# src/backtest_strategies.py
"""
Multi-strategy backtester for 5 trading strategies.

Extends the existing backtester with:
- 3 take-profit levels with custom partial close percentages
- Trailing stop for TP3 runner (trail by Nx ATR)
- Time-based exits (max candles before forced close)
- Confidence-based position sizing (1%/1.5%/2% risk)
- News event filter (reuse is_near_event from backtester)

Global risk rules:
- 5% daily drawdown halt
- 1.5:1 minimum RR check before entry
- No entries within 30 min of major news

Usage:
    python src/backtest_strategies.py
"""

import os
import json
import math
from pathlib import Path

import pandas as pd
import numpy as np
from supabase import create_client

from config_loader import load_config, get_project_root
from backtester import (
    fetch_candles_from_supabase,
    fetch_calendar_for_backtest,
    is_near_event,
    _prepare_calendar_index,
    compute_metrics,
    compute_monthly_pnl,
)
from strategies import STRATEGY_REGISTRY


# ---------------------------------------------------------------------------
# Confidence-based position sizing
# ---------------------------------------------------------------------------

def _get_risk_pct(confidence):
    """Map confidence score to risk percentage of equity."""
    if confidence >= 4:
        return 0.02   # 2% risk for high confidence
    elif confidence >= 2:
        return 0.015  # 1.5% risk for medium
    else:
        return 0.01   # 1% risk for low


# ---------------------------------------------------------------------------
# Extended backtest engine with 3 TPs + trailing stop + time exit
# ---------------------------------------------------------------------------

def run_backtest_3tp(df, strategy_cfg, tp_splits=None, trail_atr_mult=0.0,
                     max_bars=0, calendar_df=None, event_buffer_minutes=30):
    """
    Backtest engine with 3 take-profit levels, trailing stop, and time exit.

    Parameters
    ----------
    df : DataFrame with signal, sl_price, tp1_price, tp2_price, tp3_price,
         confidence columns.
    strategy_cfg : dict with max_drawdown_pct, starting_equity.
    tp_splits : list of 3 floats, e.g. [0.40, 0.40, 0.20] for TP1/TP2/TP3.
    trail_atr_mult : float, ATR multiplier for trailing stop on TP3 runner.
                     0.0 = no trailing stop.
    max_bars : int, max bars to hold before forced close. 0 = no limit.
    calendar_df : DataFrame of economic events for news filter.
    event_buffer_minutes : int, minutes buffer around news events.

    Returns dict with equity_curve, trades, metrics.
    """
    if tp_splits is None:
        tp_splits = [0.40, 0.40, 0.20]

    max_dd_pct = strategy_cfg.get("max_drawdown_pct", 0.05)
    starting_equity = strategy_cfg.get("starting_equity", 100_000.0)
    min_rr = strategy_cfg.get("min_reward_risk", 1.5)

    equity = starting_equity
    peak_equity = equity
    daily_start_equity = equity
    stopped = False

    # Position state
    in_position = False
    direction = 0
    entry_price = 0.0
    sl_price = 0.0
    tp1_price = 0.0
    tp2_price = 0.0
    tp3_price = 0.0
    position_size = 0.0
    tp1_closed = False
    tp2_closed = False
    bars_in_trade = 0
    trail_active = False
    trailing_sl = 0.0

    # Calendar filter
    cal_indexed = _prepare_calendar_index(calendar_df)
    blocked_by_calendar = 0

    equity_curve = []
    trades = []

    times = df.index.tolist()
    signals = df["signal"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    sl_col = df["sl_price"].values
    tp1_col = df["tp1_price"].values
    tp2_col = df["tp2_price"].values
    tp3_col = df["tp3_price"].values if "tp3_price" in df.columns else np.full(len(df), np.nan)
    conf_col = df["confidence"].values if "confidence" in df.columns else np.ones(len(df), dtype=int)
    atr_col = df["atr_14"].values if "atr_14" in df.columns else np.full(len(df), np.nan)

    current_date = None

    for i in range(len(df)):
        bar_time = times[i]
        bar_high = highs[i]
        bar_low = lows[i]
        bar_close = closes[i]

        # Daily drawdown reset
        bar_date = pd.Timestamp(bar_time).date()
        if bar_date != current_date:
            current_date = bar_date
            daily_start_equity = equity
            if stopped:
                stopped = False  # reset daily halt

        if stopped:
            equity_curve.append(equity)
            continue

        # --- Exit logic for active position ---
        if in_position:
            bars_in_trade += 1
            remaining_splits = []
            if not tp1_closed:
                remaining_splits = tp_splits  # all 3 portions
            elif not tp2_closed:
                remaining_splits = [0, tp_splits[1], tp_splits[2]]
            else:
                remaining_splits = [0, 0, tp_splits[2]]

            remaining_pct = sum(remaining_splits)
            remaining_size = position_size * remaining_pct

            # Update trailing stop if active
            if trail_active and trail_atr_mult > 0 and not np.isnan(atr_col[i]):
                if direction == 1:
                    new_trail = bar_high - trail_atr_mult * atr_col[i]
                    trailing_sl = max(trailing_sl, new_trail)
                elif direction == -1:
                    new_trail = bar_low + trail_atr_mult * atr_col[i]
                    trailing_sl = min(trailing_sl, new_trail)

            # Effective SL (use trailing if active and better)
            effective_sl = sl_price
            if trail_active:
                if direction == 1:
                    effective_sl = max(sl_price, trailing_sl)
                else:
                    effective_sl = min(sl_price, trailing_sl)

            # Time-based forced exit
            if max_bars > 0 and bars_in_trade >= max_bars:
                pnl = remaining_size * (bar_close - entry_price) / entry_price * direction
                equity += pnl
                trades.append({
                    "time": bar_time, "side": "TIME_EXIT",
                    "price": bar_close, "position_size": 0.0,
                    "equity": round(equity, 2), "drawdown": 0.0,
                    "pnl": round(pnl, 2),
                })
                in_position = False
                tp1_closed = False
                tp2_closed = False
                trail_active = False

            elif direction == 1:  # LONG
                # SL check
                if bar_low <= effective_sl:
                    pnl = remaining_size * (effective_sl - entry_price) / entry_price
                    equity += pnl
                    trades.append({
                        "time": bar_time,
                        "side": "TRAIL_SL_LONG" if trail_active and effective_sl == trailing_sl else "SL_EXIT_LONG",
                        "price": effective_sl, "position_size": 0.0,
                        "equity": round(equity, 2), "drawdown": 0.0,
                        "pnl": round(pnl, 2),
                    })
                    in_position = False
                    tp1_closed = False
                    tp2_closed = False
                    trail_active = False

                # TP1
                elif not tp1_closed and tp_splits[0] > 0 and bar_high >= tp1_price:
                    close_size = position_size * tp_splits[0]
                    pnl = close_size * (tp1_price - entry_price) / entry_price
                    equity += pnl
                    trades.append({
                        "time": bar_time, "side": "TP1_LONG",
                        "price": tp1_price,
                        "position_size": round(position_size * (tp_splits[1] + tp_splits[2]), 2),
                        "equity": round(equity, 2), "drawdown": 0.0,
                        "pnl": round(pnl, 2),
                    })
                    tp1_closed = True
                    sl_price = entry_price  # move to breakeven

                    # Check TP2 same bar
                    if tp_splits[1] > 0 and bar_high >= tp2_price:
                        close_size2 = position_size * tp_splits[1]
                        pnl2 = close_size2 * (tp2_price - entry_price) / entry_price
                        equity += pnl2
                        trades.append({
                            "time": bar_time, "side": "TP2_LONG",
                            "price": tp2_price,
                            "position_size": round(position_size * tp_splits[2], 2),
                            "equity": round(equity, 2), "drawdown": 0.0,
                            "pnl": round(pnl2, 2),
                        })
                        tp2_closed = True
                        # Activate trailing stop for runner
                        if tp_splits[2] > 0 and trail_atr_mult > 0:
                            trail_active = True
                            trailing_sl = tp2_price - trail_atr_mult * atr_col[i] if not np.isnan(atr_col[i]) else entry_price

                        # Check TP3 same bar
                        if tp_splits[2] > 0 and not np.isnan(tp3_price) and bar_high >= tp3_price:
                            close_size3 = position_size * tp_splits[2]
                            pnl3 = close_size3 * (tp3_price - entry_price) / entry_price
                            equity += pnl3
                            trades.append({
                                "time": bar_time, "side": "TP3_LONG",
                                "price": tp3_price, "position_size": 0.0,
                                "equity": round(equity, 2), "drawdown": 0.0,
                                "pnl": round(pnl3, 2),
                            })
                            in_position = False
                            tp1_closed = False
                            tp2_closed = False
                            trail_active = False

                # TP2 (after TP1)
                elif tp1_closed and not tp2_closed and tp_splits[1] > 0 and bar_high >= tp2_price:
                    close_size = position_size * tp_splits[1]
                    pnl = close_size * (tp2_price - entry_price) / entry_price
                    equity += pnl
                    trades.append({
                        "time": bar_time, "side": "TP2_LONG",
                        "price": tp2_price,
                        "position_size": round(position_size * tp_splits[2], 2),
                        "equity": round(equity, 2), "drawdown": 0.0,
                        "pnl": round(pnl, 2),
                    })
                    tp2_closed = True
                    if tp_splits[2] > 0 and trail_atr_mult > 0:
                        trail_active = True
                        trailing_sl = tp2_price - trail_atr_mult * atr_col[i] if not np.isnan(atr_col[i]) else entry_price

                    if tp_splits[2] <= 0:
                        in_position = False
                        tp1_closed = False
                        tp2_closed = False
                        trail_active = False
                    elif not np.isnan(tp3_price) and bar_high >= tp3_price:
                        close_size3 = position_size * tp_splits[2]
                        pnl3 = close_size3 * (tp3_price - entry_price) / entry_price
                        equity += pnl3
                        trades.append({
                            "time": bar_time, "side": "TP3_LONG",
                            "price": tp3_price, "position_size": 0.0,
                            "equity": round(equity, 2), "drawdown": 0.0,
                            "pnl": round(pnl3, 2),
                        })
                        in_position = False
                        tp1_closed = False
                        tp2_closed = False
                        trail_active = False

                # TP3 (after TP1+TP2)
                elif tp1_closed and tp2_closed and tp_splits[2] > 0:
                    if not np.isnan(tp3_price) and bar_high >= tp3_price:
                        close_size = position_size * tp_splits[2]
                        pnl = close_size * (tp3_price - entry_price) / entry_price
                        equity += pnl
                        trades.append({
                            "time": bar_time, "side": "TP3_LONG",
                            "price": tp3_price, "position_size": 0.0,
                            "equity": round(equity, 2), "drawdown": 0.0,
                            "pnl": round(pnl, 2),
                        })
                        in_position = False
                        tp1_closed = False
                        tp2_closed = False
                        trail_active = False

            elif direction == -1:  # SHORT
                if bar_high >= effective_sl:
                    pnl = remaining_size * (entry_price - effective_sl) / entry_price
                    equity += pnl
                    trades.append({
                        "time": bar_time,
                        "side": "TRAIL_SL_SHORT" if trail_active and effective_sl == trailing_sl else "SL_EXIT_SHORT",
                        "price": effective_sl, "position_size": 0.0,
                        "equity": round(equity, 2), "drawdown": 0.0,
                        "pnl": round(pnl, 2),
                    })
                    in_position = False
                    tp1_closed = False
                    tp2_closed = False
                    trail_active = False

                elif not tp1_closed and tp_splits[0] > 0 and bar_low <= tp1_price:
                    close_size = position_size * tp_splits[0]
                    pnl = close_size * (entry_price - tp1_price) / entry_price
                    equity += pnl
                    trades.append({
                        "time": bar_time, "side": "TP1_SHORT",
                        "price": tp1_price,
                        "position_size": round(position_size * (tp_splits[1] + tp_splits[2]), 2),
                        "equity": round(equity, 2), "drawdown": 0.0,
                        "pnl": round(pnl, 2),
                    })
                    tp1_closed = True
                    sl_price = entry_price

                    if tp_splits[1] > 0 and bar_low <= tp2_price:
                        close_size2 = position_size * tp_splits[1]
                        pnl2 = close_size2 * (entry_price - tp2_price) / entry_price
                        equity += pnl2
                        trades.append({
                            "time": bar_time, "side": "TP2_SHORT",
                            "price": tp2_price,
                            "position_size": round(position_size * tp_splits[2], 2),
                            "equity": round(equity, 2), "drawdown": 0.0,
                            "pnl": round(pnl2, 2),
                        })
                        tp2_closed = True
                        if tp_splits[2] > 0 and trail_atr_mult > 0:
                            trail_active = True
                            trailing_sl = tp2_price + trail_atr_mult * atr_col[i] if not np.isnan(atr_col[i]) else entry_price

                        if tp_splits[2] > 0 and not np.isnan(tp3_price) and bar_low <= tp3_price:
                            close_size3 = position_size * tp_splits[2]
                            pnl3 = close_size3 * (entry_price - tp3_price) / entry_price
                            equity += pnl3
                            trades.append({
                                "time": bar_time, "side": "TP3_SHORT",
                                "price": tp3_price, "position_size": 0.0,
                                "equity": round(equity, 2), "drawdown": 0.0,
                                "pnl": round(pnl3, 2),
                            })
                            in_position = False
                            tp1_closed = False
                            tp2_closed = False
                            trail_active = False

                elif tp1_closed and not tp2_closed and tp_splits[1] > 0 and bar_low <= tp2_price:
                    close_size = position_size * tp_splits[1]
                    pnl = close_size * (entry_price - tp2_price) / entry_price
                    equity += pnl
                    trades.append({
                        "time": bar_time, "side": "TP2_SHORT",
                        "price": tp2_price,
                        "position_size": round(position_size * tp_splits[2], 2),
                        "equity": round(equity, 2), "drawdown": 0.0,
                        "pnl": round(pnl, 2),
                    })
                    tp2_closed = True
                    if tp_splits[2] > 0 and trail_atr_mult > 0:
                        trail_active = True
                        trailing_sl = tp2_price + trail_atr_mult * atr_col[i] if not np.isnan(atr_col[i]) else entry_price

                    if tp_splits[2] <= 0:
                        in_position = False
                        tp1_closed = False
                        tp2_closed = False
                        trail_active = False
                    elif not np.isnan(tp3_price) and bar_low <= tp3_price:
                        close_size3 = position_size * tp_splits[2]
                        pnl3 = close_size3 * (entry_price - tp3_price) / entry_price
                        equity += pnl3
                        trades.append({
                            "time": bar_time, "side": "TP3_SHORT",
                            "price": tp3_price, "position_size": 0.0,
                            "equity": round(equity, 2), "drawdown": 0.0,
                            "pnl": round(pnl3, 2),
                        })
                        in_position = False
                        tp1_closed = False
                        tp2_closed = False
                        trail_active = False

                elif tp1_closed and tp2_closed and tp_splits[2] > 0:
                    if not np.isnan(tp3_price) and bar_low <= tp3_price:
                        close_size = position_size * tp_splits[2]
                        pnl = close_size * (entry_price - tp3_price) / entry_price
                        equity += pnl
                        trades.append({
                            "time": bar_time, "side": "TP3_SHORT",
                            "price": tp3_price, "position_size": 0.0,
                            "equity": round(equity, 2), "drawdown": 0.0,
                            "pnl": round(pnl, 2),
                        })
                        in_position = False
                        tp1_closed = False
                        tp2_closed = False
                        trail_active = False

        # --- New entry (only when flat) ---
        if not in_position and not stopped:
            sig = signals[i]
            if sig in (1, -1) and not np.isnan(sl_col[i]):
                # News filter
                if is_near_event(bar_time, cal_indexed, event_buffer_minutes):
                    blocked_by_calendar += 1
                    equity_curve.append(equity)
                    continue

                # Minimum RR check
                risk = abs(bar_close - sl_col[i])
                reward = abs(tp1_col[i] - bar_close) if not np.isnan(tp1_col[i]) else 0
                if risk > 0 and reward / risk < min_rr:
                    equity_curve.append(equity)
                    continue

                # Confidence-based sizing
                risk_pct = _get_risk_pct(int(conf_col[i]))
                direction = sig
                entry_price = bar_close
                sl_price = sl_col[i]
                tp1_price = tp1_col[i]
                tp2_price = tp2_col[i]
                tp3_price = tp3_col[i] if not np.isnan(tp3_col[i]) else tp2_col[i]
                position_size = equity * risk_pct
                tp1_closed = False
                tp2_closed = False
                trail_active = False
                trailing_sl = 0.0
                bars_in_trade = 0
                in_position = True

                side_label = "BUY" if sig == 1 else "SELL_SHORT"
                trades.append({
                    "time": bar_time, "side": side_label,
                    "price": bar_close,
                    "position_size": round(position_size, 2),
                    "equity": round(equity, 2), "drawdown": 0.0,
                    "pnl": 0.0,
                })

        # Update peak and drawdown
        if equity > peak_equity:
            peak_equity = equity

        # 5% daily drawdown halt
        daily_dd = (daily_start_equity - equity) / daily_start_equity if daily_start_equity > 0 else 0
        if daily_dd >= max_dd_pct:
            if in_position:
                remaining_pct = sum(tp_splits) if not tp1_closed else (
                    tp_splits[1] + tp_splits[2] if not tp2_closed else tp_splits[2]
                )
                remaining_size = position_size * remaining_pct
                pnl = remaining_size * (bar_close - entry_price) / entry_price * direction
                equity += pnl
                trades.append({
                    "time": bar_time, "side": "DD_EXIT",
                    "price": bar_close, "position_size": 0.0,
                    "equity": round(equity, 2),
                    "drawdown": round(daily_dd, 6),
                    "pnl": round(pnl, 2),
                })
                in_position = False
                tp1_closed = False
                tp2_closed = False
                trail_active = False
            stopped = True
            print(f"  Daily drawdown {max_dd_pct:.1%} breached at {bar_time}. Halting for day.")

        equity_curve.append(equity)

    equity_series = pd.Series(equity_curve, index=df.index, name="equity")
    metrics = compute_metrics(equity_series, trades, starting_equity)

    if blocked_by_calendar > 0:
        print(f"  Calendar filter blocked {blocked_by_calendar} entries.")
        metrics["blocked_by_calendar"] = blocked_by_calendar

    return {
        "equity_curve": equity_series,
        "trades": trades,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Save results (extended with strategy_name)
# ---------------------------------------------------------------------------

def save_strategy_results(strategy_key, strategy_name, instrument, granularity,
                          results, metrics):
    """Save backtest results with strategy name in the JSON summary."""
    root = get_project_root()
    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)

    trades = results["trades"]
    equity_curve = results["equity_curve"]

    # Trade log CSV
    if trades:
        trade_df = pd.DataFrame(trades)
        trade_df["instrument"] = instrument
        trade_df["granularity"] = granularity
        trade_df["strategy"] = strategy_name
        base_cols = ["time", "strategy", "instrument", "granularity", "side", "price",
                     "position_size", "equity", "drawdown", "pnl"]
        cols = [c for c in base_cols if c in trade_df.columns]
        trade_df = trade_df[cols]
        csv_path = logs_dir / f"backtest_trades_{strategy_key}_{instrument}_{granularity}.csv"
        trade_df.to_csv(csv_path, index=False)
        print(f"  Trade log: {csv_path}")

    # Monthly P&L
    starting_equity = metrics.get("starting_equity", 100_000.0)
    monthly = compute_monthly_pnl(equity_curve, starting_equity)

    # JSON summary
    summary = {
        "strategy_name": strategy_name,
        "strategy_key": strategy_key,
        "instrument": instrument,
        "granularity": granularity,
        "run_time": pd.Timestamp.now(tz="UTC").isoformat(),
        "data_range": {
            "start": str(equity_curve.index[0]) if len(equity_curve) > 0 else "",
            "end": str(equity_curve.index[-1]) if len(equity_curve) > 0 else "",
            "bars": len(equity_curve),
        },
        "metrics": metrics,
        "monthly_pnl": monthly,
    }
    json_path = logs_dir / f"backtest_summary_{strategy_key}_{strategy_name}_{instrument}_{granularity}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary: {json_path}")

    # Console summary
    print(f"\n  --- {strategy_name} | {instrument} / {granularity} ---")
    print(f"  Return: {metrics['total_return_pct']:.4f}% | DD: {metrics['max_drawdown_pct']:.4f}%")
    print(f"  Trades: {metrics['num_trades']} ({metrics['round_trips']} RTs) | Win: {metrics['win_rate_pct']:.1f}%")
    print(f"  Sharpe: {metrics['sharpe_ratio']:.4f} | Final: ${metrics['final_equity']:,.2f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_KEY in config/.env")
    sb = create_client(supabase_url, supabase_key)
    table = cfg.get("supabase", {}).get("table", "fx_candles")

    cal_cfg = cfg.get("economic_calendar", {})
    cal_enabled = cal_cfg.get("enabled", False)
    event_buffer = cal_cfg.get("event_buffer_minutes", 30)

    strategy_cfg = {
        "max_drawdown_pct": 0.05,
        "starting_equity": 100_000.0,
        "min_reward_risk": 1.5,
    }

    print("=" * 70)
    print("Multi-Strategy Backtester")
    print(f"Strategies: {list(STRATEGY_REGISTRY.keys())}")
    print("=" * 70)

    for skey, sinfo in STRATEGY_REGISTRY.items():
        strategy_name = sinfo["name"]
        signal_func = sinfo["func"]
        pairs = sinfo["pairs"]
        entry_tf = sinfo["entry_tf"]
        filter_tf = sinfo.get("filter_tf")
        tp_splits = sinfo.get("tp_splits", [0.40, 0.40, 0.20])
        trail_atr = sinfo.get("trail_atr_mult", 0.0)
        max_bars = sinfo.get("max_bars", 0)

        print(f"\n{'='*70}")
        print(f"Strategy: {strategy_name}")
        print(f"Pairs: {pairs} | Entry: {entry_tf} | Filter: {filter_tf}")
        print(f"TP splits: {tp_splits} | Trail ATR: {trail_atr} | Max bars: {max_bars}")
        print(f"{'='*70}")

        for instrument in pairs:
            print(f"\n--- {instrument} ---")

            # Load entry timeframe data
            df_entry = fetch_candles_from_supabase(instrument, entry_tf, sb, table)
            if df_entry.empty:
                print(f"  No {entry_tf} data for {instrument}. Skipping.")
                continue

            # Load filter timeframe data if needed
            df_filter = None
            if filter_tf:
                df_filter = fetch_candles_from_supabase(instrument, filter_tf, sb, table)
                if df_filter.empty:
                    print(f"  No {filter_tf} data for {instrument}. Proceeding without filter.")
                    df_filter = None

            # Load calendar
            calendar_df = None
            if cal_enabled:
                calendar_df = fetch_calendar_for_backtest(instrument, sb, cfg)

            # Generate signals
            print(f"  Generating {strategy_name} signals...")
            df_signals = signal_func(df_entry, df_filter=df_filter)

            signal_count = (df_signals["signal"] != 0).sum()
            print(f"  Signals generated: {signal_count} on {len(df_signals)} bars")

            if signal_count == 0:
                print(f"  No signals. Skipping backtest.")
                continue

            # Run backtest
            print(f"  Running backtest...")
            results = run_backtest_3tp(
                df_signals, strategy_cfg,
                tp_splits=tp_splits,
                trail_atr_mult=trail_atr,
                max_bars=max_bars,
                calendar_df=calendar_df,
                event_buffer_minutes=event_buffer,
            )

            save_strategy_results(
                skey, strategy_name, instrument, entry_tf,
                results, results["metrics"],
            )

    print("\n" + "=" * 70)
    print("All strategy backtests complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
