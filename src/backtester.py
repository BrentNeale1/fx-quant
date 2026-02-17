# src/backtester.py
"""
Backtesting engine for fx-quant.
Reads features from Supabase, simulates the SMA cross strategy,
and produces P&L, drawdown, and trade log outputs.
"""

import os
import json
import math
from pathlib import Path

import pandas as pd
import numpy as np
from supabase import create_client

from config_loader import load_config, get_project_root


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_candles_from_supabase(instrument, granularity, supabase_client, table):
    """
    Query Supabase fx_candles table filtered by instrument + granularity.
    Returns a pandas DataFrame indexed by time, sorted ascending.
    Drops rows where sma columns are NaN (warmup period).
    """
    print(f"Fetching {instrument} / {granularity} from Supabase...")

    # Supabase JS-style pagination: fetch in pages of 1000
    all_rows = []
    page_size = 1000
    offset = 0
    while True:
        resp = (
            supabase_client.table(table)
            .select("*")
            .eq("instrument", instrument)
            .eq("granularity", granularity)
            .order("time", desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = resp.data or []
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size

    if not all_rows:
        print(f"  No data found for {instrument} / {granularity}.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()

    # Convert numeric columns from possible string types
    numeric_cols = [c for c in df.columns if c not in ("instrument", "granularity")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  Fetched {len(df)} rows.")
    return df


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signals(df, strategy_cfg, ai_cfg=None):
    """
    Generate trading signals based on strategy config.
    Currently supports 'sma_cross' rule only.

    Signal logic (long-only / flat):
      sma_short > sma_long  →  signal = 1  (long)
      sma_short < sma_long  →  signal = 0  (flat)

    If ai_cfg is provided, applies RSI overbought/oversold filtering
    to block entries at extreme levels.
    """
    rule = strategy_cfg["rule"]
    if rule != "sma_cross":
        raise ValueError(f"Unsupported strategy rule: {rule}")

    short_w = strategy_cfg["params"]["short"]
    long_w = strategy_cfg["params"]["long"]
    sma_short_col = f"sma_{short_w}"
    sma_long_col = f"sma_{long_w}"

    # Drop rows where indicator columns are NaN (warmup)
    required_cols = [sma_short_col, sma_long_col, "close", "ret"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    df = df.dropna(subset=[sma_short_col, sma_long_col, "ret"]).copy()

    # Signal: 1 when short SMA above long SMA, else 0
    df["signal"] = np.where(df[sma_short_col] > df[sma_long_col], 1, 0)

    # Apply RSI filter if configured
    if ai_cfg:
        sanity = ai_cfg.get("sanity_checks", {})
        rsi_ob = sanity.get("rsi_overbought")
        rsi_os = sanity.get("rsi_oversold")

        # Find the RSI column
        rsi_col = None
        for col in df.columns:
            if col.startswith("rsi_"):
                rsi_col = col
                break

        if rsi_col and (rsi_ob or rsi_os):
            rsi_vals = df[rsi_col]
            if rsi_ob is not None:
                df.loc[(df["signal"] == 1) & (rsi_vals > rsi_ob), "signal"] = 0
            if rsi_os is not None:
                df.loc[(df["signal"] == 1) & (rsi_vals < rsi_os), "signal"] = 0

    # Position changes only on crossover (detect changes)
    df["position"] = df["signal"]

    return df


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(df, strategy_cfg):
    """
    Walk through bars, track position, equity, trades.
    Returns dict with equity_curve (Series), trades (list of dicts), metrics (dict).
    """
    trade_size_pct = strategy_cfg.get("trade_size_pct_of_equity", 0.01)
    max_dd_pct = strategy_cfg.get("max_drawdown_pct", 0.05)

    starting_equity = cfg_equity if (cfg_equity := strategy_cfg.get("starting_equity")) else 100_000.0
    equity = starting_equity
    peak_equity = equity
    position = 0  # 0 = flat, 1 = long
    position_size = 0.0
    stopped = False

    equity_curve = []
    trades = []

    times = df.index.tolist()
    signals = df["signal"].values
    closes = df["close"].values
    rets = df["ret"].values

    for i in range(len(df)):
        bar_time = times[i]
        sig = signals[i]
        price = closes[i]
        bar_ret = rets[i]

        # Check max drawdown stop
        if stopped:
            equity_curve.append(equity)
            continue

        # P&L from existing position
        if position == 1 and i > 0:
            pnl = position_size * bar_ret
            equity += pnl

        # Update peak and drawdown
        if equity > peak_equity:
            peak_equity = equity
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0

        # Max drawdown breached — stop trading
        if drawdown >= max_dd_pct:
            if position == 1:
                trades.append({
                    "time": bar_time,
                    "side": "FLAT",
                    "price": price,
                    "position_size": 0.0,
                    "equity": round(equity, 2),
                    "drawdown": round(drawdown, 6),
                })
            stopped = True
            position = 0
            position_size = 0.0
            equity_curve.append(equity)
            print(f"  Max drawdown {max_dd_pct:.1%} breached at {bar_time}. Stopping.")
            continue

        # Position change
        if sig != position:
            if sig == 1 and position == 0:
                # Enter long
                position_size = equity * trade_size_pct
                trades.append({
                    "time": bar_time,
                    "side": "BUY",
                    "price": price,
                    "position_size": round(position_size, 2),
                    "equity": round(equity, 2),
                    "drawdown": round(drawdown, 6),
                })
            elif sig == 0 and position == 1:
                # Exit long
                trades.append({
                    "time": bar_time,
                    "side": "FLAT",
                    "price": price,
                    "position_size": 0.0,
                    "equity": round(equity, 2),
                    "drawdown": round(drawdown, 6),
                })
                position_size = 0.0
            position = sig

        equity_curve.append(equity)

    equity_series = pd.Series(equity_curve, index=df.index, name="equity")
    metrics = compute_metrics(equity_series, trades, starting_equity)

    return {
        "equity_curve": equity_series,
        "trades": trades,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(equity_curve, trades, starting_equity=100_000.0):
    """
    Compute summary metrics from equity curve and trade list.
    """
    final_equity = equity_curve.iloc[-1] if len(equity_curve) > 0 else starting_equity
    total_return_pct = ((final_equity - starting_equity) / starting_equity) * 100

    # Max drawdown from equity curve
    peak = equity_curve.cummax()
    dd = (peak - equity_curve) / peak
    max_drawdown_pct = dd.max() * 100 if len(dd) > 0 else 0.0

    # Trade stats
    num_trades = len(trades)
    # Count winning round-trips (BUY followed by FLAT with higher equity)
    wins = 0
    buy_equity = None
    for t in trades:
        if t["side"] == "BUY":
            buy_equity = t["equity"]
        elif t["side"] == "FLAT" and buy_equity is not None:
            if t["equity"] > buy_equity:
                wins += 1
            buy_equity = None

    round_trips = sum(1 for t in trades if t["side"] == "FLAT")
    win_rate = (wins / round_trips * 100) if round_trips > 0 else 0.0

    # Sharpe ratio (annualized, from per-bar returns of the equity curve)
    eq_returns = equity_curve.pct_change().dropna()
    sharpe = 0.0
    if len(eq_returns) > 1 and eq_returns.std() > 0:
        sharpe = (eq_returns.mean() / eq_returns.std()) * math.sqrt(252 * 24 * 60)  # per-minute approx

    return {
        "starting_equity": round(starting_equity, 2),
        "total_return_pct": round(total_return_pct, 4),
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "num_trades": num_trades,
        "round_trips": round_trips,
        "win_rate_pct": round(win_rate, 2),
        "sharpe_ratio": round(sharpe, 4),
        "final_equity": round(final_equity, 2),
    }


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------

def compute_monthly_pnl(equity_curve, starting_equity=100_000.0):
    """
    Compute P&L for each calendar month from the equity curve.
    Returns list of dicts with month, start_equity, end_equity, pnl, pnl_pct.
    """
    monthly = []
    # Group by year-month
    grouped = equity_curve.groupby(equity_curve.index.to_period("M"))
    prev_end = starting_equity

    for period, group in grouped:
        end_eq = group.iloc[-1]
        pnl = end_eq - prev_end
        pnl_pct = (pnl / prev_end * 100) if prev_end != 0 else 0.0
        monthly.append({
            "month": str(period),
            "start_equity": round(prev_end, 2),
            "end_equity": round(end_eq, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
        })
        prev_end = end_eq

    return monthly


def save_results(instrument, granularity, results, metrics):
    """
    Save trade log CSV, monthly P&L, and a JSON summary for the dashboard.
    """
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
        cols = ["time", "instrument", "granularity", "side", "price",
                "position_size", "equity", "drawdown"]
        trade_df = trade_df[cols]
        csv_path = logs_dir / f"backtest_trades_{instrument}_{granularity}.csv"
        trade_df.to_csv(csv_path, index=False)
        print(f"  Trade log saved: {csv_path}")
    else:
        print("  No trades to log.")

    # Monthly P&L
    starting_equity = metrics.get("starting_equity", 100_000.0)
    monthly = compute_monthly_pnl(equity_curve, starting_equity)

    # Save JSON summary for dashboard
    summary = {
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
    json_path = logs_dir / f"backtest_summary_{instrument}_{granularity}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary saved: {json_path}")

    # Console summary
    print(f"\n  --- {instrument} / {granularity} Summary ---")
    print(f"  Total return:   {metrics['total_return_pct']:.4f}%")
    print(f"  Max drawdown:   {metrics['max_drawdown_pct']:.4f}%")
    print(f"  Trades:         {metrics['num_trades']}  (round-trips: {metrics['round_trips']})")
    print(f"  Win rate:       {metrics['win_rate_pct']:.2f}%")
    print(f"  Sharpe ratio:   {metrics['sharpe_ratio']:.4f}")
    print(f"  Final equity:   ${metrics['final_equity']:,.2f}")

    # Monthly P&L table
    print(f"\n  Monthly P&L:")
    print(f"  {'Month':<10} {'Start':>12} {'End':>12} {'P&L':>10} {'%':>8}")
    print(f"  {'-'*54}")
    for m in monthly:
        sign = "+" if m["pnl"] >= 0 else ""
        print(f"  {m['month']:<10} ${m['start_equity']:>11,.2f} ${m['end_equity']:>11,.2f} {sign}${m['pnl']:>8,.2f} {sign}{m['pnl_pct']:.2f}%")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()

    # Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_KEY in config/.env")
    sb = create_client(supabase_url, supabase_key)
    table = cfg.get("supabase", {}).get("table", "fx_candles")

    strategy_cfg = cfg["strategy"]
    ai_cfg = cfg.get("ai", {})
    instruments = cfg["brokers"][0]["instruments"]
    granularities = cfg["data"]["candle_granularities"]

    rsi_ob = ai_cfg.get("sanity_checks", {}).get("rsi_overbought", "off")
    rsi_os = ai_cfg.get("sanity_checks", {}).get("rsi_oversold", "off")
    print(f"Backtesting strategy: {strategy_cfg['rule']}")
    print(f"RSI filter: overbought={rsi_ob}, oversold={rsi_os}")
    print(f"Instruments: {instruments}")
    print(f"Granularities: {granularities}\n")

    for instrument in instruments:
        for granularity in granularities:
            print(f"=== {instrument} / {granularity} ===")

            df = fetch_candles_from_supabase(instrument, granularity, sb, table)
            if df.empty:
                print("  Skipping — no data.\n")
                continue

            df = generate_signals(df, strategy_cfg, ai_cfg=ai_cfg)
            if df.empty:
                print("  Skipping — no valid rows after warmup.\n")
                continue

            results = run_backtest(df, strategy_cfg)
            save_results(instrument, granularity, results, results["metrics"])

    print("Backtesting complete.")


if __name__ == "__main__":
    main()
