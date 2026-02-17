# src/param_sweep.py
"""
Parameter sweep for fx-quant strategy optimization.
Tests combinations of SMA periods, trade sizes, and confidence thresholds
against historical data and ranks by Sharpe ratio.

Usage:
    python src/param_sweep.py
"""

import os
import json
import math
import itertools
from pathlib import Path

import pandas as pd
import numpy as np
from supabase import create_client

from config_loader import load_config, get_project_root


cfg = load_config()


# ---------------------------------------------------------------------------
# Data fetching (reuse from backtester)
# ---------------------------------------------------------------------------

def fetch_data(instrument, granularity, sb, table):
    """Fetch all candles from Supabase for one instrument/granularity."""
    print(f"  Fetching {instrument} / {granularity}...")
    all_rows = []
    page_size = 1000
    offset = 0
    while True:
        resp = (
            sb.table(table)
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
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    numeric_cols = [c for c in df.columns if c not in ("instrument", "granularity")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  Got {len(df)} rows.")
    return df


# ---------------------------------------------------------------------------
# Fast backtest (no file I/O, just returns metrics)
# ---------------------------------------------------------------------------

def fast_backtest(df, sma_short, sma_long, trade_size_pct, max_dd_pct=0.05,
                  starting_equity=100_000.0, rsi_period=14,
                  rsi_overbought=None, rsi_oversold=None):
    """
    Run a quick backtest with given SMA params on raw close data.
    Computes SMAs on-the-fly so we're not limited to pre-computed columns.
    Optionally filters entries using RSI overbought/oversold thresholds.
    Returns metrics dict or None if not enough data.
    """
    # Compute SMAs from close price
    df = df.copy()
    df["sma_s"] = df["close"].rolling(sma_short).mean()
    df["sma_l"] = df["close"].rolling(sma_long).mean()

    if "ret" not in df.columns:
        df["ret"] = df["close"].pct_change()

    # Compute RSI if filtering is enabled
    rsi_values = None
    if rsi_overbought is not None or rsi_oversold is not None:
        delta = df["close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(rsi_period).mean()
        ma_down = down.rolling(rsi_period).mean()
        rs = ma_up / (ma_down.replace(0, np.nan))
        df["_rsi"] = 100 - (100 / (1 + rs))

    df = df.dropna(subset=["sma_s", "sma_l", "ret"])
    if len(df) < 100:
        return None

    # Generate signals
    signals = np.where(df["sma_s"].values > df["sma_l"].values, 1, 0)

    # Apply RSI filter: block BUY when RSI is overbought, block when oversold
    if rsi_overbought is not None or rsi_oversold is not None:
        rsi_vals = df["_rsi"].values
        for i in range(len(signals)):
            if signals[i] == 1 and not np.isnan(rsi_vals[i]):
                if rsi_overbought is not None and rsi_vals[i] > rsi_overbought:
                    signals[i] = 0  # Block buy — overbought
                if rsi_oversold is not None and rsi_vals[i] < rsi_oversold:
                    signals[i] = 0  # Block buy — oversold

    closes = df["close"].values
    rets = df["ret"].values

    equity = starting_equity
    peak_equity = equity
    position = 0
    position_size = 0.0
    stopped = False

    equity_curve = []
    wins = 0
    losses = 0
    buy_equity = None
    num_trades = 0

    for i in range(len(df)):
        sig = signals[i]

        if stopped:
            equity_curve.append(equity)
            continue

        # P&L from existing position
        if position == 1 and i > 0:
            equity += position_size * rets[i]

        # Drawdown check
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0

        if dd >= max_dd_pct:
            stopped = True
            if position == 1:
                if equity > buy_equity:
                    wins += 1
                else:
                    losses += 1
            position = 0
            position_size = 0.0
            equity_curve.append(equity)
            continue

        # Position changes
        if sig != position:
            if sig == 1 and position == 0:
                position_size = equity * trade_size_pct
                buy_equity = equity
                num_trades += 1
            elif sig == 0 and position == 1:
                num_trades += 1
                if equity > buy_equity:
                    wins += 1
                else:
                    losses += 1
                position_size = 0.0
            position = sig

        equity_curve.append(equity)

    # Metrics
    final_equity = equity_curve[-1] if equity_curve else starting_equity
    total_return_pct = ((final_equity - starting_equity) / starting_equity) * 100

    eq_series = pd.Series(equity_curve, index=df.index)
    peak = eq_series.cummax()
    dd_series = (peak - eq_series) / peak
    max_drawdown_pct = dd_series.max() * 100

    round_trips = wins + losses
    win_rate = (wins / round_trips * 100) if round_trips > 0 else 0.0

    eq_returns = eq_series.pct_change().dropna()
    sharpe = 0.0
    if len(eq_returns) > 1 and eq_returns.std() > 0:
        # Annualize based on bars per year (approximate)
        sharpe = (eq_returns.mean() / eq_returns.std()) * math.sqrt(252 * 24 * 12)

    return {
        "sma_short": sma_short,
        "sma_long": sma_long,
        "trade_size_pct": trade_size_pct,
        "rsi_overbought": rsi_overbought,
        "rsi_oversold": rsi_oversold,
        "starting_equity": starting_equity,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return_pct, 4),
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "num_trades": num_trades,
        "round_trips": round_trips,
        "win_rate_pct": round(win_rate, 2),
        "sharpe_ratio": round(sharpe, 4),
        "stopped_out": stopped,
    }


def main():
    # Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    sb = create_client(supabase_url, supabase_key)
    table = cfg.get("supabase", {}).get("table", "fx_candles")

    instruments = cfg["brokers"][0]["instruments"]
    granularities = cfg["data"]["candle_granularities"]

    # Parameter grid
    sma_combos = [
        (5, 20), (10, 30), (10, 50), (15, 50), (20, 50),
        (21, 50), (10, 100), (20, 100), (50, 100), (50, 200),
    ]
    trade_sizes = [0.01, 0.025, 0.05, 0.10]
    rsi_filters = [
        (None, None, "No RSI filter"),
        (70, 30, "RSI 70/30"),
        (75, 25, "RSI 75/25"),
        (80, 20, "RSI 80/20"),
    ]
    max_dd = 0.05

    all_results = []

    for instrument in instruments:
        for granularity in granularities:
            print(f"\n{'='*60}")
            print(f"Sweeping {instrument} / {granularity}")
            print(f"{'='*60}")

            df = fetch_data(instrument, granularity, sb, table)
            if df.empty:
                continue

            combos = list(itertools.product(sma_combos, trade_sizes, rsi_filters))
            total = len(combos)

            for idx, ((sma_s, sma_l), ts, (rsi_ob, rsi_os, rsi_label)) in enumerate(combos, 1):
                result = fast_backtest(df, sma_s, sma_l, ts, max_dd,
                                       rsi_overbought=rsi_ob, rsi_oversold=rsi_os)
                if result is None:
                    continue
                result["instrument"] = instrument
                result["granularity"] = granularity
                result["rsi_label"] = rsi_label
                all_results.append(result)

            print(f"  Tested {total} combinations.")

    # Sort by Sharpe ratio descending
    all_results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)

    # Save to JSON
    root = get_project_root()
    out_path = root / "logs" / "param_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print top 20
    print(f"\n{'='*60}")
    print(f"TOP 20 RESULTS (by Sharpe ratio)")
    print(f"{'='*60}")
    print(f"{'Rank':<5} {'Inst':<8} {'Gran':<5} {'SMA':>7} {'Size':>6} {'RSI':<15} {'Return':>9} {'MaxDD':>8} {'WinR':>6} {'Sharpe':>8} {'Trades':>7} {'Final Eq':>12}")
    print("-" * 115)

    for i, r in enumerate(all_results[:20], 1):
        sma_label = f"{r['sma_short']}/{r['sma_long']}"
        rsi_label = r.get("rsi_label", "None")
        stop_flag = " *" if r["stopped_out"] else ""
        print(f"{i:<5} {r['instrument']:<8} {r['granularity']:<5} {sma_label:>7} {r['trade_size_pct']:>6.1%} "
              f"{rsi_label:<15} {r['total_return_pct']:>+8.3f}% {r['max_drawdown_pct']:>7.3f}% {r['win_rate_pct']:>5.1f}% "
              f"{r['sharpe_ratio']:>8.2f} {r['round_trips']:>7} ${r['final_equity']:>11,.2f}{stop_flag}")

    # Focus comparison: SMA 50/100 on M15 across RSI filters
    print(f"\n{'='*60}")
    print(f"FOCUS: SMA 50/100 on M15 — RSI filter comparison (10% size)")
    print(f"{'='*60}")
    print(f"{'RSI Filter':<20} {'Return':>9} {'MaxDD':>8} {'WinR':>6} {'Sharpe':>8} {'Trades':>7} {'Final Eq':>12}")
    print("-" * 80)

    focus = [r for r in all_results
             if r["sma_short"] == 50 and r["sma_long"] == 100
             and r["granularity"] == "M15" and r["trade_size_pct"] == 0.10]
    focus.sort(key=lambda x: x["sharpe_ratio"], reverse=True)
    for r in focus:
        rsi_label = r.get("rsi_label", "None")
        print(f"{rsi_label:<20} {r['total_return_pct']:>+8.3f}% {r['max_drawdown_pct']:>7.3f}% {r['win_rate_pct']:>5.1f}% "
              f"{r['sharpe_ratio']:>8.2f} {r['round_trips']:>7} ${r['final_equity']:>11,.2f}")

    print(f"\nFull results saved to: {out_path}")
    print(f"Total combinations tested: {len(all_results)}")


if __name__ == "__main__":
    main()
