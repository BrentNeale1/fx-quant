"""
Phase 2 — Regime Analysis (Step 7).

Runs each Phase 2 strategy on the full dataset, breaks down results by year
(2021, 2022, 2023). Answers: "Is the edge strengthening or was OOS lucky?"

Output: results/phase2/regime_analysis.json + console table.
"""
import os, sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from src.indicators.technical import compute_all_indicators
from src.backtester.engine import Backtester

# Strategy imports
from src.strategies_pkg.s7_liquidity_sweep import S7_Liquidity_Sweep
from src.strategies_pkg.s9_london_session import S9_London_Session
from src.strategies_pkg.s4f_ema_ribbon import S4F_EMA_Ribbon
from src.strategies_pkg.s3_key_level_breakout import S3_KeyLevel_Breakout

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase2")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Phase 2 strategy-pair configurations
CONFIGS = [
    {"name": "S7_Tight", "pair": "GBP_JPY", "tf": "H1", "htf_tf": "H1",
     "factory": lambda: S7_Liquidity_Sweep()},
    {"name": "S9", "pair": "GBP_USD", "tf": "H1", "htf_tf": "H1",
     "factory": lambda: S9_London_Session()},
    {"name": "S9_Filtered", "pair": "GBP_AUD", "tf": "H1", "htf_tf": "H1",
     "factory": lambda: S9_London_Session(pair="GBP_AUD", filtered=True)},
    {"name": "S4F", "pair": "EUR_AUD", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: S4F_EMA_Ribbon()},
    {"name": "S3", "pair": "GBP_JPY", "tf": "H1", "htf_tf": "H1",
     "factory": lambda: S3_KeyLevel_Breakout()},
]


def load_data(pair, tf):
    """Load price data with indicators."""
    fp = os.path.join(PROCESSED_DIR, f"{pair}_{tf}.csv")
    if not os.path.exists(fp):
        print(f"  WARNING: {fp} not found")
        return None
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    return compute_all_indicators(df)


def slice_year(df, year):
    """Slice dataframe to a specific year, with 60 days warmup prepended."""
    # Match timezone awareness of the dataframe index
    if df.index.tz is not None:
        start = pd.Timestamp(f"{year}-01-01", tz=df.index.tz)
        end = pd.Timestamp(f"{year}-12-31 23:59:59", tz=df.index.tz)
    else:
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-31 23:59:59")
    warmup_start = start - pd.DateOffset(days=60)
    sliced = df[(df.index >= warmup_start) & (df.index <= end)].copy()
    return sliced, start


def run_backtest(strategy, data, htf_data, pair, start_after=None):
    """Run backtest, optionally filtering trades after a start date."""
    bt = Backtester(data=data, strategy=strategy, pair=pair,
                    starting_equity=100_000.0, htf_data=htf_data)
    report = bt.run()
    trade_log = bt.get_trade_log_df()

    # Filter trades to only those after start_after (exclude warmup trades)
    if start_after is not None and not trade_log.empty:
        ts = pd.to_datetime(trade_log["timestamp"])
        start_ts = pd.Timestamp(start_after)
        # Align timezone awareness
        if ts.dt.tz is not None and start_ts.tz is None:
            start_ts = start_ts.tz_localize(ts.dt.tz)
        elif ts.dt.tz is None and start_ts.tz is not None:
            start_ts = start_ts.tz_localize(None)
        trade_log = trade_log[ts >= start_ts]

    return report, trade_log


def compute_metrics(trade_log):
    """Compute metrics from a trade log DataFrame."""
    if trade_log.empty or len(trade_log) == 0:
        return {
            "trades": 0, "wr": 0, "pf": 0, "pnl_pips": 0,
            "max_dd_pips": 0, "expectancy": 0,
        }

    n = len(trade_log)
    wins = trade_log[trade_log["win"] == True]
    losses = trade_log[trade_log["win"] == False]

    wr = len(wins) / n * 100 if n > 0 else 0
    gross_profit = wins["pnl_pips"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl_pips"].sum()) if len(losses) > 0 else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_pnl = trade_log["pnl_pips"].sum()
    expectancy = total_pnl / n if n > 0 else 0

    # Max drawdown in pips (cumulative PnL drawdown)
    cum_pnl = trade_log["pnl_pips"].cumsum()
    peak = cum_pnl.cummax()
    dd = cum_pnl - peak
    max_dd = dd.min() if len(dd) > 0 else 0

    return {
        "trades": n,
        "wr": round(wr, 1),
        "pf": round(pf, 2),
        "pnl_pips": round(total_pnl, 1),
        "max_dd_pips": round(max_dd, 1),
        "expectancy": round(expectancy, 2),
    }


def main():
    years = [2021, 2022, 2023]
    all_results = {}

    print(f"{'='*90}")
    print("PHASE 2 — REGIME ANALYSIS (Per-Year Breakdown)")
    print(f"{'='*90}")

    for cfg in CONFIGS:
        name = cfg["name"]
        pair = cfg["pair"]
        tf = cfg["tf"]
        htf_tf = cfg["htf_tf"]

        print(f"\n{'#'*70}")
        print(f"# {name} / {pair} ({tf})")
        print(f"{'#'*70}")

        # Load data
        data = load_data(pair, tf)
        if data is None:
            continue

        htf_data = data.copy() if htf_tf == tf else load_data(pair, htf_tf)
        if htf_data is None:
            continue

        strategy_results = {"pair": pair, "timeframe": tf, "years": {}}

        # Full dataset first
        strategy = cfg["factory"]()
        _, full_log = run_backtest(strategy, data, htf_data, pair)
        full_metrics = compute_metrics(full_log)
        strategy_results["full"] = full_metrics
        print(f"  {'FULL':>6}: {full_metrics['trades']:>4} trades | "
              f"WR {full_metrics['wr']:>5.1f}% | PF {full_metrics['pf']:>5.2f} | "
              f"PnL {full_metrics['pnl_pips']:>+8.1f}p | "
              f"DD {full_metrics['max_dd_pips']:>+8.1f}p | "
              f"Exp {full_metrics['expectancy']:>+6.2f}p")

        # Per-year breakdown
        for year in years:
            sliced, start = slice_year(data, year)
            if len(sliced) < 250:
                print(f"  {year:>6}: insufficient data ({len(sliced)} bars)")
                strategy_results["years"][str(year)] = {"trades": 0}
                continue

            htf_sliced = sliced.copy() if htf_tf == tf else slice_year(htf_data, year)[0]
            strategy = cfg["factory"]()
            _, year_log = run_backtest(strategy, sliced, htf_sliced, pair, start_after=start)
            metrics = compute_metrics(year_log)
            strategy_results["years"][str(year)] = metrics

            print(f"  {year:>6}: {metrics['trades']:>4} trades | "
                  f"WR {metrics['wr']:>5.1f}% | PF {metrics['pf']:>5.2f} | "
                  f"PnL {metrics['pnl_pips']:>+8.1f}p | "
                  f"DD {metrics['max_dd_pips']:>+8.1f}p | "
                  f"Exp {metrics['expectancy']:>+6.2f}p")

        all_results[f"{name}_{pair}"] = strategy_results

    # Save results
    out_path = os.path.join(RESULTS_DIR, "regime_analysis.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")

    # Summary table
    print(f"\n{'='*90}")
    print("REGIME SUMMARY — Per-Year Profit Factor")
    print(f"{'='*90}")
    print(f"{'Strategy':<20} {'Pair':<10} {'Full':>6} {'2021':>6} {'2022':>6} {'2023':>6} {'Trend':>8}")
    print(f"{'-'*90}")
    for key, result in all_results.items():
        name_pair = key.split("_", 1)
        name = result.get("pair", key)
        pf_full = result.get("full", {}).get("pf", 0)
        pf_years = []
        for y in ["2021", "2022", "2023"]:
            pf = result.get("years", {}).get(y, {}).get("pf", 0)
            pf_years.append(pf)

        # Trend: compare first year to last year
        if pf_years[0] > 0 and pf_years[-1] > 0:
            if pf_years[-1] > pf_years[0] * 1.1:
                trend = "UP"
            elif pf_years[-1] < pf_years[0] * 0.9:
                trend = "DOWN"
            else:
                trend = "STABLE"
        else:
            trend = "N/A"

        print(f"{key:<20} {result.get('pair',''):<10} "
              f"{pf_full:>5.2f} {pf_years[0]:>5.2f} {pf_years[1]:>5.2f} "
              f"{pf_years[2]:>5.2f} {trend:>8}")


if __name__ == "__main__":
    main()
