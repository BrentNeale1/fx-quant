"""
Phase 1 Backtest Runner.

Runs all 5 strategies across their assigned pairs using TRAINING data only.
Generates rich trade logs and performance reports.
Saves results to results/phase1/.
"""
import os
import sys
import json
import time

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.indicators.technical import compute_all_indicators
from src.backtester.engine import Backtester
from src.strategies_pkg import (
    STRATEGIES, STRATEGY_PAIRS, STRATEGY_TIMEFRAMES,
    S1_MA_Breakout, S3_KeyLevel_Breakout,
    S4_EMA_Ribbon, S5_Momentum_Exhaustion,
    S6_EMA_Bounce,
)
from src.trade_pdf_report import generate_trade_pdf
# S2_VWAP_Reversal disabled — needs full redesign

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase1")


def load_data(pair: str, timeframe: str) -> pd.DataFrame:
    """Load training data and compute indicators."""
    filepath = os.path.join(PROCESSED_DIR, f"{pair}_{timeframe}.csv")
    if not os.path.exists(filepath):
        print(f"  WARNING: {filepath} not found")
        return None

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.index.name = "timestamp"

    # Compute all indicators
    df = compute_all_indicators(df)

    return df


def run_strategy_backtest(strategy_id: int, pair: str):
    """Run a single strategy on a single pair."""
    strat_cls = STRATEGIES[strategy_id]
    tf_config = STRATEGY_TIMEFRAMES[strategy_id]
    primary_tf = tf_config["primary"]
    filter_tf = tf_config["filter"]

    print(f"  Loading {pair} {primary_tf}...", end=" ")
    data = load_data(pair, primary_tf)
    if data is None:
        return None
    print(f"{len(data)} bars")

    htf_data = None
    if filter_tf:
        print(f"  Loading {pair} {filter_tf} (HTF filter)...", end=" ")
        htf_data = load_data(pair, filter_tf)
        if htf_data is not None:
            print(f"{len(htf_data)} bars")
        else:
            print("not available")

    # Instantiate strategy
    strategy = strat_cls()

    # Run backtest
    bt = Backtester(
        data=data,
        strategy=strategy,
        pair=pair,
        starting_equity=100_000.0,
        htf_data=htf_data,
    )

    print(f"  Running backtest...", end=" ")
    start_time = time.time()
    report = bt.run()
    elapsed = time.time() - start_time
    print(f"done in {elapsed:.1f}s")

    # Get trade log
    trade_log = bt.get_trade_log_df()

    return report, trade_log, data, htf_data


def save_results(strategy_id: int, strategy_name: str, pair: str,
                 report: dict, trade_log: pd.DataFrame):
    """Save results to results/phase1/."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    prefix = f"S{strategy_id}_{pair}"

    # Save report JSON
    report_path = os.path.join(RESULTS_DIR, f"{prefix}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Save trade log CSV
    if len(trade_log) > 0:
        log_path = os.path.join(RESULTS_DIR, f"{prefix}_trades.csv")
        trade_log.to_csv(log_path, index=False)

    return report_path


def print_report_summary(report: dict):
    """Print a concise report summary."""
    total = report.get("total_trades", 0)
    if total == 0:
        print(f"    No trades generated")
        return

    print(f"    Trades: {total}")
    print(f"    Win Rate: {report.get('win_rate_pct', 0):.1f}%")
    print(f"    Avg RR: {report.get('avg_rr', 0):.2f}")
    print(f"    Expectancy: {report.get('expectancy_pips', 0):.2f} pips")
    print(f"    Sharpe: {report.get('sharpe_ratio', 0):.2f}")
    print(f"    Max DD: {report.get('max_drawdown_pct', 0):.2f}%")
    print(f"    Profit Factor: {report.get('profit_factor', 0):.2f}")
    print(f"    Total PnL: {report.get('total_pnl_pips', 0):.1f} pips / ${report.get('total_pnl_dollars', 0):.2f}")
    print(f"    Final Equity: ${report.get('final_equity', 0):,.2f}")

    exits = report.get("exit_reasons", {})
    if exits:
        print(f"    Exits: {exits}")


def main():
    # Allow filtering strategies via command line: python backtest_phase1.py 2 3 4 5
    selected = None
    if len(sys.argv) > 1:
        selected = [int(x) for x in sys.argv[1:]]

    print("=" * 70)
    print("PHASE 1 BACKTEST RUNNER")
    print("Using TRAINING data only (Jan 2021 - Aug 2023)")
    print("=" * 70)

    all_results = {}

    for strategy_id in sorted(STRATEGIES.keys()):
        if selected and strategy_id not in selected:
            continue
        strat_cls = STRATEGIES[strategy_id]
        strat_name = strat_cls().name
        pairs = STRATEGY_PAIRS[strategy_id]
        tf = STRATEGY_TIMEFRAMES[strategy_id]

        print(f"\n{'='*70}")
        print(f"Strategy {strategy_id}: {strat_name}")
        print(f"Pairs: {', '.join(pairs)}")
        print(f"Primary TF: {tf['primary']}, Filter TF: {tf['filter'] or 'None'}")
        print(f"{'='*70}")

        strategy_trades_total = 0

        for pair in pairs:
            print(f"\n  --- {pair} ---")
            result = run_strategy_backtest(strategy_id, pair)
            if result is None:
                print(f"    SKIPPED (data not available)")
                continue

            report, trade_log, data, htf_data = result
            save_results(strategy_id, strat_name, pair, report, trade_log)
            print_report_summary(report)

            # Generate per-trade PDF report
            if len(trade_log) > 0:
                generate_trade_pdf(strategy_id, pair, report, trade_log,
                                   data, htf_data)

            total_trades = report.get("total_trades", 0)
            strategy_trades_total += total_trades

            key = f"S{strategy_id}_{pair}"
            all_results[key] = report

        print(f"\n  >>> S{strategy_id} TOTAL TRADES: {strategy_trades_total}")
        if strategy_trades_total < 200:
            print(f"  >>> WARNING: Below 200 trade minimum!")

    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY")
    print("=" * 70)
    print(f"\n{'Strategy':<35} {'Trades':>7} {'WinRate':>8} {'Sharpe':>8} {'PnL Pips':>10}")
    print("-" * 70)

    for key in sorted(all_results.keys()):
        r = all_results[key]
        print(f"{key:<35} {r.get('total_trades', 0):>7} "
              f"{r.get('win_rate_pct', 0):>7.1f}% "
              f"{r.get('sharpe_ratio', 0):>8.2f} "
              f"{r.get('total_pnl_pips', 0):>10.1f}")

    # Save master summary
    summary_path = os.path.join(RESULTS_DIR, "phase1_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nMaster summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
