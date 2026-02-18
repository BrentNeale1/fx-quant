"""Run S4-F-v2 on GBP_AUD, EUR_AUD, GBP_JPY using M5 entry, M15 ribbon, H1 trend.

Uses 2023-2024 data only (18 months for speed as per spec).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from src.indicators.technical import compute_all_indicators
from src.backtester.engine import Backtester
from src.strategies_pkg.s4fv2_ema_ribbon import S4Fv2_EMA_Ribbon

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
PAIRS = ["GBP_AUD", "EUR_AUD", "GBP_JPY"]

# 2023-01-01 to 2024-06-30 (18 months)
DATE_START = "2023-01-01"
DATE_END = "2024-07-01"


def load_data(pair, tf):
    """Load data, compute indicators, filter to date range."""
    fp = os.path.join(PROCESSED_DIR, f"{pair}_{tf}.csv")
    if not os.path.exists(fp):
        print(f"  WARNING: {fp} not found!")
        return None
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    df = compute_all_indicators(df)
    # Filter date range
    df = df[(df.index >= DATE_START) & (df.index < DATE_END)]
    return df


def run_pair(pair):
    """Run S4-F-v2 on a single pair."""
    print(f"\n  --- {pair} ---")

    # Load all 3 timeframes
    print(f"  Loading M5...", end=" ", flush=True)
    m5_data = load_data(pair, "M5")
    if m5_data is None:
        return None
    print(f"{len(m5_data)} bars")

    print(f"  Loading M15...", end=" ", flush=True)
    m15_data = load_data(pair, "M15")
    if m15_data is None:
        return None
    print(f"{len(m15_data)} bars")

    print(f"  Loading H1...", end=" ", flush=True)
    h1_data = load_data(pair, "H1")
    if h1_data is None:
        return None
    print(f"{len(h1_data)} bars")

    # Create strategy and attach M15 data
    strategy = S4Fv2_EMA_Ribbon()
    strategy.m15_data = m15_data

    # Run backtest: M5 as primary, H1 as HTF
    bt = Backtester(
        data=m5_data,
        strategy=strategy,
        pair=pair,
        starting_equity=100_000.0,
        htf_data=h1_data,
    )

    print(f"  Running backtest...", end=" ", flush=True)
    t0 = time.time()
    report = bt.run()
    elapsed = time.time() - t0
    trade_log = bt.get_trade_log_df()
    n = report.get("total_trades", 0)
    print(f"{n} trades in {elapsed:.0f}s")

    return report, trade_log


def print_report(report):
    """Print key metrics."""
    n = report.get("total_trades", 0)
    if n == 0:
        print(f"    No trades!")
        return
    print(f"    Trades:        {n}")
    print(f"    Win Rate:      {report.get('win_rate_pct', 0):.1f}%")
    print(f"    Avg RR:        {report.get('avg_rr', 0):.2f}")
    print(f"    Profit Factor: {report.get('profit_factor', 0):.2f}")
    print(f"    Expectancy:    {report.get('expectancy_pips', 0):.2f} pips")
    print(f"    Max DD:        {report.get('max_drawdown_pct', 0):.2f}%")
    print(f"    Total PnL:     {report.get('total_pnl_pips', 0):.1f} pips / ${report.get('total_pnl_dollars', 0):,.2f}")
    print(f"    Avg Win:       {report.get('avg_win_pips', 0):.1f}p | Avg Loss: {report.get('avg_loss_pips', 0):.1f}p")
    print(f"    Exits:         {report.get('exit_reasons', {})}")


def main():
    print("=" * 60)
    print("S4-F-v2: Trend Context Quick Tune")
    print(f"Period: {DATE_START} to {DATE_END} (18 months)")
    print(f"Pairs: {', '.join(PAIRS)}")
    print(f"Entry TF: M5 | Ribbon TF: M15 | Trend TF: H1")
    print("=" * 60)

    all_reports = []
    all_trades = []

    for pair in PAIRS:
        result = run_pair(pair)
        if result is None:
            continue
        report, trade_log = result
        print_report(report)
        all_reports.append(report)
        if len(trade_log) > 0:
            all_trades.append(trade_log)

    # Aggregate
    total_trades = sum(r.get("total_trades", 0) for r in all_reports)
    if total_trades == 0:
        print("\nNO TRADES across any pair!")
        return

    total_pnl_pips = sum(r.get("total_pnl_pips", 0) for r in all_reports)
    total_pnl_dollars = sum(r.get("total_pnl_dollars", 0) for r in all_reports)

    combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if len(combined) > 0:
        wins = combined[combined["pnl_pips"] > 0]
        losses = combined[combined["pnl_pips"] < 0]
        gross_profit = wins["pnl_pips"].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses["pnl_pips"].sum()) if len(losses) > 0 else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
        avg_win = wins["pnl_pips"].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses["pnl_pips"].mean()) if len(losses) > 0 else 0
        avg_rr = avg_win / avg_loss if avg_loss > 0 else 0
        win_rate = len(wins) / total_trades * 100
        max_dd = min(r.get("max_drawdown_pct", 0) for r in all_reports)
    else:
        pf = avg_win = avg_loss = avg_rr = win_rate = 0
        max_dd = 0

    expectancy = total_pnl_pips / total_trades if total_trades > 0 else 0

    print(f"\n{'='*60}")
    print(f"S4-F-v2 AGGREGATE RESULTS")
    print(f"{'='*60}")
    print(f"  Total Trades:    {total_trades}")
    print(f"  Win Rate:        {win_rate:.1f}%")
    print(f"  Avg RR:          {avg_rr:.2f}")
    print(f"  Expectancy:      {expectancy:.2f} pips/trade")
    print(f"  Profit Factor:   {pf:.2f}")
    print(f"  Worst Max DD:    {max_dd:.2f}%")
    print(f"  Total PnL:       {total_pnl_pips:.1f} pips / ${total_pnl_dollars:,.2f}")
    print(f"  Avg Win:         {avg_win:.1f}p | Avg Loss: {avg_loss:.1f}p")

    # Target check
    print(f"\n{'='*60}")
    print(f"TARGET CHECK")
    print(f"{'='*60}")
    t_pass = "PASS" if 150 <= total_trades <= 200 else ("CLOSE" if 100 <= total_trades <= 250 else "FAIL")
    w_pass = "PASS" if win_rate > 48 else "FAIL"
    p_pass = "PASS" if pf > 0.9 else "FAIL"
    print(f"  Trades 150-200:  {total_trades:>6}  [{t_pass}]")
    print(f"  Win Rate >48%:   {win_rate:>5.1f}%  [{w_pass}]")
    print(f"  PF >0.9:         {pf:>6.2f}  [{p_pass}]")

    # Per-pair summary
    print(f"\n{'='*60}")
    print(f"PER-PAIR BREAKDOWN")
    print(f"{'='*60}")
    print(f"{'Pair':<12} {'Trades':>7} {'WR':>7} {'PF':>7} {'PnL':>10}")
    print("-" * 45)
    for r in all_reports:
        p = r.get("pair", "?")
        print(f"{p:<12} {r.get('total_trades', 0):>7} "
              f"{r.get('win_rate_pct', 0):>6.1f}% "
              f"{r.get('profit_factor', 0):>7.2f} "
              f"{r.get('total_pnl_pips', 0):>9.1f}p")


if __name__ == "__main__":
    main()
