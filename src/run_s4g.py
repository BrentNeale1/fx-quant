"""Run S4-G Pullback-First on EUR_AUD only (quick validation).

Period: 2024-01-01 to 2024-06-30 (6 months for speed).
3 timeframes: M5 entry, M15 ribbon/ATR, H1 trend.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from src.indicators.technical import compute_all_indicators
from src.backtester.engine import Backtester
from src.strategies_pkg.s4g_pullback import S4G_Pullback

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TEST_SET_DIR = os.path.join(PROJECT_ROOT, "data", "test_set")

DATE_START = "2024-01-01"
DATE_END = "2024-07-01"
PAIR = "EUR_AUD"


def load_data(pair, tf):
    """Load data, combining processed + test_set to cover full date range."""
    dfs = []
    for d in [PROCESSED_DIR, TEST_SET_DIR]:
        fp = os.path.join(d, f"{pair}_{tf}.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            df.index.name = "timestamp"
            dfs.append(df)
    if not dfs:
        print(f"  WARNING: No data found for {pair}_{tf}")
        return None
    combined = pd.concat(dfs)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    combined = compute_all_indicators(combined)
    combined = combined[(combined.index >= DATE_START) & (combined.index < DATE_END)]
    return combined


def main():
    print("=" * 60)
    print("S4-G: Pullback-First (Quick Validation)")
    print(f"Pair: {PAIR} | Period: {DATE_START} to {DATE_END}")
    print(f"Entry TF: M5 | Ribbon TF: M15 | Trend TF: H1")
    print("=" * 60)

    print(f"\n  Loading M5...", end=" ", flush=True)
    m5 = load_data(PAIR, "M5")
    if m5 is None:
        return
    print(f"{len(m5)} bars")

    print(f"  Loading M15...", end=" ", flush=True)
    m15 = load_data(PAIR, "M15")
    if m15 is None:
        return
    print(f"{len(m15)} bars")

    print(f"  Loading H1...", end=" ", flush=True)
    h1 = load_data(PAIR, "H1")
    if h1 is None:
        return
    print(f"{len(h1)} bars")

    # Create strategy and attach M15 data
    strategy = S4G_Pullback()
    strategy.m15_data = m15

    # Run backtest: M5 primary, H1 as HTF
    bt = Backtester(
        data=m5,
        strategy=strategy,
        pair=PAIR,
        starting_equity=100_000.0,
        htf_data=h1,
    )

    print(f"\n  Running backtest...", end=" ", flush=True)
    t0 = time.time()
    report = bt.run()
    elapsed = time.time() - t0
    trade_log = bt.get_trade_log_df()
    n = report.get("total_trades", 0)
    print(f"{n} trades in {elapsed:.0f}s")

    # Report
    print(f"\n{'='*60}")
    print(f"S4-G RESULTS — {PAIR}")
    print(f"{'='*60}")

    if n == 0:
        print("  NO TRADES!")
        print("\n  VERDICT: ABANDON (0 trades)")
        return

    wr = report.get("win_rate_pct", 0)
    pf = report.get("profit_factor", 0)
    pnl = report.get("total_pnl_pips", 0)
    pnl_d = report.get("total_pnl_dollars", 0)
    dd = report.get("max_drawdown_pct", 0)

    print(f"  Total Trades:    {n}")
    print(f"  Win Rate:        {wr:.1f}%")
    print(f"  Profit Factor:   {pf:.2f}")
    print(f"  Avg RR:          {report.get('avg_rr', 0):.2f}")
    print(f"  Expectancy:      {report.get('expectancy_pips', 0):.2f} pips")
    print(f"  Max DD:          {dd:.2f}%")
    print(f"  Total PnL:       {pnl:.1f} pips / ${pnl_d:,.2f}")
    print(f"  Avg Win:         {report.get('avg_win_pips', 0):.1f}p")
    print(f"  Avg Loss:        {report.get('avg_loss_pips', 0):.1f}p")
    print(f"  Exits:           {report.get('exit_reasons', {})}")

    # Decision gate
    print(f"\n{'='*60}")
    print(f"DECISION GATE")
    print(f"{'='*60}")
    if n > 10 and pf > 0.9:
        print(f"  PASS: {n} trades, PF {pf:.2f}")
        print(f"  -> Run full dataset test")
    elif n < 10 or pf < 0.8:
        print(f"  FAIL: {n} trades, PF {pf:.2f}")
        print(f"  -> Abandon, move to Smart Money strategies")
    else:
        print(f"  BORDERLINE: {n} trades, PF {pf:.2f}")
        print(f"  -> Consider tweaks or expand test period")


if __name__ == "__main__":
    main()
