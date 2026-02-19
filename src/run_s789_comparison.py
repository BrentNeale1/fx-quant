"""
Run S7, S8, S9 (Smart Money strategies) comparison.

Uses last 6 months of available data with 70/30 train/test split.
Data range: ~March 2023 - August 2023 (last 6 months of processed data).
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from src.indicators.technical import compute_all_indicators
from src.backtester.engine import Backtester

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase1")

PAIRS = ["GBP_USD", "EUR_AUD", "GBP_JPY"]

# 6 months of data, 70/30 split
DATA_MONTHS = 6
TRAIN_RATIO = 0.70


def load_data(pair, tf, last_n_months=None):
    """Load and compute indicators. Optionally slice to last N months."""
    fp = os.path.join(PROCESSED_DIR, f"{pair}_{tf}.csv")
    if not os.path.exists(fp):
        print(f"  WARNING: {fp} not found, skipping")
        return None
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df.index.name = "timestamp"

    if last_n_months is not None and last_n_months > 0:
        end_date = df.index[-1]
        start_date = end_date - pd.DateOffset(months=last_n_months)
        # Keep extra warmup bars (250) before the start for indicator computation
        warmup_start = start_date - pd.DateOffset(days=60)
        df = df[df.index >= warmup_start]

    df = compute_all_indicators(df)

    if last_n_months is not None and last_n_months > 0:
        # Now trim to the actual date range (after indicators computed with warmup)
        df = df[df.index >= start_date]

    return df


def split_data(df, train_ratio=0.70):
    """Split data into train and test sets."""
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def run_backtest(strat_cls, data, htf_data, pair, label=""):
    """Run a single backtest and return report + trade log."""
    strategy = strat_cls()
    bt = Backtester(
        data=data,
        strategy=strategy,
        pair=pair,
        starting_equity=100_000.0,
        htf_data=htf_data,
    )
    report = bt.run()
    trade_log = bt.get_trade_log_df()
    return report, trade_log


def run_variant(label, strat_cls, phase="FULL"):
    """Run a strategy variant across all pairs."""
    print(f"\n{'=' * 60}")
    print(f"  {label} ({phase})")
    print(f"{'=' * 60}")

    all_reports = []
    all_trades = []

    for pair in PAIRS:
        # Load H1 as primary timeframe for S7-S9
        data = load_data(pair, "H1", last_n_months=DATA_MONTHS)
        if data is None:
            continue

        # HTF data: use H1 itself (strategies use htf_row for trend context)
        # The backtester passes the most recent fully-closed H1 bar as htf_row
        htf_data = data.copy()

        if phase == "TRAIN":
            data, _ = split_data(data, TRAIN_RATIO)
            htf_data, _ = split_data(htf_data, TRAIN_RATIO)
        elif phase == "TEST":
            _, data = split_data(data, TRAIN_RATIO)
            _, htf_data = split_data(htf_data, TRAIN_RATIO)

        print(f"  {pair}: H1={len(data)} bars ({data.index[0].date()} to {data.index[-1].date()})", end=" ")

        t0 = time.time()
        report, trade_log = run_backtest(strat_cls, data, htf_data, pair, label)
        elapsed = time.time() - t0

        n = report.get("total_trades", 0)
        print(f"-> {n} trades ({elapsed:.0f}s)")

        all_reports.append(report)
        if len(trade_log) > 0:
            all_trades.append(trade_log)

    # Aggregate results
    total_trades = sum(r.get("total_trades", 0) for r in all_reports)
    if total_trades == 0:
        print(f"  NO TRADES across any pair!")
        return {"total_trades": 0, "reports": all_reports}

    total_wins = sum(
        r.get("total_trades", 0) * r.get("win_rate_pct", 0) / 100
        for r in all_reports
    )
    total_pnl_pips = sum(r.get("total_pnl_pips", 0) for r in all_reports)
    total_pnl_dollars = sum(r.get("total_pnl_dollars", 0) for r in all_reports)

    # Combined trade log for aggregate PF
    combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if len(combined) > 0:
        gross_profit = combined.loc[combined["pnl_pips"] > 0, "pnl_pips"].sum()
        gross_loss = abs(combined.loc[combined["pnl_pips"] < 0, "pnl_pips"].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
        avg_win = combined.loc[combined["pnl_pips"] > 0, "pnl_pips"].mean() if (combined["pnl_pips"] > 0).any() else 0
        avg_loss = abs(combined.loc[combined["pnl_pips"] < 0, "pnl_pips"].mean()) if (combined["pnl_pips"] < 0).any() else 0
        avg_rr = avg_win / avg_loss if avg_loss > 0 else 0
        max_dd = min(r.get("max_drawdown_pct", 0) for r in all_reports)
    else:
        pf = avg_win = avg_loss = avg_rr = max_dd = 0

    win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
    expectancy = total_pnl_pips / total_trades if total_trades > 0 else 0

    print(f"\n  --- {label} {phase} AGGREGATE ---")
    print(f"  Trades: {total_trades}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Avg RR: {avg_rr:.2f}")
    print(f"  Expectancy: {expectancy:.2f} pips")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Worst Max DD: {max_dd:.2f}%")
    print(f"  Total PnL: {total_pnl_pips:.1f} pips / ${total_pnl_dollars:,.2f}")
    print(f"  Avg Win: {avg_win:.1f}p | Avg Loss: {avg_loss:.1f}p")

    # Per-pair breakdown
    for i, pair in enumerate(PAIRS):
        if i < len(all_reports):
            r = all_reports[i]
            n = r.get("total_trades", 0)
            wr = r.get("win_rate_pct", 0)
            pfp = r.get("profit_factor", 0)
            pnl = r.get("total_pnl_pips", 0)
            dd = r.get("max_drawdown_pct", 0)
            print(f"    {pair}: {n} trades, WR {wr:.1f}%, PF {pfp:.2f}, PnL {pnl:.1f}p, DD {dd:.2f}%")

    # Save trade logs
    if len(combined) > 0:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, f"{label}_{phase}_trades.csv")
        combined.to_csv(csv_path, index=False)
        print(f"  Trade log saved: {csv_path}")

    return {
        "total_trades": total_trades, "win_rate": round(win_rate, 2),
        "avg_rr": round(avg_rr, 2), "expectancy": round(expectancy, 2),
        "profit_factor": round(pf, 2), "max_dd": round(max_dd, 2),
        "total_pnl_pips": round(total_pnl_pips, 1),
        "total_pnl_dollars": round(total_pnl_dollars, 2),
        "avg_win": round(avg_win, 1), "avg_loss": round(avg_loss, 1),
        "reports": all_reports,
    }


if __name__ == "__main__":
    from src.strategies_pkg.s7_liquidity_sweep import S7_Liquidity_Sweep
    from src.strategies_pkg.s8_order_block import S8_Order_Block
    from src.strategies_pkg.s9_london_session import S9_London_Session

    strategies = {
        "S7": ("S7 (Liquidity Sweep Reversal)", S7_Liquidity_Sweep),
        "S8": ("S8 (Order Block Retest)", S8_Order_Block),
        "S9": ("S9 (London Session Gap)", S9_London_Session),
    }

    # Parse CLI args
    variant = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"

    results = {}

    for key, (label, strat_cls) in strategies.items():
        if variant != "ALL" and variant != key:
            continue

        print(f"\n{'#' * 70}")
        print(f"# {label}")
        print(f"{'#' * 70}")

        # Full 6-month run
        full = run_variant(key, strat_cls, phase="FULL")

        # 70/30 split: train
        train = run_variant(key, strat_cls, phase="TRAIN")

        # 70/30 split: test (OOS)
        test = run_variant(key, strat_cls, phase="TEST")

        results[key] = {"full": full, "train": train, "test": test}

    # Summary comparison
    if len(results) > 1:
        print(f"\n{'=' * 80}")
        print("STRATEGY COMPARISON — FULL 6 MONTHS")
        print(f"{'=' * 80}")
        print(f"{'Metric':<20}", end="")
        for k in results:
            print(f"{k:>18}", end="")
        print()
        print("-" * (20 + 18 * len(results)))
        for metric in ["total_trades", "win_rate", "avg_rr", "expectancy",
                        "profit_factor", "max_dd", "total_pnl_pips",
                        "total_pnl_dollars", "avg_win", "avg_loss"]:
            print(f"{metric:<20}", end="")
            for k in results:
                v = results[k]["full"].get(metric, 0)
                if isinstance(v, float):
                    print(f"{v:>18.2f}", end="")
                else:
                    print(f"{v:>18}", end="")
            print()

        print(f"\n{'=' * 80}")
        print("TRAIN vs TEST (70/30 Split)")
        print(f"{'=' * 80}")
        for k in results:
            train_r = results[k]["train"]
            test_r = results[k]["test"]
            print(f"\n  {k}:")
            print(f"    TRAIN: {train_r.get('total_trades',0)} trades, "
                  f"WR {train_r.get('win_rate',0):.1f}%, "
                  f"PF {train_r.get('profit_factor',0):.2f}, "
                  f"PnL {train_r.get('total_pnl_pips',0):.1f}p")
            print(f"    TEST:  {test_r.get('total_trades',0)} trades, "
                  f"WR {test_r.get('win_rate',0):.1f}%, "
                  f"PF {test_r.get('profit_factor',0):.2f}, "
                  f"PnL {test_r.get('total_pnl_pips',0):.1f}p")
            if train_r.get('profit_factor', 0) > 0 and test_r.get('profit_factor', 0) > 0:
                gen_ratio = test_r['profit_factor'] / train_r['profit_factor']
                print(f"    Generalization Ratio (PF): {gen_ratio:.2f}")

    # Save summary
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {}
    for k, v in results.items():
        summary[k] = {
            phase: {mk: mv for mk, mv in data.items() if mk != "reports"}
            for phase, data in v.items()
        }
    with open(os.path.join(RESULTS_DIR, "s789_comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {os.path.join(RESULTS_DIR, 's789_comparison.json')}")
