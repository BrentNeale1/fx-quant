"""
S9 Full Dataset Test — Additional London-centric pairs.
Tests EUR_GBP, EUR_USD, GBP_NZD on the full 2021-2023 dataset.
(GBP_AUD and GBP_USD already tested previously.)

Uses FULL dataset with 70/30 train/test split.
"""
import os
import sys
import json
import time
import io

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from src.indicators.technical import compute_all_indicators
from src.backtester.engine import Backtester

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase1")

# New pairs to test (London-centric: GBP or EUR)
PAIRS = ["EUR_GBP", "EUR_USD", "GBP_NZD"]

TRAIN_RATIO = 0.70


def load_data(pair, tf):
    """Load full dataset and compute indicators."""
    fp = os.path.join(PROCESSED_DIR, f"{pair}_{tf}.csv")
    if not os.path.exists(fp):
        print(f"  WARNING: {fp} not found, skipping")
        return None
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    df = compute_all_indicators(df)
    return df


def split_data(df, train_ratio=0.70):
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def run_pair(strat_cls, pair, data, htf_data, label, phase):
    """Run backtest on a single pair/phase."""
    if phase == "TRAIN":
        data, _ = split_data(data, TRAIN_RATIO)
        htf_data, _ = split_data(htf_data, TRAIN_RATIO)
    elif phase == "TEST":
        _, data = split_data(data, TRAIN_RATIO)
        _, htf_data = split_data(htf_data, TRAIN_RATIO)

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


if __name__ == "__main__":
    from src.strategies_pkg.s9_london_session import S9_London_Session

    results = {}

    for pair in PAIRS:
        print(f"\n{'#' * 70}")
        print(f"# S9 London Session — {pair} (Full Dataset)")
        print(f"{'#' * 70}")

        data = load_data(pair, "H1")
        if data is None:
            continue

        htf_data = data.copy()
        print(f"  Data: {len(data)} H1 bars, {data.index[0].date()} to {data.index[-1].date()}")

        pair_results = {}
        for phase in ["FULL", "TRAIN", "TEST"]:
            t0 = time.time()
            report, trade_log = run_pair(S9_London_Session, pair, data, htf_data, "S9", phase)
            elapsed = time.time() - t0

            n = report.get("total_trades", 0)
            wr = report.get("win_rate_pct", 0)
            pf = report.get("profit_factor", 0)
            pnl = report.get("total_pnl_pips", 0)
            pnl_d = report.get("total_pnl_dollars", 0)
            dd = report.get("max_drawdown_pct", 0)
            avg_w = report.get("avg_win_pips", 0)
            avg_l = report.get("avg_loss_pips", 0)
            avg_rr = avg_w / avg_l if avg_l > 0 else 0

            print(f"\n  {phase}: {n} trades, WR {wr:.1f}%, PF {pf:.2f}, "
                  f"PnL {pnl:.1f}p (${pnl_d:,.2f}), DD {dd:.2f}%, "
                  f"AvgW {avg_w:.1f}p / AvgL {avg_l:.1f}p, RR {avg_rr:.2f} ({elapsed:.0f}s)")

            pair_results[phase] = {
                "total_trades": n, "win_rate": round(wr, 2),
                "avg_rr": round(avg_rr, 2),
                "expectancy": round(pnl / n, 2) if n > 0 else 0,
                "profit_factor": round(pf, 2),
                "max_dd": round(dd, 2),
                "total_pnl_pips": round(pnl, 1),
                "total_pnl_dollars": round(pnl_d, 2),
                "avg_win": round(avg_w, 1), "avg_loss": round(avg_l, 1),
            }

            # Save trade log
            if len(trade_log) > 0:
                os.makedirs(RESULTS_DIR, exist_ok=True)
                csv_path = os.path.join(RESULTS_DIR, f"S9_{pair}_{phase}_trades.csv")
                trade_log.to_csv(csv_path, index=False)

        # Generalization check
        train_pf = pair_results.get("TRAIN", {}).get("profit_factor", 0)
        test_pf = pair_results.get("TEST", {}).get("profit_factor", 0)
        if train_pf > 0 and test_pf > 0:
            gen = test_pf / train_pf
            print(f"\n  Generalization Ratio: {gen:.2f} (OOS PF / IS PF)")

        results[pair] = pair_results

    # Summary table
    print(f"\n{'=' * 80}")
    print("S9 ADDITIONAL PAIRS — SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Pair':<12} {'Phase':<8} {'Trades':>7} {'WR%':>7} {'PF':>7} {'PnL(p)':>10} {'DD%':>7} {'AvgW':>7} {'AvgL':>7} {'RR':>6}")
    print("-" * 80)
    for pair, phases in results.items():
        for phase in ["FULL", "TRAIN", "TEST"]:
            d = phases.get(phase, {})
            print(f"{pair:<12} {phase:<8} {d.get('total_trades',0):>7} "
                  f"{d.get('win_rate',0):>6.1f}% {d.get('profit_factor',0):>6.2f} "
                  f"{d.get('total_pnl_pips',0):>9.1f}p {d.get('max_dd',0):>6.2f}% "
                  f"{d.get('avg_win',0):>6.1f} {d.get('avg_loss',0):>6.1f} "
                  f"{d.get('avg_rr',0):>5.2f}")
        print()

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "s9_additional_pairs.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out_path}")
