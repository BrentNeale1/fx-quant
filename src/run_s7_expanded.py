"""
S7 Liquidity Sweep — Expanded testing.

1. GBP_JPY: Full dataset (2021-2023) with 70/30 split
2. Other 5 pairs: Last 6 months with 70/30 split
   - GBP_USD, EUR_AUD, GBP_AUD, EUR_USD, GBP_NZD

Uses the tight SL variant (1.0 ATR) which proved best on GBP_JPY 6-month.
"""
import os, sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from src.indicators.technical import compute_all_indicators
from src.backtester.engine import Backtester
from src.strategies_pkg.s7_liquidity_sweep import S7_Liquidity_Sweep

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase1")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_RATIO = 0.70


def load_data(pair, last_n_months=None):
    """Load H1 data, optionally slice to last N months (with warmup)."""
    fp = os.path.join(PROCESSED_DIR, f"{pair}_H1.csv")
    if not os.path.exists(fp):
        print(f"  WARNING: {fp} not found, skipping")
        return None
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df.index.name = "timestamp"

    if last_n_months is not None and last_n_months > 0:
        end_date = df.index[-1]
        start_date = end_date - pd.DateOffset(months=last_n_months)
        warmup_start = start_date - pd.DateOffset(days=60)
        df = df[df.index >= warmup_start]

    df = compute_all_indicators(df)

    if last_n_months is not None and last_n_months > 0:
        df = df[df.index >= start_date]

    return df


def split_data(df):
    split_idx = int(len(df) * TRAIN_RATIO)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def run_test(pair, data, htf_data, phase="FULL"):
    if phase == "TRAIN":
        data, _ = split_data(data)
        htf_data, _ = split_data(htf_data)
    elif phase == "TEST":
        _, data = split_data(data)
        _, htf_data = split_data(htf_data)

    strategy = S7_Liquidity_Sweep()
    bt = Backtester(data=data, strategy=strategy, pair=pair,
                    starting_equity=100_000.0, htf_data=htf_data)
    report = bt.run()
    trade_log = bt.get_trade_log_df()
    return report, trade_log


def extract_metrics(report):
    n = report.get("total_trades", 0)
    wr = report.get("win_rate_pct", 0)
    pf = report.get("profit_factor", 0)
    pnl = report.get("total_pnl_pips", 0)
    pnl_d = report.get("total_pnl_dollars", 0)
    dd = report.get("max_drawdown_pct", 0)
    aw = report.get("avg_win_pips", 0)
    al = report.get("avg_loss_pips", 0)
    rr = aw / al if al > 0 else 0
    exp = pnl / n if n > 0 else 0
    return {
        "trades": n, "wr": round(wr, 1), "pf": round(pf, 2),
        "pnl_pips": round(pnl, 1), "pnl_usd": round(pnl_d, 2),
        "dd": round(dd, 2), "avg_win": round(aw, 1),
        "avg_loss": round(al, 1), "rr": round(rr, 2),
        "expectancy": round(exp, 2),
    }


if __name__ == "__main__":
    all_results = {}

    # ---- TEST 1: GBP_JPY Full Dataset ----
    pair = "GBP_JPY"
    print(f"\n{'#' * 70}")
    print(f"# S7 Tight — {pair} FULL DATASET (2021-2023)")
    print(f"{'#' * 70}")

    data = load_data(pair, last_n_months=None)
    htf = data.copy()
    print(f"  Data: {len(data)} H1 bars, {data.index[0].date()} to {data.index[-1].date()}")

    pair_results = {}
    for phase in ["FULL", "TRAIN", "TEST"]:
        t0 = time.time()
        report, trade_log = run_test(pair, data, htf, phase)
        elapsed = time.time() - t0
        m = extract_metrics(report)
        pair_results[phase] = m

        print(f"  {phase:>5}: {m['trades']:>3} trades | WR {m['wr']:>5.1f}% | "
              f"PF {m['pf']:>5.2f} | PnL {m['pnl_pips']:>+8.1f}p | "
              f"DD {m['dd']:>6.2f}% | AvgW {m['avg_win']:>5.1f} / "
              f"AvgL {m['avg_loss']:>5.1f} | RR {m['rr']:>4.2f} | "
              f"Exp {m['expectancy']:>+6.2f}p ({elapsed:.0f}s)")

        if len(trade_log) > 0:
            trade_log.to_csv(os.path.join(RESULTS_DIR, f"S7T_{pair}_{phase}_trades.csv"), index=False)

    train_pf = pair_results.get("TRAIN", {}).get("pf", 0)
    test_pf = pair_results.get("TEST", {}).get("pf", 0)
    if train_pf > 0 and test_pf > 0:
        print(f"  Gen Ratio: {test_pf / train_pf:.2f}")

    all_results[pair] = {"mode": "FULL_DATASET", **pair_results}

    # ---- TEST 2: Other 5 pairs, 6 months ----
    other_pairs = ["GBP_USD", "EUR_AUD", "GBP_AUD", "EUR_USD", "GBP_NZD"]

    for pair in other_pairs:
        print(f"\n{'#' * 70}")
        print(f"# S7 Tight — {pair} (6 months)")
        print(f"{'#' * 70}")

        data = load_data(pair, last_n_months=6)
        if data is None:
            continue
        htf = data.copy()
        print(f"  Data: {len(data)} H1 bars, {data.index[0].date()} to {data.index[-1].date()}")

        pair_results = {}
        for phase in ["FULL", "TRAIN", "TEST"]:
            t0 = time.time()
            report, trade_log = run_test(pair, data, htf, phase)
            elapsed = time.time() - t0
            m = extract_metrics(report)
            pair_results[phase] = m

            print(f"  {phase:>5}: {m['trades']:>3} trades | WR {m['wr']:>5.1f}% | "
                  f"PF {m['pf']:>5.2f} | PnL {m['pnl_pips']:>+8.1f}p | "
                  f"DD {m['dd']:>6.2f}% | AvgW {m['avg_win']:>5.1f} / "
                  f"AvgL {m['avg_loss']:>5.1f} | RR {m['rr']:>4.2f} | "
                  f"Exp {m['expectancy']:>+6.2f}p ({elapsed:.0f}s)")

            if len(trade_log) > 0:
                trade_log.to_csv(os.path.join(RESULTS_DIR, f"S7T_{pair}_{phase}_trades.csv"), index=False)

        train_pf = pair_results.get("TRAIN", {}).get("pf", 0)
        test_pf = pair_results.get("TEST", {}).get("pf", 0)
        if train_pf > 0 and test_pf > 0:
            print(f"  Gen Ratio: {test_pf / train_pf:.2f}")

        all_results[pair] = {"mode": "6_MONTHS", **pair_results}

    # ---- SUMMARY TABLE ----
    print(f"\n{'=' * 90}")
    print("S7 TIGHT (1.0 ATR SL) — ALL PAIRS SUMMARY")
    print(f"{'=' * 90}")
    print(f"{'Pair':<10} {'Mode':<12} {'Phase':<6} {'Trades':>6} {'WR%':>6} "
          f"{'PF':>6} {'PnL(p)':>9} {'RR':>5} {'Exp(p)':>7} {'DD%':>7}")
    print("-" * 90)
    for pair, data in all_results.items():
        mode = data["mode"]
        for phase in ["FULL", "TEST"]:
            m = data.get(phase, {})
            if not m:
                continue
            print(f"{pair:<10} {mode:<12} {phase:<6} {m['trades']:>6} "
                  f"{m['wr']:>5.1f}% {m['pf']:>5.2f} {m['pnl_pips']:>+8.1f}p "
                  f"{m['rr']:>4.2f} {m['expectancy']:>+6.2f}p {m['dd']:>6.2f}%")
        print()

    # Save
    out_path = os.path.join(RESULTS_DIR, "s7_expanded_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved: {out_path}")
