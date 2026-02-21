"""
S19 VWAP Deviation Mean Reversion — Multi-Period Backtest Runner.

Runs S19 across 3 periods and 4 pairs:

Periods:
  1. Training   (Jan 2021 - Aug 2023): data/processed/
  2. Test Set   (Sep 2023 - Dec 2024): warmup from processed tail + data/test_set/
  3. 2025       (Jan 2025 - Dec 2025): data/2025/ (has Oct 2024 warmup built in)

Pairs: EUR_USD, GBP_USD, GBP_JPY, GBP_AUD
"""
import os, sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from src.indicators.technical import compute_all_indicators
from src.backtester.engine import Backtester
from src.strategies_pkg.s19_vwap_deviation import S19_VWAPDeviationMeanReversion

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TEST_SET_DIR = os.path.join(BASE_DIR, "data", "test_set")
DATA_2025_DIR = os.path.join(BASE_DIR, "data", "2025")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "phase1")
os.makedirs(RESULTS_DIR, exist_ok=True)

PAIRS = ["EUR_USD", "GBP_USD", "GBP_JPY", "GBP_AUD"]

PERIODS = {
    "training": {"start": "2021-01-01", "end": "2023-08-31"},
    "test_set": {"start": "2023-09-01", "end": "2024-12-31"},
    "2025":     {"start": "2025-01-01", "end": "2025-12-31"},
}

WARMUP_ROWS = 500


def load_training_data(pair):
    """Load processed data for training period."""
    m15_fp = os.path.join(PROCESSED_DIR, f"{pair}_M15.csv")
    h1_fp = os.path.join(PROCESSED_DIR, f"{pair}_H1.csv")

    if not os.path.exists(m15_fp):
        print(f"    WARNING: {m15_fp} not found")
        return None, None

    m15 = pd.read_csv(m15_fp, index_col=0, parse_dates=True)
    m15.index.name = "timestamp"
    m15 = compute_all_indicators(m15)

    h1 = None
    if os.path.exists(h1_fp):
        h1 = pd.read_csv(h1_fp, index_col=0, parse_dates=True)
        h1.index.name = "timestamp"
        h1 = compute_all_indicators(h1)

    return m15, h1


def load_test_set_data(pair):
    """Load warmup (tail of processed) + test set data."""
    proc_fp = os.path.join(PROCESSED_DIR, f"{pair}_M15.csv")
    test_fp = os.path.join(TEST_SET_DIR, f"{pair}_M15.csv")
    proc_h1_fp = os.path.join(PROCESSED_DIR, f"{pair}_H1.csv")
    test_h1_fp = os.path.join(TEST_SET_DIR, f"{pair}_H1.csv")

    if not os.path.exists(proc_fp) or not os.path.exists(test_fp):
        print(f"    WARNING: Missing M15 data for {pair} (processed or test_set)")
        return None, None

    # M15
    proc_df = pd.read_csv(proc_fp, index_col=0, parse_dates=True)
    warmup = proc_df.tail(WARMUP_ROWS)
    test_df = pd.read_csv(test_fp, index_col=0, parse_dates=True)
    combined = pd.concat([warmup, test_df])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined.sort_index(inplace=True)
    combined.index.name = "timestamp"
    m15 = compute_all_indicators(combined)

    # H1
    h1 = None
    if os.path.exists(proc_h1_fp) and os.path.exists(test_h1_fp):
        proc_h1 = pd.read_csv(proc_h1_fp, index_col=0, parse_dates=True)
        warmup_h1 = proc_h1.tail(WARMUP_ROWS)
        test_h1 = pd.read_csv(test_h1_fp, index_col=0, parse_dates=True)
        combined_h1 = pd.concat([warmup_h1, test_h1])
        combined_h1 = combined_h1[~combined_h1.index.duplicated(keep='last')]
        combined_h1.sort_index(inplace=True)
        combined_h1.index.name = "timestamp"
        h1 = compute_all_indicators(combined_h1)

    return m15, h1


def load_2025_data(pair):
    """Load 2025 data (includes Oct 2024 warmup)."""
    m15_fp = os.path.join(DATA_2025_DIR, f"{pair}_M15.csv")
    h1_fp = os.path.join(DATA_2025_DIR, f"{pair}_H1.csv")

    if not os.path.exists(m15_fp):
        print(f"    WARNING: {m15_fp} not found")
        return None, None

    m15 = pd.read_csv(m15_fp, index_col=0, parse_dates=True)
    m15.index.name = "timestamp"
    m15 = compute_all_indicators(m15)

    h1 = None
    if os.path.exists(h1_fp):
        h1 = pd.read_csv(h1_fp, index_col=0, parse_dates=True)
        h1.index.name = "timestamp"
        h1 = compute_all_indicators(h1)

    return m15, h1


def run_backtest(data, htf_data, pair, period_start, period_end):
    """Run S19 on data, filter trades to period."""
    if data is None or len(data) < 250:
        return pd.DataFrame()

    strategy = S19_VWAPDeviationMeanReversion()
    bt = Backtester(data=data, strategy=strategy, pair=pair,
                    starting_equity=100_000.0, htf_data=htf_data)
    bt.run()
    trade_log = bt.get_trade_log_df()

    # Filter to period
    if not trade_log.empty:
        ts = pd.to_datetime(trade_log["timestamp"])
        start_ts = pd.Timestamp(period_start)
        end_ts = pd.Timestamp(f"{period_end} 23:59:59")
        if ts.dt.tz is not None:
            start_ts = start_ts.tz_localize(ts.dt.tz)
            end_ts = end_ts.tz_localize(ts.dt.tz)
        trade_log = trade_log[(ts >= start_ts) & (ts <= end_ts)]

    return trade_log


def compute_metrics(trade_log):
    if trade_log.empty or len(trade_log) == 0:
        return {
            "trades": 0, "wr": 0, "pf": 0, "sharpe": 0,
            "pnl_pips": 0, "max_dd_pips": 0, "expectancy": 0,
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

    if n > 1:
        pnl_series = trade_log["pnl_pips"]
        sharpe = (pnl_series.mean() / pnl_series.std()) * np.sqrt(252) \
            if pnl_series.std() > 0 else 0
    else:
        sharpe = 0

    cum_pnl = trade_log["pnl_pips"].cumsum()
    peak = cum_pnl.cummax()
    dd = cum_pnl - peak
    max_dd = dd.min() if len(dd) > 0 else 0

    return {
        "trades": n,
        "wr": round(wr, 1),
        "pf": round(pf, 2),
        "sharpe": round(sharpe, 2),
        "pnl_pips": round(total_pnl, 1),
        "max_dd_pips": round(max_dd, 1),
        "expectancy": round(expectancy, 2),
    }


def fmt_pf(pf):
    return f"{pf:.2f}" if pf != float('inf') else "inf"


def main():
    t0 = time.time()

    print(f"{'='*110}")
    print("S19 VWAP DEVIATION MEAN REVERSION — Multi-Period Backtest")
    print(f"  Pairs: {', '.join(PAIRS)}")
    print(f"  Periods: Training (2021-01 to 2023-08), Test Set (2023-09 to 2024-12), 2025 (2025-01 to 2025-12)")
    print(f"{'='*110}")

    all_results = {}

    for pair in PAIRS:
        print(f"\n{'─'*110}")
        print(f"  {pair}")
        print(f"{'─'*110}")

        pair_results = {}

        # --- Training period ---
        print(f"    Loading training data...", end=" ", flush=True)
        m15, h1 = load_training_data(pair)
        if m15 is not None:
            print(f"{len(m15)} M15 bars" + (f", {len(h1)} H1 bars" if h1 is not None else ""))
            trades = run_backtest(m15, h1, pair,
                                 PERIODS["training"]["start"], PERIODS["training"]["end"])
            metrics = compute_metrics(trades)
            pair_results["training"] = metrics
            if not trades.empty:
                trades.to_csv(os.path.join(RESULTS_DIR, f"S19_{pair}_training_trades.csv"),
                              index=False)
            print(f"    Training: {metrics['trades']:>4}t  WR={metrics['wr']:>5.1f}%  "
                  f"PF={fmt_pf(metrics['pf']):>5}  Sharpe={metrics['sharpe']:>6.2f}  "
                  f"PnL={metrics['pnl_pips']:>+8.1f}p  DD={metrics['max_dd_pips']:>+7.1f}p  "
                  f"Exp={metrics['expectancy']:>+6.2f}")
        else:
            print("MISSING")
            pair_results["training"] = compute_metrics(pd.DataFrame())

        # --- Test Set period ---
        print(f"    Loading test set data...", end=" ", flush=True)
        m15, h1 = load_test_set_data(pair)
        if m15 is not None:
            print(f"{len(m15)} M15 bars" + (f", {len(h1)} H1 bars" if h1 is not None else ""))
            trades = run_backtest(m15, h1, pair,
                                 PERIODS["test_set"]["start"], PERIODS["test_set"]["end"])
            metrics = compute_metrics(trades)
            pair_results["test_set"] = metrics
            if not trades.empty:
                trades.to_csv(os.path.join(RESULTS_DIR, f"S19_{pair}_test_set_trades.csv"),
                              index=False)
            print(f"    Test Set: {metrics['trades']:>4}t  WR={metrics['wr']:>5.1f}%  "
                  f"PF={fmt_pf(metrics['pf']):>5}  Sharpe={metrics['sharpe']:>6.2f}  "
                  f"PnL={metrics['pnl_pips']:>+8.1f}p  DD={metrics['max_dd_pips']:>+7.1f}p  "
                  f"Exp={metrics['expectancy']:>+6.2f}")
        else:
            print("MISSING")
            pair_results["test_set"] = compute_metrics(pd.DataFrame())

        # --- 2025 period ---
        print(f"    Loading 2025 data...", end=" ", flush=True)
        m15, h1 = load_2025_data(pair)
        if m15 is not None:
            print(f"{len(m15)} M15 bars" + (f", {len(h1)} H1 bars" if h1 is not None else ""))
            trades = run_backtest(m15, h1, pair,
                                 PERIODS["2025"]["start"], PERIODS["2025"]["end"])
            metrics = compute_metrics(trades)
            pair_results["2025"] = metrics
            if not trades.empty:
                trades.to_csv(os.path.join(RESULTS_DIR, f"S19_{pair}_2025_trades.csv"),
                              index=False)
            print(f"    2025:     {metrics['trades']:>4}t  WR={metrics['wr']:>5.1f}%  "
                  f"PF={fmt_pf(metrics['pf']):>5}  Sharpe={metrics['sharpe']:>6.2f}  "
                  f"PnL={metrics['pnl_pips']:>+8.1f}p  DD={metrics['max_dd_pips']:>+7.1f}p  "
                  f"Exp={metrics['expectancy']:>+6.2f}")
        else:
            print("MISSING")
            pair_results["2025"] = compute_metrics(pd.DataFrame())

        all_results[pair] = pair_results

    # ===== Summary comparison table =====
    print(f"\n{'='*110}")
    print("SUMMARY — S19 VWAP Deviation Mean Reversion")
    print(f"{'='*110}")
    print(f"  {'Pair':<10} {'Period':<10} {'Trades':>7} {'WR%':>7} {'PF':>7} "
          f"{'Sharpe':>7} {'PnL':>9} {'MaxDD':>9} {'Exp':>7} {'Verdict':>8}")
    print(f"  {'-'*90}")

    agg_by_period = {"training": [], "test_set": [], "2025": []}

    for pair in PAIRS:
        for period_key, period_label in [("training", "Training"),
                                          ("test_set", "TestSet"),
                                          ("2025", "2025")]:
            m = all_results[pair].get(period_key, {})
            if not m or m.get("trades", 0) == 0:
                print(f"  {pair:<10} {period_label:<10} {'—':>7} {'—':>7} {'—':>7} "
                      f"{'—':>7} {'—':>9} {'—':>9} {'—':>7} {'NO DATA':>8}")
                continue

            verdict = "PASS" if m["pf"] > 1.0 and m["trades"] >= 5 else "FAIL"
            if m["trades"] < 5 and m["pf"] > 1.0:
                verdict = "LOW_N"

            print(f"  {pair:<10} {period_label:<10} {m['trades']:>7} {m['wr']:>6.1f}% "
                  f"{fmt_pf(m['pf']):>7} {m['sharpe']:>7.2f} {m['pnl_pips']:>+9.1f} "
                  f"{m['max_dd_pips']:>+9.1f} {m['expectancy']:>+7.2f} {verdict:>8}")

    # Aggregate by period across all pairs
    print(f"\n  {'─'*90}")
    print(f"  AGGREGATE BY PERIOD:")
    print(f"  {'-'*90}")

    for period_key, period_label in [("training", "Training"),
                                      ("test_set", "TestSet"),
                                      ("2025", "2025")]:
        total_trades = 0
        total_pnl = 0
        total_gross_profit = 0
        total_gross_loss = 0
        total_wins = 0

        for pair in PAIRS:
            m = all_results[pair].get(period_key, {})
            total_trades += m.get("trades", 0)
            total_pnl += m.get("pnl_pips", 0)

        if total_trades > 0:
            # Recompute aggregate WR/PF from saved trade logs
            agg_wr = 0
            agg_pf = 0
            print(f"  {'ALL PAIRS':<10} {period_label:<10} {total_trades:>7} "
                  f"{'':>7} {'':>7} {'':>7} {total_pnl:>+9.1f} "
                  f"{'':>9} {total_pnl/total_trades:>+7.2f} "
                  f"{'PASS' if total_pnl > 0 else 'FAIL':>8}")
        else:
            print(f"  {'ALL PAIRS':<10} {period_label:<10} {'—':>7}")

    # Save JSON results
    out_path = os.path.join(RESULTS_DIR, "s19_backtest.json")

    def json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return str(obj)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=json_default)
    print(f"\nResults saved: {out_path}")

    elapsed = time.time() - t0
    print(f"Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
