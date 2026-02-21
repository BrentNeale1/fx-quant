"""
S3 Key Level Breakout — Multi-Pair Expansion Test.

Tests S3 across 4 pairs (GBP_JPY, GBP_USD, EUR_USD, GBP_AUD) on H1 across
all 4 validation periods:
  - IS:       Jan 2021 - Dec 2022  (data/processed/)
  - OOS:      Jan 2023 - Aug 2023  (data/processed/)
  - Test Set: Sep 2023 - Dec 2024  (data/processed/ warmup + data/test_set/)
  - 2025:     Jan 2025 - Dec 2025  (data/2025/)

Goal: Find pairs where S3 generalizes across all periods (PF > 1.0).
"""
import os, sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from src.indicators.technical import compute_all_indicators
from src.backtester.engine import Backtester
from src.strategies_pkg.s3_key_level_breakout import S3_KeyLevel_Breakout

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TEST_SET_DIR = os.path.join(BASE_DIR, "data", "test_set")
DATA_2025_DIR = os.path.join(BASE_DIR, "data", "2025")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "s3_expansion")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Period definitions
IS_START = "2021-01-01"
IS_END = "2022-12-31"
OOS_START = "2023-01-01"
OOS_END = "2023-08-31"
TEST_START = "2023-09-01"
TEST_END = "2024-12-31"
FWD_START = "2025-01-01"
FWD_END = "2025-12-31"
WARMUP_DAYS = 60
WARMUP_ROWS = 500

PAIRS = ["GBP_JPY", "GBP_USD", "EUR_USD", "GBP_AUD"]
TF = "H1"


def load_processed(pair):
    fp = os.path.join(PROCESSED_DIR, f"{pair}_{TF}.csv")
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    return compute_all_indicators(df)


def load_test_set(pair, processed_data):
    """Concatenate warmup from processed + test set data."""
    test_fp = os.path.join(TEST_SET_DIR, f"{pair}_{TF}.csv")
    if not os.path.exists(test_fp):
        return None
    warmup = processed_data.tail(WARMUP_ROWS)
    test_df = pd.read_csv(test_fp, index_col=0, parse_dates=True)
    combined = pd.concat([warmup, test_df])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined.sort_index(inplace=True)
    combined.index.name = "timestamp"
    return compute_all_indicators(combined)


def load_2025(pair):
    fp = os.path.join(DATA_2025_DIR, f"{pair}_{TF}.csv")
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    return compute_all_indicators(df)


def slice_period(df, start, end, warmup_days=WARMUP_DAYS):
    if df.index.tz is not None:
        start_ts = pd.Timestamp(start, tz=df.index.tz)
        end_ts = pd.Timestamp(f"{end} 23:59:59", tz=df.index.tz)
    else:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(f"{end} 23:59:59")
    warmup_start = start_ts - pd.DateOffset(days=warmup_days)
    sliced = df[(df.index >= warmup_start) & (df.index <= end_ts)].copy()
    return sliced, start_ts


def run_backtest(data, pair, start_filter):
    """Run S3 backtest on data, filter trades after start_filter timestamp."""
    if len(data) < 250:
        return pd.DataFrame()

    strategy = S3_KeyLevel_Breakout()
    bt = Backtester(data=data, strategy=strategy, pair=pair,
                    starting_equity=100_000.0, htf_data=data)
    bt.run()
    trade_log = bt.get_trade_log_df()

    if not trade_log.empty:
        ts = pd.to_datetime(trade_log["timestamp"])
        filter_ts = pd.Timestamp(start_filter)
        if ts.dt.tz is not None and filter_ts.tz is None:
            filter_ts = filter_ts.tz_localize(ts.dt.tz)
        trade_log = trade_log[ts >= filter_ts]

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
    return f"{pf:.2f}" if pf != float("inf") else "inf"


def main():
    t0 = time.time()
    all_results = {}

    print(f"{'='*110}")
    print("S3 KEY LEVEL BREAKOUT — MULTI-PAIR EXPANSION")
    print(f"  IS:       {IS_START} to {IS_END}")
    print(f"  OOS:      {OOS_START} to {OOS_END}")
    print(f"  Test Set: {TEST_START} to {TEST_END}")
    print(f"  2025:     {FWD_START} to {FWD_END}")
    print(f"  Pairs:    {', '.join(PAIRS)}")
    print(f"{'='*110}")

    for pair in PAIRS:
        print(f"\n{'─'*110}")
        print(f"  S3 / {pair} (H1)")
        print(f"{'─'*110}")

        results = {"pair": pair, "tf": TF}

        # 1. Load processed data for IS + OOS
        proc_data = load_processed(pair)
        if proc_data is None:
            print(f"    SKIP: no processed data for {pair}")
            continue
        print(f"    Processed: {len(proc_data)} bars ({proc_data.index[0].date()} to {proc_data.index[-1].date()})")

        # 2. IS period
        is_sliced, is_start_ts = slice_period(proc_data, IS_START, IS_END)
        is_log = run_backtest(is_sliced, pair, is_start_ts)
        is_m = compute_metrics(is_log)
        print(f"    IS:       {is_m['trades']:>4}t  WR={is_m['wr']:>5.1f}%  PF={fmt_pf(is_m['pf']):>5}  "
              f"PnL={is_m['pnl_pips']:>+8.1f}p  DD={is_m['max_dd_pips']:>+7.1f}p  Exp={is_m['expectancy']:>+6.2f}")
        results["is_metrics"] = is_m
        if not is_log.empty:
            is_log.to_csv(os.path.join(RESULTS_DIR, f"trades_S3_{pair}_IS.csv"), index=False)

        # 3. OOS period
        oos_sliced, oos_start_ts = slice_period(proc_data, OOS_START, OOS_END)
        oos_log = run_backtest(oos_sliced, pair, oos_start_ts)
        oos_m = compute_metrics(oos_log)
        print(f"    OOS:      {oos_m['trades']:>4}t  WR={oos_m['wr']:>5.1f}%  PF={fmt_pf(oos_m['pf']):>5}  "
              f"PnL={oos_m['pnl_pips']:>+8.1f}p  DD={oos_m['max_dd_pips']:>+7.1f}p  Exp={oos_m['expectancy']:>+6.2f}")
        results["oos_metrics"] = oos_m
        if not oos_log.empty:
            oos_log.to_csv(os.path.join(RESULTS_DIR, f"trades_S3_{pair}_OOS.csv"), index=False)

        # 4. Test Set (Sep 2023 - Dec 2024)
        test_data = load_test_set(pair, proc_data)
        if test_data is not None:
            test_log = run_backtest(test_data, pair, pd.Timestamp(TEST_START))
            # Filter to test period end
            if not test_log.empty:
                ts = pd.to_datetime(test_log["timestamp"])
                end_ts = pd.Timestamp(f"{TEST_END} 23:59:59")
                if ts.dt.tz is not None:
                    end_ts = end_ts.tz_localize(ts.dt.tz)
                test_log = test_log[ts <= end_ts]
            test_m = compute_metrics(test_log)
            print(f"    Test:     {test_m['trades']:>4}t  WR={test_m['wr']:>5.1f}%  PF={fmt_pf(test_m['pf']):>5}  "
                  f"PnL={test_m['pnl_pips']:>+8.1f}p  DD={test_m['max_dd_pips']:>+7.1f}p  Exp={test_m['expectancy']:>+6.2f}")
            results["test_metrics"] = test_m
            if not test_log.empty:
                test_log.to_csv(os.path.join(RESULTS_DIR, f"trades_S3_{pair}_test.csv"), index=False)
        else:
            print(f"    Test:     SKIP (no test set data)")
            results["test_metrics"] = compute_metrics(pd.DataFrame())

        # 5. 2025 Forward
        fwd_data = load_2025(pair)
        if fwd_data is not None:
            fwd_log = run_backtest(fwd_data, pair, pd.Timestamp(FWD_START))
            # Filter to 2025 end
            if not fwd_log.empty:
                ts = pd.to_datetime(fwd_log["timestamp"])
                end_ts = pd.Timestamp(f"{FWD_END} 23:59:59")
                if ts.dt.tz is not None:
                    end_ts = end_ts.tz_localize(ts.dt.tz)
                fwd_log = fwd_log[ts <= end_ts]
            fwd_m = compute_metrics(fwd_log)
            print(f"    2025:     {fwd_m['trades']:>4}t  WR={fwd_m['wr']:>5.1f}%  PF={fmt_pf(fwd_m['pf']):>5}  "
                  f"PnL={fwd_m['pnl_pips']:>+8.1f}p  DD={fwd_m['max_dd_pips']:>+7.1f}p  Exp={fwd_m['expectancy']:>+6.2f}")
            results["fwd_metrics"] = fwd_m
            if not fwd_log.empty:
                fwd_log.to_csv(os.path.join(RESULTS_DIR, f"trades_S3_{pair}_2025.csv"), index=False)
        else:
            print(f"    2025:     SKIP (no 2025 data)")
            results["fwd_metrics"] = compute_metrics(pd.DataFrame())

        # Verdict
        all_positive = all(
            results.get(k, {}).get("pf", 0) >= 1.0 and results.get(k, {}).get("trades", 0) >= 5
            for k in ["is_metrics", "oos_metrics", "test_metrics", "fwd_metrics"]
            if results.get(k, {}).get("trades", 0) > 0
        )
        results["verdict"] = "ALL PASS" if all_positive else "MIXED"
        print(f"    >>> Verdict: {results['verdict']}")

        all_results[f"S3_{pair}"] = results

    # Summary table
    print(f"\n{'='*110}")
    print("SUMMARY — S3 Across All Pairs and Periods")
    print(f"{'='*110}")
    print(f"  {'Pair':<12} {'IS PF':>6} {'IS t':>5} {'OOS PF':>7} {'OOS t':>6} "
          f"{'Test PF':>8} {'Test t':>7} {'2025 PF':>8} {'2025 t':>7} {'Verdict':>10}")
    print(f"  {'-'*90}")

    for pair in PAIRS:
        key = f"S3_{pair}"
        if key not in all_results:
            continue
        r = all_results[key]
        is_m = r.get("is_metrics", {})
        oos_m = r.get("oos_metrics", {})
        test_m = r.get("test_metrics", {})
        fwd_m = r.get("fwd_metrics", {})
        print(f"  {pair:<12} {fmt_pf(is_m.get('pf', 0)):>6} {is_m.get('trades', 0):>5} "
              f"{fmt_pf(oos_m.get('pf', 0)):>7} {oos_m.get('trades', 0):>6} "
              f"{fmt_pf(test_m.get('pf', 0)):>8} {test_m.get('trades', 0):>7} "
              f"{fmt_pf(fwd_m.get('pf', 0)):>8} {fwd_m.get('trades', 0):>7} "
              f"{r.get('verdict', '?'):>10}")

    # Save JSON
    out_path = os.path.join(RESULTS_DIR, "s3_expansion_results.json")

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
