"""
Held-Out Test Set Validation — Run 4 passing Phase 2 strategies on Sep 2023 - Dec 2024.

This data was held out during Phase 2 (IS: Jan 2021 - Dec 2022, OOS: Jan - Aug 2023).
The test set (Sep 2023 - Dec 2024) is truly unseen data.

We concatenate the tail of processed data (for indicator warmup) with the test set,
compute indicators, run backtests, and filter trades to Sep 2023 - Dec 2024.

Strategies:
  S7_Tight     — GBP_JPY H1 (Liquidity Sweep)
  S9_Filtered  — GBP_AUD H1 (London Session)
  S3           — GBP_JPY H1 (Key Level Breakout)
  S8_OB        — GBP_USD M15 + H1 HTF (Order Block Retest)
"""
import os, sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from src.indicators.technical import compute_all_indicators
from src.backtester.engine import Backtester

from src.strategies_pkg.s7_liquidity_sweep import S7_Liquidity_Sweep
from src.strategies_pkg.s9_london_session import S9_London_Session
from src.strategies_pkg.s3_key_level_breakout import S3_KeyLevel_Breakout
from src.strategies_pkg.s8_order_block import S8_Order_Block

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TEST_SET_DIR = os.path.join(BASE_DIR, "data", "test_set")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "test_set_validation")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Test period: Sep 2023 - Dec 2024
TEST_START = "2023-09-01"
TEST_END = "2024-12-31"

# Use last N rows of processed data for indicator warmup
WARMUP_ROWS = 500  # ~500 H1 bars = ~3 months, plenty for EMA(200) warmup


def _s8_tuned():
    s = S8_Order_Block()
    s.DISPLACEMENT_ATR = 2.5
    s.TP1_ATR_MULT = 2.0
    s.OB_RETEST_WINDOW = 40
    return s


CONFIGS = [
    {"name": "S7_Tight", "pair": "GBP_JPY", "tf": "H1", "htf_tf": "H1",
     "factory": lambda: S7_Liquidity_Sweep()},
    {"name": "S9_Filtered", "pair": "GBP_AUD", "tf": "H1", "htf_tf": "H1",
     "factory": lambda: S9_London_Session(pair="GBP_AUD", filtered=True)},
    {"name": "S3", "pair": "GBP_JPY", "tf": "H1", "htf_tf": "H1",
     "factory": lambda: S3_KeyLevel_Breakout()},
    {"name": "S8_OB", "pair": "GBP_USD", "tf": "M15", "htf_tf": "H1",
     "factory": _s8_tuned},
]


def load_data(pair, tf):
    """Load warmup (tail of processed) + full test set, concatenate, compute indicators."""
    proc_fp = os.path.join(PROCESSED_DIR, f"{pair}_{tf}.csv")
    test_fp = os.path.join(TEST_SET_DIR, f"{pair}_{tf}.csv")

    if not os.path.exists(proc_fp):
        print(f"  WARNING: {proc_fp} not found")
        return None
    if not os.path.exists(test_fp):
        print(f"  WARNING: {test_fp} not found")
        return None

    # Load processed data (tail for warmup)
    proc_df = pd.read_csv(proc_fp, index_col=0, parse_dates=True)
    warmup = proc_df.tail(WARMUP_ROWS)

    # Load test set data
    test_df = pd.read_csv(test_fp, index_col=0, parse_dates=True)

    # Concatenate
    combined = pd.concat([warmup, test_df])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined.sort_index(inplace=True)
    combined.index.name = "timestamp"

    print(f"    {pair}_{tf}: warmup={len(warmup)} + test={len(test_df)} = {len(combined)} bars")
    print(f"    Date range: {combined.index[0]} to {combined.index[-1]}")

    return compute_all_indicators(combined)


def run_backtest(cfg, data, htf_data):
    """Run backtester on full data, filter trades to test period."""
    if len(data) < 250:
        print(f"    Insufficient data ({len(data)} bars)")
        return pd.DataFrame()

    strategy = cfg["factory"]()
    bt = Backtester(data=data, strategy=strategy, pair=cfg["pair"],
                    starting_equity=100_000.0, htf_data=htf_data)
    bt.run()
    trade_log = bt.get_trade_log_df()

    # Filter to test period only
    if not trade_log.empty:
        ts = pd.to_datetime(trade_log["timestamp"])
        start_ts = pd.Timestamp(TEST_START)
        end_ts = pd.Timestamp(f"{TEST_END} 23:59:59")
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


def main():
    t0 = time.time()
    all_results = {}
    all_trade_logs = []

    print(f"{'='*100}")
    print("HELD-OUT TEST SET VALIDATION — 4 Passing Phase 2 Strategies")
    print(f"  Test period: {TEST_START} to {TEST_END}")
    print(f"  Warmup: last {WARMUP_ROWS} bars from processed data")
    print(f"{'='*100}")

    # Reference: Phase 2 IS/OOS results
    print(f"\n  Reference (Phase 2 results):")
    print(f"  {'Strategy':<14} {'IS PF':>6} {'IS t':>5} {'OOS PF':>7} {'OOS t':>6} {'Gen':>6}")
    print(f"  {'-'*50}")
    ref = {
        "S7_Tight":     (1.52, 44, 1.80, 14, 1.272),
        "S9_Filtered":  (1.31, 55, 2.26, 16, 1.712),
        "S3":           (1.22, 87, 1.23, 36, 1.068),
        "S8_OB":        (1.39, 45, 1.59, 19, 1.397),
    }
    for name, (is_pf, is_t, oos_pf, oos_t, gen) in ref.items():
        print(f"  {name:<14} {is_pf:>6.2f} {is_t:>5} {oos_pf:>7.2f} {oos_t:>6} {gen:>6.3f}")

    print(f"\n{'='*100}")
    print("TEST SET RESULTS (Sep 2023 - Dec 2024)")
    print(f"{'='*100}")

    data_cache = {}

    for cfg in CONFIGS:
        name = cfg["name"]
        pair = cfg["pair"]
        tf = cfg["tf"]
        htf_tf = cfg["htf_tf"]

        print(f"\n  {name} / {pair} ({tf})...")

        # Load primary data
        cache_key = f"{pair}_{tf}"
        if cache_key not in data_cache:
            data_cache[cache_key] = load_data(pair, tf)
        data = data_cache[cache_key]
        if data is None:
            continue

        # Load HTF data
        if htf_tf == tf:
            htf_data = data
        else:
            htf_key = f"{pair}_{htf_tf}"
            if htf_key not in data_cache:
                data_cache[htf_key] = load_data(pair, htf_tf)
            htf_data = data_cache[htf_key]
        if htf_data is None:
            continue

        # Run backtest
        trade_log = run_backtest(cfg, data, htf_data)
        metrics = compute_metrics(trade_log)

        pf_str = f"{metrics['pf']:.2f}" if metrics['pf'] != float('inf') else "inf"
        print(f"    Test: {metrics['trades']:>4}t  WR={metrics['wr']:>5.1f}%  "
              f"PF={pf_str:>5}  Sharpe={metrics['sharpe']:>6.2f}  "
              f"PnL={metrics['pnl_pips']:>+8.1f}p  DD={metrics['max_dd_pips']:>+7.1f}p  "
              f"Exp={metrics['expectancy']:>+6.2f}")

        # Save trade log
        if not trade_log.empty:
            trade_log.to_csv(
                os.path.join(RESULTS_DIR, f"trades_{name}_test.csv"), index=False)
            all_trade_logs.append(trade_log)

        all_results[name] = {
            "pair": pair, "timeframe": tf,
            "metrics_test": metrics,
        }

    # Portfolio aggregate
    print(f"\n{'='*100}")
    print("PORTFOLIO AGGREGATE — Test Set")
    print(f"{'='*100}")

    if all_trade_logs:
        combined = pd.concat(all_trade_logs, ignore_index=True)
        port_metrics = compute_metrics(combined)
        combined.to_csv(
            os.path.join(RESULTS_DIR, "trades_portfolio_test.csv"), index=False)
    else:
        port_metrics = compute_metrics(pd.DataFrame())

    pf_str = f"{port_metrics['pf']:.2f}" if port_metrics['pf'] != float('inf') else "inf"
    print(f"  PORTFOLIO: {port_metrics['trades']:>4}t  WR={port_metrics['wr']:>5.1f}%  "
          f"PF={pf_str:>5}  Sharpe={port_metrics['sharpe']:>6.2f}  "
          f"PnL={port_metrics['pnl_pips']:>+8.1f}p  DD={port_metrics['max_dd_pips']:>+7.1f}p  "
          f"Exp={port_metrics['expectancy']:>+6.2f}")

    all_results["_portfolio"] = {"metrics_test": port_metrics}

    # Full comparison table
    print(f"\n{'='*100}")
    print("FULL COMPARISON: IS → OOS → Test Set → 2025")
    print(f"{'='*100}")
    print(f"  {'Strategy':<14} {'IS PF':>6} {'OOS PF':>7} {'Test PF':>8} "
          f"{'Test t':>7} {'Test WR%':>9} {'Test PnL':>9} {'Verdict':>8}")
    print(f"  {'-'*75}")

    for name in ["S7_Tight", "S9_Filtered", "S3", "S8_OB"]:
        if name not in all_results:
            continue
        is_pf, _, oos_pf, _, _ = ref[name]
        m = all_results[name]["metrics_test"]
        pf_test = f"{m['pf']:.2f}" if m['pf'] != float('inf') else "inf"
        verdict = "PASS" if m["pf"] > 1.0 and m["trades"] >= 10 else "FAIL"
        if m["trades"] < 10 and m["pf"] > 1.0:
            verdict = "LOW_N"
        print(f"  {name:<14} {is_pf:>6.2f} {oos_pf:>7.2f} {pf_test:>8} "
              f"{m['trades']:>7} {m['wr']:>8.1f}% {m['pnl_pips']:>+9.1f} {verdict:>8}")

    pf_test = f"{port_metrics['pf']:.2f}" if port_metrics['pf'] != float('inf') else "inf"
    port_verdict = "PASS" if port_metrics["pf"] > 1.0 and port_metrics["trades"] >= 10 else "FAIL"
    print(f"  {'PORTFOLIO':<14} {'1.31':>6} {'1.55':>7} {pf_test:>8} "
          f"{port_metrics['trades']:>7} {port_metrics['wr']:>8.1f}% "
          f"{port_metrics['pnl_pips']:>+9.1f} {port_verdict:>8}")

    # Save JSON
    out_path = os.path.join(RESULTS_DIR, "validation_test_set.json")

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
