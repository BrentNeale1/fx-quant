"""
2025 Forward Validation — Run 4 passing Phase 2 strategies on 2025 data.

This is a true out-of-sample test on data the strategies have never seen.
Data includes Oct-Dec 2024 warmup for indicator computation; trades are
filtered to only count those from Jan 1 2025 onward.

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

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "2025")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "2025_validation")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 2025 test period (warmup from Oct 2024 is in the data files)
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"


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
    fp = os.path.join(DATA_DIR, f"{pair}_{tf}.csv")
    if not os.path.exists(fp):
        print(f"  WARNING: {fp} not found")
        return None
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    return compute_all_indicators(df)


def run_backtest(cfg, data, htf_data):
    """Run backtester on full data, filter trades to 2025."""
    if len(data) < 250:
        print(f"    Insufficient data ({len(data)} bars)")
        return pd.DataFrame()

    strategy = cfg["factory"]()
    bt = Backtester(data=data, strategy=strategy, pair=cfg["pair"],
                    starting_equity=100_000.0, htf_data=htf_data)
    bt.run()
    trade_log = bt.get_trade_log_df()

    # Filter to only 2025 trades
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
    print("2025 FORWARD VALIDATION — 4 Passing Phase 2 Strategies")
    print(f"  Test period: {TEST_START} to {TEST_END}")
    print(f"  Data dir: {DATA_DIR}")
    print(f"{'='*100}")

    # Reference: Phase 2 IS/OOS results for comparison
    print(f"\n  Reference (Phase 2 results):")
    print(f"  {'Strategy':<14} {'IS PF':>6} {'OOS PF':>7} {'Gen':>6}")
    print(f"  {'-'*40}")
    ref = {
        "S7_Tight":     (1.52, 1.80, 1.272),
        "S9_Filtered":  (1.31, 2.26, 1.712),
        "S3":           (1.22, 1.23, 1.068),
        "S8_OB":        (1.39, 1.59, 1.397),
    }
    for name, (is_pf, oos_pf, gen) in ref.items():
        print(f"  {name:<14} {is_pf:>6.2f} {oos_pf:>7.2f} {gen:>6.3f}")

    print(f"\n{'='*100}")
    print("2025 RESULTS")
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
        print(f"    2025: {metrics['trades']:>4}t  WR={metrics['wr']:>5.1f}%  "
              f"PF={pf_str:>5}  Sharpe={metrics['sharpe']:>6.2f}  "
              f"PnL={metrics['pnl_pips']:>+8.1f}p  DD={metrics['max_dd_pips']:>+7.1f}p  "
              f"Exp={metrics['expectancy']:>+6.2f}")

        # Save trade log
        if not trade_log.empty:
            trade_log.to_csv(
                os.path.join(RESULTS_DIR, f"trades_{name}_2025.csv"), index=False)
            all_trade_logs.append(trade_log)

        all_results[name] = {
            "pair": pair, "timeframe": tf,
            "metrics_2025": metrics,
        }

    # Portfolio aggregate
    print(f"\n{'='*100}")
    print("PORTFOLIO AGGREGATE — 2025")
    print(f"{'='*100}")

    if all_trade_logs:
        combined = pd.concat(all_trade_logs, ignore_index=True)
        port_metrics = compute_metrics(combined)
        combined.to_csv(
            os.path.join(RESULTS_DIR, "trades_portfolio_2025.csv"), index=False)
    else:
        port_metrics = compute_metrics(pd.DataFrame())

    pf_str = f"{port_metrics['pf']:.2f}" if port_metrics['pf'] != float('inf') else "inf"
    print(f"  PORTFOLIO: {port_metrics['trades']:>4}t  WR={port_metrics['wr']:>5.1f}%  "
          f"PF={pf_str:>5}  Sharpe={port_metrics['sharpe']:>6.2f}  "
          f"PnL={port_metrics['pnl_pips']:>+8.1f}p  DD={port_metrics['max_dd_pips']:>+7.1f}p  "
          f"Exp={port_metrics['expectancy']:>+6.2f}")

    all_results["_portfolio"] = {"metrics_2025": port_metrics}

    # Comparison table
    print(f"\n{'='*100}")
    print("COMPARISON: Phase 2 vs 2025")
    print(f"{'='*100}")
    print(f"  {'Strategy':<14} {'IS PF':>6} {'OOS PF':>7} {'2025 PF':>8} "
          f"{'2025 t':>7} {'2025 WR%':>9} {'2025 PnL':>9} {'Verdict':>8}")
    print(f"  {'-'*75}")

    for name in ["S7_Tight", "S9_Filtered", "S3", "S8_OB"]:
        if name not in all_results:
            continue
        is_pf, oos_pf, _ = ref[name]
        m = all_results[name]["metrics_2025"]
        pf_2025 = f"{m['pf']:.2f}" if m['pf'] != float('inf') else "inf"
        verdict = "PASS" if m["pf"] > 1.0 and m["trades"] >= 5 else "FAIL"
        print(f"  {name:<14} {is_pf:>6.2f} {oos_pf:>7.2f} {pf_2025:>8} "
              f"{m['trades']:>7} {m['wr']:>8.1f}% {m['pnl_pips']:>+9.1f} {verdict:>8}")

    pf_2025 = f"{port_metrics['pf']:.2f}" if port_metrics['pf'] != float('inf') else "inf"
    port_verdict = "PASS" if port_metrics["pf"] > 1.0 and port_metrics["trades"] >= 10 else "FAIL"
    print(f"  {'PORTFOLIO':<14} {'1.31':>6} {'1.55':>7} {pf_2025:>8} "
          f"{port_metrics['trades']:>7} {port_metrics['wr']:>8.1f}% "
          f"{port_metrics['pnl_pips']:>+9.1f} {port_verdict:>8}")

    # Save JSON
    out_path = os.path.join(RESULTS_DIR, "validation_2025.json")

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
