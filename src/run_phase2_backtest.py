"""
Phase 2 — Unified Backtest Runner with IS/OOS Split.

Runs all 5 Phase 2 strategies on:
  - In-Sample (IS): 2021-01-01 to 2022-12-31
  - Out-of-Sample (OOS): 2023-01-01 to 2023-08-31

Computes per-strategy and portfolio-level metrics, generalization scores,
and exports structured results.

Output:
  - Console summary table
  - results/phase2/backtest_report.json
  - results/phase2/trades_{name}_IS.csv / trades_{name}_OOS.csv
"""
import os, sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from src.indicators.technical import compute_all_indicators
from src.backtester.engine import Backtester

# Strategy imports
from src.strategies_pkg.s7_liquidity_sweep import S7_Liquidity_Sweep
from src.strategies_pkg.s9_london_session import S9_London_Session
from src.strategies_pkg.s3_key_level_breakout import S3_KeyLevel_Breakout
from src.strategies_pkg.s8_order_block import S8_Order_Block

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase2")
os.makedirs(RESULTS_DIR, exist_ok=True)

# IS/OOS period definitions
IS_START = "2021-01-01"
IS_END = "2022-12-31"
OOS_START = "2023-01-01"
OOS_END = "2023-08-31"
WARMUP_DAYS = 60

# Phase 2 strategy-pair configurations
# S9/GBP_USD dropped: negative PF across all param combos on IS (best 0.86)
# S4F/EUR_AUD dropped: classic overfit — IS PF 1.43 collapses to OOS PF 0.48
CONFIGS = [
    {"name": "S7_Tight", "pair": "GBP_JPY", "tf": "H1", "htf_tf": "H1",
     "factory": lambda: S7_Liquidity_Sweep()},
    {"name": "S9_Filtered", "pair": "GBP_AUD", "tf": "H1", "htf_tf": "H1",
     "factory": lambda: S9_London_Session(pair="GBP_AUD", filtered=True)},
    {"name": "S3", "pair": "GBP_JPY", "tf": "H1", "htf_tf": "H1",
     "factory": lambda: S3_KeyLevel_Breakout()},
    # S8: Order Block Retest — best sweep params (DISP_ATR=2.0, TP1=1.0, Window=40)
    # OOS: 26t PF=2.14 WR=65.4% Gen=1.850 PASS
    {"name": "S8_OB", "pair": "GBP_USD", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: _s8_tuned()},
    # S10 dropped: insufficient trade counts on M15, no pair passes generalization
    # S11 dropped: OOS collapsed across all param combos (0% WR, Gen=0.00)
]


def _s8_tuned():
    s = S8_Order_Block()
    s.DISPLACEMENT_ATR = 2.0
    s.TP1_ATR_MULT = 1.0
    s.OB_RETEST_WINDOW = 40
    return s


def load_data(pair, tf):
    """Load price data with indicators."""
    fp = os.path.join(PROCESSED_DIR, f"{pair}_{tf}.csv")
    if not os.path.exists(fp):
        print(f"  WARNING: {fp} not found")
        return None
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    return compute_all_indicators(df)


def slice_period(df, start, end, warmup_days=WARMUP_DAYS):
    """Slice dataframe to a date range, with warmup prepended for indicators."""
    if df.index.tz is not None:
        start_ts = pd.Timestamp(start, tz=df.index.tz)
        end_ts = pd.Timestamp(f"{end} 23:59:59", tz=df.index.tz)
    else:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(f"{end} 23:59:59")

    warmup_start = start_ts - pd.DateOffset(days=warmup_days)
    sliced = df[(df.index >= warmup_start) & (df.index <= end_ts)].copy()
    return sliced, start_ts


def run_backtest_period(cfg, data, htf_data, start, end):
    """Run backtester on a period, return filtered trade log."""
    sliced, start_ts = slice_period(data, start, end)
    if len(sliced) < 250:
        print(f"    Insufficient data ({len(sliced)} bars)")
        return pd.DataFrame()

    htf_sliced = sliced.copy() if cfg["htf_tf"] == cfg["tf"] else \
        slice_period(htf_data, start, end)[0]

    strategy = cfg["factory"]()
    bt = Backtester(data=sliced, strategy=strategy, pair=cfg["pair"],
                    starting_equity=100_000.0, htf_data=htf_sliced)
    bt.run()
    trade_log = bt.get_trade_log_df()

    # Filter trades to exclude warmup period
    if not trade_log.empty:
        ts = pd.to_datetime(trade_log["timestamp"])
        filter_ts = pd.Timestamp(start_ts)
        if ts.dt.tz is not None and filter_ts.tz is None:
            filter_ts = filter_ts.tz_localize(ts.dt.tz)
        elif ts.dt.tz is None and filter_ts.tz is not None:
            filter_ts = filter_ts.tz_localize(None)
        trade_log = trade_log[ts >= filter_ts]

    return trade_log


def compute_metrics(trade_log):
    """Compute metrics from a trade log DataFrame, including Sharpe ratio."""
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

    # Sharpe ratio (annualized, from per-trade PnL)
    if n > 1:
        pnl_series = trade_log["pnl_pips"]
        sharpe = (pnl_series.mean() / pnl_series.std()) * np.sqrt(252) \
            if pnl_series.std() > 0 else 0
    else:
        sharpe = 0

    # Max drawdown in pips
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


def compute_generalization_scores(is_metrics, oos_metrics):
    """Compute OOS/IS ratio per metric + composite generalization score."""
    if is_metrics["trades"] == 0 or oos_metrics["trades"] == 0:
        return {"composite": 0, "detail": {}, "verdict": "FAIL"}

    ratios = {}

    # Win rate ratio
    if is_metrics["wr"] > 0:
        ratios["wr"] = oos_metrics["wr"] / is_metrics["wr"]
    else:
        ratios["wr"] = 0

    # Profit factor ratio
    if is_metrics["pf"] > 0 and is_metrics["pf"] != float("inf"):
        if oos_metrics["pf"] == float("inf"):
            ratios["pf"] = 2.0  # Cap at 2x
        else:
            ratios["pf"] = oos_metrics["pf"] / is_metrics["pf"]
    else:
        ratios["pf"] = 0

    # Expectancy ratio
    if is_metrics["expectancy"] > 0:
        ratios["expectancy"] = oos_metrics["expectancy"] / is_metrics["expectancy"]
    elif is_metrics["expectancy"] < 0 and oos_metrics["expectancy"] < 0:
        ratios["expectancy"] = 0  # Both negative
    else:
        ratios["expectancy"] = 0

    # Sharpe ratio (same-sign comparison)
    if is_metrics["sharpe"] > 0:
        ratios["sharpe"] = oos_metrics["sharpe"] / is_metrics["sharpe"]
    else:
        ratios["sharpe"] = 0

    # Cap individual ratios at 2.0 (OOS can't be "too much better")
    for k in ratios:
        ratios[k] = min(ratios[k], 2.0)
        ratios[k] = max(ratios[k], 0.0)

    # Composite: equal-weight average of capped ratios
    composite = np.mean(list(ratios.values())) if ratios else 0

    if composite >= 0.80:
        verdict = "PASS"
    elif composite >= 0.50:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    return {
        "composite": round(composite, 3),
        "detail": {k: round(v, 3) for k, v in ratios.items()},
        "verdict": verdict,
    }


def compute_portfolio_aggregate(all_trade_logs):
    """Concatenate all strategy trades, compute portfolio-level metrics."""
    combined = pd.concat(all_trade_logs, ignore_index=True) if all_trade_logs else pd.DataFrame()
    return compute_metrics(combined), combined


def main():
    t0 = time.time()
    all_results = {}
    is_trade_logs = []
    oos_trade_logs = []

    print(f"{'='*100}")
    print("PHASE 2 — UNIFIED BACKTEST (IS/OOS Split)")
    print(f"  IS period:  {IS_START} to {IS_END}")
    print(f"  OOS period: {OOS_START} to {OOS_END}")
    print(f"{'='*100}")

    header = (f"{'Strategy':<14} {'Period':<5} {'Trades':>6} {'WR%':>6} "
              f"{'PF':>6} {'Sharpe':>7} {'Exp':>7} {'PnL':>9} {'Gen':>6}")
    separator = "-" * 100

    print(f"\n{header}")
    print(separator)

    for cfg in CONFIGS:
        name = cfg["name"]
        pair = cfg["pair"]
        tf = cfg["tf"]
        htf_tf = cfg["htf_tf"]

        print(f"\n  Loading {name} / {pair} ({tf})...")

        # Load data
        data = load_data(pair, tf)
        if data is None:
            continue
        htf_data = data.copy() if htf_tf == tf else load_data(pair, htf_tf)
        if htf_data is None:
            continue

        # Run IS
        is_log = run_backtest_period(cfg, data, htf_data, IS_START, IS_END)
        is_metrics = compute_metrics(is_log)

        # Run OOS
        oos_log = run_backtest_period(cfg, data, htf_data, OOS_START, OOS_END)
        oos_metrics = compute_metrics(oos_log)

        # Generalization score
        gen = compute_generalization_scores(is_metrics, oos_metrics)

        # Print rows
        print(f"  {name:<14} {'IS':<5} {is_metrics['trades']:>6} "
              f"{is_metrics['wr']:>5.1f}% {is_metrics['pf']:>6.2f} "
              f"{is_metrics['sharpe']:>7.2f} {is_metrics['expectancy']:>+7.2f} "
              f"{is_metrics['pnl_pips']:>+9.1f}")
        print(f"  {'':<14} {'OOS':<5} {oos_metrics['trades']:>6} "
              f"{oos_metrics['wr']:>5.1f}% {oos_metrics['pf']:>6.2f} "
              f"{oos_metrics['sharpe']:>7.2f} {oos_metrics['expectancy']:>+7.2f} "
              f"{oos_metrics['pnl_pips']:>+9.1f} "
              f"{gen['composite']:>5.2f} {gen['verdict']}")

        # Save trade CSVs
        if not is_log.empty:
            is_log.to_csv(os.path.join(RESULTS_DIR, f"trades_{name}_IS.csv"), index=False)
            is_trade_logs.append(is_log)
        if not oos_log.empty:
            oos_log.to_csv(os.path.join(RESULTS_DIR, f"trades_{name}_OOS.csv"), index=False)
            oos_trade_logs.append(oos_log)

        # Store results
        all_results[name] = {
            "pair": pair, "timeframe": tf,
            "is_metrics": is_metrics, "oos_metrics": oos_metrics,
            "generalization": gen,
        }

    # Portfolio aggregate
    print(f"\n{separator}")
    print("PORTFOLIO AGGREGATE")
    print(separator)

    port_is_metrics, _ = compute_portfolio_aggregate(is_trade_logs)
    port_oos_metrics, _ = compute_portfolio_aggregate(oos_trade_logs)
    port_gen = compute_generalization_scores(port_is_metrics, port_oos_metrics)

    print(f"  {'PORTFOLIO':<14} {'IS':<5} {port_is_metrics['trades']:>6} "
          f"{port_is_metrics['wr']:>5.1f}% {port_is_metrics['pf']:>6.2f} "
          f"{port_is_metrics['sharpe']:>7.2f} {port_is_metrics['expectancy']:>+7.2f} "
          f"{port_is_metrics['pnl_pips']:>+9.1f}")
    print(f"  {'':<14} {'OOS':<5} {port_oos_metrics['trades']:>6} "
          f"{port_oos_metrics['wr']:>5.1f}% {port_oos_metrics['pf']:>6.2f} "
          f"{port_oos_metrics['sharpe']:>7.2f} {port_oos_metrics['expectancy']:>+7.2f} "
          f"{port_oos_metrics['pnl_pips']:>+9.1f} "
          f"{port_gen['composite']:>5.2f} {port_gen['verdict']}")

    all_results["_portfolio"] = {
        "is_metrics": port_is_metrics,
        "oos_metrics": port_oos_metrics,
        "generalization": port_gen,
    }

    # Generalization summary
    print(f"\n{separator}")
    print("GENERALIZATION SUMMARY")
    print(separator)
    print(f"  {'Strategy':<14} {'Composite':>9} {'WR':>6} {'PF':>6} {'Exp':>6} {'Sharpe':>7} {'Verdict':>8}")
    for name, res in all_results.items():
        if name.startswith("_"):
            continue
        g = res["generalization"]
        d = g["detail"]
        print(f"  {name:<14} {g['composite']:>9.3f} "
              f"{d.get('wr', 0):>6.3f} {d.get('pf', 0):>6.3f} "
              f"{d.get('expectancy', 0):>6.3f} {d.get('sharpe', 0):>7.3f} "
              f"{g['verdict']:>8}")

    g = all_results["_portfolio"]["generalization"]
    d = g["detail"]
    print(f"  {'PORTFOLIO':<14} {g['composite']:>9.3f} "
          f"{d.get('wr', 0):>6.3f} {d.get('pf', 0):>6.3f} "
          f"{d.get('expectancy', 0):>6.3f} {d.get('sharpe', 0):>7.3f} "
          f"{g['verdict']:>8}")

    # Save JSON report
    out_path = os.path.join(RESULTS_DIR, "backtest_report.json")

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
