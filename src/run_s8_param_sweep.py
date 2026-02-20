"""
Parameter Sweep — S8 Order Block Retest on GBP_USD (M15 + H1 HTF).

Sweeps key tunable parameters on IS, validates top-N on OOS.

Grid (27 combos):
  DISPLACEMENT_ATR: [1.0, 1.5, 2.0]   — entry filter strictness
  TP1_ATR_MULT:     [1.0, 1.5, 2.0]   — first profit target
  OB_RETEST_WINDOW: [20, 30, 40]      — OB freshness window (bars)
"""
import os, sys, io, json, time
from itertools import product
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from src.indicators.technical import compute_all_indicators
from src.backtester.engine import Backtester
from src.strategies_pkg.s8_order_block import S8_Order_Block

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase2")
os.makedirs(RESULTS_DIR, exist_ok=True)

IS_START = "2021-01-01"
IS_END = "2022-12-31"
OOS_START = "2023-01-01"
OOS_END = "2023-08-31"
WARMUP_DAYS = 60

PAIR = "GBP_USD"
TF = "M15"
HTF_TF = "H1"

PARAM_GRID = {
    "DISPLACEMENT_ATR": [1.0, 1.5, 2.0],
    "TP1_ATR_MULT": [1.0, 1.5, 2.0],
    "OB_RETEST_WINDOW": [20, 30, 40],
}

# Number of top IS results to validate on OOS
TOP_N = 5


def load_data(pair, tf):
    fp = os.path.join(PROCESSED_DIR, f"{pair}_{tf}.csv")
    if not os.path.exists(fp):
        print(f"  WARNING: {fp} not found")
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


def run_single(data, htf_data, start, end, params):
    sliced, start_ts = slice_period(data, start, end)
    if len(sliced) < 250:
        return {"trades": 0, "wr": 0, "pf": 0, "sharpe": 0,
                "pnl_pips": 0, "max_dd_pips": 0, "expectancy": 0}

    htf_sliced = slice_period(htf_data, start, end)[0]

    strategy = S8_Order_Block()
    for key, value in params.items():
        setattr(strategy, key, value)

    bt = Backtester(data=sliced, strategy=strategy, pair=PAIR,
                    starting_equity=100_000.0, htf_data=htf_sliced)
    bt.run()
    trade_log = bt.get_trade_log_df()

    if not trade_log.empty:
        ts = pd.to_datetime(trade_log["timestamp"])
        filter_ts = pd.Timestamp(start_ts)
        if ts.dt.tz is not None and filter_ts.tz is None:
            filter_ts = filter_ts.tz_localize(ts.dt.tz)
        elif ts.dt.tz is None and filter_ts.tz is not None:
            filter_ts = filter_ts.tz_localize(None)
        trade_log = trade_log[ts >= filter_ts]

    return compute_metrics(trade_log)


def compute_metrics(trade_log):
    if trade_log.empty or len(trade_log) == 0:
        return {"trades": 0, "wr": 0, "pf": 0, "sharpe": 0,
                "pnl_pips": 0, "max_dd_pips": 0, "expectancy": 0}

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


def compute_generalization_score(is_m, oos_m):
    if is_m["trades"] == 0 or oos_m["trades"] == 0:
        return 0.0
    ratios = {}
    if is_m["wr"] > 0:
        ratios["wr"] = min(oos_m["wr"] / is_m["wr"], 2.0)
    if is_m["pf"] > 0 and is_m["pf"] != float("inf"):
        oos_pf = min(oos_m["pf"], 2 * is_m["pf"]) if oos_m["pf"] == float("inf") else oos_m["pf"]
        ratios["pf"] = min(oos_pf / is_m["pf"], 2.0)
    if is_m["expectancy"] > 0:
        ratios["expectancy"] = min(oos_m["expectancy"] / is_m["expectancy"], 2.0)
    if is_m["sharpe"] > 0:
        ratios["sharpe"] = min(oos_m["sharpe"] / is_m["sharpe"], 2.0)
    for k in ratios:
        ratios[k] = max(ratios[k], 0.0)
    return round(np.mean(list(ratios.values())), 3) if ratios else 0.0


def expand_grid(grid):
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def main():
    t0 = time.time()

    print(f"{'='*100}")
    print(f"PARAMETER SWEEP — S8 Order Block Retest / {PAIR} ({TF} + {HTF_TF})")
    print(f"  IS period:  {IS_START} to {IS_END}")
    print(f"  OOS period: {OOS_START} to {OOS_END}")
    print(f"{'='*100}")

    # Load data
    print(f"\n  Loading {PAIR} {TF}...")
    data = load_data(PAIR, TF)
    if data is None:
        return
    print(f"  Loading {PAIR} {HTF_TF}...")
    htf_data = load_data(PAIR, HTF_TF)
    if htf_data is None:
        return

    combos = expand_grid(PARAM_GRID)
    print(f"  Grid: {len(combos)} combinations\n")

    # IS sweep
    is_results = []
    for i, params in enumerate(combos, 1):
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        metrics = run_single(data, htf_data, IS_START, IS_END, params)
        is_results.append({"params": params, "metrics": metrics})
        status = f"  [{i:>2}/{len(combos)}] {param_str}"
        status += f" -> {metrics['trades']}t PF={metrics['pf']:.2f} WR={metrics['wr']:.1f}% Exp={metrics['expectancy']:+.2f}"
        print(status)

    # Rank by PF (filter out < 5 trades)
    valid = [r for r in is_results if r["metrics"]["trades"] >= 5]
    if not valid:
        print("\n  No valid IS results (all combos had <5 trades)")
        return

    valid.sort(key=lambda r: (r["metrics"]["pf"], r["metrics"]["trades"]), reverse=True)

    print(f"\n  --- IS Rankings (all valid) ---")
    print(f"  {'Rank':>4}  {'PF':>6} {'WR%':>6} {'Trades':>6} {'Sharpe':>7} {'Exp':>7} {'PnL':>9} {'DD':>8} Params")
    for rank, r in enumerate(valid, 1):
        m = r["metrics"]
        p_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        print(f"  {rank:>4}  {m['pf']:>6.2f} {m['wr']:>5.1f}% {m['trades']:>6} "
              f"{m['sharpe']:>7.2f} {m['expectancy']:>+7.2f} {m['pnl_pips']:>+9.1f} "
              f"{m['max_dd_pips']:>+8.1f}  {p_str}")

    # Validate top-N on OOS
    print(f"\n{'='*100}")
    print(f"OOS VALIDATION (Top {TOP_N} IS param sets)")
    print(f"{'='*100}")
    print(f"  {'Rank':>4}  {'IS PF':>6} {'OOS PF':>7} {'OOS WR':>7} {'OOS Trades':>10} "
          f"{'OOS Exp':>8} {'OOS PnL':>9} {'OOS DD':>8} {'Gen':>6} {'Verdict':>8} Params")

    oos_results = []
    for rank, r in enumerate(valid[:TOP_N], 1):
        params = r["params"]
        is_m = r["metrics"]
        oos_m = run_single(data, htf_data, OOS_START, OOS_END, params)
        gen = compute_generalization_score(is_m, oos_m)

        if gen >= 0.80:
            verdict = "PASS"
        elif gen >= 0.50:
            verdict = "WARN"
        else:
            verdict = "FAIL"

        p_str = ", ".join(f"{k}={v}" for k, v in params.items())
        pf_str = f"{oos_m['pf']:.2f}" if oos_m['pf'] != float('inf') else "inf"
        print(f"  {rank:>4}  {is_m['pf']:>6.2f} {pf_str:>7} {oos_m['wr']:>6.1f}% "
              f"{oos_m['trades']:>10} {oos_m['expectancy']:>+8.2f} "
              f"{oos_m['pnl_pips']:>+9.1f} {oos_m['max_dd_pips']:>+8.1f} "
              f"{gen:>5.3f} {verdict:>8}  {p_str}")

        oos_results.append({
            "rank": rank,
            "params": params,
            "is_metrics": is_m,
            "oos_metrics": oos_m,
            "generalization_score": gen,
            "verdict": verdict,
        })

    # Save results
    out_path = os.path.join(RESULTS_DIR, "s8_param_sweep.json")

    def json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return str(obj)

    results = {
        "pair": PAIR,
        "timeframe": TF,
        "htf": HTF_TF,
        "combos_tested": len(combos),
        "is_results_ranked": [
            {"rank": i + 1, "params": r["params"], "metrics": r["metrics"]}
            for i, r in enumerate(valid)
        ],
        "oos_validation": oos_results,
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=json_default)
    print(f"\nResults saved: {out_path}")

    elapsed = time.time() - t0
    print(f"Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
