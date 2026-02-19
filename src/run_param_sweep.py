"""
Phase 2 — Parameter Grid Sweep with OOS Validation.

For each strategy, defines a small param grid (~18 combos), runs all on IS,
ranks by profit factor, validates the top-1 param set on OOS.

Output: results/phase2/param_sweep.json
"""
import os, sys, io, json, time
from itertools import product
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from src.indicators.technical import compute_all_indicators
from src.backtester.engine import Backtester

# Strategy imports
from src.strategies_pkg.s7_liquidity_sweep import S7_Liquidity_Sweep
from src.strategies_pkg.s9_london_session import S9_London_Session
from src.strategies_pkg.s4f_ema_ribbon import S4F_EMA_Ribbon
from src.strategies_pkg.s3_key_level_breakout import S3_KeyLevel_Breakout

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase2")
os.makedirs(RESULTS_DIR, exist_ok=True)

# IS/OOS period definitions
IS_START = "2021-01-01"
IS_END = "2022-12-31"
OOS_START = "2023-01-01"
OOS_END = "2023-08-31"
WARMUP_DAYS = 60

# ─── Strategy configs + param grids ───────────────────────────────────────

SWEEP_CONFIGS = [
    {
        "name": "S7_Tight",
        "pair": "GBP_JPY",
        "tf": "H1",
        "htf_tf": "H1",
        "factory": lambda: S7_Liquidity_Sweep(),
        "param_mode": "setattr",
        "grid": {
            "SWEEP_MIN_ATR": [0.5, 0.7, 1.0],
            "SL_ATR_MULT": [0.8, 1.0, 1.5],
            "TP2_ATR_MULT": [2.5, 3.0],
        },
    },
    {
        "name": "S9",
        "pair": "GBP_USD",
        "tf": "H1",
        "htf_tf": "H1",
        "factory": lambda: S9_London_Session(),
        "param_mode": "setattr",
        "grid": {
            "VOLUME_MULT": [1.2, 1.5, 2.0],
            "SL_ATR_CAP": [2.0, 2.5, 3.0],
            "ENTRY_END_HOUR": [10, 11],
        },
    },
    {
        "name": "S9_Filtered",
        "pair": "GBP_AUD",
        "tf": "H1",
        "htf_tf": "H1",
        "factory": lambda: S9_London_Session(pair="GBP_AUD", filtered=True),
        "param_mode": "pair_cfg",
        "grid": {
            "min_adx": [20, 25, 30],
            "min_ema50_dist_pips": [30, 40, 50],
            "skip_friday": [True, False],
        },
    },
    {
        "name": "S4F",
        "pair": "EUR_AUD",
        "tf": "M15",
        "htf_tf": "H1",
        "factory": lambda: S4F_EMA_Ribbon(),
        "param_mode": "setattr",
        "grid": {
            "SL_ATR_MULT": [1.5, 2.0, 2.5],
            "TP_ATR_MULT": [2.5, 3.0, 4.0],
            "COMPRESSION_ATR_MULT": [0.8, 1.0],
        },
    },
    {
        "name": "S3",
        "pair": "GBP_JPY",
        "tf": "H1",
        "htf_tf": "H1",
        "factory": lambda: S3_KeyLevel_Breakout(),
        "param_mode": "setattr",
        "grid": {
            "SL_ATR_MULT": [0.3, 0.5, 0.75],
            "TP1_ATR_MULT": [1.0, 1.5, 2.0],
            "KEY_LEVEL_TOLERANCE": [0.5, 0.75],
        },
    },
]


# ─── Helpers ──────────────────────────────────────────────────────────────

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
    """Slice dataframe to a date range, with warmup prepended."""
    if df.index.tz is not None:
        start_ts = pd.Timestamp(start, tz=df.index.tz)
        end_ts = pd.Timestamp(f"{end} 23:59:59", tz=df.index.tz)
    else:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(f"{end} 23:59:59")

    warmup_start = start_ts - pd.DateOffset(days=warmup_days)
    sliced = df[(df.index >= warmup_start) & (df.index <= end_ts)].copy()
    return sliced, start_ts


def apply_params(strategy, params, mode):
    """Apply parameter dict to a strategy instance."""
    if mode == "setattr":
        for key, value in params.items():
            setattr(strategy, key, value)
    elif mode == "pair_cfg":
        for key, value in params.items():
            strategy._pair_cfg[key] = value


def run_single(cfg, data, htf_data, start, end, params):
    """Run a single backtest with given params, return metrics."""
    sliced, start_ts = slice_period(data, start, end)
    if len(sliced) < 250:
        return {"trades": 0, "wr": 0, "pf": 0, "sharpe": 0,
                "pnl_pips": 0, "expectancy": 0}

    htf_sliced = sliced.copy() if cfg["htf_tf"] == cfg["tf"] else \
        slice_period(htf_data, start, end)[0]

    strategy = cfg["factory"]()
    apply_params(strategy, params, cfg["param_mode"])

    bt = Backtester(data=sliced, strategy=strategy, pair=cfg["pair"],
                    starting_equity=100_000.0, htf_data=htf_sliced)
    bt.run()
    trade_log = bt.get_trade_log_df()

    # Filter warmup trades
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
    """Compute metrics from trade log."""
    if trade_log.empty or len(trade_log) == 0:
        return {"trades": 0, "wr": 0, "pf": 0, "sharpe": 0,
                "pnl_pips": 0, "expectancy": 0}

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

    return {
        "trades": n,
        "wr": round(wr, 1),
        "pf": round(pf, 2),
        "sharpe": round(sharpe, 2),
        "pnl_pips": round(total_pnl, 1),
        "expectancy": round(expectancy, 2),
    }


def compute_generalization_score(is_m, oos_m):
    """Quick generalization composite from IS/OOS metrics."""
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

    for k in ratios:
        ratios[k] = max(ratios[k], 0.0)

    return round(np.mean(list(ratios.values())), 3) if ratios else 0.0


def expand_grid(grid):
    """Expand a param grid dict into a list of param dicts."""
    keys = list(grid.keys())
    values = list(grid.values())
    combos = []
    for combo in product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    all_sweep_results = {}

    print(f"{'='*100}")
    print("PHASE 2 — PARAMETER SWEEP (IS Grid Search + OOS Validation)")
    print(f"  IS period:  {IS_START} to {IS_END}")
    print(f"  OOS period: {OOS_START} to {OOS_END}")
    print(f"{'='*100}")

    # Data cache to avoid reloading
    data_cache = {}

    for cfg in SWEEP_CONFIGS:
        name = cfg["name"]
        pair = cfg["pair"]
        tf = cfg["tf"]
        htf_tf = cfg["htf_tf"]

        print(f"\n{'#'*70}")
        print(f"# {name} / {pair} ({tf})")
        print(f"{'#'*70}")

        # Load data (cached)
        cache_key = f"{pair}_{tf}"
        if cache_key not in data_cache:
            data_cache[cache_key] = load_data(pair, tf)
        data = data_cache[cache_key]
        if data is None:
            continue

        htf_cache_key = f"{pair}_{htf_tf}"
        if htf_cache_key not in data_cache:
            data_cache[htf_cache_key] = load_data(pair, htf_tf)
        htf_data = data_cache[htf_cache_key] if htf_tf != tf else data
        if htf_data is None:
            continue

        # Expand parameter grid
        combos = expand_grid(cfg["grid"])
        print(f"  Grid: {len(combos)} combinations")

        # Run IS sweep
        is_results = []
        for i, params in enumerate(combos, 1):
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            metrics = run_single(cfg, data, htf_data, IS_START, IS_END, params)
            is_results.append({
                "params": params,
                "metrics": metrics,
            })
            # Progress indicator
            status = f"  [{i:>2}/{len(combos)}] {param_str}"
            status += f" -> {metrics['trades']}t PF={metrics['pf']:.2f} WR={metrics['wr']:.1f}%"
            print(status)

        # Rank by profit factor (filter out 0-trade combos)
        valid = [r for r in is_results if r["metrics"]["trades"] >= 5]
        if not valid:
            print("  No valid IS results (all combos had <5 trades)")
            all_sweep_results[name] = {
                "pair": pair, "combos_tested": len(combos),
                "best_params": None, "is_results_ranked": [],
            }
            continue

        # Sort by PF descending, then by trade count descending as tiebreak
        valid.sort(key=lambda r: (r["metrics"]["pf"], r["metrics"]["trades"]),
                   reverse=True)

        print(f"\n  --- IS Rankings (top 5) ---")
        print(f"  {'Rank':>4}  {'PF':>6} {'WR%':>6} {'Trades':>6} {'Sharpe':>7} {'Exp':>7} Params")
        for rank, r in enumerate(valid[:5], 1):
            m = r["metrics"]
            p_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
            print(f"  {rank:>4}  {m['pf']:>6.2f} {m['wr']:>5.1f}% {m['trades']:>6} "
                  f"{m['sharpe']:>7.2f} {m['expectancy']:>+7.2f}  {p_str}")

        # Validate top-1 on OOS
        best = valid[0]
        best_params = best["params"]
        is_metrics = best["metrics"]

        print(f"\n  Validating best params on OOS...")
        oos_metrics = run_single(cfg, data, htf_data, OOS_START, OOS_END, best_params)

        gen_score = compute_generalization_score(is_metrics, oos_metrics)
        if gen_score >= 0.80:
            verdict = "PASS"
        elif gen_score >= 0.50:
            verdict = "WARN"
        else:
            verdict = "FAIL"

        param_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
        print(f"\n  BEST: {param_str}")
        print(f"    IS:  {is_metrics['trades']}t PF={is_metrics['pf']:.2f} "
              f"WR={is_metrics['wr']:.1f}% Sharpe={is_metrics['sharpe']:.2f} "
              f"Exp={is_metrics['expectancy']:+.2f}")
        print(f"    OOS: {oos_metrics['trades']}t PF={oos_metrics['pf']:.2f} "
              f"WR={oos_metrics['wr']:.1f}% Sharpe={oos_metrics['sharpe']:.2f} "
              f"Exp={oos_metrics['expectancy']:+.2f}")
        print(f"    Gen: {gen_score:.3f} -> {verdict}")

        # Store
        all_sweep_results[name] = {
            "pair": pair,
            "combos_tested": len(combos),
            "best_params": best_params,
            "is_best_metrics": is_metrics,
            "oos_validation": oos_metrics,
            "generalization_score": gen_score,
            "verdict": verdict,
            "is_results_ranked": [
                {"rank": i + 1, "params": r["params"], "metrics": r["metrics"]}
                for i, r in enumerate(valid)
            ],
        }

    # Summary table
    print(f"\n{'='*100}")
    print("SWEEP SUMMARY")
    print(f"{'='*100}")
    print(f"  {'Strategy':<14} {'Combos':>6} {'Best IS PF':>10} {'OOS PF':>8} "
          f"{'Gen':>6} {'Verdict':>8}  Best Params")
    print(f"  {'-'*95}")

    for name, res in all_sweep_results.items():
        if res["best_params"] is None:
            print(f"  {name:<14} {res['combos_tested']:>6} {'N/A':>10} {'N/A':>8} "
                  f"{'N/A':>6} {'SKIP':>8}")
            continue
        is_pf = res["is_best_metrics"]["pf"]
        oos_pf = res["oos_validation"]["pf"]
        gen = res["generalization_score"]
        verdict = res["verdict"]
        p_str = ", ".join(f"{k}={v}" for k, v in res["best_params"].items())
        print(f"  {name:<14} {res['combos_tested']:>6} {is_pf:>10.2f} {oos_pf:>8.2f} "
              f"{gen:>6.3f} {verdict:>8}  {p_str}")

    # Save JSON
    out_path = os.path.join(RESULTS_DIR, "param_sweep.json")

    def json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return str(obj)

    with open(out_path, "w") as f:
        json.dump(all_sweep_results, f, indent=2, default=json_default)
    print(f"\nResults saved: {out_path}")

    elapsed = time.time() - t0
    print(f"Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
