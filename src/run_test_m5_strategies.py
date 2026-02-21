"""
Test 5-Minute Strategies on M5 Data — IS/OOS Backtest.

Strategies originally designed for 5-minute charts, now tested on actual M5 data:
  S12 - Asian Range Sweep (M5 + H1 HTF)
  S15 - Momentum Continuation (M5 + H1 HTF)
  S16 - London ORB (M5 + H1 HTF)

Parameters are scaled from M15 defaults to M5 granularity (3x more bars per hour).
"""
import os, sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from src.indicators.technical import compute_all_indicators
from src.backtester.engine import Backtester

# Strategy imports
from src.strategies_pkg.s12_asian_range_sweep import S12_AsianRangeSweep
from src.strategies_pkg.s15_momentum_continuation import S15_MomentumContinuation
from src.strategies_pkg.s16_london_orb import S16_LondonORB

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase2")
os.makedirs(RESULTS_DIR, exist_ok=True)

# IS/OOS period definitions (same as Phase 2)
IS_START = "2021-01-01"
IS_END = "2022-12-31"
OOS_START = "2023-01-01"
OOS_END = "2023-08-31"
WARMUP_DAYS = 60


def make_s12_m5():
    """S12 Asian Range Sweep adapted for M5."""
    s = S12_AsianRangeSweep()
    # M5 has 84 bars in Asian session (7h * 12 bars/h) vs 28 on M15
    s.LOOKBACK_BARS = 180      # Enough to find all Asian bars
    s.MAX_BARS = 120           # Same ~10h hold time (120 * 5min = 10h)
    return s


def make_s15_m5():
    """S15 Momentum Continuation adapted for M5."""
    s = S15_MomentumContinuation()
    # Scale bar counts by 3x (M5 has 3x more bars per hour than M15)
    s.IMPULSE_BARS = 24        # 2 hours = 24 M5 bars (was 8 on M15)
    s.PULLBACK_MIN_BARS = 6    # 30 min minimum pullback (was 2)
    s.PULLBACK_MAX_BARS = 36   # 3 hours max pullback (was 12)
    s.MAX_BARS = 120           # ~10h hold (was 40)
    return s


def make_s16_m5():
    """S16 London ORB adapted for M5."""
    s = S16_LondonORB()
    # M5 gives proper 30-min opening range (6 bars)
    s.ORB_BARS = 6             # 30 min = 6 M5 bars (was 1 on M15)
    s.VOLUME_MULT = 1.3        # Restore original volume filter (M5 volume is granular)
    s.MAX_BARS = 120           # ~10h hold (was 40)
    s.MIN_RANGE_ATR = 0.2      # Slightly tighter for M5
    s.MAX_RANGE_ATR = 1.8
    return s


# Test configs: 3 strategies x 4 pairs = 12 combos
CONFIGS = [
    # S12: Asian Range Sweep on M5
    {"name": "S12_GBP_JPY_M5", "pair": "GBP_JPY", "tf": "M5", "htf_tf": "H1",
     "factory": make_s12_m5},
    {"name": "S12_GBP_USD_M5", "pair": "GBP_USD", "tf": "M5", "htf_tf": "H1",
     "factory": make_s12_m5},
    {"name": "S12_EUR_USD_M5", "pair": "EUR_USD", "tf": "M5", "htf_tf": "H1",
     "factory": make_s12_m5},
    {"name": "S12_USD_JPY_M5", "pair": "USD_JPY", "tf": "M5", "htf_tf": "H1",
     "factory": make_s12_m5},

    # S15: Momentum Continuation on M5
    {"name": "S15_GBP_JPY_M5", "pair": "GBP_JPY", "tf": "M5", "htf_tf": "H1",
     "factory": make_s15_m5},
    {"name": "S15_GBP_USD_M5", "pair": "GBP_USD", "tf": "M5", "htf_tf": "H1",
     "factory": make_s15_m5},
    {"name": "S15_EUR_USD_M5", "pair": "EUR_USD", "tf": "M5", "htf_tf": "H1",
     "factory": make_s15_m5},
    {"name": "S15_USD_JPY_M5", "pair": "USD_JPY", "tf": "M5", "htf_tf": "H1",
     "factory": make_s15_m5},

    # S16: London ORB on M5
    {"name": "S16_GBP_JPY_M5", "pair": "GBP_JPY", "tf": "M5", "htf_tf": "H1",
     "factory": make_s16_m5},
    {"name": "S16_GBP_USD_M5", "pair": "GBP_USD", "tf": "M5", "htf_tf": "H1",
     "factory": make_s16_m5},
    {"name": "S16_EUR_USD_M5", "pair": "EUR_USD", "tf": "M5", "htf_tf": "H1",
     "factory": make_s16_m5},
    {"name": "S16_USD_JPY_M5", "pair": "USD_JPY", "tf": "M5", "htf_tf": "H1",
     "factory": make_s16_m5},
]


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


def run_backtest_period(cfg, data, htf_data, start, end):
    """Run backtester on a period, return filtered trade log."""
    sliced, start_ts = slice_period(data, start, end)
    if len(sliced) < 250:
        print(f"    Insufficient data ({len(sliced)} bars)")
        return pd.DataFrame()

    htf_sliced = slice_period(htf_data, start, end)[0] if htf_data is not None else None

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
    """Compute metrics from a trade log DataFrame."""
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


def compute_generalization_scores(is_metrics, oos_metrics):
    """Compute OOS/IS ratio per metric + composite generalization score."""
    if is_metrics["trades"] == 0 or oos_metrics["trades"] == 0:
        return {"composite": 0, "detail": {}, "verdict": "FAIL"}

    ratios = {}
    if is_metrics["wr"] > 0:
        ratios["wr"] = oos_metrics["wr"] / is_metrics["wr"]
    else:
        ratios["wr"] = 0

    if is_metrics["pf"] > 0 and is_metrics["pf"] != float("inf"):
        if oos_metrics["pf"] == float("inf"):
            ratios["pf"] = 2.0
        else:
            ratios["pf"] = oos_metrics["pf"] / is_metrics["pf"]
    else:
        ratios["pf"] = 0

    if is_metrics["expectancy"] > 0:
        ratios["expectancy"] = oos_metrics["expectancy"] / is_metrics["expectancy"]
    elif is_metrics["expectancy"] < 0 and oos_metrics["expectancy"] < 0:
        ratios["expectancy"] = 0
    else:
        ratios["expectancy"] = 0

    if is_metrics["sharpe"] > 0:
        ratios["sharpe"] = oos_metrics["sharpe"] / is_metrics["sharpe"]
    else:
        ratios["sharpe"] = 0

    for k in ratios:
        ratios[k] = min(ratios[k], 2.0)
        ratios[k] = max(ratios[k], 0.0)

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


def main():
    t0 = time.time()
    all_results = {}

    print(f"{'='*105}")
    print("TEST 5-MINUTE STRATEGIES ON M5 DATA (IS/OOS Split)")
    print(f"  IS period:  {IS_START} to {IS_END}")
    print(f"  OOS period: {OOS_START} to {OOS_END}")
    print(f"  Configs: {len(CONFIGS)} strategy-pair combos")
    print(f"{'='*105}")

    data_cache = {}

    for cfg in CONFIGS:
        name = cfg["name"]
        pair = cfg["pair"]
        tf = cfg["tf"]
        htf_tf = cfg["htf_tf"]

        print(f"\n  {name} / {pair} ({tf} + {htf_tf})...")

        # Load primary M5 data (cached)
        cache_key = f"{pair}_{tf}"
        if cache_key not in data_cache:
            print(f"    Loading {pair} {tf}...", end=" ", flush=True)
            data_cache[cache_key] = load_data(pair, tf)
            if data_cache[cache_key] is not None:
                print(f"{len(data_cache[cache_key])} bars")
            else:
                print("MISSING")
        data = data_cache[cache_key]
        if data is None:
            continue

        # Load HTF H1 data (cached)
        htf_cache_key = f"{pair}_{htf_tf}"
        if htf_cache_key not in data_cache:
            data_cache[htf_cache_key] = load_data(pair, htf_tf)
        htf_data = data_cache[htf_cache_key]
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

        # Print
        print(f"    IS:  {is_metrics['trades']:>4}t  WR={is_metrics['wr']:>5.1f}%  "
              f"PF={is_metrics['pf']:>5.2f}  Sharpe={is_metrics['sharpe']:>6.2f}  "
              f"PnL={is_metrics['pnl_pips']:>+8.1f}p  DD={is_metrics['max_dd_pips']:>+7.1f}p")
        print(f"    OOS: {oos_metrics['trades']:>4}t  WR={oos_metrics['wr']:>5.1f}%  "
              f"PF={oos_metrics['pf']:>5.2f}  Sharpe={oos_metrics['sharpe']:>6.2f}  "
              f"PnL={oos_metrics['pnl_pips']:>+8.1f}p  DD={oos_metrics['max_dd_pips']:>+7.1f}p  "
              f"Gen={gen['composite']:>5.3f} {gen['verdict']}")

        all_results[name] = {
            "pair": pair, "timeframe": tf,
            "is_metrics": is_metrics, "oos_metrics": oos_metrics,
            "generalization": gen,
        }

    # Summary table
    print(f"\n{'='*105}")
    print("SUMMARY — SORTED BY OOS PROFIT FACTOR")
    print(f"{'='*105}")
    print(f"  {'Strategy':<20} {'Pair':<10} {'IS-t':>5} {'IS PF':>6} "
          f"{'OOS-t':>6} {'OOS PF':>7} {'OOS WR%':>8} "
          f"{'Gen':>6} {'Verdict':>8}")
    print(f"  {'-'*85}")

    sorted_results = sorted(all_results.items(),
                            key=lambda x: x[1]["oos_metrics"]["pf"]
                            if x[1]["oos_metrics"]["pf"] != float("inf") else 99,
                            reverse=True)

    for name, res in sorted_results:
        is_m = res["is_metrics"]
        oos_m = res["oos_metrics"]
        gen = res["generalization"]
        pf_str = f"{oos_m['pf']:.2f}" if oos_m['pf'] != float('inf') else "inf"
        print(f"  {name:<20} {res['pair']:<10} {is_m['trades']:>5} {is_m['pf']:>6.2f} "
              f"{oos_m['trades']:>6} {pf_str:>7} {oos_m['wr']:>7.1f}% "
              f"{gen['composite']:>5.3f} {gen['verdict']:>8}")

    # Highlight promising strategies
    print(f"\n{'='*105}")
    print("PROMISING (OOS PF > 1.0, OOS trades >= 5, Gen >= 0.50)")
    print(f"{'='*105}")
    promising = [(n, r) for n, r in sorted_results
                 if r["oos_metrics"]["pf"] > 1.0
                 and r["oos_metrics"]["trades"] >= 5
                 and r["generalization"]["composite"] >= 0.50]
    if promising:
        for name, res in promising:
            is_m = res["is_metrics"]
            oos_m = res["oos_metrics"]
            gen = res["generalization"]
            print(f"  {name:<20} IS: {is_m['trades']}t PF={is_m['pf']:.2f} WR={is_m['wr']:.1f}%  "
                  f"OOS: {oos_m['trades']}t PF={oos_m['pf']:.2f} WR={oos_m['wr']:.1f}%  "
                  f"Gen={gen['composite']:.3f} {gen['verdict']}")
    else:
        print("  None found.")

    # Save JSON report
    out_path = os.path.join(RESULTS_DIR, "test_m5_strategies.json")

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
