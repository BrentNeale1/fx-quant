"""
Test New Strategies (S12-S16) — IS/OOS Backtest.

Strategies:
  S12 - Asian Range Sweep (M15 + H1 HTF) — mean reversion after sweep
  S13 - Bollinger-Keltner Squeeze (M15 + H1 HTF) — volatility breakout
  S14 - London Fix (M15 + H1 HTF) — post-fix mean reversion
  S15 - Momentum Continuation (M15 + H1 HTF) — impulse pullback entry
  S16 - London ORB (M15 + H1 HTF) — opening range breakout

Each tested across multiple pairs.
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
from src.strategies_pkg.s13_bollinger_keltner_squeeze import S13_BollingerKeltnerSqueeze
from src.strategies_pkg.s14_london_fix import S14_LondonFix
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

# All M15 strategies with H1 as HTF
CONFIGS = [
    # S12: Asian Range Sweep — best on liquid London pairs
    {"name": "S12_GBP_JPY", "pair": "GBP_JPY", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: S12_AsianRangeSweep()},
    {"name": "S12_GBP_USD", "pair": "GBP_USD", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: S12_AsianRangeSweep()},
    {"name": "S12_EUR_USD", "pair": "EUR_USD", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: S12_AsianRangeSweep()},
    {"name": "S12_USD_JPY", "pair": "USD_JPY", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: S12_AsianRangeSweep()},

    # S13: Bollinger-Keltner Squeeze — works across many pairs
    {"name": "S13_GBP_JPY", "pair": "GBP_JPY", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: S13_BollingerKeltnerSqueeze()},
    {"name": "S13_EUR_USD", "pair": "EUR_USD", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: S13_BollingerKeltnerSqueeze()},
    {"name": "S13_GBP_USD", "pair": "GBP_USD", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: S13_BollingerKeltnerSqueeze()},
    {"name": "S13_USD_JPY", "pair": "USD_JPY", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: S13_BollingerKeltnerSqueeze()},

    # S14: London Fix — best on GBP and EUR pairs
    {"name": "S14_GBP_USD", "pair": "GBP_USD", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: S14_LondonFix()},
    {"name": "S14_EUR_USD", "pair": "EUR_USD", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: S14_LondonFix()},
    {"name": "S14_EUR_GBP", "pair": "EUR_GBP", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: S14_LondonFix()},
    {"name": "S14_GBP_JPY", "pair": "GBP_JPY", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: S14_LondonFix()},

    # S15: Momentum Continuation — trending pairs
    {"name": "S15_GBP_JPY", "pair": "GBP_JPY", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: S15_MomentumContinuation()},
    {"name": "S15_USD_JPY", "pair": "USD_JPY", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: S15_MomentumContinuation()},
    {"name": "S15_GBP_USD", "pair": "GBP_USD", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: S15_MomentumContinuation()},
    {"name": "S15_EUR_USD", "pair": "EUR_USD", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: S15_MomentumContinuation()},

    # S16: London ORB — best on London-active pairs
    {"name": "S16_GBP_JPY", "pair": "GBP_JPY", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: S16_LondonORB()},
    {"name": "S16_GBP_USD", "pair": "GBP_USD", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: S16_LondonORB()},
    {"name": "S16_EUR_USD", "pair": "EUR_USD", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: S16_LondonORB()},
    {"name": "S16_USD_JPY", "pair": "USD_JPY", "tf": "M15", "htf_tf": "H1",
     "factory": lambda: S16_LondonORB()},
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
    print("TEST NEW STRATEGIES S12-S16 (IS/OOS Split)")
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

        # Load primary data (cached)
        cache_key = f"{pair}_{tf}"
        if cache_key not in data_cache:
            data_cache[cache_key] = load_data(pair, tf)
        data = data_cache[cache_key]
        if data is None:
            continue

        # Load HTF data (cached)
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
    print(f"  {'Strategy':<16} {'Pair':<10} {'IS-t':>5} {'IS PF':>6} "
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
        print(f"  {name:<16} {res['pair']:<10} {is_m['trades']:>5} {is_m['pf']:>6.2f} "
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
            print(f"  {name:<16} IS: {is_m['trades']}t PF={is_m['pf']:.2f} WR={is_m['wr']:.1f}%  "
                  f"OOS: {oos_m['trades']}t PF={oos_m['pf']:.2f} WR={oos_m['wr']:.1f}%  "
                  f"Gen={gen['composite']:.3f} {gen['verdict']}")
    else:
        print("  None found.")

    # Save JSON report
    out_path = os.path.join(RESULTS_DIR, "test_new_strategies.json")

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
