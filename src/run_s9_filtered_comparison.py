"""
S9 Filtered vs Baseline comparison on EUR_USD and GBP_AUD.

Filters applied:
  EUR_USD: ADX > 30, skip RSI 40-60, entry from Hour 8, TP1 = 1.5x range
  GBP_AUD: ADX > 25, skip Friday, EMA50 distance > 40 pips
"""
import os, sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from src.indicators.technical import compute_all_indicators
from src.backtester.engine import Backtester
from src.strategies_pkg.s9_london_session import S9_London_Session

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase1")
os.makedirs(RESULTS_DIR, exist_ok=True)

PAIRS = ["EUR_USD", "GBP_AUD"]
TRAIN_RATIO = 0.70


def load_data(pair):
    fp = os.path.join(PROCESSED_DIR, f"{pair}_H1.csv")
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    df = compute_all_indicators(df)
    return df


def split_data(df):
    split_idx = int(len(df) * TRAIN_RATIO)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def run_test(strategy, pair, data, htf_data, phase="FULL"):
    if phase == "TRAIN":
        data, _ = split_data(data)
        htf_data, _ = split_data(htf_data)
    elif phase == "TEST":
        _, data = split_data(data)
        _, htf_data = split_data(htf_data)

    bt = Backtester(data=data, strategy=strategy, pair=pair,
                    starting_equity=100_000.0, htf_data=htf_data)
    report = bt.run()
    trade_log = bt.get_trade_log_df()
    return report, trade_log


def extract_metrics(report, trade_log):
    n = report.get("total_trades", 0)
    wr = report.get("win_rate_pct", 0)
    pf = report.get("profit_factor", 0)
    pnl = report.get("total_pnl_pips", 0)
    pnl_d = report.get("total_pnl_dollars", 0)
    dd = report.get("max_drawdown_pct", 0)
    avg_w = report.get("avg_win_pips", 0)
    avg_l = report.get("avg_loss_pips", 0)
    rr = avg_w / avg_l if avg_l > 0 else 0
    exp = pnl / n if n > 0 else 0
    return {
        "trades": n, "wr": round(wr, 1), "pf": round(pf, 2),
        "pnl_pips": round(pnl, 1), "pnl_usd": round(pnl_d, 2),
        "dd": round(dd, 2), "avg_win": round(avg_w, 1),
        "avg_loss": round(avg_l, 1), "rr": round(rr, 2),
        "expectancy": round(exp, 2),
    }


if __name__ == "__main__":
    all_results = {}

    for pair in PAIRS:
        print(f"\n{'#' * 70}")
        print(f"# {pair}")
        print(f"{'#' * 70}")

        data = load_data(pair)
        htf_data = data.copy()
        print(f"  Data: {len(data)} H1 bars, {data.index[0].date()} to {data.index[-1].date()}")

        pair_results = {}

        for variant, filtered in [("BASELINE", False), ("FILTERED", True)]:
            print(f"\n  --- {variant} ---")
            variant_results = {}

            for phase in ["FULL", "TRAIN", "TEST"]:
                strat = S9_London_Session(pair=pair, filtered=filtered)
                t0 = time.time()
                report, trade_log = run_test(strat, pair, data, htf_data, phase)
                elapsed = time.time() - t0

                m = extract_metrics(report, trade_log)
                variant_results[phase] = m

                print(f"  {phase:>5}: {m['trades']:>3} trades | WR {m['wr']:>5.1f}% | "
                      f"PF {m['pf']:>5.2f} | PnL {m['pnl_pips']:>+8.1f}p | "
                      f"DD {m['dd']:>6.2f}% | AvgW {m['avg_win']:>5.1f} / "
                      f"AvgL {m['avg_loss']:>5.1f} | RR {m['rr']:>4.2f} | "
                      f"Exp {m['expectancy']:>+6.2f}p ({elapsed:.0f}s)")

                # Save trade log for filtered version
                if filtered and len(trade_log) > 0:
                    csv_path = os.path.join(RESULTS_DIR, f"S9F_{pair}_{phase}_trades.csv")
                    trade_log.to_csv(csv_path, index=False)

            # Generalization ratio
            train_pf = variant_results.get("TRAIN", {}).get("pf", 0)
            test_pf = variant_results.get("TEST", {}).get("pf", 0)
            if train_pf > 0 and test_pf > 0:
                gen = test_pf / train_pf
                print(f"  Gen Ratio: {gen:.2f}")

            pair_results[variant] = variant_results

        # Delta summary
        base_full = pair_results["BASELINE"]["FULL"]
        filt_full = pair_results["FILTERED"]["FULL"]
        base_test = pair_results["BASELINE"]["TEST"]
        filt_test = pair_results["FILTERED"]["TEST"]

        print(f"\n  IMPROVEMENT (FULL):")
        print(f"    Trades: {base_full['trades']} -> {filt_full['trades']} "
              f"({filt_full['trades'] - base_full['trades']:+d})")
        print(f"    WR:     {base_full['wr']}% -> {filt_full['wr']}% "
              f"({filt_full['wr'] - base_full['wr']:+.1f}%)")
        print(f"    PF:     {base_full['pf']} -> {filt_full['pf']} "
              f"({filt_full['pf'] - base_full['pf']:+.2f})")
        print(f"    PnL:    {base_full['pnl_pips']:+.1f}p -> {filt_full['pnl_pips']:+.1f}p "
              f"({filt_full['pnl_pips'] - base_full['pnl_pips']:+.1f}p)")
        print(f"    RR:     {base_full['rr']} -> {filt_full['rr']} "
              f"({filt_full['rr'] - base_full['rr']:+.2f})")

        print(f"\n  IMPROVEMENT (OOS TEST):")
        print(f"    Trades: {base_test['trades']} -> {filt_test['trades']}")
        print(f"    WR:     {base_test['wr']}% -> {filt_test['wr']}%")
        print(f"    PF:     {base_test['pf']} -> {filt_test['pf']}")
        print(f"    PnL:    {base_test['pnl_pips']:+.1f}p -> {filt_test['pnl_pips']:+.1f}p")

        all_results[pair] = pair_results

    # Final comparison table
    print(f"\n{'=' * 80}")
    print("FINAL COMPARISON: BASELINE vs FILTERED")
    print(f"{'=' * 80}")
    print(f"{'Pair':<10} {'Variant':<10} {'Phase':<6} {'Trades':>7} {'WR%':>6} "
          f"{'PF':>6} {'PnL(p)':>9} {'RR':>5} {'Exp(p)':>7}")
    print("-" * 80)
    for pair in PAIRS:
        for variant in ["BASELINE", "FILTERED"]:
            for phase in ["FULL", "TEST"]:
                m = all_results[pair][variant][phase]
                print(f"{pair:<10} {variant:<10} {phase:<6} {m['trades']:>7} "
                      f"{m['wr']:>5.1f}% {m['pf']:>5.2f} {m['pnl_pips']:>+8.1f}p "
                      f"{m['rr']:>4.2f} {m['expectancy']:>+6.2f}p")
            if variant == "BASELINE":
                print()
        print("-" * 80)

    # Save
    out_path = os.path.join(RESULTS_DIR, "s9_filtered_comparison.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")
