"""
Phase 2 — Correlation Analysis (4-Strategy Portfolio).

Analyzes diversification across the portfolio:
  S7_Tight   / GBP_JPY / H1   — Liquidity Sweep
  S9_Filtered/ GBP_AUD / H1   — London Session
  S3         / GBP_JPY / H1   — Key Level Breakout
  S8_OB      / GBP_USD / M15  — Order Block Retest

Computes:
  1. Signal overlap: S7 vs S3 (same pair GBP_JPY)
  2. Daily PnL correlation matrix (all strategy pairs)
  3. Temporal clustering (same-day trade entries)
  4. Portfolio-level metrics and combined equity curve

Output: results/phase2/correlation_analysis.json
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
     "factory": lambda: _s8_tuned()},
]


def load_data(pair, tf):
    fp = os.path.join(PROCESSED_DIR, f"{pair}_{tf}.csv")
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    return compute_all_indicators(df)


def run_backtest(cfg, data_cache):
    """Run backtest for a config, return trade log and equity curve."""
    pair = cfg["pair"]
    tf = cfg["tf"]
    htf_tf = cfg["htf_tf"]

    cache_key = f"{pair}_{tf}"
    if cache_key not in data_cache:
        data_cache[cache_key] = load_data(pair, tf)
    data = data_cache[cache_key]
    if data is None:
        return None, None, None

    htf_cache_key = f"{pair}_{htf_tf}"
    if htf_cache_key not in data_cache:
        data_cache[htf_cache_key] = load_data(pair, htf_tf)
    htf_data = data_cache[htf_cache_key] if htf_tf != tf else data

    strategy = cfg["factory"]()
    bt = Backtester(data=data, strategy=strategy, pair=pair,
                    starting_equity=100_000.0, htf_data=htf_data)
    report = bt.run()
    trade_log = bt.get_trade_log_df()
    eq_curve = pd.DataFrame(bt.equity_curve)
    return report, trade_log, eq_curve


def compute_signal_overlap(log_a, log_b):
    """Compute signal overlap between two strategies (time-based)."""
    if log_a.empty or log_b.empty:
        return {"overlap_count": 0, "same_dir": 0, "opposite_dir": 0, "ratio": 0}

    overlap = 0
    same_dir = 0
    opp_dir = 0

    for _, trade_a in log_a.iterrows():
        a_start = pd.Timestamp(trade_a["timestamp"])
        a_end = pd.Timestamp(trade_a["exit_time"]) if pd.notna(trade_a.get("exit_time")) else a_start

        for _, trade_b in log_b.iterrows():
            b_start = pd.Timestamp(trade_b["timestamp"])
            b_end = pd.Timestamp(trade_b["exit_time"]) if pd.notna(trade_b.get("exit_time")) else b_start

            if a_start <= b_end and b_start <= a_end:
                overlap += 1
                if trade_a["signal_direction"] == trade_b["signal_direction"]:
                    same_dir += 1
                else:
                    opp_dir += 1

    total = len(log_a) + len(log_b)
    ratio = (overlap * 2) / total if total > 0 else 0

    return {
        "overlap_count": overlap,
        "same_dir": same_dir,
        "opposite_dir": opp_dir,
        "ratio": round(ratio, 3),
    }


def compute_daily_pnl_series(trade_logs):
    """Build daily PnL series per strategy for correlation analysis."""
    daily_pnl = {}
    for name, log in trade_logs.items():
        if log is None or log.empty:
            continue
        df = log.copy()
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        daily = df.groupby("date")["pnl_pips"].sum()
        daily_pnl[name] = daily
    return daily_pnl


def compute_pnl_correlation(daily_pnl):
    """Compute pairwise correlation of daily PnL between strategies."""
    if len(daily_pnl) < 2:
        return pd.DataFrame()

    combined = pd.DataFrame(daily_pnl)
    combined = combined.fillna(0)

    return combined.corr()


def compute_temporal_overlap(trade_logs):
    """Compute pairwise temporal overlap (same-day entries) between all strategies."""
    names = list(trade_logs.keys())
    results = {}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            log_a = trade_logs[a_name]
            log_b = trade_logs[b_name]

            if log_a is None or log_a.empty or log_b is None or log_b.empty:
                continue

            dates_a = set(pd.to_datetime(log_a["timestamp"]).dt.date)
            dates_b = set(pd.to_datetime(log_b["timestamp"]).dt.date)
            shared = dates_a & dates_b
            union = dates_a | dates_b

            key = f"{a_name}_vs_{b_name}"
            results[key] = {
                "trade_days_a": len(dates_a),
                "trade_days_b": len(dates_b),
                "shared_days": len(shared),
                "jaccard_index": round(len(shared) / len(union), 3) if union else 0,
            }

    return results


def compute_portfolio_metrics(all_trade_logs):
    """Compute portfolio-level metrics from combined trade logs."""
    combined = pd.concat([log for log in all_trade_logs if not log.empty],
                         ignore_index=True)
    if combined.empty:
        return {}

    n = len(combined)
    wins = combined[combined["win"] == True]
    losses = combined[combined["win"] == False]

    wr = len(wins) / n * 100 if n > 0 else 0
    gross_profit = wins["pnl_pips"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl_pips"].sum()) if len(losses) > 0 else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_pnl = combined["pnl_pips"].sum()
    expectancy = total_pnl / n if n > 0 else 0

    # Combined max drawdown
    sorted_trades = combined.sort_values("timestamp")
    cum_pnl = sorted_trades["pnl_dollars"].cumsum()
    peak = cum_pnl.cummax()
    dd = cum_pnl - peak
    max_dd = dd.min()
    max_dd_pct = max_dd / 100_000 * 100 if max_dd < 0 else 0

    # Sharpe ratio from daily PnL
    sorted_trades = sorted_trades.copy()
    sorted_trades["date"] = pd.to_datetime(sorted_trades["timestamp"]).dt.date
    daily = sorted_trades.groupby("date")["pnl_dollars"].sum()
    if len(daily) > 1 and daily.std() > 0:
        sharpe = (daily.mean() / daily.std()) * np.sqrt(252)
    else:
        sharpe = 0

    return {
        "total_trades": n,
        "win_rate_pct": round(wr, 1),
        "profit_factor": round(pf, 2),
        "total_pnl_pips": round(total_pnl, 1),
        "total_pnl_dollars": round(combined["pnl_dollars"].sum(), 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "expectancy_pips": round(expectancy, 2),
    }


def main():
    t0 = time.time()

    print(f"{'='*80}")
    print("PHASE 2 — CORRELATION ANALYSIS (4-Strategy Portfolio)")
    print(f"{'='*80}")

    results = {}
    trade_logs = {}
    eq_curves = {}
    data_cache = {}

    # Run all backtests
    for cfg in CONFIGS:
        name = cfg["name"]
        pair = cfg["pair"]
        tf = cfg["tf"]
        print(f"\n  Running {name} / {pair} ({tf})...", end=" ", flush=True)
        t1 = time.time()
        report, log, eq = run_backtest(cfg, data_cache)
        elapsed = time.time() - t1
        n_trades = len(log) if log is not None and not log.empty else 0
        print(f"{n_trades} trades ({elapsed:.0f}s)")
        trade_logs[name] = log
        eq_curves[name] = eq

    # =====================================================================
    # 1. Signal Overlap: S7_Tight vs S3 (both on GBP_JPY)
    # =====================================================================
    print(f"\n{'#'*70}")
    print("# 1. SIGNAL OVERLAP: S7_Tight vs S3 (GBP_JPY)")
    print(f"{'#'*70}")

    log_s7 = trade_logs.get("S7_Tight", pd.DataFrame())
    log_s3 = trade_logs.get("S3", pd.DataFrame())

    if not log_s7.empty and not log_s3.empty:
        overlap = compute_signal_overlap(log_s7, log_s3)
        results["S7_S3_signal_overlap"] = overlap

        print(f"  S7_Tight trades: {len(log_s7)}")
        print(f"  S3 trades:       {len(log_s3)}")
        print(f"  Overlapping periods: {overlap['overlap_count']}")
        print(f"    Same direction:     {overlap['same_dir']}")
        print(f"    Opposite direction: {overlap['opposite_dir']}")
        print(f"  Overlap ratio: {overlap['ratio']:.1%}")

        if overlap['ratio'] < 0.15:
            print("  => LOW overlap: Good diversification within GBP_JPY")
        elif overlap['ratio'] < 0.30:
            print("  => MODERATE overlap: Some clustering on GBP_JPY")
        else:
            print("  => HIGH overlap: Significant clustering risk on GBP_JPY")
    else:
        print("  Insufficient data for overlap analysis.")

    # =====================================================================
    # 2. Daily PnL Correlation Matrix
    # =====================================================================
    print(f"\n{'#'*70}")
    print("# 2. DAILY PnL CORRELATION MATRIX")
    print(f"{'#'*70}")

    daily_pnl = compute_daily_pnl_series(trade_logs)
    corr_matrix = compute_pnl_correlation(daily_pnl)

    if not corr_matrix.empty:
        results["pnl_correlation"] = {
            f"{a}_vs_{b}": round(corr_matrix.loc[a, b], 3)
            for i, a in enumerate(corr_matrix.index)
            for j, b in enumerate(corr_matrix.columns)
            if j > i
        }

        print(f"\n  {'':>14}", end="")
        for name in corr_matrix.columns:
            print(f" {name:>12}", end="")
        print()

        for row_name in corr_matrix.index:
            print(f"  {row_name:>14}", end="")
            for col_name in corr_matrix.columns:
                val = corr_matrix.loc[row_name, col_name]
                print(f" {val:>12.3f}", end="")
            print()

        # Average pairwise correlation
        pairs = []
        for i, a in enumerate(corr_matrix.index):
            for j, b in enumerate(corr_matrix.columns):
                if j > i:
                    pairs.append(corr_matrix.loc[a, b])
        avg_corr = np.mean(pairs) if pairs else 0
        results["avg_pairwise_correlation"] = round(avg_corr, 3)

        print(f"\n  Average pairwise correlation: {avg_corr:.3f}")
        if avg_corr < 0.20:
            print("  => LOW correlation: Excellent diversification")
        elif avg_corr < 0.40:
            print("  => MODERATE correlation: Decent diversification")
        else:
            print("  => HIGH correlation: Limited diversification benefit")
    else:
        print("  Insufficient data.")

    # =====================================================================
    # 3. Temporal Overlap (Same-Day Entry Clustering)
    # =====================================================================
    print(f"\n{'#'*70}")
    print("# 3. TEMPORAL OVERLAP (Same-Day Entries)")
    print(f"{'#'*70}")

    temporal = compute_temporal_overlap(trade_logs)
    results["temporal_overlap"] = temporal

    print(f"\n  {'Pair':<30} {'Days A':>7} {'Days B':>7} {'Shared':>7} {'Jaccard':>8}")
    print(f"  {'-'*65}")
    for key, val in temporal.items():
        print(f"  {key:<30} {val['trade_days_a']:>7} {val['trade_days_b']:>7} "
              f"{val['shared_days']:>7} {val['jaccard_index']:>7.3f}")

    # =====================================================================
    # 4. Portfolio Metrics
    # =====================================================================
    print(f"\n{'#'*70}")
    print("# 4. PORTFOLIO METRICS (All 4 Strategies Combined)")
    print(f"{'#'*70}")

    all_logs = [log for log in trade_logs.values()
                if log is not None and not log.empty]
    if all_logs:
        portfolio = compute_portfolio_metrics(all_logs)
        results["portfolio"] = portfolio

        print(f"  Total trades:    {portfolio['total_trades']}")
        print(f"  Win rate:        {portfolio['win_rate_pct']}%")
        print(f"  Profit factor:   {portfolio['profit_factor']}")
        print(f"  Expectancy:      {portfolio['expectancy_pips']:+.2f} pips/trade")
        print(f"  Total PnL:       {portfolio['total_pnl_pips']:+.1f} pips "
              f"(${portfolio['total_pnl_dollars']:+,.2f})")
        print(f"  Max drawdown:    {portfolio['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe ratio:    {portfolio['sharpe_ratio']:.2f}")

    # =====================================================================
    # 5. Per-Strategy Summary
    # =====================================================================
    print(f"\n{'='*80}")
    print("STRATEGY SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Strategy':<14} {'Pair':<10} {'TF':<4} {'Trades':>6} {'WR%':>6} "
          f"{'PF':>6} {'PnL(p)':>9} {'Exp':>7}")
    print(f"  {'-'*70}")

    for cfg in CONFIGS:
        name = cfg["name"]
        pair = cfg["pair"]
        tf = cfg["tf"]
        log = trade_logs.get(name, pd.DataFrame())
        if log.empty:
            print(f"  {name:<14} {pair:<10} {tf:<4} {'N/A':>6}")
            continue
        n = len(log)
        wins = log[log["win"] == True]
        wr = len(wins) / n * 100 if n > 0 else 0
        gp = wins["pnl_pips"].sum() if len(wins) > 0 else 0
        gl = abs(log[log["win"] == False]["pnl_pips"].sum())
        pf = gp / gl if gl > 0 else 0
        pnl = log["pnl_pips"].sum()
        exp = pnl / n if n > 0 else 0
        print(f"  {name:<14} {pair:<10} {tf:<4} {n:>6} {wr:>5.1f}% "
              f"{pf:>5.2f} {pnl:>+8.1f} {exp:>+6.2f}")

    # Save
    out_path = os.path.join(RESULTS_DIR, "correlation_analysis.json")

    def json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return str(obj)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=json_default)
    print(f"\nResults saved: {out_path}")

    elapsed = time.time() - t0
    print(f"Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
