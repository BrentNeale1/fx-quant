"""
Phase 2 — Correlation Analysis (Step 8).

For strategies sharing a pair (S7_Tight + S3 on GBP_JPY):
- Compute signal overlap and simultaneous position frequency.
- Combined equity curve analysis.

Also compute portfolio-level metrics: combined PF, combined max DD,
Sharpe of combined equity curve.

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
from src.strategies_pkg.s4f_ema_ribbon import S4F_EMA_Ribbon
from src.strategies_pkg.s3_key_level_breakout import S3_KeyLevel_Breakout

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase2")
os.makedirs(RESULTS_DIR, exist_ok=True)

# All Phase 2 strategies
CONFIGS = [
    {"name": "S7_Tight", "pair": "GBP_JPY", "tf": "H1",
     "factory": lambda: S7_Liquidity_Sweep()},
    {"name": "S9", "pair": "GBP_USD", "tf": "H1",
     "factory": lambda: S9_London_Session()},
    {"name": "S9_Filtered", "pair": "GBP_AUD", "tf": "H1",
     "factory": lambda: S9_London_Session(pair="GBP_AUD", filtered=True)},
    {"name": "S4F", "pair": "EUR_AUD", "tf": "M15",
     "factory": lambda: S4F_EMA_Ribbon()},
    {"name": "S3", "pair": "GBP_JPY", "tf": "H1",
     "factory": lambda: S3_KeyLevel_Breakout()},
]


def load_data(pair, tf):
    fp = os.path.join(PROCESSED_DIR, f"{pair}_{tf}.csv")
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    return compute_all_indicators(df)


def run_backtest(cfg):
    """Run backtest for a config, return trade log and equity curve."""
    pair = cfg["pair"]
    tf = cfg["tf"]
    data = load_data(pair, tf)
    if data is None:
        return None, None, None

    htf_data = data.copy() if tf == "H1" else load_data(pair, "H1")

    strategy = cfg["factory"]()
    bt = Backtester(data=data, strategy=strategy, pair=pair,
                    starting_equity=100_000.0, htf_data=htf_data)
    report = bt.run()
    trade_log = bt.get_trade_log_df()
    eq_curve = pd.DataFrame(bt.equity_curve)
    return report, trade_log, eq_curve


def compute_signal_overlap(log_a, log_b, pair):
    """Compute signal overlap between two strategies on the same pair.

    Returns:
      - overlap_count: trades that are open at the same time
      - same_direction_count: overlapping trades in same direction
      - opposite_direction_count: overlapping trades in opposite direction
      - overlap_ratio: fraction of trades that overlap
    """
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

            # Check if time ranges overlap
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


def compute_combined_equity(eq_curves: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine equity curves from multiple strategies into portfolio equity."""
    combined = None
    for eq in eq_curves:
        if eq is None or eq.empty:
            continue
        eq = eq.set_index("timestamp")["equity"]
        # Convert to returns relative to starting equity
        returns = eq - 100_000.0
        if combined is None:
            combined = returns
        else:
            combined = combined.add(returns, fill_value=0)

    if combined is None:
        return pd.DataFrame()

    # Add back starting equity (100k per slot, or just use combined returns)
    combined = combined + 100_000.0
    return combined.reset_index()


def compute_portfolio_metrics(all_trade_logs: list[pd.DataFrame]) -> dict:
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

    # Combined max drawdown
    cum_pnl = combined.sort_values("timestamp")["pnl_dollars"].cumsum()
    peak = cum_pnl.cummax()
    dd = cum_pnl - peak
    max_dd = dd.min()
    max_dd_pct = max_dd / 100_000 * 100 if max_dd < 0 else 0

    # Sharpe ratio
    daily_pnl = combined.copy()
    daily_pnl["date"] = pd.to_datetime(daily_pnl["timestamp"]).dt.date
    daily = daily_pnl.groupby("date")["pnl_dollars"].sum()
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
    }


def main():
    print(f"{'='*80}")
    print("PHASE 2 — CORRELATION ANALYSIS")
    print(f"{'='*80}")

    results = {}
    trade_logs = {}
    eq_curves = {}

    # Run all backtests
    for cfg in CONFIGS:
        name = cfg["name"]
        pair = cfg["pair"]
        print(f"\nRunning {name} / {pair}...", end=" ", flush=True)
        t0 = time.time()
        report, log, eq = run_backtest(cfg)
        elapsed = time.time() - t0
        n_trades = len(log) if log is not None and not log.empty else 0
        print(f"{n_trades} trades ({elapsed:.0f}s)")
        trade_logs[name] = log
        eq_curves[name] = eq

    # --- Signal Overlap: S7_Tight vs S3 on GBP_JPY ---
    print(f"\n{'#'*60}")
    print("# Signal Overlap: S7_Tight vs S3 on GBP_JPY")
    print(f"{'#'*60}")

    log_s7 = trade_logs.get("S7_Tight", pd.DataFrame())
    log_s3 = trade_logs.get("S3", pd.DataFrame())

    if not log_s7.empty and not log_s3.empty:
        overlap = compute_signal_overlap(log_s7, log_s3, "GBP_JPY")
        results["S7_S3_overlap"] = overlap

        print(f"  S7 trades: {len(log_s7)}")
        print(f"  S3 trades: {len(log_s3)}")
        print(f"  Overlapping periods: {overlap['overlap_count']}")
        print(f"    Same direction: {overlap['same_dir']}")
        print(f"    Opposite direction: {overlap['opposite_dir']}")
        print(f"  Overlap ratio: {overlap['ratio']:.1%}")

        if overlap['ratio'] < 0.15:
            print("  => LOW overlap: Good diversification!")
        elif overlap['ratio'] < 0.30:
            print("  => MODERATE overlap: Some clustering.")
        else:
            print("  => HIGH overlap: Significant clustering risk.")
    else:
        print("  Insufficient data for overlap analysis.")

    # --- S9 vs S9_Filtered (different pairs, should be independent) ---
    print(f"\n{'#'*60}")
    print("# Independence Check: S9 (GBP_USD) vs S9_Filtered (GBP_AUD)")
    print(f"{'#'*60}")

    log_s9 = trade_logs.get("S9", pd.DataFrame())
    log_s9f = trade_logs.get("S9_Filtered", pd.DataFrame())

    if not log_s9.empty and not log_s9f.empty:
        # Check temporal clustering (same-day entries)
        s9_dates = set(pd.to_datetime(log_s9["timestamp"]).dt.date)
        s9f_dates = set(pd.to_datetime(log_s9f["timestamp"]).dt.date)
        shared_dates = s9_dates & s9f_dates
        temporal_overlap = len(shared_dates) / max(len(s9_dates), 1)

        results["S9_S9F_temporal"] = {
            "s9_trade_days": len(s9_dates),
            "s9f_trade_days": len(s9f_dates),
            "shared_trade_days": len(shared_dates),
            "temporal_overlap_ratio": round(temporal_overlap, 3),
        }

        print(f"  S9 trade days: {len(s9_dates)}")
        print(f"  S9_Filtered trade days: {len(s9f_dates)}")
        print(f"  Shared trade days: {len(shared_dates)}")
        print(f"  Temporal overlap: {temporal_overlap:.1%}")
    else:
        print("  Insufficient data.")

    # --- Portfolio Metrics ---
    print(f"\n{'#'*60}")
    print("# Portfolio-Level Metrics (All 5 Strategies Combined)")
    print(f"{'#'*60}")

    all_logs = [log for log in trade_logs.values()
                if log is not None and not log.empty]
    if all_logs:
        portfolio = compute_portfolio_metrics(all_logs)
        results["portfolio"] = portfolio

        print(f"  Total trades: {portfolio['total_trades']}")
        print(f"  Win rate: {portfolio['win_rate_pct']}%")
        print(f"  Profit factor: {portfolio['profit_factor']}")
        print(f"  Total PnL (pips): {portfolio['total_pnl_pips']:+.1f}")
        print(f"  Total PnL ($): {portfolio['total_pnl_dollars']:+,.2f}")
        print(f"  Max drawdown: {portfolio['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe ratio: {portfolio['sharpe_ratio']:.2f}")

    # --- Per-Strategy Summary ---
    print(f"\n{'='*80}")
    print("STRATEGY SUMMARY")
    print(f"{'='*80}")
    print(f"{'Strategy':<16} {'Pair':<10} {'Trades':>6} {'WR%':>6} {'PF':>6} {'PnL(p)':>9}")
    print(f"{'-'*60}")

    for cfg in CONFIGS:
        name = cfg["name"]
        pair = cfg["pair"]
        log = trade_logs.get(name, pd.DataFrame())
        if log.empty:
            print(f"{name:<16} {pair:<10} {'N/A':>6}")
            continue
        n = len(log)
        wins = log[log["win"] == True]
        wr = len(wins) / n * 100 if n > 0 else 0
        gp = wins["pnl_pips"].sum() if len(wins) > 0 else 0
        gl = abs(log[log["win"] == False]["pnl_pips"].sum())
        pf = gp / gl if gl > 0 else 0
        pnl = log["pnl_pips"].sum()
        print(f"{name:<16} {pair:<10} {n:>6} {wr:>5.1f}% {pf:>5.2f} {pnl:>+8.1f}")

    # Save
    out_path = os.path.join(RESULTS_DIR, "correlation_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
