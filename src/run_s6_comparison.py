"""Run S6A vs S6B comparison backtest on GBP_AUD."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.indicators.technical import compute_all_indicators
from src.backtester.engine import Backtester
from src.strategies_pkg.s6a_ema_bounce import S6A_EMA_Bounce
from src.strategies_pkg.s6b_ema_bounce import S6B_EMA_Bounce

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase1")
PAIR = "GBP_AUD"


def load_data(pair, tf):
    fp = os.path.join(PROCESSED_DIR, f"{pair}_{tf}.csv")
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    return compute_all_indicators(df)


import pandas as pd


def run_variant(label, strat_cls):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    data = load_data(PAIR, "M15")
    htf_data = load_data(PAIR, "H1")
    print(f"  M15 bars: {len(data)}, H1 bars: {len(htf_data)}")

    strategy = strat_cls()
    bt = Backtester(data=data, strategy=strategy, pair=PAIR,
                    starting_equity=100_000.0, htf_data=htf_data)

    t0 = time.time()
    report = bt.run()
    elapsed = time.time() - t0

    trade_log = bt.get_trade_log_df()

    total = report.get("total_trades", 0)
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Trades: {total}")
    print(f"  Win Rate: {report.get('win_rate_pct', 0):.1f}%")
    print(f"  Avg RR: {report.get('avg_rr', 0):.2f}")
    print(f"  Expectancy: {report.get('expectancy_pips', 0):.2f} pips")
    print(f"  Sharpe: {report.get('sharpe_ratio', 0):.2f}")
    print(f"  Max DD: {report.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Profit Factor: {report.get('profit_factor', 0):.2f}")
    print(f"  Total PnL: {report.get('total_pnl_pips', 0):.1f} pips / ${report.get('total_pnl_dollars', 0):,.2f}")
    print(f"  Final Equity: ${report.get('final_equity', 0):,.2f}")
    print(f"  Exits: {report.get('exit_reasons', {})}")

    if total > 0:
        print(f"  Avg Win: {report.get('avg_win_pips', 0):.1f}p | Avg Loss: {report.get('avg_loss_pips', 0):.1f}p")
        print(f"  Best: {report.get('best_trade_pips', 0):.1f}p | Worst: {report.get('worst_trade_pips', 0):.1f}p")
        print(f"  Max Consec Wins: {report.get('max_consecutive_wins', 0)} | Losses: {report.get('max_consecutive_losses', 0)}")

    # Save results
    prefix = label.replace(" ", "_")
    report_path = os.path.join(RESULTS_DIR, f"{prefix}_{PAIR}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    if len(trade_log) > 0:
        log_path = os.path.join(RESULTS_DIR, f"{prefix}_{PAIR}_trades.csv")
        trade_log.to_csv(log_path, index=False)

    return report, trade_log


if __name__ == "__main__":
    print("S6 MOMENTUM FILTER COMPARISON — GBP_AUD")
    print("=" * 60)

    r_a, tl_a = run_variant("S6A", S6A_EMA_Bounce)
    r_b, tl_b = run_variant("S6B", S6B_EMA_Bounce)

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'S6A (3-Check)':>15} {'S6B (2-Check)':>15}")
    print("-" * 55)
    for key in ["total_trades", "win_rate_pct", "avg_rr", "expectancy_pips",
                "sharpe_ratio", "max_drawdown_pct", "profit_factor",
                "total_pnl_pips", "total_pnl_dollars", "final_equity"]:
        va = r_a.get(key, 0)
        vb = r_b.get(key, 0)
        if isinstance(va, float):
            print(f"{key:<25} {va:>15.2f} {vb:>15.2f}")
        else:
            print(f"{key:<25} {va:>15} {vb:>15}")
