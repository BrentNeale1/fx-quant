"""Run S4-D, S4-E, S4-F comparison on GBP_AUD, EUR_AUD, GBP_JPY.
Runs all pairs for each variant (no early stopping)."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from src.indicators.technical import compute_all_indicators
from src.backtester.engine import Backtester

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase1")
PAIRS = ["GBP_AUD", "EUR_AUD", "GBP_JPY"]


def load_data(pair, tf):
    fp = os.path.join(PROCESSED_DIR, f"{pair}_{tf}.csv")
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    return compute_all_indicators(df)


def run_variant(label, strat_cls):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    all_reports = []
    all_trades = []
    total_trade_count = 0

    for pair in PAIRS:
        data = load_data(pair, "M15")
        htf_data = load_data(pair, "H1")
        print(f"  {pair}: M15={len(data)}, H1={len(htf_data)}", end=" ")

        strategy = strat_cls()
        bt = Backtester(data=data, strategy=strategy, pair=pair,
                        starting_equity=100_000.0, htf_data=htf_data)

        t0 = time.time()
        report = bt.run()
        elapsed = time.time() - t0

        trade_log = bt.get_trade_log_df()
        n = report.get("total_trades", 0)
        total_trade_count += n
        print(f"-> {n} trades ({elapsed:.0f}s)")

        all_reports.append(report)
        if len(trade_log) > 0:
            all_trades.append(trade_log)

    # Aggregate
    total_trades = sum(r.get("total_trades", 0) for r in all_reports)
    if total_trades == 0:
        print(f"  NO TRADES across any pair!")
        return {"total_trades": 0}

    total_wins = sum(r.get("total_trades", 0) * r.get("win_rate_pct", 0) / 100 for r in all_reports)
    total_pnl_pips = sum(r.get("total_pnl_pips", 0) for r in all_reports)
    total_pnl_dollars = sum(r.get("total_pnl_dollars", 0) for r in all_reports)

    # Combine trade logs for PF calculation
    combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if len(combined) > 0:
        gross_profit = combined.loc[combined["pnl_pips"] > 0, "pnl_pips"].sum()
        gross_loss = abs(combined.loc[combined["pnl_pips"] < 0, "pnl_pips"].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
        avg_win = combined.loc[combined["pnl_pips"] > 0, "pnl_pips"].mean() if (combined["pnl_pips"] > 0).any() else 0
        avg_loss = abs(combined.loc[combined["pnl_pips"] < 0, "pnl_pips"].mean()) if (combined["pnl_pips"] < 0).any() else 0
        avg_rr = avg_win / avg_loss if avg_loss > 0 else 0
        max_dd = min(r.get("max_drawdown_pct", 0) for r in all_reports)
    else:
        pf = avg_win = avg_loss = avg_rr = 0
        max_dd = 0

    win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
    expectancy = total_pnl_pips / total_trades if total_trades > 0 else 0

    print(f"\n  --- {label} AGGREGATE ---")
    print(f"  Trades: {total_trades}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Avg RR: {avg_rr:.2f}")
    print(f"  Expectancy: {expectancy:.2f} pips")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Worst Max DD: {max_dd:.2f}%")
    print(f"  Total PnL: {total_pnl_pips:.1f} pips / ${total_pnl_dollars:,.2f}")
    print(f"  Avg Win: {avg_win:.1f}p | Avg Loss: {avg_loss:.1f}p")

    # Per-pair breakdown
    for i, pair in enumerate(PAIRS):
        if i < len(all_reports):
            r = all_reports[i]
            n = r.get("total_trades", 0)
            wr = r.get("win_rate_pct", 0)
            pfp = r.get("profit_factor", 0)
            pnl = r.get("total_pnl_pips", 0)
            print(f"    {pair}: {n} trades, WR {wr:.1f}%, PF {pfp:.2f}, PnL {pnl:.1f}p")

    return {
        "total_trades": total_trades, "win_rate": win_rate, "avg_rr": avg_rr,
        "expectancy": expectancy, "profit_factor": pf, "max_dd": max_dd,
        "total_pnl_pips": total_pnl_pips, "total_pnl_dollars": total_pnl_dollars,
        "avg_win": avg_win, "avg_loss": avg_loss,
    }


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "all"

    results = {}

    if variant in ("D", "all"):
        from src.strategies_pkg.s4d_ema_ribbon import S4D_EMA_Ribbon
        results["S4-D"] = run_variant("S4-D (Volume+ADX)", S4D_EMA_Ribbon)

    if variant in ("E", "all"):
        from src.strategies_pkg.s4e_ema_ribbon import S4E_EMA_Ribbon
        results["S4-E"] = run_variant("S4-E (Compression Quality)", S4E_EMA_Ribbon)

    if variant in ("F", "all"):
        from src.strategies_pkg.s4f_ema_ribbon import S4F_EMA_Ribbon
        results["S4-F"] = run_variant("S4-F (Trend Context)", S4F_EMA_Ribbon)

    if len(results) > 1:
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        print(f"{'Metric':<20}", end="")
        for k in results:
            print(f"{k:>18}", end="")
        print()
        print("-" * (20 + 18 * len(results)))
        for metric in ["total_trades", "win_rate", "avg_rr", "expectancy",
                        "profit_factor", "max_dd", "total_pnl_pips", "total_pnl_dollars",
                        "avg_win", "avg_loss"]:
            print(f"{metric:<20}", end="")
            for k in results:
                v = results[k].get(metric, 0)
                if isinstance(v, float):
                    print(f"{v:>18.2f}", end="")
                else:
                    print(f"{v:>18}", end="")
            print()
