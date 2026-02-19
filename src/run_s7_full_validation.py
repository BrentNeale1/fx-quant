"""S7 Tight — Full dataset validation for GBP_USD and EUR_USD."""
import os, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from src.indicators.technical import compute_all_indicators
from src.backtester.engine import Backtester
from src.strategies_pkg.s7_liquidity_sweep import S7_Liquidity_Sweep

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase1")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_RATIO = 0.70

for pair in ["GBP_USD", "EUR_USD"]:
    print(f"\n{'#' * 70}")
    print(f"# S7 Tight — {pair} FULL DATASET")
    print(f"{'#' * 70}")

    fp = os.path.join(PROCESSED_DIR, f"{pair}_H1.csv")
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    df = compute_all_indicators(df)
    htf = df.copy()
    print(f"  Data: {len(df)} H1 bars, {df.index[0].date()} to {df.index[-1].date()}")

    split_idx = int(len(df) * TRAIN_RATIO)
    splits = {
        "FULL": (df, htf),
        "TRAIN": (df.iloc[:split_idx], htf.iloc[:split_idx]),
        "TEST": (df.iloc[split_idx:], htf.iloc[split_idx:]),
    }

    for phase, (d, h) in splits.items():
        t0 = time.time()
        bt = Backtester(data=d, strategy=S7_Liquidity_Sweep(), pair=pair,
                        starting_equity=100_000.0, htf_data=h)
        r = bt.run()
        tl = bt.get_trade_log_df()
        elapsed = time.time() - t0

        n = r.get("total_trades", 0)
        wr = r.get("win_rate_pct", 0)
        pf = r.get("profit_factor", 0)
        pnl = r.get("total_pnl_pips", 0)
        pnl_d = r.get("total_pnl_dollars", 0)
        dd = r.get("max_drawdown_pct", 0)
        aw = r.get("avg_win_pips", 0)
        al = r.get("avg_loss_pips", 0)
        rr = aw / al if al > 0 else 0
        exp = pnl / n if n > 0 else 0

        print(f"  {phase:>5}: {n:>3} trades | WR {wr:>5.1f}% | PF {pf:>5.2f} | "
              f"PnL {pnl:>+8.1f}p (${pnl_d:>+10,.2f}) | DD {dd:>6.2f}% | "
              f"AvgW {aw:>5.1f} / AvgL {al:>5.1f} | RR {rr:>4.2f} | "
              f"Exp {exp:>+6.2f}p ({elapsed:.0f}s)")

        if len(tl) > 0:
            tl.to_csv(os.path.join(RESULTS_DIR, f"S7T_{pair}_FULL_{phase}_trades.csv"), index=False)
