"""S9 EUR_USD: Baseline vs Light Filter (RSI skip + TP1 1.5x only)."""
import os, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from src.indicators.technical import compute_all_indicators
from src.backtester.engine import Backtester
from src.strategies_pkg.s9_london_session import S9_London_Session

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase1")
os.makedirs(RESULTS_DIR, exist_ok=True)

pair = "EUR_USD"
TRAIN_RATIO = 0.70

fp = os.path.join(PROCESSED_DIR, f"{pair}_H1.csv")
df = pd.read_csv(fp, index_col=0, parse_dates=True)
df.index.name = "timestamp"
df = compute_all_indicators(df)
htf = df.copy()
print(f"Data: {len(df)} H1 bars, {df.index[0].date()} to {df.index[-1].date()}")

split_idx = int(len(df) * TRAIN_RATIO)
train_d, test_d = df.iloc[:split_idx], df.iloc[split_idx:]
train_h, test_h = htf.iloc[:split_idx], htf.iloc[split_idx:]

variants = {
    "BASELINE":     S9_London_Session(),
    "LIGHT_FILTER": S9_London_Session(pair=pair, filtered=True),
}

print(f"\n{'Variant':<16} {'Phase':<6} {'Trades':>6} {'WR%':>6} {'PF':>6} "
      f"{'PnL(p)':>9} {'RR':>5} {'Exp(p)':>7} {'DD%':>7}")
print("-" * 80)

for label, strat in variants.items():
    for phase, d, h in [("FULL", df, htf), ("TRAIN", train_d, train_h), ("TEST", test_d, test_h)]:
        s = S9_London_Session(pair=pair, filtered=(label != "BASELINE"))
        bt = Backtester(data=d, strategy=s, pair=pair, starting_equity=100_000.0, htf_data=h)
        r = bt.run()
        tl = bt.get_trade_log_df()

        n = r.get("total_trades", 0)
        wr = r.get("win_rate_pct", 0)
        pf = r.get("profit_factor", 0)
        pnl = r.get("total_pnl_pips", 0)
        dd = r.get("max_drawdown_pct", 0)
        aw = r.get("avg_win_pips", 0)
        al = r.get("avg_loss_pips", 0)
        rr = aw / al if al > 0 else 0
        exp = pnl / n if n > 0 else 0

        print(f"{label:<16} {phase:<6} {n:>6} {wr:>5.1f}% {pf:>5.2f} "
              f"{pnl:>+8.1f}p {rr:>4.2f} {exp:>+6.2f}p {dd:>6.2f}%")

        if label == "LIGHT_FILTER" and len(tl) > 0:
            tl.to_csv(os.path.join(RESULTS_DIR, f"S9FL_{pair}_{phase}_trades.csv"), index=False)
