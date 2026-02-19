"""Quick S9 trade analysis for a single pair — runs backtest and analyzes trade log."""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from src.indicators.technical import compute_all_indicators
from src.backtester.engine import Backtester
from src.strategies_pkg.s9_london_session import S9_London_Session

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")

pair = sys.argv[1] if len(sys.argv) > 1 else "GBP_AUD"

# Load and run
fp = os.path.join(PROCESSED_DIR, f"{pair}_H1.csv")
df = pd.read_csv(fp, index_col=0, parse_dates=True)
df.index.name = "timestamp"
df = compute_all_indicators(df)

bt = Backtester(data=df, strategy=S9_London_Session(), pair=pair, starting_equity=100_000.0, htf_data=df.copy())
bt.run()
trades = bt.get_trade_log_df()

print(f"\n{'='*70}")
print(f"S9 TRADE ANALYSIS — {pair} ({len(trades)} trades)")
print(f"{'='*70}")

if len(trades) == 0:
    print("No trades!")
    sys.exit(0)

trades["win"] = trades["pnl_pips"] > 0

# 1. Exit reason breakdown
print("\n1. EXIT REASON BREAKDOWN")
for reason, grp in trades.groupby("exit_reason"):
    n = len(grp)
    wr = grp["win"].mean() * 100
    avg_pnl = grp["pnl_pips"].mean()
    print(f"   {reason:<12} {n:>4} trades ({n/len(trades)*100:5.1f}%) | WR {wr:5.1f}% | Avg PnL {avg_pnl:+7.1f}p")

# 2. Direction
print("\n2. WIN RATE BY DIRECTION")
for d, grp in trades.groupby("signal_direction"):
    print(f"   {d:<6} {len(grp):>4} trades, WR {grp['win'].mean()*100:5.1f}%, Avg PnL {grp['pnl_pips'].mean():+7.1f}p")

# 3. Day of week
print("\n3. WIN RATE BY DAY OF WEEK")
trades["dow"] = pd.to_datetime(trades["timestamp"]).dt.day_name()
for day in ["Monday","Tuesday","Wednesday","Thursday","Friday"]:
    grp = trades[trades["dow"] == day]
    if len(grp) > 0:
        print(f"   {day:<10} {len(grp):>4} trades, WR {grp['win'].mean()*100:5.1f}%, Avg PnL {grp['pnl_pips'].mean():+7.1f}p")

# 4. Hold time
print("\n4. HOLD TIME (minutes)")
winners = trades[trades["win"]]
losers = trades[~trades["win"]]
print(f"   Winners: avg {winners['hold_time_minutes'].mean():.0f}m, median {winners['hold_time_minutes'].median():.0f}m")
print(f"   Losers:  avg {losers['hold_time_minutes'].mean():.0f}m, median {losers['hold_time_minutes'].median():.0f}m")

# 5. Confluence
print("\n5. CONFLUENCE SCORE")
for score, grp in trades.groupby("confluence_score"):
    print(f"   Score {score}: {len(grp):>4} trades, WR {grp['win'].mean()*100:5.1f}%, Avg PnL {grp['pnl_pips'].mean():+7.1f}p")

# 6. Entry hour
print("\n6. ENTRY HOUR")
for h, grp in trades.groupby("hour_of_day"):
    print(f"   Hour {h:>2}: {len(grp):>4} trades, WR {grp['win'].mean()*100:5.1f}%, Avg PnL {grp['pnl_pips'].mean():+7.1f}p")

# 7. RSI zones
print("\n7. RSI AT ENTRY")
bins = [0, 30, 40, 50, 60, 70, 100]
labels = ["<30", "30-40", "40-50", "50-60", "60-70", "70+"]
trades["rsi_bin"] = pd.cut(trades["rsi_at_entry"], bins=bins, labels=labels, include_lowest=True)
for b in labels:
    grp = trades[trades["rsi_bin"] == b]
    if len(grp) > 0:
        print(f"   RSI {b:<6} {len(grp):>4} trades, WR {grp['win'].mean()*100:5.1f}%, Avg PnL {grp['pnl_pips'].mean():+7.1f}p")

# RSI neutral vs momentum
neutral = trades[(trades["rsi_at_entry"] >= 40) & (trades["rsi_at_entry"] <= 60)]
momentum = trades[(trades["rsi_at_entry"] < 40) | (trades["rsi_at_entry"] > 60)]
print(f"   RSI 40-60 (neutral):  {len(neutral):>4} trades, WR {neutral['win'].mean()*100:5.1f}%, Avg PnL {neutral['pnl_pips'].mean():+7.1f}p")
print(f"   RSI outside (momentum): {len(momentum):>4} trades, WR {momentum['win'].mean()*100:5.1f}%, Avg PnL {momentum['pnl_pips'].mean():+7.1f}p")

# 8. ADX buckets
print("\n8. ADX AT ENTRY")
adx_bins = [0, 20, 25, 30, 40, 100]
adx_labels = ["<20", "20-25", "25-30", "30-40", "40+"]
trades["adx_bin"] = pd.cut(trades["adx_at_entry"], bins=adx_bins, labels=adx_labels, include_lowest=True)
for b in adx_labels:
    grp = trades[trades["adx_bin"] == b]
    if len(grp) > 0:
        print(f"   ADX {b:<6} {len(grp):>4} trades, WR {grp['win'].mean()*100:5.1f}%, Avg PnL {grp['pnl_pips'].mean():+7.1f}p")

# 9. EMA50 distance
print("\n9. DISTANCE FROM EMA50 (absolute pips)")
trades["ema_dist_abs"] = trades["distance_from_ema50_pips"].abs()
dist_bins = [0, 20, 40, 60, 100, 500]
dist_labels = ["0-20", "20-40", "40-60", "60-100", "100+"]
trades["dist_bin"] = pd.cut(trades["ema_dist_abs"], bins=dist_bins, labels=dist_labels, include_lowest=True)
for b in dist_labels:
    grp = trades[trades["dist_bin"] == b]
    if len(grp) > 0:
        print(f"   {b:<8} {len(grp):>4} trades, WR {grp['win'].mean()*100:5.1f}%, Avg PnL {grp['pnl_pips'].mean():+7.1f}p")

# 10. Candle body ratio
print("\n10. CANDLE BODY RATIO")
br_bins = [0, 0.3, 0.5, 0.7, 1.01]
br_labels = ["<0.3", "0.3-0.5", "0.5-0.7", "0.7+"]
trades["br_bin"] = pd.cut(trades["candle_body_ratio"], bins=br_bins, labels=br_labels, include_lowest=True)
for b in br_labels:
    grp = trades[trades["br_bin"] == b]
    if len(grp) > 0:
        print(f"   {b:<8} {len(grp):>4} trades, WR {grp['win'].mean()*100:5.1f}%, Avg PnL {grp['pnl_pips'].mean():+7.1f}p")

# 11. Spread as % of avg win
avg_win_pips = winners["pnl_pips"].mean() if len(winners) > 0 else 0
avg_spread = trades["spread_at_entry"].mean()
print(f"\n11. SPREAD IMPACT")
print(f"   Avg spread: {avg_spread:.1f} pips")
print(f"   Avg win:    {avg_win_pips:.1f} pips")
print(f"   Spread/win: {avg_spread/avg_win_pips*100:.1f}%" if avg_win_pips > 0 else "   N/A")

# 12. Yearly breakdown
print("\n12. YEARLY BREAKDOWN")
trades["year"] = pd.to_datetime(trades["timestamp"]).dt.year
for y, grp in trades.groupby("year"):
    gw = grp[grp["pnl_pips"] > 0]["pnl_pips"].sum()
    gl = abs(grp[grp["pnl_pips"] < 0]["pnl_pips"].sum())
    pf = gw / gl if gl > 0 else 0
    print(f"   {y}: {len(grp):>4} trades, WR {grp['win'].mean()*100:5.1f}%, PF {pf:.2f}, PnL {grp['pnl_pips'].sum():+8.1f}p")

# 13. Session (if available)
if "session" in trades.columns:
    print("\n13. SESSION")
    for s, grp in trades.groupby("session"):
        print(f"   {s:<10} {len(grp):>4} trades, WR {grp['win'].mean()*100:5.1f}%, Avg PnL {grp['pnl_pips'].mean():+7.1f}p")

# Save trade log
out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase1", f"S9_{pair}_FULL_trades.csv")
trades.to_csv(out, index=False)
print(f"\nTrade log saved: {out}")
