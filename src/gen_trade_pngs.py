"""Generate PNG trade charts for visual review.
Shows EMA 50/100/200, Volume, ADX, with 100 bars of pre-entry context."""
import gc, os, sys
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.indicators.technical import compute_all_indicators

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase1")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")

DARK_BG = "#1e1e1e"
DARK_GRID = "#2a2a2a"
TEXT_COLOR = "white"
TEXT_DIM = "#aaaaaa"
GREEN = "#00cc66"
RED = "#ee4444"
CYAN = "#00bcd4"
BLUE = "#4488ff"
ORANGE = "#ff9900"
MAGENTA = "#ff44ff"
YELLOW = "#ffff00"

MC = mpf.make_marketcolors(up="green", down="red",
    edge={"up": "green", "down": "red"},
    wick={"up": "green", "down": "red"}, volume="in")
MPF_STYLE = mpf.make_mpf_style(marketcolors=MC, gridstyle=":", gridcolor=DARK_GRID,
    facecolor=DARK_BG, figcolor=DARK_BG,
    rc={"axes.labelcolor": TEXT_COLOR, "xtick.color": TEXT_COLOR, "ytick.color": TEXT_COLOR})


def gen_trade_png(trade_row, trade_num, total_trades, data, pair, out_dir):
    entry_time = pd.Timestamp(trade_row["timestamp"])
    exit_time = pd.Timestamp(trade_row["exit_time"])
    direction = trade_row["signal_direction"]
    entry_price = trade_row["entry_price"]
    exit_price = trade_row["exit_price"]
    sl_price = trade_row["sl_price"]
    tp1_price = trade_row["tp1_price"]
    pnl_pips = trade_row["pnl_pips"]
    exit_reason = trade_row["exit_reason"]
    pattern = trade_row.get("entry_pattern", "")

    entry_idx = data.index.get_indexer([entry_time], method="nearest")[0]
    exit_idx = data.index.get_indexer([exit_time], method="nearest")[0]

    # 100 bars before entry, 20 bars after exit
    start_idx = max(0, entry_idx - 100)
    end_idx = min(len(data) - 1, exit_idx + 20)
    if end_idx - start_idx < 40:
        end_idx = min(len(data) - 1, start_idx + 40)

    chart_data = data.iloc[start_idx:end_idx + 1].copy()
    if len(chart_data) < 3:
        return None

    addplots = []

    # EMA 50 overlay
    if "ema_50" in chart_data.columns:
        ema50 = chart_data["ema_50"]
        if ema50.notna().any():
            addplots.append(mpf.make_addplot(ema50, color=CYAN, width=1.2, panel=0))

    # EMA 100 overlay
    if "ema_100" in chart_data.columns:
        ema100 = chart_data["ema_100"]
        if ema100.notna().any():
            addplots.append(mpf.make_addplot(ema100, color=ORANGE, width=1.5, panel=0))

    # EMA 200 overlay
    if "ema_200" in chart_data.columns:
        ema200 = chart_data["ema_200"]
        if ema200.notna().any():
            addplots.append(mpf.make_addplot(ema200, color=MAGENTA, width=1.0,
                                              linestyle="--", panel=0))

    # Entry marker
    entry_markers = pd.Series(np.nan, index=chart_data.index)
    if entry_time in chart_data.index:
        entry_markers.at[entry_time] = entry_price
    elif entry_idx >= start_idx and entry_idx <= end_idx:
        entry_markers.iloc[entry_idx - start_idx] = entry_price

    if entry_markers.notna().any():
        mc = "^" if direction == "LONG" else "v"
        mcol = GREEN if direction == "LONG" else RED
        addplots.append(mpf.make_addplot(entry_markers, type="scatter", marker=mc,
            markersize=140, color=mcol, edgecolors="white", linewidths=0.8, panel=0))

    # Exit marker
    exit_markers = pd.Series(np.nan, index=chart_data.index)
    if exit_time in chart_data.index:
        exit_markers.at[exit_time] = exit_price
    elif exit_idx >= start_idx and exit_idx <= end_idx:
        exit_markers.iloc[exit_idx - start_idx] = exit_price

    if exit_markers.notna().any():
        addplots.append(mpf.make_addplot(exit_markers, type="scatter", marker="X",
            markersize=120, color=BLUE, edgecolors="white", linewidths=0.8, panel=0))

    # Volume panel (panel 1)
    if "volume" in chart_data.columns:
        vol = chart_data["volume"]
        if vol.notna().any() and vol.sum() > 0:
            # Color volume bars green/red based on candle direction
            vol_colors = pd.Series(GREEN, index=chart_data.index)
            vol_colors[chart_data["close"] < chart_data["open"]] = RED
            addplots.append(mpf.make_addplot(vol, type="bar", panel=1,
                color=GREEN, width=0.7, ylabel="Volume", alpha=0.6))

    # ADX panel (panel 2)
    if "adx_14" in chart_data.columns:
        adx = chart_data["adx_14"]
        if adx.notna().any():
            addplots.append(mpf.make_addplot(adx, panel=2, color=YELLOW,
                                              width=1.0, ylabel="ADX"))

    pnl_sign = "+" if pnl_pips >= 0 else ""
    pat_label = pattern.replace("ema_bounce_", "").replace("_", " ").title() if pattern else ""
    adx_entry = trade_row.get("adx_at_entry", 0)
    title = (f"Trade #{trade_num} {direction} {pair} | {pat_label} | "
             f"{entry_time.strftime('%Y-%m-%d %H:%M')} | "
             f"{pnl_sign}{pnl_pips:.1f}p ({exit_reason}) | ADX: {adx_entry:.1f}")

    try:
        fig, axes = mpf.plot(chart_data, type="candle", style=MPF_STYLE,
            addplot=addplots if addplots else None, volume=False,
            figsize=(18, 11), tight_layout=False, returnfig=True,
            panel_ratios=(6, 1.2, 1.2))
    except Exception as e:
        print(f"  WARNING: Could not plot trade #{trade_num}: {e}")
        return None

    ax = axes[0]
    ax.set_title(title, color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=12)

    xlim = ax.get_xlim()
    ax.hlines(y=sl_price, xmin=xlim[0], xmax=xlim[1], colors=RED,
              linestyles="dashed", linewidth=0.9, alpha=0.7)
    ax.hlines(y=tp1_price, xmin=xlim[0], xmax=xlim[1], colors=GREEN,
              linestyles="dashed", linewidth=0.8, alpha=0.8)

    ax.text(xlim[1], sl_price, " SL (200 EMA)", color=RED, fontsize=7,
            va="center", fontweight="bold")
    ax.text(xlim[1], tp1_price, " TP1 (4 ATR)", color=GREEN, fontsize=7, va="center")

    # ADX reference lines
    for a in axes:
        if hasattr(a, 'get_ylabel') and a.get_ylabel() == "ADX":
            a.axhline(y=20, color=TEXT_DIM, linewidth=0.5, linestyle="--", alpha=0.5)
            a.axhline(y=25, color=YELLOW, linewidth=0.3, linestyle=":", alpha=0.3)
            a.set_ylim(0, max(60, adx.max() * 1.1) if adx.notna().any() else 60)
            a.tick_params(colors=TEXT_DIM, labelsize=6)
            break

    # Volume axis styling
    for a in axes:
        if hasattr(a, 'get_ylabel') and a.get_ylabel() == "Volume":
            a.tick_params(colors=TEXT_DIM, labelsize=6)
            break

    # Legend
    legend_elements = [
        Line2D([0], [0], color=CYAN, lw=1.2, label="EMA 50"),
        Line2D([0], [0], color=ORANGE, lw=1.5, label="EMA 100"),
        Line2D([0], [0], color=MAGENTA, lw=1.0, linestyle="--", label="EMA 200"),
        Line2D([0], [0], marker="^" if direction == "LONG" else "v",
               color=GREEN if direction == "LONG" else RED, lw=0, markersize=8, label="Entry"),
        Line2D([0], [0], marker="X", color=BLUE, lw=0, markersize=8, label="Exit"),
        Line2D([0], [0], color=RED, lw=0.8, linestyle="dashed", label="SL"),
        Line2D([0], [0], color=GREEN, lw=0.8, linestyle="dashed", label="TP1"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7,
              facecolor=DARK_GRID, edgecolor="#444444", labelcolor=TEXT_COLOR)

    # Info box
    pnl_dollars = trade_row.get("pnl_dollars", 0)
    hold = trade_row.get("hold_time_minutes", 0)
    rsi_entry = trade_row.get("rsi_at_entry", 0)
    lot_size = trade_row.get("lot_size", 0)
    info = (f"Entry: {entry_price:.5f}   Exit: {exit_price:.5f}   "
            f"SL: {sl_price:.5f}   TP1: {tp1_price:.5f}   "
            f"PnL: {pnl_sign}{pnl_pips:.1f}p (${pnl_dollars:,.2f})   "
            f"Hold: {hold}min   RSI: {rsi_entry:.1f}   ADX: {adx_entry:.1f}   "
            f"Lots: {lot_size:,.0f}   Pattern: {pat_label}")

    fig.text(0.05, 0.01, info, color=TEXT_DIM, fontsize=7.5, fontfamily="monospace",
             va="bottom", bbox=dict(boxstyle="round,pad=0.4", facecolor=DARK_GRID,
                                    edgecolor="#444444", alpha=0.9))

    png_path = os.path.join(out_dir, f"S6_{pair}_trade{trade_num:02d}.png")
    fig.savefig(png_path, facecolor=DARK_BG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    return png_path


def main():
    strategy_id = 6
    pair = "GBP_AUD"
    n_trades = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    csv_path = os.path.join(RESULTS_DIR, f"S{strategy_id}_{pair}_trades.csv")
    trade_log = pd.read_csv(csv_path)
    total = len(trade_log)
    print(f"Loaded {total} trades from {csv_path}")

    # Load data
    data = pd.read_csv(os.path.join(PROCESSED_DIR, f"{pair}_M15.csv"),
                        index_col=0, parse_dates=True)
    data.index.name = "timestamp"
    data = compute_all_indicators(data)
    print(f"Loaded {len(data)} M15 bars")

    out_dir = os.path.join(RESULTS_DIR, "trade_pngs")
    os.makedirs(out_dir, exist_ok=True)

    # Pick trades: first n_trades (or all if fewer)
    n = min(n_trades, total)
    print(f"Generating {n} trade PNGs...")

    paths = []
    for i in range(n):
        row = trade_log.iloc[i]
        path = gen_trade_png(row, i + 1, total, data, pair, out_dir)
        if path:
            paths.append(path)
            print(f"  [{i+1}/{n}] {path}")

    print(f"\nDone. Generated {len(paths)} PNGs in {out_dir}")


if __name__ == "__main__":
    main()
