"""
Per-trade PDF report generator for fx-quant backtests.

Generates a multi-page PDF per strategy-pair:
  - Page 1: Strategy summary (metrics, equity curve, distributions)
  - Pages 2+: One page per trade (candlestick chart, indicators, info box)

Features:
  - H1 trendline projection on M15 chart (when htf_data available)
  - Confluence annotation arrow explaining trade trigger
  - Dark theme matching chart_trades.py

Usage:
  Standalone: python src/trade_pdf_report.py 1        # regenerate PDFs for strategy 1
  Standalone: python src/trade_pdf_report.py 1 3 5    # multiple strategies
  From code:  generate_trade_pdf(strategy_id, pair, report, trade_log_df, data, htf_data)
"""
import gc
import os
import sys
import json

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — avoids GDI bitmap limits on Windows

import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

plt.rcParams["figure.max_open_warning"] = 0  # Suppress warning; we close every figure

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.indicators.technical import (
    fit_trendline, swing_highs, swing_lows,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase1")

# ---------------------------------------------------------------------------
# Dark theme (matches chart_trades.py)
# ---------------------------------------------------------------------------
DARK_BG = "#1e1e1e"
DARK_GRID = "#2a2a2a"
TEXT_COLOR = "white"
TEXT_DIM = "#aaaaaa"
GREEN = "#00cc66"
RED = "#ee4444"
CYAN = "#00bcd4"
BLUE = "#4488ff"
ORANGE = "#ff9900"

MC = mpf.make_marketcolors(
    up="green", down="red",
    edge={"up": "green", "down": "red"},
    wick={"up": "green", "down": "red"},
    volume="in",
)
MPF_STYLE = mpf.make_mpf_style(
    marketcolors=MC,
    gridstyle=":",
    gridcolor=DARK_GRID,
    facecolor=DARK_BG,
    figcolor=DARK_BG,
    rc={
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
    },
)


# ---------------------------------------------------------------------------
# Trendline overlay helpers
# ---------------------------------------------------------------------------

def _tz_align(ts, target_index):
    """Align timezone of a timestamp to match the target index."""
    ts = pd.Timestamp(ts)
    if target_index.tz is None and ts.tz is not None:
        return ts.tz_localize(None)
    if target_index.tz is not None and ts.tz is None:
        return ts.tz_localize(target_index.tz)
    return ts


def _compute_trendline_overlay(chart_data, entry_time, direction, htf_data):
    """
    Detect the H1 trendline active at entry and project it onto the M15 chart.

    Returns (tl_x, tl_y, tl_info) or None.
      tl_x: list of integer x-coordinates (mplfinance bar indices)
      tl_y: list of trendline prices at those x-coords
      tl_info: dict with slope, intercept, r_squared, touch_count
    """
    if htf_data is None or len(htf_data) < 50:
        return None

    entry_ts = _tz_align(entry_time, htf_data.index)
    n_valid = int(htf_data.index.searchsorted(entry_ts, side="left"))
    if n_valid < 50:
        return None

    start = max(0, n_valid - 200)
    window = htf_data.iloc[start:n_valid]
    offset = start

    # Detect support trendline for SHORT, resistance for LONG
    if direction == "SHORT":
        mask = swing_lows(window, lookback=5)
        indices = np.where(mask.values)[0]
        if len(indices) < 3:
            return None
        recent = indices[-8:]
        prices = window["low"].values[recent]
    else:
        mask = swing_highs(window, lookback=5)
        indices = np.where(mask.values)[0]
        if len(indices) < 3:
            return None
        recent = indices[-8:]
        prices = window["high"].values[recent]

    tl = fit_trendline(recent, prices)
    if tl is None:
        return None

    # Project trendline at each M15 bar in chart_data
    htf_ts = htf_data.index[:n_valid]
    tl_x = []
    tl_y = []

    for i, ts in enumerate(chart_data.index):
        ts_aligned = _tz_align(ts, htf_ts)
        htf_idx = int(htf_ts.searchsorted(ts_aligned, side="right")) - 1
        if htf_idx < 0:
            htf_idx = 0
        window_idx = htf_idx - offset
        tl_price = tl["slope"] * window_idx + tl["intercept"]
        tl_x.append(i)
        tl_y.append(tl_price)

    return tl_x, tl_y, tl


# ---------------------------------------------------------------------------
# Confluence annotation builder
# ---------------------------------------------------------------------------

def _build_confluence_annotation(trade_row, strategy_id):
    """
    Build a 1-line annotation explaining why the trade triggered.

    Returns annotation string for the arrow label.
    """
    direction = trade_row["signal_direction"]
    confluence = int(trade_row.get("confluence_score", 0))
    rsi = trade_row.get("rsi_at_entry", 50)
    macd_hist = trade_row.get("macd_hist_at_entry", 0)
    engulf_type = "Bullish" if direction == "LONG" else "Bearish"

    if strategy_id == 1:
        # S1: Trendline Breakout-Retest
        # Confluence factors: R²>0.90, touches>=4, vol>avg, RSI 40-60, MACD confirms
        pattern = trade_row.get("entry_pattern", "engulfing")
        pattern_label = {"engulfing": "engulfing candle",
                         "pin_bar": "pin bar rejection",
                         "strong_close": "strong directional close"
                         }.get(pattern, pattern or "engulfing candle")

        factors = []
        rsi_neutral = 40 <= rsi <= 60
        macd_confirms = (direction == "LONG" and macd_hist > 0) or \
                        (direction == "SHORT" and macd_hist < 0)

        if rsi_neutral:
            factors.append(f"RSI neutral at {rsi:.0f}")
        if macd_confirms:
            factors.append("MACD confirming momentum")

        detail = ", ".join(factors) if factors else "strong trendline fit"
        return (f"{engulf_type} {pattern_label} at H1 trendline retest with "
                f"{detail} ({confluence}/5 confluence)")

    elif strategy_id == 3:
        return (f"{engulf_type} breakout at key S/R level with "
                f"{confluence}/5 confluence")

    elif strategy_id == 4:
        return (f"EMA ribbon aligned {direction.lower()} with "
                f"RSI at {rsi:.0f} ({confluence}/5 confluence)")

    elif strategy_id == 5:
        return (f"Momentum exhaustion {direction.lower()} signal with "
                f"RSI at {rsi:.0f} ({confluence}/5 confluence)")

    # Generic fallback
    return f"{direction} signal with {confluence} confluence factors"


# ---------------------------------------------------------------------------
# Summary page
# ---------------------------------------------------------------------------

def _draw_summary_page(pdf, report, trade_log_df):
    """Draw strategy summary as the first page of the PDF."""
    fig = plt.figure(figsize=(11, 8.5), facecolor=DARK_BG)

    # Title
    strat_name = report.get("strategy_name", "Unknown")
    pair = report.get("pair", "")
    fig.suptitle(f"{strat_name}  —  {pair}", color=TEXT_COLOR,
                 fontsize=16, fontweight="bold", y=0.97)

    total = report.get("total_trades", 0)
    if total == 0:
        fig.text(0.5, 0.5, "No trades generated", color=TEXT_DIM,
                 fontsize=14, ha="center", va="center")
        pdf.savefig(fig, facecolor=DARK_BG)
        plt.close(fig)
        return

    # --- Metrics table (left side) ---
    ax_table = fig.add_axes([0.04, 0.42, 0.42, 0.50])
    ax_table.set_facecolor(DARK_BG)
    ax_table.axis("off")

    metrics = [
        ("Total Trades", f"{total}"),
        ("Win Rate", f"{report.get('win_rate_pct', 0):.1f}%"),
        ("Avg R:R", f"{report.get('avg_rr', 0):.2f}"),
        ("Profit Factor", f"{report.get('profit_factor', 0):.2f}"),
        ("Sharpe Ratio", f"{report.get('sharpe_ratio', 0):.2f}"),
        ("Max Drawdown", f"{report.get('max_drawdown_pct', 0):.2f}%"),
        ("Expectancy", f"{report.get('expectancy_pips', 0):.2f} pips"),
        ("Total PnL", f"{report.get('total_pnl_pips', 0):.1f} pips / ${report.get('total_pnl_dollars', 0):,.2f}"),
        ("Avg Win", f"{report.get('avg_win_pips', 0):.1f} pips"),
        ("Avg Loss", f"{report.get('avg_loss_pips', 0):.1f} pips"),
        ("Best Trade", f"{report.get('best_trade_pips', 0):.1f} pips"),
        ("Worst Trade", f"{report.get('worst_trade_pips', 0):.1f} pips"),
        ("Avg Hold Time", f"{report.get('avg_hold_time_minutes', 0):.0f} min"),
        ("Max Consec Wins", f"{report.get('max_consecutive_wins', 0)}"),
        ("Max Consec Losses", f"{report.get('max_consecutive_losses', 0)}"),
        ("Final Equity", f"${report.get('final_equity', 0):,.2f}"),
    ]

    y_pos = 1.0
    for label, value in metrics:
        ax_table.text(0.0, y_pos, label, color=TEXT_DIM, fontsize=8.5,
                      fontfamily="monospace", va="top")
        ax_table.text(0.65, y_pos, value, color=TEXT_COLOR, fontsize=8.5,
                      fontfamily="monospace", va="top", fontweight="bold")
        y_pos -= 0.065

    # --- Equity curve (top right) ---
    if len(trade_log_df) > 0:
        ax_eq = fig.add_axes([0.55, 0.60, 0.40, 0.28])
        ax_eq.set_facecolor(DARK_BG)

        starting_eq = report.get("starting_equity", 100000)
        cumulative = starting_eq + trade_log_df["pnl_dollars"].cumsum()
        equity_series = pd.concat([pd.Series([starting_eq]), cumulative]).reset_index(drop=True)

        ax_eq.plot(equity_series, color=CYAN, linewidth=1.2)
        ax_eq.axhline(y=starting_eq, color=TEXT_DIM, linewidth=0.5, linestyle="--")
        ax_eq.fill_between(range(len(equity_series)), starting_eq, equity_series,
                           where=equity_series >= starting_eq, alpha=0.15, color=GREEN)
        ax_eq.fill_between(range(len(equity_series)), starting_eq, equity_series,
                           where=equity_series < starting_eq, alpha=0.15, color=RED)
        ax_eq.set_title("Equity Curve", color=TEXT_COLOR, fontsize=9)
        ax_eq.tick_params(colors=TEXT_DIM, labelsize=7)
        for spine in ax_eq.spines.values():
            spine.set_color(DARK_GRID)

    # --- Win/Loss distribution (bottom left) ---
    if len(trade_log_df) > 0:
        ax_dist = fig.add_axes([0.06, 0.07, 0.38, 0.28])
        ax_dist.set_facecolor(DARK_BG)

        pnl_pips = trade_log_df["pnl_pips"]
        win_pnl = pnl_pips[pnl_pips > 0]
        loss_pnl = pnl_pips[pnl_pips <= 0]

        bins = np.linspace(pnl_pips.min(), pnl_pips.max(), min(20, max(5, total // 2)))
        if len(win_pnl) > 0:
            ax_dist.hist(win_pnl, bins=bins, color=GREEN, alpha=0.7, label="Wins")
        if len(loss_pnl) > 0:
            ax_dist.hist(loss_pnl, bins=bins, color=RED, alpha=0.7, label="Losses")
        ax_dist.axvline(x=0, color=TEXT_DIM, linewidth=0.5, linestyle="--")
        ax_dist.set_title("PnL Distribution (pips)", color=TEXT_COLOR, fontsize=9)
        ax_dist.legend(fontsize=7, facecolor=DARK_GRID, edgecolor=DARK_GRID,
                       labelcolor=TEXT_COLOR)
        ax_dist.tick_params(colors=TEXT_DIM, labelsize=7)
        for spine in ax_dist.spines.values():
            spine.set_color(DARK_GRID)

    # --- Exit reasons breakdown (bottom right) ---
    exit_reasons = report.get("exit_reasons", {})
    if exit_reasons:
        ax_exit = fig.add_axes([0.55, 0.07, 0.40, 0.28])
        ax_exit.set_facecolor(DARK_BG)

        labels = list(exit_reasons.keys())
        counts = list(exit_reasons.values())
        bar_colors = []
        for label in labels:
            if "TP3" in label:
                bar_colors.append(GREEN)
            elif "SL" in label and "TP" not in label:
                bar_colors.append(RED)
            elif "TIME" in label or "END" in label:
                bar_colors.append(TEXT_DIM)
            else:
                bar_colors.append(CYAN)

        bars = ax_exit.barh(labels, counts, color=bar_colors, height=0.6)
        ax_exit.set_title("Exit Reasons", color=TEXT_COLOR, fontsize=9)
        ax_exit.tick_params(colors=TEXT_DIM, labelsize=7)
        for spine in ax_exit.spines.values():
            spine.set_color(DARK_GRID)
        # Value labels on bars
        for bar, count in zip(bars, counts):
            ax_exit.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                         str(count), color=TEXT_COLOR, fontsize=7, va="center")

    pdf.savefig(fig, facecolor=DARK_BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-trade page
# ---------------------------------------------------------------------------

def _draw_trade_page(pdf, trade_row, trade_num, total_trades, data, pair,
                     htf_data=None, strategy_id=0):
    """Draw a single trade page with candlestick chart, RSI, and info box."""
    entry_time = pd.Timestamp(trade_row["timestamp"])
    exit_time = pd.Timestamp(trade_row["exit_time"])
    direction = trade_row["signal_direction"]
    entry_price = trade_row["entry_price"]
    exit_price = trade_row["exit_price"]
    sl_price = trade_row["sl_price"]
    tp1_price = trade_row["tp1_price"]
    tp2_price = trade_row["tp2_price"]
    tp3_price = trade_row["tp3_price"]
    pnl_pips = trade_row["pnl_pips"]
    exit_reason = trade_row["exit_reason"]

    # Find bar indices for entry and exit
    try:
        entry_idx = data.index.get_indexer([entry_time], method="nearest")[0]
    except Exception:
        entry_idx = 0
    try:
        exit_idx = data.index.get_indexer([exit_time], method="nearest")[0]
    except Exception:
        exit_idx = min(entry_idx + 50, len(data) - 1)

    # Window: 50 bars before entry to exit + 10 bars after
    start_idx = max(0, entry_idx - 50)
    end_idx = min(len(data) - 1, exit_idx + 10)

    # Ensure minimum chart width
    if end_idx - start_idx < 20:
        end_idx = min(len(data) - 1, start_idx + 20)

    chart_data = data.iloc[start_idx:end_idx + 1].copy()

    if len(chart_data) < 3:
        return  # skip if not enough data

    # --- Build addplots ---
    addplots = []

    # EMA 50 overlay
    if "ema_50" in chart_data.columns:
        ema_50 = chart_data["ema_50"]
        if ema_50.notna().any():
            addplots.append(mpf.make_addplot(ema_50, color=CYAN, width=1.2,
                                             panel=0))

    # Entry marker
    entry_markers = pd.Series(np.nan, index=chart_data.index)
    entry_chart_x = None  # track entry x-coord for annotation
    if entry_time in chart_data.index:
        entry_markers.at[entry_time] = entry_price
        entry_chart_x = chart_data.index.get_loc(entry_time)
    elif entry_idx >= start_idx and entry_idx <= end_idx:
        entry_markers.iloc[entry_idx - start_idx] = entry_price
        entry_chart_x = entry_idx - start_idx

    if entry_markers.notna().any():
        marker_char = "^" if direction == "LONG" else "v"
        marker_color = GREEN if direction == "LONG" else RED
        addplots.append(mpf.make_addplot(
            entry_markers, type="scatter", marker=marker_char, markersize=120,
            color=marker_color, edgecolors="white", linewidths=0.8, panel=0))

    # Exit marker
    exit_markers = pd.Series(np.nan, index=chart_data.index)
    if exit_time in chart_data.index:
        exit_markers.at[exit_time] = exit_price
    elif exit_idx >= start_idx and exit_idx <= end_idx:
        exit_markers.iloc[exit_idx - start_idx] = exit_price

    if exit_markers.notna().any():
        addplots.append(mpf.make_addplot(
            exit_markers, type="scatter", marker="X", markersize=100,
            color=BLUE, edgecolors="white", linewidths=0.8, panel=0))

    # RSI subplot
    if "rsi_14" in chart_data.columns:
        rsi_data = chart_data["rsi_14"]
        if rsi_data.notna().any():
            addplots.append(mpf.make_addplot(rsi_data, panel=2, color="yellow",
                                             width=0.9, ylabel="RSI"))

    # --- Plot ---
    pnl_sign = "+" if pnl_pips >= 0 else ""
    title = (f"Trade #{trade_num}/{total_trades}  —  "
             f"{direction} {pair} @ {entry_price:.5f}  |  "
             f"{entry_time.strftime('%Y-%m-%d %H:%M')}  |  "
             f"{pnl_sign}{pnl_pips:.1f} pips ({exit_reason})")

    try:
        fig, axes = mpf.plot(
            chart_data, type="candle", style=MPF_STYLE,
            addplot=addplots if addplots else None,
            volume=False,
            figsize=(11, 8.5),
            tight_layout=False,
            returnfig=True,
            panel_ratios=(7, 0.3, 1.5),
        )
    except Exception:
        # Fallback without RSI panel if it fails
        addplots_filtered = []
        for ap in addplots:
            try:
                if ap.get("panel", 0) != 2:
                    addplots_filtered.append(ap)
            except Exception:
                addplots_filtered.append(ap)
        try:
            fig, axes = mpf.plot(
                chart_data, type="candle", style=MPF_STYLE,
                addplot=addplots_filtered if addplots_filtered else None,
                volume=False,
                figsize=(11, 8.5),
                tight_layout=False,
                returnfig=True,
            )
        except Exception as e:
            print(f"    WARNING: Could not plot trade #{trade_num}: {e}")
            return

    ax_main = axes[0]
    ax_main.set_title(title, color=TEXT_COLOR, fontsize=10, pad=10)

    # --- Trendline overlay ---
    has_trendline = False
    tl_result = _compute_trendline_overlay(chart_data, entry_time, direction,
                                           htf_data)
    if tl_result is not None:
        tl_x, tl_y, tl_info = tl_result
        ax_main.plot(tl_x, tl_y, color=ORANGE, linewidth=1.8,
                     linestyle="--", alpha=0.85, zorder=5)
        # Label the trendline
        label_side = "support" if direction == "SHORT" else "resistance"
        r2 = tl_info.get("r_squared", 0)
        touches = tl_info.get("touch_count", 0)
        ax_main.text(tl_x[-1] + 0.5, tl_y[-1],
                     f" H1 {label_side}\n R\u00b2={r2:.2f}, {touches}T",
                     color=ORANGE, fontsize=6.5, va="center", fontweight="bold")
        has_trendline = True

    # --- Confluence annotation arrow ---
    if entry_chart_x is not None:
        annotation_text = _build_confluence_annotation(trade_row, strategy_id)

        # Position the arrow text above for LONG, below for SHORT
        atr_val = trade_row.get("atr_at_entry", 0)
        if atr_val <= 0:
            atr_val = abs(tp1_price - entry_price) * 0.3

        if direction == "LONG":
            text_y = entry_price - 2.5 * atr_val
            arrow_y = entry_price - 0.3 * atr_val
        else:
            text_y = entry_price + 2.5 * atr_val
            arrow_y = entry_price + 0.3 * atr_val

        # Offset text horizontally to avoid candle overlap
        text_x = max(0, entry_chart_x - 15)

        ax_main.annotate(
            annotation_text,
            xy=(entry_chart_x, arrow_y),
            xytext=(text_x, text_y),
            fontsize=7, color=ORANGE, fontweight="bold",
            arrowprops=dict(
                arrowstyle="->",
                color=ORANGE,
                linewidth=1.5,
                connectionstyle="arc3,rad=0.2",
            ),
            bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_BG,
                      edgecolor=ORANGE, alpha=0.9),
            zorder=10,
        )

    # Draw horizontal SL/TP lines
    xlim = ax_main.get_xlim()
    ax_main.hlines(y=sl_price, xmin=xlim[0], xmax=xlim[1],
                   colors=RED, linestyles="dashed", linewidth=0.8, alpha=0.7)
    ax_main.hlines(y=tp1_price, xmin=xlim[0], xmax=xlim[1],
                   colors=GREEN, linestyles="dashed", linewidth=0.7, alpha=0.8)
    ax_main.hlines(y=tp2_price, xmin=xlim[0], xmax=xlim[1],
                   colors=GREEN, linestyles="dashed", linewidth=0.7, alpha=0.6)
    ax_main.hlines(y=tp3_price, xmin=xlim[0], xmax=xlim[1],
                   colors=GREEN, linestyles="dashed", linewidth=0.7, alpha=0.4)

    # Label SL/TP on right edge
    ax_main.text(xlim[1], sl_price, " SL", color=RED, fontsize=7,
                 va="center", fontweight="bold")
    ax_main.text(xlim[1], tp1_price, " TP1", color=GREEN, fontsize=7,
                 va="center", alpha=0.8)
    ax_main.text(xlim[1], tp2_price, " TP2", color=GREEN, fontsize=7,
                 va="center", alpha=0.6)
    ax_main.text(xlim[1], tp3_price, " TP3", color=GREEN, fontsize=7,
                 va="center", alpha=0.4)

    # RSI reference lines (40/60 zone)
    for ax in axes:
        if hasattr(ax, 'get_ylabel') and ax.get_ylabel() == "RSI":
            ax.axhline(y=40, color=TEXT_DIM, linewidth=0.5, linestyle="--", alpha=0.5)
            ax.axhline(y=60, color=TEXT_DIM, linewidth=0.5, linestyle="--", alpha=0.5)
            ax.axhspan(40, 60, alpha=0.05, color="yellow")
            ax.set_ylim(0, 100)
            break

    # --- Info box at bottom ---
    hold_time = trade_row.get("hold_time_minutes", 0)
    pnl_dollars = trade_row.get("pnl_dollars", 0)
    confluence = trade_row.get("confluence_score", 0)
    session = trade_row.get("session", "")
    atr_entry = trade_row.get("atr_at_entry", 0)
    rsi_entry = trade_row.get("rsi_at_entry", 0)
    macd_hist = trade_row.get("macd_hist_at_entry", 0)
    lot_size = trade_row.get("lot_size", 0)

    info_text = (
        f"Dir: {direction}   Entry: {entry_price:.5f}   Exit: {exit_price:.5f}   "
        f"SL: {sl_price:.5f}   TP1: {tp1_price:.5f}   TP2: {tp2_price:.5f}   TP3: {tp3_price:.5f}\n"
        f"PnL: {pnl_sign}{pnl_pips:.1f} pips (${pnl_dollars:,.2f})   "
        f"Hold: {hold_time} min   Lots: {lot_size:,.0f}   "
        f"Confluence: {confluence}   Session: {session}\n"
        f"ATR: {atr_entry:.5f}   RSI: {rsi_entry:.1f}   MACD Hist: {macd_hist:.6f}   "
        f"Exit: {exit_reason}"
    )

    fig.text(0.05, 0.02, info_text, color=TEXT_DIM, fontsize=7.5,
             fontfamily="monospace", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=DARK_GRID,
                       edgecolor="#444444", alpha=0.9))

    # Legend
    legend_elements = [
        Line2D([0], [0], color=CYAN, lw=1.2, label="EMA 50"),
        Line2D([0], [0], marker="^" if direction == "LONG" else "v",
               color=GREEN if direction == "LONG" else RED,
               lw=0, markersize=8, label="Entry"),
        Line2D([0], [0], marker="X", color=BLUE, lw=0, markersize=8, label="Exit"),
        Line2D([0], [0], color=RED, lw=0.8, linestyle="dashed", label="SL"),
        Line2D([0], [0], color=GREEN, lw=0.8, linestyle="dashed", label="TP"),
    ]
    if has_trendline:
        legend_elements.append(
            Line2D([0], [0], color=ORANGE, lw=1.8, linestyle="--",
                   label="H1 Trendline"))
    ax_main.legend(handles=legend_elements, loc="upper left", fontsize=7,
                   facecolor=DARK_GRID, edgecolor="#444444", labelcolor=TEXT_COLOR)

    pdf.savefig(fig, facecolor=DARK_BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_trade_pdf(strategy_id, pair, report, trade_log_df, data,
                       htf_data=None):
    """
    Generate a multi-page PDF with summary + per-trade charts.

    Args:
        strategy_id: Strategy number (1-5).
        pair: Currency pair string e.g. 'GBP_AUD'.
        report: Performance report dict from backtester.
        trade_log_df: DataFrame of trade records (from CSV or backtester).
        data: Primary timeframe OHLCV DataFrame with indicators.
        htf_data: Higher timeframe data (optional, for trendline overlay).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    prefix = f"S{strategy_id}_{pair}"
    pdf_path = os.path.join(RESULTS_DIR, f"{prefix}_trades.pdf")

    total = report.get("total_trades", 0)
    print(f"    Generating PDF: {prefix} ({total} trades)...", end=" ")

    with PdfPages(pdf_path) as pdf:
        # Page 1: Summary
        _draw_summary_page(pdf, report, trade_log_df)

        # Pages 2+: One per trade
        if len(trade_log_df) > 0:
            for idx, (_, trade_row) in enumerate(trade_log_df.iterrows(), 1):
                _draw_trade_page(pdf, trade_row, idx, total, data, pair,
                                 htf_data=htf_data, strategy_id=strategy_id)
                # Periodic cleanup to avoid GDI/memory exhaustion on large runs
                if idx % 10 == 0:
                    plt.close("all")
                    gc.collect()

    print(f"saved to {pdf_path}")
    return pdf_path


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main():
    """Regenerate PDFs from saved CSV/JSON files without re-running backtests."""
    from src.indicators.technical import compute_all_indicators
    from src.strategies_pkg import STRATEGIES, STRATEGY_PAIRS, STRATEGY_TIMEFRAMES

    PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 "data", "processed")

    selected = None
    if len(sys.argv) > 1:
        selected = [int(x) for x in sys.argv[1:]]

    print("=" * 60)
    print("PDF REPORT GENERATOR (standalone)")
    print("=" * 60)

    for strategy_id in sorted(STRATEGIES.keys()):
        if selected and strategy_id not in selected:
            continue

        pairs = STRATEGY_PAIRS[strategy_id]
        tf_config = STRATEGY_TIMEFRAMES[strategy_id]
        primary_tf = tf_config["primary"]
        filter_tf = tf_config["filter"]

        for pair in pairs:
            prefix = f"S{strategy_id}_{pair}"

            # Load report JSON
            report_path = os.path.join(RESULTS_DIR, f"{prefix}_report.json")
            if not os.path.exists(report_path):
                print(f"  SKIP {prefix}: no report JSON")
                continue
            with open(report_path) as f:
                report = json.load(f)

            # Load trade log CSV
            csv_path = os.path.join(RESULTS_DIR, f"{prefix}_trades.csv")
            if not os.path.exists(csv_path):
                print(f"  SKIP {prefix}: no trade CSV")
                continue
            trade_log_df = pd.read_csv(csv_path)

            if len(trade_log_df) == 0:
                print(f"  SKIP {prefix}: no trades")
                continue

            # Load primary data with indicators
            data_path = os.path.join(PROCESSED_DIR, f"{pair}_{primary_tf}.csv")
            if not os.path.exists(data_path):
                print(f"  SKIP {prefix}: no {primary_tf} data file")
                continue
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            data.index.name = "timestamp"
            data = compute_all_indicators(data)

            # Load HTF data if available
            htf_data = None
            if filter_tf:
                htf_path = os.path.join(PROCESSED_DIR, f"{pair}_{filter_tf}.csv")
                if os.path.exists(htf_path):
                    htf_data = pd.read_csv(htf_path, index_col=0, parse_dates=True)
                    htf_data.index.name = "timestamp"
                    htf_data = compute_all_indicators(htf_data)

            generate_trade_pdf(strategy_id, pair, report, trade_log_df,
                               data, htf_data)

    print("\nDone.")


if __name__ == "__main__":
    main()
