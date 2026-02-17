# src/chart_trades.py
"""
Generate candlestick chart with trade entry arrows, SMA 50, and pivot levels.
Usage: python src/chart_trades.py [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--granularity M5|M15]
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from supabase import create_client

from config_loader import load_config, get_project_root
from backtester import fetch_candles_from_supabase
from data_engine import add_pivot_points, add_sma, add_atr, detect_engulfing


def load_trade_log(instrument, granularity):
    """Load the backtest trade CSV for a given instrument/granularity."""
    root = get_project_root()
    csv_path = root / "logs" / f"backtest_trades_{instrument}_{granularity}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Trade log not found: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["time"])
    return df


def build_chart(instrument, granularity, start_date=None, end_date=None):
    """Build candlestick chart with trade arrows, SMA 50, and pivot levels."""
    cfg = load_config()

    # --- Fetch candle data from Supabase ---
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    sb = create_client(supabase_url, supabase_key)
    table = cfg.get("supabase", {}).get("table", "fx_candles")

    df = fetch_candles_from_supabase(instrument, granularity, sb, table)
    if df.empty:
        print("No candle data found.")
        return

    # Add indicators
    df = add_sma(df, period=50)
    df = add_atr(df, period=14)
    df = add_pivot_points(df)

    # --- Filter date range ---
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date, tz="UTC")]

    if df.empty:
        print("No data in the specified date range.")
        return

    # --- Load trade log and filter to date range ---
    trades_df = load_trade_log(instrument, granularity)
    entries = trades_df[trades_df["side"].isin(["BUY", "SELL_SHORT"])].copy()
    entries["time"] = pd.to_datetime(entries["time"], utc=True)

    if start_date:
        entries = entries[entries["time"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        entries = entries[entries["time"] <= pd.Timestamp(end_date, tz="UTC")]

    # Separate longs and shorts
    buys = entries[entries["side"] == "BUY"]
    sells = entries[entries["side"] == "SELL_SHORT"]

    # --- Build mplfinance addplots ---
    addplots = []

    # SMA 50 line
    if "sma_50" in df.columns:
        addplots.append(mpf.make_addplot(df["sma_50"], color="cyan", width=1.5,
                                         label="SMA 50"))

    # Pivot levels
    pivot_styles = {
        "r3": ("red", 0.3, "dotted"),
        "r2": ("red", 0.5, "dashed"),
        "r1": ("red", 0.8, "dashed"),
        "pivot": ("blue", 1.0, "solid"),
        "s1": ("green", 0.8, "dashed"),
        "s2": ("green", 0.5, "dashed"),
        "s3": ("green", 0.3, "dotted"),
    }

    for level, (color, alpha, linestyle) in pivot_styles.items():
        if level in df.columns:
            addplots.append(mpf.make_addplot(
                df[level], color=color, alpha=alpha, width=0.8,
                linestyle=linestyle, label=level.upper(),
            ))

    # Buy arrows (up triangles at entry price)
    buy_markers = pd.Series(np.nan, index=df.index)
    for _, row in buys.iterrows():
        t = row["time"]
        if t in df.index:
            buy_markers.at[t] = row["price"]

    if buy_markers.notna().any():
        addplots.append(mpf.make_addplot(
            buy_markers, type="scatter", marker="^", markersize=90,
            color="lime", edgecolors="black", linewidths=0.5,
        ))

    # Sell arrows (down triangles at entry price)
    sell_markers = pd.Series(np.nan, index=df.index)
    for _, row in sells.iterrows():
        t = row["time"]
        if t in df.index:
            sell_markers.at[t] = row["price"]

    if sell_markers.notna().any():
        addplots.append(mpf.make_addplot(
            sell_markers, type="scatter", marker="v", markersize=90,
            color="red", edgecolors="black", linewidths=0.5,
        ))

    # --- Style ---
    mc = mpf.make_marketcolors(
        up="green", down="red",
        edge={"up": "green", "down": "red"},
        wick={"up": "green", "down": "red"},
        volume="in",
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle=":",
        gridcolor="#2a2a2a",
        facecolor="#1e1e1e",
        figcolor="#1e1e1e",
        rc={
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
        },
    )

    # --- Plot ---
    n_bars = len(df)
    width_inches = max(16, n_bars * 0.02)
    width_inches = min(width_inches, 48)  # cap at 48"

    fig, axes = mpf.plot(
        df, type="candle", style=style,
        addplot=addplots if addplots else None,
        volume=False,
        figsize=(width_inches, 10),
        tight_layout=True,
        returnfig=True,
    )

    ax = axes[0]

    # Title and legend
    n_buys = buys.shape[0]
    n_sells = sells.shape[0]
    title = (f"{instrument} {granularity} — Pivot Retest + Engulfing Strategy\n"
             f"{df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')}  |  "
             f"Entries: {n_buys} LONG (▲), {n_sells} SHORT (▼)")
    ax.set_title(title, color="white", fontsize=13, pad=15)

    # Build manual legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="cyan", lw=1.5, label="SMA 50"),
        Line2D([0], [0], color="blue", lw=1, label="Pivot"),
        Line2D([0], [0], color="red", lw=1, linestyle="dashed", label="R1/R2/R3"),
        Line2D([0], [0], color="green", lw=1, linestyle="dashed", label="S1/S2/S3"),
        Line2D([0], [0], marker="^", color="lime", lw=0, markersize=10,
               markeredgecolor="black", label=f"BUY entry ({n_buys})"),
        Line2D([0], [0], marker="v", color="red", lw=0, markersize=10,
               markeredgecolor="black", label=f"SHORT entry ({n_sells})"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
              facecolor="#2a2a2a", edgecolor="gray", labelcolor="white")

    # Save
    root = get_project_root()
    out_dir = root / "logs"
    out_dir.mkdir(exist_ok=True)

    suffix = ""
    if start_date:
        suffix += f"_{start_date}"
    if end_date:
        suffix += f"_to_{end_date}"

    out_path = out_dir / f"chart_trades_{instrument}_{granularity}{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1e1e1e")
    print(f"Chart saved: {out_path}")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Chart trade entries on candlesticks")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--granularity", default="M5")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    # Default to a 2-week window showing a mix of trades
    if not args.start and not args.end:
        args.start = "2025-02-24"
        args.end = "2025-03-07"

    build_chart(args.instrument, args.granularity, args.start, args.end)


if __name__ == "__main__":
    main()
