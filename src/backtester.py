# src/backtester.py
"""
Backtesting engine for fx-quant.
Reads features from Supabase, simulates the SMA cross strategy,
and produces P&L, drawdown, and trade log outputs.
"""

import os
import json
import math
from pathlib import Path

import pandas as pd
import numpy as np
from supabase import create_client

from config_loader import load_config, get_project_root
from data_engine import detect_engulfing, add_pivot_points


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_candles_from_supabase(instrument, granularity, supabase_client, table):
    """
    Query Supabase fx_candles table filtered by instrument + granularity.
    Returns a pandas DataFrame indexed by time, sorted ascending.
    Drops rows where sma columns are NaN (warmup period).
    """
    print(f"Fetching {instrument} / {granularity} from Supabase...")

    # Supabase JS-style pagination: fetch in pages of 1000
    all_rows = []
    page_size = 1000
    offset = 0
    while True:
        resp = (
            supabase_client.table(table)
            .select("*")
            .eq("instrument", instrument)
            .eq("granularity", granularity)
            .order("time", desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = resp.data or []
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size

    if not all_rows:
        print(f"  No data found for {instrument} / {granularity}.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()

    # Convert numeric columns from possible string types
    numeric_cols = [c for c in df.columns if c not in ("instrument", "granularity")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  Fetched {len(df)} rows.")
    return df


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signals(df, strategy_cfg, ai_cfg=None):
    """
    Generate trading signals based on strategy config.
    Currently supports 'sma_cross' rule only.

    Signal logic (long-only / flat):
      sma_short > sma_long  →  signal = 1  (long)
      sma_short < sma_long  →  signal = 0  (flat)

    If ai_cfg is provided, applies RSI overbought/oversold filtering
    to block entries at extreme levels.
    """
    rule = strategy_cfg["rule"]
    if rule == "pivot_retest_engulfing":
        return generate_signals_pivot_retest(df, strategy_cfg)
    elif rule != "sma_cross":
        raise ValueError(f"Unsupported strategy rule: {rule}")

    short_w = strategy_cfg["params"]["short"]
    long_w = strategy_cfg["params"]["long"]
    sma_short_col = f"sma_{short_w}"
    sma_long_col = f"sma_{long_w}"

    # Drop rows where indicator columns are NaN (warmup)
    required_cols = [sma_short_col, sma_long_col, "close", "ret"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    df = df.dropna(subset=[sma_short_col, sma_long_col, "ret"]).copy()

    # Signal: 1 when short SMA above long SMA, else 0
    df["signal"] = np.where(df[sma_short_col] > df[sma_long_col], 1, 0)

    # Apply RSI filter if configured
    if ai_cfg:
        sanity = ai_cfg.get("sanity_checks", {})
        rsi_ob = sanity.get("rsi_overbought")
        rsi_os = sanity.get("rsi_oversold")

        # Find the RSI column
        rsi_col = None
        for col in df.columns:
            if col.startswith("rsi_"):
                rsi_col = col
                break

        if rsi_col and (rsi_ob or rsi_os):
            rsi_vals = df[rsi_col]
            if rsi_ob is not None:
                df.loc[(df["signal"] == 1) & (rsi_vals > rsi_ob), "signal"] = 0
            if rsi_os is not None:
                df.loc[(df["signal"] == 1) & (rsi_vals < rsi_os), "signal"] = 0

    # Position changes only on crossover (detect changes)
    df["position"] = df["signal"]

    return df


# ---------------------------------------------------------------------------
# Pivot retest + engulfing signal generation
# ---------------------------------------------------------------------------

# Ordered pivot levels from lowest to highest
PIVOT_LEVEL_ORDER = ["s3", "s2", "s1", "pivot", "r1", "r2", "r3"]


def generate_signals_pivot_retest(df, strategy_cfg):
    """
    Generate LONG/SHORT signals based on pivot level retest confirmed by
    SMA 50 alignment and an engulfing candle pattern.

    Signal values: 1 = LONG, -1 = SHORT, 0 = FLAT.
    Also populates per-row: entry_level, sl_price, tp1_price, tp2_price.
    """
    params = strategy_cfg.get("params", {})
    sma_period = params.get("sma_period", 50)
    lookback = params.get("lookback_bars", 20)
    retest_tol_atr = params.get("retest_tolerance_atr", 0.5)
    strong_close_pct = params.get("strong_close_pct", 0.30)
    sl_atr_mult = params.get("sl_atr_multiplier", 1.5)

    sma_col = f"sma_{sma_period}"
    atr_col = "atr_14"

    # Ensure required columns exist
    required = ["open", "high", "low", "close", sma_col, atr_col,
                 "pivot", "r1", "r2", "r3", "s1", "s2", "s3"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column for pivot_retest_engulfing: {col}")

    # Add engulfing pattern detection
    df = detect_engulfing(df)

    # Drop warmup rows
    df = df.dropna(subset=[sma_col, atr_col, "pivot"]).copy()

    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    sma_vals = df[sma_col].values
    atr_vals = df[atr_col].values
    bull_eng = df["bullish_engulfing"].values
    bear_eng = df["bearish_engulfing"].values

    # Build a matrix of pivot level values per bar: shape (len(df), 7)
    level_names = PIVOT_LEVEL_ORDER
    level_matrix = np.column_stack([df[lv].values for lv in level_names])

    # Use numpy arrays for output (avoids pandas CoW issues)
    n = len(df)
    out_signal = np.zeros(n, dtype=int)
    out_entry_level = np.empty(n, dtype=object)
    out_entry_level[:] = ""
    out_sl = np.full(n, np.nan)
    out_tp1 = np.full(n, np.nan)
    out_tp2 = np.full(n, np.nan)

    for i in range(lookback, n):
        atr = atr_vals[i]
        if atr <= 0 or np.isnan(atr):
            continue

        candle_range = highs[i] - lows[i]
        if candle_range <= 0:
            continue

        tolerance = retest_tol_atr * atr

        # Check each pivot level for a retest setup
        for lv_idx, lv_name in enumerate(level_names):
            level_val = level_matrix[i, lv_idx]
            if np.isnan(level_val):
                continue

            # --- LONG check (support retest) ---
            # Look for: price broke below level, then returned above it
            broke_below = False
            for j in range(i - lookback, i):
                if closes[j] < level_val:
                    broke_below = True
                    break

            if broke_below and closes[i] > level_val:
                # Price is back above the level (retest from above)
                near_level = abs(closes[i] - level_val) <= tolerance
                sma_above = sma_vals[i] >= level_val
                is_bull_eng = bool(bull_eng[i])
                strong = (closes[i] - lows[i]) >= (1 - strong_close_pct) * candle_range

                if near_level and sma_above and is_bull_eng and strong:
                    # Find TP levels: next levels above entry
                    tp1, tp2 = _find_tp_levels_long(level_matrix[i], lv_idx)
                    if not np.isnan(tp1):
                        out_signal[i] = 1
                        out_entry_level[i] = lv_name
                        out_sl[i] = closes[i] - sl_atr_mult * atr
                        out_tp1[i] = tp1
                        out_tp2[i] = tp2 if not np.isnan(tp2) else tp1
                        break  # one signal per bar

            # --- SHORT check (resistance retest) ---
            broke_above = False
            for j in range(i - lookback, i):
                if closes[j] > level_val:
                    broke_above = True
                    break

            if broke_above and closes[i] < level_val:
                near_level = abs(closes[i] - level_val) <= tolerance
                sma_below = sma_vals[i] <= level_val
                is_bear_eng = bool(bear_eng[i])
                strong = (highs[i] - closes[i]) >= (1 - strong_close_pct) * candle_range

                if near_level and sma_below and is_bear_eng and strong:
                    tp1, tp2 = _find_tp_levels_short(level_matrix[i], lv_idx)
                    if not np.isnan(tp1):
                        out_signal[i] = -1
                        out_entry_level[i] = lv_name
                        out_sl[i] = closes[i] + sl_atr_mult * atr
                        out_tp1[i] = tp1
                        out_tp2[i] = tp2 if not np.isnan(tp2) else tp1
                        break

    # Assign output arrays back to DataFrame
    df["signal"] = out_signal
    df["entry_level"] = out_entry_level
    df["sl_price"] = out_sl
    df["tp1_price"] = out_tp1
    df["tp2_price"] = out_tp2
    df["position"] = out_signal
    return df


def _find_tp_levels_long(level_values, entry_lv_idx):
    """
    For a LONG trade entered at level_values[entry_lv_idx],
    find the next two pivot levels above (higher index = higher level).
    Returns (tp1, tp2) as floats; NaN if not found.
    """
    tp1 = np.nan
    tp2 = np.nan
    found = 0
    for k in range(entry_lv_idx + 1, len(level_values)):
        val = level_values[k]
        if not np.isnan(val):
            if found == 0:
                tp1 = val
                found += 1
            elif found == 1:
                tp2 = val
                break
    return tp1, tp2


def _find_tp_levels_short(level_values, entry_lv_idx):
    """
    For a SHORT trade entered at level_values[entry_lv_idx],
    find the next two pivot levels below (lower index = lower level).
    Returns (tp1, tp2) as floats; NaN if not found.
    """
    tp1 = np.nan
    tp2 = np.nan
    found = 0
    for k in range(entry_lv_idx - 1, -1, -1):
        val = level_values[k]
        if not np.isnan(val):
            if found == 0:
                tp1 = val
                found += 1
            elif found == 1:
                tp2 = val
                break
    return tp1, tp2


# ---------------------------------------------------------------------------
# Dual take-profit backtest engine
# ---------------------------------------------------------------------------

def run_backtest_dual_tp(df, strategy_cfg):
    """
    Backtest engine supporting per-trade SL/TP with partial closes.

    Position management:
    - On entry: full position at entry price with SL, TP1, TP2.
    - On TP1 hit: close 50%, move SL to breakeven (entry price).
    - On TP2 hit: close remaining 50%.
    - On SL hit: close full remaining position.

    Returns dict with equity_curve, trades, metrics (same interface as run_backtest).
    """
    trade_size_pct = strategy_cfg.get("trade_size_pct_of_equity", 0.01)
    max_dd_pct = strategy_cfg.get("max_drawdown_pct", 0.05)
    starting_equity = strategy_cfg.get("starting_equity", 100_000.0)

    equity = starting_equity
    peak_equity = equity
    stopped = False

    # Position state
    in_position = False
    direction = 0       # 1 = long, -1 = short
    entry_price = 0.0
    sl_price = 0.0
    tp1_price = 0.0
    tp2_price = 0.0
    position_size = 0.0
    half_closed = False

    equity_curve = []
    trades = []

    times = df.index.tolist()
    signals = df["signal"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    sl_col = df["sl_price"].values
    tp1_col = df["tp1_price"].values
    tp2_col = df["tp2_price"].values

    for i in range(len(df)):
        bar_time = times[i]
        bar_high = highs[i]
        bar_low = lows[i]
        bar_close = closes[i]

        if stopped:
            equity_curve.append(equity)
            continue

        # --- Check exits for active position ---
        if in_position:
            remaining_size = position_size * (0.5 if half_closed else 1.0)

            if direction == 1:  # LONG position
                # Check SL hit (low touches SL)
                if bar_low <= sl_price:
                    pnl = remaining_size * (sl_price - entry_price) / entry_price
                    equity += pnl
                    trades.append({
                        "time": bar_time, "side": "SL_EXIT_LONG",
                        "price": sl_price,
                        "position_size": 0.0,
                        "equity": round(equity, 2),
                        "drawdown": 0.0,
                        "pnl": round(pnl, 2),
                    })
                    in_position = False
                    half_closed = False
                # Check TP1 hit
                elif not half_closed and bar_high >= tp1_price:
                    half_size = position_size * 0.5
                    pnl = half_size * (tp1_price - entry_price) / entry_price
                    equity += pnl
                    trades.append({
                        "time": bar_time, "side": "TP1_LONG",
                        "price": tp1_price,
                        "position_size": round(half_size, 2),
                        "equity": round(equity, 2),
                        "drawdown": 0.0,
                        "pnl": round(pnl, 2),
                    })
                    half_closed = True
                    sl_price = entry_price  # move SL to breakeven
                    # Check if TP2 also hit on same bar
                    if bar_high >= tp2_price:
                        pnl2 = half_size * (tp2_price - entry_price) / entry_price
                        equity += pnl2
                        trades.append({
                            "time": bar_time, "side": "TP2_LONG",
                            "price": tp2_price,
                            "position_size": 0.0,
                            "equity": round(equity, 2),
                            "drawdown": 0.0,
                            "pnl": round(pnl2, 2),
                        })
                        in_position = False
                        half_closed = False
                # Check TP2 hit (after TP1 already closed)
                elif half_closed and bar_high >= tp2_price:
                    half_size = position_size * 0.5
                    pnl = half_size * (tp2_price - entry_price) / entry_price
                    equity += pnl
                    trades.append({
                        "time": bar_time, "side": "TP2_LONG",
                        "price": tp2_price,
                        "position_size": 0.0,
                        "equity": round(equity, 2),
                        "drawdown": 0.0,
                        "pnl": round(pnl, 2),
                    })
                    in_position = False
                    half_closed = False

            elif direction == -1:  # SHORT position
                # Check SL hit (high touches SL)
                if bar_high >= sl_price:
                    pnl = remaining_size * (entry_price - sl_price) / entry_price
                    equity += pnl
                    trades.append({
                        "time": bar_time, "side": "SL_EXIT_SHORT",
                        "price": sl_price,
                        "position_size": 0.0,
                        "equity": round(equity, 2),
                        "drawdown": 0.0,
                        "pnl": round(pnl, 2),
                    })
                    in_position = False
                    half_closed = False
                # Check TP1 hit (low touches TP1)
                elif not half_closed and bar_low <= tp1_price:
                    half_size = position_size * 0.5
                    pnl = half_size * (entry_price - tp1_price) / entry_price
                    equity += pnl
                    trades.append({
                        "time": bar_time, "side": "TP1_SHORT",
                        "price": tp1_price,
                        "position_size": round(half_size, 2),
                        "equity": round(equity, 2),
                        "drawdown": 0.0,
                        "pnl": round(pnl, 2),
                    })
                    half_closed = True
                    sl_price = entry_price  # move SL to breakeven
                    # Check if TP2 also hit on same bar
                    if bar_low <= tp2_price:
                        pnl2 = half_size * (entry_price - tp2_price) / entry_price
                        equity += pnl2
                        trades.append({
                            "time": bar_time, "side": "TP2_SHORT",
                            "price": tp2_price,
                            "position_size": 0.0,
                            "equity": round(equity, 2),
                            "drawdown": 0.0,
                            "pnl": round(pnl2, 2),
                        })
                        in_position = False
                        half_closed = False
                # Check TP2 hit
                elif half_closed and bar_low <= tp2_price:
                    half_size = position_size * 0.5
                    pnl = half_size * (entry_price - tp2_price) / entry_price
                    equity += pnl
                    trades.append({
                        "time": bar_time, "side": "TP2_SHORT",
                        "price": tp2_price,
                        "position_size": 0.0,
                        "equity": round(equity, 2),
                        "drawdown": 0.0,
                        "pnl": round(pnl, 2),
                    })
                    in_position = False
                    half_closed = False

        # --- Check for new entry signal (only when flat) ---
        if not in_position and not stopped:
            sig = signals[i]
            if sig in (1, -1) and not np.isnan(sl_col[i]):
                direction = sig
                entry_price = bar_close
                sl_price = sl_col[i]
                tp1_price = tp1_col[i]
                tp2_price = tp2_col[i]
                position_size = equity * trade_size_pct
                half_closed = False
                in_position = True

                side_label = "BUY" if sig == 1 else "SELL_SHORT"
                trades.append({
                    "time": bar_time,
                    "side": side_label,
                    "price": bar_close,
                    "position_size": round(position_size, 2),
                    "equity": round(equity, 2),
                    "drawdown": 0.0,
                    "pnl": 0.0,
                })

        # Update peak and drawdown
        if equity > peak_equity:
            peak_equity = equity
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0

        # Max drawdown breached — close position and stop
        if drawdown >= max_dd_pct:
            if in_position:
                remaining_size = position_size * (0.5 if half_closed else 1.0)
                if direction == 1:
                    pnl = remaining_size * (bar_close - entry_price) / entry_price
                else:
                    pnl = remaining_size * (entry_price - bar_close) / entry_price
                equity += pnl
                trades.append({
                    "time": bar_time, "side": "DD_EXIT",
                    "price": bar_close,
                    "position_size": 0.0,
                    "equity": round(equity, 2),
                    "drawdown": round(drawdown, 6),
                    "pnl": round(pnl, 2),
                })
                in_position = False
                half_closed = False
            stopped = True
            print(f"  Max drawdown {max_dd_pct:.1%} breached at {bar_time}. Stopping.")

        equity_curve.append(equity)

    equity_series = pd.Series(equity_curve, index=df.index, name="equity")
    metrics = compute_metrics(equity_series, trades, starting_equity)

    return {
        "equity_curve": equity_series,
        "trades": trades,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Backtest engine (SMA cross — original)
# ---------------------------------------------------------------------------

def run_backtest(df, strategy_cfg):
    """
    Walk through bars, track position, equity, trades.
    Returns dict with equity_curve (Series), trades (list of dicts), metrics (dict).
    """
    trade_size_pct = strategy_cfg.get("trade_size_pct_of_equity", 0.01)
    max_dd_pct = strategy_cfg.get("max_drawdown_pct", 0.05)

    starting_equity = cfg_equity if (cfg_equity := strategy_cfg.get("starting_equity")) else 100_000.0
    equity = starting_equity
    peak_equity = equity
    position = 0  # 0 = flat, 1 = long
    position_size = 0.0
    stopped = False

    equity_curve = []
    trades = []

    times = df.index.tolist()
    signals = df["signal"].values
    closes = df["close"].values
    rets = df["ret"].values

    for i in range(len(df)):
        bar_time = times[i]
        sig = signals[i]
        price = closes[i]
        bar_ret = rets[i]

        # Check max drawdown stop
        if stopped:
            equity_curve.append(equity)
            continue

        # P&L from existing position
        if position == 1 and i > 0:
            pnl = position_size * bar_ret
            equity += pnl

        # Update peak and drawdown
        if equity > peak_equity:
            peak_equity = equity
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0

        # Max drawdown breached — stop trading
        if drawdown >= max_dd_pct:
            if position == 1:
                trades.append({
                    "time": bar_time,
                    "side": "FLAT",
                    "price": price,
                    "position_size": 0.0,
                    "equity": round(equity, 2),
                    "drawdown": round(drawdown, 6),
                })
            stopped = True
            position = 0
            position_size = 0.0
            equity_curve.append(equity)
            print(f"  Max drawdown {max_dd_pct:.1%} breached at {bar_time}. Stopping.")
            continue

        # Position change
        if sig != position:
            if sig == 1 and position == 0:
                # Enter long
                position_size = equity * trade_size_pct
                trades.append({
                    "time": bar_time,
                    "side": "BUY",
                    "price": price,
                    "position_size": round(position_size, 2),
                    "equity": round(equity, 2),
                    "drawdown": round(drawdown, 6),
                })
            elif sig == 0 and position == 1:
                # Exit long
                trades.append({
                    "time": bar_time,
                    "side": "FLAT",
                    "price": price,
                    "position_size": 0.0,
                    "equity": round(equity, 2),
                    "drawdown": round(drawdown, 6),
                })
                position_size = 0.0
            position = sig

        equity_curve.append(equity)

    equity_series = pd.Series(equity_curve, index=df.index, name="equity")
    metrics = compute_metrics(equity_series, trades, starting_equity)

    return {
        "equity_curve": equity_series,
        "trades": trades,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(equity_curve, trades, starting_equity=100_000.0):
    """
    Compute summary metrics from equity curve and trade list.
    """
    final_equity = equity_curve.iloc[-1] if len(equity_curve) > 0 else starting_equity
    total_return_pct = ((final_equity - starting_equity) / starting_equity) * 100

    # Max drawdown from equity curve
    peak = equity_curve.cummax()
    dd = (peak - equity_curve) / peak
    max_drawdown_pct = dd.max() * 100 if len(dd) > 0 else 0.0

    # Trade stats
    num_trades = len(trades)

    # Count winning round-trips
    # Supports both original (BUY→FLAT) and dual-TP (BUY→TP/SL exit) formats
    entry_sides = {"BUY", "SELL_SHORT"}
    exit_sides = {"FLAT", "SL_EXIT_LONG", "SL_EXIT_SHORT",
                  "TP1_LONG", "TP1_SHORT", "TP2_LONG", "TP2_SHORT", "DD_EXIT"}
    wins = 0
    entry_equity = None
    round_trips = 0
    for t in trades:
        if t["side"] in entry_sides:
            entry_equity = t["equity"]
        elif t["side"] in exit_sides and entry_equity is not None:
            # A round-trip completes when the full position is closed (size=0)
            if t.get("position_size", 0) == 0:
                round_trips += 1
                if t["equity"] > entry_equity:
                    wins += 1
                entry_equity = None
    win_rate = (wins / round_trips * 100) if round_trips > 0 else 0.0

    # Sharpe ratio (annualized, from per-bar returns of the equity curve)
    eq_returns = equity_curve.pct_change().dropna()
    sharpe = 0.0
    if len(eq_returns) > 1 and eq_returns.std() > 0:
        sharpe = (eq_returns.mean() / eq_returns.std()) * math.sqrt(252 * 24 * 60)  # per-minute approx

    return {
        "starting_equity": round(starting_equity, 2),
        "total_return_pct": round(total_return_pct, 4),
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "num_trades": num_trades,
        "round_trips": round_trips,
        "win_rate_pct": round(win_rate, 2),
        "sharpe_ratio": round(sharpe, 4),
        "final_equity": round(final_equity, 2),
    }


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------

def compute_monthly_pnl(equity_curve, starting_equity=100_000.0):
    """
    Compute P&L for each calendar month from the equity curve.
    Returns list of dicts with month, start_equity, end_equity, pnl, pnl_pct.
    """
    monthly = []
    # Group by year-month
    grouped = equity_curve.groupby(equity_curve.index.to_period("M"))
    prev_end = starting_equity

    for period, group in grouped:
        end_eq = group.iloc[-1]
        pnl = end_eq - prev_end
        pnl_pct = (pnl / prev_end * 100) if prev_end != 0 else 0.0
        monthly.append({
            "month": str(period),
            "start_equity": round(prev_end, 2),
            "end_equity": round(end_eq, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
        })
        prev_end = end_eq

    return monthly


def save_results(instrument, granularity, results, metrics):
    """
    Save trade log CSV, monthly P&L, and a JSON summary for the dashboard.
    """
    root = get_project_root()
    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)

    trades = results["trades"]
    equity_curve = results["equity_curve"]

    # Trade log CSV
    if trades:
        trade_df = pd.DataFrame(trades)
        trade_df["instrument"] = instrument
        trade_df["granularity"] = granularity
        base_cols = ["time", "instrument", "granularity", "side", "price",
                     "position_size", "equity", "drawdown"]
        if "pnl" in trade_df.columns:
            base_cols.append("pnl")
        cols = [c for c in base_cols if c in trade_df.columns]
        trade_df = trade_df[cols]
        csv_path = logs_dir / f"backtest_trades_{instrument}_{granularity}.csv"
        trade_df.to_csv(csv_path, index=False)
        print(f"  Trade log saved: {csv_path}")
    else:
        print("  No trades to log.")

    # Monthly P&L
    starting_equity = metrics.get("starting_equity", 100_000.0)
    monthly = compute_monthly_pnl(equity_curve, starting_equity)

    # Save JSON summary for dashboard
    summary = {
        "instrument": instrument,
        "granularity": granularity,
        "run_time": pd.Timestamp.now(tz="UTC").isoformat(),
        "data_range": {
            "start": str(equity_curve.index[0]) if len(equity_curve) > 0 else "",
            "end": str(equity_curve.index[-1]) if len(equity_curve) > 0 else "",
            "bars": len(equity_curve),
        },
        "metrics": metrics,
        "monthly_pnl": monthly,
    }
    json_path = logs_dir / f"backtest_summary_{instrument}_{granularity}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary saved: {json_path}")

    # Console summary
    print(f"\n  --- {instrument} / {granularity} Summary ---")
    print(f"  Total return:   {metrics['total_return_pct']:.4f}%")
    print(f"  Max drawdown:   {metrics['max_drawdown_pct']:.4f}%")
    print(f"  Trades:         {metrics['num_trades']}  (round-trips: {metrics['round_trips']})")
    print(f"  Win rate:       {metrics['win_rate_pct']:.2f}%")
    print(f"  Sharpe ratio:   {metrics['sharpe_ratio']:.4f}")
    print(f"  Final equity:   ${metrics['final_equity']:,.2f}")

    # Monthly P&L table
    print(f"\n  Monthly P&L:")
    print(f"  {'Month':<10} {'Start':>12} {'End':>12} {'P&L':>10} {'%':>8}")
    print(f"  {'-'*54}")
    for m in monthly:
        sign = "+" if m["pnl"] >= 0 else ""
        print(f"  {m['month']:<10} ${m['start_equity']:>11,.2f} ${m['end_equity']:>11,.2f} {sign}${m['pnl']:>8,.2f} {sign}{m['pnl_pct']:.2f}%")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()

    # Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_KEY in config/.env")
    sb = create_client(supabase_url, supabase_key)
    table = cfg.get("supabase", {}).get("table", "fx_candles")

    strategy_cfg = cfg["strategy"]
    ai_cfg = cfg.get("ai", {})
    instruments = cfg["brokers"][0]["instruments"]
    granularities = cfg["data"]["candle_granularities"]

    rsi_ob = ai_cfg.get("sanity_checks", {}).get("rsi_overbought", "off")
    rsi_os = ai_cfg.get("sanity_checks", {}).get("rsi_oversold", "off")
    print(f"Backtesting strategy: {strategy_cfg['rule']}")
    print(f"RSI filter: overbought={rsi_ob}, oversold={rsi_os}")
    print(f"Instruments: {instruments}")
    print(f"Granularities: {granularities}\n")

    for instrument in instruments:
        for granularity in granularities:
            print(f"=== {instrument} / {granularity} ===")

            df = fetch_candles_from_supabase(instrument, granularity, sb, table)
            if df.empty:
                print("  Skipping — no data.\n")
                continue

            # Add pivot points if needed for pivot_retest_engulfing strategy
            if strategy_cfg["rule"] == "pivot_retest_engulfing":
                from data_engine import add_pivot_points
                df = add_pivot_points(df)

            df = generate_signals(df, strategy_cfg, ai_cfg=ai_cfg)
            if df.empty:
                print("  Skipping — no valid rows after warmup.\n")
                continue

            if strategy_cfg["rule"] == "pivot_retest_engulfing":
                results = run_backtest_dual_tp(df, strategy_cfg)
            else:
                results = run_backtest(df, strategy_cfg)
            save_results(instrument, granularity, results, results["metrics"])

    print("Backtesting complete.")


if __name__ == "__main__":
    main()
