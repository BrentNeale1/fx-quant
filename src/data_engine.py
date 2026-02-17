# src/data_engine.py
"""
Feature builder for FX candles.
Inputs: list of OANDA candle dicts (with 'mid' prices and 'volume').
Outputs: pandas DataFrame indexed by time with indicator columns.
"""

import pandas as pd
import numpy as np


def candles_to_df(candles):
    rows = []
    for c in candles:
        mid = c["mid"]
        rows.append({
            "time": c["time"],
            "open": float(mid["o"]),
            "high": float(mid["h"]),
            "low": float(mid["l"]),
            "close": float(mid["c"]),
            "volume": c.get("volume", 0)
        })

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    return df


def add_sma(df, period=20, col="close"):
    df[f"sma_{period}"] = df[col].rolling(period).mean()
    return df


def add_ema(df, period=20, col="close"):
    df[f"ema_{period}"] = df[col].ewm(span=period, adjust=False).mean()
    return df


def add_returns(df, col="close"):
    # simple pct returns and log returns
    df["ret"] = df[col].pct_change()
    df["logret"] = np.log(df[col] / df[col].shift(1))
    return df


def add_rolling_volatility(df, period=20, source="logret"):
    # annualization is not necessary here; this gives rolling stdev of returns
    df[f"vol_{period}"] = df[source].rolling(period).std()
    return df


def add_rsi(df, period=14, col="close"):
    # Classic RSI (Wilder's)
    delta = df[col].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_atr(df, period=14):
    # Average True Range
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["tr"] = tr
    df[f"atr_{period}"] = tr.rolling(period).mean()
    return df


def add_vwap(df, period=20):
    """
    Rolling VWAP-like: uses typical price * volume over rolling window.
    VWAP usually applies intraday at tick-level; for candle-level a rolling VWAP is a useful proxy.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    df["typical_price"] = tp
    df["pv"] = pv
    # rolling VWAP: sum(price*vol)/sum(vol)
    df[f"vwap_{period}"] = df["pv"].rolling(period).sum() / df["volume"].rolling(period).sum()
    # protect inf/nan where vol sum == 0
    df[f"vwap_{period}"] = df[f"vwap_{period}"].replace([np.inf, -np.inf], np.nan)
    return df


def detect_engulfing(df):
    """
    Detect bullish and bearish engulfing candle patterns.

    Bullish engulfing: current close > prev open AND current open < prev close
                       AND current body > prev body (fully engulfs).
    Bearish engulfing: current close < prev open AND current open > prev close
                       AND current body > prev body (fully engulfs).

    Returns df with 'bullish_engulfing' and 'bearish_engulfing' boolean columns.
    """
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_body = (prev_close - prev_open).abs()
    curr_body = (df["close"] - df["open"]).abs()

    df["bullish_engulfing"] = (
        (df["close"] > prev_open)
        & (df["open"] < prev_close)
        & (curr_body > prev_body)
    )
    df["bearish_engulfing"] = (
        (df["close"] < prev_open)
        & (df["open"] > prev_close)
        & (curr_body > prev_body)
    )
    return df


def add_pivot_points(df):
    """
    Add Classic Pivot Point support/resistance levels.

    Resamples intraday OHLC data to daily bars, computes pivot levels from the
    *previous* day's High/Low/Close, and merges them back onto the original
    DataFrame so every intraday bar carries the current day's pivot levels.

    Columns added: pivot, r1, r2, r3, s1, s2, s3
    """
    # Need a DatetimeIndex to resample
    df2 = df.copy()
    daily = df2.resample("D").agg({"high": "max", "low": "min", "close": "last"}).dropna()

    # Compute pivots from previous day
    daily["pivot"] = (daily["high"] + daily["low"] + daily["close"]) / 3
    daily["r1"] = 2 * daily["pivot"] - daily["low"]
    daily["s1"] = 2 * daily["pivot"] - daily["high"]
    daily["r2"] = daily["pivot"] + (daily["high"] - daily["low"])
    daily["s2"] = daily["pivot"] - (daily["high"] - daily["low"])
    daily["r3"] = daily["high"] + 2 * (daily["pivot"] - daily["low"])
    daily["s3"] = daily["low"] - 2 * (daily["high"] - daily["pivot"])

    # Shift so today's bars use *yesterday's* pivot levels
    pivot_cols = ["pivot", "r1", "r2", "r3", "s1", "s2", "s3"]
    daily_pivots = daily[pivot_cols].shift(1)

    # Assign a date key to the original df for merging
    df["_date"] = df.index.date
    daily_pivots["_date"] = daily_pivots.index.date

    df = df.merge(daily_pivots, on="_date", how="left")
    df.drop(columns=["_date"], inplace=True)

    # Restore the original DatetimeIndex
    df.index = df2.index

    return df


def add_calendar_features(df, calendar_df, cfg=None):
    """
    Add economic calendar feature columns to a candle DataFrame using merge_asof.

    Columns added:
    - minutes_to_next_event: minutes until the next event of any impact
    - minutes_to_next_high_impact: minutes until the next High-impact event
    - upcoming_high_impact_event: boolean, True if within buffer window of a High event
    - minutes_since_last_event: minutes since the most recent event

    Parameters
    ----------
    df : DataFrame with DatetimeIndex (candle bars)
    calendar_df : DataFrame with 'event_datetime' and 'impact_numeric' columns
    cfg : dict, optional economic_calendar config section
    """
    if cfg is None:
        cfg = {}

    buffer_minutes = cfg.get("event_buffer_minutes", 30)

    if calendar_df is None or calendar_df.empty:
        df["minutes_to_next_event"] = np.nan
        df["minutes_to_next_high_impact"] = np.nan
        df["upcoming_high_impact_event"] = False
        df["minutes_since_last_event"] = np.nan
        return df

    # Prepare calendar: ensure sorted datetime index
    cal = calendar_df[["event_datetime"]].copy()
    cal["event_datetime"] = pd.to_datetime(cal["event_datetime"], utc=True)
    cal = cal.drop_duplicates(subset=["event_datetime"]).sort_values("event_datetime")

    cal_high = calendar_df.loc[
        calendar_df["impact_numeric"] >= 3, ["event_datetime"]
    ].copy()
    cal_high["event_datetime"] = pd.to_datetime(cal_high["event_datetime"], utc=True)
    cal_high = cal_high.drop_duplicates(subset=["event_datetime"]).sort_values("event_datetime")

    # Prepare candle times
    candle_times = df.index.to_series().dt.tz_localize("UTC") if df.index.tz is None else df.index.to_series()
    candle_times = candle_times.reset_index(drop=True)

    # --- Forward merge: next event after each candle ---
    candle_frame = pd.DataFrame({"candle_time": candle_times})

    # Next event (any impact)
    merged_fwd = pd.merge_asof(
        candle_frame.sort_values("candle_time"),
        cal.rename(columns={"event_datetime": "next_event_time"}),
        left_on="candle_time",
        right_on="next_event_time",
        direction="forward",
    )
    df["minutes_to_next_event"] = (
        (merged_fwd["next_event_time"] - merged_fwd["candle_time"])
        .dt.total_seconds()
        .values / 60.0
    )

    # Next high-impact event
    if not cal_high.empty:
        merged_high = pd.merge_asof(
            candle_frame.sort_values("candle_time"),
            cal_high.rename(columns={"event_datetime": "next_high_time"}),
            left_on="candle_time",
            right_on="next_high_time",
            direction="forward",
        )
        df["minutes_to_next_high_impact"] = (
            (merged_high["next_high_time"] - merged_high["candle_time"])
            .dt.total_seconds()
            .values / 60.0
        )
    else:
        df["minutes_to_next_high_impact"] = np.nan

    # Upcoming high impact: True if within buffer_minutes of a High event
    df["upcoming_high_impact_event"] = (
        df["minutes_to_next_high_impact"].notna()
        & (df["minutes_to_next_high_impact"] <= buffer_minutes)
    )

    # --- Backward merge: last event before each candle ---
    merged_bwd = pd.merge_asof(
        candle_frame.sort_values("candle_time"),
        cal.rename(columns={"event_datetime": "prev_event_time"}),
        left_on="candle_time",
        right_on="prev_event_time",
        direction="backward",
    )
    df["minutes_since_last_event"] = (
        (merged_bwd["candle_time"] - merged_bwd["prev_event_time"])
        .dt.total_seconds()
        .values / 60.0
    )

    return df


def build_all_features(df, config=None, calendar_df=None):
    """
    Run a standard set of features. config is optional dict matching the
    'features' section of system.yaml.  Supports lists of windows for
    SMA and EMA so multiple columns are produced (e.g. sma_3, sma_20).

    If calendar_df is provided, also adds economic calendar features.

    Returns DataFrame with new columns.
    """
    if config is None:
        config = {}

    # Multi-window indicators (lists)
    sma_windows = config.get("sma_windows", [3, 20])
    ema_windows = config.get("ema_windows", [20])

    # Single-value indicators (scalars)
    rsi_p = config.get("rsi_period", 14)
    atr_p = config.get("atr_period", 14)
    vwap_w = config.get("vwap_window", 20)
    vol_w = config.get("volatility_window", 20)

    df = add_returns(df)

    for w in sma_windows:
        df = add_sma(df, period=w)

    for w in ema_windows:
        df = add_ema(df, period=w)

    df = add_rsi(df, period=rsi_p)
    df = add_rolling_volatility(df, period=vol_w, source="logret")
    df = add_atr(df, period=atr_p)
    df = add_vwap(df, period=vwap_w)

    # Drop intermediate helper columns
    df.drop(columns=["pv", "typical_price", "tr"], inplace=True, errors="ignore")

    # Economic calendar features (optional)
    if calendar_df is not None:
        cal_cfg = config.get("economic_calendar", {}) if config else {}
        df = add_calendar_features(df, calendar_df, cfg=cal_cfg)

    return df
