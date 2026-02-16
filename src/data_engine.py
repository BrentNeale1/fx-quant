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


def build_all_features(df, config=None):
    """
    Run a standard set of features. config is optional dict matching the
    'features' section of system.yaml.  Supports lists of windows for
    SMA and EMA so multiple columns are produced (e.g. sma_3, sma_20).

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

    return df
