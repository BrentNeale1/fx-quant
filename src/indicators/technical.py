"""
Technical indicator library for fx-quant Phase 1.

All indicators are computed on the full dataframe upfront, but the backtester
only exposes data up to the current candle index (preventing lookahead).
Using .shift(1) where noted to ensure signals only use closed-candle data.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    # When both are positive, keep only the larger
    both_pos = (plus_dm > 0) & (minus_dm > 0)
    plus_bigger = plus_dm >= minus_dm
    plus_dm = plus_dm.where(~both_pos | plus_bigger, 0)
    minus_dm = minus_dm.where(~both_pos | ~plus_bigger, 0)

    atr_vals = atr(df, period)

    plus_di = 100 * ema(plus_dm, period) / atr_vals.replace(0, np.nan)
    minus_di = 100 * ema(minus_dm, period) / atr_vals.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return ema(dx, period)


# ---------------------------------------------------------------------------
# Stochastic Oscillator
# ---------------------------------------------------------------------------

def stochastic(df: pd.DataFrame, k_period: int = 5, k_smooth: int = 3,
               d_smooth: int = 3):
    low_min = df["low"].rolling(k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(k_period, min_periods=k_period).max()
    raw_k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    k = raw_k.rolling(k_smooth, min_periods=1).mean()
    d = k.rolling(d_smooth, min_periods=1).mean()
    return k, d


# ---------------------------------------------------------------------------
# Session VWAP with Bands (reset at London open 08:00 UTC)
# ---------------------------------------------------------------------------

def session_vwap_bands(df: pd.DataFrame, session_start_hour: int = 8):
    """Compute intra-session VWAP with standard deviation bands."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["volume"].replace(0, 1)  # avoid division by zero

    # Identify session boundaries
    hours = pd.Series(df.index.hour, index=df.index)
    prev_hours = hours.shift(1)
    session_start = (hours == session_start_hour) & (prev_hours != session_start_hour)
    session_start.iloc[0] = True

    # Assign session IDs
    session_id = session_start.astype(int).cumsum()

    # Cumulative VWAP per session
    tp_vol = typical_price * volume
    cum_tp_vol = tp_vol.groupby(session_id).cumsum()
    cum_vol = volume.groupby(session_id).cumsum()
    vwap = cum_tp_vol / cum_vol

    # Rolling std dev of typical price from VWAP within session
    deviation = typical_price - vwap
    cum_dev_sq = (deviation ** 2).groupby(session_id).cumsum()
    cum_count = deviation.groupby(session_id).cumcount() + 1
    std_dev = np.sqrt(cum_dev_sq / cum_count)

    return pd.DataFrame({
        "session_vwap": vwap,
        "vwap_std": std_dev,
        "vwap_upper_1_5": vwap + 1.5 * std_dev,
        "vwap_lower_1_5": vwap - 1.5 * std_dev,
        "vwap_upper_2": vwap + 2.0 * std_dev,
        "vwap_lower_2": vwap - 2.0 * std_dev,
        "vwap_upper_2_5": vwap + 2.5 * std_dev,
        "vwap_lower_2_5": vwap - 2.5 * std_dev,
    }, index=df.index)


# ---------------------------------------------------------------------------
# Swing High / Low Detection
# ---------------------------------------------------------------------------

def swing_highs(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """True where high[i] is highest in [i-lookback, i+lookback] window."""
    high = df["high"]
    roll_max = high.rolling(2 * lookback + 1, center=True, min_periods=lookback + 1).max()
    return high == roll_max


def swing_lows(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """True where low[i] is lowest in [i-lookback, i+lookback] window."""
    low = df["low"]
    roll_min = low.rolling(2 * lookback + 1, center=True, min_periods=lookback + 1).min()
    return low == roll_min


# ---------------------------------------------------------------------------
# Key Level Identification (horizontal S/R)
# ---------------------------------------------------------------------------

def identify_key_levels(df: pd.DataFrame, lookback: int = 5,
                        tolerance_atr_mult: float = 0.5,
                        min_touches: int = 3) -> list:
    """
    Find horizontal S/R levels by clustering swing highs/lows.
    Returns list of (price_level, touch_count) tuples.
    """
    atr_val = atr(df).iloc[-1] if len(df) > 14 else None
    if atr_val is None or np.isnan(atr_val):
        return []

    tolerance = atr_val * tolerance_atr_mult

    # Collect swing points
    sh = swing_highs(df, lookback)
    sl = swing_lows(df, lookback)
    swing_prices = pd.concat([
        df.loc[sh, "high"],
        df.loc[sl, "low"],
    ]).sort_values()

    if len(swing_prices) < min_touches:
        return []

    # Cluster swing points
    levels = []
    used = set()
    for i, price in enumerate(swing_prices):
        if i in used:
            continue
        cluster = [price]
        used.add(i)
        for j in range(i + 1, len(swing_prices)):
            if j in used:
                continue
            if abs(swing_prices.iloc[j] - price) <= tolerance:
                cluster.append(swing_prices.iloc[j])
                used.add(j)
        if len(cluster) >= min_touches:
            levels.append((np.mean(cluster), len(cluster)))

    return sorted(levels, key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Engulfing Candle Detection
# ---------------------------------------------------------------------------

def is_bullish_engulfing(df: pd.DataFrame, i: int) -> bool:
    if i < 1:
        return False
    prev = df.iloc[i - 1]
    curr = df.iloc[i]
    prev_body = abs(prev["close"] - prev["open"])
    curr_body = abs(curr["close"] - curr["open"])
    return (prev["close"] < prev["open"] and  # prev bearish
            curr["close"] > curr["open"] and   # curr bullish
            curr_body > prev_body and           # engulfs
            curr["close"] > prev["open"] and
            curr["open"] <= prev["close"])


def is_bearish_engulfing(df: pd.DataFrame, i: int) -> bool:
    if i < 1:
        return False
    prev = df.iloc[i - 1]
    curr = df.iloc[i]
    prev_body = abs(prev["close"] - prev["open"])
    curr_body = abs(curr["close"] - curr["open"])
    return (prev["close"] > prev["open"] and  # prev bullish
            curr["close"] < curr["open"] and   # curr bearish
            curr_body > prev_body and           # engulfs
            curr["close"] < prev["open"] and
            curr["open"] >= prev["close"])


# ---------------------------------------------------------------------------
# RSI Divergence Detection
# ---------------------------------------------------------------------------

def detect_rsi_divergence(df: pd.DataFrame, rsi_col: str, lookback: int = 20,
                          i: int = None) -> str:
    """
    Detect regular RSI divergence at index i.
    Returns 'bullish', 'bearish', or None.
    """
    if i is None:
        i = len(df) - 1
    if i < lookback:
        return None

    window = df.iloc[i - lookback:i + 1]
    rsi_vals = window[rsi_col]
    lows = window["low"]
    highs = window["high"]

    # Bullish divergence: price makes lower low, RSI makes higher low
    recent_low_idx = lows.idxmin()
    if recent_low_idx == window.index[-1]:
        # Current bar is the low
        prev_window = window.iloc[:-3]  # exclude last 3 bars
        if len(prev_window) > 3:
            prev_low_idx = prev_window["low"].idxmin()
            if (lows.loc[recent_low_idx] < prev_window["low"].loc[prev_low_idx] and
                    rsi_vals.loc[recent_low_idx] > rsi_vals.loc[prev_low_idx]):
                return "bullish"

    # Bearish divergence: price makes higher high, RSI makes lower high
    recent_high_idx = highs.idxmax()
    if recent_high_idx == window.index[-1]:
        prev_window = window.iloc[:-3]
        if len(prev_window) > 3:
            prev_high_idx = prev_window["high"].idxmax()
            if (highs.loc[recent_high_idx] > prev_window["high"].loc[prev_high_idx] and
                    rsi_vals.loc[recent_high_idx] < rsi_vals.loc[prev_high_idx]):
                return "bearish"

    return None


# ---------------------------------------------------------------------------
# Master Function: Compute All Indicators on a DataFrame
# ---------------------------------------------------------------------------

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to a candle dataframe.
    The caller must ensure df has columns: open, high, low, close, volume.
    """
    df = df.copy()

    # Moving Averages
    for period in [20, 50, 100, 200]:
        df[f"ema_{period}"] = ema(df["close"], period)
    df["sma_200"] = sma(df["close"], 200)

    # RSI
    df["rsi_14"] = rsi(df["close"], 14)

    # ATR
    df["atr_14"] = atr(df, 14)

    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["close"])

    # ADX
    df["adx_14"] = adx(df, 14)

    # Stochastic
    df["stoch_k"], df["stoch_d"] = stochastic(df)

    # Session VWAP (only meaningful for intraday timeframes)
    if len(df) > 0 and hasattr(df.index, "hour"):
        try:
            vwap_df = session_vwap_bands(df)
            for col in vwap_df.columns:
                df[col] = vwap_df[col]
        except Exception:
            pass

    # Swing points
    df["is_swing_high"] = swing_highs(df)
    df["is_swing_low"] = swing_lows(df)

    return df
