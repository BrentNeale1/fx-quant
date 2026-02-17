# src/strategies.py
"""
Five trading strategy signal generators.

Each function takes a primary DataFrame (entry timeframe) and optional
filter DataFrame (higher timeframe) and returns the entry df with columns:
  signal (1=long, -1=short, 0=flat), sl_price, tp1_price, tp2_price, tp3_price,
  confidence (int 0-5), strategy_name (str).
"""

import numpy as np
import pandas as pd

from data_engine import detect_engulfing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge_htf(df_entry, df_filter, cols, suffix="_htf"):
    """Forward-fill higher-timeframe columns onto entry-timeframe bars."""
    if df_filter is None or df_filter.empty:
        return df_entry
    htf = df_filter[cols].copy()
    htf.columns = [c + suffix for c in cols]
    merged = pd.merge_asof(
        df_entry, htf,
        left_index=True, right_index=True,
        direction="backward",
    )
    return merged


def _find_swing_highs(highs, order=5):
    """Return boolean array where highs[i] is a local maximum over +/-order bars."""
    n = len(highs)
    result = np.zeros(n, dtype=bool)
    for i in range(order, n - order):
        if highs[i] == np.max(highs[i - order:i + order + 1]):
            result[i] = True
    return result


def _find_swing_lows(lows, order=5):
    """Return boolean array where lows[i] is a local minimum over +/-order bars."""
    n = len(lows)
    result = np.zeros(n, dtype=bool)
    for i in range(order, n - order):
        if lows[i] == np.min(lows[i - order:i + order + 1]):
            result[i] = True
    return result


def _identify_key_levels(df, atr_col="atr_14", min_touches=3, lookback=500):
    """
    Find horizontal S/R key levels by clustering swing highs/lows.
    Returns sorted list of (level_price, touch_count).
    """
    highs = df["high"].values
    lows = df["low"].values
    atr = df[atr_col].values

    n = len(df)
    start = max(0, n - lookback)
    swing_h = _find_swing_highs(highs[start:], order=10)
    swing_l = _find_swing_lows(lows[start:], order=10)

    # Collect swing point prices
    swing_prices = []
    for i in range(len(swing_h)):
        if swing_h[i]:
            swing_prices.append(highs[start + i])
        if swing_l[i]:
            swing_prices.append(lows[start + i])

    if not swing_prices:
        return []

    # Cluster within 0.5 * median ATR tolerance
    median_atr = np.nanmedian(atr[start:])
    if np.isnan(median_atr) or median_atr <= 0:
        return []
    tolerance = 0.5 * median_atr

    swing_prices.sort()
    clusters = []
    current_cluster = [swing_prices[0]]

    for price in swing_prices[1:]:
        if price - current_cluster[-1] <= tolerance:
            current_cluster.append(price)
        else:
            if len(current_cluster) >= min_touches:
                clusters.append((np.mean(current_cluster), len(current_cluster)))
            current_cluster = [price]
    if len(current_cluster) >= min_touches:
        clusters.append((np.mean(current_cluster), len(current_cluster)))

    return sorted(clusters, key=lambda x: x[0])


def _nearest_key_level(price, key_levels, tolerance):
    """Return nearest key level within tolerance, or None."""
    best = None
    best_dist = tolerance
    for lvl, _ in key_levels:
        d = abs(price - lvl)
        if d <= best_dist:
            best = lvl
            best_dist = d
    return best


# ---------------------------------------------------------------------------
# S1: MA Breakout-Retest
# ---------------------------------------------------------------------------

def generate_signals_s1(df_entry, df_filter=None, params=None):
    """
    MA Breakout-Retest strategy.

    Entry TF: M15. Filter TF: H1.
    - Detect swing highs/lows forming channel lines
    - Break above/below channel + retest with engulfing candle
    - EMA 50 slope check, EMA 200 obstacle filter
    - H1 SMA 200 directional filter
    - Confidence scoring 0-5
    """
    if params is None:
        params = {}

    sl_atr_mult = params.get("sl_atr_multiplier", 1.5)
    swing_order = params.get("swing_order", 10)
    channel_bars = params.get("channel_lookback", 60)
    retest_bars = params.get("retest_lookback", 5)

    df = df_entry.copy()
    df = detect_engulfing(df)

    # Merge H1 SMA 200 as directional filter
    if df_filter is not None:
        df = _merge_htf(df, df_filter, ["sma_200", "close"], suffix="_htf")
    else:
        df["sma_200_htf"] = np.nan
        df["close_htf"] = np.nan

    n = len(df)
    signals = np.zeros(n, dtype=int)
    sl = np.full(n, np.nan)
    tp1 = np.full(n, np.nan)
    tp2 = np.full(n, np.nan)
    tp3 = np.full(n, np.nan)
    confidence = np.zeros(n, dtype=int)

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    ema50 = df["ema_50"].values if "ema_50" in df.columns else np.full(n, np.nan)
    ema200 = df["ema_200"].values if "ema_200" in df.columns else np.full(n, np.nan)
    sma200_htf = df["sma_200_htf"].values if "sma_200_htf" in df.columns else np.full(n, np.nan)
    close_htf = df["close_htf"].values if "close_htf" in df.columns else np.full(n, np.nan)
    atr = df["atr_14"].values
    bull_eng = df["bullish_engulfing"].values
    bear_eng = df["bearish_engulfing"].values

    for i in range(max(channel_bars, 30), n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        # Find recent swing highs/lows for channel
        window_start = max(0, i - channel_bars)
        sw_h = _find_swing_highs(highs[window_start:i + 1], order=swing_order)
        sw_l = _find_swing_lows(lows[window_start:i + 1], order=swing_order)

        swing_high_prices = [highs[window_start + j] for j in range(len(sw_h)) if sw_h[j]]
        swing_low_prices = [lows[window_start + j] for j in range(len(sw_l)) if sw_l[j]]

        if len(swing_high_prices) < 2 or len(swing_low_prices) < 2:
            continue

        # Channel resistance = line through last 2 swing highs
        channel_high = swing_high_prices[-1]
        # Channel support = line through last 2 swing lows
        channel_low = swing_low_prices[-1]

        # H1 filter: price above/below H1 SMA 200
        htf_bullish = np.isnan(sma200_htf[i]) or close_htf[i] > sma200_htf[i]
        htf_bearish = np.isnan(sma200_htf[i]) or close_htf[i] < sma200_htf[i]

        # EMA 50 slope (positive = bullish)
        ema50_slope = ema50[i] - ema50[max(0, i - 5)] if not np.isnan(ema50[i]) else 0

        # --- LONG: break above descending resistance, retest, bullish engulfing ---
        broke_above = False
        for j in range(max(0, i - retest_bars), i):
            if closes[j] > channel_high:
                broke_above = True
                break

        if (broke_above and bull_eng[i] and htf_bullish and ema50_slope > 0
                and abs(closes[i] - channel_high) <= 1.5 * atr[i]):
            # Check EMA 200 not blocking (price above it)
            if np.isnan(ema200[i]) or closes[i] > ema200[i]:
                sig_conf = 1  # base
                if ema50_slope > 0:
                    sig_conf += 1
                if not np.isnan(sma200_htf[i]) and close_htf[i] > sma200_htf[i]:
                    sig_conf += 1
                if bull_eng[i]:
                    sig_conf += 1
                if closes[i] > ema50[i] if not np.isnan(ema50[i]) else False:
                    sig_conf += 1

                signals[i] = 1
                sl[i] = closes[i] - sl_atr_mult * atr[i]
                tp1[i] = closes[i] + 1.5 * atr[i]
                tp2[i] = closes[i] + 2.5 * atr[i]
                tp3[i] = closes[i] + 4.0 * atr[i]
                confidence[i] = min(sig_conf, 5)
                continue

        # --- SHORT: break below ascending support, retest, bearish engulfing ---
        broke_below = False
        for j in range(max(0, i - retest_bars), i):
            if closes[j] < channel_low:
                broke_below = True
                break

        if (broke_below and bear_eng[i] and htf_bearish and ema50_slope < 0
                and abs(closes[i] - channel_low) <= 1.5 * atr[i]):
            if np.isnan(ema200[i]) or closes[i] < ema200[i]:
                sig_conf = 1
                if ema50_slope < 0:
                    sig_conf += 1
                if not np.isnan(sma200_htf[i]) and close_htf[i] < sma200_htf[i]:
                    sig_conf += 1
                if bear_eng[i]:
                    sig_conf += 1
                if closes[i] < ema50[i] if not np.isnan(ema50[i]) else False:
                    sig_conf += 1

                signals[i] = -1
                sl[i] = closes[i] + sl_atr_mult * atr[i]
                tp1[i] = closes[i] - 1.5 * atr[i]
                tp2[i] = closes[i] - 2.5 * atr[i]
                tp3[i] = closes[i] - 4.0 * atr[i]
                confidence[i] = min(sig_conf, 5)

    df["signal"] = signals
    df["sl_price"] = sl
    df["tp1_price"] = tp1
    df["tp2_price"] = tp2
    df["tp3_price"] = tp3
    df["confidence"] = confidence
    df["strategy_name"] = "S1_MA_Breakout_Retest"
    df["position"] = signals
    return df


# ---------------------------------------------------------------------------
# S2: Session VWAP Reversal
# ---------------------------------------------------------------------------

def generate_signals_s2(df_entry, df_filter=None, params=None):
    """
    Session VWAP Reversal strategy.

    Entry TF: M15.
    - Price crosses +/-2sigma VWAP band
    - RSI confirmation (>70 short, <30 long)
    - Target: VWAP line (TP1), opposite +/-0.5sigma (TP2)
    """
    if params is None:
        params = {}

    sl_atr_mult = params.get("sl_atr_multiplier", 1.5)

    df = df_entry.copy()
    n = len(df)
    signals = np.zeros(n, dtype=int)
    sl = np.full(n, np.nan)
    tp1 = np.full(n, np.nan)
    tp2 = np.full(n, np.nan)
    tp3 = np.full(n, np.nan)
    confidence = np.zeros(n, dtype=int)

    required = ["session_vwap", "vwap_upper_2", "vwap_lower_2", "rsi_14", "atr_14"]
    for col in required:
        if col not in df.columns:
            df["signal"] = 0
            df["sl_price"] = np.nan
            df["tp1_price"] = np.nan
            df["tp2_price"] = np.nan
            df["tp3_price"] = np.nan
            df["confidence"] = 0
            df["strategy_name"] = "S2_Session_VWAP_Reversal"
            df["position"] = 0
            return df

    closes = df["close"].values
    vwap = df["session_vwap"].values
    upper2 = df["vwap_upper_2"].values
    lower2 = df["vwap_lower_2"].values
    upper15 = df["vwap_upper_1_5"].values if "vwap_upper_1_5" in df.columns else upper2
    lower15 = df["vwap_lower_1_5"].values if "vwap_lower_1_5" in df.columns else lower2
    upper25 = df["vwap_upper_2_5"].values if "vwap_upper_2_5" in df.columns else upper2
    lower25 = df["vwap_lower_2_5"].values if "vwap_lower_2_5" in df.columns else lower2
    rsi = df["rsi_14"].values
    atr = df["atr_14"].values
    prev_close = np.roll(closes, 1)
    prev_close[0] = np.nan

    for i in range(1, n):
        if np.isnan(vwap[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or atr[i] <= 0:
            continue

        conf = 0

        # LONG: price crosses below -2sigma and RSI < 30
        if closes[i] <= lower2[i] and rsi[i] < 30:
            conf = 2  # base: touched band + RSI confirm
            if not np.isnan(lower25[i]) and closes[i] <= lower25[i]:
                conf += 1  # deeper extension
            if prev_close[i] > lower2[i]:
                conf += 1  # fresh cross

            signals[i] = 1
            sl[i] = closes[i] - sl_atr_mult * atr[i]
            tp1[i] = vwap[i]  # mean reversion to VWAP
            tp2[i] = upper15[i] if not np.isnan(upper15[i]) else vwap[i] + 0.5 * (upper2[i] - vwap[i])
            tp3[i] = upper2[i]
            confidence[i] = min(conf, 5)
            continue

        # SHORT: price crosses above +2sigma and RSI > 70
        if closes[i] >= upper2[i] and rsi[i] > 70:
            conf = 2
            if not np.isnan(upper25[i]) and closes[i] >= upper25[i]:
                conf += 1
            if prev_close[i] < upper2[i]:
                conf += 1

            signals[i] = -1
            sl[i] = closes[i] + sl_atr_mult * atr[i]
            tp1[i] = vwap[i]
            tp2[i] = lower15[i] if not np.isnan(lower15[i]) else vwap[i] - 0.5 * (vwap[i] - lower2[i])
            tp3[i] = lower2[i]
            confidence[i] = min(conf, 5)

    df["signal"] = signals
    df["sl_price"] = sl
    df["tp1_price"] = tp1
    df["tp2_price"] = tp2
    df["tp3_price"] = tp3
    df["confidence"] = confidence
    df["strategy_name"] = "S2_Session_VWAP_Reversal"
    df["position"] = signals
    return df


# ---------------------------------------------------------------------------
# S3: Key Level Momentum Breakout
# ---------------------------------------------------------------------------

def generate_signals_s3(df_entry, df_filter=None, params=None):
    """
    Key Level Momentum Breakout.

    Entry TF: H1. Filter TF: H4 (key level identification).
    - Identify H4 key levels (price clusters with 3+ reactions)
    - H1 candle closes beyond key level
    - Volume > 1.5x 20-period average
    - MACD histogram expanding + ADX > 20 rising
    """
    if params is None:
        params = {}

    sl_atr_mult = params.get("sl_atr_multiplier", 1.5)
    vol_mult = params.get("volume_multiplier", 1.5)
    adx_threshold = params.get("adx_threshold", 20)
    kl_lookback = params.get("kl_lookback", 500)

    df = df_entry.copy()
    n = len(df)

    # Identify key levels from H4 data if available, else from H1
    kl_source = df_filter if df_filter is not None and not df_filter.empty else df
    key_levels = _identify_key_levels(kl_source, lookback=kl_lookback)

    if not key_levels:
        df["signal"] = 0
        df["sl_price"] = np.nan
        df["tp1_price"] = np.nan
        df["tp2_price"] = np.nan
        df["tp3_price"] = np.nan
        df["confidence"] = 0
        df["strategy_name"] = "S3_Key_Level_Breakout"
        df["position"] = 0
        return df

    signals = np.zeros(n, dtype=int)
    sl = np.full(n, np.nan)
    tp1 = np.full(n, np.nan)
    tp2 = np.full(n, np.nan)
    tp3 = np.full(n, np.nan)
    confidence = np.zeros(n, dtype=int)

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    volumes = df["volume"].values
    atr = df["atr_14"].values
    prev_close = np.roll(closes, 1)
    prev_close[0] = np.nan

    has_macd = "macd_hist" in df.columns
    has_adx = "adx_14" in df.columns

    macd_hist = df["macd_hist"].values if has_macd else np.full(n, np.nan)
    adx = df["adx_14"].values if has_adx else np.full(n, np.nan)

    # 20-period volume average
    vol_sma = pd.Series(volumes).rolling(20).mean().values

    for i in range(30, n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        tolerance = 0.5 * atr[i]
        kl_prices = [lvl for lvl, _ in key_levels]

        # Find nearest key level
        for kl_price in kl_prices:
            # LONG breakout: price just broke above key level
            if (not np.isnan(prev_close[i]) and prev_close[i] <= kl_price
                    and closes[i] > kl_price + tolerance * 0.25):
                conf = 1

                # Volume confirmation
                if not np.isnan(vol_sma[i]) and vol_sma[i] > 0:
                    if volumes[i] > vol_mult * vol_sma[i]:
                        conf += 1

                # MACD histogram expanding
                if has_macd and i >= 2:
                    if (not np.isnan(macd_hist[i]) and not np.isnan(macd_hist[i - 1])
                            and macd_hist[i] > macd_hist[i - 1] and macd_hist[i] > 0):
                        conf += 1

                # ADX > threshold and rising
                if has_adx and i >= 1:
                    if (not np.isnan(adx[i]) and adx[i] > adx_threshold
                            and not np.isnan(adx[i - 1]) and adx[i] > adx[i - 1]):
                        conf += 1

                if conf >= 2:
                    signals[i] = 1
                    sl[i] = kl_price - sl_atr_mult * atr[i]
                    # TPs at next key levels above
                    above = sorted([p for p in kl_prices if p > kl_price])
                    tp1[i] = above[0] if len(above) > 0 else closes[i] + 2 * atr[i]
                    tp2[i] = above[1] if len(above) > 1 else closes[i] + 3 * atr[i]
                    tp3[i] = above[2] if len(above) > 2 else closes[i] + 4.5 * atr[i]
                    confidence[i] = min(conf, 5)
                    break

            # SHORT breakout: price just broke below key level
            elif (not np.isnan(prev_close[i]) and prev_close[i] >= kl_price
                    and closes[i] < kl_price - tolerance * 0.25):
                conf = 1

                if not np.isnan(vol_sma[i]) and vol_sma[i] > 0:
                    if volumes[i] > vol_mult * vol_sma[i]:
                        conf += 1

                if has_macd and i >= 2:
                    if (not np.isnan(macd_hist[i]) and not np.isnan(macd_hist[i - 1])
                            and macd_hist[i] < macd_hist[i - 1] and macd_hist[i] < 0):
                        conf += 1

                if has_adx and i >= 1:
                    if (not np.isnan(adx[i]) and adx[i] > adx_threshold
                            and not np.isnan(adx[i - 1]) and adx[i] > adx[i - 1]):
                        conf += 1

                if conf >= 2:
                    signals[i] = -1
                    sl[i] = kl_price + sl_atr_mult * atr[i]
                    below = sorted([p for p in kl_prices if p < kl_price], reverse=True)
                    tp1[i] = below[0] if len(below) > 0 else closes[i] - 2 * atr[i]
                    tp2[i] = below[1] if len(below) > 1 else closes[i] - 3 * atr[i]
                    tp3[i] = below[2] if len(below) > 2 else closes[i] - 4.5 * atr[i]
                    confidence[i] = min(conf, 5)
                    break

    df["signal"] = signals
    df["sl_price"] = sl
    df["tp1_price"] = tp1
    df["tp2_price"] = tp2
    df["tp3_price"] = tp3
    df["confidence"] = confidence
    df["strategy_name"] = "S3_Key_Level_Breakout"
    df["position"] = signals
    return df


# ---------------------------------------------------------------------------
# S4: EMA Ribbon Momentum Scalp
# ---------------------------------------------------------------------------

def generate_signals_s4(df_entry, df_filter=None, params=None):
    """
    EMA Ribbon Momentum Scalp.

    Entry TF: M15. Filter TF: H1.
    - H1 filter: EMA 20>50>100>200 fully stacked
    - M15 ribbon compression (EMAs within 8 pips) then re-expansion
    - Stochastic cross from OB/OS
    - ATR contraction during compression
    """
    if params is None:
        params = {}

    sl_atr_mult = params.get("sl_atr_multiplier", 1.5)
    ribbon_pips = params.get("ribbon_compression_pips", 0.0008)  # 8 pips
    compression_bars = params.get("compression_lookback", 5)

    df = df_entry.copy()

    # Merge H1 EMAs for stacking filter
    if df_filter is not None:
        df = _merge_htf(df, df_filter, ["ema_20", "ema_50", "ema_100", "ema_200"], suffix="_htf")

    n = len(df)
    signals = np.zeros(n, dtype=int)
    sl = np.full(n, np.nan)
    tp1 = np.full(n, np.nan)
    tp2 = np.full(n, np.nan)
    tp3 = np.full(n, np.nan)
    confidence = np.zeros(n, dtype=int)

    # Entry TF EMAs
    ema_cols = ["ema_20", "ema_50", "ema_100", "ema_200"]
    has_emas = all(c in df.columns for c in ema_cols)
    if not has_emas:
        df["signal"] = 0
        df["sl_price"] = np.nan
        df["tp1_price"] = np.nan
        df["tp2_price"] = np.nan
        df["tp3_price"] = np.nan
        df["confidence"] = 0
        df["strategy_name"] = "S4_EMA_Ribbon_Scalp"
        df["position"] = 0
        return df

    ema20 = df["ema_20"].values
    ema50 = df["ema_50"].values
    ema100 = df["ema_100"].values
    ema200 = df["ema_200"].values
    closes = df["close"].values
    atr = df["atr_14"].values

    has_stoch = "stoch_k" in df.columns and "stoch_d" in df.columns
    stoch_k = df["stoch_k"].values if has_stoch else np.full(n, 50.0)
    stoch_d = df["stoch_d"].values if has_stoch else np.full(n, 50.0)

    # H1 filter columns
    htf_ema20 = df["ema_20_htf"].values if "ema_20_htf" in df.columns else np.full(n, np.nan)
    htf_ema50 = df["ema_50_htf"].values if "ema_50_htf" in df.columns else np.full(n, np.nan)
    htf_ema100 = df["ema_100_htf"].values if "ema_100_htf" in df.columns else np.full(n, np.nan)
    htf_ema200 = df["ema_200_htf"].values if "ema_200_htf" in df.columns else np.full(n, np.nan)

    for i in range(max(compression_bars + 1, 30), n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        # H1 EMA stacking filter
        htf_bullish_stack = (
            not np.isnan(htf_ema20[i]) and
            htf_ema20[i] > htf_ema50[i] > htf_ema100[i] > htf_ema200[i]
        )
        htf_bearish_stack = (
            not np.isnan(htf_ema20[i]) and
            htf_ema20[i] < htf_ema50[i] < htf_ema100[i] < htf_ema200[i]
        )

        if not (htf_bullish_stack or htf_bearish_stack):
            # If no H1 data, allow signal but lower confidence
            if not np.isnan(htf_ema20[i]):
                continue

        # Check M15 ribbon compression in recent bars
        was_compressed = False
        for j in range(i - compression_bars, i):
            ribbon_range = max(ema20[j], ema50[j], ema100[j], ema200[j]) - \
                          min(ema20[j], ema50[j], ema100[j], ema200[j])
            if ribbon_range <= ribbon_pips:
                was_compressed = True
                break

        if not was_compressed:
            continue

        # Current ribbon must be expanding
        curr_range = max(ema20[i], ema50[i], ema100[i], ema200[i]) - \
                    min(ema20[i], ema50[i], ema100[i], ema200[i])
        prev_range = max(ema20[i - 1], ema50[i - 1], ema100[i - 1], ema200[i - 1]) - \
                    min(ema20[i - 1], ema50[i - 1], ema100[i - 1], ema200[i - 1])
        if curr_range <= prev_range:
            continue

        conf = 1

        # LONG: bullish expansion + stochastic from oversold
        if (htf_bullish_stack or np.isnan(htf_ema20[i])) and ema20[i] > ema50[i]:
            stoch_bull = (has_stoch and not np.isnan(stoch_k[i]) and
                         stoch_k[i] > stoch_d[i] and stoch_k[i] < 50)
            if stoch_bull:
                conf += 1

            # ATR contraction (current ATR < recent average)
            atr_window = atr[max(0, i - 20):i]
            if len(atr_window) > 0 and atr[i] < np.nanmean(atr_window):
                conf += 1

            if htf_bullish_stack:
                conf += 1

            signals[i] = 1
            sl[i] = closes[i] - sl_atr_mult * atr[i]
            tp1[i] = closes[i] + 1.0 * atr[i]
            tp2[i] = closes[i] + 2.0 * atr[i]
            tp3[i] = closes[i] + 3.0 * atr[i]
            confidence[i] = min(conf, 5)
            continue

        # SHORT: bearish expansion + stochastic from overbought
        if (htf_bearish_stack or np.isnan(htf_ema20[i])) and ema20[i] < ema50[i]:
            stoch_bear = (has_stoch and not np.isnan(stoch_k[i]) and
                         stoch_k[i] < stoch_d[i] and stoch_k[i] > 50)
            if stoch_bear:
                conf += 1

            atr_window = atr[max(0, i - 20):i]
            if len(atr_window) > 0 and atr[i] < np.nanmean(atr_window):
                conf += 1

            if htf_bearish_stack:
                conf += 1

            signals[i] = -1
            sl[i] = closes[i] + sl_atr_mult * atr[i]
            tp1[i] = closes[i] - 1.0 * atr[i]
            tp2[i] = closes[i] - 2.0 * atr[i]
            tp3[i] = closes[i] - 3.0 * atr[i]
            confidence[i] = min(conf, 5)

    df["signal"] = signals
    df["sl_price"] = sl
    df["tp1_price"] = tp1
    df["tp2_price"] = tp2
    df["tp3_price"] = tp3
    df["confidence"] = confidence
    df["strategy_name"] = "S4_EMA_Ribbon_Scalp"
    df["position"] = signals
    return df


# ---------------------------------------------------------------------------
# S5: Momentum Exhaustion Reversal
# ---------------------------------------------------------------------------

def generate_signals_s5(df_entry, df_filter=None, params=None):
    """
    Momentum Exhaustion Reversal.

    Entry TF: M15. Filter TF: H1 (structure).
    - RSI divergence (price new extreme, RSI doesn't confirm)
    - MACD histogram shrinking (2+ bars)
    - At H1 key level (within 10 pips)
    - Overextension: price moved 2x ATR from nearest EMA
    - Mutual exclusion with S4 ribbon conditions
    """
    if params is None:
        params = {}

    sl_atr_mult = params.get("sl_atr_multiplier", 1.5)
    divergence_lookback = params.get("divergence_lookback", 14)
    overextension_mult = params.get("overextension_atr_mult", 2.0)
    kl_tolerance_pips = params.get("kl_tolerance_pips", 0.0010)  # 10 pips

    df = df_entry.copy()
    n = len(df)

    # Get H1 key levels for structure
    kl_source = df_filter if df_filter is not None and not df_filter.empty else df
    key_levels = _identify_key_levels(kl_source, min_touches=3, lookback=500)

    signals = np.zeros(n, dtype=int)
    sl = np.full(n, np.nan)
    tp1 = np.full(n, np.nan)
    tp2 = np.full(n, np.nan)
    tp3 = np.full(n, np.nan)
    confidence = np.zeros(n, dtype=int)

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atr = df["atr_14"].values
    rsi = df["rsi_14"].values if "rsi_14" in df.columns else np.full(n, 50.0)

    has_macd = "macd_hist" in df.columns
    macd_hist = df["macd_hist"].values if has_macd else np.full(n, np.nan)

    ema50 = df["ema_50"].values if "ema_50" in df.columns else np.full(n, np.nan)
    ema100 = df["ema_100"].values if "ema_100" in df.columns else np.full(n, np.nan)
    ema200 = df["ema_200"].values if "ema_200" in df.columns else np.full(n, np.nan)

    # Check S4 mutual exclusion: if M15 EMAs are tightly stacked, skip
    ema_cols_exist = all(c in df.columns for c in ["ema_20", "ema_50", "ema_100", "ema_200"])

    for i in range(max(divergence_lookback + 2, 30), n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        # Mutual exclusion: skip if EMA ribbon is compressed (S4 territory)
        if ema_cols_exist:
            ribbon = max(df["ema_20"].iloc[i], df["ema_50"].iloc[i],
                        df["ema_100"].iloc[i], df["ema_200"].iloc[i]) - \
                    min(df["ema_20"].iloc[i], df["ema_50"].iloc[i],
                        df["ema_100"].iloc[i], df["ema_200"].iloc[i])
            if ribbon <= 0.0008:  # compressed ribbon = S4 territory
                continue

        # Check overextension from nearest EMA
        nearest_ema = np.nan
        for ema in [ema50[i], ema100[i], ema200[i]]:
            if not np.isnan(ema):
                if np.isnan(nearest_ema) or abs(closes[i] - ema) < abs(closes[i] - nearest_ema):
                    nearest_ema = ema

        if np.isnan(nearest_ema):
            continue

        dist_from_ema = abs(closes[i] - nearest_ema)
        overextended = dist_from_ema >= overextension_mult * atr[i]

        if not overextended:
            continue

        # Check at key level
        at_kl = False
        if key_levels:
            kl = _nearest_key_level(closes[i], key_levels, kl_tolerance_pips)
            at_kl = kl is not None

        # RSI divergence detection
        lb_start = max(0, i - divergence_lookback)

        # Bearish divergence: price makes new high but RSI doesn't
        price_new_high = highs[i] >= np.max(highs[lb_start:i])
        rsi_lower_high = rsi[i] < np.max(rsi[lb_start:i]) if not np.isnan(rsi[i]) else False
        bearish_div = price_new_high and rsi_lower_high

        # Bullish divergence: price makes new low but RSI doesn't
        price_new_low = lows[i] <= np.min(lows[lb_start:i])
        rsi_higher_low = rsi[i] > np.min(rsi[lb_start:i]) if not np.isnan(rsi[i]) else False
        bullish_div = price_new_low and rsi_higher_low

        # MACD histogram shrinking (2+ bars)
        macd_shrinking_bull = False
        macd_shrinking_bear = False
        if has_macd and i >= 2:
            if (not np.isnan(macd_hist[i]) and not np.isnan(macd_hist[i - 1])
                    and not np.isnan(macd_hist[i - 2])):
                # Bearish: positive histogram shrinking
                if macd_hist[i - 2] > macd_hist[i - 1] > macd_hist[i] > 0:
                    macd_shrinking_bear = True
                # Bullish: negative histogram shrinking (getting less negative)
                if macd_hist[i - 2] < macd_hist[i - 1] < macd_hist[i] < 0:
                    macd_shrinking_bull = True

        # LONG reversal: price overextended below EMA, bullish divergence
        if closes[i] < nearest_ema and bullish_div:
            conf = 1
            if macd_shrinking_bull:
                conf += 1
            if at_kl:
                conf += 1
            if rsi[i] < 30:
                conf += 1
            if overextension_mult >= 2.5:
                conf += 1

            if conf >= 2:
                signals[i] = 1
                sl[i] = closes[i] - sl_atr_mult * atr[i]
                tp1[i] = nearest_ema
                tp2[i] = closes[i] + 2.0 * atr[i]
                tp3[i] = closes[i] + 3.5 * atr[i]
                confidence[i] = min(conf, 5)
                continue

        # SHORT reversal: price overextended above EMA, bearish divergence
        if closes[i] > nearest_ema and bearish_div:
            conf = 1
            if macd_shrinking_bear:
                conf += 1
            if at_kl:
                conf += 1
            if rsi[i] > 70:
                conf += 1
            if overextension_mult >= 2.5:
                conf += 1

            if conf >= 2:
                signals[i] = -1
                sl[i] = closes[i] + sl_atr_mult * atr[i]
                tp1[i] = nearest_ema
                tp2[i] = closes[i] - 2.0 * atr[i]
                tp3[i] = closes[i] - 3.5 * atr[i]
                confidence[i] = min(conf, 5)

    df["signal"] = signals
    df["sl_price"] = sl
    df["tp1_price"] = tp1
    df["tp2_price"] = tp2
    df["tp3_price"] = tp3
    df["confidence"] = confidence
    df["strategy_name"] = "S5_Momentum_Exhaustion"
    df["position"] = signals
    return df


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY = {
    "S1": {
        "name": "S1_MA_Breakout_Retest",
        "func": generate_signals_s1,
        "pairs": ["GBP_AUD", "EUR_AUD", "EUR_CAD", "EUR_NZD"],
        "entry_tf": "M15",
        "filter_tf": "H1",
        "tp_splits": [0.40, 0.40, 0.20],  # TP1: 40%, TP2: 40%, TP3: 20% runner
        "trail_atr_mult": 1.0,
        "max_bars": 96,  # 24 hours on M15
    },
    "S2": {
        "name": "S2_Session_VWAP_Reversal",
        "func": generate_signals_s2,
        "pairs": ["GBP_USD", "EUR_USD", "GBP_JPY", "USD_JPY"],
        "entry_tf": "M15",
        "filter_tf": None,
        "tp_splits": [0.50, 0.35, 0.15],
        "trail_atr_mult": 0.75,
        "max_bars": 48,  # 12 hours on M15
    },
    "S3": {
        "name": "S3_Key_Level_Breakout",
        "func": generate_signals_s3,
        "pairs": ["GBP_JPY", "USD_JPY", "GBP_USD", "EUR_GBP"],
        "entry_tf": "H1",
        "filter_tf": "H4",
        "tp_splits": [0.70, 0.30, 0.0],  # No TP3 runner
        "trail_atr_mult": 0.0,
        "max_bars": 48,  # 48 hours on H1
    },
    "S4": {
        "name": "S4_EMA_Ribbon_Scalp",
        "func": generate_signals_s4,
        "pairs": ["GBP_AUD", "EUR_AUD", "EUR_GBP"],
        "entry_tf": "M15",
        "filter_tf": "H1",
        "tp_splits": [0.50, 0.35, 0.15],
        "trail_atr_mult": 0.5,
        "max_bars": 32,  # 8 hours on M15
    },
    "S5": {
        "name": "S5_Momentum_Exhaustion",
        "func": generate_signals_s5,
        "pairs": ["GBP_AUD", "EUR_AUD", "EUR_GBP", "GBP_CAD", "EUR_CAD"],
        "entry_tf": "M15",
        "filter_tf": "H1",
        "tp_splits": [0.40, 0.40, 0.20],
        "trail_atr_mult": 1.0,
        "max_bars": 64,  # 16 hours on M15
    },
}
