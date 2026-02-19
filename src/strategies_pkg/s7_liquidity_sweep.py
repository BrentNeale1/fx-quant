"""
Strategy S7: Liquidity Sweep Reversal.

Concept: Price sweeps past a significant swing high/low (triggering clustered
stop-loss orders), then reverses. The most empirically-supported SMC concept
(Osler 2005, NY Fed).

Entry conditions (ALL must be true):
  1. Identify significant swing high/low (5-bar fractal) within last 100 bars
  2. Price penetrates the swing level by 0.7-1.0 ATR (the sweep)
  3. Price closes back inside the previous range (reversal candle)
  4. OBV divergence: price makes new extreme but OBV doesn't confirm
     (institutional absorption signal)
  5. HTF (H1) trend alignment: only take sweeps in the direction of
     the higher-timeframe trend (200 EMA bias)
  6. Session filter: London/NY hours (08:00-17:00 UTC)
  7. RSI < 35 (for longs) or > 65 (for shorts) as a filter

Exit:
  - SL: 1.5 ATR beyond the sweep extreme
  - TP1: 1.5 ATR from entry (close 50%)
  - TP2: 3.0 ATR from entry (close 50%)
  - Max hold: 40 bars (H1 = ~40 hours)
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S7_Liquidity_Sweep(BaseStrategy):
    strategy_id = 7
    name = "S7_Liquidity_Sweep"

    # Tunable parameters
    SWING_LOOKBACK = 5        # N-bar fractal for swing detection
    SWING_HISTORY = 100       # How far back to search for swing levels
    SWEEP_MIN_ATR = 0.7       # Minimum penetration for a sweep
    SWEEP_MAX_ATR = 1.5       # Above this = genuine breakout, not a sweep
    SL_ATR_MULT = 1.0         # SL beyond sweep extreme (tightened from 1.5)
    TP1_ATR_MULT = 1.5        # First take-profit
    TP2_ATR_MULT = 3.0        # Second take-profit
    OBV_LOOKBACK = 20         # Lookback for OBV divergence detection
    MAX_BARS = 40

    def _find_swing_levels(self, data, idx):
        """Find significant swing highs and lows within lookback window."""
        start = max(0, idx - self.SWING_HISTORY)
        # Exclude the very recent bars (last 3) to avoid detecting current price action
        end = idx - 2
        if end - start < 20:
            return [], []

        swing_highs = []
        swing_lows = []
        lb = self.SWING_LOOKBACK

        for i in range(start + lb, end - lb + 1):
            # Swing high: highest high in [i-lb, i+lb]
            window_highs = data["high"].iloc[i - lb:i + lb + 1]
            if data["high"].iloc[i] == window_highs.max():
                swing_highs.append((i, data["high"].iloc[i]))

            # Swing low: lowest low in [i-lb, i+lb]
            window_lows = data["low"].iloc[i - lb:i + lb + 1]
            if data["low"].iloc[i] == window_lows.min():
                swing_lows.append((i, data["low"].iloc[i]))

        return swing_highs, swing_lows

    def _find_equal_levels(self, levels, atr_val):
        """Find clusters of equal highs/lows (within 0.1 ATR) — highest probability targets."""
        if len(levels) < 2:
            return levels
        tolerance = 0.1 * atr_val
        clustered = []
        used = set()
        for i, (idx_i, price_i) in enumerate(levels):
            if i in used:
                continue
            cluster = [(idx_i, price_i)]
            used.add(i)
            for j, (idx_j, price_j) in enumerate(levels):
                if j in used:
                    continue
                if abs(price_j - price_i) <= tolerance:
                    cluster.append((idx_j, price_j))
                    used.add(j)
            if len(cluster) >= 2:
                # Use the average price for the cluster, latest index
                avg_price = np.mean([p for _, p in cluster])
                latest_idx = max(idx for idx, _ in cluster)
                clustered.append((latest_idx, avg_price))
            else:
                clustered.append((idx_i, price_i))
        return clustered

    def _check_obv_divergence(self, data, idx, direction):
        """Check for OBV divergence (institutional absorption signal)."""
        lb = self.OBV_LOOKBACK
        if idx < lb:
            return False

        window = data.iloc[idx - lb:idx + 1]
        obv_vals = window.get("obv")
        if obv_vals is None:
            return False

        if direction == "LONG":
            # Bullish OBV divergence: price makes lower low but OBV makes higher low
            price_lows = window["low"]
            recent_low_pos = price_lows.values.argmin()
            if recent_low_pos < lb - 5:
                return False  # Low isn't recent enough
            # Find previous low in first half of window
            first_half = price_lows.iloc[:lb // 2]
            if len(first_half) < 3:
                return False
            prev_low_pos = first_half.values.argmin()
            if (price_lows.iloc[recent_low_pos] < first_half.iloc[prev_low_pos] and
                    obv_vals.iloc[recent_low_pos] > obv_vals.iloc[prev_low_pos]):
                return True
        else:
            # Bearish OBV divergence: price makes higher high but OBV makes lower high
            price_highs = window["high"]
            recent_high_pos = price_highs.values.argmax()
            if recent_high_pos < lb - 5:
                return False
            first_half = price_highs.iloc[:lb // 2]
            if len(first_half) < 3:
                return False
            prev_high_pos = first_half.values.argmax()
            if (price_highs.iloc[recent_high_pos] > first_half.iloc[prev_high_pos] and
                    obv_vals.iloc[recent_high_pos] < obv_vals.iloc[prev_high_pos]):
                return True
        return False

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        if idx < 200:
            return None

        # Session filter: 08:00-17:00 UTC
        hour = current.name.hour if hasattr(current.name, 'hour') else 0
        if hour < 8 or hour >= 17:
            return None

        atr_val = current.get("atr_14", 0)
        if atr_val <= 0 or np.isnan(atr_val):
            return None

        # HTF trend alignment
        if htf_row is None:
            return None
        htf_ema200 = htf_row.get("ema_200", np.nan)
        htf_close = htf_row.get("close", np.nan)
        if np.isnan(htf_ema200) or np.isnan(htf_close):
            return None

        htf_bullish = htf_close > htf_ema200
        htf_bearish = htf_close < htf_ema200

        price = current["close"]
        candle_high = current["high"]
        candle_low = current["low"]

        # RSI as confluence signal (not hard gate — lesson from S4)
        rsi_val = current.get("rsi_14", 50)
        if np.isnan(rsi_val):
            rsi_val = 50

        # Find swing levels
        swing_highs, swing_lows = self._find_swing_levels(data, idx)

        # ---- CHECK FOR BULLISH SWEEP (sweep below swing low, then reverse up) ----
        if htf_bullish:
            swing_lows = self._find_equal_levels(swing_lows, atr_val)
            for sw_idx, sw_price in reversed(swing_lows):  # Check most recent first
                penetration = sw_price - candle_low
                if penetration < self.SWEEP_MIN_ATR * atr_val:
                    continue
                if penetration > self.SWEEP_MAX_ATR * atr_val:
                    continue
                # Reversal confirmation: close back above the swing level
                if price <= sw_price:
                    continue
                # Strong close: in upper 40% of candle range
                candle_range = candle_high - candle_low
                if candle_range <= 0:
                    continue
                if (price - candle_low) / candle_range < 0.4:
                    continue

                # OBV divergence check (bonus confluence, not hard gate)
                obv_div = self._check_obv_divergence(data, idx, "LONG")
                confluence = 3 + (1 if obv_div else 0)

                # RSI in oversold zone adds confluence (soft, not hard gate)
                if rsi_val < 40:
                    confluence += 1

                # Volume confirmation
                vol = current.get("volume", 0)
                vol_avg = data["volume"].iloc[max(0, idx - 20):idx].mean()
                if vol_avg > 0 and vol > 1.5 * vol_avg:
                    confluence += 1

                sweep_extreme = candle_low
                sl = sweep_extreme - self.SL_ATR_MULT * atr_val
                tp1 = price + self.TP1_ATR_MULT * atr_val
                tp2 = price + self.TP2_ATR_MULT * atr_val

                return {
                    "direction": "LONG",
                    "sl": sl,
                    "tp1": tp1,
                    "tp2": tp2,
                    "tp3": tp2,
                    "confluence": confluence,
                    "entry_pattern": "liquidity_sweep_bullish",
                    "tp_splits": (0.50, 0.50, 0.0),
                    "trail_atr_mult": 1.5,
                    "max_bars": self.MAX_BARS,
                }

        # ---- CHECK FOR BEARISH SWEEP (sweep above swing high, then reverse down) ----
        if htf_bearish:
            swing_highs = self._find_equal_levels(swing_highs, atr_val)
            for sw_idx, sw_price in reversed(swing_highs):
                penetration = candle_high - sw_price
                if penetration < self.SWEEP_MIN_ATR * atr_val:
                    continue
                if penetration > self.SWEEP_MAX_ATR * atr_val:
                    continue
                # Reversal: close back below the swing level
                if price >= sw_price:
                    continue
                # Strong close: in lower 40% of candle range
                candle_range = candle_high - candle_low
                if candle_range <= 0:
                    continue
                if (candle_high - price) / candle_range < 0.4:
                    continue

                obv_div = self._check_obv_divergence(data, idx, "SHORT")
                confluence = 3 + (1 if obv_div else 0)

                if rsi_val > 60:
                    confluence += 1

                vol = current.get("volume", 0)
                vol_avg = data["volume"].iloc[max(0, idx - 20):idx].mean()
                if vol_avg > 0 and vol > 1.5 * vol_avg:
                    confluence += 1

                sweep_extreme = candle_high
                sl = sweep_extreme + self.SL_ATR_MULT * atr_val
                tp1 = price - self.TP1_ATR_MULT * atr_val
                tp2 = price - self.TP2_ATR_MULT * atr_val

                return {
                    "direction": "SHORT",
                    "sl": sl,
                    "tp1": tp1,
                    "tp2": tp2,
                    "tp3": tp2,
                    "confluence": confluence,
                    "entry_pattern": "liquidity_sweep_bearish",
                    "tp_splits": (0.50, 0.50, 0.0),
                    "trail_atr_mult": 1.5,
                    "max_bars": self.MAX_BARS,
                }

        return None
