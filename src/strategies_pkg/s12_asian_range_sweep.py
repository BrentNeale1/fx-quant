"""
Strategy S12: Asian Range Sweep (Mean Reversion).

Concept: Price sweeps past the Asian session (00:00-07:00 UTC) high or low
at London open, triggering clustered stop-loss orders, then reverses back
into the range. This is a classic liquidity sweep pattern on the session level.

Adapted to M15 from original 5-minute specification. The Asian range is well-
defined on M15 (28 bars = 7 hours), giving clean levels.

Entry conditions (ALL must be true):
  1. Asian range established (00:00-07:00 UTC, min 3 bars)
  2. Asian range not too wide (< 2.0 ATR H1)
  3. Price sweeps past Asian high/low by 0.3-1.5 ATR
  4. Price closes back inside the Asian range (reversal candle)
  5. Session: 07:00-10:00 UTC (London kill zone)
  6. HTF trend alignment (soft confluence)

Exit:
  - SL: Beyond sweep extreme + 0.5 ATR buffer
  - TP1: Asian range midpoint (close 50%)
  - TP2: Opposite side of Asian range (close 50%)
  - Max hold: 40 bars (~10 hours on M15)
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S12_AsianRangeSweep(BaseStrategy):
    strategy_id = 12
    name = "S12_AsianRangeSweep"

    # Asian session window (UTC)
    ASIAN_START_HOUR = 0
    ASIAN_END_HOUR = 7

    # Entry window (UTC)
    ENTRY_START_HOUR = 7
    ENTRY_END_HOUR = 10

    # Sweep parameters
    SWEEP_MIN_ATR = 0.3   # Min penetration past Asian range
    SWEEP_MAX_ATR = 1.5   # Max penetration (beyond = genuine breakout)

    # Risk management
    SL_ATR_BUFFER = 0.5   # Buffer beyond sweep extreme
    TP_RR_MULT = 1.5      # Minimum RR filter

    MAX_BARS = 40
    LOOKBACK_BARS = 60  # How far back to search for Asian session bars

    def __init__(self):
        super().__init__()
        self._asian_range_cache = {}

    def _compute_asian_range(self, data, idx):
        """Compute Asian session range for the current day."""
        current_time = data.index[idx]
        current_date = current_time.date()

        if current_date in self._asian_range_cache:
            return self._asian_range_cache[current_date]

        asian_bars = []
        for i in range(max(0, idx - self.LOOKBACK_BARS), idx + 1):
            bar_time = data.index[i]
            if bar_time.date() != current_date:
                continue
            if self.ASIAN_START_HOUR <= bar_time.hour < self.ASIAN_END_HOUR:
                asian_bars.append(i)

        if len(asian_bars) < 3:
            return None

        asian_data = data.iloc[asian_bars]
        asian_high = asian_data["high"].max()
        asian_low = asian_data["low"].min()
        asian_mid = (asian_high + asian_low) / 2
        asian_avg_vol = asian_data["volume"].mean()

        result = (asian_high, asian_low, asian_mid, asian_avg_vol)
        self._asian_range_cache[current_date] = result
        return result

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        if idx < 200:
            return None

        # Entry window: 07:00-10:00 UTC
        hour = current.name.hour if hasattr(current.name, 'hour') else 0
        if hour < self.ENTRY_START_HOUR or hour >= self.ENTRY_END_HOUR:
            return None

        atr_val = current.get("atr_14", 0)
        if atr_val <= 0 or np.isnan(atr_val):
            return None

        # Get Asian range
        asian = self._compute_asian_range(data, idx)
        if asian is None:
            return None

        asian_high, asian_low, asian_mid, asian_avg_vol = asian
        asian_range = asian_high - asian_low

        if asian_range <= 0:
            return None

        # Skip if Asian range is too wide (> 2.0 ATR)
        htf_atr = atr_val
        if htf_row is not None:
            htf_atr_val = htf_row.get("atr_14", np.nan)
            if not np.isnan(htf_atr_val) and htf_atr_val > 0:
                htf_atr = htf_atr_val

        if asian_range > 2.0 * htf_atr:
            return None

        price = current["close"]
        candle_high = current["high"]
        candle_low = current["low"]
        candle_range = candle_high - candle_low
        if candle_range <= 0:
            return None

        # HTF trend (soft confluence)
        htf_aligned_long = False
        htf_aligned_short = False
        if htf_row is not None:
            htf_ema200 = htf_row.get("ema_200", np.nan)
            htf_close = htf_row.get("close", np.nan)
            if not np.isnan(htf_ema200) and not np.isnan(htf_close):
                htf_aligned_long = htf_close > htf_ema200
                htf_aligned_short = htf_close < htf_ema200

        # RSI
        rsi_val = current.get("rsi_14", 50)
        if np.isnan(rsi_val):
            rsi_val = 50

        # ---- BULLISH SWEEP: price swept below Asian low, closed back inside ----
        sweep_below = candle_low < asian_low
        penetration_below = asian_low - candle_low
        closed_inside_from_below = price > asian_low

        if (sweep_below and closed_inside_from_below
                and self.SWEEP_MIN_ATR * atr_val <= penetration_below <= self.SWEEP_MAX_ATR * atr_val):
            # Strong reversal candle: close in upper 40% of range
            if (price - candle_low) / candle_range < 0.4:
                return None

            sweep_extreme = candle_low
            sl = sweep_extreme - self.SL_ATR_BUFFER * atr_val
            sl_dist = price - sl

            # TP1: Asian midpoint, TP2: Asian high
            tp1 = asian_mid
            tp2 = asian_high

            # Min RR check
            tp1_dist = tp1 - price
            if sl_dist > 0 and tp1_dist / sl_dist < 1.0:
                return None

            confluence = 3
            if htf_aligned_long:
                confluence += 1
            if rsi_val < 35:
                confluence += 1
            vol = current.get("volume", 0)
            if asian_avg_vol > 0 and vol > 1.5 * asian_avg_vol:
                confluence += 1

            return {
                "direction": "LONG",
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp2,
                "confluence": confluence,
                "entry_pattern": "asian_sweep_bullish",
                "tp_splits": (0.50, 0.50, 0.0),
                "trail_atr_mult": 1.0,
                "max_bars": self.MAX_BARS,
            }

        # ---- BEARISH SWEEP: price swept above Asian high, closed back inside ----
        sweep_above = candle_high > asian_high
        penetration_above = candle_high - asian_high
        closed_inside_from_above = price < asian_high

        if (sweep_above and closed_inside_from_above
                and self.SWEEP_MIN_ATR * atr_val <= penetration_above <= self.SWEEP_MAX_ATR * atr_val):
            # Strong reversal candle: close in lower 40% of range
            if (candle_high - price) / candle_range < 0.4:
                return None

            sweep_extreme = candle_high
            sl = sweep_extreme + self.SL_ATR_BUFFER * atr_val
            sl_dist = sl - price

            tp1 = asian_mid
            tp2 = asian_low

            tp1_dist = price - tp1
            if sl_dist > 0 and tp1_dist / sl_dist < 1.0:
                return None

            confluence = 3
            if htf_aligned_short:
                confluence += 1
            if rsi_val > 65:
                confluence += 1
            vol = current.get("volume", 0)
            if asian_avg_vol > 0 and vol > 1.5 * asian_avg_vol:
                confluence += 1

            return {
                "direction": "SHORT",
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp2,
                "confluence": confluence,
                "entry_pattern": "asian_sweep_bearish",
                "tp_splits": (0.50, 0.50, 0.0),
                "trail_atr_mult": 1.0,
                "max_bars": self.MAX_BARS,
            }

        return None
