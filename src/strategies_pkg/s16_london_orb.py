"""
Strategy S16: London Open Range Breakout (ORB).

Concept: Define the range of the first 30 minutes of the London session
(07:00-07:30 UTC = 2 M15 bars), then trade the breakout with volume
confirmation. Well-established pattern based on institutional order flow
at London open.

Adapted to M15 from original 5-minute specification (0.3-0.7 trades/day target).

Entry conditions (ALL must be true):
  1. Opening range defined: 07:00-07:30 UTC high/low (2 M15 bars)
  2. Range not too wide (< 1.5 ATR H1) and not too narrow (> 0.3 ATR)
  3. Price breaks and closes above/below opening range
  4. Volume > 1.3x opening range average
  5. Entry window: 07:30-12:00 UTC
  6. ADX > 15 (some directional context)
  7. HTF trend alignment (soft)

Exit:
  - SL: Opposite side of opening range + 0.3 ATR buffer
  - TP1: 1x opening range extension (close 50%)
  - TP2: 2x opening range extension (close 50%)
  - Time exit: 17:00 UTC
  - Max hold: 40 bars
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S16_LondonORB(BaseStrategy):
    strategy_id = 16
    name = "S16_LondonORB"

    # Opening range definition
    ORB_START_HOUR = 7
    ORB_START_MINUTE = 0
    ORB_BARS = 1             # 1 M15 bar = 15 min range (tighter on M15)

    # Entry window
    ENTRY_END_HOUR = 12      # Can enter until noon UTC

    # Range filters
    MIN_RANGE_ATR = 0.1      # Min range (too narrow = noise)
    MAX_RANGE_ATR = 2.0      # Max range (relaxed for M15 adaptation)

    # Volume
    VOLUME_MULT = 1.0        # Relaxed (M15 volume patterns differ from M5)

    # Risk management
    SL_ATR_MULT = 1.5        # SL: 1.5x ATR from entry (tighter than opposite side)
    TP1_MULT = 1.0           # TP1 = 1x range extension from breakout level
    TP2_MULT = 2.0           # TP2 = 2x range extension

    # Filters
    MIN_ADX = 15

    MAX_BARS = 40

    def __init__(self):
        super().__init__()
        self._orb_cache = {}
        self._daily_trade_cache = {}

    def _compute_opening_range(self, data, idx):
        """Compute London opening range (first 2 M15 bars at 07:00)."""
        current_time = data.index[idx]
        current_date = current_time.date()

        if current_date in self._orb_cache:
            return self._orb_cache[current_date]

        # Find the first ORB_BARS bars starting at 07:00
        orb_bars = []
        for i in range(max(0, idx - 40), idx + 1):
            bar_time = data.index[i]
            if bar_time.date() != current_date:
                continue
            if bar_time.hour == self.ORB_START_HOUR:
                orb_bars.append(i)
                if len(orb_bars) >= self.ORB_BARS:
                    break

        if len(orb_bars) < self.ORB_BARS:
            return None

        orb_data = data.iloc[orb_bars]
        orb_high = orb_data["high"].max()
        orb_low = orb_data["low"].min()
        orb_avg_vol = orb_data["volume"].mean()

        result = (orb_high, orb_low, orb_avg_vol)
        self._orb_cache[current_date] = result
        return result

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        if idx < 200:
            return None

        current_time = current.name
        hour = current_time.hour if hasattr(current_time, 'hour') else 0
        minute = current_time.minute if hasattr(current_time, 'minute') else 0

        # Entry window: after ORB defined (07:15) until noon
        # The first valid entry is the 07:15 bar (2nd bar of the session)
        if hour < self.ORB_START_HOUR:
            return None
        if hour == self.ORB_START_HOUR and minute < 15:
            return None  # ORB still forming
        if hour >= self.ENTRY_END_HOUR:
            return None

        # One trade per day
        current_date = current_time.date()
        if current_date in self._daily_trade_cache:
            return None

        atr_val = current.get("atr_14", 0)
        if atr_val <= 0 or np.isnan(atr_val):
            return None

        # Get opening range
        orb = self._compute_opening_range(data, idx)
        if orb is None:
            return None

        orb_high, orb_low, orb_avg_vol = orb
        orb_range = orb_high - orb_low

        if orb_range <= 0:
            return None

        # Range size filter
        htf_atr = atr_val
        if htf_row is not None:
            htf_atr_val = htf_row.get("atr_14", np.nan)
            if not np.isnan(htf_atr_val) and htf_atr_val > 0:
                htf_atr = htf_atr_val

        if orb_range < self.MIN_RANGE_ATR * htf_atr:
            return None
        if orb_range > self.MAX_RANGE_ATR * htf_atr:
            return None

        price = current["close"]

        # Volume confirmation
        vol = current.get("volume", 0)
        if orb_avg_vol <= 0 or vol < self.VOLUME_MULT * orb_avg_vol:
            return None

        # ADX filter
        adx_val = current.get("adx_14", 0)
        if np.isnan(adx_val):
            adx_val = 0
        if adx_val < self.MIN_ADX:
            return None

        # Direction: breakout above or below ORB
        direction = None
        if price > orb_high:
            direction = "LONG"
        elif price < orb_low:
            direction = "SHORT"

        if direction is None:
            return None

        # HTF alignment (soft)
        htf_aligned = False
        if htf_row is not None:
            htf_ema200 = htf_row.get("ema_200", np.nan)
            htf_close = htf_row.get("close", np.nan)
            if not np.isnan(htf_ema200) and not np.isnan(htf_close):
                if direction == "LONG" and htf_close > htf_ema200:
                    htf_aligned = True
                elif direction == "SHORT" and htf_close < htf_ema200:
                    htf_aligned = True

        self._daily_trade_cache[current_date] = True

        # SL / TP — ATR-based SL (tighter than opposite side of range)
        if direction == "LONG":
            sl = price - self.SL_ATR_MULT * atr_val
            tp1 = price + self.TP1_MULT * orb_range
            tp2 = price + self.TP2_MULT * orb_range
        else:
            sl = price + self.SL_ATR_MULT * atr_val
            tp1 = price - self.TP1_MULT * orb_range
            tp2 = price - self.TP2_MULT * orb_range

        # Min RR check
        sl_dist = abs(price - sl)
        tp1_dist = abs(tp1 - price)
        if sl_dist <= 0 or tp1_dist / sl_dist < 0.8:
            return None

        # Confluence
        confluence = 3  # Base: ORB breakout + volume + ADX
        if htf_aligned:
            confluence += 1
        if adx_val > 25:
            confluence += 1

        # MACD confirmation
        macd_hist = current.get("macd_hist", 0)
        if not np.isnan(macd_hist):
            if direction == "LONG" and macd_hist > 0:
                confluence += 1
            elif direction == "SHORT" and macd_hist < 0:
                confluence += 1

        return {
            "direction": direction,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp2,
            "confluence": confluence,
            "entry_pattern": f"london_orb_{direction.lower()}",
            "tp_splits": (0.50, 0.50, 0.0),
            "trail_atr_mult": 1.5,
            "max_bars": self.MAX_BARS,
        }
