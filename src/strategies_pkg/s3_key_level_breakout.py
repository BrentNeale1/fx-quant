"""
Strategy 3: Key Level Momentum Breakout.

Entry TF: H1. Key levels identified from swing point clusters.

Entry conditions (LONG):
  - H1 candle closes above a key level (horizontal S/R with 3+ touches)
  - Volume spike: current volume > 1.5x 20-bar average
  - Strong close: candle body > 50% of range (conviction candle)
  - MACD histogram same sign as direction
  - ADX > 20 (trending market)
  - Session: London + NY overlap (08:00-16:00 UTC)

SL: Back inside key level — level_price -/+ 0.5x ATR
TP1: 1.5x ATR, TP2: 2.5x ATR, TP3: 4x ATR
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy
from ..indicators.technical import identify_key_levels


class S3_KeyLevel_Breakout(BaseStrategy):
    strategy_id = 3
    name = "S3_Key_Level_Breakout"

    # Tunable parameters (defaults match original hardcoded values)
    BODY_RATIO_MIN = 0.50
    VOLUME_MULT = 1.5
    SL_ATR_MULT = 0.5
    TP1_ATR_MULT = 1.5
    TP2_ATR_MULT = 2.5
    TP3_ATR_MULT = 4.0
    MIN_ADX = 20
    KEY_LEVEL_TOLERANCE = 0.75
    KEY_LEVEL_MIN_TOUCHES = 3

    def __init__(self):
        super().__init__()
        self._cached_levels = None
        self._cache_idx = -1

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        if idx < 100:
            return None

        # Session filter: London + NY overlap (08:00-16:00 UTC)
        hour = current.name.hour if hasattr(current.name, 'hour') else 0
        if hour < 8 or hour >= 16:
            return None

        atr_val = current.get("atr_14", 0)
        if atr_val <= 0 or np.isnan(atr_val):
            return None

        # ADX filter: require trending market
        adx_val = current.get("adx_14", 0)
        if adx_val < self.MIN_ADX:
            return None

        # Strong close: candle body > 50% of range
        close = current["close"]
        body = abs(close - current["open"])
        full_range = current["high"] - current["low"]
        if full_range <= 0 or body / full_range < self.BODY_RATIO_MIN:
            return None

        # Volume spike: current volume > 1.5x 20-bar average
        vol = current.get("volume", 0)
        if vol > 0 and idx >= 20:
            vol_avg = data["volume"].iloc[idx - 20:idx].mean()
            if vol_avg > 0 and vol < self.VOLUME_MULT * vol_avg:
                return None

        prev_close = data.iloc[idx - 1]["close"]

        # MACD
        macd_h = current.get("macd_hist", 0)

        # Recalculate key levels every 20 bars using larger lookback
        if self._cached_levels is None or idx - self._cache_idx >= 20:
            start = max(0, idx - 1000)
            window = data.iloc[start:idx]  # exclude current bar
            self._cached_levels = identify_key_levels(
                window, lookback=5,
                tolerance_atr_mult=self.KEY_LEVEL_TOLERANCE,
                min_touches=self.KEY_LEVEL_MIN_TOUCHES,
            )
            self._cache_idx = idx

        if not self._cached_levels:
            return None

        for level_price, touch_count in self._cached_levels:
            tolerance = 0.3 * atr_val

            # LONG breakout: close above level, prev close was at or below
            if close > level_price + tolerance and prev_close <= level_price + tolerance:
                if macd_h <= 0:
                    continue

                # EMA alignment: 50 > 200 for LONG
                ema_50 = current.get("ema_50", 0)
                ema_200 = current.get("ema_200", 0)
                if ema_50 and ema_200 and ema_50 <= ema_200:
                    continue

                confluence = self._calc_confluence(current, data, idx,
                                                   "LONG", touch_count, vol)

                sl = level_price - self.SL_ATR_MULT * atr_val
                tp1 = close + self.TP1_ATR_MULT * atr_val
                tp2 = close + self.TP2_ATR_MULT * atr_val
                tp3 = close + self.TP3_ATR_MULT * atr_val

                return {
                    "direction": "LONG",
                    "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
                    "confluence": confluence,
                    "entry_pattern": "key_level_break",
                    "tp_splits": (0.40, 0.40, 0.20),
                    "trail_atr_mult": 2.0,
                    "max_bars": 150,
                }

            # SHORT breakout: close below level, prev close was at or above
            if close < level_price - tolerance and prev_close >= level_price - tolerance:
                if macd_h >= 0:
                    continue

                # EMA alignment: 50 < 200 for SHORT
                ema_50 = current.get("ema_50", 0)
                ema_200 = current.get("ema_200", 0)
                if ema_50 and ema_200 and ema_50 >= ema_200:
                    continue

                confluence = self._calc_confluence(current, data, idx,
                                                   "SHORT", touch_count, vol)

                sl = level_price + self.SL_ATR_MULT * atr_val
                tp1 = close - self.TP1_ATR_MULT * atr_val
                tp2 = close - self.TP2_ATR_MULT * atr_val
                tp3 = close - self.TP3_ATR_MULT * atr_val

                return {
                    "direction": "SHORT",
                    "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
                    "confluence": confluence,
                    "entry_pattern": "key_level_break",
                    "tp_splits": (0.40, 0.40, 0.20),
                    "trail_atr_mult": 2.0,
                    "max_bars": 150,
                }

        return None

    def _calc_confluence(self, current, data, idx, direction, touch_count, vol):
        confluence = 1  # breakout confirmed

        # More touches = stronger level
        if touch_count >= 3:
            confluence += 1
        if touch_count >= 5:
            confluence += 1

        # Volume spike strength (>2x avg = extra point)
        if vol > 0 and idx >= 20:
            vol_avg = data["volume"].iloc[idx - 20:idx].mean()
            if vol_avg > 0 and vol > 2.0 * vol_avg:
                confluence += 1

        rsi = current.get("rsi_14", 50)
        if direction == "LONG" and 50 < rsi < 75:
            confluence += 1
        elif direction == "SHORT" and 25 < rsi < 50:
            confluence += 1

        return min(confluence, 5)
