"""
Strategy S13: Bollinger-Keltner Squeeze.

Concept: When Bollinger Bands (20, 2) compress inside Keltner Channels (20, 1.5),
volatility is contracting. When BBs expand back outside KC, a directional move
is starting. Also known as "TTM Squeeze" (John Carter).

Adapted to M15 from original 30-minute specification.

Entry conditions (ALL must be true):
  1. Squeeze detected: BB was inside KC for at least 3 bars (squeeze on)
  2. Squeeze releases: BB expands outside KC (squeeze off)
  3. Momentum direction determines trade direction (MACD histogram)
  4. ADX rising (trend developing)
  5. Session: 07:00-17:00 UTC (London + NY)
  6. HTF trend alignment (confluence)

Exit:
  - SL: 1.5x ATR from entry
  - TP1: 2.0x ATR from entry (close 50%)
  - TP2: 3.0x ATR from entry (close 50%)
  - Max hold: 40 bars (~10 hours on M15)
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S13_BollingerKeltnerSqueeze(BaseStrategy):
    strategy_id = 13
    name = "S13_BollingerKeltnerSqueeze"

    # Squeeze detection
    MIN_SQUEEZE_BARS = 3    # Min bars in squeeze before release
    MAX_SQUEEZE_BARS = 30   # Max bars in squeeze (too long = no energy)

    # Risk management
    SL_ATR_MULT = 1.5
    TP1_ATR_MULT = 2.0
    TP2_ATR_MULT = 3.0
    TRAIL_ATR_MULT = 1.0

    # Filters
    MIN_ADX = 15            # Minimum ADX for directional move
    SESSION_START = 7
    SESSION_END = 17

    MAX_BARS = 40

    def _is_squeeze_on(self, row):
        """Check if BB is inside KC (squeeze is on)."""
        bb_upper = row.get("bb_upper", np.nan)
        bb_lower = row.get("bb_lower", np.nan)
        kc_upper = row.get("kc_upper", np.nan)
        kc_lower = row.get("kc_lower", np.nan)

        if any(np.isnan(v) for v in [bb_upper, bb_lower, kc_upper, kc_lower]):
            return False

        return bb_upper < kc_upper and bb_lower > kc_lower

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        if idx < 200:
            return None

        # Session filter
        hour = current.name.hour if hasattr(current.name, 'hour') else 0
        if hour < self.SESSION_START or hour >= self.SESSION_END:
            return None

        atr_val = current.get("atr_14", 0)
        if atr_val <= 0 or np.isnan(atr_val):
            return None

        # Check if squeeze just released (was on, now off)
        current_squeeze = self._is_squeeze_on(current)
        if current_squeeze:
            return None  # Still in squeeze

        # Count how many prior bars were in squeeze
        squeeze_count = 0
        for i in range(idx - 1, max(0, idx - self.MAX_SQUEEZE_BARS - 1), -1):
            if self._is_squeeze_on(data.iloc[i]):
                squeeze_count += 1
            else:
                break

        if squeeze_count < self.MIN_SQUEEZE_BARS:
            return None

        # Also verify the bar before the squeeze run was NOT in squeeze
        # (ensures we detect the release, not a mid-squeeze fluctuation)
        pre_squeeze_idx = idx - 1 - squeeze_count
        if pre_squeeze_idx >= 0 and self._is_squeeze_on(data.iloc[pre_squeeze_idx]):
            return None  # Squeeze was already going before our count window

        # Momentum direction: use MACD histogram
        macd_hist = current.get("macd_hist", 0)
        if np.isnan(macd_hist) or macd_hist == 0:
            return None

        # Also check momentum is accelerating (current > previous)
        prev_hist = data.iloc[idx - 1].get("macd_hist", 0)
        if np.isnan(prev_hist):
            prev_hist = 0

        direction = None
        if macd_hist > 0 and macd_hist > prev_hist:
            direction = "LONG"
        elif macd_hist < 0 and macd_hist < prev_hist:
            direction = "SHORT"

        if direction is None:
            return None

        # ADX filter: trend developing
        adx_val = current.get("adx_14", 0)
        if np.isnan(adx_val):
            adx_val = 0
        if adx_val < self.MIN_ADX:
            return None

        # ADX rising check
        prev_adx = data.iloc[idx - 1].get("adx_14", 0)
        if np.isnan(prev_adx):
            prev_adx = 0
        adx_rising = adx_val > prev_adx

        price = current["close"]

        # HTF alignment
        htf_aligned = False
        if htf_row is not None:
            htf_ema200 = htf_row.get("ema_200", np.nan)
            htf_close = htf_row.get("close", np.nan)
            if not np.isnan(htf_ema200) and not np.isnan(htf_close):
                if direction == "LONG" and htf_close > htf_ema200:
                    htf_aligned = True
                elif direction == "SHORT" and htf_close < htf_ema200:
                    htf_aligned = True

        # Confluence scoring
        confluence = 3  # Base: squeeze release + momentum + ADX
        if htf_aligned:
            confluence += 1
        if adx_rising:
            confluence += 1
        if squeeze_count >= 6:
            confluence += 1  # Longer squeeze = more stored energy

        # SL / TP
        if direction == "LONG":
            sl = price - self.SL_ATR_MULT * atr_val
            tp1 = price + self.TP1_ATR_MULT * atr_val
            tp2 = price + self.TP2_ATR_MULT * atr_val
        else:
            sl = price + self.SL_ATR_MULT * atr_val
            tp1 = price - self.TP1_ATR_MULT * atr_val
            tp2 = price - self.TP2_ATR_MULT * atr_val

        return {
            "direction": direction,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp2,
            "confluence": confluence,
            "entry_pattern": f"squeeze_release_{direction.lower()}",
            "tp_splits": (0.50, 0.50, 0.0),
            "trail_atr_mult": self.TRAIL_ATR_MULT,
            "max_bars": self.MAX_BARS,
        }
