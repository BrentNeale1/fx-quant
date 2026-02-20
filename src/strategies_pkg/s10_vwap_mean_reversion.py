"""
Strategy S10: VWAP Mean Reversion (M15).

Concept: Fade extreme deviations from session VWAP in ranging markets.
Price stretched beyond 2 SD from VWAP with RSI(9) at extremes signals
mean reversion back to VWAP.

Entry conditions (ALL must be true):
  1. Session filter: 08:00-16:00 UTC
  2. Ranging regime: ADX(14) < 25
  3. Price >= 2 SD from session VWAP (z-score)
  4. RSI(9) at extreme: <25 for longs, >75 for shorts
  5. Volume declining (fading momentum)

Exit:
  - SL: Tighter of VWAP 3-SD band and 1.5x ATR(10)
  - TP: Session VWAP (100% exit at mean)
  - Time stop: 8 bars (2 hours on M15)

Target pairs: EUR_USD, USD_JPY, EUR_GBP (range-bound majors)
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S10_VWAP_MeanReversion(BaseStrategy):
    strategy_id = 10
    name = "S10_VWAP_MeanReversion"

    # Tunable parameters
    VWAP_SD_ENTRY = 2.0       # Min SD from VWAP to enter
    VWAP_SD_SL = 3.0          # SL at this SD band
    SL_ATR_MULT = 1.5         # Alternative SL: 1.5x ATR(10)
    MAX_ADX = 30              # Ranging regime gate
    RSI9_LONG_THRESH = 25     # RSI(9) < 25 for longs
    RSI9_SHORT_THRESH = 75    # RSI(9) > 75 for shorts
    MAX_BARS = 8              # Time stop: 8 bars = 2 hours on M15

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        # Warmup
        if idx < 50:
            return None

        # Session filter: 08:00-16:00 UTC
        hour = current.name.hour if hasattr(current.name, 'hour') else 0
        if hour < 8 or hour >= 16:
            return None

        # Regime gate: ranging market only
        adx_val = current.get("adx_14", 50)
        if np.isnan(adx_val) or adx_val >= self.MAX_ADX:
            return None

        # VWAP data required
        vwap = current.get("session_vwap", np.nan)
        vwap_std = current.get("vwap_std", np.nan)
        if np.isnan(vwap) or np.isnan(vwap_std) or vwap_std <= 0:
            return None

        price = current["close"]
        atr_10 = current.get("atr_10", 0)
        if atr_10 <= 0 or np.isnan(atr_10):
            return None

        rsi9 = current.get("rsi_9", 50)
        if np.isnan(rsi9):
            return None

        # Z-score: how many SDs from VWAP
        z_score = (price - vwap) / vwap_std

        # Volume declining check (fading momentum)
        vol = current.get("volume", 0)
        vol_window = data["volume"].iloc[max(0, idx - 3):idx]
        if len(vol_window) == 0:
            return None
        vol_avg = vol_window.mean()
        if vol_avg <= 0 or vol >= vol_avg:
            return None  # Volume not declining

        # HTF trend for confluence (soft — not a gate)
        htf_bullish = False
        htf_bearish = False
        if htf_row is not None:
            htf_close = htf_row.get("close", np.nan)
            htf_ema200 = htf_row.get("ema_200", np.nan)
            if not np.isnan(htf_close) and not np.isnan(htf_ema200):
                htf_bullish = htf_close > htf_ema200
                htf_bearish = htf_close < htf_ema200

        direction = None

        # LONG: price well below VWAP, RSI oversold
        if z_score <= -self.VWAP_SD_ENTRY and rsi9 < self.RSI9_LONG_THRESH:
            direction = "LONG"
        # SHORT: price well above VWAP, RSI overbought
        elif z_score >= self.VWAP_SD_ENTRY and rsi9 > self.RSI9_SHORT_THRESH:
            direction = "SHORT"

        if direction is None:
            return None

        # Confluence scoring
        confluence = 2  # Base: VWAP deviation + RSI extreme
        if abs(z_score) >= 2.5:
            confluence += 1  # Deeper deviation
        if (direction == "LONG" and htf_bullish) or \
           (direction == "SHORT" and htf_bearish):
            confluence += 1  # HTF aligns with mean reversion direction
        if vol_avg > 0 and vol < 0.7 * vol_avg:
            confluence += 1  # Strong fade signal

        # Exit levels
        if direction == "LONG":
            sl_vwap = vwap - self.VWAP_SD_SL * vwap_std
            sl_atr = price - self.SL_ATR_MULT * atr_10
            sl = max(sl_vwap, sl_atr)  # Tighter (closer to price) for longs
            tp = vwap  # Mean reversion target
        else:
            sl_vwap = vwap + self.VWAP_SD_SL * vwap_std
            sl_atr = price + self.SL_ATR_MULT * atr_10
            sl = min(sl_vwap, sl_atr)  # Tighter (closer to price) for shorts
            tp = vwap

        return {
            "direction": direction,
            "sl": sl,
            "tp1": tp,
            "tp2": tp,
            "tp3": tp,
            "confluence": confluence,
            "entry_pattern": f"vwap_mean_reversion_{direction.lower()}",
            "tp_splits": (1.0, 0.0, 0.0),
            "trail_atr_mult": 0,
            "max_bars": self.MAX_BARS,
        }
