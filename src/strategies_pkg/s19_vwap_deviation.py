"""
Strategy S19: VWAP Deviation Mean Reversion (M15).

Concept: When price deviates >= 2.0 ATR units from session VWAP in a ranging
market (ADX < 25), with RSI(9) at extremes and declining volume, fade back
toward the VWAP mean.

Key differences from S10 (VWAP Mean Reversion):
  - Deviation normalized by ATR(14) instead of vwap_std
  - VWAP resets at 00:00 UTC (midnight) instead of 08:00
  - RSI thresholds 30/70 instead of 25/75
  - Session-based time stop at 16:00 UTC
  - SL at VWAP +/- 3.0 * ATR(14)
  - Longer time stop: 48 bars (~12h on M15)

Entry conditions (LONG — SHORT is mirror):
  1. Hour < 16 (session gate)
  2. ADX(14) < 25 (ranging)
  3. deviation = (close - session_vwap_midnight) / ATR(14) <= -2.0
  4. RSI(9) < 30
  5. Volume < 0.8 * SMA(volume, 20)
  6. SL = session_vwap_midnight - 3.0 * ATR(14)
  7. TP = session_vwap_midnight (mean reversion target)

Target pairs: EUR_USD, GBP_USD, GBP_JPY, GBP_AUD
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S19_VWAPDeviationMeanReversion(BaseStrategy):
    strategy_id = 19
    name = "S19_VWAP_Deviation_MeanReversion"

    # Parameters from spec
    DEVIATION_ENTRY = 2.0      # |deviation| >= 2.0 ATR units from VWAP
    SL_ATR_MULT = 3.0          # SL = VWAP +/- 3.0 * ATR(14)
    MAX_ADX = 25               # Ranging only
    RSI9_LONG = 30             # RSI(9) < 30 for longs
    RSI9_SHORT = 70            # RSI(9) > 70 for shorts
    VOL_MULT = 0.8             # Volume < 0.8 * SMA(vol, 20)
    SESSION_END = 16           # Exit gate at 16:00 UTC
    MAX_BARS = 48              # Fallback time stop (~12h on M15)

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        # Warmup
        if idx < 50:
            return None

        # Session gate: no new entries at or after 16:00 UTC
        hour = current.name.hour if hasattr(current.name, 'hour') else 0
        if hour >= self.SESSION_END:
            return None

        # Regime gate: ranging market only
        adx_val = current.get("adx_14", 50)
        if np.isnan(adx_val) or adx_val >= self.MAX_ADX:
            return None

        # VWAP data required (midnight-reset)
        vwap = current.get("session_vwap_midnight", np.nan)
        if np.isnan(vwap):
            return None

        price = current["close"]
        atr_14 = current.get("atr_14", 0)
        if atr_14 <= 0 or np.isnan(atr_14):
            return None

        # ATR-normalized deviation from VWAP
        deviation = (price - vwap) / atr_14

        # RSI(9)
        rsi9 = current.get("rsi_9", 50)
        if np.isnan(rsi9):
            return None

        # Volume declining check: current volume < 0.8 * SMA(volume, 20)
        vol = current.get("volume", 0)
        vol_start = max(0, idx - 20)
        vol_window = data["volume"].iloc[vol_start:idx]
        if len(vol_window) < 5:
            return None
        vol_avg = vol_window.mean()
        if vol_avg <= 0 or vol >= self.VOL_MULT * vol_avg:
            return None

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
        if deviation <= -self.DEVIATION_ENTRY and rsi9 < self.RSI9_LONG:
            direction = "LONG"
        # SHORT: price well above VWAP, RSI overbought
        elif deviation >= self.DEVIATION_ENTRY and rsi9 > self.RSI9_SHORT:
            direction = "SHORT"

        if direction is None:
            return None

        # Confluence scoring
        confluence = 2  # Base: VWAP deviation + RSI extreme
        if abs(deviation) >= 2.5:
            confluence += 1  # Deeper deviation
        if (direction == "LONG" and htf_bullish) or \
           (direction == "SHORT" and htf_bearish):
            confluence += 1  # HTF aligns with mean reversion direction
        if vol_avg > 0 and vol < 0.6 * vol_avg:
            confluence += 1  # Strong volume fade

        # Exit levels
        if direction == "LONG":
            sl = vwap - self.SL_ATR_MULT * atr_14
            tp = vwap  # Mean reversion target
        else:
            sl = vwap + self.SL_ATR_MULT * atr_14
            tp = vwap

        return {
            "direction": direction,
            "sl": sl,
            "tp1": tp,
            "tp2": tp,
            "tp3": tp,
            "confluence": confluence,
            "entry_pattern": f"vwap_deviation_mr_{direction.lower()}",
            "tp_splits": (1.0, 0.0, 0.0),
            "trail_atr_mult": 0,
            "max_bars": self.MAX_BARS,
        }
