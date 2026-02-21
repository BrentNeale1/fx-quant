"""
Strategy S21: EMA Ribbon Bounce in Range (M15).

Concept: In ranging markets, EMAs act as dynamic support/resistance. The EMA
ribbon (20/50/100 cluster) provides reversal zones when flat. Only trades
when the ribbon is flat (ranging) and exits if it starts trending.

Entry conditions (LONG — SHORT is mirror):
  1. Flat ribbon: |EMA_20_slope| < 0.0002 AND |EMA_50_slope| < 0.0002
  2. Touch cluster: low <= EMA_20 AND low >= EMA_50
  3. Rejection: close > EMA_20 AND (close - low) / (high - low) > 0.6
  4. RSI: RSI(9) > 35 AND RSI(9) < 60
  5. Volume declining: volume < SMA(volume, 20)

Exit:
  - SL: EMA_100 - 1.0 * ATR(14) for longs
  - TP: entry + 2.0 * ATR(14) — ~1.3:1 R:R
  - Time stop: 32 bars (8 hours on M15)

Target pairs: EUR_USD, GBP_USD, GBP_JPY, GBP_AUD
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S21_EMA_RibbonBounce(BaseStrategy):
    strategy_id = 21
    name = "S21_EMA_RibbonBounce"

    # Parameters from spec
    SLOPE_THRESH = 0.0002      # Max |slope| for flat ribbon
    REJECTION_RATIO = 0.6      # (close - low) / (high - low) > 0.6 for longs
    RSI_LOW = 35               # RSI(9) > 35 for longs
    RSI_HIGH = 60              # RSI(9) < 60 for longs
    RSI_SHORT_LOW = 40         # RSI(9) > 40 for shorts (mirror)
    RSI_SHORT_HIGH = 65        # RSI(9) < 65 for shorts (mirror)
    SL_ATR_MULT = 1.0          # SL = EMA_100 - 1.0 * ATR(14)
    TP_ATR_MULT = 2.0          # TP = entry + 2.0 * ATR(14)
    SLOPE_LOOKBACK = 10        # Bars to compute EMA slope over
    MAX_BARS = 32              # Time stop: 32 bars = 8 hours on M15

    def _ema_slope(self, data: pd.DataFrame, col: str, idx: int) -> float:
        """Compute slope of EMA as (current - N bars ago) / N, normalized by price."""
        if idx < self.SLOPE_LOOKBACK:
            return float('inf')
        current_val = data[col].iloc[idx]
        past_val = data[col].iloc[idx - self.SLOPE_LOOKBACK]
        if np.isnan(current_val) or np.isnan(past_val) or past_val == 0:
            return float('inf')
        # Normalize by price level so threshold works across pairs
        return (current_val - past_val) / (self.SLOPE_LOOKBACK * past_val)

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        # Warmup
        if idx < 50:
            return None

        # Required indicators
        ema_20 = current.get("ema_20", np.nan)
        ema_50 = current.get("ema_50", np.nan)
        ema_100 = current.get("ema_100", np.nan)
        rsi_9 = current.get("rsi_9", np.nan)
        atr_14 = current.get("atr_14", np.nan)

        if any(np.isnan(v) for v in [ema_20, ema_50, ema_100, rsi_9, atr_14]):
            return None
        if atr_14 <= 0:
            return None

        price = current["close"]
        high = current["high"]
        low = current["low"]

        candle_range = high - low
        if candle_range <= 0:
            return None

        # 1. Flat ribbon: both EMA_20 and EMA_50 slopes near zero
        slope_20 = self._ema_slope(data, "ema_20", idx)
        slope_50 = self._ema_slope(data, "ema_50", idx)
        if abs(slope_20) >= self.SLOPE_THRESH or abs(slope_50) >= self.SLOPE_THRESH:
            return None

        # 5. Volume declining: volume < SMA(volume, 20)
        vol = current.get("volume", 0)
        vol_start = max(0, idx - 20)
        vol_window = data["volume"].iloc[vol_start:idx]
        if len(vol_window) < 5:
            return None
        vol_avg = vol_window.mean()
        if vol_avg <= 0 or vol >= vol_avg:
            return None

        direction = None

        # LONG: price bounces off EMA ribbon from above
        # Touch: low dips into ribbon (low <= ema_20, low >= ema_50)
        # Rejection: close > ema_20, close in upper 60% of candle
        if (low <= ema_20 and low >= ema_50 and
                price > ema_20 and
                (price - low) / candle_range > self.REJECTION_RATIO and
                rsi_9 > self.RSI_LOW and rsi_9 < self.RSI_HIGH):
            direction = "LONG"

        # SHORT: price bounces off EMA ribbon from below
        # Touch: high >= ema_20, high <= ema_50 (ribbon inverted or clustered)
        # Actually for SHORT in ranging: high dips into ribbon from below
        # high >= ema_20 and high <= ema_50 (when ema_20 < ema_50 in downward lean)
        # More generally: high >= min(ema_20, ema_50) and high <= max(ema_20, ema_50)
        if direction is None:
            ema_upper = max(ema_20, ema_50)
            ema_lower = min(ema_20, ema_50)
            if (high >= ema_lower and high <= ema_upper and
                    price < ema_lower and
                    (high - price) / candle_range > self.REJECTION_RATIO and
                    rsi_9 > self.RSI_SHORT_LOW and rsi_9 < self.RSI_SHORT_HIGH):
                direction = "SHORT"

        if direction is None:
            return None

        # Confluence scoring
        confluence = 2  # Base: ribbon touch + rejection
        # Very flat ribbon
        if abs(slope_20) < 0.0001 and abs(slope_50) < 0.0001:
            confluence += 1
        # Strong rejection (close in top/bottom 80% of candle)
        if direction == "LONG" and (price - low) / candle_range > 0.8:
            confluence += 1
        elif direction == "SHORT" and (high - price) / candle_range > 0.8:
            confluence += 1
        # HTF alignment
        if htf_row is not None:
            htf_close = htf_row.get("close", np.nan)
            htf_ema200 = htf_row.get("ema_200", np.nan)
            if not np.isnan(htf_close) and not np.isnan(htf_ema200):
                if (direction == "LONG" and htf_close > htf_ema200) or \
                   (direction == "SHORT" and htf_close < htf_ema200):
                    confluence += 1

        # Exit levels
        if direction == "LONG":
            sl = ema_100 - self.SL_ATR_MULT * atr_14
            tp = price + self.TP_ATR_MULT * atr_14
        else:
            sl = ema_100 + self.SL_ATR_MULT * atr_14
            tp = price - self.TP_ATR_MULT * atr_14

        return {
            "direction": direction,
            "sl": sl,
            "tp1": tp,
            "tp2": tp,
            "tp3": tp,
            "confluence": confluence,
            "entry_pattern": f"ema_ribbon_bounce_{direction.lower()}",
            "tp_splits": (1.0, 0.0, 0.0),
            "trail_atr_mult": 0,
            "max_bars": self.MAX_BARS,
        }
