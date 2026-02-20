"""
Strategy S11: ADX Trend Pullback to 20 EMA (M15).

Concept: In strong trending environments (ADX>30, rising), enter on pullbacks
to the 20 EMA when RSI(9) has reset to a neutral zone, confirmed by a
candle closing in the trend direction.

Entry conditions (ALL must be true):
  1. Session filter: 08:00-16:00 UTC
  2. Strong trend: ADX(14) > 30 and rising
  3. Trend direction: EMA(20) > EMA(50) for uptrend (opposite for downtrend)
  4. HTF alignment: close above/below EMA(200)
  5. Pullback to EMA(20): price within 0.3x ATR(14)
  6. Candle confirmation: close in trend direction
  7. RSI(9) reset: 40-55 for longs, 45-60 for shorts
  8. Volume declining during pullback

Exit:
  - SL: Tighter of EMA(50) and 1.5x ATR(10)
  - TP1: 1.0x risk (50%), TP2: 1.8x risk (50%)
  - Trail: 1.0x ATR after TP1
  - Time stop: 40 bars (~10 hours on M15)

Target pairs: USD_JPY, GBP_USD, GBP_JPY
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S11_ADX_TrendPullback(BaseStrategy):
    strategy_id = 11
    name = "S11_ADX_TrendPullback"

    # Tunable parameters
    MIN_ADX = 30              # Strong trend gate
    EMA_PROXIMITY_ATR = 0.3   # Pullback within 0.3x ATR of EMA(20)
    SL_ATR_MULT = 1.5         # SL: 1.5x ATR(10) or beyond EMA(50)
    TP_RR_MULT = 1.8          # TP2: 1.8x risk distance
    RSI9_LONG_LOW = 40        # RSI(9) reset zone for longs
    RSI9_LONG_HIGH = 55
    RSI9_SHORT_LOW = 45       # RSI(9) reset zone for shorts
    RSI9_SHORT_HIGH = 60
    MAX_BARS = 40             # ~10 hours on M15
    TRAIL_ATR_MULT = 1.0      # Trail after 1x risk profit

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

        # ADX gate: strong and rising
        adx_val = current.get("adx_14", 0)
        if np.isnan(adx_val) or adx_val <= self.MIN_ADX:
            return None
        prev_adx = data["adx_14"].iloc[idx - 1] if idx > 0 else 0
        if np.isnan(prev_adx) or adx_val <= prev_adx:
            return None  # ADX must be rising

        # EMA values
        ema_20 = current.get("ema_20", np.nan)
        ema_50 = current.get("ema_50", np.nan)
        if np.isnan(ema_20) or np.isnan(ema_50):
            return None

        # Trend direction from EMA alignment
        uptrend = ema_20 > ema_50
        downtrend = ema_20 < ema_50
        if not uptrend and not downtrend:
            return None

        # HTF alignment required
        if htf_row is None:
            return None
        htf_close = htf_row.get("close", np.nan)
        htf_ema200 = htf_row.get("ema_200", np.nan)
        if np.isnan(htf_close) or np.isnan(htf_ema200):
            return None

        if uptrend and htf_close <= htf_ema200:
            return None
        if downtrend and htf_close >= htf_ema200:
            return None

        price = current["close"]
        open_price = current["open"]
        atr_14 = current.get("atr_14", 0)
        atr_10 = current.get("atr_10", 0)
        if atr_14 <= 0 or np.isnan(atr_14) or atr_10 <= 0 or np.isnan(atr_10):
            return None

        # Pullback proximity: price near EMA(20)
        if abs(price - ema_20) > self.EMA_PROXIMITY_ATR * atr_14:
            return None

        # RSI(9) reset check
        rsi9 = current.get("rsi_9", 50)
        if np.isnan(rsi9):
            return None

        # Volume declining during pullback
        vol = current.get("volume", 0)
        vol_window = data["volume"].iloc[max(0, idx - 3):idx]
        if len(vol_window) == 0:
            return None
        vol_avg = vol_window.mean()
        if vol_avg <= 0 or vol >= vol_avg:
            return None

        direction = None

        if uptrend:
            # Candle closes bullish
            if price <= open_price:
                return None
            # RSI(9) in reset zone for longs
            if self.RSI9_LONG_LOW <= rsi9 <= self.RSI9_LONG_HIGH:
                direction = "LONG"
        elif downtrend:
            # Candle closes bearish
            if price >= open_price:
                return None
            # RSI(9) in reset zone for shorts
            if self.RSI9_SHORT_LOW <= rsi9 <= self.RSI9_SHORT_HIGH:
                direction = "SHORT"

        if direction is None:
            return None

        # Confluence scoring
        confluence = 3  # Base: ADX strong + pullback to EMA + RSI reset
        # HTF alignment already confirmed above
        confluence += 1
        if adx_val > 40:
            confluence += 1  # Very strong trend
        # Volume spike on confirmation candle (relative to longer window)
        vol_avg_20 = data["volume"].iloc[max(0, idx - 20):idx].mean()
        if vol_avg_20 > 0 and vol > 1.5 * vol_avg_20:
            confluence += 1

        # Exit levels
        if direction == "LONG":
            sl_ema = ema_50
            sl_atr = price - self.SL_ATR_MULT * atr_10
            sl = max(sl_ema, sl_atr)  # Tighter: whichever is closer to price
            risk = price - sl
            if risk <= 0:
                return None
            tp1 = price + 1.0 * risk    # Breakeven target
            tp2 = price + self.TP_RR_MULT * risk  # Full target
        else:
            sl_ema = ema_50
            sl_atr = price + self.SL_ATR_MULT * atr_10
            sl = min(sl_ema, sl_atr)  # Tighter: whichever is closer to price
            risk = sl - price
            if risk <= 0:
                return None
            tp1 = price - 1.0 * risk
            tp2 = price - self.TP_RR_MULT * risk

        return {
            "direction": direction,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp2,
            "confluence": confluence,
            "entry_pattern": f"adx_trend_pullback_{direction.lower()}",
            "tp_splits": (0.50, 0.50, 0.0),
            "trail_atr_mult": self.TRAIL_ATR_MULT,
            "max_bars": self.MAX_BARS,
        }
