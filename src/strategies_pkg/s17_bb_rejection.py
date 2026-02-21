"""
Strategy S17: Mean Reversion Bollinger Band Rejection (M15).

Concept: In ranging markets, price respects outer Bollinger Band edges.
When price touches the outer BB and shows a rejection candle (long wick,
small body), it reverts to the mean. 2:1 R:R means only 33% WR needed.

Entry conditions (LONG — SHORT is mirror):
  1. Ranging regime: ADX(14) < 30
  2. Price touches lower BB: close <= BB_Lower(20, 2.0) OR (low <= BB_Lower AND close > BB_Lower)
  3. Rejection candle: body < 0.4 * range AND close > open (bullish)
  4. Volume confirmation: volume > 1.2 * SMA(volume, 20)
  5. HTF trend filter: H1 close > EMA(200, H1) for longs
  6. Entry: next candle open

Exit:
  - SL: low - 1.0 * ATR(14) for longs (~15-25 pips)
  - TP: entry + 2.0 * ATR(14) — 2:1 R:R, no partials
  - Time stop: 32 bars (8 hours on M15)

Target pairs: EUR_USD, GBP_USD, GBP_JPY, GBP_AUD
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S17_BB_Rejection(BaseStrategy):
    strategy_id = 17
    name = "S17_BB_Rejection"

    # Parameters from spec
    MAX_ADX = 30               # Ranging regime gate
    BB_PERIOD = 20             # BB period
    BB_STD = 2.0               # BB standard deviations
    BODY_RATIO = 0.4           # Max body/range ratio for rejection candle
    VOL_MULT = 1.2             # Volume > 1.2 * SMA(volume, 20)
    SL_ATR_MULT = 1.0          # SL = low - 1.0 * ATR(14) for longs
    TP_ATR_MULT = 2.0          # TP = entry + 2.0 * ATR(14) — 2:1 R:R
    MAX_BARS = 32              # Time stop: 32 bars = 8 hours on M15

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        # Warmup
        if idx < 50:
            return None

        # Regime gate: ranging market only
        adx_val = current.get("adx_14", 50)
        if np.isnan(adx_val) or adx_val >= self.MAX_ADX:
            return None

        # BB values required
        bb_upper = current.get("bb_upper", np.nan)
        bb_lower = current.get("bb_lower", np.nan)
        if np.isnan(bb_upper) or np.isnan(bb_lower):
            return None

        price = current["close"]
        high = current["high"]
        low = current["low"]
        open_ = current["open"]

        atr_14 = current.get("atr_14", 0)
        if atr_14 <= 0 or np.isnan(atr_14):
            return None

        # Candle geometry
        candle_range = high - low
        if candle_range <= 0:
            return None
        candle_body = abs(price - open_)
        body_ratio = candle_body / candle_range

        # Volume confirmation: volume > 1.2 * SMA(volume, 20)
        vol = current.get("volume", 0)
        vol_start = max(0, idx - 20)
        vol_window = data["volume"].iloc[vol_start:idx]
        if len(vol_window) < 5:
            return None
        vol_avg = vol_window.mean()
        if vol_avg <= 0 or vol < self.VOL_MULT * vol_avg:
            return None

        # HTF trend filter (gate, not just confluence)
        htf_bullish = False
        htf_bearish = False
        if htf_row is not None:
            htf_close = htf_row.get("close", np.nan)
            htf_ema200 = htf_row.get("ema_200", np.nan)
            if not np.isnan(htf_close) and not np.isnan(htf_ema200):
                htf_bullish = htf_close > htf_ema200
                htf_bearish = htf_close < htf_ema200

        direction = None

        # LONG: price touches lower BB + rejection candle (bullish)
        if (price <= bb_lower or (low <= bb_lower and price > bb_lower)):
            if body_ratio < self.BODY_RATIO and price > open_:  # Small body, bullish close
                if htf_bullish:  # HTF filter is a gate
                    direction = "LONG"

        # SHORT: price touches upper BB + rejection candle (bearish)
        if direction is None:
            if (price >= bb_upper or (high >= bb_upper and price < bb_upper)):
                if body_ratio < self.BODY_RATIO and price < open_:  # Small body, bearish close
                    if htf_bearish:  # HTF filter is a gate
                        direction = "SHORT"

        if direction is None:
            return None

        # Confluence scoring
        confluence = 2  # Base: BB touch + rejection candle
        if body_ratio < 0.2:
            confluence += 1  # Very strong rejection (tiny body)
        if adx_val < 20:
            confluence += 1  # Deep ranging
        if vol > 1.5 * vol_avg:
            confluence += 1  # Strong volume spike

        # Exit levels
        if direction == "LONG":
            sl = low - self.SL_ATR_MULT * atr_14
            tp = price + self.TP_ATR_MULT * atr_14
        else:
            sl = high + self.SL_ATR_MULT * atr_14
            tp = price - self.TP_ATR_MULT * atr_14

        return {
            "direction": direction,
            "sl": sl,
            "tp1": tp,
            "tp2": tp,
            "tp3": tp,
            "confluence": confluence,
            "entry_pattern": f"bb_rejection_{direction.lower()}",
            "tp_splits": (1.0, 0.0, 0.0),
            "trail_atr_mult": 0,
            "max_bars": self.MAX_BARS,
        }
