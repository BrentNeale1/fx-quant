"""
Strategy S20: Range Compression Breakout / Volatility Expansion (M15).

Concept: Volatility is cyclical. After compression (narrow ranges), markets
expand. Wait for ATR compression AND a BB-Keltner squeeze, then enter the
breakout of that compression with volume and MACD confirmation.

Entry conditions (LONG — SHORT is mirror):
  1. ATR compression: ATR(14) < 0.7 * SMA(ATR(14), 20)
  2. BB-Keltner squeeze: BB_Upper(20,2.0) < KC_Upper(20,1.5) AND BB_Lower > KC_Lower
  3. Expansion trigger: close > BB_Upper (LONG) or close < BB_Lower (SHORT)
  4. Volume confirmation: volume > 1.5 * SMA(volume, 20)
  5. Momentum: MACD histogram aligned with breakout direction

Exit:
  - SL: BB_Lower - 0.5 * ATR(14) for longs
  - TP: entry + 3.0 * ATR(14) — 3:1 R:R
  - Trailing: hold while price > BB_Middle, exit when crosses below

Target pairs: EUR_USD, GBP_USD, GBP_JPY, GBP_AUD
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S20_RangeCompression(BaseStrategy):
    strategy_id = 20
    name = "S20_Range_Compression"

    # Parameters from spec
    ATR_COMP_RATIO = 0.7       # ATR < 0.7 * SMA(ATR, 20) = compressed
    ATR_SMA_PERIOD = 20        # Period for ATR moving average
    VOL_MULT = 1.5             # Volume > 1.5 * SMA(volume, 20)
    SL_ATR_MULT = 0.5          # SL = BB_Lower - 0.5 * ATR for longs
    TP_ATR_MULT = 3.0          # TP = entry + 3.0 * ATR — 3:1 R:R
    MAX_BARS = 48              # Time stop: 48 bars = 12 hours on M15

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        # Warmup
        if idx < 50:
            return None

        # Required indicators
        atr_14 = current.get("atr_14", np.nan)
        bb_upper = current.get("bb_upper", np.nan)
        bb_lower = current.get("bb_lower", np.nan)
        bb_mid = current.get("bb_mid", np.nan)
        kc_upper = current.get("kc_upper", np.nan)
        kc_lower = current.get("kc_lower", np.nan)
        macd_hist = current.get("macd_hist", np.nan)

        if any(np.isnan(v) for v in [atr_14, bb_upper, bb_lower, bb_mid,
                                      kc_upper, kc_lower, macd_hist]):
            return None
        if atr_14 <= 0:
            return None

        price = current["close"]

        # 1. ATR compression: ATR(14) < 0.7 * SMA(ATR(14), 20)
        atr_start = max(0, idx - self.ATR_SMA_PERIOD)
        atr_window = data["atr_14"].iloc[atr_start:idx]
        if len(atr_window) < 10:
            return None
        atr_avg = atr_window.mean()
        if atr_avg <= 0 or atr_14 >= self.ATR_COMP_RATIO * atr_avg:
            return None  # Not compressed

        # 2. BB-Keltner squeeze: BB inside KC
        if not (bb_upper < kc_upper and bb_lower > kc_lower):
            return None  # No squeeze

        # 3. Expansion trigger: close breaks out of BB
        # 4. Volume confirmation
        vol = current.get("volume", 0)
        vol_start = max(0, idx - 20)
        vol_window = data["volume"].iloc[vol_start:idx]
        if len(vol_window) < 5:
            return None
        vol_avg = vol_window.mean()
        if vol_avg <= 0 or vol < self.VOL_MULT * vol_avg:
            return None

        direction = None

        # LONG: close > BB_Upper + MACD histogram positive
        if price > bb_upper and macd_hist > 0:
            direction = "LONG"
        # SHORT: close < BB_Lower + MACD histogram negative
        elif price < bb_lower and macd_hist < 0:
            direction = "SHORT"

        if direction is None:
            return None

        # Confluence scoring
        confluence = 3  # Base: squeeze + breakout + volume
        # Very tight squeeze
        squeeze_ratio = (bb_upper - bb_lower) / (kc_upper - kc_lower) if (kc_upper - kc_lower) > 0 else 1
        if squeeze_ratio < 0.5:
            confluence += 1
        # Very strong volume
        if vol > 2.0 * vol_avg:
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
            sl = bb_lower - self.SL_ATR_MULT * atr_14
            tp = price + self.TP_ATR_MULT * atr_14
        else:
            sl = bb_upper + self.SL_ATR_MULT * atr_14
            tp = price - self.TP_ATR_MULT * atr_14

        return {
            "direction": direction,
            "sl": sl,
            "tp1": tp,
            "tp2": tp,
            "tp3": tp,
            "confluence": confluence,
            "entry_pattern": f"range_compression_{direction.lower()}",
            "tp_splits": (1.0, 0.0, 0.0),
            "trail_atr_mult": 0,
            "max_bars": self.MAX_BARS,
        }
