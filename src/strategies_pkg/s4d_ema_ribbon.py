"""
Strategy S4-D: EMA Ribbon — Volume + ADX Gating.

Entry: Current S4 ribbon compression -> expansion logic PLUS:
  - Volume > 2.0x 20-period average
  - ADX_14 > 30
  - ADX rising vs 5 bars ago
  - ADX was < 35 at some point in last 10 bars (not exhausted)
  - Distance between 15min 8 EMA and 55 EMA > 1.5% of current price

Exit:
  - SL: 2.0 ATR
  - TP: 3.0 ATR (100% close)
  - No partials, no BE moves
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S4D_EMA_Ribbon(BaseStrategy):
    strategy_id = 4
    name = "S4D_Volume_ADX_Gating"

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        if idx < 50:
            return None

        # Session filter: 08:00-16:00 UTC
        hour = current.name.hour if hasattr(current.name, 'hour') else 0
        if hour < 8 or hour >= 16:
            return None

        atr_val = current.get("atr_14", 0)
        if atr_val <= 0 or np.isnan(atr_val):
            return None

        if htf_row is None:
            return None

        # H1 EMA stack check
        htf_ema20 = htf_row.get("ema_20", np.nan)
        htf_ema50 = htf_row.get("ema_50", np.nan)
        htf_ema100 = htf_row.get("ema_100", np.nan)
        htf_ema200 = htf_row.get("ema_200", np.nan)
        if any(np.isnan(v) for v in [htf_ema20, htf_ema50, htf_ema100, htf_ema200]):
            return None

        long_stack = htf_ema20 > htf_ema50 > htf_ema100 > htf_ema200
        short_stack = htf_ema20 < htf_ema50 < htf_ema100 < htf_ema200
        if not long_stack and not short_stack:
            return None

        # M15 ribbon EMAs
        ema_20 = current.get("ema_20", np.nan)
        ema_50 = current.get("ema_50", np.nan)
        ema_100 = current.get("ema_100", np.nan)
        if any(np.isnan(v) for v in [ema_20, ema_50, ema_100]):
            return None

        ribbon_width = max(ema_20, ema_50, ema_100) - min(ema_20, ema_50, ema_100)
        compression_threshold = 1.0 * atr_val

        # Check for recent compression
        was_compressed = False
        min_compression_width = float('inf')
        for j in range(max(0, idx - 20), idx):
            bar = data.iloc[j]
            e20 = bar.get("ema_20", np.nan)
            e50 = bar.get("ema_50", np.nan)
            e100 = bar.get("ema_100", np.nan)
            if any(np.isnan(v) for v in [e20, e50, e100]):
                continue
            w = max(e20, e50, e100) - min(e20, e50, e100)
            if w <= compression_threshold:
                was_compressed = True
                min_compression_width = min(min_compression_width, w)

        if not was_compressed:
            return None

        # Ribbon expanding
        if ribbon_width <= min_compression_width * 1.2:
            return None

        # Determine direction from ribbon expansion
        if long_stack:
            if not (ema_20 >= ema_50):
                return None
            direction = "LONG"
        elif short_stack:
            if not (ema_20 <= ema_50):
                return None
            direction = "SHORT"
        else:
            return None

        # ------- NEW S4-D FILTERS -------

        # Volume > 2.0x 20-period average
        if "volume" not in current.index:
            return None
        vol = current["volume"]
        vol_avg = data["volume"].iloc[max(0, idx - 20):idx].mean()
        if vol_avg <= 0 or vol <= 2.0 * vol_avg:
            return None

        # ADX_14 > 30
        adx_val = current.get("adx_14", 0)
        if np.isnan(adx_val) or adx_val <= 30:
            return None

        # ADX rising vs 5 bars ago
        if idx < 5:
            return None
        adx_5_ago = data.iloc[idx - 5].get("adx_14", 0)
        if np.isnan(adx_5_ago) or adx_val <= adx_5_ago:
            return None

        # ADX was < 35 at some point in last 10 bars (not exhausted)
        adx_was_low = False
        for j in range(max(0, idx - 10), idx):
            bar_adx = data.iloc[j].get("adx_14", 0)
            if not np.isnan(bar_adx) and bar_adx < 35:
                adx_was_low = True
                break
        if not adx_was_low:
            return None

        # Distance between 15min 8 EMA and 55 EMA > 1.5% of current price
        ema_8 = current.get("ema_8", np.nan)
        ema_55 = current.get("ema_55", np.nan)
        if np.isnan(ema_8) or np.isnan(ema_55):
            return None
        price = current["close"]
        if abs(ema_8 - ema_55) <= 0.015 * price:
            return None

        # ------- EXIT LEVELS -------
        if direction == "LONG":
            sl = price - 2.0 * atr_val
            tp1 = price + 3.0 * atr_val
        else:
            sl = price + 2.0 * atr_val
            tp1 = price - 3.0 * atr_val

        return {
            "direction": direction,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp1,  # same as tp1 — single target
            "tp3": tp1,
            "confluence": 3,
            "entry_pattern": "ribbon_vol_adx",
            "tp_splits": (1.0, 0.0, 0.0),  # 100% close at TP1
            "trail_atr_mult": 0,  # no trailing
            "max_bars": 60,
            "no_breakeven": True,
        }
