"""
Strategy S4-F: EMA Ribbon — Trend Context Filter.

Entry: Current S4 ribbon logic PLUS:
  - 1H trend for LONG: 1H close > 1H 200 EMA AND 1H 50 EMA > 1H 200 EMA
  - 1H trend for SHORT: 1H close < 1H 200 EMA AND 1H 50 EMA < 1H 200 EMA
  - Price within 1.5 ATR of 1H 50 EMA
  - 15min EMA stacking for LONG: 8 EMA > 13 EMA AND 13 EMA > 21 EMA
  - 15min EMA stacking for SHORT: 8 EMA < 13 EMA AND 13 EMA < 21 EMA
  - Volume > 1.2x average

Exit: Same as S4-D (SL 2.0 ATR, TP 3.0 ATR, 100% close, no partials)
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S4F_EMA_Ribbon(BaseStrategy):
    strategy_id = 4
    name = "S4F_Trend_Context"

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        if idx < 50:
            return None

        hour = current.name.hour if hasattr(current.name, 'hour') else 0
        if hour < 8 or hour >= 16:
            return None

        atr_val = current.get("atr_14", 0)
        if atr_val <= 0 or np.isnan(atr_val):
            return None

        if htf_row is None:
            return None

        # ------- ORIGINAL H1 EMA STACK (20 > 50 > 100 > 200) -------
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

        # ------- NEW 1H TREND FILTER -------
        htf_close = htf_row.get("close", np.nan)
        if np.isnan(htf_close):
            return None

        if long_stack:
            if not (htf_close > htf_ema200 and htf_ema50 > htf_ema200):
                return None
            direction = "LONG"
        else:
            if not (htf_close < htf_ema200 and htf_ema50 < htf_ema200):
                return None
            direction = "SHORT"

        # ------- PRICE WITHIN 1.5 ATR OF 1H 50 EMA -------
        price = current["close"]
        if abs(price - htf_ema50) > 1.5 * atr_val:
            return None

        # ------- M15 RIBBON COMPRESSION -> EXPANSION -------
        ema_20 = current.get("ema_20", np.nan)
        ema_50 = current.get("ema_50", np.nan)
        ema_100 = current.get("ema_100", np.nan)
        if any(np.isnan(v) for v in [ema_20, ema_50, ema_100]):
            return None

        ribbon_width = max(ema_20, ema_50, ema_100) - min(ema_20, ema_50, ema_100)
        compression_threshold = 1.0 * atr_val

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

        if ribbon_width <= min_compression_width * 1.2:
            return None

        # ------- 15MIN EMA STACKING: 8 > 13 > 21 (LONG) or reversed -------
        ema_8 = current.get("ema_8", np.nan)
        ema_13 = current.get("ema_13", np.nan)
        ema_21 = current.get("ema_21", np.nan)
        if any(np.isnan(v) for v in [ema_8, ema_13, ema_21]):
            return None

        if direction == "LONG":
            if not (ema_8 > ema_13 and ema_13 > ema_21):
                return None
        else:
            if not (ema_8 < ema_13 and ema_13 < ema_21):
                return None

        # ------- VOLUME > 1.2x average -------
        if "volume" not in current.index:
            return None
        vol = current["volume"]
        vol_avg = data["volume"].iloc[max(0, idx - 20):idx].mean()
        if vol_avg <= 0 or vol <= 1.2 * vol_avg:
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
            "tp2": tp1,
            "tp3": tp1,
            "confluence": 3,
            "entry_pattern": "ribbon_trend_context",
            "tp_splits": (1.0, 0.0, 0.0),
            "trail_atr_mult": 0,
            "max_bars": 60,
            "no_breakeven": True,
        }
