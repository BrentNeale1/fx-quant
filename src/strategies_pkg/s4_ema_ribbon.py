"""
Strategy 4: EMA Ribbon Momentum Scalp.

Entry TF: M15, Filter TF: H1.

H1 Filter: EMA 20 > 50 > 100 > 200 (LONG) or reversed (SHORT)

M15 Entry (LONG):
  - EMA ribbon compressed (EMAs within 1.0x ATR)
  - Ribbon re-expanding (current width > prev width)
  - Stochastic turning from oversold
  - Session filter: London/NY (08:00-16:00 UTC)

SL: Below compression zone low/high - 0.5x ATR (capped at 1.5x ATR from entry)
TP1: 2x ATR, TP2: 3x ATR, TP3: 5x ATR
Min RR: 1.0:1 at entry (TP1 dist >= SL dist)
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S4_EMA_Ribbon(BaseStrategy):
    strategy_id = 4
    name = "S4_EMA_Ribbon_Scalp"

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        if idx < 50:
            return None

        # Session filter: London + NY overlap (08:00-16:00 UTC)
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

        # Check for recent compression (look back 5-20 bars)
        was_compressed = False
        compression_low = current["low"]
        compression_high = current["high"]
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
                compression_low = min(compression_low, bar["low"])
                compression_high = max(compression_high, bar["high"])

        if not was_compressed:
            return None

        # Current bar: ribbon must be expanding (wider than min compression)
        if ribbon_width <= min_compression_width * 1.2:
            return None

        # Stochastic
        stoch_k = current.get("stoch_k", 50)
        stoch_d = current.get("stoch_d", 50)

        if long_stack:
            # Stochastic turning up or in bullish zone
            if not (stoch_k > stoch_d or stoch_k < 50):
                return None
            if not (ema_20 >= ema_50):  # Ribbon expanding upward
                return None

            confluence = self._calc_confluence(data, idx, current, atr_val, "LONG")

            # SL: below compression zone low with 0.5x ATR buffer, capped at 1.5 ATR
            sl_natural = compression_low - 0.5 * atr_val
            sl_max = current["close"] - 1.5 * atr_val
            sl = max(sl_natural, sl_max)

            tp1 = current["close"] + 2.0 * atr_val
            tp2 = current["close"] + 3.0 * atr_val
            tp3 = current["close"] + 5.0 * atr_val

            # Min 1:1 RR gate
            sl_dist = current["close"] - sl
            tp1_dist = tp1 - current["close"]
            if sl_dist <= 0 or tp1_dist / sl_dist < 1.0:
                return None

            return {
                "direction": "LONG",
                "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
                "confluence": confluence,
                "entry_pattern": "ribbon_expansion",
                "tp_splits": (0.40, 0.30, 0.30),
                "trail_atr_mult": 2.5,
                "max_bars": 60,
            }

        elif short_stack:
            if not (stoch_k < stoch_d or stoch_k > 50):
                return None
            if not (ema_20 <= ema_50):
                return None

            confluence = self._calc_confluence(data, idx, current, atr_val, "SHORT")

            # SL: above compression zone high with 0.5x ATR buffer, capped at 1.5 ATR
            sl_natural = compression_high + 0.5 * atr_val
            sl_max = current["close"] + 1.5 * atr_val
            sl = min(sl_natural, sl_max)

            tp1 = current["close"] - 2.0 * atr_val
            tp2 = current["close"] - 3.0 * atr_val
            tp3 = current["close"] - 5.0 * atr_val

            # Min 1:1 RR gate
            sl_dist = sl - current["close"]
            tp1_dist = current["close"] - tp1
            if sl_dist <= 0 or tp1_dist / sl_dist < 1.0:
                return None

            return {
                "direction": "SHORT",
                "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
                "confluence": confluence,
                "entry_pattern": "ribbon_expansion",
                "tp_splits": (0.40, 0.30, 0.30),
                "trail_atr_mult": 2.5,
                "max_bars": 60,
            }

        return None

    def _calc_confluence(self, data, idx, current, atr_val, direction):
        confluence = 2  # HTF alignment + compression/expansion

        # ATR contracting
        atr_avg = data["atr_14"].iloc[max(0, idx - 50):idx].mean()
        if atr_val < atr_avg:
            confluence += 1

        # Volume rising
        if "volume" in current.index:
            vol_avg = data["volume"].iloc[max(0, idx - 20):idx].mean()
            if vol_avg > 0 and current["volume"] > vol_avg:
                confluence += 1

        # RSI not extreme
        rsi = current.get("rsi_14", 50)
        if direction == "LONG" and 40 < rsi < 65:
            confluence += 1
        elif direction == "SHORT" and 35 < rsi < 60:
            confluence += 1

        return min(confluence, 5)
