"""
Strategy S4-E: EMA Ribbon — Compression Quality + Stochastic.

Entry: Current S4 ribbon logic PLUS:
  - All 5 EMAs (8, 13, 21, 34, 55) were within 0.3% of each other
    in at least 1 of last 10 bars (true compression)
  - Now expanding (current distance between 8 and 55 > 0.5% of price)
  - Stochastic_14 %K was < 20 in last 5 bars (LONG) AND current %K > %D
  - Stochastic_14 %K was > 80 in last 5 bars (SHORT) AND current %K < %D
  - MACD histogram expanding in trade direction
  - Volume > 1.2x average

Exit: Same as S4-D (SL 2.0 ATR, TP 3.0 ATR, 100% close, no partials)
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S4E_EMA_Ribbon(BaseStrategy):
    strategy_id = 4
    name = "S4E_Compression_Quality"

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

        # H1 EMA stack
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

        direction = "LONG" if long_stack else "SHORT"

        # ------- TRUE COMPRESSION: all 5 EMAs within 0.3% in last 10 bars -------
        had_true_compression = False
        for j in range(max(0, idx - 10), idx):
            bar = data.iloc[j]
            emas = []
            for p in [8, 13, 21, 34, 55]:
                v = bar.get(f"ema_{p}", np.nan)
                if np.isnan(v):
                    break
                emas.append(v)
            if len(emas) < 5:
                continue
            spread = max(emas) - min(emas)
            mid = np.mean(emas)
            if mid > 0 and spread / mid < 0.003:  # within 0.3%
                had_true_compression = True
                break

        if not had_true_compression:
            return None

        # ------- NOW EXPANDING: 8 EMA vs 55 EMA > 0.5% of price -------
        ema_8 = current.get("ema_8", np.nan)
        ema_55 = current.get("ema_55", np.nan)
        if np.isnan(ema_8) or np.isnan(ema_55):
            return None

        price = current["close"]
        if abs(ema_8 - ema_55) <= 0.005 * price:
            return None

        # Direction consistency: 8 EMA must be on correct side of 55 EMA
        if direction == "LONG" and ema_8 <= ema_55:
            return None
        if direction == "SHORT" and ema_8 >= ema_55:
            return None

        # ------- STOCHASTIC 14 FILTER -------
        stoch_k = current.get("stoch_k_14", np.nan)
        stoch_d = current.get("stoch_d_14", np.nan)
        if np.isnan(stoch_k) or np.isnan(stoch_d):
            return None

        if direction == "LONG":
            # %K was < 20 in last 5 bars
            stoch_was_oversold = False
            for j in range(max(0, idx - 5), idx):
                sk = data.iloc[j].get("stoch_k_14", 50)
                if not np.isnan(sk) and sk < 20:
                    stoch_was_oversold = True
                    break
            if not stoch_was_oversold:
                return None
            # Current %K > %D
            if stoch_k <= stoch_d:
                return None
        else:
            # %K was > 80 in last 5 bars
            stoch_was_overbought = False
            for j in range(max(0, idx - 5), idx):
                sk = data.iloc[j].get("stoch_k_14", 50)
                if not np.isnan(sk) and sk > 80:
                    stoch_was_overbought = True
                    break
            if not stoch_was_overbought:
                return None
            # Current %K < %D
            if stoch_k >= stoch_d:
                return None

        # ------- MACD HISTOGRAM EXPANDING -------
        if idx < 2:
            return None
        macd_curr = current.get("macd_hist", 0)
        macd_1 = data.iloc[idx - 1].get("macd_hist", 0)
        macd_2 = data.iloc[idx - 2].get("macd_hist", 0)

        if direction == "LONG":
            if not (macd_curr > macd_1 and macd_curr > macd_2):
                return None
        else:
            if not (macd_curr < macd_1 and macd_curr < macd_2):
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
            "entry_pattern": "ribbon_compression_quality",
            "tp_splits": (1.0, 0.0, 0.0),
            "trail_atr_mult": 0,
            "max_bars": 60,
            "no_breakeven": True,
        }
