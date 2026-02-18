"""
Strategy S4-G: EMA Ribbon Pullback-First.

3-timeframe: M5 entry, M15 ribbon/ATR, H1 trend.
Pullback-first approach: find the pullback FIRST, then confirm trend.

Step 1 - Find Pullback Setup (PRIMARY filter):
  - 5min Stochastic_14 %K was < 20 (LONG) or > 80 (SHORT) within last 5 bars
  - Price within 1.5 ATR (M15) of 1H 50 EMA

Step 2 - Confirm Trend Context (SECONDARY):
  - LONG: 1H close > 1H 200 EMA AND 1H 50 EMA > 1H 200 EMA
  - SHORT: 1H close < 1H 200 EMA AND 1H 50 EMA < 1H 200 EMA
  - 1H ADX_14 > 25

Step 3 - Confirm Momentum Resuming (ENTRY trigger):
  - 5min Stochastic %K crossed above %D (LONG) or below %D (SHORT)
  - 15min 50 EMA > 15min 100 EMA (LONG) or reversed (SHORT)
  - 5min 8 EMA > 13 EMA > 21 EMA (LONG) or reversed (SHORT)
  - Volume on 5min > 1.5x average

Exit:
  - SL: 2.0 ATR (15min)
  - TP: 3.0 ATR (15min), 100% close
  - No partials, no BE moves
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S4G_Pullback(BaseStrategy):
    strategy_id = 4
    name = "S4G_Pullback_First"

    def __init__(self):
        super().__init__()
        self.m15_data = None  # Set externally by runner

    def _get_m15_row(self, timestamp: pd.Timestamp) -> Optional[pd.Series]:
        """Get most recent FULLY CLOSED M15 candle before timestamp."""
        if self.m15_data is None:
            return None
        valid = self.m15_data[self.m15_data.index < timestamp]
        if len(valid) == 0:
            return None
        return valid.iloc[-1]

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        """
        data = M5 dataframe (primary)
        htf_row = most recent closed H1 candle
        self.m15_data = M15 dataframe with indicators (set externally)
        """
        if idx < 50:
            return None

        timestamp = current.name

        # Session filter: 08:00-16:00 UTC
        hour = timestamp.hour if hasattr(timestamp, 'hour') else 0
        if hour < 8 or hour >= 16:
            return None

        # ===== STEP 1: FIND PULLBACK SETUP (PRIMARY) =====

        # 5min Stochastic_14: was < 20 (LONG) or > 80 (SHORT) within last 5 bars
        stoch_k = current.get("stoch_k_14", np.nan)
        stoch_d = current.get("stoch_d_14", np.nan)
        if np.isnan(stoch_k) or np.isnan(stoch_d):
            return None

        was_oversold = False
        was_overbought = False
        for j in range(max(0, idx - 5), idx):
            sk = data.iloc[j].get("stoch_k_14", np.nan)
            if np.isnan(sk):
                continue
            if sk < 20:
                was_oversold = True
            if sk > 80:
                was_overbought = True

        if not was_oversold and not was_overbought:
            return None

        # Need M15 data for ATR and H1 data for 50 EMA
        if htf_row is None:
            return None

        m15_row = self._get_m15_row(timestamp)
        if m15_row is None:
            return None

        m15_atr = m15_row.get("atr_14", np.nan)
        if np.isnan(m15_atr) or m15_atr <= 0:
            return None

        htf_ema50 = htf_row.get("ema_50", np.nan)
        if np.isnan(htf_ema50):
            return None

        # Price within 1.5 ATR (M15) of 1H 50 EMA
        price = current["close"]
        if abs(price - htf_ema50) > 1.5 * m15_atr:
            return None

        # ===== STEP 2: CONFIRM TREND CONTEXT (SECONDARY) =====

        htf_ema200 = htf_row.get("ema_200", np.nan)
        htf_close = htf_row.get("close", np.nan)
        htf_adx = htf_row.get("adx_14", np.nan)

        if any(np.isnan(v) for v in [htf_ema200, htf_close, htf_adx]):
            return None

        # Determine direction from H1 trend
        if htf_close > htf_ema200 and htf_ema50 > htf_ema200:
            direction = "LONG"
        elif htf_close < htf_ema200 and htf_ema50 < htf_ema200:
            direction = "SHORT"
        else:
            return None

        # Verify stochastic matches direction
        if direction == "LONG" and not was_oversold:
            return None
        if direction == "SHORT" and not was_overbought:
            return None

        # H1 ADX > 25
        if htf_adx <= 25:
            return None

        # ===== STEP 3: CONFIRM MOMENTUM RESUMING (ENTRY TRIGGER) =====

        # 5min Stochastic %K crossed above %D (LONG) or below %D (SHORT)
        if direction == "LONG" and stoch_k <= stoch_d:
            return None
        if direction == "SHORT" and stoch_k >= stoch_d:
            return None

        # 15min 50 EMA > 15min 100 EMA (LONG) or reversed
        m15_ema50 = m15_row.get("ema_50", np.nan)
        m15_ema100 = m15_row.get("ema_100", np.nan)
        if np.isnan(m15_ema50) or np.isnan(m15_ema100):
            return None

        if direction == "LONG" and not (m15_ema50 > m15_ema100):
            return None
        if direction == "SHORT" and not (m15_ema50 < m15_ema100):
            return None

        # S4-G-Minimal: No 5min EMA check (contradicts pullback timing)

        # Volume on 5min > 1.5x average
        if "volume" not in current.index:
            return None
        vol = current["volume"]
        vol_avg = data["volume"].iloc[max(0, idx - 20):idx].mean()
        if vol_avg <= 0 or vol <= 1.5 * vol_avg:
            return None

        # ===== EXIT LEVELS (M15 ATR) =====
        if direction == "LONG":
            sl = price - 2.0 * m15_atr
            tp1 = price + 3.0 * m15_atr
        else:
            sl = price + 2.0 * m15_atr
            tp1 = price - 3.0 * m15_atr

        return {
            "direction": direction,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp1,
            "tp3": tp1,
            "confluence": 3,
            "entry_pattern": "pullback_first",
            "tp_splits": (1.0, 0.0, 0.0),
            "trail_atr_mult": 0,
            "max_bars": 180,  # 180 x 5min = 15 hours
            "no_breakeven": True,
        }
