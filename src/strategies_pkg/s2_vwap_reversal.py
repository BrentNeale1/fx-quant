"""
Strategy 2: Session VWAP Reversal.

Entry TF: M15, No HTF filter needed.

Entry conditions (LONG):
  - Price crosses below VWAP -2 sigma band
  - RSI < 30 (oversold confirmation)
  - Session filter: London/NY hours only (08:00-17:00 UTC)
  - Minimum band width: vwap_std > 0.3 * ATR

Entry conditions (SHORT):
  - Price crosses above VWAP +2 sigma band
  - RSI > 70 (overbought confirmation)

TP1: Return to VWAP, TP2: Opposite 0.5 sigma band
SL: Beyond session high/low OR 1x ATR (whichever tighter)
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S2_VWAP_Reversal(BaseStrategy):
    strategy_id = 2
    name = "S2_Session_VWAP_Reversal"

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        if idx < 30:
            return None

        # Session filter: only London + NY (08:00-17:00 UTC)
        hour = current.name.hour if hasattr(current.name, 'hour') else 0
        if hour < 8 or hour >= 17:
            return None

        vwap = current.get("session_vwap", None)
        upper_2 = current.get("vwap_upper_2", None)
        lower_2 = current.get("vwap_lower_2", None)

        if vwap is None or upper_2 is None or lower_2 is None:
            return None
        if any(np.isnan(v) for v in [vwap, upper_2, lower_2]):
            return None

        # Skip if bands are too tight (early in session, no deviation yet)
        vwap_std = current.get("vwap_std", 0)
        if vwap_std is None or np.isnan(vwap_std) or vwap_std <= 0:
            return None

        atr_val = current.get("atr_14", 0)
        if atr_val <= 0 or np.isnan(atr_val):
            return None

        # Minimum band width: std must be meaningful relative to ATR
        if vwap_std < 0.3 * atr_val:
            return None

        rsi = current.get("rsi_14", 50)
        if np.isnan(rsi):
            return None

        close = current["close"]
        prev = data.iloc[idx - 1]

        # Session high/low for SL
        start = max(0, idx - 80)
        session_high = data["high"].iloc[start:idx + 1].max()
        session_low = data["low"].iloc[start:idx + 1].min()

        # LONG: price below -2 sigma + RSI < 30 + price just crossed below
        if close < lower_2 and rsi < 30:
            prev_lower_2 = prev.get("vwap_lower_2", None)
            if prev_lower_2 is not None and not np.isnan(prev_lower_2):
                if prev["close"] >= prev_lower_2:
                    return self._build_long(close, vwap, atr_val, vwap_std,
                                            session_low, current)

        # SHORT: price above +2 sigma + RSI > 70 + price just crossed above
        if close > upper_2 and rsi > 70:
            prev_upper_2 = prev.get("vwap_upper_2", None)
            if prev_upper_2 is not None and not np.isnan(prev_upper_2):
                if prev["close"] <= prev_upper_2:
                    return self._build_short(close, vwap, atr_val, vwap_std,
                                             session_high, current)

        return None

    def _build_long(self, close, vwap, atr_val, vwap_std, session_low, current):
        sl_session = session_low - 0.5 * atr_val
        sl_atr = close - 1.5 * atr_val  # Wider SL for mean reversion
        sl = max(sl_session, sl_atr)

        tp1 = vwap
        tp2 = vwap + 0.5 * vwap_std
        tp3 = vwap + 1.5 * vwap_std

        confluence = 2
        if current.get("adx_14", 30) < 25:
            confluence += 1
        macd_h = current.get("macd_hist", 0)
        if macd_h > 0:
            confluence += 1

        return {
            "direction": "LONG",
            "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "confluence": confluence,
            "tp_splits": (0.50, 0.35, 0.15),
            "trail_atr_mult": 1.0,
            "max_bars": 80,
        }

    def _build_short(self, close, vwap, atr_val, vwap_std, session_high, current):
        sl_session = session_high + 0.5 * atr_val
        sl_atr = close + 1.5 * atr_val  # Wider SL for mean reversion
        sl = min(sl_session, sl_atr)

        tp1 = vwap
        tp2 = vwap - 0.5 * vwap_std
        tp3 = vwap - 1.5 * vwap_std

        confluence = 2
        if current.get("adx_14", 30) < 25:
            confluence += 1
        macd_h = current.get("macd_hist", 0)
        if macd_h < 0:
            confluence += 1

        return {
            "direction": "SHORT",
            "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "confluence": confluence,
            "tp_splits": (0.50, 0.35, 0.15),
            "trail_atr_mult": 1.0,
            "max_bars": 80,
        }
