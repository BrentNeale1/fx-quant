"""
Strategy 1: MA Breakout-Retest (Option 3 - no trendlines).

Uses MA structure + key level breaks + candle confirmation.
Entry TF: M15, Filter TF: H1 (200 SMA directional filter).

Entry conditions (LONG):
  - EMA 50 > EMA 100 > EMA 200 (trend alignment)
  - Price pulls back to EMA 50 zone (within 1.0x ATR)
  - Bullish confirmation candle (close > open, close > prev close, body > 30% range)
  - H1 close > H1 200 SMA (HTF filter)
  - Session filter: London/NY hours only (08:00-17:00 UTC)
  - Confluence >= 2

Boosters (confluence 0-5):
  - Volume above 20-period average
  - RSI between 40-60 (not overextended)
  - MACD histogram positive and rising
  - ADX > 20 (trending)
  - Price above session VWAP
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S1_MA_Breakout(BaseStrategy):
    strategy_id = 1
    name = "S1_MA_Breakout_Retest"

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        if idx < 50:
            return None

        # Session filter: only London + NY (08:00-17:00 UTC)
        hour = current.name.hour if hasattr(current.name, 'hour') else 0
        if hour < 8 or hour >= 17:
            return None

        atr_val = current.get("atr_14", 0)
        if atr_val <= 0 or np.isnan(atr_val):
            return None

        ema_50 = current.get("ema_50", np.nan)
        ema_100 = current.get("ema_100", np.nan)
        ema_200 = current.get("ema_200", np.nan)
        if any(np.isnan(v) for v in [ema_50, ema_100, ema_200]):
            return None

        close = current["close"]
        open_p = current["open"]
        prev = data.iloc[idx - 1]
        prev_close = prev["close"]

        # Candle body filter: body must be > 30% of range (no dojis)
        body = abs(close - open_p)
        full_range = current["high"] - current["low"]
        if full_range <= 0 or body / full_range < 0.3:
            return None

        # LONG setup
        if ema_50 > ema_100 > ema_200:
            # HTF filter
            if htf_row is not None:
                htf_sma200 = htf_row.get("sma_200", np.nan)
                if not np.isnan(htf_sma200) and htf_row.get("close", 0) <= htf_sma200:
                    return None

            # Pullback to EMA 50 zone (within 1.0x ATR - tightened from 1.5x)
            dist_to_ema50 = close - ema_50
            if dist_to_ema50 < 0 or dist_to_ema50 > 1.0 * atr_val:
                return None

            # Bullish confirmation candle
            if not (close > open_p and close > prev_close):
                return None

            # Not too far from EMAs (avoid chasing)
            if close - ema_200 > 5 * atr_val:
                return None

            confluence = self._calc_confluence(data, idx, current, "LONG")

            # Require minimum confluence of 2
            if confluence < 2:
                return None

            sl = current["low"] - 0.5 * atr_val
            tp1 = close + 1.5 * atr_val
            tp2 = close + 2.5 * atr_val
            tp3 = close + 4.0 * atr_val

            return {
                "direction": "LONG",
                "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
                "confluence": confluence,
                "tp_splits": (0.50, 0.30, 0.20),
                "trail_atr_mult": 1.5,
                "max_bars": 200,
            }

        # SHORT setup
        if ema_50 < ema_100 < ema_200:
            if htf_row is not None:
                htf_sma200 = htf_row.get("sma_200", np.nan)
                if not np.isnan(htf_sma200) and htf_row.get("close", 0) >= htf_sma200:
                    return None

            dist_to_ema50 = ema_50 - close
            if dist_to_ema50 < 0 or dist_to_ema50 > 1.0 * atr_val:
                return None

            if not (close < open_p and close < prev_close):
                return None

            if ema_200 - close > 5 * atr_val:
                return None

            confluence = self._calc_confluence(data, idx, current, "SHORT")

            if confluence < 2:
                return None

            sl = current["high"] + 0.5 * atr_val
            tp1 = close - 1.5 * atr_val
            tp2 = close - 2.5 * atr_val
            tp3 = close - 4.0 * atr_val

            return {
                "direction": "SHORT",
                "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
                "confluence": confluence,
                "tp_splits": (0.50, 0.30, 0.20),
                "trail_atr_mult": 1.5,
                "max_bars": 200,
            }

        return None

    def _calc_confluence(self, data, idx, current, direction):
        confluence = 0

        # Volume above average
        if "volume" in current.index:
            vol_avg = data["volume"].iloc[max(0, idx - 20):idx].mean()
            if vol_avg > 0 and current["volume"] > vol_avg:
                confluence += 1

        # RSI between 40-60
        rsi = current.get("rsi_14", 50)
        if 40 <= rsi <= 60:
            confluence += 1

        # MACD histogram confirmation
        macd_h = current.get("macd_hist", 0)
        prev_macd_h = data.iloc[idx - 1].get("macd_hist", 0)
        if direction == "LONG" and macd_h > 0 and macd_h > prev_macd_h:
            confluence += 1
        elif direction == "SHORT" and macd_h < 0 and macd_h < prev_macd_h:
            confluence += 1

        # ADX > 20
        if current.get("adx_14", 0) > 20:
            confluence += 1

        # VWAP alignment
        vwap = current.get("session_vwap", 0)
        if vwap:
            if direction == "LONG" and current["close"] > vwap:
                confluence += 1
            elif direction == "SHORT" and current["close"] < vwap:
                confluence += 1

        return min(confluence, 5)
