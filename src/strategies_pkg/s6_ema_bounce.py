"""
Strategy 6: EMA Bounce Continuation (v4).

Based on Brent's manual trading approach.
Entry TF: M15, Filter TF: H1.

Concept: Enter on pullbacks to EMAs during strong trends,
confirmed by reversal candles (hammer, strong close).

1H Trend Filter:
  LONG: Price > 200 EMA AND 50 EMA > 200 EMA
  SHORT: Price < 200 EMA AND 50 EMA < 200 EMA

15min EMA Setup (prevents counter-trend):
  LONG: 50 EMA > 100 EMA
  SHORT: 50 EMA < 100 EMA

15min EMA Separation (prevents ranging):
  |50 EMA - 100 EMA| > 0.5 ATR

15min EMA Convergence Filter:
  Current EMA separation must be >= separation from 10 bars ago.
  If shrinking, EMAs are converging and trend is weakening — void.

15min Genuine Bounce Entry:
  - Pre-pullback: >= 70% of bars [idx-10..idx-3] on CORRECT side of 100 EMA
  - Pullback: at least 1 of last 3 bars closed on OTHER side of 100 EMA
  - OHLC void: if last 3 candles ENTIRELY on wrong side of 100 EMA, void
    (sustained cross = trend change, not a pullback)
  - Bounce: current candle closes on CORRECT side of 100 EMA
  - Price within 1.0 ATR of 100 EMA
  - Reversal pattern: hammer/shooting star/strong close

Confirmations:
  - Volume > 1.2x 20-period avg (REQUIRED)
  - RSI < 40 (LONG) or > 60 (SHORT) in last 3 bars (OPTIONAL, larger position)

SL: 15min 200 EMA +/- 0.5 ATR buffer
TP1: Fixed 4.0 ATR from entry (close 60%)
Runner: 40% managed by 5.0 ATR trailing stop, floored at entry price

Session: 08:00-16:00 UTC
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S6_EMA_Bounce(BaseStrategy):
    strategy_id = 6
    name = "S6_EMA_Bounce_Continuation"

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        if idx < 200:
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

        # ---------------------------------------------------------------
        # 1H TREND FILTER
        # ---------------------------------------------------------------
        htf_close = htf_row.get("close", np.nan)
        htf_ema50 = htf_row.get("ema_50", np.nan)
        htf_ema200 = htf_row.get("ema_200", np.nan)
        if any(np.isnan(v) for v in [htf_close, htf_ema50, htf_ema200]):
            return None

        long_trend = htf_close > htf_ema200 and htf_ema50 > htf_ema200
        short_trend = htf_close < htf_ema200 and htf_ema50 < htf_ema200
        if not long_trend and not short_trend:
            return None

        direction = "LONG" if long_trend else "SHORT"

        # ---------------------------------------------------------------
        # 15MIN EMA SETUP (prevents counter-trend entries)
        # ---------------------------------------------------------------
        ema_50 = current.get("ema_50", np.nan)
        ema_100 = current.get("ema_100", np.nan)
        if np.isnan(ema_50) or np.isnan(ema_100):
            return None

        if direction == "LONG" and not (ema_50 > ema_100):
            return None
        if direction == "SHORT" and not (ema_50 < ema_100):
            return None

        # ---------------------------------------------------------------
        # 15MIN EMA SEPARATION (prevents ranging market entries)
        # ---------------------------------------------------------------
        ema_separation = abs(ema_50 - ema_100)
        if ema_separation <= 0.5 * atr_val:
            return None

        # ---------------------------------------------------------------
        # 15MIN EMA CONVERGENCE (prevents entries when trend weakening)
        # Allow minor convergence during pullback (natural), but reject
        # if EMAs have lost >30% of their separation over 20 bars.
        # ---------------------------------------------------------------
        if idx >= 20:
            past_ema50 = data.iloc[idx - 20].get("ema_50", np.nan)
            past_ema100 = data.iloc[idx - 20].get("ema_100", np.nan)
            if not np.isnan(past_ema50) and not np.isnan(past_ema100):
                past_sep = abs(past_ema50 - past_ema100)
                if past_sep > 0 and ema_separation < 0.70 * past_sep:
                    return None

        # ---------------------------------------------------------------
        # 15MIN GENUINE BOUNCE ENTRY
        # ---------------------------------------------------------------
        close = current["close"]

        # 1. PRE-PULLBACK TREND: >= 70% of bars [idx-10..idx-3] must have
        #    been on the CORRECT side of the 100 EMA.
        #    This prevents entries where price was ranging around the EMA.
        lookback_start = max(0, idx - 10)
        lookback_end = max(0, idx - 3)
        total_check_bars = lookback_end - lookback_start
        if total_check_bars < 4:
            return None

        trend_side_count = 0
        for j in range(lookback_start, lookback_end):
            bar = data.iloc[j]
            bar_ema100 = bar.get("ema_100", np.nan)
            if np.isnan(bar_ema100):
                continue
            if direction == "LONG" and bar["close"] > bar_ema100:
                trend_side_count += 1
            elif direction == "SHORT" and bar["close"] < bar_ema100:
                trend_side_count += 1

        if trend_side_count / total_check_bars < 0.70:
            return None

        # 2. PULLBACK: at least 1 of last 3 bars closed on OTHER side
        had_pullback = False
        for j in range(max(0, idx - 3), idx):
            bar_close = data.iloc[j]["close"]
            bar_ema100 = data.iloc[j].get("ema_100", np.nan)
            if np.isnan(bar_ema100):
                continue
            if direction == "LONG" and bar_close < bar_ema100:
                had_pullback = True
                break
            elif direction == "SHORT" and bar_close > bar_ema100:
                had_pullback = True
                break
        if not had_pullback:
            return None

        # 2b. OHLC VOID: if last 3 candles ALL have their ENTIRE range
        #     on the wrong side of 100 EMA, this is a sustained cross
        #     (trend change), not a brief pullback. Void the trade.
        if idx >= 3:
            all_wrong_side = True
            for j in range(idx - 3, idx):
                bar = data.iloc[j]
                bar_ema100 = bar.get("ema_100", np.nan)
                if np.isnan(bar_ema100):
                    all_wrong_side = False
                    break
                if direction == "LONG" and bar["high"] >= bar_ema100:
                    all_wrong_side = False
                    break
                elif direction == "SHORT" and bar["low"] <= bar_ema100:
                    all_wrong_side = False
                    break
            if all_wrong_side:
                return None

        # 3. BOUNCE: current candle closes on CORRECT side of 100 EMA
        if direction == "LONG" and close <= ema_100:
            return None
        if direction == "SHORT" and close >= ema_100:
            return None

        # 4. Price within 1.0 ATR of 100 EMA
        if abs(close - ema_100) > 1.0 * atr_val:
            return None

        # ---------------------------------------------------------------
        # REVERSAL PATTERN (accept ANY of these)
        # ---------------------------------------------------------------
        body = abs(current["close"] - current["open"])
        full_range = current["high"] - current["low"]
        if full_range <= 0:
            return None

        upper_wick = current["high"] - max(current["close"], current["open"])
        lower_wick = min(current["close"], current["open"]) - current["low"]
        close_position = (current["close"] - current["low"]) / full_range

        has_reversal = False
        pattern = ""

        if direction == "LONG":
            # Hammer: lower wick >= 2x body, closes in upper 25%
            if body > 0 and lower_wick >= 2.0 * body and close_position >= 0.75:
                has_reversal = True
                pattern = "hammer"
            # Strong Bullish Close: body > 60% of range, bullish candle
            elif body / full_range > 0.60 and current["close"] > current["open"]:
                has_reversal = True
                pattern = "strong_bullish_close"
        else:
            # Shooting Star: upper wick >= 2x body, closes in lower 25%
            if body > 0 and upper_wick >= 2.0 * body and close_position <= 0.25:
                has_reversal = True
                pattern = "shooting_star"
            # Strong Bearish Close: body > 60% of range, bearish candle
            elif body / full_range > 0.60 and current["close"] < current["open"]:
                has_reversal = True
                pattern = "strong_bearish_close"

        if not has_reversal:
            return None

        # ---------------------------------------------------------------
        # CONFIRMATION FILTERS (Volume required, RSI optional)
        # ---------------------------------------------------------------

        # Volume > 1.2x 20-period average (REQUIRED)
        has_volume = False
        if "volume" in current.index:
            vol = current["volume"]
            vol_avg = data["volume"].iloc[max(0, idx - 20):idx].mean()
            if vol_avg > 0 and vol > 1.2 * vol_avg:
                has_volume = True
        if not has_volume:
            return None

        # RSI < 40 (LONG) or > 60 (SHORT) in last 3 bars (OPTIONAL)
        has_rsi = False
        for j in range(max(0, idx - 2), idx + 1):
            bar_rsi = data.iloc[j].get("rsi_14", 50)
            if direction == "LONG" and bar_rsi < 40:
                has_rsi = True
                break
            elif direction == "SHORT" and bar_rsi > 60:
                has_rsi = True
                break

        # Larger position if RSI confirms (1.5% vs 1%)
        risk_pct = 0.015 if has_rsi else 0.01

        # ---------------------------------------------------------------
        # ENTRY, SL, TP LEVELS
        # ---------------------------------------------------------------
        entry = close

        # SL: 15min 200 EMA +/- 0.5 ATR buffer
        ema_200 = current.get("ema_200", np.nan)
        if np.isnan(ema_200):
            return None

        if direction == "LONG":
            sl = ema_200 - 0.5 * atr_val
        else:
            sl = ema_200 + 0.5 * atr_val

        # TP1: fixed 4.0 ATR from entry (close 60%)
        if direction == "LONG":
            tp1 = entry + 4.0 * atr_val
        else:
            tp1 = entry - 4.0 * atr_val

        # TP2: set equal to TP1 so it triggers immediately (activates trailing)
        tp2 = tp1

        # TP3: very far target — 5 ATR trailing stop manages the runner exit
        if direction == "LONG":
            tp3 = entry + 20.0 * atr_val
        else:
            tp3 = entry - 20.0 * atr_val

        confirmations = 1 + (1 if has_rsi else 0)  # Volume + optional RSI
        confluence = confirmations + 2  # +2 for trend + pattern

        return {
            "direction": direction,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "confluence": min(confluence, 5),
            "entry_pattern": f"ema_bounce_{pattern}",
            "tp_splits": (0.60, 0.0, 0.40),  # 60% at TP1, 0% at TP2, 40% runner
            "trail_atr_mult": 5.0,
            "max_bars": 120,
            "no_breakeven": False,  # Breakeven after TP1 = trailing floor at entry
            "risk_pct": risk_pct,
        }

    def _is_bullish_engulfing(self, data: pd.DataFrame, idx: int) -> bool:
        prev = data.iloc[idx - 1]
        curr = data.iloc[idx]
        prev_body = abs(prev["close"] - prev["open"])
        curr_body = abs(curr["close"] - curr["open"])
        return (prev["close"] < prev["open"] and      # prev bearish
                curr["close"] > curr["open"] and       # curr bullish
                curr_body > prev_body and              # engulfs
                curr["open"] <= prev["close"] and
                curr["close"] >= prev["open"])

    def _is_bearish_engulfing(self, data: pd.DataFrame, idx: int) -> bool:
        prev = data.iloc[idx - 1]
        curr = data.iloc[idx]
        prev_body = abs(prev["close"] - prev["open"])
        curr_body = abs(curr["close"] - curr["open"])
        return (prev["close"] > prev["open"] and      # prev bullish
                curr["close"] < curr["open"] and       # curr bearish
                curr_body > prev_body and              # engulfs
                curr["open"] >= prev["close"] and
                curr["close"] <= prev["open"])

    def _find_next_htf_level(self, entry_price: float, direction: str,
                             atr_val: float,
                             timestamp) -> Optional[float]:
        """Find next 1H key S/R level from HTF swing points."""
        if self.htf_data is None:
            return None

        htf = self.htf_data[self.htf_data.index < timestamp]
        if len(htf) < 50:
            return None

        htf_recent = htf.iloc[-200:]

        if direction == "LONG":
            mask = htf_recent.get("is_swing_high",
                                  pd.Series(False, index=htf_recent.index))
            swing_prices = htf_recent.loc[mask == True, "high"]
            if len(swing_prices) == 0:
                return None
            above = swing_prices[swing_prices > entry_price + 0.5 * atr_val]
            if len(above) == 0:
                return None
            return float(above.min())
        else:
            mask = htf_recent.get("is_swing_low",
                                  pd.Series(False, index=htf_recent.index))
            swing_prices = htf_recent.loc[mask == True, "low"]
            if len(swing_prices) == 0:
                return None
            below = swing_prices[swing_prices < entry_price - 0.5 * atr_val]
            if len(below) == 0:
                return None
            return float(below.max())
