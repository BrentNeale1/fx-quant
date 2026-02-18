"""
Strategy 6A: EMA Bounce Continuation — Three-Checkpoint Momentum Filter.

Same core logic as S6 but with:
1. Three-checkpoint momentum filter on 1H timeframe
2. Two-part EMA separation filter (historical avg + current)

1H Momentum Filter (Three Checkpoints):
  avg_price_recent = mean(1H close) from 60..40 bars ago
  avg_price_middle = mean(1H close) from 110..90 bars ago
  avg_price_old    = mean(1H close) from 160..140 bars ago
  pct_change_recent = (avg_price_recent - avg_price_middle) / avg_price_middle * 100
  pct_change_older  = (avg_price_middle - avg_price_old) / avg_price_old * 100
  LONG:  both > 0.5
  SHORT: both < -0.5

EMA Separation (Two-Part):
  1. Avg |50 EMA - 100 EMA| over last 50 bars must be > 0.05% of price
  2. Current |50 EMA - 100 EMA| must be > 0.05% of price
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S6A_EMA_Bounce(BaseStrategy):
    strategy_id = 6
    name = "S6A_EMA_Bounce_ThreeCheckpoint"

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

        if htf_row is None or self.htf_data is None:
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
        # 1H THREE-CHECKPOINT MOMENTUM FILTER
        # ---------------------------------------------------------------
        timestamp = current.name
        htf = self.htf_data[self.htf_data.index < timestamp]
        if len(htf) < 161:
            return None

        avg_price_recent = htf["close"].iloc[-60:-40].mean()
        avg_price_middle = htf["close"].iloc[-110:-90].mean()
        avg_price_old = htf["close"].iloc[-160:-140].mean()

        if any(np.isnan(v) or v <= 0 for v in [avg_price_recent, avg_price_middle, avg_price_old]):
            return None

        pct_change_recent = (avg_price_recent - avg_price_middle) / avg_price_middle * 100
        pct_change_older = (avg_price_middle - avg_price_old) / avg_price_old * 100

        if direction == "LONG":
            if not (pct_change_recent > 0.25 and pct_change_older > 0.25):
                return None
        else:
            if not (pct_change_recent < -0.25 and pct_change_older < -0.25):
                return None

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
        # TWO-PART EMA SEPARATION FILTER
        # ---------------------------------------------------------------
        price = current["close"]
        current_sep = abs(ema_50 - ema_100)

        # Part 1: Average separation over last 50 bars > 0.05% of price
        if idx >= 50:
            seps = []
            for j in range(idx - 50, idx):
                bar = data.iloc[j]
                e50 = bar.get("ema_50", np.nan)
                e100 = bar.get("ema_100", np.nan)
                if not np.isnan(e50) and not np.isnan(e100):
                    seps.append(abs(e50 - e100))
            if len(seps) > 0:
                avg_sep = np.mean(seps)
                if avg_sep < 0.0003 * price:
                    return None
            else:
                return None
        else:
            return None

        # Part 2: Current separation > 0.05% of price
        if current_sep < 0.0003 * price:
            return None

        # ---------------------------------------------------------------
        # 15MIN EMA CONVERGENCE (prevents entries when trend weakening)
        # ---------------------------------------------------------------
        if idx >= 20:
            past_ema50 = data.iloc[idx - 20].get("ema_50", np.nan)
            past_ema100 = data.iloc[idx - 20].get("ema_100", np.nan)
            if not np.isnan(past_ema50) and not np.isnan(past_ema100):
                past_sep = abs(past_ema50 - past_ema100)
                if past_sep > 0 and current_sep < 0.70 * past_sep:
                    return None

        # ---------------------------------------------------------------
        # 15MIN GENUINE BOUNCE ENTRY
        # ---------------------------------------------------------------
        close = current["close"]

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

        # Pullback check
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

        # OHLC void
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

        # Bounce confirmation
        if direction == "LONG" and close <= ema_100:
            return None
        if direction == "SHORT" and close >= ema_100:
            return None

        # Price within 1.0 ATR of 100 EMA
        if abs(close - ema_100) > 1.0 * atr_val:
            return None

        # ---------------------------------------------------------------
        # REVERSAL PATTERN
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
            if body > 0 and lower_wick >= 2.0 * body and close_position >= 0.75:
                has_reversal = True
                pattern = "hammer"
            elif body / full_range > 0.60 and current["close"] > current["open"]:
                has_reversal = True
                pattern = "strong_bullish_close"
        else:
            if body > 0 and upper_wick >= 2.0 * body and close_position <= 0.25:
                has_reversal = True
                pattern = "shooting_star"
            elif body / full_range > 0.60 and current["close"] < current["open"]:
                has_reversal = True
                pattern = "strong_bearish_close"

        if not has_reversal:
            return None

        # ---------------------------------------------------------------
        # CONFIRMATION FILTERS
        # ---------------------------------------------------------------
        has_volume = False
        if "volume" in current.index:
            vol = current["volume"]
            vol_avg = data["volume"].iloc[max(0, idx - 20):idx].mean()
            if vol_avg > 0 and vol > 1.2 * vol_avg:
                has_volume = True
        if not has_volume:
            return None

        # RSI optional
        has_rsi = False
        for j in range(max(0, idx - 2), idx + 1):
            bar_rsi = data.iloc[j].get("rsi_14", 50)
            if direction == "LONG" and bar_rsi < 40:
                has_rsi = True
                break
            elif direction == "SHORT" and bar_rsi > 60:
                has_rsi = True
                break

        risk_pct = 0.015 if has_rsi else 0.01

        # ---------------------------------------------------------------
        # ENTRY, SL, TP LEVELS
        # ---------------------------------------------------------------
        entry = close
        ema_200 = current.get("ema_200", np.nan)
        if np.isnan(ema_200):
            return None

        if direction == "LONG":
            sl = ema_200 - 0.5 * atr_val
        else:
            sl = ema_200 + 0.5 * atr_val

        if direction == "LONG":
            tp1 = entry + 4.0 * atr_val
        else:
            tp1 = entry - 4.0 * atr_val

        tp2 = tp1

        if direction == "LONG":
            tp3 = entry + 20.0 * atr_val
        else:
            tp3 = entry - 20.0 * atr_val

        confirmations = 1 + (1 if has_rsi else 0)
        confluence = confirmations + 2

        return {
            "direction": direction,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "confluence": min(confluence, 5),
            "entry_pattern": f"ema_bounce_{pattern}",
            "tp_splits": (0.60, 0.0, 0.40),
            "trail_atr_mult": 5.0,
            "max_bars": 120,
            "no_breakeven": False,
            "risk_pct": risk_pct,
        }
