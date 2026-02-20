"""
Strategy S14: London Fix Trade.

Concept: The WM/Reuters 4 PM London Fix creates predictable institutional
order flow. Large FX transactions cluster around 16:00 UTC, causing
directional moves in the 30-60 minutes before the fix, which often
reverse after completion.

This strategy trades the post-fix mean reversion: if price moved strongly
in one direction before the fix (15:00-16:00 UTC), enter the opposite
direction expecting reversion.

Adapted to M15 timeframe (1 trade per day target).

Entry conditions (ALL must be true):
  1. Time is 16:00-16:15 UTC (immediately after fix window)
  2. Pre-fix move: price moved > 0.5 ATR in one direction during 15:00-16:00
  3. Reversal candle: fix candle closes against the pre-fix direction
  4. Not Friday after 16:00 (weekend gap risk)

Exit:
  - SL: Beyond the pre-fix extreme + 0.5 ATR buffer
  - TP1: 50% retracement of pre-fix move
  - TP2: Full retracement to pre-fix level
  - Time exit: 18:00 UTC (2 hours max)
  - Max hold: 8 bars (2 hours on M15)
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S14_LondonFix(BaseStrategy):
    strategy_id = 14
    name = "S14_LondonFix"

    # Fix timing
    PRE_FIX_START_HOUR = 15   # Monitor pre-fix move from 15:00
    FIX_HOUR = 16             # Fix at 16:00 UTC
    FIX_ENTRY_MINUTE = 0      # Enter at 16:00

    # Pre-fix move threshold
    MIN_MOVE_ATR = 0.5        # Minimum pre-fix move to qualify

    # Risk management
    SL_ATR_BUFFER = 0.5       # Buffer beyond pre-fix extreme
    MAX_BARS = 8              # 2 hours on M15

    def __init__(self):
        super().__init__()
        self._daily_trade_cache = {}

    def _get_pre_fix_move(self, data, idx):
        """Calculate the pre-fix price move (15:00-16:00 UTC)."""
        current_time = data.index[idx]
        current_date = current_time.date()

        # Find bars between 15:00-16:00 UTC today
        pre_fix_bars = []
        for i in range(max(0, idx - 20), idx + 1):
            bar_time = data.index[i]
            if bar_time.date() != current_date:
                continue
            if self.PRE_FIX_START_HOUR <= bar_time.hour < self.FIX_HOUR:
                pre_fix_bars.append(i)

        if len(pre_fix_bars) < 2:
            return None

        pre_fix_data = data.iloc[pre_fix_bars]
        pre_fix_open = data.iloc[pre_fix_bars[0]]["open"]
        pre_fix_high = pre_fix_data["high"].max()
        pre_fix_low = pre_fix_data["low"].min()
        pre_fix_close = data.iloc[pre_fix_bars[-1]]["close"]

        move = pre_fix_close - pre_fix_open

        return {
            "open": pre_fix_open,
            "close": pre_fix_close,
            "high": pre_fix_high,
            "low": pre_fix_low,
            "move": move,
        }

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        if idx < 200:
            return None

        current_time = current.name
        hour = current_time.hour if hasattr(current_time, 'hour') else 0
        minute = current_time.minute if hasattr(current_time, 'minute') else 0

        # Only enter at 16:00 UTC
        if hour != self.FIX_HOUR or minute != self.FIX_ENTRY_MINUTE:
            return None

        # No Friday trades (weekend gap risk)
        dow = current_time.dayofweek if hasattr(current_time, 'dayofweek') else 0
        if dow == 4:
            return None

        # One trade per day
        current_date = current_time.date()
        if current_date in self._daily_trade_cache:
            return None

        atr_val = current.get("atr_14", 0)
        if atr_val <= 0 or np.isnan(atr_val):
            return None

        # Get pre-fix move
        pre_fix = self._get_pre_fix_move(data, idx)
        if pre_fix is None:
            return None

        move = pre_fix["move"]
        abs_move = abs(move)

        # Check minimum move threshold
        if abs_move < self.MIN_MOVE_ATR * atr_val:
            return None

        price = current["close"]

        # Trade the reversal of the pre-fix move
        if move > 0:
            # Pre-fix was bullish -> trade SHORT (reversal)
            direction = "SHORT"
            sl = pre_fix["high"] + self.SL_ATR_BUFFER * atr_val
            # TP1: 50% retracement, TP2: full retracement
            tp1 = price - 0.5 * abs_move
            tp2 = pre_fix["open"]
        else:
            # Pre-fix was bearish -> trade LONG (reversal)
            direction = "LONG"
            sl = pre_fix["low"] - self.SL_ATR_BUFFER * atr_val
            tp1 = price + 0.5 * abs_move
            tp2 = pre_fix["open"]

        # Min RR check
        sl_dist = abs(price - sl)
        tp1_dist = abs(tp1 - price)
        if sl_dist <= 0 or tp1_dist / sl_dist < 1.0:
            return None

        # Reversal candle confirmation: current candle closes against pre-fix direction
        candle_body = current["close"] - current["open"]
        if move > 0 and candle_body >= 0:
            return None  # No bearish reversal candle
        if move < 0 and candle_body <= 0:
            return None  # No bullish reversal candle

        self._daily_trade_cache[current_date] = True

        # Confluence
        confluence = 3  # Base: fix timing + pre-fix move + reversal candle

        # HTF alignment
        if htf_row is not None:
            htf_ema200 = htf_row.get("ema_200", np.nan)
            htf_close = htf_row.get("close", np.nan)
            if not np.isnan(htf_ema200) and not np.isnan(htf_close):
                if direction == "LONG" and htf_close > htf_ema200:
                    confluence += 1
                elif direction == "SHORT" and htf_close < htf_ema200:
                    confluence += 1

        # Large pre-fix move (> 1.0 ATR = stronger reversal expected)
        if abs_move > 1.0 * atr_val:
            confluence += 1

        # RSI extreme
        rsi_val = current.get("rsi_14", 50)
        if not np.isnan(rsi_val):
            if direction == "LONG" and rsi_val < 35:
                confluence += 1
            elif direction == "SHORT" and rsi_val > 65:
                confluence += 1

        return {
            "direction": direction,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp2,
            "confluence": confluence,
            "entry_pattern": f"london_fix_{direction.lower()}",
            "tp_splits": (0.50, 0.50, 0.0),
            "trail_atr_mult": 0,  # No trailing for mean reversion
            "max_bars": self.MAX_BARS,
        }
