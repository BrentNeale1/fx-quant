"""
Strategy S15: Momentum Continuation.

Concept: After a strong impulse move (>2 ATR in N bars), enter on the first
pullback in the direction of the impulse. The impulse signals institutional
commitment; the pullback offers a low-risk entry before continuation.

Adapted to M15 from original 5-minute specification (2-4 trades/day target).

Entry conditions (ALL must be true):
  1. Impulse detected: price moved > IMPULSE_ATR_MULT * ATR in last IMPULSE_BARS
  2. Pullback: price retraces to within PULLBACK_EMA_PROXIMITY * ATR of EMA(20)
  3. Pullback candle closes in impulse direction (continuation confirmation)
  4. ADX > 25 (trending environment)
  5. Session: 07:00-17:00 UTC
  6. HTF trend alignment

Exit:
  - SL: Beyond pullback extreme + 0.5 ATR
  - TP1: 1.5x risk distance (close 50%)
  - TP2: Extension to impulse length from pullback low/high (close 50%)
  - Trailing stop: 1.0 ATR after TP1
  - Max hold: 40 bars (~10 hours on M15)
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S15_MomentumContinuation(BaseStrategy):
    strategy_id = 15
    name = "S15_MomentumContinuation"

    # Impulse detection
    IMPULSE_ATR_MULT = 2.0   # Min impulse size in ATR units
    IMPULSE_BARS = 8         # Window to detect impulse (8 bars = 2 hours on M15)

    # Pullback parameters
    PULLBACK_EMA_PROXIMITY = 0.5  # Must be within 0.5 ATR of EMA(20)
    PULLBACK_MIN_BARS = 2         # Min bars of pullback
    PULLBACK_MAX_BARS = 12        # Max bars before pullback is stale

    # Risk management
    SL_ATR_BUFFER = 0.5
    TP1_RR_MULT = 1.5        # TP1 at 1.5x risk
    TP2_RR_MULT = 2.5        # TP2 at 2.5x risk
    TRAIL_ATR_MULT = 1.0

    # Filters
    MIN_ADX = 25
    SESSION_START = 7
    SESSION_END = 17

    MAX_BARS = 40

    def _detect_impulse(self, data, idx, atr_val):
        """Detect a recent impulse move."""
        if idx < self.IMPULSE_BARS + self.PULLBACK_MAX_BARS:
            return None

        # Look for impulse in the window before the current pullback zone
        # The impulse should have occurred IMPULSE_BARS bars ago, then pullback since
        for pb_end in range(idx - self.PULLBACK_MIN_BARS,
                            idx - self.PULLBACK_MAX_BARS, -1):
            imp_start = pb_end - self.IMPULSE_BARS
            if imp_start < 0:
                continue

            imp_open = data.iloc[imp_start]["open"]
            imp_close = data.iloc[pb_end]["close"]
            imp_move = imp_close - imp_open

            if abs(imp_move) >= self.IMPULSE_ATR_MULT * atr_val:
                # Found an impulse
                imp_high = data["high"].iloc[imp_start:pb_end + 1].max()
                imp_low = data["low"].iloc[imp_start:pb_end + 1].min()

                return {
                    "direction": "LONG" if imp_move > 0 else "SHORT",
                    "move": imp_move,
                    "high": imp_high,
                    "low": imp_low,
                    "end_idx": pb_end,
                }

        return None

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        if idx < 200:
            return None

        # Session filter
        hour = current.name.hour if hasattr(current.name, 'hour') else 0
        if hour < self.SESSION_START or hour >= self.SESSION_END:
            return None

        atr_val = current.get("atr_14", 0)
        if atr_val <= 0 or np.isnan(atr_val):
            return None

        # ADX filter
        adx_val = current.get("adx_14", 0)
        if np.isnan(adx_val):
            adx_val = 0
        if adx_val < self.MIN_ADX:
            return None

        # EMA(20) proximity check
        ema_20 = current.get("ema_20", np.nan)
        if np.isnan(ema_20):
            return None

        price = current["close"]
        ema_dist = abs(price - ema_20)

        if ema_dist > self.PULLBACK_EMA_PROXIMITY * atr_val:
            return None

        # Detect impulse
        impulse = self._detect_impulse(data, idx, atr_val)
        if impulse is None:
            return None

        direction = impulse["direction"]

        # Candle confirmation: closes in impulse direction
        candle_body = current["close"] - current["open"]
        if direction == "LONG" and candle_body <= 0:
            return None
        if direction == "SHORT" and candle_body >= 0:
            return None

        # Pullback confirmation: price has actually pulled back from impulse
        if direction == "LONG":
            # After bullish impulse, pullback should have brought price down
            pullback_low = data["low"].iloc[impulse["end_idx"]:idx + 1].min()
            pullback_depth = impulse["high"] - pullback_low
            if pullback_depth < 0.3 * abs(impulse["move"]):
                return None  # Not enough pullback
        else:
            pullback_high = data["high"].iloc[impulse["end_idx"]:idx + 1].max()
            pullback_depth = pullback_high - impulse["low"]
            if pullback_depth < 0.3 * abs(impulse["move"]):
                return None

        # HTF trend alignment
        htf_aligned = False
        if htf_row is not None:
            htf_ema200 = htf_row.get("ema_200", np.nan)
            htf_close = htf_row.get("close", np.nan)
            if not np.isnan(htf_ema200) and not np.isnan(htf_close):
                if direction == "LONG" and htf_close > htf_ema200:
                    htf_aligned = True
                elif direction == "SHORT" and htf_close < htf_ema200:
                    htf_aligned = True

        # SL and TP
        if direction == "LONG":
            pullback_extreme = data["low"].iloc[impulse["end_idx"]:idx + 1].min()
            sl = pullback_extreme - self.SL_ATR_BUFFER * atr_val
            risk = price - sl
            tp1 = price + self.TP1_RR_MULT * risk
            tp2 = price + self.TP2_RR_MULT * risk
        else:
            pullback_extreme = data["high"].iloc[impulse["end_idx"]:idx + 1].max()
            sl = pullback_extreme + self.SL_ATR_BUFFER * atr_val
            risk = sl - price
            tp1 = price - self.TP1_RR_MULT * risk
            tp2 = price - self.TP2_RR_MULT * risk

        if risk <= 0:
            return None

        # Confluence
        confluence = 3  # Base: impulse + pullback to EMA + continuation candle
        if htf_aligned:
            confluence += 1
        if adx_val > 35:
            confluence += 1  # Very strong trend

        # Volume confirmation
        vol = current.get("volume", 0)
        vol_avg = data["volume"].iloc[max(0, idx - 20):idx].mean()
        if vol_avg > 0 and vol > 1.3 * vol_avg:
            confluence += 1

        return {
            "direction": direction,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp2,
            "confluence": confluence,
            "entry_pattern": f"momentum_continuation_{direction.lower()}",
            "tp_splits": (0.50, 0.50, 0.0),
            "trail_atr_mult": self.TRAIL_ATR_MULT,
            "max_bars": self.MAX_BARS,
        }
