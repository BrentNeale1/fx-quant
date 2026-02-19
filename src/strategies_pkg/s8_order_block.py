"""
Strategy S8: Order Block Retest.

Concept: Price returns to the last opposing candle before a strong impulse
move (displacement), and bounces from it. Treated as classical supply/demand
zones with strict confluence filters (not mythical "institutional footprints").

Entry conditions (ALL must be true):
  1. Detect a displacement candle: body >= 1.5 ATR (strong impulse)
  2. Identify the order block: last opposing candle before displacement
  3. Price returns to retest the OB zone (touches OB body range)
  4. Rejection candle at OB: pin bar (wick >= 2x body) or engulfing pattern
  5. At least 2 of 3 confluence factors:
     a) FVG exists within the impulse move
     b) OB is at a broken S/R level (structural confluence)
     c) Volume declining on pullback into OB
  6. HTF (H1) trend alignment via 200 EMA
  7. Session filter: 08:00-17:00 UTC

Exit:
  - SL: OB body low/high + 0.3 ATR buffer (NOT the full wick)
  - TP1: 1.5 ATR from entry (close 50%)
  - TP2: 3.0 ATR from entry (close 50%)
  - Max hold: 40 bars
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S8_Order_Block(BaseStrategy):
    strategy_id = 8
    name = "S8_Order_Block"

    # Tunable parameters
    DISPLACEMENT_ATR = 1.5     # Min body size for displacement candle
    DISPLACEMENT_VOL = 1.5     # Min volume ratio for displacement
    OB_LOOKBACK = 50           # How far back to search for OBs
    OB_RETEST_WINDOW = 30      # Max bars for price to retest OB after displacement
    SL_ATR_BUFFER = 0.3        # Buffer beyond OB body for SL
    TP1_ATR_MULT = 1.5
    TP2_ATR_MULT = 3.0
    MAX_BARS = 40

    def _find_order_blocks(self, data, idx, atr_val):
        """
        Find valid order blocks: last opposing candle before a displacement move.
        Returns list of dicts: {direction, ob_idx, ob_body_high, ob_body_low,
                                ob_high, ob_low, displacement_idx, has_fvg}
        """
        order_blocks = []
        start = max(0, idx - self.OB_LOOKBACK)

        for i in range(start + 1, idx - 2):
            curr = data.iloc[i]
            body = abs(curr["close"] - curr["open"])

            # Is this a displacement candle? (body >= 1.5 ATR)
            bar_atr = curr.get("atr_14", atr_val)
            if np.isnan(bar_atr) or bar_atr <= 0:
                bar_atr = atr_val
            if body < self.DISPLACEMENT_ATR * bar_atr:
                continue

            # Volume confirmation for displacement
            vol = curr.get("volume", 0)
            vol_avg = data["volume"].iloc[max(0, i - 20):i].mean()
            if vol_avg > 0 and vol < self.DISPLACEMENT_VOL * vol_avg:
                continue

            is_bullish_displacement = curr["close"] > curr["open"]
            is_bearish_displacement = curr["close"] < curr["open"]

            if not (is_bullish_displacement or is_bearish_displacement):
                continue

            # Find the order block: last OPPOSING candle before displacement
            ob_idx = None
            for j in range(i - 1, max(start, i - 10) - 1, -1):
                ob_candle = data.iloc[j]
                ob_bullish = ob_candle["close"] > ob_candle["open"]
                ob_bearish = ob_candle["close"] < ob_candle["open"]

                if is_bullish_displacement and ob_bearish:
                    ob_idx = j
                    break
                elif is_bearish_displacement and ob_bullish:
                    ob_idx = j
                    break

            if ob_idx is None:
                continue

            ob = data.iloc[ob_idx]
            ob_body_high = max(ob["open"], ob["close"])
            ob_body_low = min(ob["open"], ob["close"])

            # Check for FVG in the impulse move
            has_fvg = False
            if i >= 2:
                candle_before = data.iloc[i - 1]
                candle_after_idx = min(i + 1, len(data) - 1)
                candle_after = data.iloc[candle_after_idx]
                if is_bullish_displacement:
                    # Bullish FVG: candle[i-1].high < candle[i+1].low
                    if candle_before["high"] < candle_after["low"]:
                        has_fvg = True
                else:
                    # Bearish FVG: candle[i-1].low > candle[i+1].high
                    if candle_before["low"] > candle_after["high"]:
                        has_fvg = True

            direction = "LONG" if is_bullish_displacement else "SHORT"
            order_blocks.append({
                "direction": direction,
                "ob_idx": ob_idx,
                "ob_body_high": ob_body_high,
                "ob_body_low": ob_body_low,
                "ob_high": ob["high"],
                "ob_low": ob["low"],
                "displacement_idx": i,
                "has_fvg": has_fvg,
            })

        return order_blocks

    def _is_rejection_candle(self, candle, direction):
        """Check if candle shows rejection (pin bar or engulfing-like)."""
        body = abs(candle["close"] - candle["open"])
        full_range = candle["high"] - candle["low"]
        if full_range <= 0:
            return False

        if direction == "LONG":
            lower_wick = min(candle["open"], candle["close"]) - candle["low"]
            # Pin bar: lower wick >= 2x body, close in upper 40%
            if lower_wick >= 2 * body and (candle["close"] - candle["low"]) / full_range >= 0.6:
                return True
            # Bullish candle with strong close
            if candle["close"] > candle["open"] and body / full_range >= 0.5:
                return True
        else:
            upper_wick = candle["high"] - max(candle["open"], candle["close"])
            # Pin bar: upper wick >= 2x body, close in lower 40%
            if upper_wick >= 2 * body and (candle["high"] - candle["close"]) / full_range >= 0.6:
                return True
            # Bearish candle with strong close
            if candle["close"] < candle["open"] and body / full_range >= 0.5:
                return True

        return False

    def _check_volume_declining(self, data, displacement_idx, idx):
        """Check if volume is declining on the pullback to OB."""
        if idx <= displacement_idx + 2:
            return False
        displacement_vol = data["volume"].iloc[displacement_idx]
        pullback_vol = data["volume"].iloc[displacement_idx + 1:idx + 1].mean()
        return pullback_vol < 0.8 * displacement_vol

    def _is_at_broken_sr(self, data, idx, ob_price, atr_val):
        """Check if OB is at a level where prior S/R was broken (structural confluence)."""
        # Look for swing highs/lows near the OB price that were broken
        tolerance = 0.5 * atr_val
        lookback_start = max(0, idx - 200)

        for i in range(lookback_start, idx - 20):
            bar = data.iloc[i]
            is_sh = bar.get("is_swing_high", False)
            is_sl_point = bar.get("is_swing_low", False)

            if is_sh and abs(bar["high"] - ob_price) < tolerance:
                return True
            if is_sl_point and abs(bar["low"] - ob_price) < tolerance:
                return True
        return False

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        if idx < 200:
            return None

        # Session filter
        hour = current.name.hour if hasattr(current.name, 'hour') else 0
        if hour < 8 or hour >= 17:
            return None

        atr_val = current.get("atr_14", 0)
        if atr_val <= 0 or np.isnan(atr_val):
            return None

        # HTF trend alignment
        if htf_row is None:
            return None
        htf_ema200 = htf_row.get("ema_200", np.nan)
        htf_close = htf_row.get("close", np.nan)
        if np.isnan(htf_ema200) or np.isnan(htf_close):
            return None

        price = current["close"]

        # Find order blocks
        order_blocks = self._find_order_blocks(data, idx, atr_val)

        for ob in order_blocks:
            # Only trade OBs aligned with HTF trend
            if ob["direction"] == "LONG" and htf_close < htf_ema200:
                continue
            if ob["direction"] == "SHORT" and htf_close > htf_ema200:
                continue

            # Check if price is retesting the OB zone
            # For LONG: price should be in or near the OB body zone (pullback down into it)
            if ob["direction"] == "LONG":
                if not (current["low"] <= ob["ob_body_high"] and price >= ob["ob_body_low"]):
                    continue
            else:
                if not (current["high"] >= ob["ob_body_low"] and price <= ob["ob_body_high"]):
                    continue

            # Check OB is not too old (retest within window)
            bars_since = idx - ob["displacement_idx"]
            if bars_since > self.OB_RETEST_WINDOW or bars_since < 3:
                continue

            # Rejection candle check
            if not self._is_rejection_candle(current, ob["direction"]):
                continue

            # Confluence scoring (need 2 of 3)
            confluence_count = 0
            if ob["has_fvg"]:
                confluence_count += 1
            if self._check_volume_declining(data, ob["displacement_idx"], idx):
                confluence_count += 1
            ob_mid = (ob["ob_body_high"] + ob["ob_body_low"]) / 2
            if self._is_at_broken_sr(data, idx, ob_mid, atr_val):
                confluence_count += 1

            if confluence_count < 2:
                continue

            # Build exit levels using OB BODY (not wick) + buffer
            if ob["direction"] == "LONG":
                sl = ob["ob_body_low"] - self.SL_ATR_BUFFER * atr_val
                tp1 = price + self.TP1_ATR_MULT * atr_val
                tp2 = price + self.TP2_ATR_MULT * atr_val
            else:
                sl = ob["ob_body_high"] + self.SL_ATR_BUFFER * atr_val
                tp1 = price - self.TP1_ATR_MULT * atr_val
                tp2 = price - self.TP2_ATR_MULT * atr_val

            return {
                "direction": ob["direction"],
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp2,
                "confluence": min(confluence_count + 2, 5),
                "entry_pattern": f"order_block_retest_{ob['direction'].lower()}",
                "tp_splits": (0.50, 0.50, 0.0),
                "trail_atr_mult": 1.5,
                "max_bars": self.MAX_BARS,
            }

        return None
