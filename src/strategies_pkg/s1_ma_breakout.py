"""
Strategy 1: Trendline Breakout-Retest.

4-step sequence identified on H1 chart with M15 entry:
  1. Identify trendline on H1 (3+ swing touches, linear regression)
  2. Breakout: H1 close beyond trendline with conviction
  3. Move away: Price moves away from trendline (confirms real break)
  4. Retest + Entry: Price pulls back to broken trendline on M15 → engulfing candle

Entry conditions (all must be true):
  - State machine in RETEST phase
  - M15 engulfing candle
  - M15 close within 1.0x ATR of projected trendline price
  - M15 EMA 50 aligns with direction
  - Session: London/NY overlap (13:00-16:00 UTC)
  - Confluence >= 2

SL: Projected trendline price +/- 0.5x ATR (behind the trendline)
TP1: Previous swing high/low (structure), fallback 1.5x ATR
TP2: Next key level or 2.5x ATR
TP3: 2x TP1 distance or 4x ATR (runner)
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy
from ..indicators.technical import (
    fit_trendline, project_trendline, swing_highs, swing_lows,
    is_bullish_engulfing, is_bearish_engulfing, identify_key_levels,
)


class S1_MA_Breakout(BaseStrategy):
    """Trendline Breakout-Retest strategy (renamed from MA Breakout)."""
    strategy_id = 1
    name = "S1_Trendline_Breakout_Retest"

    def __init__(self):
        super().__init__()
        self._trendlines = {"resistance": None, "support": None}
        self._tl_cache_idx = -1
        self._state = {
            "phase": "IDLE",
            "direction": None,
            "break_bar_idx": None,
            "break_price": None,
            "trendline": None,
            "max_dist": 0.0,
            "bars_since_break": 0,
        }
        # Performance: cached HTF timestamps for searchsorted
        self._htf_ts_cache = None
        self._last_htf_cutoff = -1  # tracks H1 bar changes for timeout

    def _htf_cutoff(self, htf: pd.DataFrame, ts: pd.Timestamp) -> int:
        """Return number of H1 bars strictly before ts, using searchsorted."""
        if self._htf_ts_cache is None:
            self._htf_ts_cache = htf.index
        # Normalize tz: strip tz from ts if HTF index is tz-naive, or vice versa
        if self._htf_ts_cache.tz is None and hasattr(ts, 'tz') and ts.tz is not None:
            ts = ts.tz_localize(None)
        elif self._htf_ts_cache.tz is not None and (not hasattr(ts, 'tz') or ts.tz is None):
            ts = ts.tz_localize(self._htf_ts_cache.tz)
        return int(self._htf_ts_cache.searchsorted(ts, side="left"))

    # ------------------------------------------------------------------
    # Trendline Detection (runs on H1 data)
    # ------------------------------------------------------------------

    def _detect_trendlines(self, htf: pd.DataFrame, n_valid: int):
        """
        Detect resistance and support trendlines from H1 swing points.
        n_valid = number of H1 bars before current M15 timestamp.
        Recalculates every 20 H1 bars.
        """
        if n_valid < 50:
            return

        if (self._tl_cache_idx >= 0 and
                n_valid - self._tl_cache_idx < 20):
            return

        self._tl_cache_idx = n_valid

        # Use last 200 H1 bars (no lookahead: only first n_valid bars)
        start = max(0, n_valid - 200)
        window = htf.iloc[start:n_valid]
        offset = start  # absolute index of window[0] in htf

        # Detect swing highs and lows
        sh_mask = swing_highs(window, lookback=5)
        sl_mask = swing_lows(window, lookback=5)

        # Resistance trendline from swing highs
        sh_indices = np.where(sh_mask.values)[0]
        if len(sh_indices) >= 3:
            recent_sh = sh_indices[-8:]
            sh_prices = window["high"].values[recent_sh]
            tl = fit_trendline(recent_sh, sh_prices)
            if tl is not None:
                tl["window_offset"] = offset
            self._trendlines["resistance"] = tl
        else:
            self._trendlines["resistance"] = None

        # Support trendline from swing lows
        sl_indices_arr = np.where(sl_mask.values)[0]
        if len(sl_indices_arr) >= 3:
            recent_sl = sl_indices_arr[-8:]
            sl_prices = window["low"].values[recent_sl]
            tl = fit_trendline(recent_sl, sl_prices)
            if tl is not None:
                tl["window_offset"] = offset
            self._trendlines["support"] = tl
        else:
            self._trendlines["support"] = None

    def _project_tl_at_htf_bar(self, tl: dict, htf_bar_idx: int) -> float:
        """Project trendline price at a given absolute HTF bar index."""
        window_rel_idx = htf_bar_idx - tl["window_offset"]
        return tl["slope"] * window_rel_idx + tl["intercept"]

    # ------------------------------------------------------------------
    # State Machine
    # ------------------------------------------------------------------

    def _update_state_machine(self, htf: pd.DataFrame, n_valid: int):
        """
        Check for breakout transitions on H1 data.
        n_valid = number of H1 bars strictly before current M15 timestamp.
        """
        if n_valid < 2:
            return

        last_h1_idx = n_valid - 1
        last_h1 = htf.iloc[last_h1_idx]

        # H1 ATR for thresholds
        h1_atr = last_h1.get("atr_14", 0)
        if h1_atr <= 0 or np.isnan(h1_atr):
            return

        phase = self._state["phase"]

        # Track H1 bar changes for timeout counter
        if phase != "IDLE":
            if last_h1_idx != self._last_htf_cutoff:
                self._last_htf_cutoff = last_h1_idx
                self._state["bars_since_break"] += 1
                if self._state["bars_since_break"] > 50:
                    self._reset_state()
                    return

        if phase == "IDLE":
            # Check for breakout above resistance -> LONG
            res_tl = self._trendlines.get("resistance")
            if res_tl is not None:
                tl_price = self._project_tl_at_htf_bar(res_tl, last_h1_idx)
                threshold = tl_price + 0.3 * h1_atr
                h1_close = last_h1["close"]
                h1_open = last_h1["open"]
                body_low = min(h1_close, h1_open)
                if h1_close > threshold and body_low > tl_price:
                    self._state = {
                        "phase": "MOVE_AWAY",
                        "direction": "LONG",
                        "break_bar_idx": last_h1_idx,
                        "break_price": h1_close,
                        "trendline": res_tl.copy(),
                        "max_dist": h1_close - tl_price,
                        "bars_since_break": 0,
                        "h1_atr": h1_atr,
                    }
                    self._last_htf_cutoff = last_h1_idx
                    return

            # Check for breakout below support -> SHORT
            sup_tl = self._trendlines.get("support")
            if sup_tl is not None:
                tl_price = self._project_tl_at_htf_bar(sup_tl, last_h1_idx)
                threshold = tl_price - 0.3 * h1_atr
                h1_close = last_h1["close"]
                h1_open = last_h1["open"]
                body_high = max(h1_close, h1_open)
                if h1_close < threshold and body_high < tl_price:
                    self._state = {
                        "phase": "MOVE_AWAY",
                        "direction": "SHORT",
                        "break_bar_idx": last_h1_idx,
                        "break_price": h1_close,
                        "trendline": sup_tl.copy(),
                        "max_dist": tl_price - h1_close,
                        "bars_since_break": 0,
                        "h1_atr": h1_atr,
                    }
                    self._last_htf_cutoff = last_h1_idx
                    return

        elif phase == "MOVE_AWAY":
            tl = self._state["trendline"]
            tl_price = self._project_tl_at_htf_bar(tl, last_h1_idx)
            h1_close = last_h1["close"]
            state_atr = self._state.get("h1_atr", h1_atr)

            if self._state["direction"] == "LONG":
                dist = h1_close - tl_price
                if dist > self._state["max_dist"]:
                    self._state["max_dist"] = dist
                if self._state["max_dist"] >= 0.5 * state_atr and dist < self._state["max_dist"]:
                    self._state["phase"] = "RETEST"
            else:  # SHORT
                dist = tl_price - h1_close
                if dist > self._state["max_dist"]:
                    self._state["max_dist"] = dist
                if self._state["max_dist"] >= 0.5 * state_atr and dist < self._state["max_dist"]:
                    self._state["phase"] = "RETEST"

    def _reset_state(self):
        self._state = {
            "phase": "IDLE",
            "direction": None,
            "break_bar_idx": None,
            "break_price": None,
            "trendline": None,
            "max_dist": 0.0,
            "bars_since_break": 0,
        }

    # ------------------------------------------------------------------
    # Confluence Scoring (0-5)
    # ------------------------------------------------------------------

    def _calc_confluence(self, data: pd.DataFrame, idx: int,
                         current: pd.Series, direction: str,
                         tl: dict) -> int:
        confluence = 0

        # Trendline R-squared > 0.90
        if tl.get("r_squared", 0) > 0.90:
            confluence += 1

        # Touch count >= 4
        if tl.get("touch_count", 0) >= 4:
            confluence += 1

        # Volume above 20-period average
        if "volume" in current.index:
            vol_avg = data["volume"].iloc[max(0, idx - 20):idx].mean()
            if vol_avg > 0 and current["volume"] > vol_avg:
                confluence += 1

        # RSI between 40-60
        rsi_val = current.get("rsi_14", 50)
        if not np.isnan(rsi_val) and 40 <= rsi_val <= 60:
            confluence += 1

        # MACD histogram confirms direction
        macd_h = current.get("macd_hist", 0)
        if not np.isnan(macd_h):
            if direction == "LONG" and macd_h > 0:
                confluence += 1
            elif direction == "SHORT" and macd_h < 0:
                confluence += 1

        return min(confluence, 5)

    # ------------------------------------------------------------------
    # Find structure-based TP levels from H1 data
    # ------------------------------------------------------------------

    def _find_structure_tp(self, htf: pd.DataFrame, n_valid: int,
                           direction: str, entry_price: float,
                           atr_val: float) -> tuple:
        """Find TP levels based on H1 swing structure and key levels."""
        if n_valid < 50:
            if direction == "LONG":
                return (entry_price + 1.5 * atr_val,
                        entry_price + 2.5 * atr_val,
                        entry_price + 4.0 * atr_val)
            else:
                return (entry_price - 1.5 * atr_val,
                        entry_price - 2.5 * atr_val,
                        entry_price - 4.0 * atr_val)

        start = max(0, n_valid - 100)
        window = htf.iloc[start:n_valid]

        if direction == "LONG":
            # TP1: previous swing high above entry
            sh_mask = swing_highs(window, lookback=5)
            sh_prices = window.loc[sh_mask, "high"]
            above = sh_prices[sh_prices > entry_price].sort_values()
            tp1 = above.iloc[0] if len(above) > 0 else entry_price + 1.5 * atr_val

            # TP2: next key level above TP1, or 2.5x ATR
            levels = identify_key_levels(window, lookback=5, min_touches=2)
            level_prices = [lv[0] for lv in levels if lv[0] > tp1]
            tp2 = min(level_prices) if level_prices else entry_price + 2.5 * atr_val

            # TP3: 2x TP1 distance or 4x ATR (runner)
            tp1_dist = tp1 - entry_price
            tp3 = entry_price + max(2.0 * tp1_dist, 4.0 * atr_val)

        else:  # SHORT
            sl_mask = swing_lows(window, lookback=5)
            sl_prices = window.loc[sl_mask, "low"]
            below = sl_prices[sl_prices < entry_price].sort_values(ascending=False)
            tp1 = below.iloc[0] if len(below) > 0 else entry_price - 1.5 * atr_val

            levels = identify_key_levels(window, lookback=5, min_touches=2)
            level_prices = [lv[0] for lv in levels if lv[0] < tp1]
            tp2 = max(level_prices) if level_prices else entry_price - 2.5 * atr_val

            tp1_dist = entry_price - tp1
            tp3 = entry_price - max(2.0 * tp1_dist, 4.0 * atr_val)

        # Ensure TP ordering makes sense
        if direction == "LONG":
            tp1 = max(tp1, entry_price + 0.5 * atr_val)
            tp2 = max(tp2, tp1 + 0.3 * atr_val)
            tp3 = max(tp3, tp2 + 0.3 * atr_val)
        else:
            tp1 = min(tp1, entry_price - 0.5 * atr_val)
            tp2 = min(tp2, tp1 - 0.3 * atr_val)
            tp3 = min(tp3, tp2 - 0.3 * atr_val)

        return tp1, tp2, tp3

    # ------------------------------------------------------------------
    # Reversal Pattern Detection
    # ------------------------------------------------------------------

    def _detect_reversal_pattern(self, data: pd.DataFrame, idx: int,
                                  current: pd.Series, direction: str) -> str:
        """
        Check for reversal patterns at the retest candle.
        Returns pattern name ('engulfing', 'pin_bar', 'strong_close') or None.
        """
        o, h, l, c = current["open"], current["high"], current["low"], current["close"]
        body = abs(c - o)
        full_range = h - l
        if full_range <= 0:
            return None

        if direction == "LONG":
            # 1. Bullish engulfing
            if is_bullish_engulfing(data, idx):
                return "engulfing"
            # 2. Bullish pin bar: lower wick >= 2x body AND close in upper 25%
            lower_wick = min(o, c) - l
            if body > 0 and lower_wick >= 2 * body and c >= l + 0.75 * full_range:
                return "pin_bar"
            # 3. Strong bullish close: body > 60% of range AND close > open
            if body > 0.60 * full_range and c > o:
                return "strong_close"
        else:  # SHORT
            # 1. Bearish engulfing
            if is_bearish_engulfing(data, idx):
                return "engulfing"
            # 2. Bearish pin bar: upper wick >= 2x body AND close in lower 25%
            upper_wick = h - max(o, c)
            if body > 0 and upper_wick >= 2 * body and c <= l + 0.25 * full_range:
                return "pin_bar"
            # 3. Strong bearish close: body > 60% of range AND close < open
            if body > 0.60 * full_range and c < o:
                return "strong_close"

        return None

    # ------------------------------------------------------------------
    # Main Signal Check
    # ------------------------------------------------------------------

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        if idx < 50:
            return None

        htf = self.htf_data
        if htf is None or len(htf) < 50:
            return None

        current_ts = current.name

        # Efficient HTF cutoff via searchsorted
        n_valid = self._htf_cutoff(htf, current_ts)
        if n_valid < 50:
            return None

        # Detect trendlines and update state machine (always, for tracking)
        self._detect_trendlines(htf, n_valid)
        self._update_state_machine(htf, n_valid)

        # Session filter: London + NY overlap (08:00-16:00 UTC)
        hour = current_ts.hour if hasattr(current_ts, 'hour') else 0
        if hour < 8 or hour >= 16:
            return None

        atr_val = current.get("atr_14", 0)
        if atr_val <= 0 or np.isnan(atr_val):
            return None

        # Only generate signals in RETEST phase
        if self._state["phase"] != "RETEST":
            return None

        direction = self._state["direction"]
        tl = self._state["trendline"]

        # Project trendline price at current H1 bar
        current_htf_idx = n_valid - 1
        tl_price = self._project_tl_at_htf_bar(tl, current_htf_idx)

        close = current["close"]

        # M15 close within 1.5x ATR of projected trendline
        dist_to_tl = abs(close - tl_price)
        if dist_to_tl > 1.5 * atr_val:
            return None

        # M15 reversal pattern confirmation (engulfing, pin bar, or strong close)
        entry_pattern = self._detect_reversal_pattern(data, idx, current, direction)
        if entry_pattern is None:
            return None

        # EMA 50 alignment
        ema_50 = current.get("ema_50", np.nan)
        if np.isnan(ema_50):
            return None
        if direction == "LONG" and close <= ema_50:
            return None
        if direction == "SHORT" and close >= ema_50:
            return None

        # Confluence scoring
        confluence = self._calc_confluence(data, idx, current, direction, tl)
        if confluence < 2:
            return None

        # SL: behind the trendline (0.5x ATR past TL)
        if direction == "LONG":
            sl = tl_price - 0.5 * atr_val
            # Validate SL is below entry (TL may have drifted above price)
            if sl >= close:
                return None
        else:
            sl = tl_price + 0.5 * atr_val
            if sl <= close:
                return None

        # TP levels: structure-based from H1 data
        tp1, tp2, tp3 = self._find_structure_tp(
            htf, n_valid, direction, close, atr_val
        )

        # Reset state after generating signal
        self._reset_state()

        return {
            "direction": direction,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "confluence": confluence,
            "entry_pattern": entry_pattern,
            "tp_splits": (0.50, 0.30, 0.20),
            "trail_atr_mult": 1.5,
            "max_bars": 200,
        }
