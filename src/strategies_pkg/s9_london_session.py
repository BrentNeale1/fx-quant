"""
Strategy S9: London Session Gap (Asian Range Breakout).

Concept: Price breaks out of the Asian session range at London open, driven
by institutional order flow from European/UK desks. Session-based volatility
patterns are among the most well-documented phenomena in FX
(Andersen & Bollerslev 1997, BIS data).

Entry conditions (ALL must be true):
  1. Asian range defined: 00:00-07:00 UTC high/low
  2. Asian range not too wide (< 1.5 ATR H1 and < pair-specific cap)
  3. Price breaks above Asian high (LONG) or below Asian low (SHORT)
     with a candle CLOSE beyond the level
  4. Volume > 3.0x Asian session average (first London candle almost always
     shows 2x, so 3x filters for meaningful surges)
  5. ADX > 20 (some trending context)
  6. Trade window: 07:00-10:00 UTC (London kill zone)

Exit:
  - SL: Opposite side of Asian range, capped at 1.5 ATR(H1) or pip limit
  - TP1: Asian range width as measured-move target (close 50%)
  - TP2: 2.0x Asian range width (close 50%)
  - Time exit: 17:00 UTC (captures full London-NY overlap)
  - Max hold: 40 bars (H1)
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S9_London_Session(BaseStrategy):
    strategy_id = 9
    name = "S9_London_Session"

    # Asian range window (UTC hours)
    ASIAN_START_HOUR = 0
    ASIAN_END_HOUR = 7

    # Entry window (UTC hours)
    ENTRY_START_HOUR = 7
    ENTRY_END_HOUR = 10

    # Exit time (UTC hour) — captures full London-NY overlap
    TIME_EXIT_HOUR = 17

    # Volume threshold (relaxed from 3.0 for H1 — Asian H1 bars aren't dramatically
    # lower volume than London H1 bars the way M15 bars would be)
    VOLUME_MULT = 1.5

    # Max Asian range (in pips) per pair category
    MAX_RANGE_PIPS = {
        "EUR_USD": 60, "GBP_USD": 80, "EUR_AUD": 80,
        "GBP_AUD": 100, "GBP_JPY": 100, "USD_JPY": 60,
        "EUR_CAD": 80, "GBP_CAD": 100, "EUR_GBP": 50,
    }

    # SL cap in ATR (widened from 1.5 — was filtering out most days)
    SL_ATR_CAP = 2.5

    MAX_BARS = 40

    # Per-pair filter overrides (set via constructor with pair= and filtered=True)
    # Each key maps to a dict of: min_adx, rsi_neutral_skip, skip_friday,
    # entry_start_hour, tp1_mult, min_ema50_dist_pips
    PAIR_FILTERS = {
        "EUR_USD": {
            "tp1_mult": 1.5,            # TP1 = 1.5x Asian range (was 1.0x)
            # NOTE: RSI filter and ADX hard gate tested but overfit — dropped
        },
        "GBP_AUD": {
            "min_adx": 25,              # require ADX > 25
            "skip_friday": True,        # drop Friday trades
            "skip_monday": True,        # Monday Asian ranges unreliable after weekend gaps
            "min_ema50_dist_pips": 40,  # require 40+ pips from EMA50
        },
    }

    def __init__(self, pair=None, filtered=False):
        super().__init__()
        self._asian_range_cache = {}  # date -> (high, low, avg_vol)
        self._pair = pair
        self._filtered = filtered
        self._pair_cfg = {}
        if filtered and pair and pair in self.PAIR_FILTERS:
            self._pair_cfg = self.PAIR_FILTERS[pair]

    def _get_pip_size(self, pair):
        if "JPY" in pair:
            return 0.01
        return 0.0001

    def _compute_asian_range(self, data, idx):
        """Compute Asian session range for the current day."""
        current_time = data.index[idx]
        current_date = current_time.date()

        if current_date in self._asian_range_cache:
            return self._asian_range_cache[current_date]

        # Find Asian session bars for today (00:00-07:00 UTC)
        asian_bars = []
        for i in range(max(0, idx - 50), idx + 1):
            bar_time = data.index[i]
            if bar_time.date() != current_date:
                continue
            bar_hour = bar_time.hour
            if self.ASIAN_START_HOUR <= bar_hour < self.ASIAN_END_HOUR:
                asian_bars.append(i)

        if len(asian_bars) < 3:
            return None

        asian_data = data.iloc[asian_bars]
        asian_high = asian_data["high"].max()
        asian_low = asian_data["low"].min()
        asian_avg_vol = asian_data["volume"].mean()

        result = (asian_high, asian_low, asian_avg_vol)
        self._asian_range_cache[current_date] = result
        return result

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        if idx < 200:
            return None

        # Entry window (per-pair override for start hour)
        hour = current.name.hour if hasattr(current.name, 'hour') else 0
        start_hour = self._pair_cfg.get("entry_start_hour", self.ENTRY_START_HOUR)
        if hour < start_hour or hour >= self.ENTRY_END_HOUR:
            return None

        # Day-of-week filters
        dow = current.name.dayofweek if hasattr(current.name, 'dayofweek') else 0
        if self._pair_cfg.get("skip_friday", False) and dow == 4:
            return None
        if self._pair_cfg.get("skip_monday", False) and dow == 0:
            return None

        atr_val = current.get("atr_14", 0)
        if atr_val <= 0 or np.isnan(atr_val):
            return None

        # Get HTF ATR for SL capping
        htf_atr = atr_val
        if htf_row is not None:
            htf_atr_val = htf_row.get("atr_14", np.nan)
            if not np.isnan(htf_atr_val) and htf_atr_val > 0:
                htf_atr = htf_atr_val

        # Compute Asian range
        asian = self._compute_asian_range(data, idx)
        if asian is None:
            return None

        asian_high, asian_low, asian_avg_vol = asian
        asian_range = asian_high - asian_low

        if asian_range <= 0:
            return None

        # Check Asian range not too wide
        pair = ""
        # Try to infer pair from strategy context; use default cap
        max_range_pips = 80  # default
        pip_size = self._get_pip_size("GBP_JPY" if atr_val > 0.005 else "EUR_USD")
        range_pips = asian_range / pip_size

        # Cap: skip if Asian range > 1.5 ATR(H1)
        if asian_range > self.SL_ATR_CAP * htf_atr:
            return None

        price = current["close"]

        # ADX filter — hard gate when filtered, soft confluence otherwise
        adx_val = current.get("adx_14", 0)
        if np.isnan(adx_val):
            adx_val = 0
        min_adx = self._pair_cfg.get("min_adx", 0)
        if min_adx > 0 and adx_val < min_adx:
            return None
        adx_strong = adx_val > 20

        # RSI neutral zone filter (EUR_USD: skip RSI 40-60 — no directional momentum)
        if self._pair_cfg.get("rsi_neutral_skip", False):
            rsi_val = current.get("rsi_14", 50)
            if not np.isnan(rsi_val) and 40 <= rsi_val <= 60:
                return None

        # EMA50 distance filter (GBP_AUD: close-to-EMA trades underperform)
        min_ema_dist = self._pair_cfg.get("min_ema50_dist_pips", 0)
        if min_ema_dist > 0:
            ema50 = current.get("ema_50", np.nan)
            if not np.isnan(ema50) and ema50 > 0:
                pip_sz = self._get_pip_size(self._pair or "EUR_USD")
                dist_pips = abs(price - ema50) / pip_sz
                if dist_pips < min_ema_dist:
                    return None

        # Volume check: current volume > 1.5x Asian average
        vol = current.get("volume", 0)
        if asian_avg_vol <= 0 or vol < self.VOLUME_MULT * asian_avg_vol:
            return None

        # Direction: breakout above or below Asian range
        direction = None
        if price > asian_high and current["close"] > asian_high:
            direction = "LONG"
        elif price < asian_low and current["close"] < asian_low:
            direction = "SHORT"

        if direction is None:
            return None

        # RSI directional filter: skip trades against RSI extremes
        if self._pair_cfg.get("rsi_directional_filter", False):
            rsi_val = current.get("rsi_14", 50)
            if not np.isnan(rsi_val):
                if direction == "LONG" and rsi_val > 60:
                    return None
                if direction == "SHORT" and rsi_val < 40:
                    return None

        # HTF trend alignment (soft: adds confluence but doesn't block)
        htf_aligned = False
        if htf_row is not None:
            htf_ema200 = htf_row.get("ema_200", np.nan)
            htf_close = htf_row.get("close", np.nan)
            if not np.isnan(htf_ema200) and not np.isnan(htf_close):
                if direction == "LONG" and htf_close > htf_ema200:
                    htf_aligned = True
                elif direction == "SHORT" and htf_close < htf_ema200:
                    htf_aligned = True

        confluence = 3 + (1 if htf_aligned else 0) + (1 if adx_strong else 0)

        # SL: opposite side of Asian range, capped
        tp1_mult = self._pair_cfg.get("tp1_mult", 1.0)

        if direction == "LONG":
            raw_sl = asian_low
            sl_distance = price - raw_sl
            max_sl_distance = self.SL_ATR_CAP * htf_atr
            if sl_distance > max_sl_distance:
                raw_sl = price - max_sl_distance
            sl = raw_sl

            tp1 = price + tp1_mult * asian_range
            tp2 = price + 2.0 * asian_range
        else:
            raw_sl = asian_high
            sl_distance = raw_sl - price
            max_sl_distance = self.SL_ATR_CAP * htf_atr
            if sl_distance > max_sl_distance:
                raw_sl = price + max_sl_distance
            sl = raw_sl

            tp1 = price - tp1_mult * asian_range
            tp2 = price - 2.0 * asian_range

        # Calculate max bars until 17:00 UTC time exit
        # On H1: roughly 17 - current_hour bars; on M15: (17-hour)*4
        # Use generic max_bars as fallback
        hours_remaining = self.TIME_EXIT_HOUR - hour
        if hours_remaining <= 0:
            return None

        return {
            "direction": direction,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp2,
            "confluence": confluence,
            "entry_pattern": f"london_breakout_{direction.lower()}",
            "tp_splits": (0.50, 0.50, 0.0),
            "trail_atr_mult": 1.5,
            "max_bars": self.MAX_BARS,
        }
