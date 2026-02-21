"""
Strategy S18: Failed Breakout Fade / Liquidity Grab Reversal (M15).

Concept: False breakouts are a permanent market feature. Retail buys breakouts,
smart money fades. This is the INVERSE of S3 — catching failures instead of
genuine breakouts.

Entry conditions (SHORT — LONG is mirror):
  1. Identify 4H high: highest high of last 16 M15 bars before signal bar
  2. Signal bar (previous bar): high > high_4h AND close < high_4h (false breakout)
  3. Volume spike on signal bar: volume > 1.5 * SMA(volume, 20)
  4. Ranging: ADX(14) < 35
  5. Confirmation (current bar): close < signal_bar.close
  6. Entry: next bar open

Exit:
  - SL: high_4h + 0.5 * ATR(14) for shorts
  - TP: entry - 1.5 * ATR(14) for shorts — ~2:1 R:R
  - Trailing: Once 1.0x ATR profit, trail by 0.5x ATR

Target pairs: EUR_USD, GBP_USD, GBP_JPY, GBP_AUD
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy


class S18_FailedBreakout(BaseStrategy):
    strategy_id = 18
    name = "S18_Failed_Breakout"

    # Parameters from spec
    LOOKBACK_BARS = 16         # 16 M15 bars = 4 hours for high/low range
    MAX_ADX = 35               # Ranging regime gate
    VOL_MULT = 1.5             # Volume > 1.5 * SMA(volume, 20) on signal bar
    SL_ATR_MULT = 0.5          # SL = range_level + 0.5 * ATR beyond breakout
    TP_ATR_MULT = 1.5          # TP = 1.5 * ATR from entry
    TRAIL_ATR_MULT = 0.5       # Trail by 0.5 * ATR after 1.0 ATR profit
    MAX_BARS = 32              # Time stop: 32 bars = 8 hours on M15

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        # Warmup: need at least lookback + 2 bars (range + signal + confirmation)
        if idx < self.LOOKBACK_BARS + 2:
            return None

        # Current bar is confirmation bar; previous bar is signal bar
        signal_bar = data.iloc[idx - 1]
        confirm_bar = current

        # Regime gate: ranging market
        adx_val = signal_bar.get("adx_14", 50)
        if np.isnan(adx_val) or adx_val >= self.MAX_ADX:
            return None

        atr_14 = signal_bar.get("atr_14", 0)
        if atr_14 <= 0 or np.isnan(atr_14):
            return None

        # Compute 4H range: highest high and lowest low of 16 bars before signal bar
        range_start = idx - 1 - self.LOOKBACK_BARS
        range_end = idx - 1  # exclusive of signal bar
        if range_start < 0:
            return None

        range_window = data.iloc[range_start:range_end]
        high_4h = range_window["high"].max()
        low_4h = range_window["low"].min()

        if np.isnan(high_4h) or np.isnan(low_4h):
            return None

        # Volume spike on signal bar: volume > 1.5 * SMA(volume, 20)
        vol_start = max(0, idx - 1 - 20)
        vol_window = data["volume"].iloc[vol_start:idx - 1]
        if len(vol_window) < 5:
            return None
        vol_avg = vol_window.mean()
        signal_vol = signal_bar.get("volume", 0)
        if vol_avg <= 0 or signal_vol < self.VOL_MULT * vol_avg:
            return None

        direction = None

        # SHORT: false breakout above 4H high
        # Signal bar: high > high_4h AND close < high_4h
        # Confirmation: current close < signal bar close
        if (signal_bar["high"] > high_4h and
                signal_bar["close"] < high_4h and
                confirm_bar["close"] < signal_bar["close"]):
            direction = "SHORT"

        # LONG: false breakdown below 4H low
        # Signal bar: low < low_4h AND close > low_4h
        # Confirmation: current close > signal bar close
        if direction is None:
            if (signal_bar["low"] < low_4h and
                    signal_bar["close"] > low_4h and
                    confirm_bar["close"] > signal_bar["close"]):
                direction = "LONG"

        if direction is None:
            return None

        # Confluence scoring
        confluence = 2  # Base: false breakout + volume spike
        # Deeper false breakout (wick beyond range is large)
        if direction == "SHORT":
            wick_beyond = signal_bar["high"] - high_4h
            if wick_beyond > 0.5 * atr_14:
                confluence += 1
        else:
            wick_beyond = low_4h - signal_bar["low"]
            if wick_beyond > 0.5 * atr_14:
                confluence += 1
        # Very strong volume
        if signal_vol > 2.0 * vol_avg:
            confluence += 1
        # HTF alignment (soft)
        if htf_row is not None:
            htf_close = htf_row.get("close", np.nan)
            htf_ema200 = htf_row.get("ema_200", np.nan)
            if not np.isnan(htf_close) and not np.isnan(htf_ema200):
                if (direction == "LONG" and htf_close > htf_ema200) or \
                   (direction == "SHORT" and htf_close < htf_ema200):
                    confluence += 1

        # Exit levels
        entry_price = confirm_bar["close"]  # approximate; actual entry is next bar open
        if direction == "SHORT":
            sl = high_4h + self.SL_ATR_MULT * atr_14
            tp = entry_price - self.TP_ATR_MULT * atr_14
        else:
            sl = low_4h - self.SL_ATR_MULT * atr_14
            tp = entry_price + self.TP_ATR_MULT * atr_14

        return {
            "direction": direction,
            "sl": sl,
            "tp1": tp,
            "tp2": tp,
            "tp3": tp,
            "confluence": confluence,
            "entry_pattern": f"failed_breakout_{direction.lower()}",
            "tp_splits": (1.0, 0.0, 0.0),
            "trail_atr_mult": self.TRAIL_ATR_MULT,
            "max_bars": self.MAX_BARS,
        }
