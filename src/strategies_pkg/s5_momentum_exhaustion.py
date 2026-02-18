"""
Strategy 5: Momentum Exhaustion Reversal.

Entry TF: M15, Filter TF: H1 (structure/key levels).

Mandatory conditions (all 4 required):
  1. RSI divergence (price new extreme, RSI doesn't confirm)
  2. MACD histogram shrinking (2+ consecutive smaller bars)
  3. Near key level (within 0.75x ATR)
  4. Price overextended: moved 1.0x+ ATR from nearest EMA
  - Session filter: London/NY (08:00-17:00 UTC)

Mutual exclusion: Cannot fire if S4 ribbon is compressed.
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseStrategy
from ..indicators.technical import identify_key_levels


class S5_Momentum_Exhaustion(BaseStrategy):
    strategy_id = 5
    name = "S5_Momentum_Exhaustion"

    def __init__(self):
        super().__init__()
        self._cached_levels = None
        self._cache_idx = -1

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        if idx < 50:
            return None

        # Session filter: London + NY overlap (08:00-16:00 UTC)
        hour = current.name.hour if hasattr(current.name, 'hour') else 0
        if hour < 8 or hour >= 16:
            return None

        atr_val = current.get("atr_14", 0)
        if atr_val <= 0 or np.isnan(atr_val):
            return None

        # Condition 1: RSI divergence
        divergence = self._detect_divergence(data, idx)
        if divergence is None:
            return None

        # Condition 2: MACD histogram shrinking
        if not self._macd_shrinking(data, idx):
            return None

        # Condition 3: Near key level
        if not self._near_key_level(data, idx, current, atr_val):
            return None

        # Condition 4: Overextension from EMA
        if not self._overextended(current, atr_val):
            return None

        # Mutual exclusion with S4 ribbon
        if self._s4_active(current, atr_val):
            return None

        confluence = 4  # All mandatory met

        # Booster: declining volume
        if self._volume_declining(data, idx):
            confluence = 5

        # Require minimum confluence of 3
        if confluence < 3:
            return None

        close = current["close"]
        if divergence == "bullish":
            sl = close - 1.5 * atr_val
            tp1 = close + 1.5 * atr_val
            tp2 = close + 2.5 * atr_val
            tp3 = close + 4.0 * atr_val
            direction = "LONG"
        else:
            sl = close + 1.5 * atr_val
            tp1 = close - 1.5 * atr_val
            tp2 = close - 2.5 * atr_val
            tp3 = close - 4.0 * atr_val
            direction = "SHORT"

        return {
            "direction": direction,
            "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "confluence": confluence,
            "entry_pattern": "momentum_exhaustion",
            "tp_splits": (0.40, 0.40, 0.20),
            "trail_atr_mult": 1.5,
            "max_bars": 120,
        }

    def _detect_divergence(self, data, idx):
        """Detect RSI divergence over lookback window."""
        lookback = 20
        if idx < lookback + 3:
            return None

        rsi_col = "rsi_14"
        if rsi_col not in data.columns:
            return None

        window = data.iloc[idx - lookback:idx + 1]
        rsi_vals = window[rsi_col]
        lows = window["low"]
        highs = window["high"]

        # Bullish: current low is near the lowest in window, but RSI is higher
        current_low = data.iloc[idx]["low"]
        min_low_idx = lows.iloc[:-3].idxmin()  # exclude last 3 bars
        if min_low_idx is not None:
            prev_low = lows.loc[min_low_idx]
            if current_low <= prev_low * 1.001:  # current near or below prev low
                if rsi_vals.iloc[-1] > rsi_vals.loc[min_low_idx]:
                    return "bullish"

        # Bearish: current high is near highest, but RSI is lower
        current_high = data.iloc[idx]["high"]
        max_high_idx = highs.iloc[:-3].idxmax()
        if max_high_idx is not None:
            prev_high = highs.loc[max_high_idx]
            if current_high >= prev_high * 0.999:
                if rsi_vals.iloc[-1] < rsi_vals.loc[max_high_idx]:
                    return "bearish"

        return None

    def _macd_shrinking(self, data, idx):
        if idx < 3:
            return False
        h0 = abs(data.iloc[idx].get("macd_hist", 0))
        h1 = abs(data.iloc[idx - 1].get("macd_hist", 0))
        h2 = abs(data.iloc[idx - 2].get("macd_hist", 0))
        return h0 < h1 and h1 < h2  # 2 consecutive shrinks

    def _near_key_level(self, data, idx, current, atr_val):
        # Recache every 50 bars
        if self._cached_levels is None or idx - self._cache_idx >= 50:
            start = max(0, idx - 500)
            window = data.iloc[start:idx]
            self._cached_levels = identify_key_levels(
                window, lookback=5, tolerance_atr_mult=0.75, min_touches=2
            )
            self._cache_idx = idx

        if not self._cached_levels:
            return False

        close = current["close"]
        tolerance = 0.75 * atr_val  # relaxed from 0.5x

        for level_price, _ in self._cached_levels:
            if abs(close - level_price) <= tolerance:
                return True
        return False

    def _overextended(self, current, atr_val):
        close = current["close"]
        ema_50 = current.get("ema_50", np.nan)
        ema_100 = current.get("ema_100", np.nan)

        distances = []
        if not np.isnan(ema_50):
            distances.append(abs(close - ema_50))
        if not np.isnan(ema_100):
            distances.append(abs(close - ema_100))

        if not distances:
            return False

        # Relaxed from 1.5x to 1.0x ATR
        return min(distances) >= 1.0 * atr_val

    def _s4_active(self, current, atr_val):
        ema_20 = current.get("ema_20", np.nan)
        ema_50 = current.get("ema_50", np.nan)
        ema_100 = current.get("ema_100", np.nan)
        if any(np.isnan(v) for v in [ema_20, ema_50, ema_100]):
            return False
        ribbon_width = max(ema_20, ema_50, ema_100) - min(ema_20, ema_50, ema_100)
        return ribbon_width <= 1.0 * atr_val  # match S4's relaxed threshold

    def _volume_declining(self, data, idx):
        if idx < 5 or "volume" not in data.columns:
            return False
        vols = data["volume"].iloc[idx - 4:idx + 1].values
        if np.any(np.isnan(vols)):
            return False
        diffs = np.diff(vols)
        return np.sum(diffs < 0) >= 3
