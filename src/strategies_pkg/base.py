"""Base strategy interface."""
from typing import Optional
import pandas as pd


class BaseStrategy:
    strategy_id: int = 0
    name: str = "BaseStrategy"

    def check_signal(self, data: pd.DataFrame, idx: int,
                     current: pd.Series,
                     htf_row: Optional[pd.Series] = None) -> Optional[dict]:
        """
        Check for a trade signal at the current candle.

        Args:
            data: Full precomputed dataframe (must only access [:idx+1]).
            idx: Current bar index into data.
            current: The current candle (data.iloc[idx]).
            htf_row: Most recent fully-closed higher timeframe candle (or None).

        Returns:
            dict with keys: direction, sl, tp1, tp2, tp3, confluence,
                            tp_splits, trail_atr_mult, max_bars
            or None if no signal.
        """
        raise NotImplementedError
