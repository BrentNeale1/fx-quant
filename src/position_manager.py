"""
Shared position management for backtester and live engine.

Extracted from backtester/engine.py to allow both the backtest engine
and the live trading engine to share the same position update logic.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Spread Configuration (in pips)
# ---------------------------------------------------------------------------
SPREAD_PIPS = {
    "EUR_USD": 1.5, "GBP_USD": 1.5,
    "GBP_AUD": 2.0, "EUR_AUD": 2.0,
    "EUR_GBP": 2.0, "GBP_JPY": 2.5,
    "USD_JPY": 1.5, "GBP_CAD": 2.5,
    "EUR_CAD": 2.5, "EUR_NZD": 2.5,
    "GBP_NZD": 2.5,
}

# Pip value per pair
PIP_SIZE = {
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "EUR_AUD": 0.0001,
    "GBP_AUD": 0.0001, "EUR_GBP": 0.0001, "GBP_CAD": 0.0001,
    "EUR_CAD": 0.0001, "EUR_NZD": 0.0001,
    "GBP_NZD": 0.0001,
    "USD_JPY": 0.01, "GBP_JPY": 0.01,
}


@dataclass
class Position:
    """Represents an open position."""
    entry_time: pd.Timestamp
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    tp3_price: float
    initial_size: float
    current_size: float
    tp_splits: tuple  # e.g. (0.40, 0.40, 0.20)
    trail_atr_mult: float
    max_bars: int
    bars_held: int = 0
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    trailing_sl: Optional[float] = None
    strategy_id: int = 0
    confluence_score: int = 0
    signal_features: dict = field(default_factory=dict)
    realized_pnl: float = 0.0
    no_breakeven: bool = False


@dataclass
class TradeRecord:
    """Rich trade log entry for Phase 3 feature engineering."""
    timestamp: pd.Timestamp = None
    strategy_id: int = 0
    pair: str = ""
    signal_direction: str = ""
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    tp3_price: float = 0.0
    lot_size: float = 0.0
    confluence_score: int = 0
    session: str = ""
    spread_at_entry: float = 0.0
    atr_at_entry: float = 0.0
    adx_at_entry: float = 0.0
    rsi_at_entry: float = 0.0
    ema_50_value: float = 0.0
    ema_200_value: float = 0.0
    vwap_deviation: float = 0.0
    news_within_60min: bool = False
    macd_hist_at_entry: float = 0.0
    stoch_k_at_entry: float = 0.0
    distance_from_ema50_pips: float = 0.0
    candle_body_ratio: float = 0.0
    hour_of_day: int = 0
    day_of_week: int = 0
    entry_pattern: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""
    exit_time: pd.Timestamp = None
    pnl_pips: float = 0.0
    pnl_dollars: float = 0.0
    hold_time_minutes: int = 0
    win: bool = False


class PositionManager:
    """Manages open positions and produces TradeRecords on close.

    Used by both the Backtester and the LiveEngine so that position
    update logic (SL/TP/trail/time exits, partial closes) is shared.
    """

    def __init__(self, pair: str, starting_equity: float = 100_000.0,
                 pip_size: float = None, spread_pips: float = None):
        self.pair = pair
        self.equity = starting_equity
        self.starting_equity = starting_equity
        self.peak_equity = starting_equity

        self.pip = pip_size if pip_size is not None else PIP_SIZE.get(pair, 0.0001)
        spread_p = spread_pips if spread_pips is not None else SPREAD_PIPS.get(pair, 2.0)
        self.spread = spread_p * self.pip

        self.open_positions: list[Position] = []
        self.closed_trades: list[TradeRecord] = []
        self.daily_pnl: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open_position(self, signal: dict, entry_price: float,
                      timestamp: pd.Timestamp, atr: float,
                      strategy_id: int = 0, signal_features: dict = None,
                      risk_pct: float = 0.01) -> Optional[Position]:
        """Create and register a new Position from a strategy signal.

        Returns the Position or None if sizing fails.
        """
        direction = signal["direction"]
        sl = signal["sl"]

        # Apply spread
        if direction == "LONG":
            adj_entry = entry_price + self.spread
        else:
            adj_entry = entry_price - self.spread

        # Apply slippage
        hour = timestamp.hour if hasattr(timestamp, 'hour') else 0
        slip = 0.4 * self.pip if 8 <= hour < 17 else 1.0 * self.pip
        if direction == "LONG":
            adj_entry += slip
        else:
            adj_entry -= slip

        sl_dist = abs(adj_entry - sl)
        if sl_dist <= 0:
            return None

        # Minimum RR check
        tp1_dist = abs(signal["tp1"] - adj_entry)
        if tp1_dist / sl_dist < 0.5:
            return None

        # Position sizing
        size = self._calculate_position_size(sl_dist, risk_pct)
        if size <= 0:
            return None

        tp_splits = signal.get("tp_splits", (0.40, 0.40, 0.20))
        trail_mult = signal.get("trail_atr_mult", 1.5)
        max_bars = signal.get("max_bars", 200)
        no_breakeven = signal.get("no_breakeven", False)
        confluence = signal.get("confluence", 0)

        pos = Position(
            entry_time=timestamp,
            direction=direction,
            entry_price=adj_entry,
            sl_price=sl,
            tp1_price=signal["tp1"],
            tp2_price=signal["tp2"],
            tp3_price=signal["tp3"],
            initial_size=size,
            current_size=size,
            tp_splits=tp_splits,
            trail_atr_mult=trail_mult,
            max_bars=max_bars,
            strategy_id=strategy_id,
            confluence_score=confluence,
            signal_features=signal_features or {},
            no_breakeven=no_breakeven,
        )
        self.open_positions.append(pos)
        return pos

    def update_positions(self, candle: pd.Series) -> list[TradeRecord]:
        """Check SL/TP/trailing/time exits for all open positions.

        Returns list of TradeRecords for positions that were fully closed.
        """
        to_close = []
        new_records_start = len(self.closed_trades)

        for pos in self.open_positions:
            pos.bars_held += 1
            high = candle["high"]
            low = candle["low"]
            close_price = candle["close"]
            current_atr = candle.get("atr_14", 0)

            effective_sl = pos.trailing_sl if pos.trailing_sl is not None else pos.sl_price

            if pos.direction == "LONG":
                if low <= effective_sl:
                    self._close_position(pos, effective_sl, "SL", candle)
                    to_close.append(pos)
                    continue

                if not pos.tp1_hit and high >= pos.tp1_price:
                    close_size = pos.initial_size * pos.tp_splits[0]
                    self._partial_close(pos, pos.tp1_price, close_size, "TP1", candle)
                    pos.tp1_hit = True
                    if not pos.no_breakeven:
                        pos.trailing_sl = pos.entry_price

                if not pos.tp2_hit and pos.tp1_hit and high >= pos.tp2_price:
                    close_size = pos.initial_size * pos.tp_splits[1]
                    if close_size > 0:
                        self._partial_close(pos, pos.tp2_price, close_size, "TP2", candle)
                    pos.tp2_hit = True
                    if pos.trailing_sl is None:
                        pos.trailing_sl = pos.sl_price

                if not pos.tp3_hit and pos.tp2_hit and high >= pos.tp3_price:
                    self._close_position(pos, pos.tp3_price, "TP3", candle)
                    to_close.append(pos)
                    continue

                if pos.tp2_hit and current_atr > 0:
                    new_trail = high - pos.trail_atr_mult * current_atr
                    if pos.trailing_sl is None or new_trail > pos.trailing_sl:
                        pos.trailing_sl = new_trail

            else:  # SHORT
                if high >= effective_sl:
                    self._close_position(pos, effective_sl, "SL", candle)
                    to_close.append(pos)
                    continue

                if not pos.tp1_hit and low <= pos.tp1_price:
                    close_size = pos.initial_size * pos.tp_splits[0]
                    self._partial_close(pos, pos.tp1_price, close_size, "TP1", candle)
                    pos.tp1_hit = True
                    if not pos.no_breakeven:
                        pos.trailing_sl = pos.entry_price

                if not pos.tp2_hit and pos.tp1_hit and low <= pos.tp2_price:
                    close_size = pos.initial_size * pos.tp_splits[1]
                    if close_size > 0:
                        self._partial_close(pos, pos.tp2_price, close_size, "TP2", candle)
                    pos.tp2_hit = True
                    if pos.trailing_sl is None:
                        pos.trailing_sl = pos.sl_price

                if not pos.tp3_hit and pos.tp2_hit and low <= pos.tp3_price:
                    self._close_position(pos, pos.tp3_price, "TP3", candle)
                    to_close.append(pos)
                    continue

                if pos.tp2_hit and current_atr > 0:
                    new_trail = low + pos.trail_atr_mult * current_atr
                    if pos.trailing_sl is None or new_trail < pos.trailing_sl:
                        pos.trailing_sl = new_trail

            if pos.max_bars > 0 and pos.bars_held >= pos.max_bars:
                self._close_position(pos, close_price, "TIME", candle)
                to_close.append(pos)
                continue

        for pos in to_close:
            if pos in self.open_positions:
                self.open_positions.remove(pos)

        return self.closed_trades[new_records_start:]

    def get_open_positions(self) -> list[Position]:
        return list(self.open_positions)

    def get_equity(self) -> float:
        return self.equity

    def force_close_all(self, candle: pd.Series) -> list[TradeRecord]:
        """Force-close all open positions at candle close price."""
        new_records_start = len(self.closed_trades)
        for pos in list(self.open_positions):
            self._close_position(pos, candle["close"], "END", candle)
        self.open_positions.clear()
        return self.closed_trades[new_records_start:]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _calculate_position_size(self, sl_distance: float,
                                 risk_pct: float = 0.01) -> float:
        if sl_distance <= 0:
            return 0.0
        risk_amount = self.equity * risk_pct
        return round(risk_amount / sl_distance, 2)

    def _partial_close(self, pos: Position, exit_price: float,
                       close_size: float, reason: str, candle: pd.Series):
        if pos.direction == "LONG":
            pnl_per_unit = exit_price - pos.entry_price
        else:
            pnl_per_unit = pos.entry_price - exit_price

        pnl = pnl_per_unit * close_size
        self.equity += pnl
        pos.current_size -= close_size
        pos.realized_pnl += pnl

        date_key = candle.name.date() if hasattr(candle.name, 'date') else None
        if date_key:
            self.daily_pnl[date_key] = self.daily_pnl.get(date_key, 0.0) + pnl

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    def _close_position(self, pos: Position, exit_price: float,
                        reason: str, candle: pd.Series):
        remaining = pos.current_size
        if remaining <= 0:
            remaining = 0.01

        if pos.direction == "LONG":
            pnl_per_unit = exit_price - pos.entry_price
        else:
            pnl_per_unit = pos.entry_price - exit_price

        final_pnl = pnl_per_unit * remaining
        self.equity += final_pnl

        total_pnl = pos.realized_pnl + final_pnl
        total_pnl_pips = total_pnl / (pos.initial_size * self.pip) if pos.initial_size > 0 else 0

        date_key = candle.name.date() if hasattr(candle.name, 'date') else None
        if date_key:
            self.daily_pnl[date_key] = self.daily_pnl.get(date_key, 0.0) + final_pnl

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        exit_time = candle.name
        hold_minutes = 0
        if hasattr(exit_time, 'timestamp') and hasattr(pos.entry_time, 'timestamp'):
            hold_minutes = int((exit_time - pos.entry_time).total_seconds() / 60)

        exit_detail = reason
        if reason != "TP3":
            parts = []
            if pos.tp1_hit:
                parts.append("TP1")
            if pos.tp2_hit:
                parts.append("TP2")
            parts.append(reason)
            exit_detail = "+".join(parts)

        features = pos.signal_features
        record = TradeRecord(
            timestamp=pos.entry_time,
            strategy_id=pos.strategy_id,
            pair=self.pair,
            signal_direction=pos.direction,
            entry_price=pos.entry_price,
            sl_price=pos.sl_price,
            tp1_price=pos.tp1_price,
            tp2_price=pos.tp2_price,
            tp3_price=pos.tp3_price,
            lot_size=pos.initial_size,
            confluence_score=pos.confluence_score,
            session=features.get("session", ""),
            spread_at_entry=features.get("spread_at_entry", 0),
            atr_at_entry=features.get("atr_at_entry", 0),
            adx_at_entry=features.get("adx_at_entry", 0),
            rsi_at_entry=features.get("rsi_at_entry", 0),
            ema_50_value=features.get("ema_50_value", 0),
            ema_200_value=features.get("ema_200_value", 0),
            vwap_deviation=features.get("vwap_deviation", 0),
            news_within_60min=features.get("news_within_60min", False),
            macd_hist_at_entry=features.get("macd_hist_at_entry", 0),
            stoch_k_at_entry=features.get("stoch_k_at_entry", 0),
            distance_from_ema50_pips=features.get("distance_from_ema50_pips", 0),
            candle_body_ratio=features.get("candle_body_ratio", 0),
            hour_of_day=features.get("hour_of_day", 0),
            day_of_week=features.get("day_of_week", 0),
            entry_pattern=features.get("entry_pattern", ""),
            exit_price=exit_price,
            exit_reason=exit_detail,
            exit_time=exit_time,
            pnl_pips=total_pnl_pips,
            pnl_dollars=total_pnl,
            hold_time_minutes=hold_minutes,
            win=total_pnl > 0,
        )
        self.closed_trades.append(record)
