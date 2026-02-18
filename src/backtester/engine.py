"""
Event-driven backtesting engine for fx-quant Phase 1.

Processes one candle at a time. No lookahead bias.
Supports:
  - Spread and slippage modeling
  - 3-level take-profit with partial closes
  - Trailing stop on runner position
  - Time-based exits
  - Rich trade logging (20+ features per signal)
  - News filter (30 min buffer)
  - Confidence-based position sizing (1-2% risk)
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
}

# Pip value per pair
PIP_SIZE = {
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "EUR_AUD": 0.0001,
    "GBP_AUD": 0.0001, "EUR_GBP": 0.0001, "GBP_CAD": 0.0001,
    "EUR_CAD": 0.0001, "EUR_NZD": 0.0001,
    "USD_JPY": 0.01, "GBP_JPY": 0.01,
}

# Major news events (simplified: first Friday of month = NFP, plus key dates)
# In production, use a calendar API. For backtesting 2021-2024, we hardcode
# a pattern: block trading around the first Friday of each month (NFP) and
# known high-impact recurring events.
MAJOR_NEWS_DAY_OF_WEEK = 4  # Friday
MAJOR_NEWS_WEEK = 1  # First full week of month


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
    realized_pnl: float = 0.0  # Tracks PnL from partial closes
    no_breakeven: bool = False  # When True, don't move SL to breakeven after TP1


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


class Backtester:
    """Event-driven backtesting engine."""

    def __init__(self, data: pd.DataFrame, strategy, pair: str,
                 starting_equity: float = 100_000.0,
                 htf_data: pd.DataFrame = None):
        """
        Args:
            data: Primary timeframe OHLCV with indicators pre-computed.
            strategy: Strategy object with check_signal(hist_data, current, htf_row) method.
            pair: Currency pair string e.g. 'EUR_USD'.
            starting_equity: Starting account equity.
            htf_data: Higher timeframe data with indicators (for multi-TF strategies).
        """
        self.data = data
        self.strategy = strategy
        self.pair = pair
        self.equity = starting_equity
        self.starting_equity = starting_equity
        self.peak_equity = starting_equity
        self.htf_data = htf_data

        self.pip = PIP_SIZE.get(pair, 0.0001)
        self.spread = SPREAD_PIPS.get(pair, 2.0) * self.pip

        self.open_positions: list[Position] = []
        self.closed_trades: list[TradeRecord] = []
        self.equity_curve = []
        self.daily_pnl = {}

    def _get_session(self, hour: int) -> str:
        if 0 <= hour < 8:
            return "ASIAN"
        elif 8 <= hour < 13:
            return "LONDON"
        elif 13 <= hour < 17:
            return "OVERLAP"
        else:
            return "NY"

    def _get_slippage(self, hour: int) -> float:
        """Slippage in price units. High-volume sessions get less slippage."""
        if 8 <= hour < 17:  # London + NY overlap
            return 0.4 * self.pip
        else:
            return 1.0 * self.pip

    def _is_near_news(self, timestamp: pd.Timestamp) -> bool:
        """Simple news filter: first Friday of each month (NFP proxy) +/- 30 min."""
        # Check if current day is first Friday of month
        if timestamp.weekday() != 4:  # Not Friday
            return False
        if timestamp.day > 7:  # Not first week
            return False
        # NFP typically at 13:30 UTC
        if 13 <= timestamp.hour <= 14:
            return True
        return False

    def _calculate_position_size(self, sl_distance: float,
                                 confluence_score: int,
                                 risk_pct: float = 0.01) -> float:
        """Risk-based position sizing (default 1%, strategies can override)."""
        if sl_distance <= 0:
            return 0.0

        risk_amount = self.equity * risk_pct
        # Position size = risk_amount / SL distance in price
        position_size = risk_amount / sl_distance
        return round(position_size, 2)

    def _get_htf_row(self, timestamp: pd.Timestamp) -> Optional[pd.Series]:
        """Get the most recent FULLY CLOSED higher-timeframe candle."""
        if self.htf_data is None:
            return None
        # Only use HTF candles that closed BEFORE current timestamp
        valid = self.htf_data[self.htf_data.index < timestamp]
        if len(valid) == 0:
            return None
        return valid.iloc[-1]

    def _apply_spread_to_entry(self, price: float, direction: str) -> float:
        """Apply spread to entry price."""
        if direction == "LONG":
            return price + self.spread  # Buy at ask
        else:
            return price - self.spread  # Sell at bid (lower)

    def _apply_slippage_to_entry(self, price: float, direction: str,
                                 hour: int) -> float:
        """Apply slippage to entry price."""
        slip = self._get_slippage(hour)
        if direction == "LONG":
            return price + slip  # Slippage works against us
        else:
            return price - slip

    def _check_daily_drawdown(self, timestamp: pd.Timestamp) -> bool:
        """Check if 5% daily drawdown has been breached."""
        date_key = timestamp.date()
        if date_key not in self.daily_pnl:
            self.daily_pnl[date_key] = 0.0
        return self.daily_pnl[date_key] <= -0.05 * self.starting_equity

    def _update_positions(self, candle: pd.Series, i: int):
        """Check SL/TP/trailing/time exits for all open positions."""
        to_close = []

        for pos in self.open_positions:
            pos.bars_held += 1
            high = candle["high"]
            low = candle["low"]
            close_price = candle["close"]
            current_atr = candle.get("atr_14", 0)

            # Determine effective SL
            effective_sl = pos.trailing_sl if pos.trailing_sl is not None else pos.sl_price

            if pos.direction == "LONG":
                # Check SL
                if low <= effective_sl:
                    self._close_position(pos, effective_sl, "SL", candle)
                    to_close.append(pos)
                    continue

                # Check TP1
                if not pos.tp1_hit and high >= pos.tp1_price:
                    close_size = pos.initial_size * pos.tp_splits[0]
                    self._partial_close(pos, pos.tp1_price, close_size, "TP1", candle)
                    pos.tp1_hit = True
                    if not pos.no_breakeven:
                        # Move SL to breakeven after TP1
                        pos.trailing_sl = pos.entry_price

                # Check TP2
                if not pos.tp2_hit and pos.tp1_hit and high >= pos.tp2_price:
                    close_size = pos.initial_size * pos.tp_splits[1]
                    if close_size > 0:
                        self._partial_close(pos, pos.tp2_price, close_size, "TP2", candle)
                    pos.tp2_hit = True
                    # If no breakeven was set, start trailing from original SL
                    if pos.trailing_sl is None:
                        pos.trailing_sl = pos.sl_price

                # Check TP3
                if not pos.tp3_hit and pos.tp2_hit and high >= pos.tp3_price:
                    self._close_position(pos, pos.tp3_price, "TP3", candle)
                    to_close.append(pos)
                    continue

                # Trailing stop for runner (after TP2)
                if pos.tp2_hit and current_atr > 0:
                    new_trail = high - pos.trail_atr_mult * current_atr
                    if pos.trailing_sl is None or new_trail > pos.trailing_sl:
                        pos.trailing_sl = new_trail

            else:  # SHORT
                # Check SL
                if high >= effective_sl:
                    self._close_position(pos, effective_sl, "SL", candle)
                    to_close.append(pos)
                    continue

                # Check TP1
                if not pos.tp1_hit and low <= pos.tp1_price:
                    close_size = pos.initial_size * pos.tp_splits[0]
                    self._partial_close(pos, pos.tp1_price, close_size, "TP1", candle)
                    pos.tp1_hit = True
                    if not pos.no_breakeven:
                        pos.trailing_sl = pos.entry_price

                # Check TP2
                if not pos.tp2_hit and pos.tp1_hit and low <= pos.tp2_price:
                    close_size = pos.initial_size * pos.tp_splits[1]
                    if close_size > 0:
                        self._partial_close(pos, pos.tp2_price, close_size, "TP2", candle)
                    pos.tp2_hit = True
                    if pos.trailing_sl is None:
                        pos.trailing_sl = pos.sl_price

                # Check TP3
                if not pos.tp3_hit and pos.tp2_hit and low <= pos.tp3_price:
                    self._close_position(pos, pos.tp3_price, "TP3", candle)
                    to_close.append(pos)
                    continue

                # Trailing stop for runner
                if pos.tp2_hit and current_atr > 0:
                    new_trail = low + pos.trail_atr_mult * current_atr
                    if pos.trailing_sl is None or new_trail < pos.trailing_sl:
                        pos.trailing_sl = new_trail

            # Time-based exit
            if pos.max_bars > 0 and pos.bars_held >= pos.max_bars:
                self._close_position(pos, close_price, "TIME", candle)
                to_close.append(pos)
                continue

        for pos in to_close:
            if pos in self.open_positions:
                self.open_positions.remove(pos)

    def _partial_close(self, pos: Position, exit_price: float,
                       close_size: float, reason: str, candle: pd.Series):
        """Close a partial portion of a position."""
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
        """Fully close remaining position and log the trade."""
        remaining = pos.current_size
        if remaining <= 0:
            remaining = 0.01  # avoid zero

        if pos.direction == "LONG":
            pnl_per_unit = exit_price - pos.entry_price
        else:
            pnl_per_unit = pos.entry_price - exit_price

        final_pnl = pnl_per_unit * remaining
        self.equity += final_pnl

        # Total PnL = partial closes + final close
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

        # Build detailed exit reason (e.g. "TP1+TP2+SL" instead of just "SL")
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

    def _build_signal_features(self, candle: pd.Series, i: int) -> dict:
        """Extract features from current candle for trade logging."""
        hour = candle.name.hour if hasattr(candle.name, 'hour') else 0
        body = abs(candle["close"] - candle["open"])
        full_range = candle["high"] - candle["low"]
        body_ratio = body / full_range if full_range > 0 else 0

        ema_50 = candle.get("ema_50", 0)
        dist_ema50 = (candle["close"] - ema_50) / self.pip if ema_50 else 0
        vwap = candle.get("session_vwap", 0)
        vwap_dev = (candle["close"] - vwap) / self.pip if vwap else 0

        return {
            "session": self._get_session(hour),
            "spread_at_entry": self.spread / self.pip,
            "atr_at_entry": candle.get("atr_14", 0),
            "adx_at_entry": candle.get("adx_14", 0),
            "rsi_at_entry": candle.get("rsi_14", 0),
            "ema_50_value": ema_50,
            "ema_200_value": candle.get("ema_200", 0),
            "vwap_deviation": vwap_dev,
            "news_within_60min": self._is_near_news(candle.name),
            "macd_hist_at_entry": candle.get("macd_hist", 0),
            "stoch_k_at_entry": candle.get("stoch_k", 0),
            "distance_from_ema50_pips": dist_ema50,
            "candle_body_ratio": body_ratio,
            "hour_of_day": hour,
            "day_of_week": candle.name.weekday() if hasattr(candle.name, 'weekday') else 0,
        }

    def run(self) -> dict:
        """Run the backtest. Returns performance report dict."""
        # Give the strategy access to full HTF data (strategy filters by timestamp)
        self.strategy.htf_data = self.htf_data

        # Need at least 200 bars for indicators to warm up
        warmup = 200

        for i in range(warmup, len(self.data)):
            candle = self.data.iloc[i]
            timestamp = self.data.index[i]

            # Update open positions first (SL/TP/trail/time checks)
            self._update_positions(candle, i)

            # Record equity
            self.equity_curve.append({
                "timestamp": timestamp,
                "equity": self.equity,
            })

            # Check daily drawdown halt
            if self._check_daily_drawdown(timestamp):
                continue

            # Skip if near news
            if self._is_near_news(timestamp):
                continue

            # Skip if already have an open position (1 at a time per strategy)
            if self.open_positions:
                continue

            # Get HTF context (only fully closed candles)
            htf_row = self._get_htf_row(timestamp)

            # Pass full data + current index. Strategy must only access [:i+1].
            signal = self.strategy.check_signal(self.data, i, candle, htf_row)

            if signal is None:
                continue

            direction = signal["direction"]
            sl = signal["sl"]
            tp1 = signal["tp1"]
            tp2 = signal["tp2"]
            tp3 = signal["tp3"]
            confluence = signal.get("confluence", 0)
            tp_splits = signal.get("tp_splits", (0.40, 0.40, 0.20))
            trail_mult = signal.get("trail_atr_mult", 1.5)
            max_bars = signal.get("max_bars", 200)
            no_breakeven = signal.get("no_breakeven", False)
            risk_pct = signal.get("risk_pct", 0.01)

            # Minimum RR check (TP1 vs SL distance)
            entry = candle["close"]
            sl_dist = abs(entry - sl)
            tp1_dist = abs(tp1 - entry)
            if sl_dist == 0 or tp1_dist / sl_dist < 0.5:
                continue

            # Apply spread and slippage to entry
            hour = timestamp.hour if hasattr(timestamp, 'hour') else 0
            adj_entry = self._apply_spread_to_entry(entry, direction)
            adj_entry = self._apply_slippage_to_entry(adj_entry, direction, hour)

            # Recalculate SL distance after adjustment
            sl_dist_adj = abs(adj_entry - sl)
            if sl_dist_adj <= 0:
                continue

            # Position sizing
            size = self._calculate_position_size(sl_dist_adj, confluence, risk_pct)
            if size <= 0:
                continue

            # Build features for logging
            features = self._build_signal_features(candle, i)
            features["entry_pattern"] = signal.get("entry_pattern", "")

            # Open position
            pos = Position(
                entry_time=timestamp,
                direction=direction,
                entry_price=adj_entry,
                sl_price=sl,
                tp1_price=tp1,
                tp2_price=tp2,
                tp3_price=tp3,
                initial_size=size,
                current_size=size,
                tp_splits=tp_splits,
                trail_atr_mult=trail_mult,
                max_bars=max_bars,
                strategy_id=self.strategy.strategy_id,
                confluence_score=confluence,
                signal_features=features,
                no_breakeven=no_breakeven,
            )
            self.open_positions.append(pos)

        # Force-close any remaining positions at last candle
        if self.open_positions:
            last_candle = self.data.iloc[-1]
            for pos in list(self.open_positions):
                self._close_position(pos, last_candle["close"], "END", last_candle)
            self.open_positions.clear()

        return self.generate_report()

    def generate_report(self) -> dict:
        """Generate comprehensive performance metrics."""
        trades = self.closed_trades
        if not trades:
            return {
                "pair": self.pair,
                "strategy_id": self.strategy.strategy_id,
                "strategy_name": self.strategy.name,
                "total_trades": 0,
                "message": "No trades generated",
            }

        wins = [t for t in trades if t.win]
        losses = [t for t in trades if not t.win]
        pnls = [t.pnl_dollars for t in trades]
        pnl_pips = [t.pnl_pips for t in trades]

        total_trades = len(trades)
        win_rate = len(wins) / total_trades * 100 if total_trades else 0
        avg_win_pips = np.mean([t.pnl_pips for t in wins]) if wins else 0
        avg_loss_pips = np.mean([abs(t.pnl_pips) for t in losses]) if losses else 0
        avg_rr = avg_win_pips / avg_loss_pips if avg_loss_pips > 0 else 0

        gross_profit = sum(t.pnl_dollars for t in wins)
        gross_loss = abs(sum(t.pnl_dollars for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        expectancy = (win_rate / 100 * avg_win_pips) - ((1 - win_rate / 100) * avg_loss_pips)

        # Sharpe ratio (annualized)
        if len(pnls) > 1:
            returns = pd.Series(pnls)
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe = 0

        # Max drawdown from equity curve
        eq = pd.Series([e["equity"] for e in self.equity_curve])
        if len(eq) > 0:
            peak = eq.cummax()
            dd = (eq - peak) / peak
            max_dd = dd.min() * 100  # negative percentage
        else:
            max_dd = 0

        # Consecutive wins/losses
        results = [t.win for t in trades]
        max_consec_wins = max_consec_losses = current_streak = 0
        current_type = None
        for r in results:
            if r == current_type:
                current_streak += 1
            else:
                current_type = r
                current_streak = 1
            if r and current_streak > max_consec_wins:
                max_consec_wins = current_streak
            if not r and current_streak > max_consec_losses:
                max_consec_losses = current_streak

        # Average hold time
        avg_hold = np.mean([t.hold_time_minutes for t in trades])

        # Best/worst trade
        best_trade_pips = max(pnl_pips)
        worst_trade_pips = min(pnl_pips)
        best_trade_dollars = max(pnls)
        worst_trade_dollars = min(pnls)

        # Session breakdown
        session_stats = {}
        for session in ["ASIAN", "LONDON", "OVERLAP", "NY"]:
            session_trades = [t for t in trades if t.session == session]
            if session_trades:
                s_wins = [t for t in session_trades if t.win]
                session_stats[session] = {
                    "trades": len(session_trades),
                    "win_rate": len(s_wins) / len(session_trades) * 100,
                    "total_pnl_pips": sum(t.pnl_pips for t in session_trades),
                }

        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        return {
            "pair": self.pair,
            "strategy_id": self.strategy.strategy_id,
            "strategy_name": self.strategy.name,
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 2),
            "avg_rr": round(avg_rr, 2),
            "expectancy_pips": round(expectancy, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "profit_factor": round(profit_factor, 2),
            "total_pnl_pips": round(sum(pnl_pips), 2),
            "total_pnl_dollars": round(sum(pnls), 2),
            "avg_win_pips": round(avg_win_pips, 2),
            "avg_loss_pips": round(avg_loss_pips, 2),
            "max_consecutive_wins": max_consec_wins,
            "max_consecutive_losses": max_consec_losses,
            "avg_hold_time_minutes": round(avg_hold, 1),
            "best_trade_pips": round(best_trade_pips, 2),
            "worst_trade_pips": round(worst_trade_pips, 2),
            "best_trade_dollars": round(best_trade_dollars, 2),
            "worst_trade_dollars": round(worst_trade_dollars, 2),
            "final_equity": round(self.equity, 2),
            "starting_equity": self.starting_equity,
            "session_breakdown": session_stats,
            "exit_reasons": exit_reasons,
        }

    def get_trade_log_df(self) -> pd.DataFrame:
        """Return closed trades as a DataFrame for CSV export."""
        if not self.closed_trades:
            return pd.DataFrame()
        records = []
        for t in self.closed_trades:
            records.append(vars(t))
        return pd.DataFrame(records)
