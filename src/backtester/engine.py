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
from typing import Optional

from src.position_manager import (
    Position, TradeRecord, PositionManager,
    SPREAD_PIPS, PIP_SIZE,
)

# Major news events (simplified: first Friday of month = NFP, plus key dates)
# In production, use a calendar API. For backtesting 2021-2024, we hardcode
# a pattern: block trading around the first Friday of each month (NFP) and
# known high-impact recurring events.
MAJOR_NEWS_DAY_OF_WEEK = 4  # Friday
MAJOR_NEWS_WEEK = 1  # First full week of month


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
        self.starting_equity = starting_equity
        self.htf_data = htf_data

        # Delegate position management to PositionManager
        self._pm = PositionManager(pair, starting_equity)

        self.equity_curve = []

    # Proxy properties so existing code that reads these still works
    @property
    def equity(self):
        return self._pm.equity

    @equity.setter
    def equity(self, value):
        self._pm.equity = value

    @property
    def peak_equity(self):
        return self._pm.peak_equity

    @peak_equity.setter
    def peak_equity(self, value):
        self._pm.peak_equity = value

    @property
    def open_positions(self):
        return self._pm.open_positions

    @property
    def closed_trades(self):
        return self._pm.closed_trades

    @property
    def daily_pnl(self):
        return self._pm.daily_pnl

    @property
    def pip(self):
        return self._pm.pip

    @property
    def spread(self):
        return self._pm.spread

    def _get_session(self, hour: int) -> str:
        if 0 <= hour < 8:
            return "ASIAN"
        elif 8 <= hour < 13:
            return "LONDON"
        elif 13 <= hour < 17:
            return "OVERLAP"
        else:
            return "NY"

    def _is_near_news(self, timestamp: pd.Timestamp) -> bool:
        """Simple news filter: first Friday of each month (NFP proxy) +/- 30 min."""
        if timestamp.weekday() != 4:
            return False
        if timestamp.day > 7:
            return False
        if 13 <= timestamp.hour <= 14:
            return True
        return False

    def _check_daily_drawdown(self, timestamp: pd.Timestamp) -> bool:
        """Check if 5% daily drawdown has been breached."""
        date_key = timestamp.date()
        if date_key not in self.daily_pnl:
            self.daily_pnl[date_key] = 0.0
        return self.daily_pnl[date_key] <= -0.05 * self.starting_equity

    def _get_htf_row(self, timestamp: pd.Timestamp) -> Optional[pd.Series]:
        """Get the most recent FULLY CLOSED higher-timeframe candle."""
        if self.htf_data is None:
            return None
        valid = self.htf_data[self.htf_data.index < timestamp]
        if len(valid) == 0:
            return None
        return valid.iloc[-1]

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
        self.strategy.htf_data = self.htf_data
        warmup = 200

        for i in range(warmup, len(self.data)):
            candle = self.data.iloc[i]
            timestamp = self.data.index[i]

            # Update open positions (SL/TP/trail/time checks)
            self._pm.update_positions(candle)

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
            if self._pm.open_positions:
                continue

            # Get HTF context
            htf_row = self._get_htf_row(timestamp)

            # Check for signal
            signal = self.strategy.check_signal(self.data, i, candle, htf_row)
            if signal is None:
                continue

            # Build features for logging
            features = self._build_signal_features(candle, i)
            features["entry_pattern"] = signal.get("entry_pattern", "")

            risk_pct = signal.get("risk_pct", 0.01)

            # Open position via PositionManager
            self._pm.open_position(
                signal=signal,
                entry_price=candle["close"],
                timestamp=timestamp,
                atr=candle.get("atr_14", 0),
                strategy_id=self.strategy.strategy_id,
                signal_features=features,
                risk_pct=risk_pct,
            )

        # Force-close any remaining positions at last candle
        if self._pm.open_positions:
            last_candle = self.data.iloc[-1]
            self._pm.force_close_all(last_candle)

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
            max_dd = dd.min() * 100
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

        avg_hold = np.mean([t.hold_time_minutes for t in trades])

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
