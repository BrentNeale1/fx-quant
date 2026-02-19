"""
Live trading engine orchestrator.

Ties together data feeds, strategies, position managers, and the executor
into a single run loop that polls OANDA and manages positions.
"""
import json
import time
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.position_manager import PositionManager, PIP_SIZE, SPREAD_PIPS
from src.live.data_feed import OandaDataFeed
from src.live.executor import OandaExecutor

ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)


@dataclass
class StrategySlot:
    """One strategy-pair combination running in the live engine."""
    name: str
    strategy: object  # BaseStrategy instance
    pair: str
    timeframe: str
    data_feed: OandaDataFeed
    position_manager: PositionManager
    htf_feed: Optional[OandaDataFeed] = None  # for multi-TF strategies
    enabled: bool = True


class LiveEngine:
    """Main orchestrator for live paper/live trading.

    Each cycle:
    1. Check kill switch
    2. Fetch latest candles for each slot
    3. Compute indicators
    4. Check for new signals (only if no open position)
    5. Update open positions (SL/TP/trail/time)
    6. Log state
    7. Print status summary
    """

    def __init__(self, slots: list[StrategySlot], executor: OandaExecutor,
                 interval_seconds: int = 60, max_daily_drawdown_pct: float = 5.0):
        self.slots = slots
        self.executor = executor
        self.interval_seconds = interval_seconds
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self._state_path = LOGS_DIR / "live_state.json"
        self._trades_path = LOGS_DIR / "live_trades.csv"
        self._cycle_count = 0

    def run_once(self):
        """Single iteration: fetch, check signals, manage positions."""
        self._cycle_count += 1
        now = datetime.now(timezone.utc)
        print(f"\n{'='*70}")
        print(f"Cycle #{self._cycle_count}  |  {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"{'='*70}")

        # 1. Kill switch
        if self.executor.check_kill_switch():
            print("KILL SWITCH ACTIVE — skipping all trading.")
            return

        for slot in self.slots:
            if not slot.enabled:
                continue

            slot_label = f"[{slot.name}/{slot.pair}]"

            try:
                # 2. Fetch latest candles
                df = slot.data_feed.fetch_latest()
                if df.empty:
                    print(f"  {slot_label} No data fetched, skipping.")
                    continue

                htf_df = None
                if slot.htf_feed is not None:
                    htf_df = slot.htf_feed.fetch_latest()

                # 3. Current candle + HTF row
                current = df.iloc[-1]
                idx = len(df) - 1
                timestamp = df.index[-1]

                htf_row = None
                if htf_df is not None and not htf_df.empty:
                    valid = htf_df[htf_df.index < timestamp]
                    if not valid.empty:
                        htf_row = valid.iloc[-1]
                elif slot.timeframe == "H1":
                    # H1 self-reference for HTF
                    valid = df[df.index < timestamp]
                    if not valid.empty:
                        htf_row = valid.iloc[-1]

                # 4. Update existing positions
                pm = slot.position_manager
                closed = pm.update_positions(current)
                for trade in closed:
                    self._log_closed_trade(trade, slot.name)
                    print(f"  {slot_label} CLOSED: {trade.signal_direction} "
                          f"exit={trade.exit_reason} pnl={trade.pnl_pips:+.1f}p")

                # 5. Check for new signal (only if no open position)
                if not pm.get_open_positions():
                    # Daily drawdown check
                    today = now.date()
                    daily_pnl = pm.daily_pnl.get(today, 0.0)
                    dd_limit = pm.starting_equity * self.max_daily_drawdown_pct / 100
                    if daily_pnl <= -dd_limit:
                        print(f"  {slot_label} Daily DD limit hit ({daily_pnl:.0f}), skipping signals.")
                        continue

                    # Give strategy access to HTF data
                    if htf_df is not None:
                        slot.strategy.htf_data = htf_df

                    signal = slot.strategy.check_signal(df, idx, current, htf_row)

                    if signal is not None:
                        atr = current.get("atr_14", 0)
                        pos = pm.open_position(
                            signal=signal,
                            entry_price=current["close"],
                            timestamp=timestamp,
                            atr=atr,
                            strategy_id=slot.strategy.strategy_id,
                            signal_features={"entry_pattern": signal.get("entry_pattern", "")},
                        )
                        if pos is not None:
                            print(f"  {slot_label} NEW {pos.direction} @ {pos.entry_price:.5f} "
                                  f"SL={pos.sl_price:.5f} TP1={pos.tp1_price:.5f}")

                            # Log to executor
                            self.executor.place_market_order(
                                pair=slot.pair,
                                units=int(pos.initial_size),
                                direction=pos.direction,
                                sl=pos.sl_price,
                                tp=pos.tp1_price,
                            )
                    else:
                        print(f"  {slot_label} No signal.")
                else:
                    open_pos = pm.get_open_positions()[0]
                    print(f"  {slot_label} Open: {open_pos.direction} "
                          f"bars={open_pos.bars_held} "
                          f"tp1={'Y' if open_pos.tp1_hit else 'N'} "
                          f"tp2={'Y' if open_pos.tp2_hit else 'N'}")

            except Exception as e:
                print(f"  {slot_label} ERROR: {e}")

        # 6. Save state
        self.save_state()

        # 7. Summary
        self._print_summary()

    def run_loop(self):
        """Continuous loop with interval sleep."""
        print(f"Starting live engine loop (interval={self.interval_seconds}s)")
        print(f"Slots: {len(self.slots)}")
        for s in self.slots:
            print(f"  - {s.name} / {s.pair} / {s.timeframe}")
        print()

        while True:
            try:
                self.run_once()
                print(f"\nSleeping {self.interval_seconds}s...")
                time.sleep(self.interval_seconds)
            except KeyboardInterrupt:
                print("\nShutting down gracefully.")
                break

    def save_state(self):
        """Persist current state to JSON for crash recovery."""
        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle": self._cycle_count,
            "slots": [],
        }
        for slot in self.slots:
            pm = slot.position_manager
            slot_state = {
                "name": slot.name,
                "pair": slot.pair,
                "equity": pm.get_equity(),
                "open_positions": len(pm.get_open_positions()),
                "closed_trades": len(pm.closed_trades),
                "enabled": slot.enabled,
            }
            if pm.get_open_positions():
                pos = pm.get_open_positions()[0]
                slot_state["position"] = {
                    "direction": pos.direction,
                    "entry_price": pos.entry_price,
                    "sl": pos.sl_price,
                    "bars_held": pos.bars_held,
                    "tp1_hit": pos.tp1_hit,
                    "tp2_hit": pos.tp2_hit,
                }
            state["slots"].append(slot_state)

        with open(self._state_path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        """Restore state after restart (basic — restores equity only)."""
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path) as f:
                state = json.load(f)
            for slot_state in state.get("slots", []):
                for slot in self.slots:
                    if slot.name == slot_state["name"] and slot.pair == slot_state["pair"]:
                        slot.position_manager.equity = slot_state.get("equity",
                                                                       slot.position_manager.equity)
            print(f"Loaded state from {self._state_path}")
        except Exception as e:
            print(f"Warning: Could not load state: {e}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_closed_trade(self, trade, slot_name: str):
        """Append a closed trade record to live_trades.csv."""
        columns = [
            "slot", "timestamp", "pair", "direction", "entry_price",
            "exit_price", "exit_reason", "pnl_pips", "pnl_dollars",
            "hold_minutes",
        ]
        row = {
            "slot": slot_name,
            "timestamp": str(trade.timestamp),
            "pair": trade.pair,
            "direction": trade.signal_direction,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "exit_reason": trade.exit_reason,
            "pnl_pips": round(trade.pnl_pips, 2),
            "pnl_dollars": round(trade.pnl_dollars, 2),
            "hold_minutes": trade.hold_time_minutes,
        }
        import csv
        file_exists = self._trades_path.exists()
        with open(self._trades_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _print_summary(self):
        """Print a compact status summary."""
        print(f"\n{'─'*50}")
        print(f"{'Slot':<18} {'Equity':>10} {'Open':>5} {'Trades':>7}")
        print(f"{'─'*50}")
        for slot in self.slots:
            if not slot.enabled:
                continue
            pm = slot.position_manager
            label = f"{slot.name}/{slot.pair}"
            print(f"{label:<18} {pm.get_equity():>10,.0f} "
                  f"{len(pm.get_open_positions()):>5} "
                  f"{len(pm.closed_trades):>7}")
        print(f"{'─'*50}")
