"""
Entry point for the Phase 2 live paper trading engine.

Usage:
    python src/live/run.py          # continuous loop
    python src/live/run.py --once   # single iteration (for testing)
"""
import os
import sys
import yaml

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.position_manager import PositionManager
from src.live.data_feed import OandaDataFeed
from src.live.executor import OandaExecutor
from src.live.engine import LiveEngine, StrategySlot

# Strategy imports
from src.strategies_pkg.s7_liquidity_sweep import S7_Liquidity_Sweep
from src.strategies_pkg.s9_london_session import S9_London_Session
from src.strategies_pkg.s4f_ema_ribbon import S4F_EMA_Ribbon
from src.strategies_pkg.s3_key_level_breakout import S3_KeyLevel_Breakout


def load_config() -> dict:
    """Load Phase 2 config from system.yaml."""
    cfg_path = os.path.join(ROOT, "config", "system.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("phase2", {})


def build_slots(config: dict) -> list[StrategySlot]:
    """Build StrategySlot instances from config."""
    starting_equity = config.get("starting_equity", 100_000)
    slots = []

    # Strategy factory
    strategy_map = config.get("strategies", [])
    for entry in strategy_map:
        if not entry.get("enabled", True):
            continue

        name = entry["name"]
        pair = entry["pair"]
        tf = entry.get("timeframe", "H1")
        htf_tf = entry.get("htf_timeframe", None)

        # Create strategy instance
        if name == "S7_Tight":
            strategy = S7_Liquidity_Sweep()
        elif name == "S9":
            strategy = S9_London_Session()
        elif name == "S9_Filtered":
            strategy = S9_London_Session(pair=pair, filtered=True)
        elif name == "S4F":
            strategy = S4F_EMA_Ribbon()
        elif name == "S3":
            strategy = S3_KeyLevel_Breakout()
        else:
            print(f"Unknown strategy: {name}, skipping.")
            continue

        # Data feeds
        data_feed = OandaDataFeed(pair=pair, granularity=tf, candle_count=250)
        htf_feed = None
        if htf_tf and htf_tf != tf:
            htf_feed = OandaDataFeed(pair=pair, granularity=htf_tf, candle_count=250)

        # Position manager (each slot gets its own)
        pm = PositionManager(pair=pair, starting_equity=starting_equity)

        slot = StrategySlot(
            name=name,
            strategy=strategy,
            pair=pair,
            timeframe=tf,
            data_feed=data_feed,
            position_manager=pm,
            htf_feed=htf_feed,
            enabled=True,
        )
        slots.append(slot)

    return slots


def main():
    from src.live.health import start_health_server
    start_health_server()

    config = load_config()
    if not config:
        print("ERROR: No phase2 config found in config/system.yaml")
        sys.exit(1)

    paper_mode = config.get("paper_mode", True)
    interval = config.get("poll_interval_seconds", 60)
    max_dd = config.get("max_daily_drawdown_pct", 5.0)

    # Check OANDA credentials
    env_path = os.path.join(ROOT, "config", ".env")
    if os.path.exists(env_path):
        for line in open(env_path).read().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    account_id = os.getenv("OANDA_ACCOUNT_ID", "")
    if not account_id and not paper_mode:
        print("ERROR: OANDA_ACCOUNT_ID not set in config/.env")
        print("Paper trading can run without it, but live mode requires it.")
        sys.exit(1)

    if not account_id:
        print("WARNING: OANDA_ACCOUNT_ID empty — paper mode only, "
              "data feeds will still work for signal testing.")

    # Build components
    executor = OandaExecutor(paper_mode=paper_mode)
    slots = build_slots(config)

    if not slots:
        print("ERROR: No strategy slots configured.")
        sys.exit(1)

    print(f"Phase 2 Live Engine")
    print(f"  Mode: {'PAPER' if paper_mode else 'LIVE'}")
    print(f"  Slots: {len(slots)}")
    print(f"  Poll interval: {interval}s")
    print(f"  Max daily DD: {max_dd}%")
    print()

    engine = LiveEngine(
        slots=slots,
        executor=executor,
        interval_seconds=interval,
        max_daily_drawdown_pct=max_dd,
    )
    engine.load_state()

    if "--once" in sys.argv:
        engine.run_once()
    else:
        engine.run_loop()


if __name__ == "__main__":
    main()
