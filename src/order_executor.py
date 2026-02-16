# src/order_executor.py
"""
Paper-first order execution engine for fx-quant.
Supports paper mode (simulated fills) and live mode (OANDA v20 market orders).
Includes kill switch, position limits, and full order logging.
"""

import os
import sys
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from supabase import create_client

from config_loader import load_config, get_project_root
from backtester import fetch_candles_from_supabase, generate_signals
from ai_wrapper import train_ensemble, validate_signal, log_ai_decision


# ---------------------------------------------------------------------------
# OANDA helpers
# ---------------------------------------------------------------------------

def get_oanda_base_url():
    """Return the OANDA API base URL based on OANDA_ENV."""
    env = os.getenv("OANDA_ENV", "practice").strip()
    if env == "practice":
        return "https://api-fxpractice.oanda.com"
    return "https://api-fxtrade.oanda.com"


def _oanda_headers():
    """Return auth headers for OANDA API calls."""
    api_key = os.getenv("OANDA_API_KEY")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

def check_kill_switch():
    """
    Check for STOP_ALL_TRADING file in project root.
    Returns True if trading should be halted.
    """
    kill_file = get_project_root() / "STOP_ALL_TRADING"
    return kill_file.exists()


# ---------------------------------------------------------------------------
# Account & trade queries
# ---------------------------------------------------------------------------

def get_account_summary():
    """
    GET /v3/accounts/{ACCOUNT_ID}/summary
    Returns dict with balance, NAV, and open position count.
    """
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    base = get_oanda_base_url()
    url = f"{base}/v3/accounts/{account_id}/summary"

    r = requests.get(url, headers=_oanda_headers())
    r.raise_for_status()
    acct = r.json()["account"]
    return {
        "balance": float(acct["balance"]),
        "NAV": float(acct["NAV"]),
        "open_position_count": int(acct["openPositionCount"]),
        "open_trade_count": int(acct["openTradeCount"]),
    }


def get_open_trades():
    """
    GET /v3/accounts/{ACCOUNT_ID}/openTrades
    Returns list of open trade dicts.
    """
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    base = get_oanda_base_url()
    url = f"{base}/v3/accounts/{account_id}/openTrades"

    r = requests.get(url, headers=_oanda_headers())
    r.raise_for_status()
    return r.json().get("trades", [])


def close_all_trades():
    """
    Close every open trade. Used when kill switch is activated.
    """
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    base = get_oanda_base_url()
    trades = get_open_trades()

    if not trades:
        print("No open trades to close.")
        return

    for trade in trades:
        trade_id = trade["id"]
        url = f"{base}/v3/accounts/{account_id}/trades/{trade_id}/close"
        r = requests.put(url, headers=_oanda_headers())
        if r.status_code == 200:
            print(f"  Closed trade {trade_id} ({trade['instrument']})")
        else:
            print(f"  Failed to close trade {trade_id}: {r.status_code} {r.text}")


# ---------------------------------------------------------------------------
# Unit sizing
# ---------------------------------------------------------------------------

def compute_units(balance, instrument, side, cfg):
    """
    Calculate order unit count from account balance and config.

    In paper mode uses trade_size_pct_of_equity.
    In live mode uses canary_size_pct (conservative).
    Returns positive int for BUY, negative int for SELL.
    """
    paper_mode = cfg.get("execution", {}).get("paper_mode", True)

    if paper_mode:
        pct = cfg["strategy"]["trade_size_pct_of_equity"]
    else:
        pct = cfg["execution"]["canary_size_pct"]

    units = int(balance * pct)
    if units < 1:
        units = 1

    if side == "SELL":
        units = -units

    return units


# ---------------------------------------------------------------------------
# Order logging
# ---------------------------------------------------------------------------

def log_order(order_details):
    """
    Append order to logs/order_log.csv.
    Creates the file with headers if it doesn't exist.
    """
    root = get_project_root()
    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)
    csv_path = logs_dir / "order_log.csv"

    columns = [
        "timestamp", "instrument", "side", "units", "price",
        "order_type", "mode", "status", "response",
    ]

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(order_details)


# ---------------------------------------------------------------------------
# Place order
# ---------------------------------------------------------------------------

def place_order(instrument, units, side, cfg, price=None):
    """
    Place an order (paper or live).

    Paper mode: simulate the order, log it, do NOT call OANDA API.
    Live mode: POST market order to OANDA, log the response.

    Returns order response dict.
    """
    paper_mode = cfg.get("execution", {}).get("paper_mode", True)
    max_positions = cfg.get("execution", {}).get("max_positions", 3)
    mode = "paper" if paper_mode else "live"
    now = datetime.now(timezone.utc).isoformat()

    # Kill switch check before every order
    if check_kill_switch():
        print(f"  KILL SWITCH active — order blocked: {side} {units} {instrument}")
        log_order({
            "timestamp": now,
            "instrument": instrument,
            "side": side,
            "units": units,
            "price": price or "",
            "order_type": "MARKET",
            "mode": mode,
            "status": "BLOCKED_KILL_SWITCH",
            "response": "Kill switch file present",
        })
        return {"status": "BLOCKED_KILL_SWITCH"}

    # Max positions check (live mode only — query open trades)
    if not paper_mode:
        try:
            open_trades = get_open_trades()
            if len(open_trades) >= max_positions and side == "BUY":
                print(f"  Max positions ({max_positions}) reached — order blocked.")
                log_order({
                    "timestamp": now,
                    "instrument": instrument,
                    "side": side,
                    "units": units,
                    "price": price or "",
                    "order_type": "MARKET",
                    "mode": mode,
                    "status": "BLOCKED_MAX_POS",
                    "response": f"Open trades: {len(open_trades)}",
                })
                return {"status": "BLOCKED_MAX_POS"}
        except Exception as e:
            print(f"  Error checking open trades: {e}")

    # --- Paper mode ---
    if paper_mode:
        print(f"  [PAPER] {side} {abs(units)} {instrument} @ {price or 'market'}")
        log_order({
            "timestamp": now,
            "instrument": instrument,
            "side": side,
            "units": units,
            "price": price or "",
            "order_type": "MARKET",
            "mode": "paper",
            "status": "FILLED",
            "response": "Paper fill simulated",
        })
        return {"status": "FILLED", "mode": "paper", "units": units, "price": price}

    # --- Live mode ---
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    base = get_oanda_base_url()
    url = f"{base}/v3/accounts/{account_id}/orders"

    order_body = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }
    }

    r = requests.post(url, headers=_oanda_headers(), json=order_body)
    resp_data = r.json()
    status = "FILLED" if r.status_code == 201 else f"ERROR_{r.status_code}"

    print(f"  [LIVE] {side} {abs(units)} {instrument} -> {status}")
    log_order({
        "timestamp": now,
        "instrument": instrument,
        "side": side,
        "units": units,
        "price": resp_data.get("orderFillTransaction", {}).get("price", ""),
        "order_type": "MARKET",
        "mode": "live",
        "status": status,
        "response": str(resp_data),
    })
    return {"status": status, "mode": "live", "response": resp_data}


# ---------------------------------------------------------------------------
# Signal execution
# ---------------------------------------------------------------------------

def execute_signals(signals_df, cfg, ai_models=None):
    """
    Takes a DataFrame with signal column (from generate_signals).
    Reads the latest signal per instrument, compares to current positions,
    and places orders for needed changes.

    If ai_models are provided, validates signals through the AI decision
    wrapper before placing orders.

    Returns list of order results.
    """
    paper_mode = cfg.get("execution", {}).get("paper_mode", True)
    max_positions = cfg.get("execution", {}).get("max_positions", 3)

    # Get account balance for sizing
    if paper_mode:
        balance = 10_000.0  # default paper balance
        try:
            summary = get_account_summary()
            balance = summary["balance"]
            print(f"Account balance: {balance:,.2f}")
        except Exception:
            print(f"Using default paper balance: {balance:,.2f}")
    else:
        summary = get_account_summary()
        balance = summary["balance"]
        print(f"Account balance: {balance:,.2f}")

    # Get latest signal row
    if signals_df.empty:
        print("No signals to execute.")
        return []

    latest = signals_df.iloc[-1]
    instrument = latest.get("instrument", signals_df.attrs.get("instrument", "UNKNOWN"))
    signal = int(latest["signal"])
    close_price = float(latest["close"])

    # Determine current position for this instrument
    current_position = 0
    if not paper_mode:
        try:
            open_trades = get_open_trades()
            for trade in open_trades:
                if trade["instrument"] == instrument:
                    current_position = 1 if float(trade["currentUnits"]) > 0 else -1
        except Exception as e:
            print(f"  Error checking positions: {e}")

    results = []

    # Signal=1 means go long, signal=0 means go flat
    if signal == 1 and current_position == 0:
        # AI validation before placing BUY order
        if ai_models:
            ai_decision = validate_signal(instrument, signal, signals_df, cfg, models=ai_models)
            log_ai_decision(ai_decision)
            if not ai_decision["approved"]:
                print(f"  AI REJECTED: confidence={ai_decision['confidence']:.2f}, {ai_decision['rationale']}")
                return results

        units = compute_units(balance, instrument, "BUY", cfg)
        result = place_order(instrument, units, "BUY", cfg, price=close_price)
        results.append(result)
    elif signal == 0 and current_position == 1:
        units = compute_units(balance, instrument, "SELL", cfg)
        result = place_order(instrument, units, "SELL", cfg, price=close_price)
        results.append(result)
    else:
        action = "LONG" if signal == 1 else "FLAT"
        print(f"  {instrument}: signal={action}, position matches — no action.")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()

    print("=" * 60)
    print("fx-quant Order Executor")
    print("=" * 60)

    paper_mode = cfg.get("execution", {}).get("paper_mode", True)
    mode_label = "PAPER" if paper_mode else "LIVE"
    print(f"Mode: {mode_label}")

    # Kill switch check at startup
    if check_kill_switch():
        print("\nKILL SWITCH ACTIVE — STOP_ALL_TRADING file detected.")
        if not paper_mode:
            print("Closing all open trades...")
            close_all_trades()
        log_order({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "instrument": "ALL",
            "side": "CLOSE_ALL",
            "units": 0,
            "price": "",
            "order_type": "KILL_SWITCH",
            "mode": mode_label.lower(),
            "status": "HALTED",
            "response": "Kill switch activated at startup",
        })
        print("Logged kill switch event. Exiting.")
        return

    # Supabase client for fetching candles
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_KEY in config/.env")
    sb = create_client(supabase_url, supabase_key)
    table = cfg.get("supabase", {}).get("table", "fx_candles")

    strategy_cfg = cfg["strategy"]
    instruments = cfg["brokers"][0]["instruments"]
    granularity = cfg["data"]["candle_granularities"][0]  # use finest granularity

    print(f"Strategy: {strategy_cfg['rule']}")
    print(f"Instruments: {instruments}")
    print(f"Granularity: {granularity}")
    print()

    all_results = []

    for instrument in instruments:
        print(f"--- {instrument} ---")

        # Fetch latest candles from Supabase
        df = fetch_candles_from_supabase(instrument, granularity, sb, table)
        if df.empty:
            print(f"  No data for {instrument}. Skipping.\n")
            continue

        # Generate signals
        df = generate_signals(df, strategy_cfg)
        if df.empty:
            print(f"  No valid rows after signal generation. Skipping.\n")
            continue

        # Attach instrument name for execute_signals
        df.attrs["instrument"] = instrument
        if "instrument" not in df.columns:
            df["instrument"] = instrument

        latest_signal = "LONG" if df["signal"].iloc[-1] == 1 else "FLAT"
        print(f"  Latest signal: {latest_signal} (close={df['close'].iloc[-1]:.5f})")

        # Train AI ensemble for this instrument
        ai_models = None
        ai_cfg = cfg.get("ai", {})
        if ai_cfg.get("model") == "local-ensemble":
            print(f"  Training AI ensemble for {instrument}...")
            models_list, val_metrics = train_ensemble(df, strategy_cfg)
            if models_list:
                ai_models = models_list

        # Execute with AI validation
        results = execute_signals(df, cfg, ai_models=ai_models)
        all_results.extend(results)
        print()

    # Summary
    print("=" * 60)
    print(f"Execution complete. Orders processed: {len(all_results)}")
    for r in all_results:
        print(f"  {r}")
    print("=" * 60)


if __name__ == "__main__":
    cfg = load_config()
    interval = cfg.get("execution", {}).get("interval_seconds", 60)

    # One-shot mode: pass --once to run a single iteration and exit
    if "--once" in sys.argv:
        main()
    else:
        print(f"Running on {interval}s loop. Press Ctrl+C to stop.\n")
        while True:
            try:
                main()
                print(f"\nSleeping {interval}s until next run...\n")
                time.sleep(interval)
            except KeyboardInterrupt:
                print("\nShutting down gracefully.")
                break
