"""
Paper/live order execution for the live engine.

Thin wrapper around OANDA v20 order API. Reuses auth patterns from
src/order_executor.py but stripped of Supabase/old strategy coupling.
"""
import os
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# Project root
ROOT = Path(__file__).resolve().parent.parent.parent

# Load .env
_env_path = ROOT / "config" / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _oanda_base_url() -> str:
    env = os.getenv("OANDA_ENV", "practice").strip()
    if env == "practice":
        return "https://api-fxpractice.oanda.com"
    return "https://api-fxtrade.oanda.com"


def _oanda_headers() -> dict:
    api_key = os.getenv("OANDA_API_KEY")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


class OandaExecutor:
    """Executes orders in paper or live mode.

    Paper mode: logs trades to CSV, tracks positions locally.
    Live mode: POSTs to OANDA v20 REST API with SL/TP attached.
    """

    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self._logs_dir = ROOT / "logs"
        self._logs_dir.mkdir(exist_ok=True)
        self._csv_path = self._logs_dir / "paper_trades.csv"
        self._account_id = os.getenv("OANDA_ACCOUNT_ID", "")

    def check_kill_switch(self) -> bool:
        """Check for STOP_ALL_TRADING file in project root."""
        return (ROOT / "STOP_ALL_TRADING").exists()

    def place_market_order(self, pair: str, units: int, direction: str,
                           sl: float = None, tp: float = None) -> dict:
        """Place a market order with optional SL/TP.

        Args:
            pair: Instrument (e.g. "GBP_JPY").
            units: Position size in units.
            direction: "LONG" or "SHORT".
            sl: Stop loss price.
            tp: Take profit price.

        Returns dict with status and details.
        """
        if self.check_kill_switch():
            self._log_trade(pair, direction, units, 0, "BLOCKED_KILL_SWITCH", sl, tp)
            return {"status": "BLOCKED_KILL_SWITCH"}

        signed_units = units if direction == "LONG" else -units

        if self.paper_mode:
            now = datetime.now(timezone.utc).isoformat()
            self._log_trade(pair, direction, units, 0, "FILLED_PAPER", sl, tp)
            return {
                "status": "FILLED",
                "mode": "paper",
                "units": signed_units,
                "sl": sl,
                "tp": tp,
            }

        # Live mode
        return self._place_live_order(pair, signed_units, sl, tp)

    def modify_trade(self, trade_id: str, sl: float = None,
                     tp: float = None) -> dict:
        """Modify SL/TP on an existing OANDA trade."""
        if self.paper_mode:
            return {"status": "MODIFIED_PAPER", "trade_id": trade_id}

        base = _oanda_base_url()
        url = f"{base}/v3/accounts/{self._account_id}/trades/{trade_id}/orders"
        body = {}
        if sl is not None:
            body["stopLoss"] = {"price": f"{sl:.5f}", "timeInForce": "GTC"}
        if tp is not None:
            body["takeProfit"] = {"price": f"{tp:.5f}", "timeInForce": "GTC"}

        r = requests.put(url, headers=_oanda_headers(), json=body, timeout=10)
        return {"status": "OK" if r.status_code == 200 else f"ERROR_{r.status_code}",
                "response": r.json()}

    def close_trade(self, trade_id: str) -> dict:
        """Close a specific OANDA trade."""
        if self.paper_mode:
            return {"status": "CLOSED_PAPER", "trade_id": trade_id}

        base = _oanda_base_url()
        url = f"{base}/v3/accounts/{self._account_id}/trades/{trade_id}/close"
        r = requests.put(url, headers=_oanda_headers(), timeout=10)
        return {"status": "OK" if r.status_code == 200 else f"ERROR_{r.status_code}",
                "response": r.json()}

    def get_open_trades(self) -> list:
        """Get all open trades from OANDA."""
        if self.paper_mode:
            return []  # paper trades tracked by PositionManager

        base = _oanda_base_url()
        url = f"{base}/v3/accounts/{self._account_id}/openTrades"
        r = requests.get(url, headers=_oanda_headers(), timeout=10)
        r.raise_for_status()
        return r.json().get("trades", [])

    def get_account_summary(self) -> dict:
        """Get account balance, NAV, open position count."""
        base = _oanda_base_url()
        url = f"{base}/v3/accounts/{self._account_id}/summary"
        r = requests.get(url, headers=_oanda_headers(), timeout=10)
        r.raise_for_status()
        acct = r.json()["account"]
        return {
            "balance": float(acct["balance"]),
            "NAV": float(acct["NAV"]),
            "open_trade_count": int(acct["openTradeCount"]),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _place_live_order(self, pair: str, signed_units: int,
                          sl: float, tp: float) -> dict:
        base = _oanda_base_url()
        url = f"{base}/v3/accounts/{self._account_id}/orders"

        order = {
            "type": "MARKET",
            "instrument": pair,
            "units": str(signed_units),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }
        if sl is not None:
            order["stopLossOnFill"] = {"price": f"{sl:.5f}", "timeInForce": "GTC"}
        if tp is not None:
            order["takeProfitOnFill"] = {"price": f"{tp:.5f}", "timeInForce": "GTC"}

        r = requests.post(url, headers=_oanda_headers(),
                          json={"order": order}, timeout=10)
        resp = r.json()
        status = "FILLED" if r.status_code == 201 else f"ERROR_{r.status_code}"
        direction = "LONG" if signed_units > 0 else "SHORT"

        fill_price = resp.get("orderFillTransaction", {}).get("price", 0)
        self._log_trade(pair, direction, abs(signed_units), fill_price, status, sl, tp)

        return {"status": status, "mode": "live", "response": resp}

    def _log_trade(self, pair: str, direction: str, units: int,
                   price: float, status: str, sl: float, tp: float):
        """Append trade to paper_trades.csv."""
        columns = [
            "timestamp", "pair", "direction", "units", "price",
            "sl", "tp", "status", "mode",
        ]
        now = datetime.now(timezone.utc).isoformat()
        row = {
            "timestamp": now,
            "pair": pair,
            "direction": direction,
            "units": units,
            "price": price,
            "sl": sl or "",
            "tp": tp or "",
            "status": status,
            "mode": "paper" if self.paper_mode else "live",
        }

        file_exists = self._csv_path.exists()
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
