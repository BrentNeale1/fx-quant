"""
OANDA candle polling + indicator computation for the live engine.

Fetches recent candles from OANDA REST API, computes technical indicators,
and exposes them for strategy signal checks.
"""
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import pandas as pd

from src.indicators.technical import compute_all_indicators

# Load OANDA credentials from config/.env
ROOT = Path(__file__).resolve().parent.parent.parent
_env_path = ROOT / "config" / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ENV = os.getenv("OANDA_ENV", "practice")
BASE_URL = (
    "https://api-fxpractice.oanda.com"
    if OANDA_ENV == "practice"
    else "https://api-fxtrade.oanda.com"
)
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def _fetch_candles(instrument: str, granularity: str, count: int) -> list[dict]:
    """Fetch the most recent `count` candles from OANDA."""
    url = f"{BASE_URL}/v3/instruments/{instrument}/candles"
    params = {
        "granularity": granularity,
        "count": count,
        "price": "M",  # mid prices
    }
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("candles", [])


def _candles_to_df(candles: list[dict]) -> pd.DataFrame:
    """Convert OANDA candle JSON to a DataFrame matching our data format."""
    rows = []
    for c in candles:
        if not c.get("complete", False):
            continue  # skip incomplete candles for indicator calculation
        mid = c["mid"]
        rows.append({
            "timestamp": c["time"],
            "open": float(mid["o"]),
            "high": float(mid["h"]),
            "low": float(mid["l"]),
            "close": float(mid["c"]),
            "volume": c.get("volume", 0),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.set_index("timestamp").sort_index()
    df.index.name = "timestamp"
    return df


class OandaDataFeed:
    """Polls OANDA for latest candles and computes indicators.

    Args:
        pair: OANDA instrument name (e.g. "GBP_JPY").
        granularity: Candle granularity (e.g. "H1", "M15").
        candle_count: How many candles to fetch (enough for indicator warmup).
    """

    def __init__(self, pair: str, granularity: str = "H1",
                 candle_count: int = 250):
        self.pair = pair
        self.granularity = granularity
        self.candle_count = candle_count
        self._cached_df: pd.DataFrame = pd.DataFrame()
        self._last_fetch_time: float = 0

    def fetch_latest(self) -> pd.DataFrame:
        """Fetch latest candles and compute all indicators.

        Returns DataFrame with OHLCV + all indicator columns.
        Only includes fully closed candles.
        """
        candles = _fetch_candles(self.pair, self.granularity, self.candle_count)
        df = _candles_to_df(candles)
        if df.empty:
            return self._cached_df

        df = compute_all_indicators(df)
        self._cached_df = df
        self._last_fetch_time = time.time()
        return df

    def get_current_candle(self) -> pd.Series:
        """Return the most recent fully closed candle with indicators."""
        if self._cached_df.empty:
            self.fetch_latest()
        if self._cached_df.empty:
            return pd.Series()
        return self._cached_df.iloc[-1]

    def get_htf_row(self, timestamp=None) -> pd.Series:
        """Get the most recent fully closed candle as HTF context.

        For H1 strategies where HTF=H1 (self-reference), this is just
        the latest closed candle. For M15 strategies needing H1 context,
        a separate OandaDataFeed for H1 should be used.
        """
        if self._cached_df.empty:
            return pd.Series()
        if timestamp is not None:
            valid = self._cached_df[self._cached_df.index < timestamp]
            if valid.empty:
                return pd.Series()
            return valid.iloc[-1]
        return self._cached_df.iloc[-1]

    @property
    def is_stale(self) -> bool:
        """True if data hasn't been fetched recently."""
        if self._last_fetch_time == 0:
            return True
        # H1 = stale after 65 min, M15 = stale after 20 min
        gran_minutes = {"H1": 65, "M15": 20, "M5": 8}
        stale_seconds = gran_minutes.get(self.granularity, 65) * 60
        return (time.time() - self._last_fetch_time) > stale_seconds
