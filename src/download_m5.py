"""Download M5 candle data from OANDA for S4-F-v2 testing.

Fetches 2022-07-01 to 2024-12-31 (extra warmup) for GBP_AUD, EUR_AUD, GBP_JPY.
Saves raw OHLCV CSVs to data/processed/{PAIR}_M5.csv.
"""
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.config_loader import load_config

# Load config + .env
cfg = load_config()
API_KEY = os.getenv("OANDA_API_KEY")
ENV = os.getenv("OANDA_ENV", "practice")
BASE = (
    "https://api-fxpractice.oanda.com"
    if ENV == "practice"
    else "https://api-fxtrade.oanda.com"
)
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

PAIRS = ["GBP_AUD", "EUR_AUD", "GBP_JPY"]
GRANULARITY = "M5"
START_DATE = datetime(2022, 7, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")


def fetch_chunk(instrument, from_dt, to_dt):
    """Fetch candles between two datetimes."""
    params = {
        "granularity": GRANULARITY,
        "from": from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to": to_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "price": "M",  # mid prices
    }
    url = f"{BASE}/v3/instruments/{instrument}/candles"
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json().get("candles", [])


def candles_to_df(candles):
    """Convert OANDA candle list to DataFrame."""
    rows = []
    for c in candles:
        if not c.get("complete", True):
            continue
        mid = c["mid"]
        rows.append({
            "time": c["time"],
            "open": float(mid["o"]),
            "high": float(mid["h"]),
            "low": float(mid["l"]),
            "close": float(mid["c"]),
            "volume": int(c.get("volume", 0)),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    return df


def download_pair(pair):
    """Download all M5 data for a pair in paginated chunks."""
    print(f"\n{'='*60}")
    print(f"Downloading {pair} {GRANULARITY}")
    print(f"Range: {START_DATE.date()} -> {END_DATE.date()}")
    print(f"{'='*60}")

    chunk_duration = timedelta(minutes=5 * 4999)  # ~4999 candles per chunk
    all_candles = []
    cursor = START_DATE
    chunk_num = 0

    while cursor < END_DATE:
        chunk_end = min(cursor + chunk_duration, END_DATE)
        chunk_num += 1
        print(f"  Chunk {chunk_num}: {cursor.strftime('%Y-%m-%d %H:%M')} -> "
              f"{chunk_end.strftime('%Y-%m-%d %H:%M')} ... ", end="", flush=True)

        try:
            candles = fetch_chunk(pair, cursor, chunk_end)
            print(f"{len(candles)} candles")
        except Exception as e:
            print(f"ERROR: {e}")
            candles = []

        if candles:
            all_candles.extend(candles)
            last_time = pd.to_datetime(candles[-1]["time"])
            cursor = last_time.to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(minutes=5)
        else:
            cursor = chunk_end

        time.sleep(0.3)

    print(f"  Total raw candles: {len(all_candles)}")

    if not all_candles:
        print(f"  NO DATA for {pair}!")
        return None

    df = candles_to_df(all_candles)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    print(f"  After dedup: {len(df)} candles")
    print(f"  Range: {df.index[0]} -> {df.index[-1]}")

    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, f"{pair}_M5.csv")
    df.to_csv(out_path)
    print(f"  Saved: {out_path}")

    return df


if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: OANDA_API_KEY not set. Check config/.env")
        sys.exit(1)

    print(f"OANDA API: {BASE}")
    print(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}")

    for pair in PAIRS:
        download_pair(pair)

    print("\nDone! M5 data downloaded.")
