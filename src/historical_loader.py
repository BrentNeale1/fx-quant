# src/historical_loader.py
"""
Fetch ~1 year of historical OANDA candles in paginated chunks (max 5000 per request),
compute features, and upload to Supabase.

Usage:
    python src/historical_loader.py
"""

import os
import time
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd

from config_loader import load_config
from data_engine import candles_to_df, build_all_features
from supabase_upload import upload_dataframe

cfg = load_config()

API_KEY = os.getenv("OANDA_API_KEY")
ENV = os.getenv("OANDA_ENV", "practice")
BASE = (
    "https://api-fxpractice.oanda.com"
    if ENV == "practice"
    else "https://api-fxtrade.oanda.com"
)
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def fetch_candles_chunk(instrument, granularity, from_dt, to_dt, count=5000):
    """Fetch up to `count` candles between from_dt and to_dt."""
    params = {
        "granularity": granularity,
        "from": from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to": to_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    url = f"{BASE}/v3/instruments/{instrument}/candles"
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json().get("candles", [])


def fetch_all_candles(instrument, granularity, days_back=365):
    """
    Paginate through OANDA history in chunks of 5000 candles.
    Returns a single merged DataFrame with all candles.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)

    # Determine candle duration for stepping forward
    gran_minutes = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240, "D": 1440, "W": 10080,
    }
    minutes = gran_minutes.get(granularity, 5)
    chunk_duration = timedelta(minutes=minutes * 4999)  # just under 5000 candles

    all_candles = []
    cursor = start
    chunk_num = 0

    while cursor < end:
        chunk_end = min(cursor + chunk_duration, end)
        chunk_num += 1
        print(f"  Chunk {chunk_num}: {cursor.strftime('%Y-%m-%d %H:%M')} -> {chunk_end.strftime('%Y-%m-%d %H:%M')} ...", end=" ")

        candles = fetch_candles_chunk(instrument, granularity, cursor, chunk_end)
        print(f"{len(candles)} candles")

        if candles:
            all_candles.extend(candles)
            # Move cursor past the last candle we received
            last_time = pd.to_datetime(candles[-1]["time"])
            cursor = last_time.to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(minutes=minutes)
        else:
            # No data in this window, skip forward
            cursor = chunk_end

        # Be polite to the API
        time.sleep(0.5)

    print(f"  Total raw candles fetched: {len(all_candles)}")

    if not all_candles:
        return pd.DataFrame()

    # Deduplicate by time (overlapping chunks)
    df = candles_to_df(all_candles)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    print(f"  After dedup: {len(df)} candles")
    print(f"  Range: {df.index[0]} -> {df.index[-1]}")

    return df


def main():
    feature_cfg = cfg.get("features", {})
    broker = cfg["brokers"][0]
    instruments = broker["instruments"]
    granularities = cfg["data"]["candle_granularities"]

    for instrument in instruments:
        for granularity in granularities:
            print(f"\n{'='*60}")
            print(f"Fetching {instrument} / {granularity} — last 365 days")
            print(f"{'='*60}")

            df = fetch_all_candles(instrument, granularity, days_back=365)
            if df.empty:
                print(f"  No data. Skipping.")
                continue

            print(f"  Computing features...")
            df = build_all_features(df, config=feature_cfg)

            print(f"  Uploading to Supabase...")
            upload_dataframe(df, instrument=instrument, granularity=granularity, chunk_size=500)
            print(f"  Done: {instrument} / {granularity}")

    print(f"\nAll historical data loaded.")


if __name__ == "__main__":
    main()
