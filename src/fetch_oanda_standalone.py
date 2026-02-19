"""
Standalone OANDA data fetcher — no Supabase dependency.
Fetches H1 candles and saves to data/processed/ in the same CSV format
as existing data files (timestamp, open, high, low, close, volume).

Usage:
    python src/fetch_oanda_standalone.py
"""

import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import pandas as pd

# Load OANDA key from config/.env
ROOT = Path(__file__).resolve().parent.parent
env_path = ROOT / "config" / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.getenv("OANDA_API_KEY")
ENV = os.getenv("OANDA_ENV", "practice")
BASE = (
    "https://api-fxpractice.oanda.com"
    if ENV == "practice"
    else "https://api-fxtrade.oanda.com"
)
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def fetch_candles_chunk(instrument, granularity, from_dt, to_dt):
    """Fetch up to 5000 candles between from_dt and to_dt."""
    params = {
        "granularity": granularity,
        "from": from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to": to_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    url = f"{BASE}/v3/instruments/{instrument}/candles"
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json().get("candles", [])


def fetch_all_candles(instrument, granularity, start_dt, end_dt):
    """Paginate through OANDA history in chunks of ~5000 candles."""
    gran_minutes = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240, "D": 1440, "W": 10080,
    }
    minutes = gran_minutes.get(granularity, 60)
    chunk_duration = timedelta(minutes=minutes * 4999)

    all_candles = []
    cursor = start_dt
    chunk_num = 0

    while cursor < end_dt:
        chunk_end = min(cursor + chunk_duration, end_dt)
        chunk_num += 1
        print(f"  Chunk {chunk_num}: {cursor.strftime('%Y-%m-%d %H:%M')} -> "
              f"{chunk_end.strftime('%Y-%m-%d %H:%M')} ...", end=" ", flush=True)

        candles = fetch_candles_chunk(instrument, granularity, cursor, chunk_end)
        print(f"{len(candles)} candles", flush=True)

        if candles:
            all_candles.extend(candles)
            last_time = pd.to_datetime(candles[-1]["time"])
            cursor = last_time.to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(minutes=minutes)
        else:
            cursor = chunk_end

        time.sleep(0.5)

    print(f"  Total raw candles fetched: {len(all_candles)}", flush=True)

    if not all_candles:
        return pd.DataFrame()

    # Convert to DataFrame
    rows = []
    for c in all_candles:
        mid = c["mid"]
        rows.append({
            "timestamp": c["time"],
            "open": float(mid["o"]),
            "high": float(mid["h"]),
            "low": float(mid["l"]),
            "close": float(mid["c"]),
            "volume": c.get("volume", 0),
        })

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"  After dedup: {len(df)} candles", flush=True)
    print(f"  Range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}", flush=True)
    return df


def main():
    if not API_KEY:
        print("ERROR: No OANDA_API_KEY found in config/.env")
        sys.exit(1)

    # Match existing data range: 2020-12-31 to 2023-08-31
    start_dt = datetime(2020, 12, 31, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(2023, 8, 31, 23, 59, tzinfo=timezone.utc)

    pairs = ["EUR_GBP", "EUR_USD", "GBP_NZD"]
    granularity = "H1"
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        out_file = out_dir / f"{pair}_{granularity}.csv"
        if out_file.exists():
            print(f"\n{pair} already exists at {out_file}, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Fetching {pair} / {granularity}  ({start_dt.date()} -> {end_dt.date()})")
        print(f"{'='*60}")

        df = fetch_all_candles(pair, granularity, start_dt, end_dt)
        if df.empty:
            print(f"  No data for {pair}. Skipping.")
            continue

        df.to_csv(out_file, index=False)
        print(f"  Saved: {out_file} ({len(df)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
