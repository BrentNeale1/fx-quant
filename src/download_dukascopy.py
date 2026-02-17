"""
Download historical OHLCV data from Dukascopy for all strategy pairs.
Date range: January 1, 2021 - December 31, 2024
Timeframes: M15, H1
Saves to data/raw/ as CSV files.
"""
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
from dukascopy_python import fetch, INTERVAL_MIN_15, INTERVAL_HOUR_1, OFFER_SIDE_BID

# All 10 currency pairs from the roadmap
PAIRS = [
    "GBP/AUD", "EUR/AUD", "EUR/GBP", "GBP/CAD", "EUR/CAD",
    "GBP/JPY", "USD/JPY", "GBP/USD", "EUR/USD", "EUR/NZD",
]

INTERVALS = {
    "M15": INTERVAL_MIN_15,
    "H1": INTERVAL_HOUR_1,
}

START = datetime(2021, 1, 1)
END = datetime(2024, 12, 31, 23, 59)

# Chunk by 6 months to stay well under the 30k row limit
CHUNK_MONTHS = 6

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


def chunk_date_range(start, end, months):
    """Split date range into chunks of N months."""
    chunks = []
    current = start
    while current < end:
        chunk_end = current + timedelta(days=months * 30)
        if chunk_end > end:
            chunk_end = end
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(hours=1)
    return chunks


def download_pair(pair, interval_name, interval_code):
    """Download data for one pair/timeframe combination."""
    pair_file = pair.replace("/", "_")
    filename = f"{pair_file}_{interval_name}.csv"
    filepath = os.path.join(RAW_DIR, filename)

    if os.path.exists(filepath):
        existing = pd.read_csv(filepath)
        print(f"  Already exists: {filename} ({len(existing)} rows) - skipping")
        return True

    chunks = chunk_date_range(START, END, CHUNK_MONTHS)
    all_dfs = []

    for i, (chunk_start, chunk_end) in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)}: {chunk_start.date()} to {chunk_end.date()}...", end=" ")
        try:
            df = fetch(
                instrument=pair,
                interval=interval_code,
                offer_side=OFFER_SIDE_BID,
                start=chunk_start,
                end=chunk_end,
                max_retries=5,
            )
            print(f"{len(df)} rows")
            if len(df) > 0:
                all_dfs.append(df)
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"ERROR: {e}")
            time.sleep(3)
            # Retry once
            try:
                df = fetch(
                    instrument=pair,
                    interval=interval_code,
                    offer_side=OFFER_SIDE_BID,
                    start=chunk_start,
                    end=chunk_end,
                    max_retries=5,
                )
                print(f"  Retry OK: {len(df)} rows")
                if len(df) > 0:
                    all_dfs.append(df)
            except Exception as e2:
                print(f"  Retry failed: {e2}")
                return False

    if not all_dfs:
        print(f"  No data retrieved for {pair} {interval_name}")
        return False

    combined = pd.concat(all_dfs)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()

    # Save with timestamp index
    combined.to_csv(filepath)
    print(f"  Saved: {filename} ({len(combined)} rows)")
    return True


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    total = len(PAIRS) * len(INTERVALS)
    completed = 0
    failed = []

    print(f"Downloading {total} pair-timeframe combinations from Dukascopy")
    print(f"Date range: {START.date()} to {END.date()}")
    print(f"Output: {RAW_DIR}")
    print("=" * 60)

    for pair in PAIRS:
        for interval_name, interval_code in INTERVALS.items():
            completed += 1
            print(f"\n[{completed}/{total}] {pair} {interval_name}")
            success = download_pair(pair, interval_name, interval_code)
            if not success:
                failed.append(f"{pair} {interval_name}")

    print("\n" + "=" * 60)
    print(f"Complete: {total - len(failed)}/{total}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    else:
        print("All downloads successful!")


if __name__ == "__main__":
    main()
