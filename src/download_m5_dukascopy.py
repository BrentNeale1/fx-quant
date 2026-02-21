"""
Download M5 (5-minute) historical data from Dukascopy.

Pairs needed for 5-minute strategies (S12, S15, S16 on M5):
  GBP_JPY, GBP_USD, EUR_USD, USD_JPY

Date range: Jan 1, 2021 - Dec 31, 2024 (matches existing M15/H1 data)
Saves to data/raw/ as CSV, then split via validate_and_split_data.py.

M5 generates ~290k bars per pair over 4 years, so we use 2-month chunks
to stay well under Dukascopy's row limit.
"""
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
from dukascopy_python import fetch, INTERVAL_MIN_5, OFFER_SIDE_BID

# Pairs needed for 5-minute strategies
PAIRS = [
    "GBP/JPY", "GBP/USD", "EUR/USD", "USD/JPY",
]

INTERVAL_NAME = "M5"
INTERVAL_CODE = INTERVAL_MIN_5

START = datetime(2021, 1, 1)
END = datetime(2024, 12, 31, 23, 59)

# Chunk by 2 months — M5 produces ~17k bars/month, so 2 months = ~34k
CHUNK_MONTHS = 2

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


def download_pair(pair):
    """Download M5 data for one pair."""
    pair_file = pair.replace("/", "_")
    filename = f"{pair_file}_{INTERVAL_NAME}.csv"
    filepath = os.path.join(RAW_DIR, filename)

    if os.path.exists(filepath):
        existing = pd.read_csv(filepath)
        print(f"  Already exists: {filename} ({len(existing)} rows) - skipping")
        return True

    chunks = chunk_date_range(START, END, CHUNK_MONTHS)
    all_dfs = []

    for i, (chunk_start, chunk_end) in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)}: {chunk_start.date()} to {chunk_end.date()}...",
              end=" ", flush=True)
        try:
            df = fetch(
                instrument=pair,
                interval=INTERVAL_CODE,
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
                    interval=INTERVAL_CODE,
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
        print(f"  No data retrieved for {pair}")
        return False

    combined = pd.concat(all_dfs)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()

    # Save
    os.makedirs(RAW_DIR, exist_ok=True)
    combined.to_csv(filepath)
    print(f"  Saved: {filename} ({len(combined)} rows)")
    print(f"  Range: {combined.index[0]} to {combined.index[-1]}")
    return True


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print(f"Downloading M5 data from Dukascopy")
    print(f"Pairs: {', '.join(PAIRS)}")
    print(f"Date range: {START.date()} to {END.date()}")
    print(f"Output: {RAW_DIR}")
    print("=" * 60)

    failed = []
    for i, pair in enumerate(PAIRS):
        print(f"\n[{i+1}/{len(PAIRS)}] {pair} M5")
        success = download_pair(pair)
        if not success:
            failed.append(pair)

    print("\n" + "=" * 60)
    print(f"Complete: {len(PAIRS) - len(failed)}/{len(PAIRS)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    else:
        print("All M5 downloads successful!")
        print(f"\nNext step: Run validate_and_split to process the data:")
        print(f"  python src/validate_and_split_data.py")


if __name__ == "__main__":
    main()
