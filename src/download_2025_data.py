"""
Download 2025 data from Dukascopy for the 4 passing Phase 2 strategies.

Pairs/TFs needed:
  - GBP_JPY H1  (S7_Tight, S3)
  - GBP_AUD H1  (S9_Filtered)
  - GBP_USD M15 (S8_OB primary)
  - GBP_USD H1  (S8_OB HTF context)

Downloads from Oct 2024 (warmup) through Dec 2025.
Saves to data/2025/ directory.
"""
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
from dukascopy_python import fetch, INTERVAL_MIN_15, INTERVAL_HOUR_1, OFFER_SIDE_BID

PAIRS_TFS = [
    ("GBP/JPY", "H1", INTERVAL_HOUR_1),
    ("GBP/AUD", "H1", INTERVAL_HOUR_1),
    ("GBP/USD", "H1", INTERVAL_HOUR_1),
    ("GBP/USD", "M15", INTERVAL_MIN_15),
]

# Oct 2024 for warmup, through end of 2025
START = datetime(2024, 10, 1)
END = datetime(2025, 12, 31, 23, 59)

CHUNK_MONTHS = 3

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "2025")


def chunk_date_range(start, end, months):
    chunks = []
    current = start
    while current < end:
        chunk_end = current + timedelta(days=months * 30)
        if chunk_end > end:
            chunk_end = end
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(hours=1)
    return chunks


def download_pair(pair, tf_name, tf_code):
    pair_file = pair.replace("/", "_")
    filename = f"{pair_file}_{tf_name}.csv"
    filepath = os.path.join(OUT_DIR, filename)

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
                interval=tf_code,
                offer_side=OFFER_SIDE_BID,
                start=chunk_start,
                end=chunk_end,
                max_retries=5,
            )
            print(f"{len(df)} rows")
            if len(df) > 0:
                all_dfs.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"ERROR: {e}")
            time.sleep(3)
            try:
                df = fetch(
                    instrument=pair,
                    interval=tf_code,
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
        print(f"  No data retrieved for {pair} {tf_name}")
        return False

    combined = pd.concat(all_dfs)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()
    combined.to_csv(filepath)
    print(f"  Saved: {filename} ({len(combined)} rows, "
          f"{combined.index[0].date()} to {combined.index[-1].date()})")
    return True


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Downloading 2025 data from Dukascopy")
    print(f"Date range: {START.date()} to {END.date()}")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)

    failed = []
    for pair, tf_name, tf_code in PAIRS_TFS:
        print(f"\n{pair} {tf_name}")
        success = download_pair(pair, tf_name, tf_code)
        if not success:
            failed.append(f"{pair} {tf_name}")

    print("\n" + "=" * 60)
    if failed:
        print(f"Failed: {', '.join(failed)}")
    else:
        print("All downloads successful!")
        print(f"\nNext: python src/run_2025_validation.py")


if __name__ == "__main__":
    main()
