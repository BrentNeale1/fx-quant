"""
Data Validation and Train/Test Split Script.

Validates raw Dukascopy data quality, then splits:
  - Training set (70%): Jan 2021 - Aug 2023 -> data/processed/
  - Test set (30%): Sep 2023 - Dec 2024 -> data/test_set/

Per the roadmap: test set must NOT be accessed during Phases 1-3.
"""
import os
import sys
import glob

import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
TEST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "test_set")

# Split date: training ends Aug 31 2023, test starts Sep 1 2023
SPLIT_DATE = "2023-09-01"

# Expected timeframe intervals
EXPECTED_INTERVALS = {
    "M5": pd.Timedelta(minutes=5),
    "M15": pd.Timedelta(minutes=15),
    "H1": pd.Timedelta(hours=1),
}

# Pip values for spread sanity checks (approximate)
PIP_VALUES = {
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "EUR_AUD": 0.0001,
    "GBP_AUD": 0.0001, "EUR_GBP": 0.0001, "GBP_CAD": 0.0001,
    "EUR_CAD": 0.0001, "EUR_NZD": 0.0001,
    "USD_JPY": 0.01, "GBP_JPY": 0.01,
}


def load_raw_csv(filepath):
    """Load a raw Dukascopy CSV file."""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    return df


def validate_data(df, pair, timeframe):
    """Run validation checks on a dataframe. Returns list of issues."""
    issues = []
    tf_delta = EXPECTED_INTERVALS[timeframe]
    pip = PIP_VALUES.get(pair, 0.0001)

    # 1. Check for missing candles (gaps in timestamp sequence)
    diffs = df.index.to_series().diff().dropna()
    # Filter to only market hours gaps (ignore weekends)
    # A "gap" is any diff > 2x expected interval (allows for some market closures)
    large_gaps = diffs[diffs > tf_delta * 4]
    # Exclude weekend gaps (Fri->Sun typically)
    weekday_gaps = []
    for ts, gap in large_gaps.items():
        # Skip if gap starts on Friday and ends on Sunday/Monday (weekend)
        prev_ts = ts - gap
        if prev_ts.weekday() == 4 and ts.weekday() in [0, 6]:  # Fri->Sun or Fri->Mon
            continue
        # Skip Christmas/New Year period
        if prev_ts.month == 12 and prev_ts.day >= 24:
            continue
        if ts.month == 1 and ts.day <= 3:
            continue
        weekday_gaps.append((ts, gap))

    if len(weekday_gaps) > 20:
        issues.append(f"WARNING: {len(weekday_gaps)} non-weekend gaps > 4x interval")
    elif len(weekday_gaps) > 0:
        issues.append(f"INFO: {len(weekday_gaps)} non-weekend gaps (likely holidays)")

    # 2. Check for zero-volume candles
    if "volume" in df.columns:
        zero_vol = (df["volume"] == 0).sum()
        if zero_vol > 0:
            issues.append(f"WARNING: {zero_vol} zero-volume candles ({zero_vol/len(df)*100:.2f}%)")

    # 3. Check for extreme outliers (candles with >10x average range)
    candle_range = df["high"] - df["low"]
    avg_range = candle_range.mean()
    extreme = (candle_range > avg_range * 10).sum()
    if extreme > 0:
        issues.append(f"WARNING: {extreme} candles with range >10x average")

    # 4. Check for duplicate timestamps
    dupes = df.index.duplicated().sum()
    if dupes > 0:
        issues.append(f"ERROR: {dupes} duplicate timestamps")

    # 5. Check date range coverage
    start = df.index.min()
    end = df.index.max()
    issues.append(f"INFO: Date range {start.date()} to {end.date()}")

    # 6. Session coverage check
    hours = df.index.hour
    asian_count = ((hours >= 0) & (hours < 8)).sum()
    london_count = ((hours >= 8) & (hours < 16)).sum()
    ny_count = ((hours >= 13) & (hours < 22)).sum()
    if asian_count == 0:
        issues.append("ERROR: No Asian session data")
    if london_count == 0:
        issues.append("ERROR: No London session data")
    if ny_count == 0:
        issues.append("ERROR: No NY session data")

    # 7. Basic OHLC sanity
    bad_ohlc = ((df["high"] < df["low"]) | (df["high"] < df["open"]) |
                (df["high"] < df["close"]) | (df["low"] > df["open"]) |
                (df["low"] > df["close"])).sum()
    if bad_ohlc > 0:
        issues.append(f"ERROR: {bad_ohlc} candles with invalid OHLC relationships")

    return issues


def split_and_save(df, pair, timeframe):
    """Split data into train/test and save to respective directories."""
    split_ts = pd.Timestamp(SPLIT_DATE, tz=df.index.tz) if df.index.tz else pd.Timestamp(SPLIT_DATE)

    train = df[df.index < split_ts]
    test = df[df.index >= split_ts]

    filename = f"{pair}_{timeframe}.csv"

    train_path = os.path.join(PROCESSED_DIR, filename)
    test_path = os.path.join(TEST_DIR, filename)

    train.to_csv(train_path)
    test.to_csv(test_path)

    return len(train), len(test)


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {RAW_DIR}")
        sys.exit(1)

    print(f"Found {len(csv_files)} files in {RAW_DIR}")
    print(f"Split date: {SPLIT_DATE}")
    print(f"Training -> {PROCESSED_DIR}")
    print(f"Test -> {TEST_DIR}")
    print("=" * 70)

    summary = []
    all_ok = True

    for filepath in csv_files:
        basename = os.path.basename(filepath).replace(".csv", "")
        parts = basename.rsplit("_", 1)
        pair = parts[0]
        timeframe = parts[1]

        print(f"\n--- {pair} {timeframe} ---")

        df = load_raw_csv(filepath)
        print(f"  Raw rows: {len(df)}")

        # Remove duplicates
        before = len(df)
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
        if len(df) < before:
            print(f"  Removed {before - len(df)} duplicate rows")

        # Validate
        issues = validate_data(df, pair, timeframe)
        for issue in issues:
            level = issue.split(":")[0]
            print(f"  {issue}")
            if level == "ERROR":
                all_ok = False

        # Split and save
        train_count, test_count = split_and_save(df, pair, timeframe)
        print(f"  Train: {train_count} rows | Test: {test_count} rows | Split: {train_count/(train_count+test_count)*100:.1f}%/{test_count/(train_count+test_count)*100:.1f}%")

        summary.append({
            "pair": pair,
            "timeframe": timeframe,
            "raw_rows": len(df),
            "train_rows": train_count,
            "test_rows": test_count,
            "issues": len([i for i in issues if i.startswith("ERROR") or i.startswith("WARNING")]),
        })

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Pair':<12} {'TF':<5} {'Raw':>8} {'Train':>8} {'Test':>8} {'Issues':>7}")
    print("-" * 55)
    for s in summary:
        print(f"{s['pair']:<12} {s['timeframe']:<5} {s['raw_rows']:>8} {s['train_rows']:>8} {s['test_rows']:>8} {s['issues']:>7}")

    total_train = sum(s["train_rows"] for s in summary)
    total_test = sum(s["test_rows"] for s in summary)
    print("-" * 55)
    print(f"{'TOTAL':<18} {sum(s['raw_rows'] for s in summary):>8} {total_train:>8} {total_test:>8}")

    if all_ok:
        print("\nAll validation checks passed. Data is ready for Phase 1.")
    else:
        print("\nSome validation checks failed. Review errors above.")


if __name__ == "__main__":
    main()
