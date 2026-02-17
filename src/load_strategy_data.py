# src/load_strategy_data.py
"""
Bulk historical data loader for all 5 trading strategies.
Fetches candles for all required pairs and timeframes, computes features,
and uploads to Supabase.

Usage:
    python src/load_strategy_data.py
"""

import sys
from datetime import datetime, timezone

from config_loader import load_config
from historical_loader import fetch_all_candles
from data_engine import build_all_features
from supabase_upload import upload_dataframe

# Pairs required per strategy
STRATEGY_PAIRS = {
    "S1_MA_Breakout_Retest":    ["GBP_AUD", "EUR_AUD", "EUR_CAD", "EUR_NZD"],
    "S2_Session_VWAP_Reversal": ["GBP_USD", "EUR_USD", "GBP_JPY", "USD_JPY"],
    "S3_Key_Level_Breakout":    ["GBP_JPY", "USD_JPY", "GBP_USD", "EUR_GBP"],
    "S4_EMA_Ribbon_Scalp":     ["GBP_AUD", "EUR_AUD", "EUR_GBP"],
    "S5_Momentum_Exhaustion":  ["GBP_AUD", "EUR_AUD", "EUR_GBP", "GBP_CAD", "EUR_CAD"],
}

# All unique pairs
ALL_PAIRS = sorted(set(p for pairs in STRATEGY_PAIRS.values() for p in pairs))

# Timeframes and their required lookback in days
TIMEFRAME_DAYS = {
    "M15": 750,   # ~70,000 candles (market hours)
    "H1":  850,   # ~20,000 candles
    "H4":  1000,  # ~6,000 candles
}


def main():
    cfg = load_config()
    feature_cfg = cfg.get("features", {})

    print("=" * 70)
    print("Strategy Data Loader")
    print(f"Pairs:      {len(ALL_PAIRS)} - {ALL_PAIRS}")
    print(f"Timeframes: {list(TIMEFRAME_DAYS.keys())}")
    print(f"Started:    {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    total = len(ALL_PAIRS) * len(TIMEFRAME_DAYS)
    done = 0
    errors = []

    for instrument in ALL_PAIRS:
        for granularity, days_back in TIMEFRAME_DAYS.items():
            done += 1
            print(f"\n[{done}/{total}] {instrument} / {granularity} - {days_back} days back")
            print("-" * 60)

            try:
                df = fetch_all_candles(instrument, granularity, days_back=days_back)
                if df.empty:
                    print(f"  No data returned. Skipping.")
                    errors.append((instrument, granularity, "no data"))
                    continue

                print(f"  Candles: {len(df)} | Range: {df.index[0]} -> {df.index[-1]}")

                print(f"  Computing features...")
                df = build_all_features(df, config=feature_cfg)

                print(f"  Uploading to Supabase ({len(df)} rows)...")
                upload_dataframe(df, instrument=instrument, granularity=granularity, chunk_size=500)
                print(f"  Done.")

            except Exception as e:
                print(f"  ERROR: {e}")
                errors.append((instrument, granularity, str(e)))

    print("\n" + "=" * 70)
    print("Load complete.")
    print(f"  Successful: {done - len(errors)} / {total}")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for inst, gran, err in errors:
            print(f"    {inst} / {gran}: {err}")
    print("=" * 70)


if __name__ == "__main__":
    main()
