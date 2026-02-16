# src/supabase_upload.py
import os
import json
import math
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
import pandas as pd
import numpy as np

from config_loader import load_config

# Load config (also loads .env as side effect)
cfg = load_config()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")   # use service_role key for server writes
TABLE = cfg.get("supabase", {}).get("table", "fx_candles")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("Missing SUPABASE_URL or SUPABASE_KEY in config/.env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def df_to_records(df: pd.DataFrame, instrument: str, granularity: str = "M1"):
    """
    Convert DataFrame to a list of dicts suitable for Supabase/Postgres.
    Ensures time is ISO string and numeric types are regular Python types.
    Replaces inf/-inf with None and converts NaN -> None.
    """
    df2 = df.copy()

    # Ensure index is time and reset to a column
    if df2.index.name is None:
        df2.index.name = "time"
    df2 = df2.reset_index()

    # Add instrument and granularity columns
    df2["instrument"] = instrument
    df2["granularity"] = granularity

    # Convert Timestamp -> ISO string (timezone-aware preserved)
    df2["time"] = df2["time"].apply(lambda t: pd.to_datetime(t).isoformat())

    # Replace infinite values with NaN so they become None later
    df2 = df2.replace([np.inf, -np.inf], np.nan)

    def _convert_cell(v):
        # None / pandas NA
        if pd.isna(v):
            return None

        # Native boolean - keep as bool
        if isinstance(v, (bool,)):
            return bool(v)

        # Integers (numpy or native)
        if isinstance(v, (np.integer, int)):
            return int(v)

        # Floats (numpy or native) - ensure finite
        if isinstance(v, (np.floating, float)):
            try:
                fv = float(v)
            except Exception:
                return None
            return fv if math.isfinite(fv) else None

        # Strings - keep, but try numeric coercion if they look numeric
        if isinstance(v, str):
            s = v.strip()
            if s.replace('.', '', 1).replace('-', '', 1).isdigit():
                try:
                    fv = float(s)
                    return fv if math.isfinite(fv) else None
                except Exception:
                    return s
            return s

        # For any other type attempt a last-resort numeric coercion
        try:
            fv = float(v)
            return fv if math.isfinite(fv) else None
        except Exception:
            try:
                return str(v)
            except Exception:
                return None

    # Apply conversion column-by-column (converts numpy dtypes -> python scalars)
    for col in df2.columns:
        df2[col] = df2[col].apply(_convert_cell)

    records = df2.to_dict(orient="records")
    return records


def upload_dataframe(df: pd.DataFrame, instrument="EUR_USD", granularity="M1", chunk_size=500):
    records = df_to_records(df, instrument, granularity)

    # debug: check first record is JSON serializable
    if records:
        try:
            json.dumps(records[0])
            print("First record JSON dump OK.")
        except Exception as e:
            print("JSON dump failed for first record:", e)
            for k, v in records[0].items():
                try:
                    json.dumps({k: v})
                except Exception as e2:
                    print(
                        "  BAD KEY:", k,
                        "VALUE TYPE:", type(v),
                        "VALUE (truncated):", repr(str(v))[:200],
                        "->", e2,
                    )

    total = len(records)
    if total == 0:
        print("No rows to upload.")
        return

    chunks = math.ceil(total / chunk_size)
    print(f"Uploading {total} rows to Supabase in {chunks} chunk(s)...")

    for i in range(chunks):
        start = i * chunk_size
        end = start + chunk_size
        batch = records[start:end]

        resp = supabase.table(TABLE).upsert(batch).execute()
        if resp and getattr(resp, "status_code", None) not in (200, 201):
            print("Upload chunk error:", resp)
            raise SystemExit("Upsert failed")
        print(f"Chunk {i+1}/{chunks} uploaded ({len(batch)} rows).")


if __name__ == "__main__":
    from data_engine import candles_to_df, build_all_features
    import requests

    API_KEY = os.getenv("OANDA_API_KEY")
    ENV = os.getenv("OANDA_ENV", "practice")
    base = (
        "https://api-fxpractice.oanda.com"
        if ENV == "practice"
        else "https://api-fxtrade.oanda.com"
    )

    feature_cfg = cfg.get("features", {})
    broker = cfg["brokers"][0]
    instruments = broker["instruments"]
    granularities = cfg["data"]["candle_granularities"]
    count = cfg["data"]["candle_count"]

    for instrument in instruments:
        for granularity in granularities:
            url = f"{base}/v3/instruments/{instrument}/candles?count={count}&granularity={granularity}"
            headers = {"Authorization": f"Bearer {API_KEY}"}
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            data = r.json()["candles"]

            df = candles_to_df(data)
            df = build_all_features(df, config=feature_cfg)

            upload_dataframe(df, instrument=instrument, granularity=granularity, chunk_size=200)

    print("Upload completed.")
