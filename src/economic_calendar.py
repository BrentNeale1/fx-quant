# src/economic_calendar.py
"""
Economic calendar pipeline using Trading Economics API.
Fetches economic events, uploads to Supabase, and provides query helpers
for the backtester and data engine.

Usage:
    python src/economic_calendar.py              # full load (~400 days)
    python src/economic_calendar.py --incremental  # fetch only recent missing data
"""

import os
import sys
import math
import time as time_mod
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
import tradingeconomics as te
from supabase import create_client

from config_loader import load_config, get_project_root

cfg = load_config()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TE_API_KEY = os.getenv("TRADING_ECONOMICS_API_KEY")

IMPACT_MAP = {"Low": 1, "Medium": 2, "High": 3}


# ---------------------------------------------------------------------------
# Country / currency helpers
# ---------------------------------------------------------------------------

def get_countries_for_instruments(instruments, currency_country_map):
    """
    Derive the set of countries to fetch from configured instrument pairs.
    E.g. ['EUR_USD'] -> {'euro area', 'united states'}
    """
    currencies = set()
    for inst in instruments:
        parts = inst.split("_")
        currencies.update(parts)

    countries = set()
    for ccy in currencies:
        country = currency_country_map.get(ccy)
        if country:
            countries.add(country)
        else:
            print(f"  Warning: no country mapping for currency {ccy}")
    return countries


# ---------------------------------------------------------------------------
# Fetch from Trading Economics
# ---------------------------------------------------------------------------

def fetch_calendar_chunk(country, init_date, end_date):
    """
    Fetch calendar events for one country and one date window.
    Returns a list of dicts from the TE API.
    """
    try:
        data = te.getCalendarData(
            country=country,
            initDate=init_date,
            endDate=end_date,
            output_type="df",
        )
        if isinstance(data, pd.DataFrame) and not data.empty:
            return data.to_dict(orient="records")
        return []
    except Exception as e:
        print(f"  Error fetching {country} ({init_date} -> {end_date}): {e}")
        return []


def fetch_all_calendar_data(countries, days_back=400, chunk_days=28,
                            sleep_between_chunks=1.0):
    """
    Paginate through Trading Economics calendar in chunk_days windows.
    Returns a deduplicated DataFrame of all events.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)

    all_records = []
    for country in sorted(countries):
        cursor = start
        chunk_num = 0
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=chunk_days), end)
            chunk_num += 1
            init_str = cursor.strftime("%Y-%m-%d")
            end_str = chunk_end.strftime("%Y-%m-%d")
            print(f"  {country}: chunk {chunk_num} ({init_str} -> {end_str}) ...", end=" ")

            records = fetch_calendar_chunk(country, init_str, end_str)
            print(f"{len(records)} events")
            all_records.extend(records)

            cursor = chunk_end
            time_mod.sleep(sleep_between_chunks)

    if not all_records:
        print("  No calendar events fetched.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    print(f"  Total raw events: {len(df)}")

    # Normalize column names (TE API returns mixed-case)
    col_map = {}
    for col in df.columns:
        col_map[col] = col.lower().replace(" ", "_")
    df = df.rename(columns=col_map)

    # Parse datetime
    date_col = None
    for candidate in ["date", "datetime", "event_datetime"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        print("  Warning: no date column found in TE response.")
        return pd.DataFrame()

    df["event_datetime"] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.dropna(subset=["event_datetime"])

    # Map category/event name
    if "category" in df.columns and "event" not in df.columns:
        df["event"] = df["category"]
    event_col = "event" if "event" in df.columns else "category"

    # Map impact to numeric
    if "importance" in df.columns and "impact" not in df.columns:
        # TE uses 'importance' with values like 1,2,3 or Low/Medium/High
        importance = df["importance"]
        if importance.dtype in (int, float, np.int64, np.float64):
            df["impact_numeric"] = importance.astype(int)
            reverse_map = {1: "Low", 2: "Medium", 3: "High"}
            df["impact"] = df["impact_numeric"].map(reverse_map).fillna("Low")
        else:
            df["impact"] = importance.astype(str)
            df["impact_numeric"] = df["impact"].map(IMPACT_MAP).fillna(1).astype(int)
    elif "impact" in df.columns:
        df["impact_numeric"] = df["impact"].map(IMPACT_MAP).fillna(1).astype(int)
    else:
        df["impact"] = "Low"
        df["impact_numeric"] = 1

    # Standardize column names for our schema
    rename = {}
    if event_col != "event_name":
        rename[event_col] = "event_name"
    for te_col, our_col in [("actual", "actual"), ("forecast", "forecast"),
                             ("previous", "previous"), ("revised", "revised"),
                             ("teforecast", "forecast"), ("teprevious", "previous"),
                             ("reference", "reference"), ("source", "source"),
                             ("currency", "currency"), ("country", "country")]:
        if te_col in df.columns and te_col != our_col:
            rename[te_col] = our_col
    df = df.rename(columns=rename)

    # Ensure required columns exist
    for col in ["country", "currency", "event_name", "actual", "forecast",
                "previous", "revised", "reference", "source"]:
        if col not in df.columns:
            df[col] = None

    # Select only the columns we need
    keep = ["event_datetime", "country", "currency", "event_name",
            "impact", "impact_numeric", "actual", "forecast",
            "previous", "revised", "reference", "source"]
    df = df[[c for c in keep if c in df.columns]]

    # Deduplicate on composite key
    df = df.drop_duplicates(subset=["event_datetime", "country", "event_name"], keep="last")
    df = df.sort_values("event_datetime").reset_index(drop=True)
    print(f"  After dedup: {len(df)} events")

    return df


# ---------------------------------------------------------------------------
# Supabase upload
# ---------------------------------------------------------------------------

def calendar_df_to_records(df):
    """
    Clean a calendar DataFrame for Supabase upsert.
    Mirrors supabase_upload.df_to_records sanitization patterns.
    """
    df2 = df.copy()

    # Convert timestamps to ISO strings
    df2["event_datetime"] = df2["event_datetime"].apply(
        lambda t: pd.to_datetime(t).isoformat() if pd.notna(t) else None
    )
    df2["last_updated"] = datetime.now(timezone.utc).isoformat()

    # Replace inf/-inf with NaN
    df2 = df2.replace([np.inf, -np.inf], np.nan)

    def _convert(v):
        if pd.isna(v):
            return None
        if isinstance(v, (bool,)):
            return bool(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            fv = float(v)
            return fv if math.isfinite(fv) else None
        if isinstance(v, str):
            return v
        try:
            return str(v)
        except Exception:
            return None

    for col in df2.columns:
        df2[col] = df2[col].apply(_convert)

    records = df2.to_dict(orient="records")

    # Final NaN/inf sweep
    def _sanitize(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    records = [{k: _sanitize(v) for k, v in row.items()} for row in records]
    return records


def upload_calendar(df, supabase_client, table="economic_calendar", chunk_size=500):
    """
    Upsert calendar records to Supabase in batches.
    """
    records = calendar_df_to_records(df)
    total = len(records)
    if total == 0:
        print("  No calendar rows to upload.")
        return

    chunks = math.ceil(total / chunk_size)
    print(f"  Uploading {total} calendar rows in {chunks} chunk(s)...")

    for i in range(chunks):
        start = i * chunk_size
        end = start + chunk_size
        batch = records[start:end]

        resp = supabase_client.table(table).upsert(batch).execute()
        if hasattr(resp, "data") and resp.data is not None:
            pass  # success
        elif getattr(resp, "status_code", None) not in (200, 201, None):
            print(f"  Upload chunk error: {resp}")
            raise SystemExit("Calendar upsert failed")
        print(f"  Chunk {i + 1}/{chunks} uploaded ({len(batch)} rows).")


# ---------------------------------------------------------------------------
# Supabase read (for backtester + data_engine)
# ---------------------------------------------------------------------------

def fetch_calendar_from_supabase(supabase_client, table="economic_calendar",
                                 currencies=None, impact_min=None,
                                 start_time=None, end_time=None):
    """
    Read calendar events from Supabase with optional filters.
    Returns a DataFrame sorted by event_datetime.
    """
    all_rows = []
    page_size = 1000
    offset = 0

    while True:
        query = supabase_client.table(table).select("*")

        if currencies:
            # Filter by currency (case-insensitive via ilike would be ideal,
            # but supabase-py .in_ works for exact match)
            query = query.in_("currency", list(currencies))
        if impact_min is not None:
            query = query.gte("impact_numeric", impact_min)
        if start_time:
            query = query.gte("event_datetime", start_time)
        if end_time:
            query = query.lte("event_datetime", end_time)

        query = query.order("event_datetime", desc=False)
        query = query.range(offset, offset + page_size - 1)
        resp = query.execute()

        rows = resp.data or []
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["event_datetime"] = pd.to_datetime(df["event_datetime"], utc=True)
    df = df.sort_values("event_datetime").reset_index(drop=True)

    # Convert numeric columns
    for col in ["impact_numeric", "actual", "forecast", "previous", "revised"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    if not TE_API_KEY:
        raise SystemExit("Missing TRADING_ECONOMICS_API_KEY in config/.env")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_KEY in config/.env")

    # Authenticate with Trading Economics
    te.login(TE_API_KEY)

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    cal_cfg = cfg.get("economic_calendar", {})
    table = cal_cfg.get("table", "economic_calendar")
    days_back = cal_cfg.get("days_back", 400)
    chunk_days = cal_cfg.get("chunk_days", 28)
    sleep_s = cal_cfg.get("sleep_between_chunks", 1.0)
    currency_country_map = cal_cfg.get("currency_country_map", {})

    instruments = cfg["brokers"][0]["instruments"]
    countries = get_countries_for_instruments(instruments, currency_country_map)

    if not countries:
        raise SystemExit("No countries derived from instruments. Check currency_country_map.")

    incremental = "--incremental" in sys.argv

    if incremental:
        # Fetch only the last 30 days
        days_back = 30
        print(f"Incremental mode: fetching last {days_back} days")
    else:
        print(f"Full load: fetching last {days_back} days")

    print(f"Countries: {sorted(countries)}")
    print(f"Instruments: {instruments}\n")

    df = fetch_all_calendar_data(
        countries=countries,
        days_back=days_back,
        chunk_days=chunk_days,
        sleep_between_chunks=sleep_s,
    )

    if df.empty:
        print("No events to upload.")
        return

    print(f"\nUploading to Supabase table '{table}'...")
    upload_calendar(df, sb, table=table)
    print("Economic calendar load complete.")


if __name__ == "__main__":
    main()
