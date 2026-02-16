# src/get_candles.py
"""
Fetch OANDA candles for every instrument × granularity defined in
config/system.yaml, build indicator features, and print the result.
"""

import os
import requests
from config_loader import load_config
from data_engine import candles_to_df, build_all_features


def fetch_candles(instrument, granularity, count, base_url, api_key):
    """Pull raw candle dicts from the OANDA v20 REST API."""
    url = (
        f"{base_url}/v3/instruments/{instrument}/candles"
        f"?count={count}&granularity={granularity}"
    )
    headers = {"Authorization": f"Bearer {api_key}"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()["candles"]


def main():
    cfg = load_config()

    # OANDA credentials (loaded into env by config_loader)
    api_key = os.getenv("OANDA_API_KEY")
    env = os.getenv("OANDA_ENV", "practice")
    base_url = (
        "https://api-fxpractice.oanda.com"
        if env == "practice"
        else "https://api-fxtrade.oanda.com"
    )

    # Config-driven parameters
    broker = cfg["brokers"][0]
    instruments = broker["instruments"]
    granularities = cfg["data"]["candle_granularities"]
    count = cfg["data"]["candle_count"]
    feature_cfg = cfg.get("features", {})

    for instrument in instruments:
        for granularity in granularities:
            print(f"\n--- {instrument} | {granularity} ---")
            candles = fetch_candles(
                instrument, granularity, count, base_url, api_key
            )
            df = candles_to_df(candles)
            df = build_all_features(df, config=feature_cfg)
            df["granularity"] = granularity
            print(df.tail(10).to_string())


if __name__ == "__main__":
    main()
