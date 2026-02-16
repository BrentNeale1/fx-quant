# src/test_connection.py
import os
from pathlib import Path
from dotenv import load_dotenv
import requests
import sys

# load .env from project-root/config/.env
env_path = Path(__file__).resolve().parents[1] / "config" / ".env"
if not env_path.exists():
    print("Missing config/.env — create it with OANDA_API_KEY and OANDA_ACCOUNT_ID")
    sys.exit(1)

load_dotenv(env_path)

API_KEY = os.getenv("OANDA_API_KEY")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
ENV = os.getenv("OANDA_ENV", "practice").strip()

if not API_KEY or not ACCOUNT_ID:
    print("Missing OANDA_API_KEY or OANDA_ACCOUNT_ID in config/.env")
    sys.exit(1)

# choose base URL for practice or live
base = "https://api-fxpractice.oanda.com" if ENV == "practice" else "https://api-fxtrade.oanda.com"
url = f"{base}/v3/accounts/{ACCOUNT_ID}/summary"

headers = {"Authorization": f"Bearer {API_KEY}"}

print("Requesting:", url)   # debug only; does not show token
r = requests.get(url, headers=headers)

print("Status:", r.status_code)
try:
    print(r.json())
except Exception:
    print(r.text)
