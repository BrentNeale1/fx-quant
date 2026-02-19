# src/dashboard.py
"""
Web dashboard for fx-quant trading bot.
Provides config editing, status monitoring, log viewing, and kill switch control.
"""

import os
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from functools import wraps

import yaml
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify

app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent.parent / "templates"))
app.secret_key = os.urandom(24)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = APP_ROOT / "config" / "system.yaml"
ENV_PATH = APP_ROOT / "config" / ".env"
RELOAD_SIGNAL = APP_ROOT / "RELOAD_CONFIG"
KILL_SWITCH_FILE = APP_ROOT / "STOP_ALL_TRADING"
ORDER_LOG = APP_ROOT / "logs" / "order_log.csv"
AI_LOG = APP_ROOT / "logs" / "ai_decisions.csv"

# When running in Docker, paths are under /app
if Path("/app/config/system.yaml").exists():
    CONFIG_PATH = Path("/app/config/system.yaml")
    ENV_PATH = Path("/app/config/.env")
    RELOAD_SIGNAL = Path("/app/RELOAD_CONFIG")
    KILL_SWITCH_FILE = Path("/app/STOP_ALL_TRADING")
    ORDER_LOG = Path("/app/logs/order_log.csv")
    AI_LOG = Path("/app/logs/ai_decisions.csv")

# Whitelisted OANDA instruments for validation
VALID_INSTRUMENTS = [
    "EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", "AUD_USD",
    "USD_CAD", "NZD_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY",
    "EUR_CHF", "AUD_JPY", "CHF_JPY", "EUR_AUD", "EUR_CAD",
    "EUR_NZD", "GBP_AUD", "GBP_CAD", "GBP_CHF", "GBP_NZD",
    "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_CHF", "CAD_JPY",
    "NZD_CAD", "NZD_CHF", "NZD_JPY",
]

VALID_GRANULARITIES = ["S5", "S10", "S15", "S30", "M1", "M2", "M4", "M5",
                       "M10", "M15", "M30", "H1", "H2", "H3", "H4", "H6",
                       "H8", "H12", "D", "W", "M"]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def check_auth(username, password):
    dashboard_pw = os.environ.get("DASHBOARD_PASSWORD", "changeme")
    return username == "admin" and password == dashboard_pw


def authenticate():
    return Response(
        "Login required.", 401,
        {"WWW-Authenticate": 'Basic realm="fx-quant dashboard"'},
    )


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg):
    """Save config with backup."""
    backup = CONFIG_PATH.with_suffix(".yaml.backup")
    shutil.copy2(CONFIG_PATH, backup)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    # Signal bot to reload
    RELOAD_SIGNAL.touch()


def read_csv_tail(csv_path, max_rows=50):
    """Read last N rows from a CSV file, return (headers, rows)."""
    if not csv_path.exists():
        return [], []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], []
    headers = rows[0]
    data = rows[1:]
    return headers, data[-max_rows:]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_config(cfg):
    """Validate config values. Returns list of error strings."""
    errors = []

    # Instruments
    instruments = cfg.get("brokers", [{}])[0].get("instruments", [])
    for inst in instruments:
        if inst not in VALID_INSTRUMENTS:
            errors.append(f"Invalid instrument: {inst}")

    # Granularities
    grans = cfg.get("data", {}).get("candle_granularities", [])
    for g in grans:
        if g not in VALID_GRANULARITIES:
            errors.append(f"Invalid granularity: {g}")

    # Numeric ranges
    strategy = cfg.get("strategy", {})
    params = strategy.get("params", {})
    if params.get("short", 1) < 1:
        errors.append("SMA short period must be >= 1")
    if params.get("long", 2) < 2:
        errors.append("SMA long period must be >= 2")
    if params.get("short", 1) >= params.get("long", 2):
        errors.append("SMA short period must be less than long period")

    trade_size = strategy.get("trade_size_pct_of_equity", 0.01)
    if not (0.001 <= trade_size <= 0.1):
        errors.append("trade_size_pct_of_equity must be between 0.001 and 0.1")

    execution = cfg.get("execution", {})
    if execution.get("max_positions", 1) < 1:
        errors.append("max_positions must be >= 1")
    if execution.get("interval_seconds", 10) < 10:
        errors.append("interval_seconds must be >= 10")

    ai = cfg.get("ai", {})
    threshold = ai.get("confidence_threshold", 0.5)
    if not (0.0 <= threshold <= 1.0):
        errors.append("confidence_threshold must be between 0.0 and 1.0")

    return errors


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/")
@requires_auth
def index():
    cfg = load_config()
    kill_active = KILL_SWITCH_FILE.exists()

    # Read recent orders for activity feed
    _, recent_orders = read_csv_tail(ORDER_LOG, max_rows=10)

    return render_template("index.html",
                           cfg=cfg,
                           kill_active=kill_active,
                           recent_orders=recent_orders)


@app.route("/config", methods=["GET", "POST"])
@requires_auth
def config_editor():
    cfg = load_config()

    if request.method == "POST":
        # Parse form into config structure
        try:
            # Instruments
            instruments_raw = request.form.get("instruments", "").strip()
            instruments = [i.strip() for i in instruments_raw.split(",") if i.strip()]
            cfg["brokers"][0]["instruments"] = instruments

            # Granularities
            grans_raw = request.form.get("granularities", "").strip()
            grans = [g.strip() for g in grans_raw.split(",") if g.strip()]
            cfg["data"]["candle_granularities"] = grans

            # Candle count
            cfg["data"]["candle_count"] = int(request.form.get("candle_count", 200))

            # Features
            cfg["features"]["sma_windows"] = _parse_int_list(request.form.get("sma_windows", "3,20"))
            cfg["features"]["ema_windows"] = _parse_int_list(request.form.get("ema_windows", "20"))
            cfg["features"]["rsi_period"] = int(request.form.get("rsi_period", 14))
            cfg["features"]["atr_period"] = int(request.form.get("atr_period", 14))
            cfg["features"]["vwap_window"] = int(request.form.get("vwap_window", 20))
            cfg["features"]["volatility_window"] = int(request.form.get("volatility_window", 20))

            # Strategy
            cfg["strategy"]["rule"] = request.form.get("strategy_rule", "sma_cross")
            cfg["strategy"]["params"]["short"] = int(request.form.get("sma_short", 3))
            cfg["strategy"]["params"]["long"] = int(request.form.get("sma_long", 20))
            cfg["strategy"]["trade_size_pct_of_equity"] = float(request.form.get("trade_size_pct", 0.01))
            cfg["strategy"]["max_drawdown_pct"] = float(request.form.get("max_drawdown_pct", 0.05))

            # AI
            cfg["ai"]["confidence_threshold"] = float(request.form.get("confidence_threshold", 0.85))
            cfg["ai"]["ensemble_models"] = [m.strip() for m in request.form.get("ensemble_models", "").split(",") if m.strip()]
            cfg["ai"]["sanity_checks"]["rsi_overbought"] = int(request.form.get("rsi_overbought", 80))
            cfg["ai"]["sanity_checks"]["rsi_oversold"] = int(request.form.get("rsi_oversold", 20))
            cfg["ai"]["sanity_checks"]["volatility_multiplier"] = float(request.form.get("volatility_multiplier", 3.0))

            # Execution
            cfg["execution"]["paper_mode"] = request.form.get("paper_mode") == "on"
            cfg["execution"]["canary_size_pct"] = float(request.form.get("canary_size_pct", 0.01))
            cfg["execution"]["max_positions"] = int(request.form.get("max_positions", 3))
            cfg["execution"]["interval_seconds"] = int(request.form.get("interval_seconds", 60))

            # Validate
            errors = validate_config(cfg)
            if errors:
                for e in errors:
                    flash(e, "danger")
                return render_template("config.html", cfg=cfg,
                                       valid_instruments=VALID_INSTRUMENTS,
                                       valid_granularities=VALID_GRANULARITIES)

            save_config(cfg)
            flash("Config saved. Bot will reload on next loop iteration.", "success")
            return redirect(url_for("config_editor"))

        except (ValueError, KeyError) as e:
            flash(f"Invalid input: {e}", "danger")
            return render_template("config.html", cfg=cfg,
                                   valid_instruments=VALID_INSTRUMENTS,
                                   valid_granularities=VALID_GRANULARITIES)

    return render_template("config.html", cfg=cfg,
                           valid_instruments=VALID_INSTRUMENTS,
                           valid_granularities=VALID_GRANULARITIES)


@app.route("/logs")
@requires_auth
def logs():
    order_headers, order_rows = read_csv_tail(ORDER_LOG, max_rows=100)
    ai_headers, ai_rows = read_csv_tail(AI_LOG, max_rows=100)
    # Reverse so newest first
    order_rows = list(reversed(order_rows))
    ai_rows = list(reversed(ai_rows))
    return render_template("logs.html",
                           order_headers=order_headers,
                           order_rows=order_rows,
                           ai_headers=ai_headers,
                           ai_rows=ai_rows)


@app.route("/backtest")
@requires_auth
def backtest():
    """Show backtest results from saved JSON summaries."""
    logs_dir = APP_ROOT / "logs"
    summaries = []
    for f in sorted(logs_dir.glob("backtest_summary_*.json")):
        with open(f, "r") as fh:
            summaries.append(json.load(fh))
    return render_template("backtest.html", summaries=summaries)


@app.route("/killswitch", methods=["POST"])
@requires_auth
def killswitch():
    action = request.form.get("action")
    if action == "activate":
        KILL_SWITCH_FILE.touch()
        flash("Kill switch ACTIVATED. All trading halted.", "warning")
    elif action == "deactivate":
        if KILL_SWITCH_FILE.exists():
            KILL_SWITCH_FILE.unlink()
        flash("Kill switch deactivated. Trading will resume on next loop.", "success")
    return redirect(url_for("index"))


@app.route("/backtest-chart")
@requires_auth
def backtest_chart():
    """Backtest trade visualization chart — scans available trade CSVs."""
    results_dir = APP_ROOT / "results" / "phase1"
    combos = []
    if results_dir.exists():
        for f in sorted(results_dir.glob("*_trades.csv")):
            # e.g. S1_GBP_AUD_trades.csv -> strategy=S1, pair=GBP_AUD
            parts = f.stem.replace("_trades", "").split("_", 1)
            if len(parts) == 2:
                combos.append({"strategy": parts[0], "pair": parts[1],
                               "label": f"{parts[0]} / {parts[1].replace('_', '/')}"})
    return render_template("backtest_chart.html", combos=combos)


@app.route("/api/backtest-chart-data")
@requires_auth
def api_backtest_chart_data():
    """Return OHLC + indicators + trades as JSON for the backtest chart."""
    strategy = request.args.get("strategy", "")
    pair = request.args.get("pair", "")
    timeframe = request.args.get("timeframe", "M15")
    start = request.args.get("start", "")
    end = request.args.get("end", "")

    # Validate
    if not strategy or not pair:
        return jsonify({"error": "strategy and pair are required"}), 400

    # Load OHLC
    ohlc_path = APP_ROOT / "data" / "processed" / f"{pair}_{timeframe}.csv"
    if not ohlc_path.exists():
        return jsonify({"error": f"OHLC file not found: {pair}_{timeframe}.csv"}), 404

    df = pd.read_csv(ohlc_path, parse_dates=["timestamp"], index_col="timestamp")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)

    # Date filter
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]

    if df.empty:
        return jsonify({"error": "No OHLC data in selected range"}), 404

    # Compute indicators
    from indicators.technical import compute_all_indicators, identify_key_levels
    df = compute_all_indicators(df)

    # Build OHLC list (round to 5 decimals)
    time_strings = df.index.strftime("%Y-%m-%dT%H:%M:%S").tolist()
    ohlc_data = []
    for i, (idx, row) in enumerate(df.iterrows()):
        ohlc_data.append({
            "time": time_strings[i],
            "open": round(row["open"], 5),
            "high": round(row["high"], 5),
            "low": round(row["low"], 5),
            "close": round(row["close"], 5),
        })

    # Build EMA indicator series
    indicators = {}
    for key in ["ema_50", "ema_100", "ema_200"]:
        if key in df.columns:
            series_data = []
            for i, (idx, row) in enumerate(df.iterrows()):
                val = row[key]
                if pd.notna(val):
                    series_data.append({"time": time_strings[i], "value": round(float(val), 5)})
            indicators[key] = series_data

    # Load trades
    trades_path = APP_ROOT / "results" / "phase1" / f"{strategy}_{pair}_trades.csv"
    trades = []
    if trades_path.exists():
        tdf = pd.read_csv(trades_path, parse_dates=["timestamp"])
        if "exit_time" in tdf.columns:
            tdf["exit_time"] = pd.to_datetime(tdf["exit_time"])
        # Apply date filter to trades
        if start:
            tdf = tdf[tdf["timestamp"] >= pd.Timestamp(start, tz="UTC")]
        if end:
            tdf = tdf[tdf["timestamp"] <= pd.Timestamp(end, tz="UTC")]

        for _, trow in tdf.iterrows():
            trade = {
                "timestamp": trow["timestamp"].strftime("%Y-%m-%dT%H:%M:%S"),
                "direction": trow.get("signal_direction", ""),
                "entry_price": round(float(trow.get("entry_price", 0)), 5),
                "sl_price": round(float(trow.get("sl_price", 0)), 5),
                "tp1_price": round(float(trow.get("tp1_price", 0)), 5),
                "tp2_price": round(float(trow.get("tp2_price", 0)), 5),
                "tp3_price": round(float(trow.get("tp3_price", 0)), 5),
                "exit_price": round(float(trow.get("exit_price", 0)), 5),
                "exit_reason": str(trow.get("exit_reason", "")),
                "pnl_pips": round(float(trow.get("pnl_pips", 0)), 1),
                "pnl_dollars": round(float(trow.get("pnl_dollars", 0)), 2),
                "hold_time_minutes": int(trow.get("hold_time_minutes", 0)),
                "confluence_score": int(trow.get("confluence_score", 0)),
                "session": str(trow.get("session", "")),
                "win": bool(trow.get("win", False)),
            }
            if pd.notna(trow.get("exit_time")):
                trade["exit_time"] = trow["exit_time"].strftime("%Y-%m-%dT%H:%M:%S")

            # Key S/R levels only for strategies that use them (S3, S5)
            trade["key_levels"] = []
            if strategy in ("S3", "S5"):
                entry_ts = trow["timestamp"]
                pre_entry = df[df.index <= entry_ts].tail(500)
                if len(pre_entry) >= 30:
                    levels = identify_key_levels(pre_entry, min_touches=2)
                    trade["key_levels"] = [
                        {"price": round(float(p), 5), "touches": int(t)}
                        for p, t in levels[:6]
                    ]

            # For all strategies: include the EMA values at entry as reference
            entry_ts = trow["timestamp"]
            entry_row = df[df.index <= entry_ts].iloc[-1] if len(df[df.index <= entry_ts]) > 0 else None
            if entry_row is not None:
                trade["ema_at_entry"] = {
                    "ema_50": round(float(entry_row.get("ema_50", 0)), 5),
                    "ema_100": round(float(entry_row.get("ema_100", 0)), 5),
                    "ema_200": round(float(entry_row.get("ema_200", 0)), 5),
                }

            trades.append(trade)

    return jsonify({"ohlc": ohlc_data, "indicators": indicators, "trades": trades})


@app.route("/chart")
@requires_auth
def chart():
    """Interactive TradingView-style candlestick chart with indicators."""
    instrument = request.args.get("instrument", "EUR_USD")
    granularity = request.args.get("granularity", "M15")

    # Validate inputs
    if instrument not in VALID_INSTRUMENTS:
        instrument = "EUR_USD"
    if granularity not in VALID_GRANULARITIES:
        granularity = "M15"

    ohlc_json = "[]"
    pivot_json = "{}"
    indicators_json = "{}"
    error_msg = None

    try:
        from supabase import create_client
        from dotenv import load_dotenv
        import numpy as np

        env_path = APP_ROOT / "config" / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")

        if url and key:
            sb = create_client(url, key)
            cfg = load_config()
            table = cfg.get("supabase", {}).get("table", "fx_candles")

            resp = (
                sb.table(table)
                .select("*")
                .eq("instrument", instrument)
                .eq("granularity", granularity)
                .order("time", desc=True)
                .limit(2000)
                .execute()
            )

            rows = resp.data if resp.data else []
            if rows:
                df = pd.DataFrame(rows)
                df["time"] = pd.to_datetime(df["time"])
                df = df.sort_values("time").set_index("time")
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col].astype(float)
                df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(int)

                # Ensure stored indicators are numeric
                for col in ["sma_3", "sma_20", "ema_20", "rsi_14", "atr_14", "vwap_20"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                # Compute SMA 50, SMA 100 on the fly (not stored in DB)
                df["sma_50"] = df["close"].rolling(50).mean()
                df["sma_100"] = df["close"].rolling(100).mean()

                # Compute EMA 50, 100, 200 on the fly (only EMA 20 stored)
                df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
                df["ema_100"] = df["close"].ewm(span=100, adjust=False).mean()
                df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

                # Compute Session VWAP (resets each trading day)
                tp = (df["high"] + df["low"] + df["close"]) / 3.0
                pv = tp * df["volume"]
                df["_date"] = df.index.date
                df["session_vwap"] = (
                    pv.groupby(df["_date"]).cumsum()
                    / df["volume"].groupby(df["_date"]).cumsum()
                )
                df["session_vwap"] = df["session_vwap"].replace([np.inf, -np.inf], np.nan)
                df.drop(columns=["_date"], inplace=True)

                # Compute pivot points
                from data_engine import add_pivot_points
                df = add_pivot_points(df)

                # Prepare OHLC JSON (with volume for client-side anchored VWAP)
                time_strings = df.index.strftime("%Y-%m-%dT%H:%M:%S").tolist()
                ohlc_data = []
                for i, (idx, row) in enumerate(df.iterrows()):
                    ohlc_data.append({
                        "time": time_strings[i],
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": int(row["volume"]),
                    })
                ohlc_json = json.dumps(ohlc_data)

                # Prepare pivot levels (latest non-null values)
                pivot_cols = ["pivot", "r1", "r2", "r3", "s1", "s2", "s3"]
                pivot_available = df[pivot_cols].dropna()
                if len(pivot_available) > 0:
                    pivot_json = pivot_available.iloc[-1].to_json()

                # Build indicators_json: { key: [{time, value}, ...] }
                indicator_cols = {
                    "sma_3": "sma_3", "sma_20": "sma_20",
                    "sma_50": "sma_50", "sma_100": "sma_100",
                    "ema_20": "ema_20", "ema_50": "ema_50",
                    "ema_100": "ema_100", "ema_200": "ema_200",
                    "rsi_14": "rsi_14", "atr_14": "atr_14",
                    "session_vwap": "session_vwap",
                }
                ind_dict = {}
                for key, col in indicator_cols.items():
                    if col in df.columns:
                        series_data = []
                        for i, (idx, row) in enumerate(df.iterrows()):
                            val = row[col]
                            if pd.notna(val):
                                series_data.append({"time": time_strings[i], "value": float(val)})
                        ind_dict[key] = series_data

                # Volume as indicator data
                vol_data = []
                for i, (idx, row) in enumerate(df.iterrows()):
                    color = "rgba(38,166,154,0.5)" if row["close"] >= row["open"] else "rgba(239,83,80,0.5)"
                    vol_data.append({
                        "time": time_strings[i],
                        "value": int(row["volume"]),
                        "color": color,
                    })
                ind_dict["volume"] = vol_data

                indicators_json = json.dumps(ind_dict)
        else:
            error_msg = "Supabase credentials not configured."
    except Exception as e:
        error_msg = f"Could not load candle data: {e}"

    return render_template("chart.html",
                           ohlc_json=ohlc_json,
                           pivot_json=pivot_json,
                           indicators_json=indicators_json,
                           instrument=instrument,
                           granularity=granularity,
                           instruments=VALID_INSTRUMENTS,
                           granularities=VALID_GRANULARITIES,
                           error_msg=error_msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_int_list(s):
    """Parse comma-separated string into list of ints."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
