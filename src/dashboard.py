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
from flask import Flask, render_template, request, redirect, url_for, flash, Response

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
    app.run(host="0.0.0.0", port=5000, debug=False)
