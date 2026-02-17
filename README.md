# fx-quant

Algorithmic FX trading system with backtesting, paper/live execution via OANDA, and a Flask web dashboard.

## Quick Start (New Machine Setup)

### 1. Clone & install dependencies

```bash
git clone <your-repo-url>
cd fx-quant
python -m pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the template and fill in your credentials:

```bash
cp config/.env.example config/.env
```

Required variables in `config/.env`:

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_KEY` | Supabase anon/service key |
| `OANDA_API_KEY` | OANDA v20 API token |
| `OANDA_ACCOUNT_ID` | OANDA account ID |
| `OANDA_ENV` | `practice` or `live` (default: practice) |
| `DASHBOARD_PASSWORD` | Password for web dashboard login |
| `TRADING_ECONOMICS_API_KEY` | Trading Economics API key (for economic calendar) |

### 3. Verify connectivity

```bash
python src/test_connection.py
```

### 4. Load historical data (if Supabase table is empty)

```bash
python src/historical_loader.py
```

### 5. Run the backtester

```bash
python src/backtester.py
```

Output goes to `logs/`:
- `backtest_trades_<INSTRUMENT>_<GRANULARITY>.csv` -- trade log
- `backtest_summary_<INSTRUMENT>_<GRANULARITY>.json` -- metrics + monthly P&L

### 6. Visualize trades on a chart

```bash
python src/chart_trades.py --start 2025-02-24 --end 2025-02-27
python src/chart_trades.py --granularity M15 --start 2025-04-01 --end 2025-04-15
```

Saves PNG charts to `logs/chart_trades_*.png`.

---

## Project Structure

```
fx-quant/
  config/
    system.yaml          # Main configuration (strategy, features, execution)
    .env                 # Secrets (not committed)
  src/
    config_loader.py     # Loads system.yaml + .env
    data_engine.py       # Feature engineering (SMA, EMA, RSI, ATR, VWAP, pivots, engulfing)
    backtester.py        # Backtesting engine (SMA cross + pivot retest strategies)
    order_executor.py    # Paper/live order execution via OANDA
    ai_wrapper.py        # ML ensemble (logistic reg, RF, gradient boosting) signal validation
    dashboard.py         # Flask web dashboard
    chart_trades.py      # Matplotlib/mplfinance trade visualization
    get_candles.py       # Fetch candles from OANDA API
    economic_calendar.py # Economic calendar fetch/upload pipeline
    historical_loader.py # Bulk historical data loader
    supabase_upload.py   # Upload candle data to Supabase
    param_sweep.py       # Strategy parameter optimization
    test_connection.py   # OANDA + Supabase connectivity check
  templates/             # Flask HTML templates (chart, backtest, config, logs)
  logs/                  # Output: trade CSVs, JSON summaries, chart PNGs
  models/                # Saved ML models
  sql/                   # Database schemas
  Dockerfile             # Bot container
  docker-compose.yml     # Bot + dashboard services
  requirements.txt       # Python dependencies (portable)
  requirements.docker.txt# Docker-specific deps
```

## Strategies

### 1. SMA Cross (original)

Long-only strategy. Goes long when short SMA > long SMA, flat otherwise.

```yaml
strategy:
  rule: sma_cross
  params:
    short: 50
    long: 100
```

### 2. Pivot Retest + Engulfing (current)

Long/short strategy with dual take-profit and ATR-based stop loss.

**Entry conditions (all must be true):**
- Price retests a pivot level (broke through, then returned within ATR tolerance)
- SMA 50 aligns with trade direction relative to the pivot level
- Engulfing candle pattern confirmed
- Strong close (in top/bottom 30% of candle range)

**Position management:**
- SL: 1.5x ATR from entry
- TP1: next pivot level in trade direction (close 50%, move SL to breakeven)
- TP2: pivot level after TP1 (close remaining 50%)

```yaml
strategy:
  rule: pivot_retest_engulfing
  params:
    sma_period: 50
    lookback_bars: 20
    retest_tolerance_atr: 0.5
    strong_close_pct: 0.30
    sl_atr_multiplier: 1.5
```

To switch strategies, edit `config/system.yaml` and change `strategy.rule`.

## Configuration Reference

All settings live in `config/system.yaml`:

| Section | Key settings |
|---------|-------------|
| `brokers[0].instruments` | Currency pairs to trade (e.g. `EUR_USD`) |
| `data.candle_granularities` | Timeframes (`M5`, `M15`, `H1`, etc.) |
| `features.*` | Indicator windows (SMA, EMA, RSI, ATR, VWAP, volatility) |
| `strategy.*` | Active strategy rule + parameters |
| `ai.*` | ML ensemble config, confidence threshold, sanity checks |
| `execution.paper_mode` | `true` for paper trading, `false` for live |
| `execution.interval_seconds` | Bot loop interval |
| `execution.max_positions` | Max concurrent open trades |

## Web Dashboard

Flask-based UI for remote management.

```bash
# Docker
docker-compose build && docker-compose up -d

# Or run directly
python src/dashboard.py
```

Access via SSH tunnel: `ssh -L 5000:localhost:5000 your-server`, then open http://localhost:5000.

Pages: Status (`/`), Backtest (`/backtest`), Chart (`/chart`), Config (`/config`), Logs (`/logs`)

## Kill Switch

Create `STOP_ALL_TRADING` in the project root to halt all trading immediately. The bot checks for this file every loop iteration. In live mode, it also closes all open trades. Toggle via the dashboard or manually:

```bash
touch STOP_ALL_TRADING   # activate
rm STOP_ALL_TRADING      # deactivate
```

## Economic Calendar

Filters trade entries near high-impact economic events (NFP, CPI, rate decisions, etc.) to reduce slippage and false signals. Uses the [Trading Economics API](https://tradingeconomics.com/api/).

**Setup:**
1. Subscribe to a Trading Economics API plan that includes Calendar data ([pricing](https://tradingeconomics.com/api/pricing.aspx))
2. Add your API key to `config/.env`: `TRADING_ECONOMICS_API_KEY=your-key-here`
3. Create the Supabase table: run `sql/create_economic_calendar.sql` in the SQL Editor
4. Fetch events: `python src/economic_calendar.py` (full load) or `python src/economic_calendar.py --incremental`

**Config** (`config/system.yaml` → `economic_calendar`):

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Toggle calendar filter on/off |
| `impact_threshold` | `High` | Minimum impact level to block entries (`Low`, `Medium`, `High`) |
| `event_buffer_minutes` | `30` | Block entries within this many minutes of an event |
| `days_back` | `400` | How far back to fetch on full load |

**Status:** Table created in Supabase. Awaiting Trading Economics API key (subscription request pending). Once the key is obtained, run the loader and the backtester will automatically filter entries near high-impact events.

## Running the Bot

```bash
# Single execution
python src/order_executor.py --once

# Continuous loop (default 60s interval)
python src/order_executor.py

# Docker
docker-compose up -d
```
