"""
Phase 2 — Kelly Criterion Sizing (Step 9).

For each Phase 2 strategy, compute:
  - Kelly optimal fraction: f* = (W * R - L) / R
  - Half-Kelly as practical recommendation
  - Monte Carlo simulation (1000 paths) to estimate drawdown distribution

Output: results/phase2/kelly_sizing.json
"""
import os, sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from src.indicators.technical import compute_all_indicators
from src.backtester.engine import Backtester

# Strategy imports
from src.strategies_pkg.s7_liquidity_sweep import S7_Liquidity_Sweep
from src.strategies_pkg.s9_london_session import S9_London_Session
from src.strategies_pkg.s4f_ema_ribbon import S4F_EMA_Ribbon
from src.strategies_pkg.s3_key_level_breakout import S3_KeyLevel_Breakout

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase2")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIGS = [
    {"name": "S7_Tight", "pair": "GBP_JPY", "tf": "H1",
     "factory": lambda: S7_Liquidity_Sweep()},
    {"name": "S9", "pair": "GBP_USD", "tf": "H1",
     "factory": lambda: S9_London_Session()},
    {"name": "S9_Filtered", "pair": "GBP_AUD", "tf": "H1",
     "factory": lambda: S9_London_Session(pair="GBP_AUD", filtered=True)},
    {"name": "S4F", "pair": "EUR_AUD", "tf": "M15",
     "factory": lambda: S4F_EMA_Ribbon()},
    {"name": "S3", "pair": "GBP_JPY", "tf": "H1",
     "factory": lambda: S3_KeyLevel_Breakout()},
]

N_SIMULATIONS = 1000
STARTING_EQUITY = 100_000.0


def load_data(pair, tf):
    fp = os.path.join(PROCESSED_DIR, f"{pair}_{tf}.csv")
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    return compute_all_indicators(df)


def run_backtest(cfg):
    pair = cfg["pair"]
    tf = cfg["tf"]
    data = load_data(pair, tf)
    if data is None:
        return None

    htf_data = data.copy() if tf == "H1" else load_data(pair, "H1")
    strategy = cfg["factory"]()
    bt = Backtester(data=data, strategy=strategy, pair=pair,
                    starting_equity=STARTING_EQUITY, htf_data=htf_data)
    bt.run()
    return bt.get_trade_log_df()


def compute_kelly(trade_log):
    """Compute Kelly criterion from trade log.

    f* = (W * R - L) / R
    where:
      W = win rate (probability of winning)
      L = loss rate (1 - W)
      R = avg_win / avg_loss (win/loss ratio)

    Returns dict with kelly_full, kelly_half, and component values.
    """
    if trade_log.empty or len(trade_log) < 10:
        return {
            "trades": len(trade_log) if trade_log is not None else 0,
            "kelly_full_pct": 0, "kelly_half_pct": 0,
            "win_rate": 0, "avg_win_loss_ratio": 0,
        }

    wins = trade_log[trade_log["win"] == True]
    losses = trade_log[trade_log["win"] == False]

    n = len(trade_log)
    w = len(wins) / n  # win probability
    l = 1 - w          # loss probability

    avg_win = wins["pnl_pips"].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses["pnl_pips"].mean()) if len(losses) > 0 else 1

    r = avg_win / avg_loss if avg_loss > 0 else 0

    # Kelly formula: f* = (W * R - L) / R
    if r > 0:
        kelly = (w * r - l) / r
    else:
        kelly = 0

    kelly = max(kelly, 0)  # clamp at 0 (never bet negative)

    return {
        "trades": n,
        "win_rate": round(w * 100, 1),
        "avg_win_pips": round(avg_win, 1),
        "avg_loss_pips": round(avg_loss, 1),
        "avg_win_loss_ratio": round(r, 2),
        "kelly_full_pct": round(kelly * 100, 2),
        "kelly_half_pct": round(kelly * 50, 2),
    }


def monte_carlo_drawdown(trade_log, fraction: float,
                         n_sims: int = N_SIMULATIONS) -> dict:
    """Monte Carlo simulation to estimate drawdown distribution.

    Randomly resamples trade outcomes (with replacement) to build
    equity paths and measure max drawdown at the given Kelly fraction.

    Returns percentile drawdown estimates.
    """
    if trade_log.empty or len(trade_log) < 10 or fraction <= 0:
        return {"p50_dd": 0, "p75_dd": 0, "p95_dd": 0, "p99_dd": 0, "ruin_pct": 0}

    # Use pnl_pips as return per trade (normalized)
    returns = trade_log["pnl_pips"].values
    n_trades = len(returns)

    max_drawdowns = []
    ruin_count = 0

    rng = np.random.default_rng(42)

    for _ in range(n_sims):
        # Resample trades with replacement
        sampled = rng.choice(returns, size=n_trades, replace=True)

        # Scale by fraction (Kelly fraction applied to risk)
        equity = STARTING_EQUITY
        peak = equity
        max_dd = 0

        for ret in sampled:
            # PnL scaled by fraction (relative to 1% base risk)
            pnl = ret * fraction / 0.01 * equity / STARTING_EQUITY
            equity += pnl

            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd

            if equity <= 0:
                ruin_count += 1
                max_dd = 100
                break

        max_drawdowns.append(max_dd)

    dd_array = np.array(max_drawdowns)
    return {
        "p50_dd": round(np.percentile(dd_array, 50), 1),
        "p75_dd": round(np.percentile(dd_array, 75), 1),
        "p95_dd": round(np.percentile(dd_array, 95), 1),
        "p99_dd": round(np.percentile(dd_array, 99), 1),
        "ruin_pct": round(ruin_count / n_sims * 100, 2),
    }


def main():
    print(f"{'='*90}")
    print("PHASE 2 — KELLY CRITERION SIZING")
    print(f"{'='*90}")

    all_results = {}

    for cfg in CONFIGS:
        name = cfg["name"]
        pair = cfg["pair"]

        print(f"\n{'#'*60}")
        print(f"# {name} / {pair}")
        print(f"{'#'*60}")

        t0 = time.time()
        trade_log = run_backtest(cfg)
        elapsed = time.time() - t0

        if trade_log is None or trade_log.empty:
            print(f"  No trades. Skipping.")
            continue

        print(f"  Backtest: {len(trade_log)} trades ({elapsed:.0f}s)")

        # Kelly computation
        kelly = compute_kelly(trade_log)
        print(f"  Win rate: {kelly['win_rate']}%")
        print(f"  Avg win/loss ratio: {kelly['avg_win_loss_ratio']}")
        print(f"  Kelly full: {kelly['kelly_full_pct']:.1f}%")
        print(f"  Kelly half: {kelly['kelly_half_pct']:.1f}% (recommended)")

        # Monte Carlo at half-Kelly
        half_kelly_frac = kelly["kelly_half_pct"] / 100
        if half_kelly_frac > 0:
            print(f"\n  Monte Carlo ({N_SIMULATIONS} paths at half-Kelly={kelly['kelly_half_pct']:.1f}%):")
            mc = monte_carlo_drawdown(trade_log, half_kelly_frac)
            print(f"    Median max DD: {mc['p50_dd']:.1f}%")
            print(f"    75th pctl DD:  {mc['p75_dd']:.1f}%")
            print(f"    95th pctl DD:  {mc['p95_dd']:.1f}%")
            print(f"    99th pctl DD:  {mc['p99_dd']:.1f}%")
            print(f"    Ruin probability: {mc['ruin_pct']:.1f}%")
        else:
            mc = {"p50_dd": 0, "p75_dd": 0, "p95_dd": 0, "p99_dd": 0, "ruin_pct": 0}
            print(f"  Kelly <= 0, no MC simulation.")

        all_results[f"{name}_{pair}"] = {
            **kelly,
            "monte_carlo_half_kelly": mc,
        }

    # Summary table
    print(f"\n{'='*90}")
    print("KELLY SIZING SUMMARY")
    print(f"{'='*90}")
    print(f"{'Strategy':<20} {'Trades':>6} {'WR%':>6} {'W/L':>5} "
          f"{'Kelly%':>7} {'HalfK%':>7} {'MC-p95DD':>9}")
    print(f"{'-'*70}")

    for key, result in all_results.items():
        mc = result.get("monte_carlo_half_kelly", {})
        print(f"{key:<20} {result['trades']:>6} {result['win_rate']:>5.1f}% "
              f"{result['avg_win_loss_ratio']:>4.2f} "
              f"{result['kelly_full_pct']:>6.1f}% {result['kelly_half_pct']:>6.1f}% "
              f"{mc.get('p95_dd', 0):>8.1f}%")

    # Save
    out_path = os.path.join(RESULTS_DIR, "kelly_sizing.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
