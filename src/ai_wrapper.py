# src/ai_wrapper.py
"""
AI Decision Wrapper for fx-quant.
Validates trading signals using a local ML ensemble (scikit-learn + LightGBM)
before allowing order execution. Low-confidence or sanity-failed signals
are rejected and logged for human review.
"""

import csv
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from config_loader import load_config, get_project_root
from backtester import run_backtest, generate_signals

# Try to use LightGBM if available, otherwise fall back to sklearn GBM
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURE_COLS = ["sma_3", "sma_20", "rsi_14", "atr_14", "vol_20", "ret", "close"]


def build_feature_vector(df, idx=-1):
    """
    Extract model input features from the signal DataFrame at a given bar index.

    Features are normalized by dividing price-scale columns by close price.
    Returns a 1D numpy array ready for model prediction.
    """
    row = df.iloc[idx]
    close = row["close"]
    if close == 0:
        close = 1e-10  # avoid division by zero

    features = []
    for col in FEATURE_COLS:
        val = row.get(col, 0.0)
        if pd.isna(val):
            val = 0.0
        # Normalize price-scale features relative to close
        if col in ("sma_3", "sma_20", "atr_14"):
            val = val / close
        features.append(float(val))

    return pd.DataFrame([features], columns=FEATURE_COLS)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_context(df, instrument, supabase_client=None, cfg=None):
    """
    Assemble retrieval context: last 50 bars of features, recent backtest trades.
    Used for rationale generation in decision logging.

    Returns dict with context summary.
    """
    cfg = cfg or {}
    ai_cfg = cfg.get("ai", {})
    context = {
        "instrument": instrument,
        "bars_available": len(df),
        "last_close": float(df["close"].iloc[-1]) if len(df) > 0 else None,
    }

    # Recent feature summary (last 50 bars)
    tail = df.tail(50)
    for col in FEATURE_COLS:
        if col in tail.columns:
            context[f"{col}_mean"] = float(tail[col].mean()) if not tail[col].isna().all() else None

    # Optional: pull recent rows from Supabase for retrieval context
    if ai_cfg.get("retriever_enabled") and supabase_client is not None:
        try:
            table = cfg.get("supabase", {}).get("table", "fx_candles")
            resp = (
                supabase_client.table(table)
                .select("time,close,ret")
                .eq("instrument", instrument)
                .order("time", desc=True)
                .limit(10)
                .execute()
            )
            recent = resp.data or []
            context["recent_supabase_rows"] = len(recent)
        except Exception as e:
            context["recent_supabase_rows"] = 0
            context["retriever_error"] = str(e)

    return context


# ---------------------------------------------------------------------------
# Ensemble training
# ---------------------------------------------------------------------------

def _build_feature_matrix(df):
    """Build feature matrix from DataFrame. Returns (X, feature_names)."""
    close = df["close"].replace(0, 1e-10)
    X = pd.DataFrame(index=df.index)
    for col in FEATURE_COLS:
        if col not in df.columns:
            X[col] = 0.0
        elif col in ("sma_3", "sma_20", "atr_14"):
            X[col] = df[col] / close
        else:
            X[col] = df[col]
    X = X.fillna(0.0)
    return X, list(X.columns)


def train_ensemble(df, strategy_cfg):
    """
    Train 3 models on historical data:
      - LogisticRegression
      - RandomForestClassifier
      - GradientBoostingClassifier (or LGBMClassifier if available)

    Target: whether the signal led to a profitable next bar
    (next-bar return > 0 when signal == 1).

    Uses walk-forward split: train on first 70%, validate on last 30%.
    Returns (list of fitted model objects, validation metrics dict).
    """
    # Need signal column
    if "signal" not in df.columns:
        df = generate_signals(df, strategy_cfg)

    # Build target: next-bar return is positive AND we have a long signal
    next_ret = df["ret"].shift(-1)
    target = ((next_ret > 0) & (df["signal"] == 1)).astype(int)

    # Drop last row (no next-bar return) and any NaN rows
    valid_mask = next_ret.notna()
    df_valid = df[valid_mask]
    target = target[valid_mask]

    X, feature_names = _build_feature_matrix(df_valid)
    y = target.values

    if len(X) < 50:
        print(f"  AI: Not enough data to train ensemble ({len(X)} rows). Skipping.")
        return [], {"error": "insufficient_data", "rows": len(X)}

    # Walk-forward split: 70/30
    split_idx = int(len(X) * 0.7)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Train models
    models = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X_train, y_train)
        models.append(("logistic_regression", lr))

        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        models.append(("random_forest", rf))

        if HAS_LGBM:
            gbm = LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
        else:
            gbm = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        gbm.fit(X_train, y_train)
        models.append(("gradient_boosting", gbm))

    # Validation metrics
    val_scores = {}
    for name, model in models:
        score = model.score(X_val, y_val)
        val_scores[name] = round(score, 4)

    print(f"  AI: Ensemble trained — validation scores: {val_scores}")
    return models, val_scores


# ---------------------------------------------------------------------------
# Ensemble prediction
# ---------------------------------------------------------------------------

def ensemble_predict(models, feature_vector):
    """
    Each model predicts probability of profitable outcome.

    Returns (confidence, predictions, agreement):
      - confidence: mean probability across models
      - predictions: list of per-model probabilities
      - agreement: all models agree on direction (all > 0.5 or all < 0.5)
    """
    if not models:
        return 0.0, [], False

    predictions = []
    for name, model in models:
        proba = model.predict_proba(feature_vector)
        # probability of class 1 (profitable)
        p = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
        predictions.append(p)

    confidence = float(np.mean(predictions))
    # Agreement: all above 0.5 or all below 0.5
    all_bullish = all(p > 0.5 for p in predictions)
    all_bearish = all(p <= 0.5 for p in predictions)
    agreement = all_bullish or all_bearish

    return confidence, predictions, agreement


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def sanity_checks(df, instrument, cfg):
    """
    Run deterministic sanity checks on the latest bar:
      - RSI not in extreme territory against the signal
      - Volatility not excessively high (ATR > 3x rolling mean)
      - Price not gapping (|ret| > 5x vol)

    Returns (passed, reasons) where reasons lists any failed checks.
    """
    ai_cfg = cfg.get("ai", {})
    sc_cfg = ai_cfg.get("sanity_checks", {})
    rsi_ob = sc_cfg.get("rsi_overbought", 80)
    rsi_os = sc_cfg.get("rsi_oversold", 20)
    vol_mult = sc_cfg.get("volatility_multiplier", 3.0)

    reasons = []
    latest = df.iloc[-1]
    signal = int(latest.get("signal", 0))

    # RSI check
    rsi = latest.get("rsi_14", 50.0)
    if pd.notna(rsi):
        if signal == 1 and rsi > rsi_ob:
            reasons.append(f"RSI={rsi:.1f} > {rsi_ob} (overbought) conflicts with BUY signal")
        elif signal == 0 and rsi < rsi_os:
            reasons.append(f"RSI={rsi:.1f} < {rsi_os} (oversold) conflicts with FLAT signal")

    # ATR volatility check
    atr = latest.get("atr_14", None)
    if atr is not None and pd.notna(atr) and "atr_14" in df.columns:
        atr_mean = df["atr_14"].rolling(50).mean().iloc[-1]
        if pd.notna(atr_mean) and atr_mean > 0 and atr > vol_mult * atr_mean:
            reasons.append(f"ATR={atr:.6f} > {vol_mult}x rolling mean ({atr_mean:.6f})")

    # Gap check: |ret| > 5x vol_20
    ret = latest.get("ret", 0.0)
    vol = latest.get("vol_20", None)
    if vol is not None and pd.notna(vol) and vol > 0 and pd.notna(ret):
        if abs(ret) > 5.0 * vol:
            reasons.append(f"|ret|={abs(ret):.6f} > 5x vol_20 ({vol:.6f}), possible gap")

    passed = len(reasons) == 0
    return passed, reasons


# ---------------------------------------------------------------------------
# Backtest recent window
# ---------------------------------------------------------------------------

def backtest_recent_window(df, strategy_cfg, window=50):
    """
    Run backtester on the last N bars to check if the strategy
    is profitable in the recent window.

    Returns (profitable, metrics).
    """
    if len(df) < window:
        window = len(df)

    if window < 10:
        return False, {"error": "insufficient_data", "bars": window}

    recent_df = df.tail(window).copy()

    try:
        results = run_backtest(recent_df, strategy_cfg)
        metrics = results["metrics"]
        profitable = metrics["total_return_pct"] > 0
        return profitable, metrics
    except Exception as e:
        return False, {"error": str(e)}


# ---------------------------------------------------------------------------
# Signal validation (main entry point)
# ---------------------------------------------------------------------------

def validate_signal(instrument, signal, df, cfg, models=None):
    """
    Main entry point — orchestrates all AI validation checks:
      1. Build feature vector -> ensemble predict -> confidence + agreement
      2. Run sanity checks
      3. Run backtest on recent window
      4. Combine into final decision

    If confidence < threshold OR sanity fails OR ensemble disagrees -> REJECT.

    Returns dict with: approved, confidence, rationale, checks.
    """
    ai_cfg = cfg.get("ai", {})
    threshold = ai_cfg.get("confidence_threshold", 0.85)
    bt_window = ai_cfg.get("backtest_validation_window", 50)
    strategy_cfg = cfg.get("strategy", {})

    checks = {}
    rationale_parts = []

    # 1. Ensemble prediction
    if models:
        fv = build_feature_vector(df)
        confidence, predictions, agreement = ensemble_predict(models, fv)
        checks["ensemble_confidence"] = round(confidence, 4)
        checks["ensemble_predictions"] = [round(p, 4) for p in predictions]
        checks["ensemble_agreement"] = agreement
    else:
        confidence = 0.0
        agreement = False
        checks["ensemble_confidence"] = 0.0
        checks["ensemble_agreement"] = False
        rationale_parts.append("No ensemble models available")

    # 2. Sanity checks
    sanity_passed, sanity_reasons = sanity_checks(df, instrument, cfg)
    checks["sanity_passed"] = sanity_passed
    checks["sanity_reasons"] = sanity_reasons
    if not sanity_passed:
        rationale_parts.append(f"Sanity failed: {'; '.join(sanity_reasons)}")

    # 3. Backtest recent window
    bt_profitable, bt_metrics = backtest_recent_window(df, strategy_cfg, window=bt_window)
    checks["backtest_profitable"] = bt_profitable
    checks["backtest_metrics"] = bt_metrics
    if not bt_profitable:
        rationale_parts.append(f"Recent backtest unprofitable: {bt_metrics.get('total_return_pct', 'N/A')}%")

    # 4. Final decision
    approved = True
    if confidence < threshold:
        approved = False
        rationale_parts.append(f"Confidence {confidence:.4f} < threshold {threshold}")
    if not agreement and models:
        approved = False
        rationale_parts.append("Ensemble models disagree on direction")
    if not sanity_passed:
        approved = False

    rationale = "; ".join(rationale_parts) if rationale_parts else "All checks passed"

    decision = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "instrument": instrument,
        "signal": signal,
        "confidence": round(confidence, 4),
        "approved": approved,
        "ensemble_agreement": agreement if models else False,
        "sanity_passed": sanity_passed,
        "backtest_profitable": bt_profitable,
        "rationale": rationale,
        "checks": checks,
    }

    return decision


# ---------------------------------------------------------------------------
# Decision logging
# ---------------------------------------------------------------------------

def log_ai_decision(decision_details):
    """
    Append AI decision to logs/ai_decisions.csv.
    Creates the file with headers if it doesn't exist.
    """
    root = get_project_root()
    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)
    csv_path = logs_dir / "ai_decisions.csv"

    columns = [
        "timestamp", "instrument", "signal", "confidence", "approved",
        "ensemble_agreement", "sanity_passed", "backtest_profitable", "rationale",
    ]

    row = {col: decision_details.get(col, "") for col in columns}

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def main():
    """
    Standalone test: train ensemble on Supabase data, run validation,
    and print results.
    """
    import os
    from supabase import create_client
    from backtester import fetch_candles_from_supabase

    cfg = load_config()

    print("=" * 60)
    print("fx-quant AI Decision Wrapper — Standalone Test")
    print("=" * 60)

    # Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_KEY in config/.env")
    sb = create_client(supabase_url, supabase_key)
    table = cfg.get("supabase", {}).get("table", "fx_candles")

    strategy_cfg = cfg["strategy"]
    instruments = cfg["brokers"][0]["instruments"]
    granularity = cfg["data"]["candle_granularities"][0]

    for instrument in instruments:
        print(f"\n--- {instrument} ---")

        df = fetch_candles_from_supabase(instrument, granularity, sb, table)
        if df.empty:
            print(f"  No data for {instrument}. Skipping.")
            continue

        df = generate_signals(df, strategy_cfg)
        if df.empty:
            print(f"  No valid rows after signal generation. Skipping.")
            continue

        # Train ensemble
        models, val_metrics = train_ensemble(df, strategy_cfg)
        print(f"  Validation metrics: {val_metrics}")

        if not models:
            print("  Skipping validation — no models trained.")
            continue

        # Build context
        context = build_context(df, instrument, sb, cfg)
        print(f"  Context: {context}")

        # Validate latest signal
        latest_signal = int(df["signal"].iloc[-1])
        decision = validate_signal(instrument, latest_signal, df, cfg, models=models)

        print(f"  Signal: {'LONG' if latest_signal == 1 else 'FLAT'}")
        print(f"  Approved: {decision['approved']}")
        print(f"  Confidence: {decision['confidence']:.4f}")
        print(f"  Agreement: {decision['ensemble_agreement']}")
        print(f"  Sanity passed: {decision['sanity_passed']}")
        print(f"  Backtest profitable: {decision['backtest_profitable']}")
        print(f"  Rationale: {decision['rationale']}")

        # Log decision
        log_ai_decision(decision)
        print(f"  Decision logged to logs/ai_decisions.csv")

    print("\n" + "=" * 60)
    print("Standalone test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
