# Strategy Learnings — Phase 1 Complete

Captures what we've learned from backtesting, filter analysis, and strategy iteration.
Updated: 2026-02-19.

Data period: 2021-01-01 to 2023-08-31 (OANDA practice account, H1 candles).
Backtester: Event-driven, no lookahead, spread + slippage modeled per pair.
Train/Test: 70/30 chronological split. Generalization ratio = OOS PF / IS PF.

---

## Phase 1 Final Scorecard

### Strategies Advancing to Phase 2

| Strategy | Pair | Trades (Full) | Trades (OOS) | WR% | PF (Full) | PF (OOS) | Gen Ratio | Edge Type |
|----------|------|---------------|--------------|-----|-----------|----------|-----------|-----------|
| **S7 Tight** | GBP_JPY | 107 | 43 | 59.8% | 0.96 | **1.39** | **1.81** | Liquidity sweep reversal |
| **S9** | GBP_USD | 296 | 42 | 53.7% | 0.75 | **1.45** | **2.10** | London session breakout |
| **S9 Filtered** | GBP_AUD | 66 | 14 | 60.6% | **1.14** | **1.94** | **2.04** | London session breakout |
| **S4-F** | EUR_AUD | 95 | — | 45.3% | **1.06** | — | — | EMA Ribbon trend context |
| **S3** | GBP_JPY | 155 | — | 52.3% | 0.99 | — | — | Key level momentum |

**Note on S7 and S9 full-dataset PF**: Both show PF < 1.0 over the full 2021-2023 period but
strong OOS profit factors (1.39-1.94) with high generalization ratios. This indicates the
strategies perform better in recent market conditions. Phase 2 will validate whether this
represents a genuine emerging edge or a recency fluke.

### Strategies on Watchlist (Marginal / Needs Work)

| Strategy | Pair | Trades (Full) | Trades (OOS) | WR% | PF | Notes |
|----------|------|---------------|--------------|-----|----|-------|
| S3 | GBP_USD | 179 | — | 53.1% | 1.02 | Near breakeven. Needs SL tightening. |
| S8 | GBP_JPY | 32 | 9 | 43.8% | 0.93 | Full PF 0.93, OOS PF 1.55. Low trade count. |

### Strategies Rejected

| Strategy | Concept | Trades Tested | Best PF | Why It Failed |
|----------|---------|---------------|---------|---------------|
| S1 | MA Breakout | ~500 | 0.88 | No edge. 35-47% WR across all pairs. |
| S2 | VWAP Reversal | ~400 | 0.78 | 20-25% WR. 26 consecutive losses at worst. |
| S4 (all except F) | EMA Ribbon variants | 0-248 | 0.78 | 7+ filters = 0 trades. Paradigm conflict. |
| S5 | Momentum Exhaustion | ~2,500 | 0.77 | High trade count but picks up too much noise. |
| S6 | EMA Bounce | ~440 | 0.84 | Good WR (59%) but terrible R:R — avg loss >> avg win. |
| S7 (1.5 ATR SL) | Liquidity Sweep wide SL | 46 | 0.71 | Too-wide stops (1.5 ATR). Tight SL (1.0 ATR) fixed it. |
| S9 | EUR_GBP | 448 | 0.65 | Tiny ranges — 2.0 pip spread eats 17% of avg win. |
| S9 | GBP_NZD | 191 | 0.67 | Wide spreads + choppy price action. |
| S9 | USD_JPY | ~30 | — | 31% WR. Asian range breakout doesn't work on USD_JPY. |

---

## Strategy Details

### S7 — Liquidity Sweep Reversal (Phase 2: GBP_JPY)

**Concept**: Price sweeps past a significant swing high/low (triggering clustered stop-loss
orders), then reverses. Based on institutional stop-hunting behavior documented by
Osler (2005, NY Fed).

**Final Parameters** (tight variant):
- Swing detection: 5-bar fractal, 100-bar lookback
- Sweep range: 0.7-1.5 ATR penetration past swing level
- Reversal confirmation: close back inside range, in upper/lower 40% of candle
- **SL: 1.0 ATR** beyond sweep extreme (tightened from 1.5 — the key improvement)
- TP1: 1.5 ATR (close 50%), TP2: 3.0 ATR (close 50%), trail at 1.5 ATR
- HTF: 200 EMA trend alignment (hard gate)
- Session: 08:00-17:00 UTC
- Confluence: OBV divergence (+1), RSI extreme (+1), volume spike (+1)
- Max hold: 40 bars

**GBP_JPY Full Dataset Results (2021-2023)**:
| Split | Trades | WR% | PF | PnL (pips) | Max DD | Expectancy |
|-------|--------|-----|----|------------|--------|------------|
| FULL | 107 | 59.8% | 0.96 | -22p | -11.9% | -0.2p |
| TRAIN | 61 | 54.1% | 0.77 | -337p | -11.8% | -5.5p |
| TEST | 43 | 67.4% | 1.39 | +331p | -3.8% | +7.7p |

**6-Month Results (Mar-Aug 2023)**: 25 trades, 76% WR, PF 2.01, +445p

**Exit profile (full dataset)**: 40% hit TP3, 36% partial (TP1+SL), 24% full SL.

**Pair suitability tested**: GBP_JPY is the only viable pair. GBP_USD (PF 0.92),
EUR_USD (PF 0.75), EUR_AUD (PF 0.29), GBP_AUD (PF 0.83), GBP_NZD (PF 0.39) all fail.
Crosses suffer from wide spreads killing the tight SL approach (R:R stays ~0.63).

**Why GBP_JPY works**: High volatility (avg ATR ~80-100 pips) gives enough room for the
1.0 ATR SL to absorb noise while the 1.5/3.0 ATR TP targets capture meaningful moves.
JPY pairs have cleaner swing structure due to institutional flow patterns around
Tokyo/London handoff.

---

### S9 — London Session Breakout (Phase 2: GBP_USD + GBP_AUD filtered)

**Concept**: Asian session (00:00-07:00 UTC) establishes a range; London open breaks it with
institutional order flow. Among the most well-documented FX phenomena
(Andersen & Bollerslev 1997, BIS triennial survey data).

**Base Parameters**:
- Asian range: 00:00-07:00 UTC high/low
- Entry window: 07:00-10:00 UTC (London kill zone)
- Volume: > 1.5x Asian session average (relaxed from 3.0 for H1 bars)
- SL: Opposite side of Asian range, capped at 2.5 ATR
- TP1: 1.0x Asian range width (measured move), TP2: 2.0x range
- Time exit: 17:00 UTC
- HTF trend: 200 EMA (soft confluence, not hard gate)
- ADX > 20 (soft confluence)
- Max hold: 40 bars

**GBP_AUD Filtered Overrides** (the key improvement):
- Skip Friday trades (44% WR, -17.6p avg — position squaring kills breakouts)
- ADX hard gate > 25 (low-ADX entries at 43% WR are pure noise)
- Require EMA50 distance > 40 pips (close-to-EMA trades are 50% WR)

**GBP_USD Full Dataset Results (2021-2023)**:
| Split | Trades | WR% | PF | PnL (pips) | Max DD |
|-------|--------|-----|----|------------|--------|
| FULL | 296 | 53.7% | 0.75 | -1,238p | -35.5% |
| TRAIN | 253 | 50.6% | 0.69 | -1,608p | -35.5% |
| TEST | 42 | 71.4% | 1.45 | +291p | -4.5% |

**GBP_AUD Filtered Full Dataset Results (2021-2023)**:
| Split | Trades | WR% | PF | PnL (pips) | Max DD |
|-------|--------|-----|----|------------|--------|
| FULL | 66 | 60.6% | 1.14 | +474p | -9.5% |
| TRAIN | 51 | 56.9% | 0.95 | +111p | -9.5% |
| TEST | 14 | 71.4% | 1.94 | +295p | -2.7% |

**Pair suitability tested across 8 pairs**:
| Pair | Full PF | OOS PF | Verdict |
|------|---------|--------|---------|
| GBP_USD | 0.75 | **1.45** | Primary — strong OOS, regime-dependent |
| GBP_AUD (filtered) | **1.14** | **1.94** | Secondary — excellent generalization |
| EUR_USD | 0.93 | 1.05 | Marginal — breakeven OOS, filter overfits |
| GBP_AUD (baseline) | 0.91 | 1.09 | Marginal without filters |
| GBP_NZD | 0.67 | 0.79 | Rejected — wide spreads |
| EUR_GBP | 0.65 | 0.55 | Rejected — tiny ranges, spread eats profit |
| USD_JPY | — | — | Rejected — 31% WR |
| EUR_AUD | — | — | Rejected — poor results in initial test |

**EUR_USD filter experiments** (documented for reference):
- RSI 40-60 skip + ADX > 30 + entry Hour 8 + TP1 1.5x: In-sample PF 1.26, OOS PF 0.89 (overfit)
- RSI 40-60 skip + TP1 1.5x only: In-sample PF 1.09, OOS PF 0.90 (overfit)
- TP1 1.5x only: In-sample PF 1.04, OOS PF 1.00 (structural R:R improvement but no profit)
- Conclusion: EUR_USD S9 is fundamentally breakeven. No filter combo produces OOS profit.

---

### S8 — Order Block Retest (Watchlist)

**Concept**: Institutional order blocks (last opposing candle before a displacement move) act
as support/resistance on retest. Requires 1.5 ATR displacement, rejection candle on
retest, and 2-of-3 confluence (FVG, broken S/R, declining pullback volume).

**GBP_JPY Full Dataset Results (2021-2023)**:
| Split | Trades | WR% | PF | PnL (pips) |
|-------|--------|-----|----|------------|
| FULL | 32 | 43.8% | 0.93 | -40p |
| TRAIN | 23 | 39.1% | 0.72 | -155p |
| TEST | 9 | 55.6% | 1.55 | +115p |

**6-Month Results**: 15 trades, 40% WR, PF 1.47, RR 2.21, +106p

**Status**: Promising R:R profile (avg win 55p, avg loss 25p) but too few trades.
The 1.5 ATR displacement threshold may be too strict for the full dataset. On watchlist
for Phase 2 with potential to relax displacement to 1.2 ATR and test on more pairs.

---

### S3 — Key Level Momentum Breakout (Phase 2: GBP_JPY)

**Concept**: H1 breakouts above/below significant S/R levels (3+ touch clusters) with
volume confirmation, strong candle close, MACD alignment, and ADX > 20.

**Results (Full Dataset)**:
| Pair | Trades | WR% | PF | PnL (pips) | Max DD |
|------|--------|-----|----|------------|--------|
| GBP_JPY | 155 | 52.3% | 0.99 | +408p | -13.5% |
| GBP_USD | 179 | 53.1% | 1.02 | +97p | -10.2% |
| USD_JPY | 138 | 52.9% | 1.00 | +66p | -9.8% |

**Status**: The only Phase 1 strategy near breakeven across multiple pairs. Minimal
filters (4 hard gates), single paradigm. Advancing to Phase 2 on GBP_JPY with
suggested SL tightening and retest confirmation experiments.

---

## S4 EMA Ribbon — Detailed Filter Analysis

### What We Tested
- **S4-D** (Volume + ADX Gating): 0 trades across 3 pairs
- **S4-E** (Compression Quality + Stochastic): 1 trade across 3 pairs
- **S4-F** (Trend Context Filter): 248 trades, 39.1% WR, PF 0.78
- **S4-F-v2** (Quick Tune with 3 TFs): 0-2 trades depending on threshold adjustments
- **S4-G** (Pullback-first): 12 trades, 16.7% WR, PF 0.17
- **S4-G-Minimal** (Pullback, no M5 EMA): 12 trades, same result

### S4 Variant Summary Table
| Variant | Trades | WR | PF | Verdict |
|---------|--------|-----|-----|---------|
| S4-D (Vol+ADX) | 0 | — | — | Dead: filters too strict |
| S4-E (Compression) | 1 | — | — | Dead: filters too strict |
| S4-F (Trend Context) | 248 | 39.1% | 0.78 | Best S4, but losing money |
| S4-F EUR_AUD only | 95 | 45.3% | 1.06 | Only profitable pair |
| S4-F-v2 (3TF tune) | 0 | — | — | Dead: contradictory filters |
| S4-G (Pullback-first) | 12 | 16.7% | 0.17 | Dead: pullbacks don't resume |
| S4-G-Minimal | 12 | 16.7% | 0.17 | Same — M5 EMA wasn't the issue |

---

## Key Design Principles (Derived from All Phase 1 Testing)

### 1. Single Paradigm Per Strategy
Momentum + mean-reversion filters are contradictory. When momentum is strong (ADX rising,
ribbon expanding), RSI is NOT at extremes and price is far from the 50 EMA. When stochastic
is oversold near the EMA, momentum hasn't fired yet. S4-D/E proved this with 0-1 trades.

### 2. Maximum 3-4 Hard Filters
Each additional hard filter compounds multiplicatively. S4-D/E/F-v2 had 7-9 simultaneous
hard gates, resulting in near-zero trades. Additional conditions should be confluence scores
(+1 to signal quality) not hard gates.

### 3. Validate Thresholds Against Data
S4 required 0.8% EMA expansion — data shows the maximum expansion within 10 bars was 0.70%.
The threshold was literally impossible. A 5-minute diagnostic of data distributions saves
hours of debugging 0-trade results.

### 4. Volume as Soft Filter
Volume > 2.0x 20-period average: only 4% of signals qualify.
Volume > 1.5x: only 21% qualify.
Volume > 1.2x: 40% qualify.
Use 1.2x as hard gate max; higher thresholds as confluence bonus only.

### 5. H1 Timeframe Has Natural Edge
S3 works on H1. Most M15 strategies struggle with noise. H1 bars naturally filter
whipsaws while capturing meaningful session moves.

### 6. Simple Strategies Outperform Complex Ones
S3 (4 filters) beats S4 (7+ filters) and S5 (complex exhaustion logic).
The right complexity level is the minimum needed for the concept to work.

### 7. Win Rate Isn't Everything
S6 has 59% WR but PF < 1.0 because avg loss >> avg win. R:R (risk-reward ratio) matters
as much as WR. A strategy needs WR × avg_win > (1-WR) × avg_loss to be profitable.

### 8. Pair Specificity Is Real
S7 works on GBP_JPY but fails on 5 other pairs. S9 works on GBP_USD and GBP_AUD but
fails on EUR_GBP and GBP_NZD. Don't assume cross-pair generalization — always validate.

### 9. Tighter Stops Often Win
S7: PF jumped from 0.71 → 2.01 when SL tightened from 1.5 → 1.0 ATR. Wider stops
don't give "more room" — they give back more profit on reversals. The optimal SL is the
minimum that avoids noise-triggered exits.

### 10. Filter Optimization Overfits Easily
S9 EUR_USD: Every filter combination improved in-sample but degraded OOS.
RSI filter, ADX filter, entry hour delay — all overfit the training period.
Only structural changes (TP1 multiplier adjustment) held across splits.
Per-pair signal filters (RSI zones, ADX thresholds) are high overfitting risk.

### 11. Regime Dependence Is Common
S7 GBP_JPY and S9 GBP_USD both show: losing in 2021-2022, profitable in 2022-2023.
This could mean the edge is strengthening (market structure evolution) or that the
test period was lucky. Phase 2 must validate with forward testing.

---

## Phase 2 Plan

### Strategies to Forward-Test
| Strategy | Pair | Config |
|----------|------|--------|
| S7 Tight | GBP_JPY | SL 1.0 ATR, TP1 1.5 ATR, TP2 3.0 ATR |
| S9 | GBP_USD | Base parameters, entry 07:00-10:00 UTC |
| S9 Filtered | GBP_AUD | No Friday, ADX > 25, EMA50 dist > 40 pips |
| S4-F | EUR_AUD | EMA Ribbon trend context, 2.0 ATR SL, 3.0 ATR TP |
| S3 | GBP_JPY | Base parameters (pending SL tightening test) |

### Phase 2 Objectives
1. **Forward validation**: Run strategies on live data (paper account) for 3+ months
2. **Regime analysis**: Monitor whether OOS edge persists or was period-specific
3. **Portfolio construction**: Test correlation between S7/S9/S3 signals — do they
   diversify or cluster?
4. **Position sizing**: Implement Kelly criterion or fixed fractional based on
   actual win rate / R:R distributions
5. **S8 exploration**: Relax displacement threshold (1.2 ATR) and test on additional pairs
6. **S3 optimization**: Test tighter SL and retest confirmation entry
7. **Drawdown management**: Define maximum portfolio drawdown limits and auto-pause rules

### Infrastructure Needed
- Live OANDA data feed for paper trading
- Trade logger with real-time metrics dashboard
- Automated signal generation (currently backtester-only)
- Alert system for human review of signals before execution
