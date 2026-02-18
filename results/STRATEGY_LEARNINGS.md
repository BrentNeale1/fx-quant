# Strategy Learnings

Captures what we've learned from backtesting, filter analysis, and strategy iteration.
Updated: 2026-02-18.

---

## Phase 1 Strategy Scorecard

### What Worked

**S3 — Key Level Momentum Breakout** (H1 timeframe)
| Pair | Trades | WR | PF | PnL (pips) | Max DD |
|------|--------|-----|-----|------------|--------|
| GBP_JPY | 155 | 52.3% | 0.99 | +408 | -13.5% |
| GBP_USD | 179 | 53.1% | 1.02 | +97 | -10.2% |
| USD_JPY | 138 | 52.9% | 1.00 | +66 | -9.8% |

S3 is the only strategy consistently near or above breakeven across multiple pairs. It uses H1 key level breakouts with volume confirmation, strong candle close, MACD alignment, and ADX > 20. Simple concept, minimal filters (4 hard gates), works on JPY crosses and GBP_USD.

**Why it works**: Single-paradigm (momentum breakout), validated levels (3+ touch S/R clusters), and the H1 timeframe naturally filters noise. ADX > 20 is a soft trend gate, not an aggressive one.

**Suggested tweaks for further testing**:
- Tighten SL from `level - 0.5 ATR` to `level - 0.3 ATR` to improve RR
- Test adding a "retest" confirmation (wait for price to pull back near the level after breakout)
- Pair-specific TP tuning: GBP_JPY may benefit from wider TPs (higher volatility)
- Extend to EUR_AUD and GBP_AUD to test cross-pair robustness

### What Has Potential But Needs Tweaks

**S4-F — EMA Ribbon Trend Context** (EUR_AUD only)
- 95 trades, 45.3% WR, PF 1.06, +173 pips
- Only EUR_AUD was profitable; GBP_AUD (PF 0.53) and GBP_JPY (PF 0.84) lost money
- The 2.0 ATR SL / 3.0 ATR TP gives RR 1.21 — needs WR > 45% to profit

**Suggested tweaks**:
- EUR_AUD-specific only (don't try to universalise)
- Consider tighter SL (1.5 ATR) with same TP (3.0 ATR) to improve RR to 2.0
- Add partial TP at 1.5 ATR with trail on remainder
- The H1 EMA stack (20>50>100>200) is probably too strict — try just 50>200

**S6 — EMA Bounce** (EUR_AUD and GBP_USD)
| Pair | Trades | WR | PF | PnL (pips) |
|------|--------|-----|-----|------------|
| EUR_AUD | 209 | 59.8% | 0.84 | -380 |
| GBP_USD | 231 | 58.4% | 0.83 | -537 |

Win rate is strong (58-60%) but profit factor is below 1.0. This means average wins are too small relative to average losses — the SL/TP ratio needs work.

**Suggested tweaks**:
- Tighten SL to reduce avg loss size (currently losing trades overwhelm winning ones)
- Try wider TP or trailing stop to let winners run
- Win rate is high enough — just need better risk/reward mechanics

### What Failed

**S1 — MA Breakout**: 35-47% WR, PF 0.40-0.88 across all pairs. No edge.

**S2 — VWAP Reversal**: 20-25% WR, PF 0.62-0.78. 26 consecutive losses at worst. Disabled.

**S4 (all variants except F on EUR_AUD)**: Extensively tested 7 variations. See detailed analysis below.

**S5 — Momentum Exhaustion**: High trade counts (240-719 per pair) but PF 0.43-0.77. Picks up too much noise.

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

### Key Findings

#### 1. Momentum + Mean-Reversion Filters Are Contradictory
The single biggest learning. Combining:
- **Momentum filters**: ADX > 28 rising, ribbon expanding, volume spike
- **Mean-reversion filters**: Stochastic < 20, price near H1 50 EMA

These almost never co-exist. When momentum is strong, stochastic is NOT at extremes and price is far from the H1 50 EMA. When stochastic is at extremes near the 50 EMA, momentum hasn't fired yet.

**Implication**: Choose one paradigm per strategy. Either build a momentum-continuation strategy OR a pullback-to-trend strategy, not both.

#### 2. EMA Compression + Expansion Thresholds Must Be Data-Validated
- Spec required all 5 EMAs within 0.4% (compression) then expanding to 0.8% distance
- **Data shows**: After 0.4% compression, the MAX expansion within 10 M15 bars was 0.70%
- 0.8% expansion literally never occurs (0 out of 163,030 samples)
- p90 expansion = 0.23%, p99 = 0.39%

**Implication**: Always validate thresholds against actual data distributions before running backtests. A 5-minute diagnostic saves hours of debugging 0-trade results.

#### 3. Volume Spike Filters Compound Aggressively
- Volume > 2.0x 20-period average: only 4% of expansion signals qualify
- Volume > 1.5x: only 21% qualify
- Volume > 1.2x: 40% qualify

When combined with 3+ other filters, even 1.5x volume becomes a near-total blocker.

**Implication**: Use volume as a soft filter (1.2x) or confluence bonus, not a hard gate, when stacking 4+ other conditions.

#### 4. Price Distance from H1 50 EMA During Strong Trends
- When H1 ADX > 28 and ribbon is expanding, median price distance from H1 50 EMA = 7.2x M15 ATR
- 0% of signals are within 1.5 ATR; only 10% are within 5.0 ATR
- The stronger the trend signal, the further price has already moved

**Implication**: "Price near slow MA" filters only work for pullback strategies, not breakout/expansion strategies.

#### 5. Filter Stacking Has Diminishing Returns
Each additional hard filter compounds multiplicatively. The S4-D/E/F-v2 specs had 7-9 simultaneous hard gates, resulting in near-zero trades.

**Rule of thumb**: 3-4 hard filters max. Additional conditions should be confluence scores or soft gates.

#### 6. Pullback-First: Right Idea, Wrong Entry Trigger
- Flipped filter priority: find stochastic pullback FIRST, then confirm trend
- Filter funnel worked well through 7 layers (332 signals at M15 ribbon stage)
- M5 EMA stacking contradicts pullback timing: after stoch < 20, fast EMAs are below slow EMAs
- Even without EMA check, pullback entries had 16.7% WR — most pullbacks were trend reversals, not continuations
- Stochastic oversold in a trend is often a warning, not a buying signal

**Implication**: Need a different entry trigger for pullbacks — possibly price action (engulfing candles, pin bars at key levels) rather than indicator-based timing.

---

## Strategy Design Principles (Derived from All Testing)

1. **Pick one paradigm**: momentum OR mean-reversion, not both in the same strategy
2. **3-4 hard filters max**: Additional conditions should be confluence scores, not hard gates
3. **Validate thresholds against data**: Run distribution checks before backtesting
4. **Volume as soft filter**: 1.2x for hard gate max; higher thresholds as confluence bonus only
5. **Test per-pair first**: Strategies perform very differently across pairs (EUR_AUD vs GBP_AUD)
6. **H1 timeframe has natural edge**: S3 works on H1; most M15 strategies struggle with noise
7. **Simple strategies outperform complex ones**: S3 (4 filters) beats S4 (7+ filters) and S5 (complex exhaustion logic)
8. **Win rate isn't everything**: S6 has 59% WR but PF < 1.0 because risk/reward is wrong. Fix the SL/TP before adding more entry filters.

---

## Next Steps

- S4 is exhausted. S4-F on EUR_AUD (PF 1.06) is the ceiling — marginal.
- Move to Smart Money / institutional flow strategies
- Revisit S3 with tweaks (tighter SL, retest confirmation, expanded pairs)
- Revisit S6 with SL/TP restructuring (win rate is already good)
