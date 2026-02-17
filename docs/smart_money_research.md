# Smart Money Research Log

## Purpose
Document observations about institutional behaviour patterns discovered during
backtesting. These insights inform Phase 2 refinements and future strategy development.

## Observations

### [Date] - Strategy 1, GBP/AUD Retest Volume Pattern
**Finding:** When tick volume at retest is >80% of the initial break volume,
win rate jumps from 58% to 71%.

**Hypothesis:** High volume at retest = institutions actively defending the level.
Low volume = retail-driven bounce, institutions absent.

**Action:** Add volume ratio check to Strategy 1 in Phase 2.
**Status:** To be tested

---

### [Date] - Strategy 2, VWAP Session Timing
**Finding:** VWAP reversals during first 60min of London session have 68% win rate.
Outside this window: 52% win rate.

**Hypothesis:** Institutional re-entry at VWAP happens predominantly during early
session price discovery.

**Action:** Add session timing filter to Strategy 2.
**Status:** To be tested

---

### [Date] - All Strategies, News Impact
**Finding:** Win rate drops 15-20% across all strategies when trading within
2 hours of major news events.

**Hypothesis:** News-driven volatility creates unpredictable institutional flow
that technical signals can't capture.

**Action:** Expand news block window from 30min to 2 hours.
**Status:** Implemented

---

## Pattern Library

### Stop Hunt Signature
- Sharp spike through obvious level (>1.5x ATR in single candle)
- Immediate reversal within 2 candles
- Volume spike on reversal candle
-> Indicates liquidity sweep, not genuine breakout

### Institutional Accumulation Signature
- Price consolidates in tight range (< 0.5x ATR) for 30+ minutes
- Volume gradually declining during consolidation
- Sudden expansion with volume spike
-> Indicates completed accumulation, strong directional move likely

---

## Failed Hypotheses
Document things that DIDN'T work to avoid retesting them:

### [Date] - Fibonacci Time Zones
**Tested:** Using Fibonacci time projections to predict reversal timing.
**Result:** No edge. Win rate unchanged.
**Conclusion:** Time-based Fib levels are noise, not institutional behaviour.
