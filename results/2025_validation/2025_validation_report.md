# 2025 Forward Validation Report

## Overview

This report presents the results of running the 4 Phase 2 passing strategies on 2025 data — a true forward test on data never seen during development or parameter tuning.

**Strategies tested:**
- S7_Tight — Liquidity Sweep (GBP_JPY, H1)
- S9_Filtered — London Session Breakout (GBP_AUD, H1)
- S3 — Key Level Breakout (GBP_JPY, H1)
- S8_OB — Order Block Retest (GBP_USD, M15)

**Data:** Downloaded from Dukascopy. Oct 2024 used for indicator warmup; trades counted from Jan 1 - Dec 31, 2025.

**Phase 2 periods for reference:**
- In-Sample (IS): Jan 2021 - Dec 2022
- Out-of-Sample (OOS): Jan 2023 - Aug 2023

---

## 1. Portfolio Summary

| Strategy     | Pair    | TF  | IS PF | OOS PF | 2025 PF | 2025 Trades | 2025 WR% | 2025 PnL (pips) | 2025 Verdict |
|--------------|---------|-----|-------|--------|---------|-------------|----------|-----------------|--------------|
| S7_Tight     | GBP_JPY | H1  | 1.52  | 1.80   | 0.52    | 18          | 55.6%    | -293.6          | FAIL         |
| S9_Filtered  | GBP_AUD | H1  | 1.31  | 2.26   | 0.74    | 18          | 55.6%    | -112.2          | FAIL         |
| S3           | GBP_JPY | H1  | 1.22  | 1.23   | 1.05    | 35          | 42.9%    | +52.2           | PASS         |
| S8_OB        | GBP_USD | M15 | 1.39  | 1.59   | 0.84    | 19          | 31.6%    | -19.9           | FAIL         |
| **Portfolio**| —       | —   | 1.31  | 1.55   | **0.83**| **90**      | **45.6%**| **-373.4**      | **FAIL**     |

---

## 2. Detailed Strategy Comparison Across Periods

### 2.1 S7_Tight — Liquidity Sweep (GBP_JPY H1)

|                    | IS (2021-2022) | OOS (2023) | 2025       |
|--------------------|----------------|------------|------------|
| Trades             | 23             | 15         | 18         |
| Win Rate           | 69.6%          | 73.3%      | 55.6%      |
| Profit Factor      | 1.52           | 1.80       | 0.52       |
| PnL (pips)         | +207.2         | +200.1     | -293.6     |
| Avg Win            | +37.8p         | +41.0p     | +31.2p     |
| Avg Loss           | -56.9p         | -62.6p     | -75.8p     |
| Reward:Risk        | 0.67           | 0.65       | 0.41       |
| Biggest Win        | +85.2p         | +94.8p     | +58.0p     |
| Biggest Loss       | -84.1p         | -73.7p     | -133.1p    |
| TP3 Hits           | 9              | 5          | 3          |
| TP1+Partial Exits  | 7              | 5          | 6          |
| SL Hits            | 7              | 4          | 8          |
| TIME Exits         | 0              | 1          | 1          |
| LONG trades (WR)   | 9 (78%)        | 11 (82%)   | 14 (57%)   |
| SHORT trades (WR)  | 14 (64%)       | 4 (50%)    | 4 (50%)    |
| Avg Hold (min)     | 798            | 1120       | 733        |
| Avg Confluence      | 3.8            | 3.5        | 3.7        |
| Avg ATR at Entry   | 0.2536         | 0.2808     | 0.3216     |
| Avg ADX at Entry   | 31.6           | 27.3       | 29.1       |

**2025 Trades:**

| # | Date       | Dir   | Entry    | Exit Reason | PnL (pips) | Hold (min) | Confluence |
|---|------------|-------|----------|-------------|------------|------------|------------|
| 1 | 2025-01-06 | LONG  | 196.795  | TP1+SL      | +20.5      | 240        | 3          |
| 2 | 2025-01-20 | SHORT | 190.368  | SL          | -64.0      | 300        | 4          |
| 3 | 2025-01-23 | LONG  | 192.630  | TP3         | +58.0      | 600        | 5          |
| 4 | 2025-02-18 | LONG  | 191.401  | TP1+SL      | +21.6      | 660        | 4          |
| 5 | 2025-02-26 | SHORT | 189.275  | TP1+SL      | +27.6      | 960        | 4          |
| 6 | 2025-03-06 | LONG  | 191.106  | SL          | -133.1     | 1560       | 4          |
| 7 | 2025-03-13 | LONG  | 192.018  | SL          | -104.8     | 480        | 4          |
| 8 | 2025-03-21 | LONG  | 193.465  | SL          | -85.8      | 240        | 4          |
| 9 | 2025-04-10 | SHORT | 187.113  | TP1+SL      | +48.6      | 1020       | 3          |
| 10| 2025-05-07 | LONG  | 191.644  | SL          | -60.6      | 360        | 3          |
| 11| 2025-09-25 | SHORT | 199.696  | SL          | -45.3      | 1020       | 4          |
| 12| 2025-10-09 | LONG  | 204.387  | SL          | -69.8      | 240        | 3          |
| 13| 2025-11-24 | LONG  | 205.687  | TIME        | +8.3       | 2400       | 3          |
| 14| 2025-12-03 | LONG  | 206.666  | TP3         | +49.9      | 720        | 4          |
| 15| 2025-12-23 | LONG  | 210.611  | TP1+SL      | +17.9      | 360        | 4          |
| 16| 2025-12-24 | LONG  | 210.406  | TP1+SL      | +18.4      | 960        | 4          |
| 17| 2025-12-30 | LONG  | 210.740  | SL          | -42.7      | 840        | 3          |
| 18| 2025-12-31 | LONG  | 210.492  | TP3         | +41.7      | 240        | 3          |

---

### 2.2 S9_Filtered — London Session Breakout (GBP_AUD H1)

|                    | IS (2021-2022) | OOS (2023) | 2025       |
|--------------------|----------------|------------|------------|
| Trades             | 42             | 9          | 18         |
| Win Rate           | 59.5%          | 66.7%      | 55.6%      |
| Profit Factor      | 1.31           | 2.26       | 0.74       |
| PnL (pips)         | +299.5         | +217.8     | -112.2     |
| Avg Win            | +50.3p         | +65.2p     | +32.7p     |
| Avg Loss           | -56.3p         | -57.7p     | -54.9p     |
| Reward:Risk        | 0.89           | 1.13       | 0.60       |
| Biggest Win        | +144.8p        | +128.3p    | +63.5p     |
| Biggest Loss       | -73.9p         | -77.5p     | -79.1p     |
| TP3 Hits           | 8              | 3          | 3          |
| TP1+Partial Exits  | 17             | 3          | 7          |
| SL Hits            | 16             | 3          | 8          |
| TIME Exits         | 1              | 0          | 0          |
| LONG trades (WR)   | 19 (58%)       | 6 (67%)    | 8 (50%)    |
| SHORT trades (WR)  | 23 (61%)       | 3 (67%)    | 10 (60%)   |
| Avg Hold (min)     | 937            | 893        | 600        |
| Avg Confluence      | 4.8            | 4.8        | 4.9        |
| Avg ATR at Entry   | 0.0028         | 0.0030     | 0.0023     |
| Avg ADX at Entry   | 33.8           | 38.1       | 35.2       |

**2025 Trades:**

| # | Date       | Dir   | Entry   | Exit Reason | PnL (pips) | Hold (min) | Confluence |
|---|------------|-------|---------|-------------|------------|------------|------------|
| 1 | 2025-01-14 | SHORT | 1.96987 | TP1+SL      | +20.5      | 540        | 5          |
| 2 | 2025-01-15 | SHORT | 1.96823 | TP1+SL      | +21.7      | 1080       | 5          |
| 3 | 2025-02-05 | SHORT | 1.99311 | TP1+SL      | +22.5      | 1080       | 5          |
| 4 | 2025-02-11 | SHORT | 1.96452 | SL          | -52.8      | 120        | 5          |
| 5 | 2025-02-12 | LONG  | 1.98255 | TP3         | +63.5      | 1320       | 4          |
| 6 | 2025-02-13 | LONG  | 1.99095 | SL          | -55.8      | 1140       | 5          |
| 7 | 2025-02-25 | LONG  | 1.99364 | TP3         | +51.0      | 360        | 5          |
| 8 | 2025-03-19 | LONG  | 2.05049 | SL          | -58.7      | 660        | 5          |
| 9 | 2025-05-27 | LONG  | 2.09807 | TP1+SL      | +23.8      | 480        | 5          |
| 10| 2025-08-21 | LONG  | 2.09756 | SL          | -60.1      | 240        | 5          |
| 11| 2025-08-28 | SHORT | 2.07018 | TP1+SL      | +12.8      | 300        | 5          |
| 12| 2025-09-02 | SHORT | 2.05877 | TP1+SL      | +19.6      | 240        | 5          |
| 13| 2025-09-11 | SHORT | 2.04354 | SL          | -39.0      | 300        | 5          |
| 14| 2025-09-25 | SHORT | 2.03833 | SL          | -43.0      | 300        | 5          |
| 15| 2025-10-08 | LONG  | 2.04341 | SL          | -50.3      | 300        | 5          |
| 16| 2025-11-12 | SHORT | 2.00723 | TP1+SL      | +28.4      | 1620       | 5          |
| 17| 2025-11-25 | LONG  | 2.03384 | TP3         | +63.0      | 300        | 5          |
| 18| 2025-12-17 | SHORT | 2.01252 | SL          | -79.1      | 420        | 5          |

---

### 2.3 S3 — Key Level Breakout (GBP_JPY H1)

|                    | IS (2021-2022) | OOS (2023) | 2025       |
|--------------------|----------------|------------|------------|
| Trades             | 84             | 32         | 35         |
| Win Rate           | 53.6%          | 59.4%      | 42.9%      |
| Profit Factor      | 1.22           | 1.23       | 1.05       |
| PnL (pips)         | +359.2         | +135.5     | +52.2      |
| Avg Win            | +43.6p         | +37.8p     | +68.8p     |
| Avg Loss           | -41.1p         | -44.8p     | -49.0p     |
| Reward:Risk        | 1.06           | 0.84       | 1.40       |
| Biggest Win        | +219.3p        | +94.4p     | +138.3p    |
| Biggest Loss       | -136.4p        | -59.4p     | -75.5p     |
| TP3 Hits           | 13             | 3          | 8          |
| TP1+Partial Exits  | 32             | 16         | 7          |
| SL Hits            | 39             | 13         | 20         |
| TIME Exits         | 0              | 0          | 0          |
| LONG trades (WR)   | 51 (51%)       | 20 (65%)   | 18 (44%)   |
| SHORT trades (WR)  | 33 (58%)       | 12 (50%)   | 17 (41%)   |
| Avg Hold (min)     | 1145           | 818        | 1106       |
| Avg Confluence      | 4.3            | 4.3        | 4.5        |
| Avg ATR at Entry   | 0.2846         | 0.3391     | 0.3467     |
| Avg ADX at Entry   | 34.6           | 35.7       | 32.7       |

**2025 Trades:**

| # | Date       | Dir   | Entry    | Exit Reason    | PnL (pips) | Hold (min) | Confluence |
|---|------------|-------|----------|----------------|------------|------------|------------|
| 1 | 2025-01-02 | SHORT | 194.497  | SL             | -48.7      | 180        | 5          |
| 2 | 2025-01-10 | SHORT | 193.326  | TP3            | +105.7     | 3660       | 5          |
| 3 | 2025-01-15 | SHORT | 191.684  | TP3            | +102.2     | 720        | 5          |
| 4 | 2025-01-16 | SHORT | 190.121  | SL             | -36.2      | 60         | 5          |
| 5 | 2025-01-24 | LONG  | 194.435  | SL             | -69.4      | 3480       | 4          |
| 6 | 2025-02-05 | SHORT | 190.408  | TP3            | +106.0     | 1260       | 5          |
| 7 | 2025-02-21 | SHORT | 189.529  | TP1+TP2+SL     | +72.3      | 3420       | 4          |
| 8 | 2025-03-12 | LONG  | 193.040  | SL             | -73.4      | 60         | 4          |
| 9 | 2025-03-17 | LONG  | 193.069  | TP3            | +86.8      | 1140       | 5          |
| 10| 2025-04-01 | SHORT | 192.294  | SL             | -53.6      | 120        | 4          |
| 11| 2025-04-03 | SHORT | 191.928  | TP3            | +138.3     | 1320       | 4          |
| 12| 2025-04-08 | SHORT | 187.910  | TP1+TP2+SL     | +125.8     | 660        | 4          |
| 13| 2025-05-13 | LONG  | 196.010  | SL             | -39.1      | 60         | 4          |
| 14| 2025-05-13 | LONG  | 196.058  | SL             | -45.5      | 840        | 4          |
| 15| 2025-05-20 | SHORT | 193.229  | SL             | -37.6      | 120        | 4          |
| 16| 2025-05-23 | SHORT | 192.468  | SL             | -38.2      | 180        | 4          |
| 17| 2025-06-05 | LONG  | 195.007  | TP1+SL         | +18.8      | 180        | 4          |
| 18| 2025-06-06 | LONG  | 195.787  | TP1+SL         | +16.4      | 5340       | 5          |
| 19| 2025-06-13 | LONG  | 195.670  | SL             | -51.3      | 240        | 4          |
| 20| 2025-06-20 | LONG  | 196.536  | TP3            | +61.7      | 3960       | 4          |
| 21| 2025-06-30 | LONG  | 197.968  | SL             | -46.7      | 420        | 4          |
| 22| 2025-08-01 | SHORT | 197.420  | TP3            | +76.6      | 120        | 5          |
| 23| 2025-08-14 | LONG  | 199.554  | TP1+SL         | +16.5      | 720        | 5          |
| 24| 2025-08-29 | SHORT | 198.084  | SL             | -28.7      | 60         | 5          |
| 25| 2025-09-03 | LONG  | 199.683  | SL             | -50.8      | 120        | 4          |
| 26| 2025-09-04 | LONG  | 199.530  | SL             | -37.2      | 540        | 5          |
| 27| 2025-09-26 | LONG  | 200.176  | TP1+SL         | +9.8       | 3420       | 5          |
| 28| 2025-09-30 | SHORT | 198.775  | SL             | -39.8      | 960        | 4          |
| 29| 2025-10-02 | SHORT | 197.699  | SL             | -66.8      | 720        | 5          |
| 30| 2025-10-31 | SHORT | 201.986  | SL             | -38.0      | 180        | 5          |
| 31| 2025-11-06 | SHORT | 200.678  | SL             | -75.5      | 1440       | 5          |
| 32| 2025-11-13 | LONG  | 203.814  | SL             | -40.4      | 120        | 4          |
| 33| 2025-11-17 | LONG  | 204.529  | SL             | -62.6      | 420        | 4          |
| 34| 2025-12-08 | LONG  | 207.684  | TP3            | +48.0      | 1080       | 5          |
| 35| 2025-12-17 | LONG  | 208.258  | TP1+TP2+SL     | +46.9      | 1380       | 5          |

---

### 2.4 S8_OB — Order Block Retest (GBP_USD M15)

|                    | IS (2021-2022) | OOS (2023) | 2025       |
|--------------------|----------------|------------|------------|
| Trades             | 67             | 17         | 19         |
| Win Rate           | 37.3%          | 52.9%      | 31.6%      |
| Profit Factor      | 1.39           | 1.59       | 0.84       |
| PnL (pips)         | +223.4         | +89.6      | -19.9      |
| Avg Win            | +31.7p         | +26.9p     | +17.1p     |
| Avg Loss           | -13.6p         | -19.0p     | -9.4p      |
| Reward:Risk        | 2.34           | 1.41       | 1.82       |
| Biggest Win        | +84.7p         | +68.7p     | +35.4p     |
| Biggest Loss       | -39.1p         | -37.5p     | -13.4p     |
| TP3 Hits           | 16             | 4          | 3          |
| TP1+Partial Exits  | 9              | 4          | 3          |
| SL Hits            | 41             | 7          | 13         |
| TIME Exits         | 1              | 2          | 0          |
| LONG trades (WR)   | 25 (28%)       | 9 (56%)    | 12 (25%)   |
| SHORT trades (WR)  | 42 (43%)       | 8 (50%)    | 7 (43%)    |
| Avg Hold (min)     | 203            | 247        | 91         |
| Avg Confluence      | 4.4            | 4.8        | 4.9        |
| Avg ATR at Entry   | 0.0014         | 0.0015     | 0.0010     |
| Avg ADX at Entry   | 29.7           | 38.2       | 29.2       |

**2025 Trades:**

| # | Date       | Dir   | Entry   | Exit Reason | PnL (pips) | Hold (min) | Confluence |
|---|------------|-------|---------|-------------|------------|------------|------------|
| 1 | 2025-01-17 | SHORT | 1.22070 | TP3         | +22.3      | 105        | 5          |
| 2 | 2025-01-17 | SHORT | 1.22051 | TP3         | +35.4      | 390        | 5          |
| 3 | 2025-02-13 | LONG  | 1.24897 | SL          | -5.7       | 135        | 4          |
| 4 | 2025-02-13 | LONG  | 1.25058 | TP1+SL      | +12.0      | 210        | 5          |
| 5 | 2025-02-18 | LONG  | 1.26054 | SL          | -11.5      | 75         | 5          |
| 6 | 2025-02-18 | LONG  | 1.25983 | SL          | -4.3       | 15         | 5          |
| 7 | 2025-02-18 | LONG  | 1.26051 | TP1+SL      | +7.3       | 165        | 5          |
| 8 | 2025-03-28 | LONG  | 1.29491 | SL          | -10.1      | 90         | 5          |
| 9 | 2025-03-28 | LONG  | 1.29473 | SL          | -9.3       | 45         | 5          |
| 10| 2025-04-29 | LONG  | 1.34259 | SL          | -8.6       | 15         | 5          |
| 11| 2025-09-23 | SHORT | 1.35107 | SL          | -12.8      | 105        | 5          |
| 12| 2025-09-23 | SHORT | 1.35144 | SL          | -8.9       | 45         | 5          |
| 13| 2025-09-23 | SHORT | 1.35120 | SL          | -11.5      | 90         | 5          |
| 14| 2025-11-07 | SHORT | 1.31134 | SL          | -13.4      | 30         | 5          |
| 15| 2025-11-07 | SHORT | 1.31222 | TP1+SL      | +5.8       | 45         | 5          |
| 16| 2025-11-18 | LONG  | 1.31568 | SL          | -9.3       | 90         | 5          |
| 17| 2025-11-26 | LONG  | 1.31552 | SL          | -7.6       | 15         | 5          |
| 18| 2025-12-04 | LONG  | 1.33390 | SL          | -9.7       | 15         | 5          |
| 19| 2025-12-04 | LONG  | 1.33364 | TP3         | +20.1      | 45         | 5          |

---

## 3. Key Findings

### 3.1 S7_Tight — Structural Failure

**What changed:** The reward:risk ratio degraded from 0.67 (IS) to 0.41 (2025). Average wins shrank from +37.8p to +31.2p while average losses expanded from -56.9p to -75.8p. ATR at entry increased from 0.2536 to 0.3216, meaning wider stop-loss distances and larger losses when hit.

**Root cause:** The strategy depends on a high win rate (70%+) to compensate for a structurally poor R:R. In 2025, win rate dropped to 55.6% — not enough to offset losses that are 2.4x the size of wins. TP3 full-target hits dropped from 9 (IS) to just 3 (2025), while the strategy mostly captured small partial profits (6 TP1+SL exits) before reversing into SL.

**Notable:** March 2025 was catastrophic — three consecutive SL hits totaling -324 pips (trades #6-8). GBP_JPY exhibited wider ranges in 2025 that exceeded the strategy's stop placement.

### 3.2 S9_Filtered — Compressed Winners

**What changed:** Average win size collapsed from +50.3p (IS) / +65.2p (OOS) to +32.7p in 2025. The biggest single win shrank from +144.8p / +128.3p to just +63.5p. Meanwhile, losses remained similar (-54.9p vs -56.3p IS).

**Root cause:** ATR on GBP_AUD decreased from 0.0028 (IS) to 0.0023 (2025), indicating tighter ranges. In lower-volatility conditions, the strategy still enters at the same confidence level but the take-profit targets are harder to reach. Most wins exit at TP1 (7 of 10 wins were TP1+SL partials) rather than running to TP3.

**Notable:** The OOS period (9 trades, PF 2.26) was likely a small-sample anomaly rather than representative performance. The IS period (42 trades, PF 1.31) was closer to reality.

### 3.3 S3 — Surviving Through Bigger Winners

**What changed:** Win rate dropped from 53.6% (IS) to 42.9% (2025), but the R:R improved from 1.06 to 1.40. Average win grew from +43.6p to +68.8p. TP3 full-target hits increased from 15.5% of trades (13/84 IS) to 22.9% (8/35 in 2025).

**Why it survived:** Key level breakout is based on support/resistance — a structural market feature that persists across regimes. When breakouts occur, they tend to produce genuinely large moves. The strategy lost more often in 2025, but its winners ran further, keeping it marginally profitable.

**Concern:** PF declined from 1.22 (IS) to 1.05 (2025), and the strategy is heavily dependent on a few large winners. Removing the top 3 trades (Apr 3: +138p, Jan 10: +106p, Feb 5: +106p) would make 2025 negative. This concentration risk warrants caution.

### 3.4 S8_OB — Win Rate Reversion

**What changed:** Win rate in 2025 (31.6%) reverted to IS levels (37.3%). The OOS win rate of 52.9% was the anomaly. Average win also compressed from +31.7p (IS) to +17.1p (2025), and biggest win shrank from +84.7p to +35.4p.

**Root cause:** ATR on GBP_USD dropped from 0.0014 (IS) to 0.0010 (2025) — lower volatility compressed the profit potential. The strategy's order block patterns still generate signals at the same rate, but the displacement moves after retests are smaller. Long trades were especially poor (25% WR in 2025 vs 28% IS).

**Notable:** Multiple instances of rapid re-entry after SL (e.g., Feb 18: three trades in one day, Sep 23: three trades in one day), suggesting the strategy doesn't adapt well to choppy conditions.

### 3.5 Portfolio-Level Observations

**Correlation risk:** S7_Tight and S3 both trade GBP_JPY H1. In months where GBP_JPY was unfavorable (e.g., March 2025), both strategies suffered simultaneously, amplifying portfolio drawdowns.

**Regime dependency:** Three of four strategies showed material degradation in 2025. The edges identified during 2021-2023 development do not appear to be persistent. Only the most fundamentally grounded pattern (key level breakout) retained marginal profitability.

**Generalization score paradox:** The strategies with the highest Phase 2 generalization scores (S9: 1.712, S8: 1.397, S7: 1.272) all failed in 2025. S3, with the lowest score (1.068), was the sole survivor. High generalization scores during a favorable OOS period may indicate regime sensitivity rather than robustness.

---

## 4. Conclusions

1. **Only S3 (Key Level Breakout) demonstrated forward viability**, and even then with marginal profitability (PF 1.05) and high concentration risk.

2. **S7_Tight and S9_Filtered failed due to poor risk:reward structure** — both strategies cut winners short (TP1 partials) while allowing full SL losses. This only works with very high win rates, which proved to be regime-dependent.

3. **S8_OB failed due to lower volatility** — the GBP_USD ATR decreased ~30% from IS levels, compressing both winning and losing magnitudes but disproportionately affecting win rate.

4. **The portfolio as a whole lost 373 pips in 2025** with a PF of 0.83, demonstrating that Phase 2 validation alone is insufficient for live deployment.

5. **Recommendation:** The Phase 2 portfolio is not ready for live trading. Future strategy development should prioritize (a) structural edge patterns over statistical edge, (b) robust R:R profiles that don't depend on high win rates, and (c) multi-year forward testing before deployment.
