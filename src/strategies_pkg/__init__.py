from .s1_ma_breakout import S1_MA_Breakout
# S2 disabled: 20-25% win rate, 26 consecutive losses, needs full redesign
# from .s2_vwap_reversal import S2_VWAP_Reversal
from .s3_key_level_breakout import S3_KeyLevel_Breakout
from .s4_ema_ribbon import S4_EMA_Ribbon
from .s5_momentum_exhaustion import S5_Momentum_Exhaustion
from .s6_ema_bounce import S6_EMA_Bounce
from .s7_liquidity_sweep import S7_Liquidity_Sweep
from .s8_order_block import S8_Order_Block
from .s9_london_session import S9_London_Session
from .s17_bb_rejection import S17_BB_Rejection
from .s18_failed_breakout import S18_FailedBreakout
from .s19_vwap_deviation import S19_VWAPDeviationMeanReversion
from .s20_range_compression import S20_RangeCompression
from .s21_ema_ribbon_bounce import S21_EMA_RibbonBounce

STRATEGIES = {
    1: S1_MA_Breakout,
    # 2: S2_VWAP_Reversal,  # DISABLED
    3: S3_KeyLevel_Breakout,
    4: S4_EMA_Ribbon,
    5: S5_Momentum_Exhaustion,
    6: S6_EMA_Bounce,
    7: S7_Liquidity_Sweep,
    8: S8_Order_Block,
    9: S9_London_Session,
    17: S17_BB_Rejection,
    18: S18_FailedBreakout,
    19: S19_VWAPDeviationMeanReversion,
    20: S20_RangeCompression,
    21: S21_EMA_RibbonBounce,
}

# Which pairs each strategy trades
# Allowed universe: GBP_AUD, EUR_AUD, EUR_CAD, GBP_CAD, GBP_USD, EUR_USD
# Removed: EUR_GBP (PF 0.38-0.43), EUR_NZD (S1 lost $25k)
STRATEGY_PAIRS = {
    1: ["GBP_AUD", "EUR_AUD", "EUR_CAD", "GBP_CAD"],
    # 2: DISABLED
    3: ["GBP_JPY", "USD_JPY", "GBP_USD"],
    4: ["GBP_AUD", "EUR_AUD", "GBP_JPY"],
    5: ["GBP_AUD", "EUR_AUD", "GBP_JPY", "USD_JPY", "GBP_USD"],
    6: ["GBP_AUD"],  # Initial test — expand to EUR_AUD, GBP_USD if passing
    7: ["GBP_USD", "GBP_JPY", "EUR_AUD"],
    8: ["GBP_USD", "EUR_AUD", "GBP_JPY"],
    9: ["GBP_USD", "EUR_AUD", "GBP_JPY"],
    17: ["EUR_USD", "GBP_USD", "GBP_JPY", "GBP_AUD"],
    18: ["EUR_USD", "GBP_USD", "GBP_JPY", "GBP_AUD"],
    19: ["EUR_USD", "GBP_USD", "GBP_JPY", "GBP_AUD"],
    20: ["EUR_USD", "GBP_USD", "GBP_JPY", "GBP_AUD"],
    21: ["EUR_USD", "GBP_USD", "GBP_JPY", "GBP_AUD"],
}

# Primary and filter timeframes
STRATEGY_TIMEFRAMES = {
    1: {"primary": "M15", "filter": "H1"},
    # 2: DISABLED
    3: {"primary": "H1", "filter": None},  # Uses internal key level detection
    4: {"primary": "M15", "filter": "H1"},
    5: {"primary": "M15", "filter": "H1"},
    6: {"primary": "M15", "filter": "H1"},
    7: {"primary": "H1", "filter": None},   # H1 primary, internal HTF via htf_data
    8: {"primary": "H1", "filter": None},   # H1 primary, internal HTF via htf_data
    9: {"primary": "H1", "filter": None},   # H1 primary, internal HTF via htf_data
    17: {"primary": "M15", "filter": "H1"},
    18: {"primary": "M15", "filter": "H1"},
    19: {"primary": "M15", "filter": "H1"},
    20: {"primary": "M15", "filter": "H1"},
    21: {"primary": "M15", "filter": "H1"},
}
