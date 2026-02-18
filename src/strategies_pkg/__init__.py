from .s1_ma_breakout import S1_MA_Breakout
# S2 disabled: 20-25% win rate, 26 consecutive losses, needs full redesign
# from .s2_vwap_reversal import S2_VWAP_Reversal
from .s3_key_level_breakout import S3_KeyLevel_Breakout
from .s4_ema_ribbon import S4_EMA_Ribbon
from .s5_momentum_exhaustion import S5_Momentum_Exhaustion
from .s6_ema_bounce import S6_EMA_Bounce

STRATEGIES = {
    1: S1_MA_Breakout,
    # 2: S2_VWAP_Reversal,  # DISABLED
    3: S3_KeyLevel_Breakout,
    4: S4_EMA_Ribbon,
    5: S5_Momentum_Exhaustion,
    6: S6_EMA_Bounce,
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
}

# Primary and filter timeframes
STRATEGY_TIMEFRAMES = {
    1: {"primary": "M15", "filter": "H1"},
    # 2: DISABLED
    3: {"primary": "H1", "filter": None},  # Uses internal key level detection
    4: {"primary": "M15", "filter": "H1"},
    5: {"primary": "M15", "filter": "H1"},
    6: {"primary": "M15", "filter": "H1"},
}
