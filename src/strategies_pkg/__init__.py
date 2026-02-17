from .s1_ma_breakout import S1_MA_Breakout
from .s2_vwap_reversal import S2_VWAP_Reversal
from .s3_key_level_breakout import S3_KeyLevel_Breakout
from .s4_ema_ribbon import S4_EMA_Ribbon
from .s5_momentum_exhaustion import S5_Momentum_Exhaustion

STRATEGIES = {
    1: S1_MA_Breakout,
    2: S2_VWAP_Reversal,
    3: S3_KeyLevel_Breakout,
    4: S4_EMA_Ribbon,
    5: S5_Momentum_Exhaustion,
}

# Which pairs each strategy trades
STRATEGY_PAIRS = {
    1: ["GBP_AUD", "EUR_AUD", "EUR_CAD", "EUR_NZD"],
    2: ["GBP_USD", "EUR_USD", "GBP_JPY", "USD_JPY"],
    3: ["GBP_JPY", "USD_JPY", "GBP_USD", "EUR_GBP"],
    4: ["GBP_AUD", "EUR_AUD", "EUR_GBP"],
    5: ["GBP_AUD", "EUR_AUD", "EUR_GBP", "GBP_CAD", "EUR_CAD"],
}

# Primary and filter timeframes
STRATEGY_TIMEFRAMES = {
    1: {"primary": "M15", "filter": "H1"},
    2: {"primary": "M15", "filter": None},
    3: {"primary": "H1", "filter": None},  # Uses internal key level detection
    4: {"primary": "M15", "filter": "H1"},
    5: {"primary": "M15", "filter": "H1"},
}
