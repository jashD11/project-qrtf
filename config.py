import hashlib
from dataclasses import dataclass, field
from typing import Any

# Toggle to "PRODUCTION_ML" to activate the Phase 2 live-data tree pipeline.
# "SANDBOX" keeps all Phase 1 GBM simulation paths active and unchanged.
MODE: str = "SANDBOX"

# Paths to historical NSE OHLCV datasets (used in PRODUCTION_ML mode only).
DATA_DAILY_CSV: str = "data/daily_ohlcv.csv"
DATA_15MIN_CSV: str = "data/15min_ohlcv.csv"


@dataclass
class _LGBMConfig:
    learning_rate: float = 0.05
    n_estimators: int = 100
    max_depth: int = -1      # -1 = no limit (LightGBM default)
    num_leaves: int = 31


@dataclass
class _XGBConfig:
    learning_rate: float = 0.05
    n_estimators: int = 100
    max_depth: int = 6


@dataclass
class _RFConfig:
    n_estimators: int = 100
    max_depth: int = 15
    min_samples_split: int = 5


@dataclass
class _MLConfig:
    lgbm: _LGBMConfig = field(default_factory=_LGBMConfig)
    xgb: _XGBConfig = field(default_factory=_XGBConfig)
    rf: _RFConfig = field(default_factory=_RFConfig)


ML_CONFIG: _MLConfig = _MLConfig()


@dataclass
class StrategyConfig:
    is_simulation: bool
    market_type: str        # 'stable_uptrend' | 'stable_downtrend' | 'stable_volatile' | 'volatile_downtrend'
    lookback_period: int    # momentum lookback in days (e.g. 5, 10, 20)
    hmm_states: int         # number of HMM hidden states (e.g. 2, 3)
    execution_style: str = "long_only"  # 'long_only' | 'long_short' | 'dynamic_tilt'
    top_n: int = 3          # number of stocks in the long leg
    bottom_n: int = 3       # number of stocks in the short leg (long_short / dynamic_tilt)
    frequency: str = "daily"

    @property
    def strategy_id(self) -> str:
        _abbrev: dict[str, str] = {"long_only": "LO", "long_short": "LS", "dynamic_tilt": "DT"}
        style_abbrev: str = _abbrev.get(self.execution_style, self.execution_style[:2].upper())
        prefix: str = (
            f"STRAT_{self.market_type.upper()}"
            f"_{style_abbrev}"
            f"_L{self.lookback_period}"
            f"_HMM{self.hmm_states}"
        )

        # All parameters included so every distinct permutation hashes uniquely
        param_string: str = (
            f"{self.is_simulation}_{self.market_type}_{self.lookback_period}"
            f"_{self.hmm_states}_{self.execution_style}"
            f"_{self.top_n}_{self.bottom_n}_{self.frequency}"
        )
        param_hash: str = hashlib.md5(param_string.encode()).hexdigest()[:8]

        return f"{prefix}_{param_hash}"
