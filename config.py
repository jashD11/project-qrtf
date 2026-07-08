import hashlib
from dataclasses import dataclass, field
from typing import Any

# Toggle to "PRODUCTION_ML" to activate the Phase 2 live-data tree pipeline.
# "SANDBOX" keeps all Phase 1 GBM simulation paths active and unchanged.
MODE: str = "SANDBOX"

# Paths to historical NSE OHLCV datasets (used in PRODUCTION_ML mode only).
# Legacy single-file selectors kept for the data_scraping dry-run; the live
# frequency selector is FREQ_REGISTRY below.
DATA_DAILY_CSV: str = "data/daily_ohlcv.csv"
DATA_15MIN_CSV: str = "data/15min_ohlcv.csv"


# --------------------------------------------------------------------------- #
# Frequency registry — the "pick a frequency flag" axis of the strategy sweep.
#
# resample_bars.py derives four aligned OHLCV Parquets from the one 15-min
# source of truth. Each frequency owns a coherent bar series; a single flag
# (``StrategyConfig.frequency``) selects which one the loader, feature factory,
# and tree engine consume. See docs/phase2_design_decisions.md §2.
#
# Unit convention (design decision D2):
#   - Feature lookbacks are counted in **bars** (one row of the selected
#     frequency). The same integer horizons auto-scale per frequency, so warm-up
#     stays small intraday. Feature names carry a bar-neutral ``_Nb`` suffix.
#   - Walk-forward train/predict windows are counted in **trading days**, so the
#     trees always get enough rows regardless of how many bars fall in a day.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _FreqSpec:
    parquet: str            # OHLCV Parquet for this frequency
    bars_per_day: int       # approx bars in one NSE session (09:15–15:30 IST)
    train_days: int         # walk-forward fit window, in unique trading days
    predict_days: int       # walk-forward out-of-sample window, in trading days
    has_institutional: bool # daily delivery joins as-is; intraday = t-1 lagged
                            # broadcast (D1). False → institutional family dropped.


# 15-min is the source itself; 30/60min/daily are session-aware resamples of it.
# All four share one timeline, so the frequency axis compares like-for-like.
FREQ_REGISTRY: dict[str, _FreqSpec] = {
    "15min": _FreqSpec("data/15min_ohlcv.parquet", 25, 504, 63, True),
    "30min": _FreqSpec("data/30min_ohlcv.parquet", 13, 504, 63, True),
    "60min": _FreqSpec("data/60min_ohlcv.parquet", 7, 504, 63, True),
    "daily": _FreqSpec("data/daily_ohlcv.parquet", 1, 504, 63, True),
}

DEFAULT_FREQUENCY: str = "daily"


def freq_spec(frequency: str) -> _FreqSpec:
    """Look up the _FreqSpec for a frequency flag, with a clear error if unknown."""
    try:
        return FREQ_REGISTRY[frequency]
    except KeyError:
        raise ValueError(
            f"frequency={frequency!r} not in FREQ_REGISTRY {sorted(FREQ_REGISTRY)}"
        ) from None


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
