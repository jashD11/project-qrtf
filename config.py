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
class _RegimeConfig:
    """Tier 2 market-regime HMM (PRODUCTION_ML). See src/production_ml/tier2_regime.py."""
    # NIFTY-50 = the market proxy: canonical barometer with full continuous
    # history here (2504 days). NIFTY-500 is only ~58% covered in this dataset
    # (a multi-year gap), so it is unusable; NIFTY-100 is the broader fallback.
    market_index: str = "NIFTY-50"
    index_parquet: str = "data/15min_index.parquet"
    stock_parquet: str = "data/daily_ohlcv.parquet"
    vol_window: int = 20                       # realized-vol lookback (trading days)
    corr_window: int = 20                      # rolling avg-pairwise-corr lookback
    zscore_min_periods: int = 252              # expanding causal z-score burn-in
    train_days: int = 504                      # walk-forward fit window (trading days)
    refit_every: int = 63                      # refit cadence / decode block (trading days)
    n_iter: int = 200                          # HMM EM iterations
    random_state: int = 42                     # reproducible fits
    # Calibration knobs (see docs/phase2_design_decisions.md §3):
    transmat_stickiness: float = 10.0          # diagonal pseudo-counts on the HMM
                                               # transition prior → more persistent
                                               # regime spells (a turnover knob; it
                                               # does NOT set the de-risk frequency).
    panic_threshold: float = 0.85              # de-risk GATE quantile. HMM posteriors
                                               # saturate (persistent vol), so the gate
                                               # is a causal trailing percentile on a
                                               # continuous stress score: Panic = the
                                               # top (1 - panic_threshold) most-stressed
                                               # days. 0.85 → ~15% Panic. → 1.0 = rarer.


@dataclass
class _ExecutionConfig:
    """
    Tier 3 (PRODUCTION_ML) capital-allocation constants. See
    src/production_ml/tier3_execution.py and docs/phase3_design_requirements.md.

    The 130/30 leverage split is a [FIXED] prior, NOT a sweep axis: it is a
    smooth/monotone exposure dial (in a bull sample 140/40 always "wins", in a
    bear 120/20 "wins"), so optimizing it just fits the sample's directional
    drift — textbook overfitting. It is chosen from the net/gross exposure budget
    and frozen here. Only ``dynamic_tilt`` reads these; long_only/long_short gate
    to cash on Panic instead.
    """
    calm_long_lev: float = 1.30    # [FIXED] Calm 130/30: 1.3x long leg
    calm_short_lev: float = 0.30   # [FIXED] Calm 130/30: 0.3x protective short
    panic_long_lev: float = 1.00   # Panic dollar-neutral: 1.0x long
    panic_short_lev: float = 1.00  # Panic dollar-neutral: 1.0x short (leg activates)


@dataclass
class _CostConfig:
    """
    Tier 3 transaction-cost model (PRODUCTION_ML). See
    src/production_ml/tier3_execution.py.

    OFF by default: current runs are gross (zero cost) so the raw edge is visible
    first; flip ``apply_costs`` on once the cost assumptions below are settled.
    Costs are turnover-based — ``oneway_bps`` is charged on notional traded
    (turnover = Sum|dw|), so a full round-trip pays it twice (once in, once out).
    """
    apply_costs: bool = False          # master switch — gross returns while False
    oneway_bps: float = 10.0           # commission + slippage + half-spread, one
                                       # way, per unit notional traded (placeholder)
    short_borrow_bps_annual: float = 50.0  # annual borrow on the short leg's gross


@dataclass
class _DSRConfig:
    """
    Tier 4 Deflated-Sharpe credibility gate. See
    src/production_ml/tier4_dsr_gate.py and docs/phase3_design_requirements.md §0.
    """
    benchmark_sharpe: float = 0.0      # SR* floor for the plain PSR (per-period)
    dsr_threshold: float = 0.95        # DSR pass line (P(true SR > deflated SR*))
    common_frequency: str = "daily"    # all columns resampled here before scoring
    trials_override: int | None = None  # N for the multiple-testing deflation;
                                        # None => ledger column count. Set to the
                                        # true number of cells searched (e.g. 12).


@dataclass
class _MLConfig:
    lgbm: _LGBMConfig = field(default_factory=_LGBMConfig)
    xgb: _XGBConfig = field(default_factory=_XGBConfig)
    rf: _RFConfig = field(default_factory=_RFConfig)
    regime: _RegimeConfig = field(default_factory=_RegimeConfig)
    execution: _ExecutionConfig = field(default_factory=_ExecutionConfig)
    cost: _CostConfig = field(default_factory=_CostConfig)
    dsr: _DSRConfig = field(default_factory=_DSRConfig)


ML_CONFIG: _MLConfig = _MLConfig()


# --------------------------------------------------------------------------- #
# PRODUCTION_ML strategy sweep — the committed grid & sensitivity bands.
# Source of truth for run_pipeline_ml.py (phase3 R1: the grid is an explicit
# Cartesian product enumerated here, not an implicit accident of config).
# See docs/phase3_design_requirements.md.
# --------------------------------------------------------------------------- #
# [SWEEP] structural forks the orchestrator iterates.
SWEEP_FREQUENCIES: list[str] = ["15min", "30min", "60min", "daily"]
SWEEP_EXECUTION_STYLES: list[str] = ["long_only", "long_short", "dynamic_tilt"]

# hmm_states is deliberately NOT a sweep axis (yet): today the execution tier
# gates only on the continuous, n_states-independent ``panic`` score, so a
# 2-state vs 3-state HMM produces return-identical cells. Frozen at 2 until
# state-conditional execution exists — see docs/phase3_deferred_hmm.md.
HEADLINE_HMM_STATES: int = 2

# [SENSITIVITY] opt-in robustness scans — a SEPARATE entrypoint, never folded
# into the headline grid (phase3 R3). Report the whole band, never the peak.
# lookback_period is intentionally absent: it is a Phase-1 GKX knob with no
# analog in the full-panel tree pipeline (see docs/phase3_deferred_hmm.md).
SENSITIVITY_BANDS: dict[str, list] = {
    "panic_threshold": [0.80, 0.85, 0.90, 0.95],
    "decile_pct": [0.05, 0.10, 0.15, 0.20],
}


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
    decile_pct: float = 0.10  # PRODUCTION_ML only — book concentration as a
                              # fraction of the daily cross-section (SANDBOX uses
                              # top_n/bottom_n instead). Matches tier1_trees.DECILE_PCT.

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
            f"_{self.top_n}_{self.bottom_n}_{self.frequency}_{self.decile_pct}"
        )
        param_hash: str = hashlib.md5(param_string.encode()).hexdigest()[:8]

        return f"{prefix}_{param_hash}"
