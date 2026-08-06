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
    # Phase 4: the rebuilt point-in-time NSE panel (~500 names, 2013→2025) from
    # src/phase4_data/. Bhavcopy is daily-only, so there is no intraday counterpart —
    # acceptable, since every intraday cell already died on costs in Phase 3. The four
    # entries above are left untouched so the Phase 3 headline stays reproducible.
    # has_institutional=False: the delivery archive only starts in 2020 and
    # create_features drops rows with any NaN feature, so enabling it would silently
    # truncate a 12-year panel to 5. Same 17 price-only features as Phase 3, which also
    # keeps the 68-name vs 500-name comparison like-for-like.
    "daily_nse500": _FreqSpec("data/nse500_daily_ohlcv.parquet", 1, 504, 63, False),
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
    # Parallel workers for RF. 15-min peaked at ~3.5 GB of 16 with n_jobs=4, so
    # cores (not RAM) are the constraint — pushed to 6 (of 8) for speed while
    # leaving headroom for the OS + OMP-pinned LGBM/XGB. n_jobs does NOT change a
    # RandomForest's output (per-tree seeds are deterministic), only speed/RAM.
    n_jobs: int = 6


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
class _NSECostConfig:
    """
    Tier 3 transaction-cost model (PRODUCTION_ML), grounded in NSE cash-equity
    DELIVERY charges — the strategy carries net positions overnight, so delivery
    rates apply, not intraday square-off. See src/production_ml/tier3_execution.py
    and docs/phase3_design_requirements.md.

    Each component is an itemized, citable line item in basis points of the
    notional traded on ONE side. ``compose_oneway_bps`` blends the asymmetric
    buy/sell legs (STT both sides, stamp buy-only, GST on the broker/exchange/SEBI
    base) into one symmetric one-way rate, since turnover (= Sum|dw|) is
    side-agnostic. Costs are charged on turnover, so a round-trip pays twice.
    """
    apply_costs: bool = True            # master switch — NET returns by default
                                        # (set False to recover gross reference)
    # --- itemized NSE delivery components (bps per side of traded notional) --- #
    brokerage_bps: float = 3.0          # institutional / discount, per side
    stt_delivery_bps: float = 10.0      # STT 0.10%, charged on BOTH buy and sell
    exchange_txn_bps: float = 0.30      # NSE cash txn charge (~0.00297%)
    sebi_bps: float = 0.01              # SEBI turnover fee (0.0001%)
    stamp_duty_bps_buy: float = 1.5     # stamp duty 0.015%, BUY side only (delivery)
    gst_pct: float = 18.0               # GST on (brokerage + exchange + SEBI)
    slippage_bps: float = 0.0           # FLAT market-impact / half-spread add-on. Kept
                                        # at 0: superseded by the per-name spread +
                                        # impact terms below (charge_per_name), which
                                        # price the same thing without a constant.
    short_borrow_bps_annual: float = 50.0  # annual borrow on the short leg's gross

    # --- size-dependent market impact (used by src/phase4_data/capacity.py) --- #
    # Square-root impact law, the shape docs/phase3_design_requirements.md §4.3 specifies
    # ("participation rate, spread, ADV, not a flat constant"):
    #
    #     impact_bps = 1e4 * impact_coef * sigma * sqrt(participation)
    #
    # `impact_coef` is THE discretionary knob in the whole cost stack — every other line
    # item is a citable statutory rate, this one is a modelling choice. 0.5 puts a 1%-of-
    # ADV trade in a 2%-daily-vol stock at ~10 bps, which is the right order for Indian
    # cash equities. Treat it as an assumption to be sensitivity-tested, not a fact.
    impact_coef: float = 0.5
    max_participation: float = 0.10     # share of ADV a single position may consume
    adv_window: int = 21                # trailing bars for the ADV estimate

    # ----------------------------------------------------------------------- #
    # Phase 4c A3/A4 — per-name execution cost, wired into the execution path.
    #
    # Phase 4b charged only the ~14.66 bps statutory stack: zero spread and zero
    # market impact, on a book turning over 0.45-0.82 per day with under 7 bps of
    # headroom. These four fields are the fix, and every one of them is OFF by
    # default — the Phase 2/3 and Phase 4b ledgers must reproduce bit-for-bit, so
    # new cost behaviour is opt-in exactly like the halt bridge.
    # ----------------------------------------------------------------------- #
    charge_per_name: bool = False       # master gate. False => flat statutory rate on
                                        # reduced turnover (the Phase 3/4b path).
                                        # True  => per-(bar,name) statutory + measured
                                        # half-spread + sqrt-law impact.
    aum_rupees: float = 1e7             # Rs 1 crore — the documented operating point.
                                        # Capacity does not bind here, so a failure at
                                        # Rs 1 cr is a failure at every larger size.
    spread_estimator: str = "cs"        # "cs" Corwin-Schultz (biased DOWN, ~6.0 bps
                                        # pooled) | "ar" Abdi-Ranaldo (biased UP, ~26.1
                                        # bps). The headline runs on "cs": if the edge
                                        # dies under the optimistic estimate, the
                                        # verdict does not depend on the choice.
    enforce_participation_cap: bool = False  # A4: cap each position at
                                        # max_participation of its ADV, redistributing
                                        # within the leg so leg gross stays 1.0.

    # --- B3 liquidity-tiered short borrow ---------------------------------- #
    # The flat 50 bps/yr assumes unlimited availability at a uniform price, which is
    # not how SLB works: borrow is cheap and deep for the large F&O names and scarce
    # and expensive down the liquidity ladder. Frozen a-priori from published SLB fee
    # ranges — a stated assumption, not a fitted parameter.
    tiered_borrow: bool = False         # opt-in, like everything else above
    borrow_bps_by_quartile: tuple[float, ...] = (25.0, 50.0, 100.0, 200.0)
                                        # by trailing-turnover quartile of the eligible
                                        # set, most liquid first

    @staticmethod
    def impact_bps(sigma: float, participation: float, coef: float = 0.5) -> float:
        """
        Market impact in bps for one trade, from daily vol and participation rate.

        Kept as a pure function so the capacity study and the (later) tier-3 integration
        share one formula instead of two copies that drift apart. Both arguments are
        fractions, not percentages: sigma=0.02 is 2% daily vol, participation=0.01 is 1%
        of average daily volume.
        """
        import math

        if participation <= 0 or sigma <= 0:
            return 0.0
        return 1e4 * coef * sigma * math.sqrt(participation)

    def compose_oneway_bps(self) -> float:
        """
        Effective symmetric one-way cost (bps) charged on turnover.

            buy  = brokerage + STT + exchange + SEBI + stamp_buy + GST(base) + slip
            sell = brokerage + STT + exchange + SEBI +           + GST(base) + slip
            oneway = (buy + sell) / 2   (a rebalance is ~half buy / half sell)

        At defaults this composes to ~14.7 bps (vs the old flat 10 placeholder).
        """
        gst_base: float = self.brokerage_bps + self.exchange_txn_bps + self.sebi_bps
        gst: float = gst_base * (self.gst_pct / 100.0)
        common: float = (
            self.brokerage_bps + self.stt_delivery_bps + self.exchange_txn_bps
            + self.sebi_bps + gst + self.slippage_bps
        )
        buy: float = common + self.stamp_duty_bps_buy
        sell: float = common
        return 0.5 * (buy + sell)


# --------------------------------------------------------------------------- #
# The trial ledger (Phase 4c D1) — the honest N for the DSR deflation.
#
# "A grid of N cells is N implicit backtests." N is not the number of columns in
# whatever file is being scored; it is the number of configurations the *program* has
# searched to arrive at the one being reported. Phase 4b deflated against N=3 and two
# cells passed. The tally below is that number, kept explicit and auditable here so it
# can be challenged line by line rather than asserted.
#
# Judgment calls, stated so they can be argued with:
#   - Phase 1 sandbox runs are EXCLUDED: synthetic data, a different question.
#   - A gross/net re-measurement of an identical configuration is NOT a new trial —
#     it is the same cell measured twice, not a new place to look.
#   - Sensitivity cells ARE counted. Searching them is searching them, regardless of
#     which ledger file they landed in.
# --------------------------------------------------------------------------- #
TRIAL_LEDGER: list[tuple[str, int, str]] = [
    # (source, cells counted, rationale)
    ("phase3 §3 net grid, 4 freq x 3 styles, 1b target", 12, "the headline search"),
    ("phase3 §2 gross, same 12 configs", 0, "same cells measured without costs"),
    ("phase3 §4 daily-only, 1b", 0, "a subset of the 12 above"),
    ("phase3 §5 daily-only, 5b", 3, "new target => new configurations"),
    ("phase3 §6 rebalance_buffer sensitivity (6 values)", 5, "mult=2.0 counted in §5"),
    ("phase3 §7.3 multi-scale panic gate", 1, "tested and rejected — still a search"),
    ("phase4b headline, daily_nse500 x 3 styles", 3, ""),
    ("phase4c frozen grid, 3 styles x 2 targets", 6, "docs/phase4c_plan.md §5"),
]
TRIALS_SEARCHED: int = sum(n for _, n, _ in TRIAL_LEDGER)   # = 30


@dataclass
class _DSRConfig:
    """
    Tier 4 Deflated-Sharpe credibility gate. See
    src/production_ml/tier4_dsr_gate.py and docs/phase3_design_requirements.md §0.
    """
    benchmark_sharpe: float = 0.0      # SR* floor for the plain PSR (per-period)
    dsr_threshold: float = 0.95        # DSR pass line (P(true SR > deflated SR*)).
                                        # FROZEN. Moving a threshold after seeing
                                        # results is the selection bias this gate
                                        # exists to correct.
    common_frequency: str = "daily"    # all columns resampled here before scoring
    trials_override: int | None = TRIALS_SEARCHED  # N for the multiple-testing
                                        # deflation (D1). None would fall back to the
                                        # ledger's column count, which is the number
                                        # of cells *reported*, not searched.

    # --- D2: Lo (2002) autocorrelation correction -------------------------- #
    # PSR/DSR assume i.i.d. daily returns. These returns are not: positions persist
    # for days under a multi-day target, and measured lag-1 autocorrelation on the
    # Phase 4b ledger is +0.09 to +0.14. Positive autocorrelation makes the naive
    # sqrt(252) annualization overstate the Sharpe, so the correction is uniform and
    # monotone AGAINST every strategy.
    autocorr_adjust: bool = True
    autocorr_horizon: int = 21          # variance-ratio horizon (trading days) behind
                                        # the Lo scaling. Long enough to contain a
                                        # whole position life at the slowest target in
                                        # the grid, short enough to keep the estimator
                                        # tight (sd ~0.05 at this sample length).


@dataclass
class _Phase4Config:
    """
    Phase 4 point-in-time database (src/phase4_data/). Additive: nothing here affects
    a Phase 2/3 run, which keeps using the 68-name Drive panel and leaves every field
    below unread.
    """
    # Acquisition window. 2013-01-01 is the later of two hard archive boundaries —
    # ISIN (the survivorship key) starts in 2012, but the index archive, which Tier 2
    # needs across the whole panel, only starts 2013-01-02.
    start_date: str = "2013-01-01"
    end_date: str = "2025-06-30"
    request_delay_s: float = 2.5       # empirically reliable pacing; see the module

    raw_dir: str = "data/bhavcopy"
    panel_parquet: str = "data/bhavcopy/panel_daily.parquet"
    lifecycle_csv: str = "data/bhavcopy/lifecycle.csv"
    index_parquet: str = "data/bhavcopy/index_daily.parquet"
    ohlcv_parquet: str = "data/nse500_daily_ohlcv.parquet"
    universe_mask_parquet: str = "data/nse500_universe_mask.parquet"

    top_n: int = 500                   # names per quarterly rebalance
    universe_lookback: int = 252       # trailing window for the liquidity rank
    universe_min_traded: int = 200     # of the lookback, days that must have traded

    # Terminal return for a name that leaves the exchange without evidence of a
    # buyout (Shumway 1997). Declared a-priori as a {0, -0.30, -1.00} sensitivity on
    # a SEPARATE ledger — freeze-before-test, like the rebalance-buffer sweep.
    delisting_return: float = -0.30

    # --- Phase 4c cost/eligibility panels (src/phase4_data/) ---------------- #
    adv_parquet: str = "data/bhavcopy/adv_daily.parquet"
    sigma_parquet: str = "data/bhavcopy/sigma_daily.parquet"
    spread_cs_parquet: str = "data/bhavcopy/spread_daily_cs.parquet"
    spread_ar_parquet: str = "data/bhavcopy/spread_daily_ar.parquet"
    # Point-in-time set of names carrying a live single-stock future, used as the
    # availability proxy for SLB borrow. An UPPER bound on shortability — real SLB is
    # thinner — because NSE publishes no historical SLB-eligibility archive.
    shortable_mask_parquet: str = "data/bhavcopy/shortable_mask.parquet"
    # Daily NIFTY-50 dividend yield (%/yr), read from the index archive's `Div Yield`
    # column. Phase 4b's long_only verdict turned on an *assumed* flat 1.3%/yr; this
    # replaces the assumption with the measurement.
    div_yield_parquet: str = "data/bhavcopy/div_yield_daily.parquet"


@dataclass
class _MLConfig:
    lgbm: _LGBMConfig = field(default_factory=_LGBMConfig)
    xgb: _XGBConfig = field(default_factory=_XGBConfig)
    rf: _RFConfig = field(default_factory=_RFConfig)
    regime: _RegimeConfig = field(default_factory=_RegimeConfig)
    execution: _ExecutionConfig = field(default_factory=_ExecutionConfig)
    cost: _NSECostConfig = field(default_factory=_NSECostConfig)
    dsr: _DSRConfig = field(default_factory=_DSRConfig)
    phase4: _Phase4Config = field(default_factory=_Phase4Config)


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
    "rebalance_buffer_mult": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
}


# --------------------------------------------------------------------------- #
# The Phase 4c frozen grid (docs/phase4c_plan.md §5) — committed BEFORE the run.
#
# Six cells: 3 execution styles x 2 targets, on the 500-name point-in-time panel.
# Everything else is fixed, and every knob below is stated here rather than passed
# on a command line, so what ran is recoverable from the repo alone.
#
# ``long_only`` has no short leg, so Track B cannot touch it — it is the control
# that isolates how much of the damage is Track A (spread + impact) by itself.
# --------------------------------------------------------------------------- #
PHASE4C_FREQUENCY: str = "daily_nse500"
PHASE4C_STYLES: list[str] = ["long_only", "long_short_slb", "dynamic_tilt_slb"]
PHASE4C_TARGETS: list[str] = ["tgt_fwd_logret_5b", "tgt_fwd_logret_21b"]
PHASE4C_LEDGER: str = "data/trial_database/phase4c_dsr_matrix.parquet"

# The cost regime the frozen grid runs under. ``spread_estimator="cs"`` is the
# deliberately CHARITABLE choice: Corwin-Schultz pools to ~6.0 bps in-universe against
# Abdi-Ranaldo's ~26.1, and on controlled synthetic data with a known planted spread it
# is the one that under-reads. If the edge dies under the optimistic estimate, the
# verdict does not depend on which estimator was picked.
PHASE4C_COST_OVERRIDES: dict[str, object] = {
    "charge_per_name": True,
    "enforce_participation_cap": True,
    "tiered_borrow": True,
    "spread_estimator": "cs",
    "aum_rupees": 1e7,      # Rs 1 crore — capacity does not bind here, so a failure
                            # at this size is a failure at every larger one
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
    rebalance_buffer_mult: float = 2.0  # PRODUCTION_ML no-trade hysteresis: a name
                                        # enters a leg at decile_pct but is only
                                        # evicted once it drifts past
                                        # decile_pct*mult. >=1.0; 1.0 disables the
                                        # buffer (enter==exit). Turnover control.
    target_col: str = "tgt_fwd_logret_1b"  # PRODUCTION_ML tree learning label. Part of
                                        # the strategy's identity, not a runtime
                                        # option: a model trained on a 5-day label is
                                        # a different strategy from one trained on a
                                        # 1-day label, and Phase 4c's frozen grid
                                        # searches both (so both are paid for in N).
                                        # Phase 3 kept them apart by writing separate
                                        # ledger FILES, which does not scale to a grid
                                        # that varies the target inside one ledger.

    @property
    def strategy_id(self) -> str:
        _abbrev: dict[str, str] = {
            "long_only": "LO", "long_short": "LS", "dynamic_tilt": "DT",
            # Phase 4c: the same two books with the short leg restricted to
            # SLB-borrowable names. Distinct ids — they are different strategies,
            # not corrections to the unrestricted ones.
            "long_short_slb": "LSB", "dynamic_tilt_slb": "DTB",
        }
        style_abbrev: str = _abbrev.get(self.execution_style, self.execution_style[:2].upper())
        # "tgt_fwd_logret_5b" -> "T5B"; kept in the human-readable prefix so a ledger
        # column says which label trained it without decoding the hash.
        tgt_abbrev: str = "T" + self.target_col.rsplit("_", 1)[-1].upper()
        prefix: str = (
            f"STRAT_{self.market_type.upper()}"
            f"_{style_abbrev}"
            f"_L{self.lookback_period}"
            f"_HMM{self.hmm_states}"
            f"_{tgt_abbrev}"
        )

        # All parameters included so every distinct permutation hashes uniquely
        param_string: str = (
            f"{self.is_simulation}_{self.market_type}_{self.lookback_period}"
            f"_{self.hmm_states}_{self.execution_style}"
            f"_{self.top_n}_{self.bottom_n}_{self.frequency}_{self.decile_pct}"
            f"_{self.rebalance_buffer_mult}_{self.target_col}"
        )
        param_hash: str = hashlib.md5(param_string.encode()).hexdigest()[:8]

        return f"{prefix}_{param_hash}"
