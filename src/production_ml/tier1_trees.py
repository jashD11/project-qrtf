"""
Phase 2 Tier 1 — Tree-based Machine Learning alpha engine.

Replaces the Phase 1 GKX momentum ranker (``src/tier1_gkx.py``) when
``config.MODE == "PRODUCTION_ML"``.  Consumes the cross-sectionally normalized
feature matrix emitted by ``src/feature_creator.py`` (MultiIndex
('date', 'ticker'), 19 features + forward-return targets) and produces a
long/short decile signal matrix via a walk-forward, multi-model tree ensemble.

Design contract
    - Walk-forward validation only: models are fit on a trailing ``train_window``
      block and used to predict the *next*, strictly out-of-sample
      ``predict_window`` block. The window then slides forward by one predict
      block. Training data is never shuffled and never overlaps the block it
      scores, so there is zero look-ahead / temporal leakage.
    - Three independent learners (LightGBM, XGBoost, RandomForest) read their
      hyperparameters from ``config.ML_CONFIG``. Their per-asset predictions are
      averaged into a single ensemble Alpha Score (single-model mode available
      via the ``model_choice`` toggle).
    - Cross-sectional decile allocation: each active trading day is bucketed
      independently — the top 10% of Alpha Scores form the long mask (+1) and
      the bottom 10% form the short mask (-1); everything else is flat (0).

The engine emits three aligned wide DataFrames (index = date, columns = ticker):
``alpha_scores``, ``long_mask`` (1/0) and ``short_mask`` (-1/0).
"""

import gc
import os
import sys
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

# LightGBM and XGBoost each vendor their own OpenMP runtime. On macOS both
# libomp copies load into the same process and collide, which segfaults / hangs
# as soon as the second library trains. Pinning a single OMP thread and allowing
# the duplicate runtime (set BEFORE the imports below) makes the trio coexist;
# RandomForest reclaims parallelism via joblib's separate process pool.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

# Make the repo root importable whether this file is run as ``python
# src/production_ml/tier1_trees.py`` (sys.path[0] == src/production_ml/) or as
# ``python -m src.production_ml.tier1_trees`` / imported by an orchestrator from
# the repo root. __file__ is src/production_ml/tier1_trees.py, so the repo root
# is three directories up.
_ROOT: str = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from src.production_ml.feature_creator import (
    DATE_LEVEL,
    TICKER_LEVEL,
    FEATURE_COLUMNS,
    PRICE_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    TARGET_HORIZONS,
    present_feature_columns,
)

# --- Walk-forward window sizing (in unique TRADING DAYS, not bars — D2) ----- #
# Windows are counted in calendar trading days so the tree gets enough history
# regardless of how many intraday bars fall in a day. Per-frequency values live
# in config.FREQ_REGISTRY; these are the daily defaults.
TRAIN_WINDOW: Final[int] = 504   # ~2 trading years of history to fit on
PREDICT_WINDOW: Final[int] = 63  # ~1 trading quarter scored out-of-sample

# --- Default learning target and decile width ------------------------------ #
DEFAULT_TARGET: Final[str] = "tgt_fwd_logret_1b"
DECILE_PCT: Final[float] = 0.10  # top / bottom 10% of the universe

# --- Shared RNG seed so every run is reproducible -------------------------- #
_SEED: Final[int] = 42

# --- Fit diagnostic: min cross-section for a day's IC to be well-defined ---- #
_IC_MIN_NAMES: Final[int] = 5


@dataclass
class WalkForwardResult:
    """Container for the full out-of-sample output of a walk-forward run."""
    alpha_scores: pd.DataFrame  # (date x ticker) ensemble Alpha Score
    long_mask: pd.DataFrame     # (date x ticker) 1 = long, 0 = flat
    short_mask: pd.DataFrame    # (date x ticker) -1 = short, 0 = flat
    ic_diagnostics: pd.DataFrame | None = None  # per-model rank-IC fit report
                                                # (read-only; see _compute_ic_diagnostics)


# --------------------------------------------------------------------------- #
# Fit diagnostic — cross-sectional rank IC (READ-ONLY)
# --------------------------------------------------------------------------- #
# IMPORTANT: the IC below is computed AFTER the out-of-sample predictions are
# made, purely as a readout of predictive skill. It never feeds back into signal
# construction, decile allocation, weights, or config/hyperparameter selection —
# so it introduces zero look-ahead. It must NOT be used to cherry-pick cells or
# tune hyperparameters on OOS data: that is the selection-on-test-statistic bias
# the DSR gate exists to correct. Legitimate hyperparameter tuning belongs on an
# INNER-validation slice of the training window, not on this OOS IC.
def _compute_ic_diagnostics(
    pred_panel: pd.DataFrame, realized: pd.Series
) -> pd.DataFrame:
    """
    Per-model cross-sectional rank Information Coefficient over the OOS panel.

    Args:
        pred_panel: (date, ticker) predictions, one column per learner plus an
                    ``ensemble`` column.
        realized:   (date, ticker) realized forward-return label for the same rows.

    For each model column, the daily IC is the Spearman rank correlation between
    that day's predictions and realized returns across the cross-section (days
    with fewer than ``_IC_MIN_NAMES`` valid pairs are skipped). Returns a frame
    indexed by model name (``ensemble`` last) with columns
    [mean_IC, IC_IR, IC_t, hit_rate, n_days].
    """
    model_cols: list[str] = list(pred_panel.columns)
    joined: pd.DataFrame = pred_panel.join(realized.rename("_realized"), how="inner")
    joined = joined.dropna(subset=["_realized"])

    def _day_ic(group: pd.DataFrame) -> pd.Series:
        y: pd.Series = group["_realized"]
        out: dict[str, float] = {}
        for m in model_cols:
            pair: pd.DataFrame = pd.concat([group[m], y], axis=1).dropna()
            out[m] = (
                pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
                if len(pair) >= _IC_MIN_NAMES
                else np.nan
            )
        return pd.Series(out)

    daily_ic: pd.DataFrame = joined.groupby(level=DATE_LEVEL).apply(_day_ic)

    rows: list[dict] = []
    for m in model_cols:
        ic: pd.Series = daily_ic[m].dropna()
        n_days: int = int(ic.shape[0])
        mean_ic: float = float(ic.mean()) if n_days else float("nan")
        ic_std: float = float(ic.std(ddof=1)) if n_days > 1 else float("nan")
        ic_ir: float = (
            mean_ic / ic_std if np.isfinite(ic_std) and ic_std > 0 else float("nan")
        )
        ic_t: float = ic_ir * np.sqrt(n_days) if np.isfinite(ic_ir) else float("nan")
        hit: float = float((ic > 0).mean()) if n_days else float("nan")
        rows.append(
            dict(model=m, mean_IC=mean_ic, IC_IR=ic_ir, IC_t=ic_t,
                 hit_rate=hit, n_days=n_days)
        )

    out: pd.DataFrame = pd.DataFrame(rows).set_index("model")
    order: list[str] = [c for c in ("lgbm", "xgb", "rf", "ensemble") if c in out.index]
    return out.reindex(order)


def _print_ic_panel(ic: pd.DataFrame) -> None:
    """Pretty-print the per-model rank-IC fit diagnostic."""
    print("[tier1_trees] fit diagnostic — cross-sectional rank IC (pred vs realized fwd ret)")
    print(f"    {'model':<10}{'meanIC':>9}{'IC_IR':>8}{'IC_t':>8}{'hit%':>7}{'n_days':>8}")
    for m, row in ic.iterrows():
        hit_pct: float = row["hit_rate"] * 100.0
        print(f"    {m:<10}{row['mean_IC']:>9.4f}{row['IC_IR']:>8.3f}"
              f"{row['IC_t']:>8.2f}{hit_pct:>6.1f}%{int(row['n_days']):>8}")


class TreeAlphaEngine:
    """
    Multi-model tree ensemble driving Tier 1 alpha via walk-forward prediction.

    Parameters
    ----------
    target_col:
        Which forward-return label to regress on. Must be one of
        ``feature_creator.TARGET_COLUMNS``. Defaults to the 1-day forward log
        return.
    model_choice:
        ``"ensemble"`` (default) averages LightGBM + XGBoost + RandomForest.
        ``"lgbm"`` / ``"xgb"`` / ``"rf"`` select a single learner.
    train_window / predict_window:
        Size (in unique trading days) of the trailing fit block and the forward
        out-of-sample scoring block.
    decile_pct:
        Fraction of the universe allocated to each side of the book per day.
    """

    _VALID_CHOICES: Final[frozenset] = frozenset(
        {"ensemble", "lgbm", "xgb", "rf"}
    )

    def __init__(
        self,
        target_col: str = DEFAULT_TARGET,
        model_choice: str = "ensemble",
        train_window: int = TRAIN_WINDOW,
        predict_window: int = PREDICT_WINDOW,
        decile_pct: float = DECILE_PCT,
    ) -> None:
        if target_col not in TARGET_COLUMNS:
            raise ValueError(
                f"target_col={target_col!r} not in TARGET_COLUMNS={TARGET_COLUMNS}"
            )
        if model_choice not in self._VALID_CHOICES:
            raise ValueError(
                f"model_choice={model_choice!r} not in {sorted(self._VALID_CHOICES)}"
            )
        if not 0.0 < decile_pct < 0.5:
            raise ValueError(f"decile_pct must be in (0, 0.5), got {decile_pct}")

        self.target_col: str = target_col
        self.model_choice: str = model_choice
        self.train_window: int = train_window
        self.predict_window: int = predict_window
        self.decile_pct: float = decile_pct
        # Feature schema is resolved from the input frame at run time (17 vs 19),
        # so a price-only intraday panel and a full daily panel both work.
        self.feature_cols: list[str] = list(FEATURE_COLUMNS)

    @classmethod
    def from_frequency(
        cls,
        frequency: str = config.DEFAULT_FREQUENCY,
        target_col: str = DEFAULT_TARGET,
        model_choice: str = "ensemble",
        decile_pct: float = DECILE_PCT,
    ) -> "TreeAlphaEngine":
        """
        Build an engine whose walk-forward windows come from ``FREQ_REGISTRY``.

        The train/predict windows are in trading days (D2), so they are identical
        across frequencies unless overridden per frequency in the registry.
        """
        spec = config.freq_spec(frequency)
        return cls(
            target_col=target_col,
            model_choice=model_choice,
            train_window=spec.train_days,
            predict_window=spec.predict_days,
            decile_pct=decile_pct,
        )

    # ---------------------------------------------------------------------- #
    # Model factory
    # ---------------------------------------------------------------------- #
    def _build_models(self) -> dict[str, object]:
        """
        Instantiate a fresh set of learners from ``config.ML_CONFIG``.

        Fresh estimators are built for every walk-forward fold so no state
        bleeds across time windows.
        """
        ml = config.ML_CONFIG
        models: dict[str, object] = {}

        if self.model_choice in ("ensemble", "lgbm"):
            models["lgbm"] = lgb.LGBMRegressor(
                learning_rate=ml.lgbm.learning_rate,
                n_estimators=ml.lgbm.n_estimators,
                max_depth=ml.lgbm.max_depth,
                num_leaves=ml.lgbm.num_leaves,
                random_state=_SEED,
                n_jobs=1,      # OMP pinned to 1 thread; keep the tree lib serial
                verbose=-1,
            )
        if self.model_choice in ("ensemble", "xgb"):
            models["xgb"] = xgb.XGBRegressor(
                learning_rate=ml.xgb.learning_rate,
                n_estimators=ml.xgb.n_estimators,
                max_depth=ml.xgb.max_depth,
                random_state=_SEED,
                n_jobs=1,      # OMP pinned to 1 thread; keep the tree lib serial
                verbosity=0,
            )
        if self.model_choice in ("ensemble", "rf"):
            models["rf"] = RandomForestRegressor(
                n_estimators=ml.rf.n_estimators,
                max_depth=ml.rf.max_depth,
                min_samples_split=ml.rf.min_samples_split,
                random_state=_SEED,
                n_jobs=ml.rf.n_jobs,   # capped to bound peak RAM; result-invariant
            )
        return models

    # ---------------------------------------------------------------------- #
    # Per-fold fit + predict
    # ---------------------------------------------------------------------- #
    def _fit_predict_block(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """
        Fit every active model on the training block and return the ensemble
        Alpha Score plus each learner's raw predictions for the OOS test block.

        Returns ``(ensemble, per_model)`` where ``ensemble`` is a Series indexed
        by the test block's ('date', 'ticker') rows (the driving Alpha Score) and
        ``per_model`` is a DataFrame of the same index with one column per active
        learner plus an ``ensemble`` column — consumed only by the read-only IC
        diagnostic. Rows whose chosen target is NaN (trailing / cross-session
        labels kept by the feature factory) are dropped from training only —
        every test row is still scored, since prediction needs features, not a label.
        """
        train_ok: pd.Series = train_df[self.target_col].notna()
        x_train: pd.DataFrame = train_df.loc[train_ok, self.feature_cols]
        y_train: pd.Series = train_df.loc[train_ok, self.target_col]
        x_test: pd.DataFrame = test_df[self.feature_cols]

        models = self._build_models()
        model_names: list[str] = list(models.keys())
        preds: list[np.ndarray] = []
        for model in models.values():
            model.fit(x_train.values, y_train.values)
            preds.append(model.predict(x_test.values))

        # Multi-model voting: average the raw return predictions per asset.
        pred_stack: np.ndarray = np.column_stack(preds)
        ensemble: np.ndarray = pred_stack.mean(axis=1)

        ensemble_series: pd.Series = pd.Series(
            ensemble, index=test_df.index, name="alpha_score"
        )
        per_model: pd.DataFrame = pd.DataFrame(
            pred_stack, index=test_df.index, columns=model_names
        )
        per_model["ensemble"] = ensemble
        return ensemble_series, per_model

    # ---------------------------------------------------------------------- #
    # Walk-forward orchestration
    # ---------------------------------------------------------------------- #
    def run_walk_forward(self, features_df: pd.DataFrame) -> WalkForwardResult:
        """
        Slide the train/predict window across the timeline and stitch together
        the full out-of-sample Alpha Score, then bucket into decile masks.

        Args:
            features_df: MultiIndex ('date', 'ticker') frame carrying at least
                ``FEATURE_COLUMNS`` and the chosen ``target_col``.

        Returns:
            WalkForwardResult with wide (date x ticker) alpha_scores plus the
            long and short decile masks.
        """
        self._validate_input(features_df)
        df: pd.DataFrame = features_df.sort_index()
        self.feature_cols = present_feature_columns(df)

        # Walk-forward windows are in TRADING DAYS (D2). The 'date' level may hold
        # intraday bar timestamps, so collapse each row to its calendar day and
        # slide over unique days. Because df is sorted by (date, ticker), the
        # per-row calendar day array is non-decreasing and folds are contiguous
        # slices found by searchsorted — no per-day boolean scan.
        row_days: np.ndarray = (
            pd.DatetimeIndex(df.index.get_level_values(DATE_LEVEL))
            .normalize()
            .to_numpy()
        )
        unique_days: np.ndarray = np.unique(row_days)
        n_dates: int = len(unique_days)
        min_required: int = self.train_window + 1
        if n_dates < min_required:
            raise ValueError(
                f"Need >= {min_required} trading days for a single walk-forward "
                f"fold (train_window={self.train_window} + 1 test day); "
                f"got {n_dates}."
            )

        def _slice(day_lo, day_hi_inclusive) -> pd.DataFrame:
            """Contiguous row slice for the calendar-day span [lo, hi] inclusive."""
            lo = int(np.searchsorted(row_days, day_lo, side="left"))
            hi = int(np.searchsorted(row_days, day_hi_inclusive, side="right"))
            return df.iloc[lo:hi]

        oos_scores: list[pd.Series] = []
        oos_per_model: list[pd.DataFrame] = []   # for the read-only IC diagnostic
        oos_realized: list[pd.Series] = []       # realized labels aligned to preds
        n_folds: int = 0

        # Start scoring the first day for which a full training window exists.
        start: int = self.train_window
        while start < n_dates:
            train_lo: int = start - self.train_window
            train_block: pd.DataFrame = _slice(
                unique_days[train_lo], unique_days[start - 1]
            )
            test_hi: int = min(start + self.predict_window, n_dates) - 1
            test_block: pd.DataFrame = _slice(unique_days[start], unique_days[test_hi])

            has_train_labels: bool = bool(train_block[self.target_col].notna().any())
            if not test_block.empty and has_train_labels:
                ensemble, per_model = self._fit_predict_block(train_block, test_block)
                oos_scores.append(ensemble)
                oos_per_model.append(per_model)
                oos_realized.append(test_block[self.target_col])
                n_folds += 1

            # Release this fold's fitted forests (largest transient) before the
            # next fold allocates — bounds peak RAM on the intraday cells.
            gc.collect()
            start += self.predict_window

        if not oos_scores:
            raise RuntimeError("Walk-forward produced no out-of-sample folds.")

        alpha_long: pd.Series = pd.concat(oos_scores).sort_index()
        alpha_scores: pd.DataFrame = alpha_long.unstack(level=TICKER_LEVEL)

        long_mask, short_mask = self._allocate_deciles(alpha_scores)

        # Read-only fit diagnostic — does NOT touch the alpha/decile path above.
        pred_panel: pd.DataFrame = pd.concat(oos_per_model).sort_index()
        realized: pd.Series = pd.concat(oos_realized).sort_index()
        ic_diagnostics: pd.DataFrame = _compute_ic_diagnostics(pred_panel, realized)

        print(
            f"[tier1_trees] Walk-forward done | model={self.model_choice} | "
            f"target={self.target_col} | folds={n_folds} | "
            f"OOS days={alpha_scores.shape[0]} | universe={alpha_scores.shape[1]}"
        )
        _print_ic_panel(ic_diagnostics)
        return WalkForwardResult(
            alpha_scores=alpha_scores,
            long_mask=long_mask,
            short_mask=short_mask,
            ic_diagnostics=ic_diagnostics,
        )

    # ---------------------------------------------------------------------- #
    # Cross-sectional decile bucketing
    # ---------------------------------------------------------------------- #
    def _allocate_deciles(
        self,
        alpha_scores: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        For each row (trading day), select the top ``decile_pct`` of assets by
        Alpha Score into the long bucket and the bottom ``decile_pct`` into the
        short bucket.

        Bucket width per day is ``k = max(1, floor(n_valid * decile_pct))`` where
        ``n_valid`` is the count of assets with a score that day, guaranteeing a
        non-empty book even on a thin cross-section.

        Returns:
            (long_mask, short_mask) — same shape/index/columns as alpha_scores,
            with values in {0, 1} and {0, -1} respectively.
        """
        # Per-day valid asset count and bucket width.
        valid_counts: pd.Series = alpha_scores.notna().sum(axis=1)
        k_per_day: pd.Series = np.floor(valid_counts * self.decile_pct).astype(int)
        k_per_day = k_per_day.clip(lower=1)

        # Descending rank -> 1 is the highest score; ascending rank -> 1 lowest.
        rank_desc: pd.DataFrame = alpha_scores.rank(
            axis=1, ascending=False, method="first"
        )
        rank_asc: pd.DataFrame = alpha_scores.rank(
            axis=1, ascending=True, method="first"
        )

        k_col = k_per_day.values[:, None]  # broadcast per-row threshold across cols
        long_mask: pd.DataFrame = (rank_desc.le(k_col)).astype(int)
        short_mask: pd.DataFrame = (rank_asc.le(k_col)).astype(int) * -1

        # A NaN score can never be selected on either side.
        valid: pd.DataFrame = alpha_scores.notna()
        long_mask = long_mask.where(valid, 0)
        short_mask = short_mask.where(valid, 0)

        return long_mask, short_mask

    # ---------------------------------------------------------------------- #
    # Input validation
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _validate_input(features_df: pd.DataFrame) -> None:
        if not isinstance(features_df.index, pd.MultiIndex):
            raise TypeError("features_df must have a ('date', 'ticker') MultiIndex.")
        if list(features_df.index.names) != [DATE_LEVEL, TICKER_LEVEL]:
            raise ValueError(
                f"Index names must be ['{DATE_LEVEL}', '{TICKER_LEVEL}'], "
                f"got {list(features_df.index.names)}."
            )
        # The price family is always required; the institutional pair is optional
        # (absent on price-only intraday panels), so validate the core only.
        missing_feats = set(PRICE_FEATURE_COLUMNS) - set(features_df.columns)
        if missing_feats:
            raise ValueError(f"features_df missing feature columns: {sorted(missing_feats)}")


# --------------------------------------------------------------------------- #
# Dry-run verification harness
# --------------------------------------------------------------------------- #
def _make_dummy_features(
    n_days: int = 800,
    n_tickers: int = 10,
    seed: int = _SEED,
) -> pd.DataFrame:
    """
    Build a synthetic feature matrix matching the FeatureFactory output schema:
    a ('date', 'ticker') MultiIndex carrying all FEATURE_COLUMNS (already in the
    [-1, +1] normalized band) plus the forward-return TARGET_COLUMNS.

    A faint linear signal is injected from the features into the targets so the
    trees have something learnable — the harness is checking plumbing and the
    decile allocation math, not predictive skill.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days, name=DATE_LEVEL)
    tickers = [f"STOCK_{i:02d}" for i in range(1, n_tickers + 1)]
    index = pd.MultiIndex.from_product([dates, tickers], names=[DATE_LEVEL, TICKER_LEVEL])

    n_rows = len(index)
    n_feats = len(FEATURE_COLUMNS)

    # Normalized features live in [-1, +1]; emulate that with a uniform draw.
    feats = rng.uniform(-1.0, 1.0, size=(n_rows, n_feats))
    feat_df = pd.DataFrame(feats, index=index, columns=FEATURE_COLUMNS)

    # Faint learnable signal + noise -> forward return targets. Each horizon scales
    # the signal by its own bar count and the noise by ~sqrt of it, so every entry in
    # TARGET_COLUMNS is populated whatever horizons the factory is configured with.
    weights = rng.normal(0.0, 0.01, size=n_feats)
    signal = feats @ weights
    for col, h in zip(TARGET_COLUMNS, TARGET_HORIZONS):
        feat_df[col] = h * signal + rng.normal(0.0, 0.02 * np.sqrt(h), size=n_rows)

    return feat_df.sort_index()


if __name__ == "__main__":
    print("=" * 70)
    print("tier1_trees dry-run — walk-forward tree ensemble + decile allocation")
    print("=" * 70)

    N_TICKERS: int = 10
    dummy: pd.DataFrame = _make_dummy_features(n_days=800, n_tickers=N_TICKERS)
    print(f"  input shape   : {dummy.shape}")
    print(f"  input columns : {dummy.columns.tolist()}")
    print(
        f"  dates={dummy.index.get_level_values(DATE_LEVEL).nunique()} "
        f"tickers={dummy.index.get_level_values(TICKER_LEVEL).nunique()}"
    )
    print()

    engine = TreeAlphaEngine(
        target_col=DEFAULT_TARGET,
        model_choice="ensemble",
        train_window=504,
        predict_window=63,
    )
    result: WalkForwardResult = engine.run_walk_forward(dummy)

    print()
    print("=" * 70)
    print("Output shapes")
    print("=" * 70)
    print(f"  alpha_scores : {result.alpha_scores.shape}")
    print(f"  long_mask    : {result.long_mask.shape}")
    print(f"  short_mask   : {result.short_mask.shape}")

    # --- Decile allocation proof ------------------------------------------- #
    expected_k: int = max(1, int(np.floor(N_TICKERS * DECILE_PCT)))
    long_per_day: pd.Series = (result.long_mask == 1).sum(axis=1)
    short_per_day: pd.Series = (result.short_mask == -1).sum(axis=1)

    print()
    print("=" * 70)
    print("Decile allocation check")
    print("=" * 70)
    print(f"  universe size           : {N_TICKERS}")
    print(f"  expected per-side count : {expected_k}  ({DECILE_PCT:.0%} of universe)")
    print(f"  long  per-day  min/max  : {long_per_day.min()} / {long_per_day.max()}")
    print(f"  short per-day  min/max  : {short_per_day.min()} / {short_per_day.max()}")
    print(f"  total long  positions   : {int(long_per_day.sum())}")
    print(f"  total short positions   : {int(short_per_day.sum())}")

    # No asset can be both long and short on the same day.
    overlap: int = int(((result.long_mask == 1) & (result.short_mask == -1)).sum().sum())

    long_ok: bool = bool((long_per_day == expected_k).all())
    short_ok: bool = bool((short_per_day == expected_k).all())
    no_overlap: bool = overlap == 0

    print()
    print(f"  long  side == {expected_k} every day : {'PASS' if long_ok else 'FAIL'}")
    print(f"  short side == {expected_k} every day : {'PASS' if short_ok else 'FAIL'}")
    print(f"  no long/short overlap        : {'PASS' if no_overlap else f'FAIL ({overlap})'}")
    print("=" * 70)

    # --- Tree-fit IC diagnostic -------------------------------------------- #
    # _make_dummy_features injects a faint feature->target signal, so a correctly
    # wired IC must be positive (the trees recover it).
    print()
    print("=" * 70)
    print("Tree-fit IC diagnostic")
    print("=" * 70)
    ic = result.ic_diagnostics
    _print_ic_panel(ic)

    ic_checks: dict[str, bool] = {
        "IC panel has all learners + ensemble":
            set(ic.index) == {"lgbm", "xgb", "rf", "ensemble"},
        "ensemble mean IC > 0 on synthetic signal":
            float(ic.loc["ensemble", "mean_IC"]) > 0,
        "ensemble IC_t is finite":
            bool(np.isfinite(ic.loc["ensemble", "IC_t"])),
    }
    print()
    for name, ok in ic_checks.items():
        print(f"  {name:<44}: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    assert all(ic_checks.values()), "tier1_trees IC diagnostic FAILED"
