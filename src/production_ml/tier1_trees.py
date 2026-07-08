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
    TARGET_COLUMNS,
)

# --- Walk-forward window sizing (in unique trading days) ------------------- #
TRAIN_WINDOW: Final[int] = 504   # ~2 trading years of history to fit on
PREDICT_WINDOW: Final[int] = 63  # ~1 trading quarter scored out-of-sample

# --- Default learning target and decile width ------------------------------ #
DEFAULT_TARGET: Final[str] = "tgt_fwd_logret_1d"
DECILE_PCT: Final[float] = 0.10  # top / bottom 10% of the universe

# --- Shared RNG seed so every run is reproducible -------------------------- #
_SEED: Final[int] = 42


@dataclass
class WalkForwardResult:
    """Container for the full out-of-sample output of a walk-forward run."""
    alpha_scores: pd.DataFrame  # (date x ticker) ensemble Alpha Score
    long_mask: pd.DataFrame     # (date x ticker) 1 = long, 0 = flat
    short_mask: pd.DataFrame    # (date x ticker) -1 = short, 0 = flat


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
                n_jobs=-1,
            )
        return models

    # ---------------------------------------------------------------------- #
    # Per-fold fit + predict
    # ---------------------------------------------------------------------- #
    def _fit_predict_block(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Fit every active model on the training block and return the ensemble
        Alpha Score for the out-of-sample test block.

        Returns a Series indexed by the test block's ('date', 'ticker') rows.
        """
        x_train: pd.DataFrame = train_df[FEATURE_COLUMNS]
        y_train: pd.Series = train_df[self.target_col]
        x_test: pd.DataFrame = test_df[FEATURE_COLUMNS]

        models = self._build_models()
        preds: list[np.ndarray] = []
        for model in models.values():
            model.fit(x_train.values, y_train.values)
            preds.append(model.predict(x_test.values))

        # Multi-model voting: average the raw return predictions per asset.
        ensemble: np.ndarray = np.mean(np.column_stack(preds), axis=1)
        return pd.Series(ensemble, index=test_df.index, name="alpha_score")

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

        unique_dates: np.ndarray = (
            df.index.get_level_values(DATE_LEVEL).unique().sort_values().to_numpy()
        )
        n_dates: int = len(unique_dates)
        min_required: int = self.train_window + 1
        if n_dates < min_required:
            raise ValueError(
                f"Need >= {min_required} trading days for a single walk-forward "
                f"fold (train_window={self.train_window} + 1 test day); "
                f"got {n_dates}."
            )

        oos_scores: list[pd.Series] = []
        n_folds: int = 0

        # Start scoring the first day for which a full training window exists.
        start: int = self.train_window
        while start < n_dates:
            train_lo: int = start - self.train_window
            train_dates = unique_dates[train_lo:start]
            test_dates = unique_dates[start:start + self.predict_window]

            train_block: pd.DataFrame = df[
                df.index.get_level_values(DATE_LEVEL).isin(train_dates)
            ]
            test_block: pd.DataFrame = df[
                df.index.get_level_values(DATE_LEVEL).isin(test_dates)
            ]

            if not test_block.empty and not train_block.empty:
                oos_scores.append(self._fit_predict_block(train_block, test_block))
                n_folds += 1

            start += self.predict_window

        if not oos_scores:
            raise RuntimeError("Walk-forward produced no out-of-sample folds.")

        alpha_long: pd.Series = pd.concat(oos_scores).sort_index()
        alpha_scores: pd.DataFrame = alpha_long.unstack(level=TICKER_LEVEL)

        long_mask, short_mask = self._allocate_deciles(alpha_scores)

        print(
            f"[tier1_trees] Walk-forward done | model={self.model_choice} | "
            f"target={self.target_col} | folds={n_folds} | "
            f"OOS days={alpha_scores.shape[0]} | universe={alpha_scores.shape[1]}"
        )
        return WalkForwardResult(
            alpha_scores=alpha_scores,
            long_mask=long_mask,
            short_mask=short_mask,
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
        missing_feats = set(FEATURE_COLUMNS) - set(features_df.columns)
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

    # Faint learnable signal + noise -> forward return targets.
    weights = rng.normal(0.0, 0.01, size=n_feats)
    signal = feats @ weights
    noise = rng.normal(0.0, 0.02, size=n_rows)
    feat_df[TARGET_COLUMNS[0]] = signal + noise
    feat_df[TARGET_COLUMNS[1]] = 5.0 * signal + rng.normal(0.0, 0.045, size=n_rows)

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
        target_col="tgt_fwd_logret_1d",
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
