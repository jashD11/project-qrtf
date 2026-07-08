"""
Phase 2 feature factory.

Consumes the MultiIndexed ('date', 'ticker') augmented daily DataFrame produced
by ``src/data_scraping.py`` (OHLCV + DeliveryQty/DeliveryPct) and emits a
fully cross-sectionally normalized feature matrix plus forward-return targets.

Design contract
    - Completely vectorized: rolling windows are computed with grouped Pandas /
      NumPy operations. There is no Python loop over the time dimension.
    - Per-ticker time-series stats use ``groupby(level='ticker')`` so windows
      never bleed across symbols.
    - Every feature is cross-sectionally rank-normalized per date into a strict
      [-1, +1] band via ``groupby(level='date')``.
    - Zero look-ahead bias: all rolling/shift features use current-or-past bars
      only. Only the *target* columns look forward (``shift(-1)`` / ``shift(-5)``).

Feature families (19 features total)
    Momentum & Reversal (10)
        mom_logret_{1,5,20,60,120,252}d, mom_close_sma{20,50,200},
        mom_max_dret_20d
    Volatility & Risk (4)
        vol_realized_{20,60}d, vol_parkinson_20d, vol_drawdown_252d
    Liquidity, Volume & Institutional (5)
        liq_amihud, liq_vol_var_20d, liq_turnover_20d,
        inst_delivery_qty, inst_delivery_pct

Targets (2)
    tgt_fwd_logret_1d, tgt_fwd_logret_5d
"""

from typing import Final

import numpy as np
import pandas as pd

# --- Index level names (must match src/data_scraping.py) ------------------- #
DATE_LEVEL: Final[str] = "date"
TICKER_LEVEL: Final[str] = "ticker"

# --- OHLCV column names on the augmented daily frame ----------------------- #
OPEN_COL: Final[str] = "open"
HIGH_COL: Final[str] = "high"
LOW_COL: Final[str] = "low"
CLOSE_COL: Final[str] = "close"
VOLUME_COL: Final[str] = "volume"

# --- Institutional columns (match src/data_scraping.py output) ------------- #
DELIVERY_QTY_COL: Final[str] = "DeliveryQty"
DELIVERY_PCT_COL: Final[str] = "DeliveryPct"

# --- Output schema --------------------------------------------------------- #
FEATURE_COLUMNS: Final[list[str]] = [
    # Momentum & Reversal
    "mom_logret_1d",
    "mom_logret_5d",
    "mom_logret_20d",
    "mom_logret_60d",
    "mom_logret_120d",
    "mom_logret_252d",
    "mom_close_sma20",
    "mom_close_sma50",
    "mom_close_sma200",
    "mom_max_dret_20d",
    # Volatility & Risk
    "vol_realized_20d",
    "vol_realized_60d",
    "vol_parkinson_20d",
    "vol_drawdown_252d",
    # Liquidity, Volume & Institutional
    "liq_amihud",
    "liq_vol_var_20d",
    "liq_turnover_20d",
    "inst_delivery_qty",
    "inst_delivery_pct",
]
TARGET_COLUMNS: Final[list[str]] = ["tgt_fwd_logret_1d", "tgt_fwd_logret_5d"]

_PARKINSON_K: Final[float] = 1.0 / (4.0 * np.log(2.0))  # Parkinson variance scalar


# --------------------------------------------------------------------------- #
# Vectorized per-ticker rolling helpers
# --------------------------------------------------------------------------- #
def _by_ticker(s: pd.Series):
    """Group a series by the ticker index level for per-symbol time series ops."""
    return s.groupby(level=TICKER_LEVEL)


def _shift(s: pd.Series, periods: int) -> pd.Series:
    """Per-ticker shift. Positive = look back, negative = look forward (targets)."""
    return _by_ticker(s).shift(periods)


def _roll(s: pd.Series, window: int, op: str) -> pd.Series:
    """
    Per-ticker trailing rolling reduction (mean/std/max/var).

    ``min_periods == window`` enforces a complete window — partial-history rows
    become NaN and are dropped downstream, so no reduced-support values leak in.
    """
    return _by_ticker(s).transform(
        lambda x: getattr(x.rolling(window, min_periods=window), op)()
    )


# --------------------------------------------------------------------------- #
# Feature computation
# --------------------------------------------------------------------------- #
def _compute_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the raw (un-normalized) feature panel from the augmented frame."""
    close = df[CLOSE_COL]
    high = df[HIGH_COL]
    low = df[LOW_COL]
    volume = df[VOLUME_COL]

    # Daily log return is the atomic building block for several families.
    daily_logret = np.log(close / _shift(close, 1))

    feats = pd.DataFrame(index=df.index)

    # ---- Momentum & Reversal ---------------------------------------------- #
    # Multi-horizon cumulative log returns: log(C_t / C_{t-n}).
    for horizon in (1, 5, 20, 60, 120, 252):
        feats[f"mom_logret_{horizon}d"] = np.log(close / _shift(close, horizon))

    # Distance of price from its own trailing simple moving averages.
    feats["mom_close_sma20"] = close / _roll(close, 20, "mean")
    feats["mom_close_sma50"] = close / _roll(close, 50, "mean")
    feats["mom_close_sma200"] = close / _roll(close, 200, "mean")

    # MAX proxy: largest single-day return over the trailing month (lottery/reversal).
    feats["mom_max_dret_20d"] = _roll(daily_logret, 20, "max")

    # ---- Volatility & Risk ------------------------------------------------ #
    # Realized volatility = rolling std of daily log returns.
    feats["vol_realized_20d"] = _roll(daily_logret, 20, "std")
    feats["vol_realized_60d"] = _roll(daily_logret, 60, "std")

    # Parkinson high-low volatility: sqrt( k * mean( ln(H/L)^2 ) ), k = 1/(4 ln2).
    hl_sq = np.square(np.log(high / low))
    feats["vol_parkinson_20d"] = np.sqrt(_PARKINSON_K * _roll(hl_sq, 20, "mean"))

    # Trailing 252-day peak drawdown: current close vs. rolling one-year high.
    peak_252 = _roll(close, 252, "max")
    feats["vol_drawdown_252d"] = (close / peak_252) - 1.0

    # ---- Liquidity, Volume & Institutional -------------------------------- #
    # Amihud illiquidity: |return| / (Close * Volume).
    feats["liq_amihud"] = daily_logret.abs() / (close * volume)

    # 20-day volume variance and turnover vs. its own trailing average.
    feats["liq_vol_var_20d"] = _roll(volume, 20, "var")
    feats["liq_turnover_20d"] = volume / _roll(volume, 20, "mean")

    # Institutional delivery metrics carried straight from the augmented frame.
    feats["inst_delivery_qty"] = df[DELIVERY_QTY_COL]
    feats["inst_delivery_pct"] = df[DELIVERY_PCT_COL]

    # Guard against divide-by-zero / log blowups before normalization so that
    # infinities cannot dominate the cross-sectional ranking.
    feats = feats.replace([np.inf, -np.inf], np.nan)
    return feats[FEATURE_COLUMNS]


def _rank_normalize(feats: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectionally rank-normalize every feature per date into strict [-1, +1].

    For each date, a feature's ranks r in [1, n] map linearly:
        norm = 2 * (r - 1) / (n - 1) - 1
    so the daily minimum -> -1 and the daily maximum -> +1. Ties take the average
    rank (never breaching the band). A date with exactly one valid observation
    maps that observation to 0.0 (no cross-section to rank against). Undefined
    values (warm-up NaNs, empty cross-sections) stay NaN and are dropped later.
    """
    by_date = feats.groupby(level=DATE_LEVEL)
    ranks = by_date.rank(method="average")
    counts = by_date.transform("count")

    denom = (counts - 1).where(counts > 1)  # NaN when counts <= 1 -> normed NaN
    normed = 2.0 * (ranks - 1.0) / denom - 1.0

    # Genuine single-observation cross-section: neutral 0.0, but only for the
    # present cell — absent tickers on that date keep NaN so their rows drop.
    single = (counts == 1) & feats.notna()
    return normed.mask(single, 0.0)


def _compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Forward 1-day and 5-day log returns (labels only — never fed as features)."""
    close = df[CLOSE_COL]
    targets = pd.DataFrame(index=df.index)
    targets["tgt_fwd_logret_1d"] = np.log(_shift(close, -1) / close)
    targets["tgt_fwd_logret_5d"] = np.log(_shift(close, -5) / close)
    return targets.replace([np.inf, -np.inf], np.nan)


def create_features(augmented_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the normalized feature matrix + forward-return targets.

    Args:
        augmented_df: MultiIndex ('date', 'ticker') daily frame with OHLCV plus
            DeliveryQty / DeliveryPct, as returned by src/data_scraping.py.

    Returns:
        MultiIndex ('date', 'ticker') DataFrame with the 19 [-1, +1] feature
        columns followed by the 2 target columns. Rows with NaN/inf features or
        NaN forward labels (early-history warm-up and the trailing target window)
        are dropped, so every returned row is fully populated and leakage-free.
    """
    if not isinstance(augmented_df.index, pd.MultiIndex):
        raise ValueError("augmented_df must have a MultiIndex ('date', 'ticker')")

    # Sort so per-ticker rolling windows are chronologically ordered.
    df = augmented_df.sort_index()

    raw = _compute_raw_features(df)
    normed = _rank_normalize(raw)
    targets = _compute_targets(df)

    out = pd.concat([normed, targets], axis=1)

    before = len(out)
    # Single dropna removes warm-up NaNs (features), inf-derived NaNs, and the
    # trailing rows whose forward labels are undefined.
    out = out.dropna(how="any")
    after = len(out)

    print(
        f"[feature_creator] Features: {len(FEATURE_COLUMNS)} | "
        f"targets: {len(TARGET_COLUMNS)} | "
        f"rows {before} -> {after} after dropna "
        f"({before - after} warm-up/label rows removed)"
    )
    return out


# --------------------------------------------------------------------------- #
# Dry-run: synthesize a scraper-schema frame and verify the [-1, +1] contract
# --------------------------------------------------------------------------- #
def _make_dummy_augmented(
    n_days: int = 400, n_tickers: int = 10, seed: int = 42
) -> pd.DataFrame:
    """Fabricate a MultiIndex ('date','ticker') OHLCV+delivery frame for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    tickers = [f"TICK_{i:02d}" for i in range(1, n_tickers + 1)]

    frames: list[pd.DataFrame] = []
    for tk in tickers:
        rets = rng.normal(0.0004, 0.02, n_days)
        close = 100.0 * np.exp(np.cumsum(rets))
        open_ = close * (1.0 + rng.normal(0.0, 0.004, n_days))
        span = np.abs(rng.normal(0.0, 0.01, n_days))
        high = np.maximum.reduce([close * (1.0 + span), open_, close])
        low = np.minimum.reduce([close * (1.0 - span), open_, close])
        volume = rng.integers(100_000, 5_000_000, n_days).astype(float)
        deliv_pct = rng.uniform(20.0, 80.0, n_days)
        deliv_qty = volume * deliv_pct / 100.0

        frames.append(
            pd.DataFrame(
                {
                    DATE_LEVEL: dates,
                    TICKER_LEVEL: tk,
                    OPEN_COL: open_,
                    HIGH_COL: high,
                    LOW_COL: low,
                    CLOSE_COL: close,
                    VOLUME_COL: volume,
                    DELIVERY_QTY_COL: deliv_qty,
                    DELIVERY_PCT_COL: deliv_pct,
                }
            )
        )

    return (
        pd.concat(frames, ignore_index=True)
        .set_index([DATE_LEVEL, TICKER_LEVEL])
        .sort_index()
    )


if __name__ == "__main__":
    print("=" * 70)
    print("feature_creator dry-run — synthetic scraper-schema panel")
    print("=" * 70)

    dummy = _make_dummy_augmented()
    print(f"  input shape   : {dummy.shape}")
    print(f"  input columns : {dummy.columns.tolist()}")
    print(
        f"  dates={dummy.index.get_level_values(DATE_LEVEL).nunique()} "
        f"tickers={dummy.index.get_level_values(TICKER_LEVEL).nunique()}"
    )
    print()

    matrix = create_features(dummy)

    print()
    print("=" * 70)
    print("Feature matrix .describe()")
    print("=" * 70)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(matrix[FEATURE_COLUMNS].describe().to_string())

    print()
    print("=" * 70)
    print("Target .describe()")
    print("=" * 70)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(matrix[TARGET_COLUMNS].describe().to_string())

    # --- Strict boundary verification -------------------------------------- #
    feat_min = matrix[FEATURE_COLUMNS].min().min()
    feat_max = matrix[FEATURE_COLUMNS].max().max()
    tol = 1e-9
    in_bounds = (feat_min >= -1.0 - tol) and (feat_max <= 1.0 + tol)
    no_nan = not matrix.isna().any().any()

    print()
    print("=" * 70)
    print(f"  feature global min : {feat_min:+.6f}")
    print(f"  feature global max : {feat_max:+.6f}")
    print(f"  [-1, +1] bounds    : {'PASS' if in_bounds else 'FAIL'}")
    print(f"  no NaN/leak rows   : {'PASS' if no_nan else 'FAIL'}")
    print("=" * 70)
