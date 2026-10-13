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

Frequency awareness (design decisions D1/D2)
    The same factory runs on any bar frequency (15min/30min/60min/daily). All
    lookback horizons are counted in **bars**, so feature names carry a
    bar-neutral ``_Nb`` suffix (``mom_logret_20b`` = 20 bars back, whatever the
    frequency). The ``date`` index level holds each bar's timestamp (a calendar
    date on daily data, an intraday timestamp otherwise); cross-sectional
    ranking per that level therefore compares stocks at the same bar.

    Two things flex by frequency:
      - Institutional family (delivery) is end-of-day only. On daily bars it
        joins directly; intraday it is a t-1 lagged broadcast (done upstream in
        augment_dataset). If the columns are absent, the family is dropped and
        the factory emits 17 features instead of 19.
      - Forward targets that would span the overnight gap are nulled on intraday
        frequencies (a target must resolve inside the same session), so the last
        few bars of each day carry no label. Daily bars never null (crossing to
        the next day *is* the target).

Feature families (19 features total)
    Momentum & Reversal (10)
        mom_logret_{1,5,20,60,120,252}b, mom_close_sma{20,50,200},
        mom_max_dret_20b
    Volatility & Risk (4)
        vol_realized_{20,60}b, vol_parkinson_20b, vol_drawdown_252b
    Liquidity, Volume & Institutional (5)
        liq_amihud, liq_vol_var_20b, liq_turnover_20b,
        inst_delivery_qty, inst_delivery_pct

Targets (3)
    tgt_fwd_logret_1b, tgt_fwd_logret_5b, tgt_fwd_logret_21b
"""

import os
import sys
from typing import Final

import numpy as np
import pandas as pd

# Make the repo root importable so ``import config`` works whether this file is
# run directly or imported by an orchestrator (mirrors tier1_trees).
_ROOT: str = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config

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

# --- Lookback horizons (counted in BARS — see D2) -------------------------- #
LOGRET_HORIZONS: Final[tuple[int, ...]] = (1, 5, 20, 60, 120, 252)
SMA_WINDOWS: Final[tuple[int, ...]] = (20, 50, 200)
MAXDRET_WINDOW: Final[int] = 20
VOL_WINDOWS: Final[tuple[int, ...]] = (20, 60)
PARKINSON_WINDOW: Final[int] = 20
DRAWDOWN_WINDOW: Final[int] = 252
VOLVAR_WINDOW: Final[int] = 20
TURNOVER_WINDOW: Final[int] = 20
# Phase 4c C1 adds the 21-bar label. Turnover is the one axis with headroom left —
# 1b -> 5b roughly halved it and was the single biggest net lift in Phase 3 — and a
# monthly horizon is the untried step. Adding it does **not** change the emitted row
# set: the keep rule is "at least one valid target", and a defined 21-bar label
# implies a defined 5- and 1-bar one, so the OR is unchanged and Phase 2/3/4b panels
# are byte-identical. It is committed as a frozen 2-point fork and paid for in N,
# not scanned (docs/phase4c_plan.md §3, §4.1).
TARGET_HORIZONS: Final[tuple[int, ...]] = (1, 5, 21)

# --- Output schema (names derived from the horizons above) ----------------- #
# Price-only families (always available, any frequency).
PRICE_FEATURE_COLUMNS: Final[list[str]] = [
    # Momentum & Reversal
    *[f"mom_logret_{h}b" for h in LOGRET_HORIZONS],
    *[f"mom_close_sma{w}" for w in SMA_WINDOWS],
    f"mom_max_dret_{MAXDRET_WINDOW}b",
    # Volatility & Risk
    *[f"vol_realized_{w}b" for w in VOL_WINDOWS],
    f"vol_parkinson_{PARKINSON_WINDOW}b",
    f"vol_drawdown_{DRAWDOWN_WINDOW}b",
    # Liquidity & Volume
    "liq_amihud",
    f"liq_vol_var_{VOLVAR_WINDOW}b",
    f"liq_turnover_{TURNOVER_WINDOW}b",
]
# End-of-day institutional family (daily direct / intraday t-1 lagged broadcast).
INSTITUTIONAL_FEATURE_COLUMNS: Final[list[str]] = [
    "inst_delivery_qty",
    "inst_delivery_pct",
]
# Full 19-feature schema. ``feature_columns(with_institutional=False)`` drops the
# trailing two when delivery data is unavailable for the chosen frequency.
FEATURE_COLUMNS: Final[list[str]] = PRICE_FEATURE_COLUMNS + INSTITUTIONAL_FEATURE_COLUMNS
TARGET_COLUMNS: Final[list[str]] = [f"tgt_fwd_logret_{h}b" for h in TARGET_HORIZONS]
# The date each label is realised: the name's own bar h steps ahead. Emitted only on
# request (``label_end_dates``), for the Tier 1 label-overlap purge. A label counts
# OWN bars, so across a halt it ends later than h calendar bars on; a purge that
# counted calendar bars let 0.02% of training labels reach the scored block
# (docs/phase6_plan.md §B.8, Stage 6).
TARGET_END_COLUMNS: Final[list[str]] = [f"tgt_end_{h}b" for h in TARGET_HORIZONS]


def feature_columns(with_institutional: bool = True) -> list[str]:
    """The feature schema for a run: all 19, or 17 when delivery is unavailable."""
    return (
        PRICE_FEATURE_COLUMNS + INSTITUTIONAL_FEATURE_COLUMNS
        if with_institutional
        else list(PRICE_FEATURE_COLUMNS)
    )


# --- Phase 6 Part B feature sets (docs/phase6_plan.md §B.2) ---------------- #
# Built by src/phase6_features/ as raw wide (date × ticker) Parquets and read here as
# plain files, so this package does not import that one. Three treatments:
#   ranked    per-stock, rank-normalized per date among in-universe names, like the 17;
#   market    one value per date for every stock, so entered raw (ranking would make
#             it a constant);
#   optional  options / futures (F2): ranked among the names that have them, 0 (the
#             band's midpoint) for the rest, with ``opt_avail`` (raw 0/1) telling the
#             trees which is which. Optional features never drop a row.
F1_RANKED: Final[list[str]] = [
    "beta_252b", "ivol_63b", "ret_overnight_21b", "ret_intraday_21b", "trade_size_21b",
    "spread_cs_63b", "sector_mom_63b", "sector_rel_21b", "sector_pe", "sector_pb",
    "sector_dy",
]
F1_MARKET: Final[list[str]] = ["mkt_vix_z", "mkt_pe_z", "mkt_ret_21b"]
F2_OPTIONAL: Final[list[str]] = [
    "opt_ivspread", "opt_skew", "opt_atm_iv", "opt_iv_chg_5b", "opt_iv_rv",
    "opt_volratio_5b", "fut_basis", "fut_oi_chg_5b",
]
F2_FLAG: Final[list[str]] = ["opt_avail"]
FEATURE_SETS: Final[dict[str, list[str]]] = {
    "p17": [],
    "p17x": F1_RANKED + F1_MARKET,
    "p17xo": F1_RANKED + F1_MARKET + F2_OPTIONAL + F2_FLAG,
}
EXTRA_FEATURE_COLUMNS: Final[list[str]] = FEATURE_SETS["p17xo"]
PHASE6_PANEL_DIRS: Final[dict[str, str]] = {"f1": "data/phase6/f1", "f2": "data/phase6/f2"}
PHASE6_MARKET_PARQUET: Final[str] = "data/phase6/f1/market.parquet"


def present_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    The subset of the feature schema actually present as columns on ``df``.

    The Phase 6 columns follow the 17 (or 19), so a ``p17`` frame resolves exactly as
    before, and so does every cache key built from this list.
    """
    return [c for c in FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS if c in df.columns]


def load_phase6_panels(feature_set: str) -> dict[str, pd.DataFrame]:
    """
    Read the raw panels a Phase 6 feature set needs. Per-stock panels are wide
    (date × ticker); the market panel is one (date × 3) frame under key ``"market"``.
    """
    names = FEATURE_SETS[feature_set]
    out: dict[str, pd.DataFrame] = {}
    for name in names:
        if name in F1_MARKET:
            continue
        sub = "f1" if name in F1_RANKED else "f2"
        path = os.path.join(PHASE6_PANEL_DIRS[sub], f"{name}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found — run `python src/phase6_features/build_{sub}.py --build`."
            )
        out[name] = pd.read_parquet(path)
    if any(n in F1_MARKET for n in names):
        if not os.path.exists(PHASE6_MARKET_PARQUET):
            raise FileNotFoundError(
                f"{PHASE6_MARKET_PARQUET} not found — run "
                "`python src/phase6_features/build_f1.py --build`."
            )
        out["market"] = pd.read_parquet(PHASE6_MARKET_PARQUET)
    for frame in out.values():
        frame.index = pd.DatetimeIndex(frame.index).normalize()
    return out


def _phase6_features(
    index: pd.MultiIndex,
    feature_set: str,
    universe_mask: pd.DataFrame | None,
    panels: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, list[str]]:
    """
    The Phase 6 columns for ``index`` (the bar frame's rows), and which are required.

    Required = every ranked F1 and market column: a row missing one is dropped, as for
    the 17. F2 is optional and filled, so it never removes a row.
    """
    names = FEATURE_SETS[feature_set]

    def _long(wide: pd.DataFrame) -> pd.Series:
        s = wide.rename_axis(index=DATE_LEVEL, columns=TICKER_LEVEL).stack(future_stack=True)
        return s.reindex(index)

    ranked = [n for n in names if n in F1_RANKED or n in F2_OPTIONAL]
    raw = pd.DataFrame({n: _long(panels[n]) for n in ranked}, index=index)
    if universe_mask is not None:
        raw = _apply_universe_mask(raw, universe_mask)
    # +inf (a loss-making sector's P/E, §B.8) is a value, not a gap: it ranks above
    # every finite P/E. Only NaN is missing.
    normed = _rank_normalize(raw)
    optional = [n for n in ranked if n in F2_OPTIONAL]
    if optional:
        normed[optional] = normed[optional].fillna(0.0)

    out = normed
    if any(n in F2_FLAG for n in names):
        out["opt_avail"] = _long(panels["opt_avail"]).fillna(0.0)
    market = [n for n in names if n in F1_MARKET]
    if market:
        dates = index.get_level_values(DATE_LEVEL)
        m = panels["market"][market].reindex(pd.DatetimeIndex(dates).normalize())
        for n in market:
            out[n] = m[n].to_numpy()
    required = [n for n in names if n in F1_RANKED or n in F1_MARKET]
    return out[names], required


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
def _compute_raw_features(df: pd.DataFrame, with_institutional: bool) -> pd.DataFrame:
    """
    Compute the raw (un-normalized) feature panel from the augmented frame.

    All windows are trailing bar counts, so this is frequency-agnostic. The
    institutional family is appended only when ``with_institutional`` and its
    columns are present.
    """
    close = df[CLOSE_COL]
    high = df[HIGH_COL]
    low = df[LOW_COL]
    volume = df[VOLUME_COL]

    # Per-bar log return is the atomic building block for several families.
    bar_logret = np.log(close / _shift(close, 1))

    feats = pd.DataFrame(index=df.index)

    # ---- Momentum & Reversal ---------------------------------------------- #
    # Multi-horizon cumulative log returns: log(C_t / C_{t-n}), n in bars.
    for h in LOGRET_HORIZONS:
        feats[f"mom_logret_{h}b"] = np.log(close / _shift(close, h))

    # Distance of price from its own trailing simple moving averages.
    for w in SMA_WINDOWS:
        feats[f"mom_close_sma{w}"] = close / _roll(close, w, "mean")

    # MAX proxy: largest single-bar return over the trailing window (lottery/reversal).
    feats[f"mom_max_dret_{MAXDRET_WINDOW}b"] = _roll(bar_logret, MAXDRET_WINDOW, "max")

    # ---- Volatility & Risk ------------------------------------------------ #
    # Realized volatility = rolling std of per-bar log returns.
    for w in VOL_WINDOWS:
        feats[f"vol_realized_{w}b"] = _roll(bar_logret, w, "std")

    # Parkinson high-low volatility: sqrt( k * mean( ln(H/L)^2 ) ), k = 1/(4 ln2).
    hl_sq = np.square(np.log(high / low))
    feats[f"vol_parkinson_{PARKINSON_WINDOW}b"] = np.sqrt(
        _PARKINSON_K * _roll(hl_sq, PARKINSON_WINDOW, "mean")
    )

    # Trailing peak drawdown: current close vs. rolling window high.
    peak = _roll(close, DRAWDOWN_WINDOW, "max")
    feats[f"vol_drawdown_{DRAWDOWN_WINDOW}b"] = (close / peak) - 1.0

    # ---- Liquidity & Volume ----------------------------------------------- #
    # Amihud illiquidity: |return| / (Close * Volume).
    feats["liq_amihud"] = bar_logret.abs() / (close * volume)

    # Rolling volume variance and turnover vs. its own trailing average.
    feats[f"liq_vol_var_{VOLVAR_WINDOW}b"] = _roll(volume, VOLVAR_WINDOW, "var")
    feats[f"liq_turnover_{TURNOVER_WINDOW}b"] = volume / _roll(volume, TURNOVER_WINDOW, "mean")

    # ---- Institutional (optional) ----------------------------------------- #
    if with_institutional:
        feats["inst_delivery_qty"] = df[DELIVERY_QTY_COL]
        feats["inst_delivery_pct"] = df[DELIVERY_PCT_COL]

    # Guard against divide-by-zero / log blowups before normalization so that
    # infinities cannot dominate the cross-sectional ranking.
    feats = feats.replace([np.inf, -np.inf], np.nan)
    return feats[feature_columns(with_institutional)]


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


def _apply_universe_mask(
    raw: pd.DataFrame, universe_mask: pd.DataFrame
) -> pd.DataFrame:
    """
    Blank out features for names outside that date's tradeable universe.

    Applied **between** the rolling stage and rank-normalization, and the ordering is
    the entire point. Each name's rolling statistics must still be computed on its own
    full history — a stock that entered the universe last quarter has a real 252-day
    volatility, and masking before the rolling stage would truncate it. But the
    cross-sectional rank has to be taken over *only* the names that were actually
    tradeable that day, or the normalized values encode a peer group the strategy could
    never have held.

    Masked cells become NaN and are dropped by the existing all-features-present rule,
    so an out-of-universe bar never reaches the trees at all.

    Args:
        raw: MultiIndex ('date', 'ticker') frame of un-normalized features.
        universe_mask: wide (date × ticker) boolean frame. Absent dates or tickers are
            treated as out-of-universe — a mask that does not mention a name is not
            evidence that the name was tradeable.
    """
    flags = (
        universe_mask.astype(bool)
        .stack()
        .rename_axis([DATE_LEVEL, TICKER_LEVEL])
        .reindex(raw.index)
        .fillna(False)
        .to_numpy()
    )
    kept = int(flags.sum())
    print(
        f"[feature_creator] universe mask: {kept:,} of {len(raw):,} bars in universe "
        f"({kept / max(len(raw), 1):.1%})"
    )
    if kept == 0:
        raise ValueError(
            "universe_mask excluded every bar — check that its dates and tickers "
            "match the price panel (a tz-aware/naive mismatch does exactly this)"
        )
    return raw.where(pd.Series(flags, index=raw.index), np.nan)


def _compute_targets(df: pd.DataFrame, null_cross_session: bool) -> pd.DataFrame:
    """
    Forward log-return labels over ``TARGET_HORIZONS`` bars (never fed as features).

    When ``null_cross_session`` (intraday frequencies), a target that would span
    the overnight gap is nulled: the forward bar must fall on the same calendar
    date as the current bar, else the label is dropped. On daily bars this is
    off — the forward bar is by definition the next session, which is the target.
    """
    close = df[CLOSE_COL]
    targets = pd.DataFrame(index=df.index)

    # Calendar date of each bar, aligned to the index for per-ticker shifting.
    cal = pd.Series(
        pd.DatetimeIndex(df.index.get_level_values(DATE_LEVEL)).normalize(),
        index=df.index,
    )

    for h in TARGET_HORIZONS:
        tgt = np.log(_shift(close, -h) / close)
        if null_cross_session:
            fwd_cal = _shift(cal, -h)
            # Keep only labels whose forward bar resolves inside the same session.
            tgt = tgt.where(fwd_cal == cal)
        targets[f"tgt_fwd_logret_{h}b"] = tgt

    return targets.replace([np.inf, -np.inf], np.nan)


def create_features(
    augmented_df: pd.DataFrame,
    frequency: str = config.DEFAULT_FREQUENCY,
    universe_mask: pd.DataFrame | None = None,
    feature_set: str = "p17",
    phase6_panels: dict[str, pd.DataFrame] | None = None,
    label_end_dates: bool = False,
) -> pd.DataFrame:
    """
    Build the normalized feature matrix + forward-return targets for one frequency.

    Args:
        augmented_df: MultiIndex ('date', 'ticker') bar frame with OHLCV, and —
            for the full 19-feature schema — DeliveryQty / DeliveryPct. The
            ``date`` level holds each bar's timestamp (calendar date on daily
            data, intraday timestamp otherwise).
        frequency:    a key in ``config.FREQ_REGISTRY``. Determines whether the
            institutional family is expected and whether forward targets are
            nulled at session boundaries (intraday only).
        universe_mask: optional wide (date × ticker) boolean point-in-time universe.
            When given, rolling features are still computed on each name's full
            history but the cross-sectional rank-normalization — and therefore the
            emitted rows — are restricted to that date's tradeable set. ``None``
            (the default) leaves behaviour byte-for-byte identical to Phase 2/3, so
            the published results stay reproducible.
        feature_set: a key in ``FEATURE_SETS``. ``"p17"`` (the default) is the
            pre-Phase-6 matrix, byte for byte. ``"p17x"`` / ``"p17xo"`` append the
            Phase 6 columns (docs/phase6_plan.md §B.2) after the 17.
        phase6_panels: the raw Phase 6 panels, as ``load_phase6_panels`` returns them.
            Read from ``data/phase6/`` when None; passed in by the causality check,
            which builds them from perturbed inputs.
        label_end_dates: also emit ``TARGET_END_COLUMNS``, each label's realisation
            date, after the targets. Off by default, so the matrix is unchanged; the
            Tier 1 purge needs it on.

    Returns:
        MultiIndex ('date', 'ticker') DataFrame with the feature columns (19, or
        17 when delivery is unavailable) followed by the target columns. Rows
        with NaN/inf features or NaN forward labels (warm-up, undefined or
        cross-session labels) are dropped, so every returned row is fully
        populated and leakage-free.
    """
    if not isinstance(augmented_df.index, pd.MultiIndex):
        raise ValueError("augmented_df must have a MultiIndex ('date', 'ticker')")

    spec = config.freq_spec(frequency)
    have_delivery = {DELIVERY_QTY_COL, DELIVERY_PCT_COL} <= set(augmented_df.columns)
    with_institutional = spec.has_institutional and have_delivery
    null_cross_session = spec.bars_per_day > 1  # only intraday spans the gap

    if spec.has_institutional and not have_delivery:
        print(
            "[feature_creator] delivery columns absent — emitting price-only "
            f"({len(PRICE_FEATURE_COLUMNS)}) features."
        )

    # Sort so per-ticker rolling windows are chronologically ordered.
    df = augmented_df.sort_index()

    raw = _compute_raw_features(df, with_institutional)
    if universe_mask is not None:
        raw = _apply_universe_mask(raw, universe_mask)
    normed = _rank_normalize(raw)
    targets = _compute_targets(df, null_cross_session)
    if label_end_dates:
        stamp = pd.Series(df.index.get_level_values(DATE_LEVEL), index=df.index)
        for h in TARGET_HORIZONS:
            targets[f"tgt_end_{h}b"] = _shift(stamp, -h).where(
                targets[f"tgt_fwd_logret_{h}b"].notna())

    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set={feature_set!r} not in {sorted(FEATURE_SETS)}")
    feat_cols = feature_columns(with_institutional)
    if feature_set == "p17":
        out = pd.concat([normed, targets], axis=1)
    else:
        panels = phase6_panels if phase6_panels is not None else load_phase6_panels(feature_set)
        extra, required = _phase6_features(df.index, feature_set, universe_mask, panels)
        out = pd.concat([normed, extra, targets], axis=1)
        print(f"[feature_creator] feature_set={feature_set}: +{extra.shape[1]} Phase 6 "
              f"columns ({len(required)} required, {extra.shape[1] - len(required)} optional)")
        feat_cols = feat_cols + required

    before = len(out)
    # Require every feature (drops warm-up / inf-derived NaNs) and at least one
    # valid target. The two targets are decoupled on purpose: a row valid for the
    # 1-bar label is kept even if its 5-bar label is undefined (undefined for the
    # trailing bars, and — intraday — for labels that would span the overnight
    # gap). tier1_trees drops the remaining per-target NaNs inside each fold, so
    # the restrictive long-horizon label never discards short-horizon training
    # rows. Any surviving target NaN is left in place as a masked (untradeable) cell.
    feat_ok = out[feat_cols].notna().all(axis=1)
    tgt_ok = out[TARGET_COLUMNS].notna().any(axis=1)
    out = out[feat_ok & tgt_ok]
    after = len(out)

    tgt_nan = {c: int(out[c].isna().sum()) for c in TARGET_COLUMNS}
    print(
        f"[feature_creator] freq={frequency} | "
        f"features: {len(feat_cols)} | "
        f"targets: {len(TARGET_COLUMNS)} | cross-session-null={null_cross_session} | "
        f"rows {before} -> {after} ({before - after} warm-up/no-label rows removed) | "
        f"per-target NaN kept: {tgt_nan}"
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
    # Features must be fully populated; targets may carry masked (NaN) cells for
    # trailing / cross-session bars, so they are checked separately.
    no_feat_nan = not matrix[FEATURE_COLUMNS].isna().any().any()
    tgt_nan = {c: int(matrix[c].isna().sum()) for c in TARGET_COLUMNS}

    print()
    print("=" * 70)
    print(f"  feature global min : {feat_min:+.6f}")
    print(f"  feature global max : {feat_max:+.6f}")
    print(f"  [-1, +1] bounds    : {'PASS' if in_bounds else 'FAIL'}")
    print(f"  no NaN feature rows: {'PASS' if no_feat_nan else 'FAIL'}")
    print(f"  target NaN (masked): {tgt_nan}")
    print("=" * 70)

    # --- Phase 6 feature sets (docs/phase6_plan.md §B.2) ------------------- #
    checks: dict[str, bool] = {
        "[-1, +1] bounds": bool(in_bounds),
        "no NaN feature rows": bool(no_feat_nan),
        "feature_set='p17' == default (byte-identical)": bool(
            create_features(dummy, feature_set="p17").equals(matrix)
        ),
    }
    rng6 = np.random.default_rng(6)
    d6 = dummy.index.get_level_values(DATE_LEVEL).unique()
    t6 = dummy.index.get_level_values(TICKER_LEVEL).unique()

    def _wide6(fill: float | None = None) -> pd.DataFrame:
        w = pd.DataFrame(rng6.normal(size=(len(d6), len(t6))), index=d6, columns=t6)
        w.iloc[:30] = np.nan                                   # warm-up
        return w

    panels6 = {n: _wide6() for n in F1_RANKED + F2_OPTIONAL}
    panels6["sector_pe"].iloc[30:, 0] = np.inf                 # a loss-making sector
    panels6["beta_252b"].iloc[300:310, 2] = np.nan             # an F1 gap after warm-up
    no_opt = list(t6[5:])                                      # half the names: no options
    for n in F2_OPTIONAL:
        panels6[n][no_opt] = np.nan
    panels6["opt_avail"] = pd.DataFrame(1.0, index=d6, columns=t6)
    panels6["opt_avail"][no_opt] = 0.0
    panels6["market"] = pd.DataFrame(
        {n: rng6.normal(size=len(d6)) for n in F1_MARKET}, index=d6)
    m_x = create_features(dummy, feature_set="p17x", phase6_panels=panels6)
    m_xo = create_features(dummy, feature_set="p17xo", phase6_panels=panels6)
    checks["p17x adds the 14 F1 columns after the 17"] = bool(
        list(m_x.columns) == FEATURE_COLUMNS + FEATURE_SETS["p17x"] + TARGET_COLUMNS)
    checks["p17xo adds the 9 F2 columns after F1"] = bool(
        list(m_xo.columns) == FEATURE_COLUMNS + FEATURE_SETS["p17xo"] + TARGET_COLUMNS)
    checks["p17 columns unchanged inside p17x"] = bool(
        m_x[FEATURE_COLUMNS].equals(matrix[FEATURE_COLUMNS].reindex(m_x.index)))
    gap = pd.MultiIndex.from_product([d6[300:310], [t6[2]]], names=[DATE_LEVEL, TICKER_LEVEL])
    checks["an F1 gap drops exactly its rows (required)"] = bool(
        gap.isin(matrix.index).all() and not gap.isin(m_x.index).any()
        and len(matrix) - len(m_x) == len(gap))
    checks["F2 never drops a row (optional)"] = bool(m_xo.index.equals(m_x.index))
    checks["ranked extras in [-1, +1]"] = bool(
        m_xo[F1_RANKED + F2_OPTIONAL].abs().max().max() <= 1.0 + tol)
    checks["+inf sector P/E ranks at +1"] = bool(
        (m_x.xs(t6[0], level=TICKER_LEVEL)["sector_pe"] == 1.0).all())
    checks["F2 = 0 where a name has no options"] = bool(
        (m_xo.loc[(slice(None), no_opt), F2_OPTIONAL] == 0.0).all().all())
    checks["opt_avail raw 0/1"] = bool(set(np.unique(m_xo["opt_avail"])) <= {0.0, 1.0})
    mk = panels6["market"].reindex(m_x.index.get_level_values(DATE_LEVEL))
    checks["market features enter raw"] = bool(
        np.allclose(m_x[F1_MARKET].to_numpy(), mk.to_numpy()))
    mask6 = pd.DataFrame(True, index=d6, columns=t6)
    mask6[t6[1]] = False
    m_mask = create_features(dummy, universe_mask=mask6, feature_set="p17xo", phase6_panels=panels6)
    checks["universe mask removes a name's rows"] = bool(
        t6[1] not in m_mask.index.get_level_values(TICKER_LEVEL))
    m_end = create_features(dummy, label_end_dates=True)
    checks["label_end_dates adds only the 3 end columns"] = bool(
        list(m_end.columns) == list(matrix.columns) + TARGET_END_COLUMNS
        and m_end[list(matrix.columns)].equals(matrix))
    tk = t6[3]
    own_dates = dummy.xs(tk, level=TICKER_LEVEL).index
    row = m_end.xs(tk, level=TICKER_LEVEL).iloc[10]
    checks["tgt_end_5b = the name's own bar 5 ahead"] = bool(
        row["tgt_end_5b"] == own_dates[own_dates.get_loc(row.name) + 5])
    for name, ok in checks.items():
        print(f"  {name:<46}: {'PASS' if ok else 'FAIL'}")
    assert all(checks.values()), "feature_creator dry-run FAILED"
