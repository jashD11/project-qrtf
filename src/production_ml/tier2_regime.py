"""
Phase 2 Tier 2 — market-regime detector (PRODUCTION_ML).

A causal, market-aware Gaussian HMM that labels each trading day with a hidden
**market mood** (Calm / Panic, optionally a 3rd stress state). It fixes the two
flaws of the sandbox regime detector (``src/sandbox_run/tier2_regime.py``):

  1. *Look-ahead in the labels* — the sandbox fits one HMM on **all** history, so
     every regime label was assigned knowing the future. Here the model is refit
     walk-forward on a trailing window and each forward block is decoded by
     **forward filtering** (past-and-present only), so labels never peek ahead.
  2. *Single-series* — the sandbox reads one index and 2 features. Here 3 inputs
     span three distinct axes of market state, one of which needs the whole
     68-stock panel (herding is invisible to any single index).

Inputs (3, daily, causally z-scored)
    0  mkt_ret            market-index daily log return       — direction / level
    1  log_realized_vol   log 20d realized vol of the index   — nervousness / risk
    2  avg_corr           mean pairwise corr of the 68 stocks — herding (systemic
                          vs idiosyncratic); the one input a single index can't see

The market index is ``config.ML_CONFIG.regime.market_index`` (NIFTY-50 — full
history here; NIFTY-500 is too sparse in this dataset).

Hidden states
    Variance-ordered so State 0 is always lowest-vol (Calm) up to State n-1
    (Panic / Stress), regardless of the label EM happened to assign. n_states is
    generic (2 default, 3 sweepable via ``StrategyConfig.hmm_states``).

Output
    ``RegimeResult`` — a daily int ``states`` Series (0..n-1) plus a ``probs``
    DataFrame of per-state posteriors (the probabilities cost nothing now and are
    exactly what the deferred regime-as-tree-feature path will consume).

This pass builds the detector only. The execution/gate tier (masks + regime →
weights → returns) and orchestrator are designed-for but deferred; a t-1 intraday
broadcast helper is included so intraday strategies can later inherit the daily
regime with no look-ahead.
"""

import os
import sys
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# Make the repo root importable whether run directly or by an orchestrator.
_ROOT: str = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config

# --- Feature layout (column order of the observation matrix) --------------- #
FEATURE_COLUMNS: Final[list[str]] = ["mkt_ret", "log_realized_vol", "avg_corr"]
VOL_IDX: Final[int] = 1  # index of log_realized_vol — the variance-ordering key

DATE_LEVEL: Final[str] = "date"
TICKER_LEVEL: Final[str] = "ticker"
_TS_COL: Final[str] = "timestamp"
_CLOSE_COL: Final[str] = "close"


@dataclass
class RegimeResult:
    """Full walk-forward output of the regime detector."""
    states: pd.Series      # int 0..n-1 (0=Calm … n-1=Panic), indexed by daily date
    probs: pd.DataFrame    # per-state posterior probabilities, same index; cols 0..n-1


# --------------------------------------------------------------------------- #
# Data loading — one index (market backbone) + the 68-stock panel (herding)
# --------------------------------------------------------------------------- #
def _load_index_daily(cfg) -> pd.Series:
    """Daily close of the broad market index, resampled from the 15-min index parquet."""
    idx = pd.read_parquet(cfg.index_parquet)
    idx = idx[idx[TICKER_LEVEL] == cfg.market_index]
    if idx.empty:
        raise ValueError(
            f"market_index={cfg.market_index!r} not found in {cfg.index_parquet!r}"
        )
    day = idx[_TS_COL].dt.normalize()
    # Session close = last 15-min close of the day; one row per calendar date.
    daily_close = idx.assign(**{DATE_LEVEL: day}).groupby(DATE_LEVEL)[_CLOSE_COL].last()
    return daily_close.sort_index()


def _load_stock_close_wide(cfg) -> pd.DataFrame:
    """Wide (date × ticker) daily close matrix for the tradeable stock panel."""
    df = pd.read_parquet(cfg.stock_parquet)
    day = df[_TS_COL].dt.normalize()
    wide = (
        df.assign(**{DATE_LEVEL: day})
        .pivot_table(index=DATE_LEVEL, columns=TICKER_LEVEL, values=_CLOSE_COL, aggfunc="last")
        .sort_index()
    )
    return wide


# --------------------------------------------------------------------------- #
# The 3 inputs
# --------------------------------------------------------------------------- #
def _rolling_avg_corr(returns_wide: pd.DataFrame, window: int) -> pd.Series:
    """
    Rolling mean of the pairwise correlations across the stock universe.

    For each day, take the trailing ``window`` daily returns of every stock, form
    the correlation matrix, and average its off-diagonal. Stocks not yet listed
    (all-NaN in the window) are dropped from that day's matrix. This is an O(days)
    loop — cheap (~2.5k days × a 68×68 corr) and the only non-vectorized step;
    everything else is column algebra.
    """
    vals = returns_wide.to_numpy()
    n_days = len(returns_wide)
    out = np.full(n_days, np.nan)
    for t in range(window - 1, n_days):
        w = vals[t - window + 1 : t + 1]           # (window × n_stocks)
        keep = ~np.isnan(w).any(axis=0)             # stocks with a full window
        wm = w[:, keep]
        if wm.shape[1] < 2:
            continue
        c = np.corrcoef(wm, rowvar=False)           # (k × k)
        k = c.shape[0]
        out[t] = (np.nansum(c) - k) / (k * (k - 1)) # mean off-diagonal
    return pd.Series(out, index=returns_wide.index, name="avg_corr")


def build_features(cfg=None) -> pd.DataFrame:
    """
    Build the 3 raw daily regime inputs, aligned on a common date index.

    Returns a DataFrame indexed by daily date with columns FEATURE_COLUMNS. Raw
    (un-standardized) — the walk-forward driver z-scores causally before fitting.
    """
    cfg = cfg or config.ML_CONFIG.regime

    index_close = _load_index_daily(cfg)
    stock_close = _load_stock_close_wide(cfg)

    mkt_ret = np.log(index_close).diff().rename("mkt_ret")
    realized_vol = mkt_ret.rolling(cfg.vol_window, min_periods=cfg.vol_window).std()
    log_rv = np.log(realized_vol.where(realized_vol > 0)).rename("log_realized_vol")

    stock_ret = np.log(stock_close).diff()
    avg_corr = _rolling_avg_corr(stock_ret, cfg.corr_window)

    feats = pd.concat([mkt_ret, log_rv, avg_corr], axis=1)
    feats = feats.replace([np.inf, -np.inf], np.nan)[FEATURE_COLUMNS]
    return feats


def _causal_zscore(feats: pd.DataFrame, min_periods: int) -> pd.DataFrame:
    """
    Standardize each input with **expanding** (past-and-present) mean/std, so the
    scaling at day t never uses future data. Puts the 3 very differently-scaled
    inputs (returns ~1e-2, log-vol ~1e0, corr ~0.5) on equal footing for the
    Gaussian, leak-free. Rows before ``min_periods`` are undefined and dropped.
    """
    exp = feats.expanding(min_periods=min_periods)
    z = (feats - exp.mean()) / exp.std()
    return z.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# Causal decode — forward filtering (no backward smoothing → no look-ahead)
# --------------------------------------------------------------------------- #
def _logsumexp(a: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
    """Numerically stable log-sum-exp (avoids a hard scipy call site)."""
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def _forward_filter(model: GaussianHMM, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Filtered state path + posteriors via the forward recursion only.

    Unlike Viterbi (global MAP) or forward-backward (uses future observations),
    the filtered posterior at day t, P(state_t | obs_0..t), depends on **past and
    present only** — the causal quantity we want. Runs over the full [train+block]
    sequence so the block's belief is carried forward from the train history;
    callers keep just the block rows.
    """
    framelogprob = model._compute_log_likelihood(obs)   # (T, S) log emission probs
    log_trans = np.log(model.transmat_)
    log_start = np.log(model.startprob_)
    T, S = framelogprob.shape

    log_alpha = np.empty((T, S))
    log_alpha[0] = log_start + framelogprob[0]
    for t in range(1, T):
        # a_t[j] = ( Σ_i a_{t-1}[i]·trans[i,j] ) · b_j(x_t)   (in log space)
        log_alpha[t] = _logsumexp(
            log_alpha[t - 1][:, None] + log_trans, axis=0
        ) + framelogprob[t]

    log_post = log_alpha - _logsumexp(log_alpha, axis=1, keepdims=True)
    states = np.argmax(log_post, axis=1)
    return states, np.exp(log_post)


def _variance_order(model: GaussianHMM) -> np.ndarray:
    """
    Map raw EM state labels → canonical order (0 = lowest vol … n-1 = highest).

    Returns ``remap`` such that ``remap[raw_label]`` is the canonical label, using
    the fitted mean of the log-realized-vol input as the ordering key.
    """
    order = np.argsort(model.means_[:, VOL_IDX])   # raw labels, ascending vol
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order))
    return remap


# --------------------------------------------------------------------------- #
# Walk-forward detector
# --------------------------------------------------------------------------- #
class RegimeDetector:
    """
    Walk-forward Gaussian-HMM market-regime detector.

    Fits on a trailing ``train_days`` window, decodes the next ``refit_every``
    block by forward filtering, then slides — so every label is leak-free. States
    are variance-ordered (0=Calm … n-1=Panic) each refit for stability.
    """

    def __init__(self, n_states: int = 2, cfg=None) -> None:
        if n_states < 2:
            raise ValueError(f"n_states must be >= 2, got {n_states}")
        self.n_states = n_states
        self.cfg = cfg or config.ML_CONFIG.regime

    def _fit_block(self, train_obs: np.ndarray) -> GaussianHMM:
        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.cfg.n_iter,
            random_state=self.cfg.random_state,
        )
        model.fit(train_obs)
        return model

    def fit_predict(self, features: pd.DataFrame) -> RegimeResult:
        """
        Args:
            features: daily DataFrame with FEATURE_COLUMNS (raw; z-scored here).

        Returns:
            RegimeResult with canonical daily states + per-state posteriors,
            starting one train window after the first standardizable day.
        """
        z = _causal_zscore(features[FEATURE_COLUMNS], self.cfg.zscore_min_periods).dropna()
        dates = z.index.to_numpy()
        obs_all = z.to_numpy()
        n = len(z)

        train_days, block = self.cfg.train_days, self.cfg.refit_every
        if n < train_days + 1:
            raise ValueError(
                f"Need >= {train_days + 1} standardized days; got {n}. "
                "Lower regime.train_days / zscore_min_periods or add history."
            )

        state_parts: list[pd.Series] = []
        prob_parts: list[pd.DataFrame] = []
        n_folds = 0

        start = train_days
        while start < n:
            lo = start - train_days
            hi = min(start + block, n)                 # block = [start, hi)
            model = self._fit_block(obs_all[lo:start])
            remap = _variance_order(model)

            # Filter over [train + block]; keep only the block's causal labels.
            seq = obs_all[lo:hi]
            raw_states, raw_post = _forward_filter(model, seq)
            b0 = start - lo                            # block offset within seq
            block_states = remap[raw_states[b0:]]
            block_post = raw_post[b0:][:, np.argsort(remap)]  # reorder cols → canonical
            block_dates = dates[start:hi]

            state_parts.append(pd.Series(block_states, index=block_dates))
            prob_parts.append(pd.DataFrame(block_post, index=block_dates))
            n_folds += 1
            start += block

        states = pd.concat(state_parts).astype(int)
        states.index.name = DATE_LEVEL
        states.name = "regime"
        probs = pd.concat(prob_parts)
        probs.index.name = DATE_LEVEL
        probs.columns = [f"p_state{c}" for c in range(self.n_states)]

        dist = states.value_counts(normalize=True).sort_index()
        dist_str = " | ".join(f"S{s}={p:.1%}" for s, p in dist.items())
        print(
            f"[tier2_regime] walk-forward done | n_states={self.n_states} | "
            f"folds={n_folds} | days={len(states)} | {dist_str}"
        )
        return RegimeResult(states=states, probs=probs)


# --------------------------------------------------------------------------- #
# t-1 intraday broadcast (designed-for the deferred execution tier)
# --------------------------------------------------------------------------- #
def broadcast_to_intraday(
    daily_states: pd.Series, bar_timestamps: pd.DatetimeIndex
) -> pd.Series:
    """
    Map each intraday bar to the most recent **completed** daily regime (t-1).

    A bar during day d inherits the regime for day d-1 (known at the prior close),
    the same lagged-broadcast pattern as delivery — no intraday look-ahead. Bars
    before the first available regime get NaN.
    """
    lagged = daily_states.shift(1)                     # day d carries d-1's label
    lagged.index = pd.DatetimeIndex(lagged.index).normalize()
    bar_days = pd.DatetimeIndex(bar_timestamps).normalize()
    return pd.Series(lagged.reindex(bar_days).to_numpy(), index=bar_timestamps, name="regime")


# --------------------------------------------------------------------------- #
# Dry-run verification on the real data
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time

    print("=" * 70)
    print("tier2_regime dry-run — causal market-regime HMM on real NSE data")
    print("=" * 70)

    feats = build_features()
    print(f"  raw features   : {feats.shape} | cols={FEATURE_COLUMNS}")
    print(f"  date span      : {feats.index.min().date()} → {feats.index.max().date()}")

    t0 = time.time()
    detector = RegimeDetector(n_states=2)
    result = detector.fit_predict(feats)
    elapsed = time.time() - t0
    print(f"  HMM fit+decode : {elapsed:.1f}s")

    states = result.states

    # --- Face validity: the Mar-2020 COVID crash must read as Panic ---------- #
    covid = states.loc["2020-03-01":"2020-04-15"]
    covid_panic = float((covid == states.max()).mean()) if len(covid) else float("nan")
    print()
    print(f"  COVID window (2020-03-01→04-15): {len(covid)} days | "
          f"Panic share {covid_panic:.0%}")

    # --- Sanity: Calm is the majority, Panic a minority ---------------------- #
    calm_share = float((states == 0).mean())
    panic_share = float((states == states.max()).mean())
    print(f"  overall        : Calm {calm_share:.0%} | Panic {panic_share:.0%}")

    # --- Causality: truncating the future must not change past labels -------- #
    cut = feats.index[int(len(feats) * 0.7)]
    trunc = RegimeDetector(n_states=2).fit_predict(feats.loc[:cut]).states
    common = states.index.intersection(trunc.index)
    # Compare only days safely before the truncation's final refit block.
    safe = common[common < cut - pd.Timedelta(days=200)]
    match = bool((states.loc[safe] == trunc.loc[safe]).all())
    print(f"  causality check: {len(safe)} pre-cut days identical after truncation "
          f"→ {'PASS' if match else 'FAIL'}")

    # --- Broadcast helper smoke (real intraday bars are tz-aware IST) -------- #
    fake_bars = pd.date_range(
        "2021-01-04 09:15", "2021-01-06 15:15", freq="60min", tz="Asia/Kolkata"
    )
    bcast = broadcast_to_intraday(states, fake_bars)
    print(f"  broadcast smoke: {bcast.notna().sum()}/{len(bcast)} intraday bars labelled")

    print()
    print("  note: a 2-state HMM bisects volatility, so the upper state (~45%) is")
    print("        'elevated-vol/risk-off', not just crashes; sweep hmm_states=3 to")
    print("        carve off a smaller top-vol stress state for a sharper gate.")
    print("=" * 70)
    checks = {
        "COVID reads as Panic (>50%)": covid_panic > 0.5,
        "Calm is the majority": calm_share > 0.5,
        "Panic is a minority (<50%)": 0.0 < panic_share < 0.5,
        "Causality (no look-ahead)": match,
        "Intraday broadcast labels bars": bool(bcast.notna().any()),
    }
    for name, ok in checks.items():
        print(f"  {name:<32}: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
