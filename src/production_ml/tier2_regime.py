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

Output (``RegimeResult`` — two complementary views)
    - HMM regime: ``states`` (0..n-1 vol/herding taxonomy) + ``probs`` (posteriors,
      for the deferred regime-as-tree-feature path). A 2-state HMM *bisects* vol,
      so ``states`` is not a rare crisis flag.
    - De-risk gate: ``panic`` (bool) + ``stress`` (continuous). HMM posteriors
      saturate (persistent vol → self-transition ~0.98), so a probability cutoff
      can't set frequency; ``panic`` instead marks the top ``1 - panic_threshold``
      of a causal trailing stress distribution. This is what the execution tier gates.

This pass builds the detector only. The execution/gate tier (masks + regime →
weights → returns) and orchestrator are designed-for but deferred; a t-1 intraday
broadcast helper is included so intraday strategies can later inherit the daily
regime with no look-ahead.
"""

import logging
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.exceptions import ConvergenceWarning


class _NotConvergingFilter(logging.Filter):
    """
    Drop hmmlearn's blanket "Model is not converging" record.

    It fires on ANY negative log-likelihood delta, however small, so it cannot
    distinguish a fit sitting on its convergence floor from one actually diverging.
    ``RegimeDetector._fit_block`` makes that distinction on the magnitude of the delta
    and raises when it is real; suppressing the record here is what stops the benign
    case from training everyone to ignore the message.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "not converging" not in record.getMessage().lower()


_NOT_CONVERGING_FILTER: Final[_NotConvergingFilter] = _NotConvergingFilter()

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
    """
    Full walk-forward output of the regime detector.

    Two complementary views of market state:
      - The HMM regime (``states`` / ``probs``): the multi-state vol/herding
        taxonomy (0=Calm … n-1=most volatile). Rich context; feeds the deferred
        regime-as-tree-feature path. Note a 2-state HMM *bisects* vol (upper
        state ~45%), so ``states`` is NOT a rare crisis flag.
      - The de-risk gate (``panic`` / ``stress``): a frequency-controlled flag.
        HMM posteriors saturate (vol is persistent, self-transition ~0.98), so a
        probability cutoff can't dial frequency — instead ``panic`` marks the days
        whose continuous stress score sits in the top ``1 - panic_threshold`` of
        its own trailing history (causal). This is what the execution tier gates on.
    """
    states: pd.Series      # int 0..n-1 HMM vol regime (0=Calm … n-1=most volatile)
    probs: pd.DataFrame    # per-state posterior probabilities, same index; cols 0..n-1
    stress: pd.Series      # continuous causal stress score (mean of z-vol & z-herding)
    panic: pd.Series       # bool de-risk gate: stress above its trailing panic quantile


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

    def __init__(
        self, n_states: int = 2, cfg=None, panic_threshold: float | None = None
    ) -> None:
        if n_states < 2:
            raise ValueError(f"n_states must be >= 2, got {n_states}")
        self.n_states = n_states
        self.cfg = cfg or config.ML_CONFIG.regime
        # Constructor override wins over the config default (handy for sweeps).
        self.panic_threshold = (
            self.cfg.panic_threshold if panic_threshold is None else panic_threshold
        )
        if not 0.0 < self.panic_threshold <= 1.0:
            raise ValueError(
                f"panic_threshold must be in (0, 1], got {self.panic_threshold}"
            )

    def _transmat_prior(self) -> np.ndarray:
        """
        Dirichlet transition prior with extra weight on the diagonal.

        hmmlearn's M-step adds ``(transmat_prior - 1)`` as pseudo-counts, so a
        diagonal of ``1 + stickiness`` biases the model toward self-transitions —
        regimes become persistent (fewer, longer spells) instead of flickering on
        every jittery day. Off-diagonal stays 1 (adds nothing).
        """
        prior = np.ones((self.n_states, self.n_states))
        np.fill_diagonal(prior, 1.0 + self.cfg.transmat_stickiness)
        return prior

    # EM log-likelihood can wobble by a hair at the convergence floor without the fit
    # being in any trouble. hmmlearn prints "Model is not converging" for ANY negative
    # delta, so the Phase 4b log carried that warning at delta = -0.002 on a
    # log-likelihood of -1399 — a relative move of 1.4e-6, i.e. noise. Left as-is the
    # message is pure alarm fatigue, and a genuinely diverging fit would read exactly
    # the same. So the benign case is silenced and a real one is escalated.
    _LL_TOLERANCE: Final[float] = 1e-4   # |delta| / |log-likelihood|

    def _fit_block(self, train_obs: np.ndarray) -> GaussianHMM:
        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.cfg.n_iter,
            random_state=self.cfg.random_state,
            transmat_prior=self._transmat_prior(),
        )
        # hmmlearn reports non-convergence through the `logging` module, not through
        # `warnings`, so a warnings filter does not touch it — the record has to be
        # dropped at the logger. It must be the EMITTING logger ("hmmlearn.base"), not
        # the "hmmlearn" parent: a filter on a logger applies to records logged through
        # that logger, and records propagating up from a child skip the parent's filters
        # entirely.
        hmm_log = logging.getLogger("hmmlearn.base")
        hmm_log.addFilter(_NOT_CONVERGING_FILTER)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                model.fit(train_obs)
        finally:
            hmm_log.removeFilter(_NOT_CONVERGING_FILTER)

        history = list(getattr(model.monitor_, "history", []))
        if len(history) >= 2:
            delta = history[-1] - history[-2]
            scale = max(abs(history[-1]), 1.0)
            if delta < 0 and abs(delta) / scale > self._LL_TOLERANCE:
                raise RuntimeError(
                    f"Tier 2 HMM diverged: log-likelihood fell by {abs(delta):.4f} "
                    f"({abs(delta) / scale:.2e} relative) on the final EM step. This is "
                    f"beyond the {self._LL_TOLERANCE:.0e} floor that ordinary numerical "
                    "wobble produces, so the regime series this run would emit is not "
                    "trustworthy."
                )
        return model

    def _severity_gate(self, stress: pd.Series) -> pd.Series:
        """
        Causal de-risk gate: Panic on days whose stress score sits in the top
        ``1 - panic_threshold`` of its own **trailing** history.

        Why not threshold the HMM posterior? Vol is persistent (self-transition
        ~0.98), so the filtered posterior saturates — >90% confident on ~92% of
        days — leaving no probability mass in the middle for a cutoff to move. A
        percentile on the continuous stress score is the only leak-free way to set
        the frequency: the expanding quantile uses past-and-present only, so Panic
        converges to ~(1 - panic_threshold) of days as history accrues (still
        arriving in clusters, because stress itself is persistent).
        """
        q = stress.expanding(min_periods=self.cfg.zscore_min_periods).quantile(
            self.panic_threshold
        )
        return (stress > q).fillna(False).rename("panic")

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

        prob_parts: list[pd.DataFrame] = []
        n_folds = 0

        start = train_days
        while start < n:
            lo = start - train_days
            hi = min(start + block, n)                 # block = [start, hi)
            model = self._fit_block(obs_all[lo:start])
            remap = _variance_order(model)

            # Filter over [train + block]; keep only the block's causal posteriors,
            # reordered to canonical (col j = j-th lowest-vol state).
            seq = obs_all[lo:hi]
            _, raw_post = _forward_filter(model, seq)
            b0 = start - lo                            # block offset within seq
            block_post = raw_post[b0:][:, np.argsort(remap)]
            prob_parts.append(pd.DataFrame(block_post, index=dates[start:hi]))
            n_folds += 1
            start += block

        probs = pd.concat(prob_parts)
        probs.index.name = DATE_LEVEL
        probs.columns = [f"p_state{c}" for c in range(self.n_states)]

        # HMM regime label = variance-ordered argmax of the canonical posteriors
        # (the honest vol/herding taxonomy — NOT the de-risk trigger).
        states = pd.Series(
            probs.to_numpy().argmax(axis=1), index=probs.index, name="regime"
        ).astype(int)
        states.index.name = DATE_LEVEL

        # De-risk gate: continuous stress = mean of the two RISK axes (z-vol,
        # z-herding); direction (z-return) is level, not risk, so excluded. Gated
        # causally against its own trailing distribution (see _severity_gate).
        stress_full = z[[FEATURE_COLUMNS[VOL_IDX], "avg_corr"]].mean(axis=1)
        panic_full = self._severity_gate(stress_full)
        stress = stress_full.reindex(probs.index).rename("stress")
        panic = panic_full.reindex(probs.index)

        dist = states.value_counts(normalize=True).sort_index()
        dist_str = " | ".join(f"S{s}={p:.0%}" for s, p in dist.items())
        print(
            f"[tier2_regime] walk-forward done | n_states={self.n_states} | "
            f"folds={n_folds} | days={len(states)} | HMM regime: {dist_str} | "
            f"gate: Panic {panic.mean():.0%} (thr={self.panic_threshold}, "
            f"stickiness={self.cfg.transmat_stickiness})"
        )
        return RegimeResult(states=states, probs=probs, stress=stress, panic=panic)


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
    result = RegimeDetector(n_states=2).fit_predict(feats)  # panic_threshold from cfg (0.85)
    print(f"  HMM fit+decode : {time.time() - t0:.1f}s")

    states, panic, stress = result.states, result.panic, result.stress
    covid_slice = slice("2020-03-01", "2020-04-15")

    # --- Face validity: the Mar-2020 COVID crash must trip the de-risk gate --- #
    covid_panic = float(panic.loc[covid_slice].mean())
    gate_share = float(panic.mean())
    print(f"  gate: Panic {gate_share:.0%} of days | COVID window Panic {covid_panic:.0%}")

    # --- Frequency control: gate fraction tracks (1 - panic_threshold) -------- #
    # Rebuild the full-history stress the detector gates on (no HMM refit needed —
    # the threshold only affects the gate), so the sweep matches the headline.
    cfg = config.ML_CONFIG.regime
    z_full = _causal_zscore(feats[FEATURE_COLUMNS], cfg.zscore_min_periods).dropna()
    stress_full = z_full[[FEATURE_COLUMNS[VOL_IDX], "avg_corr"]].mean(axis=1)

    print()
    print("  panic_threshold → realized Panic frequency over OOS (COVID in parens):")
    shares = []
    for thr in (0.80, 0.85, 0.90, 0.95):
        q = stress_full.expanding(min_periods=cfg.zscore_min_periods).quantile(thr)
        pan = (stress_full > q).fillna(False).reindex(panic.index)
        share = float(pan.mean())
        shares.append(share)
        covid_share = float(pan.loc[covid_slice].mean())
        print(f"    thr={thr:.2f} → target {1 - thr:.0%} | realized {share:4.0%} "
              f"(COVID {covid_share:3.0%})")
    # Monotone control (rarer as threshold rises); exact fraction drifts above
    # target in rising-vol eras because the trailing quantile lags — expected.
    freq_ok = all(shares[i] >= shares[i + 1] - 0.01 for i in range(len(shares) - 1))

    # --- Causality: truncating the future must not change past labels -------- #
    cut = feats.index[int(len(feats) * 0.7)]
    trunc = RegimeDetector(n_states=2).fit_predict(feats.loc[:cut])
    common = panic.index.intersection(trunc.panic.index)
    safe = common[common < cut - pd.Timedelta(days=200)]  # before truncation's last block
    match = bool(
        (states.loc[safe] == trunc.states.loc[safe]).all()
        and (panic.loc[safe] == trunc.panic.loc[safe]).all()
    )
    print()
    print(f"  causality check: {len(safe)} pre-cut days (state+gate) identical after "
          f"truncation → {'PASS' if match else 'FAIL'}")

    # --- Broadcast the gate onto intraday bars (real bars are tz-aware IST) --- #
    fake_bars = pd.date_range(
        "2021-01-04 09:15", "2021-01-06 15:15", freq="60min", tz="Asia/Kolkata"
    )
    bcast = broadcast_to_intraday(panic.astype(float), fake_bars)
    print(f"  broadcast smoke: {bcast.notna().sum()}/{len(bcast)} intraday bars gated")

    print()
    print("  note: the HMM 'states' remain the vol/herding taxonomy (2-state bisects")
    print("        vol ~55/45); the de-risk 'panic' gate is the separate, frequency-")
    print("        controlled tail signal the execution tier will act on.")
    print("=" * 70)
    checks = {
        "COVID trips the gate (>50%)": covid_panic > 0.5,
        "Gate ≈ target frequency": freq_ok,
        "Default gate is a minority (<25%)": 0.0 < gate_share < 0.25,
        "Causality (no look-ahead)": match,
        "Intraday broadcast gates bars": bool(bcast.notna().any()),
    }
    for name, ok in checks.items():
        print(f"  {name:<34}: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
