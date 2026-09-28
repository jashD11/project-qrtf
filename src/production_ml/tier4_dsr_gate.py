"""
Phase 2 Tier 4 — Deflated Sharpe Ratio credibility gate (PRODUCTION_ML).

``tier4_dsr.py`` *writes* strategy return series into a Parquet ledger; this
module *judges* them. It answers the question the raw annualized Sharpe cannot:
given how many strategies were tried, the track length, and the non-normality of
the returns, is a given Sharpe distinguishable from luck?

The instrument is the Deflated Sharpe Ratio (Bailey & López de Prado 2014):

  - PSR(SR*) — Probabilistic Sharpe Ratio: P(true SR > SR*), correcting the
    observed Sharpe for track length T and for skew / (non-normal) kurtosis.
  - Expected maximum Sharpe under the null — the benchmark SR* you would expect
    the *best* of N independent random trials to post by chance alone. Deflating
    against this is exactly the multiple-testing correction the project's design
    philosophy demands (docs/phase3_design_requirements.md §0: "a grid of N cells
    is N implicit backtests").
  - DSR = PSR(expected-max-Sharpe). A strategy "passes" when DSR exceeds the
    threshold (default 0.95): >95% confidence its true Sharpe beats what the
    luckiest of N random trials would have produced.

Multi-frequency handling: return series live at different bar frequencies whose
per-bar Sharpes are not comparable, so every column is first compounded to
**daily** returns. All statistics — and the cross-trial Sharpe variance that
sets the deflation — are then computed on one common (daily) unit, and the
deflation uses N = the honest number of cells searched across the whole grid.

Usage
    python src/production_ml/tier4_dsr_gate.py        # dry-run + real ledger
    (from the orchestrator) run_dsr_gate(ledger_path, n_trials=...)
"""

import os
import sys
from typing import Final

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

# Make the repo root importable whether run directly or by an orchestrator.
_ROOT: str = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config

_EULER_GAMMA: Final[float] = 0.5772156649015329  # Euler–Mascheroni constant
_ANN: Final[float] = 252.0 ** 0.5                 # daily -> annual Sharpe factor


# --------------------------------------------------------------------------- #
# Unit alignment — everything to daily
# --------------------------------------------------------------------------- #
def _to_daily(returns: pd.Series) -> pd.Series:
    """
    Compound a per-bar return series to per-day returns.

    Intraday bars within a calendar day are geometrically compounded
    ((1+r).prod() - 1); an already-daily column (one bar per day) passes through
    unchanged. Puts every strategy on one comparable unit before scoring.
    """
    r = returns.dropna()
    if r.empty:
        return r
    day = pd.DatetimeIndex(r.index).normalize()
    daily = (1.0 + r).groupby(day).prod() - 1.0
    daily.index.name = "date"
    return daily


# --------------------------------------------------------------------------- #
# Lo (2002) autocorrelation correction (Phase 4c D2)
# --------------------------------------------------------------------------- #
def variance_ratio(returns: pd.Series, horizon: int) -> float:
    """
    Lo-MacKinlay overlapping variance ratio at ``horizon`` days.

        VR(m) = Var(m-day return) / (m · Var(1-day return))

    VR = 1 under i.i.d. returns. VR > 1 means multi-day variance compounds faster than
    linearly — the signature of positive autocorrelation. Overlapping m-day sums are
    used (every start date, not every m-th) because the non-overlapping estimator
    throws away a factor of ``m`` of the sample for no gain in consistency.
    """
    r = returns.dropna()
    t = int(r.shape[0])
    if t <= horizon + 1 or horizon < 2:
        return 1.0
    x = r.to_numpy(dtype=float)
    mu = float(x.mean())
    var_1 = float(((x - mu) ** 2).sum() / (t - 1))
    if var_1 <= 0:
        return 1.0
    sums = np.convolve(x, np.ones(horizon), mode="valid")
    var_m = float(
        ((sums - horizon * mu) ** 2).sum()
        / (horizon * (t - horizon + 1) * (1.0 - horizon / t))
    )
    return var_m / var_1 if var_m > 0 else 1.0


def lo_scaling_factor(returns: pd.Series, horizon: int = 21) -> float:
    """
    Lo (2002) autocorrelation-corrected Sharpe scaling, as a share of the naive one.

    The usual ``SR_annual = SR_daily · sqrt(252)`` is valid only for i.i.d. returns,
    because it assumes multi-period variance grows linearly in time. These returns are
    not i.i.d. — positions persist for days under a multi-day target — so the naive
    scaling **overstates** the Sharpe. The corrected factor is ``1 / sqrt(VR)``:
    positive autocorrelation gives VR > 1 and a factor below 1, so the adjustment
    always moves against a strategy that holds its positions. It is not a tuning knob;
    it is a bias being removed.

    **Why a variance ratio and not the lag-by-lag sum.** Lo's formula can be written
    as ``eta(q) = q / sqrt(q + 2·sum_k (q-k)·rho_k)``, and evaluating it directly is
    the obvious implementation — but it multiplies every estimated ``rho_k`` by
    ~``q`` = 252. Each ``rho_k`` carries a sampling error of ~``1/sqrt(T)`` ~ 0.02, so
    ten lags of pure noise move the result by over 10%: measured on i.i.d. synthetic
    returns that form returned factors from 0.99 to 1.18 depending only on the seed.
    It was estimating its own noise. The variance ratio estimates the same quantity as
    one number and is well behaved — on i.i.d. input it centres on 1.000 (sd 0.05),
    and on AR(1) with rho=0.15 it recovers 0.866 against a theoretical 0.860.

    ``horizon`` = 21 days: long enough to contain the whole life of a position under
    the slowest target in the frozen grid, short enough to keep the estimator tight.
    Returns 1.0 for a series too short to estimate.
    """
    vr = variance_ratio(returns, horizon)
    if not np.isfinite(vr) or vr <= 0:
        return 1.0
    return float(1.0 / np.sqrt(vr))


def lag1_autocorr(returns: pd.Series) -> float:
    """Lag-1 autocorrelation — reported alongside the correction it drives."""
    r = returns.dropna()
    if r.shape[0] < 3:
        return float("nan")
    return float(r.autocorr(lag=1))


# --------------------------------------------------------------------------- #
# The statistics
# --------------------------------------------------------------------------- #
def probabilistic_sharpe_ratio(
    sr: float, n_obs: int, skew_: float, kurt: float, sr_star: float
) -> float:
    """
    PSR(sr_star) = Phi( (sr - sr_star)·sqrt(n-1) / sqrt(1 - g3·sr + (g4-1)/4·sr²) ).

    All Sharpes are per-period (daily here). ``kurt`` is NON-excess (Gaussian=3),
    so the denominator reduces to the familiar sqrt(1 + 0.5·sr²) for normal
    returns. Returns P(true SR > sr_star) in [0, 1].
    """
    if n_obs < 2:
        return float("nan")
    denom = 1.0 - skew_ * sr + ((kurt - 1.0) / 4.0) * sr ** 2
    if denom <= 0:
        return float("nan")
    z = (sr - sr_star) * np.sqrt(n_obs - 1) / np.sqrt(denom)
    return float(norm.cdf(z))


def expected_max_sharpe(sharpe_std: float, n_trials: int) -> float:
    """
    Expected maximum of N i.i.d. standard-normal-scaled Sharpe estimates — the
    benchmark the *luckiest* of N random trials would be expected to post.

        SR* = sharpe_std · [ (1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e)) ]

    ``sharpe_std`` is the cross-trial stdev of the (daily) Sharpe estimates. With
    a single trial there is nothing to deflate against, so SR* = 0.
    """
    if n_trials < 2 or not np.isfinite(sharpe_std) or sharpe_std <= 0:
        return 0.0
    q1 = norm.ppf(1.0 - 1.0 / n_trials)
    q2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sharpe_std * ((1.0 - _EULER_GAMMA) * q1 + _EULER_GAMMA * q2))


def _min_track_record_length(
    sr: float, skew_: float, kurt: float, sr_star: float, threshold: float
) -> float:
    """
    Track length (in daily obs) needed for PSR(sr_star) to reach ``threshold``.
    NaN when sr does not clear sr_star (the gate is unreachable at any length).
    """
    if sr <= sr_star:
        return float("nan")
    denom = 1.0 - skew_ * sr + ((kurt - 1.0) / 4.0) * sr ** 2
    if denom <= 0:
        return float("nan")
    return float(1.0 + denom * (norm.ppf(threshold) / (sr - sr_star)) ** 2)


def deflated_sharpe_ratio(
    daily_matrix: pd.DataFrame,
    n_trials: int | None = None,
    threshold: float = 0.95,
    benchmark_sharpe: float = 0.0,
    autocorr_adjust: bool | None = None,
    autocorr_horizon: int | None = None,
) -> pd.DataFrame:
    """
    Per-strategy DSR verdict for a matrix of DAILY return series (one column
    each). Returns a frame indexed by strategy_id, sorted by DSR descending.

    When ``autocorr_adjust`` (D2), each column's per-period Sharpe is scaled by its
    own Lo (2002) factor **before** anything else is computed — so the correction
    flows into PSR, into DSR, into the cross-trial dispersion that sets SR*, and into
    the reported annual Sharpe, rather than being bolted on at the end. It is applied
    uniformly to every column, including the deflation benchmark's inputs.

    ``T`` is deliberately left at the raw observation count. Autocorrelation also
    shrinks the *effective* sample size, so leaving T alone understates the penalty —
    a bias in the strategy's favour, kept because Lo's result is about the scaling of
    the Sharpe and inventing an effective-T on top would be a second, less standard
    adjustment stacked on the first.
    """
    cfg = config.ML_CONFIG.dsr
    autocorr_adjust = cfg.autocorr_adjust if autocorr_adjust is None else autocorr_adjust
    autocorr_horizon = (
        cfg.autocorr_horizon if autocorr_horizon is None else autocorr_horizon
    )

    cols = list(daily_matrix.columns)
    stats: dict[str, dict] = {}
    for c in cols:
        r = daily_matrix[c].dropna()
        sd = r.std(ddof=1)
        sr_naive = float(r.mean() / sd) if sd > 0 else 0.0
        factor = (
            lo_scaling_factor(r, horizon=autocorr_horizon) if autocorr_adjust else 1.0
        )
        stats[c] = dict(
            sr=sr_naive * factor,
            sr_naive=sr_naive,
            ac_factor=factor,
            ac1=lag1_autocorr(r),
            T=int(r.shape[0]),
            skew=float(skew(r)) if r.shape[0] > 2 else 0.0,
            kurt=float(kurtosis(r, fisher=False)) if r.shape[0] > 3 else 3.0,
        )

    sr_values = np.array([stats[c]["sr"] for c in cols], dtype=float)
    sharpe_std = float(np.std(sr_values, ddof=1)) if len(sr_values) > 1 else 0.0
    n = n_trials if n_trials is not None else len(cols)
    sr_star = expected_max_sharpe(sharpe_std, n)

    rows = []
    for c in cols:
        s = stats[c]
        psr0 = probabilistic_sharpe_ratio(s["sr"], s["T"], s["skew"], s["kurt"], benchmark_sharpe)
        dsr = probabilistic_sharpe_ratio(s["sr"], s["T"], s["skew"], s["kurt"], sr_star)
        rows.append(dict(
            strategy_id=c,
            SR_ann=s["sr"] * _ANN,
            SR_ann_naive=s["sr_naive"] * _ANN,
            AC1=s["ac1"],
            AC_factor=s["ac_factor"],
            SR_daily=s["sr"],
            skew=s["skew"],
            kurt=s["kurt"],
            T=s["T"],
            PSR0=psr0,
            SR_star_ann=sr_star * _ANN,
            DSR=dsr,
            minTRL=_min_track_record_length(s["sr"], s["skew"], s["kurt"], sr_star, threshold),
            passed=(dsr > threshold),
        ))

    out = pd.DataFrame(rows).set_index("strategy_id")
    out.attrs["n_trials"] = n
    out.attrs["sharpe_std_daily"] = sharpe_std
    out.attrs["threshold"] = threshold
    out.attrs["autocorr_adjust"] = autocorr_adjust
    return out.sort_values("DSR", ascending=False)


# --------------------------------------------------------------------------- #
# Ledger-level entry point
# --------------------------------------------------------------------------- #
def run_dsr_gate(
    ledger_path: str,
    n_trials: int | None = None,
    threshold: float | None = None,
    persist: bool = True,
) -> pd.DataFrame:
    """
    Read a ledger, resample every column to daily, score DSR, print a ranked
    verdict table, and (optionally) persist to ``<ledger_stem>_dsr_gate.csv``.
    """
    cfg = config.ML_CONFIG.dsr
    threshold = cfg.dsr_threshold if threshold is None else threshold
    if n_trials is None:
        n_trials = cfg.trials_override  # may still be None -> column count

    if not os.path.exists(ledger_path):
        raise FileNotFoundError(f"ledger not found: {ledger_path!r}")
    ledger = pd.read_parquet(ledger_path)
    daily = pd.DataFrame({c: _to_daily(ledger[c]) for c in ledger.columns})

    verdict = deflated_sharpe_ratio(
        daily, n_trials=n_trials, threshold=threshold,
        benchmark_sharpe=cfg.benchmark_sharpe,
    )

    n = verdict.attrs["n_trials"]
    sstd = verdict.attrs["sharpe_std_daily"]
    adj = verdict.attrs["autocorr_adjust"]
    n_pass = int(verdict["passed"].sum())
    print("=" * 104)
    print(f"  DSR CREDIBILITY GATE — {os.path.basename(ledger_path)} | "
          f"N_trials={n} | threshold={threshold:.2f} | passing {n_pass}/{len(verdict)}")
    print("=" * 104)
    print(f"  {'strategy_id':<38} {'SRnaive':>8} {'AC1':>6} {'SR_ann':>7} {'DSR':>6} "
          f"{'PSR0':>6} {'skew':>6} {'kurt':>6} {'T':>6} {'minTRL':>7}  verdict")
    print(f"  {'-'*38} {'-'*8} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6} "
          f"{'-'*6} {'-'*7}  {'-'*7}")
    for sid, row in verdict.iterrows():
        trl = "n/a" if not np.isfinite(row["minTRL"]) else f"{row['minTRL']:.0f}"
        verdict_str = "PASS" if row["passed"] else "fail"
        print(f"  {sid[:38]:<38} {row['SR_ann_naive']:>8.2f} {row['AC1']:>6.3f} "
              f"{row['SR_ann']:>7.2f} {row['DSR']:>6.3f} "
              f"{row['PSR0']:>6.3f} {row['skew']:>6.2f} {row['kurt']:>6.2f} "
              f"{int(row['T']):>6} {trl:>7}  {verdict_str}")
    print(f"\n  deflation benchmark SR* (annualized) = "
          f"{verdict['SR_star_ann'].iloc[0]:.2f}  |  cross-trial Sharpe std (daily) = {sstd:.4f}")
    if adj:
        print(f"  SR_ann is Lo (2002) autocorrelation-corrected (VR horizon="
              f"{config.ML_CONFIG.dsr.autocorr_horizon}); SRnaive is the uncorrected "
              f"sqrt(252) scaling.")
    if n <= len(verdict):
        print("  NOTE: N is not larger than the number of columns scored. The deflation")
        print("        should use the number of cells SEARCHED across the program, not")
        print("        the number reported — see config.TRIAL_LEDGER.")
    print("=" * 104)

    if persist:
        out_csv = os.path.splitext(ledger_path)[0] + "_dsr_gate.csv"
        verdict.to_csv(out_csv)
        print(f"  wrote {out_csv}")
    return verdict


# --------------------------------------------------------------------------- #
# Dry-run verification
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 96)
    print("tier4_dsr_gate dry-run — PSR/DSR math + noise-vs-edge separation")
    print("=" * 96)

    rng = np.random.default_rng(7)
    T = 1500
    dates = pd.bdate_range("2018-01-01", periods=T, name="date")

    # One genuine, strong edge + several pure-noise strategies. The edge is set
    # well above the noise dispersion so it clears the multiple-testing deflation
    # benchmark (a marginal edge SHOULD fail DSR — that is the gate working).
    edge = pd.Series(rng.normal(0.0030, 0.01, T), index=dates)      # SR/day ~0.30
    noise = {f"noise_{i}": pd.Series(rng.normal(0.0, 0.01, T), index=dates)
             for i in range(6)}
    daily = pd.DataFrame({"edge": edge, **noise})

    verdict = deflated_sharpe_ratio(daily, threshold=0.95)
    print(verdict[["SR_ann", "DSR", "PSR0", "T", "passed"]].round(3).to_string())
    print()

    edge_pass = bool(verdict.loc["edge", "passed"])
    noise_fail = bool(not verdict.loc[[c for c in daily.columns if c.startswith("noise")], "passed"].any())

    # N=1 => no deflation => DSR must equal PSR0.
    solo = deflated_sharpe_ratio(daily[["edge"]], n_trials=1)
    n1_ok = bool(np.isclose(solo.loc["edge", "DSR"], solo.loc["edge", "PSR0"]))

    # PSR of a clean series matches the direct Phi formula (skew~0, kurt~3).
    s = daily["edge"]
    sr = s.mean() / s.std(ddof=1)
    direct = float(norm.cdf(sr * np.sqrt(len(s) - 1) / np.sqrt(1 + 0.5 * sr ** 2)))
    formula = probabilistic_sharpe_ratio(sr, len(s), 0.0, 3.0, 0.0)
    phi_ok = bool(np.isclose(direct, formula, atol=1e-6))

    # --- D2: the Lo autocorrelation correction ----------------------------- #
    # An AR(1) series with the SAME mean and per-period stdev as an i.i.d. one has an
    # identical naive Sharpe but a genuinely worse risk-adjusted return, because its
    # multi-period variance compounds faster. The factor must catch that.
    def _ar1(rho: float, n: int, seed: int) -> pd.Series:
        g = np.random.default_rng(seed)
        e = g.normal(0.0, 0.01, n)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = rho * x[i - 1] + e[i]
        return pd.Series(0.003 + x, index=pd.bdate_range("2018-01-01", periods=n))

    # Averaged over seeds: a single draw of this statistic has sd ~0.05, so a
    # one-seed test would be checking the seed, not the estimator.
    f_iid = float(np.mean([lo_scaling_factor(_ar1(0.0, 2337, s)) for s in range(40)]))
    f_pos = float(np.mean([lo_scaling_factor(_ar1(0.15, 2337, s)) for s in range(40)]))
    f_neg = float(np.mean([lo_scaling_factor(_ar1(-0.15, 2337, s)) for s in range(40)]))
    # AR(1)'s asymptotic factor is sqrt((1-rho)/(1+rho)).
    th_pos = float(np.sqrt(0.85 / 1.15))
    print(f"  Lo factor (mean of 40 seeds): iid={f_iid:.3f}  AR(1)+0.15={f_pos:.3f} "
          f"(theory {th_pos:.3f})  AR(1)-0.15={f_neg:.3f}")

    # With the correction ON, a positively autocorrelated series must lose Sharpe;
    # with it OFF the two must be scored identically (the flag really is the switch).
    # rho=0.30 for the single-draw checks below: at that level the factor (~0.75,
    # sd ~0.04) is many standard errors from 1.0, so the assertion tests the
    # correction rather than the luck of one seed.
    ac_frame = pd.DataFrame({"iid": _ar1(0.0, 2337, 1), "ar_pos": _ar1(0.30, 2337, 2)})
    on = deflated_sharpe_ratio(ac_frame, n_trials=2, autocorr_adjust=True)
    off = deflated_sharpe_ratio(ac_frame, n_trials=2, autocorr_adjust=False)

    checks = {
        "edge strategy PASSES": edge_pass,
        "all noise strategies FAIL": noise_fail,
        "N=1 => DSR == PSR0 (no deflation)": n1_ok,
        "PSR matches direct Phi formula": phi_ok,
        "Lo factor unbiased on i.i.d. returns": bool(0.98 < f_iid < 1.02),
        "Lo factor recovers the AR(1) theory value": bool(abs(f_pos - th_pos) < 0.02),
        "Lo factor penalizes positive autocorrelation": bool(f_pos < 0.90),
        "Lo factor rewards negative autocorrelation": bool(f_neg > 1.10),
        "correction ON lowers an autocorrelated Sharpe": bool(
            on.loc["ar_pos", "SR_ann"] < on.loc["ar_pos", "SR_ann_naive"]
        ),
        "correction OFF is a true no-op": bool(
            np.allclose(off["SR_ann"], off["SR_ann_naive"])
        ),
        "correction applies to every column, not just one": bool(
            np.isfinite(on["AC_factor"]).all()
        ),
        "honest N is larger than a phase's column count": bool(
            config.TRIALS_SEARCHED == 42 and config.ML_CONFIG.dsr.trials_override == 42
        ),
    }
    print("-" * 96)
    for name, ok in checks.items():
        print(f"  {name:<40}: {'PASS' if ok else 'FAIL'}")
    print("-" * 96)
    assert all(checks.values()), "tier4_dsr_gate dry-run FAILED"

    # Run on the real production ledger if present.
    prod = "data/trial_database/production_dsr_matrix.parquet"
    if os.path.exists(prod):
        print("\nReal production ledger:\n")
        run_dsr_gate(prod, persist=False)
