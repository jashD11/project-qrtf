"""
Phase 4c A1 — the bid-ask spread panel. The measurement that decides the phase.

Phase 4b's two DSR passes have **under 7 bps** of headroom on a cost the backtest sets
to zero: `slippage_bps = 0.0`, and the size-dependent impact term is never called from
the execution path. With 0.45-0.82 of the book turning over per day, spread is not a
rounding error — it is the term that decides whether the result is a finding or an
artefact.

The bhavcopy carries no quotes, so the spread cannot be read; it has to be **estimated
from daily OHLC**. Two published estimators are built rather than one, because they
bracket the answer from opposite sides and the honest output is a band, not a point:

  **Corwin-Schultz (2012)** — two-day high-low ratio. The insight is that the high-low
  range over *two* days contains two days of variance but only *one* spread, while the
  sum of the two single-day ranges contains two of each; differencing isolates the
  spread. Biased **down** on this data (it reads ~0 for the most liquid quintile).

  **Abdi-Ranaldo (2017)** — close-to-mid-range covariance. ``S² = 4·E[(c_t-η_t)(c_t-η_{t+1})]``
  where ``η = (log high + log low)/2`` proxies the unobserved efficient price. Biased
  **up**, because a true spread near zero produces negative raw estimates half the time
  and clipping them at zero cannot average back down.

Two design points that are not incidental:

  **Clip once, at the window level.** Both estimators produce negative estimates on
  individual observations — that is sampling noise around a small true spread, not
  evidence of a negative spread. Clipping each observation before averaging turns
  symmetric noise into a systematic upward bias. So the window statistic is averaged
  first and clipped once, at the end.

  **Causal, even though cost is not a signal.** The window ends at ``t-1``: the spread
  charged on a trade executed at ``t`` never sees ``t``'s own bar. A cost model that
  peeks is still a backtest that could not have been run.

Output: ``data/bhavcopy/spread_daily_{cs,ar}.parquet``, wide ``(date × ticker)``
**half-spread in bps** — half, because a single execution crosses half the quoted
spread, and the engine charges per side.

Usage
    python src/phase4_data/spread.py            # dry-run + monotone-in-liquidity gate
    python src/phase4_data/spread.py --build    # build both panels from the panel
"""

import argparse
import os
import sys
from typing import Final

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_PANEL: Final[str] = "data/bhavcopy/panel_daily.parquet"
DEFAULT_MASK: Final[str] = "data/nse500_universe_mask.parquet"
DEFAULT_CS: Final[str] = "data/bhavcopy/spread_daily_cs.parquet"
DEFAULT_AR: Final[str] = "data/bhavcopy/spread_daily_ar.parquet"

WINDOW: Final[int] = 63            # trailing estimation window (one quarter)
MIN_PERIODS: Final[int] = 30       # usable estimate floor within that window

ESTIMATORS: Final[tuple[str, ...]] = ("cs", "ar")

# Corwin-Schultz constant: 3 - 2*sqrt(2).
_CS_K: Final[float] = 3.0 - 2.0 * np.sqrt(2.0)


def _wide(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    """Long panel -> wide (date × ticker) frame for one OHLC column."""
    return panel.pivot_table(
        index="date", columns="ticker", values=col, aggfunc="last"
    ).sort_index()


# --------------------------------------------------------------------------- #
# Corwin-Schultz (2012)
# --------------------------------------------------------------------------- #
def corwin_schultz(
    high: pd.DataFrame, low: pd.DataFrame,
    window: int = WINDOW, min_periods: int = MIN_PERIODS,
) -> pd.DataFrame:
    """
    Trailing Corwin-Schultz half-spread in bps.

    Per adjacent day pair (indexed at the *later* day):

        beta  = ln(H_t/L_t)^2 + ln(H_{t-1}/L_{t-1})^2      two days' ranges
        gamma = ln( max(H) / min(L) )^2                    the two-day range

    Both are averaged over the trailing window **before** the nonlinear step, so the
    single alpha is formed from window means rather than a mean of noisy per-pair
    alphas (this is the estimator's own recommended aggregation):

        alpha = (sqrt(2*beta) - sqrt(beta)) / k - sqrt(gamma / k),  k = 3 - 2*sqrt(2)
        S     = 2*(e^alpha - 1) / (1 + e^alpha)            proportional FULL spread

    A negative alpha means the estimator resolved no spread at this resolution; it is
    clipped to zero once, here, at the window level.
    """
    hl = np.log(high / low) ** 2
    beta = hl + hl.shift(1)

    two_day_high = pd.DataFrame(
        np.maximum(high.to_numpy(), high.shift(1).to_numpy()),
        index=high.index, columns=high.columns,
    )
    two_day_low = pd.DataFrame(
        np.minimum(low.to_numpy(), low.shift(1).to_numpy()),
        index=low.index, columns=low.columns,
    )
    gamma = np.log(two_day_high / two_day_low) ** 2

    roll = dict(window=window, min_periods=min_periods)
    beta_bar = beta.rolling(**roll).mean().shift(1)     # shift(1) => strictly causal
    gamma_bar = gamma.rolling(**roll).mean().shift(1)

    alpha = (np.sqrt(2.0 * beta_bar) - np.sqrt(beta_bar)) / _CS_K - np.sqrt(gamma_bar / _CS_K)
    alpha = alpha.clip(lower=0.0)                        # the one clip
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return spread / 2.0 * 1e4                            # full -> half, -> bps


# --------------------------------------------------------------------------- #
# Abdi-Ranaldo (2017)
# --------------------------------------------------------------------------- #
def abdi_ranaldo(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
    window: int = WINDOW, min_periods: int = MIN_PERIODS,
) -> pd.DataFrame:
    """
    Trailing Abdi-Ranaldo half-spread in bps.

    ``eta_t = (ln H_t + ln L_t)/2`` proxies the efficient price at t. If the close is
    the efficient price plus half a spread times a trade sign, then

        S^2 = 4 * E[ (c_t - eta_t) * (c_t - eta_{t+1}) ]

    because the *same* signed half-spread appears in both factors while the efficient
    price innovations between them are independent. The per-observation product is
    averaged over the window and the single negative-variance clip is applied once, at
    the end — clipping each product first would convert symmetric noise around a small
    true spread into a systematic overestimate.
    """
    c = np.log(close)
    eta = (np.log(high) + np.log(low)) / 2.0
    # Indexed at the later day of the pair, matching corwin_schultz's convention.
    prod = (c.shift(1) - eta.shift(1)) * (c.shift(1) - eta)

    mean_prod = prod.rolling(window=window, min_periods=min_periods).mean().shift(1)
    s_squared = (4.0 * mean_prod).clip(lower=0.0)        # the one clip
    spread = np.sqrt(s_squared)
    return spread / 2.0 * 1e4                            # full -> half, -> bps


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_panels(
    panel: pd.DataFrame, window: int = WINDOW, min_periods: int = MIN_PERIODS,
) -> dict[str, pd.DataFrame]:
    """Both half-spread panels, wide (date × ticker) in bps."""
    high, low, close = (_wide(panel, c) for c in ("high", "low", "close"))
    # A zero or missing price makes both logs undefined; drop rather than propagate.
    high, low, close = (f.where(f.gt(0)) for f in (high, low, close))

    out = {
        "cs": corwin_schultz(high, low, window, min_periods),
        "ar": abdi_ranaldo(high, low, close, window, min_periods),
    }
    for name, frame in out.items():
        vals = frame.to_numpy()
        ok = np.isfinite(vals)
        print(
            f"[spread] {name}: {frame.shape[0]:,} dates × {frame.shape[1]:,} tickers | "
            f"coverage {ok.mean():.1%} | median {np.nanmedian(vals[ok]):.1f} bps | "
            f"p90 {np.nanquantile(vals[ok], 0.90):.1f} bps"
        )
    return out


# --------------------------------------------------------------------------- #
# Validation gate — the estimate must be monotone in liquidity
# --------------------------------------------------------------------------- #
def liquidity_quintile_table(
    spreads: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    mask_path: str = DEFAULT_MASK,
) -> pd.DataFrame:
    """
    Half-spread by in-universe turnover quintile — the gate from the plan.

    A spread estimator that is *not* monotone in liquidity is measuring noise, and a
    cost model built on it would be worse than the flat constant it replaces. Quintile
    1 is the most liquid.

    Both the **mean** and the median are reported, and the gate is on the *mean*. That
    is not a cosmetic choice. Between 39% and 76% of observations clip to exactly zero
    (the estimators cannot resolve a spread below the price's own noise floor), so the
    median of the more liquid quintiles is identically 0.00 and a monotonicity test on
    it would pass degenerately on a flat row of zeros. The mean is also the correct
    statistic economically: the cost is charged on every trade, and a clipped zero is
    an unresolved spread, not a free one.
    """
    mask = pd.read_parquet(mask_path)
    mask.index = pd.DatetimeIndex(mask.index).normalize()

    turnover = _wide(panel, "turnover")
    common_d = turnover.index.intersection(mask.index)
    common_t = turnover.columns.intersection(mask.columns)
    turnover = turnover.loc[common_d, common_t].where(mask.loc[common_d, common_t])

    # Quintile by cross-sectional turnover rank on each date (1 = most liquid).
    q = turnover.rank(axis=1, ascending=False, pct=True)
    bucket = np.ceil(q * 5).clip(1, 5)

    rows = []
    for name, frame in spreads.items():
        f = frame.reindex(index=common_d, columns=common_t)
        for k in range(1, 6):
            sel = f.where(bucket == k).to_numpy()
            sel = sel[np.isfinite(sel)]
            rows.append(dict(
                estimator=name, quintile=int(k),
                mean_bps=float(sel.mean()) if sel.size else np.nan,
                median_bps=float(np.median(sel)) if sel.size else np.nan,
                zero_pct=float((sel == 0).mean() * 100.0) if sel.size else np.nan,
                n=int(sel.size),
            ))
    out = pd.DataFrame(rows).set_index(["quintile", "estimator"]).unstack("estimator")
    return out.sort_index()


def check_monotone(table: pd.DataFrame) -> dict[str, bool]:
    """
    Each estimator's **mean** half-spread must rise strictly as liquidity falls.

    Strictly, not merely non-decreasing: a flat row would satisfy "non-decreasing"
    while carrying no liquidity information at all, which is exactly the failure mode
    a gate on the (heavily zero-clipped) median would have waved through.
    """
    means = table["mean_bps"]
    return {
        est: bool((np.diff(means[est].to_numpy()) > 0).all())
        for est in means.columns
    }


def print_gate(table: pd.DataFrame) -> dict[str, bool]:
    """Print the quintile table next to the plan's reference band and gate on it."""
    print(f"\n{'=' * 78}")
    print("SPREAD GATE — half-spread (bps) by in-universe turnover quintile")
    print("=" * 78)
    print(f"  {'quintile':<18}{'CS mean':>9}{'CS med':>8}{'CS 0%':>7}"
          f"{'AR mean':>10}{'AR med':>8}{'AR 0%':>7}")
    labels = ["1 most liquid", "2", "3", "4", "5 least liquid"]
    for i, lab in enumerate(labels):
        r = table.iloc[i]
        print(f"  {lab:<18}{r[('mean_bps', 'cs')]:>9.2f}{r[('median_bps', 'cs')]:>8.2f}"
              f"{r[('zero_pct', 'cs')]:>6.0f}%"
              f"{r[('mean_bps', 'ar')]:>10.2f}{r[('median_bps', 'ar')]:>8.2f}"
              f"{r[('zero_pct', 'ar')]:>6.0f}%")
    print("-" * 78)
    # docs/phase4c_plan.md §1: the planning-stage band, measured independently.
    for est, ref in (("cs", 6.1), ("ar", 31.7)):
        n = table[("n", est)].to_numpy()
        pooled = float((table[("mean_bps", est)].to_numpy() * n).sum() / n.sum())
        print(f"  {est}: pooled in-universe mean = {pooled:5.2f} bps "
              f"(plan's reference band: {ref} bps)")
    mono = check_monotone(table)
    for est, ok in mono.items():
        print(f"  {est}: strictly monotone in liquidity -> {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    return mono


def build(
    panel_path: str = DEFAULT_PANEL,
    mask_path: str = DEFAULT_MASK,
    out_cs: str = DEFAULT_CS,
    out_ar: str = DEFAULT_AR,
) -> dict[str, pd.DataFrame]:
    """Build both spread panels, run the monotonicity gate, and persist."""
    if not os.path.exists(panel_path):
        raise FileNotFoundError(f"{panel_path} not found — run the Phase 4 build first.")
    panel = pd.read_parquet(
        panel_path, columns=["date", "ticker", "high", "low", "close", "turnover"]
    )
    spreads = build_panels(panel)

    table = liquidity_quintile_table(spreads, panel, mask_path)
    mono = print_gate(table)
    if not all(mono.values()):
        raise AssertionError(
            f"spread estimate is not monotone in liquidity {mono} — it is measuring "
            "noise, and a cost model built on it would be worse than a flat constant."
        )

    for path, key in ((out_cs, "cs"), (out_ar, "ar")):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        spreads[key].to_parquet(path)
        print(f"[spread] wrote {path}")
    return spreads


def load(estimator: str = "cs", path: str | None = None) -> pd.DataFrame:
    """Read one half-spread panel (bps), with a clear error naming the build step."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator={estimator!r} not in {list(ESTIMATORS)}")
    path = path or (DEFAULT_CS if estimator == "cs" else DEFAULT_AR)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run `python src/phase4_data/spread.py --build` first."
        )
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    return frame


# ── Dry run ─────────────────────────────────────────────────────────────────────

def _synth(
    n_days: int, spread_bps: np.ndarray, daily_vol: float = 0.015, seed: int = 7
) -> pd.DataFrame:
    """
    A panel with a **known** spread planted in it.

    An efficient price random-walks; the observed close is that price plus a signed
    half-spread (a bid or an ask hit, at random). High and low are the efficient
    price's intraday range, widened by the same half-spread on each side — which is
    what a real quoted market does, and what both estimators are built to invert.

    ``daily_vol`` is a parameter because it sets the estimators' resolution: the
    spread is recovered from a *variance decomposition*, so a spread far below the
    price's own noise cannot be separated from it at any sample size the panel has.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    rows = []
    for i, bps in enumerate(spread_bps):
        half = bps / 1e4
        eff = 100.0 * np.exp(np.cumsum(rng.normal(0, daily_vol, n_days)))
        rng_span = np.abs(rng.normal(0, daily_vol * 0.8, n_days))
        hi = eff * (1.0 + rng_span) * (1.0 + half)
        lo = eff * (1.0 - rng_span) * (1.0 - half)
        close = eff * (1.0 + half * rng.choice([-1.0, 1.0], n_days))
        rows.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i:02d}",
            "high": hi, "low": lo, "close": close,
            # Turnover ordered inversely to the planted spread, so the liquidity
            # quintile gate has a ground truth to be monotone against.
            "turnover": 10.0 ** (10 - i),
        }))
    return pd.concat(rows, ignore_index=True)


def _dry_run() -> None:
    """
    Self-test against a planted spread, checking the properties the cost model needs.

    **Recovering the level is deliberately not the bar.** Both estimators invert a
    variance decomposition, so their resolution floor is set by the price's own
    volatility: against 1.5% daily vol neither can separate a 5 bps spread from noise,
    and pretending otherwise in a test would only encode a false precision the plan
    already rejects (which is why it carries them as a **band**, 6.1 to 31.7 bps, and
    not as a number).

    What is tested is what the cost model actually relies on: the estimate is
    **ordered** by the true spread, it is causal, it is non-negative, and the wide
    names are resolved as materially costly rather than rounded to zero.
    """
    from scipy.stats import spearmanr

    planted = np.array([2.0, 10.0, 25.0, 50.0, 100.0, 200.0])
    panel = _synth(n_days=900, spread_bps=planted, daily_vol=0.012)
    spreads = build_panels(panel, window=WINDOW, min_periods=MIN_PERIODS)

    checks: dict[str, bool] = {}
    print(f"\n  planted half-spread (bps): {list(planted)}")
    for name, frame in spreads.items():
        est = frame.median(axis=0).to_numpy()
        rho = float(spearmanr(planted, est).statistic)
        print(f"  {name} recovered (median):   {np.round(est, 1).tolist()}  rho={rho:+.3f}")
        # Ordering is the property the cost model needs: charge the illiquid name more
        # than the liquid one. Rank correlation, not level accuracy.
        checks[f"{name}: rank-ordered by true spread"] = bool(rho > 0.85)
        checks[f"{name}: non-negative everywhere"] = bool(
            np.nanmin(frame.to_numpy()) >= 0.0
        )
        # The widest names must be resolved as materially costly, not rounded to zero.
        checks[f"{name}: resolves the 200 bps name"] = bool(est[-1] > 50.0)

    # Causality: an estimate on date t must not move when only bars >= t change.
    tampered = panel.copy()
    cut = tampered["date"] >= panel["date"].unique()[400]
    tampered.loc[cut, "high"] *= 1.5
    tampered.loc[cut, "low"] /= 1.5
    after = build_panels(tampered, window=WINDOW, min_periods=MIN_PERIODS)
    for name in ESTIMATORS:
        a = spreads[name].iloc[:400].to_numpy()
        b = after[name].iloc[:400].to_numpy()
        checks[f"{name}: causal (future cannot leak back)"] = bool(
            np.allclose(a, b, equal_nan=True)
        )

    # The monotonicity gate itself must fire on a non-monotone table, and — the case
    # that actually bit — on a *flat* one, which a non-strict test would have passed.
    def _tbl(cs: list[float], ar: list[float]) -> pd.DataFrame:
        cols = pd.MultiIndex.from_product([["mean_bps"], ["cs", "ar"]])
        return pd.DataFrame(list(zip(cs, ar)), index=[1, 2, 3, 4, 5], columns=cols)

    good = _tbl([2.4, 4.5, 6.2, 7.5, 9.2], [18.8, 22.5, 26.6, 30.1, 32.4])
    bad = _tbl([2.4, 4.5, 1.0, 7.5, 9.2], [18.8, 22.5, 26.6, 30.1, 32.4])
    flat = _tbl([0.0] * 5, [18.8, 22.5, 26.6, 30.1, 32.4])
    checks["gate passes a monotone table"] = all(check_monotone(good).values())
    checks["gate rejects a non-monotone table"] = not all(check_monotone(bad).values())
    checks["gate rejects a flat (all-zero) table"] = not all(check_monotone(flat).values())

    print()
    print("=" * 72)
    for name, ok in checks.items():
        print(f"  {name:<44}: {'PASS' if ok else 'FAIL'}")
    print("=" * 72)
    assert all(checks.values()), "spread dry-run FAILED"


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate the NSE half-spread panel.")
    ap.add_argument("--build", action="store_true", help="build from the panel (else dry-run)")
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--mask", default=DEFAULT_MASK)
    ap.add_argument("--out-cs", default=DEFAULT_CS)
    ap.add_argument("--out-ar", default=DEFAULT_AR)
    args = ap.parse_args()

    if not args.build:
        _dry_run()
        return
    build(args.panel, args.mask, args.out_cs, args.out_ar)


if __name__ == "__main__":
    main()
