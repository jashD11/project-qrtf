"""
Phase 4c §6 — the read-only diagnostics reported alongside the verdict.

None of these selects anything. They exist because a number can pass a gate for the
wrong reason, and each one tests a specific alternative explanation for Phase 4b's
result:

  **B4 — short-book coverage.** Of the ~50 names the model wanted to short each day,
  how many were actually borrowable? If most never were, the SLB cells are a different
  strategy rather than a corrected version of the old one, and that is the finding.

  **IC by liquidity quintile.** Phase 4b's ensemble rank-IC jumped 2.5x (0.02 -> 0.0511)
  when the universe went from 68 names to 500, and the jump was never explained. If the
  skill concentrates in the illiquid tail, then the IC jump, the 2016-2020
  concentration, and the spread problem are all *one* finding — the edge is a spread
  artefact — and no cost model can rescue it.

  **IC by year.** Is the 2021->2025 decay in the signal or in the costs? A decaying IC
  says the signal; a flat IC with decaying net returns says the costs.

  **Turnover attribution.** Turnover is what makes the result cost-fragile. Splitting it
  into signal rotation, decile-boundary churn, and panic-gate liquidation says which of
  the three is worth attacking — and whether the no-trade buffer is already exhausted.

Usage
    python src/phase4_data/phase4c_diagnostics.py            # dry-run self-test
    python src/phase4_data/phase4c_diagnostics.py --run      # all of the above
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

import config  # noqa: E402

DEFAULT_PANEL: Final[str] = "data/bhavcopy/panel_daily.parquet"
DEFAULT_MASK: Final[str] = "data/nse500_universe_mask.parquet"
N_QUINTILES: Final[int] = 5


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def liquidity_quintiles(
    turnover: pd.DataFrame, universe: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Per-(date, name) trailing-liquidity quintile, 1 = most liquid.

    Ranked on the same trailing rupee turnover the universe itself ranks on, so the
    buckets mean the same thing here as they do in ``universe.py``. Market cap is not
    available in the bhavcopy and is not needed: the question is whether the edge lives
    where the spread is wide, and turnover is the direct measure of that.
    """
    t = turnover if universe is None else turnover.where(universe)
    pct = t.rank(axis=1, ascending=False, pct=True)
    return np.ceil(pct * N_QUINTILES).clip(1, N_QUINTILES)


def _spearman_by_row(a: pd.DataFrame, b: pd.DataFrame, min_names: int = 5) -> pd.Series:
    """
    Cross-sectional Spearman correlation between two aligned wide frames, per row.

    Computed as a Pearson correlation of per-row ranks, vectorized — the equivalent
    ``scipy`` call per date would be thousands of Python-level round trips.
    """
    valid = a.notna() & b.notna()
    ra = a.where(valid).rank(axis=1)
    rb = b.where(valid).rank(axis=1)
    n = valid.sum(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    out = num / den.replace(0.0, np.nan)
    return out.where(n >= min_names)


def _ic_summary(daily_ic: pd.Series) -> dict:
    """mean IC, IC information ratio, t-stat and hit rate for a daily IC series."""
    ic = daily_ic.dropna()
    n = int(ic.shape[0])
    if n < 2:
        return dict(mean_IC=np.nan, IC_IR=np.nan, IC_t=np.nan, hit=np.nan, n_days=n)
    sd = float(ic.std(ddof=1))
    ir = float(ic.mean()) / sd if sd > 0 else np.nan
    return dict(
        mean_IC=float(ic.mean()), IC_IR=ir,
        IC_t=ir * np.sqrt(n) if np.isfinite(ir) else np.nan,
        hit=float((ic > 0).mean()), n_days=n,
    )


# --------------------------------------------------------------------------- #
# B4 — short-book coverage
# --------------------------------------------------------------------------- #
def short_book_coverage(
    alpha_scores: pd.DataFrame,
    shortable: pd.DataFrame,
    decile_pct: float = 0.10,
) -> pd.DataFrame:
    """
    Of the ``k`` names the model wanted to short each day, how many were borrowable?

    Measured on the **unrestricted** bottom decile — the book Phase 4b actually
    reported — so it answers "was that book buildable?", not "what does the restricted
    book look like?". Those are different questions and only the first one bears on
    whether Phase 4b's two passing cells were ever real.
    """
    elig = shortable.reindex(index=alpha_scores.index, columns=alpha_scores.columns)
    elig = elig.fillna(False).astype(bool)

    n_valid = alpha_scores.notna().sum(axis=1)
    k = np.floor(n_valid * decile_pct).clip(lower=1).astype(int)
    rank_asc = alpha_scores.rank(axis=1, ascending=True, method="first")
    wanted = rank_asc.le(k.to_numpy()[:, None]) & alpha_scores.notna()

    n_wanted = wanted.sum(axis=1)
    n_ok = (wanted & elig).sum(axis=1)
    out = pd.DataFrame({"wanted": n_wanted, "borrowable": n_ok})
    out = out[out["wanted"] > 0]
    out["pct"] = out["borrowable"] / out["wanted"]
    return out


def print_short_coverage(cov: pd.DataFrame) -> None:
    by_year = cov.groupby(cov.index.year).mean()
    print(f"\n{'=' * 72}")
    print("B4 — SHORT-BOOK COVERAGE: of the names the model wanted to short,")
    print("     how many were actually borrowable (F&O proxy, an UPPER bound)?")
    print("=" * 72)
    print(f"  {'year':<8}{'wanted':>9}{'borrowable':>13}{'coverage':>11}")
    for year, r in by_year.iterrows():
        print(f"  {year:<8}{r['wanted']:>9.1f}{r['borrowable']:>13.1f}{r['pct']:>11.1%}")
    print("-" * 72)
    print(f"  {'ALL':<8}{cov['wanted'].mean():>9.1f}{cov['borrowable'].mean():>13.1f}"
          f"{cov['pct'].mean():>11.1%}")
    print(f"  days with an EMPTY borrowable short book: "
          f"{(cov['borrowable'] == 0).mean():.1%}")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# IC decompositions
# --------------------------------------------------------------------------- #
def ic_by_quintile(
    alpha_scores: pd.DataFrame, realized: pd.DataFrame, quintiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensemble rank-IC computed **within** each liquidity quintile.

    Within-bucket rather than pooled: a pooled IC would partly measure the model's
    ability to rank liquid names against illiquid ones, which is a level effect, not
    the cross-sectional stock-picking skill the strategy monetizes.
    """
    q = quintiles.reindex(index=alpha_scores.index, columns=alpha_scores.columns)
    rows = []
    for k in range(1, N_QUINTILES + 1):
        sel = q == k
        ic = _spearman_by_row(alpha_scores.where(sel), realized.where(sel))
        rows.append(dict(quintile=k, **_ic_summary(ic)))
    return pd.DataFrame(rows).set_index("quintile")


def ic_by_year(alpha_scores: pd.DataFrame, realized: pd.DataFrame) -> pd.DataFrame:
    """Ensemble rank-IC per calendar year — signal decay vs cost decay."""
    ic = _spearman_by_row(alpha_scores, realized)
    rows = [
        dict(year=int(year), **_ic_summary(grp))
        for year, grp in ic.groupby(ic.index.year)
    ]
    return pd.DataFrame(rows).set_index("year")


def print_ic_table(table: pd.DataFrame, title: str, index_label: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    print(f"  {index_label:<16}{'meanIC':>9}{'IC_IR':>8}{'IC_t':>8}{'hit%':>8}{'n_days':>9}")
    for idx, r in table.iterrows():
        print(f"  {str(idx):<16}{r['mean_IC']:>9.4f}{r['IC_IR']:>8.3f}{r['IC_t']:>8.2f}"
              f"{r['hit'] * 100:>7.1f}%{int(r['n_days']):>9}")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# Turnover attribution
# --------------------------------------------------------------------------- #
def turnover_attribution(
    long_mask: pd.DataFrame,
    short_mask: pd.DataFrame,
    raw_long: pd.DataFrame,
    raw_short: pd.DataFrame,
    panic: pd.Series,
) -> dict:
    """
    Split total book turnover into its three sources.

      **panic-gate liquidation** — turnover on bars where the de-risk gate flips. The
        book goes to cash and comes back; both legs of that are forced, not chosen.
      **boundary churn** — what the no-trade buffer already removed: the difference
        between the raw decile membership and the buffered membership. If this is
        small the buffer is exhausted and widening it further buys nothing, which is
        what the Phase 3 sensitivity scan found (Sharpe flat across mult 2.0-4.0).
      **signal rotation** — the remainder: names genuinely leaving and entering the
        decile on a changed view. This is the irreducible part.
    """
    def _turn(lm: pd.DataFrame, sm: pd.DataFrame) -> pd.Series:
        book = (lm.abs() + sm.abs())
        w = book.div(book.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        t = w.diff().abs().sum(axis=1)
        if len(w):
            t.iloc[0] = w.iloc[0].abs().sum()
        return t

    buffered = _turn(long_mask, short_mask)
    raw = _turn(raw_long, raw_short)

    p = panic.reindex(buffered.index, fill_value=False).astype(bool)
    flips = p.ne(p.shift(1, fill_value=False))
    gate = float(buffered[flips].sum())
    total = float(buffered.sum())
    churn_removed = float(raw.sum() - buffered.sum())

    return {
        "total_turnover_buffered": total,
        "total_turnover_raw_deciles": float(raw.sum()),
        "churn_removed_by_buffer": churn_removed,
        "churn_removed_pct_of_raw": churn_removed / max(float(raw.sum()), 1e-12),
        "panic_gate_turnover": gate,
        "panic_gate_pct_of_total": gate / max(total, 1e-12),
        "signal_rotation_pct_of_total": (total - gate) / max(total, 1e-12),
        "mean_turnover_per_bar": float(buffered.mean()),
    }


def print_turnover(attr: dict) -> None:
    print(f"\n{'=' * 72}\nTURNOVER ATTRIBUTION\n{'=' * 72}")
    print(f"  mean turnover / bar (buffered)      : {attr['mean_turnover_per_bar']:.3f}")
    print(f"  raw-decile turnover (no buffer)     : {attr['total_turnover_raw_deciles']:,.0f}")
    print(f"  buffered turnover                   : {attr['total_turnover_buffered']:,.0f}")
    print(f"  boundary churn removed by buffer    : {attr['churn_removed_pct_of_raw']:.1%} of raw")
    print(f"  panic-gate liquidation / re-entry   : {attr['panic_gate_pct_of_total']:.1%} of total")
    print(f"  signal rotation (the irreducible)   : {attr['signal_rotation_pct_of_total']:.1%} of total")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_panel_inputs(
    dates: pd.Index, tickers: pd.Index,
    panel_path: str = DEFAULT_PANEL, mask_path: str = DEFAULT_MASK,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(turnover, universe, realized forward return) aligned to a scored grid."""
    panel = pd.read_parquet(panel_path, columns=["date", "ticker", "close", "turnover"])
    to = panel.pivot_table(index="date", columns="ticker", values="turnover", aggfunc="last")
    px = panel.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    fwd = px.bfill().shift(-1) / px - 1.0     # same convention as tier3.forward_returns

    uni = pd.read_parquet(mask_path)
    uni.index = pd.DatetimeIndex(uni.index).normalize()

    idx, cols = dates, tickers
    return (
        to.reindex(index=idx, columns=cols),
        uni.reindex(index=idx, columns=cols).fillna(False).astype(bool),
        fwd.reindex(index=idx, columns=cols),
    )


def run_all(target_col: str = "tgt_fwd_logret_5b") -> dict:
    """Run every §6 diagnostic against the cached Phase 4c walk-forward."""
    from src.production_ml.tier1_trees import TreeAlphaEngine, DECILE_PCT
    from src.production_ml.tier2_regime import RegimeDetector, build_features
    from src.production_ml.tier3_execution import apply_rebalance_buffer
    from src.production_ml.wf_cache import cached_walk_forward
    from src.phase4_data.shortable import load as load_shortable
    from run_pipeline_ml import (
        compute_signals, make_ml_config, phase4_regime_config,
    )

    frequency = config.PHASE4C_FREQUENCY
    wf, _ = compute_signals(frequency, DECILE_PCT, target_col=target_col)
    alpha = wf.alpha_scores

    turnover, universe, realized = load_panel_inputs(alpha.index, alpha.columns)
    quints = liquidity_quintiles(turnover, universe)

    out: dict = {}
    out["ic_by_quintile"] = ic_by_quintile(alpha, realized, quints)
    print_ic_table(
        out["ic_by_quintile"],
        "IC BY LIQUIDITY QUINTILE — does the skill live in the illiquid tail?",
        "quintile",
    )
    out["ic_by_year"] = ic_by_year(alpha, realized)
    print_ic_table(
        out["ic_by_year"],
        "IC BY YEAR — is the 2021->2025 decay in the signal or in the costs?",
        "year",
    )

    if os.path.exists(config.ML_CONFIG.phase4.shortable_mask_parquet):
        cov = short_book_coverage(alpha, load_shortable(), DECILE_PCT)
        print_short_coverage(cov)
        out["short_coverage"] = cov
    else:
        print("\n[diagnostics] shortable mask absent — skipping B4 coverage. Build it "
              "with `python src/phase4_data/shortable.py --build`.")

    cfg = make_ml_config(frequency, "long_short", target_col=target_col)
    lm, sm = apply_rebalance_buffer(alpha, cfg)
    panic = RegimeDetector(
        n_states=config.HEADLINE_HMM_STATES
    ).fit_predict(build_features(phase4_regime_config())).panic
    out["turnover"] = turnover_attribution(lm, sm, wf.long_mask, wf.short_mask, panic)
    print_turnover(out["turnover"])
    return out


# ── Dry run ─────────────────────────────────────────────────────────────────────

def _dry_run() -> None:
    """
    Self-test on synthetic data where each diagnostic's answer is known by design.

    The IC decompositions are the ones worth testing: a diagnostic that would report
    "the skill is uniform across liquidity" whatever the truth cannot support or refute
    the spread-artefact hypothesis, which is the whole reason it exists.
    """
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2018-01-01", periods=400)
    tickers = [f"T{i:03d}" for i in range(50)]

    alpha = pd.DataFrame(rng.normal(size=(len(dates), len(tickers))),
                         index=dates, columns=tickers)
    # Turnover falls monotonically with ticker index -> quintile 5 = last 10 names.
    turnover = pd.DataFrame(
        np.tile(np.geomspace(1e10, 1e6, len(tickers)), (len(dates), 1)),
        index=dates, columns=tickers,
    )
    quints = liquidity_quintiles(turnover)

    # Realized returns carry the alpha ONLY in the illiquid quintile; the rest is noise.
    noise = pd.DataFrame(rng.normal(scale=1.0, size=alpha.shape),
                         index=dates, columns=tickers)
    realized = noise.copy()
    illiquid = quints == 5
    realized = realized.where(~illiquid, alpha * 3.0 + noise * 0.2)

    checks: dict[str, bool] = {}
    q_ic = ic_by_quintile(alpha, realized, quints)
    checks["IC decomposition finds the planted quintile"] = bool(
        q_ic.loc[5, "mean_IC"] > 0.5 and q_ic.loc[1, "mean_IC"] < 0.15
    )

    # Signal present only in the second half of the sample -> IC by year must show it.
    realized_y = noise.copy()
    late = pd.Series(dates.year >= 2019, index=dates)
    realized_y = realized_y.where(~late, alpha * 3.0 + noise * 0.2, axis=0)
    y_ic = ic_by_year(alpha, realized_y)
    checks["IC by year finds a regime change"] = bool(
        y_ic.loc[2018, "mean_IC"] < 0.15 < y_ic.loc[2019, "mean_IC"]
    )

    # B4: with half the universe borrowable, coverage must be materially below 1.0,
    # and a fully borrowable universe must read exactly 1.0.
    half = pd.DataFrame(
        np.tile([True] * 25 + [False] * 25, (len(dates), 1)),
        index=dates, columns=tickers,
    )
    cov_half = short_book_coverage(alpha, half, 0.10)
    cov_all = short_book_coverage(alpha, half * 0 + True, 0.10)
    checks["B4 coverage < 1 on a half-borrowable universe"] = bool(
        0.2 < cov_half["pct"].mean() < 0.8
    )
    checks["B4 coverage == 1 when everything is borrowable"] = bool(
        np.allclose(cov_all["pct"], 1.0)
    )
    checks["B4 counts the full intended book"] = bool(
        (cov_all["wanted"] == max(1, int(np.floor(len(tickers) * 0.10)))).all()
    )

    # Turnover attribution: the buffer must remove churn, and a gate that never fires
    # must attribute exactly zero turnover to it.
    from src.production_ml.tier3_execution import apply_rebalance_buffer
    from run_pipeline_ml import make_ml_config
    from src.production_ml.tier1_trees import TreeAlphaEngine

    cfg_b = make_ml_config("daily", "long_short", rebalance_buffer_mult=2.0)
    cfg_r = make_ml_config("daily", "long_short", rebalance_buffer_mult=1.0)
    lm_b, sm_b = apply_rebalance_buffer(alpha, cfg_b)
    lm_r, sm_r = apply_rebalance_buffer(alpha, cfg_r)
    no_panic = pd.Series(False, index=dates)
    attr = turnover_attribution(lm_b, sm_b, lm_r, sm_r, no_panic)
    checks["turnover: buffer removes boundary churn"] = bool(
        attr["churn_removed_by_buffer"] > 0
    )
    # A gate that never trips still "flips" on bar 0 (False -> False is not a flip),
    # so the only turnover it can claim is zero.
    checks["turnover: a gate that never fires claims 0%"] = bool(
        attr["panic_gate_pct_of_total"] < 1e-9
    )
    always_panic = pd.Series(True, index=dates)
    attr2 = turnover_attribution(lm_b, sm_b, lm_r, sm_r, always_panic)
    checks["turnover: a gate flip is attributed to the gate"] = bool(
        attr2["panic_gate_pct_of_total"] > 0
    )

    print()
    print("=" * 72)
    for name, ok in checks.items():
        print(f"  {name:<50}: {'PASS' if ok else 'FAIL'}")
    print("=" * 72)
    assert all(checks.values()), "phase4c_diagnostics dry-run FAILED"


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4c §6 read-only diagnostics.")
    ap.add_argument("--run", action="store_true", help="run against real data (else dry-run)")
    ap.add_argument("--target", default="tgt_fwd_logret_5b")
    args = ap.parse_args()

    if not args.run:
        _dry_run()
        return
    run_all(args.target)


if __name__ == "__main__":
    main()
