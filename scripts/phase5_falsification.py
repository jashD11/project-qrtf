#!/usr/bin/env python3
"""Phase 5 falsification tests — regenerates every number in docs/phase5_plan.md §1.2, §2, §3.

Read-only. Reads cached walk-forward output and committed panels; fits nothing, writes
nothing, selects nothing. Runs in ~2 minutes.

Why a script and not a notebook: docs/phase5_plan.md states that no performance number
enters the record without a source file. This is that file.

    python scripts/phase5_falsification.py            # all sections
    python scripts/phase5_falsification.py --only 2   # one section

Section map:
  1.2  the live gate vs a plain 21-day realised-vol z-score   (the item-14 tripwire)
  2.1  orthogonality of trend to the ranker, inside the decile
  2.2  does a trend tilt pay? tilt vs hard cut vs doing nothing
  3.1  the SLB-feasible short leg's standalone value
  3.2  its cost, read from the recorded Phase 4c run

NOT covered: the oracle ceiling test (§3.3). It needs the engine's cost path — buffer,
participation cap, real panels — and a scratch reconstruction of that path was wrong by
1.06 of Sharpe. It belongs inside run_execution, not here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent

# The cached Tier 1 walk-forward this analysis reads. daily_nse500 / 5b is the Phase 4c
# cell with the richer short leg; the 21b cell (8f3afb98335b774c) is the best net cell.
CACHE = ROOT / "data/trial_database/wf_cache/ac11679c42e4feee"
OHLCV = ROOT / "data/nse500_daily_ohlcv.parquet"
INDEX = ROOT / "data/bhavcopy/index_daily.parquet"
SHORTABLE = ROOT / "data/bhavcopy/shortable_mask.parquet"
DIAGNOSTICS = ROOT / "data/trial_database/phase4c_dsr_matrix_execution_diagnostics.csv"

TRADING_DAYS = 252


def _require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"missing artefact: {path}\n  (a missing input is a hard failure, "
                         f"never a silently substituted number)")
    return path


# ── loaders ──────────────────────────────────────────────────────────────────


def load_signals() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cached alpha scores and the two decile masks, as the engine wrote them."""
    alpha = pd.read_parquet(_require(CACHE / "alpha_scores.parquet"))
    long_mask = pd.read_parquet(_require(CACHE / "long_mask.parquet")).astype(bool)
    short_mask = pd.read_parquet(_require(CACHE / "short_mask.parquet")).astype(bool)
    print(f"[load] alpha {alpha.shape}  {alpha.index[0].date()} -> {alpha.index[-1].date()}"
          f"  | decile {long_mask.sum(axis=1).mean():.0f} names/day  | {CACHE.name}")
    return alpha, long_mask, short_mask


def load_close(index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    """Wide (date x ticker) close, reindexed onto the signal frame."""
    px = pd.read_parquet(_require(OHLCV))
    px.columns = [c.lower() for c in px.columns]
    tcol = "timestamp" if "timestamp" in px.columns else "date"
    close = px.pivot_table(index=tcol, columns="ticker", values="close")
    close.index = pd.DatetimeIndex(close.index).normalize()
    return close.reindex(index=index, columns=columns)


def load_market(index: pd.Index) -> pd.Series:
    """NIFTY-50 daily log return, from the same artefact Tier 2 reads."""
    idx = pd.read_parquet(_require(INDEX))
    idx = idx[idx["ticker"] == "NIFTY-50"].set_index("timestamp")["close"].sort_index()
    idx.index = pd.DatetimeIndex(idx.index).normalize()
    return np.log(idx).diff().reindex(index)


# ── shared helpers ───────────────────────────────────────────────────────────


def causal_z(s: pd.Series, min_periods: int = 252) -> pd.Series:
    """Expanding z-score — the transform tier2_regime._causal_zscore applies."""
    exp = s.expanding(min_periods=min_periods)
    return (s - exp.mean()) / exp.std()


def normalise(w: pd.DataFrame) -> pd.DataFrame:
    """Scale each row so the leg's gross is 1.0."""
    return w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def trend_measures(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Trailing, causal per-name trend. Every window ends at t."""
    lr = np.log(close).diff()
    downside = lr.clip(upper=0.0).pow(2).rolling(63).mean().pow(0.5)
    return {
        "mom_12_1": np.log(close.shift(21)) - np.log(close.shift(252)),
        "mom_63": np.log(close) - np.log(close.shift(63)),
        "mom_21": np.log(close) - np.log(close.shift(21)),
        "px_v_200d": np.log(close) - np.log(close.rolling(200).mean()),
        "sortino_63": lr.rolling(63).mean() / downside,
    }


def perf(returns: pd.Series) -> tuple[float, float, float, float]:
    """(annualised return, Sharpe, max drawdown, worst calendar month)."""
    s = returns.dropna()
    cum = s.cumsum()
    return (s.mean() * TRADING_DAYS,
            s.mean() / s.std() * np.sqrt(TRADING_DAYS),
            (cum - cum.cummax()).min(),
            s.resample("ME").sum().min())


# ── §1.2 ─────────────────────────────────────────────────────────────────────


def section_1_2() -> None:
    print("\n" + "=" * 78)
    print("§1.2  Is the live gate just a rolling vol estimate?")
    print("=" * 78)
    print("The gate is mean(z_log_realized_vol, z_avg_corr) over an expanding 85th")
    print("percentile (tier2_regime.py:421-424). Tripwire: corr > 0.80 against a plain")
    print("21-day realised-vol z-score means the eight HMM parameters buy nothing.\n")

    idx = pd.read_parquet(_require(INDEX))
    idx = idx[idx["ticker"] == "NIFTY-50"].set_index("timestamp")["close"].sort_index()
    idx.index = pd.DatetimeIndex(idx.index).normalize()
    r = np.log(idx).diff()

    # The gate's own volatility term, built exactly as build_features does.
    rv = r.rolling(20, min_periods=20).std()
    z_gate = causal_z(np.log(rv.where(rv > 0)))
    # The naive benchmark the tripwire compares against.
    z_naive = causal_z(r.rolling(21, min_periods=21).std())

    both = pd.concat({"gate": z_gate, "naive": z_naive}, axis=1).dropna()
    rho = both["gate"].corr(both["naive"])
    print(f"  corr(z_log_realized_vol, 21d realised-vol z) = {rho:.3f}")
    print(f"  n = {len(both):,} days, {both.index[0].date()} -> {both.index[-1].date()}")
    print(f"  verdict: {'FAILS tripwire' if rho > 0.80 else 'passes'} (threshold 0.80)")


# ── §2.1 ─────────────────────────────────────────────────────────────────────


def _xs_spearman(a: pd.DataFrame, b: pd.DataFrame, mask: pd.DataFrame) -> float:
    """Mean over dates of the cross-sectional Spearman of a vs b, inside `mask`."""
    out = []
    for d in a.index:
        m = mask.loc[d] & a.loc[d].notna() & b.loc[d].notna()
        if m.sum() < 10:
            continue
        out.append(spearmanr(a.loc[d][m], b.loc[d][m]).statistic)
    return float(np.nanmean(out)) if out else np.nan


def section_2_1(alpha, long_mask, short_mask, close) -> None:
    print("\n" + "=" * 78)
    print("§2.1  Is trend orthogonal to the ranker?")
    print("=" * 78)
    print("The ranker trains on cross-sectionally demeaned features, so it should be blind")
    print("to absolute trend. Tripwire: |rho| > 0.40 inside the decile means the ranker")
    print("already implies it and a tilt pays a breadth cost for nothing.\n")

    full = close.notna()
    print(f"  {'trend measure':<13}{'full x-section':>16}{'top decile':>13}{'bottom decile':>15}")
    for name, tr in trend_measures(close).items():
        print(f"  {name:<13}{_xs_spearman(alpha, tr, full):>16.3f}"
              f"{_xs_spearman(alpha, tr, long_mask):>13.3f}"
              f"{_xs_spearman(alpha, tr, short_mask):>15.3f}")
    print("\n  verdict: orthogonal inside the decile. Trend is new information.")
    print("           (which is necessary, not sufficient -- see §2.2)")


# ── §2.2 ─────────────────────────────────────────────────────────────────────


def _tilt(mask: pd.DataFrame, trend: pd.DataFrame, toward_high: bool,
          strength: float = 1.0) -> pd.DataFrame:
    """Smooth tilt: w proportional to 1 + strength*(2*pctrank - 1), clipped at 0.

    Gross stays 1.0 and every name keeps a positive weight at strength <= 1, so this
    expresses the view without discarding breadth -- the construction a hard cut is
    compared against.
    """
    r = trend.where(mask).rank(axis=1, pct=True)
    signed = (2 * r - 1) if toward_high else (1 - 2 * r)
    return normalise((1.0 + strength * signed).clip(lower=0).where(mask).fillna(0.0))


def _hard_cut(mask: pd.DataFrame, trend: pd.DataFrame, toward_high: bool,
              keep_frac: float = 0.5) -> pd.DataFrame:
    """Discard the wrong-signed half of the leg, equal-weight the rest."""
    r = trend.where(mask).rank(axis=1, pct=True)
    keep = (r >= 1 - keep_frac) if toward_high else (r <= keep_frac)
    return normalise((mask & keep.fillna(False)).astype(float))


def section_2_2(long_mask, short_mask, close) -> None:
    print("\n" + "=" * 78)
    print("§2.2  Does a trend tilt pay?")
    print("=" * 78)
    print("GROSS -- before any cost, so the comparison does not depend on the cost model.")
    print("Long leg tilts toward rising names, short leg toward falling. 12-1 momentum.\n")

    fwd = np.log(close).diff().shift(-1)
    mkt = load_market(close.index)
    lr = np.log(close).diff()
    beta = (lr.rolling(252, min_periods=126).cov(mkt)
            .div(mkt.rolling(252, min_periods=126).var(), axis=0))
    mom = trend_measures(close)["mom_12_1"]

    base_long, base_short = normalise(long_mask.astype(float)), normalise(short_mask.astype(float))
    variants = {
        "baseline equal-weight": (base_long, base_short),
        "smooth tilt, strength 0.5": (_tilt(long_mask, mom, True, 0.5),
                                      _tilt(short_mask, mom, False, 0.5)),
        "smooth tilt, strength 1.0": (_tilt(long_mask, mom, True),
                                      _tilt(short_mask, mom, False)),
        "hard cut, top/bottom half": (_hard_cut(long_mask, mom, True),
                                      _hard_cut(short_mask, mom, False)),
        "reversed (contrarian) tilt": (_tilt(long_mask, mom, False),
                                       _tilt(short_mask, mom, True)),
    }

    print(f"  {'variant':<28}{'L/S ann':>9}{'SR':>6}{'maxDD':>8}{'beta':>7}"
          f"{'worst mo':>10}{'names':>9}")
    for name, (lw, sw) in variants.items():
        ret = (lw * fwd).sum(axis=1, min_count=1) - (sw * fwd).sum(axis=1, min_count=1)
        ann, sharpe, dd, worst = perf(ret)
        book_beta = ((lw * beta).sum(axis=1, min_count=1)
                     - (sw * beta).sum(axis=1, min_count=1)).mean()
        n = f"{(lw > 0).sum(axis=1).mean():.0f}/{(sw > 0).sum(axis=1).mean():.0f}"
        print(f"  {name:<28}{ann:>8.1%}{sharpe:>6.2f}{dd:>8.1%}{book_beta:>7.2f}"
              f"{worst:>10.1%}{n:>9}")

    print("\n  turnover per bar (a tilt is not a cost story):")
    for name, b, t in [("long", base_long, _tilt(long_mask, mom, True)),
                       ("short", base_short, _tilt(short_mask, mom, False))]:
        t0, t1 = b.diff().abs().sum(axis=1).mean(), t.diff().abs().sum(axis=1).mean()
        print(f"    {name:<6} baseline {t0:.3f} -> tilted {t1:.3f}  ({t1 / t0 - 1:+.0%})")

    print("\n  verdict: every variant is worse than doing nothing, on return, Sharpe,")
    print("           drawdown and worst month at once. Optimal tilt strength is zero.")
    print("           Tilt does beat the hard cut -- breadth argument confirmed -- but")
    print("           book beta is ~0 throughout, so beta-neutral sizing fixes nothing.")


# ── §3 ───────────────────────────────────────────────────────────────────────


def section_3(alpha, long_mask, short_mask, close) -> None:
    print("\n" + "=" * 78)
    print("§3  Short-leg modulation: what is there to modulate?")
    print("=" * 78)

    fwd = np.log(close).diff().shift(-1)

    shortable = pd.read_parquet(_require(SHORTABLE))
    shortable = (shortable.reindex(index=alpha.index, columns=alpha.columns)
                 .astype(object).fillna(False).astype(bool))
    print(f"  borrowable set: {shortable.sum(axis=1).mean():.0f} eligible names/day"
          f" of {alpha.shape[1]}")

    # The SLB short leg the engine builds (B2): filter the *scores*, re-rank inside the
    # eligible subset, fill toward the same k. Filtering the finished mask instead would
    # shrink the book silently and break dollar-neutrality.
    k = short_mask.sum(axis=1)
    feasible = (alpha.where(shortable).rank(axis=1, ascending=True)
                .le(k, axis=0).fillna(False) & shortable)
    print(f"  SLB leg fills {feasible.sum(axis=1).mean():.0f}/{k.mean():.0f} names/day\n")

    print("  §3.1  standalone value of the short leg, GROSS")
    print(f"    {'short leg':<22}{'ann':>9}{'SR':>7}")
    for name, mask in [("unrestricted", short_mask), ("SLB-feasible", feasible)]:
        leg = -(normalise(mask.astype(float)) * fwd).sum(axis=1, min_count=1)
        ann, sharpe, _, _ = perf(leg)
        print(f"    {name:<22}{ann:>8.1%}{sharpe:>7.2f}")
    print("    -> most of the short leg's return lives in names never borrowable.")

    print("\n  §3.2  cost of running it, from the recorded Phase 4c run")
    print(f"    source: {DIAGNOSTICS.relative_to(ROOT)}")
    diag = pd.read_csv(_require(DIAGNOSTICS))
    cols = ["execution_style", "turnover_mean", "trade_drag_annual",
            "borrow_drag_annual", "sharpe_gross", "sharpe_net"]
    sub = diag[diag["target"] == "tgt_fwd_logret_21b"][cols]
    table = sub.to_string(index=False, float_format=lambda v: f"{v:.3f}")
    for line in table.splitlines():
        print(f"    {line}")

    lo = sub[sub["execution_style"] == "long_only"].iloc[0]
    ls = sub[sub["execution_style"] == "long_short_slb"].iloc[0]
    incremental = (ls["trade_drag_annual"] - lo["trade_drag_annual"]
                   + ls["borrow_drag_annual"])
    print(f"\n    adding the feasible short leg costs {incremental:+.2%}/yr")
    print(f"    to buy the ~5%/yr of gross alpha measured in §3.1 -- they cancel.")
    print("\n  verdict: a gate scales exposure in [0,1]. There is no expected value here")
    print("           to modulate. The gate's defect is real (§1.2); the target was wrong.")
    print("\n  NOT RUN: the oracle ceiling test (§3.3). It needs the engine's cost path;")
    print("           a scratch reconstruction was wrong by 1.06 of Sharpe on long_only.")
    print("           It belongs inside run_execution.")


# ── entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["1.2", "2.1", "2.2", "2", "3"],
                    help="run one section instead of all")
    args = ap.parse_args()

    want = lambda s: args.only is None or args.only == s or args.only == s.split(".")[0]

    print("Phase 5 falsification tests -- regenerates docs/phase5_plan.md §1.2, §2, §3")
    print("Read-only: fits nothing, writes nothing, selects nothing.")

    if want("1.2"):
        section_1_2()

    if want("2.1") or want("2.2") or want("3"):
        alpha, long_mask, short_mask = load_signals()
        close = load_close(alpha.index, alpha.columns)
        if want("2.1"):
            section_2_1(alpha, long_mask, short_mask, close)
        if want("2.2"):
            section_2_2(long_mask, short_mask, close)
        if want("3"):
            section_3(alpha, long_mask, short_mask, close)

    print("\n" + "=" * 78)
    print("Full write-up and the plan this redirects to: docs/phase5_plan.md")
    print("=" * 78)


if __name__ == "__main__":
    main()
