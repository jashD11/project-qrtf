"""
Phase 4 capacity study — how much money the NSE-500 universe can actually hold.

**Why this exists.** Phase 4 rebuilt the panel at 500 names because `IR ≈ IC·√breadth`
and 68 names is too few. But breadth is only real if the names are *tradeable*, and
tradeability is not a property of the universe alone — it is a property of the universe
**and the amount of money being run**. A stock that absorbs a Rs 2 lakh order without
moving is a completely different instrument from the same stock facing a Rs 2 crore
order.

So the question "is the bottom of the 500 too thin?" has no answer until AUM is fixed.
This module answers it across four orders of magnitude, Rs 10 lakh → Rs 1,000 crore, and
writes the table for later reference.

**What it computes.** For each AUM on the grid, over every in-universe bar:

    position     = AUM / book_size,  book_size = decile_pct × N   (a 500-name universe
                   at decile_pct=10% gives a 50-name book)
    participation= position / that bar's rupee turnover
    impact       = 1e4 · coef · σ · √participation                (config.impact_bps)

plus the largest universe **N** that stays under a participation ceiling — because N is
not free either. Enlarging N shrinks each position (good, `1/N`) but reaches into thinner
names (bad, and turnover falls with rank much faster than `1/N`), so participation rises
with N and there is a largest feasible universe at every AUM.

**This is analysis only** — it reads the finished panel and universe mask, runs no trees,
and executes no backtest. Nothing here feeds the execution path yet; the per-name impact
term in `tier3_execution.py` is deliberately deferred.

Usage
    python src/phase4_data/capacity.py            # dry-run self-test
    python src/phase4_data/capacity.py --build    # full grid -> CSV + printed table
    python src/phase4_data/capacity.py --build --aum 1.0    # detail for one AUM
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
DEFAULT_OUT: Final[str] = "data/bhavcopy/capacity_curve.csv"

CRORE: Final[float] = 1e7

# Rs 10 lakh -> Rs 1,000 crore, ~3 points per decade over four decades.
AUM_GRID_CR: Final[tuple[float, ...]] = (
    0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0
)

DECILE_PCT: Final[float] = 0.10     # book = this fraction of the universe (Tier 1 decile)
VOL_WINDOW: Final[int] = 21         # trailing bars for the per-name daily vol estimate


def load_inputs(
    panel_path: str = DEFAULT_PANEL, mask_path: str = DEFAULT_MASK
) -> pd.DataFrame:
    """
    In-universe bars with the two inputs impact needs: rupee turnover and daily vol.

    Restricting to *in-universe* bars matters — the whole question is what the strategy
    would actually have traded, not what was listed.
    """
    for p in (panel_path, mask_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found — run the Phase 4 build first.")

    panel = pd.read_parquet(panel_path).sort_values(["entity", "date"])
    panel["ret"] = panel.groupby("entity", sort=False)["close"].pct_change()
    panel["sigma"] = (
        panel.groupby("entity", sort=False)["ret"]
        .transform(lambda s: s.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std())
    )

    mask = pd.read_parquet(mask_path)
    flags = mask.stack()
    keep = set(zip(*[flags[flags].index.get_level_values(i) for i in (0, 1)]))

    sub = panel[[(d, t) in keep for d, t in zip(panel["date"], panel["ticker"])]]
    sub = sub[sub["turnover"].gt(0) & sub["sigma"].notna()]
    print(
        f"[capacity] {len(sub):,} in-universe bars with turnover + vol | "
        f"{sub['ticker'].nunique():,} names | "
        f"{sub['date'].min().date()} → {sub['date'].max().date()}"
    )
    return sub[["date", "ticker", "turnover", "sigma"]]


def universe_rank_turnover(bars: pd.DataFrame) -> pd.Series:
    """
    Typical rupee turnover at each universe rank position (1 = most liquid).

    Ranked within each date, then a median across dates — so rank 500 means "the least
    liquid name the universe held that day", which is exactly the binding constraint.
    """
    r = bars.groupby("date")["turnover"].rank(ascending=False, method="first")
    return bars.assign(rank=r).groupby("rank")["turnover"].median()


MIN_N: Final[int] = 50   # below this the decile book is too small to be cross-sectional


def max_feasible_n(
    rank_turnover: pd.Series,
    aum_rupees: float,
    ceiling: float,
    decile_pct: float,
    min_n: int = MIN_N,
) -> int:
    """
    Largest universe N whose least-liquid member stays under the participation ceiling.

    Both sides of the trade-off move with N: the book is ``decile_pct·N`` names so each
    position shrinks as ``1/N``, but the marginal name is less liquid. Turnover falls
    with rank faster than ``1/N``, so participation rises and the constraint binds.

    ``min_n`` exists to reject a degenerate answer. At very small N the decile book
    rounds down to a single name, so "hold only the most liquid stock in India" passes
    the ceiling trivially and the search reports a tiny feasible N as though it were a
    strategy. It is not a cross-sectional strategy at all, so anything below ``min_n``
    (a 5-name book at decile_pct=10%) is reported as infeasible — which is the honest
    answer: at that AUM this approach does not fit in this universe.
    """
    best = 0
    for n in rank_turnover.index.astype(int):
        if n < min_n:
            continue
        book = max(1, int(round(decile_pct * n)))
        position = aum_rupees / book
        if position / rank_turnover.loc[n] <= ceiling:
            best = n
    return best


def sweep(
    bars: pd.DataFrame,
    aum_grid_cr: tuple[float, ...] = AUM_GRID_CR,
    n_universe: int = 500,
    decile_pct: float = DECILE_PCT,
    ceiling: float | None = None,
    coef: float | None = None,
) -> pd.DataFrame:
    """Run the capacity grid and return one row per AUM."""
    cost = config.ML_CONFIG.cost
    ceiling = cost.max_participation if ceiling is None else ceiling
    coef = cost.impact_coef if coef is None else coef

    rank_to = universe_rank_turnover(bars)
    book = max(1, int(round(decile_pct * n_universe)))
    turnover = bars["turnover"].to_numpy()
    sigma = bars["sigma"].to_numpy()

    rows = []
    for aum_cr in aum_grid_cr:
        aum = aum_cr * CRORE
        position = aum / book
        part = position / turnover
        # Vectorized form of config's impact_bps, same formula.
        impact = 1e4 * coef * sigma * np.sqrt(part)
        rows.append({
            "aum_cr": aum_cr,
            "position_rs": position,
            "book_size": book,
            "part_median": float(np.median(part)),
            "part_p90": float(np.quantile(part, 0.90)),
            "part_p99": float(np.quantile(part, 0.99)),
            "frac_bars_over_ceiling": float((part > ceiling).mean()),
            "impact_bps_median": float(np.median(impact)),
            "impact_bps_p90": float(np.quantile(impact, 0.90)),
            "impact_bps_mean": float(np.mean(impact)),
            "max_n_at_ceiling": max_feasible_n(rank_to, aum, ceiling, decile_pct),
        })
    out = pd.DataFrame(rows)
    # Breadth actually purchasable, vs the 68 names Phase 3 ran on. This is the number
    # that decides whether the Phase 4 thesis survives at a given size.
    out["breadth_vs_68"] = np.sqrt(out["max_n_at_ceiling"].clip(lower=1) / 68.0)
    return out


def print_table(curve: pd.DataFrame, ceiling: float) -> None:
    """Human-readable capacity table."""
    print(f"\n{'=' * 96}")
    print(f"CAPACITY CURVE — participation ceiling {ceiling:.0%}, book = "
          f"{int(curve['book_size'].iloc[0])} names")
    print("=" * 96)
    print(f"{'AUM':>10}{'position':>12}{'median':>9}{'p90':>9}{'p99':>9}"
          f"{'bars>ceil':>11}{'impact':>9}{'max N':>8}{'breadth':>9}")
    print(f"{'(Rs cr)':>10}{'(Rs)':>12}{'part':>9}{'part':>9}{'part':>9}"
          f"{'':>11}{'(bps)':>9}{'':>8}{'vs 68':>9}")
    print("-" * 96)
    for _, r in curve.iterrows():
        print(
            f"{r['aum_cr']:>10,.2f}{r['position_rs']:>12,.0f}"
            f"{r['part_median']:>8.2%}{r['part_p90']:>9.2%}{r['part_p99']:>9.2%}"
            f"{r['frac_bars_over_ceiling']:>11.2%}{r['impact_bps_median']:>9.1f}"
            f"{int(r['max_n_at_ceiling']):>8}{r['breadth_vs_68']:>9.2f}x"
        )
    print("-" * 96)
    print("max N = largest universe whose least-liquid member stays under the ceiling.")
    print("breadth vs 68 = sqrt(max_N/68), the IR ceiling this size can actually buy.")


def build(
    panel_path: str = DEFAULT_PANEL,
    mask_path: str = DEFAULT_MASK,
    out_csv: str = DEFAULT_OUT,
) -> pd.DataFrame:
    """Run the sweep and persist it."""
    bars = load_inputs(panel_path, mask_path)
    curve = sweep(bars)
    ceiling = config.ML_CONFIG.cost.max_participation

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    curve.to_csv(out_csv, index=False)
    print_table(curve, ceiling)
    print(f"\n[capacity] wrote {out_csv}")
    return curve


# ── Dry run ─────────────────────────────────────────────────────────────────────

def _dry_run() -> None:
    """
    Self-test on a synthetic universe whose right answers are known analytically.

    Checks the three properties the study depends on: participation scales linearly with
    AUM, impact scales as the square root of it, and the feasible universe shrinks as
    money grows.
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=120)
    rows = []
    for i in range(100):
        level = 10.0 ** (9 - i / 25.0)            # turnover falls steeply with rank
        rows.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i:03d}",
            "turnover": level * rng.lognormal(0, 0.1, len(dates)),
            "sigma": 0.02,
        }))
    bars = pd.concat(rows, ignore_index=True)

    curve = sweep(bars, aum_grid_cr=(0.1, 1.0, 10.0, 100.0), n_universe=100, ceiling=0.10)

    # Participation is linear in AUM: 10x the money, 10x the share of volume.
    p = curve["part_median"].to_numpy()
    assert np.allclose(p[1] / p[0], 10.0, rtol=1e-6), f"participation not linear: {p}"
    assert np.allclose(p[3] / p[2], 10.0, rtol=1e-6)

    # Impact is sqrt: 10x the money, sqrt(10) ~ 3.16x the cost.
    b = curve["impact_bps_median"].to_numpy()
    assert np.allclose(b[1] / b[0], np.sqrt(10.0), rtol=1e-6), f"impact not sqrt: {b}"

    # Feasible universe is non-increasing in AUM, and strictly shrinks somewhere.
    n = curve["max_n_at_ceiling"].to_numpy()
    assert (np.diff(n) <= 0).all(), f"feasible N grew with AUM: {n}"
    assert n[0] > n[-1], f"feasible N never shrank across 1000x of AUM: {n}"

    # Formula parity: the vectorized sweep and config's scalar function must agree.
    # Compared pointwise, NOT via the medians — sqrt is monotone but taking a median of
    # an even-length array averages two neighbours, and avg(sqrt) != sqrt(avg).
    coef = config.ML_CONFIG.cost.impact_coef
    for part in (1e-4, 1e-3, 1e-2, 0.1, 1.0):
        vec = float(1e4 * coef * 0.02 * np.sqrt(part))
        sca = config.ML_CONFIG.cost.impact_bps(0.02, part, coef)
        assert abs(vec - sca) < 1e-9, f"part={part}: sweep {vec} != config {sca}"
    assert config.ML_CONFIG.cost.impact_bps(0.02, 0.0, coef) == 0.0, "zero-trade impact"

    print("\n[capacity] dry-run OK — participation linear, impact sqrt, N shrinks with AUM")


def main() -> None:
    ap = argparse.ArgumentParser(description="NSE-500 capacity / participation study.")
    ap.add_argument("--build", action="store_true", help="run the grid (else dry-run)")
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--mask", default=DEFAULT_MASK)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.build:
        _dry_run()
        return
    build(args.panel, args.mask, args.out)


if __name__ == "__main__":
    main()
