"""
Phase 6 Part B, F1 — market-wide features (docs/phase6_plan.md §B.2).

    mkt_vix_z     India VIX close as an expanding z-score.
    mkt_pe_z      Nifty 50 P/E as an expanding z-score, spliced at the 2021-03-31
                  methodology break (§B.8).
    mkt_ret_21b   Nifty 50 21-session log return.

These are the same for every stock on a date, so they are **not** rank-normalized
(§B.2): the frame is ``(date × feature)`` and enters the model raw.

**Expanding z-scores** use every published observation up to and including t
(ddof = 1), and need at least ``Z_MIN_OBS`` observations; before that the value is 0. They
are computed on the archive's own sessions, then carried onto the panel calendar. India
VIX starts 2014-05-14 in the archive, so ``mkt_vix_z`` is 0 until mid-2015.

**The P/E break.** On 2021-03-31 NSE switched index P/E from standalone to consolidated
earnings. The Nifty 50 P/E fell 18% that day (40.43 → 33.20) while the index moved
under 1%. An expanding z across that break would read a methodology change as the market
getting cheap. The series is spliced: every value before the break is scaled by

    k = (PE_b / PE_b−1) / (N50_b / N50_b−1),

the day's P/E move net of the index's own move. Scaling every earlier value by one
constant leaves each z-score before the break unchanged, so nothing before 2021-03-31
depends on k. From the break on, k is known at the close of the break day.

    python src/phase6_features/market.py            # dry-run self-test
    python src/phase6_features/market.py --build
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Final

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.phase6_features import _common as C  # noqa: E402
from src.phase6_features import index_series as IS  # noqa: E402

Z_MIN_OBS: Final[int] = 252
RET_WINDOW: Final[int] = 21
PE_BREAK: Final[pd.Timestamp] = pd.Timestamp("2021-03-31")
# A day's P/E move net of the index move. The break is ~-0.20; no other session in the
# archive exceeds this (largest otherwise ~0.05), which build() asserts.
_BREAK_TOL: Final[float] = 0.10
FEATURES: Final[tuple[str, ...]] = ("mkt_vix_z", "mkt_pe_z", "mkt_ret_21b")
OUT: Final[str] = os.path.join(C.F1_DIR, "market.parquet")


def expanding_z(x: pd.Series, min_obs: int = Z_MIN_OBS) -> pd.Series:
    """(x_t - mean(x_<=t)) / std(x_<=t), over published values only; 0 below min_obs."""
    obs = x.dropna()
    ex = obs.expanding(min_periods=1)
    z = (obs - ex.mean()) / ex.std()
    z = z.where(ex.count() >= min_obs, 0.0)
    return z.reindex(x.index)


def splice_pe(pe: pd.Series, level: pd.Series, at: pd.Timestamp = PE_BREAK) -> pd.Series:
    """Scale the P/E history before ``at`` by the break's net-of-market jump."""
    i = pe.index.get_loc(at)
    k = (pe.iloc[i] / pe.iloc[i - 1]) / (level.loc[pe.index[i]] / level.loc[pe.index[i - 1]])
    out = pe.copy()
    out.iloc[:i] = out.iloc[:i] * k
    return out


def net_jumps(pe: pd.Series, level: pd.Series) -> pd.Series:
    """Daily log P/E change minus the index's log change, on the archive's sessions."""
    return np.log(pe).diff() - np.log(level.reindex(pe.index)).diff()


def compute(vix: pd.Series, pe: pd.Series, n50: pd.Series, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """
    ``vix``, ``pe`` and ``n50`` are on the archive's own sessions (unfilled). Output is on
    the panel calendar; z-scores are carried over the sessions the archive lacks.
    """
    z_vix = expanding_z(vix).reindex(calendar).ffill().fillna(0.0)
    z_pe = expanding_z(splice_pe(pe, n50)).reindex(calendar).ffill().fillna(0.0)
    level = n50.reindex(calendar).ffill()
    ret = np.log(level / level.shift(RET_WINDOW))
    return pd.DataFrame({"mkt_vix_z": z_vix, "mkt_pe_z": z_pe, "mkt_ret_21b": ret}, index=calendar)


def build(allow_holdout: bool = False) -> pd.DataFrame:
    panel_dates = pd.read_parquet(C.PANEL, columns=["date"])["date"]
    C.guard_holdout(panel_dates, C.PANEL, allow_holdout)
    calendar = pd.DatetimeIndex(np.sort(panel_dates.unique()))
    long = IS.load()
    C.guard_holdout(long["date"], C.INDEX_DAILY, allow_holdout)

    def series(field: str, name: str) -> pd.Series:
        return long[long["index"] == name].set_index("date")[field].sort_index().dropna()

    vix, pe, n50 = series("close", IS.VIX), series("pe", IS.NIFTY50), series("close", IS.NIFTY50)
    jumps = net_jumps(pe, n50).abs()
    off_break = jumps.drop(PE_BREAK)
    if not (jumps.loc[PE_BREAK] > _BREAK_TOL and off_break.max() < _BREAK_TOL):
        raise AssertionError(
            f"P/E break check failed: break jump {jumps.loc[PE_BREAK]:.3f}, "
            f"largest other {off_break.max():.3f} on {off_break.idxmax().date()}"
        )
    print(f"[market] P/E break {PE_BREAK.date()}: net jump {jumps.loc[PE_BREAK]:.3f} | "
          f"largest other session {off_break.max():.3f} ({off_break.idxmax().date()})")
    out = compute(vix, pe, n50, calendar)
    for f in FEATURES:
        nz = out[f].dropna()
        nz = nz[nz != 0]
        print(f"[market] {f}: first defined nonzero {nz.index.min().date()} | "
              f"range [{out[f].min():.2f}, {out[f].max():.2f}]")
    C.write(out, OUT)
    return out


# ── Dry run ─────────────────────────────────────────────────────────────────────

def _dry_run() -> None:
    checks: dict[str, bool] = {}
    rng = np.random.default_rng(9)
    cal = pd.bdate_range("2019-01-01", periods=900)
    n50 = pd.Series(10000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(cal)))), index=cal)
    vix = pd.Series(15 + np.abs(rng.normal(0, 3, len(cal))), index=cal)
    vix.iloc[:100] = np.nan                                       # starts late
    earn = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.002, len(cal)))), index=cal)
    pe = 25 * (n50 / n50.iloc[0]) / earn

    z = expanding_z(vix)
    checks["z is 0 below 252 observations"] = bool((z.iloc[100:100 + 251] == 0).all() and z.iloc[:100].isna().all())
    t = 600
    obs = vix.iloc[100:t + 1]
    checks["z formula exact"] = bool(np.isclose(z.iloc[t], (vix.iloc[t] - obs.mean()) / obs.std()))
    checks["z causal (truncation-invariant)"] = bool(expanding_z(vix.iloc[:500]).equals(z.iloc[:500]))

    # A pure methodology break on day b: P/E falls 20% with no index move that day.
    b = cal[700]
    n50_flat = n50.copy()
    n50_flat.iloc[700:] = n50_flat.iloc[700:] * (n50_flat.iloc[699] / n50_flat.iloc[700])
    earn_flat = earn.copy()          # no genuine earnings news on the break day either
    earn_flat.iloc[700:] = earn_flat.iloc[700:] * (earn_flat.iloc[699] / earn_flat.iloc[700])
    pe_true = 25 * (n50_flat / n50_flat.iloc[0]) / earn_flat
    pe_broken = pe_true.copy()
    pe_broken.iloc[700:] *= 0.8
    spliced = splice_pe(pe_broken, n50_flat, at=b)
    z_true, z_spl = expanding_z(pe_true), expanding_z(spliced)
    checks["splice removes a pure methodology break"] = bool(np.allclose(z_spl, z_true))
    checks["splice leaves pre-break z unchanged"] = bool(
        np.allclose(z_spl.iloc[:700], expanding_z(pe_broken).iloc[:700])
    )
    checks["break detected net of the index move"] = bool(
        net_jumps(pe_broken, n50_flat).abs().idxmax() == b
    )

    out = compute(vix, pe, n50, cal)
    checks["mkt_ret_21b exact"] = bool(np.isclose(out["mkt_ret_21b"].iloc[400], np.log(n50.iloc[400] / n50.iloc[379])))
    checks["output NaN-free past warm-up"] = bool(out.iloc[RET_WINDOW:].notna().all().all())

    print()
    for name, v in checks.items():
        print(f"  {name:<44}: {'PASS' if v else 'FAIL'}")
    assert all(checks.values()), "market dry-run FAILED"
    print("\n[market] dry-run OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Phase 6 F1 market-wide features.")
    ap.add_argument("--build", action="store_true", help="build (else dry-run)")
    args = ap.parse_args()
    if not args.build:
        _dry_run()
        return
    build()


if __name__ == "__main__":
    main()
