"""
Phase 6 Part B, F2 — stock-futures features (docs/phase6_plan.md §B.2).

    fut_basis       near-month annualized log(F / S) × 365 / days to expiry, minus the
                    date's cross-sectional median (which removes the common rate term).
    fut_oi_chg_5b   log change over 5 own bars of total stock-futures open interest,
                    summed over expiries.

The near month is ``options.near_month``'s (at least 7 days out, and F only if the future
traded). Futures trade on **unadjusted** prices, so S is the panel's raw close,
``close / adj_factor``. Open interest is in shares, so a split would show as a jump in
OI. Each day's OI is divided by that day's ``adj_factor``, as the panel does for volume,
so the 5-bar change is unaffected by corporate actions.

    python src/phase6_features/futures.py            # dry-run self-test
    python src/phase6_features/futures.py --build
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
from src.phase6_features import fo_archive as FO  # noqa: E402
from src.phase6_features.options import F2_DIR, near_month, symbol_map  # noqa: E402

OI_LAG: Final[int] = 5
FEATURES: Final[tuple[str, ...]] = ("fut_basis", "fut_oi_chg_5b")


def compute(panel: pd.DataFrame, fo: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    ``panel``: own-bar long panel with ``symbol`` and ``adj_factor``. ``fo``: the
    condensed archive. Returns wide (date × ticker) frames.
    """
    nm = near_month(fo)
    oi = fo[fo["kind"] == "F"].groupby(["date", "symbol"], observed=True)["oi"].sum()
    df = panel.merge(nm[["date", "symbol", "days", "F"]], on=["date", "symbol"], how="left")
    df = df.merge(oi.rename("oi").reset_index(), on=["date", "symbol"], how="left")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    spot = df["close"] / df["adj_factor"]
    raw_basis = np.log(df["F"] / spot) * 365.0 / df["days"]
    df["fut_basis"] = raw_basis - raw_basis.groupby(df["date"]).transform("median")

    oi_adj = (df["oi"] / df["adj_factor"]).where(df["oi"] > 0)
    df["fut_oi_chg_5b"] = np.log(oi_adj / oi_adj.groupby(df["ticker"], sort=False).shift(OI_LAG))
    df = df.replace([np.inf, -np.inf], np.nan)
    out = {f: C.to_wide(df, f) for f in FEATURES}
    out["_raw_basis"] = C.to_wide(df.assign(rb=raw_basis), "rb")
    out["_spot_gap"] = C.to_wide(df.assign(g=np.log(df["F"] / spot)), "g")
    return out


def build(allow_holdout: bool = False) -> dict[str, pd.DataFrame]:
    fo = FO.load()
    C.guard_holdout(fo["date"], FO.OUT, allow_holdout)
    panel = C.load_panel(allow_holdout=allow_holdout)
    adj = pd.read_parquet(C.PANEL, columns=["date", "ticker", "adj_factor"])
    panel = panel.merge(adj, on=["date", "ticker"], how="left").merge(
        symbol_map(), on=["date", "ticker"], how="left")
    out = compute(panel, fo)

    gap = out.pop("_spot_gap").stack().abs()
    rb = out.pop("_raw_basis").stack()
    print(f"[futures] |log(F/S)| near month: median {gap.median():.4f}, p99 {gap.quantile(0.99):.4f}, "
          f"share > 10% {float((gap > 0.10).mean()):.4%} | raw annualized basis median "
          f"{rb.median():.3f} (the rate it removes)")
    os.makedirs(F2_DIR, exist_ok=True)
    for name, frame in out.items():
        C.write(frame, os.path.join(F2_DIR, f"{name}.parquet"))
    return out


# ── Dry run ─────────────────────────────────────────────────────────────────────

def _dry_run() -> None:
    checks: dict[str, bool] = {}
    dates = pd.bdate_range("2021-01-04", periods=30)
    rows, panel = [], []
    rate, prem = {"AAA": 0.07, "BBB": 0.05}, {"AAA": 0.0, "BBB": 0.0}
    for s in ("AAA", "BBB", "CCC"):
        raw = 100.0 * np.exp(np.cumsum(np.full(len(dates), 0.001)))
        adjf = np.where(np.arange(len(dates)) < 15, 0.5, 1.0) if s == "AAA" else np.ones(len(dates))
        panel.append(pd.DataFrame({"date": dates, "ticker": s, "symbol": s, "open": raw * adjf,
                                   "close": raw * adjf, "turnover": 1e7, "trades": 10,
                                   "adj_factor": adjf}))
        if s == "CCC":
            continue                                   # no F&O
        for i, d in enumerate(dates):
            exp = d + pd.Timedelta(days=28)
            F = raw[i] * np.exp(rate[s] * 28 / 365 + prem[s])
            # AAA splits 2:1 on bar 15: raw OI in shares doubles, the economic position does not.
            oi = 1e6 * (2.0 if (s == "AAA" and i >= 15) else 1.0) * (1 + 0.01 * i)
            rows.append((d, s, "F", exp, 0.0, F, 50, 1e8, oi))
    fo = pd.DataFrame(rows, columns=FO.SCHEMA)
    panel = pd.concat(panel, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    out = compute(panel, fo)

    checks["raw basis = the planted rate"] = bool(
        np.allclose(out["_raw_basis"]["AAA"].dropna(), 0.07) and np.allclose(out["_raw_basis"]["BBB"].dropna(), 0.05))
    checks["demeaned basis = rate − cross-sectional median"] = bool(
        np.allclose(out["fut_basis"]["AAA"].dropna(), 0.01) and np.allclose(out["fut_basis"]["BBB"].dropna(), -0.01))
    checks["basis uses raw spot across a split"] = bool(out["fut_basis"]["AAA"].notna().all())
    oc = out["fut_oi_chg_5b"]["AAA"]
    checks["OI change unaffected by the split"] = bool(
        np.isclose(oc.iloc[17], np.log((1 + 0.17) / (1 + 0.12))))
    checks["no F&O -> NaN"] = bool(out["fut_basis"]["CCC"].isna().all())

    cut = dates[20]
    fo2 = fo.copy()
    fo2.loc[fo2["date"] > cut, ["close", "oi"]] *= 1.3
    out2 = compute(panel, fo2)
    checks["causal: features <= cut unchanged"] = all(
        out[f].loc[:cut].equals(out2[f].loc[:cut]) for f in FEATURES)

    print()
    for k, v in checks.items():
        print(f"  {k:<46}: {'PASS' if v else 'FAIL'}")
    assert all(checks.values()), "futures dry-run FAILED"
    print("\n[futures] dry-run OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Phase 6 F2 futures features.")
    ap.add_argument("--build", action="store_true", help="build (else dry-run)")
    args = ap.parse_args()
    if not args.build:
        _dry_run()
        return
    build()


if __name__ == "__main__":
    main()
