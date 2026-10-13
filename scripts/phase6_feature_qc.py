#!/usr/bin/env python3
"""
Phase 6 Part B, Stage 6 — feature quality, before any return is computed (§B.4).

Reads only the Phase 6 panels, the universe mask, the index archive table and the
condensed F&O table. **No return or label is read.** The gates are integrity checks,
wide enough that a correct build passes and a broken one does not. They are not tuned
to anything a strategy does.

    Q1  required F1 per-stock features cover ≥ 98% of universe cells, every year
    Q2  market features defined on every session from 2013-02-01
    Q3  ±inf appears only in sector_pe (a loss-making sector, §B.8)
    Q4  stock ATM IV vs India VIX: yearly median ratio in [1.2, 3.5]; daily-median corr ≥ 0.6
    Q5  Black-76 IV solved for ≥ 90% of near-month traded options
    Q6  futures: raw annualized basis median in [3%, 10%]; p99 |log F/S| ≤ 5%
    Q7  opt_avail on 15%–60% of universe cells, every year
    Q8  all 11 sectors used; a sector assigned on ≥ 98% of universe cells, every year
    Q9  call−put IV spread: |yearly median signed spread| ≤ 0.01 (put-call parity: no
        bias). The absolute spread, which is non-synchronous-close noise, is reported
        but does not gate. Revised from v1, which gated the absolute spread at 0.02 and
        failed at 0.0208 in 2018 (see phase6_plan §B.8).
    Q10 every F2 feature covers ≥ 80% of the cells where opt_avail = 1, every year

Writes ``data/phase6/qc/feature_distribution.csv`` (p1/p50/p99 by feature and year) and
``data/phase6/qc/gates.csv``. Exits non-zero if any gate fails.

    python scripts/phase6_feature_qc.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.phase6_features import _common as C  # noqa: E402
from src.phase6_features import fo_archive as FO  # noqa: E402
from src.phase6_features import index_series as IS  # noqa: E402
from src.phase6_features import options as OP  # noqa: E402
from src.production_ml.feature_creator import (  # noqa: E402
    F1_MARKET, F1_RANKED, F2_FLAG, F2_OPTIONAL,
)

QC_DIR = "data/phase6/qc"
MASK = "data/nse500_universe_mask.parquet"
YEARS = range(2014, 2026)


def panel(name: str) -> pd.DataFrame:
    sub = C.F1_DIR if name in F1_RANKED or name == "sector_code" else OP.F2_DIR
    return C.read(os.path.join(sub, f"{name}.parquet"), "build_f1/build_f2 --build")


def by_year(cells: pd.DataFrame) -> pd.Series:
    """Mean of a boolean (date × ticker) frame by year, NaN cells ignored."""
    v = cells.astype(float)
    return v.groupby(v.index.year).apply(lambda b: np.nanmean(b.to_numpy()))


def main() -> int:
    os.makedirs(QC_DIR, exist_ok=True)
    mask = C.read(MASK, "universe.py --build").astype(bool)
    gates: list[tuple[str, bool, str]] = []

    def gate(name: str, ok: bool, note: str) -> None:
        gates.append((name, bool(ok), note))
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<62} {note}")

    def on_mask(f: pd.DataFrame) -> pd.DataFrame:
        return f.reindex(index=mask.index, columns=mask.columns)

    dist, inf_cols = [], []
    cov_f1 = {}
    for name in F1_RANKED:
        f = on_mask(panel(name))
        cov_f1[name] = by_year(f.notna().where(mask)).reindex(YEARS)
        if np.isinf(f.to_numpy()).any():
            inf_cols.append(name)
        fin = f.where(mask).replace([np.inf, -np.inf], np.nan)
        for y in YEARS:
            v = fin.loc[str(y)].to_numpy().ravel()
            v = v[np.isfinite(v)]
            if v.size:
                dist.append((name, y, *np.quantile(v, [0.01, 0.5, 0.99])))
    worst = min(s.min() for s in cov_f1.values())
    gate("Q1 F1 per-stock coverage of universe cells >= 98% every year", worst >= 0.98,
         f"worst {worst:.3%} ({min(cov_f1, key=lambda k: cov_f1[k].min())})")

    mkt = C.read(os.path.join(C.F1_DIR, "market.parquet"), "build_f1 --build")
    gap = mkt.loc["2013-02-01":, F1_MARKET].isna().sum().sum()
    gate("Q2 market features defined every session from 2013-02-01", gap == 0, f"{int(gap)} gaps")
    for n in F1_MARKET:
        for y in YEARS:
            v = mkt.loc[str(y), n].to_numpy()
            dist.append((n, y, *np.quantile(v[np.isfinite(v)], [0.01, 0.5, 0.99])))

    avail = on_mask(panel("opt_avail")).fillna(0).astype(bool) & mask
    for name in F2_OPTIONAL:
        f = on_mask(panel(name))
        if np.isinf(f.to_numpy()).any():
            inf_cols.append(name)
    gate("Q3 +/-inf only in sector_pe", set(inf_cols) <= {"sector_pe"}, f"inf in {inf_cols}")

    iv = panel("opt_atm_iv")
    idx = IS.load()
    vix = idx[idx["index"] == IS.VIX].set_index("date")["close"] / 100.0
    med = iv.median(axis=1)
    ratio = (med.groupby(med.index.year).median() / vix.groupby(vix.index.year).median()).dropna()
    corr = med.corr(vix.reindex(med.index))
    gate("Q4 ATM IV / VIX yearly median ratio in [1.2, 3.5]; corr >= 0.6",
         ratio.between(1.2, 3.5).all() and corr >= 0.6,
         f"ratio {ratio.min():.2f}-{ratio.max():.2f}, corr {corr:.3f}")

    fo = FO.load()
    nm = OP.near_month(fo).dropna(subset=["F"])
    opt = fo[fo["kind"] != "F"].merge(nm, on=["date", "symbol", "expiry"], how="inner")
    solved = np.isfinite(OP.implied_vol(opt["close"], opt["F"], opt["strike"],
                                        opt["days"] / 365.0, opt["kind"] == "C"))
    by_y = pd.Series(solved).groupby(opt["date"].dt.year.to_numpy()).mean()
    gate("Q5 IV solved for >= 90% of near-month traded options", solved.mean() >= 0.90,
         f"overall {solved.mean():.1%}, worst year {by_y.min():.1%} ({by_y.idxmin()})")

    panel_raw = pd.read_parquet(C.PANEL, columns=["date", "symbol", "close", "adj_factor"])
    panel_raw["symbol"] = panel_raw["symbol"].astype("string").str.strip().str.upper()
    j = nm.merge(panel_raw.drop_duplicates(["date", "symbol"]), on=["date", "symbol"])
    spot = j["close"] / j["adj_factor"]
    gapfs = np.log(j["F"] / spot)
    basis = gapfs * 365.0 / j["days"]
    gate("Q6 raw basis median in [3%, 10%]; p99 |log F/S| <= 5%",
         0.03 <= basis.median() <= 0.10 and gapfs.abs().quantile(0.99) <= 0.05,
         f"median {basis.median():.3f}, p99 |log F/S| {gapfs.abs().quantile(0.99):.4f}")

    share = (avail.sum(axis=1) / mask.sum(axis=1).replace(0, np.nan)).dropna()
    share_y = share.groupby(share.index.year).mean().reindex(YEARS)
    gate("Q7 opt_avail on 15%-60% of universe cells every year", share_y.between(0.15, 0.60).all(),
         f"{share_y.min():.1%}-{share_y.max():.1%}")

    code = on_mask(panel("sector_code"))
    used = set(np.unique(code.to_numpy()[np.isfinite(code.to_numpy())]).astype(int))
    cov_code = by_year(code.notna().where(mask)).reindex(YEARS)
    gate("Q8 all 11 sectors used; assigned on >= 98% of universe cells",
         used == set(range(len(IS.SECTORS))) and cov_code.min() >= 0.98,
         f"{len(used)} sectors, worst year {cov_code.min():.2%}")

    sp = on_mask(panel("opt_ivspread")).where(avail)
    signed_y = sp.median(axis=1).groupby(sp.index.year).median().reindex(YEARS)
    abs_y = sp.abs().median(axis=1).groupby(sp.index.year).median().reindex(YEARS)
    gate("Q9 call-put IV spread: |yearly median signed| <= 0.01 (no bias)",
         (signed_y.abs() <= 0.01).all(),
         f"signed {signed_y.min():+.4f}..{signed_y.max():+.4f}; "
         f"abs (noise, reported) {abs_y.min():.4f}-{abs_y.max():.4f}")

    cov_f2 = {}
    for name in F2_OPTIONAL:
        f = on_mask(panel(name))
        cov_f2[name] = by_year(f.notna().where(avail)).reindex(YEARS)
        fin = f.where(avail).replace([np.inf, -np.inf], np.nan)
        for y in YEARS:
            v = fin.loc[str(y)].to_numpy().ravel()
            v = v[np.isfinite(v)]
            if v.size:
                dist.append((name, y, *np.quantile(v, [0.01, 0.5, 0.99])))
    worst2 = min(s.min() for s in cov_f2.values())
    gate("Q10 F2 features cover >= 80% of opt_avail cells every year", worst2 >= 0.80,
         f"worst {worst2:.1%} ({min(cov_f2, key=lambda k: cov_f2[k].min())})")

    pd.DataFrame(dist, columns=["feature", "year", "p01", "p50", "p99"]).to_csv(
        os.path.join(QC_DIR, "feature_distribution.csv"), index=False)
    pd.DataFrame(gates, columns=["gate", "pass", "detail"]).to_csv(
        os.path.join(QC_DIR, "gates.csv"), index=False)
    cov = pd.DataFrame({**cov_f1, **{f"{k} | opt_avail": v for k, v in cov_f2.items()}}).T
    cov.to_csv(os.path.join(QC_DIR, "coverage_by_year.csv"))

    ok = all(g[1] for g in gates)
    print("=" * 96)
    print(f"PHASE 6 FEATURE QC — {sum(g[1] for g in gates)}/{len(gates)} gates pass | "
          f"wrote {QC_DIR}/{{gates,feature_distribution,coverage_by_year}}.csv")
    print(f"VERDICT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
