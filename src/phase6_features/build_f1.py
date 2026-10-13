"""
Phase 6 Part B — build every F1 panel in order, then report coverage.

Coverage only: the share of universe cells (``nse500_universe_mask.parquet``) where each
feature is defined, by year. No return is read. Feature quality beyond coverage is
Stage 6 (``scripts/phase6_feature_qc.py``).

    python src/phase6_features/build_f1.py            # every module's dry-run
    python src/phase6_features/build_f1.py --build    # build all F1 panels + coverage
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.phase6_features import _common as C  # noqa: E402
from src.phase6_features import index_series, market, price_features, sector  # noqa: E402

MODULES = (index_series, price_features, sector, market)
UNIVERSE_MASK = "data/nse500_universe_mask.parquet"
COVERAGE_OUT = os.path.join(C.F1_DIR, "coverage_by_year.csv")


def coverage() -> pd.DataFrame:
    mask = C.read(UNIVERSE_MASK, "python src/phase4_data/universe.py --build").astype(bool)
    rows = {}
    for name in (*price_features.FEATURES, *sector.FEATURES):
        f = C.read(os.path.join(C.F1_DIR, f"{name}.parquet"), "build_f1.py --build")
        f = f.reindex(index=mask.index, columns=mask.columns)
        defined = f.notna().where(mask)
        rows[name] = defined.groupby(defined.index.year).apply(lambda b: np.nanmean(b.to_numpy(float)))
    mkt = C.read(market.OUT, "build_f1.py --build")
    for name in market.FEATURES:
        s = mkt[name].reindex(mask.index).where(mask.any(axis=1))
        rows[name] = s.notna().groupby(s.index.year).mean()
    table = pd.DataFrame(rows).T
    table.columns = [str(c) for c in table.columns]

    pe = C.read(os.path.join(C.F1_DIR, "sector_pe.parquet"), "build_f1.py --build")
    pe = pe.reindex(index=mask.index, columns=mask.columns).where(mask)
    inf_share = np.isinf(pe).sum(axis=1) / mask.sum(axis=1).replace(0, np.nan)
    print(f"[build_f1] sector_pe = +inf (loss-making sector) on {inf_share.mean():.1%} of "
          f"universe cells on average, max {inf_share.max():.1%}")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description="Build every Phase 6 F1 panel.")
    ap.add_argument("--build", action="store_true", help="build (else every dry-run)")
    args = ap.parse_args()
    if not args.build:
        for m in MODULES:
            m._dry_run()
        print("\n[build_f1] every F1 dry-run OK")
        return
    for m in MODULES:
        m.build()
    table = coverage()
    os.makedirs(C.F1_DIR, exist_ok=True)
    table.to_csv(COVERAGE_OUT)
    print("\n[build_f1] share of universe cells defined, by year:")
    print(table.round(3).to_string())
    print(f"[build_f1] wrote {COVERAGE_OUT}")


if __name__ == "__main__":
    main()
