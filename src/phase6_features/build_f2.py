"""
Phase 6 Part B — build every F2 panel in order, then report coverage.

F2 is optional per name (§B.2): about 60% of the universe has no options on a given day.
Coverage is therefore reported twice: over all universe cells, and over the cells where
``opt_avail`` = 1. No return is read; quality beyond coverage is Stage 6.

    python src/phase6_features/build_f2.py            # every module's dry-run
    python src/phase6_features/build_f2.py --build    # fo_archive -> options -> futures
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
from src.phase6_features import fo_archive, futures, options  # noqa: E402
from src.phase6_features.build_f1 import UNIVERSE_MASK  # noqa: E402

MODULES = (fo_archive, options, futures)
COVERAGE_OUT = os.path.join(options.F2_DIR, "coverage_by_year.csv")


def coverage() -> pd.DataFrame:
    mask = C.read(UNIVERSE_MASK, "python src/phase4_data/universe.py --build").astype(bool)
    avail = C.read(os.path.join(options.F2_DIR, "opt_avail.parquet"), "build_f2.py --build")
    avail = avail.reindex(index=mask.index, columns=mask.columns).fillna(0).astype(bool) & mask
    rows = {"opt_avail share of universe": avail.sum(axis=1) / mask.sum(axis=1).replace(0, np.nan)}
    for name in (*options.FEATURES, *futures.FEATURES):
        if name == "opt_avail":
            continue
        f = C.read(os.path.join(options.F2_DIR, f"{name}.parquet"), "build_f2.py --build")
        f = f.reindex(index=mask.index, columns=mask.columns)
        rows[f"{name} | where opt_avail"] = f.notna().where(avail).mean(axis=1)
    table = pd.DataFrame(rows)
    return table.groupby(table.index.year).mean().T


def main() -> None:
    ap = argparse.ArgumentParser(description="Build every Phase 6 F2 panel.")
    ap.add_argument("--build", action="store_true", help="build (else every dry-run)")
    args = ap.parse_args()
    if not args.build:
        for m in MODULES:
            m._dry_run()
        print("\n[build_f2] every F2 dry-run OK")
        return
    for m in MODULES:
        m.build()
    table = coverage()
    table.columns = [str(c) for c in table.columns]
    table.to_csv(COVERAGE_OUT)
    print("\n[build_f2] coverage by year:")
    print(table.round(3).to_string())
    print(f"[build_f2] wrote {COVERAGE_OUT}")


if __name__ == "__main__":
    main()
