#!/usr/bin/env python3
"""
Regression gate — a candidate ledger must reproduce its baseline bit-for-bit.

Every opt-in addition (Phase 4c, Phase 5) is judged by one claim: with its flag off, the
prior results come back *exactly*. "Close" is not a pass. A max abs diff of 1e-12 means the
arithmetic changed somewhere, and a later difference can then no longer be attributed to
the change under test.

Compares Parquet ledgers (bar x strategy_id) and CSV side-files (DSR gate, execution
diagnostics). Pass = identical index, identical column set, identical NaN pattern, and
max abs diff == 0.0 on every numeric column. Non-numeric columns must be equal as strings.

    python scripts/regression_check.py BASELINE CANDIDATE [BASELINE CANDIDATE ...]
    python scripts/regression_check.py --ignore-id-hash BASELINE CANDIDATE

``--ignore-id-hash`` is for baselines written before Phase 4c hashed ``target_col`` into
``strategy_id`` (commit 9f3fac3): the ids gained a ``_T5B`` token and a new hash while the
numbers stayed the same. It pairs columns on the id with the hash and target token removed
(``STRAT_LIVE_NSE_LO_L0_HMM2``), refuses an ambiguous pairing, and prints every pair — the
values must still match bit-for-bit.

Exits non-zero on any mismatch, so it can gate a script.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported ledger type: {path}")


_ID_TAIL = re.compile(r"(_T\d+B)?_[0-9a-f]{8}$")


def _strip_id(name: str) -> str:
    """``STRAT_LIVE_NSE_LO_L0_HMM2_T5B_9531394c`` -> ``STRAT_LIVE_NSE_LO_L0_HMM2``."""
    return _ID_TAIL.sub("", str(name))


def _rename_stripped(frame: pd.DataFrame, label: str) -> pd.DataFrame | None:
    stripped = [_strip_id(c) for c in frame.columns]
    if len(set(stripped)) != len(stripped):
        print(f"  {label}: ids collide once the hash is removed — pairing would be ambiguous")
        return None
    return frame.set_axis(stripped, axis=1)


def compare(baseline: str | Path, candidate: str | Path, ignore_id_hash: bool = False) -> bool:
    """Print a per-column verdict for one pair; return True only on an exact reproduction."""
    b_path, c_path = Path(baseline), Path(candidate)
    print(f"\n[regression] {b_path}  vs  {c_path}")
    base, cand = _read(b_path), _read(c_path)

    # A CSV side-file has no meaningful row order guarantee beyond its key; align on
    # strategy_id when present so a reordered-but-identical file still passes.
    if "strategy_id" in base.columns and "strategy_id" in cand.columns:
        if ignore_id_hash:
            base["strategy_id"] = base["strategy_id"].map(_strip_id)
            cand["strategy_id"] = cand["strategy_id"].map(_strip_id)
        base = base.set_index("strategy_id").sort_index()
        cand = cand.set_index("strategy_id").sort_index()
    elif ignore_id_hash:
        pairs = dict(zip(map(_strip_id, cand.columns), cand.columns))
        for col in base.columns:
            print(f"  pair  {col}  <->  {pairs.get(_strip_id(col), '(none)')}")
        base, cand = _rename_stripped(base, "baseline"), _rename_stripped(cand, "candidate")
        if base is None or cand is None:
            return False

    ok = True
    missing = sorted(set(base.columns) - set(cand.columns))
    extra = sorted(set(cand.columns) - set(base.columns))
    if missing or extra:
        print(f"  columns differ — missing {missing}, extra {extra}")
        ok = False
    if not base.index.equals(cand.index):
        print(f"  index differs — baseline {len(base.index)} rows, candidate {len(cand.index)} rows")
        return False

    for col in [c for c in base.columns if c in cand.columns]:
        b, c = base[col], cand[col]
        if pd.api.types.is_numeric_dtype(b) and pd.api.types.is_numeric_dtype(c):
            b_arr, c_arr = b.to_numpy(float), c.to_numpy(float)
            nan_match = bool(np.array_equal(np.isnan(b_arr), np.isnan(c_arr)))
            both = ~np.isnan(b_arr) & ~np.isnan(c_arr)
            diff = float(np.max(np.abs(b_arr[both] - c_arr[both]))) if both.any() else 0.0
            passed = nan_match and diff == 0.0
            detail = f"max abs diff = {diff:.3g}" + ("" if nan_match else ", NaN pattern differs")
        else:
            passed = bool((b.astype(str) == c.astype(str)).all())
            detail = "equal" if passed else "values differ"
        print(f"  {'PASS' if passed else 'FAIL'}  {str(col)[:48]:<48} {detail}")
        ok &= passed

    print(f"  → {'BIT-IDENTICAL' if ok else 'MISMATCH'}")
    return ok


def main(argv: list[str]) -> int:
    ignore_id_hash = "--ignore-id-hash" in argv
    argv = [a for a in argv if a != "--ignore-id-hash"]
    if len(argv) < 2 or len(argv) % 2:
        print(__doc__)
        return 2
    pairs = list(zip(argv[0::2], argv[1::2]))
    results = [compare(b, c, ignore_id_hash) for b, c in pairs]
    print(f"\n[regression] {sum(results)}/{len(results)} pairs bit-identical")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
