"""
Phase 6 Part B, F2 — condense the F&O archive into one long table of stock derivatives.

Reads every session archive on or before the sealed holdout boundary, in both eras:

    legacy  data/bhavcopy/fo/foDDMONYYYYbhav.csv.zip      (2013-01 → 2024-06)
    UDiFF   data/bhavcopy/fo_udiff/BhavCopy_NSE_FO_..zip  (2024-07 → 2025-06)

and keeps, in one schema:

    every stock-futures row      (legacy FUTSTK / UDiFF STF), traded or not, for open
                                 interest and the forward;
    every **traded** stock-option row (OPTSTK / STO with contracts > 0). An untraded
                                 strike carries NSE's theoretical settlement price, not
                                 a trade (§B.2), so it is dropped here.

Columns: date, symbol, kind (F / C / P), expiry, strike, close, contracts, value
(rupees), oi (shares). Two unit facts, checked against the files:

- ``value`` is NSE's reported traded value. For options that is **notional**,
  (strike + premium) × quantity, in both eras: legacy ``VAL_INLAKH`` × 1e5, UDiFF
  ``TtlTrfVal``.
- ``contracts`` is a contract count in both eras (legacy ``CONTRACTS``, UDiFF
  ``TtlTradgVol``), and ``oi`` is in shares (``OPEN_INT`` / ``OpnIntrst``).

Dates come from filenames (``shortable.date_from_filename``), and sealed files are
skipped unopened.

    python src/phase6_features/fo_archive.py            # dry-run self-test
    python src/phase6_features/fo_archive.py --build    # ~10 min, 4 processes
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.phase4_data.shortable import date_from_filename  # noqa: E402
from src.phase6_features import _common as C  # noqa: E402

RAW_DIR: Final[str] = "data/bhavcopy"
OUT: Final[str] = os.path.join(C.OUT_DIR, "fo_stock_long.parquet")
SCHEMA: Final[list[str]] = [
    "date", "symbol", "kind", "expiry", "strike", "close", "contracts", "value", "oi",
]
N_PROC: Final[int] = 4     # leaves headroom on the 8-core laptop (docs: machine limits)


def _legacy(df: pd.DataFrame) -> pd.DataFrame:
    inst = df["INSTRUMENT"].astype(str).str.strip().str.upper()
    df = df[inst.isin(["FUTSTK", "OPTSTK"])]
    kind = np.where(df["INSTRUMENT"].str.strip().str.upper() == "FUTSTK", "F",
                    df["OPTION_TYP"].astype(str).str.strip().str.upper().str[0])
    return pd.DataFrame({
        "symbol": df["SYMBOL"].astype(str).str.strip().str.upper(),
        "kind": kind,
        "expiry": pd.to_datetime(df["EXPIRY_DT"].astype(str).str.strip(), format="%d-%b-%Y"),
        "strike": pd.to_numeric(df["STRIKE_PR"], errors="coerce"),
        "close": pd.to_numeric(df["CLOSE"], errors="coerce"),
        "contracts": pd.to_numeric(df["CONTRACTS"], errors="coerce"),
        "value": pd.to_numeric(df["VAL_INLAKH"], errors="coerce") * 1e5,
        "oi": pd.to_numeric(df["OPEN_INT"], errors="coerce"),
    })


def _udiff(df: pd.DataFrame) -> pd.DataFrame:
    inst = df["FinInstrmTp"].astype(str).str.strip().str.upper()
    df = df[inst.isin(["STF", "STO"])]
    kind = np.where(df["FinInstrmTp"].str.strip().str.upper() == "STF", "F",
                    df["OptnTp"].astype(str).str.strip().str.upper().str[0])
    return pd.DataFrame({
        "symbol": df["TckrSymb"].astype(str).str.strip().str.upper(),
        "kind": kind,
        "expiry": pd.to_datetime(df["XpryDt"].astype(str).str.strip(), format="%Y-%m-%d"),
        "strike": pd.to_numeric(df["StrkPric"], errors="coerce"),
        "close": pd.to_numeric(df["ClsPric"], errors="coerce"),
        "contracts": pd.to_numeric(df["TtlTradgVol"], errors="coerce"),
        "value": pd.to_numeric(df["TtlTrfVal"], errors="coerce"),
        "oi": pd.to_numeric(df["OpnIntrst"], errors="coerce"),
    })


def parse_one(path: str) -> pd.DataFrame | None:
    """One session's stock-derivative rows in the common schema, or None if unreadable."""
    date = date_from_filename(path)
    if date is None:
        return None
    try:
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            with z.open(names[0]) as fh:
                df = pd.read_csv(io.BytesIO(fh.read()), low_memory=False)
    except (OSError, IndexError, zipfile.BadZipFile, pd.errors.ParserError, UnicodeDecodeError):
        return None
    df.columns = [c.strip() for c in df.columns]
    if "INSTRUMENT" in df.columns:
        out = _legacy(df)
    elif "FinInstrmTp" in df.columns:
        out = _udiff(df)
    else:
        return None
    out = out[(out["kind"] == "F") | (out["contracts"] > 0)]
    out.insert(0, "date", date)
    return out[SCHEMA]


def archive_paths(raw_dir: str = RAW_DIR, allow_holdout: bool = False) -> tuple[list[str], int]:
    paths = glob.glob(os.path.join(raw_dir, "fo", "*.zip")) + glob.glob(
        os.path.join(raw_dir, "fo_udiff", "*.zip"))
    dated = [(date_from_filename(p), p) for p in paths]
    dated = [(d, p) for d, p in dated if d is not None]
    n_sealed = sum(d > C.sealed_after() for d, _ in dated)
    if not allow_holdout:
        dated = [(d, p) for d, p in dated if d <= C.sealed_after()]
    return [p for _, p in sorted(dated)], n_sealed


def build(raw_dir: str = RAW_DIR, out_path: str = OUT, allow_holdout: bool = False,
          n_proc: int = N_PROC) -> pd.DataFrame:
    paths, n_sealed = archive_paths(raw_dir, allow_holdout)
    frames, bad = [], []
    with ProcessPoolExecutor(max_workers=n_proc) as ex:
        for i, (p, f) in enumerate(zip(paths, ex.map(parse_one, paths, chunksize=16)), 1):
            (bad.append(p) if f is None or f.empty else frames.append(f))
            if i % 500 == 0 or i == len(paths):
                print(f"[fo_archive] parsed {i}/{len(paths)}")
    long = pd.concat(frames, ignore_index=True)
    long["kind"] = long["kind"].astype("category")
    C.guard_holdout(long["date"], "F&O archive", allow_holdout)
    dup = long.duplicated(["date", "symbol", "kind", "expiry", "strike"])
    if dup.any():
        raise AssertionError(f"{int(dup.sum())} duplicate contract rows")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    long.to_parquet(out_path, index=False)
    per_day = long[long["kind"] != "F"].groupby("date")["symbol"].nunique()
    print(
        f"[fo_archive] {len(paths):,} sessions {long['date'].min().date()} → "
        f"{long['date'].max().date()} | unreadable {len(bad)} | sealed skipped unread {n_sealed}\n"
        f"[fo_archive] {len(long):,} rows ({int((long['kind'] == 'F').sum()):,} futures, "
        f"{int((long['kind'] != 'F').sum()):,} traded options) | "
        f"underlyings with a traded option per day: median {per_day.median():.0f}, "
        f"range {per_day.min()}–{per_day.max()}"
    )
    print(f"[fo_archive] wrote {out_path}")
    return long


def load(path: str = OUT) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run `python src/phase6_features/fo_archive.py --build` first."
        )
    return pd.read_parquet(path)


# ── Dry run ─────────────────────────────────────────────────────────────────────

def _dry_run() -> None:
    checks: dict[str, bool] = {}
    legacy = pd.DataFrame({
        "INSTRUMENT": ["FUTSTK", "FUTSTK", "OPTSTK", "OPTSTK", "OPTIDX", "OPTSTK"],
        "SYMBOL": ["INFY", "INFY", "INFY", "INFY", "NIFTY", "INFY"],
        "EXPIRY_DT": ["28-Jan-2016", "25-Feb-2016", "28-Jan-2016", "28-Jan-2016", "28-Jan-2016", "28-Jan-2016"],
        "STRIKE_PR": [0, 0, 900, 1100, 8000, 1200],
        "OPTION_TYP": ["XX", "XX", "CE", "PE", "CE", "CE"],
        "OPEN": 0, "HIGH": 0, "LOW": 0,
        "CLOSE": [1085.5, 1090.1, 192.0, 20.0, 50.0, 3.0],
        "SETTLE_PR": 0,
        "CONTRACTS": [16174, 0, 21, 5, 100, 0],
        "VAL_INLAKH": [86997.8, 0.0, 113.28, 30.0, 400.0, 0.0],
        "OPEN_INT": [12478000, 139000, 10500, 2500, 0, 0],
        "CHG_IN_OI": 0, "TIMESTAMP": "13-JAN-2016",
    })
    udiff = pd.DataFrame({
        "FinInstrmTp": ["STF", "STO", "IDO"], "TckrSymb": ["INFY", "INFY", "NIFTY"],
        "XpryDt": ["2024-07-25"] * 3, "StrkPric": [np.nan, 1740.0, 24000.0],
        "OptnTp": [np.nan, "CE", "CE"], "ClsPric": [1665.0, 13.05, 100.0],
        "TtlTradgVol": [5000, 3619, 10], "TtlTrfVal": [3.3e9, 2537147500.0, 1e6],
        "OpnIntrst": [9e6, 251600, 0],
    })
    with tempfile.TemporaryDirectory() as tmp:
        for sub in ("fo", "fo_udiff"):
            os.makedirs(os.path.join(tmp, sub))
        files = {
            os.path.join(tmp, "fo", "fo13JAN2016bhav.csv.zip"): legacy,
            os.path.join(tmp, "fo_udiff", "BhavCopy_NSE_FO_0_0_0_20240708_F_0000.csv.zip"): udiff,
            os.path.join(tmp, "fo_udiff", "BhavCopy_NSE_FO_0_0_0_20250801_F_0000.csv.zip"): udiff,
        }
        for path, df in files.items():
            with zipfile.ZipFile(path, "w") as z:
                z.writestr(os.path.basename(path)[:-4], df.to_csv(index=False))
        paths, n_sealed = archive_paths(tmp)
        checks["sealed archive skipped unopened"] = bool(len(paths) == 2 and n_sealed == 1)
        a = parse_one(paths[0])
        b = parse_one(paths[1])

    checks["legacy: index options dropped"] = bool((a["symbol"] == "INFY").all())
    checks["legacy: all futures kept, untraded options dropped"] = bool(
        (a["kind"] == "F").sum() == 2 and set(a.loc[a["kind"] != "F", "strike"]) == {900, 1100}
    )
    checks["legacy: value in rupees"] = bool(np.isclose(a.loc[a["strike"] == 900, "value"].iloc[0], 113.28e5))
    checks["legacy: CE/PE -> C/P, expiry parsed"] = bool(
        set(a["kind"]) == {"F", "C", "P"} and a["expiry"].iloc[0] == pd.Timestamp("2016-01-28")
    )
    checks["udiff: stock rows only, same schema"] = bool(
        list(b.columns) == SCHEMA and set(b["kind"]) == {"F", "C"} and (b["symbol"] == "INFY").all()
    )
    # UDiFF value is notional: contracts × lot (400) × (strike + premium).
    opt = b[b["kind"] == "C"].iloc[0]
    checks["udiff: value is notional (strike + premium)"] = bool(
        np.isclose(opt["value"], opt["contracts"] * 400 * (opt["strike"] + 12.6), rtol=1e-3)
    )
    print()
    for k, v in checks.items():
        print(f"  {k:<48}: {'PASS' if v else 'FAIL'}")
    assert all(checks.values()), "fo_archive dry-run FAILED"
    print("\n[fo_archive] dry-run OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Condense the F&O archive for Phase 6 F2.")
    ap.add_argument("--build", action="store_true", help="build (else dry-run)")
    args = ap.parse_args()
    if not args.build:
        _dry_run()
        return
    build()


if __name__ == "__main__":
    main()
