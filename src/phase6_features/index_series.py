"""
Phase 6 Part B — the index series F1 needs, from NSE's daily index archive.

Reads every ``data/bhavcopy/index/ind_close_all_<DDMMYYYY>.csv`` dated on or before the
sealed holdout boundary. Files are selected by **filename** date before any file is
opened, so a sealed file is never read. The date comes from the filename, never the
file's own date column, which changes format mid-archive (see ``dividends.py``).

Kept: the Nifty 50, the Nifty 500, India VIX, and the 11 sectoral indexes present
continuously since 2013 (§B.2). NSE renamed the CNX indexes to Nifty on 2015-11-09, and the
two broad indexes were also renamed in Feb 2013. A fixed alias map joins each into one
series. Every kept index covers all 3,082 archive sessions under exactly one name per day.
The daily log return across each rename is an ordinary move, so the level is continuous.

Output: ``data/phase6/index_daily.parquet``, long ``(date, index, close, pe, pb, dy)``,
raw. ``"-"`` and blanks read as NaN. Calendar alignment and gap filling happen in
``wide()`` / ``fill_valuation()``, so the persisted file is exactly what NSE published.

    python src/phase6_features/index_series.py            # dry-run self-test
    python src/phase6_features/index_series.py --build
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import tempfile
from typing import Final

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.phase6_features import _common as C  # noqa: E402

NIFTY50: Final[str] = "NIFTY 50"
NIFTY500: Final[str] = "NIFTY 500"
VIX: Final[str] = "INDIA VIX"
# §B.2: the 11 sectoral indexes continuous since 2013. Canonical (post-2015) names.
SECTORS: Final[tuple[str, ...]] = (
    "NIFTY AUTO", "NIFTY BANK", "NIFTY ENERGY", "NIFTY FINANCIAL SERVICES", "NIFTY FMCG",
    "NIFTY IT", "NIFTY MEDIA", "NIFTY METAL", "NIFTY PHARMA", "NIFTY PSU BANK",
    "NIFTY REALTY",
)
_ALIASES: Final[dict[str, str]] = {
    "S&P CNX NIFTY": NIFTY50, "CNX NIFTY": NIFTY50,
    "S&P CNX 500": NIFTY500, "CNX 500": NIFTY500,
    "CNX FINANCE": "NIFTY FINANCIAL SERVICES",
    **{f"CNX {s}": f"NIFTY {s}" for s in
       ("AUTO", "BANK", "ENERGY", "FMCG", "IT", "MEDIA", "METAL", "PHARMA", "PSU BANK",
        "REALTY")},
}
KEPT: Final[tuple[str, ...]] = (NIFTY50, NIFTY500, VIX, *SECTORS)
_COLS: Final[dict[str, str]] = {
    "Closing Index Value": "close", "P/E": "pe", "P/B": "pb", "Div Yield": "dy",
}
_NAME_RE: Final[re.Pattern] = re.compile(r"^ind_close_all_(\d{8})\.csv$", re.IGNORECASE)

# §B.8 (2026-10-14): a valuation ratio missing for up to this many sessions is a
# publication pause (e.g. the FY-end gap of 2021-03-31 → 2021-06-02) and is carried
# forward. A P/E missing for longer is NSE's blank for non-positive aggregate earnings
# (PSU Bank 2018-02 → 2021-01, Media 2022-09 → 2025-05) and becomes +inf: above every
# finite P/E, which is the earnings-yield ordering. Causal: a run is only treated as a
# loss once it has lasted past the limit.
PUBLICATION_GAP_LIMIT: Final[int] = 63


def canonical(name: str) -> str:
    n = str(name).strip().upper()
    return _ALIASES.get(n, n)


def build_long(raw_dir: str = C.INDEX_RAW_DIR, allow_holdout: bool = False) -> pd.DataFrame:
    """Parse the archive into the raw long table. Sealed files are skipped unopened."""
    dated: list[tuple[pd.Timestamp, str]] = []
    for path in glob.glob(os.path.join(raw_dir, "*.csv")):
        m = _NAME_RE.match(os.path.basename(path))
        if m:
            dated.append((pd.to_datetime(m.group(1), format="%d%m%Y"), path))
    if not dated:
        raise FileNotFoundError(f"No index CSVs under {raw_dir}.")
    n_sealed = sum(d > C.sealed_after() for d, _ in dated)
    if not allow_holdout:
        dated = [(d, p) for d, p in dated if d <= C.sealed_after()]

    frames = []
    for date, path in sorted(dated):
        df = pd.read_csv(path)
        df["index"] = df["Index Name"].map(canonical)
        df = df[df["index"].isin(KEPT)]
        out = df[["index"]].copy()
        for src, dst in _COLS.items():
            out[dst] = pd.to_numeric(df[src], errors="coerce") if src in df else np.nan
        out.insert(0, "date", date)
        frames.append(out)
    long = pd.concat(frames, ignore_index=True)

    dup = long.duplicated(["date", "index"])
    if dup.any():
        raise AssertionError(f"{int(dup.sum())} (date, index) rows appear under two aliases")
    print(
        f"[index_series] {long['date'].nunique():,} sessions "
        f"{long['date'].min().date()} → {long['date'].max().date()} | "
        f"{long['index'].nunique()} indexes | sealed files skipped unread: {n_sealed}"
    )
    return long


def wide(long: pd.DataFrame, field: str, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """
    One field as (calendar × index), unfilled. Sessions the archive lacks are NaN here;
    ``close_on_calendar`` and ``fill_valuation`` decide how each field fills them.
    """
    w = long.pivot(index="date", columns="index", values=field).sort_index()
    return w.reindex(calendar)


def close_on_calendar(long: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Index closes on the price panel's calendar. The 12 panel sessions with no index file
    carry the previous close forward (§B.8): causal, and the move lands on the next
    published session.
    """
    return wide(long, "close", calendar).ffill()


def fill_valuation(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    """
    Carry a valuation ratio over publication gaps (≤ ``PUBLICATION_GAP_LIMIT`` sessions).
    A P/E still missing after that is non-positive earnings: +inf. Leading NaN (before
    the first value) stays NaN.
    """
    filled = frame.ffill(limit=PUBLICATION_GAP_LIMIT)
    if field == "pe":
        seen = frame.notna().cummax()
        filled = filled.mask(seen & filled.isna(), np.inf)
    return filled


def build(raw_dir: str = C.INDEX_RAW_DIR, out_path: str = C.INDEX_DAILY,
          allow_holdout: bool = False) -> pd.DataFrame:
    long = build_long(raw_dir, allow_holdout)
    C.guard_holdout(long["date"], "index archive", allow_holdout)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    long.to_parquet(out_path, index=False)
    print(f"[index_series] wrote {out_path} | {len(long):,} rows")
    return long


def load(path: str = C.INDEX_DAILY) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run `python src/phase6_features/index_series.py --build` first."
        )
    long = pd.read_parquet(path)
    long["date"] = pd.DatetimeIndex(long["date"]).normalize()
    return long


# ── Dry run ─────────────────────────────────────────────────────────────────────

def _dry_run() -> None:
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as tmp:
        dates = pd.bdate_range("2015-11-02", periods=8)
        sealed = C.sealed_after() + pd.Timedelta(days=3)
        for i, d in enumerate([*dates, sealed]):
            renamed = d >= pd.Timestamp("2015-11-09")
            rows = [
                ("NIFTY 50" if renamed else "CNX NIFTY", 8000 + i, 22.0, 3.2, 1.3),
                ("NIFTY BANK" if renamed else "CNX BANK", 17000 + i, "-" if i in (3, 4) else 18.0, 2.5, 1.1),
                ("NIFTY MNC", 9000, 30.0, 6.0, 1.0),          # not kept
            ]
            pd.DataFrame(rows, columns=["Index Name", "Closing Index Value", "P/E", "P/B", "Div Yield"]) \
                .to_csv(os.path.join(tmp, f"ind_close_all_{d.strftime('%d%m%Y')}.csv"), index=False)
        os.remove(os.path.join(tmp, f"ind_close_all_{dates[6].strftime('%d%m%Y')}.csv"))  # a missing session

        long = build_long(tmp)
        checks["sealed file skipped"] = bool(long["date"].max() <= C.sealed_after())
        checks["renames join into one series"] = bool(
            set(long["index"]) == {NIFTY50, "NIFTY BANK"}
            and (long.groupby("index").size() == 7).all()
        )
        checks["'-' reads as NaN"] = bool(long.loc[long["index"] == "NIFTY BANK", "pe"].isna().sum() == 2)
        close = close_on_calendar(long, dates)
        checks["missing session carries previous close"] = bool(
            close.loc[dates[6], NIFTY50] == close.loc[dates[5], NIFTY50]
        )

    # Valuation fill: short gap carried, long gap -> +inf only after the limit, causal.
    cal = pd.bdate_range("2018-01-01", periods=200)
    pe = pd.DataFrame({"A": 20.0, "B": 15.0}, index=cal)
    pe.iloc[10:20, 0] = np.nan                         # 10-session pause
    pe.iloc[50:180, 1] = np.nan                        # 130-session loss run
    f = fill_valuation(pe, "pe")
    checks["short gap forward-filled"] = bool((f.iloc[10:20, 0] == 20.0).all())
    lim = PUBLICATION_GAP_LIMIT
    checks["long gap carried up to the limit"] = bool((f.iloc[50:50 + lim, 1] == 15.0).all())
    checks["long gap is +inf after the limit"] = bool(np.isinf(f.iloc[50 + lim:180, 1]).all())
    checks["value returns when published again"] = bool((f.iloc[180:, 1] == 15.0).all())
    f_trunc = fill_valuation(pe.iloc[:60], "pe")
    checks["fill is causal (truncation-invariant)"] = bool(f_trunc.equals(f.iloc[:60]))
    pb = fill_valuation(pe, "pb")
    checks["only P/E becomes +inf"] = bool(not np.isinf(pb.to_numpy()).any())

    print()
    for k, v in checks.items():
        print(f"  {k:<44}: {'PASS' if v else 'FAIL'}")
    assert all(checks.values()), "index_series dry-run FAILED"
    print("\n[index_series] dry-run OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse the index archive for Phase 6 F1.")
    ap.add_argument("--build", action="store_true", help="build (else dry-run)")
    args = ap.parse_args()
    if not args.build:
        _dry_run()
        return
    build()


if __name__ == "__main__":
    main()
