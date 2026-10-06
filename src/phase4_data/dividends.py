"""
Phase 4c E5 — the dividend guard, measured instead of assumed.

Phase 4b's `long_only` verdict was decided by a number nobody measured. The backtest
runs on **price** returns, so a net-long book's dividend income is missing; §3.4 of the
Phase 4b results added a flat **assumed** 1.3%/yr and the cell moved from DSR 0.916 to
0.9496 — a fail by 4 x 10^-4, against a 0.95 threshold. A verdict that turns on an
assumed constant is not a verdict.

The constant is unnecessary: the index archive already carries the number. Every
`ind_close_all_<DDMMYYYY>.csv` has a `Div Yield` column for each index, so the NIFTY-50
trailing dividend yield is available **per session** across the whole window. This
module extracts it.

Two deliberate limits, stated rather than buried:

  - It is an **index-level** yield applied to the book's net exposure, not a per-name
    dividend. The strategy's long book is not the index, and a book tilted toward
    high-yield names would earn more (toward growth names, less). Correcting that needs
    per-name dividend data the bhavcopy does not carry.
  - It is applied to **net** exposure, so a dollar-neutral book earns approximately
    nothing (long dividends received, short dividends paid) — which is right — and a
    book sitting in cash during a Panic earns nothing, which is also right. This is
    strictly better than Phase 4b's uniform additive constant, which credited the
    dividend even on days the strategy held no equity at all.

The accrual is deliberately **not** folded into the ledger. Keeping the ledger on price
returns keeps it comparable with Phase 3 and Phase 4b; the dividend-adjusted figure is
reported next to the headline as a stated assumption, which is what the plan asks for.

Usage
    python src/phase4_data/dividends.py            # dry-run self-test
    python src/phase4_data/dividends.py --build    # build from data/bhavcopy/index/
"""

import argparse
import glob
import os
import re
import sys
from typing import Final

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_RAW_DIR: Final[str] = "data/bhavcopy"
DEFAULT_OUT: Final[str] = "data/bhavcopy/div_yield_daily.parquet"

# The benchmark renamed itself twice over the window; all three labels are one series.
# Kept in step with bhavcopy_panel._NIFTY_ALIASES.
_NIFTY_ALIASES: Final[tuple[str, ...]] = ("S&P CNX NIFTY", "CNX NIFTY", "NIFTY 50")
_YIELD_COL: Final[str] = "Div Yield"
_INDEX_NAME_RE: Final[re.Pattern] = re.compile(
    r"^ind_close_all_(\d{8})\.csv$", re.IGNORECASE
)
# A trailing index dividend yield outside this band is a parse artefact, not a market
# event — NIFTY-50 has sat between ~0.9% and ~1.8% for the whole window.
_SANE_BAND: Final[tuple[float, float]] = (0.1, 5.0)


def build_series(raw_dir: str = DEFAULT_RAW_DIR, limit: int = 0) -> pd.Series:
    """
    Daily NIFTY-50 trailing dividend yield (percent per year) from the index archive.

    Dates come from the **filename**, never the file's own date column: that column
    changes format mid-archive and a strict parse silently dropped 62 days when the
    panel builder trusted it.
    """
    paths = sorted(glob.glob(os.path.join(raw_dir, "index", "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No index CSVs under {raw_dir}/index.")
    if limit > 0:
        paths = paths[:limit]

    rows: list[tuple[pd.Timestamp, float]] = []
    n_bad = 0
    for path in paths:
        m = _INDEX_NAME_RE.match(os.path.basename(path))
        if m is None:
            n_bad += 1
            continue
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError):
            n_bad += 1
            continue
        if "Index Name" not in df.columns or _YIELD_COL not in df.columns:
            n_bad += 1
            continue
        name = df["Index Name"].astype("string").str.strip().str.upper()
        hit = df[name.isin(_NIFTY_ALIASES)]
        if hit.empty:
            n_bad += 1
            continue
        y = pd.to_numeric(hit.iloc[0][_YIELD_COL], errors="coerce")
        if pd.isna(y):
            n_bad += 1
            continue
        rows.append((pd.to_datetime(m.group(1), format="%d%m%Y"), float(y)))

    s = (
        pd.Series(dict(rows), name="div_yield_pct")
        .sort_index()
        .pipe(lambda x: x[x.between(*_SANE_BAND)])
    )
    s.index.name = "date"
    print(
        f"[dividends] {len(s):,} sessions {s.index.min().date()} → {s.index.max().date()} "
        f"| yield mean={s.mean():.3f}%  min={s.min():.2f}%  max={s.max():.2f}% "
        f"| unusable files={n_bad}"
    )
    return s


def accrual(
    net_exposure: pd.Series, yields: pd.Series, bars_per_year: float = 252.0
) -> pd.Series:
    """
    Per-bar dividend income for a book, given its net exposure.

        accrual_t = net_exposure_t · (yield_t / 100) / bars_per_year

    The yield is reindexed onto the return series' bars and forward-filled (causal —
    ``ffill`` only), then gaps at the very start fall back to the series median. A
    dollar-neutral book has net exposure ~0 and therefore earns ~0; a book in cash
    earns 0. Both are correct, and neither was true of a flat additive constant.
    """
    y = yields.reindex(
        yields.index.union(net_exposure.index)
    ).ffill().reindex(net_exposure.index)
    y = y.fillna(float(yields.median()))
    return net_exposure * (y / 100.0) / bars_per_year


def build(raw_dir: str = DEFAULT_RAW_DIR, out_path: str = DEFAULT_OUT) -> pd.Series:
    """Extract the yield series and persist it."""
    s = build_series(raw_dir)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    s.to_frame().to_parquet(out_path)
    print(f"[dividends] wrote {out_path}")
    return s


def load(path: str = DEFAULT_OUT) -> pd.Series:
    """Read the daily yield series, with a clear error naming the build step."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run `python src/phase4_data/dividends.py --build` first."
        )
    frame = pd.read_parquet(path)
    s = frame["div_yield_pct"]
    s.index = pd.DatetimeIndex(s.index).normalize()
    return s


# ── Dry run ─────────────────────────────────────────────────────────────────────

def _dry_run() -> None:
    """
    Self-test on a synthetic archive, plus the real one when it is present.

    The properties that matter are about the *accrual*, not the parse: a cash book and
    a dollar-neutral book must earn nothing, and a fully invested book must earn the
    yield — the three cases Phase 4b's flat constant got wrong.
    """
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp()
    checks: dict[str, bool] = {}
    try:
        os.makedirs(os.path.join(tmp, "index"))
        dates = pd.bdate_range("2020-01-01", periods=10)
        for i, d in enumerate(dates):
            pd.DataFrame({
                "Index Name": ["Nifty 50", "Nifty Bank"],
                "Closing Index Value": [12000.0 + i, 30000.0],
                _YIELD_COL: [1.25, 0.80],
            }).to_csv(os.path.join(tmp, "index", f"ind_close_all_{d:%d%m%Y}.csv"), index=False)

        s = build_series(raw_dir=tmp)
        checks["parses one row per session"] = len(s) == len(dates)
        checks["picks NIFTY-50, not another index"] = bool(np.allclose(s.to_numpy(), 1.25))
        checks["dates come from the filename"] = bool(s.index[0] == dates[0])

        # The three accrual cases.
        full = pd.Series(1.0, index=dates)
        neutral = pd.Series(0.0, index=dates)
        cash = pd.Series(0.0, index=dates)
        a_full = accrual(full, s)
        checks["fully invested earns the yield"] = bool(
            np.isclose(a_full.sum() * (252.0 / len(dates)), 0.0125, rtol=1e-9)
        )
        checks["dollar-neutral book earns ~0"] = bool(np.allclose(accrual(neutral, s), 0.0))
        checks["book in cash earns 0"] = bool(np.allclose(accrual(cash, s), 0.0))
        # Half-invested earns half — linear in exposure, as an income stream should be.
        checks["accrual is linear in exposure"] = bool(
            np.allclose(accrual(full * 0.5, s).to_numpy(), a_full.to_numpy() * 0.5)
        )
        # A bar with no yield quote is forward-filled, never dropped or zeroed.
        gap = pd.Series(1.0, index=pd.bdate_range("2020-01-01", periods=15))
        checks["missing quotes forward-fill, not zero"] = bool(
            (accrual(gap, s) > 0).all()
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("=" * 62)
    for name, ok in checks.items():
        print(f"  {name:<42}: {'PASS' if ok else 'FAIL'}")
    print("=" * 62)
    assert all(checks.values()), "dividends dry-run FAILED"

    if os.path.isdir(os.path.join(DEFAULT_RAW_DIR, "index")):
        print("\nReal index archive:")
        real = build_series()
        print(f"  yearly mean yield:\n{real.groupby(real.index.year).mean().round(3).to_string()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract the NIFTY-50 dividend yield series.")
    ap.add_argument("--build", action="store_true", help="build from the archive (else dry-run)")
    ap.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.build:
        _dry_run()
        return
    build(args.raw_dir, args.out)


if __name__ == "__main__":
    main()
