"""
Phase 4c B1 — the point-in-time shortable set.

Phase 4b's two DSR passes both depend on holding ~50 short names drawn from the bottom
decile of a 500-name cross-section. In India that book cannot simply be assumed:
overnight shorting is restricted to the F&O segment plus a thin Securities Lending &
Borrowing (SLB) market, and the bottom decile of a broad universe systematically
selects small, distressed, often non-F&O names. The backtest charged a flat 50 bps/yr
borrow and assumed unlimited availability.

This module measures availability instead of assuming it. For each session it reads the
derivatives bhavcopy, keeps the **stock futures** rows, and takes the set of underlying
symbols that actually traded:

    legacy  INSTRUMENT == "FUTSTK"   symbol in SYMBOL
    UDiFF   FinInstrmTp == "STF"     symbol in TckrSymb

Two properties make this the right instrument. It is derived from *traded contracts*,
so it is genuinely point-in-time and carries **no survivorship bias** — a name that lost
F&O eligibility in 2019 stops appearing in 2019, not retroactively. And it needs no
membership list, which NSE does not publish historically.

**What it is a proxy for, and in which direction it errs.** F&O eligibility is used as
the availability proxy for SLB borrow, not because the strategy trades futures (§2.1 of
the plan closes that route: at Rs 1 cr the short book wants 0.31 lots per name, and lot
granularity and market impact push AUM in opposite directions with no overlap). SLB
activity concentrates in approximately the F&O names, so this is the honest
approximation available — and it is an **upper bound** on shortability. Real SLB is
thinner, borrow is not always locatable, and recalls happen. Every result built on this
mask is therefore a best case for the short leg.

Output: ``data/bhavcopy/shortable_mask.parquet``, wide ``(date × ticker)`` boolean on
the panel's own trading calendar and ticker namespace.

Usage
    python src/phase4_data/shortable.py            # dry-run self-test
    python src/phase4_data/shortable.py --build    # build from data/bhavcopy/fo*
"""

import argparse
import glob
import io
import os
import re
import sys
import zipfile
from typing import Final

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_RAW_DIR: Final[str] = "data/bhavcopy"
DEFAULT_PANEL: Final[str] = "data/bhavcopy/panel_daily.parquet"
DEFAULT_OUT: Final[str] = "data/bhavcopy/shortable_mask.parquet"

# Stock-futures selectors, one per archive era.
_LEGACY_INSTRUMENT: Final[str] = "FUTSTK"
_UDIFF_INSTRUMENT: Final[str] = "STF"

_MONTHS: Final[tuple[str, ...]] = (
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
)
_LEGACY_RE: Final[re.Pattern] = re.compile(
    r"^fo(\d{2})([A-Z]{3})(\d{4})bhav\.csv\.zip$", re.IGNORECASE
)
_UDIFF_RE: Final[re.Pattern] = re.compile(
    r"^BhavCopy_NSE_FO_0_0_0_(\d{8})_F_0000\.csv\.zip$", re.IGNORECASE
)


def date_from_filename(path: str) -> pd.Timestamp | None:
    """
    Session date from an F&O archive filename, either era.

    Read from the *filename*, never from the file's own date column — the same defect
    that cost the panel builder 62 days of index history (``TIMESTAMP`` drifts format
    mid-archive) applies here too.
    """
    name = os.path.basename(path)
    if (m := _LEGACY_RE.match(name)):
        day, mon, year = int(m.group(1)), m.group(2).upper(), int(m.group(3))
        return pd.Timestamp(year=year, month=_MONTHS.index(mon) + 1, day=day)
    if (m := _UDIFF_RE.match(name)):
        return pd.to_datetime(m.group(1), format="%Y%m%d")
    return None


def parse_fo_zip(path: str) -> set[str]:
    """
    Underlying symbols with a stock future that traded in one session's archive.

    Returns an empty set for an unreadable or schema-unrecognized file; the caller
    counts those and reports them rather than silently treating the day as "nothing
    was shortable", which would be a very different claim.
    """
    try:
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not names:
                return set()
            with z.open(names[0]) as fh:
                df = pd.read_csv(io.BytesIO(fh.read()), low_memory=False)
    except (OSError, zipfile.BadZipFile, pd.errors.ParserError, UnicodeDecodeError):
        return set()

    cols = {c.strip(): c for c in df.columns}
    if "INSTRUMENT" in cols and "SYMBOL" in cols:
        inst, sym = cols["INSTRUMENT"], cols["SYMBOL"]
        hit = df[df[inst].astype("string").str.strip().str.upper() == _LEGACY_INSTRUMENT]
    elif "FinInstrmTp" in cols and "TckrSymb" in cols:
        inst, sym = cols["FinInstrmTp"], cols["TckrSymb"]
        hit = df[df[inst].astype("string").str.strip().str.upper() == _UDIFF_INSTRUMENT]
    else:
        return set()

    return set(hit[sym].astype("string").str.strip().str.upper().dropna())


def symbol_to_ticker(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(date, symbol) -> panel ticker map.

    The F&O archive names an underlying by its NSE trading symbol; the panel keys on
    ``ticker``, which is that symbol except where ``resolve_tickers`` had to break a
    reuse collision (``NAME.2``). Mapping per date rather than globally also follows
    symbol renames through time, so a name that changed ticker mid-sample keeps its
    eligibility on both sides of the change.
    """
    return (
        panel[["date", "symbol", "ticker"]]
        .assign(symbol=lambda d: d["symbol"].astype("string").str.strip().str.upper())
        .drop_duplicates(subset=["date", "symbol"])
    )


def build_mask(
    raw_dir: str = DEFAULT_RAW_DIR,
    panel: pd.DataFrame | None = None,
    panel_path: str = DEFAULT_PANEL,
    limit: int = 0,
) -> pd.DataFrame:
    """
    Build the ``(date × ticker)`` boolean shortable mask on the panel's calendar.

    Sessions for which no F&O archive is on disk inherit the previous available
    session's set by forward-fill — F&O eligibility changes on a review cycle, not
    daily, so a one-day archive gap means "unknown", never "nothing was borrowable".
    The fill is causal (``ffill`` only, never ``bfill``) and its extent is reported.
    """
    if panel is None:
        if not os.path.exists(panel_path):
            raise FileNotFoundError(
                f"{panel_path} not found — run bhavcopy_panel.py --build first."
            )
        panel = pd.read_parquet(panel_path, columns=["date", "symbol", "ticker"])

    paths = sorted(
        glob.glob(os.path.join(raw_dir, "fo", "*.zip"))
        + glob.glob(os.path.join(raw_dir, "fo_udiff", "*.zip"))
    )
    if not paths:
        raise FileNotFoundError(
            f"No F&O archives under {raw_dir}/fo — run "
            "`python src/phase4_data/bhavcopy_download.py --kinds fo` first."
        )
    if limit > 0:
        paths = paths[:limit]

    per_date: dict[pd.Timestamp, set[str]] = {}
    n_bad = 0
    for i, path in enumerate(paths, start=1):
        d = date_from_filename(path)
        if d is None:
            n_bad += 1
            continue
        syms = parse_fo_zip(path)
        if not syms:
            n_bad += 1
            continue
        per_date[d] = per_date.get(d, set()) | syms
        if i % 500 == 0 or i == len(paths):
            print(f"[shortable] parsed {i}/{len(paths)} archives")

    if not per_date:
        raise ValueError("No stock-futures rows parsed from any F&O archive.")

    smap = symbol_to_ticker(panel)
    calendar = pd.DatetimeIndex(sorted(panel["date"].unique()))

    # Long (date, symbol) frame of everything with a live future, mapped to tickers.
    long = pd.DataFrame(
        [(d, s) for d, syms in per_date.items() for s in syms],
        columns=["date", "symbol"],
    )
    joined = long.merge(smap, on=["date", "symbol"], how="inner")
    matched = len(joined) / max(len(long), 1)

    wide = (
        joined.assign(flag=True)
        .pivot_table(index="date", columns="ticker", values="flag", aggfunc="any")
        .reindex(calendar)
    )
    fo_days = pd.DatetimeIndex(sorted(per_date))
    have = pd.Series(calendar.isin(fo_days), index=calendar)
    n_gap = int((~have).sum())
    # Absence has two meanings and they must not be conflated. On a session whose
    # archive we hold, a name missing from the pivot genuinely did not have a live
    # future — that is a hard False, and forward-filling it would let eligibility that
    # ENDED live on forever. Only a session with no archive at all is unknown, and
    # only those rows are carried from the previous F&O day.
    reported = wide.fillna(False).where(have, other=np.nan, axis=0)
    mask = reported.ffill().fillna(False).astype(bool)

    per_day = mask.sum(axis=1)
    active = per_day[per_day > 0]
    print(
        f"[shortable] {len(per_date):,} F&O sessions parsed (unreadable/unmatched files: "
        f"{n_bad}) | symbol->ticker match {matched:.1%}\n"
        f"[shortable] mask {mask.shape[0]:,} dates × {mask.shape[1]:,} tickers | "
        f"names/day min={int(active.min()) if len(active) else 0} "
        f"mean={active.mean():.0f} max={int(active.max()) if len(active) else 0} | "
        f"{n_gap} calendar sessions forward-filled from the prior F&O day"
    )
    return mask


def universe_coverage(mask: pd.DataFrame, universe_mask_path: str) -> pd.DataFrame:
    """
    How much of the tradeable universe was borrowable, per year — a read-only readout.

    The plan's §2.1 reference: ~171 of 500 in-universe names shortable (34%), at a
    median liquidity rank of ~105 of 500.
    """
    uni = pd.read_parquet(universe_mask_path)
    uni.index = pd.DatetimeIndex(uni.index).normalize()
    d = mask.index.intersection(uni.index)
    u = uni.loc[d]
    # Reindex the mask over the FULL universe, filling False. Intersecting the two
    # column sets instead would silently restrict the denominator to names that ever
    # had a future — turning "how much of the 500 is borrowable?" into "how much of
    # the F&O list is in the 500?", which reads ~95% and means something else entirely.
    m = mask.reindex(index=d, columns=u.columns).fillna(False).astype(bool)

    in_uni = u.sum(axis=1)
    both = (m & u).sum(axis=1)
    out = pd.DataFrame({"in_universe": in_uni, "shortable_in_universe": both})
    out = out[out["in_universe"] > 0]
    out["pct"] = out["shortable_in_universe"] / out["in_universe"]
    return out.groupby(out.index.year).mean()


def build(
    raw_dir: str = DEFAULT_RAW_DIR,
    panel_path: str = DEFAULT_PANEL,
    out_path: str = DEFAULT_OUT,
    universe_mask_path: str = "data/nse500_universe_mask.parquet",
) -> pd.DataFrame:
    """Build the mask, print the in-universe coverage readout, and persist."""
    mask = build_mask(raw_dir, panel_path=panel_path)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    mask.to_parquet(out_path)
    print(f"[shortable] wrote {out_path}")

    if os.path.exists(universe_mask_path):
        cov = universe_coverage(mask, universe_mask_path)
        print(f"\n{'=' * 60}\nIN-UNIVERSE SHORTABILITY BY YEAR\n{'=' * 60}")
        print(f"  {'year':<8}{'in universe':>13}{'shortable':>11}{'pct':>8}")
        for year, r in cov.iterrows():
            print(f"  {year:<8}{r['in_universe']:>13.0f}{r['shortable_in_universe']:>11.0f}"
                  f"{r['pct']:>8.1%}")
        print(f"  {'ALL':<8}{cov['in_universe'].mean():>13.0f}"
              f"{cov['shortable_in_universe'].mean():>11.0f}{cov['pct'].mean():>8.1%}")
        print("=" * 60)
    return mask


def load(path: str = DEFAULT_OUT) -> pd.DataFrame:
    """Read the shortable mask, with a clear error naming the build step."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run `python src/phase4_data/shortable.py --build` first."
        )
    mask = pd.read_parquet(path)
    mask.index = pd.DatetimeIndex(mask.index).normalize()
    return mask.astype(bool)


# ── Dry run ─────────────────────────────────────────────────────────────────────

def _write_zip(path: str, name: str, frame: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, frame.to_csv(index=False))


def _dry_run() -> None:
    """
    Self-test on synthesized archives in both eras, with the right answer known.

    Checks the four things the short leg depends on: only stock futures are counted
    (not options, not index futures), both era schemas parse, an archive gap is
    forward-filled rather than read as "nothing was borrowable", and — the property
    that makes this usable at all — eligibility that *ends* stays ended.
    """
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp()
    try:
        dates = pd.bdate_range("2016-02-01", periods=6)
        legacy, udiff = dates[:3], dates[3:]

        # AAA is always eligible; BBB loses eligibility after the 2nd session; CCC
        # only ever appears as an OPTION and an index future, so it is never shortable.
        for i, d in enumerate(dates):
            syms = ["AAA"] + (["BBB"] if i < 2 else [])
            rows = [{"INSTRUMENT": "FUTSTK", "SYMBOL": s} for s in syms]
            rows += [{"INSTRUMENT": "OPTSTK", "SYMBOL": "CCC"},
                     {"INSTRUMENT": "FUTIDX", "SYMBOL": "NIFTY"}]
            frame = pd.DataFrame(rows)
            if d in legacy:
                fn = f"fo{d.day:02d}{_MONTHS[d.month - 1]}{d.year}bhav.csv.zip"
                _write_zip(os.path.join(tmp, "fo", fn), fn[:-4], frame)
            else:
                frame = frame.rename(columns={"INSTRUMENT": "FinInstrmTp", "SYMBOL": "TckrSymb"})
                frame["FinInstrmTp"] = frame["FinInstrmTp"].map(
                    {"FUTSTK": "STF", "OPTSTK": "STO", "FUTIDX": "IDF"}
                )
                fn = f"BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
                _write_zip(os.path.join(tmp, "fo_udiff", fn), fn[:-4], frame)

        # A 7th panel session with NO archive — must be forward-filled, not zeroed.
        cal = list(dates) + [pd.Timestamp("2016-02-10")]
        panel = pd.DataFrame(
            [{"date": d, "symbol": s, "ticker": s}
             for d in cal for s in ("AAA", "BBB", "CCC", "DDD")]
        )
        mask = build_mask(raw_dir=tmp, panel=panel)

        checks = {
            "always-eligible name is shortable throughout": bool(mask["AAA"].all()),
            "options/index futures never count": "CCC" not in mask.columns or not bool(mask["CCC"].any()),
            "a name absent from F&O is never shortable": "DDD" not in mask.columns,
            "legacy era parsed": bool(mask.loc[legacy[0], "AAA"]),
            "udiff era parsed": bool(mask.loc[udiff[0], "AAA"]),
            "lost eligibility stays lost": not bool(mask.loc[dates[2:], "BBB"].any()),
            "held eligibility before the loss": bool(mask.loc[dates[:2], "BBB"].all()),
            "archive gap forward-fills, not zeroes": bool(
                mask.loc[pd.Timestamp("2016-02-10"), "AAA"]
            ),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("=" * 66)
    for name, ok in checks.items():
        print(f"  {name:<46}: {'PASS' if ok else 'FAIL'}")
    print("=" * 66)
    assert all(checks.values()), "shortable dry-run FAILED"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the point-in-time shortable mask.")
    ap.add_argument("--build", action="store_true", help="build from the archives (else dry-run)")
    ap.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.build:
        _dry_run()
        return
    build(args.raw_dir, args.panel, args.out)


if __name__ == "__main__":
    main()
