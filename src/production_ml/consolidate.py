"""
Phase 2 data acquisition — consolidate per-stock 15-minute CSVs into one
long-format Parquet.

``download_nse.py`` lands ~86 files named ``<SYMBOL>.NSE_15minute_ohlcv.csv`` in
``data/raw_15min/``. Each file is a single ticker's OHLCV history with schema:

    timestamp,open,high,low,close,volume,oi
    2015-02-02 09:15:00+05:30, ...

This script stacks them into one tidy long-format table
``(timestamp, ticker, open, high, low, close, volume)`` and writes it to
``data/15min_ohlcv.parquet`` — the single 15-minute source of truth that
``resample_bars.py`` later aggregates into 30min / 60min / daily.

Transformations (all structural — heavy feature-quality checks live downstream):
    - ``ticker`` is parsed from the filename (``ADANIENT.NSE_... -> ADANIENT``).
    - ``oi`` (open interest, all-zero for cash equities) is dropped.
    - ``timestamp`` is parsed tz-aware in IST (the data carries +05:30; India has
      no DST, so the fixed offset is unambiguous). Kept tz-aware so session/day
      grouping downstream is correct.
    - OHLCV coerced to numeric; rows with a non-positive/NaN close are dropped.
    - Duplicate ``(ticker, timestamp)`` rows are collapsed (keep last).
    - Sorted by ``(ticker, timestamp)``.

Usage
    python src/production_ml/consolidate.py
    python src/production_ml/consolidate.py --in data/raw_15min --out data/15min_ohlcv.parquet
    python src/production_ml/consolidate.py --limit 5      # first 5 files (smoke test)
"""

import argparse
import glob
import os
import re
from typing import Final

import pandas as pd

DEFAULT_IN_DIR: Final[str] = "data/raw_15min"
DEFAULT_OUT_PARQUET: Final[str] = "data/15min_ohlcv.parquet"
IST: Final[str] = "Asia/Kolkata"

# Index files (NIFTY-50, NIFTY-IT, ...) are basket averages, not tradeable peers of
# individual stocks, so they are excluded from the default (stocks) universe — an
# index in a cross-sectional decile ranking distorts it. A broad index can still be
# consolidated separately (``--kind indices``) to serve as the regime-tier market
# series. See docs/phase2_design_decisions.md §1.
_INDEX_PREFIX: Final[str] = "NIFTY"

# Canonical output columns (lowercase, matching feature_creator's constants).
OHLCV_COLS: Final[list[str]] = ["open", "high", "low", "close", "volume"]

# ``<TICKER>.<EXCHANGE>_15minute_ohlcv.csv`` — TICKER may contain hyphens
# (BAJAJ-AUTO) but not a dot, so the first dot separates ticker from exchange.
_FNAME_RE: Final[re.Pattern] = re.compile(
    r"^(?P<ticker>[^.]+)\.(?P<exchange>[A-Z]+)_15minute_ohlcv\.csv$"
)


def _ticker_from_filename(path: str) -> str | None:
    """Extract the ticker symbol from a raw CSV filename, or None if unmatched."""
    m = _FNAME_RE.match(os.path.basename(path))
    return m.group("ticker") if m else None


def _is_index(ticker: str) -> bool:
    """True for basket indices (NIFTY-50, NIFTY-IT, ...), False for stocks."""
    return ticker.upper().startswith(_INDEX_PREFIX)


def _select_paths(paths: list[str], kind: str) -> list[str]:
    """Filter raw CSV paths to the requested universe: stocks / indices / all."""
    if kind == "all":
        return paths
    want_index = kind == "indices"
    out = []
    for p in paths:
        t = _ticker_from_filename(p)
        if t is not None and _is_index(t) == want_index:
            out.append(p)
    return out


def _load_one(path: str) -> pd.DataFrame | None:
    """Read one raw per-stock CSV into a tidy (timestamp, ticker, OHLCV) frame."""
    ticker = _ticker_from_filename(path)
    if ticker is None:
        print(f"[consolidate] skip (name)   {os.path.basename(path)}")
        return None

    df = pd.read_csv(path)

    missing = ({"timestamp", *OHLCV_COLS}) - set(df.columns)
    if missing:
        print(f"[consolidate] skip (cols {sorted(missing)}) {os.path.basename(path)}")
        return None

    # tz-aware IST timestamps; utc=True then convert keeps a single tz dtype
    # even if any stray row parses to a different offset.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    ).dt.tz_convert(IST)

    for col in OHLCV_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ticker"] = ticker
    df = df[["timestamp", "ticker", *OHLCV_COLS]]

    # Structural cleaning: valid timestamp + tradeable close.
    df = df[df["timestamp"].notna() & df["close"].gt(0)]
    df = df.drop_duplicates(subset=["ticker", "timestamp"], keep="last")
    return df


def consolidate(
    in_dir: str, out_parquet: str, kind: str = "stocks", limit: int = 0
) -> pd.DataFrame:
    """
    Stack the raw per-symbol CSVs in ``in_dir`` into one long-format Parquet.

    Args:
        in_dir:      directory of ``<TICKER>.NSE_15minute_ohlcv.csv`` files.
        out_parquet: destination Parquet path.
        kind:        universe filter — ``"stocks"`` (default, excludes NIFTY-*),
                     ``"indices"`` (NIFTY-* only), or ``"all"``.
        limit:       if > 0, consolidate at most this many files (smoke testing).

    Returns the consolidated DataFrame (also written to ``out_parquet``).
    """
    all_paths = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
    if not all_paths:
        raise FileNotFoundError(
            f"No CSVs in {in_dir!r}. Run download_nse.py first."
        )
    paths = _select_paths(all_paths, kind)
    if not paths:
        raise FileNotFoundError(f"No {kind!r} files among {len(all_paths)} CSVs in {in_dir!r}.")
    print(f"[consolidate] universe={kind} | {len(paths)} of {len(all_paths)} files selected")
    if limit > 0:
        paths = paths[:limit]

    frames: list[pd.DataFrame] = []
    for i, path in enumerate(paths, start=1):
        one = _load_one(path)
        if one is None or one.empty:
            continue
        frames.append(one)
        print(
            f"[consolidate] [{i}/{len(paths)}] {one['ticker'].iloc[0]:<14} "
            f"rows={len(one):>7} "
            f"{one['timestamp'].min().date()} → {one['timestamp'].max().date()}"
        )

    if not frames:
        raise RuntimeError("No usable files consolidated.")

    out = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "timestamp"])
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
    out.to_parquet(out_parquet, index=False)

    # Post-write read-back assertion (mirrors the tier4 ledger pattern).
    back = pd.read_parquet(out_parquet)
    assert back.shape == out.shape, "Parquet read-back shape mismatch"

    print(
        f"\n[consolidate] wrote {out_parquet} | "
        f"rows={len(out):,} | tickers={out['ticker'].nunique()} | "
        f"{out['timestamp'].min().date()} → {out['timestamp'].max().date()}"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Consolidate per-stock 15-min CSVs → Parquet.")
    ap.add_argument("--in", dest="in_dir", default=DEFAULT_IN_DIR, help="raw CSV directory")
    ap.add_argument("--out", default=DEFAULT_OUT_PARQUET, help="output Parquet path")
    ap.add_argument(
        "--kind", choices=["stocks", "indices", "all"], default="stocks",
        help="universe filter (default: stocks, excludes NIFTY-* indices)",
    )
    ap.add_argument("--limit", type=int, default=0, help="consolidate at most N files (0=all)")
    args = ap.parse_args()

    consolidate(args.in_dir, args.out, kind=args.kind, limit=args.limit)


if __name__ == "__main__":
    main()
