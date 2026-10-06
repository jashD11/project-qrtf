"""
Phase 2 data acquisition — resample the 15-minute source into coarser bars.

Reads the consolidated 15-minute Parquet from ``consolidate.py`` and aggregates
it into **30min**, **60min** (hourly), and **daily** OHLCV Parquets. All four
frequencies therefore derive from one source of truth with perfectly aligned
timelines.

Session-aware binning
    NSE's continuous session is 09:15–15:30 IST. Intraday bars are bucketed by
    their offset from that day's 09:15 open, so a bin can **never span the
    overnight gap** (its width, ≤60min, is far smaller than the ~18h gap, and the
    bin key embeds the calendar date). Daily bars group by calendar date, which
    is exactly one session. This is fully vectorized — no per-day Python loop.

OHLCV aggregation within each bin
    open = first, high = max, low = min, close = last, volume = sum.

Outputs (alongside the source in ``data/``)
    30min_ohlcv.parquet, 60min_ohlcv.parquet, daily_ohlcv.parquet

Usage
    python src/production_ml/resample_bars.py
    python src/production_ml/resample_bars.py --in data/15min_ohlcv.parquet --outdir data
    python src/production_ml/resample_bars.py --freqs 30min daily
"""

import argparse
import os
from typing import Final

import pandas as pd

DEFAULT_IN_PARQUET: Final[str] = "data/15min_ohlcv.parquet"
DEFAULT_OUT_DIR: Final[str] = "data"

# NSE continuous session open, in minutes past midnight (09:15 = 555).
_SESSION_OPEN: Final[pd.Timedelta] = pd.Timedelta(hours=9, minutes=15)

# Target frequency → intraday bin width in minutes (None = one bar per session/day).
FREQ_RULE_MIN: Final[dict[str, int | None]] = {
    "30min": 30,
    "60min": 60,
    "daily": None,
}

# groupby aggregation reproducing an OHLCV bar; relies on chronological row order.
_OHLCV_AGG: Final[dict[str, tuple[str, str]]] = {
    "open": ("open", "first"),
    "high": ("high", "max"),
    "low": ("low", "min"),
    "close": ("close", "last"),
    "volume": ("volume", "sum"),
}


def _bar_open_ts(ts: pd.Series, rule_min: int) -> pd.Series:
    """
    Vectorized session-relative bin label for each timestamp.

    Returns, for every row, the timestamp of the start of the ``rule_min`` bin it
    falls into, measured from that day's 09:15 open. Rows on different calendar
    days can never share a bin because ``day_open`` carries the date.
    """
    day_open = ts.dt.normalize() + _SESSION_OPEN
    offset_min = (ts - day_open).dt.total_seconds() // 60
    bin_idx = (offset_min // rule_min).astype("int64")
    return day_open + pd.to_timedelta(bin_idx * rule_min, unit="m")


def resample_frequency(df: pd.DataFrame, name: str, rule_min: int | None) -> pd.DataFrame:
    """
    Aggregate the 15-min long frame into one coarser frequency.

    Args:
        df:       long-format (timestamp, ticker, OHLCV), chronologically sortable.
        name:     frequency label (for logging).
        rule_min: intraday bin width in minutes, or None for daily (per-session).

    Returns:
        Long-format (timestamp, ticker, open, high, low, close, volume) frame.
    """
    df = df.sort_values(["ticker", "timestamp"])

    if rule_min is None:
        # Daily: one bar per (ticker, calendar date); label = session date midnight.
        bar_ts = df["timestamp"].dt.normalize()
    else:
        bar_ts = _bar_open_ts(df["timestamp"], rule_min)

    grouped = (
        df.assign(bar_ts=bar_ts)
        .groupby(["ticker", "bar_ts"], sort=True)
        .agg(**_OHLCV_AGG)
        .reset_index()
        .rename(columns={"bar_ts": "timestamp"})
    )
    out = grouped[["timestamp", "ticker", "open", "high", "low", "close", "volume"]]
    out = out.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    bars_per_day = out.groupby([out["ticker"], out["timestamp"].dt.normalize()]).size()
    print(
        f"[resample] {name:<6} | rows={len(out):>9,} | tickers={out['ticker'].nunique()} | "
        f"bars/day median={int(bars_per_day.median())} max={int(bars_per_day.max())}"
    )
    return out


def resample_all(in_parquet: str, out_dir: str, freqs: list[str]) -> None:
    """Resample the source into each requested frequency and write Parquets."""
    if not os.path.exists(in_parquet):
        raise FileNotFoundError(
            f"{in_parquet!r} not found. Run consolidate.py first."
        )
    src = pd.read_parquet(in_parquet)
    print(
        f"[resample] source {in_parquet} | rows={len(src):,} | "
        f"tickers={src['ticker'].nunique()} | "
        f"{src['timestamp'].min().date()} → {src['timestamp'].max().date()}"
    )
    os.makedirs(out_dir, exist_ok=True)

    for name in freqs:
        rule_min = FREQ_RULE_MIN[name]
        out = resample_frequency(src, name, rule_min)
        path = os.path.join(out_dir, f"{name}_ohlcv.parquet")
        out.to_parquet(path, index=False)

        back = pd.read_parquet(path)
        assert back.shape == out.shape, f"{path} read-back shape mismatch"
        print(f"[resample] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Resample 15-min OHLCV → 30/60min/daily.")
    ap.add_argument("--in", dest="in_parquet", default=DEFAULT_IN_PARQUET)
    ap.add_argument("--outdir", default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--freqs",
        nargs="+",
        choices=list(FREQ_RULE_MIN),
        default=list(FREQ_RULE_MIN),
        help="which frequencies to produce (default: all)",
    )
    args = ap.parse_args()
    resample_all(args.in_parquet, args.outdir, args.freqs)


if __name__ == "__main__":
    main()
