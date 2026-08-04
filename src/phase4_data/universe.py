"""
Phase 4 universe — point-in-time top-N selection, and the engine-schema export.

**Why not just use the NIFTY 500 list?** Because today's index membership *is*
survivorship bias in its purest form: every name on it survived to today, and
backtesting on it silently assumes you knew that in 2013. The universe is therefore
rebuilt from the panel itself, using only information that existed on each selection
date.

Selection rule, applied at each quarter start (aligned to the ``PREDICT_WINDOW=63``
walk-forward refit cadence so the universe and the models turn over together):

    rank every EQ/BE name by **median daily rupee turnover over the trailing 252
    trading days**, using only bars strictly *before* the selection date;
    require ≥200 of those 252 days actually traded (drops near-dormant listings and,
    for free, names just back from a long suspension);
    take the top ``TOP_N``, and hold that set fixed for the quarter.

Holding the set fixed for a quarter means no intra-quarter look-ahead and bounded,
measurable churn. Median rather than mean turnover keeps one block-deal day from
promoting an otherwise illiquid name.

The module also writes the two artefacts the engine consumes, in exactly the schema
the existing loaders expect (``data_scraping.load_bars`` renames ``timestamp``→``date``
and MultiIndexes it unchanged), so nothing downstream has to learn a new format:

    data/nse500_daily_ohlcv.parquet   timestamp, ticker, open, high, low, close, volume
    data/nse500_universe_mask.parquet (date × ticker) bool

Only names selected at least once are exported — but with their **full** price history,
including the quarters they were not selected in, so rolling features never see a
truncated window. A name that never enters the universe can never carry a weight, and
the mask already excludes it from cross-sectional normalization, so exporting it would
cost feature-build time and change nothing.

Usage
    python src/phase4_data/universe.py            # dry-run self-test
    python src/phase4_data/universe.py --build    # build from data/bhavcopy/panel_daily.parquet
    python src/phase4_data/universe.py --build --top-n 100   # smaller universe (smoke)
"""

import argparse
import os
from typing import Final

import numpy as np
import pandas as pd

DEFAULT_PANEL: Final[str] = "data/bhavcopy/panel_daily.parquet"
DEFAULT_MASK: Final[str] = "data/nse500_universe_mask.parquet"
DEFAULT_OHLCV: Final[str] = "data/nse500_daily_ohlcv.parquet"

TOP_N: Final[int] = 500
LOOKBACK: Final[int] = 252        # trailing window for the liquidity rank (1 year)
REBALANCE_EVERY: Final[int] = 63  # matches ML_CONFIG PREDICT_WINDOW (1 quarter)
MIN_TRADED: Final[int] = 200      # of LOOKBACK days, how many must have actually traded

OHLCV_COLS: Final[list[str]] = ["open", "high", "low", "close", "volume"]


def resolve_tickers(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee a one-to-one ISIN↔ticker map.

    ``bhavcopy_panel`` sets ``ticker`` to each ISIN's last observed symbol, which is
    almost always unique — but NSE does re-issue a symbol after a delisting, and two
    different companies sharing one column name downstream would silently merge two
    price series. Collisions are broken deterministically by ISIN so the outcome does
    not depend on row order, and reported rather than fixed quietly.
    """
    out = panel.copy()
    pairs = out[["isin", "ticker"]].drop_duplicates()
    dup_names = pairs["ticker"].value_counts()
    dup_names = dup_names[dup_names > 1]
    if len(dup_names) == 0:
        return out

    print(f"[universe] WARN: {len(dup_names)} symbol(s) reused across ISINs — disambiguating")
    rename: dict[str, str] = {}
    for name in dup_names.index:
        isins = sorted(pairs.loc[pairs["ticker"] == name, "isin"])
        for i, isin in enumerate(isins[1:], start=2):  # first keeps the bare symbol
            rename[isin] = f"{name}.{i}"
            print(f"[universe]   {name} ← {isin} → {name}.{i}")
    out["ticker"] = out["isin"].map(rename).fillna(out["ticker"])
    return out


def rebalance_dates(dates: np.ndarray) -> list[int]:
    """
    Positions in the trading calendar at which the universe is re-selected.

    The first selection can only happen once a full ``LOOKBACK`` of history exists —
    ranking on a partial window would favour whichever names happened to list early.
    """
    return list(range(LOOKBACK, len(dates), REBALANCE_EVERY))


def build_mask(
    panel: pd.DataFrame,
    top_n: int = TOP_N,
    lookback: int = LOOKBACK,
    every: int = REBALANCE_EVERY,
    min_traded: int = MIN_TRADED,
) -> pd.DataFrame:
    """
    Build the ``(date × ticker)`` boolean point-in-time universe mask.

    Every selection at calendar position ``r`` reads ``turnover.iloc[r-lookback:r]`` —
    the slice ends at ``r-1``, so the selection date's own bar is never part of the
    decision that admits it. That half-open window is the whole causality argument and
    is asserted below.
    """
    turnover = panel.pivot_table(
        index="date", columns="ticker", values="turnover", aggfunc="last"
    ).sort_index()
    dates = turnover.index.to_numpy()

    mask = pd.DataFrame(
        False, index=turnover.index, columns=turnover.columns, dtype=bool
    )
    traded = turnover.gt(0)  # a row with zero turnover is a listing, not a trade

    picks = rebalance_dates(dates)
    if not picks:
        raise ValueError(
            f"panel has {len(dates)} days, need > {lookback} for one selection"
        )

    sizes: list[int] = []
    for r in picks:
        window = turnover.iloc[r - lookback:r]
        assert len(window) == lookback, "short lookback window"
        assert window.index[-1] < turnover.index[r], "lookback window touches the future"

        eligible = traded.iloc[r - lookback:r].sum(axis=0) >= min_traded
        med = window.where(window.gt(0)).median(axis=0)
        chosen = med[eligible & med.notna()].nlargest(top_n).index

        stop = min(r + every, len(dates))
        mask.iloc[r:stop, mask.columns.get_indexer(chosen)] = True
        sizes.append(len(chosen))

    n_sel = int(mask.any(axis=0).sum())
    print(
        f"[universe] {len(picks)} rebalances | per-date size "
        f"min={min(sizes)} max={max(sizes)} | {n_sel} distinct names ever selected | "
        f"active {mask.index[picks[0]].date()} → {mask.index[-1].date()}"
    )
    return mask


def churn(mask: pd.DataFrame) -> pd.Series:
    """Fraction of the universe replaced at each rebalance — a stability diagnostic."""
    picks = rebalance_dates(mask.index.to_numpy())
    rows = {}
    prev: set[str] | None = None
    for r in picks:
        cur = set(mask.columns[mask.iloc[r].to_numpy()])
        if prev is not None and prev:
            rows[mask.index[r]] = len(cur - prev) / len(prev)
        prev = cur
    return pd.Series(rows, name="churn")


def emit_engine_files(
    panel: pd.DataFrame,
    mask: pd.DataFrame,
    out_ohlcv: str = DEFAULT_OHLCV,
    out_mask: str = DEFAULT_MASK,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Write the OHLCV panel and universe mask in the engine's existing schema.

    Restricted to names selected at least once, but carrying their complete history so
    a rolling 252-day feature computed on the first selected bar still sees 252 real
    bars behind it.
    """
    selected = mask.columns[mask.any(axis=0).to_numpy()]
    sub = panel[panel["ticker"].isin(set(selected))]

    ohlcv = (
        sub.rename(columns={"date": "timestamp"})[["timestamp", "ticker", *OHLCV_COLS]]
        .sort_values(["ticker", "timestamp"])
        .reset_index(drop=True)
    )
    mask_out = mask.loc[:, selected]

    for path in (out_ohlcv, out_mask):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ohlcv.to_parquet(out_ohlcv, index=False)
    mask_out.to_parquet(out_mask)

    assert pd.read_parquet(out_ohlcv).shape == ohlcv.shape, "OHLCV read-back mismatch"
    assert pd.read_parquet(out_mask).shape == mask_out.shape, "mask read-back mismatch"

    print(
        f"[universe] wrote {out_ohlcv} | rows={len(ohlcv):,} tickers={ohlcv['ticker'].nunique():,}"
        f"\n[universe] wrote {out_mask} | {mask_out.shape[0]:,} dates × {mask_out.shape[1]:,} tickers"
    )
    return ohlcv, mask_out


def build(
    panel_path: str = DEFAULT_PANEL,
    out_ohlcv: str = DEFAULT_OHLCV,
    out_mask: str = DEFAULT_MASK,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """Load the panel, select the universe, and write both engine artefacts."""
    if not os.path.exists(panel_path):
        raise FileNotFoundError(f"{panel_path} not found — run bhavcopy_panel.py --build first.")
    panel = resolve_tickers(pd.read_parquet(panel_path))
    mask = build_mask(panel, top_n=top_n)

    c = churn(mask)
    if len(c):
        print(f"[universe] quarterly churn: mean={c.mean():.1%} max={c.max():.1%}")

    emit_engine_files(panel, mask, out_ohlcv=out_ohlcv, out_mask=out_mask)
    return mask


# ── Dry run ─────────────────────────────────────────────────────────────────────

def _dry_run() -> None:
    """
    Self-test on a synthetic panel where the right answer is known by construction.

    Names are given deliberately ordered turnover levels, so the selected set must be
    exactly the top slice; a dormant name and a late lister check the eligibility rule;
    and the causality assertion is re-checked directly against a planted future spike.
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2018-01-01", periods=400)
    n = 30
    frames = []
    for i in range(n):
        level = 10.0 ** (9 - i / 6.0)  # strictly decreasing turnover by index
        px = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
        to = level * rng.lognormal(0, 0.2, len(dates))
        d, p, t = dates, px, to
        if i == 5:                       # dormant: trades only 1 day in 4
            keep = np.arange(len(dates)) % 4 == 0
            d, p, t = dates[keep], px[keep], to[keep]
        if i == 6:                       # late lister: no history before bar 300
            d, p, t = dates[300:], px[300:], to[300:]
        frames.append(pd.DataFrame({
            "date": d, "isin": f"INE{i:09d}", "ticker": f"T{i:02d}",
            "open": p, "high": p, "low": p, "close": p,
            "volume": t / p, "turnover": t,
        }))
    panel = pd.concat(frames, ignore_index=True)

    top_n = 10
    mask = build_mask(panel, top_n=top_n, lookback=252, every=63, min_traded=200)

    first = mask.index[252]
    chosen = set(mask.columns[mask.loc[first].to_numpy()])
    assert len(chosen) == top_n, f"expected {top_n} names, got {len(chosen)}"
    assert "T05" not in chosen, "dormant name passed the traded-days filter"
    assert "T06" not in chosen, "late lister selected with no trailing history"
    expected = {f"T{i:02d}" for i in range(top_n + 2)} - {"T05", "T06"}
    assert chosen == expected, f"wrong selection\n got {sorted(chosen)}\n want {sorted(expected)}"

    # Nothing is selected before a full lookback exists.
    assert not mask.iloc[:252].to_numpy().any(), "selected a name before any history"

    # Causality, tested rather than asserted: a name whose turnover explodes only
    # *after* a rebalance must not be selected at that rebalance.
    panel2 = panel.copy()
    late = (panel2["ticker"] == "T29") & (panel2["date"] >= mask.index[252])
    panel2.loc[late, "turnover"] *= 1e6
    m2 = build_mask(panel2, top_n=top_n, lookback=252, every=63, min_traded=200)
    assert not bool(m2.loc[first, "T29"]), "future turnover leaked into the selection"
    # One rebalance later only 63 of the 252 window days are spiked, and a median is
    # unmoved by a quarter of the window — that resistance to a transient block-deal
    # burst is exactly why the rank uses the median. It takes hold at the rebalance
    # where the spike finally covers half the lookback.
    assert not bool(m2.iloc[315]["T29"]), "a quarter-length spike moved the median"
    assert bool(m2.iloc[378]["T29"]), "a persistent spike never entered the ranking"

    ohlcv, mask_out = emit_engine_files(
        panel, mask,
        out_ohlcv="/tmp/_p4_ohlcv.parquet", out_mask="/tmp/_p4_mask.parquet",
    )
    assert list(ohlcv.columns) == ["timestamp", "ticker", *OHLCV_COLS], "engine schema drift"
    assert set(ohlcv["ticker"]) == set(mask_out.columns), "OHLCV/mask ticker sets disagree"
    for p in ("/tmp/_p4_ohlcv.parquet", "/tmp/_p4_mask.parquet"):
        os.remove(p)

    print("\n[universe] dry-run OK — selection exact, filters bite, no look-ahead")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the point-in-time universe mask.")
    ap.add_argument("--build", action="store_true", help="build from the panel (else dry-run)")
    ap.add_argument("--panel", default=DEFAULT_PANEL, help="input panel Parquet")
    ap.add_argument("--out-ohlcv", default=DEFAULT_OHLCV, help="engine OHLCV output")
    ap.add_argument("--out-mask", default=DEFAULT_MASK, help="universe mask output")
    ap.add_argument("--top-n", type=int, default=TOP_N, help="names per rebalance")
    args = ap.parse_args()

    if not args.build:
        _dry_run()
        return
    build(args.panel, out_ohlcv=args.out_ohlcv, out_mask=args.out_mask, top_n=args.top_n)


if __name__ == "__main__":
    main()
