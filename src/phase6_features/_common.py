"""
Shared plumbing for the Phase 6 feature builders: paths, the holdout guard, the price
panel in "own-bar" long form, and rolling sums over each name's own bars.

**Own bars.** The 17 existing features compute every window over each name's own trading
rows (``feature_creator._shift`` / ``_roll`` group by ticker), so a halted day is skipped,
not counted as a missing bar. The Phase 6 per-stock features follow the same convention.
Where a stock return is paired with an index return, the index return is taken over the
**same span**, from the stock's previous bar date to t, so a return spanning a halt is
compared with the market move over that same halt.
"""
from __future__ import annotations

import os
import sys
from typing import Final

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config  # noqa: E402

PANEL: Final[str] = "data/bhavcopy/panel_daily.parquet"
ENGINE_OHLCV: Final[str] = "data/nse500_daily_ohlcv.parquet"
INDEX_RAW_DIR: Final[str] = "data/bhavcopy/index"
OUT_DIR: Final[str] = "data/phase6"
F1_DIR: Final[str] = os.path.join(OUT_DIR, "f1")
INDEX_DAILY: Final[str] = os.path.join(OUT_DIR, "index_daily.parquet")


def sealed_after() -> pd.Timestamp:
    return pd.Timestamp(config.PHASE6_SEALED_AFTER)


def guard_holdout(dates, what: str, allow_holdout: bool = False) -> None:
    """Refuse any input dated after the sealed holdout start (docs/phase6_plan.md §B.1)."""
    if allow_holdout:
        return
    last = pd.Timestamp(pd.Series(dates).max())
    if last > sealed_after():
        raise RuntimeError(
            f"{what} reaches {last.date()}, past the sealed holdout boundary "
            f"{sealed_after().date()} (§B.1). Pass allow_holdout=True only at the holdout step."
        )


def load_panel(
    path: str = PANEL, engine_ohlcv: str = ENGINE_OHLCV, allow_holdout: bool = False
) -> pd.DataFrame:
    """
    The adjusted bhavcopy panel, restricted to the engine's tickers, in own-bar long form.

    Prices are already split/bonus back-adjusted by ``bhavcopy_panel.apply_adjustments``
    and match ``nse500_daily_ohlcv.parquet`` exactly. Adds each row's previous own-bar
    date and close.
    """
    cols = ["date", "ticker", "open", "close", "turnover", "trades"]
    panel = pd.read_parquet(path, columns=cols)
    guard_holdout(panel["date"], path, allow_holdout)
    tickers = pd.read_parquet(engine_ohlcv, columns=["ticker"])["ticker"].unique()
    panel = panel[panel["ticker"].isin(set(tickers))]
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = panel.groupby("ticker", sort=False)
    panel["prev_date"] = g["date"].shift(1)
    panel["prev_close"] = g["close"].shift(1)
    return panel


def span_logret(level: pd.Series, start: pd.Series, end: pd.Series) -> np.ndarray:
    """log(level[end] / level[start]) for paired date arrays; NaN where either is absent."""
    a = level.reindex(pd.DatetimeIndex(start)).to_numpy(float)
    b = level.reindex(pd.DatetimeIndex(end)).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(b / a)


def own_rolling_sum(values: pd.Series, tickers: pd.Series, window: int) -> pd.Series:
    """Per-ticker trailing sum over own bars (NaN counted as 0), aligned to ``values``."""
    out = (
        values.fillna(0.0).groupby(tickers.to_numpy(), sort=False)
        .rolling(window, min_periods=1).sum()
    )
    return out.reset_index(level=0, drop=True).reindex(values.index)


def own_rolling(values: pd.Series, tickers: pd.Series, window: int, op: str) -> pd.Series:
    """Per-ticker trailing reduction with a complete window, as feature_creator._roll."""
    out = getattr(
        values.groupby(tickers.to_numpy(), sort=False).rolling(window, min_periods=window),
        op,
    )()
    return out.reset_index(level=0, drop=True).reindex(values.index)


def to_wide(long: pd.DataFrame, value: str) -> pd.DataFrame:
    """Long (date, ticker) column -> wide (date × ticker)."""
    return long.pivot(index="date", columns="ticker", values=value).sort_index()


def write(frame: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frame.to_parquet(path)
    cover = float(frame.notna().to_numpy().mean())
    print(f"[phase6] wrote {path} | {frame.shape[0]:,} × {frame.shape[1]:,} | non-NaN {cover:.1%}")


def read(path: str, builder: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run `{builder}` first.")
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    return frame
