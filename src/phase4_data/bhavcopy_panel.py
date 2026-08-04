"""
Phase 4 panel construction — raw bhavcopy files → one adjusted long-format panel.

Turns the ~3,100 daily archive files that ``bhavcopy_download.py`` fetched into a
single ``(date, isin)`` table with split/bonus-adjusted OHLCV. Four problems are
solved here, in this order:

**(a) Identity.** The panel is keyed on **ISIN**, not symbol. A rename (INFOSYSTCH →
INFY) is then invisible — same ISIN, new symbol — and needs no mapping table. The
``ticker`` exported downstream is the *last* symbol observed for that ISIN, so a
renamed company is one continuous series rather than two half-length ones.

**(b) Series migration.** EQ **and** BE are kept. NSE's surveillance framework moves
names EQ→BE→EQ; dropping BE would make a live stock look delisted and then re-IPO'd,
manufacturing phantom survivorship events. Series is recorded per row.

**(c) Corporate actions — the hard part.** NSE publishes no corporate-action file on
the archive host, and ``PREVCLOSE`` is *not* the adjusted prior close (verified on the
NESTLEIND 1:10 split of 2024-01-05: both the naive and PREVCLOSE-based returns read
−90.2%). Splits and bonuses therefore have to be **detected from the data**, using
three independent signals that must agree:

    1. price jump   — close/close₋₁ leaves ±25%. NSE enforces daily price bands (20%
                      for most names, tighter for many), so for the liquid universe
                      this alone is already strong evidence: a genuine −40% *single-day*
                      close-to-close move is close to impossible under circuit limits.
    2. rational snap— the ratio lands near a simple fraction (1/2, 2/5, 1/10 …). The
                      observed ratio is not exactly 1/k, because the stock also moves
                      on its ex-date, so the tolerance admits a plausible ex-date return
                      and the *exact* fraction becomes the adjustment factor.
    3. turnover     — a k:1 split divides price by k and multiplies quantity by k, so
       invariance     rupee turnover survives roughly intact while quantity jumps
                      inversely; a genuine wipeout takes rupee turnover with it (a name
                      bleeding out on repeated lower circuits trades almost nothing).

Every detection is written to ``corporate_actions.csv`` with all three signal values —
an auditable ledger, not a black box. Adjustment is applied **backwards** (history is
rescaled before the ex-date), which leaves every past *return* unchanged; an assertion
enforces exactly that.

*Limitation, stated plainly:* ordinary dividends (0.5–3%) are far below any detection
threshold and NSE does not adjust for them, so this is a **price-return** panel, not
total return — the same basis as the existing 68-name Drive data.

**(d) Lifecycle.** The panel is survivorship-free by construction: a stock that died
in 2019 is present until its last trading day and then simply stops. That raggedness
needs no feature or tree change (``create_features`` already drops rows with any NaN
feature). What it *does* need is an honest exit price, so ``lifecycle.csv`` records
each ISIN's terminal event and return:

    delisted      absent ≥20 trading days before the panel ends, never returns
                  → −30% terminal return (Shumway 1997)
    acquired      the final 20-day return is positive → 0% exit, not −30%
    suspended     absent, then trades again → not a delisting. Left in the panel; the
                  universe filter's "≥200 of the trailing 252 days traded" rule keeps
                  a resumed name out until it has a clean history again, so no
                  separate gap machinery is needed.
    live          still trading at the panel end

Usage
    python src/phase4_data/bhavcopy_panel.py              # dry-run self-test
    python src/phase4_data/bhavcopy_panel.py --build      # build from data/bhavcopy/
    python src/phase4_data/bhavcopy_panel.py --build --limit 200   # first 200 days
"""

import argparse
import glob
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

DEFAULT_RAW_DIR: Final[str] = "data/bhavcopy"
DEFAULT_PANEL: Final[str] = "data/bhavcopy/panel_daily.parquet"
DEFAULT_ACTIONS_CSV: Final[str] = "data/bhavcopy/corporate_actions.csv"
DEFAULT_LIFECYCLE_CSV: Final[str] = "data/bhavcopy/lifecycle.csv"
DEFAULT_INDEX_PARQUET: Final[str] = "data/bhavcopy/index_daily.parquet"

# Common equity only. Indian ISINs are prefixed by issuer class: INE = company equity,
# INF = mutual fund / ETF, IN0/IN9 = government and other paper. Filtering on the
# prefix is era-independent and exact, where a series filter alone would still admit
# ETFs (which trade under series EQ).
_EQUITY_ISIN_PREFIX: Final[str] = "INE"
_KEEP_SERIES: Final[tuple[str, ...]] = ("EQ", "BE")

PANEL_COLUMNS: Final[list[str]] = [
    "date", "isin", "symbol", "series",
    "open", "high", "low", "close", "close_vwap", "volume", "turnover", "trades",
]

# An Indian ISIN is INE + a 4-character issuer code + a 5-digit security serial:
#   INE239A01016  ->  issuer "INE239A", security "01016"
# The issuer stem is stable for the life of the company; the serial changes when the
# security itself is reissued. See link_isins().
_ISIN_STEM: Final[int] = 7


@dataclass(frozen=True)
class ActionConfig:
    """
    Corporate-action detection thresholds.

    Defaults are calibrated against ``bhavcopy_validate.py`` gate 1 — reproducing the
    existing 68-name panel's daily returns to >0.999 correlation. That is ground truth
    about *prices*, never about strategy performance, so tuning here cannot leak
    selection bias into the backtest.

    **The two error modes are not symmetric, and the thresholds lean accordingly.**
    A *missed* action leaves a fake ±50–90% return in the panel, which becomes a
    training label the trees will happily learn from — it corrupts the result. A
    *mis-snapped* action (right ex-date, slightly wrong factor) only mis-states the
    single ex-date bar's own return: back-adjustment rescales a level, so every other
    return in that name's history stays exact regardless of which factor was chosen.
    Recall therefore matters far more than factor precision, which is why the ratio
    grid stays dense and ``snap_tol`` stays generous enough to absorb a real ex-date
    move rather than rejecting the action outright.
    """
    jump: float = 0.25          # |close/close₋₁ − 1| must exceed this
    snap_tol: float = 0.10      # implied ex-date return allowed around the exact ratio
    review_tol: float = 0.05    # above this implied return, flag the row for eyeballing
    turnover_lo: float = 0.20   # rupee turnover vs trailing median: floor …
    turnover_hi: float = 5.00   # … and ceiling. A split leaves turnover roughly intact.
    turnover_window: int = 20
    min_history: int = 5        # skip the first days of a listing (no stable baseline)
    max_gap_days: int = 7       # ignore jumps measured across a break in trading
    min_price: float = 20.0     # below this the tick grid fakes rational ratios (see below)


# Price multipliers a corporate action can actually produce.
#
# Splits and bonuses only ever *reduce* the price (a 1:k split gives 1/k; an m:n bonus —
# m new shares per n held — gives n/(n+m)), so the sub-1 side of the grid is dense and
# includes the small ratios like 2/3 and 3/4 that ordinary bonus issues generate.
#
# The above-1 side is deliberately sparse. A factor > 1 is a reverse split, which in
# India is rare and always drastic — a company consolidating does 10:1, never 3:2. Left
# dense, the region just above 1 would sit right on the ±25% jump threshold and collect
# false positives from ordinary volatility, so nothing between 1 and 2 is admitted.
_MAX_BONUS: int = 12


def _rational_grid() -> np.ndarray:
    """Plausible price-multipliers for a corporate action, ascending and de-duplicated."""
    vals: set[float] = set()
    for k in (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 50, 100):
        vals.add(1.0 / k)                             # 1:k split
        vals.add(float(k))                            # k:1 reverse split (sparse, ≥2)
    for m in range(1, _MAX_BONUS + 1):                # m:n bonus → n/(n+m)
        for n in range(1, _MAX_BONUS + 1):
            vals.add(n / (n + m))
    return np.array(sorted(v for v in vals if 0.005 <= v <= 200.0 and (v <= 1.0 or v >= 2.0)))


_RATIO_GRID: Final[np.ndarray] = _rational_grid()


# ── Raw file parsing ────────────────────────────────────────────────────────────

_EQ_NAME_RE: Final[re.Pattern] = re.compile(
    r"^cm(\d{2})([A-Z]{3})(\d{4})bhav\.csv\.zip$", re.IGNORECASE
)
_UDIFF_NAME_RE: Final[re.Pattern] = re.compile(
    r"^BhavCopy_NSE_CM_0_0_0_(\d{8})_F_0000\.csv\.zip$", re.IGNORECASE
)


def _date_from_filename(path: str) -> pd.Timestamp | None:
    """
    Bar date from the archive filename — the authoritative, format-stable source.

    Preferred over the file's own date column, which drifts (one 2020 legacy file
    writes a 2-digit year) and would otherwise cost a whole trading day to a silent
    coerce-to-NaT.
    """
    name = os.path.basename(path)
    m = _EQ_NAME_RE.match(name)
    if m:
        return pd.to_datetime(f"{m.group(1)}{m.group(2).upper()}{m.group(3)}",
                              format="%d%b%Y")
    m = _UDIFF_NAME_RE.match(name)
    if m:
        return pd.to_datetime(m.group(1), format="%Y%m%d")
    return None


def _read_zip_csv(path: str) -> pd.DataFrame | None:
    """Read the single CSV inside a bhavcopy zip; None if the archive is unusable."""
    try:
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not names:
                return None
            with z.open(names[0]) as fh:
                return pd.read_csv(fh, low_memory=False)
    except (zipfile.BadZipFile, OSError, pd.errors.ParserError):
        return None


def parse_old(df: pd.DataFrame, file_date: pd.Timestamp) -> pd.DataFrame:
    """
    Legacy bhavcopy → canonical columns.

    The bar date comes from the **filename**, not the ``TIMESTAMP`` column. That column
    is not format-stable across the archive — ``cm13JUL2020bhav.csv.zip`` writes
    ``13-Jul-20`` where every other file writes ``02-JAN-2013`` — and a strict parse with
    ``errors="coerce"`` would silently turn the whole day into NaT and drop it. The
    filename encodes the same date and is uniform across 12 years.
    """
    out = pd.DataFrame({
        "date": file_date,
        "isin": df["ISIN"].astype("string").str.strip(),
        "symbol": df["SYMBOL"].astype("string").str.strip(),
        "series": df["SERIES"].astype("string").str.strip(),
        "open": pd.to_numeric(df["OPEN"], errors="coerce"),
        "high": pd.to_numeric(df["HIGH"], errors="coerce"),
        "low": pd.to_numeric(df["LOW"], errors="coerce"),
        # close = last traded price. NSE's CLOSE field is the VWAP of the final 30
        # minutes (the official settlement price), which is a *different* series: it
        # differs from the LTP by ~20 bps of daily return. The existing verified
        # 68-name panel is LTP-based, so using LAST keeps the 68 → 500 comparison a
        # pure test of breadth rather than confounding it with a price-definition
        # change. The official close is kept alongside for a future sensitivity run.
        "close": pd.to_numeric(df["LAST"], errors="coerce"),
        "close_vwap": pd.to_numeric(df["CLOSE"], errors="coerce"),
        "volume": pd.to_numeric(df["TOTTRDQTY"], errors="coerce"),
        "turnover": pd.to_numeric(df["TOTTRDVAL"], errors="coerce"),
        "trades": pd.to_numeric(df.get("TOTALTRADES"), errors="coerce"),
    })
    return out


def parse_udiff(df: pd.DataFrame, file_date: pd.Timestamp) -> pd.DataFrame:
    """
    UDiFF bhavcopy → canonical columns.

    UDiFF carries the whole CM segment, so instruments are filtered to ``STK`` here;
    the legacy format needs no equivalent because the ISIN-prefix filter handles it.
    ``LastPric``/``ClsPric`` are the UDiFF spellings of LAST/CLOSE — see parse_old for
    why the last traded price is the one that becomes ``close``.
    """
    if "FinInstrmTp" in df.columns:
        df = df[df["FinInstrmTp"].astype("string").str.strip().eq("STK")]
    out = pd.DataFrame({
        "date": file_date,
        "isin": df["ISIN"].astype("string").str.strip(),
        "symbol": df["TckrSymb"].astype("string").str.strip(),
        "series": df["SctySrs"].astype("string").str.strip(),
        "open": pd.to_numeric(df["OpnPric"], errors="coerce"),
        "high": pd.to_numeric(df["HghPric"], errors="coerce"),
        "low": pd.to_numeric(df["LwPric"], errors="coerce"),
        "close": pd.to_numeric(df["LastPric"], errors="coerce"),
        "close_vwap": pd.to_numeric(df["ClsPric"], errors="coerce"),
        "volume": pd.to_numeric(df["TtlTradgVol"], errors="coerce"),
        "turnover": pd.to_numeric(df["TtlTrfVal"], errors="coerce"),
        "trades": pd.to_numeric(df.get("TtlNbOfTxsExctd"), errors="coerce"),
    })
    return out


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the equity/series/validity filters common to both eras."""
    df = df[
        df["isin"].notna()
        & df["isin"].str.startswith(_EQUITY_ISIN_PREFIX, na=False)
        & df["series"].isin(_KEEP_SERIES)
        & df["date"].notna()
        & df["close"].gt(0)
        & df["volume"].ge(0)
    ]
    return df[PANEL_COLUMNS]


def link_isins(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Stitch a company's price history back together across ISIN reissues.

    **This is the single most consequential correction in the build.** NSE issues a
    *new ISIN* when a company changes the face value of its shares — which is exactly
    what a split does:

        NESTLEIND  INE239A01016 → INE239A01024  on the 1:10 split of 2024-01-05
        BEL        INE263A01016 → INE263A01024  on the split of 2017-03-17

    Keyed on raw ISIN, three things break at once and all of them flatter the backtest
    or corrupt it: the split becomes **structurally undetectable** (there is no prior
    bar inside the new ISIN's group, so no ratio is ever computed), the old ISIN looks
    like a **delisting** and books a −30% exit that never happened, and the new one
    looks like a **fresh IPO** whose rolling features restart from nothing.

    The issuer stem (``INE239A``) is stable across the reissue, so it does the linking.
    Two securities sharing a stem are only merged when their date ranges are
    **disjoint** — a genuine reissue hands off on consecutive trading days. Ranges that
    *overlap* mean two securities that co-existed (a second share class, a partly-paid
    line), and those are deliberately left separate.
    """
    out = panel.copy()
    out["entity"] = out["isin"]

    spans = out.groupby("isin")["date"].agg(["min", "max"])
    spans["stem"] = spans.index.str[:_ISIN_STEM]

    n_merged = n_overlap = 0
    for stem, grp in spans.groupby("stem"):
        if len(grp) < 2:
            continue
        grp = grp.sort_values("min")
        # Overlap check: each successive listing must start after the previous ends.
        starts, ends = grp["min"].to_numpy(), grp["max"].to_numpy()
        if (starts[1:] <= ends[:-1]).any():
            n_overlap += 1
            continue
        canonical = grp.index[0]          # earliest listing keeps the identity
        out.loc[out["isin"].isin(grp.index), "entity"] = canonical
        n_merged += len(grp) - 1

    print(
        f"[panel] ISIN linking: {n_merged} reissue(s) merged into their predecessor | "
        f"{n_overlap} stem(s) left split (overlapping date ranges = distinct securities)"
    )
    return out


def load_raw(raw_dir: str = DEFAULT_RAW_DIR, limit: int = 0) -> pd.DataFrame:
    """
    Parse every downloaded bhavcopy into one long frame (unadjusted).

    Both era directories are read and stacked; each file contributes one trading day.
    Unreadable archives are counted and reported rather than aborting the build — a
    single corrupt download should not cost a 12-year panel.
    """
    jobs: list[tuple[str, str]] = []
    for era, parser in (("eq", "old"), ("udiff", "udiff")):
        paths = sorted(glob.glob(os.path.join(raw_dir, era, "*.zip")))
        jobs.extend((p, parser) for p in paths)
    if not jobs:
        raise FileNotFoundError(
            f"No bhavcopy zips under {raw_dir!r}. Run bhavcopy_download.py first."
        )
    if limit > 0:
        jobs = jobs[:limit]

    frames: list[pd.DataFrame] = []
    n_bad = 0
    for i, (path, kind) in enumerate(jobs, start=1):
        file_date = _date_from_filename(path)
        if file_date is None:
            n_bad += 1
            print(f"[panel] UNPARSEABLE NAME {os.path.basename(path)}")
            continue
        raw = _read_zip_csv(path)
        if raw is None or raw.empty:
            n_bad += 1
            print(f"[panel] BAD ARCHIVE {os.path.basename(path)}")
            continue
        try:
            one = (parse_old(raw, file_date) if kind == "old"
                   else parse_udiff(raw, file_date))
        except KeyError as exc:
            n_bad += 1
            print(f"[panel] SCHEMA MISS {os.path.basename(path)} — missing {exc}")
            continue
        frames.append(_clean(one))
        if i % 250 == 0 or i == len(jobs):
            print(f"[panel] parsed {i}/{len(jobs)} files")

    if not frames:
        raise RuntimeError("No usable bhavcopy files parsed.")

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["date", "isin"], keep="last")
    out = out.sort_values(["isin", "date"]).reset_index(drop=True)
    print(
        f"[panel] raw: rows={len(out):,} isins={out['isin'].nunique():,} "
        f"days={out['date'].nunique():,} bad_files={n_bad}"
    )
    return out


# ── Corporate actions ───────────────────────────────────────────────────────────

def _snap(ratio: pd.Series, tol: float) -> pd.Series:
    """
    Nearest simple fraction to each observed ratio, or NaN if none is within ``tol``.

    The observed ratio is **not** the clean 1/k — the stock also moves on its ex-date,
    so a 1:10 split reads 0.0983 rather than 0.1000 (measured on NESTLEIND, 2024-01-05).
    Tolerance is therefore sized to admit a plausible ex-date return, and the *exact*
    fraction is what gets returned: adjusting by the exact 1/k preserves the genuine
    ex-date return, where adjusting by the raw ratio would silently erase it.
    """
    vals = ratio.to_numpy(dtype=float)
    out = np.full(vals.shape, np.nan)
    ok = np.isfinite(vals) & (vals > 0)
    if ok.any():
        idx = np.searchsorted(_RATIO_GRID, vals[ok]).clip(1, len(_RATIO_GRID) - 1)
        lo, hi = _RATIO_GRID[idx - 1], _RATIO_GRID[idx]
        # Nearest in log space — the grid spans 0.005 to 200, so absolute distance
        # would bias every choice toward the dense low end.
        cand = np.where(np.abs(np.log(vals[ok] / lo)) <= np.abs(np.log(vals[ok] / hi)), lo, hi)
        out[ok] = np.where(np.abs(vals[ok] / cand - 1.0) <= tol, cand, np.nan)
    return pd.Series(out, index=ratio.index)


def detect_corporate_actions(
    panel: pd.DataFrame, cfg: ActionConfig = ActionConfig()
) -> pd.DataFrame:
    """
    Find split/bonus ex-dates and their price multipliers.

    Returns a ledger with one row per detection, carrying every signal value so the
    decision can be audited by hand against public split records:
    ``date, isin, symbol, factor, price_ratio, snapped, turnover_ratio, volume_ratio``.

    ``factor`` is the multiplier applied to *prior* prices to make the series
    continuous — i.e. 0.1 for a 1:10 split.
    """
    df = panel.sort_values(["entity", "date"]).copy()
    g = df.groupby("entity", sort=False)

    df["_prev_close"] = g["close"].shift(1)
    df["_ratio"] = df["close"] / df["_prev_close"]
    df["_n_prior"] = g.cumcount()

    # Trailing baselines exclude the candidate day itself (shift(1)) so an ex-date's
    # own volume spike cannot inflate its own reference level.
    med_to = g["turnover"].transform(
        lambda s: s.shift(1).rolling(cfg.turnover_window, min_periods=cfg.min_history).median()
    )
    med_vol = g["volume"].transform(
        lambda s: s.shift(1).rolling(cfg.turnover_window, min_periods=cfg.min_history).median()
    )
    df["_to_ratio"] = df["turnover"] / med_to.replace(0, np.nan)
    df["_vol_ratio"] = df["volume"] / med_vol.replace(0, np.nan)

    # A price ratio only means anything when the two bars are adjacent in trading time.
    # Across a suspension — or a hole in the downloaded archive — the "jump" is just the
    # accumulated move over the break, and treating it as a split would rewrite the whole
    # prior history by a bogus factor. Requiring consecutive bars makes the detector
    # robust to both, and is what keeps a partial download from corrupting a rebuild.
    df["_gap_days"] = (df["date"] - g["date"].shift(1)).dt.days
    contiguous = df["_gap_days"].le(cfg.max_gap_days)

    # Penny stocks are where this detector goes wrong. NSE's tick is Rs 0.05, so a stock
    # at Rs 2.00 moves to Rs 3.00 and lands on *exactly* 3/2 — the tick grid manufactures
    # clean rational ratios by accident, and the snap test cannot tell them from a real
    # bonus. Requiring a meaningful price level removes that whole failure mode, and it
    # cannot mask a genuine forward action: a company does not split or issue a bonus on
    # a share already trading near its face value. Sub-Rs-20 names that are genuinely
    # liquid (IDEA, YESBANK, SUZLON) consolidate rather than split, and consolidations
    # are large enough to stay well clear of this floor's reach.
    priced = df["_prev_close"] >= cfg.min_price

    jumped = (df["_ratio"] - 1.0).abs() > cfg.jump
    seasoned = df["_n_prior"] >= cfg.min_history
    df["_snap"] = np.nan
    cand = jumped & seasoned & contiguous & priced & df["_ratio"].notna()
    if cand.any():
        df.loc[cand, "_snap"] = _snap(df.loc[cand, "_ratio"], cfg.snap_tol)

    # Turnover invariance: a split rescales price and quantity inversely, so rupee
    # turnover survives. A genuine −50% move takes rupee turnover with it (or spikes
    # it far past the ceiling on panic volume). Missing baseline → treated as passing,
    # since the snap test alone is already a strong filter for a thin-history name.
    to_ok = df["_to_ratio"].between(cfg.turnover_lo, cfg.turnover_hi) | df["_to_ratio"].isna()

    hit = cand & df["_snap"].notna() & to_ok
    ledger = (
        df.loc[hit, ["date", "entity", "isin", "symbol", "_snap", "_ratio", "_to_ratio", "_vol_ratio"]]
        .rename(columns={
            "_snap": "factor", "_ratio": "price_ratio",
            "_to_ratio": "turnover_ratio", "_vol_ratio": "volume_ratio",
        })
        .sort_values(["date", "entity"])
        .reset_index(drop=True)
    )
    # Residual between the observed ratio and the exact fraction — this is the stock's
    # genuine return on its ex-date, and it stays in the series after adjustment. A
    # large residual means the snap is doubtful, so it is flagged rather than hidden.
    ledger["implied_ex_ret"] = ledger["price_ratio"] / ledger["factor"] - 1.0
    ledger["review"] = ledger["implied_ex_ret"].abs() > cfg.review_tol

    # The funnel is printed in full so the detector can be audited from the log alone:
    # each stage should shed candidates, and a stage that sheds almost none means its
    # threshold is doing no work.
    print(
        f"[panel] corporate-action funnel: "
        f"{int(jumped.sum()):,} jumps > {cfg.jump:.0%} "
        f"→ {int((jumped & contiguous).sum()):,} contiguous "
        f"→ {int(cand.sum()):,} priced ≥ {cfg.min_price:.0f} & seasoned "
        f"→ {int((cand & df['_snap'].notna()).sum()):,} snapped "
        f"→ {len(ledger):,} turnover-invariant "
        f"({int(ledger['review'].sum()) if len(ledger) else 0} flagged for review)"
    )
    return ledger


def apply_adjustments(panel: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """
    Back-adjust prices and volumes so the series is continuous across each ex-date.

    History *before* an ex-date is multiplied by the action's factor (prices) and
    divided by it (volumes), which leaves every historical return and every rupee
    turnover unchanged — only the level moves. Multiple actions compound via a reverse
    cumulative product, so a name with three splits is handled in one pass.
    """
    out = panel.sort_values(["entity", "date"]).copy()
    if ledger.empty:
        out["adj_factor"] = 1.0
        return out

    # Mark each ex-date row with its factor, then take, for every row, the product of
    # all factors at *strictly later* ex-dates: that is this row's adjustment.
    key = pd.MultiIndex.from_frame(out[["entity", "date"]])
    fac = pd.Series(1.0, index=key)
    ex = pd.MultiIndex.from_frame(ledger[["entity", "date"]])
    fac.loc[ex] = ledger["factor"].to_numpy()
    fac.index = out.index

    grp = out.groupby("entity", sort=False)
    # cumprod of factors from the end, excluding the current row: shift(-1) within group.
    rev = fac.iloc[::-1].groupby(out["entity"].iloc[::-1], sort=False).cumprod().iloc[::-1]
    out["adj_factor"] = (rev / fac).astype(float)

    for col in ("open", "high", "low", "close", "close_vwap"):
        if col in out.columns:
            out[col] = out[col] * out["adj_factor"]
    out["volume"] = out["volume"] / out["adj_factor"]
    return out


def assert_returns_preserved(
    raw: pd.DataFrame, adj: pd.DataFrame, ledger: pd.DataFrame, tol: float = 1e-9
) -> None:
    """
    Back-adjustment must not alter any return except across the ex-dates themselves.

    This is the invariant that makes adjustment safe: rescaling a level is information-
    free, so a bug that shifts historical returns would be a real look-ahead-grade
    defect. Checked on every non-ex-date bar in the panel.
    """
    a = raw.sort_values(["entity", "date"]).set_index(["entity", "date"])["close"]
    b = adj.sort_values(["entity", "date"]).set_index(["entity", "date"])["close"]
    ra = a.groupby(level="entity").pct_change()
    rb = b.groupby(level="entity").pct_change()
    if not ledger.empty:
        ex = pd.MultiIndex.from_frame(ledger[["entity", "date"]])
        mask = ~ra.index.isin(ex)
    else:
        mask = np.ones(len(ra), dtype=bool)
    diff = (ra[mask] - rb[mask]).abs()
    worst = float(diff.max(skipna=True) or 0.0)
    assert worst < tol, f"back-adjustment changed a historical return by {worst:.2e}"
    print(f"[panel] return-preservation OK (max |Δ| = {worst:.2e})")


# ── Lifecycle ───────────────────────────────────────────────────────────────────

def classify_lifecycle(
    panel: pd.DataFrame, gap_days: int = 20, delist_return: float = -0.30
) -> pd.DataFrame:
    """
    Per-ISIN terminal event and exit return, consumed by the Tier 3 delisting fix.

    A name whose last bar precedes the panel end by more than ``gap_days`` trading days
    has left the exchange. Distinguishing a wipeout from a buyout without a corporate-
    action feed is done on the only evidence available — the final 20-day return: an
    acquisition trades *up* into the offer price, a failure bleeds out. Suspensions
    (absent then trading again) are labelled but not truncated; the universe filter's
    trailing-liquidity rule already excludes a name until its history is clean.
    """
    cal = np.sort(panel["date"].unique())
    pos = pd.Series(np.arange(len(cal)), index=cal)
    end_pos = len(cal) - 1

    df = panel.sort_values(["entity", "date"])
    g = df.groupby("entity", sort=False)

    last_date = g["date"].max()
    first_date = g["date"].min()
    n_bars = g.size()
    last_symbol = g["symbol"].last()

    # 20-bar exit return, on the adjusted close.
    def _exit_ret(s: pd.Series) -> float:
        s = s.dropna()
        if len(s) < 2:
            return np.nan
        base = s.iloc[max(0, len(s) - 21)]
        return float(s.iloc[-1] / base - 1.0) if base > 0 else np.nan

    exit_ret = g["close"].apply(_exit_ret)

    # Suspension = traded days materially short of the calendar the name spanned.
    span = pos.reindex(last_date.values).to_numpy() - pos.reindex(first_date.values).to_numpy() + 1
    suspended = (span - n_bars.to_numpy()) >= gap_days

    gone = pos.reindex(last_date.values).to_numpy() <= (end_pos - gap_days)

    kind = np.where(
        ~gone, "live",
        np.where(exit_ret.to_numpy() > 0, "acquired", "delisted"),
    )
    terminal = np.where(
        kind == "delisted", delist_return, 0.0,
    )
    terminal = np.where(kind == "live", np.nan, terminal)

    out = pd.DataFrame({
        "entity": last_date.index,
        "ticker": last_symbol.to_numpy(),
        "first_date": first_date.to_numpy(),
        "last_date": last_date.to_numpy(),
        "n_bars": n_bars.to_numpy(),
        "exit_ret_20b": exit_ret.to_numpy(),
        "suspended": suspended,
        "kind": kind,
        "terminal_return": terminal,
    }).reset_index(drop=True)

    counts = out["kind"].value_counts().to_dict()
    print(
        f"[panel] lifecycle: {counts} | suspensions={int(out['suspended'].sum())}"
    )
    return out


# ── Index series ────────────────────────────────────────────────────────────────

# The benchmark renamed itself twice over the window; all three labels are the same
# series and are matched interchangeably.
_NIFTY_ALIASES: Final[tuple[str, ...]] = ("S&P CNX NIFTY", "CNX NIFTY", "NIFTY 50")


INDEX_TICKER: Final[str] = "NIFTY-50"

_INDEX_NAME_RE: Final[re.Pattern] = re.compile(
    r"^ind_close_all_(\d{8})\.csv$", re.IGNORECASE
)


def _index_date_from_filename(path: str) -> pd.Timestamp | None:
    """Session date from an index filename (``ind_close_all_DDMMYYYY.csv``)."""
    m = _INDEX_NAME_RE.match(os.path.basename(path))
    return pd.to_datetime(m.group(1), format="%d%m%Y") if m else None


def build_index(raw_dir: str = DEFAULT_RAW_DIR, limit: int = 0) -> pd.DataFrame:
    """
    Stack the daily all-index CSVs into the Tier 2 market series.

    Emitted in the *existing* index-parquet schema (``timestamp, ticker, open, high,
    low, close, volume``) with ticker ``NIFTY-50``, so ``tier2_regime._load_index_daily``
    consumes it unchanged — repointing ``_RegimeConfig.index_parquet`` is the whole
    integration. Bhavcopy is daily, so the "session close = last bar of the day"
    resampling there collapses to the identity.
    """
    paths = sorted(glob.glob(os.path.join(raw_dir, "index", "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No index CSVs under {raw_dir}/index.")
    if limit > 0:
        paths = paths[:limit]

    rows: list[dict] = []
    n_bad = 0
    for i, path in enumerate(paths, start=1):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError):
            n_bad += 1
            continue
        if "Index Name" not in df.columns:
            n_bad += 1
            continue
        name = df["Index Name"].astype("string").str.strip().str.upper()
        hit = df[name.isin(_NIFTY_ALIASES)]
        if hit.empty:
            n_bad += 1
            continue
        r = hit.iloc[0]
        rows.append({
            # Date from the filename, not the "Index Date" column. That column drifts
            # format mid-archive — 2014-2015 files write 09/06/2014 where every other
            # era writes 02-01-2013 — and a strict parse silently dropped 62 days of
            # the market series. Exactly the same defect as the price files' TIMESTAMP.
            "timestamp": _index_date_from_filename(path),
            "open": pd.to_numeric(r.get("Open Index Value"), errors="coerce"),
            "high": pd.to_numeric(r.get("High Index Value"), errors="coerce"),
            "low": pd.to_numeric(r.get("Low Index Value"), errors="coerce"),
            "close": pd.to_numeric(r["Closing Index Value"], errors="coerce"),
        })
        if i % 500 == 0 or i == len(paths):
            print(f"[panel] index {i}/{len(paths)} files")

    out = (
        pd.DataFrame(rows)
        .dropna(subset=["timestamp", "close"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .assign(ticker=INDEX_TICKER, volume=0.0)
        .reset_index(drop=True)
    )[["timestamp", "ticker", "open", "high", "low", "close", "volume"]]

    print(f"[panel] index: {len(out):,} days {out['timestamp'].min().date()} → "
          f"{out['timestamp'].max().date()} | unmatched_files={n_bad}")
    return out


# ── Build ───────────────────────────────────────────────────────────────────────

def build(
    raw_dir: str = DEFAULT_RAW_DIR,
    out_panel: str = DEFAULT_PANEL,
    out_actions: str = DEFAULT_ACTIONS_CSV,
    out_lifecycle: str = DEFAULT_LIFECYCLE_CSV,
    out_index: str = DEFAULT_INDEX_PARQUET,
    cfg: ActionConfig = ActionConfig(),
    limit: int = 0,
) -> pd.DataFrame:
    """Run the whole panel build and write every artefact. Returns the adjusted panel."""
    raw = load_raw(raw_dir, limit=limit)
    # Must precede detection: a face-value split reissues the ISIN, so without linking
    # there is no prior bar to measure the split against and it is invisible.
    raw = link_isins(raw)
    ledger = detect_corporate_actions(raw, cfg)
    adj = apply_adjustments(raw, ledger)
    assert_returns_preserved(raw, adj, ledger)

    # Canonical display ticker = the last symbol this entity ever traded under, so a
    # rename *or* an ISIN reissue reads as one continuous series downstream.
    adj["ticker"] = adj["entity"].map(adj.groupby("entity")["symbol"].last())

    life = classify_lifecycle(adj)

    for path in (out_panel, out_actions, out_lifecycle, out_index):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    adj.to_parquet(out_panel, index=False)
    ledger.to_csv(out_actions, index=False)
    life.to_csv(out_lifecycle, index=False)
    build_index(raw_dir, limit=limit).to_parquet(out_index, index=False)

    back = pd.read_parquet(out_panel)
    assert back.shape == adj.shape, "panel read-back shape mismatch"

    print(
        f"\n[panel] wrote {out_panel} | rows={len(adj):,} "
        f"tickers={adj['ticker'].nunique():,} "
        f"{adj['date'].min().date()} → {adj['date'].max().date()}"
    )
    return adj


# ── Dry run ─────────────────────────────────────────────────────────────────────

def _dry_run() -> None:
    """
    Self-test on a synthesized panel covering every failure mode found on real data.

    Four planted cases: a clean 1:10 split; a genuine −90% wipeout that must *not* be
    flagged (the point of the turnover-invariance signal); a quiet series; and a split
    that **reissues the ISIN mid-series**, which is how NESTLEIND and BEL evaded
    detection entirely in the first real build.
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=120)
    rows = []

    plan = (
        ("INE000A01016", "SPLITCO", "split"),
        ("INE000A01024", "CRASHCO", "crash"),
        ("INE000A01032", "QUIETCO", "quiet"),
        ("INE111B01016", "REISSUE", "reissue"),   # ISIN changes at the ex-date
    )
    for isin, sym, kind in plan:
        px = 1000.0 * np.exp(np.cumsum(rng.normal(0, 0.012, len(dates))))
        qty = rng.lognormal(11, 0.3, len(dates))
        if kind in ("split", "reissue"):          # 1:10 split at bar 60
            px[60:] /= 10.0
            qty[60:] *= 10.0                      # turnover invariant, as in reality
        if kind == "crash":                       # genuine −90% with turnover collapse
            px[60:] *= 0.10
            qty[60:] *= 0.05
        frame = pd.DataFrame({
            "date": dates, "isin": isin, "symbol": sym, "series": "EQ",
            "open": px, "high": px * 1.01, "low": px * 0.99,
            "close": px, "close_vwap": px,
            "volume": qty, "turnover": px * qty, "trades": 1000.0,
        })
        if kind == "reissue":
            # Same issuer stem, new security serial, handing over on consecutive bars.
            frame.loc[60:, "isin"] = "INE111B01024"
        rows.append(frame)

    panel = link_isins(pd.concat(rows, ignore_index=True))

    # Linking must merge the reissue and leave the three unrelated stems alone.
    assert panel.loc[panel["symbol"] == "REISSUE", "entity"].nunique() == 1, \
        "ISIN reissue was not linked into one entity"
    assert panel.loc[panel["symbol"] == "SPLITCO", "entity"].nunique() == 1

    ledger = detect_corporate_actions(panel)
    found = set(zip(ledger["symbol"], ledger["date"].dt.date))

    ex = dates[60].date()
    assert ("SPLITCO", ex) in found, f"missed the planted split; ledger={found}"
    assert ("REISSUE", ex) in found, \
        f"missed the split hidden behind an ISIN reissue; ledger={found}"
    assert not any(s == "CRASHCO" for s, _ in found), \
        f"false positive: flagged a genuine crash as a corporate action; ledger={found}"
    assert not any(s == "QUIETCO" for s, _ in found), "false positive on a quiet series"

    for sym in ("SPLITCO", "REISSUE"):
        fac = float(ledger.loc[ledger["symbol"] == sym, "factor"].iloc[0])
        assert abs(fac - 0.1) < 1e-6, f"{sym}: wrong factor {fac}, expected 0.1"

    adj = apply_adjustments(panel, ledger)
    assert_returns_preserved(panel, adj, ledger)

    # Post-adjustment the split must be gone: the ex-date return is now ordinary.
    for sym in ("SPLITCO", "REISSUE"):
        s = adj[adj["symbol"] == sym].sort_values("date")["close"]
        ex_ret = float(s.pct_change().iloc[60])
        assert abs(ex_ret) < 0.10, f"{sym}: split survived adjustment ({ex_ret:.3f})"

    life = classify_lifecycle(adj)
    assert set(life["kind"]) == {"live"}, f"unexpected lifecycle: {life['kind'].tolist()}"
    # The reissued name must be ONE live entity, not a delisting plus a new listing.
    assert len(life) == len(plan), f"lifecycle split an entity: {len(life)} rows"

    print("\n[panel] dry-run OK — splits caught (incl. behind an ISIN reissue), "
          "crash rejected, returns preserved")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the adjusted bhavcopy panel.")
    ap.add_argument("--build", action="store_true", help="build from raw files (else dry-run)")
    ap.add_argument("--raw", default=DEFAULT_RAW_DIR, help="raw archive directory")
    ap.add_argument("--out", default=DEFAULT_PANEL, help="output panel Parquet")
    ap.add_argument("--limit", type=int, default=0, help="use at most N raw files (smoke)")
    ap.add_argument("--jump", type=float, default=ActionConfig.jump, help="price-jump threshold")
    ap.add_argument("--snap-tol", type=float, default=ActionConfig.snap_tol, help="snap tolerance")
    args = ap.parse_args()

    if not args.build:
        _dry_run()
        return

    build(
        raw_dir=args.raw, out_panel=args.out, limit=args.limit,
        cfg=ActionConfig(jump=args.jump, snap_tol=args.snap_tol),
    )


if __name__ == "__main__":
    main()
