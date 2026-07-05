"""
Phase 2 data ingestion.  Loads local OHLCV CSVs and enriches them with
NSE institutional delivery metrics fetched via jugaad-data's full bhavcopy API.

Full bhavcopy endpoint (jugaad-data >= 2.x):
    NSEArchives.full_bhavcopy_raw(dt) → CSV string
    URL: nsearchives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv
    Available from 2020-01-01 onward; raises ReadTimeout for earlier dates.

Expected local CSV schemas
    daily_ohlcv.csv  : columns include 'date', 'ticker', then OHLCV fields
    15min_ohlcv.csv  : columns include 'datetime', 'ticker', then OHLCV fields
"""

import io
from datetime import date, timedelta
from typing import Tuple

import pandas as pd

import config

# --- Column name constants for local CSVs ---------------------------------- #
_DAILY_DATE_COL: str = "date"
_DAILY_TICKER_COL: str = "ticker"
_15MIN_DATETIME_COL: str = "datetime"

# --- Full bhavcopy CSV column names (NSE sec_bhavdata_full_*.csv) ---------- #
_BHAV_SYMBOL_COL: str = "SYMBOL"
_BHAV_SERIES_COL: str = "SERIES"
_BHAV_DELIV_QTY_COL: str = "DELIV_QTY"
_BHAV_DELIV_PER_COL: str = "DELIV_PER"

# --- Output column names added to the augmented DataFrame ------------------ #
DELIVERY_QTY_COL: str = "DeliveryQty"
DELIVERY_PCT_COL: str = "DeliveryPct"


def load_local_csv_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads the two local OHLCV CSVs defined in config into pandas DataFrames.

    Daily data is returned with a MultiIndex of ('date', 'ticker'), sorted
    lexicographically.  15-min data uses a DatetimeIndex on the 'datetime'
    column with 'ticker' left as a regular column.

    Returns:
        Tuple of (daily_df, intraday_df):
          - daily_df:    MultiIndex ('date', 'ticker') DataFrame.
          - intraday_df: DatetimeIndex DataFrame with 'ticker' column intact.
    """
    # --- Daily OHLCV -------------------------------------------------------- #
    daily_df: pd.DataFrame = pd.read_csv(
        config.DATA_DAILY_CSV,
        parse_dates=[_DAILY_DATE_COL],
    )
    daily_df[_DAILY_DATE_COL] = pd.to_datetime(daily_df[_DAILY_DATE_COL])
    daily_df = (
        daily_df
        .set_index([_DAILY_DATE_COL, _DAILY_TICKER_COL])
        .rename_axis(index=["date", "ticker"])
        .sort_index()
    )

    # --- 15-minute OHLCV ---------------------------------------------------- #
    intraday_df: pd.DataFrame = pd.read_csv(
        config.DATA_15MIN_CSV,
        parse_dates=[_15MIN_DATETIME_COL],
    )
    intraday_df[_15MIN_DATETIME_COL] = pd.to_datetime(intraday_df[_15MIN_DATETIME_COL])
    intraday_df = intraday_df.set_index(_15MIN_DATETIME_COL).sort_index()

    print(
        f"[data_scraping] Daily   : {daily_df.shape} | "
        f"dates={daily_df.index.get_level_values('date').nunique()} | "
        f"tickers={daily_df.index.get_level_values('ticker').nunique()}"
    )
    print(
        f"[data_scraping] 15-min  : {intraday_df.shape} | "
        f"bars={len(intraday_df)}"
    )
    return daily_df, intraday_df


def scrape_nse_delivery_data(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Fetches NSE full bhavcopies for every business day in [start_date, end_date]
    and extracts institutional delivery metrics for EQ-series stocks.

    One HTTP request is made per trading day (the full bhavcopy covers every
    listed symbol), so this is efficient for bulk universe-wide enrichment.
    NSE holidays within the business-day range are caught and skipped silently.

    Only available from 2020-01-01 onward; requests for earlier dates raise
    ReadTimeout inside jugaad-data.

    Args:
        start_date: First date of the fetch window (inclusive).
        end_date:   Last date of the fetch window (inclusive).

    Returns:
        DataFrame with MultiIndex ('date', 'ticker') and columns
        ['DeliveryQty', 'DeliveryPct'].  Empty DataFrame if no data was fetched.
    """
    from jugaad_data.nse.archives import NSEArchives  # deferred: not installed in SANDBOX

    nse = NSEArchives()
    frames: list[pd.DataFrame] = []

    for ts in pd.bdate_range(start=start_date, end=end_date, freq="B"):
        dt: date = ts.date()
        try:
            csv_text: str = nse.full_bhavcopy_raw(dt)
            day_df: pd.DataFrame = pd.read_csv(io.StringIO(csv_text))

            # Validate that this file actually contains delivery columns.
            # Some holiday-adjacent files are present but stripped.
            missing = {_BHAV_DELIV_QTY_COL, _BHAV_DELIV_PER_COL} - set(day_df.columns)
            if missing:
                print(f"[data_scraping] {dt}: missing columns {missing}, skipping")
                continue

            # Restrict to EQ series to exclude SME, ETFs, etc.
            if _BHAV_SERIES_COL in day_df.columns:
                day_df = day_df[day_df[_BHAV_SERIES_COL].str.strip() == "EQ"]

            delivery: pd.DataFrame = day_df[
                [_BHAV_SYMBOL_COL, _BHAV_DELIV_QTY_COL, _BHAV_DELIV_PER_COL]
            ].copy()
            delivery.columns = ["ticker", DELIVERY_QTY_COL, DELIVERY_PCT_COL]
            delivery.insert(0, "date", pd.Timestamp(dt))
            delivery = delivery.set_index(["date", "ticker"])

            frames.append(delivery)
            print(f"[data_scraping] {dt}: {len(delivery)} EQ symbols fetched")

        except Exception as exc:
            # Covers NSE holidays (no file), network errors, and ReadTimeout
            print(f"[data_scraping] {dt}: skipped ({type(exc).__name__}: {exc})")

    if not frames:
        print("[data_scraping] No delivery data retrieved — returning empty DataFrame")
        return pd.DataFrame(
            columns=[DELIVERY_QTY_COL, DELIVERY_PCT_COL],
            index=pd.MultiIndex.from_tuples([], names=["date", "ticker"]),
        )

    result: pd.DataFrame = pd.concat(frames).sort_index()
    result[DELIVERY_QTY_COL] = pd.to_numeric(result[DELIVERY_QTY_COL], errors="coerce")
    result[DELIVERY_PCT_COL] = pd.to_numeric(result[DELIVERY_PCT_COL], errors="coerce")
    print(f"[data_scraping] Delivery data: {result.shape} | index={result.index.names}")
    return result


def augment_dataset(
    daily_df: pd.DataFrame,
    delivery_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges DeliveryQty and DeliveryPct into the local daily price DataFrame
    via a MultiIndex left-join on ('date', 'ticker').

    Gap-filling strategy:
      - Forward-fill within each ticker across dates to bridge weekends and
        NSE holidays where no bhavcopy is available.
      - Any leading NaNs (ticker appears in daily_df before delivery data
        begins) are filled with 0.0 to avoid polluting downstream features.

    Args:
        daily_df:    MultiIndex ('date', 'ticker') DataFrame from load_local_csv_data.
        delivery_df: MultiIndex ('date', 'ticker') DataFrame from scrape_nse_delivery_data.

    Returns:
        Augmented copy of daily_df with DeliveryQty and DeliveryPct appended.
    """
    delivery_cols: list[str] = [DELIVERY_QTY_COL, DELIVERY_PCT_COL]

    augmented: pd.DataFrame = daily_df.join(delivery_df[delivery_cols], how="left")

    # Forward-fill within each ticker group to propagate last known delivery
    # values across days where the bhavcopy was unavailable (holidays, gaps).
    for col in delivery_cols:
        augmented[col] = (
            augmented[col]
            .groupby(level="ticker")
            .ffill()
            .fillna(0.0)     # leading rows before first bhavcopy → zero
        )

    nan_remaining: int = int(augmented[delivery_cols].isna().sum().sum())
    print(
        f"[data_scraping] Augmented : {augmented.shape} | "
        f"NaNs remaining after fill: {nan_remaining}"
    )
    return augmented


if __name__ == "__main__":
    DRY_START: date = date(2024, 1, 2)
    DRY_END: date = date(2024, 1, 8)    # 5 business days

    # --- Step 1: local CSVs ------------------------------------------------- #
    print("=" * 60)
    print("STEP 1 — Local CSV ingestion")
    print("=" * 60)
    daily_df: pd.DataFrame | None = None
    try:
        daily_df, intraday_df = load_local_csv_data()
        print(f"  daily_df columns   : {daily_df.columns.tolist()}")
        print(f"  intraday_df columns: {intraday_df.columns.tolist()}")
    except FileNotFoundError as exc:
        print(f"  Local CSVs not present yet ({exc}). Skipping augment step.")

    # --- Step 2: delivery scrape -------------------------------------------- #
    print()
    print("=" * 60)
    print(f"STEP 2 — NSE delivery scrape  ({DRY_START} → {DRY_END})")
    print("=" * 60)
    delivery_df: pd.DataFrame = scrape_nse_delivery_data(DRY_START, DRY_END)
    print(f"\n  shape   : {delivery_df.shape}")
    print(f"  columns : {delivery_df.columns.tolist()}")
    if not delivery_df.empty:
        print(delivery_df.head(10).to_string())

    # --- Step 3: augment (only if local data is available) ------------------ #
    if daily_df is not None and not delivery_df.empty:
        print()
        print("=" * 60)
        print("STEP 3 — Augmentation")
        print("=" * 60)
        dry_dates = pd.bdate_range(DRY_START, DRY_END)
        slice_df: pd.DataFrame = daily_df[
            daily_df.index.get_level_values("date").isin(dry_dates)
        ]
        if not slice_df.empty:
            augmented: pd.DataFrame = augment_dataset(slice_df, delivery_df)
            print(f"\n  augmented shape   : {augmented.shape}")
            print(f"  augmented columns : {augmented.columns.tolist()}")
            print(augmented.head(10).to_string())
        else:
            print("  No daily rows found in the dry-run date window.")
