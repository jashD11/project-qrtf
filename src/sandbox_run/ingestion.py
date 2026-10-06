from typing import Tuple

import numpy as np
import pandas as pd

STOCK_COLS: list[str] = [f"STOCK_{i:02d}" for i in range(1, 11)]


def ingest_poc_data(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads mock daily price CSV, validates data quality, and splits into
    asset and index DataFrames.

    Validation checks:
      - All expected stock columns and INDEX_CLOSE are present.
      - No negative or zero prices (would corrupt downstream log/division ops).
      - NaN gaps are forward-filled; leading NaN rows (no prior value) are dropped.

    Args:
        filepath: Path to the raw CSV file produced by utils_simulation.

    Returns:
        Tuple of (asset_df, index_df):
          - asset_df: DataFrame of shape (days, 10) with stock close prices.
          - index_df: DataFrame of shape (days, 1) with INDEX_CLOSE prices.
    """
    df: pd.DataFrame = pd.read_csv(filepath, index_col="date", parse_dates=True)
    df.sort_index(inplace=True)

    # --- Column presence check ---
    expected_cols: list[str] = STOCK_COLS + ["INDEX_CLOSE"]
    missing: list[str] = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[ingestion] Missing expected columns: {missing}")

    df = df[expected_cols]

    # --- Value anomaly checks ---
    if (df < 0).any(axis=None):
        bad: list[str] = df.columns[(df < 0).any()].tolist()
        raise ValueError(f"[ingestion] Negative prices detected in: {bad}")

    if (df == 0).any(axis=None):
        bad = df.columns[(df == 0).any()].tolist()
        raise ValueError(f"[ingestion] Zero prices detected in: {bad}")

    # --- Forward-fill only (strict temporal causality, no bfill) ---
    nan_total: int = int(df.isna().sum().sum())
    if nan_total > 0:
        print(f"[ingestion] Forward-filling {nan_total} NaN value(s).")
    df = df.ffill()

    leading_nans: int = int(df.isna().sum().sum())
    if leading_nans > 0:
        df = df.dropna()
        print(f"[ingestion] Dropped {leading_nans} leading row(s) with no prior value.")

    asset_df: pd.DataFrame = df[STOCK_COLS].copy()
    index_df: pd.DataFrame = df[["INDEX_CLOSE"]].copy()

    print(
        f"[ingestion] Loaded {len(df)} rows | "
        f"{asset_df.shape[1]} stocks | "
        f"{df.index[0].date()} → {df.index[-1].date()}"
    )
    return asset_df, index_df
