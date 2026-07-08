import os
from typing import Final

import pandas as pd

LEDGER_PATH: Final[str] = "data/trial_database/master_dsr_matrix.parquet"


def log_to_dsr_ledger(strategy_id: str, returns_series: pd.Series) -> None:
    """
    Appends or overwrites a strategy's daily return series in the master DSR
    ledger Parquet file.

    The ledger stores one column per strategy trial, indexed by date.  Columns
    are named after the strategy_id so they remain uniquely addressable across
    the entire research lifecycle.  If the ledger does not yet exist it is
    created from scratch.  If it already exists the strategy column is
    overwritten in-place (idempotent re-runs do not duplicate data), then the
    full updated matrix is written back to disk.

    Args:
        strategy_id:    Unique string identifier for this strategy trial.
                        Used directly as the column name in the ledger.
        returns_series: Daily net portfolio return Series with a DatetimeIndex.
                        Produced by Tier 3 execute_poc_strategy.
    """
    column: pd.Series = returns_series.rename(strategy_id)

    if not os.path.exists(LEDGER_PATH):
        ledger: pd.DataFrame = column.to_frame()
        ledger.to_parquet(LEDGER_PATH)
        print(
            f"[tier4_dsr] Created new ledger at '{LEDGER_PATH}' | "
            f"strategy='{strategy_id}' | {len(column)} rows"
        )
    else:
        ledger = pd.read_parquet(LEDGER_PATH)
        # Assign aligns on index automatically; NaN fills gaps if date ranges differ
        ledger[strategy_id] = column
        ledger.to_parquet(LEDGER_PATH)
        print(
            f"[tier4_dsr] Updated ledger | "
            f"strategy='{strategy_id}' | "
            f"{ledger.shape[1]} total strategy column(s) | "
            f"{len(ledger)} rows"
        )

    # Verify the write by spot-checking the file is readable and non-empty
    verification: pd.DataFrame = pd.read_parquet(LEDGER_PATH)
    assert strategy_id in verification.columns, (
        f"[tier4_dsr] WRITE FAILED — '{strategy_id}' not found in ledger columns."
    )
    print(
        f"[tier4_dsr] Ledger verified on disk | "
        f"columns={list(verification.columns)} | "
        f"shape={verification.shape}"
    )
