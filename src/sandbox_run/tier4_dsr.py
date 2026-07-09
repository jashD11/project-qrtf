import os
from typing import Final

import pandas as pd

LEDGER_PATH: Final[str] = "data/trial_database/master_dsr_matrix.parquet"


def log_to_dsr_ledger(
    strategy_id: str,
    returns_series: pd.Series,
    ledger_path: str = LEDGER_PATH,
) -> None:
    """
    Appends or overwrites a strategy's daily return series in a DSR ledger
    Parquet file.

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
                        Produced by Tier 3 execution.
        ledger_path:    Destination Parquet. Defaults to the sandbox master
                        ledger; PRODUCTION_ML headline and sensitivity runs pass
                        their own paths so the populations never co-mingle.
    """
    column: pd.Series = returns_series.rename(strategy_id)
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)

    if not os.path.exists(ledger_path):
        ledger: pd.DataFrame = column.to_frame()
        ledger.to_parquet(ledger_path)
        print(
            f"[tier4_dsr] Created new ledger at '{ledger_path}' | "
            f"strategy='{strategy_id}' | {len(column)} rows"
        )
    else:
        ledger = pd.read_parquet(ledger_path)
        # Assign aligns on index automatically; NaN fills gaps if date ranges differ
        ledger[strategy_id] = column
        ledger.to_parquet(ledger_path)
        print(
            f"[tier4_dsr] Updated ledger | "
            f"strategy='{strategy_id}' | "
            f"{ledger.shape[1]} total strategy column(s) | "
            f"{len(ledger)} rows"
        )

    # Verify the write by spot-checking the file is readable and non-empty
    verification: pd.DataFrame = pd.read_parquet(ledger_path)
    assert strategy_id in verification.columns, (
        f"[tier4_dsr] WRITE FAILED — '{strategy_id}' not found in ledger columns."
    )
    print(
        f"[tier4_dsr] Ledger verified on disk | "
        f"columns={list(verification.columns)} | "
        f"shape={verification.shape}"
    )
