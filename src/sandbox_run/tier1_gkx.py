from typing import Final

import numpy as np
import pandas as pd

TOP_N: Final[int] = 3


def calculate_momentum_alpha(
    asset_df: pd.DataFrame,
    lookback_period: int = 5,
) -> pd.DataFrame:
    """
    Computes cross-sectional trailing momentum and ranks all stocks daily.

    Momentum formula: (close_t / close_{t-lookback_period}) - 1

    Ranking convention: rank 1 = highest momentum (best). Ties are broken by
    first occurrence (deterministic). Rows with an incomplete lookback window
    are dropped so there is no look-ahead bias.

    Args:
        asset_df:        DataFrame of daily close prices, shape (days, n_stocks).
        lookback_period: Number of days for the momentum window (from config).

    Returns:
        Integer rank DataFrame of the same column structure as asset_df.
        Rank values range from 1 (top) to n_stocks (bottom).
        Stocks with rank <= TOP_N are the top momentum picks for that day.
    """
    # Vectorized momentum signal — no per-row loop
    momentum: pd.DataFrame = asset_df.div(asset_df.shift(lookback_period)) - 1

    # Drop leading rows where the full lookback window is unavailable
    momentum = momentum.dropna(how="all")

    # Cross-sectional rank: 1 = highest momentum, descending
    ranks: pd.DataFrame = (
        momentum
        .rank(axis=1, ascending=False, method="first")
        .astype(np.int32)
    )

    print(
        f"[tier1_gkx] Momentum ranks computed | "
        f"{len(ranks)} tradeable days | "
        f"window={lookback_period} days | top-{TOP_N} threshold"
    )
    return ranks
