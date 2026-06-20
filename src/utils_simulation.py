from typing import Optional, Union

import numpy as np
import pandas as pd

NUM_STOCKS: int = 10
NUM_DAYS: int = 500
PHASE_SPLIT: int = 250   # days 1–250 = phase 1 ; days 251–500 = phase 2
START_PRICE: float = 100.0
DT: float = 1.0 / 252    # one trading day as a fraction of a year

STOCK_COLS: list[str] = [f"STOCK_{i:02d}" for i in range(1, NUM_STOCKS + 1)]

# Per-stock idiosyncratic annual drift added on top of the market-phase drift.
# Top 3 (STOCK_01–03): strong structural uptrend.
# Middle 4 (STOCK_04–07): flat / sideways junk.
# Bottom 3 (STOCK_08–10): strong structural decay.
STOCK_DRIFT_OFFSETS: np.ndarray = np.array([
    +0.35, +0.35, +0.35,   # alpha
    +0.05, +0.05, +0.05, +0.05,  # junk
    -0.25, -0.25, -0.25,   # decadent
], dtype=np.float64)

# Non-stationary, two-phase coupled GBM parameter profiles (all values annualised).
# 'phase1' governs days 1–250; 'phase2' governs days 251–500.
# Conversion to daily: drift_d = drift_a * DT,  vol_d = vol_a * sqrt(DT)
COUPLED_PARAMS: dict[str, dict[str, dict[str, float]]] = {
    "stable_uptrend": {
        "phase1": {"annual_drift":  0.05, "annual_vol": 0.10},
        "phase2": {"annual_drift":  0.25, "annual_vol": 0.10},
    },
    "stable_downtrend": {
        "phase1": {"annual_drift":  0.05, "annual_vol": 0.10},
        "phase2": {"annual_drift": -0.25, "annual_vol": 0.10},
    },
    "stable_volatile": {
        "phase1": {"annual_drift":  0.05, "annual_vol": 0.10},
        "phase2": {"annual_drift":  0.00, "annual_vol": 0.45},
    },
    "volatile_downtrend": {
        "phase1": {"annual_drift":  0.00, "annual_vol": 0.45},
        "phase2": {"annual_drift": -0.25, "annual_vol": 0.15},
    },
}


def get_output_path(market_type: str) -> str:
    """Returns the canonical CSV path for a given coupled market profile."""
    return f"data/raw_nse/mock_daily_{market_type}.csv"


def _gbm_log_increments(
    rng: np.random.Generator,
    n_days: int,
    annual_drift: Union[float, np.ndarray],
    annual_vol: float,
    n_series: int,
) -> np.ndarray:
    """
    Generates cumulative GBM log-space increments starting from zero.

    Shape of output: (n_days, n_series).
    Adding a starting log-price array of shape (n_series,) yields the full
    log-price path for each series.

    Args:
        rng:          Seeded NumPy random generator (shared across phases).
        n_days:       Number of time steps to generate.
        annual_drift: Annualised drift mu — scalar (broadcast to all series) or
                      ndarray of shape (n_series,) for per-series drift.
        annual_vol:   Annualised volatility parameter sigma (shared across series).
        n_series:     Number of independent price series (stocks or 1 for index).

    Returns:
        ndarray of shape (n_days, n_series) — cumulative log increments.
    """
    daily_vol: float = annual_vol * (DT ** 0.5)
    # log_drift is scalar or (n_series,); both broadcast cleanly against (n_days, n_series)
    log_drift: Union[float, np.ndarray] = annual_drift * DT - 0.5 * daily_vol ** 2
    noise: np.ndarray = rng.normal(loc=0.0, scale=daily_vol, size=(n_days, n_series))
    return np.cumsum(noise + log_drift, axis=0)


def generate_synthetic_daily_data(
    market_type: str = "stable_uptrend",
    num_days: int = NUM_DAYS,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generates a non-stationary synthetic daily close-price DataFrame by
    splitting the timeline into two equal phases, each governed by its own
    independent GBM drift and volatility parameters.

    Price paths are continuous at the phase boundary: Phase 2 begins from
    the exact terminal price of Phase 1 (no discontinuous jump at day 251).

    GBM log-space increment per day: (mu*dt - 0.5*sigma^2*dt) + sigma*sqrt(dt)*Z

    Args:
        market_type: Key in COUPLED_PARAMS — one of the four coupled profiles.
        num_days:    Total business days (split PHASE_SPLIT / num_days-PHASE_SPLIT).
        seed:        RNG seed for full reproducibility across runs.

    Returns:
        DataFrame with DatetimeIndex (business days ending 2025-12-31),
        10 stock columns (STOCK_01 … STOCK_10), and one INDEX_CLOSE column.
        Index is named 'date'.
    """
    if market_type not in COUPLED_PARAMS:
        raise ValueError(
            f"Unknown market_type '{market_type}'. "
            f"Valid options: {list(COUPLED_PARAMS)}"
        )

    p1_days: int = PHASE_SPLIT
    p2_days: int = num_days - PHASE_SPLIT
    p1 = COUPLED_PARAMS[market_type]["phase1"]
    p2 = COUPLED_PARAMS[market_type]["phase2"]

    rng = np.random.default_rng(seed)
    date_index = pd.bdate_range(end=pd.Timestamp("2025-12-31"), periods=num_days, freq="B")
    log_start: np.ndarray = np.full(NUM_STOCKS, np.log(START_PRICE))

    # --- Per-stock drift = market-phase drift + idiosyncratic offset (shape: (10,)) ---
    stock_drifts_p1: np.ndarray = p1["annual_drift"] + STOCK_DRIFT_OFFSETS
    stock_drifts_p2: np.ndarray = p2["annual_drift"] + STOCK_DRIFT_OFFSETS

    # --- Stock price paths (10 series) ---
    log_inc_p1: np.ndarray = _gbm_log_increments(rng, p1_days, stock_drifts_p1, p1["annual_vol"], NUM_STOCKS)
    log_px_p1: np.ndarray  = log_start + log_inc_p1                   # (250, 10)

    log_end_p1: np.ndarray = log_px_p1[-1, :]                         # (10,)  terminal log-prices
    log_inc_p2: np.ndarray = _gbm_log_increments(rng, p2_days, stock_drifts_p2, p2["annual_vol"], NUM_STOCKS)
    log_px_p2: np.ndarray  = log_end_p1 + log_inc_p2                  # (250, 10)

    stock_prices: np.ndarray = np.exp(np.vstack([log_px_p1, log_px_p2]))  # (500, 10)

    # --- Index path (1 series, half stock vol per phase for smoother benchmark) ---
    log_start_idx: float = np.log(START_PRICE)

    idx_inc_p1: np.ndarray = _gbm_log_increments(rng, p1_days, p1["annual_drift"], p1["annual_vol"] * 0.5, 1)
    log_idx_p1: np.ndarray = log_start_idx + idx_inc_p1[:, 0]         # (250,)

    log_end_idx_p1: float  = float(log_idx_p1[-1])
    idx_inc_p2: np.ndarray = _gbm_log_increments(rng, p2_days, p2["annual_drift"], p2["annual_vol"] * 0.5, 1)
    log_idx_p2: np.ndarray = log_end_idx_p1 + idx_inc_p2[:, 0]       # (250,)

    index_prices: np.ndarray = np.exp(np.concatenate([log_idx_p1, log_idx_p2]))

    data = pd.DataFrame(stock_prices, index=date_index, columns=STOCK_COLS)
    data["INDEX_CLOSE"] = index_prices
    data.index.name = "date"

    return data


def save_mock_data(df: pd.DataFrame, path: str) -> None:
    """Saves a price DataFrame to CSV at the specified path."""
    df.to_csv(path)


if __name__ == "__main__":
    for ct in COUPLED_PARAMS:
        path = get_output_path(ct)
        df = generate_synthetic_daily_data(market_type=ct)
        save_mock_data(df, path)
        p1 = COUPLED_PARAMS[ct]["phase1"]
        p2 = COUPLED_PARAMS[ct]["phase2"]
        print(
            f"[{ct:>20}] {len(df)} rows × {df.shape[1]} cols | "
            f"P1(drift={p1['annual_drift']:+.2f}, vol={p1['annual_vol']:.2f}) → "
            f"P2(drift={p2['annual_drift']:+.2f}, vol={p2['annual_vol']:.2f}) | "
            f"→ {path}"
        )
