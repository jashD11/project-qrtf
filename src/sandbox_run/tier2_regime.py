from typing import Final

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

N_ITER: Final[int] = 200
RANDOM_STATE: Final[int] = 42


def detect_vol_regime(
    index_df: pd.DataFrame,
    n_states: int = 2,
) -> pd.Series:
    """
    Fits an n-state Gaussian HMM on daily log returns of INDEX_CLOSE and
    returns a regime label series with deterministic variance-ordered encoding.

    After fitting, raw HMM state labels are remapped by ascending variance so
    that State 0 is always the lowest-volatility regime, regardless of the
    internal label the EM algorithm assigned.

    Note: The binary regime gate in Tier 3 (active vs cash) is designed for
    n_states=2.  Values other than 2 raise NotImplementedError until the
    multi-state execution logic is implemented.

    Args:
        index_df: DataFrame containing an 'INDEX_CLOSE' column with daily
                  close prices.  Must have a DatetimeIndex.
        n_states: Number of HMM hidden states (from config.hmm_states).

    Returns:
        pd.Series of dtype int indexed by date.  Values are variance-ordered
        state labels: 0 = lowest vol, n_states-1 = highest vol.
        The series starts one day after the first price row (log-return drop).
    """
    if n_states != 2:
        raise NotImplementedError(
            f"n_states={n_states} is not yet supported. "
            "The Tier 3 binary regime gate requires exactly 2 states."
        )

    log_returns: pd.Series = np.log(index_df["INDEX_CLOSE"]).diff().dropna()
    # 20-day rolling cumsum of log returns = macro trend proxy (no look-ahead: uses past 20 bars)
    trend_proxy: pd.Series = log_returns.rolling(20).sum().fillna(0.0)
    obs: np.ndarray = np.column_stack([log_returns.to_numpy(), trend_proxy.to_numpy()])

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
    )
    model.fit(obs)
    raw_states: np.ndarray = model.predict(obs)

    # covars_ shape: (n_states, 2, 2) — feature 0 is daily return variance
    state_variances: np.ndarray = model.covars_[:, 0, 0]
    low_vol_raw: int = int(np.argmin(state_variances))
    high_vol_raw: int = int(np.argmax(state_variances))

    # Remap so label 0 = lowest variance, label 1 = highest variance
    remap: np.ndarray = np.empty(n_states, dtype=np.int32)
    remap[low_vol_raw] = 0
    remap[high_vol_raw] = 1

    regimes: pd.Series = pd.Series(
        remap[raw_states].astype(np.int32),
        index=log_returns.index,
        name="vol_regime",
    )

    low_pct: float = float((regimes == 0).mean() * 100)
    print(
        f"[tier2_regime] HMM converged in {model.monitor_.iter} iter(s) | "
        f"n_states={n_states} | features=2 (ret, trend20) | "
        f"Low-vol (State 0): {low_pct:.1f}% of days | "
        f"Return variances → low={state_variances[low_vol_raw]:.2e}, "
        f"high={state_variances[high_vol_raw]:.2e}"
    )
    return regimes
