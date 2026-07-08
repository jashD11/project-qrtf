import numpy as np
import pandas as pd

from config import StrategyConfig

_VALID_STYLES: frozenset[str] = frozenset({"long_only", "long_short", "dynamic_tilt"})


def execute_poc_strategy(
    alpha_ranks: pd.DataFrame,
    regimes: pd.Series,
    asset_df: pd.DataFrame,
    cfg: StrategyConfig,
) -> pd.Series:
    """
    Simulates the POC strategy chronologically using fully vectorized operations.

    Three execution styles are supported via cfg.execution_style:

    'long_only'
        Weight +1/top_n for the top_n momentum stocks each day; 0 elsewhere.
        Gross exposure = 1.0.
        Regime gate: State 1 (High Vol) → portfolio forced to 0.0 (flat/cash).

    'long_short'
        Dollar-neutral spread: weight +1/top_n for top_n stocks,
        -1/bottom_n for bottom_n stocks, 0 for the middle.
        Net exposure = 0.0 (when top_n == bottom_n); gross exposure = 2.0.
        Regime gate: State 1 → portfolio forced to 0.0 (flat/cash).

    'dynamic_tilt'
        Regime-conditional weight matrix — no hard cash stop:
          State 0 (Low Vol)  → long-only weights (+1/top_n for top stocks, 0 elsewhere).
          State 1 (High Vol) → dollar-neutral spread (+1/top_n longs, -1/bottom_n shorts).
        State 1 activates the protective short leg rather than triggering a cash exit.
        This is implemented with a vectorized per-day flag broadcast; no Python loop.

    In all styles, portfolio return on signal date T is realized as the next-day
    (T→T+1) weighted close-to-close return.  The final signal date is dropped
    when no T+1 price row exists (shift(-1) produces NaN; min_count=1 preserves it).

    Args:
        alpha_ranks: Integer rank DataFrame (1=best) from calculate_momentum_alpha.
                     Shape (signal_days, n_stocks).  Rank 1 = highest momentum.
        regimes:     Regime label Series {0=Low Vol, 1=High Vol} from detect_vol_regime.
        asset_df:    Daily close price DataFrame. Shape (all_days, n_stocks).
        cfg:         StrategyConfig carrying execution_style, top_n, bottom_n.

    Returns:
        pd.Series of float portfolio net daily returns indexed by signal date.
    """
    if cfg.execution_style not in _VALID_STYLES:
        raise ValueError(
            f"Unknown execution_style '{cfg.execution_style}'. "
            f"Valid options: {sorted(_VALID_STYLES)}"
        )

    n_stocks: int = alpha_ranks.shape[1]

    # --- Date alignment ---
    common_dates: pd.DatetimeIndex = alpha_ranks.index.intersection(regimes.index)
    ranks: pd.DataFrame = alpha_ranks.loc[common_dates]
    regimes_aligned: pd.Series = regimes.loc[common_dates]

    # Forward return at row t: price[t+1] / price[t] - 1  (fully vectorized, no loop)
    forward_returns: pd.DataFrame = asset_df.shift(-1) / asset_df - 1
    forward_returns_aligned: pd.DataFrame = forward_returns.reindex(common_dates)

    # --- Base weight vectors (days × stocks) ---
    top_mask: pd.DataFrame = ranks <= cfg.top_n
    bot_mask: pd.DataFrame = ranks >= (n_stocks - cfg.bottom_n + 1)

    long_weights: pd.DataFrame = top_mask.astype(float) * (1.0 / cfg.top_n)
    short_weights: pd.DataFrame = bot_mask.astype(float) * (1.0 / cfg.bottom_n)

    # --- Build the final weight matrix for this execution style ---
    if cfg.execution_style == "long_only":
        weight_matrix: pd.DataFrame = long_weights

    elif cfg.execution_style == "long_short":
        # Static dollar-neutral spread every day
        weight_matrix = long_weights - short_weights

    else:  # dynamic_tilt
        # Per-day flag: 1.0 on State 1 (Panic) days, 0.0 on State 0 (Quiet) days.
        # Broadcast scalar per day → (days, stocks) matrix via outer product.
        state1_flag: pd.DataFrame = pd.DataFrame(
            np.outer((regimes_aligned == 1).astype(float).values, np.ones(n_stocks)),
            index=common_dates,
            columns=alpha_ranks.columns,
        )
        quiet_flag: pd.DataFrame = 1.0 - state1_flag   # 1.0 on State 0, 0.0 on State 1
        # Quiet (State 0): 130/30 leverage — 1.3x long, -0.3x short
        # Panic (State 1): dollar-neutral  — 1.0x long, -1.0x short
        long_scale:  pd.DataFrame = 1.3 * quiet_flag + 1.0 * state1_flag
        short_scale: pd.DataFrame = 0.3 * quiet_flag + 1.0 * state1_flag
        weight_matrix = long_scale * long_weights - short_scale * short_weights

    # --- Gross portfolio return = Σ(weight × forward_return) across stocks ---
    # min_count=1: row of all-NaN (last bar, no T+1 price) stays NaN so dropna()
    # removes it cleanly rather than collapsing it to 0.0.
    gross_returns: pd.Series = (
        (weight_matrix * forward_returns_aligned).sum(axis=1, min_count=1)
    )

    # --- Regime gate: hard cash stop for long_only and long_short only ---
    # dynamic_tilt uses State 1 to activate the short leg, not to go to cash.
    if cfg.execution_style != "dynamic_tilt":
        gross_returns = gross_returns.where(regimes_aligned == 0, other=0.0)

    portfolio_returns: pd.Series = gross_returns.dropna()
    portfolio_returns.name = "portfolio_return"
    portfolio_returns.index.name = "date"

    # --- Diagnostic summary ---
    regimes_final: pd.Series = regimes_aligned.loc[portfolio_returns.index]
    state0_days: int = int((regimes_final == 0).sum())
    state1_days: int = int((regimes_final == 1).sum())

    if cfg.execution_style == "dynamic_tilt":
        state_label = f"Long-only: {state0_days}d | Hedged: {state1_days}d"
    else:
        state_label = f"Active: {state0_days}d | Cash: {state1_days}d"

    print(
        f"[tier3_execution] style={cfg.execution_style} | "
        f"{len(portfolio_returns)} obs | {state_label} | "
        f"Cumulative: {(1 + portfolio_returns).prod() - 1:.4%}"
    )
    return portfolio_returns
