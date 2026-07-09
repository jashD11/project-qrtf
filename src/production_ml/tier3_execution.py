"""
Phase 2 Tier 3 — execution / de-risk gate (PRODUCTION_ML).

Converts the Tier 1 tree signal (long/short decile masks) and the Tier 2
market-regime signal (the causal ``panic`` de-risk gate) into a single daily —
or intraday — portfolio return series, one strategy at a time. This mirrors the
*structure* of the Phase 1 sandbox tier (``src/sandbox_run/tier3_execution.py``:
vectorized weight matrix, ``np.outer`` regime broadcast, ``min_count=1`` last-bar
drop) but with two deliberate differences:

  1. **Gate on ``panic``, not on HMM ``states``.** The Tier 2 HMM ``states`` are
     a vol/herding taxonomy that a 2-state model merely *bisects* (~55/45) — not
     a rare crisis flag. The frequency-calibrated ``panic`` gate (a causal
     trailing percentile on the continuous stress score) is the tradeable
     de-risk trigger, and is what this tier acts on. ``states``/``probs`` are
     currently unused here; state-conditional sizing is deferred — see
     docs/phase3_deferred_hmm.md.
  2. **Variable book width.** Tier 1 sizes each side by ``decile_pct`` (a
     fraction of that day's valid cross-section), so the number of names per
     leg varies day to day. Each selected name therefore gets ``1/k`` for that
     day's ``k``, rather than the sandbox's fixed ``1/top_n``.

Execution styles (``StrategyConfig.execution_style``)
    long_only     +1/k on the long decile; Panic -> flat (cash).
    long_short    +1/k long, -1/k short (dollar-neutral); Panic -> flat (cash).
    dynamic_tilt  Calm: 130/30 (1.3x long, 0.3x short). Panic: dollar-neutral
                  (1.0x/1.0x) — the short leg activates rather than exiting to
                  cash. Never goes flat. Leverage constants live in
                  ``config.ML_CONFIG.execution`` (a [FIXED] prior, not swept).

Temporal causality: signal/weight on bar T realizes as the forward return T->T+1
via ``price_wide.shift(-1)``. Unlike the tree's *training* label (which nulls
cross-session transitions), the *executed* forward return deliberately keeps the
overnight/gap move — it is a real return on a genuinely held position.

For intraday frequencies the daily ``panic`` gate is lag-broadcast onto bars via
``tier2_regime.broadcast_to_intraday`` (each bar inherits the prior completed
day's gate — no intraday look-ahead).
"""

import os
import sys
from typing import Final

import numpy as np
import pandas as pd

# Make the repo root importable whether run directly or by an orchestrator.
_ROOT: str = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from config import StrategyConfig
from src.production_ml.tier1_trees import WalkForwardResult
from src.production_ml.tier2_regime import broadcast_to_intraday

_VALID_STYLES: Final[frozenset[str]] = frozenset(
    {"long_only", "long_short", "dynamic_tilt"}
)
_DATE_LEVEL: Final[str] = "date"
_TICKER_LEVEL: Final[str] = "ticker"


# --------------------------------------------------------------------------- #
# Weight construction (fully vectorized — no loop over the time dimension)
# --------------------------------------------------------------------------- #
def build_weight_matrix(
    long_mask: pd.DataFrame,
    short_mask: pd.DataFrame,
    panic: pd.Series,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    """
    Build the (bar x ticker) portfolio weight matrix for one execution style.

    Args:
        long_mask:  (bar x ticker) 1 = long, 0 = flat (Tier 1).
        short_mask: (bar x ticker) -1 = short, 0 = flat (Tier 1).
        panic:      de-risk gate aligned to the mask's bar index (bool-ish;
                    reindexed + NaN->False here for safety).
        cfg:        carries ``execution_style``.

    Returns:
        (bar x ticker) float weight matrix. Row gross/net exposure by style:
          long_only    Calm gross 1.0 (net +1.0); Panic all-zero (cash).
          long_short   Calm gross 2.0 (net 0.0);  Panic all-zero (cash).
          dynamic_tilt Calm gross 1.6 (net +1.0); Panic gross 2.0 (net 0.0).
    """
    if cfg.execution_style not in _VALID_STYLES:
        raise ValueError(
            f"Unknown execution_style {cfg.execution_style!r}. "
            f"Valid: {sorted(_VALID_STYLES)}"
        )

    # Per-day book width k varies (decile_pct of a shifting cross-section), so
    # each selected name is 1/k that day. Empty side -> NaN count -> zero weights.
    long_count: pd.Series = long_mask.sum(axis=1).replace(0, np.nan)
    short_count: pd.Series = short_mask.abs().sum(axis=1).replace(0, np.nan)
    long_w: pd.DataFrame = long_mask.div(long_count, axis=0).fillna(0.0)
    short_w: pd.DataFrame = short_mask.abs().div(short_count, axis=0).fillna(0.0)

    panic_bool: pd.Series = (
        panic.reindex(long_mask.index, fill_value=False).astype(bool)
    )

    if cfg.execution_style == "long_only":
        weights: pd.DataFrame = long_w
        # Panic -> cash: zero the whole row (broadcast the per-bar mask down cols).
        weights = weights.where(~panic_bool, 0.0, axis=0)

    elif cfg.execution_style == "long_short":
        weights = long_w - short_w
        weights = weights.where(~panic_bool, 0.0, axis=0)

    else:  # dynamic_tilt — regime-conditional leverage, never flat.
        ex = config.ML_CONFIG.execution
        flag: np.ndarray = panic_bool.to_numpy(dtype=float)[:, None]  # (bars, 1)
        long_scale: np.ndarray = ex.calm_long_lev * (1.0 - flag) + ex.panic_long_lev * flag
        short_scale: np.ndarray = ex.calm_short_lev * (1.0 - flag) + ex.panic_short_lev * flag
        weights = pd.DataFrame(
            long_scale * long_w.to_numpy() - short_scale * short_w.to_numpy(),
            index=long_mask.index,
            columns=long_mask.columns,
        )

    return weights


# --------------------------------------------------------------------------- #
# Gate alignment (daily gate -> bar frequency)
# --------------------------------------------------------------------------- #
def _align_panic(panic: pd.Series, bar_index: pd.Index, frequency: str) -> pd.Series:
    """
    Put the daily ``panic`` gate on the strategy's bar index.

    Daily strategies reindex 1:1. Intraday strategies lag-broadcast: each bar
    inherits the prior *completed* day's gate (t-1), so there is no intraday
    look-ahead.
    """
    if frequency == config.DEFAULT_FREQUENCY or config.freq_spec(frequency).bars_per_day == 1:
        return panic.astype(bool).reindex(bar_index, fill_value=False)
    bcast: pd.Series = broadcast_to_intraday(
        panic.astype(float), pd.DatetimeIndex(bar_index)
    )
    return bcast > 0.5


# --------------------------------------------------------------------------- #
# Strategy execution
# --------------------------------------------------------------------------- #
def execute_ml_strategy(
    wf_result: WalkForwardResult,
    panic: pd.Series,
    price_wide: pd.DataFrame,
    cfg: StrategyConfig,
) -> pd.Series:
    """
    Realize one PRODUCTION_ML strategy's portfolio return series.

    Args:
        wf_result:  Tier 1 walk-forward output (long/short decile masks).
        panic:      Tier 2 daily de-risk gate (bool). Broadcast to bars for
                    intraday frequencies.
        price_wide: (bar x ticker) close prices for ``cfg.frequency``.
        cfg:        StrategyConfig (execution_style, frequency).

    Returns:
        Net portfolio return per bar, indexed by bar timestamp, NaN-free.
    """
    long_mask: pd.DataFrame = wf_result.long_mask
    short_mask: pd.DataFrame = wf_result.short_mask

    panic_bars: pd.Series = _align_panic(panic, long_mask.index, cfg.frequency)
    weights: pd.DataFrame = build_weight_matrix(long_mask, short_mask, panic_bars, cfg)

    # Forward return on bar T = price[T+1]/price[T] - 1, computed on the FULL
    # price history so "next bar" is real, then aligned to the scored bars.
    # NB: the cross-session/overnight move is intentionally KEPT (a real return
    # on a held position) — this differs from the tree label's cross-session null.
    forward_returns: pd.DataFrame = price_wide.shift(-1) / price_wide - 1

    common: pd.Index = weights.index.intersection(forward_returns.index)
    weights = weights.loc[common]
    fwd: pd.DataFrame = forward_returns.reindex(index=common, columns=weights.columns)

    # min_count=1 keeps an all-NaN row (last bar, no T+1 price) as NaN so dropna
    # removes it cleanly rather than collapsing it to 0.0.
    gross: pd.Series = (weights * fwd).sum(axis=1, min_count=1)

    # Transaction-cost model (OFF unless ML_CONFIG.cost.apply_costs). Turnover is
    # the notional traded when rebalancing into each bar's book (first bar = a
    # full entry from cash); charged one-way, so a round-trip pays twice. Panic->
    # cash liquidations and re-entries are captured automatically by the diff.
    turnover: pd.Series = weights.diff().abs().sum(axis=1)
    if len(weights):
        turnover.iloc[0] = weights.iloc[0].abs().sum()

    cost_cfg = config.ML_CONFIG.cost
    net: pd.Series = gross
    trade_drag = borrow_drag = 0.0
    if cost_cfg.apply_costs:
        ppy: float = config.freq_spec(cfg.frequency).bars_per_day * 252.0
        trade_cost: pd.Series = turnover * (cost_cfg.oneway_bps / 1e4)
        short_gross: pd.Series = weights.clip(upper=0.0).abs().sum(axis=1)
        borrow_cost: pd.Series = short_gross * (cost_cfg.short_borrow_bps_annual / 1e4) / ppy
        net = gross - trade_cost - borrow_cost
        trade_drag = float(trade_cost.reindex(gross.index).mean() * ppy)
        borrow_drag = float(borrow_cost.reindex(gross.index).mean() * ppy)

    portfolio_returns: pd.Series = net.dropna()
    portfolio_returns.name = "portfolio_return"
    portfolio_returns.index.name = _DATE_LEVEL

    panic_final: pd.Series = panic_bars.reindex(portfolio_returns.index, fill_value=False)
    calm_bars: int = int((~panic_final.astype(bool)).sum())
    panic_bar_ct: int = int(panic_final.astype(bool).sum())
    label: str = (
        f"Calm: {calm_bars}b | Hedged: {panic_bar_ct}b"
        if cfg.execution_style == "dynamic_tilt"
        else f"Active: {calm_bars}b | Cash: {panic_bar_ct}b"
    )
    cost_str = (
        f" | costs ON (drag ~{trade_drag + borrow_drag:.1%}/yr)"
        if cost_cfg.apply_costs else " | gross"
    )
    print(
        f"[tier3_execution] style={cfg.execution_style} freq={cfg.frequency} | "
        f"{len(portfolio_returns)} bars | {label} | "
        f"avg turnover/bar={turnover.reindex(portfolio_returns.index).mean():.2f}{cost_str} | "
        f"Cumulative: {(1 + portfolio_returns).prod() - 1:.4%}"
    )
    return portfolio_returns


# --------------------------------------------------------------------------- #
# Dry-run verification — synthetic masks + prices + gate, assert gate semantics
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from src.production_ml.tier1_trees import TreeAlphaEngine, _make_dummy_features

    print("=" * 70)
    print("tier3_execution dry-run — masks + panic gate -> portfolio returns")
    print("=" * 70)

    N_TICKERS: int = 10
    dummy = _make_dummy_features(n_days=800, n_tickers=N_TICKERS)
    wf = TreeAlphaEngine(model_choice="lgbm").run_walk_forward(dummy)  # 1 learner = fast

    dates = wf.long_mask.index
    tickers = wf.long_mask.columns
    rng = np.random.default_rng(0)

    # Synthetic random-walk price panel on the scored (date x ticker) grid.
    steps = rng.normal(0.0, 0.02, size=(len(dates), len(tickers)))
    price_wide = pd.DataFrame(
        100.0 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=tickers
    )

    # Synthetic daily gate: ~15% Panic, arriving in a contiguous block + noise.
    panic = pd.Series(False, index=dates)
    block = slice(len(dates) // 3, len(dates) // 3 + len(dates) // 12)
    panic.iloc[block] = True
    panic |= pd.Series(rng.random(len(dates)) < 0.05, index=dates)
    print(f"  scored grid : {len(dates)} days x {len(tickers)} tickers | "
          f"Panic {panic.mean():.0%}")
    print()

    checks: dict[str, bool] = {}
    for style in ("long_only", "long_short", "dynamic_tilt"):
        cfg = StrategyConfig(
            is_simulation=False, market_type="dryrun", lookback_period=0,
            hmm_states=2, execution_style=style, frequency="daily",
        )
        W = build_weight_matrix(wf.long_mask, wf.short_mask, panic, cfg)
        ret = execute_ml_strategy(wf, panic, price_wide, cfg)

        # Gate semantics on weights (exact, index-independent of dropna).
        panic_rows = W.index[panic.reindex(W.index).fillna(False).astype(bool)]
        calm_rows = W.index[~panic.reindex(W.index).fillna(False).astype(bool)]
        gross = W.abs().sum(axis=1)
        net = W.sum(axis=1)

        if style == "long_only":
            checks["long_only: Panic -> cash"] = bool((gross.loc[panic_rows] == 0).all())
            checks["long_only: Calm gross~1.0"] = bool(np.allclose(gross.loc[calm_rows], 1.0))
        elif style == "long_short":
            checks["long_short: Panic -> cash"] = bool((gross.loc[panic_rows] == 0).all())
            checks["long_short: Calm net~0"] = bool(np.allclose(net.loc[calm_rows], 0.0, atol=1e-9))
            checks["long_short: Calm gross~2.0"] = bool(np.allclose(gross.loc[calm_rows], 2.0))
        else:
            checks["dynamic_tilt: never flat"] = bool((gross > 0).all())
            checks["dynamic_tilt: Calm net~+1.0"] = bool(np.allclose(net.loc[calm_rows], 1.0))
            checks["dynamic_tilt: Panic net~0"] = bool(np.allclose(net.loc[panic_rows], 0.0, atol=1e-9))

        checks[f"{style}: returns NaN-free"] = bool(ret.notna().all())

    # --- Cost model: OFF reproduces gross exactly; ON strictly reduces return -- #
    cfg = StrategyConfig(
        is_simulation=False, market_type="dryrun", lookback_period=0,
        hmm_states=2, execution_style="dynamic_tilt", frequency="daily",
    )
    gross_ret = execute_ml_strategy(wf, panic, price_wide, cfg)  # apply_costs=False default
    try:
        config.ML_CONFIG.cost.apply_costs = True
        config.ML_CONFIG.cost.oneway_bps = 10.0
        net_ret = execute_ml_strategy(wf, panic, price_wide, cfg)
    finally:
        config.ML_CONFIG.cost.apply_costs = False
    common = gross_ret.index.intersection(net_ret.index)
    checks["cost ON <= gross every bar"] = bool(
        (net_ret.loc[common] <= gross_ret.loc[common] + 1e-12).all()
    )
    checks["cost ON strictly reduces some bars"] = bool(
        (net_ret.loc[common] < gross_ret.loc[common]).any()
    )

    print()
    print("=" * 70)
    for name, ok in checks.items():
        print(f"  {name:<34}: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    assert all(checks.values()), "tier3_execution dry-run FAILED"
