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
# No-trade hysteresis buffer (turnover control) — the one SEQUENTIAL step
# --------------------------------------------------------------------------- #
def apply_rebalance_buffer(
    alpha_scores: pd.DataFrame,
    cfg: StrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Two-band hysteresis on decile membership to suppress boundary churn.

    A name ENTERS a leg at the top/bottom ``decile_pct`` band but is only EVICTED
    once it drifts past the wider ``decile_pct * rebalance_buffer_mult`` exit band
    — so a name jittering across the decile edge (rank k <-> k+1) no longer forces
    a costly round-trip. Incumbent-preferring per side: retain held names still
    inside the exit band (best-ranked first), then fill remaining slots up to the
    enter width ``k_enter`` with the best-ranked names in the enter band. Book size
    stays ~``k_enter``, so 1/k weights and per-leg gross/net exposure are unchanged
    vs raw deciles.

    The LONG leg ranks by DESCENDING alpha (best first); the SHORT leg ranks by
    ASCENDING alpha (worst first) — the identical hysteresis, mirrored. Applied to
    both legs for ``long_short`` / ``dynamic_tilt``; ``long_only`` computes the long
    leg only. ``buffer_mult == 1.0`` reproduces the raw per-bar deciles exactly
    (enter band == exit band), which is why ``execute_ml_strategy`` skips this call
    entirely in that case.

    Panic is intentionally NOT handled here: the de-risk gate zeroes the book in
    ``build_weight_matrix``, which already charges the liquidation + re-entry
    turnover via ``weights.diff()``. Keeping the buffer panic-agnostic makes it a
    pure function of the alpha ranks (and makes mult==1.0 an exact no-op).

    This is the only deliberately SEQUENTIAL step in Tier 3 (hysteresis is
    path-dependent and cannot be vectorized over time), but the per-bar work is
    O(k) set ops on precomputed cross-sectional ranks — trivial vs the tree fits.
    """
    decile_pct: float = cfg.decile_pct
    mult: float = cfg.rebalance_buffer_mult
    do_short: bool = cfg.execution_style in ("long_short", "dynamic_tilt")

    index: pd.Index = alpha_scores.index
    columns: pd.Index = alpha_scores.columns
    n_bars: int = len(index)

    # Cross-sectional ranks computed once, vectorized (NaN where a name is absent
    # that bar). rank 1 = best-alpha (desc) / worst-alpha (asc).
    rank_desc: np.ndarray = alpha_scores.rank(axis=1, ascending=False, method="first").to_numpy()
    rank_asc: np.ndarray = alpha_scores.rank(axis=1, ascending=True, method="first").to_numpy()
    valid_counts: np.ndarray = alpha_scores.notna().sum(axis=1).to_numpy()

    held_long: np.ndarray = np.zeros((n_bars, len(columns)), dtype=np.int8)
    held_short: np.ndarray = np.zeros((n_bars, len(columns)), dtype=np.int8)

    def _buffer_leg(rank_row: np.ndarray, prev: list[int], k_enter: int, k_exit: int) -> list[int]:
        # Retain incumbents still inside the (wider) exit band, best-ranked first.
        # A NaN rank (name left the universe) fails ``<= k_exit`` and is dropped.
        retained: list[int] = [c for c in prev if rank_row[c] <= k_exit]
        retained.sort(key=lambda c: rank_row[c])
        if len(retained) >= k_enter:
            return retained[:k_enter]
        # Fill remaining slots with the best-ranked names in the (tighter) enter band.
        enter_cols: np.ndarray = np.where(rank_row <= k_enter)[0]
        enter_cols = enter_cols[np.argsort(rank_row[enter_cols])]
        keep: set[int] = set(retained)
        fill: list[int] = [int(c) for c in enter_cols if int(c) not in keep]
        return retained + fill[: k_enter - len(retained)]

    prev_long: list[int] = []
    prev_short: list[int] = []
    for t in range(n_bars):
        n: int = int(valid_counts[t])
        if n == 0:
            prev_long, prev_short = [], []
            continue
        k_enter: int = max(1, int(np.floor(n * decile_pct)))
        # Exit band wider by mult, capped at n//2 so the long and short hold-zones
        # can never overlap (a name can't be sticky-long and sticky-short at once).
        k_exit: int = max(k_enter, min(int(np.floor(n * decile_pct * mult)), n // 2))

        new_long: list[int] = _buffer_leg(rank_desc[t], prev_long, k_enter, k_exit)
        held_long[t, new_long] = 1
        prev_long = new_long

        if do_short:
            new_short: list[int] = _buffer_leg(rank_asc[t], prev_short, k_enter, k_exit)
            held_short[t, new_short] = 1
            prev_short = new_short

    held_long_mask: pd.DataFrame = pd.DataFrame(held_long, index=index, columns=columns)
    held_short_mask: pd.DataFrame = pd.DataFrame(-held_short, index=index, columns=columns)  # -1/0
    return held_long_mask, held_short_mask


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


def apply_terminal_returns(
    fwd: pd.DataFrame,
    price_wide: pd.DataFrame,
    terminal_returns: pd.Series,
) -> pd.DataFrame:
    """
    Give every delisted name an explicit exit return on its final bar.

    Without this the panel leaks survivorship back in through the exit. A delisted
    name's last bar has no T+1 price, so its forward return is NaN — and because
    ``.sum()`` skips NaN, a position held into a delisting is silently realized at
    **exactly 0%**. The strategy would take every delisting for free, which is the
    single most flattering bug a survivorship-free panel can still have.

    Only names whose last quote precedes the end of the sample are filled; a NaN on
    the final bar of the panel means "the backtest ended", not "the company died".

    Args:
        fwd: (bar × ticker) forward returns, already aligned to the scored bars.
        price_wide: full (bar × ticker) price history, used to find each name's
            last quote — ``fwd`` alone cannot distinguish a delisting from the
            edge of the scored window.
        terminal_returns: ticker → exit return (−0.30 for a delisting, 0.0 for an
            acquisition; see ``bhavcopy_panel.classify_lifecycle``).
    """
    out = fwd.copy()
    sample_end = price_wide.index[-1]
    n_filled = 0

    for ticker, ret in terminal_returns.dropna().items():
        if ticker not in out.columns:
            continue
        last = price_wide[ticker].last_valid_index()
        if last is None or last >= sample_end or last not in out.index:
            continue
        out.at[last, ticker] = float(ret)
        n_filled += 1

    print(f"[tier3] terminal returns applied to {n_filled} delisted/acquired names")
    return out


def assert_no_implicit_exit(weights: pd.DataFrame, fwd: pd.DataFrame) -> None:
    """
    Fail loudly if any held position would be realized at an implicit 0%.

    This is the guard that keeps the leak closed: a weight on a bar whose forward
    return is NaN contributes nothing to the sum, which reads as a flat exit. The
    final bar is exempt — there is genuinely no next price for anything there.
    """
    if len(weights) < 2:
        return
    held = weights.iloc[:-1].abs() > 0
    unpriced = fwd.iloc[:-1].isna()
    bad = held & unpriced
    n_bad = int(bad.to_numpy().sum())
    if n_bad:
        where = bad.stack()
        where = where[where]
        sample = ", ".join(f"{t} @ {d.date()}" for d, t in where.index[:5])
        raise AssertionError(
            f"{n_bad} held position(s) have no forward return and would exit at an "
            f"implicit 0% — supply terminal_returns for these names. First: {sample}"
        )


# --------------------------------------------------------------------------- #
# Strategy execution
# --------------------------------------------------------------------------- #
def execute_ml_strategy(
    wf_result: WalkForwardResult,
    panic: pd.Series,
    price_wide: pd.DataFrame,
    cfg: StrategyConfig,
    terminal_returns: pd.Series | None = None,
) -> pd.Series:
    """
    Realize one PRODUCTION_ML strategy's portfolio return series.

    Args:
        wf_result:  Tier 1 walk-forward output (long/short decile masks).
        panic:      Tier 2 daily de-risk gate (bool). Broadcast to bars for
                    intraday frequencies.
        price_wide: (bar x ticker) close prices for ``cfg.frequency``.
        cfg:        StrategyConfig (execution_style, frequency).
        terminal_returns: optional ticker -> exit return for names that leave the
                    exchange mid-sample. Supplied by the Phase 4 point-in-time
                    panel; ``None`` (the default) keeps the Phase 2/3 behaviour on
                    the 68-name panel, where no name delists inside the window.

    Returns:
        Net portfolio return per bar, indexed by bar timestamp, NaN-free.
    """
    # No-trade hysteresis buffer (turnover control). mult<=1.0 keeps the raw
    # per-bar deciles (exact no-op); >1.0 makes decile membership sticky so
    # boundary jitter (rank k <-> k+1) stops forcing round-trips.
    if cfg.rebalance_buffer_mult > 1.0:
        long_mask, short_mask = apply_rebalance_buffer(wf_result.alpha_scores, cfg)
    else:
        long_mask = wf_result.long_mask
        short_mask = wf_result.short_mask

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

    # Delisting exits. Without this a position held into a delisting is realized at
    # an implicit 0% (see apply_terminal_returns), quietly restoring the survivorship
    # bias the point-in-time panel exists to remove.
    if terminal_returns is not None:
        fwd = apply_terminal_returns(fwd, price_wide, terminal_returns)
        assert_no_implicit_exit(weights, fwd)

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
        trade_cost: pd.Series = turnover * (cost_cfg.compose_oneway_bps() / 1e4)
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
    from src.production_ml.tier1_trees import (
        TreeAlphaEngine, _make_dummy_features, DECILE_PCT,
    )

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
        rebalance_buffer_mult=1.0,  # isolate the cost effect from the buffer
    )
    try:
        config.ML_CONFIG.cost.apply_costs = False
        gross_ret = execute_ml_strategy(wf, panic, price_wide, cfg)
        config.ML_CONFIG.cost.apply_costs = True
        net_ret = execute_ml_strategy(wf, panic, price_wide, cfg)
    finally:
        config.ML_CONFIG.cost.apply_costs = True  # restore default (net is the model)
    common = gross_ret.index.intersection(net_ret.index)
    checks["cost ON <= gross every bar"] = bool(
        (net_ret.loc[common] <= gross_ret.loc[common] + 1e-12).all()
    )
    checks["cost ON strictly reduces some bars"] = bool(
        (net_ret.loc[common] < gross_ret.loc[common]).any()
    )
    checks["NSE compose_oneway_bps in ~14-15 bps"] = bool(
        13.0 < config.ML_CONFIG.cost.compose_oneway_bps() < 16.0
    )

    # --- Rebalance buffer: cuts turnover; both legs sized k_enter; mult=1.0 no-op #
    cfg_buf = StrategyConfig(
        is_simulation=False, market_type="dryrun", lookback_period=0,
        hmm_states=2, execution_style="long_short", frequency="daily",
        rebalance_buffer_mult=2.0,
    )
    cfg_nobuf = StrategyConfig(
        is_simulation=False, market_type="dryrun", lookback_period=0,
        hmm_states=2, execution_style="long_short", frequency="daily",
        rebalance_buffer_mult=1.0,
    )
    hl_b, hs_b = apply_rebalance_buffer(wf.alpha_scores, cfg_buf)
    hl_n, hs_n = apply_rebalance_buffer(wf.alpha_scores, cfg_nobuf)

    def _total_turnover(lm, sm, c):
        W = build_weight_matrix(lm, sm, panic, c)
        tv = W.diff().abs().sum(axis=1)
        if len(W):
            tv.iloc[0] = W.iloc[0].abs().sum()
        return float(tv.sum())

    turn_buf = _total_turnover(hl_b, hs_b, cfg_buf)
    turn_nobuf = _total_turnover(hl_n, hs_n, cfg_nobuf)

    n_valid = wf.alpha_scores.notna().sum(axis=1)
    k_enter = np.floor(n_valid * DECILE_PCT).clip(lower=1).astype(int)
    long_ct = (hl_b == 1).sum(axis=1)
    short_ct = (hs_b == -1).sum(axis=1)

    checks["buffer cuts turnover"] = bool(turn_buf < turn_nobuf)
    checks["buffer mult=1.0 == raw long deciles"] = bool(
        (hl_n.to_numpy() == wf.long_mask.to_numpy()).all()
    )
    checks["buffer mult=1.0 == raw short deciles"] = bool(
        (hs_n.to_numpy() == wf.short_mask.to_numpy()).all()
    )
    checks["buffered long leg sized k_enter"] = bool((long_ct == k_enter).all())
    checks["buffered short leg sized k_enter"] = bool((short_ct == k_enter).all())
    checks["short buffer engages (differs from raw)"] = bool(
        not (hs_b.to_numpy() == wf.short_mask.to_numpy()).all()
    )

    print()
    print("=" * 70)
    for name, ok in checks.items():
        print(f"  {name:<34}: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    assert all(checks.values()), "tier3_execution dry-run FAILED"
