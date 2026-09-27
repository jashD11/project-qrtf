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
from dataclasses import dataclass
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

# Phase 4c adds two styles. ``*_slb`` is the same book with the short leg restricted to
# names that were actually borrowable that day (Track B) — a different strategy, not a
# correction to the unrestricted one, so it carries its own id and its own ledger cell.
_BASE_STYLES: Final[frozenset[str]] = frozenset(
    {"long_only", "long_short", "dynamic_tilt"}
)
_SLB_SUFFIX: Final[str] = "_slb"
_VALID_STYLES: Final[frozenset[str]] = _BASE_STYLES | frozenset(
    {f"{s}{_SLB_SUFFIX}" for s in ("long_short", "dynamic_tilt")}
)
_DATE_LEVEL: Final[str] = "date"
_TICKER_LEVEL: Final[str] = "ticker"


def base_style(style: str) -> str:
    """The weighting rule behind a style name, with any ``_slb`` restriction stripped."""
    return style[: -len(_SLB_SUFFIX)] if style.endswith(_SLB_SUFFIX) else style


def restricts_shorts(style: str) -> bool:
    """True when this style's short leg must be drawn from the borrowable set only."""
    return style.endswith(_SLB_SUFFIX)


@dataclass
class ExecutionResult:
    """
    Everything one Tier 3 run produced, not just the column that lands in the ledger.

    Phase 4b returned the net series alone, so recovering gross Sharpe meant a second
    ~80-minute run and the cost drag could only be inferred (E2). Costs, turnover and
    exposure are all computed here anyway; keeping them is free and makes the §6
    diagnostics measurements rather than reconstructions.
    """
    net: pd.Series               # after costs — the ledger column
    gross: pd.Series             # before any cost
    turnover: pd.Series          # sum |dw| per bar
    trade_cost: pd.Series
    borrow_cost: pd.Series
    net_exposure: pd.Series      # sum w per bar (drives the dividend accrual, E5)
    short_gross: pd.Series
    diagnostics: dict            # scalars worth printing / recording
    weights: pd.DataFrame | None = None  # final (bar x ticker) book, for read-only
                                         # diagnostics (Phase 5 swap attribution,
                                         # short-leg oracle); never written to a ledger


# --------------------------------------------------------------------------- #
# Weight construction (fully vectorized — no loop over the time dimension)
# --------------------------------------------------------------------------- #
def cap_leg_weights(
    leg_w: pd.DataFrame, max_weight: pd.DataFrame, n_iter: int = 12
) -> pd.DataFrame:
    """
    Cap each name's weight at ``max_weight`` while keeping the leg's gross at 1.0 (A4).

    ``ML_CONFIG.cost.max_participation`` bounds the share of a name's ADV one position
    may consume, which is a bound on that name's *weight*: ``w_i <= cap · ADV_i / AUM``.
    Nothing downstream renormalizes, so simply truncating an over-cap weight would
    quietly shrink gross exposure and turn a liquidity constraint into an unannounced
    de-leveraging. The excess is therefore **redistributed within the leg**, so leg
    gross stays 1.0 and the dry-run's ``gross~1.0 / ~2.0 / net~0`` invariants continue
    to hold.

    This is water-filling: cap the offenders, spread their excess over the names with
    headroom in proportion to current weight, repeat. It converges quickly (a decile
    book is ~50 roughly equal weights, so few names bind at once).

    When a whole row's capacity ``sum(max_weight)`` is below 1.0 the leg genuinely
    cannot be built at this AUM; the row is left at its capacity and reported rather
    than forced, because forcing it would be the fiction the cap exists to prevent.
    """
    w = leg_w.copy()
    cap = max_weight.reindex_like(w)
    # A name with no ADV estimate is left uncapped rather than capped at zero: an
    # unmeasured name is not a name known to be untradeable, and zeroing it here would
    # silently drop it from the book instead of pricing it.
    cap = cap.where(cap.notna(), np.inf)

    for _ in range(n_iter):
        over = w > cap
        if not bool(over.to_numpy().any()):
            break
        excess: pd.Series = (w - cap).where(over, 0.0).sum(axis=1)
        w = w.where(~over, cap)
        # Names still below their cap and already in the leg absorb the excess.
        headroom = (w.gt(0.0) & w.lt(cap))
        base: pd.DataFrame = w.where(headroom, 0.0)
        denom: pd.Series = base.sum(axis=1)
        share = base.div(denom.replace(0.0, np.nan), axis=0).fillna(0.0)
        w = w + share.mul(excess, axis=0)

    return w


def build_weight_matrix(
    long_mask: pd.DataFrame,
    short_mask: pd.DataFrame,
    panic: pd.Series,
    cfg: StrategyConfig,
    max_weight: pd.DataFrame | None = None,
    short_scale: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Build the (bar x ticker) portfolio weight matrix for one execution style.

    Args:
        long_mask:  (bar x ticker) 1 = long, 0 = flat (Tier 1).
        short_mask: (bar x ticker) -1 = short, 0 = flat (Tier 1).
        panic:      de-risk gate aligned to the mask's bar index (bool-ish;
                    reindexed + NaN->False here for safety).
        cfg:        carries ``execution_style``.
        max_weight: optional (bar x ticker) per-name weight ceiling from the
                    participation cap (A4). Applied to each leg's ``1/k`` weights
                    **before** the style branch, so all styles inherit it, with the
                    residual redistributed inside the leg (see ``cap_leg_weights``).
        short_scale: optional per-bar multiplier in [0, 1] on the short leg's weights
                    (Phase 5 §5 step 2 — the short-leg exposure scalar the oracle
                    ceiling test drives). ``None`` takes the exact pre-Phase-5 path.

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
    style: str = base_style(cfg.execution_style)

    # Per-day book width k varies (decile_pct of a shifting cross-section), so
    # each selected name is 1/k that day. Empty side -> NaN count -> zero weights.
    # Under an SLB restriction the short leg may hold fewer than k names; dividing by
    # the *actual* count keeps short gross at 1.0, so dollar-neutrality survives and
    # the restriction shows up as concentration (and hence impact), which is correct.
    long_count: pd.Series = long_mask.sum(axis=1).replace(0, np.nan)
    short_count: pd.Series = short_mask.abs().sum(axis=1).replace(0, np.nan)
    long_w: pd.DataFrame = long_mask.div(long_count, axis=0).fillna(0.0)
    short_w: pd.DataFrame = short_mask.abs().div(short_count, axis=0).fillna(0.0)

    if max_weight is not None:
        long_w = cap_leg_weights(long_w, max_weight)
        short_w = cap_leg_weights(short_w, max_weight)

    if short_scale is not None:
        scale = short_scale.reindex(short_w.index)
        if bool(scale.isna().any()) or bool(((scale < 0.0) | (scale > 1.0)).any()):
            raise ValueError("short_scale must cover every bar with values in [0, 1]")
        short_w = short_w.mul(scale.astype(float), axis=0)

    panic_bool: pd.Series = (
        panic.reindex(long_mask.index, fill_value=False).astype(bool)
    )

    if style == "long_only":
        weights: pd.DataFrame = long_w
        # Panic -> cash: zero the whole row (broadcast the per-bar mask down cols).
        weights = weights.where(~panic_bool, 0.0, axis=0)

    elif style == "long_short":
        weights = long_w - short_w
        weights = weights.where(~panic_bool, 0.0, axis=0)

    else:  # dynamic_tilt — regime-conditional leverage, never flat.
        ex = config.ML_CONFIG.execution
        flag: np.ndarray = panic_bool.to_numpy(dtype=float)[:, None]  # (bars, 1)
        long_lev: np.ndarray = ex.calm_long_lev * (1.0 - flag) + ex.panic_long_lev * flag
        short_lev: np.ndarray = ex.calm_short_lev * (1.0 - flag) + ex.panic_short_lev * flag
        weights = pd.DataFrame(
            long_lev * long_w.to_numpy() - short_lev * short_w.to_numpy(),
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
    short_scores: pd.DataFrame | None = None,
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

    ``short_scores`` (B2) is the short leg's candidate set: the same alpha scores with
    names that were not borrowable that bar blanked to NaN. Passing it here — rather
    than filtering the finished mask — is what makes the eligibility filter *compose*
    with the buffer instead of fighting it. The short leg then re-ranks inside the
    eligible subset, so it still fills toward ``k_enter`` names and stays sticky about
    the ones it holds; filtering afterwards would instead shrink the book silently and
    break dollar-neutrality. Book width ``k`` is deliberately still taken from the
    **full** cross-section, so the restriction changes *which* names are held, not how
    many the strategy is trying to hold.

    This is the only deliberately SEQUENTIAL step in Tier 3 (hysteresis is
    path-dependent and cannot be vectorized over time), but the per-bar work is
    O(k) set ops on precomputed cross-sectional ranks — trivial vs the tree fits.
    """
    decile_pct: float = cfg.decile_pct
    mult: float = cfg.rebalance_buffer_mult
    do_short: bool = base_style(cfg.execution_style) in ("long_short", "dynamic_tilt")
    short_src: pd.DataFrame = alpha_scores if short_scores is None else short_scores

    index: pd.Index = alpha_scores.index
    columns: pd.Index = alpha_scores.columns
    n_bars: int = len(index)

    # Cross-sectional ranks computed once, vectorized (NaN where a name is absent
    # that bar). rank 1 = best-alpha (desc) / worst-alpha (asc). An ineligible name
    # carries a NaN rank, which fails every ``<= k`` test and so can never be selected
    # or retained — the filter needs no separate branch.
    rank_desc: np.ndarray = alpha_scores.rank(axis=1, ascending=False, method="first").to_numpy()
    rank_asc: np.ndarray = short_src.rank(axis=1, ascending=True, method="first").to_numpy()
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


def forward_returns(price_wide: pd.DataFrame, bridge_halts: bool = False) -> pd.DataFrame:
    """
    Forward return on bar T = price[T+1]/price[T] - 1, on the FULL price history.

    The cross-session/overnight move is intentionally KEPT (a real return on a held
    position) — this differs from the tree label's cross-session null.

    ``bridge_halts`` prices a position held into a **trading halt**. A suspended name
    stops printing for a stretch and then resumes, so its next-bar price is NaN even
    though nothing was exited — and because ``.sum()`` skips NaN, the halt would be
    realized at exactly 0%. That is the same exit-side survivorship flattery
    ``apply_terminal_returns`` closes for delistings: across this panel's 41 halt
    cells the avoided move averages **-6.1%** (range -59% to +60%). You cannot sell a
    suspended stock, so the position is carried to the next available quote and the
    move is booked on the last bar that traded.

    ``bfill`` stops at each name's last quote, so cells past a genuine delisting stay
    NaN and are still filled by ``apply_terminal_returns`` — halts and exits do not
    collide. Off by default: the Phase 2/3 panels carry internal holes of their own
    (17 daily cells), so enabling this unconditionally would silently restate Phase 3.
    """
    nxt = price_wide.bfill().shift(-1) if bridge_halts else price_wide.shift(-1)
    return nxt / price_wide - 1


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
# Phase 4c A3 — per-name execution cost panels
# --------------------------------------------------------------------------- #
@dataclass
class CostPanels:
    """The per-(date, name) inputs the Phase 4c cost model needs, already aligned."""
    adv: pd.DataFrame               # trailing median rupee turnover
    sigma: pd.DataFrame             # trailing daily return stdev
    half_spread_bps: pd.DataFrame   # measured half-spread, bps
    shortable: pd.DataFrame | None  # point-in-time borrowable set (bool), or None


def align_cost_panel(
    panel: pd.DataFrame,
    index: pd.Index,
    columns: pd.Index,
    label: str,
    traded: pd.DataFrame | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Reindex a wide cost panel onto the strategy's bars and names, filling every gap.

    **Nothing may be left NaN.** ``(traded * cost_bps).sum(axis=1)`` skips NaN, so a
    name with no spread or ADV estimate would be charged *nothing* — silently pricing
    an unmeasured name as free to trade, which is exactly the assumption Phase 4b made
    universally and this phase exists to remove. A gap is missing information, not
    evidence of a costless trade.

    A cell with no estimate takes that **date's** cross-sectional median (its liquidity
    peers on the same day); a date with no estimates at all takes the panel-wide
    median. Coverage is reported over the cells that were actually **traded** — the
    only ones whose fill can change a number — so a thin panel cannot pass unnoticed.
    """
    out = panel.reindex(index=index, columns=columns)
    if verbose:
        scope = out.where(traded.abs() > 0) if traded is not None else out
        denom = int(scope.notna().sum().sum() + scope.isna().sum().sum()) if traded is None else int((traded.abs() > 0).sum().sum())
        covered = float(scope.notna().sum().sum() / max(denom, 1))
        print(f"[tier3] cost panel '{label}': {covered:.1%} of traded cells measured "
              f"(the rest take that date's cross-sectional median)")
    out = out.T.fillna(out.median(axis=1)).T
    return out.fillna(float(np.nanmedian(panel.to_numpy())))


def load_cost_panels(need_shortable: bool = False) -> CostPanels:
    """
    Read the Phase 4c cost panels from disk, with errors naming the build step.

    Built by ``src/phase4_data/liquidity.py`` and ``src/phase4_data/spread.py``. Read
    here as plain Parquet rather than by importing those modules, so the production_ml
    package stays self-contained (the two source trees are deliberately isolated).
    """
    p4 = config.ML_CONFIG.phase4
    cost = config.ML_CONFIG.cost
    spread_path = (
        p4.spread_cs_parquet if cost.spread_estimator == "cs" else p4.spread_ar_parquet
    )
    if cost.spread_estimator not in ("cs", "ar"):
        raise ValueError(f"spread_estimator must be 'cs' or 'ar', got {cost.spread_estimator!r}")

    def _read(path: str, builder: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found — run `{builder}` first.")
        frame = pd.read_parquet(path)
        frame.index = pd.DatetimeIndex(frame.index).normalize()
        return frame

    build_liq = "python src/phase4_data/liquidity.py --build"
    build_spr = "python src/phase4_data/spread.py --build"
    shortable = None
    if need_shortable:
        shortable = _read(
            p4.shortable_mask_parquet,
            "python src/phase4_data/shortable.py --build",
        ).astype(bool)
    return CostPanels(
        adv=_read(p4.adv_parquet, build_liq),
        sigma=_read(p4.sigma_parquet, build_liq),
        half_spread_bps=_read(spread_path, build_spr),
        shortable=shortable,
    )


def per_name_cost_bps(
    traded: pd.DataFrame, panels: CostPanels, cost_cfg
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-way execution cost in bps for every (bar, name), and the participation behind it.

        cost_bps = statutory (flat, citable)
                 + measured half-spread for that name on that date   (A1)
                 + impact_bps(sigma, participation)                  (sqrt law)

    ``participation`` is the traded notional as a share of that name's trailing ADV —
    which is why ``traded`` must arrive **unreduced**. Phase 4b collapsed it to a
    per-bar scalar with ``.sum(axis=1)`` before costing, at which point no per-name term
    can be charged at all.

    The impact function is ``config._NSECostConfig.impact_bps``'s formula, evaluated
    vectorially; ``capacity.py`` already asserts the two agree pointwise, so there is
    one law here, not a second copy.
    """
    idx, cols = traded.index, traded.columns
    adv = align_cost_panel(panels.adv, idx, cols, "adv", traded)
    sigma = align_cost_panel(panels.sigma, idx, cols, "sigma", traded)
    spread = align_cost_panel(panels.half_spread_bps, idx, cols, "half_spread", traded)

    notional = traded.abs() * cost_cfg.aum_rupees
    participation = (notional / adv).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    impact = 1e4 * cost_cfg.impact_coef * sigma * np.sqrt(participation)

    cost_bps = cost_cfg.compose_oneway_bps() + spread + impact
    if bool(cost_bps.isna().to_numpy().any()):
        raise AssertionError(
            "per-name cost has NaN cells after alignment — those trades would be "
            "charged nothing at all, silently reinstating the zero-cost assumption "
            "this phase exists to remove."
        )
    return cost_bps, participation


def borrow_bps_panel(
    short_w: pd.DataFrame, panels: CostPanels, cost_cfg
) -> pd.DataFrame:
    """
    Per-name annual borrow rate in bps (B3), by trailing-turnover quartile.

    The flat ``short_borrow_bps_annual`` assumes unlimited availability at a uniform
    price. Real SLB is deep and cheap for the large F&O names and scarce and expensive
    down the ladder, so the rate is tiered on the same trailing-turnover measure the
    universe already ranks on. The tiers are frozen a-priori from published SLB fee
    ranges — a stated assumption, not a fitted parameter.
    """
    if not cost_cfg.tiered_borrow:
        return pd.DataFrame(
            cost_cfg.short_borrow_bps_annual, index=short_w.index, columns=short_w.columns
        )
    adv = align_cost_panel(
        panels.adv, short_w.index, short_w.columns, "adv(borrow)", verbose=False
    )
    # Rank only among names actually held short that bar, so the tiers describe the
    # book being borrowed rather than the whole universe.
    held_adv = adv.where(short_w.abs() > 0)
    q = held_adv.rank(axis=1, ascending=False, pct=True)
    tiers = np.asarray(cost_cfg.borrow_bps_by_quartile, dtype=float)
    bucket = np.ceil(q.to_numpy() * len(tiers))
    bucket = np.clip(np.nan_to_num(bucket, nan=len(tiers)), 1, len(tiers)).astype(int)
    return pd.DataFrame(
        tiers[bucket - 1], index=short_w.index, columns=short_w.columns
    )


def _short_mask_from_scores(
    short_scores: pd.DataFrame, k_per_day: pd.Series
) -> pd.DataFrame:
    """
    Bottom-``k`` short mask from a (possibly eligibility-filtered) score frame.

    Mirrors ``tier1_trees._allocate_deciles``'s short side exactly, but ranks within
    whatever candidate set ``short_scores`` carries. Only needed on the no-buffer path
    (``rebalance_buffer_mult == 1.0``); the buffered path filters inside
    ``apply_rebalance_buffer``.
    """
    rank_asc = short_scores.rank(axis=1, ascending=True, method="first")
    mask = rank_asc.le(k_per_day.to_numpy()[:, None]).astype(int) * -1
    return mask.where(short_scores.notna(), 0)


# --------------------------------------------------------------------------- #
# Strategy execution
# --------------------------------------------------------------------------- #
def run_execution(
    wf_result: WalkForwardResult,
    panic: pd.Series,
    price_wide: pd.DataFrame,
    cfg: StrategyConfig,
    terminal_returns: pd.Series | None = None,
    panels: CostPanels | None = None,
    short_scale: pd.Series | None = None,
) -> ExecutionResult:
    """
    Realize one PRODUCTION_ML strategy, returning gross, net and every cost component.

    Args:
        wf_result:  Tier 1 walk-forward output (alpha scores + decile masks).
        panic:      Tier 2 daily de-risk gate (bool). Broadcast to bars for
                    intraday frequencies.
        price_wide: (bar x ticker) close prices for ``cfg.frequency``.
        cfg:        StrategyConfig (execution_style, frequency).
        terminal_returns: optional ticker -> exit return for names that leave the
                    exchange mid-sample. Supplied by the Phase 4 point-in-time
                    panel; ``None`` (the default) keeps the Phase 2/3 behaviour on
                    the 68-name panel, where no name delists inside the window.
        panels:     optional pre-loaded Phase 4c cost panels. Loaded on demand when
                    absent and needed; passing them in avoids re-reading Parquet
                    once per style.
        short_scale: optional per-bar short-leg multiplier in [0, 1], passed through
                    to ``build_weight_matrix``. ``None`` (the default) is the exact
                    pre-Phase-5 path.

    Phase 4c behaviour (per-name spread/impact, participation cap, SLB restriction,
    tiered borrow) is entirely gated on ``ML_CONFIG.cost`` flags that default to False
    and on the ``_slb`` style suffix, so a default-config run reproduces Phase 3 and
    Phase 4b bit-for-bit. That is the same opt-in discipline as the halt bridge.
    """
    cost_cfg = config.ML_CONFIG.cost
    need_shortable: bool = restricts_shorts(cfg.execution_style)
    need_panels: bool = cost_cfg.apply_costs and (
        cost_cfg.charge_per_name or cost_cfg.enforce_participation_cap
        or cost_cfg.tiered_borrow or need_shortable
    )
    if need_panels and panels is None:
        panels = load_cost_panels(need_shortable=need_shortable)
    if need_shortable and (panels is None or panels.shortable is None):
        raise ValueError(
            f"execution_style={cfg.execution_style!r} restricts the short leg to "
            "borrowable names but no shortable mask was supplied — build it with "
            "`python src/phase4_data/shortable.py --build`."
        )

    alpha: pd.DataFrame = wf_result.alpha_scores

    # B2 — the short leg's candidate set. Blanking ineligible names in the *scores*
    # (rather than in the finished mask) is what makes the filter compose with the
    # buffer and keeps ``k`` intact: see apply_rebalance_buffer's docstring.
    short_scores: pd.DataFrame | None = None
    if need_shortable:
        eligible = (
            panels.shortable
            .reindex(index=alpha.index, columns=alpha.columns)
            .fillna(False)
            .astype(bool)
        )
        short_scores = alpha.where(eligible)

    # No-trade hysteresis buffer (turnover control). mult<=1.0 keeps the raw
    # per-bar deciles (exact no-op); >1.0 makes decile membership sticky so
    # boundary jitter (rank k <-> k+1) stops forcing round-trips.
    if cfg.rebalance_buffer_mult > 1.0:
        long_mask, short_mask = apply_rebalance_buffer(alpha, cfg, short_scores=short_scores)
    else:
        long_mask = wf_result.long_mask
        short_mask = wf_result.short_mask
        if short_scores is not None:
            n_valid = alpha.notna().sum(axis=1)
            k_per_day = np.floor(n_valid * cfg.decile_pct).clip(lower=1).astype(int)
            short_mask = _short_mask_from_scores(short_scores, k_per_day)

    panic_bars: pd.Series = _align_panic(panic, long_mask.index, cfg.frequency)

    # A4 — participation cap as a per-name weight ceiling, applied inside
    # build_weight_matrix before the style branch so every style inherits it.
    max_weight: pd.DataFrame | None = None
    if cost_cfg.apply_costs and cost_cfg.enforce_participation_cap:
        # Same filled ADV the cost model charges on, so a name is not simultaneously
        # "too unmeasured to cap" and "measured enough to charge".
        adv = align_cost_panel(
            panels.adv, long_mask.index, long_mask.columns, "adv(cap)", verbose=False
        )
        max_weight = cost_cfg.max_participation * adv / cost_cfg.aum_rupees

    weights: pd.DataFrame = build_weight_matrix(
        long_mask, short_mask, panic_bars, cfg, max_weight=max_weight,
        short_scale=short_scale,
    )

    # Forward returns on the full price history, then aligned to the scored bars.
    # Halts are bridged only on the Phase 4 path (where terminal_returns is supplied),
    # so the Phase 2/3 panels reproduce bit-for-bit — see forward_returns().
    fwd_full: pd.DataFrame = forward_returns(
        price_wide, bridge_halts=terminal_returns is not None
    )

    common: pd.Index = weights.index.intersection(fwd_full.index)
    weights = weights.loc[common]
    fwd: pd.DataFrame = fwd_full.reindex(index=common, columns=weights.columns)

    # Delisting exits. Without this a position held into a delisting is realized at
    # an implicit 0% (see apply_terminal_returns), quietly restoring the survivorship
    # bias the point-in-time panel exists to remove.
    if terminal_returns is not None:
        fwd = apply_terminal_returns(fwd, price_wide, terminal_returns)
        assert_no_implicit_exit(weights, fwd)

    # min_count=1 keeps an all-NaN row (last bar, no T+1 price) as NaN so dropna
    # removes it cleanly rather than collapsing it to 0.0.
    gross: pd.Series = (weights * fwd).sum(axis=1, min_count=1)

    # A3 — ``traded`` stays a (bar x ticker) FRAME. Phase 4b reduced it to a per-bar
    # scalar here, which is exactly why no per-name cost could be charged: once the
    # cross-section is summed away there is no name left to look up a spread or an ADV
    # for. Turnover is the notional traded when rebalancing into each bar's book
    # (first bar = a full entry from cash); charged one-way, so a round-trip pays
    # twice. Panic->cash liquidations and re-entries are captured by the diff.
    traded: pd.DataFrame = weights.diff().abs()
    if len(weights):
        traded.iloc[0] = weights.iloc[0].abs()
    turnover: pd.Series = traded.sum(axis=1)

    zero = pd.Series(0.0, index=weights.index)
    net: pd.Series = gross
    trade_cost = borrow_cost = zero
    trade_drag = borrow_drag = 0.0
    diagnostics: dict = {}

    if cost_cfg.apply_costs:
        ppy: float = config.freq_spec(cfg.frequency).bars_per_day * 252.0
        short_w: pd.DataFrame = weights.clip(upper=0.0).abs()
        short_gross: pd.Series = short_w.sum(axis=1)

        if cost_cfg.charge_per_name:
            cost_bps, participation = per_name_cost_bps(traded, panels, cost_cfg)
            trade_cost = (traded * cost_bps / 1e4).sum(axis=1)
            traded_sum = turnover.replace(0.0, np.nan)
            eff_bps = (traded * cost_bps).sum(axis=1) / traded_sum
            part_traded = participation.where(traded > 0).to_numpy()
            part_traded = part_traded[np.isfinite(part_traded)]
            diagnostics.update(
                effective_oneway_bps=float(eff_bps.mean()),
                statutory_bps=float(cost_cfg.compose_oneway_bps()),
                participation_median=float(np.median(part_traded)) if part_traded.size else 0.0,
                participation_p99=(
                    float(np.quantile(part_traded, 0.99)) if part_traded.size else 0.0
                ),
            )
        else:
            trade_cost = turnover * (cost_cfg.compose_oneway_bps() / 1e4)
            diagnostics["effective_oneway_bps"] = float(cost_cfg.compose_oneway_bps())

        if cost_cfg.tiered_borrow and panels is not None:
            rates = borrow_bps_panel(short_w, panels, cost_cfg)
            borrow_cost = (short_w * rates / 1e4).sum(axis=1) / ppy
        else:
            borrow_cost = short_gross * (cost_cfg.short_borrow_bps_annual / 1e4) / ppy

        net = gross - trade_cost - borrow_cost
        trade_drag = float(trade_cost.reindex(gross.index).mean() * ppy)
        borrow_drag = float(borrow_cost.reindex(gross.index).mean() * ppy)
    else:
        short_gross = weights.clip(upper=0.0).abs().sum(axis=1)

    portfolio_returns: pd.Series = net.dropna()
    portfolio_returns.name = "portfolio_return"
    portfolio_returns.index.name = _DATE_LEVEL
    live: pd.Index = portfolio_returns.index

    panic_final: pd.Series = panic_bars.reindex(live, fill_value=False)
    calm_bars: int = int((~panic_final.astype(bool)).sum())
    panic_bar_ct: int = int(panic_final.astype(bool).sum())
    label: str = (
        f"Calm: {calm_bars}b | Hedged: {panic_bar_ct}b"
        if base_style(cfg.execution_style) == "dynamic_tilt"
        else f"Active: {calm_bars}b | Cash: {panic_bar_ct}b"
    )
    if cost_cfg.apply_costs:
        eff = diagnostics.get("effective_oneway_bps", cost_cfg.compose_oneway_bps())
        cost_str = f" | costs ON ({eff:.1f} bps/side, drag ~{trade_drag + borrow_drag:.1%}/yr)"
    else:
        cost_str = " | gross"
    if need_shortable:
        held = int((short_mask.reindex(live).abs() > 0).sum(axis=1).mean())
        diagnostics["short_names_held_mean"] = held
        cost_str += f" | SLB short leg ~{held} names"

    diagnostics.update(
        trade_drag_annual=trade_drag,
        borrow_drag_annual=borrow_drag,
        turnover_mean=float(turnover.reindex(live).mean()),
        bars=int(len(portfolio_returns)),
        panic_bars=panic_bar_ct,
    )
    print(
        f"[tier3_execution] style={cfg.execution_style} freq={cfg.frequency} | "
        f"{len(portfolio_returns)} bars | {label} | "
        f"avg turnover/bar={turnover.reindex(live).mean():.2f}{cost_str} | "
        f"Cumulative: {(1 + portfolio_returns).prod() - 1:.4%}"
    )

    return ExecutionResult(
        net=portfolio_returns,
        gross=gross.reindex(live),
        turnover=turnover.reindex(live),
        trade_cost=trade_cost.reindex(live),
        borrow_cost=borrow_cost.reindex(live),
        net_exposure=weights.sum(axis=1).reindex(live),
        short_gross=short_gross.reindex(live),
        diagnostics=diagnostics,
        weights=weights.reindex(live),
    )


def execute_ml_strategy(
    wf_result: WalkForwardResult,
    panic: pd.Series,
    price_wide: pd.DataFrame,
    cfg: StrategyConfig,
    terminal_returns: pd.Series | None = None,
    panels: CostPanels | None = None,
) -> pd.Series:
    """
    The net portfolio return series for one strategy — the ledger column.

    Thin wrapper over ``run_execution`` for callers that want only the series.
    """
    return run_execution(
        wf_result, panic, price_wide, cfg,
        terminal_returns=terminal_returns, panels=panels,
    ).net


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

    # ---------------------------------------------------------------------- #
    # Phase 4c: per-name costs (A3), participation cap (A4), SLB leg (B2),
    # tiered borrow (B3). All of it is gated OFF by default, so the first check
    # is the one that protects every published result: default config must be
    # bit-identical to the old statutory-only path.
    # ---------------------------------------------------------------------- #
    from dataclasses import replace as _replace

    rng4c = np.random.default_rng(11)
    # Liquidity ordered by ticker index, so cost must rise across the cross-section.
    adv_levels = np.geomspace(5e9, 5e6, len(tickers))
    panels = CostPanels(
        adv=pd.DataFrame(
            np.tile(adv_levels, (len(dates), 1)), index=dates, columns=tickers
        ),
        sigma=pd.DataFrame(0.02, index=dates, columns=tickers),
        half_spread_bps=pd.DataFrame(
            np.tile(np.linspace(1.0, 40.0, len(tickers)), (len(dates), 1)),
            index=dates, columns=tickers,
        ),
        shortable=pd.DataFrame(
            # Only the more liquid half is borrowable — the real shape of F&O eligibility.
            np.tile([True] * (len(tickers) // 2) + [False] * (len(tickers) - len(tickers) // 2),
                    (len(dates), 1)),
            index=dates, columns=tickers,
        ),
    )

    cfg_ls = StrategyConfig(
        is_simulation=False, market_type="dryrun", lookback_period=0, hmm_states=2,
        execution_style="long_short", frequency="daily", rebalance_buffer_mult=2.0,
    )
    cfg_slb = _replace(cfg_ls, execution_style="long_short_slb")
    base_cost = config.ML_CONFIG.cost

    def _run(cfg_, **cost_kwargs):
        config.ML_CONFIG.cost = _replace(base_cost, **cost_kwargs)
        try:
            return run_execution(wf, panic, price_wide, cfg_, panels=panels)
        finally:
            config.ML_CONFIG.cost = base_cost

    c4 = {}
    flat = _run(cfg_ls)                                    # all Phase 4c flags off
    per_name = _run(cfg_ls, charge_per_name=True, aum_rupees=1e7)
    big_aum = _run(cfg_ls, charge_per_name=True, aum_rupees=1e9)
    capped = _run(cfg_ls, charge_per_name=True, aum_rupees=1e9,
                  enforce_participation_cap=True)
    slb = _run(cfg_slb, charge_per_name=True, aum_rupees=1e7)
    tiered = _run(cfg_slb, charge_per_name=True, aum_rupees=1e7, tiered_borrow=True)

    # 1. The regression guarantee.
    c4["4c OFF == statutory-only path"] = bool(
        np.allclose(flat.net.to_numpy(), net_ret.reindex(flat.net.index).to_numpy())
        if cfg_ls.execution_style == "dynamic_tilt" else
        np.isclose(flat.diagnostics["effective_oneway_bps"],
                   base_cost.compose_oneway_bps())
    )
    # 2. Spread + impact can only ADD to the statutory rate, never subtract.
    c4["per-name cost >= statutory rate"] = bool(
        per_name.diagnostics["effective_oneway_bps"] > base_cost.compose_oneway_bps()
    )
    c4["per-name costs reduce net return"] = bool(per_name.net.sum() < flat.net.sum())
    # 3. Impact is size-dependent: 100x the money must cost strictly more per share.
    c4["cost rises with AUM (impact is real)"] = bool(
        big_aum.diagnostics["effective_oneway_bps"]
        > per_name.diagnostics["effective_oneway_bps"]
    )
    c4["participation rises with AUM"] = bool(
        big_aum.diagnostics["participation_median"]
        > per_name.diagnostics["participation_median"]
    )
    # 4. E2: gross is captured and is always >= net once costs are on.
    c4["gross captured alongside net"] = bool(
        len(per_name.gross) == len(per_name.net)
        and (per_name.gross >= per_name.net - 1e-12).all()
    )
    # 5. A4: the cap binds, and gross exposure is preserved by redistribution.
    #    Tested directly on cap_leg_weights — the decile book of this 10-name toy
    #    universe is a single name, so an end-to-end test would only exercise the
    #    degenerate "nothing to redistribute to" branch.
    n_leg = 20
    leg_cols = [f"N{i:02d}" for i in range(n_leg)]
    leg_dates = pd.bdate_range("2020-01-01", periods=3)
    leg = pd.DataFrame(1.0 / n_leg, index=leg_dates, columns=leg_cols)
    # Row 0: caps bind on the thin tail but total capacity comfortably exceeds 1.0.
    # Row 1: no cap binds at all (must be an exact no-op).
    # Row 2: total capacity is BELOW 1.0 — the leg genuinely cannot be built.
    caps = pd.DataFrame(index=leg_dates, columns=leg_cols, dtype=float)
    caps.iloc[0] = [0.30] * 5 + [0.01] * 15          # capacity = 1.5 + 0.15
    caps.iloc[1] = 1.0
    caps.iloc[2] = 0.01                              # capacity = 0.20
    capped_leg = cap_leg_weights(leg, caps)

    c4["cap: never exceeds the ceiling"] = bool(
        (capped_leg <= caps + 1e-12).to_numpy().all()
    )
    c4["cap: preserves leg gross 1.0 by redistribution"] = bool(
        np.isclose(capped_leg.iloc[0].sum(), 1.0, atol=1e-9)
    )
    c4["cap: binds (weights actually moved)"] = bool(
        not np.allclose(capped_leg.iloc[0].to_numpy(), leg.iloc[0].to_numpy())
    )
    c4["cap: exact no-op when nothing binds"] = bool(
        np.allclose(capped_leg.iloc[1].to_numpy(), leg.iloc[1].to_numpy())
    )
    c4["cap: reports capacity, does not fake it"] = bool(
        np.isclose(capped_leg.iloc[2].sum(), 0.20, atol=1e-9)
    )
    # End to end, the cap must change the realized return without breaking the book.
    c4["cap changes the realized return"] = bool(
        not np.allclose(capped.net.to_numpy(), big_aum.net.to_numpy())
    )

    # 5b. A gap in a cost panel must be FILLED, never skipped. A NaN would drop out
    #     of the (traded * cost_bps).sum() and charge that trade nothing — silently
    #     reinstating the zero-cost assumption for exactly the thin names most likely
    #     to be missing an estimate.
    holed = CostPanels(
        adv=panels.adv.mask(np.eye(len(dates), len(tickers), dtype=bool)[:len(dates)]),
        sigma=panels.sigma.copy(),
        half_spread_bps=panels.half_spread_bps.copy(),
        shortable=panels.shortable,
    )
    holed.half_spread_bps.iloc[:, 0] = np.nan       # one name never measured at all
    holed.sigma.iloc[5, :] = np.nan                 # one date never measured at all
    traded_probe = pd.DataFrame(0.02, index=dates, columns=tickers)
    bps_probe, _ = per_name_cost_bps(
        traded_probe, holed, _replace(base_cost, charge_per_name=True, aum_rupees=1e7)
    )
    c4["cost panel gaps are filled, never skipped"] = bool(
        bps_probe.notna().to_numpy().all()
        and (bps_probe.to_numpy() >= base_cost.compose_oneway_bps() - 1e-9).all()
    )

    # 6. B2: the SLB short leg holds only borrowable names, and stays dollar-neutral.
    W_slb = build_weight_matrix(
        *apply_rebalance_buffer(
            wf.alpha_scores, cfg_slb,
            short_scores=wf.alpha_scores.where(panels.shortable),
        ),
        panic, cfg_slb,
    )
    calm_idx = W_slb.index[~panic.reindex(W_slb.index).fillna(False).astype(bool)]
    shorted = (W_slb < 0)
    c4["SLB leg shorts only borrowable names"] = bool(
        not (shorted & ~panels.shortable.reindex_like(shorted)).to_numpy().any()
    )
    c4["SLB leg stays dollar-neutral"] = bool(
        np.allclose(W_slb.sum(axis=1).loc[calm_idx], 0.0, atol=1e-9)
    )
    c4["SLB restriction changes the book"] = bool(
        not np.allclose(slb.net.to_numpy(), per_name.net.to_numpy())
    )
    # 7. B3: tiering an illiquid short book must cost more than the flat 50 bps.
    c4["tiered borrow differs from flat"] = bool(
        not np.isclose(tiered.borrow_cost.sum(), slb.borrow_cost.sum())
    )

    checks.update(c4)

    # 8. Phase 5 short-leg exposure scalar: all-ones must be the untouched path,
    #    all-zeros must remove the short leg and leave the long leg alone.
    cfg_ls = StrategyConfig(
        is_simulation=False, market_type="dryrun", lookback_period=0,
        hmm_states=2, execution_style="long_short", frequency="daily",
    )
    W_none = build_weight_matrix(wf.long_mask, wf.short_mask, panic, cfg_ls)
    ones = pd.Series(1.0, index=wf.long_mask.index)
    W_one = build_weight_matrix(wf.long_mask, wf.short_mask, panic, cfg_ls, short_scale=ones)
    W_zero = build_weight_matrix(wf.long_mask, wf.short_mask, panic, cfg_ls, short_scale=0.0 * ones)
    checks["short_scale=1 == no scalar (bit-for-bit)"] = bool(W_one.equals(W_none))
    checks["short_scale=0 removes the short leg"] = bool(
        (W_zero >= 0).to_numpy().all() and W_zero.equals(W_none.clip(lower=0.0))
    )
    res_none = run_execution(wf, panic, price_wide, cfg_ls)
    res_one = run_execution(wf, panic, price_wide, cfg_ls, short_scale=ones)
    checks["run_execution short_scale=1 == None"] = bool(res_one.net.equals(res_none.net))
    checks["ExecutionResult carries the weights"] = bool(
        res_none.weights is not None and res_none.weights.index.equals(res_none.net.index)
    )

    print()
    print("=" * 70)
    for name, ok in checks.items():
        print(f"  {name:<44}: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    assert all(checks.values()), "tier3_execution dry-run FAILED"
