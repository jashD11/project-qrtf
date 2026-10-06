"""
Phase 5 — cost-aware portfolio construction (docs/phase5_plan.md §7).

Phase 4c found the trees predict (IC ~0.04) but the book pays 16-47%/yr to trade on it,
and §8.2 found the average buffer swap earns about half its own round-trip cost. The
baseline buffer (``tier3_execution.apply_rebalance_buffer``) decides membership on rank
alone and charges the spread afterwards. The constructions here put the cost *into* the
decision:

- **M1 ``cost_band``** — each name's eviction band scales with its own trading cost,
  ``(c_i / median c)^(1/3)``. Rank-only.
- **M2 ``cost_swap``** — an incumbent outside the exit band is replaced only when the
  entrant's expected h-day return beats it by more than the round-trip cost (Grinold:
  ``E = IC · σ · √h · z``).

Every constant is frozen in ``config.PHASE5_*`` from theory, before any result existed.

Causality. Costs read row t of panels whose windows end at t-1. The IC is estimated from
realised returns only, with an h+1-bar embargo: the h-day return from bar s is known at
the close of s+h, and is first used on bar s+h+1. ``scripts/no_lookahead_check.py``
scrambles every input after a cut and asserts nothing before it moves.

These are pure functions of their inputs; ``run_execution`` dispatches to them on
``StrategyConfig.construction`` and the baseline path never touches this module.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config  # noqa: E402

# Below every economic magnitude (a 1e-12 return gap) and far above the ≤5.6e-17
# cold-fit jitter in raw alpha, so a refit cannot flip a threshold decision.
ROUND_DECIMALS: int = 12


def target_horizon(target_col: str) -> int:
    """``tgt_fwd_logret_21b`` -> 21: the forecast horizon the trees were trained on."""
    return int(target_col.rsplit("_", 1)[-1].rstrip("b"))


# --------------------------------------------------------------------------- #
# §7.2 — decision cost
# --------------------------------------------------------------------------- #
def decision_cost_bps(
    panels, index: pd.Index, columns: pd.Index, k_per_bar: pd.Series, cost_cfg,
) -> pd.DataFrame:
    """
    One-way cost (bps) of trading a full ``1/k`` position in each name on each bar.

    This is ``tier3_execution.per_name_cost_bps`` evaluated at the trade the decision is
    about — a whole position, ``traded = 1/k_t`` — so the rule and the ledger price a trade
    with one law (statutory + measured half-spread + square-root impact), not two. Gaps
    take the date's cross-sectional median inside ``align_cost_panel``; never NaN.
    """
    from src.production_ml.tier3_execution import per_name_cost_bps

    k = k_per_bar.reindex(index).astype(float)
    traded = pd.DataFrame(
        np.repeat((1.0 / k.to_numpy())[:, None], len(columns), axis=1),
        index=index, columns=columns,
    )
    cost_bps, _ = per_name_cost_bps(traded, panels, cost_cfg)
    return cost_bps


# --------------------------------------------------------------------------- #
# §7.4 — causal IC and expected return
# --------------------------------------------------------------------------- #
def _rowwise_spearman(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Cross-sectional Spearman correlation per row, over names valid in both frames."""
    both = a.notna() & b.notna()
    ra = a.where(both).rank(axis=1)
    rb = b.where(both).rank(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    ic = num / den.replace(0.0, np.nan)
    return ic.where(both.sum(axis=1) >= 3)


def trailing_ic(
    alpha: pd.DataFrame,
    price_wide: pd.DataFrame,
    horizon: int,
    window: int = config.PHASE5_IC_WINDOW,
    min_obs: int = config.PHASE5_IC_MIN_OBS,
) -> pd.Series:
    """
    The IC the rule may use on each bar: trailing mean of realised daily rank-ICs.

    ``IC_s`` = Spearman(alpha on bar s, h-day log return from s), which is only known at
    the close of bar s+h. On bar t the estimate averages the last ``window`` valid IC_s
    with ``pos(s) + h <= pos(t) - 1`` — an h+1-bar embargo, counted in *price-panel*
    bars so a gap in the scored index cannot shorten it. NaN until ``min_obs`` such
    observations exist. Not clamped: the caller decides what a non-positive IC means.
    """
    fwd_h = np.log(price_wide.shift(-horizon) / price_wide)
    fwd_h = fwd_h.reindex(index=alpha.index, columns=alpha.columns)
    ic = _rowwise_spearman(alpha, fwd_h)

    pos = price_wide.index.get_indexer(alpha.index)
    if (pos < 0).any():
        raise ValueError("every scored bar must exist in price_wide")
    valid = ic.notna().to_numpy()
    ic_pos = pos[valid]                      # price-panel position of each valid IC_s
    ic_val = ic.to_numpy()[valid]
    csum = np.concatenate([[0.0], np.cumsum(ic_val)])

    # Number of valid IC_s usable on bar t: those with pos(s) <= pos(t) - h - 1.
    n_known = np.searchsorted(ic_pos, pos - horizon - 1, side="right")
    lo = np.maximum(n_known - window, 0)
    count = n_known - lo
    mean = (csum[n_known] - csum[lo]) / np.where(count > 0, count, 1)
    out = pd.Series(np.where(count >= min_obs, mean, np.nan), index=alpha.index)
    out.name = "trailing_ic"
    return out


def expected_return(
    alpha: pd.DataFrame, sigma: pd.DataFrame, ic: pd.Series, horizon: int,
) -> pd.DataFrame:
    """
    Grinold expected h-day return, ``E = IC · σ · √h · z``, rounded at 1e-12.

    ``z`` is the cross-sectional z-score of alpha on each bar, ``σ`` the trailing daily
    return stdev (the cost panel's, window ending t-1). IC <= 0 is clamped to 0 — no
    demonstrated skill means no expected gain, so no discretionary swap can pay. Rows
    where the IC is not yet estimable are NaN (the caller falls back to the baseline).
    """
    z = alpha.sub(alpha.mean(axis=1), axis=0).div(alpha.std(axis=1), axis=0)
    ic_c = ic.reindex(alpha.index).clip(lower=0.0)
    sig = sigma.reindex(index=alpha.index, columns=alpha.columns)
    e = sig.mul(z).mul(ic_c * np.sqrt(horizon), axis=0)
    return e.round(ROUND_DECIMALS)


# --------------------------------------------------------------------------- #
# Shared buffer scaffolding
# --------------------------------------------------------------------------- #
def _buffer_setup(alpha_scores, cfg, short_scores):
    """Ranks, per-bar widths and output arrays, exactly as ``apply_rebalance_buffer``."""
    from src.production_ml.tier3_execution import base_style

    do_short = base_style(cfg.execution_style) in ("long_short", "dynamic_tilt")
    short_src = alpha_scores if short_scores is None else short_scores
    rank_desc = alpha_scores.rank(axis=1, ascending=False, method="first").to_numpy()
    rank_asc = short_src.rank(axis=1, ascending=True, method="first").to_numpy()
    valid_counts = alpha_scores.notna().sum(axis=1).to_numpy()
    shape = alpha_scores.shape
    return (do_short, rank_desc, rank_asc, valid_counts,
            np.zeros(shape, dtype=np.int8), np.zeros(shape, dtype=np.int8))


def _widths(n: int, decile_pct: float, mult: float) -> tuple[int, int]:
    """``k_enter`` and the baseline ``k_exit`` — the same arithmetic as the baseline."""
    k_enter = max(1, int(np.floor(n * decile_pct)))
    k_exit = max(k_enter, min(int(np.floor(n * decile_pct * mult)), n // 2))
    return k_enter, k_exit


def _fill(rank_row: np.ndarray, retained: list[int], k_enter: int) -> list[int]:
    """Best-ranked enter-band names not already held, in rank order."""
    enter_cols = np.where(rank_row <= k_enter)[0]
    enter_cols = enter_cols[np.argsort(rank_row[enter_cols])]
    keep = set(retained)
    return [int(c) for c in enter_cols if int(c) not in keep]


def _to_masks(alpha_scores, held_long, held_short):
    idx, cols = alpha_scores.index, alpha_scores.columns
    return (pd.DataFrame(held_long, index=idx, columns=cols),
            pd.DataFrame(-held_short, index=idx, columns=cols))


# --------------------------------------------------------------------------- #
# §7.3 — M1: cost-scaled buffer
# --------------------------------------------------------------------------- #
def apply_cost_band_buffer(
    alpha_scores: pd.DataFrame,
    cfg,
    cost_bps: pd.DataFrame,
    short_scores: pd.DataFrame | None = None,
    exponent: float = config.PHASE5_BAND_EXPONENT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The baseline two-band buffer with a per-name exit band scaled by trading cost.

        k_exit_i,t = max(k_enter, min(floor(n·d·mult·(c_i,t / median_t c)^(1/3)), n//2))

    ``median_t c`` is taken over the names scored on bar t. A median-cost name keeps the
    baseline band exactly; an expensive name is held through a wider rank drift, a cheap
    one is released sooner. Entry, fill order, book width and the short leg's
    eligibility handling are the baseline's, line for line — only the retention test
    reads ``k_exit[c]`` instead of one ``k_exit``. Ranks only, so the cold-fit jitter in
    raw alpha cannot move it. With every cost equal it *is* ``apply_rebalance_buffer``.
    """
    do_short, rank_desc, rank_asc, valid_counts, held_long, held_short = _buffer_setup(
        alpha_scores, cfg, short_scores
    )
    cost = cost_bps.reindex(index=alpha_scores.index, columns=alpha_scores.columns)
    cost = cost.where(alpha_scores.notna()).to_numpy()
    ratio = cost / np.nanmedian(cost, axis=1, keepdims=True) if cost.size else cost
    scale = ratio ** exponent
    d, mult = cfg.decile_pct, cfg.rebalance_buffer_mult

    def _leg(rank_row, prev, k_enter, k_exit_row):
        retained = [c for c in prev if rank_row[c] <= k_exit_row[c]]
        retained.sort(key=lambda c: rank_row[c])
        if len(retained) >= k_enter:
            return retained[:k_enter]
        return retained + _fill(rank_row, retained, k_enter)[: k_enter - len(retained)]

    prev_long: list[int] = []
    prev_short: list[int] = []
    for t in range(len(alpha_scores.index)):
        n = int(valid_counts[t])
        if n == 0:
            prev_long, prev_short = [], []
            continue
        k_enter, _ = _widths(n, d, mult)
        raw = np.floor(n * d * mult * scale[t])
        k_exit_row = np.maximum(k_enter, np.minimum(raw, n // 2))   # NaN for unscored names

        prev_long = _leg(rank_desc[t], prev_long, k_enter, k_exit_row)
        held_long[t, prev_long] = 1
        if do_short:
            prev_short = _leg(rank_asc[t], prev_short, k_enter, k_exit_row)
            held_short[t, prev_short] = 1

    return _to_masks(alpha_scores, held_long, held_short)


# --------------------------------------------------------------------------- #
# §7.4 — M2: swap only if it pays
# --------------------------------------------------------------------------- #
def _swap_leg(
    rank_row: np.ndarray, prev: list[int], k_enter: int, k_exit: int,
    e_row: np.ndarray, c_row: np.ndarray, ic_ok: bool, kappa: float,
) -> list[int]:
    """
    One leg, one bar. ``e_row`` is already signed for the leg (short: -E).

    Incumbents inside the exit band are retained exactly as in the baseline. An incumbent
    outside it but still rankable (in the universe and, on the short leg, borrowable) is
    an eviction *candidate*: slots it could hold are contested, the rest are empty. Empty
    slots are filled with the best-ranked entrants unconditionally. The next entrants,
    best first, are paired with the contested candidates, worst first, and a pair swaps
    only if ``round(E_in - E_out - kappa·(c_in + c_out)/1e4, 12) > 0``.
    """
    retained = [c for c in prev if rank_row[c] <= k_exit]
    retained.sort(key=lambda c: rank_row[c])
    if len(retained) >= k_enter:
        return retained[:k_enter]
    fill = _fill(rank_row, retained, k_enter)
    slots = k_enter - len(retained)
    if not ic_ok:
        return retained + fill[:slots]

    keep = set(retained)
    candidates = [c for c in prev if c not in keep and np.isfinite(rank_row[c])]
    candidates.sort(key=lambda c: -e_row[c])                 # best expected return first
    contested = candidates[:slots]                           # at most one per open slot
    empties = slots - len(contested)
    book = retained + fill[:empties]
    challengers = fill[empties:]

    for j, cand in enumerate(sorted(contested, key=lambda c: e_row[c])):   # worst first
        if j < len(challengers):
            ent = challengers[j]
            gain = e_row[ent] - e_row[cand] - kappa * (c_row[ent] + c_row[cand]) / 1e4
            # NaN = no estimate for this pair -> baseline behaviour (swap). Only NaN:
            # kappa=inf gives gain=-inf, which is a decision (hold), not a gap.
            if np.isnan(gain) or round(float(gain), ROUND_DECIMALS) > 0:
                book.append(ent)
                continue
        book.append(cand)
    return book


def apply_cost_swap_buffer(
    alpha_scores: pd.DataFrame,
    cfg,
    cost_bps: pd.DataFrame,
    exp_ret: pd.DataFrame,
    short_scores: pd.DataFrame | None = None,
    kappa: float = config.PHASE5_SWAP_KAPPA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The baseline 2.0 buffer, except a drifted incumbent is replaced only when it pays.

    ``exp_ret`` is ``expected_return(...)``: NaN on bars where the IC is not yet
    estimable, where this reproduces ``apply_rebalance_buffer`` exactly. With
    ``kappa = 0`` and cross-sectionally constant σ it also reproduces the baseline (every
    entrant out-ranks every candidate); with ``kappa = inf`` only forced exits and empty
    slots trade. Book width stays ``k_enter``; the short leg ranks inside the borrowable
    set and uses ``-E``.
    """
    do_short, rank_desc, rank_asc, valid_counts, held_long, held_short = _buffer_setup(
        alpha_scores, cfg, short_scores
    )
    e = exp_ret.reindex(index=alpha_scores.index, columns=alpha_scores.columns).to_numpy()
    c = cost_bps.reindex(index=alpha_scores.index, columns=alpha_scores.columns).to_numpy()
    ic_ok = np.isfinite(e).any(axis=1)
    d, mult = cfg.decile_pct, cfg.rebalance_buffer_mult

    prev_long: list[int] = []
    prev_short: list[int] = []
    for t in range(len(alpha_scores.index)):
        n = int(valid_counts[t])
        if n == 0:
            prev_long, prev_short = [], []
            continue
        k_enter, k_exit = _widths(n, d, mult)
        prev_long = _swap_leg(rank_desc[t], prev_long, k_enter, k_exit,
                              e[t], c[t], bool(ic_ok[t]), kappa)
        held_long[t, prev_long] = 1
        if do_short:
            prev_short = _swap_leg(rank_asc[t], prev_short, k_enter, k_exit,
                                   -e[t], c[t], bool(ic_ok[t]), kappa)
            held_short[t, prev_short] = 1

    return _to_masks(alpha_scores, held_long, held_short)


# --------------------------------------------------------------------------- #
# Dry-run verification
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from dataclasses import replace

    from src.production_ml.tier3_execution import CostPanels, per_name_cost_bps

    print("=" * 70)
    print("cost_aware dry-run — decision cost, causal IC, expected return")
    print("=" * 70)
    rng = np.random.default_rng(3)
    checks: dict[str, bool] = {}

    # ---- synthetic world: 400 names x 900 bars, planted IC -------------- #
    n_names, n_bars, h = 400, 900, 1
    dates = pd.bdate_range("2018-01-01", periods=n_bars)
    names = pd.Index([f"N{i:03d}" for i in range(n_names)])
    signal = rng.normal(size=(n_bars, n_names))
    rho = 0.10
    # r_{t+1} = rho * signal_t + sqrt(1 - rho^2) * noise, so corr(signal_t, r_{t+1}) = rho.
    rets = np.vstack([np.zeros((1, n_names)),
                      rho * signal[:-1] + np.sqrt(1 - rho ** 2) * rng.normal(size=(n_bars - 1, n_names))])
    price = pd.DataFrame(100 * np.exp(np.cumsum(0.01 * rets, axis=0)), index=dates, columns=names)
    alpha = pd.DataFrame(signal, index=dates, columns=names)

    # 1. Planted IC recovered. Spearman of bivariate normals = (6/pi) asin(rho/2).
    ic = trailing_ic(alpha, price, horizon=h, window=252, min_obs=126)
    expected_sp = 6 / np.pi * np.arcsin(rho / 2)
    checks["trailing IC recovers planted IC (±0.01)"] = bool(
        abs(ic.dropna().iloc[-1] - expected_sp) < 0.01
    )
    # 2. Warm-up: NaN until min_obs usable observations, i.e. before bar min_obs + h + 1.
    first = int(np.flatnonzero(ic.notna().to_numpy())[0])
    checks["IC NaN until min_obs usable obs"] = bool(first == 126 + h)

    # 3. Causality: scramble prices FROM the cut bar on (inclusive). The IC used on the
    #    cut bar reads prices up to cut-1 only, so nothing on bars <= cut may move.
    for hh in (1, 5, 21):
        cut = dates[500]
        price_p = price.copy()
        after = price_p.index >= cut
        price_p.loc[after] = price_p.loc[after] * rng.uniform(0.5, 1.5, price_p.loc[after].shape)
        a = trailing_ic(alpha, price, horizon=hh, window=252, min_obs=126)
        b = trailing_ic(alpha, price_p, horizon=hh, window=252, min_obs=126)
        checks[f"IC causal under future scramble (h={hh})"] = bool(
            a.loc[:cut].equals(b.loc[:cut]) and not a.loc[cut:].equals(b.loc[cut:])
        )
    # 4. The embargo is exactly h+1: leaking one bar earlier must be detectable.
    leaky_price = price.shift(-1)          # tomorrow's price visible today
    leak = trailing_ic(alpha, leaky_price.ffill(), horizon=h, window=252, min_obs=126)
    checks["a one-bar price leak changes the IC"] = bool(not leak.dropna().equals(ic.dropna()))

    # ---- expected return ------------------------------------------------ #
    sigma = pd.DataFrame(rng.uniform(0.01, 0.04, (n_bars, n_names)), index=dates, columns=names)
    e = expected_return(alpha, sigma, ic, horizon=21)
    z = alpha.sub(alpha.mean(axis=1), axis=0).div(alpha.std(axis=1), axis=0)
    t = 700
    manual = (ic.iloc[t] * sigma.iloc[t] * np.sqrt(21) * z.iloc[t]).round(ROUND_DECIMALS)
    checks["E = IC·σ·√h·z"] = bool(np.allclose(e.iloc[t], manual, rtol=0, atol=1e-12))
    checks["E NaN while IC not estimable"] = bool(e.iloc[: first].isna().all().all())
    e_neg = expected_return(alpha, sigma, -ic.abs(), horizon=21)
    checks["IC <= 0 clamps E to 0"] = bool((e_neg.iloc[first:].abs() == 0).all().all())

    # ---- decision cost -------------------------------------------------- #
    adv = pd.DataFrame(rng.uniform(5e6, 5e8, (n_bars, n_names)), index=dates, columns=names)
    spread = pd.DataFrame(rng.uniform(2, 40, (n_bars, n_names)), index=dates, columns=names)
    spread.iloc[:, 0] = np.nan                                    # a never-measured name
    panels = CostPanels(adv=adv, sigma=sigma, half_spread_bps=spread, shortable=None)
    cost_cfg = replace(config.ML_CONFIG.cost, charge_per_name=True, aum_rupees=1e7)
    k = pd.Series(40, index=dates)
    c = decision_cost_bps(panels, dates, names, k, cost_cfg)
    ref, _ = per_name_cost_bps(pd.DataFrame(1 / 40, index=dates, columns=names), panels, cost_cfg)
    checks["decision cost has no NaN"] = bool(c.notna().all().all())
    checks["decision cost >= statutory"] = bool((c >= cost_cfg.compose_oneway_bps()).all().all())
    checks["decision cost == per_name_cost_bps at 1/k"] = bool(c.equals(ref))

    # ---- M1: cost-scaled buffer ----------------------------------------- #
    from config import StrategyConfig
    from src.production_ml.tier1_trees import WalkForwardResult
    from src.production_ml.tier3_execution import apply_rebalance_buffer, run_execution

    def _cfg(style, construction="buffer"):
        return StrategyConfig(
            is_simulation=False, market_type="dryrun", lookback_period=0, hmm_states=2,
            execution_style=style, frequency="daily", construction=construction,
        )

    # A persistent signal (AR(1), phi=0.9) so the buffer actually has incumbents to keep;
    # one name leaves the universe mid-sample, and a third of names are unborrowable.
    nb, nn = 300, 60
    bdates = pd.bdate_range("2019-01-01", periods=nb)
    bnames = pd.Index([f"B{i:02d}" for i in range(nn)])
    sig = np.zeros((nb, nn))
    sig[0] = rng.normal(size=nn)
    for t_ in range(1, nb):
        sig[t_] = 0.9 * sig[t_ - 1] + np.sqrt(1 - 0.81) * rng.normal(size=nn)
    balpha = pd.DataFrame(sig, index=bdates, columns=bnames)
    balpha.iloc[150:, 7] = np.nan
    shortable = pd.DataFrame(rng.random((nb, nn)) > 0.33, index=bdates, columns=bnames)
    sscores = balpha.where(shortable)

    flat_cost = pd.DataFrame(20.0, index=bdates, columns=bnames)
    for style, ss in (("long_short", None), ("long_short_slb", sscores)):
        base_l, base_s = apply_rebalance_buffer(balpha, _cfg(style), short_scores=ss)
        m1_l, m1_s = apply_cost_band_buffer(balpha, _cfg(style), flat_cost, short_scores=ss)
        checks[f"M1 equal costs == baseline buffer ({style})"] = bool(
            m1_l.equals(base_l) and m1_s.equals(base_s)
        )

    het_cost = pd.DataFrame(rng.lognormal(3.0, 0.8, (nb, nn)), index=bdates, columns=bnames)
    m1_l, m1_s = apply_cost_band_buffer(balpha, _cfg("long_short_slb"), het_cost, short_scores=sscores)
    n_valid_b = balpha.notna().sum(axis=1)
    k_enter_b = np.floor(n_valid_b * 0.10).clip(lower=1)
    checks["M1 long leg sized k_enter"] = bool((m1_l.sum(axis=1) == k_enter_b).all())
    checks["M1 short leg only borrowable names"] = bool(
        not ((m1_s < 0) & ~shortable).to_numpy().any()
    )
    checks["M1 heterogeneous costs change the book"] = bool(not m1_l.equals(base_l))

    # Hand-built case: n=20, d=0.10 -> k_enter=2, baseline k_exit=4. A and B held; both
    # drift to ranks 5 and 6. Baseline evicts both. A costs 8x the median -> band x2 ->
    # k_exit 8, kept. B costs 1/8 the median -> band x0.5 -> k_exit clipped to 2, evicted.
    hn = pd.Index([f"H{i:02d}" for i in range(20)])
    hd = pd.bdate_range("2020-01-01", periods=2)
    ha = pd.DataFrame([np.arange(20, 0, -1.0), np.arange(20, 0, -1.0)], index=hd, columns=hn)
    ha.iloc[1, [0, 1, 4, 5]] = ha.iloc[1, [4, 5, 0, 1]].to_numpy()   # A,B fall to ranks 5,6
    hc = pd.DataFrame(10.0, index=hd, columns=hn)
    hc.iloc[:, 0], hc.iloc[:, 1] = 80.0, 1.25
    hb_l, _ = apply_rebalance_buffer(ha, _cfg("long_only"))
    h1_l, _ = apply_cost_band_buffer(ha, _cfg("long_only"), hc)
    held_base = set(hb_l.columns[hb_l.iloc[1] == 1])
    held_m1 = set(h1_l.columns[h1_l.iloc[1] == 1])
    checks["baseline evicts both drifted names"] = held_base == {"H04", "H05"}
    checks["M1 keeps the costly one, drops the cheap one"] = held_m1 == {"H00", "H04"}

    # End to end through run_execution under per-name costs.
    bprice = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (nb, nn)), axis=0)),
                          index=bdates, columns=bnames)
    bprice.iloc[151:, 7] = np.nan
    bpanels = CostPanels(
        adv=pd.DataFrame(rng.uniform(5e6, 5e8, (nb, nn)), index=bdates, columns=bnames),
        sigma=pd.DataFrame(rng.uniform(0.01, 0.04, (nb, nn)), index=bdates, columns=bnames),
        half_spread_bps=pd.DataFrame(rng.uniform(2, 40, (nb, nn)), index=bdates, columns=bnames),
        shortable=shortable,
    )
    bwf = WalkForwardResult(alpha_scores=balpha, long_mask=balpha * 0, short_mask=balpha * 0)
    bpanic = pd.Series(False, index=bdates)
    saved = config.ML_CONFIG.cost
    config.ML_CONFIG.cost = replace(saved, charge_per_name=True, aum_rupees=1e7)
    try:
        term = pd.Series({"B07": -0.30})
        r_base = run_execution(bwf, bpanic, bprice, _cfg("long_short_slb"),
                               terminal_returns=term, panels=bpanels)
        r_m1 = run_execution(bwf, bpanic, bprice, _cfg("long_short_slb", "cost_band"),
                             terminal_returns=term, panels=bpanels)
    finally:
        config.ML_CONFIG.cost = saved
    checks["M1 end-to-end: net NaN-free"] = bool(r_m1.net.notna().all())
    checks["M1 end-to-end: dollar-neutral"] = bool(
        np.allclose(r_m1.net_exposure.to_numpy(), 0.0, atol=1e-9)
    )
    checks["M1 end-to-end: book differs from baseline"] = bool(not r_m1.weights.equals(r_base.weights))
    # ---- M2: swap only if it pays --------------------------------------- #
    const_sigma = pd.DataFrame(0.02, index=bdates, columns=bnames)
    ic_pos = pd.Series(0.05, index=bdates)
    e_const = expected_return(balpha, const_sigma, ic_pos, horizon=5)
    for style, ss in (("long_short", None), ("long_short_slb", sscores)):
        base_l, base_s = apply_rebalance_buffer(balpha, _cfg(style), short_scores=ss)
        k0_l, k0_s = apply_cost_swap_buffer(balpha, _cfg(style), het_cost, e_const,
                                            short_scores=ss, kappa=0.0)
        checks[f"M2 kappa=0, const sigma == baseline ({style})"] = bool(
            k0_l.equals(base_l) and k0_s.equals(base_s)
        )
    e_nan = expected_return(balpha, const_sigma, pd.Series(np.nan, index=bdates), horizon=5)
    nan_l, nan_s = apply_cost_swap_buffer(balpha, _cfg("long_short_slb"), het_cost, e_nan,
                                          short_scores=sscores)
    base_l, base_s = apply_rebalance_buffer(balpha, _cfg("long_short_slb"), short_scores=sscores)
    checks["M2 no IC estimate == baseline"] = bool(nan_l.equals(base_l) and nan_s.equals(base_s))

    het_sigma = pd.DataFrame(rng.uniform(0.01, 0.04, (nb, nn)), index=bdates, columns=bnames)
    e_het = expected_return(balpha, het_sigma, ic_pos, horizon=5)
    inf_l, inf_s = apply_cost_swap_buffer(balpha, _cfg("long_short_slb"), het_cost, e_het,
                                          short_scores=sscores, kappa=np.inf)
    # kappa=inf: on bars where k is unchanged, every incumbent still rankable is kept.
    k_same = (k_enter_b == k_enter_b.shift(1)).to_numpy()
    kept_all = True
    for leg_mask, src in ((inf_l, balpha), (-inf_s, sscores)):
        held = leg_mask.to_numpy() == 1
        rankable = src.notna().to_numpy()
        for t_ in range(1, nb):
            if k_same[t_]:
                must = held[t_ - 1] & rankable[t_]
                kept_all &= bool((held[t_][must]).all())
    checks["M2 kappa=inf keeps every rankable incumbent"] = kept_all
    checks["M2 kappa=inf trades less than baseline"] = bool(
        inf_l.diff().abs().sum().sum() < base_l.diff().abs().sum().sum()
    )
    m2_l, m2_s = apply_cost_swap_buffer(balpha, _cfg("long_short_slb"), het_cost, e_het,
                                        short_scores=sscores)
    checks["M2 long leg sized k_enter"] = bool((m2_l.sum(axis=1) == k_enter_b).all())
    checks["M2 short leg only borrowable names"] = bool(
        not ((m2_s < 0) & ~shortable).to_numpy().any()
    )
    checks["M2 kappa=1 changes the book"] = bool(not m2_l.equals(base_l))

    # Hand-built pair: k_enter=2, k_exit=4. A (H00) and H04 swap values, so A drifts to
    # rank 5 and H04 jumps to rank 1 — the one entrant, contesting A's slot (H01 stays).
    # Gap in E is 10 bps: a 4-bps round trip swaps, a 16-bps one holds A.
    pa = pd.DataFrame([np.arange(20, 0, -1.0), np.arange(20, 0, -1.0)], index=hd, columns=hn)
    pa.iloc[1, [0, 4]] = pa.iloc[1, [4, 0]].to_numpy()            # A (H00) -> rank 5
    pe = pd.DataFrame(0.0, index=hd, columns=hn)
    pe.iloc[1, 0], pe.iloc[1, 4] = 0.0010, 0.0020                  # E_A 10 bps, E_in 20 bps
    for cost_side, expect in ((2.0, "H04"), (8.0, "H00")):
        pc = pd.DataFrame(cost_side, index=hd, columns=hn)
        p_l, _ = apply_cost_swap_buffer(pa, _cfg("long_only"), pc, pe)
        held1 = set(p_l.columns[p_l.iloc[1] == 1])
        checks[f"M2 pair rule: round trip {2 * cost_side:.0f} bps -> holds {expect}"] = (
            held1 == {"H01", expect}
        )

    config.ML_CONFIG.cost = replace(saved, charge_per_name=True, aum_rupees=1e7)
    try:
        r_m2 = run_execution(bwf, bpanic, bprice, _cfg("long_short_slb", "cost_swap"),
                             terminal_returns=term, panels=bpanels)
    finally:
        config.ML_CONFIG.cost = saved
    checks["M2 end-to-end: net NaN-free"] = bool(r_m2.net.notna().all())
    checks["M2 end-to-end: dollar-neutral"] = bool(
        np.allclose(r_m2.net_exposure.to_numpy(), 0.0, atol=1e-9)
    )

    try:
        run_execution(bwf, bpanic, bprice, _cfg("long_only", "cost_band"), panels=bpanels)
        checks["M1 refuses to run without per-name costs"] = False
    except ValueError:
        checks["M1 refuses to run without per-name costs"] = True

    print()
    print("=" * 70)
    for name, ok in checks.items():
        print(f"  {name:<46}: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    assert all(checks.values()), "cost_aware dry-run FAILED"
