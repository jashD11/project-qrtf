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

    print()
    print("=" * 70)
    for name, ok in checks.items():
        print(f"  {name:<46}: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    assert all(checks.values()), "cost_aware dry-run FAILED"
