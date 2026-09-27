#!/usr/bin/env python3
"""
Phase 5 ceilings and kill-tests — measurements, never candidates (docs/phase5_plan.md §8).

Three read-only measurements on the Phase 4c frozen cells, run on the cached Tier 1 fits
under the Phase 4c cost regime. Nothing here is a strategy, nothing is promoted, and every
number counts 0 in N (§7.5): the first two re-measure cells already searched, the third
uses perfect foresight and so cannot be traded.

1. ZERO-COST CEILING. The Phase 4c gross ledger scored at N=30. A cost-aware rule moves a
   cell's net towards its gross, never past it, so a cell that fails here cannot pass by
   cost reduction alone.

2. SWAP ATTRIBUTION. Every swap the baseline buffer executed (a name in, a name out, same
   leg, same bar), paired best-entrant-with-worst-exit by the ex-ante alpha z-score. For
   each pair: the realised h-day log-return gap against the round-trip cost actually
   charged. Bucketed by the *ex-ante* z gap, which is what M2 (§7.4) can see. It tells us
   whether there are swaps that do not pay for themselves for M2 to skip. It sets no
   parameter — κ was frozen in §7 before this ran.

3. PERFECT-FORESIGHT SHORT LEG (§5 step 2). ``long_short_slb`` with the short leg scaled by
   ``short_scale`` in {always on, always off, daily oracle, monthly oracle}. The daily
   oracle holds the leg on a bar only if its next-bar return is positive — a loose bound.
   The monthly oracle decides once per calendar month — the honest ceiling for a gate that
   moves at regime speed. Costs are charged on the resulting weights. Each variant is
   scored against the Phase 4c benchmark SR* (N=30), not against a benchmark its own
   inflated Sharpe would drag up. Verdict rule: if the monthly oracle fails DSR 0.95 net,
   short-leg gating is closed.

Writes only under ``data/trial_database/phase5/``. Every Tier 1 fit must be a cache HIT.

    python scripts/phase5_ceilings.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import config  # noqa: E402
from run_pipeline_ml import (  # noqa: E402
    compute_regime,
    compute_signals,
    load_terminal_returns,
    make_ml_config,
)
from src.production_ml.tier1_trees import DECILE_PCT  # noqa: E402
from src.production_ml.tier3_execution import (  # noqa: E402
    apply_terminal_returns,
    forward_returns,
    load_cost_panels,
    per_name_cost_bps,
    run_execution,
)
from src.production_ml.tier4_dsr_gate import (  # noqa: E402
    _ANN,
    _to_daily,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    run_dsr_gate,
)

FREQUENCY = config.PHASE4C_FREQUENCY
STYLES = config.PHASE4C_STYLES
TARGETS = config.PHASE4C_TARGETS
N_TRIALS = 30                      # the Phase 4c N — these measurements add nothing to it
OUT = Path("data/trial_database/phase5")
GROSS_LEDGER = config.PHASE4C_LEDGER.replace(".parquet", "_gross.parquet")
GATE_CSV = config.PHASE4C_LEDGER.replace(".parquet", "_dsr_gate.csv")


def horizon(target_col: str) -> int:
    """``tgt_fwd_logret_21b`` -> 21."""
    return int(target_col.rsplit("_", 1)[-1].rstrip("b"))


def xs_zscore(alpha: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per bar over the names scored that bar."""
    return alpha.sub(alpha.mean(axis=1), axis=0).div(alpha.std(axis=1), axis=0)


# --------------------------------------------------------------------------- #
# 2. Swap attribution
# --------------------------------------------------------------------------- #
def swap_pairs(
    weights: pd.DataFrame, alpha: pd.DataFrame, price_wide: pd.DataFrame,
    cost_bps: pd.DataFrame, panic: pd.Series, h: int, leg: str,
) -> pd.DataFrame:
    """
    One row per executed swap on ``leg``: ex-ante z gap, realised h-day gap, round-trip cost.

    Gaps are signed so that positive means the swap was a good idea for that leg (on the
    short leg, the entrant should fall more than the name it replaced). Bars on or right
    after a panic bar are excluded: there the book is liquidated or rebuilt by the gate,
    not rotated by the ranker, and those trades are not the buffer's to skip.
    """
    sign = 1.0 if leg == "long" else -1.0
    held = (weights * sign) > 0
    prev = held.shift(1, fill_value=False)
    enter, leave = (held & ~prev).to_numpy(), (~held & prev).to_numpy()

    Z = xs_zscore(alpha).reindex(index=weights.index, columns=weights.columns).to_numpy()
    fwd_h = np.log(price_wide.shift(-h) / price_wide)
    R = fwd_h.reindex(index=weights.index, columns=weights.columns).to_numpy()
    C = cost_bps.reindex(index=weights.index, columns=weights.columns).to_numpy()

    calm = ~panic.reindex(weights.index).fillna(False).astype(bool)
    rotate = (calm & calm.shift(1, fill_value=False)).to_numpy()
    rotate[0] = False

    rows = []
    for t in np.flatnonzero(rotate):
        ins, outs = np.flatnonzero(enter[t]), np.flatnonzero(leave[t])
        if not len(ins) or not len(outs):
            continue
        ins = ins[np.argsort(-sign * Z[t, ins])]        # best entrant first
        outs = outs[np.argsort(sign * Z[t, outs])]      # worst exit first
        for i, o in zip(ins, outs):
            rows.append((weights.index[t], sign * (Z[t, i] - Z[t, o]),
                         sign * (R[t, i] - R[t, o]), (C[t, i] + C[t, o]) / 1e4))
    return pd.DataFrame(rows, columns=["date", "z_gap", "realised_gap", "roundtrip_cost"])


def summarise_swaps(pairs: pd.DataFrame, label: str) -> pd.DataFrame:
    """Per ex-ante quintile: count, gaps, cost, share that paid, mean net per swap."""
    clean = pairs.dropna()
    clean = clean.assign(
        net=clean["realised_gap"] - clean["roundtrip_cost"],
        paid=clean["realised_gap"] > clean["roundtrip_cost"],
        bucket=pd.qcut(clean["z_gap"], 5, labels=[f"Q{i}" for i in range(1, 6)]),
    )
    agg = dict(n=("net", "size"), z_gap=("z_gap", "mean"),
               realised_gap=("realised_gap", "mean"), roundtrip_cost=("roundtrip_cost", "mean"),
               share_paid=("paid", "mean"), mean_net=("net", "mean"))
    by_q = clean.groupby("bucket", observed=True).agg(**agg)
    total = clean.assign(bucket="all").groupby("bucket").agg(**agg)
    out = pd.concat([by_q, total])
    out.insert(0, "cell", label)
    out.attrs["dropped"] = len(pairs) - len(clean)
    return out


# --------------------------------------------------------------------------- #
# 3. Perfect-foresight short leg
# --------------------------------------------------------------------------- #
def short_leg_return(weights, price_wide, terminal) -> pd.Series:
    """Next-bar return of the short leg alone, priced exactly as run_execution prices it."""
    fwd = forward_returns(price_wide, bridge_halts=True)
    fwd = fwd.reindex(index=weights.index, columns=weights.columns)
    fwd = apply_terminal_returns(fwd, price_wide, terminal)
    return (weights.clip(upper=0.0) * fwd).sum(axis=1, min_count=1)


def oracle_scales(leg_ret: pd.Series, bars: pd.Index) -> dict[str, pd.Series]:
    """The four short_scale series, on every bar the book is built over."""
    ones = pd.Series(1.0, index=bars)
    leg = leg_ret.reindex(bars)
    daily = (leg > 0).astype(float).where(leg.fillna(0.0) != 0.0, 1.0)
    month_sum = leg.groupby(bars.to_period("M")).transform("sum")
    monthly = (month_sum > 0).astype(float)
    return {"always_on": ones, "always_off": 0.0 * ones,
            "oracle_daily": daily, "oracle_monthly": monthly}


def score_vs_frozen_benchmark(net: pd.Series, sr_star_daily: float) -> dict:
    """Lo-corrected Sharpe and DSR against the Phase 4c SR*, without re-deriving SR*."""
    stats = deflated_sharpe_ratio(pd.DataFrame({"x": _to_daily(net)}), n_trials=1).iloc[0]
    dsr = probabilistic_sharpe_ratio(
        stats["SR_daily"], int(stats["T"]), stats["skew"], stats["kurt"], sr_star_daily
    )
    return dict(SR_ann=stats["SR_ann"], DSR=dsr, passed=dsr > config.ML_CONFIG.dsr.dsr_threshold)


# --------------------------------------------------------------------------- #
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    config.ML_CONFIG.cost = replace(config.ML_CONFIG.cost, **config.PHASE4C_COST_OVERRIDES)
    cost_cfg = config.ML_CONFIG.cost

    print("\n" + "=" * 100 + "\n1 · ZERO-COST CEILING — Phase 4c gross ledger, N=30\n" + "=" * 100)
    ceiling = run_dsr_gate(GROSS_LEDGER, n_trials=N_TRIALS, persist=False)
    ceiling.to_csv(OUT / "diagnostics_zero_cost_ceiling.csv")

    panic = compute_regime(config.HEADLINE_HMM_STATES, [FREQUENCY]).panic
    panels = load_cost_panels(need_shortable=True)
    terminal = load_terminal_returns(FREQUENCY)
    signals = {t: compute_signals(FREQUENCY, DECILE_PCT, target_col=t) for t in TARGETS}
    phase4c_net = pd.read_parquet(config.PHASE4C_LEDGER)

    def run(target, style, short_scale=None):
        wf, price_wide = signals[target]
        cfg = make_ml_config(FREQUENCY, style, target_col=target)
        res = run_execution(wf, panic, price_wide, cfg, terminal_returns=terminal,
                            panels=panels, short_scale=short_scale)
        return cfg, res

    # ---- 2 · swap attribution ------------------------------------------------ #
    tables, baseline = [], {}
    for target in TARGETS:
        for style in STYLES:
            cfg, res = run(target, style)
            baseline[(target, style)] = (cfg, res)
            wf, price_wide = signals[target]
            traded = res.weights.diff().abs()
            traded.iloc[0] = res.weights.iloc[0].abs()
            cost_bps, _ = per_name_cost_bps(traded, panels, cost_cfg)
            legs = ["long"] + (["short"] if style != "long_only" else [])
            for leg in legs:
                pairs = swap_pairs(res.weights, wf.alpha_scores, price_wide, cost_bps,
                                   panic, horizon(target), leg)
                tables.append(summarise_swaps(pairs, f"{style}/{horizon(target)}b/{leg}"))
    swaps = pd.concat(tables)
    swaps.to_csv(OUT / "diagnostics_swap_attribution.csv")

    # ---- 3 · perfect-foresight short leg ------------------------------------ #
    gate = pd.read_csv(GATE_CSV)
    sr_star_ann = float(gate["SR_star_ann"].iloc[0])
    sr_star_daily = sr_star_ann / _ANN
    oracle_rows, checks = [], {}
    for target in TARGETS:
        wf, price_wide = signals[target]
        cfg, base = baseline[(target, "long_short_slb")]
        leg_ret = short_leg_return(base.weights, price_wide, terminal)
        for name, scale in oracle_scales(leg_ret, wf.alpha_scores.index).items():
            _, res = run(target, "long_short_slb", short_scale=scale)
            if name == "always_on":
                checks[f"always_on == Phase 4c ledger ({horizon(target)}b)"] = bool(
                    _to_daily(res.net).equals(phase4c_net[cfg.strategy_id].dropna())
                )
            if name == "always_off":
                lo_cfg, lo = baseline[(target, "long_only")]
                checks[f"always_off == long_only cell ({horizon(target)}b)"] = bool(
                    np.allclose(res.net.to_numpy(), lo.net.to_numpy(), rtol=0, atol=1e-15)
                )
            d = res.diagnostics
            oracle_rows.append(dict(
                target=f"{horizon(target)}b", variant=name,
                short_on_share=float(scale.reindex(res.net.index).mean()),
                turnover=d["turnover_mean"], trade_drag=d["trade_drag_annual"],
                borrow_drag=d["borrow_drag_annual"],
                gross_SR_ann=score_vs_frozen_benchmark(res.gross, sr_star_daily)["SR_ann"],
                **score_vs_frozen_benchmark(res.net, sr_star_daily),
            ))
    oracle = pd.DataFrame(oracle_rows)
    oracle.to_csv(OUT / "diagnostics_oracle_short.csv", index=False)

    # ---- report --------------------------------------------------------------- #
    pd.set_option("display.width", 160)
    print("\n" + "=" * 100 + "\n2 · SWAP ATTRIBUTION — per executed swap, realised h-day gap vs round-trip cost\n" + "=" * 100)
    print(swaps.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n" + "=" * 100 + f"\n3 · PERFECT-FORESIGHT SHORT LEG — long_short_slb, "
          f"DSR vs Phase 4c SR* = {sr_star_ann:.3f} (N={N_TRIALS})\n" + "=" * 100)
    print(oracle.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  consistency checks:")
    for k, v in checks.items():
        print(f"    {k:<44} {'PASS' if v else 'FAIL'}")
    monthly_pass = bool(oracle.loc[oracle["variant"] == "oracle_monthly", "passed"].any())
    print(f"\n  VERDICT (§8 rule): monthly oracle {'PASSES' if monthly_pass else 'FAILS'} DSR 0.95 net "
          f"-> short-leg gating {'has a ceiling worth building for' if monthly_pass else 'is CLOSED'}")
    print("=" * 100)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
