#!/usr/bin/env python3
"""
Future-perturbation test — nothing the engine decides on bar t may depend on data after t.

Runs Tier 3 (``run_execution``) on the real Phase 4c inputs, then again with every input
after a cut date D replaced by noise: alpha scores, decile masks, prices, the panic gate,
and all four cost panels (ADV, sigma, half-spread, shortable). Everything decided on or
before D must come back bit-identical:

    turnover, net exposure, short gross, trade cost, borrow cost   on every bar <= D
    gross return                                                    on every bar <  D

Gross on bar D is excluded by construction, not by leniency: it realises the forward return
D -> D+1 (``price_wide.shift(-1)``), so it is *supposed* to read the price at D+1.

A test that cannot fail proves nothing, so a CANARY runs the same comparison with the alpha
scores shifted one bar early (``alpha.shift(-1)`` — tomorrow's prediction traded today). The
canary must be caught; if it is not, the test itself is broken and the script fails.

Reads the cached Tier 1 walk-forward (no tree fitting) and writes nothing. Guards every
Phase 5 change to Tier 3 — the cost-aware swap rule first among them.

    python scripts/no_lookahead_check.py
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
from src.production_ml.tier1_trees import DECILE_PCT, WalkForwardResult  # noqa: E402
from src.production_ml.tier3_execution import (  # noqa: E402
    CostPanels,
    ExecutionResult,
    load_cost_panels,
    run_execution,
)

FREQUENCY = "daily_nse500"
TARGET = "tgt_fwd_logret_21b"
STYLES = ["long_only", "long_short_slb"]      # the long-only control + the SLB short path
# Every portfolio construction the engine can run (docs/phase5_plan.md §7). The Phase 5
# ones read cost panels, sigma and (M2) a trailing IC from prices — all scrambled below.
CONSTRUCTIONS = ["buffer", "cost_band"]
CUT_FRACTIONS = [0.25, 0.50, 0.75]           # cut dates as a share of the scored bars
SEED = 7

# Series decided on bar t from information up to t. Gross is handled separately (see top).
DECIDED = ("turnover", "net_exposure", "short_gross", "trade_cost", "borrow_cost")


def _scramble_after(frame: pd.DataFrame, cut: pd.Timestamp, rng: np.random.Generator) -> pd.DataFrame:
    """Replace every value strictly after ``cut`` with noise of the same dtype and NaN pattern."""
    out = frame.copy()
    if pd.api.types.is_integer_dtype(out.dtypes.iloc[0]):
        out = out.astype(float)          # 0/±1 masks — the noise is fractional
    after = out.index > cut
    block = out.loc[after]
    if block.empty:
        return out
    if pd.api.types.is_bool_dtype(block.dtypes.iloc[0]):
        noise = rng.random(block.shape) < 0.5
        out.loc[after] = noise
        return out
    values = block.to_numpy(float)
    scale = np.nanstd(values) if np.isfinite(np.nanstd(values)) else 1.0
    noise = np.abs(values) * rng.uniform(0.2, 5.0, values.shape) if (values[np.isfinite(values)] >= 0).all() \
        else rng.normal(0.0, scale, values.shape)
    out.loc[after] = np.where(np.isnan(values), np.nan, noise)
    return out


def _perturb(wf, price_wide, panic, panels, cut, rng):
    """Every input the engine reads, with its future (strictly after ``cut``) replaced."""
    wf_p = WalkForwardResult(
        **{**wf.__dict__,
           "alpha_scores": _scramble_after(wf.alpha_scores, cut, rng),
           "long_mask": _scramble_after(wf.long_mask, cut, rng),
           "short_mask": _scramble_after(wf.short_mask, cut, rng)}
    )
    # Positive multiplicative noise keeps prices positive and the NaN pattern intact, so
    # the delisting guard sees the same exits — only the *values* of the future change.
    price_p = price_wide.copy()
    after = price_p.index > cut
    price_p.loc[after] = price_p.loc[after] * rng.uniform(0.5, 1.5, price_p.loc[after].shape)
    panic_p = panic.copy()
    panic_p.loc[panic_p.index > cut] = rng.random(int((panic_p.index > cut).sum())) < 0.5
    panels_p = CostPanels(
        adv=_scramble_after(panels.adv, cut, rng),
        sigma=_scramble_after(panels.sigma, cut, rng),
        half_spread_bps=_scramble_after(panels.half_spread_bps, cut, rng),
        shortable=None if panels.shortable is None else _scramble_after(panels.shortable, cut, rng),
    )
    return wf_p, price_p, panic_p, panels_p


def _first_divergence(a: ExecutionResult, b: ExecutionResult, cut: pd.Timestamp) -> list[str]:
    """Names of every series that differs on a bar it must not (empty list = causal)."""
    bad = []
    for name in DECIDED:
        x, y = getattr(a, name), getattr(b, name)
        idx = x.index[x.index <= cut].intersection(y.index)
        if not x.loc[idx].equals(y.loc[idx]):
            first = idx[(x.loc[idx] != y.loc[idx]) & ~(x.loc[idx].isna() & y.loc[idx].isna())]
            bad.append(f"{name} (first bad bar {first[0].date() if len(first) else '?'})")
    idx = a.gross.index[a.gross.index < cut].intersection(b.gross.index)
    if not a.gross.loc[idx].equals(b.gross.loc[idx]):
        bad.append("gross")
    return bad


def _changed_after(a: ExecutionResult, b: ExecutionResult, cut: pd.Timestamp) -> bool:
    """The perturbation must actually reach the engine, or an all-PASS means nothing."""
    x, y = a.turnover, b.turnover
    idx = x.index[x.index > cut].intersection(y.index)
    return not x.loc[idx].equals(y.loc[idx])


def main() -> int:
    config.ML_CONFIG.cost = replace(config.ML_CONFIG.cost, **config.PHASE4C_COST_OVERRIDES)
    wf, price_wide = compute_signals(FREQUENCY, DECILE_PCT, target_col=TARGET)
    panic = compute_regime(config.HEADLINE_HMM_STATES, [FREQUENCY]).panic
    panels = load_cost_panels(need_shortable=True)
    terminal = load_terminal_returns(FREQUENCY)
    bars = wf.alpha_scores.index
    # Each cut moves forward to the first non-panic bar. On a panic bar long_only and
    # long_short are flat, so a leak into that bar has no position to move and the check
    # would pass vacuously — the canary first failed for exactly this reason (2020-10-13).
    calm = bars[~panic.reindex(bars).fillna(False).astype(bool).to_numpy()]
    cuts = [calm[calm >= bars[int(len(bars) * f)]][0] for f in CUT_FRACTIONS]

    def run(wf_, price_, panic_, panels_, style, construction="buffer"):
        cfg = make_ml_config(FREQUENCY, style, target_col=TARGET, construction=construction)
        return run_execution(wf_, panic_, price_, cfg, terminal_returns=terminal, panels=panels_)

    ok = True
    rows = []
    for construction in CONSTRUCTIONS:
        for style in STYLES:
            base = run(wf, price_wide, panic, panels, style, construction)
            for i, cut in enumerate(cuts):
                rng = np.random.default_rng(SEED + i)
                pert = run(*_perturb(wf, price_wide, panic, panels, cut, rng), style, construction)
                bad = _first_divergence(base, pert, cut)
                reached = _changed_after(base, pert, cut)
                passed = not bad and reached
                ok &= passed
                rows.append((f"{construction}/{style}", cut.date(), "PASS" if passed else "FAIL",
                             "perturbation reached engine" if reached else "PERTURBATION HAD NO EFFECT",
                             ", ".join(bad) or "-"))

    # Canary: a one-bar leak must be caught at the middle cut, or the test has no teeth.
    # The leaked scores keep the real NaN pattern: a bare shift(-1) would also score a
    # name on its last listed bar, and the delisting guard (correctly) refuses that.
    def _leak(alpha: pd.DataFrame) -> pd.DataFrame:
        return alpha.shift(-1).where(alpha.notna())

    leaky = WalkForwardResult(**{**wf.__dict__, "alpha_scores": _leak(wf.alpha_scores)})
    cut = cuts[1]
    rng = np.random.default_rng(SEED)
    leak_base = run(leaky, price_wide, panic, panels, "long_only")
    wf_p, price_p, panic_p, panels_p = _perturb(wf, price_wide, panic, panels, cut, rng)
    leaky_p = WalkForwardResult(**{**wf_p.__dict__, "alpha_scores": _leak(wf_p.alpha_scores)})
    leak_pert = run(leaky_p, price_p, panic_p, panels_p, "long_only")
    caught = bool(_first_divergence(leak_base, leak_pert, cut))
    ok &= caught

    print("\n" + "=" * 96)
    print("NO-LOOK-AHEAD CHECK — inputs after the cut replaced by noise; decisions up to it must not move")
    print("=" * 96)
    print(f"  {'construction/style':<26} {'cut':<12} {'verdict':<8} {'sanity':<28} diverged series")
    for r in rows:
        print(f"  {r[0]:<26} {str(r[1]):<12} {r[2]:<8} {r[3]:<28} {r[4]}")
    print(f"  canary (alpha shifted one bar early) caught: {'PASS' if caught else 'FAIL — the test cannot see a leak'}")
    print("=" * 96)
    print(f"VERDICT: {'causal — no decision reads the future' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
