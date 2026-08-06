"""
Phase 4c §7 step 3 — cost sanity.

Charging zero spread and zero impact through the NEW per-name code path must
reproduce Phase 4b exactly; stepping spread in must then degrade the result
monotonically. If either fails, the per-name cost path is wrong and every number
downstream of it is meaningless.
"""
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/jash/Desktop/Quant Research/qrtf_engine")
os.chdir("/Users/jash/Desktop/Quant Research/qrtf_engine")

import config
from src.production_ml.tier1_trees import DECILE_PCT
from src.production_ml.tier2_regime import RegimeDetector, build_features
from src.production_ml.tier3_execution import CostPanels, run_execution
from run_pipeline_ml import (
    compute_signals, load_terminal_returns, make_ml_config, phase4_regime_config,
)

FREQ = "daily_nse500"
TARGET = "tgt_fwd_logret_5b"

wf, price_wide = compute_signals(FREQ, DECILE_PCT, target_col=TARGET)
panic = RegimeDetector(
    n_states=config.HEADLINE_HMM_STATES
).fit_predict(build_features(phase4_regime_config())).panic
terminal = load_terminal_returns(FREQ)

idx, cols = wf.alpha_scores.index, wf.alpha_scores.columns
adv = pd.read_parquet(config.ML_CONFIG.phase4.adv_parquet)
sig = pd.read_parquet(config.ML_CONFIG.phase4.sigma_parquet)
spr = pd.read_parquet(config.ML_CONFIG.phase4.spread_cs_parquet)
for f in (adv, sig, spr):
    f.index = pd.DatetimeIndex(f.index).normalize()

ref = pd.read_parquet("data/trial_database/production_dsr_matrix.parquet")
ref_col = {"long_only": "STRAT_LIVE_NSE_LO_L0_HMM2_f936d37f",
           "long_short": "STRAT_LIVE_NSE_LS_L0_HMM2_957f71b7",
           "dynamic_tilt": "STRAT_LIVE_NSE_DT_L0_HMM2_0a0c6a73"}

base = config.ML_CONFIG.cost


def run(style, *, spread_bps=None, coef=None, cap=False, **kw):
    """One Tier 3 run with an explicitly constructed cost environment."""
    panels = CostPanels(
        adv=adv,
        sigma=sig if coef != 0.0 else sig * 0.0,
        half_spread_bps=(
            spr if spread_bps is None
            else pd.DataFrame(float(spread_bps), index=idx, columns=cols)
        ),
        shortable=None,
    )
    cfg_cost = replace(
        base, charge_per_name=True, aum_rupees=1e7,
        enforce_participation_cap=cap,
        impact_coef=base.impact_coef if coef is None else coef, **kw,
    )
    config.ML_CONFIG.cost = cfg_cost
    try:
        return run_execution(wf, panic, price_wide, make_ml_config(FREQ, style),
                             terminal_returns=terminal, panels=panels)
    finally:
        config.ML_CONFIG.cost = base


def sharpe(s):
    s = s.dropna()
    return float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else 0.0


print("\n" + "=" * 86)
print("1. ZERO SPREAD + ZERO IMPACT through the per-name path == Phase 4b?")
print("=" * 86)
allok = True
for style in ("long_only", "long_short", "dynamic_tilt"):
    r = run(style, spread_bps=0.0, coef=0.0)
    a = ref[ref_col[style]].to_numpy()
    b = r.net.reindex(ref.index).to_numpy()
    ex = np.allclose(a, b, atol=1e-15, equal_nan=True)
    allok &= ex
    print(f"  {style:<14} identical={ex}  max|diff|={np.nanmax(np.abs(a-b)):.3e}  "
          f"SR={sharpe(r.net):.3f} (ref {sharpe(ref[ref_col[style]]):.3f})")
print(f"  => {'PASS' if allok else 'FAIL'}")

print("\n" + "=" * 86)
print("2. MONOTONE DEGRADATION as a flat half-spread is stepped in (impact off)")
print("=" * 86)
print(f"  {'spread bps':>11}" + "".join(f"{s:>16}" for s in
      ("long_only", "long_short", "dynamic_tilt")))
prev = {}
mono = True
for s_bps in (0, 2, 4, 6, 8, 10, 15, 20):
    cells = []
    for style in ("long_only", "long_short", "dynamic_tilt"):
        sr = sharpe(run(style, spread_bps=float(s_bps), coef=0.0).net)
        if style in prev and sr > prev[style] + 1e-9:
            mono = False
        prev[style] = sr
        cells.append(f"{sr:>16.3f}")
    print(f"  {s_bps:>11}" + "".join(cells))
print(f"  => strictly non-increasing in spread: {'PASS' if mono else 'FAIL'}")

print("\n" + "=" * 86)
print("3. THE REAL COST STACK — measured CS spread + sqrt-law impact at Rs 1 cr")
print("=" * 86)
print(f"  {'style':<14}{'SR gross':>10}{'SR Ph4b':>10}{'SR real':>10}"
      f"{'bps/side':>10}{'drag/yr':>10}{'part p99':>10}")
for style in ("long_only", "long_short", "dynamic_tilt"):
    r = run(style)
    rc = run(style, cap=True)
    d = r.diagnostics
    print(f"  {style:<14}{sharpe(r.gross):>10.3f}{sharpe(ref[ref_col[style]]):>10.3f}"
          f"{sharpe(r.net):>10.3f}{d['effective_oneway_bps']:>10.2f}"
          f"{d['trade_drag_annual'] + d['borrow_drag_annual']:>9.1%}"
          f"{d['participation_p99']:>10.4f}")
    print(f"  {'  + cap':<14}{'':>10}{'':>10}{sharpe(rc.net):>10.3f}"
          f"{rc.diagnostics['effective_oneway_bps']:>10.2f}"
          f"{rc.diagnostics['trade_drag_annual'] + rc.diagnostics['borrow_drag_annual']:>9.1%}"
          f"{rc.diagnostics['participation_p99']:>10.4f}")
print("=" * 86)
