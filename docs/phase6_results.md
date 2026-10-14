# Phase 6 Part B: results — new information, cheaper trading

Run 2026-10-14 (Stage 7, 14:22–18:26) against the pre-registration in
[`phase6_plan.md`](phase6_plan.md) §B, committed as `1292b1e` before any Part B number existed.
Command: `python run_pipeline_ml.py --phase6`. Ledger: `data/trial_database/phase6/phase6_dsr_matrix.parquet`
(18 columns, no NaN, 2016-01-21 → 2025-06-27), gate `phase6_dsr_matrix_dsr_gate.csv`,
execution diagnostics `phase6_dsr_matrix_execution_diagnostics.csv`. Diagnostics write-up:
[`phase6_stage7_diagnostics.md`](phase6_stage7_diagnostics.md).

## Headline: **0 of 18 cells pass at N = 57. The §B.6 stop rule fires.**

The best DSR is **0.182**: `dynamic_tilt_slb` / 5b / F0 (`p17`), net SR 0.78 after the Lo
correction, against SR\* = 1.09. `minTRL` is undefined for every cell: none clears SR\* at any
track length. **The new features made every cell worse.** Within each style × target, F0 beats
F1 and F2 in all six cases.

**The holdout was not scored** (Stage 8 not run, by decision; §B.8, 2026-10-14). The headline
had already failed, so §B.5's "passes the headline **and** survives the holdout" could not be
met. The post-2025-06-30 data stays sealed for a future pre-registration.

## All 18 cells

Net and gross SR are naive √252. SR_ann is Lo-corrected, and DSR is computed at N = 57.
Cost-swap construction throughout. The 21b cells rebalance every 21 bars (`_R21`).

| target | style | features | gross SR | net SR | SR_ann | DSR | turnover | trade drag /yr |
|---|---|---|---|---|---|---|---|---|
| 5b | dynamic_tilt_slb | p17 | 1.51 | **0.84** | 0.78 | **0.182** | 0.225 | 15.0% |
| 5b | dynamic_tilt_slb | p17x | 0.69 | 0.02 | 0.02 | 0.001 | 0.230 | 14.7% |
| 5b | dynamic_tilt_slb | p17xo | 0.82 | 0.17 | 0.15 | 0.002 | 0.228 | 14.7% |
| 5b | long_only | p17 | 1.31 | 0.81 | 0.73 | 0.140 | 0.150 | 10.1% |
| 5b | long_only | p17x | 0.67 | 0.18 | 0.16 | 0.002 | 0.149 | 9.8% |
| 5b | long_only | p17xo | 0.75 | 0.27 | 0.24 | 0.005 | 0.146 | 9.8% |
| 5b | long_short_slb | p17 | 1.28 | 0.23 | 0.23 | 0.004 | 0.217 | 13.3% |
| 5b | long_short_slb | p17x | 0.11 | −1.02 | −0.94 | 0.000 | 0.235 | 14.0% |
| 5b | long_short_slb | p17xo | 0.41 | −0.73 | −0.65 | 0.000 | 0.235 | 14.1% |
| 21b | dynamic_tilt_slb | p17 | 0.65 | 0.47 | 0.45 | 0.026 | 0.061 | 3.4% |
| 21b | dynamic_tilt_slb | p17x | 0.51 | 0.34 | 0.30 | 0.008 | 0.061 | 3.4% |
| 21b | dynamic_tilt_slb | p17xo | 0.40 | 0.25 | 0.22 | 0.004 | 0.058 | 3.3% |
| 21b | long_only | p17 | 0.65 | 0.49 | 0.45 | 0.027 | 0.048 | 3.0% |
| 21b | long_only | p17x | 0.58 | 0.44 | 0.38 | 0.017 | 0.047 | 2.9% |
| 21b | long_only | p17xo | 0.49 | 0.36 | 0.31 | 0.009 | 0.046 | 2.8% |
| 21b | long_short_slb | p17 | 0.14 | −0.28 | −0.27 | 0.000 | 0.086 | 4.8% |
| 21b | long_short_slb | p17x | 0.01 | −0.39 | −0.37 | 0.000 | 0.084 | 4.6% |
| 21b | long_short_slb | p17xo | −0.16 | −0.56 | −0.54 | 0.000 | 0.081 | 4.5% |

## Secondary (§B.5): Δ net SR from adding each feature set, all 12 reported, none selected

| target | style | F0 → F1 | F1 → F2 |
|---|---|---|---|
| 5b | dynamic_tilt_slb | −0.82 | +0.15 |
| 5b | long_only | −0.63 | +0.09 |
| 5b | long_short_slb | −1.25 | +0.29 |
| 21b | dynamic_tilt_slb | −0.13 | −0.10 |
| 21b | long_only | −0.05 | −0.08 |
| 21b | long_short_slb | −0.11 | −0.16 |

F1 hurts all six cells. F2 claws back a little at 5b and loses at 21b. No cell ends up above
its F0 version.

## Why: the signal got weaker, not the trading

Ensemble rank-IC, by fit (`tree_fit_diagnostics_daily_nse500_*_purge.csv`):

| target | p17 | p17x | p17xo |
|---|---|---|---|
| 5b | 0.049 (t 21.4) | 0.025 (t 8.2) | 0.026 (t 8.3) |
| 21b | 0.034 (t 12.3) | 0.019 (t 5.9) | 0.018 (t 5.6) |

- **Every learner drops**, Random Forest most of all: at 5b, 0.044 → 0.009. LightGBM goes
  0.049 → 0.030 and XGBoost 0.048 → 0.027.
- **Costs are not the story.** Turnover and trade drag are nearly identical across feature
  sets within each style × target. The net SR loss is the gross SR loss.
- **The new features do carry information on their own** (§B.5 read-only, per-feature IC).
  `fut_basis` is the strongest single feature in the study and strengthens after 2021: NW t
  5.6 early, 11.7 late, 6.3 recent. `opt_iv_rv` and `fut_oi_chg_5b` appear only recently.
  But the low-vol-family additions (`ivol_63b`, `beta_252b`, `opt_atm_iv`) decay the same
  way Part A found. Alignment checks pass 18/18, so this is not a wiring bug.
- **Post-2021, the aim of Part B, was not met.** At 5b, F1 helped in 2021–22 and hurt in 2023–25.
  The 2021–25 mean IC is 0.023 (p17) vs 0.015 (p17x). By 2024–25 every feature set's IC is ≈ 0
  or negative.
- **Reading, not tested:** the trees reshuffle the whole ranking instead of adding the strong
  signals on top. The rank-corr of the new fits' alpha with F0 is 0.11–0.43. The raw `mkt_*`
  columns give splits a per-date level to chase that does nothing for a cross-sectional rank:
  the per-date mean is 26–30% of label variance. Any test of this needs a new pre-registration,
  and §B.6 now forbids price/index/options-derived features anyway.

## Monthly rebalancing (21b, `_R21`)

It did what it was built for. Turnover fell to 0.05–0.09 per bar, and trade drag to 3–5%/yr
from 10–15%. But the 21b gross Sharpe is too low (≤ 0.65) for the saving to matter. The
best 21b cell nets 0.49, below every 5b F0 cell except `long_short_slb`.

## Against Phase 5

The F0 cells are Phase 5's cost-swap cells with the own-bar label-overlap purge (§A.7) turned
on. That is the only difference in their ids, `_PG`. The leak-free versions are lower:
`dynamic_tilt_slb`/5b went from net SR 1.03 to 0.84 naive (Lo 0.96 → 0.78), CAGR 23.7% → 18.1%,
max drawdown −39.2% → −43.5%. **The Phase 6 F0 numbers are the cleanest measure of the
program's best strategy.** Phase 5's DSR 0.272 at N = 42 stands as recorded, but it
includes the leak.

## What this closes, and what is left

- **§B.6 fires:** no more features derived from price, index or options data. This rules out
  the variants the diagnostics suggest (F0 + `fut_basis` only, F1 without `mkt_*`). They would
  be selection after the fact (§B.7) as well.
- **Open, each needing its own pre-registration and N:**
  - delivery data as F3 (the MTO archive reaches 2013, §B.7);
  - Part A's parked cells (a 1b horizon, a 1008-day window).
  - Any of these, or a new program, is tested on the sealed holdout, which is the only clean
    data left.
- **The program-level reading:** the trees' cross-sectional edge is real (IC ≈ 0.05 at 5b),
  but it is too thin for NSE costs and has faded to ≈ 0 by 2024–25. Neither more price-derived
  information (Part B) nor cheaper trading (M2, R21) changed that.
