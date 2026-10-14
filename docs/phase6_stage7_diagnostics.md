# Phase 6 Part B: Stage 7 diagnostics — where the new features fell short

Run 2026-10-14, after the Stage 7 grid (0/18 pass at N = 57, best DSR 0.182). Read-only:
every walk-forward was a `wf_cache` hit, nothing was fitted, N stays **57**, and the
post-2025-06-30 holdout was not read. Script, log and tables:
`data/trial_database/phase6/stage7_diag/` (`stage7_diag.py`, `stage7_diag.log`, `diag/*.csv`).

## Headline

The new features carry real information on their own, but adding them made the ensemble
**worse**: F1 roughly halved IC, and F2 changed almost nothing on top of F1. The aim was to
lift IC after 2021. At 5b it helped in 2021–22 and hurt in 2023–25. Averaged over
2021–2025, IC fell from 0.023 (p17) to 0.015 (p17x). At 21b it was worse in almost every year.

## Test 1 — alignment: 18/18 pass

Adding F1/F2 leaves the 17 base columns and the labels identical (p17x drops 238 rows,
p17xo none). `mkt_*` take one value per date and equal `market.parquet`. Every ranked
feature has rank-corr 1 with its raw panel and sits in [-1, +1]. F2 is 0 wherever the raw
value is missing. **The result is not a wiring bug.**

## Test 2 — per-feature IC of the new features (§B.5, 5b, NW t lag 4)

| feature | early IC (t) | late IC (t) | recent IC (t) |
|---|---|---|---|
| `fut_basis` | 0.025 (5.6) | 0.045 (11.7) | 0.039 (6.3) |
| `opt_avail` | 0.034 (5.6) | 0.024 (3.1) | 0.033 (2.0) |
| `opt_iv_rv` | −0.006 (−0.9) | −0.026 (−4.3) | −0.032 (−2.9) |
| `fut_oi_chg_5b` | −0.002 (−0.5) | −0.014 (−3.4) | −0.021 (−2.5) |
| `ivol_63b` | −0.067 (−7.6) | −0.029 (−3.4) | −0.033 (−1.8) |
| `beta_252b` | −0.039 (−3.5) | −0.009 (−0.8) | −0.029 (−1.3) |
| `opt_atm_iv` | −0.033 (−2.5) | −0.011 (−1.0) | 0.001 (0.1) |

- `fut_basis` is the strongest single feature in the study and **strengthens** over time.
- `ivol_63b`, `beta_252b` and `opt_atm_iv` belong to the low-vol family that Part A found
  decaying, and they decay the same way here.
- `mkt_*` have no cross-sectional IC, because they take one value per date. The per-date mean
  is 26.5% (5b) and 30.5% (21b) of label variance. Corr(`mkt_*`, per-date mean label) is
  ≤ 0.12 in absolute value.

Full tables: `t2_feature_ic_by_year_{5b,21b}.csv`, `t2_feature_ic_by_period_{5b,21b}.csv`.

## Test 3 — ensemble IC

Mean IC by year (own label):

| year | 5b p17 | 5b p17x | 5b p17xo | 21b p17 | 21b p17x | 21b p17xo |
|---|---|---|---|---|---|---|
| 2016 | 0.046 | 0.047 | 0.046 | 0.019 | 0.047 | 0.044 |
| 2017 | 0.055 | 0.024 | 0.027 | 0.035 | 0.004 | 0.009 |
| 2018 | 0.075 | 0.038 | 0.039 | 0.051 | −0.000 | −0.001 |
| 2019 | 0.104 | 0.043 | 0.045 | 0.122 | 0.081 | 0.081 |
| 2020 | 0.068 | 0.001 | 0.003 | 0.059 | 0.036 | 0.037 |
| 2021 | 0.007 | 0.024 | 0.022 | −0.002 | −0.000 | −0.005 |
| 2022 | 0.044 | 0.059 | 0.056 | 0.023 | 0.002 | 0.005 |
| 2023 | 0.047 | 0.020 | 0.020 | 0.060 | 0.043 | 0.038 |
| 2024 | 0.022 | −0.003 | −0.006 | −0.031 | −0.025 | −0.027 |
| 2025 | −0.002 | −0.024 | −0.015 | −0.034 | −0.025 | −0.022 |

- **By liquidity quintile** (1 = most liquid), 5b IC falls from 0.036–0.052 (p17) to 0.021–0.030
  (p17x). At 21b, quintile 1 goes to ≈ 0.
- **The predictions change, they are not refined**: rank-corr of alpha vs the p17 fit is
  0.11–0.43 by year.

## Reading (not tested — any check is a new pre-registration with its own N)

1. **Good ingredients, worse mix.** The trees reshuffle the whole ranking rather than adding
   the strong new signals on top, so `fut_basis` gets diluted.
2. **Redundant decaying features.** Three more low-vol-family features feed the model
   more of the signal that had stopped working.
3. **`mkt_*` likely noise for ranking.** On a raw forward-return label the trees can spend
   splits on the per-date level, which does nothing for a cross-sectional rank. This is the
   standing Stage 7 hypothesis.
4. **Coverage.** F2 exists for F&O names only (`opt_avail` = 1 on 33.9% of p17xo rows), and is
   0 for the rest.

The verdict stands: 0/18 pass and the §B.6 stop rule fires. That rule ends new features
built from price, index or options data, so the variants these readings suggest are not
available: p17 + `fut_basis` only, or p17x without `mkt_*`. A cell chosen after reading these
results would be selection anyway (§B.7). The only route left is new data: delivery, as F3,
under its own pre-registration. That design should guard against the dilution seen here.
