# Phase 6 Part A: results — why the signal decays

Run 2026-10-06 against the pre-registration in [`phase6_plan.md`](phase6_plan.md) §A
(committed `f5fefb5` before any number existed). Script: `scripts/phase6_decay.py --run`.
Tables: `data/trial_database/phase6/decay/*.csv`. Read-only: N stays **42**, and every DSR
ledger hashes the same before and after. The post-2025-06-30 holdout was not read.

## Headline: **no pre-registered rule fires.**

| # | fired | the numbers |
|---|---|---|
| H1 competed away | **no**, missed on one count | 7 of 11 skill features decayed (64%; the rule needs ⅔, i.e. 8). The other two conditions **hold**: ceiling recent/early −0.03 (needs ≤ 0.5), ridge late/early 0.29 vs ensemble 0.40 (needs within ±0.2) |
| H2 regime flip | no | 0 sign flips; the largest feature × state difference is \|t\| 2.64 (needs 3) |
| H3 training setup | no | the ceiling collapsed (recent/early −0.03, needs ≥ 0.7); the 252-day window recovers −11% of the gap (needs ≥ 50%) |
| H4 universe drift | no | the age mix explains −1% of the IC drop (needs ≥ 50%) |
| H5 data | no | disagreement late/early 0.29×, extremes 0.51× (needs ≥ 2×); removing extremes changes late IC by −0.2% |

Under §A.4 this means the decay is not explained by any of the five hypotheses *as
pre-registered*. The rule is not relaxed after the fact. What the evidence does say is in §2.
It is descriptive, it is in-sample for 2021–25 (§A.6), and Part B may use it only as an
input to its own pre-registration, judged on the holdout.

Anchors (§A.5) both passed before anything new was read:
- The ensemble 1b IC by year reproduces `phase4c_results.md` §4.2 to 4 decimals.
- The standalone lgbm, xgb and rf full-sample IC match the cached ensemble's per-learner
  IC exactly (0.0501 / 0.0488 / 0.0462).

---

## 1 · The tables

### 1.1 · A1: which features decayed (5b label, NW t, lag 4)

| feature | early IC | early t | late IC | late t | late/early | decayed |
|---|---|---|---|---|---|---|
| mom_logret_1b | −0.0216 | −6.60 | −0.0246 | −7.29 | 1.14 | no |
| mom_logret_5b | −0.0325 | −5.74 | −0.0360 | −6.50 | 1.11 | no |
| mom_close_sma20 | −0.0286 | −4.36 | −0.0242 | −3.79 | 0.85 | no |
| liq_turnover_20b | −0.0122 | −3.88 | −0.0089 | −3.07 | 0.73 | no |
| mom_logret_252b | 0.0350 | 3.67 | 0.0166 | 2.41 | 0.47 | **yes** |
| mom_max_dret_20b | −0.0511 | −8.47 | −0.0213 | −3.25 | 0.42 | **yes** |
| vol_realized_20b | −0.0604 | −7.10 | −0.0213 | −2.51 | 0.35 | **yes** |
| vol_realized_60b | −0.0657 | −6.60 | −0.0255 | −2.59 | 0.39 | **yes** |
| vol_parkinson_20b | −0.0650 | −7.07 | −0.0242 | −2.63 | 0.37 | **yes** |
| vol_drawdown_252b | 0.0451 | 3.97 | 0.0195 | 2.64 | 0.43 | **yes** |
| liq_amihud | −0.0361 | −5.42 | −0.0148 | −2.45 | 0.41 | **yes** |

The six other features do not carry skill (early |t| < 3).

### 1.2 · Model vs features vs ceiling (5b IC by year)

| year | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 H1 |
|---|---|---|---|---|---|---|---|---|---|---|
| ensemble (OOS) | 0.049 | 0.054 | 0.078 | 0.103 | 0.073 | 0.011 | 0.045 | 0.049 | 0.023 | 0.000 |
| ridge (OOS, same folds) | 0.040 | 0.062 | 0.088 | 0.112 | 0.041 | 0.009 | 0.018 | 0.051 | 0.021 | −0.023 |
| within-year ceiling | 0.037 | 0.046 | 0.087 | 0.077 | −0.027 | 0.005 | 0.022 | 0.020 | 0.011 | −0.028 |
| fixed-sign composite | 0.073 | 0.069 | 0.117 | 0.112 | 0.067 | 0.048 | 0.061 | 0.032 | 0.046 | 0.044 |
| lgbm 252-day window | — | — | 0.084 | 0.086 | 0.050 | 0.010 | 0.044 | 0.043 | 0.007 | 0.008 |
| lgbm 504-day | — | — | 0.081 | 0.100 | 0.069 | 0.007 | 0.048 | 0.052 | 0.019 | 0.000 |
| lgbm 1008-day window | — | — | 0.093 | 0.106 | 0.068 | 0.025 | 0.052 | 0.042 | 0.027 | 0.013 |

The three learners decay together. Their per-year ICs are within 0.01 of each other in every
year (`a1b_learner_ic_by_year.csv`). The window rows are on their common dates
(2018-02 → 2025-06).

The composite's signs are frozen on 2016–20, so it is in-sample there and out-of-sample from
2021. At the 1-day horizon it holds at IC 0.051–0.058 in every late year
(`a1_composite_by_year.csv`). That level is carried by the reversal features.

---

## 2 · What the evidence says (descriptive, not a fired rule)

1. **The information in the 17 features is fading, and no model rescues it.** A model fitted
   inside each year, with knowledge of the year's future, cannot find recent IC (−0.001).
   Ridge decays like the trees, and all three tree learners decay together. This is the
   H1 pattern in every respect but one: the feature count missed ⅔ by one feature.
   The construction caveat: the in-year ceiling trains on about 190 days against the
   walk-forward's 504, so it sits *below* the OOS ensemble even in early years, and it goes
   negative in 2020, when the quarters of one year are different regimes. It is weaker than
   designed, but it is built the same way every year, and its ratio is what the rule reads.
2. **It is a specific family, not every feature.** The decayed features are the low-volatility
   and lottery family: realized vol, Parkinson vol, max daily return, Amihud illiquidity,
   252-day momentum and drawdown. Their IC is down about 60%. Short-term reversal (1- and
   5-day returns, distance from the 20-day SMA) and turnover held or **strengthened**. The 5b
   label leaned on the family that decayed.
3. **The late market is one the low-vol family historically does worst in, but the state
   test does not confirm that as the cause.** Late days sit far more in the strong-uptrend
   tercile (52% vs 5% early) and the low-dispersion tercile (60% vs 35%). The feature × state
   differences peak at |t| 2.6, on avg-corr. The background story is plausible but not
   established.
4. **Longer memory beats shorter memory.** The 1008-day window is better than 504 in recent
   data (0.023 vs 0.013), and 252 is worse (0.007). Fold IC correlates −0.50 with the share of
   COVID in the training window and −0.48 with training-window vol. A frozen 2016–20 sign
   rule beats all of them from 2021 on: late 5b IC 0.047 vs 0.029 for the ensemble. This is
   the **opposite** of "the model adapts too slowly". If anything the trees chase each recent
   window's noise, and the high-vol 2020 window taught them the wrong thing.
5. **The data, the universe and the learners are not the cause.** H4 and H5 are clean nulls,
   and A1b rules out a single learner breaking.

## 3 · What this means for Part B

- The pre-registered verdict is "unexplained". Part B therefore treats the signal as
  non-stationary, and the **holdout carries the full burden of proof**.
- The direction the evidence points: **new information** (delivery, fundamentals, co-movement)
  rather than a model change. Re-tuning the window or the learner cannot recover IC that a
  within-year fit cannot find.
- Two observations are worth a pre-registered Part B cell each. Both are in-sample for
  2021–25 and each must be paid for in N:
  - a **horizon matched to the surviving reversal family** (1b, where the frozen composite
    holds at IC ≈ 0.055 through 2025), though the cost stack has already shown that 1b
    turnover is fatal without M2;
  - a **heavier-regularised or longer-memory learner**, given the 1008-day and
    frozen-composite results.
- None of this is a strategy. ICs of 0.03–0.06 are what Phase 5 already showed cannot clear
  costs at the required net SR of about 1.8.
