# Phase 6 plan: why the signal decays, then a better signal

Status: **Part A pre-registered** (2026-10-06). Part B is not yet planned.
Idea note: [`phase6_idea.md`](phase6_idea.md).

---

## A · Pre-registration: diagnose the post-2021 decay (read-only, N + 0)

**Committed before any Part A number exists.** Everything in this section is frozen. A
diagnostic whose decision rule is written after its table is seen can be made to say
anything, which defeats its purpose.

### A.0 · What is being explained

The ensemble's rank-IC against the 1-day forward return, from `docs/phase4c_results.md` §4.2
(daily_nse500, 5b target):

| year | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 H1 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean IC | 0.0475 | 0.0523 | 0.0586 | 0.0863 | 0.0564 | 0.0217 | 0.0312 | 0.0368 | 0.0183 | 0.0127 |
| IC_t | 9.12 | 11.42 | 9.85 | 10.48 | 8.00 | 4.89 | 6.28 | 7.91 | 2.31 | 1.21 |

Already ruled out: costs (IC is measured before costs), survivorship (the universe is
point-in-time and keyed on ISIN), a full-sample liquidity artefact (IC is flat across
quintiles), and a single learner breaking (each learner's full-sample IC is about 0.05).

### A.1 · Scope and the sealed holdout

- **Data.** Only the frozen panel is used: `data/bhavcopy/panel_daily.parquet`, ending
  2025-06-30. **No bhavcopy dated after 2025-06-30 is downloaded or read in Part A.** That
  period is the forward test for Part B. Looking at it now would spend it. The script asserts
  on the panel's maximum date.
- **Engine.** No engine module is edited. New fits go through the existing
  `cached_walk_forward`. Its key already separates `model_choice` and the training window, so
  every fit below is a separate cache entry and the Phase 4c/5 entries are untouched.
- **Ledgers.** Nothing is written to any DSR ledger. Tables go to
  `data/trial_database/phase6/decay/`.
- **N.** Part A adds **0** to `config.TRIAL_LEDGER` (N stays 42), on a 0-cell line with its
  rationale. The Phase 3 §7.3 precedent charged 1 for a tested-and-rejected variant. That
  charge applied because the variant was a candidate strategy. Nothing here is a candidate, and
  A.6 makes that binding.

### A.2 · Definitions

- **Periods.** Early = 2016–2020. Late = 2021-01-01 → 2025-06-30. Recent = 2024-01-01 →
  2025-06-30. All IC is out-of-sample from the walk-forward unless labelled *ceiling*.
- **IC.** The daily cross-sectional Spearman correlation inside the point-in-time universe mask
  (`phase4c_diagnostics._spearman_by_row`, minimum 5 names), summarised by period mean.
- **Horizons.** IC is measured at two horizons:
  - **1b** is the 1-day forward simple return, the §4.2 convention. Its t-stat is i.i.d.,
    which reproduces §4.2.
  - **5b** is the forward 5-day log return, the primary label. 21b, the secondary label, is
    computed the same way. Their t-stat is **Newey-West with lag h − 1**, because overlapping
    labels inflate the i.i.d. t by about √h.
- **Skill-carrying feature.** One of the 17 price features whose early 5b IC has |NW t| ≥ 3.
  Its sign is the sign of its early mean IC.
- **Decayed.** A skill-carrying feature has decayed if its late 5b |NW t| < 2, **or** its late
  signed IC is no more than 50% of its early signed IC.

### A.3 · Tests

**A1 · Per-feature IC by year (H1). Run this first.**
- Each of the 17 rank-normalized features is scored alone, by year, at 1b and 5b.
- A **fixed-sign composite** is built as the equal-weight sum of the skill-carrying features,
  each multiplied by its early sign. It is scored by year. This is a model-free signal: the
  signs are frozen on early data and never refit.

**A1b · Per-learner IC by year.** `lgbm`, `xgb` and `rf` are each refit alone on the frozen
daily_nse500 setup (504/63, 5b) via `cached_walk_forward`. They are scored by year beside the
cached ensemble.

**A2 · State conditioning (H2).** The state variables are all causal and all known by the
close of t:
- **z-vol:** z-scored log 20-day realized index vol (from `tier2_regime.build_features`).
- **avg-corr:** z-scored mean pairwise correlation (same source).
- **trend:** index 252-day log return, ending at t.
- **dispersion:** cross-sectional stdev of the trailing 21-day returns across the universe,
  ending at t.

Terciles are set on an **expanding** window, so the tercile of t uses only states up to t. For
each skill-carrying feature, the test compares daily 5b IC in the top state tercile with the
bottom one (NW t on the difference). The script also reports sign flips by year and each
state's tercile mix in early vs late.

**A3 · Training setup (H3).**
- **(a) Ridge walk-forward.** `sklearn.linear_model.Ridge(alpha=1.0)` on the same 17
  features, the same 504/63 folds and the same 5b target. The alpha is fixed and not tuned.
  Features are already in [-1, +1].
- **(b) Within-year IC ceiling.** For each calendar year Y, the year's days are split into its
  calendar quarters (2025 has two). Each quarter is predicted by an `lgbm` with the
  `ML_CONFIG.lgbm` hyperparameters, trained on the year's other quarters. Training rows within
  h days of the test quarter are purged on both sides. The ceiling is the mean 5b IC of these
  predictions. It answers how much in-year signal the 17 features contain, given in-year
  knowledge. It is not a strategy: it uses the future of the year.
- **(c) Window probe.** `lgbm` refit with `train_window` ∈ {252, 1008}. The 504 case is the
  A1b `lgbm` fit. All three are scored on their **common** OOS dates only.
- **(d) Fold composition.** Each fold's OOS IC is compared with (i) the share of its training
  window inside 2020-03-01 → 2020-12-31 and (ii) the training window's mean z-vol. The report
  gives the Spearman correlation across folds. This test is descriptive and has no rule.

**A4 · Universe drift (H4).**
- **Listing age** is the time since the ISIN's first row in the panel. It is left-censored at
  2013-01-01. Buckets are 1–3y, 3–5y and 5y+ (censored names go in 5y+). No name is scored
  under 1 year, because every feature needs 252 bars.
- IC is reported by year × age bucket, and by year × age bucket × liquidity quintile
  (`liquidity_quintiles`).
- Also reported: the age-bucket share of the scored cross-section by year, and the scored /
  universe coverage ratio by year.
- **Reweighting decomposition.** The late ensemble 5b IC is recomputed with the age-bucket
  weights held at their early mix. The share of the early-to-late drop that this removes is
  the share attributable to drift.

**A5 · Data integrity (H5).** The idea note's test ("daily closes vs bhavcopy closes")
compares a source with itself, because the daily_nse500 panel *is* bhavcopy. It is replaced
by three checks:
- **(a) Independent-source disagreement.** Daily close-to-close returns from the panel are
  compared with `data/daily_ohlcv.parquet` (the 68-name Google-Drive feed, an independent
  vendor) on matched ticker-days. The disagreement rate is the share of ticker-days with
  |r_panel − r_drive| > 1%, by year.
- **(b) Corporate-action ledger.** Rows in `data/bhavcopy/corporate_actions.csv` by year, and
  the share flagged `review`.
- **(c) Unledgered extremes.** Ticker-days in the universe with |r| > 20% and no ledger entry
  within ±1 day, by year. The script also recomputes the late ensemble 5b IC with those
  ticker-days' forward labels removed (excluded from the IC, not refit).

### A.4 · Decision rules

The hypotheses are not exclusive. Every rule that fires is reported. None is selected
because it reads best.

| # | Supported if | Points Part B to |
|---|---|---|
| H1 competed away | ≥ ⅔ of the skill-carrying features have **decayed**, **and** the recent ceiling (A3b) is ≤ 50% of the early ceiling, **and** ridge's late/early IC ratio is within ±0.2 of the ensemble's | new information: delivery, fundamentals, co-movement |
| H2 regime flip | ≥ 3 skill-carrying features have a late year with IC of the **opposite** sign at \|NW t\| ≥ 2, **or** some feature × state difference has \|NW t\| ≥ 3 **and** that state's late tercile mix differs from early by ≥ 10 pp in some tercile | regime-conditional features or model |
| H3 training setup | the recent ceiling is ≥ 70% of early **and** at least half the skill-carrying features have not decayed, **while** the recent OOS ensemble IC is ≤ 50% of early; **or** the 252-day probe recovers ≥ 50% of the gap between the 504-day `lgbm` and its early level on the recent common dates | retrain window, frequency or regularisation |
| H4 universe drift | the reweighting decomposition removes ≥ 50% of the early-to-late ensemble-IC drop | minimum-history universe filter |
| H5 data | the late (a) disagreement rate or the late (c) extremes rate is ≥ 2× its early rate, **or** removing (c) raises late ensemble 5b IC by ≥ 25% | fix the data; the decay is partly artificial |

If **no** rule fires, that is the finding: the decay is not explained by any of the five
hypotheses at this resolution. Part B must then treat the signal as non-stationary for an
unknown reason, which raises the bar for a forward test.

### A.5 · Anchors (checked before any new number is read)

1. The script's ensemble 1b IC by year must reproduce A.0 to 4 decimals (IC_t to 2).
2. The A1b single-learner full-sample 5b IC must match the cached `ic_diagnostics` (lgbm
   0.0501, xgb 0.0488, rf 0.0462) to 4 decimals. Cold-fit jitter is ≤ 5.6e-17, so this match
   is expected.

If either anchor fails, Part A stops and the discrepancy is the first thing investigated.

### A.6 · What Part A does not license

- No Part A table is promoted, and no Part A fit enters a ledger.
- Any Part B choice informed by Part A, such as a window length, a feature subset, a regime
  conditioning or a universe filter, counts as **in-sample for 2021–2025**. It carries its own
  N and is judged on the sealed post-2025-06-30 holdout, not only on the existing window.

---

## B · Find a better signal

Not yet planned. Written after Part A's result, as its own pre-registration.
