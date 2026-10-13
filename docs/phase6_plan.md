# Phase 6 plan: why the signal decays, then a better signal

Status: **Part A pre-registered** (2026-10-06) and run (results:
[`phase6_decay_results.md`](phase6_decay_results.md)). **Part B pre-registered** (2026-10-14).
Execution plan: [`phase6_partB_plan_draft.md`](phase6_partB_plan_draft.md).
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

### A.7 · Found during Part A: a Tier 1 training-label leak (fixed opt-in)

Added 2026-10-07. This is not part of the pre-registration. It is a defect found while
answering a question about the walk-forward.

**The leak.**
- `TreeAlphaEngine.run_walk_forward` trains on every row of the 504-day block.
- A row on bar d carries the h-bar label log(close[d+h] / close[d]).
- For the last h rows of the block, d+h falls inside the scored block that follows, so the
  model was fitted on returns it was then tested on.
- That affects h of 504 rows: under 1% at 5b and about 4% at 21b.
- It can only flatter a result.
- `scripts/no_lookahead_check.py` perturbs Tier 3 inputs only, so it never saw this.

**The fix** (`src/production_ml/tier1_trees.py`):
- `purge_label_overlap()` drops the last h bars of each training block. h is read from the
  target name by `target_horizon()`.
- Every training label left is then realised by the block's last bar.
- The fix is opt-in: `TreeAlphaEngine(purge_label_overlap=True)`, or
  `run_pipeline_ml.py --purge-label-overlap`, which sets `config.ML_CONFIG.purge_label_overlap`.
- When on, it adds:
  - a `_PG` tag and hash term to `strategy_id`;
  - a `purge` part to the wf_cache key;
  - its own `tree_fit_diagnostics_*_purge.csv`, so the unpurged files earlier phases cite are
    never overwritten.

**Verification.**
- The tier1 dry-run gained seven purge checks. They include a canary that records every
  fold's training rows: unpurged, a label must reach the scored block; purged, none may and
  none more than necessary are dropped.
- The wf_cache dry-run checks that the purge gives a new key.
- With the flag off, the Phase 5 grid was re-run (`data/trial_database/regression/phase6_purge_off*`).
  `scripts/regression_check.py` reports **4/4 pairs bit-identical** against
  `data/trial_database/phase5/phase5_dsr_matrix*`, and every existing `tree_fit_diagnostics_*.csv`
  is unchanged.

**Size** (`lgbm` only, daily_nse500, `data/trial_database/phase6/purge/lgbm_quick_check.csv`):

| target | full IC leaky → purged | 2016–20 | 2021–25 H1 | mean daily rank-corr (leaky vs purged alpha) |
|---|---|---|---|---|
| 5b | 0.0501 → 0.0488 (−3%) | 0.0700 → 0.0677 | 0.0281 → 0.0280 | 0.91 |
| 21b | 0.0400 → 0.0347 (−13%) | 0.0690 → 0.0590 | 0.0075 → 0.0074 | 0.89 |

**Consequences.**
- No verdict changes: Phase 4c and Phase 5 failed already, and the fix only lowers the
  numbers.
- The 21b cells were overstated more than the 5b cells.
- Part A's 5b conclusions stand: late IC is unchanged, and early IC falls slightly.
- The purged ensemble re-score of the Phase 5 grid was started and then paused. It is not
  run, and it becomes the F0 baseline of Part B.
- From Phase 6 onwards every fit runs with the purge on.

**Addendum (2026-10-14, Stage 6).** The rule above counts calendar bars. Labels count each
name's own bars, so across a halt 0.02% of labels still reached the scored block. The purge
now uses each label's own realisation date. See §B.8.

---

## B · Pre-registration: new information, cheaper trading (N 42 → 57)

**Committed before anything in it is built, and before any return is computed on a new
feature.** Everything in this section is frozen. The execution steps are in
[`phase6_partB_plan_draft.md`](phase6_partB_plan_draft.md).

### B.0 · Why

Part A fired no rule. Its evidence still says the information in the 17 price features is
fading, the low-volatility family above all, and that no model rescues it. A model fitted
inside each year, with that year's future, finds no IC in 2024–25. Part B therefore changes
the **information**, not the model, and adds a **cheaper way to trade**: a monthly rebalance
for the 21b label. Under §A.6, anything Part B takes from Part A is in-sample for 2021–25 and
is judged on the sealed holdout.

### B.1 · Scope and what is held fixed

- **Data.** The frozen panel ends 2025-06-30, and the holdout after that stays sealed until
  B.5's holdout step. **No file dated after 2025-06-30 is downloaded or read before then.**
  Every Part B builder asserts this on its inputs.
- **The in-sample OOS window** is the Phase 5 ledger's: 2016-01-21 → 2025-06-27. Every cell is
  scored on exactly this date range. The run asserts it.
- **Held fixed from Phase 5 in every cell:**
  - universe `daily_nse500`;
  - the ensemble, with walk-forward 504/63;
  - `decile_pct = 0.10`, `rebalance_buffer_mult = 2.0`, `hmm_states = 2`;
  - the Phase 4c cost regime: `aum_rupees = 1e7`, `max_participation = 0.10`, spread
    estimator `cs`, impact on, SLB eligibility on the `_slb` styles;
  - construction **`cost_swap`** (M2), with κ = 1.0, IC window 252 and minimum 126 as in
    `PHASE5_*`;
  - the label-overlap purge (§A.7), on in every fit.
- **Nothing here is tuned.** κ, windows, thresholds and the definitions in B.2 are not
  revisited after any Part B number is seen.

### B.2 · Feature sets

There are three sets. Each contains the one before it, so each layer's marginal value can be
read off.

- **F0 `p17`:** today's 17 price features, unchanged.
- **F1 `p17x`:** F0 plus the price and index features below.
- **F2 `p17xo`:** F1 plus the options and futures features below.

**Common rules.**
- Every value on date t uses only data at or before the close of t, the convention of the
  existing features.
- Per-stock features are rank-normalized per date to [-1, +1], as today.
- Prices are split- and bonus-adjusted (`adj_factor`) wherever two dates' prices meet in one
  formula.

**F1: price and index features.**

| feature | definition |
|---|---|
| `beta_252b` | OLS slope of the stock's daily log return on the Nifty 500's, over the trailing 252 bars ending at t (minimum 126) |
| `ivol_63b` | stdev of the residuals of the same regression over the trailing 63 bars (minimum 42) |
| `ret_overnight_21b` | sum over the trailing 21 bars of log(open_t / close_t−1) |
| `ret_intraday_21b` | sum over the trailing 21 bars of log(close_t / open_t) |
| `trade_size_21b` | median over the trailing 21 bars of turnover / trades, on bars with trades > 0 |
| `spread_cs_63b` | the existing Corwin-Schultz half-spread panel (`spread_daily_cs.parquet`, window ends t−1) |
| `sector_mom_63b` | the assigned sector index's 63-bar log return |
| `sector_rel_21b` | the stock's 21-bar log return minus its assigned sector index's |
| `sector_pe`, `sector_pb`, `sector_dy` | the assigned sector index's P/E, P/B and dividend yield (`-` read as missing, forward-filled per index) |
| `mkt_vix_z` | India VIX close, as an expanding z-score |
| `mkt_pe_z` | Nifty 50 P/E, as an expanding z-score |
| `mkt_ret_21b` | Nifty 50 21-bar log return |

- **Sector assignment.** The candidate set is the 11 sectoral indexes present continuously
  since 2013: Auto, Bank, Energy, Financial Services, FMCG, IT, Media, Metal, Pharma, PSU Bank
  and Realty. They were renamed from "CNX …" to "Nifty …" in November 2015, and a fixed name
  map joins the two. Each stock is assigned the index whose daily log returns it correlates
  with most over the trailing 252 bars (minimum 126). Assignment is redone on the universe
  rebalance dates (`universe.rebalance_dates`, every 63 bars) and held in between. It needs no
  constituent list and has no survivorship bias.
- **Market-wide features (`mkt_*`) are not rank-normalized.** They are identical across stocks
  on a date, so ranking would make them a constant. They enter raw, letting the trees learn
  interactions.
- **Expanding z-scores need at least 252 observations. Before that the value is 0**, the
  z-score's mean. India VIX starts on 2014-05-14 in the index archive, so the earliest
  training folds see a constant 0 for `mkt_vix_z`.

**F2: options and futures features.**

Sources: `data/bhavcopy/fo` (legacy, 2013-01 → 2024-06) and `data/bhavcopy/fo_udiff`
(2024-07 → 2025-06). Symbols map to panel tickers through the `shortable.py` mapping.

Rules applied before anything is computed:
- **Expiry.** Use the near-month expiry, rolling to the next month when fewer than 7 calendar
  days remain.
- **Traded contracts only** (contracts > 0). Untraded strikes carry NSE's theoretical
  settlement price.
- **IV by Black-76:**
  - the forward F is the same-expiry stock-futures close;
  - the rate is fixed at 6.5%;
  - the solver is vectorized bisection;
  - prices at or below intrinsic value are dropped.

| feature | definition |
|---|---|
| `opt_atm_iv` | IV interpolated linearly in strike between the two traded strikes bracketing F. At each strike, take the mean of call and put IV where both traded, otherwise whichever traded |
| `opt_ivspread` | mean of (call IV − put IV) over strikes within ±10% of F where both traded |
| `opt_skew` | IV of the traded put whose K/F is nearest 0.90 (within 0.85–0.95), minus `opt_atm_iv` |
| `opt_iv_chg_5b` | `opt_atm_iv`_t − `opt_atm_iv`_t−5 |
| `opt_iv_rv` | `opt_atm_iv` minus the 20-bar realized vol (stdev of daily log returns × √252) |
| `opt_volratio_5b` | 5-bar mean of the stock's total options traded value (all strikes and expiries) / its rupee turnover |
| `fut_basis` | near-month log(F/S) × 365 / days to expiry, minus the date's cross-sectional median |
| `fut_oi_chg_5b` | log(OI_t / OI_t−5), with stock-futures open interest summed over expiries |
| `opt_avail` | 1 if `opt_atm_iv` exists for the name on t, else 0 (raw, not ranked) |

- **Names without options.** About 60% of the universe has none on a given day. F2 features are
  rank-normalized among the names that have them. Every other name gets 0, the midpoint of the
  band, and `opt_avail` = 0 tells the trees.
- **Row-drop rule.** The rule changes from "every feature present" to "every *required*
  feature present". F0 and F1 are required. F2 is optional.
- **F0 stays byte-identical.** Under `p17`, `feature_creator` output and every wf_cache key are
  unchanged.

### B.3 · The grid and N

- **The cells.** {F0, F1, F2} × {5b traded daily, 21b rebalanced every 21 bars} ×
  {`long_only`, `long_short_slb`, `dynamic_tilt_slb`} = **18 cells**.
- **The 21b rebalance.**
  - Target weights are recomputed only on every 21st bar, counted from the first OOS bar, and
    held in between.
  - Forced exits still trade on any bar: a name leaving the universe, or a short losing its
    borrow.
  - The panic gate stays daily.
  - The M2 swap decision runs on rebalance bars only.
- **Strategy ids.** New fields are hashed only when not at their default, so every older id is
  unchanged:
  - `feature_set`: `_FX` for F1, `_FXO` for F2;
  - `rebalance_every`: `_R21`;
  - the purge keeps its `_PG` tag.
- **N.** F0/5b holds the leak-fixed versions of the 3 Phase 5 `cost_swap` 5b cells. A
  correction applied uniformly to already-searched cells counts 0, as the Phase 4c corrections
  did. F0/21b is new, because Phase 5's 21b cells traded daily, and so are all 12 F1 and F2
  cells. **N goes from 42 to 57.** This line enters `config.TRIAL_LEDGER` before the run:
  `("phase6 Part B, 3 feature sets x 2 cadences x 3 styles, cost_swap", 15, "18 cells; F0/5b
  are the 3 phase5 cost_swap 5b cells leak-fixed, a uniform correction counting 0")`.
- **The ledger** is `data/trial_database/phase6/phase6_dsr_matrix.parquet`.

### B.4 · Before the run: causality and quality gates

None of these gates computes a return.
- **Causality.** Perturb every raw input after a cut date D: prices, F&O files and index
  files. Every feature on dates ≤ D, and every Tier 1 prediction on dates ≤ D, must come back
  bit-identical. A planted canary (one feature shifted a day early) must be caught.
- **Quality.** Report, per feature, coverage by year and NaN rate. Also report the share of
  names with `opt_avail`, the median `opt_atm_iv` against India VIX, the IV solver failure
  rate, and the `fut_basis` distribution.
- **Regression.** With the new fields at their defaults, the Phase 5 ledger reproduces
  bit-for-bit (`scripts/regression_check.py`), and `scripts/no_lookahead_check.py` passes.

### B.5 · Criteria

- **Headline.** A cell passes if its DSR is > 0.95 at **N = 57** over the B.1 window. The
  scoring is that of every phase since 4c: compounded to daily, with the Lo variance-ratio
  correction at 21 days. SR\* at N = 57 is about 1.23.
- **Secondary, all reported.** Δ net Sharpe from F0 to F1 and from F1 to F2 in every cell: 12
  numbers. None is selected or promoted.
- **Read-only.** Per-feature IC by year for every new feature, using `per_feature_ic` and
  `by_year` from `scripts/phase6_decay.py`. Reported, never selected on.
- **Holdout, scored once.** The holdout runs from 2025-07-01 to the latest date available at
  that step.
  - First, the extended build must reproduce the frozen in-sample alpha ranks on dates ≤
    2025-06-30, with daily rank-correlation ≥ 0.999. Back-adjustment for new corporate actions
    rules out bit-identity.
  - Then all 18 cells are scored. A cell **survives** if all three hold on the holdout: net
    Sharpe > 0; net Sharpe ≥ ½ its in-sample net Sharpe; and ensemble IC against its own label
    with Newey-West t ≥ 2 (lag h − 1, as in §A.2).
  - The holdout can catch a collapse but cannot confirm a 1.8 Sharpe. Over about a year its
    standard error is about 0.9.
- **Part B succeeds** only for a cell that passes the headline **and** survives the holdout.

### B.6 · Stop rule

If no cell both passes and survives, the program stops adding features derived from price,
index or options data. The only next step left is genuinely new data (delivery, B.7), under
its own pre-registration.

### B.7 · What Part B does not license, and what is parked

- No κ, window, threshold or feature-definition change after a Part B number is seen.
- **A cell added after Part B's results are read is selection, not a test.** Two Part A
  candidates are parked, not dropped: a 1b horizon (where the reversal family held) and a
  longer-memory learner (1008-day window). Either can enter only through its own later
  pre-registration, with its own N.
- **Delivery data** is deferred. The known archive starts in 2020, which is too short for a
  504-day window plus a holdout. A read-only probe of NSE's older delivery files (the MTO
  archive) runs during the F2 build. If they reach 2013, delivery becomes a separately
  pre-registered F3.
  - **Probe result, 2026-10-14: they do.**
    - `archives.nseindia.com/archives/equities/mto/MTO_DDMMYYYY.DAT` returned files for
      2013-01-02, 2015-06-01, 2017-06-01, 2018-06-01 and 2020-06-02, in one format
      throughout: symbol, series, quantity traded, deliverable quantity and delivery %.
    - Only those five files were fetched.
    - F3 is feasible. It needs its own pre-registration and its own N, and it stays outside
      this grid.

### B.8 · Deviations and clarifications log

Implementation may deviate from B.2 only for data-quality reasons, never for a reason based
on returns. Every deviation, and every point where the implementation had to settle something
§B left open, is logged here with its date and reason before the run.

- **2026-10-14 · B.3, forced exits between rebalances (clarification).** §B.3 says forced exits
  trade on any bar but does not say what fills the slot. **The slot is refilled on the same
  bar** with the best-ranked eligible name in that bar's top decile not already held. Every
  construction already follows this rule for empty slots ("forced exits and empty slots
  always trade"). Nothing else trades between rebalances, and the leg keeps the size chosen
  on the rebalance bar.
  - **Why.** Found in the Stage 2 dry-run, on synthetic data, before any Part B number
    existed. Without a refill, a leg can drain between rebalances. An `_slb` book whose short
    leg has drained is net long, so it breaks the dollar-neutrality that B.4's checks require.
  - **The participation cap** is likewise set on the rebalance bar and held with the book.
    Re-capping on each day's ADV would re-weight a held book between rebalances.
- **2026-10-14 · B.2, F1 data quality (Stage 3, deviations).** Found while building, from the
  index archive alone. No return was computed against any target.
  - **Missing index sessions.** 12 panel sessions have no index file (2013-10-09 …
    2016-06-20). Index levels and valuations carry the previous session forward. The move
    lands on the next published session.
  - **Long P/E blanks are treated as losses, not forward-filled indefinitely.** NSE leaves P/E
    blank when a sector's aggregate earnings are not positive: PSU Bank 2016-05 → 2017-05 and
    2018-02 → 2021-01, Media 2020-08 → 2021-11 and 2022-09 → 2025-05. Forward-filling those
    would carry a P/E that is years stale.
    - A gap of up to 63 sessions is a publication pause and is carried. The FY-end gap of
      2021-03-31 → 2021-06-02 in Bank and PSU Bank dividend yield, and in Realty P/E, is one.
    - A P/E still blank after 63 sessions becomes +inf: above every finite P/E, which is the
      earnings-yield ordering.
    - This is causal: a run counts as a loss only once it has lasted past the limit.
    - It affects 7.1% of universe cells on average, and at most 30.2%.
  - **Nifty 50 P/E is spliced at 2021-03-31.** NSE switched index P/E from standalone to
    consolidated earnings that day. The Nifty 50 P/E fell 18% (40.43 → 33.20) on an index move
    under 1%: a net jump of 0.187 in logs, against at most 0.082 on any other session.
    - Values from the break on are divided by that day's P/E move net of the index move,
      which puts the new methodology on the old scale.
    - Values before the break are untouched. The scale is known at the break day's close.
    - *Revised in Stage 6, 2026-10-14, before any return was computed.* The first version
      scaled the history instead. That gives the same z-scores in exact arithmetic, but
      it let a later perturbation move the last bit of pre-break values, and the Stage 6
      causality check, which demands bit-identity, caught it.
    - Sector P/E is not spliced: it is only ever ranked within a date.
- **2026-10-14 · B.2, F1 readings (Stage 3, clarifications).**
  - **Own bars.** Windows count each stock's own trading bars, as the 17 existing features do.
    An index return paired with a stock return is taken over the stock's own bar span, so a
    return across a halt is compared with the market move over that halt.
  - **Complete windows.** A window with no stated minimum needs to be complete, as in
    `feature_creator._roll`.
  - **`ivol_63b`** is the residual stdev of the market model fitted on the trailing 63 bars
    itself (n − 2 degrees of freedom), not residuals from the 252-bar beta.
  - **Market z-scores** are computed on the archive's own sessions, then carried onto the panel
    calendar.
  - **Sector features** exist only on cells where the stock traded.
- **2026-10-14 · §A.7 purge tightened to own bars (Stage 6, a fix).**
  - **What the check found.** The Stage 6 causality check refits Tier 1 on real and on
    future-perturbed inputs. With the purge on, predictions before the cut still moved,
    by up to 0.0475.
  - **Why.** The purge dropped the last h *calendar* bars of each training block, but a
    label is log(close at the name's own bar t+h / close_t). Across a halt, the own bar
    t+h falls later than calendar bar t+h.
  - **Size.** 3,204 of 15.9M kept training rows (0.02%) had a 21b label ending inside the
    scored block. At the test cut, 27 labels reached past it, 21 of them from ESSDEE, a
    long halt.
  - **Fix.** `create_features(label_end_dates=True)` emits each label's own-bar
    realisation date (`tgt_end_{h}b`). The purge keeps a training row only if that date is
    before the scored block's first bar.
    - The calendar rule remains only as a fallback when the column is absent.
    - The wf_cache key gains an `ownbar` part, so no calendar-purged fit can be served.
    - Unpurged runs are unchanged.
  - **Consequence.** `_PG` now means the own-bar purge. No ledger has ever held a `_PG`
    cell. The §A.7 quick-check numbers were made under the calendar rule and are not
    re-run: the rows involved are 0.02%.
- **2026-10-14 · Stage 6 QC gate Q9 re-specified after its first run (a disclosed revision).**
  - §B.4 lists what the quality check reports but sets no thresholds. The gates in
    `scripts/phase6_feature_qc.py` were written in Stage 6.
  - **v1 failed.** It gated the yearly median *absolute* call−put IV spread at 0.02, and
    2018 came in at 0.0208.
  - **What the numbers show.** The *signed* yearly median runs −0.0002 to +0.0061: put-call
    parity holds and there is no bias, so the forward, the rate and the expiry choice are
    right. The absolute spread (2014 0.019 → 2025 0.011) is noise from non-synchronous last
    trades, and it shrinks as liquidity grows.
  - **v2** gates the bias, |signed| ≤ 0.01, and reports the absolute spread. The v1 log is
    kept at `data/trial_database/phase6/logs/stage6_feature_qc_v1_Q9abs.log`.
  - No return was read.
- **2026-10-14 · Observed in Stage 6, not changed (F0 is frozen): `liq_vol_var_20b` is not
  invariant to back-adjustment.**
  - The causality check plants a future 2:1 split. Of all 40 columns, only this one moves on
    dates before the cut. It is the variance of *volume*, and back-adjustment divides past
    volume by the split factor, so a later split rescales a name's past variance and
    therefore its rank.
  - In principle that is a small look-ahead in the original 17: a name's past feature value
    depends on a corporate action that had not happened yet.
  - It is reported, not fixed. F0 is the frozen baseline, and changing it would break the
    Phase 3–5 reproductions. Every new feature passes the same test.
- **2026-10-14 · Observed, not changed: what the sector assignment means for small caps.**
  - Names with an obvious sector are assigned correctly: HDFC Bank and ICICI Bank to Financial
    Services or Bank, SBI and PNB to PSU Bank, Infosys and TCS to IT, Sun Pharma to Pharma,
    Tata Steel to Metal, DLF to Realty, Maruti to Auto, HUL and ITC to FMCG, ONGC and Reliance
    to Energy, Zee to Media.
  - Mid and small caps in industries without one of the 11 indexes (chemicals, capital goods,
    cement and the like) take whichever index best tracks broad small-cap moves. That was
    Realty for 35% of the universe in mid-2016, and Energy and Media for 34% and 22% in 2025.
  - For those names the sector features measure "which part of the market it moves with". This
    is the §B rule working as written, so it is reported here and not changed.
- **2026-10-14 · B.2, F2 readings (Stage 4, clarifications).** These were settled from the
  archive's format alone. No return was computed against any target.
  - **Options traded value** is NSE's reported value. In both archive eras that is
    **notional**, (strike + premium) × quantity: legacy `VAL_INLAKH` × 1e5, UDiFF
    `TtlTrfVal`, checked against INFY rows. Premium-only value cannot be formed, because
    the legacy files carry no lot size.
    - `opt_volratio_5b` is 0 on a day the stock had F&O but no option traded.
    - It is NaN on a day the stock had no F&O.
  - **The forward F** is the near-month future's close only if that future traded that day.
    Otherwise the stock has no IV features that day.
  - **The IV filter** also drops prices at or above the no-arbitrage ceiling (discounted F
    for calls, discounted K for puts), and roots outside σ ∈ [0.001, 5].
  - **5- and 20-bar windows** (`opt_iv_chg_5b`, the realized vol in `opt_iv_rv`,
    `opt_volratio_5b`, `fut_oi_chg_5b`) count the stock's own bars, as in F1. A window
    with no stated minimum needs to be complete.
  - **`fut_basis`** uses the raw spot, the panel close divided by `adj_factor`, since
    futures trade on unadjusted prices. Its median is taken over the engine's tickers that
    have a basis that day. The raw annualized basis has a median of 6.5%, the rate it
    removes.
  - **Open interest** is divided by that day's `adj_factor`, as the panel does for volume,
    so a split does not register as a jump in OI.
