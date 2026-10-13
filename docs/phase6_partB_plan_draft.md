# Phase 6 Part B — execution plan (approved 2026-10-14; pre-registered in `phase6_plan.md` §B)

## Handoff — read this first (state as of 2026-10-07)

**Done and committed**
- Part A, pre-registration: `docs/phase6_plan.md` §A.
- Part A, script: `scripts/phase6_decay.py`.
- Part A, results: `docs/phase6_decay_results.md`. No rule fired; H1 missed by one feature;
  the low-vol family decayed.
- The Tier 1 label-overlap purge: `docs/phase6_plan.md` §A.7. It is opt-in, regression
  bit-identical, and worth 3% of IC at 5b and 13% at 21b.

**Not done**
- Stages 2–9. Stages 0 and 1 are complete: §B is pre-registered and committed (2026-10-14).
- The purged ensemble re-score of the Phase 5 grid was started and then paused. Stage 7's F0
  fits replace it.
- This plan is a draft. The user reviewed it but interrupted approval to deal with machine
  constraints. Confirm it with the user before Stage 1.

**Decisions the user made**
- Horizons: **5b traded daily** and **21b rebalanced monthly**.
- Training: **keep the 504/63 rolling window**.
- Financial statements: **infeasible**, out of scope.
- The holdout (data after 2025-06-30) stays **sealed** until Stage 8.
- **M2 (`cost_swap`)** is the construction.
- The Phase 4c cost regime stays on.
- The purge is on for every Phase 6 fit.
- (2026-10-14) Part A's 1b and longer-memory cells stay **out** of the grid for now and are
  parked under "Deferred, not dropped".
- (2026-10-14) How to power Stage 7 (charger vs lower `rf.n_jobs`) is **decided at Stage 7**.
- (2026-10-14) The user **approved** the draft. Stage 1 (§B in `docs/phase6_plan.md`) is
  committed. §B is now the binding text. Where it is more specific than this plan, §B wins.

**Machine constraints (measured 2026-10-07; MacBook Pro M1 Pro, 8 cores, 16 GB)**
- Measured: whole-Mac draw is about **32 W** with `rf.n_jobs = 6`, against about 5.5 W idle.
  The user's charger is **20 W (~19 W delivered)**.
- Consequence: Stage 7 (6 cold ensemble fits, about 10–14 h) **drains the battery even when
  run alone**. Either use a 67 W or 96 W charger, or lower `config._RFConfig.n_jobs` to 2–3.
  The results are invariant to `n_jobs`; Random Forest becomes about 2× slower, and power is
  roughly break-even.
- The user also runs an unrelated overnight MATLAB batch (about 1–2 cores, 12–16 W, with
  timing-sensitive runs in its last ~1.4 h). **Do not run Stage 7 at the same time.**
  Stages 1–6 (code, feature builds, dry-runs) are light enough to overlap, but stop
  CPU-heavy steps before the MATLAB timing runs.

**Resume:** start at Stage 2 (monthly rebalancing in Tier 3), then Stages 3–6, then Stage 7
on a night with a suitable charger or with lowered `rf.n_jobs`.

## Context

Part A (`docs/phase6_decay_results.md`) found that the 17 price features are losing their
information, the low-volatility family above all. A model fitted inside each year, with
knowledge of that year's future, finds no IC in 2024–25. Retuning the model cannot fix
that, so Part B adds **new information** and a **cheaper way to trade** it. Company
financial statements are out of scope (judged infeasible to obtain).

What is already on disk and unused:
- F&O option chains and futures for 2013 → June 2025, with liquid near-the-money options
  on 115–220 stocks a day;
- index files carrying sector P/E, P/B and dividend yield, plus India VIX;
- trade counts, open prices and spread estimates in the price panel.

A training-overlap leak was found and fixed opt-in (`--purge-label-overlap`). It was worth
about 3% of IC at 5b and about 13% at 21b. Every Part B fit uses the fix.

Decisions agreed with the user:
- Horizons: **5-day with daily M2 trading**, and **21-day with monthly rebalancing**.
- Training: **keep the 504/63 rolling window**, so any gain is attributable to the
  features.
- Financial statements: no.
- Holdout: the post-June-2025 data stays sealed until the final step.

## Stage 0 — commit the work so far: DONE (2026-10-07)

Part A, the purge fix with §A.7, and this draft were committed. Nothing is left uncommitted except gitignored data and `CLAUDE.md`.

## Stage 1 — pre-register Part B (`docs/phase6_plan.md` §B): DONE (2026-10-14)

The §B text freezes six things before anything is built.

**1. Feature sets** (nested, so each set's marginal value can be read off):
- **F0 `p17`**: today's 17 features.
- **F1 `p17x`**: F0 plus the price and index additions (Stage 3).
- **F2 `p17xo`**: F1 plus the options and futures additions (Stage 4).

**2. The grid.** F0, F1, F2 × {5b traded daily, 21b rebalanced every 21 bars} ×
{`long_only`, `long_short_slb`, `dynamic_tilt_slb`}:
- 18 cells, all with construction `cost_swap` (M2);
- the Phase 4c cost regime;
- purge on.

**3. N.** F0/5b holds the leak-fixed versions of the 3 Phase 5 `cost_swap` 5b cells. A
correction applied uniformly to cells already searched counts 0, as the Phase 4c corrections
did. The other **15 cells are new, so N goes from 42 to 57** (SR\* about 1.23). This enters
`config.TRIAL_LEDGER` before the run.

**4. Exact feature definitions.** Listed under Stages 3–4. Implementation may deviate from
them only for data-quality reasons, every deviation is logged in §B, and none may be based on
returns.

**5. Criteria.**
- **Headline:** any cell with DSR > 0.95 at N = 57 on the in-sample OOS window
  (2016-01 → 2025-06).
- **Secondary:** Δ net Sharpe from F0 to F1 and from F1 to F2 in every cell, all reported.
- **Holdout (Stage 8):** a cell survives if its holdout net Sharpe is > 0, is at least half
  its in-sample net Sharpe, and its holdout IC t-stat is ≥ 2.
- The holdout can catch a collapse, but it cannot confirm a 1.8 Sharpe: its standard error
  is about 0.9.

**6. Stop rule.** If no cell passes and survives, stop adding features derived from price,
index or options data. The only next step left is genuinely new data (delivery), under its
own pre-registration.

Allowed before the freeze is committed: coverage, NaN and distribution checks on new
features. **Not allowed: any computation that touches returns** (IC, Sharpe).

## Stage 2 — monthly rebalancing in Tier 3 (opt-in)

- **`StrategyConfig.rebalance_every: int = 1`.** It is hashed into `strategy_id` only when it
  is not 1 (tag `_R21`), the same pattern as `construction` and `purge_label_overlap` in
  `config.py`.
- **`tier3_execution.py`.**
  - When `rebalance_every > 1`, target weights are recomputed only on rebalance bars, which
    are every Nth bar from the first OOS bar. Weights are held constant in between, so
    `weights.diff()` charges nothing between rebalances.
  - Forced exits still happen on any bar: a name leaving the universe, or losing its borrow
    on the short leg.
  - The panic gate stays daily.
  - The M2 swap decision (`cost_aware.py`) runs on rebalance bars only.
- **Verification.**
  - `rebalance_every=1` must be bit-identical: run `regression_check.py` against the
    Phase 5 ledger.
  - New dry-run checks:
    - turnover is zero between rebalance bars;
    - a forced exit still trades between rebalances;
    - the book stays dollar-neutral for `_slb`.

## Stage 3 — F1 features: price and index additions (`src/phase6_features/`)

New package, modelled on `src/phase4_data/`: one builder per family, each writing a
`(date × ticker)` Parquet under `data/phase6/`, each with `--build` and a dry-run. **Every
value on date t uses data at or before the close of t**, the same convention as the
existing features.

| feature | definition | source and reuse |
|---|---|---|
| `beta_252b` | 252-day OLS beta against NIFTY 500 daily log returns | index files (`tier2_regime._load_index_daily` pattern), price panel |
| `ivol_63b` | standard deviation of the 63-day market-model residual | same |
| `ret_overnight_21b`, `ret_intraday_21b` | 21-day sum of log(open/prev close) and of log(close/open) | `panel_daily.parquet` open/close |
| `trade_size_21b` | 21-day median of turnover/trades | `panel_daily` `turnover`, `trades` |
| `spread_cs_63b` | Corwin-Schultz half-spread (already causal at t−1) | `src/phase4_data/spread.py` panel |
| `sector_mom_63b`, `sector_rel_21b` | the assigned sector index's 63-day return, and the stock's 21-day return minus the sector's | sector assignment below |
| `sector_pe`, `sector_pb`, `sector_dy` | the assigned sector index's valuation ratios | index-file P/E, P/B, Div Yield columns |
| `mkt_vix_z`, `mkt_pe_z`, `mkt_ret_21b` | India VIX, NIFTY P/E (both as expanding z-scores) and the NIFTY 21-day return | index files |

**Sector assignment.** Each stock is assigned to the NSE sectoral index (about 15 of them:
Bank, IT, Pharma, FMCG, Auto, Metal, …) whose daily returns it correlates with most over the
trailing 252 days. It is reassigned quarterly on `universe.rebalance_dates`. This is causal,
needs no constituent lists, and has no survivorship bias.

**Market-wide features are not rank-normalized.** `mkt_*` is the same for every stock on a
date, so ranking would flatten it to a constant. It enters the model raw, letting the trees
learn interactions, which is the GKX role of macro predictors. `feature_creator` gets a
separate unranked family for it.

## Stage 4 — F2 features: options and futures (`src/phase6_features/options.py`, `futures.py`)

Both builders read `data/bhavcopy/fo` (legacy format) and `fo_udiff` (UDiFF format).
Symbols are mapped to panel tickers with the mapping already in
`src/phase4_data/shortable.py` (99.7% match).

**Rules applied before computing anything:**
- Use the near-month expiry, rolling to the next month when fewer than 7 calendar days
  remain.
- Use **traded contracts only** (contracts > 0), because untraded strikes carry NSE's
  theoretical settlement price.
- Compute IV with **Black-76**:
  - the forward is the same-expiry stock-futures close;
  - the options are European;
  - the rate is fixed at 6.5% (rate effects largely cancel cross-sectionally);
  - the solver is vectorized bisection;
  - prices at or below intrinsic value are dropped.

| feature | definition |
|---|---|
| `opt_ivspread` | mean of (call IV − put IV) across strikes where both traded, within ±10% of the forward |
| `opt_skew` | IV of the traded put nearest K/F = 0.90 (within 0.85–0.95) minus the ATM call IV |
| `opt_atm_iv` | ATM IV, interpolated linearly in strike between the two traded strikes around the forward |
| `opt_iv_chg_5b` | 5-day change in `opt_atm_iv` |
| `opt_iv_rv` | `opt_atm_iv` minus the annualized 20-day realized volatility |
| `opt_volratio_5b` | 5-day mean of options traded value divided by stock turnover |
| `fut_basis` | annualized log(F/S) for the near month, minus the date's cross-sectional median (which removes the rate) |
| `fut_oi_chg_5b` | 5-day log change in total stock-futures open interest |
| `opt_avail` | 1 if the name has a usable chain that day, else 0 |

**Names without options.** About 60% of the universe has no options. In
`feature_creator.py`, F2 is rank-normalized among the names that have it. Elsewhere it is
filled with **0**, the midpoint of the normalized band, and `opt_avail` = 0 tells the trees.
The row-drop rule changes from "every feature present" to "every *required* feature present",
where F2 is optional. Without this change the F2 set would delete two-thirds of the
universe.

## Stage 5 — wiring

- **`feature_creator.py`.**
  - A feature-set registry: `p17` is today's list, and `p17x` and `p17xo` extend it.
  - `create_features(..., feature_set="p17")` merges the extra panels before normalization.
  - Default `p17` stays byte-identical, so cache digests are unchanged.
  - `present_feature_columns` reads the registry.
  - New feature columns change the wf_cache panel digest automatically
    (`wf_cache.panel_digest` hashes the columns the fit reads).
- **`config.py`.**
  - New `StrategyConfig.feature_set` field, hashed only when it is not `p17` (tags `_FX` and
    `_FXO`).
  - New `PHASE6_*` grid constants.
  - The `TRIAL_LEDGER` line (+15).
- **`run_pipeline_ml.py`.**
  - A `--phase6` mode, following `apply_phase5_regime`:
    - Phase 4c cost overrides;
    - purge forced on;
    - construction `cost_swap`;
    - the grid from `PHASE6_*`;
    - ledger `data/trial_database/phase6/phase6_dsr_matrix.parquet`.
  - `compute_signals` gains `feature_set`.
  - The Tier 1 outer loop runs over (feature_set, target).
  - Rebalance cadence is execution-time: it fans out over the shared fit, like construction
    does.

## Stage 6 — causality and quality gates (before any return is looked at)

- **New `scripts/phase6_causality.py`.** It perturbs every raw input after a cut date D:
  prices, F&O files, index files. Every feature on dates ≤ D must come back bit-identical. A
  canary (one feature shifted a day early) must be caught. This covers features and Tier 1,
  which `no_lookahead_check.py` (Tier 3 only) never did.
- **New `scripts/phase6_feature_qc.py`.** Per feature: coverage by year, NaN rate, the share
  of the F&O universe with `opt_avail`, IV sanity (median ATM IV against India VIX, the
  share of solver failures), basis distribution. **No returns.**
- **Regression.** The default config reproduces the Phase 5 ledger bit-for-bit, and
  `no_lookahead_check.py` passes.

## Stage 7 — the retrain and the grid (overnight)

- `python run_pipeline_ml.py --phase6` runs 6 cold ensemble fits: 3 feature sets × 2 targets.
  The F0 fits are also the leak-fixed baseline that was put aside earlier.
- Expect roughly 10–14 hours, with the laptop running hot through Random Forest.
  `rf.n_jobs` does not change results, so it can be lowered for less heat at the cost of
  time.
- Afterwards:
  - DSR gate at N = 57;
  - Δ tables from F0 to F1 and from F1 to F2;
  - **read-only** per-feature IC of the new features, using `scripts/phase6_decay.py`
    functions (`per_feature_ic`, `by_year`). This is reported, never selected on.

## Stage 8 — holdout (once)

1. Download equity, index and F&O data from 2025-07-01 to the latest date with
   `bhavcopy_download.py --start/--end`. The F&O kinds take the longest.
2. Rebuild the panel, universe, spread, shortable mask and the Phase 6 feature panels.
3. **Regression first.** On dates up to 2025-06-30, the extended build must reproduce the
   frozen alpha ranks (daily rank-correlation ≥ 0.999). New corporate actions back-adjust
   history, so bit-identity isn't expected, but the ranks must not move.
4. Score the frozen 18 cells on the holdout, once, against the §B criteria.

## Stage 9 — write-up

- New `docs/phase6_results.md`.
- Update `CLAUDE.md` and the memory notes.
- Commit the code, docs and config; data stays uncommitted.

## Deferred, not dropped

**Delivery data.** The known archive covers only 2020 onwards. That is too short to train a
504-day window and still test on the holdout. Stage 4 includes a read-only probe of NSE's
older delivery files (the MTO archive). If they reach 2013, delivery becomes a separately
pre-registered F3.

**Part A's two candidate cells** (`phase6_decay_results.md` §3). On 2026-10-14 the user chose
not to add them to the 18-cell grid yet, but to keep them for later:
- **A 1b horizon.** It matches the reversal family that held up: the frozen composite holds at
  IC ≈ 0.055 through 2025. The risk is 1b turnover, which was fatal on costs before M2.
- **A longer-memory or more heavily regularised learner**, e.g. a 1008-day training window.
  It beat 504 recently (0.023 vs 0.013). It cuts against the "keep 504/63" decision, so it
  would also blur whether any gain came from the features.

Both ideas come from the 2021–25 data, so they are in-sample. If either is taken up, it must be
added to §B **before Stage 7 produces any number**, with its cells counted in N. Otherwise it
needs its own later pre-registration. Adding a cell after Part B's results are seen is the
selection bias the DSR gate exists to correct.

## Verification summary

| Gate | Pass condition |
|---|---|
| Regression | Default config reproduces the Phase 5 ledger bit-for-bit after every engine change (Stages 2 and 5) |
| Dry-runs | Every new module's dry-run passes, with planted-effect checks where an answer is known |
| Causality | Future-perturbation test plus canary over features and Tier 1 |
| Quality | Feature QC passes before the §B freeze touches any return |
| Accounting | N = 57 recorded before the run; ledger and strategy ids are distinct (`_FX`, `_FXO`, `_R21`, `_PG`) |
| Holdout | In-sample ranks reproduce before the holdout is scored |
