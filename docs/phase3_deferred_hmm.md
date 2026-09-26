# Phase 3 Deferred Work — State-Conditional Execution & the HMM Axis

> **Later finding, recorded here so this document is not read as the last word.**
> This file frames the unused HMM states as *deferred wiring* — something to connect later.
> [`phase5_plan.md`](phase5_plan.md) §1.1 measured the stronger version: `RegimeResult.states`,
> `.probs` and `.stress` have **zero consumers anywhere in the repo**. Only `.panic` is read
> (`run_pipeline_ml.py:349`), and `panic` never touches the HMM — it is
> `mean(z_log_realized_vol, z_avg_corr)` thresholded at an expanding 85th percentile
> (`tier2_regime.py:421-424`). So the fitted Gaussian HMM is not awaiting wiring; it is
> decorative as currently built, and any estimator upgrade is a no-op until Tier 3 is
> rewired. §1.2 there also finds the gate correlates 0.907 with a plain 21-day realised-vol
> z-score.


Living record of PRODUCTION_ML execution ideas that are **designed-for but
deliberately not built yet**, so a future session can pick them up without
re-deriving the reasoning. Everything here was consciously deferred while
shipping the first working execution tier + sweep orchestrator.

---

## 1. Why `hmm_states` is not (currently) a sweep axis

The committed headline grid is `frequency × execution_style` (12 cells).
`hmm_states` was dropped from `[SWEEP]` — **not** because the fork is
uninteresting, but because it has **zero effect on returns today**:

- Tier 3 (`src/production_ml/tier3_execution.py`) gates only on the Tier 2
  `panic` flag.
- `panic` is a causal trailing percentile on a **continuous stress score**
  (mean of z-vol and z-herding) — it is computed independently of the HMM's
  state count. See `tier2_regime.RegimeDetector._severity_gate`.
- The HMM `states` / `probs` (the 0..n-1 vol/herding taxonomy) are **not read**
  by any weight logic.

So a 2-state vs 3-state HMM produces **return-identical** portfolios today —
the two cells would differ only in ledger column name. Sweeping it would be
exactly the "axis with no impact" that `docs/phase3_design_requirements.md` §0
tells us not to sweep (it inflates the multiple-testing count for nothing).
`hmm_states` is therefore frozen at `config.HEADLINE_HMM_STATES = 2` until one
of the extensions below gives it real teeth.

---

## 2. Extension A — State-conditional sizing (make `hmm_states` matter)

**Idea.** Let the HMM `states` (not just the binary `panic` gate) modulate
leverage, so a 3-state model's *middle* regime gets an intermediate tilt
distinct from both Calm and Panic.

**Sketch.**
- 2-state: Calm → 130/30; the upper state (or `panic`) → dollar-neutral. (This
  is roughly today's `dynamic_tilt`, but conditioned on `states` instead of the
  percentile gate.)
- 3-state: Calm → 130/30; Middle → an intermediate exposure (e.g. 115/15 or a
  reduced gross); Panic → dollar-neutral / cash.
- Implementation would broadcast the per-day `states` label into a
  `(days × stocks)` scale matrix — the same `np.outer` pattern already used for
  the `panic` flag in `build_weight_matrix`, but with per-state scale lookups
  instead of a boolean.

**Open design questions to resolve first (do NOT hand-wave):**
1. **What does the "middle" leverage number mean, and how is it chosen?** A 3rd
   leverage tier is a *smooth/monotone knob* — exactly the overfit-bait class
   the phase3 doc freezes by prior, never optimizes. It must be set from an
   exposure-budget argument and frozen **before** OOS scoring (R5), not tuned to
   maximize backtest Sharpe.
2. **Does the percentile `panic` gate coexist with state-conditional sizing, or
   replace it?** They answer different questions (a frequency-controlled tail
   flag vs a persistent taxonomy). Decide whether Panic overrides the state tilt
   or composes with it.
3. **Intraday.** `states` would need the same t-1 `broadcast_to_intraday`
   treatment `panic` already gets.

---

## 3. Extension B — Regime as a tree feature (the "feature" half)

Already foreshadowed across the codebase (`tier2_regime` emits `probs` for
exactly this). Inject the per-state posteriors as additional columns into the
Tier 1 feature matrix, behind a toggle, so the trees can *learn* regime-
conditional alpha rather than only having regime applied as a post-hoc execution
overlay. This is the other, complementary way `hmm_states` becomes load-bearing:
a 3-state posterior vector is a richer feature than a 2-state one.

**Guardrail:** the posteriors are already causal (walk-forward forward-filtering,
verified by the truncation test in `tier2_regime.__main__`), so wiring them as
features preserves the no-look-ahead invariant — *provided* they are joined on
matching bar timestamps with the same t-1 lag discipline used elsewhere.

---

## 4. Bias note — tuning the HMM later (temporal vs selection bias)

Requested during planning: how does eventually "tuning the HMM properly" affect
look-ahead / selection bias?

- **Temporal (look-ahead) bias — already handled.** The regime fit is causal by
  construction: trailing-window refits, forward-filtering only (no backward
  smoothing), causal expanding z-scores, and a causal expanding-quantile gate.
  `tier2_regime.__main__` proves it with a truncation test (past labels are
  unchanged when the future is chopped off). Adding states, tuning
  `transmat_stickiness`, or changing `panic_threshold` does **not** by itself
  reintroduce temporal leakage as long as these mechanisms are preserved.

- **Selection (data-snooping) bias — the real risk, and it moves up a level.**
  If a future pass picks `n_states`, `panic_threshold`, `transmat_stickiness`,
  or a middle-tilt magnitude by looking at **which value produced the best
  backtested Sharpe**, that is the same multiple-testing / data-snooping problem
  `phase3_design_requirements.md` §0 warns about — just relocated from "which
  strategy cell" to "which model hyperparameter." The discipline carries over
  unchanged:
  - Choose model-structure knobs (e.g. `n_states`) with a **causal, non-trading
    criterion** if compared at all — held-out log-likelihood / AIC computed
    *within* the walk-forward fit — never by realized strategy returns.
  - Keep every execution-facing knob (`panic_threshold`, any future tilt
    magnitude, the 130/30 split) a **[FIXED] prior frozen before OOS scoring**
    (R5). Report the whole `[SENSITIVITY]` band, never pick its peak.
  - Treat `hmm_states` the day it enters `[SWEEP]` as a *2-point structural
    fork* (does a middle regime exist?), reported as a distribution — not a knob
    to maximize.

Short version: the walk-forward machinery keeps *training* honest; the
`[SWEEP]`/`[SENSITIVITY]`/`[FIXED]` discipline is what keeps *model selection*
honest. Both must hold before `hmm_states` earns a place in the headline grid.

---

## 5. Also parked: `lookback_period` as a Phase-2 sensitivity axis

`docs/phase3_design_requirements.md` lists `lookback_period` as a
`[SENSITIVITY]` axis, inherited from the Phase-1 GKX momentum field. It has **no
direct analog** in the tree pipeline — the trees consume the full 19-feature
panel, not a single lookback window — so it is intentionally **absent** from
`config.SENSITIVITY_BANDS`. If revisited, the closest existing horizon knob is
`tier1_trees`' `target_col` (`tgt_fwd_logret_1b` vs `tgt_fwd_logret_5b`); decide
deliberately whether to reinterpret it that way or drop it for Phase 2.
