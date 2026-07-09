# Phase 3 Design Requirements — Strategy Sweep Scope

Living record of the **sweep design** for the `PRODUCTION_ML` strategy grid: which
parameters the orchestrator iterates, which are frozen by prior, and *why*. Status
tags:
- **[SWEEP]** — a real axis of the strategy grid; the orchestrator iterates it.
- **[SENSITIVITY]** — swept only to *demonstrate robustness*, then fixed by prior;
  we report the whole band, we never pick its peak.
- **[FIXED]** — set once by economic/structural reasoning and frozen. Never swept.

---

## 0. First principle — degrees of freedom are a budget

Compute is not the scarce resource; **statistical credibility** is. A grid of `N`
cells is `N` implicit backtests, and reporting the best cell cherry-picks a winner
that is partly luck (multiple-testing / data-snooping bias). So every swept axis
costs twice: runtime (visible) and false-discovery risk (hidden, worse).

The discipline is therefore **not** "sweep everything I can afford to run." It is
"sweep only the structural forks I genuinely cannot reason my way out of, and fix
everything else by prior."

### The four-question test — should I sweep `X`?

1. **Impact** — does changing `X` materially move the result? (No → don't bother.)
2. **Ignorance** — do I have a defensible economic/structural reason to just *fix*
   it? (Yes → fix it, don't sweep.)
3. **Orthogonality** — is it a distinct axis, or collinear with something already
   swept? (Collinear → redundant.)
4. **Shape of the effect** — a genuine structural fork, or a smooth/monotone knob?
   Smooth knobs are overfit bait (you just fit the sample's drift); forks are worth
   exploring.

Sweep only where the answers are: **high impact, genuine ignorance, orthogonal, and
fork-shaped.**

### Reporting rule for anything swept

Report the **whole distribution** across the axis (robustness), never just the
`max` cell. A result that only survives at one grid point is a red flag, not a
finding. For **[SENSITIVITY]** axes this is the entire point of running them.

---

## 1. The grid — bucketed

### [SWEEP] — structural "which world am I in" forks

| Axis | `StrategyConfig` field | Values | Why it earns a sweep |
|---|---|---|---|
| Bar frequency | `frequency` | 15min / 30min / 60min / daily | *The* axis. Huge impact, no defensible prior, fully orthogonal to everything else. |
| Execution style | `execution_style` | long_only / long_short / dynamic_tilt | Not a hyperparameter — it's the core research question (does the short leg earn its keep?). It selects the strategy *family*. |
| HMM state count | `hmm_states` | 2 / 3 | A real structural fork: does a middle "normal" regime exist? Coarse — a 2-point sweep, no more. |

**Optional [SWEEP]:** `model_choice` {ensemble, lgbm, xgb, rf} — but treat as a
robustness check, not a search. `ensemble` is a fine default.

### [SENSITIVITY] — sweep to prove robustness, then fix by prior

| Knob | Field / config | Band to scan | Prior to fix at |
|---|---|---|---|
| De-risk gate | `panic_threshold` (`ML_CONFIG.regime`) | 0.80 / 0.85 / 0.90 / 0.95 | Policy: "de-risk ~15% of days" → 0.85. Report the frequency/return curve; do **not** pick the peak. See phase2 §3. |
| Momentum lookback | `lookback_period` | 2–3 points around default | Feature-horizon judgment. |
| Book concentration | `top_n` / `bottom_n` | 2–3 points | = risk appetite (concentration vs breadth). |

### [FIXED] — set by prior, never swept (smooth / overfit bait / defensible default)

| Knob | Why frozen |
|---|---|
| **130/30 vs 120/20 vs 140/40** leverage split | A leverage dial with a smooth, monotone effect on exposure. In a bull sample 140/40 always "wins"; in a bear 120/20 "wins" — optimizing it just fits the sample's directional drift. Textbook overfitting. Pick from the net/gross exposure budget and freeze. |
| Tree hyperparameters (`learning_rate`, `n_estimators`, `max_depth`, `num_leaves`, `min_samples_split`) | Enormous combinatorial space, low structural interest, and financial signal-to-noise is so low that a grid search fits noise. Trees are robust within a sane range. Defaults in `_LGBMConfig`/`_XGBConfig`/`_RFConfig`, left alone. |
| `transmat_stickiness` | Persistence/turnover knob, not a frequency control. Fix by turnover appetite. |
| `vol_window`, `corr_window`, `zscore_min_periods`, `refit_every`, `train_days`, `predict_days` | Plumbing/estimation constants. Reason once, freeze. |

---

## 2. The practical ceiling

Keep the *real* grid to **2–3 axes**. The committed **[SWEEP]** set is already:

```
frequency (4) × execution_style (3) × hmm_states (2) = 24 cells
```

At 24 cells, multiplied by the single-thread OpenMP tree tax (daily ~3–10 min;
60-min ~30–60 min; 15-min ~1–3 hr per fit), that is the budget. Adding a 4th
4-value axis → 96 cells, and the odds that the best cell is noise rise sharply.

**Mental model:** the degrees of freedom are a budget you spend on multiple
testing. `130/30 → 120/20` and tree grids *feel* like free extra thoroughness, but
they are the expensive kind — they buy a better-looking backtest and a
worse-performing strategy. Spend the budget on the two or three forks where the
*structure* is genuinely uncertain; fix everything else by economic prior.

---

## 3. Requirements for the orchestrator (Phase 3 build)

- **R1 — Grid source of truth.** The **[SWEEP]** axes are enumerated from
  `StrategyConfig` fields; the orchestrator takes the Cartesian product of the
  committed lists (`frequency`, `execution_style`, `hmm_states`) and nothing else
  by default. Adding an axis to the grid is a deliberate edit, not an accident of
  config.
- **R2 — Priors live in config, not the loop.** Every **[FIXED]** value is a named
  default in `config.py` (`ML_CONFIG` / `StrategyConfig` defaults) with a one-line
  rationale comment, so "why isn't this swept?" is answerable from the code.
- **R3 — Sensitivity runs are opt-in and separate.** **[SENSITIVITY]** scans are a
  distinct, explicitly-invoked mode (e.g. a `--sensitivity panic_threshold` run),
  not folded into the headline grid — so they never inflate the main grid's
  multiple-testing count.
- **R4 — Report the distribution.** Ledger/summary output presents metrics *across*
  each swept axis (per-frequency, per-style, per-state), never a single "best
  strategy" scalar. The `strategy_id` already encodes every axis, so each cell is a
  distinct ledger column (see phase1 `tier4_dsr`).
- **R5 — Freeze before test.** The threshold/prior choices are fixed *before* the
  out-of-sample evaluation window is scored; no knob is re-tuned after seeing test
  results. (Ties back to the look-ahead / data-snooping discussion in phase2 §3.)

---

## 4. Open items

- **[OPEN]** Confirm the frozen leverage split (130/30) as the single dynamic_tilt
  default vs. exposing net/gross exposure as the *economic* parameter it derives
  from.
- **[OPEN]** Decide whether `model_choice` joins the headline **[SWEEP]** grid
  (×4 → 96 cells) or stays a one-off robustness check.
- **[OPEN]** Wire the orchestrator itself (still deferred — see phase2 §4): the grid
  iteration, mask→returns→ledger back half, and the sensitivity-run entry point all
  land here.
