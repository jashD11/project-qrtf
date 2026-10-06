# Phase 6: why the signal degrades, and a better signal (idea note, not a plan)

Status: **idea only** (2026-10-05). Nothing is pre-registered and nothing has been run.
Turn this into a frozen plan, like `docs/phase5_plan.md` §7, before any result is looked at.

## Where Phase 5 left off

- Cost-aware construction (M2) works: +0.32 to +1.38 net Sharpe on every cell, and about half the
  trade drag. The best cell reaches net SR **0.96** (`dynamic_tilt_slb`/5b/`cost_swap`).
- Passing needs net SR **≈ 1.8**: SR\* 1.17 at N = 42, plus 1.65 × 0.35 standard error over
  9.3 years. That is solved from the gate's own PSR formula in `scripts/build_phase5_deck.py`.
- The binding constraint is the **signal**, not the trading. Even at zero cost the best Sharpe
  is 1.89.
- The skill decays. Rank-IC t-stat is 8–11 in 2016–20, 4.9 in 2021, 6.3–7.9 in 2022–23,
  2.3 in 2024 and **1.2 in 2025 H1** (`docs/phase4c_results.md` §4.2).

## Goal

A signal good enough for a viable strategy: **net Sharpe ≥ 1.8, realistically aiming above 2**.
Every new trial raises SR\*, so 1.8 is a floor, not a target. A rough guide: under the current
cost stack with M2, net ran about 0.9 below gross on the best cell. That suggests a gross Sharpe
of **~2.7+** before costs. This is a back-of-envelope figure, not a measurement.

## Part A: diagnose the post-2021 decay first (read-only, counts 0 in N)

Already ruled out:
- **Costs.** IC is measured before costs.
- **Survivorship bias.** The universe is point-in-time from bhavcopy and keyed on ISIN.
- **Full-sample liquidity artefact.** IC is flat across liquidity quintiles.
- **One learner breaking.** lgbm, xgb and rf all show IC ≈ 0.05 over the full sample.

Open hypotheses, each with the test that discriminates it:

| # | Hypothesis | Test (read-only, cached fits) | If true, then |
|---|---|---|---|
| H1 | The price-pattern edge was competed away (retail/algo growth after 2020) | IC by year for **each of the 17 features alone** | No model change helps. Need new information (Part B) |
| H2 | Regime shift: feature–return relationships flip between regimes | Per-feature IC **sign** by year vs market state | Regime-conditional model or features |
| H3 | Training setup: a 504-day window spanning COVID adapts slowly | Features still predict but the model doesn't; IC by fold vs training-window composition | Retrain window, frequency or regularisation |
| H4 | Universe drift: many 2021–24 IPOs with thin feature history | IC by year × listing age × liquidity quintile | Universe filter (minimum history) |
| H5 | Data: corporate-action adjustment or price-source noise grows recently | Daily closes vs bhavcopy closes by year | Fix the data. The decay may be partly artificial |

Also store per-year IC per learner (`tree_fit_diagnostics_*` keeps only full-sample aggregates).

Run the **per-feature test first**. It decides whether this is a model problem at all.

## Part B: find a better signal

Candidates, depending on what Part A shows:
- **NSE delivery data** (`DeliveryQty`, `DeliveryPct`). Already wired in `data_scraping.py` and
  excluded since Phase 2 only to avoid scraper rate limits (memory: `delivery-data-excluded`).
- **Fundamentals**, for a slower, less-crowded signal that suits the 21-day horizon where the
  M2 swap hurdle is lower.
- **Graph / co-movement features**, i.e. the TICC correlation-graph idea from the regime work
  (memory: `regime-model-direction`).
- **Beta / sector neutralisation**, still untested. It needs its own prior and its own N.
- Model changes only if Part A points to H2/H3.

## Rules carried forward

- Pre-register every choice before seeing results. Each new cell adds to `config.TRIAL_LEDGER`
  (N starts at 42).
- **M2 (`cost_swap`) is the default construction.** Phase 4c costs (real spread, impact, SLB
  short leg) stay on.
- Anything tuned after looking at 2021–25 is in-sample for the decay. Judge on data after
  **July 2025** where possible (forward test), not only on the existing window.
- Read-only diagnostics select nothing and are never promoted.
