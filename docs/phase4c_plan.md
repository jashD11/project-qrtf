# Phase 4c — make the costs real, make the short leg feasible, make N honest

**Status: PLAN, written 2026-08-06.** Nothing here has been built. Supersedes the
deferred list in [`phase4_results.md`](phase4_results.md) §5.

---

## Context — why this phase exists

Phase 4b ran the tree stack on the 500-name point-in-time panel and produced the
project's first DSR passes: `long_short` 1.78 SR / DSR 1.000 and `dynamic_tilt` 1.37 /
0.998, against `long_only` at 0.86 / 0.916. Taken at face value that reverses Phase 3's
negative result.

It should not be taken at face value, for four reasons that this phase exists to
resolve:

1. **The model charges zero spread and zero market impact.** `slippage_bps = 0.0` and
   `impact_bps()` is never called from the execution path. With 0.45–0.82 of the book
   turning over *per day*, the passing cells have **under 7 bps** of headroom on a cost
   input the backtest sets to zero. Measured off this repo's own panel with two standard
   OHLC spread estimators, the in-universe median half-spread is **6.1 bps
   (Corwin-Schultz, optimistic) to 31.7 bps (Abdi-Ranaldo, pessimistic)** — the
   optimistic estimate alone consumes the entire headroom.
2. **Both passing cells depend on a short leg that cannot be built.** Measured from 38
   quarterly NSE F&O bhavcopy snapshots: only **171 of 500** universe names are
   shortable (34%), sitting at median liquidity rank **105 of 500**. See §2 — the
   single-stock-futures route is arithmetically impossible at Rs 1 cr and is dropped.
3. **The Sharpes are inflated by autocorrelation.** The DSR gate assumes i.i.d. daily
   returns. Measured lag-1 autocorrelation is +0.10 / +0.14 / +0.09; the Lo (2002)
   correction cuts `long_short` from **1.78 to ~1.34**.
4. **N = 3 is not the honest trial count.** The project has searched ~24 configurations
   to reach this point. At N = 24–30 the deflation benchmark roughly doubles and only
   `long_short` survives even *before* any cost correction — see §4.

Each of the first three is a correction that moves monotonically **against** the
strategy. The expected outcome of this phase is therefore that Phase 4b's passes do not
survive. That is the point: the phase is built to find out honestly, and to give a real
edge — if one exists — its best remaining chance via the one lever that helps rather
than hurts (turnover, §3).

**On method.** None of this is look-ahead bias; the walk-forward, causal-filtering and
`shift(-1)` discipline is untouched. It *is* selection bias, one level up
([`phase3_deferred_hmm.md`](phase3_deferred_hmm.md) §4), and §4 is how it gets paid for.
The lot-size and spread findings are outcome-independent measurements — they would read
the same whatever the backtest had shown — so they carry no selection cost.

---

## 0. Prerequisites — engine hygiene (build first)

These are pure engineering: no research content, no selection cost. E1 in particular
gates everything else, because every variant below currently costs a fresh ~80-minute
walk-forward fit.

**E1 — Cache the Tier 1 walk-forward to disk.** `tier1_trees.run_walk_forward` returns
`WalkForwardResult` purely in memory and nothing persists it; `compute_signals` in
`run_pipeline_ml.py` refits from scratch on every invocation. Add a disk cache keyed on
`(frequency, target_col, decile_pct, TRAIN_WINDOW, PREDICT_WINDOW, feature-panel hash)`
storing `alpha_scores` / `long_mask` / `short_mask` / `ic_diagnostics` as Parquet under
`data/trial_database/wf_cache/`. Load-on-hit, fit-and-write on miss. This turns each
Track A/B variant from 80 minutes into seconds and is the single highest-leverage change
in the phase.

**E2 — Capture gross and net in one run.** `execute_ml_strategy` returns only whichever
`ML_CONFIG.cost.apply_costs` selects, so Phase 4b never recorded gross Sharpe and
recovering it needs a second full run. Return both series (or call twice against a
`dataclasses.replace(cost, apply_costs=False)`) and log `<strategy_id>__gross` alongside
the net column.

**E3 — Fix the two `run_sensitivity` bugs** (`run_pipeline_ml.py` ~305–371) before any
scan touches this panel: it omits `terminal_returns` (so delisting exits silently vanish)
and skips `_to_daily` (so any intraday frequency lands an all-NaN ledger column). Both
are latent today only because `SENSITIVITY_BASE["frequency"] == "daily"`.

**E4 — Parameterize the ledger path.** `PRODUCTION_LEDGER` is hard-coded
(`run_pipeline_ml.py:63`), which is why Phase 4b required manually moving the Phase 3
ledger aside. Add `--ledger`.

**E5 — Dividend guard from data.** Phase 4b's `long_only` verdict turned on an assumed
flat 1.3%/yr (DSR 0.916 → 0.9496 against a 0.95 threshold). Replace with the `Div Yield`
column already present in `data/bhavcopy/index/ind_close_all_*.csv`, applied as an
index-level total-return adjustment, and state it as a stated assumption rather than a
silent one.

---

## 1. Track A — charge spread and impact per name (blocking)

This decides the phase. Until it is done no DSR pass on this panel is decision-grade.

**A1 — Spread panel.** New `src/phase4_data/spread.py`, mirroring `capacity.py`'s
analysis-module shape. The bhavcopy carries no quotes, so spread must be *estimated* from
daily OHLC. Build **both** estimators and carry them as a band:

- **Abdi-Ranaldo (2017)**: `S² = 4·E[(c_t − η_t)(c_t − η_{t+1})]`, `η = (log high + log
  low)/2`. Biased **up** by clipping negative estimates.
- **Corwin-Schultz (2012)**: two-day high-low ratio. Biased **down** on this data.

Estimate per name on a **trailing 63-day window** (causal — do not use contemporaneous
or future bars, even though cost is not a signal), clip once at the window level, not
per-observation. Emit `data/bhavcopy/spread_daily.parquet` as `(date × ticker)`
half-spread in bps, one frame per estimator.

Validation gate: the estimate must be **monotone in liquidity**. Reference values
measured on the current panel (quarterly, in-universe, by turnover quintile):

| quintile | Corwin-Schultz | Abdi-Ranaldo |
|---|---|---|
| most liquid | ~0 bps | ~0 bps |
| 2nd | 3.3 | ~0 |
| 3rd | 7.5 | 32 |
| 4th | 11.4 | 57 |
| least liquid | 16.1 | 75 |

**A2 — ADV panel.** Trailing `ML_CONFIG.cost.adv_window` (21) median rupee turnover per
name, plus trailing 21-day daily vol. `capacity.py::load_inputs` already computes both
on the fly — factor that out and persist it rather than duplicating.

**A3 — Per-name cost in Tier 3.** The surgical change. Today
(`tier3_execution.py` ~395–413):

```python
turnover: pd.Series = weights.diff().abs().sum(axis=1)   # reduced too early
trade_cost: pd.Series = turnover * (cost_cfg.compose_oneway_bps() / 1e4)
```

Keep `traded = weights.diff().abs()` **unreduced** as a `(bars × ticker)` frame, then:

```
participation = (traded · aum_rupees) / adv_name
cost_bps      = compose_oneway_bps()                      # statutory, flat
              + half_spread_name                          # A1
              + impact_bps(sigma_name, participation)     # config staticmethod
trade_cost    = (traded * cost_bps / 1e4).sum(axis=1)
```

`_NSECostConfig.impact_bps()` already exists and is already used by `capacity.py` — wire
the existing function, do not write a second one. Requires one new config field,
`aum_rupees`, frozen at **Rs 1 crore** (the documented operating point; capacity does not
bind there, so a failure at Rs 1 cr is a failure everywhere larger).

**A4 — Participation cap.** `ML_CONFIG.cost.max_participation` (0.10) exists and is
unused by the engine. Apply it in `build_weight_matrix` immediately after the `1/k`
assignment (`long_w`/`short_w`, ~lines 100–103) and **before** the style branch, so all
three styles inherit it. Note the consequence flagged during exploration: nothing
downstream renormalizes, so a cap that reduces a name's weight reduces gross exposure
unless the residual is redistributed. Decide and document which — recommend
**redistribute within the leg**, so leg gross stays 1.0 and the `gross~1.0 / ~2.0 / net~0`
dry-run invariants continue to hold.

---

## 2. Track B — an SLB-only short leg

### 2.1 The futures route is closed (measured, not assumed)

38 quarterly F&O bhavcopy snapshots, 2016-02 → 2025-05, both archive eras:

| | measured |
|---|---|
| stocks with live futures | 137–227, mean **182** |
| of those, inside the NSE-500 universe | mean **171 = 34%** |
| their median liquidity rank in the 500 | **105 of 500** |
| contract value | p10 Rs 4.39 L · **median Rs 6.44 L** · p90 Rs 9.52 L |

At Rs 1 cr, `long_short` wants Rs 2 L per short name = **0.31 lots — zero names can hold
even one lot**. `dynamic_tilt` is worse (0.09 lots; its calm short gross is 30%). A
50-name futures short book needs Rs 3.2 cr for one lot per name (weights still ±100%
wrong from integer granularity) and ~Rs 32 cr for usable weight accuracy — while the
capacity curve puts impact at 19.2 bps at Rs 25 cr against 6.8 bps of headroom. **Lot
granularity pushes AUM up, impact pushes it down, and the two do not overlap.** At Rs 3.2
cr — the lowest mechanically feasible AUM — the square-root law alone gives ~6.9 bps,
which already exceeds the entire headroom before any spread is charged.

Recorded in the `fno-short-leg-infeasible` memory. **Do not build the futures variant.**

### 2.2 What gets built instead

**B1 — Point-in-time shortable set.** Extend `src/phase4_data/bhavcopy_download.py` with
an `fo` kind — same host, same era-probe pattern, same missing-ledger and body-validation
machinery:

- legacy `content/historical/DERIVATIVES/<YYYY>/<MON>/fo<DD><MON><YYYY>bhav.csv.zip`
- UDiFF `content/fo/BhavCopy_NSE_FO_0_0_0_<YYYYMMDD>_F_0000.csv.zip`

Both verified reachable. Filter to stock futures (`INSTRUMENT == "FUTSTK"` legacy,
`FinInstrmTp == "STF"` UDiFF), take unique underlying symbols per date → `(date ×
ticker)` boolean `data/bhavcopy/shortable_mask.parquet`. Derived from *traded contracts*,
so it is genuinely point-in-time and carries no survivorship bias. Full window ≈ 3,100
days at 2.5 s ≈ 2.2 h, resumable.

F&O eligibility is used here as the **availability proxy for SLB**, not because we trade
futures. This is the honest approximation available: NSE publishes no historical
SLB-eligibility archive, and SLB activity concentrates in approximately the F&O names. It
is an *upper bound* on shortability — real SLB is thinner — and must be stated as such.

**B2 — Eligibility filter + re-rank.** Per the Tier 3 map, the filter must be applied to
`short_mask` **before** `short_count = short_mask.abs().sum(axis=1)`
(`build_weight_matrix` ~line 101), or `k` goes stale and the dollar-neutral / 130-30 math
breaks. Two-step, in `tier1_trees._allocate_deciles` or a new Tier 3 pre-step:

1. mask out non-shortable names from the short candidate set;
2. **re-rank within the eligible subset** so the short book still holds `k` names.

Step 2 matters — without it the short leg silently shrinks and `long_short` stops being
dollar-neutral. Note `apply_rebalance_buffer` runs on alpha ranks upstream, so the
eligibility filter must compose with it, not fight it: filter the candidate set, then
buffer within it.

**B3 — Realistic borrow.** The flat `short_borrow_bps_annual = 50.0` assumes unlimited
availability at a uniform price. Replace with a liquidity-tiered rate (cheapest tier for
the top-quartile-by-turnover eligible names, escalating down), frozen a-priori from
published SLB fee ranges rather than fitted. Document the tiers as an assumption.

**B4 — Coverage diagnostic (read-only, run before anything else).** On the existing
Phase 4b alpha scores: **how many of the 50 names the model wants to short each day were
actually eligible?** This needs no new engine run once E1 caches Tier 1. If the answer is
that most of the intended short book was never shortable, Track B's cells are largely
hypothetical and that is the finding — report it and stop, rather than reporting a
restricted variant's Sharpe as if it were the same strategy.

---

## 3. Track C — turnover, the one lever that helps

Every other change in this phase lowers the result. Turnover is the only axis with
headroom left, and it is where the cost fragility comes from.

Two facts constrain the design:

- **The no-trade buffer is already active and already exhausted.** Confirmed by
  reconstructing the Phase 4b ledger hash: `rebalance_buffer_mult = 2.0` ran. Phase 3's
  sensitivity scan showed net Sharpe **plateaus flat across 2.0–4.0** (0.569 / 0.567 /
  0.501), so widening it further does nothing.
- **Breadth increased turnover.** Same buffer, same 5-day target: daily `long_only` went
  0.32/bar at 68 names → **0.45/bar at 500**. A 50-name book drawn from 500 rotates far
  more than one drawn from 68. The breadth expansion bought Sharpe and cost fragility
  together.

**C1 — A 21-day target.** Add `tgt_fwd_logret_21b` to `feature_creator` alongside the
existing 1b/5b (`shift(-21)`, same construction, never fed as a feature). This is the
untried axis: 1b → 5b roughly halved turnover and was the single biggest net lift in
Phase 3.

**This is a selection-informed choice and is declared as such** — the hypothesis comes
from having seen 5b beat 1b out-of-sample. It is therefore committed a-priori as a
**2-point structural fork** in the frozen grid (§5) and paid for in N, rather than run as
a scan whose best value gets promoted.

---

## 4. Track D — honest N and the statistical gate

### 4.1 The trial ledger

`_DSRConfig.trials_override` exists and is `None`. Set it from an explicit, auditable
tally maintained in `config.py`:

| source | cells | counted | rationale |
|---|---|---|---|
| Phase 3 §3 net grid (4 freq × 3 styles, 1b) | 12 | **12** | the headline search |
| Phase 3 §2 gross, same 12 configs | 12 | 0 | same configurations measured without costs — not a new search |
| Phase 3 §4 daily-only, 1b | 3 | 0 | a subset of the 12 above |
| Phase 3 §5 daily-only, 5b | 3 | **3** | new target ⇒ new configurations |
| Phase 3 §6 buffer sensitivity (6 values) | 6 | **5** | `mult=2.0` already counted in §5 |
| Phase 3 §7.3 multi-scale panic gate | 1 | **1** | tested and rejected — still a search |
| Phase 4b headline (`daily_nse500` × 3 styles) | 3 | **3** | |
| **prior total** | | **24** | |
| Phase 4c frozen grid (§5) | 6 | **6** | |
| **N declared for Phase 4c** | | **30** | |

Judgment calls, stated so they can be challenged: Phase 1 sandbox runs are **excluded**
(synthetic data, a different question); gross/net re-measurements of identical configs
are **not** double-counted; sensitivity cells on the separate ledger **are** counted,
because searching them is searching them regardless of which file they landed in.

Expect this to bite. At N = 30 the deflation benchmark roughly doubles versus N = 3
(SR\* ≈ 0.95 annualized vs 0.39), and on the *uncorrected* Phase 4b numbers only
`long_short` still clears — before Track A charges a single basis point.

### 4.2 Autocorrelation adjustment

The gate's PSR/DSR assumes i.i.d. daily returns and uses raw `T = r.shape[0]`. Measured
on the Phase 4b ledger:

| cell | lag-1 AC | naive SR | Lo (2002) adjusted |
|---|---|---|---|
| long_only | +0.101 | 0.859 | 0.780 |
| long_short | +0.140 | 1.778 | **1.343** |
| dynamic_tilt | +0.086 | 1.365 | 1.204 |

Positive autocorrelation is structural here — positions persist for days under a 5-day
target — so the naive `√252` annualization overstates by 10–31%. Apply the Lo
autocorrelation-corrected scaling in `tier4_dsr_gate` before the PSR/DSR computation,
uniformly to every column. Under a 21-day target (C1) this correction becomes *more*
important, not less.

### 4.3 What does **not** change

**Keep the 0.95 threshold.** It is the project's frozen prior; moving it after seeing
results is precisely the selection bias the gate exists to correct.

One refinement worth considering: `sharpe_std` — which sets SR\* — is currently the
sample stdev of however many columns are in the ledger being scored, i.e. **3 points**
spanning three very different styles. With the §4.1 tally in place, estimating that
dispersion across the program's pooled ledgers rather than one 3-column file would be
more stable. Optional; do not let it block the phase.

### 4.4 Verdict on the DSR gate

**Two changes: honest N (4.1) and the autocorrelation correction (4.2).** Both are
uniform, outcome-independent, and monotone against the strategy. Threshold, PSR formula
and `_to_daily` compounding stay as they are — they are correct.

---

## 4A. Verdict on the HMM: no changes

Asked directly, and the answer is that Tier 2 is not where the problem is.

- **`hmm_states` stays frozen at 2.** Tier 3 gates only on `panic`, a causal trailing
  percentile of a continuous stress score computed independently of the state count. A
  2- vs 3-state HMM is return-identical today; sweeping it would inflate N for nothing.
- **The gate has already been tuned and the tuning failed.** Phase 3 §7.3's multi-scale
  variant degraded every window and broke the COVID protection.
- **The drawdowns are a beta problem, not a gate problem.** The old gate already tripped
  on 26% of 2018 days and the book still lost 36.5%. A de-risk gate can only reach cash;
  it cannot make a β 0.62 net-long book profit while its universe falls 36%.

The real fix for that exposure is **beta/sector neutralization in portfolio
construction**, not gate engineering. That is deliberately **out of scope for 4c** — it
is a new degree of freedom and would need its own frozen prior and its own N. Named here
so it is not rediscovered.

Minor note: the Phase 4b log emitted `Model is not converging` (delta −0.002) during the
Tier 2 fit. Benign — it is at the convergence floor, not diverging — but worth silencing
or asserting a tolerance so a real failure stays visible.

---

## 5. The frozen grid

Committed **before** the run, per R5. Six cells:

```
execution_style ∈ {long_only, long_short_slb, dynamic_tilt_slb}   (3)
target          ∈ {tgt_fwd_logret_5b, tgt_fwd_logret_21b}         (2)
```

Everything else fixed: `frequency = daily_nse500`, `decile_pct = 0.10`,
`rebalance_buffer_mult = 2.0`, `hmm_states = 2`, `aum_rupees = 1e7`,
`max_participation = 0.10`, costs on with A1–A3 wired.

`long_only` has no short leg, so it is unaffected by Track B and serves as the control:
it isolates how much of the cost damage is Track A alone.

**N = 30** (§4.1). Run once. Do not run the unrestricted `long_short` for comparison and
then report whichever reads better — if the unrestricted variant is wanted as a
diagnostic it goes on the **sensitivity ledger** and is never promoted.

---

## 6. Read-only diagnostics

None of these select anything; all are reported alongside the verdict.

- **B4 short-book coverage** — % of intended short names that were eligible, per year.
- **IC by liquidity quintile** — decomposes Phase 4b's unexplained 2.5× IC jump
  (0.02 → 0.0511). If the IC concentrates in the illiquid tail, then §1 and the
  2016-2020 concentration are the *same* finding and the edge is a spread artefact. Use
  trailing-turnover rank (the metric `universe.py` already ranks on); market cap is not
  available and is not needed for this question.
- **IC by year** — is the 2021→2025 decay in the signal or in the costs?
- **Turnover attribution** — how much is signal rotation vs decile-boundary churn vs
  panic-gate liquidation.
- **Gross vs net per cell** (E2), so cost drag is measured rather than inferred.

---

## 7. Verification

```bash
# 0. module dry-runs — each Phase 2/3 module ships a __main__ synthesizing a
#    schema-correct panel and asserting invariants. New modules must too.
python src/phase4_data/spread.py          # monotone-in-liquidity gate (§1 A1 table)
python src/production_ml/tier3_execution.py   # gross~1.0 / ~2.0 / net~0 invariants

# 1. REGRESSION GATE — Phase 3 and Phase 4b must reproduce bit-for-bit.
#    Every Track A/B change is gated behind config so the old paths are untouched:
#    spread/impact off + no eligibility filter must return the existing ledgers exactly.
python run_pipeline_ml.py --frequencies daily --target tgt_fwd_logret_5b --skip-dsr
#    -> diff against data/trial_database/production_dsr_matrix_daily5b.parquet

# 2. shortable mask sanity
#    ~137-227 names/day, ~171 in-universe, median liquidity rank ~105/500 (§2.1)

# 3. cost sanity — charge zero spread + zero impact and confirm the result is
#    identical to Phase 4b; then step spread in and confirm monotone degradation.

# 4. the run (frozen grid, N=30)
nohup python -u run_pipeline_ml.py --frequencies daily_nse500 \
  --styles long_only,long_short_slb,dynamic_tilt_slb \
  --ledger data/trial_database/phase4c_dsr_matrix.parquet \
  > logs/phase4c_run.log 2>&1 &
```

The regression gate is the important one. Phase 4b's halt-bridge precedent
(`bridge_halts` gated on `terminal_returns is not None`) is the pattern: **new behavior
is opt-in so prior results stay reproducible.**

---

## 8. Decision rules, committed in advance

- **Any cell clears DSR 0.95 at N=30 with spread + impact charged and an SLB-feasible
  short leg** → a genuine positive. Then, before any claim: the §6 diagnostics must not
  show the edge concentrated in the illiquid tail, and the recent-half DSR must be
  reported next to the full-sample one.
- **Net-positive but below the gate** → the Phase 3/4b shape at larger N and honest
  costs. Report as a rigorous negative and stop.
- **Track B4 shows most of the intended short book was never shortable** → report that
  as the finding; the restricted variant's Sharpe is a different strategy, not a
  correction to the old one.
- **Net-negative** → costs dominate at 500 names too, and the breadth thesis is closed.

The honest expectation, stated up front so a negative is not re-litigated: the measured
spread band (6–32 bps) sits well above the passing cells' ~7 bps headroom, the honest-N
benchmark roughly doubles, and the autocorrelation correction removes another 10–31% of
Sharpe. **The most likely outcome is that Phase 4b's two passes do not survive.** That
result is worth having — it is the difference between a finding and an artefact, and
[`phase3_presentation_notes.md`](phase3_presentation_notes.md) §4.6 already commits to
accepting it.

---

## 9. Sequencing

| # | Work | Gate |
|---|---|---|
| 1 | E1 Tier 1 cache, E2–E5 hygiene | regression: Phase 3 + 4b reproduce exactly |
| 2 | B1 shortable mask; **B4 coverage diagnostic** | may end Track B outright |
| 3 | A1 spread panel, A2 ADV panel | monotone-in-liquidity gate |
| 4 | A3 per-name costs, A4 participation cap | zero-spread run == Phase 4b |
| 5 | B2 eligibility filter + re-rank, B3 borrow tiers | dollar-neutrality preserved |
| 6 | C1 21-day target | targets never fed as features |
| 7 | D1 honest N, D2 autocorrelation | uniform across all columns |
| 8 | Freeze grid, run once, write `phase4c_results.md` | |

Step 2 is deliberately early: it is cheap once E1 lands and it can close Track B before
any of Track A's work is spent on it.

---

## Reference

- Phase 4b result being corrected: [`phase4_results.md`](phase4_results.md)
- Capacity / AUM curve: [`phase4_capacity.md`](phase4_capacity.md)
- Panel build and its six defects: [`phase4_data.md`](phase4_data.md)
- Phase 3 baseline and the gate dead-end: [`phase3_results.md`](phase3_results.md) §6, §7
- Discipline (freeze-before-test, honest N, sweep budget):
  [`phase3_design_requirements.md`](phase3_design_requirements.md)
- Selection vs look-ahead bias: [`phase3_deferred_hmm.md`](phase3_deferred_hmm.md) §4
