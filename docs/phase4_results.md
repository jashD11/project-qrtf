# Phase 4 — engine results on the 500-name point-in-time panel

**Status: SUPERSEDED by [`phase4c_results.md`](phase4c_results.md).** The §5 item-1 work this
document was waiting on (per-name spread + impact) was done, and it **overturned the result
below**: `long_short` goes from SR 1.78 / DSR 0.99999 here to SR −0.23 once the measured
half-spread and √-law impact are charged, and to −0.99 once the short leg is restricted to
borrowable names. 0 of 6 Phase 4c cells pass the gate.

This file is kept because it is the record of *what the corrections were applied to*, and
because its ledger reproduces bit-for-bit under Phase 4c's defaults — that regression gate is
what makes the comparison a measurement. **Read it as history, not as a current result.** Every
number in it is real and was verified; the conclusion it points toward is not.

**Run 2026-08-06.** `python run_pipeline_ml.py --frequencies daily_nse500 --target
tgt_fwd_logret_5b` — grid `daily_nse500 × {long_only, long_short, dynamic_tilt}`,
honest **N = 3**. Ledger: `data/trial_database/production_dsr_matrix.parquet` (the
Phase 3 daily ledger was moved to `archive/phase3_daily_dsr_matrix.parquet` first, so
these three columns stand alone). Log: `logs/phase4_run.log`.

**Verdict in one line: two of three cells clear the DSR gate, but the margin is smaller
than the costs the model does not charge, and essentially all of the edge is in
2016–2020. Not a validated positive.**

---

## 1. Headline

2,337 scored days (2016-01-21 → 2025-06-30), 9.27 years, universe 1,030 distinct names,
500/day. Active 2,000 bars / cash 337.

| cell | CumRet | CAGR | vol | net SR | maxDD | Calmar | beta | alpha | turnover/bar | cost drag | **DSR (N=3)** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| long_only | +308.4% | 16.39% | 20.01% | 0.86 | −36.3% | 0.45 | 0.62 | 7.97% | 0.45 | 16.5%/yr | **0.916 fail** |
| long_short | +639.2% | 24.07% | 12.58% | **1.78** | −19.9% | 1.21 | −0.02 | 22.68% | 0.82 | 30.6%/yr | **1.000 PASS** |
| dynamic_tilt | +1194.7% | 31.80% | 22.03% | 1.37 | −41.6% | 0.76 | 0.65 | 20.33% | 0.80 | 29.8%/yr | **0.998 PASS** |
| NIFTY-50 | +250.8% | 14.49% | 16.52% | 0.90 | −38.4% | — | 1.00 | — | — | — | — |

Distribution across execution_style: mean SR 1.33, min 0.86, max 1.78. Deflation
benchmark SR\* = 0.39; cross-trial Sharpe std (daily) = 0.0290.

Beta/alpha are measured against the **forward** NIFTY-50 return (T→T+1), matching the
ledger's convention of indexing each return at its signal date. Against the
contemporaneous index return `long_only` beta reads 0.10, which is an alignment
artefact, not a market-neutral long book — the correct 0.62 is consistent with Phase 3's
0.59.

Gross Sharpe was not separately captured (the run wrote net only). Adding the reported
drag back uniformly gives **approximate** gross SR 1.68 / 4.21 / 2.72 — indicative of how
much of the raw edge costs consume, not a measured figure.

---

## 2. The comparison Phase 4 existed to make

Phase 3 ended at daily `long_only` **SR +0.57 / DSR 0.737** on 68 names and named breadth
as the binding constraint. The prediction was ~SR 0.85 from `√(500/68) = 2.71×` breadth
against a required 1.49× uplift.

`long_only` came in at **SR 0.86** — almost exactly the predicted number. That is the
clean like-for-like cell (same style, same 5-day target, same frozen knobs), and the
breadth thesis called it correctly.

**But the mechanism was not the one predicted.** The run plan set the test explicitly: *if
IC is similar but Sharpe is higher, breadth did the work; if IC itself moved, something
changed about the signal and that needs explaining.* IC moved, by a lot:

| model | meanIC | IC_IR | IC_t | hit% | Phase 3 daily |
|---|---|---|---|---|---|
| lgbm | 0.0501 | 0.445 | 21.48 | 68.9% | |
| xgb | 0.0488 | 0.455 | 21.97 | 68.0% | |
| rf | 0.0462 | 0.455 | 21.99 | 68.3% | |
| **ensemble** | **0.0511** | 0.466 | 22.49 | 69.1% | **≈ 0.02** |

38 folds, 2,337 OOS days. Ensemble IC is **~2.5× Phase 3's**. So the Sharpe uplift is not
breadth acting on a constant signal — the signal itself is different, and the honest
reading is that the 500-name universe contains a *stronger and different* cross-sectional
effect than the 68 liquid names did. The obvious candidate is the small/mid-cap tail:
short-horizon reversal and illiquidity effects are mechanically stronger there, and they
are also exactly the effects that do not survive the spread. §3 is the test of that, and
it is not reassuring.

`√breadth` alone predicts IR uplift of 2.71×; observed `long_only` uplift is 0.86/0.57 =
1.51×, while IC rose 2.5×. These do not reconcile under the simple model, which is
further reason not to read this as a clean breadth confirmation.

---

## 3. Why this is not a validated positive

### 3.1 The edge dies inside the unmodelled cost band

`slippage_bps = 0.0` in `_NSECostConfig`, and the size-dependent impact term is *not
wired into the execution path*. The backtest therefore charges **zero spread and zero
market impact** — only the 14.6558 bps statutory stack. With 0.45–0.82 of the book
turning over **per day**, every extra 1 bp of one-way execution cost removes ~2.1%/yr for
`long_short`.

Net SR against extra one-way slippage (drag = avg turnover × s):

| cell | 0 bps | 2 | 4 | 6 | 8 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|
| long_only | 0.86 | 0.75 | 0.63 | 0.52 | 0.41 | 0.29 | 0.01 | −0.27 |
| long_short | 1.78 | 1.45 | 1.12 | 0.79 | 0.46 | 0.14 | −0.69 | −1.51 |
| dynamic_tilt | 1.37 | 1.18 | 1.00 | 0.82 | 0.63 | 0.45 | −0.01 | −0.47 |

Breakeven:

| cell | net SR → 0 | **DSR → 0.95** |
|---|---|---|
| long_only | 15.2 bps | already failing |
| long_short | 10.8 bps | **6.8 bps** |
| dynamic_tilt | 14.9 bps | **7.8 bps** |

`phase4_capacity.md` puts modelled impact at Rs 1 cr at **3.8 bps** — but that is the
square-root *impact* law only and excludes the bid–ask half-spread entirely. Total
realistic one-way execution cost is half-spread + impact. The passing cells survive only
if that total stays under **~7 bps**, i.e. under ~3 bps of half-spread. That is plausible
for large caps and implausible across a 500-name universe whose whole claimed advantage
comes from its smaller names. **Headroom is under 2×, on the one input the model does not
measure.**

### 3.2 The edge is concentrated in the first half

| cell | 2016–2020 SR | 2021–2025H1 SR | early DSR | **recent DSR** |
|---|---|---|---|---|
| long_only | 1.04 | 0.70 | 0.792 | 0.911 |
| long_short | 2.55 | 0.78 | 1.000 | 0.940 |
| dynamic_tilt | 2.03 | 0.76 | 0.998 | 0.931 |

**On the recent half alone, nothing clears 0.95.** `long_short` falls from SR 2.55 to
0.78. Annual returns show the same shape, including a sharp 2025 H1 reversal (long_only
−23.8%, dynamic_tilt −27.8% against NIFTY +8.4%). A decaying edge concentrated in the
older half is the signature of a microstructure effect being competed away — or of
exactly the small-cap spread effect §3.1 warns about.

### 3.3 The two passing cells depend on a short leg

Both `long_short` and `dynamic_tilt` require holding ~50 short names drawn from the
bottom decile of NSE-500. Overnight shorting in India is restricted to the F&O segment
(~180–220 names) plus a thin SLB market; the bottom decile of a 500-name cross-section
will systematically select small, distressed, often non-F&O names. The model charges a
flat 50 bps/yr borrow and assumes unlimited availability. **The feasibility of the short
leg is not established, and the only cell that avoids the question — `long_only` — is the
one that fails the gate.**

### 3.4 The `long_only` verdict is decided by the dividend assumption

Per the §6 dividend guard, adding ~1.3%/yr to the net-long books (dollar-neutral
`long_short` nets to ~0):

| cell | DSR | DSR + dividends |
|---|---|---|
| long_only | 0.916 | **0.949605** |
| long_short | 1.000 | 1.000 |
| dynamic_tilt | 0.998 | 0.999 |

`long_only` lands at **0.9496 against a 0.95 threshold** — it fails by 4 × 10⁻⁴. It does
not flip, but stating it as a clean fail would be misleading: the verdict on that cell is
set by an assumed dividend yield, not by the evidence. Flagged per §6 rather than buried.

### 3.5 N = 3 makes the deflation benchmark noisy

The gate's own note applies: with 3 trials the cross-trial Sharpe std (0.0290) that sets
SR\* = 0.39 is estimated from three points spanning very different styles. The DSR values
here are not precise to the third decimal.

---

## 4. Engine defect found and fixed during this run

The first attempt died in Tier 3 with `assert_no_implicit_exit`: 5 held positions had no
forward return. Diagnosis showed all 5 were **mid-series trading halts, not exits** — JKIL
resumes 4 days later and trades 1,953 more bars, UNITECH resumes after 66 days,
JPASSOCIAT after 7. Because `apply_terminal_returns` fills only each ticker's
`last_valid_index`, the guard was **unsatisfiable for halts**: the Phase 4 path could
never have completed as shipped.

Panel-wide there are 93 such NaN-forward cells among 1,415,561 in-universe bars — 52
genuine terminal exits and **41 halts**. The halts are not benign: they average **−6.1%**
(range −59% to +60%), so booking them at an implicit 0% would have handed the strategy
that move for free — the same exit-side survivorship flattery B2 exists to close.

**Fix** (`tier3_execution.py`): new `forward_returns(price_wide, bridge_halts=)` carries a
position held into a halt to its next available quote (`bfill().shift(-1)`), since a
suspended stock cannot be sold. `bfill` stops at each name's last quote, so genuine
delistings stay NaN and are still filled by `apply_terminal_returns` — halts and exits do
not collide.

Gated on the Phase 4 path (`terminal_returns is not None`) because the **Phase 2/3 panels
carry internal holes of their own** (17 daily cells, 192/523/1,697 at 60/30/15-min);
applying the bridge unconditionally would have silently restated Phase 3. Verified
`bridge_halts=False` is bit-identical to the previous expression on all four Phase 2/3
frequencies, and that the guard now clears all 1,415,061 candidate cells.

---

## 5. What would have to be true to believe this

In priority order — the first item is the one that decides it:

1. **Charge spread + impact per name.** §3.1 is the whole result: the passing cells have
   under 2× headroom on a cost the model sets to zero. Thread an ADV panel through
   `tier3_execution.py`, keep `traded = weights.diff().abs()` unreduced, and charge
   `config.ML_CONFIG.cost.impact_bps()` plus a per-name half-spread estimate. Until this
   is done the DSR passes are not decision-grade.
2. **Establish short-leg feasibility** (§3.3) — intersect the bottom decile with the F&O /
   SLB-eligible list per date and re-run. If the short leg is largely unshortable, both
   passing cells are void and the honest survivor is `long_only`, which fails.
3. **Explain the IC jump** (§2) — decompose IC by market-cap bucket. If it is concentrated
   in the small-cap tail, §3.1 and §3.2 are the same finding and the result is a spread
   artefact.
4. **Explain the 2021→2025 decay** (§3.2) before treating the full-sample DSR as the
   operative number.
5. Wire the participation cap in `build_weight_matrix`; add the dividend guard properly
   from the `Div Yield` column rather than a flat 1.3%; fix `run_sensitivity`'s two latent
   bugs (omits `terminal_returns`, skips `_to_daily`) before any scan on this panel.

**Do not run a `tgt_fwd_logret_1b` variant and report the better of the two** — that is a
second search and would require N=6.

---

## 6. State on disk (handoff)

Everything under `data/` is gitignored, so it exists on this machine only — it is not
recoverable from the repo. Nothing needs rebuilding to continue.

| artefact | path | note |
|---|---|---|
| Phase 4 ledger (the result) | `data/trial_database/production_dsr_matrix.parquet` | (2337, 3), the three cells above |
| DSR gate output | `data/trial_database/production_dsr_matrix_dsr_gate.csv` | N=3 verdicts |
| Tier 1 rank-IC diagnostic | `data/trial_database/tree_fit_diagnostics_daily_nse500.csv` | the 0.0511 table |
| Phase 3 daily ledger | `data/trial_database/archive/phase3_daily_dsr_matrix.parquet` | **moved aside** by this run so Phase 4 stands alone; restore to the parent dir to re-score Phase 3 |
| Panel / mask / lifecycle | `data/nse500_daily_ohlcv.parquet`, `data/nse500_universe_mask.parquet`, `data/bhavcopy/lifecycle.csv` | validated, unchanged |
| Run log | `logs/phase4_run.log` | not gitignored |

Re-running the grid costs **~80 min** (Tier 1 walk-forward is ~67 of it and is not cached
to disk — a downstream failure loses the whole fit, which happened once here). Gross
Sharpe was never captured; recovering it exactly means a second full run with
`ML_CONFIG.cost.apply_costs = False`.

Two things a fresh session should not have to rediscover:

- **Beta must be measured against the *forward* index return.** The ledger indexes each
  return at its signal date T while carrying the T→T+1 move. Using the contemporaneous
  index return makes `long_only` read beta 0.10, which looks like a market-neutral long
  book and is purely an alignment artefact (correct value 0.62).
- **`run_sensitivity` is still broken on this panel** — it omits `terminal_returns` and
  skips `_to_daily`. Fix before any `--sensitivity` scan here.

---

## Reference

- Data build and validation: [`phase4_data.md`](phase4_data.md)
- Capacity / AUM curve: [`phase4_capacity.md`](phase4_capacity.md)
- Run plan this executed: [`phase4_run_plan.md`](phase4_run_plan.md)
- Phase 3 baseline: [`phase3_results.md`](phase3_results.md)
- Discipline (freeze-before-test, honest N): [`phase3_design_requirements.md`](phase3_design_requirements.md)
