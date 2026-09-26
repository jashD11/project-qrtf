# Phase 4c — results: what the edge is worth once the costs are real

**Status: COMPLETE.** Run 2026-08-06. Executes [`phase4c_plan.md`](phase4c_plan.md);
corrects [`phase4_results.md`](phase4_results.md). Ledger:
`data/trial_database/phase4c_dsr_matrix.parquet` (2337 × 6), gross in the `_gross`
sibling, per-cell execution diagnostics in `_execution_diagnostics.csv`.

**Verdict in one line: the trees have genuine, liquidity-uniform cross-sectional skill,
and it is not worth enough to pay for the trades it demands. All six cells fail the DSR
gate — the best reaches 0.017 against a 0.95 threshold — and the short leg that carried
Phase 4b's two passes was never borrowable.**

---

## 0. The regression gate (run before anything else was believed)

Every Phase 4c addition is gated behind a `ML_CONFIG.cost` flag defaulting to `False`, or
behind an `_slb` style suffix. With those defaults, both prior results reproduce
**bit-for-bit** — not approximately, exactly:

| ledger | columns | result |
|---|---|---|
| Phase 3 daily, 5b target | 3 | all bit-identical, `max abs diff = 0.0` |
| Phase 4b `daily_nse500`, 5b target | 3 | all bit-identical, `max abs diff = 0.0` |

This is what makes the rest of the document a comparison rather than a coincidence. The
Phase 4b run also reproduced its published headline numbers exactly (+308.4341% /
+639.2251% / +1194.7144%, turnover 0.45 / 0.82 / 0.80).

---

## 1. Track A — the spread, measured

The bhavcopy carries no quotes, so the half-spread is estimated from daily OHLC by two
published estimators, carried as a **band** because they bracket the truth from opposite
sides. Measured on this panel, in-universe, by trailing-turnover quintile:

| quintile | Corwin-Schultz (mean bps) | Abdi-Ranaldo (mean bps) | CS % clipped to 0 | AR % clipped |
|---|---|---|---|---|
| 1 most liquid | 2.41 | 18.79 | 76% | 53% |
| 2 | 4.53 | 22.51 | 65% | 49% |
| 3 | 6.24 | 26.56 | 60% | 46% |
| 4 | 7.50 | 30.06 | 56% | 42% |
| 5 least liquid | 9.16 | 32.37 | 51% | 39% |
| **pooled** | **5.97** | **26.07** | 62% | 46% |

Both are **strictly monotone in liquidity** — the validation gate — and the pooled band
(5.97 → 26.07 bps) independently reproduces the plan's planning-stage estimate of
6.1 → 31.7 bps from a separate implementation.

**Two method notes that changed the answer.**

*The gate had to move from the median to the mean.* Between 39% and 76% of observations
clip to exactly zero — neither estimator can resolve a spread below the price's own noise
floor. CS's **median** is therefore `0.00` in every single quintile, and a monotonicity
test on it would have passed degenerately on a flat row of zeros. The mean is also the
economically correct statistic: cost is charged on every trade, and a clipped zero is an
unresolved spread, not a free one.

*The clip is applied once, at the window level.* Both estimators produce negative
estimates on individual observations — sampling noise around a small true spread, not
evidence of a negative spread. Clipping each observation before averaging converts
symmetric noise into systematic upward bias.

**Which estimator the headline uses, and why it is the charitable choice.** On synthetic
data with a *known* planted spread, Abdi-Ranaldo recovers the level almost exactly
(planted 2/10/25/50/100/200 bps → recovered 2.8/4.7/21.7/50.4/97.7/201.5) while
Corwin-Schultz reads far too low (0/0/0/14.9/53.5/145.0). CS is the **under**-reading
estimator, so the frozen grid runs on CS: if the edge dies under the optimistic estimate,
the verdict cannot be blamed on the choice. AR is reported as the band.

---

## 2. Track A — the cost model, verified before it was trusted

Three checks, in the order they must pass:

**Zero spread + zero impact through the new per-name path == Phase 4b.**

| cell | max abs diff vs Phase 4b | SR |
|---|---|---|
| long_only | 1.1 × 10⁻¹⁶ | 0.859 (ref 0.859) |
| long_short | 1.1 × 10⁻¹⁶ | 1.778 (ref 1.778) |
| dynamic_tilt | 1.1 × 10⁻¹⁶ | 1.365 (ref 1.365) |

Float rounding only. So every difference below is the cost model, not a refactor artefact.

**Stepping a flat half-spread in degrades the result monotonically** — and reproduces
Phase 4b's §3.1 sensitivity table digit-for-digit from a completely different code path:

| half-spread bps | 0 | 2 | 4 | 6 | 8 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|
| long_only | 0.859 | 0.747 | 0.635 | 0.523 | 0.410 | 0.298 | 0.018 | −0.263 |
| long_short | 1.778 | 1.451 | 1.123 | 0.796 | 0.469 | 0.142 | −0.673 | −1.481 |
| dynamic_tilt | 1.365 | 1.182 | 0.998 | 0.815 | 0.632 | 0.448 | −0.010 | −0.468 |

(Phase 4b reported 0.86/0.75/0.63/0.52/0.41/0.29/0.01/−0.27 for `long_only`.)

**Cost-panel coverage is essentially complete** on the cells that were actually traded:
ADV 100.0%, sigma 100.0%, half-spread 99.9%. The remainder takes that date's
cross-sectional median — never zero, and the per-name cost frame is asserted NaN-free,
because `.sum()` skips NaN and a gap would silently price an unmeasured name as free to
trade.

---

## 3. The Track A result — this is what decides the phase

Measured half-spread (CS) + square-root impact at **Rs 1 crore**, unrestricted short leg,
flat borrow — i.e. Phase 4b's exact strategies with nothing changed but the cost:

| cell | SR gross (measured) | SR Phase 4b | **SR real cost** | + participation cap | bps/side | drag/yr |
|---|---|---|---|---|---|---|
| long_only | 1.680 | 0.859 | **0.195** | 0.178 | 26.19 | 29.7% |
| long_short | 4.198 | 1.778 | **−0.225** | −0.283 | 27.08 | 55.8% |
| dynamic_tilt | 2.718 | 1.365 | **0.245** | 0.207 | 26.84 | 54.5% |

**`long_short` — Phase 4b's headline, DSR 1.000 — goes negative.** Cumulative return on
`long_only` collapses from +308.4% to **+15.4%** over 9.27 years.

The effective charge is ~26–27 bps per side against a statutory 14.66: roughly 6 bps of
spread and 5–6 bps of impact on top. Phase 4b's own breakeven analysis put the DSR-0.95
threshold at 6.8 bps of *extra* one-way cost for `long_short`; the measured extra cost is
**12.4 bps**, nearly double.

**Gross was measured, not inferred (E2).** Phase 4b could only reconstruct gross by adding
an average drag back, giving 1.68 / 4.21 / 2.72. Measured directly: **1.680 / 4.198 /
2.718.** Their reconstruction was sound — and it is now a measurement.

**Capacity is not the binding constraint at this size.** p99 participation is 4.4% against
a 10% ceiling, so the cap barely binds and costs the passing cells only ~0.02–0.06 of
Sharpe. This is a cost failure, not a sizing failure — which also means it does not get
better by running less money.

---

## 4. The read-only diagnostics (§6) — what they rule in and out

### 4.1 The edge is **not** a small-cap spread artefact

Ensemble rank-IC computed *within* each trailing-liquidity quintile:

| quintile | 1 most liquid | 2 | 3 | 4 | 5 least liquid |
|---|---|---|---|---|---|
| mean IC | 0.0408 | 0.0457 | 0.0413 | 0.0394 | **0.0484** |
| IC_t | 12.87 | 15.81 | 14.62 | 14.20 | 17.26 |

Essentially **flat**. The plan's §6 hypothesis — that Phase 4b's 2.5× IC jump was the
illiquid tail, making the spread problem and the IC jump one finding — is **refuted**. The
skill is real and roughly uniform across the liquidity spectrum.

This makes the negative result stronger, not weaker: the strategy does not fail because
its signal is a microstructure illusion. It fails because a genuine ~0.04 IC, monetized at
0.45–0.82 turnover per day, cannot pay 26 bps a side.

### 4.2 The 2021→2025 decay is in the **signal**, not the costs

| year | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean IC | 0.0475 | 0.0523 | 0.0586 | 0.0863 | 0.0564 | 0.0217 | 0.0312 | 0.0368 | 0.0183 | 0.0127 |
| IC_t | 9.12 | 11.42 | 9.85 | 10.48 | 8.00 | 4.89 | 6.28 | 7.91 | 2.31 | **1.21** |

The predictive signal itself decays — IC_t falls from ~10 to 1.21. By 2025 the
cross-sectional skill is not statistically distinguishable from zero. Phase 4b's §3.2
observation ("essentially all of the edge is in 2016–2020") is confirmed, and its cause
identified: the signal, not the cost model.

### 4.3 Turnover is irreducible

*(B4 short-book coverage is reported in §6, where the short leg is the subject.)*

| source | share |
|---|---|
| boundary churn already removed by the no-trade buffer | 47.2% of raw decile turnover |
| panic-gate liquidation / re-entry | 3.0% of total |
| **signal rotation (irreducible)** | **97.0% of total** |

The buffer is doing real work and is exhausted — consistent with Phase 3's finding that
net Sharpe is flat across `rebalance_buffer_mult` 2.0–4.0. The gate is a rounding error.
97% of turnover is the model genuinely changing its mind, so the only remaining lever is
a slower target, which is exactly what C1 tests.

---

## 5. Track D — the statistical corrections, applied alone

Applied to the **unmodified Phase 4b ledger**, so the effect of the statistics is separated
from the effect of the costs:

| cell | SR naive | lag-1 AC | SR Lo-corrected | DSR (N=30) | verdict |
|---|---|---|---|---|---|
| long_short | 1.78 | 0.140 | 1.35 | 0.986 | PASS |
| dynamic_tilt | 1.37 | 0.086 | 1.20 | 0.954 | PASS |
| long_only | 0.86 | 0.101 | 0.77 | 0.671 | fail |

Deflation benchmark SR\* = 0.62 annualized (vs 0.39 at N=3).

**The statistics alone do not overturn Phase 4b** — both cells still clear 0.95. This is
worth stating plainly, because it locates the result: honest N and the autocorrelation
correction cost `long_short` 0.43 of Sharpe and it survived; the measured spread cost it
1.56 and took it below zero. **Track A is the whole story.**

**A note on how the autocorrelation correction is computed.** The plan specified Lo's
lag-by-lag form, `eta(q) = q / sqrt(q + 2·sum_k (q−k)·rho_k)`. That form was implemented
first and rejected on measurement: it multiplies every estimated `rho_k` by ~252, and each
`rho_k` carries a sampling error of ~`1/sqrt(T)` ≈ 0.02, so ten lags of pure noise move the
answer by more than 10%. On i.i.d. synthetic returns it returned factors between 0.99 and
1.18 depending only on the random seed — it was estimating its own noise. The **overlapping
variance ratio** at 21 days estimates the same quantity as one number: it centres on 1.000
(sd 0.05) on i.i.d. input and recovers AR(1) theory to within 0.01. It yields 1.35 for
`long_short`, against the plan's independently-derived expectation of 1.343.

**The dividend guard, measured.** Phase 4b's `long_only` verdict turned on an *assumed*
flat 1.3%/yr. The index archive's own `Div Yield` column gives a measured mean of
**1.297%/yr** — the assumption was accurate. It is now applied to actual per-bar net
exposure rather than uniformly, so a dollar-neutral book earns ~0 and a book in cash earns
0, neither of which was true before. It is deliberately **not** folded into the ledger, so
the ledger stays comparable with Phase 3 and Phase 4b.

---

## 6. Track B — the short leg cannot be built

3,092 F&O sessions, 2013→2025, both archive eras, zero download failures; 99.7%
symbol→ticker match; only **2** calendar sessions needed forward-filling, so the mask is
genuinely point-in-time throughout.

**Of the 500-name universe, a mean of 177 names (35.5%) carried a live stock future** —
independently reproducing the plan's §2.1 figure of "mean 171 = 34%".

But the strategy does not want to short a random 50 names, it wants the *worst* 50, and
those are systematically less likely to be F&O-eligible:

| year | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | **ALL** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| intended short names | 49.1 | 48.9 | 49.1 | 49.0 | 49.0 | 49.1 | 49.2 | 49.0 | 49.0 | 49.0 | **49.0** |
| of which borrowable | 12.8 | 11.6 | 15.5 | 8.6 | 7.6 | 11.1 | 13.3 | 9.4 | 12.2 | 13.7 | **11.0** |
| **coverage** | 26.1% | 23.7% | 31.5% | 17.5% | 15.6% | 22.6% | 27.0% | 19.2% | 24.9% | 28.0% | **22.4%** |

**Roughly 78% of the short book Phase 4b reported was never borrowable** — and this is an
*upper* bound, because F&O eligibility over-states SLB availability. Coverage never
exceeds 32% in any year and falls to 15.6% in 2020.

Per §8's decision rule, this is itself the finding: **both of Phase 4b's passing cells
depended on a book that could not have been held.** The restricted `*_slb` variants below
are a different strategy, not a corrected version of the old one.

---

## 7. The frozen grid — the result

Six cells, committed before the run, `N = 30`, deflation benchmark **SR\* = 0.98**
annualized (against 0.39 at Phase 4b's N = 3).

| target | style | gross SR | **net SR** | Lo-corrected | CumRet | bps/side | turnover | **DSR** |
|---|---|---|---|---|---|---|---|---|
| 5b | long_only | 1.660 | 0.178 | 0.161 | +15.44% | 26.08 | 0.446 | 0.00681 |
| 5b | long_short_slb | 1.882 | −0.987 | −0.906 | −74.35% | 24.52 | 0.637 | 3.0e−09 |
| 5b | dynamic_tilt_slb | 2.014 | −0.049 | −0.045 | −29.76% | 26.04 | 0.717 | 0.00092 |
| 21b | long_only | 1.120 | **0.298** | 0.268 | +43.16% | 25.96 | 0.242 | **0.01650** |
| 21b | long_short_slb | 0.958 | −0.567 | −0.520 | −56.47% | 24.46 | 0.338 | 2.0e−06 |
| 21b | dynamic_tilt_slb | 1.308 | 0.212 | 0.192 | +22.72% | 25.81 | 0.370 | 0.00894 |

Read to the precision the artifacts carry: `phase4c_dsr_matrix_dsr_gate.csv` (SR, Lo, DSR)
and `phase4c_dsr_matrix_execution_diagnostics.csv` (gross, CumRet, bps, turnover). Net of
24.5-26.1 bps/side realised — 14.6558 statutory + measured Corwin-Schultz half-spread +
√-law impact at Rs 1 crore — plus tiered short borrow.

### **0 of 6 cells pass. The best is DSR 0.0165 against a 0.95 threshold.**

Not marginal — the best cell reaches under 2% of the required confidence. `minTRL` is
undefined for every cell: none of them clears SR\* at *any* track length, so no amount of
additional history would rescue them.

**C1 worked, and it was not enough.** The 21-day target does exactly what it was committed
to do — turnover falls from 0.45 to 0.24 per bar on `long_only`, and net Sharpe rises from
0.18 to 0.30. It is the best cell in the grid, and the target axis is the only one whose
mean improves (−0.29 → −0.02). But it buys 0.12 of Sharpe against a 0.95 gate. The lever
works; it is two orders of magnitude too small.

**The SLB restriction destroys the short leg's alpha.** Restricted to ~22% of its intended
names, `long_short`'s *gross* Sharpe falls from **4.20 to 1.88** — over half the raw edge
was in names that could never have been shorted. The restricted leg keeps the full cost
and loses most of the signal, which is why it is the worst cell in the grid.

**Borrow is a rounding error; trading is the whole cost.** Tiered borrow costs 0.4–0.8%/yr.
Trade drag is **16–47%/yr**. B3 was worth building to remove the "unlimited free borrow"
assumption, but it is not where the money goes.

**Capacity never binds.** p99 participation is 2.7–4.1% against a 10% ceiling, at Rs 1
crore. This is a cost failure, not a size failure — it does not improve by running less.

---

## 8. Verdict

**Phase 4b's two DSR passes do not survive.** The result falls in §8's third and fourth
branches simultaneously: most of the intended short book was never shortable, *and* the
surviving long-only book is net-positive but nowhere near the gate.

What each correction was worth, applied to `long_short` (Phase 4b: SR 1.78, DSR 1.000):

| correction | cost in Sharpe | still passing after? |
|---|---|---|
| honest N (30) + autocorrelation | −0.43 → 1.35 | yes, DSR 0.986 |
| measured spread + impact | −1.56 → −0.23 | **no** |
| SLB-feasible short leg | further −0.76 → −0.99 | no |

**Track A alone was decisive; Track B independently voided the same cells.** The statistical
corrections the plan expected to bite turned out to be the mildest of the three.

**This is a rigorous negative result, and a cleaner one than Phase 3's.** The signal is not
an artefact — §4.1 shows the skill is real and uniform across liquidity, which rules out
the spread-artefact explanation the plan flagged as most likely. The trees genuinely
predict cross-sectional returns at IC ≈ 0.04. That edge simply cannot pay 26 bps a side at
0.24–0.72 turnover per day, and it is decaying (§4.2): by 2025 the IC is no longer
statistically distinguishable from zero.

**The breadth thesis is closed.** Phase 4 expanded 68 → 500 names to buy `√breadth`. It
delivered the predicted gross Sharpe, and the expansion simultaneously raised turnover
(0.32 → 0.45/bar) and reached into names with wider spreads. Breadth bought edge and cost
in the same motion, and the costs won.

### What is *not* ruled out

Stated so the negative is not over-read:

- **A lower-turnover construction.** C1 moved in the right direction and was the only
  lever with headroom. A holding-period-constrained or explicitly cost-aware portfolio
  construction is a different design, not a longer target, and was out of scope here.
- **Beta/sector neutralization**, deliberately excluded (plan §4A) as a new degree of
  freedom needing its own frozen prior and its own N.
- **The cost estimate is a band.** The headline runs on Corwin-Schultz (5.97 bps pooled).
  Under Abdi-Ranaldo (26.07 bps) every cell is far worse — the choice only matters for
  *how* dead the result is.

---

## 7. Reproducing this

```bash
# panels
python src/phase4_data/liquidity.py  --build
python src/phase4_data/spread.py     --build      # prints the §1 gate table
python src/phase4_data/dividends.py  --build
python src/phase4_data/bhavcopy_download.py --kinds fo
python src/phase4_data/shortable.py  --build

# the regression gate (§0) — must be bit-identical before anything else counts
python run_pipeline_ml.py --frequencies daily --target tgt_fwd_logret_5b --skip-dsr \
  --ledger data/trial_database/regression/phase3_daily5b.parquet
python run_pipeline_ml.py --frequencies daily_nse500 --target tgt_fwd_logret_5b --skip-dsr \
  --ledger data/trial_database/regression/phase4b.parquet

# the cost verification (§2) and the Track A result (§3)
python scripts/phase4c_cost_sanity.py

# the read-only diagnostics (§4)
python src/phase4_data/phase4c_diagnostics.py --run

# the frozen grid (§6)
python run_pipeline_ml.py --phase4c
```

Module dry-runs (`python <module>.py` with no flags) self-test every component against
synthetic data with known answers, including the estimators' resolution limits.

---

## Reference

- Plan this executes: [`phase4c_plan.md`](phase4c_plan.md)
- Result being corrected: [`phase4_results.md`](phase4_results.md)
- Capacity / AUM curve: [`phase4_capacity.md`](phase4_capacity.md)
- Discipline: [`phase3_design_requirements.md`](phase3_design_requirements.md)
