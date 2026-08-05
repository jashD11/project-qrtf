# Phase 4 — engine run plan (handoff)

**Written 2026-08-06 for execution in a fresh session.** Everything below assumes no
prior context. The data is built and validated; this document covers only the run.

---

## 0. Where things stand

The point-in-time NSE panel is built, all blocking gates pass, and the engine is wired.
Nothing has been run on it yet.

| | |
|---|---|
| Panel | 5,026,224 rows · 2,606 entities · 3,094 days · 2013-01-01 → 2025-06-30 |
| Universe | 500 names/day · 1,118 distinct · 46 quarterly rebalances · churn 5.2% |
| Validation | Gates 1, 5, 6 **PASS** — 0 unexplained extremes in 1,415,561 in-universe bars |
| Frequency key | `daily_nse500` (in `config.FREQ_REGISTRY`) |

Why this run exists: Phase 3 ended with daily `long_only` at **SR +0.57 / DSR 0.737**
against a 0.95 gate, and named **breadth** as the binding constraint — 68 names is too
few for `IR ≈ IC·√breadth`. This panel has 500. The bar drops from SR 0.96 to **~0.85**
because the sample is longer (~2,338 scored days vs 1,732), and `√(500/68) = 2.71×` is
the theoretical ceiling against a required ~1.49× uplift.

Capacity is not a constraint at the Rs 1 crore operating point: median participation
0.11%, max feasible N = 488. See [`phase4_capacity.md`](phase4_capacity.md).

---

## 1. Preflight (2 minutes)

```bash
cd "/Users/jash/Desktop/Quant Research/qrtf_engine"

# Gates must still say PASS. If not, STOP — do not run the engine.
python src/phase4_data/bhavcopy_validate.py 2>&1 | grep -E "→ GATE|VERDICT"

# Artefacts present and mutually consistent.
python -c "
import pandas as pd
o=pd.read_parquet('data/nse500_daily_ohlcv.parquet')
m=pd.read_parquet('data/nse500_universe_mask.parquet')
life=pd.read_csv('data/bhavcopy/lifecycle.csv')
assert set(o.ticker)==set(m.columns), 'ohlcv/mask ticker mismatch'
assert not (set(o.ticker)-set(life.ticker)), 'tickers missing from lifecycle'
assert life.ticker.is_unique, 'duplicate tickers in lifecycle'
print('OK', len(o), 'rows', o.ticker.nunique(), 'tickers', m.shape)
"
```

Expected: `VERDICT: PASS`, and `OK 2735949 rows 1118 tickers (3094, 1118)`.

---

## 2. One decision to make first: the ledger

`run_pipeline_ml.py` hard-codes `data/trial_database/production_dsr_matrix.parquet` and
there is **no `--ledger` flag**. That file currently holds **3 columns** from the Phase 3
daily run (already written up in `phase3_results.md`).

Leaving it in place means the Phase 4 columns land alongside Phase 3's, and the DSR gate
then scores all six columns while deflating with `n_trials=3`. The N passed to the gate
is still correct — `run_pipeline_ml.py:299` passes `len(configs)` explicitly, not the
column count — but the printed table would mix two different datasets.

**Recommended: move the old ledger aside so the Phase 4 ledger stands alone.**

```bash
mkdir -p data/trial_database/archive
mv data/trial_database/production_dsr_matrix.parquet \
   data/trial_database/archive/phase3_daily_dsr_matrix.parquet
```

This is reversible and loses nothing — the Phase 3 numbers are recorded in
`docs/phase3_results.md`.

---

## 3. The run

```bash
python run_pipeline_ml.py --frequencies daily_nse500 --target tgt_fwd_logret_5b
```

**Why these flags:**

- `--frequencies daily_nse500` — bhavcopy is daily-only, so there is no intraday
  counterpart. Acceptable: every intraday cell already died on costs in Phase 3, so the
  honest family was always the three daily styles.
- `--target tgt_fwd_logret_5b` — the 5-day label was Phase 3's headline survivor because
  it roughly **halves turnover**, and turnover × cost is what killed every other cell.
  Using the same target keeps the 68 → 500 comparison a clean test of breadth.
- **No `--skip-dsr`.** This is a complete grid over its frequency, so `len(configs) = 3`
  is the honest N.

**Do not also run `tgt_fwd_logret_1b` and then report the better one.** That is two
searches, and the DSR deflation would need N=6. If both are wanted, decide *before*
running and pass the true N.

Grid: `daily_nse500` × {`long_only`, `long_short`, `dynamic_tilt`} = **3 cells**.

---

## 4. What to expect

| Stage | Time |
|---|---|
| Feature build (2.7M rows, universe-masked) | ~5 min |
| **Tier 1 walk-forward** — 37 folds × 108.5 s | **~67 min** |
| Tier 2 regime HMM (once for the run) | ~3 min |
| Tier 3 × 3 styles + Tier 4 DSR | ~5 min |
| **Total** | **~1.5 h** |

Tier 1 is fit **once per frequency** and shared across all three styles; Tier 2 runs once.
That is why three cells cost one Tier-1 fit, not three.

**RandomForest is 97% of the Tier-1 time** (105.5 s of 108.5 s per fold). Lowering
`max_depth` would cut it dramatically — but it is a *frozen* hyperparameter, so changing
it to save wall-clock is a research-integrity decision, not an engineering one. Leave it.

The 108.5 s/fold figure is measured on this machine at 252,000 train rows, which is
exactly 504 days × 500 names — not extrapolated.

Run it in the background and log it:

```bash
nohup python -u run_pipeline_ml.py --frequencies daily_nse500 \
  --target tgt_fwd_logret_5b > logs/phase4_run.log 2>&1 &
tail -f logs/phase4_run.log
```

---

## 5. Known gotchas

1. **Mixing frequencies is rejected.** `daily_nse500` and Phase 2/3 frequencies need
   different regime panels, so `compute_regime` raises if both appear in one run. Run
   them separately.
2. **`run_sensitivity` has two latent bugs** — it omits `terminal_returns` (so delisting
   exits are not applied) and skips `_to_daily`. Fine for the headline run, which uses a
   different code path, but **fix before any `--sensitivity` scan on this panel.**
3. **Slippage is not charged.** `tier3_execution.py` applies only the flat statutory
   stack (14.6558 bps one-way). At Rs 1 cr the modelled impact is ~3.8 bps, so net
   figures are optimistic by roughly that. Not wired in by design — see §6.
4. **Dividends are absent.** Price-return panel; NIFTY-50 yields 1.22–1.38%/yr, so a long
   book is understated by about that. Biases results *pessimistically*.
5. **Demergers are unadjusted** (8 events) — the parent books a fake loss. Also
   pessimistic.
6. **Memory**: the walk-forward held ~3.5 GB at 15-min in Phase 3. Daily at 500 names is
   252k train rows/fold, comfortably smaller. `rf.n_jobs=6` is already tuned for this
   machine.

---

## 6. After the run

**Record, in `docs/phase4_results.md`:**

- Per-cell gross and net Sharpe, cumulative return, max drawdown, turnover.
- The **rank-IC diagnostic** written to
  `data/trial_database/tree_fit_diagnostics_daily_nse500.csv` — compare against Phase 3's
  daily ≈ 0.02. If IC is similar but the Sharpe is higher, breadth did the work, which is
  exactly the thesis. If IC itself moved, something changed about the signal and that
  needs explaining.
- DSR at **N=3** and whether it clears 0.95.
- Report the **distribution across the three styles, never a single best cell.**

**The interpretation to prepare for:** the required uplift is ~1.49× over SR 0.57, i.e.
about **SR 0.85**. The breadth ceiling at Rs 1 cr is 2.68×, so it is reachable — but a
ceiling is not an achievement, and the honest outcomes are:

- **Clears 0.95 DSR** → the Phase 3 negative result was a breadth artefact. Then
  immediately do the deferred work in §7 before believing it.
- **Net-positive but below the gate** → same shape as Phase 3, at larger N. Report as
  another honest negative and stop.
- **Net-negative** → costs dominate at 500 names too, and the breadth thesis is dead.

**Add ~1.3%/yr and re-check** (the dividend guard). If the verdict is negative *and* the
add-back would have flipped it, that is a material finding — state it, do not bury it.

---

## 7. Deferred, and what would need doing before trusting a positive result

- **Per-name market impact in `tier3_execution.py`.** Keep `traded = weights.diff().abs()`
  unreduced, thread an ADV panel through, and charge
  `config.ML_CONFIG.cost.impact_bps()`. Not needed at Rs 1 cr; **required before any
  claim at Rs 10 cr+**, where the capacity curve shows 14.3% of bars exceeding 10%
  participation.
- **A participation weight cap** in `build_weight_matrix`, inserted right after the `1/k`
  assignment and before the style branch so all three styles inherit it.
- **The dividend guard**, using the `Div Yield` column already present in the downloaded
  index files.
- **Fix `run_sensitivity`** (gotcha 2) before scanning any axis.

---

## Reference

- Data build and the six defects found: [`phase4_data.md`](phase4_data.md)
- Capacity / AUM curve: [`phase4_capacity.md`](phase4_capacity.md)
- Phase 3 baseline being compared against: [`phase3_results.md`](phase3_results.md)
- Design discipline (freeze-before-test, honest N):
  [`phase3_design_requirements.md`](phase3_design_requirements.md)
