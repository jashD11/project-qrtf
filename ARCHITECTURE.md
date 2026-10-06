# QRTF — Phase 1 Sandbox Architecture (historical)

> **Scope.** This document describes the **Phase 1 synthetic-data sandbox** only — the
> `MODE = "SANDBOX"` path, driven by `run_pipeline.py` over GBM-simulated prices. It is kept
> as the original design record.
>
> It does **not** describe the live-data stack that carries every result in
> [`docs/`](docs/) — that is the `MODE = "PRODUCTION_ML"` path (tree ensemble, causal regime
> detector, NSE cost stack, DSR gate). For the current architecture see the repo map and
> report index in [`README.md`](README.md).
>
> The module paths below were rewritten when `src/` was split into `src/sandbox_run/` and
> `src/production_ml/`; they are corrected here, but the tier descriptions are as originally
> written.

## 1. System Topology Overview
The pipeline executes sequentially across three decoupled data-routing tiers. All data operations are strictly vectorized across the stock dimension, passing high-dimensional numpy arrays and pandas matrices to eliminate runtime looping overhead.

## 2. Tier Specifications

### Tier 1: Alpha Ranking Engine (`src/sandbox_run/tier1_gkx.py`)
* **Core Logic:** Cross-sectional momentum extraction.
* **Input:** Raw asset close prices spanning $N$ days across $M$ tickers.
* **Vectorized Operation:** Computes 5-day rolling log returns across the entire asset matrix simultaneously.
* **Output:** Daily rankings matrix. The Top $N$ performing assets form the candidate pool for long positions; the Bottom $N$ assets form the candidate pool for short positions.

### Tier 2: Regime Detection Engine (`src/sandbox_run/tier2_regime.py`)
* **Core Logic:** Non-stationary multi-variate variance and structural trend clustering.
* **Feature Grid:** A 2-dimensional feature matrix of shape `(N_days, 2)` built via `np.column_stack`:
  1. *Feature 0:* Daily index log returns (captures short-term variance/volatility).
  2. *Feature 1:* 20-day rolling cumulative index returns (acts as a macro-structural trend proxy).
* **Model:** Multi-variate `GaussianHMM(n_states=2, covariance_type='full')`.
* **State Boundaries:** Strict deterministic sorting enforced post-fit using the variance of the daily returns dimension (`model.covars_[:, 0, 0]`):
  * **State 0 (Quiet/Safe):** Lowest mathematical variance cluster.
  * **State 1 (Panic/Volatile):** Highest mathematical variance cluster.

### Tier 3: Execution & Vector Balancing Engine (`src/sandbox_run/tier3_execution.py`)
* **Core Logic:** State-dependent dynamic capital allocation.
* **Supported Styles:** 1. `long_only`: Allocates +100% weight to Top $N$ assets during State 0; pivots to 100% Cash during State 1 Panic states.
  2. `long_short`: Rigid dollar-neutral deployment (+100% Top $N$ assets / -100% Bottom $N$ assets).
  3. `dynamic_tilt`: Adaptive institutional leverage framework controlled by state vector masks:
     * **State 0 (Quiet Condition):** Deploys an institutional **130/30 leveraged structure** (Scales long weights to sum to +1.30 and short weights to sum to -0.30 via scalar masks). Retains 100% Net long market exposure while actively harvesting dispersion alpha using short proceeds.
     * **State 1 (Panic Condition):** Snaps instantly to a strict **Dollar-Neutral Split (+100% Long / -100% Short, Net 0% Exposure)** to neutralize systemic beta tail-risk.

---

## 3. Data Vault Spec (`run_pipeline.py`)
* **Storage Structure:** Unified local binary Parquet schema.
* **Dimensions:** 21 Columns $\times$ 494 Rows.
* **Tracking Rules:** Local binary ledger is permanently isolated from version control via `.gitignore` to prevent historical repository bloat. Performance verification metrics are hard-baked into text logs for presentation reproducibility.

---

## 4. What this document does not cover

The Phase 1 HMM described in Tier 2 is fitted on **all** observations and decoded with
Viterbi global MAP (`src/sandbox_run/tier2_regime.py:54-55`) — it is in-sample, and no
Phase 1 regime number should be cited as out-of-sample. The live stack fixed this: it refits
walk-forward on a trailing window and decodes by causal forward filtering
(`src/production_ml/tier2_regime.py:221`).

Phase 1 also charges **no transaction costs** and computes **no Deflated Sharpe Ratio** —
`src/sandbox_run/tier4_dsr.py` only writes the return ledger. The DSR gate is
`src/production_ml/tier4_dsr_gate.py`.
