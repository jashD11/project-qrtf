# Phase 1 POC — Final Performance Results

> **These are SIMULATED results and carry no economic meaning.** Prices are two-phase
> non-stationary GBM from `src/sandbox_run/utils_simulation.py` (seed 42) over a synthetic
> 10-stock universe — not market data. **No transaction costs are charged**, so every return
> below is gross, and **no Deflated Sharpe Ratio is computed** in Phase 1. The HMM is fitted on
> all history and decoded with Viterbi global MAP (`src/sandbox_run/tier2_regime.py:54-55`), so
> these regime numbers are **in-sample**. Phase 1 was an architecture proof-of-concept: it
> verified the plumbing of Tiers 0-4, nothing about NSE. For real results see
> [`phase4c_results.md`](phase4c_results.md).

**Run date:** 2026-06-21  
**Pipeline:** `python run_pipeline.py`  
**Configurations:** 12 (4 market profiles × 3 execution styles)  
**Observation window:** 494 trading days (2024-02-08 → 2025-12-30, per the ledger index)  
**Universe:** 10 stocks with asymmetric drift profiles (3 alpha / 4 junk / 3 decadent)

---

## Asymmetric Stock Drift Profiles

Annual drift offsets layered on top of each market-phase baseline:

| Group | Stocks | Annual Drift Offset |
|---|---|---|
| Alpha | STOCK_01–03 | +0.35 |
| Junk | STOCK_04–07 | +0.05 |
| Decadent | STOCK_08–10 | −0.25 |

---

## Market Profile Definitions

| Profile | Phase 1 (days 1–250) | Phase 2 (days 251–500) |
|---|---|---|
| `stable_uptrend` | drift +0.05, vol 0.10 | drift +0.25, vol 0.10 |
| `stable_downtrend` | drift +0.05, vol 0.10 | drift −0.25, vol 0.10 |
| `stable_volatile` | drift +0.05, vol 0.10 | drift 0.00, vol 0.45 |
| `volatile_downtrend` | drift 0.00, vol 0.45 | drift −0.25, vol 0.15 |

---

## 12-Run Execution Matrix — Final Results

| Market Profile | Style | Cum. Return | Ann. Sharpe |
|---|---|---:|---:|
| stable_uptrend | LO | +51.66% | 3.632 |
| stable_uptrend | LS | +26.86% | 1.517 |
| **stable_uptrend** | **DT** | **+62.78%** | **3.214** |
| stable_downtrend | LO | −7.65% | −0.675 |
| stable_downtrend | LS | +26.82% | 1.516 |
| **stable_downtrend** | **DT** | **−0.88%** | **−0.020** |
| stable_volatile | LO | +0.37% | 0.086 |
| stable_volatile | LS | +17.16% | 0.492 |
| **stable_volatile** | **DT** | **+41.78%** | **0.847** |
| volatile_downtrend | LO | −51.31% | −1.805 |
| volatile_downtrend | LS | −21.45% | −0.350 |
| volatile_downtrend | DT | −55.32% | −1.461 |

---

## DT vs Baseline — Outperformance Analysis

| Market | LO (baseline) | DT | Delta |
|---|---:|---:|---:|
| stable_uptrend | +51.66% | +62.78% | **+11.1 pp** |
| stable_downtrend | −7.65% | −0.88% | **+6.8 pp** |
| stable_volatile | +0.37% | +41.78% | **+41.4 pp** |
| volatile_downtrend | −51.31% | −55.32% | −4.0 pp |

### Why DT Outperforms in 3 of 4 Regimes

**Mechanism — 130/30 quiet-day leverage:**  
On days the bivariate HMM classifies as State 0 (Low Vol / Quiet), `dynamic_tilt` applies a 1.3× multiplier to the long leg and a 0.3× multiplier to the short leg. This amplifies long-side exposure to alpha stocks (STOCK_01–03, structural drift up to +0.60 annualised in `stable_uptrend` P2) without increasing net cash commitment.

**Mechanism — dollar-neutral snap on Panic days:**  
On State 1 (High Vol / Panic) days, weights snap to 1.0× / −1.0×, eliminating the leverage overhang precisely when volatility spikes. This preserves capital during the days most likely to produce outsized drawdowns.

**Why `volatile_downtrend` is the exception:**  
In this profile, 482 of 494 days are classified as Quiet (State 0), so the 1.3× long multiplier is nearly always active. But all 10 stocks experience a severe Phase 2 downtrend (baseline drift −0.25); even alpha stocks carry only a +0.10 net annual drift in Phase 2. The 30% additional long exposure amplifies losses on days when the momentum signal misfires, tipping DT marginally below LO. LS performs best here because the decadent short leg (net −0.50 annual drift) generates consistent positive P&L.

---

## Ledger

Parquet vault: `data/trial_database/master_dsr_matrix.parquet`  
Shape at Phase 1 conclusion: **494 rows × 21 strategy columns**
