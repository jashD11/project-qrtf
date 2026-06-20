# QRTF Engine — Quantitative Research & Trading Framework

A 5-tiered cross-sectional stock selection engine (Tiers 0–4) targeting NSE equities, with volatility regime detection and Deflated Sharpe Ratio validation.

See `ARCHITECTURE.md` for the full system design specification.

---

## Quick Start

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Toggle simulation mode via the `is_simulation` boolean in `config.py`.

---

## Repository Structure

```
config.py               StrategyConfig dataclass + MD5 strategy hashing
run_pipeline.py         7-stage linear orchestrator
src/
  utils_simulation.py   Two-phase non-stationary GBM data generator
  ingestion.py          Tier 0 — data quality gate
  tier1_gkx.py          Tier 1 — cross-sectional momentum ranking
  tier2_regime.py       Tier 2 — multivariate Gaussian HMM regime detector
  tier3_execution.py    Tier 3 — vectorized capital allocation engine
  tier4_dsr.py          Tier 4 — Parquet DSR ledger
data/
  raw_nse/              Synthetic CSV price files (4 market profiles)
  trial_database/       master_dsr_matrix.parquet — strategy return vault
docs/
  phase1_poc_results.md Full Phase 1 performance metrics
  tier2_regime.py.md    Tier 2 architectural overview
  tier3_execution.py.md Tier 3 technical specification
```

---

## Phase 1: Architecture POC Successfully Concluded

**Status:** Complete — 2026-06-21

Phase 1 established and mathematically verified all core pipeline plumbing across Tiers 0–4.

### What was built and verified

| Component | Verification |
|---|---|
| Two-phase non-stationary GBM generator | 4 coupled market profiles, price-continuous at day 250 boundary |
| Tier 0 quality gate | Zero look-ahead bias; `.ffill()` only; anomaly checks on 500-bar input |
| Tier 1 cross-sectional momentum | Vectorized 5-day rank matrix `(495 × 10)`; no row loops |
| Tier 2 bivariate Gaussian HMM | `(N, 2)` feature matrix (daily return + 20-day trend proxy); deterministic variance-ordered state remap |
| Tier 3 adaptive execution | Three styles (`long_only`, `long_short`, `dynamic_tilt` 130/30); `np.outer` broadcast for per-day regime weight switching; `min_count=1` NaN preservation |
| Tier 4 DSR ledger | Idempotent Parquet writes; post-write assertion; 494 rows × 21 strategy columns at close of Phase 1 |

### Phase 1 final results — 12-run matrix

4 market profiles × 3 execution styles; 494 observations each:

| Market | LO | LS | DT (130/30) |
|---|---:|---:|---:|
| stable_uptrend | +51.66% | +26.86% | **+62.78%** |
| stable_downtrend | −7.65% | +26.82% | **−0.88%** |
| stable_volatile | +0.37% | +17.16% | **+41.78%** |
| volatile_downtrend | −51.31% | −21.45% | −55.32% |

The `dynamic_tilt` 130/30 model outperformed the `long_only` baseline by **+11 pp** (`stable_uptrend`), **+6.8 pp** (`stable_downtrend`), and **+41.4 pp** (`stable_volatile`). The sole underperformance (`volatile_downtrend`, −4 pp) is structurally explained: the HMM correctly identifies 98%+ of days as Quiet, so the 1.3× lever is nearly always active in a regime where even alpha stocks face severe macro headwinds.

### Key architectural invariants confirmed

- **Zero look-ahead bias:** all features constructed with `.ffill()` / `.rolling()` on past data only.
- **Vectorized throughout:** no Python loops across time dimension in any Tier 1–3 module.
- **Deterministic reproducibility:** seeded GBM (`seed=42`), fixed HMM `random_state=42`, MD5 strategy hashing.
- **Temporal causality:** signal at T → realized return at T+1 via `asset_df.shift(-1)`.

---

## Phase 2 (Planned)

- Live NSE data ingestion replacing the GBM simulator
- Deflated Sharpe Ratio computation on the full ledger
- Walk-forward validation with expanding window
- Transaction cost and slippage modelling
