# CLAUDE.md — Project QRTF Governance Contract

## Project Status
**Phase 1 POC: Complete** (2026-06-21). All Tiers 0–4 are implemented, mathematically verified, and documented. The pipeline executes 12 strategy permutations (4 market profiles × 3 execution styles) cleanly end-to-end. See `docs/phase1_poc_results.md` for final performance metrics.

## Project Context & Architecture
- **Framework Intent:** A 5-tiered Quantitative Research & Trading Engine (Tiers 0–4) optimizing cross-sectional stock selection on the NSE while tracking volatility regimes and enforcing Deflated Sharpe Ratio validation.
- **Reference Specification:** Full design details are permanently logged in the root-level `ARCHITECTURE.md` file. Review this document before changing module interfaces.
- **State Preservation:** System tracking requires saving strategy net daily returns vertically into a single local Parquet vault (`data/trial_database/master_dsr_matrix.parquet`). This ledger must remain isolated from Git branching anomalies. Current shape: **494 rows × 21 strategy columns**.

## Implemented Module Map

| Module | Tier | Status | Key Output |
|---|---|---|---|
| `src/utils_simulation.py` | — | Done | Two-phase GBM; 4 market profiles; asymmetric stock drift (3 alpha / 4 junk / 3 decadent) |
| `src/ingestion.py` | Tier 0 | Done | Quality gate; ffill-only; returns `(asset_df, index_df)` |
| `src/tier1_gkx.py` | Tier 1 | Done | 5-day cross-sectional momentum rank matrix `(495 × 10)` |
| `src/tier2_regime.py` | Tier 2 | Done | Bivariate GaussianHMM on `[daily_return, 20d_trend]`; deterministic variance-sort |
| `src/tier3_execution.py` | Tier 3 | Done | `long_only`, `long_short`, `dynamic_tilt` (130/30 quiet / dollar-neutral panic) |
| `src/tier4_dsr.py` | Tier 4 | Done | Idempotent Parquet ledger writes with post-write assertion |
| `run_pipeline.py` | Orchestrator | Done | 7-stage linear pipeline; 12-config batch loop; summary table |
| `config.py` | Config | Done | `StrategyConfig` dataclass; MD5 strategy hashing across 8 parameters |

## Strict Quantitative Principles
1. **Zero Look-Ahead Bias:** Backward-filling (`.bfill()`) or forward linear interpolation of missing prices across vectors is strictly prohibited. Use exclusively forward-filling (`.ffill()`) to preserve temporal causality.
2. **Temporal Alignment:** Enforce exactly 25 intervals/bars per trading session (09:15 to 15:30 IST) inside Tier 0 ingestion modules before piping data to downstream modules.
3. **Environment Separation:** Pure algorithmic code changes must occur on independent feature branches. The `main` branch must remain protected and stable.
4. **No Loops Over Time:** All Tier 1–3 computations are fully vectorized via `numpy`/`pandas`. `np.outer` is used for per-day regime flag broadcasting in Tier 3. `sum(axis=1, min_count=1)` is mandatory to preserve NaN on the final row.
5. **Forward Return Indexing:** Signal at date T is always realized as the T→T+1 return via `asset_df.shift(-1) / asset_df - 1`. Never index signal and return on the same date.

## Key Architectural Invariants (Do Not Break)
- **HMM state sort:** `model.covars_[:, 0, 0]` sorts on feature-0 (daily return) variance. State 0 = low vol, State 1 = high vol. Do not change the feature ordering without updating the sort.
- **Dynamic tilt regime gate:** `dynamic_tilt` does NOT apply a hard cash stop on State 1. State 1 triggers the short leg. Only `long_only` and `long_short` apply `.where(regimes == 0, 0.0)`.
- **130/30 weight construction:** `long_scale = 1.3*quiet + 1.0*panic`, `short_scale = 0.3*quiet + 1.0*panic`. These are scalar multipliers on the normalised weight vectors, not raw position sizes.
- **Ledger idempotency:** Re-running a strategy with the same `strategy_id` overwrites its column in the Parquet file. The MD5 hash guarantees ID uniqueness across all 8 config parameters.

## Primary System Shell Commands
- **Install Requirements:** `pip install -r requirements.txt`
- **Execute Pipeline Framework:** `python run_pipeline.py`
- **Toggle Testing Context:** Modify the `is_simulation` boolean in `config.py`.

## Engineering & Coding Style Guide
- **Code Distribution:** Root scripts handle pipeline routing, parameter configurations, and entry logic. Core algorithmic math and transformations must live inside the isolated modules under `/src`.
- **Typing Paradigm:** Enforce explicit Python type hints (`from typing import ...`) across all function signatures, matrix arrays, and tensor returns.
- **Memory Optimization:** Rely strictly on vectorized `numpy` and `pandas` arithmetic operations. Avoid `for` loops across timeline rows to eliminate performance bottlenecks.

## Documentation
- `ARCHITECTURE.md` — system design specification (source of truth for module interfaces)
- `docs/phase1_poc_results.md` — full Phase 1 performance metrics and analysis
- `docs/tier2_regime.py.md` — Tier 2 HMM architectural overview
- `docs/tier3_execution.py.md` — Tier 3 capital allocation technical spec
- `README.md` — project overview and Phase 1 milestone summary