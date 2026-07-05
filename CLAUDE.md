# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full 12-run pipeline (4 market profiles × 3 execution styles)
python run_pipeline.py

# Regenerate synthetic CSV data only
python src/utils_simulation.py
```

There are no tests or linting configs in this repo. All verification is done by running `run_pipeline.py` and inspecting the printed metrics and Parquet ledger shape.

Toggle simulation mode via `is_simulation` in the `StrategyConfig` passed to `run_single_pipeline`. All current runs use `is_simulation=True`.

## Architecture

The pipeline is a **7-stage linear orchestrator** (`run_pipeline.py`) that runs one `StrategyConfig` at a time. There is no framework — stages call module functions directly.

```
Stage 1  Build StrategyConfig batch
Stage 2  Generate synthetic GBM data → data/raw_nse/mock_daily_<market>.csv
Stage 3  Tier 0 ingestion & quality gate (ingestion.py)
Stage 4  Tier 1 momentum alpha ranking (tier1_gkx.py)
Stage 5  Tier 2 Gaussian HMM regime detection (tier2_regime.py)
Stage 6  Tier 3 vectorized execution (tier3_execution.py)
Stage 7  Tier 4 append to Parquet DSR ledger (tier4_dsr.py)
```

### `config.py` — StrategyConfig

Dataclass that fully specifies one strategy trial. The `strategy_id` property generates a deterministic human-readable ID with an MD5 hash suffix, e.g. `STRAT_STABLE_UPTREND_DT_L5_HMM2_<hash8>`. This ID is used as the column name in the Parquet ledger — changing any config field produces a new column.

### `src/utils_simulation.py` — Non-stationary GBM generator

Produces 500-day price paths for 10 stocks + 1 index. The timeline is split at day 250 (PHASE_SPLIT): Phase 1 and Phase 2 each have independent drift/vol parameters from `COUPLED_PARAMS`. Price continuity at the boundary is enforced by seeding Phase 2 from the terminal log-price of Phase 1. The RNG is seeded (`seed=42`) for reproducibility.

Stocks have hard-coded asymmetric annual drift offsets (`STOCK_DRIFT_OFFSETS`): STOCK_01–03 are "alpha" (+0.35), STOCK_04–07 are "junk" (+0.05), STOCK_08–10 are "decadent" (−0.25). These offsets are layered on top of the market-phase baseline drift.

### `src/ingestion.py` — Tier 0 quality gate

Validates columns, rejects negative/zero prices, and applies **only `.ffill()`** (no `.bfill()`). This is a strict look-ahead-bias invariant: never backfill. Returns `(asset_df, index_df)` split.

### `src/tier1_gkx.py` — Tier 1 momentum

Computes `(close_t / close_{t-lookback}) - 1` fully vectorized, then cross-sectionally ranks stocks per day (rank 1 = highest momentum). Produces an integer rank matrix shape `(tradeable_days, 10)`. No Python loops over the time dimension.

### `src/tier2_regime.py` — Tier 2 HMM regime detection

Fits a bivariate `GaussianHMM(n_states=2, covariance_type='full')` on a 2-column feature matrix:
- Feature 0: daily log returns of INDEX_CLOSE (short-term variance)
- Feature 1: 20-day rolling cumulative log returns (macro trend proxy)

After fitting, raw HMM states are **deterministically remapped** by ascending variance (`model.covars_[:, 0, 0]`) so State 0 is always the low-volatility regime and State 1 is always high-volatility, regardless of which label EM assigned internally. Currently only `n_states=2` is supported; other values raise `NotImplementedError`.

### `src/tier3_execution.py` — Tier 3 capital allocation

Three execution styles, all fully vectorized with `np.outer` broadcasting (no loops over days):

| Style | State 0 (Low Vol) | State 1 (High Vol) |
|---|---|---|
| `long_only` | +1/top_n for top-N stocks | Flat (cash) |
| `long_short` | +1/top_n long, −1/bottom_n short | Flat (cash) |
| `dynamic_tilt` | 130/30: 1.3× long, 0.3× short | Dollar-neutral: 1.0× long, 1.0× short |

`dynamic_tilt` never goes to cash — it uses the Panic state to activate the protective short leg at dollar-neutral rather than exit. Signal at day T realizes as forward return T→T+1 via `asset_df.shift(-1)`. The `min_count=1` parameter on `.sum()` preserves the last-day NaN so `dropna()` removes it cleanly without collapsing it to 0.0.

### `src/tier4_dsr.py` — Tier 4 Parquet ledger

Appends or overwrites one column per `strategy_id` in `data/trial_database/master_dsr_matrix.parquet`. The ledger is **idempotent**: re-running with the same config overwrites the existing column. A post-write assertion verifies the column is readable. The Parquet file is gitignored.

## Key invariants

- **Zero look-ahead bias**: `.ffill()` only, never `.bfill()`. All rolling features use past bars only.
- **Vectorized throughout**: no Python loops over the time dimension in Tiers 1–3.
- **Temporal causality**: signal at T → realized return at T+1 via `shift(-1)`.
- **Deterministic reproducibility**: GBM seed=42, HMM `random_state=42`, MD5 strategy hashing.

## In-progress work (feature/gkx-trees-pipeline branch)

Phase 2 adds a live-data, tree-based alternative to the Phase 1 GBM/GKX simulation stack. Progress by file:

- **`config.py` — updated (complete).** Added a global `MODE` toggle (`"SANDBOX"` keeps all Phase 1 GBM simulation paths active and unchanged; `"PRODUCTION_ML"` activates the Phase 2 live-data tree pipeline), live-data CSV paths (`DATA_DAILY_CSV`, `DATA_15MIN_CSV`), and multi-model tree hyperparameter configs (`_LGBMConfig`, `_XGBConfig`, `_RFConfig`) bundled into `ML_CONFIG`. The existing `StrategyConfig` dataclass is unchanged.

- **`src/data_scraping.py` — fully implemented (complete, 241 lines).** Phase 2 data ingestion layer:
  - `load_local_csv_data()` reads the daily and 15-min OHLCV CSVs into a MultiIndex `('date', 'ticker')` daily frame and a `DatetimeIndex` intraday frame.
  - `scrape_nse_delivery_data()` fetches NSE full bhavcopies via jugaad-data (`NSEArchives.full_bhavcopy_raw`, deferred import so SANDBOX mode never requires the package), one request per trading day, restricted to EQ series, with holiday/network errors caught and skipped. Extracts `DeliveryQty` / `DeliveryPct`.
  - `augment_dataset()` left-joins delivery metrics onto the daily price frame and forward-fills within each ticker (leading NaNs zero-filled), preserving the no-backfill look-ahead invariant.
  - A `__main__` dry-run harness exercises all three steps over a January 2024 window.

- **`src/feature_creator.py` — pending (empty stub).** Will build the richer Phase 2 feature set.
- **`src/tier1_trees.py` — pending (empty stub).** Will provide the tree-based alternative to the GKX momentum alpha in Tier 1.
