# QRTF — Cross-Sectional Equity ML on NSE, Tested Against Real Costs

**Question:** can a walk-forward tree ensemble that ranks ~500 NSE stocks cross-sectionally
earn money after realistic Indian transaction costs and real short-borrow availability?
**Answer, after four phases: no** — the ranker has genuine out-of-sample skill (rank-IC 0.051,
*t* = 22.5), and that skill is still too thin to pay ~26 bps a side at 0.24–0.72 daily
turnover. This repository is the record of establishing that carefully enough to believe it.

> **No live trading.** "Live", `PRODUCTION_ML` and `STRAT_LIVE_NSE_*` throughout this repo mean
> **real historical market data**, as opposed to the synthetic Phase 1 simulator. No component
> has ever connected to a broker, placed an order, or run against paper or real money. There is
> no broker SDK, order path, or market-data socket in the tree.

---

## Headline results

Six frozen cells, committed before the run, on the 500-name point-in-time panel.
**2,337 out-of-sample daily bars, 2016-01-21 → 2025-06-27.** Deflated Sharpe Ratio deflated
against **N = 30** searched trials (`config.TRIAL_LEDGER`); pass line 0.95, frozen a priori.

| target | style | Sharpe gross | **Sharpe net** | turnover/bar | **DSR** | verdict |
|---|---|---|---|---|---|---|
| 21-day | **long_only** | 1.120 | **+0.298** | 0.242 | **0.01650** | fail |
| 21-day | dynamic_tilt_slb | 1.308 | +0.212 | 0.370 | 0.00894 | fail |
| 5-day | long_only | 1.660 | +0.178 | 0.446 | 0.00681 | fail |
| 5-day | dynamic_tilt_slb | 2.014 | −0.049 | 0.717 | 0.00092 | fail |
| 21-day | long_short_slb | 0.958 | −0.567 | 0.338 | 2.0e−06 | fail |
| 5-day | long_short_slb | 1.882 | −0.987 | 0.637 | 3.0e−09 | fail |

Source: [`data/trial_database/phase4c_dsr_matrix_dsr_gate.csv`](docs/phase4c_results.md) (Sharpe,
DSR) and `phase4c_dsr_matrix_execution_diagnostics.csv` (gross, turnover). Full record:
[`docs/phase4c_results.md`](docs/phase4c_results.md).

**0 of 6 pass, and `minTRL` is undefined for every cell** — none clears the deflated benchmark
at *any* track length, so more history cannot rescue them.

**Cost basis for every net figure.** 14.6558 bps per side statutory NSE delivery — brokerage
3.0 + STT 10.0 (both sides) + exchange 0.30 + SEBI 0.01 + stamp duty 1.5 (buy only) + 18% GST
on the brokerage/exchange/SEBI base — **plus** a measured Corwin-Schultz half-spread and
square-root market impact at Rs 1 crore, giving **24.5–26.1 bps/side realised**, plus tiered
short borrow. Itemised in `config.py::_NSECostConfig`.

### What each correction was worth

The same strategy, as three successive corrections were applied. This is the result:

| applied to `long_short` | Sharpe | DSR | source |
|---|---|---|---|
| as first measured (N = 3, no spread, unrestricted short leg) | 1.778 | 0.99999 | `production_dsr_matrix_dsr_gate.csv` |
| + honest N = 30 and Lo autocorrelation correction | 1.35 | 0.986 | [`phase4c_results.md`](docs/phase4c_results.md) §5 |
| + measured half-spread and market impact | **−0.23** | — | [`phase4c_results.md`](docs/phase4c_results.md) §3 |
| + short leg restricted to borrowable names | **−0.987** | 3.0e−09 | `phase4c_dsr_matrix_dsr_gate.csv` |

Three findings worth more than the strategy:

- **The edge is real, not a microstructure artefact.** Rank-IC is flat across liquidity
  quintiles (0.041 / 0.046 / 0.041 / 0.039 / 0.048), which refutes the leading hypothesis that
  the skill lived in the illiquid tail. It fails on cost, not on illusion.
- **~78% of the intended short book was never borrowable.** Restricting to the point-in-time
  F&O-eligible set cuts the *gross* Sharpe from 4.20 to 1.882 — over half the raw edge was in
  names that could not have been shorted.
- **The signal decays.** Rank-IC *t*-statistic falls from ~10 (2016–2019) to **1.21** by 2025 —
  no longer distinguishable from zero.

---

## Limitations

- **Single market, one regime sample.** ~9.4 years out-of-sample on NSE only, spanning roughly
  three stress episodes. The DSR corrects for multiple testing, not for having lived through
  one history.
- **17 price-only features.** No fundamentals, no analyst or macro data. NSE delivery metrics
  are implemented but excluded from production runs (the archive starts 2020 and would truncate
  a 12-year panel). This is the thinnest corner of the Gu-Kelly-Xiu information set.
- **The half-spread is estimated, not observed.** Bhavcopy carries no quotes, so it is inferred
  from daily OHLC by two published estimators carried as a band — Corwin-Schultz 5.97 bps,
  Abdi-Ranaldo 26.07 bps pooled. The headline uses the *optimistic* one deliberately.
- **Borrow availability is an upper bound.** F&O eligibility proxies SLB because NSE publishes
  no historical SLB-eligibility archive. Real borrow is thinner, so the short-leg result is
  charitable.
- **`impact_coef = 0.5` is a modelling choice, not a measurement** — the one discretionary knob
  in an otherwise statutory cost stack, and it was never sensitivity-tested. Every other line
  item is a citable rate.

---

## Repo map

```
config.py                     MODE toggle, StrategyConfig (MD5-hashed identity), ML_CONFIG,
                              FREQ_REGISTRY, the frozen grids, and TRIAL_LEDGER (N = 30)
run_pipeline.py               Phase 1 orchestrator  — synthetic sandbox, 7 linear stages
run_pipeline_ml.py            Phase 2-4c orchestrator — compute-sharing sweep

src/sandbox_run/              Phase 1: GBM simulator, momentum ranker, in-sample HMM
src/production_ml/            the live-data stack
  data_scraping.py            ingestion; OHLCV from local CSV, delivery metrics scraped
  feature_creator.py          19 features (17 in use), rank-normalised to [-1, +1] per date
  tier1_trees.py              LightGBM + XGBoost + RandomForest ensemble, walk-forward
  tier2_regime.py             causal regime detector; forward-filter decode, no backward pass
  tier3_execution.py          books, no-trade buffer, NSE cost stack, participation cap
  tier4_dsr_gate.py           PSR, Deflated Sharpe, minTRL, Lo variance-ratio correction
  wf_cache.py                 disk cache for the walk-forward, keyed on a column-scoped digest
  download_nse.py -> consolidate.py -> resample_bars.py     one-time bar construction
src/phase4_data/              point-in-time NSE panel from raw bhavcopy
  bhavcopy_download.py        archive acquisition (equity, index, F&O)
  bhavcopy_panel.py           ISIN-keyed panel, corporate actions, lifecycle
  bhavcopy_validate.py        six acceptance gates, three blocking
  universe.py                 survivorship-free quarterly top-500 by median rupee turnover
  liquidity.py spread.py      ADV / trailing vol; Corwin-Schultz + Abdi-Ranaldo half-spread
  shortable.py                point-in-time borrowable set from traded stock futures
  dividends.py capacity.py    measured dividend yield; capacity curve
  phase4c_diagnostics.py      read-only decompositions (IC by liquidity, by year, coverage)

scripts/                      deck builders, cost sanity check, Phase 5 falsification tests
docs/                         17 design and result documents, plus the slide deck
data/  logs/                  gitignored — exist only on the machine that ran the pipeline
```

26 modules under `src/`. `MODE` in `config.py` selects the stack; the two packages are kept
isolated on purpose, so a file may be duplicated rather than shared across them.

---

## Setup and reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # Python 3.12
```

> **Nothing reproduces from a bare clone.** `data/` is gitignored — the panels, ledgers and
> caches exist only on the machine that ran the pipeline, and the raw source is ~2,850 NSE
> bhavcopy archives that must be re-downloaded (~2.5 h for the F&O set alone). The commands
> below are the real provenance of each result, not a quick-start.

**Which command produced which result:**

| result file | command | runtime |
|---|---|---|
| `phase4c_dsr_matrix*.{parquet,csv}` — the headline | `python run_pipeline_ml.py --phase4c` | ~10 min warm cache |
| `production_dsr_matrix*` — Phase 4b, 500 names | `python run_pipeline_ml.py --frequencies daily_nse500 --target tgt_fwd_logret_5b` | ~1.5 h cold |
| `production_dsr_matrix_net12.parquet` — the 12-cell frequency sweep | `python run_pipeline_ml.py` | hours; 15-min peaks 3.6 GB RSS |
| `production_sensitivity_dsr_matrix.parquet` | `python run_pipeline_ml.py --sensitivity rebalance_buffer_mult` | ~1 h |
| `tree_fit_diagnostics_*.csv` | written by any run, per frequency and target | — |
| `master_dsr_matrix.parquet` — Phase 1 synthetic | `python run_pipeline.py` | < 1 min |
| §6 read-only diagnostics | `python src/phase4_data/phase4c_diagnostics.py --run` | ~2 min |
| Phase 5 falsification tables | `python scripts/phase5_falsification.py` | ~2 min, writes nothing |
| the slide deck | `python scripts/build_deck.py` / `build_beamer.py` | ~1 min |

**Panel builds, in order, before any run:**

```bash
python src/phase4_data/bhavcopy_download.py --kinds eq index   # ~2 h
python src/phase4_data/bhavcopy_download.py --kinds fo         # ~2.5 h
python src/phase4_data/bhavcopy_panel.py    --build
python src/phase4_data/bhavcopy_validate.py                    # 3 blocking gates
python src/phase4_data/universe.py          --build
python src/phase4_data/liquidity.py         --build
python src/phase4_data/spread.py            --build
python src/phase4_data/shortable.py         --build
python src/phase4_data/dividends.py         --build
```

`scripts/watch_download.sh` snapshots download progress and exits with the job.

**Determinism.** All three learners take `random_state=42`; the HMM is seeded identically; the
GBM simulator uses `seed=42`; `OMP_NUM_THREADS=1` pins thread count so tree sums are
order-stable; strategy identity is an MD5 of every config field, so changing any knob produces
a new ledger column rather than overwriting one. `QRTF_WF_CACHE=0` forces a cold Tier 1 fit.

**Tests.** There is **no test suite.** Verification is 21 module `__main__` dry-runs that
synthesise a schema-correct panel and assert invariants (decile widths, the strict [−1, +1]
normalisation band, cost frames NaN-free), plus a bit-for-bit regression gate: with every Phase
4c flag at its default, Phase 3 and Phase 4b must reproduce exactly (`max abs diff = 0.0`)
before any new number is trusted. Run one with `python -m src.production_ml.tier3_execution`.

---

## Reports

Read in this order:

| document | what it settles |
|---|---|
| [`phase4c_results.md`](docs/phase4c_results.md) | **the current headline** — all six cells fail, and why |
| [`phase5_plan.md`](docs/phase5_plan.md) | two proposed revivals falsified before being built |
| [`phase3_results.md`](docs/phase3_results.md) | the 12-cell frequency sweep; gross→net inversion |
| [`phase4_results.md`](docs/phase4_results.md) | superseded — the result Phase 4c overturned |
| [`phase4c_plan.md`](docs/phase4c_plan.md) · [`phase4_plan.md`](docs/phase4_plan.md) · [`phase4_run_plan.md`](docs/phase4_run_plan.md) | plans, written before their runs |
| [`phase2_features.md`](docs/phase2_features.md) | the 19-feature reference |
| [`phase2_design_decisions.md`](docs/phase2_design_decisions.md) · [`phase3_design_requirements.md`](docs/phase3_design_requirements.md) | design rules, including the freeze-before-test discipline |
| [`phase3_deferred_hmm.md`](docs/phase3_deferred_hmm.md) | why `hmm_states` is frozen at 2 |
| [`phase4_data.md`](docs/phase4_data.md) · [`phase4_capacity.md`](docs/phase4_capacity.md) | panel construction; capacity curve |
| [`phase3_presentation_notes.md`](docs/phase3_presentation_notes.md) | the Gu-Kelly-Xiu comparison in full |
| [`phase1_poc_results.md`](docs/phase1_poc_results.md) | Phase 1 **synthetic** sandbox results |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Phase 1 sandbox design (historical) |

Slide walkthrough: [`docs/presentation/qrtf_deck.pdf`](docs/presentation/qrtf_deck.pdf) — 29
frames, generated from the ledgers by `scripts/build_beamer.py`. **Built 2026-09-10, so it
predates the Phase 5 regime findings**; its framing of the regime layer is stale.

---

## Status

**Phase 4c closed as a negative result** and is the current headline: 0 of 6 frozen cells clear
the DSR gate, best 0.0165 against 0.95. The breadth thesis (68 → 500 names) is closed —
expanding the universe delivered the gross Sharpe it promised while raising turnover and
reaching into wider spreads in the same motion.

**Phase 5 was redirected before it was built.** Two of three proposed revivals were falsified
by cheap kill-tests: a per-name trend overlay loses Sharpe monotonically as the tilt sharpens
(4.97 → 3.45 gross, and it is not a beta story — measured book beta is ≈0), and short-leg
gating has almost nothing to modulate, since the borrowable short leg earns ~5.0%/yr gross
against ~5.46%/yr to run it. What remains open is **cost-aware portfolio construction** —
pricing the spread into the entry decision rather than charging it afterwards — against a
15.96%/yr trade drag on the one cell with positive net Sharpe.

Also open, and recorded rather than quietly fixed: the regime layer is structurally
decorative. `RegimeResult.states`, `.probs` and `.stress` have no consumers; only the `panic`
gate is read, and it correlates 0.907 with a plain 21-day realised-volatility z-score. Any
estimator upgrade is a no-op until Tier 3 is rewired.

Negative results are the deliverable here. The pipeline works; the strategy does not pay.
