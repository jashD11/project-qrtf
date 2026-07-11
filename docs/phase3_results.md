# Phase 3 Results — PRODUCTION_ML backtest record

Durable record of the PRODUCTION_ML 12-cell strategy sweep. Three run
configurations are documented: **gross** (no costs), **net** (NSE delivery costs
+ no-trade buffer), and the **daily-only** focus slice. Raw per-fold ML training
diagnostics are in §1; economic results in §2–4.

## Run parameters (common to all runs)

| | |
|---|---|
| Universe | 68 NSE tickers |
| Period | 2018-02-21 → 2025-02-13 (1,732 daily obs; 1,727 for some intraday cells) |
| Grid | 4 frequency × 3 execution_style = 12 cells |
| Features | 17 price-only (delivery family excluded — see `delivery-data-excluded`) |
| Learner | ensemble = LightGBM + XGBoost + RandomForest, raw-return predictions averaged |
| Target | `tgt_fwd_logret_1b` (1-bar forward log return) |
| Walk-forward | TRAIN 504 trading days / PREDICT 63 days → ~28–32 folds per frequency |
| Book | decile 10% long + 10% short (variable-k `1/k` weights) |
| De-risk gate | `panic_threshold = 0.85` (~15–21% Panic days), `hmm_states = 2` (frozen) |
| Buffer | `rebalance_buffer_mult = 2.0` (net runs only; gross = raw deciles) |
| Costs (net) | NSE delivery ~14.7 bps one-way (STT/exch/SEBI/stamp/GST/brokerage) + 50 bps/yr short borrow; slippage deferred |
| DSR gate | threshold 0.95, N = trials searched |

Peak RSS (per-frequency isolated process): daily 0.69 GB · 60min 1.5 GB · 30min
2.5 GB · 15min 3.6 GB (of 16 GB).

---

## 1. ML training diagnostics — per-fold cross-sectional rank IC

Predictive skill of the trees, measured as the daily cross-sectional Spearman IC
between predicted alpha and realized forward return (read-only; never used for
selection). **Tier 1 is deterministic (seed 42) and cost/buffer-independent, so
these are identical across the gross and net runs.** Metrics: `mean_IC`, `IC_IR`
= mean/std, `hit%` = share of days IC > 0. (At intraday, `IC_t` is inflated by
bar autocorrelation — read `mean_IC`/`hit%`, not `IC_t`.)

### 15-min (n_days = 47,168)
| model | mean_IC | IC_IR | hit% |
|---|---|---|---|
| lgbm | 0.0864 | 0.578 | 72.1% |
| xgb | 0.0824 | 0.565 | 71.5% |
| rf | 0.0734 | 0.514 | 69.9% |
| **ensemble** | **0.0829** | 0.567 | 71.5% |

### 30-min (n_days = 23,478)
| model | mean_IC | IC_IR | hit% |
|---|---|---|---|
| lgbm | 0.0665 | 0.425 | 67.0% |
| xgb | 0.0625 | 0.416 | 66.5% |
| rf | 0.0540 | 0.370 | 64.5% |
| **ensemble** | **0.0634** | 0.419 | 66.5% |

### 60-min (n_days = 11,636)
| model | mean_IC | IC_IR | hit% |
|---|---|---|---|
| lgbm | 0.0604 | 0.387 | 65.3% |
| xgb | 0.0572 | 0.383 | 65.0% |
| rf | 0.0448 | 0.313 | 62.3% |
| **ensemble** | **0.0569** | 0.379 | 65.1% |

### daily (n_days = 1,732)
| model | mean_IC | IC_IR | IC_t | hit% |
|---|---|---|---|---|
| lgbm | 0.0207 | 0.131 | 5.44 | 56.2% |
| xgb | 0.0212 | 0.139 | 5.80 | 56.4% |
| rf | 0.0166 | 0.108 | 4.50 | 54.6% |
| **ensemble** | **0.0234** | 0.147 | 6.12 | 55.8% |

**Read:** genuine, statistically strong predictive skill that **decays monotonically
with horizon** (ensemble IC 0.083 → 0.063 → 0.057 → 0.023). LightGBM leads at
intraday; the ensemble wins at daily (IC_t 6.12). The trees really rank stocks —
the signal is real. Whether it is *tradeable* is answered in §3.

---

## 2. GROSS N=12 — no costs, raw deciles (reference only)

Annualized Sharpe over daily-compounded returns; DSR at N=12, benchmark
SR\* = 9.17, cross-trial Sharpe std 0.347.

| freq | style | turnover/bar | gross CumRet | SR_ann | DSR | verdict |
|---|---|---|---|---|---|---|
| 15min | long_only | 1.22 | 4.8e12 % | 11.42 | 1.000 | **PASS** |
| 15min | long_short | 2.29 | 1.1e26 % | 16.29 | 1.000 | **PASS** |
| 15min | dynamic_tilt | 2.41 | 2.3e27 % | 17.16 | 1.000 | **PASS** |
| 30min | long_only | 1.16 | 2.7e6 % | 6.27 | 0.000 | fail |
| 30min | long_short | 2.11 | 1.9e13 % | 11.50 | 1.000 | **PASS** |
| 30min | dynamic_tilt | 2.25 | 3.3e13 % | 11.32 | 1.000 | **PASS** |
| 60min | long_only | 1.14 | 5,399 % | 3.17 | 0.000 | fail |
| 60min | long_short | 2.07 | 5.9e7 % | 8.04 | 0.001 | fail |
| 60min | dynamic_tilt | 2.22 | 1.9e7 % | 6.66 | 0.000 | fail |
| daily | long_only | 0.89 | 1,604 % | 1.62 | 0.000 | fail |
| daily | long_short | 1.80 | 1,800 % | 1.81 | 0.000 | fail |
| daily | dynamic_tilt | 1.87 | 4,577 % | 1.74 | 0.000 | fail |

**Passing 5/12** — all intraday. **These are cost mirages**: the astronomical
cumulatives are the artifact of compounding a tiny, consistent per-bar edge across
tens of thousands of zero-cost trades. The daily cells (the only economically
interpretable gross numbers) fail here only because the deflation benchmark is set
by the inflated intraday cells sharing the N=12 family. Costs (§3) reveal the truth.

---

## 3. NET N=12 — NSE delivery costs (~14.7 bps one-way) + buffer 2×

The real economic verdict. DSR at N=12, benchmark SR\* = 12.02, cross-trial
Sharpe std 0.455.

| freq | style | turnover/bar (vs gross) | cost drag /yr | net CumRet | SR_ann | DSR |
|---|---|---|---|---|---|---|
| daily | **long_only** | 0.59 (0.89) | 21.9% | **+97.4%** | **+0.49** | 0.000 |
| daily | dynamic_tilt | 1.23 (1.87) | 45.8% | +1.4% | +0.18 | 0.000 |
| daily | long_short | 1.21 (1.80) | 45.2% | −58.7% | −0.37 | 0.000 |
| 60min | long_only | 0.88 (1.14) | 226% | −99.998% | −6.95 | 0.000 |
| 60min | long_short | 1.44 (2.07) | 373% | −99.999% | −6.34 | 0.000 |
| 60min | dynamic_tilt | 1.61 (2.22) | 418% | −100% | −7.39 | 0.000 |
| 30min | long_only | 0.91 (1.16) | 439% | −100% | −11.32 | 0.000 |
| 30min | long_short | 1.50 (2.11) | 719% | −100% | −10.29 | 0.000 |
| 30min | dynamic_tilt | 1.69 (2.25) | 812% | −100% | −12.43 | 0.000 |
| 15min | long_only | 1.04 (1.22) | 961% | −100% | −17.48 | 0.000 |
| 15min | long_short | 1.79 (2.29) | 1,655% | −100% | −17.61 | 0.000 |
| 15min | dynamic_tilt | 1.97 (2.41) | 1,820% | −100% | −21.11 | 0.000 |

**Passing 0/12.** Complete gross→net inversion: net Sharpe now falls monotonically
with frequency. The gross Sharpe-16/17 intraday "winners" are **−100% net** —
total wipeout from cost drag of hundreds-to-thousands of %/yr on their turnover.
The buffer cut turnover ~33% at daily, ~15–22% intraday (the intraday signal moves
too fast for the buffer to help much). **Only daily long_only stays net-positive**
(+97% over ~7yr ≈ ~10%/yr).

---

## 4. DAILY-ONLY N=3 — the focus slice (net)

Intraday dropped on structural grounds (turnover × cost is deterministically fatal,
not a noisy outcome), so the honest headline family is the 3 daily style cells.
DSR at N=3, benchmark SR\* = 0.37 (un-polluted), cross-trial Sharpe std 0.027.

| style | net SR_ann | DSR | PSR₀ | skew | kurt | minTRL | verdict |
|---|---|---|---|---|---|---|---|
| **long_only** | **+0.49** | **0.626** | 0.902 | −0.20 | 5.40 | 45,453 | fail |
| dynamic_tilt | +0.18 | 0.312 | 0.685 | −0.06 | 4.43 | n/a | fail |
| long_short | −0.37 | 0.027 | 0.169 | 0.06 | 4.73 | n/a | fail |

**Passing 0/3.** With the deflation honest (SR\* 0.37 vs the polluted 12.02),
daily long_only's DSR rises to 0.626 — it fails now for the *right* reason: the raw
edge is genuine but **too weak** (`minTRL = 45,453` ≈ ~180 years of daily data to
reach credibility at SR 0.49).

---

## Key finding & next step

The signal has **real cross-sectional predictive skill** (§1, IC 0.02–0.09) but the
per-trade edge at a 1-day horizon is too small to survive **realistic NSE delivery
costs**. Gross→net destroys every intraday cell (−100%) and takes daily long_only
from SR 1.62 → 0.49. **Cost is essentially the entire gap** — the lever is turnover
reduction.

Highest-leverage next experiment: **switch the tree target `tgt_fwd_logret_1b` →
`tgt_fwd_logret_5b`** (5-day horizon → slower signal → lower turnover → less cost
drag). Note: costs here are a **lower bound** (slippage still deferred, design doc
§4.3), so net results can only worsen.

## Artifacts (gitignored `data/trial_database/`)

| File | Contents |
|---|---|
| `production_dsr_matrix.parquet` | current ledger — daily-only N=3 (net) |
| `production_dsr_matrix_net12.parquet` | archived full 12-cell NET ledger |
| `production_dsr_matrix_dsr_gate.csv` | latest DSR gate output (N=3 daily) |
| `tree_fit_diagnostics_{15min,30min,60min,daily}.csv` | per-model rank-IC diagnostics (§1) |
