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

> **Artifact note.** `tree_fit_diagnostics_daily.csv` was overwritten on 2026-08-06 by a
> 5-day-target run and is now byte-identical to `tree_fit_diagnostics_daily_5b.csv`, so the
> 1-day daily row above **no longer has a backing artifact** (the 15/30/60-min rows still
> match theirs exactly). The numbers are left as published rather than silently dropped.
> Regenerate with:
>
>     python run_pipeline_ml.py --frequencies daily --target tgt_fwd_logret_1b --skip-dsr
>
> The §4 *returns* for the 1-day target are unaffected — they still verify against
> `production_dsr_matrix_daily1b.parquet`.

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

**Cost basis for every net figure below:** 14.6558 bps per side, statutory NSE delivery
only — brokerage 3.0 + STT 10.0 (both sides) + exchange 0.30 + SEBI 0.01 + stamp duty 1.5
(buy side only) + 18% GST on the brokerage/exchange/SEBI base, plus 50 bps/yr borrow on the
short leg. **Flat slippage is 0.0 and market impact is not charged here** (`config.py`
`_NSECostConfig`), so these are a lower bound; Phase 4c charges the measured half-spread and
√-law impact and reaches 24.5-26.1 bps/side.

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

## 5. DAILY-ONLY N=3 — 5-day target (`tgt_fwd_logret_5b`, net)

The turnover-reduction experiment: retrain the trees on the **5-day** forward
log-return instead of 1-day. Everything else identical (daily, buffer 2×, NSE
delivery costs, DSR N=3). A slower learning target → slower signal → longer holds
→ lower turnover → less cost erosion. DSR benchmark SR\* = 0.33, cross-trial
Sharpe std 0.024.

| style | turnover/bar (vs 1-day) | cost drag /yr (vs 1-day) | net CumRet (vs 1-day) | net SR_ann (vs 1-day) | DSR (vs 1-day) | PSR₀ | minTRL |
|---|---|---|---|---|---|---|---|
| **long_only** | 0.32 (0.59) | 11.7% (21.9%) | **+123.1%** (+97.4%) | **+0.57** (+0.49) | **0.737** (0.626) | 0.932 | 11,615 |
| dynamic_tilt | 0.64 (1.23) | 24.0% (45.8%) | +36.1% (+1.4%) | +0.30 (+0.18) | 0.475 (0.312) | 0.786 | n/a |
| long_short | 0.66 (1.21) | 24.9% (45.2%) | −42.6% (−58.7%) | −0.19 (−0.37) | 0.090 (0.027) | 0.314 | n/a |

**Every cell improved.** The 5-day target ~**halved turnover** (LO 0.59→0.32,
−46%) and ~halved cost drag on all three styles, lifting net Sharpe across the
board. Daily long_only: SR +0.49→**+0.57**, DSR 0.626→**0.737**, and minTRL fell
from ~45,453 to **11,615** days (~180yr → ~46yr to credibility) — a materially
stronger, more cost-robust edge, though still **short of the 0.95 gate**.

The 5-day **raw IC is marginally lower** (ensemble daily mean_IC 0.019 vs 0.023 at
1-day; its IC_t is inflated by overlapping 5-day windows) — yet the *tradeable*
net result is **better**. That is the whole point: the 1-day signal is sharper
per-name but too expensive to harvest; the 5-day signal gives up a little raw skill
to trade half as much and nets out ahead. RF now leads the ensemble at daily.

## 6. SENSITIVITY — no-trade buffer width (`rebalance_buffer_mult`, daily long_only, 5-day target)

Robustness scan of the frozen buffer prior, on the headline survivor (daily
long_only, 5-day target). This is a **[SENSITIVITY]** run on the separate ledger
(`production_sensitivity_dsr_matrix.parquet`), **not** part of the headline N — we
report the whole band and never promote the peak (phase3 design §0, §4.1). One
deterministic tree fit is shared across all six values, so only the Tier-3
hysteresis varies — a clean isolation of the buffer effect.

| buffer_mult | turnover/bar | cost drag /yr | net CumRet | net SR_ann |
|---|---|---|---|---|
| 1.0 (raw deciles) | 0.63 | 23.1% | +36.3% | 0.302 |
| 1.5 | 0.41 | 15.1% | +78.1% | 0.443 |
| **2.0 (frozen prior)** | 0.32 | 11.7% | **+123.1%** | **0.569** |
| 2.5 | 0.27 | 10.0% | +90.1% | 0.485 |
| 3.0 | 0.24 | 8.7% | +120.3% | 0.567 |
| 4.0 | 0.20 | 7.3% | +94.0% | 0.501 |

**Read:** turnover and cost drag fall **monotonically** with buffer width
(0.63→0.20/bar, 23%→7%/yr) — the hysteresis does exactly what it should. But net
Sharpe is **non-monotonic**: it climbs 1.0→2.0 then **plateaus ~0.50–0.57 across
2.0–4.0**. Two opposing forces resolve here — a wider band saves cost (good) but
holds staler signal past its alpha peak (bad), and they roughly balance from 2×
onward. The frozen prior **2.0 sits in the robust plateau** (0.569, effectively
tied with 3.0's 0.567), **not** on a fragile spike.

**Methodological conclusion:** this is robustness evidence, not a tuning result.
The convention pick (2×, standard index-buffering band) was frozen a-priori and
lands near the top of a flat region — the strategy is *not* balanced on a knife-edge
buffer value. We therefore keep 2.0 and do **not** promote it or switch to 3.0:
picking the band's max on the OOS Sharpe is precisely the selection-on-test-statistic
bias the DSR gate corrects (phase3 §0). Turnover control unambiguously helps — raw
deciles (mult 1.0, Sharpe 0.302) are the worst cell in the band.

## 7. POST-HOC — beta decomposition & the panic-gate dead-end

Two follow-up studies on the headline survivor (daily long_only, 5-day target, net).
Both are **read-only diagnostics on the existing ledger** except §7.3, which was a
single tested-and-reverted gate variant. Market series = NIFTY-50 daily close,
**forward-aligned** to the ledger's T→T+1 return convention (a 1-bar lag otherwise
spuriously zeroes beta: contemporaneous R²=0.00 vs forward-aligned R²=0.15).

### 7.1 Alpha/beta — how much is skill vs market exposure

Regress each daily net series on NIFTY-50: `r = alpha + beta·r_mkt + eps`.

| style | alpha/yr | alpha t-stat | beta | R² | raw SR | beta-hedged SR |
|---|---|---|---|---|---|---|
| **long_only** | +7.6% | **0.81** | **0.586** | 0.15 | 0.57 | **0.31** |
| long_short | −4.5% | −0.46 | −0.022 | 0.00 | −0.19 | −0.17 |
| dynamic_tilt | +1.8% | 0.14 | 0.668 | 0.11 | 0.30 | 0.05 |

**Read:** long_only is *not* a pure beta bet (β 0.59, cash 19% of days) but it isn't
market-neutral either — **~half its Sharpe (0.57→0.31 hedged) is market beta** that a
bull sample flatters. The residual skill is positive but **statistically insignificant**
(alpha t 0.81 « 2) — the same verdict the DSR gate returns. long_short (β≈0, the pure
cross-sectional signal) has *negative* net alpha; dynamic_tilt is almost all beta.

### 7.2 Drawdown behavior — where the gate helps and where it doesn't

Cumulative net return of daily long_only through named crisis windows (old/production gate):

| window | NIFTY-50 | long_only |
|---|---|---|
| 2018 midcap/NBFC (Feb–Oct 18) | −0.1% | **−36.5%** |
| COVID crash (Jan–Mar 20) | −35.8% | **−1.1%** |
| COVID + recovery (→Aug 20) | −6.0% | **+25.3%** |
| 2022 rate grind (Oct 21–Jun 22) | −16.8% | **−32.7%** |

The gate protects **sharp, index-wide crashes** (COVID −36% → −1%, then +25% incl.
rebound) but **not** (a) size-localized selloffs where NIFTY-50 is flat while the
mid/small-cap book collapses (2018), nor (b) slow grinds the relative percentile never
trips on (2022).

### 7.3 Multi-scale panic gate — TESTED AND REJECTED

Hypothesis (from §7.2): the 2018 miss was the gate sensing risk on **NIFTY-50** while the
book trades mid/small-caps. Fix tried: make the gate's stress a skip-missing mean of
z-realized-vol across **three scales** — NIFTY-50, NIFTY-MIDCAP-150 (lists 2019-07), and an
equal-weight basket of the 68 traded stocks — plus the existing herding term. HMM states
left untouched (verified byte-identical); only the gate changed. Re-ran daily 5b net.

| metric | old gate (NIFTY-50 only) | multi-scale gate |
|---|---|---|
| net SR / DSR | 0.57 / 0.737 | **0.44 / 0.631** |
| net CumRet | +123% | +76% |
| COVID crash window | −1.1% | **−10.3%** |
| 2018 window | −36.5% | −40.0% |
| cash (de-risk) days | 324 | 273 |

**Refuted — reverted.** The multi-scale gate degraded every window and *broke the COVID
protection*: adding scales to a **relative top-15% percentile** reshuffled which days rank
"most stressed," pulling the trigger away from the real crashes. Deeper cause: **2018 was
never a gate-blindness problem** — the old gate already tripped **26% of 2018 days** yet the
book still lost −36.5%. A de-risk gate can only reach cash (0%); it cannot make a **net-long
book (β 0.59, §7.1)** profit while its universe falls −36%. **Gating is not the lever.** The
edge's thinness — not its drawdowns — is what fails the DSR gate, so the levers remain
breadth / features / horizon (presentation notes §4), not gate engineering.

## Key finding

The signal has **real cross-sectional predictive skill** (§1, IC 0.02–0.09) but the
per-trade edge is too small to fully clear **realistic NSE delivery costs**.
Gross→net destroys every intraday cell (−100%); the surviving edge is **daily
long_only**, and the lever is **turnover reduction**. Slowing the learning target
to 5 days (§5) halves turnover and lifts daily long_only to net SR +0.57 / DSR
0.737 — real progress, still shy of the 0.95 gate. Costs here remain a **lower
bound** (slippage still deferred, design doc §4.3), so net results can only worsen
from here; the remaining headroom is more turnover control (wider buffer, longer
horizon) and eventually a slippage/impact study on the surviving cell.

## Artifacts (gitignored `data/trial_database/`)

| File | Contents |
|---|---|
| `archive/phase3_daily_dsr_matrix.parquet` | **the §5 ledger** — daily-only N=3, 5-day target (net). Verified +0.57 / −0.19 / +0.30 |
| `production_dsr_matrix_daily5b.parquet` | identical copy of the above (§5) |
| `production_dsr_matrix_daily1b.parquet` | daily N=3 NET ledger — 1-day target (§4). Verified +0.49 / −0.37 / +0.18 |
| `production_dsr_matrix_net12.parquet` | full 12-cell NET ledger (§3). Verified against every row of §3 |
| `tree_fit_diagnostics_{15min,30min,60min}.csv` | per-model rank-IC diagnostics, 1-day (§1) |
| `tree_fit_diagnostics_daily_5b.csv` | daily per-model rank-IC diagnostics, 5-day target (§5) |
| `production_sensitivity_dsr_matrix.parquet` | sensitivity-scan ledger — buffer-width band (§6) |

**Two pointers moved after this document was written, and are corrected above.**
`production_dsr_matrix.parquet` was **reused by the Phase 4b run** on 2026-08-06 (see
[`phase4_results.md`](phase4_results.md) §0) and now holds three `daily_nse500` columns at
SR +0.86 / +1.78 / +1.37 — not the §5 cells. The Phase 3 ledger was moved to
`archive/phase3_daily_dsr_matrix.parquet` at that time, and `production_dsr_matrix_dsr_gate.csv`
is likewise now the Phase 4b gate output, not this phase's.
