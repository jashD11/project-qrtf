# RESUME_BRIEF.md — factual brief of `qrtf_engine`

Compiled 2026-09-27 at `feature/gkx-trees-pipeline` HEAD `7bb3c84`.
**[V]** = VERIFIED (machine-written results/log file, or reproduced by a cheap command this
session). **[R]** = REPORTED (prose only: docs / README / CLAUDE.md).

## A. Summary
Can a tree-ensemble cross-sectional ranker make money on NSE equities **after real Indian
costs and real short-borrow availability**? Approach: point-in-time ~500-name universe rebuilt
from raw bhavcopy (no index-membership survivorship), 17 rank-normalized price features,
walk-forward LGBM/XGB/RF ensemble on forward returns, decile long/short books gated by a causal
"panic" de-risk signal, itemized NSE cost stack plus measured half-spread and √-law impact,
short leg restricted to point-in-time borrowable names, judged by Deflated Sharpe against an
auditable count of every trial searched (N=30). **Conclusion: a rigorous negative result.** The
ranker has genuine skill (ensemble rank-IC 0.0511, IC_t 22.49 [V]) but 0 of 6 frozen cells clear
the gate — best DSR **0.0165** vs 0.95 [V]. Cost kills it, not signal quality.

## B. Data
NSE daily bhavcopy (EQ/BE) + index + F&O archives via `jugaad-data`
(`src/phase4_data/bhavcopy_download.py`); a Google-Drive 15-min panel for Phase 2/3
(`download_nse.py`). Acquisition 2013-01-01→2025-06-30 (`config.py:339`) [V]. Universe mask
3,094 dates × **1,118** distinct tickers, exactly 500/day on 2,842 active days from 2014-01-03
[V, reproduced]; 1,030 reach the trees after warm-up; Phase 2/3 panel is 68 names [V]. Panel
2,735,949 bars → 1,409,771 post-warm-up, 51.7% in-universe [V]. Scored OOS window
2016-01-21→2025-06-27, **2,337** daily bars [V]. Frequencies: 15/30/60min, daily, daily_nse500
(`config.py:43-59`). **No fixed split — walk-forward:** 504-day trailing train → disjoint 63-day
OOS block, slid; 38 folds on daily_nse500 [V]. **17** price-only features in use (19 defined;
delivery family excluded) [V]. Survivorship: universe re-ranked each quarter by trailing-252-day
*median* rupee turnover, ≥200 traded days, top-500 held fixed; delisted names get terminal
return −0.30 (`universe.py`, `config.py:345`) [V]; 107 delisted/acquired names received terminal
returns [V].

## C. Methods implemented
LightGBM / XGBoost / RandomForest + equal-weight ensemble, walk-forward
(`tier1_trees.py:45-47`); rank-IC diagnostic marked read-only; Gaussian HMM (`hmmlearn`) with a
hand-rolled **causal forward-filter** decode, no backward pass (`tier2_regime.py:221,330`);
causal expanding z-scores and an 85th-percentile panic gate (`tier2_regime.py:421-424`);
Probabilistic Sharpe, Deflated Sharpe, minTRL, and the **Lo (2002) correction via overlapping
variance ratio** (`tier4_dsr_gate.py:77-199`); Corwin-Schultz + Abdi-Ranaldo half-spread
(`spread.py`); √-law impact `1e4·coef·σ·√participation`, coef 0.5 (`config.py:236-248`);
participation cap by within-leg water-filling and a two-band no-trade buffer
(`tier3_execution.py`); point-in-time shortable set from stock-futures rows (`shortable.py`);
column-scoped walk-forward disk cache (`wf_cache.py`). Phase 1 momentum ranker
(`sandbox_run/tier1_gkx.py`). All [V].

**NOT IMPLEMENTED (in docs, absent from code):** *most Gu-Kelly-Xiu families* — only trees
(GBRT, RF); no OLS/PCR/PLS/ElasticNet/GLM/neural nets anywhere, and GKX's 94 characteristics are
discussed only as literature comparison (`phase3_presentation_notes.md` §3) [V]. *HMM states
driving execution* — `RegimeResult.states/.probs/.stress` have **zero consumers**; only `.panic`
is read (`run_pipeline_ml.py:349`) and `panic` never touches the HMM, so the fitted model is
decorative [R `phase5_plan.md` §1.1; code locations V]. *Jump model, signed-EWM observation
layer, 2007-onward history, oracle short-leg gate, index quality report* — Phase 5 plan only [R].
**No test suite** (zero pytest/unittest/`def test_`) [V]. **No live trading, paper trading or
real money** — no broker SDK, order placement or websocket code; grep for
kite/zerodha/upstox/fyers/alpaca/place_order returns 0 hits [V].

## D. Evaluation
Walk-forward only; train never overlaps the scored block; regime decoded by forward-filtering;
features trailing/current-bar; signal T → return T→T+1 via `shift(-1)`; `.ffill()` only [V].
Cost/ADV/vol/spread windows end at `t-1` [R + config V]. Metrics: Sharpe (naive and
Lo-corrected), CumRet, max DD, turnover/bar, per-component drag, IC/IC_IR/IC_t/hit, PSR, DSR,
minTRL. **Cost assumptions (`_NSECostConfig`, all V):** brokerage 3.0 bps/side, STT delivery
10.0 (both sides), exchange 0.30, SEBI 0.01, stamp 1.5 (buy only), GST 18% on
brokerage+exchange+SEBI, flat slippage **0.0**, borrow 50 bps/yr flat or tiered
25/50/100/200 by liquidity quartile → composed one-way statutory **14.6558 bps** (reproduced).
Phase 4c adds per-name Corwin-Schultz half-spread + `impact_bps(σ,participation)` at AUM Rs 1
crore, `max_participation` 0.10; realized effective **24.5–26.1 bps/side** [V]. `impact_bps(0.02,
0.01)` = 10.0 bps [V]. **Overfitting control:** DSR deflated against a line-by-line
`config.TRIAL_LEDGER` of **N=30** (reproduced) [V]; grid frozen and printed pre-run; sensitivity
scans on a separate ledger, never promoted; 0.95 threshold frozen; the charitable spread
estimator chosen so the verdict can't depend on it; and every new behaviour gated off by default
so Phase 3 and 4b reproduce **bit-for-bit** (`max abs diff = 0.0` [R]; the reproducing run is
`logs/phase4c_regression.log` [V]).

## E. Results

**E1. Phase 4c frozen grid — current headline (N=30, thr 0.95).** Source
`phase4c_dsr_matrix_dsr_gate.csv` + `..._execution_diagnostics.csv`; DD/hit reproduced from
`phase4c_dsr_matrix{,_gross}.parquet`. All [V].

| target/style | SR gross | SR net | SR Lo | CumRet | maxDD | hit | turnover | bps/side | DSR |
|---|---|---|---|---|---|---|---|---|---|
| 5b long_only | 1.660 | 0.178 | 0.161 | +15.44% | −55.73% | .486 | 0.446 | 26.08 | 0.00681 |
| 5b long_short_slb | 1.882 | −0.987 | −0.906 | −74.35% | −76.14% | .408 | 0.637 | 24.52 | 3.0e−09 |
| 5b dyn_tilt_slb | 2.014 | −0.049 | −0.045 | −29.76% | −68.35% | .543 | 0.717 | 26.04 | 0.00092 |
| **21b long_only** | **1.120** | **0.298** | **0.268** | **+43.16%** | **−44.95%** | .496 | **0.242** | 25.96 | **0.01650** |
| 21b long_short_slb | 0.958 | −0.567 | −0.520 | −56.47% | −58.83% | .429 | 0.338 | 24.46 | 2.0e−06 |
| 21b dyn_tilt_slb | 1.308 | 0.212 | 0.192 | +22.72% | −50.77% | .555 | 0.370 | 25.81 | 0.00894 |

0/6 pass. SR\*(ann) 0.9757; T=2,337; **minTRL empty for all six** — none clears SR\* at any
track length [V]. Trade drag 15.96–47.12%/yr, borrow drag ≤0.82% [V].
*Rounding conflict, minor:* docs say DSR **0.017** and net SR 0.18/0.30; machine says 0.016504
and 0.1784/0.2975. Same numbers rounded in prose.

**E2. Phase 4b (superseded, N=3)** — `production_dsr_matrix_dsr_gate.csv` + `logs/phase4_run.log`, all [V]:
long_short SR **1.7783**, +639.23%, DD −19.86%, turnover 0.82, DSR 0.99999 PASS; dynamic_tilt
1.3653, +1194.71%, DD −41.62%, 0.80, DSR 0.99763 PASS; long_only 0.8593, +308.43%, DD −36.34%,
0.45, DSR 0.91553 fail. SR\* 0.3925; minTRL 360/793/3,341.

**E3. Correction waterfall on `long_short`:** 1.78 published [V] → 1.35 with honest N=30 +
autocorrelation (DSR 0.986) [R] → **−0.23** with measured spread/impact [R] → −0.99 with the
SLB-feasible short leg [V].

**E4. Rank-IC (`tree_fit_diagnostics_*.csv`, ensemble, all V):** daily_nse500/5b 0.0511
(IC_t 22.49, n=2,333) · daily_nse500/21b 0.0426 (14.79) · daily-68/5b 0.0193 (4.63) · 60min
0.0569 (40.85) · 30min 0.0635 (64.25) · 15min 0.0829 (123.18). IC decays with horizon, rises
with breadth (68→500 at 5b: 0.0193→0.0511).

**E5. Rebalance-frequency decay (Phase 3, N=12, net; `phase3_results.md` §3 [R]).** Net SR by
frequency/style — daily +0.49/+0.18/−0.37 (turnover 0.59/1.23/1.21, drag 22/46/45%/yr);
60min −6.95/−6.34/−7.39 (drag 226/373/418%); 30min −11.32/−10.29/−12.43 (439/719/812%);
15min −17.48/−17.61/−21.11 (961/1,655/1,820%). Every intraday cell is −100% net. **Passing
0/12; gross Sharpe rises with frequency (15min 2.41) while net falls monotonically to −21.**
Machine corroboration for the daily row only (5b variant: turnover 0.32, +123.06%, SR 0.57) [V].

**E6. Half-spread (`phase4c_results.md` §1 [R]):** pooled CS **5.97 bps**, AR **26.07 bps**;
39–76% of observations clip to zero; both strictly monotone in liquidity. CS corroborated by the
realized 24.46–26.08 `effective_oneway_bps` [V].

**E7. Borrowability:** in-universe borrowable names/day over the OOS window **182.24 (36.45%)**,
whole active panel 175.64 (35.13%) [V, reproduced]. Docs quote "177 (35.5%)"
(`phase4c_results.md` §6) and "182" (`phase5_plan.md` §3.1) — the same computation over
different windows, both reproduce; **not** a conflict, but neither doc states its window.
Coverage of the *intended* short book: **22.4%** overall, 49.0 intended vs 11.0 borrowable
names/day, never >32% in any year, 15.6% in 2020 [R]; short names actually held 49.0 [V].

**E8. Other:** panic gate fires on 15% of days, 337 of 2,337 bars [V]. HMM S0 56%/S1 44%, 37
folds [V]. **HMM EM does not converge** ("Delta is −0.002") on either panel [V]. `long_only`
β 0.586, α +7.6%/yr (t=0.81), R² 0.15, beta-hedged SR 0.31 vs raw 0.57 [R]. Dividend yield
measured 1.297% vs assumed 1.300% [R]. Breakeven extra cost for `long_short` to hold DSR 0.95
was 6.8 bps; measured extra was 12.4 bps [R]. Buffer sensitivity: turnover falls monotonically
0.63→0.20/bar as mult 1.0→4.0 while net Sharpe plateaus 0.50–0.57 across 2.0–4.0 [R].
**Phase 1 (README) is synthetic GBM with an in-sample Viterbi HMM — not evidence:** 12 runs,
ledger 494×21 [V], `dynamic_tilt` beating `long_only` by up to +41.4pp [R].

## F. Negative results and dead ends
1. Every intraday cell → −100% net; intraday dropped on structural grounds, not by peeking.
2. Phase 4b's two DSR passes did not survive: they rested on `slippage_bps=0.0`, an
   `impact_bps()` never called from the execution path, an unchecked short leg, and N=3.
3. **The breadth thesis (68→500) is closed** — it delivered the gross Sharpe but raised turnover
   0.32→0.45/bar [V] and reached into wider spreads in the same motion.
4. ~78% of the intended short book was never borrowable; gross Sharpe 4.20 [R] → 1.88 [V].
5. The illiquid-tail hypothesis was **refuted** — IC is flat across liquidity quintiles
   (0.0408/0.0457/0.0413/0.0394/0.0484) [R]. The edge is real, just too thin.
6. The signal decays: IC_t ~10 (2016–19) → **1.21** by 2025 [R].
7. The textbook lag-by-lag Lo correction was built and **rejected** — it scales each `ρ_k` by
   ~252 and returned 0.99–1.18 on i.i.d. input by seed alone; replaced by the variance ratio.
8. A multi-scale panic gate was tested, rejected, and still charged 1 cell to the trial ledger [V].
9. **Phase 5's trend overlay is falsified:** Sharpe falls monotonically 4.97→4.42→4.01→3.45 as
   the tilt sharpens, losing on drawdown and worst month too; optimal strength is zero. Not
   cost (turnover *falls*), not beta (book beta ≈0, contradicting the proposed mechanism) [R].
10. Phase 5's short-leg gating is near-dead: the feasible leg earns ~5.0%/yr gross (SR 0.22) and
    costs ~5.46%/yr to run [R].
11. The regime layer is structurally decorative, and `corr(z_log_realized_vol, 21-day vol z)
    = 0.907` — a vol z-score by construction [R].
12. Self-flagged: `sandbox_run/tier2_regime.py:54-55` fits full history then uses Viterbi global
    MAP. Phase 1 only, but Phase 1 regime numbers are **not** out-of-sample [R; code V].

## G. Engineering
Two orchestrators toggled by `MODE`: `run_pipeline.py` (Phase 1, 7 linear stages) and
`run_pipeline_ml.py` (Phase 2–4c sweep). **`MODE` is committed as `"SANDBOX"`** — the production
stack is not the default (`config.py:7`) [V]. Tiers: 0 ingestion → 1 tree alpha → 2 causal
regime → 3 execution+costs → 4 DSR gate. Compute-sharing sweep: Tier 1 fit once per frequency,
Tier 2 once per run, only Tier 3 fans out over styles. `wf_cache` keys on a digest of *only the
columns the fit reads*. OpenMP workaround: LGBM and XGB each vendor `libomp` and segfault on
macOS if both train in-process, so `KMP_DUPLICATE_LIB_OK=TRUE` + `OMP_NUM_THREADS=1` and serial
training, with RF reclaiming parallelism via joblib. `StrategyConfig.strategy_id` = readable
prefix + MD5-8 of every field, used as the ledger column name. **Size: 13,406 lines of Python**
(12,058 in `src/`+`scripts/`, 1,348 in three root files) across **26** `src/` modules [V, `wc
-l`; no `cloc` available]; largest `tier3_execution.py` 1,145, `bhavcopy_panel.py` 993,
`tier1_trees.py` 622. 17 markdown design/result docs (~236 KB) plus a 29-frame Beamer deck built
from the real ledgers, **currently untracked** [V]. **Tests: none** — verification is 21
`__main__` dry-runs asserting schema invariants, decile widths and the strict [−1,+1]
normalization band, plus the bit-for-bit regression log. *The suite could not be run because
there is no suite.* Git: **26 commits**, 2026-06-21 → 2026-09-27; Jash Dalal 25 (96.2%),
`jashD11` 1 (3.8%, same person's GitHub noreply identity) [V]. `data/`, `logs/` and `CLAUDE.md`
are gitignored, so **all results artifacts exist only on the machine that ran the pipeline** [V].
Deps: pandas, numpy, hmmlearn, pyarrow, scipy, scikit-learn, lightgbm, xgboost, jugaad-data,
gdown, requests.

## H. Hardest problem solved
Making the cost model per-name and causal without breaking the guarantee that every earlier
result still reproduced exactly. Phase 4b reduced the trade matrix to a scalar turnover before
costing it, which structurally prevented charging a name-specific spread or impact — so Tier 3
was reworked to keep `traded = weights.diff().abs()` as an unreduced (bar × ticker) frame and
charge statutory + measured half-spread + `impact_bps(σ, participation)` per cell, with the
participation cap applied by water-filling *within* the leg so leg gross stays 1.0 and the
existing dry-run invariants still hold. Two traps had to be closed: `.sum()` silently skips NaN,
so any gap in the cost panel would have priced an unmeasured name as free to trade — gaps take
the date's cross-sectional median and the cost frame is asserted NaN-free; and the borrowability
filter had to hit the short leg's *alpha scores* before the hysteresis buffer, not the finished
mask, or the book would have shrunk silently and broken dollar-neutrality. Every addition was
gated off by default, and the acceptance test was bit-for-bit reproduction of Phase 3 and 4b
before any new number was believed.

## I. Six candidate resume bullets (≤105 chars)
1. `Killed a Sharpe-1.78 NSE tree strategy by measuring real spreads; DSR fell to 0.017 (N=30)` (92) — E1/E2 [V]
2. `Proved 78% of a backtest's short book was never borrowable using point-in-time NSE F&O data` (91) — §6 [R] + mask [V]
3. `Built walk-forward LGBM/XGB/RF ensemble on 500 NSE names reaching rank-IC 0.051 (t=22.5)` (88) — E4 [V]
4. `Replaced Lo's lag-by-lag Sharpe correction with a variance ratio after it estimated its noise` (93) — `tier4_dsr_gate.py:77-131` [V]
5. `Rebuilt a survivorship-free NSE universe from raw bhavcopy: 1,118 names, quarterly top-500` (90) — `universe.py` + mask [V]
6. `Falsified a per-name trend overlay: Sharpe fell 4.97 to 3.45 as the tilt sharpened, gross` (88) — `phase5_plan.md` §2.2 [R]

Backup (activity, not finding): `Cut an 80-minute walk-forward refit to seconds with a
column-scoped panel-digest disk cache` (89) — `wf_cache.py` [V], 80-min figure [R].

## J. Open questions
1. **The SURP 2026 endterm report is not in this repo** — no file, string or commit mentions
   "SURP" or "endterm". The nearest artifact is the untracked 29-frame deck. The requested
   report-vs-code reconciliation cannot be done from the repo alone; supply the report.
2. Is the deck current? Built 2026-09-10 — after Phase 4c (08-06) but before the Phase 5
   falsification (09-27), so its regime-layer claims may predate the "HMM is decorative" finding.
3. `MODE = "SANDBOX"` is committed: a fresh clone runs the synthetic pipeline. Intended?
4. **Phase 5 §1.2/§2/§3 numbers are banked nowhere** — `scripts/phase5_falsification.py` is
   explicitly read-only and "writes nothing", so all of them are REPORTED only.
5. The HMM does not converge and nothing reads its states. Keep it in the pipeline at all?
6. Should README still lead with Phase 1's synthetic, in-sample-HMM numbers while the real
   negative result lives only in `docs/`?
7. `impact_coef = 0.5` is the single discretionary knob in the cost stack, self-flagged as "an
   assumption to be sensitivity-tested, not a fact" — and never was. Does the verdict move at
   0.25 or 1.0?
8. Which drawdown gets quoted? Net max DD on the best cell is **−44.95%** [V], stated in no doc;
   the docs quote gross DDs (−10.9% baseline).
9. **Selection-gate status:** the *statistical* gate (Tier 4 DSR, N=30) is complete and in the
   loop. The *portfolio* second-selection gate proposed for Phase 5 (a trend overlay on the
   traded deciles) is **falsified and not built**, with one open caveat — it was tested on raw
   decile masks without the production `rebalance_buffer_mult=2.0` hysteresis, and closing that
   is Phase 5 step 1, unrun. Confirm which meaning a given reader is being given.
