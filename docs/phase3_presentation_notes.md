# Phase 3 — Presentation Reference Notes

**Purpose.** A self-contained, detailed narrative of the PRODUCTION_ML cross-sectional
alpha system: what it is, every result we produced, why the edge is thin, and how to
improve it. This is the *source material* for a future slide deck — deliberately more
detailed than needed for slides so nothing has to be reconstructed later. The raw
numbers also live in [`phase3_results.md`](phase3_results.md); design rationale in
[`phase3_design_requirements.md`](phase3_design_requirements.md).

**Run vintage:** all figures 2026-07-09 → 2026-07-11. Universe 68 NSE tickers,
2018-02-21 → 2025-02-13 (1,732 daily obs). 17 price-only features (delivery family
excluded to avoid NSE rate limits). Ensemble learner = LightGBM + XGBoost +
RandomForest, raw-return predictions averaged. Deterministic (seed 42).

---

## Part 1 — The system (basic idea)

### 1.1 One-sentence description

A **cross-sectional equity alpha engine**: every bar, machine-learning trees rank the
68-stock universe by predicted forward return; we go long the top decile and short the
bottom decile, gate the book down in high-volatility regimes, charge realistic NSE
transaction costs, and then **deflate the resulting Sharpe for multiple-testing** before
we believe any of it.

"Cross-sectional" is the key phrase: we do **not** predict whether the market goes up.
We predict *which stocks beat which other stocks* on the same day. The bet is relative
(long the winners, short the losers), so a rising or falling market is largely netted
out — the edge is in the ranking, not the direction.

### 1.2 The five-tier pipeline

The orchestrator (`run_pipeline_ml.py`) runs one `StrategyConfig` at a time through a
linear stack. Each tier is a pure function of the previous tier's output.

| Tier | Module | Job |
|---|---|---|
| **0 — Data** | `data_scraping.py` | Load local OHLCV CSVs (daily + intraday) into a `('date','ticker')` panel. Prices are user-supplied, not scraped; only NSE delivery metrics were ever scraped (and are currently excluded). Strict `.ffill()` only — never backfill (zero look-ahead). |
| **1 — Features + Trees** | `feature_creator.py`, `tier1_trees.py` | Build 17 price-only features, each cross-sectionally rank-normalized to [-1,+1] per day. Then a **walk-forward** tree ensemble predicts forward return and buckets each day into a top-10% long mask and bottom-10% short mask. |
| **2 — Regime gate** | `tier2_regime.py` | A 2-state Gaussian HMM on index returns labels each day Calm (S0) or high-vol. A "Panic" gate fires on the ~top 15–21% most-turbulent days and de-risks the book. |
| **3 — Execution** | `tier3_execution.py` | Turn masks into a weight matrix: 1/k decile weights, style-specific leverage, the no-trade **hysteresis buffer**, and the **NSE cost stack** charged on turnover. Emits the net return series. |
| **4 — Credibility gate** | `tier4_dsr_gate.py` | Append the return series to the Parquet ledger and compute the **Deflated Sharpe Ratio** — the Sharpe discounted for how many strategies we searched. This is the "do we actually believe it?" test. |

### 1.3 The learning engine (Tier 1 detail)

- **Walk-forward, never a single train/test split.** A trailing **504 trading days (~2y)**
  trains the trees; they predict the next **63 days (~1 quarter)** out-of-sample; then the
  window slides forward one quarter and refits. ~28 folds over the sample. Training never
  overlaps the scored block → **zero look-ahead**.
- **Ensemble of three learners** (LightGBM, XGBoost, RandomForest), fresh per fold, votes
  by averaging raw forward-return predictions into one Alpha Score per stock.
- **Decile buckets:** each out-of-sample day, `k = max(1, floor(n·0.10))` names go long
  (+1) and `k` go short (−1); no overlap.

### 1.4 The three execution styles (the core research question)

The sweep asks *does the short leg earn its keep?* by running three books:

| Style | Calm regime (S0) | Panic regime |
|---|---|---|
| `long_only` | +1/k on the top decile | flat (cash) |
| `long_short` | +1/k long, −1/k short | flat (cash) |
| `dynamic_tilt` | 130/30 (1.3× long, 0.3× short) | dollar-neutral 1.0×/1.0× (never cash) |

### 1.5 Design discipline (why the results are trustworthy, not just good-looking)

These are the guardrails that separate an honest backtest from a curve-fit:

- **Zero look-ahead bias** — `.ffill()` only; all features use trailing/current bars;
  signal at T realizes as return T→T+1.
- **Walk-forward out-of-sample** — every scored day was predicted by a model that never
  saw it or anything after it.
- **Degrees of freedom are a budget.** We sweep only genuine structural forks (frequency,
  execution style, HMM states) and *freeze everything else by prior* (tree
  hyperparameters, leverage split, buffer width). Every swept cell is an implicit
  backtest; more cells = more chances to get lucky.
- **Freeze-before-test.** Frozen params (panic threshold 0.85, decile 10%, buffer 2×) are
  committed *before* the out-of-sample window is scored, and justified by convention or
  reported as a distribution — never selected on the test statistic.
- **Deflated Sharpe (DSR).** The final gate discounts each Sharpe for N = the number of
  trials searched, so a strategy that only looks good because we tried many is caught.

### 1.6 How to read the metrics (glossary)

| Metric | What it means | Reading |
|---|---|---|
| **IC** (Information Coefficient) | Daily cross-sectional Spearman correlation of predicted alpha vs realized forward return | Predictive skill. ~0.02–0.05 is a genuinely useful signal in equities; higher is rare. |
| **IC_IR** / **IC_t** | mean IC ÷ std of IC / its t-stat | Consistency of the skill. IC_t > 2–3 = statistically real. |
| **hit%** | Share of days IC > 0 | >50% = ranks the right direction more often than not. |
| **turnover/bar** | Σ\|Δw\| per bar (fraction of book traded) | Cost driver. 1.0 = replacing the whole book each bar. |
| **cost drag /yr** | Annualized return lost to transaction costs | The gross→net gap. |
| **SR_ann** | Annualized Sharpe of net daily returns | Risk-adjusted return. |
| **DSR** | Deflated Sharpe Ratio (0–1) | Probability the true Sharpe > 0 after correcting for N trials. Gate = 0.95. |
| **PSR₀** | Probabilistic Sharpe vs zero benchmark (no deflation) | Confidence the Sharpe beats 0, ignoring multiple testing. |
| **SR\*** | Deflation benchmark — the Sharpe you'd expect from the *luckiest of N* random trials | The bar each cell must clear. Inflated when the N trials are very dispersed. |
| **minTRL** | Minimum track-record length (days) to reach DSR credibility at the observed Sharpe | Feasibility. Huge minTRL = edge too weak to ever prove with available data. |

---

## Part 2 — Detailed results (all of them)

Five run configurations, in the order we ran them: (A) predictive-skill diagnostics,
(B) gross 12-cell grid, (C) net 12-cell grid, (D) daily focus with 1-day vs 5-day target,
(E) buffer-width sensitivity.

### 2.A — Predictive skill: does the model actually rank stocks? (YES)

Per-fold cross-sectional rank IC (read-only diagnostic; never used to select cells). Ensemble row bolded.

**1-day target (`tgt_fwd_logret_1b`):**

| freq | model | mean_IC | IC_IR | IC_t | hit% |
|---|---|---|---|---|---|
| 15min | **ensemble** | **0.0829** | 0.567 | — | 71.5% |
| 30min | **ensemble** | **0.0634** | 0.419 | — | 66.5% |
| 60min | **ensemble** | **0.0569** | 0.379 | — | 65.1% |
| daily | **ensemble** | **0.0234** | 0.147 | 6.12 | 55.8% |

(Per-model at daily: lgbm 0.0207, xgb 0.0212, rf 0.0166, ensemble 0.0234. LightGBM leads
intraday; the ensemble wins at daily.)

**Read:** the skill is **real and statistically strong** (daily IC_t = 6.12), and it
**decays monotonically with horizon** (0.083 → 0.063 → 0.057 → 0.023 as bars lengthen).
The trees genuinely rank stocks. *Whether that ranking is tradeable after costs is a
separate question — answered below.* (Intraday IC_t is inflated by bar autocorrelation;
trust mean_IC/hit%, not IC_t, there.)

### 2.B — Gross economics: no costs, raw deciles (REFERENCE ONLY)

Annualized Sharpe on daily-compounded returns. DSR at N=12, deflation SR\* = 9.17.

| freq | style | turnover/bar | gross CumRet | SR_ann | DSR | verdict |
|---|---|---|---|---|---|---|
| 15min | long_only | 1.22 | 4.8e12 % | 11.42 | 1.000 | PASS |
| 15min | long_short | 2.29 | 1.1e26 % | 16.29 | 1.000 | PASS |
| 15min | dynamic_tilt | 2.41 | 2.3e27 % | 17.16 | 1.000 | PASS |
| 30min | long_only | 1.16 | 2.7e6 % | 6.27 | 0.000 | fail |
| 30min | long_short | 2.11 | 1.9e13 % | 11.50 | 1.000 | PASS |
| 30min | dynamic_tilt | 2.25 | 3.3e13 % | 11.32 | 1.000 | PASS |
| 60min | long_only | 1.14 | 5,399 % | 3.17 | 0.000 | fail |
| 60min | long_short | 2.07 | 5.9e7 % | 8.04 | 0.001 | fail |
| 60min | dynamic_tilt | 2.22 | 1.9e7 % | 6.66 | 0.000 | fail |
| daily | long_only | 0.89 | 1,604 % | 1.62 | 0.000 | fail |
| daily | long_short | 1.80 | 1,800 % | 1.81 | 0.000 | fail |
| daily | dynamic_tilt | 1.87 | 4,577 % | 1.74 | 0.000 | fail |

**Read:** 5/12 "pass" — all intraday. **These are cost mirages.** The astronomical
cumulatives (1e27 %!) are the artifact of compounding a tiny consistent per-bar edge
across tens of thousands of *zero-cost* trades. The daily cells (the only economically
interpretable gross numbers) fail here only because the deflation benchmark is dragged up
by the inflated intraday cells sharing the N=12 family. **Gross Sharpe rises with
frequency — costs (2.C) invert that completely.**

### 2.C — Net economics: NSE delivery costs + buffer (THE REAL VERDICT)

Costs = itemized NSE cash-delivery stack composed to **~14.7 bps one-way**
(brokerage 3 + STT 10 both-sides + exchange 0.3 + SEBI 0.01 + stamp 1.5 buy-only + 18%
GST), charged on turnover; + 50 bps/yr short borrow. Buffer 2×. DSR N=12, SR\* = 12.02.

| freq | style | turnover (vs gross) | cost drag /yr | net CumRet | SR_ann | DSR |
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

**Read:** 0/12 pass. **Complete gross→net inversion** — net Sharpe now falls
monotonically with frequency, the exact opposite of gross. The Sharpe-16/17 intraday
"winners" are **−100% net**: total wipeout from cost drag of hundreds-to-thousands of
%/yr on their turnover. **Only daily long_only survives** (+97% over ~7yr ≈ ~10%/yr).
Cost is essentially the *entire* gross→net gap: daily long_only goes from gross SR 1.62 →
net 0.49. **The lever is turnover.**

### 2.D — Daily focus: 1-day vs 5-day learning target

Intraday dropped on structural grounds (turnover × cost is deterministically fatal, not a
noisy outcome). The honest headline family is the 3 daily style cells. We then re-trained
the trees on the **5-day** forward return instead of 1-day — a slower learning target →
slower signal → longer holds → lower turnover.

**1-day target (DSR N=3, SR\* = 0.37):**

| style | turnover | drag/yr | net CumRet | SR_ann | DSR | PSR₀ | minTRL |
|---|---|---|---|---|---|---|---|
| **long_only** | 0.59 | 21.9% | +97.4% | **+0.49** | 0.626 | 0.902 | 45,453 |
| dynamic_tilt | 1.23 | 45.8% | +1.4% | +0.18 | 0.312 | 0.685 | n/a |
| long_short | 1.21 | 45.2% | −58.7% | −0.37 | 0.027 | 0.169 | n/a |

**5-day target (DSR N=3, SR\* = 0.33):**

| style | turnover (vs 1d) | drag/yr (vs 1d) | net CumRet (vs 1d) | SR_ann (vs 1d) | DSR (vs 1d) | PSR₀ | minTRL |
|---|---|---|---|---|---|---|---|
| **long_only** | 0.32 (0.59) | 11.7% (21.9%) | **+123.1%** (+97.4%) | **+0.57** (+0.49) | **0.737** (0.626) | 0.932 | 11,615 |
| dynamic_tilt | 0.64 (1.23) | 24.0% (45.8%) | +36.1% (+1.4%) | +0.30 (+0.18) | 0.475 (0.312) | 0.786 | n/a |
| long_short | 0.66 (1.21) | 24.9% (45.2%) | −42.6% (−58.7%) | −0.19 (−0.37) | 0.090 (0.027) | 0.314 | n/a |

**Read:** the 5-day target **improved every cell.** It ~halved turnover (long_only
0.59→0.32) and ~halved cost drag. Daily long_only: net SR +0.49→**+0.57**, DSR
0.626→**0.737**, and minTRL fell from ~45,453 to **11,615 days** (~180yr → ~46yr of daily
data to reach credibility). **The counterintuitive core insight:** the 5-day raw IC is
slightly *lower* (ensemble daily mean_IC 0.019 vs 0.023; its IC_t 4.6 is inflated by
overlapping windows), yet the *tradeable* net result is *better*. The 1-day signal is
sharper per-name but too expensive to harvest; the 5-day signal gives up a sliver of raw
skill to trade half as much and nets out ahead. (At 5-day, RandomForest narrowly leads
the ensemble at daily.)

Still **short of the 0.95 gate** — the edge is real and now more cost-robust, but too thin.

### 2.E — Buffer-width sensitivity (robustness of the frozen 2× prior)

The no-trade **hysteresis buffer**: a name *enters* a leg at the 10% decile edge but is
only *evicted* once it drifts past the wider `10% × mult` band — so marginal names
jittering across the decile boundary don't force a costly round-trip. Swept on the
headline survivor (daily long_only, 5-day target). One deterministic tree fit shared
across all six values, so *only* the Tier-3 buffer varies — a clean isolation.

| buffer_mult | turnover/bar | cost drag /yr | net CumRet | SR_ann |
|---|---|---|---|---|
| 1.0 (raw deciles) | 0.63 | 23.1% | +36.3% | 0.302 |
| 1.5 | 0.41 | 15.1% | +78.1% | 0.443 |
| **2.0 (frozen prior)** | 0.32 | 11.7% | **+123.1%** | **0.569** |
| 2.5 | 0.27 | 10.0% | +90.1% | 0.485 |
| 3.0 | 0.24 | 8.7% | +120.3% | 0.567 |
| 4.0 | 0.20 | 7.3% | +94.0% | 0.501 |

**Read:** turnover and cost drag fall **monotonically** with buffer width (0.63→0.20/bar,
23%→7%/yr) — the hysteresis does exactly what it should. But net Sharpe is
**non-monotonic**: it climbs 1.0→2.0 then **plateaus ~0.50–0.57 across 2.0–4.0.** Two
opposing forces resolve — wider band saves cost (good) but holds staler signal past its
alpha peak (bad), and they roughly balance from 2× on. **The frozen 2.0 sits in the
robust plateau** (0.569, effectively tied with 3.0's 0.567), *not* on a fragile spike.

**Methodological point (important for the deck):** this is *robustness evidence, not a
tuning result.* The 2× convention was frozen a-priori and lands near the top of a flat
region — the strategy is not balanced on a knife-edge. We therefore **keep 2.0 and do NOT
promote it or switch to 3.0**: picking the band's max on out-of-sample Sharpe is exactly
the selection-on-test-statistic bias the DSR gate exists to correct. Turnover control
unambiguously helps (raw deciles at 1.0 are the worst cell, 0.302) — but the buffer is
*not* a source of remaining edge; that plateau means there's no free Sharpe there.

### 2.F — Where the strategy stands

- **Signal skill:** real and strong (IC up to 0.08 intraday, 0.02–0.03 daily, IC_t 6).
- **Gross:** every cell profitable; Sharpe rises with frequency.
- **Net:** all intraday cells are −100% (uneconomic); **daily long_only is the sole
  survivor** at net SR ~0.57 / DSR 0.737.
- **Two turnover levers exhausted** (5-day target + buffer) — both helped, neither closes
  the gap to the 0.95 credibility gate.
- **Bottom line:** a genuine cross-sectional signal whose per-trade edge is *too thin to
  clear realistic NSE delivery costs at daily frequency* on this small universe. And note
  costs are a **lower bound** — slippage/market-impact is still deferred, so live net can
  only be worse.

---

## Part 3 — Why the edge is thin: the GKX comparison

The natural challenge: Gu, Kelly & Xiu (2020, *"Empirical Asset Pricing via Machine
Learning,"* RFS) used cross-sectional ML to produce large, consistent long-short profits.
Why do we die under costs when they didn't? **The honest answer: we don't contradict GKX
— we reproduce the sober, net-of-cost version of it on a small, expensive universe.** The
gap is structural, and almost none of it is the modeling. Ranked by impact:

### 3.1 Breadth — the single biggest factor (~20× against us)

GKX ran **~30,000 US stocks over ~60 years (1957–2016)**. We run **68 NSE tickers over
~7 years.** The Fundamental Law of Active Management: `IR ≈ IC × √breadth`. Our IC
(0.02–0.09) is *comparable to or better than* GKX's — their pooled monthly out-of-sample
R² was a famously tiny ~0.4%. Skill per bet is similar. What differs is the number of
independent bets: √30,000 ≈ 173 vs √68 ≈ 8 — a **~20× haircut on achievable Sharpe**
before anything else. With 68 names you hold ~7 per decile leg: a *concentrated*
portfolio, not a diversified cross-sectional harvester. GKX's Sharpe *requires*
law-of-large-numbers averaging over thousands of simultaneous small edges.

### 3.2 Horizon — GKX rebalanced monthly; we trade daily

GKX predicts and rebalances **monthly**. We're daily (intraday was deterministically
fatal). Monthly holding ≈ 1/21 the turnover per unit time, so cost drag is an order of
magnitude smaller for the same gross edge. Our entire 5-day-target + buffer campaign was
an attempt to crawl toward that low-turnover regime — and it helped *precisely because* it
moves us in the monthly direction.

### 3.3 Feature set — GKX had ~900 fundamental/macro signals; we have 17 price-only

GKX used **94 firm characteristics** (valuation, profitability, investment, accruals —
real accounting fundamentals) **× 8 macro predictors + industry dummies ≈ 900+ features.**
We use **17 price-only features** (momentum / reversal / volatility); delivery data was
excluded to dodge NSE rate limits. Fundamental cross-sectional anomalies are
*slower-decaying and cheaper to trade* than pure price momentum — which is the
fastest-mean-reverting, most-arbitraged, highest-turnover signal family there is. We
picked the thinnest, most cost-hostile corner of the GKX information set.

### 3.4 Gross vs net, and the microcap/short concentration

**GKX's headline Sharpes are largely gross.** The follow-up literature (Avramov, Cheng &
Metzker 2023, *"Machine Learning vs. Economic Restrictions"*; and the trading-cost
critiques) showed the profits **concentrate disproportionately in microcaps,
distressed/hard-to-short names, and the short leg** — exactly the stocks that are
expensive or impossible to trade in size. Exclude microcaps, impose short constraints, net
out costs, and GKX-style gross Sharpes of ~1.3 (value-weighted) to ~2+ (equal-weighted)
shrink dramatically. We can't lean on that engine — 68 relatively large NSE names, and our
own short leg is a net *loser* (long_short −0.19 vs long_only +0.57). **Our net verdict is
the answer to the exact question the post-GKX literature asked.**

### 3.5 Market regime — NSE costs are brutal

US institutional equity costs are single-digit bps. NSE cash-**delivery** carries **STT at
0.10% on *both* sides** (10 bps each way in one line item alone) plus stamp/exchange/GST →
our composed ~14.7 bps one-way. Overnight shorting in India is structurally harder than
borrowing US large-caps. GKX operated in about the cheapest, deepest equity market on
earth; we're in one of the more expensive ones for this style.

### 3.6 What changed, in one line

Not the HMM and not the trees (the trees work — Part 2.A proves it). It's that:

> **GKX** = {30k names × 60yr × 900 fundamental features × monthly × **gross** × cheap US
> market × microcap/short concentration}
> **Ours** = {68 names × 7yr × 17 price features × daily→5-day × **net** × expensive NSE ×
> large-cap long-only}.

Each factor costs us; **breadth + horizon + gross-vs-net dominate.** Our pipeline is
methodologically *more honest* than the GKX headline (we deflate for multiple testing and
net out realistic costs), and it lands right where the sober follow-up literature says a
cross-sectional ML signal lands once you do that on a small, expensive universe.

---

## Part 4 — Ways to improve

Ordered by expected payoff. The first two attack the factors that actually dominate
(Part 3); turnover tuning is already exhausted (flat plateau).

### 4.1 Breadth — expand the universe (highest leverage)

Go from 68 to the largest liquid slice we can support — NIFTY 200 / 500 (200–500 names).
`IR ≈ IC × √breadth`, so 68 → 500 names is a `√(500/68) ≈ 2.7×` uplift on achievable
Sharpe *at the same IC*, and it lets each decile leg hold 20–50 names instead of ~7
(diversification, less idiosyncratic noise). This is the single biggest structural lever
and directly closes the largest GKX gap. Cost: more data plumbing, longer tree fits,
liquidity screening.

### 4.2 Information — add fundamental & institutional features

The signal is currently price-only. Add the families GKX actually relied on:
- **Re-enable the delivery data** (`DeliveryQty`, `DeliveryPct`) that's already wired but
  excluded — institutional-participation signal, slower-moving.
- **Accounting fundamentals** (valuation, profitability, investment, accruals) — the
  slow-decaying, cheaper-to-trade anomalies that make monthly cross-sectional ML work.
- Slower features → slower signal → lower turnover *and* more skill: attacks both the
  feature-set gap (3.3) and the horizon/turnover problem (3.2) at once.

### 4.3 Horizon — push to 10-day / monthly rebalance

The 1→5-day jump gave the biggest single net lift. A **10-day or monthly** target is the
natural next probe — it moves us onto GKX's actual rebalance cadence, where the cost math
becomes survivable. Diminishing returns are likely past some point (staler signal), and
the buffer plateau (2.E) warns the turnover lever alone flattens out — so pair horizon
with breadth/features, don't rely on it solo. Cheap to try (a config flag already exists:
`--target`).

### 4.4 Cost realism — slippage / market-impact study

Current costs are a **lower bound** (slippage deferred). Before trusting any surviving
cell, add a size/liquidity-dependent impact model (participation rate, spread, ADV). This
only *lowers* net — it's a reality-check gate, not an improvement — but it's mandatory
before "daily long_only survives" becomes "daily long_only is tradeable." Do it *after*
breadth expansion, on whatever cells still survive.

### 4.5 Portfolio construction — beyond equal-weight deciles

Current book is naive 1/k decile weights. Options: risk-parity / inverse-vol weighting,
sector/beta neutralization (strip out unintended factor bets), and an explicit
turnover-penalized optimizer (trade off alpha vs cost *inside* the weight solve rather
than via a post-hoc buffer). Modest, incremental; do after breadth/features.

### 4.6 Honest fallback — accept the finding

If breadth + fundamentals + monthly horizon still don't clear the DSR gate net-of-slippage,
that is itself a **legitimate, publishable result**: a pure cross-sectional
momentum/price signal on a mid-cap Indian universe is structurally marginal after NSE
delivery costs. The methodology (walk-forward, freeze-before-test, DSR deflation, itemized
costs) is the deliverable — it's a rigorous *negative* result, which is worth more than an
over-fit positive one.

### Recommended sequence

1. **Expand universe to NIFTY 200/500** (4.1) — biggest lever, closes the dominant gap.
2. **Add fundamental + delivery features** (4.2) — second gap, and lowers turnover.
3. **Re-run the daily/weekly grid** on the bigger, richer universe; re-check DSR at honest N.
4. **If a cell survives**, gate it through a **slippage study** (4.4) before any claim.
5. Portfolio-construction polish (4.5) only on a survivor.

---

## Artifacts (data files, gitignored `data/trial_database/`)

| File | Contents |
|---|---|
| `production_dsr_matrix.parquet` | current ledger — daily N=3, 5-day target (net) |
| `production_dsr_matrix_daily5b.parquet` | archived daily N=3 net — 5-day target (2.D) |
| `production_dsr_matrix_daily1b.parquet` | archived daily N=3 net — 1-day target (2.D) |
| `production_dsr_matrix_net12.parquet` | archived full 12-cell net ledger (2.C) |
| `production_sensitivity_dsr_matrix.parquet` | buffer-width sweep ledger (2.E) |
| `tree_fit_diagnostics_{15min,30min,60min,daily}.csv` | per-model rank-IC, 1-day (2.A) |
| `tree_fit_diagnostics_daily_5b.csv` | per-model rank-IC, 5-day target (2.D) |

**Companion docs:** [`phase3_results.md`](phase3_results.md) (raw results tables),
[`phase3_design_requirements.md`](phase3_design_requirements.md) (sweep-design rationale,
cost model, bias methodology).
