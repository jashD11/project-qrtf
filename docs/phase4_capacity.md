# Phase 4 — capacity curve

**Status: measured, 2026-08-05.** Produced by `src/phase4_data/capacity.py --build`;
full grid in `data/bhavcopy/capacity_curve.csv`. This is an analysis of the finished
panel — no trees, no backtest. Nothing here has been wired into the execution path yet.

## The question

Phase 4 rebuilt the panel at 500 names because `IR ≈ IC·√breadth` and 68 names is too
few — a `√(500/68) = 2.71×` ceiling against a required 1.49× uplift. But breadth is only
real if the names are tradeable, and tradeability is not a property of the universe
alone. It is a property of the universe **and the money being run**. A stock that
absorbs a Rs 2 lakh order without moving is a different instrument when it faces a
Rs 2 crore order.

## Method

For each AUM, over all **1,243,608 in-universe bars** (1,072 names, 2014-01 → 2025-06):

```
position      = AUM / book,  book = decile_pct × N = 50 names at N=500
participation = position / that bar's rupee turnover
impact_bps    = 1e4 · coef · σ · √participation          (config.ML_CONFIG.cost.impact_bps)
```

`max N` is the largest universe whose *least-liquid* member stays under a 10%
participation ceiling. Both sides of that trade-off move with N — the book is
`decile_pct·N` names so each position shrinks as `1/N`, but the marginal name is thinner,
and turnover falls with rank faster than `1/N`. So participation rises with N and the
constraint binds.

Universes below N=50 are reported infeasible rather than searched: at that size the
decile book rounds to a single name, and "hold the most liquid stock in India" passes
the ceiling trivially without being a cross-sectional strategy at all.

## Result

| AUM (Rs cr) | position (Rs) | median part | p99 part | bars > 10% | impact (bps) | max N | breadth vs 68 |
|---|---|---|---|---|---|---|---|
| 0.10 | 20,000 | 0.01% | 0.73% | 0.02% | 1.2 | 498 | 2.71× |
| 0.25 | 50,000 | 0.03% | 1.81% | 0.07% | 1.9 | 496 | 2.70× |
| 0.50 | 100,000 | 0.06% | 3.63% | 0.21% | 2.7 | 493 | 2.69× |
| **1.00** | **200,000** | **0.11%** | **7.25%** | **0.61%** | **3.8** | **488** | **2.68×** |
| 2.50 | 500,000 | 0.28% | 18.1% | 2.64% | 6.1 | 448 | 2.57× |
| 5.00 | 1,000,000 | 0.57% | 36.3% | 6.95% | 8.6 | 408 | 2.45× |
| 10.0 | 2,000,000 | 1.13% | 72.5% | 14.3% | 12.2 | 362 | 2.31× |
| 25.0 | 5,000,000 | 2.83% | 181% | 27.8% | 19.2 | 248 | 1.91× |
| 50.0 | 10,000,000 | 5.67% | 363% | 39.6% | 27.2 | 155 | 1.51× |
| 100 | 20,000,000 | 11.3% | 725% | 52.4% | 38.4 | — | infeasible |
| 250 | 50,000,000 | 28.4% | 1813% | 69.4% | 60.8 | — | infeasible |
| 500 | 100,000,000 | 56.7% | 3626% | 81.3% | 85.9 | — | infeasible |
| 1,000 | 200,000,000 | 113% | 7252% | 90.5% | 121.5 | — | infeasible |

## Read

**At the Rs 1 crore operating point the capacity constraint does not bind.** Median
participation is 0.11%, only 0.61% of bars exceed 10%, impact is ~3.8 bps against a
~14.7 bps statutory cost stack, and essentially the whole 500-name universe (max N = 488)
is genuinely tradeable. The earlier concern — that thin liquidity in the bottom of the
500 made the breadth illusory — turns out to be a **scale** problem, not a universe
problem, and it does not bite here.

**The cliff is between Rs 10 cr and Rs 50 cr.** Impact roughly doubles per decade of AUM
(the √ law), but the feasible universe collapses much faster because it runs into the
steep part of the liquidity curve: 488 → 362 → 155 → infeasible across Rs 1 cr → 50 cr.

**The Phase 4 thesis is AUM-dependent, and that is the finding.** The required uplift to
clear DSR 0.95 is 1.49×. Against the breadth ceiling:

- **Rs 1 cr — 2.68×.** Comfortable headroom.
- **Rs 10 cr — 2.31×.** Still viable.
- **Rs 50 cr — 1.51×.** Essentially no margin; the ceiling barely exceeds the requirement,
  and a ceiling is not an achievement.
- **Rs 100 cr+ — infeasible.** The strategy does not fit in this universe at that size,
  regardless of how good the signal is.

So "does the tree stack work on Indian equities?" is not a well-posed question. "Does it
work at Rs 1 crore?" is, and the data says the universe can carry it. At Rs 100 crore the
answer is no on liquidity grounds alone, before any Sharpe is computed.

## Caveats

- **`impact_coef = 0.5` is the one discretionary number in the whole cost stack.** Every
  other line item (STT, stamp duty, GST, exchange fees) is a citable statutory rate; this
  is a modelling choice. It puts a 1%-of-ADV trade in a 2%-daily-vol stock at ~10 bps,
  the right order for Indian cash equities, but it should be sensitivity-tested before
  any number here is leaned on.
- **Impact is per trade, not annualized.** Turning these into a Sharpe drag needs the
  strategy's realized turnover, which requires a run.
- **Participation uses same-day turnover**, i.e. it assumes the position is executed in
  one session. Spreading a trade over several days lowers impact and is what a real
  implementation would do, so these figures are conservative.
- **Not yet wired into execution.** `tier3_execution.py` still charges the flat statutory
  stack only. Threading the per-name impact term through is deliberately deferred — it is
  not needed at Rs 1 cr and is required before any claim at Rs 10 cr+.
