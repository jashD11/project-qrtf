# Phase 2 Feature Dictionary

Complete reference for every feature produced by `src/feature_creator.py` (the
`PRODUCTION_ML` tree pipeline). **19 predictive features** across three families,
plus **2 forward-return targets**.

## Where the data comes from

The feature factory consumes an **augmented daily frame** built by
`src/data_scraping.py`. That frame is a **hybrid** of two very different sources
— worth understanding before trusting any feature:

| Data | Columns | Source | Scraped live? |
|---|---|---|---|
| Daily OHLCV | `open/high/low/close/volume` | Local CSV `data/daily_ohlcv.csv` | **No** — read from disk |
| 15-min OHLCV | `open/high/low/close/volume` | Local CSV `data/15min_ohlcv.csv` | **No** — read from disk |
| Institutional delivery | `DeliveryQty`, `DeliveryPct` | NSE full bhavcopy via jugaad-data | **Yes** |

- **Prices are not scraped.** `load_local_csv_data()` reads OHLCV from two local
  CSVs. Their provenance is **external and user-supplied** — the pipeline neither
  fetches nor generates them. Both files live under the gitignored `data/`
  directory and must be dropped in before the `PRODUCTION_ML` path can run.
- **Only delivery metrics are scraped.** `scrape_nse_delivery_data()` pulls two
  columns (`DELIV_QTY`, `DELIV_PER`) from NSE's `sec_bhavdata_full_DDMMYYYY.csv`
  (host `nsearchives.nseindia.com`) via `NSEArchives.full_bhavcopy_raw(dt)` — one
  HTTP request per business day, filtered to `SERIES == "EQ"`. Available from
  2020-01-01 onward; NSE holidays and network errors are skipped silently.
- **Augmentation (`augment_dataset`)** left-joins delivery onto prices, then:
  1. **forward-fills** delivery within each ticker to bridge holidays/gaps
     (respecting the no-backfill look-ahead invariant), and
  2. **zero-fills leading NaNs** — a ticker present in the price CSV before
     delivery coverage begins carries `DeliveryQty = 0 / DeliveryPct = 0`. This is
     the one place a stand-in value enters the panel; the two `inst_*` features
     then rank that `0` like any other observation.

No raw price value is ever synthesized or modified. Everything the model sees
downstream (the 19 features) is **derived** from this OHLCV + delivery base by
feature engineering — not fabricated.

## Conventions & guarantees

- **Input:** MultiIndex `('date', 'ticker')` augmented daily frame from
  `src/data_scraping.py` — OHLCV (`open/high/low/close/volume`) plus
  `DeliveryQty` / `DeliveryPct` (see [Where the data comes from](#where-the-data-comes-from)).
- **Per-ticker time series:** every rolling/shift stat is computed inside
  `groupby(level='ticker')`, so windows never bleed across symbols.
- **Complete windows only:** rolling ops use `min_periods == window`; partial
  warm-up rows become `NaN` and are dropped in a single terminal `dropna()`.
- **Cross-sectional normalization:** *every* raw feature is rank-normalized per
  date into a strict **`[-1, +1]`** band (see [Normalization](#cross-sectional-rank-normalization)).
  The formulas below describe the **raw** value that is then ranked — the model
  never sees the raw magnitude, only its within-day cross-sectional rank.
- **Zero look-ahead:** all features use current-or-past bars only. Only the
  *target* columns look forward.
- **Infinity guard:** raw features have `±inf → NaN` before ranking so blow-ups
  (e.g. divide-by-zero) can't dominate the cross-section.

Notation: `C_t` = close, `H_t` = high, `L_t` = low, `V_t` = volume on day `t`;
`r_t = ln(C_t / C_{t-1})` = daily log return.

---

## Family 1 — Momentum & Reversal (10 features)

Captures trend persistence across horizons, position relative to trend, and a
short-term reversal/lottery proxy.

| Feature | Formula (raw) | What it captures |
|---|---|---|
| `mom_logret_1d` | `ln(C_t / C_{t-1})` | 1-day return — very short-term momentum / reversal |
| `mom_logret_5d` | `ln(C_t / C_{t-5})` | 1-week cumulative return |
| `mom_logret_20d` | `ln(C_t / C_{t-20})` | 1-month momentum |
| `mom_logret_60d` | `ln(C_t / C_{t-60})` | 1-quarter momentum |
| `mom_logret_120d` | `ln(C_t / C_{t-120})` | 6-month momentum |
| `mom_logret_252d` | `ln(C_t / C_{t-252})` | 12-month momentum (classic academic momentum horizon) |
| `mom_close_sma20` | `C_t / SMA_20(C)` | Price vs. 1-month trend; `>1` = above trend |
| `mom_close_sma50` | `C_t / SMA_50(C)` | Price vs. ~2.5-month trend |
| `mom_close_sma200` | `C_t / SMA_200(C)` | Price vs. long-run trend (the "200-day MA" line) |
| `mom_max_dret_20d` | `max( r_{t-19..t} )` | Largest single-day log return in the trailing month — the **MAX / lottery effect** proxy; high values historically precede reversal |

Notes:
- Log returns are additive across time and symmetric, which is why every
  momentum horizon uses `ln(C_t / C_{t-n})` rather than simple returns.
- The three SMA-distance features are ratios (dimensionless), so they compare
  cleanly across stocks at very different price levels.

---

## Family 2 — Volatility & Risk (4 features)

Trailing realized risk and current drawdown state.

| Feature | Formula (raw) | What it captures |
|---|---|---|
| `vol_realized_20d` | `std( r_{t-19..t} )` | 1-month realized volatility (close-to-close) |
| `vol_realized_60d` | `std( r_{t-59..t} )` | 1-quarter realized volatility |
| `vol_parkinson_20d` | `sqrt( k · mean( ln(H/L)² ) )` over 20d, `k = 1/(4·ln 2)` | **Parkinson high-low volatility** — uses the intraday range, a more efficient vol estimator than close-to-close |
| `vol_drawdown_252d` | `C_t / max(C_{t-251..t}) − 1` | Distance below the trailing 1-year peak (≤ 0); `0` = at a new high, more negative = deeper drawdown |

Notes:
- `vol_parkinson_20d` complements the close-to-close realized-vol pair: it
  incorporates the day's full range, so it reacts to intraday risk that
  close-to-close vol misses.
- `vol_drawdown_252d` is a **state** feature (where the stock sits vs. its own
  1-year high), not a return — useful for regime/reversal interactions.

---

## Family 3 — Liquidity, Volume & Institutional (5 features)

Tradability, volume dynamics, and NSE delivery-based institutional conviction.

| Feature | Formula (raw) | What it captures |
|---|---|---|
| `liq_amihud` | `|r_t| / (C_t · V_t)` | **Amihud illiquidity** — price impact per unit traded value; high = illiquid |
| `liq_vol_var_20d` | `var( V_{t-19..t} )` | 20-day volume variance — instability/spikiness of trading activity |
| `liq_turnover_20d` | `V_t / mean(V_{t-19..t})` | Today's volume vs. its own 1-month average; `>1` = unusually active |
| `inst_delivery_qty` | `DeliveryQty` (carried through) | Shares actually taken to delivery (not intraday-squared) — institutional/positional flow magnitude |
| `inst_delivery_pct` | `DeliveryPct` (carried through) | Delivered volume as a % of traded volume — conviction / low-churn proxy |

Notes:
- Amihud uses **traded value** `C_t · V_t` in the denominator, so it's the
  return moved per rupee traded — a standard cross-sectional illiquidity signal.
- The two `inst_*` fields are NSE-specific "delivery" metrics (shares that
  changed hands for delivery rather than intraday round-trips), passed straight
  from the scraper and then rank-normalized like everything else. They are the
  pipeline's institutional-conviction signal.

---

## Cross-sectional rank normalization

Every raw feature above is transformed, **per date**, into `[-1, +1]`:

```
norm = 2 · (r − 1) / (n − 1) − 1
```

where `r` is the feature's rank within that day's cross-section (`method="average"`,
so ties share the average rank) and `n` is the number of valid observations that
day. Consequences:

- The daily **minimum → −1**, the daily **maximum → +1**, median ≈ `0`.
- Scale-free and outlier-robust: only the ordering within a day matters, so a
  single blown-up value can't dominate.
- A date with exactly one valid observation maps it to `0.0` (no cross-section
  to rank against). Warm-up / empty-cross-section cells stay `NaN` and are dropped.

This is why the model consumes pure cross-sectional ranks — comparing stocks
*against each other on the same day*, never against raw magnitudes.

---

## Targets (labels — never fed as features)

| Target | Formula | Horizon |
|---|---|---|
| `tgt_fwd_logret_1d` | `ln(C_{t+1} / C_t)` | Next-day forward log return |
| `tgt_fwd_logret_5d` | `ln(C_{t+5} / C_t)` | 5-day forward log return |

- Computed with a **negative** per-ticker shift (`shift(-1)` / `shift(-5)`) — the
  only forward-looking operation in the module.
- Kept as **raw** returns (not normalized), used only as the regression label in
  `src/tier1_trees.py`. The tree engine defaults to `tgt_fwd_logret_1d`.
- The trailing rows whose forward label is undefined are removed by the terminal
  `dropna()`, so every emitted row has both fully-populated features and a
  defined target.

---

## Output schema summary

```
19 feature columns  (all in [-1, +1])
 ├─ Momentum & Reversal ......... 10
 ├─ Volatility & Risk ...........  4
 └─ Liquidity/Volume/Inst .......  5
 2 target columns  (raw forward log returns)
```

Source of truth: `FEATURE_COLUMNS` and `TARGET_COLUMNS` in
`src/feature_creator.py`.
