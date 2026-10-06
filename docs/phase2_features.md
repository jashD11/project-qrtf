# Phase 2 Feature Dictionary

Complete reference for every feature produced by
`src/production_ml/feature_creator.py` (the `PRODUCTION_ML` tree pipeline).
**Up to 19 predictive features** across three families, plus **2 forward-return
targets**.

## Frequency awareness (bars, not days)

The factory runs on **any bar frequency** — `15min`, `30min`, `60min`, or
`daily` — selected by a single flag (`StrategyConfig.frequency`, resolved through
`config.FREQ_REGISTRY`). All four bar series are session-aware resamples of the
one 15-min source of truth, so they share one timeline.

Because of this, **every lookback is counted in _bars_, not days.** A "bar" is one
row of the selected frequency (one day on daily data, one 15-minute interval on
15-min data). The same integer horizons auto-scale per frequency, so feature
names carry a bar-neutral **`_Nb`** suffix (`mom_logret_20b` = 20 bars back,
whatever the frequency). The `date` index level holds each bar's timestamp, so
cross-sectional ranking "per date" means **per bar** — comparing stocks at the
same instant. See [`phase2_design_decisions.md`](phase2_design_decisions.md) §2.

Two things flex by frequency:
- **Institutional family is end-of-day only** (delivery has no intraday analog).
  On daily bars it joins directly; intraday it is broadcast from the **prior
  trading day** (t-1 lag, avoids seeing an EOD quantity mid-session). When
  delivery data is unavailable, the family is dropped and the factory emits **17**
  features instead of 19.
- **Cross-session targets are nulled intraday** — a forward label that would span
  the overnight gap is masked (see [Targets](#targets--labels--never-fed-as-features)).

## Where the data comes from

The feature factory consumes an **augmented bar frame** built by
`src/production_ml/data_scraping.py`. That frame is a **hybrid** of two very
different sources — worth understanding before trusting any feature:

| Data | Columns | Source | Scraped live? |
|---|---|---|---|
| OHLCV bars (all freqs) | `open/high/low/close/volume` | `data/{15min,30min,60min,daily}_ohlcv.parquet`, all resampled from the one 15-min source | **No** — read from disk |
| Institutional delivery | `DeliveryQty`, `DeliveryPct` | NSE full bhavcopy via jugaad-data | **Yes** |

- **Prices are not scraped.** `load_bars(frequency)` reads OHLCV from the Parquet
  registered for that frequency. Their provenance is **external and user-supplied**
  (per-stock CSVs pulled from a shared Drive folder, consolidated + resampled by
  `download_nse.py` → `consolidate.py` → `resample_bars.py`) — the pipeline neither
  fetches prices at feature time nor generates them. All files live under the
  gitignored `data/` directory.
- **Only delivery metrics are scraped.** `scrape_nse_delivery_data()` pulls two
  columns (`DELIV_QTY`, `DELIV_PER`) from NSE's `sec_bhavdata_full_DDMMYYYY.csv`
  (host `nsearchives.nseindia.com`) via `NSEArchives.full_bhavcopy_raw(dt)` — one
  HTTP request per business day, filtered to `SERIES == "EQ"`. Available from
  2020-01-01 onward; NSE holidays and network errors are skipped silently.
- **Augmentation (`augment_dataset`)** joins delivery onto prices (daily: exact
  join; intraday: t-1 lagged broadcast onto every bar of the day), then:
  1. **forward-fills** delivery within each ticker to bridge holidays/gaps
     (respecting the no-backfill look-ahead invariant), and
  2. **zero-fills leading NaNs** — a ticker present in prices before delivery
     coverage begins carries `DeliveryQty = 0 / DeliveryPct = 0`. This is the one
     place a stand-in value enters the panel; the two `inst_*` features then rank
     that `0` like any other observation.

No raw price value is ever synthesized or modified. Everything the model sees
downstream is **derived** from this OHLCV + delivery base by feature engineering —
not fabricated.

## Conventions & guarantees

- **Input:** MultiIndex `('date', 'ticker')` augmented bar frame from
  `src/production_ml/data_scraping.py` — OHLCV (`open/high/low/close/volume`) plus
  (when available) `DeliveryQty` / `DeliveryPct` (see [Where the data comes from](#where-the-data-comes-from)).
- **Per-ticker time series:** every rolling/shift stat is computed inside
  `groupby(level='ticker')`, so windows never bleed across symbols.
- **Complete windows only:** rolling ops use `min_periods == window`; partial
  warm-up rows become `NaN` and are dropped.
- **Cross-sectional normalization:** *every* raw feature is rank-normalized per
  bar into a strict **`[-1, +1]`** band (see [Normalization](#cross-sectional-rank-normalization)).
  The formulas below describe the **raw** value that is then ranked — the model
  never sees the raw magnitude, only its within-bar cross-sectional rank.
- **Zero look-ahead:** all features use current-or-past bars only. Only the
  *target* columns look forward.
- **Infinity guard:** raw features have `±inf → NaN` before ranking so blow-ups
  (e.g. divide-by-zero) can't dominate the cross-section.

Notation: `C_t` = close, `H_t` = high, `L_t` = low, `V_t` = volume at bar `t`;
`r_t = ln(C_t / C_{t-1})` = per-bar log return. Window subscripts count **bars**.

---

## Family 1 — Momentum & Reversal (10 features)

Captures trend persistence across horizons, position relative to trend, and a
short-term reversal/lottery proxy.

Horizon interpretations below assume **daily** bars; at intraday frequencies the
same bar counts scale down proportionally (e.g. `mom_logret_20b` is 20 bars ≈ ~3
sessions at 60-min, not a month).

| Feature | Formula (raw) | What it captures |
|---|---|---|
| `mom_logret_1b` | `ln(C_t / C_{t-1})` | 1-bar return — very short-term momentum / reversal |
| `mom_logret_5b` | `ln(C_t / C_{t-5})` | 5-bar cumulative return (~1 week daily) |
| `mom_logret_20b` | `ln(C_t / C_{t-20})` | 20-bar momentum (~1 month daily) |
| `mom_logret_60b` | `ln(C_t / C_{t-60})` | 60-bar momentum (~1 quarter daily) |
| `mom_logret_120b` | `ln(C_t / C_{t-120})` | 120-bar momentum (~6 months daily) |
| `mom_logret_252b` | `ln(C_t / C_{t-252})` | 252-bar momentum (~12 months daily — classic academic momentum horizon) |
| `mom_close_sma20` | `C_t / SMA_20(C)` | Price vs. 20-bar trend; `>1` = above trend |
| `mom_close_sma50` | `C_t / SMA_50(C)` | Price vs. 50-bar trend |
| `mom_close_sma200` | `C_t / SMA_200(C)` | Price vs. long-run trend (the "200-bar MA" line) |
| `mom_max_dret_20b` | `max( r_{t-19..t} )` | Largest single-bar log return in the trailing 20 bars — the **MAX / lottery effect** proxy; high values historically precede reversal |

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
| `vol_realized_20b` | `std( r_{t-19..t} )` | 20-bar realized volatility (close-to-close) |
| `vol_realized_60b` | `std( r_{t-59..t} )` | 60-bar realized volatility |
| `vol_parkinson_20b` | `sqrt( k · mean( ln(H/L)² ) )` over 20 bars, `k = 1/(4·ln 2)` | **Parkinson high-low volatility** — uses the bar range, a more efficient vol estimator than close-to-close |
| `vol_drawdown_252b` | `C_t / max(C_{t-251..t}) − 1` | Distance below the trailing 252-bar peak (≤ 0); `0` = at a new high, more negative = deeper drawdown |

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
| `liq_vol_var_20b` | `var( V_{t-19..t} )` | 20-bar volume variance — instability/spikiness of trading activity |
| `liq_turnover_20b` | `V_t / mean(V_{t-19..t})` | This bar's volume vs. its own 20-bar average; `>1` = unusually active |
| `inst_delivery_qty` | `DeliveryQty` (carried through) | Shares actually taken to delivery (not intraday-squared) — institutional/positional flow magnitude |
| `inst_delivery_pct` | `DeliveryPct` (carried through) | Delivered volume as a % of traded volume — conviction / low-churn proxy |

The two `inst_*` features are the only frequency-conditional family: present at
daily (direct join) and intraday (t-1 lagged broadcast) when delivery data is on
disk, otherwise dropped (17-feature schema).

Notes:
- Amihud uses **traded value** `C_t · V_t` in the denominator, so it's the
  return moved per rupee traded — a standard cross-sectional illiquidity signal.
- The two `inst_*` fields are NSE-specific "delivery" metrics (shares that
  changed hands for delivery rather than intraday round-trips), passed straight
  from the scraper and then rank-normalized like everything else. They are the
  pipeline's institutional-conviction signal.

---

## Cross-sectional rank normalization

Every raw feature above is transformed, **per bar** (i.e. per value of the `date`
index level, which holds each bar's timestamp), into `[-1, +1]`:

```
norm = 2 · (r − 1) / (n − 1) − 1
```

where `r` is the feature's rank within that bar's cross-section (`method="average"`,
so ties share the average rank) and `n` is the number of valid observations at
that bar. Consequences:

- The per-bar **minimum → −1**, the per-bar **maximum → +1**, median ≈ `0`.
- Scale-free and outlier-robust: only the ordering within a bar matters, so a
  single blown-up value can't dominate.
- A bar with exactly one valid observation maps it to `0.0` (no cross-section to
  rank against). Warm-up / empty-cross-section cells stay `NaN` and are dropped.

This is why the model consumes pure cross-sectional ranks — comparing stocks
*against each other at the same bar*, never against raw magnitudes.

---

## Targets (labels — never fed as features)

| Target | Formula | Horizon |
|---|---|---|
| `tgt_fwd_logret_1b` | `ln(C_{t+1} / C_t)` | Next-bar forward log return |
| `tgt_fwd_logret_5b` | `ln(C_{t+5} / C_t)` | 5-bar forward log return |

- Computed with a **negative** per-ticker shift (`shift(-1)` / `shift(-5)`) — the
  only forward-looking operation in the module.
- Kept as **raw** returns (not normalized), used only as the regression label in
  `src/production_ml/tier1_trees.py`. The tree engine defaults to `tgt_fwd_logret_1b`.
- **Cross-session masking (intraday only):** a forward label whose target bar
  falls in a later session (spanning the overnight gap) is set to `NaN`, so the
  last few bars of each intraday day carry no label. Daily bars are never masked —
  the next bar *is* the next session, which is exactly the target.
- **Decoupled retention:** a row is kept if its features are complete **and at
  least one** target is defined. The two horizons are independent, so a row valid
  for the 1-bar label survives even when its 5-bar label is masked (critical
  intraday, where the 5-bar same-session constraint would otherwise discard most
  rows). `tier1_trees.py` drops the remaining per-target `NaN`s inside each
  walk-forward fold, training only on rows whose chosen target is defined.

---

## Output schema summary

```
17 or 19 feature columns  (all in [-1, +1])
 ├─ Momentum & Reversal ......... 10
 ├─ Volatility & Risk ...........  4
 ├─ Liquidity / Volume ..........  3
 └─ Institutional (optional) ....  2   ← dropped when delivery unavailable
 2 target columns  (raw forward log returns; may carry masked NaN cells)
```

Source of truth: `feature_columns()` / `FEATURE_COLUMNS` and `TARGET_COLUMNS` in
`src/production_ml/feature_creator.py`; the frequency table lives in
`config.FREQ_REGISTRY`.
