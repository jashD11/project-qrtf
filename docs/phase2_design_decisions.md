# Phase 2 Design Decisions

Living record of design decisions for the `PRODUCTION_ML` (live-data, tree-based)
pipeline. Status tags:
- **[DECIDED]** — settled, safe to build against.
- **[RECOMMENDED]** — proposed default, not yet explicitly confirmed.
- **[OPEN]** — genuine fork still to resolve.

---

## 1. Data acquisition & universe

**Data facts (from the shared Drive folder "15m_dataset"):**
- 86 NSE files = **68 stocks + 18 indices**. Each also has a `.BSE` twin (176 total).
- Per file: 15-minute bars, **2015 → 2025 (~10 yr)**, ~62k rows, ~3.7 MB.
- Schema: `timestamp, open, high, low, close, volume, oi`. `timestamp` is IST
  tz-aware (`+05:30`), session 09:15–15:30 = 25 bars/day. `oi` is all-zero (equities).

**Decisions:**
- **[DECIDED] Exchange = NSE only** (drop `.BSE`). Matches the NSE bhavcopy delivery
  layer; avoids dual-listing duplicates polluting the cross-section.
- **[DECIDED] Daily is derived from the 15-min source**, not a separate file. 10 yr
  of intraday is ample for daily 252-day features + multi-year training. One source
  of truth, all frequencies aligned.
- **[DECIDED] Retrieval via rclone (authenticated)**. Anonymous gdown rate-limits
  after ~54 files ("many accesses"); rclone with the user's Google account has far
  higher quotas and is resumable. `download_nse.py` (gdown) remains for convenience.
- **[RECOMMENDED] Universe = the 68 stocks.** Indices are *averages of baskets*, not
  peers of individual stocks — including them distorts the cross-sectional decile
  ranking. So exclude the 13 sector + 2 size indices from the tradeable universe.
  **Keep one broad index (NIFTY-500 or -50) aside** as the market series for the
  regime tier (§3). Implement as a **configurable stock/index filter in
  `consolidate.py`** so nothing is deleted and the choice is reversible.
  - 18 indices: broad = `NIFTY-50/100/500`; sector = `NIFTY-AUTO/BANK/ENERGY/
    FINSERV/FMCG/HEALTHCARE/INFRA/IT/METAL/MFG/OILGAS/PHARMA/REALTY`; size =
    `NIFTY-MIDCAP-150`, `NIFTY-SMLCAP-250`.

---

## 2. Frequency-parametrized feature pipeline ("pick a frequency flag")

Goal: resample the one 15-min source into `15min / 30min / 60min / daily` and let a
single frequency flag select which bar series feeds features + trees, so the
strategy sweep's "frequency axis" is real.

**Decisions:**
- **[DECIDED] D1 — Institutional features intraday = lagged broadcast.** Carry
  prior-day (t-1) `DeliveryQty/DeliveryPct` as a within-day constant across all
  intraday bars, so every frequency keeps the full 19 features. The 1-day lag avoids
  EOD look-ahead. (Delivery has no intraday analog — it's an end-of-day bhavcopy
  quantity.)
- **[RECOMMENDED] D2 — Window units are split:**
  - **Feature lookbacks → bars.** Reuse the same integers (`1/5/20/60/120/252`, SMA
    `20/50/200`, vol `20/60`) as *bar counts*. A bar = one row of the selected
    frequency (1 day on daily, 15 min on 15-min). Same integers auto-scale the
    horizon per frequency; warm-up stays small intraday. Rename `_Nd` suffixes to
    bar-neutral (e.g. `_Nb`) so they don't mislead.
  - **Walk-forward train/predict windows → trading days.** Keep `504/63` in *days*,
    not bars, so trees get enough training rows (504 bars @15min ≈ 20 days = too
    little). The walk-forward loop slides by calendar-day groups derived from each
    bar's timestamp. Windows become **per-frequency**.
- **[DECIDED] D3 — Resampling:** `resample_bars.py` produces `30/60min/daily` from
  the 15-min source (session-aware; no bin spans the overnight gap). 15-min is the
  source itself.
- **[DECIDED] Session-boundary targets:** null out forward targets that would span
  the overnight gap (target bar must resolve in the same session); keep *features*
  spanning the gap (overnight return is real signal). **Implemented** in
  `_compute_targets(null_cross_session=...)`, on only when `bars_per_day > 1`.
- **[DECIDED] Decoupled target retention (refinement).** The old single
  `dropna(how="any")` required *both* the 1-bar and 5-bar labels, so the
  restrictive 5-bar same-session constraint discarded rows valid for the 1-bar
  label — at 60min it nuked ~72% of rows. Now `create_features` keeps a row if
  its features are complete **and ≥1 target is defined**, leaving masked (NaN)
  target cells in place; `tier1_trees` drops the remaining per-target NaNs inside
  each fold. 60min retention went 319k → 959k rows, `tgt_fwd_logret_1b` fully
  populated.
- **[DONE] `FREQ_REGISTRY`** in `config.py`: `_FreqSpec(parquet, bars_per_day,
  train_days, predict_days, has_institutional)` for 15min/30min/60min/daily.
  `config.freq_spec()` resolves it. `StrategyConfig.frequency` is the live selector;
  `data_scraping.load_bars(freq)`, `feature_creator.create_features(df, freq)`, and
  `TreeAlphaEngine.from_frequency(freq)` all read from it. Feature lookbacks are
  bar counts (names `_Nb`); walk-forward windows are trading days (loop slides by
  calendar day via `searchsorted`, not by bar). **Verified end-to-end** on the real
  parquets (daily + 60min): [-1,+1] bounds hold, decile masks correct, no leakage.

---

## 3. Regime detection (Tier 2 HMM) redesign

**Sandbox baseline (`src/sandbox_run/tier2_regime.py`):** one synthetic index series;
2 features (daily log return; 20-day cumsum trend); `GaussianHMM(2, full)`; fit once
on all history; states remapped by ascending variance (0=calm, 1=panic); consumed by
Tier 3 as a binary gate. **Two limitations that matter now:** (a) all features come
from one series — no cross-sectional view; (b) fit-on-all-history = **look-ahead in
the regime labels**.

**Newly possible with the richer data:**
- Real broad market series (`NIFTY-500`) with 10-yr history.
- **Market-internal features impossible with one index:** cross-sectional
  **dispersion** (spread of the 68 stocks' returns), **average pairwise correlation**
  (spikes in panics — risk-on/off tell), **breadth** (% above MA / advance-decline),
  plus index **realized vol**, downside semivariance, aggregate turnover.

**Decisions:**
- **[RECOMMENDED] Frequency: run the HMM at *daily*** even for intraday strategies —
  regimes are slow macro backdrops. **Broadcast the daily regime onto intraday bars**
  (same lagged-broadcast pattern as delivery).
- **[RECOMMENDED] Causal fitting:** walk-forward / expanding-window refit (fit on
  trailing data, filter current state forward) to remove the look-ahead, consistent
  with the tree engine.
- **[RECOMMENDED] Standardize features** (z-score) before fitting — raw returns
  (~0.01) and avg correlation (~0.5) are on wildly different scales and the big one
  would dominate the Gaussian.
- **[RECOMMENDED] States:** start at **2** (keeps Tier 3 compatible), keep **3**
  sweepable via `StrategyConfig.hmm_states`. NOTE: Tier 3 currently hard-codes a
  binary gate (`NotImplementedError` for >2) — >2 states needs execution changes.
- **[OPEN] Integration with the ML alpha** (pivotal fork):
  - (a) **Gate** — regime scales the tree book (calm → 130/30, panic →
    dollar-neutral/cash). Mirrors `dynamic_tilt`; minimal change.
  - (b) **Feature** — feed regime label / state-probabilities into the trees so they
    learn regime-conditional alpha. Most ML-native. *Leaning here first.*
  - (c) **Conditional models** — separate tree ensemble per regime.

---

## 4. Orchestration (not yet built)

- **[OPEN] No `PRODUCTION_ML` orchestrator exists.** `run_pipeline.py` is still 100%
  SANDBOX. Need a mode-branched path (or `run_pipeline_ml.py`) chaining:
  `load bars(freq) → augment(delivery) → create_features(freq) → TreeAlphaEngine →
  regime → execution → DSR ledger`.
- **[OPEN] Back half unwired:** `tier1_trees` emits long/short masks, but nothing
  converts masks → portfolio returns → ledger. Phase 1 `tier3_execution` / `tier4_dsr`
  expect the old momentum-rank + regime inputs, not the tree masks.

---

## Build order (current)

```
✅ download_nse.py     gdown puller (rate-limited past ~54; rclone is the real path)
✅ consolidate.py      per-stock CSVs → one long 15-min Parquet  (+ stock/index filter)
✅ resample_bars.py    15-min → 30/60min/daily  (session-aware, validated)
✅ full dataset        68 stocks consolidated + resampled to 4 aligned parquets
✅ FREQ_REGISTRY + frequency-aware loader / feature_creator / tier1_trees
⬜ Tier 2 HMM redesign (per §3)
⬜ PRODUCTION_ML orchestrator + mask→returns→ledger
```
