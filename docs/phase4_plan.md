# Phase 4 — Rebuild the NSE database at 500 names and re-run the engine on it

**Status: IN EXECUTION, started 2026-08-04.** Drafted 2026-08-03 against the Phase 3
record in [`phase3_results.md`](phase3_results.md) and the roadmap in
[`phase3_presentation_notes.md`](phase3_presentation_notes.md) §4. Numbers in the
Context and probing sections were measured before the build; Part A/B forecasts are
being replaced by outcomes as they land.

| Step | State |
|---|---|
| A1 acquisition — `src/phase4_data/bhavcopy_download.py` | **built**, both eras smoke-tested; full download running |
| A2 panel — `src/phase4_data/bhavcopy_panel.py` | **built**, dry-run + real-data smoke pass |
| A3 universe — `src/phase4_data/universe.py` | **built**, dry-run passes |
| A5 validation — `src/phase4_data/bhavcopy_validate.py` | **built + run** — gate 1 **FAILS at 93.9% vs 95%** on sub-gate 1a; 1b/1c pass. See "Gate 1 outcome" below |
| B1 `create_features(universe_mask=…)` | **built + tested**, `None` is an exact no-op |
| B2 tier-3 delisting exit | **built + tested**, guard rejects implicit 0% exits |
| B3 config wiring (`daily_nse500`, `_Phase4Config`) | **built**, Phase 2/3 paths untouched |
| B4 smoke + headline run | pending the download |

**Corrections to this plan, found while building it** (the plan text below is left as
drafted; these supersede it):

1. **A4 is folded into A2/A3, not a separate step.** Both engine artefacts are emitted
   in the schema the existing loaders already expect, so Tier 2 needed no code change
   at all — repointing `_RegimeConfig.index_parquet` / `stock_parquet` is the whole
   integration. The index parquet carries `timestamp`/`ticker` columns, not `date`.
2. **The NIFTY-50 has three names in the archive, not two.** It is `S&P CNX Nifty` in
   2013 (the plan guessed `CNX Nifty`), then `CNX Nifty`, then `Nifty 50`. All three
   are matched.
3. **Two detector failure modes the plan did not anticipate**, both fixed:
   *non-contiguous bars* (a jump measured across a suspension or a hole in the archive
   is not a corporate action — detection now requires consecutive trading days), and
   *tick quantization* (NSE's Rs 0.05 tick makes a Rs 2 stock land on exactly 3/2, so
   rational snapping fires constantly on penny stocks — detection now requires a
   prior close ≥ Rs 20, which cannot mask a real forward action because a company does
   not split a share already trading near face value). The above-1 side of the ratio
   grid was also made sparse: a factor between 1 and 2 would be a 3:2 reverse split,
   which does not happen in India, and that region sat right on the ±25% jump
   threshold collecting false positives.
4. **`PREVCLOSE` is confirmed unusable** — as the plan said, but now also for UDiFF
   (`PrvsClsgPric`), which behaves identically.
5. **No new `run_pipeline_ml.py` flags were needed.** The plan proposed `--universe-top`
   and `--end`; `universe.py --build --top-n N` already produces a smaller mask, and the
   OHLCV export is restricted to selected names, so the smoke run shrinks by itself.
6. **Mixing Phase 3 and Phase 4 frequencies in one run is now rejected**, because they
   require different regime panels — a case the plan did not consider.

## Gate 1 outcome — the build is BLOCKED pending a decision

Five defects were found by running the gate, four of them fixed:

| Defect | Effect | Fix |
|---|---|---|
| **ISIN reissued on face-value splits** (NESTLEIND `INE239A01016→…24`, BEL `INE263A01016→…24`) | 421 companies split into two histories: split undetectable, old ISIN books a phantom −30% delisting, new one restarts rolling features | issuer-stem linking, merged only when date ranges are disjoint |
| **`CLOSE` is a 30-min VWAP, not the last trade** | ~20 bps of fake daily return on *every* name | `close` = `LAST`; VWAP retained as `close_vwap` |
| **Weekend sessions never downloaded** | Budget Saturdays, Diwali Muhurat Sundays, DR drills missing → two days' return merged into one bar (TATAMOTORS read +36.4% vs +15.3%) | 13 weekend sessions recovered |
| **`TIMESTAMP` format drift** (`13-Jul-20`) | one whole trading day silently coerced to NaT and dropped | date taken from the filename |
| **The reference panel is not clean ground truth** | ≥4 physically impossible days (INFY +302.8%, BEL +195.6%); VEDL's pre-2024 history is a different series | gate re-specified — see below |

**The gate was re-specified once, for cause.** The original "correlation > 0.999 on ≥95%
of names" cannot be met against a reference that itself contains impossible days: one
such day destroys a correlation while barely moving a median. Gate 1 now tests what it
exists to test, asymmetrically:

- **1a** typical-day fidelity, tolerance in **ticks not bps** (a 1 bp threshold is
  physically unachievable — one Rs 0.05 tick is 8.5 bps on a Rs 59 stock) → **93.9%, FAIL** (needs 95%)
- **1b** *zero* days where **our** series shows an impossible (>35%) return → **0, PASS**
- **1c** overall disagreement ≤0.1% → **0.052%, PASS**

Result after all fixes: median |Δ return| **0.00e+00**, median correlation **0.9989**,
disagreements **83 of 160,111 (0.052%)**, and **not one day** where our panel carries an
impossible return the reference contradicts — against **4** where the reference does.

**The four names failing 1a are VEDL, IOC, COALINDIA, TECHM.** VEDL is the demonstrably
broken reference series. Excluding it as a documented reference defect would give
62/65 = 95.4% and a pass — but that is a post-hoc exclusion chosen *after* seeing it
would flip the verdict, so it has not been applied. **Part B (the engine run) has not
been started.** The plan's own rule is that a failing gate 1 blocks it, and thresholds
have already been re-specified once; moving them again to recover 1.1 points would be
indefensible.

## Context

Phase 3 ended on an honest negative result: the trees rank NSE stocks genuinely
(rank-IC ≈ 0.02 daily) but the daily `long_only` survivor reaches only **SR +0.57 /
DSR 0.737** against a 0.95 gate. `docs/phase3_presentation_notes.md` §4.1 names
**breadth** as the highest-leverage fix: 68 names is too few for `IR ≈ IC·√breadth`
to produce a credible Sharpe.

The binding arithmetic (computed with the repo's own `_min_track_record_length`,
reproducing the published minTRL of ~11.6k days):

| History window | Scored days | SR needed for DSR 0.95 | vs current 0.57 |
|---|---|---|---|
| current, 2018-02→2025-02 | 1,732 | 0.96 | 1.69× |
| **2013→2025 (chosen)** | **2,596** | **0.85** | **1.49×** |
| 2012→2025 | 2,846 | 0.82 | 1.44× |

**More history alone can never close this** — at SR 0.57 the gate needs ~47 years,
which does not exist for the Indian market in usable form. History only lowers the
bar; **breadth has to clear it.** Going 68 → 500 names is a `√(500/68) = 2.71×`
theoretical ceiling, so the required 1.49× is plausible but not free.

**Scope.** The goal is a database the existing engine actually runs on, end to end:
adjusted OHLCV → the same 17 features → Tiers 1–4 → a fresh DSR at honest N. That is
wider than "acquisition + validation" (which would have stopped at a validated price
panel and left the engine unwired), and the plan below is scoped accordingly.

---

## What live probing established (all verified this session)

**NSE archives are open and cheap.** `nsearchives.nseindia.com` served every era with
nothing but a `User-Agent` header — no cookies, no session priming. Critically, **one
request returns every listed stock for that day**, so cost scales with *days*, not
*stocks*: the 68-name and 500-name builds cost identically to download. The
cookie-gated `www.nseindia.com/api/*` surface — the one that actually rate-limits, and
which returned a hard connection failure when probed — is **not needed at all**.

| Stream | URL pattern | Coverage | Carries |
|---|---|---|---|
| prices (old) | `/content/historical/EQUITIES/<YYYY>/<MON>/cm<DD><MON><YYYY>bhav.csv.zip` | → 2024-07 | OHLCV, **ISIN**, turnover |
| prices (UDiFF) | `/content/cm/BhavCopy_NSE_CM_0_0_0_<YYYYMMDD>_F_0000.csv.zip` | 2024-07 → | OHLCV, **ISIN**, turnover |
| index | `/content/indices/ind_close_all_<DDMMYYYY>.csv` | **2013-01-02** → | all NSE index closes |
| delivery | `/products/content/sec_bhavdata_full_<DDMMYYYY>.csv` | 2020 → | delivery qty/pct, no ISIN |

**Two hard boundaries set the window.** ISIN — the survivorship key — begins in **2012**
(2011 files have 11 columns and no ISIN; 2012 has 13 with ISIN). The index archive
begins in **2013** (2012-01-02 is a 404). Tier 2's regime detector needs a market series
across the entire window, so the panel **starts 2013-01-01** — the later of the two.
That buys a real NIFTY-50 over the whole sample instead of a synthetic composite
spliced onto a real one, for a cost of 250 days.

**Pacing.** Zero-delay bursts failed; **3 s spacing was reliable** across ~25 probe
requests. Plan on 2.5 s with retry/backoff.

**Universe ceiling.** EQ+BE names per day: 1,306 (2010) → 1,524 (2016) → 1,876 (2022)
→ ~2,000 (2025). The *liquid* slice is far smaller; top 500 by ADV is the realistic
tradeable frontier, consistent with the Phase 3 capacity study.

### The hypothesis that failed

I expected NSE's `PREVCLOSE` to be the **corporate-action-adjusted** prior close, which
would have made split/bonus handling free. **It is not.** Tested on the NESTLEIND 1:10
split (ex-date 2024-01-05):

```
close(t-1) = 27116.40   close(t) = 2666.40   PREVCLOSE(t) = 27116.40
naive close/close-1 = -90.2%     close/PREVCLOSE-1 = -90.2%
```

Both carry the artefact identically. There is also **no static corporate-actions file**
on the archive host (all candidate paths 404), and the only API that has them is the
cookie-gated surface this design avoids.

**Consequence:** corporate-action adjustment must be *detected from the data*, and
detection accuracy is the single largest risk in this plan. It gets its own module and
its own acceptance gate.

---

## Part A — Build the database

### A1. Acquisition — `src/production_ml/bhavcopy_download.py`

Mirrors the existing `download_nse.py` contract (paced, idempotent, resumable, per-file
failures counted not fatal) so the two acquisition scripts read alike.

- Era router: date → URL, handling the 2024-07 old-bhavcopy → UDiFF cutover (discovered
  by probing, not hard-coded).
- Raw files land under `data/bhavcopy/{eq,udiff,index}/` (gitignored), stored
  as-received; an existing non-empty file is never re-fetched, so a killed run resumes.
- Holidays return 404 and are recorded in `data/bhavcopy/_missing.json` with the HTTP
  status, so a genuine outage stays distinguishable from a market holiday.
- Flags: `--start`, `--end`, `--delay` (default 2.5), `--retries`, `--kinds`, `--list-only`.

**Delivery is deliberately not downloaded.** It only exists from 2020, and
`feature_creator.create_features` drops any row with a NaN feature — so enabling the
institutional family would silently discard everything before 2020 and truncate a
12-year panel to 5. This build stays on the **same 17 price-only features** as Phase 3,
which also keeps the comparison against the 68-name result like-for-like. (Consistent
with the existing `delivery-data-excluded` decision.)

**Budget** — ~3,100 trading days, 2013-01 → 2025-06:

| Stream | Requests | Wall clock @2.5 s | Disk |
|---|---|---|---|
| prices (old + UDiFF) | ~3,100 | ~2.3 h | ~300 MB |
| index | ~3,100 | ~2.3 h | ~25 MB |
| **total** | **~6,200** | **~4.6 h** (budget 6 h with retries) | **~325 MB** |

One overnight run; resumable, so an interruption costs nothing. 155 GB free.

### A2. Panel construction — `src/production_ml/bhavcopy_panel.py`

Parses raw files into one long-format `(date, isin)` Parquet. Four problems, in order:

**(a) Identity.** Key on **ISIN**, carry `symbol` as a display alias. A rename is then
invisible (same ISIN, new symbol) and needs no mapping table. Cross-check against
`symbolchange.csv` (68 KB, live) and *report* disagreements rather than trusting either
blindly. The canonical `ticker` exported downstream is the **last symbol observed for
that ISIN**, so a renamed company is one continuous series, not two.

**(b) Series migration.** Include **EQ and BE**, recording series per row. Surveillance
moves EQ→BE→EQ; excluding BE would make a live stock look delisted and then re-IPO'd.
This is a real source of phantom delistings and is cheap to avoid.

**(c) Corporate actions — the hard part.** No external source exists, so detect with
three independent signals that must agree:

1. **Price jump** — `close_t / close_{t-1}` outside ±25%.
2. **Turnover invariance** — a k:1 split divides price by k and multiplies quantity by
   k, so rupee turnover `TOTTRDVAL` stays roughly continuous while `TOTTRDQTY` jumps
   inversely. A genuine crash drops turnover value too. This discriminates corporate
   actions from real moves **with no external data**, and is the strongest signal.
3. **Rational snapping** — the ratio lands within 1% of a simple fraction (1/2, 2/5,
   1/5, 1/10, 3/4, 2, …). Splits and bonuses are always simple ratios.

Every detected factor is written to `data/bhavcopy/corporate_actions.csv` with date,
ISIN, factor and all three signal values — an auditable ledger, not a black box.
Adjustment is applied **backwards** (rescale history before the ex-date), which leaves
all past *returns* unchanged; an assertion enforces this.

**Stated limitation:** ordinary dividends (0.5–3%) fall below any detection threshold
and NSE does not adjust for them, so the panel is a **price-return series, not total
return** — the same basis as the existing Drive data (A5 test 1 confirms this).

**(d) Delisting and gaps.** The point-in-time panel is survivorship-free *by
construction* — a stock that died in 2019 is in the files until its last trading day
and then stops. That raggedness is already handled: `create_features` drops rows with
any NaN feature, so a name enters after its warm-up and leaves when its data ends. **No
feature or tree change is needed for raggedness** (verified against LICI/ZOMATO, which
already start mid-sample in the current panel).

The panel emits an explicit terminal-return column, consumed by A6:

- **Delisting** = ISIN absent ≥20 consecutive trading days, never reappears, no
  symbol-change record. Terminal return frozen at **−30%** (Shumway 1997), with a
  declared `{0%, −30%, −100%}` sensitivity on a separate ledger — the same
  freeze-before-test discipline the buffer sweep follows.
- **Acquisition/merger** — ISIN vanishes but a symbol-change record links it, or the
  final 20-day return is positive → **0%** exit, not −30%.
- **Suspension** — absent then reappears → a gap, not a delisting. The return *across*
  the gap is nulled so a trading halt cannot manufacture a fake return.

### A3. Point-in-time universe — `src/production_ml/universe.py`

Never select on today's NIFTY 500 membership; that *is* survivorship bias. Select from
the panel itself:

- At each quarter start (aligned to the existing 63-day walk-forward refit cadence),
  rank every EQ/BE name by **median daily rupee turnover over the trailing 252 days**,
  using only data strictly before that date.
- Require ≥200 of those 252 days actually traded (drops near-dormant listings).
- Take the **top 500**, held fixed for the quarter — no intra-quarter look-ahead, and
  universe churn is bounded and measurable.
- Emit `data/bhavcopy/universe_mask.parquet` as `(date, ticker) → bool`.

### A4. Engine-schema emit

Two artefacts in exactly the shapes the existing loaders expect, so nothing downstream
has to learn a new format:

- `data/nse500_daily_ohlcv.parquet` — columns `timestamp, ticker, open, high, low,
  close, volume`, matching `data/daily_ohlcv.parquet` byte-for-byte in schema
  (`data_scraping.load_bars` renames `timestamp`→`date` and MultiIndexes it unchanged).
- `data/nse500_daily_index.parquet` — the NIFTY-50 daily close series. Handles the
  index's own rename (`CNX Nifty` in 2013 → `Nifty 50` from ~2018) by matching either
  label. This is what `_RegimeConfig.index_parquet` will point at.

### A5. Validation — `src/production_ml/bhavcopy_validate.py`

A database nobody has checked is worth nothing. Five gates:

1. **Reproduce the existing 68.** Rebuild those tickers' daily adjusted closes from
   bhavcopy over 2015-02→2025-02 and compare against `data/daily_ohlcv.parquet`.
   Gate: per-name daily-return correlation **> 0.999** and median |Δ return| **< 1 bp**
   on **≥95%** of names. This turns the corporate-action detector from a guess into a
   measured thing — any failure is almost certainly a missed adjustment. Failing names
   are listed individually, never aggregated away.
2. **Survivorship counter.** How many ISINs in the 2013 top-500 are still listed in
   2025 — the number that quantifies what the 68-name universe was blind to.
3. **Corporate-action audit.** The 20 largest detected factors printed for manual
   check against public split/bonus records.
4. **Residual discontinuity.** Post-adjustment |daily return| > 40% events should fall
   to roughly the genuine-crash rate.
5. **Causality assertions.** Universe mask at date *t* uses only data before *t*;
   back-adjustment leaves all prior returns unchanged.

**This is the gate.** If test 1 misses its threshold, the detector is wrong and Part B
must not run — a panel with undetected splits injects fake ±50% returns straight into
the tree labels.

---

## Part B — Wire it into the engine and re-run

### B1. `feature_creator.create_features` — universe-aware normalization

Rolling per-ticker features must use each name's **full** history, but the
cross-sectional rank-normalization to [−1,+1] must run on **that date's selected 500
only** — otherwise ranks encode names that were not tradeable. Today both happen in one
pass, so `create_features` gains an optional `universe_mask` argument: mask applied
after the rolling stage, before `_rank_normalize`. Default `None` preserves current
behaviour exactly, so the Phase 3 results stay reproducible.

### B2. `tier3_execution.py` — close the delisting leak

Line 289 computes `(weights * fwd).sum(axis=1, min_count=1)`, and pandas `.sum()` skips
NaN — so a delisted name's final-day return, NaN because `price_wide.shift(-1)` has no
successor, is **silently treated as zero**. Delisting losses vanish and survivorship
bias walks back in through the exit. Fix: fill the terminal cell from the panel's
delisting-return column before the sum, and assert no held position ever exits at an
implicit 0%.

### B3. Config wiring

Additive `_BhavcopyConfig` block (paths, window, delay, top-N, delisting return,
detection thresholds), plus a **new** `FREQ_REGISTRY` entry `daily_nse500` pointing at
the new Parquet, and `_RegimeConfig` repointed at the new index/stock files. The
existing `daily` entry is left untouched so Phase 3 remains re-runnable.

**Intraday cannot be expanded** — bhavcopy is daily-only. Acceptable: intraday already
died on costs (every cell → −100%), so the honest family was already the 3 daily styles.

### B4. Smoke run, then the real run

```bash
# smoke: 2 years, top 100, one style — proves the wiring, ~5 min
python run_pipeline_ml.py --frequencies daily_nse500 --styles long_only \
    --target tgt_fwd_logret_5b --skip-dsr --universe-top 100 --end 2015-01-01

# headline: full window, top 500, 3 styles, honest N=3
python run_pipeline_ml.py --frequencies daily_nse500 --target tgt_fwd_logret_5b
```

**Compute forecast** — benchmarked the actual three learners at the frozen
hyperparameters on this machine:

| Universe | Train rows/fold | Sec/fold | 41 folds (2013→2025) |
|---|---|---|---|
| 68 | 34,272 | 12.5 | 9 min |
| 200 | 100,800 | 38.8 | 27 min |
| **500** | **252,000** | **108.5** | **~74 min** |

RandomForest is 97% of that (105.5 s of 108.5 s). Tier 1 is fit once per frequency and
shared across the three styles, so a full daily grid is **~1.5 h**, not 3×. Note that
lowering `max_depth` to speed it up would change a *frozen* hyperparameter — a
research-integrity decision, not an engineering one. Keep it frozen.

The result is reported as a distribution across the 3 style cells with DSR at honest
N=3, per the existing `--sensitivity`/freeze-before-test discipline. Whether it clears
0.95 is the finding, not the target.

---

## Files

**New** (`src/production_ml/`, each with the repo's standard `__main__` dry-run):
`bhavcopy_download.py` · `bhavcopy_panel.py` · `universe.py` · `bhavcopy_validate.py`

**Modified:** `config.py` (additive: `_BhavcopyConfig`, `daily_nse500` registry entry,
`_RegimeConfig` repoint) · `feature_creator.py` (optional `universe_mask` arg) ·
`tier3_execution.py` (delisting return) · `run_pipeline_ml.py` (`--universe-top`,
`--end` flags) · `.gitignore` (`data/bhavcopy/`)

**New docs:** `docs/phase4_data.md` (the validation record) ·
`docs/phase4_results.md` (the re-run)

**Reused, not rewritten:** the pacing/retry/idempotence pattern from `download_nse.py`,
the consolidation and read-back-assertion pattern from `consolidate.py`, the
ragged-panel tolerance already in `create_features`, the whole Tier 1–4 stack.

---

## Verification

```bash
# 1. acquisition smoke — both eras, list-only first
python src/production_ml/bhavcopy_download.py --start 2013-01-01 --end 2013-01-10 --list-only
python src/production_ml/bhavcopy_download.py --start 2024-07-01 --end 2024-07-10

# 2. module dry-runs — synthesized panels, invariant assertions
python src/production_ml/bhavcopy_panel.py
python src/production_ml/universe.py

# 3. full acquisition — overnight, resumable
python src/production_ml/bhavcopy_download.py --start 2013-01-01 --end 2025-06-30

# 4. build, then GATE on validation
python src/production_ml/bhavcopy_panel.py --build
python src/production_ml/universe.py --build
python src/production_ml/bhavcopy_validate.py     # must pass test 1 before step 5

# 5. engine
python run_pipeline_ml.py --frequencies daily_nse500 --styles long_only \
    --target tgt_fwd_logret_5b --skip-dsr --universe-top 100 --end 2015-01-01
python run_pipeline_ml.py --frequencies daily_nse500 --target tgt_fwd_logret_5b

# 6. regression — Phase 3 must still reproduce unchanged
python run_pipeline_ml.py --frequencies daily --target tgt_fwd_logret_5b   # expect SR +0.57
```

Step 6 matters as much as step 5: the `create_features` and `tier3_execution` changes
must be provably no-ops on the old path, or the new result is not comparable to the old
one.
