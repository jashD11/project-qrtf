# Phase 4 — the rebuilt NSE database: validation record

**Status: all gates PASS, 2026-08-05.** Reproduce with
`python src/phase4_data/bhavcopy_validate.py`.

## What exists

| Artefact | Shape |
|---|---|
| `data/bhavcopy/panel_daily.parquet` | 5,027,424 rows · 2,821 entities · 3,094 days · 2013-01-01 → 2025-06-30 |
| `data/nse500_daily_ohlcv.parquet` | 2,684,294 rows · 1,244 tickers (engine schema) |
| `data/nse500_universe_mask.parquet` | 3,094 dates × 1,244 tickers, 500 names/day |
| `data/bhavcopy/index_daily.parquet` | 3,082 days, NIFTY-50 |
| `data/bhavcopy/corporate_actions.csv` | 796 detected actions |
| `data/bhavcopy/lifecycle.csv` | 2,821 entities: 2,096 live / 437 delisted / 288 acquired |
| `src/phase4_data/known_extremes.csv` | 25 audited extreme events (version-controlled) |

## Gate results

| Gate | Result |
|---|---|
| **1 REPRODUCTION** (blocking) | **PASS** — 1a 59/62 = 95.2%; 1b 0 impossible days; 1c 0.019% |
| 2 SURVIVORSHIP | 725 of 2,821 (25.7%) left the exchange in-window |
| 3 ACTION AUDIT | 796 actions, median implied ex-date return 0.87% |
| 4 RESIDUAL DISCONTINUITY | 0.025% of all panel bars, penny-stock dominated |
| **5 CAUSALITY** (blocking) | **PASS** — back-adjustment max \|Δ\| 2.2e-16; no look-ahead in selection |
| **6 RESIDUAL SWEEP** (blocking) | **PASS** — 25 extremes over 1,243,692 in-universe bars, **0 unexplained** |

## The six defects found, and how

Every one of these was surfaced by running the gate, not by reading the code.

### 1. NSE reissues the ISIN on a face-value split

`NESTLEIND INE239A01016 → INE239A01024` on 2024-01-05; `BEL INE263A01016 → …024`.
Keyed on raw ISIN, **421 companies** were cut into two histories: the split became
structurally undetectable (no prior bar inside the new ISIN's group), the old ISIN booked
a phantom −30% delisting, and the new one restarted with no rolling history.

**Fix:** link on the 7-character issuer stem, merging only when the two ISINs' date
ranges are **disjoint** — overlapping ranges mean two securities that co-existed (a second
share class), and those stay separate. 140 stems were correctly left split.

### 2. `CLOSE` is not the last trade

NSE's bhavcopy `CLOSE` is the **volume-weighted average of the final 30 minutes** — the
official settlement price — not the last traded price. That is ~20 bps of spurious daily
return on *every* name relative to an LTP-based feed.

**Fix:** `close` = `LAST`; the official VWAP is retained as `close_vwap` for a future
sensitivity run. Verified: median |Δ return| vs the reference went 2.09e-03 → **0.00e+00**
(bit-identical on typical days).

### 3. Weekend sessions were never downloaded

NSE trades some Saturdays and Sundays — Diwali *Muhurat* sessions, Budget Saturdays,
disaster-recovery drills. A weekday-only scan merged two days of return into one bar:
TATAMOTORS read **+36.4%** on 2019-10-29 instead of +15.3%, because the Sunday Muhurat bar
in between did not exist.

**Fix:** 13 weekend sessions recovered (2013-05-11, 2013-11-03, 2014-03-22, 2015-02-28,
2016-10-30, 2019-10-27, 2020-02-01, 2020-11-14, 2023-11-12, 2024-01-20, 2024-03-02,
2024-05-18, 2025-02-01). Calendar now matches the reference exactly — **0 missing days**.

### 4. Date columns drift format mid-archive, in *both* streams

`cm13JUL2020bhav.csv.zip` writes `13-Jul-20` where every other file writes `02-JAN-2013`;
2014–15 index files write `09/06/2014` where every other era writes `02-01-2013`. A strict
parse with `errors="coerce"` silently produced NaT and dropped **1 price day and 62 index
days**.

**Fix:** take the date from the **filename**, which is uniform across 12 years, in both
the price and index parsers.

### 5. The turnover ceiling was rejecting real splits

**ELECON 2024-07-19** has a price ratio of 0.5035 — a textbook 1:2 split — rejected
*solely* because `turnover_ratio = 7.43` exceeded a ceiling of 5.0. A split makes a share
cheaper and more accessible, so ex-date quantity routinely spikes well past 5×.

**Fix:** two-tier detection plus recalibration to 20.0. Detected actions went **633 → 796**
(+163: 17 rescued by the ISIN witness, ~146 by the ceiling), and in-universe unexplained
extremes went **56 → 25 → 0**.

**Tier A — the ISIN witness.** A reissue within ±3 bars is evidence *independent of price*
that a corporate action occurred, so the turnover test is skipped there. Of 421 handovers,
274 (65%) already matched a detected action at ±1 bar; the new ISIN typically begins the
bar *after* the ex-date (SBIN: action 2014-11-20, handover 2014-11-21). A witness alone is
never sufficient — EICHERMOT, BDL and AMRUTANJAN sit at ratio ≈ 1.00 on their handover
date (administrative reissues), and the jump and snap tests still exclude them.

**Guard against this being a blanket loosening:** the dry-run's planted genuine −90%
wipeout, which has no witness and collapsing turnover, must still be rejected. That
assertion is what makes this a calibration fix rather than switching the test off.

### 6. The reference panel is not clean ground truth

The 68-name Phase 2/3 panel — the intended measuring stick — carries days that are
physically impossible under NSE circuit limits: **INFY +302.8%** on 2015-04-24 and
**BEL +195.6%** on 2015-09-11, both on days when raw bhavcopy was flat. Its VEDL history
before 2024 is a different price series whose level ratio to ours drifts on 18% of days
before locking to exactly 1.0.

**Fix:** a **mechanical drop rule**, evaluated from properties of the reference alone
before any pass rate is computed, so it cannot be tuned to a verdict. A reference series
is disqualified if it contains a >35% day our panel contradicts, **or** its level ratio to
ours steps >1% on more than 5% of days. This disqualified 4 of 66: VEDL, ADANIENT, BEL,
INFY — each reported by name with its triggering evidence.

## Why gate 6 exists

Gate 1 can only see the 66 names the reference happens to contain — **95% of the universe
is invisible to it**. Gate 6 needs no external data: it asserts an internal property over
every in-universe bar of all 1,244 names, that no impossible return survives adjustment
unless a detected action sits within ±3 bars or the event is in the audited register.

The register (`src/phase4_data/known_extremes.csv`, version-controlled) holds 25 events in
three categories, separated by a clean diagnostic signature:

- **news (16)** — turnover *spikes* 2–80× and the ratio does **not** snap to a simple
  fraction. Real return the strategy should experience. JETAIRWAYS +123% (grounding),
  ZEEL +40% (Sony merger, 44.8×), CANBK +39% (PSU recapitalisation, 22.9×),
  INDUSINDBK/BANDHANBNK +46%/+42% on the 2020-03-26 COVID rebound.
- **demerger (8)** — turnover *collapses* to 0.0–0.4× and the ratio **does** snap.
  IIFL 0.462 (three-way demerger), EDELWEISS 0.55 (Nuvama), GFLLIMITED 0.111,
  TATACOMM 0.647 (Hemisphere), NIITLTD 0.231, STAR 0.462, OMKARCHEM 0.462, BCG 0.632.
- **penny (1)** — KSERASERA at Rs 0.40, below the Rs 20 detection floor.

## Known limitations

1. **Demergers are deliberately not adjusted.** Holders receive shares in the spun-off
   entity, so total wealth is roughly unchanged, but this panel cannot track the spin-off
   — so the parent books a fake loss. 8 events in 1.24M in-universe bars. Biases returns
   **downward**, i.e. conservative.
2. **Price return, not total return.** NSE does not adjust for ordinary dividends and they
   are far below any detection threshold. NIFTY-50 yield is 1.22–1.38%/yr, so a long book
   is understated by roughly that. Also conservative — but the *cross-sectional* tilt
   (PSUs and utilities yield far more than growth names) cannot be corrected after the
   fact and remains unmodelled.
3. **Sub-Rs-20 stocks are unadjusted by design.** The Rs 0.05 tick makes a Rs 2 stock land
   on exactly 3/2, so real and spurious ratios are indistinguishable there. Such names
   essentially never reach the top 500.
4. **12 index days are genuinely missing** (0.39%) — NSE never published those files,
   confirmed by clean 404s on re-probe. Dropped from scoring by the same mechanical rule.
5. **Slippage is not in the execution path.** See [`phase4_capacity.md`](phase4_capacity.md);
   at the Rs 1 cr operating point impact is ~3.8 bps against a ~14.7 bps statutory stack,
   but it is not yet charged.
