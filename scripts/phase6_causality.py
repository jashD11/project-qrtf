#!/usr/bin/env python3
"""
Phase 6 Part B, Stage 6 — causality of the new features and of Tier 1 (docs/phase6_plan.md §B.4).

``no_lookahead_check.py`` perturbs Tier 3's inputs only. This checks everything upstream
of it, from raw inputs.

1. **Anchor.** The in-memory rebuild of every Phase 6 panel reproduces the panels on disk
   exactly, so what is tested below is what the builders wrote.
2. **Feature causality.** Every raw input after a cut D is replaced with noise:
   - prices, volumes, turnover and trade counts;
   - index levels and valuations;
   - F&O closes, traded values and open interest;
   - the spread panel.

   Every Phase 6 panel, and the full ``p17xo`` feature matrix (the 17 included), must
   come back **bit-identical** on dates ≤ D. This runs at three cuts.
3. **Canary.** A feature shifted one day early (``ret_intraday_21b``) must be caught.
4. **Tier 1.** A walk-forward is refit (LightGBM, the 21b label) on the real and the
   perturbed matrices, with D on the second day of a scoring block:
   - With the label-overlap purge **on**, every prediction ≤ D must be identical.
   - With it **off**, the training labels of the last 21 bars reach past D and the
     predictions must move. That is the §A.7 leak, and it is this check's own canary.
5. **A future corporate action.** A 2:1 split is planted after D. Back-adjustment
   rescales the history before it, so every new panel on dates ≤ D must be unchanged
   (to 1e-9: rescaling moves the last bit of a float). Which of the 17 move is reported
   but does not gate: the 17 are F0, frozen.

Read-only: fits nothing that is kept, and writes only its log.

    python scripts/phase6_causality.py          # ~25 min
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
sys.argv = sys.argv[:1]

import run_pipeline_ml as R  # noqa: E402
from src.phase6_features import _common as C  # noqa: E402
from src.phase6_features import fo_archive as FO  # noqa: E402
from src.phase6_features import futures as FU  # noqa: E402
from src.phase6_features import index_series as IS  # noqa: E402
from src.phase6_features import market as MK  # noqa: E402
from src.phase6_features import options as OP  # noqa: E402
from src.phase6_features import price_features as PF  # noqa: E402
from src.phase6_features import sector as SE  # noqa: E402
from src.production_ml.feature_creator import (  # noqa: E402
    F1_MARKET, FEATURE_SETS, create_features, present_feature_columns,
)
from src.production_ml.tier1_trees import TreeAlphaEngine  # noqa: E402

FREQ = "daily_nse500"
FEATURE_SET = "p17xo"
TIER1_TARGET = "tgt_fwd_logret_21b"
TIER1_BLOCK = 4          # the cut sits on the 2nd day of this OOS scoring block
CUT_FRACTIONS = (0.62, 0.85)
CANARY = "ret_intraday_21b"
SPLIT_TICKER = "INFY"
SEED = 11
PANEL_COLS = ["date", "ticker", "symbol", "open", "high", "low", "close", "volume",
              "turnover", "trades", "adj_factor"]


# ── Inputs and the in-memory rebuild ───────────────────────────────────────────

def load_inputs() -> dict:
    panel = pd.read_parquet(C.PANEL, columns=PANEL_COLS)
    tickers = set(pd.read_parquet(C.ENGINE_OHLCV, columns=["ticker"])["ticker"].unique())
    panel = panel[panel["ticker"].isin(tickers)].sort_values(["ticker", "date"]).reset_index(drop=True)
    return {
        "panel": panel,
        "index": IS.load(),
        "fo": FO.load(),
        "spread": C.read(PF.SPREAD_CS, "python src/phase4_data/spread.py --build"),
    }


def build_panels(inp: dict) -> dict[str, pd.DataFrame]:
    """Every Phase 6 panel from in-memory inputs, step for step as the builders do."""
    panel = inp["panel"].copy()
    g = panel.groupby("ticker", sort=False)
    panel["prev_date"], panel["prev_close"] = g["date"].shift(1), g["close"].shift(1)
    calendar = pd.DatetimeIndex(np.sort(panel["date"].unique()))
    idx = inp["index"]

    out = PF.compute(panel, IS.close_on_calendar(idx, calendar)[IS.NIFTY500])
    grid = out["beta_252b"]
    out["spread_cs_63b"] = inp["spread"].reindex(index=grid.index, columns=grid.columns)

    sec_close = IS.close_on_calendar(idx, calendar)[list(IS.SECTORS)]
    val = {f: IS.fill_valuation(IS.wide(idx, f, calendar)[list(IS.SECTORS)], f)
           for f in ("pe", "pb", "dy")}
    sec, code = SE.compute(panel, sec_close, val)
    out.update(sec)
    out["sector_code"] = code

    def series(field: str, name: str) -> pd.Series:
        return idx[idx["index"] == name].set_index("date")[field].sort_index().dropna()

    out["market"] = MK.compute(series("close", IS.VIX), series("pe", IS.NIFTY50),
                               series("close", IS.NIFTY50), calendar)

    smap = panel[["date", "symbol", "ticker"]].assign(
        symbol=lambda d: d["symbol"].astype("string").str.strip().str.upper()
    ).drop_duplicates(subset=["date", "symbol"])
    psym = panel.drop(columns="symbol").merge(smap, on=["date", "ticker"], how="left")
    fo = inp["fo"]
    out.update(OP.compute(psym, OP.chain_features(fo), OP.option_value(fo)))
    out.update({k: v for k, v in FU.compute(psym, fo).items() if not k.startswith("_")})
    return out


def disk_path(name: str) -> str:
    if name == "market":
        return MK.OUT
    if name in FEATURE_SETS["p17x"] or name == "sector_code":
        return os.path.join(C.F1_DIR, f"{name}.parquet")
    return os.path.join(OP.F2_DIR, f"{name}.parquet")


def same(a: pd.DataFrame, b: pd.DataFrame, rtol: float = 0.0) -> bool:
    if not (a.index.equals(b.index) and list(a.columns) == list(b.columns)):
        return False
    x, y = a.to_numpy(float), b.to_numpy(float)
    if rtol == 0.0:
        return bool(np.array_equal(x, y, equal_nan=True))
    return bool(np.allclose(x, y, rtol=rtol, atol=1e-12, equal_nan=True))


def features(inp: dict, panels: dict, mask: pd.DataFrame) -> pd.DataFrame:
    bars = inp["panel"].set_index(["date", "ticker"])[["open", "high", "low", "close", "volume"]]
    fc_panels = {n: panels[n] for n in FEATURE_SETS[FEATURE_SET] if n not in F1_MARKET}
    fc_panels["market"] = panels["market"]
    return create_features(bars.sort_index(), FREQ, universe_mask=mask,
                           feature_set=FEATURE_SET, phase6_panels=fc_panels,
                           label_end_dates=True)


# ── Perturbations ───────────────────────────────────────────────────────────────

def perturb(inp: dict, cut: pd.Timestamp, rng: np.random.Generator) -> dict:
    """Every raw input strictly after ``cut`` replaced by noise; NaN pattern kept."""
    p = inp["panel"].copy()
    late = (p["date"] > cut).to_numpy()
    v = rng.uniform(0.5, 2.0, late.sum())
    # An independent factor per price column. One shared factor would leave every
    # within-day ratio (close/open, high/low) unchanged, and a leak through one of them
    # would then be invisible: the first run's canary went uncaught for exactly this.
    for c in ("open", "high", "low", "close"):
        p.loc[late, c] = p.loc[late, c].to_numpy() * rng.uniform(0.7, 1.3, late.sum())
    p.loc[late, "volume"] = p.loc[late, "volume"].to_numpy() * v
    p.loc[late, "turnover"] = p.loc[late, "turnover"].to_numpy() * v
    p.loc[late, "trades"] = np.maximum(1, np.round(p.loc[late, "trades"].to_numpy() * v))

    ix = inp["index"].copy()
    li = (ix["date"] > cut).to_numpy()
    for c in ("close", "pe", "pb", "dy"):
        ix.loc[li, c] = ix.loc[li, c].to_numpy() * rng.uniform(0.7, 1.3, li.sum())

    fo = inp["fo"].copy()
    lf = (fo["date"] > cut).to_numpy()
    for c in ("close", "value", "oi"):
        fo.loc[lf, c] = fo.loc[lf, c].to_numpy() * rng.uniform(0.7, 1.3, lf.sum())

    sp = inp["spread"].copy()
    ls = sp.index > cut
    sp.loc[ls] = sp.loc[ls].to_numpy() * rng.uniform(0.5, 2.0, sp.loc[ls].shape)
    return {"panel": p, "index": ix, "fo": fo, "spread": sp}


def plant_split(inp: dict, ticker: str, ex_date: pd.Timestamp, k: float = 0.5) -> dict:
    """A k:1 back-adjustment for a corporate action on ``ex_date``: history before it rescaled."""
    p = inp["panel"].copy()
    before = ((p["ticker"] == ticker) & (p["date"] < ex_date)).to_numpy()
    for c in ("open", "high", "low", "close"):
        p.loc[before, c] = p.loc[before, c].to_numpy() * k
    p.loc[before, "adj_factor"] = p.loc[before, "adj_factor"].to_numpy() * k
    p.loc[before, "volume"] = p.loc[before, "volume"].to_numpy() / k
    return {**inp, "panel": p}


def upto(frame: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    if isinstance(frame.index, pd.MultiIndex):
        return frame[frame.index.get_level_values(0) <= cut]
    return frame.loc[:cut]


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> int:
    t0 = time.time()
    rows: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, note: str = "") -> None:
        rows.append((name, ok, note))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {note}", flush=True)

    inp = load_inputs()
    mask = R.load_universe_mask(FREQ)
    base = build_panels(inp)
    bad = [n for n in base if not same(base[n], C.read(disk_path(n), "build"))]
    record("anchor: in-memory rebuild == every panel on disk", not bad,
           f"({len(base)} panels{'; differ: ' + ', '.join(bad) if bad else ''})")
    fb = features(inp, base, mask)
    feat_cols = present_feature_columns(fb)

    # Cuts: the Tier 1 cut (2nd day of an OOS block), then two later calendar points.
    days = np.unique(fb.index.get_level_values(0))
    spec = R.config.freq_spec(FREQ)
    block_start = spec.train_days + spec.predict_days * TIER1_BLOCK
    t1_cut = pd.Timestamp(days[block_start + 1])
    cuts = [t1_cut] + [pd.Timestamp(days[int(len(days) * f)]) for f in CUT_FRACTIONS]

    rng = np.random.default_rng(SEED)
    pert_feats: dict[pd.Timestamp, pd.DataFrame] = {}
    for cut in cuts:
        pin = perturb(inp, cut, rng)
        pp = build_panels(pin)
        moved = [n for n in base if not same(upto(base[n], cut), upto(pp[n], cut))]
        reached = any(not same(base[n].loc[cut:].iloc[1:], pp[n].loc[cut:].iloc[1:]) for n in base)
        record(f"panels causal at {cut.date()}", not moved and reached,
               f"moved: {moved}" if moved else ("" if reached else "PERTURBATION HAD NO EFFECT"))
        fp = features(pin, pp, mask)
        a, b = upto(fb, cut), upto(fp, cut)
        fmoved = [c for c in feat_cols if not same(a[[c]], b[[c]])]
        record(f"feature matrix causal at {cut.date()} ({len(feat_cols)} columns)",
               a.index.equals(b.index) and not fmoved, f"moved: {fmoved}" if fmoved else "")
        if cut == cuts[1]:
            leak_b = dict(base, **{CANARY: base[CANARY].shift(-1)})
            leak_p = dict(pp, **{CANARY: pp[CANARY].shift(-1)})
            ca, cb = upto(features(inp, leak_b, mask), cut), upto(features(pin, leak_p, mask), cut)
            caught = not same(ca[[CANARY]].reindex(cb.index), cb[[CANARY]])
            record(f"canary: {CANARY} shifted one day early is caught", caught)
        pert_feats[cut] = fp
        del pp
        print(f"  ... {time.time() - t0:,.0f}s", flush=True)

    # Tier 1: purge on must be causal; purge off must be caught (the §A.7 leak).
    end = pd.Timestamp(days[min(block_start + 2 * spec.predict_days, len(days) - 1)])
    fa_t, fp_t = upto(fb, end), upto(pert_feats[t1_cut], end)
    for purge in (True, False):
        eng = lambda: TreeAlphaEngine.from_frequency(  # noqa: E731
            FREQ, target_col=TIER1_TARGET, model_choice="lgbm", purge_label_overlap=purge)
        wa, wb = eng().run_walk_forward(fa_t), eng().run_walk_forward(fp_t)
        x, y = upto(wa.alpha_scores, t1_cut), upto(wb.alpha_scores, t1_cut)
        identical = same(x, y)
        diff = float(np.nanmax(np.abs(x.to_numpy() - y.to_numpy()))) if x.shape == y.shape else np.inf
        later = not same(wa.alpha_scores.loc[t1_cut:].iloc[1:], wb.alpha_scores.loc[t1_cut:].iloc[1:])
        if purge:
            record(f"Tier 1 (purge on): predictions <= {t1_cut.date()} identical",
                   identical and later, f"max |diff| {diff:.2e}; perturbation reached later blocks: {later}")
        else:
            record("Tier 1 canary (purge off): the label leak is caught",
                   not identical, f"max |diff| {diff:.2e}")
        print(f"  ... {time.time() - t0:,.0f}s", flush=True)

    # A corporate action after the cut must not move anything before it.
    ca_cut = cuts[1]
    ex = pd.Timestamp(days[np.searchsorted(days, np.datetime64(ca_cut)) + 10])
    split = plant_split(inp, SPLIT_TICKER, ex)
    ps = build_panels(split)
    moved = [n for n in base if not same(upto(base[n], ca_cut), upto(ps[n], ca_cut), rtol=1e-9)]
    record(f"future {SPLIT_TICKER} split ({ex.date()}) leaves new panels <= {ca_cut.date()} unchanged",
           not moved, f"moved: {moved}" if moved else "")
    fs = features(split, ps, mask)
    a, b = upto(fb, ca_cut), upto(fs, ca_cut)
    f0_moved = [c for c in feat_cols if c in FEATURE_SETS["p17"] or c not in FEATURE_SETS[FEATURE_SET]]
    f0_moved = [c for c in f0_moved if not same(a[[c]], b[[c]], rtol=1e-9)]
    new_moved = [c for c in FEATURE_SETS[FEATURE_SET] if not same(a[[c]], b[[c]], rtol=1e-9)]
    record("future split leaves the new feature columns unchanged", not new_moved,
           f"moved: {new_moved}" if new_moved else "")
    print(f"  INFO  of the 17 (F0, frozen, not gating), moved by a future split: {f0_moved or 'none'}")

    ok = all(r[1] for r in rows)
    print("\n" + "=" * 96)
    print(f"PHASE 6 CAUSALITY — {sum(r[1] for r in rows)}/{len(rows)} checks pass "
          f"({time.time() - t0:,.0f}s)")
    print(f"VERDICT: {'causal — no feature or Tier 1 prediction reads the future' if ok else 'FAIL'}")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
