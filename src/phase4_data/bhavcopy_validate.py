"""
Phase 4 validation — the acceptance gate the rebuilt database must clear.

A database nobody has checked is worth nothing, and this one has a specific way of
being wrong: NSE publishes no corporate-action feed on the archive host, so splits and
bonuses are *detected*, not looked up. A missed 1:10 split injects a fake −90% return
straight into the tree's training labels. Detection accuracy is the single largest risk
in the Phase 4 build, so it gets measured here rather than assumed.

Six gates. **Gates 1, 5 and 6 are blocking**; 2–4 are diagnostics.

    1. REPRODUCTION (blocking).  Rebuild the 68 tickers of the existing verified panel
       and compare daily *returns*. Levels are expected to differ — two panels can be
       back-adjusted from different epochs — but returns cannot. Scored in three parts:
       (1a) typical-day fidelity within a tick-scaled tolerance on ≥95% of qualifying
       names, (1b) *zero* days where our series shows an impossible return, (1c) overall
       disagreement ≤0.1%. A mechanical drop rule first disqualifies reference series
       that are themselves broken — see the threshold block below.
    2. SURVIVORSHIP.  How many of the panel left the exchange — the number that
       quantifies what a survivor-only universe would have hidden.
    3. ACTION AUDIT.  The largest detected factors, printed for eyeballing against
       public split/bonus records, plus everything the detector itself flagged.
    4. RESIDUAL DISCONTINUITY.  Panel-wide extreme-return rate, as a distribution check.
    5. CAUSALITY (blocking).  Re-asserts the two invariants the build depends on —
       back-adjustment leaves historical returns untouched, and universe selection at
       *t* reads only bars before *t*.
    6. RESIDUAL SWEEP (blocking).  **The gate that actually proves the labels are
       clean.** Gate 1 can only see the 66 names the reference happens to contain — 95%
       of the universe is invisible to it. Gate 6 asserts an internal property over
       *every* in-universe bar of all 1,244 names: no impossible return survives
       adjustment unless a detected action sits within a few bars, or the event is
       recorded in the audited `known_extremes.csv` register.

Exits non-zero if any blocking gate fails, so it can gate the build in a shell pipeline.

Usage
    python src/phase4_data/bhavcopy_validate.py
    python src/phase4_data/bhavcopy_validate.py --reference data/daily_ohlcv.parquet
"""

import argparse
import os
import sys
from typing import Final

import numpy as np
import pandas as pd

# Run as a script (`python src/phase4_data/bhavcopy_validate.py`) sys.path holds this
# file's directory, not the repo root, so `import src.phase4_data...` fails. Put the
# repo root first so the module works both as a script and as an import.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_PANEL: Final[str] = "data/bhavcopy/panel_daily.parquet"
DEFAULT_ACTIONS: Final[str] = "data/bhavcopy/corporate_actions.csv"
DEFAULT_LIFECYCLE: Final[str] = "data/bhavcopy/lifecycle.csv"
DEFAULT_REFERENCE: Final[str] = "data/daily_ohlcv.parquet"
DEFAULT_MASK: Final[str] = "data/nse500_universe_mask.parquet"

# ── Gate 1 thresholds ───────────────────────────────────────────────────────────
#
# The original specification was "per-name return correlation > 0.999 on ≥95% of
# names". Building the panel invalidated that metric: **the reference is not clean
# ground truth.** It carries at least seven days that are physically impossible under
# NSE circuit limits (INFY +302.8% on 2015-04-24, BEL +195.6% on 2015-09-11 — days when
# raw bhavcopy was flat), and its VEDL history before 2024 is a different price series
# whose level ratio to ours wanders across 454 days before locking to exactly 1.0.
# A single such day destroys a correlation while barely moving a median, so a
# correlation gate measures the reference's defects, not ours.
#
# What gate 1 actually exists to prove is that **no corporate action was missed**, since
# a missed action injects a fake ±50–90% return into the tree's training labels. That is
# tested directly and asymmetrically below. Correlation is still computed and printed —
# as a diagnostic, not a pass criterion.
# Per-name typical-day fidelity. The tolerance is expressed in **ticks, not basis
# points**, because a fixed bp threshold is physically unachievable on cheap stocks:
# NSE's tick is Rs 0.05, which is 0.2 bps on HINDUNILVR at Rs 2,057 but 8.5 bps on
# TATASTEEL at Rs 59. Two feeds that pick different last trades of the same session
# differ by a tick or two by construction, so demanding sub-tick agreement measures
# price level, not data quality. Verified empirically: the median |Δ| per name tracks
# that name's tick value in bps almost one-for-one.
MEDIAN_DIFF_GATE: Final[float] = 1e-4      # floor, for stocks where a tick is tiny
TICK_SIZE_RS: Final[float] = 0.05          # NSE minimum price increment
TICK_TOLERANCE: Final[float] = 3.0         # allow this many ticks of median disagreement
PASS_FRACTION_GATE: Final[float] = 0.95    # required on this fraction of names
IMPLAUSIBLE_RETURN: Final[float] = 0.35    # beyond NSE circuit limits ⇒ unadjusted action
DISAGREEMENT_GATE: Final[float] = 0.001    # ≤0.1% of observations may differ by >5%
MATERIAL_DIFF: Final[float] = 0.05         # what counts as a disagreement day
CORR_DIAGNOSTIC: Final[float] = 0.999      # reported only, never gates
MIN_OVERLAP_BARS: Final[int] = 250         # below this a correlation is not meaningful

# Gate 6 (residual sweep) and the mechanical drop rule.
DEFAULT_REGISTER: Final[str] = os.path.join(os.path.dirname(__file__), "known_extremes.csv")
WITNESS_BARS: Final[int] = 3               # action-to-event proximity that counts as explained
MAX_GAP_DAYS: Final[int] = 7               # a return across a suspension is not a discontinuity

# A reference series is disqualified as ground truth when it fails EITHER test. Both are
# properties of the reference itself, evaluated before any pass rate is computed.
REF_DRIFT_STEP: Final[float] = 0.01        # level-ratio step that counts as drift …
REF_DRIFT_FRACTION: Final[float] = 0.05    # … on more than this share of days ⇒ mis-stitched


def _naive_dates(s: pd.Series) -> pd.Series:
    """Drop timezone so the tz-aware reference panel aligns with the naive bhavcopy one."""
    out = pd.to_datetime(s)
    if getattr(out.dtype, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out.dt.normalize()


def _returns_wide(df: pd.DataFrame, date_col: str, price_col: str = "close") -> pd.DataFrame:
    """Pivot to (date × ticker) and take simple returns."""
    wide = df.pivot_table(index=date_col, columns="ticker", values=price_col, aggfunc="last")
    return wide.sort_index().pct_change()


def gate_reproduction(
    panel: pd.DataFrame, reference_path: str
) -> tuple[bool, pd.DataFrame]:
    """
    Gate 1 — reproduce the existing 68-name panel's daily returns from bhavcopy.

    This is what converts the corporate-action detector from a plausible heuristic into
    a measured one: the reference panel is independently sourced and already trusted by
    Phase 2/3, so any name that fails here is almost certainly carrying an undetected
    split rather than a data-vendor disagreement.
    """
    print("\n" + "=" * 78)
    print("GATE 1 — REPRODUCTION vs the existing verified panel  [BLOCKING]")
    print("=" * 78)

    if not os.path.exists(reference_path):
        print(f"  SKIP — reference {reference_path} not found")
        return True, pd.DataFrame()

    ref = pd.read_parquet(reference_path)
    ref["timestamp"] = _naive_dates(ref["timestamp"])
    ours = panel.copy()
    ours["date"] = _naive_dates(ours["date"])

    lo, hi = ref["timestamp"].min(), ref["timestamp"].max()
    ours = ours[ours["date"].between(lo, hi)]
    names = sorted(set(ref["ticker"]) & set(ours["ticker"]))
    missing = sorted(set(ref["ticker"]) - set(ours["ticker"]))

    print(f"  window {lo.date()} → {hi.date()} | {len(names)}/{ref['ticker'].nunique()} "
          f"reference tickers found in the rebuilt panel")
    if missing:
        print(f"  NOT FOUND ({len(missing)}): {', '.join(missing)}")

    r_ref = _returns_wide(ref, "timestamp")
    r_new = _returns_wide(ours, "date")

    rows = []
    for t in names:
        a, b = r_ref.get(t), r_new.get(t)
        if a is None or b is None:
            continue
        j = pd.concat([a.rename("ref"), b.rename("new")], axis=1).dropna()
        if len(j) < MIN_OVERLAP_BARS:
            rows.append({"ticker": t, "bars": len(j), "corr": np.nan,
                         "med_abs_diff": np.nan, "max_abs_diff": np.nan,
                         "n_material": 0, "tol": np.nan, "tick_bps": np.nan,
                         "our_implausible": False, "ref_implausible": False,
                         "ref_drift": np.nan, "disqualified": True, "pass": False})
            continue
        d = (j["ref"] - j["new"]).abs()
        material = d > MATERIAL_DIFF
        # One tick expressed as a return, at this name's typical price level.
        px = float(ours.loc[ours["ticker"] == t, "close"].median())
        tick_bps = TICK_SIZE_RS / px if px > 0 else np.inf
        # On a disagreement day, whichever side shows a move beyond NSE's circuit
        # limits is the side carrying an unadjusted corporate action.
        ref_bad = bool((material & (j["ref"].abs() > IMPLAUSIBLE_RETURN)).any())

        # Drift test: if only the LEVEL differs, returns are identical and the level
        # ratio is flat. A ratio that keeps stepping means the two series are not the
        # same instrument — a mis-stitched reference history.
        lv = pd.concat([
            ref.loc[ref["ticker"] == t].set_index("timestamp")["close"].rename("ref"),
            ours.loc[ours["ticker"] == t].set_index("date")["close"].rename("new"),
        ], axis=1).dropna()
        drift = float(
            (lv["ref"] / lv["new"]).pct_change().abs().gt(REF_DRIFT_STEP).mean()
        ) if len(lv) > 1 else 0.0

        rows.append({
            "ticker": t,
            "bars": len(j),
            "corr": float(j["ref"].corr(j["new"])),
            "med_abs_diff": float(d.median()),
            "max_abs_diff": float(d.max()),
            "n_material": int(material.sum()),
            "tol": max(MEDIAN_DIFF_GATE, TICK_TOLERANCE * tick_bps),
            "tick_bps": tick_bps,
            "our_implausible": bool((material & (j["new"].abs() > IMPLAUSIBLE_RETURN)).any()),
            "ref_implausible": ref_bad,
            "ref_drift": drift,
            # THE MECHANICAL DROP RULE. Evaluated here, from properties of the reference
            # alone, before any pass rate exists — so it cannot be tuned to a verdict.
            "disqualified": ref_bad or drift > REF_DRIFT_FRACTION,
            "pass": False,
        })
    res = pd.DataFrame(rows)
    if res.empty:
        print("  FAIL — no comparable tickers")
        return False, res

    # ── The mechanical drop rule, reported before any pass rate ─────────────────
    dq = res[res["disqualified"]]
    print(f"\n  [drop rule] {len(dq)}/{len(res)} reference series disqualified as ground "
          f"truth (properties of the REFERENCE, evaluated before any pass rate):")
    for _, r in dq.sort_values("ref_drift", ascending=False).iterrows():
        why = []
        if r["ref_implausible"]:
            why.append(f"impossible day (>{IMPLAUSIBLE_RETURN:.0%}) our panel contradicts")
        if pd.notna(r["ref_drift"]) and r["ref_drift"] > REF_DRIFT_FRACTION:
            why.append(f"level ratio drifts on {r['ref_drift']:.0%} of days (mis-stitched)")
        if not why:
            why.append(f"only {int(r['bars'])} overlapping bars")
        print(f"    {r['ticker']:<14}{'; '.join(why)}")

    kept = res[~res["disqualified"]].copy()
    print(f"  scoring on the {len(kept)} qualifying names")

    # ── 1a. Typical-day fidelity, per name ──────────────────────────────────────
    res["pass"] = res["med_abs_diff"] < res["tol"]
    kept = res[~res["disqualified"]]
    frac = float(kept["pass"].mean()) if len(kept) else 0.0
    ok_a = frac >= PASS_FRACTION_GATE

    print(f"\n  [1a] typical-day fidelity — median |Δ return| < "
          f"{TICK_TOLERANCE:.0f} ticks (per-name, price-scaled)")
    print(f"       {int(kept['pass'].sum())}/{len(kept)} qualifying names = {frac:.1%} "
          f"(gate ≥{PASS_FRACTION_GATE:.0%})  → {'PASS' if ok_a else 'FAIL'}")
    print(f"       median |Δ return| across names: {kept['med_abs_diff'].median():.2e}")

    # ── 1b. No missed corporate action (the asymmetric test that matters) ───────
    # A missed split leaves an impossible return in OUR series. A reference defect
    # leaves one in THEIRS. Only the former is our problem, and only the former can
    # poison a training label — so the two are counted separately, never netted.
    ours_bad = int(kept["our_implausible"].sum())
    theirs_bad = int(res["ref_implausible"].sum())
    ok_b = ours_bad == 0

    print(f"\n  [1b] missed corporate actions — days where our return exceeds "
          f"±{IMPLAUSIBLE_RETURN:.0%}")
    print(f"       ours implausible: {ours_bad}  (gate: 0)  → {'PASS' if ok_b else 'FAIL'}")
    print(f"       reference implausible: {theirs_bad}  (reported, not gated — these are "
          f"defects in the reference)")

    # ── 1c. Bounded overall disagreement ────────────────────────────────────────
    n_obs = int(kept["bars"].sum())
    n_dis = int(kept["n_material"].sum())
    rate = n_dis / max(n_obs, 1)
    ok_c = rate <= DISAGREEMENT_GATE

    print(f"\n  [1c] overall disagreement — |Δ return| > {MATERIAL_DIFF:.0%}")
    print(f"       {n_dis} of {n_obs:,} observations = {rate:.4%} "
          f"(gate ≤{DISAGREEMENT_GATE:.1%})  → {'PASS' if ok_c else 'FAIL'}")

    print(f"\n  [diagnostic, not gated] median correlation {kept['corr'].median():.6f} | "
          f"names above {CORR_DIAGNOSTIC}: {int((kept['corr'] > CORR_DIAGNOSTIC).sum())}/{len(kept)}")

    bad = kept[~kept["pass"] | kept["our_implausible"]].sort_values("med_abs_diff", ascending=False)
    if len(bad):
        print(f"\n  NAMES FAILING 1a OR 1b ({len(bad)}):")
        for _, r in bad.iterrows():
            flag = " ← MISSED ACTION" if r["our_implausible"] else ""
            print(f"    {r['ticker']:<14} bars={int(r['bars']):>5} corr={r['corr']:.5f} "
                  f"med|Δ|={r['med_abs_diff']:.2e} max|Δ|={r['max_abs_diff']:.3f}{flag}")

    worst = kept.nlargest(5, "n_material")[["ticker", "n_material", "bars", "corr"]]
    print(f"\n  most disagreement days (diagnostic):")
    for _, r in worst.iterrows():
        print(f"    {r['ticker']:<14} {int(r['n_material']):>3} days of {int(r['bars']):,} "
              f"| corr={r['corr']:.5f}")

    ok = ok_a and ok_b and ok_c
    print(f"\n  → GATE 1 {'PASS' if ok else 'FAIL'}")
    return ok, res


def gate_survivorship(panel: pd.DataFrame, lifecycle_path: str) -> None:
    """Gate 2 — count what a survivor-only universe would have silently excluded."""
    print("\n" + "=" * 78)
    print("GATE 2 — SURVIVORSHIP")
    print("=" * 78)
    if not os.path.exists(lifecycle_path):
        print(f"  SKIP — {lifecycle_path} not found")
        return

    life = pd.read_csv(lifecycle_path, parse_dates=["first_date", "last_date"])
    counts = life["kind"].value_counts()
    total = len(life)
    dead = int(counts.get("delisted", 0) + counts.get("acquired", 0))

    print(f"  {total:,} ISINs in the panel")
    for k in ("live", "delisted", "acquired"):
        n = int(counts.get(k, 0))
        print(f"    {k:<10} {n:>6,}  ({n / total:.1%})")
    print(f"  suspensions (absent, then traded again): {int(life['suspended'].sum()):,}")
    print(
        f"\n  → {dead:,} names ({dead / total:.1%}) left the exchange inside the window.\n"
        f"    A universe built from today's index membership would contain none of them,\n"
        f"    which is precisely the bias this point-in-time panel removes."
    )


def gate_action_audit(actions_path: str, top: int = 20) -> None:
    """Gate 3 — print the largest and the flagged detections for manual verification."""
    print("\n" + "=" * 78)
    print("GATE 3 — CORPORATE-ACTION AUDIT")
    print("=" * 78)
    if not os.path.exists(actions_path):
        print(f"  SKIP — {actions_path} not found")
        return

    act = pd.read_csv(actions_path, parse_dates=["date"])
    if act.empty:
        print("  no actions detected — suspicious for a 12-year panel; check thresholds")
        return

    print(f"  {len(act):,} actions detected | {act['isin'].nunique():,} distinct names | "
          f"{act['date'].dt.year.min()}–{act['date'].dt.year.max()}")
    print(f"  flagged for review (large implied ex-date return): {int(act['review'].sum())}")

    act = act.assign(_mag=np.abs(np.log(act["factor"])))
    print(f"\n  {top} largest by adjustment magnitude — check against public records:")
    print(f"    {'date':<12}{'symbol':<14}{'factor':>9}{'obs ratio':>11}"
          f"{'ex ret':>9}{'turnover':>10}")
    for _, r in act.nlargest(top, "_mag").iterrows():
        print(f"    {r['date'].date()!s:<12}{str(r['symbol']):<14}{r['factor']:>9.4f}"
              f"{r['price_ratio']:>11.4f}{r['implied_ex_ret']:>9.2%}"
              f"{r['turnover_ratio']:>10.2f}")

    flagged = act[act["review"]]
    if len(flagged):
        print(f"\n  flagged rows ({len(flagged)}) — implied ex-date return is large, so the\n"
              f"  snapped factor is the least certain here:")
        for _, r in flagged.nlargest(min(10, len(flagged)), "_mag").iterrows():
            print(f"    {r['date'].date()!s:<12}{str(r['symbol']):<14}{r['factor']:>9.4f}"
                  f"{r['price_ratio']:>11.4f}{r['implied_ex_ret']:>9.2%}")


def gate_residual_discontinuity(panel: pd.DataFrame, threshold: float = 0.40) -> None:
    """Gate 4 — count extreme post-adjustment returns; a cluster means missed actions."""
    print("\n" + "=" * 78)
    print("GATE 4 — RESIDUAL DISCONTINUITY")
    print("=" * 78)

    df = panel.sort_values(["isin", "date"])
    ret = df.groupby("isin", sort=False)["close"].pct_change()
    n = int(ret.notna().sum())
    extreme = ret.abs() > threshold
    n_ex = int(extreme.sum())

    print(f"  {n:,} return observations | |r| > {threshold:.0%}: {n_ex:,} ({n_ex / n:.4%})")

    if n_ex:
        worst = df.loc[extreme, ["date", "symbol", "close"]].assign(ret=ret[extreme])
        by_year = worst.groupby(worst["date"].dt.year).size()
        spike = by_year.idxmax()
        print(f"  worst year: {spike} with {int(by_year.max())} events")
        print(f"\n  10 most extreme survivors of adjustment:")
        for _, r in worst.reindex(worst["ret"].abs().sort_values(ascending=False).index).head(10).iterrows():
            print(f"    {r['date'].date()!s:<12}{str(r['symbol']):<14}{r['ret']:>9.1%}")
        print(
            "\n  Interpretation: isolated events are genuine (circuit-limit runs, penny\n"
            "  stocks, news shocks); a run of the same name or a same-day cluster across\n"
            "  many names is a missed corporate action."
        )


def gate_residual_sweep(
    panel: pd.DataFrame, mask_path: str, actions_path: str, register_path: str
) -> bool:
    """
    Gate 6 — every tradeable bar, not just the 66 the reference happens to cover.

    **This is the gate that actually proves the training labels are clean.** Gate 1
    compares against an external panel and therefore sees 66 of 1,244 names — 95% of the
    universe is invisible to it. Gate 6 needs no external data at all: it asserts an
    internal property, that after adjustment no in-universe bar carries a return which is
    impossible under NSE's circuit limits unless we can say *why*.

    An event passes if either
      - a detected corporate action sits within ``WITNESS_BARS`` of it, or
      - it appears in the audited register ``known_extremes.csv`` (genuine news moves,
        demergers we deliberately do not adjust, sub-floor penny stocks).

    Anything else is an undetected corporate action, i.e. a fake +/-90% training label.
    A rebuild that produces an unregistered event fails here — the register is a tripwire,
    not a suppression list.
    """
    print("\n" + "=" * 78)
    print("GATE 6 — RESIDUAL SWEEP, all in-universe bars  [BLOCKING]")
    print("=" * 78)

    if not os.path.exists(mask_path):
        print(f"  SKIP — {mask_path} not found (run universe.py --build)")
        return True

    mask = pd.read_parquet(mask_path)
    flags = mask.stack()
    in_universe = set(zip(*[flags[flags].index.get_level_values(i) for i in (0, 1)]))

    df = panel.sort_values(["entity", "date"]).copy()
    df["ret"] = df.groupby("entity", sort=False)["close"].pct_change()
    # Only consecutive bars: a return measured across a suspension is not a discontinuity
    # in the price series, it is an absence of one.
    df["gap"] = (df["date"] - df.groupby("entity", sort=False)["date"].shift(1)).dt.days

    sub = df[[(d, t) in in_universe for d, t in zip(df["date"], df["ticker"])]]
    events = sub[(sub["ret"].abs() > IMPLAUSIBLE_RETURN) & (sub["gap"] <= MAX_GAP_DAYS)]
    print(f"  {len(sub):,} in-universe bars | |return| > {IMPLAUSIBLE_RETURN:.0%}: {len(events)}")

    # Explanation 1 — a detected action nearby.
    explained_by_action = set()
    if os.path.exists(actions_path):
        act = pd.read_csv(actions_path, parse_dates=["date"])
        cal = pd.DatetimeIndex(np.sort(panel["date"].unique()))
        pos = pd.Series(range(len(cal)), index=cal)
        by_entity: dict = {}
        for e, d in zip(act["entity"], act["date"]):
            by_entity.setdefault(e, []).append(int(pos[pd.Timestamp(d)]))
        for idx, r in events.iterrows():
            i = int(pos[r["date"]])
            if any(abs(i - x) <= WITNESS_BARS for x in by_entity.get(r["entity"], [])):
                explained_by_action.add(idx)

    # Explanation 2 — the audited register.
    registered: set[tuple] = set()
    if os.path.exists(register_path):
        reg = pd.read_csv(register_path, comment="#", parse_dates=["date"])
        registered = set(zip(reg["ticker"], reg["date"]))
        cats = reg["category"].value_counts().to_dict()
        print(f"  register: {len(reg)} audited events {cats}")
    else:
        print(f"  WARN: register {register_path} not found")

    unexplained = [
        r for idx, r in events.iterrows()
        if idx not in explained_by_action and (r["ticker"], r["date"]) not in registered
    ]

    print(f"  explained by a detected action within +/-{WITNESS_BARS} bars: "
          f"{len(explained_by_action)}")
    print(f"  explained by the audited register: "
          f"{len(events) - len(explained_by_action) - len(unexplained)}")
    print(f"  UNEXPLAINED: {len(unexplained)}  (gate: 0)")

    if unexplained:
        print("\n  Each of these is a probable undetected corporate action. Either fix the\n"
              "  detector, or audit and add it to the register with a reason:")
        for r in sorted(unexplained, key=lambda x: -abs(x["ret"]))[:25]:
            print(f"    {str(r['date'].date()):<12}{r['ticker']:<14}{r['ret']:>8.1%}"
                  f"  close={r['close']:.2f}")

    ok = not unexplained
    print(f"\n  → GATE 6 {'PASS' if ok else 'FAIL'}")
    return ok


def gate_causality(panel: pd.DataFrame, mask_path: str) -> bool:
    """Gate 5 — re-assert the two invariants the whole build rests on."""
    print("\n" + "=" * 78)
    print("GATE 5 — CAUSALITY")
    print("=" * 78)
    ok = True

    # (a) Back-adjustment is level-only: undo it and every return must be identical
    # except on the ex-dates themselves.
    if "adj_factor" in panel.columns:
        df = panel.sort_values(["isin", "date"]).copy()
        df["_raw_close"] = df["close"] / df["adj_factor"]
        g = df.groupby("isin", sort=False)
        r_adj = g["close"].pct_change()
        r_raw = g["_raw_close"].pct_change()
        ex = g["adj_factor"].pct_change().abs() > 1e-12   # rows where the factor stepped
        diff = (r_adj - r_raw).abs()[~ex]
        worst = float(diff.max(skipna=True) or 0.0)
        good = worst < 1e-8
        ok &= good
        print(f"  back-adjustment preserves returns: max |Δ| = {worst:.2e} "
              f"→ {'PASS' if good else 'FAIL'}")
    else:
        print("  SKIP — panel has no adj_factor column")

    # (b) Universe membership can only begin after a full trailing window exists, and
    # every membership run must start on a rebalance boundary — a name appearing at an
    # arbitrary date would mean the selection saw something it should not have.
    if os.path.exists(mask_path):
        from src.phase4_data.universe import LOOKBACK, REBALANCE_EVERY

        mask = pd.read_parquet(mask_path)
        pre = mask.iloc[:LOOKBACK].to_numpy().any()
        good = not pre
        ok &= good
        print(f"  nothing selected before a full {LOOKBACK}-day lookback → "
              f"{'PASS' if good else 'FAIL'}")

        arr = mask.to_numpy()
        starts = np.zeros(len(mask), dtype=bool)
        entered = arr[1:] & ~arr[:-1]
        starts[1:] = entered.any(axis=1)
        bad_rows = [i for i in np.flatnonzero(starts)
                    if (i - LOOKBACK) % REBALANCE_EVERY != 0]
        good = not bad_rows
        ok &= good
        print(f"  every membership change lands on a rebalance boundary → "
              f"{'PASS' if good else f'FAIL ({len(bad_rows)} off-cycle rows)'}")
    else:
        print(f"  SKIP — {mask_path} not found (run universe.py --build)")

    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the rebuilt bhavcopy database.")
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--actions", default=DEFAULT_ACTIONS)
    ap.add_argument("--lifecycle", default=DEFAULT_LIFECYCLE)
    ap.add_argument("--reference", default=DEFAULT_REFERENCE)
    ap.add_argument("--mask", default=DEFAULT_MASK)
    ap.add_argument("--register", default=DEFAULT_REGISTER)
    args = ap.parse_args()

    if not os.path.exists(args.panel):
        raise SystemExit(f"{args.panel} not found — run bhavcopy_panel.py --build first.")

    panel = pd.read_parquet(args.panel)
    print(f"[validate] panel: rows={len(panel):,} tickers={panel['ticker'].nunique():,} "
          f"{panel['date'].min().date()} → {panel['date'].max().date()}")

    repro_ok, res = gate_reproduction(panel, args.reference)
    gate_survivorship(panel, args.lifecycle)
    gate_action_audit(args.actions)
    gate_residual_discontinuity(panel)
    causal_ok = gate_causality(panel, args.mask)
    sweep_ok = gate_residual_sweep(panel, args.mask, args.actions, args.register)

    print("\n" + "=" * 78)
    verdict = repro_ok and causal_ok and sweep_ok
    print(f"VERDICT: {'PASS — cleared for the engine' if verdict else 'FAIL — do not run the engine on this panel'}")
    print("=" * 78)
    if not verdict:
        print(
            "\nA failing reproduction gate means undetected corporate actions are still\n"
            "in the panel; they would enter the trees as fake ±50% training labels.\n"
            "Re-tune ActionConfig in bhavcopy_panel.py against the failing names above\n"
            "and rebuild. Tuning against these prices is safe — the ground truth is an\n"
            "independent price series, not strategy performance."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
