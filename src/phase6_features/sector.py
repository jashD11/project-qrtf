"""
Phase 6 Part B, F1 — sector features (docs/phase6_plan.md §B.2).

**Assignment.** Each stock is assigned the one of the 11 sectoral indexes
(``index_series.SECTORS``) whose daily log returns it correlates with most over the
trailing 252 own bars (at least 126 valid pairs), each index return taken over the stock's
own bar span. Assignment is made on the universe rebalance dates
(``universe.rebalance_dates``: every 63 sessions from session 252) using data up to that
date's close, and held until the next one. A stock with too little history on a
rebalance date is unassigned until a later rebalance. No constituent list is used, so
there is no survivorship bias.

    sector_mom_63b   the assigned index's 63-session log return.
    sector_rel_21b   the stock's 21-own-bar log return minus the assigned index's log
                     return over the same span.
    sector_pe/pb/dy  the assigned index's P/E, P/B and dividend yield, with publication
                     gaps carried and a long-blank P/E as +inf (``index_series.fill_valuation``).

Values exist only on cells where the stock traded, as for every other feature.

    python src/phase6_features/sector.py            # dry-run self-test
    python src/phase6_features/sector.py --build
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Final

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.phase4_data.universe import rebalance_dates  # noqa: E402
from src.phase6_features import _common as C  # noqa: E402
from src.phase6_features import index_series as IS  # noqa: E402

CORR_WINDOW, CORR_MIN = 252, 126
MOM_WINDOW: Final[int] = 63
REL_WINDOW: Final[int] = 21
FEATURES: Final[tuple[str, ...]] = (
    "sector_mom_63b", "sector_rel_21b", "sector_pe", "sector_pb", "sector_dy",
)


def assign(panel: pd.DataFrame, sector_close: pd.DataFrame) -> pd.DataFrame:
    """
    (calendar × ticker) float code into ``sector_close.columns`` (NaN = unassigned).

    The code on a date is the one chosen on the latest rebalance date at or before it.
    """
    calendar = sector_close.index
    t = panel["ticker"]
    y = np.log(panel["close"] / panel["prev_close"])
    corrs = []
    for name in sector_close.columns:
        x = pd.Series(
            C.span_logret(sector_close[name], panel["prev_date"], panel["date"]), index=panel.index
        )
        ok = (y.notna() & x.notna()).astype(float)
        xv, yv = x.where(ok > 0, 0.0), y.where(ok > 0, 0.0)
        S = lambda v: C.own_rolling_sum(v, t, CORR_WINDOW)  # noqa: E731
        n, sx, sy = S(ok), S(xv), S(yv)
        with np.errstate(divide="ignore", invalid="ignore"):
            cxy = S(xv * yv) - sx * sy / n
            cxx = S(xv * xv) - sx * sx / n
            cyy = S(yv * yv) - sy * sy / n
            r = (cxy / np.sqrt(cxx * cyy)).where(n >= CORR_MIN)
        wide = pd.DataFrame({"date": panel["date"], "ticker": t, "r": r}).pivot(
            index="date", columns="ticker", values="r"
        )
        # A stock that did not trade on the rebalance date uses its latest own bar before it.
        corrs.append(wide.reindex(calendar).ffill())

    stack = np.stack([c.to_numpy() for c in corrs])               # (sector, date, ticker)
    any_ok = np.isfinite(stack).any(axis=0)
    best = np.where(any_ok, np.nanargmax(np.where(np.isfinite(stack), stack, -np.inf), axis=0), -1)

    picks = rebalance_dates(calendar.to_numpy())
    on = np.zeros(len(calendar), dtype=bool)
    on[picks] = True
    code = pd.DataFrame(np.where(on[:, None], best, np.nan), index=calendar, columns=corrs[0].columns)
    # Hold each rebalance choice until the next; an unassigned stock (-1) stays unassigned.
    code = code.ffill().replace(-1, np.nan)
    return code


def gather(values: pd.DataFrame, code: pd.DataFrame) -> pd.DataFrame:
    """out[d, i] = values[d, code[d, i]] (NaN where unassigned)."""
    v = values.reindex(code.index).to_numpy(float)
    c = code.to_numpy()
    out = np.full(c.shape, np.nan)
    ok = np.isfinite(c)
    rows = np.nonzero(ok)[0]
    out[ok] = v[rows, c[ok].astype(int)]
    return pd.DataFrame(out, index=code.index, columns=code.columns)


def compute(panel: pd.DataFrame, sector_close: pd.DataFrame,
            valuation: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    calendar = sector_close.index
    code = assign(panel, sector_close)
    traded = C.to_wide(panel.assign(one=1.0), "one").reindex(index=calendar, columns=code.columns).notna()

    out = {"sector_mom_63b": gather(np.log(sector_close / sector_close.shift(MOM_WINDOW)), code)}
    for field in ("pe", "pb", "dy"):
        out[f"sector_{field}"] = gather(valuation[field], code)

    # Relative strength over the stock's own 21-bar span, against the sector it is
    # assigned to on t.
    t = panel["ticker"]
    g = panel.groupby("ticker", sort=False)
    start_date, start_close = g["date"].shift(REL_WINDOW), g["close"].shift(REL_WINDOW)
    pos = calendar.get_indexer(pd.DatetimeIndex(panel["date"]))
    cols = code.columns.get_indexer(t)
    c = code.to_numpy()[pos, cols]
    ok = np.isfinite(c) & start_date.notna().to_numpy()
    sec = np.full(len(panel), np.nan)
    sv = sector_close.to_numpy(float)
    p0 = calendar.get_indexer(pd.DatetimeIndex(start_date[ok]))
    ci = c[ok].astype(int)
    sec[ok] = np.log(sv[pos[ok], ci] / sv[p0, ci])
    rel = np.log(panel["close"] / start_close).to_numpy() - sec
    out["sector_rel_21b"] = pd.DataFrame(
        {"date": panel["date"], "ticker": t, "v": rel}
    ).pivot(index="date", columns="ticker", values="v").reindex(index=calendar, columns=code.columns)

    out = {k: v.where(traded) for k, v in out.items()}
    return {f: out[f] for f in FEATURES}, code


def build(allow_holdout: bool = False) -> dict[str, pd.DataFrame]:
    panel = C.load_panel(allow_holdout=allow_holdout)
    calendar = pd.DatetimeIndex(np.sort(panel["date"].unique()))
    long = IS.load()
    C.guard_holdout(long["date"], C.INDEX_DAILY, allow_holdout)
    close = IS.close_on_calendar(long, calendar)[list(IS.SECTORS)]
    valuation = {
        f: IS.fill_valuation(IS.wide(long, f, calendar)[list(IS.SECTORS)], f)
        for f in ("pe", "pb", "dy")
    }
    out, code = compute(panel, close, valuation)

    n_assigned = code.notna().sum(axis=1)
    share = code.apply(lambda r: r.value_counts(normalize=True), axis=1).iloc[-1]
    names = dict(enumerate(IS.SECTORS))
    print(f"[sector] assigned names per date: max {int(n_assigned.max())} | "
          f"latest mix: " + ", ".join(f"{names[int(k)].replace('NIFTY ', '')} {v:.0%}"
                                      for k, v in share.sort_values(ascending=False).items()))
    for name, frame in out.items():
        C.write(frame, os.path.join(C.F1_DIR, f"{name}.parquet"))
    C.write(code, os.path.join(C.F1_DIR, "sector_code.parquet"))
    return out


# ── Dry run ─────────────────────────────────────────────────────────────────────

def _dry_run() -> None:
    checks: dict[str, bool] = {}
    rng = np.random.default_rng(5)
    n_days, n_sec = 700, 3
    cal = pd.bdate_range("2017-01-02", periods=n_days)
    f = rng.normal(0, 0.012, (n_days, n_sec))
    sector_close = pd.DataFrame(1000 * np.exp(np.cumsum(f, axis=0)), index=cal,
                                columns=[f"SEC{j}" for j in range(n_sec)])
    switch_at = 450                       # S5 moves from sector 0 to sector 2 here
    rows = []
    for i in range(6):
        home = np.full(n_days, i % n_sec)
        if i == 5:
            home = np.where(np.arange(n_days) < switch_at, 0, 2)
        r = f[np.arange(n_days), home] + rng.normal(0, 0.006, n_days)
        close = 100 * np.exp(np.cumsum(r))
        rows.append(pd.DataFrame({"date": cal, "ticker": f"S{i}", "open": close, "close": close,
                                  "turnover": 1e7, "trades": 100}))
    panel = pd.concat(rows, ignore_index=True)
    panel = panel[~((panel["ticker"] == "S1") & (panel["date"].isin(cal[300:305])))]   # a halt
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = panel.groupby("ticker", sort=False)
    panel["prev_date"], panel["prev_close"] = g["date"].shift(1), g["close"].shift(1)
    val = {k: pd.DataFrame(v, index=cal, columns=sector_close.columns)
           for k, v in (("pe", [[20.0, 15.0, np.inf]] * n_days), ("pb", [[3.0, 2.0, 1.0]] * n_days),
                        ("dy", [[1.0, 1.5, 2.0]] * n_days))}

    out, code = compute(panel, sector_close, val)
    picks = rebalance_dates(cal.to_numpy())
    first = picks[0]
    checks["unassigned before the first rebalance"] = bool(code.iloc[:first].isna().all().all())
    planted = pd.Series([0, 1, 2, 0, 1, 0], index=[f"S{i}" for i in range(6)])
    checks["planted sectors recovered"] = bool((code.iloc[first] == planted).all())
    changes = code.ne(code.shift()).iloc[1:] & code.iloc[1:].notna() & code.shift().iloc[1:].notna()
    change_rows = set(np.nonzero(changes.any(axis=1).to_numpy())[0] + 1)
    checks["assignment changes only on rebalance dates"] = change_rows <= set(picks)
    checks["switcher reassigned at a later rebalance"] = bool(code["S5"].iloc[-1] == 2)
    checks["P/E +inf passes through to S2's sector"] = bool(
        np.isinf(out["sector_pe"]["S2"].iloc[first:]).all()
    )
    s1 = panel[panel["ticker"] == "S1"].reset_index(drop=True)
    k = int(np.flatnonzero(s1["date"] == cal[320])[0])          # window spans the halt
    a, b = s1["date"][k - REL_WINDOW], s1["date"][k]
    want = np.log(s1["close"][k] / s1["close"][k - REL_WINDOW]) - np.log(
        sector_close.loc[b, "SEC1"] / sector_close.loc[a, "SEC1"])
    checks["sector_rel_21b exact across a halt"] = bool(np.isclose(out["sector_rel_21b"].loc[b, "S1"], want))
    checks["NaN on untraded cells"] = bool(out["sector_mom_63b"].loc[cal[300:305], "S1"].isna().all())

    cut = cal[500]
    sc2 = sector_close.copy()
    sc2[sc2.index > cut] *= rng.uniform(0.7, 1.3, sc2[sc2.index > cut].shape)
    p2 = panel.copy()
    late = p2["date"] > cut
    p2.loc[late, "close"] *= rng.uniform(0.7, 1.3, int(late.sum()))
    p2["prev_close"] = p2.groupby("ticker", sort=False)["close"].shift(1)
    out2, code2 = compute(p2, sc2, val)
    checks["causal: assignment <= cut unchanged"] = bool(code.loc[:cut].equals(code2.loc[:cut]))
    checks["causal: features <= cut unchanged"] = all(out[f].loc[:cut].equals(out2[f].loc[:cut]) for f in out)

    print()
    for name, v in checks.items():
        print(f"  {name:<44}: {'PASS' if v else 'FAIL'}")
    assert all(checks.values()), "sector dry-run FAILED"
    print("\n[sector] dry-run OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Phase 6 F1 sector features.")
    ap.add_argument("--build", action="store_true", help="build (else dry-run)")
    args = ap.parse_args()
    if not args.build:
        _dry_run()
        return
    build()


if __name__ == "__main__":
    main()
