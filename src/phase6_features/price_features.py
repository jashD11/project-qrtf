"""
Phase 6 Part B, F1 — per-stock price features (docs/phase6_plan.md §B.2).

    beta_252b          OLS slope of the stock's daily log return on the Nifty 500's,
                       trailing 252 own bars ending at t, at least 126 valid pairs.
    ivol_63b           stdev of the residuals of that market-model regression fitted on
                       the trailing 63 own bars (at least 42 pairs), with n − 2 degrees
                       of freedom. Its own 63-bar fit, not the 252-bar beta (§B.8).
    ret_overnight_21b  sum over the trailing 21 own bars of log(open_t / close_prev).
    ret_intraday_21b   sum over the trailing 21 own bars of log(close_t / open_t).
    trade_size_21b     median over the trailing 21 own bars of turnover / trades.
    spread_cs_63b      the existing Corwin-Schultz half-spread panel (window ends t − 1).

Windows without a stated minimum need a complete window, as ``feature_creator._roll``
does. Prices are the panel's split/bonus-adjusted ones, so log(open_t / close_prev) is a
true overnight return even across an ex-date. Each index return is taken over the
stock's own bar span (see ``_common``).

    python src/phase6_features/price_features.py            # dry-run self-test
    python src/phase6_features/price_features.py --build
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

from src.phase6_features import _common as C  # noqa: E402
from src.phase6_features import index_series as IS  # noqa: E402

BETA_WINDOW, BETA_MIN = 252, 126
IVOL_WINDOW, IVOL_MIN = 63, 42
SUM_WINDOW: Final[int] = 21
SPREAD_CS: Final[str] = "data/bhavcopy/spread_daily_cs.parquet"
FEATURES: Final[tuple[str, ...]] = (
    "beta_252b", "ivol_63b", "ret_overnight_21b", "ret_intraday_21b",
    "trade_size_21b", "spread_cs_63b",
)


def market_model(
    y: pd.Series, x: pd.Series, tickers: pd.Series, window: int, min_n: int
) -> tuple[pd.Series, pd.Series]:
    """
    Rolling own-bar OLS of y on x: (slope, residual stdev with n − 2 dof).

    Pairs with either side missing are excluded, and the window counts own bars, so a
    pair can be missing without shrinking the window. Closed form from rolling sums.
    """
    ok = (y.notna() & x.notna()).astype(float)
    xv, yv = x.where(ok > 0, 0.0), y.where(ok > 0, 0.0)
    S = lambda v: C.own_rolling_sum(v, tickers, window)  # noqa: E731
    n, sx, sy = S(ok), S(xv), S(yv)
    sxx, sxy, syy = S(xv * xv), S(xv * yv), S(yv * yv)
    with np.errstate(divide="ignore", invalid="ignore"):
        cxx = sxx - sx * sx / n
        cxy = sxy - sx * sy / n
        cyy = syy - sy * sy / n
        slope = cxy / cxx
        ssr = (cyy - cxy * cxy / cxx).clip(lower=0.0)
        resid_sd = np.sqrt(ssr / (n - 2))
    enough = n >= min_n
    return slope.where(enough), resid_sd.where(enough)


def compute(panel: pd.DataFrame, mkt_close: pd.Series) -> dict[str, pd.DataFrame]:
    """All own-bar price features as wide (date × ticker) frames, raw."""
    t = panel["ticker"]
    y = np.log(panel["close"] / panel["prev_close"])
    x = pd.Series(C.span_logret(mkt_close, panel["prev_date"], panel["date"]), index=panel.index)

    beta, _ = market_model(y, x, t, BETA_WINDOW, BETA_MIN)
    _, ivol = market_model(y, x, t, IVOL_WINDOW, IVOL_MIN)
    overnight = np.log(panel["open"] / panel["prev_close"])
    intraday = np.log(panel["close"] / panel["open"])
    size = (panel["turnover"] / panel["trades"]).where(panel["trades"] > 0)

    long = pd.DataFrame({
        "date": panel["date"], "ticker": t,
        "beta_252b": beta,
        "ivol_63b": ivol,
        "ret_overnight_21b": C.own_rolling(overnight, t, SUM_WINDOW, "sum"),
        "ret_intraday_21b": C.own_rolling(intraday, t, SUM_WINDOW, "sum"),
        "trade_size_21b": C.own_rolling(size, t, SUM_WINDOW, "median"),
    })
    long = long.replace([np.inf, -np.inf], np.nan)
    return {f: C.to_wide(long, f) for f in FEATURES if f in long}


def build(allow_holdout: bool = False) -> dict[str, pd.DataFrame]:
    panel = C.load_panel(allow_holdout=allow_holdout)
    calendar = pd.DatetimeIndex(np.sort(panel["date"].unique()))
    mkt = IS.close_on_calendar(IS.load(), calendar)[IS.NIFTY500]
    out = compute(panel, mkt)

    grid = out["beta_252b"]
    spread = C.read(SPREAD_CS, "python src/phase4_data/spread.py --build")
    C.guard_holdout(spread.index, SPREAD_CS, allow_holdout)
    out["spread_cs_63b"] = spread.reindex(index=grid.index, columns=grid.columns)

    for name, frame in out.items():
        C.write(frame, os.path.join(C.F1_DIR, f"{name}.parquet"))
    return out


# ── Dry run ─────────────────────────────────────────────────────────────────────

def _synthetic(n_days: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    m = rng.normal(0, 0.01, n_days)
    mkt = pd.Series(1000 * np.exp(np.cumsum(m)), index=dates)
    rows = []
    for i, (b, s) in enumerate(((0.5, 0.004), (1.0, 0.004), (1.5, 0.004))):
        r = b * m + rng.normal(0, s, n_days)
        close = 100 * np.exp(np.cumsum(r))
        open_ = close * np.exp(-rng.normal(0, 0.004, n_days))     # intraday move
        rows.append(pd.DataFrame({
            "date": dates, "ticker": f"S{i}", "open": open_, "close": close,
            "turnover": rng.lognormal(16, 0.3, n_days), "trades": rng.integers(100, 1000, n_days),
        }))
    panel = pd.concat(rows, ignore_index=True)
    panel = panel[~((panel["ticker"] == "S1") & (panel["date"] == dates[200]))]   # a halt
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = panel.groupby("ticker", sort=False)
    panel["prev_date"] = g["date"].shift(1)
    panel["prev_close"] = g["close"].shift(1)
    return panel, mkt


def _dry_run() -> None:
    checks: dict[str, bool] = {}
    panel, mkt = _synthetic()
    out = compute(panel, mkt)
    d = out["beta_252b"].index

    checks["beta recovers planted 0.5/1.0/1.5 (±0.1)"] = bool(
        np.allclose(out["beta_252b"].iloc[-1].to_numpy(), [0.5, 1.0, 1.5], atol=0.1)
    )
    checks["ivol recovers planted 0.004 (±20%)"] = bool(
        np.allclose(out["ivol_63b"].iloc[-1].to_numpy(), 0.004, rtol=0.2)
    )

    # Exact OLS on one window, including the halted bar (a two-day span on both sides).
    s1 = panel[panel["ticker"] == "S1"].reset_index(drop=True)
    y = np.log(s1["close"] / s1["prev_close"])
    x = pd.Series(C.span_logret(mkt, s1["prev_date"], s1["date"]))
    end = int(np.flatnonzero(s1["date"] == d[260])[0])
    yy, xx = y.iloc[end - 62:end + 1].to_numpy(), x.iloc[end - 62:end + 1].to_numpy()
    slope, icpt = np.polyfit(xx, yy, 1)
    resid = yy - (slope * xx + icpt)
    checks["ivol == exact OLS residual sd (n-2)"] = bool(
        np.isclose(out["ivol_63b"].loc[d[260], "S1"], np.sqrt((resid ** 2).sum() / (len(yy) - 2)))
    )
    yy, xx = y.iloc[end - 251:end + 1].to_numpy(), x.iloc[end - 251:end + 1].to_numpy()
    checks["beta == exact OLS slope across a halt"] = bool(
        np.isclose(out["beta_252b"].loc[d[260], "S1"], np.polyfit(xx, yy, 1)[0])
    )
    checks["beta NaN below 126 pairs"] = bool(
        out["beta_252b"]["S0"].iloc[:126].isna().all() and out["beta_252b"]["S0"].iloc[126:].notna().all()
    )

    # Overnight + intraday = the 21-bar close-to-close log return, exactly.
    s0 = panel[panel["ticker"] == "S0"].set_index("date")
    cc = np.log(s0["close"] / s0["close"].shift(21))
    tot = out["ret_overnight_21b"]["S0"] + out["ret_intraday_21b"]["S0"]
    checks["overnight + intraday == 21b log return"] = bool(np.allclose(tot.dropna(), cc.reindex(tot.dropna().index)))
    checks["21-bar sums need a full window"] = bool(out["ret_overnight_21b"]["S0"].iloc[:21].isna().all())
    size = (s0["turnover"] / s0["trades"]).rolling(21, min_periods=21).median()
    checks["trade size == rolling median"] = bool(np.allclose(out["trade_size_21b"]["S0"].dropna(), size.dropna()))

    # Causality: perturb everything after a cut; nothing on or before it may move.
    cut = d[300]
    p2, m2 = panel.copy(), mkt.copy()
    after = p2["date"] > cut
    p2.loc[after, ["open", "close", "turnover"]] *= np.random.default_rng(1).uniform(0.5, 1.5, (int(after.sum()), 3))
    g = p2.groupby("ticker", sort=False)
    p2["prev_close"] = g["close"].shift(1)
    m2[m2.index > cut] *= 1.3
    out2 = compute(p2, m2)
    checks["causal: features <= cut unchanged"] = all(
        out[f].loc[:cut].equals(out2[f].loc[:cut]) for f in out
    )
    checks["perturbation reaches later bars"] = bool(not out["beta_252b"].loc[cut:].iloc[1:].equals(out2["beta_252b"].loc[cut:].iloc[1:]))

    print()
    for k, v in checks.items():
        print(f"  {k:<44}: {'PASS' if v else 'FAIL'}")
    assert all(checks.values()), "price_features dry-run FAILED"
    print("\n[price_features] dry-run OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Phase 6 F1 per-stock price features.")
    ap.add_argument("--build", action="store_true", help="build (else dry-run)")
    args = ap.parse_args()
    if not args.build:
        _dry_run()
        return
    build()


if __name__ == "__main__":
    main()
