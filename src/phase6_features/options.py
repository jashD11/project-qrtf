"""
Phase 6 Part B, F2 — stock-option features (docs/phase6_plan.md §B.2).

Rules applied before anything is computed (§B.2):
- **Expiry.** The near-month expiry: the earliest stock-futures expiry at least
  ``MIN_DAYS`` calendar days away (so it rolls to the next month when fewer than 7
  remain). Options and the forward use the same expiry.
- **Traded contracts only.** ``fo_archive`` keeps only options with contracts > 0. The
  forward F is that expiry's stock-futures close, and only if the future traded that day.
- **IV by Black-76**, with T = calendar days / 365 and r fixed at 6.5%, solved by
  vectorized bisection. A price at or below its discounted intrinsic value, or at or
  above its no-arbitrage ceiling, has no IV and is dropped.

Features, per (date, stock), on the stock's own bars:

    opt_atm_iv       IV interpolated linearly in strike between the two traded strikes
                     bracketing F. At each strike: the mean of call and put IV where
                     both traded, else whichever did.
    opt_ivspread     mean of (call IV − put IV) over strikes within ±10% of F where both traded.
    opt_skew         IV of the traded put with K/F nearest 0.90 (within 0.85–0.95) − opt_atm_iv.
    opt_iv_chg_5b    opt_atm_iv_t − opt_atm_iv 5 own bars earlier.
    opt_iv_rv        opt_atm_iv − 20-own-bar realized vol (stdev of daily log returns × √252).
    opt_volratio_5b  5-own-bar mean of (options traded value, all strikes and expiries) /
                     stock rupee turnover. Options value is NSE's notional (see
                     fo_archive). It is 0 on a day the stock had F&O but no option traded,
                     and NaN on a day it had no F&O at all.
    opt_avail        1 if opt_atm_iv exists on that bar, else 0 (on every traded cell).

    python src/phase6_features/options.py            # dry-run self-test
    python src/phase6_features/options.py --build
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Final

import numpy as np
import pandas as pd
from scipy.special import ndtr

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.phase6_features import _common as C  # noqa: E402
from src.phase6_features import fo_archive as FO  # noqa: E402

RATE: Final[float] = 0.065
MIN_DAYS: Final[int] = 7
SIGMA_LO, SIGMA_HI, N_BISECT = 1e-3, 5.0, 60
SPREAD_BAND: Final[float] = 0.10
SKEW_TARGET, SKEW_LO, SKEW_HI = 0.90, 0.85, 0.95
CHG_LAG, RV_WINDOW, VOL_WINDOW = 5, 20, 5
F2_DIR: Final[str] = os.path.join(C.OUT_DIR, "f2")
FEATURES: Final[tuple[str, ...]] = (
    "opt_ivspread", "opt_skew", "opt_atm_iv", "opt_iv_chg_5b", "opt_iv_rv",
    "opt_volratio_5b", "opt_avail",
)


# ── Pricing ─────────────────────────────────────────────────────────────────────

def black76(F, K, T, sigma, is_call, r: float = RATE):
    """Black-76 price of a European option on a forward."""
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    df = np.exp(-r * T)
    call = df * (F * ndtr(d1) - K * ndtr(d2))
    put = df * (K * ndtr(-d2) - F * ndtr(-d1))
    return np.where(is_call, call, put)


def implied_vol(price, F, K, T, is_call, r: float = RATE):
    """
    Vectorized bisection for Black-76 IV. NaN where the price is outside the
    no-arbitrage band (at or below discounted intrinsic, at or above the ceiling) or
    where the root lies beyond ``SIGMA_HI``.
    """
    price, F, K, T = (np.asarray(a, float) for a in (price, F, K, T))
    is_call = np.asarray(is_call, bool)
    df = np.exp(-r * T)
    intrinsic = df * np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    ceiling = df * np.where(is_call, F, K)
    ok = (price > intrinsic) & (price < ceiling) & (T > 0) & (F > 0) & (K > 0)
    lo = np.full(price.shape, SIGMA_LO)
    hi = np.full(price.shape, SIGMA_HI)
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        above = black76(F, K, T, mid, is_call, r) > price
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    iv = 0.5 * (lo + hi)
    ok &= (iv < SIGMA_HI * 0.999) & (iv > SIGMA_LO * 1.001)
    return np.where(ok, iv, np.nan)


# ── The chain on each (date, symbol) ───────────────────────────────────────────

def near_month(fo: pd.DataFrame) -> pd.DataFrame:
    """Per (date, symbol): the near-month expiry, its days to expiry, and F if it traded."""
    fut = fo[fo["kind"] == "F"].copy()
    fut["days"] = (fut["expiry"] - fut["date"]).dt.days
    fut = fut[fut["days"] >= MIN_DAYS].sort_values(["date", "symbol", "expiry"])
    nm = fut.groupby(["date", "symbol"], sort=False).head(1)
    nm = nm.assign(F=nm["close"].where(nm["contracts"] > 0))
    return nm[["date", "symbol", "expiry", "days", "F"]].reset_index(drop=True)


def chain_features(fo: pd.DataFrame) -> pd.DataFrame:
    """(date, symbol) -> opt_atm_iv, opt_ivspread, opt_skew, plus solver diagnostics."""
    nm = near_month(fo).dropna(subset=["F"])
    opt = fo[fo["kind"] != "F"].merge(nm, on=["date", "symbol", "expiry"], how="inner")
    opt["iv"] = implied_vol(opt["close"], opt["F"], opt["strike"], opt["days"] / 365.0,
                            opt["kind"] == "C")
    n_opt, n_iv = len(opt), int(opt["iv"].notna().sum())

    key = ["date", "symbol"]
    per_k = opt.pivot_table(index=[*key, "strike"], columns="kind", values="iv",
                            aggfunc="first", observed=True)
    per_k = per_k.reindex(columns=["C", "P"]).reset_index()
    per_k = per_k.merge(nm[[*key, "F"]], on=key)
    per_k["iv_k"] = per_k[["C", "P"]].mean(axis=1)            # mean of those present
    per_k = per_k.dropna(subset=["iv_k"]).sort_values([*key, "strike"])

    lo = per_k[per_k["strike"] <= per_k["F"]].groupby(key, sort=False).tail(1)
    hi = per_k[per_k["strike"] >= per_k["F"]].groupby(key, sort=False).head(1)
    br = lo[[*key, "strike", "iv_k", "F"]].merge(
        hi[[*key, "strike", "iv_k"]], on=key, suffixes=("_lo", "_hi"))
    width = br["strike_hi"] - br["strike_lo"]
    w = np.where(width > 0, (br["F"] - br["strike_lo"]) / width.where(width > 0, 1.0), 0.0)
    br["opt_atm_iv"] = br["iv_k_lo"] + w * (br["iv_k_hi"] - br["iv_k_lo"])

    both = per_k.dropna(subset=["C", "P"])
    both = both[(both["strike"] / both["F"] - 1.0).abs() <= SPREAD_BAND]
    spread = (both["C"] - both["P"]).groupby([both["date"], both["symbol"]]).mean()
    spread = spread.rename("opt_ivspread").reset_index()

    puts = per_k.dropna(subset=["P"])
    m = puts["strike"] / puts["F"]
    puts = puts[(m >= SKEW_LO) & (m <= SKEW_HI)].assign(dist=lambda d: (d["strike"] / d["F"] - SKEW_TARGET).abs())
    puts = puts.sort_values([*key, "dist", "strike"]).groupby(key, sort=False).head(1)

    out = br[[*key, "opt_atm_iv"]].merge(spread, on=key, how="outer")
    out = out.merge(puts[[*key, "P"]].rename(columns={"P": "put90_iv"}), on=key, how="outer")
    out["opt_skew"] = out["put90_iv"] - out["opt_atm_iv"]
    out.attrs["solver"] = {"options_on_near_month": n_opt, "with_iv": n_iv}
    return out.drop(columns="put90_iv")


def option_value(fo: pd.DataFrame) -> pd.DataFrame:
    """(date, symbol) -> total traded options value; 0 where the stock had only futures."""
    has_fo = fo[["date", "symbol"]].drop_duplicates()
    val = fo[fo["kind"] != "F"].groupby(["date", "symbol"], observed=True)["value"].sum()
    out = has_fo.merge(val.rename("opt_value").reset_index(), on=["date", "symbol"], how="left")
    return out.fillna({"opt_value": 0.0})


def symbol_map(panel_path: str = C.PANEL) -> pd.DataFrame:
    """Per-(date, symbol) -> engine ticker, as ``shortable.symbol_to_ticker`` maps it."""
    p = pd.read_parquet(panel_path, columns=["date", "symbol", "ticker"])
    p["symbol"] = p["symbol"].astype("string").str.strip().str.upper()
    return p.drop_duplicates(subset=["date", "symbol"])


# ── Own-bar features ────────────────────────────────────────────────────────────

def compute(panel: pd.DataFrame, chain: pd.DataFrame, value: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    ``panel`` is the own-bar long panel (with ``symbol``); ``chain`` and ``value`` are
    keyed (date, symbol). Returns wide (date × ticker) frames.
    """
    df = panel.merge(chain, on=["date", "symbol"], how="left").merge(
        value, on=["date", "symbol"], how="left")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    t = df["ticker"]
    g = df.groupby("ticker", sort=False)

    df["opt_iv_chg_5b"] = df["opt_atm_iv"] - g["opt_atm_iv"].shift(CHG_LAG)
    logret = np.log(df["close"] / df["prev_close"])
    rv = C.own_rolling(logret, t, RV_WINDOW, "std") * np.sqrt(252.0)
    df["opt_iv_rv"] = df["opt_atm_iv"] - rv
    ratio = (df["opt_value"] / df["turnover"]).where(df["turnover"] > 0)
    df["opt_volratio_5b"] = C.own_rolling(ratio, t, VOL_WINDOW, "mean")
    df["opt_avail"] = df["opt_atm_iv"].notna().astype(float)
    df = df.replace([np.inf, -np.inf], np.nan)
    return {f: C.to_wide(df, f) for f in FEATURES}


def build(allow_holdout: bool = False) -> dict[str, pd.DataFrame]:
    fo = FO.load()
    C.guard_holdout(fo["date"], FO.OUT, allow_holdout)
    panel = C.load_panel(allow_holdout=allow_holdout)
    smap = symbol_map()
    panel = panel.merge(smap, on=["date", "ticker"], how="left")

    chain = chain_features(fo)
    s = chain.attrs["solver"]
    print(f"[options] near-month traded options {s['options_on_near_month']:,} | "
          f"IV solved {s['with_iv']:,} ({s['with_iv'] / s['options_on_near_month']:.1%}) | "
          f"(date, symbol) with ATM IV {int(chain['opt_atm_iv'].notna().sum()):,}")
    out = compute(panel, chain, option_value(fo))
    os.makedirs(F2_DIR, exist_ok=True)
    for name, frame in out.items():
        C.write(frame, os.path.join(F2_DIR, f"{name}.parquet"))
    return out


# ── Dry run ─────────────────────────────────────────────────────────────────────

def _synthetic_fo(dates, symbols, rng, smile=0.10):
    rows = []
    for d in dates:
        for j, s in enumerate(symbols):
            F = 100.0 * (1 + 0.1 * j) * np.exp(rng.normal(0, 0.01))
            for exp, days in ((d + pd.Timedelta(days=5), 5), (d + pd.Timedelta(days=33), 33)):
                rows.append((d, s, "F", exp, 0.0, F, 100, 1e7, 1e6))
                T = days / 365.0
                for k in np.arange(0.80, 1.21, 0.025):
                    K = round(F * k, 2)
                    vol = 0.30 + smile * (1 - k)                       # put skew
                    for kind in "CP":
                        px = float(black76(F, K, T, vol, kind == "C"))
                        rows.append((d, s, kind, exp, K, px, 10, 1e6, 1e4))
    return pd.DataFrame(rows, columns=FO.SCHEMA)


def _dry_run() -> None:
    checks: dict[str, bool] = {}
    rng = np.random.default_rng(4)

    F = np.array([100.0, 100.0, 250.0, 80.0])
    K = np.array([90.0, 110.0, 260.0, 60.0])
    T = np.array([30, 30, 45, 10]) / 365.0
    sig = np.array([0.25, 0.40, 0.55, 0.9])
    for is_call in (True, False):
        px = black76(F, K, T, sig, np.full(4, is_call))
        checks[f"IV round-trip ({'calls' if is_call else 'puts'})"] = bool(
            np.allclose(implied_vol(px, F, K, T, np.full(4, is_call)), sig, atol=1e-8))
    df = np.exp(-RATE * 30 / 365)
    checks["price at intrinsic -> NaN"] = bool(np.isnan(implied_vol(df * 10.0, 100, 90, 30 / 365, True)))
    checks["price at ceiling -> NaN"] = bool(np.isnan(implied_vol(df * 100.0, 100, 90, 30 / 365, True)))

    dates = pd.bdate_range("2020-01-01", periods=40)
    fo = _synthetic_fo(dates, ["AAA", "BBB"], rng)
    nm = near_month(fo)
    checks["5-day expiry rolled to next month"] = bool((nm["days"] == 33).all())
    ch = chain_features(fo)
    # ATM IV: smile 0.30 + 0.10(1 - K/F) is linear in K, so interpolation is exact up
    # to which strikes bracket F; the value at K/F = 1 is 0.30.
    checks["ATM IV recovers 0.30 (±0.003)"] = bool(np.allclose(ch["opt_atm_iv"], 0.30, atol=3e-3))
    checks["call-put IV spread ~0 under parity"] = bool(np.allclose(ch["opt_ivspread"], 0.0, atol=1e-6))
    checks["skew = put IV at K/F 0.90 − ATM (0.01)"] = bool(np.allclose(ch["opt_skew"], 0.01, atol=3e-3))

    # Untraded future -> no forward -> no IV that day.
    fo2 = fo.copy()
    first = (fo2["kind"] == "F") & (fo2["date"] == dates[0]) & (fo2["symbol"] == "AAA")
    fo2.loc[first, "contracts"] = 0
    ch2 = chain_features(fo2)
    checks["no traded future -> no ATM IV"] = bool(
        ch2[(ch2["date"] == dates[0]) & (ch2["symbol"] == "AAA")]["opt_atm_iv"].isna().all())

    # Own-bar features on a synthetic panel; BBB has no F&O at all on the last 10 days.
    panel = []
    for s in ("AAA", "BBB", "CCC"):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(dates))))
        panel.append(pd.DataFrame({"date": dates, "ticker": s, "symbol": s, "open": close,
                                   "close": close, "turnover": 5e7, "trades": 100}))
    panel = pd.concat(panel, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["prev_date"] = panel.groupby("ticker")["date"].shift(1)
    panel["prev_close"] = panel.groupby("ticker")["close"].shift(1)
    fo3 = fo[~((fo["symbol"] == "BBB") & (fo["date"] >= dates[30]))]
    out = compute(panel, chain_features(fo3), option_value(fo3))
    checks["opt_avail: 1 with a chain, 0 without"] = bool(
        (out["opt_avail"]["AAA"] == 1).all() and (out["opt_avail"]["CCC"] == 0).all()
        and (out["opt_avail"]["BBB"].iloc[30:] == 0).all())
    checks["F2 NaN for a name with no options"] = bool(out["opt_atm_iv"]["CCC"].isna().all())
    a = out["opt_atm_iv"]["AAA"]
    checks["iv_chg_5b exact"] = bool(np.isclose(out["opt_iv_chg_5b"]["AAA"].iloc[20], a.iloc[20] - a.iloc[15]))
    val = option_value(fo3)
    v_aaa = val[val["symbol"] == "AAA"].set_index("date")["opt_value"] / 5e7
    checks["volratio = 5-bar mean of value/turnover"] = bool(
        np.isclose(out["opt_volratio_5b"]["AAA"].iloc[20], v_aaa.iloc[16:21].mean()))
    checks["volratio NaN once a name leaves F&O"] = bool(out["opt_volratio_5b"]["BBB"].iloc[30:].isna().all())

    cut = dates[25]
    fo4 = fo.copy()
    late = fo4["date"] > cut
    fo4.loc[late, "close"] *= rng.uniform(0.8, 1.2, int(late.sum()))
    o1 = compute(panel, chain_features(fo), option_value(fo))
    o2 = compute(panel, chain_features(fo4), option_value(fo4))
    checks["causal: features <= cut unchanged"] = all(o1[f].loc[:cut].equals(o2[f].loc[:cut]) for f in o1)

    print()
    for k, v in checks.items():
        print(f"  {k:<44}: {'PASS' if v else 'FAIL'}")
    assert all(checks.values()), "options dry-run FAILED"
    print("\n[options] dry-run OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Phase 6 F2 option features.")
    ap.add_argument("--build", action="store_true", help="build (else dry-run)")
    args = ap.parse_args()
    if not args.build:
        _dry_run()
        return
    build()


if __name__ == "__main__":
    main()
