#!/usr/bin/env python3
"""
Phase 6 Part A — why the tree signal decays after 2021 (docs/phase6_plan.md §A).

Read-only diagnostics, frozen before any of their numbers existed. Nothing here is a
strategy, nothing is promoted, and nothing is written to a DSR ledger; Part A counts 0 in
N (``config.TRIAL_LEDGER``). Each section tests one hypothesis and the decision rules of
§A.4 are evaluated mechanically at the end, so the verdict cannot drift towards whichever
story reads best.

  A1  per-feature rank-IC by year, plus a fixed-sign composite        (H1 competed away)
  A1b per-learner IC by year from single-learner refits
  A2  per-feature IC conditioned on causal market-state terciles       (H2 regime flip)
  A3  ridge walk-forward, within-year IC ceiling, window probe, folds  (H3 training setup)
  A4  IC by listing age x liquidity, reweighting decomposition         (H4 universe drift)
  A5  independent-source disagreement, CA ledger, unledgered extremes  (H5 data)

**The holdout is sealed.** The run asserts that no panel it reads extends past
2025-06-30; that period is Part B's forward test and looking at it here would spend it.

New fits (A1b learners, A3c window probe) go through ``cached_walk_forward`` — its key
separates ``model_choice`` and the training window, so they never collide with the
Phase 4c/5 entries. The cached ensemble fit must be a HIT; a miss aborts rather than
silently spending an 80-minute cold refit on something that should already exist.

Usage
    python scripts/phase6_decay.py                         # dry-run self-test
    python scripts/phase6_decay.py --run                   # every section
    python scripts/phase6_decay.py --run --sections A1,A1b
    python scripts/phase6_decay.py --run --no-fits         # skip new walk-forward fits

Writes only under ``data/trial_database/phase6/decay/``.
"""

import argparse
import os
import sys
from typing import Final

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config  # noqa: E402
from src.phase4_data.phase4c_diagnostics import (  # noqa: E402
    _spearman_by_row, liquidity_quintiles, load_panel_inputs,
)

OUT_DIR: Final[str] = "data/trial_database/phase6/decay"
FREQ: Final[str] = "daily_nse500"
HOLDOUT_START: Final[pd.Timestamp] = pd.Timestamp("2025-07-01")

EARLY: Final[tuple[str, str]] = ("2016-01-01", "2020-12-31")
LATE: Final[tuple[str, str]] = ("2021-01-01", "2025-06-30")
RECENT: Final[tuple[str, str]] = ("2024-01-01", "2025-06-30")
COVID: Final[tuple[str, str]] = ("2020-03-01", "2020-12-31")

SKILL_T: Final[float] = 3.0        # early |NW t| for a skill-carrying feature (§A.2)
DECAY_T: Final[float] = 2.0        # late |NW t| below this = decayed
DECAY_RATIO: Final[float] = 0.5    # or late signed IC <= 50% of early
EXTREME_RET: Final[float] = 0.20   # |r| for an "extreme" day (A5c)
DISAGREE_THR: Final[float] = 0.01  # |r_panel - r_drive| for a disagreement (A5a)
RIDGE_ALPHA: Final[float] = 1.0    # frozen, not tuned (§A.3 A3a)
STATE_MIN_PERIODS: Final[int] = 252

# docs/phase4c_results.md §4.2 — the anchor the ensemble 1b table must reproduce.
ANCHOR_1B: Final[dict[int, tuple[float, float]]] = {
    2016: (0.0475, 9.12), 2017: (0.0523, 11.42), 2018: (0.0586, 9.85),
    2019: (0.0863, 10.48), 2020: (0.0564, 8.00), 2021: (0.0217, 4.89),
    2022: (0.0312, 6.28), 2023: (0.0368, 7.91), 2024: (0.0183, 2.31),
    2025: (0.0127, 1.21),
}


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def nw_tstat(x: pd.Series, lag: int) -> float:
    """
    Newey-West (Bartlett) t-stat of the mean of ``x``.

    Lag ``h-1`` for an IC against an h-bar overlapping label: consecutive daily ICs share
    h-1 days of realized return, so the i.i.d. t overstates by about sqrt(h). Lag 0 is the
    plain i.i.d. t, which is what reproduces the §4.2 1-day table.
    """
    v = pd.Series(x).dropna().to_numpy(dtype=float)
    n = v.size
    if n < 3:
        return np.nan
    d = v - v.mean()
    var = d @ d / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2.0 * (1.0 - k / (lag + 1)) * (d[k:] @ d[:-k]) / n
    if lag == 0:
        var = d @ d / (n - 1)       # match _ic_summary's ddof=1 exactly
    return float(v.mean() / np.sqrt(var / n)) if var > 0 else np.nan


def ic_stats(ic: pd.Series, lag: int) -> dict:
    ic = ic.dropna()
    return dict(mean_IC=float(ic.mean()) if len(ic) else np.nan,
                t=nw_tstat(ic, lag), n_days=int(len(ic)))


def _period(ic: pd.Series, span: tuple[str, str]) -> pd.Series:
    return ic.loc[span[0]:span[1]]


def by_year(ic: pd.Series, lag: int) -> pd.DataFrame:
    rows = [dict(year=int(y), **ic_stats(g, lag)) for y, g in ic.groupby(ic.index.year)]
    return pd.DataFrame(rows).set_index("year")


def by_period(ic: pd.Series, lag: int) -> dict[str, dict]:
    return {name: ic_stats(_period(ic, span), lag)
            for name, span in (("early", EARLY), ("late", LATE), ("recent", RECENT))}


def ratio(late: float, early: float) -> float:
    return late / early if early and np.isfinite(early) and early != 0 else np.nan


# --------------------------------------------------------------------------- #
# A1 — per-feature IC (H1)
# --------------------------------------------------------------------------- #
def per_feature_ic(
    feats: dict[str, pd.DataFrame], realized: pd.DataFrame, lag: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    """(by-year long table, by-period table, daily IC per feature)."""
    daily = {f: _spearman_by_row(w, realized) for f, w in feats.items()}
    yearly = pd.concat(
        {f: by_year(ic, lag) for f, ic in daily.items()}, names=["feature"]
    )
    period = pd.DataFrame({
        f: {f"{p}_{k}": v for p, s in by_period(ic, lag).items() for k, v in s.items()}
        for f, ic in daily.items()
    }).T
    return yearly, period, daily


def classify_features(period_5b: pd.DataFrame) -> pd.DataFrame:
    """Skill-carrying and decayed flags, exactly as §A.2 defines them."""
    out = period_5b[["early_mean_IC", "early_t", "late_mean_IC", "late_t"]].copy()
    out["skill"] = out["early_t"].abs() >= SKILL_T
    out["sign"] = np.sign(out["early_mean_IC"])
    signed_early = out["sign"] * out["early_mean_IC"]
    signed_late = out["sign"] * out["late_mean_IC"]
    out["late_over_early"] = signed_late / signed_early
    out["decayed"] = out["skill"] & (
        (out["late_t"].abs() < DECAY_T) | (signed_late <= DECAY_RATIO * signed_early)
    )
    return out


def fixed_sign_composite(feats: dict[str, pd.DataFrame], signs: pd.Series) -> pd.DataFrame:
    """Equal-weight sum of skill-carrying features times their early sign — never refit."""
    parts = [feats[f] * float(s) for f, s in signs.items()]
    if not parts:
        raise ValueError("no skill-carrying features — the composite is undefined")
    total = parts[0]
    for p in parts[1:]:
        total = total + p
    return total


def sign_flips(yearly_5b: pd.DataFrame, cls: pd.DataFrame) -> pd.DataFrame:
    """Late years in which a skill-carrying feature reverses sign at |NW t| >= 2."""
    rows = []
    for f in cls.index[cls["skill"]]:
        s = cls.loc[f, "sign"]
        yr = yearly_5b.loc[f]
        late = yr[yr.index >= int(LATE[0][:4])]
        for y, r in late.iterrows():
            if np.sign(r["mean_IC"]) == -s and abs(r["t"]) >= DECAY_T:
                rows.append(dict(feature=f, year=int(y), mean_IC=r["mean_IC"], t=r["t"]))
    return pd.DataFrame(rows, columns=["feature", "year", "mean_IC", "t"])


# --------------------------------------------------------------------------- #
# A2 — state conditioning (H2)
# --------------------------------------------------------------------------- #
def expanding_terciles(state: pd.Series, min_periods: int = STATE_MIN_PERIODS) -> pd.Series:
    """Tercile 1/2/3 of each day against the state's own history up to and including t."""
    exp = state.expanding(min_periods=min_periods)
    q1, q2 = exp.quantile(1 / 3), exp.quantile(2 / 3)
    t = pd.Series(np.nan, index=state.index)
    t[state <= q1] = 1
    t[(state > q1) & (state <= q2)] = 2
    t[state > q2] = 3
    return t.where(q1.notna())


def state_difference(ic: pd.Series, terc: pd.Series, lag: int) -> dict:
    """
    Mean IC in the top state tercile minus the bottom, with a NW t on the difference.

    Built as the mean of x_t = IC_t * (1[top]/p_top - 1[bot]/p_bot), whose mean is the
    difference and whose serial dependence the NW lag handles.
    """
    df = pd.concat([ic.rename("ic"), terc.rename("s")], axis=1).dropna()
    top, bot = df["s"] == 3, df["s"] == 1
    if top.sum() < 20 or bot.sum() < 20:
        return dict(diff=np.nan, t=np.nan, ic_top=np.nan, ic_bot=np.nan)
    x = df["ic"] * (top / top.mean() - bot / bot.mean())
    return dict(diff=float(df.loc[top, "ic"].mean() - df.loc[bot, "ic"].mean()),
                t=nw_tstat(x, lag),
                ic_top=float(df.loc[top, "ic"].mean()), ic_bot=float(df.loc[bot, "ic"].mean()))


def tercile_mix(terc: pd.Series) -> pd.DataFrame:
    rows = {}
    for name, span in (("early", EARLY), ("late", LATE)):
        s = terc.loc[span[0]:span[1]].dropna()
        rows[name] = {int(k): float((s == k).mean()) for k in (1, 2, 3)}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# A3 — training setup (H3)
# --------------------------------------------------------------------------- #
def walk_forward_folds(
    unique_days: np.ndarray, train: int, predict: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train_days, test_days) per fold — the same slicing as run_walk_forward."""
    folds, start, n = [], train, len(unique_days)
    while start < n:
        folds.append((unique_days[start - train:start],
                      unique_days[start:min(start + predict, n)]))
        start += predict
    return folds


def _row_days(df: pd.DataFrame) -> np.ndarray:
    return pd.DatetimeIndex(df.index.get_level_values("date")).normalize().to_numpy()


def ridge_walk_forward(
    features_df: pd.DataFrame, feat_cols: list[str], target: str,
    train: int, predict: int, alpha: float = RIDGE_ALPHA,
) -> pd.DataFrame:
    """Walk-forward ridge on the tree stack's exact folds — a linear, low-capacity twin."""
    from sklearn.linear_model import Ridge

    df = features_df.sort_index()
    days = _row_days(df)
    preds = []
    for tr_days, te_days in walk_forward_folds(np.unique(days), train, predict):
        tr = df.iloc[np.searchsorted(days, tr_days[0], "left"):
                     np.searchsorted(days, tr_days[-1], "right")]
        te = df.iloc[np.searchsorted(days, te_days[0], "left"):
                     np.searchsorted(days, te_days[-1], "right")]
        ok = tr[target].notna()
        if not ok.any() or te.empty:
            continue
        m = Ridge(alpha=alpha).fit(tr.loc[ok, feat_cols].values, tr.loc[ok, target].values)
        preds.append(pd.Series(m.predict(te[feat_cols].values), index=te.index))
    return pd.concat(preds).sort_index().unstack("ticker")


def within_year_ceiling(
    features_df: pd.DataFrame, feat_cols: list[str], target: str, h: int,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """
    In-year lgbm predictions: each calendar quarter scored by a model trained on the
    same year's other quarters, purging h trading days either side of the test block.

    It uses the future of the year, so it is a ceiling on how much in-year signal the
    features carry — never a strategy.
    """
    from src.production_ml import tier1_trees  # noqa: F401  (sets the OpenMP env first)
    import lightgbm as lgb

    ml = config.ML_CONFIG.lgbm
    df = features_df.sort_index()
    days = pd.DatetimeIndex(_row_days(df))
    uniq = pd.DatetimeIndex(np.unique(days))
    preds = []
    for y in (years or sorted(set(uniq.year))):
        yd = uniq[uniq.year == y]
        quarters = sorted(set(yd.quarter))
        if len(quarters) < 2:
            continue
        for q in quarters:
            te_days = yd[yd.quarter == q]
            lo = uniq.get_loc(te_days[0]) - h
            hi = uniq.get_loc(te_days[-1]) + h
            purge = uniq[max(lo, 0):min(hi, len(uniq) - 1) + 1]
            tr_days = yd.difference(purge)
            tr = df[days.isin(tr_days)]
            te = df[days.isin(te_days)]
            ok = tr[target].notna()
            if ok.sum() < 100 or te.empty:
                continue
            m = lgb.LGBMRegressor(
                learning_rate=ml.learning_rate, n_estimators=ml.n_estimators,
                max_depth=ml.max_depth, num_leaves=ml.num_leaves,
                random_state=42, n_jobs=1, verbose=-1,
            ).fit(tr.loc[ok, feat_cols].values, tr.loc[ok, target].values)
            preds.append(pd.Series(m.predict(te[feat_cols].values), index=te.index))
    return pd.concat(preds).sort_index().unstack("ticker")


def fold_composition(
    daily_ic: pd.Series, unique_days: np.ndarray, train: int, predict: int,
    zvol: pd.Series,
) -> pd.DataFrame:
    """Per-fold OOS IC against the training window's COVID share and mean z-vol."""
    rows = []
    c0, c1 = pd.Timestamp(COVID[0]), pd.Timestamp(COVID[1])
    for tr_days, te_days in walk_forward_folds(unique_days, train, predict):
        tr = pd.DatetimeIndex(tr_days)
        rows.append(dict(
            test_start=te_days[0], test_end=te_days[-1],
            fold_IC=float(daily_ic.loc[te_days[0]:te_days[-1]].mean()),
            covid_share=float(((tr >= c0) & (tr <= c1)).mean()),
            train_zvol=float(zvol.reindex(tr).mean()),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# A4 — universe drift (H4)
# --------------------------------------------------------------------------- #
AGE_BUCKETS: Final[tuple[str, ...]] = ("1-3y", "3-5y", "5y+")


def age_buckets(first_seen: pd.Series, index: pd.Index, columns: pd.Index,
                censor: pd.Timestamp) -> pd.DataFrame:
    """Listing-age bucket per (date, name); names first seen at the censor date are 5y+."""
    fs = first_seen.reindex(columns)
    age = (index.values[:, None] - fs.values[None, :]) / np.timedelta64(1, "D") / 365.25
    out = np.where(age < 3, "1-3y", np.where(age < 5, "3-5y", "5y+")).astype(object)
    out[:, (fs == censor).to_numpy()] = "5y+"
    out[:, fs.isna().to_numpy()] = None
    return pd.DataFrame(out, index=index, columns=columns)


def bucket_ic(signal: pd.DataFrame, realized: pd.DataFrame,
              buckets: pd.DataFrame) -> dict[str, pd.Series]:
    return {b: _spearman_by_row(signal.where(buckets == b), realized.where(buckets == b))
            for b in AGE_BUCKETS}


def reweight_decomposition(
    signal: pd.DataFrame, realized: pd.DataFrame, buckets: pd.DataFrame,
) -> dict:
    """
    Share of the early->late IC drop explained by the age mix moving.

    IC_mix(w, P) = sum_b w_b * meanIC_b(P). Total drop D = IC_mix(w_early, early) -
    IC_mix(w_late, late); the drop with weights frozen at the early mix is
    D_fixed = IC_mix(w_early, early) - IC_mix(w_early, late). Drift share = 1 - D_fixed/D.
    """
    ics = bucket_ic(signal, realized, buckets)
    valid = signal.notna() & realized.notna()

    def w(span):
        b = buckets.loc[span[0]:span[1]].where(valid.loc[span[0]:span[1]])
        cnt = pd.Series({k: int((b == k).sum().sum()) for k in AGE_BUCKETS})
        return cnt / cnt.sum()

    def m(span):
        return pd.Series({k: float(_period(ics[k], span).mean()) for k in AGE_BUCKETS})

    we, wl, me, ml = w(EARLY), w(LATE), m(EARLY), m(LATE)

    def mix(wt, mu):
        ok = mu.notna()
        return float((wt[ok] / wt[ok].sum() * mu[ok]).sum())

    d = mix(we, me) - mix(wl, ml)
    d_fixed = mix(we, me) - mix(we, ml)
    return dict(w_early=we.to_dict(), w_late=wl.to_dict(),
                ic_early=me.to_dict(), ic_late=ml.to_dict(),
                total_drop=d, drop_at_early_mix=d_fixed,
                drift_share=(1.0 - d_fixed / d) if d > 0 else np.nan)


# --------------------------------------------------------------------------- #
# A5 — data integrity (H5)
# --------------------------------------------------------------------------- #
def disagreement_by_year(ret_a: pd.DataFrame, ret_b: pd.DataFrame,
                         thr: float = DISAGREE_THR) -> pd.DataFrame:
    a, b = ret_a.align(ret_b, join="inner")
    both = a.notna() & b.notna()
    bad = ((a - b).abs() > thr) & both
    n, k = both.sum(axis=1), bad.sum(axis=1)
    g = pd.DataFrame({"matched": n, "disagree": k}).groupby(n.index.year).sum()
    g["rate"] = g["disagree"] / g["matched"].replace(0, np.nan)
    return g


def unledgered_extremes(ret: pd.DataFrame, universe: pd.DataFrame,
                        ledger: pd.DataFrame, thr: float = EXTREME_RET) -> pd.DataFrame:
    """|r| > thr in-universe with no corporate-action entry on that name within +/-1 day."""
    led = ledger.reindex(index=ret.index, columns=ret.columns, fill_value=False).astype(bool)
    near = led | led.shift(1, fill_value=False) | led.shift(-1, fill_value=False)
    uni = universe.reindex(index=ret.index, columns=ret.columns, fill_value=False).astype(bool)
    return (ret.abs() > thr) & uni & ~near


def label_exclusion(extremes: pd.DataFrame, h: int) -> pd.DataFrame:
    """True at s when the h-bar label from s spans an extreme day (returns s+1..s+h)."""
    out = extremes.shift(-1, fill_value=False)
    for j in range(2, h + 1):
        out = out | extremes.shift(-j, fill_value=False)
    return out


# --------------------------------------------------------------------------- #
# §A.4 decision rules
# --------------------------------------------------------------------------- #
def evaluate_rules(r: dict) -> pd.DataFrame:
    """Mechanical evaluation of §A.4. A missing input reports n/a, never False."""
    rows = []

    def add(h, fired, detail):
        rows.append(dict(hypothesis=h, fired=fired, detail=detail))

    cls = r.get("classify")
    ens = r.get("ens_period_5b")
    ceil = r.get("ceiling_period")
    ridge = r.get("ridge_period")

    # H1
    if cls is not None and ceil is not None and ridge is not None and ens is not None:
        skill = cls[cls["skill"]]
        frac = float(skill["decayed"].mean()) if len(skill) else np.nan
        c_ratio = ratio(ceil["recent"]["mean_IC"], ceil["early"]["mean_IC"])
        e_ratio = ratio(ens["late"]["mean_IC"], ens["early"]["mean_IC"])
        r_ratio = ratio(ridge["late"]["mean_IC"], ridge["early"]["mean_IC"])
        fired = bool(frac >= 2 / 3 and c_ratio <= 0.5 and abs(r_ratio - e_ratio) <= 0.2)
        add("H1", fired, f"decayed {frac:.0%} of {len(skill)} skill feats; ceiling recent/early "
                         f"{c_ratio:.2f}; ridge late/early {r_ratio:.2f} vs ensemble {e_ratio:.2f}")
    else:
        add("H1", "n/a", "needs A1 + A3a + A3b")

    # H2
    flips, sd, mixes = r.get("sign_flips"), r.get("state_diff"), r.get("state_mix")
    if flips is not None and sd is not None and mixes is not None:
        n_flip = flips["feature"].nunique() if len(flips) else 0
        strong = sd[sd["t"].abs() >= 3.0]
        shifted = {s: float((mixes[s].loc["late"] - mixes[s].loc["early"]).abs().max())
                   for s in mixes}
        hit = [(f, s) for (f, s) in strong.index if shifted.get(s, 0) >= 0.10]
        fired = bool(n_flip >= 3 or len(hit) > 0)
        add("H2", fired, f"{n_flip} feats flip sign; {len(strong)} feat x state |t|>=3; "
                         f"with a >=10pp mix shift: {hit[:5]}; max mix shift {shifted}")
    else:
        add("H2", "n/a", "needs A1 + A2")

    # H3
    if cls is not None and ceil is not None and ens is not None:
        skill = cls[cls["skill"]]
        c_ratio = ratio(ceil["recent"]["mean_IC"], ceil["early"]["mean_IC"])
        held = float((~skill["decayed"]).mean()) if len(skill) else np.nan
        e_ratio = ratio(ens["recent"]["mean_IC"], ens["early"]["mean_IC"])
        part1 = bool(c_ratio >= 0.7 and held >= 0.5 and e_ratio <= 0.5)
        probe = r.get("window_probe")
        part2, rec = False, np.nan
        if probe is not None:
            gap = probe["lgbm504_early"] - probe["recent_504"]
            rec = (probe["recent_252"] - probe["recent_504"]) / gap if gap > 0 else np.nan
            part2 = bool(np.isfinite(rec) and rec >= 0.5)
        add("H3", bool(part1 or part2),
            f"ceiling recent/early {c_ratio:.2f}; skill feats held {held:.0%}; ensemble "
            f"recent/early {e_ratio:.2f}; 252-window gap recovery "
            f"{rec if probe is not None else 'n/a (no probe)'}")
    else:
        add("H3", "n/a", "needs A1 + A3b")

    # H4
    rw = r.get("reweight")
    if rw is not None:
        add("H4", bool(np.isfinite(rw["drift_share"]) and rw["drift_share"] >= 0.5),
            f"drift share of the early->late drop {rw['drift_share']:.2f}")
    else:
        add("H4", "n/a", "needs A4")

    # H5
    d5 = r.get("h5")
    if d5 is not None:
        fired = bool(d5["disagree_ratio"] >= 2 or d5["extreme_ratio"] >= 2
                     or d5["ic_lift"] >= 0.25)
        add("H5", fired, f"disagreement late/early {d5['disagree_ratio']:.2f}; extremes "
                         f"late/early {d5['extreme_ratio']:.2f}; late IC lift ex-extremes "
                         f"{d5['ic_lift']:+.1%}")
    else:
        add("H5", "n/a", "needs A5")
    return pd.DataFrame(rows).set_index("hypothesis")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _save(obj, name: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.csv")
    (obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)).to_csv(path)
    print(f"[phase6] wrote {path}")


def _banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _assert_sealed() -> None:
    """No file Part A reads may extend into the Part B holdout."""
    for path, col in ((config.freq_spec(FREQ).parquet, "timestamp"),
                      ("data/bhavcopy/panel_daily.parquet", "date")):
        mx = pd.to_datetime(pd.read_parquet(path, columns=[col])[col]).max()
        mx = mx.tz_localize(None) if getattr(mx, "tzinfo", None) else mx
        assert mx < HOLDOUT_START, f"{path} reaches {mx} — the Part B holdout is not sealed"


def _fit(engine, features, allow_cold: bool):
    from src.production_ml.wf_cache import _slot, cache_key, cached_walk_forward
    hit = os.path.exists(os.path.join(_slot(cache_key(engine, features, FREQ)), "key.txt"))
    if not hit and not allow_cold:
        raise RuntimeError(f"cache MISS for {engine.model_choice}/{engine.train_window} — "
                           "expected a hit; refusing a silent cold refit")
    print(f"[phase6] {engine.model_choice} w={engine.train_window}: "
          f"{'cache HIT' if hit else 'cold fit'}")
    return cached_walk_forward(engine, features, FREQ)


def _universe(load_universe_mask, index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    uni = load_universe_mask(FREQ)
    uni.index = pd.DatetimeIndex(uni.index).normalize()
    return uni.reindex(index=index, columns=columns).fillna(False).astype(bool)


def run(sections: set[str], new_fits: bool) -> pd.DataFrame:
    from run_pipeline_ml import load_ohlcv_multiindex, load_universe_mask, phase4_regime_config
    from src.production_ml.feature_creator import (
        create_features, present_feature_columns,
    )
    from src.production_ml.tier1_trees import DECILE_PCT, TreeAlphaEngine
    from src.production_ml.tier2_regime import _causal_zscore, build_features

    _assert_sealed()
    target, h = "tgt_fwd_logret_5b", 5
    lag = h - 1

    bars = load_ohlcv_multiindex(FREQ)
    close = bars["close"].unstack("ticker")
    features = create_features(bars, FREQ, universe_mask=load_universe_mask(FREQ))
    del bars
    feat_cols = present_feature_columns(features)
    assert len(feat_cols) == 17, f"expected the 17 price features, got {len(feat_cols)}"

    ens = _fit(TreeAlphaEngine.from_frequency(FREQ, target_col=target, decile_pct=DECILE_PCT),
               features, allow_cold=False)
    alpha = ens.alpha_scores
    idx, cols = alpha.index, alpha.columns
    turnover, universe, real_1b = load_panel_inputs(idx, cols)
    real_5b = features[target].unstack("ticker").reindex(index=idx, columns=cols)
    feats = {f: features[f].unstack("ticker").reindex(index=idx, columns=cols)
             for f in feat_cols}

    # ── Anchor 1: the ensemble must reproduce §4.2 before anything new is read ──
    _banner("ANCHOR — ensemble 1b IC by year vs docs/phase4c_results.md §4.2")
    ens_ic_1b = _spearman_by_row(alpha, real_1b)
    ey = by_year(ens_ic_1b, 0)
    ok1 = all(abs(ey.loc[y, "mean_IC"] - m) < 5e-5 and abs(ey.loc[y, "t"] - t) < 5e-3
              for y, (m, t) in ANCHOR_1B.items())
    print(ey.round(4).to_string())
    print(f"  anchor 1: {'PASS' if ok1 else 'FAIL'}")
    if not ok1:
        raise SystemExit("anchor 1 FAILED — stop and investigate before reading anything new")

    res: dict = {}
    ens_ic_5b = _spearman_by_row(alpha, real_5b)
    res["ens_period_5b"] = by_period(ens_ic_5b, lag)
    _save(pd.concat({"1b": ey, "5b": by_year(ens_ic_5b, lag)}, axis=1), "ensemble_ic_by_year")

    # ── A1 ─────────────────────────────────────────────────────────────────────
    if "A1" in sections or "A2" in sections:
        _banner("A1 — PER-FEATURE IC BY YEAR (H1)")
        y1, p1, _ = per_feature_ic(feats, real_1b, 0)
        y5, p5, daily5 = per_feature_ic(feats, real_5b, lag)
        cls = classify_features(p5)
        res["classify"], res["sign_flips"] = cls, sign_flips(y5, cls)
        signs = cls.loc[cls["skill"], "sign"]
        comp = fixed_sign_composite(feats, signs)
        comp_y = pd.concat({"1b": by_year(_spearman_by_row(comp, real_1b), 0),
                            "5b": by_year(_spearman_by_row(comp, real_5b), lag)}, axis=1)
        print(cls.round(4).to_string())
        print("\n  5b IC by year (feature x year):")
        print(y5["mean_IC"].unstack("year").round(4).to_string())
        print("\n  fixed-sign composite:")
        print(comp_y.round(4).to_string())
        print("\n  sign flips:", "none" if res["sign_flips"].empty else "")
        if not res["sign_flips"].empty:
            print(res["sign_flips"].round(4).to_string())
        for obj, name in ((y1, "a1_feature_ic_by_year_1b"), (y5, "a1_feature_ic_by_year_5b"),
                          (p1, "a1_feature_ic_by_period_1b"), (cls, "a1_classify_5b"),
                          (comp_y, "a1_composite_by_year"),
                          (res["sign_flips"], "a1_sign_flips")):
            _save(obj, name)

    # ── A1b ────────────────────────────────────────────────────────────────────
    if "A1b" in sections and new_fits:
        _banner("A1b — PER-LEARNER IC BY YEAR (single-learner refits)")
        cached = ens.ic_diagnostics
        tabs, ok2 = {}, True
        for m in ("lgbm", "xgb", "rf"):
            wf = _fit(TreeAlphaEngine.from_frequency(FREQ, target_col=target, model_choice=m,
                                                     decile_pct=DECILE_PCT),
                      features, allow_cold=True)
            got, want = wf.ic_diagnostics.loc[m, "mean_IC"], cached.loc[m, "mean_IC"]
            ok2 &= abs(got - want) < 5e-5
            print(f"  anchor 2 {m}: standalone {got:.4f} vs cached {want:.4f}")
            tabs[m] = pd.concat({"1b": by_year(_spearman_by_row(wf.alpha_scores, real_1b), 0),
                                 "5b": by_year(_spearman_by_row(wf.alpha_scores, real_5b), lag)},
                                axis=1)
            if m == "lgbm":
                res["lgbm504"] = wf.alpha_scores
        print(f"  anchor 2: {'PASS' if ok2 else 'FAIL'}")
        if not ok2:
            raise SystemExit("anchor 2 FAILED — stop and investigate")
        tabs["ensemble"] = pd.concat({"1b": ey, "5b": by_year(ens_ic_5b, lag)}, axis=1)
        out = pd.concat(tabs, axis=1)
        print(out.xs("mean_IC", axis=1, level=2).round(4).to_string())
        _save(out, "a1b_learner_ic_by_year")

    # ── A2 ─────────────────────────────────────────────────────────────────────
    if "A2" in sections:
        _banner("A2 — STATE-CONDITIONED FEATURE IC (H2)")
        rf = build_features(phase4_regime_config())
        z = _causal_zscore(rf, config.ML_CONFIG.regime.zscore_min_periods)
        uni_full = _universe(load_universe_mask, close.index, close.columns)
        r21 = np.log(close / close.shift(21)).where(uni_full)
        states = {
            "zvol": z["log_realized_vol"],
            "avgcorr": z["avg_corr"],
            "trend": rf["mkt_ret"].rolling(252, min_periods=252).sum(),
            "dispersion": r21.std(axis=1),
        }
        terc = {s: expanding_terciles(v).reindex(idx) for s, v in states.items()}
        res["state_mix"] = {s: tercile_mix(t) for s, t in terc.items()}
        skill = res["classify"].index[res["classify"]["skill"]]
        sd = pd.DataFrame({(f, s): state_difference(daily5[f], terc[s], lag)
                           for f in skill for s in terc}).T
        sd.index.names = ["feature", "state"]
        res["state_diff"] = sd
        print(sd.round(4).to_string())
        for s, mx in res["state_mix"].items():
            print(f"\n  {s} tercile mix:\n{mx.round(3).to_string()}")
        _save(sd, "a2_state_difference")
        _save(pd.concat(res["state_mix"]), "a2_state_mix")

    # ── A3 ─────────────────────────────────────────────────────────────────────
    if "A3" in sections:
        _banner("A3 — TRAINING SETUP (H3)")
        spec = config.freq_spec(FREQ)
        ridge = ridge_walk_forward(features, feat_cols, target, spec.train_days,
                                   spec.predict_days).reindex(index=idx, columns=cols)
        ridge_ic = _spearman_by_row(ridge, real_5b)
        res["ridge_period"] = by_period(ridge_ic, lag)
        ceil = within_year_ceiling(features, feat_cols, target, h,
                                   years=list(range(2016, 2026))).reindex(columns=cols)
        ceil_ic = _spearman_by_row(ceil, real_5b.reindex(ceil.index))
        res["ceiling_period"] = by_period(ceil_ic, lag)
        a3 = pd.concat({"ensemble": by_year(ens_ic_5b, lag), "ridge": by_year(ridge_ic, lag),
                        "ceiling": by_year(ceil_ic, lag)}, axis=1)
        print(a3.round(4).to_string())
        print("\n  periods:", {k: {p: round(v["mean_IC"], 4) for p, v in res[k].items()}
                               for k in ("ens_period_5b", "ridge_period", "ceiling_period")})
        _save(a3, "a3_ridge_ceiling_by_year")

        if new_fits:
            lg = {504: res.get("lgbm504")}
            if lg[504] is None:
                lg[504] = _fit(TreeAlphaEngine.from_frequency(
                    FREQ, target_col=target, model_choice="lgbm", decile_pct=DECILE_PCT),
                    features, allow_cold=True).alpha_scores
            for w in (252, 1008):
                lg[w] = _fit(TreeAlphaEngine(target_col=target, model_choice="lgbm",
                                             train_window=w, predict_window=spec.predict_days,
                                             decile_pct=DECILE_PCT),
                             features, allow_cold=True).alpha_scores
            common = lg[504].index.intersection(lg[252].index).intersection(lg[1008].index)
            ics = {w: _spearman_by_row(a.reindex(index=common, columns=cols),
                                       real_5b.reindex(common)) for w, a in lg.items()}
            ic504_full = _spearman_by_row(lg[504].reindex(columns=cols), real_5b)
            res["window_probe"] = {
                "lgbm504_early": float(_period(ic504_full, EARLY).mean()),
                **{f"recent_{w}": float(_period(ic, RECENT).mean()) for w, ic in ics.items()},
            }
            wp = pd.concat({w: by_year(ic, lag) for w, ic in ics.items()}, axis=1)
            print(f"\n  window probe on {len(common)} common dates "
                  f"({common.min().date()} -> {common.max().date()}):")
            print(wp.round(4).to_string())
            print("  ", {k: round(v, 4) for k, v in res["window_probe"].items()})
            _save(wp, "a3_window_probe")

        zvol = _causal_zscore(build_features(phase4_regime_config()),
                              config.ML_CONFIG.regime.zscore_min_periods)["log_realized_vol"]
        fc = fold_composition(ens_ic_5b, np.unique(_row_days(features.sort_index())),
                              spec.train_days, spec.predict_days, zvol)
        fc = fc.dropna(subset=["fold_IC"])
        print(f"\n  folds: {len(fc)} | spearman(fold_IC, covid_share) = "
              f"{fc['fold_IC'].corr(fc['covid_share'], method='spearman'):+.3f} | "
              f"spearman(fold_IC, train_zvol) = "
              f"{fc['fold_IC'].corr(fc['train_zvol'], method='spearman'):+.3f}")
        _save(fc, "a3_fold_composition")

    # ── A4 ─────────────────────────────────────────────────────────────────────
    if "A4" in sections:
        _banner("A4 — UNIVERSE DRIFT (H4)")
        panel = pd.read_parquet("data/bhavcopy/panel_daily.parquet", columns=["date", "ticker"])
        first = panel.groupby("ticker")["date"].min()
        censor = panel["date"].min()
        buckets = age_buckets(first, idx, cols, censor)
        valid = alpha.notna() & real_5b.notna()
        share = pd.DataFrame({b: (buckets == b).where(valid).sum(axis=1) for b in AGE_BUCKETS})
        share = share.groupby(share.index.year).sum()
        share = share.div(share.sum(axis=1), axis=0)
        cov = (alpha.notna().sum(axis=1) / universe.sum(axis=1).replace(0, np.nan))
        cov = cov.groupby(cov.index.year).mean().rename("scored_over_universe")
        bic = bucket_ic(alpha, real_5b, buckets)
        tab = pd.concat({b: by_year(ic, lag) for b, ic in bic.items()}, axis=1)
        q = liquidity_quintiles(turnover, universe)
        rows = []
        for b in AGE_BUCKETS:
            for k in range(1, 6):
                sel = (buckets == b) & (q == k)
                ic = _spearman_by_row(alpha.where(sel), real_5b.where(sel))
                for p, s in by_period(ic, lag).items():
                    rows.append(dict(bucket=b, quintile=k, period=p, **s))
        bq = pd.DataFrame(rows)
        res["reweight"] = reweight_decomposition(alpha, real_5b, buckets)
        print(tab.xs("mean_IC", axis=1, level=1).round(4).to_string())
        print("\n  age share of scored name-days:\n" + share.round(3).to_string())
        print("\n" + cov.round(3).to_string())
        print("\n  reweighting:", {k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in res["reweight"].items()})
        _save(tab, "a4_ic_by_age_year")
        _save(share.join(cov), "a4_age_share_coverage")
        _save(bq, "a4_ic_by_age_quintile_period")
        _save(pd.Series(res["reweight"]).to_frame("value"), "a4_reweight")

    # ── A5 ─────────────────────────────────────────────────────────────────────
    if "A5" in sections:
        _banner("A5 — DATA INTEGRITY (H5)")
        ret = close.pct_change(fill_method=None)
        drive = pd.read_parquet("data/daily_ohlcv.parquet")
        drive["date"] = pd.to_datetime(drive["timestamp"]).dt.tz_localize(None).dt.normalize()
        dclose = drive.pivot_table(index="date", columns="ticker", values="close",
                                   aggfunc="last")
        dret = dclose.pct_change(fill_method=None)
        dis = disagreement_by_year(ret.loc[:, ret.columns.intersection(dret.columns)], dret)

        ca = pd.read_csv("data/bhavcopy/corporate_actions.csv", parse_dates=["date"])
        pmap = pd.read_parquet("data/bhavcopy/panel_daily.parquet",
                               columns=["isin", "ticker"]).drop_duplicates()
        ca = ca.merge(pmap, on="isin", how="left")
        ca["ticker"] = ca["ticker"].fillna(ca["symbol"])
        ca_y = ca.groupby(ca["date"].dt.year).agg(actions=("isin", "size"),
                                                  review=("review", "sum"))
        led = pd.DataFrame(False, index=ret.index, columns=ret.columns)
        for d, t in ca[["date", "ticker"]].itertuples(index=False):
            if d in led.index and t in led.columns:
                led.loc[d, t] = True
        uni_full = _universe(load_universe_mask, ret.index, ret.columns)
        ext = unledgered_extremes(ret, uni_full, led)
        ext_y = pd.DataFrame({"extremes": ext.sum(axis=1), "uni_days": uni_full.sum(axis=1)})
        ext_y = ext_y.groupby(ext_y.index.year).sum()
        ext_y["rate"] = ext_y["extremes"] / ext_y["uni_days"]

        excl = label_exclusion(ext, h).reindex(index=idx, columns=cols, fill_value=False)
        ic_ex = _spearman_by_row(alpha, real_5b.where(~excl))
        late_ex = float(_period(ic_ex, LATE).mean())
        late_in = res["ens_period_5b"]["late"]["mean_IC"]

        def pr(tab, col):
            e = tab.loc[int(EARLY[0][:4]):int(EARLY[1][:4])]
            lt = tab.loc[int(LATE[0][:4]):int(LATE[1][:4])]
            return ratio(lt[col[0]].sum() / lt[col[1]].sum(), e[col[0]].sum() / e[col[1]].sum())

        res["h5"] = dict(disagree_ratio=pr(dis, ("disagree", "matched")),
                         extreme_ratio=pr(ext_y, ("extremes", "uni_days")),
                         ic_lift=late_ex / late_in - 1.0)
        print("  (a) disagreement vs the Drive feed:\n" + dis.round(4).to_string())
        print("\n  (b) corporate-action ledger:\n" + ca_y.to_string())
        print("\n  (c) unledgered |r|>20% days:\n" + ext_y.round(5).to_string())
        print(f"\n  late ensemble 5b IC {late_in:.4f} -> {late_ex:.4f} ex-extremes")
        _save(dis, "a5_disagreement")
        _save(ca_y, "a5_ca_ledger")
        _save(ext_y, "a5_extremes")
        _save(pd.Series(res["h5"]).to_frame("value"), "a5_summary")

    _banner("§A.4 DECISION RULES")
    rules = evaluate_rules(res)
    with pd.option_context("display.max_colwidth", 200):
        print(rules.to_string())
    _save(rules, "rules")
    return rules


# --------------------------------------------------------------------------- #
# Dry run — every test must find an effect planted where its answer is known
# --------------------------------------------------------------------------- #
def _dry_run() -> None:
    rng = np.random.default_rng(6)
    dates = pd.bdate_range("2016-01-01", "2025-06-30")
    tickers = [f"T{i:02d}" for i in range(60)]
    shape = (len(dates), len(tickers))
    late = pd.Series(dates >= pd.Timestamp(LATE[0]), index=dates)
    checks: dict[str, bool] = {}

    def frame(a):
        return pd.DataFrame(a, index=dates, columns=tickers)

    # NW: overlapping (5-day summed) noise must deflate the t relative to i.i.d.
    e = pd.Series(rng.normal(0.02, 1.0, 5000)).rolling(5).sum().dropna()
    checks["NW t < iid t on overlapping series"] = nw_tstat(e, 4) < 0.7 * nw_tstat(e, 0)

    # A1: f_decay predicts early only; f_stable throughout; f_noise never.
    f_decay, f_stable, f_noise = (frame(rng.normal(size=shape)) for _ in range(3))
    noise = frame(rng.normal(size=shape))
    realized = noise + 0.3 * f_stable + 0.3 * f_decay.where(~late, 0.0, axis=0)
    feats = {"f_decay": f_decay, "f_stable": f_stable, "f_noise": f_noise}
    _, p, daily = per_feature_ic(feats, realized, 0)
    cls = classify_features(p)
    checks["A1 flags the planted decay"] = bool(cls.loc["f_decay", "decayed"])
    checks["A1 keeps the stable feature"] = bool(cls.loc["f_stable", "skill"]
                                                 and not cls.loc["f_stable", "decayed"])
    checks["A1 noise is not skill-carrying"] = not bool(cls.loc["f_noise", "skill"])
    comp = fixed_sign_composite(feats, cls.loc[cls["skill"], "sign"])
    checks["A1 composite carries skill"] = bool(
        _spearman_by_row(comp, realized).mean() > 0.1)

    # A2: a feature whose sign depends on a state; terciles from an AR(1) state.
    st = pd.Series(np.cumsum(rng.normal(size=len(dates))) * 0.1, index=dates)
    terc = expanding_terciles(st)
    g = frame(rng.normal(size=shape))
    sgn = np.where(st > st.expanding(STATE_MIN_PERIODS).median(), 1.0, -1.0)
    real2 = noise + 0.3 * g.mul(sgn, axis=0)
    sd = state_difference(_spearman_by_row(g, real2), terc, 0)
    checks["A2 finds the state-conditional flip"] = abs(sd["t"]) >= 3
    checks["A2 terciles are causal (no NaN leak past burn-in)"] = bool(
        terc.iloc[:STATE_MIN_PERIODS - 1].isna().all() and terc.iloc[-100:].notna().all())

    # A3: the sign of f0's effect alternates by year — a walk-forward model is always
    # trained on the wrong regime; the within-year ceiling sees the right one.
    ysign = pd.Series(np.where(dates.year % 2 == 0, 1.0, -1.0), index=dates)
    f0, f1 = frame(rng.normal(size=shape)), frame(rng.normal(size=shape))
    tgt = noise * 0.5 + f0.mul(ysign, axis=0)
    fdf = pd.concat({"f0": f0.stack(), "f1": f1.stack(), "y": tgt.stack()}, axis=1)
    fdf.index.names = ["date", "ticker"]
    wf = ridge_walk_forward(fdf, ["f0", "f1"], "y", 504, 63)
    oos = _spearman_by_row(wf, tgt.reindex(wf.index)).mean()
    ce = within_year_ceiling(fdf, ["f0", "f1"], "y", 5, years=[2019, 2020, 2021])
    ceil_ic = _spearman_by_row(ce, tgt.reindex(ce.index)).mean()
    # (the stale model learns last year's sign, so its OOS IC is typically negative)
    checks["A3 ceiling >> stale walk-forward"] = bool(ceil_ic > 0.5 and oos < ceil_ic - 0.5)
    folds = walk_forward_folds(np.unique(dates.values), 504, 63)
    checks["A3 folds tile the OOS span with no overlap"] = bool(
        folds[0][1][0] == dates[504] and all(
            folds[i][1][-1] < folds[i + 1][1][0] for i in range(len(folds) - 1)))

    # A4: old names carry IC 0.3, young names none; young share 10% early -> 80% late.
    young_cut = np.where(late.to_numpy(), 48, 6)               # first n names are young
    buckets = frame(np.where(np.arange(60)[None, :] < young_cut[:, None], "1-3y", "5y+")
                    .astype(object))
    sig = frame(rng.normal(size=shape))
    real4 = noise + 0.6 * sig.where(buckets == "5y+", 0.0)
    rw = reweight_decomposition(sig, real4, buckets)
    checks["A4 attributes a pure-mix drop to drift"] = bool(rw["drift_share"] > 0.8)
    real4b = noise + 0.6 * sig.where(~late, 0.0, axis=0)        # decay with no mix effect
    buckets_b = frame(np.broadcast_to(np.where(np.arange(60) < 30, "1-3y", "5y+"), shape)
                      .astype(object))
    rwb = reweight_decomposition(sig, real4b, buckets_b)
    checks["A4 attributes a uniform decay to no drift"] = bool(abs(rwb["drift_share"]) < 0.1)
    fs = pd.Series([dates[0]] * 30 + [dates[400]] * 30, index=tickers)
    ab = age_buckets(fs, dates, pd.Index(tickers), dates[0])
    checks["A4 censored names are 5y+, a new listing ages"] = bool(
        (ab.iloc[:, 0] == "5y+").all() and ab.iloc[500, 59] == "1-3y"
        and ab.iloc[-1, 59] == "5y+")

    # A5: an unadjusted 1:2 split in source a only, in 2023.
    px = frame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, shape), axis=0)))
    pa = px.copy()
    sd_ = pd.Timestamp("2023-06-01")
    pa.loc[sd_:, "T07"] = pa.loc[sd_:, "T07"] / 2
    ra, rb = pa.pct_change(fill_method=None), px.pct_change(fill_method=None)
    dis = disagreement_by_year(ra, rb)
    checks["A5 disagreement lands in the split year only"] = bool(
        dis.loc[2023, "disagree"] == 1 and dis.drop(2023)["disagree"].sum() == 0)
    uni = frame(True)
    led_none = frame(False)
    led_one = led_none.copy()
    led_one.loc[sd_, "T07"] = True
    checks["A5 flags an unledgered split"] = bool(
        unledgered_extremes(ra, uni, led_none).sum().sum() == 1)
    checks["A5 a ledgered split is not flagged"] = bool(
        unledgered_extremes(ra, uni, led_one).sum().sum() == 0)
    ex = unledgered_extremes(ra, uni, led_none)
    excl = label_exclusion(ex, 5)
    pos = dates.get_loc(sd_)
    checks["A5 exclusion covers exactly the 5 spanning labels"] = bool(
        excl["T07"].sum() == 5 and excl["T07"].iloc[pos - 5:pos].all())

    # Rules: a synthetic result set where only H4 should fire.
    fake = {
        "classify": pd.DataFrame({"skill": [True, True], "decayed": [False, False]}),
        "ens_period_5b": {"early": {"mean_IC": 0.05}, "late": {"mean_IC": 0.04},
                          "recent": {"mean_IC": 0.04}},
        "ceiling_period": {"early": {"mean_IC": 0.10}, "recent": {"mean_IC": 0.09}},
        "ridge_period": {"early": {"mean_IC": 0.04}, "late": {"mean_IC": 0.03}},
        "sign_flips": pd.DataFrame(columns=["feature", "year", "mean_IC", "t"]),
        "state_diff": pd.DataFrame({"t": [1.0]}, index=pd.MultiIndex.from_tuples(
            [("f", "zvol")], names=["feature", "state"])),
        "state_mix": {"zvol": pd.DataFrame({1: [.33, .33], 2: [.33, .33], 3: [.34, .34]},
                                           index=["early", "late"])},
        "reweight": {"drift_share": 0.7},
        "h5": {"disagree_ratio": 1.0, "extreme_ratio": 1.1, "ic_lift": 0.02},
    }
    fired = evaluate_rules(fake)["fired"]
    checks["rules: only the planted H4 fires"] = bool(
        fired["H4"] == True and not any(fired[k] == True for k in ("H1", "H2", "H3", "H5")))  # noqa: E712

    print("\n" + "=" * 78)
    for name, ok in checks.items():
        print(f"  {name:<58}: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    assert all(checks.values()), "phase6_decay dry-run FAILED"


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 6 Part A decay diagnostics (read-only).")
    ap.add_argument("--run", action="store_true", help="run against real data (else dry-run)")
    ap.add_argument("--sections", default="A1,A1b,A2,A3,A4,A5")
    ap.add_argument("--no-fits", action="store_true",
                    help="skip new walk-forward fits (A1b learners, A3c window probe)")
    args = ap.parse_args()
    if not args.run:
        _dry_run()
        return
    run(set(args.sections.split(",")), new_fits=not args.no_fits)


if __name__ == "__main__":
    main()
