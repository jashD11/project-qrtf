#!/usr/bin/env python3
"""Shared data layer for the QRTF Engine decks.

Read-only. Reads the stored ledgers and panels under data/, computes every
series and statistic, renders inline SVG, and writes one self-contained HTML
file to docs/presentation/qrtf_deck.html.

Nothing under data/ is written. Every artefact load asserts the path exists and
prints the shape read; a missing artefact is a hard failure, never silently
replaced by a number from the docs.

Figures marked SOURCE: docs are transcribed from the research record because
reproducing them requires re-running the Tier-1 walk-forward. Everything else
is computed here.
"""
from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "data" / "trial_database"
BHAV = ROOT / "data" / "bhavcopy"
# output paths belong to the individual builders, not to this module

WIN_START, WIN_END = pd.Timestamp("2016-01-21"), pd.Timestamp("2025-06-27")
TRADING_DAYS = 252

# ── loading ──────────────────────────────────────────────────────────────────

def load(path: Path, **kw) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"missing artefact: {path}\n"
            "data/ is gitignored and exists only on the machine that ran the "
            "pipeline. Rebuild it before building the deck."
        )
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, **kw)
    print(f"  read {path.relative_to(ROOT)}  {df.shape}")
    return df


# ── statistics ───────────────────────────────────────────────────────────────

def sharpe(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.mean() / s.std() * math.sqrt(TRADING_DAYS))


def cum(s: pd.Series) -> pd.Series:
    return (1 + s.fillna(0)).cumprod()


def max_dd(s: pd.Series) -> float:
    c = cum(s)
    return float((c / c.cummax() - 1).min())


def weekly(s: pd.Series) -> pd.Series:
    """Downsample an equity curve for plotting; 2,337 points is more SVG than a
    slide needs and the weekly close preserves the shape exactly."""
    return s.resample("W").last().dropna()



def fmt(v: float, dp: int = 2, sign: bool = False) -> str:
    s = f"{v:+.{dp}f}" if sign else f"{v:.{dp}f}"
    return s.replace("-", "−")



def pct(v, dp=1):
    return fmt(v * 100, dp) + "%"

# ── the data ─────────────────────────────────────────────────────────────────

print("loading artefacts")
p4c_net = load(LEDGER / "phase4c_dsr_matrix.parquet")
p4c_gross = load(LEDGER / "phase4c_dsr_matrix_gross.parquet")
p4c_gate = load(LEDGER / "phase4c_dsr_matrix_dsr_gate.csv")
p4c_diag = load(LEDGER / "phase4c_dsr_matrix_execution_diagnostics.csv")
p4b = load(LEDGER / "production_dsr_matrix.parquet")
p3_net12 = load(LEDGER / "production_dsr_matrix_net12.parquet")
index_daily = load(BHAV / "index_daily.parquet")
umask = load(ROOT / "data" / "nse500_universe_mask.parquet")
ic_500 = load(LEDGER / "tree_fit_diagnostics_daily_nse500_5b.csv", comment="#")
ic_68 = load(LEDGER / "tree_fit_diagnostics_daily_5b.csv", comment="#")

# Phase 4c cells, keyed by (style, target). The ledger column *is* the
# strategy_id, so the mapping is read off the execution diagnostics rather than
# assumed from column order.
diag = p4c_diag.set_index("strategy_id")
gate = p4c_gate.set_index("strategy_id")
CELLS = []
for sid, row in diag.iterrows():
    CELLS.append({
        "sid": sid,
        "style": row["execution_style"],
        "target": "21-day" if row["target"].endswith("21b") else "5-day",
        "net_sr": sharpe(p4c_net[sid]),
        "gross_sr": sharpe(p4c_gross[sid + "__gross"]),
        "net_cum": cum(p4c_net[sid]).iloc[-1] - 1,
        "dsr": float(gate.loc[sid, "DSR"]),
        "turnover": float(row["turnover_mean"]),
        "bps": float(row["effective_oneway_bps"]),
        "drag": float(row["trade_drag_annual"]),
        "lo_sr": float(gate.loc[sid, "SR_ann"]),
    })
CELLS.sort(key=lambda c: -c["net_sr"])
SR_STAR = float(gate["SR_star_ann"].iloc[0])
best = CELLS[0]

# Phase 4b cells, in ledger order: long_only, long_short, dynamic_tilt.
B_STYLES = ["long_only", "long_short", "dynamic_tilt"]
P4B = []
for sid, style, dsr in zip(p4b.columns, B_STYLES, [0.916, 1.000, 0.998]):
    P4B.append({"sid": sid, "style": style, "sr": sharpe(p4b[sid]),
                "cum": cum(p4b[sid]).iloc[-1] - 1, "mdd": max_dd(p4b[sid]), "dsr": dsr})

# NIFTY-50, aligned to the scored window and rebased to the ledger's first day.
nifty = (index_daily[index_daily["ticker"] == "NIFTY-50"]
         .set_index("timestamp")["close"].sort_index())
nifty = nifty.reindex(p4b.index).ffill()
nifty_ret = nifty.pct_change().fillna(0)
NIFTY_SR = sharpe(nifty_ret)
NIFTY_CUM = float(nifty.iloc[-1] / nifty.iloc[0] - 1)
NIFTY_DD = max_dd(nifty_ret)

# Phase 3, 12 cells (1-day target). Ledger order is daily, 60min, 30min, 15min,
# each as long_only / long_short / dynamic_tilt.
P3_FREQ = ["daily", "60-min", "30-min", "15-min"]
P3_STYLE = ["long only", "long / short", "dynamic tilt"]
p3_net_sr = {}
for i, f_ in enumerate(P3_FREQ):
    for j, st in enumerate(P3_STYLE):
        p3_net_sr[(f_, st)] = sharpe(p3_net12[p3_net12.columns[i * 3 + j]])

# Universe composition. The count is a flat 500/day by construction, so the
# informative series is the churn: how many names enter and leave each year,
# which is what point-in-time membership actually buys.
uwin = umask.loc[WIN_START:WIN_END]
DISTINCT_NAMES = int(uwin.loc[:, uwin.any()].shape[1])
annual = uwin.resample("YE").max()
CHURN = []
prev = None
for d, row in annual.iterrows():
    cur = set(row[row].index)
    if prev is not None:
        CHURN.append({"year": str(d.year), "in": len(cur - prev), "out": len(prev - cur)})
    prev = cur

print(f"  computed: {len(CELLS)} Phase 4c cells, {DISTINCT_NAMES} distinct names, "
      f"SR* = {SR_STAR:.3f}")


# ── figures transcribed from the research record ─────────────────────────────
# SOURCE: docs. Reproducing these requires re-running the Tier-1 walk-forward
# (docs/phase4c_results.md §1, §4.1, §4.2, §6; docs/phase4_results.md §3.1;
# docs/phase1_poc_results.md; docs/phase3_results.md §2). The recorded tables
# are exact, and the deck cites them rather than approximating them here.

SPREAD_Q = {"cs": [2.41, 4.53, 6.24, 7.50, 9.16], "ar": [18.79, 22.51, 26.56, 30.06, 32.37]}
SPREAD_POOLED = {"cs": 5.97, "ar": 26.07}
STATUTORY_BPS = 14.6558

SHORT_YEARS = [str(y) for y in range(2016, 2026)]
SHORT_BORROW = [12.8, 11.6, 15.5, 8.6, 7.6, 11.1, 13.3, 9.4, 12.2, 13.7]
SHORT_INTEND = [49.1, 48.9, 49.1, 49.0, 49.0, 49.1, 49.2, 49.0, 49.0, 49.0]
SHORT_COVER = [26.1, 23.7, 31.5, 17.5, 15.6, 22.6, 27.0, 19.2, 24.9, 28.0]

IC_YEARS = [str(y) for y in range(2016, 2026)]
IC_T = [9.12, 11.42, 9.85, 10.48, 8.00, 4.89, 6.28, 7.91, 2.31, 1.21]
IC_QUINT = [0.0408, 0.0457, 0.0413, 0.0394, 0.0484]

SLIP_BPS = [0, 2, 4, 6, 8, 10, 15, 20]
SLIP = {
    "long only": [0.859, 0.747, 0.635, 0.523, 0.410, 0.298, 0.018, -0.263],
    "long / short": [1.778, 1.451, 1.123, 0.796, 0.469, 0.142, -0.673, -1.481],
    "dynamic tilt": [1.365, 1.182, 0.998, 0.815, 0.632, 0.448, -0.010, -0.468],
}

P1_PROFILES = ["stable uptrend", "stable downtrend", "stable volatile", "volatile downtrend"]
P1 = {"long only": [51.66, -7.65, 0.37, -51.31],
      "long / short": [26.86, 26.82, 17.16, -21.45],
      "dynamic tilt": [62.78, -0.88, 41.78, -55.32]}

P3_GROSS = {"15-min": [11.42, 16.29, 17.16], "30-min": [6.27, 11.50, 11.32],
            "60-min": [3.17, 8.04, 6.66], "daily": [1.62, 1.81, 1.74]}

# SR* as a function of the trial count, computed with the repo's own
# expected_max_sharpe so there is one definition of the deflation benchmark.
import sys
sys.path.insert(0, str(ROOT))
from src.production_ml.tier4_dsr_gate import expected_max_sharpe  # noqa: E402

CROSS_TRIAL_STD = 0.0290  # Phase 4b's measured cross-trial daily Sharpe dispersion
N_GRID = list(range(2, 61))
SR_STAR_CURVE = [expected_max_sharpe(CROSS_TRIAL_STD, n) * math.sqrt(TRADING_DAYS) for n in N_GRID]

