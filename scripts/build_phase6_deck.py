#!/usr/bin/env python3
"""Build the Phase 6 deck: why the edge faded, and what new data could not fix.

Read-only. Reads the Phase 6 ledgers under data/trial_database/phase6/ (the 18-cell
grid, Part A's decay tables and the Stage 7 diagnostics) plus the Phase 5 ledger for
the recap, and writes one self-contained HTML file to docs/presentation/phase6_deck.html.
Chart primitives, CSS and the page shell come from build_deck.py, so the three decks
look and behave alike.

Story first: the strategy and the attempts to improve it, not the implementation.
Nothing dated after config.PHASE6_SEALED_AFTER is read; the holdout stays sealed.

Sharpe ratios are annualized and serial-correlation adjusted (Lo, via the overlapping
variance ratio), as everywhere in the research record, except where a slide says
"naive". Forecast skill is the mean daily cross-sectional rank IC.
"""
from __future__ import annotations

import io
import math
import re
import sys
from contextlib import redirect_stdout

import pandas as pd

import build_deck as bd  # noqa: E402 - also builds its own slide list in memory; writes nothing
from deck_data import LEDGER, ROOT, TRADING_DAYS, cum, fmt, load, max_dd, pct, weekly  # noqa: E402

sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from src.production_ml.tier4_dsr_gate import probabilistic_sharpe_ratio, run_dsr_gate  # noqa: E402

esc, Frame, svg_open, gridlines, nice_ticks, ref_marks = (
    bd.esc, bd.Frame, bd.svg_open, bd.gridlines, bd.nice_ticks, bd.ref_marks)
svg_line, svg_bars, svg_waterfall, stats, table = bd.svg_line, bd.svg_bars, bd.svg_waterfall, bd.stats, bd.table

OUT = ROOT / "docs" / "presentation" / "phase6_deck.html"
P6, P5 = LEDGER / "phase6", LEDGER / "phase5"
DECAY, DIAG = P6 / "decay", P6 / "stage7_diag" / "diag"
SEALED = pd.Timestamp(config.PHASE6_SEALED_AFTER)
N_TRIALS, N_TRIALS_P5 = 57, 42

# ── data: the Phase 6 grid ───────────────────────────────────────────────────

print("loading Phase 6 artefacts")
p6_net = load(P6 / "phase6_dsr_matrix.parquet")
p6_gross = load(P6 / "phase6_dsr_matrix_gross.parquet")
p6_diag = load(P6 / "phase6_dsr_matrix_execution_diagnostics.csv").set_index("strategy_id")
p6_gate = load(P6 / "phase6_dsr_matrix_dsr_gate.csv").set_index("strategy_id")
assert p6_net.index.max() <= SEALED, "ledger reaches into the sealed holdout"


def score_gross(path, n):
    """The adjusted gross Sharpe is not persisted; score in memory (persist=False
    writes nothing). Only SR_ann is read, and it does not depend on n."""
    with redirect_stdout(io.StringIO()):
        g = run_dsr_gate(str(path), n_trials=n, persist=False)
    return g.set_index("strategy_id") if "strategy_id" in g.columns else g


g6_gate = score_gross(P6 / "phase6_dsr_matrix_gross.parquet", N_TRIALS)


def cagr(s: pd.Series) -> float:
    return float(cum(s).iloc[-1] ** (TRADING_DAYS / len(s)) - 1)


STYLES = ["long_only", "long_short_slb", "dynamic_tilt_slb"]
STYLE_LAB = {"long_only": "long-only", "long_short_slb": "long/short", "dynamic_tilt_slb": "130/30"}
STYLE_ABBR = {"long_only": "LO", "long_short_slb": "LS", "dynamic_tilt_slb": "130/30"}
TARGETS = ["5b", "21b"]
TGT_LAB = {"5b": "5-day", "21b": "21-day"}
FS = ["p17", "p17x", "p17xo"]
FS_LAB = {"p17": "base (17 price signals)", "p17x": "+ price & index", "p17xo": "+ options & futures"}
FS_SHORT = {"p17": "base", "p17x": "+ price/index", "p17xo": "+ options"}
FS_CLS = {"p17": "c2", "p17x": "c4", "p17xo": "c3"}
ORDER = [(t, s) for t in TARGETS for s in STYLES]

R = {}
for sid, row in p6_diag.iterrows():
    fs = row["feature_set"] if isinstance(row["feature_set"], str) else "p17"
    tgt = "21b" if row["target"].endswith("21b") else "5b"
    net, gross = p6_net[sid], p6_gross[sid + "__gross"]
    R[(tgt, row["execution_style"], fs)] = {
        "sid": sid, "net": net, "gross": gross,
        "sr": float(p6_gate.loc[sid, "SR_ann"]), "dsr": float(p6_gate.loc[sid, "DSR"]),
        "gsr": float(g6_gate.loc[sid + "__gross", "SR_ann"]),
        "nsr": float(row["sharpe_net"]), "ngsr": float(row["sharpe_gross"]),
        "cagr": cagr(net), "dd": max_dd(net),
        "turn": float(row["turnover_mean"]), "drag": float(row["trade_drag_annual"]),
        "bps": float(row["effective_oneway_bps"]),
    }
assert len(R) == 18, f"expected 18 strategies, got {len(R)}"


def cell(t, s, f):
    return R[(t, s, f)]


BEST_KEY = max(R, key=lambda k: R[k]["dsr"])
BEST = R[BEST_KEY]
BEST_NAME = f"{STYLE_LAB[BEST_KEY[1]]}, {TGT_LAB[BEST_KEY[0]]} forecast"
N_PASS = sum(r["dsr"] > 0.95 for r in R.values())
BASE_WINS = sum(cell(t, s, "p17")["sr"] > max(cell(t, s, "p17x")["sr"], cell(t, s, "p17xo")["sr"])
                for t, s in ORDER)
D_F1 = {(t, s): cell(t, s, "p17x")["nsr"] - cell(t, s, "p17")["nsr"] for t, s in ORDER}
D_F2 = {(t, s): cell(t, s, "p17xo")["nsr"] - cell(t, s, "p17x")["nsr"] for t, s in ORDER}
MAX_DROP = max(cell(t, s, "p17")["sr"] - cell(t, s, "p17x")["sr"] for t, s in ORDER)


def needed_sr(g):
    """The annual Sharpe at which DSR = 0.95, solved from the gate's own PSR formula
    with this column's measured T, skew and kurtosis."""
    star_d = float(g["SR_star_ann"]) / math.sqrt(TRADING_DAYS)
    T, sk, ku = int(g["T"]), float(g["skew"]), float(g["kurt"])
    lo, hi = star_d, star_d * 4
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if probabilistic_sharpe_ratio(mid, T, sk, ku, star_d) < 0.95 else (lo, mid)
    return hi * math.sqrt(TRADING_DAYS)


_gb = p6_gate.loc[BEST["sid"]]
SR_STAR = float(_gb["SR_star_ann"])
BAR_SR = needed_sr(_gb)

# ── data: the Phase 5 recap ──────────────────────────────────────────────────

print("loading Phase 5 artefacts for the recap")
p5_net = load(P5 / "phase5_dsr_matrix.parquet")
p5_diag = load(P5 / "phase5_dsr_matrix_execution_diagnostics.csv").set_index("strategy_id")
p5_gate = load(P5 / "phase5_dsr_matrix_dsr_gate.csv").set_index("strategy_id")
g5_gate = score_gross(P5 / "phase5_dsr_matrix_gross.parquet", N_TRIALS_P5)
P5R = {}
for sid, row in p5_diag.iterrows():
    cons = row["construction"] if isinstance(row["construction"], str) else "buffer"
    tgt = "21b" if row["target"].endswith("21b") else "5b"
    P5R[(tgt, row["execution_style"], cons)] = {
        "sid": sid, "sr": float(p5_gate.loc[sid, "SR_ann"]),
        "gsr": float(g5_gate.loc[sid + "__gross", "SR_ann"]),
        "cagr": cagr(p5_net[sid]), "dd": max_dd(p5_net[sid]),
        "drag": float(row["trade_drag_annual"]), "turn": float(row["turnover_mean"]),
    }
P5_BUF = P5R[("5b", "dynamic_tilt_slb", "buffer")]
P5_M2 = P5R[("5b", "dynamic_tilt_slb", "cost_swap")]
P5_STAR = float(p5_gate.loc[P5_M2["sid"], "SR_star_ann"])
P5_BAR = needed_sr(p5_gate.loc[P5_M2["sid"]])

# Phase 4c long/short, one correction at a time. SOURCE: docs/phase4c_results.md,
# the same four values build_deck.py draws.
WATERFALL = [1.78, 1.35, -0.23, -0.99]

# ── data: Part A (why the edge faded) ────────────────────────────────────────

print("loading Part A decay tables")


def by_year(path, levels=2):
    d = load(path, header=list(range(levels)), index_col=0)
    d = d[pd.to_numeric(d.index, errors="coerce").notna()]
    d.index = d.index.astype(int)
    assert d.index.max() <= SEALED.year
    return d


def wmean(d, col, years):
    """Day-weighted mean IC over the given years. col is the column tuple prefix."""
    sub = d.loc[[y for y in years if y in d.index]]
    m, n = sub[col + ("mean_IC",)], sub[col + ("n_days",)]
    return float((m * n).sum() / n.sum())


YEARS = list(range(2016, 2026))
EARLY_Y, LATE_Y, RECENT_Y = list(range(2016, 2021)), list(range(2021, 2026)), [2024, 2025]

ens = by_year(DECAY / "ensemble_ic_by_year.csv")
ridge_ceil = by_year(DECAY / "a3_ridge_ceiling_by_year.csv")
comp = by_year(DECAY / "a1_composite_by_year.csv")
window = by_year(DECAY / "a3_window_probe.csv")
classify = load(DECAY / "a1_classify_5b.csv", index_col=0)

IC_ENS = [float(ens.loc[y, ("5b", "mean_IC")]) for y in YEARS]
EARLY_IC = wmean(ens, ("5b",), EARLY_Y)
LATE_IC = wmean(ens, ("5b",), LATE_Y)
RECENT_IC = wmean(ens, ("5b",), RECENT_Y)
COMP_LATE = wmean(comp, ("5b",), LATE_Y)
COMP_RECENT = wmean(comp, ("5b",), RECENT_Y)
WIN = {w: {"late": wmean(window, (w,), LATE_Y), "recent": wmean(window, (w,), RECENT_Y)}
       for w in ("252", "504", "1008")}
CEIL_RECENT = wmean(ridge_ceil, ("ceiling",), RECENT_Y)

skill = classify[classify["skill"]].copy()
skill["early_s"] = skill["early_mean_IC"] * skill["sign"]
skill["late_s"] = skill["late_mean_IC"] * skill["sign"]
N_SKILL, N_DECAYED = len(skill), int(skill["decayed"].sum())
FADE = 1 - float(skill.loc[skill["decayed"], "late_over_early"].mean())
FEAT_NAME = {
    "mom_logret_1b": "1-day reversal", "mom_logret_5b": "5-day reversal",
    "mom_close_sma20": "gap to 20-day average", "liq_turnover_20b": "share turnover",
    "mom_logret_252b": "1-year momentum", "mom_max_dret_20b": "biggest daily jump (lottery)",
    "vol_realized_20b": "1-month volatility", "vol_realized_60b": "3-month volatility",
    "vol_parkinson_20b": "high–low volatility", "vol_drawdown_252b": "distance from 1-year high",
    "liq_amihud": "illiquidity (Amihud)",
}

# ── data: Stage 7 diagnostics (why the new data hurt) ────────────────────────

print("loading Stage 7 diagnostics")
ic6 = by_year(DIAG / "t3_ensemble_ic_by_year.csv", levels=3)
feat_ic = load(DIAG / "t2_feature_ic_by_period_5b.csv", index_col=0)
sim = load(DIAG / "t3_alpha_similarity_vs_p17.csv", index_col=0)
assert sim.index.max() <= SEALED.year


def fit_ic(tgt, fs):
    suffix = "" if fs == "p17" else f"_{fs}"
    d = load(LEDGER / f"tree_fit_diagnostics_daily_nse500_{tgt}{suffix}_purge.csv", comment="#")
    return d.set_index("model")


FIT = {(t, f): fit_ic(t, f) for t in TARGETS for f in FS}
IC_FS = {(t, f): float(FIT[(t, f)].loc["ensemble", "mean_IC"]) for t in TARGETS for f in FS}
RF_FS = {f: float(FIT[("5b", f)].loc["rf", "mean_IC"]) for f in FS}
# mean of the yearly means, as in docs/phase6_stage7_diagnostics.md
IC6_LATE = {f: float(ic6.loc[LATE_Y, ("5b", f, "mean_IC")].mean()) for f in ("p17", "p17x")}
SIM_LO, SIM_HI = float(sim.min().min()), float(sim.max().max())

NEW_FEATS = [("fut_basis", "futures|basis"), ("opt_iv_rv", "implied −|realised vol"),
             ("fut_oi_chg_5b", "futures open|interest chg"), ("ivol_63b", "stock-specific|vol"),
             ("beta_252b", "market|beta"), ("opt_atm_iv", "implied|vol")]
FEAT_SIGN = {k: (1 if feat_ic.loc[k, "late_mean_IC"] > 0 else -1) for k, _ in NEW_FEATS}

# 21-day forecast, traded daily (Phase 5, M2) vs rebalanced monthly (Phase 6, base)
CAD = {s: {"daily": P5R[("21b", s, "cost_swap")], "monthly": cell("21b", s, "p17")} for s in STYLES}

print(f"  best {BEST_NAME}: SR {BEST['sr']:.3f} DSR {BEST['dsr']:.3f}; SR* {SR_STAR:.3f}; needed {BAR_SR:.2f}")

# ── chart helpers ────────────────────────────────────────────────────────────


def ml_ticks(svg):
    """Split category tick labels on '|' into stacked lines (svg_bars draws one line)."""
    def sub(m):
        x, y, txt = m.group(1), float(m.group(2)), m.group(3)
        return "".join(f'<text class="tick" x="{x}" y="{y+i*13:.1f}" text-anchor="middle">{p}</text>'
                       for i, p in enumerate(txt.split("|")))
    return re.sub(r'<text class="tick" x="([\d.]+)" y="([\d.]+)" text-anchor="middle">([^<]*\|[^<]*)</text>', sub, svg)


def declutter(svg, gap=12):
    """Push svg_line's end-of-series labels apart vertically so they never overlap."""
    pat = re.compile(r'<text class="endlab (\w+)" x="([\d.]+)" y="([\d.]+)">')
    ys, new = [], {}
    for y, tag in sorted((float(m.group(3)), m.group(0)) for m in pat.finditer(svg)):
        y2 = max(y, ys[-1] + gap) if ys else y
        ys.append(y2)
        new[tag] = tag.replace(f'y="{m_y(tag)}"', f'y="{y2:.1f}"')
    for old, nt in new.items():
        svg = svg.replace(old, nt)
    return svg


def m_y(tag):
    return re.search(r'y="([\d.]+)"', tag).group(1)


def pts(*items):
    return '<ul class="pts">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def year_ticks(idx):
    return [(i, str(d.year)) for i, d in enumerate(idx)
            if d.year in (2016, 2018, 2020, 2022, 2024) and (i == 0 or idx[i - 1].year != d.year)]


def shade_line(svg, w, h, n, x0, x1, label):
    """Shade an x-range of an svg_line chart (whose frame margins are 62/118/16/34)
    under the lines, with a label at the top."""
    l, pw, t, ph = 62, w - 62 - 118, 16, h - 16 - 34
    X = lambda v: l + v / (n - 1) * pw  # noqa: E731
    rect = (f'<rect class="shade" x="{X(x0):.1f}" y="{t}" width="{X(x1)-X(x0):.1f}" height="{ph}"/>'
            f'<text class="shadelab" x="{X(x0)+8:.1f}" y="{t+14}">{esc(label)}</text>')
    cut = svg.index(">") + 1
    return svg[:cut] + rect + svg[cut:]


def bars_colored(labels, values, classes, *, w=620, h=320, ytitle=None, value_fmt=fmt,
                 segs=None, ref=None, tips=None, barw=None):
    """One bar per label with its own colour class. segs: [{from, to, y, label}] draws
    a horizontal average line across a range of bars."""
    vals = list(values) + [s["y"] for s in segs or []] + [r["y"] for r in ref or []]
    lo, hi = min(vals + [0]), max(vals + [0])
    pad = (hi - lo) * 0.12 or 1
    f = Frame(w, h, 62, 104, 22, 46, (0, len(labels)), (lo - pad * 0.3, hi + pad))
    out = [svg_open(w, h), gridlines(f, nice_ticks(lo, hi, 5), lambda v: fmt(v, 2)), ref_marks(f, ref)]
    gw = f.pw / len(labels)
    bw = barw or min(46, gw - 12)
    zero = f.Y(0)
    for i, (lab, v, cls) in enumerate(zip(labels, values, classes)):
        cx = f.l + gw * (i + 0.5)
        y = f.Y(v)
        tip = tips[i] if tips else f"{lab}: {value_fmt(v)}"
        out.append(f'<rect class="bar {cls}" x="{cx-bw/2:.1f}" y="{min(y, zero):.1f}" width="{bw:.1f}" '
                   f'height="{max(abs(y-zero), 1):.1f}" rx="2.5" data-tip="{esc(tip)}"/>')
        vy = y - 6 if v >= 0 else y + 14
        out.append(f'<text class="barval" x="{cx:.1f}" y="{vy:.1f}" text-anchor="middle">{value_fmt(v)}</text>')
        for li, part in enumerate(lab.split("|")):
            out.append(f'<text class="tick" x="{cx:.1f}" y="{f.t+f.ph+18+li*13:.1f}" text-anchor="middle">{esc(part)}</text>')
    for s in segs or []:
        x0, x1 = f.l + gw * s["from"] + 4, f.l + gw * (s["to"] + 1) - 4
        y = f.Y(s["y"])
        out.append(f'<line class="seg {s.get("cls", "")}" x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="seglab {s.get("cls", "")}" x="{x1+4:.1f}" y="{y-5:.1f}">{esc(s["label"])}</text>')
    if ytitle:
        out.append(f'<text class="axtitle" x="{f.l}" y="12">{esc(ytitle)}</text>')
    out.append("</svg>")
    return "".join(out)


def svg_shift3(rows, *, w=940, h=350, ytitle=None, ref=None, vfmt=fmt):
    """One column per strategy with one dot per feature set, joined by a bar from
    the lowest to the highest. rows: [{label, sub, p17, p17x, p17xo}]."""
    vals = [r[k] for r in rows for k in FS] + [x["y"] for x in ref or []]
    lo, hi = min(vals + [0]), max(vals + [0])
    pad = (hi - lo) * 0.1
    f = Frame(w, h, 62, 104, 26, 66, (0, len(rows)), (lo - pad, hi + pad))
    out = [svg_open(w, h), gridlines(f, nice_ticks(lo, hi, 5), lambda v: fmt(v, 1)), ref_marks(f, ref)]
    gw = f.pw / len(rows)
    for i, r in enumerate(rows):
        cx = f.l + gw * (i + 0.5)
        ys = {k: f.Y(r[k]) for k in FS}
        out.append(f'<line class="dbell" x1="{cx:.1f}" y1="{min(ys.values()):.1f}" x2="{cx:.1f}" y2="{max(ys.values()):.1f}"/>')
        # labels to the right, pushed apart so they never overlap
        placed = []
        for k in sorted(FS, key=lambda k: ys[k]):
            ly = ys[k] + 4
            if placed and ly - placed[-1] < 12:
                ly = placed[-1] + 12
            placed.append(ly)
            out.append(f'<text class="barval {FS_CLS[k]}" x="{cx+11:.1f}" y="{ly:.1f}">{vfmt(r[k])}</text>')
        for k in FS:
            out.append(f'<circle class="dot {FS_CLS[k]}" cx="{cx:.1f}" cy="{ys[k]:.1f}" r="5.5" '
                       f'data-tip="{esc(f"{r['label']} {r['sub']} · {FS_LAB[k]}: {vfmt(r[k])}")}"/>')
        out.append(f'<text class="tick" x="{cx:.1f}" y="{f.t+f.ph+20:.1f}" text-anchor="middle">{esc(r["label"])}</text>')
    for g in ({"from": 0, "to": 2, "label": "5-DAY FORECAST"}, {"from": 3, "to": 5, "label": "21-DAY FORECAST · MONTHLY"}):
        x0, x1 = f.l + gw * g["from"], f.l + gw * (g["to"] + 1)
        out.append(f'<line class="grpline" x1="{x0+6:.1f}" y1="{f.t+f.ph+34:.1f}" x2="{x1-6:.1f}" y2="{f.t+f.ph+34:.1f}"/>')
        out.append(f'<text class="grplab" x="{(x0+x1)/2:.1f}" y="{f.t+f.ph+50:.1f}" text-anchor="middle">{g["label"]}</text>')
    if ytitle:
        out.append(f'<text class="axtitle" x="{f.l}" y="14">{esc(ytitle)}</text>')
    out.append("</svg>")
    return "".join(out)


# ── hand-drawn diagrams ──────────────────────────────────────────────────────

def diagram_phases():
    """The program as a timeline: one node per phase, a lesson, and the best
    after-cost Sharpe it reached, coloured by whether it passed the gate."""
    w, h = 940, 250
    nodes = [
        ("PHASE 1", "Simulated market", "the machinery works|on made-up prices", "test rig", "cn"),
        ("PHASES 2–3", "Real prices,|real taxes", "fast trading dies;|daily long-only lives", "0.57 · fail", "no"),
        ("PHASE 4b", "500 stocks", "passes the luck test…", "1.78 · pass*", "warn"),
        ("PHASE 4c", "Real spreads,|real shorts", "…until trading costs|are measured", "0.30 · fail", "no"),
        ("PHASE 5", "Swap only|when it pays", "costs halved;|forecast too weak", "0.96 · fail", "no"),
        ("PHASE 6", "Why it fades;|new data", "this deck", "?", "now"),
    ]
    o = [svg_open(w, h)]
    x0, step, ly = 82, 155, 64
    o.append(f'<line class="tl-line" x1="{x0-50}" y1="{ly}" x2="{x0+step*5+50}" y2="{ly}"/>')
    for i, (ph, title, lesson, chip, cls) in enumerate(nodes):
        x = x0 + i * step
        o.append(f'<text class="tl-ph" x="{x}" y="{ly-22}" text-anchor="middle">{ph}</text>')
        o.append(f'<circle class="tl-node {cls}" cx="{x}" cy="{ly}" r="8"/>')
        for j, part in enumerate(title.split("|")):
            o.append(f'<text class="tl-t" x="{x}" y="{ly+32+j*16}" text-anchor="middle">{esc(part)}</text>')
        for j, part in enumerate(lesson.split("|")):
            o.append(f'<text class="tl-s" x="{x}" y="{ly+82+j*14}" text-anchor="middle">{esc(part)}</text>')
        o.append(f'<rect class="chip {cls}" x="{x-50}" y="{ly+122}" width="100" height="24" rx="12"/>')
        o.append(f'<text class="chip-t {cls}" x="{x}" y="{ly+138}" text-anchor="middle">{esc(chip)}</text>')
    o.append(f'<text class="tl-s" x="{x0-50}" y="{h-6}">best Sharpe after costs in each phase · '
             '*overturned once real costs and borrowable shorts were measured</text>')
    o.append("</svg>")
    return "".join(o)


def lock(x, y, s=1.0):
    return (f'<g transform="translate({x},{y}) scale({s})">'
            '<path class="lock-arc" d="M-7,0 v-6 a7,7 0 0 1 14,0 v6"/>'
            '<rect class="lock-body" x="-11" y="0" width="22" height="17" rx="3"/></g>')


def diagram_data_timeline():
    """2013 to today: training-only years, scored years (strong, fading), and the
    sealed holdout that has never been read."""
    w, h = 940, 150
    a, b = 2013.0, 2026.8
    X = lambda v: 30 + (v - a) / (b - a) * (w - 60)  # noqa: E731
    segs = [(2013.0, 2016.05, "seg-train", "training only"),
            (2016.05, 2021.0, "seg-strong", "scored · skill strong"),
            (2021.0, 2025.5, "seg-fade", "scored · skill fading"),
            (2025.5, 2026.8, "seg-sealed", "")]
    o = [svg_open(w, h),
         '<defs><pattern id="hatch6" width="8" height="8" patternUnits="userSpaceOnUse" '
         'patternTransform="rotate(45)"><line class="hatch" x1="0" y1="0" x2="0" y2="8"/></pattern></defs>']
    y0, bh = 40, 46
    for s0, s1, cls, lab in segs:
        o.append(f'<rect class="{cls}" x="{X(s0)+1:.1f}" y="{y0}" width="{X(s1)-X(s0)-2:.1f}" height="{bh}" rx="4"/>')
        if lab:
            o.append(f'<text class="seg-t" x="{(X(s0)+X(s1))/2:.1f}" y="{y0+28}" text-anchor="middle">{lab}</text>')
    sx = (X(2025.5) + X(2026.8)) / 2
    o.append(lock(sx, y0 + 12, 0.95))
    o.append(f'<text class="seal-t" x="{sx:.1f}" y="{y0-10}" text-anchor="middle">SEALED HOLDOUT</text>')
    for yr in (2013, 2016, 2019, 2021, 2023, 2025):
        o.append(f'<text class="tick" x="{X(yr):.1f}" y="{y0+bh+18}" text-anchor="middle">{yr}</text>')
    o.append(f'<text class="tick" x="{X(2025.5):.1f}" y="{y0+bh+34}" text-anchor="middle">30 Jun 2025</text>')
    o.append(f'<text class="tick" x="{X(2026.8):.1f}" y="{y0+bh+18}" text-anchor="end">today</text>')
    o.append(f'<line class="ref vert" x1="{X(2025.5):.1f}" y1="{y0-4}" x2="{X(2025.5):.1f}" y2="{y0+bh+22}"/>')
    o.append("</svg>")
    return "".join(o)


def diagram_feature_fade():
    """Each skill feature's strength (IC in its own direction), 2016–20 → 2021–25."""
    rows = skill.sort_values("late_over_early")
    n = len(rows)
    w, h = 940, 36 + n * 27 + 40
    f = Frame(w, h, 250, 120, 30, 40, (0, 0.07), (0, n))
    X = f.X
    o = [svg_open(w, h)]
    for v in (0, 0.02, 0.04, 0.06):
        o.append(f'<line class="grid" x1="{X(v):.1f}" y1="{f.t-6}" x2="{X(v):.1f}" y2="{f.t+n*27}"/>')
        o.append(f'<text class="tick" x="{X(v):.1f}" y="{f.t+n*27+16}" text-anchor="middle">{v:.2f}</text>')
    for i, (k, r) in enumerate(rows.iterrows()):
        y = f.t + i * 27 + 10
        cls = "c3" if r["decayed"] else "c2"
        e, l_ = r["early_s"], r["late_s"]
        o.append(f'<text class="flab" x="{f.l-14}" y="{y+4}" text-anchor="end">{esc(FEAT_NAME[k])}</text>')
        o.append(f'<line class="fade-ln {cls}" x1="{X(e):.1f}" y1="{y}" x2="{X(l_):.1f}" y2="{y}"/>')
        o.append(f'<circle class="dot cn" cx="{X(e):.1f}" cy="{y}" r="5" '
                 f'data-tip="{esc(f"{FEAT_NAME[k]} · 2016–20: {e:.3f} (t {abs(r['early_t']):.1f})")}"/>')
        o.append(f'<circle class="dot {cls}" cx="{X(l_):.1f}" cy="{y}" r="5.5" '
                 f'data-tip="{esc(f"{FEAT_NAME[k]} · 2021–25: {l_:.3f} (t {abs(r['late_t']):.1f})")}"/>')
        o.append(f'<text class="barval {cls}" x="{f.l+f.pw+12:.1f}" y="{y+4}">{r["late_over_early"]*100:.0f}% kept</text>')
    o.append(f'<text class="axtitle" x="{f.l}" y="16">forecast strength of each signal on its own (rank correlation, in its useful direction)</text>')
    o.append(f'<text class="axtitle" x="{f.l}" y="{h-6}">grey dot = 2016–20 · coloured dot = 2021–25</text>')
    o.append("</svg>")
    return "".join(o)


def diagram_purge():
    """The label-overlap leak: the last training days' 5-day answers reach into the
    test period. The fix removes those days."""
    w, h = 940, 250
    o = [svg_open(w, h)]
    edge, day = 600, 15
    for row, (y, title, fixed) in enumerate(((70, "BEFORE", False), (185, "AFTER THE FIX", True))):
        o.append(f'<text class="dg-t" x="20" y="{y-36}">{title}</text>')
        o.append(f'<rect class="wf-train" x="70" y="{y}" width="{edge-70-(5*day if fixed else 0)}" height="22" rx="3"/>')
        if fixed:
            o.append(f'<rect class="purged" x="{edge-5*day+1}" y="{y}" width="{5*day-2}" height="22" rx="3"/>')
            o.append(f'<text class="dg-s" x="{edge-2.5*day}" y="{y+40}" text-anchor="middle">5 days dropped</text>')
        o.append(f'<rect class="wf-pred" x="{edge+4}" y="{y}" width="{900-edge-4}" height="22" rx="3"/>')
        o.append(f'<text class="wf-key-in" x="80" y="{y+15}">training: 2 years</text>')
        o.append(f'<text class="wf-key-in" x="{edge+14}" y="{y+15}">test quarter (never seen)</text>')
        for i in range(10):
            x = edge - (10 - i) * day + day / 2
            if fixed and x > edge - 5 * day:
                continue
            x2 = x + 5 * day
            leak = x2 > edge
            cls = "arc-leak" if leak else "arc-ok"
            o.append(f'<path class="{cls}" d="M{x:.1f},{y} Q{x+2.5*day:.1f},{y-34} {x2:.1f},{y}"/>')
            o.append(f'<circle class="{cls}-d" cx="{x2:.1f}" cy="{y}" r="2.6"/>')
    o.append('<text class="dg-s c3t" x="700" y="30">red: the last training days\' 5-day</text>')
    o.append('<text class="dg-s c3t" x="700" y="44">answers land inside the test quarter</text>')
    o.append("</svg>")
    return "".join(o)


def diagram_nest():
    """Three nested feature sets. Each contains the one before."""
    w, h = 940, 300
    o = [svg_open(w, h)]
    boxes = [
        (10, 10, 920, 280, "nest3", "+ OPTIONS & FUTURES", "9 more · 40 in total", 650,
         ["implied volatility and skew", "implied minus realised vol", "options trading activity",
          "futures basis (futures vs spot)", "futures open-interest change"]),
        (30, 42, 600, 232, "nest2", "+ PRICE & INDEX", "14 more · 31 in total", 342,
         ["market beta, stock-specific vol", "overnight vs intraday returns", "trade size, bid-ask spread",
          "sector momentum and valuation", "market mood: VIX, market P/E", "and market return"]),
        (50, 74, 270, 184, "nest1", "BASE", "17 price signals", 66,
         ["momentum and reversal", "volatility", "liquidity and turnover", "", "(the Phase 5 model)"]),
    ]
    for x, y, bw, bh, cls, name, count, tx, lines in boxes:
        o.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{bw}" height="{bh}" rx="8"/>')
        o.append(f'<text class="nest-n {cls}" x="{tx}" y="{y+30}">{name}</text>')
        o.append(f'<text class="dg-s" x="{tx}" y="{y+48}">{count}</text>')
        for j, line in enumerate(lines):
            o.append(f'<text class="nest-l" x="{tx}" y="{y+76+j*20}">{esc(line)}</text>')
    o.append("</svg>")
    return "".join(o)


def diagram_holdout():
    """The sealed holdout can be used once. Two choices, two outcomes."""
    w, h = 940, 250
    o = [svg_open(w, h)]
    o.append('<rect class="seg-sealed2" x="20" y="78" width="230" height="96" rx="8"/>')
    o.append(lock(135, 96, 1.1))
    o.append('<text class="dg-t" x="135" y="148" text-anchor="middle">sealed holdout</text>')
    o.append('<text class="dg-s" x="135" y="165" text-anchor="middle">everything after 30 Jun 2025</text>')
    cards = [(16, "no", "OPEN IT NOW", ["score the 18 strategies that already failed",
                                        "→ learns nothing new, and the data is then \"seen\"",
                                        "→ no clean test left for the next idea"]),
             (138, "ok", "KEEP IT SEALED  (chosen)", ["write the next idea down first: rules, thresholds, stop rule",
                                                      "→ open the holdout once, for that idea only",
                                                      "→ one honest verdict"])]
    for y, cls, title, lines in cards:
        o.append(f'<path class="fork {cls}" d="M250,126 C290,126 290,{y+48} 330,{y+48}"/>')
        o.append(f'<rect class="card-{cls}" x="330" y="{y}" width="590" height="96" rx="8"/>')
        o.append(f'<text class="verdict-t {cls}" x="350" y="{y+26}">{title}</text>')
        for j, line in enumerate(lines):
            o.append(f'<text class="dg-s2" x="350" y="{y+50+j*18}">{esc(line)}</text>')
    o.append("</svg>")
    return "".join(o)


def cards(items, cls="cards"):
    """items: [(kicker, name, body, pill_text, tone)] → a row of HTML cards."""
    return f'<div class="{cls}" style="--n:{len(items)}">' + "".join(
        f'<div class="card {tone}"><span class="ck">{k}</span><span class="cn">{n}</span>'
        f'<span class="cb">{b}</span>{f"<span class=\"pill {tone}\">{p}</span>" if p else ""}</div>'
        for k, n, b, p, tone in items) + "</div>"


EXTRA_CSS = """
:root{--c4-soft:#F4EAD9}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--c4-soft:#33281A}}
:root[data-theme="dark"]{--c4-soft:#33281A}
.bar.cn{fill:var(--cn)}
.figfull{flex:0 0 auto}
.dg-t{fill:var(--ink);font-size:13px;font-family:var(--sans);font-weight:600;letter-spacing:.04em}
.dg-s{fill:var(--muted);font-size:11.5px;font-family:var(--sans)}
.dg-s2{fill:var(--ink2);font-size:12.5px;font-family:var(--sans)}
.c3t{fill:var(--c3)}
.verdict-t{font-size:13px;font-family:var(--sans);font-weight:650;letter-spacing:.05em}
.verdict-t.ok{fill:var(--ok)} .verdict-t.no{fill:var(--no)}
.barval.c4{fill:var(--c4)} .barval.c1{fill:var(--c1)}
ul.pts{margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:12px;max-width:62ch}
ul.pts li{position:relative;padding-left:20px;font-size:16px;line-height:1.5;color:var(--ink2)}
ul.pts li::before{content:"";position:absolute;left:0;top:.62em;width:7px;height:7px;
  border-radius:50%;background:var(--c1)}
.two ul.pts{max-width:none}
.cite{font-size:12.5px;color:var(--muted);margin:2px 0 0;max-width:none}
ol.refs{margin:0;padding-left:20px;display:flex;flex-direction:column;gap:10px;max-width:90ch}
ol.refs li{color:var(--ink2);font-size:14.5px;line-height:1.5}
ol.refs .why{display:block;color:var(--muted);font-size:12.5px}
.takeaway{font-family:var(--serif);font-size:clamp(18px,1.6vw,23px);color:var(--ink);
  max-width:60ch;line-height:1.45}
.slide>*,.split>*,.two>*,.figfull,.figwrap,.body{min-width:0}
/* timeline */
.tl-line{stroke:var(--hair);stroke-width:2}
.tl-ph{fill:var(--muted);font-size:10.5px;font-family:var(--mono);letter-spacing:.06em}
.tl-t{fill:var(--ink);font-size:13.5px;font-family:var(--sans);font-weight:600}
.tl-s{fill:var(--muted);font-size:11.5px;font-family:var(--sans)}
.tl-node{stroke:var(--surface);stroke-width:3}
.tl-node.cn{fill:var(--cn)} .tl-node.no{fill:var(--no)} .tl-node.warn{fill:var(--c4)} .tl-node.now{fill:var(--c1)}
.chip.cn{fill:var(--sunken)} .chip.no{fill:var(--c3-soft)} .chip.warn{fill:var(--c4-soft)} .chip.now{fill:var(--c1-soft)}
.chip-t{font-size:11.5px;font-family:var(--mono);font-weight:500}
.chip-t.cn{fill:var(--muted)} .chip-t.no{fill:var(--no)} .chip-t.warn{fill:var(--c4)} .chip-t.now{fill:var(--c1)}
/* data timeline, holdout, purge */
.seg-train{fill:var(--sunken);stroke:var(--hair)} .seg-strong{fill:var(--c1-soft)} .seg-fade{fill:var(--c3-soft)}
.seg-sealed{fill:url(#hatch6);stroke:var(--ink2);stroke-width:1.5}
.seg-sealed2{fill:var(--sunken);stroke:var(--ink2);stroke-width:1.5;stroke-dasharray:5 4}
.hatch{stroke:var(--hair);stroke-width:3}
.seg-t{fill:var(--ink2);font-size:12px;font-family:var(--sans);font-weight:550}
.seal-t{fill:var(--ink);font-size:10.5px;font-family:var(--mono);letter-spacing:.08em}
.lock-arc{fill:none;stroke:var(--ink);stroke-width:2.6}
.lock-body{fill:var(--ink)}
.fork{fill:none;stroke-width:2.5} .fork.ok{stroke:var(--ok)} .fork.no{stroke:var(--no)}
.card-ok{fill:var(--c2-soft)} .card-no{fill:var(--c3-soft)}
.wf-key-in{fill:var(--surface);font-size:11px;font-family:var(--sans);font-weight:600}
.purged{fill:none;stroke:var(--c3);stroke-width:1.5;stroke-dasharray:4 3}
.arc-ok{fill:none;stroke:var(--c1);stroke-width:1.5} .arc-ok-d{fill:var(--c1)}
.arc-leak{fill:none;stroke:var(--c3);stroke-width:2} .arc-leak-d{fill:var(--c3)}
/* nested feature sets */
.nest1{fill:var(--c2-soft);stroke:var(--c2);stroke-width:1.5}
.nest2{fill:var(--c4-soft);stroke:var(--c4);stroke-width:1.5}
.nest3{fill:var(--c3-soft);stroke:var(--c3);stroke-width:1.5}
.nest-n{font-size:12.5px;font-family:var(--sans);font-weight:650;letter-spacing:.07em}
.nest-n.nest1{fill:var(--c2);stroke:none} .nest-n.nest2{fill:var(--c4);stroke:none} .nest-n.nest3{fill:var(--c3);stroke:none}
.nest-l{fill:var(--ink2);font-size:12.5px;font-family:var(--sans)}
/* feature fade */
.flab{fill:var(--ink2);font-size:12px;font-family:var(--sans)}
.fade-ln{stroke-width:3;opacity:.55} .fade-ln.c2{stroke:var(--c2)} .fade-ln.c3{stroke:var(--c3)}
/* chart extras */
.shade{fill:var(--c3-soft);opacity:.7}
.shadelab{fill:var(--c3);font-size:10.5px;font-family:var(--sans);letter-spacing:.05em}
.seg{stroke:var(--ink2);stroke-width:1.5;stroke-dasharray:5 3}
.seglab{fill:var(--ink2);font-size:11px;font-family:var(--mono)}
/* html cards */
.cards{display:grid;grid-template-columns:repeat(var(--n),minmax(0,1fr));gap:12px;margin:4px 0 6px}
.card{background:var(--sunken);border-radius:6px;padding:14px 15px;border-top:3px solid var(--cn);
  display:flex;flex-direction:column;gap:6px}
.card.ok{border-top-color:var(--ok)} .card.no{border-top-color:var(--no)}
.card.warn{border-top-color:var(--c4)} .card.c1{border-top-color:var(--c1)}
.card .ck{font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:.06em;text-transform:uppercase}
.card .cn{font-family:var(--serif);font-size:18px;font-weight:600;color:var(--ink);line-height:1.2}
.card .cb{font-size:13px;color:var(--ink2);line-height:1.45;flex:1 1 auto}
.card .pill{align-self:flex-start;white-space:normal}
.pill.warn{background:var(--c4-soft);color:var(--c4)} .pill.c1{background:var(--c1-soft);color:var(--c1)}
.ruled{display:flex;flex-wrap:wrap;gap:8px 10px;align-items:center;margin-top:12px;font-size:13px;color:var(--muted)}
.ruled .x{background:var(--c3-soft);color:var(--no);border-radius:3px;padding:3px 9px;
  font-size:12.5px;text-decoration:line-through;text-decoration-thickness:1px}
@media (max-width:900px){.cards{grid-template-columns:1fr}}
@media (max-width:700px){
  html,body{overflow-x:hidden}
  .keyhint,.dotnav{display:none}
  .slide{padding-left:16px;padding-right:16px}
}
"""

SHARPE_NOTE = "Sharpe ratio: annualized, after all trading costs, adjusted for autocorrelation."

# ── slides ───────────────────────────────────────────────────────────────────

SLIDES = []


def slide(**kw):
    SLIDES.append(kw)


# 1 · title
curves = {f: weekly(cum(cell(BEST_KEY[0], BEST_KEY[1], f)["net"])) for f in FS}
xs = list(range(len(curves["p17"])))
xl = [d.strftime("%Y-%m-%d") for d in curves["p17"].index]
slide(
    layout="title",
    section="Phase 6 · NSE 500",
    title="The edge faded after 2021.<br/>New data didn't bring it back.",
    body=f"""
<p class="lede">The stock-ranking model lost most of its forecasting skill after 2021. Phase 6 asked
<strong>why</strong>, then tried to restore it by giving the model <strong>new information</strong>:
market, sector, options and futures data. Every addition made the strategy worse.</p>
{stats([
    ("best strategy, Sharpe after costs", fmt(BEST['sr'])),
    ("forecast skill, 2016–20 → 2024–25", f"{EARLY_IC:.3f} → {RECENT_IC:.3f}"),
    ("strategies that pass the luck test", f"{N_PASS} of {len(R)}"),
])}
""",
    figure=declutter(svg_line([
        {"name": FS_LAB["p17"], "x": xs, "y": list(curves["p17"].values), "cls": "c2", "width": 2.4, "xlab": xl},
        {"name": FS_LAB["p17x"], "x": xs, "y": list(curves["p17x"].values), "cls": "c4", "width": 1.8},
        {"name": FS_LAB["p17xo"], "x": xs, "y": list(curves["p17xo"].values), "cls": "c3", "width": 1.8},
    ], w=940, h=260, ylog=True, ytickvals=[0.5, 1, 2, 4, 8],
       ylab=lambda v: f"{v:g}×", xticks=year_ticks(curves["p17"].index), hover_fmt=lambda v: f"{v:.2f}×")),
    caption=f"Growth of ₹1 after costs, log scale. {BEST_NAME[0].upper() + BEST_NAME[1:]}, 2016–2025, "
            "with three versions of the model's information.",
    notes="Green is the model we started with. Amber and red are the same strategy with more data fed in. More data, less money.",
)

# 2 · the strategy
slide(
    layout="split",
    section="The story so far",
    title="How the strategy trades",
    body=pts(
        "Every day a machine-learning model ranks ~500 Indian stocks by expected return over the next few days.",
        "Buy the top tenth. Sell short the bottom tenth, but only stocks that can actually be borrowed.",
        "Three portfolio styles: <strong>long-only</strong>, <strong>long/short</strong>, and <strong>130/30</strong> "
        "(130% long, 30% short; market-neutral when markets are stressed).",
        "The model is refitted every quarter on the past two years, and only ever scored on days it has not seen.",
    ),
    figure=bd.diagram_crosssection().replace("predicted next-day return", "predicted return"),
    caption="Schematic: one day's 500 stocks, ordered by the model's predicted return.",
)

# 3 · five phases
slide(
    layout="figure",
    section="The story so far",
    title="Five phases, one question: does it survive real costs?",
    body=f"""
<div class="two">
{pts("Each phase removed one comfortable assumption: simulated prices, free trading, zero bid-ask spread, "
     "shorting anything, counting only the attempts that worked.",
     "Each correction was applied to every strategy alike, whether it helped or hurt.")}
{pts("<strong>The luck test</strong>: when many variants are tried, the best looks good partly by chance. "
     "A result passes only if it is 95% likely to beat the best that luck alone would produce "
     "(the Deflated Sharpe Ratio).")}
</div>
""",
    body_class="body",
    figure=diagram_phases(),
)

# 4 · costs, then M2
slide(
    layout="figure",
    section="The story so far",
    title="Costs ate the edge; smarter trading won half of it back",
    body=f"""
<div class="two">
{pts("<strong>Left</strong>: the 500-stock result that first passed, corrected one step at a time. "
     "Measuring the real cost of trading is what killed it.")}
{pts(f"<strong>Right</strong>: the best strategy (130/30, 5-day). Swapping a stock only when the expected "
     f"gain beats the cost (rule M2) lifted it from {fmt(P5_BUF['sr'])} to <strong>{fmt(P5_M2['sr'])}</strong>. "
     f"A pass needed about {P5_BAR:.1f}.",
     "<strong>The trading problem is largely solved. The forecast is not strong enough.</strong>")}
</div>
""",
    body_class="body",
    figure='<div class="two">' + svg_waterfall([
        {"label": "first reported", "value": WATERFALL[0], "state": "start"},
        {"label": "honest count|of attempts", "value": WATERFALL[1], "state": "pass"},
        {"label": "measured|spread & impact", "value": WATERFALL[2], "state": "fail"},
        {"label": "only borrowable|shorts", "value": WATERFALL[3], "state": "fail"},
    ], w=640, h=380) + ml_ticks(svg_bars(["no costs|(ceiling)", "after costs,|old rule", "after costs,|M2"], [
        {"name": "Sharpe", "values": [P5_BUF["gsr"], P5_BUF["sr"], P5_M2["sr"]], "cls": "c1", "neg_cls": "c3"},
    ], w=520, h=320, ytitle="Sharpe ratio, 130/30 with a 5-day forecast", value_fmt=lambda v: fmt(v, 2), barw=70,
        ref=[{"y": P5_BAR, "label": f"needed|≈ {P5_BAR:.1f}"}, {"y": P5_STAR, "label": f"luck alone|{P5_STAR:.2f}"}])) + "</div>",
    caption="Left: long/short strategy, Sharpe after each correction. Right: " + SHARPE_NOTE,
)

# 5 · forecast fading
slide(
    layout="split",
    section="The story so far",
    title="Where we stood: the forecast was fading",
    body=pts(
        "<strong>Forecast skill</strong> = each day, the rank correlation between the model's ordering of stocks "
        "and how they actually did. 0 is a coin flip; 0.02–0.05 is useful in equities.",
        f"2016–20 it averaged <strong>{EARLY_IC:.3f}</strong>. From 2021 it fell to <strong>{LATE_IC:.3f}</strong>, "
        f"and to <strong>{RECENT_IC:.3f}</strong> in 2024–25.",
        "Cheaper trading can't fix this. A weaker forecast is less to trade on, however cheaply.",
        "<strong>That is what Phase 6 set out to explain, and fix.</strong>",
    ),
    figure=bars_colored([str(y) if y < 2025 else "2025|Jan–Jun" for y in YEARS], IC_ENS,
                        ["c1" if y < 2021 else "c3" for y in YEARS], w=620, h=330,
                        ytitle="forecast skill by year (5-day forecast)", value_fmt=lambda v: fmt(v, 3),
                        segs=[{"from": 0, "to": 4, "y": EARLY_IC, "label": f"avg {EARLY_IC:.3f}"},
                              {"from": 5, "to": 9, "y": LATE_IC, "label": f"avg {LATE_IC:.3f}", "cls": ""}]),
    caption="Mean daily rank correlation between forecast and the next 5 days' returns. Dashed lines: day-weighted averages.",
)

# 6 · two questions
slide(
    layout="text",
    section="Why Phase 6",
    title="Two questions, asked in order",
    body=cards([
        ("Question 1 · diagnosis", "Why did the skill fade?",
         "Five possible causes, each with a test and a pass mark written down in advance. "
         "Nothing is tuned, so no new attempt is counted against the luck test.",
         "read-only · 42 attempts, unchanged", "c1"),
        ("Question 2 · repair", "Can new information bring it back?",
         "Give the same model more to look at: market, sector, options and futures data. "
         "Also try a cheaper way to trade the slower forecast.",
         "18 new strategies · 42 → 57 attempts", "warn"),
    ]) + diagram_data_timeline() + """
<p class="caption">The data. Everything after 30 June 2025 is held back unread: one clean test, to be used once,
for an idea fixed in writing before it is opened.</p>
<p class="aside" style="margin-top:10px">Every rule, threshold and stop condition was committed to the
repository before any result was computed.</p>
""",
)

# 7 · five suspects
slide(
    layout="text",
    section="Q1 · Why the edge faded",
    title="Five suspects",
    body=cards([
        ("1", "Competed away", "Well-known signals stop paying once enough traders use them.",
         f"almost: {N_DECAYED} of {N_SKILL} signals faded, rule needed 8", "warn"),
        ("2", "The market changed", "The signals work in some market moods and not others, and the mood shifted.",
         "cleared: no signal flipped", "ok"),
        ("3", "Model set up wrong", "The training window is too short or too long to keep up.",
         "cleared: no setup recovers it", "ok"),
        ("4", "Different stocks", "The 500-stock universe filled up with newer, different companies.",
         "cleared: explains ~0% of the drop", "ok"),
        ("5", "Bad data", "Price errors or corporate actions corrupting the inputs.",
         "cleared: data got cleaner", "ok"),
    ]) + f"""
<div class="two" style="margin-top:16px">
{pts("No suspect met its pass mark, so formally the decay is <strong>unexplained</strong>. The rule is not relaxed after the fact.",
     f"But suspect 1 missed by one signal, and its other two conditions held.")}
{pts("The rest of this section is the evidence for suspect 1.")}
</div>
""",
    notes=("Pass marks: suspect 1 needed two-thirds of the skill signals to lose half their strength; "
           "suspect 2 a sign flip or a state difference with |t| ≥ 3 (largest was 2.6); suspect 4 needed the "
           "universe's age mix to explain half the drop (it explains −1%); suspect 5 needed data disagreement "
           "to double (it fell to 0.29×)."),
)

# 8 · what faded and what held
slide(
    layout="figure",
    section="Q1 · Why the edge faded",
    title="What faded, and what held",
    body=f"""
<div class="two">
{pts(f"<strong>Faded (red)</strong>: low volatility, lottery-like jumps, illiquidity, 1-year momentum. "
     f"About <strong>{FADE*100:.0f}%</strong> of their strength is gone.",
     "<strong>Held (green)</strong>: short-term reversal and turnover. Some got stronger.")}
{pts("The faded group is the most published and most traded set of stock anomalies. "
     "Published anomalies lose about half their return after publication.",
     "<strong>Reading</strong>: the crowded signals got crowded out.")}
</div>
<p class="cite">Low volatility: Ang et al. (2006), Frazzini &amp; Pedersen (2014). Lottery: Bali, Cakici &amp;
Whitelaw (2011). Illiquidity: Amihud (2002). Decay after publication: McLean &amp; Pontiff (2016).</p>
""",
    body_class="body",
    figure=diagram_feature_fade(),
)

# 9 · no model could have found it
_rc = lambda col: [float(ridge_ceil.loc[y, (col, "mean_IC")]) for y in YEARS]  # noqa: E731
_cp = [float(comp.loc[y, ("5b", "mean_IC")]) for y in YEARS]
_fig9 = declutter(svg_line([
    {"name": "the trees", "x": list(range(10)), "y": _rc("ensemble"), "cls": "c1", "width": 2.4,
     "xlab": [str(y) for y in YEARS]},
    {"name": "linear model", "x": list(range(10)), "y": _rc("ridge"), "cls": "cn", "width": 1.6, "dash": "5 4"},
    {"name": "“cheating” model", "x": list(range(10)), "y": _rc("ceiling"), "cls": "c3", "width": 2},
    {"name": "frozen rule", "x": list(range(10)), "y": _cp, "cls": "c2", "width": 2.2},
], w=940, h=300, ylab=lambda v: f"{v:g}".replace("-", "−"), xticks=[(i, str(y)) for i, y in enumerate(YEARS)],
    hover_fmt=lambda v: fmt(v, 3)))
slide(
    layout="figure",
    section="Q1 · Why the edge faded",
    title="No model could have found it, not even one that cheats",
    body=f"""
<div class="two">
{pts("<strong>The “cheating” model</strong> is fitted inside each year, on that year's own future. "
     f"In 2024–25 even it finds about <strong>{fmt(CEIL_RECENT, 3)}</strong>.",
     "A simple linear model fades exactly like the trees, so it is not the choice of learner.")}
{pts(f"<strong>A frozen rule</strong>, just each signal's 2016–20 direction, never updated, beats the trees "
     f"from 2021 on: <strong>{COMP_LATE:.3f}</strong> vs {LATE_IC:.3f}.",
     "<strong>So: change the information, not the model.</strong>")}
</div>
""",
    body_class="body",
    figure=shade_line(_fig9, 940, 300, 10, 5, 9, "2021 ONWARD"),
    caption="Forecast skill by year, 5-day forecast. 2025 is January–June.",
    notes=("The cheating model trains on about 190 days against the real model's 504, so it sits below the real "
           "model even in early years. It is built the same way every year, and its collapse is what matters."),
)

# 10 · memory
slide(
    layout="split",
    section="Q1 · Why the edge faded",
    title="The trees chase noise; longer memory helps",
    body=pts(
        "Same model, three training windows: 1, 2 (used) and 4 years.",
        f"Recently, the longest memory does best: <strong>{WIN['1008']['recent']:.3f}</strong> vs "
        f"{WIN['504']['recent']:.3f} (2 years) and {WIN['252']['recent']:.3f} (1 year).",
        "The model isn't adapting too <em>slowly</em>. It adapts too <em>fast</em>: "
        "a training window that contains the 2020 crash teaches it the wrong lessons.",
        "A rule that never adapts at all beats every window.",
        "<strong>Parked as a future idea</strong>, not tried here: choosing it after seeing this would be cheating.",
    ),
    figure=svg_bars(["2021–25", "2024–25"], [
        {"name": "1-year memory", "values": [WIN["252"]["late"], WIN["252"]["recent"]], "cls": "cn"},
        {"name": "2-year (used)", "values": [WIN["504"]["late"], WIN["504"]["recent"]], "cls": "c1"},
        {"name": "4-year memory", "values": [WIN["1008"]["late"], WIN["1008"]["recent"]], "cls": "c2"},
        {"name": "frozen 2016–20 rule", "values": [COMP_LATE, COMP_RECENT], "cls": "c4"},
    ], w=620, h=330, ytitle="forecast skill (5-day forecast)", value_fmt=lambda v: fmt(v, 3)),
    legend_items=[("1-year memory", "cn"), ("2-year (used)", "c1"), ("4-year memory", "c2"), ("frozen 2016–20 rule", "c4")],
    caption="2025 is January–June. Window comparison uses one learner (LightGBM) on common dates; the frozen rule uses fixed signs.",
    notes="Fold skill correlates −0.50 with the share of the training window that falls in the COVID period.",
)

# 11 · the leak
slide(
    layout="figure",
    section="Q1 · Why the edge faded",
    title="One honest correction along the way",
    body=f"""
<div class="two">
{pts("A 5-day forecast is trained on 5-day answers. The last few training days' answers weren't known yet "
     "on the first test day, so the model had a small peek at the test period.",
     "The fix drops those days from training. It is applied to every strategy from here on.")}
{pts(f"Best strategy: Sharpe <strong>{fmt(P5_M2['sr'])} → {fmt(BEST['sr'])}</strong>, "
     f"annual return {pct(P5_M2['cagr'], 1)} → {pct(BEST['cagr'], 1)}.",
     "<strong>The leak-free numbers are the true baseline.</strong> Every Phase 6 result is compared against them.")}
</div>
""",
    body_class="body",
    figure=diagram_purge(),
    caption="Each arc is one training day's answer, reaching 5 trading days ahead.",
    notes=("Found during the decay study. It cost about 3% of forecast skill at 5 days and about 13% at 21 days. "
           "A uniform correction like this does not count as a new attempt."),
)

# 12 · the plan
slide(
    layout="figure",
    section="Q2 · Trying to improve it",
    title="The plan: richer information, cheaper trading",
    body=f"""
<div class="two">
{pts("<strong>Richer information</strong>: three nested sets, so each layer's value can be read off by itself.",
     "The new data was chosen to be <em>different</em> from what faded: what option traders and futures "
     "markets are pricing, how stocks move with their sector and the market.")}
{pts("<strong>Cheaper trading</strong>: the 21-day forecast is rebalanced once a month instead of daily.",
     f"<strong>The grid</strong>: 3 data sets × 2 forecasts × 3 styles = 18 strategies, all trading with M2. "
     f"A pass needs about <strong>{BAR_SR:.1f}</strong> after costs.")}
</div>
""",
    body_class="body",
    figure=diagram_nest(),
)

# 13 · the result
_rows13 = [{"label": STYLE_LAB[s], "sub": TGT_LAB[t], **{f: cell(t, s, f)["sr"] for f in FS}} for t, s in ORDER]
slide(
    layout="figure",
    section="Q2 · Trying to improve it",
    title="Adding new data made every strategy worse",
    body=f"""
<div class="two">
{pts(f"The base model beats both richer versions in <strong>{BASE_WINS} of 6</strong> strategies.",
     f"Adding price & index data cost up to {fmt(MAX_DROP, 2)} of Sharpe. "
     "Options & futures data won a little back on the 5-day forecast, never enough.")}
{pts(f"<strong>{N_PASS} of {len(R)}</strong> strategies pass. The best is still the base model at "
     f"{fmt(BEST['sr'])}, against {fmt(SR_STAR)} from luck alone.")}
</div>
""",
    body_class="body",
    figure=svg_shift3(_rows13, ytitle="Sharpe ratio after costs",
                      ref=[{"y": BAR_SR, "label": f"needed|≈ {BAR_SR:.1f}"}, {"y": SR_STAR, "label": f"luck alone|{SR_STAR:.2f}"}]),
    legend_items=[(FS_LAB[f], FS_CLS[f]) for f in FS],
    caption=SHARPE_NOTE + " 2016-01 to 2025-06.",
)

# 14 · the forecast got worse
_ic5 = {f: [float(ic6.loc[y, ("5b", f, "mean_IC")]) for y in YEARS] for f in ("p17", "p17x")}
_fig14b = declutter(svg_line([
    {"name": "base", "x": list(range(10)), "y": _ic5["p17"], "cls": "c2", "width": 2.4, "xlab": [str(y) for y in YEARS]},
    {"name": "+ price/index", "x": list(range(10)), "y": _ic5["p17x"], "cls": "c4", "width": 2},
], w=560, h=300, ylab=lambda v: f"{v:g}".replace("-", "−"), xticks=[(i, f"'{str(y)[2:]}") for i, y in enumerate(YEARS)],
    hover_fmt=lambda v: fmt(v, 3)))
_turn = {f: [cell(t, s, f)["turn"] for t, s in ORDER] for f in FS}
slide(
    layout="figure",
    section="Q2 · Trying to improve it",
    title="The forecast got worse, not the trading",
    body=f"""
<div class="two">
{pts(f"Forecast skill roughly <strong>halved</strong> with the first new layer: "
     f"{IC_FS[('5b','p17')]:.3f} → {IC_FS[('5b','p17x')]:.3f} at 5 days. The random forest fell furthest, "
     f"{RF_FS['p17']:.3f} → {RF_FS['p17x']:.3f}.",
     f"Trading barely changed: turnover moved by at most "
     f"{max(abs(a/b-1) for f in FS[1:] for a, b in zip(_turn[f], _turn['p17']))*100:.0f}%.")}
{pts("After 2021, the period it was meant to fix, it helped in 2021–22 and hurt in 2023–25. "
     f"Averaged over 2021–25: {IC6_LATE['p17']:.3f} → {IC6_LATE['p17x']:.3f}.")}
</div>
""",
    body_class="body",
    figure='<div class="two">' + svg_bars(["5-day forecast", "21-day forecast"], [
        {"name": FS_LAB[f], "values": [IC_FS[(t, f)] for t in TARGETS], "cls": FS_CLS[f]} for f in FS
    ], w=560, h=300, ytitle="forecast skill, whole period", value_fmt=lambda v: fmt(v, 3)) +
        shade_line(_fig14b, 560, 300, 10, 5, 9, "2021 ONWARD") + "</div>",
    legend_items=[(FS_LAB[f], FS_CLS[f]) for f in FS],
    caption="Right: forecast skill by year, 5-day forecast.",
)

# 15 · good ingredients, diluted
_fi = {p: [abs(float(feat_ic.loc[k, f"{p}_mean_IC"])) * (1 if feat_ic.loc[k, f"{p}_mean_IC"] * FEAT_SIGN[k] > 0 else -1)
           for k, _ in NEW_FEATS] for p in ("early", "late", "recent")}
_sim5 = [float(sim.loc[y, "5b p17x"]) for y in YEARS]
slide(
    layout="figure",
    section="Q2 · Trying to improve it",
    title="Good ingredients, diluted in the mix",
    body=f"""
<div class="two">
{pts(f"<strong>Futures basis</strong> (futures price vs spot) is the strongest single signal in the study, "
     f"and it got <em>stronger</em> after 2021: t {feat_ic.loc['fut_basis','early_t']:.1f} → "
     f"{feat_ic.loc['fut_basis','late_t']:.1f}.",
     "Two derivatives signals only appear recently. Three others are low-volatility look-alikes and fade like the rest.")}
{pts(f"But the new model doesn't add these on top. It <strong>reshuffles</strong> the ranking: it agrees "
     f"with the old one at only {SIM_LO:.2f}–{SIM_HI:.2f}.",
     "Market-wide inputs (VIX, market P/E) give the trees something to chase that doesn't help pick between stocks.")}
</div>
""",
    body_class="body",
    figure='<div class="two">' + ml_ticks(svg_bars([n for _, n in NEW_FEATS], [
        {"name": "2016–20", "values": _fi["early"], "cls": "cn"},
        {"name": "2021–25", "values": _fi["late"], "cls": "c1"},
        {"name": "2024–25", "values": _fi["recent"], "cls": "c2"},
    ], w=600, h=300, ytitle="each new signal's own forecast strength", value_fmt=lambda v: fmt(v, 3))) + bars_colored([str(y)[2:] for y in YEARS], _sim5, ["c4"] * 10, w=480, h=300,
                                 ytitle="new ranking vs old: rank correlation", value_fmt=lambda v: fmt(v, 2)) + "</div>",
    legend_items=[("2016–20", "cn"), ("2021–25", "c1"), ("2024–25", "c2"), ("agreement with base model, by year", "c4")],
    caption="Left: strength in each signal's useful direction (rank correlation with the next 5 days' returns). "
            "Right: + price & index model vs base model, 5-day forecast. 1.0 would mean the same ranking.",
)

# 16 · monthly rebalancing
slide(
    layout="figure",
    section="Q2 · Trying to improve it",
    title="Monthly trading cut costs, but gave up more than it saved",
    body=f"""
<div class="two">
{pts(f"Rebalancing the 21-day forecast once a month cut trading cost from "
     f"{min(CAD[s]['daily']['drag'] for s in STYLES)*100:.0f}–{max(CAD[s]['daily']['drag'] for s in STYLES)*100:.0f}% "
     f"a year to {min(CAD[s]['monthly']['drag'] for s in STYLES)*100:.0f}–{max(CAD[s]['monthly']['drag'] for s in STYLES)*100:.0f}%.")}
{pts("But holding a month-old ranking earns much less before costs. Net, every style ends up <strong>lower</strong> "
     "than trading the same forecast daily with M2.",
     "Cheaper trading only helps when there is a forecast worth trading.")}
</div>
""",
    body_class="body",
    figure='<div class="two">' + svg_bars([STYLE_LAB[s] for s in STYLES], [
        {"name": "traded daily", "values": [CAD[s]["daily"]["drag"] * 100 for s in STYLES], "cls": "cn"},
        {"name": "rebalanced monthly", "values": [CAD[s]["monthly"]["drag"] * 100 for s in STYLES], "cls": "c2"},
    ], w=520, h=290, ytitle="trading cost, % per year", value_fmt=lambda v: fmt(v, 1) + "%") + svg_bars(
        [STYLE_LAB[s] for s in STYLES], [
        {"name": "traded daily", "values": [CAD[s]["daily"]["sr"] for s in STYLES], "cls": "cn"},
        {"name": "rebalanced monthly", "values": [CAD[s]["monthly"]["sr"] for s in STYLES], "cls": "c2", "neg_cls": "c3"},
    ], w=520, h=290, ytitle="Sharpe ratio after costs", value_fmt=lambda v: fmt(v, 2)) + "</div>",
    legend_items=[("21-day forecast, traded daily with M2 (Phase 5)", "cn"), ("21-day forecast, rebalanced monthly (Phase 6)", "c2")],
    caption=("The daily numbers come from Phase 5, before the leak fix, which cost about 13% of the 21-day forecast's "
             "skill. That can explain only part of the gap: before costs, Sharpe falls from "
             f"{min(CAD[s]['daily']['gsr'] for s in STYLES):.2f}–{max(CAD[s]['daily']['gsr'] for s in STYLES):.2f} "
             f"to {min(CAD[s]['monthly']['gsr'] for s in STYLES):.2f}–{max(CAD[s]['monthly']['gsr'] for s in STYLES):.2f}. " + SHARPE_NOTE),
)

# 17 · scorecard
slide(
    layout="split",
    section="Q2 · Trying to improve it",
    title="Scorecard: still short of the bar",
    body=f"""
{pts(f"The best strategy is the one we started with: {BEST_NAME}, base data, leak-free.",
     f"Before costs it reaches {fmt(BEST['gsr'])}. After costs, <strong>{fmt(BEST['sr'])}</strong>.",
     f"Luck alone, across {N_TRIALS} attempts, would produce about <strong>{fmt(SR_STAR)}</strong>. "
     f"A pass needs about <strong>{BAR_SR:.1f}</strong>.",
     f"The luck-test score is <strong>{BEST['dsr']:.2f}</strong>: about a {BEST['dsr']*100:.0f}% chance it beats luck. A pass needs 0.95.")}
<p class="cite">Deflated Sharpe Ratio: Bailey &amp; López de Prado (2014). Autocorrelation adjustment: Lo (2002).</p>
""",
    figure=bars_colored(["no costs", "after costs"], [BEST["gsr"], BEST["sr"]], ["c1", "c1"],
        w=620, h=330, ytitle=f"Sharpe ratio ({BEST_NAME}, base data)", barw=100,
        ref=[{"y": BAR_SR, "label": f"needed|≈ {BAR_SR:.1f}"}, {"y": SR_STAR, "label": f"luck alone|{SR_STAR:.2f}"}]),
    caption=SHARPE_NOTE + " Benchmarks use this strategy's own track length, skew and kurtosis.",
)

# 18 · the holdout
slide(
    layout="figure",
    section="What next",
    title="The test we chose not to spend",
    body=f"""
<div class="two">
{pts("The plan's last step was to score all 18 strategies on the held-back data. It was <strong>not run</strong>.",
     "None had passed, so the step could not change the verdict. It could only use up the data.")}
{pts("A holdout is clean only until someone looks at it. After that, every later idea has been shaped by it.",
     "It stays sealed for the next idea, which must be written down in full before it is opened.")}
</div>
""",
    body_class="body",
    figure=diagram_holdout(),
)

# 19 · where profit could still come from
slide(
    layout="text",
    section="What next",
    title="Where profit could still come from after 2021",
    body=cards([
        ("1 · new information", "Delivery data",
         "The share of each day's trading that ends in actual delivery: a sign of real buyers, not day-traders. "
         "NSE's archive reaches back to 2013. This is the only new-signal route the stop rule leaves open.",
         "own pre-registration · sealed test", "c1"),
        ("2 · ideas parked from Q1", "Trade what survived",
         "A 1-day horizon, where a simple reversal-led rule still scores ~0.05, traded with M2. "
         "Or a longer-memory model. Both were spotted in this data, so they must be tested on new data.",
         "own pre-registration · sealed test", "c1"),
        ("3 · a different source of return", "Derivatives positioning",
         "Futures basis as its own strategy, not one input among 40. Or harvesting the premium that "
         "option buyers pay: most individual F&amp;O traders in India lose money (SEBI).",
         "a new program · sealed test", "warn"),
    ]) + """
<div class="ruled"><span>Ruled out by the stop rule:</span>
<span class="x">more price, index or options features</span>
<span class="x">re-tuning the model or the cost rule</span>
<span class="x">picking a winning variant from these results</span></div>
<p class="aside" style="margin-top:16px">Each idea gets its own written plan, its own count of attempts, and one look at the sealed data.</p>
""",
    notes=("Futures basis looks like the obvious candidate, but it was found in this data. Testing it here would be "
           "selection. Only the sealed holdout can say whether it is real."),
)

# 20 · takeaways
slide(
    layout="text",
    section="Takeaways",
    title="What we learned",
    body=f"""
<p class="takeaway">The edge most likely faded because it got crowded. More of the same kind of data can't buy it back.</p>
<div class="two" style="margin-top:22px">
{pts("<strong>Why it faded</strong>: the crowded low-volatility signals lost about "
     f"{FADE*100:.0f}% of their strength. Short-term reversal held. No model, not even one that sees the future, recovers it.",
     "<strong>The honest baseline</strong> is lower than we thought: "
     f"{fmt(BEST['sr'])} after fixing a small training leak, not {fmt(P5_M2['sr'])}.")}
{pts("<strong>New data made things worse</strong>: forecast skill roughly halved, and all 6 strategies lost Sharpe. "
     "The good new signals were diluted in the mix.",
     "<strong>Next</strong>: genuinely new information (delivery) or a different source of return, "
     "each tested once on the sealed data.")}
</div>
<p class="aside" style="margin-top:22px">Every rule and threshold was fixed before any result was seen,
along with a decision to stop adding price-based data if nothing passed. Nothing passed, so it stopped.</p>
""",
)

# A1 · all 18
rows = []
for t, s in ORDER:
    for f in FS:
        r = cell(t, s, f)
        rows.append([f"{STYLE_LAB[s]}, {TGT_LAB[t]}" if f == "p17" else "", FS_LAB[f],
                     fmt(r["gsr"]), fmt(r["sr"]), pct(r["cagr"]), pct(r["dd"]),
                     pct(r["turn"], 1), pct(r["drag"]), f"{r['dsr']:.3f}"])
slide(
    layout="text",
    section="Appendix",
    title="All 18 results",
    body=table(["strategy", "data", "Sharpe before costs", "Sharpe after", "return / yr", "max drawdown",
                "traded / day", "cost / yr", "luck-test score"], rows)
    + f'<p class="caption" style="margin-top:10px">{SHARPE_NOTE} Luck-test score = Deflated Sharpe Ratio at '
      f'{N_TRIALS} attempts; a pass needs 0.95. The 21-day strategies rebalance monthly. 2016-01 to 2025-06.</p>',
)

# A2 · the five tests
H = [("1 · competed away", "≥ ⅔ of skill signals lose half their strength; the cheating model collapses; a linear model fades like the trees",
      f"{N_DECAYED} of {N_SKILL} (64%); collapses; yes"),
     ("2 · the market changed", "a signal flips sign, or a market-state effect with |t| ≥ 3", "no flips; largest |t| 2.6"),
     ("3 · model set up wrong", "the cheating model keeps ≥ 70% of its skill and a 1-year window recovers ≥ half the gap",
      "keeps −3%; recovers −11%"),
     ("4 · different stocks", "the universe's age mix explains ≥ half the drop", "−1%"),
     ("5 · bad data", "data disagreement or extreme moves at least double", "0.29× and 0.51×")]
slide(
    layout="text",
    section="Appendix",
    title="The five tests, as written down in advance",
    body=table(["suspect", "convicted if", "measured", "verdict"],
               [[a, b, c, '<span class="pill no">not met</span>'] for a, b, c in H], cls="wrap")
    + '<p class="caption" style="margin-top:10px">Skill signals: features with |t| ≥ 3 in 2016–20. '
      'Early = 2016–20, late = 2021–25. All read-only: no attempt added to the count.</p>',
)

# references
slide(
    layout="text",
    section="References",
    title="References",
    body="""
<ol class="refs">
<li>Bailey, D. &amp; López de Prado, M. (2014). The Deflated Sharpe Ratio. <em>Journal of Portfolio Management</em> 40(5).
<span class="why">The luck test: the bar a result must clear after many attempts.</span></li>
<li>Lo, A. (2002). The statistics of Sharpe ratios. <em>Financial Analysts Journal</em> 58(4).
<span class="why">Sharpe ratios adjusted for autocorrelation throughout.</span></li>
<li>López de Prado, M. (2018). <em>Advances in Financial Machine Learning</em>, ch. 7. Wiley.
<span class="why">Purging training labels that overlap the test period (slide 11).</span></li>
<li>McLean, R. D. &amp; Pontiff, J. (2016). Does academic research destroy stock return predictability? <em>Journal of Finance</em> 71(1).
<span class="why">Anomalies lose much of their return after publication (slide 8).</span></li>
<li>Ang, A., Hodrick, R., Xing, Y. &amp; Zhang, X. (2006). The cross-section of volatility and expected returns. <em>Journal of Finance</em> 61(1).
<span class="why">The low-volatility anomaly.</span></li>
<li>Frazzini, A. &amp; Pedersen, L. (2014). Betting against beta. <em>Journal of Financial Economics</em> 111(1).
<span class="why">The low-risk anomaly.</span></li>
<li>Bali, T., Cakici, N. &amp; Whitelaw, R. (2011). Maxing out: stocks as lotteries. <em>Journal of Financial Economics</em> 99(2).
<span class="why">The lottery (biggest daily jump) signal.</span></li>
<li>Amihud, Y. (2002). Illiquidity and stock returns. <em>Journal of Financial Markets</em> 5(1).
<span class="why">The illiquidity signal.</span></li>
<li>Black, F. (1976). The pricing of commodity contracts. <em>Journal of Financial Economics</em> 3(1–2).
<span class="why">Implied volatility of NSE stock options from futures prices.</span></li>
<li>SEBI (2024). Analysis of profit and loss of individual traders dealing in equity F&amp;O segment.
<span class="why">Most individual F&amp;O traders lose money (slide 19).</span></li>
</ol>
""",
)

print(f"  built {len(SLIDES)} slides")


# ── page ─────────────────────────────────────────────────────────────────────

def build():
    total = len(SLIDES)
    body = "".join(bd.render_slide(s, i + 1, total) for i, s in enumerate(SLIDES))
    dotnav = "".join(f'<button type="button" aria-label="slide {i+1}"><i></i></button>'
                     for i in range(total))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Why the Edge Faded</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;650&family=Source+Serif+4:opsz,wght@8..60,600&display=swap">
<style>{bd.CSS}{EXTRA_CSS}
table.wrap td{{white-space:normal;text-align:left;vertical-align:top}}
table.wrap th{{text-align:left}}</style></head><body>
<div class="prog"></div>
<main class="deck">{body}</main>
<div class="hud">
  <span class="keyhint"><kbd>←</kbd> <kbd>→</kbd> to move · <kbd>N</kbd> for notes</span>
  <div class="hud-mid">
    <button class="navbtn prev" type="button" aria-label="previous slide">←</button>
    <span class="count">01 / {total}</span>
    <button class="navbtn next" type="button" aria-label="next slide">→</button>
    <div class="dotnav">{dotnav}</div>
  </div>
  <button class="navbtn notesbtn" type="button" aria-label="toggle presenter notes">N</button>
</div>
<div id="tip" role="status"></div>
<script>{bd.JS}</script>
</body></html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"  wrote {OUT.relative_to(ROOT)}  ({len(page)/1024:.0f} KB)")


if __name__ == "__main__":
    build()

    print("\nverification — computed values against the research record "
          "(docs/phase6_results.md, docs/phase6_decay_results.md, docs/phase6_stage7_diagnostics.md)")
    checks = [
        ("cells passing", str(N_PASS), "0"),
        ("best cell", f"{BEST_KEY[1]}/{BEST_KEY[0]}/{BEST_KEY[2]}", "dynamic_tilt_slb/5b/p17"),
        ("best DSR", fmt(BEST["dsr"], 3), "0.182"),
        ("best SR_ann", fmt(BEST["sr"]), "0.78"),
        ("best naive net SR", fmt(BEST["nsr"]), "0.84"),
        ("SR* (N=57)", fmt(SR_STAR), "1.09"),
        ("base wins (of 6)", str(BASE_WINS), "6"),
        ("5b IC p17/p17x/p17xo", "/".join(f"{IC_FS[('5b', f)]:.3f}" for f in FS), "0.049/0.025/0.026"),
        ("21b IC p17/p17x/p17xo", "/".join(f"{IC_FS[('21b', f)]:.3f}" for f in FS), "0.034/0.019/0.018"),
        ("RF 5b p17 → p17x", f"{RF_FS['p17']:.3f}/{RF_FS['p17x']:.3f}", "0.044/0.009"),
        ("2021–25 IC p17 / p17x", f"{IC6_LATE['p17']:.3f}/{IC6_LATE['p17x']:.3f}", "0.023/0.015"),
        ("Δ F0→F1 (DTB, LO, LSB; 5b)", "/".join(fmt(D_F1[('5b', s)]) for s in ("dynamic_tilt_slb", "long_only", "long_short_slb")), "−0.82/−0.63/−1.25"),
        ("Δ F1→F2 (DTB, LO, LSB; 5b)", "/".join(fmt(D_F2[('5b', s)], sign=True) for s in ("dynamic_tilt_slb", "long_only", "long_short_slb")), "+0.15/+0.09/+0.29"),
        ("Δ F0→F1 (DTB, LO, LSB; 21b)", "/".join(fmt(D_F1[('21b', s)]) for s in ("dynamic_tilt_slb", "long_only", "long_short_slb")), "−0.13/−0.05/−0.11"),
        ("Δ F1→F2 (DTB, LO, LSB; 21b)", "/".join(fmt(D_F2[('21b', s)]) for s in ("dynamic_tilt_slb", "long_only", "long_short_slb")), "−0.10/−0.08/−0.16"),
        ("fut_basis t early/late/recent", "/".join(f"{feat_ic.loc['fut_basis', p + '_t']:.1f}" for p in ("early", "late", "recent")), "5.6/11.7/6.3"),
        ("alpha similarity range", f"{SIM_LO:.2f}–{SIM_HI:.2f}", "0.11–0.43"),
        ("decayed skill features", f"{N_DECAYED}/{N_SKILL}", "7/11"),
        ("ceiling 2024–25 (weighted)", fmt(CEIL_RECENT, 3), "−0.001"),
        ("composite late 5b", fmt(COMP_LATE, 3), "0.047"),
        ("ensemble late 5b", fmt(LATE_IC, 3), "0.029"),
        ("window recent 1008/504/252", "/".join(fmt(WIN[w]["recent"], 3) for w in ("1008", "504", "252")), "0.023/0.013/0.007"),
        ("Phase 5 best M2 SR_ann", fmt(P5_M2["sr"]), "0.96"),
        ("Phase 5 best M2 CAGR", pct(P5_M2["cagr"]), "23.7%"),
        ("best CAGR / max DD", f"{pct(BEST['cagr'])} / {pct(BEST['dd'])}", "18.1% / −43.5%"),
        ("Phase 5 SR* (N=42)", fmt(P5_STAR), "1.17"),
        ("Phase 5 Sharpe needed", fmt(P5_BAR, 1), "1.8"),
        ("ledger ends on/before seal", str(p6_net.index.max() <= SEALED), "True"),
    ]
    for label, got, expect in checks:
        ok = got.replace("−", "-").replace("+", "") == expect.replace("−", "-").replace("+", "")
        print(f"  {'ok ' if ok else 'DIFF'}  {label:32} {got:>22}   record: {expect}")
