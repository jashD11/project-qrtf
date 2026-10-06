#!/usr/bin/env python3
"""Build the cost-aware rebalancing deck (M1 and M2).

Read-only. Reads the cost-aware construction ledgers under data/trial_database/
phase5/, computes every series and statistic, and writes one self-contained HTML
file to docs/presentation/phase5_deck.html. Chart primitives, CSS and the page
shell are reused from build_deck.py so the two decks look and behave alike.

Written for a viewer with no project context: no phase names, ledger names or
internal style codes on the slides. Economics only, with no significance-test
statistics; the "bar" is stated as the net Sharpe a pass would need
(docs/phase5_plan.md §9.3).

Sharpe ratios are annualized and serial-correlation adjusted (Lo, via the
overlapping variance ratio), as everywhere in the research record, except the
by-year figures, which are plain annualized Sharpe of daily net returns
(as in docs/phase5_plan.md §9.2).
"""
from __future__ import annotations

import io
import math
import sys
from contextlib import redirect_stdout

import pandas as pd

import build_deck as bd  # noqa: E402 - also builds its own slide list in memory; writes nothing
from deck_data import (LEDGER, ROOT, TRADING_DAYS, cum, fmt, load, max_dd,  # noqa: E402
                       p4c_net, pct, sharpe, weekly)

sys.path.insert(0, str(ROOT))
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402

import config  # noqa: E402
from deck_data import IC_T, IC_YEARS, STATUTORY_BPS  # noqa: E402
from src.production_ml import cost_aware  # noqa: E402
from src.production_ml.tier3_execution import load_cost_panels  # noqa: E402
from src.production_ml.tier4_dsr_gate import probabilistic_sharpe_ratio, run_dsr_gate  # noqa: E402

esc, Frame, svg_open, gridlines, nice_ticks, ref_marks = (
    bd.esc, bd.Frame, bd.svg_open, bd.gridlines, bd.nice_ticks, bd.ref_marks)
svg_line, svg_bars, stats, table = bd.svg_line, bd.svg_bars, bd.stats, bd.table

OUT = ROOT / "docs" / "presentation" / "phase5_deck.html"
P5 = LEDGER / "phase5"

# ── data ─────────────────────────────────────────────────────────────────────

print("loading cost-aware construction artefacts")
p5_net = load(P5 / "phase5_dsr_matrix.parquet")
p5_gross = load(P5 / "phase5_dsr_matrix_gross.parquet")
p5_diag = load(P5 / "phase5_dsr_matrix_execution_diagnostics.csv").set_index("strategy_id")
p5_gate = load(P5 / "phase5_dsr_matrix_dsr_gate.csv").set_index("strategy_id")
swaps = load(P5 / "diagnostics_swap_attribution.csv")

# The adjusted gross Sharpe is not persisted for the M1/M2 columns; score the gross
# ledger in memory (persist=False writes nothing). Only SR_ann is read from it.
with redirect_stdout(io.StringIO()):
    _g = run_dsr_gate(str(P5 / "phase5_dsr_matrix_gross.parquet"), n_trials=42, persist=False)
g_gate = _g.set_index("strategy_id") if "strategy_id" in _g.columns else _g
print(f"  scored gross ledger in memory  {g_gate.shape}")


def cagr(s: pd.Series) -> float:
    return float(cum(s).iloc[-1] ** (TRADING_DAYS / len(s)) - 1)


def yearly_sharpe(s: pd.Series) -> pd.Series:
    return s.groupby(s.index.year).apply(sharpe)


STYLES = ["long_only", "long_short_slb", "dynamic_tilt_slb"]
STYLE_LAB = {"long_only": "long-only", "long_short_slb": "long/short", "dynamic_tilt_slb": "130/30"}
STYLE_ABBR = {"long_only": "LO", "long_short_slb": "LS", "dynamic_tilt_slb": "130/30"}
TARGETS = ["5b", "21b"]
TGT_LAB = {"5b": "5-day", "21b": "21-day"}
TGT_ABBR = {"5b": "5d", "21b": "21d"}
CONS = ["buffer", "cost_band", "cost_swap"]
CONS_LAB = {"buffer": "baseline rule", "cost_band": "M1", "cost_swap": "M2"}
ORDER = [(t, s) for t in TARGETS for s in STYLES]

R = {}
for sid, row in p5_diag.iterrows():
    cons = row["construction"] if isinstance(row["construction"], str) else "buffer"
    tgt = "21b" if row["target"].endswith("21b") else "5b"
    net, gross = p5_net[sid], p5_gross[sid + "__gross"]
    R[(tgt, row["execution_style"], cons)] = {
        "sid": sid, "net": net, "gross": gross,
        "sr": float(p5_gate.loc[sid, "SR_ann"]),
        "gsr": float(g_gate.loc[sid + "__gross", "SR_ann"]),
        "cagr": cagr(net), "gcagr": cagr(gross), "dd": max_dd(net),
        "turn": float(row["turnover_mean"]), "drag": float(row["trade_drag_annual"]),
        "borrow": float(row["borrow_drag_annual"]), "bps": float(row["effective_oneway_bps"]),
    }
assert len(R) == 18, f"expected 18 strategies, got {len(R)}"


def cell(t, s, c):
    return R[(t, s, c)]


def short(t, s):
    return f"{STYLE_ABBR[s]} {TGT_ABBR[t]}"


turn_cut = [1 - cell(t, s, "cost_swap")["turn"] / cell(t, s, "buffer")["turn"] for t, s in ORDER]
best_key = max(ORDER, key=lambda k: cell(*k, "cost_swap")["sr"])
BEST_B, BEST_M2 = cell(*best_key, "buffer"), cell(*best_key, "cost_swap")
BEST_NAME = f"{STYLE_LAB[best_key[1]]}, {TGT_LAB[best_key[0]]} forecast"
YEARS = list(range(2016, 2026))
dsr_m1 = [cell(t, s, "cost_band")["sr"] - cell(t, s, "buffer")["sr"] for t, s in ORDER]
dsr_m2 = [cell(t, s, "cost_swap")["sr"] - cell(t, s, "buffer")["sr"] for t, s in ORDER]
neg_turned = sum(cell(t, s, "buffer")["sr"] < 0 < cell(t, s, "cost_swap")["sr"] for t, s in ORDER)

# Swap attribution: long leg of the long-only strategy (identical across styles)
def swap_rows(tgt):
    d = swaps[swaps["cell"] == f"long_only/{tgt}/long"].set_index("bucket")
    return d.loc[["Q1", "Q2", "Q3", "Q4", "Q5", "all"]]


SW5, SW21 = swap_rows("5b"), swap_rows("21b")
Q5_SHARE_LOSING = 0.8  # four of five quintiles lose

# M2 hurdle, with illustrative accuracy and volatility and the measured round-trip cost
IC_ILL, SIG_ILL = 0.04, 0.02
RT_COST = float(SW5.loc["all", "roundtrip_cost"])
PER_Z = {h: IC_ILL * SIG_ILL * math.sqrt(h) for h in (5, 21)}
HURDLE = {h: RT_COST / PER_Z[h] for h in (5, 21)}
TYPICAL_GAP = float(SW5.loc["all", "z_gap"])

# M1: how much trading costs really differ across stocks. The decision cost of a full
# position, from the causal cost panels, over every in-universe stock-day.
print("measuring per-stock cost dispersion")
with redirect_stdout(io.StringIO()):
    _ccfg = replace(config.ML_CONFIG.cost, **config.PHASE4C_COST_OVERRIDES)
    _panels = load_cost_panels()
    _um = load(ROOT / "data" / "nse500_universe_mask.parquet")
_um.index = pd.DatetimeIndex(_um.index).normalize()
_um = _um.reindex(p5_net.index).fillna(False).astype(bool)
_n = _um.sum(axis=1)
with redirect_stdout(io.StringIO()):
    _c = cost_aware.decision_cost_bps(_panels, _um.index, _um.columns,
                                      np.floor(_n * 0.10).clip(lower=1), _ccfg).where(_um)
_ratio = _c.div(_c.median(axis=1), axis=0)
_line = 100 * _ratio ** config.PHASE5_BAND_EXPONENT          # sell rank on a 100 scale
COST_Q = _c.stack().quantile([0.05, 0.95]).to_numpy()
LINE_LO, LINE_HI = (int(round(v)) for v in _line.stack().quantile([0.05, 0.95]))
M1_TURN = [cell(t, s, "cost_band")["turn"] / cell(t, s, "buffer")["turn"] - 1 for t, s in ORDER]
M1_BPS = [cell(t, s, "buffer")["bps"] - cell(t, s, "cost_band")["bps"] for t, s in ORDER]
print(f"  cost p5/p95 {COST_Q.round(1)} bps; sell line p5/p95 {LINE_LO}/{LINE_HI}")

# Slide 11: the Sharpe a pass needs, solved from the gate's own PSR formula with the
# best strategy's measured T, skew and kurtosis (deflated against SR* at N = 42).
_gb = p5_gate.loc[BEST_M2["sid"]]
SR_STAR = float(_gb["SR_star_ann"])
BEST_DSR = float(_gb["DSR"])
_T, _sk, _ku = int(_gb["T"]), float(_gb["skew"]), float(_gb["kurt"])
_star_d = SR_STAR / math.sqrt(TRADING_DAYS)
_lo, _hi = _star_d, _star_d * 4
for _ in range(80):
    _mid = (_lo + _hi) / 2
    _lo, _hi = (_mid, _hi) if probabilistic_sharpe_ratio(_mid, _T, _sk, _ku, _star_d) < 0.95 else (_lo, _mid)
BAR_SR = _hi * math.sqrt(TRADING_DAYS)
SE_BAR = math.sqrt((1 - _sk * _hi + (_ku - 1) / 4 * _hi ** 2) / (_T - 1)) * math.sqrt(TRADING_DAYS)
YEARS_T = _T / TRADING_DAYS
N_TRIALS = 42

# Forecast skill by year, 5-day forecast. SOURCE: docs/phase4c_results.md §4.2
IC_MEAN = [0.0475, 0.0523, 0.0586, 0.0863, 0.0564, 0.0217, 0.0312, 0.0368, 0.0183, 0.0127]

# ── diagrams ─────────────────────────────────────────────────────────────────

def diagram_rule(variant):
    """The selling rule on a rank axis. 'base': one sell line at 100 for every
    stock. 'm1': the sell line moves with each stock's trading cost."""
    w, h, n = 940, 180, 160
    f = Frame(w, h, 30, 30, 10, 40, (0, n), (0, 1))
    x = f.X
    o = [svg_open(w, h)]
    y0, bh = 70, 50
    zones = [(0, 50, "in", "BUY", "top 50"), (50, 100, "hold", "KEEP", "if already owned"),
             (100, n, "out", "SELL", "")]
    if variant == "m1":
        zones = zones[:2] + [(100, n, "out", "", "")]
    for a, b, cls, t1, t2 in zones:
        o.append(f'<rect class="band {cls}" x="{x(a)+1:.1f}" y="{y0}" width="{x(b)-x(a)-2:.1f}" height="{bh}" rx="3"/>')
        lx = (x(a) + x(min(b, LINE_LO) if variant == "m1" and cls == "hold" else b)) / 2
        if t1:
            o.append(f'<text class="dg-big {cls}" x="{lx:.1f}" y="{y0+22}" text-anchor="middle">{t1}</text>')
            o.append(f'<text class="dg-s" x="{lx:.1f}" y="{y0+39}" text-anchor="middle">{t2}</text>')
    o.append(f'<line class="axis" x1="{x(0):.1f}" y1="{y0+bh+8}" x2="{x(n):.1f}" y2="{y0+bh+8}"/>')
    ticks = (1, 50, 100, 150) if variant == "base" else (1, 50, LINE_LO, 100, LINE_HI, 150)
    for v in ticks:
        o.append(f'<text class="tick" x="{x(v):.1f}" y="{y0+bh+24}" text-anchor="middle">{v}</text>')
    if variant == "m1":
        for v, cls, lab, anc in ((LINE_LO, "c2", "cheapest 5% of stocks: sold sooner", "end"),
                                 (100, "cn", "median-cost stock", "start"),
                                 (LINE_HI, "c3", "most expensive 5%: held longer", "start")):
            dx = -7 if anc == "end" else 7
            ly = 22 if v != LINE_HI else 46
            o.append(f'<line class="exitmark {cls}" x1="{x(v):.1f}" y1="{ly-14}" x2="{x(v):.1f}" y2="{y0+bh+8}"/>')
            o.append(f'<text class="xslab {cls}" x="{x(v)+dx:.1f}" y="{ly}" text-anchor="{anc}">{lab}</text>')
    o.append(f'<text class="axtitle" x="{x(0):.1f}" y="{y0+bh+44}">rank today (1 = model\'s top pick, out of ~500) →</text>')
    o.append("</svg>")
    return "".join(o)


def diagram_keep_or_swap():
    """Two worked M2 decisions: extra expected return from switching vs the cost
    of switching. Same costs, different score gaps."""
    w, h = 940, 230
    o = [svg_open(w, h)]
    scale = 300 / 0.0075   # px per unit return
    cases = [("Case A · small edge", 1.2, "KEEP the old stock", "no"),
             ("Case B · big edge", 3.5, "SWAP", "ok")]
    for i, (title, zgap, verdict, cls) in enumerate(cases):
        x0 = 20 + i * 470
        gain = zgap * PER_Z[5]
        o.append(f'<text class="dg-t" x="{x0}" y="22">{title}</text>')
        o.append(f'<text class="dg-s" x="{x0}" y="40">new stock\'s score is {zgap:g} std-devs above the slipping one</text>')
        for j, (lab, v, bcls) in enumerate((("extra expected return", gain, "c1"),
                                            ("cost to switch (sell + buy)", RT_COST, "c3"))):
            y = 64 + j * 52
            o.append(f'<text class="dg-s" x="{x0}" y="{y}">{lab}</text>')
            o.append(f'<rect class="bar {bcls}" x="{x0}" y="{y+6}" width="{v*scale:.1f}" height="22" rx="3"/>')
            o.append(f'<text class="barval" x="{x0+v*scale+8:.1f}" y="{y+22}">{v*100:.2f}%</text>')
        o.append(f'<text class="verdict-t {cls}" x="{x0}" y="200">{"gain < cost" if cls == "no" else "gain > cost"}  →  {verdict}</text>')
    o.append(f'<line class="grpline" x1="470" y1="10" x2="470" y2="{h-10}"/>')
    o.append("</svg>")
    return "".join(o)


EXTRA_CSS = """
.bar.cn{fill:var(--cn)}
.figfull{flex:0 0 auto}
.m1tick{stroke:var(--c4);stroke-width:3;stroke-linecap:round}
.dg-t{fill:var(--ink);font-size:13px;font-family:var(--sans);font-weight:600}
.dg-s{fill:var(--muted);font-size:11.5px;font-family:var(--sans)}
.dg-big{font-size:15px;font-family:var(--sans);font-weight:650;letter-spacing:.06em}
.dg-big.in{fill:var(--c2)} .dg-big.hold{fill:var(--c1)} .dg-big.out{fill:var(--c3)}
.band.in{fill:var(--c2-soft)} .band.hold{fill:var(--c1-soft)} .band.out{fill:var(--c3-soft)}
.exitmark{stroke-width:2.5} .exitmark.c2{stroke:var(--c2)} .exitmark.c3{stroke:var(--c3)}
.exitmark.cn{stroke:var(--cn)}
.xslab.cn{fill:var(--ink2)}
.verdict-t{font-size:14px;font-family:var(--sans);font-weight:650}
.verdict-t.ok{fill:var(--ok)} .verdict-t.no{fill:var(--no)}
.formula{font-family:var(--mono);font-size:14px;background:var(--sunken);border-radius:4px;
  padding:10px 14px;color:var(--ink);margin:4px 0 10px;overflow-x:auto}
ul.pts{margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:12px;max-width:62ch}
ul.pts li{position:relative;padding-left:20px;font-size:16px;line-height:1.5;color:var(--ink2)}
ul.pts li::before{content:"";position:absolute;left:0;top:.62em;width:7px;height:7px;
  border-radius:50%;background:var(--c1)}
.two ul.pts{max-width:none}
.cite{font-size:12.5px;color:var(--muted);margin:2px 0 0;max-width:none}
.cite em{font-style:italic}
ol.refs{margin:0;padding-left:20px;display:flex;flex-direction:column;gap:12px;max-width:90ch}
ol.refs li{color:var(--ink2);font-size:15px;line-height:1.5}
ol.refs .why{display:block;color:var(--muted);font-size:13px}
.takeaway{font-family:var(--serif);font-size:clamp(18px,1.6vw,23px);color:var(--ink);
  max-width:60ch;line-height:1.45}
.slide>*,.split>*,.two>*,.figfull,.figwrap,.body{min-width:0}
@media (max-width:700px){
  html,body{overflow-x:hidden}
  .two{grid-template-columns:1fr}
  .keyhint,.dotnav{display:none}
  .slide{padding-left:16px;padding-right:16px}
}
"""


def pts(*items):
    return '<ul class="pts">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def svg_shift(rows, *, w=940, h=340, ylab=lambda v: f"{v:g}".replace("-", "−"), ytitle=None, ref=None,
              vfmt=lambda v: fmt(v)):
    """One column per strategy: baseline (grey) joined to M2 (green), with M1 as
    an amber tick. rows: list of {label, sub, buffer, cost_band, cost_swap}."""
    vals = [r[c] for r in rows for c in CONS] + [x["y"] for x in ref or []]
    lo, hi = min(vals + [0]), max(vals + [0])
    pad = (hi - lo) * 0.12
    f = Frame(w, h, 62, 104, 26, 66, (0, len(rows)), (lo - pad, hi + pad))
    out = [svg_open(w, h), gridlines(f, nice_ticks(lo, hi, 5), ylab), ref_marks(f, ref)]
    gw = f.pw / len(rows)
    for i, r in enumerate(rows):
        cx = f.l + gw * (i + 0.5)
        yb, y1, y2 = f.Y(r["buffer"]), f.Y(r["cost_band"]), f.Y(r["cost_swap"])
        out.append(f'<line class="dbell" x1="{cx:.1f}" y1="{yb:.1f}" x2="{cx:.1f}" y2="{y2:.1f}"/>')
        out.append(f'<line class="m1tick" x1="{cx-11:.1f}" y1="{y1:.1f}" x2="{cx+11:.1f}" y2="{y1:.1f}" '
                   f'data-tip="{esc(f"{r['label']} {r['sub']} · M1: {vfmt(r['cost_band'])}")}"/>')
        for key, cls, y in (("buffer", "cn", yb), ("cost_swap", "c2", y2)):
            out.append(f'<circle class="dot {cls}" cx="{cx:.1f}" cy="{y:.1f}" r="5.5" '
                       f'data-tip="{esc(f"{r['label']} {r['sub']} · {CONS_LAB[key]}: {vfmt(r[key])}")}"/>')
        out.append(f'<text class="barval" x="{cx+14:.1f}" y="{yb+4:.1f}">{vfmt(r["buffer"])}</text>')
        out.append(f'<text class="barval c2" x="{cx+14:.1f}" y="{y2+4:.1f}">{vfmt(r["cost_swap"])}</text>')
        out.append(f'<text class="tick" x="{cx:.1f}" y="{f.t+f.ph+20:.1f}" text-anchor="middle">{esc(r["label"])}</text>')
    for g in ({"from": 0, "to": 2, "label": "5-DAY FORECAST"}, {"from": 3, "to": 5, "label": "21-DAY FORECAST"}):
        x0, x1 = f.l + gw * g["from"], f.l + gw * (g["to"] + 1)
        out.append(f'<line class="grpline" x1="{x0+6:.1f}" y1="{f.t+f.ph+34:.1f}" x2="{x1-6:.1f}" y2="{f.t+f.ph+34:.1f}"/>')
        out.append(f'<text class="grplab" x="{(x0+x1)/2:.1f}" y="{f.t+f.ph+50:.1f}" text-anchor="middle">{g["label"]}</text>')
    if ytitle:
        out.append(f'<text class="axtitle" x="{f.l}" y="14">{esc(ytitle)}</text>')
    out.append("</svg>")
    return "".join(out)


def shift_rows(key):
    return [{"label": STYLE_LAB[s], "sub": TGT_LAB[t], **{c: cell(t, s, c)[key] for c in CONS}}
            for t, s in ORDER]


LEG3 = [("baseline rule", "cn"), ("M1", "c4"), ("M2", "c2")]
SHARPE_NOTE = "Sharpe ratio: annualized, after all trading costs, adjusted for autocorrelation."

# ── slides ───────────────────────────────────────────────────────────────────

SLIDES = []


def slide(**kw):
    SLIDES.append(kw)


def year_ticks(idx):
    return [(i, str(d.year)) for i, d in enumerate(idx)
            if d.year in (2016, 2018, 2020, 2022, 2024) and (i == 0 or idx[i - 1].year != d.year)]


# 1 · title
gw_, bw_, mw_ = (weekly(cum(s)) for s in (BEST_B["gross"], BEST_B["net"], BEST_M2["net"]))
xs = list(range(len(gw_)))
xl = [d.strftime("%Y-%m-%d") for d in gw_.index]
slide(
    layout="title",
    section="Cost-aware rebalancing · NSE 500",
    title="Trade less, keep more:<br/>two cost-aware rebalancing rules",
    body=f"""
<p class="lede">Same stock-ranking model, same trading costs. The only change is <strong>when the
portfolio is allowed to swap a stock</strong>.</p>
{stats([
    ("best strategy, Sharpe after costs", f"{fmt(BEST_B['sr'])} → {fmt(BEST_M2['sr'])}"),
    ("less trading", f"{min(turn_cut)*100:.0f}–{max(turn_cut)*100:.0f}%"),
    ("trading cost per year, best strategy", f"{pct(BEST_B['drag'], 0)} → {pct(BEST_M2['drag'], 0)}"),
])}
""",
    figure=svg_line([
        {"name": "before costs", "x": xs, "y": list(gw_.values), "cls": "c1", "width": 1.6, "xlab": xl},
        {"name": "M2, after costs", "x": xs, "y": list(mw_.values), "cls": "c2", "width": 2.4},
        {"name": "baseline, after costs", "x": xs, "y": list(bw_.values), "cls": "cn", "width": 1.8},
    ], w=940, h=280, ylog=True, ytickvals=[0.5, 1, 2, 5, 10, 20, 50],
       ylab=lambda v: f"{v:g}×", xticks=year_ticks(gw_.index), hover_fmt=lambda v: f"{v:.2f}×"),
    caption=f"Growth of ₹1, log scale. {BEST_NAME[0].upper() + BEST_NAME[1:]}, 2016–2025.",
    notes="The gap between blue and grey is what trading costs took. M2 wins back about half of it.",
)

# 2 · setup
slide(
    layout="figure",
    section="Setup",
    title="How the strategy trades",
    body=f"""
<div class="two">
{pts("Every day a model ranks ~500 Indian stocks by expected return.",
     "Buy the top 50. Keep a stock until it falls out of the top 100. The gap stops the "
     "portfolio from churning on small rank changes.")}
{pts(f"Every swap costs about <strong>{RT_COST*100:.1f}%</strong> (sell + buy: taxes, fees, bid-ask spread, price impact).",
     "Three portfolio styles are tested: <strong>long-only</strong>, <strong>long/short</strong>, "
     "<strong>130/30</strong>. Each uses a 5-day and a 21-day forecast, so there are 6 strategies.")}
</div>
""",
    body_class="body",
    figure=diagram_rule("base"),
    caption="The baseline rule: the same sell line (rank 100) for every stock.",
    notes=("Short positions use the mirror image of the same rule at the bottom of the ranking. "
           "130/30 runs 130% long / 30% short in calm markets and switches to market-neutral in stressed ones. "
           "Shorts are limited to stocks that can actually be borrowed."),
)

# 3 · problem
QB = ["Q1", "Q2", "Q3", "Q4", "Q5"]
QB_LAB = ["smallest", "2", "3", "4", "largest"]
slide(
    layout="split",
    section="The problem",
    title="Most swaps cost more than they earn",
    body=pts(
        f"We looked at all {int(SW5.loc['all','n']):,} swaps the baseline rule made.",
        f"The average swap earned <strong>{pct(SW5.loc['all','realised_gap'], 2)}</strong> "
        f"but cost <strong>{pct(SW5.loc['all','roundtrip_cost'], 2)}</strong>.",
        "Only swaps where the new stock looked <em>much</em> better paid for themselves.",
        "<strong>So: skip the swaps that don't pay.</strong>",
    ),
    figure=svg_bars(QB_LAB, [
        {"name": "return gained by swapping", "values": [SW5.loc[q, "realised_gap"] * 100 for q in QB], "cls": "c1"},
        {"name": "cost of the swap", "values": [SW5.loc[q, "roundtrip_cost"] * 100 for q in QB], "cls": "c3"},
    ], w=620, h=320, ytitle="% per swap, grouped by how much better the new stock looked",
        value_fmt=lambda v: fmt(v, 2) + "%"),
    legend_items=[("return gained by swapping", "c1"), ("cost of the swap", "c3")],
    caption="Long-only, 5-day forecast. Swaps sorted into five equal groups by the score gap between the new stock and the one it replaced.",
)

# 4 · M1
slide(
    layout="figure",
    section="Rule M1",
    title="M1: hold expensive stocks a little longer",
    body=f"""
<div class="two">
<div><h4>Why costs differ by stock</h4>
{pts(f"Taxes and fees (~{STATUTORY_BPS:.0f} bps) are the same for every stock.",
     "Bid-ask spread and price impact are not. They are larger for thinly traded stocks.",
     f"So one-way cost runs from <strong>{COST_Q[0]:.0f} bps</strong> to <strong>{COST_Q[1]:.0f} bps</strong> (5th–95th percentile).")}</div>
<div><h4>The rule</h4>
{pts("Each stock gets its own sell line: expensive stocks are held longer, cheap ones sold sooner.",
     "Buying is unchanged.")}
<div class="formula">sell line = 100 × (stock's cost ÷ median cost)<sup>1/3</sup></div>
<p class="cite">Cube-root width of the no-trade zone: Rogers (2004); Janeček &amp; Shreve (2004).
Using it for a rank-based sell line is our adaptation.</p></div>
</div>
""",
    body_class="body",
    figure=diagram_rule("m1"),
    caption=f"Measured over every stock-day, 2016–2025: 90% of sell lines fall between rank {LINE_LO} and {LINE_HI}.",
    notes=("Costs use the same model as the backtest: statutory charges, a half-spread estimated from "
           "daily highs and lows (Corwin & Schultz), and square-root price impact, all from data up to the previous day."),
)

# 5 · M2 idea
slide(
    layout="figure",
    section="Rule M2",
    title="M2: swap only if the gain beats the cost",
    body=f"""
<div class="two">
{pts("When a stock drops out of the top 100, don't sell it automatically.",
     "Compare it with the best stock waiting to come in. Swap <strong>only if</strong> "
     "the extra expected return beats the cost of switching.")}
<div>
<div class="formula">expected return = accuracy × volatility × √horizon × score</div>
{pts("<strong>accuracy</strong> is how well the model's rankings have predicted over the past year, using only past data.",
     "Forced sales (stock delisted, can no longer be shorted) and empty slots always trade.")}
<p class="cite">Expected return: Grinold (1994); Grinold &amp; Kahn (2000). The √horizon term is our
extension; it assumes daily returns are roughly independent.</p>
</div></div>
""",
    body_class="body",
    figure=diagram_keep_or_swap(),
    caption=f"Illustrative: 5-day forecast, accuracy (rank correlation) {IC_ILL}, volatility {SIG_ILL*100:.0f}%/day, "
            f"switch cost {RT_COST*100:.2f}% (the measured average).",
    notes="Expected return is Grinold's rule: IC · σ · √h · z. The swap test is E_new − E_old > cost_new + cost_old.",
)

# 6 · M2 hurdle
slide(
    layout="split",
    section="Rule M2",
    title="With a 5-day forecast, almost no swap clears the bar",
    body=pts(
        "A short forecast gives a swap only a few days to earn back its cost.",
        f"<strong>5-day:</strong> the new stock must beat the old one by about <strong>{HURDLE[5]:.0f}</strong> std-devs of score. Rare.",
        f"<strong>21-day:</strong> about <strong>{HURDLE[21]:.1f}</strong>. Some swaps get through.",
        f"A typical swap offers {TYPICAL_GAP:.2f}, so M2 mostly <strong>holds</strong>.",
    ),
    figure=svg_bars(["5-day forecast", "21-day forecast"], [
        {"name": "score gap needed to swap", "values": [HURDLE[5], HURDLE[21]], "cls": "c1"},
    ], w=620, h=320, ytitle="score gap (std-devs) needed before a swap pays", barw=110,
        value_fmt=lambda v: fmt(v, 1), ymin=0,
        ref=[{"y": TYPICAL_GAP, "label": "typical gap|offered"}]),
    caption="Same illustrative accuracy, volatility and cost as the previous slide.",
)

# 7 · result: Sharpe
slide(
    layout="figure",
    section="Result",
    title="M1 changes nothing. M2 improves every strategy.",
    body=f"""
<div class="two">
{pts(f"<strong>M2</strong>: Sharpe up {fmt(min(dsr_m2), 2, True)} to {fmt(max(dsr_m2), 2, True)} on all 6. "
     f"All {neg_turned} money-losing strategies turn profitable.")}
{pts(f"<strong>M1</strong>: within ±0.03. It is roughly cost-neutral: expensive stocks held longer, "
     f"cheap ones sold sooner, so total trading changes only {min(M1_TURN)*100:+.0f}% to {max(M1_TURN)*100:+.0f}%.")}
</div>
""",
    body_class="body",
    figure=svg_shift(shift_rows("sr"), ytitle="Sharpe ratio after costs"),
    legend_items=LEG3,
    caption=SHARPE_NOTE + " 2016–2025.",
)

# 8 · mechanism
slide(
    layout="figure",
    section="Why it works",
    title="M2 works by trading less",
    body=f"""
<div class="two">
{pts(f"Trading falls {min(turn_cut)*100:.0f}–{max(turn_cut)*100:.0f}%, and annual trading cost falls with it.")}
{pts("The cost <em>per trade</em> is unchanged, so M2 isn't picking cheaper stocks. "
     "It just trades less often.")}
</div>
""",
    body_class="body",
    figure='<div class="two">' + svg_bars(
        [short(t, s) for t, s in ORDER],
        [{"name": CONS_LAB[c], "values": [cell(t, s, c)["turn"] * 100 for t, s in ORDER], "cls": k}
         for c, k in zip(CONS, ("cn", "c4", "c2"))],
        w=520, h=280, ytitle="% of portfolio traded per day", value_fmt=lambda v: fmt(v, 0) + "%") + svg_bars(
        [short(t, s) for t, s in ORDER],
        [{"name": CONS_LAB[c], "values": [cell(t, s, c)["drag"] * 100 for t, s in ORDER], "cls": k}
         for c, k in zip(CONS, ("cn", "c4", "c2"))],
        w=520, h=280, ytitle="trading cost, % per year", value_fmt=lambda v: fmt(v, 1) + "%") + "</div>",
    legend_items=LEG3,
    caption="LO = long-only, LS = long/short; 5d / 21d = forecast horizon.",
)

# 9 · by year
YS = {k: {c: yearly_sharpe(cell(*k, c)["net"]) for c in ("buffer", "cost_swap")}
      for k in [("5b", "dynamic_tilt_slb"), ("5b", "long_only"), ("21b", "long_only")]}
wins = {k: sum(YS[k]["cost_swap"][y] > YS[k]["buffer"][y] for y in YEARS) for k in YS}
dt = YS[best_key]
slide(
    layout="split",
    section="Consistency",
    title="Better in every single year",
    body=pts(
        "M2 beats the baseline in <strong>10 out of 10 years</strong>, across every strategy checked. "
        "The gain isn't one lucky period.",
        "But 2025 is negative under both rules. The model's forecasting skill has faded since 2021.",
        f"In Jan–Jun 2025 its skill is <strong>statistically indistinguishable from zero</strong> "
        f"(t = {IC_T[-1]:.1f}, vs {min(IC_T[:5]):.0f}–{max(IC_T[:5]):.0f} in 2016–20). A trading rule can't add skill.",
    ),
    figure=svg_bars([str(y) for y in YEARS], [
        {"name": "baseline rule", "values": [dt["buffer"][y] for y in YEARS], "cls": "cn"},
        {"name": "M2", "values": [dt["cost_swap"][y] for y in YEARS], "cls": "c2"},
    ], w=620, h=230, ytitle=f"Sharpe ratio after costs, by year ({BEST_NAME})",
        value_fmt=lambda v: fmt(v, 2)) + svg_bars(IC_YEARS, [
        {"name": "forecast skill (t-stat)", "values": IC_T, "cls": "c1"},
    ], w=620, h=180, ytitle="model's forecast skill by year (t-statistic of daily rank correlation)",
        value_fmt=lambda v: fmt(v, 2), ref=[{"y": 2.0, "label": "significant|above ≈ 2"}]),
    legend_items=[("baseline rule", "cn"), ("M2", "c2"), ("forecast skill", "c1")],
    caption=("Top: plain annualized Sharpe within each year. Bottom: how strongly each day's ranking "
             "predicted the next 5 days' returns, aggregated per year. 2025 is Jan–Jun only, which by "
             "itself lowers its t-statistic by about √2."),
    notes=("Average daily rank correlation (IC) by year: " +
           ", ".join(f"{y} {v:.3f}" for y, v in zip(IC_YEARS, IC_MEAN)) +
           ". 2025 has both the lowest average IC and the lowest t-statistic. "
           "Also 10/10 for long-only on both forecasts."),
)

# 10 · drawdown
dd_best = max(ORDER, key=lambda k: cell(*k, "cost_swap")["dd"] - cell(*k, "buffer")["dd"])
slide(
    layout="figure",
    section="Risk",
    title="Smaller losses from peak",
    body=pts(
        f"Worst peak-to-trough loss shrinks everywhere. Biggest case: "
        f"<strong>{pct(cell(*dd_best, 'buffer')['dd'], 0)} → {pct(cell(*dd_best, 'cost_swap')['dd'], 0)}</strong> "
        f"({STYLE_LAB[dd_best[1]]}, {TGT_LAB[dd_best[0]]}).",
        "Much of the old drawdown wasn't market risk. It was trading costs draining the account day after day.",
    ),
    body_class="body",
    figure=svg_shift(shift_rows("dd"), ytitle="maximum drawdown after costs", ylab=lambda v: f"{v*100 + 0.0:.0f}%".replace("-0%", "0%"),
                     vfmt=lambda v: pct(v, 0)),
    legend_items=LEG3,
)

# 11 · still short
slide(
    layout="split",
    section="The catch",
    title="Better, but still not good enough",
    body=f"""
{pts(f"Even at zero cost the best strategy reaches only <strong>{fmt(BEST_B['gsr'])}</strong>. M2 gets <strong>{fmt(BEST_M2['sr'])}</strong>.",
     f"<strong>Deflated Sharpe Ratio (DSR)</strong>: test {N_TRIALS} variants and the best looks good partly by luck. "
     "DSR is the probability that the true Sharpe beats the best you'd expect from luck alone.")}
<div class="formula">luck benchmark (best of {N_TRIALS}, no skill) ≈ {SR_STAR:.2f}<br/>
needed = {SR_STAR:.2f} + 1.65 × {SE_BAR:.2f} (uncertainty over {YEARS_T:.1f} yrs) ≈ <strong>{BAR_SR:.1f}</strong></div>
{pts(f"M2's best has DSR <strong>{BEST_DSR:.2f}</strong>: about a {BEST_DSR*100:.0f}% chance it beats luck. A pass needs 0.95.",
     "<strong>The limit is the forecast, not the trading.</strong>")}
<p class="cite">Bailey &amp; López de Prado (2014). Sharpe adjusted for autocorrelation: Lo (2002).</p>
""",
    figure=svg_bars(["no costs (ceiling)", "baseline rule", "M2"], [
        {"name": "Sharpe", "values": [BEST_B["gsr"], BEST_B["sr"], BEST_M2["sr"]], "cls": "c1", "neg_cls": "c3"},
    ], w=620, h=330, ytitle=f"Sharpe ratio ({BEST_NAME})",
        value_fmt=lambda v: fmt(v, 2), barw=90,
        ref=[{"y": BAR_SR, "label": f"needed for|95% (≈ {BAR_SR:.1f})"},
             {"y": SR_STAR, "label": f"luck alone|({SR_STAR:.2f})"}]),
    caption=SHARPE_NOTE + " Benchmark and uncertainty use this strategy's measured track length, skew and kurtosis.",
    notes=("SR* = expected maximum Sharpe of N independent no-skill trials, scaled by the spread of Sharpes "
           f"across the variants actually run. 'Needed' is solved exactly from the DSR formula at 0.95: {BAR_SR:.2f}."),
)

# 12 · takeaways
slide(
    layout="text",
    section="Takeaways",
    title="What we learned",
    body=f"""
<p class="takeaway">Pricing the cost into each swap decision works. It just can't rescue a weak forecast.</p>
<div class="two" style="margin-top:22px">
{pts("<strong>M2 works</strong>: better Sharpe on every strategy, about half the trading cost, better every year.",
     "<strong>M1 barely matters</strong>: it holds expensive stocks longer but sells cheap ones sooner, so total trading hardly changes.")}
{pts(f"<strong>Not enough here</strong>: even M2's best ({fmt(BEST_M2['sr'])}) is well short of the ≈{BAR_SR:.1f} needed, "
     "and the model's skill is fading.",
     "<strong>Next</strong>: find a stronger signal (new data), and trade it with M2 from day one.")}
</div>
<p class="aside" style="margin-top:22px">The rules and settings were fixed before any results were seen,
along with a decision to stop if nothing passed. So there is no after-the-fact tuning.</p>
""",
)

# A1 · appendix
rows = []
for t, s in ORDER:
    for c in CONS:
        r = cell(t, s, c)
        rows.append([f"{STYLE_LAB[s]}, {TGT_LAB[t]}" if c == "buffer" else "", CONS_LAB[c],
                     pct(r["gcagr"]), pct(r["cagr"]), fmt(r["gsr"]), fmt(r["sr"]), pct(r["turn"], 1),
                     pct(r["drag"]), f"{r['bps']:.1f}", pct(r["dd"])])
slide(
    layout="text",
    section="Appendix",
    title="All 18 results",
    body=table(["strategy", "rule", "return before costs", "return after costs", "Sharpe before",
                "Sharpe after", "traded / day", "cost / yr", "cost per trade (bps)", "max drawdown"], rows)
         + f'<p class="caption" style="margin-top:10px">Returns are annualized. {SHARPE_NOTE} 2016-01 to 2025-06.</p>',
)

slide(
    layout="text",
    section="References",
    title="References",
    body="""
<ol class="refs">
<li>Rogers, L.C.G. (2004). Why is the effect of proportional transaction costs O(δ<sup>2/3</sup>)?
<em>Mathematics of Finance</em>, Contemporary Mathematics 351, AMS.
<span class="why">M1: the no-trade zone widens with the cube root of cost.</span></li>
<li>Janeček, K. &amp; Shreve, S. (2004). Asymptotic analysis for optimal investment and consumption with
transaction costs. <em>Finance and Stochastics</em> 8(2), 181–206.
<span class="why">M1: the same cube-root result, derived independently.</span></li>
<li>Grinold, R. (1994). Alpha is volatility times IC times score. <em>Journal of Portfolio Management</em> 20(4), 9–16.
<span class="why">M2: expected return = accuracy × volatility × score.</span></li>
<li>Grinold, R. &amp; Kahn, R. (2000). <em>Active Portfolio Management</em>, 2nd ed. McGraw-Hill.
<span class="why">M2: the same framework in textbook form.</span></li>
<li>Bailey, D. &amp; López de Prado, M. (2014). The Deflated Sharpe Ratio: correcting for selection bias,
backtest overfitting and non-normality. <em>Journal of Portfolio Management</em> 40(5).
<span class="why">Slide 11: the bar a result must clear after many variants have been tried.</span></li>
<li>Lo, A. (2002). The statistics of Sharpe ratios. <em>Financial Analysts Journal</em> 58(4).
<span class="why">Sharpe ratios adjusted for autocorrelation throughout.</span></li>
<li>Corwin, S. &amp; Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low
prices. <em>Journal of Finance</em> 67(2).
<span class="why">The per-stock spread used in every trading cost.</span></li>
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
<title>Cost-Aware Rebalancing</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;650&family=Source+Serif+4:opsz,wght@8..60,600&display=swap">
<style>{bd.CSS}{EXTRA_CSS}</style></head><body>
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

    print("\nverification — computed values against the research record (docs/phase5_plan.md §9)")
    buf_equal = all(p5_net[cell(t, s, "buffer")["sid"]].equals(p4c_net[cell(t, s, "buffer")["sid"]].reindex(p5_net.index))
                    for t, s in ORDER)
    checks = [
        ("baseline columns == prior run", str(buf_equal), "True"),
        ("130/30 5d baseline gross SR", fmt(cell("5b", "dynamic_tilt_slb", "buffer")["gsr"]), "1.89"),
        ("130/30 5d baseline net SR", fmt(cell("5b", "dynamic_tilt_slb", "buffer")["sr"]), "−0.05"),
        ("130/30 5d M2 net SR", fmt(cell("5b", "dynamic_tilt_slb", "cost_swap")["sr"]), "0.96"),
        ("130/30 5d M2 trade drag", pct(cell("5b", "dynamic_tilt_slb", "cost_swap")["drag"]), "15.5%"),
        ("LO 5d M2 turnover", fmt(cell("5b", "long_only", "cost_swap")["turn"], 3), "0.155"),
        ("LO 5d M2 trade drag", pct(cell("5b", "long_only", "cost_swap")["drag"]), "10.4%"),
        ("LS 21d M2 net SR", fmt(cell("21b", "long_short_slb", "cost_swap")["sr"]), "0.12"),
        ("swaps 5d long", f"{int(SW5.loc['all','n'])}", "23663"),
        ("M2 wins every year (3 strategies)", str(all(v == 10 for v in wins.values())), "True"),
        ("luck benchmark SR* (N=42)", fmt(SR_STAR), "1.17"),
        ("best M2 DSR", fmt(BEST_DSR, 3), "0.272"),
        ("Sharpe needed for 95%", fmt(BAR_SR, 1), "1.8"),
        ("PSR at needed Sharpe", fmt(probabilistic_sharpe_ratio(BAR_SR / math.sqrt(TRADING_DAYS), _T, _sk, _ku, _star_d), 3), "0.950"),
        ("M1 sell line p5/p95", f"{LINE_LO}/{LINE_HI}", "87/131"),
    ]
    for label, got, expect in checks:
        ok = got.replace("−", "-").lstrip("+") == expect.replace("−", "-").lstrip("+")
        print(f"  {'ok ' if ok else 'DIFF'}  {label:34} {got:>10}   record: {expect}")
