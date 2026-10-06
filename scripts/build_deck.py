#!/usr/bin/env python3
"""Build the QRTF Engine explanatory deck.

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

from deck_data import *  # noqa: F403  - loaders, statistics and every computed figure

OUT = ROOT / "docs" / "presentation" / "qrtf_deck.html"



# ── SVG primitives ───────────────────────────────────────────────────────────
# Every chart is inline SVG with no external dependency. Text takes its colour
# from theme tokens via CSS classes, never a literal, so both themes resolve.

def esc(t) -> str:
    return html.escape(str(t), quote=True)



class Frame:
    """Maps data coordinates to a fixed viewBox."""

    def __init__(self, w, h, l, r, t, b, xlim, ylim, ylog=False):
        self.w, self.h = w, h
        self.l, self.r, self.t, self.b = l, r, t, b
        self.x0, self.x1 = xlim
        self.y0, self.y1 = ylim
        self.ylog = ylog

    @property
    def pw(self):
        return self.w - self.l - self.r

    @property
    def ph(self):
        return self.h - self.t - self.b

    def X(self, v):
        if self.x1 == self.x0:
            return self.l
        return self.l + (v - self.x0) / (self.x1 - self.x0) * self.pw

    def Y(self, v):
        if self.ylog:
            v = math.log10(max(v, 1e-9))
            y0, y1 = math.log10(max(self.y0, 1e-9)), math.log10(max(self.y1, 1e-9))
        else:
            y0, y1 = self.y0, self.y1
        if y1 == y0:
            return self.t + self.ph
        return self.t + self.ph - (v - y0) / (y1 - y0) * self.ph


def svg_open(w, h, cls="fig"):
    return (f'<svg class="{cls}" viewBox="0 0 {w} {h}" role="img" '
            f'preserveAspectRatio="xMidYMid meet">')


def gridlines(f: Frame, ticks, label=lambda v: v, axis=True):
    out = []
    for v in ticks:
        if not (min(f.y0, f.y1) - 1e-9 <= v <= max(f.y0, f.y1) + 1e-9):
            continue
        y = f.Y(v)
        out.append(f'<line class="grid" x1="{f.l}" y1="{y:.1f}" x2="{f.l+f.pw}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{f.l-8:.1f}" y="{y+3.5:.1f}" text-anchor="end">{esc(label(v))}</text>')
    if axis:
        yb = f.Y(0) if f.y0 <= 0 <= f.y1 else f.t + f.ph
        out.append(f'<line class="axis" x1="{f.l}" y1="{yb:.1f}" x2="{f.l+f.pw}" y2="{yb:.1f}"/>')
    return "".join(out)


def ref_marks(f: "Frame", ref):
    """A threshold rule plus its label. The label is placed in the right margin
    rather than over the plot, so it can never collide with a mark."""
    out = []
    for r in ref or []:
        y = f.Y(r["y"])
        out.append(f'<line class="ref" x1="{f.l}" y1="{y:.1f}" x2="{f.l+f.pw}" y2="{y:.1f}"/>')
        for i, part in enumerate(r["label"].split("|")):
            out.append(f'<text class="reflab" x="{f.l+f.pw+8:.1f}" '
                       f'y="{y + 3.5 + i * 12:.1f}">{esc(part)}</text>')
    return "".join(out)


def nice_ticks(lo, hi, n=5):
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min([m * mag for m in (1, 2, 2.5, 5, 10)], key=lambda s: abs(s - raw))
    start = math.floor(lo / step) * step
    ticks, v = [], start
    while v <= hi + step * 0.5:
        if lo - step * 0.5 <= v <= hi + step * 0.5:
            ticks.append(round(v, 10))
        v += step
    return ticks


def svg_line(series, *, w=760, h=380, ylog=False, ylab=lambda v: f"{v:g}",
             xticks=None, ytickvals=None, caption_pad=0, ref=None,
             hover_fmt=None, ymin=None, ymax=None):
    """series: list of dicts {name, x (list of floats), y, cls, dash, width}.

    A shared crosshair reads every series at the hovered x, so the chart is
    inspectable rather than decorative.
    """
    xs = [v for s in series for v in s["x"]]
    ys = [v for s in series for v in s["y"]]
    if ref:
        ys += [r["y"] for r in ref]
    lo = ymin if ymin is not None else min(ys)
    hi = ymax if ymax is not None else max(ys)
    if not ylog:
        pad = (hi - lo) * 0.08 or 1
        lo, hi = lo - pad, hi + pad
    f = Frame(w, h, 62, 118, 16, 34 + caption_pad, (min(xs), max(xs)), (lo, hi), ylog)
    ticks = ytickvals if ytickvals is not None else nice_ticks(lo, hi, 5)
    out = [svg_open(w, h), gridlines(f, ticks, ylab, axis=not ylog)]

    out.append(ref_marks(f, ref))

    for s in series:
        d = " ".join(f"{'M' if i == 0 else 'L'}{f.X(x):.1f},{f.Y(y):.1f}"
                     for i, (x, y) in enumerate(zip(s["x"], s["y"])))
        dash = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        out.append(f'<path class="ln {s.get("cls","c1")}" d="{d}"{dash} '
                   f'stroke-width="{s.get("width",2)}"/>')
        # direct label at the series end, so identity is never colour-alone
        ex, ey = s["x"][-1], s["y"][-1]
        out.append(f'<circle class="dot {s.get("cls","c1")}" cx="{f.X(ex):.1f}" cy="{f.Y(ey):.1f}" r="3.2"/>')
        out.append(f'<text class="endlab {s.get("cls","c1")}" x="{f.X(ex)+9:.1f}" '
                   f'y="{f.Y(ey)+3.5:.1f}">{esc(s["name"])}</text>')

    for xv, lab in (xticks or []):
        out.append(f'<text class="tick" x="{f.X(xv):.1f}" y="{f.t+f.ph+20:.1f}" '
                   f'text-anchor="middle">{esc(lab)}</text>')

    # crosshair hit-bands
    n = min(90, len(series[0]["x"]))
    idx = np.linspace(0, len(series[0]["x"]) - 1, n).astype(int)
    bw = f.pw / max(n - 1, 1)
    for i in idx:
        cx = f.X(series[0]["x"][i])
        parts = []
        for s in series:
            j = min(i, len(s["y"]) - 1)
            parts.append(f'{s["name"]}: {(hover_fmt or (lambda v: fmt(v)))(s["y"][j])}')
        xlab = series[0].get("xlab")
        tip = xlab[i] if xlab else ""
        out.append(f'<rect class="hit" x="{cx-bw/2:.1f}" y="{f.t}" width="{bw:.1f}" '
                   f'height="{f.ph:.1f}" data-x="{cx:.1f}" data-y0="{f.t}" data-y1="{f.t+f.ph:.1f}" '
                   f'data-tip="{esc((tip + " · " if tip else "") + " · ".join(parts))}"/>')
    out.append("</svg>")
    return "".join(out)


def svg_bars(groups, series, *, w=760, h=380, ylab=lambda v: f"{v:g}", ref=None,
             value_fmt=None, ytitle=None, ymin=None, ymax=None, barw=None):
    """Grouped bars. groups: list of category labels. series: list of dicts
    {name, values, cls}. Negative values draw down from the zero line."""
    vals = [v for s in series for v in s["values"] if v is not None]
    if ref:
        vals += [r["y"] for r in ref]
    lo = min(ymin if ymin is not None else min(vals + [0]), 0)
    hi = max(ymax if ymax is not None else max(vals + [0]), 0)
    pad = (hi - lo) * 0.10 or 1
    f = Frame(w, h, 62, 104, 22, 52, (0, len(groups)), (lo - pad * 0.4, hi + pad))
    out = [svg_open(w, h), gridlines(f, nice_ticks(lo, hi, 5), ylab)]

    out.append(ref_marks(f, ref))

    gw = f.pw / len(groups)
    k = len(series)
    bw = barw or min(34, (gw - 16) / k)
    zero = f.Y(0)
    for gi, g in enumerate(groups):
        cx = f.l + gw * (gi + 0.5)
        for si, s in enumerate(series):
            v = s["values"][gi]
            if v is None:
                continue
            x = cx - (k * bw + (k - 1) * 2) / 2 + si * (bw + 2)
            y = f.Y(v)
            top, hgt = min(y, zero), abs(y - zero)
            cls = s.get("cls", "c1")
            if v < 0 and s.get("neg_cls"):
                cls = s["neg_cls"]
            vf = (value_fmt or (lambda x: fmt(x)))(v)
            out.append(f'<rect class="bar {cls}" x="{x:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                       f'height="{max(hgt,1):.1f}" rx="2.5" '
                       f'data-tip="{esc(f"{g} · {s['name']}: {vf}")}"/>')
        out.append(f'<text class="tick" x="{cx:.1f}" y="{f.t+f.ph+20:.1f}" text-anchor="middle">{esc(g)}</text>')
    if ytitle:
        out.append(f'<text class="axtitle" x="{f.l}" y="12" text-anchor="start">{esc(ytitle)}</text>')
    out.append("</svg>")
    return "".join(out)


def legend(items):
    sw = "".join(
        f'<span class="lg"><i class="sw {c}"></i>{esc(n)}</span>' for n, c in items
    )
    return f'<div class="legend">{sw}</div>'


def svg_stack(groups, series, *, w=760, h=380, ylab=lambda v: f"{v:g}",
              labels=None, ytitle=None):
    """Stacked bars with a 2px surface gap between segments."""
    totals = [sum(s["values"][i] for s in series) for i in range(len(groups))]
    hi = max(totals)
    f = Frame(w, h, 62, 52, 22, 52, (0, len(groups)), (0, hi * 1.14))
    out = [svg_open(w, h), gridlines(f, nice_ticks(0, hi, 5), ylab)]
    gw = f.pw / len(groups)
    bw = min(46, gw - 18)
    for gi, g in enumerate(groups):
        cx = f.l + gw * (gi + 0.5)
        base = 0.0
        for s in series:
            v = s["values"][gi]
            y0, y1 = f.Y(base), f.Y(base + v)
            hgt = max(y0 - y1 - 2, 1)
            out.append(f'<rect class="bar {s.get("cls","c1")}" x="{cx-bw/2:.1f}" y="{y1:.1f}" '
                       f'width="{bw:.1f}" height="{hgt:.1f}" rx="2.5" '
                       f'data-tip="{esc(f"{g} · {s['name']}: {fmt(v,1)}")}"/>')
            base += v
        out.append(f'<text class="tick" x="{cx:.1f}" y="{f.t+f.ph+20:.1f}" text-anchor="middle">{esc(g)}</text>')
    if labels:
        for gi, txt in enumerate(labels):
            out.append(f'<text class="stacklab" x="{f.l+gw*(gi+0.5):.1f}" '
                       f'y="{f.Y(totals[gi])-8:.1f}" text-anchor="middle">{esc(txt)}</text>')
    if ytitle:
        out.append(f'<text class="axtitle" x="{f.l}" y="12">{esc(ytitle)}</text>')
    out.append("</svg>")
    return "".join(out)


def svg_waterfall(steps, *, w=760, h=400):
    """steps: list of dicts {label, value, note, state} - state in
    {'start','pass','fail'}. Connectors show each correction's contribution."""
    vals = [s["value"] for s in steps]
    lo, hi = min(vals + [0]), max(vals)
    f = Frame(w, h, 62, 22, 26, 92, (0, len(steps)), (lo - 0.35, hi + 0.35))
    out = [svg_open(w, h), gridlines(f, nice_ticks(lo, hi, 5), lambda v: fmt(v, 1))]
    gw = f.pw / len(steps)
    bw = min(78, gw - 34)
    zero = f.Y(0)
    prev = None
    for i, s in enumerate(steps):
        cx = f.l + gw * (i + 0.5)
        y = f.Y(s["value"])
        cls = {"start": "c1", "pass": "c1", "fail": "c3"}[s["state"]]
        top, hgt = min(y, zero), abs(y - zero)
        out.append(f'<rect class="bar {cls}" x="{cx-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                   f'height="{max(hgt,1):.1f}" rx="2.5" '
                   f'data-tip="{esc(f"{s['label']} · Sharpe {fmt(s['value'])}")}"/>')
        vy = y - 9 if s["value"] >= 0 else y + 17
        out.append(f'<text class="barval" x="{cx:.1f}" y="{vy:.1f}" text-anchor="middle">{fmt(s["value"])}</text>')
        if prev is not None:
            out.append(f'<line class="conn" x1="{prev[0]+bw/2:.1f}" y1="{prev[1]:.1f}" '
                       f'x2="{cx-bw/2:.1f}" y2="{prev[1]:.1f}"/>')
            delta = s["value"] - steps[i - 1]["value"]
            mx = (prev[0] + bw / 2 + cx - bw / 2) / 2
            out.append(f'<text class="delta" x="{mx:.1f}" y="{prev[1]-7:.1f}" text-anchor="middle">{fmt(delta, 2, True)}</text>')
        prev = (cx, y)
        for li, line in enumerate(s["label"].split("|")):
            out.append(f'<text class="wflab" x="{cx:.1f}" y="{f.t+f.ph+22+li*13:.1f}" text-anchor="middle">{esc(line)}</text>')
        badge = "clears the gate" if s["state"] in ("start", "pass") else "fails"
        bcls = "ok" if s["state"] in ("start", "pass") else "no"
        out.append(f'<text class="wfstate {bcls}" x="{cx:.1f}" y="{f.t+f.ph+22+len(s["label"].split("|"))*13+6:.1f}" '
                   f'text-anchor="middle">{esc(badge)}</text>')
    out.append("</svg>")
    return "".join(out)




# ── hand-authored diagrams ───────────────────────────────────────────────────

def diagram_crosssection():
    """One day's cross-section, ordered by predicted return. Schematic: the
    per-name alpha scores are not retained on disk, so the shape is drawn, not
    measured. Labelled as such on the slide."""
    w, h, n = 760, 300, 500
    rng = np.random.default_rng(42)
    base = np.linspace(1, -1, n)
    y = base * 1.0 + rng.normal(0, 0.075, n)
    y.sort()
    y = y[::-1]
    mid, half = h / 2, h / 2 - 46
    out = [svg_open(w, h)]
    out.append(f'<line class="axis" x1="26" y1="{mid}" x2="{w-26}" y2="{mid}"/>')
    for i in range(n):
        x = 26 + (w - 52) * i / (n - 1)
        yy = mid - y[i] * half
        cls = "c2" if i < 50 else ("c3" if i >= n - 50 else "muted-mark")
        out.append(f'<line class="xs {cls}" x1="{x:.1f}" y1="{mid}" x2="{x:.1f}" y2="{yy:.1f}"/>')
    lx = 26 + (w - 52) * 50 / (n - 1)
    rx = 26 + (w - 52) * (n - 51) / (n - 1)
    for x in (lx, rx):
        out.append(f'<line class="ref vert" x1="{x:.1f}" y1="14" x2="{x:.1f}" y2="{h-14}"/>')
    out.append(f'<text class="xslab c2" x="26" y="22">top decile — bought (50 names)</text>')
    out.append(f'<text class="xslab c3" x="{w-26}" y="{h-8}" text-anchor="end">bottom decile — sold short (50 names)</text>')
    out.append(f'<text class="xslab muted" x="{(lx+rx)/2:.1f}" y="{mid-10:.1f}" text-anchor="middle">the other 400 names are not held</text>')
    out.append(f'<text class="axtitle" x="26" y="{h-8}">500 stocks, ranked by predicted next-day return →</text>')
    out.append("</svg>")
    return "".join(out)


def diagram_tiers():
    tiers = [
        ("Tier 0", "Data", "load prices,<br/>validate, forward-fill"),
        ("Tier 1", "Trees", "17 features → three<br/>models → one score"),
        ("Tier 2", "Regime", "calm vs. stressed;<br/>de-risk gate"),
        ("Tier 3", "Execution", "weights, costs,<br/>realized return"),
        ("Tier 4", "Gate", "is the result<br/>credible?"),
    ]
    cells = "".join(
        f'<div class="tier"><span class="tno">{t}</span>'
        f'<span class="tname">{n}</span><span class="tbody">{b}</span></div>'
        f'{"<div class=\"tarrow\">→</div>" if i < len(tiers)-1 else ""}'
        for i, (t, n, b) in enumerate(tiers)
    )
    return f'<div class="tierflow">{cells}</div>'


def diagram_walkforward():
    w, h = 760, 310
    train, pred, step = 504, 63, 63
    total = train + step * 7
    rows = [0, 1, 2, 3, None, 37]
    f = Frame(w, h, 96, 20, 26, 40, (0, train + step * 8), (0, 1))
    out = [svg_open(w, h)]
    yy = 34
    for r in rows:
        if r is None:
            out.append(f'<text class="tick" x="{f.l+80}" y="{yy+12}">⋮</text>')
            yy += 34
            continue
        s = min(r, 4) * step
        x0, x1 = f.X(s), f.X(s + train)
        x2 = f.X(s + train + pred)
        out.append(f'<rect class="wf-train" x="{x0:.1f}" y="{yy}" width="{x1-x0:.1f}" height="22" rx="2.5"/>')
        out.append(f'<rect class="wf-pred" x="{x1+2:.1f}" y="{yy}" width="{x2-x1-2:.1f}" height="22" rx="2.5"/>')
        lab = f"fold {r+1}" if r < 4 else "fold 38"
        out.append(f'<text class="wflabel" x="{f.l-12}" y="{yy+15}" text-anchor="end">{lab}</text>')
        yy += 34
    out.append(f'<text class="wfkey wf-train-t" x="{f.l}" y="20">train — 504 days (2 years)</text>')
    out.append(f'<text class="wfkey wf-pred-t" x="{f.X(train)+10:.1f}" y="20">predict — 63 days, never seen</text>')
    out.append(f'<line class="axis" x1="{f.l}" y1="{yy+2}" x2="{f.l+f.pw}" y2="{yy+2}"/>')
    out.append(f'<text class="axtitle" x="{f.l}" y="{yy+22}">time →</text>')
    out.append(f'<text class="tick" x="{f.l+f.pw}" y="{yy+22}" text-anchor="end">2016 → 2025, 38 folds, 2,337 scored days</text>')
    out.append("</svg>")
    return "".join(out)


def diagram_tracks():
    tracks = [
        ("A", "Measure the cost", "half-spread estimated from<br/>daily OHLC; market impact<br/>charged per name"),
        ("B", "Verify the shorts", "point-in-time borrowable set<br/>from traded futures<br/>contracts"),
        ("D", "Correct the statistics", "honest trial count N = 30;<br/>Lo correction for serial<br/>correlation"),
    ]
    cards = "".join(
        f'<div class="track"><span class="tkid">Track {i}</span>'
        f'<span class="tkname">{n}</span><span class="tkbody">{b}</span></div>'
        for i, n, b in tracks
    )
    return f'<div class="trackrow">{cards}</div>'


def svg_dumbbell(rows, *, w=760, h=400, ylab=lambda v: fmt(v, 1), groups=None,
                 ytitle=None, ref=None):
    """One connector per cell from its gross Sharpe to its net Sharpe. The
    length of the connector is the cost, which is the quantity the slide is
    about. rows: list of {label, sub, gross, net}."""
    vals = [v for r in rows for v in (r["gross"], r["net"])]
    if ref:
        vals += [x["y"] for x in ref]
    lo, hi = min(vals + [0]), max(vals)
    pad = (hi - lo) * 0.12
    f = Frame(w, h, 62, 104, 26, 66, (0, len(rows)), (lo - pad, hi + pad))
    out = [svg_open(w, h), gridlines(f, nice_ticks(lo, hi, 5), ylab)]
    out.append(ref_marks(f, ref))
    gw = f.pw / len(rows)
    for i, r in enumerate(rows):
        cx = f.l + gw * (i + 0.5)
        yg, yn = f.Y(r["gross"]), f.Y(r["net"])
        neg = r["net"] < 0
        out.append(f'<line class="dbell" x1="{cx:.1f}" y1="{yg:.1f}" x2="{cx:.1f}" y2="{yn:.1f}"/>')
        out.append(f'<circle class="dot c1" cx="{cx:.1f}" cy="{yg:.1f}" r="5.5" '
                   f'data-tip="{esc(f"{r['label']} · gross Sharpe {fmt(r['gross'])}")}"/>')
        out.append(f'<circle class="dot {"c3" if neg else "c2"}" cx="{cx:.1f}" cy="{yn:.1f}" r="5.5" '
                   f'data-tip="{esc(f"{r['label']} · net Sharpe {fmt(r['net'])}")}"/>')
        gy, ny = yg + 4, yn + 4
        if abs(yg - yn) < 15:          # labels would overlap; push them apart
            gy, ny = yg - 4, yn + 13
        out.append(f'<text class="barval" x="{cx+10:.1f}" y="{gy:.1f}">{fmt(r["gross"])}</text>')
        out.append(f'<text class="barval {"c3" if neg else "c2"}" x="{cx+10:.1f}" y="{ny:.1f}">{fmt(r["net"])}</text>')
        out.append(f'<text class="tick" x="{cx:.1f}" y="{f.t+f.ph+20:.1f}" text-anchor="middle">{esc(r["label"])}</text>')
        if r.get("sub"):
            out.append(f'<text class="ticksub" x="{cx:.1f}" y="{f.t+f.ph+34:.1f}" text-anchor="middle">{esc(r["sub"])}</text>')
    for g in groups or []:
        x0 = f.l + gw * g["from"]
        x1 = f.l + gw * (g["to"] + 1)
        out.append(f'<line class="grpline" x1="{x0+6:.1f}" y1="{f.t+f.ph+44:.1f}" x2="{x1-6:.1f}" y2="{f.t+f.ph+44:.1f}"/>')
        out.append(f'<text class="grplab" x="{(x0+x1)/2:.1f}" y="{f.t+f.ph+58:.1f}" text-anchor="middle">{esc(g["label"])}</text>')
    if ytitle:
        out.append(f'<text class="axtitle" x="{f.l}" y="14">{esc(ytitle)}</text>')
    out.append("</svg>")
    return "".join(out)




# ── slide helpers ────────────────────────────────────────────────────────────

def stats(items):
    return '<div class="stats">' + "".join(
        f'<div class="stat"><span class="sv">{v}</span>'
        f'<span class="sl">{l}</span></div>' for l, v in items) + "</div>"


def table(headers, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in headers)
    tb = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table></div>'



SLIDES = []


def slide(**kw):
    SLIDES.append(kw)


# ── 1 · title and result ─────────────────────────────────────────────────────
lo21 = next(c for c in CELLS if c["style"] == "long_only" and c["target"] == "21-day")
g = weekly(cum(p4c_gross[lo21["sid"] + "__gross"]))
n_ = weekly(cum(p4c_net[lo21["sid"]]))
xs = list(range(len(g)))
years = [(i, str(d.year)) for i, d in enumerate(g.index)
         if d.year in (2016, 2018, 2020, 2022, 2024) and (i == 0 or g.index[i - 1].year != d.year)]
slide(
    layout="title",
    section="QRTF Engine",
    title="A cross-sectional equity system for the NSE,<br/>and what it is worth net of costs",
    body=f"""
<p class="lede">Machine-learning models rank 500 Indian stocks each day; the strategy buys
the best-ranked tenth and sells short the worst. Over 9.3 years the ranking has genuine
predictive skill. None of the six committed configurations is profitable enough to pay for
the trading it requires.</p>
{stats([("names ranked daily", "500"), ("distinct tickers", f"{DISTINCT_NAMES:,}"),
        ("trading days scored", f"{len(p4c_net):,}"), ("years", "9.27"),
        ("configurations that clear the gate", "0 of 6")])}
""",
    figure=svg_line(
        [{"name": "before costs", "x": xs, "y": list(g.values), "cls": "c1"},
         {"name": "after costs", "x": xs, "y": list(n_.values), "cls": "c2"}],
        w=880, h=235, ylog=True, ylab=lambda v: f"{v:g}×",
        ytickvals=[1, 2, 5, 10, 20], xticks=years,
        hover_fmt=lambda v: f"{v:.2f}×"),
    caption=("Growth of ₹1 in the best of the six configurations (long only, 21-day target), "
             "before and after measured transaction costs. Log scale. 2016-01-21 to 2025-06-27."),
    notes=("Open on the gap. Everything in the deck is an account of where that gap comes from "
           "and why it cannot be closed. The skill is real; the cost of harvesting it is larger."),
)

# ── 2 · what the strategy does ───────────────────────────────────────────────
slide(
    layout="split",
    section="The system",
    title="Ranking, not forecasting",
    body="""
<p>The system does not try to predict whether the market will rise or fall. Each trading day
it ranks all 500 stocks against one another by predicted return over the next few days.</p>
<p>It then buys the top 10% and <em>sells short</em> the bottom 10% — short selling means
borrowing a share, selling it, and buying it back later, which profits when the price falls.</p>
<p>Because the strategy is long and short at the same time, a general market move affects both
sides and largely cancels. What is left depends on whether the <em>ordering</em> was right.</p>
<p class="aside">This also means the strategy can make money in a falling market and lose money
in a rising one. It is a bet on relative performance only.</p>
""",
    figure=diagram_crosssection(),
    caption=("Schematic. One day's cross-section ordered by predicted return; the per-name "
             "scores are not retained on disk, so the shape is drawn rather than measured."),
    notes=("The single most common misunderstanding is that this forecasts the index. It does "
           "not. If asked what happens in a crash: both legs fall, and the return is the "
           "difference between them."),
)

# ── 3 · the five tiers ───────────────────────────────────────────────────────
slide(
    layout="wide",
    section="The system",
    title="Five stages, each a function of the one before",
    body=f"""
{diagram_tiers()}
<div class="two">
<div><h4>Two rules hold everywhere</h4>
<p>Missing prices are <strong>carried forward, never backward</strong>. Filling a gap with a
later price would let the strategy act on information that did not exist yet.</p>
<p>A signal formed on day T earns the return from day T to T+1. It is never credited with the
move that produced it.</p></div>
<div><h4>Why the last stage exists</h4>
<p>A backtest can always be made to look good by trying configurations until one works. Stage 4
measures how many were tried and raises the bar accordingly, so that a result which is merely
the luckiest of many attempts does not pass.</p></div>
</div>
{stats([("features per stock", "17"), ("models averaged", "3"),
        ("refits over the sample", "38"), ("configurations committed", "6"),
        ("attempts counted against the gate", "30")])}
""",
    figure="",
    caption="",
    notes=("Tier 2 deserves a caveat if asked: a two-state model only bisects volatility, so "
           "the usable output is a de-risk gate on the most stressed days, not a forecast."),
)

# ── 4 · walk-forward ─────────────────────────────────────────────────────────
slide(
    layout="split",
    section="The system",
    title="The models are never tested on what they were taught",
    body="""
<p>A model that is fitted and scored on the same period will always look skilful. This one is
refitted repeatedly through history.</p>
<p>Two years of data train three models — <span class="mono">LightGBM</span>,
<span class="mono">XGBoost</span> and a random forest. They then predict the following quarter,
which they have never seen. The window advances and everything is refitted.</p>
<p>Over the sample this happens <strong>38 times</strong>. Every scored day was predicted by a
model fitted only on days before it.</p>
<p class="aside">The three models' predictions are averaged. Averaging different learners is
more stable than trusting any single one.</p>
""",
    figure=diagram_walkforward(),
    caption=("Walk-forward estimation. Each row is one refit: the training window on the "
             "left, the out-of-sample quarter it then predicts on the right."),
    notes=("This is the difference between a backtest and a demonstration. If a professor "
           "presses on look-ahead, this slide plus the forward-fill rule on slide 3 is the answer."),
)

# ── 5 · measured skill ───────────────────────────────────────────────────────
ic_rows = ic_500.set_index("model")
ic68 = float(ic_68.set_index("model").loc["ensemble", "mean_IC"])
slide(
    layout="split",
    section="The system",
    title="The ranking does predict",
    body=f"""
<p>Skill is measured by the <strong>information coefficient</strong>: on each day, the
correlation between the predicted ordering and what actually happened. Zero means the ranking
is noise. In equities, 0.02–0.05 is a genuinely useful signal — the bar is low because so
little of a single stock's daily move is predictable at all.</p>
<p>The three-model average reaches <strong>{ic_rows.loc['ensemble','mean_IC']:.4f}</strong>,
and ranks in the right direction on
<strong>{ic_rows.loc['ensemble','hit_rate']*100:.0f}%</strong> of days across
{int(ic_rows.loc['ensemble','n_days']):,} days.</p>
<p>This is the finding the rest of the deck has to be read against. <strong>The signal is
real.</strong> Everything that follows is about what it costs to act on it.</p>
""",
    figure=svg_bars(
        ["LightGBM", "XGBoost", "random forest", "all three averaged", "earlier 68-stock\nuniverse"],
        [{"name": "mean information coefficient",
          "values": [float(ic_rows.loc[m, "mean_IC"]) for m in ("lgbm", "xgb", "rf", "ensemble")] + [ic68],
          "cls": "c1"}],
        w=660, h=380, ylab=lambda v: f"{v:.02f}".rstrip("0") or "0",
        value_fmt=lambda v: f"{v:.4f}"),
    caption=("Mean daily cross-sectional rank IC by model, 500-name universe, 5-day target, "
             "2,333 scored days. The rightmost bar is the earlier 68-stock universe for scale."),
    notes=("If asked why IC rose 2.5× when the universe widened: the honest answer is that the "
           "500-name universe contains a different and stronger cross-sectional effect, and "
           "slide 16 shows it is not confined to illiquid names."),
)

# ── 6 · phases 1 and 2 ───────────────────────────────────────────────────────
slide(
    layout="split",
    section="Building a valid test",
    title="Phases 1 and 2 — the apparatus",
    body="""
<p><strong>Phase 1</strong> ran the whole five-stage stack on <em>simulated</em> prices for ten
stocks, three of which were given a hidden upward drift. It confirmed that the machinery works
end to end.</p>
<p>It says nothing about markets. The simulator decided which stocks would win, so finding them
demonstrates only that the search works — which is exactly what it was built to check.</p>
<p><strong>Phase 2</strong> replaced the simulation with real NSE data, replaced the simple
momentum ranking with the tree models, and removed a look-ahead flaw in the regime detector,
which had been fitted on the whole history at once.</p>
""",
    figure=svg_bars(
        P1_PROFILES,
        [{"name": "long only", "values": P1["long only"], "cls": "c1"},
         {"name": "long / short", "values": P1["long / short"], "cls": "c2"},
         {"name": "dynamic tilt", "values": P1["dynamic tilt"], "cls": "c4"}],
        w=660, h=380, ylab=lambda v: f"{v:g}%", value_fmt=lambda v: fmt(v, 1) + "%"),
    legend_items=[("long only", "c1"), ("long / short", "c2"), ("dynamic tilt", "c4")],
    caption=("Phase 1, cumulative return by simulated market regime and trading style. "
             "Synthetic data: a verification of pipeline mechanics, not an empirical result."),
    notes=("Be direct that this chart is not evidence. It is included because the apparatus "
           "had to be verified before it could be trusted on real data."),
)

# ── 7 · phase 3 ──────────────────────────────────────────────────────────────
p3_rows = []
for f_ in P3_FREQ:
    for j, st in enumerate(P3_STYLE):
        p3_rows.append({"label": st.replace("long / short", "long/short").replace("dynamic tilt", "dyn. tilt"),
                        "gross": P3_GROSS[f_][j], "net": p3_net_sr[(f_, st)]})
slide(
    layout="figure",
    section="Building a valid test",
    title="Phase 3 — the first collapse",
    body="""
<p>Phase 3 charged the actual Indian statutory cost of trading — securities transaction tax,
exchange and regulator fees, stamp duty, GST and brokerage — about <strong>14.7 basis points
each time a position is opened or closed</strong> (a basis point is one hundredth of one
percent).</p>
<p>Faster trading means more of those charges. Every intraday configuration went to
<strong>−100%</strong>: the strategy trades so often that the fees exceed the edge with
certainty, not by bad luck. The only survivor was the daily long-only book, at a Sharpe ratio
of 0.57 — a measure of return per unit of risk, where above 1 is good and below 0 loses
money — which still fell short of the credibility gate.</p>
""",
    figure=svg_dumbbell(
        p3_rows, w=940, h=360, ytitle="annualized Sharpe ratio",
        groups=[{"from": 0, "to": 2, "label": "daily"}, {"from": 3, "to": 5, "label": "hourly"},
                {"from": 6, "to": 8, "label": "every 30 min"}, {"from": 9, "to": 11, "label": "every 15 min"}],
        ref=[{"y": 0, "label": "break even"}]),
    legend_items=[("before costs", "c1"), ("after costs, positive", "c2"), ("after costs, negative", "c3")],
    caption=("Each line runs from a configuration's Sharpe before costs to its Sharpe after "
             "costs. Gross values from the Phase 3 record; net values recomputed from the "
             "12-cell ledger. 68 stocks, 2018–2025."),
    notes=("The length of each line is the cost. Note the ordering inverts: the fastest "
           "strategies look best before costs and worst after. The conclusion carried into "
           "Phase 4 was that 68 stocks was too small a universe."),
)

# ── 8 · the breadth hypothesis ───────────────────────────────────────────────
slide(
    layout="split",
    section="Phase 4b",
    title="Widening the universe from 68 stocks to 500",
    body=f"""
<p>Standard portfolio theory says that skill of a given quality applied to more independent
bets produces a better risk-adjusted return, scaling roughly with the square root of the number
of bets. Going from 68 names to 500 predicts an improvement of about
<strong>2.7×</strong>, against the <strong>1.5×</strong> needed to pass.</p>
<p>Testing that required rebuilding the dataset from the exchange's public daily archive as a
<strong>point-in-time</strong> index membership: a company that belonged to the index in 2017
and later collapsed is still present in 2017.</p>
<p>Without this, a backtest quietly restricts itself to firms that survived — which is
information nobody had at the time. Over the sample, <strong>{DISTINCT_NAMES:,} different
companies</strong> passed through a universe that only ever holds 500 at once.</p>
""",
    figure=svg_bars(
        [c["year"] for c in CHURN],
        [{"name": "entered the index", "values": [c["in"] for c in CHURN], "cls": "c2"},
         {"name": "left the index", "values": [-c["out"] for c in CHURN], "cls": "c3"}],
        w=660, h=380, ylab=lambda v: f"{abs(v):g}", value_fmt=lambda v: fmt(abs(v), 0)),
    legend_items=[("entered the index", "c2"), ("left the index", "c3")],
    caption=("Membership churn per year, computed from the point-in-time universe mask. "
             "Names entering above the line, names leaving below."),
    notes=("If asked why churn matters: every name below the line is a company a "
           "survivorship-biased dataset would have silently deleted from its own past."),
)

# ── 9 · phase 4b results ─────────────────────────────────────────────────────
b_curves = {r["style"]: weekly(cum(p4b[r["sid"]])) for r in P4B}
nifty_curve = weekly(cum(nifty_ret))
bx = list(range(len(nifty_curve)))
byears = [(i, str(d.year)) for i, d in enumerate(nifty_curve.index)
          if d.year % 2 == 0 and (i == 0 or nifty_curve.index[i - 1].year != d.year)]
slide(
    layout="figure",
    section="Phase 4b",
    title="Phase 4b — two of three configurations passed",
    body=f"""
<p>On the widened universe the strategy performed as the breadth argument predicted, and better.
The long/short book returned <strong>{pct(P4B[1]['cum'], 1)}</strong> over 9.3 years at a Sharpe
of <strong>{fmt(P4B[1]['sr'])}</strong>, against <strong>{pct(NIFTY_CUM, 1)}</strong> for the
NIFTY-50 index. Two of the three configurations cleared the credibility gate.</p>
<p>This result was recorded as <strong>preliminary rather than accepted</strong>. Three
quantities the backtest relied on had not been measured; they are the subject of the next
slide, and of the phase that followed.</p>
{table(["configuration", "total return", "Sharpe", "worst drawdown", "gate"],
       [[r["style"].replace("_", " "), pct(r["cum"], 1), fmt(r["sr"]), pct(r["mdd"], 1),
         f'<span class="pill {"ok" if r["dsr"] > 0.95 else "no"}">{fmt(r["dsr"], 3)}</span>']
        for r in P4B] +
       [["NIFTY-50 index", pct(NIFTY_CUM, 1), fmt(NIFTY_SR), pct(NIFTY_DD, 1), "—"]])}
""",
    figure=svg_line(
        [{"name": "dynamic tilt", "x": bx, "y": list(b_curves["dynamic_tilt"].values), "cls": "c4"},
         {"name": "long / short", "x": bx, "y": list(b_curves["long_short"].values), "cls": "c1"},
         {"name": "long only", "x": bx, "y": list(b_curves["long_only"].values), "cls": "c2"},
         {"name": "NIFTY-50", "x": bx, "y": list(nifty_curve.values), "cls": "cn", "dash": "5 4"}],
        w=940, h=330, ylog=True, ylab=lambda v: f"{v:g}×",
        ytickvals=[1, 2, 4, 8, 16], xticks=byears, hover_fmt=lambda v: f"{v:.2f}×"),
    caption=("Growth of ₹1, computed from the Phase 4b ledger; index from the exchange archive. "
             "Log scale, so equal vertical distances are equal percentage gains. "
             "Costs charged here are statutory only."),
    notes=("Present this as a success, because that is how it was recorded at the time. The "
           "gate column is the DSR: two cells above 0.95. Then turn to what was missing."),
)

# ── 10 · the three unmeasured quantities ─────────────────────────────────────
slide(
    layout="split",
    section="Phase 4b",
    title="Three quantities the backtest had not measured",
    body="""
<ol class="numbered">
<li><strong>The cost of actually transacting was set to zero.</strong> Only the statutory taxes
and fees were charged. The bid–ask spread — the gap between the price to buy and the price to
sell — and the price movement caused by one's own order were both absent.</li>
<li><strong>Nobody had checked the short positions could be borrowed.</strong> Shorting in India
requires the stock to have a futures contract or an available lender. The worst-ranked names are
precisely where that is scarce.</li>
<li><strong>The trial count was understated.</strong> The credibility gate was told three
configurations had been tried. The true figure was far higher.</li>
</ol>
<p class="aside">The first of these decides the outcome. Each additional basis point of trading
cost removes roughly 2.1% a year from the long/short book, because it turns over most of its
positions every day.</p>
""",
    figure=svg_line(
        [{"name": "long / short", "x": SLIP_BPS, "y": SLIP["long / short"], "cls": "c1"},
         {"name": "dynamic tilt", "x": SLIP_BPS, "y": SLIP["dynamic tilt"], "cls": "c4"},
         {"name": "long only", "x": SLIP_BPS, "y": SLIP["long only"], "cls": "c2"}],
        w=660, h=380, ylab=lambda v: fmt(v, 1),
        xticks=[(b, str(b)) for b in SLIP_BPS],
        ref=[{"y": 0, "label": "break even"}], caption_pad=6),
    legend_items=[("passes the gate below ~7 bps", "c1")],
    caption=("Sharpe against additional trading cost, in basis points per transaction. "
             "The passing configurations stop passing at about 6.8 and 7.8 basis points — "
             "less than 2× headroom on a quantity that had not been measured. "
             "Horizontal axis: basis points."),
    notes=("The key number: under 2× headroom. That is what made the phase necessary rather "
           "than optional."),
)

# ── 11 · phase 4c method ─────────────────────────────────────────────────────
slide(
    layout="wide",
    section="Phase 4c",
    title="Phase 4c — measuring all three, and proving nothing else changed",
    body=f"""
{diagram_tracks()}
<div class="two">
<div><h4>The control that makes the comparison valid</h4>
<p>Every addition was written so that it is <strong>switched off by default</strong>. With the
switches off, the earlier runs were re-executed and had to reproduce their previous output
before any new figure was accepted.</p>
<p>They reproduced to the last decimal: a maximum difference of
<span class="mono">1.1 × 10⁻¹⁶</span> across all three Phase 4b configurations, which is the
rounding error of the arithmetic itself, and exactly <span class="mono">0.0</span> on the
Phase 3 ledger.</p></div>
<div><h4>Why that matters</h4>
<p>Rewriting a backtest and getting a different answer leaves two possible explanations: the new
measurement, or a mistake introduced while rewriting.</p>
<p>Because the old configuration still returns the old numbers exactly, every difference
reported from here on is attributable to the cost and eligibility models, and not to the
rewrite.</p>
<p class="aside">A further rule: a strategy restricted to borrowable shorts is treated as a
<em>different</em> strategy with its own entry in the trial count, not as a corrected version of
the old one. Running both and reporting the better is the error the gate exists to catch.</p></div>
</div>
{table(["re-run with the new measurements off", "columns", "largest difference from the published result"],
       [["Phase 3, daily, 5-day target", "3", "<span class='mono'>0.0</span> — bit for bit"],
        ["Phase 4b, 500 names, 5-day target", "3", "<span class='mono'>1.1 × 10⁻¹⁶</span> — floating-point rounding"]])}
""",
    figure="",
    caption="",
    notes=("This slide is the methodological core. If there is one thing to defend, it is that "
           "the regression gate ran before any new number was believed."),
)

# ── 12 · track A ─────────────────────────────────────────────────────────────
slide(
    layout="figure",
    section="Phase 4c",
    title="Track A — measuring the spread that had been assumed away",
    body=f"""
<p>The daily exchange archive records no bid and ask prices, so the spread has to be
<em>estimated</em> from the open, high, low and close. Two published estimators were used and
kept as a range, because they are known to err in opposite directions: Corwin–Schultz reads low
at <strong>{SPREAD_POOLED['cs']} bps</strong> and Abdi–Ranaldo reads high at
<strong>{SPREAD_POOLED['ar']} bps</strong>.</p>
<p>The strategy grid was run on the <strong>lower</strong> estimate deliberately. If the result
fails under the generous assumption, the conclusion cannot be attributed to a pessimistic
choice of estimator.</p>
<p>Both estimators rise monotonically as stocks become less liquid, which was the condition set
in advance for trusting them. Once charged, the real cost of a transaction is about
<strong>26 bps</strong> against the <strong>{STATUTORY_BPS:.1f} bps</strong> of statutory
charges — and the long/short Sharpe falls from <strong>1.78 to −0.23</strong>.</p>
""",
    figure=svg_bars(
        ["most liquid", "2nd", "3rd", "4th", "least liquid"],
        [{"name": "Corwin–Schultz (low estimate)", "values": SPREAD_Q["cs"], "cls": "c2"},
         {"name": "Abdi–Ranaldo (high estimate)", "values": SPREAD_Q["ar"], "cls": "c1"}],
        w=940, h=310, ylab=lambda v: f"{v:g}", ytitle="half-spread, basis points",
        ref=[{"y": STATUTORY_BPS, "label": f"statutory cost alone — {STATUTORY_BPS:.1f} bps"}],
        value_fmt=lambda v: fmt(v, 2) + " bps"),
    legend_items=[("Corwin–Schultz — used for the headline", "c2"), ("Abdi–Ranaldo", "c1")],
    caption=("Estimated half-spread by liquidity, both estimators, 500-name universe. "
             "From the Phase 4c measurement record (docs/phase4c_results.md §1)."),
    notes=("Two method points worth mentioning only if asked: between 39% and 76% of daily "
           "estimates round to zero, so the validation used the mean rather than the median; "
           "and the negative-estimate clip is applied once per window, because clipping each "
           "observation turns symmetric noise into upward bias."),
)

# ── 13 · track B ─────────────────────────────────────────────────────────────
slide(
    layout="figure",
    section="Phase 4c",
    title="Track B — most of the short book could not have been borrowed",
    body="""
<p>Whether a stock can be sold short was determined from the exchange's derivatives archive: a
name is treated as borrowable on a given day only if futures contracts in it actually traded
that day. Because it is derived from real trading, it carries no hindsight.</p>
<p>About a third of the universe qualifies. But the strategy does not want a random third — it
wants the <strong>50 worst-ranked names</strong>, and those are systematically the ones no
lender offers.</p>
<p>Averaged over the sample the strategy intended to short <strong>49 names</strong> a day and
could have borrowed <strong>11</strong>. Coverage never exceeds 32% in any year. Roughly
<strong>78% of the short book was never available</strong>, and because futures eligibility is
more generous than actual stock lending, even that is an upper bound.</p>
<p>Restricting the strategy to what it could have borrowed cuts the long/short book's Sharpe
<em>before costs</em> from <strong>4.20 to 1.88</strong>. More than half the raw signal lived in
positions that could not have been taken.</p>
""",
    figure=svg_stack(
        SHORT_YEARS,
        [{"name": "borrowable", "values": SHORT_BORROW, "cls": "c2"},
         {"name": "not borrowable", "values": [i - b for i, b in zip(SHORT_INTEND, SHORT_BORROW)],
          "cls": "c3"}],
        w=940, h=320, ylab=lambda v: f"{v:g}", ytitle="short positions the strategy wanted to hold",
        labels=[f"{c:.0f}% borrowable" for c in SHORT_COVER]),
    legend_items=[("could be borrowed", "c2"), ("could not be borrowed", "c3")],
    caption=("Intended short positions per year, split by whether the name was borrowable. "
             "From the Phase 4c feasibility record (docs/phase4c_results.md §6); the "
             "in-universe eligible count independently recomputes to 36% here."),
    notes=("This finding voids Phase 4b's two passing cells on its own, independently of cost. "
           "Both of them required a book that could not have been held."),
)

# ── 14 · track D ─────────────────────────────────────────────────────────────
lo_pairs = [("long / short", 1.78, 1.35, 0.986), ("dynamic tilt", 1.37, 1.20, 0.954),
            ("long only", 0.86, 0.77, 0.671)]
slide(
    layout="split",
    section="Phase 4c",
    title="Track D — counting the attempts honestly",
    body=f"""
<p>If thirty strategies are tested, the best of them will look impressive even when none has any
skill, simply because it is the maximum of thirty draws. The gate corrects for this by raising
the Sharpe a strategy must beat as the number of attempts grows.</p>
<p>Phase 4b had used <strong>N = 3</strong>. Phase 4c replaced this with a line-by-line tally of
every configuration the project had ever run: <strong>N = 30</strong>. The bar rises from
<strong>0.39</strong> to <strong>0.98</strong>.</p>
<p>A second correction was applied for the fact that overlapping multi-day forecasts make
consecutive returns related to one another, which flatters the Sharpe ratio.</p>
{table(["configuration", "Sharpe as reported", "after both corrections", "gate"],
       [[n, fmt(a), fmt(b), f'<span class="pill {"ok" if d > 0.95 else "no"}">{fmt(d,3)}</span>']
        for n, a, b, d in lo_pairs])}
<p class="aside">Applied on their own, the statistical corrections do <strong>not</strong>
overturn Phase 4b — the long/short book still passes at 0.986. The change of verdict comes
from the measured costs.</p>
""",
    figure=svg_line(
        [{"name": "bar the strategy must clear", "x": N_GRID, "y": SR_STAR_CURVE, "cls": "c1"}],
        w=660, h=380, ylab=lambda v: fmt(v, 1),
        xticks=[(n, str(n)) for n in (2, 10, 20, 30, 40, 50, 60)],
        ref=[{"y": 1.35, "label": "long/short, corrected — 1.35"}],
        caption_pad=6),
    legend_items=[("required Sharpe vs. number of attempts", "c1")],
    caption=("The deflation benchmark as a function of the number of configurations searched, "
             "computed with the repository's own estimator at the measured cross-trial "
             "dispersion. Horizontal axis: configurations tried."),
    notes=("The methods point, if it comes up: the textbook lag-by-lag correction was "
           "implemented first and rejected on measurement — on random data it returned "
           "adjustments between 0.99 and 1.18 depending only on the seed, so it was measuring "
           "its own noise. The variance-ratio form centres on 1.000 and recovers theory to 0.01."),
)

# ── 15 · the committed grid ──────────────────────────────────────────────────
grid_rows = []
for tgt in ("5-day", "21-day"):
    for style in ("long_only", "long_short_slb", "dynamic_tilt_slb"):
        c = next(x for x in CELLS if x["style"] == style and x["target"] == tgt)
        grid_rows.append({"label": style.replace("_slb", "").replace("_", " "),
                          "sub": f'DSR {fmt(c["dsr"], 3)}', "gross": c["gross_sr"], "net": c["net_sr"]})
slide(
    layout="figure",
    section="Phase 4c",
    title="The committed grid — none of the six passes",
    body=f"""
<p>Six configurations were fixed in writing <em>before</em> the run: three trading styles, each
with a five-day and a twenty-one-day forecast horizon. Fixing them in advance is what stops the
search from continuing until something works.</p>
<p>All six fail. The best reaches a credibility score of <strong>{fmt(best['dsr'],3)}</strong>
against a threshold of <strong>0.95</strong> — under 2% of the confidence required, not a near
miss. For every one of the six, the minimum track record needed to reach credibility is
<strong>undefined</strong>: none clears the bar at any length of history, so collecting more
data would not change the answer.</p>
<p>The slower forecast horizon did what it was intended to do — turnover fell from
<strong>0.45 to 0.24</strong> of the book per day and the net Sharpe rose from
<strong>0.18 to 0.30</strong>. It was the one remaining lever with room in it, and it moved the
result by 0.12 against a gate that needed far more.</p>
""",
    figure=svg_dumbbell(
        grid_rows, w=940, h=355, ytitle="annualized Sharpe ratio",
        groups=[{"from": 0, "to": 2, "label": "5-day forecast"},
                {"from": 3, "to": 5, "label": "21-day forecast"}],
        ref=[{"y": 0, "label": "break even"}]),
    legend_items=[("before costs", "c1"), ("after costs, positive", "c2"), ("after costs, negative", "c3")],
    caption=("Sharpe before and after measured costs for each committed configuration, with its "
             "credibility score. Computed from the Phase 4c ledger and gate output."),
    notes=("Every gross value is comfortably positive and every net value is at or below 0.3. "
           "The line length is the cost of harvesting the signal."),
)

# ── 16 · findings ────────────────────────────────────────────────────────────
slide(
    layout="figure",
    section="Findings",
    title="What each correction was worth",
    body=f"""
<div class="findings">
<div><h4>The signal is real, and it is not a small-stock illusion</h4>
<p>Skill is flat across the liquidity spectrum — an information coefficient of
{IC_QUINT[0]:.3f} in the most traded fifth of the universe and {IC_QUINT[4]:.3f} in the least.
The leading suspicion, that the edge was an artefact of thinly traded names, is ruled out. The
strategy fails because a genuine 0.04 does not cover 26 basis points a side while replacing a
quarter to three quarters of the book every day.</p></div>
<div><h4>The signal is fading</h4>
<p>Its statistical significance falls from about 10 in 2016–2019 to <strong>1.21</strong> in
2025, at which point it is indistinguishable from zero.</p>
{svg_bars(IC_YEARS, [{"name": "significance of the daily ranking", "values": IC_T, "cls": "c1"}],
          w=560, h=130, ylab=lambda v: f"{v:g}", value_fmt=lambda v: fmt(v, 2),
          ref=[{"y": 2.0, "label": "significance|threshold"}])}</div>
<div><h4>The breadth argument is settled, negatively</h4>
<p>Widening the universe from 68 names to 500 produced the predicted improvement before costs.
It also raised turnover from 0.32 to 0.45 of the book per day and extended the strategy into
stocks with wider spreads. Breadth bought signal and cost together.</p></div>
<div><h4>What this does not rule out</h4>
<p>A portfolio construction that treats cost as part of the objective rather than a charge
applied afterwards; and neutralizing market and sector exposure, deliberately excluded here
because it adds a degree of freedom that requires its own committed prior and its own trial
count.</p>
<p class="aside">Each correction was applied to every configuration alike and regardless of
whether it improved the result. That is the basis on which these figures are reported.</p></div>
</div>
""",
    body_class="body",
    figure=svg_waterfall([
        {"label": "as reported|in Phase 4b", "value": 1.78, "state": "start"},
        {"label": "honest trial count|and serial correlation", "value": 1.35, "state": "pass"},
        {"label": "measured spread|and market impact", "value": -0.23, "state": "fail"},
        {"label": "short book restricted|to what could be borrowed", "value": -0.99, "state": "fail"},
    ], w=940, h=330),
    legend_items=[("clears the credibility gate", "c1"), ("fails", "c3")],
    caption=("The long/short configuration, Phase 4b to Phase 4c, one correction at a time. "
             "Vertical axis: annualized Sharpe ratio."),
    notes=("Close here. The statistical corrections, which were expected to be decisive, were "
           "the mildest of the three; the measured cost of trading was what settled it."),
)

print(f"  built {len(SLIDES)} slides")


# ── page ─────────────────────────────────────────────────────────────────────

CSS = """
:root{
  --ground:#EEF0F5; --surface:#FFFFFF; --sunken:#F6F7FA;
  --ink:#111621; --ink2:#414B5E; --muted:#737E93;
  --hair:#DCE1EA; --hair2:#EBEEF4;
  --c1:#3B4E9B; --c2:#00795A; --c3:#B03A2E; --c4:#9C5B00; --cn:#8B93A3;
  --c1-soft:#E4E8F5; --c2-soft:#DDEDE7; --c3-soft:#F6E3E0;
  --ok:#00795A; --no:#B03A2E;
  --serif:"Source Serif 4",Georgia,"Times New Roman",serif;
  --sans:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,monospace;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0D1117; --surface:#161B22; --sunken:#1B2029;
    --ink:#E8ECF4; --ink2:#AAB4C6; --muted:#7C879A;
    --hair:#2A313D; --hair2:#212832;
    --c1:#6E86D8; --c2:#2E9E80; --c3:#D9695C; --c4:#BE8A38; --cn:#79839A;
    --c1-soft:#1D2740; --c2-soft:#142C26; --c3-soft:#33201E;
    --ok:#2E9E80; --no:#D9695C;
  }
}
:root[data-theme="dark"]{
  --ground:#0D1117; --surface:#161B22; --sunken:#1B2029;
  --ink:#E8ECF4; --ink2:#AAB4C6; --muted:#7C879A;
  --hair:#2A313D; --hair2:#212832;
  --c1:#6E86D8; --c2:#2E9E80; --c3:#D9695C; --c4:#BE8A38; --cn:#79839A;
  --c1-soft:#1D2740; --c2-soft:#142C26; --c3-soft:#33201E;
  --ok:#2E9E80; --no:#D9695C;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:var(--sans);font-size:15px;line-height:1.55;
  -webkit-font-smoothing:antialiased}
.deck{min-height:100vh;display:flex;flex-direction:column;align-items:center;
  justify-content:flex-start;padding:0}
.slide{display:none;width:100%;max-width:1360px;min-height:100vh;
  background:var(--surface);padding:clamp(22px,3.2vw,50px) clamp(22px,3.6vw,60px) 74px;
  flex-direction:column;gap:clamp(14px,1.6vw,22px)}
.slide.on{display:flex}
.shead{display:flex;justify-content:space-between;align-items:baseline;
  border-bottom:1px solid var(--hair);padding-bottom:9px;flex:0 0 auto}
.eyebrow{font-size:11px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--muted);font-weight:600}
.sno{font-family:var(--mono);font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(30px,3.5vw,52px);
  line-height:1.14;margin:0;text-wrap:balance;letter-spacing:-.012em}
h2{font-family:var(--serif);font-weight:600;font-size:clamp(23px,2.35vw,36px);
  line-height:1.18;margin:0;text-wrap:balance;letter-spacing:-.01em;flex:0 0 auto}
h4{font-family:var(--sans);font-size:12px;font-weight:650;letter-spacing:.07em;
  text-transform:uppercase;color:var(--ink2);margin:0 0 5px}
p{margin:0 0 .68em;color:var(--ink2);max-width:66ch}
p:last-child{margin-bottom:0}
strong{color:var(--ink);font-weight:620}
em{font-style:italic}
.lede{font-family:var(--serif);font-size:clamp(16px,1.35vw,20px);line-height:1.5;
  color:var(--ink2);max-width:62ch}
.aside{border-left:2px solid var(--hair);padding-left:12px;font-size:13.5px;color:var(--muted)}
.mono{font-family:var(--mono);font-size:.92em}
.body{flex:1 1 auto;min-height:0}

/* layouts */
.split{display:grid;grid-template-columns:minmax(0,0.86fr) minmax(0,1.14fr);
  gap:clamp(20px,2.6vw,44px);align-items:start;flex:1 1 auto;min-height:0}
.figwrap{display:flex;flex-direction:column;gap:9px;min-width:0}
.figfull{flex:1 1 auto;min-height:0;display:flex;flex-direction:column;gap:9px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:clamp(20px,2.6vw,40px);margin-top:6px}
.findings{display:grid;grid-template-columns:1fr 1fr;gap:14px clamp(20px,2.4vw,38px);margin-top:4px}
.findings p{font-size:13.5px;margin-bottom:.5em}
.tcols{columns:2;column-gap:clamp(20px,2.6vw,40px)}
.tcols p,.tcols .tw{break-inside:avoid}

/* figures */
svg.fig{width:100%;height:auto;display:block;overflow:visible}
.caption{font-size:11.5px;line-height:1.45;color:var(--muted);max-width:88ch}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12px;color:var(--ink2)}
.lg{display:inline-flex;align-items:center;gap:6px}
.sw{width:11px;height:11px;border-radius:2.5px;display:inline-block;flex:0 0 auto}
.sw.c1{background:var(--c1)} .sw.c2{background:var(--c2)}
.sw.c3{background:var(--c3)} .sw.c4{background:var(--c4)}
.sw.cn{background:var(--cn)}
.sw.c3d{background:transparent;border:1.5px dashed var(--c3);border-radius:0;height:0;width:14px}

/* svg elements - every colour comes from a token so both themes resolve */
.grid{stroke:var(--hair2);stroke-width:1}
.axis{stroke:var(--hair);stroke-width:1.25}
.ref{stroke:var(--muted);stroke-width:1;stroke-dasharray:4 3;opacity:.75}
.ref.vert{stroke-dasharray:3 3;opacity:.5}
.reflab{fill:var(--muted);font-size:10.5px;font-family:var(--sans)}
.tick{fill:var(--muted);font-size:11px;font-family:var(--sans);font-variant-numeric:tabular-nums}
.ticksub{fill:var(--muted);font-size:9.5px;font-family:var(--mono)}
.axtitle{fill:var(--muted);font-size:10.5px;font-family:var(--sans);letter-spacing:.04em}
.ln{fill:none;stroke-linejoin:round;stroke-linecap:round}
.ln.c1{stroke:var(--c1)} .ln.c2{stroke:var(--c2)} .ln.c3{stroke:var(--c3)}
.ln.c4{stroke:var(--c4)} .ln.cn{stroke:var(--cn)}
.dot.c1{fill:var(--c1)} .dot.c2{fill:var(--c2)} .dot.c3{fill:var(--c3)}
.dot.c4{fill:var(--c4)} .dot.cn{fill:var(--cn)}
.endlab{font-size:11.5px;font-family:var(--sans);font-weight:600}
.endlab.c1{fill:var(--c1)} .endlab.c2{fill:var(--c2)} .endlab.c3{fill:var(--c3)}
.endlab.c4{fill:var(--c4)} .endlab.cn{fill:var(--cn)}
.bar{stroke:var(--surface);stroke-width:2}
.bar.c1{fill:var(--c1)} .bar.c2{fill:var(--c2)} .bar.c3{fill:var(--c3)} .bar.c4{fill:var(--c4)}
.barval{fill:var(--ink2);font-size:11px;font-family:var(--mono);font-variant-numeric:tabular-nums}
.barval.c2{fill:var(--c2)} .barval.c3{fill:var(--c3)}
.hit{fill:transparent}
.dbell{stroke:var(--hair);stroke-width:6;stroke-linecap:round}
.conn{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3}
.delta{fill:var(--muted);font-size:11px;font-family:var(--mono)}
.wflab{fill:var(--ink2);font-size:11px;font-family:var(--sans)}
.wfstate{font-size:10px;font-family:var(--sans);letter-spacing:.05em;text-transform:uppercase}
.wfstate.ok{fill:var(--ok)} .wfstate.no{fill:var(--no)}
.grpline{stroke:var(--hair);stroke-width:1}
.grplab{fill:var(--muted);font-size:11px;font-family:var(--sans);letter-spacing:.05em}
.stacklab{fill:var(--muted);font-size:10.5px;font-family:var(--mono)}
.xs{stroke-width:1.15}
.xs.c2{stroke:var(--c2)} .xs.c3{stroke:var(--c3)}
.xs.muted-mark{stroke:var(--hair);opacity:.9}
.xslab{font-size:11.5px;font-family:var(--sans);font-weight:600}
.xslab.c2{fill:var(--c2)} .xslab.c3{fill:var(--c3)} .xslab.muted{fill:var(--muted);font-weight:400}
.wf-train{fill:var(--c1)} .wf-pred{fill:var(--c2)}
.wf-train-t{fill:var(--c1);font-size:11.5px;font-weight:600;font-family:var(--sans)}
.wf-pred-t{fill:var(--c2);font-size:11.5px;font-weight:600;font-family:var(--sans)}
.wflabel{fill:var(--muted);font-size:11px;font-family:var(--mono)}

/* stats, tables, lists */
.stats{display:flex;flex-wrap:wrap;gap:0;border-top:1px solid var(--hair);margin-top:16px}
.stat{flex:1 1 132px;padding:12px 16px 12px 0;border-right:1px solid var(--hair);
  display:flex;flex-direction:column;gap:1px}
.stat:last-child{border-right:none}
.stat:not(:first-child){padding-left:16px}
.sv{font-family:var(--serif);font-size:clamp(20px,2vw,29px);font-weight:600;
  color:var(--ink);font-variant-numeric:tabular-nums;line-height:1.1}
.sl{font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
.tw{overflow-x:auto;margin-top:12px}
table{border-collapse:collapse;width:100%;font-size:12.5px;
  font-variant-numeric:tabular-nums}
th{text-align:right;font-weight:600;font-size:10.5px;letter-spacing:.06em;
  text-transform:uppercase;color:var(--muted);padding:0 10px 6px 0;
  border-bottom:1px solid var(--hair);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td{padding:6px 10px 6px 0;border-bottom:1px solid var(--hair2);color:var(--ink2);
  text-align:right;white-space:nowrap}
tbody tr:last-child td{border-bottom:none}
td:first-child{color:var(--ink);font-weight:550}
.pill{font-family:var(--mono);font-size:11px;padding:2px 7px;border-radius:3px;
  display:inline-block}
.pill.ok{background:var(--c2-soft);color:var(--ok)}
.pill.no{background:var(--c3-soft);color:var(--no)}
ol.numbered{margin:0;padding-left:0;list-style:none;counter-reset:n;
  display:flex;flex-direction:column;gap:11px}
ol.numbered li{counter-increment:n;padding-left:34px;position:relative;
  color:var(--ink2);font-size:14px}
ol.numbered li::before{content:counter(n);position:absolute;left:0;top:1px;
  font-family:var(--mono);font-size:11px;color:var(--muted);
  border:1px solid var(--hair);border-radius:50%;width:22px;height:22px;
  display:grid;place-items:center}

/* diagrams */
.tierflow{display:flex;align-items:stretch;gap:6px;margin:6px 0 10px;flex-wrap:wrap}
.tier{flex:1 1 150px;background:var(--sunken);border-radius:5px;padding:14px 15px;
  display:flex;flex-direction:column;gap:3px;border-top:2.5px solid var(--c1)}
.tier:nth-child(3){border-top-color:var(--c1)}
.tier:nth-child(5){border-top-color:var(--c2)}
.tier:nth-child(7){border-top-color:var(--c4)}
.tier:nth-child(9){border-top-color:var(--c3)}
.tno{font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:.05em}
.tname{font-family:var(--serif);font-size:19px;font-weight:600;color:var(--ink);line-height:1.15}
.tbody{font-size:12.5px;color:var(--ink2);line-height:1.4}
.tarrow{align-self:center;color:var(--muted);font-size:15px;flex:0 0 auto}
.trackrow{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:4px 0 12px}
.track{background:var(--sunken);border-radius:5px;padding:15px 16px;
  border-left:2.5px solid var(--c1);display:flex;flex-direction:column;gap:3px}
.track:nth-child(2){border-left-color:var(--c2)}
.track:nth-child(3){border-left-color:var(--c4)}
.tkid{font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:.05em}
.tkname{font-family:var(--serif);font-size:18px;font-weight:600;color:var(--ink)}
.tkbody{font-size:12.5px;color:var(--ink2);line-height:1.45}

/* chrome */
.hud{position:fixed;left:0;right:0;bottom:0;height:44px;display:flex;
  align-items:center;justify-content:space-between;gap:14px;
  padding:0 clamp(14px,2vw,28px);background:var(--surface);
  border-top:1px solid var(--hair);z-index:40;font-size:11.5px;color:var(--muted)}
.hud-mid{display:flex;align-items:center;gap:10px}
.navbtn{border:1px solid var(--hair);background:var(--surface);color:var(--ink2);
  border-radius:4px;width:28px;height:26px;cursor:pointer;font-size:13px;
  display:grid;place-items:center;font-family:var(--sans)}
.navbtn:hover{background:var(--sunken);color:var(--ink)}
.navbtn:focus-visible,.dotnav button:focus-visible{outline:2px solid var(--c1);outline-offset:2px}
.count{font-family:var(--mono);font-variant-numeric:tabular-nums;color:var(--ink2)}
.dotnav{display:flex;gap:3px}
.dotnav button{width:15px;height:14px;padding:0;border:none;background:transparent;cursor:pointer}
.dotnav button i{display:block;height:3px;background:var(--hair);border-radius:2px}
.dotnav button[aria-current="true"] i{background:var(--c1)}
.prog{position:fixed;top:0;left:0;height:2px;background:var(--c1);z-index:50;transition:width .18s ease}
.keyhint{font-family:var(--mono);font-size:10.5px}
kbd{font-family:var(--mono);font-size:10px;border:1px solid var(--hair);
  border-radius:3px;padding:1px 4px;color:var(--ink2)}
.notes{display:none;border-top:1px solid var(--hair);padding-top:10px;margin-top:auto;
  font-size:12.5px;color:var(--muted);max-width:88ch}
.notes strong{color:var(--ink2)}
body.shownotes .notes{display:block}
#tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;
  background:var(--ink);color:var(--surface);font-family:var(--mono);font-size:11px;
  padding:5px 9px;border-radius:4px;z-index:60;max-width:340px;line-height:1.45}
#tip.on{opacity:1}
@media (max-width:900px){
  .split,.two,.findings,.trackrow{grid-template-columns:1fr}
  .tcols{columns:1}
  .slide{min-height:auto;padding-bottom:60px}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
@media print{
  .hud,.prog{display:none}
  .slide{display:flex!important;page-break-after:always;min-height:auto;
    box-shadow:none;max-width:none;padding:24px}
  body{background:#fff}
  .notes{display:none}
}
"""

JS = """
const slides=[...document.querySelectorAll('.slide')];
const dots=[...document.querySelectorAll('.dotnav button')];
const prog=document.querySelector('.prog');
const count=document.querySelector('.count');
let i=0;
function show(n,push){
  i=Math.max(0,Math.min(slides.length-1,n));
  slides.forEach((s,k)=>s.classList.toggle('on',k===i));
  dots.forEach((d,k)=>d.setAttribute('aria-current',k===i?'true':'false'));
  prog.style.width=((i+1)/slides.length*100)+'%';
  count.textContent=String(i+1).padStart(2,'0')+' / '+slides.length;
  if(push!==false)history.replaceState(null,'','#/'+(i+1));
  window.scrollTo(0,0);
}
document.addEventListener('keydown',e=>{
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  if(e.key==='ArrowRight'||e.key==='PageDown'||e.key===' '){e.preventDefault();show(i+1)}
  else if(e.key==='ArrowLeft'||e.key==='PageUp'){e.preventDefault();show(i-1)}
  else if(e.key==='Home'){e.preventDefault();show(0)}
  else if(e.key==='End'){e.preventDefault();show(slides.length-1)}
  else if(e.key==='n'||e.key==='N'){document.body.classList.toggle('shownotes')}
});
document.querySelector('.prev').onclick=()=>show(i-1);
document.querySelector('.next').onclick=()=>show(i+1);
document.querySelector('.notesbtn').onclick=()=>document.body.classList.toggle('shownotes');
dots.forEach((d,k)=>d.onclick=()=>show(k));
const m=location.hash.match(/^#\\/(\\d+)$/);
show(m?parseInt(m[1],10)-1:0,false);

// one delegated tooltip for every mark that carries data-tip
const tip=document.getElementById('tip');
let cross=null;
document.addEventListener('mouseover',e=>{
  const t=e.target.closest('[data-tip]');
  if(!t){tip.classList.remove('on');if(cross){cross.remove();cross=null}return}
  tip.textContent=t.getAttribute('data-tip');
  tip.classList.add('on');
  if(t.classList.contains('hit')){
    const svg=t.ownerSVGElement;
    if(cross)cross.remove();
    cross=document.createElementNS('http://www.w3.org/2000/svg','line');
    cross.setAttribute('x1',t.dataset.x);cross.setAttribute('x2',t.dataset.x);
    cross.setAttribute('y1',t.dataset.y0);cross.setAttribute('y2',t.dataset.y1);
    cross.setAttribute('class','ref');cross.setAttribute('stroke-dasharray','2 2');
    svg.appendChild(cross);
  }
});
document.addEventListener('mousemove',e=>{
  if(!tip.classList.contains('on'))return;
  const r=tip.getBoundingClientRect();
  let x=e.clientX+14,y=e.clientY+16;
  if(x+r.width>innerWidth-8)x=e.clientX-r.width-14;
  if(y+r.height>innerHeight-8)y=e.clientY-r.height-14;
  tip.style.left=x+'px';tip.style.top=y+'px';
});
document.addEventListener('mouseleave',()=>{tip.classList.remove('on')});
"""


def render_figure(s):
    if not s.get("figure"):
        return ""
    bits = [s["figure"]]
    if s.get("legend_items"):
        bits.append(legend(s["legend_items"]))
    if s.get("caption"):
        bits.append(f'<p class="caption">{s["caption"]}</p>')
    return "".join(bits)


def render_slide(s, n, total):
    head = (f'<div class="shead"><span class="eyebrow">{esc(s["section"])}</span>'
            f'<span class="sno">{n:02d} / {total}</span></div>')
    notes = f'<div class="notes"><strong>Presenter note.</strong> {s["notes"]}</div>' if s.get("notes") else ""
    lay = s["layout"]
    if lay == "title":
        inner = (f'{head}<h1>{s["title"]}</h1><div class="body">{s["body"]}</div>'
                 f'<div class="figfull">{render_figure(s)}</div>')
    elif lay == "split":
        inner = (f'{head}<h2>{s["title"]}</h2><div class="split">'
                 f'<div class="body">{s["body"]}</div>'
                 f'<div class="figwrap">{render_figure(s)}</div></div>')
    elif lay == "figure":
        inner = (f'{head}<h2>{s["title"]}</h2>'
                 f'<div class="figfull">{render_figure(s)}</div>'
                 f'<div class="{s.get("body_class", "body tcols")}">{s["body"]}</div>')
    else:
        inner = f'{head}<h2>{s["title"]}</h2><div class="body">{s["body"]}</div>'
    return f'<section class="slide" id="s{n}">{inner}{notes}</section>'


def build():
    total = len(SLIDES)
    body = "".join(render_slide(s, i + 1, total) for i, s in enumerate(SLIDES))
    dotnav = "".join(f'<button type="button" aria-label="slide {i+1}"><i></i></button>'
                     for i in range(total))
    page = f"""<title>Ranking 500 Stocks, Net of Costs</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;650&family=Source+Serif+4:opsz,wght@8..60,600&display=swap">
<style>{CSS}</style>
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
<script>{JS}</script>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"  wrote {OUT.relative_to(ROOT)}  ({len(page)/1024:.0f} KB)")


if __name__ == "__main__":
    build()

    # ── verification ─────────────────────────────────────────────────────────────
    print("\nverification — computed values against the research record")
    checks = [
        ("Phase 4b long_only cum", pct(P4B[0]["cum"], 4), "+308.4341%"),
        ("Phase 4b long_short cum", pct(P4B[1]["cum"], 4), "+639.2251%"),
        ("Phase 4b dynamic_tilt cum", pct(P4B[2]["cum"], 4), "+1194.7144%"),
        ("Phase 4b long_short Sharpe", fmt(P4B[1]["sr"], 3), "1.778"),
        ("Phase 4c best net Sharpe", fmt(best["net_sr"], 2), "0.30"),
        ("Phase 4c best DSR", fmt(best["dsr"], 3), "0.017"),
        ("Phase 4c SR* (N=30)", fmt(SR_STAR, 2), "0.98"),
        ("turnover 5b long_only", fmt(next(c for c in CELLS if c["style"] == "long_only" and c["target"] == "5-day")["turnover"], 2), "0.45"),
        ("turnover 21b long_only", fmt(lo21["turnover"], 2), "0.24"),
        ("cells passing the gate", str(sum(c["dsr"] > 0.95 for c in CELLS)), "0"),
    ]
    for label, got, expect in checks:
        ok = got.replace("−", "-").lstrip("+") == expect.replace("−", "-").lstrip("+")
        print(f"  {'ok ' if ok else 'DIFF'}  {label:32} {got:>14}   record: {expect}")
