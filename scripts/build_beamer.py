#!/usr/bin/env python3
"""Build the QRTF Engine deck as a LaTeX Beamer presentation.

Reads the same data layer as the HTML deck (``deck_data``) and the same prose
(``build_deck.SLIDES``), so the words and the numbers have one definition each.
Figures are re-authored in pgfplots/TikZ rather than reused from the SVG.

    python scripts/build_beamer.py           # writes and compiles the PDF
    python scripts/build_beamer.py --no-pdf  # emit the .tex only

Compiles with tectonic (XeTeX), using fonts that ship with macOS.
"""
from __future__ import annotations

import html as _html
import re
import shutil
import subprocess
import sys

from deck_data import *  # noqa: F403  - loaders, statistics and every computed figure
import build_deck as H

TEX = ROOT / "docs" / "presentation" / "qrtf_deck.tex"
PDF = TEX.with_suffix(".pdf")

# ── palette (the validated light-mode set from the HTML deck) ────────────────
PREAMBLE = r"""
\documentclass[aspectratio=169,10pt]{beamer}
\usetheme{default}
\usepackage{fontspec}
\usepackage{pgfplots}
\usepackage{booktabs}
\usepackage{array}
\pgfplotsset{compat=1.18}
\usetikzlibrary{calc,positioning,backgrounds}

\setmainfont{Charter}
\setsansfont{Avenir Next}
\setmonofont{Menlo}[Scale=0.84]
\newfontfamily\dispfont{Charter}

\definecolor{ink}{HTML}{111621}
\definecolor{ink2}{HTML}{414B5E}
\definecolor{muted}{HTML}{737E93}
\definecolor{hair}{HTML}{DCE1EA}
\definecolor{hairb}{HTML}{EBEEF4}
\definecolor{sunken}{HTML}{F6F7FA}
\definecolor{cA}{HTML}{3B4E9B}
\definecolor{cB}{HTML}{00795A}
\definecolor{cC}{HTML}{B03A2E}
\definecolor{cD}{HTML}{9C5B00}
\definecolor{cN}{HTML}{8B93A3}
\definecolor{cAsoft}{HTML}{E4E8F5}
\definecolor{cBsoft}{HTML}{DDEDE7}
\definecolor{cCsoft}{HTML}{F6E3E0}

\setbeamercolor{normal text}{fg=ink2,bg=white}
\setbeamercolor{frametitle}{fg=ink,bg=white}
\setbeamercolor{structure}{fg=cA}
\setbeamerfont{frametitle}{family=\dispfont,series=\bfseries,size=\large}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{itemize item}{\color{muted}\textbullet}
\setbeamersize{text margin left=10mm,text margin right=10mm}
\beamertemplatenavigationsymbolsempty
\AtBeginDocument{\raggedright}

% frame title: section eyebrow and rule above, title below
\makeatletter
\newcommand{\eyebrow}[1]{\def\@eyebrow{#1}}
\eyebrow{}
\setbeamertemplate{frametitle}{%
  \vskip2mm
  {\sffamily\fontsize{6.4}{7}\selectfont\color{muted}\MakeUppercase{\@eyebrow}}%
  \hfill{\ttfamily\fontsize{6.4}{7}\selectfont\color{muted}\insertframenumber\,/\,\inserttotalframenumber}\par
  \vskip0.8mm{\color{hair}\hrule height0.4pt}\vskip2.6mm
  {\usebeamerfont{frametitle}\usebeamercolor[fg]{frametitle}\insertframetitle}\par\vskip1mm}
\makeatother
\setbeamertemplate{footline}{}

% text styles
\newcommand{\lede}[1]{{\rmfamily\fontsize{10}{13.4}\selectfont\color{ink2}#1\par}}
\newcommand{\aside}[1]{{\setlength{\leftskip}{3mm}\sffamily\fontsize{7.4}{10}\selectfont
  \color{muted}\hspace*{-3mm}\rule[-0.6mm]{0.5pt}{3.4mm}\hspace{2mm}#1\par}}
\newcommand{\bodytext}[1]{{\sffamily\fontsize{7.4}{9.7}\selectfont\color{ink2}#1\par}}
\newcommand{\subhead}[1]{{\sffamily\fontsize{6.6}{9}\selectfont\bfseries\color{ink2}%
  \MakeUppercase{#1}\par\vskip0.6mm}}
\newcommand{\figcaption}[1]{{\sffamily\fontsize{6.2}{8}\selectfont\color{muted}#1\par}}
\newcommand{\str}[1]{{\color{ink}\bfseries #1}}
\newcommand{\mn}[1]{{\ttfamily #1}}
\newcommand{\rupee}{Rs\,}
\newcommand{\swatch}[1]{\tikz[baseline=-0.5mm]\fill[#1,rounded corners=0.3pt](0,0)rectangle(1.6mm,1.6mm);}
\newcommand{\legendline}[1]{{\sffamily\fontsize{6.4}{8}\selectfont\color{ink2}#1\par}}
\newcommand{\pillok}[1]{\colorbox{cBsoft}{\color{cB}\ttfamily\fontsize{6.4}{7}\selectfont #1}}
\newcommand{\pillno}[1]{\colorbox{cCsoft}{\color{cC}\ttfamily\fontsize{6.4}{7}\selectfont #1}}

% one stat in the title-slide strip
\newcommand{\statcell}[2]{%
  \begin{minipage}[t]{\dimexpr\linewidth}\raggedright
  {\dispfont\fontsize{14}{16}\selectfont\color{ink}#1}\par\vskip0.3mm
  {\sffamily\fontsize{5.8}{7}\selectfont\color{muted}\MakeUppercase{#2}}
  \end{minipage}}

\pgfplotsset{
  qrtf/.style={
    width=\plotwidth, height=\plotheight,
    axis line style={hair,line width=0.4pt},
    tick style={draw=none},
    grid=major, grid style={hairb,line width=0.4pt},
    every axis plot/.append style={line width=1.1pt},
    label style={font=\sffamily\fontsize{6.4}{8}\selectfont,color=muted},
    tick label style={font=\sffamily\fontsize{6.4}{8}\selectfont,color=muted},
    legend style={draw=none,fill=none,font=\sffamily\fontsize{6.4}{8}\selectfont,
                  text=ink2,cells={anchor=west}},
    axis x line*=bottom, axis y line*=left,
    clip=false,
  },
  nodenear/.style={
    nodes near coords, every node near coord/.append style={
      font=\ttfamily\fontsize{5.6}{7}\selectfont, color=ink2, anchor=south, yshift=0.2mm},
  },
}
\newlength{\plotwidth}\newlength{\plotheight}
\setlength{\plotwidth}{124mm}\setlength{\plotheight}{40mm}

\begin{document}
"""

# ── HTML → LaTeX ─────────────────────────────────────────────────────────────
# The prose lives once, in build_deck.SLIDES. This converts the small, known
# subset of markup that module emits; it is not a general HTML converter.

_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}"}


def tex_escape(t: str) -> str:
    t = _html.unescape(t)
    out = "".join(_ESC.get(ch, ch) for ch in t)
    return (out.replace("10⁻¹⁶", r"$10^{-16}$")
              .replace("√", r"$\sqrt{\ }$")
              .replace("₹", r"\rupee{}"))


def inline(t: str) -> str:
    """Inline markup, applied before escaping so the tags survive."""
    t = re.sub(r"<strong>(.*?)</strong>", lambda m: "\x01" + m.group(1) + "\x02", t, flags=re.S)
    t = re.sub(r"<em>(.*?)</em>", lambda m: "\x03" + m.group(1) + "\x04", t, flags=re.S)
    t = re.sub(r'<span class="mono">(.*?)</span>', lambda m: "\x05" + m.group(1) + "\x06", t, flags=re.S)
    t = re.sub(r"<span class='mono'>(.*?)</span>", lambda m: "\x05" + m.group(1) + "\x06", t, flags=re.S)
    t = re.sub(r'<span class="pill ok">(.*?)</span>', lambda m: "\x07" + m.group(1) + "\x08", t, flags=re.S)
    t = re.sub(r'<span class="pill no">(.*?)</span>', lambda m: "\x0b" + m.group(1) + "\x0c", t, flags=re.S)
    t = re.sub(r"<br\s*/?>", "\x0e", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = tex_escape(t)
    for a, b in (("\x01", r"\str{"), ("\x03", r"\emph{"), ("\x05", r"\mn{"),
                 ("\x07", r"\pillok{"), ("\x0b", r"\pillno{")):
        t = t.replace(a, b)
    for c in "\x02\x04\x06\x08\x0c":
        t = t.replace(c, "}")
    return re.sub(r"\s+", " ", t.replace("\x0e", r"\\ ")).strip()


def _blocks(frag: str) -> str:
    """Paragraph-level markup inside one column of text."""
    out = []
    for m in re.finditer(r'<(p|h4|ol|div)([^>]*)>(.*?)</\1>', frag, flags=re.S):
        tag, attrs, inner = m.group(1), m.group(2), m.group(3)
        cls = (re.search(r'class="([^"]*)"', attrs) or [None, ""])[1]
        if tag == "h4":
            out.append(r"\subhead{%s}" % inline(inner))
        elif tag == "p":
            if "lede" in cls:
                out.append(r"\lede{%s}" % inline(inner))
            elif "aside" in cls:
                out.append(r"\aside{%s}" % inline(inner))
            elif "caption" in cls:
                out.append(r"\figcaption{%s}" % inline(inner))
            else:
                out.append(r"\bodytext{%s}" % inline(inner))
            out.append(r"\vskip0.8mm")
        elif tag == "ol":
            items = re.findall(r"<li>(.*?)</li>", inner, flags=re.S)
            body = "".join(r"\item %s" % inline(i) for i in items)
            out.append(r"{\sffamily\fontsize{7.4}{10}\selectfont\color{ink2}"
                       r"\begin{enumerate}\setlength{\itemsep}{1.1mm}%s\end{enumerate}}" % body)
        elif tag == "div" and "tw" in cls:
            out.append(_table(inner))
    return out


def _table(frag: str) -> str:
    head = re.findall(r"<th>(.*?)</th>", frag, flags=re.S)
    rows = [re.findall(r"<td>(.*?)</td>", r, flags=re.S)
            for r in re.findall(r"<tr>(.*?)</tr>", frag, flags=re.S) if "<td>" in r]
    n = len(head)
    spec = "l" + "r" * (n - 1)
    hdr = " & ".join(r"\textsc{\fontsize{5.8}{7}\selectfont %s}" % inline(h) for h in head)
    body = r" \\ ".join(" & ".join(inline(c) for c in r) for r in rows)
    return (r"{\sffamily\fontsize{6.6}{8.8}\selectfont\color{ink2}"
            r"\begin{tabular}{%s}\toprule %s \\ \midrule %s \\ \bottomrule\end{tabular}}"
            % (spec, hdr, body))


def _stats(frag: str) -> str:
    cells = re.findall(r'<span class="sv">(.*?)</span>\s*<span class="sl">(.*?)</span>',
                       frag, flags=re.S)
    w = round(1.0 / len(cells) - 0.012, 3)
    cols = "".join(r"\begin{column}{%s\textwidth}\statcell{%s}{%s}\end{column}"
                   % (w, inline(v), inline(l)) for v, l in cells)
    return (r"\vskip2mm{\color{hair}\hrule height0.4pt}\vskip2.4mm"
            r"\begin{columns}[T]%s\end{columns}" % cols)


def _split_divs(frag: str) -> list[str]:
    """Top-level <div> children of a container, by brace matching on tags."""
    parts, depth, start = [], 0, None
    for m in re.finditer(r"<div\b[^>]*>|</div>", frag):
        if m.group(0).startswith("</"):
            depth -= 1
            if depth == 0:
                parts.append(frag[start:m.start()])
        else:
            if depth == 0:
                start = m.end()
            depth += 1
    return parts


def body_parts(body: str) -> list:
    body = re.sub(r"<svg.*?</svg>", "\x0f", body, flags=re.S)   # figures are re-authored
    out = []
    pos = 0
    for m in re.finditer(r'<div class="(two|findings|stats|tierflow|trackrow)">', body):
        out.extend(_blocks(body[pos:m.start()]))
        cls = m.group(1)
        depth, i = 1, m.end()
        while depth:
            nxt = re.search(r"<div\b[^>]*>|</div>", body[i:])
            depth += -1 if nxt.group(0).startswith("</") else 1
            i += nxt.end()
        inner = body[m.end():i - len("</div>")]
        if cls == "stats":
            out.append(_stats(inner))
        elif cls in ("two", "findings"):
            kids = _split_divs(inner)
            half = (len(kids) + 1) // 2
            left = "\n".join(b for k in kids[:half] for b in _blocks(k))
            right = "\n".join(b for k in kids[half:] for b in _blocks(k))
            out.append(r"\begin{columns}[T]\begin{column}{0.475\textwidth}%s\end{column}"
                       r"\begin{column}{0.475\textwidth}%s\end{column}\end{columns}" % (left, right))
        elif cls == "tierflow":
            out.append(TIERFLOW)
        elif cls == "trackrow":
            out.append(TRACKROW)
        pos = i
    out.extend(_blocks(body[pos:]))
    return [x for x in out if x.strip()]


def layout_parts(parts: list) -> str:
    """Two columns for plain prose; stacked when a part is itself a multi-column
    block or a table, which must never be nested inside another column."""
    if any(r"\begin{columns}" in b or r"\begin{tabular}" in b for b in parts):
        return "\n".join(parts)
    return two_col(parts)


def two_col(flat: list) -> str:
    half = (len(flat) + 1) // 2
    return (r"\begin{columns}[T]\begin{column}{0.475\textwidth}%s\end{column}"
            r"\begin{column}{0.475\textwidth}%s\end{column}\end{columns}"
            % ("\n".join(flat[:half]), "\n".join(flat[half:])))


def body_to_tex(body: str, cols: int = 1) -> str:
    parts = body_parts(body)
    return two_col(parts) if cols == 2 else "\n".join(parts)


def split_at(parts: list, budget: int):
    """Split a body into what fits beside a figure and what follows it."""
    head, n = [], 0
    for i, b in enumerate(parts):
        n += len(b)
        head.append(b)
        if n > budget:
            tail = [x for x in parts[i + 1:] if not x.startswith(r"\vskip")]
            return head, tail
    return head, []


# ── TikZ diagrams ────────────────────────────────────────────────────────────

def _cards(items, colors, width="24.5mm", step=27.4):
    out = [r"\vskip1mm\begin{center}\begin{tikzpicture}[node distance=0mm]"]
    x = 0.0
    for (kicker, name, lines), col in zip(items, colors):
        body = r"\\[0.4mm]".join(lines)
        out.append(
            r"\node[anchor=north west,inner sep=1.6mm,text width=%s,fill=sunken,"
            r"rounded corners=0.6pt] (n%d) at (%.1fmm,0) {"
            r"{\ttfamily\fontsize{5.6}{7}\selectfont\color{muted}%s}\\[0.4mm]"
            r"{\dispfont\fontsize{10}{11}\selectfont\color{ink}%s}\\[0.8mm]"
            r"{\sffamily\fontsize{6.6}{8.4}\selectfont\color{ink2}\begin{tabular}{@{}l@{}}%s\end{tabular}}};"
            % (width, len(out), x, kicker, name, body))
        out.append(r"\draw[%s,line width=1.1pt] (n%d.north west)--(n%d.north east);"
                   % (col, len(out) - 1, len(out) - 1))
        x += step
    out.append(r"\end{tikzpicture}\end{center}\vskip1mm")
    return "\n".join(out)


TIERFLOW = _cards([
    ("Tier 0", "Data", ["load prices,", "validate, forward-fill"]),
    ("Tier 1", "Trees", [r"17 features $\rightarrow$ three", "models $\\rightarrow$ one score"]),
    ("Tier 2", "Regime", ["calm vs.\\ stressed;", "de-risk gate"]),
    ("Tier 3", "Execution", ["weights, costs,", "realized return"]),
    ("Tier 4", "Gate", ["is the result", "credible?"]),
], ["cA", "cA", "cB", "cD", "cC"])

TRACKROW = _cards([
    ("Track A", "Measure the cost", ["half-spread estimated from", "daily prices; market impact",
                                     "charged per name"]),
    ("Track B", "Verify the shorts", ["point-in-time borrowable set", "from traded futures",
                                      "contracts"]),
    ("Track D", "Correct the statistics", ["honest trial count $N=30$;", "Lo correction for serial",
                                           "correlation"]),
], ["cA", "cB", "cD"], width="42mm", step=45.5)


def diagram_crosssection_tex():
    rng = np.random.default_rng(42)
    n = 500
    y = np.sort(np.linspace(1, -1, n) + rng.normal(0, 0.075, n))[::-1]
    seg = []
    for i in range(0, n, 2):
        col = "cB" if i < 50 else ("cC" if i >= n - 50 else "hair")
        seg.append(r"\draw[%s,line width=0.3pt](%.2f,0)--(%.2f,%.3f);" % (col, i * 0.155, i * 0.155, y[i] * 13))
    return (r"\begin{center}\begin{tikzpicture}[x=1mm,y=1mm]" + "".join(seg) +
            r"\draw[hair,line width=0.4pt](0,0)--(77.5,0);"
            r"\node[anchor=west,font=\sffamily\fontsize{6.6}{8}\selectfont,text=cB] at (0,16){top decile --- bought (50)};"
            r"\node[anchor=east,font=\sffamily\fontsize{6.6}{8}\selectfont,text=cC] at (77.5,-16){bottom decile --- sold short (50)};"
            r"\node[font=\sffamily\fontsize{6.6}{8}\selectfont,text=muted] at (39,3.4){the other 400 are not held};"
            r"\node[anchor=west,font=\sffamily\fontsize{6.2}{8}\selectfont,text=muted] at (0,-20){500 stocks, ranked by predicted return $\rightarrow$};"
            r"\end{tikzpicture}\end{center}")


def diagram_walkforward_tex():
    rows, out, yy = [0, 1, 2, 3, None, 37], [], 0.0
    K = 0.60   # x-scale: the split layout gives this diagram about 78mm
    for r in rows:
        if r is None:
            out.append(r"\node[font=\sffamily\fontsize{7}{8}\selectfont,text=muted] at (16,%.1f){$\vdots$};" % (yy - 2.5))
            yy -= 7
            continue
        s = min(r, 4) * 6.4 * K
        out.append(r"\fill[cA,rounded corners=0.5pt](%.1f,%.1f)rectangle(%.1f,%.1f);" % (s, yy, s + 51 * K, yy - 4.4))
        out.append(r"\fill[cB,rounded corners=0.5pt](%.1f,%.1f)rectangle(%.1f,%.1f);" % (s + 51.6 * K, yy, s + 58 * K, yy - 4.4))
        out.append(r"\node[anchor=east,font=\ttfamily\fontsize{6}{7}\selectfont,text=muted] at (-2,%.1f){%s};"
                   % (yy - 2.2, "fold %d" % (r + 1) if r < 4 else "fold 38"))
        yy -= 7
    return (r"\begin{center}\begin{tikzpicture}[x=1mm,y=1mm]" + "".join(out) +
            r"\node[anchor=west,font=\sffamily\fontsize{6.6}{8}\selectfont,text=cA] at (0,5.5){train --- 504 days};"
            r"\node[anchor=west,font=\sffamily\fontsize{6.6}{8}\selectfont,text=cB] at (36,5.5){predict --- 63 days, never seen};"
            r"\draw[hair,line width=0.4pt](0,%.1f)--(70,%.1f);" % (yy + 2, yy + 2) +
            r"\node[anchor=west,font=\sffamily\fontsize{6.2}{8}\selectfont,text=muted] at (0,%.1f){time $\rightarrow$};" % (yy - 1.5) +
            r"\node[anchor=east,font=\sffamily\fontsize{6.2}{8}\selectfont,text=muted] at (70,%.1f){2016--2025, 38 folds};" % (yy - 1.5) +
            r"\end{tikzpicture}\end{center}")


# ── pgfplots helpers ─────────────────────────────────────────────────────────

def C(xs, ys):
    return " ".join("(%.4f,%.5f)" % (x, y) for x, y in zip(xs, ys))


def yearx(idx):
    return [d.year + (d.dayofyear - 1) / 365.0 for d in idx]


def axis(opts, body, w=124, h=40):
    return (r"\begin{center}\setlength{\plotwidth}{%dmm}\setlength{\plotheight}{%dmm}"
            r"\begin{tikzpicture}\begin{axis}[qrtf,%s]%s\end{axis}\end{tikzpicture}\end{center}"
            % (w, h, opts, body))


def line_chart(series, *, opts="", w=124, h=40, endlab=True):
    body = []
    for s in series:
        st = "%s,line width=1.1pt" % s["c"]
        if s.get("dash"):
            st += ",dashed"
        body.append(r"\addplot[%s,mark=none] coordinates{%s};" % (st, C(s["x"], s["y"])))
        if endlab:
            body.append(r"\node[anchor=west,font=\sffamily\fontsize{6.4}{8}\selectfont,"
                        r"text=%s,xshift=0.8mm] at (axis cs:%.4f,%.5f){%s};"
                        % (s["c"], s["x"][-1], s["y"][-1], s["name"]))
    return axis(opts, "".join(body), w, h)


def bar_chart(groups, series, *, opts="", w=124, h=40, near=True, dp=2):
    body = []
    for s in series:
        n = (r"nodes near coords,every node near coord/.append style={"
             r"font=\ttfamily\fontsize{5.4}{6}\selectfont,color=ink2,"
             r"/pgf/number format/.cd,fixed,precision=%d}," % dp) if near else ""
        body.append(r"\addplot[ybar,fill=%s,draw=none,%s] coordinates{%s};"
                    % (s["c"], n, C(range(len(groups)), s["v"])))
        body.append(r"\addlegendentry{%s}" % s["name"])
    tk = ",".join(str(i) for i in range(len(groups)))
    tl = ",".join("{%s}" % g for g in groups)
    o = (r"ybar,bar width=%.1fmm,xtick={%s},xticklabels={%s},xtick style={draw=none},"
         r"enlarge x limits=%.2f,%s" % (max(2.0, 26.0 / max(len(groups) * len(series), 1)), tk, tl,
                                        0.9 / len(groups) if len(groups) > 2 else 0.4, opts))
    return axis(o, "".join(body), w, h)


def dumbbell(rows, *, opts="", w=124, h=40, groups=()):
    body = []
    for i, r in enumerate(rows):
        body.append(r"\draw[hair,line width=2.2pt,line cap=round](axis cs:%d,%.4f)--(axis cs:%d,%.4f);"
                    % (i, r["gross"], i, r["net"]))
    body.append(r"\addplot[only marks,mark=*,mark size=1.5pt,cA] coordinates{%s};"
                % C(range(len(rows)), [r["gross"] for r in rows]))
    pos = [(i, r["net"]) for i, r in enumerate(rows) if r["net"] >= 0]
    neg = [(i, r["net"]) for i, r in enumerate(rows) if r["net"] < 0]
    for pts, col in ((pos, "cB"), (neg, "cC")):
        if pts:
            body.append(r"\addplot[only marks,mark=*,mark size=1.5pt,%s] coordinates{%s};"
                        % (col, C([p[0] for p in pts], [p[1] for p in pts])))
    for i, r in enumerate(rows):
        gy, ny = r["gross"], r["net"]
        sep = abs(gy - ny) < (max(x["gross"] for x in rows) - min(x["net"] for x in rows)) * 0.09
        body.append(r"\node[anchor=west,font=\ttfamily\fontsize{5.4}{6}\selectfont,text=ink2,"
                    r"xshift=1mm,yshift=%.1fmm] at (axis cs:%d,%.4f){%s};"
                    % (1.1 if sep else 0, i, gy, fmt(gy)))
        body.append(r"\node[anchor=west,font=\ttfamily\fontsize{5.4}{6}\selectfont,text=%s,"
                    r"xshift=1mm,yshift=%.1fmm] at (axis cs:%d,%.4f){%s};"
                    % ("cC" if ny < 0 else "cB", -1.4 if sep else 0, i, ny, fmt(ny)))
    tk = ",".join(str(i) for i in range(len(rows)))
    tl = ",".join("{%s}" % r["label"] for r in rows)
    sub = "".join(r"\node[font=\ttfamily\fontsize{5.2}{6}\selectfont,text=muted,anchor=north,"
                  r"yshift=-3.4mm] at (axis cs:%d,\pgfkeysvalueof{/pgfplots/ymin}){%s};"
                  % (i, r["sub"]) for i, r in enumerate(rows) if r.get("sub"))
    grp = "".join(r"\node[font=\sffamily\fontsize{6.2}{8}\selectfont,text=muted,anchor=north,"
                  r"yshift=-6.6mm] at (axis cs:%.1f,\pgfkeysvalueof{/pgfplots/ymin}){%s};"
                  % ((a + b) / 2, lab) for a, b, lab in groups)
    o = (r"xtick={%s},xticklabels={%s},xtick style={draw=none},x tick label style={"
         r"font=\sffamily\fontsize{6.2}{8}\selectfont,color=muted},enlarge x limits=0.06,%s"
         % (tk, tl, opts))
    return axis(o, "".join(body) + sub + grp, w, h)


def waterfall(steps, *, w=124, h=40):
    body = []
    for grp, col in ((["start", "pass"], "cA"), (["fail"], "cC")):
        pts = [(i, s["value"]) for i, s in enumerate(steps) if s["state"] in grp]
        body.append(r"\addplot[ybar,bar width=13mm,fill=%s,draw=none] coordinates{%s};"
                    % (col, C([p[0] for p in pts], [p[1] for p in pts])))
    for i, s in enumerate(steps):
        v = s["value"]
        body.append(r"\node[font=\ttfamily\fontsize{6.4}{8}\selectfont,text=ink2,anchor=%s,"
                    r"yshift=%.1fmm] at (axis cs:%d,%.4f){%s};"
                    % ("south" if v >= 0 else "north", 0.8 if v >= 0 else -0.8, i, v, fmt(v)))
        if i:
            p = steps[i - 1]["value"]
            body.append(r"\draw[muted,dashed,line width=0.4pt](axis cs:%.2f,%.4f)--(axis cs:%.2f,%.4f);"
                        % (i - 1 + 0.24, p, i - 0.24, p))
            body.append(r"\node[font=\ttfamily\fontsize{5.6}{7}\selectfont,text=muted,anchor=south,"
                        r"yshift=0.4mm] at (axis cs:%.2f,%.4f){%s};" % (i - 0.5, p, fmt(v - p, 2, True)))
        col = "cB" if s["state"] in ("start", "pass") else "cC"
        txt = "clears the gate" if s["state"] in ("start", "pass") else "fails"
        body.append(r"\node[font=\sffamily\fontsize{5.8}{7}\selectfont,text=%s,anchor=north,"
                    r"yshift=-7.6mm] at (axis cs:%d,\pgfkeysvalueof{/pgfplots/ymin}){\MakeUppercase{%s}};"
                    % (col, i, txt))
    tk = ",".join(str(i) for i in range(len(steps)))
    tl = ",".join("{\\begin{tabular}{@{}c@{}}%s\\end{tabular}}"
                  % r"\\".join(s["label"].split("|")) for s in steps)
    o = (r"ybar,xtick={%s},xticklabels={%s},xtick style={draw=none},enlarge x limits=0.14,"
         r"x tick label style={font=\sffamily\fontsize{6.4}{8}\selectfont,color=ink2},"
         r"ylabel={annualized Sharpe ratio},extra y ticks={0},extra y tick style={grid=major,"
         r"grid style={hair,line width=0.5pt}}" % (tk, tl))
    return axis(o, "".join(body), w, h)


def stack_chart(groups, series, labels, *, opts="", w=124, h=40):
    body = []
    for s in series:
        body.append(r"\addplot[ybar,fill=%s,draw=none] coordinates{%s};"
                    % (s["c"], C(range(len(groups)), s["v"])))
        body.append(r"\addlegendentry{%s}" % s["name"])
    tot = [sum(s["v"][i] for s in series) for i in range(len(groups))]
    for i, t in enumerate(tot):
        body.append(r"\node[font=\ttfamily\fontsize{5.4}{6}\selectfont,text=muted,anchor=south,"
                    r"yshift=0.6mm] at (axis cs:%d,%.2f){%s};" % (i, t, labels[i]))
    tk = ",".join(str(i) for i in range(len(groups)))
    tl = ",".join("{%s}" % g for g in groups)
    o = (r"ybar stacked,bar width=6.4mm,xtick={%s},xticklabels={%s},xtick style={draw=none},"
         r"enlarge x limits=0.06,ymin=0,%s" % (tk, tl, opts))
    return axis(o, "".join(body), w, h)


# ── the sixteen figures ──────────────────────────────────────────────────────

LEGEND_BOTTOM = (r"legend style={at={(0.5,-0.16)},anchor=north,legend columns=-1,"
                 r"/tikz/every even column/.append style={column sep=3mm}},")
YEARTICKS = "xtick={2016,2018,2020,2022,2024},xticklabel style={/pgf/number format/1000 sep=},"


def fig_equity(cols, names, colors, dashes, *, w=124, h=40, ticks=(1, 2, 5, 10, 20)):
    ser = []
    for col, nm, c, d in zip(cols, names, colors, dashes):
        s = weekly(col)
        ser.append({"x": yearx(s.index), "y": list(s.values), "name": nm, "c": c, "dash": d})
    hi = max(max(s["y"]) for s in ser)
    tk = [t for t in ticks if t <= hi * 1.35]
    return line_chart(ser, w=w, h=h, opts=(
        r"ymode=log,log basis y=10,ytick={%s},yticklabels={%s},%s"
        r"ylabel={growth of Rs 1},enlarge x limits=0.02,"
        % (",".join(str(t) for t in tk), ",".join(r"%d$\times$" % t for t in tk), YEARTICKS)))


lo21 = next(c for c in CELLS if c["style"] == "long_only" and c["target"] == "21-day")

FIGURES = {}
FIGURES[1] = fig_equity(
    [cum(p4c_gross[lo21["sid"] + "__gross"]), cum(p4c_net[lo21["sid"]])],
    ["before costs", "after costs"], ["cA", "cB"], [False, False], h=34)

FIGURES[2] = diagram_crosssection_tex()
FIGURES[4] = diagram_walkforward_tex()

_icr = ic_500.set_index("model")
_ic68 = float(ic_68.set_index("model").loc["ensemble", "mean_IC"])
FIGURES[5] = bar_chart(
    ["LightGBM", "XGBoost", "forest", r"\textbf{averaged}", "68-stock"],
    [{"name": "mean information coefficient",
      "v": [float(_icr.loc[m, "mean_IC"]) for m in ("lgbm", "xgb", "rf", "ensemble")] + [_ic68],
      "c": "cA"}],
    w=74, h=42, dp=4,
    opts=r"ylabel={mean information coefficient},ymin=0,legend style={draw=none,fill=none},"
         r"legend to name=none,")

FIGURES[6] = bar_chart(
    [r"\begin{tabular}{@{}c@{}}stable\\uptrend\end{tabular}",
     r"\begin{tabular}{@{}c@{}}stable\\downtrend\end{tabular}",
     r"\begin{tabular}{@{}c@{}}stable\\volatile\end{tabular}",
     r"\begin{tabular}{@{}c@{}}volatile\\downtrend\end{tabular}"],
    [{"name": "long only", "v": P1["long only"], "c": "cA"},
     {"name": "long / short", "v": P1["long / short"], "c": "cB"},
     {"name": "dynamic tilt", "v": P1["dynamic tilt"], "c": "cD"}],
    w=74, h=42, dp=0, opts=r"ylabel={cumulative return, \%%},%s" % LEGEND_BOTTOM)

_p3rows = []
for _f in P3_FREQ:
    for _j, _st in enumerate(P3_STYLE):
        _p3rows.append({"label": _st.replace("long / short", "long/short").replace("dynamic tilt", "dyn.\\ tilt"),
                        "gross": P3_GROSS[_f][_j], "net": p3_net_sr[(_f, _st)]})
FIGURES[7] = dumbbell(_p3rows, w=124, h=31,
                      groups=[(0, 2, "daily"), (3, 5, "hourly"), (6, 8, "every 30 min"), (9, 11, "every 15 min")],
                      opts=r"ylabel={annualized Sharpe ratio},extra y ticks={0},"
                           r"extra y tick style={grid=major,grid style={hair,line width=0.5pt}},"
                           r"x tick label style={rotate=32,anchor=east,font=\sffamily\fontsize{5.8}{7}\selectfont,color=muted},")

FIGURES[8] = bar_chart(
    [c["year"] for c in CHURN],
    [{"name": "entered the index", "v": [c["in"] for c in CHURN], "c": "cB"},
     {"name": "left the index", "v": [-c["out"] for c in CHURN], "c": "cC"}],
    w=74, h=42, dp=0,
    opts=r"ylabel={names},extra y ticks={0},extra y tick style={grid=major,"
         r"grid style={hair,line width=0.5pt}},%s" % LEGEND_BOTTOM)

FIGURES[9] = fig_equity(
    [cum(p4b[P4B[2]["sid"]]), cum(p4b[P4B[1]["sid"]]), cum(p4b[P4B[0]["sid"]]), cum(nifty_ret)],
    ["dynamic tilt", "long / short", "long only", "NIFTY-50"],
    ["cD", "cA", "cB", "cN"], [False, False, False, True], h=37, ticks=(1, 2, 4, 8, 16))

FIGURES[10] = line_chart(
    [{"x": SLIP_BPS, "y": SLIP["long / short"], "name": "long / short", "c": "cA"},
     {"x": SLIP_BPS, "y": SLIP["dynamic tilt"], "name": "dynamic tilt", "c": "cD"},
     {"x": SLIP_BPS, "y": SLIP["long only"], "name": "long only", "c": "cB"}],
    w=74, h=42,
    opts=r"xlabel={additional cost, basis points per trade},ylabel={net Sharpe},"
         r"xtick={0,5,10,15,20},extra y ticks={0},extra y tick style={grid=major,"
         r"grid style={hair,line width=0.5pt}},enlarge x limits=0.1,")

FIGURES[12] = bar_chart(
    ["most liquid", "2nd", "3rd", "4th", "least liquid"],
    [{"name": "Corwin--Schultz (used for the headline)", "v": SPREAD_Q["cs"], "c": "cB"},
     {"name": "Abdi--Ranaldo", "v": SPREAD_Q["ar"], "c": "cA"}],
    w=124, h=36, dp=2,
    opts=r"ylabel={half-spread, basis points},ymin=0,%s" % LEGEND_BOTTOM)
FIGURES[12] = FIGURES[12].replace(
    r"\end{axis}",
    r"\draw[muted,dashed,line width=0.5pt](axis cs:-0.6,%.4f)--(axis cs:4.6,%.4f);"
    r"\node[anchor=south east,font=\sffamily\fontsize{5.8}{7}\selectfont,text=muted]"
    r" at (axis cs:4.6,%.4f){statutory cost alone --- %.1f bps};\end{axis}"
    % (STATUTORY_BPS, STATUTORY_BPS, STATUTORY_BPS, STATUTORY_BPS))

FIGURES[13] = stack_chart(
    SHORT_YEARS,
    [{"name": "could be borrowed", "v": SHORT_BORROW, "c": "cB"},
     {"name": "could not be borrowed", "v": [i - b for i, b in zip(SHORT_INTEND, SHORT_BORROW)], "c": "cC"}],
    ["%.0f\\%%" % c for c in SHORT_COVER],
    w=124, h=40,
    opts=r"ylabel={short positions wanted per day},%s" % LEGEND_BOTTOM)

FIGURES[14] = line_chart(
    [{"x": N_GRID, "y": SR_STAR_CURVE, "name": "required Sharpe", "c": "cA"}],
    w=74, h=42, endlab=False,
    opts=r"xlabel={configurations tried},ylabel={Sharpe the strategy must beat},"
         r"xtick={2,10,20,30,40,50,60},enlarge x limits=0.04,")
FIGURES[14] = FIGURES[14].replace(
    r"\end{axis}",
    r"\draw[muted,dashed,line width=0.5pt](axis cs:2,1.35)--(axis cs:60,1.35);"
    r"\node[anchor=north east,font=\sffamily\fontsize{5.8}{7}\selectfont,text=muted]"
    r" at (axis cs:60,1.33){long/short, corrected --- 1.35};"
    r"\addplot[only marks,mark=*,mark size=1.4pt,cA] coordinates{(3,%.4f) (30,%.4f)};"
    r"\node[anchor=west,font=\ttfamily\fontsize{5.6}{7}\selectfont,text=ink2]"
    r" at (axis cs:3.6,%.4f){$N=3$: 0.39};"
    r"\node[anchor=east,font=\ttfamily\fontsize{5.6}{7}\selectfont,text=ink2]"
    r" at (axis cs:29,%.4f){$N=30$: 0.96};\end{axis}"
    % (SR_STAR_CURVE[1], SR_STAR_CURVE[28], SR_STAR_CURVE[1], SR_STAR_CURVE[28]))

_grid = []
for _t in ("5-day", "21-day"):
    for _s in ("long_only", "long_short_slb", "dynamic_tilt_slb"):
        _c = next(x for x in CELLS if x["style"] == _s and x["target"] == _t)
        _grid.append({"label": _s.replace("_slb", "").replace("_", " ").replace("long short", "long/short"),
                      "sub": "DSR %s" % fmt(_c["dsr"], 3), "gross": _c["gross_sr"], "net": _c["net_sr"]})
FIGURES[15] = dumbbell(_grid, w=124, h=35,
                       groups=[(0, 2, "5-day forecast"), (3, 5, "21-day forecast")],
                       opts=r"ylabel={annualized Sharpe ratio},extra y ticks={0},"
                            r"extra y tick style={grid=major,grid style={hair,line width=0.5pt}},")

FIGURES[16] = waterfall([
    {"label": "as reported|in Phase 4b", "value": 1.78, "state": "start"},
    {"label": "honest trial count|and serial correlation", "value": 1.35, "state": "pass"},
    {"label": "measured spread|and market impact", "value": -0.23, "state": "fail"},
    {"label": "short book restricted|to what could be borrowed", "value": -0.99, "state": "fail"},
], w=124, h=40)

FIGURES[17] = (bar_chart(
    IC_YEARS, [{"name": "significance of the ranking", "v": IC_T, "c": "cA"}],
    w=124, h=23, dp=1,
    opts=r"ylabel={},ymin=0,ytick={0,4,8,12},"
         r"legend style={draw=none,fill=none},legend to name=none,"))


# ── frames ───────────────────────────────────────────────────────────────────

def _frame(section: str, title: str, inner: str, note: str = "") -> str:
    return (r"\eyebrow{%s}" % inline(section) + "\n" +
            r"\begin{frame}[t]{%s}" % title + "\n" + inner + "\n" + note +
            r"\end{frame}" + "\n")


CONT = r" \textnormal{\itshape --- continued}"

# How much prose fits on one printed frame beside each kind of figure. A
# 16:9 frame gives about 70mm of text; the budgets are in characters of the
# generated LaTeX and were set by compiling and reading the overfull log.
BUDGET = {"title": 10_000, "split": 560, "figure": 300, "wide": 380}


def frames_for(i: int, s: dict) -> str:
    """One slide of the HTML deck becomes one or two printed frames."""
    title = inline(s["title"].replace("<br/>", " "))
    fig = FIGURES.get(i, "")
    cap = r"\figcaption{%s}" % inline(s["caption"]) if s.get("caption") else ""
    note = r"\note{%s}" % inline(s["notes"]) if s.get("notes") else ""
    lay = s["layout"]
    parts = body_parts(s["body"])
    head, tail = split_at(parts, s.get("budget", BUDGET[lay]))

    if lay == "title":
        inner = "\n".join(head) + r"\vskip2.6mm" + fig + cap
    elif lay == "split":
        inner = (r"\begin{columns}[T]\begin{column}{0.40\textwidth}%s\end{column}"
                 r"\begin{column}{0.575\textwidth}%s\vskip1mm%s\end{column}\end{columns}"
                 % ("\n".join(head), fig, cap))
    elif lay == "figure":
        inner = fig + cap + (r"\vskip2.2mm" + layout_parts(head) if head else "")
    else:
        inner = "\n".join(head)

    out = _frame(s["section"], title, inner, note)
    if tail:
        out += _frame(s["section"], title + CONT, layout_parts(tail))
    return out


def findings_frames(sl: dict) -> str:
    """The four findings need a frame each pair; the waterfall gets its own."""
    kids = _split_divs(re.search(r'<div class="findings">(.*)</div>\s*$',
                                 sl["body"], flags=re.S).group(1))
    out = _frame(sl["section"], inline(sl["title"]),
                 FIGURES[16] + r"\figcaption{%s}" % inline(sl["caption"]),
                 r"\note{%s}" % inline(sl["notes"]))
    pairs = [(kids[:2], FIGURES[17] + r"\figcaption{%s}" % inline(
                  "Significance of the daily ranking by year, from the Phase 4c "
                  "diagnostics. A value near 2 is the conventional threshold.")),
             (kids[2:], "")]
    for ks, extra in pairs:
        cols = r"\begin{columns}[T]%s\end{columns}" % "".join(
            r"\begin{column}{0.475\textwidth}%s\end{column}"
            % "\n".join(_blocks(k)) for k in ks)
        out += _frame(sl["section"], "Findings", (extra + r"\vskip2mm" if extra else "") + cols)
    return out


def build():
    frames = []
    for i, sl in enumerate(H.SLIDES):
        n = i + 1
        # the Phase 3 chart carries rotated labels and two label rows below it
        sl = dict(sl, budget=200) if n == 7 else sl
        frames.append(findings_frames(sl) if n == 16 else frames_for(n, sl))
    body = "".join(frames)
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text(PREAMBLE + "\n" + body + "\n" + r"\end{document}" + "\n", encoding="utf-8")
    print(f"  wrote {TEX.relative_to(ROOT)}  ({len(body)/1024:.0f} KB of frames)")


if __name__ == "__main__":
    build()
    if "--no-pdf" not in sys.argv:
        if not shutil.which("tectonic"):
            sys.exit("tectonic not found; install it or run with --no-pdf")
        print("  compiling with tectonic (first run downloads packages)")
        r = subprocess.run(["tectonic", "-X", "compile", "--keep-logs", str(TEX)],
                           capture_output=True, text=True)
        log = (r.stdout + r.stderr)
        if r.returncode:
            print("\n".join(log.strip().split("\n")[-40:]))
            sys.exit("compile failed")
        miss = sorted(set(re.findall(r"Missing character: There is no (.) ", log)))
        if miss:
            print(f"  WARNING missing glyphs: {' '.join(miss)}")
        texlog = TEX.with_suffix(".log")
        over = []
        if texlog.exists():
            over = [float(a) for a in re.findall(
                r"Overfull \\vbox \(([\d.]+)pt too high\)", texlog.read_text(errors="ignore"))]
        import pypdfium2
        n = len(pypdfium2.PdfDocument(str(PDF)))
        print(f"  wrote {PDF.relative_to(ROOT)}  ({PDF.stat().st_size/1024:.0f} KB, {n} frames)")
        bad = [o for o in over if o > 4]
        print(f"  frames overflowing their page: {len(bad)}"
              + (f"  (worst {max(bad):.0f}pt)" if bad else "  - none"))
