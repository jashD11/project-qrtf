---
name: explain
description: Produce a concrete, worked walkthrough of a QRTF pipeline tier or component (e.g. "explain tier 2", "walk me through the DSR gate", "explain feature normalization with the same concreteness"). Use when the user wants to UNDERSTAND how a piece of the pipeline actually works — mechanics traced through real numbers — not when they want code changes. Invoke for any request phrased as "walk me through / explain X with concreteness / how does X actually work".
---

# Concrete tier walkthrough

Produce an explanation in the exact style the user has approved: a mechanic
traced through **worked numbers**, anchored to the **real source code**, with the
**design rationale and gotchas** surfaced. This is a teaching artifact, not a
code summary. Match the voice and structure below precisely — it is a calibrated
quality bar, not a loose template.

## Step 0 — Always read the source first (non-negotiable)

Before writing a single number, Read the actual file(s) for the component. Every
formula, constant, threshold, and `file:line` reference in the output MUST come
from the current code, not memory. If the walkthrough spans tiers, read each
tier's module. Accuracy of the numbers and line references is what makes this
land — invented numbers destroy it. Relevant modules:

- Tier 0 data → `src/production_ml/data_scraping.py`
- Tier 1 features → `src/production_ml/feature_creator.py`
- Tier 1 trees → `src/production_ml/tier1_trees.py`
- Tier 2 regime/panic gate → `src/production_ml/tier2_regime.py`
- Tier 3 execution → `src/production_ml/tier3_execution.py`
- Tier 4 DSR gate → `src/production_ml/tier4_dsr_gate.py`
- Config / all constants → `config.py`
- Orchestrator → `run_pipeline_ml.py`

(For SANDBOX-mode requests, the analogous files live in `src/sandbox_run/`.)

## The structure (follow in order)

1. **Open with the ONE question this component answers**, and contrast its *axis*
   against the neighbouring tier. This is the spine. Examples that worked:
   - Tier 1 = *cross-sectional, per-stock, per-day*: "which stocks look good today."
   - Tier 2 = *time-series, one-market, per-day*: "is today calm or panicked."
   - Tier 3 = the *multiplication*: "what to hold × whether to hold it."
   - Tier 4 = the *judge*: "which of these returns do we believe."
   State the contrast explicitly ("Tier 1 asked X; Tier 2 asks Y").

2. **Set up a concrete scenario with real-ish specifics.** Pick a named date
   (e.g. `2021-04-12`), the real universe size (68 NSE tickers), the real
   constants from config (`decile_pct=0.10`, `panic_threshold=0.85`,
   `TRAIN_WINDOW=504`, etc.). Derive downstream numbers honestly (`k = floor(68 ×
   0.10) = 6`). Use a genuinely instructive contrast — calm day vs COVID day,
   strategy A (passes) vs strategy B (fails).

3. **Trace the numbers step by step through the actual functions**, citing
   `file:line` for each step. Show the arithmetic inline (`z = 0.1134 × 41.6 /
   1.003 = 4.70`). Round sensibly. When a formula appears, write it once in code
   fence form, then plug the numbers.

4. **Use comparison tables** for scenario contrasts (calm vs panic across the 3
   styles; strategy A vs B down the stat columns). Tables are the workhorse —
   they make the mechanic visible at a glance.

5. **Surface the WHY and the gotchas**, not just the what. Every tier has
   deliberate design decisions that look arbitrary until explained. Always hunt
   for and explain these, e.g.:
   - Why a percentile gate, not an HMM-posterior threshold (posteriors saturate).
   - Why the overnight gap is KEPT in execution but NULLED in the training label.
   - Why N in DSR is the honest search count, not the column count.
   - Why cross-sectional ranking removes time-varying scale.
   Frame each as "here's the part that surprised you / is worth pinning down."

6. **Close with a compact end-to-end recap** — an ASCII flow of how this tier
   hands off, and (if mid-series) offer the next tier as a one-line question.
   Keep the recap skimmable:
   ```
   Tier 1 → long/short decile masks   (which stocks)
   Tier 2 → panic gate                (calm or panic)
   Tier 3 → weights × forward returns (the P&L)
   Tier 4 → deflated verdict          (believe it?)
   ```

## Voice rules

- Concrete over abstract. A number beats an adjective every time.
- Bold the pivot phrases ("**This is the entire point.**", "**the overnight gap
  is KEPT**"). Sparingly — one or two per section.
- Reference code as `path:line` so it's clickable. Cite the function by name.
- Explain jargon the first time via what it *measures*, not its definition
  ("herding — are all 68 stocks moving together").
- Prefer second person for the reader's intuition ("the part that surprised you").
- Never dump code blocks of the source. Quote a formula or a single key line at
  most; the value is the *trace*, not the listing.
- End by offering the natural next step (next tier, or the run it feeds).

## Canonical quality bar

The four tier walkthroughs already delivered in this project's conversation are
the reference standard. If unsure whether the output is good enough, ask: does it
trace a specific number through a specific line of real code, and does it explain
one non-obvious design choice? If not, it's a summary, not a walkthrough — redo it.
