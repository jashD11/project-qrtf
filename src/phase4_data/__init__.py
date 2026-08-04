"""
Phase 4 — NSE bhavcopy acquisition and panel construction.

Self-contained package that rebuilds the production database from NSE's public
archive at ~500 names and 2013→2025 depth, replacing the 68-name Google-Drive
panel that Phase 2/3 ran on. Kept isolated from ``src/production_ml`` in the same
spirit as the existing SANDBOX / PRODUCTION_ML split: these modules produce data,
they do not import the tier stack.

Pipeline order:
    bhavcopy_download.py   raw archive files  → data/bhavcopy/{eq,udiff,index}/
    bhavcopy_panel.py      raw files          → adjusted long-format panel
    universe.py            panel              → point-in-time top-N mask
    bhavcopy_validate.py   panel + mask       → acceptance gates (must pass)

See docs/phase4_plan.md for the design and the acceptance criteria.
"""
