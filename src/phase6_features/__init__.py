"""
Phase 6 Part B — the new feature panels (docs/phase6_plan.md §B.2).

Self-contained package in the spirit of ``src/phase4_data``: these modules produce data
and do not import the tier stack. Each builder writes raw, un-normalized wide
``(date × ticker)`` Parquets under ``data/phase6/``; rank-normalization and the merge into
the feature matrix happen in ``feature_creator`` (Stage 5).

Every value on date t uses only data at or before the close of t, and no builder reads a
session after ``config.PHASE6_SEALED_AFTER`` unless explicitly allowed (the holdout step).

F1 (price and index) build order:
    index_series.py     index archive        -> data/phase6/index_daily.parquet
    price_features.py   panel + index        -> data/phase6/f1/{beta,ivol,ret_*,trade_size,spread}
    sector.py           panel + index        -> data/phase6/f1/sector_*.parquet
    market.py           index                -> data/phase6/f1/market.parquet

    python src/phase6_features/build_f1.py --build      # all of the above, in order
"""
