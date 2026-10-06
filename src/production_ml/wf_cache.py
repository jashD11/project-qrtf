"""
Phase 4c E1 — disk cache for the Tier 1 walk-forward fit.

``TreeAlphaEngine.run_walk_forward`` returns its ``WalkForwardResult`` purely in
memory, so every variant that only changes something *downstream* of Tier 1 — an
execution style, a cost model, a short-eligibility filter — still paid the full
~80-minute tree fit. Phase 4b lost one such fit outright to a Tier 3 failure.

The fit is a **pure function** of the feature panel and four engine knobs, so it is
safely cacheable. The key is the tuple that fully determines the output:

    (frequency, target_col, decile_pct, train_window, predict_window, panel_digest)

``panel_digest`` is a content hash of the exact feature frame handed to the engine —
its columns, its ('date','ticker') index, and every float value. A rebuilt panel, a
changed universe mask, a new feature, or one different price therefore all miss the
cache. Nothing about *when* the panel was built enters the key, so an unchanged
rebuild still hits.

This is a cache, not a store of record: deleting ``data/trial_database/wf_cache/``
only costs time. ``QRTF_WF_CACHE=0`` in the environment disables it entirely, which
is the escape hatch if a result ever needs to be reproduced from cold.

Usage
    from src.production_ml.wf_cache import cached_walk_forward
    wf = cached_walk_forward(engine, features, frequency)
"""

import hashlib
import os
import shutil
import sys
from typing import Final

import numpy as np
import pandas as pd

_ROOT: str = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.production_ml.feature_creator import (
    DATE_LEVEL,
    TICKER_LEVEL,
    present_feature_columns,
)
from src.production_ml.tier1_trees import TreeAlphaEngine, WalkForwardResult

CACHE_DIR: Final[str] = "data/trial_database/wf_cache"

# The three wide frames plus the read-only IC table, one Parquet each.
_FRAMES: Final[tuple[str, ...]] = ("alpha_scores", "long_mask", "short_mask")
_IC_FILE: Final[str] = "ic_diagnostics.parquet"
_KEY_FILE: Final[str] = "key.txt"


def _enabled() -> bool:
    """False when QRTF_WF_CACHE is set to a falsy value — the cold-run escape hatch."""
    return os.environ.get("QRTF_WF_CACHE", "1").strip().lower() not in ("0", "false", "no")


# --------------------------------------------------------------------------- #
# Keying
# --------------------------------------------------------------------------- #
def panel_digest(features: pd.DataFrame, columns: list[str]) -> str:
    """
    Content hash of a feature panel — its index, and the values of ``columns``.

    Only the columns the fit actually **reads** are hashed: the resolved feature
    schema plus the one target. That is not an optimization, it is what makes the
    key correct. Hashing the whole frame would make an unrelated column invalidate
    everything — adding the 21-day target (C1) alongside the existing 1b/5b labels
    would have thrown away every cached fit even though not one of them consumes it.

    Hashed **column by column** rather than via one ``to_numpy()`` of the whole
    frame: this runs immediately before the memory-critical tree loop, and a
    materialized (rows x cols) float copy of a 1.4M-row panel is ~200 MB of
    transient that the per-column form reduces to ~11 MB.
    """
    h = hashlib.md5()
    h.update(repr(list(columns)).encode())
    h.update(repr((features.shape[0], len(columns))).encode())
    idx = features.index
    h.update(
        pd.util.hash_pandas_object(
            pd.MultiIndex.from_arrays(
                [idx.get_level_values(DATE_LEVEL), idx.get_level_values(TICKER_LEVEL)]
            ),
            index=False,
        ).to_numpy().tobytes()
    )
    for c in columns:
        col = np.ascontiguousarray(features[c].to_numpy(dtype="float64"))
        h.update(col.tobytes())
    return h.hexdigest()


def cache_key(engine: TreeAlphaEngine, features: pd.DataFrame, frequency: str) -> str:
    """The full cache key: engine knobs that change the fit, plus the panel digest."""
    read_cols = present_feature_columns(features) + [engine.target_col]
    parts = (
        frequency,
        engine.target_col,
        f"{engine.decile_pct:g}",
        str(engine.train_window),
        str(engine.predict_window),
        engine.model_choice,
        panel_digest(features, read_cols),
    )
    return "__".join(parts)


def _slot(key: str) -> str:
    """Directory for one cache entry — a short digest of the (long) key."""
    return os.path.join(CACHE_DIR, hashlib.md5(key.encode()).hexdigest()[:16])


# --------------------------------------------------------------------------- #
# Read / write
# --------------------------------------------------------------------------- #
def _read(slot: str) -> WalkForwardResult | None:
    """Load a cache entry, or None if it is absent or incomplete (a killed write)."""
    paths = {n: os.path.join(slot, f"{n}.parquet") for n in _FRAMES}
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    frames = {n: pd.read_parquet(p) for n, p in paths.items()}
    ic_path = os.path.join(slot, _IC_FILE)
    ic = pd.read_parquet(ic_path) if os.path.exists(ic_path) else None
    return WalkForwardResult(
        alpha_scores=frames["alpha_scores"],
        long_mask=frames["long_mask"],
        short_mask=frames["short_mask"],
        ic_diagnostics=ic,
    )


def _write(slot: str, key: str, wf: WalkForwardResult) -> None:
    """
    Persist a fit. Written to a temp dir and renamed, so an interrupted write never
    leaves a half-populated slot that a later run would read as a hit.
    """
    tmp = f"{slot}.part"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    for name in _FRAMES:
        getattr(wf, name).to_parquet(os.path.join(tmp, f"{name}.parquet"))
    if wf.ic_diagnostics is not None:
        wf.ic_diagnostics.to_parquet(os.path.join(tmp, _IC_FILE))
    with open(os.path.join(tmp, _KEY_FILE), "w") as fh:
        fh.write(key + "\n")
    shutil.rmtree(slot, ignore_errors=True)
    os.replace(tmp, slot)


def cached_walk_forward(
    engine: TreeAlphaEngine, features: pd.DataFrame, frequency: str
) -> WalkForwardResult:
    """
    ``engine.run_walk_forward(features)``, served from disk when the exact same fit
    has been computed before.

    A miss fits and writes; a hit reads three Parquets in seconds. Any failure to
    read or write is reported and falls through to a live fit — the cache can cost
    time, never correctness.
    """
    if not _enabled():
        print("[wf_cache] disabled (QRTF_WF_CACHE=0) — fitting from scratch")
        return engine.run_walk_forward(features)

    key = cache_key(engine, features, frequency)
    slot = _slot(key)

    try:
        hit = _read(slot)
    except Exception as exc:                       # unreadable entry -> refit
        print(f"[wf_cache] WARN: unreadable entry {slot} ({exc}) — refitting")
        hit = None
    if hit is not None:
        print(
            f"[wf_cache] HIT {os.path.basename(slot)} | freq={frequency} "
            f"target={engine.target_col} | OOS days={hit.alpha_scores.shape[0]} "
            f"universe={hit.alpha_scores.shape[1]} (skipped the walk-forward fit)"
        )
        return hit

    print(
        f"[wf_cache] MISS {os.path.basename(slot)} | freq={frequency} "
        f"target={engine.target_col} decile_pct={engine.decile_pct} — fitting"
    )
    wf = engine.run_walk_forward(features)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        _write(slot, key, wf)
        print(f"[wf_cache] wrote {slot}")
    except Exception as exc:                       # disk full / permissions
        print(f"[wf_cache] WARN: could not persist ({exc}) — continuing uncached")
    return wf


# --------------------------------------------------------------------------- #
# Dry run — the cache must be transparent: hit == miss, and keys must discriminate
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile

    from src.production_ml.tier1_trees import _make_dummy_features

    print("=" * 70)
    print("wf_cache dry-run — round-trip fidelity + key sensitivity")
    print("=" * 70)

    dummy = _make_dummy_features(n_days=700, n_tickers=8)
    with tempfile.TemporaryDirectory() as tmpdir:
        CACHE_DIR = os.path.join(tmpdir, "wf_cache")  # noqa: F811 (module-level rebind)
        globals()["CACHE_DIR"] = CACHE_DIR

        eng = TreeAlphaEngine(model_choice="lgbm")   # one learner = fast
        cold = cached_walk_forward(eng, dummy, "daily")
        warm = cached_walk_forward(eng, dummy, "daily")

        checks: dict[str, bool] = {
            "alpha_scores round-trip exact": bool(
                np.array_equal(
                    cold.alpha_scores.to_numpy(), warm.alpha_scores.to_numpy(),
                    equal_nan=True,
                )
            ),
            "long_mask round-trip exact": bool(
                (cold.long_mask.to_numpy() == warm.long_mask.to_numpy()).all()
            ),
            "short_mask round-trip exact": bool(
                (cold.short_mask.to_numpy() == warm.short_mask.to_numpy()).all()
            ),
            "index/columns preserved": bool(
                cold.alpha_scores.index.equals(warm.alpha_scores.index)
                and cold.alpha_scores.columns.equals(warm.alpha_scores.columns)
            ),
            "ic_diagnostics preserved": bool(
                warm.ic_diagnostics is not None
                and np.allclose(
                    cold.ic_diagnostics["mean_IC"].to_numpy(),
                    warm.ic_diagnostics["mean_IC"].to_numpy(),
                )
            ),
        }

        # Key sensitivity: any input that changes the fit must change the key.
        base_key = cache_key(eng, dummy, "daily")
        perturbed = dummy.copy()
        perturbed.iloc[0, 0] += 1e-9
        checks["one changed value -> new key"] = (
            cache_key(eng, perturbed, "daily") != base_key
        )
        checks["different frequency -> new key"] = (
            cache_key(eng, dummy, "daily_nse500") != base_key
        )
        checks["different target -> new key"] = (
            cache_key(TreeAlphaEngine(target_col="tgt_fwd_logret_5b",
                                      model_choice="lgbm"), dummy, "daily") != base_key
        )
        checks["different decile_pct -> new key"] = (
            cache_key(TreeAlphaEngine(decile_pct=0.05, model_choice="lgbm"),
                      dummy, "daily") != base_key
        )
        checks["identical inputs -> same key"] = (
            cache_key(TreeAlphaEngine(model_choice="lgbm"), dummy, "daily") == base_key
        )

    print()
    print("=" * 70)
    for name, ok in checks.items():
        print(f"  {name:<38}: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    assert all(checks.values()), "wf_cache dry-run FAILED"
