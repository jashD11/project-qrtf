"""
Phase 4c A2 — the ADV and volatility panels the per-name cost model needs.

``tier3_execution`` charges market impact as ``1e4 · coef · σ · √participation``, and
participation is ``position_rupees / ADV``. Both σ and ADV are per-name, per-date
quantities that until now existed only as a transient inside
``capacity.py::load_inputs``. This module is where they live and get persisted, so the
capacity study and the execution path share one definition instead of two copies that
drift apart.

Two panels, both wide ``(date × ticker)``:

    adv_daily.parquet     trailing ``ML_CONFIG.cost.adv_window`` (21) **median** rupee
                          turnover. Median, not mean, for the same reason the universe
                          ranks on a median: one block deal must not make a thin name
                          look tradeable for a month.
    sigma_daily.parquet   trailing 21-day stdev of daily close-to-close returns.

Both are **causal**: the window ends at ``t-1``, so the estimate used to price a trade
on day ``t`` never contains day ``t``'s own bar. Cost is not a signal, but a cost model
that peeks is still a backtest that cannot be run.

Usage
    python src/phase4_data/liquidity.py            # dry-run self-test
    python src/phase4_data/liquidity.py --build    # build both panels from the panel
"""

import argparse
import os
import sys
from typing import Final

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config  # noqa: E402

DEFAULT_PANEL: Final[str] = "data/bhavcopy/panel_daily.parquet"
DEFAULT_ADV: Final[str] = "data/bhavcopy/adv_daily.parquet"
DEFAULT_SIGMA: Final[str] = "data/bhavcopy/sigma_daily.parquet"

VOL_WINDOW: Final[int] = 21   # capacity.py's published curve was built on this


def add_sigma(panel: pd.DataFrame, window: int = VOL_WINDOW) -> pd.DataFrame:
    """
    Attach per-entity daily return and trailing realized vol to a long panel.

    Factored out of ``capacity.py::load_inputs`` verbatim — same grouping key
    (``entity``, the ISIN-linked identity, not the display ticker), same
    ``min_periods == window`` full-window rule — so the published capacity curve is
    unchanged by the extraction.
    """
    out = panel.sort_values(["entity", "date"]).copy()
    out["ret"] = out.groupby("entity", sort=False)["close"].pct_change()
    out["sigma"] = (
        out.groupby("entity", sort=False)["ret"]
        .transform(lambda s: s.rolling(window, min_periods=window).std())
    )
    return out


def _wide(panel: pd.DataFrame, value: str) -> pd.DataFrame:
    """Long panel -> wide (date × ticker) frame for one column."""
    return panel.pivot_table(
        index="date", columns="ticker", values=value, aggfunc="last"
    ).sort_index()


def build_panels(
    panel: pd.DataFrame,
    adv_window: int | None = None,
    vol_window: int = VOL_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the causal ``(date × ticker)`` ADV and sigma panels.

    Both are shifted one bar after the rolling reduction, so the value carried at date
    ``t`` is computed strictly from bars ``t-window .. t-1``.
    """
    adv_window = config.ML_CONFIG.cost.adv_window if adv_window is None else adv_window

    turnover = _wide(panel, "turnover")
    # A non-traded day is absent liquidity, not zero liquidity — a zero would drag the
    # median down and make a halted name look thinner than it is when it resumes.
    turnover = turnover.where(turnover.gt(0))
    adv = turnover.rolling(adv_window, min_periods=max(2, adv_window // 2)).median().shift(1)

    px = _wide(panel, "close")
    sigma = (
        px.pct_change()
        .rolling(vol_window, min_periods=vol_window)
        .std()
        .shift(1)
    )

    print(
        f"[liquidity] adv({adv_window}d median) {adv.shape} | "
        f"sigma({vol_window}d) {sigma.shape} | "
        f"adv coverage {adv.notna().to_numpy().mean():.1%} | "
        f"sigma coverage {sigma.notna().to_numpy().mean():.1%}"
    )
    return adv, sigma


def build(
    panel_path: str = DEFAULT_PANEL,
    out_adv: str = DEFAULT_ADV,
    out_sigma: str = DEFAULT_SIGMA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the panel, build both panels, persist them."""
    if not os.path.exists(panel_path):
        raise FileNotFoundError(f"{panel_path} not found — run the Phase 4 build first.")
    panel = pd.read_parquet(panel_path, columns=["date", "ticker", "close", "turnover"])
    adv, sigma = build_panels(panel)

    for path, frame in ((out_adv, adv), (out_sigma, sigma)):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        frame.to_parquet(path)
        print(f"[liquidity] wrote {path} | {frame.shape[0]:,} dates × {frame.shape[1]:,} tickers")
    return adv, sigma


def load(
    adv_path: str = DEFAULT_ADV, sigma_path: str = DEFAULT_SIGMA
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read both panels, with a clear error naming the build step if either is absent."""
    for p in (adv_path, sigma_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found — run `python src/phase4_data/liquidity.py --build` first."
            )
    adv = pd.read_parquet(adv_path)
    sigma = pd.read_parquet(sigma_path)
    for f in (adv, sigma):
        f.index = pd.DatetimeIndex(f.index).normalize()
    return adv, sigma


# ── Dry run ─────────────────────────────────────────────────────────────────────

def _dry_run() -> None:
    """
    Self-test on a synthetic panel whose answers are known by construction.

    Checks the three properties the cost model depends on: the median is what is
    reported (not the mean), the windows are strictly causal, and a planted future
    spike cannot reach back into an earlier estimate.
    """
    dates = pd.bdate_range("2020-01-01", periods=120)
    rng = np.random.default_rng(0)
    rows = []
    for i, level in enumerate((1e9, 1e7)):
        px = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, len(dates))))
        rows.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i}", "entity": f"E{i}",
            "close": px, "turnover": np.full(len(dates), level),
        }))
    panel = pd.concat(rows, ignore_index=True)

    adv, sigma = build_panels(panel, adv_window=21, vol_window=21)

    # Constant turnover -> the median is that constant, once the window is full.
    assert np.isclose(adv["T0"].iloc[40], 1e9), f"ADV level wrong: {adv['T0'].iloc[40]}"
    assert np.isclose(adv["T1"].iloc[40], 1e7)

    # Causality: nothing is defined before a window exists...
    assert pd.isna(adv["T0"].iloc[0]), "ADV defined on the first bar"
    assert pd.isna(sigma["T0"].iloc[21]), "sigma used the contemporaneous bar"
    assert not pd.isna(sigma["T0"].iloc[22]), "sigma never becomes defined"

    # ...and a spike confined to the future cannot move an earlier estimate.
    spiked = panel.copy()
    spiked.loc[(spiked["ticker"] == "T0") & (spiked["date"] >= dates[60]), "turnover"] *= 1e3
    adv2, _ = build_panels(spiked, adv_window=21, vol_window=21)
    assert np.isclose(adv2["T0"].iloc[60], adv["T0"].iloc[60]), "future turnover leaked back"

    # Median, not mean: a single 1000x day inside the window must barely move it.
    outlier = panel.copy()
    outlier.loc[(outlier["ticker"] == "T0") & (outlier["date"] == dates[30]), "turnover"] *= 1e3
    adv3, _ = build_panels(outlier, adv_window=21, vol_window=21)
    assert np.isclose(adv3["T0"].iloc[40], 1e9), "one block deal moved the ADV (mean, not median?)"

    # The extracted sigma must match capacity.py's own computation exactly.
    cap_sigma = add_sigma(panel, window=21)
    cap_wide = (
        cap_sigma.pivot_table(index="date", columns="ticker", values="sigma")
        .reindex(sigma.index)   # pivot_table drops the all-NaN warm-up rows
        .shift(1)
    )
    assert np.allclose(
        cap_wide["T0"].to_numpy(), sigma["T0"].to_numpy(), equal_nan=True
    ), "extracted sigma diverges from capacity.py's"

    print("\n[liquidity] dry-run OK — median ADV, causal windows, no future leakage")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the ADV / volatility cost panels.")
    ap.add_argument("--build", action="store_true", help="build from the panel (else dry-run)")
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--out-adv", default=DEFAULT_ADV)
    ap.add_argument("--out-sigma", default=DEFAULT_SIGMA)
    args = ap.parse_args()

    if not args.build:
        _dry_run()
        return
    build(args.panel, args.out_adv, args.out_sigma)


if __name__ == "__main__":
    main()
