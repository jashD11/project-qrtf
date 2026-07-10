"""
PRODUCTION_ML sweep orchestrator (Phase 3).

Runs the committed strategy grid from docs/phase3_design_requirements.md and
writes one column per strategy to a dedicated production DSR ledger. Two modes:

    python run_pipeline_ml.py                      # headline [SWEEP] grid
    python run_pipeline_ml.py --sensitivity panic_threshold   # [SENSITIVITY] scan

Headline grid  = frequency x execution_style  (config.SWEEP_*), 12 cells.
    hmm_states is frozen at config.HEADLINE_HMM_STATES (2) and is NOT a sweep
    axis: the execution tier gates only on the n_states-independent ``panic``
    score, so 2- vs 3-state cells would be return-identical today. See
    docs/phase3_deferred_hmm.md.

Compute discipline (phase3 R1/R4):
    - The 12 StrategyConfigs are materialized and printed up front — the grid is
      an explicit Cartesian product, not an implicit side effect.
    - Tier 1 (tree walk-forward) is fit ONCE per frequency and Tier 2 (regime
      HMM) ONCE for the whole run; only Tier 3 fans out across the 3 execution
      styles. Naively looping all 12 cells would refit the expensive stages 3x.
    - The summary reports the distribution across each swept axis, never a single
      "best" cell.

Sensitivity runs (phase3 R3/R5) are a separate entrypoint writing to a separate
ledger, so they never inflate the headline grid's multiple-testing count. All
other knobs are frozen at SENSITIVITY_BASE before the scan (freeze-before-test).
"""

import argparse
import gc
import itertools
from typing import Final

import pandas as pd

import config
from config import StrategyConfig
from src.production_ml.feature_creator import (
    DATE_LEVEL,
    TICKER_LEVEL,
    CLOSE_COL,
    create_features,
)
from src.production_ml.tier1_trees import TreeAlphaEngine, DECILE_PCT, WalkForwardResult
from src.production_ml.tier2_regime import (
    RegimeDetector,
    RegimeResult,
    build_features as build_regime_features,
)
from src.production_ml.tier3_execution import execute_ml_strategy
from src.production_ml.tier4_dsr_gate import run_dsr_gate, _to_daily
from src.sandbox_run.tier4_dsr import log_to_dsr_ledger

DIVIDER: Final[str] = "=" * 72
PRODUCTION_LEDGER: Final[str] = "data/trial_database/production_dsr_matrix.parquet"
SENSITIVITY_LEDGER: Final[str] = "data/trial_database/production_sensitivity_dsr_matrix.parquet"
DIAGNOSTICS_PATH_TMPL: Final[str] = "data/trial_database/tree_fit_diagnostics_{frequency}.csv"

# Frozen defaults for every sandbox-only StrategyConfig field (meaningless in
# PRODUCTION_ML, but the shared dataclass requires them / hashes them into the id).
_LIVE_MARKET: Final[str] = "live_nse"
_LIVE_LOOKBACK: Final[int] = 0

# The frozen base config for [SENSITIVITY] scans (phase3 R5: freeze before test).
SENSITIVITY_BASE: Final[dict] = dict(
    frequency="daily", execution_style="dynamic_tilt", decile_pct=DECILE_PCT,
    rebalance_buffer_mult=2.0,
)


def make_ml_config(
    frequency: str,
    execution_style: str,
    decile_pct: float = DECILE_PCT,
    hmm_states: int = config.HEADLINE_HMM_STATES,
    rebalance_buffer_mult: float = 2.0,
) -> StrategyConfig:
    """A PRODUCTION_ML StrategyConfig with sandbox-only fields pinned to sentinels."""
    return StrategyConfig(
        is_simulation=False,
        market_type=_LIVE_MARKET,
        lookback_period=_LIVE_LOOKBACK,
        hmm_states=hmm_states,
        execution_style=execution_style,
        frequency=frequency,
        decile_pct=decile_pct,
        rebalance_buffer_mult=rebalance_buffer_mult,
    )


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_ohlcv_multiindex(frequency: str) -> pd.DataFrame:
    """Read a frequency's OHLCV Parquet into a sorted ('date', 'ticker') frame."""
    spec = config.freq_spec(frequency)
    df = pd.read_parquet(spec.parquet)
    df = df.rename(columns={"timestamp": DATE_LEVEL})
    return df.set_index([DATE_LEVEL, TICKER_LEVEL]).sort_index()


# --------------------------------------------------------------------------- #
# Per-frequency signal computation (the expensive stages — cached across styles)
# --------------------------------------------------------------------------- #
def compute_signals(frequency: str, decile_pct: float) -> tuple[WalkForwardResult, pd.DataFrame]:
    """Tier 1 tree walk-forward for one frequency. Returns (wf_result, price_wide)."""
    bars = load_ohlcv_multiindex(frequency)
    features = create_features(bars, frequency)
    # Extract the (small) price panel now, then drop the ~4M-row raw OHLCV frame
    # so it is not resident during the memory-critical walk-forward tree loop.
    # price_wide does not depend on the walk-forward, so this reorder is safe.
    price_wide = bars[CLOSE_COL].unstack(level=TICKER_LEVEL)
    del bars
    gc.collect()
    engine = TreeAlphaEngine.from_frequency(frequency, decile_pct=decile_pct)
    wf = engine.run_walk_forward(features)
    return wf, price_wide


def compute_regime(hmm_states: int) -> RegimeResult:
    """Tier 2 market-regime HMM — daily, frequency-independent, computed once."""
    return RegimeDetector(n_states=hmm_states).fit_predict(build_regime_features())


def _persist_ic(wf: WalkForwardResult, frequency: str) -> None:
    """
    Write the read-only Tier-1 rank-IC fit diagnostic for one frequency.

    The IC is a predictive-skill readout only; a header comment warns against
    misusing it to select cells or tune hyperparameters on OOS data (that is the
    selection-on-test-statistic bias the DSR gate corrects).
    """
    if wf.ic_diagnostics is None:
        return
    path = DIAGNOSTICS_PATH_TMPL.format(frequency=frequency)
    with open(path, "w") as fh:
        fh.write("# READ-ONLY fit diagnostic (cross-sectional rank IC). Do NOT use to\n")
        fh.write("# select cells or tune hyperparameters on OOS data — that is the\n")
        fh.write("# selection-on-test-statistic bias the DSR gate exists to correct.\n")
        wf.ic_diagnostics.to_csv(fh)
    print(f"[Tier 1] fit diagnostic written -> {path}")


# --------------------------------------------------------------------------- #
# Headline grid (phase3 R1)
# --------------------------------------------------------------------------- #
def build_headline_grid(frequencies: list[str], styles: list[str]) -> list[StrategyConfig]:
    """The explicit Cartesian product of the committed [SWEEP] axes."""
    return [
        make_ml_config(freq, style)
        for freq, style in itertools.product(frequencies, styles)
    ]


def run_headline_grid(
    frequencies: list[str], styles: list[str], skip_dsr: bool = False
) -> None:
    configs = build_headline_grid(frequencies, styles)

    print(DIVIDER)
    print(f"  PRODUCTION_ML HEADLINE SWEEP — {len(configs)} cells "
          f"({len(frequencies)} freq x {len(styles)} styles, hmm_states="
          f"{config.HEADLINE_HMM_STATES} frozen)")
    print(DIVIDER)
    for cfg in configs:
        print(f"  {cfg.strategy_id}  ->  freq={cfg.frequency}  style={cfg.execution_style}")

    # Tier 2 once for the whole run (regime is daily, frequency-independent).
    print(f"\n[Tier 2] Market-regime HMM (n_states={config.HEADLINE_HMM_STATES}) — once")
    regime = compute_regime(config.HEADLINE_HMM_STATES)
    panic = regime.panic

    # Tier 1 once per frequency; fan out Tier 3 + Tier 4 across styles.
    for frequency in frequencies:
        print(f"\n{DIVIDER}\n  FREQUENCY: {frequency}\n{DIVIDER}")
        print(f"[Tier 1] Tree walk-forward (decile_pct={DECILE_PCT}) — once for {frequency}")
        wf, price_wide = compute_signals(frequency, DECILE_PCT)
        _persist_ic(wf, frequency)

        for style in styles:
            cfg = make_ml_config(frequency, style)
            print(f"\n[Tier 3] {frequency} | {style}")
            returns = execute_ml_strategy(wf, panic, price_wide, cfg)
            # Compound intraday bars to daily BEFORE logging so every frequency
            # shares the ledger's daily index. Without this, intraday bar
            # timestamps miss the daily index labels and the column lands all-NaN
            # (T=0). Idempotent for already-daily series (one bar/day passes
            # through); the DSR gate re-applies _to_daily harmlessly on read.
            daily_returns = _to_daily(returns)
            log_to_dsr_ledger(cfg.strategy_id, daily_returns, ledger_path=PRODUCTION_LEDGER)

    _summarize(configs, PRODUCTION_LEDGER, frequencies, styles)

    # Tier 4 — the credibility gate: deflate each Sharpe for the full grid's
    # multiple-testing count (N = every cell searched, not just what's written).
    # Skipped for partial (per-frequency) runs, where len(configs) would be the
    # wrong N; run `--dsr` once at the end for the authoritative N=column-count.
    if skip_dsr:
        print("\n[Tier 4] DSR gate skipped (--skip-dsr); run `--dsr` after the "
              "full grid is banked for the authoritative N-trial verdict.")
        return
    print()
    run_dsr_gate(PRODUCTION_LEDGER, n_trials=len(configs))


# --------------------------------------------------------------------------- #
# Sensitivity scan (phase3 R3) — separate entrypoint, separate ledger
# --------------------------------------------------------------------------- #
def run_sensitivity(axis: str) -> None:
    if axis not in config.SENSITIVITY_BANDS:
        raise ValueError(
            f"axis={axis!r} not in SENSITIVITY_BANDS {sorted(config.SENSITIVITY_BANDS)}"
        )
    band = config.SENSITIVITY_BANDS[axis]
    base = dict(SENSITIVITY_BASE)

    print(DIVIDER)
    print(f"  PRODUCTION_ML SENSITIVITY SCAN — axis={axis} over {band}")
    print(f"  frozen base: {base} | hmm_states={config.HEADLINE_HMM_STATES}")
    print(DIVIDER)

    # Regime features (daily) built once; the HMM is cheap (~seconds). For a
    # panic_threshold scan the gate quantile changes per value, so we refit per
    # value (the fit is deterministic, so only the gate moves). For other axes
    # the default-threshold regime is computed once and reused.
    regime_feats = build_regime_features()
    regime_default: RegimeResult | None = None

    # Signals depend on frequency + decile_pct. Cache per (frequency, decile_pct)
    # so a decile_pct scan on one frequency reuses within the run where possible.
    signal_cache: dict[tuple[str, float], tuple[WalkForwardResult, pd.DataFrame]] = {}

    rows = []
    for value in band:
        params = dict(base)
        params[axis] = value
        frequency = params["frequency"]
        decile_pct = params["decile_pct"]

        key = (frequency, decile_pct)
        if key not in signal_cache:
            signal_cache[key] = compute_signals(frequency, decile_pct)
        wf, price_wide = signal_cache[key]

        if axis == "panic_threshold":
            panic = RegimeDetector(
                n_states=config.HEADLINE_HMM_STATES, panic_threshold=value
            ).fit_predict(regime_feats).panic
        else:
            if regime_default is None:
                regime_default = RegimeDetector(
                    n_states=config.HEADLINE_HMM_STATES
                ).fit_predict(regime_feats)
            panic = regime_default.panic

        cfg = make_ml_config(
            frequency, params["execution_style"], decile_pct=decile_pct,
            rebalance_buffer_mult=params["rebalance_buffer_mult"],
        )
        sid = f"SENS_{axis.upper()}_{value}_{cfg.strategy_id}"
        print(f"\n[Sensitivity] {axis}={value}")
        returns = execute_ml_strategy(wf, panic, price_wide, cfg)
        log_to_dsr_ledger(sid, returns, ledger_path=SENSITIVITY_LEDGER)
        rows.append((value, returns))

    print(f"\n{DIVIDER}\n  SENSITIVITY DISTRIBUTION — {axis} (report the band, not the peak)\n{DIVIDER}")
    print(f"  {axis:>16} {'CumRet':>10} {'Sharpe':>9} {'Obs':>7}")
    print(f"  {'-'*16} {'-'*10} {'-'*9} {'-'*7}")
    for value, returns in rows:
        cum, sharpe = _cum_sharpe(returns)
        print(f"  {str(value):>16} {cum:>+10.2%} {sharpe:>9.3f} {len(returns):>7}")
    print()


# --------------------------------------------------------------------------- #
# Reporting (phase3 R4) — per-axis distribution, never a single best cell
# --------------------------------------------------------------------------- #
def _cum_sharpe(returns: pd.Series) -> tuple[float, float]:
    col = returns.dropna()
    if col.empty:
        return 0.0, 0.0
    cum = (1 + col).prod() - 1
    sharpe = (col.mean() / col.std() * 252 ** 0.5) if col.std() > 0 else 0.0
    return float(cum), float(sharpe)


def _summarize(
    configs: list[StrategyConfig], ledger_path: str,
    frequencies: list[str], styles: list[str],
) -> None:
    ledger = pd.read_parquet(ledger_path)
    # Map each cell's metrics via the in-memory configs (NOT by parsing the id).
    metrics = {
        (cfg.frequency, cfg.execution_style): _cum_sharpe(ledger[cfg.strategy_id])
        for cfg in configs if cfg.strategy_id in ledger.columns
    }

    print(f"\n{DIVIDER}\n  HEADLINE SWEEP COMPLETE — {ledger.shape[1]} ledger column(s)\n{DIVIDER}")
    print("\n  Per-cell (freq x style): CumRet / Sharpe")
    print(f"  {'freq':<8}" + "".join(f"{s:>22}" for s in styles))
    for freq in frequencies:
        cells = []
        for style in styles:
            cum, sh = metrics.get((freq, style), (0.0, 0.0))
            cells.append(f"{cum:>+11.2%}/{sh:>6.2f}")
        print(f"  {freq:<8}" + "".join(f"{c:>22}" for c in cells))

    # Distribution across each swept axis (hold the other axis's members together).
    print("\n  Distribution across frequency (mean Sharpe per style):")
    for style in styles:
        shs = [metrics[(f, style)][1] for f in frequencies if (f, style) in metrics]
        if shs:
            print(f"    {style:<14} mean={sum(shs)/len(shs):>6.2f}  "
                  f"min={min(shs):>6.2f}  max={max(shs):>6.2f}")
    print("\n  Distribution across execution_style (mean Sharpe per frequency):")
    for freq in frequencies:
        shs = [metrics[(freq, s)][1] for s in styles if (freq, s) in metrics]
        if shs:
            print(f"    {freq:<14} mean={sum(shs)/len(shs):>6.2f}  "
                  f"min={min(shs):>6.2f}  max={max(shs):>6.2f}")

    print("\n  NOTE: hmm_states is not swept — the panic gate is n_states-invariant,")
    print("        so a 2- vs 3-state grid would be return-identical today. Wiring")
    print("        state-conditional execution is deferred (docs/phase3_deferred_hmm.md).")
    print()


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="PRODUCTION_ML sweep orchestrator")
    parser.add_argument(
        "--sensitivity", metavar="AXIS", default=None,
        help=f"run a sensitivity scan instead of the headline grid "
             f"(one of {sorted(config.SENSITIVITY_BANDS)})",
    )
    parser.add_argument(
        "--frequencies", nargs="+", default=config.SWEEP_FREQUENCIES,
        help="restrict the frequency axis (e.g. --frequencies daily for a fast slice)",
    )
    parser.add_argument(
        "--styles", nargs="+", default=config.SWEEP_EXECUTION_STYLES,
        help="restrict the execution_style axis",
    )
    parser.add_argument(
        "--dsr", nargs="?", const=PRODUCTION_LEDGER, default=None, metavar="LEDGER",
        help="score the DSR credibility gate on an EXISTING ledger (no recompute); "
             "optionally pass a ledger path (default: the production ledger)",
    )
    parser.add_argument(
        "--skip-dsr", action="store_true",
        help="skip the auto-DSR gate at the end of the grid (for per-frequency "
             "partial runs, whose len(configs) is the wrong N); run `--dsr` "
             "separately once the full grid is banked",
    )
    args = parser.parse_args()

    if args.dsr is not None:
        run_dsr_gate(args.dsr)
    elif args.sensitivity:
        run_sensitivity(args.sensitivity)
    else:
        run_headline_grid(args.frequencies, args.styles, skip_dsr=args.skip_dsr)


if __name__ == "__main__":
    main()
