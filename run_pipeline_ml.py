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
import os
from dataclasses import replace
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
from src.production_ml.tier1_trees import (
    TreeAlphaEngine,
    DECILE_PCT,
    DEFAULT_TARGET,
    WalkForwardResult,
)
from src.production_ml.tier2_regime import (
    RegimeDetector,
    RegimeResult,
    build_features as build_regime_features,
)
from src.production_ml.tier3_execution import (
    execute_ml_strategy,
    load_cost_panels,
    restricts_shorts,
    run_execution,
)
from src.production_ml.tier4_dsr_gate import run_dsr_gate, _to_daily
from src.production_ml.wf_cache import cached_walk_forward
from src.sandbox_run.tier4_dsr import log_to_dsr_ledger

DIVIDER: Final[str] = "=" * 72
DEFAULT_PRODUCTION_LEDGER: Final[str] = "data/trial_database/production_dsr_matrix.parquet"
# Rebound by --ledger (E4). Phase 4b had to move the Phase 3 ledger aside by hand
# because this path was hard-coded; a run should be able to name its own ledger.
PRODUCTION_LEDGER: str = DEFAULT_PRODUCTION_LEDGER
# Gross returns land in a sibling ledger rather than extra columns in the headline
# one: the DSR gate scores every column it finds, and a gross series is not a
# candidate strategy — folding it in would corrupt the cross-trial Sharpe dispersion
# that sets the deflation benchmark.
GROSS_LEDGER: str = DEFAULT_PRODUCTION_LEDGER.replace(".parquet", "_gross.parquet")
SENSITIVITY_LEDGER: Final[str] = "data/trial_database/production_sensitivity_dsr_matrix.parquet"
DIAGNOSTICS_PATH_TMPL: Final[str] = "data/trial_database/tree_fit_diagnostics_{frequency}.csv"
EXEC_DIAGNOSTICS_PATH: str = "data/trial_database/execution_diagnostics.csv"

# Frequencies served by the Phase 4 point-in-time panel (src/phase4_data/). These carry
# a universe mask, delisting exits, and their own regime panel; the Phase 2/3 entries do
# not, and the two families cannot share a run.
PHASE4_FREQUENCIES: Final[frozenset[str]] = frozenset({"daily_nse500"})

# Frozen defaults for every sandbox-only StrategyConfig field (meaningless in
# PRODUCTION_ML, but the shared dataclass requires them / hashes them into the id).
_LIVE_MARKET: Final[str] = "live_nse"
_LIVE_LOOKBACK: Final[int] = 0

# The frozen base config for [SENSITIVITY] scans (phase3 R5: freeze before test).
# Anchored on the headline survivor — daily long_only — so a buffer/threshold scan
# probes the robustness of the cell we actually report, not an off-strategy one.
SENSITIVITY_BASE: Final[dict] = dict(
    frequency="daily", execution_style="long_only", decile_pct=DECILE_PCT,
    rebalance_buffer_mult=2.0,
)


def make_ml_config(
    frequency: str,
    execution_style: str,
    decile_pct: float = DECILE_PCT,
    hmm_states: int = config.HEADLINE_HMM_STATES,
    rebalance_buffer_mult: float = 2.0,
    target_col: str = DEFAULT_TARGET,
    construction: str = "buffer",
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
        target_col=target_col,
        construction=construction,
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
def is_phase4(frequency: str) -> bool:
    """True for frequencies served by the Phase 4 point-in-time panel."""
    return frequency in PHASE4_FREQUENCIES


def load_universe_mask(frequency: str) -> pd.DataFrame | None:
    """
    Point-in-time universe mask for a Phase 4 frequency, else None.

    Missing on a Phase 4 frequency is a hard error rather than a silent fallback: a
    run without the mask would rank each name against every listed stock instead of
    the tradeable 500, which is a different (and untradeable) experiment that would
    otherwise be indistinguishable in the output.
    """
    if not is_phase4(frequency):
        return None
    path = config.ML_CONFIG.phase4.universe_mask_parquet
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run src/phase4_data/universe.py --build before a "
            f"{frequency} run."
        )
    mask = pd.read_parquet(path)
    mask.index = pd.DatetimeIndex(mask.index).normalize()
    return mask


def load_terminal_returns(frequency: str) -> pd.Series | None:
    """Ticker → delisting/acquisition exit return for a Phase 4 frequency, else None."""
    if not is_phase4(frequency):
        return None
    path = config.ML_CONFIG.phase4.lifecycle_csv
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run src/phase4_data/bhavcopy_panel.py --build first."
        )
    life = pd.read_csv(path)
    return life.set_index("ticker")["terminal_return"].dropna()


def phase4_regime_config():
    """
    A ``_RegimeConfig`` pointed at the Phase 4 index and stock panels.

    Tier 2 needs no code change — both Phase 4 artefacts are emitted in the schema its
    loaders already expect, so repointing the two paths is the whole integration.
    """
    p4 = config.ML_CONFIG.phase4
    return replace(
        config.ML_CONFIG.regime,
        index_parquet=p4.index_parquet,
        stock_parquet=p4.ohlcv_parquet,
    )


def compute_signals(
    frequency: str, decile_pct: float, target_col: str = DEFAULT_TARGET
) -> tuple[WalkForwardResult, pd.DataFrame]:
    """Tier 1 tree walk-forward for one frequency. Returns (wf_result, price_wide)."""
    bars = load_ohlcv_multiindex(frequency)
    features = create_features(bars, frequency, universe_mask=load_universe_mask(frequency))
    # Extract the (small) price panel now, then drop the ~4M-row raw OHLCV frame
    # so it is not resident during the memory-critical walk-forward tree loop.
    # price_wide does not depend on the walk-forward, so this reorder is safe.
    price_wide = bars[CLOSE_COL].unstack(level=TICKER_LEVEL)
    del bars
    gc.collect()
    engine = TreeAlphaEngine.from_frequency(
        frequency, target_col=target_col, decile_pct=decile_pct
    )
    # E1: served from disk when this exact fit has been computed before. The fit is a
    # pure function of (features, engine knobs), so the cache is transparent — see
    # src/production_ml/wf_cache.py. QRTF_WF_CACHE=0 forces a cold fit.
    wf = cached_walk_forward(engine, features, frequency)
    return wf, price_wide


def compute_regime(hmm_states: int, frequencies: list[str] | None = None) -> RegimeResult:
    """
    Tier 2 market-regime HMM — daily, frequency-independent, computed once.

    The regime is market-wide, so one fit serves every cell in a run. It does depend on
    *which* market panel is in play, though: a Phase 4 run must decode regimes over the
    2013→2025 NSE-500 history rather than the 68-name 2015→2025 one, or the panic gate
    would be undefined across the added years. Mixing Phase 3 and Phase 4 frequencies in
    one run is therefore rejected rather than silently resolved.
    """
    freqs = frequencies or []
    p4 = [f for f in freqs if is_phase4(f)]
    if p4 and len(p4) != len(freqs):
        raise ValueError(
            f"cannot mix Phase 4 {p4} and Phase 2/3 {[f for f in freqs if f not in p4]} "
            "frequencies in one run — they need different regime panels. Run them "
            "separately."
        )
    cfg = phase4_regime_config() if p4 else config.ML_CONFIG.regime
    return RegimeDetector(n_states=hmm_states).fit_predict(build_regime_features(cfg))


def _print_cost_regime() -> None:
    """
    State the cost model this run charges, up front, in the log.

    Phase 4b's headline was decided by a cost input nobody had written down
    (``slippage_bps = 0.0``, impact never called). Printing the regime before the
    grid runs makes the assumption impossible to lose track of afterwards.
    """
    c = config.ML_CONFIG.cost
    print(f"\n{DIVIDER}\n  COST REGIME\n{DIVIDER}")
    print(f"  statutory one-way          : {c.compose_oneway_bps():.4f} bps")
    print(f"  apply_costs                : {c.apply_costs}")
    print(f"  per-name spread + impact   : {c.charge_per_name}"
          f"{f' (estimator={c.spread_estimator}, AUM=Rs {c.aum_rupees:,.0f})' if c.charge_per_name else ''}")
    print(f"  participation cap          : {c.enforce_participation_cap}"
          f"{f' (<= {c.max_participation:.0%} of ADV)' if c.enforce_participation_cap else ''}")
    print(f"  short borrow               : "
          f"{'tiered ' + str(c.borrow_bps_by_quartile) if c.tiered_borrow else f'flat {c.short_borrow_bps_annual:.0f} bps/yr'}")


def _load_panels_if_needed(styles: list[str]):
    """Read the Phase 4c cost panels once per run, if any of them will be consumed."""
    c = config.ML_CONFIG.cost
    need_shortable = any(restricts_shorts(s) for s in styles)
    if not c.apply_costs or not (
        c.charge_per_name or c.enforce_participation_cap or c.tiered_borrow
        or need_shortable
    ):
        return None
    return load_cost_panels(need_shortable=need_shortable)


def _record_diagnostics(cfg: StrategyConfig, result) -> None:
    """
    Append one row of per-cell execution diagnostics to a CSV (read-only output).

    These are the §6 numbers — effective cost per side, participation, turnover, the
    gross/net split, short-book width. Measured during the run that produced the
    result, so none of them has to be inferred from it later.
    """
    row = dict(
        strategy_id=cfg.strategy_id, frequency=cfg.frequency,
        execution_style=cfg.execution_style, target=cfg.target_col,
        **({"construction": cfg.construction} if cfg.construction != "buffer" else {}),
        cum_net=float((1 + result.net).prod() - 1),
        cum_gross=float((1 + result.gross.dropna()).prod() - 1),
        sharpe_net=_cum_sharpe(result.net)[1],
        sharpe_gross=_cum_sharpe(result.gross.dropna())[1],
        **result.diagnostics,
    )
    os.makedirs(os.path.dirname(EXEC_DIAGNOSTICS_PATH) or ".", exist_ok=True)
    # Read-modify-write rather than append. Cells do not all emit the same keys — only
    # the SLB styles report a short-book width — so appending would write rows wider
    # than the header the first cell established, producing a CSV that cannot be
    # parsed. Rewriting on the union keeps the file valid whatever mix of styles ran.
    frame = pd.DataFrame([row])
    if os.path.exists(EXEC_DIAGNOSTICS_PATH):
        frame = pd.concat([pd.read_csv(EXEC_DIAGNOSTICS_PATH), frame], ignore_index=True)
    frame.to_csv(EXEC_DIAGNOSTICS_PATH, index=False)


def _persist_ic(wf: WalkForwardResult, frequency: str, target_col: str = "") -> None:
    """
    Write the read-only Tier-1 rank-IC fit diagnostic for one frequency.

    The IC is a predictive-skill readout only; a header comment warns against
    misusing it to select cells or tune hyperparameters on OOS data (that is the
    selection-on-test-statistic bias the DSR gate corrects).
    """
    if wf.ic_diagnostics is None:
        return
    tag = f"{frequency}_{target_col.rsplit('_', 1)[-1]}" if target_col else frequency
    path = DIAGNOSTICS_PATH_TMPL.format(frequency=tag)
    with open(path, "w") as fh:
        fh.write("# READ-ONLY fit diagnostic (cross-sectional rank IC). Do NOT use to\n")
        fh.write("# select cells or tune hyperparameters on OOS data — that is the\n")
        fh.write("# selection-on-test-statistic bias the DSR gate exists to correct.\n")
        wf.ic_diagnostics.to_csv(fh)
    print(f"[Tier 1] fit diagnostic written -> {path}")


# --------------------------------------------------------------------------- #
# Headline grid (phase3 R1)
# --------------------------------------------------------------------------- #
def build_headline_grid(
    frequencies: list[str], styles: list[str], targets: list[str],
    constructions: list[str] | None = None,
) -> list[StrategyConfig]:
    """The explicit Cartesian product of the committed [SWEEP] axes."""
    constructions = constructions or ["buffer"]
    return [
        make_ml_config(freq, style, target_col=target, construction=construction)
        for freq, style, target, construction in itertools.product(
            frequencies, styles, targets, constructions
        )
    ]


def run_headline_grid(
    frequencies: list[str], styles: list[str], skip_dsr: bool = False,
    targets: list[str] | None = None, constructions: list[str] | None = None,
) -> None:
    targets = targets or [DEFAULT_TARGET]
    constructions = constructions or ["buffer"]
    configs = build_headline_grid(frequencies, styles, targets, constructions)

    print(DIVIDER)
    print(f"  PRODUCTION_ML HEADLINE SWEEP — {len(configs)} cells "
          f"({len(frequencies)} freq x {len(styles)} styles x {len(targets)} targets"
          f"{f' x {len(constructions)} constructions' if len(constructions) > 1 else ''}, "
          f"hmm_states={config.HEADLINE_HMM_STATES} frozen)")
    print(DIVIDER)
    for cfg in configs:
        print(f"  {cfg.strategy_id}  ->  freq={cfg.frequency}  style={cfg.execution_style}"
              f"  target={cfg.target_col}  construction={cfg.construction}")
    _print_cost_regime()

    # Tier 2 once for the whole run (regime is daily, frequency-independent).
    print(f"\n[Tier 2] Market-regime HMM (n_states={config.HEADLINE_HMM_STATES}) — once")
    regime = compute_regime(config.HEADLINE_HMM_STATES, frequencies)
    panic = regime.panic

    # Cost panels read once per run rather than once per cell — they are the same
    # (date x ticker) Parquets for every style and every target.
    panels = _load_panels_if_needed(styles)

    # Tier 1 once per (frequency, target); fan out Tier 3 + Tier 4 across styles.
    # The target is a fit-time axis, so it joins frequency in the outer loop — only
    # the execution style is free to fan out over a shared fit.
    for frequency, target_col in itertools.product(frequencies, targets):
        print(f"\n{DIVIDER}\n  FREQUENCY: {frequency} | TARGET: {target_col}\n{DIVIDER}")
        print(f"[Tier 1] Tree walk-forward (decile_pct={DECILE_PCT}, "
              f"target={target_col}) — once for {frequency}")
        wf, price_wide = compute_signals(frequency, DECILE_PCT, target_col=target_col)
        _persist_ic(wf, frequency, target_col)
        terminal = load_terminal_returns(frequency)

        # Construction is an execution-time axis like style: it fans out over the
        # same fit, so the Phase 5 grid costs no extra Tier 1 work.
        for style, construction in itertools.product(styles, constructions):
            cfg = make_ml_config(
                frequency, style, target_col=target_col, construction=construction
            )
            print(f"\n[Tier 3] {frequency} | {style} | {target_col} | {construction}")
            result = run_execution(
                wf, panic, price_wide, cfg, terminal_returns=terminal, panels=panels,
            )
            # Compound intraday bars to daily BEFORE logging so every frequency
            # shares the ledger's daily index. Without this, intraday bar
            # timestamps miss the daily index labels and the column lands all-NaN
            # (T=0). Idempotent for already-daily series (one bar/day passes
            # through); the DSR gate re-applies _to_daily harmlessly on read.
            log_to_dsr_ledger(
                cfg.strategy_id, _to_daily(result.net), ledger_path=PRODUCTION_LEDGER
            )
            # E2 — gross banked in the same run. Phase 4b wrote net only, so its
            # gross Sharpe could only be reconstructed by adding an average drag
            # back, and recovering it exactly meant a second 80-minute fit.
            log_to_dsr_ledger(
                f"{cfg.strategy_id}__gross", _to_daily(result.gross),
                ledger_path=GROSS_LEDGER,
            )
            _record_diagnostics(cfg, result)

    _summarize(configs, PRODUCTION_LEDGER, frequencies, styles, targets, constructions)

    # Tier 4 — the credibility gate: deflate each Sharpe for the full grid's
    # multiple-testing count (N = every cell searched, not just what's written).
    # Skipped for partial (per-frequency) runs, where len(configs) would be the
    # wrong N; run `--dsr` once at the end for the authoritative N=column-count.
    if skip_dsr:
        print("\n[Tier 4] DSR gate skipped (--skip-dsr); run `--dsr` after the "
              "full grid is banked for the authoritative N-trial verdict.")
        return
    print()
    # N is the number of configurations the PROGRAM has searched, not the number this
    # run happens to write (D1). ``len(configs)`` was the old default and it silently
    # under-deflates every time: the Phase 4c grid writes 6 columns but stands on 24
    # earlier trials, and scoring it at N=6 would hand back most of the multiple-testing
    # correction the gate exists to apply. The explicit tally in config.TRIAL_LEDGER
    # wins; len(configs) is only the fallback when no tally has been declared.
    honest_n = config.ML_CONFIG.dsr.trials_override or len(configs)
    if honest_n < len(configs):
        raise ValueError(
            f"declared N={honest_n} is smaller than the {len(configs)} cells this run "
            "searched — the trial ledger in config.TRIAL_LEDGER is out of date."
        )
    print(f"[Tier 4] N = {honest_n} searched configurations "
          f"({len(configs)} written by this run) — see config.TRIAL_LEDGER")
    run_dsr_gate(PRODUCTION_LEDGER, n_trials=honest_n)


# --------------------------------------------------------------------------- #
# Sensitivity scan (phase3 R3) — separate entrypoint, separate ledger
# --------------------------------------------------------------------------- #
def run_sensitivity(axis: str, target_col: str = DEFAULT_TARGET) -> None:
    if axis not in config.SENSITIVITY_BANDS:
        raise ValueError(
            f"axis={axis!r} not in SENSITIVITY_BANDS {sorted(config.SENSITIVITY_BANDS)}"
        )
    band = config.SENSITIVITY_BANDS[axis]
    base = dict(SENSITIVITY_BASE)

    print(DIVIDER)
    print(f"  PRODUCTION_ML SENSITIVITY SCAN — axis={axis} over {band}")
    print(f"  frozen base: {base} | target={target_col} | "
          f"hmm_states={config.HEADLINE_HMM_STATES}")
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
            signal_cache[key] = compute_signals(
                frequency, decile_pct, target_col=target_col
            )
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
            target_col=target_col,
        )
        sid = f"SENS_{axis.upper()}_{value}_{cfg.strategy_id}"
        print(f"\n[Sensitivity] {axis}={value}")
        # E3, two latent bugs fixed. (1) terminal_returns was omitted, so on a Phase 4
        # frequency every delisting exit silently vanished — the exact survivorship
        # flattery the point-in-time panel exists to remove. (2) _to_daily was skipped,
        # so any intraday frequency landed an all-NaN ledger column. Both were dormant
        # only because SENSITIVITY_BASE["frequency"] happens to be daily.
        returns = execute_ml_strategy(
            wf, panic, price_wide, cfg,
            terminal_returns=load_terminal_returns(frequency),
        )
        log_to_dsr_ledger(sid, _to_daily(returns), ledger_path=SENSITIVITY_LEDGER)
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
    frequencies: list[str], styles: list[str], targets: list[str] | None = None,
    constructions: list[str] | None = None,
) -> None:
    targets = targets or [DEFAULT_TARGET]
    constructions = constructions or ["buffer"]
    ledger = pd.read_parquet(ledger_path)
    # Map each cell's metrics via the in-memory configs (NOT by parsing the id).
    metrics = {
        (cfg.frequency, cfg.execution_style, cfg.target_col, cfg.construction): _cum_sharpe(
            ledger[cfg.strategy_id]
        )
        for cfg in configs if cfg.strategy_id in ledger.columns
    }

    print(f"\n{DIVIDER}\n  HEADLINE SWEEP COMPLETE — {ledger.shape[1]} ledger column(s)\n{DIVIDER}")
    print("\n  Per-cell: CumRet / Sharpe")
    for target, construction in itertools.product(targets, constructions):
        print(f"\n  target={target}"
              + (f"  construction={construction}" if len(constructions) > 1 else ""))
        print(f"  {'freq':<14}" + "".join(f"{s:>22}" for s in styles))
        for freq in frequencies:
            cells = []
            for style in styles:
                cum, sh = metrics.get((freq, style, target, construction), (0.0, 0.0))
                cells.append(f"{cum:>+11.2%}/{sh:>6.2f}")
            print(f"  {freq:<14}" + "".join(f"{c:>22}" for c in cells))

    # Distribution across each swept axis (hold the other axes' members together).
    def _band(label: str, keyfn) -> None:
        print(f"\n  Distribution across {label}:")
        groups: dict = {}
        for key, (_, sh) in metrics.items():
            groups.setdefault(keyfn(key), []).append(sh)
        for name, shs in groups.items():
            print(f"    {str(name):<22} mean={sum(shs)/len(shs):>6.2f}  "
                  f"min={min(shs):>6.2f}  max={max(shs):>6.2f}  n={len(shs)}")

    if len(frequencies) > 1:
        _band("execution_style, held per frequency", lambda k: (k[0], k[1]))
    _band("execution_style", lambda k: k[1])
    if len(targets) > 1:
        _band("target", lambda k: k[2])
    if len(constructions) > 1:
        _band("construction", lambda k: k[3])

    print("\n  NOTE: hmm_states is not swept — the panic gate is n_states-invariant,")
    print("        so a 2- vs 3-state grid would be return-identical today. Wiring")
    print("        state-conditional execution is deferred (docs/phase3_deferred_hmm.md).")
    print()


# --------------------------------------------------------------------------- #
# Phase 4c frozen grid (docs/phase4c_plan.md §5)
# --------------------------------------------------------------------------- #
def apply_phase4c_regime(args) -> tuple[list[str], list[str], list[str]]:
    """
    Switch the run into the Phase 4c frozen grid and its cost regime.

    The grid and every cost override live in ``config`` (``PHASE4C_*``), not in this
    function and not on a command line, so the configuration that produced a result
    stays recoverable from the repo alone. Explicit ``--frequencies/--styles/--target``
    flags are honoured — a diagnostic variant is allowed — but it must then be written
    to the SENSITIVITY ledger and never promoted, which the printed warning says.
    """
    from dataclasses import replace

    config.ML_CONFIG.cost = replace(
        config.ML_CONFIG.cost, **config.PHASE4C_COST_OVERRIDES
    )
    defaults = (config.SWEEP_FREQUENCIES, config.SWEEP_EXECUTION_STYLES, [DEFAULT_TARGET])
    frequencies = (
        [config.PHASE4C_FREQUENCY] if args.frequencies == defaults[0] else args.frequencies
    )
    styles = config.PHASE4C_STYLES if args.styles == defaults[1] else args.styles
    targets = config.PHASE4C_TARGETS if args.target == defaults[2] else args.target

    if args.ledger == DEFAULT_PRODUCTION_LEDGER:
        args.ledger = config.PHASE4C_LEDGER

    frozen = (
        frequencies == [config.PHASE4C_FREQUENCY]
        and styles == config.PHASE4C_STYLES
        and targets == config.PHASE4C_TARGETS
    )
    print(DIVIDER)
    print("  PHASE 4c FROZEN GRID" if frozen else "  PHASE 4c COST REGIME — MODIFIED GRID")
    print(DIVIDER)
    if not frozen:
        print("  WARNING: the grid was overridden on the command line. This is a")
        print("  DIAGNOSTIC variant, not the frozen grid. It belongs on the")
        print("  sensitivity ledger and must never be promoted — reporting whichever")
        print("  of the two reads better is exactly the selection bias the DSR gate")
        print("  exists to correct (docs/phase4c_plan.md §5).")
    return frequencies, styles, targets


# --------------------------------------------------------------------------- #
# Phase 5 frozen grid (docs/phase5_plan.md §7)
# --------------------------------------------------------------------------- #
def apply_phase5_regime(args) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Switch the run into the Phase 5 frozen grid: the Phase 4c grid and cost regime,
    crossed with every construction in ``config.PHASE5_RUN_CONSTRUCTIONS``.

    The six ``buffer`` cells are the Phase 4c cells re-run into the same ledger, an
    in-run regression check (they must match ``phase4c_dsr_matrix.parquet`` exactly).
    They are already counted in N; only the 12 ``cost_band``/``cost_swap`` cells are
    new, which is what ``config.TRIAL_LEDGER`` charges.
    """
    if args.ledger == DEFAULT_PRODUCTION_LEDGER:
        args.ledger = config.PHASE5_LEDGER
    frequencies, styles, targets = apply_phase4c_regime(args)
    print("  PHASE 5 — constructions: " + ", ".join(config.PHASE5_RUN_CONSTRUCTIONS))
    print(f"  N = {config.TRIALS_SEARCHED} (config.TRIAL_LEDGER)")
    return frequencies, styles, targets, list(config.PHASE5_RUN_CONSTRUCTIONS)


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
        "--dsr", nargs="?", const="", default=None, metavar="LEDGER",
        help="score the DSR credibility gate on an EXISTING ledger (no recompute); "
             "optionally pass a ledger path (default: the production ledger)",
    )
    parser.add_argument(
        "--ledger", default=DEFAULT_PRODUCTION_LEDGER, metavar="PATH",
        help="ledger Parquet this run writes its columns to (E4). Naming a fresh "
             "ledger is how a phase keeps its cells separate without moving files "
             "aside by hand",
    )
    parser.add_argument(
        "--target", nargs="+", default=[DEFAULT_TARGET], metavar="COL",
        help="tree learning target column(s) — a swept axis, not a runtime option. "
             "e.g. tgt_fwd_logret_1b / _5b (a slower signal cuts turnover) / _21b. "
             "Passing more than one makes the target a structural fork of the grid, "
             "which is paid for in N (config.TRIAL_LEDGER)",
    )
    parser.add_argument(
        "--phase4c", action="store_true",
        help="run the Phase 4c frozen grid (docs/phase4c_plan.md §5): the "
             "daily_nse500 panel x {long_only, long_short_slb, dynamic_tilt_slb} x "
             "{5b, 21b}, with per-name spread + impact, the participation cap, an "
             "SLB-restricted short leg and tiered borrow all ON",
    )
    parser.add_argument(
        "--phase5", action="store_true",
        help="run the Phase 5 frozen grid (docs/phase5_plan.md §7): the Phase 4c "
             "grid and cost regime x {buffer, cost_band, cost_swap}. The buffer "
             "cells must reproduce Phase 4c bit-for-bit",
    )
    parser.add_argument(
        "--skip-dsr", action="store_true",
        help="skip the auto-DSR gate at the end of the grid (for per-frequency "
             "partial runs, whose len(configs) is the wrong N); run `--dsr` "
             "separately once the full grid is banked",
    )
    args = parser.parse_args()

    global PRODUCTION_LEDGER, GROSS_LEDGER, EXEC_DIAGNOSTICS_PATH
    frequencies, styles, targets = args.frequencies, args.styles, args.target
    constructions = ["buffer"]

    if args.phase4c and args.phase5:
        parser.error("--phase4c and --phase5 are separate frozen grids; pick one")
    if args.phase4c:
        frequencies, styles, targets = apply_phase4c_regime(args)
    elif args.phase5:
        frequencies, styles, targets, constructions = apply_phase5_regime(args)

    PRODUCTION_LEDGER = args.ledger
    GROSS_LEDGER = args.ledger.replace(".parquet", "_gross.parquet")
    EXEC_DIAGNOSTICS_PATH = args.ledger.replace(".parquet", "_execution_diagnostics.csv")

    if args.dsr is not None:
        run_dsr_gate(args.dsr or args.ledger)
    elif args.sensitivity:
        run_sensitivity(args.sensitivity, target_col=targets[0])
    else:
        run_headline_grid(
            frequencies, styles, skip_dsr=args.skip_dsr, targets=targets,
            constructions=constructions,
        )


if __name__ == "__main__":
    main()
