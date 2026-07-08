import pandas as pd

from config import StrategyConfig
from src.sandbox_run.utils_simulation import (
    generate_synthetic_daily_data,
    save_mock_data,
    get_output_path,
    COUPLED_PARAMS,
)
from src.sandbox_run.ingestion import ingest_poc_data
from src.sandbox_run.tier1_gkx import calculate_momentum_alpha
from src.sandbox_run.tier2_regime import detect_vol_regime
from src.sandbox_run.tier3_execution import execute_poc_strategy
from src.sandbox_run.tier4_dsr import log_to_dsr_ledger

DIVIDER: str = "=" * 70


def run_single_pipeline(cfg: StrategyConfig, run_index: int, total_runs: int) -> None:
    """
    Executes one full pipeline pass for a given StrategyConfig and logs the
    result to the master DSR ledger.

    Args:
        cfg:        Fully-specified strategy configuration for this trial.
        run_index:  1-based position of this run in the batch (for display).
        total_runs: Total number of runs in the batch (for display).
    """
    print(f"\n{DIVIDER}")
    print(
        f"  RUN {run_index}/{total_runs}  |  "
        f"market={cfg.market_type}  |  "
        f"style={cfg.execution_style}  |  "
        f"top_n={cfg.top_n}  bot_n={cfg.bottom_n}  |  "
        f"lookback={cfg.lookback_period}d  |  "
        f"hmm_states={cfg.hmm_states}"
    )
    print(f"  strategy_id : {cfg.strategy_id}")
    print(DIVIDER)

    # ------------------------------------------------------------------ #
    # Stage 2 — Synthetic Data Generation                                #
    # ------------------------------------------------------------------ #
    filepath: str = get_output_path(cfg.market_type)
    print(f"\n[Stage 2] Generating synthetic data  →  {filepath}")
    raw_df: pd.DataFrame = generate_synthetic_daily_data(market_type=cfg.market_type)
    save_mock_data(raw_df, filepath)
    p1 = COUPLED_PARAMS[cfg.market_type]["phase1"]
    p2 = COUPLED_PARAMS[cfg.market_type]["phase2"]
    print(
        f"  {len(raw_df)} bars × {raw_df.shape[1]} cols | "
        f"P1(drift={p1['annual_drift']:+.2f}, vol={p1['annual_vol']:.2f}) → "
        f"P2(drift={p2['annual_drift']:+.2f}, vol={p2['annual_vol']:.2f})"
    )

    # ------------------------------------------------------------------ #
    # Stage 3 — Tier 0: Data Ingestion & Quality Gate                    #
    # ------------------------------------------------------------------ #
    print(f"\n[Stage 3] Tier 0 — Ingestion & anomaly checks")
    asset_df, index_df = ingest_poc_data(filepath)
    print(
        f"  asset_df : {asset_df.shape} | NaNs={asset_df.isna().sum().sum()}\n"
        f"  index_df : {index_df.shape} | NaNs={index_df.isna().sum().sum()}"
    )

    # ------------------------------------------------------------------ #
    # Stage 4 — Tier 1: Momentum Alpha & Cross-Sectional Ranking         #
    # ------------------------------------------------------------------ #
    print(f"\n[Stage 4] Tier 1 — Momentum alpha  (lookback={cfg.lookback_period}d)")
    alpha_ranks = calculate_momentum_alpha(asset_df, lookback_period=cfg.lookback_period)
    print(
        f"  Rank matrix : {alpha_ranks.shape} | "
        f"{alpha_ranks.index[0].date()} → {alpha_ranks.index[-1].date()}"
    )

    # ------------------------------------------------------------------ #
    # Stage 5 — Tier 2: HMM Volatility Regime Detection                  #
    # ------------------------------------------------------------------ #
    print(f"\n[Stage 5] Tier 2 — Gaussian HMM  (n_states={cfg.hmm_states})")
    regimes = detect_vol_regime(index_df, n_states=cfg.hmm_states)
    regime_counts = regimes.value_counts().sort_index()
    print(
        f"  State 0 (Low  Vol) : {regime_counts.get(0, 0)} days\n"
        f"  State 1 (High Vol) : {regime_counts.get(1, 0)} days"
    )

    # ------------------------------------------------------------------ #
    # Stage 6 — Tier 3: Strategy Execution                               #
    # ------------------------------------------------------------------ #
    print(f"\n[Stage 6] Tier 3 — Execution  (style={cfg.execution_style})")
    portfolio_returns = execute_poc_strategy(alpha_ranks, regimes, asset_df, cfg)
    sharpe: float = (
        (portfolio_returns.mean() / portfolio_returns.std()) * (252 ** 0.5)
        if portfolio_returns.std() > 0 else 0.0
    )
    print(
        f"  Observations      : {len(portfolio_returns)}\n"
        f"  Mean daily return : {portfolio_returns.mean():.6f}\n"
        f"  Annualised Sharpe : {sharpe:.4f}\n"
        f"  Cumulative return : {(1 + portfolio_returns).prod() - 1:.4%}"
    )

    # ------------------------------------------------------------------ #
    # Stage 7 — Tier 4: DSR Ledger Logging                               #
    # ------------------------------------------------------------------ #
    print(f"\n[Stage 7] Tier 4 — Logging to DSR ledger")
    log_to_dsr_ledger(cfg.strategy_id, portfolio_returns)


def main() -> None:
    print(DIVIDER)
    print("  QRTF ENGINE — COUPLED-REGIME × EXECUTION STYLE TEST SUITE  (12 runs)")
    print(DIVIDER)

    # ------------------------------------------------------------------ #
    # Stage 1 — Build configuration batch                                #
    # ------------------------------------------------------------------ #
    configs: list[StrategyConfig] = [
        StrategyConfig(
            is_simulation=True,
            market_type=market_type,
            lookback_period=5,
            hmm_states=2,
            execution_style=style,
            top_n=3,
            bottom_n=3,
        )
        for market_type in ["stable_uptrend", "stable_downtrend", "stable_volatile", "volatile_downtrend"]
        for style in ["long_only", "long_short", "dynamic_tilt"]
    ]

    print(f"\n[Stage 1] {len(configs)} configurations queued:")
    for cfg in configs:
        print(f"  {cfg.strategy_id}  →  market={cfg.market_type}  style={cfg.execution_style}")

    # ------------------------------------------------------------------ #
    # Run the pipeline for each configuration                            #
    # ------------------------------------------------------------------ #
    for i, cfg in enumerate(configs, start=1):
        run_single_pipeline(cfg, run_index=i, total_runs=len(configs))

    # ------------------------------------------------------------------ #
    # Final ledger summary                                               #
    # ------------------------------------------------------------------ #
    from src.sandbox_run.tier4_dsr import LEDGER_PATH
    ledger: pd.DataFrame = pd.read_parquet(LEDGER_PATH)

    print(f"\n{DIVIDER}")
    print("  COUPLED-REGIME × EXECUTION STYLE TEST SUITE COMPLETE  (12 runs)")
    print(DIVIDER)
    print(f"\n  Ledger shape : {ledger.shape}  ({ledger.shape[1]} strategy columns)")
    print(f"  Ledger path  : {LEDGER_PATH}")
    # ------------------------------------------------------------------ #
    # Concise summary table: current-run 12 strategies only             #
    # ------------------------------------------------------------------ #
    current_ids: list[str] = [cfg.strategy_id for cfg in configs]
    rows = []
    for sid in current_ids:
        if sid not in ledger.columns:
            continue
        col = ledger[sid].dropna()
        cum = (1 + col).prod() - 1
        sharpe = (col.mean() / col.std() * 252 ** 0.5) if col.std() > 0 else 0.0
        parts = sid.split("_")
        market = "_".join(parts[1:-4])
        style  = parts[-4]
        rows.append((market, style, cum, sharpe))

    print(f"\n  {'Market':<22} {'Style':<5} {'CumRet':>10} {'Sharpe':>8}")
    print(f"  {'-'*22} {'-'*5} {'-'*10} {'-'*8}")
    for market, style, cum, sharpe in rows:
        flag = "  ★" if cum > 0 else ""
        print(f"  {market:<22} {style:<5} {cum:>+10.2%} {sharpe:>8.3f}{flag}")
    print()


if __name__ == "__main__":
    main()
