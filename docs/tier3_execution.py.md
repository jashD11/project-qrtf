# Tier 3 — Capital Allocation & Execution Engine

**Module:** `src/tier3_execution.py`  
**Entry point:** `execute_poc_strategy(alpha_ranks, regimes, asset_df, cfg) -> pd.Series`

---

## Purpose

Converts cross-sectional momentum ranks (Tier 1) and volatility regime labels (Tier 2) into a vectorized daily return series. All weight construction is matrix-level; no Python loops over time.

---

## Forward Return Convention

```python
forward_returns = asset_df.shift(-1) / asset_df - 1
```

Signal at date T is applied to the T → T+1 close-to-close return. The final row (no T+1 price) becomes all-NaN and is removed by `dropna()` via `min_count=1`.

---

## Base Weight Vectors

```python
top_mask  = ranks <= cfg.top_n                        # top_n best momentum stocks
bot_mask  = ranks >= (n_stocks - cfg.bottom_n + 1)   # bottom_n worst momentum stocks

long_weights  = top_mask.astype(float)  * (1.0 / cfg.top_n)
short_weights = bot_mask.astype(float)  * (1.0 / cfg.bottom_n)
```

Both are `(signal_days × n_stocks)` DataFrames, fully vectorized.

---

## Execution Styles

### `long_only`
```
weight_matrix = long_weights
```
Gross exposure: 1.0. Hard cash stop on State 1 days (position forced to 0).

### `long_short`
```
weight_matrix = long_weights - short_weights
```
Dollar-neutral spread; net exposure ≈ 0 when `top_n == bottom_n`. Hard cash stop on State 1 days.

### `dynamic_tilt` — Adaptive 130/30 with Panic Snap

No hard cash stop. State 1 activates the short leg instead of forcing flat.

**Quiet days (State 0):** 130/30 leverage — amplify conviction during calm markets.  
**Panic days (State 1):** Snap to pure dollar-neutral — remove leverage overhang when vol spikes.

```python
state1_flag = pd.DataFrame(
    np.outer((regimes_aligned == 1).astype(float), np.ones(n_stocks)),
    index=common_dates, columns=alpha_ranks.columns,
)
quiet_flag  = 1.0 - state1_flag          # 1.0 on State 0, 0.0 on State 1

long_scale  = 1.3 * quiet_flag + 1.0 * state1_flag   # 1.3 quiet / 1.0 panic
short_scale = 0.3 * quiet_flag + 1.0 * state1_flag   # 0.3 quiet / 1.0 panic

weight_matrix = long_scale * long_weights - short_scale * short_weights
```

| State | Long mult | Short mult | Net exposure |
|---|---|---|---|
| 0 — Quiet | 1.3× | 0.3× | +1.0 gross long bias |
| 1 — Panic | 1.0× | 1.0× | 0.0 dollar-neutral |

`np.outer` broadcasts the per-day scalar flag into a `(days, stocks)` matrix — no loop.

---

## Regime Gate

```python
# Applied to long_only and long_short only
gross_returns = gross_returns.where(regimes_aligned == 0, other=0.0)
```

`dynamic_tilt` skips this gate; the weight math itself encodes the regime response.

---

## Return Aggregation

```python
gross_returns = (weight_matrix * forward_returns_aligned).sum(axis=1, min_count=1)
portfolio_returns = gross_returns.dropna()
```

`min_count=1` preserves NaN on the last row (all-NaN from `shift(-1)`) so `dropna()` removes exactly one row, yielding 494 observations from 495 signal days.
