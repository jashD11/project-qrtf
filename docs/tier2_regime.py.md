# Tier 2 — Multivariate Gaussian HMM Regime Detector

**Module:** `src/tier2_regime.py`  
**Entry point:** `detect_vol_regime(index_df, n_states=2) -> pd.Series`

---

## Purpose

Classifies each trading day into a volatility regime (State 0 = Low Vol / Quiet, State 1 = High Vol / Panic) using a 2-state Gaussian Hidden Markov Model. The output regime series drives the execution gate and leverage switch in Tier 3.

---

## Feature Matrix

The HMM is fitted on a 2-column observation matrix of shape `(N_days, 2)`:

| Feature | Construction | Rationale |
|---|---|---|
| Daily log return | `log(INDEX_CLOSE).diff().dropna()` | Instantaneous volatility signal |
| 20-day trend proxy | `log_returns.rolling(20).sum().fillna(0.0)` | Macro momentum context; dampens false panic signals during sustained trends |

```python
trend_proxy = log_returns.rolling(20).sum().fillna(0.0)
obs = np.column_stack([log_returns.to_numpy(), trend_proxy.to_numpy()])
```

The rolling window uses only past data — no look-ahead bias.

---

## Model

```python
GaussianHMM(n_components=2, covariance_type="full", n_iter=200, random_state=42)
```

`covariance_type="full"` gives each state its own full 2×2 covariance matrix, capturing correlation between the instantaneous return and trend features per regime.

---

## Deterministic State Sorting

The EM algorithm assigns arbitrary internal labels; raw state labels are remapped by ascending daily-return variance after every fit:

```python
state_variances = model.covars_[:, 0, 0]   # feature 0 diagonal = return variance
remap[argmin(state_variances)] = 0          # lowest variance  → State 0
remap[argmax(state_variances)] = 1          # highest variance → State 1
```

`covars_` shape is `(n_states, 2, 2)` for the bivariate model; `[:, 0, 0]` isolates the daily-return variance regardless of feature count, keeping the sort invariant to future feature additions.

---

## Output

`pd.Series` of `int32`, DatetimeIndex aligned to `log_returns.index` (one row shorter than the input price series due to `.diff().dropna()`).  
Values: `0` = Low Vol, `1` = High Vol.

---

## Constraints

- `n_states != 2` raises `NotImplementedError` — Tier 3's binary gate is hardcoded for 2-state output.
- The trend proxy's initial 19-day window is zero-filled (`.fillna(0.0)`) rather than dropped, preserving date-index alignment with the return series.
