# Leak-free forecasting pipeline

`pipeline/` replaces the old modelling code's evaluation with one that can be trusted.
The old phase scripts in `load_forecast_new/` are left untouched for reference.

```bash
pip install -r requirements-pipeline.txt
pytest -q tests                      # includes the no-leakage test
python -m pipeline backtest          # writes reports/backtest.md and reports/backtest.json
```

## What was wrong, and what this changes

| # | Problem in the old pipeline | Fix here |
|---|---|---|
| 1 | Features contained the target: `load_diff_24h + load_lag_24h == load`, rolling windows included hour t, `net_load_ramp_rate` correlated 0.999 with the current change, and 19 "daily" features already knew that day's evening peak. Three of those features alone scored 0.68% MAPE. | `pipeline/features.py` builds every load feature from values at or before the forecast origin (`shift >= horizon`). `tests/test_no_leakage.py` perturbs all load values after the origin and fails if any feature moves. |
| 2 | Headline 4.09% MAPE was a hard-coded constant. | Every number in `reports/` is written by `python -m pipeline backtest`; CI re-runs it on each PR. |
| 7 | Week 2 scaler fit on train+val+test, and `bfill` pulled future values backwards. | No scaling is needed (gradient-boosted trees), and missing hours stay missing. |
| 8 | Forecast horizon never stated. | Two explicit horizons: 1 hour ahead and day-ahead (24h), one direct model each. |
| 9 | No naive baselines. | Every result is shown next to persistence (last known value) and seasonal naive (same hour on the last known day). |
| 10 | Six targets averaged into one MAPE, hiding `delhi`. | One target per run (`--target`), `delhi` by default. |
| 11 | Single split. | Walk-forward, expanding window, six 30-day folds; training stops `horizon` hours before each fold. |
| 12 | No peak or hour-of-day error. | Daily-peak MAPE and MAPE by hour of day in every report. |

The model is LightGBM trained on the change from the last known value at the forecast lag
(`y - lag_h`) rather than the level, because trees cannot extrapolate above the training range
and Delhi's peak grows every year.

## Current results (committed dataset, delhi)

| Horizon | Model | Persistence | Seasonal naive |
|---|---|---|---|
| 1h | 2.99% | 6.14% | 7.64% |
| Day-ahead | 7.15% | 7.64% | 7.64% |

On the 1,056 hours labelled `historical` (the only rows not marked as simulated): 1h **2.62%**,
day-ahead **5.55%** vs 6.38% seasonal naive.

## Caveat: the data

About 95% of rows are labelled `simulation_realistic`, and the series is stitched from regimes
that behave differently. Some simulated summer days fall to ~1,900 MW at night, which real Delhi
load does not do, and the last 216 rows (23-31 July 2025) switch to a different daily shape;
the model's 35% day-ahead error there is the data changing, not the model. These numbers prove
the method, not real-world accuracy. Real SLDC data is the next step.
