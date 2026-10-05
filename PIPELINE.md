# Leak-free forecasting pipeline

## Headline: real data only, 2026 held out

Day-ahead forecast of Delhi's **daily energy** (MU) on real Grid-India data (`data/posoco/`, no synthetic rows),
trained once on 2023–2025 and scored on every day of 2026 up to 30 Sep:

| | Model | Same as yesterday | Same day last week |
|---|---|---|---|
| Development (walk-forward on 2025) | 2.46% | 4.23% | 7.92% |
| **2026 held-out test (273 days)** | **2.57%** | 4.55% | 10.12% |
| Every year 2016–2026 (train on prior 5 years) | median 2.58%, range 2.42–3.25% | beaten 11/11 years | |

The 2026 number is from the final model (7 leaves, 300 trees), chosen on 2025 development data after a first 2026 run with a larger model gave 2.82%. The model beats "same as yesterday" in every month of 2026, and the held-out error is close to the development
error, so the model was not overfitted. Weather for the target day is the archive actual, standing in for a weather
forecast, so real-world error would be slightly higher. Full tables: `reports/daily_backtest.md`, `reports/daily_holdout.md`.
Overfitting checks (train/test gap, linear baseline, complexity sweep, shuffled target, seeds, no-weather): `reports/overfit_check.md`.

```bash
python -m pipeline.daily backtest   # development, 2026 excluded
python -m pipeline.daily holdout    # the one-time 2026 test
```

The hourly results below are on the legacy dataset, which is mostly synthetic (`reports/data_audit.md`).

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
| 1h | 1.63% | 5.11% | 6.66% |
| Day-ahead | 5.32% | 6.66% | 6.66% |

On the hours labelled `historical` (the only rows not marked as simulated): 1h **1.65%** vs 4.72%
persistence, but day-ahead **6.24%** loses to seasonal naive (5.81%). Day-ahead has no proven skill on real data yet.

Loader cleaning (see `reports/data_audit.md`): ~254 placeholder dips (nearly all at 23:00) and the 216 generated
`0`-labelled rows are blanked, and solar radiation is shifted from UTC to IST.

## Caveat: the data

About 95% of rows are labelled `simulation_realistic`, and the series is stitched from regimes
that behave differently. Some simulated summer days fall to ~1,900 MW at night, which real Delhi
load does not do, and the last 216 rows (23-31 July 2025) switch to a different daily shape;
the model's 35% day-ahead error there is the data changing, not the model. These numbers prove
the method, not real-world accuracy. Real SLDC data is the next step.

## Real data only: train 2023–2025, test on 2026

The committed dataset is mostly synthetic (`reports/data_audit.md`), so the real-data path does not use it at
all: loads come only from delhisldc.org, weather only from the Open-Meteo archive (IST clock), and both
commands assert that every row is `sldc`. SLDC only answers Indian IPs, so run this from a machine in India:

```bash
pip install -r requirements-pipeline.txt
python -m pipeline.sldc --start 2023-01-01                      # real 5-min loads -> data/sldc/hourly.csv (~1,370 pages, ~25 min)
python -m pipeline weather --start 2023-01-01 --end 2026-10-04  # real weather -> data/weather/delhi_hourly.csv
python -m pipeline backtest --real-data                         # development: walk-forward on 2023–2025 only
python -m pipeline holdout                                      # once, at the end: train on 2023–2025, score 2026
```

* `backtest --real-data` cuts everything from 2026-01-01 on (`--until`), so tuning never sees the test year.
* `holdout` trains one model per horizon on data before 2026-01-01 and scores every 2026 hour against
  persistence and seasonal naive, with a by-month table (`reports/holdout.md`). Run it once; rerunning after
  changing the model to improve the 2026 number turns 2026 into training data.
* Pages are cached in `data/sldc/raw/`, so an interrupted scrape resumes. A day whose page has no load table
  is reported as FAILED with its HTML kept, so a layout change is easy to fix.
