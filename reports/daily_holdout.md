# Daily energy: held-out test year (real data only)

Trained once on 2021-01-01 to 2025-12-31; tested day-ahead on 2026-01-01 to 2026-09-30, which was never used for fitting or tuning. Source: Grid-India daily reports (data/posoco). Weather inputs are archive actuals.

| Model | Yesterday | Same day last week | Skill vs best baseline | MAE (MU) | Days |
|---|---|---|---|---|---|
| **2.45%** | 4.55% | 10.12% | 46% | 2.83 | 273 |

## By month

| Month | Model | Yesterday | Same day last week |
|---|---|---|---|
| 2026-01 | 2.34% | 3.63% | 6.36% |
| 2026-02 | 1.62% | 3.36% | 4.75% |
| 2026-03 | 3.82% | 5.63% | 10.42% |
| 2026-04 | 2.07% | 3.97% | 11.62% |
| 2026-05 | 2.13% | 4.45% | 15.78% |
| 2026-06 | 2.86% | 4.77% | 12.90% |
| 2026-07 | 2.11% | 4.92% | 10.97% |
| 2026-08 | 2.52% | 4.86% | 7.28% |
| 2026-09 | 2.52% | 5.30% | 10.59% |

Model before bias correction: 2.57% (bias -0.94%); after: 2.45% (bias -0.41%). The correction uses only errors known the day before.

## Prediction intervals

Quantiles of the last 365 days' errors, each known by the day before.

| Nominal | 2026 coverage | Mean width (MU) |
|---|---|---|
| 80% | 77.7% | 8.3 |
| 95% | 94.1% | 16.2 |

## Most-used features

lag_2d, dow, feels_max, d_t_max, trend_7d, t_mean, lag_1d, t_min
