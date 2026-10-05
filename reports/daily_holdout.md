# Daily energy: held-out test year (real data only)

Trained once on 2021-01-01 to 2025-12-31; tested day-ahead on 2026-01-01 to 2026-09-30, which was never used for fitting or tuning. Source: Grid-India daily reports (data/posoco). Weather inputs are archive actuals.

| Model | Yesterday | Same day last week | Skill vs best baseline | MAE (MU) | Days |
|---|---|---|---|---|---|
| **2.57%** | 4.55% | 10.12% | 43% | 3.01 | 273 |

## By month

| Month | Model | Yesterday | Same day last week |
|---|---|---|---|
| 2026-01 | 2.29% | 3.63% | 6.36% |
| 2026-02 | 1.65% | 3.36% | 4.75% |
| 2026-03 | 3.88% | 5.63% | 10.42% |
| 2026-04 | 2.22% | 3.97% | 11.62% |
| 2026-05 | 2.11% | 4.45% | 15.78% |
| 2026-06 | 2.94% | 4.77% | 12.90% |
| 2026-07 | 2.28% | 4.92% | 10.97% |
| 2026-08 | 2.74% | 4.86% | 7.28% |
| 2026-09 | 2.99% | 5.30% | 10.59% |

## Prediction intervals

Split-conformal: the band is set from relative errors on the 2025 development walk-forward, then checked on 2026.

| Nominal | 2026 coverage | Band | Mean width (MU) |
|---|---|---|---|
| 80% | 76.9% | -3.1% to +4.0% | 8.3 |
| 95% | 92.7% | -6.9% to +7.0% | 16.3 |

## Most-used features

lag_2d, dow, feels_max, d_t_max, trend_7d, t_mean, lag_1d, t_min
