# Daily energy: held-out test year (real data only)

Trained once on 2023-01-01 to 2025-12-31; tested day-ahead on 2026-01-01 to 2026-09-30, which was never used for fitting or tuning. Source: Grid-India daily reports (data/posoco). Weather inputs are archive actuals.

| Model | Yesterday | Same day last week | Skill vs best baseline | MAE (MU) | Days |
|---|---|---|---|---|---|
| **2.82%** | 4.55% | 10.12% | 38% | 3.32 | 273 |

## By month

| Month | Model | Yesterday | Same day last week |
|---|---|---|---|
| 2026-01 | 2.91% | 3.63% | 6.36% |
| 2026-02 | 1.71% | 3.36% | 4.75% |
| 2026-03 | 3.69% | 5.63% | 10.42% |
| 2026-04 | 2.68% | 3.97% | 11.62% |
| 2026-05 | 2.42% | 4.45% | 15.78% |
| 2026-06 | 2.75% | 4.77% | 12.90% |
| 2026-07 | 2.59% | 4.92% | 10.97% |
| 2026-08 | 3.29% | 4.86% | 7.28% |
| 2026-09 | 3.25% | 5.30% | 10.59% |

## Most-used features

lag_1d, d_t_max, trend_7d, lag_2d, dew_mean, dow, feels_max, wind_mean
