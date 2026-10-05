# Daily energy: held-out test year (real data only)

Trained once on 2023-01-01 to 2025-12-31; tested day-ahead on 2026-01-01 to 2026-09-30, which was never used for fitting or tuning. Source: Grid-India daily reports (data/posoco). Weather inputs are archive actuals.

| Model | Yesterday | Same day last week | Skill vs best baseline | MAE (MU) | Days |
|---|---|---|---|---|---|
| **2.72%** | 4.55% | 10.12% | 40% | 3.16 | 273 |

## By month

| Month | Model | Yesterday | Same day last week |
|---|---|---|---|
| 2026-01 | 2.81% | 3.63% | 6.36% |
| 2026-02 | 1.67% | 3.36% | 4.75% |
| 2026-03 | 3.92% | 5.63% | 10.42% |
| 2026-04 | 2.45% | 3.97% | 11.62% |
| 2026-05 | 2.28% | 4.45% | 15.78% |
| 2026-06 | 2.74% | 4.77% | 12.90% |
| 2026-07 | 2.45% | 4.92% | 10.97% |
| 2026-08 | 2.97% | 4.86% | 7.28% |
| 2026-09 | 3.07% | 5.30% | 10.59% |

## Most-used features

lag_1d, lag_2d, dow, feels_max, d_t_max, dew_mean, trend_7d, t_mean
