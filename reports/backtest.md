# Backtest report

Walk-forward, expanding window, 30-day folds. Weather inputs are archive actuals, so day-ahead
numbers are slightly optimistic versus using a real weather forecast.

| Target | Horizon | Model MAPE | Persistence | Seasonal naive | Skill vs best baseline | MAE (MW) | Daily-peak MAPE (naive) |
|---|---|---|---|---|---|---|---|
| delhi | 1h | 1.63% | 5.11% | 6.66% | 68% | 70 | 1.19% (5.25%) |
| delhi | 24h | 5.32% | 6.66% | 6.66% | 20% | 231 | 4.56% (5.20%) |

Test period: 2025-01-24 00:00:00 to 2025-07-22 22:00:00

## By data source

| Horizon | Source | Hours | Model MAPE | Persistence | Seasonal naive |
|---|---|---|---|---|---|
| 1h | historical | 980 | 1.65% | 4.72% | 5.80% |
| 1h | simulation_realistic | 3158 | 1.63% | 5.24% | 6.92% |
| 24h | historical | 990 | 6.24% | 5.81% | 5.81% |
| 24h | simulation_realistic | 3178 | 5.03% | 6.92% | 6.92% |

## MAPE by hour of day

| Hour | 1h | 24h |
|---|---|---|
| 00 | 1.24% | 5.15% |
| 01 | 1.22% | 4.98% |
| 02 | 1.22% | 5.03% |
| 03 | 0.97% | 5.01% |
| 04 | 0.96% | 4.76% |
| 05 | 1.18% | 4.57% |
| 06 | 1.98% | 4.27% |
| 07 | 2.09% | 4.37% |
| 08 | 1.88% | 4.23% |
| 09 | 1.49% | 4.32% |
| 10 | 1.55% | 4.72% |
| 11 | 1.24% | 4.39% |
| 12 | 1.27% | 4.65% |
| 13 | 1.39% | 4.79% |
| 14 | 1.92% | 4.86% |
| 15 | 1.83% | 5.15% |
| 16 | 1.59% | 5.17% |
| 17 | 1.83% | 5.11% |
| 18 | 1.61% | 5.15% |
| 19 | 1.82% | 5.40% |
| 20 | 2.18% | 6.65% |
| 21 | 3.05% | 8.51% |
| 22 | 1.88% | 9.51% |
| 23 | 1.66% | 8.28% |
