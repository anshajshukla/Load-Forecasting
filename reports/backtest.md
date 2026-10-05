# Backtest report

Walk-forward, expanding window, 30-day folds. Weather inputs are archive actuals, so day-ahead
numbers are slightly optimistic versus using a real weather forecast.

| Target | Horizon | Model MAPE | Persistence | Seasonal naive | Skill vs best baseline | MAE (MW) | Daily-peak MAPE (naive) |
|---|---|---|---|---|---|---|---|
| delhi | 1h | 2.99% | 6.14% | 7.64% | 51% | 111 | 1.73% (5.33%) |
| delhi | 24h | 7.15% | 7.64% | 7.64% | 6% | 272 | 4.52% (5.33%) |

Test period: 2025-02-02 00:00:00 to 2025-07-31 23:00:00

## By data source

| Horizon | Source | Hours | Model MAPE | Persistence | Seasonal naive |
|---|---|---|---|---|---|
| 1h | historical | 1056 | 2.62% | 6.20% | 6.38% |
| 1h | simulation_realistic | 3048 | 2.42% | 5.72% | 7.34% |
| 1h | unlabelled | 216 | 12.92% | 11.76% | 18.08% |
| 24h | historical | 1056 | 5.55% | 6.38% | 6.38% |
| 24h | simulation_realistic | 3048 | 5.70% | 7.34% | 7.34% |
| 24h | unlabelled | 216 | 35.41% | 18.08% | 18.08% |

## MAPE by hour of day

| Hour | 1h | 24h |
|---|---|---|
| 00 | 6.91% | 10.18% |
| 01 | 3.12% | 8.94% |
| 02 | 1.82% | 7.57% |
| 03 | 1.41% | 6.81% |
| 04 | 1.46% | 5.98% |
| 05 | 1.31% | 5.04% |
| 06 | 2.12% | 4.43% |
| 07 | 2.26% | 4.27% |
| 08 | 2.04% | 3.99% |
| 09 | 1.87% | 4.09% |
| 10 | 2.12% | 4.52% |
| 11 | 1.37% | 4.35% |
| 12 | 1.48% | 4.43% |
| 13 | 1.64% | 4.85% |
| 14 | 2.14% | 4.86% |
| 15 | 2.06% | 5.23% |
| 16 | 2.60% | 5.94% |
| 17 | 2.10% | 5.71% |
| 18 | 2.26% | 6.11% |
| 19 | 2.66% | 6.29% |
| 20 | 2.92% | 8.74% |
| 21 | 3.16% | 11.69% |
| 22 | 2.83% | 12.69% |
| 23 | 18.16% | 24.93% |
