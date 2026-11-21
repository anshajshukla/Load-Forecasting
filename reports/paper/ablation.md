# Ablation study, 2017–2025 (development years only)

Each row removes one part of the model and re-runs the same yearly protocol (train on the 5 years before each test year, day-ahead, bias correction applied). Written by `python -m research.ablation`; 2026 is not used.

| Variant | MAPE | Change vs full | Before bias correction | DM vs full (p) |
|---|---|---|---|---|
| Full model | 2.60% | +0.00 pp | 2.67% | reference |
| No weather at all | 3.25% | +0.66 pp | 3.34% | -14.07 (1.1e-43) |
| No humidity, rain, cloud, radiation, wind (temperature only) | 2.60% | +0.00 pp | 2.67% | -0.02 (0.982) |
| No heat build-up (3/7-day weather) | 2.62% | +0.02 pp | 2.71% | -3.19 (0.00143) |
| No growth vs last year | 2.60% | -0.00 pp | 2.66% | +0.57 (0.571) |
| No weekday-ratio features | 2.63% | +0.03 pp | 2.71% | -3.69 (0.000224) |
| No holiday features | 2.77% | +0.17 pp | 2.87% | -5.77 (8.66e-09) |
| No calendar at all (weekday, holiday, season) | 3.67% | +1.07 pp | 3.78% | -24.56 (2.09e-122) |
| No long lags (14 and 364 days) | 2.59% | -0.01 pp | 2.65% | +1.23 (0.219) |
| Predict the level, not the change from yesterday | 3.05% | +0.45 pp | 3.38% | -4.32 (1.59e-05) |
| No bias correction | 2.67% | +0.07 pp | 2.67% | -6.18 (6.99e-10) |

Negative DM: the full model is more accurate than the variant.
