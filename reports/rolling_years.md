# Rolling-origin test, 2016–2026

For each year: train on the previous 5 years only, score every day of that year day-ahead. Same model, features, settings and bias correction as `pipeline.daily`; real Grid-India data and Open-Meteo archive weather. 2026 runs to 30 Sep.

| Year | Days | Model | Before bias correction | Same as yesterday | Same day last week | Bias | 95% coverage |
|---|---|---|---|---|---|---|---|
| 2016 | 360 | **2.87%** | 2.93% | 4.51% | 7.88% | -0.07% | 97% |
| 2017 | 365 | **2.61%** | 2.75% | 4.70% | 7.60% | -0.30% | 94% |
| 2018 | 365 | **2.49%** | 2.59% | 4.43% | 6.97% | -0.40% | 96% |
| 2019 | 363 | **2.43%** | 2.42% | 4.55% | 8.56% | -0.16% | 95% |
| 2020 | 364 | **3.15%** | 3.25% | 4.67% | 9.61% | +0.74% | 93% |
| 2021 | 365 | **2.94%** | 2.94% | 4.68% | 9.09% | +0.18% | 95% |
| 2022 | 365 | **2.48%** | 2.57% | 4.80% | 8.77% | -0.30% | 95% |
| 2023 | 365 | **2.48%** | 2.58% | 4.45% | 9.56% | -0.33% | 96% |
| 2024 | 366 | **2.37%** | 2.46% | 4.05% | 7.76% | -0.38% | 94% |
| 2025 | 365 | **2.45%** | 2.45% | 4.23% | 7.92% | -0.08% | 95% |
| 2026 | 273 | **2.45%** | 2.57% | 4.55% | 10.12% | -0.41% | 94% |

Median model MAPE 2.48% (range 2.37–3.15%); the model beats same-as-yesterday in 11 of 11 years.
