# Rolling-origin test, 2016–2026

For each year: train on the previous 3 years only, score every day of that year day-ahead. Same model, features and settings as `pipeline.daily`; real Grid-India data and Open-Meteo archive weather. 2026 runs to 30 Sep.

| Year | Days | Model | Same as yesterday | Same day last week | Bias |
|---|---|---|---|---|---|
| 2016 | 360 | **2.91%** | 4.51% | 7.88% | -0.17% |
| 2017 | 365 | **2.70%** | 4.70% | 7.60% | -0.67% |
| 2018 | 365 | **2.67%** | 4.43% | 6.97% | -0.85% |
| 2019 | 363 | **2.62%** | 4.55% | 8.56% | -0.43% |
| 2020 | 364 | **3.55%** | 4.67% | 9.61% | +1.78% |
| 2021 | 365 | **3.03%** | 4.68% | 9.09% | +0.21% |
| 2022 | 365 | **2.78%** | 4.80% | 8.77% | -0.61% |
| 2023 | 365 | **2.78%** | 4.45% | 9.56% | -1.05% |
| 2024 | 366 | **2.64%** | 4.05% | 7.76% | -1.08% |
| 2025 | 365 | **2.52%** | 4.23% | 7.92% | -0.34% |
| 2026 | 273 | **2.73%** | 4.55% | 10.12% | -0.95% |

Median model MAPE 2.73% (range 2.52–3.55%); the model beats same-as-yesterday in 11 of 11 years.
