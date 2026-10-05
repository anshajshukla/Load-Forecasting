# Rolling-origin test, 2016–2026

For each year: train on the previous 5 years only, score every day of that year day-ahead. Same model, features and settings as `pipeline.daily`; real Grid-India data and Open-Meteo archive weather. 2026 runs to 30 Sep.

| Year | Days | Model | Same as yesterday | Same day last week | Bias |
|---|---|---|---|---|---|
| 2016 | 360 | **2.93%** | 4.51% | 7.88% | -0.34% |
| 2017 | 365 | **2.75%** | 4.70% | 7.60% | -0.65% |
| 2018 | 365 | **2.59%** | 4.43% | 6.97% | -0.97% |
| 2019 | 363 | **2.42%** | 4.55% | 8.56% | -0.37% |
| 2020 | 364 | **3.25%** | 4.67% | 9.61% | +1.23% |
| 2021 | 365 | **2.94%** | 4.68% | 9.09% | +0.17% |
| 2022 | 365 | **2.57%** | 4.80% | 8.77% | -0.70% |
| 2023 | 365 | **2.58%** | 4.45% | 9.56% | -0.77% |
| 2024 | 366 | **2.46%** | 4.05% | 7.76% | -0.83% |
| 2025 | 365 | **2.45%** | 4.23% | 7.92% | -0.31% |
| 2026 | 273 | **2.57%** | 4.55% | 10.12% | -0.94% |

Median model MAPE 2.58% (range 2.42–3.25%); the model beats same-as-yesterday in 11 of 11 years.
