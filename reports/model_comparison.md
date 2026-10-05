# Model comparison, 2016–2026

Day-ahead MAPE per year. Trained models (Ridge, LightGBM, N-HiTS) are trained on the 5 years before each test year; Chronos models are zero-shot with 512 days of context. All models except the naive ones get the same bias correction. Weather for the target day is the archive record (stands in for a forecast). 2016–2025 is the development period; 2026 runs to 30 Sep. DM = Diebold–Mariano test against LightGBM on absolute percentage errors, all days 2016–2026 (negative = this model better).

| Model | Mean 2016–2025 | 2026 | Before correction (2016–2025) | DM vs LightGBM (p) | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LightGBM (current model) | **2.63%** | 2.47% | 2.69% | – | 2.87 | 2.61 | 2.49 | 2.43 | 3.14 | 2.94 | 2.48 | 2.48 | 2.37 | 2.45 | 2.47 |
| Chronos-2 (zero-shot, with weather) | **2.72%** | 2.82% | 2.73% | +2.9 (0.003) | 2.85 | 2.76 | 2.54 | 2.58 | 3.01 | 2.96 | 3.03 | 2.54 | 2.39 | 2.49 | 2.82 |
| Ridge (same features) | **2.95%** | 2.59% | 3.09% | +9.9 (0.000) | 3.11 | 2.85 | 2.81 | 2.68 | 3.74 | 3.30 | 2.83 | 2.66 | 2.59 | 2.89 | 2.59 |
| N-HiTS (with weather) | **3.66%** | 3.30% | 3.63% | +14.2 (0.000) | 3.66 | 3.56 | 3.07 | 3.31 | 4.30 | 4.73 | 4.44 | 3.29 | 2.88 | 3.41 | 3.30 |
| Chronos-Bolt small (zero-shot, demand only) | **4.05%** | 4.36% | 4.10% | +29.5 (0.000) | 3.93 | 4.02 | 3.66 | 4.00 | 4.42 | 4.36 | 4.35 | 4.12 | 3.74 | 3.93 | 4.36 |
| Same as yesterday | **4.51%** | 4.60% | – | +34.2 (0.000) | 4.51 | 4.70 | 4.43 | 4.55 | 4.67 | 4.68 | 4.80 | 4.45 | 4.05 | 4.23 | 4.60 |
| Same day last week | **8.37%** | 10.22% | – | +48.9 (0.000) | 7.88 | 7.60 | 6.97 | 8.56 | 9.61 | 9.09 | 8.77 | 9.56 | 7.76 | 7.92 | 10.22 |

## Combining LightGBM and Chronos-2

Run on Colab (T4 GPU). The two best models make partly different errors (correlation 0.78), so their bias-corrected
forecasts were averaged (computed from `model_comparison_errors.csv`):

| Combination | Mean 2016–2025 | 2026 | Years better than LightGBM (2016–2025) | DM vs LightGBM (p) |
|---|---|---|---|---|
| LightGBM alone | 2.63% | 2.47% | – | – |
| **50% LightGBM + 50% Chronos-2** | **2.49%** | **2.43%** | 9/10 | −6.5 (<0.001) |
| 70% / 30% | 2.51% | 2.41% | 9/10 | −9.4 (<0.001) |
| 80% / 20% | 2.54% | 2.41% | 10/10 | −10.6 (<0.001) |

The equal-weight average is the pre-set choice (no weight tuned). It is not yet in the live forecast.
