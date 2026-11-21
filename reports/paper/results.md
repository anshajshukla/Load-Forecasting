# Model comparison, 2017–2025 (development years only)

3281 days scored (2017-01-01 to 2025-12-31), the same days for every model. Each test year is forecast one day ahead by a model trained on the 5 years before it, with recorded weather for the target day. The 2026 holdout is not used. Written by `python -m research.evaluate`.

## Accuracy and significance against LightGBM

| Model | MAPE | RMSE (MU) | CRPS≈ (MU) | 80% interval coverage | DM vs LightGBM (APE) | DM (squared error) |
|---|---|---|---|---|---|---|
| LightGBM (proposed, bias-corrected) | 2.60% | 3.38 | 1.885 | 79.5% | reference | reference |
| LightGBM, no bias correction | 2.67% | 3.47 | 1.901 | 79.6% | -6.18 (p=7.17e-10) | -6.19 (p=6.6e-10) |
| Ridge, same features | 3.00% | 3.66 | 2.083 | 79.5% | -9.74 (p=4.16e-22) | -6.50 (p=9.55e-11) |
| SARIMAX + weather | 2.80% | 3.48 | 2.023 | 86.3% | -5.26 (p=1.53e-07) | -1.93 (p=0.0543) |
| Prophet + weather | 6.94% | 8.46 | 4.978 | 72.0% | -13.48 (p=2.38e-40) | -9.50 (p=3.75e-21) |
| Same as yesterday | 4.51% | 5.71 | 3.293 | 79.7% | -30.81 (p=2.83e-183) | -18.45 (p=2.04e-72) |
| Same day last week | 8.42% | 11.22 | 6.426 | 79.5% | -24.05 (p=7.86e-118) | -15.00 (p=3.2e-49) |

DM statistic: LightGBM's loss minus the other model's; negative means LightGBM is more accurate. Newey–West variance (lag 7), Harvey–Leybourne–Newbold correction, two-sided p-value. CRPS≈ is twice the mean pinball loss over the 10%, 20%, …, 90% quantiles. Naive, ridge and LightGBM quantiles come from each model's last 365 days of errors (known by the day before); SARIMAX's from its Gaussian forecast in logs; Prophet's from its posterior samples; N-HiTS from a multi-quantile loss; Chronos's from the model.

## MAPE by year

| Model | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM (proposed, bias-corrected) | 2.61 | 2.49 | 2.42 | 3.14 | 2.94 | 2.48 | 2.48 | 2.37 | 2.45 |
| LightGBM, no bias correction | 2.75 | 2.59 | 2.41 | 3.25 | 2.94 | 2.57 | 2.58 | 2.46 | 2.45 |
| Ridge, same features | 2.98 | 2.88 | 2.69 | 3.71 | 3.48 | 2.83 | 2.83 | 2.73 | 2.85 |
| SARIMAX + weather | 2.79 | 2.72 | 2.70 | 3.29 | 3.13 | 2.73 | 2.65 | 2.63 | 2.57 |
| Prophet + weather | 5.30 | 4.42 | 4.36 | 15.80 | 8.55 | 6.64 | 5.43 | 4.30 | 7.70 |
| Same as yesterday | 4.70 | 4.43 | 4.56 | 4.67 | 4.68 | 4.80 | 4.45 | 4.05 | 4.23 |
| Same day last week | 7.60 | 6.97 | 8.56 | 9.61 | 9.09 | 8.77 | 9.56 | 7.76 | 7.92 |

## Years in which LightGBM is significantly better (DM on APE, p < 0.05)

| Model | Better | Not different | Worse |
|---|---|---|---|
| LightGBM, no bias correction | 6 | 3 | 0 |
| Ridge, same features | 9 | 0 | 0 |
| SARIMAX + weather | 3 | 6 | 0 |
| Prophet + weather | 9 | 0 | 0 |
| Same as yesterday | 9 | 0 | 0 |
| Same day last week | 9 | 0 | 0 |
