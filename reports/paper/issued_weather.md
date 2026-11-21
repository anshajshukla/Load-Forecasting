# Ex-ante weather test (issued day-ahead forecasts)

712 days in 2024, 2025. Same LightGBM, trained on recorded weather; the target day's weather is the Open-Meteo forecast issued 24 hours earlier (previous-runs API, `_previous_day1`). Written by `python -m research.issued_weather eval`; 2026 not used.

| Weather for the target day | MAPE | DM vs issued (p) |
|---|---|---|
| Recorded (archive), as in every other table | 2.43% | -3.21 (0.00137) |
| **Issued a day earlier (truly ex-ante)** | **2.65%** | reference |
| No weather features at all | 3.07% | +3.91 (0.000103) |
| Same as yesterday | 4.15% | |

Issued vs recorded daily maximum temperature: mean error -0.27 °C, mean absolute error 1.14 °C.

DM statistic: the row's loss minus the issued-weather loss; negative means the row is more accurate.
