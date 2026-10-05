# Analysis: Delhi load forecasting on real data

This file summarises what was found and fixed. Detailed reports are in `reports/`; every number comes from a script in the repo.

## 1. The original dataset was mostly synthetic

`load_forecast_new/delhi_interaction_enhanced_cleaned.csv` (26,472 hourly rows, Jul 2022 – Jul 2025), from `reports/data_audit.md`:

| Rows | Label | Verdict |
|---|---|---|
| 1,056 (Mar–Apr 2025) | `historical` | Real. The daily totals match Grid-India at r = 0.998. |
| 25,200 | `simulation_realistic` | Synthetic. 74 summer days drop below 3,000 MW overnight, and the discom loads miss the total by up to ±1,950 MW. |
| 216 (Jul 2025) | `0` | Generated. BRPL's share is exactly 0.2800 every hour, and the load doesn't follow temperature. |

There were also two data bugs:
- About 250 placeholder values at 23:00, such as a repeated 1,489.24 MW.
- Solar radiation stored in UTC while everything else is in IST.

The old headline results (1.0% XGBoost, 4.09% MAPE, $4.8M/month) came from leaky features, a hard-coded constant, and figures that no code calculates.

A public GitHub dataset that claimed to be real SLDC data ([delhi-grid-intelligence](https://github.com/lokesh12kewat-source/delhi-grid-intelligence)) turned out to be the output of its own `generate_demo_data.py`, so it was rejected.

## 2. The real data used now

**Load:** `data/posoco/delhi_daily.csv` holds Delhi's daily energy met (MU) from Grid-India's daily reports, 2013 to 2 Oct 2026, with no gaps from 2023 on. It was extracted from Robbie Andrew's processed dataset (github.com/robbieandrew/robbieandrew.github.io).

**Weather:** `data/weather/delhi_hourly.csv` is the Open-Meteo archive in IST, 2023 to 30 Sep 2026.

### Why we believe the daily data is real

A generator built to look like Delhi could get the seasons and the weekly cycle right. It would not know the exact days of the 2020 curfew and lockdown, the right festival dates, or which days it rained.

| Check | Result |
|---|---|
| Janata curfew, Sun 22 Mar 2020 | Falls from 59.1 to **46.1 MU** that exact day |
| COVID lockdown from 25 Mar 2020 | Stays at 42–46 MU through early April. The same dates in 2019 rose from 54 to 74 MU. |
| COVID fiscal year (Apr 2020 – Mar 2021) | 29,385 MU, down 11% from 32,901, then recovers to 30,936 and 34,939 |
| Holi, 21 Mar 2019 | Falls from 60.1 to 46.1 MU that day |
| Diwali (12 Nov 2023, 31 Oct 2024, 20 Oct 2025) | A 15–25% dip on each actual date |
| Heavy monsoon rain (>20 mm, independent weather data) | −4.3% day over day, vs +1.7% on dry days (28 rain days) |
| Record days | 18–19 Jun 2024 (177.7 and 176.2 MU), the heatwave when Delhi set its all-time peak of about 8,656 MW |
| Weekly cycle | Weekdays about 106 MU, Saturday 102.5, Sunday 98.8 |
| Legacy `historical` rows (independent source) | Daily totals match at r = 0.998, ratio 0.994 |

A direct comparison with the source PDFs was not possible, because grid-india.in is blocked from the build environment. To spot-check: Delhi's "Energy Met" on 19 Jun 2024 should read **176.19 MU**.

## 3. Results: day-ahead daily energy, real data only

The model is LightGBM trained on the change from yesterday. Its inputs are past energy (days ≤ d−1), weather for day d and the day before, and the calendar. It was trained once on 2023–2025, and 2026 was used only as the test year.

| | Model | Same as yesterday | Same day last week |
|---|---|---|---|
| Development (walk-forward over 2025) | 2.57% | 4.23% | 7.92% |
| **2026 held-out test (273 days)** | **2.82%** | 4.55% | 10.12% |

It beats "same as yesterday" in every month of 2026, with monthly errors from 1.7% to 3.7%. The test checking that a forecast never sees energy from day d or later passes.

## 4. Overfitting checks

Full tables are in `reports/overfit_check.md`.

| Check | Result |
|---|---|
| Training vs 2026 error | 0.82% vs 2.82%. The model memorises its training data. |
| Linear model (ridge) | 2.94% on 2026. Almost all of the skill comes from the features. |
| Complexity sweep (4 to 127 leaves) | 2026 error stays between 2.7% and 3.1%. Extra capacity doesn't hurt. |
| Shuffled target | 5.03%, worse than the baseline. The skill comes from real structure. |
| 5 seeds | 2.80–2.86% |
| Without the target day's weather | 3.73% |
| Bias on 2026 | −1.37%, a slight under-forecast |

**Verdict:** the model is not overfitting in a way that hurts unseen data, but the large gap between training and test error is hard to defend. A smaller model (7 leaves, 300 trees: 1.85% on training data, 2.60% on the 2025 development data) is just as accurate. It would be chosen on the 2025 development results; since 2026 has already been seen, the switch must be disclosed as made afterwards.

**Interview line:** "On real Grid-India data, day-ahead error is about 2.8–3% on a held-out 2026, and about 3.7% without a weather forecast, against 4.6% for 'same as yesterday'. A linear model gets close, so the gain comes from the features."

## 5. Hourly pipeline (legacy dataset)

The leak-free hourly pipeline (`pipeline/`, `PIPELINE.md`) works, but its numbers are on the mostly synthetic legacy data. On the real `historical` rows: next hour 1.65% vs 4.72% for persistence; day-ahead 6.24%, which loses to seasonal naive at 5.81%. Real hourly data needs the SLDC scraper (`pipeline/sldc.py`) run from an Indian IP.
