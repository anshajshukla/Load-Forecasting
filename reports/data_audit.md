# Data authenticity audit

Dataset: `load_forecast_new/delhi_interaction_enhanced_cleaned.csv`, 26,472 hourly rows, 2022-07-25 to 2025-07-31.
Every number below comes from `python scripts/data_audit.py`. The comparisons against public records use figures reported in the press. They could not be checked live because SLDC and Open-Meteo are blocked from this environment.

## Verdict

The dataset mixes three different things, and only one of them can be used as real data.

| Rows | `data_source` | Verdict |
|---|---|---|
| 1,056 (Mar–Apr 2025) | `historical` | **Plausibly real SLDC data.** Realistic daily shapes, the discom loads sum to the Delhi total within about 34 MW, and the residuals behave like real data (kurtosis 1.9). |
| 25,200 (Jul 2022 – Jul 2025) | `simulation_realistic` | **Synthetic load, probably fitted to real anchors.** Annual peaks land close to the published records (2024: 8,566 MW on 18 Jun vs the reported ~8,656 MW on 19 Jun; 2025: 8,408 MW vs ~8,442 MW). But 74 of 283 May–Jul days dip below 3,000 MW overnight, which real Delhi summers don't do. The discom loads also miss the total by up to ±1,950 MW, and the residuals are heavy-tailed (kurtosis 5.2). |
| 216 (Jul 2025) | `0` | **Generated.** BRPL's share is exactly 0.2800 on every hour, the discoms sum to 436 MW below the total, the lag-1 residual autocorrelation is negative (−0.43, which looks like white noise), and the daily peak has no correlation with temperature (−0.07). |

The weather columns look like real reanalysis data, probably an Open-Meteo export: dew point matches the Magnus formula from temperature and RH to within 0.08 °C, and May 2024 reaches 45.9 °C, matching that year's heatwave. The `0` rows break this relationship (error 1.9 °C), so their weather was also filled in by other means.

## Bug found: radiation columns are in UTC

Shortwave radiation peaks at 07:00 on the dataset clock, while temperature peaks at 16:00. Solar noon in Delhi is about 12:10 IST, which is 06:40 UTC. So shortwave radiation (and probably the other radiation columns from the same export) is stamped in UTC while everything else is in IST: a 5.5-hour misalignment. Any solar or duck-curve feature built on them describes the wrong hour. Fix: shift these columns by +5:30, or re-download everything with `timezone=Asia/Kolkata`.

## Bug found: placeholder values at 23:00

On about 250 days the 23:00 load drops to a value like 1,489.24 MW (repeated across days) between neighbours near 5,000 MW. These dips caused an 18% error at 23:00 in the first backtest. The loader now blanks any hour below 85% of both neighbours, which brought next-hour MAPE from 2.47% to 1.63%.

## What this means for the results

* Report headline accuracy on the `historical` rows only (1h 1.65% vs 4.72% persistence; day-ahead 6.24%, which **loses** to seasonal naive at 5.81%), and call the rest a synthetic-data backtest.
* The loader already blanks the 216 `0` rows, the 23:00 dips, and shifts radiation to IST.
* Replace the synthetic history with real data, from the SLDC scraper (it needs an Indian IP) or the gated [happyman11/Delhi-SLDC](https://huggingface.co/datasets/happyman11/Delhi-SLDC) dataset, before claiming any accuracy number in an interview.

## Grid-India daily data (`data/posoco/delhi_daily.csv`): authenticity checks

The daily series used for the headline model was checked for events a generator would not reproduce
(the source PDFs at grid-india.in are blocked from this environment, so they could not be compared directly):

| Check | Result |
|---|---|
| Janata curfew, Sun 22 Mar 2020 | 59.1 → **46.1 MU** that exact day |
| COVID lockdown from 25 Mar 2020 | Stays at 42–46 MU through early April; the same dates in 2019 rose from 54 to 74 MU |
| COVID fiscal year (Apr 2020–Mar 2021) | 29,385 MU, down from 32,901 the year before, recovering to 30,936 then 34,939 |
| Holi, 21 Mar 2019 | 60.1 → 46.1 MU that day |
| Diwali (12 Nov 2023, 31 Oct 2024, 20 Oct 2025, different dates every year) | A dip of 15–25% on each actual date |
| Heavy rain in monsoon (>20 mm, independent Open-Meteo data) | Demand falls 4.3% day over day on average, vs +1.7% on dry days (28 rain days) |
| Record day | 18–19 Jun 2024 (177.7 and 176.2 MU), the heatwave when Delhi set its all-time peak of about 8,656 MW |
| Weekly cycle | Weekdays ~106 MU, Saturday 102.5, Sunday 98.8 |
| Legacy `historical` hourly rows (independent source) | Daily totals match at r = 0.998 |

Fiscal-year totals (2023-24: 35,278 MU) are in the range of CEA's planning figures (the LGBR 2023-24 projected
37,068 MU; that is a forecast, not an actual, so it is only a sanity check).
