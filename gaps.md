# Gaps and fixes

## Fixes already implemented, and their impact

| # | Problem | Fix | Impact |
|---|---|---|---|
| 1 | **Target leakage:** `load_diff_24h + load_lag_24h == load`, rolling windows included the current hour, and `net_load_ramp_rate` tracked the current change. Three such features alone scored 0.68% MAPE. | `pipeline/features.py` builds every load feature from values at or before the forecast origin. `tests/test_no_leakage.py` fails if any feature reacts to future load. | The fake 1.0% (XGBoost) dropped to an honest **2.99%** next-hour and **7.15%** day-ahead on the legacy data. |
| 2 | **Hard-coded headline** (4.09% MAPE typed into the code), and an empty results file | Every number in `reports/` is written by a script; CI re-runs it | All reported numbers are reproducible |
| 3 | **No baselines, single split, unstated horizon** | Walk-forward evaluation (6 × 30-day folds) at 1h and 24h, always shown next to persistence and seasonal naive, plus hour-of-day and daily-peak error | Shows where the model adds value: next hour yes, hourly day-ahead not on real rows |
| 4 | **216 generated rows** (`0` label, constant discom shares) | Blanked in the loader | Removes rows that scored 12.9% / 35.4% error and distorted the averages |
| 5 | **Radiation in UTC** while everything else was in IST (radiation peaked at 07:00) | Shifted by 5.5 h in the loader; new weather downloaded in IST | Together with #4: next hour 2.99% → **2.47%**, day-ahead 7.15% → **5.95%** |
| 6 | **About 250 placeholder values at 23:00** (such as a repeated 1,489.24 MW between ~5,000 MW neighbours) | The loader blanks any hour below 85% of both neighbours | 23:00 error **18% → 1.7%**; overall next hour 2.47% → **1.63%**, day-ahead 5.95% → **5.32%** |
| 7 | **Missing dependency** (`lightgbm.sklearn` needs scikit-learn), so one test failed | Added to `requirements-pipeline.txt` | Test suite passes |
| 8 | **Mostly synthetic training data** | Found and verified real Grid-India daily energy (`data/posoco/`) and real weather (`data/weather/`); a separate real-data path asserts that no synthetic row is used | First result on real data only |
| 9 | **No untouched test set** | Train once on 2023–2025, score 2026 once (`python -m pipeline.daily holdout`); development uses only 2025 inside the training years | **2.82%** day-ahead on 2026 vs 4.55% for same-as-yesterday (2.57% in development), beating the baseline every month |
| 10 | **Unverified data** | Authenticity audits of the legacy data and the daily data (curfew, lockdown, festivals, rain, records, cross-source match) | The synthetic rows are identified, the daily data is shown to be real, and a fake "real" GitHub dataset was rejected |
| 11 | **Possible overfitting** | `scripts/overfit_check.py`: train/test gap, linear baseline, complexity sweep, shuffled target, seeds, no-weather ablation | Skill confirmed real (shuffled data gives 5.03%; ridge 2.94%). The train/test gap (0.82% vs 2.82%) is flagged below. |
| 12 | **No path to real hourly data** | `pipeline/sldc.py` scraper (cached, resumable, header-based parser), `--real-only` and `holdout` modes, tests on sample pages | Ready to run from an Indian IP |

## Still missing or unproven

Roughly in priority order.

### Data

1. **No real hourly data.** The real-data result is daily energy only. Hourly forecasting still runs on the mostly synthetic legacy dataset. `pipeline/sldc.py` is written but has never run against the live site, because delhisldc.org only answers Indian IPs. It needs one run from a machine in India (`python -m pipeline.sldc --start 2023-01-01`), and the parser may need fixing once a real page is seen.
2. **The daily data hasn't been checked against the source PDFs.** The evidence for authenticity is strong (see `analysis.md`), but grid-india.in is blocked from the build environment. Spot-check a few days by hand (19 Jun 2024 should be 176.19 MU).
3. **The daily data comes through a third party.** It was scraped from Grid-India PDFs by Robbie Andrew. Any scraping errors on his side would pass through. Gaps like missing reports are possible before 2023 (none from 2023 on).
4. **No daily peak (MW).** Only energy met (MU) is available for Delhi. Daily peak demand, which grid operators care about most, isn't in this dataset.
5. **Discom-level targets (BRPL, BYPL, NDPL, NDMC, MES) have no real data.** Only the Delhi total is real.
6. **The legacy synthetic data is still in the repo** (`load_forecast_new/`), along with the old notebooks and claims built on it.

### Method

7. **Weather is the recorded actual, not a forecast.** Day-ahead results use the actual weather of the target day. A real system would use a weather forecast, so the true error lies between 2.8% (actual weather) and 3.7% (no weather). Archived weather forecasts, such as Open-Meteo's historical forecast API, would close this.
8. **2026 has now been seen.** Any model change from here (including switching to the smaller model) must be disclosed. A fresh untouched test needs data after 30 Sep 2026.
9. **The chosen model memorises training data** (0.82% train vs 2.82% test). A smaller model is just as accurate, but the switch hasn't been made.
10. **A slight under-forecast in 2026** (−1.37% mean error), probably demand growth that the change-from-yesterday target doesn't capture. Not yet addressed.
11. **One test year.** 2026 is a single held-out period. Rolling-origin tests over several years (2013 data is available) would show whether 2.8% holds in other years too, including the 2020 COVID shock.
12. **No uncertainty estimates.** Point forecasts only; no prediction intervals (for example quantile regression or conformal intervals).
13. **Hourly day-ahead has no proven skill.** On real `historical` rows it loses to seasonal naive (6.24% vs 5.81%).

### Product and repo

14. **The dashboard still shows `np.random` data** and loads no model; its data path doesn't exist.
15. **Hard-coded claims remain** in the old code and docs: 4.09% MAPE (`01_comprehensive_evaluation.py:159`), $4.8M/month and the other business figures.
16. **The Git LFS training arrays** (6 `.npy` files in `phase_3_week_2_neural_networks/data/`) are still pointers in the new repo. They were regenerated identically but couldn't be uploaded (LFS host blocked). They belong to the legacy pipeline and could simply be dropped.
17. **No automated run** of the daily model. A scheduled job (GitHub Actions) that pulls new Grid-India data, retrains, forecasts tomorrow and logs the error would turn this into a live system and give an ever-growing honest test set.
18. **The work isn't merged.** PR #1 (`claude/ml-honest-eval`) is a draft, and `claude/sldc-scraper` (real data, daily model, audits) has no PR yet.
