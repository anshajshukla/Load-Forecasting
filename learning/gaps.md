# Gaps and fixes

## Fixes implemented, and their impact

| # | Problem | Fix | Impact |
|---|---|---|---|
| 1 | **Target leakage:** `load_diff_24h + load_lag_24h == load`, rolling windows included the current hour, and `net_load_ramp_rate` tracked the current change. Three such features alone scored 0.68% MAPE. | `pipeline/features.py` builds every load feature from values at or before the forecast origin. `tests/test_no_leakage.py` fails if any feature reacts to future load. | The fake 1.0% (XGBoost) became an honest **2.99%** next-hour and **7.15%** day-ahead on the legacy data |
| 2 | **Hard-coded headline** (4.09% MAPE typed into the code), and an empty results file | Every number in `reports/` is written by a script; CI re-runs it | All reported numbers are reproducible |
| 3 | **No baselines, single split, unstated horizon** | Walk-forward at 1h and 24h, next to persistence and seasonal naive, plus hour-of-day and daily-peak error | Shows where the model adds value |
| 4 | **216 generated rows** (`0` label) | Blanked in the loader | Removed rows scoring 12.9% / 35.4% error |
| 5 | **Radiation in UTC** while everything else was in IST | Shifted in the loader; new weather downloaded in IST | Together with #4: next hour 2.99% → **2.47%**, day-ahead 7.15% → **5.95%** |
| 6 | **About 250 placeholder values at 23:00** | Loader blanks hours below 85% of both neighbours | 23:00 error **18% → 1.7%**; next hour **1.63%**, day-ahead **5.32%** |
| 7 | **Missing scikit-learn dependency** | Added to requirements | Tests pass |
| 8 | **Mostly synthetic training data** | Verified real Grid-India daily energy (`data/posoco/`) and real weather; a real-data path that asserts no synthetic row is used | First result on real data |
| 9 | **No untouched test set** | Train on 2023–2025, score 2026 (`python -m pipeline.daily holdout`) | **2.73%** on 2026 vs 4.55% for same-as-yesterday |
| 10 | **Unverified data** | Authenticity audits of the legacy and daily data | Synthetic rows identified; the daily data shown real; a fake "real" GitHub dataset rejected |
| 11 | **Possible overfitting** | `scripts/overfit_check.py` (gap, ridge, complexity, shuffled target, seeds, no-weather), now with a generated verdict | Skill confirmed real (shuffled 4.88%, ridge 2.94%) |
| 12 | **No path to real hourly data** | SLDC scraper, `--real-only`, hourly `holdout` | Ready to run from an Indian IP |
| 13 | **CI failed on every PR** (`pytest` could not import `pipeline`) | `pytest.ini` puts the repo root on the import path | CI green; PR #1 and PR #2 merged |
| 14 | **Model memorised training data** (0.82% train vs 2.82% test) | Switched to 7 leaves / 300 trees, chosen on 2025 development data, order disclosed | Train/test gap 0.82/2.82 → **1.85/2.73**; 2026 error 2.82% → **2.73%** |
| 15 | **Results changed between runs** (2.72% vs 2.73%) | Deterministic single-threaded LightGBM | Identical numbers on every run |
| 16 | **Only one test year** | `scripts/rolling_years.py`: every year 2016–2026, each trained on the prior 3 years (weather extended back to 2013) | Beats same-as-yesterday **11/11 years**, median **2.73%**; worst 2020 (COVID) 3.55% |
| 17 | **No uncertainty estimates** | Split-conformal 80%/95% intervals calibrated on 2025 | 95% band covers **93.8%** of 2026 days; 80% band 74.4% (a little narrow) |
| 18 | **Dashboard showed `np.random` data** and loaded no model | New `app/dashboard.py` reads only repo outputs: 2026 forecasts with intervals, history, every-year table, live log. Procfile, Dockerfile and docker-compose point to it. | The deployed app shows real data |
| 19 | **Hard-coded claims** (4.09% MAPE, $4.8M/month) | Legacy README banner; claims in the legacy dashboard and evaluation script marked as unmeasured | No unmarked false claim remains |
| 20 | **Broken Git LFS pointers** (6 `.npy` files) | Removed with `.gitattributes`; they belonged to the superseded legacy pipeline | No broken files in the repo |
| 21 | **No automated run, and 2026 has been seen** | `pipeline/forecast.py` + `.github/workflows/daily-forecast.yml`: each day refresh Grid-India data and weather, forecast the next day with the real weather *forecast*, and log it in `reports/live/forecasts.csv` (never overwritten), scoring it once the actual is reported | A growing, clean out-of-sample test that uses real weather forecasts |
| 22 | **No root README** | `README.md`: results, quick start, architecture, data sources, repo map, limitations | The project explains itself |
| 23 | **Model could be better** | v2: heat build-up (3/7-day), growth vs last year, weekday ratios and holiday-distance features; 5 training years; learning rate 0.02 / 600 trees. Chosen on the 2016–2025 yearly test only. | Mean 2016–2025 error **2.82% → 2.69%** (8 of 10 years better); 2026 **2.73% → 2.57%**; no-weather 3.59% → 3.35% |
| 24 | **Systematic under-forecast** (−0.94% on 2026) | Multiply each forecast by 1 + half the mean error of the last 28 days, using only errors known the day before; chosen on 2017–2025 | 2017–2025 MAPE 2.668% → **2.600%**; 2026 **2.57% → 2.45%**, bias −0.94% → **−0.41%** |
| 25 | **Intervals too narrow** (80%/95% covered 76.9%/92.7%) | Quantiles of the last 365 days' errors instead of a fixed band from the previous year | 95% band covers **94.1%** of 2026 and 93–97% in every year since 2016; 80% band 77.7% (79% on 2017–2025) |
| 26 | **Correction could overshoot after an unusual month** | Correction capped at ±2% (`CORR_CAP`) | Same accuracy (2.45% on 2026); binds on 15 days in 2017–2025 |
| 27 | **No live UI** | `pipeline/live.py` + dashboard Live tab: live download, retrain, 7-day forecast with ranges, delay labels, honest fallback | Day 1 2.26%, days 2–7 3.75–4.19% (2025) |

## Still open

These can't be closed from the build environment. Each one says what would close it.

1. **No real hourly data.** delhisldc.org only answers Indian IPs. *To close:* run `python -m pipeline.sldc --start 2023-01-01` from a machine in India and push `data/sldc/hourly.csv`; the parser may need one fix once a real page is seen. This also covers the legacy **hourly day-ahead**, which loses to seasonal naive on the real rows (6.24% vs 5.81%).
2. **Daily data not checked against the source PDFs.** grid-india.in is blocked here. *To close:* spot-check a few days by hand (19 Jun 2024 should be 176.19 MU).
3. **No strict day-ahead weather test on past data.** For recent dates, Open-Meteo's archive is itself built from short-range forecasts, so "recorded" and "forecast" weather can't be separated here. *To close:* allow `previous-runs-api.open-meteo.com` (forecasts as issued a day earlier). The live log (fix #21) uses real next-day forecasts and answers this going forward. Bounds until then: 2.45% (archive weather) to about 3.35% (no weather, before bias correction).
4. **No daily peak (MW) and no discom-level real data.** The Grid-India dataset has energy met only for Delhi. *To close:* the SLDC scrape (#1) provides peaks and discom loads.
5. **The daily workflow hasn't run on GitHub yet.** It needs to be triggered once (Actions → daily-forecast → Run workflow) and allowed to push to `main`. It depends on Robbie Andrew's dataset staying updated.
6. **The legacy code is still in the repo** (`load_forecast_new/`), marked as superseded rather than deleted, so the history of the project stays visible.
