# Changes: a learning log

Every change made to this project, in order: what was wrong, what was done, the measured impact, and the lesson.
Numbers are day-ahead MAPE unless stated. "Dev" means data used to make choices; "2026" is the held-out test year.

## Where it ended up

| Stage | 2026 error | Notes |
|---|---|---|
| Original project | "1.0%" / "4.09%" | Leaked or hard-coded; on mostly synthetic data |
| First honest model on real data (v1, 15 leaves / 600 trees) | 2.82% | 3 training years |
| v1 smaller (7 leaves / 300 trees) | 2.73% | Less memorisation |
| v2 features + 5 training years | 2.57% | Chosen on 2016–2025 |
| **v2 + recent-error bias correction (current)** | **2.45%** | Chosen on 2017–2025; 46% better than "same as yesterday" (4.55%) |

---

## Phase 1: making the numbers honest (legacy hourly data)

### 1. Removed target leakage
* **Problem:** `load_diff_24h + load_lag_24h` equalled the target exactly; rolling windows included the current hour; a ramp-rate feature tracked the current change. Three such features alone scored 0.68%.
* **Change:** every load feature uses values at or before the forecast origin (`pipeline/features.py`); a test perturbs future load and fails if any feature moves.
* **Impact:** the "1.0%" XGBoost result became an honest **2.99%** next-hour and **7.15%** day-ahead.
* **Lesson:** a result that looks too good usually is. Check whether any feature can be computed from the target, and write a test that perturbs the future.

### 2. Replaced hard-coded results with generated reports
* **Problem:** the 4.09% headline was a constant typed into the code; $4.8M/month was never calculated.
* **Change:** every number in `reports/` is written by a script and re-run in CI.
* **Lesson:** if a number isn't produced by code, it isn't a result.

### 3. Added baselines and walk-forward testing
* **Problem:** a single split, no baselines, horizon not stated.
* **Change:** walk-forward folds at 1h and 24h, always next to persistence ("same as last hour") and seasonal naive.
* **Impact:** showed where the model helps (next hour) and where it doesn't (hourly day-ahead on real rows).
* **Lesson:** a MAPE alone means nothing. Show what a trivial forecast scores.

### 4–6. Found three data bugs
* **Generated rows** (216 rows labelled `0`, with BRPL's share exactly 0.2800 every hour) were blanked.
* **Radiation in UTC** while everything else was in IST: radiation peaked at 07:00. Shifted by 5.5 hours. Together with the generated rows: next hour 2.99% → **2.47%**, day-ahead 7.15% → **5.95%**.
* **Placeholder values at 23:00** (a repeated 1,489.24 MW between ~5,000 MW neighbours): blanked. The 23:00 error went **18% → 1.7%**; next hour **1.63%**.
* **Lesson:** look at errors by hour of day; a single bad hour points straight at a data problem. Check that timestamps agree across sources (solar noon is a free sanity check).

## Phase 2: finding real data

### 7. Audited the data and found it mostly synthetic
* **Problem:** 25,200 of 26,472 rows were labelled `simulation_realistic`. Summer nights dropped below 3,000 MW and the discoms didn't add up to the total.
* **Change:** `reports/data_audit.md` classifies each row group with evidence.
* **Lesson:** verify the data before modelling it. Real data obeys physical and accounting rules (parts sum to the total, demand follows heat).

### 8. Rejected a dataset that claimed to be real
* **Problem:** a public GitHub repo labelled its series "real SLDC data". Its peaks were identical every year (~8,400 MW at 20:00), and it ran past its stated end date.
* **Change:** its own `generate_demo_data.py` produced it, so it was rejected.
* **Lesson:** a label is not provenance. Look for the generator.

### 9. Adopted and verified Grid-India daily data
* **Change:** Delhi's daily energy met from Grid-India reports (via Robbie Andrew's dataset), 2013–2026.
* **Verification:** it shows the Janata curfew (22 Mar 2020: 59.1 → 46.1 MU that day), the lockdown, Holi and Diwali on their actual dates, dips after heavy rain in independent weather data, and the June 2024 record. It matches the legacy real rows at r = 0.998.
* **Lesson:** to tell real data from fake, test for things a generator wouldn't know: dated one-off events.

### 10. Kept a held-out test year
* **Change:** train on 2023–2025, score 2026 (`pipeline.daily holdout`). Development uses only data before 2026.
* **Impact:** first real result, **2.82%** vs 4.55% for "same as yesterday".

## Phase 3: checking the model

### 11. Overfitting checks
* **Change:** `scripts/overfit_check.py` (train/test gap, ridge, complexity sweep, shuffled target, seeds, no-weather).
* **Impact:** the skill is real: shuffled data gives worse than the baseline. But the model scored 0.82% on training data against 2.82% on test.
* **Lesson:** a big train/test gap that doesn't hurt test error is still hard to defend. A linear model scoring almost as well tells you the features carry the skill.

### 12. Smaller model
* **Change:** 7 leaves / 300 trees, chosen on 2025 development data.
* **Impact:** train/test gap 0.82/2.82 → 1.85/2.73; 2026 **2.73%**.
* **Lesson:** choose on development data. When a choice is made after seeing the test, say so.

### 13. Deterministic training
* **Problem:** results moved between runs (2.72% vs 2.73%).
* **Change:** single-threaded, deterministic LightGBM.
* **Lesson:** if reruns differ, small improvements can't be trusted.

### 14. Tested every year, not one
* **Change:** `scripts/rolling_years.py` scores each year from 2016 to 2026 with a model trained on the years before it.
* **Impact:** beat "same as yesterday" in 11 of 11 years. The worst was 2020 (COVID).
* **Lesson:** one test year can be lucky. A yearly test is also a large, honest development set for later choices.

### 15. Prediction intervals
* **Change:** 80%/95% intervals from development-year errors.
* **Impact:** the 95% band covered 93.8%, but the 80% band only 74.4%, so the intervals were too narrow (fixed in #19).

## Phase 4: product and automation

### 16. CI fix
* **Problem:** every PR failed: plain `pytest` couldn't import `pipeline`. Locally, `python -m pytest` hid it.
* **Change:** `pytest.ini` with `pythonpath = .`. Reproduced with `python -P -m pytest`.
* **Lesson:** run tests the way CI runs them.

### 17. Live forecasts, dashboard, cleanup
* A daily GitHub Action logs a forecast for the next day using the real weather *forecast*, never overwritten, and scored once the actual arrives. That gives a clean test now that 2026 has been seen.
* `app/dashboard.py` shows only repo outputs (the old one showed `np.random`).
* Legacy claims are marked as unmeasured; broken LFS files removed; README added.
* **Lesson:** once you've looked at the test set, the only clean test left is the future.

### 18. Dropped a misleading weather test
* **Problem:** comparing "recorded" and "forecast" weather gave identical files. For recent dates, Open-Meteo's archive is built from short-range forecasts.
* **Change:** the comparison was deleted rather than published.
* **Lesson:** a 0.00 difference is a bug signal, not a result.

## Phase 5: improving the model

### 19. Model v2: better features and more history
* **Change:** heat build-up (3- and 7-day temperature), growth vs a year ago, weekday ratios, distance to holidays; 5 training years; learning rate 0.02 / 600 trees.
* **How chosen:** every option scored on 2016–2025 only. A ridge ensemble added nothing (2.704% vs 2.707%), so it was left out.
* **Impact:** 2016–2025 mean **2.82% → 2.69%**; 2026 **2.73% → 2.57%**; no-weather 3.59% → 3.35%.
* **Lesson:** demand responds to heat that builds up over days, not just today's temperature. Keep the simpler model when the complex one doesn't win.

### 20. Bias correction from recent errors
* **Problem:** the model under-forecast slightly in most years (−0.94% on 2026); the growth feature didn't fix it.
* **Change:** multiply each forecast by 1 + half the mean relative error of the last 28 days, using only errors known the day before. A test checks that it never sees the same day's actual.
* **How chosen:** on the 2017–2025 out-of-sample forecasts. Any window from 28 to 91 days gave about the same result (a robust choice, not a lucky one). A full correction (α = 1) overshot; half was better.
* **Impact:** 2017–2025 **2.668% → 2.600%** (bias −0.50% → −0.12%); 2026 **2.57% → 2.45%** (bias −0.94% → −0.41%). Better or equal in every year.
* **Lesson:** a model's recent mistakes are information. Partially correcting by them (shrinking toward zero) beats both ignoring and fully trusting them.

### 21. Intervals from a rolling window of errors
* **Problem:** a fixed band from one previous year was too narrow (80%/95% covered 76.9%/92.7%).
* **Change:** interval quantiles from the last 365 days' errors, each known by the day before.
* **Impact:** the 95% band covers **94.1%** of 2026 and 93–97% in every year since 2016; the 80% band 77.7% (79% on 2017–2025).
* **Lesson:** calibrate intervals on recent, out-of-sample errors. In-sample or old errors make bands too narrow.

---

## Still open

See `gaps.md`. In short: real hourly data needs the SLDC scrape from an Indian IP; the source PDFs need a hand spot-check; and a strict day-ahead weather test needs `previous-runs-api.open-meteo.com`.

## Disclosure

2026 has been scored four times as the model evolved (2.82% → 2.73% → 2.57% → 2.45%). Each choice was made on development data (2025, or the 2016–2025 yearly test) before 2026 was looked at, and the 2026 gains matched what development predicted. Still, 2026 is no longer an untouched test; the daily live log is.
