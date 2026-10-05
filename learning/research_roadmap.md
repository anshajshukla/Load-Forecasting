# Research roadmap: towards a top-journal paper

**Target:** Applied Energy (first choice), International Journal of Forecasting (alternative).
IEEE Transactions on Power Systems only once hourly/peak data exists (A6).

**Working title:** *Shifting temperature sensitivity of electricity demand across Indian states, 2013–2026:
measurement, drivers and implications for day-ahead forecasting.*

**Contributions the paper must prove:**
1. **Measurement:** how much demand rises per °C of heat, for every state and year, with uncertainty.
2. **Drivers:** why it shifts (air-conditioner ownership, income, humidity, COVID).
3. **Forecasting under change:** a method that adapts to shifting behaviour and beats strong baselines across states.

Status: ☐ not started · ◐ in progress · ☑ done. Update this file as work lands.

---

## A. Data

| # | Advancement | Done when | Effort | Status |
|---|---|---|---|---|
| A1 | **All states (~30) + all-India** daily energy from the same Grid-India dataset | One tidy table: state × day, 2013–2026; gaps counted and never filled; same authenticity checks as Delhi (COVID, festivals) run per state | 1 week | ☐ |
| A2 | **Weather per state**, weighted by where people live (not one city point) | Population-weighted daily weather for every state from Open-Meteo, IST, 2013–2026; documented weights | 1–2 weeks | ☐ |
| A3 | **True day-ahead weather forecasts** (as issued a day earlier) | Archived forecasts for every state; forecast-vs-recorded error reported. *Needs `previous-runs-api.open-meteo.com` allowed* | 1 week | ☐ |
| A4 | **Driver data**: AC ownership (NFHS-4/5 household surveys), per-capita income, Google COVID mobility, humidity / wet-bulb | One driver table per state and year, with sources cited | 2 weeks | ☐ |
| A5 | **Holiday calendars per state** (regional festivals differ) | State-level holiday features; checked against demand dips | 3 days | ☐ |
| A6 | **Hourly / peak data** (SLDC scrapes, Delhi first) | Real hourly series for at least Delhi; needed only for IEEE TPWRS | Needs an Indian IP | ☐ |

## B. Methods

| # | Advancement | Done when | Effort | Status |
|---|---|---|---|---|
| B1 | **Time-varying heat-sensitivity model** (varying-coefficient or state-space) | Sensitivity curve per state over 2013–2026 with confidence bands; Delhi's 1.3% → 5.4% → 2% reproduced | 3–4 weeks | ☐ |
| B2 | **Adaptive forecasting method**: training window chosen automatically + capped recent-error correction, generalised from the Delhi version | Method described in maths, chosen on development years only, beats the fixed 5-year window across states | 3 weeks | ☐ |
| B3 | **Probabilistic forecasts**: quantiles / intervals as a first-class output | Calibrated 80/95% intervals for every state; coverage reported per state and year | 1–2 weeks | ☐ |
| B4 | **Driver analysis**: explain sensitivity shifts with A4 | Panel regression or similar across states and years; effect sizes with uncertainty; robustness checks | 2–3 weeks | ☐ |

## C. Comparisons (every model on the same yearly tests, every state)

| # | Model | Status |
|---|---|---|
| C1 | Same as yesterday, same day last week, seasonal naive | ◐ (Delhi done) |
| C2 | SARIMAX with weather | ☐ |
| C3 | Prophet | ☐ |
| C4 | Ridge / linear with the same features | ◐ (Delhi done) |
| C5 | LightGBM (current model) | ◐ (Delhi done) |
| C6 | N-HiTS / N-BEATS | ☐ |
| C7 | Temporal Fusion Transformer | ☐ |
| C8 | A time-series foundation model (Chronos or TimeGPT), zero-shot | ☐ |
| C9 | The proposed adaptive method (B2) | ☐ |

**Done when:** one results table and figure with all models × all states, compared using the metrics in section D.

## D. Evaluation

| # | Advancement | Done when | Effort | Status |
|---|---|---|---|---|
| D1 | **Rolling-origin tests** for every state, every year | Same protocol as `scripts/rolling_years.py`, generalised to all states | 1 week | ◐ (Delhi done) |
| D2 | **Probabilistic metrics**: CRPS, pinball loss, coverage | Reported for every model that gives intervals | 1 week | ☐ |
| D3 | **Significance tests**: Diebold–Mariano, model confidence set | Every "A beats B" claim has a test behind it | 1 week | ☐ |
| D4 | **Clean, untouched test** | Method frozen and pre-registered (a dated commit) **before** scoring on held-out states and on 6+ months of live forecasts | Runs in parallel | ◐ (live log running for Delhi) |
| D5 | **Ex-ante weather test** (uses A3) | Every headline number reported with forecast weather, not recorded weather | 3 days | ☐ |
| D6 | **Error analysis**: heatwaves, festivals, COVID, monsoon onset | Table of where every model fails, and why | 1 week | ☐ |

## E. Reproducibility and writing

| # | Advancement | Done when | Status |
|---|---|---|---|
| E1 | Code + data archived on **Zenodo with a DOI**; one command reproduces every table | DOI in the paper | ☐ |
| E2 | **Literature review**: Indian and Delhi load forecasting, temperature-sensitivity studies (for example Auffhammer et al.), forecasting under change | 40–60 relevant citations; gap clearly stated | ☐ |
| E3 | **Data citations**: Grid-India, Robbie Andrew's dataset, Open-Meteo, NFHS, Google mobility | All sources and licences listed | ☐ |
| E4 | **Co-author with standing** (power systems or forecasting faculty) | Agreed authorship | ☐ |
| E5 | **Draft**: contribution statement on page 1, journal-quality figures, professional English editing | Full draft reviewed by the co-author | ☐ |
| E6 | **arXiv preprint** (timestamps the work) | Posted before submission | ☐ |
| E7 | **Submission plan**: Applied Energy → IJF → strong mid-tier fallback | Cover letter and suggested reviewers ready | ☐ |

---

## Order of work

1. **Month 1:** A1, A2, A5, D1 (all states with weather, baselines everywhere)
2. **Month 2:** C2–C8, D2, D3 (the full model comparison)
3. **Month 3:** B1, B2, A3, D5 (the sensitivity model and adaptive method; ex-ante weather)
4. **Month 4:** A4, B4, D6 (drivers and error analysis); freeze the method (D4)
5. **Month 5:** E1–E5 (reproducibility package and writing, with co-author)
6. **Month 6:** E6, E7 (preprint, submission)

## Expected odds (rough judgment, not a measurement)

| Stage | Applied Energy / IJF |
|---|---|
| Today (Delhi only) | ~5–10% |
| After sections A1–A2, C, D1–D5 | ~15–25% |
| After everything, with a strong co-author | ~30–40% |

## Already done (foundation from this project)

Leak-free features with tests; verified real data (Grid-India + Open-Meteo); rolling yearly tests 2016–2026;
overfitting checks; calibrated intervals; capped bias correction; training-window evidence; live forecasting
and dashboard (https://loaddelhi.streamlit.app/). See [`changes.md`](changes.md) and [`models.md`](models.md).
