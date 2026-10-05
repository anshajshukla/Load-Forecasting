# Delhi electricity demand forecasting

Day-ahead forecast of Delhi's daily electricity demand, trained and tested on **real data only**: Grid-India daily reports and Open-Meteo weather.

| Test | Model (MAPE) | Same as yesterday | Same day last week |
|---|---|---|---|
| Development: walk-forward over 2025 | 2.46% | 4.23% | 7.92% |
| **2026 held out (1 Jan – 30 Sep, 273 days)** | **2.57%** | 4.55% | 10.12% |
| Every year 2016–2026, each trained on the 5 years before | median 2.58% (2.42–3.25%) | beaten 11/11 years | |

* The 95% prediction interval covers 92.7% of 2026 days, and the 80% interval 76.9%; both are slightly narrow.
* Without a weather forecast for the target day, the error is 3.35%. Results above use recorded weather as a stand-in for a forecast.
* A linear model on the same features gets 2.61%, so most of the skill comes from the features, not the model's complexity.
* Model v2 (heat build-up, growth, holiday-distance and weekday features, 5 training years) was chosen on the 2016–2025 yearly test, where it cut mean error from 2.82% to 2.69%; 2026 then improved from 2.73% to 2.57%.

## Quick start

```bash
pip install -r requirements-app.txt
pytest -q                               # leak tests, holdout split, scraper parser
python -m pipeline.daily backtest       # development, 2026 excluded
python -m pipeline.daily holdout        # 2026 test + intervals
python scripts/rolling_years.py         # every year 2016–2026
python scripts/overfit_check.py         # overfitting checks
python -m pipeline.forecast             # forecast tomorrow (needs api.open-meteo.com)
streamlit run app/dashboard.py          # dashboard
```

## How it works

```
Grid-India daily energy met (data/posoco)    Open-Meteo weather, IST (data/weather)
                 │                                         │
                 └──────────────┬──────────────────────────┘
                                ▼
   features for day d: energy from days ≤ d-1 only (lags, 7/28-day means, trend, growth vs last year,
   weekday ratios), weather for d and the days before (3/7-day heat build-up), calendar, holiday distance
                                ▼
   LightGBM (7 leaves, 600 trees, last 5 years) predicts the change from yesterday
                                ▼
   forecast + split-conformal intervals (calibrated on 2025, never on 2026)
```

A test (`tests/test_daily.py`) fails if any feature for day d changes when energy on day d or later changes.

## Data

* `data/posoco/delhi_daily.csv`: Delhi's daily energy met (MU), 2013 to Oct 2026, from [Grid-India daily reports](https://posoco.in/reports/daily-reports/) via [Robbie Andrew's dataset](https://robbieandrew.github.io/india/). It was checked against events a generator would not know: the COVID curfew and lockdown, Holi and Diwali dates, rain dips and the June 2024 record (`reports/data_audit.md`).
* `data/weather/delhi_hourly.csv`: [Open-Meteo archive](https://open-meteo.com/), 2013 to Sep 2026.

## Repo map

| Path | What |
|---|---|
| `pipeline/daily.py` | Daily model: features, backtest, holdout, intervals |
| `pipeline/forecast.py` | Tomorrow's forecast and live scoring (run daily by `.github/workflows/daily-forecast.yml`) |
| `app/dashboard.py` | Streamlit dashboard on real data |
| `pipeline/sldc.py`, `pipeline/real.py` | Real hourly data: SLDC scraper (needs an Indian IP) and the all-real hourly path |
| `pipeline/features.py`, `train.py`, `legacy.py` | Leak-free hourly pipeline (legacy, mostly synthetic data; `PIPELINE.md`) |
| `reports/` | Every reported number, written by scripts |
| `analysis.md`, `gaps.md` | Audit findings, fixes and their impact, open gaps |
| `load_forecast_new/` | Original project, superseded (synthetic data, leakage, hard-coded claims) |

## Limitations

See `gaps.md`. The main ones:
* The real-data results are **daily energy only**. Hourly needs the SLDC scrape from an Indian IP.
* 2026 was looked at before the final model size was chosen (on 2025 data), and this is disclosed. The live forecasts in `reports/live/` are the clean out-of-sample test from now on.
