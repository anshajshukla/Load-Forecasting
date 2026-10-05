# Models used in this project

Every model and method the project has used, what it was for, how it did, and what it taught.

## 1. The original project (legacy, `load_forecast_new/`), superseded

Trained on the old dataset (about 95% synthetic) with leaky features, so **none of these numbers measure real
accuracy**. They are listed so the history is clear.

| Phase | Models | Reported result | What was wrong |
|---|---|---|---|
| Week 1: baselines | XGBoost, LightGBM, Random Forest, Ridge, Lasso, ElasticNet, Gradient Boosting, Prophet, ARIMA | XGBoost "1.01%" best case (6.85% average); LightGBM 0.98%–6.89% | Leaky features (one feature plus another equalled the target); synthetic data |
| Week 2: neural networks | LSTM, GRU, bidirectional LSTM, CNN-LSTM (Conv1D) | — | Sequence arrays scaled on train+val+test together, so the test set leaked into training |
| Week 3: advanced | Transformer / multi-head attention, LSTM variants, "hybrid" Random Forest + Linear, voting ensembles | Hybrid "4.09%" | 4.09% was typed into the code, not measured |
| Week 4 / Phase 4: selection | Optimised LSTM / Random Forest, final selection | "$4.8M/month savings" | Never calculated by any code |

**Lesson:** fifteen model types didn't fix bad data and leakage. Complex architectures on top of leaked features just
memorise the leak faster.

## 2. Leak-free hourly pipeline (`pipeline/`, legacy data)

| Model | Use | Result |
|---|---|---|
| **LightGBM**, one direct model per horizon, predicting the change from the last known value (`y − lag_h`) | Next hour (1h) and day-ahead (24h) | 1.63% at 1h, 5.32% at 24h after the data fixes; on the real `historical` rows 1.65% at 1h, while day-ahead (6.24%) **loses** to seasonal naive (5.81%) |
| Persistence ("same as the last known hour") | Baseline | 5.11% at 1h |
| Seasonal naive ("same hour, last known day") | Baseline | 6.66% |

**Why predict the change, not the level:** tree models can't predict above the highest value they've seen, and
Delhi's peak sets a new record most years. Predicting the change from a recent value sidesteps that.

## 3. Daily model on real data (`pipeline/daily.py`), current

Real Grid-India daily energy plus Open-Meteo weather. Day-ahead forecast of the next day's energy.

| Version | Model | 2026 error | How it was chosen |
|---|---|---|---|
| v1 | LightGBM, 15 leaves / 600 trees, 3 training years | 2.82% | First honest model |
| v1 small | LightGBM, 7 leaves / 300 trees | 2.73% | 2025 development data: same accuracy, smaller train/test gap |
| v2 | LightGBM, 7 leaves / 600 trees, learning rate 0.02, 5 training years, + heat build-up, growth, weekday and holiday features | 2.57% | 2016–2025 yearly test (2.82% → 2.69%) |
| **v2 + post-processing (current)** | v2 + bias correction (half the last 28 days' mean error, capped at ±2%) + rolling 365-day intervals | **2.45%** | 2017–2025 out-of-sample forecasts |

### Comparison models (used to judge, not deployed)

| Model | 2026 error | What it showed |
|---|---|---|
| Same as yesterday | 4.55% | The baseline to beat; the current model is 46% better |
| Same day last week | 10.12% | Weekly patterns alone are weak for daily energy |
| Ridge regression (linear, same features) | 2.61% | Most of the skill comes from the features, not from model complexity |
| 50/50 LightGBM + Ridge ensemble | no gain (2.704% vs 2.707% on 2016–2025) | Left out: no improvement, more complexity |
| Model trained on shuffled targets | 4.74% | Worse than the baseline, so the real model's skill is not memorised noise |

### Post-processing (part of the current model)

| Method | What it does | Impact |
|---|---|---|
| **Bias correction** | Multiplies the forecast by 1 + half the mean relative error of the last 28 days, using only errors known the day before | 2026: 2.57% → 2.45%; bias −0.94% → −0.41% |
| **Correction cap (±2%)** | The correction can never move a forecast by more than 2% | Same accuracy (binds on 15 days in 2017–2025); protects against one unusual month dragging every forecast |
| **Rolling conformal intervals** | 80%/95% ranges from quantiles of the last 365 days' errors | 95% range covers 94.1% of 2026 days (93–97% every year since 2016) |

### Live multi-day forecasting (`pipeline/live.py`)

The same v2 model, stepped forward one day at a time: each day's forecast becomes the next day's "yesterday".
Ranges are widened by the square root of the number of days ahead.

| Days ahead | Error (2025) | 95% range covered |
|---|---|---|
| 1 | 2.26% | 96% |
| 2–3 | 3.75–3.76% | 92–96% |
| 4–7 | 3.90–4.19% | 98–100% |

## Why LightGBM

* Works well on small tabular datasets (about 1,800 training days) with mixed features.
* Handles missing values (the 112 source-missing days and their neighbours) without filling them in.
* Fast and deterministic with a fixed seed and a single thread, so every number can be reproduced exactly.
* Kept deliberately small (7 leaves): bigger trees memorised the training years without improving the test.

## What was not tried, and why

* **Neural networks (LSTM, Transformer) on the real daily data:** about 1,800 training days is small for them, and
  a linear model already reaches 2.61%. They would add complexity with little room to gain.
* **Hourly models on real data:** waiting on the SLDC scrape (needs an Indian IP).

## Modern models compared (Phase 4, `scripts/model_comparison.py`)

Same test for all: every day of 2016–2026 forecast day-ahead, trained models trained on the 5 years before each
year, the same bias correction for all. Run on a free Colab T4 GPU.

| Model | Type | Mean 2016–2025 | 2026 | What it taught |
|---|---|---|---|---|
| **LightGBM + Chronos-2, 50/50 average** | Ensemble | **2.49%** | **2.43%** | Different good models make different mistakes; averaging beats both. Now the live model |
| LightGBM | Gradient boosting on hand-built features | 2.63% | 2.47% | Best single model; good features matter more than architecture |
| Chronos-2 | Foundation model, zero-shot, with weather covariates | 2.72% | 2.82% | Nearly as good with no training on Delhi; better in shock years (2016, 2020) |
| Ridge | Linear, same features | 2.95% | 2.59% | Most of the skill is in the features |
| N-HiTS | Deep learning, with weather | 3.66% | 3.30% | Too little data (≈1,800 days) for deep learning; worst when behaviour shifted (2020–2022) |
| Chronos-Bolt small | Foundation model, zero-shot, demand only | 4.05% | 4.36% | Without weather a foundation model barely beats "same as yesterday" |
| Same as yesterday | Naive | 4.51% | 4.60% | The bar every model must clear |
