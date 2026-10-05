# Overfitting checks: daily model

**Verdict:** training error 1.93% vs 2.57% on 2026. Test error barely moves across model sizes, is stable across seeds (2.57–2.63%), and collapses to 4.74% (worse than same-as-yesterday) when the target is shuffled, so the skill is real and not memorised noise. A linear model gets 2.61%, so most of the skill comes from the features. The model size and v2 features (7 leaves, 600 trees, training from 2021-01-01) were chosen on 2025 dev data and the 2016-2025 rolling-year test. 2026 was looked at three times in total (v1 large: 2.82%, v1 small: 2.73%, v2), each after the choice was made; that order is disclosed.

## 1. Train vs test error

| Training (2021-2025, in-sample) | Dev test (2025, trained on 2021-2024) | 2026 test |
|---|---|---|
| 1.93% | 2.49% | 2.57% |

## 2. Simpler models on the same features

| Model | Train | 2026 test |
|---|---|---|
| Ridge regression (linear) | 2.84% | 2.61% |
| Small GBM (4 leaves, 200 trees) | 2.77% | 3.01% |
| Chosen GBM (7 leaves, 600 trees) | 1.93% | 2.57% |
| Same as yesterday | - | 4.55% |

## 3. Complexity sweep

| Leaves | Trees | Train | Dev 2025 | 2026 |
|---|---|---|---|---|
| 4 | 100 | 3.12% | 3.08% | 3.33% |
| 7 | 300 | 2.27% | 2.57% | 2.72% |
| 15 | 600 | 1.39% | 2.47% | 2.57% |
| 31 | 1000 | 0.45% | 2.48% | 2.65% |
| 63 | 2000 | 0.00% | 2.48% | 2.57% |
| 127 | 3000 | 0.00% | 2.47% | 2.63% |

## 4. Shuffled-target test

Training on day-to-day changes shuffled at random gives **4.74%** on 2026, vs 2.57% for the real model and 4.55% for same-as-yesterday. The skill comes from real structure, not memorised noise.

## 5. Seed stability

2026 MAPE over 5 seeds: 2.57% to 2.63% (mean 2.60%).

## 6. Without the target day's weather

Using only yesterday's weather (no weather forecast at all): **3.35%** on 2026. Real day-ahead weather forecasts sit between this and the archive-actual result.

## 7. Bias

Mean error on 2026: -0.94% (positive = over-forecast). Days with max temperature >= 40 °C: 1.89% MAPE over 32 days; other days 2.67%.
