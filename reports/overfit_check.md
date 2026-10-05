# Overfitting checks: daily model

**Verdict:** training error 1.85% vs 2.73% on 2026. Test error barely moves across model sizes, is stable across seeds (2.73–2.75%), and collapses to 4.88% (worse than same-as-yesterday) when the target is shuffled, so the skill is real and not memorised noise. A linear model gets 2.94%, so most of the skill comes from the features. The model size (7 leaves, 300 trees) was chosen on the 2025 dev fold, after a first 2026 run with 15 leaves / 600 trees (2.82%); that order is disclosed.

## 1. Train vs test error

| Training (2023-2025, in-sample) | Dev test (2025, trained on 2023-2024) | 2026 test |
|---|---|---|
| 1.85% | 2.60% | 2.73% |

## 2. Simpler models on the same features

| Model | Train | 2026 test |
|---|---|---|
| Ridge regression (linear) | 2.73% | 2.94% |
| Small GBM (4 leaves, 200 trees) | 2.43% | 2.88% |
| Chosen GBM (7 leaves, 300 trees) | 1.85% | 2.73% |
| Same as yesterday | - | 4.55% |

## 3. Complexity sweep

| Leaves | Trees | Train | Dev 2025 | 2026 |
|---|---|---|---|---|
| 4 | 100 | 2.73% | 2.91% | 3.09% |
| 7 | 300 | 1.85% | 2.60% | 2.73% |
| 15 | 600 | 0.80% | 2.67% | 2.83% |
| 31 | 1000 | 0.12% | 2.66% | 2.93% |
| 63 | 2000 | 0.00% | 2.60% | 2.75% |
| 127 | 3000 | 0.00% | 2.58% | 2.77% |

## 4. Shuffled-target test

Training on day-to-day changes shuffled at random gives **4.88%** on 2026, vs 2.73% for the real model and 4.55% for same-as-yesterday. The skill comes from real structure, not memorised noise.

## 5. Seed stability

2026 MAPE over 5 seeds: 2.73% to 2.75% (mean 2.74%).

## 6. Without the target day's weather

Using only yesterday's weather (no weather forecast at all): **3.59%** on 2026. Real day-ahead weather forecasts sit between this and the archive-actual result.

## 7. Bias

Mean error on 2026: -0.95% (positive = over-forecast). Days with max temperature >= 40 °C: 1.90% MAPE over 32 days; other days 2.84%.
