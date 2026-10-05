# Overfitting checks: daily model

**Verdict:** the chosen GBM memorises its training data (0.82% in-sample vs 2.82% on 2026), but that memorisation
does not hurt it on unseen data: test error barely moves from 4 to 127 leaves, it is stable across seeds, and it
collapses to worse than baseline when the target is shuffled. A plain linear model gets 2.94%, so almost all of
the skill comes from the features, not from the tree model's capacity. The honest headline is "about 2.8–2.9%
day-ahead, with any reasonable model". A smaller model (7 leaves, 300 trees: 1.85% train, 2.60% on the 2025 dev
fold) gives the same accuracy with a much smaller train/test gap and is the more defensible choice. It wins on the dev fold, which
is how it should be selected, but 2026 has now been seen, so any switch must be stated as such.

## 1. Train vs test error

| Training (2023-2025, in-sample) | Dev test (2025, trained on 2023-2024) | 2026 test |
|---|---|---|
| 0.82% | 2.69% | 2.82% |

## 2. Simpler models on the same features

| Model | Train | 2026 test |
|---|---|---|
| Ridge regression (linear) | 2.74% | 2.94% |
| Small GBM (4 leaves, 200 trees) | 2.43% | 2.88% |
| Chosen GBM (15 leaves, 600 trees) | 0.82% | 2.82% |
| Same as yesterday | - | 4.55% |

## 3. Complexity sweep

| Leaves | Trees | Train | Dev 2025 | 2026 |
|---|---|---|---|---|
| 4 | 100 | 2.73% | 2.91% | 3.09% |
| 7 | 300 | 1.85% | 2.60% | 2.72% |
| 15 | 600 | 0.82% | 2.69% | 2.82% |
| 31 | 1000 | 0.12% | 2.63% | 2.92% |
| 63 | 2000 | 0.00% | 2.59% | 2.77% |
| 127 | 3000 | 0.00% | 2.56% | 2.80% |

## 4. Shuffled-target test

Training on day-to-day changes shuffled at random gives **5.03%** on 2026, vs 2.82% for the real model and 4.55% for same-as-yesterday. The skill comes from real structure, not memorised noise.

## 5. Seed stability

2026 MAPE over 5 seeds: 2.80% to 2.86% (mean 2.83%).

## 6. Without the target day's weather

Using only yesterday's weather (no weather forecast at all): **3.73%** on 2026. Real day-ahead weather forecasts sit between this and the archive-actual result.

## 7. Bias

Mean error on 2026: -1.37% (positive = over-forecast). Days with max temperature >= 40 °C: 2.29% MAPE over 32 days; other days 2.89%.
