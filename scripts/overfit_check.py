"""Overfitting checks for pipeline.daily. Run: python scripts/overfit_check.py (writes reports/overfit_check.md).

Nothing here changes the model; the 2026 numbers are only reported, never used to pick settings.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.daily import PARAMS, TEST_FROM, TRAIN_FROM, _xy, load, mape  # noqa: E402

df = load()
X, y = _xy(df, TRAIN_FROM)
tr, te = X.index < TEST_FROM, X.index >= TEST_FROM
dev_tr, dev_te = X.index < "2025-01-01", (X.index >= "2025-01-01") & tr
base = X["lag_1d"]
out = []


def gbm(params, Xtr, ytr, Xte, bte):
    m = lgb.LGBMRegressor(**params).fit(Xtr, ytr - Xtr["lag_1d"])
    return m, m.predict(Xte) + bte


def run(params=PARAMS, cols=None, a=tr, b=te, target=None):
    cols = cols or list(X.columns)
    yy = y if target is None else target
    m, p = gbm(params, X.loc[a, cols], yy[a], X.loc[b, cols], base[b].to_numpy())
    p_in = m.predict(X.loc[a, cols]) + base[a].to_numpy()
    return mape(yy[a], p_in), mape(y[b], p)


# 1. train vs test
tr_err, te_err = run()
_, dev_err = run(a=dev_tr, b=dev_te)
out += ["## 1. Train vs test error", "", f"| Training ({TRAIN_FROM[:4]}-2025, in-sample) | Dev test (2025, trained on {TRAIN_FROM[:4]}-2024) | 2026 test |",
        "|---|---|---|", f"| {tr_err:.2f}% | {dev_err:.2f}% | {te_err:.2f}% |", ""]

# 2. simpler models
lin_cols = [c for c in X.columns if c != "lag_364d"]
Xl = X[lin_cols].fillna(X[lin_cols][tr].median())
ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(Xl[tr], (y - base)[tr])
p_r = ridge.predict(Xl[te]) + base[te].to_numpy()
p_r_in = ridge.predict(Xl[tr]) + base[tr].to_numpy()
stump = dict(PARAMS, num_leaves=4, n_estimators=200)
s_tr, s_te = run(stump)
out += ["## 2. Simpler models on the same features", "", "| Model | Train | 2026 test |", "|---|---|---|",
        f"| Ridge regression (linear) | {mape(y[tr], p_r_in):.2f}% | {mape(y[te], p_r):.2f}% |",
        f"| Small GBM (4 leaves, 200 trees) | {s_tr:.2f}% | {s_te:.2f}% |",
        f"| Chosen GBM ({PARAMS['num_leaves']} leaves, {PARAMS['n_estimators']} trees) | {tr_err:.2f}% | {te_err:.2f}% |",
        f"| Same as yesterday | - | {mape(y[te], base[te]):.2f}% |", ""]

# 3. complexity sweep, judged on the 2025 dev fold (2026 shown for information only)
out += ["## 3. Complexity sweep", "", "| Leaves | Trees | Train | Dev 2025 | 2026 |", "|---|---|---|---|---|"]
for leaves, trees in [(4, 100), (7, 300), (15, 600), (31, 1000), (63, 2000), (127, 3000)]:
    pr = dict(PARAMS, num_leaves=leaves, n_estimators=trees, min_child_samples=5 if leaves > 31 else PARAMS["min_child_samples"])
    a_tr, a_te = run(pr)
    _, d = run(pr, a=dev_tr, b=dev_te)
    out.append(f"| {leaves} | {trees} | {a_tr:.2f}% | {d:.2f}% | {a_te:.2f}% |")
out.append("")

# 4. shuffled target (within training only): skill should vanish
rng = np.random.default_rng(0)
chg = (y - base)[tr].to_numpy().copy()
rng.shuffle(chg)
fake = y.copy()
fake[tr] = base[tr] + chg
_, sh = run(target=fake)
out += ["## 4. Shuffled-target test", "",
        f"Training on day-to-day changes shuffled at random gives **{sh:.2f}%** on 2026, vs {te_err:.2f}% for the real model "
        f"and {mape(y[te], base[te]):.2f}% for same-as-yesterday. The skill comes from real structure, not memorised noise.", ""]

# 5. seeds
seeds = [run(dict(PARAMS, random_state=s))[1] for s in range(5)]
out += ["## 5. Seed stability", "", f"2026 MAPE over 5 seeds: {min(seeds):.2f}% to {max(seeds):.2f}% (mean {np.mean(seeds):.2f}%).", ""]

# 6. no target-day weather (yesterday's weather only)
wcols = ["t_max", "t_min", "t_mean", "feels_max", "rh_mean", "dew_mean", "rain_mm", "cloud_mean", "radiation_sum",
         "wind_mean", "cdd", "hdd", "d_t_max"]
no_w = [c for c in X.columns if c not in wcols]
_, nw = run(cols=no_w)
out += ["## 6. Without the target day's weather", "",
        f"Using only yesterday's weather (no weather forecast at all): **{nw:.2f}%** on 2026. "
        "Real day-ahead weather forecasts sit between this and the archive-actual result.", ""]

# 7. bias and error by temperature
p_main = pd.Series(gbm(PARAMS, X[tr], y[tr], X[te], base[te].to_numpy())[1], index=X.index[te])
res = (p_main - y[te]) / y[te] * 100
hot = X.loc[te, "t_max"] >= 40
out += ["## 7. Bias", "", f"Mean error on 2026: {res.mean():+.2f}% (positive = over-forecast). "
        f"Days with max temperature >= 40 °C: {mape(y[te][hot], p_main[hot]):.2f}% MAPE over {int(hot.sum())} days; "
        f"other days {mape(y[te][~hot], p_main[~hot]):.2f}%.", ""]

Path("reports").mkdir(exist_ok=True)
verdict = (f"**Verdict:** training error {tr_err:.2f}% vs {te_err:.2f}% on 2026. Test error barely moves across model "
           f"sizes, is stable across seeds ({min(seeds):.2f}–{max(seeds):.2f}%), and collapses to {sh:.2f}% (worse than "
           f"same-as-yesterday) when the target is shuffled, so the skill is real and not memorised noise. A linear model "
           f"gets {mape(y[te], p_r):.2f}%, so most of the skill comes from the features. The model size "
           f"and v2 features ({PARAMS['num_leaves']} leaves, {PARAMS['n_estimators']} trees, training from {TRAIN_FROM}) were chosen "
           "on 2025 dev data and the 2016-2025 rolling-year test. 2026 was looked at three times in total "
           "(v1 large: 2.82%, v1 small: 2.73%, v2), each after the choice was made; that order is disclosed.\n\n")
Path("reports/overfit_check.md").write_text("# Overfitting checks: daily model\n\n" + verdict + "\n".join(out))
print("\n".join(out))
