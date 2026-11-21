"""Ablation of the LightGBM model on 2017-2025 (development years only).

    python -m research.ablation     # writes reports/paper/ablation.md and ablation.json

Each variant removes one feature group (or one design choice) and is re-run through the same
protocol: yearly refit on the 5 years before, day-ahead forecasts, the same bias correction.
The Diebold-Mariano column tests each variant against the full model on daily absolute % error.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from pipeline.daily import PARAMS, YEARS_BACK, _xy, mape, postprocess

from .common import OUT, TEST_YEARS, load_dev, year_split
from .evaluate import dm_test

WEATHER = ["t_max", "t_min", "t_mean", "feels_max", "rh_mean", "dew_mean", "rain_mm", "cloud_mean", "radiation_sum",
           "wind_mean", "cdd", "hdd", "d_t_max", "t_max_yday", "t_mean_3d", "t_mean_7d", "t_max_3d", "cdd_3d"]
GROUPS = {
    "Full model": [],
    "No weather at all": WEATHER,
    "No humidity, rain, cloud, radiation, wind (temperature only)": ["rh_mean", "dew_mean", "rain_mm", "cloud_mean",
                                                                     "radiation_sum", "wind_mean", "feels_max"],
    "No heat build-up (3/7-day weather)": ["t_mean_3d", "t_mean_7d", "t_max_3d", "cdd_3d"],
    "No growth vs last year": ["yoy_growth"],
    "No weekday-ratio features": ["dow_ratio_prev", "lag7_ratio"],
    "No holiday features": ["is_holiday", "days_to_holiday", "days_since_holiday"],
    "No calendar at all (weekday, holiday, season)": ["dow", "is_holiday", "days_to_holiday", "days_since_holiday",
                                                      "doy_sin", "doy_cos", "dow_ratio_prev", "lag7_ratio"],
    "No long lags (14 and 364 days)": ["lag_14d", "lag_364d"],
}


def run(df, drop=(), target="change", years_back=YEARS_BACK) -> pd.Series:
    X, y = _xy(df, f"{TEST_YEARS[0] - 2 - years_back}-01-01")
    cols = [c for c in X.columns if c not in drop]
    out = []
    for year in [TEST_YEARS[0] - 2, TEST_YEARS[0] - 1] + TEST_YEARS:
        tr, te = year_split(X.index, year, years_back)
        base = X["lag_1d"] if target == "change" else 0.0
        t = y - base
        m = lgb.LGBMRegressor(**PARAMS).fit(X.loc[tr, cols], t[tr])
        b = X.loc[te, "lag_1d"].to_numpy() if target == "change" else 0.0
        out.append(pd.Series(m.predict(X.loc[te, cols]) + b, index=X.index[te]))
    return pd.concat(out)


def main():
    df = load_dev()
    a = df["energy_mu"]
    variants = {name: dict(drop=cols) for name, cols in GROUPS.items()}
    variants["Predict the level, not the change from yesterday"] = dict(target="level")
    rows, ape = [], {}
    for name, kw in variants.items():
        raw = run(df, **kw)
        post = postprocess(a, raw)["pred"]
        keep = (raw.index >= f"{TEST_YEARS[0]}-01-01") & a.reindex(raw.index).notna().to_numpy()
        idx = raw.index[keep]
        ape[name] = (np.abs(post[idx] - a[idx]) / a[idx] * 100)
        rows.append({"variant": name, "mape": mape(a[idx], post[idx]), "raw_mape": mape(a[idx], raw[idx])})
        print(rows[-1])
    # No bias correction: the full model's raw forecast.
    full = rows[0]
    rows.append({"variant": "No bias correction", "mape": full["raw_mape"], "raw_mape": full["raw_mape"]})
    raw_full = run(df)
    idx = ape["Full model"].index
    ape["No bias correction"] = np.abs(raw_full[idx] - a[idx]) / a[idx] * 100
    for r in rows:
        if r["variant"] != "Full model":
            i = ape["Full model"].index.intersection(ape[r["variant"]].index)
            r["dm"], r["p"] = dm_test(ape["Full model"][i].to_numpy(), ape[r["variant"]][i].to_numpy())
        r["delta"] = r["mape"] - full["mape"]
    L = ["# Ablation study, 2017–2025 (development years only)", "",
         "Each row removes one part of the model and re-runs the same yearly protocol (train on the 5 years before each "
         "test year, day-ahead, bias correction applied). Written by `python -m research.ablation`; 2026 is not used.", "",
         "| Variant | MAPE | Change vs full | Before bias correction | DM vs full (p) |", "|---|---|---|---|---|"]
    for r in rows:
        dm = f"{r['dm']:+.2f} ({r['p']:.3g})" if "dm" in r else "reference"
        L.append(f"| {r['variant']} | {r['mape']:.2f}% | {r['delta']:+.2f} pp | {r['raw_mape']:.2f}% | {dm} |")
    L += ["", "Negative DM: the full model is more accurate than the variant."]
    (OUT / "ablation.md").write_text("\n".join(L) + "\n")
    (OUT / "ablation.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
