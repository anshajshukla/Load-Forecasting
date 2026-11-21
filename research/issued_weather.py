"""Ex-ante weather test: forecast with weather *as forecast a day earlier* instead of recorded weather.

    python -m research.issued_weather fetch   # Open-Meteo previous-runs API -> data/weather/delhi_issued_day1.csv
    python -m research.issued_weather eval    # writes reports/paper/issued_weather.md and issued_weather_predictions.csv

The previous-runs API stores, for every hour, the value of the forecast run made 24 hours earlier
(`<variable>_previous_day1`), so every hour of day d was forecast during day d-1. The model is
trained exactly as before (recorded weather); only the target day's weather is replaced with the
issued forecast, while days up to d-1 keep recorded weather (they are in the past at forecast time).
Test years: those the API covers within 2024-2025. 2026 is not used.
"""
from __future__ import annotations

import sys
from datetime import date

import numpy as np
import pandas as pd

from pipeline import LAT, LON, TZ
from pipeline.daily import DAILY_CSV, _xy, features, load, postprocess
from pipeline.features import WEATHER_COLS
from pipeline.http import get

from .common import OUT, REPO, load_dev, year_split
from .evaluate import dm_test

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ISSUED_CSV = REPO / "data/weather/delhi_issued_day1.csv"
DAILY_W = ["t_max", "t_min", "t_mean", "feels_max", "rh_mean", "dew_mean", "rain_mm", "cloud_mean", "radiation_sum",
           "wind_mean"]


def fetch(start="2023-01-01", end=None) -> pd.DataFrame:
    end = end or date.today().isoformat()
    frames = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        s, e = max(start, f"{y}-01-01"), min(end, f"{y}-12-31")
        r = get(URL, timeout=180, params={"latitude": LAT, "longitude": LON, "timezone": TZ, "start_date": s,
                                          "end_date": e, "hourly": ",".join(f"{c}_previous_day1" for c in WEATHER_COLS)})
        h = pd.DataFrame(r.json()["hourly"]).rename(columns={"time": "datetime"})
        h.columns = [c.replace("_previous_day1", "") for c in h.columns]
        frames.append(h)
        print(f"{y}: {len(h)} hours, non-null temperature {h['temperature_2m'].notna().sum()}")
    w = pd.concat(frames)
    w = w.dropna(subset=["temperature_2m"])
    ISSUED_CSV.parent.mkdir(parents=True, exist_ok=True)
    w.to_csv(ISSUED_CSV, index=False)
    print(f"{len(w)} issued-forecast hours ({w['datetime'].min()} to {w['datetime'].max()}) -> {ISSUED_CSV}")
    return w


def evaluate():
    df = load_dev()
    iw = load(DAILY_CSV, ISSUED_CSV)[DAILY_W]
    hours = pd.read_csv(ISSUED_CSV, parse_dates=["datetime"]).set_index("datetime")
    full_days = hours["temperature_2m"].resample("D").count() == 24
    years = [y for y in (2024, 2025) if full_days[str(y)].sum() >= 300]
    if not years:
        raise SystemExit("issued-forecast weather covers no full test year in 2024-2025")
    X, y = _xy(df, f"{years[0] - 7}-01-01")
    import lightgbm as lgb
    from pipeline.daily import PARAMS
    rec, iss, days_all = [], [], []
    for year in [years[0] - 2, years[0] - 1] + years:
        tr, te = year_split(X.index, year)
        m = lgb.LGBMRegressor(**PARAMS).fit(X[tr], (y - X["lag_1d"])[tr])
        test_days = X.index[te]
        rec.append(pd.Series(m.predict(X.loc[te]) + X.loc[te, "lag_1d"].to_numpy(), index=test_days))
        if year not in years:
            iss.append(rec[-1])  # warm-up years for the bias correction: recorded weather (past, not scored)
            continue
        rows = []
        for d in test_days:
            if not (d in full_days.index and full_days[d]):
                continue
            w = df.loc[d - pd.Timedelta(days=800):d].copy()
            w.loc[d, DAILY_W] = iw.loc[d, DAILY_W].to_numpy()  # only the target day uses the issued forecast
            rows.append(features(w).loc[[d]])
        Xi = pd.concat(rows)[X.columns]
        iss.append(pd.Series(m.predict(Xi) + Xi["lag_1d"].to_numpy(), index=Xi.index))
        days_all += list(Xi.index)
        print(f"{year}: {len(Xi)} days with issued weather")
    a = df["energy_mu"]
    raw_rec, raw_iss = pd.concat(rec), pd.concat(iss)
    p_rec, p_iss = postprocess(a, raw_rec)["pred"], postprocess(a, raw_iss)["pred"]
    idx = pd.DatetimeIndex(days_all).intersection(a.dropna().index)
    from research.ablation import WEATHER, run
    nw = postprocess(a, run(df, drop=WEATHER))["pred"]
    ape = {k: np.abs(p[idx] - a[idx]) / a[idx] * 100 for k, p in
           {"recorded": p_rec, "issued": p_iss, "none": nw, "yday": X["lag_1d"]}.items()}
    t_err = (iw.loc[idx, "t_max"] - df.loc[idx, "t_max"])
    dm_i = dm_test(ape["issued"].to_numpy(), ape["recorded"].to_numpy())
    dm_n = dm_test(ape["issued"].to_numpy(), ape["none"].to_numpy())
    L = ["# Ex-ante weather test (issued day-ahead forecasts)", "",
         f"{len(idx)} days in {', '.join(map(str, years))}. Same LightGBM, trained on recorded weather; the target day's "
         "weather is the Open-Meteo forecast issued 24 hours earlier (previous-runs API, `_previous_day1`). "
         "Written by `python -m research.issued_weather eval`; 2026 not used.", "",
         "| Weather for the target day | MAPE | DM vs issued (p) |", "|---|---|---|",
         f"| Recorded (archive), as in every other table | {ape['recorded'].mean():.2f}% | {-dm_i[0]:+.2f} ({dm_i[1]:.3g}) |",
         f"| **Issued a day earlier (truly ex-ante)** | **{ape['issued'].mean():.2f}%** | reference |",
         f"| No weather features at all | {ape['none'].mean():.2f}% | {-dm_n[0]:+.2f} ({dm_n[1]:.3g}) |",
         f"| Same as yesterday | {ape['yday'].mean():.2f}% | |", "",
         f"Issued vs recorded daily maximum temperature: mean error {t_err.mean():+.2f} °C, mean absolute error "
         f"{t_err.abs().mean():.2f} °C.", "",
         "DM statistic: the row's loss minus the issued-weather loss; negative means the row is more accurate."]
    pd.DataFrame({"actual": a[idx], "recorded": p_rec[idx], "issued": p_iss[idx], "no_weather": nw[idx]}).rename_axis(
        "date").to_csv(OUT / "issued_weather_predictions.csv", float_format="%.4f")
    (OUT / "issued_weather.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"fetch": fetch, "eval": evaluate}[sys.argv[1]]()
