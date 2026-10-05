"""Live day-ahead forecast: update data, forecast the next day, score earlier forecasts.

    python -m pipeline.forecast [--posoco-csv PATH/TO/POSOCO_data.csv]

1. Optionally refreshes data/posoco/delhi_daily.csv from Robbie Andrew's POSOCO_data.csv.
2. Appends recent weather and the weather *forecast* (Open-Meteo forecast API, IST) to data/weather.
3. Trains on the last 3 years of real data and forecasts the day after the last known day,
   with 80%/95% intervals from the latest 12-month walk-forward.
4. Appends to reports/live/forecasts.csv (a forecast is never overwritten) and fills in actuals
   for earlier forecasts once Grid-India reports them.

Run daily by .github/workflows/daily-forecast.yml. These forecasts are made before the day and are
never used for model selection, so they are the clean out-of-sample test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import LAT, LON, TZ
from .daily import DAILY_CSV, WEATHER_CSV, _fit_predict, _xy, dev_residuals, features, load
from .features import WEATHER_COLS

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
LIVE_CSV = Path("reports/live/forecasts.csv")
COLUMNS = ["target_date", "made_at_utc", "pred", "lo80", "hi80", "lo95", "hi95", "actual", "abs_pct_error"]


def refresh_daily(posoco_csv: Path, out: Path = DAILY_CSV) -> int:
    d = pd.read_csv(posoco_csv, usecols=["yyyymmdd", "Delhi: EnergyMet"])
    d["date"] = pd.to_datetime(d["yyyymmdd"].astype(str)).dt.date
    d = d.rename(columns={"Delhi: EnergyMet": "energy_mu"})[["date", "energy_mu"]].dropna()
    d.to_csv(out, index=False)
    return len(d)


def fetch_forecast_weather(past_days: int = 10, forecast_days: int = 3) -> pd.DataFrame:
    import requests

    r = requests.get(FORECAST_URL, timeout=60, params={
        "latitude": LAT, "longitude": LON, "hourly": ",".join(WEATHER_COLS), "timezone": TZ,
        "past_days": past_days, "forecast_days": forecast_days})
    r.raise_for_status()
    w = pd.DataFrame(r.json()["hourly"]).rename(columns={"time": "datetime"})
    w["datetime"] = pd.to_datetime(w["datetime"])
    return w.set_index("datetime")


def merge_weather(new: pd.DataFrame, path: Path = WEATHER_CSV, prefer_new: bool = False) -> None:
    """prefer_new=True for archive data (measured values replace earlier forecasts);
    False for forecasts (they only fill hours that have no value yet)."""
    old = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")
    new = new[old.columns.intersection(new.columns)]
    merged = new.combine_first(old) if prefer_new else old.combine_first(new)
    merged.reset_index().to_csv(path, index=False)


def forecast_next(df: pd.DataFrame) -> dict:
    last = df["energy_mu"].last_valid_index()
    target = last + pd.Timedelta(days=1)
    if target not in df.index or pd.isna(df.loc[target, "t_max"]):
        raise ValueError(f"no weather for {target.date()}; fetch the forecast first")
    start = (last - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    ext = df.loc[:target]
    X, y = _xy(ext, start)
    Xt = features(ext).loc[[target]]
    Xall = pd.concat([X, Xt])
    tr = Xall.index <= last
    te = Xall.index == target
    yall = y.reindex(Xall.index)
    p, _ = _fit_predict(Xall, yall, tr, te)
    pred = float(p.iloc[0])
    res = dev_residuals(df.loc[:last], start, (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    out = {"target_date": target.date().isoformat(), "made_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "pred": round(pred, 2)}
    for lvl in (80, 95):
        lo, hi = res.quantile([(1 - lvl / 100) / 2, 1 - (1 - lvl / 100) / 2])
        out[f"lo{lvl}"], out[f"hi{lvl}"] = round(pred * (1 + lo), 2), round(pred * (1 + hi), 2)
    return out


def update_log(new: dict | None, df: pd.DataFrame, path: Path = LIVE_CSV) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
    if new and new["target_date"] not in set(log["target_date"].astype(str)):
        log = pd.concat([log, pd.DataFrame([new])], ignore_index=True)
    actual = df["energy_mu"]
    for i, row in log.iterrows():
        d = pd.Timestamp(row["target_date"])
        if pd.isna(row.get("actual")) and d in actual.index and pd.notna(actual[d]):
            log.loc[i, "actual"] = actual[d]
            log.loc[i, "abs_pct_error"] = round(abs(row["pred"] - actual[d]) / actual[d] * 100, 2)
    log = log[COLUMNS].sort_values("target_date")
    log.to_csv(path, index=False)
    return log


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="pipeline.forecast")
    p.add_argument("--posoco-csv", type=Path, help="POSOCO_data.csv from robbieandrew.github.io/india/data/POSOCO_data.zip")
    p.add_argument("--skip-weather", action="store_true", help="use data/weather as is (offline runs)")
    a = p.parse_args(argv)
    if a.posoco_csv:
        print("daily rows:", refresh_daily(a.posoco_csv))
    if not a.skip_weather:
        merge_weather(fetch_forecast_weather())
    df = load()
    new = forecast_next(df)
    print("forecast:", new)
    log = update_log(new, df)
    scored = log.dropna(subset=["actual"])
    if len(scored):
        print(f"live MAPE over {len(scored)} days: {scored['abs_pct_error'].astype(float).mean():.2f}%")


if __name__ == "__main__":
    main()
