"""Live day-ahead forecast: update data, forecast the next day, score earlier forecasts.

    python -m pipeline.forecast [--posoco-csv PATH/TO/POSOCO_data.csv]

1. Optionally refreshes data/posoco/delhi_daily.csv from Robbie Andrew's POSOCO_data.csv.
2. Appends recent weather and the weather *forecast* (Open-Meteo forecast API, IST) to data/weather.
3. Trains on the last 5 years of real data and forecasts *tomorrow* (IST), stepping forward from the
   last day Grid-India has reported (pipeline.live.forecast_days), corrected by half its mean error
   over the last 28 days, with 80%/95% intervals from the last 365 days' errors.
4. Appends to reports/live/forecasts.csv (a forecast is never overwritten) and fills in actuals
   for earlier forecasts once Grid-India reports them.

Each row records `days_ahead` (days from the last reported day to the target) and `ex_ante`
(the forecast was made before the target day began in IST, so it used a real weather forecast).
Only ex-ante rows are the clean out-of-sample test. Before 2026-11-22 the script forecast the day
after the last *reported* day, which Grid-India's delay put on or after the target day itself,
so those rows are kept but marked ex_ante=False.

If Open-Meteo can't be reached after retries, the run still scores earlier forecasts and uses
weather already stored (yesterday's run fetches 3 days ahead), and only fails if tomorrow has none.

Run daily by .github/workflows/daily-forecast.yml.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import pandas as pd

from . import LAT, LON, TZ
from .daily import DAILY_CSV, WEATHER_CSV, load
from .features import WEATHER_COLS

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
LIVE_CSV = REPO_ROOT / "reports/live/forecasts.csv"
COLUMNS = ["target_date", "made_at_utc", "days_ahead", "ex_ante", "pred", "lo80", "hi80", "lo95", "hi95",
           "actual", "abs_pct_error"]


def is_ex_ante(made_at_utc: str, target_date: str) -> bool:
    """True if the forecast was made before the target day started in Delhi (IST)."""
    try:
        made = pd.Timestamp(made_at_utc)
    except ValueError:
        return False  # no usable timestamp: can't claim the forecast came first
    made = made.tz_localize("UTC") if made.tzinfo is None else made
    return bool(made < pd.Timestamp(target_date).tz_localize(TZ))


def refresh_daily(posoco_csv: Path, out: Path = DAILY_CSV) -> int:
    d = pd.read_csv(posoco_csv, usecols=["yyyymmdd", "Delhi: EnergyMet"])
    d["date"] = pd.to_datetime(d["yyyymmdd"].astype(str)).dt.date
    d = d.rename(columns={"Delhi: EnergyMet": "energy_mu"})[["date", "energy_mu"]].dropna()
    d.to_csv(out, index=False)
    return len(d)


def fetch_forecast_weather(past_days: int = 10, forecast_days: int = 3) -> pd.DataFrame:
    from .http import get

    r = get(FORECAST_URL, params={
        "latitude": LAT, "longitude": LON, "hourly": ",".join(WEATHER_COLS), "timezone": TZ,
        "past_days": past_days, "forecast_days": forecast_days})
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


def forecast_tomorrow(df: pd.DataFrame, now_utc: datetime | None = None) -> dict:
    """Forecast tomorrow in IST, however many days Grid-India's report is behind."""
    from .live import forecast_days

    now_utc = now_utc or datetime.now(timezone.utc)
    target = pd.Timestamp(pd.Timestamp(now_utc).tz_convert(TZ).date()) + pd.Timedelta(days=1)
    last = df["energy_mu"].last_valid_index()
    horizon = (target - last).days
    fc = forecast_days(df, horizon=horizon)
    if target not in fc.index:
        raise ValueError(f"no weather for {target.date()} (last reported day {last.date()}); fetch the forecast first")
    row = fc.loc[target]
    out = {"target_date": target.date().isoformat(), "made_at_utc": now_utc.isoformat(timespec="seconds"),
           "days_ahead": int(row["days_ahead"]), "ex_ante": True, "pred": round(float(row["pred"]), 2)}
    for lvl in (80, 95):
        out[f"lo{lvl}"], out[f"hi{lvl}"] = round(float(row[f"lo{lvl}"]), 2), round(float(row[f"hi{lvl}"]), 2)
    return out


def update_log(new: dict | None, df: pd.DataFrame, path: Path = LIVE_CSV) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
    # Rows from before these columns existed were always 1 day after the last reported day.
    if "days_ahead" not in log:
        log["days_ahead"] = 1
    log["ex_ante"] = [is_ex_ante(m, t) for m, t in zip(log["made_at_utc"], log["target_date"].astype(str))]
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
        try:
            merge_weather(fetch_forecast_weather())
        except Exception as e:  # noqa: BLE001 - keep going on stored weather, and say so
            print(f"::warning::weather forecast download failed ({type(e).__name__}); using stored weather")
    df = load()
    try:
        new = forecast_tomorrow(df)
        print("forecast:", new)
    except ValueError as e:
        new = None
        print(f"::error::no forecast today: {e}")
    log = update_log(new, df)  # earlier forecasts are scored even when today's forecast failed
    scored = log.dropna(subset=["actual"])
    clean = scored[scored["ex_ante"].astype(bool)]
    if len(scored):
        print(f"live MAPE, all {len(scored)} scored days: {scored['abs_pct_error'].astype(float).mean():.2f}%")
    if len(clean):
        print(f"live MAPE, {len(clean)} ex-ante days: {clean['abs_pct_error'].astype(float).mean():.2f}%")
    if new is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
