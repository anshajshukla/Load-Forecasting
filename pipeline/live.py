"""Live data + forecast for the dashboard: fetch the latest real data, forecast the coming days.

* Energy: Grid-India daily reports via Robbie Andrew's processed dataset (POSOCO_data.zip on GitHub).
* Weather: the repo's archive, extended with Open-Meteo's forecast API (recent days + up to 7 days ahead).
* Forecast: the daily model (v2, trained on the last 5 years up to the last known day), stepped forward one
  day at a time. Day 1 is a true day-ahead forecast; later days feed earlier forecasts back in as
  "yesterday", so they are less certain and their ranges are widened (by sqrt(day)).
* Ensemble: when Chronos-2 is available (pipeline/chronos.py), the forecast is the equal-weight average of the
  bias-corrected LightGBM and Chronos-2 forecasts (2.49% vs 2.63% on 2016-2025, reports/model_comparison.md).
  Each model gets its own bias correction from its own errors over the last 28 days. The ranges are LightGBM's,
  centred on the average (the average's own errors aren't logged yet, so this is conservative).

Nothing is invented: if a live source can't be reached, the repo's committed copy is used and the
result says so.
"""
from __future__ import annotations

import io
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import LAT, LON, TZ
from .daily import (CORR_ALPHA, CORR_CAP, CORR_WINDOW, DAILY_CSV, INTERVAL_WINDOW, WEATHER_CSV, YEARS_BACK,
                    _xy, features, load, oos_predictions, PARAMS)
from .features import WEATHER_COLS

POSOCO_ZIP = "https://raw.githubusercontent.com/robbieandrew/robbieandrew.github.io/master/india/data/POSOCO_data.zip"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _get(url: str, tries: int = 3, **kw):
    """GET with a few retries, so one network hiccup doesn't drop a live source."""
    import time

    import requests

    for i in range(tries):
        try:
            r = requests.get(url, timeout=60, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def fetch_energy() -> tuple[pd.Series, str]:
    """Latest Delhi daily energy met (MU). Falls back to the committed CSV if the download fails."""
    try:
        r = _get(POSOCO_ZIP)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            d = pd.read_csv(z.open("POSOCO_data.csv"), usecols=["yyyymmdd", "Delhi: EnergyMet"])
        d["date"] = pd.to_datetime(d["yyyymmdd"].astype(str))
        s = d.set_index("date")["Delhi: EnergyMet"].dropna().rename("energy_mu")
        return s, f"live: Grid-India via {POSOCO_ZIP.split('/')[2]}"
    except Exception as e:  # noqa: BLE001 - any failure falls back to the committed copy, and says so
        s = pd.read_csv(DAILY_CSV, parse_dates=["date"]).set_index("date")["energy_mu"]
        return s, f"committed copy (data/posoco/delhi_daily.csv); live download failed: {type(e).__name__}"


def fetch_weather(past_days: int = 92, forecast_days: int = 8) -> tuple[pd.DataFrame, str]:
    """Hourly weather: committed archive, plus Open-Meteo forecast for recent and coming days."""
    base = pd.read_csv(WEATHER_CSV, parse_dates=["datetime"]).set_index("datetime")
    try:
        r = _get(FORECAST_URL, params={
            "latitude": LAT, "longitude": LON, "hourly": ",".join(WEATHER_COLS), "timezone": TZ,
            "past_days": past_days, "forecast_days": forecast_days})
        f = pd.DataFrame(r.json()["hourly"]).rename(columns={"time": "datetime"})
        f["datetime"] = pd.to_datetime(f["datetime"])
        f = f.set_index("datetime")[base.columns.intersection(WEATHER_COLS)]
        return base.combine_first(f), "live: Open-Meteo archive + forecast"
    except Exception as e:  # noqa: BLE001
        return base, f"committed archive only; live forecast failed: {type(e).__name__}"


def build_frame(energy: pd.Series, weather: pd.DataFrame) -> pd.DataFrame:
    """Same daily frame as pipeline.daily.load(), built from in-memory sources."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        e_path, w_path = Path(tmp) / "e.csv", Path(tmp) / "w.csv"
        energy.rename_axis("date").reset_index().to_csv(e_path, index=False)
        weather.rename_axis("datetime").reset_index().to_csv(w_path, index=False)
        return load(e_path, w_path)


def forecast_days(df: pd.DataFrame, horizon: int = 7, ensemble: bool = False) -> pd.DataFrame:
    """Forecast from the day after the last known day, up to `horizon` days or as far as weather exists."""
    import lightgbm as lgb

    last = df["energy_mu"].last_valid_index()
    start = (last - pd.DateOffset(years=YEARS_BACK)).strftime("%Y-%m-%d")
    X, y = _xy(df.loc[:last], start)
    model = lgb.LGBMRegressor(**PARAMS).fit(X, y - X["lag_1d"])

    # Bias correction and interval width from out-of-sample errors up to the last known day.
    hist = oos_predictions(df.loc[:last], last.year - 1, last.year)
    a = df["energy_mu"].reindex(hist.index)
    err = (a / hist - 1).dropna()
    corr = float(np.clip(err.tail(CORR_WINDOW).mean() * CORR_ALPHA, -CORR_CAP, CORR_CAP)) if len(err) else 0.0
    res = (a / (hist * (1 + corr)) - 1).dropna().tail(INTERVAL_WINDOW)
    q = {lvl: res.quantile([(1 - lvl / 100) / 2, 1 - (1 - lvl / 100) / 2]).to_numpy() for lvl in (80, 95)}

    work = df.copy()
    rows = []
    for h in range(1, horizon + 1):
        d = last + pd.Timedelta(days=h)
        if d not in work.index or pd.isna(work.loc[d, "t_max"]):
            break  # no weather forecast this far out
        x = features(work.loc[:d]).loc[[d]]
        raw = float(model.predict(x[X.columns])[0] + x["lag_1d"].iloc[0])
        pred = raw * (1 + corr)
        row = {"date": d, "days_ahead": h, "pred": pred, "t_max": float(work.loc[d, "t_max"])}
        for lvl, (lo, hi) in q.items():
            row[f"lo{lvl}"], row[f"hi{lvl}"] = pred * (1 + lo * np.sqrt(h)), pred * (1 + hi * np.sqrt(h))
        rows.append(row)
        work.loc[d, "energy_mu"] = pred  # feeds the next day's "yesterday"
    out = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    out.attrs.update(last_known=last, correction_pct=corr * 100, model="LightGBM")
    if ensemble and len(out):
        out = _blend_chronos(df, out, last)
    return out


def _blend_chronos(df: pd.DataFrame, out: pd.DataFrame, last: pd.Timestamp) -> pd.DataFrame:
    """Average the LightGBM forecast with a bias-corrected Chronos-2 forecast; LightGBM only if unavailable."""
    from . import chronos

    ok, why = chronos.available()
    if not ok:
        out.attrs["model"] = f"LightGBM (Chronos-2 unavailable: {why})"
        return out
    try:
        known = df["energy_mu"].loc[:last].dropna()
        # Chronos-2's own bias correction: its one-step errors over the last CORR_WINDOW reported days.
        origins = [df.index[df.index.get_loc(t) - 1] for t in known.index[-CORR_WINDOW:]]
        past = chronos.forecast(df, origins, 1)[1]
        past.index = known.index[-CORR_WINDOW:]
        c_corr = float(np.clip((known.reindex(past.index) / past - 1).mean() * CORR_ALPHA, -CORR_CAP, CORR_CAP))
        ahead = chronos.forecast(df, [last], len(out)).iloc[0].to_numpy() * (1 + c_corr)
    except Exception as e:  # noqa: BLE001
        out.attrs["model"] = f"LightGBM (Chronos-2 failed: {type(e).__name__})"
        return out
    out = out.copy()
    out["pred_lgbm"], out["pred_chronos"] = out["pred"], ahead
    blend = (out["pred_lgbm"] + out["pred_chronos"]) / 2
    shift = blend / out["pred"]
    for c in ("lo80", "hi80", "lo95", "hi95"):
        out[c] = out[c] * shift
    out["pred"] = blend
    out.attrs.update(model="LightGBM + Chronos-2 average", chronos_correction_pct=c_corr * 100)
    return out


def run(horizon: int = 7) -> dict:
    energy, e_src = fetch_energy()
    weather, w_src = fetch_weather()
    df = build_frame(energy, weather)
    fc = forecast_days(df, horizon, ensemble=True)
    return {"df": df, "forecast": fc, "energy_source": e_src, "weather_source": w_src,
            "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
