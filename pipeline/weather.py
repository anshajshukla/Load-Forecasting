"""Hourly Delhi weather from Open-Meteo (free, no API key).

Archive API for history, forecast API for the next days. Times are local (Asia/Kolkata).
"""
from __future__ import annotations

import pandas as pd
import requests

from . import LAT, LON, TZ
from .features import WEATHER_COLS

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"


def _get(url: str, params: dict) -> pd.DataFrame:
    p = {"latitude": LAT, "longitude": LON, "timezone": TZ, "hourly": ",".join(WEATHER_COLS), **params}
    r = requests.get(url, params=p, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def archive(start: str, end: str) -> pd.DataFrame:
    return _get(ARCHIVE, {"start_date": start, "end_date": end})


def forecast(days: int = 3, past_days: int = 7) -> pd.DataFrame:
    return _get(FORECAST, {"forecast_days": days, "past_days": past_days})
