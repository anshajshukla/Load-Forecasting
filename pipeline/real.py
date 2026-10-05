"""Build an all-real hourly dataset: SLDC loads (pipeline.sldc) + Open-Meteo archive weather.

No synthetic rows, no legacy CSV. Weather is downloaded on the IST clock, which avoids the
UTC/IST radiation mix-up found in the legacy data.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from . import LAT, LON, TARGETS, TZ
from .features import WEATHER_COLS
from .sldc import load_hourly

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_CSV = Path("data/weather/delhi_hourly.csv")


def fetch_weather(start: date, end: date, out: Path = WEATHER_CSV) -> pd.DataFrame:
    import requests

    r = requests.get(WEATHER_URL, timeout=120, params={
        "latitude": LAT, "longitude": LON, "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": ",".join(WEATHER_COLS), "timezone": TZ})
    r.raise_for_status()
    h = r.json()["hourly"]
    w = pd.DataFrame(h).rename(columns={"time": "datetime"})
    w["datetime"] = pd.to_datetime(w["datetime"])
    out.parent.mkdir(parents=True, exist_ok=True)
    w.to_csv(out, index=False)
    print(f"{len(w)} weather hours -> {out}")
    return w.set_index("datetime")


def build_real(sldc_csv: str | Path, weather_csv: str | Path = WEATHER_CSV) -> pd.DataFrame:
    """Hourly frame (gaps as NaN) with the six loads, weather, and data_source='sldc'."""
    loads = load_hourly(Path(sldc_csv)).reindex(columns=TARGETS)
    w = pd.read_csv(weather_csv, parse_dates=["datetime"]).set_index("datetime")
    idx = pd.date_range(loads.index.min(), loads.index.max(), freq="h")
    df = loads.reindex(idx).join(w.reindex(idx)[[c for c in WEATHER_COLS if c in w]])
    df.index.name = "datetime"
    df["data_source"] = "sldc"
    return df
