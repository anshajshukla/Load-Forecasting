"""Load the committed legacy dataset as a clean hourly frame for the leak-free pipeline.

Only raw inputs are kept: the six loads, raw weather, and calendar events known in advance.
None of the 110 engineered features from the old pipeline are used.
"""
from __future__ import annotations

import pandas as pd

from .features import EVENT_COLS

LEGACY_CSV = "load_forecast_new/delhi_interaction_enhanced_cleaned.csv"
RENAME = {
    "delhi_load": "delhi", "brpl_load": "brpl", "bypl_load": "bypl", "ndpl_load": "ndpl",
    "ndmc_load": "ndmc", "mes_load": "mes",
    "temperature_2m (°C)": "temperature_2m", "apparent_temperature (°C)": "apparent_temperature",
    "relative_humidity_2m (%)": "relative_humidity_2m", "dew_point_2m (°C)": "dew_point_2m",
    "cloud_cover (%)": "cloud_cover", "wind_speed_10m (km/h)": "wind_speed_10m",
    "precipitation (mm)": "precipitation", "shortwave_radiation": "shortwave_radiation",
}


def load_legacy(path: str = LEGACY_CSV) -> pd.DataFrame:
    cols = ["datetime", "data_source", *RENAME, *EVENT_COLS]
    df = pd.read_csv(path, usecols=lambda c: c in cols, parse_dates=["datetime"])
    df = df.rename(columns=RENAME).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")].asfreq("h")
    df["data_source"] = df["data_source"].replace({"0": "unlabelled"}).fillna("unlabelled")
    return df
