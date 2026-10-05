"""Load the committed legacy dataset as a clean hourly frame for the leak-free pipeline.

Only raw inputs are kept: the six loads, raw weather, and calendar events known in advance.
None of the 110 engineered features from the old pipeline are used.
"""
from __future__ import annotations

import pandas as pd

from . import TARGETS
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
    # The 216 rows labelled "0" (Jul 2025) are generated (constant discom shares, white-noise
    # residuals); keep the hours but blank their loads. See reports/data_audit.md.
    df.loc[df["data_source"] == "unlabelled", TARGETS] = float("nan")
    # ~240 hours (nearly all at 23:00) are placeholder dips such as a repeated 1,489.24 MW between ~5,000 MW
    # neighbours; blank any hour below 85% of both neighbours.
    for c in TARGETS:
        v = df[c]
        df.loc[v < 0.85 * pd.concat([v.shift(1), v.shift(-1)], axis=1).min(axis=1), c] = float("nan")
    # Radiation was exported on a UTC clock while everything else is IST (+5:30): the value
    # for IST hour t is the average of the UTC rows 5 and 6 hours earlier.
    rad = df["shortwave_radiation"]
    df["shortwave_radiation"] = (rad.shift(5) + rad.shift(6)) / 2
    return df
