"""Shared protocol for every model in the paper comparison.

* Data: real Grid-India daily energy + Open-Meteo archive weather, cut at DEV_END (2026 never loaded).
* Test years 2017-2025. For each test year Y a model is trained on the 5 years before Y only, then
  forecasts every day of Y one day ahead: energy known up to d-1, weather for day d.
* Every model writes one row per (date, model) with the point forecast and 9 quantiles (10%..90%),
  or point only, to reports/paper/pred_<model>.csv. research/evaluate.py scores them all the same way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.daily import YEARS_BACK, load

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports/paper"
DEV_END = "2026-01-01"  # exclusive; the 2026 holdout is not used in this folder
TEST_YEARS = list(range(2017, 2026))
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QCOLS = [f"q{int(q * 100)}" for q in QUANTILES]


def load_dev() -> pd.DataFrame:
    """Daily frame (energy + daily weather) with every day from DEV_END on removed."""
    df = load()
    df = df[df.index < DEV_END].copy()
    assert df.index.max() < pd.Timestamp(DEV_END)
    return df


def year_split(index: pd.DatetimeIndex, year: int, years_back: int = YEARS_BACK):
    tr = (index >= f"{year - years_back}-01-01") & (index < f"{year}-01-01")
    te = (index >= f"{year}-01-01") & (index < f"{year + 1}-01-01")
    return tr, te


def empirical_quantiles(actual: pd.Series, point: pd.Series, window: int = 365) -> pd.DataFrame:
    """Quantiles from the last `window` days' relative errors, each known by the day before."""
    err = (actual.reindex(point.index) / point - 1).shift(1).rolling(window, min_periods=window // 2)
    return pd.DataFrame({c: point * (1 + err.quantile(q)) for c, q in zip(QCOLS, QUANTILES)})


def save(model: str, point: pd.Series, quantiles: pd.DataFrame | None = None) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"date": point.index, "model": model, "pred": point.to_numpy()})
    if quantiles is not None:
        q = quantiles.reindex(point.index)
        for c in QCOLS:
            out[c] = q[c].to_numpy()
    assert pd.to_datetime(out["date"]).max() < pd.Timestamp(DEV_END), "2026 must not be scored here"
    path = OUT / f"pred_{model}.csv"
    out.to_csv(path, index=False, float_format="%.4f")
    print(f"{model}: {len(out)} days -> {path.relative_to(REPO)}")
    return path


def gap_free(y: pd.Series) -> pd.Series:
    """Linear interpolation of source gaps, ONLY as input for models that cannot take NaN (N-HiTS,
    Chronos). Interpolated days are never scored: evaluate.py scores only days with a reported value."""
    return y.interpolate(limit_area="inside")


def calendar_exog(index: pd.DatetimeIndex) -> pd.DataFrame:
    from pipeline.daily import features
    return features(pd.DataFrame({"energy_mu": np.nan, "t_max": 0.0, "t_min": 0.0, "t_mean": 0.0, "feels_max": 0.0,
                                  "rh_mean": 0.0, "dew_mean": 0.0, "rain_mm": 0.0, "cloud_mean": 0.0,
                                  "radiation_sum": 0.0, "wind_mean": 0.0}, index=index))[
        ["dow", "is_holiday", "days_to_holiday", "days_since_holiday", "doy_sin", "doy_cos"]]


WEATHER_EXOG = ["t_max", "t_min", "t_mean", "rh_mean", "rain_mm", "cloud_mean", "radiation_sum", "wind_mean"]


def exog(df: pd.DataFrame) -> pd.DataFrame:
    """Weather of day d (known as a forecast at the end of d-1) plus calendar; no energy values."""
    e = df[WEATHER_EXOG].copy()
    e["cdd"] = (df["t_mean"] - 24).clip(lower=0)
    e["hdd"] = (16 - df["t_mean"]).clip(lower=0)
    e["t_mean_3d"] = df["t_mean"].rolling(3).mean()
    return e.join(calendar_exog(df.index))
