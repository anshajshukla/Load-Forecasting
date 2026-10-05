"""Feature construction that can only see the past.

Every load-derived feature for a forecast made at origin time t for target time t+h
uses load values at or before t. In row terms (one row per target hour), that means
load shifted by at least `horizon` hours. Weather columns are for the target hour;
in production they come from a weather *forecast*, in backtests from the archive
(stated as an optimistic assumption in the report).
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "apparent_temperature", "dew_point_2m",
    "cloud_cover", "shortwave_radiation", "wind_speed_10m", "precipitation",
]
# Calendar events known in advance (festivals, school vacations). Passed through when present.
EVENT_COLS = [
    "is_diwali_period", "is_major_festival", "is_national_holiday", "is_religious_festival",
    "pre_festival_day", "post_festival_day", "is_summer_vacation", "is_winter_vacation",
]


def _calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    years = range(idx.year.min() - 1, idx.year.max() + 2)
    hol = holidays.India(years=years, subdiv="DL")
    f = pd.DataFrame(index=idx)
    f["hour"] = idx.hour
    f["dow"] = idx.dayofweek
    f["month"] = idx.month
    doy = idx.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    f["is_holiday"] = pd.Index(idx.date).isin(list(hol.keys())).astype(int)
    return f


def build_features(df: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """Return features for predicting df[target] `horizon` hours after the origin.

    df: hourly, DatetimeIndex without gaps (missing hours as NaN), columns = targets + weather.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    y = df[target]
    f = _calendar(df.index)

    # Same-hour lags. For lag L to be known at origin t = target - horizon, need L >= horizon.
    lags = {horizon, horizon + 1, horizon + 2, 24, 48, 168, 336}
    for lag in sorted(l for l in lags if l >= horizon):
        f[f"lag_{lag}h"] = y.shift(lag)
    # Same hour on the most recent fully known day(s).
    day_lag = 24 * int(np.ceil(horizon / 24))
    f["same_hour_last_known_day"] = y.shift(day_lag)
    f["same_hour_last_week"] = y.shift(168 if horizon <= 168 else 168 * int(np.ceil(horizon / 168)))

    known = y.shift(horizon)  # last value known at origin
    for w in (24, 168):
        f[f"roll_mean_{w}h"] = known.rolling(w, min_periods=w // 2).mean()
        f[f"roll_max_{w}h"] = known.rolling(w, min_periods=w // 2).max()
        f[f"roll_min_{w}h"] = known.rolling(w, min_periods=w // 2).min()
    f["roll_std_24h"] = known.rolling(24, min_periods=12).std()
    f["ramp_known_1h"] = known - y.shift(horizon + 1)

    for c in WEATHER_COLS:
        if c in df:
            f[c] = df[c]
    if "temperature_2m" in df:
        t = df["temperature_2m"]
        f["cdh"] = (t - 24).clip(lower=0)  # cooling degree-hours
        f["hdh"] = (18 - t).clip(lower=0)
        f["temp_x_hour"] = t * f["hour_sin"]
        f["temp_mean_prev24"] = t.shift(1).rolling(24, min_periods=12).mean()
        f["temp_mean_72h"] = t.rolling(72, min_periods=24).mean()  # heat build-up over days
        day = df.index.normalize()
        # Whole-day temperature stats for the target day: available from a day-ahead weather forecast.
        f["tmax_day"] = t.groupby(day).transform("max")
        f["tmin_day"] = t.groupby(day).transform("min")
        f["tmax_change_vs_yesterday"] = f["tmax_day"] - f["tmax_day"].shift(24)
        f["temp_change_vs_yesterday"] = t - t.shift(24)
    if "apparent_temperature" in df:
        f["apparent_cdh"] = (df["apparent_temperature"] - 26).clip(lower=0)
    for c in EVENT_COLS:
        if c in df:
            f[c] = df[c]
    return f


def assert_no_future_leak(df: pd.DataFrame, target: str, horizon: int, at: int | None = None) -> None:
    """Perturb target values after the origin and check features at the target row don't change."""
    at = at if at is not None else len(df) - 1
    base = build_features(df, target, horizon).iloc[at]
    pert = df.copy()
    origin = at - horizon
    pert.iloc[origin + 1:, pert.columns.get_loc(target)] *= 1.5
    after = build_features(pert, target, horizon).iloc[at]
    diff = (base - after).abs()
    bad = diff[diff > 1e-9]
    if len(bad):
        raise AssertionError(f"features use load after the origin: {list(bad.index)}")
