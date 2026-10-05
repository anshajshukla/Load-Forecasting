import numpy as np
import pandas as pd
import pytest

from pipeline.features import assert_no_future_leak, build_features


@pytest.fixture
def frame():
    idx = pd.date_range("2023-01-01", periods=24 * 60, freq="h")
    rng = np.random.default_rng(0)
    load = 4000 + 800 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 50, len(idx))
    temp = 25 + 8 * np.sin(2 * np.pi * (idx.hour - 4) / 24)
    return pd.DataFrame({"delhi": load, "temperature_2m": temp}, index=idx)


@pytest.mark.parametrize("h", [1, 3, 24, 48])
def test_features_never_see_load_after_origin(frame, h):
    for at in (len(frame) - 1, len(frame) - 200):
        assert_no_future_leak(frame, "delhi", h, at=at)


def test_target_itself_is_not_recoverable(frame):
    f = build_features(frame, "delhi", 1).dropna()
    f = f.loc[:, f.std() > 0]
    y = frame.loc[f.index, "delhi"]
    corr = f.corrwith(y - f["lag_1h"]).abs().max()
    assert corr < 0.9  # the shipped dataset had a 0.999 feature (net_load_ramp_rate)


def test_forecast_next_returns_24_hours(frame):
    from pipeline.train import forecast_next

    small = frame.iloc[-24 * 40:]
    fc = forecast_next(small, "delhi", horizons=[1, 24])
    assert list(fc["horizon_h"]) == [1, 24]
    assert (fc["target_time"] - fc["origin"]).dt.total_seconds().tolist() == [3600, 86400]
    assert fc["pred"].between(2000, 6000).all()
