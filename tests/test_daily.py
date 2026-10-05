"""Daily model: features for day d must not change when energy on day d or later changes."""
import numpy as np
import pandas as pd

from pipeline.daily import features, holdout


def frame(n=900):
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    t = 25 + 10 * np.sin(2 * np.pi * (idx.dayofyear - 100) / 365)
    df = pd.DataFrame({c: t for c in ("t_max", "t_min", "t_mean", "feels_max", "rh_mean", "dew_mean",
                                      "rain_mm", "cloud_mean", "radiation_sum", "wind_mean")}, index=idx)
    df["energy_mu"] = 90 + 3 * (t - 25) + rng.normal(0, 2, n)
    return df


def test_no_future_energy_in_features():
    df = frame()
    d = 600
    base = features(df).iloc[d]
    bumped = df.copy()
    bumped.iloc[d:, bumped.columns.get_loc("energy_mu")] *= 3
    assert features(bumped).iloc[d].equals(base)


def test_holdout_trains_only_before_test_year():
    r, f = holdout(frame(), "2023-01-01", "2025-01-01")
    assert r["train"][1] < "2025-01-01" <= r["test"][0]
    assert r["model_mape"] < r["yesterday_mape"]


def test_postprocess_uses_only_past_errors():
    from pipeline.daily import postprocess
    idx = pd.date_range("2024-01-01", periods=500, freq="D")
    rng = np.random.default_rng(1)
    actual = pd.Series(100 + rng.normal(0, 3, 500), index=idx)
    raw = pd.Series(100.0, index=idx)
    base = postprocess(actual, raw).iloc[400]
    bumped = actual.copy()
    bumped.iloc[400:] *= 2
    assert postprocess(bumped, raw).iloc[400].equals(base)
