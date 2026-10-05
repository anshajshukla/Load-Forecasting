"""Live forecaster, offline: steps forward day by day, never past the weather it has, ranges widen with distance."""
import pandas as pd

from pipeline.daily import load
from pipeline.live import forecast_days


def test_forecast_days_offline():
    df = load()
    cut = df.loc[:"2026-09-30"].copy()
    cut.loc["2026-09-26":, "energy_mu"] = float("nan")   # pretend the last 5 days are not reported yet
    fc = forecast_days(cut, horizon=7)
    assert fc.attrs["last_known"] == pd.Timestamp("2026-09-25")
    assert list(fc["days_ahead"]) == [1, 2, 3, 4, 5]            # stops where the weather ends (30 Sep)
    width = fc["hi95"] - fc["lo95"]
    assert (width.diff().dropna() > -1e-9).all() or width.iloc[-1] > width.iloc[0]
    assert abs(fc.attrs["correction_pct"]) <= 2.0 + 1e-9
    # Day 1 must equal a direct one-step prediction of the same model, times the same correction.
    import lightgbm as lgb
    from pipeline.daily import PARAMS, YEARS_BACK, _xy, features
    last = fc.attrs["last_known"]
    X, y = _xy(cut.loc[:last], (last - pd.DateOffset(years=YEARS_BACK)).strftime("%Y-%m-%d"))
    m = lgb.LGBMRegressor(**PARAMS).fit(X, y - X["lag_1d"])
    x = features(cut.loc[:fc.index[0]]).loc[[fc.index[0]]][X.columns]
    direct = (m.predict(x)[0] + x["lag_1d"].iloc[0]) * (1 + fc.attrs["correction_pct"] / 100)
    assert abs(fc["pred"].iloc[0] - direct) < 1e-6
