"""Live forecast log: forecasts are never overwritten, actuals are filled in once known."""
import pandas as pd

from pipeline.forecast import update_log


def test_log_keeps_first_forecast_and_scores_it(tmp_path):
    path = tmp_path / "live.csv"
    actual = pd.DataFrame({"energy_mu": [100.0, float("nan")]}, index=pd.to_datetime(["2026-10-01", "2026-10-02"]))
    f = {"target_date": "2026-10-02", "made_at_utc": "x", "pred": 110.0, "lo80": 1, "hi80": 2, "lo95": 1, "hi95": 2}
    update_log(f, actual, path)
    update_log(dict(f, pred=999.0), actual, path)  # a later run must not replace it
    log = pd.read_csv(path)
    assert log["pred"].tolist() == [110.0] and log["actual"].isna().all()
    actual.loc["2026-10-02", "energy_mu"] = 100.0
    log = update_log(None, actual, path)
    assert log["actual"].tolist() == [100.0] and log["abs_pct_error"].tolist() == [10.0]


def test_archive_replaces_forecast_but_forecast_only_fills(tmp_path):
    from pipeline.forecast import merge_weather
    path = tmp_path / "w.csv"
    idx = pd.to_datetime(["2026-10-01 00:00", "2026-10-01 01:00"])
    pd.DataFrame({"datetime": idx[:1], "temperature_2m": [20.0]}).to_csv(path, index=False)
    merge_weather(pd.DataFrame({"temperature_2m": [99.0, 21.0]}, index=idx), path)              # forecast
    assert pd.read_csv(path)["temperature_2m"].tolist() == [20.0, 21.0]
    merge_weather(pd.DataFrame({"temperature_2m": [25.0]}, index=idx[1:]), path, prefer_new=True)  # archive
    assert pd.read_csv(path)["temperature_2m"].tolist() == [20.0, 25.0]
