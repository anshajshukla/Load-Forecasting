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


def test_ex_ante_means_made_before_the_day_starts_in_ist():
    from pipeline.forecast import is_ex_ante
    assert is_ex_ante("2026-11-20T15:25:03+00:00", "2026-11-21")       # 20:55 IST the evening before
    assert not is_ex_ante("2026-11-20T15:25:03+00:00", "2026-11-20")   # same day: weather already observed
    assert not is_ex_ante("2026-11-20T18:31:00+00:00", "2026-11-21")   # 00:01 IST on the day itself


def test_forecast_tomorrow_steps_over_the_reporting_delay():
    from datetime import datetime, timezone

    from pipeline.daily import load
    from pipeline.forecast import forecast_tomorrow
    df = load().loc[:"2026-09-30"].copy()
    df.loc["2026-09-27":, "energy_mu"] = float("nan")       # Grid-India two days behind on the 28th
    now = datetime(2026, 9, 28, 14, 0, tzinfo=timezone.utc)  # 19:30 IST on 28 Sep
    f = forecast_tomorrow(df, now)
    assert f["target_date"] == "2026-09-29" and f["days_ahead"] == 3 and f["ex_ante"]
    assert f["lo95"] < f["lo80"] < f["pred"] < f["hi80"] < f["hi95"]


def test_old_rows_get_marked_and_new_columns_added(tmp_path):
    path = tmp_path / "live.csv"
    pd.DataFrame([{"target_date": "2026-11-19", "made_at_utc": "2026-11-19T15:42:16+00:00", "pred": 79.63,
                   "lo80": 1, "hi80": 2, "lo95": 1, "hi95": 2, "actual": None, "abs_pct_error": None}]).to_csv(path, index=False)
    actual = pd.DataFrame({"energy_mu": [77.67]}, index=pd.to_datetime(["2026-11-19"]))
    log = update_log(None, actual, path)
    assert log["days_ahead"].tolist() == [1] and log["ex_ante"].tolist() == [False]
    assert log["abs_pct_error"].tolist() == [2.52]


def test_weather_outage_still_scores_and_fails_loudly(tmp_path, monkeypatch):
    import pytest

    import pipeline.forecast as fc

    def down():
        raise TimeoutError("open-meteo down")

    log_path = tmp_path / "live.csv"
    orig = fc.update_log
    monkeypatch.setattr(fc, "fetch_forecast_weather", down)
    monkeypatch.setattr(fc, "update_log", lambda new, df: orig(new, df, log_path))

    def no_weather_for_tomorrow(df):
        raise ValueError("no weather for tomorrow")

    monkeypatch.setattr(fc, "forecast_tomorrow", no_weather_for_tomorrow)
    # The weather download fails and there is no forecast, but the log is still written and the run fails.
    with pytest.raises(SystemExit) as e:
        fc.main([])
    assert e.value.code == 1 and log_path.exists()
