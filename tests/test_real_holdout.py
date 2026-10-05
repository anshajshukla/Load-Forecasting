"""All-real path: SLDC loads + weather only, test year held out, no legacy/synthetic rows."""
import numpy as np
import pandas as pd

from pipeline import TARGETS
from pipeline.__main__ import main
from pipeline.real import build_real
from pipeline.train import holdout


def write_inputs(tmp_path):
    idx = pd.date_range("2024-01-01", "2026-03-31 23:00", freq="h")
    rng = np.random.default_rng(1)
    base = 4000 + 1500 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 60, len(idx))
    loads = pd.DataFrame({t: base * f for t, f in zip(TARGETS, [1, .4, .2, .3, .05, .01])}, index=idx)
    loads.index.name = "datetime"
    loads.to_csv(tmp_path / "hourly.csv")
    w = pd.DataFrame({"datetime": idx, "temperature_2m": 25 + 10 * np.sin(2 * np.pi * idx.dayofyear / 365)})
    w.to_csv(tmp_path / "weather.csv", index=False)
    return tmp_path / "hourly.csv", tmp_path / "weather.csv"


def test_build_real_has_only_sldc_rows(tmp_path):
    s, w = write_inputs(tmp_path)
    df = build_real(s, w)
    assert set(df["data_source"]) == {"sldc"}
    assert "temperature_2m" in df and df["delhi"].notna().all()


def test_holdout_never_trains_on_test_year(tmp_path):
    s, w = write_inputs(tmp_path)
    r = holdout(build_real(s, w), "delhi", 24, "2026-01-01")
    assert pd.Timestamp(r["train_period"][1]) < pd.Timestamp("2026-01-01") - pd.Timedelta(hours=23)
    assert r["test_period"][0].startswith("2026-01-01")
    assert r["model_mape"] < r["seasonal_naive_mape"] * 1.5


def test_cli_holdout_writes_report(tmp_path):
    s, w = write_inputs(tmp_path)
    main(["holdout", "--real", str(s), "--weather", str(w), "--horizons", "1", "--out", str(tmp_path / "rep")])
    assert "never used for fitting" in (tmp_path / "rep" / "holdout.md").read_text()
