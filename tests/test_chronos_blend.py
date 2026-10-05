"""Live ensemble: the forecast is the average of LightGBM and Chronos-2, or LightGBM alone if Chronos-2 is missing."""
import pandas as pd
import pytest

from pipeline import chronos
from pipeline.daily import load
from pipeline.live import forecast_days


def _cut():
    df = load()
    cut = df.loc[:"2026-09-30"].copy()
    cut.loc["2026-09-26":, "energy_mu"] = float("nan")
    return cut


def test_falls_back_to_lightgbm(monkeypatch):
    monkeypatch.setattr(chronos, "available", lambda: (False, "not installed"))
    cut = _cut()
    a, b = forecast_days(cut, 7), forecast_days(cut, 7, ensemble=True)
    assert (a["pred"] == b["pred"]).all()
    assert b.attrs["model"].startswith("LightGBM (Chronos-2 unavailable")


def test_blend_is_average(monkeypatch):
    monkeypatch.setattr(chronos, "available", lambda: (True, "fake"))
    def fake(df, origins, horizon):  # constant 100 MU forecast
        return pd.DataFrame(100.0, index=pd.to_datetime(origins), columns=range(1, horizon + 1))
    monkeypatch.setattr(chronos, "forecast", fake)
    cut = _cut()
    fc = forecast_days(cut, 7, ensemble=True)
    assert fc.attrs["model"] == "LightGBM + Chronos-2 average"
    assert ((fc["pred"] - (fc["pred_lgbm"] + fc["pred_chronos"]) / 2).abs() < 1e-9).all()
    assert (fc["lo95"] < fc["pred"]).all() and (fc["pred"] < fc["hi95"]).all()


def test_real_chronos_runs():
    if not chronos.available()[0]:
        pytest.skip("Chronos-2 not installed")
    fc = forecast_days(_cut(), 7, ensemble=True)
    assert fc.attrs["model"] == "LightGBM + Chronos-2 average"
    assert ((fc["pred_chronos"] / fc["pred_lgbm"] - 1).abs() < 0.15).all()
