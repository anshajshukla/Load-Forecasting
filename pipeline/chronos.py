"""Chronos-2 (Amazon's time-series foundation model, zero-shot) for the live LightGBM + Chronos-2 average.

Chronos-2 is never trained on this data: it reads the last 512 days of energy plus the weather and calendar
covariates, and forecasts the next days given those covariates. On 2016-2025 the equal-weight average of the
bias-corrected LightGBM and Chronos-2 forecasts scored 2.49% vs 2.63% for LightGBM alone
(reports/model_comparison.md). Optional: if torch/chronos-forecasting are missing or the model can't load,
available() is False and the live forecast stays LightGBM-only, and says so.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .daily import features

MODEL, CTX = "amazon/chronos-2", 512
EXOG = ["t_max", "t_min", "t_mean", "feels_max", "rh_mean", "rain_mm", "cdd", "dow", "is_holiday",
        "doy_sin", "doy_cos"]


@lru_cache(maxsize=1)
def _pipe():
    import torch
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(MODEL, device_map="cuda" if torch.cuda.is_available() else "cpu")


def available() -> tuple[bool, str]:
    try:
        _pipe()
        return True, MODEL
    except Exception as e:  # noqa: BLE001 - optional model; the caller falls back to LightGBM only
        return False, f"{type(e).__name__}: {e}"[:200]


def _inputs(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    ex = features(df)[EXOG].copy()
    ex[["cdd", "rain_mm"]] = ex[["cdd", "rain_mm"]].fillna(0)
    ex = ex.interpolate(limit_direction="both")
    y = df["energy_mu"].interpolate(limit_area="inside")  # source gaps filled as inputs only, never scored
    return y, ex


def forecast(df: pd.DataFrame, origins: list[pd.Timestamp], horizon: int = 1) -> pd.DataFrame:
    """For each origin o (last known day), forecast days o+1..o+horizon from energy <= o and covariates.
    Returns a frame indexed by origin with columns 1..horizon."""
    y, ex = _inputs(df)
    hist, fut = [], []
    for o in origins:
        i = df.index.get_loc(o) + 1
        h = ex.iloc[max(0, i - CTX):i].assign(target=y.iloc[max(0, i - CTX):i].to_numpy())
        f = ex.iloc[i:i + horizon]
        hist.append(h.assign(item_id=str(o.date()), timestamp=h.index))
        fut.append(f.assign(item_id=str(o.date()), timestamp=f.index))
    p = _pipe().predict_df(pd.concat(hist, ignore_index=True), future_df=pd.concat(fut, ignore_index=True),
                           prediction_length=horizon, quantile_levels=[0.5])
    p["step"] = p.groupby("item_id").cumcount() + 1
    out = p.pivot(index="item_id", columns="step", values="predictions")
    out.index = pd.to_datetime(out.index)
    return out
