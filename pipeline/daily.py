"""Day-ahead forecast of Delhi's daily energy (MU) on real Grid-India data only.

    python -m pipeline.daily backtest   # development: walk-forward inside 2023-2025
    python -m pipeline.daily holdout    # once: train on 2023-2025, score 2026

Inputs: data/posoco/delhi_daily.csv (real daily energy met) and data/weather/delhi_hourly.csv
(Open-Meteo archive, IST). A forecast for day d is made at the end of day d-1: energy features
use days <= d-1 only; weather for day d is the archive actual (stands in for a weather forecast,
so the numbers are slightly optimistic).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

DAILY_CSV = Path("data/posoco/delhi_daily.csv")
WEATHER_CSV = Path("data/weather/delhi_hourly.csv")
TRAIN_FROM, TEST_FROM = "2023-01-01", "2026-01-01"
PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=15,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)


def load(daily_csv=DAILY_CSV, weather_csv=WEATHER_CSV) -> pd.DataFrame:
    e = pd.read_csv(daily_csv, parse_dates=["date"]).set_index("date")["energy_mu"]
    w = pd.read_csv(weather_csv, parse_dates=["datetime"]).set_index("datetime")
    g = w.resample("D")
    wd = pd.DataFrame({
        "t_max": g["temperature_2m"].max(), "t_min": g["temperature_2m"].min(),
        "t_mean": g["temperature_2m"].mean(), "feels_max": g["apparent_temperature"].max(),
        "rh_mean": g["relative_humidity_2m"].mean(), "dew_mean": g["dew_point_2m"].mean(),
        "rain_mm": g["precipitation"].sum(), "cloud_mean": g["cloud_cover"].mean(),
        "radiation_sum": g["shortwave_radiation"].sum(), "wind_mean": g["wind_speed_10m"].mean(),
    })
    idx = pd.date_range(e.index.min(), e.index.max(), freq="D")
    df = wd.reindex(idx).join(e.reindex(idx))
    df.index.name = "date"
    return df


def features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per target day d; energy features only from days <= d-1."""
    y = df["energy_mu"]
    f = pd.DataFrame(index=df.index)
    for k in (1, 2, 3, 7, 14, 364):
        f[f"lag_{k}d"] = y.shift(k)
    f["mean_7d"] = y.shift(1).rolling(7, min_periods=5).mean()
    f["mean_28d"] = y.shift(1).rolling(28, min_periods=20).mean()
    f["trend_7d"] = f["lag_1d"] - y.shift(8)
    for c in ("t_max", "t_min", "t_mean", "feels_max", "rh_mean", "dew_mean", "rain_mm", "cloud_mean",
              "radiation_sum", "wind_mean"):
        f[c] = df[c]
    f["cdd"] = (df["t_mean"] - 24).clip(lower=0)
    f["hdd"] = (16 - df["t_mean"]).clip(lower=0)
    f["d_t_max"] = df["t_max"] - df["t_max"].shift(1)  # weather change vs the last known day
    f["t_max_yday"] = df["t_max"].shift(1)
    idx = df.index
    hol = holidays.India(years=range(idx.year.min() - 1, idx.year.max() + 2), subdiv="DL")
    f["dow"] = idx.dayofweek
    f["is_holiday"] = pd.Index(idx.date).isin(list(hol)).astype(int)
    f["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    return f


def mape(a, p) -> float:
    a, p = np.asarray(a, float), np.asarray(p, float)
    m = ~(np.isnan(a) | np.isnan(p))
    return float(np.mean(np.abs(a[m] - p[m]) / a[m]) * 100)


def _fit_predict(X, y, tr, te):
    m = lgb.LGBMRegressor(**PARAMS).fit(X[tr], (y - X["lag_1d"])[tr])  # learn the change from yesterday
    return pd.Series(m.predict(X[te]) + X.loc[te, "lag_1d"].to_numpy(), index=X.index[te]), m


def _score(y, X, pred, te) -> dict:
    a = y[te]
    out = {"days": int(te.sum()), "model_mape": mape(a, pred),
           "yesterday_mape": mape(a, X.loc[te, "lag_1d"]), "last_week_mape": mape(a, X.loc[te, "lag_7d"]),
           "mae_mu": float(np.nanmean(np.abs(a - pred)))}
    out["skill_vs_best_baseline_pct"] = 100 * (1 - out["model_mape"] / min(out["yesterday_mape"], out["last_week_mape"]))
    return out


def _xy(df, train_from):
    X, y = features(df), df["energy_mu"]
    ok = y.notna() & X["lag_1d"].notna() & X["t_max"].notna() & (X.index >= train_from)
    return X[ok], y[ok]


def backtest(df, train_from=TRAIN_FROM, until=TEST_FROM, folds=12) -> dict:
    """Walk-forward over the last `folds` months before `until`; never touches days from `until` on."""
    X, y = _xy(df[df.index < until], train_from)
    months = pd.period_range(end=pd.Timestamp(until) - pd.Timedelta(days=1), periods=folds, freq="M")
    preds = []
    for mth in months:
        te = (X.index.to_period("M") == mth)
        tr = X.index < mth.start_time
        p, _ = _fit_predict(X, y, tr, te)
        preds.append(p)
    p = pd.concat(preds)
    te = X.index.isin(p.index)
    r = _score(y, X, p.reindex(X.index[te]), te)
    r.update(kind="walk-forward", period=[str(p.index.min().date()), str(p.index.max().date())])
    return r


def holdout(df, train_from=TRAIN_FROM, test_from=TEST_FROM) -> tuple[dict, pd.DataFrame]:
    X, y = _xy(df, train_from)
    tr, te = X.index < test_from, X.index >= test_from
    p, m = _fit_predict(X, y, tr, te)
    r = _score(y, X, p, te)
    r.update(kind="holdout", train=[str(X.index[tr].min().date()), str(X.index[tr].max().date())],
             test=[str(X.index[te].min().date()), str(X.index[te].max().date())])
    frame = pd.DataFrame({"actual": y[te], "pred": p, "yesterday": X.loc[te, "lag_1d"], "last_week": X.loc[te, "lag_7d"]})
    r["by_month"] = {str(k): {"model": mape(g.actual, g.pred), "yesterday": mape(g.actual, g.yesterday),
                              "last_week": mape(g.actual, g.last_week)}
                     for k, g in frame.groupby(frame.index.to_period("M"))}
    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
    r["top_features"] = imp.head(8).to_dict()
    return r, frame


def _table(r: dict) -> list[str]:
    return ["| Model | Yesterday | Same day last week | Skill vs best baseline | MAE (MU) | Days |",
            "|---|---|---|---|---|---|",
            f"| **{r['model_mape']:.2f}%** | {r['yesterday_mape']:.2f}% | {r['last_week_mape']:.2f}% | "
            f"{r['skill_vs_best_baseline_pct']:.0f}% | {r['mae_mu']:.2f} | {r['days']} |"]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="pipeline.daily")
    p.add_argument("cmd", choices=["backtest", "holdout"])
    p.add_argument("--train-from", default=TRAIN_FROM)
    p.add_argument("--test-from", default=TEST_FROM)
    p.add_argument("--out", type=Path, default=Path("reports"))
    a = p.parse_args(argv)
    df = load()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "backtest":
        r = backtest(df, a.train_from, a.test_from)
        lines = ["# Daily energy: development backtest (real data, 2026 excluded)", "",
                 f"Walk-forward, monthly folds {r['period'][0]} to {r['period'][1]}, training from {a.train_from}, "
                 "expanding window. Day-ahead MAPE of Delhi's daily energy met.", "", *_table(r)]
        (a.out / "daily_backtest.json").write_text(json.dumps(r, indent=2))
        (a.out / "daily_backtest.md").write_text("\n".join(lines) + "\n")
    else:
        r, frame = holdout(df, a.train_from, a.test_from)
        lines = ["# Daily energy: held-out test year (real data only)", "",
                 f"Trained once on {r['train'][0]} to {r['train'][1]}; tested day-ahead on {r['test'][0]} to "
                 f"{r['test'][1]}, which was never used for fitting or tuning. Source: Grid-India daily reports "
                 "(data/posoco). Weather inputs are archive actuals.", "", *_table(r), "", "## By month", "",
                 "| Month | Model | Yesterday | Same day last week |", "|---|---|---|---|"]
        for k, v in r["by_month"].items():
            lines.append(f"| {k} | {v['model']:.2f}% | {v['yesterday']:.2f}% | {v['last_week']:.2f}% |")
        lines += ["", "## Most-used features", "", ", ".join(r["top_features"])]
        (a.out / "daily_holdout.json").write_text(json.dumps(r, indent=2))
        (a.out / "daily_holdout.md").write_text("\n".join(lines) + "\n")
        frame.to_csv(a.out / "daily_holdout_predictions.csv")
    print((a.out / f"daily_{a.cmd}.md").read_text())


if __name__ == "__main__":
    main()
