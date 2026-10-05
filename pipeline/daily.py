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
TRAIN_FROM, TEST_FROM = "2021-01-01", "2026-01-01"  # 5 training years (v2; v1 used 3)
# v2 (chosen on the 2016-2025 rolling-year test, mean MAPE 2.82% -> 2.71%; 2026 not used):
# extra heat build-up, growth, holiday-distance and weekday-ratio features, 5 training years,
# learning rate 0.02 with 600 trees. v1 (7 leaves / 300 trees, lr 0.03, 3 years) is in git history.
PARAMS = dict(n_estimators=600, learning_rate=0.02, num_leaves=7, min_child_samples=15,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0,
              n_jobs=1, deterministic=True)  # identical results on every run


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
    # Extend past the last known energy day while weather exists, so tomorrow can be forecast.
    idx = pd.date_range(e.index.min(), max(e.index.max(), wd.dropna(subset=["t_max"]).index.max()), freq="D")
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
    # Heat build-up over the last days (weather of d and earlier), and growth vs a year ago (energy <= d-1).
    f["t_mean_3d"] = df["t_mean"].rolling(3).mean()
    f["t_mean_7d"] = df["t_mean"].rolling(7).mean()
    f["t_max_3d"] = df["t_max"].rolling(3).mean()
    f["cdd_3d"] = (f["t_mean_3d"] - 24).clip(lower=0)
    f["yoy_growth"] = y.shift(1).rolling(28).mean() / y.shift(365).rolling(28).mean()
    # Typical ratio of this weekday to the trailing week, from the last 8 same weekdays (all <= d-1).
    rel = (y / y.rolling(7).mean()).shift(1)
    prev_dow = (df.index.dayofweek - 1) % 7
    f["dow_ratio_prev"] = rel.groupby(prev_dow).transform(lambda s: s.rolling(8, min_periods=4).median())
    f["lag7_ratio"] = y.shift(7) / y.shift(8)
    idx = df.index
    hol = holidays.India(years=range(idx.year.min() - 1, idx.year.max() + 2), subdiv="DL")
    hd = np.array(sorted(pd.to_datetime(list(hol)).values))
    pos = np.searchsorted(hd, idx.values)
    nxt = hd[np.minimum(pos, len(hd) - 1)]
    prv = hd[np.maximum(pos - 1, 0)]
    f["days_to_holiday"] = np.clip((nxt - idx.values) / np.timedelta64(1, "D"), 0, 10)
    f["days_since_holiday"] = np.clip((idx.values - prv) / np.timedelta64(1, "D"), 0, 10)
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


# Post-processing, chosen on the 2017-2025 out-of-sample errors (2026 not used):
# * bias correction: multiply by 1 + 0.5 x mean relative error of the last 28 days (MAPE 2.668% -> 2.600%,
#   bias -0.50% -> -0.12%; any window from 28 to 91 days gives about the same);
# * intervals: quantiles of the last 365 days' corrected errors (80%/95% coverage 79%/95% in every year,
#   vs 77%/93% for a fixed band from the previous year).
CORR_WINDOW, CORR_ALPHA, INTERVAL_WINDOW, YEARS_BACK = 28, 0.5, 365, 5


def oos_predictions(df, first_year: int, last_year: int, years_back: int = YEARS_BACK) -> pd.Series:
    """Day-ahead predictions for each year from a model trained on the `years_back` years before it."""
    X, y = _xy(df, f"{first_year - years_back}-01-01")
    out = []
    for year in range(first_year, last_year + 1):
        tr = (X.index >= f"{year - years_back}-01-01") & (X.index < f"{year}-01-01")
        te = (X.index >= f"{year}-01-01") & (X.index < f"{year + 1}-01-01")
        if te.any() and tr.sum() >= 365:  # skip years without at least a year of history to train on
            out.append(_fit_predict(X, y, tr, te)[0])
    return pd.concat(out) if out else pd.Series(dtype=float)


def postprocess(actual: pd.Series, raw: pd.Series) -> pd.DataFrame:
    """Bias-corrected forecast and 80%/95% intervals, each using only errors known by the day before."""
    a = actual.reindex(raw.index)
    err = a / raw - 1
    corr = (err.shift(1).rolling(CORR_WINDOW, min_periods=CORR_WINDOW // 2).mean() * CORR_ALPHA).fillna(0)
    pred = raw * (1 + corr)
    res = (a / pred - 1).shift(1).rolling(INTERVAL_WINDOW, min_periods=INTERVAL_WINDOW // 2)
    out = pd.DataFrame({"raw": raw, "pred": pred})
    for lvl in (80, 95):
        q = (1 - lvl / 100) / 2
        out[f"lo{lvl}"], out[f"hi{lvl}"] = pred * (1 + res.quantile(q)), pred * (1 + res.quantile(1 - q))
    return out


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
    # Earlier years' out-of-sample forecasts feed the bias correction and intervals (errors known by d-1 only).
    start_year = pd.Timestamp(test_from).year
    hist = oos_predictions(df, start_year - 2, start_year - 1)
    post = postprocess(df["energy_mu"], pd.concat([hist, p])).loc[p.index]
    frame = pd.DataFrame({"actual": y[te], "raw": p, "yesterday": X.loc[te, "lag_1d"], "last_week": X.loc[te, "lag_7d"]})
    frame = frame.join(post.drop(columns="raw"))
    r["raw_model_mape"] = r["model_mape"]
    r["model_mape"] = mape(frame.actual, frame.pred)
    r["mae_mu"] = float(np.nanmean(np.abs(frame.actual - frame.pred)))
    r["bias_pct"] = float(((frame.pred - frame.actual) / frame.actual).mean() * 100)
    r["raw_bias_pct"] = float(((frame.raw - frame.actual) / frame.actual).mean() * 100)
    r["skill_vs_best_baseline_pct"] = 100 * (1 - r["model_mape"] / min(r["yesterday_mape"], r["last_week_mape"]))
    r["intervals"] = {}
    for level in (80, 95):
        inside = (frame.actual >= frame[f"lo{level}"]) & (frame.actual <= frame[f"hi{level}"])
        r["intervals"][level] = {"coverage": float(inside.mean() * 100),
                                 "mean_width_mu": float((frame[f"hi{level}"] - frame[f"lo{level}"]).mean())}
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
        lines += ["", f"Model before bias correction: {r['raw_model_mape']:.2f}% (bias {r['raw_bias_pct']:+.2f}%); "
                  f"after: {r['model_mape']:.2f}% (bias {r['bias_pct']:+.2f}%). The correction uses only errors known the "
                  "day before.", "", "## Prediction intervals", "",
                  "Quantiles of the last 365 days' errors, each known by the day before.", "",
                  "| Nominal | 2026 coverage | Mean width (MU) |", "|---|---|---|"]
        for lvl, v in r["intervals"].items():
            lines.append(f"| {lvl}% | {v['coverage']:.1f}% | {v['mean_width_mu']:.1f} |")
        lines += ["", "## Most-used features", "", ", ".join(r["top_features"])]
        (a.out / "daily_holdout.json").write_text(json.dumps(r, indent=2))
        (a.out / "daily_holdout.md").write_text("\n".join(lines) + "\n")
        frame.to_csv(a.out / "daily_holdout_predictions.csv")
    print((a.out / f"daily_{a.cmd}.md").read_text())


if __name__ == "__main__":
    main()
