"""Walk-forward backtest, final model training and forecasting."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import build_features

LGB_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)


def mape(a, p) -> float:
    a, p = np.asarray(a, float), np.asarray(p, float)
    m = np.isfinite(a) & np.isfinite(p) & (a > 0)
    return float(np.mean(np.abs((a[m] - p[m]) / a[m])) * 100)


def mae(a, p) -> float:
    a, p = np.asarray(a, float), np.asarray(p, float)
    m = np.isfinite(a) & np.isfinite(p)
    return float(np.mean(np.abs(a[m] - p[m])))


@dataclass
class FoldResult:
    fold_start: str
    fold_end: str
    n: int
    model_mape: float
    persistence_mape: float
    seasonal_naive_mape: float
    model_mae: float
    daily_peak_mape: float


def _xy(df, target, horizon):
    X = build_features(df, target, horizon)
    y = df[target]
    ok = y.notna() & X[[c for c in X if c.startswith("lag_")]].notna().all(axis=1)
    return X[ok], y[ok]


def backtest(df: pd.DataFrame, target: str, horizon: int, n_folds: int = 6, fold_days: int = 30,
             min_train_days: int = 365) -> dict:
    """Expanding-window walk-forward: train on everything before each fold, test on the fold."""
    X, y = _xy(df, target, horizon)
    end = y.index.max()
    folds, preds = [], []
    for k in range(n_folds, 0, -1):
        f_start = end - pd.Timedelta(days=fold_days * k) + pd.Timedelta(hours=1)
        f_end = f_start + pd.Timedelta(days=fold_days) - pd.Timedelta(hours=1)
        # Gap of `horizon` hours so no training label lies after the first test origin.
        tr = y.index < f_start - pd.Timedelta(hours=horizon - 1)
        te = (y.index >= f_start) & (y.index <= f_end)
        if tr.sum() < min_train_days * 24 or te.sum() == 0:
            continue
        m = lgb.LGBMRegressor(**LGB_PARAMS).fit(X[tr], y[tr])
        p = pd.Series(m.predict(X[te]), index=y.index[te])
        a = y[te]
        persist = X.loc[te, f"lag_{horizon}h"]
        seasonal = X.loc[te, "same_hour_last_known_day"]
        dp = pd.DataFrame({"a": a, "p": p}).groupby(a.index.date).max()
        folds.append(FoldResult(str(f_start), str(f_end), int(te.sum()), mape(a, p), mape(a, persist),
                                mape(a, seasonal), mae(a, p), mape(dp.a, dp.p)))
        preds.append(pd.DataFrame({"actual": a, "pred": p, "persistence": persist, "seasonal_naive": seasonal}))
    if not folds:
        raise ValueError("not enough history for a backtest")
    allp = pd.concat(preds)
    by_hour = allp.groupby(allp.index.hour).apply(lambda g: mape(g.actual, g.pred)).round(2).to_dict()
    summary = {
        "target": target, "horizon_h": horizon, "folds": [asdict(f) for f in folds],
        "model_mape": mape(allp.actual, allp.pred),
        "persistence_mape": mape(allp.actual, allp.persistence),
        "seasonal_naive_mape": mape(allp.actual, allp.seasonal_naive),
        "model_mae_mw": mae(allp.actual, allp.pred),
        "mape_by_hour": by_hour,
        "test_period": [str(allp.index.min()), str(allp.index.max())],
        "n_test_hours": int(len(allp)),
    }
    summary["skill_vs_best_baseline_pct"] = 100 * (1 - summary["model_mape"] /
                                                   min(summary["persistence_mape"], summary["seasonal_naive_mape"]))
    return summary


def fit_final(df: pd.DataFrame, target: str, horizon: int) -> lgb.LGBMRegressor:
    X, y = _xy(df, target, horizon)
    return lgb.LGBMRegressor(**LGB_PARAMS).fit(X, y)


def forecast_next(df: pd.DataFrame, target: str, horizons=range(1, 25), models: dict | None = None) -> pd.DataFrame:
    """Forecast the next len(horizons) hours after the last observed hour, one direct model per horizon."""
    last = df[target].last_valid_index()
    rows = []
    for h in horizons:
        m = models[h] if models and h in models else fit_final(df.loc[:last], target, h)
        ts = last + pd.Timedelta(hours=h)
        ext = df.reindex(pd.date_range(df.index.min(), ts, freq="h"))
        X = build_features(ext, target, h).loc[[ts]]
        rows.append({"target_time": ts, "horizon_h": h, "origin": last, "pred": float(m.predict(X)[0])})
    return pd.DataFrame(rows)


def write_report(results: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "backtest.json").write_text(json.dumps(results, indent=2, default=str))
    lines = ["# Backtest report", "",
             "Walk-forward, expanding window, 30-day folds. Weather inputs are archive actuals, so day-ahead",
             "numbers are slightly optimistic versus using a real weather forecast.", "",
             "| Target | Horizon | Model MAPE | Persistence | Seasonal naive | Skill vs best baseline | MAE (MW) |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['target']} | {r['horizon_h']}h | {r['model_mape']:.2f}% | {r['persistence_mape']:.2f}% | "
                     f"{r['seasonal_naive_mape']:.2f}% | {r['skill_vs_best_baseline_pct']:.0f}% | {r['model_mae_mw']:.0f} |")
    lines += ["", f"Test period: {results[0]['test_period'][0]} to {results[0]['test_period'][1]}"]
    (out / "backtest.md").write_text("\n".join(lines) + "\n")
