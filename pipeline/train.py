"""Walk-forward backtest, final model training and forecasting."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import build_features

LGB_PARAMS = dict(n_estimators=800, learning_rate=0.03, num_leaves=63, min_child_samples=30,
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


# The model learns the change from the last known value at the same lag (y - lag_h), not the level.
# Trees can't extrapolate levels, and Delhi's load grows every year, so a level model under-predicts
# record peaks; the residual is closer to stationary. It also improved both horizons in the backtest ablation.
def _fit(X, y, horizon):
    return lgb.LGBMRegressor(**LGB_PARAMS).fit(X, y - X[f"lag_{horizon}h"])


def _predict(m, X, horizon):
    return m.predict(X) + X[f"lag_{horizon}h"].to_numpy()


def _daily_peak(p: pd.DataFrame, col: str) -> float:
    """MAPE of each day's predicted maximum vs the actual maximum (what peak scheduling cares about)."""
    d = p.groupby(p.index.date).agg(a=("actual", "max"), p=(col, "max"))
    return mape(d.a, d.p)


def backtest(df: pd.DataFrame, target: str, horizon: int, n_folds: int = 6, fold_days: int = 30,
             min_train_days: int = 365, groups: pd.Series | None = None) -> dict:
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
        m = _fit(X[tr], y[tr], horizon)
        p = pd.Series(_predict(m, X[te], horizon), index=y.index[te])
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
        "daily_peak_mape": _daily_peak(allp, "pred"),
        "daily_peak_mape_seasonal_naive": _daily_peak(allp, "seasonal_naive"),
        "mape_by_hour": by_hour,
        "test_period": [str(allp.index.min()), str(allp.index.max())],
        "n_test_hours": int(len(allp)),
    }
    if groups is not None:
        g = groups.reindex(allp.index).fillna("unknown").astype(str)
        summary["by_group"] = {
            k: {"n": int(len(v)), "model_mape": mape(v.actual, v.pred),
                "persistence_mape": mape(v.actual, v.persistence), "seasonal_naive_mape": mape(v.actual, v.seasonal_naive)}
            for k, v in allp.groupby(g)}
    summary["skill_vs_best_baseline_pct"] = 100 * (1 - summary["model_mape"] /
                                                   min(summary["persistence_mape"], summary["seasonal_naive_mape"]))
    return summary


def fit_final(df: pd.DataFrame, target: str, horizon: int) -> lgb.LGBMRegressor:
    X, y = _xy(df, target, horizon)
    return _fit(X, y, horizon)


def forecast_next(df: pd.DataFrame, target: str, horizons=range(1, 25), models: dict | None = None) -> pd.DataFrame:
    """Forecast the next len(horizons) hours after the last observed hour, one direct model per horizon."""
    last = df[target].last_valid_index()
    rows = []
    for h in horizons:
        m = models[h] if models and h in models else fit_final(df.loc[:last], target, h)
        ts = last + pd.Timedelta(hours=h)
        ext = df.reindex(pd.date_range(df.index.min(), ts, freq="h"))
        X = build_features(ext, target, h).loc[[ts]]
        rows.append({"target_time": ts, "horizon_h": h, "origin": last, "pred": float(_predict(m, X, h)[0])})
    return pd.DataFrame(rows)


def write_report(results: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "backtest.json").write_text(json.dumps(results, indent=2, default=str))
    lines = ["# Backtest report", "",
             "Walk-forward, expanding window, 30-day folds. Weather inputs are archive actuals, so day-ahead",
             "numbers are slightly optimistic versus using a real weather forecast.", "",
             "| Target | Horizon | Model MAPE | Persistence | Seasonal naive | Skill vs best baseline | MAE (MW) | Daily-peak MAPE (naive) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['target']} | {r['horizon_h']}h | {r['model_mape']:.2f}% | {r['persistence_mape']:.2f}% | "
                     f"{r['seasonal_naive_mape']:.2f}% | {r['skill_vs_best_baseline_pct']:.0f}% | {r['model_mae_mw']:.0f} | "
                     f"{r['daily_peak_mape']:.2f}% ({r['daily_peak_mape_seasonal_naive']:.2f}%) |")
    lines += ["", f"Test period: {results[0]['test_period'][0]} to {results[0]['test_period'][1]}"]
    if any("by_group" in r for r in results):
        lines += ["", "## By data source", "",
                  "| Horizon | Source | Hours | Model MAPE | Persistence | Seasonal naive |", "|---|---|---|---|---|---|"]
        for r in results:
            for k, v in r.get("by_group", {}).items():
                lines.append(f"| {r['horizon_h']}h | {k} | {v['n']} | {v['model_mape']:.2f}% | "
                             f"{v['persistence_mape']:.2f}% | {v['seasonal_naive_mape']:.2f}% |")
    lines += ["", "## MAPE by hour of day", "", "| Hour | " + " | ".join(f"{r['horizon_h']}h" for r in results) + " |",
              "|---|" + "---|" * len(results)]
    for h in range(24):
        lines.append(f"| {h:02d} | " + " | ".join(f"{r['mape_by_hour'].get(h, float('nan')):.2f}%" for r in results) + " |")
    (out / "backtest.md").write_text("\n".join(lines) + "\n")
