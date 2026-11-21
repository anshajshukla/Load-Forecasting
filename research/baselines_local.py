"""Classical baselines and the LightGBM model on the common protocol (research/common.py).

    python -m research.baselines_local [--only sarimax,prophet,...]

Writes reports/paper/pred_<model>.csv for: lightgbm (the paper's model, with bias correction and
empirical intervals), lightgbm_raw, yesterday, last_week, ridge, sarimax, prophet.
SARIMAX and Prophet are refit once per test year on the 5 years before it. SARIMAX then filters
through the test year with fixed parameters, so each forecast uses energy up to d-1 only (one-step
ahead); Prophet has no autoregressive term and forecasts the whole year from weather and calendar.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from pipeline.daily import _xy, oos_predictions, postprocess

from .common import QCOLS, QUANTILES, TEST_YEARS, empirical_quantiles, exog, load_dev, save, year_split


def run_lightgbm(df):
    raw = oos_predictions(df, TEST_YEARS[0] - 2, TEST_YEARS[-1])  # two warm-up years for correction/intervals
    post = postprocess(df["energy_mu"], raw)
    keep = post.index >= f"{TEST_YEARS[0]}-01-01"
    q = empirical_quantiles(df["energy_mu"], post["pred"])
    save("lightgbm", post["pred"][keep], q[keep])
    save("lightgbm_raw", raw[keep], empirical_quantiles(df["energy_mu"], raw)[keep])


def run_naive(df):
    X, y = _xy(df, f"{TEST_YEARS[0] - 2}-01-01")
    for name, col in (("yesterday", "lag_1d"), ("last_week", "lag_7d")):
        p = X[col]
        keep = p.index >= f"{TEST_YEARS[0]}-01-01"
        save(name, p[keep], empirical_quantiles(y, p)[keep])


def run_ridge(df):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _xy(df, f"{TEST_YEARS[0] - 7}-01-01")
    cols = [c for c in X.columns if c != "lag_364d"]
    out = []
    for year in [TEST_YEARS[0] - 2, TEST_YEARS[0] - 1] + TEST_YEARS:
        tr, te = year_split(X.index, year)
        if tr.sum() < 365:
            continue
        Xl = X[cols].fillna(X[cols][tr].median())
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(Xl[tr], (y - X["lag_1d"])[tr])
        out.append(pd.Series(m.predict(Xl[te]) + X.loc[te, "lag_1d"].to_numpy(), index=X.index[te]))
    p = pd.concat(out)
    keep = p.index >= f"{TEST_YEARS[0]}-01-01"
    save("ridge", p[keep], empirical_quantiles(y, p)[keep])


def run_sarimax(df):
    """log(energy) ~ SARIMA(1,1,1)(1,0,1)7 with weather + calendar regressors."""
    from scipy.stats import norm
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    ly = np.log(df["energy_mu"])
    ex = exog(df).drop(columns=["dow", "doy_sin", "doy_cos"])  # weekly and yearly cycles: seasonal AR + differencing
    ok = ex.notna().all(axis=1)
    pts, qs = [], []
    for year in TEST_YEARS:
        tr, te = year_split(df.index, year)
        tr, te = tr & ok.to_numpy(), te & ok.to_numpy()
        idx_tr, idx_all = df.index[tr], df.index[tr | te]
        mu, sd = ex[tr].mean(), ex[tr].std().replace(0, 1)
        z = (ex - mu) / sd
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(ly[idx_tr], exog=z.loc[idx_tr], order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
                          freq="D").fit(disp=False, maxiter=200)
            # Same parameters, filtered over train + test: in-sample one-step predictions on the test year
            # use observations up to d-1 only.
            full = res.apply(ly[idx_all].asfreq("D"), exog=z.reindex(pd.date_range(idx_all[0], idx_all[-1])).ffill())
        pr = full.get_prediction(start=f"{year}-01-01", end=f"{year}-12-31")
        m, s = pr.predicted_mean, np.sqrt(pr.var_pred_mean)
        pts.append(np.exp(m))
        qs.append(pd.DataFrame({c: np.exp(m + s * norm.ppf(q)) for c, q in zip(QCOLS, QUANTILES)}))
        print(f"sarimax {year}: aic {res.aic:.0f}")
    save("sarimax", pd.concat(pts), pd.concat(qs))


def run_prophet(df):
    import logging

    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    ex = exog(df).drop(columns=["dow", "doy_sin", "doy_cos"])
    pts, qs = [], []
    for year in TEST_YEARS:
        tr, te = year_split(df.index, year)
        d = ex.assign(ds=df.index, y=df["energy_mu"].to_numpy())
        train = d[tr].dropna()
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                    seasonality_mode="multiplicative", uncertainty_samples=500)
        for c in ex.columns:
            m.add_regressor(c)
        m.fit(train)
        test = d[te].dropna(subset=list(ex.columns))
        samples = m.predictive_samples(test)["yhat"]
        f = m.predict(test)
        idx = pd.DatetimeIndex(test["ds"])
        pts.append(pd.Series(f["yhat"].to_numpy(), index=idx))
        qs.append(pd.DataFrame({c: np.quantile(samples, q, axis=1) for c, q in zip(QCOLS, QUANTILES)}, index=idx))
        print(f"prophet {year} done")
    save("prophet", pd.concat(pts), pd.concat(qs))


RUNNERS = {"lightgbm": run_lightgbm, "naive": run_naive, "ridge": run_ridge, "sarimax": run_sarimax,
           "prophet": run_prophet}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=",".join(RUNNERS))
    a = p.parse_args()
    df = load_dev()
    for name in a.only.split(","):
        RUNNERS[name](df)
