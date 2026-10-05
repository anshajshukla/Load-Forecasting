"""Compare the LightGBM model with deep-learning and foundation models on the same day-ahead test.

Run: python scripts/model_comparison.py (writes reports/model_comparison.md and reports/model_comparison.csv).
Needs requirements-research.txt (torch, neuralforecast, chronos-forecasting); the Chronos weights are
downloaded from Hugging Face on first run.

Protocol (same as scripts/rolling_years.py): every day d of every year 2016-2026 is forecast from data up to
d-1 plus the archive weather of d. Trained models (Ridge, LightGBM, N-HiTS) are trained on the 5 years before
the test year only. Chronos models are zero-shot (never trained on this data) and see the last 512 days.
Every model gets the same bias correction (pipeline.daily.postprocess). Neural models need a gap-free input,
so the 112 days missing at the source (all before 2023) are linearly interpolated *as inputs only*; they are
never scored.
"""
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline.daily as d  # noqa: E402

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

FIRST, LAST, BACK, CTX = 2015, 2026, d.YEARS_BACK, 512  # 2015 only warms up the bias correction
EXOG = ["t_max", "t_min", "t_mean", "feels_max", "rh_mean", "rain_mm", "cdd", "dow", "is_holiday",
        "doy_sin", "doy_cos"]

df = d.load()
y = df["energy_mu"]
X, yy = d._xy(df, "2010-01-01")
feat = d.features(df)
y_in = y.interpolate(limit_area="inside")  # inputs only; scoring always uses the real y
test_days = X.index[(X.index >= f"{FIRST}-01-01") & (X.index < f"{LAST + 1}-01-01")]


def years():
    for yr in range(FIRST, LAST + 1):
        te = test_days[(test_days >= f"{yr}-01-01") & (test_days < f"{yr + 1}-01-01")]
        if len(te):
            yield yr, te


def ridge():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xf = X.fillna(X.median())
    out = []
    for yr, te in years():
        tr = (X.index >= f"{yr - BACK}-01-01") & (X.index < f"{yr}-01-01")
        m = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(Xf[tr], (yy - X["lag_1d"])[tr])
        out.append(pd.Series(m.predict(Xf.loc[te]) + X.loc[te, "lag_1d"].to_numpy(), index=te))
    return pd.concat(out)


def nhits(seed=0):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    ex = feat[EXOG].copy()
    ex[["cdd", "rain_mm"]] = ex[["cdd", "rain_mm"]].fillna(0)
    ex = ex.interpolate(limit_direction="both")
    full = pd.DataFrame({"unique_id": "delhi", "ds": df.index, "y": y_in}).join(ex, on="ds")
    out = []
    for yr, te in years():
        start = pd.Timestamp(f"{yr - BACK}-01-01")
        part = full[(full.ds >= start) & (full.ds <= te.max())].dropna(subset=["y"])
        n_test = int((part.ds >= f"{yr}-01-01").sum())
        model = NHITS(h=1, input_size=56, futr_exog_list=EXOG, scaler_type="robust", max_steps=800,
                      learning_rate=1e-3, batch_size=64, random_seed=seed, val_check_steps=100,
                      enable_progress_bar=False, enable_model_summary=False, logger=False)
        nf = NeuralForecast(models=[model], freq="D")
        cv = nf.cross_validation(part, n_windows=n_test, step_size=1, refit=False)
        out.append(cv.set_index("ds")["NHITS"].reindex(te))
        print(f"  N-HiTS {yr}: {d.mape(y.reindex(te), out[-1]):.2f}%", flush=True)
    return pd.concat(out)


def chronos(name, covariates):
    from chronos import BaseChronosPipeline
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"  # Colab GPU if present
    pipe = BaseChronosPipeline.from_pretrained(name, device_map=dev)
    yv, idx = y_in.to_numpy(), {t: i for i, t in enumerate(df.index)}
    preds = {}
    if not covariates:  # Chronos-Bolt: energy history only
        ctx = [torch.tensor(yv[max(0, idx[t] - CTX):idx[t]], dtype=torch.float32) for t in test_days]
        for b in range(0, len(ctx), 256):
            q, mean = pipe.predict_quantiles(ctx[b:b + 256], prediction_length=1, quantile_levels=[0.5])
            for t, v in zip(test_days[b:b + 256], mean[:, 0].numpy()):
                preds[t] = float(v)
        return pd.Series(preds).reindex(test_days)
    # Chronos-2: energy history plus the same weather/calendar covariates, known for day d.
    ex = feat[EXOG].copy()
    ex[["cdd", "rain_mm"]] = ex[["cdd", "rain_mm"]].fillna(0)
    ex = ex.interpolate(limit_direction="both")
    for b in range(0, len(test_days), 200):
        hist, fut = [], []
        for t in test_days[b:b + 200]:
            i = idx[t]
            h = ex.iloc[max(0, i - CTX):i].assign(target=yv[max(0, i - CTX):i])
            hist.append(h.assign(item_id=str(t.date()), timestamp=h.index))
            fut.append(ex.iloc[i:i + 1].assign(item_id=str(t.date()), timestamp=ex.index[i:i + 1]))
        p = pipe.predict_df(pd.concat(hist, ignore_index=True), future_df=pd.concat(fut, ignore_index=True),
                            prediction_length=1, quantile_levels=[0.5])
        for _, r in p.iterrows():
            preds[pd.Timestamp(r["item_id"])] = float(r["predictions"])
    return pd.Series(preds).reindex(test_days)


def dm_test(e1, e2):
    """Diebold-Mariano test on absolute percentage errors (h=1); negative stat = model 1 better."""
    from scipy import stats
    dlt = (np.abs(e1) - np.abs(e2)).dropna()
    s = dlt.mean() / (dlt.std(ddof=1) / np.sqrt(len(dlt)))
    return float(s), float(2 * stats.norm.sf(abs(s)))


raw = {"Same as yesterday": X["lag_1d"].reindex(test_days), "Same day last week": X["lag_7d"].reindex(test_days)}
print("Ridge"); raw["Ridge (same features)"] = ridge()
print("LightGBM"); raw["LightGBM (current model)"] = d.oos_predictions(df, FIRST, LAST)
print("Chronos-Bolt"); raw["Chronos-Bolt small (zero-shot, demand only)"] = chronos("amazon/chronos-bolt-small", False)
print("Chronos-2"); raw["Chronos-2 (zero-shot, with weather)"] = chronos("amazon/chronos-2", True)
print("N-HiTS"); raw["N-HiTS (with weather)"] = nhits()

scored = test_days[test_days >= "2016-01-01"]
a = y.reindex(scored)
rows, errs = [], {}
for name, r in raw.items():
    naive = name.startswith("Same")
    p = r.reindex(scored) if naive else d.postprocess(y, r)["pred"].reindex(scored)
    e = (p - a) / a * 100
    errs[name] = e
    yr_mape = e.abs().groupby(scored.year).mean()
    rows.append({"model": name, "dev": yr_mape[yr_mape.index < 2026].mean(), "y2026": yr_mape.get(2026),
                 "raw_dev": np.nan if naive else d.mape(a[scored.year < 2026], r.reindex(scored)[scored.year < 2026]),
                 **{str(k): v for k, v in yr_mape.items()}})
res = pd.DataFrame(rows)
Path("reports").mkdir(exist_ok=True)
res.to_csv("reports/model_comparison.csv", index=False)
pd.DataFrame(errs).to_csv("reports/model_comparison_errors.csv", index_label="date", float_format="%.4f")

ref = errs["LightGBM (current model)"]
yrs = [c for c in res.columns if c.isdigit()]
lines = ["# Model comparison, 2016–2026", "",
         "Day-ahead MAPE per year. Trained models (Ridge, LightGBM, N-HiTS) are trained on the 5 years before each test "
         "year; Chronos models are zero-shot with 512 days of context. All models except the naive ones get the same bias "
         "correction. Weather for the target day is the archive record (stands in for a forecast). 2016–2025 is the "
         "development period; 2026 runs to 30 Sep. DM = Diebold–Mariano test against LightGBM on absolute percentage "
         "errors, all days 2016–2026 (negative = this model better).", "",
         "| Model | Mean 2016–2025 | 2026 | Before correction (2016–2025) | DM vs LightGBM (p) | " + " | ".join(yrs) + " |",
         "|---|---|---|---|---|" + "---|" * len(yrs)]
for _, r in res.sort_values("dev").iterrows():
    dm = "–" if r.model.startswith("LightGBM") else "{:+.1f} ({:.3f})".format(*dm_test(errs[r.model], ref))
    rawc = "–" if np.isnan(r.raw_dev) else f"{r.raw_dev:.2f}%"
    lines.append(f"| {r.model} | **{r.dev:.2f}%** | {r.y2026:.2f}% | {rawc} | {dm} | "
                 + " | ".join(f"{r[c]:.2f}" for c in yrs) + " |")
Path("reports/model_comparison.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
