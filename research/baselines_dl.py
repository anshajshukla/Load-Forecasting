"""Neural and foundation-model baselines on the common protocol (research/common.py).

    python -m research.baselines_dl [--only nhits,chronos_bolt,chronos2]

Runs on GitHub Actions (.github/workflows/paper-eval.yml): it needs PyTorch and model weights from
huggingface.co. Writes reports/paper/pred_<model>.csv.

* nhits: N-HiTS (neuralforecast), refit per test year on the 5 years before it, weather + calendar of
  day d as future covariates, 56-day input window, multi-quantile loss; one-step forecasts across the
  test year without refitting (cutoff moves one day at a time, so only energy up to d-1 is seen).
* chronos_bolt: Chronos-Bolt (base), zero-shot, energy history only (last 512 days up to d-1).
* chronos2: Chronos-2, zero-shot, energy history plus the same weather + calendar covariates.
Both need gap-free input: the source gaps (all before 2023) are interpolated for the input only
(common.gap_free) and never scored.
"""
from __future__ import annotations

import argparse

import pandas as pd

from .common import QCOLS, QUANTILES, TEST_YEARS, exog, gap_free, load_dev, save, year_split

CONTEXT = 512


def run_nhits(df):
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import NHITS

    ex = exog(df).ffill().bfill()
    cols = list(ex.columns)
    long = ex.assign(unique_id="delhi", ds=df.index, y=gap_free(df["energy_mu"]).to_numpy()).reset_index(drop=True)
    levels = [20, 40, 60, 80]
    pts, qs = [], []
    for year in TEST_YEARS:
        tr, te = year_split(df.index, year)
        part = long[tr | te].dropna(subset=["y"]).reset_index(drop=True)
        n_test = int(te.sum())
        model = NHITS(h=1, input_size=56, futr_exog_list=cols, loss=MQLoss(level=levels), scaler_type="robust",
                      max_steps=1000, learning_rate=1e-3, batch_size=64, random_seed=1, val_check_steps=100,
                      enable_progress_bar=False, logger=False)
        nf = NeuralForecast(models=[model], freq="D")
        cv = nf.cross_validation(df=part, test_size=n_test, n_windows=None, step_size=1, refit=False, val_size=0)
        cv = cv.set_index("ds")
        name = "NHITS"
        q = {"q50": cv[f"{name}-median"]}
        for lv in levels:
            lo, hi = (100 - lv) / 200, 1 - (100 - lv) / 200
            q[f"q{round(lo * 100)}"], q[f"q{round(hi * 100)}"] = cv[f"{name}-lo-{lv}"], cv[f"{name}-hi-{lv}"]
        q = pd.DataFrame(q)[QCOLS]
        pts.append(q["q50"])
        qs.append(q)
        print(f"nhits {year}: {len(q)} days")
    save("nhits", pd.concat(pts), pd.concat(qs))


def _targets(df):
    idx = df.index[(df.index >= f"{TEST_YEARS[0]}-01-01")]
    return idx[df["energy_mu"].reindex(idx).notna()]


def run_chronos_bolt(df):
    import torch
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", device_map="cpu", torch_dtype=torch.float32)
    y = gap_free(df["energy_mu"])
    days = _targets(df)
    out = []
    for i in range(0, len(days), 256):
        chunk = days[i:i + 256]
        ctx = [torch.tensor(y[:d - pd.Timedelta(days=1)].dropna().to_numpy()[-CONTEXT:], dtype=torch.float32) for d in chunk]
        q, _ = pipe.predict_quantiles(ctx, prediction_length=1, quantile_levels=QUANTILES)
        out.append(pd.DataFrame(q[:, 0, :].numpy(), index=chunk, columns=QCOLS))
    q = pd.concat(out)
    save("chronos_bolt", q["q50"], q)


def run_chronos2(df):
    from chronos import Chronos2Pipeline

    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    y = gap_free(df["energy_mu"])
    ex = exog(df).ffill().bfill()
    days = _targets(df)
    out = []
    for i in range(0, len(days), 128):
        chunk = days[i:i + 128]
        ctx, fut = [], []
        for d in chunk:
            hist = ex.loc[:d - pd.Timedelta(days=1)].iloc[-CONTEXT:]
            c = hist.assign(id=str(d.date()), timestamp=hist.index, target=y.reindex(hist.index).to_numpy())
            ctx.append(c.dropna(subset=["target"]))
            fut.append(ex.loc[[d]].assign(id=str(d.date()), timestamp=[d]))
        pred = pipe.predict_df(pd.concat(ctx, ignore_index=True), future_df=pd.concat(fut, ignore_index=True),
                               prediction_length=1, quantile_levels=QUANTILES, id_column="id",
                               timestamp_column="timestamp", target="target")
        qcols = [c for c in pred.columns if _is_float(c)]
        qcols = sorted(qcols, key=float)
        p = pred.set_index(pd.to_datetime(pred["timestamp"]))[qcols]
        p.columns = QCOLS
        out.append(p)
        print(f"chronos2 {i + len(chunk)}/{len(days)}")
    q = pd.concat(out)
    save("chronos2", q["q50"], q)


def _is_float(s) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


RUNNERS = {"nhits": run_nhits, "chronos_bolt": run_chronos_bolt, "chronos2": run_chronos2}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=",".join(RUNNERS))
    a = p.parse_args()
    df = load_dev()
    failed = []
    for name in a.only.split(","):
        try:
            RUNNERS[name](df)
        except Exception as e:  # noqa: BLE001 - report every model, then fail at the end
            import traceback
            traceback.print_exc()
            failed.append(f"{name}: {type(e).__name__}: {e}")
    if failed:
        raise SystemExit("FAILED " + " | ".join(failed))
