"""Score every model in reports/paper/pred_*.csv on the same days, with significance tests.

    python -m research.evaluate          # writes reports/paper/results.md and results.json

* Point accuracy: MAPE, RMSE (MU) per test year and pooled 2017-2025.
* Probabilistic: mean pinball loss over the 9 quantiles 10%..90% (x2 approximates CRPS, in MU),
  and the coverage of the 10-90% interval (nominal 80%).
* Significance: Diebold-Mariano test of each model against LightGBM on daily absolute percentage
  error and squared error, Newey-West variance (lag 7) and the Harvey-Leybourne-Newbold correction.
Only days with a reported actual and a forecast from every compared model are scored.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from .common import DEV_END, OUT, QCOLS, QUANTILES, TEST_YEARS, load_dev

REF = "lightgbm"
ORDER = ["lightgbm", "lightgbm_raw", "ridge", "sarimax", "nhits", "chronos2", "chronos_bolt", "prophet",
         "yesterday", "last_week"]
LABEL = {"lightgbm": "LightGBM (proposed, bias-corrected)", "lightgbm_raw": "LightGBM, no bias correction",
         "ridge": "Ridge, same features", "sarimax": "SARIMAX + weather", "nhits": "N-HiTS + weather",
         "chronos2": "Chronos-2 + weather (zero-shot)", "chronos_bolt": "Chronos-Bolt (zero-shot, no weather)",
         "prophet": "Prophet + weather", "yesterday": "Same as yesterday", "last_week": "Same day last week"}


def load_preds() -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(OUT.glob("pred_*.csv")):
        d = pd.read_csv(f, parse_dates=["date"]).set_index("date")
        assert d.index.max() < pd.Timestamp(DEV_END), f"{f.name} contains 2026"
        out[f.stem[5:]] = d
    return dict(sorted(out.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99))


def pinball(actual: pd.Series, q: pd.DataFrame) -> pd.Series:
    loss = []
    for c, tau in zip(QCOLS, QUANTILES):
        u = actual - q[c]
        loss.append(np.maximum(tau * u, (tau - 1) * u))
    return pd.concat(loss, axis=1).mean(axis=1)


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, lag: int = 7) -> tuple[float, float]:
    """DM statistic and two-sided p-value for H0: equal expected loss (negative stat: a is better)."""
    d = loss_a - loss_b
    T = len(d)
    dc = d - d.mean()
    lrv = dc @ dc / T
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / T
    stat = d.mean() / np.sqrt(lrv / T)
    h = 1
    stat *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)  # Harvey, Leybourne and Newbold (1997)
    return float(stat), float(2 * stats.t.sf(abs(stat), df=T - 1))


def main() -> dict:
    df = load_dev()
    actual = df["energy_mu"]
    preds = load_preds()
    days = actual.dropna().index
    days = days[(days >= f"{TEST_YEARS[0]}-01-01")]
    for d in preds.values():
        days = days.intersection(d["pred"].dropna().index)
    a = actual.reindex(days)
    res = {"days": len(days), "period": [str(days.min().date()), str(days.max().date())], "models": {}}
    ape = {m: (np.abs(d.loc[days, "pred"] - a) / a * 100) for m, d in preds.items()}
    se = {m: (d.loc[days, "pred"] - a) ** 2 for m, d in preds.items()}
    for m, d in preds.items():
        r = {"mape": float(ape[m].mean()), "rmse_mu": float(np.sqrt(se[m].mean())),
             "by_year": {int(y): float(g.mean()) for y, g in ape[m].groupby(ape[m].index.year)}}
        if set(QCOLS) <= set(d.columns) and d.loc[days, QCOLS].notna().all(axis=None):
            q = d.loc[days, QCOLS]
            pl = pinball(a, q)
            r["pinball_mu"] = float(pl.mean())
            r["crps_approx_mu"] = float(2 * pl.mean())
            r["cov80"] = float(((a >= q["q10"]) & (a <= q["q90"])).mean() * 100)
        if m != REF:
            r["dm_ape"] = dm_test(ape[REF].to_numpy(), ape[m].to_numpy())
            r["dm_se"] = dm_test(se[REF].to_numpy(), se[m].to_numpy())
            yr = {}
            for y in TEST_YEARS:
                sel = ape[REF].index.year == y
                if sel.sum() > 30:
                    yr[y] = dm_test(ape[REF].to_numpy()[sel], ape[m].to_numpy()[sel])
            r["dm_ape_by_year"] = yr
        res["models"][m] = r

    L = ["# Model comparison, 2017–2025 (development years only)", "",
         f"{res['days']} days scored ({res['period'][0]} to {res['period'][1]}), the same days for every model. Each test year is "
         "forecast one day ahead by a model trained on the 5 years before it, with recorded weather for the target day. "
         "The 2026 holdout is not used. Written by `python -m research.evaluate`.", "",
         "## Accuracy and significance against LightGBM", "",
         "| Model | MAPE | RMSE (MU) | CRPS≈ (MU) | 80% interval coverage | DM vs LightGBM (APE) | DM (squared error) |",
         "|---|---|---|---|---|---|---|"]
    for m, r in res["models"].items():
        crps = f"{r['crps_approx_mu']:.3f}" if "crps_approx_mu" in r else "–"
        cov = f"{r['cov80']:.1f}%" if "cov80" in r else "–"
        dm = lambda t: f"{t[0]:+.2f} (p={t[1]:.3g})"
        L.append(f"| {LABEL.get(m, m)} | {r['mape']:.2f}% | {r['rmse_mu']:.2f} | {crps} | {cov} | "
                 f"{dm(r['dm_ape']) if 'dm_ape' in r else 'reference'} | {dm(r['dm_se']) if 'dm_se' in r else 'reference'} |")
    L += ["", "DM statistic: LightGBM's loss minus the other model's; negative means LightGBM is more accurate. "
          "Newey–West variance (lag 7), Harvey–Leybourne–Newbold correction, two-sided p-value. "
          "CRPS≈ is twice the mean pinball loss over the 10%, 20%, …, 90% quantiles. Naive, ridge and LightGBM quantiles "
          "come from each model's last 365 days of errors (known by the day before); SARIMAX's from its Gaussian "
          "forecast in logs; Prophet's from its posterior samples; N-HiTS from a multi-quantile loss; Chronos's from the model.", "",
          "## MAPE by year", "", "| Model | " + " | ".join(map(str, TEST_YEARS)) + " |", "|---|" + "---|" * len(TEST_YEARS)]
    for m, r in res["models"].items():
        L.append(f"| {LABEL.get(m, m)} | " + " | ".join(f"{r['by_year'].get(y, float('nan')):.2f}" for y in TEST_YEARS) + " |")
    L += ["", "## Years in which LightGBM is significantly better (DM on APE, p < 0.05)", "",
          "| Model | Better | Not different | Worse |", "|---|---|---|---|"]
    for m, r in res["models"].items():
        if "dm_ape_by_year" not in r:
            continue
        v = r["dm_ape_by_year"].values()
        L.append(f"| {LABEL.get(m, m)} | {sum(s < 0 and p < .05 for s, p in v)} | {sum(p >= .05 for s, p in v)} | "
                 f"{sum(s > 0 and p < .05 for s, p in v)} |")
    (OUT / "results.md").write_text("\n".join(L) + "\n")
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print("\n".join(L))
    return res


if __name__ == "__main__":
    main()
