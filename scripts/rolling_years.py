"""Rolling-origin test over many years: for each year Y, train on the 5 years before Y and score Y day-ahead.

Run: python scripts/rolling_years.py (writes reports/rolling_years.md). Same model and features as pipeline.daily.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.daily import _xy, load, mape, oos_predictions, postprocess  # noqa: E402

df = load()
X, y = _xy(df, "2013-01-01")
raw_all = oos_predictions(df, 2015, 2026)  # 2015 only warms up the correction and intervals for 2016
post = postprocess(df["energy_mu"], raw_all)
rows = []
for year in range(2016, 2027):
    te = (X.index >= f"{year}-01-01") & (X.index < f"{year + 1}-01-01")
    a = y[te]
    p, raw = post["pred"].reindex(a.index), raw_all.reindex(a.index)
    cov = ((a >= post["lo95"].reindex(a.index)) & (a <= post["hi95"].reindex(a.index))).mean() * 100
    rows.append({"year": year, "days": int(te.sum()), "model": mape(a, p), "raw": mape(a, raw),
                 "yesterday": mape(a, X.loc[te, "lag_1d"]), "last_week": mape(a, X.loc[te, "lag_7d"]),
                 "bias": float(((p - a) / a).mean() * 100), "cov95": float(cov)})
r = pd.DataFrame(rows)
lines = ["# Rolling-origin test, 2016–2026", "",
         "For each year: train on the previous 5 years only, score every day of that year day-ahead. Same model, features, "
         "settings and bias correction as `pipeline.daily`; real Grid-India data and Open-Meteo archive weather. 2026 runs to 30 Sep.", "",
         "| Year | Days | Model | Before bias correction | Same as yesterday | Same day last week | Bias | 95% coverage |",
         "|---|---|---|---|---|---|---|---|"]
for x in rows:
    lines.append(f"| {x['year']} | {x['days']} | **{x['model']:.2f}%** | {x['raw']:.2f}% | {x['yesterday']:.2f}% | "
                 f"{x['last_week']:.2f}% | {x['bias']:+.2f}% | {x['cov95']:.0f}% |")
lines += ["", f"Median model MAPE {r.model.median():.2f}% (range {r.model.min():.2f}–{r.model.max():.2f}%); the model beats "
          f"same-as-yesterday in {int((r.model < r.yesterday).sum())} of {len(r)} years."]
Path("reports/rolling_years.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
