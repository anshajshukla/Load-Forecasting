"""Rolling-origin test over many years: for each year Y, train on the 3 years before Y and score Y day-ahead.

Run: python scripts/rolling_years.py (writes reports/rolling_years.md). Same model and features as pipeline.daily.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.daily import _fit_predict, _xy, load, mape  # noqa: E402

df = load()
X, y = _xy(df, "2013-01-01")
rows = []
for year in range(2016, 2027):
    tr = (X.index >= f"{year - 3}-01-01") & (X.index < f"{year}-01-01")
    te = (X.index >= f"{year}-01-01") & (X.index < f"{year + 1}-01-01")
    p, _ = _fit_predict(X, y, tr, te)
    a = y[te]
    rows.append({"year": year, "days": int(te.sum()), "model": mape(a, p),
                 "yesterday": mape(a, X.loc[te, "lag_1d"]), "last_week": mape(a, X.loc[te, "lag_7d"]),
                 "bias": float(((p - a) / a).mean() * 100)})
r = pd.DataFrame(rows)
lines = ["# Rolling-origin test, 2016–2026", "",
         "For each year: train on the previous 3 years only, score every day of that year day-ahead. Same model, features "
         "and settings as `pipeline.daily`; real Grid-India data and Open-Meteo archive weather. 2026 runs to 30 Sep.", "",
         "| Year | Days | Model | Same as yesterday | Same day last week | Bias |", "|---|---|---|---|---|---|"]
for x in rows:
    lines.append(f"| {x['year']} | {x['days']} | **{x['model']:.2f}%** | {x['yesterday']:.2f}% | {x['last_week']:.2f}% | {x['bias']:+.2f}% |")
lines += ["", f"Median model MAPE {r.model.median():.2f}% (range {r.model.min():.2f}–{r.model.max():.2f}%); the model beats "
          f"same-as-yesterday in {int((r.model < r.yesterday).sum())} of {len(r)} years."]
Path("reports/rolling_years.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
