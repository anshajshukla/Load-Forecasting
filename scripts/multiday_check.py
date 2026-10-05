"""How accurate are the live app's multi-day forecasts? Run: python scripts/multiday_check.py
(writes reports/multiday.md).

From every Monday of 2025, pretend that was the latest real day and forecast the next 7 days exactly as the live
app does (pipeline.live.forecast_days), then compare with what Grid-India reported. 2026 is not used.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.daily import load  # noqa: E402
from pipeline.live import forecast_days  # noqa: E402

df = load()
rows = []
for origin in pd.date_range("2025-01-06", "2025-12-22", freq="7D"):
    fc = forecast_days(df.loc[: origin + pd.Timedelta(days=7)].assign(
        energy_mu=lambda d: d["energy_mu"].where(d.index <= origin)), horizon=7)
    a = df["energy_mu"].reindex(fc.index)
    for d, r in fc.iterrows():
        if pd.notna(a[d]):
            rows.append({"days_ahead": r.days_ahead, "ape": abs(r.pred - a[d]) / a[d] * 100,
                         "in95": r.lo95 <= a[d] <= r.hi95})
r = pd.DataFrame(rows).groupby("days_ahead").agg(mape=("ape", "mean"), coverage95=("in95", "mean"), n=("ape", "size"))
lines = ["# Multi-day forecast accuracy (live app method, 2025)", "",
         "From every Monday of 2025, forecast the next 7 days the way the live app does and compare with Grid-India.", "",
         "| Days ahead | MAPE | 95% range covered | Forecasts |", "|---|---|---|---|"]
lines += [f"| {int(h)} | {x.mape:.2f}% | {x.coverage95 * 100:.0f}% | {int(x.n)} |" for h, x in r.iterrows()]
Path("reports/multiday.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
