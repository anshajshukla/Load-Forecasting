"""Second page: what the dashboard uses, which models, how they were tested, and the limits."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.daily import CORR_ALPHA, CORR_CAP, CORR_WINDOW, INTERVAL_WINDOW, PARAMS, YEARS_BACK  # noqa: E402

st.set_page_config(page_title="How it works", layout="wide")
st.title("How it works")
st.caption("Everything below is read from the repo: code settings, generated reports and the learning notes.")


def read(rel: str) -> str:
    p = ROOT / rel
    return p.read_text() if p.exists() else f"_{rel} not found_"


def section(md: str, start: str, stop: str | None = None) -> str:
    """Slice a markdown file from a heading to the next given heading."""
    i = md.find(start)
    if i < 0:
        return ""
    j = md.find(stop, i + len(start)) if stop else -1
    return md[i:j] if j > 0 else md[i:]


holdout = json.loads(read("reports/daily_holdout.json")) if (ROOT / "reports/daily_holdout.json").exists() else {}

st.header("1. Data (real only)")
st.markdown("""
| Input | Source | Coverage |
|---|---|---|
| Delhi daily energy met (MU) | [Grid-India daily reports](https://posoco.in/reports/daily-reports/), via [Robbie Andrew's dataset](https://robbieandrew.github.io/india/) | 2013 → latest report (downloaded live) |
| Hourly weather (temperature, humidity, dew point, cloud, radiation, wind, rain) | [Open-Meteo](https://open-meteo.com/) archive + forecast, Delhi, IST | 2013 → 7 days ahead (forecast downloaded live) |
| Holidays | `holidays` package, India / Delhi | All years |

**Not used:** the original project's dataset, which is about 95% synthetic. Nothing is simulated or filled in:
the 112 days missing at the source (all before 2023) are skipped. Verified against events a generator wouldn't
know: the Janata curfew (22 Mar 2020: 59.1 → 46.1 MU), the lockdown, Holi and Diwali dates, rain dips, and the
June 2024 record.
""")

st.header("2. The model in use")
c = st.columns(3)
c[0].metric("Algorithm", "LightGBM")
c[1].metric("Trees / leaves", f"{PARAMS['n_estimators']} / {PARAMS['num_leaves']}")
c[2].metric("Training window", f"last {YEARS_BACK} years")
st.markdown(f"""
* **Target:** the change in daily energy from yesterday (tree models can't predict above the highest value they've seen).
* **Features for day d:** energy up to day d−1 only (lags of 1, 2, 3, 7, 14 and 364 days, 7/28-day means, 7-day trend,
  growth vs a year ago, weekday ratios); weather for day d and the days before (max/min/mean temperature, feels-like,
  humidity, dew point, rain, cloud, radiation, wind, cooling/heating degrees, 3- and 7-day heat build-up);
  calendar (weekday, season, holiday, days to/since a holiday).
* **Bias correction:** forecast × (1 + {CORR_ALPHA} × mean error of the last {CORR_WINDOW} days), capped at ±{CORR_CAP:.0%}.
* **Ranges:** 80% and 95% from the last {INTERVAL_WINDOW} days' errors; widened by √(days ahead) beyond day 1.
* **Live tab:** retrains on the latest data each refresh; day 1 is a true day-ahead forecast, later days feed
  earlier forecasts back in.
""")

st.header("3. Results")
if holdout:
    c = st.columns(4)
    c[0].metric("2026 held-out error", f"{holdout['model_mape']:.2f}%")
    c[1].metric("Same as yesterday", f"{holdout['yesterday_mape']:.2f}%")
    c[2].metric("Better than baseline by", f"{holdout['skill_vs_best_baseline_pct']:.0f}%")
    c[3].metric("95% range covered", f"{holdout['intervals']['95']['coverage']:.1f}%")
tabs = st.tabs(["Every year 2016–2026", "By days ahead (live method)", "Training window", "Overfitting checks"])
with tabs[0]:
    st.markdown(read("reports/rolling_years.md").split("\n", 2)[2])
with tabs[1]:
    st.markdown(read("reports/multiday.md").split("\n", 2)[2])
with tabs[2]:
    st.markdown(read("reports/training_window.md").split("\n", 2)[2])
with tabs[3]:
    st.markdown(read("reports/overfit_check.md").split("\n", 2)[2])

st.header("4. Every model used in this project")
models = read("learning/models.md")
st.markdown(section(models, "## 1. The original project", "## Why LightGBM"))
with st.expander("Why LightGBM, and what was not tried"):
    st.markdown(section(models, "## Why LightGBM"))

st.header("5. How it got here")
progress = pd.DataFrame({
    "Stage": ["Original project", "First honest model (v1)", "v1 smaller", "v2 features, 5 years", "v2 + bias correction (current)"],
    "2026 error": ["'1.0%' / '4.09%' (leaked / hard-coded)", "2.82%", "2.73%", "2.57%", "2.45%"],
})
st.dataframe(progress, hide_index=True, width="stretch")
with st.expander("Full learning log (every change, its impact and the lesson)"):
    st.markdown(read("learning/changes.md"))

st.header("6. Limits")
st.markdown(section(read("learning/gaps.md"), "## Still open"))
st.caption("Source code: github.com/anshajshuklaa/Load-Forecasting · Disclosure: 2026 was scored four times as the "
           "model evolved; each choice was made on development data first. The daily forecast log is the clean test.")
