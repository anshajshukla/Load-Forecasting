"""Streamlit dashboard on real data only: Delhi daily energy, the 2026 held-out forecasts, live forecasts.

    streamlit run app/dashboard.py

Every number on the page is read from files that scripts in this repo write; nothing is typed in.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data/posoco/delhi_daily.csv"
HOLDOUT_JSON = ROOT / "reports/daily_holdout.json"
HOLDOUT_PRED = ROOT / "reports/daily_holdout_predictions.csv"
ROLLING = ROOT / "reports/rolling_years.md"
LIVE = ROOT / "reports/live/forecasts.csv"

st.set_page_config(page_title="Delhi Load Forecast", layout="wide")
st.title("Delhi daily electricity demand: day-ahead forecast")
st.caption("Real data only: Grid-India daily reports (energy met, MU) and Open-Meteo weather. "
           "Model trained on 2023–2025, tested on 2026.")


@st.cache_data
def read_csv(path: Path, date_col: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=[date_col]).set_index(date_col)


if not HOLDOUT_JSON.exists():
    st.error("reports/daily_holdout.json is missing. Run `python -m pipeline.daily holdout` first.")
    st.stop()

r = json.loads(HOLDOUT_JSON.read_text())
c1, c2, c3, c4 = st.columns(4)
c1.metric("2026 day-ahead MAPE", f"{r['model_mape']:.2f}%")
c2.metric("Same as yesterday", f"{r['yesterday_mape']:.2f}%")
c3.metric("Skill vs best baseline", f"{r['skill_vs_best_baseline_pct']:.0f}%")
if "intervals" in r:
    c4.metric("95% interval coverage", f"{r['intervals']['95']['coverage']:.1f}%")

tab_live, tab_test, tab_hist, tab_years = st.tabs(["Live forecasts", "2026 test", "History", "Every year"])

with tab_live:
    if LIVE.exists():
        live = read_csv(LIVE, "target_date")
        st.subheader("Forecasts made before the day (not seen during development)")
        st.dataframe(live.sort_index(ascending=False), width="stretch")
        scored = live.dropna(subset=["actual"])
        if len(scored):
            err = ((scored.pred - scored.actual).abs() / scored.actual * 100).mean()
            st.metric(f"Live MAPE over {len(scored)} scored days", f"{err:.2f}%")
    else:
        st.info("No live forecasts yet. The daily GitHub Action writes reports/live/forecasts.csv.")

with tab_test:
    if HOLDOUT_PRED.exists():
        p = read_csv(HOLDOUT_PRED, "date")
        fig = go.Figure()
        if "lo95" in p:
            fig.add_trace(go.Scatter(x=list(p.index) + list(p.index[::-1]), y=list(p.hi95) + list(p.lo95[::-1]),
                                     fill="toself", line=dict(width=0), name="95% interval", opacity=0.2))
        fig.add_trace(go.Scatter(x=p.index, y=p.actual, name="Actual", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=p.index, y=p.pred, name="Forecast", line=dict(dash="dot")))
        fig.update_layout(yaxis_title="MU per day", height=450, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
        months = pd.DataFrame(r["by_month"]).T.round(2)
        st.dataframe(months, width="stretch")

with tab_hist:
    d = read_csv(DAILY, "date")["energy_mu"]
    fig = go.Figure(go.Scatter(x=d.index, y=d, line=dict(width=1)))
    fig.update_layout(yaxis_title="MU per day", height=420, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.caption("Note the 2020 COVID lockdown dip (from 22 Mar 2020) and the summer peaks.")

with tab_years:
    if ROLLING.exists():
        st.markdown(ROLLING.read_text())
