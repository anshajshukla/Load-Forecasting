"""Streamlit dashboard on real data only: Delhi daily energy, the 2026 held-out forecasts, live forecasts.

    streamlit run app/dashboard.py

Every number on the page is read from files that scripts in this repo write; nothing is typed in.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `streamlit run app/dashboard.py` does not put the repo root on the import path
DAILY = ROOT / "data/posoco/delhi_daily.csv"
HOLDOUT_JSON = ROOT / "reports/daily_holdout.json"
HOLDOUT_PRED = ROOT / "reports/daily_holdout_predictions.csv"
ROLLING = ROOT / "reports/rolling_years.md"
MULTIDAY = ROOT / "reports/multiday.md"
LIVE = ROOT / "reports/live/forecasts.csv"

st.set_page_config(page_title="Delhi Load Forecast", layout="wide")
st.title("Delhi daily electricity demand: day-ahead forecast")
st.caption("Real data only: Grid-India daily reports (energy met, MU) and Open-Meteo weather. "
           "Model trained on the last 5 years (2021–2025 for the 2026 test); the live tab retrains on the latest 5 years.")


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

tab_now, tab_live, tab_test, tab_hist, tab_years = st.tabs(
    ["Live forecast", "Forecast log", "2026 test", "History", "Every year"])


@st.cache_data(ttl=3600, show_spinner="Fetching the latest Grid-India data and weather forecast, retraining...")
def live_run():
    from pipeline.live import run
    return run(horizon=7)


with tab_now:
    top = st.columns([3, 1])
    if top[1].button("Refresh now", use_container_width=False):
        live_run.clear()
    try:
        L = live_run()
    except Exception as e:  # noqa: BLE001
        st.error(f"Live run failed: {type(e).__name__}: {e}")
        st.stop()
    fc, hist = L["forecast"], L["df"]["energy_mu"].dropna()
    last = fc.attrs.get("last_known", hist.index.max())
    top[0].caption(f"Fetched {L['fetched_at_utc']} (refreshes hourly). Energy: {L['energy_source']}. "
                   f"Weather: {L['weather_source']}. Latest real day: {pd.Timestamp(last).date()}.")
    if fc.empty:
        st.warning("No weather forecast is available beyond the latest real day, so no forecast can be made.")
    else:
        today = pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)
        status = ["past: awaiting Grid-India report" if d < today else ("today" if d == today else "upcoming")
                  for d in fc.index]
        upcoming = fc[fc.index > today]
        key = upcoming.index[0] if len(upcoming) else fc.index[-1]
        d1 = fc.loc[key]
        m = st.columns(4)
        m[0].metric(f"Forecast for {key:%a %d %b}" + (" (tomorrow)" if key == today + pd.Timedelta(days=1) else ""),
                    f"{d1.pred:.1f} MU", f"{(d1.pred / hist.iloc[-1] - 1) * 100:+.1f}% vs latest real day")
        m[1].metric("80% range", f"{d1.lo80:.0f}–{d1.hi80:.0f} MU")
        m[2].metric("95% range", f"{d1.lo95:.0f}–{d1.hi95:.0f} MU")
        m[3].metric("Forecast max temperature", f"{d1.t_max:.1f} °C")
        recent = hist.loc[pd.Timestamp(last) - pd.Timedelta(days=90):]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(fc.index) + list(fc.index[::-1]), y=list(fc.hi95) + list(fc.lo95[::-1]),
                                 fill="toself", line=dict(width=0), name="95% range", opacity=0.2))
        fig.add_trace(go.Scatter(x=recent.index, y=recent, name="Actual (Grid-India)", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=fc.index, y=fc.pred, name="Forecast", mode="lines+markers", line=dict(dash="dot")))
        fig.update_layout(yaxis_title="MU per day", height=430, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
        show = fc[["days_ahead", "pred", "lo80", "hi80", "lo95", "hi95", "t_max"]].round(1)
        show.insert(0, "status", status)
        show.index = show.index.strftime("%a %d %b %Y")
        st.dataframe(show.rename(columns={"days_ahead": "days ahead", "pred": "forecast (MU)", "t_max": "max temp °C"}),
                     width="stretch")
        if len(upcoming) < len(fc):
            st.info(f"Grid-India's daily reports arrive with a delay: the latest real day is {pd.Timestamp(last):%d %b}, "
                    "so the first rows are days that have passed but are not reported yet.")
        if MULTIDAY.exists():
            with st.expander("How accurate is each day? (tested on 2025)"):
                st.markdown(MULTIDAY.read_text().split("\n", 2)[2])
                st.caption("Tested with recorded weather; real forecasts further out also carry weather-forecast error.")
        st.caption("The first day after the latest real day is a true day-ahead forecast (about 2.3% error in 2025). Later days use earlier forecasts as "
                   "'yesterday' and a weather forecast further out, so they are less certain; their ranges are widened. "
                   f"Bias correction applied: {fc.attrs.get('correction_pct', 0):+.2f}% (capped at ±2%).")

with tab_live:
    if LIVE.exists():
        live = read_csv(LIVE, "target_date")
        st.subheader("Daily forecasts logged before the day by the GitHub Action")
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
