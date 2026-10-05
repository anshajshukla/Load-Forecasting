"""Model performance page: how the model did on data it never saw."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_JSON = ROOT / "reports/daily_holdout.json"
HOLDOUT_PRED = ROOT / "reports/daily_holdout_predictions.csv"
ROLLING = ROOT / "reports/rolling_years.md"
LIVE = ROOT / "reports/live/forecasts.csv"
DAILY = ROOT / "data/posoco/delhi_daily.csv"
ACTUAL, FORECAST = "#2a78d6", "#eb6834"


def clean(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      yaxis=dict(title="MU per day", gridcolor="rgba(128,128,128,0.15)", zeroline=False),
                      xaxis=dict(showgrid=False), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


st.title("Model performance")
st.caption("Tested on 2026 (1 Jan – 30 Sep), which the model was not trained on, and on every year since 2016.")

if not HOLDOUT_JSON.exists():
    st.error("reports/daily_holdout.json is missing. Run `python -m pipeline.daily holdout`.")
    st.stop()
r = json.loads(HOLDOUT_JSON.read_text())

with st.container(border=True):
    c = st.columns(4)
    c[0].metric("Average error, 2026", f"{r['model_mape']:.2f}%")
    c[1].metric("“Same as yesterday” error", f"{r['yesterday_mape']:.2f}%")
    c[2].metric("Improvement", f"{r['skill_vs_best_baseline_pct']:.0f}%")
    if "intervals" in r:
        c[3].metric("95% range hit rate", f"{r['intervals']['95']['coverage']:.0f}%")

t_test, t_years, t_log, t_hist = st.tabs(["2026 test", "Every year", "Forecast log", "Full history"])

with t_test:
    p = pd.read_csv(HOLDOUT_PRED, parse_dates=["date"]).set_index("date")
    fig = go.Figure()
    if "lo95" in p:
        fig.add_trace(go.Scatter(x=list(p.index) + list(p.index[::-1]), y=list(p.hi95) + list(p.lo95[::-1]),
                                 fill="toself", fillcolor=FORECAST, opacity=0.15, line=dict(width=0),
                                 hoverinfo="skip", name="95% range"))
    fig.add_trace(go.Scatter(x=p.index, y=p.actual, name="Actual", line=dict(color=ACTUAL, width=2),
                             hovertemplate="%{y:.1f} MU<extra>Actual</extra>"))
    fig.add_trace(go.Scatter(x=p.index, y=p.pred, name="Forecast", line=dict(color=FORECAST, width=2, dash="dot"),
                             hovertemplate="%{y:.1f} MU<extra>Forecast</extra>"))
    st.plotly_chart(clean(fig), width="stretch", config={"displayModeBar": False})
    months = pd.DataFrame(r["by_month"]).T
    months.index = pd.PeriodIndex(months.index, freq="M").strftime("%b %Y")
    st.dataframe(months.rename(columns={"model": "Model", "yesterday": "Same as yesterday",
                                        "last_week": "Same day last week"}),
                 width="stretch", column_config={k: st.column_config.NumberColumn(k, format="%.2f%%")
                                                 for k in ["Model", "Same as yesterday", "Same day last week"]})

with t_years:
    if ROLLING.exists():
        st.markdown(ROLLING.read_text().split("\n", 2)[2])

with t_log:
    if LIVE.exists():
        live = pd.read_csv(LIVE, parse_dates=["target_date"]).set_index("target_date")
        scored = live.dropna(subset=["actual"])
        if len(scored):
            st.metric(f"Live error over {len(scored)} scored days",
                      f"{((scored.pred - scored.actual).abs() / scored.actual * 100).mean():.2f}%")
        st.dataframe(live.sort_index(ascending=False), width="stretch")
    else:
        st.info("No logged forecasts yet. The daily GitHub Action records one forecast per day before the day "
                "happens, then scores it once Grid-India reports the actual.")

with t_hist:
    d = pd.read_csv(DAILY, parse_dates=["date"]).set_index("date")["energy_mu"]
    fig = go.Figure(go.Scatter(x=d.index, y=d, line=dict(color=ACTUAL, width=1),
                               hovertemplate="%{x|%d %b %Y}: %{y:.1f} MU<extra></extra>"))
    st.plotly_chart(clean(fig, 340), width="stretch", config={"displayModeBar": False})
    st.caption("Delhi's daily energy met since 2013. Note the 2020 COVID lockdown dip and the summer peaks.")
