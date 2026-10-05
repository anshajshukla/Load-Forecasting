"""Live forecast page: tomorrow's demand, the next 7 days, and the last 60 days of real data."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MULTIDAY = ROOT / "reports/multiday.md"
ACTUAL, FORECAST = "#2a78d6", "#eb6834"  # reference palette slots 1 and 2 (validated pair)


@st.cache_data(ttl=3600, show_spinner="Fetching the latest Grid-India data and weather forecast…")
def live_run():
    from pipeline.live import run
    return run(horizon=7)


st.title("Delhi electricity demand")
st.caption("Day-ahead forecast from real Grid-India data and the Open-Meteo weather forecast.")

try:
    L = live_run()
except Exception as e:  # noqa: BLE001
    st.error(f"Couldn't run the live forecast: {type(e).__name__}: {e}")
    st.stop()

fc, hist = L["forecast"], L["df"]["energy_mu"].dropna()
last = pd.Timestamp(fc.attrs.get("last_known", hist.index.max()))
today = pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)

bar = st.columns([6, 1], vertical_alignment="center")
live_ok = L["energy_source"].startswith("live") and L["weather_source"].startswith("live")
bar[0].caption(("🟢 Live data" if live_ok else "🟡 Using saved data (a live source is unavailable)")
               + f" · latest reported day **{last:%a %d %b}** · updated {L['fetched_at_utc']}")
if bar[1].button("Refresh", icon=":material/refresh:", width="stretch"):
    live_run.clear()
    st.rerun()

if fc.empty:
    st.warning("No weather forecast is available yet beyond the latest reported day, so there's nothing to forecast.")
    st.stop()

upcoming = fc[fc.index > today]
key = upcoming.index[0] if len(upcoming) else fc.index[-1]
d = fc.loc[key]
week_ago = key - pd.Timedelta(days=7)

# --- Headline: tomorrow --------------------------------------------------------------------------------------
with st.container(border=True):
    label = "Tomorrow" if key == today + pd.Timedelta(days=1) else "Next forecast"
    c = st.columns([1.4, 1, 1, 1])
    delta = None
    if week_ago in hist.index:
        delta = f"{(d.pred / hist[week_ago] - 1) * 100:+.1f}% vs same day last week"
    c[0].metric(f"{label} · {key:%A %d %b}", f"{d.pred:.0f} MU", delta, delta_color="off")
    c[1].metric("Likely range (80%)", f"{d.lo80:.0f}–{d.hi80:.0f} MU")
    c[2].metric("Wide range (95%)", f"{d.lo95:.0f}–{d.hi95:.0f} MU")
    c[3].metric("Max temperature", f"{d.t_max:.0f} °C")

# --- Chart: last 60 days + forecast --------------------------------------------------------------------------
recent = hist.loc[last - pd.Timedelta(days=60):]
x_band = list(fc.index) + list(fc.index[::-1])
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_band, y=list(fc.hi95) + list(fc.lo95[::-1]), fill="toself", fillcolor=FORECAST,
                         opacity=0.12, mode="lines", line=dict(width=0), hoverinfo="skip", name="95% range"))
fig.add_trace(go.Scatter(x=x_band, y=list(fc.hi80) + list(fc.lo80[::-1]), fill="toself", fillcolor=FORECAST,
                         opacity=0.22, mode="lines", line=dict(width=0), hoverinfo="skip", name="80% range"))
fig.add_trace(go.Scatter(x=recent.index, y=recent, name="Actual", line=dict(color=ACTUAL, width=2),
                         hovertemplate="%{y:.1f} MU<extra>Actual</extra>"))
fig.add_trace(go.Scatter(x=[last, *fc.index], y=[hist[last], *fc.pred], name="Forecast", mode="lines+markers",
                         line=dict(color=FORECAST, width=2, dash="dot"), marker=dict(size=8),
                         hovertemplate="%{y:.1f} MU<extra>Forecast</extra>"))
fig.add_vline(x=today, line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dash"))
fig.add_annotation(x=today, y=1, yref="paper", text="today", showarrow=False, yanchor="bottom",
                   font=dict(size=11, color="gray"))
fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                  yaxis=dict(title="MU per day", gridcolor="rgba(128,128,128,0.15)", zeroline=False),
                  xaxis=dict(showgrid=False), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# --- Next 7 days ---------------------------------------------------------------------------------------------
st.subheader("Next days", divider="gray")


def status(day: pd.Timestamp) -> str:
    if day < today:
        return "Awaiting report"
    return "Today" if day == today else "Upcoming"


table = pd.DataFrame({
    "Day": [f"{x:%a %d %b}" for x in fc.index],
    "Status": [status(x) for x in fc.index],
    "Forecast (MU)": fc["pred"].round(0).astype(int),
    "Likely range (80%)": [f"{a:.0f}–{b:.0f}" for a, b in zip(fc.lo80, fc.hi80)],
    "Max temp (°C)": fc["t_max"].round(0).astype(int),
})
st.dataframe(table, hide_index=True, width="stretch")
if (fc.index < today).any():
    st.caption(f"Grid-India reports arrive a few days late, so days after {last:%d %b} that have already passed "
               "are still forecasts.")

with st.expander("How accurate are these forecasts?"):
    st.markdown("The first day after the latest report is a true day-ahead forecast. Later days reuse earlier "
                "forecasts, so they are less certain and their ranges are wider.")
    if MULTIDAY.exists():
        st.markdown(MULTIDAY.read_text().split("\n", 4)[4])
    st.caption(f"Tested on 2025 with recorded weather. Bias correction applied today: "
               f"{fc.attrs.get('correction_pct', 0):+.2f}% (capped at ±2%). "
               f"Sources: {L['energy_source']}; {L['weather_source']}.")
