"""Delhi electricity demand dashboard (entry point).

    streamlit run app/dashboard.py

Three pages: the live forecast, the model's tested performance, and how it works. Every number is read from
live sources or from files that scripts in this repo write; nothing is typed in.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `pipeline`

st.set_page_config(page_title="Delhi Load Forecast", page_icon="⚡", layout="wide")
st.navigation([
    st.Page("views/live.py", title="Live forecast", icon=":material/bolt:", default=True),
    st.Page("views/performance.py", title="Model performance", icon=":material/insights:"),
    st.Page("views/how_it_works.py", title="How it works", icon=":material/menu_book:"),
]).run()
