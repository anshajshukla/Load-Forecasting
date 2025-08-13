#!/bin/bash
cd load_forecast_new/delhi_forecasting_dashboard
pip install -r ../../requirements.txt
streamlit run main.py --server.port $PORT --server.address 0.0.0.0
