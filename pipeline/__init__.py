"""Leak-free Delhi load forecasting pipeline: scrape SLDC, add weather, train, forecast, analyse."""

TARGETS = ["delhi", "brpl", "bypl", "ndpl", "ndmc", "mes"]
LAT, LON, TZ = 28.6139, 77.2090, "Asia/Kolkata"
