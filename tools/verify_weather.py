"""Compare the dataset's weather columns with Open-Meteo's archive for Delhi.

If the columns were downloaded from Open-Meteo they should match almost exactly at some grid
point; if they were generated, they won't. Tries a few common Delhi coordinates.
"""
import pandas as pd, requests, numpy as np
CSV = "load_forecast_new/delhi_interaction_enhanced_cleaned.csv"
COLS = {"temperature_2m (°C)": "temperature_2m", "relative_humidity_2m (%)": "relative_humidity_2m",
        "cloud_cover (%)": "cloud_cover", "precipitation (mm)": "precipitation", "shortwave_radiation": "shortwave_radiation",
        "wind_speed_10m (km/h)": "wind_speed_10m"}
df = pd.read_csv(CSV, usecols=["datetime", "data_source", *COLS], parse_dates=["datetime"]).set_index("datetime")
for lat, lon, name in [(28.6139, 77.2090, "Connaught Place"), (28.7041, 77.1025, "Delhi (Google)"),
                       (28.5665, 77.1031, "Palam/IGI"), (28.65, 77.23, "Old Delhi")]:
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=120, params={
        "latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata", "start_date": "2022-07-25",
        "end_date": "2025-07-31", "hourly": ",".join(COLS.values())})
    r.raise_for_status()
    om = pd.DataFrame(r.json()["hourly"]); om["time"] = pd.to_datetime(om["time"]); om = om.set_index("time")
    print(f"\n=== {name} ({lat},{lon}) grid used: {r.json().get('latitude')},{r.json().get('longitude')}")
    for src_col, om_col in COLS.items():
        a = df[src_col]; b = om[om_col].reindex(a.index)
        m = a.notna() & b.notna()
        d = (a[m] - b[m]).abs()
        line = f"  {om_col:22s} corr={a[m].corr(b[m]):.4f} median|diff|={d.median():.3f} exact(<0.05)={(d<0.05).mean()*100:.1f}%"
        by = {s: f"{((d[df.loc[m[m].index,'data_source']==s])<0.05).mean()*100:.0f}%" for s in df.data_source.dropna().unique()}
        print(line, "exact by source:", by)
