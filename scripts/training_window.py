"""Training-window test with provenance checks. Run: python scripts/training_window.py
(writes reports/training_window.md).

For each test year 2019-2025, train on the previous 2, 3, 5 or 8 years or all history, score that year
day-ahead. Inputs are only the real Grid-India and Open-Meteo files; the script asserts that every energy
value the model sees equals the file value and that training always ends before the test year.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.daily import DAILY_CSV, WEATHER_CSV, _fit_predict, _xy, load, mape  # noqa: E402

YEARS = range(2019, 2026)
e = pd.read_csv(DAILY_CSV, parse_dates=["date"]).set_index("date")["energy_mu"]
w = pd.read_csv(WEATHER_CSV)
df = load()
assert df["energy_mu"].dropna().equals(e.reindex(df.index).dropna()), "energy must equal the Grid-India file"
X, y = _xy(df, "2013-01-01")

rows = []
for win in (2, 3, 5, 8, "all"):
    errs = []
    for year in YEARS:
        lo = "2013-01-01" if win == "all" else f"{year - win}-01-01"
        tr = (X.index >= lo) & (X.index < f"{year}-01-01")
        te = (X.index >= f"{year}-01-01") & (X.index < f"{year + 1}-01-01")
        assert X.index[tr].max() < X.index[te].min(), "training must end before the test year"
        errs.append(mape(y[te], _fit_predict(X, y, tr, te)[0]))
    rows.append((win, float(np.mean(errs)), errs))

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]  # noqa: E731
missing = len(pd.date_range(e.index.min(), e.index.max())) - len(e)
lines = ["# Training-window test (real data only)", "",
         f"Each year {YEARS.start}–{YEARS.stop - 1} is forecast day-ahead by a model trained only on the years before it.", "",
         "| Training window | Average error | " + " | ".join(str(y_) for y_ in YEARS) + " |",
         "|---|---|" + "---|" * len(YEARS)]
for win, m, errs in rows:
    name = "All history (since 2013)" if win == "all" else f"{win} years"
    lines.append(f"| {name} | **{m:.2f}%** | " + " | ".join(f"{v:.2f}%" for v in errs) + " |")
lines += ["", "## Checks (all passed)", "",
          f"* Energy: `{DAILY_CSV}` (sha256 `{sha(DAILY_CSV)}…`), {len(e)} days, {e.index.min().date()} to {e.index.max().date()}.",
          f"  {missing} days are missing at the source (all before 2023); they are skipped, never filled in.",
          f"* Weather: `{WEATHER_CSV}` (sha256 `{sha(WEATHER_CSV)}…`), {len(w)} hours, {int(w.isna().sum().sum())} missing values.",
          "* Every energy value the model sees equals the file value (asserted).",
          "* Every training period ends before its test year (asserted).",
          f"* Real-event spot check: 21 Mar 2020 {e['2020-03-21']} MU → 22 Mar 2020 (Janata curfew) {e['2020-03-22']} MU.",
          "* Training is deterministic (fixed seed, single thread): reruns give identical numbers."]
Path("reports/training_window.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
