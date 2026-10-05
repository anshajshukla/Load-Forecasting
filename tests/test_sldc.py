"""Parser and loader tests for the SLDC scraper, on a page shaped like the live Loaddata.aspx table."""
from datetime import date

import pandas as pd
import pytest

from pipeline.legacy import load_legacy
from pipeline.sldc import parse_day, to_hourly


def page(rows):
    body = "".join(f"<tr><td>{t}</td>" + "".join(f"<td>{v}</td>" for v in vals) + "</tr>" for t, vals in rows)
    return ("<html><table><tr><td>menu</td></tr></table>"
            "<table id='ContentPlaceHolder3_DGGridAv'><tr><th>TIMESLOT</th><th>DELHI</th><th>BRPL</th>"
            "<th>BYPL</th><th>NDPL</th><th>NDMC</th><th>MES</th></tr>" + body + "</table></html>")


def five_min_day(base=4000.0):
    return [(f"{m // 60:02d}:{m % 60:02d}", (base + m, 1700, 900, 1200, 150, 30)) for m in range(0, 1440, 5)]


def test_parse_day_reads_the_load_table():
    df = parse_day(page(five_min_day()), date(2024, 6, 1))
    assert len(df) == 288
    assert df.index[0] == pd.Timestamp("2024-06-01 00:00")
    assert df.loc["2024-06-01 00:05", "delhi"] == 4005
    assert list(df.columns) == ["delhi", "brpl", "bypl", "ndpl", "ndmc", "mes"]


def test_tpddl_header_maps_to_ndpl_and_commas_are_stripped():
    html = page([("00:00", ("4,100", 1, 2, 3, 4, 5))]).replace("<th>NDPL</th>", "<th>TPDDL</th>")
    df = parse_day(html, date(2024, 6, 1))
    assert df.iloc[0]["delhi"] == 4100 and df.iloc[0]["ndpl"] == 3


def test_missing_table_raises():
    with pytest.raises(ValueError):
        parse_day("<html><table><tr><td>no data</td></tr></table></html>", date(2024, 6, 1))


def test_to_hourly_means_each_hour_and_drops_sparse_hours():
    df = parse_day(page(five_min_day()), date(2024, 6, 1))
    h = to_hourly(df)
    assert len(h) == 24
    assert h.loc["2024-06-01 00:00", "delhi"] == pytest.approx(4000 + 27.5)
    assert len(to_hourly(df.iloc[::3])) == 0  # 4 readings per hour < 6


def test_real_loads_replace_synthetic(tmp_path):
    idx = pd.date_range("2025-03-10", periods=48, freq="h")
    real = pd.DataFrame({"datetime": idx, "delhi": 1234.0, "brpl": 1.0, "bypl": 1.0, "ndpl": 1.0,
                         "ndmc": 1.0, "mes": 1.0})
    path = tmp_path / "hourly.csv"
    real.to_csv(path, index=False)
    df = load_legacy(real=str(path), real_only=True)
    assert (df.loc[idx, "delhi"] == 1234.0).all()
    assert (df.loc[idx, "data_source"] == "sldc").all()
    assert df["delhi"].notna().sum() == 48
