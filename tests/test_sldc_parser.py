from datetime import date

import pytest

from pipeline.sldc import ParseError, parse_day

HTML = """<html><body><table><tr><td>menu</td></tr></table>
<table id="grid"><tr><th>TIMESLOT</th><th>DELHI</th><th>BRPL</th><th>BYPL</th><th>NDPL</th><th>NDMC</th><th>MES</th></tr>
{rows}
</table></body></html>"""


def _page(n=12):
    rows = "".join(
        f"<tr><td>{(5 * i) // 60:02d}:{(5 * i) % 60:02d}</td><td>{4000 + i}</td><td>1800</td><td>900</td>"
        f"<td>1300</td><td>200</td><td>{'-' if i == 3 else 30}</td></tr>"
        for i in range(n))
    return HTML.format(rows=rows)


def test_parses_by_header_names():
    df = parse_day(_page(), date(2024, 6, 1))
    assert list(df.columns) == ["delhi", "brpl", "bypl", "ndpl", "ndmc", "mes"]
    assert len(df) == 12
    assert df.index[0].isoformat() == "2024-06-01T00:00:00"
    assert df["delhi"].iloc[5] == 4005
    assert df["mes"].isna().sum() == 1  # non-numeric cell becomes NaN, not a guess


def test_low_winter_values_are_kept():
    # The old fetcher dropped DELHI values outside 4000-8000 MW; winter nights are ~2000 MW.
    page = _page().replace("<td>4000</td>", "<td>1950</td>")
    assert parse_day(page, date(2024, 1, 1))["delhi"].iloc[0] == 1950


def test_missing_table_raises():
    with pytest.raises(ParseError):
        parse_day("<html><table><tr><td>x</td></tr></table></html>", date(2024, 6, 1))
