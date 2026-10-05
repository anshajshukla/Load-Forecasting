"""Scraper for Delhi SLDC load data (https://www.delhisldc.org).

`Loaddata.aspx?mode=DD/MM/YYYY` serves one day of readings (5-minute slots) per DISCOM.
Columns are matched by their header text, never by guessing from value ranges.
The site drops connections from outside India, so this must run on an Indian IP
(the `sldc` self-hosted runner in Actions).
"""
from __future__ import annotations

import re
import time
from datetime import date, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://www.delhisldc.org/Loaddata.aspx?mode={d:%d/%m/%Y}"
HEADERS = {"User-Agent": "Mozilla/5.0 (load-forecasting research; contact via GitHub anshajshuklaa/Load-Forecasting)"}

# Header text -> our column name. TPDDL was called NDPL; both map to "ndpl".
COLUMN_ALIASES = {
    "delhi": "delhi", "brpl": "brpl", "bypl": "bypl", "ndpl": "ndpl", "tpddl": "ndpl",
    "ndmc": "ndmc", "mes": "mes",
}
TIME_HEADERS = ("timeslot", "time slot", "time", "timeblock")


class ParseError(RuntimeError):
    pass


def _norm(s: str) -> str:
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def parse_day(html: bytes | str, day: date) -> pd.DataFrame:
    """Parse one Loaddata page into a DataFrame indexed by local datetime."""
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 10:
            continue
        header = [_norm(c.get_text(" ", strip=True)) for c in rows[0].find_all(["th", "td"])]
        if not header or header[0] not in TIME_HEADERS:
            continue
        cols = {i: COLUMN_ALIASES[h] for i, h in enumerate(header) if h in COLUMN_ALIASES}
        if "delhi" not in cols.values():
            continue
        recs = []
        for row in rows[1:]:
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
            if len(cells) < len(header):
                continue
            m = re.match(r"^(\d{1,2}):(\d{2})", cells[0])
            if not m:
                continue
            hh, mm = int(m.group(1)), int(m.group(2))
            ts = pd.Timestamp(day) + pd.Timedelta(hours=hh, minutes=mm)  # 24:00 rolls to next day
            rec = {"datetime": ts}
            for i, name in cols.items():
                v = cells[i].replace(",", "")
                rec[name] = float(v) if re.fullmatch(r"-?\d+(\.\d+)?", v) else float("nan")
            recs.append(rec)
        if recs:
            df = pd.DataFrame(recs).set_index("datetime").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            return df[df.index.normalize() == pd.Timestamp(day)] if len(df) > 1 else df
    raise ParseError(f"no load table with a time column and a DELHI column found for {day}")


def fetch_day(day: date, session: requests.Session | None = None, retries: int = 3) -> pd.DataFrame:
    s = session or requests.Session()
    last = None
    for attempt in range(retries):
        try:
            r = s.get(BASE.format(d=day), headers=HEADERS, timeout=45)
            r.raise_for_status()
            return parse_day(r.content, day)
        except (requests.RequestException, ParseError) as e:
            last = e
            time.sleep(2 ** attempt * 3)
    raise last


def fetch_range(start: date, end: date, pause: float = 1.5) -> tuple[pd.DataFrame, list[date]]:
    """Fetch [start, end] inclusive. Returns the data and the days that failed (to retry later)."""
    s = requests.Session()
    parts, failed = [], []
    d = start
    while d <= end:
        try:
            parts.append(fetch_day(d, s))
        except Exception as e:  # keep going; a missing day is retried on the next run
            print(f"[sldc] {d}: {type(e).__name__}: {e}")
            failed.append(d)
        time.sleep(pause)
        d += timedelta(days=1)
    return (pd.concat(parts) if parts else pd.DataFrame()), failed
