"""Scrape real Delhi load from delhisldc.org (5-minute values) and turn it into hourly data.

The site drops connections from outside India, so run this from an Indian IP:

    python -m pipeline.sldc --start 2022-07-25 --end 2025-07-31

Each day's page is cached under data/sldc/raw/ so reruns only fetch missing days, and the
parsed hourly loads go to data/sldc/hourly.csv. `load_legacy(real=...)` then replaces the
synthetic loads with these wherever a real hour exists.

The parser finds the table by its headers (TIMESLOT plus the DELHI/BRPL/... columns) instead
of a fixed element id. If a page has no such table it raises and keeps the raw HTML, so a
layout change is easy to diagnose.
"""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from html.parser import HTMLParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import pandas as pd

from . import TARGETS

URL = "https://www.delhisldc.org/Loaddata.aspx?mode={d:%d/%m/%Y}"
OUT = REPO_ROOT / "data/sldc"
# Page header -> pipeline column. NDPL is shown as TPDDL on newer pages.
HEADERS = {"DELHI": "delhi", "BRPL": "brpl", "BYPL": "bypl", "NDPL": "ndpl", "TPDDL": "ndpl",
           "NDMC": "ndmc", "MES": "mes"}


class _Tables(HTMLParser):
    """Collect every <table> as a list of rows of cell text."""

    def __init__(self):
        super().__init__()
        self.tables, self._stack, self._cell = [], [], None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._stack.append([])
        elif tag == "tr" and self._stack:
            self._stack[-1].append([])
        elif tag in ("td", "th") and self._stack and self._stack[-1]:
            self._cell = []

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self._stack[-1][-1].append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "table" and self._stack:
            self.tables.append(self._stack.pop())


def parse_day(html: str, day: date) -> pd.DataFrame:
    """Return the day's 5-minute loads (MW) indexed by timestamp, one column per target."""
    p = _Tables()
    p.feed(html)
    for rows in p.tables:
        for i, row in enumerate(rows):
            head = [c.upper() for c in row]
            if any("TIMESLOT" in c.replace(" ", "") for c in head) and "DELHI" in head:
                cols = {j: HEADERS[c] for j, c in enumerate(head) if c in HEADERS}
                recs = []
                for r in rows[i + 1:]:
                    if len(r) < len(head) or ":" not in r[0]:
                        continue
                    hh, mm = (int(x) for x in r[0].split(":")[:2])
                    ts = pd.Timestamp(day) + pd.Timedelta(hours=hh, minutes=mm)
                    rec = {"timestamp": ts}
                    for j, name in cols.items():
                        rec[name] = pd.to_numeric(r[j].replace(",", ""), errors="coerce")
                    recs.append(rec)
                if recs:
                    return pd.DataFrame(recs).set_index("timestamp").sort_index()
    raise ValueError(f"no load table found for {day}")


def to_hourly(five_min: pd.DataFrame) -> pd.DataFrame:
    """Mean over [hh:00, hh+1:00); hours with fewer than 6 of 12 readings are left out."""
    g = five_min.resample("h")
    out = g.mean()
    return out[g.count()["delhi"] >= 6]


def fetch(day: date, session, raw_dir: Path) -> str:
    path = raw_dir / f"{day:%Y-%m-%d}.html"
    if path.exists():
        return path.read_text(encoding="utf-8")
    r = session.get(URL.format(d=day), timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    path.write_text(r.text, encoding="utf-8")
    return r.text


def scrape(start: date, end: date, out: Path = OUT, delay: float = 1.0) -> pd.DataFrame:
    import requests

    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    s, frames, failed = requests.Session(), [], []
    d = start
    while d <= end:
        cached = (raw / f"{d:%Y-%m-%d}.html").exists()
        try:
            frames.append(to_hourly(parse_day(fetch(d, s, raw), d)))
        except Exception as e:  # keep going; report at the end
            failed.append((d, f"{type(e).__name__}: {e}"))
        if not cached:
            time.sleep(delay)
        d += timedelta(days=1)
    hourly = pd.concat(frames).sort_index() if frames else pd.DataFrame(columns=TARGETS)
    hourly = hourly[~hourly.index.duplicated(keep="last")]
    hourly.index.name = "datetime"
    hourly.to_csv(out / "hourly.csv")
    print(f"{len(frames)} days parsed, {len(hourly)} hours -> {out / 'hourly.csv'}")
    for d, why in failed[:20]:
        print("FAILED", d, why)
    if failed:
        print(f"{len(failed)} days failed; raw pages kept in {raw}")
    return hourly


def load_hourly(path: Path = OUT / "hourly.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="pipeline.sldc")
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, default=date.today() - timedelta(days=1))
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    a = p.parse_args(argv)
    scrape(a.start, a.end, a.out, a.delay)


if __name__ == "__main__":
    main()
