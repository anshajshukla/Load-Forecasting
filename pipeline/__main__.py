"""CLI: python -m pipeline backtest [--target delhi] [--csv PATH] [--real data/sldc/hourly.csv] [--out reports]

Runs the leak-free walk-forward backtest at 1h and 24h (day-ahead) horizons and writes
reports/backtest.md and reports/backtest.json.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from . import TARGETS
from .legacy import LEGACY_CSV, load_legacy
from .real import WEATHER_CSV, build_real, fetch_weather
from .train import backtest, holdout, write_holdout_report, write_report


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--real-data", action="store_true",
                   help="walk-forward on the all-real dataset (--real + data/weather) instead of the legacy CSV")
    b.add_argument("--until", default="2026-01-01",
                   help="with --real-data: drop everything from this date on (the held-out test year)")
    b.add_argument("--target", default="delhi", choices=TARGETS)
    b.add_argument("--csv", default=LEGACY_CSV)
    b.add_argument("--real", help="hourly CSV from `python -m pipeline.sldc` (replaces synthetic loads)")
    b.add_argument("--real-only", action="store_true", help="with --real: drop all synthetic loads")
    b.add_argument("--out", default="reports")
    b.add_argument("--horizons", default="1,24")
    w = sub.add_parser("weather", help="download Open-Meteo archive weather (IST clock)")
    w.add_argument("--start", type=date.fromisoformat, required=True)
    w.add_argument("--end", type=date.fromisoformat, required=True)
    t = sub.add_parser("holdout", help="real data only: train before --test-from, score the year after once")
    t.add_argument("--real", default="data/sldc/hourly.csv")
    t.add_argument("--weather", default=str(WEATHER_CSV))
    t.add_argument("--test-from", default="2026-01-01")
    t.add_argument("--target", default="delhi", choices=TARGETS)
    t.add_argument("--horizons", default="1,24")
    t.add_argument("--out", default="reports")
    a = p.parse_args(argv)
    if a.cmd == "weather":
        fetch_weather(a.start, a.end)
        return
    if a.cmd == "holdout":
        df = build_real(a.real, a.weather)
        # Guard: this path must never see synthetic or legacy rows.
        assert (df["data_source"] == "sldc").all(), "non-SLDC rows in holdout data"
        res = [holdout(df, a.target, int(h), a.test_from) for h in a.horizons.split(",")]
        write_holdout_report(res, Path(a.out))
        print((Path(a.out) / "holdout.md").read_text())
        return
    if a.real_data:
        df = build_real(a.real or "data/sldc/hourly.csv")
        assert (df["data_source"] == "sldc").all(), "non-SLDC rows in real data"
        df = df[df.index < a.until]  # development never sees the test year
    else:
        df = load_legacy(a.csv, real=a.real, real_only=a.real_only)
    results = [backtest(df, a.target, int(h), groups=df["data_source"]) for h in a.horizons.split(",")]
    write_report(results, Path(a.out))
    print((Path(a.out) / "backtest.md").read_text())


if __name__ == "__main__":
    main()
