"""CLI: python -m pipeline <command> --data DIR

  scrape    fetch recent SLDC days (default: yesterday and today) plus any recorded gaps
  backfill  fetch a historical range from SLDC
  weather   refresh Open-Meteo archive + forecast for the stored range
  forecast  train per-horizon models on all data and log the next 24h forecast
  backtest  walk-forward evaluation vs persistence and seasonal-naive baselines
  analyze   score logged forecasts against actuals, flag drift, write reports/latest.md
  seed      import the legacy CSV's delhi/DISCOM columns as raw data (to bootstrap before backfill)
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from . import TARGETS, TZ
from .store import Store


def _today() -> date:
    return pd.Timestamp.now(tz=TZ).date()


def cmd_scrape(st: Store, a) -> None:
    from .sldc import fetch_range

    gaps_file = st.root / "sldc" / "failed_days.json"
    gaps = [date.fromisoformat(d) for d in json.loads(gaps_file.read_text())] if gaps_file.exists() else []
    end = _today()
    start = end - timedelta(days=a.days - 1)
    df, failed = fetch_range(start, end)
    for d in sorted(set(gaps) - set(failed))[: a.max_gap_days]:
        part, f2 = fetch_range(d, d)
        df = pd.concat([df, part]) if len(part) else df
        failed += f2
    if len(df):
        st.add_raw(df)
    gaps_file.parent.mkdir(parents=True, exist_ok=True)
    gaps_file.write_text(json.dumps(sorted({d.isoformat() for d in failed})))
    print(f"[scrape] {len(df)} readings, {len(failed)} failed days")


def cmd_backfill(st: Store, a) -> None:
    from .sldc import fetch_range

    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    chunk = timedelta(days=30)
    while start <= end:
        e = min(start + chunk - timedelta(days=1), end)
        df, failed = fetch_range(start, e)
        if len(df):
            st.add_raw(df)
        print(f"[backfill] {start}..{e}: {len(df)} readings, failed {[d.isoformat() for d in failed]}")
        start = e + timedelta(days=1)


def cmd_weather(st: Store, a) -> None:
    from . import weather

    load = st.hourly()
    if load.empty:
        raise SystemExit("no load data yet")
    have = st.weather()
    start = (have.index.max() - pd.Timedelta(days=3)).date() if len(have) else load.index.min().date()
    arch_end = _today() - timedelta(days=6)  # archive lags a few days; forecast API covers the rest
    if start <= arch_end:
        st.add_weather(weather.archive(start.isoformat(), arch_end.isoformat()))
    st.add_weather(weather.forecast(days=3, past_days=7))
    print(f"[weather] rows: {len(st.weather())}")


def cmd_forecast(st: Store, a) -> None:
    from .train import forecast_next

    df = st.frame()
    fc = forecast_next(df, a.target, horizons=range(1, 25))
    fc["target"] = a.target
    fc["made_at"] = pd.Timestamp.now(tz=TZ).tz_localize(None).floor("min")
    st.log_forecasts(fc)
    print(fc.to_string(index=False))


def cmd_backtest(st: Store, a) -> None:
    from .train import backtest, write_report

    df = st.frame()
    results = [backtest(df, a.target, h) for h in (1, 24)]
    write_report(results, st.reports)
    print((st.reports / "backtest.md").read_text())


def cmd_analyze(st: Store, a) -> None:
    from . import analyze

    df = st.frame()
    fc = st.forecasts()
    fc = fc[fc.get("target", a.target) == a.target] if len(fc) else fc
    now = df[a.target].last_valid_index()
    scores = analyze.score(fc, df[a.target], now)
    bt_file = st.reports / "backtest.json"
    bt = json.loads(bt_file.read_text()) if bt_file.exists() else []
    flags = analyze.drift_flags(scores, bt)
    latest = fc[fc["made_at"] == fc["made_at"].max()] if len(fc) else fc
    bt_md = (st.reports / "backtest.md").read_text() if (st.reports / "backtest.md").exists() else ""
    print(analyze.write(scores, flags, latest, st.reports, bt_md))


def cmd_seed(st: Store, a) -> None:
    cols = {f"{t}_load": t for t in TARGETS}
    df = pd.read_csv(a.csv, usecols=["datetime", *cols], parse_dates=["datetime"]).rename(columns=cols)
    st.add_raw(df.set_index("datetime"))
    print(f"[seed] {len(df)} rows from {a.csv}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="pipeline")
    p.add_argument("--data", default="store", help="data directory (the checked-out data branch)")
    p.add_argument("--target", default="delhi", choices=TARGETS)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scrape"); s.add_argument("--days", type=int, default=2); s.add_argument("--max-gap-days", type=int, default=10)
    b = sub.add_parser("backfill"); b.add_argument("--start", required=True); b.add_argument("--end", required=True)
    sub.add_parser("weather"); sub.add_parser("forecast"); sub.add_parser("backtest"); sub.add_parser("analyze")
    sd = sub.add_parser("seed"); sd.add_argument("--csv", required=True)
    a = p.parse_args(argv)
    st = Store(a.data)
    globals()[f"cmd_{a.cmd}"](st, a)


if __name__ == "__main__":
    main()
