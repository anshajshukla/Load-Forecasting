"""CLI: python -m pipeline backtest [--target delhi] [--csv PATH] [--real data/sldc/hourly.csv] [--out reports]

Runs the leak-free walk-forward backtest at 1h and 24h (day-ahead) horizons and writes
reports/backtest.md and reports/backtest.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import TARGETS
from .legacy import LEGACY_CSV, load_legacy
from .train import backtest, write_report


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--target", default="delhi", choices=TARGETS)
    b.add_argument("--csv", default=LEGACY_CSV)
    b.add_argument("--real", help="hourly CSV from `python -m pipeline.sldc` (replaces synthetic loads)")
    b.add_argument("--real-only", action="store_true", help="with --real: drop all synthetic loads")
    b.add_argument("--out", default="reports")
    b.add_argument("--horizons", default="1,24")
    a = p.parse_args(argv)
    df = load_legacy(a.csv, real=a.real, real_only=a.real_only)
    results = [backtest(df, a.target, int(h), groups=df["data_source"]) for h in a.horizons.split(",")]
    write_report(results, Path(a.out))
    print((Path(a.out) / "backtest.md").read_text())


if __name__ == "__main__":
    main()
