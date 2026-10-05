"""Plain-CSV storage under a data directory (the `data` branch in Actions)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import TARGETS


def _read(path: Path, idx: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=[idx]).set_index(idx).sort_index()


def _merge_write(path: Path, new: pd.DataFrame, idx: str) -> pd.DataFrame:
    old = _read(path, idx)
    both = pd.concat([old, new]) if len(old) else new
    both = both[~both.index.duplicated(keep="last")].sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    both.rename_axis(idx).to_csv(path)
    return both


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.raw = self.root / "sldc" / "raw"          # native SLDC resolution, one file per month
        self.hourly_path = self.root / "sldc" / "hourly.csv"
        self.weather_path = self.root / "weather" / "hourly.csv"
        self.forecast_log = self.root / "forecasts" / "log.csv"
        self.reports = self.root / "reports"

    def add_raw(self, df: pd.DataFrame) -> None:
        for month, part in df.groupby(df.index.to_period("M")):
            _merge_write(self.raw / f"{month}.csv", part, "datetime")
        self.rebuild_hourly()

    def rebuild_hourly(self) -> pd.DataFrame:
        files = sorted(self.raw.glob("*.csv"))
        if not files:
            return pd.DataFrame()
        raw = pd.concat(_read(f, "datetime") for f in files)
        cols = [c for c in TARGETS if c in raw]
        # Readings in [h:00, h+1:00) are averaged and labelled h:00.
        hourly = raw[cols].resample("h").mean()
        hourly["n_readings"] = raw[cols[0]].resample("h").count()
        self.hourly_path.parent.mkdir(parents=True, exist_ok=True)
        hourly.rename_axis("datetime").to_csv(self.hourly_path)
        return hourly

    def hourly(self) -> pd.DataFrame:
        return _read(self.hourly_path, "datetime")

    def add_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        return _merge_write(self.weather_path, df.rename_axis("datetime"), "datetime")

    def weather(self) -> pd.DataFrame:
        return _read(self.weather_path, "datetime")

    def log_forecasts(self, fc: pd.DataFrame) -> None:
        self.forecast_log.parent.mkdir(parents=True, exist_ok=True)
        fc.to_csv(self.forecast_log, mode="a", header=not self.forecast_log.exists(), index=False)

    def forecasts(self) -> pd.DataFrame:
        if not self.forecast_log.exists():
            return pd.DataFrame(columns=["target_time", "horizon_h", "origin", "pred"])
        return pd.read_csv(self.forecast_log)

    def frame(self) -> pd.DataFrame:
        """Hourly load joined with weather on a gap-free hourly index."""
        load = self.hourly()
        if load.empty:
            return load
        w = self.weather()
        idx = pd.date_range(load.index.min(), max(load.index.max(), w.index.max() if len(w) else load.index.max()), freq="h")
        out = load.reindex(idx)
        # Treat thinly covered hours as missing rather than trusting 1-2 readings.
        if "n_readings" in out:
            thin = out["n_readings"] < out["n_readings"].median() / 2
            out.loc[thin, [c for c in out if c != "n_readings"]] = float("nan")
        if len(w):
            out = out.join(w.reindex(idx))
        return out.rename_axis("datetime")
