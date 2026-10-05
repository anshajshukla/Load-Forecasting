"""Score past forecasts against the actuals that have since arrived, and flag drift.

Runs every cycle. Writes reports/latest.md and appends a row to reports/history.csv.
If ANTHROPIC_API_KEY is set, Claude adds a short plain-English reading of the numbers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .train import mae, mape


def score(forecasts: pd.DataFrame, actual: pd.Series, now: pd.Timestamp) -> dict:
    f = forecasts.copy()
    f["target_time"] = pd.to_datetime(f["target_time"])
    f["actual"] = f["target_time"].map(actual)
    f = f.dropna(subset=["actual"])
    out = {"as_of": str(now), "scored_forecasts": int(len(f))}
    for days in (1, 7, 30):
        w = f[f["target_time"] > now - pd.Timedelta(days=days)]
        for h in (1, 24):
            g = w[w["horizon_h"] == h]
            if len(g):
                out[f"mape_{h}h_{days}d"] = round(mape(g.actual, g.pred), 2)
                out[f"mae_{h}h_{days}d"] = round(mae(g.actual, g.pred), 1)
    return out


def drift_flags(scores: dict, backtest: list[dict]) -> list[str]:
    flags = []
    for r in backtest:
        h = r["horizon_h"]
        live = scores.get(f"mape_{h}h_7d")
        if live is not None and live > 1.5 * r["model_mape"]:
            flags.append(f"{h}h-ahead 7-day MAPE {live:.2f}% is over 1.5x the backtest {r['model_mape']:.2f}%: retrain or inspect inputs")
    return flags


def llm_commentary(report_md: str) -> str | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic

    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=2000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low"},
        system="You are a grid-operations analyst. Read the forecast monitoring report and write 3-5 short bullet "
               "points: how accurate the forecasts were, whether accuracy is drifting, and anything an operator "
               "should check. Only use numbers that appear in the report.",
        messages=[{"role": "user", "content": report_md}],
    )
    if resp.stop_reason == "refusal":
        return None
    return "".join(b.text for b in resp.content if b.type == "text").strip() or None


def write(scores: dict, flags: list[str], latest_forecast: pd.DataFrame, out: Path, backtest_md: str = "") -> str:
    out.mkdir(parents=True, exist_ok=True)
    hist = out / "history.csv"
    row = pd.DataFrame([{**scores, "drift_flags": len(flags)}])
    row.to_csv(hist, mode="a", header=not hist.exists(), index=False)
    md = [f"# Forecast monitor ({scores['as_of']})", "",
          "## Live accuracy (forecasts scored against SLDC actuals)", "",
          "```json", json.dumps(scores, indent=2), "```", ""]
    md += ["## Drift", ""] + ([f"- {f}" for f in flags] or ["- No drift flags."]) + [""]
    if len(latest_forecast):
        md += ["## Next 24 hours (delhi, MW)", "", "| Time | Horizon | Forecast |", "|---|---|---|"]
        md += [f"| {r.target_time} | {r.horizon_h}h | {r.pred:.0f} |" for r in latest_forecast.itertuples()]
        md.append("")
    if backtest_md:
        md += ["## Latest backtest", "", backtest_md]
    text = "\n".join(md)
    note = llm_commentary(text)
    if note:
        text = text.replace("## Live accuracy", f"## Analyst summary (Claude)\n\n{note}\n\n## Live accuracy", 1)
    (out / "latest.md").write_text(text + "\n")
    return text
