"""Feature-group ablation: does each new group of features lower the day-ahead error?

Run: python scripts/feature_ablation.py (writes reports/feature_ablation.md). Same protocol as
scripts/rolling_years.py: each year trained on the 5 years before it, bias correction applied.
Groups are judged on 2016-2025 only; 2026 is shown but not used to choose.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline.daily as d  # noqa: E402

df = d.load()
y = df["energy_mu"]
YEARS = range(2016, 2027)


def run(groups):
    d.FEATURE_GROUPS[:] = groups
    post = d.postprocess(y, d.oos_predictions(df, 2015, 2026))
    out = {}
    for yr in YEARS:
        p = post["pred"][(post.index >= f"{yr}-01-01") & (post.index < f"{yr + 1}-01-01")]
        out[yr] = d.mape(y.reindex(p.index), p)
    s = pd.Series(out)
    return s


saved = list(d.FEATURE_GROUPS)
base = run([])
rows = [("Current model", base)]
for g in d.EXTRA:
    rows.append((f"+ {g}", run([g])))
dev = [yr for yr in YEARS if yr < 2026]
keep = [g for (name, s), g in zip(rows[1:], d.EXTRA)
        if s[dev].mean() < base[dev].mean() and (s[dev] < base[dev]).sum() >= 6]
rows.append((f"+ kept groups ({', '.join(keep) or 'none'})", run(keep)))
rows.append(("+ all groups", run(list(d.EXTRA))))
d.FEATURE_GROUPS[:] = saved

lines = ["# Feature-group ablation", "",
         "Day-ahead MAPE per year; each year trained on the 5 years before it, bias correction applied. A group is kept "
         "if it lowers the 2016–2025 mean and improves at least 6 of those 10 years. 2026 (to 30 Sep) is reported, "
         "not used to choose.", "",
         "| Features | Mean 2016–2025 | Years better | 2026 | " + " | ".join(str(yr) for yr in dev) + " |",
         "|---|---|---|---|" + "---|" * len(dev)]
for name, s in rows:
    better = "–" if s is base else f"{int((s[dev] < base[dev]).sum())}/10"
    lines.append(f"| {name} | **{s[dev].mean():.3f}%** | {better} | {s[2026]:.2f}% | "
                 + " | ".join(f"{s[yr]:.2f}" for yr in dev) + " |")
Path("reports/feature_ablation.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
