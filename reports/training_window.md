# Training-window test (real data only)

Each year 2019–2025 is forecast day-ahead by a model trained only on the years before it.

| Training window | Average error | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|
| 2 years | **2.79%** | 2.49% | 3.43% | 3.04% | 2.83% | 2.63% | 2.53% | 2.58% |
| 3 years | **2.72%** | 2.56% | 3.37% | 2.94% | 2.58% | 2.62% | 2.44% | 2.50% |
| 5 years | **2.67%** | 2.42% | 3.25% | 2.94% | 2.57% | 2.58% | 2.46% | 2.45% |
| 8 years | **2.68%** | 2.39% | 3.41% | 2.92% | 2.63% | 2.50% | 2.46% | 2.44% |
| All history (since 2013) | **2.69%** | 2.39% | 3.41% | 2.92% | 2.66% | 2.52% | 2.51% | 2.42% |

## Checks (all passed)

* Energy: `data/posoco/delhi_daily.csv` (sha256 `275630ce48db3dff…`), 4910 days, 2013-01-02 to 2026-10-02.
  112 days are missing at the source (all before 2023); they are skipped, never filled in.
* Weather: `data/weather/delhi_hourly.csv` (sha256 `435d867d977bf7e7…`), 120504 hours, 0 missing values.
* Every energy value the model sees equals the file value (asserted).
* Every training period ends before its test year (asserted).
* Real-event spot check: 21 Mar 2020 59.1 MU → 22 Mar 2020 (Janata curfew) 46.1 MU.
* Training is deterministic (fixed seed, single thread): reruns give identical numbers.
