# Delhi daily energy met (real)

`delhi_daily.csv`: Delhi's daily energy met in MU (GWh), 2013-01-02 to 2026-10-02.

* Source: Grid-India (formerly POSOCO) daily PSP reports, https://posoco.in/reports/daily-reports/
* Extracted from Robbie Andrew's processed dataset (`india/data/POSOCO_data.zip` in
  https://github.com/robbieandrew/robbieandrew.github.io, Table A/G scraped from the PDFs, no further processing).
  Please cite both Grid-India and Robbie Andrew (https://robbieandrew.github.io/india/) when using it.
* Checked against the legacy `historical` hourly rows (44 overlapping days): correlation 0.998, ratio 0.994.
