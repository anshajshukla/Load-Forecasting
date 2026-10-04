"""One-off probe: print the structure of Delhi SLDC load pages so the scraper can be written against real HTML."""
import re, sys, requests
from bs4 import BeautifulSoup

URLS = [
    "https://www.delhisldc.org/Redirect.aspx?Loc=0804",
    "https://www.delhisldc.org/Redirect.aspx?Loc=0805",
    "https://www.delhisldc.org/Loaddata.aspx?mode=01/06/2024",
    "https://www.delhisldc.org/Loaddata.aspx?mode=15/01/2023",
]
H = {"User-Agent": "Mozilla/5.0 (load-forecasting research scraper)"}
for url in URLS:
    print("=" * 100, "\nURL:", url)
    try:
        r = requests.get(url, headers=H, timeout=60, allow_redirects=True)
    except Exception as e:
        print("ERROR", repr(e)); continue
    print("status", r.status_code, "final", r.url, "bytes", len(r.content))
    soup = BeautifulSoup(r.content, "html.parser")
    print("title:", soup.title.get_text(strip=True) if soup.title else None)
    for f in soup.find_all(["iframe", "frame"]):
        print("FRAME src:", f.get("src"))
    for a in soup.find_all("a", href=True)[:0]:
        pass
    for inp in soup.find_all(["input", "select"])[:30]:
        print("FORM", inp.name, inp.get("id"), inp.get("name"), inp.get("type"), (inp.get("value") or "")[:40])
    tables = soup.find_all("table")
    print("tables:", len(tables))
    for i, t in enumerate(tables):
        rows = t.find_all("tr")
        if len(rows) < 3:
            continue
        print(f"--- table {i} id={t.get('id')} rows={len(rows)}")
        for row in rows[:6] + rows[-3:]:
            print("   |", " | ".join(c.get_text(" ", strip=True) for c in row.find_all(["td", "th"]))[:300])
    text = soup.get_text(" ", strip=True)
    print("text sample:", re.sub(r"\s+", " ", text)[:1500])
