"""HTTP GET with retries and backoff, shared by every live download (Open-Meteo, Grid-India)."""
from __future__ import annotations

import time


def get(url: str, tries: int = 4, timeout: float = 60, backoff: float = 10, **kw):
    """GET `url`, retrying on timeouts, connection errors and HTTP errors.

    Open-Meteo sometimes stalls for minutes; one slow answer used to fail the whole daily run.
    Waits backoff, 2*backoff, ... seconds between tries and raises the last error if all fail.
    """
    import requests

    for i in range(tries):
        try:
            r = requests.get(url, timeout=timeout, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if i == tries - 1:
                raise
            print(f"GET {url.split('?')[0]} failed ({type(e).__name__}), retry {i + 1}/{tries - 1}")
            time.sleep(backoff * (i + 1))
