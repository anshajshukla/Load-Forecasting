"""Probe: is delhisldc.org reachable from this runner at all (DNS, port 80/443)?"""
import socket, requests
host = "www.delhisldc.org"
for h in (host, "delhisldc.org"):
    try:
        print(h, "resolves to", sorted({a[4][0] for a in socket.getaddrinfo(h, None)}))
    except Exception as e:
        print(h, "DNS error", e)
for port in (80, 443):
    s = socket.socket(); s.settimeout(15)
    try:
        s.connect((host, port)); print("tcp", port, "OPEN")
    except Exception as e:
        print("tcp", port, "FAIL", repr(e))
    finally:
        s.close()
for url in ("http://www.delhisldc.org/Loaddata.aspx?mode=01/06/2024", "https://delhisldc.org/"):
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        print(url, r.status_code, len(r.content))
    except Exception as e:
        print(url, "ERROR", type(e).__name__)
print("runner public IP:", requests.get("https://api.ipify.org", timeout=10).text)
