#!/usr/bin/env python3
"""Refresh-disabled bulk load, then time the first refresh with concurrent searches.

usage: python3 -I refresh_after_ingest.py http://HOST:9200 DOCS [CLIENTS] [BULK]
"""
import json, random, statistics, sys, threading, time, urllib.request

url, total = sys.argv[1].rstrip("/"), int(sys.argv[2])
clients = int(sys.argv[3]) if len(sys.argv) > 3 else 8
bulk = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
idx = "refresh-repro"

def call(method, path, body=None, ctype="application/json", timeout=600):
    data = body.encode() if isinstance(body, str) else body
    req = urllib.request.Request(url + path, data=data, method=method, headers={"Content-Type": ctype})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read() or b"{}")
    return out, time.time() - t0

try: call("DELETE", "/" + idx)
except Exception: pass
call("PUT", "/" + idx, json.dumps({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}}))

words = [("w%d" % i) for i in range(5000)]
def doc(i):
    return {"id": i, "actor": {"login": random.choice(words), "id": random.randint(1, 10**6)},
            "repo": {"name": random.choice(words) + "/" + random.choice(words)},
            "type": random.choice(["PushEvent", "IssuesEvent", "WatchEvent", "ForkEvent"]),
            "payload": {"text": " ".join(random.choice(words) for _ in range(120))},
            "created_at": "2026-10-%02dT%02d:00:00Z" % (random.randint(1, 28), random.randint(0, 23))}

lat, lock, next_i = [], threading.Lock(), [0]
def loader():
    while True:
        with lock:
            start = next_i[0]; next_i[0] += bulk
        if start >= total: return
        body = "".join('{"index":{}}\n' + json.dumps(doc(i)) + "\n" for i in range(start, min(start + bulk, total)))
        r, t = call("POST", "/%s/_bulk" % idx, body, "application/x-ndjson")
        assert not r.get("errors"), str(r)[:500]
        with lock: lat.append(t)

t0 = time.time()
ts = [threading.Thread(target=loader) for _ in range(clients)]
[t.start() for t in ts]; [t.join() for t in ts]
ingest = time.time() - t0
lat.sort()
print("ingest: %d docs in %.1fs = %.0f docs/s; bulk p50 %.2fs p99 %.2fs max %.2fs"
      % (total, ingest, total / ingest, lat[len(lat) // 2], lat[int(len(lat) * 0.99)], lat[-1]))

searches, refresh_done = [], threading.Event()
def searcher():
    time.sleep(0.2)
    while not refresh_done.is_set():
        _, t = call("POST", "/%s/_search" % idx, json.dumps({"query": {"match_all": {}}, "size": 10}), timeout=120)
        searches.append(t)
        time.sleep(0.1)
s = threading.Thread(target=searcher); s.start()
_, refresh = call("POST", "/%s/_refresh" % idx)
refresh_done.set(); s.join()
count = call("GET", "/%s/_count" % idx)[0]["count"]
print("first refresh: %.2fs; %d searches during it, max %.2fs; count after refresh %d"
      % (refresh, len(searches), max(searches or [0]), count))
_, t = call("POST", "/%s/_search" % idx, json.dumps({"query": {"match": {"payload.text": "w42"}}, "size": 10}))
print("first match query after refresh: %.2fs" % t)
ok = refresh <= 10 and max(searches or [0]) <= 1 and count == total and t <= 5
print("PASS" if ok else "FAIL")
