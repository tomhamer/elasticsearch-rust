#!/usr/bin/env python3
"""load.py base corpus [n]: create osmgeopoints (track mapping, 5 shards) and bulk-load."""
import json, sys, urllib.request
base, corpus = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else None
def req(m, p, body=None, ct="application/json"):
    r = urllib.request.Request(base + p, data=body, method=m, headers={"Content-Type": ct})
    try:
        return json.loads(urllib.request.urlopen(r, timeout=600).read() or b"{}")
    except urllib.error.HTTPError as e:
        return json.loads(e.read() or b"{}")
req("DELETE", "/osmgeopoints")
print(req("PUT", "/osmgeopoints", json.dumps({"settings": {"index.number_of_shards": 5, "index.number_of_replicas": 0, "index.requests.cache.enable": False}, "mappings": {"dynamic": "strict", "properties": {"location": {"type": "geo_point"}}}}).encode()))
buf, k = [], 0
with open(corpus) as f:
    for line in f:
        buf.append('{"index":{}}\n' + line.rstrip("\n") + "\n")
        k += 1
        if len(buf) == 5000:
            r = req("POST", "/osmgeopoints/_bulk", "".join(buf).encode(), "application/x-ndjson"); assert not r.get("errors"), str(r)[:300]; buf = []
        if n and k >= n: break
if buf:
    r = req("POST", "/osmgeopoints/_bulk", "".join(buf).encode(), "application/x-ndjson"); assert not r.get("errors"), str(r)[:300]
req("POST", "/osmgeopoints/_refresh")
print(req("POST", "/osmgeopoints/_forcemerge?max_num_segments=1"))
req("POST", "/osmgeopoints/_refresh")
print(req("GET", "/osmgeopoints/_count"))
