#!/usr/bin/env python3
"""Check identical float graph results on the same index before timing a kernel change."""
import argparse
import http.client
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--requests", required=True)
ap.add_argument("--before-port", type=int, required=True)
ap.add_argument("--after-port", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
connections = [http.client.HTTPConnection("127.0.0.1", p, timeout=120)
               for p in (a.before_port, a.after_port)]
checks = []
for line in Path(a.requests).read_text().splitlines():
    entry = json.loads(line)
    body = json.dumps(entry["request"]).encode()
    results = []
    for c in connections:
        c.request("POST", "/candidate/_search", body, {"Content-Type": "application/json"})
        response = c.getresponse()
        result = json.loads(response.read())
        assert response.status == 200, (entry["id"], response.status, result)
        results.append(result)
    assert results[0]["hits"] == results[1]["hits"], (entry["id"], results)
    checks.append(entry["id"])
for c in connections:
    c.close()
Path(a.out).write_text(json.dumps({"requests": len(checks), "exact_hit_parity": True,
                                "ids": checks}, indent=2) + "\n")
print(f"{len(checks)} identical hit/score/total checks")
