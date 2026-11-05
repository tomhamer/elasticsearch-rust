#!/usr/bin/env python3
import json, sys, urllib.request
port = sys.argv[1]
for q in sys.argv[2:]:
    r = urllib.request.Request(f"http://127.0.0.1:{port}/_query?format=txt", data=json.dumps({"query": q}).encode(), headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(r)
        print("==", q); print(resp.read().decode()); print("warnings:", resp.headers.get_all("Warning"))
    except urllib.error.HTTPError as e:
        print("==", q); print(e.read().decode()[:500])
