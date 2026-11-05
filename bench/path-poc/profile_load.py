#!/usr/bin/env python3
"""Steady closed-loop load for perf sampling; deliberately not a latency report."""
import argparse
import concurrent.futures
import http.client
import json
import time
from compare import dsl, request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=19213)
ap.add_argument("--path", default="/path-poc/_search?request_cache=false")
inputs = ap.add_mutually_exclusive_group(required=True)
inputs.add_argument("--queries")
inputs.add_argument("--requests", help="prebuilt benchmark request JSONL")
ap.add_argument("--request-id", help="request ID selected from --requests")
ap.add_argument("--seconds", type=int, default=20)
ap.add_argument("--clients", type=int, default=8)
a = ap.parse_args()
if a.requests:
    entries = [json.loads(line) for line in open(a.requests)]
    selected = [entry for entry in entries if a.request_id is None or entry["id"] == a.request_id]
    if not selected:
        ap.error("request ID was not found")
    body = selected[0]["request"]
else:
    if a.request_id:
        ap.error("--request-id requires --requests")
    q = next(json.loads(line) for line in open(a.queries) if json.loads(line)["class"] == "term")
    body = {"query": dsl(q), "size": 10, "_source": False, "track_total_hits": True}
deadline = time.monotonic() + a.seconds
def load(_):
    conn = http.client.HTTPConnection("127.0.0.1", a.port, timeout=120)
    n = 0
    while time.monotonic() < deadline:
        request(conn, "POST", a.path, body)
        n += 1
    return n
with concurrent.futures.ThreadPoolExecutor(max_workers=a.clients) as pool:
    print("completed", sum(pool.map(load, range(a.clients))), "requests")
