#!/usr/bin/env python3
"""Median ES|QL latency per query against one or more servers.

    latency.py queries.tsv http://127.0.0.1:19201 [http://...] [--warmup 3] [--runs 20]
        [--max-seconds 120] [--only name]

`queries.tsv`: `name<TAB>query` lines (compare.py --dump). A query whose
first run exceeds --max-seconds is timed once (slow builds).
"""
import argparse
import json
import statistics
import time
import urllib.request


def run(base, q, timeout):
    req = urllib.request.Request(base + "/_query", data=json.dumps({"query": q}).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        urllib.request.urlopen(req, timeout=timeout).read()
        ok = True
    except Exception:
        ok = False
    return (time.perf_counter() - t0) * 1000, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries")
    ap.add_argument("servers", nargs="+")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--max-seconds", type=float, default=120)
    ap.add_argument("--only")
    a = ap.parse_args()
    qs = [l.rstrip("\n").split("\t", 1) for l in open(a.queries) if "\t" in l]
    print("query".ljust(40) + "".join(s.rsplit(":", 1)[-1].rjust(14) for s in a.servers) + "   (median ms)")
    for name, q in qs:
        if a.only and a.only not in name:
            continue
        cells = []
        for s in a.servers:
            first, ok = run(s, q, 3600)
            if first > a.max_seconds * 1000:
                cells.append(f"{first:12.0f}*" if ok else "   error*")
                continue
            for _ in range(a.warmup - 1):
                run(s, q, 3600)
            ts = [run(s, q, 3600)[0] for _ in range(a.runs)]
            cells.append(f"{statistics.median(ts):13.2f}" + ("" if ok else "!"))
        print(name.ljust(40) + " ".join(cells), flush=True)
    print("* single run (over --max-seconds); ! the query failed")


if __name__ == "__main__":
    main()
