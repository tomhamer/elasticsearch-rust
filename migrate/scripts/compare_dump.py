#!/usr/bin/env python3
"""Compare `es-lucene-import dump` output against ground truth from
fixtures.py (or any {target: {id: {routing, source}}} JSON), or against a
live server holding the imported data.

usage: compare_dump.py <expected.json> --dump <dir>
       compare_dump.py <expected.json> --server <url>
"""
import json, sys, urllib.request

expected = json.load(open(sys.argv[1]))
mode, where = sys.argv[2], sys.argv[3]


def from_dump(target):
    docs = {}
    try:
        lines = open(f"{where}/{target}.bulk.ndjson").read().splitlines()
    except FileNotFoundError:
        return docs
    for a, s in zip(lines[::2], lines[1::2]):
        meta = list(json.loads(a).values())[0]
        docs[meta["_id"]] = {"routing": meta.get("routing"), "source": json.loads(s)}
    return docs


def call(method, path, body=None):
    r = urllib.request.Request(where.rstrip("/") + path, data=json.dumps(body).encode() if body else None, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r) as resp:
        return json.loads(resp.read())


def from_server(target):
    # one page (fixtures stay under index.max_result_window): avoids depending
    # on the target's scroll implementation
    docs = {}
    try:
        call("POST", f"/{target}/_open")
    except Exception:
        pass
    r = call("POST", f"/{target}/_search?size=10000", {"query": {"match_all": {}}})
    for h in r["hits"]["hits"]:
        docs[h["_id"]] = {"routing": h.get("_routing"), "source": h["_source"]}
    return docs


bad = 0
for target, exp in expected.items():
    got = from_dump(target) if mode == "--dump" else from_server(target)
    missing = sorted(set(exp) - set(got))
    extra = sorted(set(got) - set(exp))
    diff = [i for i in exp if i in got and exp[i] != got[i]]
    ok = not (missing or extra or diff)
    bad += not ok
    print(f"{target:12} expected {len(exp):5} got {len(got):5} missing {len(missing)} extra {len(extra)} differing {len(diff)} {'OK' if ok else 'FAIL'}")
    for i in (missing[:3] + extra[:3] + diff[:3]):
        print("   ", repr(i), json.dumps(exp.get(i))[:200], "|", json.dumps(got.get(i))[:200])
sys.exit(1 if bad else 0)
