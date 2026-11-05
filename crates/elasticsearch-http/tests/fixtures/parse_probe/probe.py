#!/usr/bin/env python3
"""Parse-error fidelity probe: sends every case in cases.jsonl to a reference
Elasticsearch and to our server and diffs the error responses.

    probe.py --ref http://localhost:9430 --ours http://localhost:9431 [-v] [--only substr]
    probe.py --ref ... --record expected.jsonl     # record the reference only

Compared: status, error.type/reason/line/col, the caused_by chain
(type/reason/line/col) and root_cause[0] type/reason. Success bodies are
compared on status only.
"""
import argparse, json, os, sys, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

SETUP = [
    ("POST", "/_license/start_trial?acknowledge=true", None),
    ("DELETE", "/probe,probe-*", None),
    ("DELETE", "/_index_template/probe-tpl", None),
    ("PUT", "/probe", json.dumps({
        "aliases": {"probe-alias": {}},
        "mappings": {"properties": {
            "f": {"type": "keyword"}, "t": {"type": "text"}, "n": {"type": "long"}, "d": {"type": "date"},
            "g": {"type": "geo_point"}, "o": {"type": "nested"}, "p": {"type": "keyword"},
            "v": {"type": "dense_vector", "dims": 2, "similarity": "l2_norm"}}}})),
    ("PUT", "/probe/_doc/1?refresh=true", json.dumps({"f": "a", "n": 1})),
]


def call(base, method, path, body):
    data = body.encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method)
    if data is not None:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            st = r.status
    except urllib.error.HTTPError as e:
        raw = e.read()
        st = e.code
    try:
        js = json.loads(raw) if raw else None
    except ValueError:
        js = raw.decode(errors="replace")
    return st, js


def shape(st, js):
    out = {"status": st}
    if not (isinstance(js, dict) and isinstance(js.get("error"), dict)):
        return out
    e = js["error"]
    chain = []
    cur = e
    while isinstance(cur, dict):
        chain.append({k: cur.get(k) for k in ("type", "reason", "line", "col") if k in cur})
        cur = cur.get("caused_by")
    out["chain"] = chain
    rc = e.get("root_cause") or []
    if rc:
        out["root"] = {k: rc[0].get(k) for k in ("type", "reason")}
    return out


def load_cases(only):
    cases = [json.loads(l) for l in open(os.path.join(HERE, "cases.jsonl")) if l.strip()]
    return [c for c in cases if not only or any(o in c["name"] for o in only)]


def setup(base):
    for m, p, b in SETUP:
        call(base, m, p, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ours")
    ap.add_argument("--record")
    ap.add_argument("--only", action="append")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    cases = load_cases(a.only)
    setup(a.ref)
    if a.ours:
        setup(a.ours)
    if a.record:
        with open(a.record, "w") as f:
            for c in cases:
                f.write(json.dumps({"name": c["name"], "expect": shape(*call(a.ref, c["method"], c["path"], c["body"]))}) + "\n")
        return
    ok = 0
    for c in cases:
        r = shape(*call(a.ref, c["method"], c["path"], c["body"]))
        o = shape(*call(a.ours, c["method"], c["path"], c["body"]))
        if r == o:
            ok += 1
            continue
        print("DIFF", c["name"], c["method"], c["path"], c["body"] if c["body"] is None or len(c["body"]) < 200 else c["body"][:200])
        if a.v:
            print("   ref :", json.dumps(r))
            print("   ours:", json.dumps(o))
    print(f"{ok}/{len(cases)} match")


if __name__ == "__main__":
    main()
