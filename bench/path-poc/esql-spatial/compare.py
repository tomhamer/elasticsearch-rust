#!/usr/bin/env python3
"""Compare ES|QL responses of two servers for every `esql` operation of a
Rally track (default: the geopoint track).

    compare.py --track ~/rally-1pct/tracks/geopoint --a http://127.0.0.1:19200 \
        --b http://127.0.0.1:19201 [--dump queries.tsv] [--only name]

Prints, per operation, whether `columns` and `values` are identical, and
the first difference otherwise. Wall time of each call is shown for both.
`--dump` writes `name<TAB>query` lines (the POC's `POC_QUERIES` input).
"""
import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.request


def operations(track):
    """`(name, query)` of the track's esql operations (Rally's Jinja
    templating is not evaluated: the geopoint operations are plain JSON
    once `{{ ... }}` defaults are substituted)."""
    out = []
    for path in sorted(glob.glob(os.path.join(track, "operations", "*.json"))) + [os.path.join(track, "track.json")]:
        if not os.path.exists(path):
            continue
        text = open(path).read()
        # `{{ var | default(x) }}` -> x
        text = re.sub(r"\{\{\s*[\w.]+\s*\|\s*default\(([^)]*)\)\s*\}\}", lambda m: m.group(1).strip("'\""), text)
        for m in re.finditer(r'"operation-type"\s*:\s*"esql"', text):
            # The enclosing object: scan back to its "name".
            start = text.rfind("{", 0, m.start())
            depth, i = 0, start
            while i < len(text):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            try:
                op = json.loads(text[start:i + 1])
            except json.JSONDecodeError:
                continue
            if "query" in op:
                out.append((op["name"], op["query"]))
    seen, uniq = set(), []
    for n, q in out:
        if n not in seen:
            seen.add(n)
            uniq.append((n, q))
    return uniq


SAVED = {}


def query(base, q):
    if base.startswith("file:"):
        body, warn = SAVED[base][q]
        return body, warn, 0.0
    req = urllib.request.Request(base + "/_query?format=json", data=json.dumps({"query": q}).encode(), headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            body = json.loads(r.read())
            warn = r.headers.get_all("Warning") or []
    except urllib.error.HTTPError as e:
        body = {"error": json.loads(e.read() or b"{}"), "status": e.code}
        warn = []
    return body, warn, time.time() - t0


def first_diff(a, b, path=""):
    if type(a) != type(b):
        return f"{path}: {a!r} vs {b!r}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: missing on {'a' if k not in a else 'b'}"
            d = first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: len {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {a!r} vs {b!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default=os.path.expanduser("~/rally-1pct/tracks/geopoint"))
    ap.add_argument("--a", default="http://127.0.0.1:19200")
    ap.add_argument("--b", default="http://127.0.0.1:19201")
    ap.add_argument("--dump")
    ap.add_argument("--only")
    ap.add_argument("--sorted", action="store_true", help="also compare with rows sorted (order-insensitive)")
    ap.add_argument("--save-a", help="save server a's responses (JSON) for a later --a file:PATH")
    args = ap.parse_args()
    if args.a.startswith("file:"):
        SAVED[args.a] = {q: (b, w) for q, b, w in json.load(open(args.a[5:]))}
    saved = []
    ops = operations(args.track)
    if args.dump:
        with open(args.dump, "w") as f:
            for n, q in ops:
                f.write(f"{n}\t{q}\n")
    same = 0
    for name, q in ops:
        if args.only and args.only not in name:
            continue
        a, wa, ta = query(args.a, q)
        saved.append((q, a, wa))
        b, wb, tb = query(args.b, q)
        ka = {k: a.get(k) for k in ("columns", "values", "error")}
        kb = {k: b.get(k) for k in ("columns", "values", "error")}
        d = first_diff(ka, kb)
        if d and args.sorted and "values" in a and "values" in b:
            if a.get("columns") == b.get("columns") and sorted(map(json.dumps, a["values"])) == sorted(map(json.dumps, b["values"])):
                d = "same rows, different order"
        m = re.match(r"^(FROM \S+)\s*\| WHERE (.*) \| LIMIT (\d+)$", q)
        if d and m and "SORT" not in q and "values" in a and "values" in b and a.get("columns") == b.get("columns") and len(a["values"]) == len(b["values"]):
            # LIMIT without SORT: any matching rows are right (ES's pick
            # varies run to run). Check that a sample of b's rows match the
            # predicate on server a.
            bad = 0
            rows_to_check = [] if args.a.startswith("file:") else b["values"][:: max(1, len(b["values"]) // 25)]
            for row in rows_to_check:
                pt = row[0]
                chk = f'{m.group(1)} | WHERE ({m.group(2)}) AND ST_INTERSECTS(location, TO_GEOSHAPE("{pt}")) | STATS c = COUNT(*)'
                r, _, _ = query(args.a, chk)
                if not r.get("values") or r["values"][0][0] < 1:
                    bad += 1
            d = None if bad == 0 else f"{bad} sampled rows of b do not match on a"
            if d is None:
                name += " (LIMIT rows verified)" if rows_to_check else " (LIMIT: same columns and row count)"
        wdiff = "" if wa == wb else f"  warnings differ: {wa} vs {wb}"
        status = "SAME" if not d else "DIFF"
        same += not d
        print(f"{status} {name:40} a {ta*1000:9.1f} ms  b {tb*1000:9.1f} ms  rows {len(a.get('values', []))}/{len(b.get('values', []))}{wdiff}")
        if d:
            print(f"     {d[:400]}")
    print(f"{same}/{len(ops)} identical")
    if args.save_a:
        json.dump(saved, open(args.save_a, "w"))


if __name__ == "__main__":
    sys.exit(main())
