#!/usr/bin/env python3
"""Differential Painless harness.

Runs every case of corpus.py through POST /_scripts/painless/_execute on a
reference Elasticsearch 8.19 and on this server, and asserts the responses
are identical (status + JSON body; only Java identity hashes such as
`[I@6d104abf` are normalised).

  ES_REF=http://localhost:9290 ES_OURS=http://localhost:9281 python3 run.py [-v] [-k substr]
"""
import json, os, re, sys, urllib.request, urllib.error

REF = os.environ.get("ES_REF", "http://localhost:9290")
OURS = os.environ.get("ES_OURS", "http://localhost:9281")
IDX = "painless_diff"

MAPPING = {"mappings": {"properties": {
    "n": {"type": "long"}, "i": {"type": "integer"}, "s": {"type": "keyword"}, "d": {"type": "double"},
    "f": {"type": "float"}, "b": {"type": "boolean"}, "dt": {"type": "date"}, "ip": {"type": "ip"},
    "geo": {"type": "geo_point"}, "arr": {"type": "long"}, "strs": {"type": "keyword"},
    "txt": {"type": "text"}, "ver": {"type": "keyword"}, "dn": {"type": "date_nanos"},
    "vec": {"type": "dense_vector", "dims": 3, "index": False},
    "bvec": {"type": "dense_vector", "dims": 3, "element_type": "byte", "index": False}}}}
DOC = {"n": 10, "i": 7, "s": "hello", "d": 3.14, "f": 1.5, "b": True, "dt": "2020-01-15T10:30:00Z",
       "ip": "192.168.1.1", "geo": {"lat": 40.7, "lon": -74.0}, "arr": [1, 2, 3], "strs": ["a", "b"],
       "txt": "quick brown fox", "dn": "2020-01-15T10:30:00.123456789Z", "vec": [1.0, 2.0, 3.0], "bvec": [1, 2, 3]}


def http(base, method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:  # connection problems
        return 0, str(e)


def setup(base):
    http(base, "DELETE", "/" + IDX)
    http(base, "PUT", "/" + IDX, MAPPING)
    http(base, "PUT", f"/{IDX}/_doc/1?refresh=true", DOC)


def norm(text):
    text = re.sub(r"@[0-9a-f]{5,8}\b", "@H", text)
    try:
        return json.loads(text)
    except Exception:
        return text


def request(case):
    name, ctx, src, params, doc = case
    script = {"source": src}
    if params:
        script["params"] = params
    body = {"script": script}
    if ctx != "painless_test":
        body["context"] = ctx
        body["context_setup"] = {"index": IDX, "document": DOC if doc is None else doc}
    elif doc is not None:
        body["context_setup"] = {"index": IDX, "document": doc}
    return body


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from corpus import CASES
    verbose = "-v" in sys.argv
    sub = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else None
    for b in (REF, OURS):
        setup(b)
    ok, bad = 0, []
    for case in CASES:
        if sub and sub not in case[0]:
            continue
        body = request(case)
        r = http(REF, "POST", "/_scripts/painless/_execute", body)
        o = http(OURS, "POST", "/_scripts/painless/_execute", body)
        if r[0] == o[0] and norm(r[1]) == norm(o[1]):
            ok += 1
        else:
            bad.append((case[0], case[2], r, o))
    total = ok + len(bad)
    print(f"PASS {ok}/{total}  FAIL {len(bad)}")
    for name, src, r, o in bad:
        print(f"--- {name}: {src!r}")
        if verbose:
            print("   REF ", r[0], r[1][:700])
            print("   OURS", o[0], o[1][:700])
        else:
            def short(x):
                j = norm(x[1])
                if isinstance(j, dict) and "error" in j:
                    e = j["error"]
                    c = e.get("caused_by", {})
                    return f"{x[0]} {e.get('type')}: {e.get('reason')} / {c.get('type')}: {c.get('reason')}"
                return f"{x[0]} {x[1][:150]}"
            print("   REF ", short(r))
            print("   OURS", short(o))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
