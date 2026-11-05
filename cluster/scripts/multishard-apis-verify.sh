#!/usr/bin/env bash
# Multi-shard API check (kNN cases avoid score ties and BM25: tie order and
# per-shard term stats legitimately differ from the 1-shard twin)
# on a local 3-node cluster (HTTP PORT_BASE..PORT_BASE+2, default 9300):
# _mget, _msearch, _search_shards, _terms_enum, _cat/indices + _stats docs,
# scroll, PIT and _async_search on a 3-shard x 1-replica index.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PORT_BASE="${PORT_BASE:-9300}" TAG=msapis
export P0=$PORT_BASE P1=$((PORT_BASE + 1)) P2=$((PORT_BASE + 2))
C="$HERE/snap-cluster.sh"
H=127.0.0.1:$P0
N=300

"$C" start || exit 1
for i in $(seq 1 60); do curl -s "$H/_cluster/health" >/dev/null 2>&1 && break; sleep 1; done
curl -s -XPUT "$H/ms" -H 'content-type: application/json' \
  -d '{"settings":{"number_of_shards":3,"number_of_replicas":1},"mappings":{"properties":{"tag":{"type":"keyword"},"n":{"type":"integer"}}}}' | head -c 300; echo
curl -s "$H/_cluster/health/ms?wait_for_status=green&timeout=60s" | head -c 300; echo
python3 - "$N" >/tmp/msapis-bulk.ndjson <<'EOF'
import json, sys
for i in range(int(sys.argv[1])):
    print(json.dumps({"index": {"_index": "ms", "_id": str(i)}}))
    print(json.dumps({"n": i, "tag": f"t{i:03d}"}))
EOF
curl -s -XPOST "127.0.0.1:$P1/_bulk?refresh=true" -H 'content-type: application/x-ndjson' \
  --data-binary @/tmp/msapis-bulk.ndjson | head -c 120; echo

j() { python3 -c "import json,sys; r=json.load(sys.stdin); print($1)"; }
echo "--- _mget (ids)"
curl -s -XPOST "$H/ms/_mget" -H 'content-type: application/json' -d '{"ids":["0","1","2","3","4","5","299","nope"]}' \
  | j '[(d["_id"], d.get("found"), d.get("_source",{}).get("n")) for d in r["docs"]]'
echo "--- _mget (docs, via node 3)"
curl -s -XPOST "127.0.0.1:$P2/_mget" -H 'content-type: application/json' -d '{"docs":[{"_index":"ms","_id":"7"},{"_index":"ms","_id":"150"},{"_index":"missing","_id":"1"}]}' \
  | head -c 900; echo
echo "--- _msearch"
printf '{"index":"ms"}\n{"query":{"match_all":{}},"size":0,"track_total_hits":true}\n{}\n{"query":{"term":{"tag":"t042"}}}\n{"index":"missing"}\n{}\n' \
  | curl -s -XPOST "$H/ms/_msearch" -H 'content-type: application/x-ndjson' --data-binary @- \
  | j '[(x.get("status"), x.get("hits",{}).get("total"), x.get("_shards"), x.get("error",{}).get("type")) for x in r["responses"]], "SHARDS_OK" if all(x["_shards"]["total"] == 3 for x in r["responses"][:2]) else "SHARDS_BAD"'
echo "--- _search_shards"
curl -s "$H/ms/_search_shards" | j '[[ (c["shard"], c["primary"], c["node"], c["state"]) for c in s] for s in r["shards"]], sorted(r["nodes"])'
curl -s "$H/ms/_search_shards?routing=5" | j '[[c["shard"] for c in s] for s in r["shards"]]'
echo "--- _terms_enum"
curl -s -XPOST "$H/ms/_terms_enum" -H 'content-type: application/json' -d '{"field":"tag","string":"t1","size":5}' | head -c 400; echo
echo "--- _count / _cat/indices / _stats docs"
curl -s "$H/ms/_count" | head -c 200; echo
curl -s "$H/_cat/indices/ms?v&h=index,pri,rep,docs.count" ;
curl -s "$H/ms/_stats/docs" | j 'r["_shards"], r["_all"]["primaries"]["docs"], r["_all"]["total"]["docs"], r["indices"]["ms"]["primaries"]["docs"]'
echo "--- _count preference=_shards (each subset sums to the total)"
for p in 0 1 2 0,2; do echo -n "_shards:$p "; curl -s "$H/ms/_count?preference=_shards:$p" | head -c 200; echo; done
echo "--- _stats level=shards (shard -> [(primary, routing.node)])"
curl -s "$H/ms/_stats/docs?level=shards" \
  | j '{k: [(c["routing"]["primary"], c["routing"]["node"], c["docs"]["count"]) for c in v] for k, v in sorted(r["indices"]["ms"]["shards"].items())}'
curl -s "$H/_nodes" | j 'sorted(r["nodes"])'
echo "--- explain=true hit labels"
curl -s -XPOST "$H/ms/_search?explain=true" -H 'content-type: application/json' -d '{"size":6,"query":{"match_all":{}}}' \
  | j 'sorted({(h["_shard"], h["_node"]) for h in r["hits"]["hits"]})'
echo "--- scroll"
python3 - <<'EOF'
import json, urllib.request
import os
P0, P1, P2 = (int(os.environ[k]) for k in ("P0", "P1", "P2"))
def call(m, p, b=None):
    rq = urllib.request.Request(f"http://127.0.0.1:{P0}" + p, method=m, data=json.dumps(b).encode() if b is not None else None, headers={"content-type": "application/json"})
    try: return json.load(urllib.request.urlopen(rq))
    except urllib.error.HTTPError as e: return {"status": e.code, "err": e.read()[:300].decode()}
r = call("POST", "/ms/_search?scroll=1m", {"size": 40, "sort": ["n"]})
if "hits" not in r: print("scroll start failed", r); raise SystemExit
seen = [h["_source"]["n"] for h in r["hits"]["hits"]]; sid = r["_scroll_id"]; print("first page _shards", r["_shards"], "total", r["hits"]["total"], "SHARDS_OK" if r["_shards"]["total"] == 3 else "SHARDS_BAD")
while True:
    r = call("POST", "/_search/scroll", {"scroll": "1m", "scroll_id": sid})
    if "hits" not in r: print("scroll page failed", r); break
    if not r["hits"]["hits"]: break
    seen += [h["_source"]["n"] for h in r["hits"]["hits"]]; sid = r["_scroll_id"]
print("scroll docs", len(seen), "unique", len(set(seen)), "sorted", seen == sorted(seen))
print("clear", call("DELETE", "/_search/scroll", {"scroll_id": sid}))
p = call("POST", "/ms/_pit?keep_alive=1m")
if "id" not in p: print("pit open failed", p); raise SystemExit
print("pit open _shards", p.get("_shards"), "SHARDS_OK" if p.get("_shards", {}).get("total") == 3 else "SHARDS_BAD")
r = call("POST", "/_search", {"pit": {"id": p["id"]}, "size": 1000, "sort": ["n"]})
print("pit search", r.get("_shards"), len(r.get("hits", {}).get("hits", [])), r.get("err"), "SHARDS_OK" if r.get("_shards", {}).get("total") == 3 else "SHARDS_BAD")
after = r["hits"]["hits"][99]["sort"] if len(r.get("hits", {}).get("hits", [])) > 99 else [99]
r2 = call("POST", "/_search", {"pit": {"id": p["id"]}, "size": 50, "sort": ["n"], "search_after": after})
print("pit search_after", after, "first", [h["_source"]["n"] for h in r2.get("hits", {}).get("hits", [])][:3], r2.get("err"))
print("pit close", call("DELETE", "/_pit", {"id": p["id"]}))
a = call("POST", "/ms/_async_search?size=0&track_total_hits=true", {"query": {"match_all": {}}})
print("async", a.get("is_running"), a.get("response", {}).get("hits", {}).get("total"), a.get("response", {}).get("_shards"), a.get("err"), "SHARDS_OK" if a.get("response", {}).get("_shards", {}).get("total") == 3 else "SHARDS_BAD")
EOF
echo "--- knn DFS (3-shard kv vs 1-shard twin kv1: same ids, scores, totals, aggs)"
python3 - <<'EOF'
import json, math, urllib.request
import os
P0, P1, P2 = (int(os.environ[k]) for k in ("P0", "P1", "P2"))
def call(m, p, b=None, port=P0):
    rq = urllib.request.Request(f"http://127.0.0.1:{port}" + p, method=m, data=json.dumps(b).encode() if b is not None else None, headers={"content-type": "application/json"})
    try: return json.load(urllib.request.urlopen(rq))
    except urllib.error.HTTPError as e: return {"status": e.code, "err": e.read()[:300].decode()}
mp = {"properties": {"v": {"type": "dense_vector", "dims": 3, "similarity": "l2_norm"}, "w": {"type": "dense_vector", "dims": 2, "similarity": "cosine"}, "n": {"type": "integer"}, "tag": {"type": "keyword"}}}
for name, shards in (("kv", 3), ("kv1", 1)):
    print(name, call("PUT", "/" + name, {"settings": {"number_of_shards": shards, "number_of_replicas": 1}, "mappings": mp}).get("acknowledged"))
call("GET", "/_cluster/health/kv,kv1?wait_for_status=green&timeout=60s")
lines = []
for i in range(200):
    d = {"n": i, "tag": "even" if i % 2 == 0 else "odd", "v": [math.sin(i), math.cos(i * 0.7), (i % 17) / 17.0], "w": [1.0 + (i % 13) + i / 1000.0, 1.0 + (i % 7)]}
    for name in ("kv", "kv1"):
        lines += [json.dumps({"index": {"_index": name, "_id": str(i)}}), json.dumps(d)]
rq = urllib.request.Request(f"http://127.0.0.1:{P1}/_bulk?refresh=true", method="POST", data=("\n".join(lines) + "\n").encode(), headers={"content-type": "application/x-ndjson"})
print("bulk errors", json.load(urllib.request.urlopen(rq))["errors"])
K1 = {"field": "v", "query_vector": [0.3, -0.2, 0.5], "k": 10, "num_candidates": 400}
K2 = {"field": "w", "query_vector": [3.0, 1.0], "k": 7, "num_candidates": 400, "boost": 0.5, "_name": "wk"}
cases = [
    ("lone", "", {"knn": K1, "size": 10}),
    ("hybrid", "", {"knn": K1, "query": {"bool": {"should": [{"term": {"n": {"value": 5, "boost": 0.3}}}, {"term": {"n": {"value": 7, "boost": 0.2}}}, {"term": {"n": {"value": 38, "boost": 0.1}}}]}}, "size": 15}),
    ("two clauses", "", {"knn": [K1, K2], "size": 20}),
    ("knn + aggs", "", {"knn": K1, "size": 3, "aggs": {"t": {"terms": {"field": "tag"}}, "s": {"sum": {"field": "n"}}}}),
    ("knn + sort", "", {"knn": K2, "sort": [{"n": "desc"}], "size": 5}),
    ("knn + q", "?q=n:5", {"knn": K1, "size": 12}),
    ("knn filter + post_filter", "", {"knn": dict(K1, filter={"term": {"tag": "odd"}}), "post_filter": {"range": {"n": {"gte": 50}}}, "size": 10}),
    ("knn from", "", {"knn": [K1, K2], "from": 4, "size": 6}),
]
for label, qs, body in cases:
    out = []
    for name in ("kv", "kv1"):
        r = call("POST", f"/{name}/_search{qs}", body)
        if "hits" not in r: out.append(("ERR", r)); continue
        out.append(([(h["_id"], round(h["_score"], 5) if h.get("_score") is not None else None, h.get("matched_queries"), h.get("sort")) for h in r["hits"]["hits"]],
                    r["hits"]["total"], json.dumps(r.get("aggregations"), sort_keys=True), r["_shards"]["total"]))
    a, b = out
    ok = a[0] != "ERR" and b[0] != "ERR" and a[:3] == b[:3]
    print(("KNN_OK " if ok else "KNN_BAD ") + label, "shards", a[3] if a[0] != "ERR" else a, "hits", len(a[0]) if a[0] != "ERR" else "", "" if ok else f"\n  kv:  {a}\n  kv1: {b}")
r = call("POST", "/kv/_search?scroll=1m", {"knn": K1, "query": {"match_all": {}}})
print("knn DFS in scroll rejected:", r.get("status"), r.get("err", "")[:160])
EOF
echo "--- collapse inner_hits (3-shard cv vs 1-shard twin cv1: hits, inner hits, totals; field sorts avoid ties)"
python3 - <<'EOF'
import json, urllib.request
import os
P0, P1, P2 = (int(os.environ[k]) for k in ("P0", "P1", "P2"))
def call(m, p, b=None, port=P0):
    rq = urllib.request.Request(f"http://127.0.0.1:{port}" + p, method=m, data=json.dumps(b).encode() if b is not None else None, headers={"content-type": "application/json"})
    try: return json.load(urllib.request.urlopen(rq))
    except urllib.error.HTTPError as e: return {"status": e.code, "err": json.loads(e.read().decode() or "{}")}
mp = {"properties": {"g": {"type": "keyword"}, "n": {"type": "integer"}, "tag": {"type": "keyword"}}}
for name, shards in (("cv", 3), ("cv1", 1)):
    print(name, call("PUT", "/" + name, {"settings": {"number_of_shards": shards, "number_of_replicas": 1}, "mappings": mp}).get("acknowledged"))
call("GET", "/_cluster/health/cv,cv1?wait_for_status=green&timeout=60s")
lines = []
for i in range(150):
    d = {"n": i, "tag": "abc"[i % 3]}
    if i % 11 != 0: d["g"] = f"g{i % 7}"
    for name in ("cv", "cv1"):
        lines += [json.dumps({"index": {"_index": name, "_id": str(i)}}), json.dumps(d)]
rq = urllib.request.Request(f"http://127.0.0.1:{P1}/_bulk?refresh=true", method="POST", data=("\n".join(lines) + "\n").encode(), headers={"content-type": "application/x-ndjson"})
print("bulk errors", json.load(urllib.request.urlopen(rq))["errors"])
def norm(v):
    if isinstance(v, dict): return {k: norm(x) for k, x in v.items() if k not in ("_index", "took", "_shards", "_shard", "_node")}
    if isinstance(v, list): return [norm(x) for x in v]
    return v
ASC, DESC = [{"n": "asc"}], [{"n": "desc"}]
cases = [
    ("single inner_hits + sort", "", {"collapse": {"field": "g", "inner_hits": {"name": "top", "size": 2, "sort": ASC}}, "sort": DESC, "size": 10}),
    ("multiple inner_hits", "", {"collapse": {"field": "g", "inner_hits": [{"name": "lo", "size": 2, "sort": ASC}, {"name": "hi", "size": 4, "sort": DESC, "_source": False, "version": True}]}, "sort": DESC, "size": 10}),
    ("inner from/size", "", {"collapse": {"field": "g", "inner_hits": {"name": "page", "from": 2, "size": 4, "sort": ASC}}, "sort": ASC, "size": 10}),
    ("inner default size", "", {"collapse": {"field": "g", "inner_hits": {"name": "d", "sort": DESC}}, "sort": ASC}),
    ("_source filtering", "", {"collapse": {"field": "g", "inner_hits": {"name": "s", "size": 3, "sort": ASC, "_source": {"excludes": ["tag"]}, "docvalue_fields": ["tag"]}}, "sort": DESC, "size": 10}),
    ("missing collapse value", "", {"collapse": {"field": "g", "inner_hits": {"name": "m", "size": 5, "sort": DESC}}, "sort": ASC, "size": 3}),
    ("query + from", "", {"query": {"constant_score": {"filter": {"range": {"n": {"gte": 40, "lt": 120}}}}}, "collapse": {"field": "g", "inner_hits": {"name": "q", "size": 3, "sort": ASC}}, "sort": DESC, "from": 2, "size": 4}),
    ("q param", "?q=tag:b", {"collapse": {"field": "g", "inner_hits": {"name": "q", "size": 3, "sort": ASC}}, "sort": DESC, "size": 10}),
    ("post_filter", "", {"post_filter": {"term": {"tag": "c"}}, "collapse": {"field": "g", "inner_hits": {"name": "p", "size": 3, "sort": ASC}}, "sort": DESC, "size": 10}),
    ("nested collapse", "", {"collapse": {"field": "g", "inner_hits": {"name": "bytag", "size": 5, "sort": ASC, "collapse": {"field": "tag"}}}, "sort": DESC, "size": 10}),
    ("rest_total_hits_as_int", "?rest_total_hits_as_int=true", {"collapse": {"field": "g", "inner_hits": {"name": "t", "size": 1, "sort": ASC}}, "sort": DESC, "size": 4}),
    ("err: missing name", "", {"collapse": {"field": "g", "inner_hits": {"size": 1}}, "sort": DESC}),
    ("err: inner collapse inner_hits", "", {"collapse": {"field": "g", "inner_hits": {"name": "x", "collapse": {"field": "tag", "inner_hits": {"name": "y"}}}}, "sort": DESC}),
]
for label, qs, body in cases:
    a, b = (norm(call("POST", f"/{name}/_search{qs}", body)) for name in ("cv", "cv1"))
    ok = a == b and (label.startswith("err") or ("hits" in a and all("inner_hits" in h for h in a["hits"]["hits"]) and a["hits"]["hits"]))
    print(("COLLAPSE_OK " if ok else "COLLAPSE_BAD ") + label, "hits", len(a.get("hits", {}).get("hits", [])) if "hits" in a else a.get("err", a)["error"]["type"] if "err" in a else a,
          "" if ok else f"\n  cv:  {json.dumps(a)[:1500]}\n  cv1: {json.dumps(b)[:1500]}")
r = call("POST", "/cv/_search?scroll=1m", {"collapse": {"field": "g", "inner_hits": {"name": "x"}}, "sort": DESC})
r1 = call("POST", "/cv1/_search?scroll=1m", {"collapse": {"field": "g", "inner_hits": {"name": "x"}}, "sort": DESC})
ok = r.get("status") == r1.get("status") == 400 and r["err"]["error"]["type"] == r1["err"]["error"]["type"] and r["err"]["error"].get("reason") == r1["err"]["error"].get("reason")
print(("COLLAPSE_OK " if ok else "COLLAPSE_BAD ") + "scroll rejected", r.get("status"), json.dumps(r.get("err"))[:200], "" if ok else f"\n  cv1: {json.dumps(r1)[:300]}")
EOF
echo "--- DFS global term stats (3-shard dv vs 1-shard twin dv1, skewed corpus: ids + _score within 1e-4)"
python3 - <<'EOF'
import json, math, urllib.request
import os
P0, P1, P2 = (int(os.environ[k]) for k in ("P0", "P1", "P2"))
def call(m, p, b=None, port=P0, nd=None):
    data = nd.encode() if nd is not None else json.dumps(b).encode() if b is not None else None
    ct = "application/x-ndjson" if nd is not None else "application/json"
    rq = urllib.request.Request(f"http://127.0.0.1:{port}" + p, method=m, data=data, headers={"content-type": ct})
    try: return json.load(urllib.request.urlopen(rq))
    except urllib.error.HTTPError as e: return {"status": e.code, "err": e.read()[:300].decode()}
mp = {"properties": {"t": {"type": "text"}, "u": {"type": "text"}, "v": {"type": "dense_vector", "dims": 3, "similarity": "l2_norm"}, "n": {"type": "integer"}}}
for name, shards in (("dv", 3), ("dv1", 1)):
    print(name, call("PUT", "/" + name, {"settings": {"number_of_shards": shards, "number_of_replicas": 1}, "mappings": mp}).get("acknowledged"))
call("GET", "/_cluster/health/dv,dv1?wait_for_status=green&timeout=60s")
# Skew: "fox" / "quick" docs all share one routing key (one shard of dv);
# the rest spread by id. Doc lengths grow with the id, so no two docs tie.
lines = []
for i in range(240):
    skew = i < 90
    words = (["fox"] * (1 + i % 4) if skew else []) + (["quick"] * (1 + i % 3) if skew or i % 9 == 0 else []) + (["dog"] * (1 + i % 2) if i % 3 == 0 else [])
    d = {"n": i, "t": " ".join(words + ["pad"] * i), "u": "brown " * (1 + i % 5) + "pad " * i,
         "v": [math.sin(i), math.cos(i * 0.7), (i % 17) / 17.0]}
    for name in ("dv", "dv1"):
        meta = {"_index": name, "_id": str(i)}
        if skew: meta["routing"] = "skew"
        lines += [json.dumps({"index": meta}), json.dumps(d)]
print("bulk errors", call("POST", "/_bulk?refresh=true", port=P1, nd="\n".join(lines) + "\n").get("errors"))
print("dv shard docs", sorted(c["docs"]["count"] for v in call("GET", "/dv/_stats/docs?level=shards")["indices"]["dv"]["shards"].values() for c in v if c["routing"]["primary"]))
DFS = "search_type=dfs_query_then_fetch"
K = {"field": "v", "query_vector": [0.3, -0.2, 0.5], "k": 10, "num_candidates": 400}
# (label, body, knn): ES rejects an explicit search_type with knn (the search
# type is determined automatically: DFS), so knn cases run without one.
cases = [
    ("match", {"query": {"match": {"t": "fox dog"}}, "size": 25}, False),
    ("multi field bool", {"query": {"bool": {"should": [{"match": {"t": "quick dog"}}, {"match": {"u": "brown"}}], "filter": {"range": {"n": {"lt": 200}}}}}, "size": 30}, False),
    ("match operator and + from", {"query": {"match": {"t": {"query": "fox quick", "operator": "and"}}}, "from": 5, "size": 10}, False),
    ("hybrid query + knn", {"query": {"match": {"t": "fox dog"}}, "knn": K, "size": 20}, True),
    ("hybrid two knn clauses + bool", {"query": {"bool": {"should": [{"match": {"t": "quick"}}, {"match": {"u": "brown"}}]}}, "knn": [K, dict(K, query_vector=[-0.4, 0.9, 0.1], k=6, boost=0.5)], "size": 20}, True),
]
def hits(r): return [(h["_id"], h.get("_score")) for h in r["hits"]["hits"]] if "hits" in r else None
def same(a, b): return a is not None and b is not None and len(a) == len(b) and len(a) > 0 and all(x[0] == y[0] and abs(x[1] - y[1]) <= 1e-4 for x, y in zip(a, b))
def diff(a, b):
    i = next((i for i, (x, y) in enumerate(zip(a or [], b or [])) if x[0] != y[0] or abs(x[1] - y[1]) > 1e-4), min(len(a or []), len(b or [])))
    return f"\n  first difference at {i}\n  dv:  {(a or [])[max(0, i - 2):i + 5]}\n  dv1: {(b or [])[max(0, i - 2):i + 5]}"
def check(label, a, b): print(("DFS_OK " if same(a, b) else "DFS_BAD ") + label, len(a or []), "" if same(a, b) else diff(a, b))
for label, body, knn in cases:
    qs = "" if knn else f"?{DFS}"
    for port in (P0, P2):
        check(f"{label} via {port}", hits(call("POST", f"/dv/_search{qs}", body, port)), hits(call("POST", f"/dv1/_search{qs}", body, port)))
# Explicit search_type with knn: the same 400 on both, as ES.
a, b = call("POST", f"/dv/_search?{DFS}", cases[3][1]), call("POST", f"/dv1/_search?{DFS}", cases[3][1])
ok = a.get("status") == b.get("status") == 400 and a.get("err") == b.get("err")
print(("DFS_OK " if ok else "DFS_BAD ") + "search_type with knn rejected", a.get("status"), (a.get("err") or "")[:160], "" if ok else f"\n  dv1: {b}")
# query_then_fetch keeps per-shard stats: the skewed corpus scores differently, as in ES.
for label, body, knn in cases:
    if knn: continue
    a, b = hits(call("POST", "/dv/_search", body)), hits(call("POST", "/dv1/_search", body))
    print(("QTF_DIFFERS_OK " if a and b and not same(a, b) else "QTF_DIFFERS_BAD ") + f"{label} without dfs", a[:2] if a else a, b[:2] if b else b)
# msearch: search_type per header (knn entries without), and as a URL param (text entries).
def msearch(idx, url_dfs):
    nd, sel = "", [c for c in cases if not (url_dfs and c[2])]
    for label, body, knn in sel:
        hdr = {"index": idx} if url_dfs or knn else {"index": idx, "search_type": "dfs_query_then_fetch"}
        nd += json.dumps(hdr) + "\n" + json.dumps(body) + "\n"
    r = call("POST", "/_msearch" + (f"?{DFS}" if url_dfs else ""), nd=nd)
    return sel, ([hits(x) for x in r["responses"]] if "responses" in r else [None] * len(sel))
for url_dfs in (False, True):
    sel, ra = msearch("dv", url_dfs)
    _, rb = msearch("dv1", url_dfs)
    for (label, _, _), a, b in zip(sel, ra, rb):
        check(f"msearch ({'url' if url_dfs else 'header'}) {label}", a, b)
# Scroll (first and later pages), PIT, async search.
body = {"query": {"match": {"t": "fox dog quick"}}, "size": 15}
def scroll_all(idx):
    r = call("POST", f"/{idx}/_search?scroll=1m&{DFS}", body); out = hits(r) or []
    while r.get("hits", {}).get("hits"):
        r = call("POST", "/_search/scroll", {"scroll": "1m", "scroll_id": r["_scroll_id"]}); out += hits(r) or []
    call("DELETE", "/_search/scroll", {"scroll_id": r.get("_scroll_id", "")})
    return out
# Deep pages reach long docs whose BM25 norms quantize to the same byte, so
# equal scores tie; ES breaks ties by shard then doc, which legitimately
# differs between 1 and 3 shards. Compare with ties in id order.
def untie(h): return sorted(h, key=lambda x: (-round(x[1], 4), x[0]))
a, b = scroll_all("dv"), scroll_all("dv1")
check("scroll all pages (ties in id order)", untie(a), untie(b))
def pit(idx):
    p = call("POST", f"/{idx}/_pit?keep_alive=1m")
    r = hits(call("POST", f"/_search?{DFS}", dict(body, pit={"id": p["id"]}))) if "id" in p else None
    call("DELETE", "/_pit", {"id": p.get("id", "")}); return r
a, b = pit("dv"), pit("dv1")
check("pit", a, b)
def asy(idx):
    r = call("POST", f"/{idx}/_async_search?{DFS}&wait_for_completion_timeout=30s", body)
    return hits(r["response"]) if "response" in r else None
a, b = asy("dv"), asy("dv1")
check("async search", a, b)
# Stored async search (keep_on_completion): submit on node 1, fetch by id via
# node 3 (forwarded to the owner), then status + delete.
def asy_kept(idx):
    r = call("POST", f"/{idx}/_async_search?{DFS}&keep_on_completion=true&wait_for_completion_timeout=30s", body)
    if "id" not in r: return None, r
    g = call("GET", f"/_async_search/{r['id']}?wait_for_completion_timeout=30s", port=P2)
    st = call("GET", f"/_async_search/status/{r['id']}", port=P2)
    d = call("DELETE", f"/_async_search/{r['id']}", port=P2)
    gone = call("GET", f"/_async_search/{r['id']}")
    meta = (g.get("id") == r["id"], st.get("completion_status"), d.get("acknowledged"), gone.get("status"))
    return (hits(g["response"]) if "response" in g else None), meta
(a, ma), (b, mb) = asy_kept("dv"), asy_kept("dv1")
check("async search kept (GET by id)", a, b)
print(("DFS_OK " if ma == (True, 200, True, 404) else "DFS_BAD ") + "async kept id/status/delete", ma, mb)
EOF
"$C" stop
