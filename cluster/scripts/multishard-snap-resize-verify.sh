#!/usr/bin/env bash
# Multi-shard snapshot/restore, _split/_shrink/_clone, _reindex /
# _update_by_query / _delete_by_query and _stats / _cat/indices store size on
# a local 3-node cluster (HTTP 10400-10402, transport 10410-10412).
# Prints PASS/FAIL per check; exit status = number of failures.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PORT_BASE="${PORT_BASE:-10400}" TAG=msrr
C="$HERE/snap-cluster.sh"

"$C" start || exit 1
python3 - "$PORT_BASE" <<'EOF'
import json, sys, time, urllib.request
base = int(sys.argv[1]); fails = 0
def call(m, p, b=None, node=0, nd=False):
    data = None
    if b is not None:
        data = b.encode() if nd else json.dumps(b).encode()
    ct = "application/x-ndjson" if nd else "application/json"
    rq = urllib.request.Request(f"http://127.0.0.1:{base + node}{p}", method=m, data=data, headers={"content-type": ct})
    try:
        r = urllib.request.urlopen(rq, timeout=120); return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try: return e.code, json.loads(e.read())
        except Exception: return e.code, {}
def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  -- {str(detail)[:400]}"))
    if not ok: fails += 1
    return ok
def green(idx):
    return call("GET", f"/_cluster/health/{idx}?wait_for_status=green&timeout=60s")[1].get("status") == "green"
def count(idx):
    call("POST", f"/{idx}/_refresh")
    return call("GET", f"/{idx}/_count")[1].get("count")
def all_found(idx, n):
    s, r = call("POST", f"/{idx}/_mget", {"docs": [{"_id": str(i), **({"routing": f"r{i % 7}"} if i % 5 == 0 else {})} for i in range(n)]}, node=1)
    return sum(1 for d in r.get("docs", []) if d.get("found"))

N = 300
for _ in range(60):
    if call("GET", "/_cluster/health")[0] == 200: break
    time.sleep(1)
call("PUT", "/src", {"settings": {"number_of_shards": 3, "number_of_replicas": 1},
                     "mappings": {"properties": {"n": {"type": "integer"}, "tag": {"type": "keyword"}}}})
check("src green", green("src"))
lines = []
for i in range(N):
    a = {"_index": "src", "_id": str(i)}
    if i % 5 == 0: a["routing"] = f"r{i % 7}"
    lines += [json.dumps({"index": a}), json.dumps({"n": i, "tag": f"t{i % 10}"})]
call("POST", "/_bulk?refresh=true", "\n".join(lines) + "\n", node=2, nd=True)
check("src count", count("src") == N, count("src"))

# ---- (4) _stats / _cat/indices store size summed across shards
s, st = call("GET", "/src/_stats/store,docs?level=shards")
per = [c["store"]["size_in_bytes"] for sh in st["indices"]["src"]["shards"].values() for c in sh if c["routing"]["primary"]]
pri = st["indices"]["src"]["primaries"]["store"]["size_in_bytes"]
tot = st["indices"]["src"]["total"]["store"]["size_in_bytes"]
check("stats primaries store = sum of 3 primary shards", len(per) == 3 and pri == sum(per) and pri > max(per), (per, pri))
check("stats total store >= primaries (replicas)", tot >= pri and tot > 0, (tot, pri))
check("stats _all primaries store", st["_all"]["primaries"]["store"]["size_in_bytes"] == pri, st["_all"]["primaries"]["store"])
check("stats docs primaries", st["indices"]["src"]["primaries"]["docs"]["count"] == N, st["indices"]["src"]["primaries"]["docs"])
s, cat = call("GET", "/_cat/indices/src?format=json&bytes=b&h=index,docs.count,store.size,pri.store.size")
check("_cat/indices store.size = stats total", cat and int(cat[0]["store.size"]) == tot and int(cat[0]["pri.store.size"]) == pri, (cat, tot, pri))

# ---- (3) _reindex / _update_by_query / _delete_by_query on multi-shard sources
call("PUT", "/dst", {"settings": {"number_of_shards": 2, "number_of_replicas": 0}})
green("dst")
s, r = call("POST", "/_reindex?refresh=true", {"source": {"index": "src"}, "dest": {"index": "dst"}})
check("reindex 3sh->2sh created", s == 200 and r.get("created") == N and r.get("total") == N, (s, r))
check("reindex dst count + routed gets", count("dst") == N and all_found("dst", N) == N, (count("dst"), all_found("dst", N)))
s, r = call("POST", "/_reindex?refresh=true", {"source": {"index": "src", "query": {"term": {"tag": "t3"}}}, "dest": {"index": "dst1"},
                                               "script": {"source": "ctx._source.n += 1000"}})
check("reindex query+script into new index", s == 200 and r.get("created") == N // 10, (s, r))
s, r = call("POST", "/_reindex", {"source": {"index": "src"}, "dest": {"index": "dst", "op_type": "create"}, "conflicts": "proceed"})
check("reindex op_type=create conflicts=proceed", s == 200 and r.get("version_conflicts") == N and r.get("created") == 0, (s, r))
s, r = call("POST", "/src/_update_by_query?refresh=true", {"query": {"term": {"tag": "t1"}}, "script": {"source": "ctx._source.n += 10000"}})
check("update_by_query across shards", s == 200 and r.get("updated") == N // 10, (s, r))
s, r = call("POST", "/src/_count", {"query": {"range": {"n": {"gte": 10000}}}})
check("update_by_query visible", r.get("count") == N // 10, r)
call("PUT", "/dbq", {"settings": {"number_of_shards": 3, "number_of_replicas": 1}})
green("dbq")
call("POST", "/_reindex?refresh=true", {"source": {"index": "src"}, "dest": {"index": "dbq"}})
# n < 100 matches 90 docs: the t1 ones were moved by the UBQ above.
s, r = call("POST", "/dbq/_delete_by_query?refresh=true", {"query": {"range": {"n": {"lt": 100}}}})
check("delete_by_query across shards", s == 200 and r.get("deleted") == 90 and count("dbq") == N - 90, (s, r, count("dbq")))

# ---- (3b) wait_for_completion=false: cluster-run tasks over every shard
node_ids = set(call("GET", "/_nodes")[1].get("nodes", {}).keys())
def shard_docs(idx):
    call("POST", f"/{idx}/_refresh")
    st = call("GET", f"/{idx}/_stats/docs?level=shards")[1]
    return sorted(c["docs"]["count"] for sh in st["indices"][idx]["shards"].values() for c in sh if c["routing"]["primary"])
def run_task(p, b, key, want, node=0):
    s, r = call("POST", p, b, node=node)
    tid = r.get("task", "")
    owner = tid.split(":")[0]
    check(f"async {key}: task id node:n of a cluster node", s == 200 and owner in node_ids and tid.split(":")[-1].isdigit(), (s, r, node_ids))
    s, t = call("GET", f"/_tasks/{tid}?wait_for_completion=true&timeout=60s", node=(node + 1) % 3)
    resp = t.get("response", {})
    check(f"async {key}: _tasks result ({key}={want})", s == 200 and t.get("completed") is True and t.get("task", {}).get("node") == owner
          and resp.get(key) == want and resp.get("failures") == [], (s, t))
call("PUT", "/adst", {"settings": {"number_of_shards": 3, "number_of_replicas": 0}})
green("adst")
run_task("/_reindex?wait_for_completion=false&refresh=true", {"source": {"index": "src"}, "dest": {"index": "adst"}}, "created", N)
per = shard_docs("adst")
check("async reindex: docs on all 3 shards + routed gets", len(per) == 3 and min(per) > 0 and sum(per) == N and all_found("adst", N) == N, (per, all_found("adst", N)))
run_task("/adst/_update_by_query?wait_for_completion=false&refresh=true", {"query": {"term": {"tag": "t2"}}, "script": {"source": "ctx._source.n += 20000"}}, "updated", N // 10, node=1)
check("async update_by_query visible", call("POST", "/adst/_count", {"query": {"range": {"n": {"gte": 20000}}}})[1].get("count") == N // 10)
# n in [200, 10000): 200..299 minus the t1/t2 docs moved above = 80.
run_task("/adst/_delete_by_query?wait_for_completion=false&refresh=true", {"query": {"range": {"n": {"gte": 200, "lt": 10000}}}}, "deleted", 80, node=2)
check("async delete_by_query count", count("adst") == N - 80, count("adst"))

# ---- (3c) multi-shard dest created by an index template, 1-shard source
call("PUT", "/one", {"settings": {"number_of_shards": 1, "number_of_replicas": 0}})
green("one")
call("POST", "/_reindex?refresh=true", {"source": {"index": "src"}, "dest": {"index": "one"}})
call("PUT", "/_index_template/tdst", {"index_patterns": ["tdst*"], "template": {"settings": {"number_of_shards": 3, "number_of_replicas": 0}}})
s, r = call("POST", "/_reindex?refresh=true", {"source": {"index": "one"}, "dest": {"index": "tdst1"}})
check("reindex 1sh -> template 3sh dest created", s == 200 and r.get("created") == N, (s, r))
s, r = call("GET", "/tdst1/_settings")
check("template dest number_of_shards=3", r.get("tdst1", {}).get("settings", {}).get("index", {}).get("number_of_shards") == "3", r)
per = shard_docs("tdst1")
check("template dest: docs on all 3 shards + routed gets", len(per) == 3 and min(per) > 0 and sum(per) == N and all_found("tdst1", N) == N, (per, all_found("tdst1", N)))

# ---- (1) snapshot / restore of a 3-shard index
call("PUT", "/_snapshot/r", {"type": "fs", "settings": {"location": "r"}})
s, r = call("PUT", "/_snapshot/r/s1?wait_for_completion=true", {"indices": "src", "include_global_state": False})
sh = r.get("snapshot", {}).get("shards", {})
check("snapshot shards total=3 successful=3", s == 200 and sh.get("total") == 3 and sh.get("successful") == 3, (s, r))
s, r = call("GET", "/_snapshot/r/s1")
check("get snapshot shards total=3", r.get("snapshots", [{}])[0].get("shards", {}).get("total") == 3, r)
s, r = call("GET", "/_snapshot/r/s1/_status")
ss = r.get("snapshots", [{}])[0]
idx = ss.get("indices", {}).get("src", {})
check("status shards_stats total=3 done=3", ss.get("shards_stats", {}).get("total") == 3 and ss.get("shards_stats", {}).get("done") == 3, ss.get("shards_stats"))
check("status per-shard entries 0,1,2", sorted(idx.get("shards", {}).keys()) == ["0", "1", "2"]
      and all(v.get("stats", {}).get("total", {}).get("file_count", 0) > 0 for v in idx.get("shards", {}).values()), idx.get("shards"))
call("DELETE", "/src")
s, r = call("POST", "/_snapshot/r/s1/_restore?wait_for_completion=true", {"indices": "src"})
check("restore shards total=3", s == 200 and r.get("snapshot", {}).get("shards", {}).get("total") == 3, (s, r))
check("restored green", green("src"))
s, r = call("GET", "/src/_settings/index.number_of_shards?flat_settings=true")
check("restored number_of_shards=3", r.get("src", {}).get("settings", {}).get("index.number_of_shards") == "3", r)
check("restored count + routed gets", count("src") == N and all_found("src", N) == N, (count("src"), all_found("src", N)))
s, r = call("POST", "/src/_count", {"query": {"range": {"n": {"gte": 10000}}}})
check("restored keeps UBQ updates", r.get("count") == N // 10, r)

# ---- (2) _split / _shrink / _clone
call("PUT", "/src/_settings", {"index.blocks.write": True})
for tgt, n in (("split6", 6), ("clone3", 3), ("shrink1", 1)):
    verb = {"split6": "_split", "clone3": "_clone", "shrink1": "_shrink"}[tgt]
    s, r = call("POST", f"/src/{verb}/{tgt}", {"settings": {"index.number_of_shards": n, "index.number_of_replicas": 1}})
    check(f"{verb} -> {n} acknowledged", s == 200 and r.get("acknowledged"), (s, r))
    green(tgt)
    s, r = call("GET", f"/{tgt}/_settings/index.number_of_shards?flat_settings=true")
    check(f"{verb} number_of_shards={n}", r.get(tgt, {}).get("settings", {}).get("index.number_of_shards") == str(n), r)
    s, r = call("GET", f"/{tgt}/_search_shards")
    check(f"{verb} routing table has {n} shards", len(r.get("shards", [])) == n, len(r.get("shards", [])))
    ok = check(f"{verb} count + routed gets", count(tgt) == N and all_found(tgt, N) == N, (count(tgt), all_found(tgt, N)))
    if not ok:
        print("   cat:", call("GET", f"/_cat/shards/{tgt}?h=shard,prirep,state,node,docs&format=json")[1])
# Split 3 -> 9: 9 is a multiple of 3, but not a factor of the 768 routing
# shards; ES throws IllegalStateException from selectSplitShard (HTTP 500).
for verb, n, code, want in (("_split", 9, 500, "illegal_state_exception"), ("_split", 2, 400, "illegal_argument_exception"),
                            ("_shrink", 2, 400, "illegal_argument_exception"), ("_clone", 4, 400, "illegal_argument_exception")):
    s, r = call("POST", f"/src/{verb}/bad{n}{verb}", {"settings": {"index.number_of_shards": n}})
    check(f"{verb} -> {n} rejected {code}", s == code and r.get("error", {}).get("type") == want, (s, r))

print(f"failures: {fails}")
sys.exit(min(fails, 100))
EOF
rc=$?
"$C" stop
exit $rc
