#!/usr/bin/env bash
# Multi-shard check on a local 3-node cluster (HTTP 9300-9302, transport
# 9310-9312): 3 shards x 1 replica, 1000-doc bulk, GET every id, count,
# _cat/shards, then kill -9 one node and re-check.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PORT_BASE=9300 TAG=multishard
C="$HERE/snap-cluster.sh"
H=127.0.0.1
N=1000

"$C" start || exit 1
curl -s -XPUT "$H:9300/ms" -H 'content-type: application/json' \
  -d '{"settings":{"number_of_shards":3,"number_of_replicas":1}}' | head -c 300; echo
curl -s "$H:9300/_cluster/health/ms?wait_for_status=green&timeout=60s" | head -c 600; echo

python3 - "$N" >/tmp/multishard-bulk.ndjson <<'EOF'
import json, sys
for i in range(int(sys.argv[1])):
    print(json.dumps({"index": {"_index": "ms", "_id": str(i)}}))
    print(json.dumps({"n": i, "t": f"doc {i}"}))
EOF
curl -s -XPOST "$H:9301/_bulk?refresh=true" -H 'content-type: application/x-ndjson' \
  --data-binary @/tmp/multishard-bulk.ndjson | python3 -c '
import json,sys,collections
r=json.load(sys.stdin); c=collections.Counter()
for it in r["items"]:
    x=it["index"]; c[(x.get("status"), x.get("_shards",{}).get("successful"))]+=1
print("bulk errors:", r["errors"], "items:", len(r["items"]), "(status, _shards.successful):", dict(c))'

check() {
  echo "--- $1"
  for p in "${@:2}"; do
    echo -n "count via :$p "; curl -s "$H:$p/ms/_count" | head -c 300; echo
  done
  curl -s "$H:$2/ms/_search?size=0&track_total_hits=true" | python3 -c '
import json,sys; r=json.load(sys.stdin); print("search total:", r["hits"]["total"], "_shards:", r["_shards"])'
  python3 - "$2" "${@:2}" <<'EOF'
import json, sys, urllib.request
ports = sys.argv[2:]
found = missing = 0
for i in range(1000):
    p = ports[i % len(ports)]
    try:
        r = json.load(urllib.request.urlopen(f"http://127.0.0.1:{p}/ms/_doc/{i}"))
        ok = r.get("found") and r["_source"]["n"] == i
    except Exception:
        ok = False
    found += bool(ok); missing += (not ok)
print(f"GET by id: found={found} missing={missing}")
EOF
  echo "_cat/shards:"; curl -s "$H:$2/_cat/shards/ms?v&h=index,shard,prirep,state,docs,node&s=shard,prirep"
  curl -s "$H:$2/_cluster/health/ms" | head -c 600; echo
}

check "3 nodes" 9300 9301 9302
"$C" kill 3
echo "killed node-3"
curl -s "$H:9300/_cluster/health/ms?wait_for_status=yellow&wait_for_no_initializing_shards=true&timeout=60s" | head -c 600; echo
check "after kill -9 node-3" 9300 9301
"$C" up 3
curl -s "$H:9300/_cluster/health/ms?wait_for_status=green&timeout=90s" | head -c 600; echo
check "node-3 rejoined" 9300 9301 9302
"$C" stop
