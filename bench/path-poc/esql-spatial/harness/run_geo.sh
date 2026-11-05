#!/usr/bin/env bash
# 1%-data / 1%-query Rally across all pilot tracks, one engine at a time.
# Server on CPUs 0-13, Rally on 14-15, matching the 2026-10-12 pilot.
set -u
engine="${1:?engine: es9 or rust}"
tracks="${TRACKS:-metricbeat vectors100k percolator geonames nested geopoint}"
root="$HOME/rally-1pct"
res="$HOME/esql-rally/${RES_TAG:-$engine}"
data="$HOME/esql-rally/server-data-${RES_TAG:-$engine}"
rm -rf "$res" "$data"; mkdir -p "$res" "$data"
if [ "$engine" = es9 ]; then
  conf="$res/es9-conf"; cp -r "$HOME/ref/elasticsearch-9.5.5/config" "$conf"
  cat > "$conf/elasticsearch.yml" <<YML
cluster.name: rally-1pct
node.name: es9-0
network.host: 127.0.0.1
http.port: 19200
transport.port: 19300
discovery.type: single-node
path.data: $data
path.logs: $res/es9-logs
xpack.security.enabled: false
YML
  ES_PATH_CONF="$conf" ES_JAVA_OPTS="-Xms16g -Xmx16g" setsid taskset -c 0-13 \
    "$HOME/ref/elasticsearch-9.5.5/bin/elasticsearch" > "$res/server.log" 2>&1 < /dev/null &
  sha256sum "$HOME/ref/elasticsearch-9.5.5/bin/elasticsearch" > "$res/server.sha256"
else
  bin="${RUST_BIN:-$HOME/es9-perf/target/release/elasticsearch}"
  sha256sum "$bin" > "$res/server.sha256"
  setsid taskset -c 0-13 "$bin" -E cluster.name=rally-1pct -E node.name=rust-0 \
    -E network.host=127.0.0.1 -E http.port=19200 -E transport.port=19300 \
    -E discovery.type=single-node -E path.data="$data" -E xpack.security.enabled=false \
    > "$res/server.log" 2>&1 < /dev/null &
fi
spid=$!; echo "$spid" > "$res/server.pid"
for i in $(seq 1 180); do curl -sf 'localhost:19200/_cluster/health?wait_for_status=yellow&timeout=1s' >/dev/null && break; sleep 1; done
curl -sf 'localhost:19200/_cluster/health?wait_for_status=yellow&timeout=1s' >/dev/null || { echo "server not healthy" > "$res/server.failed"; kill "$spid" 2>/dev/null; echo "FATAL: $engine server not healthy" >&2; exit 1; }
curl -sS 'localhost:19200/' > "$res/root.json"
# geopoint's ES|QL polygon task needs a non-basic licence on ES9; apply the same trial to both.
curl -sS -XPOST 'localhost:19200/_license/start_trial?acknowledge=true' > "$res/license.json"
for track in $tracks; do
  python3 - <<'PY' > "$res/$track.cleanup.json"
import json, urllib.request
base = "http://127.0.0.1:19200"
for index in json.load(urllib.request.urlopen(base + "/_cat/indices?format=json")):
    if not index["index"].startswith("."):
        urllib.request.urlopen(urllib.request.Request(base + "/" + index["index"], method="DELETE")).read()
PY
  start=$(date +%s)
  taskset -c 14,15 "$HOME/rally-venv/bin/esrally" race --pipeline=benchmark-only \
    --target-hosts=127.0.0.1:19200 --track-path="${TRACK_ROOT:-$root/tracks}/$track" \
    ${RALLY_EXTRA:-} --report-format=csv --report-file="$res/$track.csv" \
    --user-tags="engine:$engine,data:0.01,queries:0.01" --on-error=continue \
    > "$res/$track.log" 2>&1
  echo "$? $(( $(date +%s) - start ))" > "$res/$track.exit"
  python3 "$HOME/rally-pilot/audit_queries.py" "$root/tracks/$track" > "$res/$track.audit.json" 2> "$res/$track.audit-errors.log"
  curl -sS 'localhost:19200/_cat/indices?format=json' > "$res/$track.indices.json"
  curl -sS 'localhost:19200/_all/_count' > "$res/$track.count.json"
  kill -0 "$spid" 2>/dev/null || { echo "server died during $track" > "$res/server.died"; break; }
done
ps -o rss= -p "$spid" > "$res/server-final-rss-kb.txt" 2>/dev/null
grep -E "VmHWM|VmRSS" /proc/$spid/status > "$res/server-mem.txt" 2>/dev/null; for p in $(pgrep -P $spid); do grep -E "VmHWM|VmRSS" /proc/$p/status >> "$res/server-mem.txt"; done
kill "$spid"; sleep 10; kill -9 "$spid" 2>/dev/null
rm -rf "$data"
touch "$res/suite.done"
