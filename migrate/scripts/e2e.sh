#!/usr/bin/env bash
# End-to-end check of the upgrade path against a real Elasticsearch 8.x:
#   1. start a fresh ES node and load fixtures (fixtures.py records ground truth)
#   2. stop it cleanly (ES flushes on shutdown)
#   3. dump the data dir with es-lucene-import and diff every doc
#   4. import into the Rust server and diff every doc again
#
# usage: ES_HOME=<es-8.x dir> RUST_ES=<elasticsearch binary> IMPORTER=<es-lucene-import> scripts/e2e.sh [workdir]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
W=${1:-$HOME/lucimport}
: "${ES_HOME:?set ES_HOME to an unpacked Elasticsearch 8.x}"
: "${RUST_ES:?set RUST_ES to the Rust elasticsearch binary}"
: "${IMPORTER:?set IMPORTER to the es-lucene-import binary}"
ES_PORT=${ES_PORT:-9871}
RUST_PORT=${RUST_PORT:-9872}

rm -rf "$W" && mkdir -p "$W"
cp -r "$ES_HOME/config" "$W/config"
cat > "$W/config/elasticsearch.yml" <<EOF
cluster.name: lucimport
path.data: $W/es-data
path.logs: $W/es-logs
network.host: 127.0.0.1
http.port: $ES_PORT
transport.port: $((ES_PORT + 100))
discovery.type: single-node
xpack.security.enabled: false
xpack.ml.enabled: false
ingest.geoip.downloader.enabled: false
cluster.routing.allocation.disk.threshold_enabled: false
EOF
printf -- "-Xms1g\n-Xmx1g\n" > "$W/config/jvm.options.d/heap.options"

echo "== starting Elasticsearch"
ES_PATH_CONF="$W/config" setsid nohup "$ES_HOME/bin/elasticsearch" -p "$W/es.pid" > "$W/es.out" 2>&1 < /dev/null &
for _ in $(seq 1 90); do curl -sf "localhost:$ES_PORT" > /dev/null && break; sleep 2; done
python3 "$here/fixtures.py" "http://localhost:$ES_PORT" "$W/expected.json"

echo "== stopping Elasticsearch (flushes on shutdown)"
es_pid=$(cat "$W/es.pid")   # ES removes the pid file early in shutdown: wait on the process
kill "$es_pid"
while kill -0 "$es_pid" 2> /dev/null; do sleep 1; done

echo "== dump + diff"
"$IMPORTER" dump --data "$W/es-data" --out "$W/dump" > "$W/dump.log"
python3 "$here/compare_dump.py" "$W/expected.json" --dump "$W/dump"

echo "== import into the Rust server + diff"
ES_HTTP_ADDR="127.0.0.1:$RUST_PORT" ES_DATA_DIR="$W/rust-data" setsid nohup "$RUST_ES" -E xpack.security.enabled=false > "$W/rust.out" 2>&1 < /dev/null &
rust_pid=$!
trap 'kill $rust_pid 2>/dev/null || true' EXIT
for _ in $(seq 1 30); do curl -sf "localhost:$RUST_PORT" > /dev/null && break; sleep 1; done
"$IMPORTER" import --data "$W/es-data" --target "http://127.0.0.1:$RUST_PORT" | tee "$W/import.log" | sed -n '/^verify:/,$p'
python3 "$here/compare_dump.py" "$W/expected.json" --server "http://127.0.0.1:$RUST_PORT"
echo "== PASS"
