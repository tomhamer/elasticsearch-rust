#!/usr/bin/env bash
# Kibana-against-rustic harness (run on the Linux build box).
#
#   tests/kibana/run.sh start   # build + start our server (:9201) and Kibana (:5601)
#   tests/kibana/run.sh ref     # start reference ES 8.19 (:9250) + Kibana (:5602)
#   tests/kibana/run.sh wait    # wait until Kibana(s) report status "available"
#   tests/kibana/run.sh errors  # summarize >=400 responses from our trace log
#   tests/kibana/run.sh restart # rebuild + restart our server (FRESH=1 wipes data) and its Kibana
#   tests/kibana/run.sh stop    # stop everything started here
#
# Env: KIBANA_HOME (default ~/kibana), KIBANA_VERSION (8.19.0),
#      ES_REF_TGZ (an elasticsearch-8.19.x linux tarball for `ref`),
#      CARGO_TARGET_DIR (default ~/targets/kibana).
set -euo pipefail
K="${KIBANA_HOME:-$HOME/kibana}"
KV="${KIBANA_VERSION:-8.19.0}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
TARGET="${CARGO_TARGET_DIR:-$HOME/targets/kibana}"
mkdir -p "$K"

kibana_dist() {
  if [ ! -x "$K/kibana-$KV/bin/kibana" ]; then
    curl -sfL -o "$K/kibana.tgz" "https://artifacts.elastic.co/downloads/kibana/kibana-$KV-linux-x86_64.tar.gz"
    tar xzf "$K/kibana.tgz" -C "$K"
  fi
}

start_kibana() { # name port es_url
  local name=$1 port=$2 es=$3 dir="$K/$1"
  mkdir -p "$dir/data"
  cat > "$dir/kibana.yml" <<YML
server.port: $port
server.host: "127.0.0.1"
server.name: "$name"
elasticsearch.hosts: ["$es"]
path.data: "$dir/data"
pid.file: "$dir/kibana.pid"
telemetry.optIn: false
telemetry.enabled: false
newsfeed.enabled: false
xpack.encryptedSavedObjects.encryptionKey: "kibana-compat-harness-key-0123456789abcdef"
xpack.reporting.encryptionKey: "kibana-compat-harness-key-0123456789abcdef"
xpack.security.encryptionKey: "kibana-compat-harness-key-0123456789abcdef"
logging:
  appenders:
    file:
      type: file
      fileName: "$dir/kibana.log"
      layout:
        type: json
  root:
    appenders: [file]
    level: info
  loggers:
    - name: elasticsearch.query
      level: debug
YML
  : > "$dir/kibana.log"
  KBN_PATH_CONF="$dir" setsid nohup "$K/kibana-$KV/bin/kibana" > "$dir/stdout.log" 2>&1 < /dev/null &
  echo "kibana $name on :$port -> $es"
}

case "${1:-}" in
  start)
    kibana_dist
    (cd "$REPO" && CARGO_TARGET_DIR="$TARGET" cargo build --release -q)
    rm -rf "$K/rustic-data"; mkdir -p "$K/rustic-data"
    : > "$K/trace.ndjson"
    ES_DATA_DIR="$K/rustic-data" ES_HTTP_ADDR=127.0.0.1:9201 ES_TRACE_HTTP="$K/trace.ndjson" ES_TRACE_HTTP_MAX=20000 \
      setsid nohup "$TARGET/release/elasticsearch" -E xpack.security.enabled=false > "$K/rustic.log" 2>&1 < /dev/null &
    echo $! > "$K/rustic.pid"
    sleep 1
    curl -sf localhost:9201/ > /dev/null && echo "rustic on :9201"
    start_kibana kbn-rustic 5601 http://127.0.0.1:9201
    ;;
  restart)
    (cd "$REPO" && CARGO_TARGET_DIR="$TARGET" cargo build --release -q 2>/dev/null)
    [ -f "$K/kbn-rustic/kibana.pid" ] && kill "$(cat "$K/kbn-rustic/kibana.pid")" 2>/dev/null || true
    [ -f "$K/rustic.pid" ] && kill "$(cat "$K/rustic.pid")" 2>/dev/null || true
    sleep 2
    if [ "${FRESH:-0}" = 1 ]; then rm -rf "$K/rustic-data"; mkdir -p "$K/rustic-data"; rm -rf "$K/kbn-rustic/data"; fi
    : > "$K/trace.ndjson"
    ES_DATA_DIR="$K/rustic-data" ES_HTTP_ADDR=127.0.0.1:9201 ES_TRACE_HTTP="$K/trace.ndjson" ES_TRACE_HTTP_MAX=20000 \
      setsid nohup "$TARGET/release/elasticsearch" -E xpack.security.enabled=false > "$K/rustic.log" 2>&1 < /dev/null &
    echo $! > "$K/rustic.pid"
    sleep 1
    curl -sf localhost:9201/ > /dev/null && echo "rustic on :9201"
    start_kibana kbn-rustic 5601 http://127.0.0.1:9201
    ;;
  ref)
    kibana_dist
    if [ ! -x "$K/es-ref/bin/elasticsearch" ]; then
      mkdir -p "$K/es-ref" && tar xzf "${ES_REF_TGZ:?set ES_REF_TGZ}" -C "$K/es-ref" --strip-components=1
    fi
    if ! curl -sf localhost:9250/ > /dev/null; then
      (cd "$K/es-ref" && ES_JAVA_OPTS="-Xms1g -Xmx1g" setsid nohup ./bin/elasticsearch -Expack.security.enabled=false \
        -Ehttp.port=9250 -Etransport.port=9350 -Ediscovery.type=single-node -Ecluster.name=ref -Expack.ml.enabled=false \
        > "$K/es-ref.out" 2>&1 < /dev/null &)
      until curl -sf localhost:9250/ > /dev/null; do sleep 2; done
    fi
    start_kibana kbn-ref 5602 http://127.0.0.1:9250
    ;;
  wait)
    for p in ${2:-5601}; do
      for i in $(seq 1 120); do
        s=$(curl -s "localhost:$p/api/status" | python3 -c 'import sys,json; print(json.load(sys.stdin)["status"]["overall"]["level"])' 2>/dev/null || true)
        [ "$s" = "available" ] && { echo "kibana :$p available"; break; }
        sleep 2
      done
      [ "$s" = "available" ] || { echo "kibana :$p NOT available (last: ${s:-none})"; exit 1; }
    done
    ;;
  errors)
    python3 - "$K/trace.ndjson" <<'PY'
import json, sys, re, collections
c = collections.Counter(); ex = {}
for line in open(sys.argv[1]):
    try: r = json.loads(line)
    except Exception: continue
    if r["status"] < 400: continue
    path = re.sub(r"\?.*", "", r["uri"])
    path = re.sub(r"/[A-Za-z0-9_\-=]{20,}", "/{id}", path)
    k = (r["method"], path, r["status"]); c[k] += 1; ex.setdefault(k, r["resp"][:300])
for k, n in c.most_common():
    print(n, *k, "|", ex[k])
PY
    ;;
  stop)
    for d in kbn-rustic kbn-ref; do
      [ -f "$K/$d/kibana.pid" ] && kill "$(cat "$K/$d/kibana.pid")" 2>/dev/null || true
    done
    pkill -f "$K/kibana-$KV/node" 2>/dev/null || true
    [ -f "$K/rustic.pid" ] && kill "$(cat "$K/rustic.pid")" 2>/dev/null || true
    pkill -f "$K/es-ref/" 2>/dev/null || true
    echo stopped
    ;;
  *) sed -n 2,12p "$0"; exit 2 ;;
esac
