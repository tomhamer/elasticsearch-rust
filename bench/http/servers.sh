#!/usr/bin/env bash
# Start/stop the two servers compared by bench/http/run.sh (Linux).
#
#   servers.sh start-es | stop-es | start-rustic <binary> | stop-rustic | status
#
# Both servers are pinned to the same CPU set ($SERVER_CPUS, default 0-7) and
# the load generator to another ($CLIENT_CPUS, see run.sh) so neither steals
# cycles from the other. Real Elasticsearch: official 8.19.x tarball with its
# bundled JDK, security off, single node, $ES_HEAP heap (default 8g).
set -euo pipefail
ROOT="${HTTPPERF_HOME:-$HOME/httpperf}"
ES_VERSION="${ES_VERSION:-8.19.22}"
ES_HOME="$ROOT/elasticsearch-$ES_VERSION"
ES_PORT="${ES_PORT:-9211}"
RUSTIC_PORT="${RUSTIC_PORT:-9212}"
SERVER_CPUS="${SERVER_CPUS:-0-7}"
ES_HEAP="${ES_HEAP:-8g}"

install_es() {
  [ -d "$ES_HOME" ] && return
  mkdir -p "$ROOT" && cd "$ROOT"
  local tgz="elasticsearch-$ES_VERSION-linux-x86_64.tar.gz"
  curl -sSfO "https://artifacts.elastic.co/downloads/elasticsearch/$tgz"
  curl -sSfO "https://artifacts.elastic.co/downloads/elasticsearch/$tgz.sha512"
  sha512sum -c "$tgz.sha512"
  tar xzf "$tgz"
}

wait_port() {
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$1/" >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "server on :$1 did not come up" >&2; return 1
}

case "${1:-}" in
  start-es)
    install_es
    cat > "$ES_HOME/config/elasticsearch.yml" <<YML
cluster.name: httpperf
node.name: es-bench
path.data: $ROOT/es-data
path.logs: $ROOT/es-logs
network.host: 127.0.0.1
http.port: $ES_PORT
transport.port: 9311
discovery.type: single-node
xpack.security.enabled: false
xpack.ml.enabled: false
xpack.watcher.enabled: false
ingest.geoip.downloader.enabled: false
cluster.routing.allocation.disk.threshold_enabled: false
action.destructive_requires_name: false
YML
    printf -- "-Xms%s\n-Xmx%s\n" "$ES_HEAP" "$ES_HEAP" > "$ES_HOME/config/jvm.options.d/heap.options"
    rm -rf "$ROOT/es-data"
    ES_JAVA_HOME= JAVA_HOME= taskset -c "$SERVER_CPUS" "$ES_HOME/bin/elasticsearch" -d -p "$ROOT/es.pid" >"$ROOT/es-stdout.log" 2>&1
    wait_port "$ES_PORT"
    curl -s "http://127.0.0.1:$ES_PORT/" | grep '"number"'
    ;;
  stop-es)
    [ -f "$ROOT/es.pid" ] && kill "$(cat "$ROOT/es.pid")" 2>/dev/null || true
    for _ in $(seq 1 30); do pgrep -f "es.path.home=$ES_HOME" >/dev/null || break; sleep 1; done
    rm -f "$ROOT/es.pid"
    ;;
  start-rustic)
    bin="${2:?path to the elasticsearch (rustic) release binary}"
    rm -rf "$ROOT/rustic-data"
    ES_DATA_DIR="$ROOT/rustic-data" ES_HTTP_ADDR="127.0.0.1:$RUSTIC_PORT" RUST_LOG=warn \
      nohup taskset -c "$SERVER_CPUS" "$bin" -E xpack.security.enabled=false >"$ROOT/rustic.log" 2>&1 &
    echo $! > "$ROOT/rustic.pid"
    wait_port "$RUSTIC_PORT"
    ;;
  stop-rustic)
    [ -f "$ROOT/rustic.pid" ] && kill "$(cat "$ROOT/rustic.pid")" 2>/dev/null || true
    rm -f "$ROOT/rustic.pid"
    ;;
  status)
    curl -s "http://127.0.0.1:$ES_PORT/" | grep '"number"' || echo "es: down"
    curl -s "http://127.0.0.1:$RUSTIC_PORT/" | grep '"number"' || echo "rustic: down"
    ;;
  *) echo "usage: $0 start-es|stop-es|start-rustic <bin>|stop-rustic|status" >&2; exit 2 ;;
esac
