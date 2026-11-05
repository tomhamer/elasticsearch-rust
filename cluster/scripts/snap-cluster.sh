#!/usr/bin/env bash
# 3-node cluster for the snapshot tests (ports 9500-9590 only).
#   snap-cluster.sh start        start all nodes (fresh data + repo dirs)
#   snap-cluster.sh stop         stop all nodes (by pid file)
#   snap-cluster.sh kill N       kill -9 node N
#   snap-cluster.sh up N         (re)start node N keeping its data
#   snap-cluster.sh urls
# Node i: HTTP PORT_BASE+i-1, transport PORT_BASE+10+i-1.
set -euo pipefail
PORT_BASE="${PORT_BASE:-9500}"
N="${NODES:-3}"
TAG="${TAG:-snap}"
DIR="${CLUSTER_DIR:-/tmp/snap-esn-$TAG}"
REPO="${REPO_DIR:-/tmp/snap-repo-$TAG}"
BIN="${BIN:-$(cd "$(dirname "$0")/../.." && pwd)/target/debug/elasticsearch}"

seeds=""; masters=""
for i in $(seq 1 "$N"); do
  seeds+="127.0.0.1:$((PORT_BASE + 10 + i - 1)),"
  masters+="node-$i,"
done

up() {
  local i="$1"
  mkdir -p "$DIR/conf-none" "$REPO"
  ES_PATH_CONF="$DIR/conf-none" nohup "$BIN" \
    -E cluster.name="esn-$TAG" -E node.name="node-$i" \
    -E path.data="$DIR/node-$i" -E path.repo="$REPO" \
    -E xpack.license.self_generated.type="${ES_LICENSE:-trial}" \
    -E xpack.security.enabled=false \
    -E network.host=127.0.0.1 \
    -E http.port=$((PORT_BASE + i - 1)) -E transport.port=$((PORT_BASE + 10 + i - 1)) \
    -E discovery.seed_hosts="${seeds%,}" -E cluster.initial_master_nodes="${masters%,}" \
    -E node.attr.testattr=test \
    -E cluster.routing.allocation.disk.watermark.low=1b \
    -E cluster.routing.allocation.disk.watermark.high=1b \
    -E cluster.routing.allocation.disk.watermark.flood_stage=1b \
    >>"$DIR/node-$i.log" 2>&1 &
  echo $! >"$DIR/node-$i.pid"
}

stop_all() {
  for p in "$DIR"/node-*.pid; do
    [[ -f "$p" ]] && kill "$(cat "$p")" 2>/dev/null || true
  done
  sleep 1
  for p in "$DIR"/node-*.pid; do
    [[ -f "$p" ]] && kill -9 "$(cat "$p")" 2>/dev/null || true
  done
  rm -f "$DIR"/node-*.pid
}

case "${1:-}" in
  start)
    [[ -d "$DIR" ]] && stop_all
    rm -rf "$DIR" "$REPO"; mkdir -p "$DIR" "$REPO"
    for i in $(seq 1 "$N"); do up "$i"; done
    for _ in $(seq 1 120); do
      n=$(curl -s "127.0.0.1:$PORT_BASE/_cat/nodes" 2>/dev/null | wc -l || true)
      [[ "$n" -ge "$N" ]] && { echo "cluster up ($n nodes)"; exit 0; }
      sleep 0.5
    done
    echo "cluster did not form" >&2; exit 1 ;;
  stop) stop_all ;;
  kill) kill -9 "$(cat "$DIR/node-$2.pid")" 2>/dev/null || true; rm -f "$DIR/node-$2.pid" ;;
  up) up "$2" ;;
  urls)
    u=""; for i in $(seq 1 "$N"); do u+="http://127.0.0.1:$((PORT_BASE + i - 1)),"; done
    echo "${u%,}" ;;
  *) echo "usage: $0 start|stop|kill N|up N|urls" >&2; exit 2 ;;
esac
