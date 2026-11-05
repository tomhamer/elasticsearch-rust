#!/usr/bin/env bash
# Start / stop an N-node cluster configured like ES's REST-test clusters
# (`test/test-clusters` DefaultSettingsProvider: `node.attr.testattr=test`,
# `path.repo`, tiny absolute disk watermarks), for running the YAML suite.
#
#   test-cluster.sh start [N=2]      # fresh data dirs every time
#   test-cluster.sh stop
#   test-cluster.sh urls             # comma-separated HTTP URLs
#
# Node i: HTTP PORT_BASE+i-1, transport PORT_BASE+100+i-1.
# BIN defaults to target/debug/elasticsearch (repository root).
set -euo pipefail
PORT_BASE="${PORT_BASE:-9400}"
DIR="${CLUSTER_DIR:-/tmp/es-yaml-cluster-$PORT_BASE}"
BIN="${BIN:-$(cd "$(dirname "$0")/../../../.." && pwd)/target/debug/elasticsearch}"

n_nodes() { cat "$DIR/n" 2>/dev/null || echo 2; }

case "${1:-}" in
  start)
    n="${2:-2}"
    "$0" stop >/dev/null 2>&1 || true
    rm -rf "$DIR"; mkdir -p "$DIR/repo"; echo "$n" >"$DIR/n"
    seeds=""; masters=""
    for j in $(seq 1 "$n"); do seeds+="127.0.0.1:$((PORT_BASE + 100 + j - 1)),"; masters+="node-$j,"; done
    for i in $(seq 1 "$n"); do
      mkdir -p "$DIR/node-$i"
      ES_PATH_CONF="$DIR/conf-none" nohup "$BIN" \
        -E cluster.name=yamlRestTest -E node.name="node-$i" \
        -E path.data="$DIR/node-$i" -E path.repo="$DIR/repo" \
        -E network.host=127.0.0.1 \
        -E http.port=$((PORT_BASE + i - 1)) -E transport.port=$((PORT_BASE + 100 + i - 1)) \
        -E discovery.seed_hosts="${seeds%,}" -E cluster.initial_master_nodes="${masters%,}" \
        -E node.attr.testattr=test \
        -E cluster.routing.allocation.disk.watermark.low=1b \
        -E cluster.routing.allocation.disk.watermark.high=1b \
        -E cluster.routing.allocation.disk.watermark.flood_stage=1b \
        -E xpack.license.self_generated.type="${ES_YAML_LICENSE:-trial}" \
        -E action.destructive_requires_name=false \
        -E xpack.security.enabled=false \
        >"$DIR/node-$i.log" 2>&1 &
      echo $! >"$DIR/node-$i.pid"
    done
    for _ in $(seq 1 120); do
      if curl -fsS "127.0.0.1:$PORT_BASE/_cluster/health?wait_for_nodes=$n&timeout=1s" >/dev/null 2>&1; then
        echo "cluster formed with $n nodes"; exit 0
      fi
      sleep 0.5
    done
    echo "cluster did not form; logs in $DIR" >&2; exit 1 ;;
  stop)
    for p in "$DIR"/node-*.pid; do [[ -f "$p" ]] && kill "$(cat "$p")" 2>/dev/null || true; done
    sleep 1; rm -f "$DIR"/node-*.pid ;;
  urls)
    u=""; for i in $(seq 1 "$(n_nodes)"); do u+="http://127.0.0.1:$((PORT_BASE + i - 1)),"; done; echo "${u%,}" ;;
  *) sed -n '2,11p' "$0"; exit 2 ;;
esac
