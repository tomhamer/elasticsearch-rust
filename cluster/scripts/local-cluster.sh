#!/usr/bin/env bash
# Start / stop an N-node cluster of the `elasticsearch` binary on localhost.
#
#   local-cluster.sh start [N=3] [BIN=target/debug/elasticsearch]
#   local-cluster.sh stop
#   local-cluster.sh kill I          # kill -9 node I (1-based)
#   local-cluster.sh restart I       # start node I again with its data
#
# EXTRA_ARGS (word-split) is appended to every node's command line, e.g.
# EXTRA_ARGS="-E path.repo=/tmp/repos".
#
# Node i: HTTP PORT_BASE+i-1, transport PORT_BASE+100+i-1, data in
# $CLUSTER_DIR/node-i. All nodes are master-eligible data nodes.
set -euo pipefail
# PORT_BASE lets several clusters share a host: HTTP PORT_BASE+i-1,
# transport PORT_BASE+100+i-1.
PORT_BASE="${PORT_BASE:-9200}"
CLUSTER_DIR="${CLUSTER_DIR:-/tmp/es-local-cluster-$PORT_BASE}"
BIN="${BIN:-$(cd "$(dirname "$0")/../.." && pwd)/target/debug/elasticsearch}"

start_node() {
  local i=$1 n=$2
  local seeds="" masters=""
  for j in $(seq 1 "$n"); do
    seeds+="127.0.0.1:$((PORT_BASE + 100 + j - 1)),"
    masters+="node-$j,"
  done
  mkdir -p "$CLUSTER_DIR/node-$i"
  # `cluster.initial_master_nodes` only matters on the very first start.
  local bootstrap=()
  if [[ ! -f "$CLUSTER_DIR/node-$i/_cluster/coordination.json" ]] && ! compgen -G "$CLUSTER_DIR/node-$i/_cluster/state-*.json" >/dev/null; then
    bootstrap=(-E "cluster.initial_master_nodes=${masters%,}")
  fi
  ES_PATH_CONF="$CLUSTER_DIR/conf-none" nohup "$BIN" \
    -E cluster.name=local -E node.name="node-$i" \
    -E path.data="$CLUSTER_DIR/node-$i" \
    -E network.host=127.0.0.1 \
    -E http.port=$((PORT_BASE + i - 1)) -E transport.port=$((PORT_BASE + 100 + i - 1)) \
    -E discovery.seed_hosts="${seeds%,}" "${bootstrap[@]}" \
    -E node.attr.zone="zone-$(( (i - 1) % 3 ))" \
    -E xpack.license.self_generated.type="${ES_LICENSE:-trial}" \
    -E xpack.security.enabled=false ${EXTRA_ARGS:-} \
    >"$CLUSTER_DIR/node-$i.log" 2>&1 &
  echo $! >"$CLUSTER_DIR/node-$i.pid"
}

case "${1:-}" in
  start)
    n="${2:-3}"
    echo "$n" >"$CLUSTER_DIR.n" 2>/dev/null || { mkdir -p "$CLUSTER_DIR"; echo "$n" >"$CLUSTER_DIR.n"; }
    mkdir -p "$CLUSTER_DIR"
    for i in $(seq 1 "$n"); do start_node "$i" "$n"; done
    for _ in $(seq 1 120); do
      if curl -fsS "localhost:$PORT_BASE/_cluster/health?wait_for_nodes=$n&timeout=1s" >/dev/null 2>&1; then
        echo "cluster formed with $n nodes"; exit 0
      fi
      sleep 0.5
    done
    echo "cluster did not form; logs in $CLUSTER_DIR" >&2; exit 1 ;;
  stop)
    for p in "$CLUSTER_DIR"/node-*.pid; do [[ -f "$p" ]] && kill "$(cat "$p")" 2>/dev/null || true; done
    sleep 1; rm -f "$CLUSTER_DIR"/node-*.pid ;;
  kill)
    kill -9 "$(cat "$CLUSTER_DIR/node-$2.pid")" ;;
  restart)
    start_node "$2" "$(cat "$CLUSTER_DIR.n")" ;;
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac
