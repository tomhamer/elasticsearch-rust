#!/usr/bin/env bash
# Measure master-failover timing on a local 3-node cluster.
#
#   failover-timing.sh <kill|stop|blackhole> [trials=3] [window_secs=90]
#
# kill = `kill -9` of the elected master (disconnect: connections reset,
#        reconnects refused),
# stop = `kill -STOP` (hung process: connections stay open, nothing answers),
# blackhole = `kill -9` plus an iptables rule dropping SYNs to the dead
#        node's transport port: connections reset, but reconnects hang
#        instead of being refused (a killed container / vanished host).
#        Needs passwordless sudo; the rule is tagged esfo-$PORT_BASE and
#        removed after the trial.
# Each trial starts a fresh cluster, then polls the two survivors every
# 200 ms and reports, relative to the signal:
#   master  - a survivor reports a surviving node in `_cat/master`
#   removed - a survivor's `_cat/nodes` no longer lists the dead node
#   writes  - start of the first write after which every write succeeds in <1s
#
# HTTP ports PORT_BASE..+2, transport PORT_BASE+10..+12, data in DIR.
# Only PIDs recorded in DIR/node-*.pid are ever signalled.
set -uo pipefail
MODE="${1:?kill|stop}"; TRIALS="${2:-3}"; WINDOW="${3:-90}"
PORT_BASE="${PORT_BASE:-9400}"
DIR="${CLUSTER_DIR:-/tmp/esfo-$PORT_BASE}"
BIN="${BIN:-$(cd "$(dirname "$0")/../.." && pwd)/target/debug/elasticsearch}"
EXTRA=(${EXTRA_SETTINGS:-})   # e.g. "-E cluster.election.max_timeout=2s"

now() { date +%s.%N; }
http() { echo "127.0.0.1:$((PORT_BASE + $1 - 1))"; }

unblackhole() {
  [[ -f "$DIR-blackhole-port" ]] || return 0
  local port; port="$(cat "$DIR-blackhole-port")"
  while sudo -n iptables -w -D INPUT -i lo -p tcp --dport "$port" --syn -m comment --comment "esfo-$PORT_BASE" -j DROP 2>/dev/null; do :; done
  rm -f "$DIR-blackhole-port"
}
trap 'unblackhole; stop_all' EXIT

stop_all() {
  for p in "$DIR"/node-*.pid; do
    [[ -f "$p" ]] || continue
    kill -CONT "$(cat "$p")" 2>/dev/null; kill -9 "$(cat "$p")" 2>/dev/null
    rm -f "$p"
  done
}

start_all() {
  stop_all; sleep 0.5; rm -rf "$DIR"; mkdir -p "$DIR"
  local seeds="" masters=""
  for j in 1 2 3; do seeds+="127.0.0.1:$((PORT_BASE + 10 + j - 1)),"; masters+="node-$j,"; done
  for i in 1 2 3; do
    mkdir -p "$DIR/node-$i"
    ES_PATH_CONF="$DIR/conf-none" RUST_LOG="${NODE_LOG:-info}" nohup "$BIN" \
      -E cluster.name="esfo-$PORT_BASE" -E node.name="node-$i" -E path.data="$DIR/node-$i" \
      -E network.host=127.0.0.1 -E http.port=$((PORT_BASE + i - 1)) -E transport.port=$((PORT_BASE + 10 + i - 1)) \
      -E discovery.seed_hosts="${seeds%,}" -E cluster.initial_master_nodes="${masters%,}" \
      -E xpack.security.enabled=false "${EXTRA[@]}" >"$DIR/node-$i.log" 2>&1 &
    echo $! >"$DIR/node-$i.pid"
  done
  for _ in $(seq 1 120); do
    curl -fsS -m 2 "$(http 1)/_cluster/health?wait_for_nodes=3&timeout=1s" >/dev/null 2>&1 && return 0
    sleep 0.5
  done
  echo "cluster did not form; logs in $DIR" >&2; return 1
}

trial() {
  local t="$1"
  start_all || return 1
  curl -fsS -m 30 -XPUT "$(http 1)/fo" -H 'content-type: application/json' \
    -d '{"settings":{"number_of_shards":3,"number_of_replicas":1}}' >/dev/null
  curl -fsS -m 30 "$(http 1)/_cluster/health/fo?wait_for_status=green&timeout=20s" >/dev/null
  sleep 3   # let fault-detection checks settle into steady state
  local master; master="$(curl -fsS -m 5 "$(http 1)/_cat/master?h=node" | tr -d ' \n')"
  local mi="${master#node-}" pid; pid="$(cat "$DIR/node-$mi.pid")"
  local survivors=(); for i in 1 2 3; do [[ "$i" != "$mi" ]] && survivors+=("$i"); done
  local out="$DIR/trial-$MODE-$t"; rm -rf "$out"; mkdir -p "$out"
  local t0; t0="$(now)"
  case "$MODE" in
    kill) kill -9 "$pid" ;;
    stop) kill -STOP "$pid" ;;
    blackhole)
      local port=$((PORT_BASE + 10 + mi - 1))
      echo "$port" >"$DIR-blackhole-port"
      sudo -n iptables -w -I INPUT -i lo -p tcp --dport "$port" --syn -m comment --comment "esfo-$PORT_BASE" -j DROP || return 1
      t0="$(now)"; kill -9 "$pid" ;;
    *) echo "unknown mode $MODE" >&2; return 1 ;;
  esac
  echo "$t0" >"$out/t0"
  local end=$(( $(date +%s) + WINDOW )) n=0
  (  # subshell: `wait` must cover the probes only, not the node processes
  while (( $(date +%s) < end )); do
    n=$((n + 1))
    for s in "${survivors[@]}"; do
      ( r="$(curl -s -m 2 "$(http "$s")/_cat/master?h=node" | tr -d ' \n')"; echo "$(now) $r" >>"$out/master-$s" ) &
      ( r="$(curl -s -m 2 "$(http "$s")/_cat/nodes?h=name" | tr '\n' ',')"; echo "$(now) $r" >>"$out/nodes-$s" ) &
    done
    local s="${survivors[$((n % 2))]}"
    ( b="$(now)"; c="$(curl -s -m 60 -o /dev/null -w '%{http_code}' -XPUT "$(http "$s")/fo/_doc/$n" -H 'content-type: application/json' -d '{"n":1}')"; echo "$b $(now) $c" >>"$out/writes" ) &
    sleep 0.2
  done
  wait
  )
  local tm tr tw
  tm="$(cat "$out"/master-* | sort -n | awk -v t0="$t0" -v dead="$master" '$2 ~ /^node-[0-9]+$/ && $2 != dead { printf "%.1f", $1 - t0; exit }')"
  tr="$(cat "$out"/nodes-* | sort -n | awk -v t0="$t0" -v dead="$master" '$2 ~ /node-/ && index($2, dead ",") == 0 { printf "%.1f", $1 - t0; exit }')"
  tw="$(sort -n "$out/writes" | awk -v t0="$t0" '{ ok = ($3 ~ /^20[01]$/ && $2 - $1 < 1.0); if (!ok) { first = ""; bad++ } else if (first == "") first = $1 - t0 } END { if (first == "") print "never"; else printf "%.1f (%d slow/failed)", (bad ? first : 0), bad }')"
  echo "trial $t mode=$MODE dead=$master  master_elected=${tm:-never}s  node_removed=${tr:-never}s  writes_ok=${tw}s"
  mkdir -p "$out/logs"; cp "$DIR"/node-*.log "$out/logs/" 2>/dev/null
  # Keep a copy outside DIR: the next trial wipes it.
  rm -rf "$DIR-last-$MODE-$t"; cp -r "$out" "$DIR-last-$MODE-$t"
  unblackhole
}

for t in $(seq 1 "$TRIALS"); do trial "$t"; done
stop_all
