#!/usr/bin/env bash
# Measure master failover in the docker compose deployment
# (cluster/deploy/docker/docker-compose.yml) when the elected master is
# killed with `docker compose kill` (SIGKILL; the container's IP vanishes,
# so reconnects hang instead of being refused).
#
#   ES_IMAGE=elasticsearch-rs:dev compose-failover-timing.sh [trials=3] [window_secs=90]
#
# Each trial starts the compose cluster from scratch (`down -v`, `up -d`),
# creates index `fo` (3 shards, 2 replicas: every primary has a copy on every
# node), then kills the elected master and polls the two survivors every
# 200 ms over their container IPs. Reported relative to the kill:
#   master  - a survivor reports a surviving node in `_cat/master`
#   removed - a survivor's `_cat/nodes` no longer lists the dead node
#   writes  - start of the first write after which every write is 2xx in <1s
#   stall   - longest single write (seconds) and number of slow/failed writes
#   lost    - acknowledged (2xx) writes that _mget cannot find after the window
# Only the compose project's own containers are touched.
set -uo pipefail
TRIALS="${1:-3}"; WINDOW="${2:-90}"
COMPOSE_DIR="$(cd "$(dirname "$0")/../deploy/docker" && pwd)"
export ES_IMAGE="${ES_IMAGE:-elasticsearch-rs:dev}" ES_PORT="${ES_PORT:-30921}"
OUT="${OUT_DIR:-$HOME/compose-fo}"
dc() { docker compose -f "$COMPOSE_DIR/docker-compose.yml" "$@"; }
now() { date +%s.%N; }
ip() { docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$1"; }

trial() {
  local t="$1" out="$OUT/trial-$1"; rm -rf "$out"; mkdir -p "$out"
  dc down -v >/dev/null 2>&1
  dc up -d --no-build >/dev/null 2>&1 || { echo "compose up failed"; return 1; }
  local e1; e1="$(ip es01)"
  for _ in $(seq 1 120); do
    curl -fsS -m 2 "$e1:9200/_cluster/health?wait_for_nodes=3&timeout=1s" >/dev/null 2>&1 && break; sleep 0.5
  done
  local ok=0
  for _ in $(seq 1 30); do   # 503 until the first master is elected
    curl -fsS -m 30 -XPUT "$e1:9200/fo" -H 'content-type: application/json' \
      -d '{"settings":{"number_of_shards":3,"number_of_replicas":2}}' >/dev/null 2>&1 && { ok=1; break; }
    sleep 1
  done
  (( ok )) || { echo "create index failed"; return 1; }
  curl -fsS -m 30 "$e1:9200/_cluster/health/fo?wait_for_status=green&timeout=30s" >/dev/null || { echo "not green"; return 1; }
  sleep 3   # let fault detection settle
  local master; master="$(curl -fsS -m 5 "$e1:9200/_cat/master?h=node" | tr -d ' \n')"
  [[ "$master" =~ ^es0[123]$ ]] || { echo "no master: '$master'"; return 1; }
  local surv=() ips=()
  for n in es01 es02 es03; do [[ "$n" != "$master" ]] && { surv+=("$n"); ips+=("$(ip "$n")"); }; done
  local t0 t1; t0="$(now)"; dc kill "$master" >/dev/null 2>&1; t1="$(now)"
  echo "$t0 $t1 $master" >"$out/t0"
  local end=$(( $(date +%s) + WINDOW )) n=0
  (
  while (( $(date +%s) < end )); do
    n=$((n + 1))
    for k in 0 1; do
      local h="${ips[$k]}"
      ( r="$(curl -s -m 2 "$h:9200/_cat/master?h=node" | tr -d ' \n')"; echo "$(now) $r" >>"$out/master-$k" ) &
      ( r="$(curl -s -m 2 "$h:9200/_cat/nodes?h=name" | tr -d ' ' | tr '\n' ',')"; echo "$(now) $r" >>"$out/nodes-$k" ) &
    done
    local h="${ips[$((n % 2))]}"
    ( b="$(now)"; c="$(curl -s -m 60 -o /dev/null -w '%{http_code}' -XPUT "$h:9200/fo/_doc/$n" -H 'content-type: application/json' -d '{"n":1}')"; echo "$b $(now) $c $n $h" >>"$out/writes" ) &
    sleep 0.2
  done
  wait
  )
  local tm tr tw st
  tm="$(cat "$out"/master-* | sort -n | awk -v t0="$t0" -v dead="$master" '$2 ~ /^es0[123]$/ && $2 != dead { printf "%.1f", $1 - t0; exit }')"
  tr="$(cat "$out"/nodes-* | sort -n | awk -v t0="$t0" -v dead="$master" '$2 ~ /es0/ && index($2, dead ",") == 0 { printf "%.1f", $1 - t0; exit }')"
  tw="$(sort -n "$out/writes" | awk -v t0="$t0" '{ ok = ($3 ~ /^20[01]$/ && $2 - $1 < 1.0); if (!ok) { first = ""; bad++ } else if (first == "") first = $1 - t0 } END { if (first == "") print "never"; else printf "%.1f (%d slow/failed)", (bad ? first : 0), bad }')"
  st="$(awk '{ d = $2 - $1; if (d > m) m = d; if ($3 !~ /^20[01]$/) f++ } END { printf "max_write=%.1fs non2xx=%d total=%d", m, f, NR }' "$out/writes")"
  echo "trial $t dead=$master kill_cmd=$(awk '{printf "%.1f", $2-$1}' "$out/t0")s master_elected=${tm:-never}s node_removed=${tr:-never}s writes_ok=${tw}s $st"
  # Acknowledged writes (2xx) must all be readable from a survivor.
  local acked ids found
  acked="$(awk '$3 ~ /^20[01]$/' "$out/writes" | wc -l)"
  ids="$(awk '$3 ~ /^20[01]$/ { printf "%s\"%s\"", (k++ ? "," : ""), $4 }' "$out/writes")"
  curl -s -m 30 -XPOST "${ips[0]}:9200/fo/_refresh" >/dev/null
  found="$(curl -s -m 60 -XPOST "${ips[0]}:9200/fo/_mget" -H 'content-type: application/json' -d "{\"ids\":[$ids]}" | grep -o '"found":true' | wc -l)"
  echo "  acknowledged=$acked present=$found lost=$((acked - found)) timed_out_by_host: $(awk '$3 !~ /^20[01]$/ { c[$5]++ } END { for (h in c) printf "%s=%d ", h, c[h] }' "$out/writes")"
  dc logs --no-color >"$out/compose.log" 2>&1
  docker inspect -f '{{.State.Status}} restarts={{.RestartCount}}' "$master" >"$out/dead-state" 2>&1
  echo "  dead container after window: $(cat "$out/dead-state")"
}

for t in $(seq 1 "$TRIALS"); do trial "$t"; done
dc down -v >/dev/null 2>&1
