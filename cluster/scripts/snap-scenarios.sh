#!/usr/bin/env bash
# Snapshot / restore correctness scenarios on a 3-node cluster with a shared
# fs repository (ports 9500-9590, /tmp/snap-esn-*, /tmp/snap-repo-*).
#   snap-scenarios.sh [indexing|replicas|master|master_primary|finalize|restore_failover|orphan_delete|primary|delete|incremental|all]
# Every scenario starts a fresh cluster via snap-cluster.sh and prints
# "PASS <scenario>" or "FAIL <scenario>: <why>" lines; exit 1 on any FAIL.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PORT_BASE="${PORT_BASE:-9500}" TAG="${TAG:-snap}"
DIR="/tmp/snap-esn-$TAG"; REPO="/tmp/snap-repo-$TAG"
H='Content-Type: application/json'
LIVE=$PORT_BASE
fails=0; cur=""

es() { curl -s -m "${TMO:-120}" -X "$1" -H "$H" "127.0.0.1:$LIVE$2" ${3:+-d "$3"}; }
code() { curl -s -m "${TMO:-120}" -o /dev/null -w '%{http_code}' -X "$1" -H "$H" "127.0.0.1:$LIVE$2" ${3:+-d "$3"}; }
jq_() { python3 -c "import sys,json
try: d=json.load(sys.stdin)
except Exception: print(''); sys.exit()
try: print($1)
except Exception: print('')"; }
bad() { fails=$((fails + 1)); echo "FAIL $cur: $*"; }
ok() { echo "  ok   $cur: $*"; }
expect() { # desc actual wanted
  if [[ "$2" == "$3" ]]; then ok "$1 = $2"; else bad "$1: got '$2' want '$3'"; fi
}
master() { es GET '/_cat/master?h=node' | tr -d ' \n' | sed 's/node-//'; }
primary_of() { es GET "/_cat/shards/$1?h=prirep,node" | awk '$1=="p"{print $2}' | sed 's/node-//'; }
port_of() { echo $((PORT_BASE + $1 - 1)); }
wait_new_master() { local nm; for _ in $(seq 1 120); do nm=$(TMO=5 master); [[ -n "$nm" && "$nm" != "$1" ]] && return 0; sleep 0.5; done; return 1; }
use_live_except() { local i; for i in 1 2 3; do [[ "$i" != "$1" ]] && { LIVE=$(port_of "$i"); return; }; done; }
wait_green() { # index [timeout_s]
  local t=$(( ${2:-60} * 2 ))
  for _ in $(seq 1 "$t"); do
    [[ "$(es GET "/_cluster/health/$1" | jq_ 'd["status"]')" == green ]] && return 0
    sleep 0.5
  done
  return 1
}
wait_nodes() { for _ in $(seq 1 120); do [[ "$(es GET /_cat/nodes | wc -l)" -ge "$1" ]] && return 0; sleep 0.5; done; return 1; }
count() { es POST "/$1/_refresh" >/dev/null; es GET "/$1/_count" | jq_ 'd["count"]'; }
snap_state() { es GET "/_snapshot/$1/$2" | jq_ 'd["snapshots"][0]["state"] if "snapshots" in d else d["error"]["type"]'; }
wait_snap_done() { # repo snap timeout_s -> echoes final state
  local s=""
  for _ in $(seq 1 $(( ${3:-120} * 2 ))); do
    s=$(snap_state "$1" "$2")
    # anything else (no master yet, timeouts) is transient during failover
    case "$s" in SUCCESS|PARTIAL|FAILED|snapshot_missing_exception) echo "$s"; return 0 ;; esac
    sleep 0.5
  done
  echo "TIMEOUT($s)"; return 1
}
load() { # index from to [payload_bytes]  -- bulk load docs {n:i, pad:random}
  python3 - "$1" "$2" "$3" "${4:-200}" "$LIVE" <<'EOF'
import sys, json, random, string, urllib.request
idx, a, b, pad, port = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
rnd = random.Random(a)
for s in range(a, b + 1, 1000):
    lines = []
    for i in range(s, min(s + 1000, b + 1)):
        lines.append(json.dumps({"index": {"_id": str(i)}}))
        lines.append(json.dumps({"n": i, "pad": "".join(rnd.choices(string.ascii_letters + string.digits, k=pad))}))
    req = urllib.request.Request("http://127.0.0.1:%s/%s/_bulk" % (port, idx), data=("\n".join(lines) + "\n").encode(), headers={"Content-Type": "application/x-ndjson"}, method="POST")
    r = json.load(urllib.request.urlopen(req, timeout=300))
    if r.get("errors"):
        print("bulk errors", file=sys.stderr); sys.exit(1)
EOF
}
mkrepo() { es PUT "/_snapshot/$1" "{\"type\":\"fs\",\"settings\":{\"location\":\"$REPO/$1\"${2:+,$2}}}" | jq_ 'd.get("acknowledged", d)'; }
set_range() { # index -> "count min max" of field n
  es POST "/$1/_refresh" >/dev/null
  es POST "/$1/_search" '{"size":0,"track_total_hits":true,"aggs":{"mn":{"min":{"field":"n"}},"mx":{"max":{"field":"n"}}}}' | jq_ '"%d %d %d" % (d["hits"]["total"]["value"], d["aggregations"]["mn"]["value"] or 0, d["aggregations"]["mx"]["value"] or 0)'
}
# Repository integrity: listing works, every SUCCESS snapshot restores, a new
# snapshot can be taken, no in-progress entries are left behind.
repo_ok() { # repo index expected_count_for_new_snapshot
  local repo="$1" idx="$2" want="$3" names n st c
  [[ "$(code GET "/_snapshot/$repo/_all")" == 200 ]] || { bad "repo listing failed: $(es GET "/_snapshot/$repo/_all" | head -c 300)"; return; }
  [[ "$(code POST "/_snapshot/$repo/_verify")" == 200 ]] || bad "repo verify failed"
  for _ in $(seq 1 60); do
    c=$(es GET '/_cluster/state/metadata' | jq_ 'len((d.get("snapshots") or {}).get("snapshots", []))')
    n=$(es GET "/_snapshot/$repo/_current" | jq_ 'len(d["snapshots"])')
    [[ "$n" == 0 ]] && break; sleep 0.5
  done
  expect "in-progress snapshots left" "$n" 0
  names=$(es GET "/_snapshot/$repo/_all" | jq_ '" ".join(s["snapshot"]+":"+s["state"] for s in d["snapshots"])')
  ok "snapshots in repo: $names"
  expect "post-fault snapshot" "$(es PUT "/_snapshot/$repo/zz_after?wait_for_completion=true" "{\"indices\":\"$idx\"}" | jq_ 'd["snapshot"]["state"]')" SUCCESS
  for n in $names zz_after:SUCCESS; do
    st="${n#*:}"; n="${n%%:*}"
    [[ "$st" == SUCCESS ]] || continue
    [[ -n "${ONLY_NEW:-}" && "$n" != zz_after ]] && continue
    es DELETE "/chk_$idx" >/dev/null
    c=$(es POST "/_snapshot/$repo/$n/_restore?wait_for_completion=true" "{\"indices\":\"$idx\",\"rename_pattern\":\"(.+)\",\"rename_replacement\":\"chk_\$1\"}" | jq_ 'd["snapshot"]["shards"]["failed"]')
    expect "restore of $n failed shards" "$c" 0
    c=$(count "chk_$idx")
    if [[ "$n" == zz_after ]]; then expect "restored zz_after doc count" "$c" "$want"; else ok "restored $n doc count $c"; fi
    [[ -n "${SNAP_WANT[$n]:-}" ]] && expect "restored $n doc count" "$c" "${SNAP_WANT[$n]}"
  done
  es DELETE "/chk_$idx" >/dev/null
  expect "cleanup after fault" "$(code POST "/_snapshot/$repo/_cleanup")" 200
  expect "delete zz_after" "$(code DELETE "/_snapshot/$repo/zz_after")" 200
}
declare -A SNAP_WANT=()
begin() { cur="$1"; SNAP_WANT=(); LIVE=$PORT_BASE; before=$fails; "$HERE/snap-cluster.sh" start >/dev/null || { bad "cluster did not start"; return 1; }; }
finish() { if [[ "$fails" == "$before" ]]; then echo "PASS $cur"; else echo "FAILED $cur"; fi; }

# 1. Snapshot while indexing: each snapshot must hold exactly a prefix
#    {1..k} of the single sequential writer's acknowledged documents, with
#    acked-before-start <= k <= acked-after-end (+1 for the in-flight doc).
sc_indexing() {
  begin indexing || return
  mkrepo r >/dev/null
  es PUT /w '{"settings":{"number_of_shards":1,"number_of_replicas":1}}' >/dev/null
  wait_green w || bad "w not green"
  : >"$DIR/acked"
  ( i=0; while [[ ! -e "$DIR/stop-writer" ]]; do
      i=$((i + 1))
      c=$(curl -s -m 30 -o /dev/null -w '%{http_code}' -X PUT -H "$H" "127.0.0.1:$((PORT_BASE + i % 3))/w/_doc/$i" -d "{\"n\":$i}")
      [[ "$c" == 201 || "$c" == 200 ]] || { echo "writer: doc $i -> $c" >>"$DIR/writer.err"; i=$((i - 1)); sleep 0.2; continue; }
      echo "$i" >"$DIR/acked.tmp" && mv "$DIR/acked.tmp" "$DIR/acked"
    done ) &
  local writer=$! k lo hi
  local -a los his
  sleep 2
  for k in 1 2 3 4 5 6; do
    lo=$(cat "$DIR/acked" 2>/dev/null || echo 0)
    LIVE=$((PORT_BASE + k % 3))
    st=$(es PUT "/_snapshot/r/w$k?wait_for_completion=true" '{"indices":"w"}' | jq_ 'd["snapshot"]["state"]')
    hi=$(cat "$DIR/acked")
    los[$k]=${lo:-0}; his[$k]=$hi
    [[ "$st" == SUCCESS ]] || bad "snapshot w$k state '$st'"
    sleep 1
  done
  LIVE=$PORT_BASE
  touch "$DIR/stop-writer"; wait "$writer" 2>/dev/null
  [[ -s "$DIR/writer.err" ]] && bad "writer errors: $(head -3 "$DIR/writer.err" | tr '\n' ';')"
  local total; total=$(cat "$DIR/acked")
  expect "live doc count" "$(count w)" "$total"
  for k in 1 2 3 4 5 6; do
    es DELETE /rw >/dev/null
    es POST "/_snapshot/r/w$k/_restore?wait_for_completion=true" '{"indices":"w","rename_pattern":"w","rename_replacement":"rw"}' >/dev/null
    read -r c mn mx <<<"$(set_range rw)"
    if [[ "$c" -gt 0 && "$mn" == 1 && "$c" == "$mx" && "$c" -ge "${los[$k]}" && "$c" -le $(( ${his[$k]} + 1 )) ]]; then
      ok "w$k restored exactly docs 1..$c (acked before start ${los[$k]}, after end ${his[$k]})"
    else
      bad "w$k not prefix-consistent: count=$c min=$mn max=$mx acked-before=${los[$k]} acked-after=${his[$k]}"
    fi
  done
  finish
}

# 2. Restore into a cluster with replicas: every copy converges, green.
sc_replicas() {
  begin replicas || return
  mkrepo r >/dev/null
  es PUT /rep '{"settings":{"number_of_shards":1,"number_of_replicas":2}}' >/dev/null
  es PUT /rep0 '{"settings":{"number_of_shards":1,"number_of_replicas":0}}' >/dev/null
  load rep 1 5000; load rep0 1 3000
  for i in $(seq 1 200); do es DELETE "/rep/_doc/$i" >/dev/null; done   # deletes must not resurrect
  es POST /rep/_update/300 '{"doc":{"n":300,"upd":true}}' >/dev/null
  expect "snapshot" "$(es PUT '/_snapshot/r/s1?wait_for_completion=true' '{"indices":"rep,rep0"}' | jq_ 'd["snapshot"]["state"]')" SUCCESS
  es DELETE /rep,rep0 >/dev/null
  expect "restore failed shards" "$(es POST '/_snapshot/r/s1/_restore?wait_for_completion=true' '{"indices":"rep"}' | jq_ 'd["snapshot"]["shards"]["failed"]')" 0
  expect "restore rep0 with 2 replicas" "$(es POST '/_snapshot/r/s1/_restore?wait_for_completion=true' '{"indices":"rep0","index_settings":{"index.number_of_replicas":2}}' | jq_ 'd["snapshot"]["shards"]["failed"]')" 0
  wait_green rep 90 || bad "rep not green after restore: $(es GET /_cat/shards/rep | tr '\n' ';')"
  wait_green rep0 90 || bad "rep0 not green after restore: $(es GET /_cat/shards/rep0 | tr '\n' ';')"
  expect "rep count" "$(count rep)" 4800
  expect "rep0 count" "$(count rep0)" 3000
  expect "updated doc survived" "$(es GET /rep/_source/300 | jq_ 'd["upd"]')" True
  expect "deleted doc stays deleted" "$(code GET /rep/_doc/7)" 404
  expect "rep copies" "$(es GET '/_cat/shards/rep?h=state' | grep -c STARTED)" 3
  expect "rep0 copies" "$(es GET '/_cat/shards/rep0?h=state' | grep -c STARTED)" 3
  # every copy holds the full doc set: take each node down to the last one
  # holding the data is too disruptive, so read each copy via its own node.
  local p c
  for p in 0 1 2; do
    LIVE=$((PORT_BASE + p))
    c=$(es GET '/rep/_search?preference=_only_local&size=0&track_total_hits=true' | jq_ 'd["hits"]["total"]["value"]')
    expect "rep count on node-$((p + 1)) (_only_local)" "$c" 4800
    c=$(es GET '/rep0/_search?preference=_only_local&size=0&track_total_hits=true' | jq_ 'd["hits"]["total"]["value"]')
    expect "rep0 count on node-$((p + 1)) (_only_local)" "$c" 3000
  done
  LIVE=$PORT_BASE
  # converged copies survive losing the primary's node
  p=$(primary_of rep); "$HERE/snap-cluster.sh" kill "$p"; use_live_except "$p"
  for _ in $(seq 1 60); do c=$(count rep); [[ "$c" == 4800 ]] && break; sleep 0.5; done
  expect "rep count after killing primary node-$p" "$c" 4800
  for _ in $(seq 1 60); do c=$(count rep0); [[ "$c" == 3000 ]] && break; sleep 0.5; done
  expect "rep0 count after killing node-$p" "$c" 3000
  finish
}

# Create a 1-shard, 1-replica index whose primary is not on node $2.
mk_index_off() { # index node
  for _ in $(seq 1 12); do
    es PUT "/$1" "{\"settings\":{\"number_of_shards\":1,\"number_of_replicas\":1,\"index.routing.allocation.exclude._name\":\"node-$2\"}}" >/dev/null
    wait_green "$1" 30; [[ -n "$(primary_of "$1")" && "$(primary_of "$1")" != "$2" ]] && return 0
    es DELETE "/$1" >/dev/null
  done
  bad "could not place the primary of $1 off node-$2"; return 1
}
# Create a 1-shard, 1-replica index and fail the master over until the
# master is the node holding the primary.
mk_index_on_master() { # index
  local mm nm
  es PUT "/$1" '{"settings":{"number_of_shards":1,"number_of_replicas":1}}' >/dev/null
  wait_green "$1" 30
  for _ in $(seq 1 8); do
    mm=$(master); [[ -n "$mm" && "$(primary_of "$1")" == "$mm" ]] && return 0
    "$HERE/snap-cluster.sh" kill "$mm"; use_live_except "$mm"
    for _ in $(seq 1 120); do nm=$(TMO=5 master); [[ -n "$nm" && "$nm" != "$mm" ]] && break; sleep 0.5; done
    "$HERE/snap-cluster.sh" up "$mm"; wait_nodes 3; wait_green "$1" 90
  done
  bad "could not get the primary of $1 onto the master"; return 1
}
in_progress() { es GET "/_snapshot/$1/_current" | jq_ 'len(d["snapshots"])'; }
wait_idle() { # repo -> 0 once no snapshot is in progress
  local n=""
  for _ in $(seq 1 120); do n=$(TMO=10 in_progress "$1"); [[ "$n" == 0 ]] && break; sleep 0.5; done
  echo "$n"
}
restores_running() { es GET '/_cluster/state' | jq_ 'len((d.get("restore") or {}).get("snapshots", []))'; }

slow_setup() { # index docs [off:N|on-master] -> repo r (throttled) + baseline snapshot s_before in repo r
  mkrepo r >/dev/null
  case "${3:-}" in
    off:*) mk_index_off "$1" "${3#off:}" ;;
    on-master) mk_index_on_master "$1" ;;
    *) es PUT "/$1" '{"settings":{"number_of_shards":1,"number_of_replicas":1}}' >/dev/null ;;
  esac
  load "$1" 1 "$2" 400
  es POST "/$1/_flush" >/dev/null
  wait_green "$1" || bad "$1 not green"
  expect "baseline snapshot" "$(es PUT '/_snapshot/r/s_before?wait_for_completion=true' "{\"indices\":\"$1\"}" | jq_ 'd["snapshot"]["state"]')" SUCCESS
  SNAP_WANT[s_before]=$2
  load "$1" $(( $2 + 1 )) $(( $2 * 2 )) 400
  es POST "/$1/_flush" >/dev/null
  mkrepo r '"max_snapshot_bytes_per_sec":"150kb"' >/dev/null
}
unthrottle() { for _ in $(seq 1 40); do [[ "$(mkrepo r)" == True ]] && return; sleep 0.5; done; bad "could not update repository settings: $(mkrepo r | head -c 200)"; }

# 3. Master failover mid-snapshot, shard data nodes survive: the new master
#    resumes the snapshot and it ends SUCCESS with every document.
sc_master() {
  begin master || return
  local m st t0; m=$(master)
  slow_setup m 6000 "off:$m"
  expect "start slow snapshot" "$(es PUT /_snapshot/r/s_mid '{"indices":"m"}' | jq_ 'd["accepted"]')" True
  sleep 3
  expect "snapshot running before kill" "$(snap_state r s_mid)" IN_PROGRESS
  "$HERE/snap-cluster.sh" kill "$m"; use_live_except "$m"; t0=$(date +%s)
  ok "killed master node-$m (primary of m on node-$(TMO=10 primary_of m))"
  st=$(TMO=10 wait_snap_done r s_mid 180)
  expect "s_mid under the new master" "$st" SUCCESS
  ok "s_mid finished $(( $(date +%s) - t0 ))s after the kill"
  expect "in-progress entry removed" "$(wait_idle r)" 0
  ok "in-progress entry gone $(( $(date +%s) - t0 ))s after the kill"
  unthrottle
  expect "s_mid failed shards" "$(es GET /_snapshot/r/s_mid | jq_ 'd["snapshots"][0]["shards"]["failed"]')" 0
  SNAP_WANT[s_mid]=12000
  repo_ok r m 12000
  "$HERE/snap-cluster.sh" up "$m"; wait_nodes 3 || bad "node-$m did not rejoin"
  wait_green m 90 || bad "m not green after node-$m rejoined"
  LIVE=$(port_of "$m")
  expect "old master sees final state" "$(snap_state r s_mid)" "$st"
  expect "snapshot via restarted node" "$(es PUT '/_snapshot/r/s_rejoin?wait_for_completion=true' '{"indices":"m"}' | jq_ 'd["snapshot"]["state"]')" SUCCESS
  expect "s_before still restorable" "$(es POST '/_snapshot/r/s_before/_restore?wait_for_completion=true' '{"rename_pattern":"m","rename_replacement":"m_b"}' | jq_ 'd["snapshot"]["shards"]["failed"]')" 0
  expect "s_before count" "$(count m_b)" 6000
  expect "delete s_mid" "$(code DELETE /_snapshot/r/s_mid)" 200
  finish
}

# 3b. The killed master also held the only copy of the shard being
#     snapshotted: the shard fails and the resumed snapshot ends PARTIAL.
sc_master_primary() {
  begin master_primary || return
  local m st
  slow_setup m 6000 on-master
  m=$(master)
  expect "primary of m is on the master" "$(primary_of m)" "$m"
  expect "start slow snapshot" "$(es PUT /_snapshot/r/s_mid '{"indices":"m"}' | jq_ 'd["accepted"]')" True
  sleep 3
  expect "snapshot running before kill" "$(snap_state r s_mid)" IN_PROGRESS
  "$HERE/snap-cluster.sh" kill "$m"; use_live_except "$m"
  st=$(TMO=10 wait_snap_done r s_mid 120)
  expect "s_mid under the new master" "$st" PARTIAL
  expect "in-progress entry removed" "$(wait_idle r)" 0
  expect "s_mid failed shards" "$(es GET /_snapshot/r/s_mid | jq_ 'd["snapshots"][0]["shards"]["failed"]')" 1
  ok "shard failure: $(es GET /_snapshot/r/s_mid | jq_ 'd["snapshots"][0]["failures"][0]["reason"]' | head -c 160)"
  unthrottle
  "$HERE/snap-cluster.sh" up "$m"; wait_nodes 3 || bad "node-$m did not rejoin"
  wait_green m 90 || bad "m not green after node-$m rejoined"
  expect "m count after rejoin" "$(count m)" 12000
  repo_ok r m 12000
  expect "delete s_mid" "$(code DELETE /_snapshot/r/s_mid)" 200
  finish
}

# 3c. Master killed around the finalize write. Odd attempts kill when the
#     snapshot's snap-<uuid>.dat appears (before the root index-N write), even
#     attempts when the new index-N appears (after it, before the in-progress
#     entry is removed). Either way: exactly one SUCCESS snapshot, restorable.
sc_finalize() {
  begin finalize || return
  mkrepo r >/dev/null
  local att m st gen nsnap landed idx
  shopt -s nullglob
  for att in 1 2 3 4 5 6; do
    wait_new_master 0; m=$(master); idx="f$att"
    mk_index_off "$idx" "$m" || break
    load "$idx" 1 3000 200; es POST "/$idx/_flush" >/dev/null
    use_live_except "$m"
    gen=$(ls "$REPO/r" | sed -n 's/^index-\([0-9][0-9]*\)$/\1/p' | sort -n | tail -1); gen=${gen:--1}
    set -- "$REPO"/r/snap-*.dat; nsnap=$#
    expect "attempt $att: start snapshot" "$(es PUT "/_snapshot/r/s_f$att" "{\"indices\":\"$idx\"}" | jq_ 'd["accepted"]')" True
    SECONDS=0
    while (( SECONDS < 30 )); do
      if (( att % 2 )); then set -- "$REPO"/r/snap-*.dat; (( $# > nsnap )) && break
      else [[ -e "$REPO/r/index-$((gen + 1))" ]] && break; fi
    done
    "$HERE/snap-cluster.sh" kill "$m"
    if [[ -e "$REPO/r/index-$((gen + 1))" ]]; then landed="after the root index-$((gen + 1)) write"; else set -- "$REPO"/r/snap-*.dat; landed="before the root index write (snap blob written: $(( $# > nsnap )))"; fi
    ok "attempt $att: killed master node-$m $landed"
    wait_new_master "$m" || bad "attempt $att: no new master"
    st=$(TMO=10 wait_snap_done r "s_f$att" 120)
    expect "attempt $att: s_f$att under the new master" "$st" SUCCESS
    expect "attempt $att: in-progress snapshots left" "$(wait_idle r)" 0
    expect "attempt $att: snapshots named s_f$att" "$(es GET /_snapshot/r/_all | jq_ "sum(1 for s in d['snapshots'] if s['snapshot']=='s_f$att')")" 1
    expect "attempt $att: restore failed shards" "$(es POST "/_snapshot/r/s_f$att/_restore?wait_for_completion=true" "{\"rename_pattern\":\"(.+)\",\"rename_replacement\":\"chk_\$1\"}" | jq_ 'd["snapshot"]["shards"]["failed"]')" 0
    expect "attempt $att: restored doc count" "$(count "chk_$idx")" 3000
    es DELETE "/chk_$idx" >/dev/null
    "$HERE/snap-cluster.sh" up "$m"; wait_nodes 3 || bad "attempt $att: node-$m did not rejoin"
    wait_green "$idx" 90 || bad "attempt $att: $idx not green after node-$m rejoined"
  done
  # A master killed right after writing index-N leaves index-(N-1) behind
  # (removed by the next write); what must hold is one snap blob per
  # snapshot and index.latest naming the highest generation.
  set -- "$REPO"/r/snap-*.dat
  expect "snap blobs in the repo root" "$#" 6
  shopt -u nullglob
  ONLY_NEW=1 repo_ok r f1 3000
  # (a kill between the index-N write and the index.latest update leaves
  # index.latest one behind until the next write; readers list index-N)
  gen=$(ls "$REPO/r" | sed -n 's/^index-\([0-9][0-9]*\)$/\1/p' | sort -n | tail -1)
  expect "after the next writes index.latest names the only root generation ($gen)" "$((16#$(od -An -tx1 "$REPO/r/index.latest" | tr -d ' \n')))" "$gen"
  expect "root index generations left" "$(ls "$REPO/r" | grep -c -E '^index-[0-9]+$')" 1
  finish
}

# 3d. Master killed while a restore is running: the new master runs the
#     restore again; the index ends green with every document. If it does
#     not, the restore must be re-issuable.
sc_restore_failover() {
  begin restore_failover || return
  mkrepo r >/dev/null
  es PUT /rs '{"settings":{"number_of_shards":1,"number_of_replicas":1}}' >/dev/null
  load rs 1 30000 100; es POST /rs/_flush >/dev/null
  wait_green rs || bad "rs not green"
  expect "snapshot" "$(es PUT '/_snapshot/r/s1?wait_for_completion=true' '{"indices":"rs"}' | jq_ 'd["snapshot"]["state"]')" SUCCESS
  local m c n t0; m=$(master); use_live_except "$m"
  expect "start restore" "$(es POST '/_snapshot/r/s1/_restore' '{"indices":"rs","rename_pattern":"rs","rename_replacement":"rs_r"}' | jq_ 'd["accepted"]')" True
  c=0
  for _ in $(seq 1 200); do c=$(count rs_r); [[ "${c:-0}" -gt 0 ]] && break; sleep 0.05; done
  expect "restore entry present before kill" "$(restores_running)" 1
  "$HERE/snap-cluster.sh" kill "$m"; t0=$(date +%s)
  ok "killed master node-$m with ${c:-0} of 30000 docs restored"
  for _ in $(seq 1 360); do n=$(TMO=10 restores_running); [[ "$n" == 0 ]] && break; sleep 0.5; done
  expect "restore entries left" "$n" 0
  ok "restore entry gone $(( $(date +%s) - t0 ))s after the kill"
  wait_green rs_r 60 || true
  c=$(count rs_r)
  if [[ "$c" == 30000 ]]; then ok "restore completed under the new master ($c docs)"
  else
    bad "restore not completed by the new master ($c docs); re-issuing"
    es DELETE /rs_r >/dev/null
    expect "re-issued restore failed shards" "$(es POST '/_snapshot/r/s1/_restore?wait_for_completion=true' '{"indices":"rs","rename_pattern":"rs","rename_replacement":"rs_r"}' | jq_ 'd["snapshot"]["shards"]["failed"]')" 0
    expect "re-issued restore count" "$(count rs_r)" 30000
  fi
  expect "min/max of restored docs" "$(set_range rs_r)" "30000 1 30000"
  "$HERE/snap-cluster.sh" up "$m"; wait_nodes 3 || bad "node-$m did not rejoin"
  wait_green rs_r 120 || bad "rs_r not green after node-$m rejoined"
  expect "rs_r health" "$(es GET /_cluster/health/rs_r | jq_ 'd["status"]')" green
  expect "rs_r count after rejoin" "$(count rs_r)" 30000
  expect "a second restore into a new name" "$(es POST '/_snapshot/r/s1/_restore?wait_for_completion=true' '{"indices":"rs","rename_pattern":"rs","rename_replacement":"rs_2"}' | jq_ 'd["snapshot"]["shards"]["failed"]')" 0
  expect "rs_2 count" "$(count rs_2)" 30000
  finish
}

# 3e. DELETE of a snapshot whose master died: aborts and removes it.
sc_orphan_delete() {
  begin orphan_delete || return
  local m c t0; m=$(master)
  slow_setup m 6000 "off:$m"
  expect "start slow snapshot" "$(es PUT /_snapshot/r/s_mid '{"indices":"m"}' | jq_ 'd["accepted"]')" True
  sleep 3
  expect "snapshot running before kill" "$(snap_state r s_mid)" IN_PROGRESS
  "$HERE/snap-cluster.sh" kill "$m"; use_live_except "$m"; t0=$(date +%s)
  # the repo stays throttled: without the abort the snapshot would run for minutes
  c=""
  for _ in $(seq 1 60); do c=$(TMO=60 code DELETE /_snapshot/r/s_mid); [[ "$c" == 200 || "$c" == 404 ]] && break; sleep 0.5; done
  expect "DELETE of the orphaned running snapshot" "$c" 200
  ok "DELETE answered $(( $(date +%s) - t0 ))s after the kill"
  expect "s_mid after delete" "$(snap_state r s_mid)" snapshot_missing_exception
  expect "in-progress snapshots left" "$(wait_idle r)" 0
  unthrottle
  repo_ok r m 12000
  "$HERE/snap-cluster.sh" up "$m"; wait_nodes 3 || bad "node-$m did not rejoin"
  LIVE=$(port_of "$m")
  expect "old master sees it gone" "$(snap_state r s_mid)" snapshot_missing_exception
  finish
}

# 4. Kill the node holding the primary mid-snapshot (non-master if possible).
sc_primary() {
  begin primary || return
  slow_setup p 6000
  local m pn st
  m=$(master); pn=$(primary_of p)
  if [[ "$pn" == "$m" ]]; then
    # move mastership away is not possible from here; move the primary instead
    es PUT /p/_settings '{"index.number_of_replicas":0}' >/dev/null; sleep 1
    es PUT /p/_settings '{"index.number_of_replicas":1}' >/dev/null; wait_green p
    pn=$(primary_of p)
  fi
  ok "master node-$m, primary on node-$pn"
  expect "start slow snapshot" "$(es PUT /_snapshot/r/s_mid '{"indices":"p"}' | jq_ 'd["accepted"]')" True
  "$HERE/snap-cluster.sh" kill "$pn"; use_live_except "$pn"
  ok "killed primary holder node-$pn"
  st=$(TMO=10 wait_snap_done r s_mid 180)
  case "$st" in SUCCESS|PARTIAL|FAILED|snapshot_missing_exception) ok "s_mid ended as $st" ;; *) bad "s_mid state after primary loss: $st" ;; esac
  [[ "$st" == SUCCESS ]] && SNAP_WANT[s_mid]=12000
  unthrottle
  for _ in $(seq 1 120); do [[ "$(es GET /_cluster/health/p | jq_ 'd["active_primary_shards"]')" == 1 ]] && break; sleep 0.5; done
  expect "live count after primary loss" "$(count p)" 12000
  repo_ok r p 12000
  "$HERE/snap-cluster.sh" up "$pn"; wait_nodes 3 || bad "node-$pn did not rejoin"
  wait_green p 90 || bad "p not green after node-$pn rejoined"
  expect "snapshot after rejoin" "$(es PUT '/_snapshot/r/s_rejoin?wait_for_completion=true' '{"indices":"p"}' | jq_ 'd["snapshot"]["state"]')" SUCCESS
  finish
}

# 5. Delete a snapshot while another is running; delete the running one.
sc_delete() {
  begin delete || return
  slow_setup d 5000
  local c st
  expect "start slow snapshot" "$(es PUT /_snapshot/r/s_run '{"indices":"d"}' | jq_ 'd["accepted"]')" True
  sleep 2
  expect "s_run running" "$(snap_state r s_run)" IN_PROGRESS
  expect "status of running snapshot" "$(es GET /_snapshot/r/s_run/_status | jq_ 'd["snapshots"][0]["state"]')" STARTED
  expect "_current lists it" "$(es GET /_snapshot/r/_current | jq_ 'd["snapshots"][0]["snapshot"]')" s_run
  expect "same name while running" "$(es PUT /_snapshot/r/s_run | jq_ 'd["error"]["type"]')" invalid_snapshot_name_exception
  expect "repo delete while running" "$(code DELETE /_snapshot/r)" 500
  LIVE=$((PORT_BASE + 1))
  c=$(TMO=300 code DELETE /_snapshot/r/s_before)
  LIVE=$PORT_BASE
  case "$c" in
    200) ok "delete of s_before during s_run -> 200" ;;
    *) bad "delete of s_before during s_run -> $c $(es DELETE /_snapshot/r/s_before | head -c 300)" ;;
  esac
  st=$(wait_snap_done r s_run 300)
  expect "s_run final state" "$st" SUCCESS
  SNAP_WANT[s_run]=10000
  expect "s_before gone" "$(code GET /_snapshot/r/s_before)" 404
  # delete the running snapshot itself -> aborted and removed
  es POST /d/_doc '{"n":0}' >/dev/null; load d 10001 14000 400; es POST /d/_flush >/dev/null
  expect "start second slow snapshot" "$(es PUT /_snapshot/r/s_abort '{"indices":"d"}' | jq_ 'd["accepted"]')" True
  sleep 2
  expect "s_abort running" "$(snap_state r s_abort)" IN_PROGRESS
  expect "delete running snapshot" "$(TMO=120 code DELETE /_snapshot/r/s_abort)" 200
  expect "s_abort gone" "$(code GET /_snapshot/r/s_abort)" 404
  unthrottle
  repo_ok r d 14001
  finish
}

# 6. Incremental snapshots share files; deleting one does not break another.
sc_incremental() {
  begin incremental || return
  mkrepo r >/dev/null
  es PUT /inc '{"settings":{"number_of_shards":1,"number_of_replicas":1}}' >/dev/null
  load inc 1 8000 400; es POST /inc/_flush >/dev/null
  expect "s1" "$(es PUT '/_snapshot/r/s1?wait_for_completion=true' '{"indices":"inc"}' | jq_ 'd["snapshot"]["state"]')" SUCCESS
  local sz1 sz2 sz3 inc tot f2
  sz1=$(du -sk "$REPO/r" | cut -f1)
  expect "s1 again unchanged (s1b)" "$(es PUT '/_snapshot/r/s1b?wait_for_completion=true' '{"indices":"inc"}' | jq_ 'd["snapshot"]["state"]')" SUCCESS
  inc=$(es GET /_snapshot/r/s1b/_status | jq_ 'd["snapshots"][0]["stats"]["incremental"]["file_count"]')
  expect "unchanged shard uploads no files" "$inc" 0
  load inc 8001 8200 400; es POST /inc/_flush >/dev/null
  expect "s2" "$(es PUT '/_snapshot/r/s2?wait_for_completion=true' '{"indices":"inc"}' | jq_ 'd["snapshot"]["state"]')" SUCCESS
  sz2=$(du -sk "$REPO/r" | cut -f1)
  read -r inc tot <<<"$(es GET /_snapshot/r/s2/_status | jq_ '"%d %d" % (d["snapshots"][0]["stats"]["incremental"]["file_count"], d["snapshots"][0]["stats"]["total"]["file_count"])')"
  if [[ "$inc" -lt "$tot" && $((sz2 - sz1)) -lt $((sz1 / 2)) ]]; then
    ok "s2 incremental: $inc of $tot files uploaded, repo ${sz1}k -> ${sz2}k"
  else
    bad "s2 not incremental: $inc of $tot files uploaded, repo ${sz1}k -> ${sz2}k"
  fi
  expect "delete s1" "$(code DELETE /_snapshot/r/s1)" 200
  expect "delete s1b" "$(code DELETE /_snapshot/r/s1b)" 200
  expect "restore s2 after deleting s1" "$(es POST '/_snapshot/r/s2/_restore?wait_for_completion=true' '{"rename_pattern":"inc","rename_replacement":"inc_r2"}' | jq_ 'd["snapshot"]["shards"]["failed"]')" 0
  expect "s2 count" "$(count inc_r2)" 8200
  load inc 8201 8400 400; es POST /inc/_flush >/dev/null
  expect "s3" "$(es PUT '/_snapshot/r/s3?wait_for_completion=true' '{"indices":"inc"}' | jq_ 'd["snapshot"]["state"]')" SUCCESS
  expect "clone s3 -> c3" "$(code PUT /_snapshot/r/s3/_clone/c3 '{"indices":"inc"}')" 200
  expect "delete s3 (newest)" "$(code DELETE /_snapshot/r/s3)" 200
  expect "cleanup" "$(code POST /_snapshot/r/_cleanup)" 200
  es DELETE /inc_r2 >/dev/null
  expect "restore s2 after deleting s3 + cleanup" "$(es POST '/_snapshot/r/s2/_restore?wait_for_completion=true' '{"rename_pattern":"inc","rename_replacement":"inc_r2"}' | jq_ 'd["snapshot"]["shards"]["failed"]')" 0
  expect "s2 count" "$(count inc_r2)" 8200
  expect "restore clone c3 after deleting its source" "$(es POST '/_snapshot/r/c3/_restore?wait_for_completion=true' '{"rename_pattern":"inc","rename_replacement":"inc_r3"}' | jq_ 'd["snapshot"]["shards"]["failed"]')" 0
  expect "c3 count" "$(count inc_r3)" 8400
  expect "delete s2,c3" "$(code DELETE /_snapshot/r/s2,c3)" 200
  sz3=$(find "$REPO/r/indices" -type f 2>/dev/null | wc -l | tr -d ' ')
  expect "no shard blobs left after deleting every snapshot" "$sz3" 0
  expect "empty repo still usable" "$(es PUT '/_snapshot/r/s4?wait_for_completion=true' '{"indices":"inc"}' | jq_ 'd["snapshot"]["state"]')" SUCCESS
  finish
}

what="${1:-all}"
[[ "$what" == all ]] && what="indexing replicas master master_primary finalize restore_failover orphan_delete primary delete incremental"
for s in $what; do "sc_$s"; done
"$HERE/snap-cluster.sh" stop
echo "snap-scenarios: $fails failed checks"
[[ "$fails" -eq 0 ]]
