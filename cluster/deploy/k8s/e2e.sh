#!/usr/bin/env bash
# End-to-end test of the elasticsearch Helm chart on a local kind cluster.
#
# Requires: docker (running), kind, kubectl, helm, jq. Run from anywhere:
#
#   cluster/deploy/k8s/e2e.sh
#
# What it does:
#   1. creates a kind cluster with 3 workers labelled zone-a/b/c
#   2. builds (or reuses) the node image and loads it into kind
#   3. installs the chart: 3 dedicated masters + 2 data nodes, zone awareness
#   4. waits for green; indexes documents into 3 indices (1 replica)
#      and checks the node logs carry no ANSI colour codes
#   5. force-kills the elected master while a writer indexes documents;
#      checks a new master is elected, the cluster returns to green, no
#      acknowledged write is lost and every pod and shard copy agrees on
#      the document counts
#   6. the same for the data pod holding a primary of the write index
#   7. scales data nodes 2 -> 3 -> 2 (drain with allocation filtering
#      before shrinking), checking green + doc count each time
#   8. `kubectl rollout restart` of every StatefulSet, checking the doc
#      count throughout (whenever all shards answer) and green at the end
#   9. rolling upgrade with `helm upgrade` (new image tag + JSON logs) while
#      the writer runs; checks acknowledged writes and that logs are JSON
#  10. full restart (every pod deleted at once): same PVCs, same cluster
#      UUID, same documents afterwards
#  11. runs `helm test`
#
# Environment:
#   CLUSTER_NAME   kind cluster name                  (default es-e2e)
#   IMAGE          node image                         (default elasticsearch-rs:e2e)
#   BUILD_IMAGE    1 = docker build the image first   (default 1 unless the image exists)
#   KEEP_CLUSTER   1 = do not delete the kind cluster on exit (default 0)
#   REUSE_CLUSTER  1 = reuse an existing kind cluster of that name (default 0)
#   DOCS           documents per index                (default 300)
#   TIMEOUT        seconds for each wait              (default 600)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CHART="$REPO_ROOT/cluster/deploy/helm/elasticsearch"

CLUSTER_NAME="${CLUSTER_NAME:-es-e2e}"
IMAGE="${IMAGE:-elasticsearch-rs:e2e}"
KEEP_CLUSTER="${KEEP_CLUSTER:-0}"
REUSE_CLUSTER="${REUSE_CLUSTER:-0}"
DOCS="${DOCS:-300}"
TIMEOUT="${TIMEOUT:-600}"
NS=es-e2e
RELEASE=e2e
ES_NAME=e2e                      # clusterName -> client Service "e2e"
URL="http://${ES_NAME}:9200"
CLIENT_POD=es-e2e-client
INDICES=(e2e-a e2e-b e2e-c)
WRITE_INDEX=e2e-w                # written to while pods are killed / upgraded
VALUES_IMAGE="$IMAGE"            # image the chart is deployed with
VALUES_JSON_LOGS=0               # 1 = ES_LOG_FORMAT=json on every node
WARNINGS=()
WORK="$(mktemp -d)"
KUBECTL=(kubectl --context "kind-${CLUSTER_NAME}")
HELM=(helm --kube-context "kind-${CLUSTER_NAME}")
STEP=0
CREATED_CLUSTER=0

# ------------------------------------------------------------------ output
bold() { printf '\033[1m%s\033[0m\n' "$*"; }
step() { STEP=$((STEP + 1)); echo; bold "==> [$STEP] $*"; }
info() { echo "    $*"; }
pass() { printf '    \033[32mOK\033[0m %s\n' "$*"; }
warn() { printf '    \033[33mWARN\033[0m %s\n' "$*"; WARNINGS+=("$*"); }
fail() { printf '\n\033[31mFAIL\033[0m %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------ cleanup
diagnostics() {
  echo
  bold "---- diagnostics ----"
  "${KUBECTL[@]}" get nodes -L topology.kubernetes.io/zone || true
  "${KUBECTL[@]}" -n "$NS" get pods,sts,svc,pdb,pvc -o wide || true
  "${KUBECTL[@]}" -n "$NS" get events --sort-by=.lastTimestamp | tail -40 || true
  for p in $("${KUBECTL[@]}" -n "$NS" get pods -l elasticsearch/cluster="$ES_NAME" -o name 2>/dev/null); do
    echo "---- logs $p (last 60 lines)"
    "${KUBECTL[@]}" -n "$NS" logs "$p" -c elasticsearch --tail=60 || true
  done
  es "$URL/_cat/nodes?v" 2>/dev/null || true
  es "$URL/_cat/shards?v" 2>/dev/null || true
}

cleanup() {
  local rc=$?
  set +e
  if [[ $rc -ne 0 ]]; then
    diagnostics
  fi
  rm -rf "$WORK"
  if [[ "$KEEP_CLUSTER" == "1" ]]; then
    info "keeping kind cluster ${CLUSTER_NAME} (KEEP_CLUSTER=1)"
  elif [[ "$CREATED_CLUSTER" == "1" ]]; then
    info "deleting kind cluster ${CLUSTER_NAME}"
    kind delete cluster --name "$CLUSTER_NAME" >/dev/null 2>&1
  fi
  if [[ ${#WARNINGS[@]} -gt 0 ]]; then
    bold "warnings:"; printf '    %s\n' "${WARNINGS[@]}"
  fi
  if [[ $rc -eq 0 ]]; then bold "E2E PASSED"; else bold "E2E FAILED (exit $rc)"; fi
  exit $rc
}
trap cleanup EXIT
trap 'fail "interrupted"' INT TERM

# ------------------------------------------------------------------ helpers
# es <curl args...>: curl from a pod inside the cluster (survives pod kills,
# unlike kubectl port-forward).
es() {
  "${KUBECTL[@]}" -n "$NS" exec -i "$CLIENT_POD" -- \
    curl -sS --max-time 30 -H 'Content-Type: application/json' "$@"
}

client_sh() {
  "${KUBECTL[@]}" -n "$NS" exec -i "$CLIENT_POD" -- bash -c "$1"
}

es_pods() {   # "<pod> <ip>" per Elasticsearch pod
  "${KUBECTL[@]}" -n "$NS" get pods -l elasticsearch/cluster="$ES_NAME" \
    -o jsonpath='{range .items[*]}{.metadata.name} {.status.podIP}{"\n"}{end}'
}

# wait_until <description> <command...>: retry until success or TIMEOUT.
wait_until() {
  local what="$1"; shift
  local deadline=$(( $(date +%s) + TIMEOUT ))
  until "$@" >/dev/null 2>&1; do
    if [[ $(date +%s) -ge $deadline ]]; then
      fail "timed out after ${TIMEOUT}s waiting for: $what"
    fi
    sleep 3
  done
  pass "$what"
}

health_is() {
  local want="$1"
  [[ "$(es "$URL/_cluster/health" | jq -r .status)" == "$want" ]]
}

wait_green() {
  wait_until "cluster health green" health_is green
  info "$(es "$URL/_cluster/health" | jq -c '{status,number_of_nodes,active_shards,relocating_shards,unassigned_shards}')"
}

all_pods_ready() {
  local want have
  want=$("${KUBECTL[@]}" -n "$NS" get sts -l elasticsearch/cluster="$ES_NAME" -o json | jq '[.items[].spec.replicas] | add')
  have=$("${KUBECTL[@]}" -n "$NS" get pods -l elasticsearch/cluster="$ES_NAME" -o json \
    | jq '[.items[] | select(.metadata.deletionTimestamp == null) | select(any(.status.conditions[]?; .type == "Ready" and .status == "True"))] | length')
  [[ "$want" == "$have" ]]
}

# total docs over the test indices; prints nothing unless every shard answered.
# One `_count` per index: a multi-index search needs one node holding a copy
# of every index (cross-node fan-out is phase 2 of the cluster layer).
doc_count() {
  local out idx total=0
  for idx in "${INDICES[@]}"; do
    out="$(es "$URL/$idx/_count")" || return 1
    [[ "$(jq -r '._shards.failed' <<< "$out" 2>/dev/null)" == "0" ]] || return 1
    total=$(( total + $(jq -r .count <<< "$out") ))
  done
  echo "$total"
}

assert_doc_count() {
  local c
  c="$(doc_count)" || fail "count query failed (not all shards available)"
  [[ "$c" == "$EXPECTED_DOCS" ]] || fail "document count $c, expected $EXPECTED_DOCS (data loss?)"
  pass "document count $c"
}

# ---- writer: indexes documents one at a time from the client pod and logs
# "<id> <http status>" per attempt. A 200/201 is an acknowledged write.
install_writer() {
  client_sh 'cat > /tmp/writer.sh' <<EOF
i=0
while [ ! -f /tmp/w-stop ]; do
  id="\$1-\$i"
  code=\$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 2 --max-time 10 -H 'Content-Type: application/json' \\
    -X PUT "${URL}/${WRITE_INDEX}/_doc/\$id" -d "{\\"n\\":\$i,\\"phase\\":\\"\$1\\"}")
  echo "\$id \$code" >> /tmp/w-log
  i=\$((i + 1))
done
touch /tmp/w-done
EOF
}

start_writer() {   # start_writer <phase>
  client_sh "rm -f /tmp/w-stop /tmp/w-done; : > /tmp/w-log; nohup bash /tmp/writer.sh '$1' >/dev/null 2>&1 &" < /dev/null
  sleep 5
  [[ "$(client_sh 'grep -c " 20[01]$" /tmp/w-log' < /dev/null)" -gt 0 ]] || fail "writer made no acknowledged write in 5s"
  info "writer running (phase $1)"
}

writer_done() { client_sh 'test -f /tmp/w-done' < /dev/null; }

# stop_writer <phase>: stop, fetch the log, add acknowledged ids to $WORK/acked.
stop_writer() {
  local log="$WORK/w-$1.log"
  client_sh 'touch /tmp/w-stop' < /dev/null
  wait_until "writer stopped" writer_done
  client_sh 'cat /tmp/w-log' < /dev/null > "$log"
  awk '$2 == 200 || $2 == 201 { print $1 }' "$log" >> "$WORK/acked"
  awk '{ print $1 }' "$log" >> "$WORK/attempted"
  info "phase $1: $(wc -l < "$log" | tr -d ' ') attempts, by HTTP status: $(awk '{ c[$2]++ } END { for (k in c) printf "%s=%d ", k, c[k] }' "$log")"
}

# Every acknowledged id (all phases so far) must be readable.
assert_acked_present() {
  local chunk resp missing=0 acked attempted count
  es -X POST "$URL/$WRITE_INDEX/_refresh" < /dev/null > /dev/null || fail "refresh of $WRITE_INDEX failed"
  rm -f "$WORK"/chunk-*; split -l 500 "$WORK/acked" "$WORK/chunk-"
  : > "$WORK/missing"
  for chunk in "$WORK"/chunk-*; do
    resp="$(jq -Rn '{ids: [inputs]}' < "$chunk" | es -X POST "$URL/$WRITE_INDEX/_mget" -d @-)" || fail "_mget failed"
    jq -e '.docs | type == "array"' <<< "$resp" > /dev/null || fail "_mget: unexpected response: ${resp:0:300}"
    jq -r '.docs[] | select(.found != true) | ._id' <<< "$resp" >> "$WORK/missing"
  done
  acked=$(wc -l < "$WORK/acked" | tr -d ' '); attempted=$(wc -l < "$WORK/attempted" | tr -d ' ')
  missing=$(wc -l < "$WORK/missing" | tr -d ' ')
  [[ "$missing" == "0" ]] || fail "$missing of $acked acknowledged writes are missing, e.g. $(head -5 "$WORK/missing" | tr '\n' ' ')"
  count="$(es "$URL/$WRITE_INDEX/_count" < /dev/null | jq -r .count)"
  [[ "$count" -ge "$acked" && "$count" -le "$attempted" ]] || fail "$WRITE_INDEX holds $count documents; acknowledged $acked, attempted $attempted"
  pass "all $acked acknowledged writes present ($WRITE_INDEX holds $count documents, $attempted attempted)"
}

# Every shard of the test indices: 2 STARTED copies with the same doc count.
copies_consistent() {
  local out
  es -X POST "$URL/_refresh" < /dev/null > /dev/null || return 1
  out="$(es "$URL/_cat/shards?h=index,shard,prirep,state,docs,node" < /dev/null)" || return 1
  awk '$1 ~ /^e2e-/ { k = $1 "/" $2; if ($4 != "STARTED") bad = 1; if ((k in d) && d[k] != $5) bad = 1; d[k] = $5; n[k]++ }
       END { if (length(n) == 0) bad = 1; for (k in n) if (n[k] != 2) bad = 1; exit bad }' <<< "$out"
}

# `_count` of every test index asked through every pod must agree.
assert_same_on_all_pods() {
  local pod ip idx c line ref="" pods=0
  wait_until "primary and replica of every shard hold the same number of documents" copies_consistent
  es "$URL/_cat/shards?h=index,shard,prirep,state,docs,node" < /dev/null | grep '^e2e-' | sort | sed 's/^/    /'
  while read -r pod ip <&3; do
    [[ -n "$ip" ]] || fail "pod $pod has no IP"
    line=""
    for idx in "${INDICES[@]}" "$WRITE_INDEX"; do
      c="$(es "http://$ip:9200/$idx/_count" < /dev/null | jq -r 'if ._shards.failed == 0 then .count else "ERR" end' 2>/dev/null)" || c=ERR
      line+="$idx=${c:-ERR} "
    done
    info "$pod: $line"
    [[ "$line" != *ERR* ]] || fail "count through $pod failed"
    [[ -z "$ref" ]] && ref="$line"
    [[ "$line" == "$ref" ]] || fail "pod $pod reports different counts"
    pods=$((pods + 1))
  done 3< <(es_pods)
  pass "identical document counts through all $pods pods"
}

# watch_rollout <statefulset>...: wait for the rollouts, checking the count
# whenever every shard answers.
watch_rollout() {
  local sts c deadline checks=0
  deadline=$(( $(date +%s) + TIMEOUT ))
  for sts in "$@"; do
    until "${KUBECTL[@]}" -n "$NS" rollout status "statefulset/$sts" --timeout=10s >/dev/null 2>&1; do
      [[ $(date +%s) -lt $deadline ]] || fail "rollout of $sts did not finish in ${TIMEOUT}s"
      if c="$(doc_count 2>/dev/null)"; then
        [[ "$c" == "$EXPECTED_DOCS" ]] || fail "document count $c during rollout of $sts, expected $EXPECTED_DOCS"
        checks=$((checks + 1))
      fi
    done
  done
  pass "rollout of $* complete ($checks consistent counts observed during it)"
}

current_master() {
  es "$URL/_cat/master?h=node" | tr -d '[:space:]'
}

master_is_not() {
  local m
  m="$(current_master)" || return 1
  [[ -n "$m" && "$m" != "$1" ]]
}

cluster_term() {
  es "$URL/_cluster/state/metadata?filter_path=metadata.cluster_coordination.term" | jq -r '.metadata.cluster_coordination.term // empty'
}

# A master is elected in a term newer than $1. The killed pod is recreated
# within a second under the same name (and keeps its node id on its PVC), so
# it may legitimately win the new election: the term, not the name, proves
# that an election happened.
elected_after_term() {
  local t m
  t="$(cluster_term)" || return 1
  m="$(current_master)" || return 1
  [[ -n "$m" && -n "$t" && "$t" -gt "$1" ]]
}

no_shards_on() {
  ! es "$URL/_cat/shards?h=node" | tr -d ' ' | grep -qx "$1"
}

values_file() {
  local data_replicas="$1" f="$WORK/values-${1}.yaml"
  cat > "$f" <<EOF
clusterName: ${ES_NAME}
image:
  repository: ${VALUES_IMAGE%:*}
  tag: "${VALUES_IMAGE##*:}"
  pullPolicy: Never
zoneAwareness:
  enabled: true
  setAllocationAwareness: true
nodeGroupDefaults:
  resources:
    requests: { cpu: 100m, memory: 512Mi }
    limits: { memory: 1Gi }
  persistence:
    enabled: true
    size: 1Gi
    storageClassName: standard
  antiAffinity: soft
$( [[ "$VALUES_JSON_LOGS" == "1" ]] && printf '  extraEnv:\n    - { name: ES_LOG_FORMAT, value: json }\n' )
terminationGracePeriodSeconds: 60
drainOnShutdown:
  enabled: true
  allocationDelay: 1m
  timeoutSeconds: 45
nodeGroups:
  - name: master
    replicas: 3
    roles: [master]
    clientService: false
  - name: data
    replicas: ${data_replicas}
    roles: [data, ingest]
    clientService: true
tests:
  waitForStatus: green
  timeout: 120s
EOF
  echo "$f"
}

helm_deploy() {
  local data_replicas="$1"
  "${HELM[@]}" upgrade --install "$RELEASE" "$CHART" -n "$NS" --create-namespace \
    -f "$(values_file "$data_replicas")" --wait --timeout "${TIMEOUT}s"
}

# ------------------------------------------------------------------ 0. preflight
step "preflight"
for tool in docker kind kubectl helm jq; do
  command -v "$tool" >/dev/null || fail "$tool is required"
done
docker info >/dev/null 2>&1 || fail "docker daemon is not running"
pass "docker, kind, kubectl, helm, jq present"

# ------------------------------------------------------------------ 1. kind
step "kind cluster ${CLUSTER_NAME}"
if kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
  [[ "$REUSE_CLUSTER" == "1" ]] || fail "kind cluster ${CLUSTER_NAME} exists (set REUSE_CLUSTER=1 or delete it)"
  info "reusing existing cluster"
else
  kind create cluster --name "$CLUSTER_NAME" --config "$SCRIPT_DIR/kind-config.yaml" --wait 180s
  CREATED_CLUSTER=1
fi
# kind versions without nodes[].labels support: label explicitly.
zones=(zone-a zone-b zone-c); i=0
for n in $("${KUBECTL[@]}" get nodes -l '!node-role.kubernetes.io/control-plane' -o name); do
  "${KUBECTL[@]}" label --overwrite "$n" "topology.kubernetes.io/zone=${zones[$((i % 3))]}" >/dev/null
  i=$((i + 1))
done
"${KUBECTL[@]}" get nodes -L topology.kubernetes.io/zone
# The pods bind 0.0.0.0, so ES's bootstrap checks are enforced and
# `vm.max_map_count` must be at least 262144 (the kernel default is 65530).
# The sysctl is not namespaced: setting it in a kind node sets it on the host.
for n in $(kind get nodes --name "$CLUSTER_NAME"); do
  current=$(docker exec "$n" cat /proc/sys/vm/max_map_count)
  if (( current < 262144 )); then
    docker exec "$n" sysctl -w vm.max_map_count=262144 >/dev/null || fail "vm.max_map_count is ${current}; set it to at least 262144 on the docker host"
  fi
done

# ------------------------------------------------------------------ 2. image
step "image ${IMAGE}"
if [[ "${BUILD_IMAGE:-}" == "1" ]] || { [[ -z "${BUILD_IMAGE:-}" ]] && ! docker image inspect "$IMAGE" >/dev/null 2>&1; }; then
  info "building from $REPO_ROOT (this compiles the node in release mode)"
  docker build -f "$REPO_ROOT/cluster/deploy/docker/Dockerfile" -t "$IMAGE" "$REPO_ROOT"
fi
kind load docker-image "$IMAGE" --name "$CLUSTER_NAME"
pass "image loaded into kind"

# ------------------------------------------------------------------ 3. install
step "helm install (3 masters + 2 data nodes)"
helm_deploy 2
"${KUBECTL[@]}" -n "$NS" get pods -o wide
"${KUBECTL[@]}" -n "$NS" delete pod "$CLIENT_POD" --ignore-not-found >/dev/null
"${KUBECTL[@]}" -n "$NS" run "$CLIENT_POD" --image="$IMAGE" --image-pull-policy=Never --restart=Never \
  --labels=app=es-e2e-client --command -- sleep 86400 >/dev/null
"${KUBECTL[@]}" -n "$NS" wait --for=condition=Ready "pod/$CLIENT_POD" --timeout="${TIMEOUT}s" >/dev/null
pass "client pod ready"
wait_green
es "$URL/_cat/nodes?v&h=name,node.role,master"
attrs="$(es "$URL/_cat/nodeattrs?h=node,attr,value" | grep -c ' zone ' || true)"
[[ "$attrs" -ge 5 ]] || fail "expected node.attr.zone on every node, found $attrs"
pass "node.attr.zone set on all nodes"
while read -r pod _ <&3; do
  logs="$("${KUBECTL[@]}" -n "$NS" logs "$pod" -c elasticsearch < /dev/null)"
  [[ -n "$logs" ]] || fail "no logs from $pod"
  ! grep -q $'\033\[' <<< "$logs" || fail "logs of $pod contain ANSI escape codes"
done 3< <(es_pods)
pass "node logs are plain text without ANSI colour codes"
CLUSTER_UUID="$(es "$URL/" | jq -r .cluster_uuid)"
info "cluster UUID $CLUSTER_UUID"

# ------------------------------------------------------------------ 4. data
step "index ${DOCS} documents into each of ${INDICES[*]}"
for idx in "${INDICES[@]}"; do
  es -X PUT "$URL/$idx" -d '{"settings":{"number_of_shards":1,"number_of_replicas":1}}' | jq -c .
  bulk="$WORK/bulk-$idx.ndjson"
  : > "$bulk"
  for ((d = 0; d < DOCS; d++)); do
    printf '{"index":{"_id":"%s"}}\n{"n":%d,"title":"document %d of %s","tag":"t%d"}\n' "$d" "$d" "$d" "$idx" $((d % 7)) >> "$bulk"
  done
  resp="$("${KUBECTL[@]}" -n "$NS" exec -i "$CLIENT_POD" -- curl -sS --max-time 60 \
    -H 'Content-Type: application/x-ndjson' -X POST "$URL/$idx/_bulk?refresh=true" --data-binary @- < "$bulk")"
  [[ "$(jq -r .errors <<< "$resp")" == "false" ]] || fail "bulk into $idx reported errors: $(jq -c '[.items[] | .[] | select(.error)][0]' <<< "$resp")"
done
EXPECTED_DOCS=$(( DOCS * ${#INDICES[@]} ))
es -X PUT "$URL/$WRITE_INDEX" -d '{"settings":{"number_of_shards":1,"number_of_replicas":1}}' | jq -c .
wait_green
assert_doc_count
es "$URL/_cat/shards?v"
multi="$(es "$URL/$(IFS=,; echo "${INDICES[*]}")/_count" | jq -r '.count // empty' 2>/dev/null || true)"
if [[ "$multi" == "$EXPECTED_DOCS" ]]; then pass "multi-index _count over ${INDICES[*]}: $multi"
else warn "multi-index _count returned '${multi}', expected $EXPECTED_DOCS"; fi
install_writer
: > "$WORK/acked"; : > "$WORK/attempted"

# ------------------------------------------------------------------ 5. master failover
step "kill the elected master with writes in flight"
old_master="$(current_master)"
[[ -n "$old_master" ]] || fail "no elected master"
old_term="$(cluster_term)"
old_uid="$("${KUBECTL[@]}" -n "$NS" get pod "$old_master" -o jsonpath='{.metadata.uid}')"
[[ -n "$old_term" ]] || fail "could not read the cluster term"
start_writer master-kill
info "elected master: $old_master (term $old_term) — force-deleting its pod"
"${KUBECTL[@]}" -n "$NS" delete pod "$old_master" --grace-period=0 --force >/dev/null 2>&1
wait_until "a master is elected in a term after $old_term" elected_after_term "$old_term"
[[ "$("${KUBECTL[@]}" -n "$NS" get pod "$old_master" -o jsonpath='{.metadata.uid}' 2>/dev/null)" != "$old_uid" ]] || fail "pod $old_master was not replaced"
info "master after the election: $(current_master) (term $(cluster_term))"
wait_until "all pods ready again" all_pods_ready
wait_green
sleep 5
stop_writer master-kill
assert_doc_count
assert_acked_present
assert_same_on_all_pods

# ------------------------------------------------------------------ 6. data pod failure
step "kill a data pod with writes in flight"
victim="$(es "$URL/_cat/shards/$WRITE_INDEX?h=shard,prirep,node" | awk '$1 == 0 && $2 == "p" { print $3 }')"
[[ -n "$victim" ]] || fail "no primary found for $WRITE_INDEX shard 0"
start_writer data-kill
info "$victim holds the primary of $WRITE_INDEX shard 0 — force-deleting its pod"
"${KUBECTL[@]}" -n "$NS" delete pod "$victim" --grace-period=0 --force >/dev/null 2>&1
sleep 5
wait_until "all pods ready again" all_pods_ready
wait_green
sleep 5
stop_writer data-kill
assert_doc_count
assert_acked_present
assert_same_on_all_pods

# ------------------------------------------------------------------ 7. scale data nodes
step "scale data nodes 2 -> 3"
helm_deploy 3
wait_until "all pods ready" all_pods_ready
wait_green
assert_doc_count
es "$URL/_cat/allocation?v"

step "scale data nodes 3 -> 2 (drain e2e-data-2 first)"
leaving="${ES_NAME}-data-2"
es -X PUT "$URL/_cluster/settings" -d "{\"persistent\":{\"cluster.routing.allocation.exclude._name\":\"${leaving}\"}}" | jq -c .
wait_until "no shard left on ${leaving}" no_shards_on "$leaving"
helm_deploy 2
es -X PUT "$URL/_cluster/settings" -d '{"persistent":{"cluster.routing.allocation.exclude._name":null}}' | jq -c .
wait_until "all pods ready" all_pods_ready
wait_green
assert_doc_count
[[ "$(es "$URL/_cat/nodes?h=name" | grep -c "${ES_NAME}-data-")" == "2" ]] || fail "expected 2 data nodes"
pass "2 data nodes"

# ------------------------------------------------------------------ 8. rolling restart
step "rolling restart of every StatefulSet"
for sts in "${ES_NAME}-data" "${ES_NAME}-master"; do
  info "kubectl rollout restart statefulset/$sts"
  "${KUBECTL[@]}" -n "$NS" rollout restart "statefulset/$sts" >/dev/null
  watch_rollout "$sts"
  wait_green
  assert_doc_count
done
assert_acked_present
assert_same_on_all_pods

# ------------------------------------------------------------------ 9. rolling upgrade
step "rolling upgrade (helm upgrade: new image tag, JSON logs) with writes in flight"
VALUES_IMAGE="${IMAGE}-v2"
VALUES_JSON_LOGS=1
docker tag "$IMAGE" "$VALUES_IMAGE"
kind load docker-image "$VALUES_IMAGE" --name "$CLUSTER_NAME"
start_writer upgrade
"${HELM[@]}" upgrade "$RELEASE" "$CHART" -n "$NS" -f "$(values_file 2)" --timeout "${TIMEOUT}s" >/dev/null
watch_rollout "${ES_NAME}-data" "${ES_NAME}-master"
wait_until "all pods ready" all_pods_ready
wait_green
sleep 5
stop_writer upgrade
while read -r pod _ <&3; do
  img="$("${KUBECTL[@]}" -n "$NS" get pod "$pod" -o jsonpath='{.spec.containers[?(@.name=="elasticsearch")].image}' < /dev/null)"
  [[ "$img" == "$VALUES_IMAGE" ]] || fail "$pod runs $img, expected $VALUES_IMAGE"
  logs="$("${KUBECTL[@]}" -n "$NS" logs "$pod" -c elasticsearch < /dev/null)"
  total=$(grep -c . <<< "$logs" || true)
  json=$(jq -R 'fromjson? | select(type == "object" and has("timestamp") and has("level"))' <<< "$logs" | jq -s length)
  ! grep -q $'\033\[' <<< "$logs" || fail "logs of $pod contain ANSI escape codes"
  [[ "$json" -gt 0 && $((total - json)) -le 2 ]] || fail "$pod: only $json of $total log lines are JSON objects"
  info "$pod: image $img, $json of $total log lines are JSON objects"
done 3< <(es_pods)
pass "every pod runs $VALUES_IMAGE and logs JSON"
assert_doc_count
assert_acked_present
assert_same_on_all_pods

# ------------------------------------------------------------------ 10. full restart
step "full restart: delete every pod at once, data must come back from the PVCs"
pvcs_before="$("${KUBECTL[@]}" -n "$NS" get pvc -o jsonpath='{range .items[*]}{.metadata.name}={.metadata.uid}={.spec.volumeName}{"\n"}{end}' | sort)"
pods_before="$("${KUBECTL[@]}" -n "$NS" get pods -l elasticsearch/cluster="$ES_NAME" -o jsonpath='{range .items[*]}{.metadata.uid}{"\n"}{end}' | sort)"
write_count_before="$(es "$URL/$WRITE_INDEX/_count" | jq -r .count)"
"${KUBECTL[@]}" -n "$NS" delete pod -l elasticsearch/cluster="$ES_NAME" --wait=true --timeout="${TIMEOUT}s" >/dev/null
pods_after="$("${KUBECTL[@]}" -n "$NS" get pods -l elasticsearch/cluster="$ES_NAME" -o jsonpath='{range .items[*]}{.metadata.uid}{"\n"}{end}' | sort)"
[[ -z "$(comm -12 <(echo "$pods_before") <(echo "$pods_after"))" ]] || fail "some pods were not replaced"
pass "all $(wc -l <<< "$pods_before" | tr -d ' ') pods deleted and replaced"
wait_until "all pods ready" all_pods_ready
wait_green
pvcs_after="$("${KUBECTL[@]}" -n "$NS" get pvc -o jsonpath='{range .items[*]}{.metadata.name}={.metadata.uid}={.spec.volumeName}{"\n"}{end}' | sort)"
[[ "$pvcs_before" == "$pvcs_after" ]] || fail "PVCs changed across the restart"
pass "same $(wc -l <<< "$pvcs_after" | tr -d ' ') PVCs / volumes"
uuid="$(es "$URL/" | jq -r .cluster_uuid)"
[[ "$uuid" == "$CLUSTER_UUID" ]] || fail "cluster UUID changed: $CLUSTER_UUID -> $uuid (a new cluster was bootstrapped)"
pass "cluster UUID unchanged ($uuid)"
assert_doc_count
assert_acked_present
[[ "$(es "$URL/$WRITE_INDEX/_count" | jq -r .count)" == "$write_count_before" ]] || fail "$WRITE_INDEX count changed across the restart"
pass "$WRITE_INDEX still holds $write_count_before documents"
assert_same_on_all_pods

# ------------------------------------------------------------------ 11. helm test
step "helm test"
"${HELM[@]}" test "$RELEASE" -n "$NS" --timeout "${TIMEOUT}s" --logs
pass "helm test"
