#!/usr/bin/env bash
# End-to-end test: the Rust engine under the STOCK Elastic Cloud on
# Kubernetes operator (unmodified, installed from Elastic's published
# manifests) on a local kind cluster. Nothing rustic-specific runs in the
# cluster except the node image named in `spec.image`.
#
# Requires: docker (running; the user in the docker group), kind, kubectl,
# jq, curl, openssl. Run from anywhere:
#
#   cluster/deploy/eck/e2e.sh
#
# Steps:
#   1. kind cluster (1 control plane + 3 workers)
#   2. node image: docker build (or reuse), `kind load`
#   3. stock ECK: crds.yaml + operator.yaml from download.elastic.co
#   4. a plain `Elasticsearch` resource (3 nodes, spec.image = our image);
#      waits for ECK to report it Ready and green
#   5. HTTPS through the ECK `-es-http` Service, verifying ECK's CA and the
#      service name, as `elastic` with the password ECK generated; checks
#      `GET /`, `_cat/nodes`, the file-realm user and that wrong passwords fail
#   6. indexes documents (1 replica) and searches them
#   7. scale 3 -> 5 -> 3 (ECK: node shutdown `remove` + voting exclusions)
#      keeping every document
#   8. rolling config change (a `node.attr` in `nodeSets[].config`): every
#      pod restarted by ECK, one at a time (node shutdown `restart`); the
#      new attribute is visible on every node and no document is lost
#   9. rolling image change (same binary, new tag), as a version upgrade is
#  10. a pod force-killed (no pre-stop hook); ECK/StatefulSet recreate it,
#      the cluster returns to green with every document
#  11. certificate rotation: ECK re-issues the HTTP certificate (its
#      internal Secret deleted); the running pods serve the new one
#  12. optional (KIBANA=1): a stock `Kibana` resource pointing at the
#      cluster (elastic/kibana service account token from ECK) reaches
#      `available`
#
# Environment:
#   CLUSTER_NAME   kind cluster name                     (default eck-e2e)
#   IMAGE          node image                            (default elasticsearch-rs:eck)
#   BUILD_IMAGE    1 = docker build the image first      (default 1 unless the image exists)
#   PREBUILT_DIR   build the image from these binaries   (elasticsearch,
#                  elasticsearch-reset-password, elasticsearch-keystore,
#                  elasticsearch-create-enrollment-token;
#                  default: compile inside docker). With it, RUNTIME_IMAGE
#                  must carry the glibc they were linked against.
#   RUNTIME_IMAGE  runtime base for PREBUILT_DIR         (default ubuntu:24.04)
#   ECK_VERSION    stock ECK release                     (default 3.5.0)
#   ES_VERSION     spec.version                          (default 9.5.5)
#   KIBANA         1 = also run the Kibana step          (default 0)
#   KIBANA_IMAGE   Kibana image (multi-arch)             (default docker.elastic.co/kibana/kibana:$ES_VERSION)
#   KEEP_CLUSTER   1 = keep the kind cluster on exit     (default 0)
#   REUSE_CLUSTER  1 = reuse an existing kind cluster    (default 0)
#   DOCS           documents indexed                     (default 500)
#   TIMEOUT        seconds for each wait                 (default 900)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CLUSTER_NAME="${CLUSTER_NAME:-eck-e2e}"
IMAGE="${IMAGE:-elasticsearch-rs:eck}"
ECK_VERSION="${ECK_VERSION:-3.5.0}"
ES_VERSION="${ES_VERSION:-9.5.5}"
KIBANA="${KIBANA:-0}"
KIBANA_IMAGE="${KIBANA_IMAGE:-docker.elastic.co/kibana/kibana:${ES_VERSION}}"
KEEP_CLUSTER="${KEEP_CLUSTER:-0}"
REUSE_CLUSTER="${REUSE_CLUSTER:-0}"
DOCS="${DOCS:-500}"
TIMEOUT="${TIMEOUT:-900}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-ubuntu:24.04}"
NS=eck-e2e
ES=rustic
INDEX=e2e
KUBECTL=(kubectl --context "kind-${CLUSTER_NAME}")
WORK="$(mktemp -d)"
STEP=0
PASSED=0
CREATED_CLUSTER=0
PF_PID=""
LOCAL_PORT="${LOCAL_PORT:-19200}"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
step() { STEP=$((STEP + 1)); echo; bold "==> [$STEP] $*  ($(date -u +%H:%M:%S))"; }
info() { echo "    $*"; }
pass() { PASSED=$((PASSED + 1)); printf '    \033[32mOK\033[0m %s\n' "$*"; }
fail() { printf '\n\033[31mFAIL\033[0m [step %s] %s\n' "$STEP" "$*" >&2; exit 1; }

diagnostics() {
  echo
  bold "---- diagnostics ----"
  "${KUBECTL[@]}" -n "$NS" get elasticsearch,kibana,pods,sts,svc,pvc -o wide 2>/dev/null || true
  "${KUBECTL[@]}" -n "$NS" get events --sort-by=.lastTimestamp 2>/dev/null | tail -40 || true
  for p in $("${KUBECTL[@]}" -n "$NS" get pods -l elasticsearch.k8s.elastic.co/cluster-name="$ES" -o name 2>/dev/null); do
    echo "--- $p (init: elastic-internal-init-filesystem)"
    "${KUBECTL[@]}" -n "$NS" logs "$p" -c elastic-internal-init-filesystem --tail=20 2>/dev/null || true
    echo "--- $p"
    "${KUBECTL[@]}" -n "$NS" logs "$p" -c elasticsearch --tail=60 2>/dev/null || true
    "${KUBECTL[@]}" -n "$NS" logs "$p" -c elasticsearch --previous --tail=30 2>/dev/null || true
  done
  echo "--- ECK operator"
  "${KUBECTL[@]}" -n elastic-system logs statefulset/elastic-operator --tail=80 2>/dev/null || true
}

cleanup() {
  local rc=$?
  [[ -n "$PF_PID" ]] && kill "$PF_PID" 2>/dev/null || true
  if [[ $rc -ne 0 ]]; then diagnostics; fi
  if [[ $CREATED_CLUSTER -eq 1 && $KEEP_CLUSTER -ne 1 && $rc -eq 0 ]]; then
    kind delete cluster --name "$CLUSTER_NAME" >/dev/null 2>&1 || true
  fi
  rm -rf "$WORK"
  echo
  if [[ $rc -eq 0 ]]; then bold "PASSED: $PASSED checks in $STEP steps"; else bold "FAILED after $PASSED passed checks (step $STEP); cluster kept: kind-${CLUSTER_NAME}"; fi
  exit $rc
}
trap cleanup EXIT

# Wait until `cmd` succeeds, at most TIMEOUT seconds.
wait_for() { # description cmd...
  local what="$1"; shift
  local deadline=$((SECONDS + TIMEOUT))
  until "$@" >/dev/null 2>&1; do
    (( SECONDS < deadline )) || fail "timed out after ${TIMEOUT}s waiting for: $what"
    sleep 5
  done
}

# ------------------------------------------------------------------ REST
# Through `kubectl port-forward` to the ECK `-es-http` Service, verifying
# the ECK CA and the Service's DNS name (`--connect-to`).
start_port_forward() {
  [[ -n "$PF_PID" ]] && kill "$PF_PID" 2>/dev/null || true
  "${KUBECTL[@]}" -n "$NS" port-forward "svc/${ES}-es-http" "${LOCAL_PORT}:9200" >"$WORK/pf.log" 2>&1 &
  PF_PID=$!
  sleep 2
}
es() { # METHOD PATH [curl args...]
  local method="$1" path="$2"; shift 2
  curl -sS --max-time 60 --cacert "$WORK/ca.crt" \
    --connect-to "${ES}-es-http.${NS}.svc:9200:127.0.0.1:${LOCAL_PORT}" \
    -u "elastic:${PASSWORD}" -X "$method" "https://${ES}-es-http.${NS}.svc:9200${path}" "$@"
}
es_ok() { # retries through port-forward restarts (pods behind the Service change)
  local out
  for _ in 1 2 3 4 5; do
    if out="$(es "$@" -f 2>/dev/null)"; then printf '%s' "$out"; return 0; fi
    kill -0 "$PF_PID" 2>/dev/null || start_port_forward
    sleep 3
  done
  es "$@"; return 1
}

es_phase() { "${KUBECTL[@]}" -n "$NS" get elasticsearch "$ES" -o jsonpath='{.status.phase}/{.status.health}/{.status.availableNodes}' 2>/dev/null; }
es_ready() { # nodes
  [[ "$(es_phase)" == "Ready/green/$1" ]]
}
cluster_green_with() { # nodes
  local h; h="$(es_ok GET '/_cluster/health?wait_for_status=green&timeout=5s')" || return 1
  [[ "$(jq -r .status <<<"$h")" == green && "$(jq -r .number_of_nodes <<<"$h")" == "$1" ]]
}
doc_count() { es_ok GET "/${INDEX}/_count" | jq -r .count; }
check_docs() { # expected
  local n; n="$(doc_count)"
  [[ "$n" == "$1" ]] || fail "expected $1 documents, found $n"
  pass "$n documents"
}
pods_revision() { "${KUBECTL[@]}" -n "$NS" get pods -l elasticsearch.k8s.elastic.co/cluster-name="$ES" -o jsonpath='{range .items[*]}{.metadata.labels.controller-revision-hash}{"\n"}{end}' | sort -u; }
sts_revision() { "${KUBECTL[@]}" -n "$NS" get sts "${ES}-es-default" -o jsonpath='{.status.updateRevision}'; }
rolled_out() { # every pod at the StatefulSet's update revision
  local want; want="$(sts_revision)"
  [[ -n "$want" && "$(pods_revision)" == "$want" ]]
}

manifest() { # count image [extra config yaml lines]
  local count="$1" image="$2" config="${3:-}"
  cat <<EOF
apiVersion: elasticsearch.k8s.elastic.co/v1
kind: Elasticsearch
metadata:
  name: ${ES}
  namespace: ${NS}
spec:
  version: ${ES_VERSION}
  image: ${image}
  nodeSets:
    - name: default
      count: ${count}
      config:
        node.attr.e2e: "true"${config}
      podTemplate:
        spec:
          containers:
            - name: elasticsearch
              resources:
                requests: {memory: 2Gi, cpu: 500m}
                limits: {memory: 2Gi}
      volumeClaimTemplates:
        - metadata:
            name: elasticsearch-data
          spec:
            accessModes: [ReadWriteOnce]
            resources:
              requests:
                storage: 2Gi
EOF
}

# ======================================================================
step "kind cluster ${CLUSTER_NAME}"
for t in docker kind kubectl jq curl openssl; do command -v "$t" >/dev/null || fail "missing $t"; done
if kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
  [[ $REUSE_CLUSTER -eq 1 ]] || fail "kind cluster $CLUSTER_NAME exists (REUSE_CLUSTER=1 to reuse it)"
  info "reusing"
else
  cat >"$WORK/kind.yaml" <<'EOF'
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
  - role: worker
  - role: worker
EOF
  kind create cluster --name "$CLUSTER_NAME" --config "$WORK/kind.yaml"
  CREATED_CLUSTER=1
fi
pass "kind-${CLUSTER_NAME}"

step "node image ${IMAGE}"
BUILD_IMAGE="${BUILD_IMAGE:-$(docker image inspect "$IMAGE" >/dev/null 2>&1 && echo 0 || echo 1)}"
if [[ "$BUILD_IMAGE" == 1 ]]; then
  if [[ -n "${PREBUILT_DIR:-}" ]]; then
    dst="$REPO_ROOT/cluster/deploy/docker/.prebuilt"
    mkdir -p "$dst"
    for b in elasticsearch elasticsearch-reset-password elasticsearch-keystore elasticsearch-create-enrollment-token; do install -m 0755 "$PREBUILT_DIR/$b" "$dst/$b"; done
    DOCKER_BUILDKIT=1 docker build -f "$REPO_ROOT/cluster/deploy/docker/Dockerfile" \
      --build-arg BINARIES=prebuilt --build-arg RUNTIME_IMAGE="$RUNTIME_IMAGE" -t "$IMAGE" "$REPO_ROOT"
  else
    DOCKER_BUILDKIT=1 docker build -f "$REPO_ROOT/cluster/deploy/docker/Dockerfile" -t "$IMAGE" "$REPO_ROOT"
  fi
fi
docker image inspect "$IMAGE" >/dev/null || fail "no image $IMAGE"
# The same bits under a second tag, for the rolling image change.
IMAGE2="${IMAGE%:*}:eck-rolled"
docker tag "$IMAGE" "$IMAGE2"
kind load docker-image "$IMAGE" "$IMAGE2" --name "$CLUSTER_NAME"
pass "loaded $IMAGE and $IMAGE2"

step "stock ECK ${ECK_VERSION} from download.elastic.co"
"${KUBECTL[@]}" create -f "https://download.elastic.co/downloads/eck/${ECK_VERSION}/crds.yaml" 2>/dev/null \
  || "${KUBECTL[@]}" replace -f "https://download.elastic.co/downloads/eck/${ECK_VERSION}/crds.yaml"
"${KUBECTL[@]}" apply -f "https://download.elastic.co/downloads/eck/${ECK_VERSION}/operator.yaml"
"${KUBECTL[@]}" -n elastic-system rollout status statefulset/elastic-operator --timeout="${TIMEOUT}s"
pass "elastic-operator $("${KUBECTL[@]}" -n elastic-system get sts elastic-operator -o jsonpath='{.spec.template.spec.containers[0].image}')"

step "Elasticsearch resource: 3 nodes, spec.image=${IMAGE}"
if "${KUBECTL[@]}" get namespace "$NS" >/dev/null 2>&1; then
  info "deleting the namespace of a previous run"
  "${KUBECTL[@]}" delete namespace "$NS" --wait=true --timeout="${TIMEOUT}s"
fi
"${KUBECTL[@]}" create namespace "$NS" --dry-run=client -o yaml | "${KUBECTL[@]}" apply -f -
manifest 3 "$IMAGE" | "${KUBECTL[@]}" apply -f -
wait_for "ECK reports Ready/green/3" es_ready 3
pass "phase/health/nodes = $(es_phase)"

step "HTTPS with ECK's CA and the ECK-generated elastic password"
"${KUBECTL[@]}" -n "$NS" get secret "${ES}-es-http-certs-public" -o jsonpath='{.data.ca\.crt}' | base64 -d >"$WORK/ca.crt"
PASSWORD="$("${KUBECTL[@]}" -n "$NS" get secret "${ES}-es-elastic-user" -o jsonpath='{.data.elastic}' | base64 -d)"
[[ -n "$PASSWORD" ]] || fail "no elastic password"
start_port_forward
root="$(es_ok GET /)" || fail "GET / failed: $root"
info "$(jq -c '{name, cluster_name, version: .version.number}' <<<"$root")"
[[ "$(jq -r .cluster_name <<<"$root")" == "$ES" ]] || fail "cluster_name"
pass "GET / over verified TLS"
who="$(es_ok GET /_security/_authenticate)"
[[ "$(jq -r .username <<<"$who")" == elastic && "$(jq -r '.roles|index("superuser")' <<<"$who")" != null ]] || fail "authenticate: $who"
pass "elastic is a superuser from realm $(jq -r .authentication_realm.type <<<"$who")"
code="$(curl -s -o /dev/null -w '%{http_code}' --cacert "$WORK/ca.crt" --connect-to "${ES}-es-http.${NS}.svc:9200:127.0.0.1:${LOCAL_PORT}" -u elastic:wrong "https://${ES}-es-http.${NS}.svc:9200/")"
[[ "$code" == 401 ]] || fail "wrong password answered $code"
pass "wrong password -> 401"
info "$(es_ok GET '/_cat/nodes?h=name,node.role,master,version' | tr '\n' ';')"
catn="$(es_ok GET '/_cat/nodes?full_id=true&h=id,name')"
# ECK's pre-stop hook greps `<id> <name>` lines out of this.
[[ "$(grep -cE "^[A-Za-z0-9_-]{22} ${ES}-es-default-[0-9]+\$" <<<"$catn")" -eq 3 ]] || fail "_cat/nodes: $catn"
pass "_cat/nodes lists 3 nodes"

step "index ${DOCS} documents (1 primary shard x 3, 1 replica) and search"
es_ok PUT "/${INDEX}" -H 'Content-Type: application/json' -d '{"settings":{"number_of_shards":3,"number_of_replicas":1}}' >/dev/null
for ((i = 0; i < DOCS; i += 100)); do
  : >"$WORK/bulk"
  for ((j = i; j < i + 100 && j < DOCS; j++)); do
    printf '{"index":{"_id":"%d"}}\n{"n":%d,"title":"doc %d","tag":"t%d"}\n' "$j" "$j" "$j" $((j % 5)) >>"$WORK/bulk"
  done
  r="$(es_ok POST "/${INDEX}/_bulk" -H 'Content-Type: application/x-ndjson' --data-binary @"$WORK/bulk")"
  [[ "$(jq -r .errors <<<"$r")" == false ]] || fail "bulk errors: $(head -c 400 <<<"$r")"
done
es_ok POST "/${INDEX}/_refresh" >/dev/null
check_docs "$DOCS"
hits="$(es_ok POST "/${INDEX}/_search" -H 'Content-Type: application/json' -d '{"query":{"term":{"tag.keyword":"t3"}},"size":0,"track_total_hits":true}' | jq -r .hits.total.value)"
[[ "$hits" == $((DOCS / 5)) ]] || fail "search t3: $hits"
pass "search: $hits hits for tag t3"
wait_for "green with 3 nodes" cluster_green_with 3
pass "green"

step "scale 3 -> 5"
manifest 5 "$IMAGE" | "${KUBECTL[@]}" apply -f -
wait_for "ECK reports Ready/green/5" es_ready 5
wait_for "green with 5 nodes" cluster_green_with 5
check_docs "$DOCS"

step "scale 5 -> 3"
manifest 3 "$IMAGE" | "${KUBECTL[@]}" apply -f -
wait_for "5 -> 3 pods" bash -c "[[ \$(${KUBECTL[*]} -n $NS get pods -l elasticsearch.k8s.elastic.co/cluster-name=$ES --no-headers | wc -l) -eq 3 ]]"
wait_for "ECK reports Ready/green/3" es_ready 3
wait_for "green with 3 nodes" cluster_green_with 3
check_docs "$DOCS"
excl="$(es_ok GET '/_cluster/state/metadata?filter_path=metadata.cluster_coordination.voting_config_exclusions' | jq -c '.metadata.cluster_coordination.voting_config_exclusions // []')"
info "voting exclusions after scale-down: $excl"
pass "scaled down"

step "rolling config change (node.attr.e2e_rack)"
before="$(sts_revision)"
manifest 3 "$IMAGE" $'\n        node.attr.e2e_rack: r1' | "${KUBECTL[@]}" apply -f -
wait_for "new StatefulSet revision" bash -c "[[ \$(${KUBECTL[*]} -n $NS get sts ${ES}-es-default -o jsonpath='{.status.updateRevision}') != '$before' ]]"
wait_for "every pod restarted at the new revision" rolled_out
wait_for "ECK reports Ready/green/3" es_ready 3
wait_for "green with 3 nodes" cluster_green_with 3
attrs="$(es_ok GET '/_nodes/_all/no-metrics' | jq -r '[.nodes[].attributes.e2e_rack] | unique | join(",")')"
[[ "$attrs" == r1 ]] || fail "node.attr.e2e_rack on every node: [$attrs]"
pass "every node has node.attr.e2e_rack=r1"
check_docs "$DOCS"

step "rolling image change (${IMAGE2})"
before="$(sts_revision)"
manifest 3 "$IMAGE2" $'\n        node.attr.e2e_rack: r1' | "${KUBECTL[@]}" apply -f -
wait_for "new StatefulSet revision" bash -c "[[ \$(${KUBECTL[*]} -n $NS get sts ${ES}-es-default -o jsonpath='{.status.updateRevision}') != '$before' ]]"
wait_for "every pod restarted at the new revision" rolled_out
wait_for "ECK reports Ready/green/3" es_ready 3
wait_for "green with 3 nodes" cluster_green_with 3
pass "pods run $("${KUBECTL[@]}" -n "$NS" get pods -l elasticsearch.k8s.elastic.co/cluster-name="$ES" -o jsonpath='{.items[*].spec.containers[0].image}')"
check_docs "$DOCS"

step "force-kill a pod (no pre-stop hook)"
victim="${ES}-es-default-1"
uid="$("${KUBECTL[@]}" -n "$NS" get pod "$victim" -o jsonpath='{.metadata.uid}')"
"${KUBECTL[@]}" -n "$NS" delete pod "$victim" --grace-period=0 --force
wait_for "$victim recreated" bash -c "[[ \$(${KUBECTL[*]} -n $NS get pod $victim -o jsonpath='{.metadata.uid}') != '$uid' ]]"
wait_for "ECK reports Ready/green/3" es_ready 3
wait_for "green with 3 nodes" cluster_green_with 3
check_docs "$DOCS"

step "certificate rotation without a restart (ECK re-issues the HTTP certificate)"
served_cert() {
  kill -0 "$PF_PID" 2>/dev/null || start_port_forward
  openssl s_client -connect "127.0.0.1:${LOCAL_PORT}" -servername "${ES}-es-http.${NS}.svc" </dev/null 2>/dev/null \
    | openssl x509 -noout -fingerprint -sha256 2>/dev/null | cut -d= -f2
}
pods_uids() { "${KUBECTL[@]}" -n "$NS" get pods -l elasticsearch.k8s.elastic.co/cluster-name="$ES" -o jsonpath='{range .items[*]}{.metadata.uid}{"\n"}{end}' | sort; }
before_cert="$(served_cert)"; before_pods="$(pods_uids)"
[[ -n "$before_cert" ]] || fail "no certificate served"
"${KUBECTL[@]}" -n "$NS" delete secret "${ES}-es-http-certs-internal"
wait_for "a new HTTP certificate served" bash -c "c=\$(openssl s_client -connect 127.0.0.1:${LOCAL_PORT} -servername ${ES}-es-http.${NS}.svc </dev/null 2>/dev/null | openssl x509 -noout -fingerprint -sha256 2>/dev/null | cut -d= -f2); [[ -n \"\$c\" && \"\$c\" != '$before_cert' ]]"
[[ "$(pods_uids)" == "$before_pods" ]] || fail "pods were restarted"
es_ok GET /_cluster/health >/dev/null || fail "HTTPS with the rotated certificate"
pass "new certificate $(served_cert | cut -c1-23)... served by the running pods"

if [[ "$KIBANA" == 1 ]]; then
  step "stock Kibana resource (${KIBANA_IMAGE}) against the cluster"
  # The kind nodes pull it (`kind load` of a multi-arch image fails with
  # "content digest ... not found").
  cat <<EOF | "${KUBECTL[@]}" apply -f -
apiVersion: kibana.k8s.elastic.co/v1
kind: Kibana
metadata:
  name: kb
  namespace: ${NS}
spec:
  version: ${ES_VERSION}
  image: ${KIBANA_IMAGE}
  count: 1
  elasticsearchRef:
    name: ${ES}
EOF
  wait_for "Kibana green" bash -c "[[ \$(${KUBECTL[*]} -n $NS get kibana kb -o jsonpath='{.status.health}') == green ]]"
  pass "Kibana health green"
fi
