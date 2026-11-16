#!/bin/bash
# Renders the chart's `es.formed` decision (templates/_helpers.tpl) over
# fixture objects with `helm template`, since `lookup` returns nothing
# offline. Production review #416: an existing master StatefulSet whose pods
# never turned ready is not evidence that the cluster formed.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/c/templates"
printf 'apiVersion: v2\nname: formed-test\nversion: 0.0.1\n' > "$work/c/Chart.yaml"
cp "$here/elasticsearch/templates/_helpers.tpl" "$work/c/templates/"
cat > "$work/c/templates/cases.yaml" <<'T'
{{- range .Values.cases }}
{{ .name }}: {{ include "es.formed" (dict "clusterBootstrapped" .clusterBootstrapped "configMap" .configMap "masterStatefulSets" .sts) | default "false" | quote }}
{{- end }}
T
cat > "$work/values.yaml" <<'V'
cases:
  - {name: fresh, clusterBootstrapped: false, configMap: {}, sts: []}
  - {name: never_started, clusterBootstrapped: false, configMap: {metadata: {name: es-bootstrap}}, sts: [{metadata: {name: es-master}, status: {replicas: 3}}]}
  - {name: never_ready, clusterBootstrapped: false, configMap: {metadata: {name: es-bootstrap}}, sts: [{metadata: {name: es-master}, status: {replicas: 3, readyReplicas: 0}}]}
  - {name: ready_master, clusterBootstrapped: false, configMap: {}, sts: [{}, {metadata: {name: es-master}, status: {readyReplicas: 1}}]}
  - {name: recorded, clusterBootstrapped: false, configMap: {metadata: {annotations: {elasticsearch/cluster-formed: "true"}}}, sts: [{metadata: {name: es-master}, status: {readyReplicas: 0}}]}
  - {name: explicit, clusterBootstrapped: true, configMap: {}, sts: []}
V
out="$(helm template t "$work/c" -f "$work/values.yaml" --show-only templates/cases.yaml)"
want='fresh: "false"
never_started: "false"
never_ready: "false"
ready_master: "true"
recorded: "true"
explicit: "true"'
got="$(grep -v '^---\|^#\|^$' <<<"$out")"
if [[ "$got" != "$want" ]]; then
  echo "FAIL: es.formed decisions differ" >&2
  diff <(echo "$want") <(echo "$got") >&2 || true
  exit 1
fi
# The full chart still renders (offline: bootstrap list present).
helm template t "$here/elasticsearch" --show-only templates/configmap.yaml | grep -q 'cluster.initial_master_nodes' || { echo "FAIL: offline render lacks the bootstrap list" >&2; exit 1; }
echo "PASS: 6 es.formed cases + offline render"
