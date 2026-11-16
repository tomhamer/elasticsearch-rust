#!/usr/bin/env bash
# Create the two Secrets the kustomize overlays need, once per namespace,
# before `kubectl apply -k cluster/deploy/k8s/overlays/<overlay>`:
#
#   elasticsearch-certs        CA + one certificate for HTTP and transport
#                              (tls.crt, tls.key, ca.crt), from es-certgen
#   elasticsearch-credentials  `password`: the elastic user's bootstrap
#                              password (ELASTIC_PASSWORD, or random)
#
#   cluster/deploy/k8s/create-secrets.sh [NAMESPACE]      # default: elasticsearch
#
# Existing Secrets are left alone. Requires kubectl and openssl. To use your
# own PKI instead, create `elasticsearch-certs` with the same keys; its
# certificate must name the client Service `elasticsearch` (the probes
# verify it).
set -euo pipefail

NS="${1:-elasticsearch}"
KUBECTL=(${KUBECTL:-kubectl})
DOMAIN="${CLUSTER_DOMAIN:-cluster.local}"
CERTGEN="$(cd "$(dirname "${BASH_SOURCE[0]}")/../docker/bin" && pwd)/es-certgen"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

"${KUBECTL[@]}" get namespace "$NS" >/dev/null 2>&1 || "${KUBECTL[@]}" create namespace "$NS"

if "${KUBECTL[@]}" -n "$NS" get secret elasticsearch-certs >/dev/null 2>&1; then
  echo "secret $NS/elasticsearch-certs exists; keeping it"
else
  bash "$CERTGEN" ca "$WORK/ca" --cn "elasticsearch ${NS} CA" >/dev/null
  dns="elasticsearch,elasticsearch.${NS},elasticsearch.${NS}.svc,elasticsearch.${NS}.svc.${DOMAIN}"
  dns+=",elasticsearch-headless,elasticsearch-headless.${NS}.svc,*.elasticsearch-headless.${NS}.svc,*.elasticsearch-headless.${NS}.svc.${DOMAIN},localhost"
  bash "$CERTGEN" cert "$WORK/ca" "$WORK/node" node --dns "$dns" --ip 127.0.0.1 >/dev/null
  "${KUBECTL[@]}" -n "$NS" create secret generic elasticsearch-certs \
    --from-file=tls.crt="$WORK/node/node.crt" --from-file=tls.key="$WORK/node/node.key" \
    --from-file=ca.crt="$WORK/ca/ca.crt"
fi

if "${KUBECTL[@]}" -n "$NS" get secret elasticsearch-credentials >/dev/null 2>&1; then
  echo "secret $NS/elasticsearch-credentials exists; keeping it"
else
  pw="${ELASTIC_PASSWORD:-$(openssl rand -base64 48 | LC_ALL=C tr -dc 'A-Za-z0-9' | cut -c1-24)}"
  "${KUBECTL[@]}" -n "$NS" create secret generic elasticsearch-credentials \
    --from-literal=username=elastic --from-file=password=<(printf '%s' "$pw")
fi

cat <<EOF
Secrets ready in namespace ${NS}. After applying an overlay:
  PASSWORD=\$(kubectl -n ${NS} get secret elasticsearch-credentials -o go-template='{{.data.password | base64decode}}')
  kubectl -n ${NS} get secret elasticsearch-certs -o go-template='{{index .data "ca.crt" | base64decode}}' > ca.crt
  kubectl -n ${NS} port-forward svc/elasticsearch 9200 &
  curl --cacert ca.crt -u "elastic:\$PASSWORD" https://localhost:9200/_cluster/health?pretty
EOF
