#!/usr/bin/env bash
# End-to-end check of the secure docker-compose.yml: the setup service
# creates the CA, certificates and password; es01..es03 form a green cluster
# that requires authentication, serves HTTPS only and uses transport TLS.
#
#   ES_IMAGE=elasticsearch-rs:dev cluster/deploy/docker/compose-e2e.sh
#
# Uses a throwaway compose project (`es-compose-e2e`) and host port
# ES_PORT (default 29200); everything it creates is removed on exit
# (KEEP=1 keeps it). Requires docker compose, curl, jq, openssl.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ES_IMAGE="${ES_IMAGE:-elasticsearch-rs:dev}" ES_PORT="${ES_PORT:-29200}"
PROJECT=es-compose-e2e
WORK="$(mktemp -d)"
dc() { docker compose -p "$PROJECT" -f "$DIR/docker-compose.yml" "$@"; }
ok() { printf '    \033[32mOK\033[0m %s\n' "$*"; }
fail() { printf '\033[31mFAIL\033[0m %s\n' "$*" >&2; dc ps -a >&2 || true; dc logs --tail 40 >&2 || true; exit 1; }
cleanup() {
  local rc=$?
  [[ "${KEEP:-0}" == 1 ]] || dc down -v --remove-orphans >/dev/null 2>&1 || true
  rm -rf "$WORK"
  [[ $rc -eq 0 ]] && echo "COMPOSE E2E PASSED" || echo "COMPOSE E2E FAILED"
}
trap cleanup EXIT

# Container names are fixed in the compose file (es01..es03): refuse to
# clobber a running deployment.
for c in es01 es02 es03; do
  docker inspect "$c" >/dev/null 2>&1 && fail "a container named $c already exists"
done

dc up -d --no-build
deadline=$((SECONDS + 300))
until [[ "$(docker inspect -f '{{.State.Health.Status}}' es01 2>/dev/null)" == healthy && "$(docker inspect -f '{{.State.Health.Status}}' es03 2>/dev/null)" == healthy ]]; do
  (( SECONDS < deadline )) || fail "nodes not healthy after 300s"
  sleep 3
done
ok "setup completed, es01..es03 healthy (verified-TLS, authenticated healthchecks)"
[[ "$(docker inspect -f '{{.State.ExitCode}}' "$(dc ps -aq setup)")" == 0 ]] || fail "setup exited non-zero"

dc cp es01:/usr/share/elasticsearch/config/certs/ca/ca.crt "$WORK/ca.crt" >/dev/null
PW="$(dc exec -T es01 cat config/certs/elastic-password)"
[[ ${#PW} -ge 20 ]] || fail "generated password too short"
URL="https://localhost:${ES_PORT}"
es() { curl -sS --cacert "$WORK/ca.crt" -K <(printf 'user = "elastic:%s"\n' "$PW") "$@"; }

health="$(es "$URL/_cluster/health?wait_for_status=green&wait_for_nodes=3&timeout=60s")"
[[ "$(jq -r .status <<<"$health")/$(jq -r .number_of_nodes <<<"$health")" == green/3 ]] || fail "health: $health"
ok "green, 3 nodes, over HTTPS verified with the setup CA"
[[ "$(es "$URL/_security/_authenticate" | jq -r .username)" == elastic ]] || fail "authenticate"
ok "authenticated as elastic with the generated password"
code="$(curl -s -o /dev/null -w '%{http_code}' --cacert "$WORK/ca.crt" "$URL/")"
[[ "$code" == 401 ]] || fail "anonymous -> $code"
ok "anonymous request -> 401"
code="$(curl -s -o /dev/null -w '%{http_code}' --cacert "$WORK/ca.crt" -u elastic:wrong "$URL/")"
[[ "$code" == 401 ]] || fail "wrong password -> $code"
ok "wrong password -> 401"
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://localhost:${ES_PORT}/" || true)"
[[ "$code" != 200 ]] || fail "plain HTTP answered 200"
ok "plain HTTP refused (curl status $code)"
if curl -s -o /dev/null --max-time 5 "$URL/"; then fail "TLS accepted without the CA"; fi
ok "HTTPS without the CA fails verification"
tls="$(dc exec -T es01 bash -c 'openssl s_client -connect es02:9300 -CAfile config/certs/ca/ca.crt </dev/null 2>&1 | grep "Verify return code"')"
grep -q 'Verify return code: 0 (ok)' <<<"$tls" || fail "transport es02:9300: $tls"
ok "transport es02:9300 speaks TLS with a certificate from the setup CA"
es -XPUT "$URL/e2e" -H 'Content-Type: application/json' -d '{"settings":{"number_of_shards":3,"number_of_replicas":2}}' >/dev/null
es -XPOST "$URL/e2e/_bulk?refresh=true" -H 'Content-Type: application/x-ndjson' --data-binary $'{"index":{"_id":"1"}}\n{"a":1}\n{"index":{"_id":"2"}}\n{"a":2}\n' >/dev/null
[[ "$(es "$URL/e2e/_count" | jq -r .count)" == 2 ]] || fail "count"
ok "index + search across the secured cluster"
