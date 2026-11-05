#!/usr/bin/env bash
# Fetch Elasticsearch's YAML REST test suite + API JSON specs for a pinned tag.
# Keeps only:
#   rest-api-spec/src/yamlRestTest/resources/rest-api-spec/test/**  -> tests/es-yaml/test/
#   rest-api-spec/src/main/resources/rest-api-spec/api/*.json       -> tests/es-yaml/api/
# Usage: tests/es-yaml/fetch.sh [tag] [dest]
#   tag:  default the pinned tag below; `v8.19.22` fetches the 8.x suite that
#         `es-yaml-runner --compat 8` runs (ES's yamlRestCompatTest).
#   dest: default tests/es-yaml (e.g. tests/es-yaml/v8 to keep both side by side;
#         run it with `es-yaml-runner --suite tests/es-yaml/v8`).
set -euo pipefail

ES_TAG="${1:-v9.5.5}"   # latest 9.x GA at the time of pinning (2026-10-08)
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -n "${2:-}" ]; then mkdir -p "$2"; HERE="$(cd "$2" && pwd)"; fi
VER="${ES_TAG#v}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "fetching elasticsearch ${ES_TAG} ..."
curl -fsSL "https://github.com/elastic/elasticsearch/archive/refs/tags/${ES_TAG}.tar.gz" -o "$TMP/es.tar.gz"
PREFIX="elasticsearch-${VER}/rest-api-spec/src"
tar -xzf "$TMP/es.tar.gz" -C "$TMP" \
  "${PREFIX}/yamlRestTest/resources/rest-api-spec/test" \
  "${PREFIX}/main/resources/rest-api-spec/api"

rm -rf "$HERE/test" "$HERE/api"
mv "$TMP/${PREFIX}/yamlRestTest/resources/rest-api-spec/test" "$HERE/test"
mv "$TMP/${PREFIX}/main/resources/rest-api-spec/api" "$HERE/api"
# Module / x-pack suites that live outside rest-api-spec (ILM, SLM, data
# streams + their lifecycle). Copied as test/<dir>.
# ILM's suite moved from ilm/qa/rest (8.x) to the plugin itself (9.x).
case "$VER" in
  8.*) ILM="x-pack/plugin/ilm/qa/rest/src/yamlRestTest/resources/rest-api-spec/test/ilm" ;;
  *) ILM="x-pack/plugin/ilm/src/yamlRestTest/resources/rest-api-spec/test/ilm" ;;
esac
for d in \
  "$ILM" \
  "x-pack/plugin/slm/src/yamlRestTest/resources/rest-api-spec/test/slm" \
  "modules/data-streams/src/yamlRestTest/resources/rest-api-spec/test/data_stream" \
  "x-pack/plugin/src/yamlRestTest/resources/rest-api-spec/test/dlm" \
  "x-pack/plugin/src/yamlRestTest/resources/rest-api-spec/test/data_streams"; do
  tar -xzf "$TMP/es.tar.gz" -C "$TMP" "elasticsearch-${VER}/$d"
  rm -rf "$HERE/test/$(basename "$d")"
  mv "$TMP/elasticsearch-${VER}/$d" "$HERE/test/$(basename "$d")"
done
# x-pack security suites (run with `es-yaml-runner --security`)
XP="elasticsearch-${VER}/x-pack/plugin/src/yamlRestTest/resources/rest-api-spec/test"
XDIRS="api_key authenticate change_password privileges role_mapping roles security service_accounts token users set_security_user user_profile"
tar -xzf "$TMP/es.tar.gz" -C "$TMP" $(for d in $XDIRS; do echo "$XP/$d"; done)
rm -rf "$HERE/xpack-test" && mkdir -p "$HERE/xpack-test"
for d in $XDIRS; do mv "$TMP/$XP/$d" "$HERE/xpack-test/"; done
echo "$ES_TAG" > "$HERE/VERSION"
echo "done: $(find "$HERE/test" -name '*.yml' | wc -l | tr -d ' ') test files, $(ls "$HERE/api" | wc -l | tr -d ' ') api specs"
