#!/usr/bin/env bash
# Assemble the cross-cluster YAML corpus `ccs_ccr_yaml.py` runs, from an
# Elasticsearch v9.5.5 checkout (a sparse one with these paths is enough):
#   rest-api-spec/src/main/resources/rest-api-spec/api
#   rest-api-spec/src/yamlRestTest/resources/rest-api-spec/test/{cluster.stats,cluster.remote_info}
#   x-pack/plugin/ccr/src/yamlRestTest/resources/rest-api-spec/test/ccr
#   x-pack/qa/multi-cluster-search-security/src/yamlRestTest/resources/rest-api-spec/test
# Output: <out>/{mcss,plug,oss}/{api,test,VERSION} and <out>/SHA256SUMS.
set -euo pipefail
src=${1:?ES v9.5.5 checkout}
out=${2:?output directory}
want=48acba663b09e70b115dd879329141506c377890
have=$(git -C "$src" rev-parse HEAD)
[ "$have" = "$want" ] || { echo "checkout is $have, not v9.5.5 ($want)" >&2; exit 1; }
[ ! -e "$out" ] || { echo "refusing to reuse $out" >&2; exit 1; }
api="$src/rest-api-spec/src/main/resources/rest-api-spec/api"
for s in mcss plug oss; do
  mkdir -p "$out/$s/test"
  cp -R "$api" "$out/$s/api"
  echo "v9.5.5" > "$out/$s/VERSION"
done
cp -R "$src/x-pack/qa/multi-cluster-search-security/src/yamlRestTest/resources/rest-api-spec/test/basic_license" "$out/mcss/test/"
cp -R "$src/x-pack/qa/multi-cluster-search-security/src/yamlRestTest/resources/rest-api-spec/test/full_license" "$out/mcss/test/"
cp -R "$src/x-pack/plugin/ccr/src/yamlRestTest/resources/rest-api-spec/test/ccr" "$out/plug/test/"
mkdir -p "$out/oss/test/cluster.stats"
cp "$src/rest-api-spec/src/yamlRestTest/resources/rest-api-spec/test/cluster.stats/30_ccs_stats.yml" "$out/oss/test/cluster.stats/"
cp -R "$src/rest-api-spec/src/yamlRestTest/resources/rest-api-spec/test/cluster.remote_info" "$out/oss/test/"
(cd "$out" && find . -type f -name '*.yml' | sort | xargs sha256sum > SHA256SUMS)
echo "assembled $(wc -l < "$out/SHA256SUMS") test files into $out"
