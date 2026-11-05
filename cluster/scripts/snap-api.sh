#!/usr/bin/env bash
# Snapshot REST API conformance probes against a running 3-node cluster
# (scripts/snap-cluster.sh start). Each check asserts the HTTP status and a
# substring of the body (error type or response field), rotating over nodes
# so every request also exercises non-master forwarding.
set -uo pipefail
PORT_BASE="${PORT_BASE:-9500}"
REPO_DIR="${REPO_DIR:-/tmp/snap-repo-${TAG:-snap}}"
pass=0; fail=0; n=0
H='Content-Type: application/json'

chk() { # name status substr method path [body]
  local name="$1" want="$2" sub="$3" m="$4" path="$5" body="${6:-}"
  local port=$((PORT_BASE + n % 3)); n=$((n + 1))
  local out code
  if [[ -n "$body" ]]; then
    out=$(curl -s -w '\n%{http_code}' -X "$m" -H "$H" "127.0.0.1:$port$path" -d "$body")
  else
    out=$(curl -s -w '\n%{http_code}' -X "$m" "127.0.0.1:$port$path")
  fi
  code="${out##*$'\n'}"; out="${out%$'\n'*}"
  if [[ "$code" == "$want" && "$out" == *"$sub"* ]]; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1)); echo "FAIL [$name] $m $path -> $code (want $want, '$sub'): ${out:0:300}"
  fi
}
q() { curl -s -X "$1" -H "$H" "127.0.0.1:$PORT_BASE$2" ${3:+-d "$3"}; }

q DELETE '/api_*,restored_*,ren_*?expand_wildcards=all' >/dev/null
q DELETE '/_snapshot/api_*' >/dev/null

# ---- repositories -------------------------------------------------------
chk repo-missing 404 repository_missing_exception GET /_snapshot/api_nope
chk repo-put 200 '"acknowledged":true' PUT /_snapshot/api_r "{\"type\":\"fs\",\"settings\":{\"location\":\"$REPO_DIR/api_r\"}}"
chk repo-put-post 200 '"acknowledged":true' POST /_snapshot/api_r2 "{\"type\":\"fs\",\"settings\":{\"location\":\"$REPO_DIR/api_r2\",\"compress\":true}}"
chk repo-get 200 '"api_r":{"type":"fs"' GET /_snapshot/api_r
chk repo-get-all 200 '"api_r2"' GET /_snapshot
chk repo-get-_all 200 '"api_r2"' GET /_snapshot/_all
chk repo-get-wild 200 '"api_r2"' GET '/_snapshot/api_*'
chk repo-get-list 200 '"api_r2"' GET '/_snapshot/api_r,api_r2'
chk repo-no-type 400 action_request_validation_exception PUT /_snapshot/api_bad '{"settings":{"location":"x"}}'
chk repo-bad-type 500 repository_exception PUT /_snapshot/api_bad '{"type":"nosuch","settings":{}}'
chk repo-outside-path 500 repository_exception PUT /_snapshot/api_bad '{"type":"fs","settings":{"location":"/tmp/esn-not-allowed"}}'
chk repo-no-location 500 repository_exception PUT /_snapshot/api_bad '{"type":"fs","settings":{}}'
chk repo-verify 200 '"nodes":{' POST /_snapshot/api_r/_verify
chk repo-verify-missing 404 repository_missing_exception POST /_snapshot/api_nope/_verify
chk repo-delete-missing 404 repository_missing_exception DELETE /_snapshot/api_nope
chk cat-repos 200 'api_r ' GET /_cat/repositories

# ---- create --------------------------------------------------------------
q PUT /api_a '{"settings":{"number_of_shards":1,"number_of_replicas":1,"index.refresh_interval":"7s"}}' >/dev/null
q PUT /api_b '{"settings":{"number_of_shards":1,"number_of_replicas":0}}' >/dev/null
for i in $(seq 1 20); do q PUT "/api_a/_doc/$i" "{\"n\":$i}" >/dev/null; q PUT "/api_b/_doc/$i" "{\"n\":$i}" >/dev/null; done
q POST /api_a,api_b/_refresh >/dev/null

chk snap-create-wait 200 '"state":"SUCCESS"' PUT '/_snapshot/api_r/s1?wait_for_completion=true' '{"indices":"api_*","metadata":{"who":"me"}}'
chk snap-create-async 200 '"accepted":true' PUT /_snapshot/api_r/s2 '{"indices":"api_a"}'
for _ in $(seq 1 50); do [[ "$(q GET /_snapshot/api_r/s2)" == *'"state":"SUCCESS"'* ]] && break; sleep 0.2; done
chk snap-create-post 200 '"state":"SUCCESS"' POST '/_snapshot/api_r/s3?wait_for_completion=true' '{"indices":"api_b","include_global_state":false}'
chk snap-dup 400 invalid_snapshot_name_exception PUT '/_snapshot/api_r/s1?wait_for_completion=true'
chk snap-upper 400 invalid_snapshot_name_exception PUT '/_snapshot/api_r/BadName?wait_for_completion=true'
chk snap-underscore 400 invalid_snapshot_name_exception PUT '/_snapshot/api_r/_bad?wait_for_completion=true'
chk snap-hash 400 invalid_snapshot_name_exception PUT '/_snapshot/api_r/a%23b?wait_for_completion=true'
chk snap-norepo 404 repository_missing_exception PUT '/_snapshot/api_nope/s1?wait_for_completion=true'
chk snap-noindex 404 index_not_found_exception PUT '/_snapshot/api_r/s9?wait_for_completion=true' '{"indices":"api_missing"}'
chk snap-noindex-ignore 200 '"state":"SUCCESS"' PUT '/_snapshot/api_r/s4?wait_for_completion=true' '{"indices":"api_missing,api_a","ignore_unavailable":true}'

# ---- get -----------------------------------------------------------------
chk get-one 200 '"snapshot":"s1"' GET /_snapshot/api_r/s1
chk get-meta 200 '"metadata":{"who":"me"}' GET /_snapshot/api_r/s1
chk get-all 200 '"total":4' GET /_snapshot/api_r/_all
chk get-wild 200 '"total":4' GET '/_snapshot/api_r/s*'
chk get-list 200 '"total":2' GET '/_snapshot/api_r/s1,s3'
chk get-exclude 200 '"total":3' GET '/_snapshot/api_r/*,-s1'
chk get-missing 404 snapshot_missing_exception GET /_snapshot/api_r/nope
chk get-missing-ignore 200 '"snapshots":[]' GET '/_snapshot/api_r/nope?ignore_unavailable=true'
chk get-missing-repo 404 repository_missing_exception GET /_snapshot/api_nope/_all
chk get-nonverbose 200 '"state":"SUCCESS"' GET '/_snapshot/api_r/s1?verbose=false'
chk get-size 200 '"remaining":3' GET '/_snapshot/api_r/_all?size=1'
chk get-sort-desc 200 '"snapshots":[{"snapshot":"s4"' GET '/_snapshot/api_r/_all?sort=name&order=desc'
chk get-multi-repo 200 '"repository":"api_r"' GET '/_snapshot/api_*/_all'
chk get-current 200 '"snapshots":[]' GET /_snapshot/api_r/_current
chk get-index-details 200 '"index_details"' GET '/_snapshot/api_r/s1?index_details=true'
chk cat-snaps 200 's1 ' GET /_cat/snapshots/api_r
chk cat-snaps-missing 404 repository_missing_exception GET /_cat/snapshots/api_nope

# ---- status --------------------------------------------------------------
chk status-one 200 '"shards_stats"' GET /_snapshot/api_r/s1/_status
chk status-list 200 '"snapshot":"s3"' GET /_snapshot/api_r/s1,s3/_status
chk status-repo 200 '"snapshots":[]' GET /_snapshot/api_r/_status
chk status-all 200 '"snapshots":[]' GET /_snapshot/_status
chk status-missing 404 snapshot_missing_exception GET /_snapshot/api_r/nope/_status
chk status-missing-ignore 200 '"snapshots":[]' GET '/_snapshot/api_r/nope/_status?ignore_unavailable=true'
chk status-missing-repo 404 repository_missing_exception GET /_snapshot/api_nope/s1/_status

# ---- restore -------------------------------------------------------------
chk restore-open 500 snapshot_restore_exception POST '/_snapshot/api_r/s1/_restore?wait_for_completion=true'
chk restore-missing 404 snapshot_missing_exception POST '/_snapshot/api_r/nope/_restore?wait_for_completion=true'
chk restore-missing-repo 404 repository_missing_exception POST '/_snapshot/api_nope/s1/_restore?wait_for_completion=true'
chk restore-noindex 404 index_not_found_exception POST '/_snapshot/api_r/s1/_restore?wait_for_completion=true' '{"indices":"zzz"}'
chk restore-rename 200 '"indices":["restored_api_a"]' POST '/_snapshot/api_r/s1/_restore?wait_for_completion=true' '{"indices":"api_a","rename_pattern":"(.+)","rename_replacement":"restored_$1"}'
chk restore-rename-count 200 '"count":20' GET /restored_api_a/_count
chk restore-rename-collide 500 snapshot_restore_exception POST '/_snapshot/api_r/s1/_restore?wait_for_completion=true' '{"indices":"api_a","rename_pattern":"(.+)","rename_replacement":"restored_$1"}'
chk restore-same-target 500 snapshot_restore_exception POST '/_snapshot/api_r/s1/_restore?wait_for_completion=true' '{"indices":"api_*","rename_pattern":"api_.","rename_replacement":"ren_same"}'
chk restore-settings 200 '"shards":{' POST '/_snapshot/api_r/s1/_restore?wait_for_completion=true' '{"indices":"api_a","rename_pattern":"api_","rename_replacement":"ren_","index_settings":{"index.number_of_replicas":0},"ignore_index_settings":["index.refresh_interval"]}'
chk restore-settings-replicas 200 '"number_of_replicas":"0"' GET /ren_a/_settings
s=$(q GET /ren_a/_settings); [[ "$s" == *refresh_interval* ]] && { fail=$((fail + 1)); echo "FAIL [ignore_index_settings] refresh_interval kept: ${s:0:300}"; } || pass=$((pass + 1))
chk restore-bad-setting 500 snapshot_restore_exception POST '/_snapshot/api_r/s1/_restore?wait_for_completion=true' '{"indices":"api_a","rename_pattern":"api_","rename_replacement":"ren_x_","index_settings":{"index.number_of_shards":2}}'
chk restore-bad-ignore 500 snapshot_restore_exception POST '/_snapshot/api_r/s1/_restore?wait_for_completion=true' '{"indices":"api_a","rename_pattern":"api_","rename_replacement":"ren_y_","ignore_index_settings":["index.number_of_shards"]}'
chk restore-unknown-param 400 illegal_argument_exception POST '/_snapshot/api_r/s1/_restore?wait_for_completion=true' '{"indices":"api_a","bogus":1}'
q POST /api_b/_close >/dev/null
chk restore-closed 200 '"indices":["api_b"]' POST '/_snapshot/api_r/s1/_restore?wait_for_completion=true' '{"indices":"api_b","include_global_state":false,"partial":true}'
chk restore-closed-count 200 '"count":20' GET /api_b/_count
q DELETE /ren_a >/dev/null
chk restore-async 200 '"accepted":true' POST /_snapshot/api_r/s1/_restore '{"indices":"api_a","rename_pattern":"api_","rename_replacement":"ren_"}'
for _ in $(seq 1 50); do [[ "$(q GET /ren_a/_count)" == *'"count":20'* ]] && break; sleep 0.2; done
chk restore-async-count 200 '"count":20' GET /ren_a/_count
chk restore-global 200 '"snapshot":{' POST '/_snapshot/api_r/s1/_restore?wait_for_completion=true' '{"indices":"-*","include_global_state":true}'
chk restore-global-on-noglobal 500 snapshot_restore_exception POST '/_snapshot/api_r/s3/_restore?wait_for_completion=true' '{"indices":"-*","include_global_state":true}'

# ---- clone ---------------------------------------------------------------
chk clone 200 '"acknowledged":true' PUT /_snapshot/api_r/s1/_clone/c1 '{"indices":"api_a"}'
chk clone-get 200 '"indices":["api_a"]' GET /_snapshot/api_r/c1
chk clone-exists 400 invalid_snapshot_name_exception PUT /_snapshot/api_r/s1/_clone/c1 '{"indices":"*"}'
chk clone-missing-src 404 snapshot_missing_exception PUT /_snapshot/api_r/nope/_clone/c2 '{"indices":"*"}'
chk clone-no-indices 400 action_request_validation_exception PUT /_snapshot/api_r/s1/_clone/c2 '{}'
chk clone-no-match 400 'No indices in the source snapshot [s1/' PUT /_snapshot/api_r/s1/_clone/c2 '{"indices":"zzz"}'
chk clone-bad-name 400 invalid_snapshot_name_exception PUT /_snapshot/api_r/s1/_clone/BAD '{"indices":"*"}'
chk clone-restore 200 '"indices":["ren_c_a"]' POST '/_snapshot/api_r/c1/_restore?wait_for_completion=true' '{"rename_pattern":"api_","rename_replacement":"ren_c_"}'
chk clone-restore-count 200 '"count":20' GET /ren_c_a/_count

# ---- delete / cleanup ----------------------------------------------------
chk delete-missing 404 snapshot_missing_exception DELETE /_snapshot/api_r/nope
chk delete-one 200 '"acknowledged":true' DELETE /_snapshot/api_r/s2
chk delete-gone 404 snapshot_missing_exception GET /_snapshot/api_r/s2
chk delete-wild 200 '"acknowledged":true' DELETE '/_snapshot/api_r/s3,s4*'
chk delete-left 200 '"total":2' GET /_snapshot/api_r/_all
mkdir -p "$REPO_DIR/api_r/indices/stale-dir/0" && echo junk >"$REPO_DIR/api_r/indices/stale-dir/0/__junk" && echo junk >"$REPO_DIR/api_r/stale-root-blob"
chk cleanup 200 '"deleted_blobs"' POST /_snapshot/api_r/_cleanup
[[ -e "$REPO_DIR/api_r/indices/stale-dir" ]] && { fail=$((fail + 1)); echo "FAIL [cleanup] stale index dir not removed"; } || pass=$((pass + 1))
chk cleanup-missing 404 repository_missing_exception POST /_snapshot/api_nope/_cleanup
chk after-cleanup-restore 200 '"indices":["ren_d_a"]' POST '/_snapshot/api_r/c1/_restore?wait_for_completion=true' '{"rename_pattern":"api_","rename_replacement":"ren_d_"}'
chk repo-update-in-use 200 '"acknowledged":true' PUT /_snapshot/api_r "{\"type\":\"fs\",\"settings\":{\"location\":\"$REPO_DIR/api_r\",\"max_snapshot_bytes_per_sec\":\"100mb\"}}"
chk repo-delete 200 '"acknowledged":true' DELETE /_snapshot/api_r2
chk repo-delete-wild 200 '"acknowledged":true' DELETE '/_snapshot/api_*'
chk repo-gone 404 repository_missing_exception GET /_snapshot/api_r
chk bad-method 405 '' PATCH /_snapshot/api_r

q DELETE '/api_*,restored_*,ren_*?expand_wildcards=all' >/dev/null
echo "snap-api: pass $pass fail $fail"
[[ "$fail" -eq 0 ]]
