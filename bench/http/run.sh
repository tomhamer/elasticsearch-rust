#!/usr/bin/env bash
# Reproduce bench/results/HTTP_VS_ES.md: this server vs real Elasticsearch
# over HTTP on the bench corpus (Linux; run from the repo root).
#
#   bench/http/run.sh                 # everything (~25 min on 16 vCPU)
#   HB_PHASES=load,check,search HB_ONLY=match,knn bench/http/run.sh
#
# Servers share CPUs $SERVER_CPUS (default 0-7), the load generator runs on
# $CLIENT_CPUS (default 8-11). Knobs of the harness (HB_*) are documented in
# crates/elasticsearch-http/benches/http_compare.rs. Output: $OUT_DIR
# (default bench/results/http) gets the JSON lines and the Markdown summary.
set -euo pipefail
cd "$(dirname "$0")/../.."
HERE=bench/http
OUT_DIR="${OUT_DIR:-bench/results/http}"
CLIENT_CPUS="${CLIENT_CPUS:-8-11}"
export ES_PORT="${ES_PORT:-9211}" RUSTIC_PORT="${RUSTIC_PORT:-9212}"
mkdir -p "$OUT_DIR"

cargo build --release --bin elasticsearch
cargo bench -p elasticsearch-http --bench http_compare --no-run 2>&1 | tail -1
TARGET_DIR="${CARGO_TARGET_DIR:-target}"
HARNESS=$(ls -t "$TARGET_DIR"/release/deps/http_compare-* | grep -v '\.d$' | head -1)

trap '$HERE/servers.sh stop-es; $HERE/servers.sh stop-rustic' EXIT
$HERE/servers.sh start-es
$HERE/servers.sh start-rustic "$TARGET_DIR/release/elasticsearch"

stamp=$(date +%Y%m%d-%H%M%S)
HB_TARGETS="rustic=http://127.0.0.1:$RUSTIC_PORT,es=http://127.0.0.1:$ES_PORT" \
HB_CORPUS="${HB_CORPUS:-bench/corpus}" HB_OUT="$OUT_DIR/summary-$stamp.md" \
  taskset -c "$CLIENT_CPUS" "$HARNESS" | tee "$OUT_DIR/run-$stamp.jsonl"
echo "summary: $OUT_DIR/summary-$stamp.md"
