#!/usr/bin/env bash
# Run the full benchmark (rustic 4-seg + 1-seg, tantivy 1-seg + 4-seg, Lucene)
# sequentially and regenerate results/RESULTS.md.  Assumes corpus/ is generated,
# `cargo build --release -p rustic-bench-compare` is done, and (for Lucene)
# JAVA_HOME points at a JDK 21 and lucene/lib holds the jars (see README).
#
# `run_all.sh --vectors-only` re-runs just the vector benchmarks (rustic,
# Lucene, Qdrant, hnswlib; unfiltered + filtered) back-to-back and
# regenerates RESULTS.md from all result files.  Qdrant / hnswlib need the
# venv in tools/venv (numpy, hnswlib, qdrant-client) and the qdrant binary in
# tools/qdrant-*/; each step is skipped with a message when missing.
#
# `run_all.sh --hybrid-only` re-runs just the hybrid (BM25 + k-NN) benchmark:
# rustic, Lucene, Qdrant interleaved for ROUNDS (default 2) rounds, best of
# rounds kept, then regenerates RESULTS.md.
set -euo pipefail
cd "$(dirname "$0")"
C=${COMPARE:-../target/release/compare}
Q=corpus/queries.jsonl
R=results
VEC=corpus/vectors.f32
VEC100=corpus/vectors-q100.f32
VECTORS_ONLY=0
HYBRID_ONLY=0
[[ "${1:-}" == "--vectors-only" ]] && VECTORS_ONLY=1
[[ "${1:-}" == "--hybrid-only" ]] && HYBRID_ONLY=1

# Hybrid search (BM25 + k-NN): rustic, Lucene, Qdrant back-to-back, ROUNDS
# rounds (default 2); each system's better-of-rounds result (highest QPS per
# mode/filter) is kept in <results>/<system>-hybrid.json by `compare
# hybrid-merge`. Needs corpus/vectors-q100.f32 (vectors.py --queries 100).
# Env: HYBRID_RESULTS (default results/), HYBRID_INDEXES (default indexes/),
# ROUNDS, BENCH_MACHINE (host label recorded in the JSON), COMPARE (binary).
run_hybrid() {
  local B HR HI
  B=$(pwd)
  mkdir -p "${HYBRID_RESULTS:-$R}" "${HYBRID_INDEXES:-indexes}"
  HR=$(cd "${HYBRID_RESULTS:-$R}" && pwd)
  HI=$(cd "${HYBRID_INDEXES:-indexes}" && pwd)
  if [[ ! -f $VEC100 ]]; then
    python3 corpus/vectors.py --n 200000 --dim 128 --queries 100 --seed 42 --out $VEC100
  fi
  if [[ -z "${JAVA_HOME:-}" ]] && ls -d tools/jdk-21*/Contents/Home >/dev/null 2>&1; then
    export JAVA_HOME=$(cd "$(ls -d tools/jdk-21*/Contents/Home | head -1)" && pwd)
  fi
  PY=$B/tools/venv/bin/python
  QBIN=$(ls -d "$B"/tools/qdrant-*/qdrant 2>/dev/null | tail -1 || true)
  rm -f "$HR"/*-hybrid.json
  $C hybrid --system tantivy --results "$HR"
  $C hybrid --system hnswlib --results "$HR"
  local first=${ROUND_START:-1}
  for round in $(seq "$first" $((first + ${ROUNDS:-2} - 1))); do
    D=$HR/hybrid-rounds/r$round
    mkdir -p "$D"
    $C hybrid --system rustic --corpus corpus/corpus.jsonl --queries $Q --vectors $VEC100 --vec-queries 100 --results "$D"
    if [[ -d lucene/lib ]] && { [[ -n "${JAVA_HOME:-}" ]] || command -v java >/dev/null; }; then
      lucene/run.sh hybrid --corpus "$B/corpus/corpus.jsonl" --queries "$B/$Q" --vectors "$B/$VEC100" \
        --vec-queries 100 --out "$HI/lucene-hybrid" --reuse --results "$D"
    else
      echo "[run_all] skipping Lucene hybrid (no java / lucene/lib)" >&2
    fi
    if [[ -x $PY && -n "$QBIN" ]] && $PY -c 'import qdrant_client, httpx, numpy' 2>/dev/null; then
      (cd qdrant && $PY hybrid.py --corpus "$B/corpus/corpus.jsonl" --queries "$B/$Q" --vectors "$B/$VEC100" \
        --vec-queries 100 --qdrant-bin "$QBIN" --storage "$HI/qdrant-hybrid" --reuse --results "$D")
    else
      echo "[run_all] skipping Qdrant hybrid (no venv / qdrant binary)" >&2
    fi
  done
  $C hybrid-merge --inputs "$HR"/hybrid-rounds/r*/*-hybrid.json --results "$HR"
}

run_vectors() {
  rm -f $R/*vectors*.json
  $C vectors --system rustic  --vectors $VEC --dim 128 --queries 20 --threads 1 --results $R
  $C vectors --system rustic  --vectors $VEC --dim 128 --queries 20 --threads 0 --label rustic-mt --results $R
  $C vectors --system tantivy --results $R
  $C vectors-filtered --vectors $VEC --dim 128 --queries 20 --ef 100 --results $R

  if [[ -z "${JAVA_HOME:-}" ]] && ls -d tools/jdk-21*/Contents/Home >/dev/null 2>&1; then
    export JAVA_HOME=$(cd "$(ls -d tools/jdk-21*/Contents/Home | head -1)" && pwd)
  fi
  if [[ -n "${JAVA_HOME:-}" && -d lucene/lib ]]; then
    lucene/run.sh vectors          --vectors ../corpus/vectors.f32 --out ../indexes/lucene-vec --dim 128 --queries 20 --results ../results
    lucene/run.sh vectors-filtered --vectors ../corpus/vectors.f32 --out ../indexes/lucene-vec-filtered --dim 128 --queries 20 --ef 100 --results ../results
  else
    echo "[run_all] skipping Lucene vectors (no JAVA_HOME / lucene/lib)" >&2
  fi

  PY=tools/venv/bin/python
  if [[ -x $PY ]] && $PY -c 'import qdrant_client, httpx, numpy' 2>/dev/null; then
    QBIN=$(ls -d tools/qdrant-*/qdrant 2>/dev/null | tail -1 || true)
    if [[ -n "$QBIN" ]]; then
      (cd qdrant && ../$PY bench.py --vectors ../$VEC --qdrant-bin ../$QBIN --storage ../indexes/qdrant --results ../$R)
    else
      echo "[run_all] skipping Qdrant (no tools/qdrant-*/qdrant binary)" >&2
    fi
  else
    echo "[run_all] skipping Qdrant (no tools/venv with qdrant-client)" >&2
  fi
  if [[ -x $PY ]] && $PY -c 'import hnswlib, numpy' 2>/dev/null; then
    (cd hnswlib && ../$PY bench.py --vectors ../$VEC --results ../$R)
  else
    echo "[run_all] skipping hnswlib (no tools/venv with hnswlib)" >&2
  fi
}

if [[ $HYBRID_ONLY == 1 ]]; then
  run_hybrid
  if [[ -n "${HYBRID_RESULTS:-}" ]]; then
    $C report --inputs $R/*.json "$HYBRID_RESULTS"/*-hybrid.json --out "$HYBRID_RESULTS/RESULTS.md"   # later inputs win
  else
    $C report --inputs $R/*.json --out $R/RESULTS.md
  fi
  exit 0
fi

if [[ $VECTORS_ONLY == 1 ]]; then
  run_vectors
  $C report --inputs $R/*.json --out $R/RESULTS.md
  exit 0
fi

rm -f $R/*.json
$C index --system rustic  --corpus corpus/corpus.jsonl --out indexes/rustic      --refresh-every 50000 --results $R
$C index --system rustic  --corpus corpus/corpus.jsonl --out indexes/rustic-1seg --refresh-every 50000 --force-merge --label rustic-1seg --results $R
$C index --system tantivy --corpus corpus/corpus.jsonl --out indexes/tantivy     --results $R
$C index --system tantivy --corpus corpus/corpus.jsonl --out indexes/tantivy-4seg --tantivy-commit-every 50000 --label tantivy-4seg --results $R

for spec in "rustic indexes/rustic rustic" "rustic indexes/rustic-1seg rustic-1seg" \
            "tantivy indexes/tantivy tantivy" "tantivy indexes/tantivy-4seg tantivy-4seg"; do
  set -- $spec
  $C query   --system $1 --index $2 --queries $Q --warmup 2 --iters 5 --label $3 --results $R
  $C startup --system $1 --index $2 --queries $Q --runs 5 --label $3 --results $R
done


if [[ -n "${JAVA_HOME:-}" && -d lucene/lib ]]; then
  lucene/run.sh index   --corpus ../corpus/corpus.jsonl --out ../indexes/lucene --results ../results
  lucene/run.sh query   --index ../indexes/lucene --queries ../corpus/queries.jsonl --warmup 2 --iters 5 --results ../results
  lucene/run.sh startup --index ../indexes/lucene --queries ../corpus/queries.jsonl --runs 5 --results ../results
fi

run_vectors
run_hybrid

$C report --inputs $R/*.json --out $R/RESULTS.md
