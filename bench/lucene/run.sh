#!/usr/bin/env bash
# Build and run the Lucene harness without Gradle.
#
#   ./run.sh index   --corpus ../corpus/corpus.jsonl --out ../indexes/lucene --results ../results
#   ./run.sh startup --index ../indexes/lucene --queries ../corpus/queries.jsonl --runs 5 --results ../results
#   ./run.sh query   --index ../indexes/lucene --queries ../corpus/queries.jsonl --warmup 2 --iters 5 --results ../results
#   ./run.sh vectors --vectors ../corpus/vectors.f32 --out ../indexes/lucene-vec --dim 128 --results ../results
#   ./run.sh hybrid  --corpus ../corpus/corpus.jsonl --queries ../corpus/queries.jsonl --vectors ../corpus/vectors-q100.f32 --vec-queries 100 --out ../indexes/lucene-hybrid [--reuse] --results ../results
#
# Requires: a JDK 21 (set JAVA_HOME, or put `java`/`javac` on PATH) and the
# Lucene + Gson jars in ./lib (see README section 4 for the curl commands).
# Compiles into ./out when sources are newer than the classes.
set -euo pipefail
cd "$(dirname "$0")"

if [[ -n "${JAVA_HOME:-}" ]]; then
  JAVA="$JAVA_HOME/bin/java"; JAVAC="$JAVA_HOME/bin/javac"
else
  JAVA=java; JAVAC=javac
fi
BENCH_CLASSPATH='lib/*:out'
if [[ -n "${ES_NATIVE_HOME:-}" ]]; then
  BENCH_CLASSPATH="$BENCH_CLASSPATH:$ES_NATIVE_HOME/lib/*"
fi

if [[ ! -d out ]] || [[ -n "$(find src -newer out -name '*.java' 2>/dev/null)" ]]; then
  echo "[run.sh] compiling ..." >&2
  rm -rf out && mkdir -p out
  "$JAVAC" --release 21 -Xlint:deprecation -cp 'lib/*' -d out src/main/java/bench/*.java
  touch out
fi

# Same JVM flags as build.gradle.kts: fixed 4 GB heap, parallel GC, Panama
# native access for MMapDirectory and the incubating Vector API (SIMD) that
# Lucene uses for postings decoding and vector distance.
exec "$JAVA" -Xms4g -Xmx4g -XX:+UseParallelGC \
  --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector \
  -Dorg.apache.lucene.vectorization.upperJavaFeatureVersion=27 \
  -cp "$BENCH_CLASSPATH" bench.Main "$@"
