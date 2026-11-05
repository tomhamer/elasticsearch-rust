> - This file documents the in-process library harness (`compare/`, `corpus/`,
>   `lucene/`, `qdrant/`, `hnswlib/`, `run_all.sh`).
> - [path-poc/](path-poc/README.md): Elasticsearch 9.5.5 comparison and
>   correctness tooling. Benchmark results are in the
>   [top-level README](../README.md#performance).

# rustic vs Tantivy vs Lucene (vs Qdrant / hnswlib for vectors) benchmark harness

Head-to-head benchmark of `rustic-core` (our Rust Lucene replacement in
`crates/rustic-core`), [Tantivy 0.22](https://github.com/quickwit-oss/tantivy)
and [Apache Lucene 10](https://lucene.apache.org/) on identical synthetic
corpora and query sets.  It measures:

| metric | subcommand | what is timed |
|---|---|---|
| indexing throughput, index size, segment count | `index` | corpus read + analyze + index + commit, wall clock |
| cold startup | `startup` | fresh process: open index → first term query |
| warm query latency p50 / p99 per query class, exact hit counts | `query` | per-query latency after warm-up |
| vector build time, QPS, recall@10 | `vectors` | HNSW build; search at ef = 50/100/200 vs brute force |
| filtered vector QPS, recall@10 | `vectors-filtered` | k-NN with a payload filter at 1 / 10 / 50 % selectivity, ef = 100 |
| hybrid (BM25 + k-NN) QPS, p50/p99, k-NN recall, cross-system overlap@10 | `hybrid` | score-sum and RRF fusion, unfiltered and with a `price` pre-filter (~10 / 50 %); see section 4d |

Vectors are also run against [Qdrant](https://qdrant.tech) (local server,
`qdrant/bench.py`) and [hnswlib](https://github.com/nmslib/hnswlib)
(Python binding, `hnswlib/bench.py`); see section 4c.

Everything lives under `bench/`; nothing under `crates/` is touched.  The only
edit outside `bench/` is the root `Cargo.toml` (workspace member
`bench/compare` and the `tantivy = "0.22"` workspace dependency).

```
bench/
  corpus/     gen.py, queries.py, vectors.py   (std-lib Python 3 only)
  compare/    Rust binary: rustic + tantivy drivers, startup orchestration, report
  lucene/     Gradle project: the same four subcommands against Lucene 10
  indexes/    generated indexes (not checked in)
  results/    JSON results + RESULTS.md + ISSUES.md
```

## 1. Machine info

Every result JSON embeds the machine description, captured with:

```sh
sysctl -n machdep.cpu.brand_string      # CPU
sysctl -n hw.ncpu hw.physicalcpu        # logical / physical cores
sysctl -n hw.memsize                    # RAM bytes
sw_vers                                 # macOS version
uname -m                                # arch
```

The results in `results/RESULTS.md` were produced on: Apple M1, 8 cores
(8 logical / 8 physical), 8 GB RAM, macOS 13.0 (arm64), rustc 1.87.0,
Python 3.14.  Nothing else of note was running, except where stated.

## 2. Generate the corpora

```sh
cd bench/corpus
python3 gen.py     --docs 200000 --seed 42 --out corpus.jsonl          # ~10 s / 200k docs, 170 MB
python3 queries.py --corpus corpus.jsonl --out queries.jsonl --seed 42  # ~15 s
python3 vectors.py --n 200000 --dim 128 --queries 20 --seed 42 --out vectors.f32
```

`gen.py` is deterministic for a given `(--docs, --seed)` and supports
`--docs 1000000` (the default) if you have the disk (~850 MB JSONL).

Corpus shape (see the docstring in `gen.py` for every knob):

* 200 000-word purely alphabetic vocabulary, Zipf(s = 1.0) term distribution.
  Words are alphabetic only so that all three analyzers tokenize identically.
* Body length lognormal, mean 120 words (σ = 0.5, clamped 5..2000); title 5–10 words.
* 10 % "hot" documents draw 40 % of their body tokens from a 64-term band,
  producing dense postings (df ≈ 5–15 % of the corpus) that a plain Zipf
  corpus would not have.
* `price` lognormal around 50 (2 decimals), `ts` epoch-ms within a 3-year
  window, `tag` one of 50 values (Zipf 0.8).

Query set (`queries.jsonl`, 1000 queries, all on `body` unless noted):

| class | n | construction |
|---|---:|---|
| `term` | 200 | 66 rare (df < 5), 66 medium (0.5–2 % of docs), 68 frequent (5–15 %) |
| `and2` | 200 | medium AND frequent |
| `or2` | 200 | medium OR frequent |
| `and3` | 100 | medium AND frequent AND frequent |
| `phrase` | 100 | adjacent bigrams actually present in the corpus (50 frequent, 50 rare) |
| `range` | 100 | `price` ranges: 50 windows of ~1 % selectivity, 50 of ~10 % (from the exact price quantiles) |
| `tag` | 100 | keyword term queries on `tag` (cycling the 50 tags) |

Vectors: 200 000 base vectors + 20 queries, dim 128, little-endian f32, no
header (`n = filesize / (4·dim) − 20`).  Mildly clustered (1000 Gaussian
centres, σ = 0.6 noise) so recall is meaningful.

## 3. Build and run rustic / tantivy

One command runs everything (both rustic layouts, both tantivy layouts,
Lucene if `JAVA_HOME` and `lucene/lib` are set up) sequentially and writes
`results/RESULTS.md`:

```sh
cargo build --release -p rustic-bench-compare && bench/run_all.sh
```

Individual steps:

```sh
cargo build --release -p rustic-bench-compare        # binary: target/release/compare
cd bench

# indexing (rustic refreshes = one segment file every N docs; tantivy commits once at the end)
../target/release/compare index --system rustic  --corpus corpus/corpus.jsonl --out indexes/rustic  --refresh-every 50000 --results results
../target/release/compare index --system tantivy --corpus corpus/corpus.jsonl --out indexes/tantivy --threads 1 --heap-mb 1024 --results results
#   --tantivy-commit-every 50000                 -> 4-segment tantivy index (record with --label tantivy-4seg)
#   --force-merge (rustic)                       -> merge_segments into 1 segment after the last refresh
#   --label NAME                                 -> name recorded in results (e.g. rustic-1seg)

# cold startup: 5 fresh processes each (spawns itself as `startup-child`)
../target/release/compare startup --system rustic  --index indexes/rustic  --queries corpus/queries.jsonl --runs 5 --results results
../target/release/compare startup --system tantivy --index indexes/tantivy --queries corpus/queries.jsonl --runs 5 --results results

# warm query latency + hit-count cross-check (prints mismatches vs any other *-query.json already present)
../target/release/compare query --system tantivy --index indexes/tantivy --queries corpus/queries.jsonl --warmup 2 --iters 5 --results results
../target/release/compare query --system rustic  --index indexes/rustic  --queries corpus/queries.jsonl --warmup 2 --iters 5 --results results

# vectors (rustic only; tantivy writes an N/A record)
../target/release/compare vectors --system rustic  --vectors corpus/vectors.f32 --dim 128 --queries 20 --results results
../target/release/compare vectors --system tantivy --results results

# merge everything (including Lucene's JSON if present) into a markdown table
../target/release/compare report --inputs results/*.json --out results/RESULTS.md
```

Each subcommand writes `results/<system>-<subcommand>.json` with the envelope
`{system, subcommand, timestamp_unix, machine, params, metrics}`.

## 4. Build and run Lucene

### 4a. Without Gradle (what produced the checked-in results)

```sh
# Any JDK 21 works; the results used a Temurin 21 unpacked into bench/tools/.
export JAVA_HOME=$(ls -d "$PWD"/tools/jdk-21*/Contents/Home)     # or: brew install openjdk@21
mkdir -p lucene/lib && cd lucene/lib
V=10.5.1
for a in lucene-core lucene-analysis-common lucene-queryparser lucene-queries lucene-sandbox; do
  curl -fsSO "https://repo1.maven.org/maven2/org/apache/lucene/$a/$V/$a-$V.jar"
done
curl -fsSO https://repo1.maven.org/maven2/com/google/code/gson/gson/2.11.0/gson-2.11.0.jar
cd ..
./run.sh index --corpus ../corpus/corpus.jsonl --out ../indexes/lucene --results ../results   # compiles into ./out on first use
./run.sh query   --index ../indexes/lucene --queries ../corpus/queries.jsonl --warmup 2 --iters 5 --results ../results
./run.sh startup --index ../indexes/lucene --queries ../corpus/queries.jsonl --runs 5 --results ../results
./run.sh vectors --vectors ../corpus/vectors.f32 --out ../indexes/lucene-vec --dim 128 --results ../results
```

`run.sh` compiles with `javac --release 21 -cp 'lib/*'` and runs with
`-Xms4g -Xmx4g -XX:+UseParallelGC --enable-native-access=ALL-UNNAMED
--add-modules jdk.incubator.vector` (the last flag enables Lucene's SIMD
postings/vector code paths).

### 4b. With Gradle

Install a JDK and Gradle:

```sh
brew install openjdk@21 gradle
export JAVA_HOME="$(brew --prefix openjdk@21)/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"
cd bench/lucene
gradle wrapper --gradle-version 8.10          # once: generates ./gradlew (+ gradle/wrapper/*)
```

If you prefer not to generate the wrapper, replace `./gradlew` with `gradle`
below.  Lucene version is `luceneVersion` in `gradle.properties` (10.5.1);
bump to the newest 10.x if that artifact is no longer on Maven Central.

```sh
./gradlew run --args="index   --corpus ../corpus/corpus.jsonl --out ../indexes/lucene --results ../results"
./gradlew run --args="startup --index ../indexes/lucene --queries ../corpus/queries.jsonl --runs 5 --results ../results"
./gradlew run --args="query   --index ../indexes/lucene --queries ../corpus/queries.jsonl --warmup 2 --iters 5 --results ../results"
./gradlew run --args="vectors --vectors ../corpus/vectors.f32 --out ../indexes/lucene-vec --dim 128 --queries 20 --results ../results"
cd .. && ../target/release/compare report --inputs results/*.json --out results/RESULTS.md
```

Lucene specifics (see the Javadoc in each class):

* Analyzer: custom `CharTokenizer.fromTokenCharPredicate(Character::isLetterOrDigit)`
  + `LowerCaseFilter` = lowercase + split on non-alphanumerics (StandardAnalyzer
  applies Unicode word-break rules and SimpleAnalyzer splits on digits, so
  neither matches the other two systems exactly).
* `MMapDirectory`, `IndexWriterConfig` with 1 GB RAM buffer, no compound
  files, single indexing thread, default `TieredMergePolicy`, one commit.
* `price`/`ts` are `NumericDocValuesField` only (a columnar fast field like
  the other systems); ranges use `NumericDocValuesField.newSlowRangeQuery`.
  `tag` is `StringField` + `SortedDocValuesField`.
* `TopScoreDocCollectorManager(10, Integer.MAX_VALUE)` so `totalHits` is exact;
  no executor, so segments are searched sequentially like the others.
* Vectors: `KnnFloatVectorField` with `Lucene99HnswVectorsFormat(maxConn = 16,
  beamWidth = 100)` via a `FilterCodec`, EUCLIDEAN, `forceMerge(1)`.  Lucene has
  no separate `ef`; `KnnFloatVectorQuery(field, q, k = ef)` is truncated to 10.
* Startup: the parent spawns a fresh JVM (`java -cp <same classpath>
  bench.Main startup-child`) per run; the child reports JVM start→`main()`
  separately (`jvm_startup_ms`) from `DirectoryReader.open` and first query.

### 4c. Qdrant and hnswlib (vectors only)

```sh
cd bench/tools
python3 -m venv venv && venv/bin/pip install numpy hnswlib qdrant-client   # hnswlib builds from source on py3.14
curl -fsSLO https://github.com/qdrant/qdrant/releases/download/v1.19.1/qdrant-aarch64-apple-darwin.tar.gz
mkdir -p qdrant-1.19.1 && tar -xzf qdrant-aarch64-apple-darwin.tar.gz -C qdrant-1.19.1
cd ../qdrant  && ../tools/venv/bin/python bench.py --vectors ../corpus/vectors.f32 --qdrant-bin ../tools/qdrant-1.19.1/qdrant --storage ../indexes/qdrant --results ../results
cd ../hnswlib && ../tools/venv/bin/python bench.py --vectors ../corpus/vectors.f32 --results ../results
```

`qdrant/bench.py` starts the binary itself (storage under `indexes/qdrant`,
telemetry off) and stops it when done.  `bench/run_all.sh --vectors-only`
re-runs every vector benchmark (rustic 1-thread + all-thread build, Lucene,
Qdrant, hnswlib; unfiltered + filtered) back-to-back and regenerates
`RESULTS.md`; steps whose tools are missing are skipped.  Shared Python
helpers (loading, payload, filters, ground truth, stats, JSON envelope) are in
`pycommon.py`.

* **Qdrant**: Euclid, hnsw m = 16 / ef_construct = 100,
  `default_segment_number = 1` (Qdrant's optimizer still settled on 2
  segments for 200k vectors; the count is in the result JSON), integer
  payload indexes on `tag`/`bucket`, everything else default.  Build = gRPC upload + wait until green with all
  vectors indexed (multi-threaded; there is no 1-thread number).  Queries go
  over one keep-alive HTTP/1.1 connection from a single-threaded Python client,
  so QPS/p50 include the round trip and JSON; `server p50` is the `time` field
  Qdrant returns in each response.
* **hnswlib** 0.8: M = 16, ef_construction = 100, built with 1 thread (the
  searched index) and with all threads; `knn_query` with `num_threads = 1`.
  hnswlib's L2 SIMD kernels are SSE/AVX only, so on arm64 it runs scalar
  distance code.  Filtered search uses `filter=<Python callable>`, called per
  visited candidate: that measures Python callback overhead, not hnswlib's C++.

### 4d. Hybrid search (BM25 + k-NN in one request)

```sh
python3 corpus/vectors.py --n 200000 --dim 128 --queries 100 --seed 42 --out corpus/vectors-q100.f32   # same base vectors, 100 queries
bench/run_all.sh --hybrid-only          # rustic, Lucene, Qdrant interleaved, ROUNDS=2 (better of rounds kept)
# or individually:
../target/release/compare hybrid --system rustic --corpus corpus/corpus.jsonl --queries corpus/queries.jsonl --vectors corpus/vectors-q100.f32 --vec-queries 100 --results results
lucene/run.sh hybrid --corpus ../corpus/corpus.jsonl --queries ../corpus/queries.jsonl --vectors ../corpus/vectors-q100.f32 --vec-queries 100 --out ../indexes/lucene-hybrid [--reuse] --results ../results
(cd qdrant && ../tools/venv/bin/python hybrid.py --corpus ../corpus/corpus.jsonl --queries ../corpus/queries.jsonl --vectors ../corpus/vectors-q100.f32 --vec-queries 100 --qdrant-bin ../tools/qdrant-1.19.1/qdrant --storage ../indexes/qdrant-hybrid [--reuse] --results ../results)
../target/release/compare hybrid --system tantivy --results results   # N/A record (no vectors); likewise --system hnswlib (no text)
```

`vectors.f32` keeps its 20 query vectors (the vector benchmarks use them);
`vectors-q100.f32` is the same generator run with `--queries 100` (identical
base vectors, the first 20 queries identical too).  Corpus doc `i` carries
base vector `i`.  Workload (identical in `compare/src/hybrid.rs`,
`lucene/.../Hybrid.java`, `qdrant/hybrid.py`):

* Index: `body` BM25 (k1 1.2, b 0.75), `price` numeric, vector L2 with HNSW
  m = 16 / ef_construction = 100; one text segment and one graph.
* Query vector `i` (100) is paired with text query `or2[i/2]` (even `i`) or
  `term[i/2]` (odd `i`).  k = 10, num_candidates = 100.
* `sum` (ES `query` + `knn`): BM25 + k-NN score (1/(1+d²)) for docs in the
  global k-NN top-10.  rustic `Index::hybrid_search`; Lucene
  `BooleanQuery(SHOULD text, SHOULD KnnFloatVectorQuery(k=100))` whose merged
  k-NN result is truncated to the top 10 (subclass overriding
  `mergeLeafResults`, ES's per-shard num_candidates -> global k); Qdrant a
  `formula` query (approximation, see `hybrid.py`).
* `rrf`: text top-100 + k-NN top-100 fused with RRF (rank_constant 60, window
  100): rustic `fusion::rrf`, Lucene in Java, Qdrant `prefetch` + `rrf`
  (k = 60).  Qdrant's BM25 = client-built sparse TF part + server `idf`
  modifier.
* Filters `price <= q10` / `price <= q50` (nearest-rank price quantiles): text
  becomes `bool { must: text, filter: range }`; the same range is the k-NN
  pre-filter inside graph traversal.
* The checked-in hybrid numbers were produced on the EC2 build box
  (m7i.4xlarge, see `results/ISSUES.md` harness notes) with
  `COMPARE=<binary> HYBRID_RESULTS=~/hybrid-work/results HYBRID_INDEXES=~/hybrid-work/indexes
  BENCH_MACHINE="EC2 m7i.4xlarge" ROUNDS=2 ./run_all.sh --hybrid-only`
  (`BENCH_MACHINE` is recorded as `machine.label`; per-round JSONs are in
  `results/hybrid-rounds/`, the better round per mode/filter is merged by
  `compare hybrid-merge`).  `RESULTS.md` flags that section as a different
  machine from the M1 tables.
* 3 warm-up rounds + 10 timed rounds over the 100 queries per (mode,
  filter); k-NN-leg recall@10 vs brute force (same filter); fused top-10 ids
  kept for the overlap@10 table.  Lucene's query cache is disabled.

## 5. Methodology

* **Same analysis everywhere**: lowercase + split on non-alphanumerics
  (rustic `standard`, tantivy `default`, Lucene custom analyzer above).  The
  corpus contains only `[a-z]+` words, so apostrophe/digit handling differences
  between the three cannot show up.
* **Same scoring**: BM25 k1 = 1.2, b = 0.75 with per-segment statistics
  (rustic `Bm25Params::default()` via `segment_term_scorer`; tantivy default;
  Lucene `BM25Similarity(1.2, 0.75)`).  Only hit *counts* are cross-checked,
  because Lucene ≥ 8 drops the `(k1+1)` numerator factor so scores differ by
  a constant.
* **Same top-k = 10** everywhere; the total hit count is exact (rustic
  `collect_top_k` total, tantivy `Count`, Lucene `totalHitsThreshold = MAX`).
* **Same segment search**: all three iterate segments sequentially on one
  thread.  rustic builds per-segment scorers from the public query layer
  (`segment_term_scorer`, `IntersectionN`, `Union`, `segment_phrase_scorer`,
  fast-field `range_query` → `ConstScorer<VecDocSet>`), collects with
  `collect_top_k(10, |d| !deleted)` and merges by score.
* **Stored fields** are enabled in all three (`TEXT | STORED`, `Field.Store.YES`)
  because rustic always stores documents; otherwise on-disk sizes would not be
  comparable.
* **Indexing**: single writer thread, 1 GB heap for tantivy/Lucene.  rustic has
  no memory budget; `--refresh-every N` seals one segment file every N docs
  (default 100 000; 50 000 was used for the 200k run to get 4 segments).
  Throughput = docs / wall time where wall time covers reading the JSONL,
  analysis, indexing and the final commit.  Segment counts are reported because
  they affect query latency: tantivy and Lucene produce 1 segment for 200k docs
  with a 1 GB buffer, rustic 4.  Both layouts are therefore reported for both
  Rust systems: `rustic` (4 seg) / `rustic-1seg` (force-merged; the merge time
  is included in its indexing wall time) and `tantivy` (1 seg) /
  `tantivy-4seg` (commit every 50k).  Compare like with like.
* **Warm query latency**: first one untimed pass over the whole query set
  (`--global-warmup 1`, so Lucene's JIT is not in the samples; done for all
  systems), then for every query `--warmup 2` untimed executions,
  then `--iters 5` timed executions.  Latencies of all queries of a class are
  pooled (200 queries × 5 iters = 1000 samples for `term`) and p50/p99 are the
  nearest-rank percentiles (`ceil(p·n)`-th smallest).  Per-query p50 and hit
  counts are kept in the JSON for the cross-check.
* **Cold startup**: `--runs 5` fresh processes.  Before each run the harness
  tries `sudo -n purge` then `purge` to drop the page cache; on this machine
  neither succeeded without a password, so the numbers are labelled **warm**
  (page cache hot).  Run `sudo -v` first to pre-authorise sudo if you want a
  real cold-cache measurement.  Reported: median over runs of `open`, `first
  query`, `open + first query` and total process wall time (the first run of
  each series is noticeably slower because the executable itself is cold; the
  per-run values are in the JSON).  The first query is the first `term` query
  in `queries.jsonl` (a rare term).
* **Vectors**: HNSW m = 16, ef_construction = 100, L2, deterministic seed.
  Ground truth is an exact brute-force scan over the same vectors.  For each
  ef the 20 queries are run for 5 untimed warm-up rounds and 20 timed rounds;
  QPS = executions / total time, recall@10 from the first round.
* **Filtered vectors**: every base vector `i` carries `tag = i % 100` and
  `bucket = i % 10` (cluster membership is random, so these are uncorrelated
  with position).  Per query `qi` the filters are `tag == qi % 100` (1 %),
  `bucket == qi % 10` (10 %) and `bucket < 5` (50 %); ground truth is brute
  force among the matching vectors; ef = 100.  rustic goes through
  `Index::knn_search_with` with a `RangeI64` filter and the default
  `PlannerConfig` (the engine path: filter query -> point ids ->
  `VectorStore::search`, whose planner picks exact scan vs filtered HNSW); the
  JSON/report also show `VectorStore::search` timed alone with precomputed
  point ids.  Lucene: `KnnFloatVectorQuery` with an `IntField` exact/range
  filter.  Qdrant: `match` / `range` filter on integer payload indexes.

## 6. Reading the results

`results/RESULTS.md` has one table per metric with one row (or column) per
system; "n/a" means that system did not produce the result (e.g. Lucene not
run, or tantivy for vectors).  The *Hit-count cross-check* section lists every
query whose exact hit count differs between two systems.  `results/ISSUES.md`
records harness notes and the fixes made during the benchmark work.
