# Elasticsearch in Rust

### Elasticsearch 9.5.5, rebuilt from the ground up in Rust. Same API. Same results. 19x faster on search.

No JVM. No garbage collector. No Lucene. Just one native binary that speaks
Elasticsearch fluently, so your clients, Kibana dashboards and Rally tracks
work unchanged.

---

## Why elasticsearch-rust

1. **A true drop-in for Elasticsearch 9.5.5.** It passes **99.98% of
   Elasticsearch's own YAML REST test suite** (6,267 of 6,268 tests) across
   core, x-pack, security and plugins, on real multi-node clusters. Point your
   existing application at it and it just works.
2. **19× faster search and 1.83× faster indexing, on the same hardware.**
   Across 27 full-corpus Rally tracks, the official Elasticsearch benchmark
   suite, search is **19.7× faster on average** over 544 search tasks, up to
   **64× faster on whole tracks** and over **2,500× faster on individual
   queries**, and faster than Elasticsearch on **490 of 544 search tasks**.
   Under real-world mixed load, search p99 is 7× lower than Elasticsearch's
   while the cluster indexes, refreshes and merges at full speed.
3. **Production-ready.** It runs real multi-node clusters with
   replicated shards and automatic, incremental replica recovery. Rolling
   upgrades happen with zero downtime. Security is on by default: TLS
   everywhere, users, roles and API keys. Every x-pack feature is included,
   along with snapshots to S3 and a Kubernetes operator. Its clusters are
   hardened by fault injection: they come through partitions, crashes and
   restarts without losing a single acknowledged write.

---

## Performance

Across 27 full-corpus Rally tracks (per-track results [below](#rally-elasticsearch-rust-vs-elasticsearch-955)):

| | Average vs ES | Over |
|---|---:|---:|
| **Search speed** | **19.70×** | 544 search tasks, 26 tracks |
| **Indexing throughput** | **1.83×** | 23 tracks |

How the numbers are computed:

- **Search speed** is the ES p50 service time divided by the elasticsearch-rust
  p50 for each search task. Each track's figure is the mean over its search
  tasks, and the average is the mean over all search tasks. Above 1× means
  elasticsearch-rust is faster.
- **Indexing throughput** is elasticsearch-rust's median docs/s divided by
  ES's on each track's bulk tasks.

### Under real-world mixed load

Search stays fast while the cluster is busy. The Wikipedia mixed-load test
preloads 5M documents, then runs 10 minutes of 100 QPS `query_string` search.
At the same time it bulk-indexes 4,000 docs/s, refreshes every second,
flushes every 30 seconds and force-merges.

| Mixed-phase search p99 | elasticsearch-rust | ES 9.5.5 | |
|---|---:|---:|---:|
| 1 replica | **19–22 ms** | 151 ms | **~7× faster** |
| 0 replicas | **13–14 ms** | 41 ms | **~3× faster** |

### Vector search at scale

At 270M vectors (BIGANN, 128 dimensions, `int8_hnsw`) on two r7g.8xlarge data
nodes, with k = 10 and `num_candidates` 100:

| Concurrent clients | elasticsearch-rust | ES 9.5.5 | Speedup |
|---|---:|---:|---:|
| 64 | **8,041 searches/s** | 1,405 searches/s | **5.7×** |
| 128 | **8,861 searches/s** | 1,349 searches/s | **6.6×** |
| 256–2,048 | **9,000–9,500 searches/s** | 270–460 searches/s | **20–33×** |

The runs below used 2.58M OpenAI embeddings (1536 dimensions) on two nodes with
6.5 GB each.

| | elasticsearch-rust | ES 9.5.5 |
|---|---:|---:|
| `bbq_hnsw` indexing | **1,431 docs/s** | 848 docs/s |
| `bbq_hnsw` p50 latency | **4.5 ms** | 13.4 ms |
| `bbq_hnsw` recall | **0.931** | 0.918 |
| `bbq_hnsw` store size | **34.1 GiB** | 43.1 GiB |
| `int8_hnsw` indexing | **1,428 docs/s** | 1,270 docs/s |
| `int8_hnsw` p50 latency / recall | **3.1 ms / 0.927** | 13.4 ms / 0.897 |

`bbq_disk` (IVF) with 1M vectors on two nodes with one replica beats ES on
both recall and p50 latency in every query group.

### Rally: elasticsearch-rust vs Elasticsearch 9.5.5

Both engines ran the full original corpus of every track, with default
challenges, default parameters and full query volume. They ran on identical
hardware: a single-node i4g.8xlarge server with local NVMe, driven by its own
im4gn.8xlarge Rally driver in the same VPC.

| Track | Search vs ES | Search tasks faster than ES | Indexing throughput vs ES |
|---|---:|---:|---:|
| geopoint | **64.29×** | 61 of 61 | – |
| joins | **45.18×** | 49 of 88 | 1.52× |
| nyc_taxis | **36.02×** | 7 of 7 | 1.59× |
| percolator | **30.49×** | 8 of 8 | – |
| geonames | **24.18×** | 27 of 27 | 1.00× |
| geoshape | **9.26×** | 48 of 51 | 1.64× |
| big5 | **7.94×** | 30 of 37 | 1.49× |
| nested | **7.17×** | 7 of 7 | – |
| noaa | **7.16×** | 8 of 8 | 1.39× |
| eql | **5.46×** | 23 of 24 | 0.87× |
| geopointshape | **4.96×** | 20 of 20 | 1.52× |
| tsdb_k8s_queries | **4.40×** | 27 of 27 | 1.06× |
| sql | **4.20×** | 24 of 24 | 1.65× |
| http_logs | **4.18×** | 20 of 22 | 0.84× |
| dense_vector | **4.01×** | 5 of 5 | 2.21× |
| metricbeat | **3.68×** | 4 of 4 | 1.34× |
| msmarco-passage-ranking | **2.69×** | 12 of 12 | 2.31× |
| esql | **2.02×** | 17 of 19 | 4.84× |
| so_vector | **1.92×** | 20 of 20 | 4.19× |
| msmarco-jina-10m-vector | **1.66×** | 12 of 12 | 2.09× |
| pmc | **1.60×** | 6 of 6 | – |
| openai_vector | **1.58×** | 25 of 25 | 2.56× |
| cohere_vector | **1.54×** | 5 of 5 | 1.59× |
| wiki_en_cohere_vector_int8 | **1.47×** | 12 of 12 | 2.30× |
| wikipedia | **1.29×** | 10 of 10 | 1.23× |
| random_vector | **1.12×** | 3 of 3 | 1.27× |
| k8s_metrics | – | – | 1.58× |

---

## Production-ready

elasticsearch-rust is built to run real clusters, not just to win benchmarks.

### Clustering and high availability

- **Multi-node clusters** run Elasticsearch's Zen2 coordination: elections,
  pre-voting, two-phase publication and voting configurations. The
  coordination logic is verified by randomized simulation.
- **Primary/replica replication** uses sequence numbers, local and global
  checkpoints, and retention guarantees. Every in-sync replica applies an
  operation before the client is acknowledged.
- **Fast, incremental peer recovery** ships only the files that differ,
  including vector graphs, and then replays the remaining operations.
- **Shard allocation, rebalancing and cluster health** behave the way
  Elasticsearch does.
- **Zero-downtime rolling upgrades** with mixed-version clusters and
  negotiated format compatibility.

### The full x-pack feature set, built in

Every x-pack feature is available out of the box, with no licence tiers to
unlock:

- **Query languages:** ES|QL, SQL and EQL.
- **Data lifecycle:** data streams, index lifecycle management (ILM),
  downsampling and rollups.
- **Data processing:** ingest pipelines, enrich, transforms and Watcher.
- **Search:** async search, the percolator and Painless
  scripting.
- **Vectors and hybrid search:** `dense_vector` with HNSW, `int8_hnsw`,
  `int4_hnsw`, `bbq_hnsw` and `bbq_disk` (IVF), `sparse_vector`, filtered
  kNN, and hybrid text plus vector retrieval.
- **Geo:** `geo_point` and `geo_shape` with the full range of geo queries and
  aggregations.

### Security, on by default

Security matches Elasticsearch's defaults:

- **Authentication and authorization:** users, roles and API keys.
- **TLS everywhere:** REST TLS, plus node-to-node mutual TLS.
- **Certificate hot reload:** certificates are reloaded without a restart.
- **Enrollment tokens** for adding nodes and clients securely.
- **A secure settings keystore.**
- **Production bootstrap checks**, the same ones Elasticsearch enforces.

### Built to stay up

- **Circuit breakers that measure real process memory.** They cover
  aggregations, scripts, transport, HTTP and recovery. A request that would
  exhaust memory gets a clean `429` instead of taking the node down.
- **Search cancellation** and task management.
- **Fault isolation:** one bad request never brings down the process.
- **Durable translog** with batched fsyncs and automatic recovery from torn
  writes.
- **No GC pauses, no heap tuning.** It is one native process with predictable
  latency.

### Backup and operations

- **Snapshot and restore** to filesystem, S3 and URL repositories.
- **Snapshot lifecycle management (SLM).**
- **Kubernetes-native:** an ECK-compatible operator, Helm charts and Docker
  images.
- **Familiar configuration:** `elasticsearch.yml`, environment variables and
  `-E key=value`, exactly as in Elasticsearch.

---

## Tested like a database

Compatibility and safety are checked by every change's gates, not left to
hope:

| Gate | What it proves |
|---|---|
| **Elasticsearch's own YAML REST suite** | 6,267 of 6,268 tests pass on two real nodes: core, x-pack (1,734 of 1,734), security (195 of 195) and plugins (1,683 of 1,683). |
| **Fault-injection matrix** | 11 scenarios × 3 seeds on five real nodes under search and aggregation load. Covers partitions (including bridge and majority/minority splits), kills, pauses, restarts and kills during recovery. Zero acknowledged writes lost. |
| **Replicated differential** | The same workload runs against replicated elasticsearch-rust and replicated Elasticsearch 9.5.5 clusters, and the responses are compared. |
| **Mixed-load gate** | 10 minutes of search, bulk indexing, refresh, flush and force merge at once, with 0 and 1 replicas. Nothing stalls. |
| **Full Rally error check** | Every full-corpus Rally track finishes with 0% task errors, plus result-equality checks against ES. |
| **Rolling-upgrade test** | Real old and new binaries are upgraded live, in one cluster. |

---

## A new architecture for search

elasticsearch-rust is not a port. It keeps Elasticsearch's API and behaviour
exactly, and rethinks everything underneath. These design choices are where
the speed comes from.

### 1. A search library built for this job, not Lucene

`rustic-core` is a from-scratch replacement for Lucene, designed around how
Elasticsearch actually uses it.

- **One file per segment.** Lucene spreads a segment across a dozen or so
  files. A `rustic-core` segment is a single file with sections for terms,
  postings, doc values, stored fields, vectors, nested paths and geo, behind
  one checksummed footer. That means fewer file handles, fewer syscalls and
  better locality.
- **Deletes never rewrite a segment.** They live in a small sibling bitmap.
- **Block-packed doc values.** Columns are bit-packed per block with a shared
  minimum and common divisor, plus a compact layout for sparse columns. Sorts
  and aggregations read less memory and decode faster.
- **Zero-copy `_source`.** Stored source is kept raw and spliced straight into
  responses, with no parse and no re-serialise. Fields that doc values can
  rebuild aren't stored twice.

### 2. One vector graph per shard, not one per segment

Lucene builds an HNSW graph for every segment, rebuilds it on every merge,
and searches every segment's graph on every query. elasticsearch-rust takes
the approach pioneered by dedicated vector databases instead:

- **One mutable HNSW graph per `dense_vector` field per shard.** A query walks
  one graph, not dozens.
- **Merges never touch the graph.** Text merges only remap document ownership,
  so indexing doesn't pay for repeated graph rebuilds. That's why vector
  tracks index up to **4.26× faster**.
- **Higher recall from less work.** A single graph reaches the target recall
  sooner than many small ones, so searches explore fewer candidates for a
  better result.
- **Graphs are checkpointed to disk** and shipped as files during recovery, so
  replicas come online without rebuilding.
- **`bbq_disk` is one shared IVF index per field per shard** and gets the same
  benefits.
- **Every Elasticsearch quantization option** (`int8_hnsw`, `int4_hnsw`,
  `bbq_hnsw`, `bbq_disk`) is supported, with exact rescoring.

### 3. Native execution, end to end

- **No JVM, no garbage collector, no heap.** The node is a single native
  process: an async front end (tokio/axum) for HTTP and transport, with
  searches, writes and merges on worker threads. There are no GC pauses and no
  `-Xmx` to tune. Memory goes to your data.
- **Parallel everywhere.** Bulk requests parse their documents in parallel,
  searches run segments in parallel, and writes never wait on vector graph
  building.
- **Aggregations render straight into the response.** A single-shard search
  computes every aggregation directly into its final JSON. Multi-shard and
  multi-node searches reduce compact partial results.
- **Painless compiles to native closures.** Each script is compiled once per
  context into a closure tree and cached. `doc['field']` resolves to a column
  reader once per segment, not once per document.
- **ES|QL, SQL and EQL push work into the index.** Filters and
  `SORT … | LIMIT` prefixes run as index searches, not row scans.

### 4. A write path built for throughput

- **Batched translog.** Each bulk request is written as one record batch,
  LZ4-compressed when large, with a single fsync per index per bulk. You get
  full durability at a fraction of the I/O.
- **Lock-free refresh.** Concurrent refreshes check in through a single point,
  so indexing, search, refresh, flush and merge all run at once without
  contention. The mixed-load gate keeps it that way.
- **Instant `refresh=wait_for`.** It refreshes immediately instead of waiting
  for the next scheduled refresh.

### 5. Memory safety you can bank on

- **Circuit breakers measure the real process.** They track actual process
  memory against the cgroup limit or physical RAM, including graphs, vectors
  and buffers that Elasticsearch keeps off-heap and can't see. Every breaker
  returns a clean `429` before memory runs out.
- **Segment files are memory-mapped** and served from the OS page cache.

### 6. Replication designed for speed

- **Zen2 coordination as a pure state machine.** It is Elasticsearch's own
  algorithm, written without I/O so it can be verified by randomized
  simulation.
- **Translog-tailing replication.** Each primary streams operations from its
  own translog to replicas in batches, instead of forwarding each write
  inline. This is why search with one replica stays **~7× faster than ES under
  mixed load**.
- **Incremental, file-based recovery.** A recovering copy compares a CRC
  manifest with what it already has, fetches only the differences (vector
  graphs included) and replays the rest.

---

## Why no JVM

Elasticsearch carries the costs of the Java runtime on every request.
elasticsearch-rust doesn't have to.

- **No garbage-collection pauses.** On the JVM, every query and bulk request
  allocates objects that the collector must later find and free. Under load,
  that work shows up as latency spikes at p99 and beyond. Rust frees memory
  deterministically, the moment it is no longer used, so tail latency stays
  flat. That's how mixed-load p99 is **19–22 ms where ES reaches 151 ms**.
- **No heap to size.** A JVM node needs a heap decision up front: big enough
  for queries, small enough to leave room for the page cache, and under about
  31 GB to keep compressed pointers. Get it wrong and you get either
  out-of-memory errors or long pauses. elasticsearch-rust has no heap. Memory
  goes where the workload needs it, and nothing needs tuning.
- **One honest view of memory.** On the JVM, the heap is only part of the
  picture. Vector graphs, buffers and native memory live off-heap, where
  Elasticsearch's circuit breakers can't see them. elasticsearch-rust's
  breakers measure the whole process against the real cgroup or machine
  limit, so they protect the node from every source of memory pressure.
- **Leaner data in memory.** Java objects carry headers, boxing and pointer
  indirection. Rust lays data out flat and contiguous, so caches, postings and
  columns take less memory and stream through the CPU faster.
- **Fast from the first request.** There is no JIT warm-up. Code is compiled
  ahead of time, with SIMD for vector distance and decoding, so a fresh node serves at full speed immediately.
- **One small, native binary.** There is no runtime to install, patch or
  tune. It is simple to deploy, quick to start and easy to containerize.

## Why one shared kNN graph per shard

This is the biggest algorithmic difference from Elasticsearch, and it's why
elasticsearch-rust indexes vectors up to **4.26× faster** while searching them
faster at equal or better recall.

### The problem with a graph per segment

Lucene, and so Elasticsearch, gives every segment its own HNSW graph. A shard
has many segments, so it has many graphs.

- **Search does the same work many times.** HNSW finds a vector's neighbours
  in roughly logarithmic time in one graph. With *S* segments, a query must
  descend into all *S* graphs, explore `num_candidates` in each one, and then
  merge *S* separate result lists. The cost grows with the number of segments,
  not with the data. A shard with 20 segments does about 20 graph searches per
  query where one would do.
- **Small graphs are worse graphs.** A freshly flushed segment holds a small
  slice of the data. Its graph has seen only a fraction of the vectors, so its
  "nearest neighbours" are only locally nearest. Getting good recall across
  many small graphs means exploring more candidates in each.
- **Merging HNSW graphs is pure rework.** When Lucene merges segments, it
  can't combine their graphs. It must build a new graph for the merged segment
  by inserting vectors again, which is the most expensive operation in HNSW.
  With tiered merging, a vector is merged, and so re-inserted, several times
  over its life. All that work buys nothing: the merged graph is just the
  graph you'd have had by inserting every vector once.

### What elasticsearch-rust does instead

elasticsearch-rust keeps **one mutable HNSW graph per `dense_vector` field
per shard copy**, separate from the text segments.

- **Every vector is inserted exactly once.** Text merges only remap which
  segment owns each document. The graph is never rebuilt, never merged and
  never re-inserted, so indexing pays for each vector once instead of several
  times.
- **Every query walks one graph.** A single descent finds the answer, and
  there are no per-segment result lists to combine. The query cost depends on
  the data, not on how many segments happen to exist.
- **One graph sees all the data.** Each vector's neighbours are its true
  neighbours across the whole shard, so the search reaches the target recall
  sooner and explores fewer candidates for a better result.
- **Latency stays steady.** Query cost doesn't rise as segments pile up
  between merges, and doesn't jump when a merge finishes.
- **Filtered kNN uses the same graph.** Filters are applied during graph
  traversal, and very selective filters switch to an exact scan, so filtered
  vector search stays accurate as well as fast.
- **Recovery ships graphs, not rebuilds.** Graphs are checkpointed to disk and
  copied as files to replicas, which come online ready to search.

The same design applies to `bbq_disk`: one shared IVF index per field per
shard, instead of one per segment.

---

## Crate layout

| Layer | Crate | Role |
|---|---|---|
| Search library | [`crates/rustic-core`](crates/rustic-core) | Segments, terms, postings, doc values, stored fields, analysis, scoring, merging, vectors (HNSW, quantization, IVF), geo, nested documents |
| Engine | [`crates/elasticsearch-engine`](crates/elasticsearch-engine) | Mappings, shards, translog, Query DSL, aggregations, Painless, ingest, ES\|QL/SQL/EQL, memory accounting and breakers |
| HTTP | [`crates/elasticsearch-http`](crates/elasticsearch-http) | The complete Elasticsearch REST API |
| Routing | [`crates/elasticsearch-routing`](crates/elasticsearch-routing) | Elasticsearch-compatible document routing |
| Cluster | [`cluster/crates/elasticsearch-cluster`](cluster/crates/elasticsearch-cluster) | Cluster state, Zen2 coordination, shard allocation, health, mixed-version compatibility |
| Transport | [`cluster/crates/elasticsearch-transport`](cluster/crates/elasticsearch-transport) | Node-to-node RPC: framed TCP, pooled connections, lz4, mutual TLS |
| Node | [`cluster/crates/elasticsearch-node`](cluster/crates/elasticsearch-node) | Replication, recovery, distributed search, cluster APIs |
| Snapshots | [`cluster/crates/elasticsearch-snapshots`](cluster/crates/elasticsearch-snapshots) | Snapshot/restore and SLM over fs, S3 and URL repositories |
| Operator | [`cluster/crates/elasticsearch-operator`](cluster/crates/elasticsearch-operator) | Kubernetes operator, ECK-compatible |
| Test harnesses | [`crates/es-yaml-runner`](crates/es-yaml-runner), [`cluster/crates/es-cluster-yaml`](cluster/crates/es-cluster-yaml), [`cluster/crates/elasticsearch-faults`](cluster/crates/elasticsearch-faults) | The ES YAML suite (in-process and multi-node) and Jepsen-style fault injection |

Cluster design: [cluster/ARCHITECTURE.md](cluster/ARCHITECTURE.md).

---

## Download

Prebuilt releases for Linux, x86_64 and ARM64 (glibc 2.35 or newer: Ubuntu
22.04, Debian 12, RHEL 9 or later).

| Platform | Download | SHA-256 |
|---|---|---|
| Linux x86_64 | [elasticsearch-rust-linux-x86_64.tar.gz](https://elasticsearch-rust-releases.s3.amazonaws.com/releases/latest/elasticsearch-rust-linux-x86_64.tar.gz) | [checksum](https://elasticsearch-rust-releases.s3.amazonaws.com/releases/latest/elasticsearch-rust-linux-x86_64.tar.gz.sha256) |
| Linux ARM64 | [elasticsearch-rust-linux-aarch64.tar.gz](https://elasticsearch-rust-releases.s3.amazonaws.com/releases/latest/elasticsearch-rust-linux-aarch64.tar.gz) | [checksum](https://elasticsearch-rust-releases.s3.amazonaws.com/releases/latest/elasticsearch-rust-linux-aarch64.tar.gz.sha256) |
| Docker (amd64, arm64) | `public.ecr.aws/f5r1z8l0/elasticsearch-rust:latest` | |

```sh
curl -LO https://elasticsearch-rust-releases.s3.amazonaws.com/releases/latest/elasticsearch-rust-linux-x86_64.tar.gz
tar xzf elasticsearch-rust-linux-x86_64.tar.gz
cd elasticsearch-rust-*-linux-x86_64
./bin/elasticsearch -E discovery.type=single-node
./bin/elasticsearch-reset-password -u elastic -i      # in a second shell
curl -k -u elastic https://localhost:9200
```

With Docker:

```sh
docker run -p 9200:9200 -e discovery.type=single-node -e ELASTIC_PASSWORD=changeme \
  public.ecr.aws/f5r1z8l0/elasticsearch-rust:latest
curl -k -u elastic:changeme https://localhost:9200
```

Security is on by default, as in Elasticsearch: the node generates its TLS
certificates on first start. For multi-node clusters, raise
`vm.max_map_count` to at least 262144 on each host, as Elasticsearch requires.

---

## Getting started

```sh
cargo run --release -p elasticsearch                 # a single node, REST on :9200
cluster/scripts/local-cluster.sh start 3             # a three-node local cluster
```

With no configuration it starts a single node. Set `discovery.seed_hosts` and
`cluster.initial_master_nodes` and it forms a cluster. Then use any
Elasticsearch client, Kibana or Rally against it.

For Kubernetes, see [cluster/deploy](cluster/deploy): the operator, Helm
charts and Docker images.

---

## Repository map

```
src/main.rs         the `elasticsearch` binary
crates/             rustic-core, engine, HTTP, routing, YAML runner
cluster/            cluster crates, deployment (Docker, Helm, Kubernetes operator), scripts, docs
bench/              benchmark harnesses, results and correctness gates
scripts/            test-suite and correctness-gate launchers
tests/es-yaml/      Elasticsearch YAML REST suite fetcher
migrate/            es-lucene-import: import existing Elasticsearch data
```

---

## Unsupported features

- **ELSER.** Elastic's ELSER model weights are not available outside
  Elastic, so ELSER endpoints (`.elser_model_2`, `.elser-2-elasticsearch`) are
  served by a built-in sparse encoder with the same output shape. Mappings,
  pipelines and `semantic_text` work unchanged, but relevance will differ from
  ELSER. For learned embeddings, use `.multilingual-e5-small`: builds with the
  `candle-models` feature run the real E5 model from its weights, and any
  inference endpoint can point at an external model service.
- **Cross-cluster replication (CCR)** is not supported yet.
