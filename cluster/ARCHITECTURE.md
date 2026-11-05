# Cluster layer architecture

The single-node engine (`crates/elasticsearch-engine`, `crates/elasticsearch-http`)
gives one process the full ES REST surface.
This workspace turns N such processes into an Elasticsearch **cluster**: master
election, cluster-state publication, shard allocation, primary/replica
replication, peer recovery, failover, snapshots, and Kubernetes deployment.

Its crates are members of the root cargo workspace, and the root binary
(`src/main.rs`) always runs `elasticsearch_node::cli::main()`.

## Crates

| Crate | Owns | I/O? |
|---|---|---|
| `elasticsearch-cluster` | `state` (model), `coordination` (Zen2 election + publication state machine), `allocation` (deciders, balancer, routing-table transitions, explain, reroute), `routing` (ES-compatible Murmur3 shard routing), `health` | none — pure, deterministic, simulation-tested |
| `elasticsearch-transport` | framed TCP RPC between nodes: handshake, request/response, typed action handlers, connection manager, timeouts, mutual TLS | tokio |
| `elasticsearch-snapshots` | blob stores (`fs`, `s3`, `url`), repository format, incremental shard snapshots, restore, SLM scheduling logic | tokio + object stores |
| `elasticsearch-node` | the cluster-aware server: config (`elasticsearch.yml`, `-E`, env), discovery, drives coordination over transport, applies cluster states to local shards, replication + recovery (`replication/`), distributed request routing, cluster REST APIs, the `elasticsearch` binary | everything |
| `elasticsearch-operator` | Kubernetes operator (CRD `Elasticsearch`, StatefulSets per nodeSet, rolling restarts, scale-down with voting exclusions + shard draining) | kube API |

`deploy/` holds the Dockerfile, Helm chart and raw manifests.

## How a node is put together

```
            HTTP :9200 (axum)                    transport :9300 (TCP)
                 │                                      │
        ┌────────▼─────────┐                 ┌──────────▼──────────┐
        │  cluster front    │──forward/proxy─▶│  action handlers     │
        │  (node crate)     │                 │  (node crate)        │
        └──┬──────────┬────┘                 └──┬─────────┬────────┘
   local   │          │ cluster APIs           │         │
   dispatch▼          ▼                        ▼         ▼
  elasticsearch-http  ClusterService ◀── Coordinator (cluster crate, sans-IO)
     router(engine)        │ applies states
           │               ▼
   elasticsearch-engine  ShardManager ── replication/ (tailer, replica applier,
     Node (one copy of        │            checkpoints, recovery)
     each local index)        ▼
                          engine IndexService per local shard copy
```

* **Engine mapping.** Each node runs one engine `Node` (data dir
  `<path.data>/engine`). A shard copy of a single-shard index is the engine
  index of the same name, so every engine API (search, aggs, mappings, ...)
  served locally through `elasticsearch_http::router(engine_node)` is exact.
  Shard `k >= 1` of a multi-shard index lives in a separate engine under
  `<path.data>/engine-shard-<k>` (`shard_engines.rs`); requests select it with
  an internal shard header. Searches fan out per `index#shard` and are merged
  by the coordinating node (`front/fanout.rs`, `front/agg_reduce.rs`).
  Cross-node `query_then_fetch` keeps reader contexts on each shard copy
  between the query and fetch phases (`front/qtf.rs`, `front/contexts.rs`).
  Aggregations are reduced from rendered partial results with mergeable
  sketch state.
* **Requests.** The cluster front classifies each REST request:
  cluster APIs (`_cluster/*`, `_nodes/*`, `_cat/{nodes,shards,allocation,recovery,master,health}`,
  index create/delete/settings/mappings/aliases/templates) are handled by the
  node crate through the master; document writes are forwarded to the node
  holding the primary; reads are served locally if a local active copy
  exists, otherwise proxied to a node that has one (round robin over active
  copies, preferring `_local`/`preference`).
* **Metadata changes** (create index, put mapping, aliases, settings,
  templates) are master tasks: the master computes the new `ClusterState`,
  publishes it, and every node's applier makes its engine match.
  Dynamic mapping updates discovered on a primary are sent to the master as a
  put-mapping task before replicas see the op.

## Replication (primary → replicas)

ES model, built on the engine's translog:

1. A write arrives at the primary's node and is executed locally through the
   engine (any write API: index, bulk, update, delete-by-query, ...). The
   engine assigns `_seq_no` and appends to its translog.
2. The **translog tailer** (`replication::tailer`) reads new records from the
   primary's translog files (`[len u32 LE][crc32 u32 LE][bincode TranslogOp]`,
   files `translog-<gen:08>.tlog`) in seq_no order.
3. Ops are shipped to every in-sync replica (`indices:data/write/replica`),
   batched, with the primary term and the primary's mapping version.
4. Each replica appends the ops to its own replica WAL and applies them with
   `Shard::replay_index` / `replay_delete`, skipping ops whose seq_no is not
   newer than what it already has for that `_id` (tombstones kept for deletes
   until the global checkpoint passes them).
5. The primary acks the client once every in-sync replica has acked up to the
   op's seq_no (or the replica was failed via the master), and rewrites
   `_shards.{total,successful,failed}` in the response.
6. Replicas that fail are reported to the master (`shard-failed`), removed
   from the in-sync set, and re-recovered.

Checkpoints: each copy tracks its **local checkpoint** (highest contiguous
seq_no applied); the primary computes the **global checkpoint** (min over
in-sync copies) and piggybacks it on replica requests.

The engine flushes (and trims its translog) only when the node crate asks it
to: in cluster mode we do not start the engine's own background loop; the
node's maintenance loop flushes only below the min retained seq_no of all
replicas and ongoing recoveries (the equivalent of ES retention leases).

## Peer recovery

Recovery is **file-based** by default, as in ES
(`elasticsearch-node/src/replication/file_recovery.rs`). The source flushes,
pins its last commit and ships its files, including vector graph
checkpoints. The target diffs the manifest against what it already holds by
name, length and CRC-32 (plus a 64-byte tail check), reuses identical files,
fetches the rest, and installs each one only once its whole-file CRC matches.
Then the translog operations above the commit are replayed, and the target is
marked in-sync and STARTED via the master. A target that already holds the
history gets an operations-only recovery. Segment-format and vector-file
compatibility is checked before files are shipped to an older node, which is
what lets mixed-version clusters recover copies during a rolling upgrade.

Node settings that are not in ES: `indices.recovery.file_based` (default
`true`; `false` selects operations-based recovery, which
re-indexes every live document and rebuilds graphs) and
`indices.recovery.graph_checkpoint_lag_ops` (default 50,000: operations a
commit may hold above its graph checkpoint before the source saves graphs
first).

## Coordination

ES "Zen2" semantics: master-eligible nodes, terms, pre-voting, joins
(votes), two-phase publication (publish → quorum of accepts → commit →
apply), voting configuration with automatic shrinking/growing to keep an odd
number of voters, `cluster.initial_master_nodes` bootstrap, leader/follower
fault detection, `cluster.no_master_block`. Implemented as a sans-IO state
machine in `elasticsearch-cluster::coordination`: inputs are messages and
timer ticks, outputs are messages to send and states to apply. The node crate
wires it to the transport.

Safety properties tested by randomized simulation: at most one master per
term; a committed state is never lost; every node applies states in version
order; the voting configuration only changes via committed states.

## Allocation

`elasticsearch-cluster::allocation`: deciders (same-shard, replica-after-
primary-active, max retries (`index.allocation.max_retries` 5), node filters
`index.routing.allocation.{require,include,exclude}.*` and cluster-level
equivalents, awareness attributes (forced and unforced), disk watermarks
(85/90/95%), throttling (`cluster.routing.allocation.node_concurrent_recoveries` 2,
`node_initial_primaries_recoveries` 4), shards-limit
(`index.routing.allocation.total_shards_per_node`, `cluster.…`), enable
(`cluster.routing.allocation.enable` / `cluster.routing.rebalance.enable`),
node shutdown / tier preference) + a weight-based balancer (shard count per
node and per index, ES's default weights 0.45/0.55) + delayed allocation on
node left (`index.unassigned.node_left.delayed_timeout` 1m). Primary failover
promotes an in-sync replica and bumps the primary term.

## Snapshots, resize and remote clusters

Snapshot restore runs on the elected master: each restored primary's node
downloads, verifies and adopts the shard's files from the snapshot, and
replicas then recover from it by file-based peer recovery
(`elasticsearch-node/src/snapshot/restore.rs`).

Shrink, split and `_clone` in cluster mode copy documents into the target
index through scroll + `_bulk` (`elasticsearch-node/src/resize.rs`).

Cross-cluster replication is deliberately out of scope: each cluster is
self-contained, with in-cluster replicas and snapshots for durability and
recovery. Requests that need a leader cluster are rejected with
`connect_transport_exception`.
