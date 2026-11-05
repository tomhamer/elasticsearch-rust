# es-lucene-import: upgrading from Lucene-backed Elasticsearch

An offline upgrade path from Elasticsearch 8.x (Lucene 9) to the Rust
server. It reads a stopped node's `path.data` directly, so it needs no Java,
no running Elasticsearch and no extra load on a production cluster, and it
replays everything through the target's REST API. Indices that use synthetic
`_source` are reindexed to stored `_source` in Elasticsearch first, so every
document is imported from its original source exactly as written.

```
es-lucene-import plan   --data /var/lib/elasticsearch
es-lucene-import import --data /var/lib/elasticsearch --target http://127.0.0.1:9200
es-lucene-import dump   --data /var/lib/elasticsearch --out ./export   # bulk NDJSON + requests, for review/replay
es-lucene-import inspect /var/lib/elasticsearch/indices/<uuid>/0/index # debug a single Lucene index
```

## What it reads

| Source | Lucene / ES format |
|---|---|
| Latest commit | `segments_N` (Lucene 9.x) |
| Segments | `.si`, compound `.cfs`/`.cfe`, `.fnm` (incl. per-commit field-info generations) |
| Hard deletes | `.liv` generations |
| Soft deletes (how ES deletes/updates) | `__soft_deletes` doc values: `Lucene90DocValuesFormat` metadata + `IndexedDISI`, incl. doc-values update generations and force-merged segments |
| Documents | `Lucene90CompressingStoredFieldsFormat`: LZ4 (default), DEFLATE (`best_compression`), ES zstd; sliced chunks for large docs |
| `_id` | ES `Uid` binary encoding (numeric, UTF-8, base64 / auto ids) |
| `_source`, `_routing` | stored fields; SMILE `_source` is converted to JSON |
| Cluster metadata | the node's `_state` Lucene index (`PersistedClusterStateService`): SMILE + DEFLATE-compressed x-content, paged docs, deduplicated mappings |
| Translog safety | `translog.ckp` and `translog-N.ckp` (format 3 BE / 4 LE) |

Nothing in the search structures (postings, points, vectors, other doc
values) is read. The Rust engine rebuilds them from `_source`.

## What it imports, in order

1. Component templates, composable and legacy index templates, ingest
   pipelines, stored scripts and ILM policies. Anything that already exists on
   the target is kept unless `--overwrite` is passed. Elasticsearch's own
   built-ins (`_meta.managed: true`) are skipped because the target ships its
   own versions; `--include-managed` overrides this.
2. Each index is created with its mappings, aliases and settings, minus
   source-cluster-only settings (uuid, version.created, resize/recovery
   state).
3. Every live document is loaded via `_bulk` with its original `_id` and
   `routing`. `?pipeline=_none` is set and `index.default_pipeline` /
   `final_pipeline` / `index.blocks.*` are applied only after the load, so
   stored `_source` (already processed by ES) is not run through pipelines a
   second time.
4. Data stream backing indices are written into the data stream (`create`
   ops), so streams are recreated rather than copied index by index.
5. Closed indices are loaded, verified, then closed again.
6. Per-target `_count` is checked against the number of docs read. Any
   mismatch, failed doc or failed index creation exits non-zero.

## Safety

- **Unflushed writes.** If any translog generation holds operations beyond
  the last Lucene commit, the tool refuses to run, because those writes would
  be silently lost. A clean Elasticsearch shutdown flushes. Otherwise, start
  ES, `POST /_flush`, and stop it. `--allow-unflushed` overrides this.
- **Soft deletes.** The decoded soft-delete count must equal the count in the
  commit, otherwise the segment is rejected rather than guessed at.
- **Read-only.** The tool never writes to the Elasticsearch data directory.
- **Default selection.** Dot-prefixed (system) indices are skipped unless
  named with `--include`. For example, `--include '.kibana*'` carries Kibana
  saved objects across.
- **Authentication.** `--user u:p` sends basic auth to the `--target`.

## Testing

`scripts/e2e.sh` starts a real Elasticsearch 8.x and loads
`scripts/fixtures.py`. The fixtures cover:
- numeric/UTF-8/base64/auto ids
- routing
- updates and deletes across 8 segments
- 2,600 soft-deletes inside a force-merged segment
- best_compression
- nested docs
- 1 MB+ docs (sliced chunks)
- 3 shards
- a default pipeline
- a closed index
- a data stream across a rollover
- templates, pipelines, scripts and ILM

The script records what ES returns, stops ES, then diffs every document twice:
once from `dump` and once after `import` into the Rust server.

```
ES_HOME=~/elasticsearch-8.19.22 RUST_ES=target/release/elasticsearch \
IMPORTER=migrate/target/release/es-lucene-import migrate/scripts/e2e.sh
```

Last run: ES 8.19.22 (Lucene 9.12.2), 6,917 docs in 11 targets, all identical
on both paths.

Builds as its own cargo workspace (like `cluster/`), with no dependency on
the engine crates: `cd migrate && cargo build --release`.
