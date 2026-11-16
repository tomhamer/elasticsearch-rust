# Migrating from Elasticsearch 9.5.5 with remote reindex

The Rust server implements `POST /_reindex` with `source.remote`. It reads
documents from a running Elasticsearch HTTP/HTTPS endpoint using search
and scroll, then indexes them into the Rust cluster. The source cluster
reads its own disk formats; no offline Lucene importer is needed.

This is a document-copy operation. It does not copy cluster configuration
or provide continuous replication or an automated cutover. Elasticsearch
transport interoperability (including CCR with Elasticsearch) is not
implemented.

## Scope

The workflow below targets an ordinary index with retrievable `_source`.
The implementation includes remote authentication, TLS, host allowlisting,
asynchronous tasks, throttling, and writes routed to destination shards.

Reindex copies documents from their `_source`, so, as in Elasticsearch
itself, an index whose mapping disables `_source` (`"_source":
{"enabled": false}`) cannot be migrated this way. Neither can a request
that does not fetch it (`"_source": false`). As in Elasticsearch, the
first hit without `_source` fails the reindex with
`illegal_argument_exception` "[index][id] didn't store _source"; no
document is ever written as an empty object. Batches copied before
that hit stay in the destination. Rebuild such indices from their
original data instead.

What does migrate:

- Synthetic source (including logsdb and time series indices) is copied
  as Elasticsearch returns it. It is reconstructed content and may differ
  from the original submitted JSON, e.g. in field order, array order and
  number formatting.
- `_source` includes/excludes in the source mapping, or a `_source`
  filter in the request, copy only the fields that remain. A document
  left with no fields is copied as `{}`, as Elasticsearch does.

Data streams, system indices and feature-owned state need their own
migration plan.

## 1. Prepare the destination and source connection

Review and recreate the required mappings, analysis settings, shard/replica
settings, templates, aliases, pipelines and lifecycle policies through
their APIs. Remote reindex does not transfer them. Create an empty
destination index explicitly so that dynamic mapping does not determine
the migrated schema accidentally. Keep default/final ingest pipelines
disabled while copying documents that the source has already processed;
apply the intended pipeline settings after the copy.

Allow the source host on the Rust nodes that may execute the job:

```yaml
reindex.remote.whitelist: ["es-source.example:9200"]
# For a source using a private CA, install its CA and configure:
reindex.ssl.certificate_authorities: ["/etc/elasticsearch/certs/source-ca.crt"]
```

Use source credentials with `read` access to the selected index, permission
to search and scroll, and the cluster permission needed for the initial
`GET /` version probe. Requests to the Rust destination also require its
own credentials and permissions. The source host and credentials below are
placeholders.

## 2. Establish a consistent copy boundary

For a straightforward cutover, pause writes and deletes to the source
index, refresh it, and keep writes paused through copying and verification.
For an online backfill, separately arrange capture and reconciliation of
writes and deletes. Reindex is a point-in-time copy, not a change stream;
rerunning it alone does not remove documents deleted from the source.

## 3. Copy the documents

Send this request to the **Rust destination**, once for each source index:

```http
POST /_reindex?wait_for_completion=false&refresh=true
{
  "source": {
    "remote": {
      "host": "https://es-source.example:9200",
      "username": "migration-reader",
      "password": "REPLACE_WITH_SOURCE_PASSWORD"
    },
    "index": "source-index",
    "_source": {"exclude_vectors": false}
  },
  "dest": {
    "index": "destination-index",
    "pipeline": "_none"
  }
}
```

`exclude_vectors: false` explicitly requests vector values from source
indices that omit them from default source responses. IDs and returned
routing values are preserved unless the request overrides them. Remote
reindex accepts one automatic slice; `slices > 1` is rejected.

The response returns a task ID. Poll `GET /_tasks/<task-id>` until
`completed: true`. Check the top-level `error`, then `response.failures`,
`response.timed_out`, `response.version_conflicts` and the created/updated
counts. A failed copy can leave documents in the destination; there is no
durable remote-scroll resume. Reconcile partial work before retrying. Bulk
rejection retries are distinct from resuming a failed job.

## 4. Verify and cut over

Compare document counts, IDs, routing, representative document contents,
vector fields and application query results against the source. Equal
counts alone do not establish a correct migration. Check destination shard
health and replication, then apply the intended pipeline/lifecycle settings
and switch application endpoints and writes. Retain the source until the
migration has been accepted and a rollback plan is in place.
