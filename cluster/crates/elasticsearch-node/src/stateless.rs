//! Stateless object store (ES `x-pack/plugin/stateless`: `ObjectStoreService`,
//! `TranslogReplicator`, stateless commits).
//!
//! On a `stateless.enabled` cluster the index node makes every write
//! durable in the object store before acknowledging it, and copies recover
//! from the object store instead of from another node:
//!
//! * **Translog**: the primary's replication pump (the one place every
//!   operation passes through exactly once, in `_seq_no` order) uploads each
//!   batch of operations as an immutable blob
//!   `indices/<index uuid>/<shard>/translog/<first>-<last>-<term>` before it
//!   ships the batch to the replicas, so `replicate_through` (which every
//!   write waits on before its response) returns only once the write is in
//!   the object store (ES acknowledges after the translog upload).
//! * **Commits**: once `stateless.translog.flush_size` bytes (or
//!   [`COMMIT_AFTER_BLOBS`] blobs) of translog accumulated, and when a
//!   primary starts, the primary uploads a commit:
//!   the live documents of a point-in-time snapshot of the shard
//!   (`commits/<generation>/batch-<n>`) and `commit.json` naming it, then
//!   deletes the translog blobs and commit it covers.
//! * **Recovery**: a primary allocated to an index node (the old one died)
//!   and a search copy both replay the latest commit and the translog blobs
//!   above it through the regular recovery target; a search copy then
//!   catches up from the primary's translog only from where the object
//!   store ended.
//!
//! Settings and defaults are ES's: `stateless.object_store.{type,bucket,
//! client,base_path}` (`fs` resolves `bucket/base_path` against
//! `path.repo`, as an fs repository `location`; `s3` uses the named S3
//! client) and `stateless.translog.flush_size` (16mb).
//!
//! Deliberate differences from ES: commits hold documents (the recovery
//! snapshot format) rather than Lucene segment files; the translog is
//! uploaded per replication batch as soon as the batch is taken, so
//! `stateless.translog.flush.interval` (ES batches uploads every 200ms) is
//! accepted but not used; search copies still receive live operations by
//! replication (they do not refresh from uploaded commits).

use crate::replication::ops::ReplicaOp;
use crate::replication::recovery::{RecoveryOptions, RecoverySource, RecoveryStartInfo};
use bytes::Bytes;
use elasticsearch_engine::index::IndexService;
use elasticsearch_snapshots::blobstore::BlobContainer;
use elasticsearch_snapshots::repository::settings::build_store;
use elasticsearch_snapshots::{RepositoryContext, RepositoryMetadata};
use serde::{Deserialize, Serialize};
use serde::de::DeserializeSeed;
use elasticsearch_engine::translog::AdmittedBatch;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

pub const ENABLED: &str = "stateless.enabled";
pub const TYPE: &str = "stateless.object_store.type";
pub const BUCKET: &str = "stateless.object_store.bucket";
pub const CLIENT: &str = "stateless.object_store.client";
pub const BASE_PATH: &str = "stateless.object_store.base_path";
pub const FLUSH_SIZE: &str = "stateless.translog.flush_size";
/// A commit also follows this many translog blobs (per-batch uploads of
/// small writes would otherwise leave recovery replaying a blob per write).
pub const COMMIT_AFTER_BLOBS: u64 = 1024;

/// A search copy's object store recovery could not continue from the
/// primary's translog (trimmed): it falls back to a full peer recovery.
pub const TAIL_UNAVAILABLE: &str = "the primary's translog no longer covers the object store";

const COMMIT: &str = "commit.json";
const TRANSLOG: &str = "translog";
const COMMITS: &str = "commits";

fn setting(settings: &BTreeMap<String, Value>, key: &str) -> Option<String> {
    match settings.get(key)? {
        Value::String(s) => Some(s.clone()),
        Value::Null => None,
        v => Some(v.to_string()),
    }
}

/// The repository settings ES derives for the object store type
/// (`ObjectStoreType.createRepositorySettings`).
pub fn repository_metadata(settings: &BTreeMap<String, Value>) -> Result<RepositoryMetadata, String> {
    let kind = setting(settings, TYPE).unwrap_or_else(|| "fs".into()).to_ascii_lowercase();
    let bucket = setting(settings, BUCKET).filter(|b| !b.is_empty()).ok_or_else(|| format!("setting [{BUCKET}] must be set for an object store of type [{kind}]"))?;
    let base = setting(settings, BASE_PATH).filter(|b| !b.is_empty());
    let repo = match kind.as_str() {
        "fs" => json!({"location": match &base { Some(b) => format!("{bucket}/{b}"), None => bucket }}),
        "s3" => {
            let mut s = json!({"bucket": bucket});
            if let Some(c) = setting(settings, CLIENT) {
                s["client"] = json!(c);
            }
            if let Some(b) = base {
                s["base_path"] = json!(b);
            }
            s
        }
        other => return Err(format!("unsupported object store type [{other}]")),
    };
    Ok(RepositoryMetadata::new("stateless_object_store", kind, repo))
}

/// ES byte-size value (`16mb`, `512kb`, `1gb`, bytes).
pub fn parse_bytes(v: &str) -> Option<u64> {
    let v = v.trim().to_ascii_lowercase();
    let (num, mult) = [("tb", 1u64 << 40), ("gb", 1 << 30), ("mb", 1 << 20), ("kb", 1 << 10), ("b", 1)]
        .iter()
        .find_map(|(s, m)| v.strip_suffix(s).map(|n| (n.trim().to_string(), *m)))
        .unwrap_or((v.clone(), 1));
    num.parse::<f64>().ok().map(|n| (n * mult as f64) as u64)
}

/// The node's object store (`None` on a regular node).
pub struct ObjectStore {
    root: BlobContainer,
    flush_size: u64,
}

impl ObjectStore {
    pub async fn open(settings: &BTreeMap<String, Value>, ctx: &RepositoryContext) -> Result<Option<Arc<ObjectStore>>, String> {
        if !matches!(settings.get(ENABLED), Some(Value::Bool(true))) && setting(settings, ENABLED).as_deref() != Some("true") {
            return Ok(None);
        }
        let meta = repository_metadata(settings)?;
        let (store, _) = build_store(&meta, ctx).await.map_err(|e| format!("failed to open the stateless object store: {}", e.reason))?;
        let flush_size = setting(settings, FLUSH_SIZE).and_then(|v| parse_bytes(&v)).unwrap_or(16 << 20);
        tracing::info!("stateless object store: {}", store.describe());
        Ok(Some(Arc::new(ObjectStore { root: BlobContainer::new(store, ""), flush_size })))
    }

    /// The object store directory of one shard.
    pub fn shard(&self, index_uuid: &str, shard: u32) -> Arc<ShardObjectStore> {
        Arc::new(ShardObjectStore {
            dir: self.root.child("indices").child(index_uuid).child(&shard.to_string()),
            flush_size: self.flush_size,
            since_commit: AtomicU64::new(0),
            blobs_since_commit: AtomicU64::new(0),
            committing: AtomicBool::new(false),
            commit_lock: tokio::sync::Mutex::new(()),
        })
    }
}

/// `commit.json`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitManifest {
    pub generation: u64,
    pub primary_term: u64,
    /// Every operation `<=` this is reflected in the commit.
    pub max_seq_no: i64,
    pub batches: u32,
    /// The recovery snapshot's header (settings, mapping, document count).
    pub info: RecoveryStartInfo,
}

/// A translog blob's name: `<first>-<last>-<term>` (zero-padded so names
/// sort by first seq_no).
pub fn translog_blob_name(first: i64, last: i64, term: u64) -> String {
    format!("{:020}-{:020}-{:010}", first.max(0), last.max(0), term)
}

pub fn parse_translog_blob_name(name: &str) -> Option<(i64, i64, u64)> {
    let mut it = name.split('-');
    let first = it.next()?.parse().ok()?;
    let last = it.next()?.parse().ok()?;
    let term = it.next()?.parse().ok()?;
    it.next().is_none().then_some((first, last, term))
}

/// What a recovery replays: the latest commit (if any) and the translog
/// blobs above it, in seq_no order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecoveryPlan {
    pub commit: Option<CommitManifest>,
    pub translog: Vec<String>,
    /// Highest primary term among the commit and the translog blobs.
    pub max_term: u64,
}

impl RecoveryPlan {
    /// Choose the blobs: everything above the commit's `max_seq_no`.
    pub fn from_listing(commit: Option<CommitManifest>, names: impl IntoIterator<Item = String>) -> RecoveryPlan {
        let floor = commit.as_ref().map_or(-1, |c| c.max_seq_no);
        let mut blobs: Vec<(i64, i64, u64, String)> = names.into_iter().filter_map(|n| parse_translog_blob_name(&n).map(|(f, l, t)| (f, l, t, n))).filter(|(_, l, _, _)| *l > floor).collect();
        blobs.sort();
        let max_term = blobs.iter().map(|b| b.2).chain(commit.iter().map(|c| c.primary_term)).max().unwrap_or(0);
        RecoveryPlan { commit, translog: blobs.into_iter().map(|b| b.3).collect(), max_term }
    }

    /// Highest seq_no the plan reaches (`-1` for an empty store).
    pub fn max_seq_no(&self) -> i64 {
        let from_blobs = self.translog.iter().filter_map(|n| parse_translog_blob_name(n)).map(|(_, l, _)| l).max().unwrap_or(-1);
        from_blobs.max(self.commit.as_ref().map_or(-1, |c| c.max_seq_no))
    }

    pub fn is_empty(&self) -> bool {
        self.commit.is_none() && self.translog.is_empty()
    }
}

/// One shard's directory in the object store.
pub struct ShardObjectStore {
    dir: BlobContainer,
    flush_size: u64,
    since_commit: AtomicU64,
    blobs_since_commit: AtomicU64,
    committing: AtomicBool,
    commit_lock: tokio::sync::Mutex<()>,
}

fn err(e: elasticsearch_snapshots::SnapshotError) -> String {
    e.reason
}

impl ShardObjectStore {
    /// Upload one batch of operations (ES translog upload).
    pub async fn append(&self, ops: &[ReplicaOp], term: u64) -> Result<(), String> {
        let (Some(first), Some(last)) = (ops.iter().map(|o| o.seq_no()).min(), ops.iter().map(|o| o.seq_no()).max()) else { return Ok(()) };
        let body = serde_json::to_vec(ops).map_err(|e| e.to_string())?;
        let len = body.len() as u64;
        self.dir.child(TRANSLOG).write_blob(&translog_blob_name(first as i64, last as i64, term), Bytes::from(body), false).await.map_err(err)?;
        self.since_commit.fetch_add(len, Ordering::Relaxed);
        self.blobs_since_commit.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Enough translog for a new commit (`stateless.translog.flush_size`,
    /// or [`COMMIT_AFTER_BLOBS`] blobs).
    pub fn wants_commit(&self) -> bool {
        (self.since_commit.load(Ordering::Relaxed) >= self.flush_size || self.blobs_since_commit.load(Ordering::Relaxed) >= COMMIT_AFTER_BLOBS) && !self.committing.load(Ordering::Relaxed)
    }

    /// Upload a commit of `svc`'s copy of `shard`: a point-in-time snapshot
    /// of its live documents, then `commit.json`, then delete what it
    /// supersedes.
    pub async fn commit(&self, svc: Arc<IndexService>, shard: u32, term: u64, gate: Arc<tokio::sync::RwLock<()>>) -> Result<CommitManifest, String> {
        let _g = self.commit_lock.lock().await;
        self.committing.store(true, Ordering::Relaxed);
        let r = self.commit_inner(svc, shard, term, gate).await;
        self.committing.store(false, Ordering::Relaxed);
        r
    }

    async fn commit_inner(&self, svc: Arc<IndexService>, shard: u32, term: u64, gate: Arc<tokio::sync::RwLock<()>>) -> Result<CommitManifest, String> {
        let before = self.since_commit.load(Ordering::Relaxed);
        let blobs_before = self.blobs_since_commit.load(Ordering::Relaxed);
        let previous = self.latest_commit().await?;
        let generation = previous.as_ref().map_or(1, |c| c.generation + 1);
        let gen_dir = self.dir.child(COMMITS).child(&format!("{generation:010}"));
        // Capture the reader and its checkpoint under the same primary gate
        // as peer recovery. Later uploads read the retained immutable snapshot.
        let permit = gate.write_owned().await;
        let mut source = tokio::task::spawn_blocking(move || {
            let result = RecoverySource::start(svc, shard, term, RecoveryOptions::default());
            drop(permit);
            result
        }).await.map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        let mut info = source.info().clone();
        info.shard = shard;
        // Do not collect an entire shard's documents in memory before upload.
        let mut batches = 0u32;
        loop {
            let (next, batch) = tokio::task::spawn_blocking(move || {
                let batch = source.next_snapshot_batch();
                (source, batch)
            }).await.map_err(|e| e.to_string())?;
            source = next;
            let Some(batch) = batch.map_err(|e| e.to_string())? else { break };
            gen_dir.write_blob(&format!("batch-{batches:06}"), Bytes::from(serde_json::to_vec(&batch).map_err(|e| e.to_string())?), false).await.map_err(err)?;
            batches += 1;
        }
        let manifest = CommitManifest { generation, primary_term: term, max_seq_no: info.snapshot_max_seq_no, batches, info };
        self.dir.write_blob_atomic(COMMIT, Bytes::from(serde_json::to_vec(&manifest).map_err(|e| e.to_string())?), false).await.map_err(err)?;
        self.since_commit.fetch_sub(before.min(self.since_commit.load(Ordering::Relaxed)), Ordering::Relaxed);
        self.blobs_since_commit.fetch_sub(blobs_before.min(self.blobs_since_commit.load(Ordering::Relaxed)), Ordering::Relaxed);
        // Garbage: the translog the commit covers and the previous commit.
        let covered: Vec<String> = self.dir.child(TRANSLOG).list_blobs().await.map_err(err)?.into_keys().filter(|n| parse_translog_blob_name(n).is_some_and(|(_, l, _)| l <= manifest.max_seq_no)).collect();
        if !covered.is_empty() {
            self.dir.child(TRANSLOG).delete_blobs_ignoring_if_not_exists(&covered).await.map_err(err)?;
        }
        if let Some(p) = previous {
            let old = self.dir.child(COMMITS).child(&format!("{:010}", p.generation));
            let names: Vec<String> = old.list_blobs().await.map_err(err)?.into_keys().collect();
            old.delete_blobs_ignoring_if_not_exists(&names).await.map_err(err)?;
        }
        Ok(manifest)
    }

    pub async fn latest_commit(&self) -> Result<Option<CommitManifest>, String> {
        if !self.dir.blob_exists(COMMIT).await.map_err(err)? {
            return Ok(None);
        }
        let b = self.dir.read_blob(COMMIT).await.map_err(err)?;
        serde_json::from_slice(&b).map(Some).map_err(|e| format!("corrupt [{COMMIT}]: {e}"))
    }

    pub async fn recovery_plan(&self) -> Result<RecoveryPlan, String> {
        let commit = self.latest_commit().await?;
        let names = self.dir.child(TRANSLOG).list_blobs().await.map_err(err)?.into_keys();
        Ok(RecoveryPlan::from_listing(commit, names))
    }

    pub async fn commit_batch(&self, commit: &CommitManifest, i: u32) -> Result<AdmittedBatch<ReplicaOp>, String> {
        let b = self.dir.child(COMMITS).child(&format!("{:010}", commit.generation)).read_blob_admitted(&format!("batch-{i:06}"), blob_read_admission()).await.map_err(err)?;
        decode_operations(&b, elasticsearch_engine::memory::breakers())
    }

    pub async fn translog_batch(&self, name: &str) -> Result<AdmittedBatch<ReplicaOp>, String> {
        let b = self.dir.child(TRANSLOG).read_blob_admitted(name, blob_read_admission()).await.map_err(err)?;
        decode_operations(&b, elasticsearch_engine::memory::breakers())
    }
}

fn blob_read_admission() -> elasticsearch_snapshots::blobstore::ReadAdmission {
    std::sync::Arc::new(|bytes| {
        elasticsearch_engine::memory::breakers().reserve_request(bytes as u64, "stateless encoded blob")
            .map(|memory| Box::new(memory) as Box<dyn Send + Sync>)
            .map_err(|e| elasticsearch_snapshots::error::SnapshotError::new("circuit_breaking_exception", e.status, e.reason))
    })
}

/// Typed operation ownership is separate from the blob store's encoded byte
/// IO. JSON may unescape into an internal temporary buffer; reserve its maximum
/// geometric scratch before invoking the parser, then admit each owned field
/// and operation slot through the same seeds as transport decoding.
fn decode_operations(bytes: &[u8], breakers: &'static elasticsearch_engine::memory::Breakers) -> Result<AdmittedBatch<ReplicaOp>, String> {
    let scratch_bytes = (bytes.len() as u64).checked_mul(4).and_then(|n| n.checked_add(256))
        .ok_or_else(|| "object-store JSON scratch size overflow".to_string())?;
    let _scratch = breakers.reserve_request(scratch_bytes, "object-store operation JSON scratch").map_err(|e| e.to_string())?;
    let mut decoder = serde_json::Deserializer::from_slice(bytes);
    let result = crate::replication::decode::Operations(breakers).deserialize(&mut decoder).map_err(|e| e.to_string())?;
    decoder.end().map_err(|e| e.to_string())?;
    Ok(result)
}

/// The recovery header of a replay of `plan` (the commit's, or an empty
/// snapshot when there is no commit yet), under the plan's highest term.
pub fn plan_info(plan: &RecoveryPlan, svc: &IndexService, shard: u32) -> Result<RecoveryStartInfo,String> {
    let mut info = match &plan.commit {
        Some(c) => c.info.clone(),
        None => RecoveryStartInfo::try_new(&svc.name,&svc.uuid,shard,1,"{}","{}",0,-1,0,0,0).map_err(|e|e.to_string())?,
    };
    info.shard = shard;
    info.primary_term = info.primary_term.max(plan.max_term).max(1);
    Ok(info)
}

/// Batches of a replay of `plan` for `run_recovery_target`: the commit's
/// documents (snapshot phase), then each translog blob, then done. Each op
/// keeps the term that wrote it (every copy stores an op's own
/// `_primary_term`); the recovery fences on its own term (`_term`, the
/// recovery header's), not the ops'.
pub fn plan_puller(os: Arc<ShardObjectStore>, plan: RecoveryPlan, _term: u64) -> impl FnMut() -> futures::future::BoxFuture<'static, Result<crate::shards::RecoveryBatch, String>> {
    use crate::shards::RecoveryBatch;
    let cursor = Arc::new(tokio::sync::Mutex::new((0u32, 0usize)));
    let plan = Arc::new(plan);
    move || {
        let (os, plan, cursor) = (os.clone(), plan.clone(), cursor.clone());
        Box::pin(async move {
            let mut c = cursor.lock().await;
            let batch = |snapshot: bool, ops: AdmittedBatch<ReplicaOp>, phase_done: bool| -> Result<RecoveryBatch, String> {
                let memory = crate::replication::decode::DecodeMemory::reserve_texts(elasticsearch_engine::memory::breakers(), &[]).map_err(|e| e.to_string())?;
                Ok(RecoveryBatch { snapshot, ops, phase_done, mapping_json: String::new(), global_checkpoint: -1, mapping_version: 0, memory })
            };
            if let Some(commit) = &plan.commit {
                if c.0 < commit.batches {
                    let ops = os.commit_batch(commit, c.0).await?;
                    let result = batch(true, ops, false)?;
                    c.0 += 1;
                    return Ok(result);
                }
            }
            if c.1 < plan.translog.len() {
                let ops = os.translog_batch(&plan.translog[c.1]).await?;
                let result = batch(false, ops, false)?;
                c.1 += 1;
                return Ok(result);
            }
            batch(false, AdmittedBatch::new(elasticsearch_engine::memory::breakers()).map_err(|e| e.to_string())?, true)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_store_decode_retains_owned_batch_and_rejects_before_large_copy() {
        use elasticsearch_engine::memory::{Breakers, Kind};
        let bytes = serde_json::to_vec(&vec![ReplicaOp::NoOp { seq_no: 7, primary_term: 1, reason: "x".repeat(4096) }]).unwrap();
        // Private allocation tests exclude unrelated process RSS; child/parent limits remain finite.
        let tiny = Breakers::leak_for_tests(1024);
        tiny.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        assert!(decode_operations(&bytes, tiny).is_err());
        assert_eq!(tiny.get(Kind::Request).used(), 0);
        let roomy = Breakers::leak_for_tests(1 << 20);
        roomy.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        let batch = decode_operations(&bytes, roomy).unwrap();
        drop(bytes);
        assert_eq!(batch[0].seq_no(), 7);
        assert!(roomy.get(Kind::Request).used() >= 4096);
        drop(batch);
        assert_eq!(roomy.get(Kind::Request).used(), 0);
        assert!(decode_operations(b"[] trailing", roomy).is_err());
        assert_eq!(roomy.get(Kind::Request).used(), 0);
    }

    fn s(pairs: &[(&str, &str)]) -> BTreeMap<String, Value> {
        pairs.iter().map(|(k, v)| (k.to_string(), json!(v))).collect()
    }

    #[test]
    fn repository_settings_follow_es() {
        let m = repository_metadata(&s(&[(BUCKET, "stateless"), (BASE_PATH, "base_path")])).unwrap();
        assert_eq!((m.repo_type.as_str(), m.setting("location")), ("fs", Some("stateless/base_path".into())));
        let m = repository_metadata(&s(&[(TYPE, "fs"), (BUCKET, "b")])).unwrap();
        assert_eq!(m.setting("location"), Some("b".into()));
        let m = repository_metadata(&s(&[(TYPE, "s3"), (BUCKET, "b"), (CLIENT, "c"), (BASE_PATH, "p")])).unwrap();
        assert_eq!((m.setting("bucket"), m.setting("client"), m.setting("base_path")), (Some("b".into()), Some("c".into()), Some("p".into())));
        assert!(repository_metadata(&s(&[])).unwrap_err().contains("stateless.object_store.bucket"));
        assert!(repository_metadata(&s(&[(TYPE, "gcs"), (BUCKET, "b")])).is_err());
        assert_eq!(parse_bytes("16mb"), Some(16 << 20));
        assert_eq!(parse_bytes("512"), Some(512));
    }

    #[test]
    fn translog_names_sort_and_parse() {
        let n = translog_blob_name(7, 12, 3);
        assert_eq!(parse_translog_blob_name(&n), Some((7, 12, 3)));
        assert!(translog_blob_name(9, 10, 1) < translog_blob_name(11, 11, 1));
        assert_eq!(parse_translog_blob_name("junk"), None);
    }

    fn info() -> RecoveryStartInfo {
        RecoveryStartInfo::try_new("i","u",0,2,"{}","{}",6,5,3,0,0).unwrap()
    }

    #[test]
    fn recovery_plan_replays_what_the_commit_does_not_cover() {
        let commit = CommitManifest { generation: 3, primary_term: 2, max_seq_no: 5, batches: 1, info: info() };
        let names = vec![translog_blob_name(0, 5, 1), translog_blob_name(10, 12, 3), translog_blob_name(4, 9, 2), "other".to_string()];
        let p = RecoveryPlan::from_listing(Some(commit.clone()), names.clone());
        assert_eq!(p.translog, vec![translog_blob_name(4, 9, 2), translog_blob_name(10, 12, 3)], "covered blobs dropped, rest in seq order");
        assert_eq!((p.max_term, p.max_seq_no()), (3, 12));
        let p = RecoveryPlan::from_listing(None, names);
        assert_eq!(p.translog.len(), 3);
        assert!(RecoveryPlan::from_listing(None, Vec::new()).is_empty());
        assert_eq!(RecoveryPlan::default().max_seq_no(), -1);
    }

    /// Commit + translog round trip on an fs object store, against a real
    /// engine copy: the commit holds the live documents, `commit.json` names
    /// it, covered translog blobs are deleted.
    #[tokio::test]
    async fn commit_and_translog_round_trip() {
        use elasticsearch_engine::index::WriteReq;
        use elasticsearch_engine::shard::WriteCond;
        let tmp = tempfile::tempdir().unwrap();
        let ctx = RepositoryContext { path_repo: vec![tmp.path().join("repo")], ..Default::default() };
        std::fs::create_dir_all(tmp.path().join("repo")).unwrap();
        let os = ObjectStore::open(&s(&[(ENABLED, "true"), (BUCKET, "stateless"), (BASE_PATH, "bp"), (FLUSH_SIZE, "1b")]), &ctx).await.unwrap().unwrap();
        assert!(ObjectStore::open(&s(&[(BUCKET, "x")]), &ctx).await.unwrap().is_none(), "regular nodes have no object store");
        let engine = elasticsearch_engine::node::Node::open(&tmp.path().join("e")).unwrap();
        engine.create_index("i", &json!({})).unwrap();
        let svc = engine.get_index("i").unwrap();
        for id in ["a", "b", "c"] {
            let r = svc.write(vec![WriteReq::Index { id: Some(id.into()), source: r#"{"f":1}"#.into(), routing: None, cond: WriteCond::default() }], false);
            assert!(r[0].is_ok());
        }
        let shard = os.shard("uuid-i", 0);
        let op = |seq: i64| ReplicaOp::Index { id: format!("d{seq}"), source_json: "{}".into(), routing: None, seq_no: seq, version: 1, primary_term: 1 };
        shard.append(&[op(0), op(1)], 1).await.unwrap();
        shard.append(&[op(2)], 1).await.unwrap();
        shard.append(&[op(3), op(4)], 1).await.unwrap();
        assert!(shard.wants_commit());
        let plan = shard.recovery_plan().await.unwrap();
        assert_eq!((plan.commit.is_none(), plan.translog.len(), plan.max_seq_no()), (true, 3, 4));
        assert_eq!(shard.translog_batch(&plan.translog[1]).await.unwrap(), vec![op(2)]);
        let gate = Arc::new(tokio::sync::RwLock::new(()));
        let writer = gate.read().await;
        let commit = shard.commit(svc.clone(), 0, 1, gate.clone());
        tokio::pin!(commit);
        assert!(tokio::time::timeout(std::time::Duration::from_millis(20), &mut commit).await.is_err(), "snapshot must wait for active primary writes");
        drop(writer);
        let c = commit.await.unwrap();
        assert_eq!((c.generation, c.info.snapshot_docs, c.max_seq_no), (1, 3, 2));
        let docs: usize = { let mut n = 0; for i in 0..c.batches { n += shard.commit_batch(&c, i).await.unwrap().len(); } n };
        assert_eq!(docs, 3);
        let plan = shard.recovery_plan().await.unwrap();
        assert_eq!(plan.commit.as_ref().map(|c| c.generation), Some(1));
        assert_eq!(plan.translog, vec![translog_blob_name(3, 4, 1)], "blobs at or below the commit are gone");
        let c2 = shard.commit(svc, 0, 1, Arc::new(tokio::sync::RwLock::new(()))).await.unwrap();
        assert_eq!(c2.generation, 2);
        assert!(!tmp.path().join("repo/stateless/bp/indices/uuid-i/0/commits/0000000001/batch-000000").exists(), "previous commit deleted");
    }
}
