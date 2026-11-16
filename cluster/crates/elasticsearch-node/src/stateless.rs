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

/// The single, overwritten manifest of commits written before commits were
/// fenced (read for recovery when no fenced manifest exists).
const COMMIT: &str = "commit.json";
const TRANSLOG: &str = "translog";
const COMMITS: &str = "commits";
const MANIFESTS: &str = "manifests";

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
            initial: tokio::sync::watch::channel(Initial::NotRequired).0,
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
    /// The commit's own directory under `commits/` (term-qualified and
    /// unique to the upload: [`commit_dir_name`]); `None`: a commit written
    /// before commits were fenced, in `commits/<generation>`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dir: Option<String>,
}

impl CommitManifest {
    /// Directory (under `commits/`) holding this commit's batches.
    pub fn dir_name(&self) -> String {
        self.dir.clone().unwrap_or_else(|| format!("{:010}", self.generation))
    }

    /// Commits order by primary term, then generation: a newer term's
    /// commit supersedes any commit of an older one, whenever that was
    /// published.
    fn order(&self) -> (u64, u64) {
        (self.primary_term, self.generation)
    }
}

/// `manifests/<term>-<generation>.json`: one immutable manifest per
/// commit, created only if absent.
pub fn manifest_blob_name(term: u64, generation: u64) -> String {
    format!("{term:010}-{generation:010}.json")
}

pub fn parse_manifest_blob_name(name: &str) -> Option<(u64, u64)> {
    let (term, generation) = name.strip_suffix(".json")?.split_once('-')?;
    Some((term.parse().ok()?, generation.parse().ok()?))
}

/// `commits/<term>-<generation>-<nonce>`: a commit's batches, never shared
/// with another upload (a deposed primary's late upload or a concurrent
/// one of the same term cannot mix batches into another commit).
pub fn commit_dir_name(term: u64, generation: u64, nonce: &str) -> String {
    format!("{term:010}-{generation:010}-{nonce}")
}

fn parse_commit_dir_name(name: &str) -> Option<(u64, u64)> {
    let mut it = name.splitn(3, '-');
    let term = it.next()?.parse().ok()?;
    let generation = it.next()?.parse().ok()?;
    it.next()?;
    Some((term, generation))
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
/// blobs above it, in seq_no order, with each seq_no's history resolved.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecoveryPlan {
    pub commit: Option<CommitManifest>,
    pub translog: Vec<String>,
    /// Per `translog` blob: only its operations below this seq_no belong to
    /// the history (`i64::MAX`: all of them); see [`RecoveryPlan::from_listing`].
    pub keep_below: Vec<i64>,
    /// Blobs (fully or partly) superseded by a newer primary term's history:
    /// `(blob, first superseded seq_no)`. Never replayed past that seq_no.
    pub superseded: Vec<(String, i64)>,
    /// Highest primary term among the commit and the translog blobs.
    pub max_term: u64,
}

impl RecoveryPlan {
    /// Choose the blobs: everything above the commit's `max_seq_no`, with
    /// ES's history semantics for a reused seq_no: a primary promoted under
    /// term `t` continues the history from where its own copy ended and
    /// trims every older term's operation above that point (ES
    /// `trimOperationsAboveSeqNo` on resync, and `MultiSnapshot` reading the
    /// newest translog first). An older term's operation at or above the
    /// seq_no where a newer term's history starts was never part of the
    /// history the newer primary acknowledged, whatever order the uploads
    /// became visible in (a deposed primary's delayed upload: production
    /// review #109), so it is never replayed. A term's history starts no
    /// later than its lowest blob, and no later than right above its commit
    /// (the commit is the term's claim, published before it acknowledges
    /// anything: [`ShardObjectStore::begin_initial_commit`]).
    pub fn from_listing(commit: Option<CommitManifest>, names: impl IntoIterator<Item = String>) -> RecoveryPlan {
        let floor = commit.as_ref().map_or(-1, |c| c.max_seq_no);
        let mut blobs: Vec<(i64, i64, u64, String)> = names.into_iter().filter_map(|n| parse_translog_blob_name(&n).map(|(f, l, t)| (f, l, t, n))).filter(|(_, l, _, _)| *l > floor).collect();
        let max_term = blobs.iter().map(|b| b.2).chain(commit.iter().map(|c| c.primary_term)).max().unwrap_or(0);
        let mut starts: BTreeMap<u64, i64> = BTreeMap::new();
        if let Some(c) = &commit {
            starts.insert(c.primary_term, c.max_seq_no + 1);
        }
        for (f, _, t, _) in &blobs {
            let s = starts.entry(*t).or_insert(*f);
            *s = (*s).min(*f);
        }
        let superseded_from = |t: u64| starts.range(t + 1..).map(|(_, s)| *s).min().unwrap_or(i64::MAX);
        // seq_no order; at the same first seq_no the newer term first, so
        // even a first-seen deduplication keeps the newer term's operation.
        blobs.sort_by(|a, b| (a.0, std::cmp::Reverse(a.2), a.1, &a.3).cmp(&(b.0, std::cmp::Reverse(b.2), b.1, &b.3)));
        let mut plan = RecoveryPlan { commit, max_term, ..RecoveryPlan::default() };
        for (f, l, t, n) in blobs {
            let bound = superseded_from(t);
            if l >= bound {
                plan.superseded.push((n.clone(), bound.max(f)));
            }
            if f >= bound {
                continue;
            }
            plan.keep_below.push(if l >= bound { bound } else { i64::MAX });
            plan.translog.push(n);
        }
        plan
    }

    /// Highest seq_no the plan reaches (`-1` for an empty store).
    pub fn max_seq_no(&self) -> i64 {
        let from_blobs = self.translog.iter().zip(&self.keep_below).filter_map(|(n, k)| parse_translog_blob_name(n).map(|(_, l, _)| l.min(k.saturating_sub(1)))).max().unwrap_or(-1);
        from_blobs.max(self.commit.as_ref().map_or(-1, |c| c.max_seq_no))
    }

    pub fn is_empty(&self) -> bool {
        self.commit.is_none() && self.translog.is_empty()
    }
}

/// A primary's first commit under its term (see
/// [`ShardObjectStore::begin_initial_commit`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Initial {
    /// Not a primary's store (recovery readers, tests).
    NotRequired,
    /// The primary's term has no commit yet: nothing is acknowledged.
    Pending,
    Ready,
    /// A newer primary term owns the shard's object store, or the first
    /// commit failed for good: this store acknowledges nothing.
    Fenced,
}

/// One shard's directory in the object store.
pub struct ShardObjectStore {
    dir: BlobContainer,
    flush_size: u64,
    since_commit: AtomicU64,
    blobs_since_commit: AtomicU64,
    committing: AtomicBool,
    commit_lock: tokio::sync::Mutex<()>,
    initial: tokio::sync::watch::Sender<Initial>,
}

/// A commit's point-in-time snapshot, captured under the primary gate and
/// not yet uploaded ([`ShardObjectStore::prepare_commit`]).
pub struct PreparedCommit {
    source: RecoverySource,
    info: RecoveryStartInfo,
    term: u64,
    translog_bytes: u64,
    translog_blobs: u64,
}

/// Create blob `name` in `container` (never overwriting: an existing blob
/// is someone else's), retried until it succeeds when `retry`, as
/// translog uploads are: a primary's first commit must not give up on a
/// transient store error, since its acknowledgements wait for it.
async fn create_blob(container: &BlobContainer, name: &str, data: Bytes, atomic: bool, retry: bool) -> Result<(), String> {
    let mut delay = std::time::Duration::from_millis(50);
    loop {
        let r = if atomic { container.write_blob_atomic(name, data.clone(), true).await } else { container.write_blob(name, data.clone(), true).await };
        match r {
            Ok(()) => return Ok(()),
            Err(e) if e.kind == "file_already_exists_exception" || !retry => return Err(e.reason),
            Err(e) => {
                tracing::warn!("object store commit upload of [{name}] failed, retrying in {delay:?}: {}", e.reason);
                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(std::time::Duration::from_secs(5));
            }
        }
    }
}

/// The outcome of an uncertain manifest create ([`ShardObjectStore::manifest_owner`]).
enum Publication {
    Ours,
    Theirs,
    Unknown(String),
}

fn err(e: elasticsearch_snapshots::SnapshotError) -> String {
    e.reason
}

impl ShardObjectStore {
    /// A primary's store: its first commit under its term is published
    /// before any of its operations is uploaded (and so acknowledged). The
    /// commit is the term's claim on the object store: from then on a
    /// recovery never chooses an older term's commit, whenever a deposed
    /// primary publishes one, and that primary sees the newer term and
    /// collects nothing (production review #93).
    pub fn begin_initial_commit(&self) {
        self.initial.send_replace(Initial::Pending);
    }

    pub fn initial_state(&self) -> Initial {
        *self.initial.borrow()
    }

    /// Wait for the primary's first commit (see [`Self::begin_initial_commit`]).
    async fn initial_ready(&self) -> Result<(), String> {
        let mut rx = self.initial.subscribe();
        let state = *rx.wait_for(|s| *s != Initial::Pending).await.map_err(|e| e.to_string())?;
        match state {
            Initial::Fenced => Err("this primary's object store commits are fenced: a newer primary term owns the shard".into()),
            _ => Ok(()),
        }
    }

    /// Upload one batch of operations (ES translog upload).
    pub async fn append(&self, ops: &[ReplicaOp], term: u64) -> Result<(), String> {
        let (Some(first), Some(last)) = (ops.iter().map(|o| o.seq_no()).min(), ops.iter().map(|o| o.seq_no()).max()) else { return Ok(()) };
        self.initial_ready().await?;
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
    /// of its live documents, then its manifest, then delete what it
    /// supersedes.
    pub async fn commit(&self, svc: Arc<IndexService>, shard: u32, term: u64, gate: Arc<tokio::sync::RwLock<()>>) -> Result<CommitManifest, String> {
        let _g = self.commit_lock.lock().await;
        self.committing.store(true, Ordering::Relaxed);
        let permit = gate.write_owned().await;
        let r = match self.prepare_commit(svc, shard, term, permit).await {
            Ok(prepared) => self.publish_commit(prepared, false).await,
            Err(e) => Err(e),
        };
        self.committing.store(false, Ordering::Relaxed);
        r
    }

    /// The primary's first commit under its term, with the gate permit the
    /// caller took before the primary accepted any write. Uploads retry
    /// until they succeed; the store then acknowledges (or, fenced or
    /// failed, never does).
    pub async fn initial_commit(&self, svc: Arc<IndexService>, shard: u32, term: u64, permit: tokio::sync::OwnedRwLockWriteGuard<()>) -> Result<CommitManifest, String> {
        let _g = self.commit_lock.lock().await;
        self.committing.store(true, Ordering::Relaxed);
        let r = match self.prepare_commit(svc, shard, term, permit).await {
            Ok(prepared) => self.publish_commit(prepared, true).await,
            Err(e) => Err(e),
        };
        self.committing.store(false, Ordering::Relaxed);
        if self.initial_state() == Initial::Pending {
            self.initial.send_replace(if r.is_ok() { Initial::Ready } else { Initial::Fenced });
        }
        r
    }

    /// Capture the commit's snapshot under `permit` (the primary gate held
    /// for writing: the same consistency as peer recovery's).
    pub async fn prepare_commit(&self, svc: Arc<IndexService>, shard: u32, term: u64, permit: tokio::sync::OwnedRwLockWriteGuard<()>) -> Result<PreparedCommit, String> {
        let translog_bytes = self.since_commit.load(Ordering::Relaxed);
        let translog_blobs = self.blobs_since_commit.load(Ordering::Relaxed);
        let source = tokio::task::spawn_blocking(move || {
            let result = RecoverySource::start(svc, shard, term, RecoveryOptions::default());
            drop(permit);
            result
        }).await.map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        let mut info = source.info().clone();
        info.shard = shard;
        Ok(PreparedCommit { source, info, term, translog_bytes, translog_blobs })
    }

    /// Every fenced manifest: `(term, generation, blob name)`.
    async fn manifests(&self) -> Result<Vec<(u64, u64, String)>, String> {
        let names = self.dir.child(MANIFESTS).list_blobs().await.map_err(err)?;
        Ok(names.into_keys().filter_map(|n| parse_manifest_blob_name(&n).map(|(t, g)| (t, g, n))).collect())
    }

    /// The commit written before commits were fenced, if any.
    async fn legacy_commit(&self) -> Result<Option<CommitManifest>, String> {
        if !self.dir.blob_exists(COMMIT).await.map_err(err)? {
            return Ok(None);
        }
        let b = self.dir.read_blob(COMMIT).await.map_err(err)?;
        serde_json::from_slice(&b).map(Some).map_err(|e| format!("corrupt [{COMMIT}]: {e}"))
    }

    /// Upload `prepared` and publish its manifest under its term, then
    /// collect what it supersedes, unless a newer term owns the store.
    ///
    /// The batches go to a directory no other upload uses; the manifest is
    /// created only if absent (never overwriting another commit); the
    /// ownership check runs after publication, so a primary deposed at any
    /// point before it deletes nothing of a newer term. A recovery reads
    /// the commit of the highest term (then generation): a deposed
    /// primary's late commit, of an older term, is never chosen over the
    /// commit the newer primary published before acknowledging a write.
    pub async fn publish_commit(&self, prepared: PreparedCommit, retry: bool) -> Result<CommitManifest, String> {
        let PreparedCommit { mut source, info, term, translog_bytes, translog_blobs } = prepared;
        let fenced = |newer: u64| format!("object store commit of primary term [{term}] fenced: primary term [{newer}] owns the shard's object store");
        let existing = self.manifests().await?;
        let legacy = self.legacy_commit().await?;
        if let Some(newer) = existing.iter().map(|m| m.0).chain(legacy.iter().map(|c| c.primary_term)).filter(|t| *t > term).max() {
            self.fence();
            return Err(fenced(newer));
        }
        let generation = existing.iter().map(|m| m.1).chain(legacy.iter().map(|c| c.generation)).max().unwrap_or(0) + 1;
        let dir_name = commit_dir_name(term, generation, &elasticsearch_snapshots::util::random_base64_uuid());
        let gen_dir = self.dir.child(COMMITS).child(&dir_name);
        // Do not collect an entire shard's documents in memory before upload.
        let mut batches = 0u32;
        loop {
            let (next, batch) = tokio::task::spawn_blocking(move || {
                let batch = source.next_snapshot_batch();
                (source, batch)
            }).await.map_err(|e| e.to_string())?;
            source = next;
            let Some(batch) = batch.map_err(|e| e.to_string())? else { break };
            let body = Bytes::from(serde_json::to_vec(&batch).map_err(|e| e.to_string())?);
            let name = format!("batch-{batches:06}");
            create_blob(&gen_dir, &name, body, false, retry).await?;
            batches += 1;
        }
        let manifest = CommitManifest { generation, primary_term: term, max_seq_no: info.snapshot_max_seq_no, batches, info, dir: Some(dir_name.clone()) };
        let body = Bytes::from(serde_json::to_vec(&manifest).map_err(|e| e.to_string())?);
        let manifests = self.dir.child(MANIFESTS);
        let name = manifest_blob_name(term, generation);
        if let Err(e) = create_blob(&manifests, &name, body, true, retry).await {
            // An error does not mean the manifest is absent: the store can
            // install it and then fail (the fs backend's directory fsync
            // after the hard link, an S3 PUT whose response is lost), and a
            // retry then finds this upload's own manifest "already exists".
            // Read it back: only another upload's manifest under this name
            // makes this commit's batches garbage (production review #108).
            match self.manifest_owner(&name, &dir_name, retry).await {
                Publication::Ours => tracing::warn!("object store commit [{name}] reported [{e}] but its manifest is installed: published"),
                Publication::Theirs => {
                    let _ = gen_dir.delete().await;
                    return Err(format!("object store commit [{name}] not published: {e}"));
                }
                Publication::Unknown(why) => {
                    // Possibly published (or still being installed): keep
                    // the batches; a later commit collects them if not.
                    return Err(format!("object store commit [{name}] not confirmed ({why}): {e}"));
                }
            }
        }
        self.since_commit.fetch_sub(translog_bytes.min(self.since_commit.load(Ordering::Relaxed)), Ordering::Relaxed);
        self.blobs_since_commit.fetch_sub(translog_blobs.min(self.blobs_since_commit.load(Ordering::Relaxed)), Ordering::Relaxed);
        // Ownership, after publication: a newer term's commit means this
        // primary was deposed; its commit is never read, and it must not
        // delete anything the newer primary relies on.
        let after = self.manifests().await?;
        if let Some(newer) = after.iter().map(|m| m.0).filter(|t| *t > term).max() {
            self.fence();
            return Err(fenced(newer));
        }
        self.collect(&manifest, &after, legacy).await?;
        Ok(manifest)
    }

    /// Whose manifest `name` is after an uncertain create: this upload's
    /// (it names `dir_name`, unique to the upload), another upload's, or
    /// unknown (absent, unreadable). Read errors are retried when `retry`.
    async fn manifest_owner(&self, name: &str, dir_name: &str, retry: bool) -> Publication {
        let mut delay = std::time::Duration::from_millis(50);
        loop {
            match self.dir.child(MANIFESTS).read_blob(name).await {
                Ok(b) => {
                    return match serde_json::from_slice::<CommitManifest>(&b) {
                        Ok(m) if m.dir.as_deref() == Some(dir_name) => Publication::Ours,
                        Ok(_) => Publication::Theirs,
                        Err(e) => Publication::Unknown(format!("unreadable manifest: {e}")),
                    };
                }
                Err(e) if e.kind == "no_such_file_exception" => return Publication::Unknown("manifest absent".into()),
                Err(e) if !retry => return Publication::Unknown(e.reason),
                Err(e) => {
                    tracing::warn!("object store manifest [{name}] read-back failed, retrying in {delay:?}: {}", e.reason);
                    tokio::time::sleep(delay).await;
                    delay = (delay * 2).min(std::time::Duration::from_secs(5));
                }
            }
        }
    }

    fn fence(&self) {
        if self.initial_state() != Initial::NotRequired {
            self.initial.send_replace(Initial::Fenced);
        }
    }

    /// Garbage the commit `manifest` supersedes: translog blobs it covers
    /// and older commits, of its own term or older terms only (a newer
    /// term's objects are never this primary's to delete).
    async fn collect(&self, manifest: &CommitManifest, manifests: &[(u64, u64, String)], legacy: Option<CommitManifest>) -> Result<(), String> {
        let term = manifest.primary_term;
        let covered: Vec<String> = self.dir.child(TRANSLOG).list_blobs().await.map_err(err)?.into_keys()
            .filter(|n| parse_translog_blob_name(n).is_some_and(|(_, l, t)| l <= manifest.max_seq_no && t <= term)).collect();
        if !covered.is_empty() {
            self.dir.child(TRANSLOG).delete_blobs_ignoring_if_not_exists(&covered).await.map_err(err)?;
        }
        let older = |t: u64, g: u64| t <= term && (t, g) < manifest.order();
        let stale: Vec<String> = manifests.iter().filter(|(t, g, _)| older(*t, *g)).map(|m| m.2.clone()).collect();
        if !stale.is_empty() {
            self.dir.child(MANIFESTS).delete_blobs_ignoring_if_not_exists(&stale).await.map_err(err)?;
        }
        for d in self.dir.child(COMMITS).children().await.map_err(err)? {
            if parse_commit_dir_name(&d).is_some_and(|(t, g)| older(t, g)) {
                self.dir.child(COMMITS).child(&d).delete().await.map_err(err)?;
            }
        }
        if let Some(l) = legacy.filter(|l| older(l.primary_term, l.generation)) {
            self.dir.delete_blobs_ignoring_if_not_exists(&[COMMIT.to_string()]).await.map_err(err)?;
            self.dir.child(COMMITS).child(&l.dir_name()).delete().await.map_err(err)?;
        }
        Ok(())
    }

    /// The commit a recovery replays: the highest term's newest one.
    pub async fn latest_commit(&self) -> Result<Option<CommitManifest>, String> {
        let mut newest: Option<CommitManifest> = None;
        let mut manifests = self.manifests().await?;
        manifests.sort();
        if let Some((_, _, name)) = manifests.last() {
            let b = self.dir.child(MANIFESTS).read_blob(name).await.map_err(err)?;
            newest = Some(serde_json::from_slice(&b).map_err(|e| format!("corrupt [{name}]: {e}"))?);
        }
        match self.legacy_commit().await? {
            Some(l) if newest.as_ref().is_none_or(|n| l.order() > n.order()) => Ok(Some(l)),
            _ => Ok(newest),
        }
    }

    pub async fn recovery_plan(&self) -> Result<RecoveryPlan, String> {
        let commit = self.latest_commit().await?;
        let names = self.dir.child(TRANSLOG).list_blobs().await.map_err(err)?.into_keys();
        let plan = RecoveryPlan::from_listing(commit, names);
        for (blob, from) in &plan.superseded {
            tracing::warn!("object store translog [{blob}]: operations from seq_no [{from}] are superseded by a newer primary term's history and are not replayed");
        }
        Ok(plan)
    }

    /// Translog blob `i` of `plan`, holding only the operations that belong
    /// to the recovered history ([`RecoveryPlan::keep_below`]). An operation
    /// of a newer term than the blob's own is corrupt history: an error.
    pub async fn plan_translog_batch(&self, plan: &RecoveryPlan, i: usize) -> Result<AdmittedBatch<ReplicaOp>, String> {
        let name = &plan.translog[i];
        let term = parse_translog_blob_name(name).map_or(u64::MAX, |(_, _, t)| t);
        let mut ops = self.translog_batch(name).await?;
        if let Some(op) = ops.iter().find(|o| o.primary_term() > term) {
            return Err(format!("object store translog [{name}] of primary term [{term}] holds operation [{}] of newer primary term [{}]", op.seq_no(), op.primary_term()));
        }
        let keep_below = plan.keep_below.get(i).copied().unwrap_or(i64::MAX);
        if keep_below != i64::MAX {
            ops.retain(|o| o.seq_no() < keep_below);
        }
        Ok(ops)
    }

    pub async fn commit_batch(&self, commit: &CommitManifest, i: u32) -> Result<AdmittedBatch<ReplicaOp>, String> {
        let b = self.dir.child(COMMITS).child(&commit.dir_name()).read_blob_admitted(&format!("batch-{i:06}"), blob_read_admission()).await.map_err(err)?;
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
                let ops = os.plan_translog_batch(&plan, c.1).await?;
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
        let commit = CommitManifest { generation: 3, primary_term: 2, max_seq_no: 5, batches: 1, info: info(), dir: None };
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
        assert!(!tmp.path().join(format!("repo/stateless/bp/indices/uuid-i/0/commits/{}/batch-000000", c.dir_name())).exists(), "previous commit deleted");
        assert!(!tmp.path().join(format!("repo/stateless/bp/indices/uuid-i/0/manifests/{}", manifest_blob_name(1, 1))).exists(), "previous manifest deleted");
    }

    /// An object store under `tmp` and an engine node of its own.
    async fn store_and_engine(tmp: &std::path::Path, engine: &str) -> (Arc<ObjectStore>, Arc<elasticsearch_engine::node::Node>) {
        let ctx = RepositoryContext { path_repo: vec![tmp.join("repo")], ..Default::default() };
        std::fs::create_dir_all(tmp.join("repo")).unwrap();
        let os = ObjectStore::open(&s(&[(ENABLED, "true"), (BUCKET, "stateless"), (BASE_PATH, "bp"), (FLUSH_SIZE, "1b")]), &ctx).await.unwrap().unwrap();
        (os, elasticsearch_engine::node::Node::open(&tmp.join(engine)).unwrap())
    }

    fn index_doc(svc: &IndexService, id: &str, source: Value) {
        use elasticsearch_engine::index::WriteReq;
        let r = svc.write(vec![WriteReq::Index { id: Some(id.into()), source: source.to_string(), routing: None, cond: elasticsearch_engine::shard::WriteCond::default() }], false);
        assert!(r[0].is_ok(), "{r:?}");
    }

    fn hits(svc: &Arc<IndexService>, query: Value) -> u64 {
        svc.refresh().unwrap();
        let body = json!({"query": query, "track_total_hits": true});
        let sreq = elasticsearch_engine::search::parse_request(&body, &std::collections::HashMap::new()).unwrap();
        let opts = elasticsearch_engine::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
        let out = elasticsearch_engine::search::execute(&[elasticsearch_engine::search::Target::new(svc.clone())], &sreq, &opts).unwrap();
        let v: Value = serde_json::from_str(&out.json).unwrap();
        v["hits"]["total"]["value"].as_u64().unwrap()
    }

    /// Recover a copy from the object store into an empty `svc` (what a new
    /// primary or a rebuilt search copy replays): the plan's commit, then
    /// its translog blobs.
    async fn replay_plan(store: &ShardObjectStore, svc: &Arc<IndexService>) -> RecoveryPlan {
        let plan = store.recovery_plan().await.unwrap();
        let copy = svc.shard(0).unwrap();
        let mut apply = |ops: AdmittedBatch<ReplicaOp>| {
            for op in ops.iter() {
                match op {
                    ReplicaOp::Index { id, source_json, routing, seq_no, version, .. } => copy.replay_source(id, source_json, routing.clone(), *seq_no, *version).unwrap(),
                    ReplicaOp::Delete { id, seq_no, version, .. } => copy.shard.replay_delete(id, *seq_no, *version).unwrap(),
                    _ => {}
                }
            }
        };
        if let Some(c) = &plan.commit {
            for i in 0..c.batches {
                apply(store.commit_batch(c, i).await.unwrap());
            }
        }
        for i in 0..plan.translog.len() {
            apply(store.plan_translog_batch(&plan, i).await.unwrap());
        }
        plan
    }

    /// Production review #92: a stateless commit of a `_source`-disabled
    /// index holds the documents' indexed data (ES `_recovery_source`), so
    /// a recovery after the commit deleted the translog blobs it covers
    /// re-indexes every field. Before, the commit held the empty stored
    /// source: the recovered copy kept ids and counts but matched nothing.
    #[tokio::test]
    async fn source_disabled_commit_recovers_indexed_fields_after_translog_collection() {
        let tmp = tempfile::tempdir().unwrap();
        let (os, engine) = store_and_engine(tmp.path(), "e").await;
        let mapping = json!({"mappings": {"_source": {"enabled": false}, "properties": {"k": {"type": "keyword"}, "n": {"type": "long"}}}});
        let svc = engine.create_index("i", &mapping).unwrap();
        // Filtered stored source too: `excludes` keeps `secret` out of `_source` only.
        let filtered = engine.create_index("f", &json!({"mappings": {"_source": {"excludes": ["secret"]}, "properties": {"secret": {"type": "keyword"}}}})).unwrap();
        for i in 0..5 {
            index_doc(&svc, &format!("d{i}"), json!({"k": format!("v{i}"), "n": i}));
            index_doc(&filtered, &format!("d{i}"), json!({"secret": format!("s{i}"), "open": i}));
        }
        for (name, svc) in [("i", &svc), ("f", &filtered)] {
            let store = os.shard(&format!("uuid-{name}"), 0);
            let op = |seq: i64| ReplicaOp::Index { id: format!("d{seq}"), source_json: "{}".into(), routing: None, seq_no: seq, version: 1, primary_term: 1 };
            store.append(&[op(0), op(1), op(2)], 1).await.unwrap();
            let c = store.commit(svc.clone(), 0, 1, Arc::new(tokio::sync::RwLock::new(()))).await.unwrap();
            assert_eq!(c.max_seq_no, 4);
            assert!(store.recovery_plan().await.unwrap().translog.is_empty(), "the covered translog blobs are deleted");
            let (_, fresh_engine) = store_and_engine(tmp.path(), &format!("fresh-{name}")).await;
            let fresh = fresh_engine.create_index(name, &if name == "i" { mapping.clone() } else { json!({"mappings": {"_source": {"excludes": ["secret"]}, "properties": {"secret": {"type": "keyword"}}}}) }).unwrap();
            replay_plan(&store, &fresh).await;
            assert_eq!(hits(&fresh, json!({"match_all": {}})), 5);
            if name == "i" {
                for q in [json!({"term": {"k": "v3"}}), json!({"range": {"n": {"gte": 2}}})] {
                    assert_eq!(hits(&fresh, q.clone()), hits(svc, q.clone()), "{q}");
                }
                assert_eq!(hits(&fresh, json!({"term": {"k": "v3"}})), 1);
            } else {
                assert_eq!(hits(&fresh, json!({"term": {"secret": "s2"}})), 1, "an excluded field is still indexed after recovery");
                let got = fresh.shard(0).unwrap().get("d2", None, true).unwrap().unwrap();
                assert_eq!(serde_json::from_str::<Value>(got.source.as_deref().unwrap()).unwrap(), json!({"open": 2}), "the stored source stays filtered");
            }
        }
    }

    fn top_id(svc: &Arc<IndexService>, body: Value) -> Option<String> {
        svc.refresh().unwrap();
        let sreq = elasticsearch_engine::search::parse_request(&body, &std::collections::HashMap::new()).unwrap();
        let opts = elasticsearch_engine::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
        let out = elasticsearch_engine::search::execute(&[elasticsearch_engine::search::Target::new(svc.clone())], &sreq, &opts).unwrap();
        let v: Value = serde_json::from_str(&out.json).unwrap();
        v["hits"]["hits"][0]["_id"].as_str().map(str::to_string)
    }

    /// `_recovery_source` never holds a second copy of indexed dense
    /// vectors, and is not kept at all when the filter dropped nothing but
    /// them (the common `_source.excludes: [<vector>]`): a rebuild takes the
    /// vectors back from the vector store, as ES rebuilds them from their
    /// vector values. ES 9's default (`index.mapping.exclude_source_vectors`)
    /// and synthetic source never get a `_recovery_source` field at all.
    #[tokio::test]
    async fn recovery_source_keeps_no_vectors_and_rebuilds_them_from_the_vector_store() {
        let tmp = tempfile::tempdir().unwrap();
        let (os, engine) = store_and_engine(tmp.path(), "e").await;
        let props = json!({"emb": {"type": "dense_vector", "dims": 3, "similarity": "l2_norm"}, "t": {"type": "keyword"}});
        // ES 9 default vector mapping, synthetic source: no `_recovery_source` field.
        for (name, body) in [("vdef", json!({"mappings": {"properties": props}})), ("vsyn", json!({"settings": {"index.mapping.source.mode": "synthetic"}, "mappings": {"properties": props}}))] {
            let svc = engine.create_index(name, &body).unwrap();
            assert_eq!(svc.lookup().recovery_source_id, None, "[{name}]");
        }
        let cases = [
            ("ve", json!({"_source": {"excludes": ["emb"]}, "properties": props}), false),
            ("vd", json!({"_source": {"enabled": false}, "properties": props}), true),
            ("vx", json!({"_source": {"excludes": ["emb", "t"]}, "properties": props}), true),
        ];
        for (name, m, keeps) in cases {
            let svc = engine.create_index(name, &json!({"mappings": m.clone()})).unwrap();
            for i in 0..5 {
                index_doc(&svc, &format!("d{i}"), json!({"t": format!("x{i}"), "emb": [i as f64, 1.0, 0.0]}));
            }
            let rid = svc.lookup().recovery_source_id.expect("filtered or disabled source");
            let got = svc.shard(0).unwrap().get("d1", None, true).unwrap().unwrap();
            let kept = got.doc.get(rid).and_then(|v| v.as_text()).map(str::to_string);
            assert_eq!(kept.is_some(), keeps, "[{name}] {kept:?}");
            if let Some(k) = &kept {
                assert!(!k.contains("emb") && k.contains("x1"), "[{name}] no vector copy: {k}");
            }
            let store = os.shard(&format!("uuid-{name}"), 0);
            store.commit(svc.clone(), 0, 1, Arc::new(tokio::sync::RwLock::new(()))).await.unwrap();
            let (_, fresh_engine) = store_and_engine(tmp.path(), &format!("fresh-{name}")).await;
            let fresh = fresh_engine.create_index(name, &json!({"mappings": m})).unwrap();
            replay_plan(&store, &fresh).await;
            let knn = json!({"knn": {"field": "emb", "query_vector": [3.1, 1.0, 0.0], "k": 1, "num_candidates": 10}, "size": 1});
            assert_eq!(top_id(&fresh, knn.clone()), Some("d3".into()), "[{name}] vectors rebuilt from the vector store");
            assert_eq!(hits(&fresh, json!({"term": {"t": "x2"}})), 1, "[{name}]");
            assert_eq!(hits(&fresh, json!({"exists": {"field": "emb"}})), 5, "[{name}]");
        }
    }

    /// Production review #93: primary A (term 1) captures a commit snapshot
    /// and stalls; B (term 2) takes over, acknowledges writes, publishes its
    /// commit and collects the translog it covers; then A resumes. A must
    /// neither replace B's commit nor delete anything of B's, and a
    /// recovery from an empty path holds every write B acknowledged.
    #[tokio::test]
    async fn a_deposed_primary_cannot_replace_or_collect_a_newer_terms_commit() {
        let tmp = tempfile::tempdir().unwrap();
        let (os, engine_a) = store_and_engine(tmp.path(), "a").await;
        let (_, engine_b) = store_and_engine(tmp.path(), "b").await;
        let mapping = json!({"mappings": {"properties": {"k": {"type": "keyword"}}}});
        let a = engine_a.create_index("i", &mapping).unwrap();
        let b = engine_b.create_index("i", &mapping).unwrap();
        let op = |id: &str, seq: i64, term: u64| ReplicaOp::Index { id: id.into(), source_json: json!({"k": id}).to_string(), routing: None, seq_no: seq, version: 1, primary_term: term };
        // Term 1: A holds a0..a2 (seq 0..2), uploaded.
        let store_a = os.shard("uuid-i", 0);
        store_a.begin_initial_commit();
        let gate_a = Arc::new(tokio::sync::RwLock::new(()));
        store_a.initial_commit(a.clone(), 0, 1, gate_a.clone().try_write_owned().unwrap()).await.unwrap();
        for (i, id) in ["a0", "a1", "a2"].iter().enumerate() {
            index_doc(&a, id, json!({"k": id}));
            store_a.append(&[op(id, i as i64, 1)], 1).await.unwrap();
        }
        // A also executed a3 (seq 3), never uploaded nor acknowledged.
        index_doc(&a, "a3", json!({"k": "a3"}));
        // A captures its next commit's snapshot (max seq 3) and stalls.
        let stalled = store_a.prepare_commit(a.clone(), 0, 1, gate_a.clone().write_owned().await).await.unwrap();
        // B recovers from the object store (a0..a2), then becomes the
        // primary of term 2: its first commit, then writes b3..b5, reusing
        // seq 3 (A's unacknowledged a3 is not in the history B continues).
        let store_b = os.shard("uuid-i", 0);
        replay_plan(&store_b, &b).await;
        store_b.begin_initial_commit();
        let gate_b = Arc::new(tokio::sync::RwLock::new(()));
        store_b.initial_commit(b.clone(), 0, 2, gate_b.clone().try_write_owned().unwrap()).await.unwrap();
        assert_eq!(store_b.initial_state(), Initial::Ready);
        for (i, id) in ["b3", "b4", "b5"].iter().enumerate() {
            index_doc(&b, id, json!({"k": id}));
            store_b.append(&[op(id, 3 + i as i64, 2)], 2).await.unwrap();
        }
        // B commits (covers b3..b5, deletes their translog blobs).
        let cb = store_b.commit(b.clone(), 0, 2, gate_b.clone()).await.unwrap();
        assert_eq!((cb.primary_term, cb.max_seq_no), (2, 5));
        // A resumes: refused, and it fences its store (it acknowledges nothing more).
        let late = store_a.publish_commit(stalled, false).await;
        assert!(late.as_ref().is_err_and(|e| e.contains("fenced")), "{late:?}");
        assert_eq!(store_a.initial_state(), Initial::Fenced);
        assert!(store_a.append(&[op("a4", 4, 1)], 1).await.is_err(), "a deposed primary uploads nothing");
        // Even a deposed commit that slips through (published before A could
        // see B's term) is never chosen over B's, nor collects B's objects.
        let reader = os.shard("uuid-i", 0);
        let plan = reader.recovery_plan().await.unwrap();
        assert_eq!(plan.commit.as_ref().map(|c| (c.primary_term, c.generation)), Some((2, cb.generation)));
        let (_, engine_c) = store_and_engine(tmp.path(), "c").await;
        let c = engine_c.create_index("i", &mapping).unwrap();
        replay_plan(&reader, &c).await;
        for id in ["a0", "a1", "a2", "b3", "b4", "b5"] {
            assert_eq!(hits(&c, json!({"term": {"k": id}})), 1, "acknowledged write [{id}] survives");
        }
        assert_eq!(hits(&c, json!({"term": {"k": "a3"}})), 0, "the deposed primary's unacknowledged write is not resurrected");
        // An older term's late manifest (as if A had published before seeing
        // term 2) does not change what a recovery reads.
        let mut forged = cb.clone();
        forged.primary_term = 1;
        forged.generation = cb.generation + 7;
        forged.dir = Some(commit_dir_name(1, forged.generation, "late"));
        reader.dir.child(MANIFESTS).write_blob_atomic(&manifest_blob_name(1, forged.generation), Bytes::from(serde_json::to_vec(&forged).unwrap()), true).await.unwrap();
        assert_eq!(reader.latest_commit().await.unwrap().map(|c| c.order()), Some((2, cb.generation)));
    }

    /// Recover `svc`'s empty copy of shard 0 from `store` through the
    /// production path: [`plan_info`], [`plan_puller`] and the regular
    /// recovery target (its replica applier and seq_no deduplication).
    async fn recover_through_target(store: &Arc<ShardObjectStore>, svc: &Arc<IndexService>) -> Result<RecoveryPlan, String> {
        let plan = store.recovery_plan().await?;
        let applier = Arc::new(crate::replication::replica::ReplicaApplier::open(svc.clone(), 0).map_err(|e| e.to_string())?);
        let info = plan_info(&plan, svc, 0)?;
        let pull = plan_puller(store.clone(), plan.clone(), info.primary_term);
        crate::shards::run_recovery_target(applier, info, Arc::new(parking_lot::Mutex::new(None)), &|_| true, pull).await?;
        Ok(plan)
    }

    fn source_of(svc: &Arc<IndexService>, id: &str) -> Option<Value> {
        let got = svc.shard(0).unwrap().get(id, None, true).unwrap()?;
        got.source.as_deref().map(|s| serde_json::from_str(s).unwrap())
    }

    /// An fs object store whose directory fsyncs of `manifests/` (the sync
    /// after the hard link that installs a manifest) and of pending manifest
    /// temp files fail once each time the matching switch is armed.
    async fn hooked_store(tmp: &std::path::Path, after_link: Arc<AtomicBool>, before_link: Arc<AtomicBool>, race: Arc<parking_lot::Mutex<Option<(std::path::PathBuf, Vec<u8>)>>>) -> Arc<ObjectStore> {
        use elasticsearch_snapshots::blobstore::fs::FsBlobStore;
        let root = tmp.join("repo/stateless/bp");
        let hook: elasticsearch_snapshots::blobstore::fs::SyncHook = Arc::new(move |p: &std::path::Path| {
            let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            // Another upload publishes its manifest while this one uploads
            // its batches (taking the generation this one chose).
            if name.contains("batch-") {
                if let Some((path, body)) = race.lock().take() {
                    std::fs::write(path, body)?;
                }
            }
            let in_manifests = p.parent().and_then(|d| d.file_name()).is_some_and(|d| d == MANIFESTS);
            if name == MANIFESTS && after_link.swap(false, Ordering::SeqCst) {
                return Err(std::io::Error::other("injected fsync failure after the manifest was linked"));
            }
            if in_manifests && name.starts_with("pending-") && before_link.swap(false, Ordering::SeqCst) {
                return Err(std::io::Error::other("injected fsync failure before the manifest was linked"));
            }
            Ok(())
        });
        let fs = FsBlobStore::open(root, false).await.unwrap().with_sync_hook(hook);
        Arc::new(ObjectStore { root: BlobContainer::new(Arc::new(fs), ""), flush_size: 1 })
    }

    fn commit_dirs(tmp: &std::path::Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(tmp.join("repo/stateless/bp/indices/uuid-i/0/commits")).map(|it| it.map(|e| e.unwrap().file_name().to_string_lossy().to_string()).collect()).unwrap_or_default();
        v.sort();
        v
    }

    /// Production review #108: the store installs a commit's manifest and
    /// then reports an error (the fs backend's directory fsync after the
    /// hard link; an S3 PUT whose response is lost). The commit is
    /// published: its batches must survive, in both retry modes (the
    /// retrying first commit then sees its own manifest "already exists"),
    /// and a recovery before any later commit reads it. Only another
    /// upload's manifest under the same name makes the batches garbage; an
    /// unconfirmed publication keeps them.
    #[tokio::test]
    async fn an_ambiguous_manifest_publication_keeps_its_committed_batches() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("repo")).unwrap();
        let after_link = Arc::new(AtomicBool::new(false));
        let before_link = Arc::new(AtomicBool::new(false));
        let race = Arc::new(parking_lot::Mutex::new(None));
        let os = hooked_store(tmp.path(), after_link.clone(), before_link.clone(), race.clone()).await;
        let (_, engine) = store_and_engine(tmp.path(), "a").await;
        let mapping = json!({"mappings": {"properties": {"k": {"type": "keyword"}}}});
        let svc = engine.create_index("i", &mapping).unwrap();
        for id in ["a", "b", "c"] {
            index_doc(&svc, id, json!({"k": id}));
        }
        let store = os.shard("uuid-i", 0);
        // Retrying mode (a primary's first commit): the first attempt
        // installs the manifest and fails, the retry finds it existing.
        store.begin_initial_commit();
        after_link.store(true, Ordering::SeqCst);
        let gate = Arc::new(tokio::sync::RwLock::new(()));
        let first = store.initial_commit(svc.clone(), 0, 1, gate.clone().try_write_owned().unwrap()).await;
        assert!(!after_link.load(Ordering::SeqCst), "the fault was injected");
        let first = first.expect("its own installed manifest means the commit is published");
        assert_eq!(store.initial_state(), Initial::Ready);
        assert_eq!(commit_dirs(tmp.path()), vec![first.dir_name()], "the published commit's batches are kept");
        // Recovery before any later commit.
        let (_, fresh) = store_and_engine(tmp.path(), "r1").await;
        let r1 = fresh.create_index("i", &mapping).unwrap();
        let plan = recover_through_target(&store, &r1).await.expect("recovery reads the published commit");
        assert_eq!(plan.commit.map(|c| c.dir), Some(first.dir.clone()));
        assert_eq!(hits(&r1, json!({"match_all": {}})), 3);
        // Non-retrying mode (a regular commit): the single attempt fails
        // after installation.
        for id in ["d", "e"] {
            index_doc(&svc, id, json!({"k": id}));
        }
        after_link.store(true, Ordering::SeqCst);
        let second = store.commit(svc.clone(), 0, 1, gate.clone()).await;
        assert!(!after_link.load(Ordering::SeqCst), "the fault was injected");
        let second = second.expect("its own installed manifest means the commit is published");
        assert_eq!(second.generation, first.generation + 1);
        assert_eq!(commit_dirs(tmp.path()), vec![second.dir_name()], "kept, and the superseded commit collected");
        let (_, fresh) = store_and_engine(tmp.path(), "r2").await;
        let r2 = fresh.create_index("i", &mapping).unwrap();
        let plan = recover_through_target(&store, &r2).await.expect("recovery reads the published commit");
        assert_eq!(plan.commit.map(|c| c.generation), Some(second.generation));
        assert_eq!(hits(&r2, json!({"match_all": {}})), 5);
        // Unconfirmed: the manifest was never installed (the temp file's
        // fsync failed). Not proven to be someone else's: batches kept, and
        // the published predecessor still recovers.
        before_link.store(true, Ordering::SeqCst);
        let unconfirmed = store.commit(svc.clone(), 0, 1, gate.clone()).await;
        assert!(!before_link.load(Ordering::SeqCst), "the fault was injected");
        assert!(unconfirmed.as_ref().is_err_and(|e| e.contains("not confirmed")), "{unconfirmed:?}");
        assert_eq!(commit_dirs(tmp.path()).len(), 2, "unconfirmed batches are not deleted");
        let (_, fresh) = store_and_engine(tmp.path(), "r3").await;
        let r3 = fresh.create_index("i", &mapping).unwrap();
        assert_eq!(recover_through_target(&store, &r3).await.unwrap().commit.map(|c| c.generation), Some(second.generation));
        assert_eq!(hits(&r3, json!({"match_all": {}})), 5);
        // Another upload's manifest won the generation: this upload's
        // batches are garbage and deleted.
        let generation = second.generation + 1;
        let mut theirs = second.clone();
        theirs.generation = generation;
        theirs.dir = Some(commit_dir_name(1, generation, "theirs"));
        let manifest_path = tmp.path().join("repo/stateless/bp/indices/uuid-i/0/manifests").join(manifest_blob_name(1, generation));
        *race.lock() = Some((manifest_path, serde_json::to_vec(&theirs).unwrap()));
        let before = commit_dirs(tmp.path());
        let lost = store.commit(svc.clone(), 0, 1, gate.clone()).await;
        assert!(race.lock().is_none(), "the race was injected");
        assert!(lost.as_ref().is_err_and(|e| e.contains("not published")), "{lost:?}");
        assert_eq!(commit_dirs(tmp.path()), before, "the losing upload's own directory is deleted");
    }

    /// Production review #109: primary A (term 1) has a translog upload in
    /// flight across failover. B recovers without it, publishes its first
    /// commit under term 2 (covering seq 10), and acknowledges its own
    /// write at seq 11. A's delayed term-1 uploads (seq 11, a different
    /// operation, and seq 12, never part of B's history) then become
    /// visible, and B stops before another commit. A recovery from empty
    /// local storage installs B's acknowledged history, never A's
    /// operation at the reused seq_no, and reports the superseded blobs.
    #[tokio::test]
    async fn recovery_prefers_the_newer_terms_history_at_a_reused_seq_no() {
        let tmp = tempfile::tempdir().unwrap();
        let (os, engine_a) = store_and_engine(tmp.path(), "a").await;
        let (_, engine_b) = store_and_engine(tmp.path(), "b").await;
        let mapping = json!({"mappings": {"properties": {"k": {"type": "keyword"}}}});
        let a = engine_a.create_index("i", &mapping).unwrap();
        let b = engine_b.create_index("i", &mapping).unwrap();
        let op = |id: &str, k: &str, seq: i64, term: u64| ReplicaOp::Index { id: id.into(), source_json: json!({"k": k}).to_string(), routing: None, seq_no: seq, version: 1, primary_term: term };
        let store_a = os.shard("uuid-i", 0);
        store_a.begin_initial_commit();
        store_a.initial_commit(a.clone(), 0, 1, Arc::new(tokio::sync::RwLock::new(())).try_write_owned().unwrap()).await.unwrap();
        let acked: Vec<ReplicaOp> = (0..=10).map(|i| op(&format!("d{i}"), "a", i, 1)).collect();
        for o in &acked {
            if let ReplicaOp::Index { id, source_json, .. } = o {
                index_doc(&a, id, serde_json::from_str(source_json).unwrap());
            }
        }
        store_a.append(&acked, 1).await.unwrap();
        // A executes x (seq 11) and ghost (seq 12); their uploads stall.
        let stale_x = op("x", "old", 11, 1);
        let stale_ghost = op("ghost", "old", 12, 1);
        // B recovers seq 0..10 and is promoted under term 2.
        let store_b = os.shard("uuid-i", 0);
        replay_plan(&store_b, &b).await;
        store_b.begin_initial_commit();
        let gate_b = Arc::new(tokio::sync::RwLock::new(()));
        let cb = store_b.initial_commit(b.clone(), 0, 2, gate_b.clone().try_write_owned().unwrap()).await.unwrap();
        assert_eq!((cb.primary_term, cb.max_seq_no), (2, 10));
        // B acknowledges a different operation at seq 11.
        index_doc(&b, "x", json!({"k": "new"}));
        store_b.append(&[op("x", "new", 11, 2)], 2).await.unwrap();
        // A's in-flight uploads complete (its readiness check passed long ago).
        store_a.append(&[stale_x], 1).await.unwrap();
        store_a.append(&[stale_ghost], 1).await.unwrap();
        // B stops before another commit. Recover from empty local storage.
        let reader = os.shard("uuid-i", 0);
        let (_, engine_c) = store_and_engine(tmp.path(), "c").await;
        let c = engine_c.create_index("i", &mapping).unwrap();
        let plan = recover_through_target(&reader, &c).await.unwrap();
        assert_eq!(source_of(&c, "x"), Some(json!({"k": "new"})), "the acknowledged new-term operation wins seq 11");
        assert_eq!(source_of(&c, "ghost"), None, "an older term's operation above the newer term's start is not history");
        assert_eq!(hits(&c, json!({"term": {"k": "old"}})), 0);
        assert_eq!(hits(&c, json!({"match_all": {}})), 12);
        // Never silent: the plan names exactly the superseded blobs.
        let mut superseded = plan.superseded.clone();
        superseded.sort();
        assert_eq!(superseded, vec![(translog_blob_name(11, 11, 1), 11), (translog_blob_name(12, 12, 1), 12)]);
        assert_eq!(plan.translog, vec![translog_blob_name(11, 11, 2)]);
        assert_eq!(plan.max_seq_no(), 11);
        // Arrival order does not matter: with the new term's blob listed
        // before or after, and a multi-op stale blob straddling the start,
        // only the operations below the newer term's start survive.
        let names = vec![translog_blob_name(9, 12, 1), translog_blob_name(11, 11, 2), translog_blob_name(11, 11, 1)];
        for order in [names.clone(), names.iter().rev().cloned().collect()] {
            let p = RecoveryPlan::from_listing(None, order);
            assert_eq!(p.translog, vec![translog_blob_name(9, 12, 1), translog_blob_name(11, 11, 2)]);
            assert_eq!(p.keep_below, vec![11, i64::MAX]);
            assert_eq!(p.max_seq_no(), 11);
        }
    }
}
