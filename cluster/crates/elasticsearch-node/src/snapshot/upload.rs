//! Shard snapshots written by the node holding the primary (ES
//! `SnapshotShardsService`), and the master's side of them.
//!
//! **Data node.** [`ACTION_UPLOAD`] starts (or joins) the snapshot of one
//! local primary: the shard is frozen ([`super::shard::prepare`], hard links
//! under `snapshot-staging/<uuid>/<index>/<shard>`), then its files are
//! uploaded incrementally straight from the frozen image through this
//! node's own instance of the repository (opened with the node's
//! `path.repo` / S3 client settings, as ES requires every node to reach the
//! repository), which writes the shard-level blobs: the `__*` data blobs,
//! `snap-{uuid}.dat` and a new `index-{gen}` built on the generation the
//! master handed over. The node then drops the shard's links. Uploads are
//! throttled per node by the repository's `max_snapshot_bytes_per_sec`
//! (ES default `40mb`, one limiter per repository instance, so per node) and
//! at most `thread_pool.snapshot.max` (ES default 10) shard snapshots run at
//! once on a node; the rest wait (`INIT`). [`ACTION_UPLOAD_STATUS`] reports a
//! shard snapshot's stage and statistics (long-polled until it finishes or
//! `wait_ms` passes) and aborts it on request. A job is keyed by snapshot
//! uuid, index and shard, so a request repeated by a new master after a
//! failover joins the running (or finished) upload instead of starting
//! another, but only when it asks for the same repository index id and
//! base shard generation (a new master resolves its own id for an index
//! new to the repository); otherwise, and for a failed one, the old upload
//! is stopped and a new one started. The master accepts a reported success
//! only after checking the shard's `snap-{uuid}.dat`, shard index and data
//! blobs are in the repository where it finalizes
//! (`Repository::verify_shard_snapshot`). Jobs are dropped when the master
//! releases the snapshot ([`super::shard::ACTION_RELEASE`]); nothing of
//! them survives a restart (the staging directory is wiped at startup).
//!
//! **Master** ([`Uploader`], the snapshots service's
//! [`RemoteShardSnapshots`]): under the repository's shard lock it hands
//! the node the current shard generation, polls the node's status into the
//! shard's `_status` entry and returns the new generation for the finalize
//! (the root `index-N` is still written only by the master, with its
//! compare-and-set). A node that left the cluster, keeps failing to answer,
//! or no longer knows the upload (it restarted) fails the shard: the
//! snapshot ends `PARTIAL`, as in ES. An abort (snapshot `DELETE`) is passed
//! on to the node, which stops at the next chunk.
//!
//! **Mixed versions.** A node that does not advertise
//! [`Capability::SnapshotShardUploadV1`] (or does not know the action)
//! cannot upload: as before, the master pulls the frozen files into its own
//! staging directory and uploads them. It first checks that the copy fits:
//! the files plus the bytes other recoveries and pulls in flight still
//! write must leave the high disk watermark's free space on the master
//! ([`check_pull_space`], `file_recovery::reserve_disk`); otherwise that
//! shard fails with the reason rather than filling the master's disk
//! (which would put a read-only block on its indices).

use super::common::{self, repo_context};
use super::shard::{self, PrepareReq, PrepareResp, ReadReq, CHUNK};
use crate::front::SharedText;
use crate::internal_capabilities::{self, Capability};
use crate::node::ClusterNode;
use crate::shared_sequence::SharedSequence;
use elasticsearch_engine::error::EsError;
use elasticsearch_snapshots::repository::shard::ShardSnapshotTask;
use elasticsearch_snapshots::repository::ShardSnapshotResult;
use elasticsearch_snapshots::service::SnapshotFiles;
use elasticsearch_snapshots::snapshot::{ShardStage, ShardStatus, SnapshotStats};
use elasticsearch_snapshots::{CancelToken, RemoteShardRequest, RemoteShardSnapshots, Repository, RepositoryMetadata, SnapshotError};
use elasticsearch_transport::{RemoteError, TransportError};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;

/// Start (or join) the snapshot of a local primary.
pub const ACTION_UPLOAD: &str = "internal:admin/snapshot/shard/upload";
/// Status of (and abort for) a shard snapshot started by [`ACTION_UPLOAD`].
pub const ACTION_UPLOAD_STATUS: &str = "internal:admin/snapshot/shard/upload/status";

/// ES `snapshot` thread pool maximum on nodes with at least 750mb of heap.
const DEFAULT_SNAPSHOT_THREADS: usize = 10;
const POLL_WAIT_MS: u64 = 2_000;
const MAX_POLL_WAIT_MS: u64 = 5_000;
/// Consecutive unanswered status polls before the shard fails.
const POLL_FAILURES: u32 = 3;
/// Status polls an abort waits for the node's upload to stop.
const ABORT_POLLS: usize = 6;
/// Shards the master pulls at once from nodes that cannot upload.
const LEGACY_PULLS: usize = 4;
/// Finished jobs whose snapshot was never released are forgotten after this.
const FINISHED_JOB_TTL: Duration = Duration::from_secs(3600);

const RUNNING: u8 = 0;
const DONE: u8 = 1;
const FAILED: u8 = 2;
const UNKNOWN: u8 = 3;

/// Test seam (hidden): when not `u64::MAX`, the free space (bytes) a pull
/// to the master must leave instead of the high watermark's.
#[doc(hidden)]
pub static PULL_DISK_RESERVE: AtomicU64 = AtomicU64::new(u64::MAX);
/// Bytes masters of this process pulled from nodes that cannot upload.
#[doc(hidden)]
pub static PULLED_BYTES: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UploadReq {
    pub repository: SharedText,
    pub repo_type: SharedText,
    /// The repository's settings (JSON object) as registered.
    pub settings_json: SharedText,
    pub snapshot_name: SharedText,
    pub snapshot_uuid: SharedText,
    pub index: SharedText,
    pub shard: u32,
    /// The index's id in the repository.
    pub index_id: SharedText,
    /// Shard generation to build on; empty when the shard has none.
    pub base_gen: SharedText,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UploadStatusReq {
    pub snapshot_uuid: SharedText,
    pub index: SharedText,
    pub shard: u32,
    /// Longest the node waits for the upload to finish before answering.
    pub wait_ms: u64,
    /// Stop the upload (snapshot aborted).
    pub abort: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UploadStatusResp {
    /// `RUNNING`, `DONE`, `FAILED` or `UNKNOWN` (no such upload here).
    pub state: u8,
    pub stage: ShardStage,
    pub stats: SnapshotStats,
    pub new_gen: SharedText,
    pub total_size: u64,
    pub segment_count: u32,
    pub failure_kind: SharedText,
    pub failure_status: u16,
    pub failure: SharedText,
}

impl UploadStatusResp {
    fn unknown() -> Self {
        Self {
            state: UNKNOWN,
            stage: ShardStage::Failure,
            stats: SnapshotStats::default(),
            new_gen: SharedText::default(),
            total_size: 0,
            segment_count: 0,
            failure_kind: SharedText::default(),
            failure_status: 0,
            failure: SharedText::default(),
        }
    }
}

// ----- data node ------------------------------------------------------------

type JobKey = (String, String, u32);

struct Job {
    /// What the job writes: the index id in the repository and the shard
    /// generation it builds on. A master asking for anything else (a new
    /// master that resolved another id for an index new to the repository,
    /// or another base generation) gets a new job, never this one's result.
    index_id: String,
    base_gen: String,
    status: Arc<Mutex<ShardStatus>>,
    cancel: CancelToken,
    outcome: Mutex<Option<(Instant, Result<ShardSnapshotResult, SnapshotError>)>>,
    done: tokio::sync::watch::Sender<bool>,
}

impl Job {
    fn new(node_id: &str, r: &UploadReq) -> Arc<Self> {
        let mut st = ShardStatus::new(ShardStage::Init);
        st.node = Some(node_id.to_string());
        Arc::new(Self {
            index_id: r.index_id.to_string(),
            base_gen: r.base_gen.to_string(),
            status: Arc::new(Mutex::new(st)),
            cancel: CancelToken::new(),
            outcome: Mutex::new(None),
            done: tokio::sync::watch::channel(false).0,
        })
    }

    /// Whether a start request for the same shard of the same snapshot can
    /// join this job.
    fn serves(&self, r: &UploadReq) -> bool {
        !self.failed() && self.index_id == r.index_id.as_str() && self.base_gen == r.base_gen.as_str()
    }

    fn finished(&self) -> bool {
        self.outcome.lock().is_some()
    }

    fn failed(&self) -> bool {
        matches!(&*self.outcome.lock(), Some((_, Err(_))))
    }

    fn expired(&self, now: Instant) -> bool {
        matches!(&*self.outcome.lock(), Some((at, _)) if now.duration_since(*at) > FINISHED_JOB_TTL)
    }

    fn finish(&self, res: Result<ShardSnapshotResult, SnapshotError>) {
        if let Err(e) = &res {
            let mut st = self.status.lock();
            st.stage = ShardStage::Failure;
            st.reason = Some(e.to_string());
        }
        *self.outcome.lock() = Some((Instant::now(), res));
        self.done.send_replace(true);
    }

    /// Until the job finished, at most `max`.
    async fn wait(&self, max: Duration) {
        let mut rx = self.done.subscribe();
        let _ = tokio::time::timeout(max, async move {
            while !*rx.borrow() {
                if rx.changed().await.is_err() {
                    break;
                }
            }
        })
        .await;
    }

    fn report(&self) -> Result<UploadStatusResp, EsError> {
        let st = self.status.lock().clone();
        let mut r = UploadStatusResp { state: RUNNING, stage: st.stage, stats: st.stats, ..UploadStatusResp::unknown() };
        if let Some((_, outcome)) = &*self.outcome.lock() {
            match outcome {
                Ok(res) => {
                    r.state = DONE;
                    r.stage = ShardStage::Done;
                    r.stats = res.stats;
                    r.new_gen = SharedText::try_new(&res.new_gen)?;
                    r.total_size = res.total_size;
                    r.segment_count = res.segment_count;
                }
                Err(e) => {
                    r.state = FAILED;
                    r.stage = ShardStage::Failure;
                    r.failure_kind = SharedText::try_new(e.kind)?;
                    r.failure_status = e.status;
                    r.failure = SharedText::try_new(&e.reason)?;
                }
            }
        }
        Ok(r)
    }
}

#[derive(Default)]
struct NodeUploads {
    jobs: Mutex<HashMap<JobKey, Arc<Job>>>,
    /// This node's repository instances (one rate limiter per repository).
    repos: tokio::sync::Mutex<HashMap<String, (RepositoryMetadata, Arc<Repository>)>>,
    /// Shard snapshots running on this node (`thread_pool.snapshot.max`).
    slots: OnceLock<Arc<Semaphore>>,
    /// Pulls from nodes that cannot upload, when this node is master.
    pulls: OnceLock<Arc<Semaphore>>,
}

static UPLOADS: OnceLock<Mutex<HashMap<String, Arc<NodeUploads>>>> = OnceLock::new();

fn uploads_of(node: &ClusterNode) -> Arc<NodeUploads> {
    UPLOADS.get_or_init(Default::default).lock().entry(node.local.id.clone()).or_default().clone()
}

fn snapshot_slots(node: &ClusterNode, u: &NodeUploads) -> Arc<Semaphore> {
    u.slots
        .get_or_init(|| {
            let n = node.effective_setting("thread_pool.snapshot.max").and_then(|v| v.trim().parse::<usize>().ok()).filter(|n| *n > 0).unwrap_or(DEFAULT_SNAPSHOT_THREADS);
            Arc::new(Semaphore::new(n))
        })
        .clone()
}

fn remote_of(e: EsError) -> RemoteError {
    RemoteError::new(&e.kind, e.reason, e.status)
}

pub fn register(node: &Arc<ClusterNode>) {
    let t = &node.transport;
    let n = node.clone();
    t.register_handler(ACTION_UPLOAD, move |r: UploadReq| {
        let n = n.clone();
        async move { start(&n, r).map_err(remote_of) }
    });
    let n = node.clone();
    t.register_handler(ACTION_UPLOAD_STATUS, move |r: UploadStatusReq| {
        let n = n.clone();
        async move { status(&n, r).await.map_err(remote_of) }
    });
}

fn start(node: &Arc<ClusterNode>, r: UploadReq) -> Result<UploadStatusResp, EsError> {
    let u = uploads_of(node);
    let key: JobKey = (r.snapshot_uuid.to_string(), r.index.to_string(), r.shard);
    let now = Instant::now();
    let job = {
        let mut jobs = u.jobs.lock();
        jobs.retain(|_, j| !j.expired(now));
        match jobs.get(&key).cloned() {
            Some(j) if j.serves(&r) => j,
            previous => {
                // A different upload of this shard (another index id or
                // base generation, or a failed one) is stopped first: it
                // shares the shard's staging directory.
                let previous = previous.filter(|p| !p.finished());
                if let Some(p) = &previous {
                    p.cancel.cancel("superseded by a new shard snapshot request");
                }
                let j = Job::new(&node.local.id, &r);
                jobs.insert(key, j.clone());
                tokio::spawn(run_job(node.clone(), r, j.clone(), previous));
                j
            }
        }
    };
    job.report()
}

async fn status(node: &ClusterNode, r: UploadStatusReq) -> Result<UploadStatusResp, EsError> {
    let job = uploads_of(node).jobs.lock().get(&(r.snapshot_uuid.to_string(), r.index.to_string(), r.shard)).cloned();
    let Some(job) = job else { return Ok(UploadStatusResp::unknown()) };
    if r.abort {
        job.cancel.cancel("Snapshot was aborted by deletion");
    }
    if r.wait_ms > 0 {
        job.wait(Duration::from_millis(r.wait_ms.min(MAX_POLL_WAIT_MS))).await;
    }
    job.report()
}

async fn run_job(node: Arc<ClusterNode>, r: UploadReq, job: Arc<Job>, previous: Option<Arc<Job>>) {
    if let Some(p) = previous {
        // Its cleanup removes the staging directory this job freezes into.
        p.wait(Duration::from_secs(24 * 3600)).await;
    }
    let res = upload_shard(&node, &r, &job).await;
    match &res {
        Ok(res) => tracing::debug!(index = %r.index, shard = r.shard, snapshot = %r.snapshot_name, files = res.stats.incremental_file_count, bytes = res.stats.incremental_size, "shard snapshot uploaded"),
        Err(e) => tracing::warn!(index = %r.index, shard = r.shard, snapshot = %r.snapshot_name, "shard snapshot failed: {e}"),
    }
    // The frozen image is no longer needed once its files are in the
    // repository (or the upload gave up).
    let _ = tokio::fs::remove_dir_all(shard::staging_dir(&node, &r.snapshot_uuid, &r.index, r.shard)).await;
    job.finish(res);
}

/// Freeze the local primary and upload it through this node's repository.
async fn upload_shard(node: &Arc<ClusterNode>, r: &UploadReq, job: &Job) -> Result<ShardSnapshotResult, SnapshotError> {
    let _slot = snapshot_slots(node, &uploads_of(node)).acquire_owned().await.map_err(|e| SnapshotError::illegal_state(e.to_string()))?;
    job.cancel.check()?;
    let settings: serde_json::Value = serde_json::from_str(r.settings_json.as_str())?;
    let repo = data_repository(node, RepositoryMetadata::new(r.repository.as_str(), r.repo_type.as_str(), settings)).await?;
    let prepared = shard::prepare(node, &PrepareReq { index: r.index.clone(), shard: r.shard, snapshot_uuid: r.snapshot_uuid.clone() }).await.map_err(snapshot_error)?;
    job.cancel.check()?;
    let dir = PathBuf::from(prepared.dir.as_str());
    let files = shard::snapshot_names(prepared.files).map_err(snapshot_error)?;
    let task = ShardSnapshotTask {
        snapshot_name: r.snapshot_name.to_string(),
        snapshot_uuid: r.snapshot_uuid.to_string(),
        index_id: r.index_id.to_string(),
        shard: r.shard,
        dir,
        files,
        status: job.status.clone(),
        cancel: job.cancel.clone(),
    };
    let base = if r.base_gen.is_empty() { None } else { Some(r.base_gen.to_string()) };
    repo.snapshot_shard_at(base, task).await
}

/// This node's instance of repository `meta` (reopened when its settings
/// change).
async fn data_repository(node: &ClusterNode, meta: RepositoryMetadata) -> Result<Arc<Repository>, SnapshotError> {
    let u = uploads_of(node);
    let mut repos = u.repos.lock().await;
    if let Some((known, repo)) = repos.get(&meta.name) {
        if *known == meta {
            return Ok(repo.clone());
        }
    }
    let repo = Repository::open(meta.clone(), &repo_context(node)).await?;
    repos.insert(meta.name.clone(), (meta, repo.clone()));
    Ok(repo)
}

/// The master released snapshot `snapshot_uuid` (finished or aborted):
/// stop and forget its uploads on this node.
pub(crate) fn forget(node: &ClusterNode, snapshot_uuid: &str) {
    let u = uploads_of(node);
    u.jobs.lock().retain(|(uuid, _, _), job| {
        if uuid == snapshot_uuid {
            job.cancel.cancel("Snapshot was released");
            false
        } else {
            true
        }
    });
}

/// This node, now master, snapshots `index`/`shard` itself: an upload a
/// previous master started here for the same snapshot must stop (and drop
/// its staging links) before the shard is frozen again.
pub(crate) async fn cancel_local_job(node: &ClusterNode, snapshot_uuid: &str, index: &str, shard: u32) {
    let job = uploads_of(node).jobs.lock().remove(&(snapshot_uuid.to_string(), index.to_string(), shard));
    if let Some(job) = job {
        job.cancel.cancel("the shard is snapshotted by the elected master");
        job.wait(Duration::from_secs(120)).await;
    }
}

// ----- master ---------------------------------------------------------------

/// Error kinds a shard failure can carry back from a node (the snapshots
/// crate's errors have static kinds).
const KINDS: &[&str] = &[
    "snapshot_exception",
    "task_cancelled_exception",
    "i_o_exception",
    "no_such_file_exception",
    "file_already_exists_exception",
    "repository_exception",
    "repository_verification_exception",
    "circuit_breaking_exception",
    "illegal_state_exception",
    "illegal_argument_exception",
    "corrupt_index_exception",
    "corrupt_state_exception",
    "parse_exception",
    "index_not_found_exception",
    "connect_transport_exception",
    "node_disconnected_exception",
    "receive_timeout_transport_exception",
    "action_not_found_transport_exception",
    "transport_serialization_exception",
];

fn static_kind(kind: &str) -> &'static str {
    KINDS.iter().find(|k| **k == kind).copied().unwrap_or("snapshot_exception")
}

fn snapshot_error(e: EsError) -> SnapshotError {
    SnapshotError::new(static_kind(&e.kind), e.status, e.reason)
}

fn left(node_id: &str) -> SnapshotError {
    SnapshotError::new("snapshot_exception", 500, format!("node [{node_id}] left the cluster while the snapshot was running"))
}

fn transport_failure(node_id: &str, e: &TransportError) -> SnapshotError {
    let remote = e.to_remote_error();
    SnapshotError::new(static_kind(&remote.error_type), remote.status, format!("[{node_id}] {}", remote.reason))
}

fn is_action_not_found(e: &TransportError) -> bool {
    matches!(e, TransportError::ActionNotFound { .. }) || e.to_remote_error().error_type == "action_not_found_transport_exception"
}

/// The master's [`RemoteShardSnapshots`]: the node holding the primary
/// uploads it, or (an older node) the master pulls and uploads it.
pub(crate) struct Uploader {
    pub node: Arc<ClusterNode>,
}

enum Remote {
    /// The node cannot upload (no such action): pull instead.
    Unsupported,
    Failed(SnapshotError),
}

#[async_trait::async_trait]
impl RemoteShardSnapshots for Uploader {
    async fn snapshot_shard(&self, repo: Arc<Repository>, req: RemoteShardRequest, status: Arc<Mutex<ShardStatus>>, cancel: CancelToken) -> Result<ShardSnapshotResult, SnapshotError> {
        let node = &self.node;
        let Some(peer) = node.state().nodes.get(&req.node_id).cloned() else { return Err(left(&req.node_id)) };
        if internal_capabilities::supports(&peer, Capability::SnapshotShardUploadV1) && crate::capability_probe::verify(&node.transport, &peer, false).await.is_ok() {
            match remote_upload(node, &repo, &req, &status, &cancel).await {
                Ok(res) => return Ok(res),
                Err(Remote::Failed(e)) => return Err(e),
                Err(Remote::Unsupported) => {
                    tracing::info!(node = %req.node_id, index = %req.index_name, shard = req.shard, "node cannot upload its shard snapshot; copying the shard's files to the master");
                }
            }
        }
        legacy_upload(node, &repo, req, status, cancel).await
    }
}

/// Apply a node's report to the shard's status; `Some` once it finished.
fn apply(status: &Mutex<ShardStatus>, resp: &UploadStatusResp, r: &RemoteShardRequest) -> Option<Result<ShardSnapshotResult, SnapshotError>> {
    if resp.state != UNKNOWN {
        let mut st = status.lock();
        st.stage = resp.stage;
        st.stats = resp.stats;
    }
    match resp.state {
        RUNNING => None,
        DONE => Some(Ok(ShardSnapshotResult {
            index_id: r.index_id.clone(),
            shard: r.shard,
            new_gen: resp.new_gen.to_string(),
            old_gen: r.base_gen.clone(),
            stats: resp.stats,
            total_size: resp.total_size,
            segment_count: resp.segment_count,
        })),
        FAILED => Some(Err(SnapshotError::new(static_kind(&resp.failure_kind), if resp.failure_status == 0 { 500 } else { resp.failure_status }, resp.failure.to_string()))),
        _ => Some(Err(SnapshotError::new("snapshot_exception", 500, format!("[{}][{}] the shard snapshot is no longer running on node [{}]", r.index_name, r.shard, r.node_id)))),
    }
}

async fn remote_upload(node: &Arc<ClusterNode>, repo: &Repository, r: &RemoteShardRequest, status: &Mutex<ShardStatus>, cancel: &CancelToken) -> Result<ShardSnapshotResult, Remote> {
    let text = |s: &str| SharedText::try_new(s).map_err(|e| Remote::Failed(snapshot_error(e)));
    let meta = repo.metadata();
    let settings = serde_json::to_string(&meta.settings).map_err(|e| Remote::Failed(SnapshotError::from(e)))?;
    let start = UploadReq {
        repository: text(&meta.name)?,
        repo_type: text(&meta.repo_type)?,
        settings_json: text(&settings)?,
        snapshot_name: text(&r.snapshot_name)?,
        snapshot_uuid: text(&r.snapshot_uuid)?,
        index: text(&r.index_name)?,
        shard: r.shard,
        index_id: text(&r.index_id)?,
        base_gen: text(r.base_gen.as_deref().unwrap_or(""))?,
    };
    let addr = node.address_of(&r.node_id).ok_or_else(|| Remote::Failed(left(&r.node_id)))?;
    let first = match node.transport.send_request::<_, UploadStatusResp>(addr, ACTION_UPLOAD, &start, Duration::from_secs(30)).await {
        Ok(resp) => resp,
        Err(e) if is_action_not_found(&e) => return Err(Remote::Unsupported),
        Err(e) => return Err(Remote::Failed(transport_failure(&r.node_id, &e))),
    };
    if let Some(done) = apply(status, &first, r) {
        return done.map_err(Remote::Failed);
    }
    let mut failures = 0u32;
    loop {
        if cancel.is_cancelled() {
            // Stop the node's upload and wait (bounded) until it stopped, so
            // nothing is written to the repository once the abort returns.
            let abort = UploadStatusReq { snapshot_uuid: start.snapshot_uuid.clone(), index: start.index.clone(), shard: r.shard, wait_ms: MAX_POLL_WAIT_MS, abort: true };
            for _ in 0..ABORT_POLLS {
                let Some(addr) = node.address_of(&r.node_id) else { break };
                match node.transport.send_request::<_, UploadStatusResp>(addr, ACTION_UPLOAD_STATUS, &abort, Duration::from_millis(MAX_POLL_WAIT_MS + 10_000)).await {
                    Ok(resp) if resp.state == RUNNING => continue,
                    _ => break,
                }
            }
            return Err(Remote::Failed(SnapshotError::cancelled(cancel.reason().unwrap_or_else(|| "Snapshot was aborted".into()))));
        }
        let Some(addr) = node.address_of(&r.node_id) else { return Err(Remote::Failed(left(&r.node_id))) };
        let poll = UploadStatusReq { snapshot_uuid: start.snapshot_uuid.clone(), index: start.index.clone(), shard: r.shard, wait_ms: POLL_WAIT_MS, abort: false };
        match node.transport.send_request::<_, UploadStatusResp>(addr, ACTION_UPLOAD_STATUS, &poll, Duration::from_millis(POLL_WAIT_MS + 30_000)).await {
            Ok(resp) => {
                failures = 0;
                if let Some(done) = apply(status, &resp, r) {
                    return done.map_err(Remote::Failed);
                }
            }
            Err(e) => {
                failures += 1;
                if !node.state().nodes.contains_key(&r.node_id) {
                    return Err(Remote::Failed(left(&r.node_id)));
                }
                if failures >= POLL_FAILURES {
                    return Err(Remote::Failed(transport_failure(&r.node_id, &e)));
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    }
}

/// Whether the master can take a copy of `needed` bytes on a filesystem
/// with `disk` usage while keeping `reserve` bytes free (the high disk
/// watermark's). Unknown disk statistics pass.
pub fn check_pull_space(disk: crate::stats::DiskUsage, needed: u64, reserve: u64) -> Result<(), String> {
    if disk.total_bytes == 0 || disk.available_bytes >= needed.saturating_add(reserve) {
        return Ok(());
    }
    Err(format!(
        "insufficient disk space on the master to stage the shard's files for the snapshot (the node holding the primary cannot upload them itself): the files need [{needed}] bytes, [{}] are available and [{reserve}] must stay free (cluster.routing.allocation.disk.watermark.high)",
        disk.available_bytes
    ))
}

fn pull_reserve(node: &ClusterNode, root: &Path) -> u64 {
    match PULL_DISK_RESERVE.load(Ordering::Relaxed) {
        u64::MAX => node.disk_reserve(root),
        v => v,
    }
}

/// An older node: copy the frozen shard into the master's staging
/// directory (when it fits) and upload it from there.
async fn legacy_upload(node: &Arc<ClusterNode>, repo: &Repository, r: RemoteShardRequest, status: Arc<Mutex<ShardStatus>>, cancel: CancelToken) -> Result<ShardSnapshotResult, SnapshotError> {
    let pulls = uploads_of(node).pulls.get_or_init(|| Arc::new(Semaphore::new(LEGACY_PULLS))).clone();
    let _slot = pulls.acquire_owned().await.map_err(|e| SnapshotError::illegal_state(e.to_string()))?;
    cancel.check()?;
    let dest = common::staging_root(node).join(format!("{}-recv", common::dir_name(&r.snapshot_uuid))).join(common::dir_name(&r.index_name)).join(r.shard.to_string());
    let res = pull_and_upload(node, repo, &r, &dest, status, &cancel).await;
    let _ = tokio::fs::remove_dir_all(&dest).await;
    res
}

async fn pull_and_upload(node: &Arc<ClusterNode>, repo: &Repository, r: &RemoteShardRequest, dest: &Path, status: Arc<Mutex<ShardStatus>>, cancel: &CancelToken) -> Result<ShardSnapshotResult, SnapshotError> {
    let files = pull_shard(node, r, dest, cancel).await?;
    let task = ShardSnapshotTask {
        snapshot_name: r.snapshot_name.clone(),
        snapshot_uuid: r.snapshot_uuid.clone(),
        index_id: r.index_id.clone(),
        shard: r.shard,
        dir: dest.to_path_buf(),
        files,
        status,
        cancel: cancel.clone(),
    };
    repo.snapshot_shard_at(r.base_gen.clone(), task).await
}

async fn pull_shard(node: &Arc<ClusterNode>, r: &RemoteShardRequest, dest: &Path, cancel: &CancelToken) -> Result<SnapshotFiles, SnapshotError> {
    let fail = |m: String| SnapshotError::new("snapshot_exception", 500, m);
    let addr = node.address_of(&r.node_id).ok_or_else(|| left(&r.node_id))?;
    let req = PrepareReq {
        index: SharedText::try_new(&r.index_name).map_err(snapshot_error)?,
        shard: r.shard,
        snapshot_uuid: SharedText::try_new(&r.snapshot_uuid).map_err(snapshot_error)?,
    };
    let prepared: PrepareResp = node.transport.send_request(addr, shard::ACTION_PREPARE, &req, Duration::from_secs(300)).await.map_err(|e| transport_failure(&r.node_id, &e))?;
    let mut needed = 0u64;
    for (rel, len) in &prepared.files {
        if !common::safe_rel(rel) {
            return Err(fail(format!("invalid file name [{rel}]")));
        }
        needed = needed.saturating_add(*len);
    }
    let root = common::staging_root(node);
    tokio::fs::create_dir_all(&root).await?;
    let reserve = pull_reserve(node, &root);
    check_pull_space(crate::stats::disk_usage(&root), needed, reserve).map_err(|m| fail(format!("[{}][{}] {m}", r.index_name, r.shard)))?;
    // Counted with the file recoveries in flight on this filesystem, so
    // neither overcommits the space the other checked.
    let reservation = crate::replication::file_recovery::reserve_disk(&root, needed, reserve)
        .map_err(|m| fail(format!("[{}][{}] cannot stage the shard's files on the master for the snapshot: {m}", r.index_name, r.shard)))?;
    let _ = tokio::fs::remove_dir_all(dest).await;
    for (rel, len) in &prepared.files {
        let path = dest.join(rel.as_str());
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let mut f = tokio::fs::File::create(&path).await?;
        let mut off = 0u64;
        while off < *len {
            cancel.check()?;
            let want = CHUNK.min(len - off);
            let rr = ReadReq { snapshot_uuid: req.snapshot_uuid.clone(), index: req.index.clone(), shard: r.shard, path: rel.clone(), offset: off, len: want };
            let chunk: SharedSequence<u8> = node.transport.send_request(addr, shard::ACTION_READ, &rr, Duration::from_secs(120)).await.map_err(|e| transport_failure(&r.node_id, &e))?;
            if chunk.is_empty() {
                return Err(fail(format!("[{rel}] truncated at {off} of {len} bytes")));
            }
            f.write_all(&chunk).await?;
            let n = chunk.len() as u64;
            off += n;
            reservation.written(n);
            PULLED_BYTES.fetch_add(n, Ordering::Relaxed);
        }
        f.flush().await?;
    }
    shard::snapshot_names(prepared.files).map_err(snapshot_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::DiskUsage;

    #[test]
    fn pull_guard_keeps_the_high_watermark_free() {
        let disk = DiskUsage { total_bytes: 1000, free_bytes: 300, available_bytes: 300 };
        assert!(check_pull_space(disk, 100, 200).is_ok(), "exactly at the reserve");
        let e = check_pull_space(disk, 101, 200).unwrap_err();
        assert!(e.starts_with("insufficient disk space on the master"), "{e}");
        assert!(e.contains("[101] bytes, [300] are available and [200] must stay free"), "{e}");
        assert!(e.contains("cluster.routing.allocation.disk.watermark.high"), "{e}");
        // Overflowing sums do not wrap into a pass.
        assert!(check_pull_space(disk, u64::MAX, 1).is_err());
        // Unknown disk statistics pass (as file recovery's check does).
        assert!(check_pull_space(DiskUsage::default(), u64::MAX, u64::MAX).is_ok());
    }

    #[test]
    fn remote_failures_keep_their_kind() {
        assert_eq!(static_kind("i_o_exception"), "i_o_exception");
        assert_eq!(static_kind("some_new_exception"), "snapshot_exception");
        let r = RemoteShardRequest { snapshot_name: "s".into(), snapshot_uuid: "u".into(), index_name: "i".into(), index_id: "id".into(), shard: 2, node_id: "n1".into(), base_gen: Some("g0".into()) };
        let status = Mutex::new(ShardStatus::new(ShardStage::Init));
        let mut resp = UploadStatusResp::unknown();
        resp.state = RUNNING;
        resp.stage = ShardStage::Started;
        resp.stats.processed_size = 7;
        assert!(apply(&status, &resp, &r).is_none());
        let seen = status.lock().clone();
        assert_eq!((seen.stage, seen.stats.processed_size), (ShardStage::Started, 7));
        resp.state = FAILED;
        resp.stage = ShardStage::Failure;
        resp.failure_kind = SharedText::try_new("i_o_exception").unwrap();
        resp.failure_status = 500;
        resp.failure = SharedText::try_new("disk gone").unwrap();
        let e = apply(&status, &resp, &r).unwrap().unwrap_err();
        assert_eq!(e.to_string(), "i_o_exception: disk gone");
        resp.state = DONE;
        resp.new_gen = SharedText::try_new("g1").unwrap();
        let done = apply(&status, &resp, &r).unwrap().unwrap();
        assert_eq!((done.new_gen.as_str(), done.old_gen.as_deref(), done.shard), ("g1", Some("g0"), 2));
        let gone = apply(&status, &UploadStatusResp::unknown(), &r).unwrap().unwrap_err();
        assert!(gone.reason.contains("no longer running on node [n1]"), "{gone}");
    }

    /// Regression (master failover): a new master that resolved another
    /// repository index id (or base generation) for the shard must not join
    /// the previous master's upload, whose blobs live under the old id.
    #[test]
    fn a_job_serves_only_the_same_index_id_and_base_generation() {
        let t = |s: &str| SharedText::try_new(s).unwrap();
        let req = |index_id: &str, base: &str| UploadReq {
            repository: t("repo"), repo_type: t("fs"), settings_json: t("{}"), snapshot_name: t("s"), snapshot_uuid: t("u"),
            index: t("idx"), shard: 0, index_id: t(index_id), base_gen: t(base),
        };
        let job = Job::new("n1", &req("old-id", ""));
        assert!(job.serves(&req("old-id", "")));
        assert!(!job.serves(&req("new-id", "")), "another index id");
        assert!(!job.serves(&req("old-id", "g1")), "another base generation");
        job.finish(Err(SnapshotError::new("i_o_exception", 500, "x")));
        assert!(!job.serves(&req("old-id", "")), "a failed upload is redone");
    }

    #[test]
    fn capability_is_advertised() {
        let mut attrs = std::collections::BTreeMap::new();
        crate::internal_capabilities::advertise(&mut attrs);
        assert_eq!(attrs.get("rustic.internal.snapshot_shard_upload_v1").map(String::as_str), Some("1"));
    }
}
