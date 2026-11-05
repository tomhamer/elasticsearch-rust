//! The master's metadata service: the engine side of ES's `MasterService`.
//!
//! The engine implements every metadata API (index creation with
//! templates, mappings, settings, aliases, data streams, ...), so the
//! elected master runs metadata requests against its own engine and
//! publishes what results (see [`crate::mirror`]). That engine work used to
//! run *inside* the coordinator's state task, i.e. on the coordination
//! actor, before the state was committed: a slow index creation stalled
//! fault detection, and a failed publication left the change in the
//! master's engine.
//!
//! Here it runs on one dedicated task, never on the actor, and no
//! uncommitted metadata reaches a live engine:
//!
//! 1. **Batch.** Everything queued is taken at once (up to [`MAX_BATCH`]):
//!    index-level REST metadata requests (explicit mappings included),
//!    engine mutations (auto-create, system indices), dynamic mapping
//!    updates, registry reconciliation, "publish what the engine has"
//!    requests. Identical mapping updates for the same index collapse into
//!    one. Jobs whose change already lives in the master's engine ("kept":
//!    registry resources, engine APIs that ran outside the service, a
//!    lifecycle pass) form their own group, published first.
//! 2. **Execute.** Index-level requests are *dry runs* on the scratch copy
//!    (`crate::scratch`), synced to the master engine's metadata first: they
//!    produce the engine's own response and the metadata to publish, and
//!    never touch a live engine (a delete destroys nothing, a new index or
//!    alias is invisible to readers until committed). Dynamic mapping
//!    updates merge as a pure function ([`merge_mapping`]).
//! 3. **Publish** one pure state task ([`crate::master::engine_metadata_task`])
//!    with the [`MetadataDelta`], applied to the *current* master state; it
//!    runs on the actor in microseconds, batched there with every other
//!    queued task into one publication.
//! 4. **Apply.** The cluster applier applies the committed state on every
//!    node, the master included. A kept group re-syncs the master's engine
//!    afterwards: a failed publication is rolled back, pure-task changes the
//!    applier skipped while it was pending are caught up.
//! 5. **Answer** every job of the batch with the one publication result.
//!
//! While a kept group is pending the applier does not mirror state metadata
//! into the master's engine 0 (an older committed state applied in between
//! would transiently erase the kept change before it is published). Shard
//! engines and shard reconciliation are unaffected.

use crate::mirror::{self, MetadataDelta};
use crate::node::ClusterNode;
use elasticsearch_cluster::state::{ClusterState, IndexMetadata};
use elasticsearch_transport::RemoteError;
use serde_json::Value;
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use tokio::sync::{mpsc, oneshot};

/// Jobs taken into one batch at most (bounds the time one batch holds the
/// engine before it publishes).
pub const MAX_BATCH: usize = 1024;

thread_local! {
    static ON_SERVICE: Cell<bool> = const { Cell::new(false) };
}

/// Is the current thread executing a metadata batch? A dynamic mapping
/// update raised by engine work inside a batch (a metadata API writing to
/// a system index) is applied locally and published with the batch; asking
/// the master would wait for this very batch.
pub fn on_metadata_service() -> bool {
    ON_SERVICE.with(|c| c.get())
}

struct ServiceScope;

impl ServiceScope {
    fn enter() -> ServiceScope {
        ON_SERVICE.with(|c| c.set(true));
        ServiceScope
    }
}

impl Drop for ServiceScope {
    fn drop(&mut self) {
        ON_SERVICE.with(|c| c.set(false));
    }
}

/// Receives the batch's publication result: the committed version (or the
/// current one when nothing changed), or why it was not committed.
pub type Done = Box<dyn FnOnce(Result<u64, String>) + Send>;

pub enum Job {
    /// Engine work on the master's engine. Returns whether metadata may
    /// have changed (false skips the extraction when the whole batch says
    /// so). A `dry_run` job (index-level metadata APIs: create / delete /
    /// settings / aliases / mappings / templates / data streams) runs on the
    /// scratch copy (`crate::scratch`, [`On::Scratch`]), never on a live
    /// engine; its result is published and applied by the applier. A job
    /// that is not a dry run publishes changes an engine API already made outside the service
    /// (`publish_engine_metadata`) or that must stay (registry resources).
    /// `before`: for a kept job, engine 0 as it was before the change the
    /// job publishes (an engine API that ran outside the service); `None`:
    /// the job makes the change itself.
    Run { run: Box<dyn FnOnce(&ClusterNode, On) -> bool + Send>, done: Done, dry_run: bool, before: Option<Box<crate::scratch::Snapshot>>, unpublished: Option<Arc<std::sync::atomic::AtomicBool>> },
    /// A dynamic mapping update from a primary (`MappingUpdatedAction`):
    /// the full proposed mapping of `index`.
    Mapping { index: String, mapping: crate::owned_json::OwnedJson, waiters: Vec<oneshot::Sender<Result<u64, RemoteError>>>, attempt: u8, uuid: Option<String>, exact_delta: bool },
    /// A pure metadata change computed from the applied state and the
    /// batch's delta so far (e.g. an explicit `PUT _mapping`); it may read
    /// the engine but never changes it.
    Pure { compute: Box<dyn FnOnce(&ClusterNode, &ClusterState, &mut MetadataDelta) + Send>, done: Done },
}

/// The engine a [`Job::Run`] works on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum On {
    /// The master's live engine 0 (kept jobs).
    Live,
    /// The scratch copy (dry runs, `crate::scratch`).
    Scratch,
}


impl ClusterNode {
    /// The engine of `on` (the scratch engine must be open: jobs on it only
    /// run once it is).
    pub fn engine_on(&self, on: On) -> Arc<elasticsearch_engine::node::Node> {
        match (on, self.scratch()) {
            (On::Scratch, Ok(s)) => s.engine.clone(),
            _ => self.engine.clone(),
        }
    }

    pub fn router_on(&self, on: On) -> axum::Router {
        match (on, self.scratch()) {
            (On::Scratch, Ok(s)) => s.router.clone(),
            _ => self.router.clone(),
        }
    }

    pub fn uuids_on(&self, on: On) -> &parking_lot::Mutex<mirror::UuidMap> {
        match (on, self.scratch()) {
            (On::Scratch, Ok(s)) => &s.uuids,
            _ => &self.uuids,
        }
    }
}

/// Counters for tests and `_nodes/stats`-style diagnostics.
#[derive(Default)]
pub struct ServiceStats {
    pub batches: AtomicU64,
    pub jobs: AtomicU64,
    pub publications: AtomicU64,
    pub deduplicated: AtomicU64,
    /// Batches whose publication met a newer base (compare-and-set).
    pub stale_bases: AtomicU64,
    /// Batches that panicked (failed; the service kept running).
    pub panics: AtomicU64,
}

#[derive(Clone)]
pub struct MetadataService {
    tx: mpsc::UnboundedSender<Job>,
    pub stats: Arc<ServiceStats>,
}

impl MetadataService {
    pub fn start(node: Weak<ClusterNode>) -> MetadataService {
        let (tx, rx) = mpsc::unbounded_channel();
        let stats = Arc::new(ServiceStats::default());
        tokio::spawn(run(node, rx, stats.clone()));
        MetadataService { tx, stats }
    }

    /// Queue a job; false when the node is shutting down.
    pub fn submit(&self, job: Job) -> bool {
        self.tx.send(job).is_ok()
    }

    /// [`Self::submit`], handing the job back when the service is gone.
    pub fn submit_job(&self, job: Job) -> Result<(), Job> {
        self.tx.send(job).map_err(|e| e.0)
    }

    /// Queue a dynamic mapping update and wait for its publication.
    pub(crate) async fn put_mapping(&self, index: String, mapping: crate::owned_json::OwnedJson, uuid: Option<String>) -> Result<u64, RemoteError> {
        self.queue_mapping(index, mapping, uuid, false).await
    }

    pub(crate) async fn put_mapping_delta(&self, index: String, mapping: crate::owned_json::OwnedJson, uuid: Option<String>) -> Result<u64, RemoteError> {
        self.queue_mapping(index, mapping, uuid, true).await
    }

    async fn queue_mapping(&self, index: String, mut mapping: crate::owned_json::OwnedJson, uuid: Option<String>, exact_delta: bool) -> Result<u64, RemoteError> {
        mapping.reserve_extra(std::mem::size_of::<Job>() + 512, "queued mapping job and reply")
            .map_err(|e| RemoteError::new(&e.kind,e.reason,e.status))?;
        let (reply, rx) = oneshot::channel();
        if !self.submit(Job::Mapping { index, mapping, waiters: vec![reply], attempt: 0, uuid, exact_delta }) {
            return Err(RemoteError::new("node_closed_exception", "node is shutting down", 503));
        }
        rx.await.unwrap_or_else(|_| Err(RemoteError::new("node_closed_exception", "node is shutting down", 503)))
    }

    /// Queue an engine job and wait for the batch's publication result
    /// (see [`Job::Run`] for `dry_run`).
    pub async fn run(&self, run: Box<dyn FnOnce(&ClusterNode, On) -> bool + Send>, dry_run: bool) -> Result<u64, String> {
        self.run_with_proof(run, dry_run, None).await
    }

    /// Proof is set only after the coordinator returns without ever invoking
    /// the proposed state task. It does not classify errors by message text.
    pub async fn run_with_proof(&self, run: Box<dyn FnOnce(&ClusterNode, On) -> bool + Send>, dry_run: bool, unpublished: Option<Arc<std::sync::atomic::AtomicBool>>) -> Result<u64, String> {
        assert!(dry_run || unpublished.is_none(), "live mutations cannot be replayed using scratch proof");
        let (reply, rx) = oneshot::channel();
        let done: Done = Box::new(move |r| {
            let _ = reply.send(r);
        });
        if !self.submit(Job::Run { run, done, dry_run, before: None, unpublished }) {
            return Err("node is shutting down".into());
        }
        rx.await.unwrap_or_else(|_| Err("metadata task did not execute".into()))
    }

    /// Publish what an engine API changed in the master's engine 0 since
    /// `before` (taken before the API ran): a kept job that changes nothing
    /// itself.
    pub async fn publish_kept_since(&self, before: crate::scratch::Snapshot) -> Result<u64, String> {
        let (reply, rx) = oneshot::channel();
        let done: Done = Box::new(move |r| {
            let _ = reply.send(r);
        });
        let run: Box<dyn FnOnce(&ClusterNode, On) -> bool + Send> = Box::new(|_: &ClusterNode, _: On| true);
        if !self.submit(Job::Run { run, done, dry_run: false, before: Some(Box::new(before)), unpublished: None }) {
            return Err("node is shutting down".into());
        }
        rx.await.unwrap_or_else(|_| Err("metadata task did not execute".into()))
    }

    /// Queue a pure job and wait for the batch's publication result.
    pub async fn pure(&self, compute: Box<dyn FnOnce(&ClusterNode, &ClusterState, &mut MetadataDelta) + Send>) -> Result<u64, String> {
        let (reply, rx) = oneshot::channel();
        let done: Done = Box::new(move |r| {
            let _ = reply.send(r);
        });
        if !self.submit(Job::Pure { compute, done }) {
            return Err("node is shutting down".into());
        }
        rx.await.unwrap_or_else(|_| Err("metadata task did not execute".into()))
    }
}

async fn run(node: Weak<ClusterNode>, mut rx: mpsc::UnboundedReceiver<Job>, stats: Arc<ServiceStats>) {
    use futures::FutureExt;
    while let Some(first) = rx.recv().await {
        let mut jobs = vec![first];
        while jobs.len() < MAX_BATCH {
            match rx.try_recv() {
                Ok(j) => jobs.push(j),
                Err(_) => break,
            }
        }
        let Some(node) = node.upgrade() else { break };
        // A panicking batch fails (its callers see the dropped reply) and
        // the service keeps serving: the scratch copy is rebuilt and the
        // master's engine re-synced to the committed state.
        let batch = std::panic::AssertUnwindSafe(execute_batch(&node, jobs, &stats)).catch_unwind().await;
        if batch.is_err() {
            tracing::error!("a metadata batch panicked; it failed, the service continues");
            stats.panics.fetch_add(1, Ordering::Relaxed);
            if let Ok(scratch) = node.scratch() {
                scratch.reset();
            }
            let n = node.clone();
            tokio::task::block_in_place(move || n.resync_engine_metadata());
        }
    }
}

/// Collapse identical mapping updates (same index, same proposed mapping)
/// into one job answering every waiter.
fn dedupe(jobs: Vec<Job>, stats: &ServiceStats) -> Vec<Job> {
    use std::hash::{BuildHasher, Hash, Hasher};
    struct HashWriter<'a,H>(&'a mut H);
    impl<H:Hasher> std::io::Write for HashWriter<'_,H> {
        fn write(&mut self,bytes:&[u8])->std::io::Result<usize>{self.0.write(bytes);Ok(bytes.len())}
        fn flush(&mut self)->std::io::Result<()>{Ok(())}
    }
    let keys=std::collections::hash_map::RandomState::new();
    let mut out: Vec<Job> = Vec::with_capacity(jobs.len());
    let mut seen: HashMap<u64, Vec<usize>> = HashMap::new();
    for job in jobs {
        match job {
            Job::Mapping { index, mut mapping, waiters, attempt, uuid, exact_delta } => {
                let mut hasher=keys.build_hasher();index.hash(&mut hasher);uuid.hash(&mut hasher);exact_delta.hash(&mut hasher);
                // Stream into the hash; never duplicate an arbitrarily large
                // mapping string merely to collapse identical queued updates.
                let hashed=serde_json::to_writer(HashWriter(&mut hasher), &*mapping);
                if let Err(error)=hashed {
                    let error=RemoteError::new("serialization_exception",error.to_string(),500);
                    for waiter in waiters { let _=waiter.send(Err(error.clone())); }
                    continue;
                }
                let key=hasher.finish();
                let same=seen.get(&key).and_then(|positions|positions.iter().copied().find(|&i|matches!(&out[i],
                    Job::Mapping{index:other,mapping:tree,uuid:other_uuid,exact_delta:other_delta,..}
                        if other==&index && other_uuid==&uuid && *other_delta==exact_delta && **tree==*mapping)));
                if let Some(i)=same {
                    if let Job::Mapping { waiters: first, mapping: owner, .. } = &mut out[i] {
                        let required=first.len().saturating_add(waiters.len());
                        let admitted=if required>first.capacity() {
                            owner.reserve_extra(required.saturating_mul(std::mem::size_of::<oneshot::Sender<Result<u64, RemoteError>>>()),"mapping waiter replacement")
                                .and_then(|_|first.try_reserve_exact(required-first.len()).map_err(|e|elasticsearch_engine::error::EsError::new(429,"circuit_breaking_exception",e.to_string())))
                        } else {Ok(())};
                        let admitted=admitted.and_then(|_|owner.prepare_reply_owner(&mapping));
                        if let Err(error)=admitted {
                            let error=RemoteError::new(&error.kind,error.reason,error.status);
                            for waiter in waiters {let _=waiter.send(Err(error.clone()));}
                            continue;
                        }
                        owner.retain_reply_owner(mapping);
                        first.extend(waiters);
                    }
                    stats.deduplicated.fetch_add(1, Ordering::Relaxed);
                } else {
                    // Hash table entries include control bytes and spare capacity;
                    // sixteen entry units cover geometric old/new table overlap
                    // plus the collision-position Vec (including its minimum
                    // allocation). Each distinct queued tree retains its charge.
                    let index_bytes=16*(std::mem::size_of::<(u64,Vec<usize>)>()+std::mem::size_of::<usize>());
                    if let Err(error)=mapping.reserve_extra(index_bytes,"mapping dedup index") {
                        let error=RemoteError::new(&error.kind,error.reason,error.status);
                        for waiter in waiters {let _=waiter.send(Err(error.clone()));}
                        continue;
                    }
                    seen.entry(key).or_default().push(out.len());
                    out.push(Job::Mapping { index, mapping, waiters, attempt, uuid, exact_delta });
                }
            }
            other => out.push(other),
        }
    }
    out
}

enum Outcome {
    Done(Done),
    /// A kept job publishing an engine API's change since `befores[i]`:
    /// retried from the same snapshot when its publication met a newer base.
    KeptSince(Done, usize),
    /// A mapping job: its waiters, its merge error, and the job itself (to
    /// retry it when the publication found a newer base).
    Mapping(Vec<oneshot::Sender<Result<u64, RemoteError>>>, Option<RemoteError>, String, crate::owned_json::OwnedJson, u8, Option<String>, bool),
}

/// Re-publications of an engine API's kept change that met a newer base.
const KEPT_RETRIES: u8 = 5;

/// Retries of a dynamic mapping update whose publication met a newer base.
const MAPPING_RETRIES: u8 = 3;

fn fail_all(jobs: Vec<Job>, reason: &str) {
    for job in jobs {
        match job {
            Job::Run { done, .. } | Job::Pure { done, .. } => done(Err(reason.to_string())),
            Job::Mapping { waiters, .. } => {
                for w in waiters {
                    let _ = w.send(Err(RemoteError::new("not_master_exception", reason, 503)));
                }
            }
        }
    }
}

async fn execute_batch(node: &Arc<ClusterNode>, jobs: Vec<Job>, stats: &ServiceStats) {
    stats.batches.fetch_add(1, Ordering::Relaxed);
    stats.jobs.fetch_add(jobs.len() as u64, Ordering::Relaxed);
    let jobs = dedupe(jobs, stats);
    // Elected, and the applied state is of this node's own term: a node
    // re-elected while its applier still holds an older state never
    // computes a batch on it.
    if !node.await_own_master_state(MASTER_STATE_WAIT).await {
        fail_all(jobs, &elasticsearch_cluster::coordination::CoordinationError::not_master(&node.local.name).to_string());
        if node.take_metadata_resync() {
            let n = node.clone();
            tokio::task::block_in_place(|| n.resync_engine_metadata());
        }
        return;
    }
    // Registries are cluster-state listeners. Make their startup/reinstall
    // work visible before any request derives mappings from templates; a
    // periodic timer alone permits auto-creation with incorrect dynamic types.
    let (done, completed) = oneshot::channel();
    execute_group(node, vec![Job::Run {
        run: Box::new(|node, _| elasticsearch_engine::registry::ensure(&node.engine)),
        done: Box::new(move |result| { let _ = done.send(result); }),
        dry_run: false, before: None, unpublished: None,
    }], stats).await;
    if let Err(error) = completed.await.unwrap_or_else(|_| Err("registry reconciliation did not complete".into())) {
        fail_all(jobs, &error);
        return;
    }
    // Changes that already live in the engine publish first, in their own
    // group: the dry-run group then starts from what they committed.
    let (keep, rest): (Vec<Job>, Vec<Job>) = jobs.into_iter().partition(|j| matches!(j, Job::Run { dry_run: false, .. }));
    if !keep.is_empty() {
        execute_group(node, keep, stats).await;
    }
    if !rest.is_empty() {
        execute_group(node, rest, stats).await;
    }
}

/// How long a batch waits for the applier to apply this master's own term.
const MASTER_STATE_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// One publication: engine jobs, extraction, pure jobs, publish, answer.
async fn execute_group(node: &Arc<ClusterNode>, jobs: Vec<Job>, stats: &ServiceStats) {
    let proofs: Vec<_> = jobs.iter().filter_map(|job| match job {
        Job::Run { dry_run: true, unpublished: Some(proof), .. } => Some(proof.clone()),
        _ => None,
    }).collect();
    let keeps = jobs.iter().any(|j| matches!(j, Job::Run { dry_run: false, .. }));
    let dry_runs = jobs.iter().filter(|j| matches!(j, Job::Run { dry_run: true, .. })).count();
    let n = node.clone();
    let (outcomes, delta, from_scratch, kept, befores) = tokio::task::block_in_place(move || {
        let _scope = ServiceScope::enter();
        // Only kept jobs work on (and read) the live engine 0: dry runs and
        // pure jobs leave the applier free (the scratch copy and the batch
        // are this task's alone).
        let _metadata = keeps.then(|| n.metadata_mutex.lock());
        let base: Arc<ClusterState> = n.state();
        // Kept changes are diffed against the state engine 0 mirrors (what
        // it held before them), never against a newer committed state: an
        // index committed since but not yet in engine 0 is not "deleted".
        let mirrored: Arc<ClusterState> = if keeps { n.mirrored_state().unwrap_or_else(|| base.clone()) } else { base.clone() };
        // Engine 0 before the kept jobs: their delta is what they changed
        // (before → after), never engine 0's drift from the mirrored state
        // (an index or mapping that failed to apply here must not be
        // published as a delete or a revert).
        // A job that brings its own `before` changed engine 0 before the
        // batch; the group's snapshot covers the jobs that change it here.
        let own_changes = jobs.iter().any(|j| matches!(j, Job::Run { dry_run: false, before: None, .. }));
        let kept_before = (keeps && own_changes).then(|| crate::scratch::snapshot_of(&n.engine, true));
        let mut befores: Vec<Box<crate::scratch::Snapshot>> = Vec::new();
        // Dry runs execute on the scratch copy, synced to the committed
        // metadata (a kept group of this batch has published before, or
        // failed and is not there): only what changed since the last batch
        // is applied to the copy.
        let scratch = if dry_runs > 0 {
            match n.scratch() {
                Ok(scratch) => {
                    scratch.sync(&base.metadata);
                    let before = scratch.snapshot();
                    Some((scratch, before))
                }
                Err(e) => {
                    tracing::error!("cannot open the metadata scratch engine: {e}");
                    None
                }
            }
        } else {
            None
        };
        let mut changed = false;
        let mut kept_changed = false;
        // The batch's successful dry runs so far (a failed one is undone to
        // this, not to the base).
        let mut overlay = crate::scratch::Overlay::default();
        let mut outcomes = Vec::with_capacity(jobs.len());
        let mut pure = Vec::new();
        for job in jobs {
            match job {
                Job::Run { run, done, dry_run: false, before, .. } => {
                    let c = run(&n, On::Live);
                    changed |= c;
                    kept_changed |= c;
                    match before {
                        Some(b) => {
                            outcomes.push(Outcome::KeptSince(done, befores.len()));
                            befores.push(b);
                        }
                        None => outcomes.push(Outcome::Done(done)),
                    }
                }
                Job::Run { run, done, dry_run: true, .. } => {
                    let Some((scratch, batch_before)) = &scratch else {
                        done(Err("the metadata scratch engine is unavailable".into()));
                        continue;
                    };
                    // What this request touched, if it fails: its partial
                    // changes are undone on the copy (and never published).
                    let job_before = (dry_runs > 1).then(|| scratch.snapshot());
                    let ok = run(&n, On::Scratch);
                    if ok {
                        changed = true;
                        if let Some(job_before) = &job_before {
                            scratch.record(job_before, &base.metadata, &mut overlay);
                        }
                    } else {
                        let (names, globals) = scratch.touched(job_before.as_ref().unwrap_or(batch_before));
                        if !names.is_empty() || globals {
                            scratch.restore_over(&names, globals, &base.metadata, &overlay);
                        }
                    }
                    outcomes.push(Outcome::Done(done));
                }
                other => pure.push(other),
            }
        }
        // A kept change is in the live engine until it is published: the
        // applier leaves engine 0 to the re-sync after the publication. (The
        // applier is held off by the metadata lock until here.)
        if kept_changed {
            n.set_engine_metadata_pending(true);
        }
        // What the copy's requests changed, per touched index (no other
        // index is extracted or compared).
        let mut from_scratch: BTreeMap<String, Option<IndexMetadata>> = BTreeMap::new();
        let mut delta = match (&scratch, changed, keeps) {
            (_, false, _) => MetadataDelta::default(),
            (_, true, true) => {
                let all: Vec<&crate::scratch::Snapshot> = kept_before.iter().chain(befores.iter().map(|b| &**b)).collect();
                // The customs the befores captured: one family when they all
                // name the same one (an engine API of that family), else all.
                let mut scopes = all.iter().map(|b| b.customs().flatten());
                let first = scopes.next().flatten();
                let scope = if first.is_some() && scopes.all(|s| s == first) { first } else { None };
                let after = crate::scratch::snapshot_scoped(&n.engine, Some(scope));
                // Only what the jobs changed (before != after) and what is
                // not already the mirrored state (after != mirrored: a
                // state applied meanwhile is not a change of theirs).
                let indices = crate::scratch::changes_between(&n.engine, &mut n.uuids.lock(), &all, &after, &mirrored.metadata);
                let mut delta = MetadataDelta { indices, ..MetadataDelta::default() };
                // Globals: what the jobs changed (before -> after) laid onto
                // the latest committed state, entry by entry, as ES's master
                // runs a task against the latest state: a concurrent change
                // of another entry (another pipeline) is no conflict.
                let b = &base.metadata;
                let mut persistent = b.persistent_settings.clone();
                let mut transient = b.transient_settings.clone();
                for before in &all {
                    let (bg, ag) = (before.globals(), after.globals());
                    for (from, to, out, committed) in [
                        (&bg.index_templates, &ag.index_templates, &mut delta.index_templates, &b.index_templates),
                        (&bg.component_templates, &ag.component_templates, &mut delta.component_templates, &b.component_templates),
                        (&bg.legacy_templates, &ag.legacy_templates, &mut delta.legacy_templates, &b.legacy_templates),
                    ] {
                        for k in from.keys().chain(to.keys()) {
                            if from.get(k) != to.get(k) && committed.get(k) != to.get(k) {
                                out.insert(k.clone(), to.get(k).cloned());
                            }
                        }
                    }
                    for (from, to, out) in [(&bg.persistent_settings, &ag.persistent_settings, &mut persistent), (&bg.transient_settings, &ag.transient_settings, &mut transient)] {
                        for k in from.keys().chain(to.keys()).cloned().collect::<Vec<String>>() {
                            if from.get(&k) != to.get(&k) {
                                match to.get(&k) {
                                    Some(v) => {
                                        out.insert(k, v.clone());
                                    }
                                    None => {
                                        out.remove(&k);
                                    }
                                }
                            }
                        }
                    }
                    let family = before.customs().flatten();
                    for k in bg.customs.keys().chain(ag.customs.keys()).filter(|k| family.is_none_or(|f| f == k.as_str())).cloned().collect::<std::collections::BTreeSet<String>>() {
                        if bg.customs.get(&k) == ag.customs.get(&k) {
                            continue;
                        }
                        let current = delta.customs.get(&k).cloned().unwrap_or_else(|| b.customs.get(&k).cloned());
                        let merged = crate::meta_extra::merge_custom_change(&k, current.as_ref(), bg.customs.get(&k), ag.customs.get(&k));
                        delta.customs.insert(k, merged);
                    }
                }
                delta.customs.retain(|k, v| b.customs.get(k) != v.as_ref());
                if persistent != b.persistent_settings {
                    delta.persistent_settings = Some(persistent);
                }
                if transient != b.transient_settings {
                    delta.transient_settings = Some(transient);
                }
                // Indices against engine 0's mirrored base (drift is never a
                // change), globals against the committed base they were
                // laid onto.
                delta.expect_base(&mirrored.metadata);
                delta.expect_globals_base(b);
                delta
            }
            (Some((scratch, before)), true, false) => {
                let (indices, globals) = scratch.changes(before, &base.metadata);
                scratch.close_copies(indices.iter().filter(|(_, c)| c.is_some()).map(|(k, _)| k.clone()));
                from_scratch = indices.clone();
                let mut delta = MetadataDelta { indices, ..MetadataDelta::default() };
                delta.set_globals(&base.metadata, &globals);
                if delta.has_globals() {
                    scratch.globals_dirty();
                }
                delta
            }
            (None, true, false) => MetadataDelta::default(),
        };
        // Dynamic mapping updates and pure jobs, in queue order, computed as
        // pure functions onto the metadata this batch publishes (never into
        // an engine: the applier applies the committed result on every
        // node, this one included).
        for job in pure {
            match job {
                Job::Mapping { index, mapping, waiters, attempt, uuid, exact_delta } => {
                    let r = merge_into_delta(&n, &base, &mut delta, &index, &mapping, uuid.as_deref(), exact_delta);
                    outcomes.push(Outcome::Mapping(waiters, r.err(), index, mapping, attempt, uuid, exact_delta));
                }
                Job::Pure { compute, done } => {
                    compute(&n, &base, &mut delta);
                    outcomes.push(Outcome::Done(done));
                }
                Job::Run { .. } => unreachable!("engine jobs ran above"),
            }
        }
        if !keeps {
            // Expectations for everything the dry runs and pure jobs touched.
            delta.expect_base(&base.metadata);
        }
        (outcomes, Some(delta).filter(|d| !d.is_empty()), from_scratch, kept_changed, befores)
    });
    let published: BTreeMap<String, Option<IndexMetadata>> = delta.as_ref().map(|d| d.indices.clone()).unwrap_or_default();
    let result = match delta {
        None => Ok(node.state().version),
        Some(delta) => {
            stats.publications.fetch_add(1, Ordering::Relaxed);
            let invoked = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let entered = invoked.clone();
            let task = crate::master::engine_metadata_task(node.master.clone(), delta);
            let result = node.coord().submit(Box::new(move |current| {
                entered.store(true, Ordering::Release);
                task(current)
            })).await;
            if result.is_err() && !invoked.load(Ordering::Acquire) {
                for proof in &proofs { proof.store(true, Ordering::Release); }
            }
            result
        }
    };
    let stale = result.as_ref().is_err_and(|e| e.starts_with(crate::master::STALE_BASE));
    if stale {
        stats.stale_bases.fetch_add(1, Ordering::Relaxed);
    }
    // The scratch copy now holds what the batch published (committed) or
    // needs those copies re-created (not committed, or changed further by
    // a pure job of the batch after the copy produced them).
    if !from_scratch.is_empty() {
        if let Ok(scratch) = node.scratch() {
            if result.is_ok() {
                let diverged: Vec<String> = from_scratch.iter().filter(|(k, v)| published.get(*k) != Some(*v)).map(|(k, _)| k.clone()).collect();
                let settled: BTreeMap<String, Option<IndexMetadata>> = from_scratch.into_iter().filter(|(k, _)| !diverged.contains(k)).collect();
                scratch.settle(&settled, diverged);
            } else {
                scratch.settle(&BTreeMap::new(), from_scratch.into_keys());
                scratch.globals_dirty();
            }
        }
    }
    if let Err(e) = &result {
        tracing::warn!("metadata batch was not committed: {e}");
    }
    // Kept changes: roll back (failure) or catch engine 0 up to the
    // committed state. Dry runs and pure jobs left no engine change; the
    // applier applies their committed state as usual.
    let resync = node.take_metadata_resync();
    // A stale base is retried from the kept snapshots (engine 0 keeps the
    // change; it only catches up), never rolled back.
    if resync || (kept && result.is_err() && !stale) {
        // Roll engine 0 back to the committed state (a full pass), once no
        // engine API is in flight (an immediate re-sync would erase their
        // unpublished changes); the applier keeps applying meanwhile.
        node.set_engine_metadata_pending(false);
        node.request_engine_rollback();
    } else if kept {
        // Committed: engine 0 already holds the change; it only catches up
        // with what the applier left to it meanwhile (incremental).
        let n = node.clone();
        tokio::task::block_in_place(move || n.catch_up_engine_metadata());
    }
    let mut befores: Vec<Option<Box<crate::scratch::Snapshot>>> = befores.into_iter().map(Some).collect();
    // A kept change that ends unpublished (not retried) is rolled back out of
    // engine 0 below: the master must not keep state no other node has.
    // A kept change that ends unpublished (a stale kept job without a
    // snapshot, retries exhausted) is rolled back out of engine 0: the
    // master must not keep state no other node has. Then the group's other
    // stale kept jobs are not retried either (a retry would diff its snapshot
    // against the rolled-back engine 0 and publish the rollback): they all
    // fail, unapplied.
    let rollback = stale && kept && outcomes.iter().any(|o| match o {
        Outcome::Done(_) => true,
        Outcome::KeptSince(_, i) => befores.get(*i).and_then(Option::as_ref).is_none_or(|b| b.attempt >= KEPT_RETRIES),
        Outcome::Mapping(..) => false,
    });
    // Before any caller learns its request failed: a retry must not take its
    // snapshot from the engine 0 this rollback resets.
    if rollback {
        node.request_engine_rollback();
    }
    for o in outcomes {
        match o {
            // The engine API's change is still in engine 0: computed again
            // from the same snapshot against the newer state (ES's master runs
            // a task against the latest state), not reported stale.
            Outcome::KeptSince(done, i) if stale && !rollback => match befores.get_mut(i).and_then(Option::take) {
                Some(mut before) => {
                    before.attempt += 1;
                    let run: Box<dyn FnOnce(&ClusterNode, On) -> bool + Send> = Box::new(|_: &ClusterNode, _: On| true);
                    if let Err(Job::Run { done, .. }) = node.metadata_service().submit_job(Job::Run { run, done, dry_run: false, before: Some(before), unpublished: None }) {
                        done(result.clone());
                    }
                }
                None => done(result.clone()),
            },
            Outcome::KeptSince(done, _) if stale => done(result.clone().map_err(|e| format!("{e}; the request was not applied, retry it"))),
            Outcome::KeptSince(done, _) => done(result.clone()),
            Outcome::Done(done) => {
                done(result.clone().map_err(|e| if e.starts_with(crate::master::STALE_BASE) {
                    format!("{e}; the request was not applied, retry it")
                } else {
                    e
                }))
            }
            Outcome::Mapping(waiters, err, index, mapping, attempt, uuid, exact_delta) => {
                // Data, not a closure: retried against the newer base.
                if err.is_none() && stale && attempt < MAPPING_RETRIES {
                    let _ = node.metadata_service().submit(Job::Mapping { index, mapping, waiters, attempt: attempt + 1, uuid, exact_delta });
                    continue;
                }
                let r = match err {
                    Some(e) => Err(e),
                    None => result.clone().map_err(|e| RemoteError::new("failed_to_commit_cluster_state_exception", e, 503)),
                };
                for w in waiters {
                    let _ = w.send(r.clone());
                }
            }
        }
    }
}
/// `proposal` (a primary's complete proposed mapping) merged into `base`
/// with the index's own analysis, creation version and mapping limits, as
/// `IndexService::put_mapping` merges, but on detached values: nothing is
/// applied to any engine.
pub fn merge_mapping(svc: &elasticsearch_engine::index::IndexService, base: &Value, proposal: &Value) -> elasticsearch_engine::error::Result<Value> {
    use elasticsearch_engine::mapping::Mappings;
    let registry = svc.registry();
    let mut merged = Mappings::parse(base, &registry)?;
    let update = Mappings::parse_update(proposal, &registry)?;
    // The runtime-field rules every copy's put_mapping enforces (a time
    // series index's routing_path and dimensions, the index sort). Merged
    // here without them, a `dynamic: runtime` field matching routing_path
    // was committed, every copy refused to apply it and quarantined the
    // index, and the write got "the local copy ... was replaced" instead of
    // ES's routing_path error (tsdb/20_mapping.yml).
    if let Some(rt) = &update.runtime {
        let settings = svc.meta().settings;
        elasticsearch_engine::tsdb::check_runtime_shadow(&settings, &merged, rt)?;
        elasticsearch_engine::indexsort::check_runtime_shadows_sort(&settings, Some(rt))?;
    }
    elasticsearch_engine::inference::semantic_map::with_created(svc.created_version(), || merged.merge(&update))?;
    let limits = elasticsearch_engine::fieldtypes::limits::Limits::from_settings(|k| svc.setting(k));
    elasticsearch_engine::fieldtypes::limits::check(&merged, &limits)?;
    Ok(merged.to_json())
}

/// Merge one dynamic mapping update into the index metadata the batch
/// publishes (the batch's own change to the index, else the applied one).
/// Concurrent conflicting updates resolve here, in queue order: the first
/// wins and the later one gets the merge error (ES's
/// `illegal_argument_exception` / `mapper_parsing_exception`).
fn merge_into_delta(node: &ClusterNode, base: &ClusterState, delta: &mut MetadataDelta, index: &str, proposal: &Value, uuid: Option<&str>, exact_delta: bool) -> Result<(), RemoteError> {
    let missing = || RemoteError::new("index_not_found_exception", format!("no such index [{index}]"), 404);
    let current: IndexMetadata = match delta.indices.get(index) {
        Some(Some(im)) => im.clone(),
        Some(None) => return Err(missing()),
        None => base.metadata.indices.get(index).cloned().ok_or_else(missing)?,
    };
    // A proposal from another incarnation of the name (deleted and
    // re-created meanwhile) never merges into this one.
    if uuid.is_some_and(|u| u != current.uuid) {
        return Err(missing());
    }
    // The index's analysis and limits (the engine copy is metadata the
    // applier keeps in step; it is only read here).
    let svc = node.engine.get_index(index).ok_or_else(missing)?;
    // `index.blocks.metadata` / `read_only` / `read_only_allow_delete`
    // block mapping updates, dynamic ones included (ES checks
    // METADATA_WRITE on the put-mapping request).
    svc.check_metadata_block(true).map_err(|e| RemoteError::new(e.kind.clone(), e.reason.clone(), e.status))?;
    // Only what the proposal adds (an older or stale primary's whole
    // mapping must not revert root or field parameters, e.g. `dynamic`).
    let proposal = &if exact_delta { crate::mapping_update::materialize_update(&current.mappings, proposal) } else { crate::mapping_update::additive_update(&current.mappings, proposal) };
    let merged = merge_mapping(&svc, &current.mappings, proposal).map_err(|e| RemoteError::new(e.kind.clone(), e.reason.clone(), e.status))?;
    if merged == *current.mappings {
        return Ok(());
    }
    let mut im = current;
    im.mappings = merged.into();
    im.mapping_version += 1;
    im.version += 1;
    delta.indices.insert(index.to_string(), Some(im));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn mapping_job(index: &str, mapping: Value) -> (Job, oneshot::Receiver<Result<u64, RemoteError>>) {
        let (tx, rx) = oneshot::channel();
        (Job::Mapping { index: index.into(), mapping: crate::owned_json::OwnedJson::parse(&mapping.to_string()).unwrap(), waiters: vec![tx], attempt: 0, uuid: None, exact_delta: false }, rx)
    }

    #[test]
    fn identical_mapping_updates_collapse_per_index() {
        let stats = ServiceStats::default();
        let m = json!({"properties": {"a": {"type": "keyword"}}});
        let (a, _ra) = mapping_job("i", m.clone());
        let (b, _rb) = mapping_job("i", m.clone());
        let (c, _rc) = mapping_job("j", m.clone());
        let (d, _rd) = mapping_job("i", json!({"properties": {"a": {"type": "long"}}}));
        let out = dedupe(vec![a, b, c, d], &stats);
        assert_eq!(out.len(), 3, "same index + same mapping collapse; other index / other mapping stay");
        match &out[0] {
            Job::Mapping { index, waiters, .. } => {
                assert_eq!(index, "i");
                assert_eq!(waiters.len(), 2);
            }
            _ => panic!("mapping job expected"),
        }
        assert_eq!(stats.deduplicated.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn mapping_merge_is_pure_and_the_first_type_wins() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = elasticsearch_engine::node::Node::open(&tmp.path().join("e")).unwrap();
        let svc = engine.create_index("m", &json!({"mappings": {"properties": {"a": {"type": "keyword"}}}})).unwrap();
        let base = svc.mapping_json();
        let merged = merge_mapping(&svc, &base, &json!({"properties": {"a": {"type": "keyword"}, "b": {"type": "long"}}})).unwrap();
        assert_eq!(merged["properties"]["b"]["type"], "long");
        assert_eq!(merged["properties"]["a"]["type"], "keyword");
        assert!(svc.mapping_json()["properties"].get("b").is_none(), "nothing reaches the engine before the commit");
        assert!(merge_mapping(&svc, &merged, &json!({"properties": {"b": {"type": "text"}}})).is_err(), "a later conflicting type loses");
        assert_eq!(merge_mapping(&svc, &merged, &merged).unwrap(), merged, "re-proposing the committed mapping changes nothing");
    }

    #[test]
    fn service_scope_is_thread_local_and_cleared() {
        assert!(!on_metadata_service());
        {
            let _s = ServiceScope::enter();
            assert!(on_metadata_service());
            std::thread::spawn(|| assert!(!on_metadata_service())).join().unwrap();
        }
        assert!(!on_metadata_service());
    }
}
