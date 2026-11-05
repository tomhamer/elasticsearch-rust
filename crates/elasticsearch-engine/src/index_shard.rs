//! One local copy of logical shard `n` of an index ([`IndexShard`]):
//! `indices/<uuid>/shard-<n>/`, its write path, replica apply, file sets
//! and searchers. Compiled as a child module of `crate::index` (it shares
//! the index core's private parsing / mapping machinery).
//!
//! Lifecycle guarantees (the cluster layer relies on them):
//! * **A1 fencing**: every operation takes an op permit; once
//!   [`IndexShard::close`] ran, operations fail with the typed
//!   `index_shard_closed_exception` ([`EsError::is_already_closed`]).
//!   `close` waits for in-flight permits before releasing the store.
//! * **A2 primary mode**: `write` / `update` on a replica copy fail with
//!   `shard_not_in_primary_mode_exception`; replica apply works on both.
//! * **A3 exclusive lock**: an open copy holds `shard-<n>.lock` (flock);
//!   a second open of the same shard fails with
//!   `shard_lock_obtain_failed_exception`.
//! * **A4 durable close**: `close` commits (flush) so a later
//!   `ExistingStore` open recovers every acknowledged operation.

use super::*;
use crate::hooks::changes::{ChangeOp, Durability, ReplicaApplyResult};
use crate::hooks::files::{FileSource, IndexFileSet, IndexManifest};
use crate::shard::PreparedReplicaOp;
use parking_lot::Condvar;
use serde::{Deserialize, Serialize};
use std::sync::atomic::AtomicI64;

/// Overall bound of the per-item dynamic-mapping fallback of one shard
/// request (each item may need its own master round trip).
const DYNAMIC_FALLBACK_BUDGET: std::time::Duration = std::time::Duration::from_secs(60);

/// How to bring up a shard copy ([`IndexService::open_shard`]).
#[derive(Clone, Debug)]
pub struct ShardOpenSpec {
    /// ES allocation id of this copy (reported in shard stats / routing).
    pub allocation_id: String,
    /// Primary (accepts `write` / `update`) or replica (replica apply only).
    pub primary: bool,
    /// Operating primary term; the copy moves up to it (never down).
    pub primary_term: u64,
    pub recovery: RecoverySource,
}

impl ShardOpenSpec {
    /// A local primary at term 1 (single-node indices).
    pub fn local_primary(recovery: RecoverySource) -> ShardOpenSpec {
        ShardOpenSpec { allocation_id: new_uuid(), primary: true, primary_term: 1, recovery }
    }
}

/// Where a shard copy's data comes from.
#[derive(Clone, Debug)]
pub enum RecoverySource {
    /// A new, empty shard (`shard-<n>/` must not hold a shard yet).
    Empty,
    /// The shard already on disk (commit + translog replay).
    ExistingStore,
    /// Files received from a peer / snapshot into `staged` (e.g. by
    /// [`IndexFileSet::write_to_dir`]): verified against `manifest`, then
    /// atomically renamed into `shard-<n>/` (which must not exist). On
    /// failure nothing is left under `shard-<n>/`.
    FromFiles { staged: PathBuf, manifest: IndexManifest },
    /// [`FromFiles`](Self::FromFiles) of a peer recovery's staged copy:
    /// every file was verified (chunk checksums and the whole-file CRC-32)
    /// before it was renamed into the staging directory, so installation
    /// checks only that each is present with its length (no second read of
    /// tens of GB).
    FromRecoveredFiles { staged: PathBuf, manifest: IndexManifest },
}

/// Per-copy statistics (`_stats?level=shards`, `_cat/shards`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardStats {
    pub shard: u32,
    pub allocation_id: String,
    pub primary: bool,
    pub docs: u64,
    pub deleted_docs: u64,
    pub store_size: u64,
    pub segments: usize,
    pub translog_size: u64,
    pub uncommitted_ops: i64,
    pub max_seq_no: i64,
    pub local_checkpoint: i64,
    pub global_checkpoint: i64,
    pub primary_term: u64,
}

// ------------------------------------------------------------- shard lock

#[cfg(unix)]
fn try_flock(f: &std::fs::File) -> std::io::Result<bool> {
    use std::os::unix::io::AsRawFd;
    unsafe extern "C" {
        fn flock(fd: i32, operation: i32) -> i32;
    }
    const LOCK_EX: i32 = 2;
    const LOCK_NB: i32 = 4;
    // SAFETY: `flock` on a file descriptor we own for the call's duration.
    let r = unsafe { flock(f.as_raw_fd(), LOCK_EX | LOCK_NB) };
    if r == 0 {
        return Ok(true);
    }
    let e = std::io::Error::last_os_error();
    if e.kind() == std::io::ErrorKind::WouldBlock {
        Ok(false)
    } else {
        Err(e)
    }
}

#[cfg(not(unix))]
fn try_flock(_f: &std::fs::File) -> std::io::Result<bool> {
    Ok(true)
}

/// Lock paths held by this process (flock also conflicts within a process
/// on Linux / macOS; this also covers filesystems without flock).
fn held_locks() -> &'static Mutex<std::collections::HashSet<PathBuf>> {
    static H: std::sync::OnceLock<Mutex<std::collections::HashSet<PathBuf>>> = std::sync::OnceLock::new();
    H.get_or_init(|| Mutex::new(std::collections::HashSet::new()))
}

/// The exclusive lock of one shard directory (`<index>/shard-<n>.lock`,
/// beside `shard-<n>/` so it can be held before that directory exists).
pub(crate) struct ShardLock {
    path: PathBuf,
    _file: std::fs::File,
}

impl ShardLock {
    fn path_of(index_dir: &Path, n: u32) -> PathBuf {
        index_dir.join(format!("shard-{n}.lock"))
    }

    /// `None` when another copy holds it.
    fn try_acquire(index_dir: &Path, n: u32) -> Result<Option<ShardLock>> {
        let path = Self::path_of(index_dir, n);
        {
            let mut h = held_locks().lock();
            if !h.insert(path.clone()) {
                return Ok(None);
            }
        }
        let release = |p: &PathBuf| {
            held_locks().lock().remove(p);
        };
        if let Err(e) = std::fs::create_dir_all(index_dir) {
            release(&path);
            return Err(e.into());
        }
        let file = match std::fs::OpenOptions::new().create(true).truncate(false).read(true).write(true).open(&path) {
            Ok(f) => f,
            Err(e) => {
                release(&path);
                return Err(e.into());
            }
        };
        match try_flock(&file) {
            Ok(true) => Ok(Some(ShardLock { path, _file: file })),
            Ok(false) => {
                release(&path);
                Ok(None)
            }
            Err(e) => {
                release(&path);
                Err(e.into())
            }
        }
    }
}

impl Drop for ShardLock {
    fn drop(&mut self) {
        // Closing the file releases the flock.
        held_locks().lock().remove(&self.path);
    }
}

// ------------------------------------------------------------- the copy

/// One local copy of logical shard `id`.
pub struct IndexShard {
    pub id: u32,
    pub allocation_id: String,
    pub shard: Shard,
    core: Arc<IndexCore>,
    primary: AtomicBool,
    closed: AtomicBool,
    /// In-flight operations (op permits).
    inflight: Mutex<usize>,
    drained: Condvar,
    lock: Mutex<Option<ShardLock>>,
    /// [`IndexCore::schema_generation`] this copy's core schema matches.
    synced_schema: AtomicU64,
    /// Whether the mapping had a `@timestamp` date field when this copy
    /// first synced its merge settings (its open): ES chooses the merge
    /// policy when the engine is created, so a `@timestamp` mapped later
    /// takes effect at the next open.
    timestamp_at_open: std::sync::OnceLock<bool>,
    global_checkpoint: AtomicI64,
    /// Reservations of this copy's local recovery (translog validation and
    /// replay, the publish) that went past a breaker's limit, forced.
    recovery_over_limit: AtomicU64,
}

/// ES `MappingLookup.getTimestampFieldType() != null`: `@timestamp` is a
/// mapped `date` or `date_nanos` field (a concrete one, not runtime).
pub(crate) fn timestamp_field(lookup: &crate::mapping::FieldLookup) -> bool {
    lookup.get("@timestamp").is_some_and(|f| f.core_id.is_some()
        && matches!(f.ty, crate::mapping::EsType::Date | crate::mapping::EsType::DateNanos))
}

impl std::ops::Deref for IndexShard {
    type Target = IndexCore;
    fn deref(&self) -> &IndexCore {
        &self.core
    }
}

impl std::fmt::Debug for IndexShard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IndexShard([{}][{}], {})", self.core.name, self.id, if self.is_primary() { "primary" } else { "replica" })
    }
}

/// An operation permit (A1): held for the duration of one operation.
pub(crate) struct OpPermit<'a>(&'a IndexShard);

impl Drop for OpPermit<'_> {
    fn drop(&mut self) {
        let mut n = self.0.inflight.lock();
        *n -= 1;
        if *n == 0 {
            self.0.drained.notify_all();
        }
    }
}

/// Where the shard's files sit in a staged directory: `staged/` itself
/// (shard-level file set) or `staged/shard-<k>/` (index-level file set from
/// the deprecated [`IndexService::snapshot_files`]).
fn staged_shard_root(staged: &Path, manifest: &IndexManifest) -> PathBuf {
    let mut prefix: Option<&str> = None;
    for f in &manifest.files {
        let Some((first, _)) = f.path.split_once('/') else { continue };
        if first.starts_with("shard-") {
            match prefix {
                None => prefix = Some(first),
                Some(p) if p == first => {}
                Some(_) => return staged.to_path_buf(),
            }
        } else {
            return staged.to_path_buf();
        }
    }
    match prefix {
        Some(p) if manifest.files.iter().all(|f| f.path.starts_with(&format!("{p}/")) || !f.path.contains('/')) => staged.join(p),
        _ => staged.to_path_buf(),
    }
}

/// FromFiles: verify, then move the staged shard into `dir` atomically
/// (`verified`: the files were checksummed as they were received; only
/// their presence and lengths are checked).
fn install_staged(dir: &Path, staged: &Path, manifest: &IndexManifest, verified: bool) -> Result<()> {
    if dir.exists() {
        return Err(EsError::new(500, "illegal_state_exception", format!("cannot recover from files into [{}]: the shard directory already exists", dir.display())));
    }
    if verified {
        crate::hooks::files::verify_index_dir_lengths(staged, manifest)?;
    } else {
        crate::hooks::files::verify_index_dir(staged, manifest)?;
    }
    let root = staged_shard_root(staged, manifest);
    if std::fs::rename(&root, dir).is_err() {
        // Another filesystem: copy beside the target, then rename.
        let tmp = dir.with_extension("recovering");
        let _ = std::fs::remove_dir_all(&tmp);
        let r = crate::ilm::resize::copy_tree(&root, &tmp).and_then(|_| std::fs::rename(&tmp, dir).map_err(EsError::from));
        if let Err(e) = r {
            let _ = std::fs::remove_dir_all(&tmp);
            let _ = std::fs::remove_dir_all(dir);
            return Err(e);
        }
    }
    crate::hooks::sync_dir(dir.parent());
    // Per-copy state never transfers.
    let _ = std::fs::remove_file(dir.join(crate::hooks::files::MANIFEST_FILE));
    let _ = std::fs::remove_file(dir.join(crate::hooks::retention::LEASES_FILE));
    let _ = std::fs::remove_file(dir.join(META_FILE));
    Ok(())
}

impl IndexShard {
    /// Open (create / recover) copy `n` of the index at `core` (see
    /// [`IndexService::open_shard`]).
    pub(crate) fn open(core: &Arc<IndexCore>, n: u32, spec: ShardOpenSpec) -> Result<Arc<IndexShard>> {
        let lock = ShardLock::try_acquire(&core.path, n)?.ok_or_else(|| EsError::shard_lock_failed(&core.name, &core.uuid, n, "starting shard"))?;
        let dir = core.path.join(format!("shard-{n}"));
        // The generation is read before the schema: a mapping update landing
        // in between leaves the copy behind the generation it records, so its
        // first write syncs the new fields (read after, the copy would claim
        // a generation whose fields it never got and index them as unknown).
        let schema_generation = core.schema_generation();
        let schema = core.lookup().core_schema();
        let mut gcp_from_manifest = None;
        let from_files = matches!(spec.recovery, RecoverySource::FromFiles { .. } | RecoverySource::FromRecoveredFiles { .. });
        // FromFiles: whether `shard-<n>/` is ours to clean up on failure.
        let mut installed = false;
        // Recovering the copy from its store reads and applies acknowledged
        // ops (translog validation and replay, and the refresh that makes
        // them searchable): a [`crate::memory::Replay`], never refused or
        // delayed by a circuit breaker (ES `recoverFromTranslog`).
        let replay = crate::memory::Replay::new(format!("local recovery of [{}][{n}]", core.name));
        let opened = replay.run(|| -> Result<(Shard, crate::translog::TranslogReplay)> {
            match &spec.recovery {
                RecoverySource::Empty => {
                    if dir.join("shard_state.json").exists() {
                        return Err(EsError::new(500, "illegal_state_exception", format!("shard [{}][{n}] already has data at [{}]; remove it first", core.name, dir.display())));
                    }
                    Ok((Shard::create(&dir, &core.name, schema)?, crate::translog::TranslogReplay::empty()))
                }
                RecoverySource::ExistingStore => Shard::open_for_replay(&dir, schema),
                RecoverySource::FromFiles { staged, manifest } | RecoverySource::FromRecoveredFiles { staged, manifest } => {
                    install_staged(&dir, staged, manifest, matches!(spec.recovery, RecoverySource::FromRecoveredFiles { .. }))?;
                    installed = true;
                    gcp_from_manifest = Some(manifest.local_checkpoint);
                    Shard::open_for_replay(&dir, schema)
                }
            }
        });
        let (shard, ops) = match opened {
            Ok(x) => x,
            Err(e) => {
                if installed {
                    let _ = std::fs::remove_dir_all(&dir);
                }
                return Err(e);
            }
        };
        shard.core.read().vector_store().set_gpu_gate(crate::gpu::gate_for_path(&dir));
        let s = Arc::new(IndexShard {
            id: n,
            allocation_id: spec.allocation_id.clone(),
            shard,
            core: core.clone(),
            primary: AtomicBool::new(spec.primary),
            closed: AtomicBool::new(false),
            inflight: Mutex::new(0),
            drained: Condvar::new(),
            lock: Mutex::new(Some(lock)),
            synced_schema: AtomicU64::new(schema_generation),
            global_checkpoint: AtomicI64::new(-1),
            recovery_over_limit: AtomicU64::new(0),
            timestamp_at_open: std::sync::OnceLock::new(),
        });
        s.sync_index_sort(&core.lookup());
        s.sync_merge_config();
        s.sync_stored_compression();
        {
            // Gets load the vectors kept out of the stored `_source`.
            let weak = Arc::downgrade(&core);
            // Synthetic-source residuals get their column-backed fields back.
            s.shard.set_source_patch(Some(Arc::new(move |raw: &str, vector: &dyn Fn(u16) -> Option<Vec<f32>>, cols: &dyn crate::synthetic::DocColumns| {
                let core = weak.upgrade()?;
                let lookup = core.lookup();
                if crate::synthetic::is_residual(raw) {
                    let full = crate::synthetic::rehydrate(raw, &lookup, cols)?;
                    return Some(crate::vector_source::rehydrate_stored(&full, &lookup, vector, cols).unwrap_or(full));
                }
                crate::vector_source::rehydrate_stored(raw, &lookup, vector, cols)
            })));
        }
        let finish = replay.run(|| -> Result<()> {
            s.replay(&ops)?;
            if let RecoverySource::FromFiles { manifest, .. } | RecoverySource::FromRecoveredFiles { manifest, .. } = &spec.recovery {
                if s.shard.local_checkpoint() < manifest.local_checkpoint {
                    return Err(EsError::new(500, "corrupt_index_exception", format!(
                        "recovered shard [{}][{n}] has local checkpoint [{}] below the manifest's [{}]",
                        core.name,
                        s.shard.local_checkpoint(),
                        manifest.local_checkpoint
                    )));
                }
            }
            if spec.primary_term > s.shard.primary_term() {
                s.shard.set_primary_term(spec.primary_term)?;
            }
            // A primary recovered from its own store fills the holes its
            // translog replay left (a later batch durable, an earlier one
            // never written). Replicas get theirs from the primary.
            if spec.primary && matches!(spec.recovery, RecoverySource::ExistingStore) {
                let filled = s.shard.fill_gaps_to_max_seq_no(s.shard.primary_term())?;
                if filled > 0 {
                    tracing::info!(index = %core.name, shard = n, filled, "filled seq_no gaps left by the translog with no-ops");
                }
            }
            Ok(())
        });
        if let Err(e) = finish {
            s.closed.store(true, Ordering::Release);
            s.lock.lock().take();
            if from_files {
                let _ = std::fs::remove_dir_all(&dir);
            }
            return Err(e);
        }
        s.global_checkpoint.store(gcp_from_manifest.unwrap_or_else(|| s.shard.local_checkpoint()), Ordering::Release);
        // A recovered copy is searchable as recovered (ES refreshes at the
        // end of recovery): everything its commit and translog replay
        // processed is visible, so a fleet wait for any of it returns at
        // once (ES RefreshListeners: up to the last refreshed location).
        // Nothing writes to the copy before it is registered.
        let processed = s.shard.processed_checkpoint();
        replay.run(|| s.shard.publish())?;
        s.recovery_over_limit.store(replay.over_limit(), Ordering::Relaxed);
        s.shard.note_opened_refresh(processed);
        crate::refresh::register(&s);
        Ok(s)
    }

    /// Reservations of this copy's local recovery that went past a breaker's
    /// limit (forced, never refused): proof of which breaker it ran under.
    pub fn recovery_over_limit(&self) -> u64 {
        self.recovery_over_limit.load(Ordering::Relaxed)
    }

    /// Replay translog ops recovered by `Shard::open_for_replay` (sources
    /// re-parsed against the current mapping), a bounded chunk at a time: a
    /// translog of any size replays without holding it all in memory.
    fn replay(&self, ops: &crate::translog::TranslogReplay) -> Result<()> {
        ops.for_each_chunk(crate::translog::REPLAY_CHUNK_OPS, |chunk| self.replay_chunk(chunk))
    }

    fn replay_chunk(&self, ops: crate::translog::AdmittedBatch<TranslogOp>) -> Result<()> {
        ops.try_for_each(|op| {
            match op {
                TranslogOp::Index { id, source_json, seq_no, version, routing } => {
                    let mut pd = self.core.parse_for_replica(&source_json, &id)?;
                    let source_json = pd.source.take().unwrap_or(source_json);
                    self.sync_schema()?;
                    self.shard.replay_index(id, source_json, pd.doc, pd.ignored, routing, seq_no as i64, version)?;
                }
                TranslogOp::Delete { id, seq_no, version } => self.shard.replay_delete(&id, seq_no as i64, version)?,
            }
            Ok(())
        })?;
        // The copy is not served yet: write its buffer out as it fills, so a
        // restart with gigabytes unflushed does not hold them all in
        // memtables until the final publish (forced under the replay).
        self.shard.seal_replay_if_buffer_full();
        Ok(())
    }

    /// Delete `shard-<n>/` of the index at `core` under the shard lock
    /// (fails while a copy of `n` is open).
    pub(crate) fn delete_store(core: &IndexCore, n: u32) -> Result<()> {
        let lock = ShardLock::try_acquire(&core.path, n)?.ok_or_else(|| EsError::shard_lock_failed(&core.name, &core.uuid, n, "deleting shard data"))?;
        let dir = core.path.join(format!("shard-{n}"));
        match std::fs::remove_dir_all(&dir) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        crate::hooks::sync_dir(Some(&core.path));
        drop(lock);
        Ok(())
    }

    /// Move `shard-<n>/` of the index at `core` to `dest` (same
    /// filesystem; `dest` must not exist) under the shard lock (fails while
    /// a copy of `n` is open). `false` when there is no `shard-<n>/`.
    pub(crate) fn move_store(core: &IndexCore, n: u32, dest: &Path) -> Result<bool> {
        let lock = ShardLock::try_acquire(&core.path, n)?.ok_or_else(|| EsError::shard_lock_failed(&core.name, &core.uuid, n, "moving shard data"))?;
        let dir = core.path.join(format!("shard-{n}"));
        if !dir.exists() {
            return Ok(false);
        }
        std::fs::rename(&dir, dest)?;
        crate::hooks::sync_dir(Some(&core.path));
        drop(lock);
        Ok(true)
    }

    /// The shared index core (mapping, settings, analysis).
    pub fn core(&self) -> &Arc<IndexCore> {
        &self.core
    }

    // ------------------------------------------------------- lifecycle

    /// Take an op permit (A1).
    pub(crate) fn permit(&self) -> Result<OpPermit<'_>> {
        let mut n = self.inflight.lock();
        self.core.check_attached()?;
        if self.closed.load(Ordering::Acquire) {
            return Err(EsError::shard_closed(&self.core.name, &self.core.uuid, self.id));
        }
        *n += 1;
        Ok(OpPermit(self))
    }

    /// Keep this exact physical copy alive during a compound external
    /// operation (for example, replica WAL append + raw shard apply/flush).
    /// Close/delete fences new permits and waits for this permit to drop
    /// before its directory can be reused. Do not hold it across network
    /// waits; acquire it for each local operation instead.
    pub fn operation_permit(&self) -> Result<impl Drop + '_> {
        self.permit()
    }

    /// Fails with the typed closed error once the copy is closed.
    pub fn check_usable(&self) -> Result<()> {
        self.core.check_attached()?;
        if self.closed.load(Ordering::Acquire) {
            return Err(EsError::shard_closed(&self.core.name, &self.core.uuid, self.id));
        }
        Ok(())
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// Stop new operation permits without draining existing operations or
    /// releasing the store lock. `close` must subsequently finish retirement.
    pub(crate) fn fence_new_operations(&self) {
        let _inflight = self.inflight.lock();
        self.closed.store(true, Ordering::Release);
    }

    fn check_primary(&self) -> Result<()> {
        if !self.is_primary() {
            return Err(EsError::shard_not_primary(&self.core.name, &self.core.uuid, self.id));
        }
        Ok(())
    }

    /// Fence the copy (A1), wait for in-flight operations, commit
    /// everything acknowledged (A4), release the shard lock (A3).
    /// Idempotent.
    pub fn close(&self) -> Result<()> {
        {
            let mut n = self.inflight.lock();
            if self.closed.swap(true, Ordering::AcqRel) && self.lock.lock().is_none() {
                return Ok(());
            }
            while *n > 0 {
                self.drained.wait(&mut n);
            }
        }
        let r = self.shard.flush().and_then(|_| self.shard.sync_translog());
        self.shard.close_refresh_listeners();
        // Before the shard lock goes: a copy opened next on these files
        // never meets this one's detached seal / merge output.
        self.shard.close_files();
        self.lock.lock().take();
        r
    }

    /// Fence the copy and wait for in-flight operations without
    /// committing (its files are about to be deleted); releases the lock.
    pub(crate) fn fence(&self) {
        {
            let mut n = self.inflight.lock();
            self.closed.store(true, Ordering::Release);
            while *n > 0 {
                self.drained.wait(&mut n);
            }
        }
        self.shard.close_refresh_listeners();
        self.shard.close_files();
        self.lock.lock().take();
    }

    /// Primary copy (accepts `write` / `update`)?
    pub fn is_primary(&self) -> bool {
        self.primary.load(Ordering::Acquire)
    }

    /// A2: promote this copy to primary at `new_term` (durable term bump
    /// first). `new_term` must be above the copy's current term.
    pub fn promote(&self, new_term: u64) -> Result<()> {
        let _p = self.permit()?;
        let cur = self.shard.primary_term();
        if new_term <= cur {
            return Err(EsError::new(409, "illegal_state_exception", format!("cannot promote [{}][{}] to primary term [{new_term}]: current term is [{cur}]", self.core.name, self.id)));
        }
        self.shard.set_primary_term(new_term)?;
        self.primary.store(true, Ordering::Release);
        Ok(())
    }

    /// Demote to a replica copy (writes are refused from now on).
    pub fn demote(&self) {
        self.primary.store(false, Ordering::Release);
    }

    pub fn global_checkpoint(&self) -> i64 {
        self.global_checkpoint.load(Ordering::Acquire)
    }

    /// Advance the global checkpoint (never moves backwards).
    pub fn update_global_checkpoint(&self, gcp: i64) {
        self.global_checkpoint.fetch_max(gcp, Ordering::AcqRel);
    }

    /// Add mapping fields the core learned since this copy last synced.
    pub fn sync_schema(&self) -> Result<()> {
        let g = self.core.schema_generation();
        if self.synced_schema.load(Ordering::Acquire) != g {
            let lookup = self.core.lookup();
            self.shard.sync_schema(&lookup.core_schema())?;
            self.sync_index_sort(&lookup);
            self.synced_schema.store(g, Ordering::Release);
        }
        Ok(())
    }

    /// [`IndexShard::sync_schema`] for the search path: never waits behind a
    /// background seal or an append-only batch holding the shard's seal /
    /// batch locks (see [`Shard::try_sync_schema`]); the schema then stays
    /// unsynced until a write or a later search syncs it.
    pub fn sync_schema_for_search(&self) -> Result<()> {
        let g = self.core.schema_generation();
        if self.synced_schema.load(Ordering::Acquire) != g {
            let lookup = self.core.lookup();
            if self.shard.try_sync_schema(&lookup.core_schema())? && self.try_sync_index_sort(&lookup) {
                self.synced_schema.store(g, Ordering::Release);
            }
        }
        Ok(())
    }

    /// [`IndexShard::sync_index_sort`] without blocking on the core lock;
    /// whether the core's sort is now in sync.
    fn try_sync_index_sort(&self, lookup: &crate::mapping::FieldLookup) -> bool {
        let sort = crate::indexsort::core_sort(&self.core.meta.read().settings, lookup);
        // A later reopen enables timestamp leaf ordering even for a dynamically
        // added mapping. Keep already published leaves separate until then too.
        let preserve = sort.is_none() && lookup.get("@timestamp").is_some_and(|f|
            matches!(f.ty, crate::mapping::EsType::Date | crate::mapping::EsType::DateNanos));
        let Some(core) = self.shard.core.try_read() else { return false };
        if core.index_sort() == sort.as_ref() && core.preserves_published_leaves() == preserve {
            return true;
        }
        drop(core);
        let Some(mut core) = self.shard.core.try_write() else { return false };
        core.set_index_sort(sort);
        core.set_preserve_published_leaves(preserve);
        true
    }

    /// Push the dynamic `index.merge.*` settings into the shard's merge
    /// scheduler and policy (at open, then by the refresh scheduler about
    /// once a second: dynamic like ES). An invalid value keeps the current
    /// configuration (updates are validated before they are stored).
    pub(crate) fn sync_merge_config(&self) {
        match crate::shard::MergeConfig::from_settings(|k| self.core.setting(k)) {
            Ok(mut config) => {
                config.timestamp_field = *self.timestamp_at_open.get_or_init(|| timestamp_field(&self.core.lookup()));
                self.shard.set_merge_config(config)
            }
            Err(error) => tracing::warn!(index = %self.core.name, shard = self.id, "ignoring invalid merge settings: {error}"),
        }
    }

    /// Push `index.codec` into the shard's memtables: `best_compression`
    /// (and `legacy_best_compression`) stores fields in ES's large zstd
    /// blocks, anything else in the default small LZ4 chunks. A static
    /// setting: read when the copy opens.
    pub(crate) fn sync_stored_compression(&self) {
        let codec = self.core.setting("index.codec");
        let codec = codec.as_deref().map(|c| if c.eq_ignore_ascii_case("legacy_best_compression") { "best_compression" } else { c });
        self.shard.set_stored_compression(rustic_core::codec::store::StoredCompression::from_codec(codec));
    }

    /// Push `index.sort.*` (resolved against the mapping) into the core.
    fn sync_index_sort(&self, lookup: &crate::mapping::FieldLookup) {
        let sort = crate::indexsort::core_sort(&self.core.meta.read().settings, lookup);
        // A later reopen enables timestamp leaf ordering even for a dynamically
        // added mapping. Keep already published leaves separate until then too.
        let preserve = sort.is_none() && lookup.get("@timestamp").is_some_and(|f|
            matches!(f.ty, crate::mapping::EsType::Date | crate::mapping::EsType::DateNanos));
        let core = self.shard.core.read();
        if core.index_sort() != sort.as_ref() || core.preserves_published_leaves() != preserve {
            drop(core);
            let mut core = self.shard.core.write();
            core.set_index_sort(sort);
            core.set_preserve_published_leaves(preserve);
        }
    }

    // ----------------------------------------------------------- writes

    /// Apply a batch of already routed writes (A2: primary only; hook 12
    /// retry semantics of [`WriteCond`] unchanged).
    pub fn write(&self, reqs: Vec<WriteReq>, sync: bool) -> Vec<Result<WriteResult>> {
        let _p = match self.permit() {
            Ok(p) => p,
            Err(e) => return reqs.iter().map(|_| Err(e.clone())).collect(),
        };
        if let Err(e) = self.check_primary() {
            return reqs.iter().map(|_| Err(e.clone())).collect();
        }
        self.write_inner(reqs, sync).into_iter().map(|result| self.core.attached_or(result)).collect()
    }

    /// The write path of this copy (blocks, parsing, dynamic mapping,
    /// concurrency checks, translog), for already routed requests.
    fn write_inner(&self, reqs: Vec<WriteReq>, sync: bool) -> Vec<Result<WriteResult>> {
        let profile = tracing::enabled!(target: "primary_profile", tracing::Level::DEBUG).then(std::time::Instant::now);
        let profile_rows = reqs.len();
        if let Err(e) = self.check_open() {
            return reqs.iter().map(|_| Err(e.clone())).collect();
        }
        let (idx_block, del_block) = (self.blocked_writes(false).err(), self.blocked_writes(true).err());
        if idx_block.is_some() || del_block.is_some() {
            let blocked: Vec<Option<EsError>> = reqs.iter().map(|r| if matches!(r, WriteReq::Delete { .. }) { del_block.clone() } else { idx_block.clone() }).collect();
            if blocked.iter().all(|b| b.is_some()) {
                return blocked.into_iter().map(|b| Err(b.unwrap())).collect();
            }
            if blocked.iter().any(|b| b.is_some()) {
                // Mixed batch: run the allowed ops, fail the blocked ones in place.
                let mut allowed = Vec::new();
                let mut slots = Vec::new();
                for (r, b) in reqs.into_iter().zip(blocked) {
                    match b {
                        Some(e) => slots.push(Err(e)),
                        None => {
                            slots.push(Ok(allowed.len()));
                            allowed.push(r);
                        }
                    }
                }
                let mut res: Vec<Option<Result<WriteResult>>> = self.write_inner(allowed, sync).into_iter().map(Some).collect();
                return slots.into_iter().map(|s| match s {
                    Ok(i) => res[i].take().unwrap(),
                    Err(e) => Err(e),
                }).collect();
            }
        }
        let (routing_required, source_enabled) = {
            let m = self.meta.read();
            (m.mappings.routing_required(), crate::synthetic::source_mode(settings::get_str(&m.settings, "index.mapping.source.mode"), &m.mappings, settings::get_str(&m.settings, "index.mode")) != crate::synthetic::SourceMode::Disabled)
        };
        // A join field: a child document needs its parent's routing (ES's
        // `ParentJoinFieldMapper`: "[routing] is missing for join field").
        let join = crate::fieldtypes::join::join_field(&self.lookup()).map(|f| {
            let rel = crate::fieldtypes::join::Relations::of(&f);
            (f, rel)
        });
        let limits = crate::fieldtypes::limits::Limits::from_settings(|k| self.setting(k));
        let ts_bounds = crate::tsdb::bounds(&self.meta.read().settings);
        // Auto ids: k-ordered in logsdb / columnar indices, as ES makes them.
        let k_ordered = crate::index::k_ordered_ids(crate::settings::get_str(&self.meta.read().settings, "index.mode"));
        // Phase 1: parse (parallel for big batches). In a cluster (mapping
        // hook installed), documents needing a dynamic mapping update are
        // deferred: their updates go to the master as ONE request for the
        // whole shard request, acknowledged before any of them is indexed
        // (ES `TransportShardBulkAction`; ES asks per item).
        use rayon::prelude::*;
        // New dynamic fields are collected over the whole request and applied
        // as one mapping update (through the cluster when it has a hook).
        let defer = true;
        let hooked = self.has_mapping_hook();
        let deferred: Mutex<Vec<(usize, WriteReq, Value)>> = Mutex::new(Vec::new());
        let mut prepared: Vec<Result<WriteOp>> = {
            let f = |position: usize, r: WriteReq, defer: bool| -> Result<WriteOp> {
                match r {
                    WriteReq::Index { id, source, routing, cond } => {
                        cond.validate(false)?;
                        let requested_id = if defer { id.clone() } else { None };
                        let (id, auto) = match id {
                            Some(i) => {
                                if i.is_empty() {
                                    return Err(EsError::action_validation(&["if _id is specified it must not be empty"]));
                                }
                                if i.len() > 512 {
                                    return Err(EsError::action_validation(&[&format!("id [{i}] is too long, must be no longer than 512 bytes but was: {}", i.len())]));
                                }
                                (i, cond.append_only && cond.create && cond.if_seq_no.is_none() && cond.version.is_none() && !cond.is_retry)
                            }
                            None => (if k_ordered { crate::index::auto_id_k_ordered() } else { auto_id() }, true),
                        };
                        if routing.is_some() && ts_bounds.is_some() {
                            return Err(crate::tsdb::mode_error("specifying routing", &self.name));
                        }
                        if routing_required && routing.is_none() {
                            return Err(EsError::new(400, "routing_missing_exception", format!("routing is required for [{}]/[{id}]", self.name)).with("index_uuid", "_na_").with("index", self.name.clone()));
                        }
                        let plain = crate::jsonpos::parse_plain(source.as_bytes());
                        let v: Value = match plain.map_or_else(|| crate::jsonpos::validate(source.as_bytes()).map(|_| None), |v| Ok(Some(v))) {
                            Ok(Some(v)) => v,
                            Err(e) => {
                                return Err(e.doc_failure(source.as_bytes()));
                            }
                            Ok(None) => crate::jsonpos::parse_source_value(&source).map_err(|e| EsError::document_parsing(format!("[{}:{}] failed to parse", e.line(), e.column())).caused_by(EsError::json_parse(&e)))?,
                        };
                        if routing.is_none() {
                            if let Some((field, rel)) = &join {
                                let value = v.get(field.name.as_str());
                                let name = match value {
                                    Some(Value::String(s)) => Some(s.as_str()),
                                    Some(Value::Object(o)) => o.get("name").and_then(Value::as_str),
                                    _ => None,
                                };
                                if name.is_some_and(|n| rel.parent_of(n).is_some()) {
                                    let cause = EsError::illegal_argument(format!("[routing] is missing for join field [{}]", field.name));
                                    return Err(crate::fieldtypes::index::parse_err(field, &id, value.unwrap_or(&Value::Null), cause));
                                }
                            }
                        }
                        if let Some(b) = ts_bounds {
                            crate::tsdb::check_timestamp(&v, b)?;
                            self.lookup().tsdb_fields.check_positions(&v)?;
                            let meta = self.meta.read();
                            crate::tsdb::check_routing_source(&meta.mappings, &meta.settings, &v)?;
                        }
                        let (mut pd, _) = match self.parse_source_mode(&v, &id, &limits, defer) {
                            Err(e) if e.kind == crate::index::DEFERRED_MAPPING => {
                                deferred.lock().push((position, WriteReq::Index { id: requested_id, source, routing, cond }, v));
                                return Err(e);
                            }
                            other => other?,
                        };
                        let (mut id, mut auto) = (id, auto);
                        if ts_bounds.is_some() {
                            // The mapping may have just learned dynamic dimensions: compute
                            // the ids against the updated mapping.
                            let ids = {
                                let meta = self.meta.read();
                                self.lookup().tsdb_fields.build_ids(&meta.mappings, &meta.settings, &v)?
                            };
                            if !auto && id != ids.id {
                                return Err(EsError::illegal_argument(format!("_id must be unset or set to [{}] but was [{id}] because [{}] is in time_series mode", ids.id, self.name)));
                            }
                            id = ids.id.clone();
                            auto = false;
                            // The parsed document does not depend on its
                            // `_id` (only error messages name it, and the
                            // parse above succeeded): keep it rather than
                            // parsing and synthesizing the source again.
                            crate::tsdb::push_hidden_fields(&self.lookup(), &ids, &mut pd.doc);
                        }
                        // `_source` disabled: nothing is stored, but the
                        // operation (translog, replication, recovery) keeps
                        // the original source, as ES's translog does: a
                        // replica re-parses it into the same documents
                        // (nested ones included).
                        let source = if source_enabled {
                            pd.source.take().unwrap_or(source)
                        } else {
                            pd.doc.fields.push((crate::mapping::SOURCE_FIELD, rustic_core::FieldValue::Keyword(String::new())));
                            source
                        };
                        Ok(WriteOp::Index { id, source, doc: pd.doc, ignored: pd.ignored, routing, cond, auto_id: auto })
                    }
                    WriteReq::Delete { id, cond, routing } => {
                        if routing.is_some() && ts_bounds.is_some() {
                            return Err(crate::tsdb::mode_error("specifying routing", &self.name));
                        }
                        cond.validate(true)?;
                        if ts_bounds.is_some() && !crate::tsdb::valid_id(&id) {
                            return Err(EsError::new(404, "resource_not_found_exception", format!("invalid id [{id}] for index [{}] in time_series mode", self.name)));
                        }
                        Ok(WriteOp::Delete { id, cond })
                    }
                }
            };
            // A batch fans its documents out on the shared pool while fewer
            // batches parse at once than half the cores: then the pool has
            // workers to spare for each. With more, every batch parses on its
            // own thread (as an ES write thread does), since fanning them all
            // out queued them behind the workers, which spin and yield while
            // they wait. (Parallel only when no other batch parsed left 8
            // concurrent k8s_metrics bulks each on one core, with most of a
            // 16- or 32-core host idle: a bulk's latency was its serial parse.)
            static PARSING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            static CORES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            let cores = *CORES.get_or_init(|| std::thread::available_parallelism().map_or(4, usize::from));
            let busy = PARSING.fetch_add(1, Ordering::AcqRel);
            // ES counts a document's whole engine index operation in
            // `index_time_in_millis`, its parse into fields (here also the
            // shape tessellation) included, summed over threads: each index
            // request's parse time is added to the shard's counter.
            let parse_nanos = std::sync::atomic::AtomicU64::new(0);
            let timed = |position: usize, r: WriteReq| -> Result<WriteOp> {
                let index = matches!(r, WriteReq::Index { .. });
                let started = std::time::Instant::now();
                let out = f(position, r, defer);
                if index {
                    parse_nanos.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
                out
            };
            // Concurrent batches of large documents (k8s_metrics: ~10 KB
            // each) still fan out while the pool has workers to spare;
            // batches of small ones (metricbeat: ~1 KB, 10,000 a bulk) only
            // when alone: there the per-document work is too small to pay
            // for sharing the pool (0.7x throughput when they all fanned out).
            static MIN_BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            let min_bytes = *MIN_BYTES.get_or_init(|| std::env::var("RUSTIC_PARSE_PARALLEL_MIN_DOC_BYTES").ok().and_then(|v| v.parse().ok()).unwrap_or(4096));
            let bytes: usize = reqs.iter().map(|r| match r { WriteReq::Index { source, .. } => source.len(), _ => 0 }).sum();
            let large = !reqs.is_empty() && bytes / reqs.len() >= min_bytes;
            let parallel = busy == 0 || (large && (busy + 1) * 2 <= cores);
            // Enough work to share: 64 documents, or fewer large ones (a
            // 50-document bulk of 3.5 KB metrics docs parsed serially for
            // most of its latency).
            let enough = reqs.len() >= 64 || (reqs.len() >= 8 && bytes >= 64 << 10);
            let out: Vec<Result<WriteOp>> = if enough && parallel {
                reqs.into_par_iter().enumerate().map(|(position, r)| timed(position, r)).collect()
            } else {
                reqs.into_iter().enumerate().map(|(position, r)| timed(position, r)).collect()
            };
            PARSING.fetch_sub(1, Ordering::AcqRel);
            crate::shard::Counters::add_time(&self.shard.counters.index_time_nanos, std::time::Duration::from_nanos(parse_nanos.load(Ordering::Relaxed)));
            let mut out = out;
            let mut deferred = std::mem::take(&mut *deferred.lock());
            // Request order (the parallel parse pushes in completion
            // order): within one request, the first item to introduce a
            // field decides its type, as in ES.
            deferred.sort_by_key(|(position, _, _)| *position);
            // Unavailable (or rejected for load): the deferred items fail at
            // once rather than each retrying on its own.
            let unavailable = |error: &EsError| error.status == 503 || error.status == 429 || error.kind.contains("timeout") || error.kind.contains("master_not_discovered");
            if !deferred.is_empty() {
                let items = deferred.iter().map(|(_, r, v)| {
                    let id = match r { WriteReq::Index { id, .. } => id.as_deref().unwrap_or(""), WriteReq::Delete { id, .. } => id.as_str() };
                    (v, id)
                });
                let accepted = if hooked {
                    match &self.propose_dynamic_mapping(items, &limits) {
                        Some((base, mapping)) => self.request_dynamic_mapping(base, mapping),
                        None => Ok(()),
                    }
                } else {
                    self.apply_dynamic_mapping_batch(items, &limits).map(|_| ())
                };
                match accepted {
                    // The master could not be reached in time: only the items
                    // needing the new fields fail (bounded wait, ES error).
                    Err(error) if unavailable(&error) => {
                        for (position, _, _) in deferred { out[position] = Err(error.clone()); }
                    }
                    // Accepted: the fields are known now. Rejected (e.g. a
                    // conflicting type another request introduced first):
                    // each item resolves on its own against the master's
                    // mapping (per-item update, ES's error per loser).
                    // Bounded overall: once the master is unavailable for one
                    // item, or the budget is spent, the rest fail at once
                    // instead of each waiting out its own timeout.
                    _ => {
                        let started = std::time::Instant::now();
                        let mut gave_up: Option<EsError> = None;
                        for (position, r, _) in deferred {
                            if gave_up.is_none() && started.elapsed() >= DYNAMIC_FALLBACK_BUDGET {
                                gave_up = Some(EsError::new(503, "process_cluster_event_timeout_exception", format!(
                                    "failed to process cluster event (put-mapping [{}]) within {}ms (timed out while waiting for a dynamic mapping update)", self.name, DYNAMIC_FALLBACK_BUDGET.as_millis())));
                            }
                            if let Some(error) = &gave_up {
                                out[position] = Err(error.clone());
                                continue;
                            }
                            let item = f(position, r, false);
                            if let Err(error) = &item {
                                if unavailable(error) {
                                    gave_up = Some(error.clone());
                                }
                            }
                            out[position] = item;
                        }
                    }
                }
            }
            out
        };
        // Phase 2: apply valid ops in order.
        let mut ops = Vec::new();
        let mut slots = Vec::with_capacity(prepared.len());
        // Time series creates: their sources, to tag a version conflict with `[_tsid@timestamp]`.
        let mut ts_creates: Vec<(usize, String)> = Vec::new();
        for p in prepared {
            match p {
                Ok(op) => {
                    if ts_bounds.is_some() {
                        if let WriteOp::Index { source, cond, .. } = &op {
                            if cond.create {
                                ts_creates.push((ops.len(), source.clone()));
                            }
                        }
                    }
                    slots.push(Ok(ops.len()));
                    ops.push(op);
                }
                Err(e) => slots.push(Err(e)),
            }
        }
        // New dynamic fields reach this copy's schema before its docs do.
        if let Err(error) = self.sync_schema() {
            // Preserve item-level parse failures and fail otherwise valid rows
            // before applying any writes under an incompatible vector schema.
            return slots.into_iter().map(|slot| Err(match slot {
                Ok(_) => error.clone(), Err(original) => original,
            })).collect();
        }
        let prepared_us = profile.map(|t| t.elapsed().as_micros() as u64).unwrap_or(0);
        let auto_rows = if profile.is_some() { ops.iter().filter(|o| matches!(o, WriteOp::Index { auto_id: true, .. })).count() } else { 0 };
        let mut applied = self.shard.apply(&self.name, &self.uuid, ops, sync).into_iter().map(Some).collect::<Vec<_>>();
        if let Some(t) = profile {
            tracing::debug!(target: "primary_profile", index=%self.name, shard=self.id, rows=profile_rows, auto_rows,
                prepared_us, apply_us=(t.elapsed().as_micros() as u64).saturating_sub(prepared_us), "primary write stages");
        }
        for (i, src) in ts_creates {
            if let Some(Err(e)) = &mut applied[i] {
                if e.kind == "version_conflict_engine_exception" && e.reason.contains("document already exists") {
                    let tag = serde_json::from_str::<Value>(&src).ok().and_then(|v| {
                        let ts = v.get("@timestamp").and_then(Value::as_str)?.to_string();
                        let meta = self.meta.read();
                        let ids = self.lookup().tsdb_fields.build_ids(&meta.mappings, &meta.settings, &v).ok()?;
                        Some(format!("[{}@{ts}]", ids.tsid))
                    });
                    if let (Some(tag), Some(pos)) = (tag, e.reason.find("]: ")) {
                        e.reason.insert_str(pos + 1, &tag);
                    }
                }
            }
        }
        slots.into_iter().map(|s| match s {
            Ok(i) => applied[i].take().unwrap(),
            Err(e) => Err(e),
        }).collect()
    }


    /// Whether this index keeps semantic_text inference results in `_source`.
    fn semantic_legacy(&self) -> bool {
        crate::settings::get_str(&self.meta.read().settings, "index.mapping.semantic_text.use_legacy_format") == Some("true")
    }

    /// `_update`: partial doc / upsert / doc_as_upsert / script with
    /// retries (A2: primary copies only).
    pub fn update(&self, id: &str, body: &Value, routing: Option<String>, retry_on_conflict: usize, cond: WriteCond) -> Result<(WriteResult, Option<String>)> {
        self.update_source(id,body,None,routing,retry_on_conflict,cond)
    }
    pub fn update_source(&self,id:&str,body:&Value,raw:Option<&crate::source_json::Source>,routing:Option<String>,retry_on_conflict:usize,cond:WriteCond)->Result<(WriteResult,Option<String>)>{
        let _p = self.permit()?;
        self.check_primary()?;
        self.check_open()?;
        if crate::tsdb::is_time_series(&self.meta.read().settings) {
            return Err(crate::tsdb::mode_error("update", &self.name));
        }
        let o = body.as_object().ok_or_else(|| EsError::action_validation(&["script or doc is missing"]))?;
        // `TransportUpdateAction.deleteInferenceResults`: the legacy
        // semantic_text format cannot re-run inference after a script.
        if o.contains_key("script") && self.semantic_legacy() && self.lookup().fields.values().any(|f| f.declared.is_semantic()) {
            return Err(EsError::new(400, "status_exception", "Cannot apply update with a script on indices that contain inference field(s)"));
        }
        // Hook 12: a retried update that was already applied must not run
        // again (a scripted update would apply twice).
        if let (true, Some(m)) = (cond.is_retry, cond.coordinator_id.as_deref()) {
            if let Some(orig) = self.shard.retry_lookup(id, m) {
                let src = self.shard.get(id)?.and_then(|g| g.source);
                return self.core.attached_or(Ok((orig, src)));
            }
        }
        let marker = cond.coordinator_id.clone();
        let r = self.update_inner(id, o, body, raw, routing, retry_on_conflict, cond);
        if let (Ok((w, _)), Some(m)) = (&r, marker) {
            if w.result != "noop" {
                self.shard.retry_record(&m, w, w.result == "deleted");
            }
        }
        self.core.attached_or(r)
    }

    fn update_inner(&self, id: &str, o: &serde_json::Map<String, Value>, _body: &Value, raw:Option<&crate::source_json::Source>, routing: Option<String>, retry_on_conflict: usize, cond: WriteCond) -> Result<(WriteResult, Option<String>)> {
        if o.contains_key("script") {
            return crate::script::update::scripted_update(self, id, o, raw, routing, retry_on_conflict, cond);
        }
        let doc = o.get("doc");
        let upsert = o.get("upsert");
        let doc_as_upsert = o.get("doc_as_upsert").and_then(Value::as_bool).unwrap_or(false);
        let detect_noop = o.get("detect_noop").and_then(Value::as_bool).unwrap_or(true);
        if doc.is_none() && upsert.is_none() {
            return Err(EsError::action_validation(&["script or doc is missing"]));
        }
        let mut attempt = 0;
        loop {
            let cur = self.shard.get(id)?;
            let (new_src, c, created) = match &cur {
                Some(g) => {
                    let stored = g.source.as_deref().unwrap_or("{}");
                    let parsed=serde_json::from_str::<Value>(stored);
                    let (src, unchanged) = match parsed {
                        Ok(mut src) if raw.is_none() => {
                            let before = src.clone();
                            if let Some(d) = doc {
                                merge_json(&mut src, d);
                                if self.semantic_legacy() {
                                    crate::inference::semantic::strip_stale_legacy(&self.lookup(), d, &mut src);
                                }
                            }
                            let unchanged = src == before;
                            (src.to_string(), unchanged)
                        }
                        _ => {
                            // Semantic transformations still need their own lossless path.
                            if self.semantic_legacy() { return Err(EsError::illegal_argument("lossless semantic source updates are not supported")); }
                            let mut src = crate::source_json::Source::parse(stored)?;
                            let before = src.clone();
                            if let Some(d)=raw.and_then(|r|r.get("doc")){src.merge(d);}
                            else if let Some(d) = doc { src.merge(&crate::source_json::Source::from_value(d)); }
                            let text = src.text();
                            let unchanged = src.equivalent(&before);
                            (text, unchanged)
                        }
                    };
                    if detect_noop && unchanged {
                        self.shard.counters.noop_update_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        return Ok((WriteResult { id: id.to_string(), version: g.version, seq_no: g.seq_no, primary_term: g.primary_term, result: "noop" }, Some(src)));
                    }
                    let mut c = cond.clone();
                    if c.if_seq_no.is_none() {
                        c.if_seq_no = Some(g.seq_no);
                        c.if_primary_term = Some(self.shard.doc_primary_term(id).unwrap_or_else(|| self.shard.primary_term()));
                    }
                    (src, c, false)
                }
                None => {
                    if cond.if_seq_no.is_some() {
                        return Err(EsError::document_missing(&self.name, &self.uuid, id));
                    }
                    let src = if doc_as_upsert {
                        doc.cloned().unwrap_or(json!({}))
                    } else if let Some(u) = upsert {
                        u.clone()
                    } else {
                        return Err(EsError::document_missing(&self.name, &self.uuid, id));
                    };
                    let src=raw.and_then(|r|r.get(if doc_as_upsert{"doc"}else{"upsert"})).map(|s|s.text()).unwrap_or_else(||src.to_string());
                    (src, WriteCond { create: true, ..Default::default() }, true)
                }
            };
            let s = new_src.clone();
            let r = self.write(vec![WriteReq::Index { id: Some(id.to_string()), source: s, routing: routing.clone(), cond: c }], true).remove(0);
            match r {
                Ok(mut w) => {
                    if created {
                        w.result = "created";
                    }
                    return Ok((w, Some(new_src)));
                }
                Err(e) if e.status == 409 && attempt < retry_on_conflict && cond.if_seq_no.is_none() => {
                    attempt += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Realtime GET on this copy: `_routing: {required: true}` is
    /// enforced; `realtime=false` on an index with refresh disabled only
    /// sees refreshed documents.
    pub fn get(&self, id: &str, routing: Option<&str>, realtime: bool) -> Result<Option<GetResult>> {
        let _p = self.permit()?;
        self.check_open()?;
        self.check_read_block()?;
        if routing.is_none() && self.meta.read().mappings.routing_required() {
            return Err(EsError::new(400, "routing_missing_exception", format!("routing is required for [{}]/[{id}]", self.name))
                .with("index_uuid", "_na_")
                .with("index", self.name.clone()));
        }
        if routing.is_some() && crate::tsdb::is_time_series(&self.meta.read().settings) {
            return Err(crate::tsdb::mode_error("specifying routing", &self.name));
        }
        self.shard.get_mode(id, realtime)
    }

    // ---------------------------------------------------- replica apply

    /// Hook 3 on this copy (see [`IndexService::apply_replica_ops`]); allowed
    /// on primaries and replicas, fenced by `request_term`.
    pub fn apply_replica_ops(&self, request_term: u64, ops: Vec<ChangeOp>, durability: Durability) -> Result<ReplicaApplyResult> {
        let n = ops.len();
        self.apply_replica_ops_marked(request_term, ops, vec![None; n], durability)
    }

    /// Hook 12 replica apply with the primary's coordinator markers.
    pub fn apply_replica_ops_marked(&self, request_term: u64, ops: Vec<ChangeOp>, markers: Vec<Option<String>>, durability: Durability) -> Result<ReplicaApplyResult> {
        if markers.len() != ops.len() {
            return Err(EsError::illegal_argument("markers must have one entry per op"));
        }
        let ops: Vec<(ChangeOp, Option<String>)> = ops.into_iter().zip(markers).collect();
        let parse = |(op, marker): (ChangeOp, Option<String>)| -> Result<PreparedReplicaOp> {
            let parsed = match &op {
                ChangeOp::Index { id, source_json, .. } => {
                    let pd = self.core.parse_for_replica(source_json, id)?;
                    Some((pd.doc, pd.ignored))
                }
                _ => None,
            };
            Ok(PreparedReplicaOp { op, parsed, marker })
        };
        // Parsed before the operation permit is taken, in parallel on a pool
        // of its own (never the global one, never stealing): a parse that
        // waited on rayon work while holding the permit could steal a job
        // that waits for every permit (a relocation's or a block's), a
        // deferred hold the locking objective rules out.
        let prepared: Vec<PreparedReplicaOp> = if ops.len() >= 64 {
            crate::shard::on_replica_parse_pool(|| {
                use rayon::prelude::*;
                ops.into_par_iter().map(parse).collect::<Result<_>>()
            })?
        } else {
            ops.into_iter().map(parse).collect::<Result<_>>()?
        };
        let _p = self.permit()?;
        self.sync_schema()?;
        self.shard.apply_replica(request_term, prepared, durability)
    }

    /// Hook 12: the original result of the last op on `id` that carried
    /// `coordinator_id`.
    pub fn retried_result(&self, id: &str, coordinator_id: &str) -> Option<WriteResult> {
        self.shard.retry_lookup(id, coordinator_id)
    }

    /// Replay one raw `_source` through this copy's full index path:
    /// parse against the current mapping (applying dynamic mapping updates
    /// like the primary write path), sync the schema, then
    /// [`Shard::replay_index`] with the parsed doc and the source to store
    /// (the canonical form on synthetic-source indices). The public
    /// counterpart of the translog-replay path, for embedders that replay
    /// documents themselves (e.g. a replication layer).
    pub fn replay_source(&self, id: &str, source_json: &str, routing: Option<String>, seq_no: i64, version: u64) -> Result<()> {
        let _p = self.permit()?;
        self.check_open()?;
        let pd = self.core.parse_for_replica(source_json, id)?;
        let source = pd.source.clone().unwrap_or_else(|| source_json.to_string());
        self.sync_schema()?;
        self.shard.replay_index(id.to_string(), source, pd.doc, pd.ignored, routing, seq_no, version)
    }

    /// Apply a document prepared from a stable mapping snapshot. Replication
    /// still owns WAL durability, staleness checks and ordered application.
    pub fn replay_prepared(&self, id: &str, source_json: &str, pd: crate::document::ParsedDoc,
        routing: Option<String>, seq_no: i64, version: u64) -> Result<()> {
        let _p = self.permit()?;
        self.check_open()?;
        let source = pd.source.unwrap_or_else(|| source_json.to_string());
        self.sync_schema()?;
        self.shard.replay_index(id.to_string(), source, pd.doc, pd.ignored, routing, seq_no, version)
    }

    /// Replicated index / delete ops applied concurrently with other
    /// batches ([`Shard::replay_staged`]): staleness by seq_no per id, so
    /// batches may apply in any order. Index ops parse here unless
    /// `prepared` (a document prepared from a stable mapping snapshot).
    /// Returns, per op, whether it applied. The caller owns WAL durability
    /// and replication fencing.
    pub fn replay_ops(&self, ops: Vec<ReplayOp<'_>>) -> Result<crate::translog::AdmittedBatch<bool>> {
        let _p = self.permit()?;
        self.check_open()?;
        let slots = ops.len().checked_mul(std::mem::size_of::<crate::shard::StagedReplay>())
            .ok_or_else(|| EsError::new(429, "circuit_breaking_exception", "replica staged slots overflow"))?;
        let mut staged_memory = crate::memory::breakers().reserve_request(slots as u64, "replica staged operations")?;
        let mut staged = Vec::new();
        staged.try_reserve_exact(ops.len()).map_err(|e| EsError::new(429, "circuit_breaking_exception", format!("replica staged allocation: {e}")))?;
        fn copy_text(text: &str, memory: &mut crate::memory::Reservation) -> Result<String> {
            memory.grow(text.len() as u64, "replica staged text")?;
            let mut out = String::new();
            out.try_reserve_exact(text.len()).map_err(|e| EsError::new(429, "circuit_breaking_exception", format!("replica staged text: {e}")))?;
            out.push_str(text); Ok(out)
        }
        let mut failed = None;
        for op in ops {
            staged.push(match op {
                ReplayOp::Index { id, source, routing, seq_no, version, prepared } => {
                    // A stale op is ignored before parsing (its source may
                    // not parse against the current mapping).
                    if self.shard.known_seq_no(id).is_some_and(|known| known >= seq_no) {
                        crate::shard::StagedReplay::Skip { seq_no }
                    } else {
                        let pd = match prepared {
                            Some(pd) => pd,
                            None => match self.core.parse_for_replica(source, id) {
                                Ok(pd) => pd,
                                // Apply what precedes the failing op, in
                                // order, then fail (as one-by-one replay).
                                Err(error) => { failed = Some(error); break; }
                            },
                        };
                        let source = match pd.source { Some(source) => source, None => copy_text(source, &mut staged_memory)? };
                        crate::shard::StagedReplay::Index { id: copy_text(id, &mut staged_memory)?, source, doc: pd.doc, ignored: pd.ignored,
                            routing: routing.map(|r| copy_text(r, &mut staged_memory)).transpose()?, seq_no, version }
                    }
                }
                ReplayOp::Delete { id, seq_no, version } => crate::shard::StagedReplay::Delete { id: copy_text(id, &mut staged_memory)?, seq_no, version },
            });
        }
        self.sync_schema()?;
        let applied = self.shard.replay_staged(staged)?;
        match failed {
            Some(error) => Err(error),
            None => Ok(applied),
        }
    }

    /// The caller owns WAL durability and replication fencing. Unsupported or
    /// non-new batches return false without consuming prepared documents.
    pub fn try_replay_new_batch(&self, ops: &[crate::shard::ReplayIndex<'_>], prepared: &mut Vec<crate::document::ParsedDoc>) -> Result<bool> {
        let _p = self.permit()?;
        self.check_open()?;
        self.sync_schema()?;
        self.shard.try_replay_new_batch(ops, prepared)
    }

    // ------------------------------------------------------- file sets

    /// Hook 2 for this copy: pin the shard's last commit (+ translog and
    /// history sidecar with `include_translog`). Paths are relative to the
    /// shard directory, so `write_to_dir(staged)` + `RecoverySource::FromFiles
    /// { staged, manifest }` recreates the copy. Holds a retention lock on
    /// the history above the commit while the set lives.
    pub fn snapshot_files(&self, include_translog: bool) -> Result<IndexFileSet> {
        // The permit only while pinning: checksumming (`build`) reads every
        // file and must not hold off this copy's close.
        let (mut sources, [committed, commit_cp, local_checkpoint, max_seq_no], primary_term, lock) = {
            let _p = self.permit()?;
            self.shard.capture_files(include_translog, "")?
        };
        sources.reserve_entry(0,self.name.len().saturating_add(self.uuid.len()))?;
        let manifest = IndexManifest {
            index: self.name.clone(),
            uuid: self.uuid.clone(),
            files: Vec::new(),
            committed_seq_no: committed,
            local_checkpoint: if include_translog { local_checkpoint } else { commit_cp },
            max_seq_no: if include_translog { max_seq_no } else { committed },
            primary_term,
            includes_translog: include_translog,
        };
        IndexFileSet::build(manifest, sources, Some(lock))
    }

    /// The index-level file set (`index_meta.json` + `shard-<n>/...`) of the
    /// deprecated [`IndexService::snapshot_files`].
    pub(crate) fn snapshot_index_files(&self, include_translog: bool) -> Result<IndexFileSet> {
        // The metadata join and decimal shard prefix are independent temporary
        // producers, before capture_files starts owning its source registry.
        // Joining may keep the cloned base and its replacement allocation
        // simultaneously; cover both plus the bounded decimal shard prefix.
        let path_bytes = self.path().as_os_str().as_encoded_bytes().len()
            .checked_add(META_FILE.len()).and_then(|bytes| bytes.checked_add(1))
            .and_then(|bytes| bytes.checked_mul(3)).and_then(|bytes| bytes.checked_add(64))
            .ok_or_else(|| EsError::new(429, "circuit_breaking_exception", "snapshot path size overflow"))?;
        let _paths = rustic_core::admission::shared().reserve(path_bytes as u64, "index snapshot path construction")?;
        let _p = self.permit()?;
        let meta = crate::hooks::files::read_metadata_file(&self.path().join(META_FILE))?;
        let (mut sources, [committed, commit_cp, local_checkpoint, max_seq_no], primary_term, lock) = self.shard.capture_files(include_translog, &format!("shard-{}/", self.id))?;
        sources.reserve_entry(META_FILE.len(),0)?;
        sources.values.insert(META_FILE.to_string(), FileSource::Bytes(meta));
        sources.reserve_entry(0,self.name.len().saturating_add(self.uuid.len()))?;
        let manifest = IndexManifest {
            index: self.name.clone(),
            uuid: self.uuid.clone(),
            files: Vec::new(),
            committed_seq_no: committed,
            local_checkpoint: if include_translog { local_checkpoint } else { commit_cp },
            max_seq_no: if include_translog { max_seq_no } else { committed },
            primary_term,
            includes_translog: include_translog,
        };
        IndexFileSet::build(manifest, sources, Some(lock))
    }

    // ------------------------------------------------------------ stats

    /// Live and deleted documents of this copy. Like ES (which counts on
    /// its internal reader, not subject to external refreshes), every
    /// write so far counts, refreshed or not.
    ///
    /// Waits for the core read lock only briefly (writers, refreshes and
    /// maintenance installs hold the core write lock for short swaps; long
    /// work runs detached): only if it stays busy for that long do the
    /// counts come from the published reader view (as of the last refresh)
    /// instead.
    /// Live and deleted documents as searches see them (ES `docs.count`
    /// is refresh-based: writes count once a refresh made them visible).
    /// Reads the published view: never waits on the core or a write.
    pub fn doc_count(&self) -> (u64, u64) {
        let view = self.shard.reader_view();
        let (live, total) = (view.num_live_docs(), view.num_docs());
        // ES counts docs over its internal reader, which flushes and merges
        // update without an external refresh: a search-idle index whose
        // merges installed after its last refresh still counts every
        // sealed doc (the published view alone missed the merged ones).
        // Only when the core is free: stats never wait for a flush, merge
        // install or write batch holding it (the locking objective); then
        // the published view answers, as it always did.
        let (sealed_live, sealed_total) = match self.shard.core.try_read() {
            // Searchable and hidden (sealed, not yet published) segments.
            Some(core) => core.all_leaves().iter().fold((0u64, 0u64), |(l, t), leaf| match leaf {
                rustic_core::query::LeafReader::Segment(s) => (l + s.num_live_docs() as u64, t + s.num_docs as u64),
                _ => (l, t),
            }),
            None => (0, 0),
        };
        if sealed_live > live {
            (sealed_live, sealed_total - sealed_live)
        } else {
            (live, total - live)
        }
    }

    /// Live and deleted documents over every completed write, refreshed or
    /// not (internal callers needing realtime counts): completed writes
    /// join the index first (the check-in a refresh does).
    pub fn doc_count_realtime(&self) -> (u64, u64) {
        self.shard.checkin_completed_writes(None);
        let core = self.shard.core.read();
        let (mut live, mut total) = (0u64, 0u64);
        for l in core.all_leaves() {
            live += l.num_live_docs_latest() as u64;
            total += l.num_docs() as u64;
        }
        (live, total - live)
    }

    pub fn stats(&self) -> ShardStats {
        let (docs, deleted) = self.doc_count();
        let write = self.shard.write_stats();
        ShardStats {
            shard: self.id,
            allocation_id: self.allocation_id.clone(),
            primary: self.is_primary(),
            docs,
            deleted_docs: deleted,
            store_size: self.shard.store_size(),
            // Closed memtables count as the segments they seal into; read
            // from the published view (no core lock).
            segments: self.shard.reader_view().segments_for_stats(),
            translog_size: write.translog_bytes,
            uncommitted_ops: write.uncommitted_ops,
            max_seq_no: write.max_seq_no,
            local_checkpoint: write.local_checkpoint,
            global_checkpoint: self.global_checkpoint(),
            primary_term: self.shard.primary_term(),
        }
    }

    // ------------------------------------------- local segment surgery

    /// Live root documents of this copy: (`_id`, `_routing`, `_source`,
    /// `_seq_no`, `_version`, address).
    fn live_docs(&self) -> Result<Vec<(String, Option<String>, String, i64, u64, rustic_core::types::DocAddress)>> {
        use crate::mapping::{ID_FIELD, ROUTING_FIELD, SEQ_NO_FIELD, SOURCE_FIELD, VERSION_FIELD};
        // Every completed write, pooled ones included (copies, splits).
        self.shard.checkin_completed_writes(None);
        let core = self.shard.core.read();
        let mut out = Vec::new();
        for (li, leaf) in core.all_leaves().iter().enumerate() {
            for d in 0..leaf.num_docs() {
                if leaf.is_deleted_latest(d) || !leaf.is_root(d) {
                    continue;
                }
                let Some(seq) = leaf.i64_value(SEQ_NO_FIELD, d) else { continue };
                let addr = rustic_core::types::DocAddress::new(li as u32, d);
                let doc = core.get_doc(&addr)?;
                let Some(id) = doc.get(ID_FIELD).and_then(|v| v.as_text()).map(str::to_string) else { continue };
                let source = doc.get(SOURCE_FIELD).and_then(|v| v.as_text()).map(|raw| crate::vector_source::load(raw, &self.core.lookup(), leaf, d).into_owned()).unwrap_or_default();
                let routing = doc.get(ROUTING_FIELD).and_then(|v| v.as_text()).map(str::to_string);
                let version = leaf.i64_value(VERSION_FIELD, d).unwrap_or(1).max(1) as u64;
                out.push((id, routing, source, seq, version, addr));
            }
        }
        Ok(out)
    }

    /// Re-index into this copy every live document of `src` for which
    /// `keep(_id, _routing)` holds, keeping `_seq_no` / `_version` /
    /// `_routing` (shrink, emulated-layout migration). Sources are
    /// re-parsed against this copy's mapping.
    pub(crate) fn copy_docs_from(&self, src: &IndexShard, keep: &dyn Fn(&str, Option<&str>) -> bool) -> Result<usize> {
        self.shard.files_changed();
        let mut n = 0;
        for (id, routing, source, seq, version, _) in src.live_docs()? {
            if !keep(&id, routing.as_deref()) {
                continue;
            }
            let mut pd = self.core.parse_for_replica(&source, &id)?;
            let source = pd.source.take().unwrap_or(source);
            self.sync_schema()?;
            self.shard.replay_index(id, source, pd.doc, pd.ignored, routing, seq, version)?;
            n += 1;
        }
        Ok(n)
    }

    /// Delete (from the segments, without translog ops) every live
    /// document for which `keep(_id, _routing)` does not hold (split's
    /// shard-splitting step). Commit afterwards.
    pub(crate) fn retain_docs(&self, keep: &dyn Fn(&str, Option<&str>) -> bool) -> Result<usize> {
        self.shard.files_changed();
        let drop: Vec<rustic_core::types::DocAddress> = self.live_docs()?.into_iter().filter(|(id, r, ..)| !keep(id, r.as_deref())).map(|d| d.5).collect();
        let mut core = self.shard.core.write();
        for a in &drop {
            core.delete_doc(a)?;
        }
        // Sorted publication may need to seal pending writers. Use the shard
        // protocol so it coordinates with detached seals and releases the core
        // lock during encoding, just like an ordinary refresh.
        std::mem::drop(core);
        self.shard.publish()?;
        Ok(drop.len())
    }
}

impl IndexService {
    /// An ephemeral service exposing exactly `shard` of its index (searcher
    /// handles / search targets over one copy). Never persisted; refuses
    /// whole-index operations (write / get / update through the service).
    pub(crate) fn view_of(shard: &Arc<IndexShard>) -> Arc<IndexService> {
        let n = shard_count(&shard.core.meta.read().settings).max(shard.id + 1);
        let mut slots = vec![None; n as usize];
        slots[shard.id as usize] = Some(shard.clone());
        Arc::new(IndexService { core: shard.core.clone(), preparation_epochs: (0..slots.len()).map(|_| AtomicU64::new(0)).collect(), shards: RwLock::new(slots), view: true })
    }
}

#[cfg(test)]
mod replay_source_tests {
    use super::*;

    #[test]
    fn replay_source_indexes_like_the_primary_write_path() {
        let tmp = tempfile::tempdir().unwrap();
        let node = crate::node::Node::open(tmp.path()).unwrap();
        let svc = node.create_index("rs", &json!({})).unwrap();
        let shard = svc.shard(0).unwrap();
        // A dynamic field: the replay must learn it into the mapping.
        shard.replay_source("a", r#"{"n":1,"s":"hello"}"#, None, 0, 1).unwrap();
        let g = shard.get("a", None, true).unwrap().unwrap();
        assert_eq!(g.version, 1);
        // Searches see it after a refresh (realtime get above did not).
        shard.shard.refresh().unwrap();
        let mut target = crate::search::Target::new(svc.clone());
        target.shard = Some(shard.clone());
        let search = |q: Value| crate::search::execute(&[target.clone()], &crate::search::SearchRequest { query: Some(q), ..Default::default() }, &crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None }).unwrap();
        assert_eq!(search(json!({"match_all": {}})).total, 1);
        // A term query on the dynamically learned field: postings exist.
        assert_eq!(search(json!({"term": {"s": "hello"}})).total, 1);
        // A later replay updates the doc in place.
        shard.replay_source("a", r#"{"n":2,"s":"world"}"#, None, 1, 2).unwrap();
        let g = shard.get("a", None, true).unwrap().unwrap();
        assert_eq!(g.version, 2);
        assert_eq!(serde_json::from_str::<Value>(g.source.as_deref().unwrap()).unwrap()["n"], 2);
    }
}

#[cfg(test)]
#[path = "index_shard_tests.rs"]
mod tests;
