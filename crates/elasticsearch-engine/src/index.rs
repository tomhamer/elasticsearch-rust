//! One ES index: metadata (settings, mappings, aliases, state) shared by
//! its logical shards ([`IndexCore`]) plus the local shard copies
//! ([`IndexShard`], `indices/<uuid>/shard-<n>/`). The core owns source
//! parsing and dynamic mapping updates; each shard copy owns its write
//! path (concurrency checks, translog, updates/upserts) and recovery.
//! [`IndexService`] routes single-document operations to their shard
//! (murmur3, `elasticsearch-routing`) and fans whole-index operations out
//! over the local shards.

use crate::analysis::AnalysisRegistry;
use crate::error::{EsError, Result};
use crate::mapping::{FieldLookup, Mappings};
use crate::settings::{self, FlatSettings};
use crate::shard::{GetResult, Shard, WriteCond, WriteOp, WriteResult};
use crate::translog::TranslogOp;
use parking_lot::{Mutex, RwLock};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

const META_FILE: &str = "index_meta.json";
/// Present while an index written by the single-shard emulation is being
/// migrated to real shards.
const MIGRATION_MARKER: &str = "shard_migration.json";

// Hooks 2/3/8 (file sets, replica apply, public parse): `hooks/index_ext.rs`.
#[path = "hooks/index_ext.rs"]
mod hooks_ext;

// Logical shards: one local copy of shard `n` (`index_shard.rs`).
#[path = "index_shard.rs"]
mod index_shard;
pub use index_shard::{IndexShard, RecoverySource, ShardOpenSpec, ShardStats};

#[derive(Clone)]
pub struct IndexMeta {
    pub settings: FlatSettings,
    pub mappings: Mappings,
    /// alias name → alias definition (`filter`, `routing`, `is_write_index`, ...).
    pub aliases: BTreeMap<String, Value>,
    pub open: bool,
}

/// Error kind of a parse whose dynamic mapping update is deferred to its
/// shard request (see `IndexCore::parse_source_mode`); never returned to a
/// client.
pub(crate) const DEFERRED_MAPPING: &str = "deferred_dynamic_mapping_update";

/// Longest a `refresh=wait_for` listener waits before forcing a refresh
/// (a safety bound: refreshes normally run every `index.refresh_interval`).
const WAIT_FOR_REFRESH_MAX: std::time::Duration = std::time::Duration::from_secs(30);

/// One replicated op for [`IndexShard::replay_ops`].
pub enum ReplayOp<'a> {
    Index { id: &'a str, source: &'a str, routing: Option<&'a str>, seq_no: i64, version: u64, prepared: Option<crate::document::ParsedDoc> },
    Delete { id: &'a str, seq_no: i64, version: u64 },
}

/// One write against the index.
#[derive(Clone)]
pub enum WriteReq {
    Index { id: Option<String>, source: String, routing: Option<String>, cond: WriteCond },
    Delete { id: String, routing: Option<String>, cond: WriteCond },
}

/// Node-level interception point for dynamic mapping updates (A6). A
/// cluster node installs one ([`crate::node::Node::set_mapping_update_hook`])
/// so the master acknowledges a mapping before the write that introduced it
/// is acknowledged. Without a hook the update is applied locally.
/// Per-request authorization of dynamic mapping updates (ES runs the
/// `indices:admin/mapping/auto_put` action as the writing user): called with
/// the concrete index name before a document's dynamic update is applied.
pub type MappingUpdateAuthz = Arc<dyn Fn(&str) -> Result<()> + Send + Sync>;

thread_local! {
    static MAPPING_AUTHZ: std::cell::RefCell<Option<MappingUpdateAuthz>> = const { std::cell::RefCell::new(None) };
}

/// Restores the previous [`MappingUpdateAuthz`] when dropped.
pub struct MappingAuthzGuard(Option<MappingUpdateAuthz>);

impl Drop for MappingAuthzGuard {
    fn drop(&mut self) {
        let prev = self.0.take();
        MAPPING_AUTHZ.with(|c| *c.borrow_mut() = prev);
    }
}

/// Make `authz` the dynamic-mapping authorizer on this thread until the
/// guard drops.
pub fn enter_mapping_authz(authz: Option<MappingUpdateAuthz>) -> MappingAuthzGuard {
    MappingAuthzGuard(MAPPING_AUTHZ.with(|c| c.replace(authz)))
}

pub fn current_mapping_authz() -> Option<MappingUpdateAuthz> {
    MAPPING_AUTHZ.with(|c| c.borrow().clone())
}

pub trait MappingUpdateHook: Send + Sync {
    /// `mappings` is the index's complete mapping (ES JSON, as `GET
    /// _mapping` renders it) including the dynamic update. Return `Ok` once
    /// the update is accepted (e.g. acked by the master); the engine then
    /// applies it to the shared index core, at most once (it re-checks under
    /// the mapping lock, so an update that already arrived through
    /// [`IndexService::apply_metadata`] is not applied twice). An error fails
    /// the write that needed the update.
    fn mapping_update(&self, index: &str, uuid: &str, mappings: &Value) -> Result<()>;

    /// [`MappingUpdateHook::mapping_update`] from the copy `core` itself (a
    /// host holding several copies of one index, e.g. one per shard, tells
    /// which one asks). Defaults to the name / uuid form.
    fn mapping_update_for(&self, core: &IndexCore, mappings: &Value) -> Result<()> {
        self.mapping_update(&core.name, &core.uuid, mappings)
    }

    /// Preserve the parser snapshot so concurrent publications cannot turn
    /// unchanged parameters into stale updates.
    fn mapping_update_from(&self, core: &IndexCore, _base: &Value, mappings: &Value) -> Result<()> {
        self.mapping_update_for(core, mappings)
    }
}

/// Recoveries of a failed copy from its store before it is left failed
/// (ES `index.allocation.max_retries`).
const MAX_RECOVERY_ATTEMPTS: u32 = 5;
/// The first retry's delay; doubled for each further one.
#[cfg(not(test))]
const RECOVERY_BACKOFF: std::time::Duration = std::time::Duration::from_secs(1);
#[cfg(test)]
const RECOVERY_BACKOFF: std::time::Duration = std::time::Duration::ZERO;
/// A copy that stayed healthy this long after a successful recovery starts
/// counting again.
const RECOVERY_ATTEMPTS_RESET: std::time::Duration = std::time::Duration::from_secs(300);

pub(crate) struct RecoveryAttempts {
    pub(crate) count: u32,
    pub(crate) next: std::time::Instant,
    /// When the last recovery succeeded (`None`: it failed, or the copy
    /// failed again since).
    pub(crate) healthy_since: Option<std::time::Instant>,
}

/// The copy failed again at `now`: its count restarts (`true`) only if it
/// had stayed healthy for [`RECOVERY_ATTEMPTS_RESET`] since its last
/// successful recovery.
pub(crate) fn restart_count(a: &mut RecoveryAttempts, now: std::time::Instant) -> bool {
    match a.healthy_since.take() {
        Some(since) if now.duration_since(since) >= RECOVERY_ATTEMPTS_RESET => {
            a.count = 0;
            true
        }
        _ => false,
    }
}

/// Forget the recovery attempts of an index's copies (it is deleted).
pub(crate) fn forget_recovery_attempts(uuid: &str) {
    recovery_attempts().lock().retain(|(u, _), _| u != uuid);
}

/// Recovery attempts per (index uuid, shard).
pub(crate) fn recovery_attempts() -> &'static Mutex<std::collections::HashMap<(String, u32), RecoveryAttempts>> {
    static A: std::sync::OnceLock<Mutex<std::collections::HashMap<(String, u32), RecoveryAttempts>>> = std::sync::OnceLock::new();
    A.get_or_init(Default::default)
}

/// What the logical shards of one index share: name, uuid, settings,
/// mappings, aliases, analysis. Every [`IndexShard`] holds an `Arc` to it;
/// [`IndexService`] and [`IndexShard`] deref to it.
pub struct IndexCore {
    pub name: String,
    pub uuid: String,
    pub creation_date: i64,
    path: PathBuf,
    meta: RwLock<IndexMeta>,
    lookup: RwLock<Arc<FieldLookup>>,
    registry: RwLock<AnalysisRegistry>,
    mapping_lock: Mutex<()>,
    /// Segments are ordered by max `@timestamp` descending (ES leaf sorter): the
    /// index had a `@timestamp` date field when it was created / opened and no `index.sort`.
    ts_sorted: AtomicBool,
    /// Bumped whenever the mapping grows; shard copies add the new core
    /// fields to their schema before their next write / search.
    schema_gen: AtomicU64,
    /// Version of the last metadata applied through
    /// [`IndexService::apply_metadata`] (0 = none).
    metadata_version: AtomicU64,
    /// Highest cluster `mapping_version` this copy's mapping is known to
    /// match or exceed (0 = unknown; not persisted). A host that installs
    /// mappings from several sources (the cluster state, a primary's
    /// replication requests) keeps an older one from overwriting a newer.
    mapping_version: AtomicU64,
    /// The host's id of this copy's index incarnation (a cluster's index
    /// uuid), set once.
    cluster_uuid: std::sync::OnceLock<String>,
    mapping_hook: RwLock<Option<Arc<dyn MappingUpdateHook>>>,
    /// Created sparse (cluster-managed copies): whole-index conveniences
    /// (write / get / update / search over "the index") act on the copies
    /// open here; a document op whose shard has no local copy fails with
    /// `shard_not_found_exception`. Persisted in `index_meta.json`.
    sparse: AtomicBool,
    /// Shards whose primary could not be allocated here (ES
    /// `UnassignedInfo` reason `ALLOCATION_FAILED`): the analyzers did not
    /// build when the shard was to recover (a `lenient: false` synonym
    /// filter with a missing set / invalid rule). The copy stays unopened
    /// until [`IndexService::try_allocate`] succeeds. Not persisted: a
    /// restart retries the allocation (ES: `CLUSTER_RECOVERED`).
    failed: RwLock<BTreeMap<u32, EsError>>,
    /// The node replaced this copy (a cluster wipes a copy before peer
    /// recovery): its files are gone, so whole-index metadata operations on
    /// it fail with [`COPY_REPLACED`] for the caller to retry against the
    /// index the node holds now.
    detached: AtomicBool,
    /// Serialize physical preparation/installation with index retirement.
    preparation_lifecycle: RwLock<()>,
    /// Weak references include unpublished copies: deletion must fence them too.
    prepared_copies: Mutex<Vec<std::sync::Weak<IndexShard>>>,
    /// Shard opens in progress in [`IndexService::prepare_shard`] (shard,
    /// cancellation token): run without the lifecycle lock, cancelled by a
    /// detach or a reset of their slot ([`IndexCore::cancel_openings`]).
    openings: Mutex<Vec<(u32, Arc<AtomicBool>)>>,
}

/// Error type of a metadata operation on an index copy the node replaced
/// while the operation held it (retry against the current copy).
pub const COPY_REPLACED: &str = "index_copy_replaced_exception";

impl IndexCore {
    /// Mark this copy replaced by the node (see [`IndexCore::detached`]).
    pub fn detach(&self) {
        let _lifecycle = self.preparation_lifecycle.write();
        self.detached.store(true, Ordering::SeqCst);
        // An open in progress is not waited for (it can be a long graph
        // catch-up): it stops at its next batch and closes what it opened.
        self.cancel_openings(None);
        let copies = std::mem::take(&mut *self.prepared_copies.lock());
        for copy in copies.into_iter().filter_map(|weak| weak.upgrade()) {
            // No new preparation can enter, and existing WAL operations hold
            // permits. Drain before the caller deletes or reuses the directory.
            copy.fence();
        }
    }

    /// Fence metadata and new writes without draining on the cluster applier.
    /// Recovery must still call `detach` before replacing files. In-flight
    /// operations fail their `attached_or` check rather than acknowledging a
    /// write under a mapping the cluster rejected.
    pub fn quarantine_mapping(&self) {
        self.detached.store(true, Ordering::SeqCst);
        self.cancel_openings(None);
        // Do not drain: a mapping publication can run inside an operation
        // holding a permit. Retirement still performs the blocking fence.
        for copy in self.prepared_copies.lock().iter().filter_map(|weak| weak.upgrade()) {
            copy.fence_new_operations();
        }
    }

    /// Cancel the shard opens in progress (of `shard`; `None`: all): each
    /// fails at its next cancellation point and releases its shard lock, so
    /// whoever resets the slot waits only for that.
    fn cancel_openings(&self, shard: Option<u32>) {
        for (n, token) in self.openings.lock().iter() {
            if shard.is_none_or(|s| s == *n) {
                token.store(true, Ordering::Release);
            }
        }
    }

    /// Whether the node replaced (or is deleting) this copy.
    pub fn is_detached(&self) -> bool {
        self.detached.load(Ordering::SeqCst)
    }

    pub(crate) fn check_attached(&self) -> Result<()> {
        if self.detached.load(Ordering::SeqCst) {
            return Err(EsError::new(503, COPY_REPLACED, format!("the local copy of index [{}] was replaced; retry", self.name)));
        }
        Ok(())
    }

    /// Recheck the copy at the operation's acknowledgement boundary, including
    /// successful operations that raced with mapping quarantine.
    pub(crate) fn attached_or<T>(&self, r: Result<T>) -> Result<T> {
        self.check_attached()?;
        r
    }

}

/// One index on this node: the shared [`IndexCore`] plus its local shard
/// copies. A service from [`IndexService::create`] / [`IndexService::open`]
/// holds every shard; one from
/// [`Node::create_index_sparse`](crate::node::Node::create_index_sparse)
/// holds only the copies the cluster opened ([`IndexService::open_shard`]).
pub struct IndexService {
    core: Arc<IndexCore>,
    /// Slot `n` holds the local copy of shard `n` (sparse: `None` when not
    /// allocated here).
    shards: RwLock<Vec<Option<Arc<IndexShard>>>>,
    preparation_epochs: Vec<AtomicU64>,
    /// An ephemeral view over one shard copy ([`IndexService::view_of`]):
    /// whole-index operations are refused.
    view: bool,
}

/// Unpublished physical shard, exclusively owned until installed or discarded.
pub struct PreparedShard {
    owner: Arc<IndexCore>,
    epoch: u64,
    copy: Option<Arc<IndexShard>>,
}

impl Drop for PreparedShard {
    fn drop(&mut self) {
        if let Some(copy) = self.copy.take() {
            if let Err(error) = copy.close() {
                tracing::warn!(shard=copy.id, %error, "failed to close discarded prepared shard");
            }
        }
    }
}

impl std::ops::Deref for IndexService {
    type Target = IndexCore;
    fn deref(&self) -> &IndexCore {
        &self.core
    }
}

/// The sequence, timestamp and node address ES's time-based id
/// generators share: a 24-bit sequence (started at a random value), a
/// timestamp that never goes backwards on this process (advanced when the
/// sequence wraps within a millisecond), and a 6-byte node address fixed
/// per process (ES: the MAC munged with random bytes; here random).
fn id_parts() -> (u32, u64, [u8; 6]) {
    use std::sync::atomic::AtomicU32;
    static SEQ: std::sync::LazyLock<AtomicU32> = std::sync::LazyLock::new(|| {
        let r = uuid::Uuid::new_v4();
        AtomicU32::new(u32::from_le_bytes([r.as_bytes()[0], r.as_bytes()[1], r.as_bytes()[2], r.as_bytes()[3]]))
    });
    static LAST_TS: AtomicU64 = AtomicU64::new(0);
    static ADDRESS: std::sync::LazyLock<[u8; 6]> = std::sync::LazyLock::new(|| {
        let r = uuid::Uuid::new_v4();
        let mut a = [0u8; 6];
        a.copy_from_slice(&r.as_bytes()[0..6]);
        a
    });
    let sequence_id = SEQ.fetch_add(1, Ordering::Relaxed).wrapping_add(1) & 0xff_ffff;
    let now = crate::date::now_millis().max(0) as u64;
    let previous = LAST_TS.fetch_update(Ordering::AcqRel, Ordering::Acquire, |last| {
        Some(if sequence_id == 0 { last.max(now) + 1 } else { last.max(now) })
    }).unwrap_or(0);
    let timestamp = if sequence_id == 0 { previous.max(now) + 1 } else { previous.max(now) };
    (sequence_id, timestamp, *ADDRESS)
}

/// ES-style auto id: 20 chars of URL-safe base64 of ES's time-based UUID
/// (`TimeBasedUUIDGenerator`). Its byte order shares prefixes between ids
/// and keeps the per-id bytes few, so the `_id` dictionary compresses:
/// random bytes in every id cost ~6 bytes an id more (the noaa store gap).
pub fn auto_id() -> String {
    use base64::Engine;
    let (sequence_id, timestamp, address) = id_parts();
    let mut b = [0u8; 15];
    b[0] = sequence_id as u8;
    b[1] = (sequence_id >> 16) as u8;
    b[2] = (timestamp >> 16) as u8;
    b[3] = (timestamp >> 24) as u8;
    b[4] = (timestamp >> 32) as u8;
    b[5] = (timestamp >> 40) as u8;
    b[6..12].copy_from_slice(&address);
    b[12] = (timestamp >> 8) as u8;
    b[13] = (sequence_id >> 8) as u8;
    b[14] = timestamp as u8;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b)
}

/// ES's k-ordered auto id (`TimeBasedKOrderedUUIDGenerator`, used by
/// logsdb and columnar indices): the timestamp's high bytes, the node
/// address, then the varying bytes, so ids sort roughly by time and share
/// long prefixes (the `_id` terms of a logs index compress, and its inline
/// docs follow doc order).
pub fn auto_id_k_ordered() -> String {
    use base64::Engine;
    let (sequence_id, timestamp, address) = id_parts();
    let mut b = [0u8; 15];
    b[0] = (timestamp >> 40) as u8;
    b[1] = (timestamp >> 32) as u8;
    b[2] = (timestamp >> 24) as u8;
    b[3] = (timestamp >> 16) as u8;
    b[4..10].copy_from_slice(&address);
    b[10] = (sequence_id >> 16) as u8;
    b[11] = (timestamp >> 8) as u8;
    b[12] = (sequence_id >> 8) as u8;
    b[13] = timestamp as u8;
    b[14] = sequence_id as u8;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b)
}

/// Whether auto ids of an index in `index.mode` `mode` are k-ordered (ES
/// `IndexRouting.shouldUseTimeBasedId`: logsdb, logsdb_columnar and
/// columnar indices).
pub fn k_ordered_ids(mode: Option<&str>) -> bool {
    mode.is_some_and(|m| matches!(m.to_ascii_lowercase().as_str(), "logsdb" | "logsdb_columnar" | "columnar"))
}

/// An auto id for an index in `index.mode` `mode` ([`auto_id`] or
/// [`auto_id_k_ordered`], as ES chooses).
pub fn auto_id_for_mode(mode: Option<&str>) -> String {
    if k_ordered_ids(mode) { auto_id_k_ordered() } else { auto_id() }
}

pub fn new_uuid() -> String {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&uuid::Uuid::new_v4().as_bytes()[..16])[..22].to_string()
}

/// The analysis registry of `flat` with every `synonyms_set` filter built
/// empty (ES validates analysis settings this way: the sets load when a
/// shard recovers, see [`registry_for_in`]).
pub(crate) fn registry_for(flat: &FlatSettings) -> Result<AnalysisRegistry> {
    registry_for_in(flat, None, false)
}

/// The node data directory an index directory (`<data>/indices/<uuid>`,
/// `<data>/staging/<uuid>`) belongs to.
fn data_dir_of(path: &Path) -> Option<PathBuf> {
    path.parent().and_then(Path::parent).map(Path::to_path_buf)
}

/// The analysis registry of `flat`; with `load_sets`, `synonyms_set`
/// filters read the synonyms sets of the node at `data` (a missing set or
/// an invalid rule fails a `lenient: false` filter, exactly what makes an
/// ES shard recovery fail).
pub(crate) fn registry_for_in(flat: &FlatSettings, data: Option<&Path>, load_sets: bool) -> Result<AnalysisRegistry> {
    let analysis: FlatSettings = flat
        .iter()
        .filter_map(|(k, v)| k.strip_prefix("index.analysis.").map(|r| (r.to_string(), v.clone())))
        .collect();
    let ctx = crate::analysis::AnalysisCtx { data_dir: data.map(Path::to_path_buf), load_sets };
    let mut reg = AnalysisRegistry::from_settings_in(&settings::unflatten(&analysis), &ctx)?;
    reg.sims = crate::similarity::SimConfig::from_flat(flat)?;
    // `index.mapping.ignore_above` defaults to 8191 in `logsdb` mode
    // (`Mapper.IgnoreAbove.getIgnoreAboveDefaultValue`).
    let logsdb = flat.get("index.mode").and_then(Value::as_str).map(|m| m.eq_ignore_ascii_case("logsdb") || m.eq_ignore_ascii_case("logsdb_columnar")).unwrap_or(false);
    reg.default_ignore_above = flat
        .get("index.mapping.ignore_above")
        .and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
        .map(|v| v as usize)
        .or(if logsdb { Some(8191) } else { None });
    reg.default_ignore_malformed = flat.get("index.mapping.ignore_malformed").map(|v| v.as_bool().unwrap_or_else(|| v.as_str() == Some("true"))).unwrap_or(false);
    reg.downsample_interval = flat.get("index.downsample.interval").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string);
    reg.index_mode = flat.get("index.mode").and_then(Value::as_str).map(str::to_string);
    reg.created = flat.get("index.version.created").map(|v| crate::version::IndexVersion::from_setting(Some(v)));
    reg.ts_doc_values_format = match flat.get("index.use_time_series_doc_values_format").map(|v| v.as_bool().unwrap_or_else(|| v.as_str() == Some("true"))) {
        Some(b) => b,
        None => matches!(reg.index_mode.as_deref().map(str::to_ascii_lowercase).as_deref(), Some("time_series" | "logsdb" | "columnar" | "logsdb_columnar")),
    };
    let flag = |k: &str| flat.get(k).map(|v| v.as_bool().unwrap_or_else(|| v.as_str() == Some("true"))).unwrap_or(true);
    reg.dv_multi_value = flag("index.mapping.doc_values.multi_value");
    reg.dv_nullability = flag("index.mapping.doc_values.nullability");
    reg.sort_fields = crate::indexsort::effective_sort_fields(flat);
    reg.synthetic_id = crate::tsdb::synthetic_id(flat)
        && crate::indexsort::mode_default_sort_enabled()
        && reg.sort_fields.iter().map(String::as_str).eq([crate::tsdb::TSID_FIELD, "@timestamp"]);
    reg.use_dv_skipper = match flat.get("index.mapping.use_doc_values_skipper") {
        Some(v) => v.as_bool().unwrap_or_else(|| v.as_str() == Some("true")),
        None => crate::settings::use_doc_values_skipper_default(reg.index_mode.as_deref(), flat.get("index.version.created").map(|v| crate::version::IndexVersion::from_setting(Some(v))).unwrap_or(crate::version::IndexVersion::CURRENT)),
    };
    reg.index_disabled_by_default = match flat.get("index.mapping.index_disabled_by_default") {
        Some(v) => v.as_bool().unwrap_or_else(|| v.as_str() == Some("true")),
        None => reg.strict_columnar(),
    };
    reg.dynamic_strings_keyword_only = match flat.get("index.mapping.dynamic_strings.auto_text").map(|v| v.as_bool().unwrap_or_else(|| v.as_str() == Some("true"))) {
        Some(auto_text) => !auto_text,
        None => matches!(reg.index_mode.as_deref().map(str::to_ascii_lowercase).as_deref(), Some("columnar" | "logsdb_columnar")),
    };
    reg.created_version = match flat.get("index.version.created") {
        None => crate::version::IndexVersion::CURRENT,
        v => crate::version::IndexVersion::from_setting(v),
    };
    reg.exclude_source_vectors = match flat.get("index.mapping.exclude_source_vectors") {
        Some(v) => v.as_bool().unwrap_or_else(|| v.as_str() == Some("true")),
        None => reg.created_version.on_or_after(crate::version::IndexVersion::EXCLUDE_SOURCE_VECTORS_DEFAULT),
    };
    // Indices created before Lucene 10 keep the non-stemming `persian`
    // analyzer (`PersianAnalyzerProvider` / `LegacyPersianAnalyzer`).
    // Absent while an index is being created: it gets the current version.
    {
        use crate::version::IndexVersion;
        let created = reg.created_version;
        if created.before(IndexVersion::UPGRADE_TO_LUCENE_10_0_0) {
            reg.legacy_builtin("persian", "persian_v8");
        }
        reg.malformed_sorted = created.on_or_after(IndexVersion::STORE_IGNORED_MALFORMED_IN_BINARY_DOC_VALUES);
    }
    Ok(reg)
}

/// `index.auto_expand_replicas: "<min>-<max|all>"`: ES sets
/// `number_of_replicas` to `clamp(data_nodes - 1, min, max)`, which on a
/// single node is `min` (system indices use `0-1` and stay green).
fn apply_auto_expand_replicas(s: &mut FlatSettings) {
    let lo = s
        .get("index.auto_expand_replicas")
        .and_then(Value::as_str)
        .and_then(|v| v.split_once('-'))
        .and_then(|(lo, _)| lo.trim().parse::<u64>().ok());
    if let Some(lo) = lo {
        s.insert("index.number_of_replicas".into(), json!(lo.to_string()));
    }
}

/// Post-parse work that needs the final mapping: the stored-fields block
/// and, for synthetic-source indices, the canonical `_source` to store.
#[cfg(test)]
thread_local! {
    /// Tests: nanoseconds in `synthesize` and in `store_forms` ([`finish_parsed`]).
    pub(crate) static PARSE_PHASES: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
    /// Tests: skip the reference checks of `finish_parsed` (microbenches).
    pub(crate) static SKIP_SYNTH_CHECKS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn finish_parsed(pd: &mut crate::document::ParsedDoc, v: &Value, m: &Mappings, lk: &FieldLookup, settings: &FlatSettings) -> Result<()> {
    crate::stored::attach(&mut pd.doc, v, lk);
    let synthetic = crate::synthetic::source_mode(settings::get_str(settings, "index.mapping.source.mode"), m, settings::get_str(settings, "index.mode")) == crate::synthetic::SourceMode::Synthetic;
    // `exclude_source_vectors`: store the source ES would load back (the
    // vectors in their indexed form, patched in at the end of their object).
    let patched = if !synthetic && crate::vector_source::has_synthetic_vectors(lk) && crate::vector_source::excludes_in(settings) { crate::vector_source::rehydrate(v, lk) } else { None };
    if let Some(p) = &patched {
        pd.source = Some(p.to_string());
    }
    let v = patched.as_ref().unwrap_or(v);
    if synthetic {
        let keep = crate::synthetic::Keep::of_index(settings);
        #[cfg(test)]
        let t_synth = std::time::Instant::now();
        // A plain document's stored forms in one walk; the two passes for
        // the rest (and in tests, to check the one walk's bytes).
        let plain = crate::synthetic::plain_forms(v, m, lk, keep, &pd.doc);
        #[cfg(test)]
        if plain.is_some() {
            PARSE_PHASES.with(|p| { let mut x = p.get(); x.0 += t_synth.elapsed().as_nanos() as u64; p.set(x); });
        }
        #[cfg(test)]
        if let (Some((full, residual)), false) = (&plain, SKIP_SYNTH_CHECKS.with(|c| c.get())) {
            crate::synthetic::FORCE_GENERAL.with(|f| f.set(true));
            let (synth, mut ignored) = crate::synthetic::synthesize(v, m, lk, keep).expect("the general walk accepts a plain document");
            crate::synthetic::FORCE_GENERAL.with(|f| f.set(false));
            assert!(ignored.is_none(), "a plain document records no ignored source");
            let (gfull, gresidual) = crate::synthetic::store_forms(&synth, lk, &pd.doc, &mut ignored);
            assert_eq!(full, &gfull);
            assert_eq!(residual, &gresidual);
            let reference = crate::synthetic::residual_for(synth.clone(), lk, &pd.doc);
            let strip = |r: &Option<String>| r.as_deref().map(|r| match crate::synthetic::residual_parts(r) {
                Some((ids, _, json)) => format!("{ids}|{json}"),
                None => r.to_string(),
            });
            assert_eq!(strip(residual), strip(&reference));
        }
        // From v34 an empty residual whose fields the columns tell back
        // is stored as two bytes.
        let compact = rustic_core::codec::write_version() >= rustic_core::codec::EMPTY_RESIDUAL_SINCE;
        if let Some((full, residual)) = plain {
            pd.source = Some(full);
            if let Some(residual) = residual {
                let residual = if compact { crate::synthetic::compact_residual(residual, lk, &pd.doc) } else { residual };
                pd.doc.fields.push((crate::mapping::SOURCE_FIELD, rustic_core::types::FieldValue::Keyword(residual)));
            }
            return Ok(());
        }
        let (synth, mut ignored) = crate::synthetic::synthesize(v, m, lk, keep)?;
        #[cfg(test)]
        let t_forms = std::time::Instant::now();
        // ES stores no `_source` for synthetic source: the stored one is
        // the residual the doc-value columns do not rebuild (the translog
        // keeps the canonical form, `pd.source`).
        #[cfg(test)]
        let had_ignored = ignored.clone();
        let (full, residual) = crate::synthetic::store_forms(&synth, lk, &pd.doc, &mut ignored);
        #[cfg(test)]
        PARSE_PHASES.with(|p| { let mut p = p.get(); p.0 += (t_forms - t_synth).as_nanos() as u64; p.1 += t_forms.elapsed().as_nanos() as u64; PARSE_PHASES.with(|c| c.set(p)); });
        // Every engine test writing synthetic source checks the one-walk
        // forms against the reference implementation, and that folded
        // `_ignored_source` entries rebuild exactly.
        #[cfg(test)]
        if !SKIP_SYNTH_CHECKS.with(|c| c.get()) {
            // The in-place walk of plain documents is the general walk's.
            crate::synthetic::FORCE_GENERAL.with(|f| f.set(true));
            let general = crate::synthetic::synthesize(v, m, lk, keep);
            crate::synthetic::FORCE_GENERAL.with(|f| f.set(false));
            let (general, general_ignored) = general.expect("the general walk accepts what the plain walk did");
            assert_eq!(general, synth);
            assert_eq!(general_ignored.map(|e| e.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>()), had_ignored.as_ref().map(|e| e.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>()));
            assert_eq!(full, synth.to_string());
            let reference = crate::synthetic::residual_for(synth.clone(), lk, &pd.doc);
            let strip = |r: &Option<String>| r.as_deref().map(|r| match crate::synthetic::residual_parts(r) {
                Some((ids, _, json)) => format!("{ids}|{json}"),
                None => r.to_string(),
            });
            assert_eq!(strip(&residual), strip(&reference));
            if let (Some(entries), None, Some(r)) = (&had_ignored, &ignored, &residual) {
                let format = crate::synthetic::IgnoredSourceFormat::Singular;
                assert_eq!(crate::synthetic::folded_ignored_source(r, &full, lk, format), Some(crate::synthetic::encode_ignored_source(entries, format)));
            }
        }
        if let Some(residual) = residual {
            let residual = if compact { crate::synthetic::compact_residual(residual, lk, &pd.doc) } else { residual };
            pd.doc.fields.push((crate::mapping::SOURCE_FIELD, rustic_core::types::FieldValue::Keyword(residual)));
        } else if let Some(stored) = crate::vector_source::synthetic_stored_form(&synth, lk, &pd.doc) {
            // The indexed vectors left to the vector store (the stored
            // fields would otherwise keep each one again as JSON text),
            // and the unmapped strings kept once (not also as
            // `_ignored_source`); `pd.source`, which the translog keeps,
            // stays whole.
            let stored = crate::synthetic::fold_ignored(stored, &synth, lk, &mut ignored);
            pd.doc.fields.push((crate::mapping::SOURCE_FIELD, rustic_core::types::FieldValue::Keyword(stored)));
        }
        pd.source = Some(full);
        if let Some(entries) = ignored {
            let created = match settings.get("index.version.created") {
                None => crate::version::IndexVersion::CURRENT,
                v => crate::version::IndexVersion::from_setting(v),
            };
            let ts_dv = match settings.get("index.use_time_series_doc_values_format").map(|v| v.as_bool().unwrap_or_else(|| v.as_str() == Some("true"))) {
                Some(b) => b,
                None => matches!(settings::get_str(settings, "index.mode").map(str::to_ascii_lowercase).as_deref(), Some("time_series" | "logsdb" | "columnar" | "logsdb_columnar")),
            };
            let format = crate::synthetic::IgnoredSourceFormat::of(created, ts_dv);
            for value in crate::synthetic::encode_ignored_source(&entries, format) {
                pd.doc.fields.push((crate::mapping::IGNORED_SOURCE_FIELD, rustic_core::types::FieldValue::Keyword(value)));
            }
        }
    } else if let Some(src) = m.source.as_ref().filter(|s| s.get("includes").is_some() || s.get("excludes").is_some()) {
        // Mapping-level `_source.includes` / `excludes` filter what is stored.
        let list = |k: &str| -> Vec<String> { src.get(k).and_then(Value::as_array).map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default() };
        pd.source = Some(crate::search::filter_source(v, &list("includes"), &list("excludes")).to_string());
    } else if settings::get_str(settings, "index.mapping.semantic_text.use_legacy_format") == Some("true") {
        // Legacy semantic_text format: the inference results live in `_source`.
        if let Some(src) = crate::inference::semantic::legacy_source(v, lk, &pd.doc) {
            pd.source = Some(src.to_string());
        }
    } else if let Some(p) = &patched {
        // The stored `_source` leaves the indexed vectors to the vector
        // store (`crate::vector_source::stored_form`); `pd.source`, which
        // the translog keeps, stays whole.
        let enabled = crate::synthetic::source_mode(settings::get_str(settings, "index.mapping.source.mode"), m, settings::get_str(settings, "index.mode")) != crate::synthetic::SourceMode::Disabled;
        let doc = &pd.doc;
        if let Some((stored, sparse)) = enabled.then(|| crate::vector_source::stored_form_marked(p, lk, &doc)).flatten() {
            pd.doc.fields.push((crate::mapping::SOURCE_FIELD, rustic_core::types::FieldValue::Keyword(stored)));
            // Sparse vectors served from their doc values (as ES serves
            // stored ones from term vectors) leave the stored-fields block.
            crate::stored::drop_from_block(&mut pd.doc, lk, &sparse);
        }
    }
    Ok(())
}

impl IndexService {
    /// Default settings ES reports for every index.
    fn complete_settings(name: &str, uuid: &str, created: i64, mut s: FlatSettings) -> FlatSettings {
        s.entry("index.number_of_shards".into()).or_insert(json!("1"));
        s.entry("index.number_of_replicas".into()).or_insert(json!("1"));
        s.insert("index.uuid".into(), json!(uuid));
        s.insert("index.provided_name".into(), json!(name));
        s.insert("index.creation_date".into(), json!(created.to_string()));
        s.insert("index.version.created".into(), json!(crate::version::IndexVersion::CURRENT.to_string()));
        s.entry("index.routing.allocation.include._tier_preference".into()).or_insert(json!("data_content"));
        apply_auto_expand_replicas(&mut s);
        s
    }

    pub fn create(path: &Path, name: &str, settings_in: FlatSettings, mappings_json: &Value, aliases: BTreeMap<String, Value>) -> Result<IndexService> {
        Self::create_with_uuid(path, name, None, settings_in, mappings_json, aliases)
    }

    /// [`IndexService::create`] with a given `index.uuid` (hook 1; `None`
    /// = a random one).
    pub fn create_with_uuid(path: &Path, name: &str, uuid: Option<&str>, settings_in: FlatSettings, mappings_json: &Value, aliases: BTreeMap<String, Value>) -> Result<IndexService> {
        settings::validate(&settings_in)?;
        crate::fielddata_cache::clear_index(name);
        if settings_in.get("index.number_of_shards").and_then(Value::as_str).map(|s| s != "1").unwrap_or(false) {
            // Single-shard node: accept the setting (clients pass it) but run one shard.
        }
        // ES validates the analysis settings without loading synonyms sets
        // (`CREATE_INDEX`); the sets load when the shards recover, and a
        // failure there leaves the primaries unassigned, not the create
        // request rejected.
        let (registry, recovery_err) = match registry_for_in(&settings_in, data_dir_of(path).as_deref(), true) {
            Ok(r) => (r, None),
            Err(e) => (registry_for(&settings_in)?, Some(e)),
        };
        let mut mappings = Mappings::parse(mappings_json, &registry)?;
        crate::indexsort::check_runtime_shadows_sort(&settings_in, mappings.runtime.as_ref())?;
        crate::tsdb::prepare_mappings(&settings_in, &mut mappings, &registry)?;
        crate::tsdb::validate_create(&settings_in, &mappings)?;
        // The `index.mapping.source.mode` setting wins over the mapping's
        // `_source.mode`. Record it on the mappings so `get_mapping` reports
        // the effective mode and `put_mapping` sees it as the current value.
        // When the mapping never declared a mode, the mapping stays empty
        // (no `_source` key in get_mapping) unless the setting is present.
        if let Some(v) = settings::get_str(&settings_in, "index.mapping.source.mode") {
            mappings.source_mode_setting = Some(v.to_ascii_lowercase());
        }
        crate::synthetic::validate_create(&settings_in, &mappings)?;
        let limits = crate::fieldtypes::limits::Limits::from_settings(|k| {
            let v = settings_in.get(k).map(|v| match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            });
            // A new index is created with the current index version.
            if k == "index.version.created" { v.or_else(|| Some(crate::version::IndexVersion::CURRENT.to_string())) } else { v }
        });
        crate::fieldtypes::limits::check(&mappings, &limits)?;
        mappings.assign_core_ids();
        let lookup = FieldLookup::build(&mappings, &registry);
        crate::indexsort::validate_sort(&settings_in, &lookup)?;
        crate::inference::semantic::validate_mapping(&settings_in, &lookup)?;
        crate::inference::semantic_map::validate_fields(&mappings, &lookup, None)?;
        let uuid = uuid.map(str::to_string).unwrap_or_else(new_uuid);
        let created = crate::date::now_millis();
        let settings = Self::complete_settings(name, &uuid, created, settings_in);
        let n = shard_count(&settings);
        let core = Arc::new(IndexCore::new(name.to_string(), uuid, created, path, IndexMeta { settings, mappings, aliases, open: true }, lookup, registry));
        let svc = IndexService { core, shards: RwLock::new(vec![None; n as usize]), preparation_epochs: (0..n).map(|_| AtomicU64::new(0)).collect(), view: false };
        // Shard opens create their directories; an index whose shards stay
        // unallocated still needs its own for `index_meta.json`.
        std::fs::create_dir_all(path)?;
        match recovery_err {
            None => {
                for i in 0..n {
                    svc.open_shard(i, ShardOpenSpec::local_primary(RecoverySource::Empty))?;
                }
            }
            Some(e) => svc.mark_unallocated_failed(&e),
        }
        svc.persist()?;
        Ok(svc)
    }

    /// An index service over `core` with no shard open yet (the cluster
    /// opens the copies allocated here with [`IndexService::open_shard`]).
    pub(crate) fn sparse_from(core: IndexCore) -> IndexService {
        let n = shard_count(&core.meta.read().settings);
        core.sparse.store(true, Ordering::Release);
        IndexService { core: Arc::new(core), shards: RwLock::new(vec![None; n as usize]), preparation_epochs: (0..n).map(|_| AtomicU64::new(0)).collect(), view: false }
    }

    /// Reopen from disk and replay the translogs. A sparse index (created
    /// by [`Node::create_index_sparse`](crate::node::Node::create_index_sparse))
    /// reopens with no shard open: the cluster reopens the copies it
    /// allocates here. An index written by the single-shard emulation (one
    /// `shard-0/` holding every document of an `N > 1` index) is migrated to
    /// real shards on open: documents move to the shard they route to.
    pub fn open(path: &Path) -> Result<IndexService> {
        let raw: Value = serde_json::from_slice(&std::fs::read(path.join(META_FILE))?)?;
        let settings: FlatSettings = raw["settings"].as_object().map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect()).unwrap_or_default();
        // Recovery builds the analyzers with the synonyms sets; a failure
        // leaves the shards of an open index unassigned (a closed index
        // recovers without them, ES keeps closed indices green).
        let (registry, recovery_err) = match registry_for_in(&settings, data_dir_of(path).as_deref(), true) {
            Ok(r) => (r, None),
            Err(e) => (registry_for(&settings)?, Some(e)),
        };
        let mut mappings = Mappings::from_persisted(&raw["mappings"], &registry)?;
        crate::tsdb::prepare_mappings(&settings, &mut mappings, &registry)?;
        let lookup = FieldLookup::build(&mappings, &registry);
        let aliases = raw["aliases"].as_object().map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect()).unwrap_or_default();
        let n = shard_count(&settings);
        let meta = IndexMeta { settings, mappings, aliases, open: raw["open"].as_bool().unwrap_or(true) };
        let core = IndexCore::new(
            raw["name"].as_str().unwrap_or_default().to_string(),
            raw["uuid"].as_str().unwrap_or_default().to_string(),
            raw["creation_date"].as_i64().unwrap_or(0),
            path,
            meta,
            lookup,
            registry,
        );
        core.metadata_version.store(raw["metadata_version"].as_u64().unwrap_or(0), Ordering::Relaxed);
        let sparse = raw["sparse"].as_bool().unwrap_or(false);
        core.sparse.store(sparse, Ordering::Relaxed);
        let svc = IndexService { core: Arc::new(core), shards: RwLock::new(vec![None; n as usize]), preparation_epochs: (0..n).map(|_| AtomicU64::new(0)).collect(), view: false };
        if sparse {
            return Ok(svc);
        }
        if let (Some(e), true) = (recovery_err, svc.is_open()) {
            svc.mark_unallocated_failed(&e);
            return Ok(svc);
        }
        // A migration interrupted by a crash restarts from shard-0 (still
        // complete: it is only trimmed after the other shards committed).
        let marker = path.join(MIGRATION_MARKER);
        let legacy = n > 1 && (marker.exists() || (1..n).all(|i| !path.join(format!("shard-{i}")).exists()));
        if legacy {
            crate::hooks::write_durable(&marker, b"{}")?;
            for i in 1..n {
                let _ = std::fs::remove_dir_all(path.join(format!("shard-{i}")));
            }
        }
        for i in 0..n {
            let rec = if legacy && i > 0 { RecoverySource::Empty } else { RecoverySource::ExistingStore };
            if let Err(e) = svc.open_shard(i, ShardOpenSpec::local_primary(rec)) {
                // Another holder owns the store: opening it here is the
                // error, not a failed copy.
                if e.kind == "shard_lock_obtain_failed_exception" {
                    return Err(e);
                }
                // The index stays registered with this copy failed (red,
                // like an unassigned ES primary) instead of vanishing from
                // the node: its files are untouched.
                tracing::error!("failed to open shard [{}][{i}] at {}: {e}", svc.name, path.display());
                svc.failed.write().insert(i, e);
            }
        }
        if legacy && svc.failed.read().is_empty() {
            svc.migrate_emulated_layout()?;
        }
        Ok(svc)
    }

    /// Move every document of `shard-0` that routes elsewhere to its shard
    /// (indices written while `number_of_shards > 1` was emulated on one
    /// shard), then commit all shards.
    fn migrate_emulated_layout(&self) -> Result<()> {
        let shards = self.local_shards();
        let src = shards[0].clone();
        for dst in &shards[1..] {
            let id = dst.id;
            dst.copy_docs_from(&src, &|doc_id, routing| self.route(doc_id, routing).map(|s| s == id).unwrap_or(false))?;
            dst.shard.flush()?;
        }
        src.retain_docs(&|doc_id, routing| self.route(doc_id, routing).map(|s| s == 0).unwrap_or(true))?;
        src.shard.flush()?;
        let _ = std::fs::remove_file(self.path().join(MIGRATION_MARKER));
        Ok(())
    }
}

/// `index.number_of_shards` of a settings map (1 when absent / invalid).
pub(crate) fn shard_count(s: &FlatSettings) -> u32 {
    s.get("index.number_of_shards")
        .and_then(|v| v.as_str().and_then(|x| x.parse().ok()).or_else(|| v.as_u64().map(|x| x as u32)))
        .unwrap_or(1)
        .max(1)
}

impl IndexCore {
    fn new(name: String, uuid: String, creation_date: i64, path: &Path, meta: IndexMeta, lookup: FieldLookup, registry: AnalysisRegistry) -> IndexCore {
        let core = IndexCore {
            name,
            uuid,
            creation_date,
            path: path.to_path_buf(),
            meta: RwLock::new(meta),
            lookup: RwLock::new(Arc::new(lookup)),
            registry: RwLock::new(registry),
            mapping_lock: Mutex::new(()),
            ts_sorted: AtomicBool::new(false),
            schema_gen: AtomicU64::new(1),
            metadata_version: AtomicU64::new(0),
            mapping_version: AtomicU64::new(0),
            cluster_uuid: std::sync::OnceLock::new(),
            mapping_hook: RwLock::new(None),
            sparse: AtomicBool::new(false),
            failed: RwLock::new(BTreeMap::new()),
            detached: AtomicBool::new(false),
            preparation_lifecycle: RwLock::new(()),
            prepared_copies: Mutex::new(Vec::new()),
            openings: Mutex::new(Vec::new()),
        };
        core.refresh_ts_sorted();
        core
    }

    /// The core of a sparse (cluster-managed) index: metadata verbatim.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_sparse(name: &str, uuid: &str, created: i64, path: &Path, settings: FlatSettings, mappings: Mappings, lookup: FieldLookup, registry: AnalysisRegistry, aliases: BTreeMap<String, Value>) -> IndexCore {
        IndexCore::new(name.to_string(), uuid.to_string(), created, path, IndexMeta { settings, mappings, aliases, open: true }, lookup, registry)
    }

    /// Install (or remove) the dynamic-mapping hook (A6).
    pub fn set_mapping_update_hook(&self, hook: Option<Arc<dyn MappingUpdateHook>>) {
        *self.mapping_hook.write() = hook;
    }

    /// Version of the last metadata applied with
    /// [`IndexService::apply_metadata`] (0 = never).
    pub fn metadata_version(&self) -> u64 {
        self.metadata_version.load(Ordering::Acquire)
    }

    /// Record the host's id of this copy's incarnation (first call wins).
    pub fn set_cluster_uuid(&self, uuid: &str) {
        let _ = self.cluster_uuid.set(uuid.to_string());
    }

    /// The host's id of this copy's incarnation, if recorded.
    pub fn cluster_uuid(&self) -> Option<&str> {
        self.cluster_uuid.get().map(String::as_str)
    }

    /// See the `mapping_version` field (0 = unknown).
    pub fn mapping_version(&self) -> u64 {
        self.mapping_version.load(Ordering::Acquire)
    }

    /// Record that the mapping now matches cluster mapping version `v`
    /// (only ever raised).
    pub fn advance_mapping_version(&self, v: u64) {
        self.mapping_version.fetch_max(v, Ordering::AcqRel);
    }

    /// Generation of the mapping's core schema (bumped when fields are added).
    pub(crate) fn schema_generation(&self) -> u64 {
        self.schema_gen.load(Ordering::Acquire)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn persist(&self) -> Result<()> {
        let m = self.meta.read();
        let mut v = json!({
            "name": self.name,
            "uuid": self.uuid,
            "creation_date": self.creation_date,
            "settings": settings::as_flat_json(&m.settings),
            "mappings": m.mappings.to_persisted(),
            "aliases": m.aliases,
            "open": m.open,
        });
        // Only cluster-managed (sparse / versioned) indices carry these keys.
        if self.sparse.load(Ordering::Acquire) {
            v["sparse"] = json!(true);
        }
        if self.metadata_version() > 0 {
            v["metadata_version"] = json!(self.metadata_version());
        }
        crate::hooks::write_durable(&self.path.join(META_FILE), &serde_json::to_vec_pretty(&v)?)?;
        Ok(())
    }

    pub fn meta(&self) -> IndexMeta {
        self.meta.read().clone()
    }

    /// Whether `_source` is stored as sent (no `_source` settings beyond
    /// `mode: stored` / `enabled: true`, stored source mode), so doc values
    /// can stand in for it. Read under the lock: [`Self::meta`] clones every
    /// mapping, and ES|QL / SQL scans ask once per batch of documents.
    /// Whether `_source` is synthetic (rebuilt from doc values and the
    /// stored residual).
    pub fn synthetic_source(&self) -> bool {
        let m = self.meta.read();
        crate::synthetic::source_mode(crate::settings::get_str(&m.settings, "index.mapping.source.mode"), &m.mappings, crate::settings::get_str(&m.settings, "index.mode"))
            == crate::synthetic::SourceMode::Synthetic
    }

    pub fn source_stored_as_sent(&self) -> bool {
        let m = self.meta.read();
        m.mappings.source.as_ref().is_none_or(|source| source.as_object().is_some_and(|fields|
            fields.iter().all(|(name, value)| (name == "mode" && value.as_str() == Some("stored"))
                || (name == "enabled" && value.as_bool() == Some(true)))))
            && matches!(m.mappings.effective_source_mode(), None | Some(crate::synthetic::SourceMode::Stored))
    }

    /// The mapping's `runtime` section, without cloning the rest of the
    /// metadata (every search of an index with scripts reads it).
    pub fn mapping_runtime(&self) -> Option<Value> {
        self.meta.read().mappings.runtime.clone()
    }

    /// Whether the index is a time-series index (`index.mode`), without
    /// cloning its metadata (checked by every search).
    pub fn is_time_series(&self) -> bool {
        crate::tsdb::is_time_series(&self.meta.read().settings)
    }

    /// Serialize the current mapping without first cloning the full index
    /// metadata. The read guard keeps this snapshot consistent with updates.
    pub fn mapping_json(&self) -> Value {
        self.meta.read().mappings.to_json()
    }

    pub fn mapping_json_admitted(&self) -> Result<(String, crate::memory::Reservation)> {
        self.meta.read().mappings.to_json_admitted()
    }


    pub fn lookup(&self) -> Arc<FieldLookup> {
        self.lookup.read().clone()
    }

    /// Hash of the index settings and aliases: a component of the shard
    /// request cache key (dynamic settings and alias changes take effect
    /// without touching the data).
    pub fn meta_fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let m = self.meta.read();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for (k, v) in &m.settings {
            k.hash(&mut h);
            v.to_string().hash(&mut h);
        }
        for (k, v) in &m.aliases {
            k.hash(&mut h);
            v.to_string().hash(&mut h);
        }
        m.open.hash(&mut h);
        h.finish()
    }

    pub fn is_open(&self) -> bool {
        self.meta.read().open
    }

    pub fn setting(&self, key: &str) -> Option<String> {
        self.meta.read().settings.get(key).and_then(Value::as_str).map(str::to_string)
    }

    /// `index.query.default_field` (`*` when unset). The setting is a list
    /// (kept as a JSON array, e.g. big5's `["message"]`) or a string.
    pub fn query_default_fields(&self) -> Vec<String> {
        let raw = match self.meta.read().settings.get("index.query.default_field") {
            Some(Value::Array(a)) => {
                let v: Vec<String> = a.iter().map(crate::query::dsl::scalar_string).filter(|x| !x.is_empty()).collect();
                return if v.is_empty() { vec!["*".to_string()] } else { v };
            }
            Some(Value::String(s)) => s.clone(),
            _ => return vec!["*".to_string()],
        };
        let raw = raw.trim().trim_start_matches('[').trim_end_matches(']');
        let v: Vec<String> = raw.split(',').map(|x| x.trim().trim_matches('"').to_string()).filter(|x| !x.is_empty()).collect();
        if v.is_empty() { vec!["*".to_string()] } else { v }
    }

    /// The version this index was created with (`index.version.created`):
    /// the key every version-dependent default is gated on (see
    /// [`crate::version`]). Indices written by older builds without the
    /// setting read as 8.x-created. A read lock + parse: callers on a
    /// per-document path should hoist it.
    pub fn created_version(&self) -> crate::version::IndexVersion {
        crate::version::IndexVersion::from_setting(self.meta.read().settings.get("index.version.created"))
    }

    /// `index.disable_sequence_numbers`: sequence numbers and primary terms
    /// are hidden (`-2` / `0` in write responses, absent from get), cannot
    /// be queried, and optimistic concurrency control / updates are
    /// refused. (Replication keeps its internal sequence numbers.)
    pub fn seq_no_disabled(&self) -> bool {
        let explicit = match self.meta.read().settings.get("index.disable_sequence_numbers") {
            Some(Value::Bool(b)) => Some(*b),
            Some(Value::String(v)) => Some(v == "true"),
            _ => None,
        };
        match explicit {
            Some(b) => b,
            // Default: on for strict columnar indices
            // (`IndexSettings.DISABLE_SEQUENCE_NUMBERS`).
            None => {
                let columnar = self.setting("index.mode").is_some_and(|m| matches!(m.to_ascii_lowercase().as_str(), "columnar" | "logsdb_columnar"));
                columnar && self.created_version().on_or_after(crate::version::IndexVersion::DISABLE_SEQUENCE_NUMBERS)
            }
        }
    }

    pub fn setting_usize(&self, key: &str, default: usize) -> usize {
        self.setting(key).and_then(|s| s.parse().ok()).unwrap_or(default)
    }

    pub fn aliases(&self) -> BTreeMap<String, Value> {
        self.meta.read().aliases.clone()
    }

    pub fn set_aliases(&self, a: BTreeMap<String, Value>) -> Result<()> {
        self.meta.write().aliases = a;
        self.persist()
    }

    /// Re-evaluate whether segments are sorted on `@timestamp` (index create / open).
    fn refresh_ts_sorted(&self) {
        let has_index_sort = self.meta.read().settings.keys().any(|k| k.starts_with("index.sort."));
        let ts = !has_index_sort && self.lookup.read().get("@timestamp").is_some_and(|f| matches!(f.ty, crate::mapping::EsType::Date | crate::mapping::EsType::DateNanos));
        self.ts_sorted.store(ts, Ordering::Relaxed);
    }

    /// Whether this index orders its segments by max `@timestamp` descending.
    pub fn segments_sorted_on_timestamp(&self) -> bool {
        self.ts_sorted.load(Ordering::Relaxed)
    }

    /// Whether this index was created sparse (cluster-managed shard copies).
    pub fn is_sparse(&self) -> bool {
        self.sparse.load(Ordering::Acquire)
    }

    pub(crate) fn check_open(&self) -> Result<()> {
        self.check_attached()?;
        if !self.is_open() {
            return Err(EsError::index_closed(&self.name, &self.uuid));
        }
        Ok(())
    }

    fn block_err(&self, status: u16, id: &str) -> EsError {
        EsError::new(status, "cluster_block_exception", format!("index [{}] blocked by: [{id}];", self.name))
    }

    /// Write blocks (`index.blocks.*`, ES `ClusterBlockLevel.WRITE`);
    /// `read_only_allow_delete` lets deletes through.
    fn blocked_writes(&self, is_delete: bool) -> Result<()> {
        let m = self.meta.read();
        let b = |k: &str| settings::get_bool(&m.settings, k, false);
        if b("index.blocks.read_only") {
            return Err(self.block_err(403, "FORBIDDEN/5/index read-only (api)"));
        }
        if b("index.blocks.write") {
            return Err(self.block_err(403, "FORBIDDEN/8/index write (api)"));
        }
        if !is_delete && b("index.blocks.read_only_allow_delete") {
            return Err(self.block_err(429, "TOO_MANY_REQUESTS/12/disk usage exceeded flood-stage watermark, index has read-only-allow-delete block"));
        }
        // A CCR follower only takes the leader's operations (`FollowingEngine`).
        if b(crate::ccr::FOLLOWING_INDEX_SETTING) {
            return Err(crate::ccr::following_write_error());
        }
        Ok(())
    }

    /// `index.blocks.read` (searches, gets, explain, ...).
    pub fn check_read_block(&self) -> Result<()> {
        if settings::get_bool(&self.meta.read().settings, "index.blocks.read", false) {
            return Err(self.block_err(403, "FORBIDDEN/7/index read (api)"));
        }
        Ok(())
    }

    /// `index.blocks.metadata` (mapping / settings reads and writes) and,
    /// for writes, `read_only` / `read_only_allow_delete`.
    pub fn check_metadata_block(&self, write: bool) -> Result<()> {
        let m = self.meta.read();
        let b = |k: &str| settings::get_bool(&m.settings, k, false);
        if b("index.blocks.metadata") {
            return Err(self.block_err(403, "FORBIDDEN/9/index metadata (api)"));
        }
        if write && b("index.blocks.read_only") {
            return Err(self.block_err(403, "FORBIDDEN/5/index read-only (api)"));
        }
        if write && b("index.blocks.read_only_allow_delete") {
            return Err(self.block_err(429, "TOO_MANY_REQUESTS/12/disk usage exceeded flood-stage watermark, index has read-only-allow-delete block"));
        }
        Ok(())
    }

    /// Parse a source with dynamic mapping; applies (and persists) mapping
    /// updates under the mapping lock.
    fn parse_source(&self, v: &Value, id: &str) -> Result<(crate::document::ParsedDoc, bool)> {
        let limits = crate::fieldtypes::limits::Limits::from_settings(|k| self.setting(k));
        self.parse_source_with(v, id, &limits)
    }

    /// [`Self::parse_source`] with the mapping limits already resolved (a
    /// bulk resolves them once, not per document).
    fn parse_source_with(&self, v: &Value, id: &str, limits: &crate::fieldtypes::limits::Limits) -> Result<(crate::document::ParsedDoc, bool)> {
        self.parse_source_mode(v, id, limits, false)
    }

    /// [`Self::parse_source_with`]; `defer`: when a dynamic mapping update
    /// is needed and a cluster hook is installed, fail with
    /// [`DEFERRED_MAPPING`] instead of requesting it, so the caller batches
    /// the updates of a whole shard request into one master round trip
    /// ([`Self::request_dynamic_mapping`]).
    fn parse_source_mode(&self, v: &Value, id: &str, limits: &crate::fieldtypes::limits::Limits, defer: bool) -> Result<(crate::document::ParsedDoc, bool)> {
        // `ShardBulkInferenceActionFilter`: the semantic_text inference runs
        // once per document, before it is parsed.
        // One lookup snapshot per document, read before the metadata guard as
        // before; without semantic fields `infer_document` is empty, so the
        // settings read it needs is skipped.
        let lk = self.lookup();
        let inferred = if lk.may_have_semantic_fields {
            let legacy = settings::get_str(&self.meta.read().settings, "index.mapping.semantic_text.use_legacy_format") == Some("true");
            crate::inference::semantic::infer_document(v, &lk, legacy)?
        } else {
            serde_json::Map::new()
        };
        // Inference can take a while: after it, parse against a fresh
        // snapshot as before rather than one from before the inference.
        let lk = if lk.may_have_semantic_fields { self.lookup() } else { lk };
        crate::inference::semantic::with_inferred(inferred, || self.parse_source_inner(v, id, limits, defer, lk))
    }

    /// Whether dynamic mapping updates go through a cluster hook (the
    /// master acknowledges them first).
    pub(crate) fn has_mapping_hook(&self) -> bool {
        self.mapping_hook.read().is_some()
    }

    /// The mapping the sources `items` would grow the current one into
    /// (all their new dynamic fields, in order), if any. Items that fail to
    /// parse contribute nothing (they fail again when indexed).
    pub(crate) fn propose_dynamic_mapping<'a>(&self, items: impl IntoIterator<Item = (&'a Value, &'a str)>,
        limits: &crate::fieldtypes::limits::Limits) -> Option<(Value, Value)>
    {
        // One grown mapping for every item, no lookup built per item.
        let (base, lk) = (self.meta.read().mappings.clone(), self.lookup());
        let grown = crate::document::grow_mapping_for(items, &base, &lk, limits)?;
        Some((base.to_json(), grown.to_json()))
    }

    /// Apply the new fields every item of a shard request introduces as one
    /// mapping update (one lookup build, one metadata write), as ES applies
    /// one dynamic mapping update per bulk shard request; for an index
    /// without a cluster hook. Returns whether the mapping grew. Items whose
    /// fields it leaves out (limits, second-round fields) take the per-item
    /// path when parsed again.
    pub(crate) fn apply_dynamic_mapping_batch<'a>(&self, items: impl IntoIterator<Item = (&'a Value, &'a str)>,
        limits: &crate::fieldtypes::limits::Limits) -> Result<bool>
    {
        if let Some(authz) = current_mapping_authz() {
            authz(&self.name)?;
        }
        let _g = self.mapping_lock.lock();
        let (base, lk) = (self.meta.read().mappings.clone(), self.lookup());
        let Some(grown) = crate::document::grow_mapping_for(items, &base, &lk, limits) else { return Ok(false) };
        let nlk = Arc::new(FieldLookup::build(&grown, &lk.registry));
        {
            let mut meta = self.meta.write();
            meta.mappings = grown;
            *self.lookup.write() = nlk;
            self.schema_gen.fetch_add(1, Ordering::AcqRel);
        }
        self.persist()?;
        Ok(true)
    }

    /// Ask the cluster to accept `mappings` (the index's complete mapping
    /// with dynamic additions) once for a whole shard request: the
    /// request's authorization, then the hook (master round trip, bounded).
    pub(crate) fn request_dynamic_mapping(&self, base: &Value, mappings: &Value) -> Result<()> {
        if let Some(authz) = current_mapping_authz() {
            authz(&self.name)?;
        }
        let hook = self.mapping_hook.read().clone();
        match hook {
            Some(h) => h.mapping_update_from(self, base, mappings),
            None => Ok(()),
        }
    }

    /// `lk`: the lookup snapshot [`Self::parse_source_mode`] just took (the
    /// fast path's; the slow path below re-reads the latest).
    fn parse_source_inner(&self, v: &Value, id: &str, limits: &crate::fieldtypes::limits::Limits, defer: bool, lk: Arc<FieldLookup>) -> Result<(crate::document::ParsedDoc, bool)> {
        // Common case (no mapping change): parse under the read guard
        // instead of cloning the whole mapping per document. Mapping
        // writers only take the meta write lock briefly.
        // A document that needs new fields is not parsed further here: the
        // grown mapping and its lookup (a walk of the whole mapping, for
        // thousands of fields) are built once below under the mapping lock,
        // or once per shard request for deferred updates, not by every
        // document of a parallel batch that meets the same new field.
        {
            let meta = self.meta.read();
            if let Some(mut pd) = crate::document::parse_unless_new_fields(v, id, &meta.mappings, &lk, limits)? {
                finish_parsed(&mut pd, v, &meta.mappings, &lk, &meta.settings)?;
                return Ok((pd, false));
            }
        }
        if defer {
            return Err(EsError::new(500, DEFERRED_MAPPING, "dynamic mapping update deferred to its shard request"));
        }
        if let Some(authz) = current_mapping_authz() {
            authz(&self.name)?;
        }
        let hook = self.mapping_hook.read().clone();
        if let Some(h) = hook {
            // A6: the cluster accepts the update first (outside the mapping
            // lock: it may come back through `apply_metadata`), then it is
            // applied here unless it already arrived.
            let proposed = {
                let _g = self.mapping_lock.lock();
                let (m, lk) = (self.meta.read().mappings.clone(), self.lookup());
                crate::document::parse_with_limits(v, id, &m, &lk, limits)?.1.map(|(nm, _)| (m.to_json(), nm.to_json()))
            };
            if let Some((base, p)) = proposed {
                h.mapping_update_from(self, &base, &p)?;
            }
        }
        let _g = self.mapping_lock.lock();
        // Re-run against the latest mapping (another writer may have changed it).
        let (m, lk) = {
            let meta = self.meta.read();
            (meta.mappings.clone(), self.lookup())
        };
        let (mut pd, upd) = crate::document::parse_with_limits(v, id, &m, &lk, limits)?;
        match upd {
            Some((nm, nlk)) => {
                let nlk = Arc::new(nlk);
                let settings = {
                    let mut meta = self.meta.write();
                    meta.mappings = nm.clone();
                    // Publish the matching plan under the metadata guard, while
                    // preserving mapping publication before source finalization.
                    *self.lookup.write() = nlk.clone();
                    // Published fields must reach every shard even if source
                    // finalization subsequently rejects this document.
                    self.schema_gen.fetch_add(1, Ordering::AcqRel);
                    meta.settings.clone()
                };
                finish_parsed(&mut pd, v, &nm, &nlk, &settings)?;
                self.persist()?;
            }
            None => {
                let settings = self.meta.read().settings.clone();
                finish_parsed(&mut pd, v, &m, &lk, &settings)?;
            }
        }
        Ok((pd, true))
    }

    /// Put mapping (merge).
    pub fn put_mapping(&self, body: &Value) -> Result<()> {
        self.check_attached()?;
        let r = self.put_mapping_attached(body);
        self.attached_or(r)
    }

    fn put_mapping_attached(&self, body: &Value) -> Result<()> {
        self.check_metadata_block(true)?;
        let _g = self.mapping_lock.lock();
        let reg = self.registry.read().clone();
        let mut new = Mappings::parse_update(body, &reg)?;
        if let Some(rt) = &new.runtime {
            let meta = self.meta.read();
            crate::tsdb::check_runtime_shadow(&meta.settings, &meta.mappings, rt)?;
            crate::indexsort::check_runtime_shadows_sort(&meta.settings, Some(rt))?;
        }
        let mut m = self.meta.read().mappings.clone();
        // `_source.mode` may only be set on an index that has no effective
        // mode yet (no setting, no declared mode, source enabled); the
        // setting always wins over the mapping.
        if let Some(nv) = new.source.as_ref().and_then(|s| s.get("mode")).and_then(Value::as_str) {
            let meta = self.meta.read();
            let setting = settings::get_str(&meta.settings, "index.mapping.source.mode");
            let im = settings::get_str(&meta.settings, "index.mode");
            if setting.is_none() && !matches!(im, Some("logsdb" | "time_series")) && m.effective_source_mode().is_none() {
                let v = nv.to_ascii_lowercase();
                if let Some(s) = m.source.as_mut() {
                    s["mode"] = Value::String(v);
                } else {
                    m.source = Some(serde_json::json!({ "mode": v }));
                }
            }
        }
        // When the `index.mapping.source.mode` setting is set, the mapping's
        // `_source.mode` is locked to the setting's value: a `put_mapping`
        // that tries to change it is silently ignored (the setting wins),
        // except on logsdb / time_series indices where specific errors apply.
        if let Some(nv) = new.source.as_ref().and_then(|s| s.get("mode")).and_then(Value::as_str) {
            let meta = self.meta.read();
            let setting = settings::get_str(&meta.settings, "index.mapping.source.mode");
            let im = settings::get_str(&meta.settings, "index.mode");
            // The effective current mode: the setting if present, else the
            // index-mode default (logsdb/time_series default to synthetic),
            // else the stored mapping mode, else null.
            // The "from" value in the conflict message is the stored mapping
            // mode (null when the mapping never declared one), not the
            // effective mode.
            let stored_mode = m.source.as_ref().and_then(|s| s.get("mode")).and_then(Value::as_str).map(|s| s.to_ascii_lowercase());
            let cur = setting
                .map(|s| s.to_ascii_lowercase())
                .or_else(|| {
                    if matches!(im, Some("logsdb" | "time_series")) {
                        Some("synthetic".to_string())
                    } else {
                        None
                    }
                })
                .or(stored_mode.clone());
            let nv = nv.to_ascii_lowercase();
            if let Some(ref cur) = cur {
                if nv != *cur {
                    if matches!(im, Some("logsdb" | "time_series")) {
                        if nv == "disabled" {
                            let im = im.unwrap();
                            return Err(crate::error::EsError::mapper_parsing(format!("Failed to parse mapping: _source can not be disabled in index using [{im}] index mode")));
                        }
                        let from = stored_mode.as_deref().unwrap_or("null");
                        return Err(crate::error::EsError::illegal_argument(format!("Mapper for [_source] conflicts with existing mapper:\n\tCannot update parameter [mode] from [{}] to [{}]", from, nv)));
                    }
                    // Standard index: the setting wins; drop the mode from the
                    // update so the merge keeps the setting's value.
                    if let Some(s) = new.source.as_mut() {
                        s.as_object_mut().unwrap().remove("mode");
                    }
                }
            }
        }
        crate::inference::semantic_map::with_created(self.created_version(), || m.merge(&new))?;
        let limits = crate::fieldtypes::limits::Limits::from_settings(|k| self.setting(k));
        crate::fieldtypes::limits::check(&m, &limits)?;
        let lk = FieldLookup::build(&m, &reg);
        crate::inference::semantic_map::validate_fields(&m, &lk, Some(&crate::inference::semantic_map::declared_fields(body)))?;
        {
            let mut meta = self.meta.write();
            meta.mappings = m;
            *self.lookup.write() = Arc::new(lk);
        }
        self.schema_gen.fetch_add(1, Ordering::AcqRel);
        self.persist()
    }

    /// Dry run of [`IndexService::put_mapping`]: parse, merge and limit-check
    /// without applying (multi-index updates are all-or-nothing).
    pub fn check_put_mapping(&self, body: &Value) -> Result<()> {
        self.check_metadata_block(true)?;
        let reg = self.registry.read().clone();
        let mut new = Mappings::parse_update(body, &reg)?;
        if let Some(rt) = &new.runtime {
            let meta = self.meta.read();
            crate::tsdb::check_runtime_shadow(&meta.settings, &meta.mappings, rt)?;
            crate::indexsort::check_runtime_shadows_sort(&meta.settings, Some(rt))?;
        }
        let mut m = self.meta.read().mappings.clone();
        if let Some(nv) = new.source.as_ref().and_then(|s| s.get("mode")).and_then(Value::as_str) {
            let meta = self.meta.read();
            let setting = settings::get_str(&meta.settings, "index.mapping.source.mode");
            let im = settings::get_str(&meta.settings, "index.mode");
            if setting.is_none() && !matches!(im, Some("logsdb" | "time_series")) && m.effective_source_mode().is_none() {
                let v = nv.to_ascii_lowercase();
                if let Some(s) = m.source.as_mut() {
                    s["mode"] = Value::String(v);
                } else {
                    m.source = Some(serde_json::json!({ "mode": v }));
                }
            }
        }
        if let Some(nv) = new.source.as_ref().and_then(|s| s.get("mode")).and_then(Value::as_str) {
            let meta = self.meta.read();
            let setting = settings::get_str(&meta.settings, "index.mapping.source.mode");
            let im = settings::get_str(&meta.settings, "index.mode");
            let stored_mode = m.source.as_ref().and_then(|s| s.get("mode")).and_then(Value::as_str).map(|s| s.to_ascii_lowercase());
            let cur = setting
                .map(|s| s.to_ascii_lowercase())
                .or_else(|| {
                    if matches!(im, Some("logsdb" | "time_series")) {
                        Some("synthetic".to_string())
                    } else {
                        None
                    }
                })
                .or(stored_mode.clone());
            let nv = nv.to_ascii_lowercase();
            if let Some(ref cur) = cur {
                if nv != *cur {
                    if matches!(im, Some("logsdb" | "time_series")) {
                        if nv == "disabled" {
                            let im = im.unwrap();
                            return Err(crate::error::EsError::mapper_parsing(format!("Failed to parse mapping: _source can not be disabled in index using [{im}] index mode")));
                        }
                        let from = stored_mode.as_deref().unwrap_or("null");
                        return Err(crate::error::EsError::illegal_argument(format!("Mapper for [_source] conflicts with existing mapper:\n\tCannot update parameter [mode] from [{}] to [{}]", from, nv)));
                    }
                    if let Some(s) = new.source.as_mut() {
                        s.as_object_mut().unwrap().remove("mode");
                    }
                }
            }
        }
        crate::inference::semantic_map::with_created(self.created_version(), || m.merge(&new))?;
        let limits = crate::fieldtypes::limits::Limits::from_settings(|k| self.setting(k));
        crate::fieldtypes::limits::check(&m, &limits)
    }

}

impl IndexService {
    /// Update settings; static settings on an open index are rejected.
    pub fn put_settings(&self, flat: FlatSettings, preserve_existing: bool) -> Result<()> {
        let r = self.put_settings_reopen(flat, preserve_existing, false);
        self.attached_or(r)
    }

    /// `PUT /{index}/_settings`; with `reopen`, non-dynamic (but not final)
    /// settings close the index, apply the change and reopen it.
    pub fn put_settings_reopen(&self, flat: FlatSettings, preserve_existing: bool, reopen: bool) -> Result<()> {
        self.put_settings_impl(flat, preserve_existing, reopen, false)
    }

    /// Apply a committed cluster metadata delta. Archived values are inert
    /// persisted metadata, not user-writable settings. Public APIs must use
    /// `put_settings` so they cannot manufacture archived values.
    pub fn apply_settings_delta(&self, flat: FlatSettings) -> Result<()> {
        self.put_settings_impl(flat, false, false, true)
    }

    fn put_settings_impl(&self, flat: FlatSettings, preserve_existing: bool, reopen: bool, committed: bool) -> Result<()> {
        // Archived settings are read-only except for null resets. Expand the
        // archived wildcard against this index, never treating it as a new key.
        let flat: FlatSettings = {
            let meta = self.meta.read();
            flat.into_iter().flat_map(|(key, value)| {
                if value.is_null() && key.starts_with("archived.") && key.ends_with('*') {
                    let prefix = key.trim_end_matches('*');
                    meta.settings.keys().filter(|stored| stored.starts_with(prefix))
                        .map(|stored| (stored.clone(), Value::Null)).collect::<Vec<_>>()
                } else { vec![(key, value)] }
            }).collect()
        };
        let checked: FlatSettings = flat.iter().filter(|(key, _)| !committed || !key.starts_with("archived.")).map(|(key, value)| (key.clone(), value.clone())).collect();
        settings::validate(&checked)?;
        const FINAL: &[&str] = &["index.number_of_shards", "index.number_of_routing_shards", "index.uuid", "index.creation_date", "index.version.created", "index.mapping.source.mode"];
        if reopen && self.is_open() && flat.keys().any(|k| !settings::is_dynamic(k)) && !flat.keys().any(|k| FINAL.contains(&k.as_str())) {
            self.set_open(false)?;
            let r = self.put_settings_impl(flat, preserve_existing, false, committed);
            self.set_open(true)?;
            return r;
        }
        self.check_attached()?;
        // `index.mapping.source.mode` is final: it cannot be changed after
        // the index is created (even with `reopen`).
        if let Some(v) = flat.get("index.mapping.source.mode") {
            if !v.is_null() {
                let cur = self.setting("index.mapping.source.mode").unwrap_or_else(|| "null".to_string());
                let new = v.as_str().unwrap_or("null").to_string();
                return Err(EsError::illegal_argument(format!("Can't update non dynamic setting [index.mapping.source.mode] from [{cur}] to [{new}]")));
            }
        }
        // Changing the blocks themselves is always allowed (ES exempts them).
        if !flat.keys().all(|k| k.starts_with("index.blocks.")) {
            self.check_metadata_block(true)?;
        }
        let open = self.is_open();
        crate::tsdb::validate_put_settings(&self.meta.read().settings, &flat, open, &self.name)?;
        let mut bad = Vec::new();
        for k in flat.keys() {
            if !settings::is_dynamic(k) && open {
                bad.push(k.clone());
            }
        }
        if !bad.is_empty() {
            return Err(EsError::illegal_argument(format!(
                "Can't update non dynamic settings [[{}]] for open indices [[{}/{}]]{}",
                bad.join(", "),
                self.name,
                self.uuid,
                if bad.iter().any(|k| FINAL.contains(&k.as_str())) {
                    ""
                } else {
                    " unless the `reopen` query parameter is set to true. Alternatively, close the indices, apply the settings changes, and reopen the indices"
                }
            )));
        }
        // Only a change to `auto_expand_replicas` itself re-derives the
        // replica count here. Any other update keeps it: on a cluster the
        // master derives it from the cluster's nodes and publishes it, and
        // re-deriving it from this single engine would undo that.
        let expand = flat.contains_key("index.auto_expand_replicas");
        let changed: Vec<String> = flat.keys().filter(|k| k.starts_with("index.similarity.")).cloned().collect();
        {
            let mut m = self.meta.write();
            // Validate the merged settings before changing any (ES
            // `verifyIndexMetadata`): a rejected update leaves the index as
            // it was, in memory and on disk.
            let mut merged = m.settings.clone();
            for (k, v) in flat {
                if preserve_existing && merged.contains_key(&k) {
                    continue;
                }
                if v.is_null() {
                    merged.remove(&k);
                } else {
                    merged.insert(k, v);
                }
            }
            if expand {
                apply_auto_expand_replicas(&mut merged);
            }
            // Similarity ranges only of the similarities the update changes:
            // an index an older build created out of range still takes
            // other updates.
            crate::similarity::check_changed_ranges(&merged, &changed)?;
            let reg = registry_for(&merged);
            if !open {
                let reg = reg?;
                m.settings = merged;
                *self.registry.write() = reg;
            } else {
                m.settings = merged;
                // Dynamic index-level mapping defaults the registry carries.
                let keyword_only = reg.map(|r| r.dynamic_strings_keyword_only).unwrap_or(false);
                let changed = {
                    let mut reg = self.registry.write();
                    let changed = reg.dynamic_strings_keyword_only != keyword_only;
                    reg.dynamic_strings_keyword_only = keyword_only;
                    changed
                };
                if changed {
                    let reg = self.registry.read().clone();
                    *self.lookup.write() = Arc::new(FieldLookup::build(&m.mappings, &reg));
                }
            }
        }
        self.persist()
    }

}

impl IndexCore {
    /// Rebuild analyzers (and the field lookup that holds them) from the
    /// current settings, e.g. after a synonyms set changed
    /// (`_reload_search_analyzers`).
    pub fn reload_search_analyzers(&self) -> Result<()> {
        let _g = self.mapping_lock.lock();
        let reg = registry_for_in(&self.meta.read().settings, data_dir_of(&self.path).as_deref(), true)?;
        let lk = FieldLookup::build(&self.meta.read().mappings, &reg);
        *self.registry.write() = reg;
        *self.lookup.write() = Arc::new(lk);
        Ok(())
    }

    pub fn registry(&self) -> AnalysisRegistry {
        self.registry.read().clone()
    }

    /// The index's `index.similarity.*` configuration.
    pub fn sims(&self) -> crate::similarity::SimConfig {
        self.registry.read().sims.clone()
    }

    pub fn refresh_interval_ms(&self) -> i64 {
        self.setting("index.refresh_interval").and_then(|s| crate::date::parse_time_value(&s)).unwrap_or(1000)
    }

}

fn as_u32(v: &Value) -> Option<u32> {
    v.as_str().and_then(|x| x.parse().ok()).or_else(|| v.as_u64().map(|x| x as u32))
}

/// Logical shards: routing, the local shard copies and whole-index
/// operations fanned out over them.
impl IndexService {
    pub fn set_open(&self, open: bool) -> Result<()> {
        self.check_attached()?;
        let r = self.set_open_attached(open);
        self.attached_or(r)
    }

    fn set_open_attached(&self, open: bool) -> Result<()> {
        self.meta.write().open = open;
        if open {
            // Reopening recovers the shards: the analyzers are rebuilt with
            // the current synonyms sets first (ES `beforeIndexShardRecovery`
            // → `reloadSearchAnalyzers`); a failure leaves every primary
            // unassigned (`shards_acknowledged: false`, health red).
            match self.reload_search_analyzers() {
                Ok(()) => {
                    self.open_unallocated();
                }
                Err(e) => {
                    for sh in self.local_shards() {
                        self.close_shard(sh.id)?;
                    }
                    self.mark_unallocated_failed(&e);
                }
            }
            self.refresh_ts_sorted();
        }
        if !open {
            for s in self.local_shards() {
                s.shard.flush()?;
            }
            // A closed index has no allocation failure: its shards recover
            // without analyzers (ES reports it green).
            self.failed.write().clear();
        }
        self.persist()
    }

    /// Shards whose primary is unassigned after a failed allocation
    /// (shard → the failure), see [`IndexService::try_allocate`].
    pub fn failed_shards(&self) -> BTreeMap<u32, EsError> {
        self.failed.read().clone()
    }

    /// Number of shard copies open on this node.
    pub fn active_shards(&self) -> usize {
        self.shards.read().iter().flatten().count()
    }

    /// ES `unassigned.details` of failed shard `n` (`None` when it is not failed).
    pub fn allocation_failure_details(&self, n: u32) -> Option<String> {
        let e = self.failed.read().get(&n)?.clone();
        let mut out = format!(
            "failed shard on node [{node}]: failed recovery, failure org.elasticsearch.indices.recovery.RecoveryFailedException: [{idx}][{n}]: Recovery failed on {{{node}}}\nCaused by: [{idx}/{uuid}][[{idx}][{n}]] org.elasticsearch.index.shard.IndexShardRecoveryException: failed to recover from gateway",
            node = crate::node::NODE_ID,
            idx = self.name,
            uuid = self.uuid
        );
        let mut cur = Some(&e);
        while let Some(c) = cur {
            let class = match c.kind.as_str() {
                "resource_not_found_exception" => "org.elasticsearch.ResourceNotFoundException".to_string(),
                "illegal_argument_exception" => "java.lang.IllegalArgumentException".to_string(),
                "parse_exception" => "java.text.ParseException".to_string(),
                other => format!("org.elasticsearch.{}", other.split('_').map(|p| { let mut ch = p.chars(); ch.next().map(|f| f.to_uppercase().collect::<String>() + ch.as_str()).unwrap_or_default() }).collect::<String>()),
            };
            out.push_str(&format!("\nCaused by: {class}: {}", c.reason));
            cur = c.caused_by.as_deref();
        }
        Some(out)
    }

    /// Record `e` as the allocation failure of every shard not open here.
    fn mark_unallocated_failed(&self, e: &EsError) {
        let n = self.shards.read().len() as u32;
        let mut f = self.failed.write();
        for i in 0..n {
            if self.shard(i).is_none() {
                f.insert(i, e.clone());
            }
        }
    }

    /// Open every shard copy not open here (from its store when one is on
    /// disk, else empty); a copy that fails to open stays failed.
    fn open_unallocated(&self) {
        let n = self.shards.read().len() as u32;
        for i in 0..n {
            if self.shard(i).is_some() {
                self.failed.write().remove(&i);
                continue;
            }
            let rec = if self.path.join(format!("shard-{i}")).join("shard_state.json").exists() { RecoverySource::ExistingStore } else { RecoverySource::Empty };
            match self.open_shard(i, ShardOpenSpec::local_primary(rec)) {
                Ok(_) => {
                    self.failed.write().remove(&i);
                }
                Err(e) => {
                    self.failed.write().insert(i, e);
                }
            }
        }
    }

    /// A cluster mirror (no local copies) whose analyzers now build: forget
    /// the shard failures recorded when the index was created with them
    /// broken. The cluster layer allocates and opens the real copies; the
    /// mirror must stop answering `no_shard_available` for e.g. `_analyze`.
    pub fn clear_failed_without_copies(&self) {
        if self.local_shards().is_empty() {
            self.failed.write().clear();
        }
    }

    /// Retry allocating the failed primaries (`POST _cluster/reroute?retry_failed`):
    /// rebuild the analyzers with the current synonyms sets and open the
    /// unopened copies. Returns whether every shard is now allocated.
    pub fn try_allocate(&self) -> bool {
        if self.failed.read().is_empty() {
            return true;
        }
        match self.reload_search_analyzers() {
            Ok(()) => self.open_unallocated(),
            Err(e) => self.mark_unallocated_failed(&e),
        }
        self.failed.read().is_empty()
    }

    /// The error a document operation on the missing copy of shard `n` gets:
    /// `unavailable_shards_exception` for an unassigned primary (ES
    /// `TransportReplicationAction`), else `shard_not_found_exception`.
    fn missing_copy(&self, n: u32) -> EsError {
        if self.failed.read().contains_key(&n) {
            return EsError::new(503, "unavailable_shards_exception", format!("[{}][{n}] primary shard is not active Timeout: [1m], request: [BulkShardRequest [[{}][{n}]] containing [1] requests]", self.name, self.name));
        }
        EsError::shard_not_found(&self.name, &self.uuid, n)
    }

    /// `index.number_of_shards`.
    pub fn number_of_shards(&self) -> u32 {
        shard_count(&self.meta.read().settings)
    }

    /// The index's routing table parameters (shards, routing shards,
    /// partition size), as `elasticsearch-routing` computes shard ids.
    pub fn index_routing(&self) -> elasticsearch_routing::IndexRouting {
        let m = self.meta.read();
        let n = shard_count(&m.settings);
        let rns = m.settings.get("index.number_of_routing_shards").and_then(as_u32).filter(|x| *x > 0).unwrap_or_else(|| elasticsearch_routing::calculate_num_routing_shards(n));
        let ps = m.settings.get("index.routing_partition_size").and_then(as_u32).unwrap_or(1);
        let created = crate::version::IndexVersion::from_setting(m.settings.get("index.version.created"));
        elasticsearch_routing::IndexRouting::new(n, rns, ps).with_modulo(created.on_or_after(crate::version::IndexVersion::MOD_ROUTING_FUNCTION))
    }

    /// The shard a single-document operation on `id` (with `routing`)
    /// goes to: murmur3 over the routing value (or `_id`), ES
    /// `OperationRouting`. Time series indices route by the routing hash
    /// carried in their `_id`.
    pub fn route(&self, id: &str, routing: Option<&str>) -> Result<u32> {
        let r = self.index_routing();
        if r.number_of_shards == 1 {
            return Ok(0);
        }
        if routing.is_none() && crate::tsdb::is_time_series(&self.meta.read().settings) {
            if let Some(h) = tsdb_routing_hash(id) {
                return Ok(r.hash_to_shard(h));
            }
        }
        r.shard_id(id, routing).map_err(|_| {
            EsError::new(400, "routing_missing_exception", format!("routing is required for [{}]/[{id}]", self.name)).with("index_uuid", "_na_").with("index", self.name.clone())
        })
    }

    /// Shards a search with `?routing=r1,r2` visits (every shard when `None`).
    pub fn search_shards_for(&self, routing: Option<&str>) -> Vec<u32> {
        let r = self.index_routing();
        match routing.map(|s| s.split(',').map(str::trim).filter(|s| !s.is_empty()).collect::<Vec<_>>()) {
            Some(vals) if !vals.is_empty() => {
                let mut out: Vec<u32> = vals.iter().flat_map(|v| r.search_shards(v)).collect();
                out.sort_unstable();
                out.dedup();
                out
            }
            _ => (0..r.number_of_shards).collect(),
        }
    }

    /// The local copy of shard `n`, if open on this node.
    pub fn shard(&self, n: u32) -> Option<Arc<IndexShard>> {
        self.shards.read().get(n as usize).cloned().flatten()
    }

    /// Every shard copy open on this node, by shard number.
    pub fn local_shards(&self) -> Vec<Arc<IndexShard>> {
        self.shards.read().iter().flatten().cloned().collect()
    }

    /// Deprecated shim (cluster code pinned to the single-shard engine):
    /// the local copy of shard 0.
    ///
    /// # Panics
    /// When shard 0 is not open on this node.
    #[deprecated(note = "use IndexService::shard(n) / local_shards()")]
    pub fn primary_shard(&self) -> Arc<IndexShard> {
        self.shard(0).unwrap_or_else(|| panic!("shard 0 of [{}] is not open on this node", self.name))
    }

    /// Open the local copy of shard `n` (A3: fails with
    /// `shard_lock_obtain_failed_exception` while another copy of `n` is
    /// open in this or another process).
    pub fn open_shard(&self, n: u32, spec: ShardOpenSpec) -> Result<Arc<IndexShard>> {
        let prepared = self.prepare_shard(n, spec)?;
        self.install_prepared_shard(prepared)
    }

    /// Open a physical copy without publishing it to searches or cluster roles.
    /// The returned owner closes it if abandoned. Callers must invalidate pending
    /// preparation when routing/term changes, then revalidate authority before
    /// installation; the engine epoch protects slot replacement, not routing.
    ///
    /// The lifecycle lock is held only to validate and register the open and
    /// to register its result, never across the open itself (a recovered
    /// copy's graph catch-up or rebuild can take long, and a detach, which
    /// the cluster applier runs to delete or close the index, takes that
    /// lock exclusively). Meanwhile a detach or a reset of the slot
    /// (`invalidate_shard_preparation`, `remove_shard`,
    /// `set_aside_shard_store`) cancels the open: it fails at its next
    /// cancellation point ([`rustic_core::cancel`]) and releases its shard
    /// lock.
    pub fn prepare_shard(&self, n: u32, spec: ShardOpenSpec) -> Result<PreparedShard> {
        let token = Arc::new(AtomicBool::new(false));
        let epoch = {
            let _lifecycle = self.preparation_lifecycle.read();
            self.check_attached()?;
            let slots = self.shards.read();
            if self.view || n as usize >= slots.len() {
                return Err(EsError::illegal_argument("invalid shard preparation target"));
            }
            if slots[n as usize].is_some() {
                return Err(EsError::shard_lock_failed(&self.name, &self.uuid, n, "preparing shard"));
            }
            self.openings.lock().push((n, token.clone()));
            self.preparation_epochs[n as usize].load(Ordering::Relaxed)
        };
        let opened = rustic_core::cancel::with_cancel(token.clone(), || {
            rustic_core::cancel::check()?;
            IndexShard::open(&self.core, n, spec)
        });
        let _lifecycle = self.preparation_lifecycle.read();
        self.openings.lock().retain(|(_, t)| !Arc::ptr_eq(t, &token));
        let copy = opened.map_err(|e| if token.load(Ordering::Acquire) { self.cancelled_open(n) } else { e })?;
        // Cancelled after the open finished (or detached meanwhile): the
        // copy goes before anyone reuses its files.
        if token.load(Ordering::Acquire) || self.detached.load(Ordering::SeqCst) || self.preparation_epochs[n as usize].load(Ordering::Relaxed) != epoch {
            copy.fence();
            return Err(self.cancelled_open(n));
        }
        let mut registered = self.prepared_copies.lock();
        registered.retain(|weak| weak.upgrade().is_some_and(|copy| !copy.is_closed()));
        registered.push(Arc::downgrade(&copy));
        Ok(PreparedShard { owner: self.core.clone(), epoch, copy: Some(copy) })
    }

    /// The error of an open that a detach or a slot reset cancelled.
    fn cancelled_open(&self, n: u32) -> EsError {
        if self.detached.load(Ordering::SeqCst) {
            return EsError::new(503, COPY_REPLACED, format!("the local copy of index [{}] was replaced; retry", self.name));
        }
        EsError::shard_lock_failed(&self.name, &self.uuid, n, "opening a shard whose slot was reset meanwhile")
    }

    /// Borrow the physical copy for unpublished WAL preparation. This does not
    /// authorize installation: callers must retain the owner and revalidate
    /// routing plus the slot epoch at installation. Dropping the owner fences
    /// and drains the copy even if a preparation worker still holds this Arc.
    pub fn prepared_shard_copy(&self, prepared: &PreparedShard) -> Result<Arc<IndexShard>> {
        let _lifecycle = self.preparation_lifecycle.read();
        self.check_attached()?;
        let copy = prepared.copy.as_ref().ok_or_else(|| EsError::illegal_argument("prepared shard already installed"))?;
        let slots = self.shards.read();
        let n = copy.id as usize;
        if self.view || !Arc::ptr_eq(&self.core, &prepared.owner) || n >= slots.len()
            || slots[n].is_some() || copy.is_closed()
            || self.preparation_epochs[n].load(Ordering::Relaxed) != prepared.epoch {
            return Err(EsError::shard_lock_failed(&self.name, &self.uuid, copy.id, "accessing stale prepared shard"));
        }
        Ok(copy.clone())
    }

    /// Cancel any unpublished opening for this slot. Does not close a serving copy.
    pub fn invalidate_shard_preparation(&self, n: u32) -> Result<()> {
        let slots = self.shards.write();
        if n as usize >= slots.len() || self.view {
            return Err(EsError::illegal_argument("invalid shard preparation target"));
        }
        self.preparation_epochs[n as usize].fetch_add(1, Ordering::Relaxed);
        self.cancel_openings(Some(n));
        Ok(())
    }

    pub fn install_prepared_shard(&self, mut prepared: PreparedShard) -> Result<Arc<IndexShard>> {
        self.try_install_prepared_shard(&mut prepared)
    }

    /// Attempt installation without dropping a rejected owner. Ordered cluster
    /// callbacks can release their authority lock before draining that owner.
    pub fn try_install_prepared_shard(&self, prepared: &mut PreparedShard) -> Result<Arc<IndexShard>> {
        let _lifecycle = self.preparation_lifecycle.read();
        self.check_attached()?;
        if prepared.copy.is_none() {
            return Err(EsError::illegal_argument("prepared shard already installed"));
        }
        let copy = prepared.copy.as_ref().expect("prepared copy is owned");
        let n = copy.id as usize;
        let mut slots = self.shards.write();
        if self.view || !Arc::ptr_eq(&self.core, &prepared.owner) || n >= slots.len()
            || slots[n].is_some() || copy.is_closed()
            || self.preparation_epochs[n].load(Ordering::Relaxed) != prepared.epoch {
            return Err(EsError::shard_lock_failed(&self.name, &self.uuid, copy.id, "installing stale prepared shard"));
        }
        let copy = prepared.copy.take().expect("prepared copy is owned");
        slots[n] = Some(copy.clone());
        self.preparation_epochs[n].fetch_add(1, Ordering::Relaxed);
        Ok(copy)
    }

    /// Close the local copy of shard `n`: fence new operations (they fail
    /// with `index_shard_closed_exception`), wait for in-flight ones, make
    /// every acknowledged operation durable (A4), release the shard lock.
    pub fn close_shard(&self, n: u32) -> Result<()> {
        let s = {
            let mut slots = self.shards.write();
            let slot = slots.get_mut(n as usize).ok_or_else(|| EsError::shard_not_found(&self.name, &self.uuid, n))?;
            self.preparation_epochs[n as usize].fetch_add(1, Ordering::Relaxed);
            slot.take().ok_or_else(|| EsError::shard_not_found(&self.name, &self.uuid, n))?
        };
        s.close()
    }

    /// Recover the copy `failed` (its engine failed: a tragic event) from
    /// its own store, as ES reallocates a failed shard and recovers it from
    /// the store: fence it (no commit: its translog is closed), then open it
    /// again (translog replay, seq_no gap fill). While it cannot be opened
    /// the shard is reported failed (red).
    pub fn recover_failed_shard(&self, failed: &Arc<IndexShard>) -> Result<()> {
        // ES `index.allocation.max_retries`: a copy that keeps failing (a
        // tragedy its own recovery reproduces) is retried with backoff, at
        // most MAX_RECOVERY_ATTEMPTS times, then left failed (red).
        {
            let now = std::time::Instant::now();
            let mut attempts = recovery_attempts().lock();
            let a = attempts.entry((self.uuid.clone(), failed.id)).or_insert(RecoveryAttempts { count: 0, next: now, healthy_since: None });
            // The count restarts only after the copy stayed healthy for a
            // while since its last successful recovery; once over the
            // limit the shard stays failed (until the index is reopened or
            // deleted).
            restart_count(a, now);
            if a.count >= MAX_RECOVERY_ATTEMPTS {
                let reason = failed.shard.tragic_error().unwrap_or_default();
                let e = EsError::new(500, "shard_recovery_exception", format!(
                    "shard [{}][{}] failed {MAX_RECOVERY_ATTEMPTS} times recovering from its store, not retrying: {reason}", self.name, failed.id));
                self.failed.write().insert(failed.id, e.clone());
                return Err(e);
            }
            if now < a.next {
                return Ok(());
            }
            a.count += 1;
            a.next = now + RECOVERY_BACKOFF * (1u32 << (a.count - 1).min(10));
        }
        let Some(removed) = self.detach_shard_if_same(failed) else { return Ok(()) };
        // Waits for and stops the failed copy's writes to its files (its
        // detached seal / merge jobs included) before it opens again.
        removed.fence();
        let n = removed.id;
        let spec = ShardOpenSpec { allocation_id: removed.allocation_id.clone(), primary: removed.is_primary(), primary_term: removed.shard.primary_term(), recovery: RecoverySource::ExistingStore };
        drop(removed);
        match self.open_shard(n, spec) {
            Ok(_) => {
                self.failed.write().remove(&n);
                if let Some(a) = recovery_attempts().lock().get_mut(&(self.uuid.clone(), n)) {
                    a.healthy_since = Some(std::time::Instant::now());
                }
                Ok(())
            }
            Err(e) => {
                self.failed.write().insert(n, e.clone());
                Err(e)
            }
        }
    }

    /// Retire only the physical copy owned by the caller. A stale recovery
    /// completion must not close a replacement installed at the same shard ID.
    /// The identity check and removal are atomic; draining happens outside the
    /// slot lock so unrelated shard lookups can continue.
    pub fn close_shard_if_same(&self, expected: &Arc<IndexShard>) -> Result<bool> {
        let Some(removed) = self.detach_shard_if_same(expected) else { return Ok(false) };
        removed.close()?;
        Ok(true)
    }

    /// Fence and detach without waiting for operations or filesystem work.
    /// The caller must drain the returned copy with `close` before reusing its
    /// directory. The OS shard lock remains held until that close completes.
    pub fn detach_shard_if_same(&self, expected: &Arc<IndexShard>) -> Option<Arc<IndexShard>> {
        let removed = {
            let mut slots = self.shards.write();
            let slot = slots.get_mut(expected.id as usize)?;
            if !slot.as_ref().is_some_and(|current| Arc::ptr_eq(current, expected)) {
                return None;
            }
            self.preparation_epochs[expected.id as usize].fetch_add(1, Ordering::Relaxed);
            expected.fence_new_operations();
            slot.take().expect("identity checked while holding slot lock")
        };
        Some(removed)
    }

    /// Close (if open) the local copy of shard `n` and move its data
    /// (`shard-<n>/`) to `dest` under the shard lock: a peer recovery keeps
    /// the old files aside to reuse the ones identical to the source's
    /// (ES `Store.MetadataSnapshot.recoveryDiff`). `false` when there was no
    /// data.
    pub fn set_aside_shard_store(&self, n: u32, dest: &Path) -> Result<bool> {
        let open = {
            let mut slots = self.shards.write();
            if let Some(epoch) = self.preparation_epochs.get(n as usize) {
                epoch.fetch_add(1, Ordering::Relaxed);
            }
            // An open of this slot in progress lets go of the shard lock.
            self.cancel_openings(Some(n));
            slots.get_mut(n as usize).and_then(Option::take)
        };
        if let Some(s) = open {
            s.close()?;
        }
        IndexShard::move_store(&self.core, n, dest)
    }

    /// Close (if open) and forget the local copy of shard `n`; with
    /// `delete_data`, delete `shard-<n>/` under the shard lock.
    pub fn remove_shard(&self, n: u32, delete_data: bool) -> Result<()> {
        let open = {
            let mut slots = self.shards.write();
            if let Some(epoch) = self.preparation_epochs.get(n as usize) {
                epoch.fetch_add(1, Ordering::Relaxed);
            }
            self.cancel_openings(Some(n));
            slots.get_mut(n as usize).and_then(Option::take)
        };
        if let Some(s) = open {
            s.close()?;
        }
        if delete_data {
            IndexShard::delete_store(&self.core, n)?;
        }
        Ok(())
    }

    /// Take the field ids a shard copy's files were written with (core
    /// field key → core id: the schema of their core commit) as this
    /// index's, so files received from a peer or restored from a snapshot
    /// are read under the fields they hold. Ids are assigned as fields first
    /// appear (in sorted order within one mapping change), so a copy whose
    /// mapping grew over several changes numbers its fields differently
    /// from a shell made from the final mapping. Fields the files do not
    /// name keep their ids unless the files use one, then they move above
    /// every id in use; keys this mapping lacks (`_doc_count`, a field whose
    /// mapping has not arrived yet) are reserved under the files' ids. A copy
    /// already open here must hold no operation: it is reopened empty with
    /// the new ids (the other shards of a full-open index); one that holds
    /// any refuses the change. Returns whether the ids changed.
    pub fn adopt_core_ids(&self, ids: &BTreeMap<String, u16>) -> Result<bool> {
        // Commits written before the stored-fields block's field was named
        // by its key call it `_stored_fields`: the same field.
        let renamed;
        let ids = if ids.contains_key(crate::mapping::LEGACY_STORED_BLOCK_NAME) && !ids.contains_key(crate::mapping::STORED_BLOCK_KEY) {
            renamed = ids.iter().map(|(k, v)| (if k == crate::mapping::LEGACY_STORED_BLOCK_NAME { crate::mapping::STORED_BLOCK_KEY.to_string() } else { k.clone() }, *v)).collect::<BTreeMap<String, u16>>();
            &renamed
        } else {
            ids
        };
        let reopen = {
            let _g = self.mapping_lock.lock();
            let mut m = self.meta.read().mappings.clone();
            if ids.iter().all(|(k, id)| m.core_ids.get(k) == Some(id)) {
                return Ok(false);
            }
            let mut used = std::collections::BTreeSet::new();
            for id in ids.values() {
                if *id < crate::mapping::FIRST_USER_FIELD || !used.insert(*id) {
                    return Err(EsError::new(500, "corrupt_index_exception", format!("invalid field id [{id}] in the schema of the files adopted by [{}]", self.name)));
                }
            }
            let copies = self.local_shards();
            if let Some(busy) = copies.iter().find(|c| c.shard.max_seq_no() >= 0 || c.doc_count_realtime() != (0, 0)) {
                return Err(EsError::new(500, "illegal_state_exception", format!("cannot change the field ids of [{}]: its copy of shard [{}] holds operations", self.name, busy.id)));
            }
            let above = |v: Option<&u16>| v.map_or(0, |m| m + 1);
            let mut next = m.next_core_id.max(above(used.iter().next_back())).max(above(m.core_ids.values().max()));
            let mut adopted = BTreeMap::new();
            for (k, id) in &m.core_ids {
                let id = match ids.get(k) {
                    Some(theirs) => *theirs,
                    None if used.contains(id) => {
                        next += 1;
                        next - 1
                    }
                    None => *id,
                };
                adopted.insert(k.clone(), id);
            }
            for (k, id) in ids {
                adopted.entry(k.clone()).or_insert(*id);
            }
            tracing::info!(index = %self.name, "taking the field ids of adopted files");
            m.core_ids = adopted;
            m.next_core_id = next;
            let lookup = FieldLookup::build(&m, &self.registry.read());
            {
                let mut meta = self.meta.write();
                meta.mappings = m;
                *self.lookup.write() = Arc::new(lookup);
            }
            self.schema_gen.fetch_add(1, Ordering::AcqRel);
            self.persist()?;
            copies
        };
        // Their cores were created with the old ids.
        for copy in reopen {
            let (n, spec) = (copy.id, ShardOpenSpec { allocation_id: copy.allocation_id.clone(), primary: copy.is_primary(), primary_term: copy.shard.primary_term(), recovery: RecoverySource::Empty });
            drop(copy);
            self.remove_shard(n, true)?;
            self.open_shard(n, spec)?;
        }
        Ok(true)
    }

    /// Apply cluster metadata verbatim: `mappings` (ES JSON, merged into
    /// the current mapping, which only grows) and the complete `settings`.
    /// Monotonic in `version`: a version not above the last applied one is
    /// a no-op.
    pub fn apply_metadata(&self, mappings: &Value, settings_in: &FlatSettings, version: u64) -> Result<()> {
        let _g = self.mapping_lock.lock();
        if version <= self.metadata_version() {
            return Ok(());
        }
        let reg = registry_for_in(settings_in, data_dir_of(&self.path).as_deref(), true).or_else(|_| registry_for(settings_in))?;
        let body = match mappings.as_object() {
            Some(o) if o.len() == 1 && o.contains_key("_doc") => &mappings["_doc"],
            _ => mappings,
        };
        let mut new = Mappings::parse_update(body, &reg)?;
        let mut m = self.meta.read().mappings.clone();
        crate::inference::semantic_map::with_created(self.created_version(), || m.merge(&new))?;
        crate::tsdb::prepare_mappings(settings_in, &mut m, &reg)?;
        let lk = FieldLookup::build(&m, &reg);
        {
            let mut meta = self.meta.write();
            let mut s = settings_in.clone();
            s.insert("index.uuid".into(), json!(self.uuid));
            meta.settings = s;
            meta.mappings = m;
            *self.lookup.write() = Arc::new(lk);
        }
        *self.registry.write() = reg;
        self.schema_gen.fetch_add(1, Ordering::AcqRel);
        self.metadata_version.store(version, Ordering::Release);
        self.refresh_ts_sorted();
        self.persist()
    }

    /// Whole-index conveniences route each operation to its shard and act
    /// on the local copy (a sparse, cluster-managed service: an operation
    /// whose shard has no local copy fails with `shard_not_found_exception`).
    /// An ephemeral single-shard view refuses them.
    fn whole(&self) -> Result<()> {
        if self.view {
            return Err(EsError::new(500, "illegal_state_exception", format!("index [{}] is a single shard view: whole-index operations must go through its shard copies", self.name)));
        }
        Ok(())
    }

    /// The local copy of the shard `id` routes to (`shard_not_found_exception`
    /// when that shard has no copy open here).
    fn routed(&self, id: &str, routing: Option<&str>) -> Result<Arc<IndexShard>> {
        self.whole()?;
        let n = self.route(id, routing)?;
        self.shard(n).ok_or_else(|| self.missing_copy(n))
    }

    /// Apply a batch of writes (bulk / single doc APIs): each is routed to
    /// its shard (murmur3 over `routing` or `_id`; auto ids are generated
    /// first) and every shard applies its sub-batch. Results keep the
    /// request order.
    pub fn write(&self, reqs: Vec<WriteReq>, sync: bool) -> Vec<Result<WriteResult>> {
        if let Err(e) = self.whole() {
            return reqs.iter().map(|_| Err(e.clone())).collect();
        }
        let occ = |r: &WriteReq| match r {
            WriteReq::Index { cond, .. } | WriteReq::Delete { cond, .. } => cond.if_seq_no.is_some() || cond.if_primary_term.is_some(),
        };
        if reqs.iter().any(occ) && self.seq_no_disabled() {
            let n = reqs.len();
            let mut out: Vec<Option<Result<WriteResult>>> = (0..n).map(|_| None).collect();
            let (mut pos, mut rest) = (Vec::new(), Vec::new());
            for (i, r) in reqs.into_iter().enumerate() {
                if occ(&r) {
                    let (id, routing) = match &r {
                        WriteReq::Index { id, routing, .. } => (id.clone().unwrap_or_default(), routing.clone()),
                        WriteReq::Delete { id, routing, .. } => (id.clone(), routing.clone()),
                    };
                    out[i] = Some(Err(self.seq_no_disabled_err("o_c_c_not_supported_exception", "Optimistic concurrency control is not supported on indices with sequence numbers disabled", &id, routing.as_deref())));
                } else {
                    pos.push(i);
                    rest.push(r);
                }
            }
            if !rest.is_empty() {
                for (i, r) in pos.into_iter().zip(self.write(rest, sync)) {
                    out[i] = Some(r);
                }
            }
            return out.into_iter().map(|r| r.unwrap_or_else(|| Err(EsError::internal("missing write result")))).collect();
        }
        let slots = self.shards.read().clone();
        if slots.len() == 1 {
            return match &slots[0] {
                Some(s) => s.write(reqs, sync),
                None => reqs.iter().map(|_| Err(self.missing_copy(0))).collect(),
            };
        }
        if let Err(e) = self.check_open() {
            return reqs.iter().map(|_| Err(e.clone())).collect();
        }
        let ts = crate::tsdb::is_time_series(&self.meta.read().settings);
        // Auto ids: k-ordered in logsdb / columnar indices, as ES makes them.
        let k_ordered = k_ordered_ids(crate::settings::get_str(&self.meta.read().settings, "index.mode"));
        let n = reqs.len();
        let mut out: Vec<Option<Result<WriteResult>>> = (0..n).map(|_| None).collect();
        let mut groups: Vec<(Vec<usize>, Vec<WriteReq>)> = (0..slots.len()).map(|_| (Vec::new(), Vec::new())).collect();
        for (i, r) in reqs.into_iter().enumerate() {
            let (r, shard) = match r {
                WriteReq::Index { id, source, routing, cond } => {
                    let shard = match (&id, ts) {
                        (_, true) if routing.is_none() => Ok(self.tsdb_route_source(&source)),
                        (Some(id), _) => self.route(id, routing.as_deref()),
                        (None, _) => Ok(u32::MAX),
                    };
                    let (id, shard) = match (id, shard) {
                        (None, Ok(u32::MAX)) => {
                            let a = if k_ordered { auto_id_k_ordered() } else { auto_id() };
                            let s = self.route(&a, routing.as_deref());
                            (Some(a), s)
                        }
                        (id, s) => (id, s),
                    };
                    (WriteReq::Index { id, source, routing, cond }, shard)
                }
                WriteReq::Delete { id, routing, cond } => {
                    let s = self.route(&id, routing.as_deref());
                    (WriteReq::Delete { id, routing, cond }, s)
                }
            };
            match shard {
                Ok(s) => {
                    groups[s as usize].0.push(i);
                    groups[s as usize].1.push(r);
                }
                Err(e) => out[i] = Some(Err(e)),
            }
        }
        let authz = current_mapping_authz();
        let run = |(sn, (pos, batch)): (usize, (Vec<usize>, Vec<WriteReq>))| -> Vec<(usize, Result<WriteResult>)> {
            if batch.is_empty() {
                return Vec::new();
            }
            // (rayon workers: carry the request's mapping authorizer over)
            let _authz = enter_mapping_authz(authz.clone());
            let res = match &slots[sn] {
                Some(s) => s.write(batch, sync),
                None => batch.iter().map(|_| Err(self.missing_copy(sn as u32))).collect(),
            };
            pos.into_iter().zip(res).collect()
        };
        // In parallel only when two or more shards have writes. A shard-level
        // bulk (what a coordinating node sends each copy) has one group: the
        // rayon hop gained nothing and queued the whole write behind the
        // global pool's other jobs (a mean ~70 ms of a ~190 ms explicit-id
        // vector bulk on a 32-core node), with the write thread held.
        let parallel = groups.iter().filter(|(_, batch)| !batch.is_empty()).nth(1).is_some();
        let done: Vec<Vec<(usize, Result<WriteResult>)>> = if n >= 64 && parallel {
            use rayon::prelude::*;
            groups.into_par_iter().enumerate().map(run).collect()
        } else {
            groups.into_iter().enumerate().map(run).collect()
        };
        for (i, r) in done.into_iter().flatten() {
            out[i] = Some(r);
        }
        out.into_iter().map(|r| r.unwrap_or_else(|| Err(EsError::internal("write not routed")))).collect()
    }

    /// Shard of a time series document (its routing hash over the
    /// `routing_path` dimensions); shard 0 when the source cannot be
    /// routed (the shard reports the parse error).
    fn tsdb_route_source(&self, source: &str) -> u32 {
        let id = serde_json::from_str::<Value>(source).ok().and_then(|v| {
            let meta = self.meta.read();
            self.lookup().tsdb_fields.build_ids(&meta.mappings, &meta.settings, &v).ok().map(|ids| ids.id)
        });
        id.and_then(|id| tsdb_routing_hash(&id)).map(|h| self.index_routing().hash_to_shard(h)).unwrap_or(0)
    }

    /// Realtime get of `id` on the shard it routes to (by `_id`).
    pub fn get(&self, id: &str) -> Result<Option<GetResult>> {
        self.check_open()?;
        self.check_read_block()?;
        let s = self.routed(id, None)?;
        s.check_usable()?;
        s.shard.get(id)
    }

    /// GET as ES serves it: routed by `routing` (or `_id`) to one shard;
    /// `_routing: {required: true}` is enforced and `realtime=false` on an
    /// index with refresh disabled only sees refreshed documents.
    pub fn get_routed(&self, id: &str, routing: Option<&str>, realtime: bool) -> Result<Option<GetResult>> {
        self.check_open()?;
        self.check_read_block()?;
        if routing.is_none() && self.meta.read().mappings.routing_required() {
            return Err(EsError::new(400, "routing_missing_exception", format!("routing is required for [{}]/[{id}]", self.name))
                .with("index_uuid", "_na_")
                .with("index", self.name.clone()));
        }
        self.routed(id, routing)?.get(id, routing, realtime)
    }

    /// Realtime lookup without `get` stats, on the shard `id` routes to.
    pub fn peek(&self, id: &str, routing: Option<&str>) -> Result<Option<GetResult>> {
        self.routed(id, routing)?.shard.peek(id)
    }

    /// `_update`: partial doc / upsert / doc_as_upsert / script with
    /// retries, on the shard `id` routes to.
    fn seq_no_disabled_err(&self, kind: &str, reason: &str, id: &str, routing: Option<&str>) -> EsError {
        let shard = self.route(id, routing).unwrap_or(0);
        EsError::new(400, kind, reason).with("index_uuid", self.uuid.clone()).with("shard", shard.to_string()).with("index", self.name.clone())
    }

    pub fn update(&self, id: &str, body: &Value, routing: Option<String>, retry_on_conflict: usize, cond: WriteCond) -> Result<(WriteResult, Option<String>)> {
        self.update_source(id,body,None,routing,retry_on_conflict,cond)
    }
    pub fn update_source(&self,id:&str,body:&Value,raw:Option<&crate::source_json::Source>,routing:Option<String>,retry_on_conflict:usize,cond:WriteCond)->Result<(WriteResult,Option<String>)>{
        self.check_open()?;
        if self.seq_no_disabled() {
            return Err(self.seq_no_disabled_err("update_not_supported_exception", "Updates are not supported on indices with sequence numbers disabled", id, routing.as_deref()));
        }
        let s = self.routed(id, routing.as_deref())?;
        s.update_source(id, body, raw, routing, retry_on_conflict, cond)
    }

    /// Live and deleted documents over the local shards, as searches see
    /// them (refresh-based, as ES `docs.count`).
    pub fn doc_count(&self) -> (u64, u64) {
        self.local_shards().iter().map(|s| s.doc_count()).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    }

    /// [`IndexService::doc_count`] over every completed write, refreshed or
    /// not.
    pub fn doc_count_realtime(&self) -> (u64, u64) {
        self.local_shards().iter().map(|s| s.doc_count_realtime()).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    }

    /// Refresh the shard document `id` (with `routing`) lives on (a
    /// single-document write's `?refresh`); every local shard when it
    /// cannot be routed.
    pub fn refresh_doc_shard(&self, id: &str, routing: Option<&str>) -> Result<()> {
        match self.route(id, routing).ok().and_then(|n| self.shard(n)) {
            Some(s) => s.shard.refresh(),
            None => self.refresh(),
        }
    }

    /// `refresh=wait_for` on the shard holding `(id, routing)` (every
    /// local shard when unknown); whether a refresh had to be forced.
    pub fn wait_for_refresh_doc(&self, id: &str, routing: Option<&str>) -> Result<bool> {
        match self.route(id, routing).ok().and_then(|n| self.shard(n)) {
            Some(s) => s.shard.wait_for_refresh(self.max_refresh_listeners(), WAIT_FOR_REFRESH_MAX),
            None => self.wait_for_refresh(),
        }
    }

    /// `refresh=wait_for` on every local shard.
    pub fn wait_for_refresh(&self) -> Result<bool> {
        let mut forced = false;
        for s in self.local_shards() {
            forced |= s.shard.wait_for_refresh(self.max_refresh_listeners(), WAIT_FOR_REFRESH_MAX)?;
        }
        Ok(forced)
    }

    /// [`IndexService::wait_for_refresh_doc`] without waiting: `done` is
    /// called by the refresh that makes the write visible (ES refresh
    /// listeners; the calling write thread returns at once).
    pub fn add_refresh_listener_doc(&self, id: &str, routing: Option<&str>, done: crate::shard::RefreshCallback) {
        match self.route(id, routing).ok().and_then(|n| self.shard(n)) {
            Some(s) => s.shard.add_refresh_listener(self.max_refresh_listeners(), WAIT_FOR_REFRESH_MAX, done),
            None => self.add_refresh_listener(done),
        }
    }

    /// [`IndexService::wait_for_refresh`] without waiting: `done` is called
    /// once every local shard's listener completed (forced if any was).
    pub fn add_refresh_listener(&self, done: crate::shard::RefreshCallback) {
        let shards = self.local_shards();
        if shards.is_empty() {
            done(Ok(false));
            return;
        }
        struct Join { remaining: usize, forced: bool, error: Option<EsError>, done: Option<crate::shard::RefreshCallback> }
        let join = std::sync::Arc::new(parking_lot::Mutex::new(Join { remaining: shards.len(), forced: false, error: None, done: Some(done) }));
        for s in shards {
            let join = join.clone();
            s.shard.add_refresh_listener(self.max_refresh_listeners(), WAIT_FOR_REFRESH_MAX, Box::new(move |result| {
                let finished = {
                    let mut j = join.lock();
                    match result {
                        Ok(forced) => j.forced |= forced,
                        Err(error) => { j.error.get_or_insert(error); }
                    }
                    j.remaining -= 1;
                    if j.remaining == 0 { j.done.take().map(|done| (done, j.forced, j.error.take())) } else { None }
                };
                if let Some((done, forced, error)) = finished {
                    done(match error { Some(error) => Err(error), None => Ok(forced) });
                }
            }));
        }
    }

    /// ES `index.max_refresh_listeners` (default 1000).
    fn max_refresh_listeners(&self) -> usize {
        self.setting("index.max_refresh_listeners").and_then(|v| v.parse().ok()).unwrap_or(1000)
    }

    /// Refresh every local shard.
    pub fn refresh(&self) -> Result<()> {
        for s in self.local_shards() {
            s.shard.refresh()?;
        }
        Ok(())
    }

    /// Flush (commit) every local shard.
    pub fn flush(&self) -> Result<()> {
        for s in self.local_shards() {
            s.shard.flush()?;
        }
        Ok(())
    }

    /// Force merge every local shard.
    pub fn force_merge(&self, max_segments: usize, only_expunge_deletes: bool) -> Result<()> {
        self.force_merge_with(max_segments, only_expunge_deletes, true)
    }

    /// Force merge every local shard; `flush` as ES `_forcemerge?flush`.
    pub fn force_merge_with(&self, max_segments: usize, only_expunge_deletes: bool, flush: bool) -> Result<()> {
        for s in self.local_shards() {
            s.shard.force_merge_with(max_segments, only_expunge_deletes, flush)?;
        }
        Ok(())
    }

    /// Make every local shard's translog appends durable.
    pub fn sync_translog(&self) -> Result<()> {
        for s in self.local_shards() {
            s.shard.sync_translog()?;
        }
        Ok(())
    }

    /// Store size over the local shards.
    pub fn store_size(&self) -> u64 {
        self.local_shards().iter().map(|s| s.shard.store_size()).sum()
    }

    /// Translog bytes over the local shards (stats: see [`crate::shard::Shard::write_stats`]).
    pub fn translog_size(&self) -> u64 {
        self.local_shards().iter().map(|s| s.shard.write_stats().translog_bytes).sum()
    }

    /// Uncommitted ops over the local shards (stats, as [`Self::translog_size`]).
    pub fn uncommitted_ops(&self) -> i64 {
        self.local_shards().iter().map(|s| s.shard.write_stats().uncommitted_ops).sum()
    }

    pub fn num_segments(&self) -> usize {
        self.local_shards()
            .iter()
            // Closed memtables count as the segments they seal into (stats
            // never seal), from the published view (no core lock).
            .map(|s| s.shard.reader_view().segments_for_stats())
            .sum()
    }

    /// Operation counters summed over the local shards.
    pub fn counters(&self) -> crate::shard::Counters {
        let shards = self.local_shards();
        let c = crate::shard::Counters::sum(shards.iter().map(|s| &s.shard.counters));
        // `merges.current` counts merges the policy wants but has not
        // started too: a wait for it to reach 0 (Rally's
        // wait-until-merges-finish) then really waits for the merges.
        let queued: u64 = shards.iter().map(|s| s.shard.queued_merges()).sum();
        c.merge_current.fetch_add(queued, std::sync::atomic::Ordering::Relaxed);
        c
    }
}

/// The routing hash a time series `_id` starts with (4 bytes, LE).
fn tsdb_routing_hash(id: &str) -> Option<i32> {
    crate::tsdb::routing_hash_of_id(id)
}

/// Recursive merge of a partial doc into a source (objects merge, the rest replaces).
pub fn merge_json(dst: &mut Value, src: &Value) {
    match (dst, src) {
        (Value::Object(d), Value::Object(s)) => {
            for (k, v) in s {
                match d.get_mut(k) {
                    Some(dv) if dv.is_object() && v.is_object() => merge_json(dv, v),
                    _ => {
                        d.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (d, s) => *d = s.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn svc(dir: &Path) -> IndexService {
        IndexService::create(dir, "test", settings::normalize_index_settings(&json!({})), &json!({"properties": {"n": {"type": "long"}}}), BTreeMap::new()).unwrap()
    }

    #[test]
    fn archived_index_settings_can_be_cleared_on_an_open_index() {
        let tmp = tempfile::tempdir().unwrap();
        let s = svc(tmp.path());
        {
            let mut meta = s.meta.write();
            meta.settings.insert("archived.index.max_regex_length".into(), json!("bad"));
            meta.settings.insert("archived.index.other".into(), json!("old"));
        }
        let before = s.meta().settings;
        assert!(s.put_settings(settings::normalize_index_settings(&json!({"archived.index.other": "new"})), false).is_err());
        assert_eq!(s.meta().settings, before, "rejected archival writes cannot change metadata");
        s.put_settings(settings::normalize_index_settings(&json!({"archived.index.max_regex_length": null})), false).unwrap();
        assert!(!s.meta().settings.contains_key("archived.index.max_regex_length"));
        assert!(s.meta().settings.contains_key("archived.index.other"));
        s.put_settings(settings::normalize_index_settings(&json!({"archived.*": null})), false).unwrap();
        assert!(!s.meta().settings.keys().any(|key| key.starts_with("archived.")));
        assert!(s.is_open());
    }

    #[test]
    fn resetting_one_shard_keeps_the_index_and_its_mapping() {
        let tmp = tempfile::tempdir().unwrap();
        let s = svc(tmp.path());
        s.write(vec![WriteReq::Index { id: Some("1".into()), source: r#"{"n": 1, "v": "x"}"#.into(), routing: None, cond: WriteCond::default() }], true);
        assert!(s.lookup().get("v").is_some());
        s.remove_shard(0, true).unwrap();
        assert!(!tmp.path().join("shard-0").exists());
        // The mapping (incl. the dynamic field) survives; only the data went.
        assert!(s.lookup().get("v").is_some());
        assert!(s.shard(0).is_none());
    }

    #[test]
    fn quarantine_cancels_inflight_open_tokens() {
        let tmp = tempfile::tempdir().unwrap();
        let s = svc(tmp.path());
        let token = Arc::new(AtomicBool::new(false));
        s.openings.lock().push((0, token.clone()));
        s.quarantine_mapping();
        assert!(token.load(Ordering::Acquire));
        assert_eq!(s.attached_or(Ok(())).unwrap_err().kind, COPY_REPLACED);
        s.detach();
    }

    #[test]
    fn replaced_copies_refuse_metadata_operations() {
        let tmp = tempfile::tempdir().unwrap();
        let s = svc(tmp.path());
        s.detach();
        // A failure of an operation that was running when the copy was
        // replaced (its files deleted under it) is reported as replaced too.
        let io = s.attached_or::<()>(Err(EsError::internal("IO error: No such file or directory (os error 2)"))).unwrap_err();
        assert_eq!((io.status, io.kind.as_str()), (503, COPY_REPLACED));
        for e in [s.set_open(false).unwrap_err(), s.put_mapping(&json!({"properties": {"x": {"type": "long"}}})).unwrap_err(),
                  s.put_settings(settings::normalize_index_settings(&json!({"index": {"refresh_interval": "2s"}})), false).unwrap_err()] {
            assert_eq!((e.status, e.kind.as_str()), (503, COPY_REPLACED));
        }
    }

    /// A rejected settings update (BM25 `b` out of range) changes nothing:
    /// the index stays open with its settings, and reopens from disk.
    #[test]
    fn rejected_settings_update_leaves_the_index_as_it_was() {
        let tmp = tempfile::tempdir().unwrap();
        let flat = |v: Value| settings::normalize_index_settings(&v);
        let s = IndexService::create(tmp.path(), "e", flat(json!({"index": {"similarity": {"my": {"type": "BM25", "b": 0.5}}}})), &json!({}), BTreeMap::new()).unwrap();
        let before = s.meta.read().settings.clone();
        let bad = flat(json!({"index": {"similarity": {"my": {"type": "BM25", "b": 2}}}}));
        // Through a close / reopen, and on a closed index.
        let e = s.put_settings_reopen(bad.clone(), false, true).unwrap_err();
        assert_eq!(e.reason, "illegal b value: 2.0, must be between 0 and 1");
        assert!(s.is_open());
        assert_eq!(s.meta.read().settings, before);
        s.set_open(false).unwrap();
        assert!(s.put_settings_reopen(bad, false, false).is_err());
        assert_eq!(s.meta.read().settings, before);
        s.set_open(true).unwrap();
        // Other updates still apply; the index reopens from disk as it was.
        s.put_settings(flat(json!({"index": {"refresh_interval": "2s"}})), false).unwrap();
        drop(s);
        let s = IndexService::open(tmp.path()).unwrap();
        assert_eq!(s.setting("index.similarity.my.b").as_deref(), Some("0.5"));
        assert!(s.is_open());
    }

    /// An index an older build created with BM25 parameters out of
    /// Lucene's ranges still opens (its scorers are never pruned), and
    /// takes updates that leave its similarities alone.
    #[test]
    fn existing_out_of_range_similarity_still_opens() {
        let tmp = tempfile::tempdir().unwrap();
        let flat = |v: Value| settings::normalize_index_settings(&v);
        let s = IndexService::create(tmp.path(), "e", flat(json!({"index": {"similarity": {"my": {"type": "BM25", "b": 2, "k1": -1}}}})), &json!({}), BTreeMap::new()).unwrap();
        s.put_settings(flat(json!({"index": {"refresh_interval": "2s"}})), false).unwrap();
        drop(s);
        let s = IndexService::open(tmp.path()).unwrap();
        assert_eq!(s.sims().resolve(Some("my")).sim.as_bm25().unwrap().b, 2.0);
        // New indices are held to the ranges.
        let dir = tempfile::tempdir().unwrap();
        let node = crate::node::Node::open(dir.path()).unwrap();
        let e = node.create_index("n", &json!({"settings": {"index": {"similarity": {"my": {"type": "BM25", "k1": -1}}}}})).unwrap_err();
        assert_eq!(e.reason, "illegal k1 value: -1.0, must be a non-negative finite value");
    }

    #[test]
    fn auto_expand_rederives_replicas_only_when_it_changes() {
        let tmp = tempfile::tempdir().unwrap();
        let flat = |v: Value| settings::normalize_index_settings(&v);
        let s = IndexService::create(tmp.path(), "e", flat(json!({"index": {"number_of_replicas": 0, "auto_expand_replicas": "0-all"}})), &json!({}), BTreeMap::new()).unwrap();
        assert_eq!(s.setting("index.number_of_replicas").as_deref(), Some("0"));
        // A cluster master publishes the count it derived from the cluster's nodes.
        s.put_settings(flat(json!({"index": {"number_of_replicas": 1}})), false).unwrap();
        assert_eq!(s.setting("index.number_of_replicas").as_deref(), Some("1"));
        // Unrelated updates (a write block) keep it.
        s.put_settings(flat(json!({"index": {"blocks": {"write": true}}})), false).unwrap();
        assert_eq!(s.setting("index.number_of_replicas").as_deref(), Some("1"));
        // Changing auto_expand_replicas re-derives it (single engine: the minimum).
        s.put_settings(flat(json!({"index": {"auto_expand_replicas": "1-all"}})), false).unwrap();
        assert_eq!(s.setting("index.number_of_replicas").as_deref(), Some("1"));
        s.put_settings(flat(json!({"index": {"auto_expand_replicas": "0-all"}})), false).unwrap();
        assert_eq!(s.setting("index.number_of_replicas").as_deref(), Some("0"));
    }

    #[test]
    fn write_get_update_reopen() {
        let tmp = tempfile::tempdir().unwrap();
        {
            let s = svc(tmp.path());
            let r = s.write(vec![WriteReq::Index { id: Some("1".into()), source: r#"{"n": 1, "t": "hello"}"#.into(), routing: None, cond: WriteCond::default() }], true);
            assert_eq!(r[0].as_ref().unwrap().result, "created");
            assert!(s.lookup().get("t.keyword").is_some(), "dynamic mapping applied");
            let bad = s.write(vec![WriteReq::Index { id: Some("2".into()), source: r#"{"n": "x"}"#.into(), routing: None, cond: WriteCond::default() }], true);
            assert_eq!(bad[0].as_ref().unwrap_err().kind, "document_parsing_exception");
            let (w, src) = s.update("1", &json!({"doc": {"n": 2}}), None, 0, WriteCond::default()).unwrap();
            assert_eq!(w.result, "updated");
            assert_eq!(serde_json::from_str::<Value>(&src.unwrap()).unwrap()["n"], 2);
            let (w, _) = s.update("1", &json!({"doc": {"n": 2}}), None, 0, WriteCond::default()).unwrap();
            assert_eq!(w.result, "noop");
            let (w, _) = s.update("9", &json!({"doc": {"n": 9}, "doc_as_upsert": true}), None, 0, WriteCond::default()).unwrap();
            assert_eq!(w.result, "created");
            let e = s.update("8", &json!({"doc": {"n": 9}}), None, 0, WriteCond::default()).unwrap_err();
            assert_eq!(e.kind, "document_missing_exception");
            let r = s.write(vec![WriteReq::Index { id: None, source: "{}".into(), routing: None, cond: WriteCond::default() }], true);
            assert_eq!(r[0].as_ref().unwrap().id.len(), 20);
        }
        let s = IndexService::open(tmp.path()).unwrap();
        let g = s.get("1").unwrap().unwrap();
        assert_eq!(g.version, 2);
        assert!(g.source.unwrap().contains("\"n\":2"));
        assert!(s.lookup().get("t").is_some());
        assert!(s.get("9").unwrap().is_some());
    }

    #[test]
    fn a_reopened_copy_has_refreshed_its_commit_and_translog() {
        // A fleet wait for a checkpoint the copy had before it opened
        // returns at once: the open's refresh made it visible.
        let tmp = tempfile::tempdir().unwrap();
        {
            let s = svc(tmp.path());
            let index = |id: &str| WriteReq::Index { id: Some(id.into()), source: r#"{"n": 1}"#.into(), routing: None, cond: WriteCond::default() };
            assert!(s.write(vec![index("committed")], false)[0].is_ok());
            s.flush().unwrap();
            assert!(s.write(vec![index("replayed")], false)[0].is_ok());
        }
        let s = IndexService::open(tmp.path()).unwrap();
        let copy = s.shard(0).unwrap();
        assert_eq!(copy.shard.local_checkpoint(), 1);
        assert_eq!(copy.shard.refreshed_checkpoint(), 1, "commit plus translog replay");
        assert!(crate::fleet::caught_up(&s, &copy, 0));
        assert!(crate::fleet::caught_up(&s, &copy, 1));
        assert!(!crate::fleet::caught_up(&s, &copy, 2));
    }

    #[test]
    fn settings_rules() {
        let tmp = tempfile::tempdir().unwrap();
        let s = svc(tmp.path());
        s.put_settings(settings::normalize_index_settings(&json!({"refresh_interval": "5s"})), false).unwrap();
        assert_eq!(s.setting("index.refresh_interval").as_deref(), Some("5s"));
        let e = s.put_settings(settings::normalize_index_settings(&json!({"number_of_shards": 2})), false).unwrap_err();
        assert!(e.reason.starts_with("Can't update non dynamic settings [[index.number_of_shards]] for open indices [[test/"));
        let e = s.put_mapping(&json!({"properties": {"n": {"type": "keyword"}}})).unwrap_err();
        assert_eq!(e.reason, "mapper [n] cannot be changed from type [long] to [keyword]");
    }

    /// `_primary_term` is the document's own term in write / get results,
    /// and `_seq_no` / `_primary_term` can be sorted and aggregated on.
    #[test]
    fn per_doc_primary_term_sort_and_aggs() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Arc::new(svc(tmp.path()));
        let idx = |id: &str, n: i64| WriteReq::Index { id: Some(id.into()), source: format!(r#"{{"n": {n}}}"#), routing: None, cond: WriteCond::default() };
        // Term 1: two docs.
        assert_eq!(s.shard(0).unwrap().shard.primary_term(), 1);
        let r = s.write(vec![idx("a", 1), idx("b", 2)], true);
        for w in &r {
            assert_eq!(w.as_ref().unwrap().primary_term, s.shard(0).unwrap().shard.primary_term());
        }
        assert_eq!(r[0].as_ref().unwrap().primary_term, 1);
        // Term 2: one new doc + one overwrite.
        s.shard(0).unwrap().shard.set_primary_term(2).unwrap();
        let r = s.write(vec![idx("c", 3), idx("b", 4)], true);
        assert_eq!(r[0].as_ref().unwrap().primary_term, 2);
        assert_eq!((r[1].as_ref().unwrap().result, r[1].as_ref().unwrap().primary_term), ("updated", 2));
        // GET returns each document's own term, not the shard's.
        assert_eq!(s.get("a").unwrap().unwrap().primary_term, 1);
        assert_eq!(s.get("b").unwrap().unwrap().primary_term, 2);
        assert_eq!(s.get("c").unwrap().unwrap().primary_term, 2);
        // A noop update carries the document's term.
        let (w, _) = s.update("a", &json!({"doc": {"n": 1}}), None, 0, WriteCond::default()).unwrap();
        assert_eq!((w.result, w.primary_term), ("noop", 1));
        // Delete carries the shard's term.
        let r = s.write(vec![WriteReq::Delete { id: "zz".into(), routing: None, cond: WriteCond::default() }], true);
        assert_eq!((r[0].as_ref().unwrap().result, r[0].as_ref().unwrap().primary_term), ("not_found", 2));

        s.refresh().unwrap();
        let search = |body: Value| -> Value {
            let req = crate::search::parse_request(&body, &std::collections::HashMap::new()).unwrap();
            let targets = vec![crate::search::Target::new(s.clone())];
            let out = crate::search::execute(&targets, &req, &crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65536, scroll_id: None, pit_id: None }).unwrap();
            serde_json::from_str(&out.json).unwrap()
        };
        // Sort by _seq_no (desc): b (seq 3), c (seq 2), a (seq 0).
        let v = search(json!({"sort": [{"_seq_no": "desc"}]}));
        let ids: Vec<&str> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["b", "c", "a"]);
        assert_eq!(v["hits"]["hits"][0]["sort"], json!([3]));
        // Sort by _primary_term then _seq_no.
        let v = search(json!({"sort": [{"_primary_term": "asc"}, {"_seq_no": "asc"}]}));
        let ids: Vec<&str> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["a", "c", "b"]);
        assert_eq!(v["hits"]["hits"][2]["sort"], json!([2, 3]));
        // terms agg on _primary_term: term 1 -> 1 doc (a), term 2 -> 2 docs (b, c).
        let v = search(json!({"size": 0, "aggs": {"t": {"terms": {"field": "_primary_term", "order": {"_key": "asc"}}}, "s": {"terms": {"field": "_seq_no", "order": {"_key": "asc"}}}}}));
        assert_eq!(v["aggregations"]["t"]["buckets"], json!([{"key": 1, "doc_count": 1}, {"key": 2, "doc_count": 2}]));
        assert_eq!(v["aggregations"]["s"]["buckets"], json!([{"key": 0, "doc_count": 1}, {"key": 2, "doc_count": 1}, {"key": 3, "doc_count": 1}]));
    }
}

impl std::fmt::Debug for IndexService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IndexService({}/{})", self.name, self.uuid)
    }
}

#[cfg(test)]
mod auto_id_tests {
    use base64::Engine;

    /// ES's time-based UUID layout: unique, 20 characters, the node address
    /// (bytes 6..12) the same in every id, the sequence's low byte first.
    #[test]
    fn auto_ids_follow_es_time_based_layout() {
        let ids: Vec<String> = (0..100_000).map(|_| super::auto_id()).collect();
        let unique: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len());
        let bytes: Vec<Vec<u8>> = ids.iter().map(|id| { assert_eq!(id.len(), 20); base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(id).unwrap() }).collect();
        assert!(bytes.iter().all(|b| b.len() == 15 && b[6..12] == bytes[0][6..12]));
        // Consecutive ids (one thread) differ in the sequence's low byte.
        assert!(bytes.windows(2).filter(|w| w[0][0] == w[1][0]).count() < bytes.len() / 100);
    }

    /// ES's k-ordered layout (logsdb, columnar): unique, 20 characters, the
    /// timestamp's high bytes then the node address leading, so ids made in
    /// order sort in order within a millisecond's byte and share 10 bytes.
    #[test]
    fn k_ordered_auto_ids_follow_es_layout() {
        let ids: Vec<String> = (0..100_000).map(|_| super::auto_id_k_ordered()).collect();
        let unique: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len());
        let bytes: Vec<Vec<u8>> = ids.iter().map(|id| { assert_eq!(id.len(), 20); base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(id).unwrap() }).collect();
        assert!(bytes.iter().all(|b| b.len() == 15 && b[4..10] == bytes[0][4..10]));
        // Timestamp bytes never go backwards.
        assert!(bytes.windows(2).all(|w| w[0][..4] <= w[1][..4]));
        assert!(super::k_ordered_ids(Some("logsdb")) && super::k_ordered_ids(Some("LogsDB")) && super::k_ordered_ids(Some("columnar")));
        assert!(!super::k_ordered_ids(Some("standard")) && !super::k_ordered_ids(Some("time_series")) && !super::k_ordered_ids(None));
    }
}

/// Per-document parse phases on real corpus docs:
/// `MB_DIR=~/tsdb/mb MB_CORPUS=<file> MB_DS=k8s-pod MB_DOCS=50000 cargo test --profile fast -p elasticsearch-engine --lib parse_phase_microbench -- --ignored --nocapture`
/// (`MB_DIR/<ds>.mapping.json` and `<ds>.settings.json`: the index's
/// mapping and flat settings).
#[cfg(test)]
mod parse_phase_microbench {
    use super::*;
    use std::io::BufRead;

    #[test]
    #[ignore]
    fn parse_phase_microbench() {
        let dir = std::env::var("MB_DIR").expect("MB_DIR");
        let ds = std::env::var("MB_DS").unwrap_or_else(|_| "k8s-pod".into());
        let n: usize = std::env::var("MB_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(50_000);
        let mapping: Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/{ds}.mapping.json")).unwrap()).unwrap();
        let settings: Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/{ds}.settings.json")).unwrap()).unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let node = crate::node::Node::open(tmp.path()).unwrap();
        node.create_index("p", &serde_json::json!({"settings": settings, "mappings": mapping})).unwrap();
        let svc = node.get_index("p").unwrap();
        let file = std::fs::File::open(std::env::var("MB_CORPUS").expect("MB_CORPUS")).unwrap();
        let docs: Vec<Value> = std::io::BufReader::new(file).lines().take(n).map(|l| serde_json::from_str(&l.unwrap()).unwrap()).collect();
        let limits = crate::fieldtypes::limits::Limits::from_settings(|k| svc.core.setting(k));
        // `MB_VERIFY=1`: every document's one-walk forms checked against
        // the two passes (and the reference residual), not timed.
        let verify = std::env::var("MB_VERIFY").is_ok();
        SKIP_SYNTH_CHECKS.with(|c| c.set(!verify));
        if verify {
            for (i, d) in docs.iter().enumerate() { svc.core.parse_source_mode(d, &i.to_string(), &limits, false).unwrap(); }
            println!("verified {} docs", docs.len());
            return;
        }
        // Warm (dynamic mappings settle).
        for (i, d) in docs.iter().take(2000).enumerate() { let _ = svc.core.parse_source_mode(d, &i.to_string(), &limits, false); }
        for round in 0..3 {
            PARSE_PHASES.with(|c| c.set((0, 0)));
            let t = std::time::Instant::now();
            for (i, d) in docs.iter().enumerate() {
                svc.core.parse_source_mode(d, &i.to_string(), &limits, false).unwrap();
            }
            let total = t.elapsed().as_nanos() as f64 / docs.len() as f64;
            let (synth, forms) = PARSE_PHASES.with(|c| c.get());
            let per = |x: u64| x as f64 / docs.len() as f64;
            println!("round {round}: {} docs  parse_source {:.0} ns/doc  synthesize {:.0}  store_forms {:.0}  rest {:.0}", docs.len(), total, per(synth), per(forms), total - per(synth) - per(forms));
        }
    }
}
