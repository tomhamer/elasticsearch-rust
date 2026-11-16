//! Index lifecycle features of a single node:
//!
//! * ILM (`policy`, `runner`): policies stored in `_state/ilm.json`, a
//!   per-index execution state stored next to the index metadata
//!   (`<index dir>/lifecycle.json`), and a background executor that walks
//!   ES's step graph (`new/init` → `new/complete` → phase actions → ...).
//! * Data stream lifecycle (`dlm`): `_data_stream/{n}/_lifecycle`,
//!   explain, stats and the background rollover / retention run, plus
//!   `_data_stream/_modify`, `_migrate`, `_promote`, `_stats`.
//! * Resize (`resize`): `_clone`, `_shrink`, `_split` (segment copy).
//! * SLM (`slm`): policies, stats and status; `_execute` and the cron
//!   schedule take real snapshots in the fs repository, retention
//!   (`expire_after` / `min_count` / `max_count`) deletes old ones.
//!
//! The executor runs on one thread per node (`start`): every second it
//! runs the "cluster state changed" pass (every step except async wait
//! steps), and every `indices.lifecycle.poll_interval` (default 10m) the
//! periodic pass that also evaluates wait steps (rollover conditions,
//! `wait_for_snapshot`). Data stream lifecycle runs every
//! `data_streams.lifecycle.poll_interval` (default 5m).

pub mod dlm;
pub mod policy;
pub mod resize;
pub mod runner;
pub mod slm;
pub mod snapshot_steps;
pub mod frozen;

use crate::error::{EsError, Result};
use crate::node::Node;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const RUNNING: &str = "RUNNING";
pub const STOPPING: &str = "STOPPING";
pub const STOPPED: &str = "STOPPED";

fn running() -> String {
    RUNNING.to_string()
}

/// A stored ILM policy (`LifecyclePolicyMetadata`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolicyMeta {
    /// Normalized `{"phases": {...}, "_meta"?: {...}, "deprecated"?: bool}`.
    pub policy: Value,
    pub version: u64,
    pub modified_date: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IlmState {
    #[serde(default)]
    pub policies: BTreeMap<String, PolicyMeta>,
    #[serde(default = "running")]
    pub mode: String,
    #[serde(default)]
    pub slm: BTreeMap<String, slm::SlmPolicyMeta>,
    #[serde(default = "running")]
    pub slm_mode: String,
    #[serde(default)]
    pub slm_stats: slm::SlmStats,
    #[serde(default)]
    pub builtins_installed: bool,
}

impl Default for IlmState {
    fn default() -> Self {
        IlmState {
            policies: BTreeMap::new(),
            mode: running(),
            slm: BTreeMap::new(),
            slm_mode: running(),
            slm_stats: slm::SlmStats::default(),
            builtins_installed: false,
        }
    }
}

/// Last data stream lifecycle run (`GET /_lifecycle/stats`).
#[derive(Default, Clone)]
pub struct DlmRunStats {
    pub last_run_start: Option<i64>,
    pub last_run_duration: Option<i64>,
    pub time_between_starts: Option<i64>,
    /// backing index name → error (rendered in explain + stats).
    pub errors: HashMap<String, String>,
    /// backing index name → error of its `frozen_after` conversion
    /// ([`frozen::run`], every lifecycle pass), shown like `errors`.
    pub frozen_errors: HashMap<String, String>,
}

pub struct IlmService {
    path: PathBuf,
    pub(crate) st: RwLock<IlmState>,
    /// Execution states by index uuid (cache of `<index dir>/lifecycle.json`).
    pub(crate) exec: Mutex<HashMap<String, runner::IndexLifecycle>>,
    /// Bumped per index uuid whenever its stored lifecycle state changes
    /// (a cheap change marker for hosts that diff engine metadata).
    pub(crate) exec_gen: Mutex<HashMap<String, u64>>,
    /// The host drives lifecycle passes ([`IlmService::drive_externally`]).
    external: AtomicBool,
    /// A cluster's own implementation of data / placement steps.
    cluster_steps: RwLock<Option<Arc<dyn ClusterLifecycleSteps>>>,
    /// Serializes executor runs (background thread vs API-triggered runs).
    pub(crate) run_lock: Mutex<()>,
    /// Serializes [`IlmService::persist`].
    persist_lock: Mutex<()>,
    started: AtomicBool,
    /// The cluster's SLM metadata (`snapshot_lifecycle` custom), replicated
    /// by the host; `None` on a standalone node (the local catalog is then
    /// authoritative).
    cluster_slm: RwLock<Option<ClusterSlm>>,
    pub(crate) dlm_stats: Mutex<DlmRunStats>,
    /// Backing indices whose `frozen_after` conversion was submitted by a
    /// transition scan and advances on every pass until it ends
    /// (`DLMFrozenTransitionExecutor`'s submitted transitions; a new master
    /// starts empty and resubmits on its first scan).
    pub(crate) frozen_submitted: Mutex<std::collections::BTreeSet<String>>,
}

impl IlmService {
    pub fn load(data: &Path) -> IlmService {
        let path = data.join("_state/ilm.json");
        let mut st: IlmState = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        let fresh = !st.builtins_installed;
        if fresh {
            let now = crate::date::now_millis();
            for (name, p) in builtin_policies() {
                st.policies.entry(name.to_string()).or_insert(PolicyMeta { policy: p, version: 1, modified_date: now });
            }
            st.builtins_installed = true;
        }
        let svc = IlmService {
            path,
            st: RwLock::new(st),
            exec: Mutex::new(HashMap::new()),
            exec_gen: Mutex::new(HashMap::new()),
            external: AtomicBool::new(false),
            cluster_steps: RwLock::new(None),
            run_lock: Mutex::new(()),
            persist_lock: Mutex::new(()),
            started: AtomicBool::new(false),
            cluster_slm: RwLock::new(None),
            dlm_stats: Mutex::new(DlmRunStats::default()),
            frozen_submitted: Mutex::new(Default::default()),
        };
        if fresh {
            let _ = svc.persist();
        }
        svc
    }

    pub fn persist(&self) -> Result<()> {
        // One writer at a time, serializing the state inside the lock: two
        // concurrent changes (policy PUTs) neither share a temp file nor let
        // an older snapshot land after a newer one.
        let _writer = self.persist_lock.lock();
        let bytes = serde_json::to_vec_pretty(&*self.st.read())?;
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        crate::hooks::write_durable(&self.path, &bytes)?;
        Ok(())
    }

    pub fn mode(&self) -> String {
        self.st.read().mode.clone()
    }

    pub fn policy(&self, name: &str) -> Option<PolicyMeta> {
        self.st.read().policies.get(name).cloned()
    }

    pub fn policy_count(&self) -> usize {
        self.st.read().policies.len()
    }

    /// Replicate the cluster's SLM metadata (the coordinator executes SLM
    /// and records each policy's last success there): ILM's
    /// `wait_for_snapshot` validation and step read it, not the local
    /// catalog.
    pub fn restore_cluster_slm(&self, slm: ClusterSlm) {
        *self.cluster_slm.write() = Some(slm);
    }

    fn slm_health_state(&self) -> (String, usize) {
        if let Some(state) = self.cluster_slm.read().as_ref() {
            return (state.mode.clone(), state.policies.len());
        }
        let state = self.st.read();
        (state.slm_mode.clone(), state.slm.len())
    }

    /// An SLM policy's last successful snapshot (`None`: no such policy;
    /// `Some(None)`: none yet), from the authoritative SLM metadata: the
    /// cluster's when the host replicates it, else the local catalog.
    pub fn slm_last_success(&self, policy: &str) -> Option<Option<SlmSuccess>> {
        if let Some(cluster) = self.cluster_slm.read().as_ref() {
            return cluster.policies.get(policy).cloned();
        }
        let st = self.st.read();
        let p = st.slm.get(policy)?;
        let repository = p.policy["repository"].as_str().unwrap_or_default().to_string();
        Some(p.last_success.as_ref().and_then(|s| {
            Some(SlmSuccess {
                repository: repository.clone(),
                snapshot_name: s["snapshot_name"].as_str().unwrap_or_default().to_string(),
                start_time: s["start_time"].as_i64().or_else(|| s["time"].as_i64())?,
            })
        }))
    }

    pub fn slm_policy_count(&self) -> usize {
        self.st.read().slm.len()
    }

    pub fn slm_mode(&self) -> String {
        self.st.read().slm_mode.clone()
    }

    pub fn st_snapshot(&self) -> IlmState {
        self.st.read().clone()
    }

    /// `PUT /_ilm/policy/{name}` after request validation: prerequisites
    /// (`TransportPutLifecycleAction.validatePrerequisites`), no-op when
    /// unchanged, version + 1 otherwise.
    pub fn put_policy(&self, node: &Node, name: &str, policy: Value) -> Result<()> {
        if let Some(existing) = self.policy(name) {
            if existing.policy == policy {
                return Ok(());
            }
        }
        let phases = policy["phases"].as_object().cloned().unwrap_or_default();
        // `PutLifecycleMetadataService`: the action needs an enterprise (or trial) license.
        let licensed = crate::license::current(node).effective_mode(crate::date::now_millis()).allows(crate::license::Mode::Enterprise);
        if !licensed && phases.values().any(|p| p["actions"].get("searchable_snapshot").is_some()) {
            return Err(EsError::illegal_argument(format!(
                "policy [{name}] defines the [searchable_snapshot] action but the current license is non-compliant for [searchable-snapshots]"
            )));
        }
        // The `searchable_snapshot` action is stored as ES does (`PUT` /
        // `GET _ilm/policy` round-trip it). ILM here does not snapshot,
        // mount and swap: an index that reaches the action stops in ERROR
        // there (`branch-check-prerequisites`), never skipping it.
        for (pn, p) in &phases {
            if let Some(slm_policy) = p["actions"]["wait_for_snapshot"]["policy"].as_str() {
                if self.slm_last_success(slm_policy).is_none() {
                    return Err(EsError::illegal_argument(format!(
                        "no such snapshot lifecycle policy [{slm_policy}], the snapshot lifecycle policy referenced by the [wait_for_snapshot] action in the [{pn}] phase must exist before it can be referenced by an ILM policy"
                    )));
                }
            }
        }
        let _ = node;
        self.replace_policy(name, policy)
    }

    /// Store a policy (version + 1, new modified date).
    pub fn replace_policy(&self, name: &str, policy: Value) -> Result<()> {
        {
            let mut st = self.st.write();
            let version = st.policies.get(name).map(|p| p.version + 1).unwrap_or(1);
            st.policies.insert(name.to_string(), PolicyMeta { policy, version, modified_date: crate::date::now_millis() });
        }
        self.persist()
    }

    pub fn delete_policy(&self, name: &str) -> Result<()> {
        if self.st.write().policies.remove(name).is_none() {
            return Err(EsError::resource_not_found(format!("Lifecycle policy not found: {name}")));
        }
        self.persist()
    }

    /// STOPPING → STOPPED as decided by the cluster (the elected master's
    /// executor found no index inside an unsafe action).
    pub fn set_stopped(&self) -> Result<()> {
        {
            let mut st = self.st.write();
            if st.mode != STOPPING {
                return Ok(());
            }
            st.mode = STOPPED.into();
        }
        self.persist()
    }

    /// `_ilm/start` → RUNNING; `_ilm/stop` → STOPPING (the executor moves to
    /// STOPPED once no index is inside an unsafe action).
    pub fn set_mode(&self, start: bool) -> Result<()> {
        {
            let mut st = self.st.write();
            st.mode = if start {
                RUNNING.into()
            } else if st.mode == STOPPED {
                STOPPED.into()
            } else {
                STOPPING.into()
            };
        }
        self.persist()
    }
}

/// `indices.lifecycle.*` / `data_streams.lifecycle.*` cluster settings
/// (transient wins over persistent), as strings.
/// Whether `svc` is in its lifecycle's `frozen` phase (ES
/// `FrozenExistenceDeciderService.isFrozenPhase`).
pub fn frozen_phase(node: &Node, svc: &crate::index::IndexService) -> bool {
    runner::load(node, svc).ilm.phase.as_deref() == Some("frozen")
}

pub fn cluster_setting(node: &Node, key: &str) -> Option<String> {
    let cs = node.cluster_settings.read();
    let get = |m: &serde_json::Map<String, Value>| -> Option<String> {
        let v = m.get(key).cloned().or_else(|| {
            // nested form `{"indices": {"lifecycle": {...}}}`
            let mut cur = Value::Object(m.clone());
            for part in key.split('.') {
                cur = cur.get(part)?.clone();
            }
            Some(cur)
        })?;
        match v {
            Value::Null => None,
            Value::String(s) => Some(s),
            other => Some(other.to_string()),
        }
    };
    get(&cs.1).or_else(|| get(&cs.0))
}

fn setting_millis(node: &Node, key: &str, default: i64) -> i64 {
    cluster_setting(node, key).and_then(|s| parse_time_millis(&s)).filter(|v| *v > 0).unwrap_or(default)
}

/// The `searchable_snapshot` ILM action (and so the `frozen` phase) is not
/// supported: refused at `PUT _ilm/policy`, and an index whose stored
/// policy has it stops in the ERROR step there instead of skipping it.
pub fn searchable_snapshot_unsupported(policy: &str, phase: &str) -> EsError {
    EsError::illegal_argument(format!(
        "policy [{policy}] defines the [searchable_snapshot] action in the [{phase}] phase, but the [searchable_snapshot] action is not supported by this cluster"
    ))
}

/// A snapshot lifecycle policy's last successful execution
/// (`SnapshotInvocationRecord`).
#[derive(Clone, Debug, PartialEq)]
pub struct SlmSuccess {
    /// The policy's repository (`WaitForSnapshotStep` looks the snapshot
    /// up there).
    pub repository: String,
    pub snapshot_name: String,
    /// When the snapshot started (`snapshot_start_timestamp`).
    pub start_time: i64,
}

/// The cluster's SLM metadata as ILM reads it: the operation mode and, per
/// policy, its last success.
#[derive(Clone, Debug, Default)]
pub struct ClusterSlm {
    pub mode: String,
    pub policies: BTreeMap<String, Option<SlmSuccess>>,
}

/// An index's lifecycle execution state and rollover info (what ES keeps in
/// the index metadata of the cluster state: `LifecycleExecutionState`,
/// `rollover_info`), as JSON; `None` when the index has none. A cluster
/// publishes it with the index's metadata so that a failed publication
/// reverts it together with the step's other effects, and a new master
/// continues from it.
pub fn index_lifecycle_state(node: &Node, svc: &crate::index::IndexService) -> Option<Value> {
    let state = serde_json::to_value(runner::load(node, svc)).ok()?;
    let none = serde_json::to_value(runner::IndexLifecycle::default()).ok()?;
    (state != none).then_some(state)
}

/// A marker that changes whenever an index's stored lifecycle state does
/// (0 = never changed in this process). O(1): no state is serialized.
pub fn index_lifecycle_generation(node: &Node, svc: &crate::index::IndexService) -> u64 {
    node.ilm.exec_gen.lock().get(&svc.uuid).copied().unwrap_or(0)
}

/// ES `LifecycleExecutionState` keys of the index metadata custom `ilm`
/// (string values) for the `ExecState` fields; the lifecycle date is
/// ES's `creation_date` key.
const ES_ILM_KEYS: &[(&str, &str)] = &[
    ("phase", "phase"), ("action", "action"), ("step", "step"), ("failed_step", "failed_step"),
    ("is_auto_retryable_error", "is_auto_retryable_error"), ("failed_step_retry_count", "failed_step_retry_count"),
    ("step_info", "step_info"), ("previous_step_info", "previous_step_info"), ("phase_definition", "phase_definition"),
    ("lifecycle_date", "creation_date"), ("phase_time", "phase_time"), ("action_time", "action_time"), ("step_time", "step_time"),
    ("snapshot_repository", "snapshot_repository"), ("snapshot_name", "snapshot_name"), ("shrink_index_name", "shrink_index_name"),
];

/// An index's lifecycle state (as [`index_lifecycle_state`] returns it) in
/// the shape ES stores with the index metadata in a snapshot: the `ilm`
/// custom (every value a string; objects as JSON text) and
/// `rollover_info`. Either may be absent.
pub fn lifecycle_snapshot_fields(state: &Value) -> (Option<Value>, Option<Value>) {
    let ilm = state.get("ilm").and_then(Value::as_object).map(|o| {
        let mut out = serde_json::Map::new();
        for (ours, es) in ES_ILM_KEYS {
            let text = match o.get(*ours) {
                None | Some(Value::Null) => continue,
                Some(Value::String(s)) => s.clone(),
                Some(other) => other.to_string(),
            };
            out.insert((*es).to_string(), Value::String(text));
        }
        out
    }).filter(|o| !o.is_empty()).map(Value::Object);
    let rollover = state.get("rollover_info").filter(|r| r.as_object().is_some_and(|o| !o.is_empty())).cloned();
    (ilm, rollover)
}

/// The lifecycle state of an index restored from a snapshot whose index
/// metadata is `metadata` (ES `IndexMetadata` JSON: the `ilm` custom,
/// `rollover_info`, `settings`, `creation_date`), as
/// [`restore_index_lifecycle_state`] takes it. ES restores both with the
/// index: its phases continue where they were, aged from the original
/// lifecycle date. A managed index the snapshot holds no execution state
/// for (its lifecycle had not started) keeps its original creation date as
/// the lifecycle date. `None`: nothing to restore.
pub fn lifecycle_from_snapshot(metadata: &Value) -> Option<Value> {
    let mut ilm = serde_json::Map::new();
    if let Some(o) = metadata.get("ilm").and_then(Value::as_object) {
        for (ours, es) in ES_ILM_KEYS {
            let Some(v) = o.get(*es).filter(|v| !v.is_null()) else { continue };
            let text = v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string());
            let value = match *ours {
                "lifecycle_date" | "phase_time" | "action_time" | "step_time" | "failed_step_retry_count" => match text.parse::<i64>() {
                    Ok(n) => json!(n),
                    Err(_) => continue,
                },
                "is_auto_retryable_error" => json!(text == "true"),
                "step_info" | "previous_step_info" | "phase_definition" => match v {
                    Value::Object(_) => v.clone(),
                    _ => match serde_json::from_str::<Value>(&text) {
                        Ok(parsed) => parsed,
                        Err(_) => continue,
                    },
                },
                _ => Value::String(text),
            };
            ilm.insert((*ours).to_string(), value);
        }
    }
    let settings = &metadata["settings"];
    let setting = |k: &str| settings.get(k).or_else(|| settings.pointer(&format!("/{}", k.replace('.', "/")))).and_then(|v| v.as_str().map(str::to_string).or_else(|| (!v.is_null()).then(|| v.to_string())));
    let managed = setting("index.lifecycle.name").is_some_and(|p| !p.is_empty());
    if managed && !ilm.contains_key("lifecycle_date") {
        let created = metadata.get("creation_date").and_then(|v| v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
            .or_else(|| setting("index.creation_date").and_then(|s| s.parse().ok()))
            .filter(|ms| *ms > 0);
        if let Some(ms) = created {
            ilm.insert("lifecycle_date".into(), json!(ms));
        }
    }
    let rollover = metadata.get("rollover_info").and_then(Value::as_object).filter(|o| !o.is_empty()).cloned();
    if ilm.is_empty() && rollover.is_none() {
        return None;
    }
    let state = json!({"ilm": ilm, "rollover_info": rollover.unwrap_or_default()});
    // Only what the execution state can hold.
    serde_json::from_value::<runner::IndexLifecycle>(state).ok().and_then(|lc| serde_json::to_value(lc).ok())
}

/// Make an index's lifecycle state the published one (`None`: no state).
pub fn restore_index_lifecycle_state(node: &Node, svc: &crate::index::IndexService, state: Option<&Value>) -> Result<()> {
    let want: runner::IndexLifecycle = match state {
        Some(v) => serde_json::from_value(v.clone())?,
        None => runner::IndexLifecycle::default(),
    };
    if serde_json::to_value(runner::load(node, svc))? == serde_json::to_value(&want)? {
        return Ok(());
    }
    runner::save(node, svc, &want)
}

/// Progress of a lifecycle step a cluster executes itself.
#[derive(Clone, Debug, PartialEq)]
pub enum StepProgress {
    Done,
    /// Not yet (optional `step_info`); asked again on the next pass.
    Pending(Option<Value>),
    /// The step fails with this reason (the index moves to ERROR).
    Failed(String),
}

/// A cluster's implementation of the lifecycle steps that move documents
/// or depend on where shard copies are. The engine's own forms act on the
/// local shard copies, which on a cluster's master (or on its metadata
/// copy) may hold no documents at all. Every method returns at once:
/// long work is started on the first call and reported on later ones.
pub trait ClusterLifecycleSteps: Send + Sync {
    /// A node id to collect one copy of every shard of `index` on (the
    /// shrink's `index.routing.allocation.require._id`).
    fn single_node(&self, index: &str) -> Option<String>;
    /// Every shard of `index` has a started copy on `node_id`.
    fn allocated_on(&self, index: &str, node_id: &str) -> StepProgress;
    /// Every primary of `index` is active.
    fn primaries_active(&self, index: &str) -> StepProgress;
    /// Force-merge every copy of `index` to at most `max_num_segments`.
    fn forcemerge(&self, index: &str, max_num_segments: usize) -> StepProgress;
    /// Shrink `source` into `target` (`body`: the resize request body),
    /// documents included.
    fn shrink(&self, source: &str, target: &str, body: &Value) -> StepProgress;
    /// `target` holds at least as many documents as `source` (checked
    /// before `source` is deleted).
    fn holds_all_docs(&self, source: &str, target: &str) -> StepProgress;
    /// Index ILM history documents.
    fn write_history(&self, items: Vec<Value>);
    /// ILM `check-allocation` (`migrate == false`, ES
    /// `AllocationRoutedStep`) or `check-migration` (`migrate == true`,
    /// `DataTierMigrationRoutedStep`) judged on the cluster's routing
    /// table: every shard copy active and started on a node the index's
    /// allocation filters / tier preference allow. `settings`: the index's
    /// `index.routing.allocation.*` and `index.number_of_replicas` as this
    /// lifecycle pass sees them (the preceding `allocate` / `migrate` step
    /// may have just changed them); the routing is judged only once the
    /// cluster state carries them.
    fn allocation_routed(&self, index: &str, settings: &BTreeMap<String, String>, migrate: bool) -> StepProgress;
    /// Data stream lifecycle downsampling: downsample `source` (write
    /// blocked) into `target` (`body`: the `_downsample` request body),
    /// reading every source shard copy. `Done` once the target holds the
    /// downsampled documents.
    fn downsample(&self, source: &str, target: &str, body: &Value) -> StepProgress;
    /// `WaitForSnapshotStep`'s snapshot check: `snapshot` of `repository`
    /// (SLM policy `policy`'s last success) still exists, succeeded and includes
    /// `index`. `Done` once verified; otherwise `Pending` with the reason as
    /// step info (`Failed` is read the same way: the step keeps waiting).
    fn snapshot_includes(&self, policy: &str, repository: &str, snapshot: &str, index: &str) -> StepProgress;
    /// Clone `source` (write blocked) into `target` (`body`: the `_clone`
    /// request body), documents included. `Done` once `target` is the
    /// completed clone of this `source` incarnation (see
    /// [`snapshot_steps::clone_index`]).
    fn clone_index(&self, source: &str, target: &str, body: &Value) -> StepProgress;
    /// Snapshot `index` alone into `repository` as `snapshot` (`body`: the
    /// create-snapshot body) and wait for it. A snapshot already there
    /// under the name that completed with `index` is kept; one that failed
    /// or is partial is deleted and taken again; one in progress is waited
    /// for. `Done` once a complete snapshot is there (see
    /// [`snapshot_steps::snapshot_index`]).
    fn snapshot_index(&self, repository: &str, snapshot: &str, index: &str, body: &Value) -> StepProgress;
    /// Mount `snapshot` of `repository` as a searchable snapshot (`body`:
    /// the `_mount` request body; `partial`: `storage=shared_cache`), waiting
    /// for the restore. `Done` once every shard was restored.
    fn mount_snapshot(&self, repository: &str, snapshot: &str, body: &Value, partial: bool) -> StepProgress;
    /// Delete `snapshot` of `repository` (one already gone is `Done`).
    fn delete_snapshot(&self, repository: &str, snapshot: &str) -> StepProgress;
    /// ILM `unfollow` (ES `UnfollowAction`) on follower `index`: `step` is
    /// `wait-for-follow-shard-tasks` (every follow task caught up with its
    /// leader shard), `pause-follower-index`, `close-follower-index`,
    /// `unfollow-follower-index` or `open-follower-index`. A follower that
    /// is already paused or converted counts as done.
    fn unfollow_step(&self, index: &str, step: &str) -> StepProgress {
        StepProgress::Failed(format!("[{step}] of [{index}] is not supported on this node"))
    }
    /// Delete the snapshots of `repository` matching `pattern` that
    /// `select` picks from their info (see
    /// [`snapshot_steps::delete_snapshots_matching`]).
    fn delete_snapshots_matching(&self, repository: &str, pattern: &str, select: snapshot_steps::SnapshotSelect) -> StepProgress;
}

/// `WaitForSnapshotStep`: judge a `GET _snapshot/{repository}/{snapshot}`
/// listing (`snapshots`: the snapshot objects found, each with `state` and
/// `indices`) for SLM policy `policy`'s last successful snapshot. `Ok` when
/// exactly that snapshot was found, it succeeded and it includes `index`;
/// else the reason the index is not yet backed up (ES's messages).
pub fn verify_snapshot_listing(snapshots: &[Value], policy: &str, repository: &str, snapshot: &str, index: &str) -> std::result::Result<(), String> {
    let found: Vec<&Value> = snapshots.iter().filter(|s| s["snapshot"].as_str().is_none_or(|n| n == snapshot)).collect();
    if found.len() != 1 {
        return Err(format!("unexpected number of snapshots retrieved for repository '{repository}' and snapshot '{snapshot}' (expected 1, found {})", found.len()));
    }
    let snap = found[0];
    let state = snap["state"].as_str().unwrap_or_default();
    if state != "SUCCESS" {
        return Err(format!("the last successful snapshot [{snapshot}] of policy '{policy}' in repository '{repository}' is in state [{state}], not [SUCCESS]"));
    }
    let included = snap["indices"].as_array().is_some_and(|a| a.iter().any(|i| i.as_str() == Some(index)));
    if !included {
        return Err(format!("the last successful snapshot of policy '{policy}' does not include index '{index}'"));
    }
    Ok(())
}

/// A setting value as a string (lists comma-joined, spaces around the
/// commas dropped): how lifecycle steps compare an engine's settings with
/// the published ones.
pub fn setting_string(v: &Value) -> Option<String> {
    match v {
        Value::Null => None,
        Value::String(s) => Some(s.split(',').map(str::trim).collect::<Vec<_>>().join(",")),
        Value::Array(a) => Some(a.iter().filter_map(setting_string).collect::<Vec<_>>().join(",")),
        other => Some(other.to_string()),
    }
}

/// When the lifecycle executor last ran its periodic ILM and data stream
/// lifecycle polls (see [`tick`]).
pub struct TickClock {
    last_ilm_poll: Instant,
    last_dlm_poll: Instant,
    /// The `frozen_after` transition and cleanup scans
    /// (`dlm.frozen.transition.poll_interval`, `dlm.frozen.cleanup.poll_interval`).
    last_frozen_poll: Instant,
    last_frozen_cleanup: Instant,
    frozen_cleanup_ran: bool,
}

impl Default for TickClock {
    fn default() -> Self {
        let now = Instant::now();
        TickClock { last_ilm_poll: now, last_dlm_poll: now, last_frozen_poll: now, last_frozen_cleanup: now, frozen_cleanup_ran: false }
    }
}

/// The frozen transition service's scans that are due on this tick (ES
/// schedules them on the master with these node settings: the transition
/// scan every `dlm.frozen.transition.poll_interval` (5m, first after one
/// interval), the orphan cleanup every `dlm.frozen.cleanup.poll_interval`
/// (1d, first after min(5m, interval))); submitted conversions advance on
/// every tick in between.
fn frozen_tick(node: &Node, clock: &mut TickClock) {
    let transition_every = frozen::node_time_setting(node, "dlm.frozen.transition.poll_interval", 300_000) as u128;
    let transition_due = clock.last_frozen_poll.elapsed().as_millis() >= transition_every;
    if transition_due {
        clock.last_frozen_poll = Instant::now();
    }
    let cleanup_every = frozen::node_time_setting(node, "dlm.frozen.cleanup.poll_interval", 86_400_000) as u128;
    // First run after min(5m, interval), then every interval.
    let wait = if clock.frozen_cleanup_ran { cleanup_every } else { cleanup_every.min(300_000) };
    let cleanup_due = clock.last_frozen_cleanup.elapsed().as_millis() >= wait;
    if cleanup_due {
        clock.last_frozen_cleanup = Instant::now();
        clock.frozen_cleanup_ran = true;
    }
    frozen::tick(node, transition_due, cleanup_due);
}

/// One iteration of the lifecycle executor (the `es-lifecycle` thread runs
/// it every second): the ILM pass (periodic per
/// `indices.lifecycle.poll_interval`), data stream lifecycle, SLM. A host
/// that drives lifecycle itself ([`IlmService::drive_externally`]) calls this.
pub fn tick(node: &Node, clock: &mut TickClock) {
    let ilm_every = setting_millis(node, "indices.lifecycle.poll_interval", 600_000) as u128;
    let periodic = clock.last_ilm_poll.elapsed().as_millis() >= ilm_every;
    if periodic {
        clock.last_ilm_poll = Instant::now();
    }
    runner::run_all(node, periodic);
    let dlm_every = setting_millis(node, "data_streams.lifecycle.poll_interval", 300_000) as u128;
    if clock.last_dlm_poll.elapsed().as_millis() >= dlm_every {
        clock.last_dlm_poll = Instant::now();
        dlm::run(node);
    }
    frozen_tick(node, clock);
    slm::tick(node);
}

impl TickClock {
    /// Whether the next [`tick`] / [`tick_lifecycle`] runs a periodic ILM
    /// poll or a data stream lifecycle pass (rollover conditions are judged
    /// only then).
    pub fn polls_due(&self, node: &Node) -> bool {
        let ilm_every = setting_millis(node, "indices.lifecycle.poll_interval", 600_000) as u128;
        let dlm_every = setting_millis(node, "data_streams.lifecycle.poll_interval", 300_000) as u128;
        self.last_ilm_poll.elapsed().as_millis() >= ilm_every || self.last_dlm_poll.elapsed().as_millis() >= dlm_every
    }
}

/// [`tick`] without SLM: ILM and data stream lifecycle only (a cluster
/// runs snapshot lifecycle itself).
pub fn tick_lifecycle(node: &Node, clock: &mut TickClock) {
    let ilm_every = setting_millis(node, "indices.lifecycle.poll_interval", 600_000) as u128;
    let periodic = clock.last_ilm_poll.elapsed().as_millis() >= ilm_every;
    if periodic {
        clock.last_ilm_poll = Instant::now();
    }
    runner::run_all(node, periodic);
    let dlm_every = setting_millis(node, "data_streams.lifecycle.poll_interval", 300_000) as u128;
    if clock.last_dlm_poll.elapsed().as_millis() >= dlm_every {
        clock.last_dlm_poll = Instant::now();
        dlm::run(node);
    }
    frozen_tick(node, clock);
}

impl IlmService {
    /// The host runs lifecycle passes itself ([`tick`]): [`start`] spawns no
    /// executor thread for this node. Call before the node's router starts.
    pub fn drive_externally(&self) {
        self.started.store(true, Ordering::SeqCst);
        self.external.store(true, Ordering::SeqCst);
    }

    /// Whether the host drives lifecycle passes (read APIs then never run
    /// steps themselves).
    pub fn driven_externally(&self) -> bool {
        self.external.load(Ordering::SeqCst)
    }

    /// Install (or remove) a cluster's [`ClusterLifecycleSteps`].
    pub fn set_cluster_steps(&self, steps: Option<Arc<dyn ClusterLifecycleSteps>>) {
        *self.cluster_steps.write() = steps;
    }

    pub(crate) fn cluster_steps(&self) -> Option<Arc<dyn ClusterLifecycleSteps>> {
        self.cluster_steps.read().clone()
    }

    /// Hold off lifecycle passes (they serialize on the same lock) for as
    /// long as the guard lives; `None` if a pass did not finish within
    /// `timeout`.
    pub fn pause_for(&self, timeout: Duration) -> Option<parking_lot::MutexGuard<'_, ()>> {
        self.run_lock.try_lock_for(timeout)
    }
}

/// Start the lifecycle executor thread for `node` (idempotent).
pub fn start(node: &Arc<Node>) {
    if node.ilm.started.swap(true, Ordering::SeqCst) {
        return;
    }
    let weak = Arc::downgrade(node);
    let spawned = std::thread::Builder::new()
        .name("es-lifecycle".into())
        .spawn(move || {
            let mut clock = TickClock::default();
            loop {
                std::thread::sleep(Duration::from_millis(1000));
                let node = match weak.upgrade() {
                    Some(n) => n,
                    None => break,
                };
                tick(&node, &mut clock);
            }
        });
    if spawned.is_err() {
        node.ilm.started.store(false, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// TimeValue / ByteSizeValue (ES string representations)
// ---------------------------------------------------------------------------

/// A parsed ES `TimeValue`: millis plus the original string representation
/// (`TimeValue.getStringRep()`: `300s` stays `300s`).
#[derive(Clone, Debug, PartialEq)]
pub struct TimeVal {
    pub nanos: i128,
    pub rep: String,
}

impl TimeVal {
    pub fn millis(&self) -> i64 {
        (self.nanos / 1_000_000) as i64
    }
    pub fn zero() -> TimeVal {
        TimeVal { nanos: 0, rep: "0ms".into() }
    }
}

/// `TimeValue.parseTimeValue(s, setting)`.
pub fn parse_time(s: &str, setting: &str) -> Result<TimeVal> {
    let bad = || {
        EsError::new(400, "parse_exception", format!("failed to parse setting [{setting}] with value [{s}] as a time value: unit is missing or unrecognized"))
    };
    let lower = s.trim().to_ascii_lowercase();
    if lower == "-1" {
        return Ok(TimeVal { nanos: -1, rep: "-1".into() });
    }
    if lower == "0" {
        return Ok(TimeVal::zero());
    }
    let units: [(&str, i128); 7] = [
        ("nanos", 1),
        ("micros", 1_000),
        ("ms", 1_000_000),
        ("s", 1_000_000_000),
        ("m", 60_000_000_000),
        ("h", 3_600_000_000_000),
        ("d", 86_400_000_000_000),
    ];
    for (u, mult) in units {
        if let Some(n) = lower.strip_suffix(u) {
            // `ms` must not be read as minutes+`s`, `nanos`/`micros` before `s`.
            if u == "s" && (n.ends_with('m') || n.ends_with("nano") || n.ends_with("micro")) {
                continue;
            }
            let n = n.trim();
            return match n.parse::<i64>() {
                Ok(v) => Ok(TimeVal { nanos: v as i128 * mult, rep: format!("{v}{u}") }),
                Err(_) => {
                    if n.parse::<f64>().is_ok() {
                        Err(EsError::new(400, "parse_exception", format!("failed to parse [{s}], fractional time values are not supported")))
                    } else {
                        Err(EsError::new(400, "parse_exception", format!("failed to parse [{s}]")))
                    }
                }
            };
        }
    }
    Err(bad())
}

pub fn parse_time_millis(s: &str) -> Option<i64> {
    parse_time(s, "").ok().map(|t| t.millis())
}

/// `TimeValue.toHumanReadableString(2)`.
pub fn human_time(millis: i64) -> String {
    if millis < 0 {
        return millis.to_string();
    }
    let nanos = millis as f64 * 1e6;
    if millis == 0 {
        return "0s".into();
    }
    let (v, suffix) = if nanos >= 86_400e9 {
        (nanos / 86_400e9, "d")
    } else if nanos >= 3_600e9 {
        (nanos / 3_600e9, "h")
    } else if nanos >= 60e9 {
        (nanos / 60e9, "m")
    } else if nanos >= 1e9 {
        (nanos / 1e9, "s")
    } else {
        (nanos / 1e6, "ms")
    };
    format!("{}{suffix}", format_decimal(v))
}

fn format_decimal(v: f64) -> String {
    if v.fract() == 0.0 {
        return format!("{}", v as i64);
    }
    // ES `formatDecimal`: a leading zero fraction digit truncates.
    let s = format!("{v}");
    if let Some(dot) = s.find('.') {
        if s.as_bytes().get(dot + 1) == Some(&b'0') {
            return s[..dot].to_string();
        }
    }
    let r = format!("{:.2}", (v * 100.0).round() / 100.0);
    let r = r.trim_end_matches('0').trim_end_matches('.').to_string();
    r
}

/// `ByteSizeValue.parseBytesSizeValue` → (bytes, normalized string rep).
pub fn parse_size(s: &str, setting: &str) -> Result<(u64, String)> {
    let lower = s.trim().to_ascii_lowercase();
    let bad = || {
        EsError::new(400, "parse_exception", format!("failed to parse setting [{setting}] with value [{s}] as a size in bytes: unit is missing or unrecognized"))
    };
    if lower == "0" || lower == "-1" {
        return Ok((0, lower));
    }
    let units: [(&str, u64); 6] = [("pb", 1 << 50), ("tb", 1 << 40), ("gb", 1 << 30), ("mb", 1 << 20), ("kb", 1 << 10), ("b", 1)];
    for (u, m) in units {
        if let Some(n) = lower.strip_suffix(u) {
            let n = n.trim();
            let v: f64 = n.parse().map_err(|_| bad())?;
            if v < 0.0 {
                return Err(bad());
            }
            return Ok(((v * m as f64) as u64, format!("{n}{u}")));
        }
    }
    Err(bad())
}

/// ISO-8601 UTC rendering of epoch millis (`2024-01-01T00:00:00.000Z`).
pub fn iso_date(millis: i64) -> String {
    crate::date::DateFormatter::new("strict_date_optional_time")
        .map(|f| f.format(millis, crate::date::Tz::UTC))
        .unwrap_or_default()
}

/// Exception → `step_info` JSON (`ElasticsearchException.generateThrowableXContent`).
pub fn error_json(e: &EsError) -> Value {
    let mut m = serde_json::Map::new();
    m.insert("type".into(), json!(e.kind));
    m.insert("reason".into(), json!(e.reason));
    for (k, v) in &e.extra {
        m.insert(k.clone(), v.clone());
    }
    if let Some(c) = &e.caused_by {
        m.insert("caused_by".into(), error_json(c));
    }
    Value::Object(m)
}

/// A policy body's `_meta.version` (ES `IndexTemplateRegistry`
/// `isUpgradeRequired`: the profiling registry's versioned policy).
fn meta_version(policy: &Value) -> Option<i64> {
    policy["_meta"]["version"].as_i64()
}

#[cfg(test)]
thread_local! {
    /// Test seam: extra built-ins (a versioned one, as profiling's).
    static TEST_EXTRA_BUILTINS: std::cell::RefCell<Vec<(&'static str, Value)>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl IlmService {
    /// ES `IndexTemplateRegistry.addIndexLifecyclePoliciesIfMissing`, on
    /// every reconcile (not only on a fresh data directory, as before): a
    /// built-in policy missing from the cluster is installed (a policy a
    /// later release added reaches existing clusters), and one whose
    /// `_meta.version` is older than the built-in's is upgraded. A present
    /// policy without a newer version is left alone. Returns whether
    /// anything changed.
    pub fn ensure_builtins(&self) -> bool {
        #[allow(unused_mut)]
        let mut wanted = builtin_policies();
        #[cfg(test)]
        TEST_EXTRA_BUILTINS.with(|e| wanted.extend(e.borrow().iter().cloned()));
        let mut changed = false;
        {
            let mut st = self.st.write();
            let now = crate::date::now_millis();
            for (name, body) in wanted {
                match st.policies.get_mut(name) {
                    None => {
                        st.policies.insert(name.to_string(), PolicyMeta { policy: body, version: 1, modified_date: now });
                        changed = true;
                    }
                    Some(cur) if meta_version(&body).is_some_and(|want| meta_version(&cur.policy).is_none_or(|have| want > have)) => {
                        cur.policy = body;
                        cur.version += 1;
                        cur.modified_date = now;
                        changed = true;
                    }
                    Some(_) => {}
                }
            }
            st.builtins_installed = true;
        }
        if changed {
            if let Err(e) = self.persist() {
                tracing::warn!("persisting built-in lifecycle policies failed: {e}");
            }
        }
        changed
    }
}

/// The policies ES 8.19 installs on a fresh cluster (stack + x-pack
/// template registries), dumped verbatim from a reference node.
fn builtin_policies() -> Vec<(&'static str, Value)> {
    static RAW: &str = include_str!("builtin_policies.json");
    let m: serde_json::Map<String, Value> = serde_json::from_str(RAW).unwrap_or_default();
    m.into_iter().map(|(k, v)| (&*Box::leak(k.into_boxed_str()), v)).collect()
}

/// `_health_report` indicators `ilm` / `slm`.
pub fn health_indicator(node: &Node, name: &str) -> Value {
    match name {
        "ilm" => {
            let mode = node.ilm.mode();
            let policies = node.ilm.policy_count();
            let stagnating = runner::stagnating_indices(node);
            if mode == RUNNING || policies == 0 {
                let symptom = if mode == RUNNING { "Index Lifecycle Management is running" } else { "No Index Lifecycle Management policies configured" };
                json!({"status": "green", "symptom": symptom, "details": {"policies": policies, "stagnating_indices": stagnating, "ilm_status": mode}})
            } else {
                json!({"status": "yellow", "symptom": "Index Lifecycle Management is not running",
                       "details": {"policies": policies, "stagnating_indices": stagnating, "ilm_status": mode},
                       "impacts": [{"id": "elasticsearch:health:ilm:impact:stale_indices", "severity": 3,
                                    "description": "Automatic index lifecycle and data retention management is disabled. The performance and stability of the cluster could be impacted.",
                                    "impact_areas": ["deployment_management"]}],
                       "diagnosis": [{"id": "elasticsearch:health:ilm:diagnosis:ilm_disabled", "cause": "Index Lifecycle Management is stopped",
                                      "action": "Start Index Lifecycle Management using [POST /_ilm/start].", "help_url": "https://ela.st/start-ilm"}]})
            }
        }
        _ => {
            let (mode, policies) = node.ilm.slm_health_state();
            if policies == 0 {
                json!({"status": "green", "symptom": "No Snapshot Lifecycle Management policies configured", "details": {"slm_status": mode, "policies": 0}})
            } else if mode == RUNNING {
                json!({"status": "green", "symptom": "Snapshot Lifecycle Management is running", "details": {"slm_status": mode, "policies": policies}})
            } else {
                json!({"status": "yellow", "symptom": "Snapshot Lifecycle Management is not running", "details": {"slm_status": mode, "policies": policies},
                       "impacts": [{"id": "elasticsearch:health:slm:impact:automation_disabled", "severity": 3,
                                    "description": "Scheduled snapshots are not running. New backup snapshots will not be created automatically.",
                                    "impact_areas": ["backup"]}],
                       "diagnosis": [{"id": "elasticsearch:health:slm:diagnosis:slm_disabled", "cause": "Snapshot Lifecycle Management is stopped",
                                      "action": "Start Snapshot Lifecycle Management using [POST /_slm/start].", "help_url": "https://ela.st/fix-slm"}]})
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Built-in policies on an existing (older) install: a missing one
    /// (`ml-size-based-ilm-policy`, added later) is installed; a versioned
    /// one older than the built-in is upgraded; an unversioned one a user
    /// changed is left alone (ES `IndexTemplateRegistry`).
    #[test]
    fn builtin_policies_upgrade_from_an_older_install() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("_state")).unwrap();
        let mine = json!({"phases": {"hot": {"min_age": "0ms", "actions": {"rollover": {"max_age": "1d"}}}}});
        let old = IlmState {
            policies: [
                ("logs".to_string(), PolicyMeta { policy: mine.clone(), version: 3, modified_date: 1 }),
                ("profiling-like".to_string(), PolicyMeta { policy: json!({"phases": {}, "_meta": {"managed": true, "version": 14}}), version: 2, modified_date: 1 }),
            ].into_iter().collect(),
            builtins_installed: true,
            ..Default::default()
        };
        std::fs::write(tmp.path().join("_state/ilm.json"), serde_json::to_vec(&old).unwrap()).unwrap();
        let newer = json!({"phases": {"hot": {"min_age": "0ms", "actions": {"rollover": {"max_age": "30d"}}}}, "_meta": {"managed": true, "version": 15}});
        TEST_EXTRA_BUILTINS.with(|e| e.borrow_mut().push(("profiling-like", newer.clone())));
        let svc = IlmService::load(tmp.path());
        assert!(svc.policy("ml-size-based-ilm-policy").is_none(), "an older install lacks it");
        assert!(svc.ensure_builtins());
        let builtins: BTreeMap<&str, Value> = builtin_policies().into_iter().collect();
        assert_eq!(svc.policy("ml-size-based-ilm-policy").map(|p| p.policy), Some(builtins["ml-size-based-ilm-policy"].clone()));
        let up = svc.policy("profiling-like").unwrap();
        assert_eq!((up.version, up.policy), (3, newer), "a newer _meta.version upgrades it");
        let logs = svc.policy("logs").unwrap();
        assert_eq!((logs.version, logs.policy), (3, mine), "no newer version: left alone");
        assert!(!svc.ensure_builtins(), "nothing left to do");
        // A deleted built-in comes back on the next reconcile (ES).
        svc.st.write().policies.remove("ml-size-based-ilm-policy");
        assert!(svc.ensure_builtins());
        TEST_EXTRA_BUILTINS.with(|e| e.borrow_mut().clear());
    }

    #[test]
    fn lifecycle_state_round_trips_through_snapshot_metadata() {
        let state = json!({
            "ilm": {"phase": "warm", "action": "complete", "step": "complete", "lifecycle_date": 1_700_000_000_000i64,
                "phase_time": 1_700_000_100_000i64, "action_time": 1_700_000_200_000i64, "step_time": 1_700_000_300_000i64,
                "phase_definition": {"policy": "p", "phase_definition": {"min_age": "0ms", "actions": {}}, "version": 3, "modified_date_in_millis": 5},
                "step_info": {"message": "m"}, "failed_step_retry_count": 2, "is_auto_retryable_error": true},
            "rollover_info": {"logs": {"met_conditions": {"max_docs": 1}, "time": 1_700_000_050_000i64}},
        });
        let (ilm, rollover) = lifecycle_snapshot_fields(&state);
        let ilm = ilm.expect("ilm custom");
        assert_eq!(ilm["creation_date"], "1700000000000");
        assert_eq!(ilm["phase"], "warm");
        assert!(ilm["phase_definition"].is_string() && ilm.as_object().unwrap().values().all(Value::is_string), "{ilm}");
        let metadata = json!({"settings": {"index.lifecycle.name": "p"}, "creation_date": 1_600_000_000_000i64, "ilm": ilm, "rollover_info": rollover});
        let back = lifecycle_from_snapshot(&metadata).expect("restored state");
        let want = serde_json::to_value(serde_json::from_value::<runner::IndexLifecycle>(state).unwrap()).unwrap();
        assert_eq!(back, want);
        // Never-started lifecycle: aged from the original creation date.
        let fresh = lifecycle_from_snapshot(&json!({"settings": {"index": {"lifecycle": {"name": "p"}}}, "creation_date": 1_600_000_000_000i64})).unwrap();
        assert_eq!(fresh["ilm"]["lifecycle_date"], 1_600_000_000_000i64);
        assert!(fresh["ilm"].get("phase").is_none(), "{fresh}");
        // Unmanaged, nothing recorded: nothing to restore.
        assert!(lifecycle_from_snapshot(&json!({"settings": {}, "creation_date": 1})).is_none());
    }

    /// Concurrent policy PUTs persist without sharing a temp file (cmb1: a
    /// rename found the shared `ilm.json.tmp` gone, "No such file or
    /// directory"), and the file ends with every policy.
    #[test]
    fn concurrent_policy_writes_all_persist() {
        let dir = tempfile::tempdir().unwrap();
        let ilm = Arc::new(IlmService::load(dir.path()));
        let writers: Vec<_> = (0..16)
            .map(|i| {
                let ilm = ilm.clone();
                std::thread::spawn(move || {
                    for j in 0..20 {
                        ilm.replace_policy(&format!("p{i}-{j}"), json!({"phases": {}})).expect("persisted");
                    }
                })
            })
            .collect();
        for w in writers {
            w.join().unwrap();
        }
        let reloaded = IlmService::load(dir.path());
        let st = reloaded.st_snapshot();
        for i in 0..16 {
            for j in 0..20 {
                assert!(st.policies.contains_key(&format!("p{i}-{j}")), "policy p{i}-{j} missing after reload");
            }
        }
    }

    #[test]
    fn time_values() {
        assert_eq!(parse_time("300s", "min_age").unwrap().rep, "300s");
        assert_eq!(parse_time("10ms", "x").unwrap().millis(), 10);
        assert_eq!(parse_time("1d", "x").unwrap().millis(), 86_400_000);
        assert_eq!(parse_time("0", "x").unwrap().rep, "0ms");
        assert_eq!(parse_time("5x", "min_age").unwrap_err().reason, "failed to parse setting [min_age] with value [5x] as a time value: unit is missing or unrecognized");
        assert_eq!(human_time(1234), "1.23s");
        assert_eq!(human_time(2050), "2s");
        assert_eq!(human_time(90_000), "1.5m");
        assert_eq!(human_time(12), "12ms");
        assert_eq!(parse_size("50GB", "x").unwrap(), (50 << 30, "50gb".into()));
    }
}
