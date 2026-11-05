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
    cluster_slm_health: RwLock<Option<(String, usize)>>,
    pub(crate) dlm_stats: Mutex<DlmRunStats>,
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
            cluster_slm_health: RwLock::new(None),
            dlm_stats: Mutex::new(DlmRunStats::default()),
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

    /// Cluster SLM is executed by the coordinator; mirror only its health inputs.
    pub fn restore_cluster_slm_health(&self, mode: String, policies: usize) {
        *self.cluster_slm_health.write() = Some((mode, policies));
    }

    fn slm_health_state(&self) -> (String, usize) {
        if let Some(state) = self.cluster_slm_health.read().as_ref() {
            return state.clone();
        }
        let state = self.st.read();
        (state.slm_mode.clone(), state.slm.len())
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
        for (pn, p) in &phases {
            if let Some(slm_policy) = p["actions"]["wait_for_snapshot"]["policy"].as_str() {
                if !self.st.read().slm.contains_key(slm_policy) {
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
}

/// When the lifecycle executor last ran its periodic ILM and data stream
/// lifecycle polls (see [`tick`]).
pub struct TickClock {
    last_ilm_poll: Instant,
    last_dlm_poll: Instant,
}

impl Default for TickClock {
    fn default() -> Self {
        TickClock { last_ilm_poll: Instant::now(), last_dlm_poll: Instant::now() }
    }
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
