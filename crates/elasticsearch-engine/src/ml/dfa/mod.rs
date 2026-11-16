//! Data frame analytics configs (`_ml/data_frame/analytics`): CRUD,
//! `_explain`, `_preview`, start/stop state, stats.
//!
//! ES keeps DFA configs in `.ml-config` and their runtime state in a
//! persistent task. Here both live in `<data>/_state/ml_dfa.json`.
//!
//! Starting an analytics job runs every check ES runs at start (source /
//! dest validation, field detection, data presence, empty dest index),
//! creates the dest index, marks the job `started` and launches its worker:
//! `outlier_detection` runs in process ([`outlier`]). `regression` and
//! `classification` need ml-cpp's boosted-tree trainer, which this node does
//! not have: their configs can be created, explained and previewed, but
//! `_start` rejects them rather than acknowledging a job that never trains.

pub mod config;
pub mod fields;
pub mod outlier;

use crate::error::{EsError, Result};
use crate::node::{Node, NODE_ID};
use config::{analysis_kind, conflict, status, Mode};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

/// How long `_stop` / reset wait for a worker to exit (ES `_stop` timeout).
const STOP_TIMEOUT: Duration = Duration::from_secs(20);

/// Extra memory ES adds to a running job's `model_memory_limit`
/// (`DataFrameAnalyticsConfig.PROCESS_MEMORY_OVERHEAD`).
const PROCESS_MEMORY_OVERHEAD: u64 = 5 * 1024 * 1024;

/// Runtime state of one analytics job (ES: its persistent task + stats docs).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Rt {
    /// `started`, `stopping`, `stopped`, `failed` (missing = `stopped`).
    #[serde(default)]
    pub state: String,
    #[serde(default)]
    pub start_time: Option<i64>,
    #[serde(default)]
    pub failure_reason: Option<String>,
    /// Per-phase progress percentages reported by the analytics process.
    #[serde(default)]
    pub progress: BTreeMap<String, u64>,
    /// `data_counts` / `memory_usage` / `analysis_stats` of the process.
    #[serde(default)]
    pub data_counts: Map<String, Value>,
    #[serde(default)]
    pub memory_usage: Map<String, Value>,
    #[serde(default)]
    pub analysis_stats: Option<Value>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct State {
    /// id → rendered config (with `create_time` and `version`).
    #[serde(default)]
    configs: BTreeMap<String, Value>,
    #[serde(default)]
    rt: BTreeMap<String, Rt>,
}

impl State {
    pub(super) fn state_of(&self, id: &str) -> &str {
        self.rt.get(id).map(|r| r.state.as_str()).filter(|s| !s.is_empty()).unwrap_or("stopped")
    }
}

struct Service {
    path: PathBuf,
    st: Mutex<State>,
    /// Jobs persisted as running when this state was loaded (the node
    /// restarted under them), with failed resume attempts so far.
    interrupted: Mutex<Vec<(String, u32)>>,
}

fn registry() -> &'static Mutex<HashMap<PathBuf, Arc<Service>>> {
    static R: OnceLock<Mutex<HashMap<PathBuf, Arc<Service>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(super) fn service(node: &Node) -> Arc<Service> {
    registry()
        .lock()
        .entry(node.data.clone())
        .or_insert_with(|| {
            let path = node.data.join("_state").join("ml_dfa.json");
            let mut st: State = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
            let interrupted = interrupted_by_restart(&mut st);
            Arc::new(Service { path, st: Mutex::new(st), interrupted: Mutex::new(interrupted.into_iter().map(|id| (id, 0)).collect()) })
        })
        .clone()
}

/// Jobs persisted as running have no worker after a node restart. A
/// `started` one stays `started` and is resumed by [`resume_interrupted`]
/// (ES reassigns its persistent task); a `stopping` one is `stopped`.
fn interrupted_by_restart(st: &mut State) -> Vec<String> {
    let mut resume = Vec::new();
    for (id, rt) in st.rt.iter_mut() {
        match rt.state.as_str() {
            "started" => resume.push(id.clone()),
            "stopping" => rt.state = "stopped".into(),
            _ => {}
        }
    }
    resume
}

/// Resume attempts (one per call, the node calls every second) before an
/// interrupted job is marked `failed`.
const RESUME_ATTEMPTS: u32 = 300;

/// Whether jobs interrupted by a restart are waiting to resume.
pub fn has_interrupted(node: &Node) -> bool {
    !service(node).interrupted.lock().is_empty()
}

/// Relaunch the workers of jobs that were running when the node restarted,
/// from their first phase over the existing dest index (ES reassigns a
/// data frame analytics task and the job restarts). A job whose sources are
/// not available yet is tried again on the next call; after
/// [`RESUME_ATTEMPTS`] it is `failed` with the reason. Returns the ids
/// resumed.
pub fn resume_interrupted(node: &Arc<Node>) -> Vec<String> {
    let svc = service(node);
    let pending = std::mem::take(&mut *svc.interrupted.lock());
    let mut resumed = Vec::new();
    let mut retry = Vec::new();
    for (id, attempts) in pending {
        match resume_one(node, &id) {
            Ok(true) => resumed.push(id),
            Ok(false) => {}
            Err(e) if attempts + 1 < RESUME_ATTEMPTS => {
                tracing::debug!(id, "data frame analytics resume deferred: {e}");
                retry.push((id, attempts + 1));
            }
            Err(e) => {
                let reason = format!("could not resume the analytics job after a node restart: {}", e.reason);
                let _ = report_progress_if_started(node, &id, |rt| {
                    rt.state = "failed".into();
                    rt.failure_reason = Some(reason.clone());
                });
                crate::ml::resultsidx::audit(node, &id, "data_frame_analytics", "error", &reason);
            }
        }
    }
    svc.interrupted.lock().extend(retry);
    resumed
}

/// Ok(false): nothing to resume (deleted, stopped meanwhile, or running).
fn resume_one(node: &Arc<Node>, id: &str) -> Result<bool> {
    let st = service(node).read();
    let Some(cfg) = st.configs.get(id).cloned() else { return Ok(false) };
    if st.state_of(id) != "started" || outlier::running(node, id) {
        return Ok(false);
    }
    drop(st);
    let kind = analysis_kind(&cfg).to_string();
    if kind != "outlier_detection" {
        // Never started by this version; nothing can run it.
        let _ = report_progress_if_started(node, id, |rt| {
            rt.state = "failed".into();
            rt.failure_reason = Some(format!("[{kind}] analysis is not supported by this node"));
        });
        return Ok(false);
    }
    fields::validate_source_dest(node, &cfg)?;
    let sel = fields::detect(node, &cfg)?;
    fields::check_has_features(&sel)?;
    create_dest(node, id, &cfg)?;
    let features: Vec<String> = sel.included.iter().filter(|(n, _, _)| !sel.required.contains(n)).map(|(n, _, _)| n.clone()).collect();
    crate::ml::resultsidx::audit(node, id, "data_frame_analytics", "info", "Resuming analytics after a node restart");
    outlier::launch(node.clone(), id.to_string(), cfg, features);
    Ok(true)
}

impl Service {
    fn read(&self) -> State {
        self.st.lock().clone()
    }

    fn mutate<R>(&self, f: impl FnOnce(&mut State) -> Result<R>) -> Result<R> {
        let mut st = self.st.lock();
        let mut draft = st.clone();
        let out = f(&mut draft)?;
        if let Some(d) = self.path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(&draft)?)?;
        std::fs::rename(tmp, &self.path)?;
        *st = draft;
        Ok(out)
    }
}

/// Reset persisted analytics configurations and runtime state.
pub fn reset(node: &Node) -> Result<()> {
    outlier::stop_all(node, STOP_TIMEOUT);
    service(node).mutate(|state| { *state = State::default(); Ok(()) })
}

/// Plain string of a scalar JSON value.
pub(crate) fn plain(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        x => x.to_string(),
    }
}

fn not_found(id: &str) -> EsError {
    EsError::resource_not_found(format!("No known data frame analytics with id [{id}]"))
}

/// ML upgrade mode refuses DFA writes (`MlUpgradeModeActionFilter`).
fn check_upgrade_mode(node: &Node, action: &str) -> Result<()> {
    if crate::ml::upgrade_mode(node) {
        return Err(EsError::new(429, "status_exception", format!("Cannot perform cluster:admin/xpack/ml/data_frame/analytics/{action} action while upgrade mode is enabled")));
    }
    Ok(())
}

/// `xpack.ml.max_model_memory_limit` (transient over persistent), if set.
fn max_mml(node: &Node) -> Option<(u64, String)> {
    let cs = node.cluster_settings.read();
    let key = "xpack.ml.max_model_memory_limit";
    let nested = |m: &Map<String, Value>| m.get("xpack").and_then(|x| x.get("ml")).and_then(|x| x.get("max_model_memory_limit")).cloned();
    let v = cs.1.get(key).cloned().or_else(|| nested(&cs.1)).or_else(|| cs.0.get(key).cloned()).or_else(|| nested(&cs.0)).filter(|v| !v.is_null())?;
    let v = &v;
    crate::ilm::parse_size(&plain(v), key).ok()
}

/// Ids matching an expression (`ExpandedIdsMatcher`): concrete ids must
/// exist; wildcards must match unless `allow_no_match`.
fn expand(st: &State, expr: &str, allow_no_match: bool) -> Result<Vec<String>> {
    let expr = if expr.is_empty() { "_all" } else { expr };
    let mut out: Vec<String> = Vec::new();
    let mut missing: Vec<&str> = Vec::new();
    for tok in expr.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let wild = tok == "_all" || tok.contains('*');
        let hits: Vec<&String> = st.configs.keys().filter(|id| if wild { tok == "_all" || crate::mapping::wildcard_match(tok, id) } else { id.as_str() == tok }).collect();
        if hits.is_empty() && (!wild || !allow_no_match) {
            missing.push(tok);
        }
        out.extend(hits.into_iter().cloned());
    }
    if !missing.is_empty() {
        return Err(not_found(&missing.join(",")));
    }
    out.sort();
    out.dedup();
    Ok(out)
}

/// The single config an `_explain` / `_preview` id expression names.
fn single_config(st: &State, expr: &str) -> Result<Value> {
    let ids = expand(st, expr, false)?;
    if ids.len() > 1 {
        return Err(status(format!("expected only one config but matched [{}]", ids.join(", "))));
    }
    Ok(st.configs[&ids[0]].clone())
}

/// A stored config as the APIs return it.
fn render(cfg: &Value, exclude_generated: bool) -> Value {
    let mut m = cfg.as_object().cloned().unwrap_or_default();
    if exclude_generated {
        m.remove("create_time");
        m.remove("version");
    } else {
        m.insert("authorization".into(), json!({"roles": ["superuser"]}));
    }
    Value::Object(m)
}

fn page<T: Clone>(v: &[T], from: usize, size: usize) -> Vec<T> {
    v.iter().skip(from).take(size).cloned().collect()
}

fn check_page(from: i64, size: i64) -> Result<(usize, usize)> {
    if from < 0 {
        return Err(EsError::illegal_argument("Parameter [from] cannot be < 0"));
    }
    if size < 0 {
        return Err(EsError::illegal_argument("Parameter [size] cannot be < 0"));
    }
    Ok((from as usize, size as usize))
}

// ---- CRUD ----

/// `PUT _ml/data_frame/analytics/{id}`.
pub fn put(node: &Node, id: &str, body: &Value) -> Result<Value> {
    check_upgrade_mode(node, "put")?;
    let parsed = config::parse(body, Some(id), Mode::Put)?;
    let mut cfg = parsed.config;
    config::validate_put(&cfg)?;
    fields::validate_source_dest(node, &Value::Object(cfg.clone()))?;
    config::apply_max_mml(&mut cfg, parsed.mml_explicit, max_mml(node))?;
    cfg.insert("create_time".into(), json!(crate::date::now_millis()));
    cfg.insert("version".into(), json!(config::VERSION));
    let cfg = Value::Object(config::canonical(cfg));
    service(node).mutate(|st| {
        if st.configs.contains_key(id) {
            return Err(EsError::new(400, "resource_already_exists_exception", format!("A data frame analytics with id [{id}] already exists")));
        }
        st.configs.insert(id.to_string(), cfg.clone());
        st.rt.remove(id);
        Ok(())
    })?;
    Ok(render(&cfg, false))
}

/// `GET _ml/data_frame/analytics[/{id}]`.
pub fn get(node: &Node, expr: &str, allow_no_match: bool, from: i64, size: i64, exclude_generated: bool) -> Result<Value> {
    let (from, size) = check_page(from, size)?;
    let st = service(node).read();
    let ids = expand(&st, expr, allow_no_match)?;
    let docs: Vec<Value> = page(&ids, from, size).iter().map(|id| render(&st.configs[id], exclude_generated)).collect();
    Ok(json!({"count": ids.len(), "data_frame_analytics": docs}))
}

/// `POST _ml/data_frame/analytics/{id}/_update`.
pub fn update(node: &Node, id: &str, body: &Value) -> Result<Value> {
    check_upgrade_mode(node, "update")?;
    let u = config::parse_update(body, id)?;
    let max = max_mml(node);
    let svc = service(node);
    let out = svc.mutate(|st| {
        let cur = st.configs.get(id).cloned().ok_or_else(|| not_found(id))?;
        let running = st.state_of(id) != "stopped";
        let mut m = cur.as_object().cloned().unwrap_or_default();
        let mml_changed = u.mml.as_ref().is_some_and(|(_, r)| m.get("model_memory_limit").and_then(Value::as_str) != Some(r.as_str()));
        let threads_changed = u.max_num_threads.is_some_and(|t| m.get("max_num_threads").and_then(Value::as_i64) != Some(t));
        if running && (mml_changed || threads_changed) {
            return Err(conflict(format!("Cannot update analytics [{id}] unless it's stopped")));
        }
        if let Some(d) = &u.description {
            m.insert("description".into(), json!(d));
        }
        if let Some((_, r)) = &u.mml {
            m.insert("model_memory_limit".into(), json!(r));
            config::apply_max_mml(&mut m, true, max.clone())?;
        }
        if let Some(b) = u.allow_lazy_start {
            m.insert("allow_lazy_start".into(), json!(b));
        }
        if let Some(t) = u.max_num_threads {
            m.insert("max_num_threads".into(), json!(t));
        }
        if let Some(meta) = &u.meta {
            m.insert("_meta".into(), meta.clone());
        }
        let cfg = Value::Object(config::canonical(m));
        st.configs.insert(id.to_string(), cfg.clone());
        Ok(cfg)
    })?;
    Ok(render(&out, false))
}

/// `DELETE _ml/data_frame/analytics/{id}`.
pub fn delete(node: &Node, id: &str, force: bool) -> Result<Value> {
    check_upgrade_mode(node, "delete")?;
    let svc = service(node);
    let target = {
        let st = svc.read();
        if id.contains('*') || id == "_all" {
            let ids = expand(&st, id, true)?;
            if ids.len() != 1 {
                return Err(status(format!("Expected a single match for data frame analytics [{id}] but got [{}]", ids.len())));
            }
            ids[0].clone()
        } else {
            id.to_string()
        }
    };
    svc.mutate(|st| {
        if !st.configs.contains_key(&target) {
            return Err(not_found(&target));
        }
        let state = st.state_of(&target).to_string();
        if state != "stopped" && !force {
            return Err(conflict(format!("Cannot delete data frame analytics [{target}] while its status is [{state}]")));
        }
        Ok(())
    })?;
    if !outlier::stop_and_wait(node, &target, STOP_TIMEOUT) {
        return Err(EsError::new(503, "timeout_exception", format!("timed out stopping data frame analytics [{target}] before deleting it")));
    }
    svc.mutate(|st| {
        st.configs.remove(&target);
        st.rt.remove(&target);
        Ok(())
    })?;
    // #534: a start or restart resume that read the job before the delete
    // may have launched a worker since the first stop; a launch now sees
    // the job gone (`outlier::launch`), so this one is the last.
    if !outlier::stop_and_wait(node, &target, STOP_TIMEOUT) {
        return Err(EsError::new(503, "timeout_exception", format!("timed out stopping data frame analytics [{target}] before deleting it")));
    }
    Ok(json!({"acknowledged": true}))
}

// ---- start / stop ----

/// Record progress / counts reported by the analytics process.
pub fn report_progress(node: &Node, id: &str, f: impl FnOnce(&mut Rt)) -> Result<()> {
    service(node).mutate(|st| {
        if st.configs.contains_key(id) {
            f(st.rt.entry(id.to_string()).or_default());
        }
        Ok(())
    })
}

/// [`report_progress`] from a worker: only while the job is still
/// `started` (a stopped, failed or deleted job's state is not overwritten).
fn report_progress_if_started(node: &Node, id: &str, f: impl FnOnce(&mut Rt)) -> Result<()> {
    service(node).mutate(|st| {
        if st.configs.contains_key(id) {
            if let Some(rt) = st.rt.get_mut(id).filter(|rt| rt.state == "started") {
                f(rt);
            }
        }
        Ok(())
    })
}

/// The ES-shaped refusal to start an analysis kind this node cannot run.
fn unsupported_kind(id: &str, kind: &str) -> EsError {
    status(format!(
        "Cannot start data frame analytics [{id}]: [{kind}] analysis is not supported by this node (it needs the boosted tree trainer); only [outlier_detection] analytics can be started"
    ))
}

/// Create the dest index from the source mappings and settings, or add the
/// result mappings to an existing one (ES `DestinationIndex`).
fn create_dest(node: &Node, id: &str, cfg: &Value) -> Result<()> {
    let dest = cfg["dest"]["index"].as_str().unwrap_or("");
    if dest.is_empty() {
        return Ok(());
    }
    // Conflicting source mappings (#537) and a results field the copied
    // documents already have (#538) are refused before anything is written.
    let mut props = fields::source_properties(node, cfg)?;
    fields::check_results_field_absent(cfg, &props)?;
    let results = cfg["dest"]["results_field"].as_str().unwrap_or("ml");
    let kind = analysis_kind(cfg);
    let mut res = Map::new();
    if kind == "outlier_detection" {
        res.insert("feature_influence".into(), json!({"type": "nested", "dynamic": "false", "properties": {"feature_name": {"type": "keyword"}, "influence": {"type": "double"}}}));
        res.insert("outlier_score".into(), json!({"type": "double"}));
    } else {
        let a = &cfg["analysis"][kind];
        let dep = a["dependent_variable"].as_str().unwrap_or("");
        let dep_mapping = props.get(dep).cloned().unwrap_or_else(|| json!({"type": "keyword"}));
        res.insert("is_training".into(), json!({"type": "boolean"}));
        res.insert(a["prediction_field_name"].as_str().unwrap_or("prediction").to_string(), dep_mapping);
    }
    let existing = node.get_index(dest).map(|s| vec![s]).unwrap_or_else(|| node.all_indices().into_iter().filter(|s| s.aliases().contains_key(dest)).collect());
    if !existing.is_empty() {
        // ES `updateMappingsToDestIndex` (#536): the result and incremental
        // id mappings go onto the existing dest (its write index) before the
        // job writes, so scores are searchable whatever its `dynamic` is.
        let write = if existing.len() == 1 {
            existing[0].clone()
        } else {
            existing.iter().find(|s| s.aliases().get(dest).and_then(|d| d.get("is_write_index")).and_then(Value::as_bool) == Some(true)).cloned().unwrap_or_else(|| existing[0].clone())
        };
        let m = write.meta().mappings.to_json();
        let have = m.get("properties").and_then(Value::as_object).cloned().unwrap_or_default();
        if have.contains_key("ml__incremental_id") {
            // This job's own dest (a restart or resume): already prepared.
            return Ok(());
        }
        fields::check_results_field_absent(cfg, &have)?;
        let mut add = Map::new();
        if !results.is_empty() {
            add.insert(results.to_string(), json!({"properties": res}));
        }
        add.insert("ml__incremental_id".into(), json!({"type": "long"}));
        let body = json!({"properties": add});
        return match crate::coord::current() {
            Some(c) => c("PUT", &format!("/{}/_mapping", crate::coord::path_segment(&write.name)), Some(body)).map(|_| ()),
            None => write.put_mapping(&body),
        };
    }
    if !results.is_empty() {
        props.insert(results.to_string(), json!({"properties": res}));
    }
    props.insert("ml__incremental_id".into(), json!({"type": "long"}));
    let settings = fields::dest_settings(node, cfg)?;
    let body = json!({
        "settings": settings,
        "mappings": {
            "dynamic": "false",
            "_meta": {"analytics": id, "created_by": "data-frame-analytics", "version": {"created": config::VERSION}, "creation_date_in_millis": crate::date::now_millis()},
            "properties": props
        }
    });
    let created = match crate::coord::current() {
        Some(c) => c("PUT", &format!("/{}", crate::coord::path_segment(dest)), Some(body)).map(|_| ()),
        None => node.create_index(dest, &body).map(|_| ()),
    };
    match created {
        Ok(_) => Ok(()),
        Err(e) if e.kind == "resource_already_exists_exception" => Ok(()),
        Err(e) => Err(e),
    }
}

/// `POST _ml/data_frame/analytics/{id}/_start`.
pub fn start(node: &Arc<Node>, id: &str, body_id: Option<&str>) -> Result<Value> {
    if let Some(b) = body_id {
        if b != id {
            return Err(EsError::illegal_argument(format!("Inconsistent id; '{b}' specified in the body differs from '{id}' specified as a URL argument")));
        }
    }
    check_upgrade_mode(node, "start")?;
    let svc = service(node);
    let (cfg, state, prior) = {
        let st = svc.read();
        let cfg = st.configs.get(id).cloned().ok_or_else(|| not_found(id))?;
        (cfg, st.state_of(id).to_string(), st.rt.get(id).cloned())
    };
    if state != "stopped" && state != "failed" {
        return Err(conflict(format!("Cannot start data frame analytics [{id}] because it has already been started"))
            .caused_by(EsError::new(400, "resource_already_exists_exception", format!("task with id {{data_frame_analytics-{id}}} already exist"))));
    }
    let kind = analysis_kind(&cfg).to_string();
    // ES `DataFrameAnalyticsManager`: a job whose every phase completed
    // finishes again at once; one with progress resumes (its dest index
    // already holds the reindexed documents).
    let phase_list = phases(&kind);
    let progress_of = |p: &str| prior.as_ref().and_then(|r| r.progress.get(p).copied()).unwrap_or(0);
    let finished = phase_list.iter().all(|p| progress_of(p) == 100);
    let restarting = !finished && phase_list.iter().any(|p| progress_of(p) > 0);
    if finished {
        return Ok(json!({"acknowledged": true, "node": NODE_ID}));
    }
    fields::validate_source_dest(node, &cfg)?;
    let sel = fields::detect(node, &cfg)?;
    fields::check_has_features(&sel)?;
    let rows = fields::usable_docs(node, &cfg, &sel)?;
    if rows == 0 {
        return Err(status(format!("Unable to start {id} as no documents in the source indices {} {}", fields::source_list(&cfg), fields::NO_DATA_SUFFIX)));
    }
    if kind != "outlier_detection" {
        return Err(unsupported_kind(id, &kind));
    }
    if rows > outlier::MAX_ROWS {
        return Err(status(format!("Cannot start data frame analytics [{id}]: outlier detection supports at most [{}] documents but [{rows}] have every analysed field", outlier::MAX_ROWS)));
    }
    if outlier::running(node, id) {
        return Err(conflict(format!("Cannot start data frame analytics [{id}] because it has already been started")));
    }
    let features: Vec<String> = sel.included.iter().filter(|(n, _, _)| !sel.required.contains(n)).map(|(n, _, _)| n.clone()).collect();
    // ES `TransportStartDataFrameAnalyticsAction`: an estimate above the
    // configured limit only warns at start; the analysis itself then fails
    // before it allocates past the limit (ml-cpp's hard limit, here
    // `outlier::admit`).
    let expected = outlier::estimate_for(&cfg["analysis"]["outlier_detection"], rows, features.len() as u64);
    if expected > outlier::memory_limit(&cfg) {
        let mml = cfg["model_memory_limit"].as_str().unwrap_or(config::DEFAULT_MML);
        crate::ml::resultsidx::audit(
            node,
            id,
            "data_frame_analytics",
            "warning",
            &format!(
                "Configured model memory limit [{mml}] is lower than the expected memory usage [{}]. The analytics job may fail due to configured memory constraints.",
                crate::memory::byte_size(expected)
            ),
        );
    }
    let dest = cfg["dest"]["index"].as_str().unwrap_or("").to_string();
    if !restarting && (node.get_index(&dest).is_some() || node.all_indices().iter().any(|s| s.aliases().contains_key(&dest))) {
        let r = crate::ml::search(node, &dest, &json!({"size": 0, "track_total_hits": true}))?;
        if r["hits"]["total"]["value"].as_u64().unwrap_or(0) > 0 {
            return Err(status(format!("dest index [{dest}] must be empty")));
        }
    }
    create_dest(node, id, &cfg)?;
    svc.mutate(|st| {
        if !st.configs.contains_key(id) {
            return Err(not_found(id));
        }
        let rt = st.rt.entry(id.to_string()).or_default();
        let kept = if restarting { std::mem::take(&mut rt.progress) } else { BTreeMap::new() };
        *rt = Rt { state: "started".into(), start_time: Some(crate::date::now_millis()), progress: kept, ..Default::default() };
        Ok(())
    })?;
    outlier::launch(node.clone(), id.to_string(), cfg, features);
    Ok(json!({"acknowledged": true, "node": NODE_ID}))
}

/// `POST _ml/data_frame/analytics/{id}/_stop`.
pub fn stop(node: &Node, expr: &str, body_id: Option<&str>, allow_no_match: bool, _force: bool) -> Result<Value> {
    if let Some(b) = body_id {
        if b != expr {
            return Err(EsError::illegal_argument(format!("Inconsistent id; '{b}' specified in the body differs from '{expr}' specified as a URL argument")));
        }
    }
    check_upgrade_mode(node, "stop")?;
    let svc = service(node);
    let ids = expand(&svc.read(), expr, allow_no_match)?;
    // The worker exits at its next page / chunk; progress is kept so a
    // later `_start` resumes.
    for id in &ids {
        if !outlier::stop_and_wait(node, id, STOP_TIMEOUT) {
            return Err(EsError::new(503, "timeout_exception", format!("timed out waiting for data frame analytics [{id}] to stop")));
        }
    }
    svc.mutate(|st| {
        for id in &ids {
            if let Some(rt) = st.rt.get_mut(id) {
                if rt.state != "stopped" {
                    rt.state = "stopped".into();
                    rt.failure_reason = None;
                }
            }
        }
        Ok(())
    })?;
    // #534: a start or restart resume that read the job as started before
    // the stop may have launched a worker since; a launch now sees it
    // stopped (`outlier::launch`), so stopping again leaves none.
    for id in &ids {
        if !outlier::stop_and_wait(node, id, STOP_TIMEOUT) {
            return Err(EsError::new(503, "timeout_exception", format!("timed out waiting for data frame analytics [{id}] to stop")));
        }
    }
    Ok(json!({"stopped": true}))
}

// ---- stats ----

fn phases(kind: &str) -> &'static [&'static str] {
    if kind == "outlier_detection" {
        &["reindexing", "loading_data", "computing_outliers", "writing_results"]
    } else {
        &["reindexing", "loading_data", "feature_selection", "coarse_parameter_search", "fine_tuning_parameters", "final_training", "writing_results", "inference"]
    }
}

fn progress(cfg: &Value, rt: Option<&Rt>) -> Vec<(&'static str, u64)> {
    phases(analysis_kind(cfg)).iter().map(|p| (*p, rt.and_then(|r| r.progress.get(*p).copied()).unwrap_or(0))).collect()
}

fn node_json(node: &Node) -> Value {
    json!({"id": NODE_ID, "name": node.name, "ephemeral_id": NODE_ID, "transport_address": "127.0.0.1:9300", "attributes": {}})
}

fn stats_doc(node: &Node, id: &str, cfg: &Value, st: &State, verbose: bool) -> Value {
    let rt = st.rt.get(id);
    let state = st.state_of(id);
    let mut m = Map::new();
    m.insert("id".into(), json!(id));
    m.insert("state".into(), json!(state));
    if let Some(r) = rt.and_then(|r| r.failure_reason.clone()) {
        m.insert("failure_reason".into(), json!(r));
    }
    m.insert("progress".into(), Value::Array(progress(cfg, rt).into_iter().map(|(p, v)| json!({"phase": p, "progress_percent": v})).collect()));
    let mut dc = json!({"training_docs_count": 0, "test_docs_count": 0, "skipped_docs_count": 0});
    for (k, v) in rt.map(|r| &r.data_counts).into_iter().flatten() {
        dc[k] = v.clone();
    }
    m.insert("data_counts".into(), dc);
    let mut mu = json!({"peak_usage_bytes": 0, "status": "ok"});
    for (k, v) in rt.map(|r| &r.memory_usage).into_iter().flatten() {
        mu[k] = v.clone();
    }
    m.insert("memory_usage".into(), mu);
    if let Some(a) = rt.and_then(|r| r.analysis_stats.clone()) {
        let _ = verbose;
        m.insert("analysis_stats".into(), a);
    }
    if state != "stopped" && state != "failed" {
        if crate::ml::upgrade_mode(node) {
            m.insert("assignment_explanation".into(), json!("persistent task cannot be assigned while upgrade mode is enabled."));
        } else {
            m.insert("node".into(), node_json(node));
            m.insert("assignment_explanation".into(), json!(""));
        }
    }
    Value::Object(m)
}

/// `GET _ml/data_frame/analytics[/{id}]/_stats`.
pub fn stats(node: &Node, expr: &str, allow_no_match: bool, from: i64, size: i64, verbose: bool) -> Result<Value> {
    let (from, size) = check_page(from, size)?;
    let st = service(node).read();
    let ids = expand(&st, expr, allow_no_match)?;
    let docs: Vec<Value> = page(&ids, from, size).iter().map(|id| stats_doc(node, id, &st.configs[id], &st, verbose)).collect();
    Ok(json!({"count": ids.len(), "data_frame_analytics": docs}))
}

// ---- explain / preview ----

/// The config `_explain` / `_preview` work on: a stored one by id, or one
/// given in the body (exactly one of the two).
fn resolve_config(node: &Node, id: Option<&str>, body: Option<&Value>) -> Result<Value> {
    match (id.filter(|i| !i.is_empty()), body.filter(|b| !b.is_null() && b.as_object().map(|o| !o.is_empty()).unwrap_or(true))) {
        (Some(_), Some(_)) => Err(status("Please provide either a job [id] or the config object but not both")),
        (None, None) => Err(status("Please provide a job [id] or the config object")),
        (Some(i), None) => single_config(&service(node).read(), i),
        (None, Some(b)) => Ok(Value::Object(config::parse(b, None, Mode::Explain)?.config)),
    }
}

/// `_ml/data_frame/analytics[/{id}]/_explain` (body: a config).
pub fn explain(node: &Node, id: Option<&str>, body: Option<&Value>) -> Result<Value> {
    let cfg = resolve_config(node, id, body)?;
    fields::explain(node, &cfg)
}

/// `_ml/data_frame/analytics[/{id}]/_preview` (body: `{"config": {..}}`).
pub fn preview(node: &Node, id: Option<&str>, body: Option<&Value>) -> Result<Value> {
    let cfg_body = match body {
        Some(Value::Object(o)) => {
            if let Some(k) = o.keys().find(|k| k.as_str() != "config") {
                return Err(EsError::x_content_parse(format!("[1:1] [preview_data_frame_analytics_response] unknown field [{k}]")));
            }
            o.get("config").cloned()
        }
        _ => None,
    };
    let cfg = resolve_config(node, id, cfg_body.as_ref())?;
    fields::preview(node, &cfg)
}

// ---- node-level views (cat, memory stats, upgrade mode) ----

/// One `_cat/ml/data_frame/analytics` row.
pub struct CatRow {
    pub id: String,
    pub kind: String,
    pub create_time: Option<i64>,
    pub version: String,
    pub source_index: String,
    pub dest_index: String,
    pub description: String,
    pub model_memory_limit: String,
    pub state: String,
    pub failure_reason: String,
    /// `reindexing:0,loading_data:0,...`.
    pub progress: String,
    pub assignment_explanation: String,
    /// Whether the job is assigned to this node (`node.*` columns).
    pub assigned: bool,
}

/// Rows of `_cat/ml/data_frame/analytics[/{id}]`, sorted by id.
pub fn cat_rows(node: &Node, expr: &str, allow_no_match: bool) -> Result<Vec<CatRow>> {
    let st = service(node).read();
    let ids = expand(&st, expr, allow_no_match)?;
    let upgrade = crate::ml::upgrade_mode(node);
    Ok(ids
        .iter()
        .map(|id| {
            let cfg = &st.configs[id];
            let rt = st.rt.get(id);
            let state = st.state_of(id).to_string();
            let running = state != "stopped" && state != "failed";
            CatRow {
                id: id.clone(),
                kind: analysis_kind(cfg).to_string(),
                create_time: cfg["create_time"].as_i64(),
                version: cfg["version"].as_str().unwrap_or("").to_string(),
                source_index: cfg["source"]["index"].as_array().map(|a| a.iter().map(plain).collect::<Vec<_>>().join(",")).unwrap_or_default(),
                dest_index: cfg["dest"]["index"].as_str().unwrap_or("").to_string(),
                description: cfg["description"].as_str().unwrap_or("").to_string(),
                model_memory_limit: cfg["model_memory_limit"].as_str().unwrap_or("").to_string(),
                failure_reason: rt.and_then(|r| r.failure_reason.clone()).unwrap_or_default(),
                progress: progress(cfg, rt).iter().map(|(p, v)| format!("{p}:{v}")).collect::<Vec<_>>().join(","),
                assignment_explanation: if running && upgrade { "persistent task cannot be assigned while upgrade mode is enabled.".into() } else { String::new() },
                assigned: running && !upgrade,
                state,
            }
        })
        .collect())
}

/// Ids of analytics jobs that are not stopped (ES: DFA persistent tasks).
pub fn running_ids(node: &Node) -> Vec<String> {
    let st = service(node).read();
    st.configs.keys().filter(|id| !matches!(st.state_of(id), "stopped" | "failed")).cloned().collect()
}

/// Memory reserved by running analytics jobs (`_ml/memory/_stats`
/// `data_frame_analytics_in_bytes`): `model_memory_limit` plus the process
/// overhead for each running job; 0 in upgrade mode (tasks unassigned).
pub fn memory_in_bytes(node: &Node) -> u64 {
    if crate::ml::upgrade_mode(node) {
        return 0;
    }
    let st = service(node).read();
    st.configs
        .iter()
        .filter(|(id, _)| !matches!(st.state_of(id), "stopped" | "failed"))
        .map(|(_, c)| crate::ilm::parse_size(c["model_memory_limit"].as_str().unwrap_or(config::DEFAULT_MML), "model_memory_limit").map(|(b, _)| b).unwrap_or(0) + PROCESS_MEMORY_OVERHEAD)
        .sum()
}

/// The model memory limit of each running analytics job, for the
/// autoscaling ML decider (which adds ES's process overhead).
pub fn running_memory(node: &Node) -> Vec<u64> {
    let st = service(node).read();
    st.configs
        .iter()
        .filter(|(id, _)| !matches!(st.state_of(id), "stopped" | "failed"))
        .map(|(_, c)| crate::ilm::parse_size(c["model_memory_limit"].as_str().unwrap_or(config::DEFAULT_MML), "model_memory_limit").map(|(b, _)| b).unwrap_or(0))
        .collect()
}

/// Number of configured analytics jobs (ES `_xpack/usage`
/// `data_frame_analytics_jobs._all.count`).
pub fn count(node: &Node) -> usize {
    service(node).read().configs.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::WriteReq;
    use crate::shard::WriteCond;

    /// 40 inliers on a small grid of (x, y) and one far point `far`; a
    /// document missing `y` (skipped).
    fn source(node: &Node) {
        let svc = node.create_index("od-src", &json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0},
            "mappings": {"properties": {"x": {"type": "double"}, "y": {"type": "double"}, "label": {"type": "keyword"}}}})).unwrap();
        let mut reqs: Vec<WriteReq> = (0..40)
            .map(|i| WriteReq::Index { id: Some(format!("in-{i}")), source: json!({"x": (i % 8) as f64, "y": (i / 8) as f64, "label": "a"}).to_string(), routing: None, cond: WriteCond::default() })
            .collect();
        reqs.push(WriteReq::Index { id: Some("far".into()), source: json!({"x": 80.0, "y": 2.0, "label": "b"}).to_string(), routing: None, cond: WriteCond::default() });
        reqs.push(WriteReq::Index { id: Some("partial".into()), source: json!({"x": 1.0, "label": "c"}).to_string(), routing: None, cond: WriteCond::default() });
        assert!(svc.write(reqs, false).iter().all(|r| r.is_ok()));
        svc.refresh().unwrap();
    }

    fn put_od(node: &Node, id: &str) {
        put(node, id, &json!({"source": {"index": "od-src"}, "dest": {"index": format!("{id}-dest")}, "analysis": {"outlier_detection": {}}, "analyzed_fields": {"includes": ["x", "y"]}})).unwrap();
    }

    fn wait_worker(node: &Node, id: &str) {
        let deadline = std::time::Instant::now() + Duration::from_secs(60);
        while outlier::running(node, id) {
            assert!(std::time::Instant::now() < deadline, "worker did not finish");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn rt(node: &Node, id: &str) -> Rt {
        service(node).read().rt.get(id).cloned().unwrap_or_default()
    }

    fn dest_docs(node: &Node, index: &str) -> HashMap<String, Value> {
        let r = crate::ml::search(node, index, &json!({"size": 100})).unwrap();
        r["hits"]["hits"].as_array().unwrap().iter().map(|h| (h["_id"].as_str().unwrap().to_string(), h["_source"].clone())).collect()
    }

    fn check_results(node: &Node, id: &str) {
        let r = rt(node, id);
        assert_eq!(r.state, "stopped", "terminal state");
        assert!(phases("outlier_detection").iter().all(|p| r.progress.get(*p) == Some(&100)), "{:?}", r.progress);
        assert_eq!(r.data_counts["training_docs_count"], json!(41));
        assert_eq!(r.data_counts["skipped_docs_count"], json!(1));
        let docs = dest_docs(node, &format!("{id}-dest"));
        assert_eq!(docs.len(), 42, "every source document is in dest");
        assert!(docs["partial"].get("ml").is_none(), "a skipped document has no results");
        assert_eq!(docs["far"]["label"], "b", "source fields are copied");
        let score = |d: &str| docs[d]["ml"]["outlier_score"].as_f64().unwrap();
        assert!(score("far") > 0.9, "{}", score("far"));
        assert!(docs.keys().filter(|k| k.starts_with("in-")).all(|k| score(k) < score("far")));
        let fi = docs["far"]["ml"]["feature_influence"].as_array().unwrap();
        let x = fi.iter().find(|f| f["feature_name"] == "x").unwrap()["influence"].as_f64().unwrap();
        assert!(x > 0.9, "{fi:?}");
        let s = stats(node, id, false, 0, 100, false).unwrap();
        assert_eq!(s["data_frame_analytics"][0]["state"], "stopped");
        assert!(s["data_frame_analytics"][0]["progress"].as_array().unwrap().iter().all(|p| p["progress_percent"] == 100));
    }

    /// #353: `source._source` excludes a sparse column: it is neither an
    /// analysis input (no document is skipped for lacking it) nor copied to
    /// the destination.
    #[test]
    fn source_filtering_selects_and_copies() {
        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        source(&node);
        // `label` is a keyword (not analyzed anyway); `y` is sparse: the
        // `partial` doc lacks it. Excluding `y` analyzes `x` alone.
        put(&node, "sf", &json!({"source": {"index": "od-src", "_source": {"excludes": ["y", "label"]}}, "dest": {"index": "sf-dest"}, "analysis": {"outlier_detection": {}}})).unwrap();
        start(&node, "sf", None).unwrap();
        wait_worker(&node, "sf");
        let r = rt(&node, "sf");
        assert_eq!(r.state, "stopped");
        assert_eq!(r.data_counts["skipped_docs_count"], json!(0), "no doc lacks an analyzed field");
        assert_eq!(r.data_counts["training_docs_count"], json!(42));
        let docs = dest_docs(&node, "sf-dest");
        assert!(docs.values().all(|d| d.get("y").is_none() && d.get("label").is_none()), "excluded fields are not copied");
        assert!(docs["far"]["ml"]["outlier_score"].as_f64().unwrap() > 0.9);
        let fi = docs["far"]["ml"]["feature_influence"].as_array().cloned().unwrap_or_default();
        assert!(fi.iter().all(|f| f["feature_name"] != "y"), "{fi:?}");
    }

    /// #51: a started outlier detection job really runs: dest results,
    /// progress, data counts and a terminal `stopped` state.
    #[test]
    fn outlier_detection_runs_to_completion() {
        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        source(&node);
        put_od(&node, "od");
        let r = start(&node, "od", None).unwrap();
        assert_eq!(r["acknowledged"], true);
        wait_worker(&node, "od");
        check_results(&node, "od");
        // Starting a finished job completes at once.
        start(&node, "od", None).unwrap();
        assert_eq!(rt(&node, "od").state, "stopped");
    }

    /// Stopping part way keeps progress; starting again resumes over the
    /// non-empty dest index and finishes with the same results.
    #[test]
    fn outlier_detection_stop_and_restart() {
        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        source(&node);
        put_od(&node, "odr");
        outlier::hold::set(&node, "odr", true);
        start(&node, "odr", None).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while rt(&node, "odr").progress.get("loading_data") != Some(&100) {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(stats(&node, "odr", false, 0, 100, false).unwrap()["data_frame_analytics"][0]["state"], "started");
        stop(&node, "odr", None, false, false).unwrap();
        assert!(!outlier::running(&node, "odr"), "stop waits for the worker");
        let r = rt(&node, "odr");
        assert_eq!(r.state, "stopped");
        assert_eq!(r.progress.get("reindexing"), Some(&100));
        assert_ne!(r.progress.get("writing_results"), Some(&100));
        outlier::hold::set(&node, "odr", false);
        start(&node, "odr", None).unwrap();
        wait_worker(&node, "odr");
        check_results(&node, "odr");
    }

    fn index_docs(node: &Node, index: &str, docs: &[(&str, Value)]) {
        let svc = node.get_index(index).unwrap();
        let reqs = docs.iter().map(|(id, d)| WriteReq::Index { id: Some(id.to_string()), source: d.to_string(), routing: None, cond: WriteCond::default() }).collect();
        assert!(svc.write(reqs, false).iter().all(|r| r.is_ok()));
        svc.refresh().unwrap();
    }

    /// #534: a launch for a job no longer started (stopped, or deleted,
    /// after its caller read it) starts no worker.
    #[test]
    fn launch_after_stop_or_delete_starts_no_worker() {
        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        source(&node);
        put_od(&node, "late");
        let cfg = service(&node).read().configs.get("late").cloned().unwrap();
        // Never started (or stopped meanwhile).
        outlier::launch(node.clone(), "late".into(), cfg.clone(), vec!["x".into(), "y".into()]);
        assert!(!outlier::running(&node, "late"));
        assert!(node.get_index("late-dest").is_none(), "nothing was written");
        delete(&node, "late", true).unwrap();
        outlier::launch(node.clone(), "late".into(), cfg, vec!["x".into(), "y".into()]);
        assert!(!outlier::running(&node, "late"));
    }

    /// #535: the dest index keeps the analysis settings its copied mappings
    /// refer to (an unused text field with a custom analyzer).
    #[test]
    fn dest_keeps_source_analysis_settings() {
        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        node.create_index("an-src", &json!({"settings": {"analysis": {"analyzer": {"my_an": {"type": "custom", "tokenizer": "whitespace", "filter": ["lowercase"]}}}},
            "mappings": {"properties": {"x": {"type": "double"}, "t": {"type": "text", "analyzer": "my_an"}}}})).unwrap();
        let docs: Vec<(String, Value)> = (0..10).map(|i| (format!("d{i}"), json!({"x": i as f64, "t": "Some Text"}))).collect();
        index_docs(&node, "an-src", &docs.iter().map(|(i, d)| (i.as_str(), d.clone())).collect::<Vec<_>>());
        put(&node, "an", &json!({"source": {"index": "an-src"}, "dest": {"index": "an-dest"}, "analysis": {"outlier_detection": {}}, "analyzed_fields": {"includes": ["x"]}})).unwrap();
        start(&node, "an", None).unwrap();
        wait_worker(&node, "an");
        assert_eq!(rt(&node, "an").state, "stopped", "{:?}", rt(&node, "an").failure_reason);
        let dest = node.get_index("an-dest").expect("dest created");
        assert_eq!(dest.meta().settings.get("index.analysis.analyzer.my_an.tokenizer"), Some(&json!("whitespace")));
    }

    /// #536: a pre-created empty dest with `dynamic: false` gets the result
    /// mappings, so scores are searchable.
    #[test]
    fn existing_dest_gets_result_mappings() {
        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        source(&node);
        node.create_index("pre-dest", &json!({"mappings": {"dynamic": false, "properties": {"x": {"type": "double"}, "y": {"type": "double"}}}})).unwrap();
        put(&node, "pre", &json!({"source": {"index": "od-src"}, "dest": {"index": "pre-dest"}, "analysis": {"outlier_detection": {}}, "analyzed_fields": {"includes": ["x", "y"]}})).unwrap();
        start(&node, "pre", None).unwrap();
        wait_worker(&node, "pre");
        assert_eq!(rt(&node, "pre").state, "stopped", "{:?}", rt(&node, "pre").failure_reason);
        let r = crate::ml::search(&node, "pre-dest", &json!({"size": 0, "track_total_hits": true, "query": {"range": {"ml.outlier_score": {"gte": 0.9}}}})).unwrap();
        assert!(r["hits"]["total"]["value"].as_u64().unwrap() >= 1, "{r}");
        let r = crate::ml::search(&node, "pre-dest", &json!({"size": 0, "track_total_hits": true, "query": {"exists": {"field": "ml.outlier_score"}}})).unwrap();
        assert_eq!(r["hits"]["total"]["value"], 41);
    }

    /// #537: sources mapping a field differently are refused before the
    /// dest is created.
    #[test]
    fn conflicting_source_mappings_are_refused() {
        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        for (i, t) in [("cf-a", "long"), ("cf-b", "double")] {
            node.create_index(i, &json!({"mappings": {"properties": {"x": {"type": t}}}})).unwrap();
            let docs: Vec<(String, Value)> = (0..5).map(|n| (format!("{i}-{n}"), json!({"x": n}))).collect();
            index_docs(&node, i, &docs.iter().map(|(i, d)| (i.as_str(), d.clone())).collect::<Vec<_>>());
        }
        for order in [["cf-a", "cf-b"], ["cf-b", "cf-a"]] {
            let id = format!("cf-{}", order[0]);
            put(&node, &id, &json!({"source": {"index": order}, "dest": {"index": format!("{id}-dest")}, "analysis": {"outlier_detection": {}}})).unwrap();
            let e = start(&node, &id, None).unwrap_err();
            assert!(e.reason.contains("cannot merge [properties] mappings because of differences for field [x]"), "{}", e.reason);
            assert!(node.get_index(&format!("{id}-dest")).is_none());
        }
    }

    /// #538: a results field the source documents already have is refused
    /// (the results would replace it).
    #[test]
    fn results_field_present_in_source_is_refused() {
        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        node.create_index("rf-src", &json!({"mappings": {"properties": {"x": {"type": "double"}, "ml": {"properties": {"business_tag": {"type": "keyword"}}}}}})).unwrap();
        let docs: Vec<(String, Value)> = (0..5).map(|n| (format!("d{n}"), json!({"x": n, "ml": {"business_tag": "keep"}}))).collect();
        index_docs(&node, "rf-src", &docs.iter().map(|(i, d)| (i.as_str(), d.clone())).collect::<Vec<_>>());
        put(&node, "rf", &json!({"source": {"index": "rf-src"}, "dest": {"index": "rf-dest"}, "analysis": {"outlier_detection": {}}, "analyzed_fields": {"includes": ["x"]}})).unwrap();
        let e = start(&node, "rf", None).unwrap_err();
        assert_eq!(e.reason, "A field that matches the dest.results_field [ml] already exists; please set a different results_field");
        // Another results field works, and keeps the source object.
        put(&node, "rf2", &json!({"source": {"index": "rf-src"}, "dest": {"index": "rf2-dest", "results_field": "out"}, "analysis": {"outlier_detection": {}}, "analyzed_fields": {"includes": ["x"]}})).unwrap();
        start(&node, "rf2", None).unwrap();
        wait_worker(&node, "rf2");
        let docs = dest_docs(&node, "rf2-dest");
        assert!(docs.values().all(|d| d["ml"]["business_tag"] == "keep" && d["out"]["outlier_score"].is_number()), "{docs:?}");
    }

    /// #572 / #573: a duplicate `_id` across sources is analysed once (the
    /// copy the dest keeps); a multi-valued feature is skipped, not read as
    /// its first value.
    #[test]
    fn duplicate_ids_and_multi_valued_features_follow_the_dest() {
        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        for i in ["dup-a", "dup-b"] {
            node.create_index(i, &json!({"mappings": {"properties": {"x": {"type": "double"}}}})).unwrap();
        }
        let a: Vec<(String, Value)> = (0..10).map(|n| (format!("a{n}"), json!({"x": n as f64}))).chain([("dup".to_string(), json!({"x": 100.0})), ("multi".to_string(), json!({"x": [0.0, 100.0]}))]).collect();
        index_docs(&node, "dup-a", &a.iter().map(|(i, d)| (i.as_str(), d.clone())).collect::<Vec<_>>());
        index_docs(&node, "dup-b", &[("dup", json!({"x": 0.0}))]);
        put(&node, "dup", &json!({"source": {"index": ["dup-a", "dup-b"]}, "dest": {"index": "dup-dest"}, "analysis": {"outlier_detection": {}}, "analyzed_fields": {"includes": ["x"]}})).unwrap();
        start(&node, "dup", None).unwrap();
        wait_worker(&node, "dup");
        let r = rt(&node, "dup");
        assert_eq!(r.state, "stopped", "{:?}", r.failure_reason);
        assert_eq!(r.data_counts["training_docs_count"], json!(11), "10 + one dup");
        assert_eq!(r.data_counts["skipped_docs_count"], json!(1), "the multi-valued doc");
        let docs = dest_docs(&node, "dup-dest");
        assert!(docs["multi"].get("ml").is_none());
    }

    /// A job that was running when its node restarted resumes by itself
    /// (ES reassigns the task): `started` throughout, then finished.
    #[test]
    fn restart_resumes_running_jobs() {
        let mut st = State::default();
        st.rt.insert("a".into(), Rt { state: "started".into(), progress: [("reindexing".to_string(), 100u64)].into_iter().collect(), ..Default::default() });
        st.rt.insert("b".into(), Rt { state: "stopped".into(), ..Default::default() });
        st.rt.insert("c".into(), Rt { state: "stopping".into(), ..Default::default() });
        assert_eq!(interrupted_by_restart(&mut st), vec!["a".to_string()]);
        assert_eq!(st.rt["a"].state, "started");
        assert_eq!(st.rt["a"].progress.get("reindexing"), Some(&100));
        assert_eq!(st.rt["b"].state, "stopped");
        assert_eq!(st.rt["c"].state, "stopped");

        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        source(&node);
        put_od(&node, "odf");
        outlier::hold::set(&node, "odf", true);
        start(&node, "odf", None).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while rt(&node, "odf").progress.get("loading_data") != Some(&100) {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
        // The process dies under the worker: the persisted state still says
        // started. Restart the node from its data path.
        assert!(outlier::stop_and_wait(&node, "odf", Duration::from_secs(10)));
        assert_eq!(rt(&node, "odf").state, "started");
        outlier::hold::set(&node, "odf", false);
        node.flush_all();
        drop(node);
        registry().lock().remove(data.path());
        let node = Node::open(data.path()).unwrap();
        assert!(has_interrupted(&node));
        assert_eq!(rt(&node, "odf").state, "started", "a reassigned job is still started");
        assert_eq!(resume_interrupted(&node), vec!["odf".to_string()]);
        assert!(!has_interrupted(&node));
        // A second call (the next tick) does not launch it again.
        assert!(resume_interrupted(&node).is_empty());
        wait_worker(&node, "odf");
        check_results(&node, "odf");
    }

    /// A resume that cannot run yet (source gone) is retried, not dropped.
    #[test]
    fn resume_waits_for_missing_sources() {
        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        source(&node);
        put_od(&node, "odw");
        service(&node).mutate(|st| { st.rt.insert("odw".into(), Rt { state: "started".into(), ..Default::default() }); Ok(()) }).unwrap();
        node.delete_index("od-src").unwrap();
        service(&node).interrupted.lock().push(("odw".into(), 0));
        assert!(resume_interrupted(&node).is_empty());
        assert!(has_interrupted(&node), "kept for the next attempt");
        assert_eq!(rt(&node, "odw").state, "started");
        source(&node);
        assert_eq!(resume_interrupted(&node), vec!["odw".to_string()]);
        wait_worker(&node, "odw");
        check_results(&node, "odw");
    }

    /// Regression / classification are rejected at `_start` (after ES's own
    /// start checks) instead of acknowledging a job that never trains.
    #[test]
    fn supervised_analyses_are_rejected_at_start() {
        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        source(&node);
        for (id, analysis) in [("reg", json!({"regression": {"dependent_variable": "x"}})), ("cls", json!({"classification": {"dependent_variable": "label"}}))] {
            put(&node, id, &json!({"source": {"index": "od-src"}, "dest": {"index": format!("{id}-dest")}, "analysis": analysis})).unwrap();
            let e = start(&node, id, None).unwrap_err();
            assert_eq!(e.status, 400);
            assert_eq!(e.kind, "status_exception");
            assert!(e.reason.contains("analysis is not supported by this node"), "{}", e.reason);
            assert_eq!(service(&node).read().state_of(id), "stopped", "never started");
            assert!(node.get_index(&format!("{id}-dest")).is_none(), "no dest index created");
        }
    }

    /// #114: a large explicit `n_neighbors` is admitted against the job's
    /// `model_memory_limit` before the N x K neighbour lists are built: the
    /// job fails with ml-cpp's hard-limit error and writes no results,
    /// while the same job under a sufficient limit completes.
    #[test]
    fn outlier_neighbour_lists_are_admitted_against_model_memory_limit() {
        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        source(&node);
        let job = |id: &str, mml: &str| {
            put(&node, id, &json!({"source": {"index": "od-src"}, "dest": {"index": format!("{id}-dest")},
                "analysis": {"outlier_detection": {"n_neighbors": 40}}, "analyzed_fields": {"includes": ["x", "y"]}, "model_memory_limit": mml})).unwrap();
        };
        // 41 rows x 40 neighbours: about 36kb, far over a 1kb limit.
        let need = outlier::estimate_for(&json!({"n_neighbors": 40}), 41, 2);
        assert!(need >= 16 * 41 * 40 && need > 1024 && need < 1 << 20, "{need}");
        job("od-small", "1kb");
        start(&node, "od-small", None).expect("ES only warns at start");
        wait_worker(&node, "od-small");
        let r = rt(&node, "od-small");
        assert_eq!(r.state, "failed");
        let reason = r.failure_reason.clone().unwrap();
        assert!(reason.starts_with("Input error: memory limit [1 kb] has been exceeded. Please force stop the job, increase to new estimated limit ["), "{reason}");
        assert_ne!(r.progress.get("computing_outliers"), Some(&100));
        assert!(dest_docs(&node, "od-small-dest").values().all(|d| d.get("ml").is_none()), "no results were written");
        // Enough memory: the same analysis runs to completion.
        job("od-big", "10mb");
        start(&node, "od-big", None).unwrap();
        wait_worker(&node, "od-big");
        let r = rt(&node, "od-big");
        assert_eq!(r.state, "stopped", "{:?}", r.failure_reason);
        assert_eq!(r.data_counts["training_docs_count"], json!(41));
        let docs = dest_docs(&node, "od-big-dest");
        assert!(docs["far"]["ml"]["outlier_score"].as_f64().unwrap() > 0.5);
    }

    /// #143: features are read as typed values: a runtime override of a
    /// source field and a runtime-only feature are what is analysed.
    #[test]
    fn outlier_detection_analyses_runtime_feature_values() {
        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        source(&node);
        let rm = json!({
            // `x` is overridden (it emits `y`, 0 when absent): `far` (x = 80
            // in its source, y = 2) moves into the grid...
            "x": {"type": "double", "script": {"source": "if (doc['y'].size() == 0) { emit(0.0) } else { emit(doc['y'].value) }"}},
            // ...and `partial` (no `y`) is the one far out on a runtime-only feature.
            "z": {"type": "double", "script": {"source": "if (doc['y'].size() == 0) { emit(1000.0) } else { emit(doc['y'].value) }"}}
        });
        put(&node, "odrt", &json!({"source": {"index": "od-src", "runtime_mappings": rm}, "dest": {"index": "odrt-dest"},
            "analysis": {"outlier_detection": {}}, "analyzed_fields": {"includes": ["x", "z"]}})).unwrap();
        start(&node, "odrt", None).unwrap();
        wait_worker(&node, "odrt");
        let r = rt(&node, "odrt");
        assert_eq!(r.state, "stopped", "{:?}", r.failure_reason);
        assert_eq!(r.data_counts["training_docs_count"], json!(42), "every document has both runtime features");
        assert_eq!(r.data_counts["skipped_docs_count"], json!(0));
        let docs = dest_docs(&node, "odrt-dest");
        let score = |d: &str| docs[d]["ml"]["outlier_score"].as_f64().unwrap();
        assert!(score("partial") > 0.9, "{}", score("partial"));
        assert!(score("far") < 0.5, "the runtime x, not the source x, was analysed: {}", score("far"));
        let fi = docs["partial"]["ml"]["feature_influence"].as_array().unwrap();
        let z = fi.iter().find(|f| f["feature_name"] == "z").unwrap()["influence"].as_f64().unwrap();
        assert!(z > 0.9, "{fi:?}");
        assert_eq!(docs["far"]["x"], 80.0, "the dest copy keeps the source");
    }
}
