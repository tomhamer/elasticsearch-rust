//! Data frame analytics configs (`_ml/data_frame/analytics`): CRUD,
//! `_explain`, `_preview`, start/stop state, stats.
//!
//! ES keeps DFA configs in `.ml-config` and their runtime state in a
//! persistent task. Here both live in `<data>/_state/ml_dfa.json`.
//!
//! Starting an analytics job runs every check ES runs at start (source /
//! dest validation, field detection, data presence, empty dest index),
//! creates the dest index and marks the job `started`.

pub mod config;
pub mod fields;

use crate::error::{EsError, Result};
use crate::node::{Node, NODE_ID};
use config::{analysis_kind, conflict, status, Mode};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

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
    fn state_of(&self, id: &str) -> &str {
        self.rt.get(id).map(|r| r.state.as_str()).filter(|s| !s.is_empty()).unwrap_or("stopped")
    }
}

struct Service {
    path: PathBuf,
    st: Mutex<State>,
}

fn registry() -> &'static Mutex<HashMap<PathBuf, Arc<Service>>> {
    static R: OnceLock<Mutex<HashMap<PathBuf, Arc<Service>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn service(node: &Node) -> Arc<Service> {
    registry()
        .lock()
        .entry(node.data.clone())
        .or_insert_with(|| {
            let path = node.data.join("_state").join("ml_dfa.json");
            let st: State = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
            Arc::new(Service { path, st: Mutex::new(st) })
        })
        .clone()
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
        st.configs.remove(&target);
        st.rt.remove(&target);
        Ok(())
    })?;
    Ok(json!({"acknowledged": true}))
}

// ---- start / stop ----

/// Called once the job passed every start check, its dest index exists and
/// its state is `started`.
fn launch_training(_node: &Node, _id: &str, _cfg: &Value) {}

/// Record progress / counts reported by the analytics process.
pub fn report_progress(node: &Node, id: &str, f: impl FnOnce(&mut Rt)) -> Result<()> {
    service(node).mutate(|st| {
        if st.configs.contains_key(id) {
            f(st.rt.entry(id.to_string()).or_default());
        }
        Ok(())
    })
}

/// Create the dest index from the source mappings (ES `DestinationIndex`).
fn create_dest(node: &Node, id: &str, cfg: &Value) -> Result<()> {
    let dest = cfg["dest"]["index"].as_str().unwrap_or("");
    if dest.is_empty() || node.get_index(dest).is_some() || node.all_indices().iter().any(|s| s.aliases().contains_key(dest)) {
        return Ok(());
    }
    let mut props = fields::source_properties(node, cfg);
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
    if !results.is_empty() {
        props.insert(results.to_string(), json!({"properties": res}));
    }
    props.insert("ml__incremental_id".into(), json!({"type": "long"}));
    let body = json!({
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
pub fn start(node: &Node, id: &str, body_id: Option<&str>) -> Result<Value> {
    if let Some(b) = body_id {
        if b != id {
            return Err(EsError::illegal_argument(format!("Inconsistent id; '{b}' specified in the body differs from '{id}' specified as a URL argument")));
        }
    }
    check_upgrade_mode(node, "start")?;
    let svc = service(node);
    let (cfg, state) = {
        let st = svc.read();
        let cfg = st.configs.get(id).cloned().ok_or_else(|| not_found(id))?;
        (cfg, st.state_of(id).to_string())
    };
    if state != "stopped" && state != "failed" {
        return Err(conflict(format!("Cannot start data frame analytics [{id}] because it has already been started"))
            .caused_by(EsError::new(400, "resource_already_exists_exception", format!("task with id {{data_frame_analytics-{id}}} already exist"))));
    }
    fields::validate_source_dest(node, &cfg)?;
    let sel = fields::detect(node, &cfg)?;
    fields::check_has_features(&sel)?;
    if fields::usable_docs(node, &cfg, &sel)? == 0 {
        return Err(status(format!("Unable to start {id} as no documents in the source indices {} {}", fields::source_list(&cfg), fields::NO_DATA_SUFFIX)));
    }
    let dest = cfg["dest"]["index"].as_str().unwrap_or("").to_string();
    if node.get_index(&dest).is_some() || node.all_indices().iter().any(|s| s.aliases().contains_key(&dest)) {
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
        *rt = Rt { state: "started".into(), start_time: Some(crate::date::now_millis()), ..Default::default() };
        Ok(())
    })?;
    launch_training(node, id, &cfg);
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

/// Number of configured analytics jobs (ES `_xpack/usage`
/// `data_frame_analytics_jobs._all.count`).
pub fn count(node: &Node) -> usize {
    service(node).read().configs.len()
}
