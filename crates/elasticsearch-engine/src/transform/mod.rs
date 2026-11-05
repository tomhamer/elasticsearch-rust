//! Transforms (`_transform`): pivot and latest, batch and continuous.
//!
//! Configs and checkpoint state live in `<data>/_state/transforms.json`
//! (ES keeps them in `.transform-internal-*`). A started transform is
//! driven by a per-node scheduler thread (holding only a `Weak<Node>`,
//! exiting when nothing is started): each trigger runs a checkpoint on a
//! worker thread, paging the source with a composite aggregation
//! (`max_page_search_size` buckets per page), converting buckets into
//! documents and bulk-indexing them (through `dest.pipeline`) into the
//! destination index, which `_start` creates with deduced mappings.
//! Continuous transforms (`sync.time`) checkpoint up to `now - delay`,
//! only when the source changed, and recompute just the changed buckets
//! (terms keys / date histogram range of the new data).

pub mod config;
pub mod pivot;

use crate::error::{EsError, Result};
use crate::index::WriteReq;
use crate::node::{Node, ES_VERSION, NODE_ID};
use crate::shard::WriteCond;
use config::{Config, Function};
use parking_lot::{Condvar, Mutex};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::{Duration, Instant};

/// Preview size (ES `PreviewTransformAction` default).
const PREVIEW_SIZE: usize = 100;
/// Delay before a batch transform's first trigger (the task assignment
/// latency of ES: `_start` returns before the run begins).
const BATCH_START_DELAY_MS: i64 = 250;

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub pages_processed: u64,
    pub documents_processed: u64,
    pub documents_indexed: u64,
    pub documents_deleted: u64,
    pub trigger_count: u64,
    pub index_time_in_ms: u64,
    pub index_total: u64,
    pub index_failures: u64,
    pub search_time_in_ms: u64,
    pub search_total: u64,
    pub search_failures: u64,
    pub processing_time_in_ms: u64,
    pub processing_total: u64,
    pub delete_time_in_ms: u64,
    pub exp_avg_duration: f64,
    pub exp_avg_indexed: f64,
    pub exp_avg_processed: f64,
}

impl Stats {
    fn to_json(&self) -> Value {
        json!({
            "pages_processed": self.pages_processed,
            "documents_processed": self.documents_processed,
            "documents_indexed": self.documents_indexed,
            "documents_deleted": self.documents_deleted,
            "trigger_count": self.trigger_count,
            "index_time_in_ms": self.index_time_in_ms,
            "index_total": self.index_total,
            "index_failures": self.index_failures,
            "search_time_in_ms": self.search_time_in_ms,
            "search_total": self.search_total,
            "search_failures": self.search_failures,
            "processing_time_in_ms": self.processing_time_in_ms,
            "processing_total": self.processing_total,
            "delete_time_in_ms": self.delete_time_in_ms,
            "exponential_avg_checkpoint_duration_ms": self.exp_avg_duration,
            "exponential_avg_documents_indexed": self.exp_avg_indexed,
            "exponential_avg_documents_processed": self.exp_avg_processed,
        })
    }
    fn from_json(v: &Value) -> Stats {
        let u = |k: &str| v.get(k).and_then(Value::as_u64).unwrap_or(0);
        let f = |k: &str| v.get(k).and_then(Value::as_f64).unwrap_or(0.0);
        Stats {
            pages_processed: u("pages_processed"),
            documents_processed: u("documents_processed"),
            documents_indexed: u("documents_indexed"),
            documents_deleted: u("documents_deleted"),
            trigger_count: u("trigger_count"),
            index_time_in_ms: u("index_time_in_ms"),
            index_total: u("index_total"),
            index_failures: u("index_failures"),
            search_time_in_ms: u("search_time_in_ms"),
            search_total: u("search_total"),
            search_failures: u("search_failures"),
            processing_time_in_ms: u("processing_time_in_ms"),
            processing_total: u("processing_total"),
            delete_time_in_ms: u("delete_time_in_ms"),
            exp_avg_duration: f("exponential_avg_checkpoint_duration_ms"),
            exp_avg_indexed: f("exponential_avg_documents_indexed"),
            exp_avg_processed: f("exponential_avg_documents_processed"),
        }
    }
}

/// Exponential moving average over ~10 checkpoints (ES `TransformIndexerStats`).
fn exp_avg(prev: f64, v: f64, first: bool) -> f64 {
    const ALPHA: f64 = 2.0 / 11.0;
    if first {
        v
    } else {
        ALPHA * v + (1.0 - ALPHA) * prev
    }
}

#[derive(Clone, Debug, Default)]
struct TState {
    /// The persistent task exists (started, maybe failed).
    started: bool,
    /// A checkpoint is executing on a worker.
    running: bool,
    stop_requested: bool,
    stop_at_checkpoint: bool,
    failed: Option<String>,
    checkpoint: u64,
    last_cp_millis: i64,
    last_upper: Option<i64>,
    /// Next trigger (epoch millis), when started.
    next_run: i64,
    /// Progress of the running checkpoint: (docs processed, total docs).
    progress: Option<(u64, u64)>,
    last_search_time: Option<i64>,
    changes_last_detected_at: Option<i64>,
    health_issue: Option<String>,
    stats: Stats,
}

impl TState {
    fn state_name(&self) -> &'static str {
        if self.failed.is_some() {
            "failed"
        } else if !self.started {
            "stopped"
        } else if self.running && self.stop_requested {
            "stopping"
        } else if self.running {
            "indexing"
        } else {
            "started"
        }
    }
}

struct Entry {
    config: Value,
    st: TState,
}

pub struct TransformService {
    path: PathBuf,
    node: Mutex<Weak<Node>>,
    map: Mutex<BTreeMap<String, Entry>>,
    cv: Condvar,
    scheduler_alive: AtomicBool,
    upgrade_mode: AtomicBool,
    /// Per started transform: the coordinator its checkpoints run through
    /// on a cluster (captured from `_start`).
    run_rest: Mutex<HashMap<String, RestCall>>,
}

/// A REST request through a cluster coordinator: `(method, uri, body)` →
/// response body (a `Value::String` body is sent as NDJSON).
pub type RestCall = Arc<dyn Fn(&str, &str, Option<Value>) -> Result<Value> + Send + Sync>;

/// Where a transform's index work runs: on a cluster through its
/// coordinator (the source is read from every shard, the destination is
/// created in the cluster metadata and filled through its primaries),
/// otherwise on the node's own engine.
#[derive(Clone, Default)]
pub struct Ops(pub Option<RestCall>);

/// `expr` for a REST path: index expressions keep `,` and `*`.
fn path_expr(expr: &str) -> String {
    let mut out = String::new();
    for b in expr.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'*' | b',' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// The NDJSON `_bulk` body indexing `docs` (`(index, id, source)`).
fn bulk_body(docs: &[(String, String, Value)]) -> String {
    let mut out = String::new();
    for (index, id, d) in docs {
        out.push_str(&json!({"index": {"_index": index, "_id": id}}).to_string());
        out.push('\n');
        out.push_str(&d.to_string());
        out.push('\n');
    }
    out
}

/// `(succeeded, failed)` items of a `_bulk` response.
fn bulk_counts(resp: &Value) -> (u64, u64) {
    let mut ok = 0;
    let mut failed = 0;
    for item in resp["items"].as_array().into_iter().flatten() {
        match item.as_object().and_then(|o| o.values().next()) {
            Some(r) if r.get("error").is_none() => ok += 1,
            _ => failed += 1,
        }
    }
    (ok, failed)
}

impl Ops {
    /// One search over `expr` (`targets`: its local resolution).
    fn search(&self, node: &Node, expr: &str, targets: &[crate::search::Target], body: &Value) -> Result<Value> {
        match &self.0 {
            Some(rest) => rest("POST", &format!("/{}/_search?ignore_unavailable=true&allow_no_indices=true", path_expr(expr)), Some(body.clone())),
            None => pivot::search(node, targets, body),
        }
    }

    fn count(&self, expr: &str, targets: &[crate::search::Target], query: &Value) -> Result<u64> {
        match &self.0 {
            Some(rest) => {
                let r = rest("POST", &format!("/{}/_count?ignore_unavailable=true&allow_no_indices=true", path_expr(expr)), Some(json!({"query": query})))?;
                Ok(r["count"].as_u64().unwrap_or(0))
            }
            None => crate::search::count(targets, Some(query), None),
        }
    }

    fn create_index(&self, node: &Node, name: &str, body: &Value) -> Result<()> {
        match &self.0 {
            Some(rest) => rest("PUT", &format!("/{}", path_expr(name)), Some(body.clone())).map(|_| ()),
            None => node.create_index(name, body).map(|_| ()),
        }
    }

    fn delete_index(&self, node: &Node, name: &str) -> Result<()> {
        match &self.0 {
            Some(rest) => rest("DELETE", &format!("/{}", path_expr(name)), None).map(|_| ()),
            None => node.delete_index(name),
        }
    }

    fn refresh(&self, node: &Node, expr: &str) -> Result<()> {
        match &self.0 {
            Some(rest) => rest("POST", &format!("/{}/_refresh?ignore_unavailable=true&allow_no_indices=true", path_expr(expr)), None).map(|_| ()),
            None => node.refresh_indices(expr),
        }
    }
}

fn registry() -> &'static Mutex<HashMap<PathBuf, Arc<TransformService>>> {
    static R: OnceLock<Mutex<HashMap<PathBuf, Arc<TransformService>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The node's transform service (opened on first use).
pub fn service(node: &Arc<Node>) -> Arc<TransformService> {
    let mut r = registry().lock();
    let svc = r.entry(node.data.clone()).or_insert_with(|| Arc::new(TransformService::open(&node.data))).clone();
    drop(r);
    *svc.node.lock() = Arc::downgrade(node);
    if svc.map.lock().values().any(|e| e.st.started) {
        svc.ensure_scheduler();
    }
    svc
}

/// The service without binding the scheduler to the node (read-only uses
/// such as `_cat/transforms`, which only see `&Node`).
pub fn service_ref(node: &Node) -> Arc<TransformService> {
    let mut r = registry().lock();
    r.entry(node.data.clone()).or_insert_with(|| Arc::new(TransformService::open(&node.data))).clone()
}

/// Compile the scripts of `scripted_metric` aggregations (ES fails the
/// validation query with a 400 when one does not compile).
fn check_scripts(aggs: &Map<String, Value>) -> Result<()> {
    for def in aggs.values() {
        if let Some(sm) = def.get("scripted_metric").and_then(Value::as_object) {
            for k in ["init_script", "map_script", "combine_script", "reduce_script"] {
                let src = match sm.get(k) {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Object(o)) => o.get("source").and_then(Value::as_str).unwrap_or("").to_string(),
                    _ => continue,
                };
                if src.trim().is_empty() {
                    continue;
                }
                if let Err(e) = crate::script::compile(&src, &json!({}), crate::script::ScriptContext::Aggs) {
                    if e.status < 500 && !e.reason.contains("Variable [state]") && !e.reason.contains("Variable [states]") {
                        return Err(EsError::new(400, "status_exception", "Failed to test query, received status: BAD_REQUEST").caused_by(e));
                    }
                }
            }
        }
        if let Some(sub) = def.get("aggs").or_else(|| def.get("aggregations")).and_then(Value::as_object) {
            check_scripts(sub)?;
        }
    }
    Ok(())
}

fn not_found(id: &str) -> EsError {
    EsError::resource_not_found(format!("Transform with id [{id}] could not be found"))
}

fn status_err(status: u16, msg: impl Into<String>) -> EsError {
    EsError::new(status, "status_exception", msg)
}

fn matches_expr(expr: &str, id: &str) -> bool {
    expr.split(',').map(str::trim).any(|p| p == "_all" || p == "*" || crate::mapping::wildcard_match(p, id))
}

fn is_pattern(expr: &str) -> bool {
    expr.split(',').any(|p| p == "_all" || p.contains('*'))
}

impl TransformService {
    fn open(data: &Path) -> TransformService {
        let path = data.join("_state").join("transforms.json");
        let mut map = BTreeMap::new();
        if let Ok(b) = std::fs::read(&path) {
            if let Ok(Value::Object(m)) = serde_json::from_slice::<Value>(&b) {
                for (id, e) in m {
                    let s = &e["state"];
                    let st = TState {
                        started: s["started"].as_bool().unwrap_or(false),
                        checkpoint: s["checkpoint"].as_u64().unwrap_or(0),
                        last_cp_millis: s["last_cp_millis"].as_i64().unwrap_or(0),
                        last_upper: s["last_upper"].as_i64(),
                        last_search_time: s["last_search_time"].as_i64(),
                        changes_last_detected_at: s["changes_last_detected_at"].as_i64(),
                        stats: Stats::from_json(&s["stats"]),
                        next_run: crate::date::now_millis(),
                        ..Default::default()
                    };
                    map.insert(id, Entry { config: e["config"].clone(), st });
                }
            }
        }
        TransformService { path, node: Mutex::new(Weak::new()), map: Mutex::new(map), cv: Condvar::new(), scheduler_alive: AtomicBool::new(false), upgrade_mode: AtomicBool::new(false), run_rest: Mutex::new(HashMap::new()) }
    }

    fn persist(&self, map: &BTreeMap<String, Entry>) -> Result<()> {
        let mut out = Map::new();
        for (id, e) in map {
            let s = &e.st;
            out.insert(
                id.clone(),
                json!({
                    "config": e.config,
                    "state": {
                        "started": s.started && s.failed.is_none(),
                        "checkpoint": s.checkpoint,
                        "last_cp_millis": s.last_cp_millis,
                        "last_upper": s.last_upper,
                        "last_search_time": s.last_search_time,
                        "changes_last_detected_at": s.changes_last_detected_at,
                        "stats": s.stats.to_json(),
                    }
                }),
            );
        }
        if let Some(d) = self.path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(&Value::Object(out))?)?;
        std::fs::rename(tmp, &self.path)?;
        Ok(())
    }

    fn config_of(&self, id: &str) -> Result<(Value, Config)> {
        let raw = self.map.lock().get(id).map(|e| e.config.clone()).ok_or_else(|| not_found(id))?;
        let cfg = config::parse(Some(id), &config::strip_generated(&raw), false, true)?;
        Ok((raw, cfg))
    }

    /// Ids matching an expression; a concrete id that does not exist (or
    /// an expression matching nothing with `allow_no_match=false`) is a 404.
    fn expand(&self, expr: &str, allow_no_match: bool) -> Result<Vec<String>> {
        let map = self.map.lock();
        let mut out = Vec::new();
        for part in expr.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            if part == "_all" || part == "*" || part.contains('*') {
                let found: Vec<String> = map.keys().filter(|k| crate::mapping::wildcard_match(if part == "_all" { "*" } else { part }, k)).cloned().collect();
                if found.is_empty() && !allow_no_match {
                    return Err(not_found(part));
                }
                out.extend(found);
            } else if map.contains_key(part) {
                out.push(part.to_string());
            } else {
                return Err(not_found(part));
            }
        }
        out.sort();
        out.dedup();
        Ok(out)
    }

    // ------------------------------------------------------------ validation

    /// Source/dest/pipeline checks (skipped with `defer_validation`).
    fn validate_env(&self, node: &Node, ops: &Ops, cfg: &Config, check_dest: bool) -> Result<()> {
        let mut errs: Vec<String> = Vec::new();
        let targets = match pivot::source_targets(node, cfg, false) {
            Ok(t) => t,
            Err(e) if e.kind == "index_not_found_exception" => {
                errs.push(e.reason.clone());
                Vec::new()
            }
            Err(e) => return Err(e),
        };
        if check_dest {
            if let Some(d) = &cfg.dest_index {
                if let Some(m) = config::validate_dest_name(d) {
                    errs.push(m);
                } else if errs.is_empty() {
                    let mut srcs: Vec<String> = targets.iter().map(|t| t.svc.name.clone()).collect();
                    srcs.sort();
                    srcs.dedup();
                    match node.write_index(d, false) {
                        Ok(svc) => {
                            if srcs.contains(&svc.name) {
                                errs.push(format!("Destination index [{}] is included in source expression [{}]", svc.name, srcs.join(",")));
                            }
                        }
                        Err(e) if e.kind == "index_not_found_exception" => {
                            if cfg.source_indices.iter().any(|p| crate::mapping::wildcard_match(p, d)) {
                                errs.push(format!("Destination index [{d}] is included in source expression [{}]", cfg.source_indices.join(",")));
                            }
                        }
                        Err(e) => errs.push(e.reason.clone()),
                    }
                }
            }
        }
        if !errs.is_empty() {
            let refs: Vec<&str> = errs.iter().map(String::as_str).collect();
            return Err(EsError::action_validation(&refs));
        }
        self.check_pipeline(node, cfg)?;
        if let Some(rm) = cfg.runtime_mappings.as_ref().and_then(Value::as_object) {
            for (name, def) in rm {
                let t = def.get("type").and_then(Value::as_str).unwrap_or("");
                if !matches!(t, "long" | "double" | "keyword" | "boolean" | "date" | "ip" | "geo_point" | "composite" | "lookup") {
                    return Err(EsError::new(400, "mapper_parsing_exception", format!("The mapper type [{t}] declared on runtime field [{name}] does not exist. It might have been created within a future version or requires a plugin to be installed. Check the documentation.")));
                }
            }
        }
        if let Function::Pivot { aggs, .. } = &cfg.function {
            check_scripts(aggs)?;
        }
        let types = pivot::field_types(cfg, &targets);
        pivot::check_latest_keys(cfg, &types)?;
        // A test query: the first bucket.
        if !targets.is_empty() {
            let body = pivot::page_body(cfg, &cfg.query, 1, None);
            ops.search(node, &cfg.source_indices.join(","), &targets, &body).map_err(|e| {
                if e.status >= 500 {
                    e
                } else {
                    EsError::new(400, "status_exception", format!("Failed to test query, received status: BAD_REQUEST")).caused_by(e)
                }
            })?;
        }
        Ok(())
    }

    fn check_pipeline(&self, node: &Node, cfg: &Config) -> Result<()> {
        if let Some(p) = &cfg.dest_pipeline {
            if !node.ingest().exists(p) {
                return Err(status_err(400, format!("Pipeline with id [{p}] could not be found")));
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------ CRUD

    /// `PUT /_transform/{id}`; returns deprecation-style warnings.
    pub fn put(&self, node: &Node, ops: &Ops, id: &str, body: &Value, defer: bool) -> Result<Vec<String>> {
        config::validate_id(id)?;
        let cfg = config::parse(Some(id), body, true, true)?;
        if self.map.lock().contains_key(id) {
            return Err(EsError::new(409, "resource_already_exists_exception", format!("Transform with id [{id}] already exists")));
        }
        if !defer {
            self.validate_env(node, ops, &cfg, true)?;
        } else if let Some(d) = &cfg.dest_index {
            if let Some(m) = config::validate_dest_name(d) {
                return Err(EsError::action_validation(&[&m]));
            }
        }
        let stored = config::normalize(body, &cfg, ES_VERSION, Some(crate::date::now_millis()));
        let mut map = self.map.lock();
        map.insert(id.to_string(), Entry { config: stored, st: TState::default() });
        self.persist(&map)?;
        Ok(config::runtime_warnings(&cfg))
    }

    /// `GET /_transform[/{id}]`.
    pub fn get(&self, expr: &str, from: usize, size: usize, allow_no_match: bool, exclude_generated: bool) -> Result<Value> {
        if size > 1000 {
            return Err(EsError::action_validation(&["Param [size] has a max acceptable value of [1000]"]));
        }
        let ids = self.expand(expr, allow_no_match)?;
        let map = self.map.lock();
        let mut out = Vec::new();
        for id in ids.iter().skip(from).take(size) {
            if let Some(e) = map.get(id) {
                let mut c = e.config.clone();
                if exclude_generated {
                    if let Some(o) = c.as_object_mut() {
                        for k in ["version", "create_time", "authorization"] {
                            o.remove(k);
                        }
                    }
                }
                out.push(c);
            }
        }
        Ok(json!({"count": ids.len(), "transforms": out}))
    }

    /// `POST /_features/_reset`: stop and delete every transform.
    pub fn reset_all(&self) -> Result<()> {
        let ids: Vec<String> = self.map.lock().keys().cloned().collect();
        for id in &ids {
            self.stop_one(id, true, false, true, Duration::from_secs(30));
        }
        let mut map = self.map.lock();
        map.clear();
        self.persist(&map)
    }

    /// `DELETE /_transform/{id}`.
    pub fn delete(&self, node: &Node, ops: &Ops, id: &str, force: bool, delete_dest: bool) -> Result<()> {
        let dest = {
            let map = self.map.lock();
            let e = map.get(id).ok_or_else(|| not_found(id))?;
            if e.st.started && !force {
                return Err(status_err(409, format!("Cannot delete transform [{id}] as the task is running. Stop the task first")));
            }
            e.config["dest"]["index"].as_str().map(str::to_string)
        };
        if force {
            self.stop_one(id, true, false, true, Duration::from_secs(30));
        }
        if delete_dest {
            if let Some(d) = dest {
                if node.get_index(&d).is_some() {
                    ops.delete_index(node, &d)?;
                }
            }
        }
        self.run_rest.lock().remove(id);
        let mut map = self.map.lock();
        map.remove(id);
        self.persist(&map)?;
        Ok(())
    }

    /// `POST /_transform/{id}/_update`; returns the updated config.
    pub fn update(&self, node: &Node, ops: &Ops, id: &str, body: &Value, defer: bool) -> Result<(Value, Vec<String>)> {
        let (raw, old) = self.config_of(id)?;
        let upd = body.as_object().ok_or_else(|| EsError::parsing("request body is required"))?;
        for k in upd.keys() {
            if !matches!(k.as_str(), "source" | "dest" | "frequency" | "sync" | "description" | "settings" | "_meta" | "retention_policy") {
                return Err(EsError::x_content_parse(format!("[1:1] [data_frame_transform_config_update] unknown field [{k}]")));
            }
        }
        let mut merged = config::strip_generated(&raw);
        let mo = merged.as_object_mut().unwrap();
        for (k, v) in upd {
            if k == "settings" {
                let mut s = mo.get("settings").cloned().filter(Value::is_object).unwrap_or_else(|| json!({}));
                if let (Some(so), Some(vo)) = (s.as_object_mut(), v.as_object()) {
                    for (sk, sv) in vo {
                        if sv.is_null() {
                            so.remove(sk);
                        } else {
                            so.insert(sk.clone(), sv.clone());
                        }
                    }
                }
                mo.insert(k.clone(), s);
            } else if v.is_null() {
                mo.remove(k);
            } else {
                mo.insert(k.clone(), v.clone());
            }
        }
        let cfg = config::parse(Some(id), &merged, false, true)?;
        if old.sync.is_none() != cfg.sync.is_none() {
            let (from, to) = (if old.sync.is_some() { "time" } else { "null" }, if cfg.sync.is_some() { "time" } else { "null" });
            return Err(EsError::new(409, "status_exception", format!("Cannot change the current sync configuration of transform [{id}] from [{from}] to [{to}]")));
        }
        if !defer {
            let check_dest = upd.contains_key("dest") || upd.contains_key("source");
            self.validate_env(node, ops, &cfg, check_dest)?;
        } else if let Some(d) = &cfg.dest_index {
            if let Some(m) = config::validate_dest_name(d) {
                return Err(EsError::action_validation(&[&m]));
            }
        }
        let create_time = raw.get("create_time").and_then(Value::as_i64);
        let stored = config::normalize(&merged, &cfg, ES_VERSION, create_time);
        let mut map = self.map.lock();
        if let Some(e) = map.get_mut(id) {
            e.config = stored.clone();
        }
        self.persist(&map)?;
        Ok((stored, config::runtime_warnings(&cfg)))
    }

    /// `POST /_transform/_upgrade`.
    pub fn upgrade(&self) -> Value {
        json!({"updated": 0, "no_action": self.map.lock().len(), "needs_update": 0})
    }

    pub fn set_upgrade_mode(&self, enabled: bool) {
        self.upgrade_mode.store(enabled, Ordering::Relaxed);
    }

    // ------------------------------------------------------------ preview

    /// `POST /_transform[/{id}]/_preview`.
    pub fn preview(&self, node: &Node, ops: &Ops, id: Option<&str>, body: &Value) -> Result<(Value, Vec<String>)> {
        // An empty object is a (incomplete) config object: only an absent body is none.
        let has_body = !body.is_null();
        let cfg = match (id, has_body) {
            (None, false) => return Err(status_err(400, "Please provide a transform [id] or the config object")),
            (Some(_), true) => return Err(status_err(400, "Please provide either a transform [id] or the config object but not both")),
            (Some(id), false) => self.config_of(id)?.1,
            (None, true) => config::parse(None, body, false, false)?,
        };
        let warnings = config::runtime_warnings(&cfg);
        let targets = match pivot::source_targets(node, &cfg, false) {
            Ok(t) => t,
            Err(e) if e.kind == "index_not_found_exception" => return Err(EsError::action_validation(&[&e.reason])),
            Err(e) => return Err(e),
        };
        self.check_pipeline(node, &cfg)?;
        let types = pivot::field_types(&cfg, &targets);
        pivot::check_latest_keys(&cfg, &types)?;
        let mut docs: Vec<Value> = Vec::new();
        if !targets.is_empty() {
            let q = if let Some((f, delay)) = &cfg.sync {
                let upper = crate::date::now_millis() - delay;
                json!({"bool": {"filter": [cfg.query.clone(), {"range": {f: {"lt": upper, "format": "epoch_millis"}}}]}})
            } else {
                cfg.query.clone()
            };
            let body = pivot::page_body(&cfg, &q, PREVIEW_SIZE, None);
            let resp = ops.search(node, &cfg.source_indices.join(","), &targets, &body)?;
            let page = pivot::convert_page(&cfg, &resp, &types)?;
            docs = page.docs.into_iter().map(|(_, d)| Value::Object(d)).collect();
        }
        if let Some(p) = &cfg.dest_pipeline {
            if !docs.is_empty() {
                let sim_docs: Vec<Value> = docs.iter().map(|d| json!({"_source": d})).collect();
                let out = node.simulate_pipeline(Some(p), &json!({"docs": sim_docs}), false)?;
                docs = out["docs"].as_array().cloned().unwrap_or_default().into_iter().filter(|d| !d.is_null()).map(|d| d["doc"]["_source"].clone()).collect();
            }
        }
        let generated = pivot::dest_index_body(&cfg, &types, ES_VERSION, true);
        Ok((json!({"preview": docs, "generated_dest_index": generated}), warnings))
    }

    // ------------------------------------------------------------ lifecycle

    /// `POST /_transform/{id}/_start`.
    pub fn start(self: &Arc<Self>, node: &Node, ops: &Ops, id: &str, from: Option<&str>) -> Result<()> {
        let (_, cfg) = self.config_of(id)?;
        if let Some(f) = from {
            if !cfg.is_continuous() {
                return Err(EsError::action_validation(&["[from] parameter is currently not supported for batch (non-continuous) transforms"]));
            }
            let fmt = crate::date::DateFormatter::default();
            if f.is_empty() || crate::date::parse_date_math(f, &fmt, crate::date::Tz::UTC, crate::date::now_millis(), false).is_err() {
                return Err(EsError::new(400, "parse_exception", "Failed to parse date for [from]"));
            }
        }
        {
            let map = self.map.lock();
            let e = map.get(id).ok_or_else(|| not_found(id))?;
            if e.st.started {
                return Err(status_err(409, format!("Cannot start transform [{id}] as it is already started.")));
            }
        }
        self.check_pipeline(node, &cfg)?;
        let unattended = cfg.settings.unattended;
        let targets = match pivot::source_targets(node, &cfg, unattended) {
            Ok(t) => t,
            Err(e) if e.kind == "index_not_found_exception" => return Err(EsError::action_validation(&[&e.reason])),
            Err(e) => return Err(e),
        };
        // Create the destination index (ES does this before the task starts).
        let dest = cfg.dest_index.clone().unwrap_or_default();
        if !dest.is_empty() {
            match node.write_index(&dest, false) {
                Ok(_) => {}
                Err(e) if e.kind == "index_not_found_exception" => {
                    let types = pivot::field_types(&cfg, &targets);
                    let body = pivot::dest_index_body(&cfg, &types, ES_VERSION, false);
                    match ops.create_index(node, &dest, &body) {
                        Ok(_) => {}
                        Err(e) if e.kind == "resource_already_exists_exception" => {}
                        Err(e) => {
                            // Retry without the generated settings (older engines may reject them).
                            let b2 = json!({"mappings": body["mappings"], "aliases": body["aliases"]});
                            ops.create_index(node, &dest, &b2).map_err(|_| e)?;
                        }
                    }
                }
                Err(e) => return Err(EsError::action_validation(&[&e.reason])),
            }
        }
        let continuous = cfg.is_continuous();
        match &ops.0 {
            Some(rest) => {
                self.run_rest.lock().insert(id.to_string(), rest.clone());
            }
            None => {
                self.run_rest.lock().remove(id);
            }
        }
        {
            let mut map = self.map.lock();
            let e = map.get_mut(id).ok_or_else(|| not_found(id))?;
            let now = crate::date::now_millis();
            e.st.started = true;
            e.st.failed = None;
            e.st.stop_requested = false;
            e.st.stop_at_checkpoint = false;
            e.st.health_issue = if targets.is_empty() && unattended { Some("source index missing".into()) } else { None };
            e.st.next_run = if continuous { now } else { now + BATCH_START_DELAY_MS };
            if let (Some(f), true) = (from, continuous) {
                if e.st.checkpoint == 0 {
                    e.st.last_upper = crate::date::parse_date_math(f, &crate::date::DateFormatter::default(), crate::date::Tz::UTC, now, false).ok();
                }
            }
            self.persist(&map)?;
        }
        self.ensure_scheduler();
        if continuous {
            // `_start` returns once the task is assigned; give the first
            // checkpoint a moment so stats reflect it.
            let deadline = Instant::now() + Duration::from_secs(2);
            let mut map = self.map.lock();
            loop {
                let done = map.get(id).map(|e| !e.st.started || e.st.checkpoint > 0 || e.st.failed.is_some()).unwrap_or(true);
                if done || Instant::now() >= deadline {
                    break;
                }
                self.cv.wait_for(&mut map, Duration::from_millis(50));
            }
        }
        Ok(())
    }

    /// Stop one transform; returns whether it was waited for.
    fn stop_one(&self, id: &str, force: bool, wait_for_checkpoint: bool, wait: bool, timeout: Duration) {
        {
            let mut map = self.map.lock();
            let Some(e) = map.get_mut(id) else { return };
            if !e.st.started {
                return;
            }
            if force || !e.st.running {
                e.st.started = false;
                e.st.failed = None;
            }
            e.st.stop_requested = true;
            e.st.stop_at_checkpoint = wait_for_checkpoint;
            let _ = self.persist(&map);
        }
        self.cv.notify_all();
        if wait {
            let deadline = Instant::now() + timeout;
            let mut map = self.map.lock();
            loop {
                let busy = map.get(id).map(|e| e.st.running || e.st.started).unwrap_or(false);
                if !busy || Instant::now() >= deadline {
                    break;
                }
                self.cv.wait_for(&mut map, Duration::from_millis(50));
            }
        }
    }

    /// `POST /_transform/{id}/_stop`.
    pub fn stop(&self, expr: &str, force: bool, wait_for_checkpoint: bool, wait: bool, allow_no_match: bool, timeout: Duration) -> Result<()> {
        if force && wait_for_checkpoint {
            return Err(EsError::action_validation(&["cannot set both [force] and [wait_for_checkpoint] to true"]));
        }
        let ids = self.expand(expr, allow_no_match)?;
        if !force {
            let map = self.map.lock();
            for id in &ids {
                if let Some(e) = map.get(id) {
                    if e.st.failed.is_some() {
                        return Err(status_err(409, format!("Unable to stop transform [{id}] as it is in a failed state. Use force stop to stop the transform. More details: [{}]", e.st.failed.clone().unwrap_or_default())));
                    }
                }
            }
        }
        for id in &ids {
            self.stop_one(id, force, wait_for_checkpoint, wait, timeout);
        }
        Ok(())
    }

    /// `POST /_transform/{id}/_reset`.
    pub fn reset(&self, node: &Node, ops: &Ops, id: &str, force: bool) -> Result<()> {
        let (_, cfg) = self.config_of(id)?;
        let started = self.map.lock().get(id).map(|e| e.st.started).unwrap_or(false);
        if started && !force {
            return Err(status_err(409, format!("Cannot reset transform [{id}] as the task is running. Stop the task first")));
        }
        if started {
            self.stop_one(id, true, false, true, Duration::from_secs(30));
        }
        if let Some(d) = &cfg.dest_index {
            if let Some(svc) = node.get_index(d) {
                let created_by_us = svc.meta().mappings.to_json().get("_meta").and_then(|m| m.get("created_by")).and_then(Value::as_str) == Some("transform");
                if created_by_us {
                    ops.delete_index(node, d)?;
                }
            }
        }
        let mut map = self.map.lock();
        if let Some(e) = map.get_mut(id) {
            e.st = TState::default();
        }
        self.persist(&map)?;
        Ok(())
    }

    /// `POST /_transform/{id}/_schedule_now`.
    pub fn schedule_now(&self, id: &str) -> Result<()> {
        if id == "_all" || id == "*" {
            return Err(EsError::action_validation(&["_schedule_now API does not support _all wildcard"]));
        }
        let mut map = self.map.lock();
        let e = map.get_mut(id).ok_or_else(|| not_found(id))?;
        if e.st.started {
            e.st.next_run = crate::date::now_millis();
        }
        drop(map);
        self.cv.notify_all();
        Ok(())
    }

    // ------------------------------------------------------------ stats

    fn stats_json(&self, node: &Node, id: &str, e: &Entry) -> Value {
        let st = &e.st;
        let mut m = Map::new();
        m.insert("id".into(), json!(id));
        m.insert("state".into(), json!(st.state_name()));
        if let Some(r) = &st.failed {
            m.insert("reason".into(), json!(r));
        }
        if st.started {
            m.insert(
                "node".into(),
                json!({"id": NODE_ID, "name": node.name, "ephemeral_id": NODE_ID, "transport_address": "127.0.0.1:9300", "attributes": {}}),
            );
        }
        m.insert("stats".into(), st.stats.to_json());
        let mut cp = Map::new();
        let mut last = Map::new();
        last.insert("checkpoint".into(), json!(st.checkpoint));
        if st.checkpoint > 0 {
            last.insert("timestamp_millis".into(), json!(st.last_cp_millis));
            if let Some(u) = st.last_upper {
                last.insert("time_upper_bound_millis".into(), json!(u));
            }
        }
        cp.insert("last".into(), Value::Object(last));
        if st.running {
            let mut next = Map::new();
            next.insert("checkpoint".into(), json!(st.checkpoint + 1));
            if let Some((done, total)) = st.progress {
                let pct = if total == 0 { 100.0 } else { (done as f64 * 100.0 / total as f64).min(100.0) };
                next.insert("checkpoint_progress".into(), json!({"total_docs": total, "docs_remaining": total.saturating_sub(done), "percent_complete": pct, "docs_indexed": st.stats.documents_indexed, "docs_processed": done}));
            }
            next.insert("timestamp_millis".into(), json!(crate::date::now_millis()));
            cp.insert("next".into(), Value::Object(next));
        }
        if e.config.get("sync").is_some() {
            cp.insert("operations_behind".into(), json!(0));
            if let Some(t) = st.changes_last_detected_at {
                cp.insert("changes_last_detected_at".into(), json!(t));
            }
            if let Some(t) = st.last_search_time {
                cp.insert("last_search_time".into(), json!(t));
            }
        }
        m.insert("checkpointing".into(), Value::Object(cp));
        let health = if let Some(r) = &st.failed {
            json!({"status": "red", "issues": [{"type": "transform_task_failed", "issue": "Transform task state is [failed]", "details": r, "count": 1}]})
        } else if let Some(i) = &st.health_issue {
            json!({"status": "yellow", "issues": [{"type": "transform_indexer_failed", "issue": "Transform indexer failed", "details": i, "count": 1}]})
        } else {
            json!({"status": "green"})
        };
        m.insert("health".into(), health);
        Value::Object(m)
    }

    /// `GET /_transform/{id}/_stats`.
    pub fn stats(&self, node: &Node, expr: &str, from: usize, size: usize, allow_no_match: bool) -> Result<Value> {
        if size > 1000 {
            return Err(EsError::action_validation(&["Param [size] has a max acceptable value of [1000]"]));
        }
        let ids: Vec<String> = {
            let map = self.map.lock();
            map.keys().filter(|k| matches_expr(expr, k)).cloned().collect()
        };
        if ids.is_empty() && !allow_no_match {
            return Err(not_found(expr));
        }
        let map = self.map.lock();
        let list: Vec<Value> = ids.iter().skip(from).take(size).filter_map(|id| map.get(id).map(|e| self.stats_json(node, id, e))).collect();
        Ok(json!({"count": ids.len(), "transforms": list}))
    }

    /// Rows for `_cat/transforms`: (config, stats) per matching transform.
    pub fn cat_rows(&self, node: &Node, expr: Option<&str>, allow_no_match: bool) -> Result<Vec<(Value, Value)>> {
        let expr = expr.unwrap_or("_all");
        let ids = self.expand(expr, allow_no_match || is_pattern(expr))?;
        let map = self.map.lock();
        Ok(ids.iter().filter_map(|id| map.get(id).map(|e| (e.config.clone(), self.stats_json(node, id, e)))).collect())
    }

    /// `GET /_transform/_node_stats`.
    pub fn node_stats(&self) -> Value {
        let n = self.map.lock().values().filter(|e| e.st.started).count();
        let peek = self.map.lock().iter().filter(|(_, e)| e.st.started).min_by_key(|(_, e)| e.st.next_run).map(|(k, _)| k.clone());
        json!({NODE_ID: {"scheduler": {"registered_transform_count": n, "peek_transform": peek}}, "total": {"scheduler": {"registered_transform_count": n}}})
    }

    // ------------------------------------------------------------ scheduler

    fn ensure_scheduler(self: &Arc<Self>) {
        if self.scheduler_alive.swap(true, Ordering::SeqCst) {
            self.cv.notify_all();
            return;
        }
        let svc = self.clone();
        let spawned = std::thread::Builder::new().name("transform-scheduler".into()).spawn(move || svc.scheduler_loop());
        if spawned.is_err() {
            self.scheduler_alive.store(false, Ordering::SeqCst);
        }
    }

    fn scheduler_loop(self: Arc<Self>) {
        loop {
            let node = self.node.lock().upgrade();
            let Some(node) = node else { break };
            let now = crate::date::now_millis();
            let mut due = Vec::new();
            {
                let mut map = self.map.lock();
                if !map.values().any(|e| e.st.started || e.st.running) {
                    self.scheduler_alive.store(false, Ordering::SeqCst);
                    break;
                }
                if !self.upgrade_mode.load(Ordering::Relaxed) {
                    for (id, e) in map.iter_mut() {
                        if e.st.started && !e.st.running && e.st.failed.is_none() && e.st.next_run <= now && !e.st.stop_requested {
                            e.st.running = true;
                            due.push(id.clone());
                        }
                    }
                }
                if due.is_empty() {
                    self.cv.wait_for(&mut map, Duration::from_millis(100));
                }
            }
            for id in due {
                let svc = self.clone();
                let n = node.clone();
                let r = std::thread::Builder::new().name(format!("transform-{id}")).spawn(move || svc.run_trigger(&n, &id));
                if r.is_err() {
                    break;
                }
            }
            drop(node);
        }
    }

    fn update_state<F: FnOnce(&mut TState)>(&self, id: &str, f: F) -> bool {
        let mut map = self.map.lock();
        match map.get_mut(id) {
            Some(e) => {
                f(&mut e.st);
                true
            }
            None => false,
        }
    }

    fn should_abort(&self, id: &str) -> bool {
        let map = self.map.lock();
        match map.get(id) {
            Some(e) => (e.st.stop_requested && !e.st.stop_at_checkpoint) || !e.st.started,
            None => true,
        }
    }

    /// One trigger: detect changes and maybe run a checkpoint.
    fn run_trigger(self: &Arc<Self>, node: &Node, id: &str) {
        let r = self.checkpoint(node, id);
        let mut map = self.map.lock();
        if let Some(e) = map.get_mut(id) {
            e.st.running = false;
            e.st.progress = None;
            match r {
                Ok(()) => {}
                Err(err) => {
                    e.st.stats.search_failures += 1;
                    let unattended = e.config["settings"]["unattended"].as_bool().unwrap_or(false);
                    if unattended && e.st.started {
                        e.st.health_issue = Some(err.reason.clone());
                    } else if e.st.started {
                        e.st.failed = Some(format!("task encountered irrecoverable failure: {}", err.reason));
                    }
                }
            }
            let freq = config::parse(Some(id), &config::strip_generated(&e.config), false, true).map(|c| c.frequency_ms).unwrap_or(config::DEFAULT_FREQUENCY_MS);
            e.st.next_run = crate::date::now_millis() + freq;
            if e.st.stop_requested {
                e.st.started = false;
                e.st.stop_requested = false;
                e.st.stop_at_checkpoint = false;
            }
            let _ = self.persist(&map);
        }
        drop(map);
        self.cv.notify_all();
    }

    fn checkpoint(&self, node: &Node, id: &str) -> Result<()> {
        let (_, cfg) = self.config_of(id)?;
        let ops = Ops(self.run_rest.lock().get(id).cloned());
        let src = cfg.source_indices.join(",");
        let started = Instant::now();
        let (prev_cp, prev_upper) = {
            let map = self.map.lock();
            let e = map.get(id).ok_or_else(|| not_found(id))?;
            (e.st.checkpoint, e.st.last_upper)
        };
        self.update_state(id, |s| s.stats.trigger_count += 1);
        let targets = pivot::source_targets(node, &cfg, cfg.settings.unattended)?;
        if targets.is_empty() && cfg.settings.unattended {
            return Err(EsError::index_not_found(&cfg.source_indices.join(",")));
        }
        let now = crate::date::now_millis();
        let types = pivot::field_types(&cfg, &targets);
        let mut query = cfg.query.clone();
        let mut upper = None;
        if let Some((field, delay)) = &cfg.sync {
            let mut u = now - delay;
            // align_checkpoints: round down to a fixed date_histogram interval on the sync field.
            if cfg.settings.align_checkpoints {
                if let Function::Pivot { groups, .. } = &cfg.function {
                    if let Some(g) = groups.iter().find(|g| g.kind == "date_histogram" && g.field() == Some(field.as_str())) {
                        if let Some(iv) = g.params.get("fixed_interval").and_then(Value::as_str).and_then(crate::date::parse_fixed_interval) {
                            u = u.div_euclid(iv) * iv;
                        }
                    }
                }
            }
            upper = Some(u);
            let mut filters = vec![cfg.query.clone(), json!({"range": {field.clone(): {"lt": u, "format": "epoch_millis"}}})];
            if prev_cp > 0 || prev_upper.is_some() {
                let lower = prev_upper.unwrap_or(0);
                // Change detection: any new data in [lower, upper)?
                let probe = json!({"bool": {"filter": [cfg.query.clone(), {"range": {field.clone(): {"gte": lower, "lt": u, "format": "epoch_millis"}}}]}});
                let t0 = Instant::now();
                let changed_filter = self.changed_buckets(node, &ops, &cfg, &targets, &probe, lower)?;
                self.update_state(id, |s| {
                    s.stats.search_total += 1;
                    s.stats.search_time_in_ms += t0.elapsed().as_millis() as u64;
                    s.last_search_time = Some(now);
                });
                match changed_filter {
                    None => {
                        if prev_cp > 0 {
                            return Ok(());
                        }
                    }
                    Some(f) => {
                        self.update_state(id, |s| s.changes_last_detected_at = Some(now));
                        if prev_cp > 0 {
                            filters.extend(f);
                        }
                    }
                }
            } else {
                self.update_state(id, |s| s.last_search_time = Some(now));
            }
            query = json!({"bool": {"filter": filters}});
        }
        // Progress total.
        let total = if targets.is_empty() { 0 } else { ops.count(&src, &targets, &query).unwrap_or(0) };
        self.update_state(id, |s| s.progress = Some((0, total)));
        let dest = cfg.dest_index.clone().unwrap_or_default();
        let mut after: Option<Value> = None;
        let (mut cp_indexed, mut cp_processed) = (0u64, 0u64);
        if !targets.is_empty() {
            loop {
                if self.should_abort(id) {
                    return Ok(());
                }
                let body = pivot::page_body(&cfg, &query, cfg.page_size(), after.as_ref());
                let t0 = Instant::now();
                let resp = ops.search(node, &src, &targets, &body)?;
                let st_ms = t0.elapsed().as_millis() as u64;
                let t1 = Instant::now();
                let page = pivot::convert_page(&cfg, &resp, &types)?;
                let proc_ms = t1.elapsed().as_millis() as u64;
                let n = page.docs.len() as u64;
                let t2 = Instant::now();
                let (ok, failed) = if n > 0 { self.index_docs(node, &ops, &cfg, &dest, page.docs)? } else { (0, 0) };
                let idx_ms = t2.elapsed().as_millis() as u64;
                cp_indexed += ok;
                cp_processed += page.docs_processed;
                let dp = page.docs_processed;
                self.update_state(id, |s| {
                    s.stats.search_total += 1;
                    s.stats.search_time_in_ms += st_ms;
                    s.stats.processing_total += 1;
                    s.stats.processing_time_in_ms += proc_ms;
                    s.stats.pages_processed += 1;
                    s.stats.documents_processed += dp;
                    if n > 0 {
                        s.stats.index_total += 1;
                        s.stats.index_time_in_ms += idx_ms;
                        s.stats.documents_indexed += ok;
                        s.stats.index_failures += failed;
                    }
                    s.progress = Some((cp_processed, total));
                });
                if failed > 0 {
                    return Err(EsError::new(500, "bulk_index_exception", format!("Bulk index experienced [{failed}] failures")));
                }
                match page.after_key {
                    Some(a) if n as usize >= cfg.page_size() => after = Some(a),
                    _ => break,
                }
            }
        }
        // ES refreshes the destination at the end of every checkpoint
        // (`TransformIndexer.onFinish`), before the retention policy runs.
        if !dest.is_empty() {
            ops.refresh(node, &dest)?;
        }
        // Retention policy.
        if let Some((field, max_age)) = &cfg.retention {
            let t0 = Instant::now();
            let deleted = self.apply_retention(node, &ops, &dest, field, now - max_age)?;
            self.update_state(id, |s| {
                s.stats.documents_deleted += deleted;
                s.stats.delete_time_in_ms += t0.elapsed().as_millis() as u64;
            });
        }
        let dur = started.elapsed().as_secs_f64() * 1000.0;
        let continuous = cfg.is_continuous();
        self.update_state(id, |s| {
            let first = s.checkpoint == 0;
            s.checkpoint += 1;
            s.last_cp_millis = crate::date::now_millis();
            s.last_upper = upper;
            s.stats.exp_avg_duration = exp_avg(s.stats.exp_avg_duration, dur.max(0.001), first);
            s.stats.exp_avg_indexed = exp_avg(s.stats.exp_avg_indexed, cp_indexed as f64, first);
            s.stats.exp_avg_processed = exp_avg(s.stats.exp_avg_processed, cp_processed as f64, first);
            if !continuous || s.stop_at_checkpoint {
                s.started = false;
                s.stop_requested = false;
                s.stop_at_checkpoint = false;
            }
        });
        Ok(())
    }

    /// Filters restricting a continuous checkpoint to the buckets touched by
    /// new data (`None` when there is no new data).
    fn changed_buckets(&self, node: &Node, ops: &Ops, cfg: &Config, targets: &[crate::search::Target], probe: &Value, lower: i64) -> Result<Option<Vec<Value>>> {
        if targets.is_empty() {
            return Ok(None);
        }
        let src = cfg.source_indices.join(",");
        let n = ops.count(&src, targets, probe)?;
        if n == 0 {
            return Ok(None);
        }
        let sync_field = cfg.sync.as_ref().map(|s| s.0.clone()).unwrap_or_default();
        let mut filters = Vec::new();
        let term_fields: Vec<(String, String)> = match &cfg.function {
            Function::Pivot { groups, .. } => {
                for g in groups {
                    if g.kind == "date_histogram" && g.field() == Some(sync_field.as_str()) {
                        filters.push(json!({"range": {sync_field.clone(): {"gte": lower, "format": "epoch_millis"}}}));
                    }
                }
                groups.iter().filter(|g| g.kind == "terms" && g.params.get("script").is_none()).filter_map(|g| g.field().map(|f| (g.name.clone(), f.to_string()))).collect()
            }
            Function::Latest { unique_key, .. } => unique_key.iter().map(|k| (k.clone(), k.clone())).collect(),
        };
        for (_name, field) in term_fields {
            let mut vals: HashSet<String> = HashSet::new();
            let mut list: Vec<Value> = Vec::new();
            let mut has_missing = false;
            let mut after: Option<Value> = None;
            loop {
                let mut comp = json!({"size": 1000, "sources": [{"k": {"terms": {"field": field, "missing_bucket": true}}}]});
                if let Some(a) = &after {
                    comp["after"] = a.clone();
                }
                let mut body = json!({"size": 0, "query": probe, "aggs": {"c": {"composite": comp}}});
                if let Some(rm) = &cfg.runtime_mappings {
                    body["runtime_mappings"] = rm.clone();
                }
                let resp = ops.search(node, &src, targets, &body)?;
                let buckets = resp["aggregations"]["c"]["buckets"].as_array().cloned().unwrap_or_default();
                for b in &buckets {
                    let k = &b["key"]["k"];
                    if k.is_null() {
                        has_missing = true;
                    } else if vals.insert(k.to_string()) {
                        list.push(k.clone());
                    }
                }
                if buckets.len() < 1000 || vals.len() > 65_536 {
                    break;
                }
                after = resp["aggregations"]["c"].get("after_key").cloned();
                if after.is_none() {
                    break;
                }
            }
            if vals.len() > 65_536 {
                continue;
            }
            let terms = json!({"terms": {field.clone(): list}});
            filters.push(if has_missing {
                json!({"bool": {"should": [terms, {"bool": {"must_not": {"exists": {"field": field}}}}], "minimum_should_match": 1}})
            } else {
                terms
            });
        }
        Ok(Some(filters))
    }

    /// Bulk-index converted documents into `dest` (through the pipelines).
    fn index_docs(&self, node: &Node, ops: &Ops, cfg: &Config, dest: &str, docs: Vec<(String, Map<String, Value>)>) -> Result<(u64, u64)> {
        if let Some(rest) = &ops.0 {
            // The coordinator runs the pipelines (`dest.pipeline`, the
            // index's default/final pipelines) and routes to the primaries.
            let items: Vec<(String, String, Value)> = docs.into_iter().map(|(id, d)| (dest.to_string(), id, Value::Object(d))).collect();
            let uri = match &cfg.dest_pipeline {
                Some(p) => format!("/_bulk?pipeline={}", path_expr(p)),
                None => "/_bulk".to_string(),
            };
            let resp = rest("POST", &uri, Some(Value::String(bulk_body(&items))))?;
            return Ok(bulk_counts(&resp));
        }
        let mut by_index: BTreeMap<String, Vec<WriteReq>> = BTreeMap::new();
        let mut failed = 0u64;
        for (id, d) in docs {
            let src = serde_json::to_string(&Value::Object(d))?;
            let out = node.run_ingest(crate::ingest::IngestRequest { index: dest, id: Some(&id), routing: None, source: &src, pipeline: cfg.dest_pipeline.as_deref() });
            match out {
                Ok(crate::ingest::IngestOutcome::Unchanged) => by_index.entry(dest.to_string()).or_default().push(WriteReq::Index { id: Some(id), source: src, routing: None, cond: WriteCond::default() }),
                Ok(crate::ingest::IngestOutcome::Changed { index, id: nid, routing, source, .. }) => {
                    by_index.entry(index).or_default().push(WriteReq::Index { id: nid.or(Some(id)), source, routing, cond: WriteCond::default() })
                }
                Ok(crate::ingest::IngestOutcome::Dropped { .. }) => {}
                Err(_) => failed += 1,
            }
        }
        let mut ok = 0u64;
        for (index, reqs) in by_index {
            let svc = node.write_index(&index, true)?;
            for r in svc.write(reqs, true) {
                if r.is_ok() {
                    ok += 1;
                } else {
                    failed += 1;
                }
            }
        }
        Ok((ok, failed))
    }

    fn apply_retention(&self, node: &Node, ops: &Ops, dest: &str, field: &str, cutoff: i64) -> Result<u64> {
        if let Some(rest) = &ops.0 {
            if node.get_index(dest).is_none() {
                return Ok(0);
            }
            // ES runs a delete-by-query and refreshes the destination.
            let body = json!({"query": {"range": {field: {"lt": cutoff, "format": "epoch_millis"}}}});
            let r = rest("POST", &format!("/{}/_delete_by_query?refresh=true&conflicts=proceed", path_expr(dest)), Some(body))?;
            return Ok(r["deleted"].as_u64().unwrap_or(0));
        }
        let Some(svc) = node.get_index(dest) else { return Ok(0) };
        let targets = vec![crate::search::Target::new(svc.clone())];
        let mut deleted = 0;
        loop {
            let body = json!({"size": 1000, "_source": false, "query": {"range": {field: {"lt": cutoff, "format": "epoch_millis"}}}});
            let resp = pivot::search(node, &targets, &body)?;
            let hits = resp["hits"]["hits"].as_array().cloned().unwrap_or_default();
            if hits.is_empty() {
                break;
            }
            let reqs: Vec<WriteReq> = hits.iter().filter_map(|h| h["_id"].as_str()).map(|id| WriteReq::Delete { id: id.to_string(), routing: None, cond: WriteCond::default() }).collect();
            let n = reqs.len();
            for r in svc.write(reqs, true) {
                if r.is_ok() {
                    deleted += 1;
                }
            }
            // ES refreshes the destination after the retention
            // delete-by-query; the next page must not see these again.
            svc.refresh()?;
            if n < 1000 {
                break;
            }
        }
        Ok(deleted)
    }
}

#[cfg(test)]
mod cluster_ops_tests {
    use super::*;

    #[test]
    fn path_expr_keeps_index_expression_syntax() {
        assert_eq!(path_expr("logs-*,metrics"), "logs-*,metrics");
        assert_eq!(path_expr("<logs-{now/d}>"), "%3Clogs-%7Bnow%2Fd%7D%3E");
    }

    #[test]
    fn bulk_body_carries_ids_and_index() {
        let b = bulk_body(&[("dest".into(), "abc".into(), json!({"x": 1}))]);
        let lines: Vec<Value> = b.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines, vec![json!({"index": {"_index": "dest", "_id": "abc"}}), json!({"x": 1})]);
        assert!(b.ends_with('\n'));
    }

    #[test]
    fn bulk_counts_splits_failures() {
        let r = json!({"errors": true, "items": [{"index": {"status": 201}}, {"index": {"status": 400, "error": {"type": "x"}}}, {"index": {"status": 200}}]});
        assert_eq!(bulk_counts(&r), (2, 1));
    }
}
