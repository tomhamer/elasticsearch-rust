//! Watcher (x-pack): watches stored in `.watches`, a scheduler thread that
//! fires them, execution (input -> condition -> transform -> actions) with
//! history in `.watcher-history-*`, throttling and acking.
//!
//! * `schedule`  trigger schedules (interval / hourly / daily / ... / cron)
//! * `model`     watch parsing, validation and normalization
//! * `exec`      the execution pipeline (input, condition, transform)
//! * `actions`   logging, index, webhook, email, slack, pagerduty, jira
//! * `httpc`     minimal blocking HTTP client
//! * `api`       the REST-level operations (put/get/delete/execute/ack/...)

pub mod actions;
pub mod api;
pub mod exec;
pub mod httpc;
pub mod model;
pub mod normalize;
pub mod schedule;

use crate::error::{EsError, Result};
use crate::index::{IndexService, WriteReq};
use crate::node::{Node, ResolveOpts};
use crate::shard::WriteCond;
use parking_lot::{Mutex, RwLock};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

pub const WATCHES_INDEX: &str = ".watches";
pub const TRIGGERED_INDEX: &str = ".triggered_watches";
/// `.watcher-history-<HISTORY_VERSION>-<yyyy.MM.dd>`.
pub const HISTORY_VERSION: &str = "17";
pub const HISTORY_RETENTION_MS: i64 = 7 * 86_400_000;

/// ISO-8601 with millis, UTC (`2026-10-04T18:00:00.000Z`).
pub fn iso(ms: i64) -> String {
    crate::date::from_millis(ms).format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()
}

pub fn parse_iso(s: &str) -> Option<i64> {
    crate::date::DateFormatter::new("strict_date_optional_time||epoch_millis").ok()?.parse(s).ok()
}

/// A watch held in memory (the source of truth is its `.watches` document).
pub struct CachedWatch {
    pub id: String,
    /// Stored document source (`trigger`, `input`, ..., `status`).
    pub src: Value,
    pub schedule: Arc<schedule::Schedule>,
    pub version: u64,
    pub seq_no: i64,
    pub primary_term: u64,
    pub next_fire: Option<i64>,
}

impl CachedWatch {
    pub fn active(&self) -> bool {
        self.src["status"]["state"]["active"].as_bool().unwrap_or(true)
    }
    pub fn start_time(&self) -> i64 {
        self.src["status"]["state"]["timestamp"].as_str().and_then(parse_iso).unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
pub struct CurrentExec {
    pub watch_id: String,
    pub record_id: String,
    pub triggered_time: i64,
    pub execution_time: i64,
    pub phase: &'static str,
}

#[derive(Default)]
pub struct Counters {
    pub executed: AtomicUsize,
    pub failed: AtomicUsize,
}

pub struct WatcherService {
    path: PathBuf,
    /// `manually_stopped` (persisted): the service is STOPPED until `_start`.
    stopped: AtomicBool,
    pub cache: RwLock<BTreeMap<String, CachedWatch>>,
    loaded: AtomicBool,
    pub settings: Mutex<Map<String, Value>>,
    pub current: Mutex<HashMap<String, CurrentExec>>,
    pub counters: Counters,
    /// Executed actions by type: (count, total millis).
    pub action_stats: Mutex<HashMap<String, (u64, u64)>>,
    pub running_threads: AtomicUsize,
    last_cleanup: AtomicI64,
    started: AtomicBool,
    pub load_lock: Mutex<()>,
}

fn registry() -> &'static Mutex<HashMap<PathBuf, Arc<WatcherService>>> {
    static R: OnceLock<Mutex<HashMap<PathBuf, Arc<WatcherService>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The watcher service of `node` (created on first use, state in
/// `<data>/_state/watcher.json`).
pub fn service(node: &Node) -> Arc<WatcherService> {
    let mut r = registry().lock();
    r.entry(node.data.clone())
        .or_insert_with(|| {
            let path = node.data.join("_state/watcher.json");
            let st: Value = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(json!({}));
            Arc::new(WatcherService {
                path,
                stopped: AtomicBool::new(st["manually_stopped"].as_bool().unwrap_or(false)),
                cache: RwLock::new(BTreeMap::new()),
                loaded: AtomicBool::new(false),
                settings: Mutex::new(st["settings"].as_object().cloned().unwrap_or_default()),
                current: Mutex::new(HashMap::new()),
                counters: Counters::default(),
                action_stats: Mutex::new(HashMap::new()),
                running_threads: AtomicUsize::new(0),
                last_cleanup: AtomicI64::new(0),
                started: AtomicBool::new(false),
                load_lock: Mutex::new(()),
            })
        })
        .clone()
}

impl WatcherService {
    pub fn manually_stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }

    pub fn set_stopped(&self, stopped: bool) {
        self.stopped.store(stopped, Ordering::SeqCst);
        self.persist();
    }

    /// Forget every cached watch (`POST /_features/_reset` deleted the
    /// watcher indices); the cache reloads from `.watches` on next use.
    pub fn reset(&self) {
        let _g = self.load_lock.lock();
        self.cache.write().clear();
        self.loaded.store(false, Ordering::SeqCst);
    }

    pub fn persist(&self) {
        let v = json!({"manually_stopped": self.manually_stopped(), "settings": &*self.settings.lock()});
        if let Some(d) = self.path.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let tmp = self.path.with_extension("json.tmp");
        if std::fs::write(&tmp, serde_json::to_vec_pretty(&v).unwrap_or_default()).is_ok() {
            let _ = std::fs::rename(tmp, &self.path);
        }
    }

    pub fn state_name(&self) -> &'static str {
        if self.manually_stopped() {
            "stopped"
        } else {
            "started"
        }
    }

    /// Load every stored watch into the cache (once).
    pub fn ensure_loaded(&self, node: &Node) {
        if self.loaded.load(Ordering::SeqCst) {
            return;
        }
        let _g = self.load_lock.lock();
        if self.loaded.load(Ordering::SeqCst) {
            return;
        }
        if let Some(svc) = node.get_index(WATCHES_INDEX) {
            let _ = svc.refresh();
            for (id, src, version, seq_no, pt) in scan_watches(node).unwrap_or_default() {
                match model::build_cached(&id, src, version, seq_no, pt) {
                    Ok(cw) => {
                        self.cache.write().insert(id, cw);
                    }
                    Err(e) => tracing::warn!("failed to load watch [{id}]: {e}"),
                }
            }
        }
        self.loaded.store(true, Ordering::SeqCst);
    }

    pub fn record_action(&self, ty: &str, millis: u64) {
        let mut s = self.action_stats.lock();
        let e = s.entry(ty.to_string()).or_insert((0, 0));
        e.0 += 1;
        e.1 += millis;
    }

    pub fn watch_count(&self) -> usize {
        self.cache.read().len()
    }
}

/// `(id, source, version, seq_no, primary_term)` of every document in `.watches`.
pub fn scan_watches(node: &Node) -> Result<Vec<(String, Value, u64, i64, u64)>> {
    let mut out = Vec::new();
    let Some(svc) = node.get_index(WATCHES_INDEX) else { return Ok(out) };
    let _ = svc.refresh();
    let targets = node.resolve(WATCHES_INDEX, ResolveOpts { expand_hidden: true, ..Default::default() })?;
    let body = json!({"size": 10000, "query": {"match_all": {}}, "seq_no_primary_term": true, "version": true});
    let v: Value = match remote() {
        Some(rest) => {
            let _ = rest("POST", &format!("/{WATCHES_INDEX}/_refresh"), None);
            rest("POST", &format!("/{WATCHES_INDEX}/_search?expand_wildcards=all"), Some(body))?
        }
        None => {
            let req = crate::search::parse_request(&body, &HashMap::new())?;
            let o = crate::search::execute(&targets, &req, &search_opts(node))?;
            serde_json::from_str(&o.json).map_err(|e| EsError::internal(e.to_string()))?
        }
    };
    for h in v["hits"]["hits"].as_array().cloned().unwrap_or_default() {
        out.push((
            h["_id"].as_str().unwrap_or("").to_string(),
            h["_source"].clone(),
            h["_version"].as_u64().unwrap_or(1),
            h["_seq_no"].as_i64().unwrap_or(0),
            h["_primary_term"].as_u64().unwrap_or(1),
        ));
    }
    Ok(out)
}

/// A REST request through a cluster coordinator: `(method, uri, body)` →
/// response body (a `Value::String` body is sent as NDJSON).
pub type RestCall = Arc<dyn Fn(&str, &str, Option<Value>) -> Result<Value> + Send + Sync>;

fn remote_slot() -> &'static Mutex<Option<RestCall>> {
    static R: OnceLock<Mutex<Option<RestCall>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(None))
}

/// On a cluster, watch inputs, transforms and index actions read and write
/// the cluster's indices through its coordinator (set from the watcher
/// requests this node serves; scheduled executions use the latest).
pub fn set_remote(remote: Option<RestCall>) {
    if remote.is_some() {
        *remote_slot().lock() = remote;
    }
}

pub(crate) fn remote() -> Option<RestCall> {
    remote_slot().lock().clone()
}

/// An index expression in a REST path (`,` and `*` kept).
pub(crate) fn path_expr(expr: &str) -> String {
    expr.bytes().map(|b| if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'*' | b',') { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}

pub fn search_opts(node: &Node) -> crate::search::ExecOpts<'static> {
    crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None }
}

/// Mapping of `.watches` (as ES's `watches.json`).
fn watches_mapping() -> Value {
    let off = json!({"type": "object", "enabled": false, "dynamic": true});
    json!({
        "dynamic": "strict",
        "_meta": {"managed_index_mappings_version": 1, "version": "8.12.0"},
        "properties": {
            "status": off, "trigger": off, "input": off, "condition": off, "actions": off, "transform": off,
            "throttle_period": {"type": "keyword", "index": false, "doc_values": false},
            "throttle_period_in_millis": {"type": "long", "index": false, "doc_values": false},
            "metadata": {"type": "object", "dynamic": true}
        }
    })
}

fn system_settings() -> Value {
    json!({"index": {"number_of_shards": 1, "auto_expand_replicas": "0-1", "format": 6, "priority": 800, "hidden": true}})
}

/// `.watches`, created on first write.
/// Create a system index through the coordinator (on a cluster), so it is
/// in the cluster metadata and allocated before documents are written.
fn create_remote(node: &Node, name: &str, body: &Value) -> Option<Result<Arc<IndexService>>> {
    let rest = remote()?;
    Some(match rest("PUT", &format!("/{}", path_expr(name)), Some(body.clone())) {
        Ok(_) => node.get_index(name).ok_or_else(|| EsError::internal(format!("[{name}] was created but is not known here"))),
        Err(e) if e.kind == "resource_already_exists_exception" || e.kind == "index_already_exists_exception" => node.get_index(name).ok_or(e),
        Err(e) => Err(e),
    })
}

pub fn watches_index(node: &Node) -> Result<Arc<IndexService>> {
    if let Some(s) = node.get_index(WATCHES_INDEX) {
        return Ok(s);
    }
    if let Some(r) = create_remote(node, WATCHES_INDEX, &json!({"settings": system_settings(), "mappings": watches_mapping()})) {
        return r;
    }
    match node.create_index(WATCHES_INDEX, &json!({"settings": system_settings(), "mappings": watches_mapping()})) {
        Ok(s) => Ok(s),
        Err(e) if e.kind == "resource_already_exists_exception" || e.kind == "index_already_exists_exception" => {
            node.get_index(WATCHES_INDEX).ok_or(e)
        }
        Err(e) => Err(e),
    }
}

pub fn triggered_index(node: &Node) -> Result<Arc<IndexService>> {
    if let Some(s) = node.get_index(TRIGGERED_INDEX) {
        return Ok(s);
    }
    let m = json!({"dynamic": "strict", "properties": {
        "trigger_event": {"type": "object", "dynamic": true, "properties": {"type": {"type": "keyword"}, "triggered_time": {"type": "date"},
            "manual": {"type": "object", "dynamic": true, "properties": {"schedule": {"type": "object", "dynamic": true}}},
            "schedule": {"type": "object", "dynamic": true, "properties": {"scheduled_time": {"type": "date"}}}}},
        "state": {"type": "keyword"}}});
    if let Some(r) = create_remote(node, TRIGGERED_INDEX, &json!({"settings": system_settings(), "mappings": m})) {
        return r;
    }
    match node.create_index(TRIGGERED_INDEX, &json!({"settings": system_settings(), "mappings": m})) {
        Ok(s) => Ok(s),
        Err(_) => node.get_index(TRIGGERED_INDEX).ok_or_else(|| EsError::internal("cannot create .triggered_watches")),
    }
}

/// The write URI of one document of `index` with `cond` (`?if_seq_no=`,
/// `op_type=create`, ...).
fn doc_uri(index: &str, id: &str, cond: &WriteCond) -> String {
    let mut q: Vec<String> = Vec::new();
    if cond.create {
        q.push("op_type=create".into());
    }
    if let Some(s) = cond.if_seq_no {
        q.push(format!("if_seq_no={s}"));
    }
    if let Some(t) = cond.if_primary_term {
        q.push(format!("if_primary_term={t}"));
    }
    if let Some(v) = cond.version {
        q.push(format!("version={v}"));
    }
    format!("/{}/_doc/{}{}{}", path_expr(index), path_expr(id), if q.is_empty() { "" } else { "?" }, q.join("&"))
}

/// Index one document; returns `(version, seq_no, primary_term, created)`.
/// On a cluster the write goes through the coordinator to the primary.
pub fn put_doc(svc: &IndexService, id: &str, src: &Value, cond: WriteCond) -> Result<(u64, i64, u64, bool)> {
    if let Some(rest) = remote() {
        let r = rest("PUT", &doc_uri(&svc.name, id, &cond), Some(src.clone()))?;
        return Ok((r["_version"].as_u64().unwrap_or(1), r["_seq_no"].as_i64().unwrap_or(0), r["_primary_term"].as_u64().unwrap_or(1), r["result"].as_str() == Some("created")));
    }
    put_doc_local(svc, id, src, cond)
}

/// [`put_doc`] on this node's engine.
pub fn put_doc_local(svc: &IndexService, id: &str, src: &Value, cond: WriteCond) -> Result<(u64, i64, u64, bool)> {
    let mut r = svc.write(vec![WriteReq::Index { id: Some(id.to_string()), source: src.to_string(), routing: None, cond }], false);
    let w = r.pop().ok_or_else(|| EsError::internal("no write result"))??;
    Ok((w.version, w.seq_no, w.primary_term, w.result == "created"))
}

pub fn delete_doc(svc: &IndexService, id: &str) -> Result<Option<u64>> {
    if let Some(rest) = remote() {
        let r = rest("DELETE", &format!("/{}/_doc/{}", path_expr(&svc.name), path_expr(id)), None)?;
        return Ok(if r["result"].as_str() == Some("deleted") { r["_version"].as_u64() } else { None });
    }
    let mut r = svc.write(vec![WriteReq::Delete { id: id.to_string(), routing: None, cond: WriteCond::default() }], false);
    let w = r.pop().ok_or_else(|| EsError::internal("no write result"))??;
    Ok(if w.result == "deleted" { Some(w.version) } else { None })
}

/// The history data stream `.watcher-history-17`.
pub fn history_stream() -> String {
    format!(".watcher-history-{HISTORY_VERSION}")
}

fn history_mapping() -> Value {
    let off = json!({"type": "object", "enabled": false});
    json!({
        "dynamic": false,
        "properties": {
            "watch_id": {"type": "keyword"},
            "node": {"type": "keyword"},
            "trigger_event": {"type": "object", "dynamic": true, "properties": {
                "type": {"type": "keyword"}, "triggered_time": {"type": "date"},
                "manual": {"type": "object", "dynamic": true, "properties": {"schedule": {"type": "object", "dynamic": true, "properties": {"scheduled_time": {"type": "date"}}}}},
                "schedule": {"type": "object", "dynamic": true, "properties": {"scheduled_time": {"type": "date"}}}}},
            "state": {"type": "keyword"},
            "status": off.clone(), "input": off.clone(), "condition": off.clone(), "transform": off.clone(), "actions": off.clone(), "metadata": {"type": "object", "dynamic": true},
            "messages": {"type": "text"},
            "user": {"type": "text"},
            "exception": off.clone(),
            "result": {"type": "object", "dynamic": true, "properties": {
                "execution_time": {"type": "date"}, "execution_duration": {"type": "long"},
                "input": {"type": "object", "dynamic": true, "properties": {"type": {"type": "keyword"}, "status": {"type": "keyword"}, "payload": off.clone(), "search": off.clone(), "http": off.clone(), "reason": {"type": "text"}}},
                "condition": {"type": "object", "dynamic": true, "properties": {"type": {"type": "keyword"}, "status": {"type": "keyword"}, "met": {"type": "boolean"}, "compare": off.clone(), "array_compare": off.clone(), "script": off.clone(), "reason": {"type": "text"}}},
                "transform": {"type": "object", "dynamic": true, "properties": {"type": {"type": "keyword"}, "status": {"type": "keyword"}, "payload": off.clone(), "search": off.clone(), "reason": {"type": "text"}}},
                "actions": {"type": "nested", "dynamic": true, "properties": {"id": {"type": "keyword"}, "type": {"type": "keyword"}, "status": {"type": "keyword"}, "reason": {"type": "keyword"},
                    "email": off.clone(), "webhook": off.clone(), "index": off.clone(), "logging": off.clone(), "slack": off.clone(), "pagerduty": off.clone(), "jira": off.clone()}}}}
        }
    })
}

/// Write one history document (data stream and template created on demand).
pub fn write_history(node: &Node, id: &str, record: &Value, _now: i64) {
    let name = history_stream();
    if let Some(rest) = remote() {
        // On a cluster: the template is cluster metadata and the record goes
        // to the data stream's primary (the stream is created on demand).
        if !node.templates.read().index.contains_key(&name) {
            if let Err(e) = rest("PUT", &format!("/_index_template/{}", path_expr(&name)), Some(history_template(&name))) {
                tracing::warn!("failed to install the watch history template: {e}");
            }
        }
        if let Err(e) = rest("PUT", &history_doc_uri(&name, id), Some(record.clone())) {
            tracing::warn!("failed to write watch history [{id}]: {e}");
        }
        return;
    }
    if !node.templates.read().index.contains_key(&name) {
        let tpl = history_template(&name);
        node.templates.write().index.insert(name.clone(), tpl);
        let _ = node.persist_state();
    }
    let svc = match node.write_index(&name, true) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("failed to write watch history [{id}]: {e}");
            return;
        }
    };
    if let Err(e) = put_doc_local(&svc, id, record, WriteCond { create: true, ..Default::default() }) {
        tracing::warn!("failed to write watch history [{id}]: {e}");
    }
}

/// The index template of the watch history data stream.
fn history_template(name: &str) -> Value {
    json!({
        "index_patterns": [format!("{name}*")],
        "priority": 2147483647i64,
        "data_stream": {"hidden": true},
        "template": {
            "settings": {"index.number_of_shards": 1, "index.auto_expand_replicas": "0-1", "index.hidden": true, "index.format": 6},
            "mappings": history_mapping()
        },
        "version": 17,
        "_meta": {"description": "index template for watcher history indices", "managed": true}
    })
}

/// The create-only write of one history record into the data stream.
fn history_doc_uri(stream: &str, id: &str) -> String {
    format!("/{}/_create/{}", path_expr(stream), path_expr(id))
}

/// Setting lookup: cluster settings, then `ES_<KEY>` environment variable.
pub fn setting(node: &Node, key: &str) -> Option<String> {
    crate::ilm::cluster_setting(node, key)
        .or_else(|| node.node_settings.read().get(key).map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())))
        .or_else(|| std::env::var(format!("ES_{}", key.to_ascii_uppercase().replace(['.', '-'], "_"))).ok())
}

pub fn watcher_enabled(node: &Node) -> bool {
    setting(node, "xpack.watcher.enabled").map(|s| s != "false").unwrap_or(true)
}

pub fn default_throttle_ms(node: &Node) -> i64 {
    setting(node, "xpack.watcher.execution.default_throttle_period").and_then(|s| crate::ilm::parse_time_millis(&s)).unwrap_or(5000)
}

/// Delete history backing indices older than the retention (daily cleaner).
fn cleanup_history(node: &Node, now: i64) {
    if setting(node, "xpack.watcher.history.cleaner_service.enabled").map(|s| s == "false").unwrap_or(false) {
        return;
    }
    let retention = setting(node, "xpack.watcher.history.cleaner_service.retention").and_then(|s| crate::ilm::parse_time_millis(&s)).unwrap_or(HISTORY_RETENTION_MS);
    let Some(ds) = node.data_stream(&history_stream()) else { return };
    for idx in ds.indices.iter().filter(|i| i.as_str() != ds.write_index()) {
        if let Some(svc) = node.get_index(idx) {
            if now - svc.creation_date > retention {
                let _ = node.delete_index(idx);
            }
        }
    }
}

/// One scheduler pass at `now`: fire due watches. Returns the watches fired
/// as `(id, scheduled_time)`; exposed for tests with a fake clock.
pub fn due_watches(svc: &WatcherService, now: i64) -> Vec<(String, i64)> {
    let mut fired = Vec::new();
    let mut cache = svc.cache.write();
    for (id, w) in cache.iter_mut() {
        if !w.active() {
            w.next_fire = None;
            continue;
        }
        if w.next_fire.is_none() {
            w.next_fire = w.schedule.next_after(w.start_time(), now);
        }
        if let Some(nf) = w.next_fire {
            if nf <= now {
                fired.push((id.clone(), nf));
                w.next_fire = w.schedule.next_after(w.start_time(), nf.max(now - 1));
            }
        }
    }
    fired
}

/// Start the scheduler thread for `node` (idempotent).
pub fn start(node: &Arc<Node>) {
    let svc = service(node);
    if svc.started.swap(true, Ordering::SeqCst) {
        return;
    }
    let weak = Arc::downgrade(node);
    let spawned = std::thread::Builder::new().name("watcher-scheduler".into()).spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_millis(250));
        let Some(node) = weak.upgrade() else { break };
        let svc = service(&node);
        if !watcher_enabled(&node) {
            continue;
        }
        svc.ensure_loaded(&node);
        let now = crate::date::now_millis();
        if now - svc.last_cleanup.load(Ordering::Relaxed) > 3_600_000 {
            svc.last_cleanup.store(now, Ordering::Relaxed);
            cleanup_history(&node, now);
        }
        if svc.manually_stopped() {
            continue;
        }
        for (id, scheduled) in due_watches(&svc, now) {
            if svc.running_threads.load(Ordering::SeqCst) >= 64 {
                tracing::warn!("watcher execution pool is saturated; skipping watch [{id}]");
                continue;
            }
            svc.running_threads.fetch_add(1, Ordering::SeqCst);
            let node2 = node.clone();
            let svc2 = svc.clone();
            let ok = std::thread::Builder::new().name(format!("watcher-exec-{id}")).spawn(move || {
                if let Err(e) = exec::run_scheduled(&node2, &svc2, &id, scheduled, now) {
                    tracing::warn!("watch [{id}] execution failed: {e}");
                }
                svc2.running_threads.fetch_sub(1, Ordering::SeqCst);
            });
            if ok.is_err() {
                svc.running_threads.fetch_sub(1, Ordering::SeqCst);
            }
        }
    });
    if spawned.is_err() {
        svc.started.store(false, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod doc_uri_tests {
    use super::*;

    #[test]
    fn history_goes_to_the_data_stream_create_only() {
        assert_eq!(history_doc_uri(".watcher-history-17", "w_1-2026"), "/.watcher-history-17/_create/w_1-2026");
        let t = history_template(".watcher-history-17");
        assert_eq!(t["index_patterns"], json!([".watcher-history-17*"]));
        assert_eq!(t["data_stream"], json!({"hidden": true}));
    }

    #[test]
    fn doc_uri_carries_conditions() {
        assert_eq!(doc_uri(".watches", "w 1", &WriteCond::default()), "/.watches/_doc/w%201");
        let c = WriteCond { create: true, if_seq_no: Some(3), if_primary_term: Some(1), ..Default::default() };
        assert_eq!(doc_uri(".watches", "w", &c), "/.watches/_doc/w?op_type=create&if_seq_no=3&if_primary_term=1");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(s: &str) -> i64 {
        crate::date::DateFormatter::new("strict_date_optional_time").unwrap().parse(s).unwrap()
    }

    fn stored(trigger: Value, created: &str) -> Value {
        let mut src = model::parse_watch("x", &json!({"trigger": {"schedule": trigger}}), ms(created), true).unwrap();
        src["status"]["state"]["timestamp"] = json!(created);
        src
    }

    /// Fake-clock scheduler test: every schedule type fires at the expected instants.
    #[test]
    fn scheduler_fake_clock() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let svc = service(&node);
        let created = "2020-01-01T00:00:00.500Z";
        let cases: Vec<(&str, Value, Vec<&str>)> = vec![
            ("iv", json!({"interval": "10s"}), vec!["2020-01-01T00:00:10.500Z", "2020-01-01T00:00:20.500Z"]),
            ("hourly", json!({"hourly": {"minute": [15, 45]}}), vec!["2020-01-01T00:15:00Z", "2020-01-01T00:45:00Z", "2020-01-01T01:15:00Z"]),
            ("daily", json!({"daily": {"at": "02:30"}}), vec!["2020-01-01T02:30:00Z", "2020-01-02T02:30:00Z"]),
            ("weekly", json!({"weekly": {"on": "friday", "at": "noon"}}), vec!["2020-01-03T12:00:00Z", "2020-01-10T12:00:00Z"]),
            ("monthly", json!({"monthly": {"on": 15, "at": "midnight"}}), vec!["2020-01-15T00:00:00Z", "2020-02-15T00:00:00Z"]),
            ("yearly", json!({"yearly": {"in": "mar", "on": 1, "at": "00:00"}}), vec!["2020-03-01T00:00:00Z", "2021-03-01T00:00:00Z"]),
            ("cron", json!({"cron": "0 0/30 * * * ?"}), vec!["2020-01-01T00:30:00Z", "2020-01-01T01:00:00Z"]),
        ];
        for (id, trig, _) in &cases {
            let cw = model::build_cached(id, stored(trig.clone(), created), 1, 0, 1).unwrap();
            svc.cache.write().insert(id.to_string(), cw);
        }
        // advance a fake clock in 1s steps and record every firing per watch
        let mut fired: HashMap<String, Vec<i64>> = HashMap::new();
        let mut t = ms("2020-01-01T00:00:01Z");
        let end = ms("2021-03-02T00:00:00Z");
        // coarse steps keep the loop short; each fire's scheduled time is exact
        while t <= end {
            for (id, scheduled) in due_watches(&svc, t) {
                fired.entry(id).or_default().push(scheduled);
            }
            t += if t < ms("2020-01-02T00:00:00Z") { 1000 } else { 60_000 };
        }
        for (id, _, want) in &cases {
            let got = &fired[*id];
            for (i, w) in want.iter().enumerate() {
                assert_eq!(got[i], ms(w), "watch {id} firing #{i}");
            }
        }
        // an inactive watch never fires
        let mut src = stored(json!({"interval": "1s"}), created);
        src["status"]["state"]["active"] = json!(false);
        svc.cache.write().insert("off".into(), model::build_cached("off", src, 1, 0, 1).unwrap());
        assert!(due_watches(&svc, end + 10_000).iter().all(|(id, _)| id != "off"));
    }
}
