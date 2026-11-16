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
use crate::security::owner::StoredAuthn;
use crate::shard::WriteCond;
use parking_lot::{Mutex, RwLock};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

pub const WATCHES_INDEX: &str = ".watches";
pub const TRIGGERED_INDEX: &str = ".triggered_watches";
/// `.watcher-history-<HISTORY_VERSION>-<yyyy.MM.dd>`.
pub const HISTORY_VERSION: &str = "18";
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
    started: AtomicBool,
    /// Pending triggered executions were recovered (once per start).
    pub recovered: AtomicBool,
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
                started: AtomicBool::new(false),
                recovered: AtomicBool::new(false),
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

    /// Make `watches` the cached set (a cluster node reloading `.watches`).
    /// A watch whose trigger, start time and activity did not change keeps
    /// its next fire time, so a reload never fires a trigger again.
    pub fn replace_all(&self, watches: Vec<CachedWatch>) {
        let _g = self.load_lock.lock();
        let mut cache = self.cache.write();
        let mut next = BTreeMap::new();
        for mut w in watches {
            if let Some(old) = cache.get(&w.id) {
                if old.src["trigger"] == w.src["trigger"] && old.start_time() == w.start_time() && old.active() == w.active() {
                    w.next_fire = old.next_fire;
                }
            }
            next.insert(w.id.clone(), w);
        }
        *cache = next;
        self.loaded.store(true, Ordering::SeqCst);
    }

    /// Whether the stored watches were loaded.
    pub fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
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
            // A scan that fails (or is incomplete) is retried on the next
            // tick: it never counts as loading an empty schedule.
            let scanned = match scan_watches(node) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("watcher: cannot load .watches yet: {e}");
                    return;
                }
            };
            for (id, src, version, seq_no, pt) in scanned {
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

/// Watches read per page of a `.watches` scan.
pub const WATCH_PAGE: usize = 1000;

/// Every hit of `.watches`, page by page (`search` runs one page's body):
/// no cap on the number of watches (finding 201), and a page that timed out
/// or had failed shards fails the whole scan, so an incomplete list never
/// replaces the schedule (finding 203).
pub fn all_watch_hits(search: &dyn Fn(&Value) -> Result<Value>) -> Result<Vec<Value>> {
    let mut out: Vec<Value> = Vec::new();
    let mut after: Option<Value> = None;
    loop {
        let mut body = json!({"size": WATCH_PAGE, "query": {"match_all": {}}, "seq_no_primary_term": true, "version": true, "sort": [{"_seq_no": "asc"}], "track_total_hits": false});
        if let Some(a) = &after {
            body["search_after"] = a.clone();
        }
        let v = search(&body)?;
        let failed = v["_shards"]["failed"].as_u64().unwrap_or(0);
        if v["timed_out"] == true || failed > 0 {
            return Err(EsError::new(503, "status_exception", format!("the .watches scan returned incomplete results (timed_out: {}, failed shards: {failed})", v["timed_out"] == true)));
        }
        let hits = v["hits"]["hits"].as_array().cloned().unwrap_or_default();
        let full = hits.len() >= WATCH_PAGE;
        after = hits.last().map(|h| h["sort"].clone()).filter(Value::is_array);
        out.extend(hits);
        if !full {
            return Ok(out);
        }
        if after.is_none() {
            return Err(EsError::internal("the .watches scan cannot page past a full page (no sort values)"));
        }
    }
}

/// `(id, source, version, seq_no, primary_term)` of every document in `.watches`.
pub fn scan_watches(node: &Node) -> Result<Vec<(String, Value, u64, i64, u64)>> {
    let mut out = Vec::new();
    let Some(svc) = node.get_index(WATCHES_INDEX) else { return Ok(out) };
    let _ = svc.refresh();
    let targets = node.resolve(WATCHES_INDEX, ResolveOpts { expand_hidden: true, ..Default::default() })?;
    let rest = remote();
    if let Some(rest) = &rest {
        let _ = rest("POST", &format!("/{WATCHES_INDEX}/_refresh"), None);
    }
    let hits = all_watch_hits(&|body: &Value| match &rest {
        Some(rest) => rest("POST", &format!("/{WATCHES_INDEX}/_search?expand_wildcards=all"), Some(body.clone())),
        None => {
            let req = crate::search::parse_request(body, &HashMap::new())?;
            let o = crate::search::execute(&targets, &req, &search_opts(node))?;
            serde_json::from_str(&o.json).map_err(|e| EsError::internal(e.to_string()))
        }
    })?;
    for h in hits {
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

/// Builds [`RestCall`]s through the cluster coordinator running as a stored
/// subject (`None`: the watcher's internal origin, `_system`). It carries no
/// caller credentials: every execution names the identity it runs as.
pub type RestFactory = Arc<dyn Fn(Option<&StoredAuthn>) -> Result<RestCall> + Send + Sync>;

fn factory_slot() -> &'static Mutex<Option<RestFactory>> {
    static R: OnceLock<Mutex<Option<RestFactory>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(None))
}

/// On a cluster, watch inputs, transforms and index actions read and write
/// the cluster's indices through its coordinator (set from the watcher
/// requests this node serves). Each call runs as an explicit identity:
/// the watcher's own indices as `_system` ([`remote`]), a watch's searches
/// and writes as the subject that stored the watch ([`remote_as`]).
pub fn set_rest_factory(factory: Option<RestFactory>) {
    if factory.is_some() {
        *factory_slot().lock() = factory;
    }
}

fn factory() -> Option<RestFactory> {
    factory_slot().lock().clone()
}

/// Coordinator calls as the watcher's internal origin (`.watches`,
/// `.triggered_watches`, the history), as ES writes them with
/// `WATCHER_ORIGIN`. `None` off a cluster.
pub(crate) fn remote() -> Option<RestCall> {
    let f = factory()?;
    match f(None) {
        Ok(r) => Some(r),
        Err(e) => {
            tracing::warn!("watcher coordinator call unavailable: {e}");
            None
        }
    }
}

/// Coordinator calls as `identity` (`None`: the internal origin, for watches
/// stored with security off). `Ok(None)` off a cluster.
pub(crate) fn remote_as(identity: Option<&StoredAuthn>) -> Result<Option<RestCall>> {
    match factory() {
        Some(f) => f(identity).map(Some),
        None => Ok(None),
    }
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

/// Realtime read of one document: `(source, version, seq_no, primary_term)`,
/// `None` when it does not exist. On a cluster the read goes through the
/// coordinator (the primary's copy is the one conditional writes check).
pub fn get_doc(svc: &IndexService, id: &str) -> Result<Option<(Value, u64, i64, u64)>> {
    if let Some(rest) = remote() {
        let v = match rest("GET", &format!("/{}/_doc/{}", path_expr(&svc.name), path_expr(id)), None) {
            Ok(v) => v,
            Err(e) if e.kind == "index_not_found_exception" || e.status == 404 => return Ok(None),
            Err(e) => return Err(e),
        };
        if v["found"] != true {
            return Ok(None);
        }
        return Ok(Some((v["_source"].clone(), v["_version"].as_u64().unwrap_or(1), v["_seq_no"].as_i64().unwrap_or(0), v["_primary_term"].as_u64().unwrap_or(1))));
    }
    let Some(g) = svc.get(id)? else { return Ok(None) };
    let src: Value = match g.source.as_deref() {
        Some(s) => serde_json::from_str(s).map_err(|e| EsError::internal(e.to_string()))?,
        None => return Ok(None),
    };
    Ok(Some((src, g.version, g.seq_no, g.primary_term)))
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

/// The history data stream `.watcher-history-18`.
pub fn history_stream() -> String {
    format!(".watcher-history-{HISTORY_VERSION}")
}

/// The composable template of the history stream (ES
/// `WatcherIndexTemplateRegistry.HISTORY_TEMPLATE_NAME`): `.watch-history-18`.
pub fn history_template_name() -> String {
    format!(".watch-history-{HISTORY_VERSION}")
}

/// Install the watcher registry's lifecycle policy and history template
/// when missing or older, as ES's `WatcherIndexTemplateRegistry` does on
/// startup (they used to appear only with the first history record, so
/// `GET _ilm/policy/watch-history-ilm-policy-16` answered 404). Called by
/// [`crate::registry::ensure`]. Returns whether anything changed.
pub fn ensure_registry(node: &Node) -> bool {
    if !watcher_enabled(node) {
        return false;
    }
    let ilm = use_ilm(node);
    let mut changed = false;
    // `WatcherIndexTemplateRegistry.getLifecyclePolicies`: the history
    // policy, installed (if missing) only by the watcher's registry, so a
    // node with watcher disabled has none.
    if ilm && node.ilm.policy(HISTORY_POLICY).is_none() {
        match node.ilm.put_policy(node, HISTORY_POLICY, history_policy()) {
            Ok(_) => changed = true,
            Err(e) => tracing::warn!("failed to install the watch history lifecycle policy: {e}"),
        }
    }
    let name = history_template_name();
    let want = history_template(&history_stream(), ilm);
    let older = node.templates.read().index.get(&name).is_none_or(|t| t["version"].as_i64().unwrap_or(0) < want["version"].as_i64().unwrap_or(0));
    if older {
        let mut t = want;
        let now = crate::date::now_millis();
        t["created_date_millis"] = now.into();
        t["modified_date_millis"] = now.into();
        node.templates.write().index.insert(name, t);
        changed = true;
    }
    changed
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
    let ilm = use_ilm(node);
    if let Some(rest) = remote() {
        // On a cluster: the policy and template are cluster metadata and the
        // record goes to the data stream's primary (created on demand).
        if ilm && node.ilm.policy(HISTORY_POLICY).is_none() {
            if let Err(e) = rest("PUT", &format!("/_ilm/policy/{HISTORY_POLICY}"), Some(json!({"policy": history_policy()}))) {
                tracing::warn!("failed to install the watch history lifecycle policy: {e}");
            }
        }
        if !node.templates.read().index.contains_key(&history_template_name()) {
            if let Err(e) = rest("PUT", &format!("/_index_template/{}", path_expr(&history_template_name())), Some(history_template(&name, ilm))) {
                tracing::warn!("failed to install the watch history template: {e}");
            }
        }
        if let Err(e) = rest("PUT", &history_doc_uri(&name, id), Some(record.clone())) {
            tracing::warn!("failed to write watch history [{id}]: {e}");
        }
        return;
    }
    if ilm && node.ilm.policy(HISTORY_POLICY).is_none() {
        if let Err(e) = node.ilm.put_policy(node, HISTORY_POLICY, history_policy()) {
            tracing::warn!("failed to install the watch history lifecycle policy: {e}");
        }
    }
    if !node.templates.read().index.contains_key(&history_template_name()) {
        let tpl = history_template(&name, ilm);
        node.templates.write().index.insert(history_template_name(), tpl);
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

/// The lifecycle policy of the watch history (ES
/// `WatcherIndexTemplateRegistry`, `watch-history-ilm-policy.json`): roll
/// over at 50gb or 3 days, delete 4 days after rollover.
pub const HISTORY_POLICY: &str = "watch-history-ilm-policy-16";

pub fn history_policy() -> Value {
    // ES's `watch-history-ilm-policy.json`, as `GET _ilm/policy` shows it.
    json!({
        "phases": {
            "hot": {"min_age": "0ms", "actions": {"rollover": {"max_age": "3d", "max_primary_shard_size": "50gb"}}},
            "delete": {"min_age": "4d", "actions": {"delete": {"delete_searchable_snapshot": true}}}
        },
        "_meta": {"managed": true, "description": "default policy for the watcher history indices"}
    })
}

/// `xpack.watcher.use_ilm_index_management` (default true): the history
/// template names [`HISTORY_POLICY`], which ILM applies on every node set.
fn use_ilm(node: &Node) -> bool {
    setting(node, "xpack.watcher.use_ilm_index_management").map(|s| s != "false").unwrap_or(true)
}

/// The index template of the watch history data stream.
fn history_template(name: &str, ilm: bool) -> Value {
    let mut settings = json!({"index.number_of_shards": 1, "index.auto_expand_replicas": "0-1", "index.hidden": true, "index.format": 6});
    if ilm {
        settings["index.lifecycle.name"] = json!(HISTORY_POLICY);
    }
    json!({
        "index_patterns": [format!("{name}*")],
        "priority": 2147483647i64,
        "data_stream": {"hidden": true},
        "template": {
            "settings": settings,
            "mappings": history_mapping()
        },
        "allow_auto_create": true,
        "version": HISTORY_VERSION.parse::<i64>().unwrap_or(18),
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


/// One scheduler pass at `now`: fire due watches. Returns the watches fired
/// as `(id, scheduled_time)`; exposed for tests with a fake clock.
pub fn due_watches(svc: &WatcherService, now: i64) -> Vec<(String, i64)> {
    due_watches_where(svc, now, 0, &|_| true)
}

/// [`due_watches`] for the watches `owned` says this node triggers (on a
/// cluster: the watches whose `.watches` copy it was assigned, see
/// `elasticsearch_node::watcher_sched`). A watch this node does not own has
/// no next fire time; one it starts owning is scheduled from `now + grace`
/// (milliseconds), so a trigger the previous owner may still fire before it
/// sees the new assignment is not fired twice.
pub fn due_watches_where(svc: &WatcherService, now: i64, grace: i64, owned: &dyn Fn(&str) -> bool) -> Vec<(String, i64)> {
    let mut fired = Vec::new();
    let mut cache = svc.cache.write();
    for (id, w) in cache.iter_mut() {
        if !w.active() || !owned(id) {
            w.next_fire = None;
            continue;
        }
        if w.next_fire.is_none() {
            w.next_fire = w.schedule.next_after(w.start_time(), now + grace.max(0));
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

/// Watch executions running at once.
pub const MAX_RUNNING: usize = 64;

/// Make watch `id` due again at `scheduled` (a trigger that could not be
/// admitted): the next scheduler pass fires it with that scheduled time.
fn keep_due(svc: &WatcherService, id: &str, scheduled: i64) {
    if let Some(w) = svc.cache.write().get_mut(id) {
        w.next_fire = Some(w.next_fire.map_or(scheduled, |nf| nf.min(scheduled)));
    }
}

/// Execute the `due` watches (`(id, scheduled time)`), each on its own
/// thread (at most [`MAX_RUNNING`] at once, as the local scheduler).
pub fn fire(node: &Arc<Node>, svc: &Arc<WatcherService>, due: Vec<(String, i64)>, now: i64) {
    for (id, scheduled) in due {
        if svc.running_threads.load(Ordering::SeqCst) >= MAX_RUNNING {
            // The trigger stays due (retried on the next tick with its own
            // scheduled time) instead of being dropped (finding 202).
            tracing::debug!("watcher execution pool is saturated; watch [{id}] stays due");
            keep_due(svc, &id, scheduled);
            continue;
        }
        svc.running_threads.fetch_add(1, Ordering::SeqCst);
        let node2 = node.clone();
        let svc2 = svc.clone();
        let id2 = id.clone();
        let ok = std::thread::Builder::new().name(format!("watcher-exec-{id}")).spawn(move || {
            if let Err(e) = exec::run_scheduled(&node2, &svc2, &id2, scheduled, now) {
                tracing::warn!("watch [{id2}] execution failed: {e}");
            }
            svc2.running_threads.fetch_sub(1, Ordering::SeqCst);
        });
        if ok.is_err() {
            svc.running_threads.fetch_sub(1, Ordering::SeqCst);
            keep_due(svc, &id, scheduled);
        }
    }
}

/// The watch id of a `.triggered_watches` record id
/// (`<watch id>_<uuid>-<time>`; a watch id may itself hold `_`).
pub fn record_watch_id(record_id: &str) -> Option<&str> {
    let b = record_id.as_bytes();
    record_id.match_indices('_').rev().map(|(i, _)| i).find(|&i| {
        let uuid = record_id.get(i + 1..i + 37);
        uuid.is_some_and(|u| uuid::Uuid::parse_str(u).is_ok()) && b.get(i + 37) == Some(&b'-')
    }).map(|i| &record_id[..i])
}

/// The pending triggered executions in `.triggered_watches`: `(watch id,
/// record id, scheduled time, triggered time)`.
pub fn triggered_records(node: &Node) -> Result<Vec<(String, String, i64, i64)>> {
    let rest = remote();
    let local = node.get_index(TRIGGERED_INDEX);
    if rest.is_none() && local.is_none() {
        return Ok(Vec::new());
    }
    if let Some(svc) = &local {
        let _ = svc.refresh();
    }
    let hits = all_watch_hits(&|body: &Value| match &rest {
        Some(rest) => rest("POST", &format!("/{TRIGGERED_INDEX}/_search?expand_wildcards=all&ignore_unavailable=true"), Some(body.clone())),
        None => {
            let targets = node.resolve(TRIGGERED_INDEX, ResolveOpts { expand_hidden: true, ..Default::default() })?;
            let req = crate::search::parse_request(body, &HashMap::new())?;
            let o = crate::search::execute(&targets, &req, &search_opts(node))?;
            serde_json::from_str(&o.json).map_err(|e| EsError::internal(e.to_string()))
        }
    })?;
    let mut out = Vec::new();
    for h in hits {
        let rid = h["_id"].as_str().unwrap_or("").to_string();
        let Some(watch) = record_watch_id(&rid).map(str::to_string) else { continue };
        let ev = &h["_source"]["trigger_event"];
        let triggered = ev["triggered_time"].as_str().and_then(parse_iso).unwrap_or_else(crate::date::now_millis);
        let scheduled = ev["schedule"]["scheduled_time"].as_str().and_then(parse_iso).unwrap_or(triggered);
        out.push((watch, rid, scheduled, triggered));
    }
    Ok(out)
}

/// Execute the triggered watches a previous run left pending (ES
/// `WatcherService.reloadInner`: `findTriggeredWatches`, then
/// `executeTriggeredWatches`), for the watches `owned` says this node runs,
/// once per start (finding 211). Records of executions running here are
/// left alone; a record whose watch no longer exists is retired.
pub fn recover_triggered(node: &Arc<Node>, svc: &Arc<WatcherService>, owned: &dyn Fn(&str) -> bool) {
    let records = match triggered_records(node) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("watcher: cannot read pending triggered watches yet: {e}");
            return;
        }
    };
    svc.recovered.store(true, Ordering::SeqCst);
    for (watch, rid, scheduled, triggered) in records {
        if svc.current.lock().contains_key(&rid) || !owned(&watch) {
            continue;
        }
        if !svc.cache.read().contains_key(&watch) {
            if let Ok(t) = triggered_index(node) {
                let _ = delete_doc(&t, &rid);
            }
            continue;
        }
        let (node2, svc2) = (node.clone(), svc.clone());
        svc.running_threads.fetch_add(1, Ordering::SeqCst);
        let spawned = std::thread::Builder::new().name(format!("watcher-recover-{watch}")).spawn(move || {
            if let Err(e) = exec::run_recovered(&node2, &svc2, &watch, &rid, scheduled, triggered) {
                tracing::warn!("recovered execution of watch [{watch}] failed: {e}");
            }
            svc2.running_threads.fetch_sub(1, Ordering::SeqCst);
        });
        if spawned.is_err() {
            svc.running_threads.fetch_sub(1, Ordering::SeqCst);
            svc.recovered.store(false, Ordering::SeqCst);
        }
    }
}

/// Every watch in `.watches`, read through the cluster coordinator as the
/// watcher's internal origin (refreshed first).
pub fn load_from_cluster() -> Result<Vec<CachedWatch>> {
    let rest = remote().ok_or_else(|| EsError::internal("no cluster coordinator for watcher"))?;
    let _ = rest("POST", &format!("/{WATCHES_INDEX}/_refresh?ignore_unavailable=true"), None);
    let hits = all_watch_hits(&|body: &Value| rest("POST", &format!("/{WATCHES_INDEX}/_search?expand_wildcards=all&ignore_unavailable=true"), Some(body.clone())))?;
    let mut out = Vec::new();
    for h in hits {
        let id = h["_id"].as_str().unwrap_or("").to_string();
        match model::build_cached(&id, h["_source"].clone(), h["_version"].as_u64().unwrap_or(1), h["_seq_no"].as_i64().unwrap_or(0), h["_primary_term"].as_u64().unwrap_or(1)) {
            Ok(cw) => out.push(cw),
            Err(e) => tracing::warn!("failed to load watch [{id}]: {e}"),
        }
    }
    Ok(out)
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
        // History retention is the history template's lifecycle policy
        // (ILM), as in ES: nothing to clean up here.
        let now = crate::date::now_millis();
        if svc.manually_stopped() {
            continue;
        }
        if svc.is_loaded() && !svc.recovered.load(Ordering::SeqCst) {
            recover_triggered(&node, &svc, &|_| true);
        }
        fire(&node, &svc, due_watches(&svc, now), now);
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
        assert_eq!(history_doc_uri(".watcher-history-18", "w_1-2026"), "/.watcher-history-18/_create/w_1-2026");
        let t = history_template(".watcher-history-18", true);
        assert_eq!(t["index_patterns"], json!([".watcher-history-18*"]));
        assert_eq!(t["data_stream"], json!({"hidden": true}));
        // Retention is the lifecycle policy's (ES WatcherIndexTemplateRegistry).
        assert_eq!(t["template"]["settings"]["index.lifecycle.name"], HISTORY_POLICY);
        assert!(history_template(".watcher-history-18", false)["template"]["settings"].get("index.lifecycle.name").is_none());
        assert_eq!(history_policy()["phases"]["delete"]["min_age"], "4d");
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

    /// Finding 211: a triggered execution persisted in `.triggered_watches`
    /// but never finished (a restart) runs on recovery, under its own record,
    /// which is then retired; a record whose watch is gone is retired too.
    #[test]
    fn pending_triggered_watches_are_recovered() {
        assert_eq!(record_watch_id("my_watch_0f8fad5b-d9cb-469f-a165-70867728950e-2026-11-09T00:00:00.000000000Z"), Some("my_watch"));
        assert_eq!(record_watch_id("nouuid_here-2026"), None);
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let svc = service(&node);
        let src = model::parse_watch("w1", &json!({
            "trigger": {"schedule": {"interval": "1h"}},
            "input": {"simple": {"x": 1}},
            "actions": {"to_out": {"index": {"index": "recovered-out"}}}
        }), crate::date::now_millis(), true).unwrap();
        svc.cache.write().insert("w1".into(), model::build_cached("w1", src, 1, 0, 1).unwrap());
        let t = triggered_index(&node).unwrap();
        let rid = "w1_0f8fad5b-d9cb-469f-a165-70867728950e-2026-11-09T00:00:00.000000000Z";
        let orphan = "gone_6ba7b810-9dad-11d1-80b4-00c04fd430c8-2026-11-09T00:00:00.000000000Z";
        for id in [rid, orphan] {
            put_doc(&t, id, &json!({"trigger_event": {"type": "schedule", "triggered_time": "2026-11-09T00:00:00.000Z", "schedule": {"scheduled_time": "2026-11-09T00:00:00.000Z"}}, "state": "waiting"}), Default::default()).unwrap();
        }
        assert_eq!(triggered_records(&node).unwrap().len(), 2);
        recover_triggered(&node, &svc, &|_| true);
        assert!(svc.recovered.load(Ordering::SeqCst));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            let ran = node.get_index("recovered-out").is_some_and(|s| {
                let _ = s.refresh();
                s.doc_count().0 > 0
            });
            if ran && triggered_records(&node).unwrap().is_empty() {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "recovered execution did not run: {:?}", triggered_records(&node));
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }

    /// Finding 202: triggers the saturated pool cannot admit stay due with
    /// their scheduled time instead of being dropped.
    #[test]
    fn saturated_triggers_stay_due() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let svc = service(&node);
        let created = "2020-01-01T00:00:00.000Z";
        for i in 0..65 {
            let id = format!("w{i}");
            let cw = model::build_cached(&id, stored(json!({"daily": {"at": "02:30"}}), created), 1, 0, 1).unwrap();
            svc.cache.write().insert(id, cw);
        }
        let t = ms("2020-01-01T02:30:00Z");
        // A first pass schedules each watch's next fire time (02:30).
        assert!(due_watches(&svc, t - 60_000).is_empty());
        let due = due_watches(&svc, t);
        assert_eq!(due.len(), 65);
        // The pool is full: nothing is admitted, nothing is lost.
        svc.running_threads.store(MAX_RUNNING, Ordering::SeqCst);
        fire(&node, &svc, due.clone(), t);
        let again = due_watches(&svc, t + 1000);
        assert_eq!(again.len(), 65);
        assert!(again.iter().all(|(_, s)| *s == ms("2020-01-01T02:30:00Z")), "{again:?}");
        svc.running_threads.store(0, Ordering::SeqCst);
    }

    /// Finding 201 / 203: the scan pages past any page size and refuses an
    /// incomplete page.
    #[test]
    fn watch_scans_page_and_refuse_incomplete_results() {
        let total = WATCH_PAGE * 2 + 5;
        let pages = std::cell::Cell::new(0);
        let hits = all_watch_hits(&|body: &Value| {
            pages.set(pages.get() + 1);
            let from = body["search_after"][0].as_u64().map_or(0, |s| s as usize + 1);
            let hits: Vec<Value> = (from..(from + WATCH_PAGE).min(total)).map(|i| json!({"_id": format!("w{i}"), "sort": [i]})).collect();
            Ok(json!({"timed_out": false, "_shards": {"failed": 0}, "hits": {"hits": hits}}))
        })
        .unwrap();
        assert_eq!(hits.len(), total);
        assert_eq!(pages.get(), 3);
        let partial = all_watch_hits(&|_: &Value| Ok(json!({"timed_out": true, "_shards": {"failed": 0}, "hits": {"hits": []}})));
        assert!(partial.is_err());
        let failed = all_watch_hits(&|_: &Value| Ok(json!({"timed_out": false, "_shards": {"failed": 1}, "hits": {"hits": []}})));
        assert!(failed.is_err());
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
