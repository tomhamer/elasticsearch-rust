//! Transforms (`_transform`): pivot and latest, batch and continuous.
//!
//! On a cluster (a [`cluster::TransformCluster`] is installed) configs,
//! state/stats and checkpoints are documents in `.transform-internal-007`
//! and a started transform is a persistent task in the cluster state,
//! assigned to and fenced by the elected master (see [`cluster`]). A
//! standalone node keeps them in `<data>/_state/transforms.json`. A
//! started transform is driven by a per-node scheduler thread (holding only a `Weak<Node>`,
//! exiting when nothing is started): each trigger runs a checkpoint on a
//! worker thread, paging the source with a composite aggregation
//! (`max_page_search_size` buckets per page), converting buckets into
//! documents and bulk-indexing them (through `dest.pipeline`) into the
//! destination index, which `_start` creates with deduced mappings.
//! Continuous transforms (`sync.time`) checkpoint up to `now - delay`,
//! only when the source changed, and recompute just the changed buckets
//! (terms keys / date histogram range of the new data).

pub mod cluster;
pub mod config;
pub mod pivot;

use crate::error::{EsError, Result};
use crate::index::WriteReq;
use crate::node::{Node, ES_VERSION, NODE_ID};
use crate::shard::WriteCond;
use crate::security::owner::StoredAuthn;
use cluster::TransformCluster;
use config::{Config, Function};
use parking_lot::{Condvar, Mutex};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::{Duration, Instant};

/// Page size of the master's recovery read of the transform catalog.
const RECOVERY_PAGE: usize = 1000;

/// Every transform config and state document of the internal index (the
/// hits, with `_seq_no`/`_primary_term`), read in `page`-sized pages with
/// `search_after` on the unique `(doc_type, id)` key until a page comes back
/// empty. The caller holds `lifecycle` with an empty map, so no transform is
/// created or deleted while it pages; a stale executor may still rewrite a
/// state document, which keeps its key (and is fenced by its seq_no). A
/// read that ends with fewer documents than the first page counted is an
/// error, never a smaller catalog.
fn load_catalog(rest: &RestCall, page: usize) -> Result<Vec<Value>> {
    let uri = format!("/{}/_search?ignore_unavailable=true&allow_no_indices=true&allow_partial_search_results=false", cluster::INTERNAL_INDEX);
    let mut out: Vec<Value> = Vec::new();
    let mut after: Option<Value> = None;
    let mut expected: Option<u64> = None;
    loop {
        let mut body = json!({
            "size": page,
            "seq_no_primary_term": true,
            "track_total_hits": true,
            "sort": [{"doc_type": "asc"}, {"id": "asc"}],
            "query": {"terms": {"doc_type": [cluster::CONFIG_DOC_TYPE, cluster::STATE_DOC_TYPE]}}
        });
        if let Some(a) = &after {
            body["search_after"] = a.clone();
        }
        let resp = match rest("POST", &uri, Some(body)) {
            Ok(r) => r,
            Err(e) if e.kind == "index_not_found_exception" && out.is_empty() => return Ok(out),
            Err(e) => return Err(e),
        };
        pivot::check_complete(&resp)?;
        if expected.is_none() && resp["hits"]["total"]["relation"] == "eq" {
            expected = resp["hits"]["total"]["value"].as_u64();
        }
        let hits = resp["hits"]["hits"].as_array().cloned().unwrap_or_default();
        let Some(last) = hits.last() else { break };
        let next = last.get("sort").cloned().filter(|s| s.as_array().is_some_and(|a| !a.is_empty()));
        let Some(next) = next else {
            return Err(EsError::new(500, "illegal_state_exception", format!("transform recovery: a [{}] page came back without sort values", cluster::INTERNAL_INDEX)));
        };
        if after.as_ref() == Some(&next) {
            return Err(EsError::new(500, "illegal_state_exception", format!("transform recovery: paging [{}] made no progress", cluster::INTERNAL_INDEX)));
        }
        after = Some(next);
        out.extend(hits);
    }
    if let Some(n) = expected {
        if (out.len() as u64) < n {
            return Err(EsError::new(503, "search_phase_execution_exception", format!("transform recovery read [{}] of [{n}] documents of [{}]; retrying", out.len(), cluster::INTERNAL_INDEX)));
        }
    }
    Ok(out)
}

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
    /// Consecutive failed triggers (ES `TransformContext.failureCount`).
    failure_count: u64,
    /// Destination id encoding of the documents written so far
    /// ([`pivot::ID_ENCODING`]; 1 for state stored before it existed).
    id_encoding: u32,
    /// Cluster: allocation id of the persistent task this node executes.
    alloc: Option<u64>,
    /// Cluster: `(_seq_no, _primary_term)` of the stored state document.
    doc_seq: Option<(u64, u64)>,
    /// Cluster: the allocation id under which this node last wrote the
    /// state document itself (its `doc_seq` is then its own write). Under a
    /// new allocation the first conditional write may find the document
    /// moved on by the former executor and takes it over once; after that
    /// a lost conditional write means another executor took the task.
    seq_alloc: Option<u64>,
}

impl TState {
    /// A new transform's state.
    fn fresh() -> TState {
        TState { id_encoding: pivot::ID_ENCODING, ..Default::default() }
    }

    /// The stored form (file entry `state`, internal index document `state`).
    fn to_json(&self) -> Value {
        let task_state = if self.failed.is_some() { "failed" } else if self.started { "started" } else { "stopped" };
        let indexer_state = if self.running { "indexing" } else if self.started { "started" } else { "stopped" };
        json!({
            "task_state": task_state,
            "indexer_state": indexer_state,
            "should_stop_at_checkpoint": self.stop_at_checkpoint,
            "reason": self.failed,
            "started": self.started && self.failed.is_none(),
            "checkpoint": self.checkpoint,
            "last_cp_millis": self.last_cp_millis,
            "last_upper": self.last_upper,
            "last_search_time": self.last_search_time,
            "changes_last_detected_at": self.changes_last_detected_at,
            "id_encoding": self.id_encoding,
            "failure_count": self.failure_count,
            "stats": self.stats.to_json(),
        })
    }

    /// Restore stored state (not started: who executes is decided by the
    /// caller; encoding 1 when the state predates `id_encoding`).
    fn from_json(s: &Value, stats: &Value) -> TState {
        TState {
            checkpoint: s["checkpoint"].as_u64().unwrap_or(0),
            last_cp_millis: s["last_cp_millis"].as_i64().unwrap_or(0),
            last_upper: s["last_upper"].as_i64(),
            last_search_time: s["last_search_time"].as_i64(),
            changes_last_detected_at: s["changes_last_detected_at"].as_i64(),
            stats: Stats::from_json(if stats.is_object() { stats } else { &s["stats"] }),
            id_encoding: s["id_encoding"].as_u64().unwrap_or(1) as u32,
            failure_count: s["failure_count"].as_u64().unwrap_or(0),
            next_run: crate::date::now_millis(),
            ..Default::default()
        }
    }

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
    /// Bumped (under the transform's write fence) by a forced stop, reset
    /// or delete: a worker of an older generation can no longer write.
    generation: u64,
}

/// What a running checkpoint executes under: the service epoch and the
/// transform generation it was started in.
#[derive(Clone, Copy, Debug)]
struct Run {
    epoch: u64,
    generation: u64,
}

pub struct TransformService {
    path: PathBuf,
    node: Mutex<Weak<Node>>,
    map: Mutex<BTreeMap<String, Entry>>,
    cv: Condvar,
    scheduler_alive: AtomicBool,
    upgrade_mode: AtomicBool,
    /// Per started transform: the coordinator its checkpoints run through
    /// on a cluster (captured from `_start`; a transform resumed by a new
    /// master runs through the internal `_system` coordinator).
    run_rest: Mutex<HashMap<String, RestCall>>,
    /// The cluster this node is part of (`None`: standalone, file storage).
    cluster: Mutex<Option<Arc<dyn TransformCluster>>>,
    /// Cluster: the master term the cached transforms were loaded in
    /// (`None`: not loaded; this node does not serve or execute them).
    loaded_term: Mutex<Option<u64>>,
    /// Bumped whenever this node gives up its transforms: a checkpoint
    /// started under an older epoch is fenced.
    epoch: AtomicU64,
    /// Serializes start / stop / adoption (task publication + local state).
    lifecycle: Mutex<()>,
    /// Serializes state document writes (their `if_seq_no` chain).
    persist_lock: Mutex<()>,
    /// Transforms of the master-local file that predates cluster storage,
    /// imported into the internal index by the next master load here.
    legacy: Mutex<BTreeMap<String, Entry>>,
    /// The internal index is known to exist (this load).
    internal_ready: AtomicBool,
    /// Per transform: the generation fence every destination, state and
    /// checkpoint write of a worker holds (read) and a forced stop, reset
    /// or delete takes (write) to bump the generation. Taking it write
    /// proves no older worker is writing, or ever will.
    fences: Mutex<HashMap<String, Arc<parking_lot::RwLock<u64>>>>,
}

/// Error kind of a checkpoint or write stopped because this node no longer
/// owns the transform (never reported as a transform failure).
const FENCED: &str = "transform_task_fenced_exception";

/// Error kind of a checkpoint write refused while upgrade mode halts the
/// transforms: the checkpoint stops (to run again once it is disabled),
/// it does not fail.
const HALTED: &str = "transform_upgrade_mode_halted";

fn halted_err(id: &str) -> EsError {
    EsError::new(429, HALTED, format!("transform [{id}] is halted: the Transform feature is upgrading"))
}

/// The upgrade mode document in the internal index (cluster).
const UPGRADE_MODE_DOC: &str = "transform-upgrade-mode";
const UPGRADE_MODE_DOC_TYPE: &str = "transform_upgrade_mode";

fn fenced_err(id: &str) -> EsError {
    EsError::new(409, FENCED, format!("transform [{id}] is no longer assigned to this node"))
}

fn not_master() -> EsError {
    EsError::new(503, "not_master_exception", "transforms are served and executed by the elected master; this node is not it")
}

/// ES `TransformIndexer.shouldAuditOnFinish`: checkpoints 1-9, then every
/// 10th below 100, every 100th below 1000, every 1000th after.
fn audit_checkpoint(n: u64) -> bool {
    match n {
        0..=9 => true,
        10..=99 => n % 10 == 0,
        100..=999 => n % 100 == 0,
        _ => n % 1000 == 0,
    }
}

/// ES `TransformFailureHandler`: these statuses fail the transform at once;
/// anything else (shard failures, timeouts, rejections, 5xx) is retried.
fn irrecoverable(e: &EsError) -> bool {
    matches!(e.status, 400 | 401 | 403 | 404 | 405 | 406 | 410 | 501)
}

/// Delay before retrying a failed checkpoint (ES backs off exponentially,
/// capped at an hour; never longer than an hour, never shorter than 5 s).
fn retry_delay_ms(failures: u64) -> i64 {
    let exp = failures.saturating_sub(1).min(10) as u32;
    (5_000i64 << exp).min(3_600_000)
}

/// A REST request through a cluster coordinator: `(method, uri, body)` →
/// response body (a `Value::String` body is sent as NDJSON).
pub type RestCall = Arc<dyn Fn(&str, &str, Option<Value>) -> Result<Value> + Send + Sync>;

/// Where a transform's index work runs: on a cluster through its
/// coordinator (the source is read from every shard, the destination is
/// created in the cluster metadata and filled through its primaries),
/// otherwise on the node's own engine.
#[derive(Clone, Default)]
/// The transform's REST calls (`None`: this node's engine) and, while a
/// checkpoint reads its source over a point in time, that PIT's id.
pub struct Ops(pub Option<RestCall>, pub Option<String>);

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

/// The `_aliases` actions setting up `dest.aliases` on `dest` (ES
/// `TransformIndex.setUpDestinationAliases`): a `remove` from every index
/// (`*`) for each `move_on_creation` alias, then an `add` on the
/// destination for each alias.
fn dest_alias_actions(dest: &str, aliases: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    for a in aliases {
        if let (Some(name), Some(true)) = (a.get("alias").and_then(Value::as_str), a.get("move_on_creation").and_then(Value::as_bool)) {
            out.push(json!({"remove": {"index": "*", "alias": name}}));
        }
    }
    for a in aliases {
        if let Some(name) = a.get("alias").and_then(Value::as_str) {
            out.push(json!({"add": {"index": dest, "alias": name}}));
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

/// NDJSON `_bulk` delete actions (an encoding-1 rebuild's old ids).
fn bulk_delete_body(index: &str, ids: &[String]) -> String {
    let mut out = String::new();
    for id in ids {
        out.push_str(&json!({"delete": {"_index": index, "_id": id}}).to_string());
        out.push('\n');
    }
    out
}

/// `(indexed, failed)` items of a `_bulk` response. Deletes are not counted
/// as indexed; a delete fails only with an error (a missing document is not).
fn bulk_counts(resp: &Value) -> (u64, u64) {
    let mut ok = 0;
    let mut failed = 0;
    for item in resp["items"].as_array().into_iter().flatten() {
        let Some((op, r)) = item.as_object().and_then(|o| o.iter().next()) else {
            failed += 1;
            continue;
        };
        let error = r.get("error").is_some();
        match (op.as_str(), error) {
            ("delete", false) => {}
            (_, true) => failed += 1,
            _ => ok += 1,
        }
    }
    (ok, failed)
}

impl Ops {
    /// One search over `expr` (`targets`: its local resolution).
    fn search(&self, node: &Node, expr: &str, targets: &[crate::search::Target], body: &Value) -> Result<Value> {
        match &self.0 {
            Some(rest) if self.1.is_some() => {
                // Over the checkpoint's point in time (its targets are the PIT's).
                let mut b = body.clone();
                b["pit"] = json!({"id": self.1, "keep_alive": PIT_KEEP_ALIVE});
                let r = rest("POST", "/_search?allow_partial_search_results=false", Some(b))?;
                pivot::check_complete(&r)?;
                Ok(r)
            }
            Some(rest) => {
                let r = rest("POST", &format!("/{}/_search?ignore_unavailable=true&allow_no_indices=true&allow_partial_search_results=false", path_expr(expr)), Some(body.clone()))?;
                pivot::check_complete(&r)?;
                Ok(r)
            }
            None => pivot::search(node, targets, body),
        }
    }

    /// Documents of `expr` matching `query`, in the transform's source
    /// search context: with `runtime_mappings` (the source's runtime
    /// fields, which a sync field or the source query may use) the count is
    /// a size-0 search carrying them, as every page search does.
    fn count(&self, node: &Node, expr: &str, targets: &[crate::search::Target], query: &Value, runtime_mappings: Option<&Value>) -> Result<u64> {
        if runtime_mappings.is_some() || self.1.is_some() {
            let mut body = json!({"size": 0, "track_total_hits": true, "query": query});
            if let Some(rm) = runtime_mappings {
                body["runtime_mappings"] = rm.clone();
            }
            let r = self.search(node, expr, targets, &body)?;
            return Ok(r["hits"]["total"]["value"].as_u64().or_else(|| r["hits"]["total"].as_u64()).unwrap_or(0));
        }
        match &self.0 {
            Some(rest) => {
                let r = rest("POST", &format!("/{}/_count?ignore_unavailable=true&allow_no_indices=true", path_expr(expr)), Some(json!({"query": query})))?;
                pivot::check_complete(&r)?;
                Ok(r["count"].as_u64().unwrap_or(0))
            }
            None => crate::search::count(targets, Some(query), None),
        }
    }

    /// Open the point in time a checkpoint reads its source through (on a
    /// cluster; a standalone node pins snapshot targets instead).
    /// ES `ClientTransformIndexer`: when the point in time cannot be
    /// opened, the checkpoint falls back to plain searches.
    fn open_pit(&self, expr: &str) -> Result<Option<PitGuard>> {
        let Some(rest) = &self.0 else { return Ok(None) };
        match rest("POST", &format!("/{}/_pit?keep_alive={PIT_KEEP_ALIVE}&ignore_unavailable=true&allow_no_indices=true", path_expr(expr)), None) {
            Ok(r) => match r["id"].as_str() {
                Some(id) => Ok(Some(PitGuard { rest: rest.clone(), id: id.to_string() })),
                None => {
                    tracing::warn!("transform source [{expr}]: the point in time answer has no id; falling back to normal search");
                    Ok(None)
                }
            },
            Err(e) => {
                tracing::warn!("transform source [{expr}]: failed to create a point in time reader, falling back to normal search: {}", e.reason);
                Ok(None)
            }
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

/// How long a checkpoint's point in time is kept between its searches.
const PIT_KEEP_ALIVE: &str = "5m";

/// A checkpoint's open point in time, closed when the checkpoint ends.
struct PitGuard {
    rest: RestCall,
    id: String,
}

impl Drop for PitGuard {
    fn drop(&mut self) {
        let _ = (self.rest)("DELETE", "/_pit", Some(json!({"id": self.id})));
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

/// Replace the document of every latest bucket whose runtime-sorted
/// candidates were truncated with the group's true latest document (the
/// one at the group's maximum sort value, found by its own search).
fn resolve_latest(node: &Node, ops: &Ops, cfg: &Config, src: &str, targets: &[crate::search::Target], query: &Value, page: &mut pivot::Page) -> Result<()> {
    for u in std::mem::take(&mut page.unresolved) {
        let Some(body) = pivot::latest_lookup(cfg, query, &u) else { continue };
        let resp = ops.search(node, src, targets, &body)?;
        if let Some(d) = resp["hits"]["hits"].get(0).and_then(pivot::latest_doc) {
            if let Some(slot) = page.docs.get_mut(u.pos) {
                slot.1 = d;
            }
        }
    }
    Ok(())
}

/// Who a transform runs as: the subject that created (or last updated) it.
/// ES stores the creating request's authentication with the config (the
/// `_xpack_security_authentication` header) and runs every source search
/// and destination write with it; rustic stores the same subject
/// ([`StoredAuthn`]) and runs as it, so its roles, including document- and
/// field-level security, apply to every checkpoint. An API key's transform
/// runs as the key: limited by the key's role descriptors and its owner's
/// roles together.
#[derive(Clone, Debug)]
pub struct CallerAuth {
    /// The subject the transform's work runs as.
    pub identity: StoredAuthn,
    /// The config's `authorization` (ES: `{"roles": [...]}` or `{"api_key": {...}}`).
    pub authorization: Value,
}

/// What a transform's work runs as.
#[derive(Clone, Debug)]
pub enum RunAs {
    /// The stored subject (user, run-as user or API key).
    Identity(StoredAuthn),
    /// A user name stored by an earlier build (`es-security-runas-user`).
    User(String),
}

/// The stored config key holding the subject (never returned by GET).
const HEADERS_KEY: &str = "headers";
/// ES's header for a stored authentication.
const AUTH_HEADER: &str = "_xpack_security_authentication";
const RUN_AS_HEADER: &str = "es-security-runas-user";

/// The subject stored with a config.
fn stored_run_as(config: &Value) -> Option<RunAs> {
    let h = config.get(HEADERS_KEY)?;
    if let Some(s) = h.get(AUTH_HEADER).and_then(Value::as_str).and_then(StoredAuthn::decode) {
        return Some(RunAs::Identity(s));
    }
    h.get(RUN_AS_HEADER).and_then(Value::as_str).map(|u| RunAs::User(u.to_string()))
}

/// Restricts a standalone node's source targets to what a subject may read
/// (DLS/FLS stamped on each target); installed by the HTTP layer when
/// security is enabled. A cluster runs source searches through its
/// coordinator as the subject instead.
pub type SourceAccess = Arc<dyn Fn(&Node, &RunAs, &[String], Vec<crate::search::Target>) -> Result<Vec<crate::search::Target>> + Send + Sync>;

fn source_access() -> &'static Mutex<Option<SourceAccess>> {
    static A: OnceLock<Mutex<Option<SourceAccess>>> = OnceLock::new();
    A.get_or_init(|| Mutex::new(None))
}

/// Install the standalone source access check (see [`SourceAccess`]).
pub fn set_source_access(hook: Option<SourceAccess>) {
    *source_access().lock() = hook;
}

/// `targets` as `subject` may read them (standalone; unchanged without a
/// subject or without security).
pub fn secure_targets(node: &Node, subject: Option<&RunAs>, names: &[String], targets: Vec<crate::search::Target>) -> Result<Vec<crate::search::Target>> {
    let hook = source_access().lock().clone();
    match (hook, subject) {
        (Some(h), Some(u)) => h(node, u, names, targets),
        _ => Ok(targets),
    }
}

/// One cluster maintenance pass for the transforms of `node` (called every
/// second by the cluster node): the elected master loads them from the
/// internal index and takes over their persistent tasks; any other node
/// stops executing them.
pub fn cluster_tick(node: &Arc<Node>) {
    service(node).cluster_tick();
}

impl TransformService {
    fn open(data: &Path) -> TransformService {
        let path = data.join("_state").join("transforms.json");
        let mut map = BTreeMap::new();
        if let Ok(b) = std::fs::read(&path) {
            if let Ok(Value::Object(m)) = serde_json::from_slice::<Value>(&b) {
                for (id, e) in m {
                    let s = &e["state"];
                    let mut st = TState::from_json(s, &Value::Null);
                    st.started = s["started"].as_bool().unwrap_or(false);
                    map.insert(id, Entry { config: e["config"].clone(), st, generation: 0 });
                }
            }
        }
        // Standalone upgrade mode survives a restart (its own file).
        let upgrading = std::fs::read(path.with_extension("upgrade_mode")).map(|b| b == b"true").unwrap_or(false);
        TransformService {
            path,
            node: Mutex::new(Weak::new()),
            map: Mutex::new(map),
            cv: Condvar::new(),
            scheduler_alive: AtomicBool::new(false),
            upgrade_mode: AtomicBool::new(upgrading),
            run_rest: Mutex::new(HashMap::new()),
            cluster: Mutex::new(None),
            loaded_term: Mutex::new(None),
            epoch: AtomicU64::new(0),
            lifecycle: Mutex::new(()),
            persist_lock: Mutex::new(()),
            legacy: Mutex::new(BTreeMap::new()),
            internal_ready: AtomicBool::new(false),
            fences: Mutex::new(HashMap::new()),
        }
    }

    // ------------------------------------------------------------ cluster ownership

    /// Run on a cluster: storage moves to the internal index and execution
    /// to the persistent task's owner. Transforms read from this node's old
    /// local file are imported by the first master load on this node.
    pub fn set_cluster(&self, c: Option<Arc<dyn TransformCluster>>) {
        let mut map = self.map.lock();
        if c.is_some() {
            let old = std::mem::take(&mut *map);
            let mut legacy = self.legacy.lock();
            for (id, mut e) in old {
                e.st.running = false;
                legacy.insert(id, e);
            }
            self.epoch.fetch_add(1, Ordering::SeqCst);
        }
        drop(map);
        self.run_rest.lock().clear();
        *self.cluster.lock() = c;
        *self.loaded_term.lock() = None;
        self.internal_ready.store(false, Ordering::SeqCst);
        self.cv.notify_all();
    }

    fn cluster(&self) -> Option<Arc<dyn TransformCluster>> {
        self.cluster.lock().clone()
    }

    /// On a cluster: make this node's view current before serving an API
    /// (the elected master loads the transforms of its term once; any other
    /// node refuses).
    pub fn ensure_current(self: &Arc<Self>) -> Result<()> {
        let Some(c) = self.cluster() else { return Ok(()) };
        let Some(term) = c.master_term() else {
            if self.loaded_term.lock().is_some() {
                let _g = self.lifecycle.lock();
                self.relinquish("this node is no longer the elected master");
            }
            return Err(not_master());
        };
        if *self.loaded_term.lock() == Some(term) {
            return Ok(());
        }
        let _g = self.lifecycle.lock();
        if *self.loaded_term.lock() == Some(term) {
            return Ok(());
        }
        self.relinquish("a new master term began");
        self.load_from_cluster(&c, term)
    }

    /// See [`cluster_tick`].
    pub fn cluster_tick(self: &Arc<Self>) {
        let Some(c) = self.cluster() else { return };
        if c.master_term().is_none() {
            let loaded = self.loaded_term.lock().is_some() || !self.map.lock().is_empty();
            if loaded {
                let _g = self.lifecycle.lock();
                self.relinquish("this node is no longer the elected master");
            }
            return;
        }
        if let Err(e) = self.ensure_current() {
            tracing::debug!("transforms not loaded on the elected master yet: {}", e.reason);
            return;
        }
        let _g = self.lifecycle.lock();
        if let Err(e) = self.adopt_tasks(&c) {
            tracing::debug!("transform tasks not taken over yet: {}", e.reason);
        }
    }

    /// Stop executing and serving every transform here (caller holds
    /// `lifecycle`). Running checkpoints are fenced by the epoch.
    fn relinquish(&self, why: &str) {
        let mut map = self.map.lock();
        self.epoch.fetch_add(1, Ordering::SeqCst);
        for (id, e) in map.iter() {
            if e.st.alloc.is_some() {
                tracing::info!("[{id}] transform stopped executing on this node: {why}");
            }
        }
        map.clear();
        drop(map);
        self.run_rest.lock().clear();
        *self.loaded_term.lock() = None;
        self.internal_ready.store(false, Ordering::SeqCst);
        self.cv.notify_all();
    }

    /// Load every transform from the internal index (caller holds
    /// `lifecycle`; the map is empty), import the old local file, then take
    /// over the persistent tasks.
    fn load_from_cluster(self: &Arc<Self>, c: &Arc<dyn TransformCluster>, term: u64) -> Result<()> {
        // Every config and state document, paged to the end: a capped
        // single search would drop transforms past the window, and the
        // orphan cleanup below would then remove their persistent tasks.
        let hits = load_catalog(&c.system_rest(), RECOVERY_PAGE)?;
        let mut configs: BTreeMap<String, Value> = BTreeMap::new();
        let mut states: HashMap<String, (Value, Option<(u64, u64)>)> = HashMap::new();
        for h in &hits {
            let src = &h["_source"];
            let Some(id) = src["id"].as_str() else { continue };
            match src["doc_type"].as_str() {
                Some(cluster::CONFIG_DOC_TYPE) => {
                    let mut cfg = src.clone();
                    if let Some(o) = cfg.as_object_mut() {
                        o.remove("doc_type");
                    }
                    configs.insert(id.to_string(), cfg);
                }
                Some(cluster::STATE_DOC_TYPE) => {
                    let seq = h["_seq_no"].as_u64().zip(h["_primary_term"].as_u64());
                    states.insert(id.to_string(), (src.clone(), seq));
                }
                Some(UPGRADE_MODE_DOC_TYPE) => {
                    // Upgrade mode is cluster state: a new master keeps it.
                    self.upgrade_mode.store(src["enabled"].as_bool().unwrap_or(false), Ordering::SeqCst);
                }
                _ => {}
            }
        }
        let mut map = BTreeMap::new();
        for (id, config) in configs {
            let st = match states.remove(&id) {
                Some((doc, seq)) => {
                    let s = &doc["state"];
                    let mut st = TState::from_json(s, &doc["stats"]);
                    st.doc_seq = seq;
                    // A failed task stays failed on its next owner (ES keeps
                    // the failure in the task state).
                    if s["task_state"] == "failed" {
                        st.failed = s["reason"].as_str().map(str::to_string).or_else(|| Some("task failed".into()));
                    }
                    st.stop_at_checkpoint = s["should_stop_at_checkpoint"].as_bool().unwrap_or(false);
                    st
                }
                None => TState::fresh(),
            };
            let generation = *self.fence(&id).read();
            map.insert(id, Entry { config, st, generation });
        }
        // Transforms of the master-local file this node wrote before
        // cluster storage existed: imported once (ids already in the index win).
        let legacy: Vec<(String, Value, TState)> = self.legacy.lock().iter().filter(|(id, _)| !map.contains_key(*id)).map(|(id, e)| (id.clone(), e.config.clone(), e.st.clone())).collect();
        let had_legacy = !self.legacy.lock().is_empty();
        let mut imported_started = Vec::new();
        let mut import_ids = Vec::new();
        for (id, config, mut st) in legacy {
            if st.started {
                imported_started.push(id.clone());
            }
            st.started = false;
            st.alloc = None;
            st.doc_seq = None;
            import_ids.push(id.clone());
            let generation = *self.fence(&id).read();
            map.insert(id, Entry { config, st, generation });
        }
        *self.map.lock() = map;
        *self.loaded_term.lock() = Some(term);
        if had_legacy {
            for id in &import_ids {
                self.save_config(id)?;
                self.save_state(id)?;
            }
            let me = c.local_node();
            for id in &imported_started {
                let (freq, idc, mec) = (self.map.lock().get(id).and_then(|e| e.config["frequency"].as_str().map(str::to_string)), id.clone(), me.clone());
                c.update_persistent_tasks(Arc::new(move |cur| cluster::create_task(cur, &idc, cluster::task_params(&idc, freq.as_deref(), None), &mec, term).map(|(v, _)| v)))?;
            }
            self.legacy.lock().clear();
            let _ = std::fs::rename(&self.path, self.path.with_extension("json.imported"));
            tracing::info!("imported {} transform(s) from {} into {}", import_ids.len(), self.path.display(), cluster::INTERNAL_INDEX);
        }
        self.adopt_tasks(c)
    }

    /// Take over the transform tasks of the committed cluster state (caller
    /// holds `lifecycle`): tasks assigned elsewhere are re-assigned here with
    /// new allocation ids, tasks without a config are removed, and local
    /// execution follows exactly the tasks assigned to this node.
    fn adopt_tasks(self: &Arc<Self>, c: &Arc<dyn TransformCluster>) -> Result<()> {
        let Some(term) = c.master_term() else { return Ok(()) };
        if *self.loaded_term.lock() != Some(term) {
            return Ok(());
        }
        let me = c.local_node();
        let mut custom = c.persistent_tasks();
        let (orphans, foreign): (Vec<String>, Vec<String>) = {
            let map = self.map.lock();
            let tasks = cluster::transform_tasks(&custom);
            let orphans = tasks.iter().filter(|(id, _)| !map.contains_key(id)).map(|(id, _)| id.clone()).collect();
            let foreign = tasks.iter().filter(|(id, t)| map.contains_key(id) && t.executor.as_deref() != Some(me.as_str())).map(|(id, _)| id.clone()).collect();
            (orphans, foreign)
        };
        if !orphans.is_empty() || !foreign.is_empty() {
            let mec = me.clone();
            custom = c.update_persistent_tasks(Arc::new(move |cur| {
                let mut v = cur.clone();
                let mut changed = false;
                for id in &orphans {
                    if let Some(n) = cluster::remove_task(&v, id) {
                        v = n;
                        changed = true;
                    }
                }
                if let Some(n) = cluster::reassign(&v, &foreign, &mec, term) {
                    v = n;
                    changed = true;
                }
                changed.then_some(v)
            }))?;
        }
        let now = crate::date::now_millis();
        let mut adopted = false;
        let mut map = self.map.lock();
        for (id, e) in map.iter_mut() {
            match cluster::task_of(&custom, id) {
                Some(t) if t.executor.as_deref() == Some(me.as_str()) => {
                    if e.st.alloc != Some(t.allocation_id) {
                        tracing::info!("[{id}] transform task assigned to this node (allocation id [{}], term [{term}]); resuming after checkpoint [{}]", t.allocation_id, e.st.checkpoint);
                        e.st.alloc = Some(t.allocation_id);
                        e.st.started = true;
                        e.st.running = false;
                        e.st.stop_requested = false;
                        e.st.next_run = now;
                        adopted = true;
                    }
                }
                _ => {
                    if e.st.alloc.is_some() || e.st.started {
                        e.st.started = false;
                        e.st.alloc = None;
                    }
                }
            }
        }
        drop(map);
        if adopted {
            self.ensure_scheduler();
        }
        self.cv.notify_all();
        Ok(())
    }

    /// Whether the checkpoint started under `epoch` must stop: this node
    /// gave the transforms up, is no longer the elected master, or the task
    /// is no longer assigned here with the allocation id it executes under.
    fn fenced(&self, id: &str, epoch: u64) -> bool {
        if self.epoch.load(Ordering::SeqCst) != epoch {
            return true;
        }
        if self.cluster().is_none() {
            return false;
        }
        !self.owns(id)
    }

    /// Cluster: this node is the elected master of the term it loaded and
    /// the latest applied cluster state assigns `id`'s persistent task to
    /// it under the allocation id it executes.
    fn owns(&self, id: &str) -> bool {
        let Some(c) = self.cluster() else { return true };
        let Some(term) = c.master_term() else { return false };
        if *self.loaded_term.lock() != Some(term) {
            return false;
        }
        let alloc = self.map.lock().get(id).and_then(|e| e.st.alloc);
        let me = c.local_node();
        match (alloc, cluster::task_of(&c.persistent_tasks(), id)) {
            (Some(a), Some(t)) => t.allocation_id == a && t.executor.as_deref() == Some(me.as_str()),
            _ => false,
        }
    }

    /// Remove `id`'s persistent task (cluster; caller holds `lifecycle`) and
    /// forget its allocation.
    fn remove_task(&self, id: &str) -> Result<()> {
        if let Some(c) = self.cluster() {
            let idc = id.to_string();
            c.update_persistent_tasks(Arc::new(move |cur| cluster::remove_task(cur, &idc)))?;
        }
        if let Some(e) = self.map.lock().get_mut(id) {
            e.st.alloc = None;
        }
        Ok(())
    }

    // ------------------------------------------------------------ storage

    /// Standalone: the whole map in the local file.
    fn persist_file(&self) -> Result<()> {
        let map = self.map.lock();
        let mut out = Map::new();
        for (id, e) in map.iter() {
            out.insert(id.clone(), json!({"config": e.config, "state": e.st.to_json()}));
        }
        drop(map);
        if let Some(d) = self.path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(&Value::Object(out))?)?;
        std::fs::rename(tmp, &self.path)?;
        Ok(())
    }

    fn ensure_internal_index(&self, rest: &RestCall) -> Result<()> {
        if self.internal_ready.load(Ordering::SeqCst) {
            return Ok(());
        }
        match rest("PUT", &format!("/{}", cluster::INTERNAL_INDEX), Some(cluster::internal_index_body())) {
            Ok(_) => {}
            Err(e) if e.kind == "resource_already_exists_exception" => {}
            Err(e) => return Err(e),
        }
        self.internal_ready.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn doc_uri(id: &str) -> String {
        format!("/{}/_doc/{}", cluster::INTERNAL_INDEX, path_expr(id))
    }

    /// Persist `id`'s config (durable before the API answers).
    fn save_config(&self, id: &str) -> Result<()> {
        let Some(c) = self.cluster() else { return self.persist_file() };
        let mut doc = self.map.lock().get(id).map(|e| e.config.clone()).ok_or_else(|| not_found(id))?;
        if let Some(o) = doc.as_object_mut() {
            o.insert("id".into(), json!(id));
            o.insert("doc_type".into(), json!(cluster::CONFIG_DOC_TYPE));
        }
        let rest = c.system_rest();
        self.ensure_internal_index(&rest)?;
        rest("PUT", &format!("{}?refresh=true", Self::doc_uri(&cluster::config_doc_id(id))), Some(doc))?;
        Ok(())
    }

    /// Persist `id`'s state. On a cluster the write is fenced: it must
    /// follow the state document version this node last read or wrote (a
    /// stale executor's write fails with a version conflict).
    fn save_state(&self, id: &str) -> Result<()> {
        let Some(c) = self.cluster() else { return self.persist_file() };
        let _g = self.persist_lock.lock();
        let (doc, seq) = {
            let map = self.map.lock();
            let Some(e) = map.get(id) else { return Ok(()) };
            (json!({"id": id, "doc_type": cluster::STATE_DOC_TYPE, "state": e.st.to_json(), "stats": e.st.stats.to_json()}), e.st.doc_seq)
        };
        let rest = c.system_rest();
        self.ensure_internal_index(&rest)?;
        let base = Self::doc_uri(&cluster::state_doc_id(id));
        let r = match seq {
            None => rest("PUT", &format!("{base}?refresh=true"), Some(doc))?,
            Some((s, t)) => match rest("PUT", &format!("{base}?refresh=true&if_seq_no={s}&if_primary_term={t}"), Some(doc.clone())) {
                Ok(r) => r,
                Err(e) if e.status == 409 => {
                    // Gone (a wiped index) differs from written by another owner.
                    let current = rest("GET", &base, None);
                    let missing = match &current {
                        Ok(v) => v["found"] == false,
                        Err(e) => e.kind == "index_not_found_exception",
                    };
                    if !missing {
                        // The first write under a newly assigned allocation
                        // takes the document over from the former executor
                        // (which may have written after this node loaded
                        // it), once, conditionally on what it read; the
                        // former executor's next conditional write then
                        // fails and fences it. Any other lost write means
                        // this node no longer executes the transform.
                        let first = self.map.lock().get(id).is_some_and(|e| e.st.alloc.is_some() && e.st.seq_alloc != e.st.alloc);
                        let seen = current.as_ref().ok().and_then(|v| v["_seq_no"].as_u64().zip(v["_primary_term"].as_u64()));
                        match (first && self.owns(id), seen) {
                            (true, Some((s2, t2))) => match rest("PUT", &format!("{base}?refresh=true&if_seq_no={s2}&if_primary_term={t2}"), Some(doc)) {
                                Ok(r) => r,
                                Err(e) if e.status == 409 => return Err(fenced_err(id)),
                                Err(e) => return Err(e),
                            },
                            _ => return Err(fenced_err(id)),
                        }
                    } else {
                        self.internal_ready.store(false, Ordering::SeqCst);
                        self.ensure_internal_index(&rest)?;
                        rest("PUT", &format!("{base}?refresh=true"), Some(doc))?
                    }
                }
                Err(e) => return Err(e),
            },
        };
        if let (Some(s), Some(t)) = (r["_seq_no"].as_u64(), r["_primary_term"].as_u64()) {
            if let Some(e) = self.map.lock().get_mut(id) {
                e.st.doc_seq = Some((s, t));
                e.st.seq_alloc = e.st.alloc;
            }
        }
        Ok(())
    }

    /// Persist a completed checkpoint (cluster: `data_frame_transform_checkpoint` document).
    fn save_checkpoint(&self, id: &str, checkpoint: u64, at: i64, upper: Option<i64>) -> Result<()> {
        let Some(c) = self.cluster() else { return Ok(()) };
        let rest = c.system_rest();
        self.ensure_internal_index(&rest)?;
        let doc = json!({"transform_id": id, "doc_type": cluster::CHECKPOINT_DOC_TYPE, "checkpoint": checkpoint, "timestamp_millis": at, "time_upper_bound_millis": upper});
        rest("PUT", &format!("{}?refresh=true", Self::doc_uri(&cluster::checkpoint_doc_id(id, checkpoint))), Some(doc))?;
        Ok(())
    }

    /// Delete `id`'s stored documents (cluster) or rewrite the file.
    fn delete_stored(&self, id: &str) -> Result<()> {
        let Some(c) = self.cluster() else { return self.persist_file() };
        let rest = c.system_rest();
        for doc in [cluster::config_doc_id(id), cluster::state_doc_id(id)] {
            match rest("DELETE", &format!("{}?refresh=true", Self::doc_uri(&doc)), None) {
                Ok(_) => {}
                Err(e) if e.status == 404 => {}
                Err(e) => return Err(e),
            }
        }
        self.delete_checkpoints(&rest, id)
    }

    fn delete_checkpoints(&self, rest: &RestCall, id: &str) -> Result<()> {
        let body = json!({"query": {"bool": {"filter": [{"term": {"transform_id": id}}, {"term": {"doc_type": cluster::CHECKPOINT_DOC_TYPE}}]}}});
        match rest("POST", &format!("/{}/_delete_by_query?refresh=true&conflicts=proceed&ignore_unavailable=true&allow_no_indices=true", cluster::INTERNAL_INDEX), Some(body)) {
            Ok(_) => Ok(()),
            Err(e) if e.status == 404 => Ok(()),
            Err(e) => Err(e),
        }
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
    fn validate_env(&self, node: &Node, ops: &Ops, cfg: &Config, check_dest: bool, run_as: Option<&RunAs>) -> Result<()> {
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
            let targets = if ops.0.is_none() { secure_targets(node, run_as, &cfg.source_indices, targets)? } else { targets };
            let body = pivot::page_body(cfg, &cfg.query, 1, None);
            ops.search(node, &cfg.source_indices.join(","), &targets, &body).map_err(|e| {
                if e.status >= 500 || e.status == 403 {
                    e
                } else {
                    EsError::new(400, "status_exception", "Failed to test query, received status: BAD_REQUEST").caused_by(e)
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

    /// The stored config with the caller's identity (or the previous one).
    fn with_auth(mut stored: Value, auth: Option<&CallerAuth>, previous: Option<&Value>) -> Value {
        if let Some(o) = stored.as_object_mut() {
            match auth {
                Some(a) => {
                    o.insert("authorization".into(), a.authorization.clone());
                    o.insert(HEADERS_KEY.into(), json!({AUTH_HEADER: a.identity.encode()}));
                }
                None => {
                    if let Some(p) = previous {
                        for k in ["authorization", HEADERS_KEY] {
                            if let Some(v) = p.get(k) {
                                o.insert(k.into(), v.clone());
                            }
                        }
                    }
                }
            }
        }
        stored
    }

    // ------------------------------------------------------------ CRUD

    /// `PUT /_transform/{id}`; returns deprecation-style warnings.
    pub fn put(&self, node: &Node, ops: &Ops, id: &str, body: &Value, defer: bool, auth: Option<&CallerAuth>) -> Result<Vec<String>> {
        config::validate_id(id)?;
        let cfg = config::parse(Some(id), body, true, true)?;
        if self.map.lock().contains_key(id) {
            return Err(EsError::new(409, "resource_already_exists_exception", format!("Transform with id [{id}] already exists")));
        }
        let run_as = auth.map(|a| RunAs::Identity(a.identity.clone()));
        if !defer {
            self.validate_env(node, ops, &cfg, true, run_as.as_ref())?;
        } else if let Some(d) = &cfg.dest_index {
            if let Some(m) = config::validate_dest_name(d) {
                return Err(EsError::action_validation(&[&m]));
            }
        }
        let stored = Self::with_auth(config::normalize(body, &cfg, ES_VERSION, Some(crate::date::now_millis())), auth, None);
        let generation = *self.fence(id).read();
        {
            let mut map = self.map.lock();
            if map.contains_key(id) {
                return Err(EsError::new(409, "resource_already_exists_exception", format!("Transform with id [{id}] already exists")));
            }
            map.insert(id.to_string(), Entry { config: stored, st: TState::fresh(), generation });
        }
        let saved = self.save_config(id).and_then(|_| self.save_state(id));
        if let Err(e) = saved {
            self.map.lock().remove(id);
            return Err(e);
        }
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
                if let Some(o) = c.as_object_mut() {
                    o.remove(HEADERS_KEY);
                    if exclude_generated {
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
    pub fn reset_all(self: &Arc<Self>) -> Result<()> {
        let ids: Vec<String> = self.map.lock().keys().cloned().collect();
        for id in &ids {
            self.stop_one(id, true, false, true, Duration::from_secs(30))?;
        }
        if let Some(c) = self.cluster() {
            // The internal index was deleted with the feature's indices;
            // the tasks of transforms not loaded here go too.
            let _g = self.lifecycle.lock();
            c.update_persistent_tasks(Arc::new(|cur| cluster::remove_all(cur)))?;
            self.map.lock().clear();
            self.run_rest.lock().clear();
            self.internal_ready.store(false, Ordering::SeqCst);
            return Ok(());
        }
        self.map.lock().clear();
        self.persist_file()
    }

    /// `DELETE /_transform/{id}`.
    pub fn delete(self: &Arc<Self>, node: &Node, ops: &Ops, id: &str, force: bool, delete_dest: bool) -> Result<()> {
        let dest = {
            let map = self.map.lock();
            let e = map.get(id).ok_or_else(|| not_found(id))?;
            if e.st.started && !force {
                return Err(status_err(409, format!("Cannot delete transform [{id}] as the task is running. Stop the task first")));
            }
            e.config["dest"]["index"].as_str().map(str::to_string)
        };
        if force {
            self.stop_one(id, true, false, true, Duration::from_secs(30))?;
        }
        self.disown(id);
        if delete_dest {
            if let Some(d) = dest {
                if node.get_index(&d).is_some() {
                    ops.delete_index(node, &d)?;
                }
            }
        }
        {
            // A task left behind (a stop racing a completing batch) goes too.
            let _g = self.lifecycle.lock();
            self.remove_task(id)?;
        }
        self.run_rest.lock().remove(id);
        let _g = self.persist_lock.lock();
        self.map.lock().remove(id);
        drop(_g);
        self.delete_stored(id)?;
        // The fence's generation outlives the entry: a worker of the deleted
        // transform never matches a transform re-created with its id.
        Ok(())
    }

    /// `POST /_transform/{id}/_update`; returns the updated config.
    pub fn update(&self, node: &Node, ops: &Ops, id: &str, body: &Value, defer: bool, auth: Option<&CallerAuth>) -> Result<(Value, Vec<String>)> {
        let (raw, old) = self.config_of(id)?;
        let upd = body.as_object().ok_or_else(|| EsError::parsing("request body is required"))?;
        for k in upd.keys() {
            if !matches!(k.as_str(), "source" | "dest" | "frequency" | "sync" | "description" | "settings" | "_meta" | "retention_policy") {
                return Err(EsError::x_content_parse(format!("[1:1] [data_frame_transform_config_update] unknown field [{k}]")));
            }
        }
        let mut merged = config::strip_generated(&raw);
        let mo = merged.as_object_mut().unwrap();
        mo.remove(HEADERS_KEY);
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
            let run_as = auth.map(|a| RunAs::Identity(a.identity.clone())).or_else(|| stored_run_as(&raw));
            self.validate_env(node, ops, &cfg, check_dest, run_as.as_ref())?;
        } else if let Some(d) = &cfg.dest_index {
            if let Some(m) = config::validate_dest_name(d) {
                return Err(EsError::action_validation(&[&m]));
            }
        }
        let create_time = raw.get("create_time").and_then(Value::as_i64);
        // ES stores the updating user's headers: the transform now runs as them.
        let stored = Self::with_auth(config::normalize(&merged, &cfg, ES_VERSION, create_time), auth, Some(&raw));
        {
            let mut map = self.map.lock();
            if let Some(e) = map.get_mut(id) {
                e.config = stored.clone();
            }
        }
        self.save_config(id)?;
        let mut shown = stored;
        if let Some(o) = shown.as_object_mut() {
            o.remove(HEADERS_KEY);
        }
        Ok((shown, config::runtime_warnings(&cfg)))
    }

    /// `POST /_transform/_upgrade`.
    pub fn upgrade(&self) -> Value {
        json!({"updated": 0, "no_action": self.map.lock().len(), "needs_update": 0})
    }

    /// `POST _transform/set_upgrade_mode` (#568): durable (the internal
    /// index on a cluster, a file standalone), and when enabling, answered
    /// only once no checkpoint write can follow: every write of a running
    /// checkpoint holds its transform's fence and checks the mode under it,
    /// so taking each fence once waits out the writes in flight, and later
    /// ones are refused (the checkpoint halts, to run again once the mode
    /// is disabled). Starts are refused while it is enabled.
    pub fn set_upgrade_mode(&self, enabled: bool) -> Result<()> {
        match self.cluster() {
            Some(c) => {
                let rest = c.system_rest();
                self.ensure_internal_index(&rest)?;
                rest("PUT", &format!("{}?refresh=true", Self::doc_uri(UPGRADE_MODE_DOC)), Some(json!({"id": UPGRADE_MODE_DOC, "doc_type": UPGRADE_MODE_DOC_TYPE, "enabled": enabled})))?;
            }
            None => {
                let f = self.path.with_extension("upgrade_mode");
                if let Some(d) = f.parent() {
                    std::fs::create_dir_all(d)?;
                }
                std::fs::write(&f, if enabled { "true" } else { "false" })?;
            }
        }
        self.upgrade_mode.store(enabled, Ordering::SeqCst);
        if enabled {
            let fences: Vec<_> = self.fences.lock().values().cloned().collect();
            for f in fences {
                drop(f.write());
            }
        }
        self.cv.notify_all();
        Ok(())
    }

    /// Whether upgrade mode halts the transforms.
    pub fn upgrade_mode(&self) -> bool {
        self.upgrade_mode.load(Ordering::SeqCst)
    }

    /// The stored config of `id` (for privilege checks by the HTTP layer).
    pub fn stored_config(&self, id: &str) -> Option<Value> {
        self.map.lock().get(id).map(|e| e.config.clone())
    }

    // ------------------------------------------------------------ preview

    /// `POST /_transform[/{id}]/_preview` (`run_as`: the caller, whose read
    /// access the preview runs with on a standalone node).
    pub fn preview(&self, node: &Node, ops: &Ops, id: Option<&str>, body: &Value, run_as: Option<&RunAs>) -> Result<(Value, Vec<String>)> {
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
            let targets = if ops.0.is_none() { secure_targets(node, run_as, &cfg.source_indices, targets)? } else { targets };
            let q = if let Some((f, delay)) = &cfg.sync {
                let upper = crate::date::now_millis() - delay;
                json!({"bool": {"filter": [cfg.query.clone(), {"range": {f: {"lt": upper, "format": "epoch_millis"}}}]}})
            } else {
                cfg.query.clone()
            };
            let body = pivot::page_body(&cfg, &q, PREVIEW_SIZE, None);
            let resp = ops.search(node, &cfg.source_indices.join(","), &targets, &body)?;
            let mut page = pivot::convert_page(&cfg, &resp, &types)?;
            resolve_latest(node, ops, &cfg, &cfg.source_indices.join(","), &targets, &q, &mut page)?;
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
        if self.upgrade_mode() {
            return Err(EsError::new(429, "status_exception", "Cannot start any Transform while the Transform feature is upgrading."));
        }
        let (raw, cfg) = self.config_of(id)?;
        let mut from_millis = None;
        if let Some(f) = from {
            if !cfg.is_continuous() {
                return Err(EsError::action_validation(&["[from] parameter is currently not supported for batch (non-continuous) transforms"]));
            }
            let fmt = crate::date::DateFormatter::default();
            match crate::date::parse_date_math(f, &fmt, crate::date::Tz::UTC, crate::date::now_millis(), false) {
                Ok(v) if !f.is_empty() => from_millis = Some(v),
                _ => return Err(EsError::new(400, "parse_exception", "Failed to parse date for [from]")),
            }
        }
        let first_run = {
            let map = self.map.lock();
            let e = map.get(id).ok_or_else(|| not_found(id))?;
            if e.st.started {
                return Err(status_err(409, format!("Cannot start transform [{id}] as it is already started.")));
            }
            e.st.checkpoint == 0
        };
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
            // ES `TransformIndex.setUpDestinationAliases`: when the transform
            // sets up its destination (before its first checkpoint), the
            // `dest.aliases` point at it whether the transform created the
            // index or the user did; `move_on_creation` first removes every
            // other index from that alias (without deleting it). One
            // `_aliases` request, so the move is atomic.
            if first_run && !cfg.dest_aliases.is_empty() {
                let actions = dest_alias_actions(&dest, &cfg.dest_aliases);
                match &ops.0 {
                    Some(rest) => {
                        rest("POST", "/_aliases", Some(json!({"actions": actions})))?;
                    }
                    None => {
                        node.update_aliases(&actions)?;
                    }
                }
            }
        }
        let continuous = cfg.is_continuous();
        let lifecycle = self.lifecycle.lock();
        let alloc = match self.cluster() {
            Some(c) => {
                let term = c.master_term().ok_or_else(not_master)?;
                let me = c.local_node();
                let params = cluster::task_params(id, raw["frequency"].as_str(), from_millis);
                let cell: Arc<Mutex<Option<u64>>> = Arc::new(Mutex::new(None));
                let (idc, out) = (id.to_string(), cell.clone());
                c.update_persistent_tasks(Arc::new(move |cur| {
                    let (v, a) = cluster::create_task(cur, &idc, params.clone(), &me, term)?;
                    *out.lock() = Some(a);
                    Some(v)
                }))?;
                let a = cell.lock().take();
                Some(a.ok_or_else(|| status_err(409, format!("Cannot start transform [{id}] as it is already started.")))?)
            }
            None => {
                match &ops.0 {
                    Some(rest) => self.run_rest.lock().insert(id.to_string(), rest.clone()),
                    None => self.run_rest.lock().remove(id),
                };
                None
            }
        };
        {
            let mut map = self.map.lock();
            let e = map.get_mut(id).ok_or_else(|| not_found(id))?;
            let now = crate::date::now_millis();
            e.st.started = true;
            e.st.alloc = alloc;
            e.st.failed = None;
            e.st.failure_count = 0;
            e.st.stop_requested = false;
            e.st.stop_at_checkpoint = false;
            e.st.health_issue = if targets.is_empty() && unattended { Some("source index missing".into()) } else { None };
            e.st.next_run = if continuous { now } else { now + BATCH_START_DELAY_MS };
            if let (Some(f), true) = (from_millis, continuous) {
                if e.st.checkpoint == 0 {
                    e.st.last_upper = Some(f);
                }
            }
        }
        // Durable before the API answers; a failed write undoes the start.
        if let Err(err) = self.save_state(id) {
            if let Some(e) = self.map.lock().get_mut(id) {
                e.st.started = false;
            }
            let _ = self.remove_task(id);
            return Err(err);
        }
        drop(lifecycle);
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

    /// Stop one transform; waits (with `wait`) until it stopped executing
    /// and its task is gone, and reports a timeout (ES answers 408). A
    /// forced stop disowns a running worker at once (see [`Self::disown`]):
    /// whatever it is blocked on, it can no longer write.
    fn stop_one(&self, id: &str, force: bool, wait_for_checkpoint: bool, wait: bool, timeout: Duration) -> Result<()> {
        {
            let _g = self.lifecycle.lock();
            let immediate = {
                let mut map = self.map.lock();
                let Some(e) = map.get_mut(id) else { return Ok(()) };
                if !e.st.started {
                    let running = e.st.running;
                    drop(map);
                    if force && running {
                        self.disown(id);
                    }
                    return Ok(());
                }
                let immediate = force || !e.st.running;
                if immediate {
                    e.st.started = false;
                    e.st.failed = None;
                }
                e.st.stop_requested = true;
                e.st.stop_at_checkpoint = wait_for_checkpoint;
                immediate
            };
            if force {
                self.disown(id);
            }
            if immediate {
                self.remove_task(id)?;
            }
            self.save_state(id)?;
        }
        self.cv.notify_all();
        if wait {
            let deadline = Instant::now() + timeout;
            let mut map = self.map.lock();
            loop {
                let busy = map.get(id).map(|e| e.st.running || e.st.started || e.st.alloc.is_some()).unwrap_or(false);
                if !busy {
                    break;
                }
                if Instant::now() >= deadline {
                    return Err(EsError::new(408, "status_exception", format!("Timed out after [{}] while waiting for transform [{id}] to stop", {
                        let ms = timeout.as_millis();
                        if ms % 1000 == 0 { format!("{}s", ms / 1000) } else { format!("{ms}ms") }
                    })));
                }
                self.cv.wait_for(&mut map, Duration::from_millis(50));
            }
        }
        Ok(())
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
            self.stop_one(id, force, wait_for_checkpoint, wait, timeout)?;
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
            self.stop_one(id, true, false, true, Duration::from_secs(30))?;
        }
        // No worker of this transform writes from here on, even one still
        // blocked in a search (it is fenced by the new generation).
        self.disown(id);
        if let Some(d) = &cfg.dest_index {
            if let Some(svc) = node.get_index(d) {
                let created_by_us = svc.meta().mappings.to_json().get("_meta").and_then(|m| m.get("created_by")).and_then(Value::as_str) == Some("transform");
                if created_by_us {
                    ops.delete_index(node, d)?;
                }
            }
        }
        {
            let mut map = self.map.lock();
            if let Some(e) = map.get_mut(id) {
                let doc_seq = e.st.doc_seq;
                e.st = TState::fresh();
                e.st.doc_seq = doc_seq;
            }
        }
        if let Some(c) = self.cluster() {
            self.delete_checkpoints(&c.system_rest(), id)?;
        }
        self.save_state(id)
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
            let node_id = self.cluster().map(|c| c.local_node()).unwrap_or_else(|| NODE_ID.to_string());
            m.insert(
                "node".into(),
                json!({"id": node_id, "name": node.name, "ephemeral_id": node_id, "transport_address": "127.0.0.1:9300", "attributes": {}}),
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
            json!({"status": "yellow", "issues": [{"type": "transform_indexer_failed", "issue": "Transform indexer failed", "details": i, "count": st.failure_count.max(1)}]})
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
        let node_id = self.cluster().map(|c| c.local_node()).unwrap_or_else(|| NODE_ID.to_string());
        json!({node_id: {"scheduler": {"registered_transform_count": n, "peek_transform": peek}}, "total": {"scheduler": {"registered_transform_count": n}}})
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
            let Some(node) = node else {
                self.scheduler_alive.store(false, Ordering::SeqCst);
                break;
            };
            let now = crate::date::now_millis();
            let on_cluster = self.cluster.lock().is_some();
            let mut due = Vec::new();
            let epoch;
            {
                let mut map = self.map.lock();
                epoch = self.epoch.load(Ordering::SeqCst);
                if !map.values().any(|e| e.st.started || e.st.running) {
                    self.scheduler_alive.store(false, Ordering::SeqCst);
                    break;
                }
                if !self.upgrade_mode.load(Ordering::SeqCst) {
                    for (id, e) in map.iter_mut() {
                        // On a cluster only a task assigned here executes.
                        let owned = !on_cluster || e.st.alloc.is_some();
                        if owned && e.st.started && !e.st.running && e.st.failed.is_none() && e.st.next_run <= now && !e.st.stop_requested {
                            e.st.running = true;
                            due.push((id.clone(), e.generation));
                        }
                    }
                }
                if due.is_empty() {
                    self.cv.wait_for(&mut map, Duration::from_millis(100));
                }
            }
            for (id, generation) in due {
                let svc = self.clone();
                let n = node.clone();
                let run = Run { epoch, generation };
                let r = std::thread::Builder::new().name(format!("transform-{id}")).spawn(move || svc.run_trigger(&n, &id, run));
                if r.is_err() {
                    break;
                }
            }
            drop(node);
        }
    }

    /// Update `id`'s state when nothing gave the transforms up since
    /// `epoch` (a fenced checkpoint never touches its successor's state).
    fn update_state<F: FnOnce(&mut TState)>(&self, id: &str, run: Run, f: F) -> bool {
        let mut map = self.map.lock();
        if self.epoch.load(Ordering::SeqCst) != run.epoch {
            return false;
        }
        match map.get_mut(id) {
            Some(e) if e.generation == run.generation => {
                f(&mut e.st);
                true
            }
            _ => false,
        }
    }

    fn should_abort(&self, id: &str, run: Run) -> bool {
        let map = self.map.lock();
        if self.epoch.load(Ordering::SeqCst) != run.epoch {
            return true;
        }
        match map.get(id) {
            Some(e) => e.generation != run.generation || (e.st.stop_requested && !e.st.stop_at_checkpoint) || !e.st.started,
            None => true,
        }
    }

    fn fence(&self, id: &str) -> Arc<parking_lot::RwLock<u64>> {
        self.fences.lock().entry(id.to_string()).or_insert_with(|| Arc::new(parking_lot::RwLock::new(0))).clone()
    }

    /// Run a write of the checkpoint `run` while holding `id`'s fence, only
    /// if `run` is still the transform's current generation and owner.
    fn fenced_write<T>(&self, id: &str, run: Run, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let fence = self.fence(id);
        let g = fence.read();
        let current = *g == run.generation && self.map.lock().get(id).is_some_and(|e| e.generation == run.generation);
        if !current || self.fenced(id, run.epoch) {
            return Err(fenced_err(id));
        }
        if self.upgrade_mode() {
            return Err(halted_err(id));
        }
        let r = f();
        drop(g);
        r
    }

    /// A destination mutation of the checkpoint `run` (a page's bulk with
    /// its rebuild deletes, the retention delete-by-query), fenced as ES
    /// fences a transform executor: under the generation fence, and on a
    /// cluster only after proving, immediately before dispatch, that this
    /// node still executes the task: the latest applied cluster state
    /// assigns the task here under this allocation id, and the conditional
    /// (`if_seq_no` / `if_primary_term`) state document write succeeds,
    /// then the assignment is checked once more. A former executor (the
    /// task re-assigned, or a new owner's state write in between) stops
    /// here: it mutates nothing further and never completes the checkpoint.
    fn fenced_mutation<T>(&self, id: &str, run: Run, f: impl FnOnce() -> Result<T>) -> Result<T> {
        self.fenced_write(id, run, || {
            if self.cluster().is_some() {
                self.save_state(id)?;
                if self.fenced(id, run.epoch) {
                    return Err(fenced_err(id));
                }
            }
            f()
        })
    }

    /// Disown every worker of `id` (forced stop, reset, delete): waits for
    /// any write in progress, then bumps the generation so no older worker
    /// writes again. Returns once that is proven.
    fn disown(&self, id: &str) {
        let fence = self.fence(id);
        let mut g = fence.write();
        let mut map = self.map.lock();
        if let Some(e) = map.get_mut(id) {
            e.generation += 1;
            *g = e.generation;
            e.st.running = false;
            e.st.progress = None;
        } else {
            *g += 1;
        }
        drop(map);
        drop(g);
        self.cv.notify_all();
    }

    /// One trigger: detect changes and maybe run a checkpoint, then record
    /// the outcome (a retryable failure is retried with back-off; an
    /// irrecoverable one, or more than `num_failure_retries`, fails the
    /// transform; a fenced checkpoint just stops executing here).
    fn run_trigger(self: &Arc<Self>, node: &Node, id: &str, run: Run) {
        let epoch = run.epoch;
        let r = self.checkpoint(node, id, run);
        let default_retries = crate::ilm::cluster_setting(node, "xpack.transform.num_transform_failure_retries").and_then(|v| v.parse::<i64>().ok()).unwrap_or(10);
        let need_remove;
        {
            let mut map = self.map.lock();
            if self.epoch.load(Ordering::SeqCst) != epoch {
                drop(map);
                self.cv.notify_all();
                return;
            }
            let Some(e) = map.get_mut(id).filter(|e| e.generation == run.generation) else {
                // Deleted, or disowned by a forced stop / reset: nothing of
                // this worker's is recorded.
                drop(map);
                self.cv.notify_all();
                return;
            };
            e.st.running = false;
            e.st.progress = None;
            if matches!(&r, Err(err) if err.kind == HALTED) {
                // Upgrade mode halted the checkpoint: it runs again (from
                // its last completed checkpoint) once the mode is disabled.
                tracing::info!("[{id}] transform checkpoint halted by upgrade mode");
                e.st.next_run = 0;
                drop(map);
                self.cv.notify_all();
                return;
            }
            if matches!(&r, Err(err) if err.kind == FENCED) {
                tracing::info!("[{id}] transform stopped executing on this node: its checkpoint was fenced (the task is no longer assigned here)");
                e.st.started = false;
                e.st.alloc = None;
                drop(map);
                self.cv.notify_all();
                return;
            }
            let parsed = config::parse(Some(id), &config::strip_generated(&e.config), false, true).ok();
            let freq = parsed.as_ref().map(|c| c.frequency_ms).unwrap_or(config::DEFAULT_FREQUENCY_MS);
            let mut next = crate::date::now_millis() + freq;
            match r {
                Ok(()) => {
                    if e.st.failure_count > 0 {
                        e.st.failure_count = 0;
                        e.st.health_issue = None;
                    }
                }
                Err(err) => {
                    e.st.stats.search_failures += 1;
                    e.st.failure_count += 1;
                    let unattended = parsed.as_ref().map(|c| c.settings.unattended).unwrap_or(false);
                    let retries = parsed.as_ref().and_then(|c| c.settings.num_failure_retries).unwrap_or(default_retries);
                    if !e.st.started {
                    } else if unattended {
                        e.st.health_issue = Some(err.reason.clone());
                        next = crate::date::now_millis() + retry_delay_ms(e.st.failure_count);
                    } else if irrecoverable(&err) {
                        e.st.failed = Some(format!("task encountered irrecoverable failure: {}", err.reason));
                    } else if retries != -1 && e.st.failure_count as i64 > retries {
                        e.st.failed = Some(format!("task encountered more than {retries} failures; latest failure: {}", err.reason));
                    } else {
                        tracing::warn!("[{id}] transform checkpoint failed, retrying (attempt {}): {}", e.st.failure_count, err.reason);
                        e.st.health_issue = Some(err.reason.clone());
                        next = crate::date::now_millis() + retry_delay_ms(e.st.failure_count).min(freq.max(5_000));
                    }
                }
            }
            e.st.next_run = next;
            if e.st.stop_requested {
                e.st.started = false;
                e.st.stop_requested = false;
                e.st.stop_at_checkpoint = false;
            }
            need_remove = !e.st.started && e.st.alloc.is_some();
        }
        if need_remove {
            let _g = self.lifecycle.lock();
            if let Err(e) = self.fenced_write(id, run, || self.remove_task(id)) {
                tracing::warn!("[{id}] transform task not removed: {}", e.reason);
            }
        }
        if let Err(e) = self.fenced_write(id, run, || self.save_state(id)) {
            if e.kind == FENCED {
                if let Some(en) = self.map.lock().get_mut(id) {
                    en.st.started = false;
                    en.st.alloc = None;
                }
            }
            tracing::warn!("[{id}] transform state not stored: {}", e.reason);
        }
        self.cv.notify_all();
    }

    /// The coordinator a checkpoint runs through: on a cluster, as the
    /// stored user through the local coordinator; standalone, the `_start`
    /// caller's (if any) or the local engine.
    fn run_ops(&self, id: &str, raw: &Value) -> Ops {
        match self.cluster() {
            Some(c) => Ops(Some(c.rest_as(stored_run_as(raw).as_ref())), None),
            None => Ops(self.run_rest.lock().get(id).cloned(), None),
        }
    }

    fn checkpoint(&self, node: &Node, id: &str, run: Run) -> Result<()> {
        let epoch = run.epoch;
        if self.fenced(id, epoch) {
            return Err(fenced_err(id));
        }
        let (raw, cfg) = self.config_of(id)?;
        let ops = self.run_ops(id, &raw);
        let run_as = stored_run_as(&raw);
        let src = cfg.source_indices.join(",");
        let started = Instant::now();
        let (prev_cp, prev_upper, encoding) = {
            let map = self.map.lock();
            let e = map.get(id).ok_or_else(|| not_found(id))?;
            (e.st.checkpoint, e.st.last_upper, e.st.id_encoding)
        };
        // Documents written with an older id encoding: this checkpoint
        // rebuilds the destination (every group recomputed, each old id
        // deleted), then the transform is on the current encoding.
        let rebuild = encoding < pivot::ID_ENCODING;
        self.update_state(id, run, |s| s.stats.trigger_count += 1);
        let targets = pivot::source_targets(node, &cfg, cfg.settings.unattended)?;
        if targets.is_empty() && cfg.settings.unattended {
            return Err(EsError::index_not_found(&cfg.source_indices.join(",")));
        }
        let targets = if ops.0.is_none() { secure_targets(node, run_as.as_ref(), &cfg.source_indices, targets)? } else { targets };
        // #567 `use_point_in_time` (ES's default): every source search of
        // this checkpoint (change detection, count, pages, latest
        // follow-ups) reads one view, so a document moving between groups
        // across a page boundary is neither lost nor counted twice.
        let pinned = cfg.settings.use_point_in_time && !targets.is_empty();
        let targets = if pinned && ops.0.is_none() { node.snapshot(&targets)? } else { targets };
        let pit = if pinned { ops.open_pit(&src)? } else { None };
        let sops = Ops(ops.0.clone(), pit.as_ref().map(|p| p.id.clone()));
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
            if (prev_cp > 0 || prev_upper.is_some()) && !rebuild {
                let lower = prev_upper.unwrap_or(0);
                // Change detection: any new data in [lower, upper)?
                let probe = json!({"bool": {"filter": [cfg.query.clone(), {"range": {field.clone(): {"gte": lower, "lt": u, "format": "epoch_millis"}}}]}});
                let t0 = Instant::now();
                let changed_filter = self.changed_buckets(node, &sops, &cfg, &targets, &probe, lower)?;
                self.update_state(id, run, |s| {
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
                        self.update_state(id, run, |s| s.changes_last_detected_at = Some(now));
                        if prev_cp > 0 {
                            filters.extend(f);
                        }
                    }
                }
            } else {
                self.update_state(id, run, |s| s.last_search_time = Some(now));
            }
            query = json!({"bool": {"filter": filters}});
        }
        // Progress total.
        let total = if targets.is_empty() { 0 } else { sops.count(node, &src, &targets, &query, cfg.runtime_mappings.as_ref())? };
        self.update_state(id, run, |s| s.progress = Some((0, total)));
        let dest = cfg.dest_index.clone().unwrap_or_default();
        let mut after: Option<Value> = None;
        let (mut cp_indexed, mut cp_processed) = (0u64, 0u64);
        let filtered = pivot::filters_buckets(&cfg);
        if !targets.is_empty() {
            loop {
                if self.should_abort(id, run) {
                    return Ok(());
                }
                let body = pivot::page_body(&cfg, &query, cfg.page_size(), after.as_ref());
                let t0 = Instant::now();
                let resp = sops.search(node, &src, &targets, &body)?;
                let st_ms = t0.elapsed().as_millis() as u64;
                let t1 = Instant::now();
                let mut page = pivot::convert_page(&cfg, &resp, &types)?;
                resolve_latest(node, &sops, &cfg, &src, &targets, &query, &mut page)?;
                let proc_ms = t1.elapsed().as_millis() as u64;
                let n = page.docs.len() as u64;
                let deletes: Vec<String> = if rebuild { page.legacy_ids.clone() } else { Vec::new() };
                if self.fenced(id, epoch) {
                    return Err(fenced_err(id));
                }
                let t2 = Instant::now();
                let (ok, failed) = if n > 0 { self.fenced_mutation(id, run, || self.index_docs(node, &ops, &cfg, &dest, page.docs, &deletes))? } else { (0, 0) };
                let idx_ms = t2.elapsed().as_millis() as u64;
                cp_indexed += ok;
                cp_processed += page.docs_processed;
                let dp = page.docs_processed;
                self.update_state(id, run, |s| {
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
                // A bucket_selector filters after the composite paged: the
                // cursor is followed whatever survived (a page filtered empty
                // still has a cursor; only a page without one is the last).
                match page.after_key {
                    Some(a) if filtered && after.as_ref() != Some(&a) => after = Some(a),
                    Some(a) if !filtered && n as usize >= cfg.page_size() => after = Some(a),
                    _ => break,
                }
            }
        }
        // ES refreshes the destination at the end of every checkpoint
        // (`TransformIndexer.onFinish`), before the retention policy runs.
        if !dest.is_empty() {
            self.fenced_write(id, run, || ops.refresh(node, &dest))?;
        }
        // Retention policy.
        if let Some((field, max_age)) = &cfg.retention {
            if self.fenced(id, epoch) {
                return Err(fenced_err(id));
            }
            let t0 = Instant::now();
            let deleted = self.fenced_mutation(id, run, || self.apply_retention_until(node, &ops, &dest, field, now - max_age, &|| self.should_abort(id, run)))?;
            self.update_state(id, run, |s| {
                s.stats.documents_deleted += deleted;
                s.stats.delete_time_in_ms += t0.elapsed().as_millis() as u64;
            });
        }
        if self.fenced(id, epoch) {
            return Err(fenced_err(id));
        }
        let at = crate::date::now_millis();
        let dur = started.elapsed().as_secs_f64() * 1000.0;
        let continuous = cfg.is_continuous();
        let mut completed = 0;
        self.fenced_write(id, run, || {
            self.save_checkpoint(id, prev_cp + 1, at, upper)?;
            self.update_state(id, run, |s| {
            let first = s.checkpoint == 0;
            s.checkpoint += 1;
            completed = s.checkpoint;
            s.last_cp_millis = at;
            s.last_upper = upper;
            s.id_encoding = pivot::ID_ENCODING;
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
        })?;
        if rebuild {
            tracing::info!("[{id}] transform destination rebuilt with document id encoding [{}]", pivot::ID_ENCODING);
        }
        if audit_checkpoint(completed) {
            let on = self.cluster().map(|c| c.local_node()).unwrap_or_else(|| NODE_ID.to_string());
            tracing::info!("[{id}] finished indexing for transform checkpoint [{completed}] on node [{on}]");
        }
        Ok(())
    }

    /// Filters restricting a continuous checkpoint to the buckets touched by
    /// new data (`None` when there is no new data).
    fn changed_buckets(&self, node: &Node, ops: &Ops, cfg: &Config, targets: &[crate::search::Target], probe: &Value, lower: i64) -> Result<Option<Vec<Value>>> {
        if targets.is_empty() {
            return Ok(None);
        }
        let src = cfg.source_indices.join(",");
        let n = ops.count(node, &src, targets, probe, cfg.runtime_mappings.as_ref())?;
        if n == 0 {
            return Ok(None);
        }
        let sync_field = cfg.sync.as_ref().map(|s| s.0.clone()).unwrap_or_default();
        let mut filters = Vec::new();
        let term_fields: Vec<(String, String)> = match &cfg.function {
            Function::Pivot { groups, .. } => {
                for g in groups {
                    if g.kind == "date_histogram" && g.field() == Some(sync_field.as_str()) {
                        // ES `DateHistogramGroupSource.getIncrementalBucketUpdateFilterQuery`:
                        // from the start of the bucket holding the previous
                        // upper bound (calendar or fixed interval, offset and
                        // time zone), so a bucket the new data falls into is
                        // recomputed from all of its documents, not only
                        // those since the last checkpoint.
                        let from = crate::aggs::date_histogram_bucket_start(&g.params, lower)?.min(lower);
                        filters.push(json!({"range": {sync_field.clone(): {"gte": from, "format": "epoch_millis"}}}));
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

    /// Bulk-index converted documents into `dest` (through the pipelines),
    /// deleting the `deletes` ids (an encoding-1 rebuild) first.
    fn index_docs(&self, node: &Node, ops: &Ops, cfg: &Config, dest: &str, docs: Vec<(String, Map<String, Value>)>, deletes: &[String]) -> Result<(u64, u64)> {
        let new_ids: HashSet<&str> = docs.iter().map(|(id, _)| id.as_str()).collect();
        let deletes: Vec<String> = deletes.iter().filter(|d| !new_ids.contains(d.as_str())).cloned().collect();
        if let Some(rest) = &ops.0 {
            // The coordinator runs the pipelines (`dest.pipeline`, the
            // index's default/final pipelines) and routes to the primaries.
            let items: Vec<(String, String, Value)> = docs.into_iter().map(|(id, d)| (dest.to_string(), id, Value::Object(d))).collect();
            let uri = match &cfg.dest_pipeline {
                Some(p) => format!("/_bulk?pipeline={}", path_expr(p)),
                None => "/_bulk".to_string(),
            };
            let mut body = bulk_delete_body(dest, &deletes);
            body.push_str(&bulk_body(&items));
            let resp = rest("POST", &uri, Some(Value::String(body)))?;
            return Ok(bulk_counts(&resp));
        }
        let mut by_index: BTreeMap<String, Vec<WriteReq>> = BTreeMap::new();
        let mut failed = 0u64;
        if !deletes.is_empty() {
            if let Some(svc) = node.get_index(dest) {
                let reqs = deletes.iter().map(|id| WriteReq::Delete { id: id.clone(), routing: None, cond: WriteCond::default() }).collect();
                for r in svc.write(reqs, true) {
                    if let Err(e) = r {
                        if e.status != 404 {
                            failed += 1;
                        }
                    }
                }
            }
        }
        for (id, d) in docs {
            let src = serde_json::to_string(&Value::Object(d))?;
            let out = node.run_ingest(crate::ingest::IngestRequest { index: dest, id: Some(&id), routing: None, source: &src, pipeline: cfg.dest_pipeline.as_deref(), cond: Default::default() });
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

    /// The retention policy: delete the destination documents older than
    /// `cutoff`. ES runs a delete-by-query on the configured destination
    /// expression (`dest.index`, an index or an alias: every index behind
    /// it, alias filters applied) with lenient-expand-open options, then
    /// refreshes.
    fn apply_retention(&self, node: &Node, ops: &Ops, dest: &str, field: &str, cutoff: i64) -> Result<u64> {
        self.apply_retention_until(node, ops, dest, field, cutoff, &|| false)
    }

    /// The retention delete over `dest`; `abort` is checked between pages
    /// (a stopped or fenced checkpoint stops deleting).
    fn apply_retention_until(&self, node: &Node, ops: &Ops, dest: &str, field: &str, cutoff: i64, abort: &dyn Fn() -> bool) -> Result<u64> {
        let query = json!({"range": {field: {"lt": cutoff, "format": "epoch_millis"}}});
        if let Some(rest) = &ops.0 {
            let uri = format!("/{}/_delete_by_query?refresh=true&conflicts=proceed&ignore_unavailable=true&allow_no_indices=true", path_expr(dest));
            return match rest("POST", &uri, Some(json!({"query": query.clone()}))) {
                Ok(r) => Ok(r["deleted"].as_u64().unwrap_or(0)),
                Err(e) if e.kind == "index_not_found_exception" => Ok(0),
                Err(e) => Err(e),
            };
        }
        let opts = crate::node::ResolveOpts { ignore_unavailable: true, allow_no_indices: true, ..Default::default() };
        let targets = node.resolve(dest, opts)?;
        if targets.is_empty() {
            return Ok(0);
        }
        let mut deleted = 0;
        loop {
            if abort() {
                return Err(EsError::new(409, "status_exception", "the transform checkpoint was stopped during its retention policy"));
            }
            let body = json!({"size": 1000, "_source": false, "query": query.clone()});
            let resp = pivot::search(node, &targets, &body)?;
            let hits = resp["hits"]["hits"].as_array().cloned().unwrap_or_default();
            if hits.is_empty() {
                break;
            }
            let n = hits.len();
            let mut by_index: BTreeMap<String, Vec<WriteReq>> = BTreeMap::new();
            for h in &hits {
                let Some(id) = h["_id"].as_str() else { continue };
                let index = match h["_index"].as_str() {
                    Some(i) => i.to_string(),
                    None if targets.len() == 1 => targets[0].svc.name.clone(),
                    None => continue,
                };
                // #591: the document's own routing (a destination pipeline
                // may route it to another shard than its `_id` selects).
                let routing = h["_routing"].as_str().map(str::to_string);
                by_index.entry(index).or_default().push(WriteReq::Delete { id: id.to_string(), routing, cond: WriteCond::default() });
            }
            let mut round = 0u64;
            for (index, reqs) in by_index {
                let Some(t) = targets.iter().find(|t| t.svc.name == index) else { continue };
                for r in t.svc.write(reqs, true) {
                    match r {
                        Ok(_) => round += 1,
                        // Deleted meanwhile: not a failure.
                        Err(e) if e.status == 404 => {}
                        Err(e) => return Err(e),
                    }
                }
            }
            if round == 0 {
                // Hits that cannot be deleted would be searched again forever.
                return Err(EsError::new(500, "status_exception", format!("the retention policy of [{dest}] deleted none of [{n}] expired documents")));
            }
            deleted += round;
            // ES refreshes the destination after the retention
            // delete-by-query; the next page must not see these again.
            for t in &targets {
                t.svc.refresh()?;
            }
            if n < 1000 || round == 0 {
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

    #[test]
    fn rebuild_deletes_precede_the_new_documents_and_missing_ones_are_not_failures() {
        let mut b = bulk_delete_body("dest", &["old".into()]);
        b.push_str(&bulk_body(&[("dest".into(), "new".into(), json!({"x": 1}))]));
        let lines: Vec<Value> = b.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines[0], json!({"delete": {"_index": "dest", "_id": "old"}}));
        assert_eq!(lines[1], json!({"index": {"_index": "dest", "_id": "new"}}));
        let r = json!({"items": [{"delete": {"status": 404, "result": "not_found"}}, {"delete": {"status": 500, "error": {"type": "x"}}}, {"index": {"status": 201}}]});
        assert_eq!(bulk_counts(&r), (1, 1));
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn write_docs(node: &Node, index: &str, docs: &[(String, Value)]) {
        let svc = node.write_index(index, true).unwrap();
        let reqs = docs.iter().map(|(id, d)| WriteReq::Index { id: Some(id.clone()), source: d.to_string(), routing: None, cond: WriteCond::default() }).collect();
        for r in svc.write(reqs, true) {
            r.unwrap();
        }
        node.refresh_indices(index).unwrap();
    }

    fn ids_of(node: &Node, index: &str) -> Vec<String> {
        let t = vec![crate::search::Target::new(node.get_index(index).unwrap())];
        let r = pivot::search(node, &t, &json!({"size": 100, "query": {"match_all": {}}})).unwrap();
        let mut out: Vec<String> = r["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
        out.sort();
        out
    }

    fn stats_of(svc: &TransformService, node: &Node, id: &str) -> Value {
        svc.stats(node, id, 0, 10, true).unwrap()["transforms"][0].clone()
    }

    fn wait_for(mut f: impl FnMut() -> bool, what: &str) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while !f() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// A coordinator whose source search answers with one shard failed (the
    /// response a cluster gives while a source shard is lost).
    fn partial_cluster(node: Arc<Node>, bulks: Arc<AtomicUsize>) -> RestCall {
        Arc::new(move |method: &str, uri: &str, body: Option<Value>| {
            let path = uri.split('?').next().unwrap_or("");
            if method == "PUT" {
                node.create_index(path.trim_start_matches('/'), &body.unwrap_or(json!({})))?;
                return Ok(json!({"acknowledged": true}));
            }
            if path.ends_with("/_count") {
                return Ok(json!({"count": 2, "_shards": {"total": 2, "successful": 2, "failed": 0}}));
            }
            if path.ends_with("/_search") {
                return Ok(json!({
                    "timed_out": false,
                    "_shards": {"total": 2, "successful": 1, "failed": 1, "failures": [{"shard": 1, "index": "src", "status": 503, "reason": {"type": "no_shard_available_action_exception", "reason": "shard lost"}}]},
                    "aggregations": {"_transform": {"buckets": [{"key": {"k": "a"}, "doc_count": 1, "n": {"value": 1}}], "after_key": {"k": "a"}}}
                }));
            }
            if path == "/_bulk" {
                bulks.fetch_add(1, Ordering::SeqCst);
                return Ok(json!({"items": []}));
            }
            Ok(json!({"acknowledged": true}))
        })
    }

    fn body(dest: &str, retries: Option<i64>) -> Value {
        let mut b = json!({
            "source": {"index": "src"},
            "dest": {"index": dest},
            "pivot": {"group_by": {"k": {"terms": {"field": "k"}}}, "aggs": {"n": {"value_count": {"field": "k"}}}}
        });
        if let Some(r) = retries {
            b["settings"] = json!({"num_failure_retries": r});
        }
        b
    }

    #[test]
    fn a_partial_source_page_never_completes_a_checkpoint() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("src", &json!({"mappings": {"properties": {"k": {"type": "keyword"}}}})).unwrap();
        write_docs(&node, "src", &[("1".into(), json!({"k": "a"})), ("2".into(), json!({"k": "b"}))]);
        let bulks = Arc::new(AtomicUsize::new(0));
        let ops = Ops(Some(partial_cluster(node.clone(), bulks.clone())), None);
        let svc = service(&node);
        svc.put(&node, &ops, "retrying", &body("dst-retrying", None), true, None).unwrap();
        svc.put(&node, &ops, "no-retries", &body("dst-no-retries", Some(0)), true, None).unwrap();
        svc.start(&node, &ops, "retrying", None).unwrap();
        svc.start(&node, &ops, "no-retries", None).unwrap();
        // Shard lost: the checkpoint fails and is retried, never completed
        // with the groups of the remaining shard.
        wait_for(|| stats_of(&svc, &node, "retrying")["stats"]["search_failures"].as_u64().unwrap_or(0) >= 1, "a failed checkpoint");
        let s = stats_of(&svc, &node, "retrying");
        assert_eq!(s["checkpointing"]["last"]["checkpoint"], 0, "{s}");
        assert_eq!(s["state"], "started", "a shard failure is retried, not a completed or failed transform: {s}");
        assert_eq!(s["health"]["status"], "yellow", "{s}");
        assert!(s["health"]["issues"][0]["details"].as_str().unwrap().contains("Partial shards failure"), "{s}");
        assert_eq!(bulks.load(Ordering::SeqCst), 0, "nothing of a partial page is indexed");
        // `num_failure_retries: 0`: the first failure fails the transform.
        wait_for(|| stats_of(&svc, &node, "no-retries")["state"] == "failed", "the transform to fail");
        let s = stats_of(&svc, &node, "no-retries");
        assert!(s["reason"].as_str().unwrap().contains("more than 0 failures"), "{s}");
        assert_eq!(s["checkpointing"]["last"]["checkpoint"], 0, "{s}");
        svc.stop("retrying,no-retries", true, false, true, true, Duration::from_secs(10)).unwrap();
    }

    #[test]
    fn colliding_groups_get_distinct_documents_and_old_ids_are_rebuilt() {
        let tmp = tempfile::tempdir().unwrap();
        // A transform stored before id encoding 2 (no `id_encoding`), with
        // one completed checkpoint whose documents used the ambiguous ids.
        let cfg = json!({
            "id": "legacy",
            "source": {"index": ["src"], "query": {"match_all": {}}},
            "dest": {"index": "dst"},
            "pivot": {"group_by": {"a": {"terms": {"field": "a", "missing_bucket": true}}, "b": {"terms": {"field": "b"}}}, "aggregations": {"n": {"value_count": {"field": "b"}}}}
        });
        std::fs::create_dir_all(tmp.path().join("_state")).unwrap();
        std::fs::write(tmp.path().join("_state").join("transforms.json"), json!({"legacy": {"config": cfg, "state": {"started": false, "checkpoint": 1, "last_cp_millis": 1}}}).to_string()).unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("src", &json!({"mappings": {"properties": {"a": {"type": "keyword"}, "b": {"type": "keyword"}}}})).unwrap();
        node.create_index("dst", &json!({"mappings": {"properties": {"a": {"type": "keyword"}, "b": {"type": "keyword"}, "n": {"type": "long"}}}})).unwrap();
        let groups = [(json!("x$x"), json!("x")), (json!("x"), json!("x$x")), (Value::Null, json!("n")), (json!("__NULL_VALUE__"), json!("n"))];
        let src: Vec<(String, Value)> = groups.iter().enumerate().map(|(i, (a, b))| (i.to_string(), if a.is_null() { json!({"b": b}) } else { json!({"a": a, "b": b}) })).collect();
        write_docs(&node, "src", &src);
        let key = |a: &Value, b: &Value| -> BTreeMap<String, Value> { [("a".to_string(), a.clone()), ("b".to_string(), b.clone())].into_iter().collect() };
        let legacy: Vec<(String, Value)> = groups.iter().map(|(a, b)| (pivot::legacy_doc_id(&key(a, b)), json!({"a": a, "b": b, "n": 1}))).collect();
        write_docs(&node, "dst", &legacy);
        let mut legacy_ids: Vec<String> = legacy.iter().map(|(id, _)| id.clone()).collect();
        legacy_ids.sort();
        legacy_ids.dedup();
        assert_eq!(legacy_ids.len(), 2, "encoding 1 merged the four groups into two documents");
        assert_eq!(ids_of(&node, "dst"), legacy_ids);
        let svc = service(&node);
        svc.start(&node, &Ops::default(), "legacy", None).unwrap();
        wait_for(|| stats_of(&svc, &node, "legacy")["checkpointing"]["last"]["checkpoint"] == 2 && stats_of(&svc, &node, "legacy")["state"] == "stopped", "the rebuild checkpoint");
        let mut want: Vec<String> = groups.iter().map(|(a, b)| pivot::doc_id(&key(a, b))).collect();
        want.sort();
        want.dedup();
        assert_eq!(want.len(), 4, "every group has its own id");
        assert_eq!(ids_of(&node, "dst"), want, "old ids deleted, one document per group");
        assert_eq!(svc.map.lock()["legacy"].st.id_encoding, pivot::ID_ENCODING);
        let stored: Value = serde_json::from_slice(&std::fs::read(tmp.path().join("_state").join("transforms.json")).unwrap()).unwrap();
        assert_eq!(stored["legacy"]["state"]["id_encoding"], pivot::ID_ENCODING, "the marker is persisted: no second rebuild");
    }

    /// A coordinator whose source search blocks until released (a stalled
    /// shard), then answers a full page; it records every write.
    struct Stalled {
        entered: std::sync::mpsc::Receiver<()>,
        release: std::sync::mpsc::Sender<()>,
        writes: Arc<Mutex<Vec<String>>>,
    }

    fn stalled_cluster(node: Arc<Node>) -> (RestCall, Stalled) {
        let (entered_tx, entered) = std::sync::mpsc::channel();
        let (release, release_rx) = std::sync::mpsc::channel::<()>();
        let release_rx = Arc::new(Mutex::new(release_rx));
        let entered_tx = Arc::new(Mutex::new(entered_tx));
        let writes: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let w = writes.clone();
        let rest: RestCall = Arc::new(move |method: &str, uri: &str, body: Option<Value>| {
            let path = uri.split('?').next().unwrap_or("").to_string();
            // Opening and closing the checkpoint's point in time are reads.
            if path.ends_with("/_pit") {
                return Ok(if method == "POST" { json!({"id": "stalled-pit"}) } else { json!({"succeeded": true}) });
            }
            if method == "PUT" {
                node.create_index(path.trim_start_matches('/'), &body.unwrap_or(json!({})))?;
                return Ok(json!({"acknowledged": true}));
            }
            if method == "DELETE" {
                node.delete_index(path.trim_start_matches('/'))?;
                return Ok(json!({"acknowledged": true}));
            }
            if path.ends_with("/_count") {
                return Ok(json!({"count": 1, "_shards": {"total": 1, "successful": 1, "failed": 0}}));
            }
            if path.ends_with("/_search") {
                let _ = entered_tx.lock().send(());
                let _ = release_rx.lock().recv();
                return Ok(json!({
                    "timed_out": false,
                    "_shards": {"total": 1, "successful": 1, "failed": 0},
                    "aggregations": {"_transform": {"buckets": [{"key": {"k": "a"}, "doc_count": 1, "n": {"value": 1}}]}}
                }));
            }
            w.lock().push(format!("{method} {path}"));
            Ok(json!({"items": [{"index": {"status": 201}}]}))
        });
        (rest, Stalled { entered, release, writes })
    }

    /// #568: upgrade mode is durable, refuses starts, and once enabled no
    /// checkpoint write follows (a write is refused as halted).
    #[test]
    fn upgrade_mode_halts_writes_refuses_starts_and_persists() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("src", &json!({"mappings": {"properties": {"k": {"type": "keyword"}}}})).unwrap();
        write_docs(&node, "src", &[("1".into(), json!({"k": "a"}))]);
        let svc = service(&node);
        svc.put(&node, &Ops::default(), "up", &body("dst-up", None), true, None).unwrap();
        svc.set_upgrade_mode(true).unwrap();
        let e = svc.start(&node, &Ops::default(), "up", None).unwrap_err();
        assert_eq!(e.status, 429, "{}", e.reason);
        assert_eq!(e.reason, "Cannot start any Transform while the Transform feature is upgrading.");
        let run = Run { epoch: svc.epoch.load(Ordering::SeqCst), generation: svc.map.lock()["up"].generation };
        let mut wrote = false;
        let e = svc.fenced_write("up", run, || { wrote = true; Ok(()) }).unwrap_err();
        assert_eq!(e.kind, HALTED);
        assert!(!wrote, "no write after the acknowledgement");
        // A restart keeps the mode.
        assert!(TransformService::open(&node.data).upgrade_mode());
        svc.set_upgrade_mode(false).unwrap();
        assert!(!TransformService::open(&node.data).upgrade_mode());
        svc.fenced_write("up", run, || Ok(())).unwrap();
        svc.start(&node, &Ops::default(), "up", None).unwrap();
    }

    #[test]
    fn a_forced_reset_fences_a_worker_blocked_in_its_search() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("src", &json!({"mappings": {"properties": {"k": {"type": "keyword"}}}})).unwrap();
        write_docs(&node, "src", &[("1".into(), json!({"k": "a"}))]);
        let (rest, stalled) = stalled_cluster(node.clone());
        let ops = Ops(Some(rest), None);
        let svc = service(&node);
        svc.put(&node, &ops, "stuck", &body("dst-stuck", None), true, None).unwrap();
        svc.start(&node, &ops, "stuck", None).unwrap();
        stalled.entered.recv_timeout(Duration::from_secs(20)).expect("the worker reached its search");
        // A stop that cannot complete in time says so.
        let e = svc.stop("stuck", false, false, true, true, Duration::from_millis(300)).unwrap_err();
        assert_eq!(e.status, 408, "{}", e.reason);
        // A forced reset returns at once (no waiting out the search) and
        // proves no write of the old worker can follow.
        let t0 = Instant::now();
        svc.reset(&node, &ops, "stuck", true).unwrap();
        assert!(t0.elapsed() < Duration::from_secs(10), "reset waited for the stalled worker");
        assert!(node.get_index("dst-stuck").is_none(), "the transform-created destination is gone");
        stalled.release.send(()).unwrap();
        std::thread::sleep(Duration::from_secs(2));
        assert!(stalled.writes.lock().is_empty(), "the old worker wrote after the reset: {:?}", stalled.writes.lock());
        assert!(node.get_index("dst-stuck").is_none(), "the old worker re-created the destination");
        let s = stats_of(&svc, &node, "stuck");
        assert_eq!(s["state"], "stopped", "{s}");
        assert_eq!(s["checkpointing"]["last"]["checkpoint"], 0, "{s}");
        assert_eq!(s["stats"]["trigger_count"], 0, "the old worker changed the reset state: {s}");
        assert_eq!(s["stats"]["documents_indexed"], 0, "{s}");
    }

    #[test]
    fn a_page_filtered_empty_by_bucket_selector_does_not_end_the_checkpoint() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("src", &json!({"mappings": {"properties": {"k": {"type": "keyword"}}}})).unwrap();
        // k00..k09 one document each (filtered out), k10 two documents.
        let mut docs: Vec<(String, Value)> = (0..10).map(|i| (format!("d{i}"), json!({"k": format!("k{i:02}")}))).collect();
        docs.push(("d10".into(), json!({"k": "k10"})));
        docs.push(("d11".into(), json!({"k": "k10"})));
        write_docs(&node, "src", &docs);
        let svc = service(&node);
        let b = json!({
            "source": {"index": "src"},
            "dest": {"index": "dst-filtered"},
            "settings": {"max_page_search_size": 10},
            "pivot": {"group_by": {"k": {"terms": {"field": "k"}}}, "aggs": {
                "n": {"value_count": {"field": "k"}},
                "keep": {"bucket_selector": {"buckets_path": {"n": "n"}, "script": "params.n > 1"}}
            }}
        });
        svc.put(&node, &Ops::default(), "filtered", &b, true, None).unwrap();
        svc.start(&node, &Ops::default(), "filtered", None).unwrap();
        wait_for(|| stats_of(&svc, &node, "filtered")["state"] == "stopped" && stats_of(&svc, &node, "filtered")["checkpointing"]["last"]["checkpoint"] == 1, "the checkpoint");
        let t = vec![crate::search::Target::new(node.get_index("dst-filtered").unwrap())];
        let r = pivot::search(&node, &t, &json!({"size": 20})).unwrap();
        let keys: Vec<&str> = r["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_source"]["k"].as_str().unwrap()).collect();
        assert_eq!(keys, vec!["k10"], "the eligible group on the second page: {r}");
    }

    #[test]
    fn latest_by_runtime_field_finds_the_maximum_beyond_the_candidates() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("src", &json!({"mappings": {"properties": {"u": {"type": "keyword"}, "t": {"type": "long"}, "seq": {"type": "long"}}}})).unwrap();
        // 150 versions of one key; the latest (largest `t`) is the last
        // document, outside the first 100 candidates.
        let docs: Vec<(String, Value)> = (0..150).map(|i| (format!("d{i:03}"), json!({"u": "only", "t": if i == 149 { 10_000 } else { i }, "seq": i}))).collect();
        write_docs(&node, "src", &docs);
        let svc = service(&node);
        let b = json!({
            "source": {"index": "src", "runtime_mappings": {"rt": {"type": "long", "script": {"source": "emit(doc['t'].value)"}}}},
            "dest": {"index": "dst-latest"},
            "latest": {"unique_key": ["u"], "sort": "rt"}
        });
        svc.put(&node, &Ops::default(), "latest-rt", &b, true, None).unwrap();
        svc.start(&node, &Ops::default(), "latest-rt", None).unwrap();
        wait_for(|| stats_of(&svc, &node, "latest-rt")["state"] == "stopped" && stats_of(&svc, &node, "latest-rt")["checkpointing"]["last"]["checkpoint"] == 1, "the checkpoint");
        let t = vec![crate::search::Target::new(node.get_index("dst-latest").unwrap())];
        let r = pivot::search(&node, &t, &json!({"size": 5})).unwrap();
        let hits = r["hits"]["hits"].as_array().unwrap();
        assert_eq!(hits.len(), 1, "{r}");
        assert_eq!(hits[0]["_source"]["seq"], 149, "the true latest document: {r}");
        assert_eq!(hits[0]["_source"]["t"], 10_000, "{r}");
    }

    /// Two generations behind the write alias `reporting` (the newer one is
    /// its write index), each with an aged and a recent row.
    fn aliased_generations(node: &Node, now: i64) {
        for g in ["rep-v1", "rep-v2"] {
            node.create_index(g, &json!({"mappings": {"properties": {"t": {"type": "date"}}}})).unwrap();
            write_docs(node, g, &[(format!("{g}-old"), json!({"t": 1_000})), (format!("{g}-new"), json!({"t": now}))]);
        }
        node.update_aliases(&[
            json!({"add": {"index": "rep-v1", "alias": "reporting", "is_write_index": false}}),
            json!({"add": {"index": "rep-v2", "alias": "reporting", "is_write_index": true}}),
        ])
        .unwrap();
    }

    #[test]
    fn retention_deletes_aged_rows_behind_a_destination_write_alias() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let now = crate::date::now_millis();
        aliased_generations(&node, now);
        assert!(node.get_index("reporting").is_none(), "the destination is an alias, not an index");
        let svc = service(&node);
        let deleted = svc.apply_retention(&node, &Ops::default(), "reporting", "t", now - 3_600_000).unwrap();
        assert_eq!(deleted, 2, "the aged row of every index behind the alias");
        assert_eq!(ids_of(&node, "rep-v1"), vec!["rep-v1-new".to_string()]);
        assert_eq!(ids_of(&node, "rep-v2"), vec!["rep-v2-new".to_string()]);
        // A missing destination is lenient (nothing to delete).
        assert_eq!(svc.apply_retention(&node, &Ops::default(), "missing-dest", "t", now).unwrap(), 0);
    }

    #[test]
    fn clustered_retention_runs_the_delete_by_query_on_the_destination_alias() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let calls: Arc<Mutex<Vec<(String, String, Value)>>> = Arc::new(Mutex::new(Vec::new()));
        let rec = calls.clone();
        let rest: RestCall = Arc::new(move |method: &str, uri: &str, body: Option<Value>| -> Result<Value> {
            rec.lock().push((method.to_string(), uri.to_string(), body.unwrap_or(Value::Null)));
            Ok(json!({"deleted": 3}))
        });
        let svc = service(&node);
        // The coordinator's node holds no index named `reporting`.
        let deleted = svc.apply_retention(&node, &Ops(Some(rest), None), "reporting", "t", 5_000).unwrap();
        assert_eq!(deleted, 3);
        let calls = calls.lock();
        assert_eq!(calls.len(), 1, "{:?}", *calls);
        let (method, uri, body) = &calls[0];
        assert_eq!(method, "POST");
        assert!(uri.starts_with("/reporting/_delete_by_query?"), "{uri}");
        assert!(uri.contains("refresh=true") && uri.contains("ignore_unavailable=true"), "{uri}");
        assert_eq!(body["query"]["range"]["t"]["lt"], 5_000, "{body}");
    }

    /// #591: standalone retention deletes routed documents (their own
    /// routing, not the `_id`'s shard) and ends.
    #[test]
    fn standalone_retention_deletes_routed_documents() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("routed", &json!({"settings": {"number_of_shards": 4}, "mappings": {"properties": {"t": {"type": "date"}}}})).unwrap();
        let svc_i = node.get_index("routed").unwrap();
        let reqs = (0..20).map(|i| WriteReq::Index { id: Some(format!("d{i}")), source: json!({"t": 1_000}).to_string(), routing: Some("tenant-a".into()), cond: WriteCond::default() }).collect();
        for r in svc_i.write(reqs, true) {
            r.unwrap();
        }
        node.refresh_indices("routed").unwrap();
        let svc = service(&node);
        assert_eq!(svc.apply_retention(&node, &Ops::default(), "routed", "t", 5_000).unwrap(), 20);
        let t = vec![crate::search::Target::new(node.get_index("routed").unwrap())];
        let r = pivot::search(&node, &t, &json!({"size": 0, "track_total_hits": true})).unwrap();
        assert_eq!(r["hits"]["total"]["value"], 0, "{r}");
        // A stopped checkpoint stops deleting.
        let e = svc.apply_retention_until(&node, &Ops::default(), "routed", "t", 5_000, &|| true).unwrap_err();
        assert_eq!(e.status, 409);
    }

    /// #567: a checkpoint over a cluster reads its source through one point
    /// in time (opened, searched with, closed); `use_point_in_time` is on by
    /// default and can be turned off.
    #[test]
    fn checkpoint_source_searches_read_one_point_in_time() {
        let calls: Arc<Mutex<Vec<(String, String, Value)>>> = Arc::new(Mutex::new(Vec::new()));
        let rec = calls.clone();
        let rest: RestCall = Arc::new(move |method: &str, uri: &str, body: Option<Value>| -> Result<Value> {
            rec.lock().push((method.to_string(), uri.to_string(), body.clone().unwrap_or(Value::Null)));
            if uri.contains("/_pit") && method == "POST" {
                return Ok(json!({"id": "pit-1"}));
            }
            Ok(json!({"timed_out": false, "_shards": {"total": 1, "successful": 1, "failed": 0}, "hits": {"total": {"value": 7, "relation": "eq"}, "hits": []}}))
        });
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let ops = Ops(Some(rest), None);
        {
            let pit = ops.open_pit("src").unwrap().unwrap();
            let sops = Ops(ops.0.clone(), Some(pit.id.clone()));
            sops.search(&node, "src", &[], &json!({"size": 1})).unwrap();
            assert_eq!(sops.count(&node, "src", &[], &json!({"match_all": {}}), None).unwrap(), 7);
        }
        let calls = calls.lock();
        assert!(calls[0].1.starts_with("/src/_pit?keep_alive="), "{:?}", calls[0]);
        for (_, uri, body) in &calls[1..3] {
            assert!(uri.starts_with("/_search?"), "{uri}");
            assert_eq!(body["pit"]["id"], "pit-1", "{body}");
        }
        assert_eq!((calls[3].0.as_str(), calls[3].1.as_str()), ("DELETE", "/_pit"));
        assert_eq!(calls[3].2["id"], "pit-1");
        let on = config::parse(Some("t"), &json!({"source": {"index": "s"}, "dest": {"index": "d"}, "pivot": {"group_by": {"k": {"terms": {"field": "k"}}}, "aggs": {"n": {"value_count": {"field": "k"}}}}}), true, true).unwrap();
        assert!(on.settings.use_point_in_time);
        let off = config::parse(Some("t"), &json!({"source": {"index": "s"}, "dest": {"index": "d"}, "settings": {"use_point_in_time": false}, "pivot": {"group_by": {"k": {"terms": {"field": "k"}}}, "aggs": {"n": {"value_count": {"field": "k"}}}}}), true, true).unwrap();
        assert!(!off.settings.use_point_in_time);
    }

    #[test]
    fn dest_aliases_are_moved_onto_created_and_precreated_destinations() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("src", &json!({"mappings": {"properties": {"k": {"type": "keyword"}}}})).unwrap();
        write_docs(&node, "src", &[("1".into(), json!({"k": "a"}))]);
        // The previous generation holds both aliases.
        node.create_index("report-v1", &json!({})).unwrap();
        node.update_aliases(&[json!({"add": {"index": "report-v1", "alias": "reporting"}}), json!({"add": {"index": "report-v1", "alias": "archive"}})]).unwrap();
        let aliases = |i: &str| node.get_index(i).unwrap().aliases();
        let svc = service(&node);
        // A destination the transform creates.
        let mut b = body("report-v2", None);
        b["dest"]["aliases"] = json!([{"alias": "reporting", "move_on_creation": true}, {"alias": "archive"}]);
        svc.put(&node, &Ops::default(), "v2", &b, true, None).unwrap();
        svc.start(&node, &Ops::default(), "v2", None).unwrap();
        let v2 = aliases("report-v2");
        assert!(v2.contains_key("reporting") && v2.contains_key("archive"), "{v2:?}");
        let v1 = aliases("report-v1");
        assert!(!v1.contains_key("reporting"), "move_on_creation removes the old generation from the alias: {v1:?}");
        assert!(v1.contains_key("archive"), "without move_on_creation the alias keeps the old generation: {v1:?}");
        assert!(node.get_index("report-v1").is_some(), "the old generation is not deleted");
        // A destination the user created before the transform started.
        node.create_index("report-v3", &json!({"mappings": {"properties": {"k": {"type": "keyword"}, "n": {"type": "long"}}}})).unwrap();
        let mut b = body("report-v3", None);
        b["dest"]["aliases"] = json!([{"alias": "reporting", "move_on_creation": true}]);
        svc.put(&node, &Ops::default(), "v3", &b, true, None).unwrap();
        svc.start(&node, &Ops::default(), "v3", None).unwrap();
        assert!(aliases("report-v3").contains_key("reporting"), "the alias is added to a pre-created destination");
        let v2 = aliases("report-v2");
        assert!(!v2.contains_key("reporting") && v2.contains_key("archive"), "{v2:?}");
        assert!(!aliases("report-v1").contains_key("reporting"));
        for id in ["v2", "v3"] {
            wait_for(|| stats_of(&svc, &node, id)["state"] == "stopped" && stats_of(&svc, &node, id)["checkpointing"]["last"]["checkpoint"] == 1, "the batch checkpoint");
        }
    }

    #[test]
    fn clustered_dest_aliases_go_out_as_one_aliases_request() {
        let actions = dest_alias_actions("report-v2", &[json!({"alias": "reporting", "move_on_creation": true}), json!({"alias": "archive", "move_on_creation": false})]);
        assert_eq!(
            actions,
            vec![
                json!({"remove": {"index": "*", "alias": "reporting"}}),
                json!({"add": {"index": "report-v2", "alias": "reporting"}}),
                json!({"add": {"index": "report-v2", "alias": "archive"}}),
            ]
        );
    }

    #[test]
    fn retry_policy_matches_es() {
        assert!(irrecoverable(&EsError::new(404, "index_not_found_exception", "x")));
        assert!(irrecoverable(&EsError::new(400, "x", "x")));
        assert!(!irrecoverable(&EsError::new(503, "search_phase_execution_exception", "x")));
        assert!(!irrecoverable(&EsError::new(429, "es_rejected_execution_exception", "x")));
        assert_eq!(retry_delay_ms(1), 5_000);
        assert_eq!(retry_delay_ms(2), 10_000);
        assert_eq!(retry_delay_ms(30), 3_600_000);
        assert!(audit_checkpoint(1) && audit_checkpoint(9) && audit_checkpoint(20) && !audit_checkpoint(21) && audit_checkpoint(3000));
    }
}

/// Review findings 129, 141 and 142.
#[cfg(test)]
mod continuity_tests {
    use super::*;

    const HOUR: i64 = 3_600_000;
    const MIN: i64 = 60_000;

    fn write_docs(node: &Node, index: &str, docs: &[(String, Value)]) {
        let svc = node.write_index(index, true).unwrap();
        let reqs = docs.iter().map(|(id, d)| WriteReq::Index { id: Some(id.clone()), source: d.to_string(), routing: None, cond: WriteCond::default() }).collect();
        for r in svc.write(reqs, true) {
            r.unwrap();
        }
        node.refresh_indices(index).unwrap();
    }

    fn source(node: &Node, times: &[i64]) {
        node.create_index("src", &json!({"mappings": {"properties": {"time": {"type": "date"}, "ms": {"type": "long"}, "k": {"type": "keyword"}}}})).unwrap();
        let docs: Vec<(String, Value)> = times.iter().enumerate().map(|(i, t)| (format!("d{i}"), json!({"time": t, "ms": t, "k": "x"}))).collect();
        write_docs(node, "src", &docs);
    }

    fn cfg(body: Value) -> Config {
        config::parse(Some("t"), &body, true, true).unwrap()
    }

    /// The counts the pivot writes for `query` (bucket key → count).
    fn bucket_counts(node: &Node, cfg: &Config, query: &Value) -> BTreeMap<String, i64> {
        let targets = pivot::source_targets(node, cfg, false).unwrap();
        let types = pivot::field_types(cfg, &targets);
        let resp = pivot::search(node, &targets, &pivot::page_body(cfg, query, 100, None)).unwrap();
        let page = pivot::convert_page(cfg, &resp, &types).unwrap();
        page.docs.into_iter().map(|(_, d)| (d["h"].to_string(), d["n"].as_i64().unwrap_or(-1))).collect()
    }

    /// The query of a continuous checkpoint with upper bound `upper` after
    /// one whose upper bound was `lower` (as `checkpoint` builds it).
    fn incremental_query(svc: &TransformService, node: &Node, cfg: &Config, lower: i64, upper: i64) -> Value {
        let targets = pivot::source_targets(node, cfg, false).unwrap();
        let (field, _) = cfg.sync.clone().unwrap();
        let probe = json!({"bool": {"filter": [cfg.query.clone(), {"range": {field.clone(): {"gte": lower, "lt": upper, "format": "epoch_millis"}}}]}});
        let changed = svc.changed_buckets(node, &Ops::default(), cfg, &targets, &probe, lower).unwrap().expect("new data is detected");
        let mut filters = vec![cfg.query.clone(), json!({"range": {field: {"lt": upper, "format": "epoch_millis"}}})];
        filters.extend(changed);
        json!({"bool": {"filter": filters}})
    }

    fn pivot_body(histogram: Value, settings: Value) -> Value {
        json!({
            "source": {"index": "src"},
            "dest": {"index": "dst"},
            "sync": {"time": {"field": "time", "delay": "60s"}},
            "settings": settings,
            "pivot": {"group_by": {"h": {"date_histogram": histogram}}, "aggs": {"n": {"value_count": {"field": "k"}}}}
        })
    }

    /// Review finding 141: a continuous checkpoint whose previous upper
    /// bound fell inside a date_histogram bucket recomputes that whole
    /// bucket: its count equals a complete rebuild's (calendar interval,
    /// a calendar interval in a half-hour time zone, and a fixed interval
    /// with an offset and `align_checkpoints: false`).
    #[test]
    fn a_checkpoint_inside_a_bucket_recomputes_the_whole_bucket() {
        let base = 1_767_225_600_000i64; // 2026-01-01T00:00:00Z
        let cases = [
            ("calendar 1h", json!({"field": "time", "calendar_interval": "1h"}), json!({}), [5, 20, 40], 30, 45),
            ("calendar 1h +05:30", json!({"field": "time", "calendar_interval": "1h", "time_zone": "+05:30"}), json!({}), [35, 50, 70], 60, 80),
            ("fixed 1h offset 10m unaligned", json!({"field": "time", "fixed_interval": "1h", "offset": "10m"}), json!({"align_checkpoints": false}), [15, 25, 40], 30, 45),
        ];
        for (name, histogram, settings, minutes, lower_min, upper_min) in cases {
            let tmp = tempfile::tempdir().unwrap();
            let node = Node::open(tmp.path()).unwrap();
            source(&node, &minutes.map(|m| base + 2 * HOUR + m * MIN));
            let cfg = cfg(pivot_body(histogram, settings));
            let svc = service(&node);
            let (lower, upper) = (base + 2 * HOUR + lower_min * MIN, base + 2 * HOUR + upper_min * MIN);
            let full = bucket_counts(&node, &cfg, &json!({"bool": {"filter": [cfg.query.clone(), {"range": {"time": {"lt": upper, "format": "epoch_millis"}}}]}}));
            assert_eq!(full.values().sum::<i64>(), 3, "{name}: {full:?}");
            let incremental = bucket_counts(&node, &cfg, &incremental_query(&svc, &node, &cfg, lower, upper));
            assert!(!incremental.is_empty(), "{name}");
            for (key, n) in &incremental {
                assert_eq!(Some(n), full.get(key), "{name}: bucket {key} rewritten with a partial aggregate: {incremental:?} vs {full:?}");
            }
        }
    }

    /// Review finding 142: change detection runs in the source's search
    /// context: a runtime sync field, and a runtime field in the source
    /// query, find the new data (a physical field is the control).
    #[test]
    fn change_detection_carries_the_runtime_mappings() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let base = 1_767_225_600_000i64;
        source(&node, &[base + 5 * MIN, base + 40 * MIN]);
        let svc = service(&node);
        let rt = json!({"rt": {"type": "date", "script": {"source": "emit(doc['ms'].value)"}}, "rk": {"type": "keyword", "script": {"source": "emit(doc['k'].value)"}}});
        let make = |sync: &str, query: Value| {
            cfg(json!({
                "source": {"index": "src", "runtime_mappings": rt.clone(), "query": query},
                "dest": {"index": "dst"},
                "sync": {"time": {"field": sync, "delay": "60s"}},
                "pivot": {"group_by": {"k": {"terms": {"field": "k"}}}, "aggs": {"n": {"value_count": {"field": "k"}}}}
            }))
        };
        let (lower, upper) = (base + 30 * MIN, base + 50 * MIN);
        for (name, c) in [
            ("runtime sync field", make("rt", json!({"match_all": {}}))),
            ("runtime source query", make("time", json!({"term": {"rk": "x"}}))),
            ("physical control", make("time", json!({"term": {"k": "x"}}))),
        ] {
            let targets = pivot::source_targets(&node, &c, false).unwrap();
            let field = c.sync.clone().unwrap().0;
            let probe = json!({"bool": {"filter": [c.query.clone(), {"range": {field: {"gte": lower, "lt": upper, "format": "epoch_millis"}}}]}});
            let changed = svc.changed_buckets(&node, &Ops::default(), &c, &targets, &probe, lower).unwrap();
            assert!(changed.is_some(), "{name}: the record at +40m is new data");
            let total = Ops::default().count(&node, "src", &targets, &probe, c.runtime_mappings.as_ref()).unwrap();
            assert_eq!(total, 1, "{name}");
        }
    }

    /// A cluster coordinator over a shared data node: the internal index
    /// documents (conditional writes, gets), `_bulk`, `_delete_by_query`
    /// and refreshes, as the real REST API answers them. `muts` counts the
    /// destination mutations dispatched (bulks and delete-by-queries).
    fn data_rest(d: Arc<Node>, muts: Arc<std::sync::atomic::AtomicUsize>) -> RestCall {
        Arc::new(move |method: &str, uri: &str, body: Option<Value>| {
            let (path, query) = uri.split_once('?').unwrap_or((uri, ""));
            let param = |k: &str| query.split('&').find_map(|kv| kv.strip_prefix(&format!("{k}="))).map(str::to_string);
            let segs: Vec<&str> = path.trim_start_matches('/').split('/').collect();
            match (method, segs.as_slice()) {
                ("PUT", [index]) => {
                    if d.get_index(index).is_some() {
                        return Err(EsError::new(400, "resource_already_exists_exception", format!("index [{index}] already exists")));
                    }
                    d.create_index(index, &body.unwrap_or(json!({})))?;
                    Ok(json!({"acknowledged": true}))
                }
                ("PUT", [index, "_doc", id]) => {
                    let cond = WriteCond { if_seq_no: param("if_seq_no").and_then(|v| v.parse().ok()), if_primary_term: param("if_primary_term").and_then(|v| v.parse().ok()), ..WriteCond::default() };
                    let svc = d.write_index(index, true)?;
                    let w = svc.write(vec![WriteReq::Index { id: Some(id.to_string()), source: body.unwrap_or(json!({})).to_string(), routing: None, cond }], true).pop().unwrap()?;
                    d.refresh_indices(index)?;
                    Ok(json!({"_id": id, "_version": w.version, "_seq_no": w.seq_no, "_primary_term": w.primary_term, "result": w.result}))
                }
                ("GET", [index, "_doc", id]) => {
                    let svc = d.get_index(index).ok_or_else(|| EsError::index_not_found(index))?;
                    let r = pivot::search(&d, &[crate::search::Target::new(svc)], &json!({"query": {"ids": {"values": [id]}}, "seq_no_primary_term": true, "version": true}))?;
                    Ok(match r["hits"]["hits"].get(0) {
                        Some(h) => json!({"found": true, "_seq_no": h["_seq_no"], "_primary_term": h["_primary_term"], "_version": h["_version"], "_source": h["_source"]}),
                        None => json!({"found": false}),
                    })
                }
                ("POST", ["_bulk"]) => {
                    muts.fetch_add(1, Ordering::SeqCst);
                    let text = body.and_then(|b| b.as_str().map(str::to_string)).unwrap_or_default();
                    let mut lines = text.lines();
                    let mut items = Vec::new();
                    while let Some(action) = lines.next() {
                        let a: Value = serde_json::from_str(action).unwrap();
                        let (op, meta) = a.as_object().and_then(|o| o.iter().next()).map(|(k, v)| (k.clone(), v.clone())).unwrap();
                        let id = meta["_id"].as_str().unwrap().to_string();
                        let index = meta["_index"].as_str().unwrap().to_string();
                        let req = if op == "delete" {
                            WriteReq::Delete { id, routing: None, cond: WriteCond::default() }
                        } else {
                            WriteReq::Index { id: Some(id), source: lines.next().unwrap().to_string(), routing: None, cond: WriteCond::default() }
                        };
                        let r = d.write_index(&index, true)?.write(vec![req], true).pop().unwrap();
                        d.refresh_indices(&index)?;
                        items.push(match r {
                            Ok(w) => json!({op.clone(): {"status": 200, "_version": w.version}}),
                            Err(e) => json!({op.clone(): {"status": e.status, "error": {"type": e.kind, "reason": e.reason}}}),
                        });
                    }
                    Ok(json!({"items": items}))
                }
                ("POST", [index, "_delete_by_query"]) => {
                    muts.fetch_add(1, Ordering::SeqCst);
                    let svc = d.get_index(index).ok_or_else(|| EsError::index_not_found(index))?;
                    let q = body.as_ref().map(|b| b["query"].clone()).unwrap_or(json!({"match_all": {}}));
                    let r = pivot::search(&d, &[crate::search::Target::new(svc.clone())], &json!({"size": 1000, "query": q}))?;
                    let ids: Vec<String> = r["hits"]["hits"].as_array().into_iter().flatten().filter_map(|h| h["_id"].as_str().map(str::to_string)).collect();
                    let reqs = ids.iter().map(|id| WriteReq::Delete { id: id.clone(), routing: None, cond: WriteCond::default() }).collect();
                    let deleted = svc.write(reqs, true).into_iter().filter(|r| r.is_ok()).count();
                    d.refresh_indices(index)?;
                    Ok(json!({"deleted": deleted}))
                }
                ("POST", [index, "_refresh"]) => {
                    d.refresh_indices(index)?;
                    Ok(json!({"_shards": {"failed": 0}}))
                }
                _ => Err(EsError::illegal_argument(format!("unexpected {method} {uri}"))),
            }
        })
    }

    /// One node's view of the cluster: its id, the master term it loaded
    /// and the persistent tasks as it last applied them.
    struct View {
        me: String,
        term: u64,
        tasks: Mutex<Value>,
        rest: RestCall,
    }

    impl TransformCluster for View {
        fn system_rest(&self) -> RestCall {
            self.rest.clone()
        }
        fn rest_as(&self, _: Option<&RunAs>) -> RestCall {
            self.rest.clone()
        }
        fn local_node(&self) -> String {
            self.me.clone()
        }
        fn master_term(&self) -> Option<u64> {
            Some(self.term)
        }
        fn persistent_tasks(&self) -> Value {
            self.tasks.lock().clone()
        }
        fn update_persistent_tasks(&self, f: cluster::TasksChange) -> Result<Value> {
            let mut t = self.tasks.lock();
            if let Some(n) = f(&*t) {
                *t = n;
            }
            Ok(t.clone())
        }
    }

    /// The persistent tasks with `t` assigned to `executor` under `alloc`.
    fn tasks_assigned(executor: &str, alloc: u64, term: u64) -> Value {
        let (v, a) = cluster::create_task(&json!({"last_allocation_id": alloc - 1, "tasks": []}), "t", cluster::task_params("t", None, None), executor, term).unwrap();
        assert_eq!(a, alloc);
        v
    }

    /// An executor of transform `t` on its own node: the service with its
    /// cluster view, executing allocation `alloc`.
    fn executor(dir: &Path, view: Arc<View>, config: Value, alloc: u64, doc_seq: Option<(u64, u64)>) -> (TransformService, Run) {
        let svc = TransformService::open(dir);
        let term = view.term;
        let c: Arc<dyn TransformCluster> = view;
        *svc.cluster.lock() = Some(c);
        *svc.loaded_term.lock() = Some(term);
        let mut st = TState::fresh();
        st.started = true;
        st.alloc = Some(alloc);
        st.doc_seq = doc_seq;
        svc.map.lock().insert("t".into(), Entry { config, st, generation: 0 });
        let run = Run { epoch: svc.epoch.load(Ordering::SeqCst), generation: 0 };
        (svc, run)
    }

    fn dest_doc(d: &Node) -> (Value, Value) {
        d.refresh_indices("dst").unwrap();
        let t = vec![crate::search::Target::new(d.get_index("dst").unwrap())];
        let r = pivot::search(d, &t, &json!({"size": 5, "version": true})).unwrap();
        let h = &r["hits"]["hits"][0];
        (h["_source"]["n"].clone(), h["_version"].clone())
    }

    /// Review finding 129: when the task moves to a new master (a new
    /// allocation id), a former executor's next destination write is
    /// refused at the source before dispatch, whether it has applied the
    /// new assignment (allocation check) or not yet (its conditional state
    /// write loses to the new owner's): its held bulk and its retention
    /// delete-by-query never reach the destination and it cannot record a
    /// checkpoint. Destination `_version`s stay ES's (1, 2, 3, ...).
    #[test]
    fn a_former_executors_held_writes_never_land_after_the_takeover() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("data")).unwrap();
        let d = Node::open(&tmp.path().join("data")).unwrap();
        d.create_index("dst", &json!({"mappings": {"properties": {"k": {"type": "keyword"}, "n": {"type": "long"}, "time": {"type": "date"}}}})).unwrap();
        let muts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let rest = data_rest(d.clone(), muts.clone());
        let body = json!({"source": {"index": "src"}, "dest": {"index": "dst"},
            "pivot": {"group_by": {"k": {"terms": {"field": "k"}}}, "aggs": {"n": {"value_count": {"field": "k"}}}}});
        let c = cfg(body.clone());
        let ops = Ops(Some(rest.clone()), None);
        let page = |n: i64| vec![("group-x".to_string(), json!({"k": "x", "n": n, "time": crate::date::now_millis() - 2 * 86_400_000}).as_object().unwrap().clone())];
        // Node A executes allocation 1 (master term 1).
        let view_a = Arc::new(View { me: "A".into(), term: 1, tasks: Mutex::new(tasks_assigned("A", 1, 1)), rest: rest.clone() });
        let (a, run_a) = executor(&tmp.path().join("a"), view_a.clone(), body.clone(), 1, None);
        assert_eq!(a.fenced_mutation("t", run_a, || a.index_docs(&d, &ops, &c, "dst", page(1), &[])).unwrap(), (1, 0));
        assert_eq!(dest_doc(&d), (json!(1), json!(1)));
        // Node B, elected for term 2, loads the stored state...
        let state_uri = TransformService::doc_uri(&cluster::state_doc_id("t"));
        let stored = rest("GET", &state_uri, None).unwrap();
        let loaded = stored["_seq_no"].as_u64().zip(stored["_primary_term"].as_u64());
        assert!(loaded.is_some(), "{stored}");
        // ... while A, not yet aware, writes another page.
        assert_eq!(a.fenced_mutation("t", run_a, || a.index_docs(&d, &ops, &c, "dst", page(2), &[])).unwrap(), (1, 0));
        assert_eq!(dest_doc(&d), (json!(2), json!(2)));
        // B takes the task over (allocation 2) and writes its output: its
        // first conditional state write takes the document over from A.
        let view_b = Arc::new(View { me: "B".into(), term: 2, tasks: Mutex::new(tasks_assigned("B", 2, 2)), rest: rest.clone() });
        let (b, run_b) = executor(&tmp.path().join("b"), view_b.clone(), body.clone(), 2, loaded);
        assert_eq!(b.fenced_mutation("t", run_b, || b.index_docs(&d, &ops, &c, "dst", page(3), &[])).unwrap(), (1, 0));
        assert_eq!(dest_doc(&d), (json!(3), json!(3)));
        let before = muts.load(Ordering::SeqCst);
        // A's held page, released while A still believes it owns the task:
        // its conditional state write fails, the bulk is never sent.
        let e = a.fenced_mutation("t", run_a, || a.index_docs(&d, &ops, &c, "dst", page(99), &[])).unwrap_err();
        assert_eq!(e.kind, FENCED, "{e:?}");
        // Its retention delete-by-query (the destination document is past
        // max_age) is fenced the same way.
        let e = a.fenced_mutation("t", run_a, || a.apply_retention(&d, &ops, "dst", "time", crate::date::now_millis() - 86_400_000)).unwrap_err();
        assert_eq!(e.kind, FENCED, "{e:?}");
        // Nor can it record its checkpoint or state.
        assert_eq!(a.fenced_write("t", run_a, || a.save_state("t")).unwrap_err().kind, FENCED);
        assert_eq!(muts.load(Ordering::SeqCst), before, "nothing of the former executor was dispatched");
        assert_eq!(dest_doc(&d), (json!(3), json!(3)), "the new owner's output and ES versions are intact");
        // Once A applied the new assignment, the allocation check alone stops it.
        *view_a.tasks.lock() = tasks_assigned("B", 2, 2);
        assert_eq!(a.fenced_mutation("t", run_a, || a.index_docs(&d, &ops, &c, "dst", page(98), &[])).unwrap_err().kind, FENCED);
        assert_eq!(muts.load(Ordering::SeqCst), before);
        // The new owner keeps writing, its versions incrementing normally,
        // and its own retention runs.
        assert_eq!(b.fenced_mutation("t", run_b, || b.index_docs(&d, &ops, &c, "dst", page(4), &[])).unwrap(), (1, 0));
        assert_eq!(dest_doc(&d), (json!(4), json!(4)));
        assert_eq!(b.fenced_mutation("t", run_b, || b.apply_retention(&d, &ops, "dst", "time", crate::date::now_millis() - 86_400_000)).unwrap(), 1);
    }
}

#[cfg(test)]
mod recovery_tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// A cluster whose internal index holds `docs` (sorted by `(doc_type,
    /// id)`), served as a coordinator does: `search_after` honoured, at most
    /// `window` hits per response whatever `size` asks (a capped result
    /// window), and only the first `visible` documents ever returned while
    /// `hits.total` counts all of them.
    struct FakeCluster {
        docs: Arc<Vec<Value>>,
        window: usize,
        visible: usize,
        tasks: Mutex<Value>,
        searches: Arc<AtomicUsize>,
    }

    fn key(d: &Value) -> (String, String) {
        (d["_source"]["doc_type"].as_str().unwrap_or("").to_string(), d["_source"]["id"].as_str().unwrap_or("").to_string())
    }

    impl TransformCluster for FakeCluster {
        fn system_rest(&self) -> RestCall {
            let (docs, window, visible, searches) = (self.docs.clone(), self.window, self.visible, self.searches.clone());
            Arc::new(move |method: &str, uri: &str, body: Option<Value>| -> Result<Value> {
                let path = uri.split('?').next().unwrap_or("");
                if method == "POST" && path.ends_with("/_search") {
                    searches.fetch_add(1, Ordering::SeqCst);
                    let body = body.unwrap_or(Value::Null);
                    let size = body["size"].as_u64().unwrap_or(10) as usize;
                    let after = body.get("search_after").and_then(Value::as_array).map(|a| (a[0].as_str().unwrap_or("").to_string(), a[1].as_str().unwrap_or("").to_string()));
                    let hits: Vec<Value> = docs
                        .iter()
                        .take(visible)
                        .filter(|d| after.as_ref().map_or(true, |a| key(d) > *a))
                        .take(size.min(window))
                        .map(|d| {
                            let (t, i) = key(d);
                            let mut h = d.clone();
                            h["sort"] = json!([t, i]);
                            h
                        })
                        .collect();
                    return Ok(json!({
                        "timed_out": false,
                        "_shards": {"total": 1, "successful": 1, "failed": 0},
                        "hits": {"total": {"value": docs.len(), "relation": "eq"}, "hits": hits}
                    }));
                }
                Ok(json!({"acknowledged": true}))
            })
        }
        fn rest_as(&self, _subject: Option<&RunAs>) -> RestCall {
            self.system_rest()
        }
        fn local_node(&self) -> String {
            "me".into()
        }
        fn master_term(&self) -> Option<u64> {
            Some(1)
        }
        fn persistent_tasks(&self) -> Value {
            self.tasks.lock().clone()
        }
        fn update_persistent_tasks(&self, f: cluster::TasksChange) -> Result<Value> {
            let mut t = self.tasks.lock();
            if let Some(v) = f(&*t) {
                *t = v;
            }
            Ok(t.clone())
        }
    }

    fn tid(i: usize) -> String {
        format!("t{i:04}")
    }

    /// `n` started transforms: a config and a state document each (state
    /// at checkpoint 7), and a persistent task assigned to this node.
    fn fake(n: usize, window: usize, visible: usize) -> Arc<FakeCluster> {
        let mut docs = Vec::new();
        let mut tasks = Value::Null;
        for i in 0..n {
            let id = tid(i);
            docs.push(json!({"_id": cluster::config_doc_id(&id), "_source": {
                "id": id, "doc_type": cluster::CONFIG_DOC_TYPE,
                "source": {"index": ["src"]}, "dest": {"index": format!("dst-{id}")},
                "pivot": {"group_by": {"k": {"terms": {"field": "k"}}}, "aggs": {"n": {"value_count": {"field": "k"}}}}
            }}));
            docs.push(json!({"_id": cluster::state_doc_id(&id), "_seq_no": i, "_primary_term": 1, "_source": {
                "id": id, "doc_type": cluster::STATE_DOC_TYPE,
                "state": {"task_state": "started", "indexer_state": "stopped", "checkpoint": 7}, "stats": {}
            }}));
            tasks = cluster::create_task(&tasks, &id, cluster::task_params(&id, None, None), "me", 1).unwrap().0;
        }
        docs.sort_by_key(key);
        let visible = visible.min(docs.len());
        Arc::new(FakeCluster { docs: Arc::new(docs), window, visible, tasks: Mutex::new(tasks), searches: Arc::new(AtomicUsize::new(0)) })
    }

    #[test]
    fn recovery_pages_past_the_search_window_and_keeps_every_transform_and_task() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = Arc::new(TransformService::open(tmp.path()));
        // 250 transforms = 500 documents, served at most 100 per response:
        // one capped search would see 50 configs and orphan 200 tasks.
        let n = 250;
        let c = fake(n, 100, usize::MAX);
        svc.set_cluster(Some(c.clone() as Arc<dyn TransformCluster>));
        svc.ensure_current().unwrap();
        {
            let map = svc.map.lock();
            assert_eq!(map.len(), n, "every config recovered");
            for i in 0..n {
                let e = &map[&tid(i)];
                assert_eq!(e.st.checkpoint, 7, "[{}] state recovered, not a fresh one", tid(i));
                assert_eq!(e.st.doc_seq, Some((i as u64, 1)), "[{}] state seq_no kept for the write fence", tid(i));
                assert!(e.st.started, "[{}] task adopted", tid(i));
            }
        }
        assert_eq!(cluster::transform_tasks(&c.tasks.lock()).len(), n, "no task removed as an orphan");
        assert!(c.searches.load(Ordering::SeqCst) >= 6, "the catalog was paged");
        svc.set_cluster(None);
    }

    #[test]
    fn a_catalog_read_short_of_its_total_fails_recovery_and_removes_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = Arc::new(TransformService::open(tmp.path()));
        // The index counts 40 documents but only 10 are ever returned.
        let c = fake(20, 100, 10);
        svc.set_cluster(Some(c.clone() as Arc<dyn TransformCluster>));
        let e = svc.ensure_current().unwrap_err();
        assert_eq!(e.status, 503, "{}", e.reason);
        assert!(svc.loaded_term.lock().is_none(), "an incomplete catalog is never installed");
        assert!(svc.map.lock().is_empty());
        assert_eq!(cluster::transform_tasks(&c.tasks.lock()).len(), 20, "no task removed");
        svc.set_cluster(None);
    }
}
