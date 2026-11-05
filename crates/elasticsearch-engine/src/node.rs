//! The single node: index registry, index-expression resolution (lists,
//! wildcards, exclusions, date math, aliases), aliases, index templates
//! (composable, component, legacy), cluster settings, scroll / PIT contexts
//! and the background maintenance loop. Cluster features (election,
//! replication, allocation) are out of scope: this is a 1-node cluster with
//! one primary shard per index.

use crate::error::{EsError, Result};
use crate::index::{new_uuid, IndexService};
use crate::search::{Cursor, SearchRequest, Target};
use crate::settings::{self, FlatSettings};
use parking_lot::{Mutex, RwLock};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const NODE_ID: &str = "rUst1cN0d3Id000000000w";
/// Re-exported from [`crate::version`], the one source of truth (the
/// cluster crate takes its version constants from here).
pub use crate::version::{ES_VERSION, LUCENE_VERSION, MIN_INDEX_COMPAT, MIN_WIRE_COMPAT};

// Hooks 1/2 (create with uuid, adopt / reopen from files): `hooks/node_ext.rs`.
#[path = "hooks/node_ext.rs"]
mod hooks_ext;
pub use hooks_ext::AdoptOptions;

#[derive(Clone, Copy, Debug)]
pub struct ResolveOpts {
    pub ignore_unavailable: bool,
    pub allow_no_indices: bool,
    pub expand_open: bool,
    pub expand_closed: bool,
    pub expand_hidden: bool,
}

impl Default for ResolveOpts {
    fn default() -> Self {
        Self { ignore_unavailable: false, allow_no_indices: true, expand_open: true, expand_closed: false, expand_hidden: false }
    }
}

impl ResolveOpts {
    pub fn from_params(p: &HashMap<String, String>) -> ResolveOpts {
        let mut o = ResolveOpts::default();
        if let Some(v) = p.get("ignore_unavailable") {
            o.ignore_unavailable = v == "true";
        }
        if let Some(v) = p.get("allow_no_indices") {
            o.allow_no_indices = v != "false";
        }
        if let Some(v) = p.get("expand_wildcards") {
            o.expand_open = false;
            for w in v.split(',') {
                match w {
                    "open" => o.expand_open = true,
                    "closed" => o.expand_closed = true,
                    "hidden" => o.expand_hidden = true,
                    "all" => {
                        o.expand_open = true;
                        o.expand_closed = true;
                        o.expand_hidden = true;
                    }
                    _ => {}
                }
            }
        }
        o
    }
}

pub struct ScrollCtx {
    pub targets: Vec<Target>,
    pub req: SearchRequest,
    pub cursors: Vec<Cursor>,
    pub keep_alive: Duration,
    pub expires: Instant,
    /// The context's own memory on the `accounting` breaker ([`context_charge`]).
    pub accounting: Option<crate::memory::Reservation>,
    /// `(index, shard)` targets freed with their copy
    /// ([`Node::free_shard_contexts`]): reported failed on every page.
    pub lost: Vec<(String, u32)>,
}

pub struct PitCtx {
    pub targets: Vec<Target>,
    pub keep_alive: Duration,
    pub expires: Instant,
    /// The context's own memory on the `accounting` breaker ([`context_charge`]).
    pub accounting: Option<crate::memory::Reservation>,
    /// See [`ScrollCtx::lost`].
    pub lost: Vec<(String, u32)>,
}

/// Report a context's `lost` targets in a page's `_shards` (each failed,
/// with ES's `shard_not_found_exception`), as ES reports a shard whose
/// reader context closed with it.
pub fn report_lost_targets(json: String, lost: &[(String, u32)]) -> String {
    if lost.is_empty() {
        return json;
    }
    let Ok(mut v) = serde_json::from_str::<Value>(&json) else { return json };
    let Some(s) = v.get_mut("_shards").and_then(Value::as_object_mut) else { return json };
    let n = lost.len() as u64;
    let get = |s: &serde_json::Map<String, Value>, k: &str| s.get(k).and_then(Value::as_u64).unwrap_or(0);
    let (total, failed) = (get(s, "total") + n, get(s, "failed") + n);
    s.insert("total".into(), json!(total));
    s.insert("failed".into(), json!(failed));
    let mut failures = s.remove("failures").and_then(|f| f.as_array().cloned()).unwrap_or_default();
    failures.extend(lost.iter().map(|(index, shard)| json!({"shard": shard, "index": index, "node": crate::node::NODE_ID,
        "reason": {"type": "shard_not_found_exception", "reason": "no such shard", "shard": shard.to_string(), "index": index}})));
    s.insert("failures".into(), Value::Array(failures));
    serde_json::to_string(&v).unwrap_or(json)
}

/// The `accounting` breaker's charge for a search context kept between
/// requests (PIT, scroll): its own bookkeeping, released when the context
/// is closed or expires. The views it retains (frozen deletions, copied
/// point tables, memtables and graph generations they keep alive) are
/// accounted by their owners.
pub fn context_charge(targets: &[Target], cursors: usize) -> crate::memory::Reservation {
    let per_target: u64 = targets.iter().map(|t| 256 + t.snapshot.as_ref().map_or(0, |v| v.num_leaves() as u64 * 128)).sum();
    let bytes = 1024 + per_target + cursors as u64 * 128;
    crate::memory::process_breakers().reserve_without_breaking(crate::memory::Kind::Accounting, bytes)
}

/// The reader context of a cluster coordinator's query_then_fetch query
/// phase (ES `ReaderContext`): the pinned targets its fetch phase reads.
/// Freed by the fetch (or an explicit free); abandoned ones expire.
pub struct FetchCtx {
    pub targets: Vec<Target>,
    /// Who opened it, for what: only the same requester may fetch or free it.
    pub owner: FetchOwner,
    pub expires: Instant,
    /// Estimated memory the context holds (see [`Node::fetch_contexts_bytes`]).
    pub bytes: u64,
    /// Its share of the node's [`FetchBudget`], given back on drop.
    pub charge: FetchCharge,
    /// The same `bytes` on the `accounting` breaker, as PIT and scroll
    /// contexts' [`context_charge`]: released on drop (fetched, freed,
    /// reaped or dropped with its copy). The budget only caps contexts; this
    /// is the one memory figure, and the views it pins are their owners'.
    pub accounting: crate::memory::Reservation,
}

impl FetchCtx {
    /// Every target still on the copy it pinned: the same open copy, not
    /// removed, reset or reopened since. ES frees a shard's reader contexts
    /// when the shard closes; a context over a copy gone since must neither
    /// be read nor keep its files. A promotion (a new primary term on the
    /// same copy) keeps them, as in ES: the copy stays open.
    pub fn copies_current(&self) -> bool {
        self.targets.iter().all(|t| match &t.shard {
            None => true,
            Some(s) => t.svc.shard(s.id).is_some_and(|now| Arc::ptr_eq(&now, s)),
        })
    }
}

/// The query_then_fetch reader contexts a node holds over all its engines
/// (a cluster node shares engine 0's with every shard engine, see
/// [`Node::fetch_budget`]): the caps are the node's, not an engine's.
#[derive(Default)]
pub struct FetchBudget {
    contexts: std::sync::atomic::AtomicUsize,
    bytes: std::sync::atomic::AtomicU64,
    /// Bytes past which contexts are refused; [`MAX_FETCH_CONTEXT_BYTES`]
    /// when unset.
    byte_limit: Option<u64>,
}

impl FetchBudget {
    /// A budget refusing contexts past `limit` estimated bytes (instead of
    /// [`MAX_FETCH_CONTEXT_BYTES`]).
    pub fn with_byte_limit(limit: u64) -> Self {
        Self { byte_limit: Some(limit), ..Self::default() }
    }

    /// Charge one context of `bytes`, unless that would pass (past 256
    /// contexts) the byte limit. No count cap: ES bounds query_then_fetch
    /// reader contexts by memory only (`search.max_open_scroll_context`
    /// counts scrolls alone), and a refused context fails its shard.
    fn charge(self: &Arc<Self>, bytes: u64) -> Option<FetchCharge> {
        use std::sync::atomic::Ordering::Relaxed;
        let n = self.contexts.fetch_add(1, Relaxed);
        let b = self.bytes.fetch_add(bytes, Relaxed);
        let charge = FetchCharge { budget: self.clone(), bytes };
        (n < 256 || b + bytes <= self.byte_limit.unwrap_or(MAX_FETCH_CONTEXT_BYTES)).then_some(charge)
    }

    /// Contexts and estimated bytes held.
    pub fn held(&self) -> (usize, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        (self.contexts.load(Relaxed), self.bytes.load(Relaxed))
    }
}

/// One context's share of a [`FetchBudget`].
pub struct FetchCharge {
    budget: Arc<FetchBudget>,
    bytes: u64,
}

impl Drop for FetchCharge {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        self.budget.contexts.fetch_sub(1, Relaxed);
        self.budget.bytes.fetch_sub(self.bytes, Relaxed);
    }
}

/// Drop reaped reader contexts off the request thread: the views they pin
/// can be the last references to whole segments (rare: only a full context
/// budget reaps there). One process-wide thread drops them in turn.
fn drop_off_thread<T: Send + 'static>(dead: Vec<T>) {
    static REAPER: std::sync::OnceLock<Option<std::sync::Mutex<std::sync::mpsc::Sender<Box<dyn Send>>>>> = std::sync::OnceLock::new();
    if dead.is_empty() {
        return;
    }
    let reaper = REAPER.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Box<dyn Send>>();
        std::thread::Builder::new().name("drop-contexts".into()).spawn(move || rx.into_iter().for_each(drop)).ok().map(|_| std::sync::Mutex::new(tx))
    });
    // Without the thread, or once it is gone, they drop here.
    if let Some(tx) = reaper {
        let _ = tx.lock().map(|tx| tx.send(Box::new(dead)));
    }
}

/// Remove the entries of `m` that fail `keep` and hand them back, so the
/// caller drops them (and the reader views they pin) outside `m`'s lock.
pub fn take_dead<V>(m: &mut HashMap<String, V>, keep: impl Fn(&V) -> bool) -> Vec<V> {
    let dead: Vec<String> = m.iter().filter(|(_, v)| !keep(v)).map(|(k, _)| k.clone()).collect();
    dead.iter().filter_map(|k| m.remove(k)).collect()
}

/// What a query_then_fetch reader context was opened by and for (ES keeps
/// the security context and shard of a `ReaderContext`): a fetch or free
/// that differs in any part finds no context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FetchOwner {
    /// The authenticated subject (user, realm, API key id, run-as);
    /// `None` with security off.
    pub principal: Option<String>,
    /// The index expression of the request.
    pub index: Option<String>,
    /// The cluster shard the request was routed to.
    pub shard: Option<u32>,
}

/// How long an unfetched query_then_fetch reader context lives unless the
/// dynamic cluster setting `search.default_keep_alive` says otherwise (ES's
/// default for a search's reader contexts).
pub const FETCH_CTX_KEEP_ALIVE: Duration = Duration::from_secs(300);
/// Most freed-before-registration query_then_fetch context ids a node
/// remembers (over all its engines).
pub const MAX_FETCH_CONTEXTS: usize = 4096;
/// Most estimated bytes they hold.
pub const MAX_FETCH_CONTEXT_BYTES: u64 = 64 << 20;

#[derive(Default)]
pub struct Templates {
    pub index: BTreeMap<String, Value>,
    pub component: BTreeMap<String, Value>,
    pub legacy: BTreeMap<String, Value>,
}

pub struct Node {
    pub data: PathBuf,
    pub name: String,
    pub cluster_name: String,
    pub cluster_uuid: String,
    indices: RwLock<BTreeMap<String, Arc<IndexService>>>,
    pub templates: RwLock<Templates>,
    pub cluster_settings: RwLock<(Map<String, Value>, Map<String, Value>)>,
    /// Node-level settings (`elasticsearch.yml` / `-E`): the fallback below
    /// persistent and transient cluster settings, never reported by
    /// `GET _cluster/settings`.
    pub node_settings: RwLock<Map<String, Value>>,
    /// Published transport endpoint; standalone HTTP tests use the virtual default.
    pub transport_address: RwLock<String>,
    /// Transport addresses of every node in this node's cluster (set by the
    /// cluster layer): a remote seeded with any of them is this cluster.
    pub cluster_transport_addresses: RwLock<Vec<String>>,
    pub scrolls: Mutex<HashMap<String, ScrollCtx>>,
    pub pits: Mutex<HashMap<String, PitCtx>>,
    /// Internal reader contexts of folds that page another node's shards
    /// (ES keeps these apart from user PITs: not in the open-PIT stats, not
    /// closed by `DELETE _pit`); see `esql::remote::FetchPit`.
    pub fold_contexts: Mutex<HashMap<String, PitCtx>>,
    /// query_then_fetch reader contexts by id.
    pub fetch_contexts: Mutex<HashMap<String, FetchCtx>>,
    /// Ids freed before their query phase registered them (a coordinator
    /// gave up on this copy while the phase ran): never registered, until
    /// they expire.
    pub fetch_freed: Mutex<HashMap<String, Instant>>,
    /// What every engine of this node's contexts hold together: its own, or
    /// (a cluster's shard engine) engine 0's.
    pub fetch_budget: Mutex<Arc<FetchBudget>>,
    /// Circuit breakers (the process's, configured from this node's settings).
    pub breakers: &'static crate::memory::Breakers,
    /// Task manager (`_tasks`, reindex/by-query tasks).
    pub tasks: crate::tasks::TaskManager,
    /// Async searches by id.
    pub async_searches: Mutex<HashMap<String, Arc<crate::async_search::AsyncSearch>>>,
    pub ccs_stats: Mutex<crate::ccs_stats::CcsStats>,
    /// Data streams by name (persisted in the cluster state).
    pub data_streams: RwLock<BTreeMap<String, crate::datastream::DataStream>>,
    create_lock: Mutex<()>,
    pub ds_create_lock: Mutex<()>,
    pub started: Instant,
    /// Security (authn/authz state); inert unless `xpack.security.enabled`.
    pub security: Arc<crate::security::Security>,
    /// ILM / SLM / data stream lifecycle state (`crate::ilm`).
    pub ilm: crate::ilm::IlmService,
    /// Snapshot repositories and snapshots.
    pub snapshots: crate::snapshot::SnapshotService,
    /// Node-level dynamic mapping hook (A6), installed on every index.
    mapping_hook: RwLock<Option<Arc<dyn crate::index::MappingUpdateHook>>>,
    /// Where rollover conditions read an index's stats (a cluster: the
    /// cluster-wide primaries, see [`crate::datastream::RolloverStatsHook`]).
    pub(crate) rollover_stats_hook: RwLock<Option<crate::datastream::RolloverStatsHook>>,
    /// Per local shard copy: the max seq_no last seen and since when (the
    /// idle flush in [`Node::maintenance`]).
    write_activity: Mutex<HashMap<usize, (i64, Instant)>>,
}

/// `invalid_index_name_exception` for a name that cannot be created.
pub fn validate_index_name(name: &str) -> Result<()> {
    is_valid_name(name).map_err(|why| EsError::invalid_index_name(name, &why))
}

fn is_valid_name(name: &str) -> std::result::Result<(), String> {
    if name.is_empty() {
        return Err("must not be empty".into());
    }
    if name != name.to_lowercase() {
        return Err("must be lowercase".into());
    }
    if name.starts_with(['-', '_', '+']) {
        return Err("must not start with '_', '-', or '+'".into());
    }
    if name == "." || name == ".." {
        return Err("must not be '.' or '..'".into());
    }
    for c in ['\\', '/', '*', '?', '"', '<', '>', '|', ' ', ',', '#', ':'] {
        if name.contains(c) {
            return Err(format!("must not contain the following characters [ , \", *, \\, <, |, ,, >, /, ?]"));
        }
    }
    if name.len() > 255 {
        return Err("index name is too long, (> 255)".into());
    }
    Ok(())
}

/// Resolve `<static{date_math{format}}>` index names.
pub fn resolve_date_math(name: &str) -> Result<String> {
    let name = if name.contains('%') { percent_decode(name) } else { name.to_string() };
    if !(name.starts_with('<') && name.ends_with('>')) {
        return Ok(name);
    }
    let inner = &name[1..name.len() - 1];
    let mut out = String::new();
    let mut rest = inner;
    while let Some(start) = rest.find('{') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        // find matching close brace (format may contain nested braces)
        let mut depth = 1;
        let mut end = 0;
        for (i, c) in after.char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i;
                        break;
                    }
                }
                _ => {}
            }
        }
        let expr = &after[..end];
        let (math, fmt, tz) = match expr.find('{') {
            Some(i) => {
                let f = &expr[i + 1..expr.len() - 1];
                let (f, tz) = match f.split_once('|') {
                    Some((a, b)) => (a, Some(b)),
                    None => (f, None),
                };
                (&expr[..i], f.to_string(), tz)
            }
            None => (expr, "yyyy.MM.dd".to_string(), None),
        };
        let tz = match tz {
            Some(z) => crate::date::Tz::parse(z)?,
            None => crate::date::Tz::UTC,
        };
        let df = crate::date::DateFormatter::new(&fmt.replace("YYYY", "yyyy"))?;
        let ms = crate::date::parse_date_math(math, &df, tz, crate::date::now_millis(), false)?;
        out.push_str(&df.format(ms, tz));
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Merge template mappings: like `deep_merge`, but a field whose `type`
/// changes is replaced wholesale (parameters of the old type are dropped).
pub(crate) fn merge_mapping(dst: &mut Value, src: &Value) {
    match (dst.as_object_mut(), src.as_object()) {
        (Some(d), Some(s)) => {
            if let (Some(Value::String(dt)), Some(Value::String(st))) = (d.get("type"), s.get("type")) {
                if dt != st {
                    *dst = src.clone();
                    return;
                }
            }
            for (k, v) in s {
                match d.get_mut(k) {
                    // Template merges combine the raw mappings
                    // (`MapperService.merge` -> `XContentHelper.merge`):
                    // dynamic templates by name, a same-named one merged
                    // into the earlier one in place.
                    Some(cur) if k == "dynamic_templates" && cur.is_array() && v.is_array() => raw_merge_value(cur, v),
                    Some(cur) if cur.is_object() && v.is_object() => merge_mapping(cur, v),
                    _ => {
                        d.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        _ => {
            if !src.is_null() {
                *dst = src.clone();
            }
        }
    }
}

/// Composable template mapping merge (`MapperService.merge` of the raw
/// component mappings, `XContentHelper.merge` with `RawFieldMappingMerge`).
pub(crate) fn compose_mapping(dst: &mut Value, src: &Value) {
    let src = unwrap_typed(src);
    if src.is_null() {
        return;
    }
    if !dst.is_object() || !src.is_object() {
        *dst = src.clone();
        return;
    }
    raw_merge_in(dst, &src, None);
}

/// `XContentHelper.merge(parent, first, second, RawFieldMappingMerge)`.
fn raw_merge_in(first: &mut Value, second: &Value, parent: Option<&str>) {
    let (Some(f), Some(s)) = (first.as_object_mut(), second.as_object()) else { return };
    for (k, v) in s {
        let Some(base) = f.get_mut(k) else {
            f.insert(k.clone(), v.clone());
            continue;
        };
        match (base.is_object(), v.is_object(), base.is_array() && v.is_array()) {
            (true, true, _) => {
                if parent == Some("properties") {
                    // Two mappings of the same field: objects merge their
                    // properties (and keep `subobjects`); anything else is
                    // replaced by the later mapping.
                    let ty = |m: &Value| m.get("type").and_then(Value::as_str).map(str::to_string).or_else(|| m.get("properties").is_some().then(|| "object".to_string()));
                    let mergeable = |t: Option<String>| matches!(t.as_deref(), Some("object" | "nested"));
                    if mergeable(ty(base)) && mergeable(ty(v)) {
                        let mut merged = Map::new();
                        if let Some(p) = base.get("properties") {
                            merged.insert("properties".into(), p.clone());
                        }
                        if let Some(so) = base.get("subobjects") {
                            merged.insert("subobjects".into(), so.clone());
                        }
                        let mut merged = Value::Object(merged);
                        raw_merge_in(&mut merged, v, None);
                        *base = merged;
                    } else {
                        *base = v.clone();
                    }
                } else {
                    raw_merge_in(base, v, Some(k));
                }
            }
            (_, _, true) => raw_merge_value(base, v),
            _ => *base = v.clone(),
        }
    }
}

/// `XContentHelper.merge` with `RawFieldMappingMerge` outside `properties`:
/// maps merge recursively; lists of single-key maps merge by key (new keys
/// appended); other lists combine (the second list's values first, then the
/// first's not already present); anything else is replaced.
fn raw_merge_value(first: &mut Value, second: &Value) {
    match (first, second) {
        (Value::Object(f), Value::Object(s)) => {
            for (k, v) in s {
                match f.get_mut(k) {
                    Some(cur) if (cur.is_object() && v.is_object()) || (cur.is_array() && v.is_array()) => raw_merge_value(cur, v),
                    _ => {
                        f.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (Value::Array(f), Value::Array(s)) => {
            let one = |a: &Vec<Value>| a.iter().all(|x| x.as_object().is_some_and(|o| o.len() == 1));
            if one(f) && one(s) {
                let key = |t: &Value| t.as_object().and_then(|o| o.keys().next().cloned()).unwrap_or_default();
                for t in s {
                    match f.iter_mut().find(|c| key(c) == key(t)) {
                        Some(slot) => raw_merge_value(slot, t),
                        None => f.push(t.clone()),
                    }
                }
            } else {
                let mut merged = s.clone();
                for x in f.iter() {
                    if !merged.contains(x) {
                        merged.push(x.clone());
                    }
                }
                *f = merged;
            }
        }
        (f, s) => *f = s.clone(),
    }
}

fn deep_merge(dst: &mut Value, src: &Value) {
    if src.is_null() {
        return;
    }
    crate::index::merge_json(dst, src)
}

impl Node {
    /// `action.destructive_requires_name`: transient, then persistent, then
    /// node setting; ES 8 defaults it to `true`.
    pub fn destructive_requires_name(&self) -> bool {
        let as_bool = |v: &Value| v.as_str().map(|s| s != "false").unwrap_or(v.as_bool().unwrap_or(true));
        let key = "action.destructive_requires_name";
        let cs = self.cluster_settings.read();
        cs.1.get(key).or(cs.0.get(key)).map(as_bool).or_else(|| self.node_settings.read().get(key).map(as_bool)).unwrap_or(true)
    }

    pub fn open(data: &Path) -> Result<Arc<Node>> {
        Self::open_named(data, &hostname())
    }

    pub fn open_named(data: &Path, name: &str) -> Result<Arc<Node>> {
        Self::open_with_security_named(data, Arc::new(crate::security::Security::open(data)), name)
    }

    /// Shards on the same node share native users, roles, keys and tokens.
    pub fn open_with_security(data: &Path, security: Arc<crate::security::Security>) -> Result<Arc<Node>> {
        Self::open_with_security_named(data, security, &hostname())
    }

    pub fn open_with_security_named(data: &Path, security: Arc<crate::security::Security>, name: &str) -> Result<Arc<Node>> {
        std::fs::create_dir_all(data.join("indices"))?;
        std::fs::create_dir_all(data.join("_state"))?;
        let state = load_cluster_state(&data.join("_state/cluster.json"))?;
        let cluster_uuid = state["cluster_uuid"].as_str().map(str::to_string).unwrap_or_else(new_uuid);
        let mut templates = Templates::default();
        for (k, field) in [("index_templates", 0), ("component_templates", 1), ("legacy_templates", 2)] {
            if let Some(m) = state[k].as_object() {
                for (n, v) in m {
                    match field {
                        0 => templates.index.insert(n.clone(), v.clone()),
                        1 => templates.component.insert(n.clone(), v.clone()),
                        _ => templates.legacy.insert(n.clone(), v.clone()),
                    };
                }
            }
        }
        let mut persistent = state["persistent"].as_object().cloned().unwrap_or_default();
        crate::settings::archive_unknown_or_invalid_settings("persistent", &mut persistent);
        let data_streams: BTreeMap<String, crate::datastream::DataStream> = serde_json::from_value(state["data_streams"].clone()).unwrap_or_default();
        let node = Node {
            data: data.to_path_buf(),
            name: name.to_string(),
            cluster_name: std::env::var("ES_CLUSTER_NAME").unwrap_or_else(|_| "elasticsearch".into()),
            cluster_uuid,
            indices: RwLock::new(BTreeMap::new()),
            templates: RwLock::new(templates),
            cluster_settings: RwLock::new((persistent, Map::new())),
            node_settings: RwLock::new(Map::new()),
            transport_address: RwLock::new(crate::ccr::remote::TRANSPORT_ADDRESS.into()),
            cluster_transport_addresses: RwLock::new(Vec::new()),
            scrolls: Mutex::new(HashMap::new()),
            pits: Mutex::new(HashMap::new()),
            fold_contexts: Mutex::new(HashMap::new()),
            fetch_contexts: Mutex::new(HashMap::new()),
            fetch_freed: Mutex::new(HashMap::new()),
            fetch_budget: Mutex::new(Arc::default()),
            breakers: crate::memory::process_breakers(),
            tasks: Default::default(),
            async_searches: Mutex::new(HashMap::new()),
            ccs_stats: Mutex::new(Default::default()),
            data_streams: RwLock::new(data_streams),
            create_lock: Mutex::new(()),
            ds_create_lock: Mutex::new(()),
            started: Instant::now(),
            security,
            ilm: crate::ilm::IlmService::load(data),
            snapshots: crate::snapshot::SnapshotService::load(data),
            mapping_hook: RwLock::new(None),
            rollover_stats_hook: RwLock::new(None),
            write_activity: Mutex::new(HashMap::new()),
        };
        // Publish the synonyms sets before the indices recover: a
        // `synonyms_set` filter reads them when its shard's analyzers build.
        let _ = crate::search_ext::ext(&node);
        for e in std::fs::read_dir(data.join("indices"))?.flatten() {
            if e.path().join("index_meta.json").exists() {
                match IndexService::open(&e.path()) {
                    Ok(svc) => {
                        node.indices.write().insert(svc.name.clone(), Arc::new(svc));
                    }
                    Err(err) => tracing::error!("failed to open index at {}: {err}", e.path().display()),
                }
            }
        }
        node.persist_state()?;
        // Breakers are the process's: only the first node opened applies
        // its persisted breaker settings.
        crate::memory::configure_at_open(&node);
        crate::script::rate_limit::configure_at_open(&node)?;
        Ok(Arc::new(node))
    }

    pub fn persist_state(&self) -> Result<()> {
        // Concurrent callers (Kibana PUTs many templates at once) share the
        // tmp path; serialize snapshot + write + rename so none fails with
        // ENOENT and an older snapshot never lands after a newer one.
        static PERSIST: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _g = PERSIST.lock().unwrap_or_else(|e| e.into_inner());
        let t = self.templates.read();
        let v = json!({
            "cluster_uuid": self.cluster_uuid,
            "index_templates": t.index,
            "component_templates": t.component,
            "legacy_templates": t.legacy,
            "persistent": self.cluster_settings.read().0,
            "data_streams": serde_json::to_value(&*self.data_streams.read())?,
        });
        crate::hooks::write_durable(&self.data.join("_state/cluster.json"), &serde_json::to_vec_pretty(&v)?)?;
        Ok(())
    }

    /// Borrow each open service without allocating a parallel ownership table.
    /// Callers must not mutate this node's index registry inside the callback.
    pub fn try_visit_indices<E>(&self, mut visit: impl FnMut(&Arc<IndexService>)->std::result::Result<(),E>)->std::result::Result<(),E> {
        for service in self.indices.read().values() { visit(service)?; } Ok(())
    }

    pub fn all_indices(&self) -> Vec<Arc<IndexService>> {
        self.indices.read().values().cloned().collect()
    }

    pub fn get_index(&self, name: &str) -> Option<Arc<IndexService>> {
        self.indices.read().get(name).cloned()
    }

    /// `svc` is still the index registered as `name` (not deleted or
    /// replaced since it was resolved). No Arc clone.
    pub fn is_registered_index(&self, name: &str, svc: &Arc<IndexService>) -> bool {
        self.indices.read().get(name).is_some_and(|s| Arc::ptr_eq(s, svc))
    }

    /// Skipped by wildcards unless `expand_wildcards` includes hidden: ES
    /// skips `index.hidden` and system indices, not every dot-prefixed name.
    fn is_hidden(svc: &IndexService) -> bool {
        crate::system_indices::is_system(&svc.name) || svc.setting("index.hidden").map(|s| s == "true").unwrap_or(false)
    }

    /// An internal refresh of `expr` (what ES features issue as a
    /// system-origin `RefreshRequest`): missing indices are ignored.
    pub fn refresh_indices(&self, expr: &str) -> Result<()> {
        let opts = ResolveOpts { ignore_unavailable: true, allow_no_indices: true, ..Default::default() };
        for t in self.resolve(expr, opts).unwrap_or_default() {
            t.svc.refresh()?;
        }
        Ok(())
    }

    /// Resolve an index expression to concrete targets (alias filters kept).
    pub fn resolve(&self, expr: &str, opts: ResolveOpts) -> Result<Vec<Target>> {
        let indices = self.indices.read();
        let mut out: Vec<(String, Option<Value>)> = Vec::new();
        // Alias `search_routing` per index (`None`: every shard).
        let mut routing: HashMap<String, Option<String>> = HashMap::new();
        let alias_routing = |def: &Value| def.get("search_routing").and_then(|v| v.as_str().map(str::to_string).or_else(|| (!v.is_null()).then(|| v.to_string())));
        let parts: Vec<String> = if expr.is_empty() { vec!["_all".into()] } else { split_expr(expr) };
        let mut any_wildcard = false;
        // A data stream named with `::failures` resolves even when its failure store is empty.
        let mut empty_failure_store = false;
        for raw in parts {
            let (exclude, p) = match raw.strip_prefix('-') {
                Some(r) if !out.is_empty() => (true, r.to_string()),
                _ => (false, raw.clone()),
            };
            let p = resolve_date_math(&p)?;
            // Index component selectors: `name::data` (the default) and `name::failures`.
            let (p, failures) = match crate::datastream::split_selector(&p)? {
                (base, Some(crate::datastream::Selector::Failures)) => (base.to_string(), Some(p.clone())),
                (base, _) => (base.to_string(), None),
            };
            let mut matched: Vec<(String, Option<Value>)> = Vec::new();
            // `expand_wildcards=none`: a wildcard pattern is a literal name.
            let no_expand = !opts.expand_open && !opts.expand_closed && !opts.expand_hidden && p != "_all" && p != "*";
            if let Some(full) = &failures {
                // Failure store indices of the matching data streams.
                let is_pattern = (p == "_all" || p == "*" || p.contains('*')) && !no_expand;
                if is_pattern {
                    any_wildcard = true;
                }
                let pat = if p == "_all" { "*" } else { p.as_str() };
                let dss = if is_pattern { self.data_streams_matching(pat, opts.expand_hidden) } else { self.data_stream(&p).into_iter().collect() };
                if dss.is_empty() && !is_pattern && !exclude && !opts.ignore_unavailable {
                    let exists = indices.contains_key(&p) || indices.values().any(|s| s.aliases().contains_key(&p));
                    return Err(EsError::index_not_found(if exists { full } else { &p }));
                }
                if !is_pattern && !dss.is_empty() {
                    empty_failure_store = true;
                }
                for ds in dss {
                    for fi in &ds.failure_indices {
                        if indices.get(fi).map(|s| s.is_open() || opts.expand_closed).unwrap_or(false) {
                            note_routing(&mut routing, fi, None);
                            matched.push((fi.clone(), None));
                        }
                    }
                }
            } else if (p == "_all" || p == "*" || p.contains('*')) && !no_expand {
                any_wildcard = true;
                let pat = if p == "_all" { "*".to_string() } else { p.clone() };
                for (n, svc) in indices.iter() {
                    let hidden = Self::is_hidden(svc);
                    if hidden && !opts.expand_hidden && !(pat.starts_with('.') && n.starts_with('.')) {
                        continue;
                    }
                    let open = svc.is_open();
                    if (open && !opts.expand_open) || (!open && !opts.expand_closed) {
                        continue;
                    }
                    if crate::mapping::wildcard_match(&pat, n) {
                        matched.push((n.clone(), None));
                        note_routing(&mut routing, n, None);
                    }
                }
                // Data streams matching the pattern expand to their backing indices.
                for ds in self.data_streams_matching(&pat, opts.expand_hidden) {
                    for bi in &ds.indices {
                        if indices.get(bi).map(|s| s.is_open() || opts.expand_closed).unwrap_or(false) {
                            matched.push((bi.clone(), None));
                        }
                    }
                }
                // Aliases matching the pattern too.
                if p != "_all" && p != "*" {
                    for (n, svc) in indices.iter() {
                        for (a, def) in svc.aliases() {
                            if crate::mapping::wildcard_match(&pat, &a) {
                                matched.push((n.clone(), def.get("filter").cloned()));
                                note_routing(&mut routing, n, alias_routing(&def));
                            }
                        }
                    }
                }
            } else if let Some(svc) = indices.get(&p) {
                if !svc.is_open() && !opts.ignore_unavailable && !opts.expand_closed && !exclude {
                    return Err(EsError::index_closed(&svc.name, &svc.uuid));
                }
                if svc.is_open() || opts.expand_closed {
                    matched.push((p.clone(), None));
                    note_routing(&mut routing, &p, None);
                }
            } else if let Some(backing) = self.data_stream_indices(&p) {
                for b in backing.into_iter().filter(|b| indices.contains_key(b)) {
                    note_routing(&mut routing, &b, None);
                    matched.push((b, None));
                }
            } else {
                let mut found = false;
                for (n, svc) in indices.iter() {
                    if let Some(def) = svc.aliases().get(&p) {
                        found = true;
                        matched.push((n.clone(), def.get("filter").cloned()));
                        note_routing(&mut routing, n, alias_routing(def));
                    }
                }
                if !found && !exclude && !opts.ignore_unavailable {
                    return Err(EsError::index_not_found(&p));
                }
            }
            if exclude {
                out.retain(|(n, _)| !matched.iter().any(|(m, _)| m == n));
            } else {
                for m in matched {
                    if !out.iter().any(|(n, _)| *n == m.0) {
                        out.push(m);
                    } else if m.1.is_none() {
                        // Direct reference wins over a filtered alias.
                        if let Some(e) = out.iter_mut().find(|(n, _)| *n == m.0) {
                            e.1 = None;
                        }
                    } else if let Some(e) = out.iter_mut().find(|(n, _)| *n == m.0) {
                        // Two filtered aliases of one index: the filters are OR-ed.
                        if let (Some(a), Some(b)) = (e.1.as_ref(), m.1.as_ref()) {
                            if a != b {
                                e.1 = Some(json!({"bool": {"should": [a.clone(), b.clone()], "minimum_should_match": 1}}));
                            }
                        }
                    }
                }
            }
        }
        if out.is_empty() && !empty_failure_store && (!opts.allow_no_indices || (!any_wildcard && !opts.ignore_unavailable)) {
            return Err(EsError::index_not_found(if expr.is_empty() { "_all" } else { expr }));
        }
        out.into_iter()
            .filter_map(|(n, f)| indices.get(&n).map(|svc| Target { alias_filter: f, routing: routing.get(&n).cloned().flatten(), ..Target::new(svc.clone()) }))
            .map(|mut target| {
                if let Some(shard) = crate::search::selected_read_shard() {
                    target.shard = Some(target.svc.shard(shard).ok_or_else(|| EsError::new(503, "no_shard_available_action_exception", format!("[{}][{shard}] no active shard copy", target.svc.name)))?);
                }
                Ok(target)
            }).collect()
    }

    /// Concrete index for a write against `name` (alias → write index).
    /// Auto-creates the index when it does not exist.
    pub fn write_index(&self, name: &str, auto_create: bool) -> Result<Arc<IndexService>> {
        let name = resolve_date_math(name)?;
        if let Some(svc) = self.get_index(&name) {
            return Ok(svc);
        }
        if let Some(ds) = self.data_stream(&name) {
            if auto_create && ds.rollover_on_write {
                return self.data_stream_write_index(&ds);
            }
            return self.get_index(ds.write_index()).ok_or_else(|| EsError::index_not_found(&name));
        }
        let mut candidates = Vec::new();
        for svc in self.all_indices() {
            if let Some(def) = svc.aliases().get(&name) {
                candidates.push((svc.clone(), def.get("is_write_index").and_then(Value::as_bool)));
            }
        }
        if !candidates.is_empty() {
            if let Some((s, _)) = candidates.iter().find(|(_, w)| *w == Some(true)) {
                return Ok(s.clone());
            }
            if candidates.len() == 1 && candidates[0].1 != Some(false) {
                return Ok(candidates[0].0.clone());
            }
            return Err(EsError::illegal_argument(format!(
                "no write index is defined for alias [{name}]. The write index may be explicitly disabled using is_write_index=false or the alias points to multiple indices without one being designated as a write index"
            )));
        }
        let auto = self.cluster_settings.read().0.get("action.auto_create_index").and_then(Value::as_str).map(|s| s != "false").unwrap_or(true);
        // ES|QL views are not write targets.
        if !auto_create || !auto || crate::esql::views::exists(self, &name) {
            return Err(EsError::index_not_found(&name));
        }
        if self.data_stream_template(&name).is_some() {
            return match self.create_data_stream(&name) {
                Ok(ds) => self.get_index(ds.write_index()).ok_or_else(|| EsError::index_not_found(&name)),
                Err(e) if e.kind == "resource_already_exists_exception" => self.write_index(&name, false),
                Err(e) => Err(e),
            };
        }
        match self.create_index(&name, &json!({})) {
            Ok(svc) => Ok(svc),
            Err(e) if e.kind == "resource_already_exists_exception" => self.get_index(&name).ok_or(e),
            Err(e) => Err(e),
        }
    }

    /// The routing of a single-document operation addressed to `name`: an
    /// alias with `index_routing` imposes it (ES `resolveIndexRouting`; a
    /// different explicit routing is rejected), otherwise `routing`.
    pub fn index_routing_for(&self, name: &str, routing: Option<&str>) -> Result<Option<String>> {
        let name = resolve_date_math(name)?;
        if self.get_index(&name).is_some() || self.data_stream(&name).is_some() {
            return Ok(routing.map(str::to_string));
        }
        let defs: Vec<Value> = self.all_indices().iter().filter_map(|s| s.aliases().get(&name).cloned()).collect();
        let def = match defs.as_slice() {
            [one] => one.clone(),
            many => match many.iter().find(|d| d.get("is_write_index").and_then(Value::as_bool) == Some(true)) {
                Some(w) => w.clone(),
                None => return Ok(routing.map(str::to_string)),
            },
        };
        let Some(ar) = def.get("index_routing").and_then(|v| v.as_str().map(str::to_string).or_else(|| (!v.is_null()).then(|| v.to_string()))) else {
            return Ok(routing.map(str::to_string));
        };
        if ar.contains(',') {
            return Err(EsError::illegal_argument(format!("index/indexing routing value [{ar}] cannot contain multiple routing values for alias [{name}]")));
        }
        match routing {
            Some(r) if r != ar => Err(EsError::illegal_argument(format!("Alias [{name}] has index routing associated with it [{ar}], and was provided with routing value [{r}], rejecting operation"))),
            _ => Ok(Some(ar)),
        }
    }

    /// Settings, mappings and aliases from matching templates.
    pub fn template_for(&self, name: &str) -> (Value, Value, Value) {
        let t = self.templates.read();
        let mut settings = json!({});
        let mut mappings = json!({});
        let mut aliases = json!({});
        // Legacy templates by ascending order.
        let mut legacy: Vec<&Value> = t
            .legacy
            .values()
            .filter(|v| pattern_list(&v["index_patterns"]).iter().any(|p| crate::mapping::wildcard_match(p, name)))
            .collect();
        legacy.sort_by_key(|v| v["order"].as_i64().unwrap_or(0));
        for v in legacy {
            deep_merge(&mut settings, &normalize_settings_json(&v["settings"]));
            merge_mapping(&mut mappings, &unwrap_typed(&v["mappings"]));
            deep_merge(&mut aliases, &v["aliases"]);
        }
        // Composable: highest priority match wins.
        if let Some((_, it)) = Self::best_index_template(&t, name) {
            Self::compose_into(&t, it, &mut settings, &mut mappings, &mut aliases);
        }
        (settings, mappings, aliases)
    }

    /// The highest-priority composable template whose patterns match `name`.
    fn best_index_template<'a>(t: &'a Templates, name: &str) -> Option<(&'a String, &'a Value)> {
        t.index
            .iter()
            .filter(|(_, v)| pattern_list(&v["index_patterns"]).iter().any(|p| crate::mapping::wildcard_match(p, name)))
            .max_by_key(|(_, v)| v["priority"].as_i64().unwrap_or(0))
    }

    /// Merge a composable template's component templates (in order) and
    /// then its own `template` block into the accumulators.
    fn compose_into(t: &Templates, it: &Value, settings: &mut Value, mappings: &mut Value, aliases: &mut Value) -> Option<Value> {
        let mut parts: Vec<Value> = Vec::new();
        for c in it["composed_of"].as_array().cloned().unwrap_or_default() {
            if let Some(ct) = c.as_str().and_then(|n| t.component.get(n)) {
                parts.push(ct["template"].clone());
            }
        }
        parts.push(it["template"].clone());
        let mut lifecycle = None;
        for p in parts {
            deep_merge(settings, &normalize_settings_json(&p["settings"]));
            compose_mapping(mappings, &p["mappings"]);
            // An alias redefined by a later template replaces the earlier definition.
            if let (Some(dst), Some(src)) = (aliases.as_object_mut(), p["aliases"].as_object()) {
                for (k, v) in src {
                    dst.insert(k.clone(), v.clone());
                }
            }
            if p["lifecycle"].is_object() {
                lifecycle = Some(p["lifecycle"].clone());
            }
        }
        lifecycle
    }

    /// `POST /_index_template/_simulate[/{name}]`: resolve a composable
    /// template (`body`, or the stored `name` when `body` is null) into
    /// the `{template: {settings, mappings, aliases}, overlapping}` shape.
    /// `ComposableIndexTemplate.PARSER` structure checks for a simulate body.
    fn check_template_body(body: &Value) -> Result<()> {
        const FIELDS: &[&str] = &["index_patterns", "composed_of", "template", "priority", "version", "_meta", "data_stream", "allow_auto_create", "ignore_missing_component_templates", "deprecated"];
        const TEMPLATE: &[&str] = &["settings", "mappings", "aliases", "lifecycle", "data_stream_options"];
        let Some(o) = body.as_object() else { return Ok(()) };
        for (k, v) in o {
            if !FIELDS.contains(&k.as_str()) {
                return Err(EsError::unknown_field_xc("index_template", k, None, FIELDS));
            }
            if k == "template" {
                if let Some(t) = v.as_object() {
                    for tk in t.keys() {
                        if !TEMPLATE.contains(&tk.as_str()) {
                            return Err(EsError::field_failure("index_template", "template", None, EsError::unknown_field_xc("template", tk, Some("template"), TEMPLATE)));
                        }
                    }
                }
            }
        }
        if !o.contains_key("index_patterns") {
            return Err(EsError::illegal_argument("Required [index_patterns]"));
        }
        Ok(())
    }

    pub fn simulate_template(&self, name: Option<&str>, body: &Value, include_defaults: bool) -> Result<Value> {
        let t = self.templates.read();
        let it = if body.is_null() {
            let n = name.ok_or_else(|| EsError::action_validation(&["index template body or name is required"]))?;
            t.index.get(n).cloned().ok_or_else(|| EsError::illegal_argument(format!("unable to simulate template [{n}] that does not exist")))?
        } else {
            Self::check_template_body(body)?;
            body.clone()
        };
        Self::check_missing_components(&t, &it, name.unwrap_or("simulate_template"))?;
        let patterns = pattern_list(&it["index_patterns"]);
        let overlaps = |other: &Value| {
            let other = pattern_list(&other["index_patterns"]);
            patterns.iter().any(|a| other.iter().any(|b| crate::mapping::wildcard_match(a, b) || crate::mapping::wildcard_match(b, a)))
        };
        let mut overlapping: Vec<Value> = Vec::new();
        for (k, v) in t.legacy.iter().filter(|(_, v)| overlaps(v)) {
            overlapping.push(json!({"name": k, "index_patterns": pattern_list(&v["index_patterns"])}));
        }
        for (k, v) in t.index.iter().filter(|(k, v)| Some(k.as_str()) != name && overlaps(v)) {
            overlapping.push(json!({"name": k, "index_patterns": pattern_list(&v["index_patterns"])}));
        }
        let (mut settings, mut mappings, mut aliases) = (json!({}), json!({}), json!({}));
        let lifecycle = Self::compose_into(&t, &it, &mut settings, &mut mappings, &mut aliases);
        let dso = Self::simulated_dso(&t, &it);
        let mut out = Self::simulated(&settings, mappings, aliases, lifecycle, overlapping, include_defaults, self);
        if let Some(d) = dso {
            out["template"]["data_stream_options"] = d;
        }
        Ok(out)
    }

    /// `data_stream_options` of a simulated data stream template.
    fn simulated_dso(t: &Templates, it: &Value) -> Option<Value> {
        if !it.get("data_stream").map(Value::is_object).unwrap_or(false) {
            return None;
        }
        let mut parts = Vec::new();
        for c in it["composed_of"].as_array().cloned().unwrap_or_default() {
            if let Some(ct) = c.as_str().and_then(|n| t.component.get(n)) {
                parts.push(ct["template"].get("data_stream_options").cloned());
            }
        }
        parts.push(it["template"].get("data_stream_options").cloned());
        Some(crate::datastream::compose_data_stream_options(parts.into_iter()).unwrap_or_else(|| json!({})))
    }

    /// `POST /_index_template/_simulate_index/{index}`: the template that
    /// would apply if `index` were created now. A `body` template competes
    /// with the stored ones by priority.
    pub fn simulate_index(&self, index: &str, body: &Value, include_defaults: bool) -> Result<Value> {
        let t = self.templates.read();
        let matches = |v: &Value| pattern_list(&v["index_patterns"]).iter().any(|p| crate::mapping::wildcard_match(p, index));
        let prio = |v: &Value| v["priority"].as_i64().unwrap_or(0);
        let new_body = body.as_object().map(|_| body);
        if let Some(b) = new_body {
            Self::check_template_body(b)?;
            Self::check_missing_components(&t, b, "simulate_index_template")?;
        }
        let mut cands: Vec<(String, &Value)> = t.index.iter().filter(|(_, v)| matches(v)).map(|(k, v)| (k.clone(), v)).collect();
        if let Some(b) = new_body.filter(|b| matches(b)) {
            cands.push((String::new(), b));
        }
        let Some(best) = cands.iter().max_by_key(|(_, v)| prio(v)).cloned() else {
            return Ok(json!({}));
        };
        let mut overlapping: Vec<Value> = Vec::new();
        let best_patterns = pattern_list(&best.1["index_patterns"]);
        for (k, v) in t.legacy.iter() {
            let other = pattern_list(&v["index_patterns"]);
            if best_patterns.iter().any(|a| other.iter().any(|b| crate::mapping::wildcard_match(a, b) || crate::mapping::wildcard_match(b, a))) {
                overlapping.push(json!({"name": k, "index_patterns": other}));
            }
        }
        for (k, v) in cands.iter().filter(|(k, _)| !k.is_empty() && *k != best.0) {
            overlapping.push(json!({"name": k, "index_patterns": pattern_list(&v["index_patterns"])}));
        }
        let (mut settings, mut mappings, mut aliases) = (json!({}), json!({}), json!({}));
        let lifecycle = Self::compose_into(&t, best.1, &mut settings, &mut mappings, &mut aliases);
        // An existing data stream of that name adds its setting and mapping
        // overrides (`TransportSimulateIndexTemplateAction.collectSettings` /
        // `collectMappings`).
        if best.1.get("data_stream").is_some_and(Value::is_object) {
            if let Some(ds) = self.data_stream(index) {
                if !ds.settings.is_empty() {
                    let mut flat = crate::settings::normalize_index_settings(&settings);
                    for (k, v) in &ds.settings {
                        if v.is_null() {
                            flat.remove(k);
                        } else {
                            flat.insert(k.clone(), v.clone());
                        }
                    }
                    flat.retain(|_, v| !v.is_null());
                    settings = crate::settings::unflatten(&flat);
                }
                if !ds.mappings.is_empty() {
                    if !mappings.is_object() {
                        mappings = json!({});
                    }
                    merge_mapping(&mut mappings, &Value::Object(ds.mappings.clone()));
                }
            }
        }
        let dso = Self::simulated_dso(&t, best.1);
        let mut out = Self::simulated(&settings, mappings, aliases, lifecycle, overlapping, include_defaults, self);
        if let Some(d) = dso {
            out["template"]["data_stream_options"] = d;
        }
        Ok(out)
    }

    /// Reject a composable template naming component templates that do not
    /// exist (unless listed in `ignore_missing_component_templates`).
    pub fn validate_composed_of(&self, it: &Value, name: &str) -> Result<()> {
        Self::check_missing_components(&self.templates.read(), it, name)
    }

    /// Compose a composable template body (component templates + its own
    /// `template` block) into a single `{settings, mappings, aliases}` value.
    pub fn template_for_composition(&self, it: &Value) -> Value {
        let t = self.templates.read();
        let mut settings = json!({});
        let mut mappings = json!({});
        let mut aliases = json!({});
        Self::compose_into(&t, it, &mut settings, &mut mappings, &mut aliases);
        json!({"settings": settings, "mappings": mappings, "aliases": aliases})
    }

    fn check_missing_components(t: &Templates, it: &Value, n: &str) -> Result<()> {
        let ignore_all = it["ignore_missing_component_templates"].as_bool().unwrap_or(false);
        let ignored: Vec<&str> = it["ignore_missing_component_templates"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
        let missing: Vec<String> = it["composed_of"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .filter(|c| !t.component.contains_key(*c) && !ignore_all && !ignored.contains(c))
            .map(str::to_string)
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        Err(EsError::new(
            400,
            "invalid_index_template_exception",
            format!(
                "index_template [{n}] invalid, cause [index template [{n}] specifies a missing component templates [{}] that does not exist and is not part of 'ignore_missing_component_templates']",
                missing.join(", ")
            ),
        ))
    }

    fn simulated(settings: &Value, mappings: Value, aliases: Value, lifecycle: Option<Value>, overlapping: Vec<Value>, include_defaults: bool, node: &Node) -> Value {
        let flat = settings::normalize_index_settings(settings);
        let settings = if flat.is_empty() { json!({}) } else { settings::unflatten(&flat) };
        let mut template = json!({"settings": settings, "mappings": mappings, "aliases": aliases});
        if let Some(lc) = lifecycle {
            template["lifecycle"] = crate::ilm::dlm::render_lifecycle(node, "", &lc, include_defaults);
        }
        json!({"template": template, "overlapping": overlapping})
    }

    /// Create an index (`PUT /{index}` body: settings, mappings, aliases).
    pub fn create_index(&self, name: &str, body: &Value) -> Result<Arc<IndexService>> {
        if !name.starts_with(".ds-") {
            if let Some((tname, _)) = self.data_stream_template(name) {
                return Err(EsError::illegal_argument(format!(
                    "cannot create index with name [{name}], because it matches with template [{tname}] that creates data streams only, use create data stream api instead"
                )));
            }
        }
        self.create_index_impl(name, body, None, true)
    }

    pub(crate) fn create_index_impl(&self, name: &str, body: &Value, uuid: Option<&str>, initialize_lifecycle: bool) -> Result<Arc<IndexService>> {
        let name = resolve_date_math(name)?;
        if let Err(why) = is_valid_name(&name) {
            return Err(EsError::invalid_index_name(&name, &why));
        }
        // Names are shared with ES|QL views.
        if crate::esql::views::exists(self, &name) {
            return Err(EsError::invalid_index_name(&name, "already exists as an ESQL view"));
        }
        let _g = self.create_lock.lock();
        if let Some(existing) = self.get_index(&name) {
            return Err(EsError::index_already_exists(&name, &existing.uuid));
        }
        for svc in self.all_indices() {
            if svc.aliases().contains_key(&name) {
                return Err(EsError::new(400, "invalid_index_name_exception", format!("Invalid index name [{name}], already exists as alias")).with("index_uuid", "_na_").with("index", name.clone()));
            }
        }
        if let Some(o) = body.as_object() {
            for k in o.keys() {
                if !matches!(k.as_str(), "settings" | "mappings" | "aliases") {
                    return Err(EsError::new(400, "parse_exception", format!("unknown key [{k}] for create index")));
                }
            }
        }
        for (k, what) in [("settings", "settings"), ("mappings", "mappings"), ("aliases", "aliases")] {
            if body.get(k).is_some_and(|v| !v.is_object() && !v.is_null()) {
                return Err(EsError::new(400, "x_content_parse_exception", format!("[1:1] [create_index_request] failed to parse field [{what}]")));
            }
        }
        if let Some(a) = body.get("aliases").and_then(Value::as_object) {
            for (name, def) in a {
                validate_request_alias(name, def)?;
            }
        }
        if body["mappings"].get("properties").is_some_and(|v| !v.is_object() && !v.is_null()) {
            return Err(EsError::new(400, "mapper_parsing_exception", "Failed to parse mapping: [properties] must be an object"));
        }
        if let Some(a) = body.get("aliases").and_then(Value::as_object) {
            for (alias, def) in a {
                validate_alias_definition(alias, def)?;
            }
        }
        let (mut settings, mut mappings, mut aliases) = self.template_for(&name);
        deep_merge(&mut settings, &normalize_settings_json(&body["settings"]));
        let req_mappings = unwrap_typed(&body["mappings"]);
        deep_merge(&mut mappings, &req_mappings);
        deep_merge(&mut aliases, &body["aliases"]);
        let mut flat = settings::normalize_index_settings(&settings);
        crate::synthetic::license_fallback(self, &name, &mut flat, &mappings);
        // `cluster.columnar.enabled: false` (logsdb plugin) blocks new
        // indices in a strict columnar index mode.
        if let Some(m) = flat.get("index.mode").and_then(Value::as_str).map(str::to_ascii_lowercase).filter(|m| matches!(m.as_str(), "columnar" | "logsdb_columnar")) {
            let enabled = crate::ilm::cluster_setting(self, "cluster.columnar.enabled").or_else(|| self.node_settings.read().get("cluster.columnar.enabled").map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())));
            if enabled.as_deref() == Some("false") {
                return Err(EsError::illegal_argument(format!("creation of indices with a columnar index mode [{m}] is disabled; set [cluster.columnar.enabled] to [true] to allow it")));
            }
        }
        // `IndexSettings.SYNTHETIC_ID` validation.
        if flat.get("index.mapping.synthetic_id").is_some_and(|v| v.as_bool() == Some(true) || v.as_str() == Some("true")) {
            let mode = flat.get("index.mode").and_then(Value::as_str).unwrap_or("standard").to_ascii_uppercase();
            if mode != "TIME_SERIES" {
                return Err(EsError::illegal_argument(format!("The setting [index.mapping.synthetic_id] is only permitted when [index.mode] is set to [TIME_SERIES]. Current mode: [{mode}].")));
            }
            let codec = flat.get("index.codec").and_then(Value::as_str).unwrap_or("default");
            if !matches!(codec, "default" | "best_compression") {
                return Err(EsError::illegal_argument(format!("The setting [index.mapping.synthetic_id] is only permitted when [index.codec] is set to [default] or [best_compression]. Current mode: [{codec}].")));
            }
        }
        if flat.get("index.mode").and_then(Value::as_str) == Some("vectordb_document") {
            // `IndexMode.IndexModeSettingsProvider`.
            let key = "index.mapping.exclude_source_vectors";
            if flat.get(key).is_some_and(|v| v.as_bool() == Some(false) || v.as_str() == Some("false")) {
                return Err(EsError::illegal_argument(format!("[{key}] cannot be set to [false] when [index.mode=vectordb_document]")));
            }
            flat.insert(key.into(), json!("true"));
            flat.entry("index.store.preload".into()).or_insert(json!(["vex", "veq", "veb", "cenivf"]));
            flat.entry("index.merge.intra_merge_parallelism_enabled".into()).or_insert(json!("true"));
            flat.entry("index.merge.scheduler.auto_throttle".into()).or_insert(json!("false"));
        }
        if flat.get("index.mode").and_then(Value::as_str) == Some("lookup") {
            if let Some(n) = flat.get("index.number_of_shards") {
                let n = n.as_str().map(str::to_string).unwrap_or_else(|| n.to_string());
                if n != "1" {
                    return Err(EsError::illegal_argument("index with [lookup] mode must have [index.number_of_shards] set to 1 or unset"));
                }
            }
        }
        // ES `IndexModule`: with `node.store.allow_mmap: false` the mapping
        // store types are refused.
        if let Some(store_type) = flat.get("index.store.type").and_then(Value::as_str) {
            if !rustic_core::codec::directory::mmap_allowed() && matches!(store_type, "mmapfs" | "hybridfs") {
                return Err(EsError::illegal_argument(format!("store type [{store_type}] is not allowed because mmap is disabled")));
            }
        }
        let mut alias_map = BTreeMap::new();
        if let Some(a) = aliases.as_object() {
            for (k, v) in a {
                alias_map.insert(resolve_date_math(k).unwrap_or_else(|_| k.clone()), clean_alias(v));
            }
        }
        // A new index's similarities are built as ES builds them (strict
        // ranges); an existing one opens whatever an older build accepted.
        crate::similarity::check_ranges(&flat)?;
        let dir = self.data.join("indices").join(new_uuid());
        let svc = IndexService::create_with_uuid(&dir, &name, uuid, flat, &mappings, alias_map).map_err(|e| {
            let _ = std::fs::remove_dir_all(&dir);
            e
        })?;
        let svc = Arc::new(svc);
        svc.set_mapping_update_hook(self.mapping_hook.read().clone());
        if let Err(error) = if initialize_lifecycle { crate::ilm::runner::initialize_new_index(self, &svc) } else { Ok(()) } {
            self.ilm.exec.lock().remove(&svc.uuid);
            self.ilm.exec_gen.lock().remove(&svc.uuid);
            drop(svc);
            let _ = std::fs::remove_dir_all(&dir);
            return Err(error);
        }
        self.indices.write().insert(name, svc.clone());
        Ok(svc)
    }

    /// Register an index whose directory was prepared outside
    /// `create_index` (resize: clone / shrink / split).
    pub fn register_index(&self, svc: Arc<IndexService>) -> Result<()> {
        let _g = self.create_lock.lock();
        if let Some(existing) = self.get_index(&svc.name) {
            return Err(EsError::index_already_exists(&svc.name, &existing.uuid));
        }
        svc.set_mapping_update_hook(self.mapping_hook.read().clone());
        self.indices.write().insert(svc.name.clone(), svc);
        Ok(())
    }

    pub fn delete_index(&self, name: &str) -> Result<()> {
        // A data stream backing index: the write index cannot be deleted,
        // other backing indices leave the stream (generation + 1).
        // Match resolve's registry -> data-stream lock order. The callback
        // only removes from the already locked registry; all blocking cleanup
        // runs after both metadata guards have been released.
        let mut indices = self.indices.write();
        let (svc, changed) = self.detach_backing_index_with(name, || {
            indices.remove(name).ok_or_else(|| EsError::index_not_found(name))
        })?;
        drop(indices);
        self.finish_local_index_deletion(name, svc);
        if changed { self.persist_state()?; }
        Ok(())
    }

    /// Whether a search context (scroll, PIT, fold, query_then_fetch) still
    /// reads copy `shard` of index `name`: ES deletes a shard's data only
    /// once no reader pins it (the shard lock is released).
    pub fn copy_has_contexts(&self, name: &str, shard: u32) -> bool {
        let now = Instant::now();
        let reads = |targets: &[Target]| targets.iter().any(|t| t.svc.name == name && t.shard.as_ref().is_none_or(|s| s.id == shard));
        self.scrolls.lock().values().any(|s| s.expires > now && reads(&s.targets))
            || [&self.pits, &self.fold_contexts].iter().any(|m| m.lock().values().any(|s| s.expires > now && reads(&s.targets)))
            || self.fetch_contexts.lock().values().any(|c| c.expires > now && reads(&c.targets))
    }

    /// A context that also reads other shards keeps them: only the freed
    /// copy's targets leave it, and its pages report that shard failed
    /// (`lost`); a context left with nothing to read is freed.
    pub fn free_shard_contexts(&self, name: &str, shard: u32) -> usize {
        let reads = |t: &crate::search::Target| t.svc.name == name && t.shard.as_ref().is_none_or(|s| s.id == shard);
        let mut freed = 0;
        {
            let mut m = self.scrolls.lock();
            m.retain(|_, s| {
                if !s.targets.iter().any(reads) {
                    return true;
                }
                let keep: Vec<bool> = s.targets.iter().map(|t| !reads(t)).collect();
                freed += 1;
                if !keep.iter().any(|k| *k) {
                    return false;
                }
                let aligned = s.cursors.len() == s.targets.len();
                let mut i = 0;
                s.targets.retain(|_| { i += 1; keep[i - 1] });
                if aligned {
                    let mut i = 0;
                    s.cursors.retain(|_| { i += 1; keep[i - 1] });
                }
                s.lost.push((name.to_string(), shard));
                true
            });
        }
        for contexts in [&self.pits, &self.fold_contexts] {
            let mut m = contexts.lock();
            m.retain(|_, s| {
                if !s.targets.iter().any(reads) {
                    return true;
                }
                freed += 1;
                s.targets.retain(|t| !reads(t));
                if s.targets.is_empty() {
                    return false;
                }
                s.lost.push((name.to_string(), shard));
                true
            });
        }
        freed
    }

    /// Remove local shard files for recovery without altering logical data
    /// stream membership or generation. Cluster metadata owns that membership.
    pub fn delete_local_index_copy(&self, name: &str) -> Result<()> {
        let svc = self.indices.write().remove(name).ok_or_else(|| EsError::index_not_found(name))?;
        self.finish_local_index_deletion(name, svc);
        Ok(())
    }

    fn finish_local_index_deletion(&self, name: &str, svc: Arc<IndexService>) {
        crate::index::forget_recovery_attempts(&svc.uuid);
        self.scrolls.lock().retain(|_, s| !s.targets.iter().any(|t| t.svc.name == name));
        self.pits.lock().retain(|_, s| !s.targets.iter().any(|t| t.svc.name == name));
        self.fold_contexts.lock().retain(|_, s| !s.targets.iter().any(|t| t.svc.name == name));
        self.drop_fetch_contexts(name);
        let path = svc.path().to_path_buf();
        // Fence the copies (in-flight operations finish) before the files go;
        // a metadata operation still holding this service retries.
        svc.detach();
        for sh in svc.local_shards() {
            sh.fence();
        }
        drop(svc);
        let _ = std::fs::remove_dir_all(path);
    }

    /// Apply `_aliases` actions atomically.
    pub fn update_aliases(&self, actions: &[Value]) -> Result<Vec<Value>> {
        let mut results: Vec<Value> = Vec::new();
        // Indices a later `remove_index` action deletes may be replaced by an alias of the same name.
        let mut pending_removals: Vec<String> = Vec::new();
        for a in actions {
            if let Some(b) = a.get("remove_index") {
                let mut names: Vec<String> = b.get("index").and_then(Value::as_str).map(|s| vec![s.to_string()]).unwrap_or_default();
                if let Some(arr) = b.get("indices").and_then(Value::as_array) {
                    names.extend(arr.iter().filter_map(|x| x.as_str().map(str::to_string)));
                }
                for n in names {
                    if let Ok(ts) = self.resolve(&n, ResolveOpts { expand_closed: true, ..Default::default() }) {
                        pending_removals.extend(ts.into_iter().map(|t| t.svc.name.clone()));
                    }
                }
            }
        }
        let mut changes: BTreeMap<String, BTreeMap<String, Value>> = BTreeMap::new();
        let mut removals: Vec<String> = Vec::new();
        for a in actions {
            let o = a.as_object().ok_or_else(|| EsError::parsing("[aliases] actions must be objects"))?;
            let (op, body) = o.iter().next().ok_or_else(|| EsError::parsing("empty alias action"))?;
            let names = |k: &str, ks: &str| -> Vec<String> {
                let mut v = Vec::new();
                if let Some(s) = body.get(k).and_then(Value::as_str) {
                    v.push(s.to_string());
                }
                match body.get(ks) {
                    Some(Value::Array(a)) => v.extend(a.iter().filter_map(|x| x.as_str().map(str::to_string))),
                    Some(Value::String(s)) => v.push(s.clone()),
                    _ => {}
                }
                if k == "alias" {
                    v = v.into_iter().map(|a| resolve_date_math(&a).unwrap_or(a)).collect();
                }
                v
            };
            if op != "remove" && body.get("must_exist").is_some_and(|v| !v.is_null()) {
                return Err(EsError::new(400, "illegal_argument_exception", "[must_exist] is only valid for remove actions"));
            }
            let idx_names = names("index", "indices");
            if idx_names.is_empty() {
                return Err(EsError::action_validation(&["One of [index] or [indices] is required"]));
            }
            let mut targets = Vec::new();
            for n in &idx_names {
                for t in self.resolve(n, ResolveOpts { expand_closed: true, ..Default::default() })? {
                    targets.push(t.svc.name.clone());
                }
            }
            // Backing indices are reported under their data stream name.
            let mut shown: Vec<String> = Vec::new();
            for t in &targets {
                let n = self.data_stream_of_index(t).unwrap_or_else(|| t.clone());
                if !shown.contains(&n) {
                    shown.push(n);
                }
            }
            match op.as_str() {
                "add" => {
                    let aliases = names("alias", "aliases");
                    if aliases.is_empty() {
                        return Err(EsError::action_validation(&["One of [alias] or [aliases] is required"]));
                    }
                    let mut def = Map::new();
                    for k in ["filter", "is_write_index", "is_hidden"] {
                        if let Some(v) = body.get(k) {
                            def.insert(k.into(), v.clone());
                        }
                    }
                    // `routing` sets both routings unless given explicitly; values are strings.
                    let rs = |v: &Value| match v {
                        Value::String(s) => Value::String(s.clone()),
                        other => Value::String(other.to_string()),
                    };
                    for k in ["index_routing", "search_routing"] {
                        if let Some(v) = body.get(k).or(body.get("routing")).filter(|v| !v.is_null()) {
                            if targets.iter().any(|t| self.get_index(t).is_some_and(|s| s.is_time_series())) {
                                return Err(crate::tsdb::routing_forbidden());
                            }
                            def.insert(k.into(), rs(v));
                        }
                    }
                    results.push(json!({"action": {"type": "add", "indices": shown, "aliases": aliases}, "status": 200, "error": null}));
                    for t in &targets {
                        for al in &aliases {
                            if self.get_index(al).is_some() && !removals.contains(al) && !pending_removals.contains(al) {
                                return Err(EsError::new(400, "invalid_alias_name_exception", format!("Invalid alias name [{al}]: an index or data stream exists with the same name as the alias")));
                            }
                            if crate::esql::views::exists(self, al) {
                                return Err(EsError::new(400, "invalid_alias_name_exception", format!("Invalid alias name [{al}]. an index, data stream, or ESQL view exists with the same name as the alias")));
                            }
                            changes.entry(t.clone()).or_insert_with(|| self.get_index(t).map(|s| s.aliases()).unwrap_or_default()).insert(al.clone(), Value::Object(def.clone()));
                        }
                    }
                }
                "remove" => {
                    let aliases = names("alias", "aliases");
                    let must_exist = body.get("must_exist").and_then(Value::as_bool).unwrap_or(false);
                    let mut removed_any = false;
                    for t in &targets {
                        let cur = changes.entry(t.clone()).or_insert_with(|| self.get_index(t).map(|s| s.aliases()).unwrap_or_default());
                        let mut removed = false;
                        for al in &aliases {
                            let keys: Vec<String> = cur.keys().filter(|k| crate::mapping::wildcard_match(al, k)).cloned().collect();
                            for k in keys {
                                cur.remove(&k);
                                removed = true;
                            }
                        }
                        if !removed && must_exist {
                            return Err(EsError::alias_missing(&aliases.join(",")));
                        }
                        removed_any |= removed;
                    }
                    let action = json!({"type": "remove", "indices": shown, "aliases": aliases});
                    if removed_any {
                        results.push(json!({"action": action, "status": 200, "error": null}));
                    } else {
                        let msg = if aliases.len() == 1 { format!("alias [{}] missing", aliases[0]) } else { format!("aliases [{}] missing", aliases.join(",")) };
                        results.push(json!({"action": action, "status": 404, "error": {"type": "aliases_not_found_exception", "reason": msg}}));
                    }
                }
                "remove_index" => {
                    results.push(json!({"action": {"type": "remove_index", "indices": idx_names}, "status": 200, "error": null}));
                    removals.extend(targets)
                }
                other => return Err(EsError::parsing(format!("Unknown alias action [{other}]"))),
            }
        }
        for (idx, a) in changes {
            if let Some(svc) = self.get_index(&idx) {
                svc.set_aliases(a)?;
            }
        }
        for r in removals {
            self.delete_index(&r)?;
        }
        Ok(results)
    }

    /// Register a query_then_fetch reader context over `targets` (pinned
    /// views) for `owner`, sweeping expired ones once the table grows.
    ///
    /// The node's budget is bounded ([`MAX_FETCH_CONTEXTS`] contexts,
    /// [`MAX_FETCH_CONTEXT_BYTES`] estimated, over all its engines): when
    /// it is full, even after dropping this engine's expired contexts,
    /// nothing is registered and `false` comes back. Query callers admit
    /// this context before executing so refusal cannot require scoring the
    /// same request again against a different snapshot.
    ///
    /// The estimate covers the context's own handles, not the segments and
    /// memtables its views keep alive: those are shared by every view of a
    /// publication, so a per-context share would count them many times.
    pub fn register_fetch_context(&self, id: String, targets: Vec<Target>, owner: FetchOwner) -> bool {
        let Some(charge) = self.reserve_fetch_context(&targets) else { return false; };
        self.register_reserved_fetch_context(id, targets, owner, charge)
    }

    /// Reserve capacity before query execution without publishing a context
    /// that could expire or be fetched while its query is still running.
    pub fn reserve_fetch_context(&self, targets: &[Target]) -> Option<FetchCharge> {
        let bytes = 256 + targets.iter().map(|t| 128 + 64 * t.snapshot.as_ref().map_or(0, |v| v.num_leaves() as u64)).sum::<u64>();
        let budget = self.fetch_budget.lock().clone();
        budget.charge(bytes).or_else(|| {
            drop_off_thread(self.reap_fetch_contexts(Instant::now()));
            budget.charge(bytes)
        })
    }

    /// Publish an admitted reader after its query completes. Cancellation
    /// and duplicate IDs still refuse publication under the same lock.
    pub fn register_reserved_fetch_context(&self, id: String, targets: Vec<Target>, owner: FetchOwner, charge: FetchCharge) -> bool {
        // A known cancellation releases admission synchronously, without
        // constructing a retained reader or waiting for the reaper.
        if self.fetch_freed.lock().remove(&id).is_some() { return false; }
        let now = Instant::now();
        let bytes = charge.bytes;
        let accounting = crate::memory::process_breakers().reserve_without_breaking(crate::memory::Kind::Accounting, bytes);
        let ctx = FetchCtx { targets, owner, expires: now + self.fetch_keep_alive(), bytes, charge, accounting };
        // Under `fetch_freed` (taken before `fetch_contexts`, as
        // `free_unregistered_fetch_context` does): a free is either seen
        // here or sees this context.
        let mut freed = self.fetch_freed.lock();
        if freed.remove(&id).is_some() {
            drop(freed);
            drop_off_thread(vec![ctx]);
            return false;
        }
        let mut ctxs = self.fetch_contexts.lock();
        // An id already in use (a duplicate a client slipped past an edge
        // of an earlier build) never replaces its context.
        if ctxs.contains_key(&id) {
            drop((ctxs, freed));
            drop_off_thread(vec![ctx]);
            return false;
        }
        ctxs.insert(id, ctx);
        true
    }

    /// `search.default_keep_alive` (transient over persistent), else
    /// [`FETCH_CTX_KEEP_ALIVE`].
    pub fn fetch_keep_alive(&self) -> Duration {
        let cs = self.cluster_settings.read();
        cs.1.get("search.default_keep_alive")
            .or(cs.0.get("search.default_keep_alive"))
            .and_then(|v| v.as_str().map(str::to_string).or_else(|| v.as_i64().map(|n| n.to_string())))
            .and_then(|s| crate::date::parse_time_value(&s))
            .and_then(|ms| u64::try_from(ms).ok())
            .map_or(FETCH_CTX_KEEP_ALIVE, Duration::from_millis)
    }

    /// Drop the query_then_fetch reader contexts expired at `now`
    /// (`Node::maintenance`; the cluster's engine maintenance on every
    /// shard engine).
    ///
    /// Contexts over an index closed since are dropped too, whoever closed it
    /// (`_close`, a cluster metadata change, ILM): their views must not
    /// outlive the open index.
    ///
    /// The reaped contexts come back, for the caller to drop (with the
    /// views they pin) outside the table's lock.
    pub fn reap_fetch_contexts(&self, now: Instant) -> Vec<FetchCtx> {
        self.fetch_freed.lock().retain(|_, expires| *expires > now);
        take_dead(&mut self.fetch_contexts.lock(), |c| c.expires > now && c.targets.iter().all(|t| t.svc.is_open()) && c.copies_current())
    }

    /// Drop query_then_fetch reader context `id`, whoever opened it: the
    /// cluster front end's own cleanup of a context whose answer it refused
    /// (internal; requests go through [`Self::take_fetch_context`]).
    pub fn drop_fetch_context(&self, id: &str) {
        let dead = self.fetch_contexts.lock().remove(id);
        drop_off_thread(dead.into_iter().collect());
    }

    /// Drop the query_then_fetch reader contexts over copy `shard` of index
    /// `name` at once (the cluster removes, resets or promotes it): its
    /// pinned views must not keep wiped files until the keep-alive.
    pub fn drop_fetch_contexts_of_copy(&self, name: &str, shard: u32) {
        let dead = take_dead(&mut self.fetch_contexts.lock(), |c| !c.targets.iter().any(|t| t.svc.name == name && t.shard.as_ref().is_some_and(|s| s.id == shard)));
        drop_off_thread(dead);
    }

    /// Drop the query_then_fetch reader contexts over index `name` (it is
    /// deleted, reopened or closed: its pinned views must not outlive it).
    pub fn drop_fetch_contexts(&self, name: &str) {
        let dead = take_dead(&mut self.fetch_contexts.lock(), |c| !c.targets.iter().any(|t| t.svc.name == name));
        drop(dead);
    }

    /// Take (free) the query_then_fetch reader context `id` if `owner`
    /// opened it and it has not expired; a request from anyone else, or for
    /// another index or shard, neither reads nor frees it.
    pub fn take_fetch_context(&self, id: &str, owner: &FetchOwner) -> Option<FetchCtx> {
        let mut ctxs = self.fetch_contexts.lock();
        let c = ctxs.get(id)?;
        if c.owner != *owner || c.expires <= Instant::now() {
            return None;
        }
        let ctx = ctxs.remove(id)?;
        drop(ctxs);
        // A closed index, or a copy gone or replaced since, is not read (its
        // context is freed all the same, outside the table's lock).
        (ctx.targets.iter().all(|t| t.svc.is_open()) && ctx.copies_current()).then_some(ctx)
    }

    /// A free of `id` (by `owner`) that found no context: should its query
    /// phase still arrive, it registers none (the oldest of
    /// [`MAX_FETCH_CONTEXTS`] such ids make room; each expires with the
    /// keep-alive). One registered since by the same owner is freed now;
    /// someone else's is left alone. Atomic with
    /// [`Self::register_fetch_context`]: both decide under `fetch_freed`.
    pub fn free_unregistered_fetch_context(&self, id: &str, owner: &FetchOwner) {
        let expires = Instant::now() + self.fetch_keep_alive();
        let mut freed = self.fetch_freed.lock();
        let late = {
            let mut ctxs = self.fetch_contexts.lock();
            match ctxs.get(id) {
                Some(c) if c.owner == *owner => ctxs.remove(id),
                Some(_) => return,
                None => None,
            }
        };
        if let Some(ctx) = late {
            drop(freed);
            drop_off_thread(vec![ctx]);
            return;
        }
        // Full: the oldest eighth makes room, one scan per 512 inserts.
        if freed.len() >= MAX_FETCH_CONTEXTS {
            let mut by_age: Vec<(Instant, String)> = freed.iter().map(|(k, e)| (*e, k.clone())).collect();
            by_age.sort_unstable();
            for (_, k) in by_age.into_iter().take(MAX_FETCH_CONTEXTS / 8) {
                freed.remove(&k);
            }
        }
        freed.insert(id.to_string(), expires);
    }

    /// Estimated bytes held by this engine's open query_then_fetch reader
    /// contexts (for memory accounting).
    pub fn fetch_contexts_bytes(&self) -> u64 {
        self.fetch_contexts.lock().values().map(|c| c.bytes).sum()
    }

    /// Create a point-in-time snapshot for the targets: one target per shard
    /// copy searched, each retaining its copy's current reader view
    /// (segments and published memtables, vector store captured). Nothing is
    /// sealed and no shard lock is taken (see [`rustic_core::ViewSlot::retain`]).
    pub fn snapshot(&self, targets: &[Target]) -> Result<Vec<Target>> {
        let mut out = Vec::new();
        for t in crate::search::expand_targets(targets)? {
            let sh = t.shard_copy().clone();
            sh.sync_schema_for_search()?;
            sh.shard.before_search();
            let view = sh.shard.retain_view()?;
            out.push(Target { snapshot: Some(Arc::new(view)), ..t });
        }
        Ok(out)
    }

    /// Cluster setting `search.allow_expensive_queries` (default true).
    pub fn allow_expensive_queries(&self) -> bool {
        let cs = self.cluster_settings.read();
        match cs.1.get("search.allow_expensive_queries").or(cs.0.get("search.allow_expensive_queries")) {
            Some(v) => v.as_bool().unwrap_or_else(|| v.as_str() != Some("false")),
            None => true,
        }
    }

    pub fn max_buckets(&self) -> usize {
        let cs = self.cluster_settings.read();
        cs.1.get("search.max_buckets").or(cs.0.get("search.max_buckets")).and_then(|v| v.as_str().and_then(|s| s.parse().ok()).or(v.as_u64().map(|x| x as usize))).unwrap_or(65_536)
    }

    /// `search.default_search_timeout` (dynamic; transient over persistent
    /// over the node's own setting; ES default `-1`: none): the timeout of a
    /// search whose request names none. One read lock per search, as
    /// [`Self::max_buckets`]; the common unset case parses nothing.
    pub fn default_search_timeout(&self) -> Option<Duration> {
        const KEY: &str = "search.default_search_timeout";
        let parse = |v: &Value| v.as_str().and_then(crate::date::parse_time_value).filter(|ms| *ms > 0).map(|ms| Duration::from_millis(ms as u64));
        {
            let cs = self.cluster_settings.read();
            if let Some(v) = cs.1.get(KEY).or(cs.0.get(KEY)) {
                return parse(v);
            }
        }
        self.node_settings.read().get(KEY).and_then(parse)
    }

    /// `indices.memory.shard_inactive_time` (ES default 5m).
    pub fn shard_inactive_time(&self) -> Duration {
        self.node_settings
            .read()
            .get("indices.memory.shard_inactive_time")
            .and_then(Value::as_str)
            .and_then(crate::date::parse_time_value)
            .map_or(Duration::from_secs(300), |ms| Duration::from_millis(ms.max(0) as u64))
    }

    /// One maintenance tick: expire contexts, flush big translogs and idle
    /// shards, optimize vector segments.
    /// Expire stale durable retention leases on every local copy, per
    /// index `index.soft_deletes.retention_lease.period` (default 12h).
    /// Run periodically (engine [`Node::maintenance`], cluster maintenance).
    pub fn expire_retention_leases(&self) {
        for svc in self.all_indices() {
            let period = svc.setting("index.soft_deletes.retention_lease.period")
                .and_then(|v| crate::date::parse_time_value(&v))
                .filter(|ms| *ms > 0)
                .unwrap_or(crate::hooks::retention::DEFAULT_LEASE_PERIOD_MILLIS);
            for sh in svc.local_shards() {
                match sh.shard.expire_retention_leases(period) {
                    Ok(ids) if !ids.is_empty() => tracing::info!("expired retention leases {ids:?} on [{}][{}]", svc.name, sh.id),
                    Ok(_) => {}
                    Err(e) => tracing::warn!("expiring retention leases on [{}][{}] failed: {e}", svc.name, sh.id),
                }
            }
        }
    }

    pub fn maintenance(&self) {
        /// ES `index.flush_after_merge` default.
        const FLUSH_AFTER_MERGE_BYTES: u64 = 512 * 1024 * 1024;
        let now = Instant::now();
        {
            // About once a minute (ES syncs leases every 30s).
            static LAST: Mutex<Option<Instant>> = Mutex::new(None);
            let mut last = LAST.lock();
            if last.is_none_or(|t| t.elapsed() >= std::time::Duration::from_secs(60)) {
                *last = Some(now);
                drop(last);
                self.expire_retention_leases();
            }
        }
        drop(crate::search_ext::take_expired_sql_cursors(self, now));
        self.scrolls.lock().retain(|_, s| s.expires > now);
        self.pits.lock().retain(|_, s| s.expires > now);
        self.fold_contexts.lock().retain(|_, s| s.expires > now);
        self.reap_fetch_contexts(now);
        self.reap_async_searches();
        let inactive = self.shard_inactive_time();
        let prev = std::mem::take(&mut *self.write_activity.lock());
        let mut seen = HashMap::new();
        for svc in self.all_indices() {
            if !svc.is_open() {
                continue;
            }
            // ES `index.translog.flush_threshold_age` (default 1m): a copy
            // still being written to is committed once its last flush is
            // that old (`InternalEngine#shouldPeriodicallyFlush`), so its
            // translog stays short and merged-away segment files leave the
            // disk within about a minute instead of waiting for idleness.
            let flush_age = svc
                .setting("index.translog.flush_threshold_age")
                .and_then(|v| crate::date::parse_time_value(&v))
                .filter(|ms| *ms > 0)
                .map_or(Duration::from_secs(60), |ms| Duration::from_millis(ms as u64));
            for sh in svc.local_shards() {
                // A copy whose engine failed (tragic event) is recovered from
                // its store, as ES reallocates and store-recovers a failed
                // shard; until then the index is red.
                if sh.shard.tragic_error().is_some() {
                    if let Err(e) = svc.recover_failed_shard(&sh) {
                        tracing::error!("recovery of failed shard [{}][{}] from its store failed: {e}", svc.name, sh.id);
                    }
                    continue;
                }
                // Uncommitted bytes: generations kept for history retention
                // would otherwise trigger a flush every tick.
                let big = sh.shard.uncommitted_translog_size() > 512 * 1024 * 1024
                    // ES `index.flush_after_merge` (512mb): a big merge
                    // makes the next periodic check flush, freeing its inputs.
                    || sh.shard.merged_away_committed_bytes() >= FLUSH_AFTER_MERGE_BYTES;
                sh.shard.sweep_retired();
                // ES `flushOnIdle`: commit a copy with uncommitted ops or
                // vector swaps once no write came for the inactive time, so
                // a restart neither replays its tail nor redoes the seals.
                let key = Arc::as_ptr(&sh) as usize;
                let max = sh.shard.max_seq_no();
                let since = match prev.get(&key) {
                    Some(&(m, t)) if m == max => t,
                    _ => now,
                };
                seen.insert(key, (max, since));
                let pending = max > sh.shard.committed_seq_no() || sh.shard.has_uncommitted_vector_changes() || sh.shard.has_uncommitted_merges();
                let aged = pending && sh.shard.elapsed_since_flush() >= flush_age;
                if big || aged || (pending && now.duration_since(since) >= inactive) {
                    if let Err(e) = sh.shard.flush() {
                        tracing::error!("flush of [{}][{}] failed: {e}", svc.name, sh.id);
                    }
                }
            }
        }
        *self.write_activity.lock() = seen;
        self.optimize_vectors_all();
    }

    /// Run the vector optimizers (seal appendable chunks into HNSW-indexed
    /// segments, vacuum, merge) on every open local shard. Blocking: a seal
    /// builds graphs. Without it kNN brute-forces every vector, so a host
    /// that does not run [`Node::maintenance`] (cluster mode) must call this.
    pub fn optimize_vectors_all(&self) {
        for svc in self.all_indices() {
            if !svc.is_open() {
                continue;
            }
            for sh in svc.local_shards() {
                // Published memtables of a write-idle copy are sealed first
                // (Lucene writes a segment, graph included, at refresh).
                sh.shard.seal_published_if_idle(Duration::from_secs(5));
                // Graph builds run without the core lock, so writes to the
                // shard are not held up by them.
                let job = sh.shard.core.read().vector_optimizer();
                if let Err(e) = job.run() {
                    tracing::warn!("vector optimize of [{}][{}] failed: {e}", svc.name, sh.id);
                }
            }
        }
    }

    /// Collect shards with legacy segments or mutable graph owners. Mutable
    /// owners still need reclamation after their last live vector is deleted.
    fn vector_background_jobs(engines: &[Arc<Self>]) -> Vec<(String, u32, rustic_core::index::VectorOptimizer)> {
        let mut jobs = Vec::new();
        for engine in engines {
            for svc in engine.all_indices() {
                if !svc.is_open() {
                    continue;
                }
                for sh in svc.local_shards() {
                    sh.shard.seal_published_if_idle(Duration::from_secs(5));
                    let job = sh.shard.core.read().vector_optimizer();
                    let backlog = job.backlog();
                    if backlog.mutable_fields > 0 || backlog.unsealed_segments + backlog.sealed_segments > 0 {
                        jobs.push((svc.name.clone(), sh.id, job));
                    }
                }
            }
        }
        jobs
    }

    /// Index every eligible shard before running its bounded maintenance pass.
    /// Legacy maintenance performs at most one rebuild per shard; mutable
    /// maintenance compacts storage without rebuilding graph topology. Shards
    /// remain sequential, with HNSW retaining its internal parallelism.
    pub fn optimize_vectors_background(engines: &[Arc<Self>]) {
        let jobs = Self::vector_background_jobs(engines);
        run_vector_background_phases(&jobs, |(index, shard, job)| {
            let before = job.backlog();
            let started = Instant::now();
            match job.index_pending() {
                Ok(report) => {
                    tracing::debug!(target: "vector_optimizer", %index, shard, phase = "indexing", elapsed_ms = started.elapsed().as_millis() as u64,
                        ?before, after = ?job.backlog(), ?report, "vector background pass");
                    true
                }
                Err(e) => {
                    tracing::warn!(%index, shard, "vector indexing pass failed: {e}");
                    false
                }
            }
        }, |(index, shard, job)| {
            let before = job.backlog();
            let started = Instant::now();
            match job.maintain_once() {
                Ok(report) => tracing::debug!(target: "vector_optimizer", %index, shard, phase = "maintenance", elapsed_ms = started.elapsed().as_millis() as u64,
                    ?before, after = ?job.backlog(), ?report, "vector background pass"),
                Err(e) => tracing::warn!(%index, shard, "vector maintenance pass failed: {e}"),
            }
        });
    }

    /// Background maintenance thread (1s tick; ES's refresh interval).
    pub fn start_background(self: &Arc<Self>) {
        let weak = Arc::downgrade(self);
        std::thread::Builder::new()
            .name("es-maintenance".into())
            .spawn(move || loop {
                std::thread::sleep(Duration::from_secs(1));
                match weak.upgrade() {
                    Some(n) => n.maintenance(),
                    None => break,
                }
            })
            .ok();
    }

    /// Flush every index (shutdown).
    pub fn flush_all(&self) {
        for svc in self.all_indices() {
            let _ = svc.flush();
        }
    }

    pub fn uptime_millis(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }
}

/// A failed indexing job does not prevent other shards from indexing. It
/// also does not spend more work on that shard's maintenance in this round.
fn run_vector_background_phases<T>(jobs: &[T], mut indexing: impl FnMut(&T) -> bool, mut maintenance: impl FnMut(&T)) {
    let indexed: Vec<bool> = jobs.iter().map(&mut indexing).collect();
    for (job, ready) in jobs.iter().zip(indexed) {
        if ready {
            maintenance(job);
        }
    }
}

/// Merge an index's search routing reached through one more expression
/// part: no routing (a direct reference, an alias without
/// `search_routing`) wins; alias routings union.
fn note_routing(routing: &mut HashMap<String, Option<String>>, n: &str, r: Option<String>) {
    let merged = match (routing.get(n), r) {
        (None, r) => r,
        (Some(None), _) | (_, None) => None,
        (Some(Some(a)), Some(b)) => {
            let mut v: Vec<String> = a.split(',').chain(b.split(',')).map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect();
            v.sort();
            v.dedup();
            Some(v.join(","))
        }
    };
    routing.insert(n.to_string(), merged);
}

fn hostname() -> String {
    std::env::var("HOSTNAME").ok().filter(|h| !h.is_empty()).unwrap_or_else(|| "rustic-node".into())
}

fn split_expr(expr: &str) -> Vec<String> {
    // Commas inside date-math `<...>` are part of the name.
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut depth = 0;
    for c in expr.chars() {
        match c {
            '<' => {
                depth += 1;
                cur.push(c)
            }
            '>' => {
                depth -= 1;
                cur.push(c)
            }
            ',' if depth == 0 => {
                if !cur.trim().is_empty() {
                    out.push(cur.trim().to_string());
                }
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

pub fn pattern_list(v: &Value) -> Vec<String> {
    match v {
        Value::String(s) => s.split(',').map(str::to_string).collect(),
        Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
        _ => Vec::new(),
    }
}

fn normalize_settings_json(v: &Value) -> Value {
    if v.is_null() {
        return json!({});
    }
    settings::unflatten(&settings::normalize_index_settings(v))
}

fn unwrap_typed(v: &Value) -> Value {
    if let Some(o) = v.as_object() {
        if o.len() == 1 {
            if let Some(inner) = o.get("_doc") {
                return inner.clone();
            }
        }
    }
    if v.is_null() { json!({}) } else { v.clone() }
}

/// `IndexNameExpressionResolver.checkForRemoteIndexExpression`: APIs that
/// do not allow remote indices reject `cluster:index` expressions.
pub fn reject_remote_indices(expr: &str) -> Result<()> {
    let is_remote = |e: &str| {
        let e = e.strip_prefix('-').unwrap_or(e);
        match e.find(':') {
            Some(i) => i > 0 && !e[i..].starts_with("::"),
            None => false,
        }
    };
    let parts: Vec<&str> = expr.split(',').collect();
    if parts.iter().any(|p| is_remote(p)) {
        let remote: Vec<&str> = parts.into_iter().filter(|p| is_remote(p)).collect();
        return Err(EsError::illegal_argument(format!("Cross-cluster calls are not supported in this context but remote indices were requested: [{}]", remote.join(", "))));
    }
    Ok(())
}

/// `Alias.fromXContent`: the fields and value types a create-index request
/// alias accepts.
fn validate_request_alias(name: &str, def: &Value) -> Result<()> {
    let Some(o) = def.as_object() else { return Ok(()) };
    for (k, v) in o {
        if !matches!(k.as_str(), "filter" | "routing" | "index_routing" | "search_routing" | "is_write_index" | "is_hidden") {
            return Err(EsError::illegal_argument(format!("Unknown field [{k}] in alias [{name}]")));
        }
        match v {
            Value::Object(_) => {}
            Value::String(s) => {
                if !matches!(k.as_str(), "routing" | "index_routing" | "search_routing") {
                    return Err(EsError::illegal_argument(format!("Unsupported String type value [{s}] for field [{k}] in alias [{name}]")));
                }
            }
            Value::Bool(b) => {
                if !matches!(k.as_str(), "is_write_index" | "is_hidden") {
                    return Err(EsError::illegal_argument(format!("Unsupported boolean type value [{b}] for field [{k}] in alias [{name}]")));
                }
            }
            other => return Err(EsError::illegal_argument(format!("Unknown token [{}] in alias [{name}]", crate::jsonpos::token_name(other)))),
        }
    }
    Ok(())
}

fn clean_alias(v: &Value) -> Value {
    let mut m = v.as_object().cloned().unwrap_or_default();
    if let Some(r) = m.remove("routing") {
        m.insert("index_routing".into(), r.clone());
        m.insert("search_routing".into(), r);
    }
    Value::Object(m)
}

pub type FlatSettingsMap = FlatSettings;

/// The node's persisted cluster state (`_state/cluster.json`): `{}` for a
/// new node. An existing file that cannot be read or parsed is an error:
/// starting from `{}` would persist an empty state over it and lose the
/// templates, data streams, persistent settings and cluster UUID (ES also
/// refuses to start on unreadable local state).
fn load_cluster_state(path: &Path) -> Result<Value> {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| {
            EsError::internal(format!("cannot parse cluster state [{}] ({} bytes): {e}; refusing to start rather than overwrite it", path.display(), bytes.len()))
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(json!({})),
        Err(e) => Err(EsError::internal(format!("cannot read cluster state [{}]: {e}; refusing to start rather than overwrite it", path.display()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unreadable_cluster_state_is_not_overwritten() {
        let tmp = tempfile::tempdir().unwrap();
        {
            let node = Node::open(tmp.path()).unwrap();
            node.templates.write().legacy.insert("keep".into(), json!({"index_patterns": ["k*"]}));
            node.persist_state().unwrap();
        }
        let state = tmp.path().join("_state/cluster.json");
        assert!(std::fs::read_to_string(&state).unwrap().contains("keep"));
        // A torn / truncated state file: the node refuses to start and
        // leaves the file as it was.
        std::fs::write(&state, b"{\"legacy_templates\": {\"keep\"").unwrap();
        let err = Node::open(tmp.path()).err().expect("unreadable cluster state must fail the open");
        assert!(err.reason.contains("refusing to start"), "{}", err.reason);
        assert_eq!(std::fs::read(&state).unwrap(), b"{\"legacy_templates\": {\"keep\"");
        // A missing file is a new node.
        std::fs::remove_file(&state).unwrap();
        Node::open(tmp.path()).unwrap();
        assert!(state.exists());
    }

    #[test]
    fn index_with_unopenable_shard_stays_registered_and_red() {
        let tmp = tempfile::tempdir().unwrap();
        let path = {
            let node = Node::open(tmp.path()).unwrap();
            let svc = node.create_index("broken", &json!({"settings": {"number_of_shards": 2}})).unwrap();
            svc.path().to_path_buf()
        };
        std::fs::write(path.join("shard-1").join("shard_state.json"), b"not json").unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let svc = node.get_index("broken").expect("index must not vanish when one shard fails to open");
        assert_eq!(svc.failed_shards().keys().copied().collect::<Vec<_>>(), vec![1]);
        assert_eq!(svc.active_shards(), 1);
        // Its files are untouched.
        assert_eq!(std::fs::read(path.join("shard-1").join("shard_state.json")).unwrap(), b"not json");
    }

    #[cfg(feature = "mutable-hnsw")]
    #[test]
    fn vector_background_schedules_mutable_only_shards() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("text-only", &json!({})).unwrap();
        let svc = node.create_index("mutable", &json!({
            "mappings": {"properties": {"v": {"type": "dense_vector", "dims": 4,
                "index_options": {"type": "hnsw"}}}}
        })).unwrap();
        let shard = svc.local_shards()[0].clone();
        assert!(shard.shard.core.read().vector_optimizer().backlog().sealed_segments == 0);
        assert!(shard.shard.core.read().vector_optimizer().backlog().unsealed_segments == 0);
        let jobs = Node::vector_background_jobs(&[node.clone()]);
        assert_eq!(jobs.len(), 1, "mutable-only owner must receive maintenance; text-only must not");
        assert_eq!(jobs[0].0, "mutable");
        // The real pass must remain safe when there are no legacy segments.
        Node::optimize_vectors_background(&[node.clone()]);
        svc.set_open(false).unwrap();
        assert!(Node::vector_background_jobs(&[node]).is_empty(), "closed indices are not scheduled");
    }

    #[test]
    fn vector_background_round_indexes_all_shards_before_bounded_maintenance() {
        let events = std::cell::RefCell::new(Vec::new());
        run_vector_background_phases(&[0, 1, 2], |shard| {
            events.borrow_mut().push((*shard, "index"));
            *shard != 1 // A failed shard must not starve a later shard.
        }, |shard| {
            events.borrow_mut().push((*shard, "maintain"));
        });
        assert_eq!(*events.borrow(), vec![(0, "index"), (1, "index"), (2, "index"), (0, "maintain"), (2, "maintain")]);
    }

    #[test]
    fn create_resolve_aliases_templates() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("logs-1", &json!({})).unwrap();
        node.create_index("logs-2", &json!({"aliases": {"logs": {}}})).unwrap();
        node.create_index("other", &json!({})).unwrap();
        assert_eq!(node.create_index("logs-1", &json!({})).unwrap_err().kind, "resource_already_exists_exception");
        assert_eq!(node.create_index("Bad", &json!({})).unwrap_err().kind, "invalid_index_name_exception");
        let names = |e: &str| node.resolve(e, ResolveOpts::default()).unwrap().iter().map(|t| t.svc.name.clone()).collect::<Vec<_>>();
        assert_eq!(names("logs-*"), vec!["logs-1", "logs-2"]);
        assert_eq!(names("*,-logs-1"), vec!["logs-2", "other"]);
        assert_eq!(names("logs"), vec!["logs-2"]);
        assert_eq!(names("_all").len(), 3);
        assert_eq!(node.resolve("nope", ResolveOpts::default()).unwrap_err().kind, "index_not_found_exception");
        assert!(node.resolve("nope*", ResolveOpts::default()).unwrap().is_empty());
        node.update_aliases(&[json!({"add": {"index": "logs-1", "alias": "logs", "is_write_index": true}})]).unwrap();
        assert_eq!(node.write_index("logs", true).unwrap().name, "logs-1");
        node.templates.write().index.insert("t".into(), json!({"index_patterns": ["tpl-*"], "priority": 1, "template": {"settings": {"number_of_replicas": 0}, "mappings": {"properties": {"x": {"type": "keyword"}}}}}));
        let svc = node.write_index("tpl-a", true).unwrap();
        assert_eq!(svc.setting("index.number_of_replicas").as_deref(), Some("0"));
        assert!(svc.lookup().get("x").is_some());
        drop(svc);
        drop(node);
        let node = Node::open(tmp.path()).unwrap();
        assert_eq!(node.all_indices().len(), 4);
        assert!(node.resolve("tpl-a", ResolveOpts::default()).is_ok());
    }

    #[test]
    fn date_math_names() {
        let n = resolve_date_math("<logs-{now/d}>").unwrap();
        assert!(n.starts_with("logs-20") && n.len() == "logs-2024.01.01".len(), "{n}");
        let n = resolve_date_math("%3Clogs-%7Bnow%2Fd%7ByyyyMM%7D%7D%3E").unwrap();
        assert_eq!(n.len(), "logs-202401".len(), "{n}");
        assert_eq!(resolve_date_math("plain").unwrap(), "plain");
    }
}

/// `Alias.fromXContent` (a create-index request's `aliases` entry): known
/// fields only, string routings, boolean flags.
pub fn validate_alias_definition(alias: &str, def: &Value) -> Result<()> {
    let Some(o) = def.as_object() else { return Ok(()) };
    for (k, v) in o {
        if !matches!(k.as_str(), "filter" | "routing" | "index_routing" | "search_routing" | "is_write_index" | "is_hidden") {
            return Err(EsError::illegal_argument(format!("Unknown field [{k}] in alias [{alias}]")));
        }
        match v {
            Value::Object(_) | Value::Null => {}
            Value::String(x) if !matches!(k.as_str(), "routing" | "index_routing" | "search_routing") => {
                return Err(EsError::illegal_argument(format!("Unsupported String type value [{x}] for field [{k}] in alias [{alias}]")));
            }
            Value::String(_) => {}
            Value::Bool(b) if !matches!(k.as_str(), "is_write_index" | "is_hidden") => {
                return Err(EsError::illegal_argument(format!("Unsupported boolean type value [{b}] for field [{k}] in alias [{alias}]")));
            }
            Value::Bool(_) => {}
            Value::Number(_) => return Err(EsError::illegal_argument(format!("Unknown token [VALUE_NUMBER] in alias [{alias}]"))),
            Value::Array(_) => return Err(EsError::illegal_argument(format!("Unknown token [START_ARRAY] in alias [{alias}]"))),
        }
    }
    Ok(())
}
