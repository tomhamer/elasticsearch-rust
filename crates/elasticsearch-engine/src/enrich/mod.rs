//! Enrich policies (`_enrich/policy`), policy execution into hidden
//! `.enrich-<policy>-<epochmillis>` indices (alias `.enrich-<policy>`), and
//! the lookups used by the `enrich` ingest processor and ES|QL `ENRICH`.
//!
//! Policies persist in `<data>/_state/enrich.json`. Lookups need the node:
//! ingest only carries the pipeline store, so the node APIs that run
//! pipelines enter a [`NodeScope`] (a thread-local `&Node` for the duration
//! of the call) which the processor reads.

pub mod processor;

use crate::error::{EsError, Result};
use crate::node::{Node, ResolveOpts};
use crate::security::owner::StoredAuthn;
use crate::security::{Authn, Permission};
use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

pub const SUPPORTED_TYPES: [&str; 3] = ["match", "geo_match", "range"];
pub const README: &str = "This index is managed by Elasticsearch and should not be modified in any way.";
pub const VERSION_DEPRECATION: &str = "the [elasticsearch_version] field of an enrich policy has no effect and will be removed in a future version of Elasticsearch";
const CACHE_MAX: usize = 1000;
/// Bytes the lookup cache may retain (keys and results, serialized size):
/// entries are evicted by bytes as well as by count (finding 163).
pub const CACHE_MAX_BYTES: u64 = 64 << 20;
/// A result larger than this is returned but never cached.
pub const CACHE_MAX_ENTRY_BYTES: u64 = CACHE_MAX_BYTES / 16;

/// One enrich policy.
#[derive(Clone, Debug)]
pub struct Policy {
    pub name: String,
    /// `match`, `range` or `geo_match`.
    pub kind: String,
    pub indices: Vec<String>,
    pub match_field: String,
    pub enrich_fields: Vec<String>,
    pub query: Option<Value>,
    /// Security: the subject that stored the policy (ES keeps the creating
    /// request's security headers); executions read the sources as it.
    /// Never part of the policy's API configuration.
    pub headers: Option<StoredAuthn>,
}

/// Key of the stored subject in a policy's persisted / published form.
const HEADERS_KEY: &str = "_rustic_security_headers";

impl Policy {
    /// The persisted and published form: [`Policy::config_json`] plus the
    /// stored subject.
    pub fn stored_json(&self) -> Value {
        let mut v = self.config_json();
        if let (Some(h), Some(o)) = (&self.headers, v.as_object_mut()) {
            o.insert(HEADERS_KEY.into(), json!(h.encode()));
        }
        v
    }

    /// `{"<type>": {name, query?, indices, match_field, enrich_fields}}`.
    pub fn config_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("name".into(), json!(self.name));
        if let Some(q) = &self.query {
            m.insert("query".into(), q.clone());
        }
        m.insert("indices".into(), json!(self.indices));
        m.insert("match_field".into(), json!(self.match_field));
        m.insert("enrich_fields".into(), json!(self.enrich_fields));
        let mut o = Map::new();
        o.insert(self.kind.clone(), Value::Object(m));
        Value::Object(o)
    }

    pub fn alias(&self) -> String {
        alias_name(&self.name)
    }
}

pub fn alias_name(policy: &str) -> String {
    format!(".enrich-{policy}")
}

/// Is `index` a concrete enrich index of `policy` (`.enrich-<policy>-<digits>`)?
pub fn is_policy_index(policy: &str, index: &str) -> bool {
    index
        .strip_prefix(&format!(".enrich-{policy}-"))
        .map(|r| !r.is_empty() && r.bytes().all(|b| b.is_ascii_digit()))
        .unwrap_or(false)
}

fn str_array(v: &Value, field: &str) -> Result<Vec<String>> {
    match v {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::Array(a) => a
            .iter()
            .map(|x| x.as_str().map(str::to_string).ok_or_else(|| EsError::x_content_parse(format!("[1:2] [policy] failed to parse field [{field}]"))))
            .collect(),
        _ => Err(EsError::x_content_parse(format!("[1:2] [policy] failed to parse field [{field}]"))),
    }
}

/// Parse a PUT body; returns the policy and whether `elasticsearch_version`
/// was given (deprecation warning).
pub fn parse_policy(name: &str, body: &Value) -> Result<(Policy, bool)> {
    let o = match body {
        Value::Object(o) => o,
        Value::Null => return Err(EsError::new(400, "parse_exception", "request body is required")),
        _ => return Err(EsError::new(400, "parsing_exception", "unexpected token")),
    };
    let (kind, inner) = match o.iter().next() {
        Some(x) => x,
        None => return Err(EsError::new(400, "parsing_exception", "unexpected token")),
    };
    if o.len() > 1 {
        return Err(EsError::new(400, "parsing_exception", "unexpected token"));
    }
    let inner = inner.as_object().ok_or_else(|| EsError::new(400, "parsing_exception", "unexpected token"))?;
    let mut indices = None;
    let mut match_field = None;
    let mut enrich_fields = None;
    let mut query = None;
    let mut version = false;
    for (k, v) in inner {
        match k.as_str() {
            "indices" => indices = Some(str_array(v, "indices")?),
            "match_field" => {
                match_field = Some(v.as_str().map(str::to_string).ok_or_else(|| EsError::x_content_parse("[1:2] [policy] failed to parse field [match_field]"))?)
            }
            "enrich_fields" => enrich_fields = Some(str_array(v, "enrich_fields")?),
            "query" => {
                if !v.is_object() {
                    return Err(EsError::x_content_parse("[1:2] [policy] failed to parse field [query]"));
                }
                query = Some(v.clone());
            }
            "elasticsearch_version" => version = true,
            other => return Err(EsError::x_content_parse(format!("[1:2] [policy] unknown field [{other}]"))),
        }
    }
    let mut missing = Vec::new();
    if indices.is_none() {
        missing.push("indices");
    }
    if match_field.is_none() {
        missing.push("match_field");
    }
    if enrich_fields.is_none() {
        missing.push("enrich_fields");
    }
    if !missing.is_empty() {
        return Err(EsError::illegal_argument(format!("Required [{}]", missing.join(", "))));
    }
    let p = Policy { name: name.to_string(), kind: kind.clone(), indices: indices.unwrap(), match_field: match_field.unwrap(), enrich_fields: enrich_fields.unwrap(), query, headers: None };
    Ok((p, version))
}

/// `MetadataCreateIndexService.validateIndexOrAliasName` + lowercase, as
/// `EnrichStore.putPolicy` applies them to policy names.
/// A policy as [`Policy::config_json`] stores it (its `name` included).
fn parse_stored(name: &str, cfg: &Value) -> Result<Policy> {
    let mut cfg = cfg.clone();
    let headers = cfg.as_object_mut().and_then(|o| o.remove(HEADERS_KEY));
    if let Some(inner) = cfg.as_object_mut().and_then(|o| o.values_mut().next()).and_then(Value::as_object_mut) {
        inner.remove("name");
    }
    let mut p = parse_policy(name, &cfg)?.0;
    p.headers = match headers {
        None => None,
        // Unreadable stored headers: refuse the policy rather than run it unbound.
        Some(h) => Some(h.as_str().and_then(StoredAuthn::decode).ok_or_else(|| EsError::internal(format!("unreadable security headers of enrich policy [{name}]")))?),
    };
    Ok(p)
}

fn validate_policy(p: &Policy) -> Result<()> {
    let name = &p.name;
    let bad = |why: &str| EsError::illegal_argument(format!("Invalid policy name [{name}], {why}"));
    if name.is_empty() {
        return Err(EsError::illegal_argument("name is missing or empty"));
    }
    const INVALID: &[char] = &['\\', '/', '*', '?', '"', '<', '>', '|', ' ', ','];
    if name.contains(INVALID) {
        return Err(bad("must not contain the following characters [ , \", *, \\, <, |, ,, >, /, ?]"));
    }
    if name.contains('#') {
        return Err(bad("must not contain '#'"));
    }
    if name.contains(':') {
        return Err(bad("must not contain ':'"));
    }
    if name.starts_with(['_', '-', '+']) {
        return Err(bad("must not start with '_', '-', or '+'"));
    }
    if name.len() > 255 {
        return Err(bad(&format!("index name is too long, ({} > 255)", name.len())));
    }
    if name == "." || name == ".." {
        return Err(bad("must not be '.' or '..'"));
    }
    if name.to_lowercase() != *name {
        return Err(bad("must be lowercase"));
    }
    if !SUPPORTED_TYPES.contains(&p.kind.as_str()) {
        return Err(EsError::illegal_argument(format!("unsupported policy type [{}], supported types are [match, geo_match, range]", p.kind)));
    }
    if p.indices.is_empty() {
        return Err(EsError::illegal_argument("indices is missing or empty"));
    }
    if p.match_field.is_empty() {
        return Err(EsError::illegal_argument("matchField is missing or empty"));
    }
    if p.enrich_fields.is_empty() {
        return Err(EsError::illegal_argument("enrichFields is missing or empty"));
    }
    Ok(())
}

// ------------------------------------------------------------ store

#[derive(Default)]
pub struct Stats {
    pub remote_requests_total: AtomicU64,
    pub executed_searches_total: AtomicU64,
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
    pub hits_nanos: AtomicU64,
    pub misses_nanos: AtomicU64,
}

/// The lookup cache: results by key with their charged bytes, the bytes
/// charged in all and the insertion order (eviction oldest first).
#[derive(Default)]
struct LookupCache {
    entries: HashMap<String, (Arc<Vec<Value>>, u64)>,
    order: std::collections::VecDeque<String>,
    bytes: u64,
}

impl LookupCache {
    fn get(&self, key: &str) -> Option<Arc<Vec<Value>>> {
        self.entries.get(key).map(|(v, _)| v.clone())
    }

    fn clear(&mut self) {
        *self = LookupCache::default();
    }

    /// Cache `hits` under `key` within `max_entries` and `max_bytes`;
    /// returns how many entries were evicted. A result over
    /// [`CACHE_MAX_ENTRY_BYTES`] (or `max_bytes`) is not cached.
    fn insert(&mut self, key: String, hits: Arc<Vec<Value>>, max_entries: usize, max_bytes: u64) -> u64 {
        let charge = key.len() as u64 + 64 + hits.iter().map(|h| serde_json::to_vec(h).map(|b| b.len() as u64 + 32).unwrap_or(64)).sum::<u64>();
        if charge > CACHE_MAX_ENTRY_BYTES.min(max_bytes) || self.entries.contains_key(&key) {
            return 0;
        }
        let mut evicted = 0;
        while !self.order.is_empty() && (self.entries.len() >= max_entries || self.bytes + charge > max_bytes) {
            if let Some(old) = self.order.pop_front() {
                if let Some((_, b)) = self.entries.remove(&old) {
                    self.bytes -= b;
                    evicted += 1;
                }
            }
        }
        self.bytes += charge;
        self.order.push_back(key.clone());
        self.entries.insert(key, (hits, charge));
        evicted
    }
}

/// A node's policies, executions in flight and lookup cache.
pub struct EnrichStore {
    path: PathBuf,
    policies: Mutex<BTreeMap<String, Policy>>,
    /// Policy name → task id of the running execution.
    pub executing: Mutex<BTreeMap<String, (u64, i64)>>,
    pub stats: Stats,
    cache: Mutex<LookupCache>,
    /// Searches an enrich index on the nodes that hold it (installed by a
    /// cluster node; see [`RemoteLookup`]).
    remote: parking_lot::RwLock<Option<RemoteLookup>>,
}

/// A search of an enrich index (concrete name, search body) on a node that
/// holds a copy of it, answering the search response. ES sends every enrich
/// lookup through `EnrichCoordinatorProxyAction`, which searches the enrich
/// index wherever a copy is allocated; a cluster node installs this so a
/// node without a local copy (a dedicated ingest or coordinating node) can
/// still look up. Called on blocking (never async) threads.
pub type RemoteLookup = Arc<dyn Fn(&str, &Value) -> Result<Value> + Send + Sync>;

/// Install (or remove) the node's [`RemoteLookup`].
pub fn set_remote_lookup(node: &Node, lookup: Option<RemoteLookup>) {
    *store(node).remote.write() = lookup;
}

fn registry() -> &'static Mutex<HashMap<PathBuf, Arc<EnrichStore>>> {
    static R: OnceLock<Mutex<HashMap<PathBuf, Arc<EnrichStore>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The node's enrich store.
pub fn store(node: &Node) -> Arc<EnrichStore> {
    let mut r = registry().lock();
    r.entry(node.data.clone()).or_insert_with(|| Arc::new(EnrichStore::open(&node.data))).clone()
}

impl EnrichStore {
    fn open(data: &Path) -> EnrichStore {
        let path = data.join("_state").join("enrich.json");
        let mut map = BTreeMap::new();
        if let Ok(b) = std::fs::read(&path) {
            if let Ok(Value::Object(m)) = serde_json::from_slice::<Value>(&b) {
                for (name, cfg) in m {
                    if let Ok(p) = parse_stored(&name, &cfg) {
                        map.insert(name, p);
                    }
                }
            }
        }
        EnrichStore { path, policies: Mutex::new(map), executing: Mutex::new(BTreeMap::new()), stats: Stats::default(), cache: Mutex::new(LookupCache::default()), remote: parking_lot::RwLock::new(None) }
    }

    fn persist(&self, map: &BTreeMap<String, Policy>) -> Result<()> {
        let mut v = Map::new();
        for (k, p) in map {
            v.insert(k.clone(), p.stored_json());
        }
        if let Some(d) = self.path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(&Value::Object(v))?)?;
        std::fs::rename(tmp, &self.path)?;
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<Policy> {
        self.policies.lock().get(name).cloned()
    }

    /// Policies by comma-separated names (all when empty); unknown names are skipped.
    pub fn list(&self, names: &[String]) -> Vec<Policy> {
        let m = self.policies.lock();
        if names.is_empty() || names.iter().any(|n| n == "_all" || n == "*") {
            return m.values().cloned().collect();
        }
        m.values().filter(|p| names.iter().any(|n| crate::mapping::wildcard_match(n, &p.name))).cloned().collect()
    }

    pub fn put(&self, p: Policy) -> Result<()> {
        validate_policy(&p)?;
        let mut m = self.policies.lock();
        if m.contains_key(&p.name) {
            return Err(EsError::new(400, "resource_already_exists_exception", format!("policy [{}] already exists", p.name)));
        }
        m.insert(p.name.clone(), p);
        self.persist(&m)
    }

    /// Every policy's configuration by name, as published in cluster
    /// metadata (ES keeps enrich policies in `EnrichMetadata`).
    pub fn cluster_state(&self) -> Value {
        Value::Object(self.policies.lock().iter().map(|(k, p)| (k.clone(), p.stored_json())).collect())
    }

    /// Make this node's policies the published set.
    pub fn restore_cluster_state(&self, published: &Value) -> Result<()> {
        let mut want = BTreeMap::new();
        for (name, cfg) in published.as_object().into_iter().flatten() {
            want.insert(name.clone(), parse_stored(name, cfg)?);
        }
        let mut m = self.policies.lock();
        let same = m.len() == want.len() && m.iter().all(|(k, p)| want.get(k).is_some_and(|w| w.stored_json() == p.stored_json()));
        if same {
            return Ok(());
        }
        *m = want;
        self.persist(&m)?;
        drop(m);
        self.clear_cache();
        Ok(())
    }

    fn remove(&self, name: &str) -> Result<()> {
        let mut m = self.policies.lock();
        m.remove(name);
        self.persist(&m)
    }

    fn clear_cache(&self) {
        self.cache.lock().clear();
    }

    pub fn cache_json(&self) -> (usize, u64) {
        let c = self.cache.lock();
        (c.entries.len(), c.bytes)
    }
}

// ------------------------------------------------------------ node scope

thread_local! {
    static CURRENT: Cell<*const Node> = const { Cell::new(std::ptr::null()) };
}

/// Makes `node` visible to enrich processors compiled or run on this
/// thread until dropped (restores the previous scope).
pub struct NodeScope {
    prev: *const Node,
}

impl NodeScope {
    pub fn enter(node: &Node) -> NodeScope {
        NodeScope { prev: CURRENT.with(|c| c.replace(node as *const Node)) }
    }
}

impl Drop for NodeScope {
    fn drop(&mut self) {
        CURRENT.with(|c| c.set(self.prev));
    }
}

/// Run `f` with the node of the current [`NodeScope`], if any.
pub fn with_current<T>(f: impl FnOnce(&Node) -> T) -> Option<T> {
    let p = CURRENT.with(|c| c.get());
    if p.is_null() {
        return None;
    }
    // SAFETY: the pointer is only set while a `NodeScope` borrowing the node
    // is alive on this thread.
    Some(f(unsafe { &*p }))
}

// ------------------------------------------------------------ APIs

/// `PUT /_enrich/policy/{name}`; returns whether to emit the
/// `elasticsearch_version` deprecation warning.
pub fn put_policy(node: &Node, name: &str, body: &Value, headers: Option<StoredAuthn>) -> Result<bool> {
    let (mut p, version) = parse_policy(name, body)?;
    p.headers = headers;
    store(node).put(p)?;
    Ok(version)
}

/// `GET /_enrich/policy[/{names}]`.
pub fn get_policies(node: &Node, names: &[String]) -> Value {
    let ps: Vec<Value> = store(node).list(names).iter().map(|p| json!({"config": p.config_json()})).collect();
    json!({ "policies": ps })
}

/// Pipelines (ids) with an `enrich` processor using `policy`.
fn pipelines_using(node: &Node, policy: &str) -> Vec<String> {
    fn uses(v: &Value, policy: &str) -> bool {
        match v {
            Value::Object(m) => {
                if let Some(Value::Object(e)) = m.get("enrich") {
                    if e.get("policy_name").and_then(Value::as_str) == Some(policy) {
                        return true;
                    }
                }
                m.values().any(|x| uses(x, policy))
            }
            Value::Array(a) => a.iter().any(|x| uses(x, policy)),
            _ => false,
        }
    }
    node.ingest().get(None).into_iter().filter(|(_, cfg)| uses(cfg, policy)).map(|(id, _)| id).collect()
}

/// `DELETE /_enrich/policy/{name}`.
pub fn delete_policy(node: &Node, name: &str) -> Result<()> {
    let st = store(node);
    if st.get(name).is_none() {
        return Err(EsError::resource_not_found(format!("policy [{name}] not found")));
    }
    if st.executing.lock().contains_key(name) {
        return Err(EsError::new(429, "es_rejected_execution_exception", format!("Could not delete policy [{name}] because it is currently executing")));
    }
    let users = pipelines_using(node, name);
    if !users.is_empty() {
        return Err(EsError::illegal_argument(format!("Could not delete policy [{name}] because a pipeline is referencing it [{}]", users.join(", "))));
    }
    for svc in node.all_indices() {
        if is_policy_index(name, &svc.name) {
            let _ = node.delete_index(&svc.name);
        }
    }
    st.remove(name)?;
    st.clear_cache();
    Ok(())
}

/// Validate the source indices before an execution.
fn validate_sources(node: &Node, p: &Policy) -> Result<Vec<crate::search::Target>> {
    let mut opts = ResolveOpts::default();
    opts.allow_no_indices = true;
    let targets = node.resolve(&p.indices.join(","), opts)?;
    for t in &targets {
        let lk = t.svc.lookup();
        let f = lk.get(&p.match_field);
        let ty = f.map(|f| f.declared.name().to_string());
        let ok = match (p.kind.as_str(), ty.as_deref()) {
            (_, None) => {
                return Err(EsError::new(
                    500,
                    "elasticsearch_exception",
                    format!("Could not traverse mapping to field [{}]. Could not find the [{}] field under [{}]", p.match_field, p.match_field, t.svc.name),
                ))
            }
            ("range", Some(t)) => t.ends_with("_range"),
            // ES 9: any source type (values must parse as shapes when indexed).
            _ => true,
        };
        if !ok {
            return Err(EsError::new(
                500,
                "elasticsearch_exception",
                format!("Field '{}' has type [{}] which doesn't appear to be a {} type", p.match_field, ty.unwrap_or_default(), if p.kind == "range" { "range" } else { "geo" }),
            ));
        }
    }
    Ok(targets)
}

/// Merge property tree `add` (as [`nest_property`] builds) into `into`.
fn merge_properties(into: &mut Map<String, Value>, add: Value) {
    let Value::Object(add) = add else { return };
    for (k, v) in add {
        match (into.get_mut(&k), v) {
            (Some(Value::Object(cur)), Value::Object(v)) if cur.contains_key("properties") && v.contains_key("properties") => {
                if let (Some(Value::Object(cp)), Some(vp)) = (cur.get_mut("properties"), v.get("properties").cloned()) {
                    merge_properties(cp, vp);
                }
            }
            (Some(_), _) => {}
            (None, v) => {
                into.insert(k, v);
            }
        }
    }
}

/// The source mapping of enrich field `field`, as the enrich index keeps
/// it (ES `EnrichPolicyRunner`: the enrich fields are mapped like their
/// source fields, neither indexed nor with doc values), so that the
/// executed policy's output keeps its types whatever later happens to its
/// source indices. `None`: a type kept out of the enrich index mapping.
fn enrich_field_mapping(targets: &[crate::search::Target], field: &str) -> Option<Value> {
    for t in targets {
        let m = t.svc.meta().mappings.to_json();
        let def = field.split('.').try_fold(&m, |d, k| d["properties"].get(k))?;
        let ty = def["type"].as_str().unwrap_or("object");
        let mut def = def.clone();
        let o = def.as_object_mut()?;
        o.remove("fields");
        o.remove("copy_to");
        match ty {
            "keyword" | "long" | "integer" | "short" | "byte" | "double" | "float" | "half_float" | "scaled_float" | "unsigned_long" | "date" | "date_nanos" | "boolean" | "ip" | "geo_point" => {
                o.insert("index".into(), json!(false));
                o.insert("doc_values".into(), json!(false));
                o.remove("time_series_dimension");
                o.remove("time_series_metric");
            }
            "text" | "match_only_text" => {
                o.insert("type".into(), json!("text"));
                o.insert("index".into(), json!(false));
            }
            _ => return None,
        }
        return Some(def);
    }
    None
}

fn nest_property(path: &str, mapping: Value) -> Value {
    let parts: Vec<&str> = path.split('.').collect();
    let mut cur = mapping;
    for (i, part) in parts.iter().enumerate().rev() {
        let mut m = Map::new();
        m.insert(part.to_string(), cur);
        cur = if i == 0 { Value::Object(m) } else { json!({ "properties": Value::Object(m) }) };
    }
    cur
}

/// Run a policy: build `.enrich-<policy>-<millis>`, point the alias at it
/// and delete the previous enrich indices of the policy.
/// The index operations a policy execution performs. Locally they act on
/// the node's engine; a cluster performs them through its coordinator, so
/// the source is read from every shard and the enrich index replicates.
pub trait PolicyIndexOps {
    fn create_index(&self, name: &str, body: &Value) -> Result<()>;
    /// `_reindex` with `refresh=true`; returns its response.
    fn reindex(&self, body: &Value) -> Result<Value>;
    fn update_aliases(&self, actions: &[Value]) -> Result<()>;
    /// Sets `index.blocks.write`.
    fn block_writes(&self, index: &str) -> Result<()>;
    fn delete_index(&self, index: &str) -> Result<()>;
}

/// [`PolicyIndexOps`] on the node's own engine.
pub struct LocalPolicyOps<'a> {
    pub node: &'a Node,
    pub task: &'a crate::tasks::Task,
}

impl PolicyIndexOps for LocalPolicyOps<'_> {
    fn create_index(&self, name: &str, body: &Value) -> Result<()> {
        self.node.create_index(name, body).map(|_| ())
    }
    fn reindex(&self, body: &Value) -> Result<Value> {
        let mut params = HashMap::new();
        params.insert("refresh".to_string(), "true".to_string());
        let r = crate::reindex::parse_reindex(body, &body.to_string(), &params)?;
        let (status, out) = crate::reindex::execute(self.node, &r, self.task)?;
        if status >= 400 {
            return Err(EsError::new(500, "elasticsearch_exception", "Encountered bulk failures during reindex process"));
        }
        Ok(out)
    }
    fn update_aliases(&self, actions: &[Value]) -> Result<()> {
        self.node.update_aliases(actions).map(|_| ())
    }
    fn block_writes(&self, index: &str) -> Result<()> {
        if let Some(svc) = self.node.get_index(index) {
            let flat = crate::settings::normalize_index_settings(&json!({"index": {"blocks": {"write": true}}}));
            svc.put_settings(flat, false)?;
        }
        Ok(())
    }
    fn delete_index(&self, index: &str) -> Result<()> {
        self.node.delete_index(index)
    }
}

/// The subject a policy execution reads its source indices as (security
/// on): its document and field level security shape the enrich index.
pub struct SourceReader {
    pub authn: Authn,
    pub perm: Permission,
}

pub fn execute_policy(node: &Node, name: &str, task: &crate::tasks::Task, reader: Option<&SourceReader>) -> Result<String> {
    execute_policy_with(node, name, task, &LocalPolicyOps { node, task }, reader)
}

pub fn execute_policy_with(node: &Node, name: &str, task: &crate::tasks::Task, ops: &dyn PolicyIndexOps, reader: Option<&SourceReader>) -> Result<String> {
    let st = store(node);
    let p = st.get(name).ok_or_else(|| EsError::resource_not_found(format!("policy [{name}] does not exist")))?;
    {
        let mut ex = st.executing.lock();
        if ex.contains_key(name) {
            return Err(EsError::new(429, "es_rejected_execution_exception", format!("Policy execution failed. Policy execution for [{name}] is already in progress.")));
        }
        ex.insert(name.to_string(), (task.id, task.start_millis));
    }
    let r = run_policy(node, &p, task, ops, reader);
    st.executing.lock().remove(name);
    st.clear_cache();
    r
}

/// The source query and `_source` fields of an execution reading as
/// `reader`: its document level security restricts each source index's
/// documents (ANDed in per index), and fields its field level security
/// hides are left out. A field FLS hides cannot be the match field, and an
/// object field under FLS (whose sub-fields FLS may partly hide) is refused
/// rather than copied whole.
fn restricted_sources(p: &Policy, targets: &[crate::search::Target], reader: Option<&SourceReader>) -> Result<(Option<Value>, Vec<String>)> {
    const SEARCH: &str = "indices:data/read/search";
    let mut fields = vec![p.match_field.clone()];
    fields.extend(p.enrich_fields.iter().cloned());
    let Some(r) = reader.filter(|r| r.perm.has_dls_fls()) else { return Ok((p.query.clone(), fields)) };
    let denied = |why: String| EsError::new(403, "security_exception", why);
    let mut per_index: Vec<(String, Option<Value>)> = Vec::new();
    for t in targets {
        let name = t.svc.name.clone();
        if per_index.iter().any(|(n, _)| *n == name) {
            continue;
        }
        let access = r.perm.index_access(SEARCH, &[name.as_str()], &r.authn)
            .map_err(|_| denied(format!("unable to read the source index [{name}] of enrich policy [{}] with document or field level security", p.name)))?;
        if let Some(fa) = &access.fields {
            if !fa.allows(&p.match_field) {
                return Err(denied(format!("field level security on the source index [{name}] hides the match field [{}] of enrich policy [{}]", p.match_field, p.name)));
            }
            let lookup = t.svc.lookup();
            for f in &p.enrich_fields {
                if !fa.allows(f) {
                    fields.retain(|k| k != f);
                } else if lookup.get(f).is_none() {
                    return Err(denied(format!("enrich field [{f}] of enrich policy [{}] is not a leaf field of the source index [{name}] and cannot be read under field level security", p.name)));
                }
            }
        }
        per_index.push((name, access.dls));
    }
    if per_index.iter().all(|(_, d)| d.is_none()) {
        return Ok((p.query.clone(), fields));
    }
    let should: Vec<Value> = per_index
        .into_iter()
        .map(|(n, d)| match d {
            Some(d) => json!({"bool": {"filter": [{"term": {"_index": n}}, d]}}),
            None => json!({"term": {"_index": n}}),
        })
        .collect();
    let restrict = json!({"bool": {"should": should, "minimum_should_match": 1}});
    let query = match &p.query {
        Some(q) => json!({"bool": {"filter": [q, restrict]}}),
        None => restrict,
    };
    Ok((Some(query), fields))
}

/// The enrich row id of a source row: its index, routing and id, unique
/// across all the policy's source indices and shards.
const ROW_ID_SCRIPT: &str = "ctx._id = ctx._index + '|' + (ctx._routing == null ? '' : ctx._routing) + '|' + ctx._id";

fn run_policy(node: &Node, p: &Policy, task: &crate::tasks::Task, ops: &dyn PolicyIndexOps, reader: Option<&SourceReader>) -> Result<String> {
    task.set_status(json!({"phase": "RUNNING"}));
    let targets = validate_sources(node, p)?;
    let (source_query, source_fields) = restricted_sources(p, &targets, reader)?;
    // Mapping of the match field in the enrich index.
    let mf_mapping = match p.kind.as_str() {
        "match" => json!({"type": "keyword", "doc_values": false}),
        "geo_match" => json!({"type": "geo_shape"}),
        _ => {
            let mut m = json!({"type": "long_range"});
            if let Some(t) = targets.first() {
                if let Some(f) = t.svc.lookup().get(&p.match_field) {
                    m = json!({"type": f.declared.name()});
                    if f.declared.name() == "date_range" {
                        m["format"] = json!(f.date_format.spec);
                    }
                }
            }
            m
        }
    };
    let mut millis = crate::date::now_millis();
    let mut index = format!(".enrich-{}-{millis}", p.name);
    while node.get_index(&index).is_some() {
        millis += 1;
        index = format!(".enrich-{}-{millis}", p.name);
    }
    // The enrich fields' types as executed: bound to this generation.
    let mut properties = match nest_property(&p.match_field, mf_mapping) {
        Value::Object(o) => o,
        _ => Map::new(),
    };
    for f in p.enrich_fields.iter().filter(|f| **f != p.match_field) {
        if let Some(def) = enrich_field_mapping(&targets, f) {
            merge_properties(&mut properties, nest_property(f, def));
        }
    }
    let body = json!({
        "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0, "auto_expand_replicas": "0-all", "hidden": true}},
        "mappings": {
            "dynamic": false,
            "_source": {"enabled": true},
            "_meta": {"enrich_readme": README, "enrich_policy_name": p.name, "enrich_match_field": p.match_field, "enrich_policy_type": p.kind, "enrich_field_types": true},
            "properties": properties,
        }
    });
    ops.create_index(&index, &body)?;
    task.set_status(json!({"phase": "RUNNING", "step": "REINDEX"}));
    let fail = |e: EsError| {
        let _ = ops.delete_index(&index);
        e
    };
    if !targets.is_empty() {
        let mut src = json!({"index": p.indices, "_source": source_fields});
        if let Some(q) = &source_query {
            src["query"] = q.clone();
        }
        // Source `_id`s are local to their index (and routing): two source
        // rows may share one. ES's enrich reindex pipeline removes `_id` so
        // every row is kept; here each row gets an id unique across the
        // policy's sources (finding 165).
        let rb = json!({"source": src, "dest": {"index": index}, "script": {"lang": "painless", "source": ROW_ID_SCRIPT}});
        let out = ops.reindex(&rb).map_err(fail)?;
        if out.get("failures").and_then(Value::as_array).map(|a| !a.is_empty()).unwrap_or(false) {
            let _ = ops.delete_index(&index);
            return Err(EsError::new(500, "elasticsearch_exception", "Encountered bulk failures during reindex process"));
        }
        // Every source row must be a row of its own.
        let (total, created) = (out["total"].as_u64(), out["created"].as_u64());
        let overwritten = out["updated"].as_u64().unwrap_or(0);
        if overwritten > 0 || matches!((total, created), (Some(t), Some(c)) if c < t.saturating_sub(out["noops"].as_u64().unwrap_or(0))) {
            let _ = ops.delete_index(&index);
            return Err(EsError::new(500, "elasticsearch_exception", format!("the enrich index of policy [{}] lost source rows ({} of {} written)", p.name, created.unwrap_or(0), total.unwrap_or(0))));
        }
    }
    task.set_status(json!({"phase": "RUNNING", "step": "UPDATE_ALIAS"}));
    let alias = p.alias();
    // The alias moves from exactly the indices that hold it now, each
    // removal `must_exist`: if another execution moved or deleted it in the
    // meantime, the whole swap fails and this execution deletes only its own
    // index. Only the predecessors it removed are deleted, never another
    // execution's candidate (finding 164).
    let holders: Vec<String> = node.all_indices().into_iter().filter(|s| s.name != index && s.aliases().contains_key(&alias)).map(|s| s.name.clone()).collect();
    let mut actions = vec![json!({"add": {"index": index, "alias": alias, "is_hidden": true}})];
    for o in &holders {
        actions.push(json!({"remove": {"index": o, "alias": alias, "must_exist": true}}));
    }
    ops.update_aliases(&actions).map_err(fail)?;
    let _ = ops.block_writes(&index);
    for o in holders {
        let _ = ops.delete_index(&o);
    }
    Ok(index)
}

/// `GET /_enrich/_stats`.
pub fn stats(node: &Node) -> Value {
    let st = store(node);
    let now = crate::date::now_millis();
    let executing: Vec<Value> = st
        .executing
        .lock()
        .iter()
        .map(|(name, (id, start))| {
            json!({
                "name": name,
                "task": {
                    "node": crate::node::NODE_ID, "id": id, "type": "transport", "action": "cluster:admin/xpack/enrich/execute",
                    "description": format!("policy execution [{name}]"), "start_time_in_millis": start,
                    "running_time_in_nanos": ((now - start).max(0) as u64) * 1_000_000, "cancellable": false, "headers": {}
                }
            })
        })
        .collect();
    let s = &st.stats;
    let (count, bytes) = st.cache_json();
    json!({
        "executing_policies": executing,
        "coordinator_stats": [{
            "node_id": crate::node::NODE_ID,
            "queue_size": 0,
            "remote_requests_current": 0,
            "remote_requests_total": s.remote_requests_total.load(Ordering::Relaxed),
            "executed_searches_total": s.executed_searches_total.load(Ordering::Relaxed),
        }],
        "cache_stats": [{
            "node_id": crate::node::NODE_ID,
            "count": count,
            "hits": s.hits.load(Ordering::Relaxed),
            "hits_time_in_millis": s.hits_nanos.load(Ordering::Relaxed) / 1_000_000,
            "misses": s.misses.load(Ordering::Relaxed),
            "misses_time_in_millis": s.misses_nanos.load(Ordering::Relaxed) / 1_000_000,
            "evictions": s.evictions.load(Ordering::Relaxed),
            "size_in_bytes": bytes,
        }]
    })
}

// ------------------------------------------------------------ lookups

/// Test seam (hidden): the next `n` lookups of policy `p` fail as an
/// unavailable enrich shard would.
#[doc(hidden)]
pub static FAIL_LOOKUPS: parking_lot::Mutex<Option<(String, usize)>> = parking_lot::Mutex::new(None);

/// Search the policy's enrich index (through its alias) and return the
/// `_source` of up to `size` matches. Results are cached until the next
/// execution of the policy.
pub fn lookup(node: &Node, policy: &str, query: Value, size: usize) -> Result<Arc<Vec<Value>>> {
    {
        let mut fail = FAIL_LOOKUPS.lock();
        if let Some((p, n)) = fail.as_mut().filter(|(p, n)| p == policy && *n > 0) {
            *n -= 1;
            return Err(EsError::new(503, "no_shard_available_action_exception", format!("injected lookup failure for enrich policy [{p}]")));
        }
    }
    let st = store(node);
    let alias = alias_name(policy);
    let start = Instant::now();
    let mut opts = ResolveOpts::default();
    opts.expand_hidden = true;
    let targets = node.resolve(&alias, opts)?;
    let key = format!("{}|{size}|{query}", targets.first().map(|t| t.svc.name.as_str()).unwrap_or(""));
    if let Some(v) = st.cache.lock().get(&key) {
        st.stats.hits.fetch_add(1, Ordering::Relaxed);
        st.stats.hits_nanos.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        return Ok(v);
    }
    st.stats.remote_requests_total.fetch_add(1, Ordering::Relaxed);
    st.stats.executed_searches_total.fetch_add(1, Ordering::Relaxed);
    let body = json!({"query": {"constant_score": {"filter": query}}, "size": size, "track_total_hits": false});
    // No local copy of the enrich index (a cluster's dedicated ingest or
    // coordinating node holds only its metadata): search it where it is
    // allocated, as ES's coordinator proxy does.
    let remote = if crate::search::require_local_shards(&targets).is_err() { st.remote.read().clone() } else { None };
    let v: Value = match (remote, targets.first()) {
        (Some(remote), Some(target)) => remote(&target.svc.name, &body)?,
        _ => {
            let req = crate::search::parse_request(&body, &HashMap::new())?;
            let out = crate::search::execute(&targets, &req, &crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None })?;
            serde_json::from_str(&out.json)?
        }
    };
    // An incomplete answer (timed out, or shards failed) is no lookup
    // result: neither applied nor cached as an absence (finding 162).
    complete_lookup(&v, &alias)?;
    let hits: Vec<Value> = v["hits"]["hits"].as_array().map(|a| a.iter().map(|h| h["_source"].clone()).collect()).unwrap_or_default();
    let hits = Arc::new(hits);
    let evicted = st.cache.lock().insert(key, hits.clone(), CACHE_MAX, CACHE_MAX_BYTES);
    st.stats.evictions.fetch_add(evicted, Ordering::Relaxed);
    st.stats.misses.fetch_add(1, Ordering::Relaxed);
    st.stats.misses_nanos.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(hits)
}

/// A lookup search response that covers the whole enrich index: not timed
/// out, no shard failed.
fn complete_lookup(v: &Value, alias: &str) -> Result<()> {
    let failed = v["_shards"]["failed"].as_u64().unwrap_or(0);
    if v["timed_out"] == true || failed > 0 {
        return Err(EsError::new(
            500,
            "elasticsearch_exception",
            format!("enrich lookup in [{alias}] returned incomplete results ({})", if failed > 0 { format!("{failed} shards failed") } else { "timed out".to_string() }),
        ));
    }
    Ok(())
}

/// Does the policy's enrich index (alias) exist?
pub fn has_enrich_index(node: &Node, policy: &str) -> bool {
    let alias = alias_name(policy);
    node.all_indices().iter().any(|s| s.aliases().contains_key(&alias))
}

/// Turn an ingest/ES|QL value into the GeoJSON shape a `geo_match`
/// lookup intersects (points as `"lat,lon"`, `[lon, lat]`, `{lat, lon}`,
/// WKT, GeoJSON, or a list of points → multipoint).
pub fn to_shape(v: &Value) -> Result<Value> {
    fn point(v: &Value) -> Option<[f64; 2]> {
        match v {
            Value::String(s) => {
                let (a, b) = s.split_once(',')?;
                Some([b.trim().parse().ok()?, a.trim().parse().ok()?])
            }
            Value::Array(a) if a.len() == 2 => Some([a[0].as_f64()?, a[1].as_f64()?]),
            Value::Object(m) if m.contains_key("lat") => Some([m.get("lon")?.as_f64()?, m.get("lat")?.as_f64()?]),
            _ => None,
        }
    }
    if let Some(p) = point(v) {
        return Ok(json!({"type": "point", "coordinates": p}));
    }
    match v {
        Value::Array(a) => {
            let pts: Option<Vec<[f64; 2]>> = a.iter().map(point).collect();
            match pts {
                Some(p) => Ok(json!({"type": "multipoint", "coordinates": p})),
                None => Err(EsError::illegal_argument(format!("unsupported value [{v}] for geo_match"))),
            }
        }
        Value::String(_) | Value::Object(_) => Ok(v.clone()),
        _ => Err(EsError::illegal_argument(format!("unsupported value [{v}] for geo_match"))),
    }
}

/// The lookup query of a policy type for `value`.
pub fn lookup_query(kind: &str, match_field: &str, value: &Value, shape_relation: &str) -> Result<Value> {
    Ok(match kind {
        "geo_match" => json!({"geo_shape": {match_field: {"shape": to_shape(value)?, "relation": shape_relation}}}),
        _ => match value {
            Value::Array(a) => json!({"terms": {match_field: a}}),
            v => json!({"term": {match_field: v}}),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_policies_round_trip_and_reload() {
        let master = tempfile::tempdir().unwrap();
        let peer = tempfile::tempdir().unwrap();
        let a = EnrichStore::open(master.path());
        let b = EnrichStore::open(peer.path());
        let (p, _) = parse_policy("p1", &json!({"match": {"indices": "src", "match_field": "k", "enrich_fields": ["a"], "query": {"match_all": {}}}})).unwrap();
        a.put(p).unwrap();
        // What the master publishes, another node applies.
        b.restore_cluster_state(&a.cluster_state()).unwrap();
        assert_eq!(b.cluster_state(), a.cluster_state());
        assert_eq!(b.get("p1").unwrap().match_field, "k");
        // ... and keeps across a restart (stored with its `name`).
        assert_eq!(EnrichStore::open(peer.path()).cluster_state(), a.cluster_state());
        // A delete of the last policy is published as an empty set.
        b.restore_cluster_state(&json!({})).unwrap();
        assert!(b.get("p1").is_none());
        assert!(EnrichStore::open(peer.path()).get("p1").is_none());
    }

    fn source_node() -> (tempfile::TempDir, Arc<Node>) {
        let d = tempfile::tempdir().unwrap();
        let node = Node::open(d.path()).unwrap();
        for (index, docs) in [("src-a", vec![("1", None, r#"{"key":"alice","payload":"from-a"}"#)]), ("src-b", vec![("1", None, r#"{"key":"bob","payload":"from-b"}"#), ("2", Some("r1"), r#"{"key":"carol","payload":"r1"}"#), ("2", Some("r2"), r#"{"key":"dave","payload":"r2"}"#)])] {
            let svc = node.create_index(index, &json!({"settings": {"number_of_shards": 2}, "mappings": {"properties": {"key": {"type": "keyword"}, "payload": {"type": "keyword"}}}})).unwrap();
            let reqs = docs.into_iter().map(|(id, routing, src)| crate::index::WriteReq::Index { id: Some(id.into()), source: src.into(), routing: routing.map(str::to_string), cond: Default::default() }).collect();
            for r in svc.write(reqs, true) {
                r.unwrap();
            }
            svc.refresh().unwrap();
        }
        (d, node)
    }

    fn found(node: &Node, policy: &str, key: &str) -> Vec<Value> {
        lookup(node, policy, json!({"term": {"key": key}}), 10).unwrap().to_vec()
    }

    /// Finding 165: rows of different source indices (and of one index's
    /// routings) that share an `_id` are all kept.
    #[test]
    fn colliding_source_ids_are_all_kept() {
        let (_d, node) = source_node();
        put_policy(&node, "people", &json!({"match": {"indices": ["src-a", "src-b"], "match_field": "key", "enrich_fields": ["payload"]}}), None).unwrap();
        for round in 0..2 {
            let task = node.tasks.register("cluster:admin/xpack/enrich/execute", None, false, false, Map::new());
            execute_policy(&node, "people", &task, None).unwrap();
            for (key, payload) in [("alice", "from-a"), ("bob", "from-b"), ("carol", "r1"), ("dave", "r2")] {
                let hits = found(&node, "people", key);
                assert_eq!(hits.len(), 1, "round {round}: {key}");
                assert_eq!(hits[0]["payload"], payload, "round {round}: {key}");
            }
        }
    }

    /// Runs another execution of the same policy to completion just before
    /// this one's alias swap (an execution overlapping across masters).
    struct Overlapping<'a> {
        inner: LocalPolicyOps<'a>,
        policy: Policy,
        other: std::cell::RefCell<Option<String>>,
    }

    impl PolicyIndexOps for Overlapping<'_> {
        fn create_index(&self, name: &str, body: &Value) -> Result<()> {
            self.inner.create_index(name, body)
        }
        fn reindex(&self, body: &Value) -> Result<Value> {
            self.inner.reindex(body)
        }
        fn update_aliases(&self, actions: &[Value]) -> Result<()> {
            if self.other.borrow().is_none() {
                let b = run_policy(self.inner.node, &self.policy, self.inner.task, &self.inner, None).unwrap();
                *self.other.borrow_mut() = Some(b);
            }
            self.inner.update_aliases(actions)
        }
        fn block_writes(&self, index: &str) -> Result<()> {
            self.inner.block_writes(index)
        }
        fn delete_index(&self, index: &str) -> Result<()> {
            self.inner.delete_index(index)
        }
    }

    /// Finding 164: an execution whose predecessor was replaced by an
    /// overlapping execution fails without touching the winner: the
    /// winner's index keeps the alias and its data.
    #[test]
    fn an_overlapped_execution_never_deletes_the_winner() {
        let (_d, node) = source_node();
        put_policy(&node, "people", &json!({"match": {"indices": ["src-a", "src-b"], "match_field": "key", "enrich_fields": ["payload"]}}), None).unwrap();
        let task = node.tasks.register("cluster:admin/xpack/enrich/execute", None, false, false, Map::new());
        let first = execute_policy(&node, "people", &task, None).unwrap();
        let policy = store(&node).get("people").unwrap();
        let ops = Overlapping { inner: LocalPolicyOps { node: &node, task: &task }, policy: policy.clone(), other: Default::default() };
        let loser = run_policy(&node, &policy, &task, &ops, None);
        assert!(loser.is_err(), "{loser:?}");
        let winner = ops.other.borrow().clone().unwrap();
        let holders: Vec<String> = node.all_indices().into_iter().filter(|s| s.aliases().contains_key(&alias_name("people"))).map(|s| s.name.clone()).collect();
        assert_eq!(holders, vec![winner.clone()]);
        assert!(node.get_index(&first).is_none(), "the replaced predecessor is deleted");
        let candidates: Vec<String> = node.all_indices().into_iter().map(|s| s.name.clone()).filter(|n| is_policy_index("people", n)).collect();
        assert_eq!(candidates, vec![winner], "the loser deletes only its own index");
        store(&node).clear_cache();
        assert_eq!(found(&node, "people", "alice").len(), 1);
    }

    /// Finding 162: a timed-out or shard-failed lookup answer is an error.
    #[test]
    fn incomplete_lookups_are_refused() {
        assert!(complete_lookup(&json!({"timed_out": false, "_shards": {"failed": 0}, "hits": {"hits": []}}), "a").is_ok());
        assert!(complete_lookup(&json!({"timed_out": true, "_shards": {"failed": 0}, "hits": {"hits": []}}), "a").is_err());
        assert!(complete_lookup(&json!({"timed_out": false, "_shards": {"failed": 1}, "hits": {"hits": []}}), "a").is_err());
    }

    /// Finding 163: the lookup cache is bounded by bytes, evicts oldest
    /// first, and never caches a result over the per-entry limit.
    #[test]
    fn lookup_cache_is_bounded_by_bytes() {
        let mut c = LookupCache::default();
        let big = |n: usize| Arc::new(vec![json!({"payload": "x".repeat(n)})]);
        let mut evicted = 0;
        for i in 0..200 {
            evicted += c.insert(format!("k{i}"), big(10_000), CACHE_MAX, 100_000);
            assert!(c.bytes <= 100_000, "{} bytes", c.bytes);
        }
        assert!(evicted > 0 && c.entries.len() < 10);
        assert!(c.get("k199").is_some() && c.get("k0").is_none());
        // Oversized: returned to the caller, never retained.
        assert_eq!(c.insert("huge".into(), big(CACHE_MAX_ENTRY_BYTES as usize), CACHE_MAX, CACHE_MAX_BYTES), 0);
        assert!(c.get("huge").is_none());
        // The entry count still bounds it.
        let mut c = LookupCache::default();
        for i in 0..20 {
            c.insert(format!("k{i}"), big(10), 5, CACHE_MAX_BYTES);
        }
        assert_eq!(c.entries.len(), 5);
        c.clear();
        assert_eq!((c.entries.len(), c.bytes), (0, 0));
    }

    #[test]
    fn policy_parsing() {
        let (p, v) = parse_policy("p1", &json!({"match": {"indices": "src", "match_field": "k", "enrich_fields": ["a"], "elasticsearch_version": "x"}})).unwrap();
        assert!(v);
        assert_eq!(p.indices, vec!["src"]);
        assert_eq!(p.config_json()["match"]["name"], "p1");
        assert!(validate_policy(&Policy { kind: "foo".into(), ..p.clone() }).unwrap_err().reason.contains("unsupported policy type [foo]"));
        assert!(validate_policy(&Policy { name: "Up".into(), ..p.clone() }).unwrap_err().reason.contains("must be lowercase"));
        assert!(is_policy_index("p1", ".enrich-p1-123"));
        assert!(!is_policy_index("p1", ".enrich-p1-b-123"));
        assert_eq!(to_shape(&json!("40.0,-70.5")).unwrap()["coordinates"], json!([-70.5, 40.0]));
    }
}
