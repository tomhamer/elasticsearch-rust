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
}

impl Policy {
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
    let p = Policy { name: name.to_string(), kind: kind.clone(), indices: indices.unwrap(), match_field: match_field.unwrap(), enrich_fields: enrich_fields.unwrap(), query };
    Ok((p, version))
}

/// `MetadataCreateIndexService.validateIndexOrAliasName` + lowercase, as
/// `EnrichStore.putPolicy` applies them to policy names.
/// A policy as [`Policy::config_json`] stores it (its `name` included).
fn parse_stored(name: &str, cfg: &Value) -> Result<Policy> {
    let mut cfg = cfg.clone();
    if let Some(inner) = cfg.as_object_mut().and_then(|o| o.values_mut().next()).and_then(Value::as_object_mut) {
        inner.remove("name");
    }
    Ok(parse_policy(name, &cfg)?.0)
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

/// A node's policies, executions in flight and lookup cache.
pub struct EnrichStore {
    path: PathBuf,
    policies: Mutex<BTreeMap<String, Policy>>,
    /// Policy name → task id of the running execution.
    pub executing: Mutex<BTreeMap<String, (u64, i64)>>,
    pub stats: Stats,
    cache: Mutex<HashMap<String, Arc<Vec<Value>>>>,
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
        EnrichStore { path, policies: Mutex::new(map), executing: Mutex::new(BTreeMap::new()), stats: Stats::default(), cache: Mutex::new(HashMap::new()) }
    }

    fn persist(&self, map: &BTreeMap<String, Policy>) -> Result<()> {
        let mut v = Map::new();
        for (k, p) in map {
            v.insert(k.clone(), p.config_json());
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
        Value::Object(self.policies.lock().iter().map(|(k, p)| (k.clone(), p.config_json())).collect())
    }

    /// Make this node's policies the published set.
    pub fn restore_cluster_state(&self, published: &Value) -> Result<()> {
        let mut want = BTreeMap::new();
        for (name, cfg) in published.as_object().into_iter().flatten() {
            want.insert(name.clone(), parse_stored(name, cfg)?);
        }
        let mut m = self.policies.lock();
        let same = m.len() == want.len() && m.iter().all(|(k, p)| want.get(k).is_some_and(|w| w.config_json() == p.config_json()));
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
        let bytes: u64 = c.iter().map(|(k, v)| k.len() as u64 + v.iter().map(|x| x.to_string().len() as u64).sum::<u64>()).sum();
        (c.len(), bytes)
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
pub fn put_policy(node: &Node, name: &str, body: &Value) -> Result<bool> {
    let (p, version) = parse_policy(name, body)?;
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

pub fn execute_policy(node: &Node, name: &str, task: &crate::tasks::Task) -> Result<String> {
    execute_policy_with(node, name, task, &LocalPolicyOps { node, task })
}

pub fn execute_policy_with(node: &Node, name: &str, task: &crate::tasks::Task, ops: &dyn PolicyIndexOps) -> Result<String> {
    let st = store(node);
    let p = st.get(name).ok_or_else(|| EsError::resource_not_found(format!("policy [{name}] does not exist")))?;
    {
        let mut ex = st.executing.lock();
        if ex.contains_key(name) {
            return Err(EsError::new(429, "es_rejected_execution_exception", format!("Policy execution failed. Policy execution for [{name}] is already in progress.")));
        }
        ex.insert(name.to_string(), (task.id, task.start_millis));
    }
    let r = run_policy(node, &p, task, ops);
    st.executing.lock().remove(name);
    st.clear_cache();
    r
}

fn run_policy(node: &Node, p: &Policy, task: &crate::tasks::Task, ops: &dyn PolicyIndexOps) -> Result<String> {
    task.set_status(json!({"phase": "RUNNING"}));
    let targets = validate_sources(node, p)?;
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
    let body = json!({
        "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0, "auto_expand_replicas": "0-all", "hidden": true}},
        "mappings": {
            "dynamic": false,
            "_source": {"enabled": true},
            "_meta": {"enrich_readme": README, "enrich_policy_name": p.name, "enrich_match_field": p.match_field, "enrich_policy_type": p.kind},
            "properties": nest_property(&p.match_field, mf_mapping),
        }
    });
    ops.create_index(&index, &body)?;
    task.set_status(json!({"phase": "RUNNING", "step": "REINDEX"}));
    let fail = |e: EsError| {
        let _ = ops.delete_index(&index);
        e
    };
    if !targets.is_empty() {
        let mut fields = vec![p.match_field.clone()];
        fields.extend(p.enrich_fields.iter().cloned());
        let mut src = json!({"index": p.indices, "_source": fields});
        if let Some(q) = &p.query {
            src["query"] = q.clone();
        }
        let rb = json!({"source": src, "dest": {"index": index}});
        let out = ops.reindex(&rb).map_err(fail)?;
        if out.get("failures").and_then(Value::as_array).map(|a| !a.is_empty()).unwrap_or(false) {
            let _ = ops.delete_index(&index);
            return Err(EsError::new(500, "elasticsearch_exception", "Encountered bulk failures during reindex process"));
        }
    }
    task.set_status(json!({"phase": "RUNNING", "step": "UPDATE_ALIAS"}));
    let alias = p.alias();
    let mut actions = vec![json!({"add": {"index": index, "alias": alias, "is_hidden": true}})];
    let old: Vec<String> = node.all_indices().into_iter().map(|s| s.name.clone()).filter(|n| *n != index && is_policy_index(&p.name, n)).collect();
    for o in &old {
        if node.get_index(o).map(|s| s.aliases().contains_key(&alias)).unwrap_or(false) {
            actions.push(json!({"remove": {"index": o, "alias": alias}}));
        }
    }
    ops.update_aliases(&actions).map_err(fail)?;
    let _ = ops.block_writes(&index);
    for o in old {
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

/// Search the policy's enrich index (through its alias) and return the
/// `_source` of up to `size` matches. Results are cached until the next
/// execution of the policy.
pub fn lookup(node: &Node, policy: &str, query: Value, size: usize) -> Result<Arc<Vec<Value>>> {
    let st = store(node);
    let alias = alias_name(policy);
    let start = Instant::now();
    let mut opts = ResolveOpts::default();
    opts.expand_hidden = true;
    let targets = node.resolve(&alias, opts)?;
    let key = format!("{}|{size}|{query}", targets.first().map(|t| t.svc.name.as_str()).unwrap_or(""));
    if let Some(v) = st.cache.lock().get(&key).cloned() {
        st.stats.hits.fetch_add(1, Ordering::Relaxed);
        st.stats.hits_nanos.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        return Ok(v);
    }
    st.stats.remote_requests_total.fetch_add(1, Ordering::Relaxed);
    st.stats.executed_searches_total.fetch_add(1, Ordering::Relaxed);
    let body = json!({"query": {"constant_score": {"filter": query}}, "size": size, "track_total_hits": false});
    let req = crate::search::parse_request(&body, &HashMap::new())?;
    let out = crate::search::execute(&targets, &req, &crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None })?;
    let v: Value = serde_json::from_str(&out.json)?;
    let hits: Vec<Value> = v["hits"]["hits"].as_array().map(|a| a.iter().map(|h| h["_source"].clone()).collect()).unwrap_or_default();
    let hits = Arc::new(hits);
    {
        let mut c = st.cache.lock();
        if c.len() >= CACHE_MAX {
            let k = c.keys().next().cloned();
            if let Some(k) = k {
                c.remove(&k);
                st.stats.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        c.insert(key, hits.clone());
    }
    st.stats.misses.fetch_add(1, Ordering::Relaxed);
    st.stats.misses_nanos.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(hits)
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
