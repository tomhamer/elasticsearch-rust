//! Desired nodes (`_internal/desired_nodes`), ES 8.19 semantics.
//!
//! * `PUT /_internal/desired_nodes/{history_id}/{version}[?dry_run]`
//! * `GET /_internal/desired_nodes/_latest`
//! * `DELETE /_internal/desired_nodes`
//!
//! The latest definition is cluster metadata (ES `DesiredNodesMetadata`):
//! custom [`CUSTOM`] = `{"latest": {history_id, version, nodes: [...]}}`,
//! each node carrying a `status` (0 = PENDING, 1 = ACTUALIZED). Writes run as
//! a master state task so the definition is published, replicated and
//! survives a master failover; `GET` reads the local copy of the state.
//!
//! Parsing and request validation happen on the receiving node (as in ES,
//! where the REST layer parses and the transport action validates before
//! routing to the master); the history/version conflict checks run inside
//! the master task against the live state.

use crate::front::{RestReq, RestResp};
use crate::node::{ClusterNode, ES_VERSION};
use elasticsearch_cluster::state::ClusterState;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Metadata custom name (ES `DesiredNodesMetadata.TYPE`).
pub const CUSTOM: &str = "desired_nodes";

/// Same value as `rest/indices_apis.rs` (the build hash ES puts in warnings).
use elasticsearch_engine::version::BUILD_HASH;

const NODE_VERSION_DEPRECATION: &str = "[version removal] Specifying node_version in desired nodes requests is deprecated.";

const STATUS_PENDING: u8 = 0;
const STATUS_ACTUALIZED: u8 = 1;

/// `DiscoveryNodeRole` names a desired node may declare.
const ROLES: &[&str] = &[
    "data",
    "data_cold",
    "data_content",
    "data_frozen",
    "data_hot",
    "data_warm",
    "ingest",
    "master",
    "ml",
    "remote_cluster_client",
    "transform",
    "voting_only",
];

const COMMON_PARAMS: &[&str] = &["pretty", "human", "error_trace", "filter_path", "format"];

// ------------------------------------------------------------------ model

#[derive(Clone, Debug, PartialEq)]
pub enum Processors {
    Exact(f64),
    Range { min: f64, max: Option<f64> },
}

#[derive(Clone, Debug, PartialEq)]
pub struct DesiredNode {
    /// Flattened (`node.name` style) settings, values as given.
    pub settings: BTreeMap<String, Value>,
    pub processors: Processors,
    pub memory: String,
    pub storage: String,
    pub node_version: Option<String>,
}

impl DesiredNode {
    /// `node.external_id`, defaulting to `node.name`.
    pub fn external_id(&self) -> Option<String> {
        let get = |k: &str| self.settings.get(k).map(setting_string).filter(|s| !s.trim().is_empty());
        get("node.external_id").or_else(|| get("node.name"))
    }

    fn roles(&self) -> Result<Option<Vec<String>>, String> {
        let Some(v) = self.settings.get("node.roles") else { return Ok(None) };
        let list: Vec<String> = match v {
            Value::Array(a) => a.iter().map(setting_string).collect(),
            Value::Null => return Ok(None),
            other => {
                let s = setting_string(other);
                s.split(',').map(|r| r.trim().to_string()).filter(|r| !r.is_empty()).collect()
            }
        };
        for r in &list {
            if !ROLES.contains(&r.as_str()) {
                return Err(format!("unknown role [{r}]"));
            }
        }
        Ok(Some(list))
    }

    /// Default roles (no `node.roles`) include `master`.
    pub fn has_master_role(&self) -> bool {
        match self.roles() {
            Ok(Some(r)) => r.iter().any(|r| r == "master"),
            _ => true,
        }
    }

    fn to_json(&self, status: Option<u8>) -> Value {
        let mut o = Map::new();
        o.insert("settings".into(), nest_settings(&self.settings));
        match &self.processors {
            Processors::Exact(p) => {
                o.insert("processors".into(), json!(p));
            }
            Processors::Range { min, max } => {
                let mut r = Map::new();
                r.insert("min".into(), json!(min));
                if let Some(m) = max {
                    r.insert("max".into(), json!(m));
                }
                o.insert("processors_range".into(), Value::Object(r));
            }
        }
        o.insert("memory".into(), json!(self.memory));
        o.insert("storage".into(), json!(self.storage));
        if let Some(v) = &self.node_version {
            o.insert("node_version".into(), json!(v));
        }
        if let Some(s) = status {
            o.insert("status".into(), json!(s));
        }
        Value::Object(o)
    }

    fn from_json(v: &Value) -> Option<DesiredNode> {
        let o = v.as_object()?;
        let mut settings = BTreeMap::new();
        flatten("", o.get("settings")?, &mut settings);
        let processors = if let Some(p) = o.get("processors").and_then(Value::as_f64) {
            Processors::Exact(p)
        } else {
            let r = o.get("processors_range")?;
            Processors::Range { min: r.get("min")?.as_f64()?, max: r.get("max").and_then(Value::as_f64) }
        };
        Some(DesiredNode {
            settings,
            processors,
            memory: o.get("memory")?.as_str()?.to_string(),
            storage: o.get("storage")?.as_str()?.to_string(),
            node_version: o.get("node_version").and_then(Value::as_str).map(str::to_string),
        })
    }

    /// `DesiredNode.equalsWithProcessorsCloseTo`.
    fn equals_close(&self, other: &DesiredNode) -> bool {
        let close = |a: f64, b: f64| (a - b).abs() < 1e-5;
        let procs = match (&self.processors, &other.processors) {
            (Processors::Exact(a), Processors::Exact(b)) => close(*a, *b),
            (Processors::Range { min: a, max: am }, Processors::Range { min: b, max: bm }) => {
                close(*a, *b)
                    && match (am, bm) {
                        (Some(x), Some(y)) => close(*x, *y),
                        (None, None) => true,
                        _ => false,
                    }
            }
            _ => false,
        };
        procs
            && self.settings == other.settings
            && same_bytes(&self.memory, &other.memory)
            && same_bytes(&self.storage, &other.storage)
            && self.node_version == other.node_version
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DesiredNodes {
    pub history_id: String,
    pub version: i64,
    pub nodes: Vec<DesiredNode>,
}

impl DesiredNodes {
    pub fn from_state(st: &ClusterState) -> Option<DesiredNodes> {
        let l = st.metadata.customs.get(CUSTOM)?.get("latest")?;
        Some(DesiredNodes {
            history_id: l.get("history_id")?.as_str()?.to_string(),
            version: l.get("version")?.as_i64()?,
            nodes: l.get("nodes")?.as_array()?.iter().filter_map(DesiredNode::from_json).collect(),
        })
    }

    /// `GET _latest` body (no per-node status).
    pub fn to_api_json(&self) -> Value {
        json!({"history_id": self.history_id, "version": self.version, "nodes": self.nodes.iter().map(|n| n.to_json(None)).collect::<Vec<_>>()})
    }

    /// Cluster-state custom with each node's status, as ES computes it:
    /// a node keeps the status it had in the definition currently stored in
    /// `st` (`DesiredNodes.createIncludingStatusFromPreviousVersion`, matched
    /// by external id, any history), then every node whose external id is a
    /// member of `st.nodes` becomes ACTUALIZED
    /// (`DesiredNodes.updateDesiredNodesStatusIfNeeded`). ACTUALIZED means
    /// "has joined at least once": a node leaving never reverts it.
    pub fn to_custom(&self, st: &ClusterState) -> Value {
        // `DiscoveryNode.getExternalId()`: `node.external_id`, else the name.
        let present: BTreeSet<&str> = st.nodes.values().map(|n| n.attributes.get("external_id").map(String::as_str).unwrap_or(n.name.as_str())).collect();
        let previous = previous_statuses(st);
        let nodes: Vec<Value> = self
            .nodes
            .iter()
            .map(|n| {
                let s = match n.external_id() {
                    Some(id) if present.contains(id.as_str()) => STATUS_ACTUALIZED,
                    Some(id) => previous.get(&id).copied().unwrap_or(STATUS_PENDING),
                    None => STATUS_PENDING,
                };
                n.to_json(Some(s))
            })
            .collect();
        json!({"latest": {"history_id": self.history_id, "version": self.version, "nodes": nodes}})
    }

    fn by_external_id(&self) -> BTreeMap<String, &DesiredNode> {
        self.nodes.iter().filter_map(|n| Some((n.external_id()?, n))).collect()
    }

    /// `DesiredNodes.equalsWithProcessorsCloseTo` (order-insensitive).
    fn equals_close(&self, other: &DesiredNodes) -> bool {
        if self.history_id != other.history_id || self.version != other.version || self.nodes.len() != other.nodes.len() {
            return false;
        }
        let a = self.by_external_id();
        let b = other.by_external_id();
        a.len() == b.len() && a.iter().all(|(k, n)| b.get(k).map(|m| n.equals_close(m)).unwrap_or(false))
    }
}

/// Per-external-id status of the definition stored in `st`.
fn previous_statuses(st: &ClusterState) -> BTreeMap<String, u8> {
    let Some(nodes) = st.metadata.customs.get(CUSTOM).and_then(|c| c.get("latest")).and_then(|l| l.get("nodes")).and_then(Value::as_array) else {
        return BTreeMap::new();
    };
    nodes
        .iter()
        .filter_map(|v| {
            let status = v.get("status").and_then(Value::as_u64).map(|s| s as u8).unwrap_or(STATUS_PENDING);
            Some((DesiredNode::from_json(v)?.external_id()?, status))
        })
        .collect()
}

/// Re-evaluate desired node statuses against the current membership
/// (ES `DesiredNodes.updateDesiredNodesStatusIfNeeded`, run by
/// `JoinTaskExecutor` when nodes join and when a master is elected).
/// Pending nodes whose external id is now a member flip to ACTUALIZED;
/// nothing ever reverts. Bumps the metadata version only when a status
/// changed. Returns whether the state changed.
pub(crate) fn refresh_status(state: &mut ClusterState) -> bool {
    let Some(latest) = DesiredNodes::from_state(state) else { return false };
    let custom = latest.to_custom(state);
    if state.metadata.customs.get(CUSTOM) == Some(&custom) {
        return false;
    }
    state.metadata.customs.insert(CUSTOM.into(), custom);
    state.metadata.version += 1;
    true
}

// ------------------------------------------------------------ errors

/// An ES error response: status + the `error` object (with `root_cause`).
#[derive(Debug, Clone, PartialEq)]
pub struct ApiError {
    pub status: u16,
    pub error: Value,
}

impl ApiError {
    fn simple(status: u16, kind: &str, reason: &str) -> ApiError {
        ApiError { status, error: json!({"root_cause": [{"type": kind, "reason": reason}], "type": kind, "reason": reason}) }
    }

    fn bad(kind: &str, reason: &str) -> ApiError {
        ApiError::simple(400, kind, reason)
    }

    /// `x_content_parse_exception` raised while parsing node `idx`
    /// (`[update_desired_nodes_request] failed to parse field [nodes]`).
    fn node_parse(idx: usize, reason: String) -> ApiError {
        let col = 12 + idx;
        let inner = json!({"type": "x_content_parse_exception", "reason": format!("[1:{col}] {reason}")});
        let reason_top = format!("[1:{col}] [update_desired_nodes_request] failed to parse field [nodes]");
        ApiError {
            status: 400,
            error: json!({"root_cause": [inner.clone()], "type": "x_content_parse_exception", "reason": reason_top, "caused_by": inner}),
        }
    }

    /// A failure building node `idx` from its settings: ES wraps the
    /// constructor's `IllegalArgumentException` twice.
    fn node_settings(idx: usize, reason: &str) -> ApiError {
        let col = 12 + idx;
        let iae = json!({"type": "illegal_argument_exception", "reason": reason});
        let mid = json!({"type": "x_content_parse_exception", "reason": format!("[1:{col}] [desired_node] failed to parse field [settings]"), "caused_by": iae});
        ApiError {
            status: 400,
            error: json!({
                "root_cause": [{"type": "x_content_parse_exception", "reason": format!("[1:{col}] [desired_node] failed to parse field [settings]")}],
                "type": "x_content_parse_exception",
                "reason": format!("[1:{col}] [update_desired_nodes_request] failed to parse field [nodes]"),
                "caused_by": mid,
            }),
        }
    }

    pub fn resp(&self) -> RestResp {
        RestResp::json(self.status, &json!({"error": self.error, "status": self.status}))
    }

    /// Task errors cross the coordinator as strings.
    fn encode(&self) -> String {
        format!("desired_nodes|{}|{}", self.status, self.error)
    }

    fn decode(s: &str) -> Option<ApiError> {
        let rest = s.strip_prefix("desired_nodes|")?;
        let (status, err) = rest.split_once('|')?;
        Some(ApiError { status: status.parse().ok()?, error: serde_json::from_str(err).ok()? })
    }
}

// ------------------------------------------------------------- parsing

fn setting_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Settings as ES `Settings` holds them: flat keys, string (or string
/// list) values.
fn flatten(prefix: &str, v: &Value, out: &mut BTreeMap<String, Value>) {
    match v {
        Value::Object(o) => {
            for (k, v) in o {
                let key = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                flatten(&key, v, out);
            }
        }
        Value::Array(a) => {
            out.insert(prefix.to_string(), Value::Array(a.iter().map(|x| Value::String(setting_string(x))).collect()));
        }
        Value::Null => {
            out.insert(prefix.to_string(), Value::Null);
        }
        other => {
            out.insert(prefix.to_string(), Value::String(setting_string(other)));
        }
    }
}

/// `Settings.toXContent` without `flat_settings`: nested objects.
fn nest_settings(flat: &BTreeMap<String, Value>) -> Value {
    let mut root = Map::new();
    for (k, v) in flat {
        let parts: Vec<&str> = k.split('.').collect();
        let mut cur = &mut root;
        for p in &parts[..parts.len() - 1] {
            let e = cur.entry(p.to_string()).or_insert_with(|| Value::Object(Map::new()));
            if !e.is_object() {
                *e = Value::Object(Map::new());
            }
            cur = e.as_object_mut().expect("object");
        }
        cur.insert(parts[parts.len() - 1].to_string(), v.clone());
    }
    Value::Object(root)
}

/// A JSON number or numeric string (ES `ValueType.DOUBLE` accepts both).
fn number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

/// `Processors.of`: finite and strictly positive.
fn processors_value(v: &Value, field: &str) -> Result<f64, String> {
    let p = number(v).ok_or_else(|| format!("[{field}] failed to parse field [{field}]"))?;
    if !p.is_finite() {
        return Err(format!("Only a positive number of [{field}] are allowed and [{p}] was provided"));
    }
    if p <= 0.0 {
        return Err(format!("Only a positive number of [{field}] are allowed and [{p}] was provided"));
    }
    Ok(p)
}

/// Parse an ES byte size (`64gb`, `1tb`, `512`, `-1`) to bytes.
fn parse_bytes(s: &str) -> Option<f64> {
    let t = s.trim().to_ascii_lowercase();
    if t == "-1" {
        return Some(-1.0);
    }
    let units: [(&str, f64); 6] = [("pb", 1125899906842624.0), ("tb", 1099511627776.0), ("gb", 1073741824.0), ("mb", 1048576.0), ("kb", 1024.0), ("b", 1.0)];
    for (u, mult) in units {
        if let Some(n) = t.strip_suffix(u) {
            let n: f64 = n.trim().parse().ok()?;
            return (n >= 0.0 && n.is_finite()).then_some(n * mult);
        }
    }
    let n: f64 = t.parse().ok()?;
    (n >= 0.0 && n.fract() == 0.0).then_some(n)
}

fn same_bytes(a: &str, b: &str) -> bool {
    a == b || matches!((parse_bytes(a), parse_bytes(b)), (Some(x), Some(y)) if x == y)
}

fn byte_field(o: &Map<String, Value>, field: &str) -> Result<String, String> {
    let v = match o.get(field) {
        None => return Err(format!("Required [{field}]")),
        Some(Value::Null) => return Err(format!("[desired_node] {field} doesn't support values of type: VALUE_NULL")),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => n.to_string(),
        Some(_) => return Err(format!("[desired_node] failed to parse field [{field}]")),
    };
    if parse_bytes(&v).is_none() {
        return Err(format!("[desired_node] failed to parse field [{field}]"));
    }
    Ok(v)
}

const NODE_FIELDS: &[&str] = &["settings", "processors", "processors_range", "memory", "storage", "node_version"];

/// Parse one node object (`DesiredNode.fromXContent`).
fn parse_node(idx: usize, v: &Value) -> Result<DesiredNode, ApiError> {
    let perr = |r: String| ApiError::node_parse(idx, r);
    let o = v.as_object().ok_or_else(|| perr("[desired_node] Expected START_OBJECT".into()))?;
    for k in o.keys() {
        if !NODE_FIELDS.contains(&k.as_str()) {
            return Err(perr(format!("[desired_node] unknown field [{k}]")));
        }
    }
    let settings = match o.get("settings") {
        None => return Err(perr("Required [settings]".into())),
        Some(s @ Value::Object(_)) => {
            let mut m = BTreeMap::new();
            flatten("", s, &mut m);
            m
        }
        Some(_) => return Err(perr("[desired_node] settings doesn't support values of type: VALUE_NULL".into())),
    };
    let processors = match (o.get("processors"), o.get("processors_range")) {
        (Some(_), Some(_)) => return Err(perr("[desired_node] processors and processors_range are mutually exclusive".into())),
        (None, None) => return Err(perr("Required one of fields [processors, processors_range], but none were specified.".into())),
        (Some(p), None) => Processors::Exact(processors_value(p, "processors").map_err(perr)?),
        (None, Some(r)) => {
            let r = r.as_object().ok_or_else(|| perr("[processors_range] failed to parse field [processors_range]".into()))?;
            for k in r.keys() {
                if k != "min" && k != "max" {
                    return Err(perr(format!("[processors_range] unknown field [{k}]")));
                }
            }
            let min = match r.get("min") {
                None | Some(Value::Null) => return Err(perr("Required [min]".into())),
                Some(m) => processors_value(m, "min").map_err(perr)?,
            };
            let max = match r.get("max") {
                None | Some(Value::Null) => None,
                Some(m) => Some(processors_value(m, "max").map_err(perr)?),
            };
            if let Some(m) = max {
                if min > m {
                    return Err(perr(format!("min processors must be less than or equal to max processors and it was: min: {min} max: {m}")));
                }
            }
            Processors::Range { min, max }
        }
    };
    let memory = byte_field(o, "memory").map_err(perr)?;
    let storage = byte_field(o, "storage").map_err(perr)?;
    let node_version = match o.get("node_version") {
        None | Some(Value::Null) => None,
        Some(v) => Some(setting_string(v)),
    };
    let node = DesiredNode { settings, processors, memory, storage, node_version };
    if node.external_id().is_none() {
        return Err(ApiError::node_settings(idx, "[node.name] or [node.external_id] is missing or empty"));
    }
    if let Err(e) = node.roles() {
        return Err(ApiError::node_settings(idx, &e));
    }
    Ok(node)
}

/// Parse the request body (`UpdateDesiredNodesRequest.fromXContent`).
pub fn parse_body(body: &Value) -> Result<Vec<DesiredNode>, ApiError> {
    let o = body.as_object().ok_or_else(|| ApiError::bad("x_content_parse_exception", "[1:1] [update_desired_nodes_request] Expected START_OBJECT"))?;
    for k in o.keys() {
        if k != "nodes" {
            return Err(ApiError::bad("x_content_parse_exception", &format!("[1:2] [update_desired_nodes_request] unknown field [{k}]")));
        }
    }
    let nodes = match o.get("nodes") {
        None => return Err(ApiError::bad("x_content_parse_exception", "Required [nodes]")),
        Some(Value::Array(a)) => a,
        Some(_) => return Err(ApiError::bad("x_content_parse_exception", "[1:11] [update_desired_nodes_request] nodes doesn't support values of that type")),
    };
    nodes.iter().enumerate().map(|(i, n)| parse_node(i, n)).collect()
}

/// `UpdateDesiredNodesRequest.validate`.
pub fn validate(history_id: &str, version: i64, nodes: &[DesiredNode]) -> Result<(), ApiError> {
    let mut errs = Vec::new();
    if history_id.trim().is_empty() {
        errs.push("historyID should not be empty");
    }
    if version < 0 {
        errs.push("version must be positive");
    }
    if !nodes.iter().any(DesiredNode::has_master_role) {
        errs.push("nodes must contain at least one master node");
    }
    if errs.is_empty() {
        return Ok(());
    }
    let reason: String = "Validation Failed: ".to_string() + &errs.iter().enumerate().map(|(i, e)| format!("{}: {e};", i + 1)).collect::<String>();
    Err(ApiError::bad("action_request_validation_exception", &reason))
}

/// `DesiredNodes` constructor check.
fn check_duplicates(nodes: &[DesiredNode]) -> Result<(), ApiError> {
    let mut seen = BTreeSet::new();
    for n in nodes {
        if let Some(id) = n.external_id() {
            if !seen.insert(id.clone()) {
                return Err(ApiError::bad("illegal_argument_exception", &format!("Some nodes contain the same setting value [{id}] for [node.external_id]")));
            }
        }
    }
    Ok(())
}

/// The master-side update (`TransportUpdateDesiredNodesAction.updateDesiredNodes`).
/// Returns the definition to store and `replaced_existing_history_id`.
pub fn apply(latest: Option<&DesiredNodes>, proposed: DesiredNodes) -> Result<(DesiredNodes, bool), ApiError> {
    let Some(latest) = latest else { return Ok((proposed, false)) };
    let replaced = latest.history_id != proposed.history_id;
    if latest.equals_close(&proposed) {
        return Ok((latest.clone(), false));
    }
    if !replaced && latest.version == proposed.version {
        return Err(ApiError::bad(
            "illegal_argument_exception",
            &format!("Desired nodes with history [{}] and version [{}] already exists with a different definition", latest.history_id, latest.version),
        ));
    }
    if !replaced && proposed.version < latest.version {
        return Err(ApiError::simple(
            409,
            "version_conflict_exception",
            &format!("version [{}] has been superseded by version [{}] for history [{}]", proposed.version, latest.version, latest.history_id),
        ));
    }
    Ok((proposed, replaced))
}

// ------------------------------------------------------------ handlers

pub async fn handle(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let segs = req.segments();
    let method = req.method();
    match (method.as_str(), segs.len()) {
        ("GET", 3) if segs[2] == "_latest" => get(node),
        ("DELETE", 2) => delete(node, req).await,
        ("PUT", 4) | ("POST", 4) => put(node, req, &segs[2], &segs[3]).await,
        _ => RestResp::json(
            405,
            &json!({"error": format!("Incorrect HTTP method for uri [{}] and method [{}]", req.uri, method), "status": 405}),
        ),
    }
}

fn unrecognized(req: &RestReq, allowed: &[&str]) -> Option<RestResp> {
    let params = req.params();
    let mut bad: Vec<&String> = params.keys().filter(|k| !allowed.contains(&k.as_str()) && !COMMON_PARAMS.contains(&k.as_str())).collect();
    if bad.is_empty() {
        return None;
    }
    bad.sort();
    let parts: Vec<String> = bad
        .iter()
        .map(|b| match allowed.iter().find(|a| similarity(b, a) > 0.5) {
            Some(s) => format!("[{b}] -> did you mean [{s}]?"),
            None => format!("[{b}]"),
        })
        .collect();
    let noun = if bad.len() == 1 { "parameter" } else { "parameters" };
    Some(RestResp::error(400, "illegal_argument_exception", &format!("request [{}] contains unrecognized {noun}: {}", req.path(), parts.join(", "))))
}

/// Lucene `LevenshteinDistance.getDistance` (1 - edits / max length).
fn similarity(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        prev = cur;
    }
    let max = a.len().max(b.len()).max(1);
    1.0 - prev[b.len()] as f64 / max as f64
}

fn get(node: &Arc<ClusterNode>) -> RestResp {
    match DesiredNodes::from_state(&node.state()) {
        Some(d) => RestResp::json(200, &d.to_api_json()),
        None => RestResp::error(404, "resource_not_found_exception", "Desired nodes not found"),
    }
}

fn task_error(e: String) -> RestResp {
    match ApiError::decode(&e) {
        Some(a) => a.resp(),
        None => RestResp::error(500, "illegal_state_exception", &e),
    }
}

async fn delete(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    if let Some(r) = unrecognized(req, &["master_timeout", "timeout"]) {
        return r;
    }
    if !node.is_master() {
        return super::forward_to_master(node, req).await;
    }
    let task: elasticsearch_cluster::coordination::StateTask = Box::new(move |s: &ClusterState| {
        let mut next = s.clone();
        if next.metadata.customs.remove(CUSTOM).is_some() {
            next.metadata.version += 1;
        }
        Ok(next)
    });
    match node.coord().submit(task).await {
        // `plain_text_empty_response`: 200 with no body.
        Ok(_) => RestResp::new(200,vec![],vec![]),
        Err(e) => task_error(e),
    }
}

async fn put(node: &Arc<ClusterNode>, req: &RestReq, history_id: &str, version: &str) -> RestResp {
    if let Some(r) = unrecognized(req, &["dry_run", "master_timeout", "timeout"]) {
        return r;
    }
    let Ok(version) = version.parse::<i64>() else {
        return RestResp::error(400, "illegal_argument_exception", &format!("Failed to parse long parameter [version] with value [{version}]"));
    };
    let dry_run = match req.params().get("dry_run").map(String::as_str) {
        None | Some("false") => false,
        Some("") | Some("true") => true,
        Some(v) => return RestResp::error(400, "illegal_argument_exception", &format!("Failed to parse value [{v}] as only [true] or [false] are allowed.")),
    };
    if req.body.is_empty() {
        return RestResp::error(400, "parse_exception", "request body is required");
    }
    let Some(body) = req.json() else {
        return RestResp::error(400, "x_content_parse_exception", "[1:1] [update_desired_nodes_request] failed to parse JSON body");
    };
    let nodes = match parse_body(&body) {
        Ok(n) => n,
        Err(e) => return e.resp(),
    };
    let warn = nodes.iter().any(|n| n.node_version.is_some());
    if let Err(e) = validate(history_id, version, &nodes) {
        return with_warning(e.resp(), warn);
    }
    if !node.is_master() {
        return super::forward_to_master(node, req).await;
    }
    if let Err(e) = check_duplicates(&nodes) {
        return with_warning(e.resp(), warn);
    }
    let proposed = DesiredNodes { history_id: history_id.to_string(), version, nodes };

    let replaced = if dry_run {
        match apply(DesiredNodes::from_state(&node.state()).as_ref(), proposed) {
            Ok((_, r)) => r,
            Err(e) => return with_warning(e.resp(), warn),
        }
    } else {
        let out = Arc::new(parking_lot::Mutex::new(false));
        let out2 = out.clone();
        let task: elasticsearch_cluster::coordination::StateTask = Box::new(move |s: &ClusterState| {
            let latest = DesiredNodes::from_state(s);
            let (stored, replaced) = apply(latest.as_ref(), proposed).map_err(|e| e.encode())?;
            *out2.lock() = replaced;
            let custom = stored.to_custom(s);
            let mut next = s.clone();
            if next.metadata.customs.get(CUSTOM) != Some(&custom) {
                next.metadata.customs.insert(CUSTOM.into(), custom);
                next.metadata.version += 1;
            }
            Ok(next)
        });
        if let Err(e) = node.coord().submit(task).await {
            return with_warning(task_error(e), warn);
        }
        let r = *out.lock();
        r
    };
    with_warning(RestResp::json(200, &json!({"replaced_existing_history_id": replaced, "dry_run": dry_run})), warn)
}

fn with_warning(mut r: RestResp, warn: bool) -> RestResp {
    if warn {
        r.push_header(("Warning".into(), format!("299 Elasticsearch-{ES_VERSION}-{BUILD_HASH} \"{NODE_VERSION_DEPRECATION}\"")));
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(v: Value) -> Result<DesiredNode, ApiError> {
        parse_node(0, &v)
    }

    fn ds(h: &str, v: i64, nodes: Vec<Value>) -> DesiredNodes {
        DesiredNodes { history_id: h.into(), version: v, nodes: nodes.into_iter().map(|x| n(x).unwrap()).collect() }
    }

    fn kind(e: &ApiError) -> &str {
        e.error["type"].as_str().unwrap()
    }

    #[test]
    fn parses_and_renders_nested_settings() {
        let d = n(json!({"settings": {"node.name": "i-1"}, "processors": 8, "memory": "64gb", "storage": "128gb"})).unwrap();
        assert_eq!(d.to_json(None), json!({"settings": {"node": {"name": "i-1"}}, "processors": 8.0, "memory": "64gb", "storage": "128gb"}));
        assert_eq!(serde_json::to_string(&d.to_json(None)["processors"]).unwrap(), "8.0");
        let r = n(json!({"settings": {"node": {"external_id": "x"}}, "processors_range": {"min": 8.0}, "memory": "1gb", "storage": "1tb"})).unwrap();
        assert_eq!(r.to_json(None)["processors_range"], json!({"min": 8.0}));
        assert_eq!(r.external_id().as_deref(), Some("x"));
    }

    #[test]
    fn parse_errors_are_x_content() {
        let bad = [
            json!({"processors": 64.0, "memory": "64gb", "storage": "128gb"}),
            json!({"settings": null, "processors": 64.0, "memory": "64gb", "storage": "128gb"}),
            json!({"settings": {}, "memory": "64gb", "storage": "128gb"}),
            json!({"settings": {"node.name": "a"}, "processors": 8.0, "storage": "128gb"}),
            json!({"settings": {"node.name": "a"}, "processors": 8.0, "memory": null, "storage": "128gb"}),
            json!({"settings": {"node.name": "a"}, "processors": 8, "memory": "64gb", "storage": null}),
            json!({"settings": {"node.name": "a"}, "processors_range": {}, "memory": "64gb", "storage": "128gb"}),
            json!({"settings": {"node.name": "a"}, "processors_range": {"max": 8.0}, "memory": "64gb", "storage": "128gb"}),
            json!({"settings": {"node.name": "a"}, "processors_range": {"min": 8.0, "max": 1.0}, "memory": "64gb", "storage": "128gb"}),
            json!({"settings": {"node.name": "a"}, "processors": "NaN", "memory": "64gb", "storage": "128gb"}),
            json!({"settings": {"node.name": "a"}, "processors": "-Infinity", "memory": "64gb", "storage": "128gb"}),
            json!({"settings": {"node.name": "a"}, "processors_range": {"min": 0.0, "max": 1.0}, "memory": "64gb", "storage": "128gb"}),
            json!({"settings": {"node.name": "a"}, "processors_range": {"min": 1.0, "max": "Infinity"}, "memory": "64gb", "storage": "128gb"}),
            json!({"settings": {"node.name": "a"}, "processors": 1.0, "processors_range": {"min": 1.0, "max": 2.0}, "memory": "64gb", "storage": "128gb"}),
            json!({"settings": {"node.name": "a", "node.roles": "unknown,other"}, "processors": 8.5, "memory": "64gb", "storage": "128gb"}),
        ];
        for b in bad {
            let e = n(b.clone()).unwrap_err();
            assert_eq!((e.status, kind(&e)), (400, "x_content_parse_exception"), "{b}");
        }
    }

    #[test]
    fn missing_name_has_nested_cause() {
        for s in [json!({}), json!({"node.name": "   "}), json!({"node.external_id": ""})] {
            let e = n(json!({"settings": s, "processors": 8, "memory": "64gb", "storage": "128gb"})).unwrap_err();
            assert_eq!(e.error["caused_by"]["caused_by"]["reason"], "[node.name] or [node.external_id] is missing or empty");
            assert_eq!(e.error["root_cause"][0]["type"], "x_content_parse_exception");
        }
    }

    #[test]
    fn request_validation() {
        let one = vec![n(json!({"settings": {"node.name": "a"}, "processors": 8, "memory": "64gb", "storage": "128gb"})).unwrap()];
        assert_eq!(validate("    ", 1, &one).unwrap_err().error["reason"], "Validation Failed: 1: historyID should not be empty;");
        assert_eq!(validate("h", -1, &one).unwrap_err().error["reason"], "Validation Failed: 1: version must be positive;");
        assert_eq!(validate("h", 1, &[]).unwrap_err().error["reason"], "Validation Failed: 1: nodes must contain at least one master node;");
        let hot = vec![n(json!({"settings": {"node.name": "a", "node.roles": "data_hot"}, "processors": 8, "memory": "64gb", "storage": "128gb"})).unwrap()];
        assert_eq!(kind(&validate("h", 1, &hot).unwrap_err()), "action_request_validation_exception");
        let master = vec![n(json!({"settings": {"node.name": "a", "node.roles": ["data_hot", "master"]}, "processors": 8, "memory": "64gb", "storage": "128gb"})).unwrap()];
        assert!(validate("h", 1, &master).is_ok());
        assert!(validate("h", 0, &one).is_ok());
    }

    #[test]
    fn duplicate_external_ids() {
        let a = json!({"settings": {"node.name": "instance-000187"}, "processors": 8, "memory": "64gb", "storage": "128gb"});
        let b = json!({"settings": {"node.external_id": "instance-000187"}, "processors": 4, "memory": "64gb", "storage": "128gb"});
        let d = ds("h", 1, vec![a, b]);
        assert_eq!(check_duplicates(&d.nodes).unwrap_err().error["reason"], "Some nodes contain the same setting value [instance-000187] for [node.external_id]");
    }

    #[test]
    fn conflicts() {
        let a = json!({"settings": {"node.name": "a"}, "processors": 8, "memory": "64gb", "storage": "128gb"});
        let b = json!({"settings": {"node.name": "b"}, "processors": 16, "memory": "128gb", "storage": "1tb"});
        let v2 = ds("test", 2, vec![a.clone(), b.clone()]);
        // Lower version → 409.
        let e = apply(Some(&v2), ds("test", 1, vec![a.clone()])).unwrap_err();
        assert_eq!((e.status, e.error["reason"].as_str().unwrap()), (409, "version [1] has been superseded by version [2] for history [test]"));
        // Same version, different definition → 400.
        let e = apply(Some(&v2), ds("test", 2, vec![a.clone()])).unwrap_err();
        assert_eq!(e.error["reason"], "Desired nodes with history [test] and version [2] already exists with a different definition");
        // Same definition, different order → idempotent.
        let (s, r) = apply(Some(&v2), ds("test", 2, vec![b.clone(), a.clone()])).unwrap();
        assert_eq!((s, r), (v2.clone(), false));
        // New history replaces regardless of version.
        let (s, r) = apply(Some(&v2), ds("other", 1, vec![a.clone()])).unwrap();
        assert!(r);
        assert_eq!(s.history_id, "other");
        // Higher version.
        assert!(!apply(Some(&v2), ds("test", 3, vec![a])).unwrap().1);
    }

    #[test]
    fn custom_roundtrip_and_status() {
        let mut st = ClusterState::empty("c");
        let d = ds("h", 1, vec![json!({"settings": {"node.name": "n1", "node.roles": ["master"]}, "processors_range": {"min": 1.0, "max": 2.0}, "memory": "1gb", "storage": "1gb", "node_version": "8.19.0"})]);
        let c = d.to_custom(&st);
        assert_eq!(c["latest"]["nodes"][0]["status"], STATUS_PENDING);
        st.metadata.customs.insert(CUSTOM.into(), c);
        let back = DesiredNodes::from_state(&st).unwrap();
        assert_eq!(back, d);
        assert!(back.to_api_json()["nodes"][0].get("status").is_none());
        assert_eq!(back.to_api_json()["nodes"][0]["settings"], json!({"node": {"name": "n1", "roles": ["master"]}}));
    }

    fn member(id: &str, name: &str) -> elasticsearch_cluster::state::DiscoveryNode {
        elasticsearch_cluster::state::DiscoveryNode {
            id: id.into(),
            name: name.into(),
            ephemeral_id: format!("{id}-e"),
            transport_address: format!("127.0.0.1:{}", 9300 + id.len()),
            http_address: String::new(),
            host: "127.0.0.1".into(),
            ip: "127.0.0.1".into(),
            roles: elasticsearch_cluster::state::Role::defaults(),
            attributes: BTreeMap::new(),
            version: ES_VERSION.into(),
        }
    }

    fn statuses(st: &ClusterState) -> Vec<Value> {
        st.metadata.customs[CUSTOM]["latest"]["nodes"].as_array().unwrap().iter().map(|n| n["status"].clone()).collect()
    }

    fn node_named(name: &str) -> Value {
        json!({"settings": {"node.name": name}, "processors": 8, "memory": "64gb", "storage": "128gb"})
    }

    #[test]
    fn refresh_status_on_join_and_leave() {
        let mut st = ClusterState::empty("c");
        st.nodes.insert("id1".into(), member("id1", "n1"));
        // Stored while only n1 is a member.
        let d = ds("h", 1, vec![node_named("n1"), node_named("n2")]);
        let c = d.to_custom(&st);
        st.metadata.customs.insert(CUSTOM.into(), c);
        assert_eq!(statuses(&st), vec![json!(STATUS_ACTUALIZED), json!(STATUS_PENDING)]);
        // No membership change: no-op, no metadata version bump.
        let v = st.metadata.version;
        assert!(!refresh_status(&mut st));
        assert_eq!(st.metadata.version, v);
        // n2 joins: flips to ACTUALIZED and bumps the metadata version.
        st.nodes.insert("id2".into(), member("id2", "n2"));
        assert!(refresh_status(&mut st));
        assert_eq!(st.metadata.version, v + 1);
        assert_eq!(statuses(&st), vec![json!(STATUS_ACTUALIZED), json!(STATUS_ACTUALIZED)]);
        // Both leave: ACTUALIZED is sticky ("joined at least once").
        st.nodes.clear();
        assert!(!refresh_status(&mut st));
        assert_eq!(statuses(&st), vec![json!(STATUS_ACTUALIZED), json!(STATUS_ACTUALIZED)]);
        // A new version keeps statuses by external id (any history); new
        // ids start PENDING.
        let d2 = ds("other", 1, vec![node_named("n2"), node_named("n3")]);
        let c2 = d2.to_custom(&st);
        assert_eq!(c2["latest"]["nodes"][0]["status"], json!(STATUS_ACTUALIZED));
        assert_eq!(c2["latest"]["nodes"][1]["status"], json!(STATUS_PENDING));
        // A node with `node.external_id` matches on that, not its name.
        let mut ext = member("id3", "host-a");
        ext.attributes.insert("external_id".into(), "n3".into());
        st.metadata.customs.insert(CUSTOM.into(), c2);
        st.nodes.insert("id3".into(), ext);
        assert!(refresh_status(&mut st));
        assert_eq!(statuses(&st), vec![json!(STATUS_ACTUALIZED), json!(STATUS_ACTUALIZED)]);
        // No desired nodes: nothing to do.
        let mut empty = ClusterState::empty("c");
        assert!(!refresh_status(&mut empty));
    }

    #[test]
    fn task_error_roundtrip() {
        let e = ApiError::simple(409, "version_conflict_exception", "x");
        assert_eq!(ApiError::decode(&e.encode()), Some(e));
        assert!(ApiError::decode("other failure").is_none());
    }

    #[test]
    fn did_you_mean() {
        assert!(similarity("diy_run", "dry_run") > 0.5);
        assert!(similarity("foo", "master_timeout") < 0.5);
    }
}
