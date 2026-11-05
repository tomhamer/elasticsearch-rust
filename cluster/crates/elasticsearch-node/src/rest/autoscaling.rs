//! Autoscaling APIs (x-pack `autoscaling` plugin, ES 8.19):
//! `PUT|GET|DELETE /_autoscaling/policy/{name}` and
//! `GET /_autoscaling/capacity`.
//!
//! Policies live in the `autoscaling` metadata custom as
//! `{"policies": {name: {"policy": {"roles": [..], "deciders": {d: {k: "v"}}}}}}`.
//! Policy validation follows `PolicyValidator` /
//! `AutoscalingCalculateCapacityService.validate`; capacity follows
//! `AutoscalingCalculateCapacityService.calculate`.

use super::node_shutdown::{check_params, loc_str, method_not_allowed, scan_fields, task_error, token_name, ApiError, TimeValue};
use crate::front::{RestReq, RestResp};
use crate::node::ClusterNode;
use elasticsearch_cluster::state::{ClusterState, Role};
use elasticsearch_transport::RemoteError;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

pub const CUSTOM: &str = "autoscaling";
pub const ACTION_NODE_INFO: &str = "internal:autoscaling/node_info";
const PARSER: &str = "put_autocaling_policy_request";
const INVALID_NAME: &str = "name must not contain the following characters [ , \", *, \\, <, |, ,, >, /, ?]";

// ------------------------------------------------------------ policy model

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Policy {
    pub roles: BTreeSet<String>,
    pub deciders: BTreeMap<String, BTreeMap<String, String>>,
}

impl Policy {
    fn to_json(&self) -> Value {
        let deciders: Map<String, Value> =
            self.deciders.iter().map(|(d, s)| (d.clone(), Value::Object(s.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect()))).collect();
        json!({"roles": self.roles.iter().collect::<Vec<_>>(), "deciders": deciders})
    }

    fn from_json(v: &Value) -> Policy {
        let roles = v["roles"].as_array().map(|a| a.iter().filter_map(|r| r.as_str().map(String::from)).collect()).unwrap_or_default();
        let deciders = v["deciders"]
            .as_object()
            .map(|m| {
                m.iter()
                    .map(|(d, s)| {
                        let settings = s.as_object().map(|o| o.iter().map(|(k, v)| (k.clone(), v.as_str().map(String::from).unwrap_or_else(|| v.to_string()))).collect()).unwrap_or_default();
                        (d.clone(), settings)
                    })
                    .collect()
            })
            .unwrap_or_default();
        Policy { roles, deciders }
    }
}

/// `PutAutoscalingPolicyAction.Request`: `None` means "not given".
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PutRequest {
    pub roles: Option<BTreeSet<String>>,
    pub deciders: Option<BTreeMap<String, BTreeMap<String, String>>>,
}

pub fn policies(state: &ClusterState) -> BTreeMap<String, Policy> {
    state.metadata.customs.get(CUSTOM).and_then(|c| c["policies"].as_object()).map(|m| m.iter().map(|(n, p)| (n.clone(), Policy::from_json(&p["policy"]))).collect()).unwrap_or_default()
}

fn store(state: &mut ClusterState, all: &BTreeMap<String, Policy>) {
    let p: Map<String, Value> = all.iter().map(|(n, p)| (n.clone(), json!({"policy": p.to_json()}))).collect();
    state.metadata.customs.insert(CUSTOM.into(), json!({"policies": p}));
}

fn role_names() -> BTreeSet<&'static str> {
    Role::ALL.iter().map(|r| r.name()).collect()
}

/// Java `SortedSet.toString()`.
fn set_str(s: &BTreeSet<String>) -> String {
    format!("[{}]", s.iter().cloned().collect::<Vec<_>>().join(", "))
}

// ------------------------------------------------------------ body parsing

fn bad(reason: String) -> ApiError {
    ApiError::simple(400, "x_content_parse_exception", &reason)
}

/// Settings value to its string form (`Settings.fromXContent`).
fn scalar_str(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn flatten(prefix: &str, v: &Value, out: &mut BTreeMap<String, String>) {
    match v {
        Value::Object(m) => {
            for (k, x) in m {
                let key = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                flatten(&key, x, out);
            }
        }
        Value::Array(a) => {
            let items: Vec<String> = a.iter().map(|x| scalar_str(x).unwrap_or_else(|| x.to_string())).collect();
            out.insert(prefix.to_string(), items.join(","));
        }
        Value::Null => {}
        other => {
            out.insert(prefix.to_string(), scalar_str(other).unwrap_or_default());
        }
    }
}

pub fn parse_body(body: &[u8]) -> Result<PutRequest, ApiError> {
    if body.iter().all(u8::is_ascii_whitespace) {
        return Err(ApiError::simple(400, "parse_exception", "request body is required"));
    }
    if serde_json::from_slice::<Value>(body).is_err() {
        return Err(bad(format!("[1:1] [{PARSER}] failed to parse JSON body")));
    }
    let fields = scan_fields(body).map_err(|(tok, loc)| bad(format!("{} [{PARSER}] Expected START_OBJECT but was: {tok}", loc_str(loc))))?;
    let mut r = PutRequest::default();
    for f in &fields {
        let wrong = || bad(format!("{} [{PARSER}] {} doesn't support values of type: {}", loc_str(f.value_loc), f.name, token_name(f.value_tok)));
        match f.name.as_str() {
            "roles" => {
                let items: Vec<&Value> = match &f.value {
                    Value::Array(a) => a.iter().collect(),
                    Value::String(_) => vec![&f.value],
                    _ => return Err(wrong()),
                };
                let mut roles = BTreeSet::new();
                for v in items {
                    let Some(s) = scalar_str(v) else {
                        return Err(bad(format!("{} [{PARSER}] failed to parse field [roles]", loc_str(f.value_loc))));
                    };
                    roles.insert(s);
                }
                r.roles = Some(roles);
            }
            "deciders" => {
                let Value::Object(m) = &f.value else { return Err(wrong()) };
                let mut deciders = BTreeMap::new();
                for (name, v) in m {
                    if !v.is_object() {
                        return Err(bad(format!("{} [{PARSER}] failed to parse field [deciders]", loc_str(f.value_loc))));
                    }
                    let mut s = BTreeMap::new();
                    flatten("", v, &mut s);
                    deciders.insert(name.clone(), s);
                }
                r.deciders = Some(deciders);
            }
            other => return Err(bad(format!("{} [{PARSER}] unknown field [{other}]", loc_str(f.name_loc)))),
        }
    }
    Ok(r)
}

/// `Strings.validFileName`.
fn valid_name(name: &str) -> bool {
    !name.chars().any(|c| matches!(c, ' ' | '"' | '*' | '\\' | '<' | '|' | ',' | '>' | '/' | '?'))
}

/// `PutAutoscalingPolicyAction.Request.validate`.
pub fn validate(name: &str, r: &PutRequest) -> Result<(), ApiError> {
    let known = role_names();
    let mut errors: Vec<String> = r.roles.iter().flatten().filter(|x| !known.contains(x.as_str())).cloned().collect();
    if !valid_name(name) {
        errors.push(INVALID_NAME.into());
    }
    if errors.is_empty() {
        return Ok(());
    }
    let msg: String = errors.iter().enumerate().map(|(i, e)| format!("{}: {e};", i + 1)).collect();
    Err(ApiError::simple(400, "action_request_validation_exception", &format!("Validation Failed: {msg}")))
}

// ------------------------------------------------------------ deciders

#[derive(Clone, Copy)]
enum Kind {
    Bytes,
    Int { min: i64 },
    Double { min: Option<f64> },
    Time,
}

struct Decider {
    name: &'static str,
    /// `None` = every role (and applies to a policy with no roles).
    roles: Option<&'static [&'static str]>,
    default_on: bool,
    settings: &'static [(&'static str, Kind)],
}

const DECIDERS: &[Decider] = &[
    Decider {
        name: "fixed",
        roles: None,
        default_on: false,
        settings: &[("storage", Kind::Bytes), ("memory", Kind::Bytes), ("nodes", Kind::Int { min: 0 }), ("processors", Kind::Double { min: Some(0.0) })],
    },
    Decider { name: "frozen_existence", roles: Some(&["data_frozen"]), default_on: true, settings: &[] },
    Decider { name: "frozen_shards", roles: Some(&["data_frozen"]), default_on: true, settings: &[("memory_per_shard", Kind::Bytes)] },
    Decider { name: "frozen_storage", roles: Some(&["data_frozen"]), default_on: true, settings: &[("percentage", Kind::Double { min: Some(0.0) })] },
    Decider {
        name: "ml",
        roles: Some(&["ml"]),
        default_on: true,
        settings: &[("num_anomaly_jobs_in_queue", Kind::Int { min: 0 }), ("num_analytics_jobs_in_queue", Kind::Int { min: 0 }), ("down_scale_delay", Kind::Time)],
    },
    Decider { name: "proactive_storage", roles: Some(&["data", "data_hot"]), default_on: true, settings: &[("forecast_window", Kind::Time)] },
    Decider { name: "reactive_storage", roles: Some(&["data", "data_content", "data_hot", "data_warm", "data_cold"]), default_on: true, settings: &[] },
];

fn decider(name: &str) -> Option<&'static Decider> {
    DECIDERS.iter().find(|d| d.name == name)
}

fn applies(d: &Decider, roles: &BTreeSet<String>) -> bool {
    match d.roles {
        None => true,
        Some(_) if roles.is_empty() => false,
        Some(rs) => rs.iter().any(|r| roles.contains(*r)),
    }
}

fn iae(reason: String) -> ApiError {
    ApiError::simple(400, "illegal_argument_exception", &reason)
}

/// `ByteSizeValue.parseBytesSizeValue`.
pub fn parse_bytes(v: &str, key: &str) -> Result<i64, String> {
    let s = v.trim().to_ascii_lowercase();
    if s == "-1" {
        return Ok(-1);
    }
    if s == "0" {
        return Ok(0);
    }
    let units: [(&str, i64); 11] = [
        ("pb", 1 << 50),
        ("tb", 1 << 40),
        ("gb", 1 << 30),
        ("mb", 1 << 20),
        ("kb", 1 << 10),
        ("p", 1 << 50),
        ("t", 1 << 40),
        ("g", 1 << 30),
        ("m", 1 << 20),
        ("k", 1 << 10),
        ("b", 1),
    ];
    let Some((num, mult)) = units.iter().find_map(|(u, m)| s.strip_suffix(u).map(|n| (n.trim(), *m))) else {
        return Err(format!("failed to parse setting [{key}] with value [{v}] as a size in bytes: unit is missing or unrecognized"));
    };
    if let Ok(n) = num.parse::<i64>() {
        return n.checked_mul(mult).filter(|b| *b >= -1).ok_or_else(|| format!("failed to parse setting [{key}] with value [{v}] as a size in bytes"));
    }
    match num.parse::<f64>() {
        Ok(f) if f.is_finite() && f >= 0.0 => Ok((f * mult as f64) as i64),
        _ => Err(format!("failed to parse [{v}]")),
    }
}

/// `ByteSizeValue.toString()` (largest unit, one truncated decimal).
pub fn humanize_bytes(b: i64) -> String {
    let units = [("pb", 1i64 << 50), ("tb", 1 << 40), ("gb", 1 << 30), ("mb", 1 << 20), ("kb", 1 << 10)];
    let (suffix, div) = units.iter().find(|(_, d)| b >= *d).copied().unwrap_or(("b", 1));
    if div == 1 {
        return format!("{b}b");
    }
    let value = b as f64 / div as f64;
    let p = java_double(value);
    match p.split_once('.') {
        Some((int, frac)) if !frac.starts_with('0') => format!("{int}.{}{suffix}", &frac[..1]),
        Some((int, _)) => format!("{int}{suffix}"),
        None => format!("{p}{suffix}"),
    }
}

/// `Double.toString` for ordinary magnitudes.
fn java_double(f: f64) -> String {
    if f.fract() == 0.0 && f.abs() < 1e7 {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

fn check_setting(key: &str, value: &str, kind: Kind) -> Result<(), String> {
    match kind {
        Kind::Bytes => parse_bytes(value, key).map(|_| ()),
        Kind::Int { min } => {
            let n: i64 = value.trim().parse().map_err(|_| format!("Failed to parse value [{value}] for setting [{key}]"))?;
            if n < min {
                return Err(format!("Failed to parse value [{value}] for setting [{key}] must be >= {min}"));
            }
            Ok(())
        }
        Kind::Double { min } => {
            let f: f64 = value.trim().parse().map_err(|_| format!("Failed to parse value [{value}] for setting [{key}]"))?;
            if let Some(m) = min {
                if f < m {
                    return Err(format!("Failed to parse value [{value}] for setting [{key}] must be >= {}", java_double(m)));
                }
            }
            Ok(())
        }
        Kind::Time => TimeValue::parse(value, key).map(|_| ()).map_err(|e| e["reason"].as_str().unwrap_or("").to_string()),
    }
}

/// `PolicyValidator` / `AutoscalingCalculateCapacityService.validate`.
pub fn validate_policy(name: &str, p: &Policy) -> Result<(), ApiError> {
    for (dname, settings) in &p.deciders {
        let Some(d) = decider(dname) else { return Err(iae(format!("unknown decider [{dname}]"))) };
        if !applies(d, &p.roles) {
            return Err(iae(format!("decider [{dname}] not applicable to policy with roles [ {}]", set_str(&p.roles))));
        }
        for (k, v) in settings {
            let Some((_, kind)) = d.settings.iter().find(|(s, _)| s == k) else {
                return Err(iae(format!("unknown setting [{k}] for decider [{dname}]")));
            };
            check_setting(k, v, *kind).map_err(iae)?;
        }
    }
    if with_defaults(p).is_empty() {
        return Err(iae(format!("no default nor user configured deciders for policy [{name}] with roles [{}]", set_str(&p.roles))));
    }
    Ok(())
}

fn with_defaults(p: &Policy) -> BTreeMap<String, BTreeMap<String, String>> {
    let mut d = p.deciders.clone();
    for dec in DECIDERS.iter().filter(|d| d.default_on && applies(d, &p.roles)) {
        d.entry(dec.name.to_string()).or_default();
    }
    d
}

/// `TransportPutAutoscalingPolicyAction.putAutoscalingPolicy` minus the
/// state write: `Ok(None)` is a no-op.
pub fn merge(existing: Option<&Policy>, name: &str, r: &PutRequest) -> Result<Option<Policy>, ApiError> {
    let updated = match existing {
        None => {
            let Some(roles) = r.roles.clone() else {
                return Err(iae(format!("new policy {name} with no roles defined, must provide empty list for no roles")));
            };
            Policy { roles, deciders: r.deciders.clone().unwrap_or_default() }
        }
        Some(e) => Policy { roles: r.roles.clone().unwrap_or_else(|| e.roles.clone()), deciders: r.deciders.clone().unwrap_or_else(|| e.deciders.clone()) },
    };
    validate_policy(name, &updated)?;
    if existing == Some(&updated) {
        return Ok(None);
    }
    Ok(Some(updated))
}

// ------------------------------------------------------------ capacity

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Resources {
    pub storage: Option<i64>,
    pub memory: Option<i64>,
    pub processors: Option<f64>,
}

impl Resources {
    const ZERO: Resources = Resources { storage: Some(0), memory: Some(0), processors: Some(0.0) };

    fn to_json(self) -> Value {
        let mut m = Map::new();
        if let Some(s) = self.storage {
            m.insert("storage".into(), json!(s));
        }
        if let Some(s) = self.memory {
            m.insert("memory".into(), json!(s));
        }
        if let Some(p) = self.processors {
            m.insert("processors".into(), json!(p));
        }
        Value::Object(m)
    }

    fn combine(a: Resources, b: Resources, fi: fn(i64, i64) -> i64, ff: fn(f64, f64) -> f64) -> Resources {
        fn opt<T: Copy>(x: Option<T>, y: Option<T>, f: impl Fn(T, T) -> T) -> Option<T> {
            match (x, y) {
                (Some(x), Some(y)) => Some(f(x, y)),
                (x, None) => x,
                (None, y) => y,
            }
        }
        Resources { storage: opt(a.storage, b.storage, fi), memory: opt(a.memory, b.memory, fi), processors: opt(a.processors, b.processors, ff) }
    }

    fn sum(a: Resources, b: Resources) -> Resources {
        Self::combine(a, b, |x, y| x + y, |x, y| x + y)
    }

    fn max(a: Resources, b: Resources) -> Resources {
        Self::combine(a, b, i64::max, f64::max)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Capacity {
    pub total: Resources,
    pub node: Option<Resources>,
}

impl Capacity {
    const ZERO: Capacity = Capacity { total: Resources::ZERO, node: Some(Resources::ZERO) };

    fn to_json(self) -> Value {
        let mut m = Map::new();
        if let Some(n) = self.node {
            m.insert("node".into(), n.to_json());
        }
        m.insert("total".into(), self.total.to_json());
        Value::Object(m)
    }

    /// `AutoscalingCapacity.upperBound`.
    fn upper_bound(a: Capacity, b: Capacity) -> Capacity {
        let node = match (a.node, b.node) {
            (Some(x), Some(y)) => Some(Resources::max(x, y)),
            (x, None) => x,
            (None, y) => y,
        };
        Capacity { total: Resources::max(a.total, b.total), node }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeciderResult {
    pub required: Option<Capacity>,
    pub reason: Option<(String, Value)>,
}

impl DeciderResult {
    fn to_json(&self) -> Value {
        let mut m = Map::new();
        if let Some(c) = self.required {
            m.insert("required_capacity".into(), c.to_json());
        }
        if let Some((summary, details)) = &self.reason {
            m.insert("reason_summary".into(), json!(summary));
            m.insert("reason_details".into(), details.clone());
        }
        Value::Object(m)
    }
}

/// `FixedAutoscalingDeciderService.scale`.
pub fn fixed(settings: &BTreeMap<String, String>) -> DeciderResult {
    let bytes = |k: &str| settings.get(k).and_then(|v| parse_bytes(v, k).ok());
    let storage = bytes("storage");
    let memory = bytes("memory");
    let processors = settings.get("processors").and_then(|v| v.trim().parse::<f64>().ok());
    let nodes: i64 = settings.get("nodes").and_then(|v| v.trim().parse().ok()).unwrap_or(1);
    let required = if storage.is_none() && memory.is_none() && processors.is_none() {
        None
    } else {
        let node = Resources { storage, memory, processors };
        let total = Resources { storage: storage.map(|s| s * nodes), memory: memory.map(|m| m * nodes), processors: processors.map(|p| p * nodes as f64) };
        Some(Capacity { total, node: Some(node) })
    };
    let hb = |b: Option<i64>| b.map(humanize_bytes);
    let summary = format!(
        "fixed storage [{}] memory [{}] processors [{}] nodes [{nodes}]",
        hb(storage).unwrap_or_else(|| "null".into()),
        hb(memory).unwrap_or_else(|| "null".into()),
        processors.map(java_double).unwrap_or_else(|| "null".into()),
    );
    let mut details = Map::new();
    details.insert("storage".into(), hb(storage).map(Value::String).unwrap_or(Value::Null));
    details.insert("memory".into(), hb(memory).map(Value::String).unwrap_or(Value::Null));
    details.insert("nodes".into(), json!(nodes));
    if let Some(p) = processors {
        details.insert("processors".into(), json!(p));
    }
    DeciderResult { required, reason: Some((summary, Value::Object(details))) }
}

/// What this node contributes to `current_capacity`
/// (`AutoscalingNodeInfo` + the ClusterInfo disk total).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct NodeInfo {
    pub memory: u64,
    pub processors: f64,
    pub storage: u64,
}

pub fn local_info(node: &ClusterNode) -> NodeInfo {
    let mem = crate::stats::mem_info();
    let disk = crate::stats::disk_usage(&node.config.data_path);
    let procs = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    NodeInfo { memory: mem.total_bytes, processors: procs as f64, storage: disk.total_bytes }
}

pub fn register(node: &Arc<ClusterNode>) {
    let n = node.clone();
    node.transport.register_handler(ACTION_NODE_INFO, move |_: ()| {
        let n = n.clone();
        async move { Ok::<_, RemoteError>(local_info(&n)) }
    });
}

async fn node_info(node: &Arc<ClusterNode>, id: &str) -> Option<NodeInfo> {
    if id == node.local.id {
        return Some(local_info(node));
    }
    let addr = node.address_of(id)?;
    node.transport.send_request::<_, NodeInfo>(addr, ACTION_NODE_INFO, &(), Duration::from_secs(10)).await.ok()
}

/// A current node as the capacity calculation sees it.
#[derive(Debug, Clone)]
pub struct CurrentNode {
    pub name: String,
    pub id: String,
    pub data: bool,
    pub info: Option<NodeInfo>,
}

/// `calculateForPolicy` given the nodes whose role set equals the policy's.
pub fn calculate_policy(p: &Policy, nodes: &[CurrentNode]) -> Value {
    let known = role_names();
    if p.roles.iter().any(|r| !known.contains(r.as_str())) {
        return json!({"current_capacity": Capacity::ZERO.to_json(), "current_nodes": [], "deciders": {"_unknown_role": {}}});
    }
    let current = nodes
        .iter()
        .map(|n| {
            let r = Resources {
                storage: Some(if n.data { n.info.map(|i| i.storage as i64).unwrap_or(0) } else { 0 }),
                memory: Some(n.info.map(|i| i.memory as i64).unwrap_or(0)),
                processors: n.info.map(|i| i.processors),
            };
            Capacity { total: r, node: Some(r) }
        })
        .reduce(|a, b| Capacity { total: Resources::sum(a.total, b.total), node: Some(Resources::max(a.node.unwrap_or_default(), b.node.unwrap_or_default())) })
        .unwrap_or(Capacity::ZERO);
    let results: BTreeMap<String, DeciderResult> =
        with_defaults(p).iter().map(|(d, s)| (d.clone(), if d == "fixed" { fixed(s) } else { DeciderResult { required: None, reason: None } })).collect();
    let required = results.values().map(|r| r.required).collect::<Option<Vec<_>>>().and_then(|v| v.into_iter().reduce(Capacity::upper_bound));
    let mut sorted: Vec<&CurrentNode> = nodes.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name).then(a.id.cmp(&b.id)));
    let mut m = Map::new();
    if let Some(r) = required {
        m.insert("required_capacity".into(), r.to_json());
    }
    m.insert("current_capacity".into(), current.to_json());
    m.insert("current_nodes".into(), Value::Array(sorted.iter().map(|n| json!({"name": n.name})).collect()));
    m.insert("deciders".into(), Value::Object(results.iter().map(|(k, v)| (k.clone(), v.to_json())).collect()));
    Value::Object(m)
}

// ------------------------------------------------------------ handlers

pub async fn handle(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let segs = req.segments();
    let m = req.method();
    match (segs.get(1).map(String::as_str), segs.get(2).cloned(), segs.len()) {
        (Some("policy"), Some(name), 3) => match m.as_str() {
            "PUT" => put(node, req, name).await,
            "GET" => get(node, req, name).await,
            "DELETE" => delete(node, req, name).await,
            _ => method_not_allowed(req, "GET, PUT, DELETE"),
        },
        (Some("capacity"), None, 2) => match m.as_str() {
            "GET" => capacity(node, req).await,
            _ => method_not_allowed(req, "GET"),
        },
        _ => RestResp::error(400, "illegal_argument_exception", &format!("no handler found for uri [{}] and method [{}]", req.uri, req.method)),
    }
}

async fn put(node: &Arc<ClusterNode>, req: &RestReq, name: String) -> RestResp {
    if let Some(r) = check_params(req, &["master_timeout", "timeout"]) {
        return r;
    }
    let r = match parse_body(&req.body) {
        Ok(r) => r,
        Err(e) => return e.resp(),
    };
    if let Err(e) = validate(&name, &r) {
        return e.resp();
    }
    if !node.is_master() {
        return super::forward_to_master(node, req).await;
    }
    let ctx = node.master.clone();
    let task: elasticsearch_cluster::coordination::StateTask = Box::new(move |s: &ClusterState| {
        let mut all = policies(s);
        match merge(all.get(&name), &name, &r) {
            Err(e) => Err(e.encode()),
            Ok(None) => Ok(s.clone()),
            Ok(Some(p)) => {
                all.insert(name.clone(), p);
                let mut next = s.clone();
                store(&mut next, &all);
                next.metadata.version += 1;
                Ok(ctx.reroute(&next))
            }
        }
    });
    match node.coord().submit(task).await {
        Ok(_) => RestResp::json(200, &json!({"acknowledged": true})),
        Err(e) => task_error(e),
    }
}

fn not_found(name: &str) -> ApiError {
    ApiError::simple(404, "resource_not_found_exception", &format!("autoscaling policy with name [{name}] does not exist"))
}

/// `Regex.simpleMatch` (`*` wildcards).
fn simple_match(pattern: &str, s: &str) -> bool {
    match pattern.split_once('*') {
        None => pattern == s,
        Some((head, rest)) => {
            let Some(tail) = s.strip_prefix(head) else { return false };
            if rest.is_empty() {
                return true;
            }
            (0..=tail.len()).filter(|i| tail.is_char_boundary(*i)).any(|i| simple_match(rest, &tail[i..]))
        }
    }
}

async fn delete(node: &Arc<ClusterNode>, req: &RestReq, name: String) -> RestResp {
    if let Some(r) = check_params(req, &["master_timeout", "timeout"]) {
        return r;
    }
    if !node.is_master() {
        return super::forward_to_master(node, req).await;
    }
    let ctx = node.master.clone();
    let task: elasticsearch_cluster::coordination::StateTask = Box::new(move |s: &ClusterState| {
        let mut all = policies(s);
        let before = all.len();
        if name.contains('*') {
            all.retain(|k, _| !simple_match(&name, k));
        } else if all.remove(&name).is_none() {
            return Err(not_found(&name).encode());
        }
        if all.len() == before {
            return Ok(s.clone());
        }
        let mut next = s.clone();
        store(&mut next, &all);
        next.metadata.version += 1;
        Ok(ctx.reroute(&next))
    });
    match node.coord().submit(task).await {
        Ok(_) => RestResp::json(200, &json!({"acknowledged": true})),
        Err(e) => task_error(e),
    }
}

async fn get(node: &Arc<ClusterNode>, req: &RestReq, name: String) -> RestResp {
    if let Some(r) = check_params(req, &["master_timeout"]) {
        return r;
    }
    match policies(&node.state()).get(&name) {
        Some(p) => RestResp::json(200, &p.to_json()),
        None => not_found(&name).resp(),
    }
}

async fn capacity(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    if let Some(r) = check_params(req, &["master_timeout"]) {
        return r;
    }
    if !node.is_master() {
        return super::forward_to_master(node, req).await;
    }
    let state = node.state();
    let all = policies(&state);
    let mut info_cache: BTreeMap<String, Option<NodeInfo>> = BTreeMap::new();
    let mut out = Map::new();
    for (name, p) in &all {
        let mut current = Vec::new();
        for (id, n) in &state.nodes {
            let roles: BTreeSet<String> = n.roles.iter().map(|r| r.name().to_string()).collect();
            if roles != p.roles {
                continue;
            }
            if !info_cache.contains_key(id) {
                let i = node_info(node, id).await;
                info_cache.insert(id.clone(), i);
            }
            current.push(CurrentNode { name: n.name.clone(), id: id.clone(), data: n.is_data(), info: info_cache[id] });
        }
        out.insert(name.clone(), calculate_policy(p, &current));
    }
    RestResp::json(200, &json!({"policies": out}))
}

// ------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;

    fn reason(e: &ApiError) -> String {
        e.error["reason"].as_str().unwrap().to_string()
    }

    fn roles(r: &[&str]) -> BTreeSet<String> {
        r.iter().map(|s| s.to_string()).collect()
    }

    fn put_req(body: &str) -> PutRequest {
        parse_body(body.as_bytes()).unwrap()
    }

    #[test]
    fn parse_roles_and_deciders() {
        let r = put_req(r#"{"roles":["ml","data"],"deciders":{"fixed":{"storage":"1kb","nodes":3}}}"#);
        assert_eq!(r.roles, Some(roles(&["data", "ml"])));
        let d = r.deciders.unwrap();
        assert_eq!(d["fixed"]["storage"], "1kb");
        assert_eq!(d["fixed"]["nodes"], "3");
        let r = put_req(r#"{"roles":"data"}"#);
        assert_eq!(r.roles, Some(roles(&["data"])));
        assert_eq!(r.deciders, None);
        let r = put_req("{}");
        assert_eq!(r, PutRequest::default());
    }

    #[test]
    fn parse_errors() {
        let e = parse_body(b"").unwrap_err();
        assert_eq!(e.error["type"], "parse_exception");
        let e = parse_body(br#"{"policy":{}}"#).unwrap_err();
        assert_eq!(reason(&e), "[1:2] [put_autocaling_policy_request] unknown field [policy]");
        let e = parse_body(br#"{"deciders":1}"#).unwrap_err();
        assert!(reason(&e).contains("deciders doesn't support values of type: VALUE_NUMBER"), "{}", reason(&e));
        assert_eq!(e.status, 400);
    }

    #[test]
    fn request_validation() {
        let r = PutRequest { roles: Some(roles(&["non-existing", "data"])), deciders: None };
        let e = validate("p", &r).unwrap_err();
        assert_eq!(e.error["type"], "action_request_validation_exception");
        assert_eq!(reason(&e), "Validation Failed: 1: non-existing;");
        let e = validate("a,b", &PutRequest::default()).unwrap_err();
        assert_eq!(reason(&e), format!("Validation Failed: 1: {INVALID_NAME};"));
        assert!(INVALID_NAME.ends_with(r#"[ , ", *, \, <, |, ,, >, /, ?]"#));
        assert!(validate("ok", &PutRequest { roles: Some(roles(&["data_content"])), deciders: None }).is_ok());
    }

    #[test]
    fn merge_rules() {
        let e = merge(None, "p", &put_req("{}")).unwrap_err();
        assert_eq!(reason(&e), "new policy p with no roles defined, must provide empty list for no roles");
        let e = merge(None, "p", &put_req(r#"{"roles":[]}"#)).unwrap_err();
        assert_eq!(reason(&e), "no default nor user configured deciders for policy [p] with roles [[]]");
        let p = merge(None, "p", &put_req(r#"{"roles":[],"deciders":{"fixed":{}}}"#)).unwrap().unwrap();
        assert!(p.roles.is_empty());
        let created = merge(None, "p", &put_req(r#"{"roles":["data"]}"#)).unwrap().unwrap();
        // Update keeps the roles when only deciders are given.
        let upd = merge(Some(&created), "p", &put_req(r#"{"deciders":{"fixed":{}}}"#)).unwrap().unwrap();
        assert_eq!(upd.roles, roles(&["data"]));
        assert!(upd.deciders.contains_key("fixed"));
        assert_eq!(merge(Some(&upd), "p", &put_req(r#"{"roles":["data"]}"#)).unwrap(), None);
        let e = merge(None, "p", &put_req(r#"{"roles":["data"],"deciders":{"nope":{}}}"#)).unwrap_err();
        assert_eq!(reason(&e), "unknown decider [nope]");
        let e = merge(None, "p", &put_req(r#"{"roles":[],"deciders":{"fixed":{"unknown":"x"}}}"#)).unwrap_err();
        assert_eq!(reason(&e), "unknown setting [unknown] for decider [fixed]");
        let e = merge(None, "p", &put_req(r#"{"roles":["master"],"deciders":{"ml":{}}}"#)).unwrap_err();
        assert_eq!(reason(&e), "decider [ml] not applicable to policy with roles [ [master]]");
        let e = merge(None, "p", &put_req(r#"{"roles":["master"],"deciders":{"fixed":{"storage":"12"}}}"#)).unwrap_err();
        assert_eq!(reason(&e), "failed to parse setting [storage] with value [12] as a size in bytes: unit is missing or unrecognized");
        let e = merge(None, "p", &put_req(r#"{"roles":["ml"],"deciders":{"ml":{"num_anomaly_jobs_in_queue":-1}}}"#)).unwrap_err();
        assert_eq!(e.error["type"], "illegal_argument_exception");
    }

    #[test]
    fn byte_sizes() {
        assert_eq!(parse_bytes("1337b", "k").unwrap(), 1337);
        assert_eq!(parse_bytes("2KB", "k").unwrap(), 2048);
        assert_eq!(parse_bytes("1g", "k").unwrap(), 1 << 30);
        assert_eq!(parse_bytes("0", "k").unwrap(), 0);
        assert!(parse_bytes("10", "k").is_err());
        assert_eq!(humanize_bytes(1337), "1.3kb");
        assert_eq!(humanize_bytes(7331), "7.1kb");
        assert_eq!(humanize_bytes(1024), "1kb");
        assert_eq!(humanize_bytes(1000), "1000b");
        assert_eq!(humanize_bytes(3 << 30), "3gb");
    }

    #[test]
    fn fixed_decider_matches_yaml() {
        let s: BTreeMap<String, String> = [("storage", "1337b"), ("memory", "7331b"), ("nodes", "10")].iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        let r = fixed(&s);
        let c = r.required.unwrap();
        assert_eq!(c.total.storage, Some(13370));
        assert_eq!(c.total.memory, Some(73310));
        assert_eq!(c.node.unwrap().memory, Some(7331));
        assert_eq!(r.reason.as_ref().unwrap().0, "fixed storage [1.3kb] memory [7.1kb] processors [null] nodes [10]");
        let mut s2 = s.clone();
        s2.insert("processors".into(), "2".into());
        let r = fixed(&s2);
        assert_eq!(r.reason.as_ref().unwrap().0, "fixed storage [1.3kb] memory [7.1kb] processors [2.0] nodes [10]");
        assert_eq!(r.required.unwrap().total.processors, Some(20.0));
        assert_eq!(fixed(&BTreeMap::new()).required, None);
    }

    #[test]
    fn capacity_shapes() {
        let p = Policy { roles: roles(&["voting_only"]), deciders: [("fixed".to_string(), [("storage".to_string(), "1337b".to_string())].into_iter().collect())].into_iter().collect() };
        let v = calculate_policy(&p, &[]);
        assert_eq!(v["required_capacity"]["total"]["storage"], 1337);
        assert_eq!(v["current_capacity"]["total"]["processors"], 0.0);
        assert_eq!(v["current_capacity"]["node"]["storage"], 0);
        assert_eq!(v["current_nodes"], json!([]));
        assert_eq!(v["deciders"]["fixed"]["reason_details"]["nodes"], 1);
        let unknown = Policy { roles: roles(&["warp_drive"]), deciders: BTreeMap::new() };
        assert_eq!(calculate_policy(&unknown, &[])["deciders"], json!({"_unknown_role": {}}));
        // Default deciders without a computed capacity make the policy's
        // required capacity unknown.
        let data = Policy { roles: roles(&["data"]), deciders: BTreeMap::new() };
        let info = NodeInfo { memory: 100, processors: 4.0, storage: 1000 };
        let nodes = vec![
            CurrentNode { name: "b".into(), id: "2".into(), data: true, info: Some(info) },
            CurrentNode { name: "a".into(), id: "1".into(), data: true, info: Some(info) },
        ];
        let v = calculate_policy(&data, &nodes);
        assert!(v.get("required_capacity").is_none());
        assert_eq!(v["current_capacity"]["total"]["storage"], 2000);
        assert_eq!(v["current_capacity"]["node"]["storage"], 1000);
        assert_eq!(v["current_capacity"]["total"]["processors"], 8.0);
        assert_eq!(v["current_nodes"], json!([{"name": "a"}, {"name": "b"}]));
        assert_eq!(v["deciders"]["reactive_storage"], json!({}));
        assert_eq!(v["deciders"]["proactive_storage"], json!({}));
    }

    #[test]
    fn wildcard_match() {
        assert!(simple_match("*", "x"));
        assert!(simple_match("my_*_delete*", "my_policy_delete_1"));
        assert!(!simple_match("my_autoscaling_policy_delete*", "my_autoscaling_policy_keep"));
    }

    #[test]
    fn storage_roundtrip() {
        let mut s = ClusterState::empty("test");
        let p = Policy { roles: roles(&["data", "ml"]), deciders: [("fixed".to_string(), BTreeMap::new())].into_iter().collect() };
        store(&mut s, &[("p".to_string(), p.clone())].into_iter().collect());
        assert_eq!(policies(&s)["p"], p);
        assert_eq!(p.to_json(), json!({"roles": ["data", "ml"], "deciders": {"fixed": {}}}));
    }
}
