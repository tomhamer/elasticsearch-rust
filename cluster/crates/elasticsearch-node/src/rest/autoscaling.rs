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

/// One shard copy as the storage deciders see it.
#[derive(Debug, Clone)]
pub struct WorkShard {
    pub index: String,
    pub shard: u32,
    /// The node holding it (`None`: unassigned).
    pub node: Option<String>,
    /// Its store size (an unassigned copy: its primary's).
    pub bytes: u64,
    pub primary: bool,
    /// `index.routing.allocation.include._tier_preference`.
    pub tiers: Vec<String>,
}

/// One index as the frozen and proactive deciders see it.
#[derive(Debug, Clone, Default)]
pub struct WorkIndex {
    pub name: String,
    pub creation_ms: i64,
    /// `number_of_shards` and copies per shard (`number_of_replicas + 1`).
    pub shards: u64,
    pub copies: u64,
    /// `FrozenUtils.isFrozenIndex` / in the ILM `frozen` phase.
    pub frozen: bool,
    pub frozen_phase: bool,
}

/// What the master gathered for the deciders: shard copies and their
/// sizes, indices, the data nodes' disks, the data streams, the ML jobs and
/// the settings they read.
#[derive(Debug, Clone)]
pub struct Workload {
    pub shards: Vec<WorkShard>,
    pub indices: BTreeMap<String, WorkIndex>,
    pub disks: BTreeMap<String, elasticsearch_engine::autoscaling::NodeDisk>,
    /// Each data stream's backing indices, oldest first.
    pub streams: Vec<Vec<String>>,
    pub ml_jobs: Vec<elasticsearch_engine::autoscaling::MlJob>,
    pub thresholds: elasticsearch_engine::autoscaling::DiskThresholds,
    pub ml: elasticsearch_engine::autoscaling::MlSettings,
    pub now_ms: i64,
}

impl Default for Workload {
    fn default() -> Self {
        Workload { shards: Vec::new(), indices: BTreeMap::new(), disks: BTreeMap::new(), streams: Vec::new(), ml_jobs: Vec::new(),
            thresholds: Default::default(), ml: Default::default(), now_ms: 0 }
    }
}

/// Whether an unassigned copy with tier preference `tiers` belongs to a
/// policy with `roles`: its first preferred tier is one of them (`data`
/// holds every data tier).
fn wants_tier(tiers: &[String], roles: &BTreeSet<String>) -> bool {
    let first = tiers.first().map(String::as_str).unwrap_or("data_content");
    roles.contains(first) || (roles.contains("data") && first.starts_with("data_"))
}

/// Each policy's ML scale-down timer (ES `ScaleTimer`), by its roles.
fn ml_timers() -> &'static parking_lot::Mutex<BTreeMap<String, elasticsearch_engine::autoscaling::ScaleTimer>> {
    static TIMERS: std::sync::OnceLock<parking_lot::Mutex<BTreeMap<String, elasticsearch_engine::autoscaling::ScaleTimer>>> = std::sync::OnceLock::new();
    TIMERS.get_or_init(Default::default)
}

fn decision(d: elasticsearch_engine::autoscaling::Decision) -> DeciderResult {
    let required = d.capacity.map(|c| {
        let total = Resources { storage: c.total_storage.map(|b| b as i64), memory: c.total_memory.map(|b| b as i64), processors: None };
        let node = (c.node_storage.is_some() || c.node_memory.is_some())
            .then(|| Resources { storage: c.node_storage.map(|b| b as i64), memory: c.node_memory.map(|b| b as i64), processors: None });
        Capacity { total, node }
    });
    DeciderResult { required, reason: Some((d.summary, d.details)) }
}

/// One decider of policy `p` over its current `nodes`.
fn run_decider(name: &str, settings: &BTreeMap<String, String>, p: &Policy, nodes: &[CurrentNode], current: Capacity, work: &Workload) -> DeciderResult {
    use elasticsearch_engine::autoscaling as a;
    let tier: Vec<&CurrentNode> = nodes.iter().filter(|n| n.data).collect();
    let slot = |id: &str| tier.iter().position(|n| n.id == id);
    let disks: Vec<a::NodeDisk> = tier.iter().map(|n| work.disks.get(&n.id).copied()
        .unwrap_or_else(|| { let total = n.info.map(|i| i.storage).unwrap_or(0); a::NodeDisk { total, free: total } })).collect();
    let copies: Vec<a::ShardCopy> = work.shards.iter().filter_map(|s| {
        let node = match &s.node { Some(n) => Some(slot(n)?), None => { if !wants_tier(&s.tiers, &p.roles) { return None; } None } };
        Some(a::ShardCopy { id: format!("[{}][{}]", s.index, s.shard), bytes: s.bytes, node })
    }).collect();
    let current_storage = current.total.storage.map(|b| b.max(0) as u64);
    let time = |key: &str, default: u64| settings.get(key).and_then(|v| elasticsearch_engine::date::parse_time_value(v)).filter(|v| *v >= 0).map(|v| v as u64).unwrap_or(default);
    let frozen = || work.indices.values().filter(|i| i.frozen);
    match name {
        "fixed" => fixed(settings),
        "reactive_storage" => decision(a::reactive_storage(current_storage, &disks, &copies, &work.thresholds)),
        "proactive_storage" => {
            let ids: BTreeSet<&str> = tier.iter().map(|n| n.id.as_str()).collect();
            let streams: Vec<a::StreamIndices> = work.streams.iter().map(|backing| a::StreamIndices {
                indices: backing.iter().filter_map(|name| work.indices.get(name)).map(|i| {
                    let index_copies: Vec<&WorkShard> = work.shards.iter().filter(|s| s.index == i.name).collect();
                    a::BackingIndex {
                        name: i.name.clone(),
                        creation_ms: i.creation_ms,
                        bytes: index_copies.iter().map(|s| s.bytes.max(1024)).sum(),
                        copies: i.shards * i.copies,
                        on_tier: index_copies.iter().any(|s| s.node.as_deref().is_some_and(|n| ids.contains(n))),
                        off_tier: index_copies.iter().any(|s| s.node.as_deref().is_some_and(|n| !ids.contains(n))),
                    }
                }).collect(),
            }).collect();
            decision(a::proactive_storage(current_storage, &disks, &copies, &streams, time("forecast_window", 1_800_000), work.now_ms, &work.thresholds))
        }
        "frozen_existence" => {
            let names: Vec<String> = work.indices.values().filter(|i| i.frozen_phase).map(|i| i.name.clone()).collect();
            decision(a::frozen_existence(&names))
        }
        "frozen_shards" => {
            let per_shard = settings.get("memory_per_shard").and_then(|v| parse_bytes(v, "memory_per_shard").ok()).map(|b| b as u64).unwrap_or(a::DEFAULT_MEMORY_PER_SHARD);
            decision(a::frozen_shards(frozen().map(|i| i.shards * i.copies).sum(), per_shard))
        }
        "frozen_storage" => {
            let pct = settings.get("percentage").and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(a::DEFAULT_FROZEN_STORAGE_PERCENTAGE);
            // `estimateSize`: each shard's size times its copies.
            let data_set: u64 = frozen().map(|i| work.shards.iter().filter(|s| s.index == i.name && s.primary).map(|s| s.bytes * i.copies).sum::<u64>()).sum();
            decision(a::frozen_storage(data_set, pct))
        }
        "ml" => {
            let queue = |key: &str| settings.get(key).and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(0);
            let s = a::MlSettings { anomaly_queue: queue("num_anomaly_jobs_in_queue"), analytics_queue: queue("num_analytics_jobs_in_queue"),
                down_scale_delay_ms: time("down_scale_delay", 3_600_000), ..work.ml };
            let memory: Vec<u64> = nodes.iter().filter_map(|n| n.info.map(|i| i.memory)).collect();
            let input = a::MlInput { jobs: &work.ml_jobs, node_memory: &memory,
                current_tier_memory: current.total.memory.unwrap_or(0).max(0) as u64, current_node_memory: current.node.and_then(|n| n.memory).unwrap_or(0).max(0) as u64 };
            let key = p.roles.iter().cloned().collect::<Vec<_>>().join(",");
            let mut timers = ml_timers().lock();
            decision(a::ml(&input, &s, timers.entry(key).or_default(), work.now_ms))
        }
        // Validation accepts only the deciders above.
        other => DeciderResult { required: None, reason: Some((format!("unknown decider [{other}]"), json!({}))) },
    }
}

/// `calculateForPolicy` given the nodes whose role set equals the policy's.
pub fn calculate_policy(p: &Policy, nodes: &[CurrentNode], work: &Workload) -> Value {
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
        with_defaults(p).iter().map(|(d, s)| (d.clone(), run_decider(d, s, p, nodes, current, work))).collect();
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
    let work = match workload(node, &state).await {
        Ok(w) => w,
        Err(e) => return RestResp::from_es_error(&e),
    };
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
        out.insert(name.clone(), calculate_policy(p, &current, &work));
    }
    RestResp::json(200, &json!({"policies": out}))
}

/// The deciders' inputs, gathered by the master (shard sizes and disks from
/// every data node, the ML jobs and data streams from its engine).
async fn workload(node: &Arc<ClusterNode>, state: &ClusterState) -> Result<Workload, elasticsearch_engine::error::EsError> {
    use elasticsearch_engine::autoscaling as a;
    let stats = super::gather_shard_doc_stats(node, state).await?;
    let size_of = |c: &elasticsearch_cluster::state::ShardRouting| c.allocation_id().and_then(|id| stats.get(id)).map(|s| s.store_bytes).unwrap_or(0);
    let setting = |key: &str| state.metadata.setting(key).and_then(Value::as_str).map(str::to_string)
        .or_else(|| node.config.settings.get(key).and_then(Value::as_str).map(str::to_string));
    let mut work = Workload {
        thresholds: a::DiskThresholds::from_settings(&setting),
        ml: a::MlSettings {
            max_machine_memory_percent: setting("xpack.ml.max_machine_memory_percent").and_then(|v| v.trim().parse().ok()).unwrap_or(30),
            use_auto: setting("xpack.ml.use_auto_machine_memory_percent").as_deref() == Some("true"),
            max_ml_node_size: setting("xpack.ml.max_ml_node_size").and_then(|v| parse_bytes(&v, "xpack.ml.max_ml_node_size").ok()).map(|b| b.max(0) as u64).unwrap_or(0),
            ..Default::default()
        },
        now_ms: elasticsearch_engine::date::now_millis(),
        ..Default::default()
    };
    for (name, meta) in &state.metadata.indices {
        let tier = meta.setting("index.routing.allocation.include._tier_preference");
        work.indices.insert(name.clone(), WorkIndex {
            name: name.clone(),
            creation_ms: meta.creation_date,
            shards: meta.number_of_shards as u64,
            copies: meta.number_of_replicas as u64 + 1,
            frozen: a::is_frozen_index(tier),
            frozen_phase: meta.lifecycle.as_ref().is_some_and(|l| l["ilm"]["phase"].as_str() == Some("frozen")),
        });
    }
    for (index, irt) in &state.routing_table.indices {
        let tiers: Vec<String> = state.metadata.indices.get(index).and_then(|m| m.setting("index.routing.allocation.include._tier_preference"))
            .map(|s| s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect())
            .unwrap_or_else(|| vec!["data_content".into()]);
        for t in &irt.shards {
            let primary_bytes = t.copies.iter().filter(|c| c.primary && c.node_id.is_some()).map(&size_of).max().unwrap_or(0);
            for c in &t.copies {
                let bytes = match size_of(c) { 0 if !c.primary || c.node_id.is_none() => primary_bytes, b => b };
                work.shards.push(WorkShard { index: index.clone(), shard: t.shard, node: c.node_id.clone(), bytes, primary: c.primary, tiers: tiers.clone() });
            }
        }
    }
    for n in state.data_nodes() {
        if let Ok(Some(d)) = crate::rest::nodes::fetch_disk(node, &n.id).await {
            work.disks.insert(n.id.clone(), a::NodeDisk { total: d.total_bytes, free: d.free_bytes });
        }
    }
    work.streams = node.engine.data_streams.read().values().map(|ds| ds.indices.clone()).collect();
    work.ml_jobs = a::ml_jobs(&node.engine);
    Ok(work)
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
        let v = calculate_policy(&p, &[], &Workload::default());
        assert_eq!(v["required_capacity"]["total"]["storage"], 1337);
        assert_eq!(v["current_capacity"]["total"]["processors"], 0.0);
        assert_eq!(v["current_capacity"]["node"]["storage"], 0);
        assert_eq!(v["current_nodes"], json!([]));
        assert_eq!(v["deciders"]["fixed"]["reason_details"]["nodes"], 1);
        let unknown = Policy { roles: roles(&["warp_drive"]), deciders: BTreeMap::new() };
        assert_eq!(calculate_policy(&unknown, &[], &Workload::default())["deciders"], json!({"_unknown_role": {}}));
        // #417: the default storage deciders of a data policy compute its
        // required capacity.
        let data = Policy { roles: roles(&["data"]), deciders: BTreeMap::new() };
        let info = NodeInfo { memory: 100, processors: 4.0, storage: 1000 };
        let nodes = vec![
            CurrentNode { name: "b".into(), id: "2".into(), data: true, info: Some(info) },
            CurrentNode { name: "a".into(), id: "1".into(), data: true, info: Some(info) },
        ];
        let v = calculate_policy(&data, &nodes, &Workload::default());
        assert_eq!(v["required_capacity"]["total"]["storage"], 2000, "{v}");
        assert_eq!(v["current_capacity"]["total"]["storage"], 2000);
        assert_eq!(v["current_capacity"]["node"]["storage"], 1000);
        assert_eq!(v["current_capacity"]["total"]["processors"], 8.0);
        assert_eq!(v["current_nodes"], json!([{"name": "a"}, {"name": "b"}]));
        assert_eq!(v["deciders"]["reactive_storage"]["reason_summary"], "storage ok");
        assert_eq!(v["deciders"]["proactive_storage"]["reason_summary"], "storage ok");
    }

    /// #417 closure: an ordinary data policy, the same with a fixed
    /// override, and an ML policy with healthy nodes and pending work all
    /// return a required capacity from their deciders.
    #[test]
    fn data_and_ml_policies_calculate_their_required_capacity() {
        use elasticsearch_engine::autoscaling::{MlJob, NodeDisk};
        const GB: u64 = 1 << 30;
        let info = NodeInfo { memory: 8 * GB, processors: 4.0, storage: 100 * GB };
        let nodes = vec![CurrentNode { name: "d1".into(), id: "d1".into(), data: true, info: Some(info) },
            CurrentNode { name: "d2".into(), id: "d2".into(), data: true, info: Some(info) }];
        let mut work = Workload::default();
        work.disks.insert("d1".into(), NodeDisk { total: 100 * GB, free: 40 * GB });
        work.disks.insert("d2".into(), NodeDisk { total: 100 * GB, free: 40 * GB });
        let shard = |node: Option<&str>, bytes: u64| WorkShard { index: "logs".into(), shard: 0, node: node.map(str::to_string), bytes, primary: node.is_some(), tiers: vec!["data_hot".into()] };
        work.shards = vec![shard(Some("d1"), 50 * GB), shard(None, 100 * GB)];
        let data = Policy { roles: roles(&["data_hot"]), deciders: BTreeMap::new() };
        let v = calculate_policy(&data, &nodes, &work);
        // 200 GB now; the 100 GB replica fits no node (40 GB free each).
        assert_eq!(v["required_capacity"]["total"]["storage"], json!(300 * GB), "{v}");
        assert!(v["deciders"]["reactive_storage"]["reason_summary"].as_str().unwrap().starts_with("not enough storage"), "{v}");
        // A fixed override is combined, not suppressed by the defaults.
        let fixed_data = Policy { roles: roles(&["data_hot"]), deciders: [("fixed".to_string(), [("storage".to_string(), "1tb".to_string()), ("memory".to_string(), "1gb".to_string())].into_iter().collect())].into_iter().collect() };
        let v = calculate_policy(&fixed_data, &nodes, &work);
        assert_eq!(v["required_capacity"]["total"]["storage"], json!(1u64 << 40), "{v}");
        assert_eq!(v["required_capacity"]["total"]["memory"], json!(GB), "{v}");
        // ML: a running job and one waiting past the queue allowance.
        work.ml_jobs = vec![MlJob { bytes: 330 << 20, assigned: true, analytics: false }, MlJob { bytes: 630 << 20, assigned: false, analytics: false }];
        let ml_nodes = vec![CurrentNode { name: "m1".into(), id: "m1".into(), data: false, info: Some(NodeInfo { memory: 2 * GB, processors: 2.0, storage: 0 }) }];
        let ml = Policy { roles: roles(&["ml"]), deciders: BTreeMap::new() };
        let v = calculate_policy(&ml, &ml_nodes, &work);
        // ES `checkForScaleUp`: current ML memory (2gb x 30% less the 30mb
        // code overhead) less the free memory left, plus the unassignable
        // job: 960mb native, as a node size at 30% with the overhead.
        let wanted = ((100.0 / 30.0) * (990u64 << 20) as f64).ceil() as u64;
        assert_eq!(v["required_capacity"]["total"]["memory"], json!(wanted), "{v}");
        assert!(v["deciders"]["ml"]["reason_summary"].as_str().unwrap().contains("scale up"), "{v}");
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
