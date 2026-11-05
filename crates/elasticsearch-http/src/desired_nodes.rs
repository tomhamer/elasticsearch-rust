//! `_internal/desired_nodes`: the orchestrator-facing description of the nodes a
//! cluster is meant to have (history id + monotonically increasing version).
//!
//! The single node keeps the latest definition in memory per cluster; ES keeps it
//! in the cluster state, so it is equally non-durable across a full restart of a
//! test cluster (and survives for the life of the node here).

use crate::{Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::{Node, ES_VERSION};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, OnceLock};

struct Desired {
    history_id: String,
    version: i64,
    /// Rendered nodes, in request order.
    nodes: Vec<Value>,
}

fn store() -> &'static Mutex<HashMap<String, Desired>> {
    static S: OnceLock<Mutex<HashMap<String, Desired>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashMap::new()))
}

const ROLES: &[&str] = &["master", "data", "data_content", "data_hot", "data_warm", "data_cold", "data_frozen", "ingest", "ml", "remote_cluster_client", "transform", "voting_only", "search", "index"];
const NODE_VERSION_DEPRECATION: &str = "[version removal] Specifying node_version in desired nodes requests is deprecated.";

pub fn handle(node: &Node, req: &Req) -> R {
    match (req.seg(2), &req.method) {
        ("", &Method::DELETE) => {
            store().lock().unwrap().remove(&node.cluster_uuid);
            Ok(Resp::text(String::new()))
        }
        ("_latest", &Method::GET) => match store().lock().unwrap().get(&node.cluster_uuid) {
            Some(d) => Ok(Resp::ok(json!({"history_id": d.history_id, "version": d.version, "nodes": d.nodes}))),
            None => Err(EsError::resource_not_found("Desired nodes not found")),
        },
        (history, &Method::PUT) if !history.is_empty() && !req.seg(3).is_empty() => update(node, req, history, req.seg(3)),
        _ => crate::method_not_allowed(req, &["GET", "PUT", "DELETE"]),
    }
}

fn xcontent(reason: impl Into<String>) -> EsError {
    EsError::x_content_parse(reason)
}

/// ES's strict object parser reports a failed node as nested parse exceptions.
fn node_error(inner: impl Into<String>) -> EsError {
    xcontent("[1:1] [desired_node] failed to parse field [nodes]")
        .caused_by(EsError::illegal_argument("[desired_node] failed to parse object").caused_by(EsError::illegal_argument(inner)))
}

fn positive_finite(v: &Value, what: &str) -> Result<f64, EsError> {
    match v.as_f64() {
        Some(f) if f.is_finite() && f > 0.0 => Ok(f),
        Some(_) => Err(node_error(format!("[{what}] must be positive and finite"))),
        None => Err(node_error(format!("[{what}] must be a number"))),
    }
}

fn byte_size(v: &Value, what: &str) -> Result<String, EsError> {
    let s = v.as_str().ok_or_else(|| node_error(format!("[{what}] is required")))?;
    let t = s.trim().to_ascii_lowercase();
    let digits: String = t.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
    let unit = &t[digits.len()..];
    if digits.is_empty() || digits.parse::<f64>().is_err() || !matches!(unit, "" | "b" | "kb" | "k" | "mb" | "m" | "gb" | "g" | "tb" | "t" | "pb" | "p") {
        return Err(node_error(format!("failed to parse setting [{what}] with value [{s}] as a size in bytes")));
    }
    Ok(s.to_string())
}

struct Parsed {
    external_id: String,
    id_setting: &'static str,
    master: bool,
    rendered: Value,
    node_version: bool,
}

fn parse_node(n: &Value) -> Result<Parsed, EsError> {
    let o = n.as_object().ok_or_else(|| xcontent("[1:1] [desired_node] failed to parse field [nodes]"))?;
    for k in o.keys() {
        if !matches!(k.as_str(), "settings" | "processors" | "processors_range" | "memory" | "storage" | "node_version") {
            return Err(xcontent(format!("[1:1] [desired_node] unknown field [{k}]")));
        }
    }
    let settings = match o.get("settings") {
        Some(Value::Object(_)) => o["settings"].clone(),
        _ => return Err(node_error("[settings] is required")),
    };
    let flat = elasticsearch_engine::settings::flatten(&settings);
    let text = |k: &str| flat.get(k).and_then(Value::as_str).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    let (external_id, id_setting) = match (text("node.external_id"), text("node.name")) {
        (Some(e), _) => (e, "node.external_id"),
        (None, Some(n)) => (n, "node.name"),
        _ => return Err(node_error("[node.name] or [node.external_id] is missing or empty")),
    };
    let mut master = true;
    if let Some(roles) = flat.get("node.roles") {
        let list: Vec<String> = match roles {
            Value::String(s) => s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect(),
            Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
            _ => vec![],
        };
        for r in &list {
            if !ROLES.contains(&r.as_str()) {
                return Err(node_error(format!("unknown role [{r}]")));
            }
        }
        master = list.iter().any(|r| r == "master");
    }
    let mut rendered = Map::new();
    rendered.insert("settings".into(), elasticsearch_engine::settings::unflatten(&flat));
    match (o.get("processors"), o.get("processors_range")) {
        (Some(_), Some(_)) => return Err(node_error("Only one of [processors] and [processors_range] is allowed")),
        (Some(p), None) => {
            rendered.insert("processors".into(), json!(positive_finite(p, "processors")?));
        }
        (None, Some(Value::Object(r))) => {
            for k in r.keys() {
                if k != "min" && k != "max" {
                    return Err(node_error(format!("[processors_range] unknown field [{k}]")));
                }
            }
            let min = positive_finite(r.get("min").ok_or_else(|| node_error("[min] is required"))?, "min")?;
            let mut range = Map::new();
            range.insert("min".into(), json!(min));
            if let Some(mx) = r.get("max") {
                let max = positive_finite(mx, "max")?;
                if min > max {
                    return Err(node_error("min processors must be less than or equal to max"));
                }
                range.insert("max".into(), json!(max));
            }
            rendered.insert("processors_range".into(), Value::Object(range));
        }
        _ => return Err(node_error("[processors] or [processors_range] is required")),
    }
    rendered.insert("memory".into(), json!(byte_size(o.get("memory").unwrap_or(&Value::Null), "memory")?));
    rendered.insert("storage".into(), json!(byte_size(o.get("storage").unwrap_or(&Value::Null), "storage")?));
    let node_version = match o.get("node_version") {
        Some(Value::String(v)) if !v.trim().is_empty() => {
            rendered.insert("node_version".into(), json!(v));
            true
        }
        Some(_) => return Err(node_error("[node_version] must not be empty")),
        None => false,
    };
    Ok(Parsed { external_id, id_setting, master, rendered: Value::Object(rendered), node_version })
}

/// Order-insensitive identity of a node definition.
fn canonical(nodes: &[Value]) -> BTreeMap<String, String> {
    nodes
        .iter()
        .map(|n| {
            let flat = elasticsearch_engine::settings::flatten(&n["settings"]);
            let id = flat.get("node.external_id").or_else(|| flat.get("node.name")).and_then(Value::as_str).unwrap_or("").to_string();
            (id, n.to_string())
        })
        .collect()
}

fn update(node: &Node, req: &Req, history_id: &str, version: &str) -> R {
    let version: i64 = version.parse().map_err(|_| EsError::illegal_argument(format!("Failed to parse long parameter [version] with value [{version}]")))?;
    let dry_run = match req.param("dry_run") {
        None => false,
        Some("true") | Some("") => true,
        Some("false") => false,
        Some(o) => return Err(EsError::illegal_argument(format!("Failed to parse value [{o}] as only [true] or [false] are allowed."))),
    };
    let body = req.json()?;
    let body = body.as_object().ok_or_else(|| xcontent("[1:1] [desired_nodes] unexpected token"))?;
    for k in body.keys() {
        if k != "nodes" {
            return Err(xcontent(format!("[1:1] [desired_nodes] unknown field [{k}]")));
        }
    }
    let nodes_in = body.get("nodes").and_then(Value::as_array).ok_or_else(|| xcontent("[1:1] [desired_nodes] failed to parse field [nodes]"))?;
    let parsed: Vec<Parsed> = nodes_in.iter().map(parse_node).collect::<Result<_, _>>()?;

    let mut errors: Vec<&str> = Vec::new();
    if history_id.trim().is_empty() {
        errors.push("historyID should not be empty");
    }
    if version <= 0 {
        errors.push("version must be positive");
    }
    if !parsed.iter().any(|p| p.master) {
        errors.push("nodes must contain at least one master node");
    }
    if !errors.is_empty() {
        return Err(EsError::action_validation(&errors));
    }

    let mut seen: HashMap<&str, &'static str> = HashMap::new();
    for p in &parsed {
        if seen.insert(p.external_id.as_str(), p.id_setting).is_some() {
            return Err(EsError::illegal_argument(format!("Some nodes contain the same setting value [{}] for [{}]", p.external_id, p.id_setting)));
        }
    }
    let rendered: Vec<Value> = parsed.iter().map(|p| p.rendered.clone()).collect();

    let mut st = store().lock().unwrap();
    let mut replaced = false;
    if let Some(cur) = st.get(&node.cluster_uuid) {
        if cur.history_id != history_id {
            replaced = true;
        } else if version < cur.version {
            return Err(EsError::new(409, "version_conflict_exception", format!("version [{version}] has been superseded by version [{}] for history [{history_id}]", cur.version)));
        } else if version == cur.version && canonical(&cur.nodes) != canonical(&rendered) {
            return Err(EsError::illegal_argument(format!("Desired nodes with history [{history_id}] and version [{version}] already exists with a different definition")));
        }
    }
    if !dry_run {
        st.insert(node.cluster_uuid.clone(), Desired { history_id: history_id.to_string(), version, nodes: rendered });
    }
    let mut resp = Resp::ok(json!({"replaced_existing_history_id": replaced, "dry_run": dry_run}));
    if parsed.iter().any(|p| p.node_version) {
        resp.headers.push(("Warning", format!("299 Elasticsearch-{ES_VERSION}-{} \"{NODE_VERSION_DEPRECATION}\"", crate::cluster::BUILD_HASH)));
    }
    Ok(resp)
}
