//! Autoscaling APIs (`_autoscaling/policy/{name}`, `_autoscaling/capacity`).
//!
//! Policies (roles + decider settings) live in the extension state (ES:
//! `AutoscalingMetadata` in cluster state). The capacity report runs the
//! `fixed` decider exactly as ES does (required capacity = the configured
//! per-node resources times `nodes`); the storage, shard and ML deciders ES
//! enables by default for data / frozen / ml roles only report a reason, as
//! a single node has nothing to scale.

use crate::{method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::{EsError, Result};
use elasticsearch_engine::mapping::wildcard_match;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search_ext::ext;
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
use std::sync::Arc;

/// `DiscoveryNodeRole.roles()` names.
const KNOWN_ROLES: &[&str] = &["data", "data_cold", "data_content", "data_frozen", "data_hot", "data_warm", "index", "ingest", "master", "ml", "remote_cluster_client", "search", "transform", "voting_only"];
/// Roles of this node (what `_nodes` reports).
const NODE_ROLES: &[&str] = &["data", "data_cold", "data_content", "data_frozen", "data_hot", "data_warm", "ingest", "master", "ml", "remote_cluster_client", "transform"];
const FIXED_SETTINGS: &[&str] = &["storage", "memory", "processors", "nodes"];

/// The deciders, the roles each applies to (`fixed` applies to any policy)
/// and whether ES enables it by default for a policy with such roles.
const DECIDERS: &[(&str, &[&str], bool)] = &[
    ("fixed", &[], false),
    ("reactive_storage", &["data", "data_content", "data_hot", "data_warm", "data_cold"], true),
    ("proactive_storage", &["data", "data_content", "data_hot", "data_warm", "data_cold"], true),
    ("frozen_storage", &["data_frozen"], true),
    ("frozen_shards", &["data_frozen"], true),
    ("frozen_existence", &["data_frozen"], true),
    ("ml", &["ml"], true),
];

fn ia(msg: impl Into<String>) -> EsError {
    EsError::illegal_argument(msg)
}

fn not_found(name: &str) -> EsError {
    EsError::resource_not_found(format!("autoscaling policy with name [{name}] does not exist"))
}

/// `ByteSizeValue.parseBytesSizeValue`.
fn parse_bytes(setting: &str, s: &str) -> Result<i64> {
    let t = s.trim().to_ascii_lowercase();
    let err = || ia(format!("failed to parse setting [{setting}] with value [{s}] as a size in bytes: unit is missing or unrecognized"));
    let units: &[(&str, i64)] = &[("pb", 1 << 50), ("tb", 1 << 40), ("gb", 1 << 30), ("mb", 1 << 20), ("kb", 1 << 10), ("b", 1)];
    for (u, mul) in units {
        if let Some(num) = t.strip_suffix(u) {
            let n: f64 = num.trim().parse().map_err(|_| err())?;
            return Ok((n * *mul as f64) as i64);
        }
    }
    if t == "-1" || t == "0" {
        return t.parse().map_err(|_| err());
    }
    Err(err())
}

/// `ByteSizeValue.toString()` (one truncated decimal, `.0` dropped).
fn human_bytes(bytes: i64) -> String {
    let units: &[(&str, f64)] = &[("pb", (1u64 << 50) as f64), ("tb", (1u64 << 40) as f64), ("gb", (1u64 << 30) as f64), ("mb", (1u64 << 20) as f64), ("kb", 1024.0)];
    for (u, div) in units {
        if bytes as f64 >= *div {
            let v = (bytes as f64 / div * 10.0).floor() / 10.0;
            return if v.fract() == 0.0 { format!("{}{u}", v as i64) } else { format!("{v:.1}{u}") };
        }
    }
    format!("{bytes}b")
}

/// Parse and validate a decider's settings (`Settings.fromXContent` + the
/// decider's setting definitions).
fn validate_decider(name: &str, settings: &Value, roles: &BTreeSet<String>) -> Result<Value> {
    let Some((_, decider_roles, _)) = DECIDERS.iter().find(|(n, _, _)| *n == name) else {
        return Err(ia(format!("unknown decider [{name}]")));
    };
    if !decider_roles.is_empty() && !decider_roles.iter().any(|r| roles.contains(*r)) {
        return Err(ia(format!("decider [{name}] not applicable to policy with roles [ {} ]", roles.iter().cloned().collect::<Vec<_>>().join(", "))));
    }
    let o = settings.as_object().ok_or_else(|| EsError::x_content_parse(format!("[1:2] [autoscaling_policy] failed to parse field [deciders]: [{name}] must be an object")))?;
    if name == "fixed" {
        for (k, v) in o {
            if !FIXED_SETTINGS.contains(&k.as_str()) {
                return Err(ia(format!("unknown setting [{k}] for decider [{name}]")));
            }
            let text = match v {
                Value::String(s) => s.clone(),
                Value::Number(n) => n.to_string(),
                other => return Err(ia(format!("Failed to parse value [{other}] for setting [{k}]"))),
            };
            match k.as_str() {
                "storage" | "memory" => {
                    parse_bytes(k, &text)?;
                }
                "processors" => {
                    let p: f64 = text.parse().map_err(|_| ia(format!("Failed to parse value [{text}] for setting [processors]")))?;
                    if p <= 0.0 {
                        return Err(ia(format!("Failed to parse value [{text}] for setting [processors] must be > 0")));
                    }
                }
                "nodes" => {
                    let n: i64 = text.parse().map_err(|_| ia(format!("Failed to parse value [{text}] for setting [nodes]")))?;
                    if n < 0 {
                        return Err(ia(format!("Failed to parse value [{text}] for setting [nodes] must be >= 0")));
                    }
                }
                _ => {}
            }
        }
    } else if !o.is_empty() {
        return Err(ia(format!("unknown setting [{}] for decider [{name}]", o.keys().next().unwrap())));
    }
    // ES stores settings flattened as strings.
    Ok(Value::Object(o.iter().map(|(k, v)| (k.clone(), json!(match v { Value::String(s) => s.clone(), other => other.to_string() }))).collect()))
}

fn default_deciders(roles: &BTreeSet<String>) -> Map<String, Value> {
    DECIDERS.iter().filter(|(_, rs, on)| *on && rs.iter().any(|r| roles.contains(*r))).map(|(n, _, _)| (n.to_string(), json!({}))).collect()
}

fn put_policy(node: &Node, req: &Req, name: &str) -> R {
    let body = req.json()?;
    let o = body.as_object().ok_or_else(|| EsError::new(400, "parse_exception", "request body is required"))?;
    for k in o.keys() {
        if !matches!(k.as_str(), "roles" | "deciders") {
            return Err(EsError::x_content_parse(format!("[1:2] [autoscaling_policy] unknown field [{k}]")));
        }
    }
    let roles: Option<BTreeSet<String>> = match o.get("roles") {
        None | Some(Value::Null) => None,
        Some(Value::Array(a)) => {
            let mut set = BTreeSet::new();
            for r in a {
                let r = r.as_str().ok_or_else(|| EsError::x_content_parse("[1:2] [autoscaling_policy] failed to parse field [roles]"))?;
                if !KNOWN_ROLES.contains(&r) {
                    return Err(ia(format!("unknown node roles [{r}]")));
                }
                set.insert(r.to_string());
            }
            Some(set)
        }
        Some(other) => return Err(EsError::x_content_parse(format!("[1:2] [autoscaling_policy] roles doesn't support values of type: {}", crate::rollup::token_name(other)))),
    };
    let st = ext(node);
    let existing = st.autoscaling_policies.read().get(name).cloned();
    let roles = match (roles, &existing) {
        (Some(r), _) => r,
        (None, Some(e)) => e["roles"].as_array().into_iter().flatten().filter_map(Value::as_str).map(str::to_string).collect(),
        (None, None) => return Err(ia(format!("new policy {name} with no roles defined, must provide empty list for no roles"))),
    };
    let deciders: Map<String, Value> = match o.get("deciders") {
        Some(Value::Object(d)) => {
            let mut out = Map::new();
            for (dn, ds) in d {
                out.insert(dn.clone(), validate_decider(dn, ds, &roles)?);
            }
            out
        }
        Some(Value::Null) | None => match &existing {
            Some(e) => e["deciders"].as_object().cloned().unwrap_or_default(),
            None => {
                let d = default_deciders(&roles);
                if d.is_empty() {
                    return Err(ia(format!("no default deciders for policy [{name}] with roles [ {} ]", roles.iter().cloned().collect::<Vec<_>>().join(", "))));
                }
                d
            }
        },
        Some(other) => return Err(EsError::x_content_parse(format!("[1:2] [autoscaling_policy] deciders doesn't support values of type: {}", crate::rollup::token_name(other)))),
    };
    if deciders.is_empty() {
        return Err(ia(format!("no default deciders for policy [{name}] with roles [ {} ]", roles.iter().cloned().collect::<Vec<_>>().join(", "))));
    }
    st.autoscaling_policies.write().insert(name.to_string(), json!({"roles": roles.iter().collect::<Vec<_>>(), "deciders": deciders}));
    st.persist()?;
    Ok(Resp::ack())
}

fn get_policy(node: &Node, name: &str) -> R {
    let st = ext(node);
    let p = st.autoscaling_policies.read().get(name).cloned().ok_or_else(|| not_found(name))?;
    Ok(Resp::ok(p))
}

fn delete_policy(node: &Node, name: &str) -> R {
    let st = ext(node);
    {
        let mut g = st.autoscaling_policies.write();
        if name.contains('*') {
            g.retain(|k, _| !wildcard_match(name, k));
        } else if g.remove(name).is_none() {
            return Err(not_found(name));
        }
    }
    st.persist()?;
    Ok(Resp::ack())
}

fn capacity_json(storage: Option<i64>, memory: Option<i64>, processors: Option<f64>) -> Value {
    let mut m = Map::new();
    if let Some(s) = storage {
        m.insert("storage".into(), json!(s));
    }
    if let Some(mem) = memory {
        m.insert("memory".into(), json!(mem));
    }
    if let Some(p) = processors {
        m.insert("processors".into(), json!(p));
    }
    Value::Object(m)
}

/// `FixedAutoscalingDeciderService.scale`.
fn fixed_result(settings: &Value) -> (Option<Value>, Value) {
    let get = |k: &str| settings.get(k).and_then(Value::as_str);
    let storage = get("storage").and_then(|s| parse_bytes("storage", s).ok());
    let memory = get("memory").and_then(|s| parse_bytes("memory", s).ok());
    let processors = get("processors").and_then(|s| s.parse::<f64>().ok());
    let nodes = get("nodes").and_then(|s| s.parse::<i64>().ok()).unwrap_or(1);
    let required = if storage.is_some() || memory.is_some() || processors.is_some() {
        Some(json!({
            "node": capacity_json(storage, memory, processors),
            "total": capacity_json(storage.map(|s| s * nodes), memory.map(|m| m * nodes), processors.map(|p| p * nodes as f64)),
        }))
    } else {
        None
    };
    let fmt_bytes = |b: Option<i64>| b.map(human_bytes).unwrap_or_else(|| "null".into());
    let summary = format!(
        "fixed storage [{}] memory [{}] processors [{}] nodes [{nodes}]",
        fmt_bytes(storage),
        fmt_bytes(memory),
        processors.map(|p| format!("{p:?}")).unwrap_or_else(|| "null".into())
    );
    let mut details = json!({"storage": storage.map(human_bytes), "memory": memory.map(human_bytes), "nodes": nodes});
    if let Some(p) = processors {
        details["processors"] = json!(p);
    }
    (required, json!({"reason_summary": summary, "reason_details": details}))
}

fn get_capacity(node: &Node) -> R {
    let st = ext(node);
    let policies = st.autoscaling_policies.read().clone();
    let sys = crate::sysinfo::sys();
    let fs = crate::sysinfo::fs(&node.data);
    let node_roles: BTreeSet<&str> = NODE_ROLES.iter().copied().collect();
    let mut out = Map::new();
    for (name, p) in &policies {
        let roles: BTreeSet<&str> = p["roles"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
        let matches_node = roles == node_roles;
        let (storage, memory, processors) = if matches_node { (fs.total as i64, sys.total_mem as i64, sys.cpus as f64) } else { (0, 0, 0.0) };
        let current = json!({"node": capacity_json(Some(storage), Some(memory), Some(processors)), "total": capacity_json(Some(storage), Some(memory), Some(processors))});
        let current_nodes: Vec<Value> = if matches_node { vec![json!({"name": node.name})] } else { Vec::new() };
        let mut deciders = Map::new();
        let mut required: Option<Value> = None;
        let mut all_required = true;
        for (dn, ds) in p["deciders"].as_object().into_iter().flatten() {
            let (req, mut res) = if dn == "fixed" {
                fixed_result(ds)
            } else {
                (None, json!({"reason_summary": format!("{dn} decider has nothing to scale on a single node")}))
            };
            match req {
                Some(r) => {
                    required = Some(match required {
                        None => r.clone(),
                        Some(cur) => upper_bound(&cur, &r),
                    });
                    res["required_capacity"] = r;
                }
                None => all_required = false,
            }
            deciders.insert(dn.clone(), res);
        }
        let mut entry = Map::new();
        if all_required {
            if let Some(r) = required {
                entry.insert("required_capacity".into(), r);
            }
        }
        entry.insert("current_capacity".into(), current);
        entry.insert("current_nodes".into(), Value::Array(current_nodes));
        entry.insert("deciders".into(), Value::Object(deciders));
        out.insert(name.clone(), Value::Object(entry));
    }
    Ok(Resp::ok(json!({"policies": out})))
}

/// `AutoscalingCapacity.upperBound`: the per-resource maximum.
fn upper_bound(a: &Value, b: &Value) -> Value {
    let max = |k: &str| -> Value {
        let mut m = Map::new();
        for res in ["storage", "memory", "processors"] {
            match (a[k].get(res).and_then(Value::as_f64), b[k].get(res).and_then(Value::as_f64)) {
                (Some(x), Some(y)) => {
                    let v = x.max(y);
                    m.insert(res.into(), if res == "processors" { json!(v) } else { json!(v as i64) });
                }
                _ => {}
            }
        }
        Value::Object(m)
    };
    json!({"node": max("node"), "total": max("total")})
}

/// `/_autoscaling/...`
pub async fn handle(node: Arc<Node>, req: Req) -> R {
    let m = req.method.clone();
    match (req.seg(1), req.segs.len()) {
        ("policy", 3) => {
            let name = req.seg(2).to_string();
            crate::util::check_params(&req, &["master_timeout", "timeout"])?;
            match m {
                Method::PUT => crate::blocking(move || put_policy(&node, &req, &name)).await,
                Method::GET => crate::blocking(move || get_policy(&node, &name)).await,
                Method::DELETE => crate::blocking(move || delete_policy(&node, &name)).await,
                _ => method_not_allowed(&req, &["GET", "PUT", "DELETE"]),
            }
        }
        ("capacity", 2) => {
            if m != Method::GET {
                return method_not_allowed(&req, &["GET"]);
            }
            crate::util::check_params(&req, &["master_timeout"])?;
            crate::blocking(move || get_capacity(&node)).await
        }
        _ => crate::not_found_handler(&req),
    }
}
