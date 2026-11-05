//! Small ES 9 endpoints without a home elsewhere: dangling indices, the
//! Painless context listing, node shutdown metadata, logs streams, EQL stats.

use crate::observability::EPHEMERAL_ID;
use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::{Node, NODE_ID};
use std::sync::Mutex;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

/// Cluster-state customs this module keeps per node (`NodesShutdownMetadata`,
/// `StreamsMetadata`), persisted next to the node's data.
#[derive(Default, serde::Serialize, serde::Deserialize, Clone)]
struct State {
    #[serde(default)]
    shutdowns: Vec<Value>,
    #[serde(default)]
    streams: Map<String, Value>,
}

fn states() -> &'static Mutex<HashMap<PathBuf, State>> {
    static S: OnceLock<Mutex<HashMap<PathBuf, State>>> = OnceLock::new();
    S.get_or_init(Default::default)
}

fn state_path(node: &Node) -> PathBuf {
    node.data.join("_misc9_state.json")
}

fn with_state<T>(node: &Node, f: impl FnOnce(&mut State) -> T) -> Result<T, EsError> {
    let mut g = states().lock().unwrap_or_else(|e| e.into_inner());
    let st = g.entry(node.data.clone()).or_insert_with(|| std::fs::read(state_path(node)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default());
    let before = serde_json::to_vec(&*st).unwrap_or_default();
    let out = f(st);
    let after = serde_json::to_vec(&*st).unwrap_or_default();
    if before != after {
        let tmp = state_path(node).with_extension("json.tmp");
        std::fs::write(&tmp, &after)?;
        std::fs::rename(tmp, state_path(node))?;
    }
    Ok(out)
}

// ------------------------------------------------------------- dangling

/// `GET /_dangling`, `POST|DELETE /_dangling/{index_uuid}`: this node never
/// has dangling indices (every index on disk is in its metadata).
pub async fn dangling(node: Arc<Node>, req: Req) -> R {
    let uuid = req.seg(1).to_string();
    match (uuid.as_str(), &req.method) {
        ("", &Method::GET) => Ok(Resp::ok(json!({"_nodes": {"total": 1, "successful": 1, "failed": 0}, "cluster_name": node.cluster_name, "dangling_indices": []}))),
        ("", _) => method_not_allowed(&req, &["GET"]),
        (u, &Method::POST) | (u, &Method::DELETE) if req.segs.len() == 2 => Err(EsError::illegal_argument(format!("No dangling index found for UUID [{u}]"))),
        _ => crate::not_found_handler(&req),
    }
}

// ------------------------------------------------------------- painless

/// `PainlessScriptEngine` contexts (ES 9.5.5 default distribution).
const PAINLESS_CONTEXTS: &[&str] = &[
    "aggregation_selector", "aggs", "aggs_combine", "aggs_init", "aggs_map", "aggs_reduce", "analysis", "boolean_field", "bucket_aggregation",
    "bytesref_sort", "composite_field", "date_field", "double_field", "double_values", "field", "filter", "geo_point_field", "geometry_field",
    "ingest", "ingest_template", "interval", "ip_field", "keyword_field", "long_field", "moving-function", "number_sort", "painless_test",
    "processor_conditional", "reindex", "score", "script_heuristic", "similarity", "similarity_weight", "string_sort", "template", "terms_set",
    "update", "update_by_query", "watcher_condition", "watcher_transform", "xpack_template",
];

/// `GET /_scripts/painless/_context[?context=]` (`PainlessContextAction`).
pub fn painless_context(req: &Req) -> R {
    match req.param("context") {
        None => Ok(Resp::ok(json!({"contexts": PAINLESS_CONTEXTS}))),
        Some(c) if !PAINLESS_CONTEXTS.contains(&c) => Err(EsError::illegal_argument(format!("script context [{c}] not found"))),
        // The whitelist of a context (classes, methods, bindings) is not
        // modelled: the context answers with its name and no entries.
        Some(c) => Ok(Resp::ok(json!({"name": c, "classes": [], "imported_methods": [], "class_bindings": [], "instance_bindings": []}))),
    }
}

// ------------------------------------------------------------- shutdown

const SHUTDOWN_TYPES: &[&str] = &["RESTART", "REMOVE", "REPLACE", "SIGTERM"];

/// `PUT|GET|DELETE /_nodes/{node_id}/shutdown`, `GET /_nodes/shutdown`.
pub async fn shutdown(node: Arc<Node>, req: Req) -> R {
    let id = if req.seg(1) == "shutdown" { String::new() } else { req.seg(1).to_string() };
    blocking(move || match (req.method.clone(), id.as_str()) {
        (Method::GET, "") => shutdown_get(&node, None),
        (Method::GET, i) => shutdown_get(&node, Some(i)),
        (Method::PUT, i) if !i.is_empty() => shutdown_put(&node, &req, i),
        (Method::DELETE, i) if !i.is_empty() => {
            let removed = with_state(&node, |st| {
                let before = st.shutdowns.len();
                st.shutdowns.retain(|s| s["node_id"].as_str() != Some(i));
                before != st.shutdowns.len()
            })?;
            if !removed {
                return Err(EsError::resource_not_found(format!("node [{i}] is not currently shutting down")));
            }
            Ok(Resp::ack())
        }
        _ => method_not_allowed(&req, &["GET", "PUT", "DELETE"]),
    })
    .await
}

fn shutdown_put(node: &Node, req: &Req, id: &str) -> R {
    let body = req.json()?;
    let o = body.as_object().cloned().unwrap_or_default();
    for k in o.keys() {
        if !matches!(k.as_str(), "type" | "reason" | "allocation_delay" | "target_node_name" | "grace_period") {
            return Err(EsError::x_content_parse(format!("[1:2] [put_node_shutdown_request] unknown field [{k}]")));
        }
    }
    let s = |k: &str| o.get(k).filter(|v| !v.is_null()).map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()));
    let missing: Vec<&str> = ["type", "reason"].into_iter().filter(|k| s(k).is_none()).collect();
    if !missing.is_empty() {
        return Err(EsError::illegal_argument(format!("Required [{}]", missing.join(", "))));
    }
    let ty = s("type").unwrap_or_default().to_ascii_uppercase();
    if !SHUTDOWN_TYPES.contains(&ty.as_str()) {
        return Err(EsError::x_content_parse("Failed to build [put_node_shutdown_request] after last required field arrived")
            .caused_by(EsError::illegal_argument(format!("unknown shutdown type: {}", s("type").unwrap_or_default()))));
    }
    let (delay, target, grace) = (s("allocation_delay"), s("target_node_name"), s("grace_period"));
    let mut errs: Vec<String> = Vec::new();
    if delay.is_some() && ty != "RESTART" {
        errs.push("allocation_delay is only allowed for RESTART-type shutdown requests".into());
    }
    match (&target, ty.as_str()) {
        (Some(t), x) if x != "REPLACE" => errs.push(format!("target node name is only valid for REPLACE type shutdowns, but was given type [{ty}] and target node name [{t}]")),
        (None, "REPLACE") => errs.push("target node name is required for REPLACE type shutdowns".into()),
        _ => {}
    }
    if ty == "SIGTERM" {
        if grace.is_none() {
            errs.push("grace period is required for SIGTERM shutdowns".into());
        }
    } else if grace.is_some() {
        errs.push(format!("grace period is only valid for SIGTERM type shutdowns, but was given type [{ty}] and target node name [{}]", target.clone().unwrap_or_else(|| "null".into())));
    }
    if !errs.is_empty() {
        let refs: Vec<&str> = errs.iter().map(String::as_str).collect();
        return Err(EsError::action_validation(&refs));
    }
    let mut rec = json!({"node_id": id, "type": ty, "reason": s("reason"), "shutdown_started_millis": elasticsearch_engine::date::now_millis()});
    for (k, v) in [("allocation_delay", delay), ("target_node_name", target), ("grace_period", grace)] {
        if let Some(v) = v {
            rec[k] = json!(v);
        }
    }
    with_state(node, |st| match st.shutdowns.iter_mut().find(|r| r["node_id"].as_str() == Some(id)) {
        // An update keeps the original start time.
        Some(cur) => {
            let started = cur["shutdown_started_millis"].clone();
            *cur = rec;
            cur["shutdown_started_millis"] = started;
        }
        None => st.shutdowns.push(rec),
    })?;
    Ok(Resp::ack())
}

/// `SingleNodeShutdownStatus` of each registered shutdown (all, or the
/// comma-separated node ids).
fn shutdown_get(node: &Node, ids: Option<&str>) -> R {
    let wanted: Option<Vec<&str>> = ids.map(|i| i.split(',').map(str::trim).filter(|x| !x.is_empty() && *x != "_all").collect()).filter(|v: &Vec<&str>| !v.is_empty());
    let recs = with_state(node, |st| st.shutdowns.clone())?;
    let shards: usize = node.all_indices().iter().map(|s| s.number_of_shards() as usize).sum();
    let first_index = node.all_indices().into_iter().map(|s| s.name.clone()).min();
    let mut out = Vec::new();
    for r in recs {
        let nid = r["node_id"].as_str().unwrap_or("");
        if wanted.as_ref().is_some_and(|w| !w.contains(&nid)) {
            continue;
        }
        let member = nid == NODE_ID;
        let ty = r["type"].as_str().unwrap_or("");
        let shard_migration = if ty == "RESTART" {
            json!({"status": "COMPLETE", "shard_migrations_remaining": 0, "explanation": "no shard relocation is necessary for a node restart"})
        } else if !member {
            json!({"status": "NOT_STARTED", "shard_migrations_remaining": 0, "explanation": "node is not currently part of the cluster"})
        } else if shards > 0 {
            json!({"status": "STALLED", "shard_migrations_remaining": shards,
                   "explanation": format!("shard [0] [primary] of index [{}] cannot move, use the Cluster Allocation Explain API on this shard for details", first_index.clone().unwrap_or_default())})
        } else {
            json!({"status": "COMPLETE", "shard_migrations_remaining": 0})
        };
        let started = if member { "COMPLETE" } else { "NOT_STARTED" };
        let parts = [shard_migration["status"].as_str().unwrap_or(""), started, "COMPLETE", started];
        let overall = if parts.contains(&"STALLED") {
            "STALLED"
        } else if parts.iter().all(|p| *p == "COMPLETE") {
            "COMPLETE"
        } else if parts.contains(&"IN_PROGRESS") {
            "IN_PROGRESS"
        } else {
            "NOT_STARTED"
        };
        let mut m = Map::new();
        m.insert("node_id".into(), json!(nid));
        m.insert("node_ephemeral_id".into(), if member { json!(EPHEMERAL_ID) } else { Value::Null });
        m.insert("type".into(), json!(ty));
        m.insert("reason".into(), r["reason"].clone());
        if let Some(d) = r.get("allocation_delay") {
            m.insert("allocation_delay".into(), d.clone());
        }
        m.insert("shutdown_started_millis".into(), r["shutdown_started_millis"].clone());
        m.insert("status".into(), json!(overall));
        m.insert("shard_migration".into(), shard_migration);
        m.insert("persistent_tasks".into(), json!({"status": started, "persistent_tasks_remaining": 0, "auto_reassignable_tasks_remaining": 0}));
        m.insert("plugins".into(), json!({"status": "COMPLETE"}));
        m.insert("shard_snapshots".into(), json!({"status": started, "completed_shards": 0, "paused_shards": 0, "running_shards": 0}));
        for k in ["target_node_name", "grace_period"] {
            if let Some(v) = r.get(k) {
                m.insert(k.into(), v.clone());
            }
        }
        out.push(Value::Object(m));
    }
    Ok(Resp::ok(json!({"nodes": out})))
}

// ------------------------------------------------------------- streams

/// `StreamTypes` (ES 9.5.5).
const STREAM_TYPES: &[&str] = &["logs", "logs.ecs", "logs.otel"];

/// `GET /_streams/status`, `POST /_streams/{name}/_enable|_disable`.
pub async fn streams(node: Arc<Node>, req: Req) -> R {
    let (a, b) = (req.seg(1).to_string(), req.seg(2).to_string());
    blocking(move || match (a.as_str(), b.as_str(), &req.method) {
        ("status", "", &Method::GET) => {
            let st = with_state(&node, |s| s.streams.clone())?;
            let mut out = Map::new();
            for t in STREAM_TYPES {
                out.insert((*t).into(), json!({"enabled": st.get(*t).and_then(Value::as_bool).unwrap_or(false)}));
            }
            Ok(Resp::ok(Value::Object(out)))
        }
        (name, op @ ("_enable" | "_disable"), &Method::POST) => {
            if !STREAM_TYPES.contains(&name) {
                return Err(EsError::illegal_argument(format!("Unknown stream type [{name}]")));
            }
            let on = op == "_enable";
            with_state(&node, |s| {
                s.streams.insert(name.to_string(), json!(on));
            })?;
            Ok(Resp::ack())
        }
        _ => crate::not_found_handler(&req),
    })
    .await
}

// ------------------------------------------------------------- eql

/// `GET /_eql/stats` (`EqlStatsAction`): the per-node feature and query counters.
pub fn eql_stats(node: &Node) -> R {
    let stats = json!({
        "features": {
            "joins": {"join_queries_three": 0, "join_queries_two": 0, "join_until": 0, "join_queries_five_or_more": 0, "join_queries_four": 0},
            "sequence": 0,
            "keys": {"join_keys_two": 0, "join_keys_one": 0, "join_keys_three": 0, "join_keys_five_or_more": 0, "join_keys_four": 0},
            "join": 0,
            "sequences": {"sequence_queries_three": 0, "sequence_queries_four": 0, "sequence_queries_two": 0, "sequence_until": 0, "sequence_maxspan": 0, "sequence_queries_five_or_more": 0},
            "event": 0,
            "pipes": {"pipe_tail": 0, "pipe_head": 0},
        },
        "queries": {"all": {"total": 0, "failed": 0}, "_all": {"total": 0, "failed": 0}},
    });
    Ok(Resp::ok(json!({"_nodes": {"total": 1, "successful": 1, "failed": 0}, "cluster_name": node.cluster_name, "stats": [{"stats": stats}]})))
}
