//! Where trained models live (ES `TrainedModelProvider`).
//!
//! As in ES, model configs and vocabularies are documents of the system
//! index `.ml-inference-000005`, and the definition of an uploaded model is
//! a series of chunked documents in `.ml-inference-native-000002`. Both are
//! ordinary replicated indices (one shard, `auto_expand_replicas: 0-1`), so a
//! model survives the loss of the node that stored it, and every node reads
//! models from them. On a cluster the reads and writes go through the
//! cluster coordinator (every shard copy, replicated before the response);
//! a single node acts on its own indices. A write is acknowledged only once
//! the documents are written (replicated on a cluster) and refreshed.
//!
//! Deployments and model aliases are cluster-state metadata (ES
//! `TrainedModelAssignmentMetadata`, `ModelAliasMetadata`): the master keeps
//! them, publishes them ([`cluster_state`]) and every other node applies
//! them ([`restore_cluster_state`]).
//!
//! A node serves models from the indices once the HTTP layer bound it
//! ([`bind`]); unbound (an engine used as a library, unit tests) the store is
//! the legacy `<path.data>/ml-models` files. On first use, models found in
//! those files are migrated into the indices.

use crate::coord::{self, path_segment};
use crate::error::{EsError, Result};
use crate::index::WriteReq;
use crate::node::Node;
use crate::shard::WriteCond;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

pub const CONFIG_INDEX: &str = ".ml-inference-000005";
pub const NATIVE_INDEX: &str = ".ml-inference-native-000002";

const CONFIG_TYPE: &str = "trained_model_config";
const VOCAB_TYPE: &str = "trained_model_vocabulary";
const DEFINITION_TYPE: &str = "trained_model_definition_doc";
/// Marker written into the legacy model directory once its models are in the indices.
const MIGRATED_MARKER: &str = ".migrated-to-ml-inference-index";

fn nodes() -> &'static Mutex<HashMap<PathBuf, Weak<Node>>> {
    static N: OnceLock<Mutex<HashMap<PathBuf, Weak<Node>>>> = OnceLock::new();
    N.get_or_init(Default::default)
}

/// Serve the models of the node whose `_state` dir is `state_dir` from the
/// model indices (called by the HTTP layer for every request; cheap).
pub fn bind(state_dir: &Path, node: &Arc<Node>) {
    let mut n = nodes().lock().unwrap_or_else(|e| e.into_inner());
    if n.get(state_dir).is_some_and(|w| w.strong_count() > 0) {
        return;
    }
    n.insert(state_dir.to_path_buf(), Arc::downgrade(node));
}

/// [`bind`] a node (its `<path.data>/_state`), from the per-request path:
/// a thread remembers the node it bound last.
pub fn bind_node(node: &Arc<Node>) {
    thread_local! {
        // A `Weak` keeps the allocation: no other node can reuse its address.
        static LAST: std::cell::RefCell<Weak<Node>> = const { std::cell::RefCell::new(Weak::new()) };
    }
    if LAST.with(|l| std::ptr::eq(l.borrow().as_ptr(), Arc::as_ptr(node))) {
        return;
    }
    bind(&node.data.join("_state"), node);
    LAST.with(|l| *l.borrow_mut() = Arc::downgrade(node));
}

/// The node bound to `state_dir`, when models are served from the indices.
pub fn bound(state_dir: &Path) -> Option<Arc<Node>> {
    nodes().lock().unwrap_or_else(|e| e.into_inner()).get(state_dir).and_then(Weak::upgrade)
}

fn index_settings() -> Value {
    json!({
        "settings": {"index": {"hidden": true, "number_of_shards": 1, "auto_expand_replicas": "0-1"}},
        "mappings": {"dynamic": false, "properties": {
            "doc_type": {"type": "keyword"},
            "model_id": {"type": "keyword"},
            "doc_num": {"type": "long"}
        }}
    })
}

/// Create the model indices when missing.
fn ensure_indices(node: &Node) -> Result<()> {
    for index in [CONFIG_INDEX, NATIVE_INDEX] {
        if node.get_index(index).is_some() {
            continue;
        }
        let r = match coord::internal() {
            Some(c) => c("PUT", &format!("/{}", path_segment(index)), Some(index_settings())).map(|_| ()),
            None => node.create_index(index, &index_settings()).map(|_| ()),
        };
        match r {
            Err(e) if e.kind != "resource_already_exists_exception" => return Err(e),
            _ => {}
        }
    }
    Ok(())
}

enum Op {
    Index(String, Value),
    Delete(String),
}

/// Write documents to `index`; returns once they are durable (replicated
/// on a cluster) and visible.
fn write(node: &Node, index: &str, ops: Vec<Op>) -> Result<()> {
    if ops.is_empty() {
        return Ok(());
    }
    ensure_indices(node)?;
    if let Some(c) = coord::internal() {
        let mut lines = String::new();
        for op in &ops {
            match op {
                Op::Index(id, v) => {
                    lines.push_str(&json!({"index": {"_index": index, "_id": id}}).to_string());
                    lines.push('\n');
                    lines.push_str(&v.to_string());
                    lines.push('\n');
                }
                Op::Delete(id) => {
                    lines.push_str(&json!({"delete": {"_index": index, "_id": id}}).to_string());
                    lines.push('\n');
                }
            }
        }
        let resp = c("POST", "/_bulk?refresh=true", Some(Value::String(lines)))?;
        if resp["errors"].as_bool() == Some(true) {
            for item in resp["items"].as_array().into_iter().flatten() {
                let op = item.as_object().and_then(|o| o.values().next()).cloned().unwrap_or(Value::Null);
                if let Some(err) = op.get("error").filter(|e| e.is_object()) {
                    let status = op["status"].as_u64().unwrap_or(500) as u16;
                    return Err(EsError::new(status, err["type"].as_str().unwrap_or("exception"), err["reason"].as_str().unwrap_or("model store write failed")));
                }
            }
        }
        return Ok(());
    }
    let svc = node.get_index(index).ok_or_else(|| EsError::index_not_found(index))?;
    let reqs: Vec<WriteReq> = ops
        .into_iter()
        .map(|op| match op {
            Op::Index(id, v) => WriteReq::Index { id: Some(id), source: v.to_string(), routing: None, cond: WriteCond::default() },
            Op::Delete(id) => WriteReq::Delete { id, routing: None, cond: WriteCond::default() },
        })
        .collect();
    for r in svc.write(reqs, true) {
        match r {
            Ok(_) => {}
            Err(e) if e.kind == "document_missing_exception" || e.status == 404 => {}
            Err(e) => return Err(e),
        }
    }
    svc.refresh()?;
    Ok(())
}

/// Every `(id, source)` of `index` matching `query`, `page` at a time, sorted.
/// A model store search, made with the internal origin as its writes are
/// (the model indices are internal: the caller needs no privilege there).
fn internal_search(node: &Node, index: &str, body: &Value) -> Result<Value> {
    coord::scoped(coord::internal(), || crate::ml::resultsidx::search(node, index, body))
}

fn search_all(node: &Node, index: &str, query: Value, sort: Value, page: usize) -> Result<Vec<(String, Value)>> {
    let mut out = Vec::new();
    loop {
        let body = json!({"from": out.len(), "size": page, "query": query, "sort": sort, "track_total_hits": false});
        let resp = internal_search(node, index, &body)?;
        let h = crate::ml::resultsidx::hits(&resp);
        let n = h.len();
        out.extend(h);
        if n < page || out.len() >= 10_000 {
            break;
        }
    }
    Ok(out)
}

fn config_doc_id(id: &str) -> String {
    format!("{CONFIG_TYPE}-{id}")
}

fn vocab_doc_id(id: &str) -> String {
    format!("{VOCAB_TYPE}-{id}")
}

fn definition_doc_id(id: &str, part: u64) -> String {
    format!("{DEFINITION_TYPE}-{id}-{part}")
}

// ---------------------------------------------------------------- configs

/// Every stored model config, by model id.
pub fn load_configs(node: &Node) -> Result<Map<String, Value>> {
    let docs = search_all(node, CONFIG_INDEX, json!({"term": {"doc_type": CONFIG_TYPE}}), json!([{"model_id": "asc"}]), 1000)?;
    Ok(docs.into_iter().filter_map(|(_, s)| Some((s["model_id"].as_str()?.to_string(), s.get("config")?.clone()))).collect())
}

/// One stored model config.
pub fn get_config(node: &Node, id: &str) -> Result<Option<Value>> {
    let resp = internal_search(node, CONFIG_INDEX, &json!({"size": 1, "query": {"ids": {"values": [config_doc_id(id)]}}}))?;
    Ok(crate::ml::resultsidx::hits(&resp).into_iter().next().and_then(|(_, s)| s.get("config").cloned()))
}

/// Make the stored configs `next` (`prev`: what is stored now).
pub fn save_configs(node: &Node, prev: &Map<String, Value>, next: &Map<String, Value>) -> Result<()> {
    let mut ops = Vec::new();
    for (id, cfg) in next {
        if prev.get(id) != Some(cfg) {
            ops.push(Op::Index(config_doc_id(id), json!({"doc_type": CONFIG_TYPE, "model_id": id, "config": cfg})));
        }
    }
    for id in prev.keys().filter(|k| !next.contains_key(*k)) {
        ops.push(Op::Delete(config_doc_id(id)));
    }
    write(node, CONFIG_INDEX, ops)
}

// ---------------------------------------------------------------- artifacts

pub fn put_vocabulary(node: &Node, id: &str, body: &Value) -> Result<()> {
    write(node, CONFIG_INDEX, vec![Op::Index(vocab_doc_id(id), json!({"doc_type": VOCAB_TYPE, "model_id": id, "body": body}))])
}

/// The stored vocabulary of `id` (`Ok(None)`: none stored). A store that
/// cannot be read is an error, never an absent vocabulary.
pub fn vocabulary(node: &Node, id: &str) -> Result<Option<Value>> {
    let resp = internal_search(node, CONFIG_INDEX, &json!({"size": 1, "query": {"ids": {"values": [vocab_doc_id(id)]}}}))?;
    if resp["timed_out"] == true || resp["_shards"]["failed"].as_u64().unwrap_or(0) > 0 {
        return Err(EsError::new(503, "status_exception", format!("the vocabulary of model [{id}] could not be read completely")));
    }
    Ok(crate::ml::resultsidx::hits(&resp).into_iter().next().and_then(|(_, s)| s.get("body").cloned()))
}

pub fn put_definition_part(node: &Node, id: &str, part: u64, body: &Value) -> Result<()> {
    write(node, NATIVE_INDEX, vec![Op::Index(definition_doc_id(id, part), json!({"doc_type": DEFINITION_TYPE, "model_id": id, "doc_num": part, "body": body}))])
}

/// The definition parts of `id`, by part number.
pub fn definition_parts(node: &Node, id: &str) -> Result<Vec<(u64, Value)>> {
    // Parts can be megabytes each: a few per page.
    let docs = search_all(node, NATIVE_INDEX, json!({"bool": {"filter": [{"term": {"doc_type": DEFINITION_TYPE}}, {"term": {"model_id": id}}]}}), json!([{"doc_num": "asc"}]), 8)?;
    Ok(docs.into_iter().filter_map(|(_, s)| Some((s["doc_num"].as_u64()?, s.get("body")?.clone()))).collect())
}

/// Delete the vocabulary and definition of `id`.
pub fn delete_artifacts(node: &Node, id: &str) -> Result<()> {
    write(node, CONFIG_INDEX, vec![Op::Delete(vocab_doc_id(id))])?;
    let parts: Vec<Op> = definition_parts(node, id)?.into_iter().map(|(n, _)| Op::Delete(definition_doc_id(id, n))).collect();
    write(node, NATIVE_INDEX, parts)
}

/// Delete every stored model (feature reset).
pub fn reset(node: &Node) -> Result<()> {
    for index in [CONFIG_INDEX, NATIVE_INDEX] {
        if node.get_index(index).is_none() {
            continue;
        }
        match coord::internal() {
            Some(c) => {
                c("DELETE", &format!("/{}", path_segment(index)), None)?;
            }
            None => node.delete_index(index)?,
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- migration

/// Import the models of the legacy `ml-models` directory into the indices,
/// once (configs, vocabularies and definition parts; nothing is overwritten).
pub fn migrate_legacy(node: &Node, root: &Path) -> Result<usize> {
    static DONE: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    let done = DONE.get_or_init(Default::default);
    if done.lock().unwrap_or_else(|e| e.into_inner()).contains(root) || root.join(MIGRATED_MARKER).exists() {
        return Ok(0);
    }
    let legacy: Map<String, Value> = std::fs::read(root.join("models.json")).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).and_then(|v| v.as_object().cloned()).unwrap_or_default();
    let mut moved = 0;
    if !legacy.is_empty() {
        let stored = load_configs(node)?;
        let mut configs = stored.clone();
        for (id, cfg) in &legacy {
            if stored.contains_key(id) || !super::models::acceptable_model_id(id) {
                continue;
            }
            configs.insert(id.clone(), cfg.clone());
            let dir = root.join(id);
            if let Ok(b) = std::fs::read(dir.join("vocabulary.json")) {
                if let Ok(v) = serde_json::from_slice::<Value>(&b) {
                    put_vocabulary(node, id, &v)?;
                }
            }
            for e in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                let Some(part) = name.strip_prefix("definition-").and_then(|n| n.strip_suffix(".json")).and_then(|n| n.parse::<u64>().ok()) else { continue };
                if let Some(v) = std::fs::read(e.path()).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()) {
                    put_definition_part(node, id, part, &v)?;
                }
            }
            moved += 1;
        }
        save_configs(node, &stored, &configs)?;
        tracing::info!("migrated {moved} trained model(s) from {} into [{CONFIG_INDEX}] / [{NATIVE_INDEX}]", root.display());
    }
    if root.exists() {
        let _ = std::fs::write(root.join(MIGRATED_MARKER), b"");
    }
    done.lock().unwrap_or_else(|e| e.into_inner()).insert(root.to_path_buf());
    Ok(moved)
}
