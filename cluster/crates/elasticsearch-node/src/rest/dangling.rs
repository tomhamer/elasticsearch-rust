//! Dangling indices APIs (ES 8.19): `GET /_dangling`,
//! `POST /_dangling/{index_uuid}?accept_data_loss=true` (import) and
//! `DELETE /_dangling/{index_uuid}?accept_data_loss=true`.
//!
//! An index copy on a node is dangling when the cluster metadata does not
//! know it: its name is absent from `metadata.indices`, or the node's
//! recorded cluster uuid for that name differs from the metadata's uuid.
//! Every node reports its own through `internal:dangling/list`; the
//! coordinating node merges them by uuid (`TransportListDanglingIndicesAction`).

use super::node_shutdown::{check_params, method_not_allowed, ApiError};
use crate::front::{RestReq, RestResp, SharedText};
use crate::shared_sequence::SharedSequence;
use elasticsearch_engine::error::EsError;
use crate::node::ClusterNode;
use elasticsearch_cluster::state::ClusterState;
use elasticsearch_transport::RemoteError;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

pub const ACTION_LIST: &str = "internal:dangling/list";
/// The previous release's delete action. Its coordinators send it for
/// `DELETE /_dangling` *and* as the first step of an import (delete every
/// holder's copy, then recreate the index empty), with the same payload, so
/// a holder cannot tell the two apart. This build refuses it: an old
/// coordinator in a mixed cluster can no longer delete a copy on an
/// upgraded holder, which is lost only when an operator deletes it from an
/// upgraded coordinator (or after the upgrade).
pub const ACTION_DELETE: &str = "internal:dangling/delete";
/// The delete action of builds whose import never deletes: only ever sent
/// for `DELETE /_dangling?accept_data_loss=true`.
pub const ACTION_DELETE_V2: &str = "internal:dangling/delete/v2";
/// What a node holds of one dangling index, for its import: the index's
/// own metadata and the shard copies (shard, allocation id) on the node.
pub const ACTION_IMPORT_INFO: &str = "internal:dangling/import_info";
/// Import a dangling index (sent to the elected master).
pub const ACTION_IMPORT: &str = "internal:dangling/import";
/// Tombstone a dangling index being deleted (sent to the elected master):
/// `(index name, index uuid)`.
pub const ACTION_TOMBSTONE: &str = "internal:dangling/tombstone";

/// A node's part of a dangling index import ([`ACTION_IMPORT_INFO`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportInfo {
    /// The index metadata as the copy holds it (its local identity; the
    /// importer sets the dangling uuid and the in-sync copies).
    pub index: elasticsearch_cluster::state::IndexMetadata,
    /// (shard, allocation id) of every copy of the index on the node.
    pub copies: Vec<(u32, String)>,
}

/// One dangling copy as a node reports it. Settings and mappings travel as
/// JSON text (the transport codec is not self-describing).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Dangling {
    pub name: SharedText,
    pub uuid: SharedText,
    pub creation_date: i64,
    pub settings: SharedText,
    pub mappings: SharedText,
}

/// Indices on this node the cluster metadata does not know, from the
/// node's (name, local uuid, creation date) list and its name to cluster
/// uuid map.
pub fn find_dangling(state: &ClusterState, local: &[(String, String, i64)], cluster_uuids: &BTreeMap<String, (String, String)>) -> Vec<(String, String, i64)> {
    local
        .iter()
        .filter_map(|(name, local_uuid, created)| {
            let uuid = cluster_uuids.get(name).map(|(c, _)| c.clone()).unwrap_or_else(|| local_uuid.clone());
            let dangling = match state.metadata.indices.get(name) {
                None => true,
                Some(m) => m.uuid != uuid,
            };
            dangling.then(|| (name.clone(), uuid, *created))
        })
        .collect()
}

pub fn local_dangling(node: &ClusterNode) -> Result<SharedSequence<Dangling>, EsError> {
    let state = node.state();
    let all = node.engine.all_indices();
    SharedSequence::try_generate(|push| {
        for svc in &all {
            // Borrow the registry under its lock; copy only accepted names into
            // admitted owners, rather than cloning the entire registry first.
            let uuid = {
                let uuids = node.uuids.lock();
                let uuid = uuids.indices.get(&svc.name).map(|(c, _)| c.as_str()).unwrap_or(&svc.uuid);
                if state.metadata.indices.get(&svc.name).is_some_and(|m| m.uuid == uuid) { continue; }
                SharedText::try_new(uuid)?
            };
            let meta = svc.meta();
            push(Dangling {
                name: SharedText::try_new(&svc.name)?, uuid,
                creation_date: svc.creation_date,
                settings: SharedText::try_json(&meta.settings)?,
                mappings: { let (text,memory)=meta.mappings.to_json_admitted()?;SharedText::from_reserved(text,memory)? },
            })?;
        }
        Ok(())
    })
}

/// Index copies this node quarantined because their identity could not be
/// established (`mirror::reconcile_incarnation`), in every engine: listed
/// as dangling (under the cluster index uuid the copy records, else its
/// local uuid) with their location, deletable with `DELETE _dangling`.
///
/// Every engine directory on disk is scanned, opened or not: a shard engine
/// opens only for shard numbers the metadata has, so the copy of shard
/// k > 0 of an index the metadata lacks would otherwise never be seen. In
/// an engine that is not open, an index directory whose index uuid the
/// metadata does not hold is listed too (ES lists every shard's copy).
pub fn quarantined(node: &ClusterNode) -> Vec<(Dangling, std::path::PathBuf)> {
    let state = node.state();
    let known = |uuid: &str| state.metadata.indices.values().any(|im| im.uuid == uuid);
    let mut out = Vec::new();
    for root in engine_roots(node) {
        out.extend(quarantined_in(&root));
        let closed_shard_engine = shard_engine_number(&root).is_some_and(|k| node.shard_engines.get(k).is_none());
        if closed_shard_engine {
            out.extend(unopened_dangling_in(&root, &known));
        }
    }
    out
}

/// Delete what a tombstone of `md` names in the shard engines of this node
/// that are not open (index directories and set-aside copies alike): an
/// engine opens only for the shard numbers the metadata has, so a copy of a
/// deleted dangling index there would otherwise outlive its tombstone.
/// Open engines purge their own (`mirror::purge_tombstoned_set_aside`).
pub fn purge_tombstoned_unopened(node: &ClusterNode, md: &elasticsearch_cluster::state::Metadata) {
    if elasticsearch_cluster::graveyard::tombstones(md).is_empty() {
        return;
    }
    for root in engine_roots(node) {
        if !shard_engine_number(&root).is_some_and(|k| node.shard_engines.get(k).is_none()) {
            continue;
        }
        let tombstoned = |_: &str| false;
        let mut doomed = unopened_dangling_in(&root, &tombstoned);
        doomed.extend(quarantined_in(&root));
        for (d, path) in doomed.into_iter().filter(|(d, _)| elasticsearch_cluster::graveyard::contains(md, d.uuid.as_str())) {
            // Re-checked: the engine may have opened meanwhile.
            if shard_engine_number(&root).is_some_and(|k| node.shard_engines.get(k).is_some()) {
                break;
            }
            match std::fs::remove_dir_all(&path) {
                Ok(()) => tracing::info!(index = %d.name, uuid = %d.uuid, path = %path.display(), "deleted a copy its tombstone names"),
                Err(e) => tracing::warn!(index = %d.name, uuid = %d.uuid, "cannot delete a tombstoned copy: {e}"),
            }
        }
    }
}

/// `k` of a shard engine directory `.../engine-shard-<k>`.
fn shard_engine_number(root: &std::path::Path) -> Option<u32> {
    root.file_name()?.to_str()?.strip_prefix("engine-shard-")?.parse().ok()
}

/// The index directories of an engine that is not open whose index uuid
/// `known` does not accept (not in the cluster metadata).
fn unopened_dangling_in(root: &std::path::Path, known: &dyn Fn(&str) -> bool) -> Vec<(Dangling, std::path::PathBuf)> {
    let Ok(entries) = std::fs::read_dir(root.join("indices")) else { return Vec::new() };
    let mut out: Vec<(Dangling, std::path::PathBuf)> = entries.flatten()
        .filter(|e| crate::mirror::holds_shard_data(&e.path()))
        .filter_map(|e| quarantined_copy(&e.path()).map(|d| (d, e.path())))
        .filter(|(d, _)| !known(d.uuid.as_str()))
        .collect();
    out.sort_by(|a, b| a.1.cmp(&b.1));
    out
}

/// The set-aside copies of the engine whose data directory is `root`.
pub fn quarantined_in(root: &std::path::Path) -> Vec<(Dangling, std::path::PathBuf)> {
    let Ok(entries) = std::fs::read_dir(root.join(elasticsearch_engine::node::Node::QUARANTINE_DIR)) else { return Vec::new() };
    let mut out: Vec<(Dangling, std::path::PathBuf)> = entries.flatten().filter_map(|e| quarantined_copy(&e.path()).map(|d| (d, e.path()))).collect();
    out.sort_by(|a, b| a.1.cmp(&b.1));
    out
}

/// The dangling entry of one quarantined index directory.
pub fn quarantined_copy(path: &std::path::Path) -> Option<Dangling> {
    let raw: Value = serde_json::from_slice(&std::fs::read(path.join("index_meta.json")).ok()?).ok()?;
    let uuid = crate::shards::read_copy_record(path).map(|r| r.index_uuid)
        .or_else(|| raw.get("cluster_uuid").and_then(Value::as_str).map(str::to_string))
        .or_else(|| raw.get("uuid").and_then(Value::as_str).map(str::to_string))?;
    Some(Dangling {
        name: SharedText::try_new(raw.get("name").and_then(Value::as_str).unwrap_or_default()).ok()?,
        uuid: SharedText::try_new(&uuid).ok()?,
        creation_date: raw.get("creation_date").and_then(Value::as_i64).unwrap_or(0),
        settings: SharedText::try_json(raw.get("settings").unwrap_or(&Value::Null)).ok()?,
        mappings: SharedText::try_json(raw.get("mappings").unwrap_or(&Value::Null)).ok()?,
    })
}

/// Delete this node's quarantined copies of `uuid`; whether any existed.
fn delete_quarantined(node: &ClusterNode, uuid: &str) -> Result<bool, String> {
    let mut found = false;
    for (d, path) in quarantined(node) {
        if d.uuid.as_str() == uuid {
            found = true;
            std::fs::remove_dir_all(&path).map_err(|e| format!("cannot delete quarantined copy [{}]: {e}", path.display()))?;
        }
    }
    Ok(found)
}

/// This node's dangling copies: engine indices the metadata does not know,
/// set-aside copies and copies in shard engines that are not open.
pub fn local_dangling_all(node: &ClusterNode) -> Result<SharedSequence<Dangling>, EsError> {
    let quarantined = quarantined(node);
    if quarantined.is_empty() {
        return local_dangling(node);
    }
    let engine = local_dangling(node)?;
    SharedSequence::try_generate(|push| {
        for d in engine.into_iter() {
            push(d)?;
        }
        for (d, _) in quarantined {
            push(d)?;
        }
        Ok(())
    })
}

/// The data directories of this node's engines (engine 0 and every shard
/// engine directory on disk).
fn engine_roots(node: &ClusterNode) -> Vec<std::path::PathBuf> {
    let mut roots = vec![node.engine.data.clone()];
    if let Ok(entries) = std::fs::read_dir(&node.config.data_path) {
        let mut shard_engines: Vec<std::path::PathBuf> = entries.flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("engine-shard-")).map(|e| e.path()).collect();
        shard_engines.sort();
        roots.extend(shard_engines);
    }
    roots
}

/// This node's part of dangling index `uuid` (`None`: it holds none).
pub fn local_import_info(node: &ClusterNode, uuid: &str) -> Option<ImportInfo> {
    let mut index: Option<elasticsearch_cluster::state::IndexMetadata> = None;
    let mut copies: Vec<(u32, String)> = Vec::new();
    let mut found: Option<std::path::PathBuf> = None;
    let mut note_record = |path: &std::path::Path| {
        if let Some(rec) = crate::shards::read_copy_record(path).filter(|r| r.index_uuid == uuid) {
            if !copies.contains(&(rec.shard, rec.allocation_id.clone())) {
                copies.push((rec.shard, rec.allocation_id));
            }
            found.get_or_insert_with(|| path.to_path_buf());
        }
    };
    // Engine 0 indices the metadata does not know.
    if let Ok(listed) = local_dangling(node) {
        for d in listed.into_iter().filter(|d| d.uuid.as_str() == uuid) {
            if let Some(svc) = node.engine.get_index(d.name.as_str()) {
                note_record(svc.path());
                if index.is_none() {
                    index = Some(crate::mirror::extract_index(&node.engine, &svc, &Default::default(), &mut Default::default()));
                }
            }
        }
    }
    // Shard copies of it in every engine directory on disk, opened or not:
    // a shard engine opens only for shards the metadata has, so the copy of
    // shard k > 0 is seen here only on disk.
    for root in engine_roots(node) {
        for sub in ["indices", elasticsearch_engine::node::Node::QUARANTINE_DIR] {
            let Ok(entries) = std::fs::read_dir(root.join(sub)) else { continue };
            for e in entries.flatten() {
                note_record(&e.path());
            }
        }
    }
    // Copies set aside (one per engine holding a shard of it).
    for (d, path) in quarantined(node).into_iter().filter(|(d, _)| d.uuid.as_str() == uuid) {
        note_record(&path);
        if index.is_none() {
            match elasticsearch_engine::index::IndexService::open(&path) {
                Ok(svc) => index = Some(crate::mirror::extract_index(&node.engine, &svc, &Default::default(), &mut Default::default())),
                Err(e) => tracing::warn!(index = %d.name, "cannot read the dangling copy at [{}]: {}", path.display(), e.reason),
            }
        }
    }
    // Only a shard engine's copy here: the index metadata from it.
    if index.is_none() {
        if let Some(path) = found.filter(|_| !copies.is_empty()) {
            if let Ok(svc) = elasticsearch_engine::index::IndexService::open(&path) {
                index = Some(crate::mirror::extract_index(&node.engine, &svc, &Default::default(), &mut Default::default()));
            }
        }
    }
    index.map(|index| ImportInfo { index, copies })
}

/// ES `LocalAllocateDangledIndices` on the elected master: the dangling
/// index's own metadata is added under its original uuid, with the copies
/// found on the nodes as its in-sync copies, so its primaries are allocated
/// to those copies (`EXISTING_STORE`, reason `DANGLING_INDEX_IMPORTED`) and
/// nothing is recreated empty. Each holder re-registers its copy when it
/// applies the index (`mirror::adopt_set_aside`).
pub async fn import_on_master(node: &Arc<ClusterNode>, uuid: &str) -> Result<u64, RemoteError> {
    let state = node.state();
    if state.master_node_id.as_deref() != Some(node.local.id.as_str()) {
        return Err(RemoteError::new("not_master_exception", format!("node [{}] is not the elected master", node.local.name), 503));
    }
    if elasticsearch_cluster::graveyard::contains(&state.metadata, uuid) {
        return Err(RemoteError::new("illegal_argument_exception", format!("refusing to import dangling index [{uuid}]: a tombstone names it (it was deleted)"), 400));
    }
    let mut infos: Vec<ImportInfo> = Vec::new();
    for id in state.nodes.keys() {
        if *id == node.local.id {
            infos.extend(local_import_info(node, uuid));
            continue;
        }
        let Some(addr) = node.address_of(id) else { continue };
        let target = uuid.to_string();
        match node.transport.send_request::<_, Option<ImportInfo>>(addr, ACTION_IMPORT_INFO, &target, Duration::from_secs(30)).await {
            Ok(info) => infos.extend(info),
            Err(e) => tracing::warn!(node = %id, "no dangling index information from node: {e}"),
        }
    }
    let Some(first) = infos.first() else {
        return Err(RemoteError::new("illegal_argument_exception", format!("No dangling index found for UUID [{uuid}]"), 400));
    };
    let mut im = first.index.clone();
    im.uuid = uuid.to_string();
    im.in_sync_allocation_ids.clear();
    for (shard, aid) in infos.iter().flat_map(|i| i.copies.iter()) {
        if *shard < im.number_of_shards {
            im.in_sync_allocation_ids.entry(*shard).or_default().insert(aid.clone());
        }
    }
    let name = im.name.clone();
    if let Some(existing) = state.metadata.indices.get(&name) {
        return Err(RemoteError::new("illegal_argument_exception", format!(
            "refusing to import dangling index [[{name}/{uuid}]] as an index with the same name already exists [[{name}/{}]]", existing.uuid), 400));
    }
    let (n, add) = (name.clone(), im);
    node.metadata_service().pure(Box::new(move |_, base, delta| {
        // Compare-and-set on the name: the batch is refused if it appeared.
        if !base.metadata.indices.contains_key(&n) {
            delta.indices.insert(n, Some(add));
        }
    })).await.map_err(|e| RemoteError::new("illegal_state_exception", format!("dangling index import of [{name}/{uuid}] failed: {e}"), 500))?;
    match node.state().metadata.indices.get(&name) {
        Some(im) if im.uuid == uuid => Ok(node.state().version),
        Some(im) => Err(RemoteError::new("illegal_argument_exception", format!("refusing to import dangling index [[{name}/{uuid}]]: index [{name}/{}] was created meanwhile", im.uuid), 400)),
        None => Err(RemoteError::new("illegal_state_exception", format!("dangling index import of [{name}/{uuid}] was not applied"), 500)),
    }
}

/// ES `TransportDeleteDanglingIndexAction` on the elected master: a
/// tombstone for the index, so that every node holding a copy deletes it,
/// including nodes that are away now (when they rejoin).
pub async fn tombstone_on_master(node: &Arc<ClusterNode>, name: &str, uuid: &str) -> Result<u64, RemoteError> {
    let state = node.state();
    if state.master_node_id.as_deref() != Some(node.local.id.as_str()) {
        return Err(RemoteError::new("not_master_exception", format!("node [{}] is not the elected master", node.local.name), 503));
    }
    if let Some(im) = state.metadata.indices.values().find(|im| im.uuid == uuid) {
        return Err(RemoteError::new("illegal_argument_exception", format!("Refusing to delete dangling index [{}/{uuid}]: the cluster metadata holds it", im.name), 400));
    }
    if elasticsearch_cluster::graveyard::contains(&state.metadata, uuid) {
        return Ok(state.version);
    }
    let max = node.config.settings.get("cluster.indices.tombstones.size")
        .and_then(|v| v.as_str().map(str::to_string).or_else(|| v.as_u64().map(|n| n.to_string())))
        .and_then(|v| v.parse::<usize>().ok()).unwrap_or(elasticsearch_cluster::graveyard::DEFAULT_SIZE);
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64);
    let (n, u) = (name.to_string(), uuid.to_string());
    node.metadata_service().pure(Box::new(move |_, base, delta| {
        let g = elasticsearch_cluster::graveyard::with_tombstone(&base.metadata, &n, &u, now, max);
        delta.customs.insert(elasticsearch_cluster::graveyard::CUSTOM.into(), Some(g));
    })).await.map_err(|e| RemoteError::new("illegal_state_exception", format!("cannot tombstone dangling index [{name}/{uuid}]: {e}"), 500))?;
    if elasticsearch_cluster::graveyard::contains(&node.state().metadata, uuid) {
        Ok(node.state().version)
    } else {
        Err(RemoteError::new("illegal_state_exception", format!("the tombstone of dangling index [{name}/{uuid}] was not applied"), 500))
    }
}

pub fn register(node: &Arc<ClusterNode>) {
    let n = node.clone();
    node.transport.register_handler(ACTION_LIST, move |_: ()| {
        let n = n.clone();
        async move { local_dangling_all(&n).map_err(super::wire_memory) }
    });
    let n = node.clone();
    node.transport.register_handler(ACTION_DELETE_V2, move |uuid: SharedText| {
        let n = n.clone();
        async move {
            if delete_quarantined(&n, uuid.as_str()).map_err(|e| RemoteError::new("illegal_state_exception", e, 500))? {
                return Ok::<_, RemoteError>(true);
            }
            let Some(d) = local_dangling(&n).map_err(super::wire_memory)?.into_iter().find(|d| d.uuid == uuid) else { return Ok::<_, RemoteError>(false) };
            crate::meta_extra::delete_local_copy(&n.engine, &d.name).map_err(|e| RemoteError::new("illegal_state_exception", e, 500))?;
            n.uuids.lock().indices.remove(d.name.as_str());
            Ok(true)
        }
    });
    let n = node.clone();
    node.transport.register_handler(ACTION_IMPORT_INFO, move |uuid: String| {
        let n = n.clone();
        async move { Ok::<_, RemoteError>(local_import_info(&n, &uuid)) }
    });
    let n = node.clone();
    node.transport.register_handler(ACTION_TOMBSTONE, move |(name, uuid): (String, String)| {
        let n = n.clone();
        async move { tombstone_on_master(&n, &name, &uuid).await }
    });
    let n = node.clone();
    node.transport.register_handler(ACTION_IMPORT, move |uuid: String| {
        let n = n.clone();
        async move { import_on_master(&n, &uuid).await }
    });
    // Registered (not left missing) so an old coordinator gets a clear
    // refusal rather than `action_not_found`.
    node.transport.register_handler(ACTION_DELETE, move |uuid: SharedText| async move {
        Err::<bool, RemoteError>(legacy_delete_refusal(uuid.as_str()))
    });
}

/// See [`ACTION_DELETE`].
fn legacy_delete_refusal(uuid: &str) -> RemoteError {
    RemoteError::new("unsupported_operation_exception", format!(
        "refusing to delete dangling index [{uuid}] for a coordinating node of an older version: its dangling import deletes the copy before recreating the index empty; delete it from an upgraded node"), 400)
}

/// Fan out the list action: per-node results (node id → dangling copies)
/// and the number of nodes that failed.
async fn gather(node: &Arc<ClusterNode>) -> (usize, BTreeMap<String, SharedSequence<Dangling>>, Vec<(String, String)>) {
    let state = node.state();
    let mut ok = BTreeMap::new();
    let mut failed = Vec::new();
    for id in state.nodes.keys() {
        if *id == node.local.id {
            match local_dangling_all(node) {
                Ok(value) => { ok.insert(id.clone(), value); },
                Err(e) => failed.push((id.clone(), e.to_string())),
            }
            continue;
        }
        let Some(addr) = node.address_of(id) else {
            failed.push((id.clone(), format!("node [{id}] has no known address")));
            continue;
        };
        match node.transport.send_request::<_, SharedSequence<Dangling>>(addr, ACTION_LIST, &(), Duration::from_secs(10)).await {
            Ok(d) => {
                ok.insert(id.clone(), d);
            }
            Err(e) => failed.push((id.clone(), e.to_string())),
        }
    }
    (state.nodes.len(), ok, failed)
}

/// Merge per-node results by uuid (`ListDanglingIndicesResponse.resultsByIndexUUID`).
pub fn merge(per_node: &BTreeMap<String, SharedSequence<Dangling>>) -> Vec<(Dangling, Vec<String>)> {
    let mut by_uuid: BTreeMap<SharedText, (Dangling, Vec<String>)> = BTreeMap::new();
    for (id, list) in per_node {
        for d in list {
            let holders = &mut by_uuid.entry(d.uuid.clone()).or_insert_with(|| (d.clone(), Vec::new())).1;
            // One entry per node (a node lists one copy per shard engine).
            if !holders.contains(id) {
                holders.push(id.clone());
            }
        }
    }
    by_uuid.into_values().collect()
}

pub fn list_json(cluster_name: &str, total: usize, merged: &[(Dangling, Vec<String>)], failed: &[(String, String)]) -> Value {
    let mut nodes = Map::new();
    nodes.insert("total".into(), json!(total));
    nodes.insert("successful".into(), json!(total - failed.len()));
    nodes.insert("failed".into(), json!(failed.len()));
    if !failed.is_empty() {
        let f: Vec<Value> = failed
            .iter()
            .map(|(id, reason)| json!({"type": "failed_node_exception", "reason": format!("Failed node [{id}]"), "node_id": id, "caused_by": {"type": "exception", "reason": reason}}))
            .collect();
        nodes.insert("failures".into(), Value::Array(f));
    }
    let list: Vec<Value> = merged
        .iter()
        .map(|(d, ids)| {
            json!({
                "index_name": d.name,
                "index_uuid": d.uuid,
                "creation_date": super::util::iso_millis(d.creation_date),
                "creation_date_millis": d.creation_date,
                "node_ids": ids,
            })
        })
        .collect();
    json!({"_nodes": nodes, "cluster_name": cluster_name, "dangling_indices": list})
}

fn iae(reason: &str) -> RestResp {
    ApiError::simple(400, "illegal_argument_exception", reason).resp()
}

/// `paramAsBoolean("accept_data_loss", false)`.
fn accept_data_loss(req: &RestReq) -> Result<bool, RestResp> {
    match req.params().get("accept_data_loss").map(String::as_str) {
        None | Some("false") => Ok(false),
        Some("") | Some("true") => Ok(true),
        Some(v) => Err(iae(&format!("Failed to parse value [{v}] as only [true] or [false] are allowed."))),
    }
}

pub async fn handle(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let segs = req.segments();
    let m = req.method();
    match (segs.len(), m.as_str()) {
        (1, "GET") => list(node, req).await,
        (1, _) => method_not_allowed(req, "GET"),
        (2, "POST") => import(node, req, &segs[1]).await,
        (2, "DELETE") => delete(node, req, &segs[1]).await,
        (2, _) => method_not_allowed(req, "POST, DELETE"),
        _ => RestResp::error(400, "illegal_argument_exception", &format!("no handler found for uri [{}] and method [{}]", req.uri, req.method)),
    }
}

async fn list(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    if let Some(r) = check_params(req, &[]) {
        return r;
    }
    let (total, ok, failed) = gather(node).await;
    let state = node.state();
    RestResp::json(200, &list_json(&state.cluster_name, total, &merge(&ok), &failed))
}

/// Shared prologue of import and delete: params, the dangling copy and
/// the nodes holding it.
async fn find(node: &Arc<ClusterNode>, req: &RestReq, uuid: &str) -> Result<(Dangling, Vec<String>), RestResp> {
    if let Some(r) = check_params(req, &["accept_data_loss", "master_timeout", "timeout"]) {
        return Err(r);
    }
    let accept = accept_data_loss(req)?;
    let (_, ok, _) = gather(node).await;
    let Some(found) = merge(&ok).into_iter().find(|(d, _)| d.uuid == uuid) else {
        return Err(iae(&format!("No dangling index found for UUID [{uuid}]")));
    };
    if !accept {
        return Err(iae("accept_data_loss must be set to true"));
    }
    Ok(found)
}

/// `POST /_dangling/{uuid}?accept_data_loss=true` (ES
/// `TransportImportDanglingIndexAction`): the elected master imports the
/// index with its existing copies ([`import_on_master`]).
async fn import(node: &Arc<ClusterNode>, req: &RestReq, uuid: &str) -> RestResp {
    if let Err(r) = find(node, req, uuid).await {
        return r;
    }
    let target = uuid.to_string();
    let result = if node.is_master() { import_on_master(node, uuid).await } else { node.to_master::<_, u64>(ACTION_IMPORT, &target, Duration::from_secs(60)).await };
    match result {
        Ok(_) => RestResp::json(202, &json!({"acknowledged": true})),
        Err(e) => ApiError::simple(e.status, &e.error_type, &e.reason).resp(),
    }
}

async fn delete_on(node: &Arc<ClusterNode>, id: &str, uuid: &str) -> bool {
    if id == node.local.id {
        if let Ok(true) = delete_quarantined(node, uuid) {
            return true;
        }
        let Ok(copies) = local_dangling(node) else { return false };
        let Some(d) = copies.into_iter().find(|d| d.uuid == uuid) else { return false };
        let ok = crate::meta_extra::delete_local_copy(&node.engine, &d.name).is_ok();
        node.uuids.lock().indices.remove(d.name.as_str());
        return ok;
    }
    let Some(addr) = node.address_of(id) else { return false };
    match node.transport.send_request::<_, bool>(addr, ACTION_DELETE_V2, &uuid, Duration::from_secs(30)).await {
        Ok(deleted) => deleted,
        // A holder of the previous release knows only the legacy action;
        // this build sends it only for an explicit delete, never an import.
        Err(elasticsearch_transport::TransportError::ActionNotFound { .. }) =>
            node.transport.send_request::<_, bool>(addr, ACTION_DELETE, &uuid, Duration::from_secs(30)).await.unwrap_or(false),
        Err(_) => false,
    }
}

async fn delete(node: &Arc<ClusterNode>, req: &RestReq, uuid: &str) -> RestResp {
    let (d, holders) = match find(node, req, uuid).await {
        Ok(f) => f,
        Err(r) => return r,
    };
    // ES: the delete is a tombstone, so copies on nodes that are away now
    // are deleted when they come back. The holders delete theirs now.
    let request = (d.name.to_string(), uuid.to_string());
    let tombstoned = if node.is_master() { tombstone_on_master(node, &request.0, &request.1).await } else { node.to_master::<_, u64>(ACTION_TOMBSTONE, &request, Duration::from_secs(60)).await };
    if let Err(e) = tombstoned {
        return ApiError::simple(e.status, &e.error_type, &e.reason).resp();
    }
    for id in &holders {
        delete_on(node, id, uuid).await;
    }
    RestResp::json(202, &json!({"acknowledged": true}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use elasticsearch_cluster::state::{IndexMetadata, IndexState};

    fn meta(name: &str, uuid: &str) -> IndexMetadata {
        IndexMetadata {
            name: name.into(),
            uuid: uuid.into(),
            version: 1,
            mapping_version: 1,
            settings_version: 1,
            aliases_version: 1,
            state: IndexState::Open,
            settings: Map::new().into(),
            mappings: json!({}).into(),
            aliases: BTreeMap::new(),
            number_of_shards: 1,
            number_of_replicas: 0,
            routing_num_shards: 1,
            primary_terms: vec![1],
            in_sync_allocation_ids: BTreeMap::new(),
            creation_date: 0,
            lifecycle: None,
        }
    }

    fn state_with(name: &str, uuid: &str) -> ClusterState {
        let mut s = ClusterState::empty("test");
        s.metadata.indices.insert(name.into(), meta(name, uuid));
        s
    }

    #[test]
    fn detects_unknown_and_mismatched_uuid() {
        let s = state_with("known", "U1");
        let local = vec![("known".to_string(), "L1".to_string(), 1), ("gone".to_string(), "L2".to_string(), 2), ("stale".to_string(), "L3".to_string(), 3)];
        let mut map = BTreeMap::new();
        map.insert("known".to_string(), ("U1".to_string(), "L1".to_string()));
        map.insert("stale".to_string(), ("OLD".to_string(), "L3".to_string()));
        let mut s2 = s.clone();
        s2.metadata.indices.insert("stale".into(), meta("stale", "NEW"));
        let d = find_dangling(&s2, &local, &map);
        assert_eq!(d, vec![("gone".to_string(), "L2".to_string(), 2), ("stale".to_string(), "OLD".to_string(), 3)]);
        assert!(find_dangling(&s, &local[..1], &map).is_empty());
    }

    fn dangling(name: &str, uuid: &str) -> Dangling {
        Dangling { name: SharedText::try_new(name).unwrap(), uuid: SharedText::try_new(uuid).unwrap(), creation_date: 1_700_000_000_000, settings: SharedText::try_new(r#"{"index.uuid":"x","index.number_of_shards":"2","index.provided_name":"a"}"#).unwrap(), mappings: SharedText::try_new(r#"{"properties":{}}"#).unwrap() }
    }

    #[test]
    fn merges_by_uuid_and_renders() {
        let mut per = BTreeMap::new();
        per.insert("n1".to_string(), SharedSequence::try_collect([dangling("a", "U")]).unwrap());
        per.insert("n2".to_string(), SharedSequence::try_collect([dangling("a", "U"), dangling("b", "V")]).unwrap());
        let merged = merge(&per);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].1, vec!["n1".to_string(), "n2".to_string()]);
        let v = list_json("c", 3, &merged, &[]);
        assert_eq!(v["_nodes"], json!({"total": 3, "successful": 3, "failed": 0}));
        assert_eq!(v["cluster_name"], "c");
        assert_eq!(v["dangling_indices"][0]["index_name"], "a");
        assert_eq!(v["dangling_indices"][0]["creation_date"], "2023-11-14T22:13:20.000Z");
        assert_eq!(v["dangling_indices"][0]["creation_date_millis"], 1_700_000_000_000i64);
        assert_eq!(list_json("c", 1, &[], &[])["dangling_indices"], json!([]));
    }

    /// An import brings the dangling copy back under its own uuid with
    /// every document (it is never deleted and recreated empty), and it is
    /// no longer listed as dangling.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn import_keeps_the_dangling_copy_and_its_documents() {
        use tower::ServiceExt;
        // The node outlives the test (its runtime is not shut down), so keep
        // its data directory: a deleted one makes cluster-persist exit the process.
        let dir: &'static tempfile::TempDir = Box::leak(Box::new(tempfile::tempdir().unwrap()));
        let mut settings = crate::config::Settings::new();
        for (key, value) in [
            ("cluster.name", json!("dangling-import")),
            ("node.name", json!("dangling-import")),
            ("path.data", json!(dir.path().join("data"))),
            ("path.logs", json!(dir.path().join("logs"))),
            ("network.host", json!("127.0.0.1")),
            ("http.port", json!("0")),
            ("transport.port", json!("0")),
            ("discovery.type", json!("single-node")),
            ("xpack.security.enabled", json!("false")),
        ] { settings.insert(key.into(), value); }
        let (node, _listener) = ClusterNode::start(crate::config::NodeConfig::from_settings(settings).unwrap()).await.unwrap();
        for _ in 0..300 {
            if node.is_master() && node.state().nodes.contains_key(&node.local.id) { break; }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert!(node.is_master(), "the single node never became master");
        // A copy on this node's disk the cluster metadata does not know.
        // Mapped up front: a dynamic update would go to the master.
        node.engine.create_index("orphan", &json!({"settings": {"number_of_shards": 1}, "mappings": {"properties": {"v": {"type": "long"}}}})).unwrap();
        for i in 0..5 {
            let r = axum::http::Request::builder().method("PUT").uri(format!("/orphan/_doc/{i}?refresh=true"))
                .header("content-type", "application/json").body(axum::body::Body::from(json!({"v": i}).to_string())).unwrap();
            let resp = node.router.clone().oneshot(r).await.unwrap();
            let status = resp.status();
            let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
            assert!(status.is_success(), "{status} {}", String::from_utf8_lossy(&body));
        }
        let uuid = node.engine.get_index("orphan").unwrap().uuid.clone();
        let listed = local_dangling(&node).unwrap();
        assert!(listed.into_iter().any(|d| d.uuid == uuid.as_str()), "the copy is dangling");

        let req = RestReq::try_new_uri("POST", format_args!("/_dangling/{uuid}?accept_data_loss=true"), &[], &[], 0, Default::default(), None).unwrap();
        let r = handle(&node, &req).await;
        assert_eq!(r.status, 202, "{}", String::from_utf8_lossy(&r.body));
        assert_eq!(r.body_json().unwrap(), json!({"acknowledged": true}));
        assert_eq!(node.state().metadata.indices.get("orphan").map(|im| im.uuid.clone()), Some(uuid.clone()), "imported under its own uuid");
        let mut started = false;
        for _ in 0..300 {
            let st = node.state();
            if st.routing_table.shard_table("orphan", 0).and_then(|t| t.active_primary()).is_some() { started = true; break; }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert!(started, "the imported primary never started");
        let svc = node.engine.get_index("orphan").expect("the copy is the index");
        assert_eq!(svc.uuid, uuid, "the same copy, not a recreated one");
        assert_eq!(svc.doc_count_realtime().0, 5, "every document survives");
        assert!(!local_dangling(&node).unwrap().into_iter().any(|d| d.uuid == uuid.as_str()), "no longer dangling");
        // A second import is refused: the name exists.
        let r = handle(&node, &req).await;
        assert_eq!(r.status, 400, "{}", String::from_utf8_lossy(&r.body));

        // Another dangling copy, for the delete actions.
        node.engine.create_index("orphan2", &json!({"settings": {"number_of_shards": 1}})).unwrap();
        let uuid = node.engine.get_index("orphan2").unwrap().uuid.clone();
        // A coordinator of the previous release sends the legacy delete for
        // DELETE /_dangling and as the first step of its (data-losing)
        // import alike: it is refused and the copy stays. The v2 delete,
        // sent only for DELETE /_dangling, removes it. `ClusterNode::start`
        // does not install the REST-layer transport handlers (`lib::run`
        // does, through `rest::register`), so install this module's here.
        register(&node);
        let mut peer = node.local.clone();
        peer.id = "older-coordinator".into();
        peer.ephemeral_id = "older-coordinator-process".into();
        peer.transport_address = "127.0.0.1:0".into();
        let client = elasticsearch_transport::TransportService::bind(elasticsearch_transport::TransportConfig::new("dangling-import", peer, "127.0.0.1:0".parse().unwrap())).await.unwrap();
        let addr: std::net::SocketAddr = node.local.transport_address.parse().unwrap();
        let target = SharedText::try_new(uuid.as_str()).unwrap();
        let legacy = client.send_request::<_, bool>(addr, ACTION_DELETE, &target, Duration::from_secs(10)).await;
        assert!(legacy.as_ref().is_err_and(|e| e.to_string().contains("older version")), "{legacy:?}");
        assert!(node.engine.get_index("orphan2").is_some(), "the legacy delete keeps the copy");
        assert!(client.send_request::<_, bool>(addr, ACTION_DELETE_V2, &target, Duration::from_secs(10)).await.unwrap());
        assert!(node.engine.get_index("orphan2").is_none(), "an explicit delete still deletes");
        assert!(node.engine.get_index("orphan").is_some(), "the imported index is not dangling");
        client.close().await;
    }
}
