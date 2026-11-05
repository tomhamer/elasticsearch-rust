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

pub fn register(node: &Arc<ClusterNode>) {
    let n = node.clone();
    node.transport.register_handler(ACTION_LIST, move |_: ()| {
        let n = n.clone();
        async move { local_dangling(&n).map_err(super::wire_memory) }
    });
    let n = node.clone();
    node.transport.register_handler(ACTION_DELETE_V2, move |uuid: SharedText| {
        let n = n.clone();
        async move {
            let Some(d) = local_dangling(&n).map_err(super::wire_memory)?.into_iter().find(|d| d.uuid == uuid) else { return Ok::<_, RemoteError>(false) };
            crate::meta_extra::delete_local_copy(&n.engine, &d.name).map_err(|e| RemoteError::new("illegal_state_exception", e, 500))?;
            n.uuids.lock().indices.remove(d.name.as_str());
            Ok(true)
        }
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
            match local_dangling(node) {
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
            by_uuid.entry(d.uuid.clone()).or_insert_with(|| (d.clone(), Vec::new())).1.push(id.clone());
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

/// `POST /_dangling/{uuid}`. ES (`LocalAllocateDangledIndices`) adds the
/// copy's own metadata under its original uuid and allocates the primaries
/// from the stores already on disk; nothing is deleted. Here the cluster uuid
/// of an index is minted by the master's engine and a dangling copy carries
/// no in-sync allocation ids, so adopting a store of unknown provenance could
/// resurrect stale data. The import is therefore refused by design and the
/// copy stays where it is, still listed and still deletable: no data is ever
/// destroyed by an import. Restore from a snapshot to bring the data back.
async fn import(node: &Arc<ClusterNode>, req: &RestReq, uuid: &str) -> RestResp {
    let (d, holders) = match find(node, req, uuid).await {
        Ok(f) => f,
        Err(r) => return r,
    };
    iae(&format!(
        "cannot import dangling index [{}/{}]: importing a dangling index with its existing data is not supported by this cluster; the copy on nodes [{}] was left in place",
        d.name, d.uuid, holders.join(", ")
    ))
}

async fn delete_on(node: &Arc<ClusterNode>, id: &str, uuid: &str) -> bool {
    if id == node.local.id {
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
    let (_, holders) = match find(node, req, uuid).await {
        Ok(f) => f,
        Err(r) => return r,
    };
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

    /// An import never loses the dangling copy's documents: the copy on
    /// disk is kept, with every document, and is still listed afterwards
    /// (it used to be deleted and the index recreated empty).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn import_keeps_the_dangling_copy_and_its_documents() {
        use tower::ServiceExt;
        let dir = tempfile::tempdir().unwrap();
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
        assert_eq!(r.status, 400, "{}", String::from_utf8_lossy(&r.body));
        let v = r.body_json().unwrap();
        assert_eq!(v["error"]["type"], "illegal_argument_exception", "{v}");
        assert!(v["error"]["reason"].as_str().unwrap().contains("left in place"), "{v}");

        let svc = node.engine.get_index("orphan").expect("the dangling copy survives the import");
        assert_eq!(svc.uuid, uuid, "the same copy, not a recreated one");
        assert_eq!(svc.doc_count_realtime().0, 5, "every document survives");
        assert!(local_dangling(&node).unwrap().into_iter().any(|d| d.uuid == uuid.as_str()), "still listed");
        assert!(node.state().metadata.indices.get("orphan").is_none(), "no empty index was created in its place");

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
        assert_eq!(node.engine.get_index("orphan").expect("the legacy delete keeps the copy").doc_count_realtime().0, 5);
        assert!(client.send_request::<_, bool>(addr, ACTION_DELETE_V2, &target, Duration::from_secs(10)).await.unwrap());
        assert!(node.engine.get_index("orphan").is_none(), "an explicit delete still deletes");
        client.close().await;
    }
}
