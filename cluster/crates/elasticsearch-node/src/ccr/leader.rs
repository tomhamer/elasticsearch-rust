//! The leader side of CCR: actions a follower cluster calls over its
//! remote-cluster connection (ES `ShardChangesAction`,
//! `RetentionLeaseActions`, `PutCcrRestoreSessionAction`,
//! `GetCcrRestoreFileChunkAction`, `ClearCcrRestoreSessionAction` and the
//! remote cluster state request).
//!
//! The node a follower reaches (a gateway) authorizes a remote caller as
//! the user it names, with this cluster's role definitions, and runs the
//! request where the data is: shard changes and restore sessions on the
//! leader shard's primary, retention leases on every active copy. The hop
//! inside this cluster (`routed`) goes over the cluster's own transport and
//! is not authorized again.

use crate::node::ClusterNode;
use crate::remote_cluster::{remote_error, to_remote, transport_error};
use elasticsearch_cluster::state::{ClusterState, IndexState};
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::hooks::changes::ChangeOp;
use elasticsearch_engine::index::IndexShard;
use elasticsearch_transport::{RemoteError, RemotePeer};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

pub const ACTION_METADATA: &str = "cluster:monitor/xpack/ccr/leader_metadata";
pub const ACTION_SHARD_CHANGES: &str = "indices:data/read/xpack/ccr/shard_changes";
pub const ACTION_LEASE: &str = "indices:admin/seq_no/ccr_retention_lease";
pub const ACTION_SESSION_PUT: &str = "internal:admin/ccr/restore/session/put";
pub const ACTION_FILE_CHUNK: &str = "internal:admin/ccr/restore/file_chunk/get";
pub const ACTION_SESSION_CLEAR: &str = "internal:admin/ccr/restore/session/clear";

/// `ccr.indices.recovery.recovery_activity_timeout` (60s): an idle restore
/// session is released.
const SESSION_IDLE: Duration = Duration::from_secs(60);

// ------------------------------------------------------------- wire types

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MetadataReq {
    pub identity: Option<String>,
    /// Exact names to describe; `None`: every index (auto-follow).
    pub names: Option<Vec<String>>,
    /// Check the follow privileges on the named indices (put follow).
    pub privileges: bool,
}

/// A leader index as a follower needs it (JSON as text: bincode carries no
/// self-describing values).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LeaderIndex {
    pub name: String,
    pub uuid: String,
    pub open: bool,
    pub shards: u32,
    pub routing_shards: u32,
    /// Flat settings.
    pub settings_json: String,
    pub mappings_json: String,
    pub aliases_json: String,
    pub mapping_version: u64,
    pub settings_version: u64,
    pub aliases_version: u64,
    pub primaries_active: bool,
    pub primary_term: u64,
    pub data_stream: Option<String>,
}

impl LeaderIndex {
    pub fn settings(&self) -> elasticsearch_engine::settings::FlatSettings {
        serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&self.settings_json).map(|m| m.into_iter().collect()).unwrap_or_default()
    }
    pub fn setting(&self, key: &str) -> Option<String> {
        self.settings().get(key).map(|v| match v {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        })
    }
    pub fn mappings(&self) -> serde_json::Value {
        serde_json::from_str(&self.mappings_json).unwrap_or(serde_json::Value::Null)
    }
    pub fn aliases(&self) -> BTreeMap<String, serde_json::Value> {
        serde_json::from_str(&self.aliases_json).unwrap_or_default()
    }
    pub fn soft_deletes(&self) -> bool {
        self.setting("index.soft_deletes.enabled").as_deref() != Some("false")
    }
    pub fn searchable_snapshot(&self) -> bool {
        self.setting("index.store.type").as_deref() == Some("snapshot")
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MetadataResp {
    pub cluster_name: String,
    pub metadata_version: u64,
    pub indices: Vec<LeaderIndex>,
    /// Requested names that are not concrete indices: (name, `ALIAS` |
    /// `DATA_STREAM`).
    pub abstractions: Vec<(String, String)>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ShardChangesReq {
    pub identity: Option<String>,
    pub index: String,
    pub uuid: String,
    pub shard: u32,
    pub from_seq_no: i64,
    pub max_operation_count: u64,
    pub max_batch_size: u64,
    pub poll_timeout_ms: u64,
    pub routed: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ShardChangesResp {
    pub operations: Vec<ChangeOp>,
    pub global_checkpoint: i64,
    pub max_seq_no: i64,
    pub max_seq_no_of_updates: i64,
    pub mapping_version: u64,
    pub settings_version: u64,
    pub aliases_version: u64,
    pub took_millis: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeaseOp {
    /// Add (renew when it exists): restore and follow.
    Add,
    /// Renew (add when missing): the follow task's periodic renewal.
    Renew,
    Remove,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeaseReq {
    pub identity: Option<String>,
    pub index: String,
    pub uuid: String,
    /// `None`: every shard.
    pub shard: Option<u32>,
    pub op: LeaseOp,
    pub id: String,
    pub retaining_seq_no: i64,
    pub routed: bool,
}

/// One copy's result: (shard, primary, error type and reason).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LeaseResp {
    pub results: Vec<(u32, bool, Option<(String, String)>)>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SessionPutReq {
    pub identity: Option<String>,
    pub index: String,
    pub uuid: String,
    pub shard: u32,
    pub lease_id: String,
    pub routed: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SessionPutResp {
    pub session: String,
    pub node_id: String,
    /// The pinned commit's `IndexManifest` (paths relative to the shard).
    pub manifest_json: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ChunkReq {
    pub identity: Option<String>,
    pub index: String,
    pub session: String,
    pub node_id: String,
    pub path: String,
    pub offset: u64,
    pub len: u64,
    pub routed: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ChunkResp {
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SessionClearReq {
    pub identity: Option<String>,
    pub index: String,
    pub session: String,
    pub node_id: String,
    pub routed: bool,
}

// ---------------------------------------------------------- authorization

/// Authorize a remote cluster's request as the user it names (security on).
/// `check` gets the permission and the user's name and roles.
fn authorize(node: &ClusterNode, peer: &Option<RemotePeer>, identity: &Option<String>, check: impl FnOnce(&elasticsearch_engine::security::Permission) -> Result<(), String>) -> Result<(), EsError> {
    if peer.is_none() || !node.engine.security.enabled() {
        return Ok(());
    }
    // Certificate-based (the user's roles here) or API key based (the
    // user's granted descriptors limited by the cross-cluster API key).
    let Some(stored) = crate::remote_cluster::authenticate_remote(node, peer, identity)? else {
        return Err(EsError::new(401, "security_exception", "missing authentication credentials for a cross-cluster replication request"));
    };
    let authn = stored.to_authn();
    let perm = node.engine.security.permission(&authn);
    check(&perm).map_err(|why| {
        EsError::new(403, "security_exception", format!("{why} for user [{}] with effective roles [{}]", authn.username, perm.effective_roles.join(",")))
    })
}

fn index_action(perm: &elasticsearch_engine::security::Permission, action: &str, index: &str) -> Result<(), String> {
    if perm.check_index(action, index) {
        Ok(())
    } else {
        Err(format!("action [{action}] is unauthorized on indices [{index}]"))
    }
}

fn es_remote(e: EsError) -> RemoteError {
    to_remote(&e)
}

// --------------------------------------------------------------- metadata

fn leader_index(st: &ClusterState, name: &str) -> LeaderIndex {
    let im = &st.metadata.indices[name];
    let ds = st.metadata.customs.get(crate::mirror::DATA_STREAMS_CUSTOM).and_then(|c| {
        let streams = c.get("data_streams").or(Some(c))?.as_object()?;
        streams.iter().find(|(_, d)| d["indices"].as_array().is_some_and(|a| a.iter().any(|i| i.as_str() == Some(name) || i["index_name"].as_str() == Some(name)))).map(|(k, _)| k.clone())
    });
    LeaderIndex {
        name: name.to_string(),
        uuid: im.uuid.clone(),
        open: im.state == IndexState::Open,
        shards: im.number_of_shards.max(1),
        routing_shards: im.routing_num_shards,
        settings_json: serde_json::Value::Object((*im.settings).clone()).to_string(),
        mappings_json: (*im.mappings).to_string(),
        aliases_json: serde_json::to_string(&im.aliases).unwrap_or_else(|_| "{}".into()),
        mapping_version: im.mapping_version,
        settings_version: im.settings_version,
        aliases_version: im.aliases_version,
        primaries_active: (0..im.number_of_shards.max(1)).all(|k| st.routing_table.shard_table(name, k).and_then(|t| t.active_primary()).is_some()),
        primary_term: im.primary_terms.iter().copied().max().unwrap_or(1),
        data_stream: ds,
    }
}

fn abstraction(node: &ClusterNode, st: &ClusterState, name: &str) -> Option<&'static str> {
    if st.metadata.customs.get(crate::mirror::DATA_STREAMS_CUSTOM).is_some_and(|c| c.get("data_streams").unwrap_or(c).get(name).is_some()) {
        return Some("DATA_STREAM");
    }
    if st.metadata.indices.values().any(|im| im.aliases.contains_key(name)) || node.engine.all_indices().iter().any(|s| s.aliases().contains_key(name)) {
        return Some("ALIAS");
    }
    None
}

async fn metadata(node: Arc<ClusterNode>, req: MetadataReq, peer: Option<RemotePeer>) -> Result<MetadataResp, RemoteError> {
    authorize(&node, &peer, &req.identity, |perm| {
        if !perm.check_cluster("cluster:monitor/state") {
            return Err("action [cluster:monitor/state] is unauthorized".into());
        }
        if req.privileges {
            for n in req.names.iter().flatten() {
                for action in ["indices:monitor/stats", ACTION_SHARD_CHANGES] {
                    if !perm.check_index(action, n) {
                        return Err(format!("insufficient privileges to follow index [{n}], privilege for action [{action}] is missing"));
                    }
                }
            }
        }
        Ok(())
    })
    .map_err(|mut e| {
        // ES reports a missing follow privilege as this exact message.
        if e.reason.starts_with("insufficient privileges") {
            if let Some(cut) = e.reason.find(" for user [") {
                e.reason.truncate(cut);
            }
        }
        es_remote(e)
    })?;
    let st = node.state();
    let mut out = MetadataResp { cluster_name: st.cluster_name.clone(), metadata_version: st.metadata.version, ..Default::default() };
    match &req.names {
        Some(names) => {
            for n in names {
                if st.metadata.indices.contains_key(n) {
                    out.indices.push(leader_index(&st, n));
                } else if let Some(t) = abstraction(&node, &st, n) {
                    out.abstractions.push((n.clone(), t.to_string()));
                }
            }
        }
        None => {
            for n in st.metadata.indices.keys() {
                out.indices.push(leader_index(&st, n));
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- routing

/// The node holding the primary of `index` shard `shard`.
fn primary_node(st: &ClusterState, index: &str, shard: u32) -> Option<String> {
    st.routing_table.shard_table(index, shard)?.active_primary()?.node_id.clone()
}

fn check_index(st: &ClusterState, index: &str, uuid: &str) -> Result<(), EsError> {
    match st.metadata.indices.get(index) {
        Some(im) if uuid.is_empty() || im.uuid == uuid => {
            if im.state != IndexState::Open {
                return Err(EsError::index_closed(index, &im.uuid));
            }
            Ok(())
        }
        _ => Err(EsError::index_not_found(index)),
    }
}

fn no_primary(index: &str, shard: u32) -> EsError {
    EsError::new(503, "no_shard_available_action_exception", format!("[{index}][{shard}] no active primary shard copy"))
}

/// Forward a request inside this cluster to `node_id`.
async fn hop<Req: Serialize + Sync, Resp: serde::de::DeserializeOwned>(node: &ClusterNode, node_id: &str, action: &str, req: &Req, timeout: Duration) -> Result<Resp, RemoteError> {
    let addr = node.address_of(node_id).ok_or_else(|| RemoteError::new("node_not_connected_exception", format!("[{node_id}] node not connected"), 502))?;
    node.transport.send_request::<Req, Resp>(addr, action, req, timeout).await.map_err(|e| to_remote(&transport_error(&e, node_id)))
}

// ----------------------------------------------------------- shard changes

fn local_primary(node: &ClusterNode, index: &str, shard: u32) -> Result<Arc<crate::shards::PrimaryCtx>, EsError> {
    node.shards.primary(index, shard).ok_or_else(|| no_primary(index, shard))
}

/// ES `ShardChangesAction`'s missing-history failure.
fn missing_history(copy: &IndexShard, e: &EsError) -> EsError {
    let leases: Vec<String> = copy.shard.retention_leases().iter().map(|l| format!("RetentionLease{{id='{}', retainingSequenceNumber={}, timestamp={}, source='{}'}}", l.id, l.retaining_seq_no, l.timestamp_millis, l.source)).collect();
    EsError::resource_not_found(format!(
        "Operations are no longer available for replicating. Existing retention leases [RetentionLeases{{primaryTerm={}, version=0, leases={{{}}}}}]; maybe increase the retention lease period setting [index.soft_deletes.retention_lease.period]? ({})",
        copy.shard.primary_term(),
        leases.join(", "),
        e.reason
    ))
    .with("es.requested_operations_missing", "true")
}

async fn shard_changes_local(node: &ClusterNode, req: &ShardChangesReq) -> Result<ShardChangesResp, EsError> {
    let started = Instant::now();
    if req.from_seq_no < 0 {
        return Err(EsError::action_validation(&[format!("fromSeqNo [{}] cannot be lower than 0", req.from_seq_no).as_str()]));
    }
    if req.max_batch_size == 0 {
        return Err(EsError::action_validation(&[format!("maxBatchSize [{}b] must be larger than 0", req.max_batch_size).as_str()]));
    }
    let poll = Duration::from_millis(req.poll_timeout_ms);
    loop {
        let st = node.state();
        check_index(&st, &req.index, &req.uuid)?;
        let p = local_primary(node, &req.index, req.shard)?;
        let copy = p.copy();
        let gcp = p.global_checkpoint();
        let im = &st.metadata.indices[&req.index];
        let mut resp = ShardChangesResp {
            operations: Vec::new(),
            global_checkpoint: gcp,
            max_seq_no: copy.shard.max_seq_no(),
            max_seq_no_of_updates: copy.shard.max_seq_no_of_updates(),
            mapping_version: im.mapping_version,
            settings_version: im.settings_version,
            aliases_version: im.aliases_version,
            took_millis: 0,
        };
        if req.from_seq_no <= gcp {
            let from = req.from_seq_no;
            let max_ops = req.max_operation_count.max(1) as usize;
            let to = gcp.min(from.saturating_add(max_ops as i64 - 1));
            let max_bytes = req.max_batch_size;
            let c2 = copy.clone();
            let ops = tokio::task::spawn_blocking(move || -> Result<Vec<ChangeOp>, EsError> {
                let mut cursor = None;
                let snap = c2.shard.changes_from_cursor_sized(&mut cursor, from, to, max_ops, max_bytes, elasticsearch_engine::ccr::op_size)?;
                Ok(snap.map(|item| ChangeOp::clone(&item)).filter(|o| o.seq_no() <= to).collect())
            })
            .await
            .map_err(|e| EsError::internal(e.to_string()))?;
            resp.operations = match ops {
                Ok(o) => o,
                Err(e) if e.kind == "missing_history_operations_exception" => return Err(missing_history(&copy, &e)),
                Err(e) => return Err(e),
            };
            // An incomplete read (history not contiguous from `from`) is
            // reported as missing operations, as ES does.
            if resp.operations.first().is_some_and(|o| o.seq_no() != from) {
                let e = EsError::new(500, "missing_history_operations_exception", format!("expected seq_no [{from}]"));
                return Err(missing_history(&copy, &e));
            }
            resp.took_millis = started.elapsed().as_millis() as u64;
            return Ok(resp);
        }
        if started.elapsed() >= poll {
            resp.took_millis = started.elapsed().as_millis() as u64;
            return Ok(resp);
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn shard_changes(node: Arc<ClusterNode>, req: ShardChangesReq, peer: Option<RemotePeer>) -> Result<ShardChangesResp, RemoteError> {
    authorize(&node, &peer, &req.identity, |perm| index_action(perm, ACTION_SHARD_CHANGES, &req.index)).map_err(es_remote)?;
    let st = node.state();
    check_index(&st, &req.index, &req.uuid).map_err(es_remote)?;
    let target = primary_node(&st, &req.index, req.shard).ok_or_else(|| es_remote(no_primary(&req.index, req.shard)))?;
    if target == node.local.id {
        return shard_changes_local(&node, &req).await.map_err(es_remote);
    }
    if req.routed {
        return Err(es_remote(no_primary(&req.index, req.shard)));
    }
    let mut fwd = req.clone();
    fwd.routed = true;
    hop(&node, &target, ACTION_SHARD_CHANGES, &fwd, Duration::from_millis(req.poll_timeout_ms) + Duration::from_secs(60)).await
}

// ------------------------------------------------------- retention leases

fn lease_on_copy(copy: &IndexShard, op: LeaseOp, id: &str, retaining: i64) -> Result<(), EsError> {
    match op {
        LeaseOp::Add => match copy.shard.add_retention_lease(id, retaining, "ccr") {
            Ok(_) => Ok(()),
            Err(e) if e.kind == "retention_lease_already_exists_exception" => copy.shard.renew_retention_lease(id, retaining, "ccr").map(|_| ()),
            Err(e) => Err(e),
        },
        LeaseOp::Renew => copy.shard.renew_retention_lease(id, retaining, "ccr").map(|_| ()),
        LeaseOp::Remove => match copy.shard.remove_retention_lease(id) {
            Ok(true) => Ok(()),
            Ok(false) => Err(EsError::new(404, "retention_lease_not_found_exception", format!("retention lease with ID [{id}] not found"))),
            Err(e) => Err(e),
        },
    }
}

/// The lease change on this node's copies of the shards.
fn lease_local(node: &ClusterNode, st: &ClusterState, req: &LeaseReq) -> LeaseResp {
    let n = st.metadata.indices.get(&req.index).map(|im| im.number_of_shards.max(1)).unwrap_or(0);
    let mut out = LeaseResp::default();
    for k in (0..n).filter(|k| req.shard.is_none_or(|s| s == *k)) {
        let Some(slot) = node.shards.get(&req.index, k) else { continue };
        let (svc, primary) = {
            let g = slot.lock();
            (g.svc.clone(), matches!(g.role, crate::shards::Role::Primary(_)))
        };
        let Some(copy) = svc.shard(k) else { continue };
        let r = lease_on_copy(&copy, req.op, &req.id, req.retaining_seq_no);
        out.results.push((k, primary, r.err().map(|e| (e.kind, e.reason))));
    }
    out
}

async fn lease(node: Arc<ClusterNode>, req: LeaseReq, peer: Option<RemotePeer>) -> Result<LeaseResp, RemoteError> {
    let action = match req.op {
        LeaseOp::Add => "indices:admin/seq_no/add_retention_lease",
        LeaseOp::Renew => "indices:admin/seq_no/renew_retention_lease",
        LeaseOp::Remove => "indices:admin/seq_no/remove_retention_lease",
    };
    authorize(&node, &peer, &req.identity, |perm| {
        // `read` covers the follower's own leases (ES's leader-side follow
        // privileges); `manage_leader_index` covers forget_follower.
        if perm.check_index(action, &req.index) || perm.check_index(ACTION_SHARD_CHANGES, &req.index) {
            Ok(())
        } else {
            Err(format!("action [{action}] is unauthorized on indices [{}]", req.index))
        }
    })
    .map_err(es_remote)?;
    let st = node.state();
    match st.metadata.indices.get(&req.index) {
        Some(im) if req.uuid.is_empty() || im.uuid == req.uuid => {}
        _ => return Err(es_remote(EsError::index_not_found(&req.index))),
    }
    if req.routed {
        return Ok(lease_local(&node, &st, &req));
    }
    // Every node holding an active copy of the shards.
    let n = st.metadata.indices[&req.index].number_of_shards.max(1);
    let mut nodes: Vec<String> = Vec::new();
    for k in (0..n).filter(|k| req.shard.is_none_or(|s| s == *k)) {
        for c in st.routing_table.shard_table(&req.index, k).map(|t| t.copies.clone()).unwrap_or_default() {
            if c.active() {
                if let Some(id) = c.node_id {
                    if !nodes.contains(&id) {
                        nodes.push(id);
                    }
                }
            }
        }
    }
    let mut out = LeaseResp::default();
    let mut fwd = req.clone();
    fwd.routed = true;
    let answers = futures::future::join_all(nodes.iter().map(|id| {
        let (node, fwd, st) = (node.clone(), fwd.clone(), st.clone());
        async move {
            if *id == node.local.id {
                Ok(lease_local(&node, &st, &fwd))
            } else {
                hop::<_, LeaseResp>(&node, id, ACTION_LEASE, &fwd, Duration::from_secs(30)).await
            }
        }
    }))
    .await;
    for a in answers {
        match a {
            Ok(r) => out.results.extend(r.results),
            Err(e) => tracing::debug!(index = %req.index, "retention lease change on a leader copy failed: {}", e.reason),
        }
    }
    Ok(out)
}

// ------------------------------------------------------- restore sessions

struct Session {
    files: elasticsearch_engine::hooks::files::IndexFileSet,
    last: Instant,
}

fn sessions() -> &'static Mutex<HashMap<String, Arc<Mutex<Session>>>> {
    static S: OnceLock<Mutex<HashMap<String, Arc<Mutex<Session>>>>> = OnceLock::new();
    S.get_or_init(Default::default)
}

async fn session_put(node: Arc<ClusterNode>, req: SessionPutReq, peer: Option<RemotePeer>) -> Result<SessionPutResp, RemoteError> {
    authorize(&node, &peer, &req.identity, |perm| index_action(perm, ACTION_SHARD_CHANGES, &req.index)).map_err(es_remote)?;
    let st = node.state();
    check_index(&st, &req.index, &req.uuid).map_err(es_remote)?;
    let target = primary_node(&st, &req.index, req.shard).ok_or_else(|| es_remote(no_primary(&req.index, req.shard)))?;
    if target != node.local.id {
        if req.routed {
            return Err(es_remote(no_primary(&req.index, req.shard)));
        }
        let mut fwd = req.clone();
        fwd.routed = true;
        return hop(&node, &target, ACTION_SESSION_PUT, &fwd, Duration::from_secs(120)).await;
    }
    let p = local_primary(&node, &req.index, req.shard).map_err(es_remote)?;
    let copy = p.copy();
    // Flush first (ES restores the leader's last commit): the follower then
    // replays few operations.
    let c2 = copy.clone();
    // Writes hold the gate's read side: the flush and the pin see a quiet copy.
    let _gate = p.gate.write().await;
    let files = tokio::task::spawn_blocking(move || -> Result<_, EsError> {
        crate::shards::flush_within_graph_lag(&c2, 50_000).map_err(|e| EsError::new(500, "flush_failed_engine_exception", e))?;
        c2.snapshot_files(true)
    })
    .await
    .map_err(|e| es_remote(EsError::internal(e.to_string())))?
    .map_err(es_remote)?;
    drop(_gate);
    let retain = files.manifest.local_checkpoint + 1;
    let manifest_json = serde_json::to_string(&files.manifest).map_err(|e| es_remote(EsError::internal(e.to_string())))?;
    // The follower's lease on every copy, before the pinned commit's
    // retention lock goes with the session.
    let lr = LeaseReq { identity: None, index: req.index.clone(), uuid: req.uuid.clone(), shard: Some(req.shard), op: LeaseOp::Add, id: req.lease_id.clone(), retaining_seq_no: retain, routed: false };
    let _ = lease(node.clone(), lr, None).await;
    let id = elasticsearch_snapshots::util::random_base64_uuid();
    sessions().lock().insert(id.clone(), Arc::new(Mutex::new(Session { files, last: Instant::now() })));
    tracing::info!(index = %req.index, shard = req.shard, session = %id, "opened a CCR restore session");
    Ok(SessionPutResp { session: id, node_id: node.local.id.clone(), manifest_json })
}

async fn file_chunk(node: Arc<ClusterNode>, req: ChunkReq, peer: Option<RemotePeer>) -> Result<ChunkResp, RemoteError> {
    authorize(&node, &peer, &req.identity, |perm| index_action(perm, ACTION_SHARD_CHANGES, &req.index)).map_err(es_remote)?;
    if req.node_id != node.local.id {
        if req.routed {
            return Err(RemoteError::new("illegal_state_exception", format!("restore session [{}] is not on this node", req.session), 500));
        }
        let mut fwd = req.clone();
        fwd.routed = true;
        let target = req.node_id.clone();
        return hop(&node, &target, ACTION_FILE_CHUNK, &fwd, Duration::from_secs(120)).await;
    }
    let s = sessions().lock().get(&req.session).cloned().ok_or_else(|| RemoteError::new("resource_not_found_exception", format!("missing session [{}]", req.session), 404))?;
    let (path, off, len) = (req.path.clone(), req.offset, req.len);
    let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, EsError> {
        let mut g = s.lock();
        g.last = Instant::now();
        let a = g.files.read_range(&path, off, len)?;
        Ok(a.to_vec())
    })
    .await
    .map_err(|e| es_remote(EsError::internal(e.to_string())))?
    .map_err(es_remote)?;
    Ok(ChunkResp { bytes })
}

async fn session_clear(node: Arc<ClusterNode>, req: SessionClearReq, peer: Option<RemotePeer>) -> Result<(), RemoteError> {
    authorize(&node, &peer, &req.identity, |perm| index_action(perm, ACTION_SHARD_CHANGES, &req.index)).map_err(es_remote)?;
    if req.node_id != node.local.id {
        if req.routed {
            return Ok(());
        }
        let mut fwd = req.clone();
        fwd.routed = true;
        let target = req.node_id.clone();
        return hop(&node, &target, ACTION_SESSION_CLEAR, &fwd, Duration::from_secs(30)).await;
    }
    sessions().lock().remove(&req.session);
    Ok(())
}

// ---------------------------------------------------------- registration

pub fn register(node: &Arc<ClusterNode>) {
    let t = node.transport.clone();
    macro_rules! serve {
        ($action:expr, $req:ty, $f:ident) => {{
            let n = node.clone();
            t.register_handler($action, move |req: $req| {
                let peer = elasticsearch_transport::inbound_remote_peer();
                let n = n.clone();
                async move { $f(n, req, peer).await }
            });
            t.allow_remote_action($action);
        }};
    }
    serve!(ACTION_METADATA, MetadataReq, metadata);
    serve!(ACTION_SHARD_CHANGES, ShardChangesReq, shard_changes);
    serve!(ACTION_LEASE, LeaseReq, lease);
    serve!(ACTION_SESSION_PUT, SessionPutReq, session_put);
    serve!(ACTION_FILE_CHUNK, ChunkReq, file_chunk);
    serve!(ACTION_SESSION_CLEAR, SessionClearReq, session_clear);
    // Idle restore sessions release their pinned commits.
    let weak = Arc::downgrade(node);
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(10)).await;
            if weak.strong_count() == 0 {
                return;
            }
            sessions().lock().retain(|id, s| {
                let keep = s.try_lock().map(|g| g.last.elapsed() < SESSION_IDLE).unwrap_or(true);
                if !keep {
                    tracing::info!(session = %id, "released an idle CCR restore session");
                }
                keep
            });
        }
    });
}

// ------------------------------------------------------- follower helpers

/// Send a leader action to remote cluster `alias` (connecting first).
pub async fn call<Req: Serialize + Sync, Resp: serde::de::DeserializeOwned>(node: &ClusterNode, alias: &str, action: &str, req: &Req, timeout: Duration) -> Result<Resp, EsError> {
    let svc = crate::remote_cluster::service(node).ok_or_else(|| elasticsearch_engine::ccr::remote::connection_error(&node.engine, alias))?;
    svc.send(alias, action, req, Some(timeout), true).await
}

/// The leader cluster's metadata of `names` (`None`: every index).
pub async fn fetch_metadata(node: &ClusterNode, alias: &str, identity: Option<String>, names: Option<Vec<String>>, privileges: bool) -> Result<MetadataResp, EsError> {
    let identity = crate::remote_cluster::outgoing_identity(node, alias, identity);
    call(node, alias, ACTION_METADATA, &MetadataReq { identity, names, privileges }, Duration::from_secs(60)).await
}

/// Add / renew / remove a lease on every copy of a leader index's shards.
#[allow(clippy::too_many_arguments)]
pub async fn change_lease(node: &ClusterNode, alias: &str, identity: Option<String>, index: &str, uuid: &str, shard: Option<u32>, op: LeaseOp, id: &str, retaining: i64) -> Result<LeaseResp, EsError> {
    let identity = crate::remote_cluster::outgoing_identity(node, alias, identity);
    let req = LeaseReq { identity, index: index.to_string(), uuid: uuid.to_string(), shard, op, id: id.to_string(), retaining_seq_no: retaining, routed: false };
    call(node, alias, ACTION_LEASE, &req, Duration::from_secs(60)).await
}

/// A [`RemoteError`]-shaped failure as the follower reports it.
pub fn from_remote(e: &RemoteError) -> EsError {
    remote_error(e)
}

/// The lease change on this cluster's own copies (`forget_follower` on the
/// leader cluster).
pub async fn change_lease_here(node: &Arc<ClusterNode>, req: LeaseReq) -> Result<LeaseResp, EsError> {
    lease(node.clone(), req, None).await.map_err(|e| remote_error(&e))
}
