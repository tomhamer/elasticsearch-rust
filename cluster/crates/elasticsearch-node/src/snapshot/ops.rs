//! Master-side snapshot operations: create, get, status, delete, clone, and
//! resuming snapshots orphaned by a master failover.
//!
//! **Create.** The master resolves the indices, records the snapshot in the
//! cluster state (`IN_PROGRESS`) and hands the shards to its snapshots
//! service. Each shard is snapshotted where its primary is (ES
//! `SnapshotShardsService`, see [`super::upload`]): the node holding it
//! freezes the shard ([`super::shard`]) and uploads the frozen files
//! incrementally to the repository itself, writing the shard-level blobs
//! on top of the shard generation the master gives it, and reports its
//! progress (`_status`) and new generation to the master; the master only
//! writes the index / global metadata and finalizes the root `index-N`. A
//! primary on the master itself is frozen and uploaded locally. No shard
//! data of another node is ever copied to the master, except from a node
//! that cannot upload (an older build during a rolling upgrade): the
//! master then pulls that shard's frozen files into its staging directory,
//! only if they fit under the high disk watermark (the shard fails
//! otherwise). A shard that cannot be snapshotted (its primary is gone,
//! its node left or failed the upload) becomes a shard failure and the
//! snapshot ends `PARTIAL`, as in ES; with `partial: false` a snapshot is
//! refused up front when a primary is not allocated. `DELETE` of a running
//! snapshot aborts the uploads on the data nodes.
//!
//! **Master failover.** The in-progress entry survives in the cluster
//! state, with the node that held each primary when the snapshot started.
//! The new master finds it (no local task owns it) and runs it again under
//! the same uuid ([`resume_orphans`]): it asks the data nodes for the
//! shards again, and a node still running (or done with) the upload the
//! old master started joins it rather than starting over; blobs written
//! for the old master's attempt are overwritten or left unreferenced
//! (removed by `_cleanup`). A shard whose recorded node has left the
//! cluster fails and the snapshot ends `PARTIAL`, as in ES. If the old
//! master had already published the snapshot (it died between the finalize
//! write and removing the entry) the entry is only removed. The root
//! `index-N` compare-and-set plus the same-name check make a second
//! finalize of the same snapshot (deposed master, duplicate resume) fail
//! rather than overwrite anything; the loser then reports the snapshot the
//! winner published.

use super::common::{self, err, now_millis, service, state_of, Pending, PROGRESS_CUSTOM, REPOS_CUSTOM};
use super::shard::{self, PrepareReq, ReleaseReq};
use crate::front::{RestReq, RestResp, SharedText};
use crate::node::ClusterNode;
use elasticsearch_cluster::state::{ClusterState, IndexMetadata, IndexState};
use elasticsearch_snapshots::snapshot::{ShardStage, ShardStatus, SnapshotStatus, ES_VERSION, ES_VERSION_ID};
use elasticsearch_snapshots::util::{random_base64_uuid, resolve_names, simple_match, CancelToken};
use elasticsearch_snapshots::{CreateSnapshotParams, CreateSnapshotRequest, GetSnapshotsRequest, IndexSnapshotInput, RemoteShard, RemoteUploader, RenderOptions, ShardSnapshotInput, SnapshotError, SnapshotInfo, SnapshotState, SnapshotsService};
use futures::StreamExt;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

/// Index metadata as stored in a snapshot (ES `IndexMetadata` JSON shape;
/// settings flat).
pub fn index_metadata_json(im: &IndexMetadata) -> Value {
    let mut settings = im.settings.clone();
    settings.insert("index.uuid".into(), json!(im.uuid));
    settings.insert("index.number_of_shards".into(), json!(im.number_of_shards.to_string()));
    let mappings = if im.mappings.as_object().map(|o| o.len() == 1 && o.contains_key("_doc")).unwrap_or(false) { (*im.mappings).clone() } else { json!({"_doc": im.mappings}) };
    let primary_terms: Map<String, Value> = (0..im.number_of_shards.max(1)).map(|k| (k.to_string(), json!(im.primary_term(k)))).collect();
    json!({
        "version": im.version,
        "mapping_version": im.mapping_version,
        "settings_version": im.settings_version,
        "aliases_version": im.aliases_version,
        "state": if im.state == IndexState::Open { "open" } else { "close" },
        "settings": settings,
        "mappings": mappings,
        "aliases": im.aliases,
        // Every shard's term, and the routing shards (ES `IndexMetadata`
        // keys): a file restore continues the terms and must route
        // documents to the shards whose files it adopts.
        "primary_terms": primary_terms,
        "routing_num_shards": im.routing_num_shards,
        "in_sync_allocations": {},
        "creation_date": im.creation_date,
    })
}

/// Global metadata stored with a snapshot.
pub fn global_metadata_json(st: &ClusterState) -> Value {
    let customs: Map<String, Value> = st
        .metadata
        .customs
        .iter()
        // The segment format floor belongs to the live cluster's members,
        // not to its data: never snapshotted, never restored.
        .filter(|(k, _)| !matches!(k.as_str(), REPOS_CUSTOM | PROGRESS_CUSTOM | "node_shutdown" | elasticsearch_cluster::compatibility::SEGMENT_FORMAT_CUSTOM))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    json!({
        "cluster_uuid": st.metadata.cluster_uuid,
        "persistent_settings": st.metadata.persistent_settings,
        "index_templates": st.metadata.index_templates,
        "component_templates": st.metadata.component_templates,
        "legacy_templates": st.metadata.legacy_templates,
        "customs": customs,
    })
}

fn expand_opts(p: &CreateSnapshotParams) -> elasticsearch_engine::node::ResolveOpts {
    let mut o = elasticsearch_engine::node::ResolveOpts { ignore_unavailable: p.ignore_unavailable, allow_no_indices: p.allow_no_indices, expand_open: false, expand_closed: false, expand_hidden: false };
    for w in &p.expand_wildcards {
        match w.as_str() {
            "open" => o.expand_open = true,
            "closed" => o.expand_closed = true,
            "hidden" => o.expand_hidden = true,
            "all" => {
                o.expand_open = true;
                o.expand_closed = true;
                o.expand_hidden = true;
            }
            _ => {}
        }
    }
    o
}

/// Concrete indices (and data streams) a create request selects.
fn resolve_indices(node: &ClusterNode, p: &CreateSnapshotParams) -> Result<(Vec<String>, Vec<String>), RestResp> {
    let exprs: Vec<String> = if p.indices.is_empty() { vec!["_all".into()] } else { p.indices.clone() };
    let expr = exprs.join(",");
    let mut names: BTreeSet<String> = BTreeSet::new();
    match node.engine.resolve(&expr, expand_opts(p)) {
        Ok(ts) => names.extend(ts.iter().map(|t| t.svc.name.clone())),
        Err(e) => return Err(RestResp::json(e.status, &e.to_json_with_trace(false))),
    }
    let mut streams = Vec::new();
    for (name, ds) in node.engine.data_streams.read().iter() {
        if ds.indices.iter().any(|i| names.contains(i)) {
            streams.push(name.clone());
        }
    }
    Ok((names.into_iter().collect(), streams))
}

/// A running create: the pending entry and the task finishing it.
pub struct Started {
    pub pending: Arc<Pending>,
    pub handle: tokio::task::JoinHandle<Result<SnapshotInfo, SnapshotError>>,
}

/// Validate and start a snapshot (shared by REST and SLM).
pub async fn start_create(node: &Arc<ClusterNode>, repo: &str, name: &str, params: CreateSnapshotParams) -> Result<Started, RestResp> {
    let svc = service(node).await;
    let r = svc.repository(repo).map_err(|e| err(&e))?;
    elasticsearch_snapshots::repository::settings::validate_snapshot_name(repo, name).map_err(|e| err(&e))?;
    if r.is_readonly() {
        return Err(err(&SnapshotError::repository(repo, "cannot create snapshot in a readonly repository")));
    }
    let s = state_of(node);
    let busy = svc.running_snapshots(Some(repo)).iter().any(|x| x.name == name) || s.pending.lock().iter().any(|p| p.repository == repo && p.name == name)
        // also one started by a previous master and not yet finalized
        || common::progress_entries(&node.state(), "snapshots").iter().any(|e| e["repository"].as_str() == Some(repo) && e["snapshot"].as_str() == Some(name));
    if busy {
        return Err(err(&SnapshotError::invalid_snapshot_name(repo, name, "snapshot with the same name is already in-progress")));
    }
    let data = r.repository_data().await.map_err(|e| err(&e))?;
    if data.snapshot_by_name(name).is_some() {
        return Err(err(&SnapshotError::invalid_snapshot_name(repo, name, "snapshot with the same name already exists")));
    }
    let (indices, data_streams) = resolve_indices(node, &params)?;
    // A primary that is assigned but still initializing is waited for, as
    // ES's snapshot shard state `WAITING` does: only a primary with no copy
    // anywhere makes the snapshot fail. (The master may also apply the
    // state that started a primary after the node that wrote to it did.)
    let st = wait_initializing_primaries(node, &indices).await;
    let missing: Vec<String> = indices.iter().filter(|i| (0..shards_of(&st.metadata, i)).any(|k| primary_node(&st, i, k).is_none())).cloned().collect();
    if !params.partial && !missing.is_empty() {
        return Err(err(&SnapshotError::snapshot(repo, name, format!("Indices don't have primary shards [{}]", missing.join(", ")))));
    }
    let uuid = random_base64_uuid();
    let (done, _) = tokio::sync::watch::channel(false);
    let pending = Arc::new(Pending {
        repository: repo.to_string(),
        name: name.to_string(),
        uuid: uuid.clone(),
        include_global_state: params.include_global_state,
        partial: params.partial,
        indices: indices.clone(),
        data_streams: data_streams.clone(),
        metadata: params.metadata.clone(),
        start_time: now_millis(),
        assigned: indices.iter().flat_map(|i| (0..shards_of(&st.metadata, i)).filter_map(|k| primary_node(&st, i, k).map(|n| (assigned_key(i, k), n)))).collect(),
        cancel: CancelToken::new(),
        in_service: AtomicBool::new(false),
        done,
    });
    s.pending.lock().push(pending.clone());
    let entry = json!({
        "repository": repo, "snapshot": name, "uuid": uuid, "state": "STARTED",
        "include_global_state": params.include_global_state, "partial": params.partial,
        "start_time_millis": pending.start_time, "indices": indices, "data_streams": data_streams,
        "master": node.local.id, "repository_state_id": data.generation,
        "metadata": params.metadata, "shards": pending.assigned,
    });
    if let Err(e) = common::add_progress(node, "snapshots", entry).await {
        s.pending.lock().retain(|p| !Arc::ptr_eq(p, &pending));
        let _ = pending.done.send(true);
        if elasticsearch_cluster::coordination::CoordinationError::is_not_master(&e) {
            return Err(crate::front::task_error(&e));
        }
        return Err(err(&SnapshotError::concurrent_snapshot_execution(repo, name, format!("failed to start snapshot: {e}"))));
    }
    let node2 = node.clone();
    let p2 = pending.clone();
    let handle = tokio::spawn(async move { run_create(node2, svc, p2).await });
    Ok(Started { pending, handle })
}

async fn run_create(node: Arc<ClusterNode>, svc: Arc<SnapshotsService>, p: Arc<Pending>) -> Result<SnapshotInfo, SnapshotError> {
    let result = async {
        let inputs: Vec<IndexSnapshotInput> = futures::stream::iter(p.indices.clone().into_iter().map(|i| {
            let node = node.clone();
            let p = p.clone();
            async move { gather_index(&node, &p, &i).await }
        }))
        .buffer_unordered(4)
        .collect()
        .await;
        if p.cancel.is_cancelled() {
            return Err(SnapshotError::snapshot(&p.repository, &p.name, p.cancel.reason().unwrap_or_else(|| "Snapshot was aborted".into())));
        }
        let mut inputs = inputs;
        inputs.sort_by(|a, b| a.index_name.cmp(&b.index_name));
        let st = node.state();
        let req = CreateSnapshotRequest {
            repository: p.repository.clone(),
            snapshot: p.name.clone(),
            uuid: Some(p.uuid.clone()),
            indices: inputs,
            global_metadata_json: Some(global_metadata_json(&st)),
            include_global_state: p.include_global_state,
            // Up-front primary checks are done; shards failing from here on
            // make the snapshot PARTIAL (ES semantics).
            partial: true,
            metadata: p.metadata.clone(),
            feature_states: vec![],
            data_streams: p.data_streams.clone(),
            node_id: Some(node.local.id.clone()),
            // Every other node's primaries are snapshotted by their node.
            remote: Some(RemoteUploader(Arc::new(super::upload::Uploader { node: node.clone() }))),
        };
        let (_, h) = svc.start_snapshot(req).await?;
        p.in_service.store(true, std::sync::atomic::Ordering::SeqCst);
        h.await.map_err(|e| SnapshotError::illegal_state(format!("snapshot task failed: {e}")))?
    }
    .await;
    // Lost the finalize to another master working on the same snapshot (a
    // resume after failover, or we are the deposed one): the published
    // snapshot is the result.
    let result = match result {
        Err(e) => match published(&svc, &p.repository, &p.uuid).await {
            Some(info) => {
                tracing::info!(repository = %p.repository, snapshot = %p.name, "snapshot was already finalized ({e})");
                Ok(info)
            }
            None => Err(e),
        },
        ok => ok,
    };
    // Staging copies are no longer needed anywhere.
    shard::release(&node, &p.uuid);
    let _ = tokio::fs::remove_dir_all(common::staging_root(&node).join(format!("{}-recv", common::dir_name(&p.uuid)))).await;
    // Best effort and off the completion path: a node that just died must
    // not keep the snapshot "in progress" for its request timeout.
    let st = node.state();
    for n in st.nodes.values().filter(|n| n.id != node.local.id) {
        if let Ok(addr) = n.transport_address.parse::<std::net::SocketAddr>() {
            // Accounted, never refused: a skipped release keeps that node's
            // finished upload job and staging until it restarts.
            let Ok(uuid) = crate::control_memory::unrefused("snapshot release", || SharedText::try_new(&p.uuid)) else { continue };
            let node=node.clone();
            tokio::spawn(async move {
                let _ = node.transport.send_request::<_, ()>(addr, shard::ACTION_RELEASE, &ReleaseReq { snapshot_uuid: uuid }, Duration::from_secs(10)).await;
            });
        }
    }
    let cleanup = std::time::Instant::now();
    common::after_repo_write(&node, &svc, &p.repository).await;
    state_of(&node).finished.lock().insert(p.uuid.clone());
    if let Err(e) = common::remove_progress(&node, "snapshots", p.uuid.clone()).await {
        tracing::warn!(snapshot = %p.name, "failed to remove in-progress entry: {e}");
    }
    state_of(&node).pending.lock().retain(|x| !Arc::ptr_eq(x, &p));
    let _ = p.done.send(true);
    match &result {
        Ok(i) => tracing::info!(repository = %p.repository, snapshot = %p.name, state = %i.state.map(|s| s.as_str()).unwrap_or("?"), publish_ms = cleanup.elapsed().as_millis() as u64, "snapshot completed"),
        Err(e) => tracing::warn!(repository = %p.repository, snapshot = %p.name, "snapshot failed: {e}"),
    }
    result
}

/// The snapshot `uuid` as published in the repository, if it is.
async fn published(svc: &SnapshotsService, repo: &str, uuid: &str) -> Option<SnapshotInfo> {
    let r = svc.repository(repo).ok()?;
    let data = r.repository_data().await.ok()?;
    data.snapshot_by_uuid(uuid)?;
    r.read_snapshot_info(uuid).await.ok()
}

/// Number of shards of `index` (1 when it is gone from the metadata).
fn shards_of(md: &elasticsearch_cluster::state::Metadata, index: &str) -> u32 {
    crate::shard_engines::shards_of(md, index)
}

/// Node holding the active primary of `index`/`shard`.
/// The state once no primary of `indices` is initializing (or after 30s).
async fn wait_initializing_primaries(node: &Arc<ClusterNode>, indices: &[String]) -> Arc<ClusterState> {
    let mut rx = node.subscribe();
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let st = rx.borrow_and_update().clone();
        if !primaries_initializing(&st, indices) || !matches!(tokio::time::timeout_at(deadline, rx.changed()).await, Ok(Ok(()))) {
            return st;
        }
    }
}

/// Whether a primary of `indices` is assigned to a node but not active yet.
fn primaries_initializing(st: &ClusterState, indices: &[String]) -> bool {
    indices.iter().any(|i| {
        (0..shards_of(&st.metadata, i)).any(|k| st.routing_table.shard_table(i, k).is_some_and(|t| t.active_primary().is_none() && t.primary().is_some_and(|p| p.node_id.is_some())))
    })
}

fn primary_node(st: &ClusterState, index: &str, shard: u32) -> Option<String> {
    st.routing_table.shard_table(index, shard).and_then(|t| t.active_primary()).and_then(|r| r.node_id.clone())
}

/// Key of a shard in `Pending::assigned` / the in-progress entry's
/// `shards`: the bare index name for shard 0 (the pre-multi-shard format,
/// so entries written by an older master still resume), `index/k` otherwise
/// (`/` cannot appear in an index name).
pub(crate) fn assigned_key(index: &str, shard: u32) -> String {
    if shard == 0 {
        index.to_string()
    } else {
        format!("{index}/{shard}")
    }
}

async fn gather_index(node: &Arc<ClusterNode>, p: &Arc<Pending>, index: &str) -> IndexSnapshotInput {
    let st = node.state();
    let md = st.metadata.indices.get(index).map(index_metadata_json).unwrap_or_else(|| json!({"settings": {"index.number_of_shards": "1"}}));
    let n = shards_of(&st.metadata, index);
    // Shards are planned independently (ES snapshots shards in parallel;
    // one failing shard only makes the snapshot PARTIAL). Only the
    // master's own primaries are frozen here; the others are frozen and
    // uploaded by their nodes once the snapshot runs.
    let mut shards: Vec<ShardSnapshotInput> = futures::stream::iter((0..n).map(|k| {
        let st = st.clone();
        async move {
            match plan_shard(node, p, &st, index, k).await {
                Ok(Planned::Local(dir, files)) => ShardSnapshotInput { shard: k, dir, files, unavailable: None, remote: None },
                Ok(Planned::Remote(node_id)) => ShardSnapshotInput { shard: k, dir: PathBuf::new(), files: vec![].into(), unavailable: None, remote: Some(RemoteShard { node_id }) },
                Err(reason) => {
                    tracing::warn!(%index, shard = k, snapshot = %p.name, "shard cannot be snapshotted: {reason}");
                    ShardSnapshotInput { shard: k, dir: PathBuf::new(), files: vec![].into(), unavailable: Some(reason), remote: None }
                }
            }
        }
    }))
    .buffer_unordered(4)
    .collect()
    .await;
    shards.sort_by_key(|s| s.shard);
    IndexSnapshotInput { index_name: index.to_string(), index_metadata_json: md, shards }
}

/// Where a shard is snapshotted.
enum Planned {
    /// The master's own primary, frozen into this staging directory.
    Local(PathBuf, elasticsearch_snapshots::service::SnapshotFiles),
    /// By the node holding the primary.
    Remote(String),
}

async fn plan_shard(node: &Arc<ClusterNode>, p: &Pending, st: &ClusterState, index: &str, shard: u32) -> Result<Planned, String> {
    if !st.metadata.indices.contains_key(index) {
        return Err(format!("index [{index}] was deleted"));
    }
    // Resumed after a master change: the copy being snapshotted went away
    // with its node (ES fails such a shard; the snapshot ends PARTIAL).
    if let Some(orig) = p.assigned.get(&assigned_key(index, shard)) {
        if !st.nodes.contains_key(orig) {
            return Err(format!("node [{orig}] left the cluster while the snapshot was running"));
        }
    }
    let primary = primary_node(st, index, shard).ok_or_else(|| "primary shard is not allocated".to_string())?;
    if primary != node.local.id {
        if !st.nodes.contains_key(&primary) {
            return Err(format!("node [{primary}] left the cluster"));
        }
        return Ok(Planned::Remote(primary));
    }
    // An upload a previous master started on this node for the same shard
    // must not race the local freeze of the same staging directory.
    super::upload::cancel_local_job(node, &p.uuid, index, shard).await;
    let req = PrepareReq { index: SharedText::try_new(index).map_err(|e|e.to_string())?, shard, snapshot_uuid: SharedText::try_new(&p.uuid).map_err(|e|e.to_string())? };
    let r = shard::prepare(node, &req).await.map_err(|e|e.to_string())?;
    Ok(Planned::Local(PathBuf::from(r.dir.as_str()), shard::snapshot_names(r.files).map_err(|e|e.to_string())?))
}

/// `PUT|POST _snapshot/{repo}/{snapshot}`.
pub async fn create(node: &Arc<ClusterNode>, req: &RestReq, repo: &str, name: &str) -> RestResp {
    let body = req.json();
    if !req.body.is_empty() && body.is_none() {
        return RestResp::error(400, "parse_exception", "request body is required to be valid JSON");
    }
    let params = match CreateSnapshotParams::from_json(body.as_ref()) {
        Ok(p) => p,
        Err(e) => return err(&e),
    };
    let wait = req.params().get("wait_for_completion").map(|v| v != "false").unwrap_or(false);
    let started = match start_create(node, repo, name, params).await {
        Ok(s) => s,
        Err(r) => return r,
    };
    if !wait {
        return RestResp::json(200, &json!({"accepted": true}));
    }
    match started.handle.await {
        Ok(Ok(info)) => RestResp::json(200, &info.create_response()),
        Ok(Err(e)) => err(&e),
        Err(e) => RestResp::error(500, "illegal_state_exception", &e.to_string()),
    }
}

/// Pending snapshots of `repo` not yet reported by the service.
fn pending_of(node: &ClusterNode, svc: &SnapshotsService, repo: &str) -> Vec<Arc<Pending>> {
    let running: BTreeSet<String> = svc.running_snapshots(Some(repo)).iter().map(|r| r.name.clone()).collect();
    let s = state_of(node);
    let mut out: Vec<Arc<Pending>> = s.pending.lock().iter().filter(|p| p.repository == repo && !running.contains(&p.name)).cloned().collect();
    // Snapshots started by a previous master stay in the cluster state until
    // `resume_orphans` has adopted them; until then they must still be
    // reported as in progress rather than as missing.
    let finished = s.finished.lock().clone();
    for e in common::progress_entries(&node.state(), "snapshots") {
        let (Some(name), Some(uuid)) = (e["snapshot"].as_str(), e["uuid"].as_str()) else { continue };
        if e["repository"].as_str() != Some(repo) || running.contains(name) || finished.contains(uuid) || out.iter().any(|p| p.uuid == uuid || p.name == name) {
            continue;
        }
        let strs = |k: &str| -> Vec<String> { e[k].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default() };
        out.push(Arc::new(Pending {
            repository: repo.to_string(),
            name: name.to_string(),
            uuid: uuid.to_string(),
            include_global_state: e["include_global_state"].as_bool().unwrap_or(true),
            partial: e["partial"].as_bool().unwrap_or(false),
            indices: strs("indices"),
            data_streams: strs("data_streams"),
            metadata: e.get("metadata").and_then(Value::as_object).cloned(),
            start_time: e["start_time_millis"].as_i64().unwrap_or_else(now_millis),
            assigned: BTreeMap::new(),
            cancel: CancelToken::new(),
            in_service: AtomicBool::new(false),
            done: tokio::sync::watch::channel(false).0,
        }));
    }
    out
}

fn pending_info(p: &Pending, st: &ClusterState) -> SnapshotInfo {
    SnapshotInfo {
        repository: p.repository.clone(),
        snapshot: p.name.clone(),
        uuid: p.uuid.clone(),
        version_id: Some(ES_VERSION_ID),
        version: Some(ES_VERSION.into()),
        indices: p.indices.clone(),
        data_streams: p.data_streams.clone(),
        feature_states: vec![],
        include_global_state: Some(p.include_global_state),
        metadata: p.metadata.clone(),
        state: Some(SnapshotState::InProgress),
        reason: None,
        start_time: p.start_time,
        end_time: 0,
        total_shards: p.indices.iter().map(|i| shards_of(&st.metadata, i)).sum(),
        successful_shards: 0,
        failures: vec![],
        index_details: BTreeMap::new(),
    }
}

fn pending_status(p: &Pending, st: &ClusterState, human: bool) -> Value {
    let mut indices = BTreeMap::new();
    for i in &p.indices {
        let mut m = BTreeMap::new();
        for k in 0..shards_of(&st.metadata, i) {
            m.insert(k, ShardStatus::new(ShardStage::Init));
        }
        indices.insert(i.clone(), m);
    }
    SnapshotStatus { snapshot: p.name.clone(), repository: p.repository.clone(), uuid: p.uuid.clone(), state: "STARTED".into(), include_global_state: p.include_global_state, indices, start_time: p.start_time, time: now_millis() - p.start_time }.to_json(human)
}

fn name_matches(exprs: &[String], name: &str, current: bool) -> bool {
    exprs.iter().any(|e| e == "_all" || (e == "_current" && current) || simple_match(e, name))
}

fn params_btree(req: &RestReq) -> BTreeMap<String, String> {
    req.params().into_iter().collect()
}

/// `GET _snapshot/{repo}/{snapshots}`.
pub async fn get(node: &Arc<ClusterNode>, req: &RestReq, repo: &str, snaps: &str) -> RestResp {
    let svc = service(node).await;
    let mut gr = match GetSnapshotsRequest::from_params(repo, snaps, &params_btree(req)) {
        Ok(r) => r,
        Err(e) => return err(&e),
    };
    // Snapshots still gathering files are reported as IN_PROGRESS.
    let repo_names = svc.repository_names();
    let (sel_repos, _) = resolve_names(&gr.repositories, &repo_names);
    let mut extra: Vec<SnapshotInfo> = Vec::new();
    let mut taken: BTreeSet<String> = BTreeSet::new();
    for r in &sel_repos {
        for p in pending_of(node, &svc, r) {
            if name_matches(&gr.snapshots, &p.name, true) && (gr.slm_policy_filter.is_empty() || p.metadata.as_ref().and_then(|m| m.get("policy")).is_some()) {
                taken.insert(p.name.clone());
                extra.push(pending_info(&p, &node.state()));
            }
        }
    }
    let original = gr.snapshots.clone();
    gr.snapshots.retain(|s| !taken.contains(s));
    let mut out = if gr.snapshots.is_empty() && !original.is_empty() {
        // Only pending names were asked for; still validate the repository.
        if let Err(e) = svc.get_repositories_json(&gr.repositories) {
            return err(&e);
        }
        json!({"snapshots": [], "total": 0, "remaining": 0})
    } else {
        match svc.get_snapshots(&gr).await {
            Ok(v) => v,
            Err(e) => return err(&e),
        }
    };
    if !extra.is_empty() {
        let opts = RenderOptions { verbose: gr.verbose, include_repository: gr.include_repository, index_names: gr.index_names, index_details: gr.index_details, human: gr.human };
        let known: BTreeSet<String> = out["snapshots"].as_array().map(|a| a.iter().filter_map(|s| s["snapshot"].as_str().map(str::to_string)).collect()).unwrap_or_default();
        let mut added = 0;
        if let Some(a) = out["snapshots"].as_array_mut() {
            for i in extra.iter().filter(|i| !known.contains(&i.snapshot)) {
                a.push(i.render(opts));
                added += 1;
            }
        }
        if let Some(t) = out["total"].as_u64() {
            out["total"] = json!(t + added);
        }
    }
    RestResp::json(200, &out)
}

/// `GET _snapshot/_status`, `GET _snapshot/{repo}/_status`,
/// `GET _snapshot/{repo}/{snapshots}/_status`.
pub async fn status(node: &Arc<ClusterNode>, req: &RestReq, repo: Option<&str>, snaps: &str) -> RestResp {
    let svc = service(node).await;
    let params = req.params();
    let ignore = params.get("ignore_unavailable").map(|v| v != "false").unwrap_or(false);
    let human = params.get("human").map(|v| v != "false").unwrap_or(false);
    let names: Vec<String> = snaps.split(',').filter(|s| !s.is_empty()).map(str::to_string).collect();
    let repos: Vec<String> = match repo {
        Some(r) => vec![r.to_string()],
        None => svc.repository_names(),
    };
    let mut extra = Vec::new();
    let mut taken = BTreeSet::new();
    for r in &repos {
        for p in pending_of(node, &svc, r) {
            if names.is_empty() || name_matches(&names, &p.name, true) {
                taken.insert(p.name.clone());
                extra.push(pending_status(&p, &node.state(), human));
            }
        }
    }
    let rest: Vec<String> = names.iter().filter(|n| !taken.contains(*n)).cloned().collect();
    let mut out = if !names.is_empty() && rest.is_empty() {
        if let Some(r) = repo {
            if let Err(e) = svc.repository(r) {
                return err(&e);
            }
        }
        json!({"snapshots": []})
    } else {
        match svc.snapshot_status(repo, &rest, ignore, human).await {
            Ok(v) => v,
            Err(e) => return err(&e),
        }
    };
    if let Some(a) = out["snapshots"].as_array_mut() {
        a.extend(extra);
    }
    RestResp::json(200, &out)
}

/// `DELETE _snapshot/{repo}/{snapshots}`.
pub async fn delete(node: &Arc<ClusterNode>, repo: &str, snaps: &str) -> RestResp {
    let svc = service(node).await;
    if let Err(e) = svc.repository(repo) {
        return err(&e);
    }
    let names: Vec<String> = snaps.split(',').filter(|s| !s.is_empty()).map(str::to_string).collect();
    // Snapshots orphaned by a master change become local tasks first, so
    // they are aborted like any other running snapshot.
    resume_orphans(node).await;
    let pend: Vec<Arc<Pending>> = state_of(node).pending.lock().iter().filter(|p| p.repository == repo && name_matches(&names, &p.name, false)).cloned().collect();
    for p in &pend {
        p.cancel.cancel("Snapshot was aborted by deletion");
    }
    for p in &pend {
        // Once the snapshots service runs it, `delete_snapshots` aborts it.
        while !p.in_service() && tokio::time::timeout(Duration::from_millis(100), p.wait_done()).await.is_err() {}
    }
    let aborted: BTreeSet<String> = pend.iter().map(|p| p.name.clone()).collect();
    let mut todo = names.clone();
    loop {
        match svc.delete_snapshots(repo, &todo).await {
            Ok(v) => {
                for p in &pend {
                    p.wait_done().await;
                }
                common::after_repo_write(node, &svc, repo).await;
                return RestResp::json(200, &v);
            }
            Err(e) if e.kind == "snapshot_missing_exception" && !aborted.is_empty() => {
                let before = todo.len();
                todo.retain(|n| !aborted.contains(n));
                if todo.is_empty() {
                    for p in &pend {
                        p.wait_done().await;
                    }
                    return RestResp::json(200, &json!({"acknowledged": true}));
                }
                if todo.len() == before {
                    return err(&e);
                }
            }
            Err(e) => return err(&e),
        }
    }
}

/// `PUT _snapshot/{repo}/{source}/_clone/{target}`.
pub async fn clone(node: &Arc<ClusterNode>, req: &RestReq, repo: &str, source: &str, target: &str) -> RestResp {
    let svc = service(node).await;
    let body = req.json().unwrap_or(json!({}));
    match svc.clone_snapshot(repo, source, target, &body).await {
        Ok(v) => {
            common::after_repo_write(node, &svc, repo).await;
            RestResp::json(200, &v)
        }
        Err(e) => err(&e),
    }
}

/// Master only: take over snapshots and restores whose coordinating master
/// is gone. Safe to call concurrently and repeatedly (the 1 s master tick
/// and `delete` both do).
pub async fn resume_orphans(node: &Arc<ClusterNode>) {
    if !node.is_master() {
        return;
    }
    let st = node.state();
    let entries = common::progress_entries(&st, "snapshots");
    let restores = common::progress_entries(&st, "restore");
    if entries.is_empty() && restores.is_empty() {
        return;
    }
    let s = state_of(node);
    let svc = service(node).await;
    let strings = |v: &Value| -> Vec<String> { v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default() };
    for e in entries {
        let uuid = e["uuid"].as_str().unwrap_or_default().to_string();
        let repo = e["repository"].as_str().unwrap_or_default().to_string();
        let name = e["snapshot"].as_str().unwrap_or_default().to_string();
        if s.finished.lock().contains(&uuid) || s.pending.lock().iter().any(|p| p.uuid == uuid) || svc.running_snapshots(Some(&repo)).iter().any(|r| r.name == name) {
            continue;
        }
        let Ok(r) = svc.repository(&repo) else {
            tracing::warn!(repository = %repo, snapshot = %name, "orphaned snapshot's repository is gone; dropping its in-progress entry");
            let _ = common::remove_progress(node, "snapshots", uuid).await;
            continue;
        };
        let data = match r.repository_data().await {
            Ok(d) => d,
            Err(err) => {
                tracing::warn!(repository = %repo, snapshot = %name, "cannot read the repository to resume the snapshot (will retry): {err}");
                continue;
            }
        };
        if data.snapshot_by_uuid(&uuid).is_some() || data.snapshot_by_name(&name).is_some() {
            // The previous master published it and died before removing
            // the entry.
            tracing::info!(repository = %repo, snapshot = %name, "snapshot orphaned by a master change was already finalized");
            common::after_repo_write(node, &svc, &repo).await;
            s.finished.lock().insert(uuid.clone());
            let _ = common::remove_progress(node, "snapshots", uuid).await;
            continue;
        }
        let (done, _) = tokio::sync::watch::channel(false);
        let p = Arc::new(Pending {
            repository: repo.clone(),
            name: name.clone(),
            uuid: uuid.clone(),
            include_global_state: e["include_global_state"].as_bool().unwrap_or(true),
            partial: e["partial"].as_bool().unwrap_or(false),
            indices: strings(&e["indices"]),
            data_streams: strings(&e["data_streams"]),
            metadata: e.get("metadata").and_then(Value::as_object).cloned(),
            start_time: e["start_time_millis"].as_i64().unwrap_or_else(now_millis),
            assigned: e["shards"].as_object().map(|m| m.iter().filter_map(|(k, v)| v.as_str().map(|n| (k.clone(), n.to_string()))).collect()).unwrap_or_default(),
            cancel: CancelToken::new(),
            in_service: AtomicBool::new(false),
            done,
        });
        {
            let mut pending = s.pending.lock();
            if pending.iter().any(|x| x.uuid == uuid) || s.finished.lock().contains(&uuid) {
                continue;
            }
            pending.push(p.clone());
        }
        tracing::info!(repository = %repo, snapshot = %name, "resuming snapshot orphaned by a master change");
        tokio::spawn(run_create(node.clone(), svc.clone(), p));
    }
    for e in restores {
        super::restore::resume_orphan(node, &e);
    }
}

#[cfg(test)]
mod tests {
    /// A snapshot waits for an initializing primary (ES `WAITING`) and fails
    /// only for a primary with no copy.
    #[test]
    fn initializing_primaries_are_waited_for() {
        use super::{primaries_initializing, primary_node};
        use elasticsearch_cluster::state::{AllocationId, ClusterState, IndexMetadata, IndexRoutingTable, IndexShardRoutingTable, IndexState, ShardRouting, ShardState};
        use serde_json::json;
        let state = |shard_state: ShardState, node: Option<&str>| {
            let mut s = ClusterState::empty("test");
            s.metadata.indices.insert("i".into(), IndexMetadata { name: "i".into(), uuid: "u".into(), version: 1, mapping_version: 1, settings_version: 1,
                aliases_version: 1, state: IndexState::Open, settings: Default::default(), mappings: json!({}).into(), aliases: Default::default(), number_of_shards: 1,
                number_of_replicas: 0, routing_num_shards: 1024, primary_terms: vec![1], in_sync_allocation_ids: Default::default(), creation_date: 0, lifecycle: None });
            let copy = ShardRouting { index: "i".into(), shard: 0, primary: true, state: shard_state, node_id: node.map(str::to_string), relocating_node_id: None,
                allocation_id: node.map(|n| AllocationId { id: format!("a-{n}"), relocation_id: None }), recovery_source: None, unassigned_info: None, expected_shard_size_in_bytes: None };
            s.routing_table.indices.insert("i".into(), IndexRoutingTable { index_uuid: "u".into(), shards: vec![IndexShardRoutingTable { shard: 0, copies: vec![copy] }] });
            s
        };
        let i = vec!["i".to_string()];
        let initializing = state(ShardState::Initializing, Some("n1"));
        assert!(primaries_initializing(&initializing, &i));
        assert_eq!(primary_node(&initializing, "i", 0), None);
        let started = state(ShardState::Started, Some("n1"));
        assert!(!primaries_initializing(&started, &i));
        assert_eq!(primary_node(&started, "i", 0).as_deref(), Some("n1"));
        // No copy anywhere: not waited for (the snapshot fails).
        assert!(!primaries_initializing(&state(ShardState::Unassigned, None), &i));
    }

    use super::assigned_key;

    #[test]
    fn assigned_key_is_bare_for_shard_zero() {
        assert_eq!(assigned_key("idx", 0), "idx");
        assert_eq!(assigned_key("idx", 2), "idx/2");
    }
}
