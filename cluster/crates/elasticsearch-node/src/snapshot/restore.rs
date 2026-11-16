//! `POST _snapshot/{repo}/{snapshot}/_restore`, run by the elected master.
//!
//! For every index of the plan the master
//! 1. deletes the closed index being replaced, if any;
//! 2. creates the target index through the normal metadata path (settings,
//!    mappings and — with `include_aliases` — aliases from the snapshot,
//!    adjusted by `index_settings` / `ignore_index_settings`; write blocks
//!    are applied only after the data is in);
//! 3. restores its shards from the snapshot's files (ES): the index is
//!    armed in the restore entry first, so the task creating it gives its
//!    primaries `RecoverySource::Snapshot` routing; each primary's node
//!    downloads, verifies and adopts the shard's files and starts the copy,
//!    and replicas then recover from it by file-based peer recovery
//!    ([`super::file_restore`]). The master waits for every primary to
//!    start or fail its restore (ES `RestoreInProgress`).
//!
//! Re-indexing is the fallback, logged with its reason, when the files
//! cannot be adopted: a source-only repository, a node that cannot restore
//! files, an index created by another index version (its routing and
//! source handling follow that version), a shard count, routing shards or
//! routing / sort / source-mode setting that differs from the snapshot's,
//! or segment formats some member cannot read (found by the primary's node,
//! which then fails the shard with `file_restore::REINDEX_TAG`; the master
//! deletes the index and starts again). Field ids are no reason: the restored index takes the
//! snapshot's (`file_restore`). The re-index path
//! 1. downloads the shard files into a staging directory, opens them
//!    read-only as an engine `IndexService` (replaying a staged translog if
//!    the snapshot was taken without a flush) and streams every live
//!    document out of it with the peer-recovery snapshot reader;
//! 2. writes the documents through the cluster write path (`_bulk` with
//!    their `_id`, `_routing` and `_version` as an external version, and no
//!    ingest pipeline), so the primary replicates them to every replica.
//!
//! A shard whose files fail verification (length or SHA-256 against the
//! snapshot) fails its restore on either path: its primary never starts.
//!
//! Global state (persistent settings, templates, metadata customs) is
//! restored with one master task. A data stream the request names is
//! restored whole, under a free name only (renamed with its backing and
//! failure indices); a backing index restored by its own name stays a
//! plain index and never joins or replaces a live stream.
//!
//! Progress is persisted per target index in the restore entry
//! (`targets.<index>`: `pending` → `creating` → `restoring` → `done` or
//! `failed`, ES `RestoreInProgress`). The master task that creates a target
//! records its incarnation (`index_uuid`, [`file_restore::claim_target`]).
//! A target is published `done` once its data is in, and only then becomes
//! writable (a file restore creates it write-blocked; its final blocks are
//! applied after the publication). A restore resumed by a new master
//! ([`resume_orphan`]) never touches a `done` or `failed` target: it deletes
//! only the recorded incarnation of an incomplete one (fenced by its uuid,
//! never by name) and restores the incomplete targets again.
//!
//! No restore runs an ingest pipeline: the re-index path writes each
//! document exactly as the snapshot holds it ([`crate::front::bypass_pipelines`]),
//! and the target gets its default / final pipeline settings only after
//! its documents are in.

use super::common::{self, err, now_millis, service, state_of, PROGRESS_CUSTOM, REPOS_CUSTOM};
use super::file_restore::{self, FileRestoreSpec};
use crate::front::{RestReq, RestResp};
use crate::node::ClusterNode;
use crate::replication::ops::ReplicaOp;
use crate::replication::recovery::{RecoveryOptions, RecoverySource};
use elasticsearch_cluster::state::{ClusterState, IndexState};
use elasticsearch_engine::index::{IndexService, RecoverySource as EngineRecovery, ShardOpenSpec};
use elasticsearch_snapshots::util::{random_base64_uuid, CancelToken};
use elasticsearch_snapshots::{ExistingIndex, RestoreIndexPlan, RestorePlan, RestoreRequest, SnapshotError, SnapshotsService};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// `targets.<index>.state` values of a restore entry (see the module doc).
pub const TARGET_PENDING: &str = "pending";
pub const TARGET_CREATING: &str = "creating";
pub const TARGET_RESTORING: &str = "restoring";
pub const TARGET_DONE: &str = "done";
pub const TARGET_FAILED: &str = "failed";

/// Index settings that run ingest pipelines: a re-indexed target gets them
/// only after its documents are in.
const PIPELINE_SETTINGS: &[&str] = &["index.default_pipeline", "index.final_pipeline"];

const PER_COPY: &[&str] = &["index.uuid", "index.creation_date", "index.provided_name", "index.version.created", "index.history.uuid", "index.version.compatibility"];

tokio::task_local! {
    /// The restore caller's credential headers: its sub-requests (index
    /// creation, document bulk loads) are authorized as the caller. A
    /// restore resumed after a master change has none.
    static CREDENTIALS: crate::front::SharedHeaders;
}

fn rest(method: &str, uri: std::fmt::Arguments<'_>, body: Option<&Value>) -> elasticsearch_engine::error::Result<RestReq> {
    let borrowed=CREDENTIALS.try_with(|headers| {
        if headers.is_empty(){None}else{Some(RestReq::try_new_uri(method,uri,headers,&[],0,Default::default(),None))}
    }).ok().flatten();
    let mut req=match borrowed {Some(req)=>req?,None=>{let headers=crate::front::SharedHeaders::try_pairs(&[(crate::front::SYSTEM_HEADER,"true")])?;RestReq::try_new_uri(method,uri,&headers,&[],0,Default::default(),None)?}};
    req.push_header("content-type","application/json")?;
    if let Some(body)=body {req.set_json(body)?;} Ok(req)
}

/// Run a REST request through this node's cluster front, as a client
/// request (`hops` 0: a forwarded request may be answered by this node's
/// own engine copy, which need not hold the target's shards). Boxed so the
/// future does not embed `front::dispatch` (which embeds this module: a
/// recursive async type).
fn dispatch(node: &Arc<ClusterNode>, req: elasticsearch_engine::error::Result<RestReq>) -> std::pin::Pin<Box<dyn std::future::Future<Output = RestResp> + Send>> {
    let node = node.clone();
    Box::pin(async move { match req { Ok(req)=>crate::front::dispatch(&node,req).await, Err(e)=>RestResp::json(e.status,&e.to_json()) } })
}

fn resp_error(r: &RestResp, what: &str) -> SnapshotError {
    let v = r.body_json().unwrap_or(Value::Null);
    let reason = v["error"]["reason"].as_str().map(str::to_string).unwrap_or_else(|| String::from_utf8_lossy(&r.body).into_owned());
    let kind = v["error"]["type"].as_str().unwrap_or("exception").to_string();
    SnapshotError::new("snapshot_restore_exception", 500, format!("{what}: [{kind}] {reason}"))
}

/// `POST _snapshot/{repo}/{snapshot}/_restore`.
pub async fn restore(node: &Arc<ClusterNode>, req: &RestReq, repo: &str, snap: &str) -> RestResp {
    let svc = service(node).await;
    let body = req.json();
    if !req.body.is_empty() && body.is_none() {
        return RestResp::error(400, "parse_exception", "request body is required to be valid JSON");
    }
    let rr = match RestoreRequest::from_json(body.as_ref()) {
        Ok(r) => r,
        Err(e) => return err(&e),
    };
    let wait = req.params().get("wait_for_completion").map(|v| v != "false").unwrap_or(false);
    let st = node.state();
    let existing: BTreeMap<String, ExistingIndex> = st.metadata.indices.iter().map(|(n, m)| (n.clone(), ExistingIndex { open: m.state == IndexState::Open, number_of_shards: m.number_of_shards.max(1) })).collect();
    let closed = closed_incarnations(&st);
    let plan = match svc.plan_restore(repo, snap, &rr, &existing).await {
        Ok(p) => p,
        Err(e) => return err(&e),
    };
    if let Some(e) = data_stream_conflict(&st, &plan) {
        return err(&e);
    }
    // An index can only be the target of one restore at a time.
    for e in common::progress_entries(&st, "restore") {
        let busy: Vec<&str> = e["indices"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
        if let Some(i) = plan.indices.iter().find(|i| busy.contains(&i.target_index.as_str())) {
            return err(&SnapshotError::snapshot_restore(&plan.snapshot.snapshot_ref(), format!("cannot restore index [{}] because it's being restored", i.target_index)));
        }
    }
    let creds=match crate::reindex_multi::credentials(req) {Ok(v)=>v,Err(e)=>return RestResp::json(e.status,&e.to_json())};
    let uuid = random_base64_uuid();
    let s = state_of(node);
    s.restoring.lock().insert(uuid.clone());
    let entry = json!({
        "snapshot": plan.snapshot.snapshot, "repository": repo, "uuid": uuid, "state": "STARTED",
        "indices": plan.indices.iter().map(|i| i.target_index.clone()).collect::<Vec<_>>(),
        "targets": plan.indices.iter().map(|i| (i.target_index.clone(), json!({"state": TARGET_PENDING}))).collect::<Map<String, Value>>(),
        "master": node.local.id,
        // Lets a new master run the restore again after a failover.
        "request": body.clone().unwrap_or(Value::Null),
    });
    if let Err(e) = common::add_progress(node, "restore", entry).await {
        s.restoring.lock().remove(&uuid);
        if elasticsearch_cluster::coordination::CoordinationError::is_not_master(&e) {
            return crate::front::task_error(&e);
        }
        return RestResp::error(503, "concurrent_snapshot_execution_exception", &format!("failed to start restore: {e}"));
    }
    // Feature states replace the feature's current system indices: delete
    // them before restoring (ES `RestoreService`, production review #393).
    for (index, index_uuid) in feature_indices_to_delete(&node.state(), &plan) {
        if let Err(e) = delete_incarnation(node, &index, &index_uuid).await {
            s.restoring.lock().remove(&uuid);
            let _ = common::remove_progress(node, "restore", uuid.clone()).await;
            return err(&SnapshotError::snapshot_restore(&plan.snapshot.snapshot_ref(), format!("failed to delete the existing index [{index}] of a restored feature state: {e}")));
        }
    }
    let node2 = node.clone();
    let plan2 = plan.clone();
    let handle = tokio::spawn(CREDENTIALS.scope(creds, async move { run_restore(node2, svc, plan2, rr, uuid, false, closed).await }));
    if !wait {
        return RestResp::json(200, &json!({"accepted": true}));
    }
    match handle.await {
        Ok(Ok(ok)) => RestResp::json(200, &plan.response_json(ok)),
        Ok(Err(e)) => err(&e),
        Err(e) => RestResp::error(500, "illegal_state_exception", &e.to_string()),
    }
}

/// `resumed`: run again by a new master ([`rerun`]).
/// The closed indices of `st`, by name: the incarnation a plan validated
/// for replacement ([`replace_closed_incarnation`]).
fn closed_incarnations(st: &ClusterState) -> BTreeMap<String, String> {
    st.metadata.indices.iter().filter(|(_, m)| m.state == IndexState::Close).map(|(n, m)| (n.clone(), m.uuid.clone())).collect()
}

/// `closed`: the closed indices the plan validated ([`closed_incarnations`]).
async fn run_restore(node: Arc<ClusterNode>, svc: Arc<SnapshotsService>, plan: RestorePlan, rr: RestoreRequest, uuid: String, resumed: bool, closed: BTreeMap<String, String>) -> Result<u32, SnapshotError> {
    let result = async {
        let mut ok = 0u32;
        let mut first_err: Option<SnapshotError> = None;
        // Global state first (ES applies it with the indices in one
        // update): a restored index's lifecycle resumes under the restored
        // policies, never a same-name destination policy in the meantime
        // (production review #379).
        if let Some(g) = &plan.global_metadata {
            restore_global(&node, g).await?;
        }
        for ip in &plan.indices {
            let hook = BEFORE_RESTORE_INDEX.lock().clone();
            if let Some(hook) = hook {
                hook().await;
            }
            match restore_index(&node, &svc, &plan, ip, &rr, &uuid, resumed, closed.get(&ip.target_index).map(String::as_str)).await {
                Ok(n) => ok += n,
                Err(e) => {
                    tracing::warn!(index = %ip.target_index, "restore failed: {e}");
                    // Terminal: a resumed restore leaves a failed target
                    // as it is. (Not persisted when this node lost the
                    // mastership: the next master then restores it again.)
                    if let Err(pe) = set_target(&node, &uuid, &ip.target_index, TARGET_FAILED, None).await {
                        tracing::warn!(index = %ip.target_index, "cannot record the failed restore: {pe}");
                    }
                    first_err.get_or_insert(e);
                }
            }
        }
        if !plan.data_streams.is_empty() {
            restore_data_streams(&node, &plan, &uuid).await;
        }
        if ok == 0 {
            if let Some(e) = first_err {
                return Err(e);
            }
        }
        Ok(ok)
    }
    .await;
    if let Err(e) = common::remove_progress(&node, "restore", uuid.clone()).await {
        tracing::warn!("failed to remove restore entry: {e}");
    }
    state_of(&node).restoring.lock().remove(&uuid);
    result
}

/// Master only: run a restore again whose coordinating master is gone.
/// Targets the entry records as `done` or `failed` are left exactly as they
/// are (a completed target may have taken writes since). An incomplete
/// target's recorded incarnation, and only that one, is deleted; then the
/// stored request is planned again and its incomplete targets restored
/// under the same restore uuid. An entry without a stored request is
/// dropped.
pub fn resume_orphan(node: &Arc<ClusterNode>, e: &Value) {
    let uuid = e["uuid"].as_str().unwrap_or_default().to_string();
    if !state_of(node).restoring.lock().insert(uuid.clone()) {
        return;
    }
    let (node, e) = (node.clone(), e.clone());
    tokio::spawn(async move {
        tracing::info!(snapshot = %e["snapshot"], "restore was orphaned by a master change; running it again");
        if let Err(err) = rerun(&node, &e, &uuid).await {
            tracing::warn!(snapshot = %e["snapshot"], "orphaned restore could not be completed; restored indices may be incomplete: {err}");
        }
        // `run_restore` removes both itself; this covers the early exits.
        if common::progress_entries(&node.state(), "restore").iter().any(|x| x["uuid"].as_str() == Some(uuid.as_str())) {
            let _ = common::remove_progress(&node, "restore", uuid.clone()).await;
        }
        state_of(&node).restoring.lock().remove(&uuid);
    });
}

/// What a new master does with an orphaned restore entry's targets.
#[derive(Debug, Default, PartialEq)]
struct RerunPlan {
    /// `done` / `failed` targets: never deleted nor restored again.
    finished: std::collections::BTreeSet<String>,
    /// `done` targets whose final blocks were not applied yet.
    unreleased: Vec<(String, String, Map<String, Value>)>,
    /// (name, incarnation) of incomplete targets the restore created and
    /// that still exist as that incarnation.
    delete: Vec<(String, String)>,
}

fn rerun_plan(e: &Value, st: &ClusterState) -> Result<RerunPlan, String> {
    let Some(targets) = e["targets"].as_object() else {
        return Err("the restore entry records no per-index progress (an older version wrote it); its target indices are left as they are".into());
    };
    let mut plan = RerunPlan::default();
    for (name, t) in targets {
        let recorded = t["index_uuid"].as_str();
        let current = st.metadata.indices.get(name).map(|im| im.uuid.as_str());
        match t["state"].as_str().unwrap_or(TARGET_PENDING) {
            TARGET_DONE | TARGET_FAILED => {
                plan.finished.insert(name.clone());
                if t["state"] == TARGET_DONE && t["released"] != true {
                    if let (Some(u), Some(blocks)) = (recorded.filter(|u| current == Some(*u)), t["final_blocks"].as_object()) {
                        plan.unreleased.push((name.clone(), u.to_string(), blocks.clone()));
                    }
                }
            }
            // Created (claimed) by this restore and still that incarnation.
            _ => {
                if let Some(u) = recorded.filter(|u| current == Some(*u)) {
                    plan.delete.push((name.clone(), u.to_string()));
                }
            }
        }
    }
    Ok(plan)
}

async fn rerun(node: &Arc<ClusterNode>, e: &Value, uuid: &str) -> Result<u32, SnapshotError> {
    let repo = e["repository"].as_str().unwrap_or_default();
    let snap = e["snapshot"].as_str().unwrap_or_default();
    let body = e.get("request").ok_or_else(|| SnapshotError::new("snapshot_restore_exception", 500, "the restore entry carries no request".to_string()))?;
    let rr = RestoreRequest::from_json(if body.is_null() { None } else { Some(body) })?;
    let rp = rerun_plan(e, &node.state()).map_err(|m| SnapshotError::new("snapshot_restore_exception", 500, m))?;
    for (t, index_uuid, blocks) in &rp.unreleased {
        release_target(node, uuid, t, index_uuid, blocks.clone()).await?;
    }
    for (t, index_uuid) in &rp.delete {
        delete_incarnation(node, t, index_uuid).await.map_err(|m| SnapshotError::new("snapshot_restore_exception", 500, format!("failed to remove half-restored index [{t}]: {m}")))?;
    }
    let st = node.state();
    // Finished targets are not planned again (they are not restored over).
    let existing: BTreeMap<String, ExistingIndex> = st.metadata.indices.iter().filter(|(n, _)| !rp.finished.contains(*n))
        .map(|(n, m)| (n.clone(), ExistingIndex { open: m.state == IndexState::Open, number_of_shards: m.number_of_shards.max(1) })).collect();
    let closed = closed_incarnations(&st);
    let svc = service(node).await;
    let mut plan = svc.plan_restore(repo, snap, &rr, &existing).await?;
    plan.indices.retain(|i| !rp.finished.contains(&i.target_index));
    run_restore(node.clone(), svc, plan, rr, uuid.to_string(), true, closed).await
}

/// Test seam (hidden): awaited by a restore right before it replaces a
/// closed index (after its plan was validated).
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub static BEFORE_CLOSED_REPLACEMENT: parking_lot::Mutex<Option<Arc<dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync>>> = parking_lot::Mutex::new(None);

/// What [`replace_closed_incarnation`] found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Replaced {
    /// The validated closed incarnation, deleted.
    Deleted,
    /// No index of that name any more: nothing to replace.
    Gone,
    /// The validated incarnation, open again: left alone.
    Open,
    /// Another incarnation of the name: left alone.
    Recreated,
}

/// Delete `index` for a restore over it only while it is still incarnation
/// `index_uuid` and still closed, deciding and deleting in one master
/// metadata task.
async fn replace_closed_incarnation(node: &Arc<ClusterNode>, index: &str, index_uuid: &str) -> Result<Replaced, String> {
    let (name, uuid) = (index.to_string(), index_uuid.to_string());
    let outcome = Arc::new(parking_lot::Mutex::new(Replaced::Gone));
    let seen = outcome.clone();
    let version = node.metadata_service().pure(Box::new(move |_, state, delta| {
        let current = match delta.indices.get(&name) {
            Some(change) => change.as_ref().map(|im| (im.uuid.clone(), im.state)),
            None => state.metadata.indices.get(&name).map(|im| (im.uuid.clone(), im.state)),
        };
        *seen.lock() = match current {
            None => Replaced::Gone,
            Some((u, _)) if u != uuid => Replaced::Recreated,
            Some((_, IndexState::Close)) => {
                delta.indices.insert(name, None);
                Replaced::Deleted
            }
            Some(_) => Replaced::Open,
        };
    })).await?;
    let mut rx = node.subscribe();
    while rx.borrow().version < version {
        if rx.changed().await.is_err() {
            return Err("the node is shutting down".into());
        }
    }
    let outcome = *outcome.lock();
    Ok(outcome)
}

/// The current indices (name, uuid) of the feature states `plan` restores:
/// every index matching one of the feature's system index patterns.
fn feature_indices_to_delete(st: &ClusterState, plan: &RestorePlan) -> Vec<(String, String)> {
    let patterns: Vec<&str> = plan.feature_states.iter().filter_map(|f| elasticsearch_http::system_features::feature_index_patterns(f)).flatten().copied().collect();
    st.metadata.indices.iter().filter(|(n, _)| patterns.iter().any(|p| elasticsearch_snapshots::util::simple_match(p, n))).map(|(n, m)| (n.clone(), m.uuid.clone())).collect()
}

/// Delete incarnation `index_uuid` of `index`, and no other: an index of
/// that name created meanwhile by anyone else is left alone. One master
/// metadata task decides and deletes, so nothing can replace the index in
/// between. `Ok(false)`: that incarnation is gone already.
async fn delete_incarnation(node: &Arc<ClusterNode>, index: &str, index_uuid: &str) -> Result<bool, String> {
    let (name, uuid) = (index.to_string(), index_uuid.to_string());
    let deleted = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let marked = deleted.clone();
    let version = node.metadata_service().pure(Box::new(move |_, state, delta| {
        let current = match delta.indices.get(&name) {
            Some(change) => change.as_ref().map(|im| im.uuid.clone()),
            None => state.metadata.indices.get(&name).map(|im| im.uuid.clone()),
        };
        if current.as_deref() == Some(uuid.as_str()) {
            delta.indices.insert(name, None);
            marked.store(true, std::sync::atomic::Ordering::Release);
        }
    })).await?;
    // Applied here before a later task re-creates the name.
    let mut rx = node.subscribe();
    while rx.borrow().version < version {
        if rx.changed().await.is_err() {
            return Err("the node is shutting down".into());
        }
    }
    let deleted = deleted.load(std::sync::atomic::Ordering::Acquire);
    if deleted {
        tracing::info!(index, index_uuid, "deleted the incomplete restore target");
    }
    Ok(deleted)
}

/// Persist target `target`'s progress in restore `restore_uuid`'s entry.
/// `creating` forgets any recorded incarnation (the creating task records
/// the new one). `fence`: the change applies only while the target's
/// recorded incarnation is that one.
async fn set_target(node: &ClusterNode, restore_uuid: &str, target: &str, state: &'static str, fence: Option<(&str, Map<String, Value>)>) -> Result<(), String> {
    let (uuid, target) = (restore_uuid.to_string(), target.to_string());
    let fence = fence.map(|(u, extra)| (u.to_string(), extra));
    common::update_custom(node, PROGRESS_CUSTOM, move |v| {
        let Some(e) = v.get_mut("restore").and_then(Value::as_array_mut).and_then(|a| a.iter_mut().find(|e| e["uuid"].as_str() == Some(uuid.as_str()))) else {
            return Err(format!("restore [{uuid}] is no longer in progress"));
        };
        if !e["targets"].is_object() {
            e["targets"] = json!({});
        }
        let t = &mut e["targets"][&target];
        if !t.is_object() {
            *t = json!({});
        }
        if let Some((index_uuid, extra)) = fence {
            if t["index_uuid"].as_str() != Some(index_uuid.as_str()) {
                return Err(format!("restore target [{target}] is no longer incarnation [{index_uuid}]"));
            }
            for (k, x) in extra {
                t[&k] = x;
            }
        }
        if state == TARGET_CREATING {
            if let Some(o) = t.as_object_mut() {
                o.remove("index_uuid");
            }
        }
        t["state"] = json!(state);
        Ok(true)
    })
    .await
}

/// The incarnation of `target` restore `restore_uuid` created (recorded
/// by the creating task).
fn claimed_uuid(st: &ClusterState, restore_uuid: &str, target: &str) -> Option<String> {
    common::progress_entries(st, "restore").into_iter().find(|e| e["uuid"].as_str() == Some(restore_uuid))
        .and_then(|e| e["targets"][target]["index_uuid"].as_str().map(str::to_string))
}

/// After a target is published `done`: apply its final blocks (the
/// snapshot's, lifting the restore's own write block) and record that.
async fn release_target(node: &Arc<ClusterNode>, restore_uuid: &str, target: &str, index_uuid: &str, blocks: Map<String, Value>) -> Result<(), SnapshotError> {
    if !node.state().metadata.indices.get(target).is_some_and(|m| m.uuid == index_uuid) {
        return Ok(());
    }
    apply_blocks(node, target, blocks).await?;
    let mut released = Map::new();
    released.insert("released".into(), Value::Bool(true));
    set_target(node, restore_uuid, target, TARGET_DONE, Some((index_uuid, released))).await.map_err(|m| SnapshotError::illegal_state(format!("restore target [{target}]: {m}")))
}

/// Test seam (hidden): `RUSTIC_TEST_RESTORE_STALL` names targets (comma
/// separated) whose first restore (not one resumed by a new master) stops
/// for good right before the target is published `done`, as if its master
/// died there.
async fn stall_seam(target: &str, resumed: bool) {
    if resumed {
        return;
    }
    let stalled = std::env::var("RUSTIC_TEST_RESTORE_STALL").is_ok_and(|v| v.split(',').any(|t| t == target));
    if stalled {
        tracing::warn!(index = %target, "test seam: the restore stalls before publishing the target");
        std::future::pending::<()>().await;
    }
}

/// Publish target `target` (incarnation `index_uuid`) as restored: from
/// now on no resumed restore deletes or restores it again. Then its final
/// `blocks` are applied.
async fn publish_done(node: &Arc<ClusterNode>, restore_uuid: &str, target: &str, index_uuid: &str, blocks: Map<String, Value>, resumed: bool) -> Result<(), SnapshotError> {
    stall_seam(target, resumed).await;
    let mut extra = Map::new();
    extra.insert("final_blocks".into(), Value::Object(blocks.clone()));
    set_target(node, restore_uuid, target, TARGET_DONE, Some((index_uuid, extra))).await.map_err(|m| SnapshotError::illegal_state(format!("cannot publish restored index [{target}]: {m}")))?;
    release_target(node, restore_uuid, target, index_uuid, blocks).await
}

/// Wait until every primary of `index` (`shards` of them) is active.
async fn wait_for_primaries(node: &ClusterNode, index: &str, shards: u32, timeout: Duration) -> bool {
    let mut rx = node.subscribe();
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let ready = {
            let st = rx.borrow();
            (0..shards.max(1)).all(|k| st.routing_table.shard_table(index, k).and_then(|t| t.active_primary()).is_some())
        };
        if ready {
            return true;
        }
        if tokio::time::timeout_at(deadline, rx.changed()).await.is_err() {
            return false;
        }
    }
}

fn unwrap_mappings(m: &Value) -> Value {
    match m.as_object() {
        Some(o) if o.len() == 1 && o.contains_key("_doc") => o["_doc"].clone(),
        Some(_) => m.clone(),
        None => json!({}),
    }
}

/// Test seam (hidden): awaited by a running restore before each index.
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub static BEFORE_RESTORE_INDEX: parking_lot::Mutex<Option<Arc<dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync>>> = parking_lot::Mutex::new(None);

/// Node setting (not ES): `false` restores every index by re-indexing.
pub const FILE_BASED: &str = "snapshot.restore.file_based";

/// The engine's internal header for a restore's index create: no current
/// index template applies to the restored metadata (ES; production review
/// #378). Same name as `elasticsearch_http`'s.
const RESTORE_NO_TEMPLATES_HEADER: &str = "x-rustic-internal-restore-no-templates";

/// Index settings whose change on restore (`index_settings` /
/// `ignore_index_settings`) changes where documents route, how segments are
/// sorted or what they store: such a restore re-indexes (ES copies the
/// files regardless).
const LAYOUT_SETTINGS: &[&str] = &[
    "index.number_of_routing_shards", "index.routing_partition_size", "index.routing_path", "index.mode", "index.mapping.source.mode",
    "index.sort.field", "index.sort.order", "index.sort.mode", "index.sort.missing",
];

/// Restore one index; returns the number of shards restored.
#[allow(clippy::too_many_arguments)]
async fn restore_index(node: &Arc<ClusterNode>, svc: &Arc<SnapshotsService>, plan: &RestorePlan, ip: &RestoreIndexPlan, rr: &RestoreRequest, restore_uuid: &str, resumed: bool, closed_uuid: Option<&str>) -> Result<u32, SnapshotError> {
    let target = ip.target_index.clone();
    let started = now_millis();
    if ip.replaces_closed_index {
        let hook = BEFORE_CLOSED_REPLACEMENT.lock().clone();
        if let Some(hook) = hook {
            hook().await;
        }
        // Only the closed incarnation the plan validated, still closed, in
        // the one metadata task that deletes it: an index opened (and
        // written) or re-created under the name meanwhile is never
        // replaced (production review #130).
        let Some(uuid) = closed_uuid else {
            return Err(SnapshotError::snapshot_restore(&plan.snapshot.snapshot_ref(), format!("cannot restore index [{target}]: the closed index it replaces is unknown")));
        };
        match replace_closed_incarnation(node, &target, uuid).await.map_err(|m| SnapshotError::snapshot_restore(&plan.snapshot.snapshot_ref(), format!("failed to replace closed index [{target}]: {m}")))? {
            Replaced::Deleted | Replaced::Gone => {}
            Replaced::Open => return Err(SnapshotError::snapshot_restore(&plan.snapshot.snapshot_ref(), format!("cannot restore index [{target}] because an open index with same name already exists in the cluster. Either close or delete the existing index or restore the index under a different name by providing a rename pattern and replacement name"))),
            Replaced::Recreated => return Err(SnapshotError::snapshot_restore(&plan.snapshot.snapshot_ref(), format!("cannot restore index [{target}] because the closed index it was planned to replace was deleted and the name re-created meanwhile"))),
        }
    }
    let mut settings: Map<String, Value> = ip.metadata["settings"].as_object().cloned().unwrap_or_default();
    let mut blocks = Map::new();
    for k in settings.keys().cloned().collect::<Vec<_>>() {
        if PER_COPY.contains(&k.as_str()) || k.starts_with("index.resize.") || k.starts_with("index.shrink.") {
            settings.remove(&k);
        } else if k.starts_with("index.blocks.") {
            if let Some(v) = settings.remove(&k) {
                blocks.insert(k, v);
            }
        }
    }
    // Same shard count as the snapshotted index (ES restores the shard
    // layout as-is).
    settings.insert("index.number_of_shards".into(), json!(ip.number_of_shards.max(1).to_string()));
    // The lifecycle state is restored with the index (ES keeps the `ilm`
    // custom and rollover info of the snapshot's index metadata): the
    // target is created unmanaged, then gets its state and its policy in
    // one metadata change, so no lifecycle pass starts it from a new age.
    let lifecycle = restored_lifecycle(&mut settings, &ip.metadata);
    let mut body = json!({"settings": settings, "mappings": unwrap_mappings(&ip.metadata["mappings"])});
    if let Some(a) = ip.metadata.get("aliases").filter(|a| a.as_object().map(|o| !o.is_empty()).unwrap_or(false)) {
        body["aliases"] = a.clone();
    }
    // A `source` repository's shards are stored-fields-only segments.
    let source_only = svc.repository(&plan.repository).map(|r| r.settings().source_only).unwrap_or(false);
    match file_restore_blocker(node, ip, rr, source_only) {
        None => {
            // Created write-blocked: nothing writes to it before it is
            // published `done` (a resumed restore may delete it until then).
            let mut files_body = body.clone();
            files_body["settings"]["index.blocks.write"] = json!("true");
            match restore_index_files(node, plan, ip, restore_uuid, &files_body, lifecycle.as_ref()).await? {
                FilesOutcome::Restored(n, index_uuid) => {
                    let mut final_blocks = blocks;
                    final_blocks.entry("index.blocks.write").or_insert(Value::Null);
                    publish_done(node, restore_uuid, &target, &index_uuid, final_blocks, resumed).await?;
                    return Ok(n);
                }
                FilesOutcome::Reindex(reason, index_uuid) => {
                    tracing::warn!(index = %target, snapshot = %plan.snapshot.snapshot, "restoring by re-indexing the snapshot's documents: {reason}");
                    delete_incarnation(node, &target, &index_uuid).await
                        .map_err(|m| SnapshotError::snapshot_restore(&plan.snapshot.snapshot_ref(), format!("failed to remove [{target}] before re-indexing it: {m}")))?;
                }
            }
        }
        Some(reason) => tracing::info!(index = %target, snapshot = %plan.snapshot.snapshot, "restoring by re-indexing the snapshot's documents: {reason}"),
    }
    // Pipelines never run on restored documents: the target gets its
    // pipeline settings once the documents are in (and the documents are
    // written with every pipeline bypassed, `load_shard_docs`).
    let mut pipelines = Map::new();
    let mut docs_body = body;
    if let Some(settings) = docs_body["settings"].as_object_mut() {
        for k in PIPELINE_SETTINGS {
            if let Some(v) = settings.remove(*k) {
                pipelines.insert((*k).to_string(), v);
            }
        }
        // Write-blocked like a file restore's target: client writes would
        // be lost if a new master deleted the incomplete target. The
        // replay's own bulks pass the block (`load_shard_docs`).
        settings.insert("index.blocks.write".into(), json!("true"));
    }
    let mut final_blocks = blocks;
    final_blocks.entry("index.blocks.write").or_insert(Value::Null);
    restore_index_docs(node, svc, plan, ip, restore_uuid, &docs_body, DocsFinish { pipelines, blocks: final_blocks, resumed, lifecycle }, started, source_only).await
}

/// What a re-indexed target gets once its documents are in, in order.
struct DocsFinish {
    /// Its default / final pipeline settings.
    pipelines: Map<String, Value>,
    /// The snapshot's index blocks.
    blocks: Map<String, Value>,
    resumed: bool,
    /// The lifecycle state and policy the target adopts once created.
    lifecycle: Option<RestoredLifecycle>,
}

/// What a restored index's lifecycle continues from.
#[derive(Clone)]
struct RestoredLifecycle {
    /// `index.lifecycle.name` (taken out of the creation settings).
    policy: Option<Value>,
    /// The execution state and rollover info (`ilm::restore_index_lifecycle_state`).
    state: Value,
}

/// Take the lifecycle of the snapshot's index (`metadata`) out of the
/// target's creation `settings`: `None` when it has none to restore.
fn restored_lifecycle(settings: &mut Map<String, Value>, metadata: &Value) -> Option<RestoredLifecycle> {
    let mut adjusted = metadata.clone();
    // The policy as the restore request left it (`index_settings` /
    // `ignore_index_settings` are applied to `settings`).
    adjusted["settings"] = Value::Object(settings.clone());
    let state = elasticsearch_engine::ilm::lifecycle_from_snapshot(&adjusted)?;
    let policy = settings.remove("index.lifecycle.name").filter(|p| p.as_str().is_some_and(|p| !p.is_empty()));
    Some(RestoredLifecycle { policy, state })
}

/// Give the just-created target (incarnation `index_uuid`) its restored
/// lifecycle state, then its policy, in one published metadata change.
async fn adopt_lifecycle(node: &Arc<ClusterNode>, target: &str, index_uuid: &str, lifecycle: Option<&RestoredLifecycle>) -> Result<(), SnapshotError> {
    let Some(lc) = lifecycle.cloned() else { return Ok(()) };
    let (t, u) = (target.to_string(), index_uuid.to_string());
    let applied = node.mutate_engine_metadata(move |engine| {
        let svc = engine.get_index(&t).filter(|s| s.uuid == u).ok_or_else(|| elasticsearch_engine::error::EsError::index_not_found(&t))?;
        elasticsearch_engine::ilm::restore_index_lifecycle_state(engine, &svc, Some(&lc.state))?;
        if let Some(policy) = lc.policy {
            let mut flat = elasticsearch_engine::settings::FlatSettings::new();
            flat.insert("index.lifecycle.name".into(), policy);
            svc.put_settings(flat, false)?;
        }
        Ok(())
    }).await;
    applied.map(|_| ()).map_err(|e| SnapshotError::illegal_state(format!("cannot restore the lifecycle state of [{target}]: {}", e.reason)))
}

/// Mark `target` `creating` in its restore entry (the creating task then
/// records its incarnation).
async fn announce_creation(node: &ClusterNode, plan: &RestorePlan, restore_uuid: &str, target: &str) -> Result<(), SnapshotError> {
    set_target(node, restore_uuid, target, TARGET_CREATING, None).await.map_err(|e| SnapshotError::snapshot_restore(&plan.snapshot.snapshot_ref(), format!("cannot start restoring [{target}]: {e}")))
}

/// The incarnation of `target` this restore just created: the one the
/// creating task recorded, if `target` still is it.
fn created_incarnation(node: &ClusterNode, plan: &RestorePlan, restore_uuid: &str, target: &str) -> Result<String, SnapshotError> {
    let st = node.state();
    let current = st.metadata.indices.get(target).map(|m| m.uuid.clone());
    match (current, claimed_uuid(&st, restore_uuid, target)) {
        (Some(c), Some(claimed)) if c == claimed => Ok(c),
        (None, _) => Err(SnapshotError::snapshot_restore(&plan.snapshot.snapshot_ref(), format!("index [{target}] was removed while it was being restored"))),
        _ => Err(SnapshotError::snapshot_restore(&plan.snapshot.snapshot_ref(), format!("index [{target}] was not created by this restore"))),
    }
}

/// Why `ip` cannot be restored from the snapshot's files here (it is
/// re-indexed instead), if anything the master can tell before creating it.
fn file_restore_blocker(node: &ClusterNode, ip: &RestoreIndexPlan, rr: &RestoreRequest, source_only: bool) -> Option<String> {
    if source_only {
        return Some("the repository is source-only (its shards hold stored fields only)".into());
    }
    let disabled = |v: &Value| matches!(v, Value::Bool(false)) || v.as_str() == Some("false");
    if node.config.settings.get(FILE_BASED).is_some_and(disabled) {
        return Some(format!("[{FILE_BASED}] is false"));
    }
    let st = node.state();
    if let Some(n) = st.data_nodes().find(|n| !crate::internal_capabilities::supports(n, crate::internal_capabilities::Capability::SnapshotFileRestoreV1)) {
        return Some(format!("node [{}] cannot restore snapshot files (an older build)", n.name));
    }
    let settings = &ip.metadata["settings"];
    let n = ip.number_of_shards.max(1);
    let declared = settings.get("index.number_of_shards").and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok())));
    if declared.is_some_and(|d| d != u64::from(n)) {
        return Some(format!("the snapshot's index declares [{}] shards but has [{n}] shard snapshots", declared.unwrap_or_default()));
    }
    use elasticsearch_engine::version::IndexVersion;
    let created = IndexVersion::from_setting(settings.get("index.version.created"));
    if created.0 != IndexVersion::CURRENT.0 {
        return Some(format!("the snapshot's index was created with index version [{}], this cluster creates [{}]", created.0, IndexVersion::CURRENT.0));
    }
    if n > 1 {
        let routing_shards = |v: &Value| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok()));
        let expected = settings.get("index.number_of_routing_shards").and_then(routing_shards)
            .unwrap_or_else(|| u64::from(elasticsearch_cluster::routing::calculate_num_routing_shards(n)));
        if let Some(recorded) = ip.metadata.get("routing_num_shards").and_then(routing_shards).filter(|r| *r > 0 && *r != expected) {
            return Some(format!("the snapshot's index routes over [{recorded}] routing shards, the restored index over [{expected}]"));
        }
    }
    let sensitive = |k: &str| LAYOUT_SETTINGS.contains(&k) || k.starts_with("index.sort.");
    if let Some(k) = rr.index_settings.keys().find(|k| sensitive(k.as_str())) {
        return Some(format!("the restore changes [{k}]"));
    }
    if let Some(p) = rr.ignore_index_settings.iter().find(|p| LAYOUT_SETTINGS.iter().any(|k| elasticsearch_snapshots::util::simple_match(p, k))) {
        return Some(format!("the restore removes [{p}]"));
    }
    None
}

enum FilesOutcome {
    /// Shards whose primaries started from the snapshot's files, and the
    /// restored incarnation.
    Restored(u32, String),
    /// The files cannot be adopted (found by a primary's node): re-index
    /// (after deleting the incarnation created for the files).
    Reindex(String, String),
}

/// The highest primary term the snapshot's index metadata records.
fn snapshot_primary_term(metadata: &Value) -> u64 {
    metadata["primary_terms"].as_object().map(|m| m.values().filter_map(Value::as_u64).max().unwrap_or(0)).unwrap_or(0)
}

/// Set (or with `None` remove) the file restore spec of `target` in
/// restore `restore_uuid`'s entry.
async fn arm(node: &ClusterNode, restore_uuid: &str, target: &str, spec: Option<&FileRestoreSpec>) -> Result<(), String> {
    let (uuid, target) = (restore_uuid.to_string(), target.to_string());
    let spec = spec.map(serde_json::to_value).transpose().map_err(|e| e.to_string())?;
    common::update_custom(node, PROGRESS_CUSTOM, move |v| {
        let Some(e) = v.get_mut("restore").and_then(Value::as_array_mut).and_then(|a| a.iter_mut().find(|e| e["uuid"].as_str() == Some(uuid.as_str()))) else {
            return Err(format!("restore [{uuid}] is no longer in progress"));
        };
        match spec {
            Some(spec) => {
                if !e["files"].is_object() {
                    e["files"] = json!({});
                }
                e["files"][&target] = spec;
            }
            None => {
                let Some(files) = e.get_mut("files").and_then(Value::as_object_mut) else { return Ok(false) };
                if files.remove(&target).is_none() {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    })
    .await
}

/// Create `ip`'s target index with ES's restore routing and wait for each
/// restored primary to start or fail.
async fn restore_index_files(node: &Arc<ClusterNode>, plan: &RestorePlan, ip: &RestoreIndexPlan, restore_uuid: &str, body: &Value, lifecycle: Option<&RestoredLifecycle>) -> Result<FilesOutcome, SnapshotError> {
    let target = ip.target_index.as_str();
    let snap_ref = plan.snapshot.snapshot_ref();
    let spec = FileRestoreSpec {
        repository: plan.repository.clone(),
        snapshot: plan.snapshot.snapshot.clone(),
        snapshot_uuid: plan.snapshot.uuid.clone(),
        version: plan.snapshot.version.clone(),
        source_index: ip.source_index.clone(),
        index_id: ip.index_id.clone(),
        primary_term: snapshot_primary_term(&ip.metadata),
        empty_shards: ip.failed_shards.clone(),
        replaces_closed: ip.replaces_closed_index,
        armed: true,
        index_uuid: None,
    };
    arm(node, restore_uuid, target, Some(&spec)).await.map_err(|e| SnapshotError::snapshot_restore(&snap_ref, format!("cannot start restoring [{target}]: {e}")))?;
    announce_creation(node, plan, restore_uuid, target).await?;
    // Primaries start only once their files are in: no wait here.
    let prepared_request = rest("PUT", format_args!("/{target}?wait_for_active_shards=0"), Some(body)).and_then(|mut r| { r.push_header(RESTORE_NO_TEMPLATES_HEADER, "1")?; Ok(r) });
    let r = dispatch(node, prepared_request).await;
    if !r.is_success() {
        let _ = arm(node, restore_uuid, target, None).await;
        return Err(resp_error(&r, &format!("failed to create index [{target}]")));
    }
    let uuid = created_incarnation(node, plan, restore_uuid, target)?;
    adopt_lifecycle(node, target, &uuid, lifecycle).await?;
    let st = node.state();
    if file_restore::file_restore_of(&st, target, &uuid).is_none() {
        // Created by another path than the armed one: its primaries are
        // new empty shards.
        let _ = arm(node, restore_uuid, target, None).await;
        return Ok(FilesOutcome::Reindex("the index was created without restore routing".into(), uuid));
    }
    let outcomes = wait_for_restored_primaries(node, target, &uuid, &ip.shards).await.map_err(|e| SnapshotError::snapshot_restore(&snap_ref, e))?;
    let mut restored = 0u32;
    let mut failures = Vec::new();
    for (shard, outcome) in outcomes {
        match outcome {
            Ok(()) => restored += 1,
            Err(details) if file_restore::requires_reindex(&details) => return Ok(FilesOutcome::Reindex(format!("shard [{shard}]: {details}"), uuid)),
            Err(details) => failures.push(format!("[{target}][{shard}] {details}")),
        }
    }
    if restored == 0 && !failures.is_empty() {
        return Err(SnapshotError::snapshot_restore(&snap_ref, format!("failed to restore [{target}]: {}", failures.join("; "))));
    }
    if !failures.is_empty() {
        tracing::warn!(index = %target, failed = failures.len(), "shards failed to restore and stay unassigned: {}", failures.join("; "));
    }
    tracing::info!(index = %target, shards = restored, "restored from the snapshot's files");
    Ok(FilesOutcome::Restored(restored, uuid))
}

/// Wait until every primary of `shards` of `target` (incarnation `uuid`)
/// started from its restore (`Ok`) or failed it (`Err`: the failure's
/// details). A failed restore is never retried (`restore_in_progress`
/// decider), so its primary stays unassigned with the failure recorded.
async fn wait_for_restored_primaries(node: &ClusterNode, target: &str, uuid: &str, shards: &[u32]) -> Result<Vec<(u32, Result<(), String>)>, String> {
    use elasticsearch_cluster::state::{RecoverySource as Source, ShardState};
    let mut rx = node.subscribe();
    loop {
        let outcome = {
            let st = rx.borrow();
            if !st.metadata.indices.get(target).is_some_and(|m| m.uuid == uuid) {
                return Err(format!("index [{target}] was deleted while it was being restored"));
            }
            let mut out = Vec::with_capacity(shards.len());
            for &k in shards {
                let Some(t) = st.routing_table.shard_table(target, k) else { break };
                if t.active_primary().is_some() {
                    out.push((k, Ok(())));
                    continue;
                }
                let failed = t.primary().filter(|p| p.state == ShardState::Unassigned && matches!(p.recovery_source, Some(Source::Snapshot { .. })))
                    .and_then(|p| p.unassigned_info.as_ref()).filter(|u| u.failed_allocations > 0);
                match failed {
                    Some(u) => out.push((k, Err(u.details.clone().unwrap_or_else(|| "failed to restore".into())))),
                    None => break,
                }
            }
            (out.len() == shards.len()).then_some(out)
        };
        if let Some(out) = outcome {
            return Ok(out);
        }
        if rx.changed().await.is_err() {
            return Err("the node is shutting down".into());
        }
    }
}

/// Restore the index blocks of the snapshot (once its data is in).
async fn apply_blocks(node: &Arc<ClusterNode>, target: &str, blocks: Map<String, Value>) -> Result<(), SnapshotError> {
    if blocks.is_empty() {
        return Ok(());
    }
    let prepared_request = rest("PUT", format_args!("/{target}/_settings"), Some(&Value::Object(blocks)));
    let r = dispatch(node, prepared_request).await;
    if !r.is_success() {
        return Err(resp_error(&r, "restoring index blocks"));
    }
    Ok(())
}

/// The re-indexing restore of one index (see the module doc).
#[allow(clippy::too_many_arguments)]
async fn restore_index_docs(node: &Arc<ClusterNode>, svc: &Arc<SnapshotsService>, plan: &RestorePlan, ip: &RestoreIndexPlan, restore_uuid: &str, body: &Value, finish: DocsFinish, started: i64, source_only: bool) -> Result<u32, SnapshotError> {
    let target = ip.target_index.clone();
    announce_creation(node, plan, restore_uuid, &target).await?;
    let prepared_request = rest("PUT", format_args!("/{target}?wait_for_active_shards=1"), Some(body)).and_then(|mut r| { r.push_header(RESTORE_NO_TEMPLATES_HEADER, "1")?; Ok(r) });
    let r = dispatch(node, prepared_request).await;
    if !r.is_success() {
        return Err(resp_error(&r, &format!("failed to create index [{target}]")));
    }
    let created = created_incarnation(node, plan, restore_uuid, &target)?;
    adopt_lifecycle(node, &target, &created, finish.lifecycle.as_ref()).await?;
    if !wait_for_primaries(node, &target, ip.number_of_shards, Duration::from_secs(60)).await {
        return Err(SnapshotError::snapshot_restore(&plan.snapshot.snapshot_ref(), format!("primary of [{target}] did not become active")));
    }
    let mut restored = 0u32;
    let mut docs = 0u64;
    // `_recovery` per restored shard: (shard, files, bytes).
    let mut per_shard: Vec<(u32, u64, u64)> = Vec::new();
    for shard in &ip.shards {
        let dir = common::staging_root(node).join(format!("restore-{}-{}-{shard}", common::dir_name(restore_uuid), common::dir_name(&target)));
        let _ = tokio::fs::remove_dir_all(&dir).await;
        let cancel = CancelToken::new();
        let res = svc.restore_shard(&plan.repository, &plan.snapshot.snapshot, &ip.source_index, *shard, &dir, &cancel).await;
        let res = match res {
            Ok(r) => r,
            Err(e) => {
                let _ = tokio::fs::remove_dir_all(&dir).await;
                return Err(e);
            }
        };
        // Bytes are the staged store files' (the index/shard bookkeeping
        // sidecars are not store files).
        let mut bytes = 0u64;
        match common::list_files(&dir) {
            Ok(l) => {
                for (_, len) in l.iter().filter(|(n, _)| !(n == "index_meta.json" || n.ends_with("shard_state.json"))) {
                    bytes += len;
                }
            }
            Err(_) => bytes += res.bytes,
        }
        let loaded = load_docs(node, &target, &created, dir.clone()).await;
        let _ = tokio::fs::remove_dir_all(&dir).await;
        let shard_docs = loaded.map_err(|e| SnapshotError::snapshot_restore(&plan.snapshot.snapshot_ref(), format!("failed to restore [{target}][{shard}]: {e}")))?;
        docs += shard_docs;
        // `_recovery` files: the Lucene files ES would report for the shard
        // (as the single-node engine does), not our store's own layout.
        per_shard.push((*shard, elasticsearch_engine::snapshot::lucene_recovered_files(shard_docs, source_only), bytes));
        restored += 1;
    }
    // Restored documents are searchable at once (ES restores whole
    // segments) and must survive the copy being reopened by the settings
    // change below: refresh and commit them first.
    let prepared_request = rest("POST", format_args!("/{target}/_flush"), None);
    let flushed = dispatch(node, prepared_request).await;
    if !flushed.is_success() { return Err(resp_error(&flushed,"flushing restored index")); }
    // Visibility must be checked before restoring a saved read block.
    refresh_until_visible(node, &target, docs).await?;
    if !finish.pipelines.is_empty() {
        let prepared_request = rest("PUT", format_args!("/{target}/_settings"), Some(&Value::Object(finish.pipelines)));
        let r = dispatch(node, prepared_request).await;
        if !r.is_success() {
            return Err(resp_error(&r, "restoring the index's pipeline settings"));
        }
    }
    let st = node.state();
    let index_uuid = created.clone();
    let primary_node = |k: u32| st.routing_table.shard_table(&target, k).and_then(|t| t.active_primary()).and_then(|p| p.node_id.clone()).unwrap_or_default();
    let (files, bytes) = per_shard.iter().fold((0, 0), |(f, b), (_, sf, sb)| (f + sf, b + sb));
    let shards: Vec<Value> = per_shard.iter().map(|(k, f, b)| json!({"id": k, "files": f, "bytes": b, "node": primary_node(*k)})).collect();
    let rec = json!({
        "index_uuid": index_uuid, "repository": plan.repository, "snapshot": plan.snapshot.snapshot,
        "snapshot_uuid": plan.snapshot.uuid, "version": plan.snapshot.version, "source_index": ip.source_index,
        "restore_uuid": restore_uuid, "files": files, "bytes": bytes, "docs": docs, "shards": shards,
        "start_time_millis": started, "stop_time_millis": now_millis(), "node": primary_node(0),
    });
    let t2 = target.clone();
    let _ = common::update_custom(node, PROGRESS_CUSTOM, move |v| {
        if !v.is_object() {
            *v = json!({});
        }
        if !v["recoveries"].is_object() {
            v["recoveries"] = json!({});
        }
        v["recoveries"][&t2] = rec;
        Ok(true)
    })
    .await;
    drop(st);
    publish_done(node, restore_uuid, &target, &created, finish.blocks, finish.resumed).await?;
    Ok(restored)
}

/// Refresh `target` until its `docs` restored documents are searchable. A
/// refresh can land on a shard copy that the node replaces right after
/// (the copy reopened from its store), so the count is checked.
async fn refresh_until_visible(node: &Arc<ClusterNode>, target: &str, docs: u64) -> Result<(),SnapshotError> {
    for attempt in 0..50 {
        let prepared_request = rest("POST", format_args!("/{target}/_refresh"), None);
        let refreshed = dispatch(node, prepared_request).await;
        if !refreshed.is_success() { return Err(resp_error(&refreshed,"refreshing restored index")); }
        let prepared_request = rest("GET", format_args!("/{target}/_search?size=0&request_cache=false&track_total_hits=true"), None);
        let r = dispatch(node, prepared_request).await;
        let n = r.body_json().and_then(|v| v["hits"]["total"]["value"].as_u64());
        if !r.is_success() { return Err(resp_error(&r,"checking restored visibility")); }
        if n.is_some_and(|n| n >= docs) { return Ok(()); }
        if attempt == 49 {
            tracing::warn!(index = %target, "restored documents are not all searchable yet");
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Ok(())
}

/// Stream every live document of the staged shard into `target`.
async fn load_docs(node: &Arc<ClusterNode>, target: &str, index_uuid: &str, dir: PathBuf) -> Result<u64, String> {
    let svc = tokio::task::spawn_blocking(move || {
        let svc = IndexService::open(&dir)?;
        // A copy staged from a sparse (cluster-managed) index keeps
        // `"sparse": true` in its `index_meta.json`, so the engine reopens it
        // with no shard open. The image is that one copy under `shard-0/`:
        // open it explicitly (its docs are exactly the snapshotted shard's).
        if svc.is_sparse() && svc.local_shards().is_empty() {
            svc.open_shard(0, ShardOpenSpec { allocation_id: "snapshot-restore".to_string(), primary: true, primary_term: 1, recovery: EngineRecovery::ExistingStore })?;
        }
        Ok::<_, elasticsearch_engine::error::EsError>(svc)
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| format!("cannot open snapshot shard: {}", e.reason))?;
    let svc = Arc::new(svc);
    let mut total = 0u64;
    // The image is one shard copy under `shard-0/`; with N > 1 the engine
    // migrated it to the shards its docs route to, so read every shard.
    let ids: Vec<u32> = svc.local_shards().iter().map(|s| s.id).collect();
    for sid in ids {
        total += load_shard_docs(node, target, index_uuid, svc.clone(), sid).await?;
    }
    Ok(total)
}

async fn load_shard_docs(node: &Arc<ClusterNode>, target: &str, index_uuid: &str, svc: Arc<IndexService>, sid: u32) -> Result<u64, String> {
    // A timestamp-sorted index searches its segments newest first: the
    // restored copy gets the source's segments (a refresh after each).
    let keep_segments = svc.segments_sorted_on_timestamp();
    let src = RecoverySource::start(svc, sid, 1, RecoveryOptions { segment_batches: keep_segments, ..Default::default() }).map_err(|e| e.to_string())?;
    let mut src = Some(src);
    let mut total = 0u64;
    loop {
        let mut s = src.take().expect("source");
        let (s, batch) = tokio::task::spawn_blocking(move || {
            let b = s.next_snapshot_batch();
            (s, b)
        })
        .await
        .map_err(|e| e.to_string())?;
        src = Some(s);
        let Some(ops) = batch.map_err(|e| e.to_string())? else { break };
        let n=ops.iter().filter(|op|matches!(op,ReplicaOp::Index{..})).count() as u64;
        if n==0 {continue;}
        // Every pipeline bypassed (default and final included): the
        // snapshot holds the documents as ingest left them.
        let routings: Vec<Option<String>> = ops.iter().filter_map(|op| match op { ReplicaOp::Index { routing, .. } => Some(routing.clone()), _ => None }).collect();
        let req=rest("POST",format_args!("/_bulk"),None)
            .and_then(|mut req|{req.replace_header("content-type","application/x-ndjson")?;crate::front::bypass_pipelines(&mut req, routings)?;
                // The target is write-blocked until it is published: only
                // this replay writes to it.
                crate::front::restore_replay(&mut req, target, index_uuid)?;req.try_with_encoded_body(|out| {
                #[derive(serde::Serialize)] struct Meta<'a>{_index:&'a str,_id:&'a str,version:u64,version_type:&'static str,#[serde(skip_serializing_if="Option::is_none")]routing:Option<&'a str>}
                #[derive(serde::Serialize)] struct Action<'a>{index:Meta<'a>}
                for op in ops.iter() {
                    let ReplicaOp::Index{id,source_json,routing,version,..}=op else{continue};
                    let action=Action{index:Meta{_index:target,_id:id,version:(*version).max(1) as u64,version_type:"external",routing:routing.as_deref()}};
                    serde_json::to_writer(&mut *out,&action)?;out.write_all(b"\n")?;
                    out.write_all(if source_json.is_empty(){b"{}"}else{source_json.as_bytes()})?;out.write_all(b"\n")?;
                } Ok(())
            })});
        let r = dispatch(node, req).await;
        let v = r.body_json().unwrap_or(Value::Null);
        if !r.is_success() {
            return Err(format!("bulk failed: {}", String::from_utf8_lossy(&r.body)));
        }
        if v["errors"].as_bool().unwrap_or(false) {
            let first = v["items"].as_array().and_then(|a| a.iter().find_map(|i| i.as_object().and_then(|o| o.values().next()).and_then(|x| x.get("error")).cloned()));
            return Err(format!("bulk item failed: {}", first.unwrap_or(Value::Null)));
        }
        total += n;
        if keep_segments && src.as_ref().is_some_and(|s| s.at_segment_boundary()) {
            let prepared_request = rest("POST", format_args!("/{target}/_refresh"), None);
            let refreshed = dispatch(node, prepared_request).await;
            if !refreshed.is_success() { return Err(format!("refresh failed: {}",String::from_utf8_lossy(&refreshed.body))); }
        }
    }
    Ok(total)
}

/// Restore persistent settings, templates and metadata customs.
async fn restore_global(node: &Arc<ClusterNode>, g: &Value) -> Result<(), SnapshotError> {
    let g = g.clone();
    let task: elasticsearch_cluster::coordination::StateTask = Box::new(move |s: &ClusterState| {
        let mut next = s.clone();
        if let Some(ps) = g["persistent_settings"].as_object() {
            next.metadata.persistent_settings = ps.clone();
        }
        // ES: composable and component templates are replaced by the
        // snapshot's (a destination-only template must not survive a
        // global-state rollback: production review #377); legacy ones merge.
        for (key, dst, replace) in [("index_templates", &mut next.metadata.index_templates, true), ("component_templates", &mut next.metadata.component_templates, true), ("legacy_templates", &mut next.metadata.legacy_templates, false)] {
            if let Some(m) = g[key].as_object() {
                if replace {
                    dst.clear();
                }
                for (k, v) in m {
                    dst.insert(k.clone(), v.clone());
                }
            }
        }
        if let Some(c) = g["customs"].as_object() {
            for (k, v) in c {
                if matches!(k.as_str(), REPOS_CUSTOM | PROGRESS_CUSTOM | "node_shutdown" | crate::mirror::DATA_STREAMS_CUSTOM | elasticsearch_cluster::compatibility::SEGMENT_FORMAT_CUSTOM) {
                    continue;
                }
                next.metadata.customs.insert(k.clone(), v.clone());
            }
        }
        next.metadata.version += 1;
        Ok(next)
    });
    node.coord().submit(task).await.map(|_| ()).map_err(|e| SnapshotError::illegal_state(format!("failed to restore global state: {e}")))
}

/// Why `plan` would replace or join a live data stream (#103), checked
/// before anything is mutated: a restored stream's name is taken by a live
/// stream or index, or a restored index is a live stream's name or one of
/// its backing / failure indices. ES never restores into a live stream:
/// a backing index restored by its own name stays a plain index, and a
/// whole stream is restored only under a free name.
fn data_stream_conflict(st: &ClusterState, plan: &RestorePlan) -> Option<SnapshotError> {
    let live = st.metadata.customs.get(crate::mirror::DATA_STREAMS_CUSTOM).and_then(Value::as_object);
    let snap = plan.snapshot.snapshot_ref();
    let member_of = |index: &str| -> Option<String> {
        live?.iter().find(|(_, d)| ["indices", "failure_indices"].iter().any(|k| d[*k].as_array().is_some_and(|a| a.iter().any(|i| i.as_str() == Some(index))))).map(|(n, _)| n.clone())
    };
    for ds in &plan.data_streams {
        let t = &ds.target_name;
        if live.is_some_and(|m| m.contains_key(t)) {
            return Some(SnapshotError::snapshot_restore(&snap, format!("cannot restore data stream [{t}] because a data stream with the same name already exists in the cluster. Either delete the existing data stream or restore the data stream under a different name by providing a rename pattern and replacement name")));
        }
        if st.metadata.indices.contains_key(t) {
            return Some(SnapshotError::snapshot_restore(&snap, format!("cannot restore data stream [{t}] because an index with the same name already exists in the cluster")));
        }
    }
    for ip in &plan.indices {
        let t = &ip.target_index;
        if live.is_some_and(|m| m.contains_key(t)) {
            return Some(SnapshotError::snapshot_restore(&snap, format!("cannot restore index [{t}] because a data stream with the same name already exists in the cluster")));
        }
        if let Some(ds) = member_of(t) {
            return Some(SnapshotError::snapshot_restore(&snap, format!("cannot restore index [{t}] because it is a backing index of the live data stream [{ds}]. Either delete the data stream or restore the index under a different name by providing a rename pattern and replacement name")));
        }
    }
    None
}

/// Create the data streams the request restores whole, each naming only
/// its own restored members (plain indices restored by name never join a
/// stream). Checked in the publication task itself, against the state it
/// publishes: a member is attached only if it is still the incarnation this
/// restore (`restore_uuid`) created, and no other data stream holds it
/// (an index replaced, or added to another stream, while later targets
/// were still restoring is left out); a stream that exists by now is never
/// replaced.
async fn restore_data_streams(node: &Arc<ClusterNode>, plan: &RestorePlan, restore_uuid: &str) {
    let streams: Vec<(String, Value)> = plan.data_streams.iter().map(|ds| (ds.target_name.clone(), ds.record.clone())).collect();
    if streams.is_empty() {
        return;
    }
    let restore_uuid = restore_uuid.to_string();
    let task: elasticsearch_cluster::coordination::StateTask = Box::new(move |st: &ClusterState| {
        let key = crate::mirror::DATA_STREAMS_CUSTOM;
        let mut v = st.metadata.customs.get(key).cloned().unwrap_or_else(|| json!({}));
        if !v.is_object() {
            v = json!({});
        }
        // Indices other streams already hold.
        let held: std::collections::HashSet<String> = v.as_object().into_iter().flatten()
            .flat_map(|(_, d)| ["indices", "failure_indices"].into_iter().flat_map(move |k| d[k].as_array().cloned().unwrap_or_default()))
            .filter_map(|i| i.as_str().map(str::to_string))
            .collect();
        let mut changed = false;
        for (name, record) in &streams {
            if v.get(name).is_some() {
                tracing::warn!(data_stream = %name, "a data stream with this name exists; the restored one is not applied over it");
                continue;
            }
            let mut d = record.clone();
            for k in ["indices", "failure_indices"] {
                let kept: Vec<Value> = d[k].as_array().map(|a| a.iter().filter(|i| {
                    let Some(n) = i.as_str() else { return false };
                    let ours = st.metadata.indices.get(n).is_some_and(|im| claimed_uuid(st, &restore_uuid, n).as_deref() == Some(im.uuid.as_str()));
                    if !ours || held.contains(n) {
                        tracing::warn!(data_stream = %name, index = %n, "a member is no longer the index this restore created (or another data stream holds it): left out of the restored data stream");
                        return false;
                    }
                    true
                }).cloned().collect()).unwrap_or_default();
                d[k] = Value::Array(kept);
            }
            if d["indices"].as_array().is_none_or(|a| a.is_empty()) {
                tracing::warn!(data_stream = %name, "no backing index of the data stream was restored; not restoring it");
                continue;
            }
            v[name] = d;
            changed = true;
        }
        if !changed {
            return Err("no change".into());
        }
        let mut next = st.clone();
        next.metadata.customs.insert(key.to_string(), v);
        next.metadata.version += 1;
        Ok(next)
    });
    match node.coord().submit(task).await {
        Ok(_) => {}
        Err(e) if e == "no change" => {}
        Err(e) => tracing::warn!("failed to restore data streams: {e}"),
    }
}

/// Patch the admitted response from borrowed published metadata. Replacements
/// keep their actual decoded owners with the response until serialization.
/// A re-indexed restore's primaries report the restore (`SNAPSHOT`), each
/// shard from its own record (older records describe shard 0 only).
pub fn patch_recovery_response(st:&ClusterState,body:&mut crate::owned_json::OwnedJson)->Result<(),elasticsearch_engine::error::EsError>{
    use elasticsearch_engine::error::EsError;
    let Some(records)=st.metadata.customs.get(PROGRESS_CUSTOM).and_then(|v|v["recoveries"].as_object())else{return Ok(())};
    for (index,r) in records {
        if !st.metadata.indices.get(index).is_some_and(|im|Some(im.uuid.as_str())==r["index_uuid"].as_str()){continue}
        if body.get(index).and_then(|v|v["shards"].as_array()).is_none(){continue}
        let shards:Vec<(u64,u64,u64,&Value)>=match r["shards"].as_array(){
            Some(a)=>a.iter().map(|s|(s["id"].as_u64().unwrap_or(0),s["files"].as_u64().unwrap_or(0),s["bytes"].as_u64().unwrap_or(0),&s["node"])).collect(),
            None=>vec![(0,r["files"].as_u64().unwrap_or(0),r["bytes"].as_u64().unwrap_or(0),&r["node"])],
        };
        let start=r["start_time_millis"].as_i64().unwrap_or(0);let stop=r["stop_time_millis"].as_i64().unwrap_or(start);let elapsed=(stop-start).max(0);
        for (id,files,bytes,node) in shards {
            let node_name=st.nodes.get(node.as_str().unwrap_or_default()).map(|n|n.name.as_str()).unwrap_or_default();
            let name=crate::front::SharedText::try_json(&node_name)?;
            // ES restores through `addRecoveredBytesToFile`: nothing counts as
            // recovered *from a snapshot* (that is snapshot-based peer recovery).
            let text=crate::front::SharedText::try_format(format_args!(r#"{{
              "id":{id},"type":"SNAPSHOT","stage":"DONE","primary":true,
              "start_time_in_millis":{start},"stop_time_in_millis":{stop},"total_time_in_millis":{elapsed},
              "source":{{"repository":{},"snapshot":{},"version":{},"index":{},"restoreUUID":{}}},
              "target":{{"id":{},"name":{}}},
              "index":{{"size":{{"total_in_bytes":{bytes},"reused_in_bytes":0,"recovered_in_bytes":{bytes},"recovered_from_snapshot_in_bytes":0,"percent":"100.0%"}},
                "files":{{"total":{files},"reused":0,"recovered":{files},"percent":"100.0%"}},"total_time_in_millis":{elapsed},"source_throttle_time_in_millis":0,"target_throttle_time_in_millis":0}},
              "translog":{{"recovered":0,"total":0,"percent":"100.0%","total_on_start":0,"total_time_in_millis":0}},
              "verify_index":{{"check_index_time_in_millis":0,"total_time_in_millis":0}}
            }}"#,&r["repository"],&r["snapshot"],&r["version"],&r["source_index"],&r["restore_uuid"],node,name.as_str()))?;
            let mut patch=crate::owned_json::OwnedJson::parse(&text)?;
            body.prepare_reply_owner(&patch)?;
            let mut value=Value::Null;patch.transform(|v,_|{value=std::mem::take(v);Ok(())})?;
            body.transform(|body,memory|{
                let shards=body.get_mut(index).and_then(|v|v["shards"].as_array_mut()).expect("checked recovery shape");
                if let Some(entry)=shards.iter_mut().find(|v|v["primary"].as_bool()==Some(true)&&v["id"].as_u64()==Some(id)){*entry=std::mem::take(&mut value);}
                else {
                    if shards.len()==shards.capacity(){let n=shards.len().checked_add(1).ok_or_else(||EsError::new(429,"circuit_breaking_exception","snapshot recovery slots overflow"))?;let bytes=n.checked_mul(std::mem::size_of::<Value>()).ok_or_else(||EsError::new(429,"circuit_breaking_exception","snapshot recovery bytes overflow"))?;memory.grow(bytes as u64,"snapshot recovery replacement slots")?;shards.try_reserve_exact(1).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;}
                    shards.insert(0,std::mem::take(&mut value));
                }Ok(())
            })?;
            body.retain_reply_owner(patch);
        }
    }Ok(())
}

#[cfg(test)]
mod credential_tests {
    use super::*;

    #[tokio::test]
    async fn restore_sub_requests_carry_the_callers_credentials() {
        let creds = crate::front::SharedHeaders::try_new(&[("authorization".to_string(), "Basic dTpw".to_string())]).unwrap();
        let req = CREDENTIALS.scope(creds, async { rest("PUT", format_args!("/restored"), None) }).await.unwrap();
        assert!(req.headers.iter().any(|(k, v)| k == "authorization" && v == "Basic dTpw"));
        assert!(req.headers.iter().any(|(k, v)| k == "content-type" && v == "application/json"));
        // The caller's own credentials, not the system identity.
        assert!(!req.headers.iter().any(|(k, _)| k == crate::front::SYSTEM_HEADER));
        // A resumed restore (no caller) runs as `_system` instead.
        let resumed = rest("PUT", format_args!("/r"), None).unwrap();
        assert!(!resumed.headers.iter().any(|(k, _)| k == "authorization"));
        assert!(resumed.headers.iter().any(|(k, v)| k == crate::front::SYSTEM_HEADER && v == "true"));
        let empty = CREDENTIALS.scope(crate::front::SharedHeaders::try_new(&[]).unwrap(), async { rest("PUT", format_args!("/r"), None) }).await.unwrap();
        assert!(empty.headers.iter().any(|(k, _)| k == crate::front::SYSTEM_HEADER));
    }
}

#[cfg(test)]
mod rerun_tests {
    use super::*;
    use elasticsearch_cluster::state::IndexMetadata;

    fn state(indices: &[(&str, &str)]) -> ClusterState {
        let mut s = ClusterState::empty("test");
        for (name, uuid) in indices {
            s.metadata.indices.insert((*name).into(), IndexMetadata { name: (*name).into(), uuid: (*uuid).into(), version: 1, mapping_version: 1, settings_version: 1,
                aliases_version: 1, state: IndexState::Open, settings: Default::default(), mappings: json!({}).into(), aliases: Default::default(), number_of_shards: 1,
                number_of_replicas: 0, routing_num_shards: 1024, primary_terms: vec![1], in_sync_allocation_ids: Default::default(), creation_date: 0, lifecycle: None });
        }
        s
    }

    /// A new master never deletes a completed (or failed) target, nor an
    /// index of an incomplete target's name that is not the incarnation the
    /// restore created; it deletes the incomplete targets it created.
    #[test]
    fn rerun_deletes_only_incomplete_incarnations_the_restore_created() {
        let e = json!({"targets": {
            "done": {"state": "done", "index_uuid": "d1", "released": true},
            "unreleased": {"state": "done", "index_uuid": "u1", "final_blocks": {"index.blocks.write": null}},
            "failed": {"state": "failed", "index_uuid": "f1"},
            "restoring": {"state": "restoring", "index_uuid": "r1"},
            "replaced": {"state": "restoring", "index_uuid": "old"},
            "creating": {"state": "creating"},
            "pending": {"state": "pending"},
            "gone": {"state": "restoring", "index_uuid": "g1"},
        }});
        let st = state(&[("done", "d1"), ("unreleased", "u1"), ("failed", "f1"), ("restoring", "r1"), ("replaced", "someone-elses"), ("creating", "c-unclaimed"), ("pending", "p-user")]);
        let plan = rerun_plan(&e, &st).unwrap();
        assert_eq!(plan.finished.iter().map(String::as_str).collect::<Vec<_>>(), vec!["done", "failed", "unreleased"]);
        assert_eq!(plan.delete, vec![("restoring".to_string(), "r1".to_string())]);
        assert_eq!(plan.unreleased.len(), 1);
        assert_eq!((plan.unreleased[0].0.as_str(), plan.unreleased[0].1.as_str()), ("unreleased", "u1"));
    }

    /// An entry an older version wrote has no progress: nothing is deleted.
    #[test]
    fn rerun_of_an_entry_without_progress_deletes_nothing() {
        let e = json!({"indices": ["a", "b"]});
        assert!(rerun_plan(&e, &state(&[("a", "x"), ("b", "y")])).is_err());
    }
}

/// CCR (`crate::ccr`): create follower `target` from its leader through
/// this restore flow (ES restores a follower from the `_ccr_<alias>`
/// repository, snapshot `_latest_`). A restore entry (without a stored
/// request: a new master drops it rather than planning it again) arms the
/// target for a file restore, `body` creates it (`PUT /{target}`), each
/// primary stages the leader primary's files (`file_restore::stage_shard`)
/// and starts; returns once every primary started (`Ok`: the restored
/// shard count and the index uuid) or one failed.
pub async fn restore_follower(node: &Arc<ClusterNode>, target: &str, alias: &str, leader_index: &str, leader_uuid: &str, primary_term: u64, body: &Value) -> Result<(u32, String), String> {
    let repository = format!("_ccr_{alias}");
    let uuid = random_base64_uuid();
    let entry = json!({
        "snapshot": "_latest_", "repository": repository, "uuid": uuid, "state": "STARTED",
        "indices": [target], "targets": {target: {"state": TARGET_PENDING}}, "master": node.local.id, "ccr": true,
    });
    // Not `add_progress`: concurrent follows share the repository and snapshot name.
    common::update_custom(node, PROGRESS_CUSTOM, move |v| {
        if !v.is_object() {
            *v = json!({});
        }
        if !v["restore"].is_array() {
            v["restore"] = json!([]);
        }
        v["restore"].as_array_mut().expect("array").push(entry);
        Ok(true)
    })
    .await?;
    state_of(node).restoring.lock().insert(uuid.clone());
    let result = async {
        let spec = FileRestoreSpec {
            repository: repository.clone(),
            snapshot: "_latest_".into(),
            snapshot_uuid: "_latest_".into(),
            version: None,
            source_index: leader_index.to_string(),
            index_id: leader_uuid.to_string(),
            primary_term,
            empty_shards: Vec::new(),
            replaces_closed: false,
            armed: true,
            index_uuid: None,
        };
        arm(node, &uuid, target, Some(&spec)).await?;
        set_target(node, &uuid, target, TARGET_CREATING, None).await?;
        let prepared_request = rest("PUT", format_args!("/{target}?wait_for_active_shards=0"), Some(body));
        let r = dispatch(node, prepared_request).await;
        if !r.is_success() {
            let v = r.body_json().unwrap_or(Value::Null);
            return Err(format!("failed to create the follower index [{target}]: {}", v["error"]["reason"].as_str().unwrap_or("unknown failure")));
        }
        let st = node.state();
        let index_uuid = match (st.metadata.indices.get(target).map(|m| m.uuid.clone()), claimed_uuid(&st, &uuid, target)) {
            (Some(c), Some(claimed)) if c == claimed => c,
            _ => return Err(format!("index [{target}] was not created by this restore")),
        };
        if file_restore::file_restore_of(&st, target, &index_uuid).is_none() {
            return Err(format!("index [{target}] was created without restore routing"));
        }
        let shards: Vec<u32> = (0..st.metadata.indices[target].number_of_shards.max(1)).collect();
        let outcomes = wait_for_restored_primaries(node, target, &index_uuid, &shards).await?;
        let failures: Vec<String> = outcomes.iter().filter_map(|(k, o)| o.as_ref().err().map(|e| format!("[{target}][{k}] {e}"))).collect();
        if !failures.is_empty() {
            return Err(format!("failed to restore the follower from the leader: {}", failures.join("; ")));
        }
        let _ = set_target(node, &uuid, target, TARGET_DONE, Some((&index_uuid, Map::new()))).await;
        Ok((shards.len() as u32, index_uuid))
    }
    .await;
    if let Err(e) = common::remove_progress(node, "restore", uuid.clone()).await {
        tracing::warn!("failed to remove the follower's restore entry: {e}");
    }
    state_of(node).restoring.lock().remove(&uuid);
    result
}
