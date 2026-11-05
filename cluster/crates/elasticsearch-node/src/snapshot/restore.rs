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
//! restored with one master task; data streams whose backing indices were
//! restored are re-created (renamed like their indices).

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
    let plan = match svc.plan_restore(repo, snap, &rr, &existing).await {
        Ok(p) => p,
        Err(e) => return err(&e),
    };
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
    let node2 = node.clone();
    let plan2 = plan.clone();
    let handle = tokio::spawn(CREDENTIALS.scope(creds, async move { run_restore(node2, svc, plan2, rr, uuid).await }));
    if !wait {
        return RestResp::json(200, &json!({"accepted": true}));
    }
    match handle.await {
        Ok(Ok(ok)) => RestResp::json(200, &plan.response_json(ok)),
        Ok(Err(e)) => err(&e),
        Err(e) => RestResp::error(500, "illegal_state_exception", &e.to_string()),
    }
}

async fn run_restore(node: Arc<ClusterNode>, svc: Arc<SnapshotsService>, plan: RestorePlan, rr: RestoreRequest, uuid: String) -> Result<u32, SnapshotError> {
    let result = async {
        let mut ok = 0u32;
        let mut first_err: Option<SnapshotError> = None;
        for ip in &plan.indices {
            match restore_index(&node, &svc, &plan, ip, &rr, &uuid).await {
                Ok(n) => ok += n,
                Err(e) => {
                    tracing::warn!(index = %ip.target_index, "restore failed: {e}");
                    first_err.get_or_insert(e);
                }
            }
        }
        if let Some(g) = &plan.global_metadata {
            restore_global(&node, g).await?;
        }
        if !plan.snapshot.data_streams.is_empty() {
            restore_data_streams(&node, &svc, &plan, &rr).await;
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

/// Master only: run a restore again whose coordinating master is gone. The
/// target indices it had created are deleted first (they hold an unknown
/// part of the documents), then the stored request is planned and executed
/// under the same restore uuid. An entry without a stored request is dropped.
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

async fn rerun(node: &Arc<ClusterNode>, e: &Value, uuid: &str) -> Result<u32, SnapshotError> {
    let repo = e["repository"].as_str().unwrap_or_default();
    let snap = e["snapshot"].as_str().unwrap_or_default();
    let body = e.get("request").ok_or_else(|| SnapshotError::new("snapshot_restore_exception", 500, "the restore entry carries no request".to_string()))?;
    let rr = RestoreRequest::from_json(if body.is_null() { None } else { Some(body) })?;
    let targets: Vec<String> = e["indices"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
    for t in &targets {
        if !node.state().metadata.indices.contains_key(t) {
            continue;
        }
        // Finish admitted URI construction before suspension: fmt::Arguments is not Send.
        let prepared_request = rest("DELETE", format_args!("/{t}"), None);
        let r = dispatch(node, prepared_request).await;
        if r.status >= 300 && r.status != 404 {
            return Err(resp_error(&r, &format!("failed to remove half-restored index [{t}]")));
        }
    }
    let st = node.state();
    let existing: BTreeMap<String, ExistingIndex> = st.metadata.indices.iter().map(|(n, m)| (n.clone(), ExistingIndex { open: m.state == IndexState::Open, number_of_shards: m.number_of_shards.max(1) })).collect();
    let svc = service(node).await;
    let plan = svc.plan_restore(repo, snap, &rr, &existing).await?;
    run_restore(node.clone(), svc, plan, rr, uuid.to_string()).await
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

/// Node setting (not ES): `false` restores every index by re-indexing.
pub const FILE_BASED: &str = "snapshot.restore.file_based";

/// Index settings whose change on restore (`index_settings` /
/// `ignore_index_settings`) changes where documents route, how segments are
/// sorted or what they store: such a restore re-indexes (ES copies the
/// files regardless).
const LAYOUT_SETTINGS: &[&str] = &[
    "index.number_of_routing_shards", "index.routing_partition_size", "index.routing_path", "index.mode", "index.mapping.source.mode",
    "index.sort.field", "index.sort.order", "index.sort.mode", "index.sort.missing",
];

/// Restore one index; returns the number of shards restored.
async fn restore_index(node: &Arc<ClusterNode>, svc: &Arc<SnapshotsService>, plan: &RestorePlan, ip: &RestoreIndexPlan, rr: &RestoreRequest, restore_uuid: &str) -> Result<u32, SnapshotError> {
    let target = ip.target_index.clone();
    let started = now_millis();
    if ip.replaces_closed_index {
        let prepared_request = rest("DELETE", format_args!("/{target}"), None);
        let r = dispatch(node, prepared_request).await;
        if !r.is_success() {
            return Err(resp_error(&r, &format!("failed to replace closed index [{target}]")));
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
    let mut body = json!({"settings": settings, "mappings": unwrap_mappings(&ip.metadata["mappings"])});
    if let Some(a) = ip.metadata.get("aliases").filter(|a| a.as_object().map(|o| !o.is_empty()).unwrap_or(false)) {
        body["aliases"] = a.clone();
    }
    // A `source` repository's shards are stored-fields-only segments.
    let source_only = svc.repository(&plan.repository).map(|r| r.settings().source_only).unwrap_or(false);
    match file_restore_blocker(node, ip, rr, source_only) {
        None => match restore_index_files(node, plan, ip, restore_uuid, &body).await? {
            FilesOutcome::Restored(n) => {
                apply_blocks(node, &target, blocks).await?;
                return Ok(n);
            }
            FilesOutcome::Reindex(reason) => {
                tracing::warn!(index = %target, snapshot = %plan.snapshot.snapshot, "restoring by re-indexing the snapshot's documents: {reason}");
                let prepared_request = rest("DELETE", format_args!("/{target}"), None);
                let r = dispatch(node, prepared_request).await;
                if !r.is_success() && r.status != 404 {
                    return Err(resp_error(&r, &format!("failed to remove [{target}] before re-indexing it")));
                }
            }
        },
        Some(reason) => tracing::info!(index = %target, snapshot = %plan.snapshot.snapshot, "restoring by re-indexing the snapshot's documents: {reason}"),
    }
    restore_index_docs(node, svc, plan, ip, restore_uuid, &body, blocks, started, source_only).await
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
    /// Shards whose primaries started from the snapshot's files.
    Restored(u32),
    /// The files cannot be adopted (found by a primary's node): re-index.
    Reindex(String),
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
async fn restore_index_files(node: &Arc<ClusterNode>, plan: &RestorePlan, ip: &RestoreIndexPlan, restore_uuid: &str, body: &Value) -> Result<FilesOutcome, SnapshotError> {
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
    // Primaries start only once their files are in: no wait here.
    let prepared_request = rest("PUT", format_args!("/{target}?wait_for_active_shards=0"), Some(body));
    let r = dispatch(node, prepared_request).await;
    if !r.is_success() {
        let _ = arm(node, restore_uuid, target, None).await;
        return Err(resp_error(&r, &format!("failed to create index [{target}]")));
    }
    let st = node.state();
    let Some(uuid) = st.metadata.indices.get(target).map(|m| m.uuid.clone()) else {
        return Err(SnapshotError::snapshot_restore(&snap_ref, format!("index [{target}] was removed while it was being restored")));
    };
    if file_restore::file_restore_of(&st, target, &uuid).is_none() {
        // Created by another path than the armed one: its primaries are
        // new empty shards.
        let _ = arm(node, restore_uuid, target, None).await;
        return Ok(FilesOutcome::Reindex("the index was created without restore routing".into()));
    }
    let outcomes = wait_for_restored_primaries(node, target, &uuid, &ip.shards).await.map_err(|e| SnapshotError::snapshot_restore(&snap_ref, e))?;
    let mut restored = 0u32;
    let mut failures = Vec::new();
    for (shard, outcome) in outcomes {
        match outcome {
            Ok(()) => restored += 1,
            Err(details) if file_restore::requires_reindex(&details) => return Ok(FilesOutcome::Reindex(format!("shard [{shard}]: {details}"))),
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
    Ok(FilesOutcome::Restored(restored))
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
async fn restore_index_docs(node: &Arc<ClusterNode>, svc: &Arc<SnapshotsService>, plan: &RestorePlan, ip: &RestoreIndexPlan, restore_uuid: &str, body: &Value, blocks: Map<String, Value>, started: i64, source_only: bool) -> Result<u32, SnapshotError> {
    let target = ip.target_index.clone();
    let prepared_request = rest("PUT", format_args!("/{target}?wait_for_active_shards=1"), Some(body));
    let r = dispatch(node, prepared_request).await;
    if !r.is_success() {
        return Err(resp_error(&r, &format!("failed to create index [{target}]")));
    }
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
        let loaded = load_docs(node, &target, dir.clone()).await;
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
    apply_blocks(node, &target, blocks).await?;
    let st = node.state();
    let index_uuid = st.metadata.indices.get(&target).map(|m| m.uuid.clone()).unwrap_or_default();
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
async fn load_docs(node: &Arc<ClusterNode>, target: &str, dir: PathBuf) -> Result<u64, String> {
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
        total += load_shard_docs(node, target, svc.clone(), sid).await?;
    }
    Ok(total)
}

async fn load_shard_docs(node: &Arc<ClusterNode>, target: &str, svc: Arc<IndexService>, sid: u32) -> Result<u64, String> {
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
        let req=rest("POST",format_args!("/_bulk?pipeline=_none"),None)
            .and_then(|mut req|{req.replace_header("content-type","application/x-ndjson")?;req.try_with_encoded_body(|out| {
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
        for (key, dst) in [("index_templates", &mut next.metadata.index_templates), ("component_templates", &mut next.metadata.component_templates), ("legacy_templates", &mut next.metadata.legacy_templates)] {
            if let Some(m) = g[key].as_object() {
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

/// Re-create data streams whose backing indices were restored.
async fn restore_data_streams(node: &Arc<ClusterNode>, svc: &SnapshotsService, plan: &RestorePlan, rr: &RestoreRequest) {
    let Ok(g) = svc.read_global_metadata(&plan.repository, &plan.snapshot.snapshot).await else { return };
    let Some(streams) = g["customs"][crate::mirror::DATA_STREAMS_CUSTOM].as_object().cloned() else { return };
    let renamed: BTreeMap<String, String> = plan.indices.iter().map(|i| (i.source_index.clone(), i.target_index.clone())).collect();
    let mut add = Map::new();
    for (name, ds) in streams {
        let backing: Vec<String> = ds["indices"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
        let kept: Vec<String> = backing.iter().filter_map(|b| renamed.get(b).cloned()).collect();
        if kept.is_empty() {
            continue;
        }
        let new_name = match (&rr.rename_pattern, &rr.rename_replacement) {
            (Some(p), Some(r)) => elasticsearch_snapshots::restore::rename_index(&name, p, r).unwrap_or(name.clone()),
            _ => name.clone(),
        };
        let mut d = ds.clone();
        d["name"] = json!(new_name);
        d["indices"] = json!(kept);
        add.insert(new_name, d);
    }
    if add.is_empty() {
        return;
    }
    if let Err(e) = common::update_custom(node, crate::mirror::DATA_STREAMS_CUSTOM, move |v| {
        if !v.is_object() {
            *v = json!({});
        }
        for (k, d) in add {
            v[&k] = d;
        }
        Ok(true)
    })
    .await
    {
        tracing::warn!("failed to restore data streams: {e}");
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
