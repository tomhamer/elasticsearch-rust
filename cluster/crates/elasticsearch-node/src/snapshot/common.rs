//! Shared snapshot state of a node: the master's [`SnapshotsService`]
//! (rebuilt per master term and kept in step with the `repositories`
//! metadata custom), the snapshots still gathering shard files, restores in
//! progress, and helpers to edit the snapshot-related cluster-state customs.
//!
//! Every repository write goes through the elected master's single
//! `SnapshotsService`, which is what the snapshots crate assumes (its shard
//! generations, locks and in-progress registry are per process). A new
//! master term gets a fresh service, so nothing learnt under an older term
//! (e.g. unpublished shard generations) can leak into a later snapshot.

use crate::front::RestResp;
use crate::node::ClusterNode;
use elasticsearch_cluster::state::ClusterState;
use elasticsearch_snapshots::util::CancelToken;
use elasticsearch_snapshots::{RepositoryContext, RepositoryMetadata, S3ClientSettings, SnapshotError, SnapshotsService};
use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

/// `metadata.repositories` (ES `RepositoriesMetadata`).
pub const REPOS_CUSTOM: &str = "repositories";
/// `metadata.snapshot_lifecycle` (ES `SnapshotLifecycleMetadata`).
pub const SLM_CUSTOM: &str = "snapshot_lifecycle";
/// Snapshots / restores in progress. ES keeps these as cluster-state
/// customs (`snapshots`, `restore`); the cluster state model here only has
/// metadata customs, so they live under this internal key and are rendered
/// at the top level of `GET _cluster/state` by [`render_state_customs`].
pub const PROGRESS_CUSTOM: &str = "_snapshots_in_progress";

pub fn now_millis() -> i64 {
    elasticsearch_snapshots::util::now_millis()
}

pub fn err(e: &SnapshotError) -> RestResp {
    RestResp::json(e.status, &e.to_json())
}

/// A snapshot that is still collecting shard files on the master (before
/// the snapshots crate takes over and lists it as running).
pub struct Pending {
    pub repository: String,
    pub name: String,
    pub uuid: String,
    pub include_global_state: bool,
    pub partial: bool,
    pub indices: Vec<String>,
    pub data_streams: Vec<String>,
    pub metadata: Option<Map<String, Value>>,
    pub start_time: i64,
    /// Index -> id of the node that held its primary when the snapshot
    /// started (recorded in the cluster state). Empty for entries written
    /// before this was tracked.
    pub assigned: std::collections::BTreeMap<String, String>,
    pub cancel: CancelToken,
    /// Handed to the snapshots service (it reports the snapshot from then on).
    pub in_service: AtomicBool,
    pub done: tokio::sync::watch::Sender<bool>,
}

impl Pending {
    pub fn in_service(&self) -> bool {
        self.in_service.load(Ordering::SeqCst)
    }

    pub async fn wait_done(&self) {
        let mut rx = self.done.subscribe();
        while !*rx.borrow() {
            if rx.changed().await.is_err() {
                break;
            }
        }
    }
}

/// Per-node snapshot runtime.
#[derive(Default)]
pub struct SnapState {
    svc: Mutex<Option<(u64, Arc<SnapshotsService>)>>,
    /// Repositories currently registered in `svc`.
    registered: tokio::sync::Mutex<BTreeMap<String, RepositoryMetadata>>,
    pub pending: Mutex<Vec<Arc<Pending>>>,
    /// Snapshots this node finished (their in-progress entry is going away).
    pub finished: Mutex<BTreeSet<String>>,
    /// Restores this node coordinates, by restore uuid.
    pub restoring: Mutex<BTreeSet<String>>,
    /// Running SLM snapshots by policy id: (snapshot name, uuid, start).
    pub slm_running: Mutex<BTreeMap<String, (String, String, i64)>>,
    /// Next SLM fire time per (policy id, version).
    pub slm_next: Mutex<HashMap<String, (u64, i64)>>,
    pub slm_retention_next: Mutex<Option<(String, i64)>>,
    pub background: AtomicBool,
}

static STATES: OnceLock<Mutex<HashMap<String, Arc<SnapState>>>> = OnceLock::new();

pub fn state_of(node: &ClusterNode) -> Arc<SnapState> {
    STATES.get_or_init(Default::default).lock().entry(node.local.id.clone()).or_default().clone()
}

/// The node's repository context: `path.repo`, URL allow-list, S3 clients.
pub fn repo_context(node: &ClusterNode) -> RepositoryContext {
    let mut flat: BTreeMap<String, String> = node.config.settings.iter().map(|(k, v)| (k.clone(), v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()))).collect();
    // Secure settings as of the last `_nodes/reload_secure_settings`.
    crate::keystore::overlay_secure(&node.config.settings, &mut flat);
    RepositoryContext {
        path_repo: node.config.repo_paths.clone(),
        allowed_urls: crate::config::get_list(&node.config.settings, "repositories.url.allowed_urls"),
        s3_clients: S3ClientSettings::from_node_settings(&flat),
    }
}

/// Repositories registered in the cluster state.
pub fn repos_in_state(st: &ClusterState) -> BTreeMap<String, RepositoryMetadata> {
    let mut out = BTreeMap::new();
    if let Some(m) = st.metadata.customs.get(REPOS_CUSTOM).and_then(Value::as_object) {
        for (name, r) in m {
            out.insert(name.clone(), RepositoryMetadata::new(name.clone(), r["type"].as_str().unwrap_or_default(), r.get("settings").cloned().unwrap_or(json!({}))));
        }
    }
    out
}

/// The master's snapshots service, in step with the cluster state's
/// repositories.
pub async fn service(node: &Arc<ClusterNode>) -> Arc<SnapshotsService> {
    let s = state_of(node);
    let term = node.mode.lock().1;
    let mut reg = s.registered.lock().await;
    let svc = {
        let mut g = s.svc.lock();
        match &*g {
            Some((t, svc)) if *t == term => svc.clone(),
            _ => {
                let svc = SnapshotsService::new(repo_context(node));
                *g = Some((term, svc.clone()));
                reg.clear();
                svc
            }
        }
    };
    let wanted = repos_in_state(&node.state());
    for (name, meta) in &wanted {
        if reg.get(name) == Some(meta) || !svc.running_snapshots(Some(name)).is_empty() {
            continue;
        }
        match svc.put_repository(meta.clone(), false).await {
            Ok(()) => {
                reg.insert(name.clone(), meta.clone());
            }
            Err(e) => tracing::warn!(repository = %name, "cannot open repository on this node: {e}"),
        }
    }
    let stale: Vec<String> = reg.keys().filter(|n| !wanted.contains_key(*n)).cloned().collect();
    for n in stale {
        if svc.delete_repositories(std::slice::from_ref(&n)).is_ok() {
            reg.remove(&n);
        }
    }
    svc
}

/// After `_nodes/reload_secure_settings`: the master's service builds its
/// repositories again from the reloaded secure settings (S3 credentials),
/// as ES's `S3Service.refreshAndClearCache` does. Every other repository use
/// opens its repository from [`repo_context`] and so sees them already.
pub async fn reload_repository_clients(node: &ClusterNode) {
    let s = state_of(node);
    let mut reg = s.registered.lock().await;
    let svc = s.svc.lock().as_ref().map(|(_, svc)| svc.clone());
    if let Some(svc) = svc {
        svc.set_context(repo_context(node));
        reg.clear();
    }
}

/// Forget the cached service (e.g. after a repository was re-registered with
/// different settings while snapshots were running).
pub async fn forget_repository(node: &ClusterNode, name: &str) {
    let s = state_of(node);
    s.registered.lock().await.remove(name);
}

/// Submit a master task that edits one metadata custom. `f` returns whether
/// it changed anything. An empty object removes the custom.
pub async fn update_custom<F>(node: &ClusterNode, key: &'static str, f: F) -> Result<(), String>
where
    F: FnOnce(&mut Value) -> Result<bool, String> + Send + 'static,
{
    let task: elasticsearch_cluster::coordination::StateTask = Box::new(move |s: &ClusterState| {
        let mut next = s.clone();
        let mut v = next.metadata.customs.get(key).cloned().unwrap_or_else(|| json!({}));
        if !f(&mut v)? {
            return Err("no change".into());
        }
        if v.as_object().map(|o| o.is_empty()).unwrap_or(false) {
            next.metadata.customs.remove(key);
        } else {
            next.metadata.customs.insert(key.to_string(), v);
        }
        next.metadata.version += 1;
        Ok(next)
    });
    match node.coord().submit(task).await {
        Ok(_) => Ok(()),
        Err(e) if e == "no change" => Ok(()),
        Err(e) => Err(e),
    }
}

fn arr<'a>(v: &'a mut Value, key: &str) -> &'a mut Vec<Value> {
    if !v.is_object() {
        *v = json!({});
    }
    if !v[key].is_array() {
        v[key] = json!([]);
    }
    v[key].as_array_mut().expect("array")
}

/// Record an in-progress snapshot in the cluster state.
pub async fn add_progress(node: &ClusterNode, kind: &'static str, entry: Value) -> Result<(), String> {
    update_custom(node, PROGRESS_CUSTOM, move |v| {
        let a = arr(v, kind);
        a.retain(|e| !(e["repository"] == entry["repository"] && e["snapshot"] == entry["snapshot"]));
        a.push(entry);
        Ok(true)
    })
    .await
}

pub async fn remove_progress(node: &ClusterNode, kind: &'static str, uuid: String) -> Result<(), String> {
    update_custom(node, PROGRESS_CUSTOM, move |v| {
        let a = arr(v, kind);
        let before = a.len();
        a.retain(|e| e["uuid"].as_str() != Some(uuid.as_str()));
        let changed = a.len() != before;
        if a.is_empty() {
            if let Some(o) = v.as_object_mut() {
                o.remove(kind);
            }
        }
        Ok(changed)
    })
    .await
}

pub fn progress_entries(st: &ClusterState, kind: &str) -> Vec<Value> {
    st.metadata.customs.get(PROGRESS_CUSTOM).and_then(|v| v[kind].as_array().cloned()).unwrap_or_default()
}

/// Repository `uuid` / `generation` as last published.
pub async fn set_repo_generation(node: &ClusterNode, repo: &str, uuid: String, generation: i64) {
    let repo = repo.to_string();
    let r = update_custom(node, REPOS_CUSTOM, move |v| {
        let Some(e) = v.get_mut(&repo).filter(|e| e.is_object()) else { return Ok(false) };
        let new_uuid = if uuid == "_na_" { e["uuid"].clone() } else { json!(uuid) };
        if e["generation"] == json!(generation) && e["pending_generation"] == json!(generation) && e["uuid"] == new_uuid {
            return Ok(false);
        }
        e["uuid"] = new_uuid;
        e["generation"] = json!(generation);
        e["pending_generation"] = json!(generation);
        Ok(true)
    })
    .await;
    if let Err(e) = r {
        tracing::warn!("failed to publish repository generation: {e}");
    }
}

/// After any repository write: give the repository a uuid if it has none
/// yet (ES assigns it with the first `index-N`), then publish uuid and
/// generation in the cluster state.
pub async fn after_repo_write(node: &Arc<ClusterNode>, svc: &SnapshotsService, repo: &str) {
    let Ok(r) = svc.repository(repo) else { return };
    let Ok(mut data) = r.repository_data().await else { return };
    if data.generation >= 0 && (data.uuid.is_empty() || data.uuid == "_na_") && !r.is_readonly() {
        data.uuid = elasticsearch_snapshots::util::random_base64_uuid();
        let expected = data.generation;
        match r.write_repository_data(&data, expected).await {
            Ok(g) => data.generation = g,
            Err(e) => tracing::warn!(repository = %repo, "failed to assign repository uuid: {e}"),
        }
    }
    set_repo_generation(node, repo, data.uuid.clone(), data.generation).await;
}

/// Staging directory for shard files of snapshots and restores.
pub fn staging_root(node: &ClusterNode) -> PathBuf {
    node.config.data_path.join("snapshot-staging")
}

/// `GET _cluster/state` top-level customs: `snapshots`, `restore`,
/// `snapshot_deletions` (ES `SnapshotsInProgress`, `RestoreInProgress`,
/// `SnapshotDeletionsInProgress`).
pub fn render_state_customs(st: &ClusterState) -> Map<String, Value> {
    let fmt = |ms: i64| elasticsearch_snapshots::util::format_millis(ms);
    let uuid_of = |name: &str| st.metadata.indices.get(name).map(|m| m.uuid.clone()).unwrap_or_else(|| "_na_".into());
    let snaps: Vec<Value> = progress_entries(st, "snapshots")
        .iter()
        .map(|e| {
            let indices: Vec<String> = e["indices"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
            let start = e["start_time_millis"].as_i64().unwrap_or(0);
            json!({
                "repository": e["repository"],
                "snapshot": e["snapshot"],
                "uuid": e["uuid"],
                "include_global_state": e["include_global_state"],
                "partial": e["partial"],
                "state": e["state"].as_str().unwrap_or("STARTED"),
                "indices": indices.iter().map(|i| json!({"name": i, "id": uuid_of(i)})).collect::<Vec<_>>(),
                "start_time": fmt(start),
                "start_time_millis": start,
                "repository_state_id": e["repository_state_id"].as_i64().unwrap_or(-1),
                "shards": indices.iter().map(|i| {
                    let node = st.routing_table.shard_table(i, 0).and_then(|t| t.active_primary()).and_then(|p| p.node_id.clone());
                    json!({"index": {"index_name": i, "index_uuid": uuid_of(i)}, "shard": 0, "state": if node.is_some() { "INIT" } else { "MISSING" }, "node": node})
                }).collect::<Vec<_>>(),
                "feature_states": [],
                "data_streams": e["data_streams"].clone(),
            })
        })
        .collect();
    let restores: Vec<Value> = progress_entries(st, "restore")
        .iter()
        .map(|e| {
            json!({
                "snapshot": e["snapshot"],
                "repository": e["repository"],
                "uuid": e["uuid"],
                "state": e["state"].as_str().unwrap_or("STARTED"),
                "indices": e["indices"],
                "shards": e["indices"].as_array().map(|a| a.iter().map(|i| json!({"index": i, "shard": 0, "state": "INIT"})).collect::<Vec<_>>()).unwrap_or_default(),
            })
        })
        .collect();
    let mut m = Map::new();
    m.insert("snapshots".into(), json!({"snapshots": snaps}));
    m.insert("restore".into(), json!({"snapshots": restores}));
    m.insert("snapshot_deletions".into(), json!({"snapshot_deletions": []}));
    m
}

/// Validate a relative path inside a staging directory.
pub fn safe_rel(p: &str) -> bool {
    let path = std::path::Path::new(p);
    !p.is_empty() && !path.is_absolute() && path.components().all(|c| matches!(c, std::path::Component::Normal(_)))
}

/// File-system-safe directory name for an index.
pub fn dir_name(index: &str) -> String {
    index.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' }).collect()
}

/// Every file under `dir` as (relative path, length), sorted.
pub fn list_files(dir: &std::path::Path) -> std::io::Result<Vec<(String, u64)>> {
    fn walk(base: &std::path::Path, d: &std::path::Path, out: &mut Vec<(String, u64)>) -> std::io::Result<()> {
        for e in std::fs::read_dir(d)? {
            let e = e?;
            let p = e.path();
            let ft = e.file_type()?;
            if ft.is_dir() {
                walk(base, &p, out)?;
            } else if ft.is_file() {
                let rel = p.strip_prefix(base).map_err(std::io::Error::other)?.to_string_lossy().replace('\\', "/");
                out.push((rel, e.metadata()?.len()));
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out)?;
    out.sort();
    Ok(out)
}
