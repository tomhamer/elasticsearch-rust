//! REST routing of `_snapshot/*`, `_slm/*`, `_cat/snapshots`,
//! `_cat/repositories`, plus repository registration / verification.
//!
//! Repository reads (`GET _snapshot[/{repo}]`, `_cat/repositories`) are
//! served from the local cluster state; everything that touches repository
//! contents runs on the elected master (other nodes forward).

use super::common::{self, err, repo_context, service, state_of, REPOS_CUSTOM};
use super::{ops, restore, shard, slm};
use crate::front::{forward, local_engine, RestReq, RestResp};
use crate::node::ClusterNode;
use crate::rest::cat::{self, cols, ColDef, Row};
use elasticsearch_snapshots::util::{format_duration_millis, random_base64_uuid, resolve_names, stringify_settings};
use elasticsearch_snapshots::{GetSnapshotsRequest, Repository, RepositoryMetadata, SnapshotError};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

async fn on_master(node: &Arc<ClusterNode>, req: &RestReq) -> Option<RestResp> {
    if node.is_master() {
        return None;
    }
    Some(match node.state().master_node_id.clone() {
        Some(m) if m != node.local.id => forward(node, &m, req).await,
        _ => RestResp::error(503, "master_not_discovered_exception", ""),
    })
}

fn es_error(e: &elasticsearch_engine::error::EsError) -> RestResp {
    RestResp::json(e.status, &e.to_json())
}

/// The parsed `_mount` request (`RestMountSearchableSnapshotAction`).
#[derive(Debug, PartialEq)]
struct MountRequest {
    index: String,
    mounted: String,
    partial: bool,
    wait: bool,
    index_settings: Map<String, Value>,
    ignore: Vec<String>,
}

fn parse_mount(params: &std::collections::HashMap<String, String>, body: Option<&Value>) -> Result<MountRequest, elasticsearch_engine::error::EsError> {
    use elasticsearch_engine::error::EsError;
    let storage = params.get("storage").map(String::as_str).unwrap_or("full_copy");
    let partial = match storage.to_uppercase().as_str() {
        "FULL_COPY" => false,
        "SHARED_CACHE" => true,
        other => return Err(EsError::illegal_argument(format!("No enum constant org.elasticsearch.xpack.core.searchablesnapshots.MountSearchableSnapshotRequest.Storage.{other}"))),
    };
    let wait = match params.get("wait_for_completion").map(String::as_str) {
        None => false,
        Some("" | "true") => true,
        Some("false") => false,
        Some(v) => return Err(EsError::illegal_argument(format!("Failed to parse value [{v}] as only [true] or [false] are allowed."))),
    };
    let obj = body.and_then(Value::as_object).cloned().unwrap_or_default();
    let parse_err = |msg: String| EsError::x_content_parse(format!("[1:2] {msg}"));
    for k in obj.keys() {
        if !matches!(k.as_str(), "index" | "renamed_index" | "index_settings" | "ignore_index_settings" | "ignored_index_settings") {
            return Err(parse_err(format!("[mount_searchable_snapshot] unknown field [{k}]")));
        }
    }
    let index = obj.get("index").and_then(Value::as_str).ok_or_else(|| EsError::illegal_argument("Required [index]"))?.to_string();
    let mounted = obj.get("renamed_index").and_then(Value::as_str).unwrap_or(&index).to_string();
    let index_settings = match obj.get("index_settings") {
        Some(Value::Object(o)) => o.clone(),
        Some(Value::Null) | None => Map::new(),
        Some(_) => return Err(parse_err("[mount_searchable_snapshot] failed to parse field [index_settings]".into())),
    };
    let ignore = match obj.get("ignore_index_settings") {
        Some(Value::Array(a)) => a.iter().map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())).collect(),
        Some(Value::String(s)) => vec![s.clone()],
        _ => Vec::new(),
    };
    Ok(MountRequest { index, mounted, partial, wait, index_settings, ignore })
}

/// The restore body mounting `m` (`TransportMountSearchableSnapshotAction`):
/// the index under its mounted name, read-only, with the searchable
/// snapshot store settings.
fn mount_restore_body(m: &MountRequest, repo: &str, snap: &str, snap_uuid: &str, source_settings: &Map<String, Value>) -> Value {
    let mut ignore = m.ignore.clone();
    ignore.push("index.data_path".into());
    for k in source_settings.keys() {
        if ["index.routing.allocation.require.", "index.routing.allocation.include.", "index.routing.allocation.exclude."].iter().any(|p| k.starts_with(p)) {
            ignore.push(k.clone());
        }
    }
    let mut settings = Map::new();
    settings.insert("index.number_of_replicas".into(), json!("0"));
    settings.insert("index.auto_expand_replicas".into(), json!("false"));
    settings.insert("index.shard.check_on_startup".into(), json!("false"));
    settings.insert("index.routing.allocation.include._tier_preference".into(), json!(if m.partial { "data_frozen" } else { "data_cold,data_warm,data_hot" }));
    for (k, v) in elasticsearch_engine::settings::flatten(&Value::Object(m.index_settings.clone())) {
        let k = if k.starts_with("index.") { k } else { format!("index.{k}") };
        settings.insert(k, v);
    }
    settings.insert("index.store.snapshot.repository_name".into(), json!(repo));
    settings.insert("index.store.snapshot.snapshot_name".into(), json!(snap));
    settings.insert("index.store.snapshot.snapshot_uuid".into(), json!(snap_uuid));
    settings.insert("index.store.snapshot.index_name".into(), json!(m.index));
    settings.insert("index.store.snapshot.index_uuid".into(), source_settings.get("index.uuid").cloned().unwrap_or_else(|| json!(m.index)));
    settings.insert("index.store.type".into(), json!("snapshot"));
    settings.insert("index.blocks.write".into(), json!("true"));
    settings.insert("index.allocation.existing_shards_allocator".into(), json!("searchable_snapshot_allocator"));
    settings.insert("index.recovery.type".into(), json!("snapshot_prewarm"));
    if m.partial {
        settings.insert("index.store.snapshot.partial".into(), json!("true"));
        settings.insert("index.routing.allocation.disk.watermark.ignore".into(), json!("true"));
        settings.insert("index.shard_limit.group".into(), json!("frozen"));
    }
    json!({
        "indices": [m.index],
        "rename_pattern": ".+",
        "rename_replacement": m.mounted.replace('$', "$$"),
        "index_settings": settings,
        "ignore_index_settings": ignore,
        "include_global_state": false,
        "include_aliases": false,
    })
}

/// `POST _snapshot/{repo}/{snapshot}/_mount`: a restore of one index with
/// the searchable snapshot settings.
async fn mount(node: &Arc<ClusterNode>, req: &RestReq, repo: &str, snap: &str) -> RestResp {
    use elasticsearch_snapshots::{ExistingIndex, RestoreRequest};
    let body = req.json();
    if !req.body.is_empty() && body.is_none() {
        return RestResp::error(400, "parse_exception", "request body is required to be valid JSON");
    }
    let m = match parse_mount(&req.params(), body.as_ref()) {
        Ok(m) => m,
        Err(e) => return es_error(&e),
    };
    // The source index as snapshotted (settings, snapshot uuid).
    let svc = service(node).await;
    let probe = match RestoreRequest::from_json(Some(&json!({"indices": [m.index], "include_global_state": false, "include_aliases": false}))) {
        Ok(r) => r,
        Err(e) => return err(&e),
    };
    let none: BTreeMap<String, ExistingIndex> = BTreeMap::new();
    let plan = match svc.plan_restore(repo, snap, &probe, &none).await {
        Ok(p) => p,
        Err(e) if e.kind == "index_not_found_exception" => {
            return es_error(&elasticsearch_engine::error::EsError::index_not_found(&format!("index [{}] not found in repository [{repo}]", m.index)));
        }
        Err(e) => return err(&e),
    };
    let source_settings = plan.indices.first().and_then(|i| i.metadata["settings"].as_object().cloned()).unwrap_or_default();
    if source_settings.get("index.store.type").and_then(Value::as_str) == Some("snapshot") {
        let g = |k: &str| source_settings.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        return es_error(&elasticsearch_engine::error::EsError::illegal_argument(format!(
            "index [{}] in snapshot [{repo}/{}:{snap}] is a snapshot of a searchable snapshot index backed by index [{}] in snapshot [{}/{}:{}] and cannot be mounted; did you mean to restore it instead?",
            m.index, g("index.store.snapshot.repository_uuid"), g("index.store.snapshot.index_name"), g("index.store.snapshot.repository_name"), g("index.store.snapshot.repository_uuid"), g("index.store.snapshot.snapshot_name")
        )));
    }
    let restore_body = mount_restore_body(&m, repo, snap, &plan.snapshot.uuid, &source_settings);
    let restore_req = (|| -> elasticsearch_engine::error::Result<RestReq> {
        let mut r=req.clone(); r.set_method("POST")?;
        r.set_uri_fmt(format_args!("/_snapshot/{repo}/{snap}/_restore?wait_for_completion={}",m.wait))?;
        r.set_headers(&crate::reindex_multi::credentials(req)?)?; r.push_header("content-type","application/json")?;
        r.set_json(&restore_body)?; Ok(r)
    })();
    let restore_req=match restore_req {Ok(r)=>r,Err(e)=>return es_error(&e)};
    let r = restore::restore(node, &restore_req, repo, snap).await;
    if !m.wait && r.is_success() {
        return RestResp::json(202, &json!({"accepted": true}));
    }
    r
}

/// `_analyze` / `_verify_integrity` run on the master's engine, which
/// knows only the repositories registered with it: register the cluster's
/// repository there first (unverified; analysis verifies it itself).
async fn engine_repository_op(node: &Arc<ClusterNode>, req: &RestReq, repo: &str) -> RestResp {
    if let Some(meta) = common::repos_in_state(&node.state()).get(repo) {
        let body = engine_repository_body(meta);
        let put=(|| -> elasticsearch_engine::error::Result<RestReq> {
            let mut r=req.clone(); r.set_method("PUT")?; r.set_uri_fmt(format_args!("/_snapshot/{repo}?verify=false"))?;
            r.retain_headers(|(k,_)|!k.eq_ignore_ascii_case("content-length")&&!k.eq_ignore_ascii_case("content-type"))?;
            r.push_header("content-type","application/json")?;r.set_json(&body)?;Ok(r)
        })();
        let put=match put {Ok(r)=>r,Err(e)=>return es_error(&e)};
        let r = local_engine(node, &put).await;
        if !r.is_success() {
            return r;
        }
    }
    let r = local_engine(node, req).await;
    // The engine validated the request but reads only its own snapshot
    // format: the cluster's snapshots are verified here.
    let empty = r.status == 400 && r.body_json().is_some_and(|v| v["error"]["reason"] == "repository is empty, cannot verify its integrity");
    if empty && req.segments().last().map(String::as_str) == Some("_verify_integrity") {
        return verify_integrity(node, repo).await;
    }
    r
}

/// `POST _snapshot/{repo}/_verify_integrity` over the cluster's snapshots:
/// every snapshot's info and every snapshotted index's metadata must read.
async fn verify_integrity(node: &Arc<ClusterNode>, repo: &str) -> RestResp {
    let svc = service(node).await;
    let r = match svc.repository(repo) {
        Ok(r) => r,
        Err(e) => return err(&e),
    };
    let data = match r.repository_data().await {
        Ok(d) => d,
        Err(e) => return err(&e),
    };
    if data.snapshots.is_empty() {
        return RestResp::error(400, "illegal_argument_exception", "repository is empty, cannot verify its integrity");
    }
    let now = common::now_millis();
    let mut log = Vec::new();
    let mut per_index: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let (mut snaps_ok, mut index_snapshots, mut index_snapshots_ok, mut anomalies) = (0usize, 0usize, 0usize, 0usize);
    for e in &data.snapshots {
        let info = match r.read_snapshot_info(&e.uuid).await {
            Ok(i) => i,
            Err(err) => {
                anomalies += 1;
                log.push(json!({"timestamp_in_millis": now, "anomaly": "failed to load snapshot info", "snapshot": {"snapshot": e.name, "uuid": e.uuid}, "exception": err.reason}));
                continue;
            }
        };
        let rendered = info.render(elasticsearch_snapshots::RenderOptions::default());
        let failed: Vec<&str> = info.failures.iter().map(|f| f.index.as_str()).collect();
        let mut snap_ok = true;
        for index in &info.indices {
            index_snapshots += 1;
            let entry = per_index.entry(index.clone()).or_default();
            entry.0 += 1;
            if failed.contains(&index.as_str()) {
                continue;
            }
            match svc.read_index_metadata(repo, &info.snapshot, index).await {
                Ok(_) => {
                    entry.1 += 1;
                    index_snapshots_ok += 1;
                }
                Err(err) => {
                    snap_ok = false;
                    anomalies += 1;
                    log.push(json!({"timestamp_in_millis": now, "anomaly": "failed to load index metadata", "snapshot": rendered.clone(), "index": {"name": index}, "exception": err.reason}));
                }
            }
        }
        if snap_ok {
            snaps_ok += 1;
        }
        log.push(json!({"timestamp_in_millis": now, "snapshot": rendered}));
    }
    for (name, (n, ok)) in &per_index {
        log.push(json!({"timestamp_in_millis": now, "index": {"name": name}, "snapshot_restorability": {"total_snapshot_count": n, "restorable_snapshot_count": ok}}));
    }
    RestResp::json(200, &integrity_response(repo, &data.uuid, data.generation, log, (snaps_ok, data.snapshots.len()), &per_index, (index_snapshots_ok, index_snapshots), anomalies))
}

#[allow(clippy::too_many_arguments)]
fn integrity_response(repo: &str, uuid: &str, generation: i64, log: Vec<Value>, snapshots: (usize, usize), per_index: &BTreeMap<String, (usize, usize)>, index_snapshots: (usize, usize), anomalies: usize) -> Value {
    let indices_ok = per_index.values().filter(|(n, ok)| n == ok).count();
    json!({
        "log": log,
        "results": {
            "status": {
                "repository": {"name": repo, "uuid": uuid, "generation": generation},
                "snapshots": {"verified": snapshots.0, "total": snapshots.1},
                "indices": {"verified": indices_ok, "total": per_index.len()},
                "index_snapshots": {"verified": index_snapshots.0, "total": index_snapshots.1},
                "blobs": {"verified": 0}
            },
            "final_repository_generation": generation,
            "total_anomalies": anomalies,
            "result": if anomalies == 0 { "pass" } else { "fail" }
        }
    })
}

/// The `PUT _snapshot/{repo}` body registering `meta` with an engine.
fn engine_repository_body(meta: &RepositoryMetadata) -> Value {
    json!({"type": meta.repo_type, "settings": Value::Object(meta.settings.clone())})
}

pub async fn handle(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let segs = req.segments();
    let s: Vec<&str> = segs.iter().map(String::as_str).collect();
    let m = req.method.to_ascii_uppercase();
    let m = m.as_str();
    if s.first() == Some(&"_cat") {
        return match s.get(1).copied() {
            Some("repositories") => cat_repositories(node, req),
            _ => {
                if let Some(r) = on_master(node, req).await {
                    return r;
                }
                cat_snapshots(node, req, s.get(2).copied()).await
            }
        };
    }
    if s.first() == Some(&"_slm") {
        if let Some(r) = on_master(node, req).await {
            return r;
        }
        return crate::front::retry_on_new_master(node, req, move || slm::handle(node, req)).await;
    }
    // `_snapshot/...`
    let local_read = matches!((m, s.as_slice()), ("GET" | "HEAD", ["_snapshot"]) | ("GET" | "HEAD", ["_snapshot", _]));
    if local_read && s.get(1).copied() != Some("_status") {
        return get_repositories(node, s.get(1).copied());
    }
    if let Some(r) = on_master(node, req).await {
        return r;
    }
    crate::front::retry_on_new_master(node, req, move || master_op(node, req)).await
}

async fn master_op(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let segs = req.segments();
    let s: Vec<&str> = segs.iter().map(String::as_str).collect();
    let m = req.method.to_ascii_uppercase();
    let m = m.as_str();
    match (m, s.as_slice()) {
        ("GET", ["_snapshot", "_status"]) => ops::status(node, req, None, "").await,
        ("PUT" | "POST", ["_snapshot", repo]) => put_repository(node, req, repo).await,
        ("DELETE", ["_snapshot", repo]) => delete_repositories(node, repo).await,
        ("POST", ["_snapshot", repo, "_verify"]) => verify_repository(node, repo).await,
        ("POST", ["_snapshot", repo, "_cleanup"]) => cleanup_repository(node, repo).await,
        ("GET", ["_snapshot", repo, "_status"]) => ops::status(node, req, Some(repo), "").await,
        ("GET", ["_snapshot", repo, snaps, "_status"]) => ops::status(node, req, Some(repo), snaps).await,
        ("POST", ["_snapshot", repo, "_analyze" | "_verify_integrity"]) => engine_repository_op(node, req, repo).await,
        ("PUT" | "POST", ["_snapshot", repo, snap]) => ops::create(node, req, repo, snap).await,
        ("GET" | "HEAD", ["_snapshot", repo, snaps]) => ops::get(node, req, repo, snaps).await,
        ("DELETE", ["_snapshot", repo, snaps]) => ops::delete(node, repo, snaps).await,
        ("POST", ["_snapshot", repo, snap, "_restore"]) => restore::restore(node, req, repo, snap).await,
        ("POST", ["_snapshot", repo, snap, "_mount"]) => mount(node, req, repo, snap).await,
        ("PUT" | "POST", ["_snapshot", repo, source, "_clone", target]) => ops::clone(node, req, repo, source, target).await,
        _ => RestResp::error(405, "method_not_allowed", &format!("Incorrect HTTP method for uri [{}] and method [{m}], allowed: [GET, PUT, POST, DELETE]", req.uri)),
    }
}

// ------------------------------------------------------------ repositories

fn repo_json(e: &Value) -> Value {
    let mut settings = e["settings"].as_object().cloned().unwrap_or_default();
    // ES's `SettingsFilter` applies to repository settings too: the
    // deprecated insecure S3 `access_key` / `secret_key` are never returned.
    settings.retain(|k, _| !elasticsearch_engine::settings_filter::is_filtered(k));
    json!({"type": e["type"], "uuid": e["uuid"].as_str().unwrap_or("_na_"), "settings": Value::Object(stringify_settings(&settings))})
}

/// `GET _snapshot[/{repos}]` from the cluster state.
fn get_repositories(node: &ClusterNode, names: Option<&str>) -> RestResp {
    let st = node.state();
    let all: Map<String, Value> = st.metadata.customs.get(REPOS_CUSTOM).and_then(Value::as_object).cloned().unwrap_or_default();
    let available: Vec<String> = all.keys().cloned().collect();
    let exprs: Vec<String> = names.map(|n| n.split(',').map(str::to_string).collect()).unwrap_or_else(|| vec!["_all".into()]);
    let (selected, missing) = resolve_names(&exprs, &available);
    if let Some(m) = missing.first() {
        return err(&SnapshotError::repository_missing(m));
    }
    let mut out = Map::new();
    for n in selected {
        out.insert(n.clone(), repo_json(&all[&n]));
    }
    RestResp::json(200, &Value::Object(out))
}

fn default_repository(node: &ClusterNode) -> Option<String> {
    let state = node.state();
    state.metadata.transient_settings.get("repositories.default_repository")
        .or_else(|| state.metadata.persistent_settings.get("repositories.default_repository"))
        .and_then(Value::as_str).filter(|value| !value.is_empty()).map(str::to_string)
}

fn in_use(node: &ClusterNode, svc: &elasticsearch_snapshots::SnapshotsService, name: &str) -> bool {
    let st = node.state();
    !svc.running_snapshots(Some(name)).is_empty()
        || state_of(node).pending.lock().iter().any(|p| p.repository == name)
        || common::progress_entries(&st, "snapshots").iter().any(|e| e["repository"].as_str() == Some(name))
        || common::progress_entries(&st, "restore").iter().any(|e| e["repository"].as_str() == Some(name))
}

async fn put_repository(node: &Arc<ClusterNode>, req: &RestReq, name: &str) -> RestResp {
    let Some(body) = req.json() else {
        return RestResp::error(400, "parse_exception", "request body is required");
    };
    let meta = match RepositoryMetadata::from_put_body(name, &body) {
        Ok(m) => m,
        Err(e) => return err(&e),
    };
    if default_repository(node).as_deref() == Some(name) && (meta.settings.get("readonly").is_some_and(|v| v == &Value::Bool(true) || v.as_str() == Some("true"))) {
        let current = common::repos_in_state(&node.state());
        let action = if current.contains_key(name) { "mark" } else { "create" };
        let reason = if action == "mark" {
            format!("Repository [{name}] is the default repository. Cannot mark the default repository as read-only. Please change the default repository setting before marking this repository as read-only.")
        } else {
            format!("Repository [{name}] is the default repository. Cannot create the default repository as read-only. Please change the default repository setting before creating this repository as read-only.")
        };
        return RestResp::error(400, "illegal_argument_exception", &reason);
    }
    let verify = req.params().get("verify").map(|v| v != "false").unwrap_or(true);
    let svc = service(node).await;
    let current = common::repos_in_state(&node.state());
    if current.get(name).map(|c| c != &meta).unwrap_or(false) && in_use(node, &svc, name) {
        return err(&SnapshotError::illegal_state(format!("trying to modify or unregister repository [{name}] that is currently used")));
    }
    let repo = match Repository::open(meta.clone(), &repo_context(node)).await {
        Ok(r) => r,
        Err(e) => return err(&e),
    };
    let (uuid, generation) = match repo.repository_data().await {
        Ok(d) => (d.uuid.clone(), d.generation),
        Err(_) => ("_na_".to_string(), -1),
    };
    if verify {
        if let Err(e) = verify_nodes(node, &repo, &meta).await {
            return err(&e);
        }
    }
    let entry = json!({"type": meta.repo_type, "uuid": if uuid.is_empty() { "_na_".to_string() } else { uuid }, "settings": Value::Object(meta.settings.clone()), "generation": generation, "pending_generation": generation});
    let name2 = name.to_string();
    let r = common::update_custom(node, REPOS_CUSTOM, move |v| {
        if !v.is_object() {
            *v = json!({});
        }
        if v.get(&name2) == Some(&entry) {
            return Ok(false);
        }
        v[&name2] = entry;
        Ok(true)
    })
    .await;
    match r {
        Ok(()) => RestResp::json(200, &json!({"acknowledged": true})),
        Err(e) => crate::front::task_error(&e),
    }
}

/// `(name, uuid)` of the searchable snapshot indices mounted from `repo`.
fn mounted_from(st: &elasticsearch_cluster::state::ClusterState, repo: &str) -> Vec<(String, String)> {
    let setting = |im: &elasticsearch_cluster::state::IndexMetadata, k: &str| im.settings.get(k).and_then(|v| v.as_str().map(str::to_string));
    st.metadata
        .indices
        .iter()
        .filter(|(_, im)| setting(im, "index.store.type").as_deref() == Some("snapshot") && setting(im, "index.store.snapshot.repository_name").as_deref() == Some(repo))
        .map(|(n, im)| (n.clone(), im.uuid.clone()))
        .collect()
}

async fn delete_repositories(node: &Arc<ClusterNode>, names: &str) -> RestResp {
    let st = node.state();
    let available: Vec<String> = common::repos_in_state(&st).keys().cloned().collect();
    let exprs: Vec<String> = names.split(',').map(str::to_string).collect();
    let (selected, missing) = resolve_names(&exprs, &available);
    if let Some(m) = missing.first() {
        return err(&SnapshotError::repository_missing(m));
    }
    if selected.is_empty() {
        return err(&SnapshotError::repository_missing(names));
    }
    for name in &selected {
        if default_repository(node).as_deref() == Some(name) {
            return RestResp::error(409, "repository_conflict_exception", &format!("[{name}] trying to modify or unregister repository that is currently used (cannot delete the default repository. Please change the default repository cluster setting repositories.default_repository before deleting this repository.)"));
        }
    }
    for name in &selected {
        // A repository backing searchable snapshot indices cannot be unregistered.
        let used = mounted_from(&st, name);
        if !used.is_empty() {
            let shown: Vec<String> = used.iter().take(5).map(|(n, u)| format!("[{n}/{u}]")).collect();
            let more = if used.len() > 5 { ",..." } else { "" };
            return RestResp::error(409, "repository_conflict_exception", &format!("[{name}] trying to modify or unregister repository that is currently used (found {} searchable snapshots indices that use the repository: {}{more})", used.len(), shown.join(",")));
        }
    }
    let svc = service(node).await;
    for n in &selected {
        if in_use(node, &svc, n) {
            return err(&SnapshotError::illegal_state(format!("trying to modify or unregister repository [{n}] that is currently used")));
        }
    }
    let sel = selected.clone();
    let r = common::update_custom(node, REPOS_CUSTOM, move |v| {
        let Some(o) = v.as_object_mut() else { return Ok(false) };
        let before = o.len();
        for n in &sel {
            o.remove(n);
        }
        Ok(o.len() != before)
    })
    .await;
    for n in &selected {
        common::forget_repository(node, n).await;
    }
    match r {
        Ok(()) => RestResp::json(200, &json!({"acknowledged": true})),
        Err(e) => crate::front::task_error(&e),
    }
}

/// ES verification: the master writes a seed blob, every master-eligible
/// and data node reads it and writes its own blob, then the test container
/// is removed.
async fn verify_nodes(node: &Arc<ClusterNode>, repo: &Arc<Repository>, meta: &RepositoryMetadata) -> Result<Value, SnapshotError> {
    let st = node.state();
    let seed = random_base64_uuid();
    let readonly = repo.is_readonly();
    let c = repo.root().child(&format!("tests-{seed}"));
    if readonly {
        repo.latest_generation().await.map_err(|e| SnapshotError::repository_verification(&meta.name, format!("path [{}] is not accessible on master node: {}", repo.store().describe(), e.reason)))?;
    } else {
        c.write_blob_atomic("master.dat", bytes::Bytes::from(seed.clone().into_bytes()), true)
            .await
            .map_err(|e| SnapshotError::repository_verification(&meta.name, format!("path [{}] is not accessible on master node: {}", repo.store().describe(), e.reason)))?;
    }
    let req = (|| -> Result<_,elasticsearch_engine::error::EsError> {
        Ok(shard::VerifyReq {
            repository: crate::front::SharedText::try_new(&meta.name)?,
            repo_type: crate::front::SharedText::try_new(&meta.repo_type)?,
            settings_json: crate::front::SharedText::try_json(&meta.settings)?,
            seed: crate::front::SharedText::try_new(&seed)?,
        })
    })().map_err(|e|SnapshotError::repository_verification(&meta.name,e.to_string()))?;
    let targets: Vec<_> = st.nodes.values().filter(|n| n.is_master_eligible() || n.is_data()).cloned().collect();
    let futs = targets.iter().map(|n| {
        let req = req.clone();
        let node = node.clone();
        async move {
            let r = if n.id == node.local.id {
                shard::verify_local(&node, &req).await.map_err(|e|e.to_string())
            } else {
                match n.transport_address.parse::<std::net::SocketAddr>() {
                    Ok(a) => node.transport.send_request::<_, ()>(a, shard::ACTION_VERIFY, &req, Duration::from_secs(30)).await.map_err(|e| e.to_string()),
                    Err(_) => Err("bad transport address".into()),
                }
            };
            (n.id.clone(), n.name.clone(), r)
        }
    });
    let results = futures::future::join_all(futs).await;
    if !readonly {
        let _ = c.delete().await;
    }
    let mut ok = Map::new();
    let mut failures = Vec::new();
    for (id, name, r) in results {
        match r {
            Ok(()) => {
                ok.insert(id, json!({"name": name}));
            }
            Err(e) => failures.push(format!("[{id}, 'org.elasticsearch.repositories.RepositoryVerificationException: [{}] {e}']", meta.name)),
        }
    }
    if !failures.is_empty() {
        return Err(SnapshotError::repository_verification(&meta.name, format!("[{}]", failures.join(", "))));
    }
    Ok(json!({"nodes": ok}))
}

async fn verify_repository(node: &Arc<ClusterNode>, name: &str) -> RestResp {
    let svc = service(node).await;
    let repo = match svc.repository(name) {
        Ok(r) => r,
        Err(e) => return err(&e),
    };
    let meta = repo.metadata().clone();
    match verify_nodes(node, &repo, &meta).await {
        Ok(v) => RestResp::json(200, &v),
        Err(e) => err(&e),
    }
}

async fn cleanup_repository(node: &Arc<ClusterNode>, name: &str) -> RestResp {
    let svc = service(node).await;
    if svc.repository(name).is_ok() && in_use(node, &svc, name) {
        return err(&SnapshotError::illegal_state(format!("Cannot cleanup [{name}] - a snapshot is currently running")));
    }
    match svc.cleanup_repository(name).await {
        Ok(v) => {
            common::after_repo_write(node, &svc, name).await;
            RestResp::json(200, &v)
        }
        Err(e) => err(&e),
    }
}

// --------------------------------------------------------------------- cat

fn cat_repositories(node: &ClusterNode, req: &RestReq) -> RestResp {
    let rows: Vec<Row> = common::repos_in_state(&node.state()).values().map(|m| vec![("id", Some(m.name.clone())), ("type", Some(m.repo_type.clone()))]).collect();
    const REPO_COLS: &[ColDef] = &[("id", &["id", "repoId"], "unique repository id", true, false), ("type", &["t", "type"], "repository type", true, false)];
    cat::render(req, &cols(REPO_COLS), rows)
}

const SNAP_COLS: &[ColDef] = &[
    ("id", &["id", "snapshot"], "unique snapshot", true, false),
    ("repository", &["re", "repo"], "repository name", true, false),
    ("status", &["s", "status"], "snapshot name", true, false),
    ("start_epoch", &["ste", "startEpoch"], "start time in seconds since 1970-01-01 00:00:00", true, true),
    ("start_time", &["sti", "startTime"], "start time in HH:MM:SS", true, false),
    ("end_epoch", &["ete", "endEpoch"], "end time in seconds since 1970-01-01 00:00:00", true, true),
    ("end_time", &["eti", "endTime"], "end time in HH:MM:SS", true, false),
    ("duration", &["dur", "duration"], "duration", true, true),
    ("indices", &["i", "indices"], "number of indices", true, true),
    ("successful_shards", &["ss", "successful_shards"], "number of successful shards", true, true),
    ("failed_shards", &["fs", "failed_shards"], "number of failed shards", true, true),
    ("total_shards", &["ts", "total_shards"], "number of total shards", true, true),
    ("reason", &["r", "reason"], "reason for failures", true, false),
];

fn hms(ms: i64) -> String {
    chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ms).map(|d| d.format("%H:%M:%S").to_string()).unwrap_or_default()
}

async fn cat_snapshots(node: &Arc<ClusterNode>, req: &RestReq, repo: Option<&str>) -> RestResp {
    let cols = cols(SNAP_COLS);
    if req.params().get("help").map(|v| v != "false").unwrap_or(false) {
        return cat::render(req, &cols, vec![]);
    }
    let params = req.params();
    let repo = repo.map(str::to_string).or_else(|| params.get("repository").cloned()).unwrap_or_else(|| "_all".into());
    let svc = service(node).await;
    let mut p: BTreeMap<String, String> = BTreeMap::new();
    if let Some(v) = params.get("ignore_unavailable") {
        p.insert("ignore_unavailable".into(), v.clone());
    }
    let gr = match GetSnapshotsRequest::from_params(&repo, "_all", &p) {
        Ok(r) => r,
        Err(e) => return err(&e),
    };
    let v = match svc.get_snapshots(&gr).await {
        Ok(v) => v,
        Err(e) => return err(&e),
    };
    let mut rows = Vec::new();
    for s in v["snapshots"].as_array().cloned().unwrap_or_default() {
        let start = s["start_time_in_millis"].as_i64().unwrap_or(0);
        let end = s["end_time_in_millis"].as_i64().unwrap_or(0);
        let reason = s["reason"].as_str().map(str::to_string).or_else(|| s["failures"].as_array().filter(|a| !a.is_empty()).map(|a| a.iter().filter_map(|f| f["reason"].as_str()).collect::<Vec<_>>().join(", ")));
        let row: Row = vec![
            ("id", s["snapshot"].as_str().map(str::to_string)),
            ("repository", s["repository"].as_str().map(str::to_string)),
            ("status", s["state"].as_str().map(str::to_string)),
            ("start_epoch", Some((start / 1000).to_string())),
            ("start_time", Some(hms(start))),
            ("end_epoch", Some((end / 1000).to_string())),
            ("end_time", Some(hms(end))),
            ("duration", Some(format_duration_millis(s["duration_in_millis"].as_i64().unwrap_or((end - start).max(0))))),
            ("indices", Some(s["indices"].as_array().map(|a| a.len()).unwrap_or(0).to_string())),
            ("successful_shards", Some(s["shards"]["successful"].as_u64().unwrap_or(0).to_string())),
            ("failed_shards", Some(s["shards"]["failed"].as_u64().unwrap_or(0).to_string())),
            ("total_shards", Some(s["shards"]["total"].as_u64().unwrap_or(0).to_string())),
            ("reason", reason),
        ];
        rows.push(row);
    }
    cat::render(req, &cols, rows)
}

#[cfg(test)]
mod engine_repository_tests {
    use super::*;

    #[test]
    fn integrity_result_counts() {
        let per: BTreeMap<String, (usize, usize)> = [("a".to_string(), (2, 2)), ("b".to_string(), (1, 0))].into_iter().collect();
        let v = integrity_response("r", "u", 3, vec![], (2, 2), &per, (2, 3), 1);
        assert_eq!(v["results"]["status"]["indices"], json!({"verified": 1, "total": 2}));
        assert_eq!(v["results"]["result"], "fail");
        assert_eq!(integrity_response("r", "u", 0, vec![], (1, 1), &BTreeMap::new(), (0, 0), 0)["results"]["result"], "pass");
    }

    #[test]
    fn mount_request_parses_like_es() {
        let params: std::collections::HashMap<String, String> = [("storage".to_string(), "shared_cache".to_string()), ("wait_for_completion".to_string(), "true".to_string())].into_iter().collect();
        let m = parse_mount(&params, Some(&json!({"index": "docs", "renamed_index": "m", "index_settings": {"index.number_of_replicas": 0}}))).unwrap();
        assert_eq!((m.index.as_str(), m.mounted.as_str(), m.partial, m.wait), ("docs", "m", true, true));
        let e = parse_mount(&Default::default(), Some(&json!({"index": "docs", "foo": 1}))).unwrap_err();
        assert_eq!(e.kind, "x_content_parse_exception");
        let bad: std::collections::HashMap<String, String> = [("storage".to_string(), "x".to_string())].into_iter().collect();
        assert_eq!(parse_mount(&bad, Some(&json!({"index": "d"}))).unwrap_err().kind, "illegal_argument_exception");
    }

    #[test]
    fn mount_restore_body_sets_snapshot_store() {
        let m = parse_mount(&Default::default(), Some(&json!({"index": "docs"}))).unwrap();
        let src: Map<String, Value> = [("index.uuid".to_string(), json!("u1")), ("index.routing.allocation.require.box".to_string(), json!("hot"))].into_iter().collect();
        let b = mount_restore_body(&m, "repo", "snap", "su", &src);
        assert_eq!(b["index_settings"]["index.store.type"], "snapshot");
        assert_eq!(b["index_settings"]["index.store.snapshot.index_uuid"], "u1");
        assert_eq!(b["index_settings"]["index.blocks.write"], "true");
        assert_eq!(b["rename_replacement"], "docs");
        assert!(b["ignore_index_settings"].as_array().unwrap().contains(&json!("index.routing.allocation.require.box")));
    }

    #[test]
    fn engine_registration_body_carries_type_and_settings() {
        let meta = RepositoryMetadata::new("r", "fs", json!({"location": "/tmp/x", "readonly": true}));
        assert_eq!(engine_repository_body(&meta), json!({"type": "fs", "settings": {"location": "/tmp/x", "readonly": true}}));
    }
}
