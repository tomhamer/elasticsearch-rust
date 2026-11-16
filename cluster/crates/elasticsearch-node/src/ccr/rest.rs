//! The `_ccr` REST APIs in a cluster. Changes run on the elected master
//! (another node forwards them), as do the stats (the auto-follow
//! coordinator's are the master's; every node's shard follow task
//! statuses are gathered); follow info and pattern reads are served from
//! the cluster state anywhere. Request parsing, validation and error
//! strings are the engine's (`elasticsearch_engine::ccr`), as ES 9.5.5.

use super::follow;
use super::leader::{self, LeaderIndex, LeaseOp};
use super::state::{self, CcrState, Follower};
use crate::front::{RestReq, RestResp};
use crate::node::ClusterNode;
use elasticsearch_cluster::state::{ClusterState, IndexState};
use elasticsearch_engine::ccr::{self as eccr, FollowParams, FollowerMeta, PutFollowRequest, FOLLOWING_INDEX_SETTING};
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::settings::FlatSettings;
use serde_json::{json, Map, Value};
use std::sync::Arc;
use std::time::Duration;

fn err(e: &EsError) -> RestResp {
    RestResp::from_es_error(e)
}

fn ack() -> RestResp {
    RestResp::json(200, &json!({"acknowledged": true}))
}

fn state_err(e: String) -> EsError {
    if e.contains("not master") || e.contains("not_master") {
        EsError::new(503, "not_master_exception", e)
    } else {
        EsError::new(500, "illegal_state_exception", e)
    }
}

fn method_not_allowed(req: &RestReq, allowed: &[&str]) -> RestResp {
    RestResp::json(405, &json!({"error": format!("Incorrect HTTP method for uri [{}] and method [{}], allowed: [{}]", req.uri, req.method, allowed.join(", ")), "status": 405}))
}

/// Route a `_ccr` request (`None`: not a CCR API).
pub async fn handle(node: &Arc<ClusterNode>, req: &RestReq) -> Option<RestResp> {
    let segs = req.segments();
    let s: Vec<&str> = segs.iter().map(String::as_str).collect();
    let m = req.method.as_str();
    let resp = match s.as_slice() {
        ["_ccr", "stats"] => match m {
            "GET" => on_master(node, req, |n, r| Box::pin(async move { ccr_stats(&n, &r).await })).await,
            _ => method_not_allowed(req, &["GET"]),
        },
        ["_ccr", "auto_follow"] => match m {
            "GET" => get_patterns(node, None),
            _ => method_not_allowed(req, &["GET"]),
        },
        ["_ccr", "auto_follow", name] => {
            let name = name.to_string();
            match m {
                "GET" => get_patterns(node, Some(&name)),
                "PUT" => on_master(node, req, move |n, r| Box::pin(async move { put_pattern(&n, &r, &name).await })).await,
                "DELETE" => on_master(node, req, move |n, _| Box::pin(async move { delete_pattern(&n, &name).await })).await,
                _ => method_not_allowed(req, &["GET", "PUT", "DELETE"]),
            }
        }
        ["_ccr", "auto_follow", name, action @ ("pause" | "resume")] => {
            let (name, active) = (name.to_string(), *action == "resume");
            match m {
                "POST" => on_master(node, req, move |n, _| Box::pin(async move { activate_pattern(&n, &name, active).await })).await,
                _ => method_not_allowed(req, &["POST"]),
            }
        }
        [index, "_ccr", action] => {
            let index = index.to_string();
            match (*action, m) {
                ("follow", "PUT") => on_master(node, req, move |n, r| Box::pin(async move { follow(&n, &r, &index).await })).await,
                ("pause_follow", "POST") => on_master(node, req, move |n, _| Box::pin(async move { pause(&n, &index).await })).await,
                ("resume_follow", "POST") => on_master(node, req, move |n, r| Box::pin(async move { resume(&n, &r, &index).await })).await,
                ("unfollow", "POST") => on_master(node, req, move |n, _| Box::pin(async move { unfollow(&n, &index).await })).await,
                ("info", "GET") => info(node, &index),
                ("stats", "GET") => on_master(node, req, move |n, _| Box::pin(async move { index_stats(&n, &index).await })).await,
                ("forget_follower", "POST") => forget_follower(node, req, &index).await,
                ("follow", _) => method_not_allowed(req, &["PUT"]),
                ("info" | "stats", _) => method_not_allowed(req, &["GET"]),
                ("pause_follow" | "resume_follow" | "unfollow" | "forget_follower", _) => method_not_allowed(req, &["POST"]),
                _ => return None,
            }
        }
        _ => return None,
    };
    Some(crate::rest::util::postprocess(req, resp))
}

type Run = Box<dyn FnOnce(Arc<ClusterNode>, RestReq) -> futures::future::BoxFuture<'static, RestResp> + Send>;

/// Run on the elected master (forwarding from any other node).
async fn on_master<F>(node: &Arc<ClusterNode>, req: &RestReq, f: F) -> RestResp
where
    F: FnOnce(Arc<ClusterNode>, RestReq) -> futures::future::BoxFuture<'static, RestResp> + Send + 'static,
{
    if !node.is_master() {
        return crate::rest::forward_to_master(node, req).await;
    }
    let run: Run = Box::new(f);
    run(node.clone(), req.clone()).await
}

// ----------------------------------------------------------------- follow

/// `ActiveShardCount` of `wait_for_active_shards`: whether put-follow waits.
fn waits_for_shards(req: &RestReq) -> Result<bool, EsError> {
    match req.params().get("wait_for_active_shards").map(String::as_str) {
        None | Some("0") | Some("") | Some("index-setting") => Ok(false),
        Some("all") => Ok(true),
        Some(v) => match v.parse::<i64>() {
            Ok(n) if n >= 0 => Ok(n > 0),
            _ => Err(EsError::illegal_argument(format!("cannot parse ActiveShardCount[{v}]"))),
        },
    }
}

async fn follow(node: &Arc<ClusterNode>, req: &RestReq, index: &str) -> RestResp {
    let body = req.json().unwrap_or(Value::Null);
    let parsed = PutFollowRequest::parse(index, &body).and_then(|mut r| {
        r.wait_for_active_shards = waits_for_shards(req)?;
        Ok(r)
    });
    let r = match parsed {
        Ok(r) => r,
        Err(e) => return err(&e),
    };
    let identity = crate::front::ccs::identity(node, req).await;
    match put_follow(node, r, identity).await {
        Ok((created, acked, started)) => RestResp::json(200, &json!({"follow_index_created": created, "follow_index_shards_acked": acked, "index_following_started": started})),
        Err(e) => err(&e),
    }
}

/// The leader index of a follow request (`CcrLicenseChecker`): concrete,
/// open.
async fn leader_index(node: &ClusterNode, alias: &str, identity: Option<String>, name: &str, privileges: bool) -> Result<LeaderIndex, EsError> {
    let meta = leader::fetch_metadata(node, alias, identity, Some(vec![name.to_string()]), privileges).await?;
    if let Some(l) = meta.indices.into_iter().find(|i| i.name == name) {
        if !l.open {
            return Err(EsError::index_closed(&l.name, &l.uuid));
        }
        return Ok(l);
    }
    Err(match meta.abstractions.iter().find(|(n, _)| n == name) {
        Some((_, t)) => EsError::illegal_argument(format!("cannot follow [{name}], because it is a {t}")),
        None => EsError::index_not_found(name),
    })
}

/// Settings a follower never takes from its leader at creation.
const NOT_COPIED: &[&str] = &[
    "index.uuid", "index.history.uuid", "index.provided_name", "index.creation_date", "index.version.created", "index.version.upgraded",
    "index.version.upgraded_string", "index.verified_before_close", "index.routing.allocation.initial_recovery._id",
];

fn follower_settings(l: &LeaderIndex, overrides: &FlatSettings) -> Map<String, Value> {
    let mut s: Map<String, Value> = l
        .settings()
        .into_iter()
        .filter(|(k, _)| !NOT_COPIED.contains(&k.as_str()) && !k.starts_with("index.resize.") && !k.starts_with("index.shrink.") && !k.starts_with("index.history."))
        .collect();
    s.insert(FOLLOWING_INDEX_SETTING.into(), json!("true"));
    for (k, v) in overrides {
        if v.is_null() {
            s.remove(k);
        } else {
            s.insert(k.clone(), v.clone());
        }
    }
    s
}

/// `TransportPutFollowAction` on the master: the follower restored from
/// its leader (ES's `_ccr_<alias>` repository), then following. Returns
/// `(follow_index_created, follow_index_shards_acked,
/// index_following_started)`; without `wait_for_active_shards` it answers
/// before the restore completes, and following still starts.
pub async fn put_follow(node: &Arc<ClusterNode>, req: PutFollowRequest, identity: Option<String>) -> Result<(bool, bool, bool), EsError> {
    req.validate()?;
    let alias = req.remote_cluster.clone().unwrap_or_default();
    let leader_name = req.leader_index.clone().unwrap_or_default();
    let l = leader_index(node, &alias, identity.clone(), &leader_name, true).await?;
    if !l.soft_deletes() {
        return Err(EsError::illegal_argument(format!("leader index [{leader_name}] does not have soft deletes enabled")));
    }
    if l.searchable_snapshot() {
        return Err(EsError::illegal_argument(format!(
            "leader index [{leader_name}] is a searchable snapshot index and cannot be used as a leader index for cross-cluster replication purpose"
        )));
    }
    let replicated = eccr::filter_settings(&req.settings);
    if !replicated.is_empty() {
        let unknown: Vec<&str> = replicated.keys().map(String::as_str).filter(|k| !eccr::known_index_setting(k)).collect();
        let msg = if unknown.is_empty() {
            format!("can not put follower index that could override leader settings {}", eccr::settings_string(&replicated))
        } else {
            format!("unknown setting{} [{}]", if unknown.len() == 1 { "" } else { "s" }, unknown.join(","))
        };
        return Err(EsError::illegal_argument(msg));
    }
    let follower = req.follower_index.clone();
    elasticsearch_engine::node::validate_index_name(&follower)?;
    let st = node.state();
    if let Some(existing) = st.metadata.indices.get(&follower) {
        if existing.state == IndexState::Open {
            return Err(EsError::new(
                500,
                "snapshot_restore_exception",
                format!(
                    "[default:_ccr_{alias}:_latest_/_latest_] cannot restore index [{follower}] because an open index with same name already exists in the cluster. Either close or delete the existing index or restore the index under a different name by providing a rename pattern and replacement name"
                ),
            ));
        }
        let r = crate::front::dispatch(node, RestReq::try_new("DELETE", &format!("/{follower}"), &crate::front::system_headers(), &[], 0, Default::default(), None)?).await;
        if !r.is_success() {
            return Err(EsError::new(r.status, "illegal_state_exception", format!("cannot replace the closed index [{follower}]")));
        }
    }
    // The follower's CCR metadata first: its primaries' restore reads the
    // identity to fetch the leader's files with.
    let entry = Follower {
        uuid: String::new(),
        meta: FollowerMeta {
            follower_index: follower.clone(),
            remote_cluster: alias.clone(),
            leader_index: l.name.clone(),
            leader_uuid: l.uuid.clone(),
            leader_history_uuids: (0..l.shards).map(|_| l.uuid.clone()).collect(),
            created_millis: elasticsearch_engine::date::now_millis(),
        },
        identity: identity.clone(),
        params: None,
    };
    let name = follower.clone();
    state::update(node, move |s| {
        s.followers.insert(name, entry);
        Ok(true)
    })
    .await
    .map_err(state_err)?;
    let body = json!({"settings": Value::Object(follower_settings(&l, &req.settings)), "mappings": l.mappings()});
    let params = req.params.resolve();
    let restore = {
        let (node, follower, alias, l) = (node.clone(), follower.clone(), alias.clone(), l.clone());
        async move {
            let r = crate::snapshot_api::restore::restore_follower(&node, &follower, &alias, &l.name, &l.uuid, l.primary_term, &body).await;
            match r {
                Ok((_, uuid)) => {
                    let name = follower.clone();
                    state::update(&node, move |s| {
                        let Some(f) = s.followers.get_mut(&name) else { return Err(format!("follower [{name}] was removed while it was restored")) };
                        f.uuid = uuid;
                        f.params = Some(params);
                        Ok(true)
                    })
                    .await
                    .map_err(state_err)?;
                    // Aliases (and anything the leader changed meanwhile).
                    let identity = CcrState::of(&node.state()).followers.get(&follower).and_then(|f| f.identity.clone());
                    if let Err(e) = follow::sync_metadata(&node, &follower, &alias, &l.name, identity).await {
                        tracing::warn!(%follower, "initial metadata sync of the follower failed: {e}");
                    }
                    Ok(())
                }
                Err(e) => {
                    let name = follower.clone();
                    let _ = state::update(&node, move |s| Ok(s.followers.remove(&name).is_some())).await;
                    Err(EsError::new(500, "snapshot_restore_exception", format!("[{}:_latest_/_latest_] {e}", super::restore::repository(&alias))))
                }
            }
        }
    };
    if !req.wait_for_active_shards {
        tokio::spawn(async move {
            if let Err(e) = restore.await {
                tracing::warn!("follower restore failed: {e}");
            }
        });
        return Ok((true, false, false));
    }
    restore.await?;
    Ok((true, true, true))
}

// ------------------------------------------------------- pause / resume

fn follower_of<'a>(ccr: &'a CcrState, st: &ClusterState, index: &str) -> Result<&'a Follower, EsError> {
    if !st.metadata.indices.contains_key(index) {
        return Err(EsError::index_not_found(index));
    }
    ccr.follower(st, index).ok_or_else(|| EsError::illegal_argument(format!("index [{index}] is not a follower index")))
}

async fn pause(node: &Arc<ClusterNode>, index: &str) -> RestResp {
    let st = node.state();
    let ccr = CcrState::of(&st);
    let f = match follower_of(&ccr, &st, index) {
        Ok(f) => f,
        Err(e) => return err(&e),
    };
    if f.params.is_none() {
        return err(&EsError::illegal_argument(format!("no shard follow tasks for [{index}]")));
    }
    let name = index.to_string();
    match state::update(node, move |s| Ok(s.followers.get_mut(&name).is_some_and(|f| f.params.take().is_some()))).await {
        Ok(()) => ack(),
        Err(e) => err(&state_err(e)),
    }
}

async fn resume(node: &Arc<ClusterNode>, req: &RestReq, index: &str) -> RestResp {
    let params = match eccr::parse_resume_body(&req.json().unwrap_or(Value::Null)) {
        Ok(p) => p,
        Err(e) => return err(&e),
    };
    let mut errs = Vec::new();
    params.validate(&mut errs);
    if !errs.is_empty() {
        return err(&EsError::action_validation(&errs.iter().map(String::as_str).collect::<Vec<_>>()));
    }
    match resume_follow(node, index, &params).await {
        Ok(()) => ack(),
        Err(e) => err(&e),
    }
}

/// `TransportResumeFollowAction.validate` + starting the shard follow tasks.
async fn resume_follow(node: &Arc<ClusterNode>, follower: &str, params: &FollowParams) -> Result<(), EsError> {
    let st = node.state();
    let ccr = CcrState::of(&st);
    let f = follower_of(&ccr, &st, follower).map_err(|e| {
        if e.kind == "illegal_argument_exception" { EsError::illegal_argument(format!("follow index [{follower}] does not have ccr metadata")) } else { e }
    })?.clone();
    let fim = st.metadata.indices[follower].clone();
    let l = leader_index(node, &f.meta.remote_cluster, f.identity.clone(), &f.meta.leader_index, false).await.map_err(|e| {
        if e.kind == "index_closed_exception" { EsError::illegal_argument("leader and follow index must be open") } else { e }
    })?;
    if l.uuid != f.meta.leader_uuid {
        return Err(EsError::illegal_argument(format!("follow index [{follower}] should reference [{}] as leader index but instead reference [{}] as leader index", l.uuid, f.meta.leader_uuid)));
    }
    if !l.soft_deletes() {
        return Err(EsError::illegal_argument(format!("leader index [{}] does not have soft deletes enabled", l.name)));
    }
    if l.searchable_snapshot() {
        return Err(EsError::illegal_argument(format!("leader index [{}] is a searchable snapshot index and cannot be used for cross-cluster replication purpose", l.name)));
    }
    if fim.setting("index.soft_deletes.enabled") == Some("false") {
        return Err(EsError::illegal_argument(format!("follower index [{follower}] does not have soft deletes enabled")));
    }
    if l.shards != fim.number_of_shards.max(1) {
        return Err(EsError::illegal_argument(format!("leader index primary shards [{}] does not match with the number of shards of the follow index [{}]", l.shards, fim.number_of_shards)));
    }
    if l.routing_shards != fim.routing_num_shards {
        return Err(EsError::illegal_argument(format!(
            "leader index number_of_routing_shards [{}] does not match with the number_of_routing_shards of the follow index [{}]",
            l.routing_shards, fim.routing_num_shards
        )));
    }
    if fim.state != IndexState::Open {
        return Err(EsError::illegal_argument("leader and follow index must be open"));
    }
    if fim.setting(FOLLOWING_INDEX_SETTING) != Some("true") {
        return Err(EsError::illegal_argument(format!("the following index [{follower}] is not ready to follow; the setting [{FOLLOWING_INDEX_SETTING}] must be enabled.")));
    }
    let strip = |s: &FlatSettings| {
        let mut f = eccr::filter_settings(s);
        f.remove("index.lifecycle.indexing_complete");
        f
    };
    let fs: FlatSettings = fim.settings.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    let (ls, fs) = (strip(&l.settings()), strip(&fs));
    if ls != fs {
        return Err(EsError::illegal_argument(format!("the leader index settings [{}] and follower index settings [{}] must be identical", eccr::settings_string(&ls), eccr::settings_string(&fs))));
    }
    if let Some(svc) = node.engine.get_index(follower) {
        svc.check_put_mapping(&l.mappings())?;
    }
    if f.params.is_some() {
        let exists = |n: u32| format!("task with id {{{}-{n}}} already exist", fim.uuid);
        let mut e = EsError::new(400, "resource_already_exists_exception", exists(0));
        if fim.number_of_shards > 1 {
            let sup: Vec<Value> = (1..fim.number_of_shards).map(|n| json!({"type": "resource_already_exists_exception", "reason": exists(n)})).collect();
            e = e.with("suppressed", Value::Array(sup));
        }
        return Err(e);
    }
    let (name, resolved) = (follower.to_string(), params.resolve());
    state::update(node, move |s| {
        let Some(f) = s.followers.get_mut(&name) else { return Err(format!("follower [{name}] is gone")) };
        f.params = Some(resolved);
        Ok(true)
    })
    .await
    .map_err(state_err)
}

// ---------------------------------------------------------------- unfollow

async fn unfollow(node: &Arc<ClusterNode>, index: &str) -> RestResp {
    let st = node.state();
    let ccr = CcrState::of(&st);
    let f = match follower_of(&ccr, &st, index) {
        Ok(f) => f.clone(),
        Err(e) => return err(&e),
    };
    if st.metadata.indices[index].state == IndexState::Open {
        return err(&EsError::illegal_argument(format!("cannot convert the follower index [{index}] to a non-follower, because it has not been closed")));
    }
    if f.params.is_some() {
        return err(&EsError::illegal_argument(format!("cannot convert the follower index [{index}] to a non-follower, because it has not been paused")));
    }
    // The following setting goes (as any follower metadata change, through the master).
    let mut change = eccr::FollowerMetadataChange::default();
    change.settings.insert(FOLLOWING_INDEX_SETTING.into(), Value::Null);
    if let Err(e) = crate::node::commit_follower_metadata(node, index, &change).await {
        return err(&crate::remote_cluster::remote_error(&e));
    }
    let name = index.to_string();
    if let Err(e) = state::update(node, move |s| Ok(s.followers.remove(&name).is_some())).await {
        return err(&state_err(e));
    }
    // The follower's retention leases on the leader.
    let lease_id = eccr::retention_lease_id(&st.cluster_name, index, &f.uuid, &f.meta.remote_cluster, &f.meta.leader_index, &f.meta.leader_uuid);
    let removed = leader::change_lease(node, &f.meta.remote_cluster, f.identity.clone(), &f.meta.leader_index, &f.meta.leader_uuid, None, LeaseOp::Remove, &lease_id, 0).await;
    if let Err(e) = removed {
        let wrapped = EsError::new(500, "exception", format!("{}: {}", eccr::java_exception_class(&e.kind), e.reason)).with("failed_to_remove_retention_leases", lease_id.clone()).caused_by(e);
        return err(&wrapped);
    }
    ack()
}

// ------------------------------------------------------------ info / stats

fn info(node: &Arc<ClusterNode>, expr: &str) -> RestResp {
    let opts = elasticsearch_engine::node::ResolveOpts { ignore_unavailable: false, allow_no_indices: true, expand_open: true, expand_closed: true, expand_hidden: false };
    let names: Vec<String> = match node.engine.resolve(expr, opts) {
        Ok(t) => t.into_iter().map(|t| t.svc.name.clone()).collect(),
        Err(e) => return err(&e),
    };
    let st = node.state();
    let ccr = CcrState::of(&st);
    let mut out = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for n in names {
        if !seen.insert(n.clone()) {
            continue;
        }
        let Some(f) = ccr.follower(&st, &n) else { continue };
        let mut m = Map::new();
        m.insert("follower_index".into(), json!(n));
        m.insert("remote_cluster".into(), json!(f.meta.remote_cluster));
        m.insert("leader_index".into(), json!(f.meta.leader_index));
        match &f.params {
            Some(p) => {
                m.insert("status".into(), json!("active"));
                let mut pm = Map::new();
                p.fragment(&mut pm);
                m.insert("parameters".into(), Value::Object(pm));
            }
            None => {
                m.insert("status".into(), json!("paused"));
            }
        }
        out.push(Value::Object(m));
    }
    RestResp::json(200, &json!({"follower_indices": out}))
}

/// `FollowStatsAction.StatsResponses` of `names` (sorted).
async fn follow_stats_json(node: &Arc<ClusterNode>, names: &[String]) -> Value {
    let st = node.state();
    let ccr = CcrState::of(&st);
    let statuses = follow::gather(node).await;
    let mut indices = Vec::new();
    for name in names {
        let Some(f) = ccr.follower(&st, name) else { continue };
        let n = st.metadata.indices.get(name).map(|im| im.number_of_shards.max(1)).unwrap_or(1);
        let mut shards = Vec::new();
        let mut lag = 0i64;
        for k in 0..n {
            let v = statuses.get(&(name.clone(), k)).cloned().unwrap_or_else(|| follow::idle_status(f, name, k, &st));
            lag += v["leader_global_checkpoint"].as_i64().unwrap_or(0) - v["follower_global_checkpoint"].as_i64().unwrap_or(0);
            shards.push(v);
        }
        indices.push(json!({"index": name, "total_global_checkpoint_lag": lag, "shards": shards}));
    }
    json!({"indices": indices})
}

fn following_names(st: &ClusterState, filter: Option<&[String]>) -> Vec<String> {
    let mut v: Vec<String> = CcrState::of(st).following(st).into_iter().map(|(n, _)| n).filter(|n| filter.is_none_or(|f| f.contains(n))).collect();
    v.sort();
    v
}

async fn index_stats(node: &Arc<ClusterNode>, expr: &str) -> RestResp {
    let names: Vec<String> = expr.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    let all = names.is_empty() || (names.len() == 1 && (names[0] == "_all" || names[0] == "*"));
    let st = node.state();
    let list = following_names(&st, if all { None } else { Some(&names) });
    if !all && list.is_empty() {
        return err(&EsError::resource_not_found(format!("No shard follow tasks for follower indices [{}]", names.join(","))));
    }
    RestResp::json(200, &follow_stats_json(node, &list).await)
}

async fn ccr_stats(node: &Arc<ClusterNode>, _req: &RestReq) -> RestResp {
    let st = node.state();
    let list = following_names(&st, None);
    let follow = follow_stats_json(node, &list).await;
    RestResp::json(200, &json!({"auto_follow_stats": super::auto_follow::stats_json(node), "follow_stats": follow}))
}

// ---------------------------------------------------------- forget follower

const FORGET_FOLLOWER_PARSER: &str = "forget_follower";

/// `POST /{index}/_ccr/forget_follower` on the leader cluster
/// (`TransportForgetFollowerAction`): remove a follower's retention lease
/// from every copy of every shard; the broadcast header counts primaries.
async fn forget_follower(node: &Arc<ClusterNode>, req: &RestReq, expr: &str) -> RestResp {
    let body = req.json().unwrap_or(Value::Null);
    let o = match &body {
        Value::Object(o) => o.clone(),
        Value::Null => return err(&EsError::new(400, "parse_exception", "request body is required")),
        other => return err(&EsError::x_content_parse(format!("[{FORGET_FOLLOWER_PARSER}] Expected START_OBJECT but was: {}", elasticsearch_engine::jsonpos::token_name(other)))),
    };
    const FIELDS: [&str; 4] = ["follower_cluster", "follower_index", "follower_index_uuid", "leader_remote_cluster"];
    let mut vals: [Option<String>; 4] = Default::default();
    for (k, v) in &o {
        let Some(i) = FIELDS.iter().position(|f| f == k) else {
            return err(&EsError::unknown_field_xc(FORGET_FOLLOWER_PARSER, k, None, &FIELDS));
        };
        vals[i] = Some(match v {
            Value::String(s) => s.clone(),
            Value::Number(_) | Value::Bool(_) => v.to_string(),
            _ => return err(&EsError::unsupported_type_xc(FORGET_FOLLOWER_PARSER, k, v, None)),
        });
    }
    if vals.iter().any(Option::is_none) {
        return err(&EsError::new(500, "null_pointer_exception", elasticsearch_engine::error::NULL_REASON));
    }
    let [fc, fi, fu, lrc] = vals.map(Option::unwrap_or_default);
    let names: Vec<String> = match node.engine.resolve(expr, elasticsearch_engine::node::ResolveOpts::default()) {
        Ok(t) => t.into_iter().map(|t| t.svc.name.clone()).collect(),
        Err(e) => return err(&e),
    };
    let st = node.state();
    let (mut total, mut ok) = (0u64, 0u64);
    let mut failures: Vec<Value> = Vec::new();
    for name in names {
        let Some(im) = st.metadata.indices.get(&name) else { continue };
        let id = eccr::retention_lease_id(&fc, &fi, &fu, &lrc, &name, &im.uuid);
        let lreq = leader::LeaseReq { identity: None, index: name.clone(), uuid: im.uuid.clone(), shard: None, op: LeaseOp::Remove, id: id.clone(), retaining_seq_no: 0, routed: false };
        let results = leader::change_lease_here(node, lreq).await.map(|r| r.results).unwrap_or_default();
        for k in 0..im.number_of_shards.max(1) {
            total += 1;
            match results.iter().find(|(s, primary, _)| *s == k && *primary) {
                Some((_, _, None)) => ok += 1,
                Some((_, _, Some((kind, reason)))) if kind == "retention_lease_not_found_exception" => failures.push(json!({
                    "shard": k, "index": name, "status": "NOT_FOUND", "reason": {"type": kind, "reason": reason}
                })),
                Some((_, _, Some((kind, reason)))) => failures.push(json!({"shard": k, "index": name, "status": "INTERNAL_SERVER_ERROR", "reason": {"type": kind, "reason": reason}})),
                None => failures.push(json!({"shard": k, "index": name, "status": "SERVICE_UNAVAILABLE", "reason": {"type": "no_shard_available_action_exception", "reason": format!("[{name}][{k}] no active primary shard copy")}})),
            }
        }
    }
    let failed = failures.len();
    let mut grouped: Vec<Value> = Vec::new();
    for f in failures {
        if !grouped.iter().any(|g| g["index"] == f["index"] && g["reason"] == f["reason"]) {
            grouped.push(f);
        }
    }
    let mut shards = json!({"total": total, "successful": ok, "failed": failed});
    if !grouped.is_empty() {
        shards["failures"] = Value::Array(grouped);
    }
    RestResp::json(200, &json!({"_shards": shards}))
}

// ------------------------------------------------------- auto-follow patterns

fn get_patterns(node: &Arc<ClusterNode>, name: Option<&str>) -> RestResp {
    let ccr = CcrState::of(&node.state());
    let mut out = Vec::new();
    match name {
        None => {
            for (n, p) in &ccr.patterns {
                out.push(json!({"name": n, "pattern": p.to_json()}));
            }
        }
        Some(n) => match ccr.patterns.get(n) {
            Some(p) => out.push(json!({"name": n, "pattern": p.to_json()})),
            None => return err(&eccr::pattern_missing(n)),
        },
    }
    RestResp::json(200, &json!({"patterns": out}))
}

async fn put_pattern(node: &Arc<ClusterNode>, req: &RestReq, name: &str) -> RestResp {
    let pattern = match eccr::parse_auto_follow_pattern(name, &req.json().unwrap_or(Value::Null)) {
        Ok(p) => p,
        Err(e) => return err(&e),
    };
    let identity = crate::front::ccs::identity(node, req).await;
    // ES reads the leader cluster's state: existing matching indices count
    // as followed (only those of new patterns on an update).
    let meta = match leader::fetch_metadata(node, &pattern.remote_cluster, identity.clone(), None, false).await {
        Ok(m) => m,
        Err(e) => return err(&e),
    };
    let name = name.to_string();
    let r = state::update(node, move |s| {
        let previous = s.patterns.get(&name).cloned();
        let mark: Vec<String> = match &previous {
            Some(p) => pattern.leader_index_patterns.iter().filter(|x| !p.leader_index_patterns.contains(x)).cloned().collect(),
            None => pattern.leader_index_patterns.clone(),
        };
        let mut followed = s.followed.get(&name).cloned().unwrap_or_default();
        for l in &meta.indices {
            if eccr::auto_follow_match(&mark, &pattern.leader_index_exclusion_patterns, &l.name, l.data_stream.as_deref()) && !followed.contains(&l.uuid) {
                followed.push(l.uuid.clone());
            }
        }
        s.followed.insert(name.clone(), followed);
        s.pattern_identity.insert(name.clone(), identity);
        s.patterns.insert(name, pattern);
        Ok(true)
    })
    .await;
    super::auto_follow::wake(node);
    match r {
        Ok(()) => ack(),
        Err(e) => err(&state_err(e)),
    }
}

async fn delete_pattern(node: &Arc<ClusterNode>, name: &str) -> RestResp {
    if !CcrState::of(&node.state()).patterns.contains_key(name) {
        return err(&eccr::pattern_missing(name));
    }
    let n = name.to_string();
    match state::update(node, move |s| {
        s.followed.remove(&n);
        s.pattern_identity.remove(&n);
        Ok(s.patterns.remove(&n).is_some())
    })
    .await
    {
        Ok(()) => ack(),
        Err(e) => err(&state_err(e)),
    }
}

async fn activate_pattern(node: &Arc<ClusterNode>, name: &str, active: bool) -> RestResp {
    if !CcrState::of(&node.state()).patterns.contains_key(name) {
        return err(&eccr::pattern_missing(name));
    }
    let n = name.to_string();
    let r = state::update(node, move |s| match s.patterns.get_mut(&n) {
        Some(p) if p.active != active => {
            p.active = active;
            Ok(true)
        }
        _ => Ok(false),
    })
    .await;
    super::auto_follow::wake(node);
    match r {
        Ok(()) => ack(),
        Err(e) => err(&state_err(e)),
    }
}

/// Wait (bounded) for the state to show `f` holds.
#[allow(dead_code)]
pub(crate) async fn wait_for(node: &ClusterNode, timeout: Duration, f: impl Fn(&ClusterState) -> bool) -> bool {
    let mut rx = node.subscribe();
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if f(&rx.borrow_and_update()) {
            return true;
        }
        if tokio::time::timeout_at(deadline, rx.changed()).await.is_err() {
            return false;
        }
    }
}
