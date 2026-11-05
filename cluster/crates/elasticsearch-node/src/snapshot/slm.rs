//! Snapshot lifecycle management (`_slm/*`).
//!
//! Policies, the operation mode and the stats live in the
//! `snapshot_lifecycle` metadata custom (ES `SnapshotLifecycleMetadata`),
//! so they survive restarts and master changes. The elected master runs
//! the schedules: every second it fires the policies whose next execution
//! time passed (taking snapshots through the same path as
//! `PUT _snapshot/{repo}/{snap}`), records `last_success` / `last_failure`
//! and runs retention on `slm.retention_schedule` (ES default
//! `0 30 1 * * ?`).

use super::common::{self, err, now_millis, service, state_of, SLM_CUSTOM};
use super::ops;
use crate::front::{RestReq, RestResp};
use crate::node::ClusterNode;
use elasticsearch_cluster::state::ClusterState;
use elasticsearch_snapshots::slm::{self, Cron, OperationMode, PolicyMetadata, PolicyStats, SlmInProgress, SlmStats, SnapshotInvocationRecord, SnapshotLifecyclePolicy};
use elasticsearch_snapshots::util::parse_time_value;
use elasticsearch_snapshots::{SnapshotError, SnapshotState};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

const DEFAULT_RETENTION_SCHEDULE: &str = "0 30 1 * * ?";

#[derive(Default)]
pub struct SlmModel {
    pub policies: BTreeMap<String, PolicyMetadata>,
    pub mode: OperationMode,
    pub stats: SlmStats,
}

fn record_from(v: &Value) -> Option<SnapshotInvocationRecord> {
    let o = v.as_object()?;
    Some(SnapshotInvocationRecord {
        snapshot_name: o.get("snapshot_name")?.as_str()?.to_string(),
        start_time: o.get("start_time").and_then(Value::as_i64),
        time: o.get("time").and_then(Value::as_i64).unwrap_or(0),
        details: o.get("details").and_then(Value::as_str).map(str::to_string),
    })
}

impl SlmModel {
    pub fn from_value(v: Option<&Value>) -> SlmModel {
        let mut m = SlmModel::default();
        let Some(v) = v else { return m };
        if let Some(ps) = v["policies"].as_object() {
            for (id, p) in ps {
                let Ok(policy) = SnapshotLifecyclePolicy::from_json(id, &p["policy"]) else { continue };
                m.policies.insert(
                    id.clone(),
                    PolicyMetadata {
                        policy,
                        version: p["version"].as_u64().unwrap_or(1),
                        modified_date: p["modified_date_millis"].as_i64().unwrap_or(0),
                        last_success: record_from(&p["last_success"]),
                        last_failure: record_from(&p["last_failure"]),
                        invocations_since_last_success: p["invocations_since_last_success"].as_u64().unwrap_or(0),
                    },
                );
            }
        }
        m.mode = match v["operation_mode"].as_str() {
            Some("STOPPING") => OperationMode::Stopping,
            Some("STOPPED") => OperationMode::Stopped,
            _ => OperationMode::Running,
        };
        let s = &v["stats"];
        m.stats.retention_runs = s["retention_runs"].as_u64().unwrap_or(0);
        m.stats.retention_failed = s["retention_failed"].as_u64().unwrap_or(0);
        m.stats.retention_timed_out = s["retention_timed_out"].as_u64().unwrap_or(0);
        m.stats.retention_deletion_time_millis = s["retention_deletion_time_millis"].as_i64().unwrap_or(0);
        if let Some(a) = s["policy_stats"].as_array() {
            for p in a {
                let Some(id) = p["policy"].as_str() else { continue };
                m.stats.policies.insert(
                    id.to_string(),
                    PolicyStats {
                        snapshots_taken: p["snapshots_taken"].as_u64().unwrap_or(0),
                        snapshots_failed: p["snapshots_failed"].as_u64().unwrap_or(0),
                        snapshots_deleted: p["snapshots_deleted"].as_u64().unwrap_or(0),
                        snapshot_deletion_failures: p["snapshot_deletion_failures"].as_u64().unwrap_or(0),
                    },
                );
            }
        }
        m
    }

    pub fn to_value(&self) -> Value {
        let rec = |r: &SnapshotInvocationRecord| r.to_json(false);
        let mut ps = Map::new();
        for (id, p) in &self.policies {
            let mut e = Map::new();
            e.insert("policy".into(), p.policy.to_json());
            e.insert("headers".into(), json!({}));
            e.insert("version".into(), json!(p.version));
            e.insert("modified_date_millis".into(), json!(p.modified_date));
            if let Some(s) = &p.last_success {
                e.insert("last_success".into(), rec(s));
            }
            if let Some(f) = &p.last_failure {
                e.insert("last_failure".into(), rec(f));
            }
            e.insert("invocations_since_last_success".into(), json!(p.invocations_since_last_success));
            ps.insert(id.clone(), Value::Object(e));
        }
        json!({"policies": ps, "operation_mode": self.mode.as_str(), "stats": self.stats.to_json()})
    }
}

pub fn model(st: &ClusterState) -> SlmModel {
    SlmModel::from_value(st.metadata.customs.get(SLM_CUSTOM))
}

/// Edit the SLM custom on the master. `f` returns whether it changed.
async fn mutate<F>(node: &ClusterNode, f: F) -> Result<(), String>
where
    F: FnOnce(&mut SlmModel) -> Result<bool, String> + Send + 'static,
{
    common::update_custom(node, SLM_CUSTOM, move |v| {
        let mut m = SlmModel::from_value(if v.as_object().map(|o| o.is_empty()).unwrap_or(true) { None } else { Some(&*v) });
        if !f(&mut m)? {
            return Ok(false);
        }
        *v = m.to_value();
        Ok(true)
    })
    .await
}

fn minimum_interval(st: &ClusterState) -> Option<i64> {
    match st.metadata.setting("slm.minimum_interval").and_then(Value::as_str) {
        Some(s) => parse_time_value("slm.minimum_interval", s).ok(),
        None => Some(slm::DEFAULT_MINIMUM_INTERVAL_MILLIS),
    }
}

fn bool_param(req: &RestReq, k: &str) -> bool {
    req.params().get(k).map(|v| v != "false").unwrap_or(false)
}

fn in_progress(node: &ClusterNode) -> BTreeMap<String, SlmInProgress> {
    state_of(node).slm_running.lock().iter().map(|(id, (name, uuid, start))| (id.clone(), SlmInProgress { name: name.clone(), uuid: uuid.clone(), state: "STARTED".into(), start_time: *start, failure: None })).collect()
}

/// `_slm/*` (master).
pub async fn handle(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let segs = req.segments();
    let s: Vec<&str> = segs.iter().map(String::as_str).collect();
    let method = req.method.as_str();
    match (method, s.as_slice()) {
        ("GET", ["_slm", "policy"]) => get_policies(node, req, &[]),
        ("GET", ["_slm", "policy", ids]) => {
            let ids: Vec<String> = ids.split(',').map(str::to_string).collect();
            get_policies(node, req, &ids)
        }
        ("PUT", ["_slm", "policy", id]) => put_policy(node, req, id).await,
        ("DELETE", ["_slm", "policy", id]) => delete_policy(node, id).await,
        ("PUT" | "POST", ["_slm", "policy", id, "_execute"]) => match execute_policy(node, id).await {
            Ok(name) => RestResp::json(200, &json!({"snapshot_name": name})),
            Err(r) => r,
        },
        ("POST", ["_slm", "_execute_retention"]) => {
            let n = node.clone();
            tokio::spawn(async move { run_retention(&n).await });
            RestResp::json(200, &json!({"acknowledged": true}))
        }
        ("GET", ["_slm", "stats"]) => RestResp::json(200, &model(&node.state()).stats.to_json()),
        ("GET", ["_slm", "status"]) => RestResp::json(200, &model(&node.state()).mode.status_json()),
        ("POST", ["_slm", "start"]) => set_mode(node, OperationMode::Running).await,
        ("POST", ["_slm", "stop"]) => {
            let running = !state_of(node).slm_running.lock().is_empty();
            set_mode(node, if running { OperationMode::Stopping } else { OperationMode::Stopped }).await
        }
        _ => RestResp::error(405, "method_not_allowed", &format!("Incorrect HTTP method for uri [{}] and method [{method}]", req.uri)),
    }
}

fn get_policies(node: &ClusterNode, req: &RestReq, ids: &[String]) -> RestResp {
    let m = model(&node.state());
    if m.policies.is_empty() && !ids.is_empty() && !ids.iter().all(|i| i == "_all" || i == "*") {
        return err(&SnapshotError::resource_not_found(format!("snapshot lifecycle policy or policies [{}] not found, no policies are configured", ids.join(", "))));
    }
    match slm::policies_json(&m.policies, ids, &m.stats, &in_progress(node), now_millis(), bool_param(req, "human")) {
        Ok(v) => RestResp::json(200, &v),
        Err(e) => err(&e),
    }
}

async fn put_policy(node: &Arc<ClusterNode>, req: &RestReq, id: &str) -> RestResp {
    let Some(body) = req.json() else {
        return RestResp::error(400, "parse_exception", "request body is required");
    };
    let policy = match SnapshotLifecyclePolicy::from_json(id, &body) {
        Ok(p) => p,
        Err(e) => return err(&e),
    };
    let st = node.state();
    if let Err(e) = policy.validate(minimum_interval(&st)) {
        return err(&e);
    }
    if !common::repos_in_state(&st).contains_key(&policy.repository) {
        return err(&SnapshotError::illegal_argument(format!("no such repository [{}]", policy.repository)));
    }
    let now = now_millis();
    let r = mutate(node, move |m| {
        let prev = m.policies.get(&policy.id);
        if prev.map(|p| p.policy == policy).unwrap_or(false) {
            return Ok(false);
        }
        let pm = PolicyMetadata::new(policy.clone(), now, prev);
        m.policies.insert(policy.id.clone(), pm);
        Ok(true)
    })
    .await;
    match r {
        Ok(()) => RestResp::json(200, &json!({"acknowledged": true})),
        Err(e) => crate::front::task_error(&e),
    }
}

async fn delete_policy(node: &Arc<ClusterNode>, id: &str) -> RestResp {
    if !model(&node.state()).policies.contains_key(id) {
        return err(&SnapshotError::resource_not_found(format!("Snapshot lifecycle policy not found: {id}")));
    }
    let id2 = id.to_string();
    match mutate(node, move |m| Ok(m.policies.remove(&id2).is_some())).await {
        Ok(()) => RestResp::json(200, &json!({"acknowledged": true})),
        Err(e) => crate::front::task_error(&e),
    }
}

async fn set_mode(node: &Arc<ClusterNode>, mode: OperationMode) -> RestResp {
    match mutate(node, move |m| {
        let changed = m.mode != mode;
        m.mode = mode;
        Ok(changed)
    })
    .await
    {
        Ok(()) => RestResp::json(200, &json!({"acknowledged": true})),
        Err(e) => crate::front::task_error(&e),
    }
}

/// Take a policy's snapshot now; returns its name.
pub async fn execute_policy(node: &Arc<ClusterNode>, id: &str) -> Result<String, RestResp> {
    let m = model(&node.state());
    let Some(pm) = m.policies.get(id).cloned() else {
        return Err(err(&SnapshotError::resource_not_found(format!("no such snapshot lifecycle policy [{id}]"))));
    };
    let start = now_millis();
    let name = pm.policy.generate_snapshot_name(start).map_err(|e| err(&e))?;
    let params = pm.policy.snapshot_params().map_err(|e| err(&e))?;
    let started = match ops::start_create(node, &pm.policy.repository, &name, params).await {
        Ok(s) => s,
        Err(r) => {
            let details = String::from_utf8_lossy(&r.body).into_owned();
            let (id2, name2) = (id.to_string(), name.clone());
            let _ = mutate(node, move |m| {
                if let Some(p) = m.policies.get_mut(&id2) {
                    p.record_failure(&name2, start, now_millis(), &details);
                }
                m.stats.snapshot_failed(&id2);
                Ok(true)
            })
            .await;
            return Err(r);
        }
    };
    state_of(node).slm_running.lock().insert(id.to_string(), (name.clone(), started.pending.uuid.clone(), start));
    let node2 = node.clone();
    let (id2, name2) = (id.to_string(), name.clone());
    tokio::spawn(async move {
        let outcome = started.handle.await;
        let failure: Option<String> = match &outcome {
            Ok(Ok(info)) if info.state == Some(SnapshotState::Success) => None,
            Ok(Ok(info)) => Some(json!({"type": "snapshot_exception", "reason": format!("[{}:{}] failed to create snapshot successfully, {} out of {} total shards failed", info.repository, info.snapshot, info.failed_shards(), info.total_shards)}).to_string()),
            Ok(Err(e)) => Some(e.to_json()["error"].to_string()),
            Err(e) => Some(json!({"type": "exception", "reason": e.to_string()}).to_string()),
        };
        state_of(&node2).slm_running.lock().remove(&id2);
        let r = mutate(&node2, move |m| {
            if let Some(p) = m.policies.get_mut(&id2) {
                match &failure {
                    None => p.record_success(&name2, start, now_millis()),
                    Some(d) => p.record_failure(&name2, start, now_millis(), d),
                }
            }
            match failure {
                None => m.stats.snapshot_taken(&id2),
                Some(_) => m.stats.snapshot_failed(&id2),
            }
            if m.mode == OperationMode::Stopping {
                m.mode = OperationMode::Stopped;
            }
            Ok(true)
        })
        .await;
        if let Err(e) = r {
            tracing::warn!("failed to record SLM outcome: {e}");
        }
    });
    Ok(name)
}

/// Delete the snapshots retention selects, for every policy.
pub async fn run_retention(node: &Arc<ClusterNode>) {
    let t0 = now_millis();
    let m = model(&node.state());
    let svc = service(node).await;
    let mut failed = false;
    let mut deleted: Vec<(String, bool)> = Vec::new();
    for (id, pm) in &m.policies {
        let Some(ret) = pm.policy.retention.as_ref().filter(|r| !r.is_empty()) else { continue };
        let Ok(r) = svc.repository(&pm.policy.repository) else {
            failed = true;
            continue;
        };
        let Ok(data) = r.repository_data().await else {
            failed = true;
            continue;
        };
        let mut infos = Vec::new();
        for e in &data.snapshots {
            if !e.slm_policy.is_empty() && e.slm_policy != *id {
                continue;
            }
            if let Ok(i) = r.read_snapshot_info(&e.uuid).await {
                infos.push(i);
            }
        }
        for s in ret.snapshots_to_delete(id, &infos, now_millis()) {
            let resp = ops::delete(node, &pm.policy.repository, &s.snapshot).await;
            deleted.push((id.clone(), resp.is_success()));
            if !resp.is_success() {
                failed = true;
            }
        }
    }
    let took = now_millis() - t0;
    let r = mutate(node, move |m| {
        for (id, ok) in &deleted {
            if *ok {
                m.stats.snapshot_deleted(id);
            } else {
                m.stats.snapshot_deletion_failure(id);
            }
        }
        m.stats.retention_run(took, failed, false);
        Ok(true)
    })
    .await;
    if let Err(e) = r {
        tracing::warn!("failed to record SLM retention stats: {e}");
    }
}

/// One scheduler round (elected master, every second).
pub async fn tick(node: &Arc<ClusterNode>) {
    let st = node.state();
    let Some(_) = st.metadata.customs.get(SLM_CUSTOM) else { return };
    let m = model(&st);
    // `_slm/stop` completes once no policy's snapshot runs (here, on the
    // elected master: a snapshot started by a previous master no longer
    // counts).
    if m.mode == OperationMode::Stopping && state_of(node).slm_running.lock().is_empty() {
        let r = mutate(node, |m| {
            if m.mode == OperationMode::Stopping {
                m.mode = OperationMode::Stopped;
                Ok(true)
            } else {
                Ok(false)
            }
        })
        .await;
        if let Err(e) = r {
            tracing::warn!("failed to complete the SLM stop: {e}");
        }
        return;
    }
    if m.mode != OperationMode::Running {
        return;
    }
    let s = state_of(node);
    let now = now_millis();
    let mut due = Vec::new();
    {
        let mut next = s.slm_next.lock();
        next.retain(|id, _| m.policies.contains_key(id));
        for (id, pm) in &m.policies {
            match next.get(id) {
                Some((v, at)) if *v == pm.version => {
                    if now >= *at {
                        due.push(id.clone());
                        next.insert(id.clone(), (pm.version, pm.next_execution_millis(now).unwrap_or(i64::MAX)));
                    }
                }
                _ => {
                    next.insert(id.clone(), (pm.version, pm.next_execution_millis(now).unwrap_or(i64::MAX)));
                }
            }
        }
    }
    for id in due {
        if s.slm_running.lock().contains_key(&id) {
            continue;
        }
        tracing::info!(policy = %id, "SLM policy triggered");
        if let Err(r) = execute_policy(node, &id).await {
            tracing::warn!(policy = %id, "SLM snapshot failed to start: {}", String::from_utf8_lossy(&r.body));
        }
    }
    // Retention.
    if !m.policies.values().any(|p| p.policy.retention.as_ref().map(|r| !r.is_empty()).unwrap_or(false)) {
        return;
    }
    let sched = st.metadata.setting("slm.retention_schedule").and_then(Value::as_str).unwrap_or(DEFAULT_RETENTION_SCHEDULE).to_string();
    let Ok(cron) = Cron::parse(&sched) else { return };
    let fire = {
        let mut g = s.slm_retention_next.lock();
        match &*g {
            Some((sc, at)) if *sc == sched => {
                if now >= *at {
                    *g = Some((sched.clone(), cron.next_after(now).unwrap_or(i64::MAX)));
                    true
                } else {
                    false
                }
            }
            _ => {
                *g = Some((sched.clone(), cron.next_after(now).unwrap_or(i64::MAX)));
                false
            }
        }
    };
    if fire {
        let n = node.clone();
        tokio::spawn(async move { run_retention(&n).await });
    }
}
