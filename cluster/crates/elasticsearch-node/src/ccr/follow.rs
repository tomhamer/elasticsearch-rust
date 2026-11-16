//! Shard follow tasks (ES `ShardFollowNodeTask`), run by the node holding
//! each follower primary.
//!
//! A reconcile loop (on every applied cluster state, and every second)
//! starts a task for each primary of a following follower this node holds
//! and stops tasks whose follower was paused, unfollowed or deleted, or
//! whose primary moved (a new holder starts its own). A task:
//! 1. renews its retention lease on every copy of the leader shard every
//!    `index.ccr.retention_lease.renew_interval` (30s), retaining from the
//!    follower's global checkpoint + 1 (the lease is added if missing);
//! 2. reads the leader shard's operations from the follower primary's
//!    local checkpoint + 1 (`max_read_request_operation_count`,
//!    `max_read_request_size`, a long poll of `read_poll_timeout`);
//! 3. replicates the leader's mapping, settings and aliases onto the
//!    follower (through the master) whenever the leader's metadata versions
//!    move, before writing operations that may need them;
//! 4. writes the operations to the follower primary with the leader's
//!    seq_no, version and the follower's primary term, in batches of
//!    `max_write_request_operation_count` / `max_write_request_size`, under
//!    the primary's write gate, and waits until its replicas have them.
//! Failures ES retries are retried with its backoff (capped by
//! `max_retry_delay`) and listed in `read_exceptions`; any other failure is
//! the task's `fatal_exception` (it then stops until paused and resumed).

use super::leader::{self, LeaseOp, ShardChangesReq, ShardChangesResp};
use super::state::{CcrState, Follower};
use crate::node::{ClusterNode, TransportNet};
use crate::shards::PrimaryCtx;
use elasticsearch_engine::ccr::{op_size, retermed, retry_delay, shard_status_json, should_retry, FollowParams, ShardTaskView};
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::hooks::changes::{ChangeOp, Durability};
use elasticsearch_transport::RemoteError;
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::{Duration, Instant};

/// Gathers every node's shard follow task statuses.
pub const ACTION_TASK_STATS: &str = "internal:ccr/shard_follow_task_stats";

/// `ShardFollowNodeTaskStatus` inputs.
#[derive(Debug)]
pub struct TaskStatus {
    pub leader_global_checkpoint: i64,
    pub leader_max_seq_no: i64,
    pub follower_global_checkpoint: i64,
    pub follower_max_seq_no: i64,
    pub last_requested_seq_no: i64,
    pub total_read_time_millis: u64,
    pub total_read_remote_exec_time_millis: u64,
    pub successful_read_requests: u64,
    pub failed_read_requests: u64,
    pub operations_read: u64,
    pub bytes_read: u64,
    pub total_write_time_millis: u64,
    pub successful_write_requests: u64,
    pub failed_write_requests: u64,
    pub operations_written: u64,
    /// from_seq_no → (retries, exception)
    pub read_exceptions: BTreeMap<i64, (u32, EsError)>,
    pub last_read: Option<Instant>,
    pub fatal: Option<EsError>,
}

impl TaskStatus {
    fn new(checkpoint: i64, max: i64) -> TaskStatus {
        TaskStatus {
            leader_global_checkpoint: checkpoint,
            leader_max_seq_no: max,
            follower_global_checkpoint: checkpoint,
            follower_max_seq_no: max,
            last_requested_seq_no: checkpoint,
            total_read_time_millis: 0,
            total_read_remote_exec_time_millis: 0,
            successful_read_requests: 0,
            failed_read_requests: 0,
            operations_read: 0,
            bytes_read: 0,
            total_write_time_millis: 0,
            successful_write_requests: 0,
            failed_write_requests: 0,
            operations_written: 0,
            read_exceptions: BTreeMap::new(),
            last_read: Some(Instant::now()),
            fatal: None,
        }
    }

    /// The status view (follower metadata versions from the follower index).
    pub fn view(&self, shard: u32, versions: (i64, i64, i64)) -> ShardTaskView {
        let failing = self.fatal.is_some() || !self.read_exceptions.is_empty();
        ShardTaskView {
            shard,
            leader_global_checkpoint: self.leader_global_checkpoint,
            leader_max_seq_no: self.leader_max_seq_no,
            follower_global_checkpoint: self.follower_global_checkpoint,
            follower_max_seq_no: self.follower_max_seq_no,
            last_requested_seq_no: self.last_requested_seq_no,
            outstanding_read_requests: if failing { 0 } else { 1 },
            mapping_version: versions.0,
            settings_version: versions.1,
            aliases_version: versions.2,
            total_read_time_millis: self.total_read_time_millis,
            total_read_remote_exec_time_millis: self.total_read_remote_exec_time_millis,
            successful_read_requests: self.successful_read_requests,
            failed_read_requests: self.failed_read_requests,
            operations_read: self.operations_read,
            bytes_read: self.bytes_read,
            total_write_time_millis: self.total_write_time_millis,
            successful_write_requests: self.successful_write_requests,
            failed_write_requests: self.failed_write_requests,
            operations_written: self.operations_written,
            read_exceptions: self.read_exceptions.iter().map(|(k, v)| (*k, v.clone())).collect(),
            time_since_last_read_millis: self.last_read.map(|i| i.elapsed().as_millis() as i64).unwrap_or(-1),
            fatal: self.fatal.clone(),
        }
    }
}

struct Task {
    /// What the task was started for: a change restarts it.
    allocation_id: String,
    term: u64,
    follower_uuid: String,
    params: String,
    stop: Arc<AtomicBool>,
    status: Arc<Mutex<TaskStatus>>,
}

type Tasks = Mutex<BTreeMap<(String, u32), Task>>;

/// Per node (by node id): its running tasks.
fn runtimes() -> &'static Mutex<BTreeMap<String, Arc<Tasks>>> {
    static R: OnceLock<Mutex<BTreeMap<String, Arc<Tasks>>>> = OnceLock::new();
    R.get_or_init(Default::default)
}

fn tasks_of(node: &ClusterNode) -> Arc<Tasks> {
    runtimes().lock().entry(node.local.id.clone()).or_default().clone()
}

/// `index.ccr.retention_lease.renew_interval` (node setting, 30s).
fn renew_interval(node: &ClusterNode) -> Duration {
    node.config
        .settings
        .get("index.ccr.retention_lease.renew_interval")
        .and_then(Value::as_str)
        .and_then(elasticsearch_engine::date::parse_time_value)
        .map(|ms| Duration::from_millis(ms.max(0) as u64))
        .unwrap_or(Duration::from_secs(30))
}

fn still_primary(node: &ClusterNode, p: &Arc<PrimaryCtx>) -> bool {
    node.shards.primary(&p.index, p.shard).is_some_and(|cur| Arc::ptr_eq(&cur, p))
}

fn params_fp(p: &FollowParams) -> String {
    serde_json::to_string(p).unwrap_or_default()
}

/// Start / stop tasks to match the state.
fn reconcile(node: &Arc<ClusterNode>) {
    let st = node.state();
    let ccr = CcrState::of(&st);
    let tasks = tasks_of(node);
    let mut desired: BTreeMap<(String, u32), (Arc<PrimaryCtx>, Follower)> = BTreeMap::new();
    for (name, f) in ccr.following(&st) {
        let Some(im) = st.metadata.indices.get(&name) else { continue };
        if im.state != elasticsearch_cluster::state::IndexState::Open {
            continue;
        }
        for k in 0..im.number_of_shards.max(1) {
            if let Some(p) = node.shards.primary(&name, k) {
                desired.insert((name.clone(), k), (p, f.clone()));
            }
        }
    }
    let mut g = tasks.lock();
    g.retain(|key, t| {
        let keep = desired.get(key).is_some_and(|(p, f)| {
            p.allocation_id == t.allocation_id && p.term == t.term && f.uuid == t.follower_uuid && f.params.as_ref().map(params_fp).unwrap_or_default() == t.params
        });
        if !keep {
            t.stop.store(true, Ordering::Release);
        }
        keep
    });
    for (key, (p, f)) in desired {
        if g.contains_key(&key) {
            continue;
        }
        let Some(params) = f.params.clone() else { continue };
        let copy = p.copy();
        let status = Arc::new(Mutex::new(TaskStatus::new(p.global_checkpoint(), copy.shard.max_seq_no())));
        let stop = Arc::new(AtomicBool::new(false));
        g.insert(key.clone(), Task { allocation_id: p.allocation_id.clone(), term: p.term, follower_uuid: f.uuid.clone(), params: params_fp(&params), stop: stop.clone(), status: status.clone() });
        tracing::info!(follower = %key.0, shard = key.1, leader = %f.meta.leader_index, remote = %f.meta.remote_cluster, "starting shard follow task");
        let weak = Arc::downgrade(node);
        tokio::spawn(run(weak, key.0, key.1, p, f, params, status, stop));
    }
}

/// A failed read (or write): retried with backoff, or fatal.
async fn failed(status: &Mutex<TaskStatus>, from: i64, e: EsError, retries: &mut u32, params: &FollowParams, stop: &AtomicBool) {
    let delay = {
        let mut s = status.lock();
        s.failed_read_requests += 1;
        if should_retry(&e) {
            *retries += 1;
            // ES keeps at most `max_outstanding_read_requests` of them.
            let entry = s.read_exceptions.entry(from).or_insert((0, e.clone()));
            entry.0 = *retries;
            entry.1 = e;
            Some(retry_delay(*retries, params.retry_delay_ms()))
        } else {
            tracing::warn!("shard follow task failed: {e}");
            s.fatal = Some(e);
            None
        }
    };
    match delay {
        Some(d) => tokio::time::sleep(d).await,
        None => {
            while !stop.load(Ordering::Acquire) {
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run(weak: Weak<ClusterNode>, name: String, shard: u32, p: Arc<PrimaryCtx>, f: Follower, params: FollowParams, status: Arc<Mutex<TaskStatus>>, stop: Arc<AtomicBool>) {
    let alias = f.meta.remote_cluster.clone();
    let leader = f.meta.leader_index.clone();
    let leader_uuid = f.meta.leader_uuid.clone();
    let identity = weak.upgrade().and_then(|n| crate::remote_cluster::outgoing_identity(&n, &alias, f.identity.clone()));
    let mut retries = 0u32;
    let mut last_renewal: Option<Instant> = None;
    // The leader metadata versions last replicated (`None`: sync first).
    let mut synced: Option<(u64, u64, u64)> = None;
    loop {
        if stop.load(Ordering::Acquire) {
            break;
        }
        let Some(node) = weak.upgrade() else { break };
        if !still_primary(&node, &p) {
            break;
        }
        let st = node.state();
        let lease_id = elasticsearch_engine::ccr::retention_lease_id(&st.cluster_name, &name, &f.uuid, &alias, &leader, &leader_uuid);
        if last_renewal.is_none_or(|t| t.elapsed() >= renew_interval(&node)) {
            let retain = p.global_checkpoint() + 1;
            match leader::change_lease(&node, &alias, identity.clone(), &leader, &leader_uuid, Some(shard), LeaseOp::Renew, &lease_id, retain).await {
                Ok(_) => last_renewal = Some(Instant::now()),
                Err(e) => tracing::debug!(follower = %name, shard, "retention lease renewal failed: {e}"),
            }
        }
        let copy = p.copy();
        let from = copy.shard.local_checkpoint() + 1;
        let req = ShardChangesReq {
            identity: identity.clone(),
            index: leader.clone(),
            uuid: leader_uuid.clone(),
            shard,
            from_seq_no: from,
            max_operation_count: params.read_ops() as u64,
            max_batch_size: params.read_bytes(),
            poll_timeout_ms: params.poll_timeout_ms() as u64,
            routed: false,
        };
        let t0 = Instant::now();
        let timeout = Duration::from_millis(params.poll_timeout_ms() as u64) + Duration::from_secs(60);
        let read: Result<ShardChangesResp, EsError> = leader::call(&node, &alias, leader::ACTION_SHARD_CHANGES, &req, timeout).await;
        let resp = match read {
            Ok(r) => r,
            Err(e) => {
                drop(node);
                failed(&status, from, e, &mut retries, &params, &stop).await;
                continue;
            }
        };
        // Paused (or moved) during the long poll: nothing is written.
        if stop.load(Ordering::Acquire) || !still_primary(&node, &p) {
            break;
        }
        let bytes: u64 = resp.operations.iter().map(op_size).sum();
        {
            let mut s = status.lock();
            s.successful_read_requests += 1;
            s.total_read_time_millis += t0.elapsed().as_millis() as u64;
            s.total_read_remote_exec_time_millis += resp.took_millis;
            s.operations_read += resp.operations.len() as u64;
            s.bytes_read += bytes;
            s.leader_global_checkpoint = resp.global_checkpoint;
            s.leader_max_seq_no = resp.max_seq_no;
            s.last_read = Some(Instant::now());
            s.last_requested_seq_no = s.last_requested_seq_no.max(resp.global_checkpoint.min(from + params.read_ops() as i64 - 1));
            s.read_exceptions.clear();
        }
        retries = 0;
        // The leader's metadata first: the operations may use what it added.
        let versions = (resp.mapping_version, resp.settings_version, resp.aliases_version);
        if synced != Some(versions) {
            match sync_metadata(&node, &name, &alias, &leader, identity.clone()).await {
                Ok(()) => synced = Some(versions),
                Err(e) => {
                    drop(node);
                    failed(&status, from, e, &mut retries, &params, &stop).await;
                    continue;
                }
            }
        }
        if !resp.operations.is_empty() {
            if let Err(e) = write(&node, &p, &name, resp.operations, &params, &status).await {
                drop(node);
                failed(&status, from, e, &mut retries, &params, &stop).await;
                continue;
            }
        }
        let mut s = status.lock();
        s.follower_global_checkpoint = p.global_checkpoint();
        s.follower_max_seq_no = p.copy().shard.max_seq_no();
    }
    tracing::info!(follower = %name, shard, "shard follow task stopped");
}

/// ES `TransportBulkShardOperationsAction` on the follower primary.
async fn write(node: &Arc<ClusterNode>, p: &Arc<PrimaryCtx>, name: &str, ops: Vec<ChangeOp>, params: &FollowParams, status: &Mutex<TaskStatus>) -> Result<(), EsError> {
    let durability = match node.state().metadata.indices.get(name).and_then(|im| im.setting("index.translog.durability").map(str::to_string)).as_deref() {
        Some("async") => Durability::Async,
        _ => Durability::Request,
    };
    let (max_ops, max_bytes) = (params.write_ops(), params.write_bytes());
    let mut batches: Vec<Vec<ChangeOp>> = Vec::new();
    let mut cur: Vec<ChangeOp> = Vec::new();
    let mut cur_bytes = 0u64;
    for op in ops {
        cur_bytes += op_size(&op);
        cur.push(op);
        if cur.len() >= max_ops || cur_bytes >= max_bytes {
            batches.push(std::mem::take(&mut cur));
            cur_bytes = 0;
        }
    }
    if !cur.is_empty() {
        batches.push(cur);
    }
    for batch in batches {
        let w0 = Instant::now();
        let n = batch.len() as u64;
        let gate = p.gate.read().await;
        if !still_primary(node, p) || p.relocated.load(Ordering::SeqCst) {
            return Err(EsError::new(503, "shard_not_in_primary_mode_exception", format!("[{name}][{}] the follower primary moved", p.shard)));
        }
        let (copy, term) = (p.copy(), p.term);
        let batch: Vec<ChangeOp> = batch.into_iter().map(|o| retermed(o, term)).collect();
        let applied = tokio::task::spawn_blocking(move || copy.apply_replica_ops(term, batch, durability)).await.map_err(|e| EsError::internal(e.to_string()))?;
        let res = match applied {
            Ok(r) => r,
            Err(e) => {
                let mut s = status.lock();
                s.failed_write_requests += 1;
                s.total_write_time_millis += w0.elapsed().as_millis() as u64;
                return Err(e);
            }
        };
        // The follower's replicas get them as any write on this primary.
        let replicated = p.replicate_through(res.max_seq_no, &TransportNet(node.clone())).await;
        drop(gate);
        let mut s = status.lock();
        s.total_write_time_millis += w0.elapsed().as_millis() as u64;
        s.successful_write_requests += 1;
        s.operations_written += n;
        s.follower_max_seq_no = res.max_seq_no;
        drop(s);
        if let Err(e) = replicated {
            return Err(EsError::new(503, "unavailable_shards_exception", format!("[{name}][{}] replication of followed operations failed: {e}", p.shard)));
        }
    }
    Ok(())
}

/// Replicate the leader's mapping, the replicated settings that differ and
/// the aliases onto follower `name` (ES `ShardFollowTasksExecutor`
/// `innerUpdateMapping` / `innerUpdateSettings` / `innerUpdateAliases`),
/// through the master.
pub async fn sync_metadata(node: &Arc<ClusterNode>, name: &str, alias: &str, leader: &str, identity: Option<String>) -> Result<(), EsError> {
    let meta = leader::fetch_metadata(node, alias, identity, Some(vec![leader.to_string()]), false).await?;
    let Some(l) = meta.indices.into_iter().find(|i| i.name == leader) else {
        return Err(EsError::index_not_found(leader));
    };
    let st = node.state();
    let Some(fim) = st.metadata.indices.get(name) else { return Err(EsError::index_not_found(name)) };
    let change = metadata_change(&l, &*fim.settings, &*fim.mappings, &fim.aliases, fim.state == elasticsearch_cluster::state::IndexState::Open);
    if change.mapping.is_none() && change.settings.is_empty() && change.aliases.is_none() {
        return Ok(());
    }
    crate::node::commit_follower_metadata(node, name, &change).await.map_err(|e: RemoteError| crate::remote_cluster::remote_error(&e))
}

/// What replicating `l` onto a follower with these settings, mapping and
/// aliases changes.
pub fn metadata_change(l: &leader::LeaderIndex, settings: &serde_json::Map<String, Value>, mappings: &Value, aliases: &BTreeMap<String, Value>, open: bool) -> elasticsearch_engine::ccr::FollowerMetadataChange {
    use elasticsearch_engine::ccr::filter_settings;
    let mut change = elasticsearch_engine::ccr::FollowerMetadataChange::default();
    let lmap = l.mappings();
    if !lmap.is_null() && &lmap != mappings && open {
        change.mapping = Some(lmap);
    }
    let lf = filter_settings(&l.settings());
    let ff = filter_settings(&settings.iter().map(|(k, v)| (k.clone(), v.clone())).collect::<elasticsearch_engine::settings::FlatSettings>());
    change.settings = lf.iter().filter(|(k, v)| ff.get(*k) != Some(*v)).map(|(k, v)| (k.clone(), v.clone())).collect();
    change.reopen = change.settings.keys().any(|k| !elasticsearch_engine::settings::is_dynamic(k));
    let laliases: BTreeMap<String, Value> = l
        .aliases()
        .into_iter()
        .map(|(k, mut v)| {
            if let Some(o) = v.as_object_mut() {
                o.insert("is_write_index".into(), json!(false));
            }
            (k, v)
        })
        .collect();
    if &laliases != aliases {
        change.aliases = Some(laliases);
    }
    change
}

/// This node's task statuses: (follower, shard, status JSON).
fn local_stats(node: &ClusterNode) -> Vec<(String, u32, String)> {
    let st = node.state();
    let ccr = CcrState::of(&st);
    let tasks = tasks_of(node);
    let g = tasks.lock();
    g.iter()
        .filter_map(|((name, k), t)| {
            let f = ccr.followers.get(name)?;
            let im = st.metadata.indices.get(name)?;
            let versions = (im.mapping_version as i64, im.settings_version as i64, im.aliases_version as i64);
            let v = t.status.lock().view(*k, versions);
            Some((name.clone(), *k, shard_status_json(&f.meta.remote_cluster, &f.meta.leader_index, name, &v).to_string()))
        })
        .collect()
}

/// Every node's shard follow task statuses, by (follower, shard).
pub async fn gather(node: &Arc<ClusterNode>) -> BTreeMap<(String, u32), Value> {
    let st = node.state();
    let mut out = BTreeMap::new();
    let answers = futures::future::join_all(st.nodes.keys().map(|id| {
        let node = node.clone();
        let id = id.clone();
        async move {
            if id == node.local.id {
                return local_stats(&node);
            }
            let Some(addr) = node.address_of(&id) else { return Vec::new() };
            node.transport.send_request::<_, Vec<(String, u32, String)>>(addr, ACTION_TASK_STATS, &(), Duration::from_secs(30)).await.unwrap_or_default()
        }
    }))
    .await;
    for (name, k, v) in answers.into_iter().flatten() {
        if let Ok(v) = serde_json::from_str(&v) {
            out.insert((name, k), v);
        }
    }
    out
}

/// The status of a shard whose task does not run (yet) anywhere.
pub fn idle_status(f: &Follower, name: &str, shard: u32, st: &elasticsearch_cluster::state::ClusterState) -> Value {
    let im = st.metadata.indices.get(name);
    let versions = im.map(|im| (im.mapping_version as i64, im.settings_version as i64, im.aliases_version as i64)).unwrap_or((1, 1, 1));
    let s = TaskStatus::new(-1, -1);
    shard_status_json(&f.meta.remote_cluster, &f.meta.leader_index, name, &s.view(shard, versions))
}

pub fn register(node: &Arc<ClusterNode>) {
    let n = node.clone();
    node.transport.register_handler(ACTION_TASK_STATS, move |_: ()| {
        let n = n.clone();
        async move { Ok::<_, RemoteError>(local_stats(&n)) }
    });
    let weak = Arc::downgrade(node);
    let mut states = node.subscribe();
    tokio::spawn(async move {
        loop {
            let Some(node) = weak.upgrade() else { break };
            if node.transport.is_closed() {
                break;
            }
            reconcile(&node);
            drop(node);
            let closed = tokio::select! {
                r = states.changed() => r.is_err(),
                _ = tokio::time::sleep(Duration::from_secs(1)) => false,
            };
            if closed {
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
        // The node is gone: stop its tasks.
        if let Some(node_id) = weak.upgrade().map(|n| n.local.id.clone()) {
            if let Some(t) = runtimes().lock().remove(&node_id) {
                for task in t.lock().values() {
                    task.stop.store(true, Ordering::Release);
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leader(settings: Value, mappings: Value, aliases: Value) -> leader::LeaderIndex {
        leader::LeaderIndex { name: "l".into(), settings_json: settings.to_string(), mappings_json: mappings.to_string(), aliases_json: aliases.to_string(), ..Default::default() }
    }

    #[test]
    fn metadata_change_replicates_mapping_settings_and_read_only_aliases() {
        let l = leader(
            json!({"index.number_of_shards": "1", "index.number_of_replicas": "1", "index.analysis.analyzer.a.type": "standard", "index.uuid": "x"}),
            json!({"properties": {"f": {"type": "keyword"}, "g": {"type": "long"}}}),
            json!({"logs": {"is_write_index": true}}),
        );
        let fset = serde_json::from_value(json!({"index.number_of_shards": "1", "index.number_of_replicas": "0", "index.uuid": "y"})).unwrap();
        let c = metadata_change(&l, &fset, &json!({"properties": {"f": {"type": "keyword"}}}), &BTreeMap::new(), true);
        assert_eq!(c.mapping.unwrap()["properties"]["g"]["type"], "long");
        // Replicas and uuids are not replicated; analysis is (and needs a reopen).
        assert!(!c.settings.contains_key("index.number_of_replicas"));
        assert!(!c.settings.contains_key("index.uuid"));
        assert!(c.settings.contains_key("index.analysis.analyzer.a.type"));
        assert!(c.reopen);
        assert_eq!(c.aliases.unwrap()["logs"]["is_write_index"], false);
    }

    #[test]
    fn identical_metadata_changes_nothing() {
        let l = leader(json!({"index.number_of_shards": "1"}), json!({"properties": {}}), json!({}));
        let fset = serde_json::from_value(json!({"index.number_of_shards": "1"})).unwrap();
        let c = metadata_change(&l, &fset, &json!({"properties": {}}), &BTreeMap::new(), true);
        assert!(c.mapping.is_none() && c.settings.is_empty() && c.aliases.is_none());
    }
}
