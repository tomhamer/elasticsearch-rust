//! Shard follow tasks (ES `ShardFollowNodeTask`) and the auto-follow
//! coordinator, driven by one `es-ccr` thread per node that holds only a
//! `Weak<Node>` and exits when the node goes away or nothing is followed.

use super::*;
use crate::hooks::changes::{ChangeOp, Durability};
use crate::index::IndexShard;
use std::collections::BTreeSet;
use std::sync::Weak;
use std::time::{Duration, Instant};

/// `index.ccr.retention_lease.renew_interval` (node setting, tests only).
fn renew_interval(node: &Node) -> Duration {
    node.node_settings
        .read()
        .get("index.ccr.retention_lease.renew_interval")
        .and_then(Value::as_str)
        .and_then(crate::date::parse_time_value)
        .map(|ms| Duration::from_millis(ms.max(0) as u64))
        .unwrap_or(Duration::from_secs(30))
}

/// Runtime state of one follower shard (`ShardFollowNodeTaskStatus` inputs).
#[derive(Debug)]
pub(crate) struct ShardTask {
    pub shard: u32,
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
    /// Start of the current (long-)poll for new operations.
    pub poll_started: Instant,
    pub fatal: Option<EsError>,
    pub retries: u32,
    pub next_attempt: Instant,
    pub last_lease_renewal: Option<Instant>,
    /// Where the previous read stopped in the leader shard's translog.
    pub cursor: Option<crate::shard::ChangesCursor>,
}

impl ShardTask {
    fn new(shard: u32, fsh: Option<&IndexShard>) -> ShardTask {
        let (cp, max) = fsh.map(|s| (s.shard.local_checkpoint(), s.shard.max_seq_no())).unwrap_or((-1, -1));
        let now = Instant::now();
        ShardTask {
            shard,
            leader_global_checkpoint: cp,
            leader_max_seq_no: max,
            follower_global_checkpoint: cp,
            follower_max_seq_no: max,
            last_requested_seq_no: cp,
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
            last_read: Some(now),
            poll_started: now,
            fatal: None,
            retries: 0,
            next_attempt: now,
            last_lease_renewal: None,
            cursor: None,
        }
    }
}

/// The leader metadata last replicated onto a follower index.
#[derive(Debug)]
pub(crate) struct MetaSync {
    leader_schema_gen: u64,
    last_check: Option<Instant>,
    pub mapping_version: i64,
    pub settings_version: i64,
    pub aliases_version: i64,
    /// A change the cluster is committing (its hook answered pending).
    pending: bool,
    /// That change includes the leader's mapping: the follow task writes
    /// no operation until it is applied (they may use its fields).
    pending_mapping: bool,
    /// A failure of the periodic sync, for the follow task to report.
    last_error: Option<EsError>,
    /// The version bumps of that change, counted once it is applied.
    pending_bumps: (i64, i64, i64),
}

impl Default for MetaSync {
    fn default() -> Self {
        MetaSync { leader_schema_gen: u64::MAX, last_check: None, mapping_version: 1, settings_version: 1, aliases_version: 1, pending: false, pending_mapping: false, last_error: None, pending_bumps: (0, 0, 0) }
    }
}

#[derive(Debug, Default)]
struct AutoCluster {
    last_check: Option<Instant>,
    metadata_version: i64,
    last_fp: Option<u64>,
    last_patterns: Vec<String>,
    retry_at: Option<Instant>,
}

#[derive(Debug, Default)]
struct AutoState {
    failed_indices: u64,
    failed_remote: u64,
    successful: u64,
    /// `pattern` or `pattern:index` → (timestamp, exception)
    errors: BTreeMap<String, (i64, EsError)>,
    clusters: BTreeMap<String, AutoCluster>,
    dirty: bool,
}

/// In-memory CCR runtime of a node.
#[derive(Default)]
pub(crate) struct Runtime {
    shards: Mutex<BTreeMap<(String, u32), Arc<Mutex<ShardTask>>>>,
    meta: Mutex<HashMap<String, MetaSync>>,
    auto: Mutex<AutoState>,
}

impl Runtime {
    /// Forget the runtime state of a follower index's shard tasks.
    pub(crate) fn drop_index(&self, uuid: &str) {
        self.shards.lock().retain(|(u, _), _| u != uuid);
        self.meta.lock().remove(uuid);
    }

    pub(crate) fn patterns_changed(&self) {
        self.auto.lock().dirty = true;
    }

    fn task(&self, uuid: &str, n: u32, fsh: Option<&IndexShard>) -> Arc<Mutex<ShardTask>> {
        self.shards.lock().entry((uuid.to_string(), n)).or_insert_with(|| Arc::new(Mutex::new(ShardTask::new(n, fsh)))).clone()
    }

    /// `AutoFollowStats.toXContentFragment`.
    pub(crate) fn auto_stats_json(&self) -> Value {
        let a = self.auto.lock();
        let errors: Vec<Value> = a
            .errors
            .iter()
            .map(|(k, (ts, e))| json!({"leader_index": k, "timestamp": ts, "auto_follow_exception": crate::ilm::error_json(e)}))
            .collect();
        let clusters: Vec<Value> = a
            .clusters
            .iter()
            .map(|(name, c)| {
                json!({
                    "cluster_name": name,
                    "time_since_last_check_millis": c.last_check.map(|t| t.elapsed().as_millis() as i64).unwrap_or(-1),
                    "last_seen_metadata_version": c.metadata_version,
                })
            })
            .collect();
        json!({
            "number_of_failed_follow_indices": a.failed_indices,
            "number_of_failed_remote_cluster_state_requests": a.failed_remote,
            "number_of_successful_follow_indices": a.successful,
            "recent_auto_follow_errors": errors,
            "auto_followed_clusters": clusters,
        })
    }
}

// ---------------------------------------------------------------------------
// Status JSON
// ---------------------------------------------------------------------------

/// `ShardFollowNodeTaskStatus.toXContentFragment`.
pub fn shard_status_json(remote: &str, leader: &str, follower: &str, t: &ShardTaskView) -> Value {
    let mut m = Map::new();
    m.insert("remote_cluster".into(), json!(remote));
    m.insert("leader_index".into(), json!(leader));
    m.insert("follower_index".into(), json!(follower));
    m.insert("shard_id".into(), json!(t.shard));
    m.insert("leader_global_checkpoint".into(), json!(t.leader_global_checkpoint));
    m.insert("leader_max_seq_no".into(), json!(t.leader_max_seq_no));
    m.insert("follower_global_checkpoint".into(), json!(t.follower_global_checkpoint));
    m.insert("follower_max_seq_no".into(), json!(t.follower_max_seq_no));
    m.insert("last_requested_seq_no".into(), json!(t.last_requested_seq_no));
    m.insert("outstanding_read_requests".into(), json!(t.outstanding_read_requests));
    m.insert("outstanding_write_requests".into(), json!(0));
    m.insert("write_buffer_operation_count".into(), json!(0));
    m.insert("write_buffer_size_in_bytes".into(), json!(0));
    m.insert("follower_mapping_version".into(), json!(t.mapping_version));
    m.insert("follower_settings_version".into(), json!(t.settings_version));
    m.insert("follower_aliases_version".into(), json!(t.aliases_version));
    m.insert("total_read_time_millis".into(), json!(t.total_read_time_millis));
    m.insert("total_read_remote_exec_time_millis".into(), json!(t.total_read_remote_exec_time_millis));
    m.insert("successful_read_requests".into(), json!(t.successful_read_requests));
    m.insert("failed_read_requests".into(), json!(t.failed_read_requests));
    m.insert("operations_read".into(), json!(t.operations_read));
    m.insert("bytes_read".into(), json!(t.bytes_read));
    m.insert("total_write_time_millis".into(), json!(t.total_write_time_millis));
    m.insert("successful_write_requests".into(), json!(t.successful_write_requests));
    m.insert("failed_write_requests".into(), json!(t.failed_write_requests));
    m.insert("operations_written".into(), json!(t.operations_written));
    m.insert(
        "read_exceptions".into(),
        Value::Array(t.read_exceptions.iter().map(|(from, (n, e))| json!({"from_seq_no": from, "retries": n, "exception": crate::ilm::error_json(e)})).collect()),
    );
    m.insert("time_since_last_read_millis".into(), json!(t.time_since_last_read_millis));
    if let Some(e) = &t.fatal {
        m.insert("fatal_exception".into(), crate::ilm::error_json(e));
    }
    Value::Object(m)
}

/// A point-in-time copy of a shard task's status.
pub struct ShardTaskView {
    pub shard: u32,
    pub leader_global_checkpoint: i64,
    pub leader_max_seq_no: i64,
    pub follower_global_checkpoint: i64,
    pub follower_max_seq_no: i64,
    pub last_requested_seq_no: i64,
    pub outstanding_read_requests: u32,
    pub mapping_version: i64,
    pub settings_version: i64,
    pub aliases_version: i64,
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
    pub read_exceptions: Vec<(i64, (u32, EsError))>,
    pub time_since_last_read_millis: i64,
    pub fatal: Option<EsError>,
}

fn view(t: &ShardTask, ms: Option<&MetaSync>) -> ShardTaskView {
    let failing = t.fatal.is_some() || !t.read_exceptions.is_empty();
    ShardTaskView {
        shard: t.shard,
        leader_global_checkpoint: t.leader_global_checkpoint,
        leader_max_seq_no: t.leader_max_seq_no,
        follower_global_checkpoint: t.follower_global_checkpoint,
        follower_max_seq_no: t.follower_max_seq_no,
        last_requested_seq_no: t.last_requested_seq_no,
        // A read for new operations is always outstanding (long poll) unless
        // the task failed or is backing off.
        outstanding_read_requests: if failing { 0 } else { 1 },
        mapping_version: ms.map_or(1, |m| m.mapping_version),
        settings_version: ms.map_or(1, |m| m.settings_version),
        aliases_version: ms.map_or(1, |m| m.aliases_version),
        total_read_time_millis: t.total_read_time_millis,
        total_read_remote_exec_time_millis: t.total_read_remote_exec_time_millis,
        successful_read_requests: t.successful_read_requests,
        failed_read_requests: t.failed_read_requests,
        operations_read: t.operations_read,
        bytes_read: t.bytes_read,
        total_write_time_millis: t.total_write_time_millis,
        successful_write_requests: t.successful_write_requests,
        failed_write_requests: t.failed_write_requests,
        operations_written: t.operations_written,
        read_exceptions: t.read_exceptions.iter().map(|(k, v)| (*k, v.clone())).collect(),
        time_since_last_read_millis: t.last_read.map(|i| i.elapsed().as_millis() as i64).unwrap_or(-1),
        fatal: t.fatal.clone(),
    }
}

/// `FollowStatsAction.StatsResponses` for the given follower indices
/// (`(name, uuid)`, sorted by name).
pub fn follow_stats_json(node: &Node, svc: &CcrService, list: &[(String, String)]) -> Value {
    let mut indices = Vec::new();
    for (name, uuid) in list {
        let Some(meta) = svc.st.lock().followers.get(uuid).cloned() else { continue };
        let Some(fsvc) = find_by_uuid(node, name, uuid) else { continue };
        let mut shards = Vec::new();
        let mut lag = 0i64;
        for n in 0..fsvc.number_of_shards() {
            let fsh = fsvc.shard(n);
            let t = svc.rt.task(uuid, n, fsh.as_deref());
            let v = {
                let t = t.lock();
                let ms = svc.rt.meta.lock();
                view(&t, ms.get(uuid))
            };
            lag += v.leader_global_checkpoint - v.follower_global_checkpoint;
            shards.push(shard_status_json(&meta.remote_cluster, &meta.leader_index, name, &v));
        }
        indices.push(json!({"index": name, "total_global_checkpoint_lag": lag, "shards": shards}));
    }
    json!({"indices": indices})
}

// ---------------------------------------------------------------------------
// The worker
// ---------------------------------------------------------------------------

pub(crate) fn worker_loop(weak: Weak<Node>, key: (PathBuf, usize), svc: Arc<CcrService>) {
    let mut idle_ticks: u32 = 0;
    loop {
        let wait = if idle_ticks == 0 { Duration::from_millis(1) } else { Duration::from_millis((5u64 << idle_ticks.min(3)).min(40)) };
        std::thread::sleep(wait);
        let Some(node) = weak.upgrade() else {
            svc.worker_stopped();
            let mut r = registry().lock();
            if r.get(&key).is_some_and(|s| Arc::ptr_eq(s, &svc)) {
                r.remove(&key);
            }
            break;
        };
        if !svc.st.lock().has_work() && svc.worker_idle_exit() {
            break;
        }
        let busy = run_once(&node, &svc);
        drop(node);
        idle_ticks = if busy { 0 } else { idle_ticks.saturating_add(1) };
    }
}

fn run_once(node: &Arc<Node>, svc: &Arc<CcrService>) -> bool {
    cleanup_orphans(node, svc);
    let tasks: Vec<(String, FollowParams, FollowerMeta)> = {
        let st = svc.st.lock();
        st.tasks.iter().filter_map(|(u, p)| st.followers.get(u).map(|m| (u.clone(), p.clone(), m.clone()))).collect()
    };
    let mut busy = false;
    for (uuid, params, meta) in tasks {
        let Some(fsvc) = find_by_uuid(node, &meta.follower_index, &uuid) else { continue };
        for n in 0..fsvc.number_of_shards() {
            let fsh = fsvc.shard(n);
            let t = svc.rt.task(&uuid, n, fsh.as_deref());
            busy |= step(node, svc, &fsvc, &uuid, &meta, &params, n, &t);
        }
    }
    busy |= run_auto_follow(node, svc);
    busy
}

/// Drop the CCR state of follower indices that no longer exist
/// (`ShardFollowTaskCleaner`).
fn cleanup_orphans(node: &Node, svc: &CcrService) {
    let gone: Vec<String> = {
        let st = svc.st.lock();
        st.followers.iter().filter(|(u, m)| find_by_uuid(node, &m.follower_index, u).is_none()).map(|(u, _)| u.clone()).collect()
    };
    if gone.is_empty() {
        return;
    }
    let mut st = svc.st.lock();
    for u in &gone {
        // Re-check under the lock (the index may just have been registered).
        if let Some(m) = st.followers.get(u) {
            if find_by_uuid(node, &m.follower_index, u).is_some() {
                continue;
            }
        }
        st.followers.remove(u);
        st.tasks.remove(u);
        svc.rt.drop_index(u);
    }
    let _ = svc.persist(&st);
}

/// `ShardFollowNodeTask.shouldRetry`.
fn should_retry(e: &EsError) -> bool {
    matches!(
        e.kind.as_str(),
        "shard_not_found_exception"
            | "illegal_index_shard_state_exception"
            | "index_shard_closed_exception"
            | "index_shard_not_started_exception"
            | "index_shard_not_recovering_exception"
            | "no_shard_available_action_exception"
            | "unavailable_shards_exception"
            | "already_closed_exception"
            | "security_exception"
            | "cluster_block_exception"
            | "index_closed_exception"
            | "connect_transport_exception"
            | "node_disconnected_exception"
            | "receive_timeout_transport_exception"
            | "transport_exception"
            | "node_closed_exception"
            | "no_such_remote_cluster_exception"
            | "no_seed_node_left_exception"
            | "es_rejected_execution_exception"
            | "circuit_breaking_exception"
            // A cluster's commit of the follower's metadata that did not go
            // through this time (a stale base, a failed or timed-out
            // publication, no master yet): retried with backoff.
            | "failed_to_commit_cluster_state_exception"
            | "process_cluster_event_timeout_exception"
            | "master_not_discovered_exception"
            | "not_master_exception"
            | "node_not_connected_exception"
    )
}

/// `ShardFollowNodeTask.computeDelay` (without the jitter).
fn retry_delay(retry: u32, max_ms: i64) -> Duration {
    let n = 1u64 << retry.saturating_sub(1).min(24);
    Duration::from_millis((50 * n).min(max_ms.max(1) as u64))
}

fn read_failure(t: &mut ShardTask, from: i64, e: EsError, params: &FollowParams) {
    t.failed_read_requests += 1;
    if should_retry(&e) {
        t.retries += 1;
        let entry = t.read_exceptions.entry(from).or_insert((0, e.clone()));
        entry.0 = t.retries;
        entry.1 = e;
        t.next_attempt = Instant::now() + retry_delay(t.retries, params.retry_delay_ms());
    } else {
        t.fatal = Some(e);
    }
}

fn shard_of(svc: &IndexService, n: u32) -> Result<Arc<IndexShard>> {
    svc.shard(n).ok_or_else(|| EsError::shard_not_found(&svc.name, &svc.uuid, n))
}

/// One round of a shard follow task: read the next operations from the
/// leader shard and apply them to the follower shard. True when ops moved.
#[allow(clippy::too_many_arguments)]
fn step(node: &Arc<Node>, svc: &CcrService, fsvc: &Arc<IndexService>, uuid: &str, meta: &FollowerMeta, params: &FollowParams, n: u32, task: &Arc<Mutex<ShardTask>>) -> bool {
    let now = Instant::now();
    {
        let t = task.lock();
        if t.fatal.is_some() || now < t.next_attempt {
            return false;
        }
    }
    // A closed follower is not read into (ES fails its writes and retries).
    if !fsvc.is_open() {
        return false;
    }
    let Some(fsh) = fsvc.shard(n) else { return false };
    let from = fsh.shard.local_checkpoint() + 1;
    let fail = |e: EsError| {
        read_failure(&mut task.lock(), from, e, params);
        false
    };
    let leader = match svc.connect(node, &meta.remote_cluster) {
        Ok(l) => l,
        Err(e) => return fail(e),
    };
    let lsvc = match leader.index(&meta.leader_index).filter(|s| s.uuid == meta.leader_uuid) {
        Some(s) => s,
        None => return fail(EsError::index_not_found(&meta.leader_index)),
    };
    if !lsvc.is_open() {
        return fail(EsError::new(400, "cluster_block_exception", format!("index [{}] blocked by: [FORBIDDEN/4/index closed];", lsvc.name)));
    }
    let lsh = match shard_of(&lsvc, n) {
        Ok(s) => s,
        Err(e) => return fail(e),
    };
    let leader_gcp = crate::fleet::shard_checkpoint(&lsvc, &lsh);
    let leader_max = lsh.shard.max_seq_no();
    // Retention lease renewal (`CcrRetentionLeases`, every 30s).
    let renew_due = task.lock().last_lease_renewal.is_none_or(|t| t.elapsed() >= renew_interval(node));
    if renew_due {
        let id = retention_lease_id(&node.cluster_name, &fsvc.name, uuid, &meta.remote_cluster, &lsvc.name, &lsvc.uuid);
        let retain = crate::fleet::shard_checkpoint(fsvc, &fsh) + 1;
        let _ = lsh.shard.renew_retention_lease(&id, retain, "ccr");
        task.lock().last_lease_renewal = Some(Instant::now());
    }
    // Periodic settings / aliases / mapping replication; a failure it met
    // (or one recorded by an earlier sync) is the task's read failure.
    let _ = sync_metadata(svc, &lsvc, fsvc, uuid, false);
    let sync_error = svc.rt.meta.lock().get_mut(uuid).and_then(|m| m.last_error.take());
    if let Some(e) = sync_error {
        return fail(e);
    }
    {
        let mut t = task.lock();
        t.leader_global_checkpoint = leader_gcp;
        t.leader_max_seq_no = leader_max;
        t.follower_global_checkpoint = crate::fleet::shard_checkpoint(fsvc, &fsh);
        t.follower_max_seq_no = fsh.shard.max_seq_no();
    }
    // A pending mapping cannot accept these operations yet. Keep renewing
    // leases and polling metadata above, but do not reread the same translog
    // range (and inflate read counters) on every retry while it commits.
    if svc.rt.meta.lock().get(uuid).is_some_and(|m| m.pending_mapping) {
        task.lock().next_attempt = Instant::now() + Duration::from_millis(100);
        return false;
    }
    if from > leader_gcp {
        // Nothing to read: a long poll completes empty after `read_poll_timeout`.
        let mut t = task.lock();
        if t.poll_started.elapsed() >= Duration::from_millis(params.poll_timeout_ms() as u64) {
            t.successful_read_requests += 1;
            t.last_read = Some(Instant::now());
            t.poll_started = Instant::now();
            t.retries = 0;
            t.read_exceptions.clear();
        }
        return false;
    }
    let to = leader_gcp.min(from.saturating_add(params.read_ops() as i64 - 1));
    let t0 = Instant::now();
    let mut cursor = task.lock().cursor.take();
    let read = lsh.shard.changes_from_cursor_sized(&mut cursor, from, to, params.read_ops(), params.read_bytes(), op_size);
    task.lock().cursor = cursor;
    let snap = match read {
        Ok(s) => s,
        Err(e) => {
            let e = if e.kind == "missing_history_operations_exception" {
                EsError::resource_not_found(format!(
                    "Operations are no longer available for replicating. Maybe increase the retention setting [index.soft_deletes.retention.operations]? [{}]",
                    e.reason
                ))
            } else {
                e
            };
            return fail(e);
        }
    };
    let follower_term = fsh.shard.primary_term();
    let count=snap.len();
    let capacity_bytes=(count as u64).saturating_mul((3*std::mem::size_of::<ChangeOp>()+std::mem::size_of::<crate::memory::Reservation>()+8) as u64).saturating_add(1024);
    let _slots=match crate::memory::breakers().reserve_request(capacity_bytes,"change feed follower staging") {Ok(g)=>g,Err(e)=>return fail(e)};
    let mut copies=Vec::with_capacity(count);
    let mut ops: Vec<ChangeOp> = Vec::with_capacity(count);
    let mut sizes: Vec<u64> = Vec::with_capacity(count);
    let mut bytes = 0u64;
    for op in snap {
        if op.seq_no() > to {
            break;
        }
        let size = op_size(&op);
        bytes += size;
        sizes.push(size);
        // The follower primary stamps its own term (TransportBulkShardOperationsAction).
        let (value,memory)=match op.try_copy(){Ok(v)=>v,Err(e)=>return fail(e)};
        copies.push(memory);
        ops.push(retermed(value, follower_term));
    }
    let read_ms = t0.elapsed().as_millis() as u64;
    let last = ops.last().map(|o| o.seq_no()).unwrap_or(from - 1);
    {
        let mut t = task.lock();
        t.successful_read_requests += 1;
        t.total_read_time_millis += read_ms;
        t.total_read_remote_exec_time_millis += read_ms;
        t.operations_read += ops.len() as u64;
        t.bytes_read += bytes;
        t.last_read = Some(Instant::now());
        t.poll_started = Instant::now();
        t.last_requested_seq_no = t.last_requested_seq_no.max(to);
        t.read_exceptions.remove(&from);
        t.retries = 0;
    }
    if ops.is_empty() {
        return false;
    }
    // The leader's mapping first: the ops may use fields it added.
    if lsvc.schema_generation() != svc.rt.meta.lock().get(uuid).map_or(u64::MAX, |m| m.leader_schema_gen) {
        if let Err(e) = sync_metadata(svc, &lsvc, fsvc, uuid, true) {
            // Reported here; not again by the next periodic sync.
            if let Some(m) = svc.rt.meta.lock().get_mut(uuid) {
                m.last_error = None;
            }
            return fail(e);
        }
    }
    // Still being committed: no operation is written before it (they are
    // read again, from the follower's checkpoint, at the next poll).
    if svc.rt.meta.lock().get(uuid).is_some_and(|m| m.pending_mapping) {
        task.lock().next_attempt = Instant::now() + Duration::from_millis(100);
        return false;
    }
    let durability = if fsvc.setting("index.translog.durability").as_deref() == Some("async") { Durability::Async } else { Durability::Request };
    let (max_ops, max_wbytes) = (params.write_ops(), params.write_bytes());
    let mut batch: Vec<ChangeOp> = Vec::with_capacity(count);
    let mut batch_bytes = 0u64;
    let mut iter = ops.into_iter().zip(sizes).peekable();
    while let Some((op, size)) = iter.next() {
        batch_bytes += size;
        batch.push(op);
        let full = batch.len() >= max_ops || batch_bytes >= max_wbytes || iter.peek().is_none();
        if !full {
            continue;
        }
        let w0 = Instant::now();
        let n_ops = batch.len() as u64;
        let r = fsh.apply_replica_ops(follower_term, std::mem::replace(&mut batch, Vec::with_capacity(count)), durability);
        batch_bytes = 0;
        let mut t = task.lock();
        t.total_write_time_millis += w0.elapsed().as_millis() as u64;
        match r {
            Ok(res) => {
                t.successful_write_requests += 1;
                t.operations_written += n_ops;
                t.follower_max_seq_no = res.max_seq_no;
            }
            Err(e) => {
                t.failed_write_requests += 1;
                drop(t);
                return fail(e);
            }
        }
    }
    let mut t = task.lock();
    t.follower_global_checkpoint = crate::fleet::shard_checkpoint(fsvc, &fsh);
    t.follower_max_seq_no = fsh.shard.max_seq_no().max(last);
    true
}

/// `Translog.Operation.estimateSize` (ids / reasons in UTF-16 code units).
fn op_size(op: &ChangeOp) -> u64 {
    let u16len = |s: &str| s.encode_utf16().count();
    (match op {
        ChangeOp::Index { id, source_json, routing, .. } => 2 * u16len(id) + source_json.len() + routing.as_deref().map_or(0, |r| 2 * u16len(r)) + 32,
        ChangeOp::Delete { id, .. } => 2 * u16len(id) + 24,
        ChangeOp::NoOp { reason, .. } => 2 * u16len(reason) + 16,
    }) as u64
}

fn retermed(op: ChangeOp, term: u64) -> ChangeOp {
    match op {
        ChangeOp::Index { id, source_json, routing, seq_no, version, .. } => ChangeOp::Index { id, source_json, routing, seq_no, version, primary_term: term },
        ChangeOp::Delete { id, seq_no, version, .. } => ChangeOp::Delete { id, seq_no, version, primary_term: term },
        ChangeOp::NoOp { seq_no, reason, .. } => ChangeOp::NoOp { seq_no, primary_term: term, reason },
    }
}

/// The first metadata replication onto a new follower (aliases arrive
/// here: the restore does not copy them).
pub(crate) fn initial_sync(svc: &CcrService, lsvc: &IndexService, fsvc: &IndexService) {
    if let Err(e) = sync_metadata(svc, lsvc, fsvc, &fsvc.uuid, true) {
        tracing::warn!("initial metadata sync of follower [{}] failed: {e}", fsvc.name);
    }
}

/// Replicate the leader's mapping, replicated settings and aliases onto
/// the follower (`ShardFollowTasksExecutor.innerUpdateMapping /
/// innerUpdateSettings / innerUpdateAliases`). Without `force`, at most
/// once a second.
fn sync_metadata(svc: &CcrService, lsvc: &IndexService, fsvc: &IndexService, uuid: &str, force: bool) -> Result<()> {
    {
        let ms = svc.rt.meta.lock();
        if let Some(m) = ms.get(uuid) {
            if !force && m.last_check.is_some_and(|t| t.elapsed() < Duration::from_secs(1)) {
                return Ok(());
            }
        }
    }
    let schema_gen = lsvc.schema_generation();
    let lmeta = lsvc.meta();
    let fmeta = fsvc.meta();
    let mut change = FollowerMetadataChange::default();
    // Mapping.
    let lmap = lmeta.mappings.to_json();
    if lmap != fmeta.mappings.to_json() && fsvc.is_open() {
        change.mapping = Some(lmap);
    }
    // Settings that differ (`getUpdatedSettings`): dynamic ones in place,
    // others with the follower closed and reopened.
    let lf = filter_settings(&lmeta.settings);
    let ff = filter_settings(&fmeta.settings);
    change.settings = lf.iter().filter(|(k, v)| ff.get(*k) != Some(*v)).map(|(k, v)| (k.clone(), v.clone())).collect();
    change.reopen = change.settings.keys().any(|k| !crate::settings::is_dynamic(k));
    // Aliases (never write aliases on a follower).
    let laliases: BTreeMap<String, Value> = lmeta
        .aliases
        .iter()
        .map(|(k, v)| {
            let mut v = v.clone();
            if let Some(o) = v.as_object_mut() {
                o.insert("is_write_index".into(), json!(false));
            }
            (k.clone(), v)
        })
        .collect();
    if laliases != fmeta.aliases {
        change.aliases = Some(laliases);
    }
    let (dm, ds, da) = (i64::from(change.mapping.is_some()), i64::from(!change.settings.is_empty()), i64::from(change.aliases.is_some()));
    let first_err = if dm + ds + da == 0 {
        None
    } else {
        // A cluster commits the change through its master first.
        let hook = svc.metadata_hook.read().clone();
        let r = match hook {
            Some(h) => h(&fsvc.name, change),
            None => apply_follower_in_place(fsvc, &change),
        };
        r.err()
    };
    // Being committed by the cluster: not an error (the next sync finds it
    // applied, or sends it again), and not applied yet either.
    let pending = first_err.as_ref().is_some_and(|e| e.kind == FOLLOWER_METADATA_PENDING);
    let first_err = first_err.filter(|_| !pending);
    let mut ms = svc.rt.meta.lock();
    let m = ms.entry(uuid.to_string()).or_default();
    m.last_check = Some(Instant::now());
    if pending {
        m.pending = true;
        m.pending_mapping |= dm > 0;
        m.pending_bumps = (m.pending_bumps.0.max(dm), m.pending_bumps.1.max(ds), m.pending_bumps.2.max(da));
    }
    // A failed commit: nothing is pending any more; the error is kept for
    // the follow task (the periodic sync's caller ignores its result).
    if let Some(e) = &first_err {
        m.pending = false;
        m.pending_mapping = false;
        m.last_error = Some(e.clone());
    }
    // Counted once applied (follow stats' versions): now, or when a change
    // reported pending turns out applied (nothing left to change).
    if first_err.is_none() && !pending {
        m.leader_schema_gen = schema_gen;
        let (pm, ps, pa) = std::mem::take(&mut m.pending_bumps);
        m.mapping_version += dm.max(pm);
        m.settings_version += ds.max(ps);
        m.aliases_version += da.max(pa);
        m.pending = false;
        m.pending_mapping = false;
    }
    match first_err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Auto-follow coordinator
// ---------------------------------------------------------------------------

/// Fingerprint of a leader cluster's index metadata (what ES watches via
/// the remote cluster state's metadata version).
fn leader_fingerprint(leader: &LeaderCluster) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let mut idx: Vec<(String, String, bool)> = leader.indices().iter().map(|s| (s.name.clone(), s.uuid.clone(), s.is_open())).collect();
    idx.sort();
    idx.hash(&mut h);
    match leader {
        LeaderCluster::Local(n) => {
            for (k, d) in n.data_streams.read().iter() {
                k.hash(&mut h);
                d.indices.hash(&mut h);
            }
        }
    }
    h.finish()
}

fn run_auto_follow(node: &Arc<Node>, svc: &Arc<CcrService>) -> bool {
    let (by_cluster, all_names): (BTreeMap<String, Vec<String>>, BTreeSet<String>) = {
        let st = svc.st.lock();
        let mut m: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (name, p) in &st.patterns {
            if p.active {
                m.entry(p.remote_cluster.clone()).or_default().push(name.clone());
            }
        }
        (m, st.patterns.keys().cloned().collect())
    };
    let dirty = {
        let mut a = svc.rt.auto.lock();
        // Purge the stats of removed patterns and auto-followers without patterns.
        a.errors.retain(|k, _| all_names.contains(k.split(':').next().unwrap_or("")));
        a.clusters.retain(|c, _| by_cluster.contains_key(c));
        std::mem::take(&mut a.dirty)
    };
    let mut busy = false;
    for (alias, names) in by_cluster {
        let now = Instant::now();
        {
            let a = svc.rt.auto.lock();
            if let Some(c) = a.clusters.get(&alias) {
                if c.retry_at.is_some_and(|r| now < r) && !dirty {
                    continue;
                }
            }
        }
        let leader = match svc.connect(node, &alias).and_then(|l| l.check_license().map(|_| l)) {
            Ok(l) => l,
            Err(e) => {
                let mut a = svc.rt.auto.lock();
                if e.kind == "no_such_remote_cluster_exception" {
                    // The auto-follower stops until the connection reappears.
                    let c = a.clusters.entry(alias.clone()).or_default();
                    c.retry_at = Some(now + Duration::from_secs(1));
                    continue;
                }
                let ts = crate::date::now_millis();
                for p in &names {
                    a.failed_remote += 1;
                    a.errors.insert(p.clone(), (ts, e.clone()));
                }
                let c = a.clusters.entry(alias.clone()).or_default();
                c.last_check = Some(now);
                c.retry_at = Some(now + Duration::from_secs(1));
                continue;
            }
        };
        let fp = leader_fingerprint(&leader);
        {
            let mut a = svc.rt.auto.lock();
            let c = a.clusters.entry(alias.clone()).or_default();
            if !dirty && c.last_fp == Some(fp) && c.last_patterns == names && c.retry_at.is_none() {
                continue;
            }
            if c.last_fp != Some(fp) {
                c.metadata_version += 1;
            }
            c.last_fp = Some(fp);
            c.last_patterns = names.clone();
            c.last_check = Some(now);
            c.retry_at = None;
        }
        for pname in &names {
            let Some(pattern) = svc.st.lock().patterns.get(pname).cloned() else { continue };
            if !pattern.active {
                continue;
            }
            svc.rt.auto.lock().errors.remove(pname);
            let followed: Vec<String> = svc.st.lock().followed.get(pname).cloned().unwrap_or_default();
            let leader_indices = leader.indices();
            let mut newly: Vec<String> = Vec::new();
            for l in &leader_indices {
                if !l.is_open() || followed.contains(&l.uuid) || l.active_shards() < l.number_of_shards() as usize {
                    continue;
                }
                let ds = leader.parent_data_stream(&l.name);
                if !pattern.matches(&l.name, ds.as_deref()) {
                    continue;
                }
                let fname = follower_index_name(&pattern, &l.name);
                // Already followed by an existing follower of this leader (`leaderIndexAlreadyFollowed`).
                let already = node.get_index(&fname).is_some_and(|f| svc.st.lock().followers.get(&f.uuid).is_some_and(|m| m.leader_uuid == l.uuid));
                if already {
                    newly.push(l.uuid.clone());
                    continue;
                }
                // `generateRequest`: a renamed data stream follows the pattern too.
                let ds_name = match (&ds, &pattern.follow_index_pattern) {
                    (Some(d), Some(fp)) => Some(fp.replace(AUTO_FOLLOW_PATTERN_REPLACEMENT, d)),
                    _ => None,
                };
                let req = PutFollowRequest {
                    follower_index: fname.clone(),
                    remote_cluster: Some(alias.clone()),
                    leader_index: Some(l.name.clone()),
                    data_stream_name: ds_name,
                    settings: pattern.settings.clone(),
                    params: pattern.params.clone(),
                    wait_for_active_shards: false,
                };
                let key = format!("{pname}:{}", l.name);
                match put_follow(node, &req) {
                    Ok(_) => {
                        let mut a = svc.rt.auto.lock();
                        a.successful += 1;
                        a.errors.remove(&key);
                        newly.push(l.uuid.clone());
                        busy = true;
                    }
                    Err(e) => {
                        let mut a = svc.rt.auto.lock();
                        a.failed_indices += 1;
                        a.errors.insert(key, (crate::date::now_millis(), e));
                    }
                }
            }
            // Record the newly followed leader indices and forget the ones
            // that no longer exist (`cleanFollowedRemoteIndices`).
            let mut st = svc.st.lock();
            if let Some(list) = st.followed.get_mut(pname) {
                list.extend(newly);
                list.retain(|u| leader_indices.iter().any(|l| &l.uuid == u));
                list.dedup();
            } else if st.patterns.contains_key(pname) {
                st.followed.insert(pname.clone(), newly);
            }
            let _ = svc.persist(&st);
        }
        // Following created indices on the leader (self-following): look again.
        if busy {
            svc.rt.auto.lock().dirty = true;
        }
    }
    busy
}

/// [`apply_follower_change`] on the follower service itself.
pub(crate) fn apply_follower_in_place(fsvc: &IndexService, change: &FollowerMetadataChange) -> Result<()> {
    let mut first_err: Option<EsError> = None;
    if let Some(m) = &change.mapping {
        if let Err(e) = fsvc.put_mapping(m) {
            first_err = Some(e);
        }
    }
    if !change.settings.is_empty() {
        if let Err(e) = fsvc.put_settings_reopen(change.settings.clone(), false, change.reopen && fsvc.is_open()) {
            first_err = first_err.or(Some(e));
        }
    }
    if let Some(a) = &change.aliases {
        if let Err(e) = fsvc.set_aliases(a.clone()) {
            first_err = first_err.or(Some(e));
        }
    }
    match first_err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}
