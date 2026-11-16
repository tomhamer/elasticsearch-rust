//! The auto-follow coordinator (ES `AutoFollowCoordinator`), run by the
//! elected master: about every second (and at once when patterns change)
//! it reads each remote cluster's indices that active patterns name and
//! follows every new open leader index matching a pattern (and none of its
//! exclusion patterns; system indices never) whose primaries are all
//! active, as `PUT {follower}/_ccr/follow` would with the pattern's
//! settings and parameters, as the user who put the pattern. It also drops
//! the CCR metadata of follower indices that were deleted
//! (`ShardFollowTaskCleaner`).

use super::leader::{self, MetadataResp};
use super::state::{self, CcrState};
use crate::node::ClusterNode;
use elasticsearch_engine::ccr::{follower_index_name, PutFollowRequest, AUTO_FOLLOW_PATTERN_REPLACEMENT};
use elasticsearch_engine::error::EsError;
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

#[derive(Debug, Default)]
struct ClusterStats {
    last_check: Option<Instant>,
    metadata_version: i64,
    last_seen: Option<u64>,
}

#[derive(Debug, Default)]
struct AutoState {
    failed_indices: u64,
    failed_remote: u64,
    successful: u64,
    /// `pattern` or `pattern:index` → (timestamp, exception)
    errors: BTreeMap<String, (i64, EsError)>,
    clusters: BTreeMap<String, ClusterStats>,
}

fn states() -> &'static Mutex<BTreeMap<String, Arc<Mutex<AutoState>>>> {
    static S: OnceLock<Mutex<BTreeMap<String, Arc<Mutex<AutoState>>>>> = OnceLock::new();
    S.get_or_init(Default::default)
}

fn auto(node: &ClusterNode) -> Arc<Mutex<AutoState>> {
    states().lock().entry(node.local.id.clone()).or_default().clone()
}

fn wakers() -> &'static Mutex<BTreeMap<String, Arc<tokio::sync::Notify>>> {
    static W: OnceLock<Mutex<BTreeMap<String, Arc<tokio::sync::Notify>>>> = OnceLock::new();
    W.get_or_init(Default::default)
}

fn waker(node: &ClusterNode) -> Arc<tokio::sync::Notify> {
    wakers().lock().entry(node.local.id.clone()).or_default().clone()
}

/// Run the coordinator now (patterns changed).
pub fn wake(node: &ClusterNode) {
    waker(node).notify_one();
}

/// `AutoFollowStats.toXContentFragment`.
pub fn stats_json(node: &ClusterNode) -> Value {
    let a = auto(node);
    let a = a.lock();
    let errors: Vec<Value> = a
        .errors
        .iter()
        .map(|(k, (ts, e))| json!({"leader_index": k, "timestamp": ts, "auto_follow_exception": elasticsearch_engine::ilm::error_json(e)}))
        .collect();
    let clusters: Vec<Value> = a
        .clusters
        .iter()
        .map(|(name, c)| json!({"cluster_name": name, "time_since_last_check_millis": c.last_check.map(|t| t.elapsed().as_millis() as i64).unwrap_or(-1), "last_seen_metadata_version": c.metadata_version}))
        .collect();
    json!({
        "number_of_failed_follow_indices": a.failed_indices,
        "number_of_failed_remote_cluster_state_requests": a.failed_remote,
        "number_of_successful_follow_indices": a.successful,
        "recent_auto_follow_errors": errors,
        "auto_followed_clusters": clusters,
    })
}

/// Drop the CCR metadata of follower indices that no longer exist.
async fn cleanup(node: &Arc<ClusterNode>) {
    let st = node.state();
    let ccr = CcrState::of(&st);
    let gone: Vec<String> = ccr
        .followers
        .iter()
        .filter(|(n, f)| !f.uuid.is_empty() && st.metadata.indices.get(*n).is_none_or(|im| im.uuid != f.uuid))
        .map(|(n, _)| n.clone())
        .collect();
    if gone.is_empty() {
        return;
    }
    let _ = state::update(node, move |s| {
        let mut changed = false;
        for n in &gone {
            changed |= s.followers.remove(n).is_some();
        }
        Ok(changed)
    })
    .await;
}

async fn run_once(node: &Arc<ClusterNode>) {
    cleanup(node).await;
    let st = node.state();
    let ccr = CcrState::of(&st);
    let mut by_cluster: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, p) in &ccr.patterns {
        if p.active {
            by_cluster.entry(p.remote_cluster.clone()).or_default().push(name.clone());
        }
    }
    {
        let a = auto(node);
        let mut a = a.lock();
        a.errors.retain(|k, _| ccr.patterns.contains_key(k.split(':').next().unwrap_or("")));
        a.clusters.retain(|c, _| by_cluster.contains_key(c));
    }
    for (alias, names) in by_cluster {
        let identity = names.iter().find_map(|n| ccr.pattern_identity.get(n).cloned().flatten());
        let meta: MetadataResp = match leader::fetch_metadata(node, &alias, identity, None, false).await {
            Ok(m) => m,
            Err(e) => {
                if e.kind == "no_such_remote_cluster_exception" {
                    continue;
                }
                let a = auto(node);
                let mut a = a.lock();
                let ts = elasticsearch_engine::date::now_millis();
                for p in &names {
                    a.failed_remote += 1;
                    a.errors.insert(p.clone(), (ts, e.clone()));
                }
                a.clusters.entry(alias.clone()).or_default().last_check = Some(Instant::now());
                continue;
            }
        };
        {
            let a = auto(node);
            let mut a = a.lock();
            let c = a.clusters.entry(alias.clone()).or_default();
            c.last_check = Some(Instant::now());
            if c.last_seen != Some(meta.metadata_version) {
                c.last_seen = Some(meta.metadata_version);
                c.metadata_version = meta.metadata_version as i64;
            }
            for p in &names {
                a.errors.remove(p);
            }
        }
        for pname in &names {
            let ccr = CcrState::of(&node.state());
            let Some(pattern) = ccr.patterns.get(pname).cloned().filter(|p| p.active) else { continue };
            let identity = ccr.pattern_identity.get(pname).cloned().flatten();
            let followed = ccr.followed.get(pname).cloned().unwrap_or_default();
            let mut newly: Vec<String> = Vec::new();
            for l in &meta.indices {
                if !l.open || !l.primaries_active || followed.contains(&l.uuid) || !pattern.matches(&l.name, l.data_stream.as_deref()) {
                    continue;
                }
                let fname = follower_index_name(&pattern, &l.name);
                // Already followed by a follower of this leader index.
                if ccr.followers.get(&fname).is_some_and(|f| f.meta.leader_uuid == l.uuid) {
                    newly.push(l.uuid.clone());
                    continue;
                }
                let ds_name = match (&l.data_stream, &pattern.follow_index_pattern) {
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
                match super::rest::put_follow(node, req, identity.clone()).await {
                    Ok(_) => {
                        let a = auto(node);
                        let mut a = a.lock();
                        a.successful += 1;
                        a.errors.remove(&key);
                        newly.push(l.uuid.clone());
                    }
                    Err(e) => {
                        tracing::warn!(pattern = %pname, leader = %l.name, "auto-follow failed: {e}");
                        let a = auto(node);
                        let mut a = a.lock();
                        a.failed_indices += 1;
                        a.errors.insert(key, (elasticsearch_engine::date::now_millis(), e));
                    }
                }
            }
            // Record the newly followed leader indices; forget leader indices
            // that are gone (`cleanFollowedRemoteIndices`).
            let live: Vec<String> = meta.indices.iter().map(|l| l.uuid.clone()).collect();
            let pname = pname.clone();
            let _ = state::update(node, move |s| {
                if !s.patterns.contains_key(&pname) {
                    return Ok(false);
                }
                let list = s.followed.entry(pname).or_default();
                let before = list.clone();
                list.extend(newly);
                list.retain(|u| live.contains(u));
                list.sort();
                list.dedup();
                Ok(*list != before)
            })
            .await;
        }
    }
}

pub fn register(node: &Arc<ClusterNode>) {
    let weak = Arc::downgrade(node);
    let wake = waker(node);
    tokio::spawn(async move {
        loop {
            {
                let Some(node) = weak.upgrade() else { return };
                if node.transport.is_closed() {
                    return;
                }
                if node.is_master() {
                    run_once(&node).await;
                }
            }
            tokio::select! {
                _ = wake.notified() => {}
                _ = tokio::time::sleep(Duration::from_secs(1)) => {}
            }
        }
    });
}
