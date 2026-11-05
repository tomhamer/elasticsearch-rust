//! Snapshot / restore in the cluster: `_snapshot/*`, `_slm/*`,
//! `_cat/snapshots`, `_cat/repositories`, built on `elasticsearch-snapshots`.
//!
//! * Repositories are cluster metadata (`metadata.repositories`), written by
//!   master tasks and validated / verified on every node.
//! * The elected master coordinates snapshots ([`ops`]): each primary's node
//!   freezes its shard files (gated flush + hard links) and uploads them to
//!   the repository itself ([`upload`]); the master tracks the shards and
//!   finalizes the snapshot through its `SnapshotsService`.
//! * Restores adopt the snapshot's shard files on each primary's node
//!   (replicas then recover from it by file-based peer recovery), and
//!   re-index the snapshot's documents through the cluster write path only
//!   when the files cannot be adopted ([`restore`], [`file_restore`]).
//! * SLM schedules run on the elected master ([`snapshot::slm`]).
//!
//! See the module docs of each part for the protocol details.

#[path = "snapshot/common.rs"]
pub mod common;
#[path = "snapshot/file_restore.rs"]
pub mod file_restore;
#[path = "snapshot/ops.rs"]
pub mod ops;
#[path = "snapshot/rest.rs"]
mod rest;
#[path = "snapshot/restore.rs"]
pub mod restore;
#[path = "snapshot/shard.rs"]
pub mod shard;
#[path = "snapshot/slm.rs"]
pub mod slm;
#[path = "snapshot/upload.rs"]
pub mod upload;

use crate::front::{RestReq, RestResp};
use crate::node::ClusterNode;
use elasticsearch_cluster::state::ClusterState;
use serde_json::{Map, Value};
use std::sync::Arc;
use std::time::Duration;

pub async fn handle(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    ensure_registered(node);
    rest::handle(node, req).await
}

/// Register the snapshot transport actions and start the master-side
/// background work (resuming orphaned snapshots / restores, SLM schedules). Called once
/// at node start (`lib.rs` `run()`); idempotent.
pub fn register(node: &Arc<ClusterNode>) {
    ensure_registered(node);
}

fn ensure_registered(node: &Arc<ClusterNode>) {
    let st = common::state_of(node);
    if st.background.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return;
    }
    shard::register(node);
    upload::register(node);
    let weak = Arc::downgrade(node);
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(1));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            let Some(node) = weak.upgrade() else { return };
            if !node.is_master() {
                continue;
            }
            ops::resume_orphans(&node).await;
            slm::tick(&node).await;
        }
    });
}

/// Top-level `GET _cluster/state` customs (`snapshots`, `restore`,
/// `snapshot_deletions`) for the integrator's `state_api`.
pub fn render_state_customs(st: &ClusterState) -> Map<String, Value> {
    common::render_state_customs(st)
}

/// The `snapshots` section of `GET _cluster/stats` (ES 8.8+): in-progress
/// counts overall and per registered repository.
pub fn cluster_stats_section(st:&ClusterState,human:bool)->Result<crate::owned_json::OwnedJson,elasticsearch_engine::error::EsError>{
    use crate::front::SharedText;
    use crate::owned_json::OwnedJson;
    use elasticsearch_engine::error::EsError;
    let running=st.metadata.customs.get(common::PROGRESS_CUSTOM).and_then(|v|v["snapshots"].as_array()).map(Vec::as_slice).unwrap_or(&[]);
    let shards_of=|name:Option<&str>|->usize{running.iter().filter(|e|name.is_none()||e["repository"].as_str()==name).map(|e|e["indices"].as_array().map_or(0,Vec::len)).sum()};
    let text=SharedText::try_format(format_args!(r#"{{"current_counts":{{"snapshots":{},"shard_snapshots":{},"snapshot_deletions":0,"concurrent_operations":{},"cleanups":0}},"repositories":{{}}}}"#,running.len(),shards_of(None),running.len()))?;
    let mut output=OwnedJson::parse(&text)?;
    if let Some(repos)=st.metadata.customs.get(common::REPOS_CUSTOM).and_then(Value::as_object){for(name,meta)in repos{
        let count=running.iter().filter(|e|e["repository"].as_str()==Some(name)).count();let shards=shards_of(Some(name));
        let oldest=running.iter().filter(|e|e["repository"].as_str()==Some(name)).filter_map(|e|e["start_time_millis"].as_i64().or_else(||e["start_time_in_millis"].as_i64())).min().unwrap_or_else(common::now_millis);
        let kind=SharedText::try_json(&meta["type"].as_str().unwrap_or_default())?;
        let human_time=if human{chrono::DateTime::<chrono::Utc>::from_timestamp_millis(oldest).map(|t|SharedText::try_format(format_args!(",\"oldest_start_time\":\"{}\"",t.format("%Y-%m-%dT%H:%M:%S%.3fZ")))).transpose()?}else{None};
        let text=SharedText::try_format(format_args!(r#"{{"type":{},"current_counts":{{"snapshots":{count},"clones":0,"finalizations":0,"deletions":0,"snapshot_deletions":0,"active_deletions":0,"shards":{{"total":{shards},"complete":0,"incomplete":{shards},"states":{{"INIT":{shards},"SUCCESS":0,"FAILED":0,"ABORTED":0,"MISSING":0,"WAITING":0,"QUEUED":0,"PAUSED_FOR_NODE_REMOVAL":0}}}}}},"oldest_start_time_millis":{oldest}{}}}"#,kind.as_str(),human_time.as_deref().unwrap_or("")))?;
        let mut entry=OwnedJson::parse(&text)?;output.prepare_reply_owner(&entry)?;
        let mut value=Value::Null;entry.transform(|v,_|{value=std::mem::take(v);Ok(())})?;
        output.transform(|output,memory|{
            let bytes=name.len().checked_add(2*(12*std::mem::size_of::<(String,Value)>()+16*std::mem::size_of::<usize>())).ok_or_else(||EsError::new(429,"circuit_breaking_exception","snapshot stats map overflow"))?;
            memory.grow(bytes as u64,"snapshot stats repository key and map node")?;
            let mut key=String::new();key.try_reserve_exact(name.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;key.push_str(name);
            output["repositories"].as_object_mut().expect("snapshot repository object").insert(key,std::mem::take(&mut value));Ok(())
        })?;
        output.retain_reply_owner(entry);
    }}Ok(output)
}

/// Metadata customs that are internal and must not be rendered under
/// `metadata` in `GET _cluster/state`.
pub fn is_internal_custom(key: &str) -> bool {
    key == common::PROGRESS_CUSTOM
}

/// Mark restored indices' primary entries as `SNAPSHOT` recoveries in a
/// `_recovery` response body.
pub fn patch_recovery_response(st: &ClusterState, body: &mut crate::owned_json::OwnedJson) -> Result<(),elasticsearch_engine::error::EsError> {
    restore::patch_recovery_response(st, body)
}
