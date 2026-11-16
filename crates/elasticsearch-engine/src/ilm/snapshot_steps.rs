//! The steps that turn an index into a searchable snapshot, shared by the
//! data stream lifecycle's `frozen_after` conversion (ES 9.5.5
//! `dlm-frozen-transition`, [`super::frozen`]) and ILM's
//! `searchable_snapshot` action: write block, clone, force-merge, snapshot,
//! mount, and the checks between them.
//!
//! Every step returns at once with a [`StepProgress`]: on a cluster the work
//! goes through the cluster's own APIs ([`ClusterLifecycleSteps`]: the
//! lifecycle pass runs on the master's metadata copy, which holds no
//! documents), started on the first call and reported on later passes;
//! on a standalone node it runs here, on the node's own shards, and is
//! `Done` (or failed) when the call returns. Each step is idempotent: a
//! step whose result is already there reports `Done` without redoing it,
//! so a conversion interrupted anywhere (a restart, a master change)
//! resumes where it was.

use super::{ClusterLifecycleSteps, StepProgress};
use crate::error::{EsError, Result};
use crate::node::Node;
use serde_json::{json, Map, Value};
use std::sync::Arc;

fn steps(node: &Node) -> Option<Arc<dyn ClusterLifecycleSteps>> {
    node.ilm.cluster_steps()
}

/// A step's failure as an error (`Done` / `Pending` pass through).
pub fn check(progress: StepProgress) -> Result<StepProgress> {
    match progress {
        StepProgress::Failed(reason) => Err(EsError::new(500, "exception", reason)),
        other => Ok(other),
    }
}

fn setting_true(v: Option<String>) -> bool {
    v.as_deref() == Some("true")
}

/// Whether `index` carries a write block (`index.blocks.write`).
pub fn write_blocked(node: &Node, index: &str) -> bool {
    node.get_index(index).is_some_and(|svc| setting_true(svc.setting("index.blocks.write")))
}

/// Add the write block to `index` (`AddIndexBlockRequest(WRITE)`), if it
/// has none. On a cluster the setting is published with the lifecycle pass;
/// the steps that need it ([`clone_index`]) wait until it is.
pub fn add_write_block(node: &Node, index: &str) -> Result<()> {
    let svc = node.get_index(index).ok_or_else(|| EsError::index_not_found(index))?;
    if setting_true(svc.setting("index.blocks.write")) {
        return Ok(());
    }
    let mut block = crate::settings::FlatSettings::new();
    block.insert("index.blocks.write".into(), json!("true"));
    svc.put_settings(block, false)
}

/// Every primary of `index` is active (yellow). Standalone: the index
/// exists (its shards are local).
pub fn primaries_active(node: &Node, index: &str) -> StepProgress {
    match steps(node) {
        Some(steps) => steps.primaries_active(index),
        None if node.get_index(index).is_some() => StepProgress::Done,
        None => StepProgress::Pending(Some(json!({"message": format!("waiting for index [{index}] to exist")}))),
    }
}

/// Clone `source` (write blocked) into `target` with `settings` on top of
/// the source's (`ResizeType.CLONE`). `Done` once `target` is this source
/// incarnation's completed clone; an index under the name that is not is
/// never adopted (the step fails).
pub fn clone_index(node: &Node, source: &str, target: &str, settings: &Value) -> StepProgress {
    let body = json!({"settings": settings});
    if let Some(steps) = steps(node) {
        return steps.clone_index(source, target, &body);
    }
    let Some(src) = node.get_index(source) else { return StepProgress::Failed(format!("no such index [{source}]")) };
    if let Some(t) = node.get_index(target) {
        return if t.setting("index.resize.source.uuid").as_deref() == Some(src.uuid.as_str()) {
            StepProgress::Done
        } else {
            StepProgress::Failed(format!("index [{target}] ([{}]) exists but was not created by a clone of [{source}] ([{}]): it is neither reused nor deleted", t.uuid, src.uuid))
        };
    }
    drop(src);
    match super::resize::resize(node, source, target, super::resize::ResizeType::Clone, &body) {
        Ok(_) => StepProgress::Done,
        Err(e) => StepProgress::Failed(e.reason),
    }
}

/// Force-merge every copy of `index` to at most `max_num_segments`.
pub fn force_merge(node: &Node, index: &str, max_num_segments: usize) -> StepProgress {
    if let Some(steps) = steps(node) {
        return steps.forcemerge(index, max_num_segments);
    }
    let Some(svc) = node.get_index(index) else { return StepProgress::Failed(format!("no such index [{index}]")) };
    match svc.force_merge(max_num_segments, false) {
        Ok(()) => StepProgress::Done,
        Err(e) => StepProgress::Failed(format!("force merge of [{index}] failed: {}", e.reason)),
    }
}

/// ES `snapshotCompletedSuccessfully`: `info` (a snapshot's info or
/// metadata) succeeded with no failed shard, at least one successful one,
/// and includes `index`.
pub fn snapshot_complete(info: &Value, index: &str) -> bool {
    info["state"] == "SUCCESS"
        && info["shards"]["failed"].as_u64().unwrap_or(0) == 0
        && info["shards"]["successful"].as_u64().unwrap_or(0) > 0
        && info["indices"].as_array().is_some_and(|a| a.iter().any(|i| i.as_str() == Some(index)))
}

/// Whether `info` is a snapshot still running.
pub fn snapshot_in_progress(info: &Value) -> bool {
    matches!(info["state"].as_str(), Some("IN_PROGRESS" | "STARTED" | "INIT"))
}

/// ES `checkSnapshotInfoSuccess`: `Ok` for a complete snapshot of `index`,
/// else its failure as ES words it.
pub fn check_snapshot_info(index: &str, snapshot: &str, info: Option<&Value>) -> std::result::Result<(), String> {
    let Some(info) = info.filter(|i| !i.is_null()) else {
        return Err(format!("DLM snapshot [{snapshot}] for index [{index}] did not return snapshot info"));
    };
    if snapshot_complete(info, index) {
        return Ok(());
    }
    let successful = info["shards"]["successful"].as_u64().unwrap_or(0);
    let failed = info["shards"]["failed"].as_u64().unwrap_or(0);
    let state = info["state"].as_str().unwrap_or("unknown");
    let indices = info["indices"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(", ")).unwrap_or_default();
    match info["reason"].as_str().filter(|r| !r.is_empty()) {
        Some(reason) => Err(format!(
            "DLM snapshot [{snapshot}] for index [{index}] finished with [{successful}] successful shards and [{failed}] failed shards, state [{state}], indices [{indices}], reason [{reason}]"
        )),
        None => Err(format!(
            "DLM snapshot [{snapshot}] for index [{index}] finished with [{successful}] successful shards and [{failed}] failed shards, state [{state}], indices [{indices}]"
        )),
    }
}

/// The create-snapshot body of a snapshot holding `index` alone (ES
/// `buildCreateSnapshotRequest`: no global state, partial allowed,
/// `metadata` as the user metadata).
pub fn snapshot_body(index: &str, metadata: &Value) -> Value {
    json!({"indices": [index], "include_global_state": false, "partial": true, "metadata": metadata})
}

/// Snapshot `index` alone into `repository` as `snapshot` (`metadata`: the
/// snapshot's user metadata), waiting for it. A complete snapshot already
/// under the name is kept; a failed or partial one is deleted and taken
/// again (ES `checkForOrphanedSnapshotAndStart`).
pub fn snapshot_index(node: &Node, repository: &str, snapshot: &str, index: &str, metadata: &Value) -> StepProgress {
    let body = snapshot_body(index, metadata);
    if let Some(steps) = steps(node) {
        return steps.snapshot_index(repository, snapshot, index, &body);
    }
    let existing = match node.snapshots.get_snapshots(repository, snapshot, true) {
        Ok(found) => found.into_iter().next(),
        Err(e) => return StepProgress::Failed(format!("DLM failed to start snapshot [{snapshot}] for index [{index}]: {}", e.reason)),
    };
    if let Some(info) = existing {
        if snapshot_complete(&info, index) {
            return StepProgress::Done;
        }
        if let Err(e) = node.snapshots.delete_snapshots(repository, snapshot) {
            return StepProgress::Failed(format!("DLM failed to delete incomplete snapshot [{snapshot}] for index [{index}]: {}", e.reason));
        }
    }
    match node.snapshots.create_snapshot(node, repository, snapshot, &body) {
        Ok(info) => match check_snapshot_info(index, snapshot, Some(&info)) {
            Ok(()) => StepProgress::Done,
            Err(e) => StepProgress::Failed(e),
        },
        Err(e) => StepProgress::Failed(format!("DLM failed to start snapshot [{snapshot}] for index [{index}]: {}", e.reason)),
    }
}

/// The `_mount` body mounting `index` of a snapshot as `mounted`.
pub fn mount_body(index: &str, mounted: &str, index_settings: &Map<String, Value>, ignore_index_settings: &[&str]) -> Value {
    json!({"index": index, "renamed_index": mounted, "index_settings": index_settings, "ignore_index_settings": ignore_index_settings})
}

/// ES's mount response check: the restore reported, with no failed shard
/// and at least one successful one.
pub fn check_mount_response(snapshot: &str, response: &Value) -> std::result::Result<(), String> {
    let shards = &response["snapshot"]["shards"];
    if shards.is_null() {
        return Err(format!("DLM failed to mount snapshot [{snapshot}] because the restore info was missing"));
    }
    if shards["failed"].as_u64().unwrap_or(0) > 0 || shards["successful"].as_u64().unwrap_or(0) == 0 {
        return Err(format!(
            "DLM failed to mount snapshot [{snapshot}] because there were failed shards or no successful shards. Restore info: [{shards}]"
        ));
    }
    Ok(())
}

/// Mount `index` of `repository`/`snapshot` as the searchable snapshot
/// index `mounted` (`partial`: `storage=shared_cache`, the frozen tier),
/// waiting for the restore.
pub fn mount_snapshot(node: &Node, repository: &str, snapshot: &str, index: &str, mounted: &str, index_settings: &Map<String, Value>, ignore_index_settings: &[&str], partial: bool) -> StepProgress {
    if let Some(steps) = steps(node) {
        return steps.mount_snapshot(repository, snapshot, &mount_body(index, mounted, index_settings, ignore_index_settings), partial);
    }
    let ignore: Vec<String> = ignore_index_settings.iter().map(|s| s.to_string()).collect();
    match node.snapshots.mount(node, repository, snapshot, index, mounted, index_settings, &ignore, partial) {
        Ok(r) => match check_mount_response(snapshot, &r) {
            Ok(()) => StepProgress::Done,
            Err(e) => StepProgress::Failed(e),
        },
        Err(e) => StepProgress::Failed(format!("DLM failed while mounting snapshot [{snapshot}]: {}", e.reason)),
    }
}

/// Delete `snapshot` of `repository` (gone already: `Done`).
pub fn delete_snapshot(node: &Node, repository: &str, snapshot: &str) -> StepProgress {
    if let Some(steps) = steps(node) {
        return steps.delete_snapshot(repository, snapshot);
    }
    match node.snapshots.get_snapshots(repository, snapshot, true) {
        Ok(found) if found.is_empty() => StepProgress::Done,
        Ok(_) => match node.snapshots.delete_snapshots(repository, snapshot) {
            Ok(()) => StepProgress::Done,
            Err(e) => StepProgress::Failed(e.reason),
        },
        Err(e) => StepProgress::Failed(e.reason),
    }
}

/// A selection of a repository's snapshots (`info`: the snapshot's info,
/// with `uuid`, `indices` and `metadata`).
pub type SnapshotSelect = Arc<dyn Fn(&Value) -> bool + Send + Sync>;

/// Delete the snapshots of `repository` matching `pattern` that `select`
/// picks (best effort: a repository without matches is `Done`).
pub fn delete_snapshots_matching(node: &Node, repository: &str, pattern: &str, select: SnapshotSelect) -> StepProgress {
    if let Some(steps) = steps(node) {
        return steps.delete_snapshots_matching(repository, pattern, select);
    }
    let found = match node.snapshots.get_snapshots(repository, pattern, true) {
        Ok(found) => found,
        Err(e) => return StepProgress::Failed(e.reason),
    };
    let names: Vec<String> = found.iter().filter(|info| select(info)).filter_map(|info| info["snapshot"].as_str().map(str::to_string)).collect();
    if names.is_empty() {
        return StepProgress::Done;
    }
    match node.snapshots.delete_snapshots(repository, &names.join(",")) {
        Ok(()) => StepProgress::Done,
        Err(e) => StepProgress::Failed(e.reason),
    }
}

/// Whether repository `name` is registered (`Some(readonly)`), from the
/// cluster's catalog on a cluster node.
pub fn repository_readonly(node: &Node, name: &str) -> Option<bool> {
    node.snapshots.registered_readonly(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_info_checks_match_es() {
        let ok = json!({"state": "SUCCESS", "shards": {"failed": 0, "successful": 1}, "indices": ["a"]});
        assert!(snapshot_complete(&ok, "a"));
        assert!(!snapshot_complete(&ok, "b"));
        assert_eq!(check_snapshot_info("a", "s", Some(&ok)), Ok(()));
        let partial = json!({"state": "PARTIAL", "shards": {"failed": 1, "successful": 1}, "indices": ["a"]});
        assert_eq!(
            check_snapshot_info("a", "s", Some(&partial)).unwrap_err(),
            "DLM snapshot [s] for index [a] finished with [1] successful shards and [1] failed shards, state [PARTIAL], indices [a]"
        );
        assert_eq!(check_snapshot_info("a", "s", None).unwrap_err(), "DLM snapshot [s] for index [a] did not return snapshot info");
        assert!(check_mount_response("s", &json!({"snapshot": {"shards": {"total": 1, "failed": 0, "successful": 1}}})).is_ok());
        assert!(check_mount_response("s", &json!({"snapshot": {"shards": {"total": 1, "failed": 1, "successful": 0}}})).is_err());
        assert_eq!(check_mount_response("s", &json!({"accepted": true})).unwrap_err(), "DLM failed to mount snapshot [s] because the restore info was missing");
    }
}
