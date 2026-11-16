//! Data stream lifecycle `frozen_after` (ES 9.5.5 `dlm-frozen-transition`):
//! a backing index older than its stream's `frozen_after` is converted to a
//! partially mounted (frozen tier) searchable snapshot.
//!
//! * Marking (`DataStreamLifecycleService.candidatesForFrozen` /
//!   `maybeMarkIndicesForFrozen`): each data stream lifecycle run marks the
//!   due backing indices with the default repository
//!   (`repositories.default_repository`) in the index's
//!   `data_stream_lifecycle` custom (`dlm_freeze_with`). Without a default
//!   repository nothing is marked (ES only logs it).
//! * Conversion (`DLMConvertToFrozen`), for every marked index, resumed on
//!   every lifecycle pass until it completes: write block, clone to
//!   `dlm-clone-{index}` without replicas (skipped for an index that has
//!   none), force-merge to one segment, snapshot `dlm-frozen-{merged}` into
//!   the marked repository, mount it as `dlm-frozen-{index}`
//!   (`storage=shared_cache`), wait for its primaries, swap it into the data
//!   stream in the original's place, delete the clone and the original.
//!   Each step skips work already done, so an interrupted conversion
//!   resumes where it was.
//! * Errors: ES's unrecoverable conditions (no repository marked, the
//!   repository unregistered or read-only) unmark the index; any other
//!   failure is recorded for the index (DLM explain `error`) and retried on
//!   the next pass. An original deleted meanwhile ends its conversion.
//! * A frozen index DLM created that retention deletes takes its backing
//!   snapshot with it; DLM-created indices left orphaned (their source gone
//!   or out of its data stream) are deleted.
//!
//! The steps themselves are [`super::snapshot_steps`], shared with ILM's
//! `searchable_snapshot` action.

use super::snapshot_steps as ss;
use super::{cluster_setting, runner, StepProgress};
use crate::datastream::DataStream;
use crate::error::EsError;
use crate::index::IndexService;
use crate::node::Node;
use serde_json::{json, Map, Value};

/// `DLMConvertToFrozen.CLONE_INDEX_PREFIX`.
pub const CLONE_INDEX_PREFIX: &str = "dlm-clone-";
/// `DLMConvertToFrozen.SNAPSHOT_NAME_PREFIX` (snapshot and mounted index).
pub const SNAPSHOT_NAME_PREFIX: &str = "dlm-frozen-";
/// `DataStreamLifecycleService.DLM_CREATED_SETTING`: set on the clone and
/// the mounted index (never candidates themselves).
pub const DLM_CREATED_SETTING: &str = "index.dlm.frozen.created";
/// `DLMConvertToFrozen.DLM_CREATED_METADATA_KEY` (snapshot user metadata).
pub const DLM_CREATED_METADATA_KEY: &str = "dlm-created";
/// `DataStreamLifecycleService.FROZEN_CANDIDATE_REPOSITORY_METADATA_KEY`.
pub const FROZEN_CANDIDATE_REPOSITORY_METADATA_KEY: &str = "dlm_freeze_with";
/// `DLMFrozenTransitionSettings.TRANSITION_ENABLED_SETTING`.
pub const TRANSITION_ENABLED_SETTING: &str = "dlm.frozen_transitions.enabled";
/// Settings of the merged index the mount drops
/// (`maybeMountSearchableSnapshot`).
const IGNORED_INDEX_SETTINGS: [&str; 3] = ["index.routing.allocation.total_shards_per_node", "index.lifecycle.name", "index.routing.allocation.include._tier_preference"];

fn is_true(v: Option<String>) -> bool {
    v.as_deref() == Some("true")
}

/// Created by the conversion (a clone or a mounted index).
pub fn dlm_created(svc: &IndexService) -> bool {
    is_true(svc.setting(DLM_CREATED_SETTING))
}

/// The repository `svc` is marked to be frozen with (`Some("")`: marked
/// without one).
pub fn marked_repository(node: &Node, svc: &IndexService) -> Option<String> {
    runner::load(node, svc).data_stream_lifecycle.get(FROZEN_CANDIDATE_REPOSITORY_METADATA_KEY).cloned()
}

fn set_mark(node: &Node, svc: &IndexService, repository: Option<&str>) -> crate::error::Result<()> {
    let mut lc = runner::load(node, svc);
    let changed = match repository {
        Some(r) => lc.data_stream_lifecycle.insert(FROZEN_CANDIDATE_REPOSITORY_METADATA_KEY.into(), r.to_string()).as_deref() != Some(r),
        None => lc.data_stream_lifecycle.remove(FROZEN_CANDIDATE_REPOSITORY_METADATA_KEY).is_some(),
    };
    if changed {
        runner::save(node, svc, &lc)?;
    }
    Ok(())
}

/// Mark `svc` for conversion with `repository` (`MarkIndicesForFrozenTask`).
pub fn mark(node: &Node, svc: &IndexService, repository: &str) -> crate::error::Result<()> {
    set_mark(node, svc, Some(repository))
}

/// Remove the mark (`UnmarkIndexForFrozenTask`).
pub fn unmark(node: &Node, svc: &IndexService) -> crate::error::Result<()> {
    set_mark(node, svc, None)
}

/// The data stream lifecycle's marking step for one data stream, after its
/// retention and downsampling (`candidatesForFrozen`): every backing index
/// but the write index, managed by the lifecycle, older than `frozen_after`
/// (generation time), not marked yet, not created by the conversion and not
/// excluded from this run (`excluded`: lifecycle skip, time series bounds,
/// downsampling) is marked with the default repository, if one is set.
pub fn mark_candidates(node: &Node, ds: &DataStream, excluded: &dyn Fn(&IndexService) -> bool) -> Vec<(String, EsError)> {
    let mut errors = Vec::new();
    let Some(frozen_after) = ds.lifecycle.as_ref().and_then(|l| l.get("frozen_after")).and_then(Value::as_str).and_then(|f| super::parse_time(f, "frozen_after").ok()).map(|t| t.millis()) else {
        return errors;
    };
    let Some(repository) = cluster_setting(node, "repositories.default_repository").filter(|r| !r.is_empty()) else { return errors };
    let now = crate::date::now_millis();
    for idx in ds.indices.iter().filter(|i| i.as_str() != ds.write_index()) {
        let Some(svc) = node.get_index(idx) else { continue };
        if !super::dlm::managed_by_dlm(node, &svc) || excluded(&svc) || dlm_created(&svc) || marked_repository(node, &svc).is_some() {
            continue;
        }
        if now - super::dlm::generation_time(node, &svc) < frozen_after {
            continue;
        }
        if let Err(e) = mark(node, &svc, &repository) {
            errors.push((idx.clone(), e));
        }
    }
    errors
}

/// Why a conversion stopped.
#[derive(Debug)]
pub enum Failure {
    /// `DLMUnrecoverableException`: the index is unmarked, nothing recorded.
    Unrecoverable(String),
    /// The original index is gone: the conversion ends quietly.
    Gone,
    /// Recorded for the index and retried on the next pass.
    Retry(EsError),
}

fn retry(reason: String) -> Failure {
    Failure::Retry(EsError::new(500, "exception", reason))
}

/// `checkIfEligibleForConvertToFrozen`: the index exists and is marked
/// with a registered, writable repository. Returns the repository.
pub fn check_eligible(node: &Node, index: &str) -> Result<String, Failure> {
    let svc = node.get_index(index).ok_or(Failure::Gone)?;
    let repository = marked_repository(node, &svc).filter(|r| !r.is_empty()).ok_or_else(|| {
        Failure::Unrecoverable(format!("Default repository is required for convert-to-frozen steps but was not configured for index [{index}]"))
    })?;
    match ss::repository_readonly(node, &repository) {
        None => Err(Failure::Unrecoverable(format!("Repository [{repository}] required for convert-to-frozen steps is no longer registered in project [default]"))),
        Some(true) => Err(Failure::Unrecoverable(format!("Repository [{repository}] required for convert-to-frozen steps is configured as read-only in project [default]"))),
        Some(false) => Ok(repository),
    }
}

pub fn clone_name(index: &str) -> String {
    format!("{CLONE_INDEX_PREFIX}{index}")
}

pub fn frozen_name(index: &str) -> String {
    format!("{SNAPSHOT_NAME_PREFIX}{index}")
}

/// The index the conversion force-merges, snapshots and mounts
/// (`getIndexForForceMerge`): the clone, unless the original has no
/// replicas and no clone was made.
fn index_for_force_merge(node: &Node, index: &str) -> String {
    let clone = clone_name(index);
    let replicas = node.get_index(index).and_then(|s| s.setting("index.number_of_replicas")).and_then(|r| r.parse::<u32>().ok()).unwrap_or(1);
    if node.get_index(&clone).is_some() || replicas != 0 {
        clone
    } else {
        index.to_string()
    }
}

fn pending(progress: StepProgress) -> Option<StepProgress> {
    matches!(progress, StepProgress::Pending(_)).then_some(progress)
}

/// One pass of `DLMConvertToFrozen.run` for `index`: every step that can
/// complete now does. `Done` once the frozen index replaced the original.
pub fn advance(node: &Node, index: &str) -> Result<StepProgress, Failure> {
    let repository = check_eligible(node, index)?;
    // maybeMarkIndexReadOnly
    if !ss::write_blocked(node, index) {
        ss::add_write_block(node, index).map_err(|e| retry(format!("DLM unable to mark index [{index}] with read only block: {}", e.reason)))?;
    }
    let mounted = frozen_name(index);
    let merged = index_for_force_merge(node, index);
    if node.get_index(&mounted).is_none() {
        // maybeCloneIndex
        if merged != index {
            let mut settings = Map::new();
            settings.insert("index.number_of_replicas".into(), json!(0));
            settings.insert("index.auto_expand_replicas".into(), Value::Null);
            settings.insert(DLM_CREATED_SETTING.into(), json!(true));
            match ss::clone_index(node, index, &merged, &Value::Object(settings)) {
                StepProgress::Done => {}
                StepProgress::Pending(info) => return Ok(StepProgress::Pending(info)),
                StepProgress::Failed(reason) => {
                    // A clone this conversion made is cleaned up (never an
                    // index it did not create).
                    if node.get_index(&merged).is_some_and(|c| dlm_created(&c) && c.setting("index.resize.source.name").as_deref() == Some(index)) {
                        let _ = node.delete_index(&merged);
                    }
                    return Err(retry(format!("DLM failed to clone index [{index}] to index [{merged}]. [{merged}] has been cleaned up by DLM. {reason}")));
                }
            }
            if let Some(p) = pending(ss::primaries_active(node, &merged)) {
                return Ok(p);
            }
        }
        // maybeForceMergeIndex
        match ss::force_merge(node, &merged, 1) {
            StepProgress::Done => {}
            StepProgress::Pending(info) => return Ok(StepProgress::Pending(info)),
            StepProgress::Failed(reason) => return Err(retry(format!("DLM failed to force merge index [{merged}]: {reason}"))),
        }
        // maybeTakeSnapshot
        let snapshot = frozen_name(&merged);
        let mut metadata = Map::new();
        metadata.insert(DLM_CREATED_METADATA_KEY.into(), json!(true));
        match ss::snapshot_index(node, &repository, &snapshot, &merged, &Value::Object(metadata)) {
            StepProgress::Done => {}
            StepProgress::Pending(info) => return Ok(StepProgress::Pending(info)),
            StepProgress::Failed(reason) => return Err(retry(reason)),
        }
        // maybeMountSearchableSnapshot
        let mut settings = Map::new();
        settings.insert(DLM_CREATED_SETTING.into(), json!(true));
        match ss::mount_snapshot(node, &repository, &snapshot, &merged, &mounted, &settings, &IGNORED_INDEX_SETTINGS, true) {
            StepProgress::Done => {}
            StepProgress::Pending(info) => return Ok(StepProgress::Pending(info)),
            StepProgress::Failed(reason) => return Err(retry(reason)),
        }
        if node.get_index(&mounted).is_none() {
            return Ok(StepProgress::Pending(Some(json!({"message": format!("waiting for the mounted index [{mounted}]")}))));
        }
    }
    // waitForMountedIndexToBeAvailable
    if let Some(p) = pending(ss::primaries_active(node, &mounted)) {
        return Ok(p);
    }
    // maybeCleanup: swap the frozen index in, then drop the clone and the
    // original.
    if let Some(ds) = node.data_stream_of_index(index) {
        super::dlm::replace_backing_index(node, &ds, index, &mounted).map_err(Failure::Retry)?;
    }
    if merged != index && node.get_index(&merged).is_some() {
        node.delete_index(&merged).map_err(|e| retry(format!("DLM unable to delete index [{merged}]: {}", e.reason)))?;
    }
    node.delete_index(index).map_err(|e| retry(format!("DLM unable to delete index [{index}]: {}", e.reason)))?;
    Ok(StepProgress::Done)
}

/// A node time setting in milliseconds (`default` when unset or invalid;
/// ES's minimum for these is 1s).
pub fn node_time_setting(node: &Node, key: &str, default: i64) -> i64 {
    let v = node.node_settings.read().get(key).map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()));
    v.and_then(|s| super::parse_time(&s, key).ok()).map(|t| t.millis()).filter(|ms| *ms >= 1000).unwrap_or(default)
}

fn node_int_setting(node: &Node, key: &str) -> Option<usize> {
    node.node_settings.read().get(key).and_then(|v| v.as_u64().map(|n| n as usize).or_else(|| v.as_str().and_then(|s| s.parse().ok())))
}

/// How many conversions may be submitted at once
/// (`DLMFrozenTransitionPlugin`: the `dlm_frozen_transition` pool's max
/// size plus its queue, `min(2 × processors, 100) + min(20 × processors, 1000)`
/// unless `dlm.frozen.transition.thread_pool.size` / `.queue_size` say otherwise).
fn max_submitted(node: &Node) -> usize {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let size = node_int_setting(node, "dlm.frozen.transition.thread_pool.size").unwrap_or((2 * cores).min(100));
    let queue = node_int_setting(node, "dlm.frozen.transition.thread_pool.queue_size").unwrap_or((20 * cores).min(1000));
    size + queue
}

/// `DLMFrozenTransitionService.checkForFrozenIndices`: submit every marked
/// backing index not created by the conversion and not held
/// (`index.lifecycle.skip`), up to the executor's capacity, unless
/// `dlm.frozen_transitions.enabled` is false (submitted ones continue).
fn submit_marked(node: &Node) {
    if cluster_setting(node, TRANSITION_ENABLED_SETTING).is_some_and(|v| v == "false") {
        return;
    }
    let capacity = max_submitted(node);
    let streams: Vec<DataStream> = node.data_streams.read().values().cloned().collect();
    let mut submitted = node.ilm.frozen_submitted.lock();
    for ds in &streams {
        for idx in &ds.indices {
            if submitted.contains(idx) {
                continue;
            }
            if submitted.len() >= capacity {
                return;
            }
            let Some(svc) = node.get_index(idx) else { continue };
            if dlm_created(&svc) || marked_repository(node, &svc).is_none() || is_true(svc.setting("index.lifecycle.skip")) {
                continue;
            }
            submitted.insert(idx.clone());
        }
    }
}

/// One lifecycle pass of the frozen transition (`DLMFrozenTransitionService`
/// / `DLMFrozenTransitionExecutor`): a due transition scan submits the
/// marked indices; every submitted conversion advances
/// ([`advance`]) until it completes, fails or ends. A failure is recorded
/// for the index (DLM explain `error`, ES's `recordAndLogError`) and the
/// conversion leaves the submitted set, so the next scan retries it; an
/// unrecoverable one unmarks the index. A due cleanup scan removes what
/// conversions left orphaned.
pub fn tick(node: &Node, transition_due: bool, cleanup_due: bool) {
    if transition_due {
        submit_marked(node);
    }
    let submitted: Vec<String> = node.ilm.frozen_submitted.lock().iter().cloned().collect();
    if !submitted.is_empty() {
        let mut errors = node.ilm.dlm_stats.lock().frozen_errors.clone();
        let mut ended = Vec::new();
        for idx in &submitted {
            match advance(node, idx) {
                Ok(StepProgress::Pending(_)) => {}
                Ok(_) | Err(Failure::Gone) => {
                    errors.remove(idx);
                    ended.push(idx.clone());
                }
                Err(Failure::Unrecoverable(_)) => {
                    errors.remove(idx);
                    ended.push(idx.clone());
                    if let Some(svc) = node.get_index(idx) {
                        if let Err(e) = unmark(node, &svc) {
                            errors.insert(idx.clone(), super::error_json(&EsError::new(500, "exception", format!("Error unmarking index [{idx}] for conversion to frozen index: {}", e.reason))).to_string());
                        }
                    }
                }
                Err(Failure::Retry(e)) => {
                    errors.insert(idx.clone(), super::error_json(&e).to_string());
                    ended.push(idx.clone());
                }
            }
        }
        // Errors of indices gone or no longer marked are forgotten.
        errors.retain(|k, _| node.get_index(k).is_some_and(|svc| marked_repository(node, &svc).is_some()));
        node.ilm.dlm_stats.lock().frozen_errors = errors;
        node.ilm.frozen_submitted.lock().retain(|i| !ended.contains(i));
    }
    if cleanup_due {
        cleanup_orphans(node);
    }
}

/// A transition scan, every submitted conversion advanced, and a cleanup
/// scan, at once (tests and hosts that drive the steps themselves).
pub fn run(node: &Node) {
    tick(node, true, true);
}

/// The repository and snapshot backing a frozen index the conversion
/// mounted (`dlmCreatedBackingSnapshot`), deleted with it by retention.
pub fn backing_snapshot(svc: &IndexService) -> Option<(String, String)> {
    if !crate::snapshot::is_searchable_snapshot(svc) || !dlm_created(svc) {
        return None;
    }
    let repository = svc.setting("index.store.snapshot.repository_name").filter(|r| !r.is_empty())?;
    let snapshot = svc.setting("index.store.snapshot.snapshot_name").filter(|s| !s.is_empty())?;
    Some((repository, snapshot))
}

/// Best-effort deletion of a deleted frozen index's backing snapshot.
pub fn delete_backing_snapshot(node: &Node, backing: Option<(String, String)>) {
    if let Some((repository, snapshot)) = backing {
        let _ = ss::delete_snapshot(node, &repository, &snapshot);
    }
}

/// `DLMFrozenCleanupService.getOrphanedIndicesInProject`: a DLM-created
/// index outside any data stream whose source (the mounted index's
/// snapshotted index, a clone's resize source) is gone or no longer in a
/// data stream is deleted.
fn cleanup_orphans(node: &Node) {
    let orphans: Vec<String> = node.all_indices().iter().filter(|svc| dlm_created(svc) && is_orphaned(node, &svc.name)).map(|svc| svc.name.clone()).collect();
    for name in orphans {
        let _ = node.delete_index(&name);
    }
    // `getOrphanedSnapshotsInProject`: DLM's snapshots in the default
    // repository that no mounted index uses, holding one index that is
    // orphaned.
    let Some(repository) = cluster_setting(node, "repositories.default_repository").filter(|r| !r.is_empty()) else { return };
    let mounted: std::collections::HashSet<String> = node.all_indices().iter().filter(|svc| crate::snapshot::is_searchable_snapshot(svc)).filter_map(|svc| svc.setting("index.store.snapshot.snapshot_uuid")).collect();
    // Every existing index that is not orphaned (a name that does not
    // exist is: its source is gone).
    let live: std::collections::HashSet<String> = node.all_indices().iter().filter(|svc| !is_orphaned(node, &svc.name)).map(|svc| svc.name.clone()).collect();
    let select = std::sync::Arc::new(move |info: &Value| -> bool {
        let created = info["metadata"][DLM_CREATED_METADATA_KEY] == json!(true);
        let in_use = info["uuid"].as_str().is_some_and(|u| mounted.contains(u));
        let indices: Vec<&str> = info["indices"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
        created && !in_use && indices.len() == 1 && !live.contains(indices[0])
    });
    let _ = ss::delete_snapshots_matching(node, &repository, &format!("{SNAPSHOT_NAME_PREFIX}*"), select);
}

/// `DLMFrozenCleanupService.isIndexOrphaned`: not in a data stream, and its
/// source (the mounted index's snapshotted index, then a clone's resize
/// source) is gone or no longer in a data stream.
fn is_orphaned(node: &Node, name: &str) -> bool {
    let in_stream = |n: &str| node.data_stream_of_index(n).is_some();
    if in_stream(name) {
        return false;
    }
    let mut source = node.get_index(name).and_then(|s| s.setting("index.store.snapshot.index_name")).filter(|s| !s.is_empty()).unwrap_or_else(|| name.to_string());
    if let Some(resized) = node.get_index(&source).and_then(|s| s.setting("index.resize.source.name")).filter(|s| !s.is_empty()) {
        source = resized;
    }
    node.get_index(&source).is_none() || !in_stream(&source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::ResolveOpts;
    use std::collections::HashMap;

    fn node_with_stream(tmp: &std::path::Path, replicas: u32) -> (std::sync::Arc<Node>, String) {
        let node = Node::open(tmp).unwrap();
        node.ilm.drive_externally();
        node.templates.write().index.insert("fz".into(), json!({"index_patterns": ["fz-*"], "data_stream": {}, "template": {
            "settings": {"index.number_of_replicas": replicas.to_string()},
            "mappings": {"properties": {"@timestamp": {"type": "date"}, "n": {"type": "long"}}}}}));
        node.create_data_stream("fz-a").unwrap();
        super::super::dlm::put_lifecycle(&node, "fz-a", false, Some(&json!({"frozen_after": "1d", "data_retention": "30d"}))).unwrap();
        let first = node.data_stream("fz-a").unwrap().write_index().to_string();
        let svc = node.get_index(&first).unwrap();
        let reqs = (0..5).map(|i| crate::index::WriteReq::Index { id: Some(i.to_string()), source: json!({"@timestamp": "2026-01-01T00:00:00Z", "n": i}).to_string(), routing: None, cond: Default::default() }).collect();
        for r in svc.write(reqs, true) {
            r.unwrap();
        }
        node.rollover("fz-a", None, &json!({}), false).unwrap();
        let mut origin = crate::settings::FlatSettings::new();
        origin.insert("index.lifecycle.origination_date".into(), json!((crate::date::now_millis() - 2 * 86_400_000).to_string()));
        svc.put_settings(origin, false).unwrap();
        (node, first)
    }

    fn register_repo(node: &Node, tmp: &std::path::Path) {
        let loc = tmp.join("repo");
        node.snapshots.put_repository("frozen-repo", &json!({"type": "fs", "settings": {"location": loc.to_string_lossy()}})).unwrap();
        node.cluster_settings.write().0.insert("repositories.default_repository".into(), json!("frozen-repo"));
    }

    fn explain_error(node: &Node, index: &str) -> String {
        let ex = super::super::dlm::explain(node, index, ResolveOpts::from_params(&HashMap::new()), false).unwrap();
        ex["indices"][index]["error"].as_str().unwrap_or_default().to_string()
    }

    /// The full conversion on a standalone node against an fs repository:
    /// marked by the lifecycle run, then write block, clone (the index has
    /// a replica), force-merge, snapshot `dlm-frozen-dlm-clone-{index}`,
    /// partial mount `dlm-frozen-{index}`, swap, cleanup. The frozen index
    /// holds the documents in the original's place; the clone and the
    /// original are gone.
    #[test]
    fn frozen_after_converts_a_backing_index_to_a_frozen_searchable_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let (node, first) = node_with_stream(tmp.path(), 1);
        register_repo(&node, tmp.path());
        super::super::dlm::run(&node);
        assert_eq!(marked_repository(&node, &node.get_index(&first).unwrap()).as_deref(), Some("frozen-repo"));
        run(&node);
        let mounted = frozen_name(&first);
        let ds = node.data_stream("fz-a").unwrap();
        assert_eq!(ds.indices[0], mounted, "{:?}", ds.indices);
        assert!(node.get_index(&first).is_none() && node.get_index(&clone_name(&first)).is_none());
        let svc = node.get_index(&mounted).unwrap();
        assert!(crate::snapshot::is_searchable_snapshot(&svc));
        assert_eq!(svc.setting("index.store.snapshot.partial").as_deref(), Some("true"));
        assert_eq!(svc.setting(DLM_CREATED_SETTING).as_deref(), Some("true"));
        assert_eq!(svc.setting("index.store.snapshot.snapshot_name").as_deref(), Some(frozen_name(&clone_name(&first)).as_str()));
        assert_eq!(svc.doc_count().0, 5);
        let snaps = node.snapshots.get_snapshots("frozen-repo", "_all", false).unwrap();
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0]["metadata"][DLM_CREATED_METADATA_KEY], json!(true));
        assert!(explain_error(&node, &mounted).is_empty());
        // Never converted again.
        super::super::dlm::run(&node);
        run(&node);
        assert_eq!(node.data_stream("fz-a").unwrap().indices[0], mounted);
    }

    /// Without replicas the original is force-merged and snapshotted
    /// itself (no clone).
    #[test]
    fn frozen_after_without_replicas_skips_the_clone() {
        let tmp = tempfile::tempdir().unwrap();
        let (node, first) = node_with_stream(tmp.path(), 0);
        register_repo(&node, tmp.path());
        super::super::dlm::run(&node);
        run(&node);
        let svc = node.get_index(&frozen_name(&first)).unwrap();
        assert_eq!(svc.setting("index.store.snapshot.snapshot_name").as_deref(), Some(frozen_name(&first).as_str()));
        assert_eq!(svc.doc_count().0, 5);
    }

    /// No default repository: nothing is marked or converted (ES only
    /// logs). An index marked without a repository, or with one no longer
    /// registered, fails with ES's unrecoverable error and is unmarked.
    #[test]
    fn frozen_after_without_a_repository() {
        let tmp = tempfile::tempdir().unwrap();
        let (node, first) = node_with_stream(tmp.path(), 1);
        super::super::dlm::run(&node);
        let svc = node.get_index(&first).unwrap();
        assert_eq!(marked_repository(&node, &svc), None);
        run(&node);
        assert!(node.get_index(&first).is_some() && !ss::write_blocked(&node, &first));
        mark(&node, &svc, "").unwrap();
        match check_eligible(&node, &first) {
            Err(Failure::Unrecoverable(m)) => assert_eq!(m, format!("Default repository is required for convert-to-frozen steps but was not configured for index [{first}]")),
            other => panic!("{other:?}"),
        }
        mark(&node, &svc, "gone").unwrap();
        match check_eligible(&node, &first) {
            Err(Failure::Unrecoverable(m)) => assert_eq!(m, "Repository [gone] required for convert-to-frozen steps is no longer registered in project [default]"),
            other => panic!("{other:?}"),
        }
        run(&node);
        assert_eq!(marked_repository(&node, &svc), None, "unmarked");
        assert!(node.get_index(&first).is_some());
    }

    /// A step failing records the error for the index in the DLM explain
    /// and is retried on the next pass: here a foreign index holds the
    /// clone's name (never adopted or deleted); once it is gone the
    /// conversion completes and the error clears.
    #[test]
    fn frozen_after_failures_are_recorded_and_retried() {
        let tmp = tempfile::tempdir().unwrap();
        let (node, first) = node_with_stream(tmp.path(), 1);
        register_repo(&node, tmp.path());
        node.create_index(&clone_name(&first), &json!({})).unwrap();
        super::super::dlm::run(&node);
        run(&node);
        let err = explain_error(&node, &first);
        assert!(err.contains(&format!("DLM failed to clone index [{first}] to index [{}]", clone_name(&first))), "{err}");
        assert!(node.get_index(&first).is_some() && node.get_index(&clone_name(&first)).is_some());
        run(&node);
        assert!(!explain_error(&node, &first).is_empty(), "still failing, still recorded");
        node.delete_index(&clone_name(&first)).unwrap();
        run(&node);
        assert!(node.get_index(&first).is_none());
        assert_eq!(node.data_stream("fz-a").unwrap().indices[0], frozen_name(&first));
        assert!(explain_error(&node, &frozen_name(&first)).is_empty());
        assert!(node.ilm.dlm_stats.lock().frozen_errors.is_empty());
    }

    /// The transition scan submits; between scans submitted conversions
    /// advance and nothing new starts (`dlm.frozen.transition.poll_interval`).
    /// The cleanup scan deletes DLM's snapshot once nothing mounts it and
    /// its index is gone (`DLMFrozenCleanupService`), never while mounted.
    #[test]
    fn frozen_transition_scan_and_orphaned_snapshot_cleanup() {
        let tmp = tempfile::tempdir().unwrap();
        let (node, first) = node_with_stream(tmp.path(), 1);
        register_repo(&node, tmp.path());
        super::super::dlm::run(&node);
        tick(&node, false, false);
        assert!(!ss::write_blocked(&node, &first), "a conversion started without a transition scan");
        tick(&node, true, false);
        let mounted = frozen_name(&first);
        assert_eq!(node.data_stream("fz-a").unwrap().indices[0], mounted);
        assert!(node.ilm.frozen_submitted.lock().is_empty(), "a completed conversion stays submitted");
        tick(&node, false, true);
        assert_eq!(node.snapshots.get_snapshots("frozen-repo", "_all", false).unwrap().len(), 1, "a mounted snapshot was cleaned up");
        // The frozen index goes (the data stream is deleted): its snapshot
        // is orphaned.
        node.delete_data_streams("fz-a").unwrap();
        assert!(node.get_index(&mounted).is_none());
        tick(&node, false, true);
        assert!(node.snapshots.get_snapshots("frozen-repo", "_all", false).unwrap().is_empty(), "the orphaned snapshot was kept");
    }
}
