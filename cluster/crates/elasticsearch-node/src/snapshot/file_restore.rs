//! File-based restore of a snapshotted shard (ES `RestoreService`,
//! `RecoverySource.SnapshotRecoverySource`, `StoreRecovery.restore`,
//! `BlobStoreRepository.restoreShard`).
//!
//! **Master** ([`super::restore`]). Before it creates a target index it
//! *arms* the index in its restore entry (the `restore` list of the
//! `_snapshots_in_progress` custom): a [`FileRestoreSpec`] naming the
//! snapshot, the repository's index id and the shards that start empty.
//! The master task that adds the index's routing
//! ([`new_index_routing`], from `master.rs`) gives its primaries ES's
//! restore routing (`RecoverySource::Snapshot`, `NEW_INDEX_RESTORED` or
//! `EXISTING_INDEX_RESTORED`, the snapshot's primary terms continued) and
//! *claims* the spec in the same cluster state (records the index uuid,
//! disarms it), so only the incarnation the restore created is restored
//! from files, and a node holding its primary finds the spec in the very
//! state that assigned the copy.
//!
//! **The primary's node** (`ClusterNode::restore_primary_from_snapshot`)
//! resets its copy, downloads the shard's files from the repository into
//! the shard's staging directory (`shard-<n>.recovery/`; every file
//! verified against the snapshot's length and SHA-256 as it is written,
//! its CRC-32 computed over the same bytes, [`stage_shard`]), checks that
//! every member reads their segment formats, makes the target index take
//! the field ids the files were written with (`IndexService::adopt_core_ids`:
//! a new index numbers its fields in sorted order, the snapshotted one as
//! its mapping grew), then opens them as the copy
//! (`RecoverySource::FromRecoveredFiles`: the engine moves the staged shard
//! into `shard-<n>/` after checking each manifest file is present with its
//! length, and replays the translog of a snapshot staged without a flush),
//! promotes it under the shard's primary term and reports it started. The
//! graphs and quantized codes are the snapshotted ones: nothing is
//! re-indexed or re-inserted (beyond the bounded graph lag a snapshot
//! freeze leaves, as a file recovery does).
//!
//! **Replicas** wait for the started primary (the `replica_after_primary_active`
//! decider) and recover from it by file-based peer recovery
//! (`replication::file_recovery`), as for any new replica.
//!
//! **Failures.** A primary whose restore fails is reported failed; the
//! `restore_in_progress` decider never allocates it again (ES marks the
//! shard's restore FAILURE), so the index stays red: nothing corrupt or
//! partial ever starts. A failure tagged [`REINDEX_TAG`] means the files
//! are sound but this cluster cannot adopt them; the master then deletes
//! the index and restores it again by re-indexing.

use super::common::{self, PROGRESS_CUSTOM};
use crate::node::ClusterNode;
use crate::replication::file_recovery as fr;
use crate::replication::recovery::{FileRecoveryStats, RecoveryState, RecoveryText, SnapshotRecoverySource};
use elasticsearch_cluster::allocation::AllocationService;
use elasticsearch_cluster::state::{ClusterState, RecoverySource};
use elasticsearch_engine::hooks::files::{note_checksum, IndexFileEntry, IndexManifest};
use elasticsearch_engine::index::IndexService;
use elasticsearch_snapshots::util::CancelToken;
use elasticsearch_snapshots::Repository;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

/// Prefix of a primary's restore failure that only re-indexing gets past
/// (the files are intact but this cluster cannot adopt them as they are).
pub const REINDEX_TAG: &str = "snapshot files cannot be adopted";

/// What a target index restores from files, kept in its restore entry
/// (`files.<target>`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FileRestoreSpec {
    pub repository: String,
    pub snapshot: String,
    pub snapshot_uuid: String,
    #[serde(default)]
    pub version: Option<String>,
    /// The snapshotted index (the target's name before any rename).
    pub source_index: String,
    /// The repository's id of the snapshotted index.
    pub index_id: String,
    /// The snapshot's highest primary term: the restored index continues it.
    #[serde(default)]
    pub primary_term: u64,
    /// Shards that failed in a partial snapshot: they start empty.
    #[serde(default)]
    pub empty_shards: Vec<u32>,
    /// The target replaced a closed index (`EXISTING_INDEX_RESTORED`).
    #[serde(default)]
    pub replaces_closed: bool,
    /// Set until the master task that creates the index claims it.
    #[serde(default)]
    pub armed: bool,
    /// The incarnation restored from files (set by the claim).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_uuid: Option<String>,
}

fn restore_entries(st: &ClusterState) -> Vec<Value> {
    common::progress_entries(st, "restore")
}

/// The armed spec of `index` and its restore's uuid.
pub fn armed_file_restore(st: &ClusterState, index: &str) -> Option<(String, FileRestoreSpec)> {
    restore_entries(st).into_iter().find_map(|e| {
        let spec: FileRestoreSpec = serde_json::from_value(e["files"].get(index)?.clone()).ok()?;
        spec.armed.then(|| (e["uuid"].as_str().unwrap_or_default().to_string(), spec))
    })
}

/// The spec that restores incarnation `index_uuid` of `index` from files.
pub fn file_restore_of(st: &ClusterState, index: &str, index_uuid: &str) -> Option<(String, FileRestoreSpec)> {
    restore_entries(st).into_iter().find_map(|e| {
        let spec: FileRestoreSpec = serde_json::from_value(e["files"].get(index)?.clone()).ok()?;
        (spec.index_uuid.as_deref() == Some(index_uuid)).then(|| (e["uuid"].as_str().unwrap_or_default().to_string(), spec))
    })
}

/// Record in `st` that incarnation `index_uuid` of `index` is the one
/// restore `restore_uuid` restores from files (and disarm the spec).
fn claim(st: &mut ClusterState, restore_uuid: &str, index: &str, index_uuid: &str) {
    let Some(custom) = st.metadata.customs.get_mut(PROGRESS_CUSTOM) else { return };
    let Some(entries) = custom.get_mut("restore").and_then(Value::as_array_mut) else { return };
    for e in entries.iter_mut().filter(|e| e["uuid"].as_str() == Some(restore_uuid)) {
        if let Some(spec) = e.get_mut("files").and_then(|f| f.get_mut(index)).filter(|s| s.is_object()) {
            spec["armed"] = Value::Bool(false);
            spec["index_uuid"] = Value::String(index_uuid.to_string());
        }
    }
}

/// Routing of the index `name` a master task just added to `next`: ES's
/// restore routing when a restore armed `name` for a file restore
/// (claimed in the same state), else that of a new index.
pub fn new_index_routing(alloc: &AllocationService, mut next: ClusterState, name: &str, now: i64) -> Result<ClusterState, String> {
    let armed = armed_file_restore(&next, name).filter(|_| next.metadata.indices.contains_key(name));
    let Some((restore_uuid, spec)) = armed else { return alloc.add_index_routing(&next, name, now) };
    let im = next.metadata.indices.get_mut(name).expect("the index is in the state");
    let index_uuid = im.uuid.clone();
    // ES restores the snapshot's index metadata, terms included (at least
    // the first term); the primaries' initialization then bumps them, so a
    // restored primary always promotes its copy to a newer term.
    let n = im.number_of_shards.max(1) as usize;
    if im.primary_terms.len() < n {
        im.primary_terms.resize(n, 0);
    }
    for t in im.primary_terms.iter_mut() {
        *t = (*t).max(spec.primary_term.max(1));
    }
    let source = RecoverySource::Snapshot { repository: spec.repository.clone(), snapshot: spec.snapshot.clone(), index: spec.source_index.clone() };
    let mut s = alloc.add_restored_index_routing(&next, name, source, !spec.replaces_closed, now)?;
    // ES `initializeAsRestore`: a shard that failed in a partial snapshot
    // starts as a new empty shard.
    if let Some(irt) = s.routing_table.indices.get_mut(name) {
        for t in irt.shards.iter_mut().filter(|t| spec.empty_shards.contains(&t.shard)) {
            for c in t.copies.iter_mut().filter(|c| c.primary) {
                c.recovery_source = Some(RecoverySource::EmptyStore);
            }
        }
    }
    claim(&mut s, &restore_uuid, name, &index_uuid);
    tracing::info!(index = %name, snapshot = %spec.snapshot, source = %spec.source_index, "restoring the index's primaries from the snapshot's files");
    Ok(s)
}

/// Why a primary's restore failed. `reindex`: the files are intact but
/// cannot be adopted here ([`REINDEX_TAG`]).
#[derive(Debug)]
pub struct RestoreFailure {
    pub reindex: bool,
    pub reason: String,
}

impl RestoreFailure {
    fn hard(reason: impl Into<String>) -> Self {
        RestoreFailure { reindex: false, reason: reason.into() }
    }

    fn reindex(reason: impl Into<String>) -> Self {
        RestoreFailure { reindex: true, reason: reason.into() }
    }

    /// The shard failure's message.
    pub fn message(&self) -> String {
        if self.reindex { format!("{REINDEX_TAG}: {}", self.reason) } else { format!("failed to restore from snapshot: {}", self.reason) }
    }
}

/// Whether a failed shard's details carry [`REINDEX_TAG`].
pub fn requires_reindex(details: &str) -> bool {
    details.contains(REINDEX_TAG)
}

/// The staged files of one shard, ready to be adopted.
pub struct Staged {
    /// The shard's files, as the engine installs them (paths under the
    /// staging directory, CRC-32s of the downloaded bytes).
    pub manifest: IndexManifest,
    /// Store bytes (the shard state file is bookkeeping, not a store file).
    pub bytes: u64,
    /// The local checkpoint of the snapshotted commit.
    pub commit_checkpoint: i64,
    pub download_millis: u64,
}

/// Download shard `shard` of `spec`'s snapshot into `staging` (emptied
/// first) and check that this cluster can adopt it as copy `shard` of
/// `svc` (`readable`: the segment format every member reads). Every file is
/// verified against the snapshot's length and SHA-256 as it is written; a
/// mismatch is a hard failure.
pub async fn stage_shard(node: &ClusterNode, svc: &Arc<IndexService>, shard: u32, spec: &FileRestoreSpec, staging: &Path, readable: u32) -> Result<Staged, RestoreFailure> {
    use elasticsearch_http::pools::{self, Pool};
    let started = std::time::Instant::now();
    match tokio::fs::remove_dir_all(staging).await {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(RestoreFailure::hard(format!("cannot clear [{}]: {e}", staging.display()))),
    }
    let meta = common::repos_in_state(&node.state()).remove(&spec.repository).ok_or_else(|| RestoreFailure::hard(format!("[{}] missing", spec.repository)))?;
    let repo = Repository::open(meta, &common::repo_context(node)).await.map_err(|e| RestoreFailure::hard(format!("[{}] {}", e.kind, e.reason)))?;
    let res = repo
        .restore_shard(&spec.snapshot_uuid, &spec.index_id, shard, staging, &CancelToken::new())
        .await
        .map_err(|e| RestoreFailure::hard(format!("[{}:{}] shard [{}][{shard}]: [{}] {}", spec.repository, spec.snapshot, spec.source_index, e.kind, e.reason)))?;
    let download_millis = started.elapsed().as_millis() as u64;
    // The image is one shard directory (`shard-0/` as staged by the
    // snapshot freeze) beside the source's `index_meta.json`.
    let mut prefixes: Vec<&str> = res.restored.iter().filter_map(|f| f.physical_name.split_once('/').map(|(p, _)| p)).collect();
    prefixes.sort_unstable();
    prefixes.dedup();
    let prefix = match prefixes.as_slice() {
        [p] if p.starts_with("shard-") => p.to_string(),
        other => return Err(RestoreFailure::reindex(format!("the snapshot's shard image has an unexpected layout {other:?}"))),
    };
    let under = format!("{prefix}/");
    let files: Vec<IndexFileEntry> = res.restored.iter().filter(|f| f.physical_name.starts_with(&under))
        .map(|f| IndexFileEntry { path: f.physical_name.clone(), len: f.length, crc32: f.crc32 }).collect();
    let state_path = format!("{prefix}/shard_state.json");
    if !files.iter().any(|f| f.path == state_path) {
        return Err(RestoreFailure::hard(format!("the snapshot of shard [{}][{shard}] has no shard state", spec.source_index)));
    }
    let bytes = files.iter().filter(|f| f.path != state_path).map(|f| f.len).sum();
    let includes_translog = files.iter().any(|f| f.path.starts_with(&format!("{prefix}/translog/")));
    let (check_staging, check_svc, check_prefix) = (staging.to_path_buf(), svc.clone(), prefix.clone());
    let checked = pools::run(Pool::Recovery, move || -> Result<(IndexManifest, i64), RestoreFailure> {
        let state: Value = std::fs::read(check_staging.join(&check_prefix).join("shard_state.json")).ok().and_then(|b| serde_json::from_slice(&b).ok())
            .ok_or_else(|| RestoreFailure::hard("the snapshot's shard state is unreadable"))?;
        let committed = state["committed_seq_no"].as_i64().unwrap_or(-1);
        let checkpoint = state["seqno"]["local_checkpoint"].as_i64().unwrap_or(committed);
        let manifest = IndexManifest {
            index: check_svc.name.clone(), uuid: check_svc.uuid.clone(), files,
            committed_seq_no: committed, local_checkpoint: checkpoint, max_seq_no: checkpoint,
            primary_term: 1, includes_translog,
        };
        // Every member must read the files (a replica, any later primary),
        // not only this node.
        fr::validate_staged_formats_upto(&check_staging, &manifest, readable).map_err(|e| {
            // Not tagged as a segment format refusal: the master would not
            // count it as a failed restore (and allocate it again).
            match elasticsearch_cluster::compatibility::refused_segment_format(&e) {
                Some(required) => RestoreFailure::reindex(format!("the snapshot's files need segment format v{required}, but a member reads only through v{readable}")),
                None => RestoreFailure::reindex(e),
            }
        })?;
        // The restored index takes the field ids the snapshotted files were
        // written with (the source's mapping may have grown in another order
        // than the sorted one a new index assigns).
        let ids = fr::staged_core_ids(&check_staging.join(&check_prefix)).map_err(|e| RestoreFailure::hard(e))?;
        check_svc.adopt_core_ids(&ids).map_err(|e| RestoreFailure::reindex(format!("cannot take the snapshot's field ids: [{}] {}", e.kind, e.reason)))?;
        // Checksummed as downloaded: the copy's later file recoveries (its
        // replicas) find them by file identity instead of reading every file.
        for f in &manifest.files {
            note_checksum(&check_staging.join(&f.path), f.crc32);
        }
        Ok((manifest, checkpoint))
    }).await.map_err(|e| RestoreFailure::hard(e.reason))??;
    let (manifest, commit_checkpoint) = checked;
    tracing::info!(index = %svc.name, shard, files = manifest.files.len(), bytes, reused = res.reused_files, millis = download_millis, "snapshot shard files downloaded and verified");
    Ok(Staged { manifest, bytes, commit_checkpoint, download_millis })
}

/// The `_recovery` record of a restored primary. `files` is the Lucene file
/// count ES reports for the shard (as the single-node engine and a
/// re-indexing restore do), `bytes` the adopted store bytes.
#[allow(clippy::too_many_arguments)]
pub fn recovery_state(index: &str, shard: u32, spec: &FileRestoreSpec, restore_uuid: &str, target_node: &str, staged: &Staged, docs: u64, translog_ops: u64, start_millis: u64, stop_millis: u64) -> Result<RecoveryState, String> {
    let text = |s: &str| RecoveryText::try_new(s).map_err(|e| e.reason.to_string());
    let source = SnapshotRecoverySource {
        repository: text(&spec.repository)?,
        snapshot: text(&spec.snapshot)?,
        version: spec.version.as_deref().map(text).transpose()?,
        index: text(&spec.source_index)?,
        restore_uuid: text(restore_uuid)?,
    };
    let count = elasticsearch_engine::snapshot::lucene_recovered_files(docs, false);
    let files = FileRecoveryStats {
        files_total: count, files_recovered: count, bytes_total: staged.bytes, bytes_recovered: staged.bytes,
        time_millis: staged.download_millis, ..Default::default()
    };
    RecoveryState::snapshot_restore(index, shard, source, target_node, files, translog_ops, start_millis, stop_millis).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn state_with(entry: Value) -> ClusterState {
        let mut st = ClusterState::empty("c");
        st.metadata.customs.insert(PROGRESS_CUSTOM.into(), json!({"restore": [entry]}));
        st
    }

    /// Only the armed name is restored from files, and only the incarnation
    /// the claim recorded.
    #[test]
    fn armed_spec_is_claimed_by_one_incarnation() {
        let spec = FileRestoreSpec { repository: "r".into(), snapshot: "s".into(), snapshot_uuid: "su".into(), source_index: "src".into(), index_id: "id".into(), armed: true, ..Default::default() };
        let mut st = state_with(json!({"uuid": "ru", "files": {"dst": serde_json::to_value(&spec).unwrap()}}));
        assert_eq!(armed_file_restore(&st, "dst").map(|(u, s)| (u, s.source_index)), Some(("ru".into(), "src".into())));
        assert!(armed_file_restore(&st, "other").is_none());
        assert!(file_restore_of(&st, "dst", "u1").is_none());
        claim(&mut st, "ru", "dst", "u1");
        assert!(armed_file_restore(&st, "dst").is_none(), "claimed: a later index of that name is a new index");
        assert_eq!(file_restore_of(&st, "dst", "u1").map(|(_, s)| s.snapshot_uuid), Some("su".into()));
        assert!(file_restore_of(&st, "dst", "u2").is_none());
    }

    #[test]
    fn failure_messages_tag_reindex_fallbacks() {
        assert!(requires_reindex(&RestoreFailure::reindex("field ids differ").message()));
        assert!(!requires_reindex(&RestoreFailure::hard("checksum failed").message()));
        assert!(RestoreFailure::hard("checksum failed").message().starts_with("failed to restore from snapshot"));
    }
}
