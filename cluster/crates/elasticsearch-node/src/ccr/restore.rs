//! A follower primary's restore from its leader (ES `CcrRepository`): the
//! snapshot file restore's staging step, with the files of the leader
//! primary's pinned commit (a restore session on the leader) instead of a
//! blob store. Everything after it — verification of the staged formats,
//! the field ids the files were written with, the open of the staged copy,
//! its promotion and the `_recovery` record (source `SNAPSHOT`, repository
//! `_ccr_<alias>`, snapshot `_latest_`) — is the snapshot restore's.

use super::leader::{self, ChunkReq, ChunkResp, SessionClearReq, SessionPutReq, SessionPutResp};
use super::state::CcrState;
use crate::node::ClusterNode;
use crate::replication::file_recovery as fr;
use crate::snapshot_api::file_restore::{FileRestoreSpec, RestoreFailure, Staged};
use elasticsearch_engine::hooks::files::{note_checksum, IndexManifest};
use elasticsearch_engine::index::IndexService;
use elasticsearch_snapshots::util::CancelToken;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

/// ES's repository name of a remote cluster's leader indices.
pub fn repository(alias: &str) -> String {
    format!("_ccr_{alias}")
}

/// The snapshot name ES restores followers from.
pub const LATEST: &str = "_latest_";

/// `ccr.indices.recovery.chunk_size` (1mb).
fn chunk_size(node: &ClusterNode) -> u64 {
    node.effective_setting("ccr.indices.recovery.chunk_size")
        .and_then(|s| elasticsearch_engine::ilm::parse_size(&s, "ccr.indices.recovery.chunk_size").ok())
        .map(|s| s.0.clamp(1 << 10, 1 << 30))
        .unwrap_or(1 << 20)
}

/// Download shard `shard` of the leader index `spec` names from remote
/// cluster `alias` into `staging`, and check that this cluster can adopt
/// the files as copy `shard` of `svc` (as `file_restore::stage_shard`).
#[allow(clippy::too_many_arguments)]
pub async fn stage_shard(node: &ClusterNode, svc: &Arc<IndexService>, shard: u32, spec: &FileRestoreSpec, alias: &str, staging: &Path, readable: u32, cancel: &CancelToken) -> Result<Staged, RestoreFailure> {
    use elasticsearch_http::pools::{self, Pool};
    let started = std::time::Instant::now();
    let st = node.state();
    let follower = CcrState::of(&st).followers.get(&svc.name).cloned();
    let identity = crate::remote_cluster::outgoing_identity(node, alias, follower.as_ref().and_then(|f| f.identity.clone()));
    let follower_uuid = st.metadata.indices.get(&svc.name).map(|im| im.uuid.clone()).unwrap_or_default();
    let lease_id = elasticsearch_engine::ccr::retention_lease_id(&st.cluster_name, &svc.name, &follower_uuid, alias, &spec.source_index, &spec.index_id);
    match tokio::fs::remove_dir_all(staging).await {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(RestoreFailure::hard(format!("cannot clear [{}]: {e}", staging.display()))),
    }
    let put = SessionPutReq { identity: identity.clone(), index: spec.source_index.clone(), uuid: spec.index_id.clone(), shard, lease_id, routed: false };
    let session: SessionPutResp = leader::call(node, alias, leader::ACTION_SESSION_PUT, &put, Duration::from_secs(120)).await
        .map_err(|e| RestoreFailure::hard(format!("[{}:{LATEST}] shard [{}][{shard}]: [{}] {}", repository(alias), spec.source_index, e.kind, e.reason)))?;
    let leader_manifest: IndexManifest = serde_json::from_str(&session.manifest_json).map_err(|e| RestoreFailure::hard(format!("unreadable leader manifest: {e}")))?;
    let chunk = chunk_size(node);
    let fetched = async {
        for f in &leader_manifest.files {
            if cancel.is_cancelled() {
                return Err(RestoreFailure::hard("the restore was aborted: the index was deleted or its restore is no longer in progress"));
            }
            let dest = staging.join(&f.path);
            if let Some(parent) = dest.parent() {
                tokio::fs::create_dir_all(parent).await.map_err(|e| RestoreFailure::hard(format!("cannot create [{}]: {e}", parent.display())))?;
            }
            let mut out = tokio::fs::File::create(&dest).await.map_err(|e| RestoreFailure::hard(format!("cannot create [{}]: {e}", dest.display())))?;
            let mut offset = 0u64;
            while offset < f.len {
                let len = chunk.min(f.len - offset);
                let req = ChunkReq { identity: identity.clone(), index: spec.source_index.clone(), session: session.session.clone(), node_id: session.node_id.clone(), path: f.path.clone(), offset, len, routed: false };
                let r: ChunkResp = leader::call(node, alias, leader::ACTION_FILE_CHUNK, &req, Duration::from_secs(60)).await
                    .map_err(|e| RestoreFailure::hard(format!("fetching [{}] from the leader: [{}] {}", f.path, e.kind, e.reason)))?;
                if r.bytes.len() as u64 != len {
                    return Err(RestoreFailure::hard(format!("short chunk of [{}] at [{offset}]: [{}] of [{len}] bytes", f.path, r.bytes.len())));
                }
                tokio::io::AsyncWriteExt::write_all(&mut out, &r.bytes).await.map_err(|e| RestoreFailure::hard(format!("writing [{}]: {e}", dest.display())))?;
                offset += len;
            }
            tokio::io::AsyncWriteExt::flush(&mut out).await.map_err(|e| RestoreFailure::hard(e.to_string()))?;
            out.sync_all().await.map_err(|e| RestoreFailure::hard(e.to_string()))?;
            drop(out);
            // Every file is verified against the leader's manifest.
            let (check, crc_expected, len_expected) = (dest.clone(), f.crc32, f.len);
            let (len, crc) = pools::run(Pool::Recovery, move || elasticsearch_engine::hooks::files::file_crc32(&check))
                .await
                .map_err(|e| RestoreFailure::hard(e.reason))?
                .map_err(|e| RestoreFailure::hard(e.reason))?;
            if len != len_expected || crc != crc_expected {
                return Err(RestoreFailure::hard(format!("file [{}] does not match the leader's: length [{len}] crc32 [{crc:08x}], expected [{len_expected}] [{crc_expected:08x}]", f.path)));
            }
        }
        Ok(())
    }
    .await;
    let clear = SessionClearReq { identity, index: spec.source_index.clone(), session: session.session.clone(), node_id: session.node_id.clone(), routed: false };
    if let Err(e) = leader::call::<_, ()>(node, alias, leader::ACTION_SESSION_CLEAR, &clear, Duration::from_secs(30)).await {
        tracing::debug!(index = %svc.name, shard, "clearing the CCR restore session failed: {e}");
    }
    fetched?;
    let download_millis = started.elapsed().as_millis() as u64;
    let bytes = leader_manifest.total_bytes();
    let manifest = IndexManifest { index: svc.name.clone(), uuid: svc.uuid.clone(), primary_term: 1, ..leader_manifest };
    let commit_checkpoint = manifest.committed_seq_no;
    let (check_staging, check_svc, check_manifest) = (staging.to_path_buf(), svc.clone(), manifest.clone());
    pools::run(Pool::Recovery, move || -> Result<(), RestoreFailure> {
        fr::validate_staged_formats_upto(&check_staging, &check_manifest, readable).map_err(|e| match elasticsearch_cluster::compatibility::refused_segment_format(&e) {
            Some(required) => RestoreFailure::reindex(format!("the leader's files need segment format v{required}, but a member reads only through v{readable}")),
            None => RestoreFailure::reindex(e),
        })?;
        let ids = fr::staged_core_ids(&check_staging).map_err(RestoreFailure::hard)?;
        check_svc.adopt_core_ids(&ids).map_err(|e| RestoreFailure::reindex(format!("cannot take the leader's field ids: [{}] {}", e.kind, e.reason)))?;
        for f in &check_manifest.files {
            note_checksum(&check_staging.join(&f.path), f.crc32);
        }
        Ok(())
    })
    .await
    .map_err(|e| RestoreFailure::hard(e.reason))??;
    tracing::info!(index = %svc.name, shard, files = manifest.files.len(), bytes, millis = download_millis, "follower shard files fetched from the leader and verified");
    Ok(Staged { manifest, bytes, commit_checkpoint, download_millis })
}
