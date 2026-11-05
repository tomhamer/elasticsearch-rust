//! Shards snapshotted by "another node" (`ShardSnapshotInput::remote`): the
//! uploader writes the shard blobs through its own repository instance on
//! top of the base generation the coordinator hands it, the coordinator
//! records the new generation and finalizes. Failures make the snapshot
//! PARTIAL (with the remote node in the failure) and a delete aborts the
//! remote upload.

mod common;

use common::*;
use elasticsearch_snapshots::repository::shard::ShardSnapshotTask;
use elasticsearch_snapshots::repository::ShardSnapshotResult;
use elasticsearch_snapshots::snapshot::ShardStatus;
use elasticsearch_snapshots::{
    CancelToken, IndexSnapshotInput, RemoteShard, RemoteShardRequest, RemoteShardSnapshots, RemoteUploader, Repository,
    RepositoryContext, RepositoryMetadata, ShardSnapshotInput, SnapshotError, SnapshotState, SnapshotsService,
};
use parking_lot::Mutex;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

fn ctx(root: &Path) -> RepositoryContext {
    RepositoryContext { path_repo: vec![root.join("repos")], ..Default::default() }
}

enum Mode {
    Upload,
    /// Writes the shard under another index id than the coordinator's (as
    /// an upload started by a previous master that reserved another id).
    WrongIndexId,
    Fail,
    BlockUntilCancelled,
}

/// A "data node": its own repository instance over the same store.
struct Node {
    repo: Arc<Repository>,
    dir: PathBuf,
    files: Vec<String>,
    mode: Mode,
    calls: AtomicUsize,
    bases: Mutex<Vec<Option<String>>>,
    started: AtomicBool,
}

#[async_trait::async_trait]
impl RemoteShardSnapshots for Node {
    async fn snapshot_shard(
        &self,
        _coordinator: Arc<Repository>,
        req: RemoteShardRequest,
        status: Arc<Mutex<ShardStatus>>,
        cancel: CancelToken,
    ) -> Result<ShardSnapshotResult, SnapshotError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.bases.lock().push(req.base_gen.clone());
        self.started.store(true, Ordering::SeqCst);
        match self.mode {
            Mode::Fail => Err(SnapshotError::new("i_o_exception", 500, "disk on the data node failed")),
            Mode::BlockUntilCancelled => {
                while !cancel.is_cancelled() {
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
                Err(SnapshotError::cancelled(cancel.reason().unwrap_or_default()))
            }
            Mode::Upload | Mode::WrongIndexId => {
                let index_id = if matches!(self.mode, Mode::WrongIndexId) { format!("{}-previous-master", req.index_id) } else { req.index_id };
                let task = ShardSnapshotTask {
                    snapshot_name: req.snapshot_name,
                    snapshot_uuid: req.snapshot_uuid,
                    index_id,
                    shard: req.shard,
                    dir: self.dir.clone(),
                    files: self.files.clone().into(),
                    status,
                    cancel,
                };
                self.repo.snapshot_shard_at(req.base_gen, task).await
            }
        }
    }
}

async fn setup(tmp: &Path, mode: Mode) -> (Arc<SnapshotsService>, Arc<Node>, PathBuf, PathBuf) {
    let svc = SnapshotsService::new(ctx(tmp));
    let meta = RepositoryMetadata::new("r", "fs", json!({"location": "r"}));
    svc.put_repository(meta.clone(), true).await.unwrap();
    let local = tmp.join("local");
    write_file(&local, "a", &random_bytes(5_000));
    let remote = tmp.join("remote");
    write_file(&remote, "b", &random_bytes(70_000));
    write_file(&remote, "sub/c", &random_bytes(3_000));
    let node = Arc::new(Node {
        repo: Repository::open(meta, &ctx(tmp)).await.unwrap(),
        dir: remote.clone(),
        files: vec!["b".into(), "sub/c".into()],
        mode,
        calls: AtomicUsize::new(0),
        bases: Mutex::new(Vec::new()),
        started: AtomicBool::new(false),
    });
    (svc, node, local, remote)
}

fn input(local: &Path) -> Vec<IndexSnapshotInput> {
    vec![IndexSnapshotInput {
        index_name: "i".into(),
        index_metadata_json: index_meta("i", 2),
        shards: vec![
            shard(local, 0, &["a"]),
            ShardSnapshotInput { shard: 1, dir: PathBuf::new(), files: Vec::<String>::new().into(), unavailable: None, remote: Some(RemoteShard { node_id: "data-1".into() }) },
        ],
    }]
}

fn remote_request(svc_node: &Arc<Node>, local: &Path, name: &str) -> elasticsearch_snapshots::CreateSnapshotRequest {
    let mut req = request("r", name, input(local));
    req.partial = true;
    req.remote = Some(RemoteUploader(svc_node.clone()));
    req
}

#[tokio::test]
async fn remote_shard_is_uploaded_by_its_node_and_restores() {
    let tmp = tempfile::tempdir().unwrap();
    let (svc, node, local, remote) = setup(tmp.path(), Mode::Upload).await;
    let info = svc.create_snapshot(remote_request(&node, &local, "s1")).await.unwrap();
    assert_eq!(info.state, Some(SnapshotState::Success));
    assert_eq!((info.total_shards, info.successful_shards), (2, 2));
    assert_eq!(node.calls.load(Ordering::SeqCst), 1);
    assert_eq!(node.bases.lock()[0], None, "a new shard has no base generation");

    let out = tmp.path().join("out1");
    svc.restore_shard("r", "s1", "i", 1, &out, &CancelToken::new()).await.unwrap();
    assert_same_dir_files(&remote, &["b", "sub/c"], &out);
    let out = tmp.path().join("out0");
    svc.restore_shard("r", "s1", "i", 0, &out, &CancelToken::new()).await.unwrap();
    assert_same_dir_files(&local, &["a"], &out);

    let st = svc.snapshot_status(Some("r"), &["s1".into()], false, false).await.unwrap();
    let s1 = &st["snapshots"][0]["indices"]["i"]["shards"]["1"];
    assert_eq!(s1["stage"], "DONE", "{st}");
    assert_eq!(s1["stats"]["total"]["file_count"], 2);
    assert_eq!(s1["stats"]["incremental"]["file_count"], 2);

    // Second snapshot: the node builds on the generation the coordinator
    // recorded, so nothing is uploaded again.
    let info = svc.create_snapshot(remote_request(&node, &local, "s2")).await.unwrap();
    assert_eq!(info.state, Some(SnapshotState::Success));
    let bases = node.bases.lock().clone();
    assert!(bases[1].is_some(), "the second upload builds on the first generation");
    let st = svc.snapshot_status(Some("r"), &["s2".into()], false, false).await.unwrap();
    assert_eq!(st["snapshots"][0]["indices"]["i"]["shards"]["1"]["stats"]["incremental"]["file_count"], 0, "{st}");
    assert_eq!(st["snapshots"][0]["indices"]["i"]["shards"]["1"]["stats"]["total"]["file_count"], 2);
    // Data blobs: a, b, sub/c once each.
    assert_eq!(data_blobs(&svc, "r").await.len(), 3);

    // Deleting the first snapshot keeps everything the second references.
    svc.delete_snapshots("r", &["s1".into()]).await.unwrap();
    let out = tmp.path().join("out2");
    svc.restore_shard("r", "s2", "i", 1, &out, &CancelToken::new()).await.unwrap();
    assert_same_dir_files(&remote, &["b", "sub/c"], &out);
}

#[tokio::test]
async fn remote_shard_failure_makes_the_snapshot_partial() {
    let tmp = tempfile::tempdir().unwrap();
    let (svc, node, local, _) = setup(tmp.path(), Mode::Fail).await;
    let info = svc.create_snapshot(remote_request(&node, &local, "p")).await.unwrap();
    assert_eq!(info.state, Some(SnapshotState::Partial));
    assert_eq!((info.total_shards, info.successful_shards), (2, 1));
    let f = info.failures[0].to_json();
    assert_eq!(f["shard_id"], 1);
    assert_eq!(f["node_id"], "data-1", "the failure names the node that held the shard");
    assert_eq!(f["reason"], "i_o_exception: disk on the data node failed");
    let st = svc.snapshot_status(Some("r"), &["p".into()], false, false).await.unwrap();
    assert_eq!(st["snapshots"][0]["indices"]["i"]["shards"]["1"]["stage"], "FAILURE");

    // No uploader at all: the shard fails, the snapshot still finalizes.
    let mut req = request("r", "q", input(&local));
    req.partial = true;
    let info = svc.create_snapshot(req).await.unwrap();
    assert_eq!(info.state, Some(SnapshotState::Partial));
}

#[tokio::test]
async fn deleting_a_running_snapshot_cancels_remote_uploads() {
    let tmp = tempfile::tempdir().unwrap();
    let (svc, node, local, _) = setup(tmp.path(), Mode::BlockUntilCancelled).await;
    let (running, handle) = svc.start_snapshot(remote_request(&node, &local, "s")).await.unwrap();
    while !node.started.load(Ordering::SeqCst) {
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    let st = svc.snapshot_status(Some("r"), &["s".into()], false, false).await.unwrap();
    assert_eq!(st["snapshots"][0]["indices"]["i"]["shards"]["1"]["node"], "data-1", "{st}");
    let ack = tokio::time::timeout(std::time::Duration::from_secs(20), svc.delete_snapshots("r", &["s".into()])).await.expect("abort finishes").unwrap();
    assert_eq!(ack, json!({"acknowledged": true}));
    assert!(running.cancel.is_cancelled());
    assert!(handle.await.unwrap().is_err());
    let data = svc.repository("r").unwrap().repository_data().await.unwrap();
    assert!(data.snapshot_by_name("s").is_none());
}

/// Regression (master failover): a node reporting success for blobs that
/// are not where the coordinator finalizes (another index id) must fail the
/// shard, never be recorded as successful.
#[tokio::test]
async fn remote_success_is_verified_in_the_repository() {
    let tmp = tempfile::tempdir().unwrap();
    let (svc, node, local, _) = setup(tmp.path(), Mode::WrongIndexId).await;
    let info = svc.create_snapshot(remote_request(&node, &local, "w")).await.unwrap();
    assert_eq!(info.state, Some(SnapshotState::Partial));
    assert_eq!((info.total_shards, info.successful_shards), (2, 1));
    let f = info.failures[0].to_json();
    assert_eq!(f["shard_id"], 1);
    let reason = f["reason"].as_str().unwrap();
    assert!(reason.contains("is not in the repository"), "{reason}");
    // The finalized snapshot's successful shard restores; the failed one is
    // not referenced.
    let out = tmp.path().join("w0");
    svc.restore_shard("r", "w", "i", 0, &out, &CancelToken::new()).await.unwrap();
    assert!(svc.restore_shard("r", "w", "i", 1, &tmp.path().join("w1"), &CancelToken::new()).await.is_err());
}
