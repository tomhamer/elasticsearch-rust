//! Repository operations racing each other:
//!
//! * a cleanup that has passed its idle check excludes new snapshots and
//!   clones (ES rejects them while a cleanup runs), so it can never delete
//!   the files of a snapshot that then reports success;
//! * a snapshot being cloned cannot be deleted (ES `cannot delete snapshot
//!   while it is being cloned`), so the clone's files survive and the clone
//!   restores completely.

mod common;

use common::*;
use elasticsearch_snapshots::repository::TestHook;
use elasticsearch_snapshots::{CancelToken, IndexSnapshotInput, RepositoryContext, RepositoryMetadata, SnapshotState, SnapshotsService};
use serde_json::json;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;

async fn service(root: &Path) -> Arc<SnapshotsService> {
    let svc = SnapshotsService::new(RepositoryContext { path_repo: vec![root.join("repos")], ..Default::default() });
    svc.put_repository(RepositoryMetadata::new("r", "fs", json!({"location": "r"})), true).await.unwrap();
    svc
}

fn index(name: &str, dir: &Path, files: &[&str]) -> Vec<IndexSnapshotInput> {
    vec![IndexSnapshotInput { index_name: name.into(), index_metadata_json: index_meta(name, 1), shards: vec![shard(dir, 0, files)] }]
}

/// A hook that reports it was reached and waits to be released.
fn pause() -> (TestHook, Arc<Notify>, Arc<Notify>) {
    let (entered, release) = (Arc::new(Notify::new()), Arc::new(Notify::new()));
    let (e, r) = (entered.clone(), release.clone());
    let hook: TestHook = Arc::new(move || {
        let (e, r) = (e.clone(), r.clone());
        Box::pin(async move {
            e.notify_one();
            r.notified().await;
        })
    });
    (hook, entered, release)
}

async fn restore_matches(svc: &SnapshotsService, snap: &str, index: &str, src: &Path, files: &[&str], out: &Path) {
    let res = svc.restore_shard("r", snap, index, 0, out, &CancelToken::new()).await.unwrap_or_else(|e| panic!("restore {snap}: {e}"));
    assert_eq!(res.files as usize, files.len());
    assert_same_dir_files(src, files, out);
}

#[tokio::test]
async fn cleanup_after_its_idle_check_excludes_new_snapshots_and_clones() {
    let tmp = tempfile::tempdir().unwrap();
    let svc = service(tmp.path()).await;
    let a = tmp.path().join("a");
    write_file(&a, "seg", &random_bytes(20_000));
    svc.create_snapshot(request("r", "base", index("a", &a, &["seg"]))).await.unwrap();
    let b = tmp.path().join("b");
    write_file(&b, "seg", &random_bytes(30_000));
    write_file(&b, "segments_1", b"commit");

    let repo = svc.repository("r").unwrap();
    let (hook, entered, release) = pause();
    repo.set_test_hook("cleanup_admitted", Some(hook));
    let svc2 = svc.clone();
    let cleanup = tokio::spawn(async move { svc2.cleanup_repository("r").await });
    tokio::time::timeout(Duration::from_secs(10), entered.notified()).await.expect("cleanup reaches its pause point");

    // Cleanup has passed its idle check: a new snapshot (new index, whose
    // directory cleanup would consider unreferenced) and a clone are
    // rejected rather than racing it.
    let e = svc.create_snapshot(request("r", "new", index("b", &b, &["seg", "segments_1"]))).await.unwrap_err();
    assert_eq!(e.kind, "concurrent_snapshot_execution_exception", "{e}");
    assert_eq!(e.reason, "[r:new] cannot snapshot while a repository cleanup is in-progress in [r]");
    let e = svc.clone_snapshot("r", "base", "copy", &json!({"indices": "a"})).await.unwrap_err();
    assert_eq!(e.kind, "concurrent_snapshot_execution_exception", "{e}");

    release.notify_one();
    let r = tokio::time::timeout(Duration::from_secs(10), cleanup).await.unwrap().unwrap().unwrap();
    assert_eq!(r, json!({"results": {"deleted_bytes": 0, "deleted_blobs": 0}}));
    repo.set_test_hook("cleanup_admitted", None);

    // Admission is released: the snapshot runs, survives another cleanup,
    // and restores completely, as does the earlier one.
    let info = svc.create_snapshot(request("r", "new", index("b", &b, &["seg", "segments_1"]))).await.unwrap();
    assert_eq!(info.state, Some(SnapshotState::Success));
    svc.cleanup_repository("r").await.unwrap();
    restore_matches(&svc, "new", "b", &b, &["seg", "segments_1"], &tmp.path().join("out-new")).await;
    restore_matches(&svc, "base", "a", &a, &["seg"], &tmp.path().join("out-base")).await;
}

#[tokio::test]
async fn cleanup_is_refused_while_a_snapshot_runs() {
    let tmp = tempfile::tempdir().unwrap();
    let svc = SnapshotsService::new(RepositoryContext { path_repo: vec![tmp.path().join("repos")], ..Default::default() });
    // ~5s at 200kb/s: still running when cleanup is asked for.
    svc.put_repository(RepositoryMetadata::new("r", "fs", json!({"location": "r", "max_snapshot_bytes_per_sec": "200kb"})), true)
        .await
        .unwrap();
    let d = tmp.path().join("d");
    write_file(&d, "big", &random_bytes(1 << 20));
    let (_, handle) = svc.start_snapshot(request("r", "s", index("i", &d, &["big"]))).await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    let e = svc.cleanup_repository("r").await.unwrap_err();
    assert_eq!(e.reason, "Cannot cleanup [r] - a snapshot is currently running");
    let info = handle.await.unwrap().unwrap();
    assert_eq!(info.state, Some(SnapshotState::Success));
    svc.cleanup_repository("r").await.unwrap();
    restore_matches(&svc, "s", "i", &d, &["big"], &tmp.path().join("out")).await;
}

#[tokio::test]
async fn deleting_the_source_of_a_running_clone_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let svc = service(tmp.path()).await;
    let d = tmp.path().join("d");
    write_file(&d, "seg", &random_bytes(40_000));
    write_file(&d, "segments_1", b"commit");
    let files = ["seg", "segments_1"];
    svc.create_snapshot(request("r", "src", index("i", &d, &files))).await.unwrap();
    let blobs_before = data_blobs(&svc, "r").await;

    let repo = svc.repository("r").unwrap();
    let (hook, entered, release) = pause();
    repo.set_test_hook("clone_metadata_read", Some(hook));
    let svc2 = svc.clone();
    let clone = tokio::spawn(async move { svc2.clone_snapshot("r", "src", "copy", &json!({"indices": "i"})).await });
    tokio::time::timeout(Duration::from_secs(10), entered.notified()).await.expect("clone reaches its pause point");

    // The clone has read the source's metadata and written its shard
    // references; the source is its only owner of the files.
    let e = svc.delete_snapshots("r", &["src".into()]).await.unwrap_err();
    assert_eq!(e.kind, "concurrent_snapshot_execution_exception", "{e}");
    assert_eq!(e.reason, "[r:src] cannot delete snapshot while it is being cloned");
    // Wildcard deletes that match the source are rejected the same way.
    assert!(svc.delete_snapshots("r", &["*".into()]).await.is_err());

    release.notify_one();
    tokio::time::timeout(Duration::from_secs(10), clone).await.unwrap().unwrap().unwrap();
    repo.set_test_hook("clone_metadata_read", None);
    restore_matches(&svc, "copy", "i", &d, &files, &tmp.path().join("out1")).await;

    // Once the clone is done the source can go; the clone keeps every file.
    svc.delete_snapshots("r", &["src".into()]).await.unwrap();
    assert_eq!(data_blobs(&svc, "r").await, blobs_before);
    svc.cleanup_repository("r").await.unwrap();
    restore_matches(&svc, "copy", "i", &d, &files, &tmp.path().join("out2")).await;
}

#[tokio::test]
async fn clone_of_an_already_deleted_source_is_an_explicit_error() {
    let tmp = tempfile::tempdir().unwrap();
    let svc = service(tmp.path()).await;
    let d = tmp.path().join("d");
    write_file(&d, "seg", b"data");
    svc.create_snapshot(request("r", "src", index("i", &d, &["seg"]))).await.unwrap();
    svc.delete_snapshots("r", &["src".into()]).await.unwrap();
    let e = svc.clone_snapshot("r", "src", "copy", &json!({"indices": "i"})).await.unwrap_err();
    assert_eq!((e.kind, e.status), ("snapshot_missing_exception", 404), "{e}");
    assert!(data_blobs(&svc, "r").await.is_empty());
}
