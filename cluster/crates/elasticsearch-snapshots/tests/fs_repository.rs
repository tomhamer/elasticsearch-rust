mod common;

use common::*;
use elasticsearch_snapshots::blobstore::LZ4_MAGIC;
use elasticsearch_snapshots::repository::RepositoryData;
use elasticsearch_snapshots::{
    CancelToken, GetSnapshotsRequest, IndexSnapshotInput, Repository, RepositoryContext, RepositoryMetadata, RestoreRequest,
    ShardSnapshotInput, SnapshotState, SnapshotsService,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

fn service(root: &Path) -> Arc<SnapshotsService> {
    SnapshotsService::new(RepositoryContext { path_repo: vec![root.join("repos")], ..Default::default() })
}

async fn fs_repo(svc: &SnapshotsService, name: &str, settings: serde_json::Value) {
    svc.put_repository(RepositoryMetadata::new(name, "fs", settings), true).await.unwrap();
}

#[tokio::test]
async fn fs_full_cycle_chunked_compressed() {
    let tmp = tempfile::tempdir().unwrap();
    let svc = service(tmp.path());
    fs_repo(&svc, "backup", json!({"location": "backup", "compress": true, "chunk_size": "64kb"})).await;
    full_cycle(&svc, "backup", tmp.path(), 64 * 1024).await;
}

#[tokio::test]
async fn fs_full_cycle_uncompressed_unchunked() {
    let tmp = tempfile::tempdir().unwrap();
    let svc = service(tmp.path());
    // chunk_size bigger than any file → single-part blobs; full_cycle expects 4 parts,
    // so use a dedicated smaller check here instead.
    fs_repo(&svc, "plain", json!({"location": "plain", "compress": false})).await;
    let dir = tmp.path().join("d");
    write_file(&dir, "big", &random_bytes(300_000));
    let req = request(
        "plain",
        "s",
        vec![IndexSnapshotInput {
            index_name: "i".into(),
            index_metadata_json: index_meta("i", 1),
            shards: vec![shard(&dir, 0, &["big"])],
        }],
    );
    svc.create_snapshot(req).await.unwrap();
    // Filtering happens before total/pagination, including an empty result.
    for (state, count) in [(SnapshotState::Success, 1), (SnapshotState::Failed, 0)] {
        let filtered = svc.get_snapshots(&GetSnapshotsRequest {
            repositories: vec!["plain".into()], states: Some(vec![state]),
            size: Some(1), ..Default::default()
        }).await.unwrap();
        assert_eq!(filtered["snapshots"].as_array().unwrap().len(), count);
        assert_eq!(filtered["total"], count);
    }
    let blobs = data_blobs(&svc, "plain").await;
    assert_eq!(blobs.len(), 1);
    assert!(!blobs.keys().next().unwrap().contains(".part"));
    // Metadata blobs are plain JSON.
    let root = tmp.path().join("repos/plain");
    let idx = std::fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .find(|n| n.starts_with("index-"))
        .unwrap();
    let raw = std::fs::read(root.join(&idx)).unwrap();
    assert_eq!(raw[0], b'{');
    let rd: RepositoryData = serde_json::from_slice(&raw).unwrap();
    assert_eq!(rd.snapshots[0].name, "s");
    assert_eq!(rd.snapshots[0].state, 1);
    let latest = std::fs::read(root.join("index.latest")).unwrap();
    assert_eq!(format!("index-{}", i64::from_be_bytes(latest[..8].try_into().unwrap())), idx);

    // Compressed repository: metadata blobs carry the LZ4 header.
    fs_repo(&svc, "packed", json!({"location": "packed"})).await;
    svc.create_snapshot(request(
        "packed",
        "s",
        vec![IndexSnapshotInput { index_name: "i".into(), index_metadata_json: index_meta("i", 1), shards: vec![shard(&dir, 0, &["big"])] }],
    ))
    .await
    .unwrap();
    let raw = std::fs::read(tmp.path().join("repos/packed/index-0")).unwrap();
    assert!(raw.starts_with(LZ4_MAGIC));
    let out = tmp.path().join("restored");
    svc.restore_shard("packed", "s", "i", 0, &out, &CancelToken::new()).await.unwrap();
    assert_same_dir_files(&dir, &["big"], &out);
}

#[tokio::test]
async fn path_repo_validation_and_repository_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let svc = SnapshotsService::new(RepositoryContext::default());
    let e = svc
        .put_repository(RepositoryMetadata::new("r", "fs", json!({"location": "/tmp/x"})), true)
        .await
        .unwrap_err();
    assert_eq!(e.kind, "repository_exception");
    assert_eq!(
        e.reason,
        "[r] location [/tmp/x] doesn't match any of the locations specified by path.repo because this setting is empty"
    );
    let svc = service(tmp.path());
    let e = svc
        .put_repository(RepositoryMetadata::new("r", "fs", json!({"location": "/elsewhere"})), true)
        .await
        .unwrap_err();
    assert_eq!(
        e.reason,
        format!(
            "[r] location [/elsewhere] doesn't match any of the locations specified by path.repo: [{}]",
            tmp.path().join("repos").display()
        )
    );
    let e = svc.repository("nope").unwrap_err();
    assert_eq!((e.kind, e.reason.as_str(), e.status), ("repository_missing_exception", "[nope] missing", 404));
    fs_repo(&svc, "r1", json!({"location": "r1", "compress": "false"})).await;
    let g = svc.get_repositories_json(&[]).unwrap();
    assert_eq!(g, json!({"r1": {"type": "fs", "settings": {"location": "r1", "compress": "false"}}}));
    assert_eq!(svc.delete_repositories(&["r*".into()]).unwrap(), vec!["r1"]);
    assert!(svc.delete_repositories(&["r1".into()]).is_err());
}

#[tokio::test]
async fn concurrent_modification_is_detected() {
    let tmp = tempfile::tempdir().unwrap();
    let ctx = RepositoryContext { path_repo: vec![tmp.path().to_path_buf()], ..Default::default() };
    let meta = RepositoryMetadata::new("shared", "fs", json!({"location": "shared"}));
    // Two "nodes" with their own Repository instances over the same location.
    let a = Repository::open(meta.clone(), &ctx).await.unwrap();
    let b = Repository::open(meta, &ctx).await.unwrap();
    let da = a.repository_data().await.unwrap();
    let db = b.repository_data().await.unwrap();
    assert_eq!(da.generation, -1);
    assert_eq!(a.write_repository_data(&da, da.generation).await.unwrap(), 0);
    let e = b.write_repository_data(&db, db.generation).await.unwrap_err();
    assert_eq!(e.kind, "repository_exception");
    assert_eq!(
        e.reason,
        "[shared] concurrent modification of the index-N file, expected current generation [-1], actual current generation [0]"
    );
    // After re-reading, b can write.
    let db = b.repository_data().await.unwrap();
    assert_eq!(db.generation, 0);
    assert_eq!(b.write_repository_data(&db, 0).await.unwrap(), 1);
    // Old generation was removed, index.latest moved.
    assert_eq!(a.latest_generation().await.unwrap(), 1);
    assert!(!a.root().blob_exists("index-0").await.unwrap());
    // A stale writer that races on the same new generation also fails.
    let e = a.write_repository_data(&da, 0).await.unwrap_err();
    assert!(e.reason.contains("concurrent modification of the index-N file"));
}

#[tokio::test]
async fn readonly_and_url_repositories() {
    let tmp = tempfile::tempdir().unwrap();
    let svc = service(tmp.path());
    fs_repo(&svc, "rw", json!({"location": "loc"})).await;
    let dir = tmp.path().join("d");
    write_file(&dir, "f", b"hello");
    let one = || {
        vec![IndexSnapshotInput { index_name: "i".into(), index_metadata_json: index_meta("i", 1), shards: vec![shard(&dir, 0, &["f"])] }]
    };
    svc.create_snapshot(request("rw", "s1", one())).await.unwrap();

    fs_repo(&svc, "ro", json!({"location": "loc", "readonly": true})).await;
    let e = svc.create_snapshot(request("ro", "s2", one())).await.unwrap_err();
    assert_eq!(e.reason, "[ro] cannot create snapshot in a readonly repository");
    let e = svc.delete_snapshots("ro", &["s1".into()]).await.unwrap_err();
    assert_eq!(e.reason, "[ro] cannot delete snapshot in a readonly repository");

    let url = format!("file:{}", tmp.path().join("repos/loc").display());
    svc.put_repository(RepositoryMetadata::new("url", "url", json!({"url": url})), true).await.unwrap();
    let g = svc
        .get_snapshots(&GetSnapshotsRequest { repositories: vec!["url".into()], ..Default::default() })
        .await
        .unwrap();
    assert_eq!(g["snapshots"][0]["snapshot"], "s1");
    let out = tmp.path().join("out");
    svc.restore_shard("url", "s1", "i", 0, &out, &CancelToken::new()).await.unwrap();
    assert_eq!(std::fs::read(out.join("f")).unwrap(), b"hello");
    // Multi-repo get.
    let g = svc.get_snapshots(&GetSnapshotsRequest::default()).await.unwrap();
    assert_eq!(g["total"], 3);
}

#[tokio::test]
async fn abort_running_snapshot_and_current_status() {
    let tmp = tempfile::tempdir().unwrap();
    let svc = service(tmp.path());
    // 200kb/s with a 1mb file → ~5s; we abort long before.
    fs_repo(&svc, "slow", json!({"location": "slow", "max_snapshot_bytes_per_sec": "200kb"})).await;
    let dir = tmp.path().join("d");
    write_file(&dir, "big", &random_bytes(1 << 20));
    let req = request(
        "slow",
        "s",
        vec![IndexSnapshotInput { index_name: "i".into(), index_metadata_json: index_meta("i", 1), shards: vec![shard(&dir, 0, &["big"])] }],
    );
    let (running, handle) = svc.start_snapshot(req).await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(400)).await;

    let cur = svc.snapshot_status(Some("slow"), &[], false, false).await.unwrap();
    let s = &cur["snapshots"][0];
    assert_eq!(s["state"], "STARTED");
    assert_eq!(s["uuid"], running.uuid);
    assert_eq!(s["indices"]["i"]["shards"]["0"]["stage"], "STARTED");
    assert_eq!(s["indices"]["i"]["shards"]["0"]["node"], "node-1");
    let processed = s["stats"]["processed"]["size_in_bytes"].as_u64().unwrap();
    assert!(processed > 0 && processed < (1 << 20), "processed={processed}");

    let g = svc
        .get_snapshots(&GetSnapshotsRequest {
            repositories: vec!["slow".into()],
            snapshots: vec!["_current".into()],
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(g["snapshots"][0]["state"], "IN_PROGRESS");
    assert_eq!(g["snapshots"][0]["end_time_in_millis"], 0);

    // Same name while running.
    let e = svc
        .start_snapshot(request("slow", "s", vec![]))
        .await
        .err()
        .unwrap();
    assert_eq!(e.reason, "[slow:s] Invalid snapshot name [s], snapshot with the same name is already in-progress");
    // Cleanup refused while running.
    let e = svc.cleanup_repository("slow").await.unwrap_err();
    assert_eq!(e.reason, "Cannot cleanup [slow] - a snapshot is currently running");

    svc.delete_snapshots("slow", &["s".into()]).await.unwrap();
    let e = handle.await.unwrap().unwrap_err();
    assert_eq!(e.reason, "[slow:s] Snapshot was aborted by deletion");
    let g = svc
        .get_snapshots(&GetSnapshotsRequest { repositories: vec!["slow".into()], ..Default::default() })
        .await
        .unwrap();
    assert_eq!(g["total"], 0);
    assert_eq!(svc.snapshot_status(None, &[], false, false).await.unwrap(), json!({"snapshots": []}));
    // Nothing of the aborted snapshot survives a cleanup (fs writes are
    // atomic, so the interrupted part never became a blob).
    svc.cleanup_repository("slow").await.unwrap();
    assert!(data_blobs(&svc, "slow").await.is_empty());
    assert!(svc.repository("slow").unwrap().indices_container().children().await.unwrap().is_empty());
}

#[tokio::test]
async fn concurrent_snapshots_share_shards() {
    let tmp = tempfile::tempdir().unwrap();
    let svc = service(tmp.path());
    fs_repo(&svc, "r", json!({"location": "r"})).await;
    let dir = tmp.path().join("d");
    for i in 0..5 {
        write_file(&dir, &format!("f{i}"), &random_bytes(10_000 + i));
    }
    let files: Vec<String> = (0..5).map(|i| format!("f{i}")).collect();
    let mk = |name: &str| {
        request(
            "r",
            name,
            vec![IndexSnapshotInput {
                index_name: "i".into(),
                index_metadata_json: index_meta("i", 1),
                shards: vec![ShardSnapshotInput { shard: 0, dir: dir.clone(), files: files.clone().into(), unavailable: None, remote: None }],
            }],
        )
    };
    let (a, b, c) = tokio::join!(svc.create_snapshot(mk("a")), svc.create_snapshot(mk("b")), svc.create_snapshot(mk("c")));
    a.unwrap();
    b.unwrap();
    c.unwrap();
    // Files deduplicated across the three snapshots.
    assert_eq!(data_blobs(&svc, "r").await.len(), 5);
    for n in ["a", "b", "c"] {
        let out = tmp.path().join(format!("out-{n}"));
        svc.restore_shard("r", n, "i", 0, &out, &CancelToken::new()).await.unwrap();
        assert_same_dir_files(&dir, &files.iter().map(|s| s.as_str()).collect::<Vec<_>>(), &out);
    }
    svc.delete_snapshots("r", &["a".into(), "b".into()]).await.unwrap();
    assert_eq!(data_blobs(&svc, "r").await.len(), 5);
    let c = svc.cleanup_repository("r").await.unwrap();
    assert_eq!(c, json!({"results": {"deleted_bytes": 0, "deleted_blobs": 0}}));
}

#[tokio::test]
async fn partial_snapshots() {
    let tmp = tempfile::tempdir().unwrap();
    let svc = service(tmp.path());
    fs_repo(&svc, "r", json!({"location": "r"})).await;
    let dir = tmp.path().join("d");
    write_file(&dir, "ok", b"ok");
    let inputs = || {
        vec![IndexSnapshotInput {
            index_name: "i".into(),
            index_metadata_json: index_meta("i", 3),
            shards: vec![
                shard(&dir, 0, &["ok"]),
                ShardSnapshotInput { shard: 1, dir: dir.clone(), files: vec![].into(), unavailable: Some("primary shard is not allocated".into()), remote: None },
                // shard 2 fails while snapshotting (missing file)
                shard(&dir, 2, &["missing-file"]),
            ],
        }]
    };
    let e = svc.create_snapshot(request("r", "p", inputs())).await.unwrap_err();
    assert_eq!(e.reason, "[r:p] Indices don't have primary shards [i]");

    let mut req = request("r", "p", inputs());
    req.partial = true;
    let info = svc.create_snapshot(req).await.unwrap();
    assert_eq!(info.state, Some(SnapshotState::Partial));
    assert_eq!((info.total_shards, info.successful_shards), (3, 1));
    assert_eq!(info.failures.len(), 2);
    let f = info.failures[0].to_json();
    assert_eq!(f["index"], "i");
    assert_eq!(f["index_uuid"], "uuid-i");
    assert_eq!(f["shard_id"], 1);
    assert_eq!(f["reason"], "primary shard is not allocated");
    assert_eq!(f["node_id"], "node-1");
    assert_eq!(f["status"], "INTERNAL_SERVER_ERROR");

    let st = svc.snapshot_status(Some("r"), &["p".into()], false, false).await.unwrap();
    assert_eq!(st["snapshots"][0]["state"], "SUCCESS");
    assert_eq!(st["snapshots"][0]["shards_stats"]["failed"], 2);
    assert_eq!(st["snapshots"][0]["indices"]["i"]["shards"]["1"]["stage"], "FAILURE");

    let e = svc.plan_restore("r", "p", &RestoreRequest::from_json(None).unwrap(), &BTreeMap::new()).await.unwrap_err();
    assert!(e.reason.ends_with("index [i] wasn't fully snapshotted - cannot restore"), "{}", e.reason);
    let plan = svc
        .plan_restore("r", "p", &RestoreRequest::from_json(Some(&json!({"partial": true}))).unwrap(), &BTreeMap::new())
        .await
        .unwrap();
    assert_eq!(plan.indices[0].shards, vec![0]);
    assert_eq!(plan.indices[0].failed_shards, vec![1, 2]);
    // Clone of a partially snapshotted index is refused.
    let e = svc.clone_snapshot("r", "p", "p2", &json!({"indices": "i"})).await.unwrap_err();
    assert_eq!(e.reason, "[r:p2] Can't clone index [i] because its snapshot was not successful.");
}

#[tokio::test]
async fn restore_detects_corruption() {
    let tmp = tempfile::tempdir().unwrap();
    let svc = service(tmp.path());
    fs_repo(&svc, "r", json!({"location": "r"})).await;
    let dir = tmp.path().join("d");
    write_file(&dir, "f", b"original content");
    svc.create_snapshot(request(
        "r",
        "s",
        vec![IndexSnapshotInput { index_name: "i".into(), index_metadata_json: index_meta("i", 1), shards: vec![shard(&dir, 0, &["f"])] }],
    ))
    .await
    .unwrap();
    let blob = data_blobs(&svc, "r").await.into_keys().next().unwrap();
    std::fs::write(tmp.path().join("repos/r/indices").join(&blob), b"tampered content").unwrap();
    let e = svc.restore_shard("r", "s", "i", 0, &tmp.path().join("out"), &CancelToken::new()).await.unwrap_err();
    assert_eq!(e.kind, "corrupt_index_exception");
    assert!(!tmp.path().join("out/f").exists());
}
