//! Durability and restore drills for the `fs` repository:
//!
//! * fsync failures (injected through the store's sync hook) on directories
//!   (the root `index-N` / `index.latest` publication, shard directories)
//!   and on files fail the snapshot instead of reporting success, and the
//!   repository stays usable once the fault clears;
//! * restores from a damaged repository (truncated, flipped, missing data
//!   blobs, unreadable shard / root metadata) fail with a clear error and
//!   leave neither the target file nor temporary files behind.

mod common;

use common::*;
use elasticsearch_snapshots::blobstore::fs::FsBlobStore;
use elasticsearch_snapshots::repository::RepositorySettings;
use elasticsearch_snapshots::{
    CancelToken, GetSnapshotsRequest, IndexSnapshotInput, Repository, RepositoryContext, RepositoryMetadata, SnapshotState,
    SnapshotsService,
};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

const OK: u8 = 0;
const FAIL_DIRS: u8 = 1;
const FAIL_FILES: u8 = 2;

/// An fs repository whose fsyncs fail on demand (`mode`).
async fn faulty_repo(root: &Path, name: &str) -> (Arc<SnapshotsService>, Arc<AtomicU8>, PathBuf) {
    let svc = SnapshotsService::new(RepositoryContext::default());
    let mode = Arc::new(AtomicU8::new(OK));
    let location = root.join(name);
    let m = mode.clone();
    let store = FsBlobStore::open(location.clone(), false).await.unwrap().with_sync_hook(Arc::new(move |p: &Path| {
        let fail = match m.load(Ordering::SeqCst) {
            FAIL_DIRS => p.is_dir(),
            FAIL_FILES => p.is_file(),
            _ => false,
        };
        if fail { Err(std::io::Error::other(format!("injected fsync failure on [{}]", p.display()))) } else { Ok(()) }
    }));
    svc.register_repository(Repository::with_store(
        RepositoryMetadata::new(name, "fs", json!({"location": location.to_string_lossy()})),
        Arc::new(store),
        RepositorySettings::default(),
    ));
    (svc, mode, location)
}

fn one_index(dir: &Path, files: &[&str]) -> Vec<IndexSnapshotInput> {
    vec![IndexSnapshotInput { index_name: "i".into(), index_metadata_json: index_meta("i", 1), shards: vec![shard(dir, 0, files)] }]
}

async fn snapshot_names(svc: &SnapshotsService, repo: &str) -> Vec<String> {
    let g = svc.get_snapshots(&GetSnapshotsRequest { repositories: vec![repo.into()], ..Default::default() }).await.unwrap();
    g["snapshots"].as_array().unwrap().iter().map(|s| s["snapshot"].as_str().unwrap().to_string()).collect()
}

fn leftovers(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return out };
    for e in rd.flatten() {
        if e.file_type().unwrap().is_dir() {
            out.extend(leftovers(&e.path()));
        } else {
            let n = e.file_name().to_string_lossy().to_string();
            if n.starts_with("pending-") || n.starts_with("restore-") {
                out.push(e.path().display().to_string());
            }
        }
    }
    out
}

#[tokio::test]
async fn directory_fsync_failure_fails_the_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    let (svc, mode, location) = faulty_repo(tmp.path(), "r").await;
    let data = tmp.path().join("data");
    write_file(&data, "seg", &random_bytes(10_000));
    write_file(&data, "segments_1", b"commit");

    // A good snapshot first, so the repository has a generation to keep.
    let ok = svc.create_snapshot(request("r", "s1", one_index(&data, &["seg", "segments_1"]))).await.unwrap();
    assert_eq!(ok.state, Some(SnapshotState::Success));
    let latest_before = std::fs::read(location.join("index.latest")).unwrap();

    // Every directory fsync fails: the shard's blobs and the root
    // generation cannot be published durably, so the snapshot must not
    // report success and the repository generation must not advance.
    mode.store(FAIL_DIRS, Ordering::SeqCst);
    write_file(&data, "seg2", &random_bytes(5_000));
    let r = svc.create_snapshot(request("r", "s2", one_index(&data, &["seg", "seg2", "segments_1"]))).await;
    match &r {
        Err(e) => assert!(e.reason.contains("injected fsync failure"), "{e}"),
        Ok(info) => {
            assert_ne!(info.state, Some(SnapshotState::Success), "snapshot reported success despite fsync failures");
            assert!(info.failures.iter().any(|f| f.reason.contains("injected fsync failure")), "{:?}", info.failures);
        }
    }
    mode.store(OK, Ordering::SeqCst);
    assert_eq!(std::fs::read(location.join("index.latest")).unwrap(), latest_before, "generation advanced without a durable index-N");
    assert_eq!(snapshot_names(&svc, "r").await, vec!["s1"]);

    // Recovered: the repository is consistent and usable.
    let ok = svc.create_snapshot(request("r", "s3", one_index(&data, &["seg", "seg2", "segments_1"]))).await.unwrap();
    assert_eq!(ok.state, Some(SnapshotState::Success));
    let out = tmp.path().join("out");
    svc.restore_shard("r", "s3", "i", 0, &out, &CancelToken::new()).await.unwrap();
    assert_same_dir_files(&data, &["seg", "seg2", "segments_1"], &out);
    assert!(leftovers(&location).is_empty(), "{:?}", leftovers(&location));
}

#[tokio::test]
async fn root_metadata_fsync_failure_fails_finalization() {
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("data");
    write_file(&data, "f", b"content");
    // Only the repository root's fsync fails: the shard's blobs are
    // durable, but `index-N` / `index.latest` publication is not.
    let location = tmp.path().join("root");
    let root = location.clone();
    let svc = SnapshotsService::new(RepositoryContext::default());
    let store = FsBlobStore::open(location.clone(), false).await.unwrap().with_sync_hook(Arc::new(move |p: &Path| {
        if p == root.as_path() { Err(std::io::Error::other("injected fsync failure on the repository root")) } else { Ok(()) }
    }));
    svc.register_repository(Repository::with_store(
        RepositoryMetadata::new("root", "fs", json!({"location": location.to_string_lossy()})),
        Arc::new(store),
        RepositorySettings::default(),
    ));
    let e = svc.create_snapshot(request("root", "s", one_index(&data, &["f"]))).await.unwrap_err();
    assert!(e.reason.contains("injected fsync failure on the repository root"), "{e}");
    assert!(snapshot_names(&svc, "root").await.is_empty());
}

#[tokio::test]
async fn file_fsync_failure_fails_the_shard() {
    let tmp = tempfile::tempdir().unwrap();
    let (svc, mode, location) = faulty_repo(tmp.path(), "r").await;
    let data = tmp.path().join("data");
    write_file(&data, "seg", &random_bytes(4_000));
    mode.store(FAIL_FILES, Ordering::SeqCst);
    let r = svc.create_snapshot(request("r", "s", one_index(&data, &["seg"]))).await;
    mode.store(OK, Ordering::SeqCst);
    match r {
        Err(e) => assert!(e.reason.contains("injected fsync failure"), "{e}"),
        Ok(info) => {
            assert_ne!(info.state, Some(SnapshotState::Success));
            assert_eq!(info.successful_shards, 0);
            assert!(info.failures[0].reason.contains("injected fsync failure"), "{:?}", info.failures);
        }
    }
    // Temporary files of the failed writes are gone.
    assert!(leftovers(&location).is_empty(), "{:?}", leftovers(&location));
}

/// Snapshot one shard (two data blobs) into a plain fs repository.
async fn snapshotted(tmp: &Path) -> (Arc<SnapshotsService>, PathBuf, PathBuf) {
    let svc = SnapshotsService::new(RepositoryContext { path_repo: vec![tmp.join("repos")], ..Default::default() });
    svc.put_repository(RepositoryMetadata::new("r", "fs", json!({"location": "r"})), true).await.unwrap();
    let data = tmp.join("data");
    write_file(&data, "a", &random_bytes(20_000));
    write_file(&data, "b", &random_bytes(30_000));
    let info = svc.create_snapshot(request("r", "s", one_index(&data, &["a", "b"]))).await.unwrap();
    assert_eq!(info.state, Some(SnapshotState::Success));
    (svc, data, tmp.join("repos/r"))
}

async fn restore_err(svc: &SnapshotsService, out: &Path) -> elasticsearch_snapshots::SnapshotError {
    let e = svc.restore_shard("r", "s", "i", 0, out, &CancelToken::new()).await.unwrap_err();
    assert!(leftovers(out).is_empty(), "temporary restore files left: {:?}", leftovers(out));
    e
}

#[tokio::test]
async fn restore_drill_damaged_repository_fails_clearly() {
    let tmp = tempfile::tempdir().unwrap();
    let (svc, data, repo) = snapshotted(tmp.path()).await;
    let blobs: Vec<String> = data_blobs(&svc, "r").await.into_keys().collect();
    assert_eq!(blobs.len(), 2);
    let blob_path = |k: &str| repo.join("indices").join(k);

    // Partial (truncated) data blob.
    let victim = blob_path(&blobs[0]);
    let original = std::fs::read(&victim).unwrap();
    std::fs::write(&victim, &original[..original.len() / 2]).unwrap();
    let out = tmp.path().join("out1");
    let e = restore_err(&svc, &out).await;
    assert!(matches!(e.kind, "corrupt_index_exception" | "i_o_exception"), "{e}");
    assert!(e.reason.contains("unexpected end of blob") || e.reason.contains("checksum failed"), "{e}");

    // Flipped byte: same length, wrong checksum.
    let mut flipped = original.clone();
    flipped[original.len() / 3] ^= 0xff;
    std::fs::write(&victim, &flipped).unwrap();
    let out = tmp.path().join("out2");
    let e = restore_err(&svc, &out).await;
    assert_eq!(e.kind, "corrupt_index_exception", "{e}");
    assert!(e.reason.contains("checksum failed"), "{e}");

    // Missing data blob.
    std::fs::remove_file(&victim).unwrap();
    let out = tmp.path().join("out3");
    let e = restore_err(&svc, &out).await;
    assert!(e.is_not_found(), "{e}");
    std::fs::write(&victim, &original).unwrap();

    // Unreadable shard-level metadata (`snap-{uuid}.dat`).
    let shard_dir = victim.parent().unwrap().to_path_buf();
    let snap_meta = std::fs::read_dir(&shard_dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.file_name().unwrap().to_string_lossy().starts_with("snap-"))
        .unwrap();
    let meta_bytes = std::fs::read(&snap_meta).unwrap();
    std::fs::write(&snap_meta, b"{not json").unwrap();
    let out = tmp.path().join("out4");
    let e = restore_err(&svc, &out).await;
    assert!(!e.reason.is_empty(), "{e}");
    std::fs::write(&snap_meta, &meta_bytes).unwrap();

    // A healthy repository again restores every file intact.
    let out = tmp.path().join("out5");
    svc.restore_shard("r", "s", "i", 0, &out, &CancelToken::new()).await.unwrap();
    assert_same_dir_files(&data, &["a", "b"], &out);

    // Damaged root generation: listing / restoring fails with an error, not a hang or a panic.
    let generation = std::fs::read_dir(&repo)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .find(|n| n.starts_with("index-"))
        .unwrap();
    std::fs::write(repo.join(&generation), b"\x00garbage").unwrap();
    let svc2 = SnapshotsService::new(RepositoryContext { path_repo: vec![tmp.path().join("repos")], ..Default::default() });
    let _ = svc2.put_repository(RepositoryMetadata::new("r", "fs", json!({"location": "r"})), false).await;
    let e = svc2.restore_shard("r", "s", "i", 0, &tmp.path().join("out6"), &CancelToken::new()).await.unwrap_err();
    assert!(!e.reason.is_empty(), "{e}");
    assert!(!tmp.path().join("out6/a").exists());
}

#[tokio::test]
async fn restore_cancellation_stops_and_cleans_up() {
    let tmp = tempfile::tempdir().unwrap();
    let svc = SnapshotsService::new(RepositoryContext { path_repo: vec![tmp.path().join("repos")], ..Default::default() });
    // 100kb/s restore of 1mb: ~10s unless cancelled.
    svc.put_repository(RepositoryMetadata::new("r", "fs", json!({"location": "r", "max_restore_bytes_per_sec": "100kb"})), true)
        .await
        .unwrap();
    let data = tmp.path().join("data");
    write_file(&data, "big", &random_bytes(1 << 20));
    svc.create_snapshot(request("r", "s", one_index(&data, &["big"]))).await.unwrap();
    let cancel = CancelToken::new();
    let c2 = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        c2.cancel("restore aborted");
    });
    let out = tmp.path().join("out");
    let started = std::time::Instant::now();
    let e = svc.restore_shard("r", "s", "i", 0, &out, &cancel).await.unwrap_err();
    assert_eq!((e.kind, e.reason.as_str()), ("task_cancelled_exception", "restore aborted"));
    assert!(started.elapsed() < std::time::Duration::from_secs(5), "{:?}", started.elapsed());
    assert!(!out.join("big").exists());
    assert!(leftovers(&out).is_empty(), "{:?}", leftovers(&out));
}

/// Production review #133: a snapshot deletion whose root `index-N` never
/// publishes (its temp file's fsync fails) leaves the committed catalog
/// listing the snapshot, and nothing of the failed transaction may become
/// current: a later successful snapshot and a cleanup keep the snapshot
/// restorable.
#[tokio::test]
async fn a_failed_delete_does_not_contaminate_the_next_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    let location = tmp.path().join("r");
    let root = location.clone();
    let fail_root_publication = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let armed = fail_root_publication.clone();
    let svc = SnapshotsService::new(RepositoryContext::default());
    let store = FsBlobStore::open(location.clone(), false).await.unwrap().with_sync_hook(Arc::new(move |p: &Path| {
        let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        if armed.load(Ordering::SeqCst) && p.parent() == Some(root.as_path()) && name.starts_with("pending-index-") {
            return Err(std::io::Error::other("injected fsync failure before the root generation was published"));
        }
        Ok(())
    }));
    svc.register_repository(Repository::with_store(
        RepositoryMetadata::new("r", "fs", json!({"location": location.to_string_lossy()})),
        Arc::new(store),
        RepositorySettings::default(),
    ));
    let data = tmp.path().join("data");
    write_file(&data, "seg_a", &random_bytes(8_000));
    write_file(&data, "seg_b", &random_bytes(8_000));
    write_file(&data, "segments_1", b"commit");
    svc.create_snapshot(request("r", "s0", one_index(&data, &["seg_a", "segments_1"]))).await.unwrap();
    svc.create_snapshot(request("r", "s1", one_index(&data, &["seg_a", "seg_b", "segments_1"]))).await.unwrap();
    fail_root_publication.store(true, Ordering::SeqCst);
    let e = svc.delete_snapshots("r", &["s1".into()]).await.unwrap_err();
    assert!(e.reason.contains("injected fsync failure"), "{e}");
    fail_root_publication.store(false, Ordering::SeqCst);
    assert_eq!(snapshot_names(&svc, "r").await, vec!["s0", "s1"], "the delete did not commit");
    // Same repository instance: a successful snapshot, then a cleanup.
    let s2 = svc.create_snapshot(request("r", "s2", one_index(&data, &["seg_a", "segments_1"]))).await.unwrap();
    assert_eq!(s2.state, Some(SnapshotState::Success));
    svc.cleanup_repository("r").await.unwrap();
    assert_eq!(snapshot_names(&svc, "r").await, vec!["s0", "s1", "s2"]);
    let out = tmp.path().join("out1");
    svc.restore_shard("r", "s1", "i", 0, &out, &CancelToken::new()).await.expect("s1 is still restorable");
    assert_same_dir_files(&data, &["seg_a", "seg_b", "segments_1"], &out);
    // The delete retried now commits and removes s1's unique data.
    svc.delete_snapshots("r", &["s1".into()]).await.unwrap();
    let out = tmp.path().join("out2");
    svc.restore_shard("r", "s2", "i", 0, &out, &CancelToken::new()).await.unwrap();
    assert_same_dir_files(&data, &["seg_a", "segments_1"], &out);
}

/// Production review #134: a clone of a snapshot holding a data stream
/// needs the stream's definition from the source's root metadata. A failed
/// read fails the clone (nothing published); once the read works the
/// clone carries the definition. A clone without data streams stays
/// independent of the source's global metadata.
#[tokio::test]
async fn a_data_stream_clone_fails_when_its_metadata_cannot_be_read() {
    let tmp = tempfile::tempdir().unwrap();
    let location = tmp.path().join("r");
    let svc = SnapshotsService::new(RepositoryContext::default());
    svc.register_repository(Repository::with_store(
        RepositoryMetadata::new("r", "fs", json!({"location": location.to_string_lossy()})),
        Arc::new(FsBlobStore::open(location.clone(), false).await.unwrap()),
        RepositorySettings::default(),
    ));
    let data = tmp.path().join("data");
    write_file(&data, "seg", &random_bytes(4_000));
    let backing = ".ds-logs-2026.11.09-000001";
    let definition = json!({"name": "logs", "indices": [{"index_name": backing, "index_uuid": "u"}], "generation": 1});
    let mut req = request("r", "src", vec![IndexSnapshotInput { index_name: backing.into(), index_metadata_json: index_meta(backing, 1), shards: vec![shard(&data, 0, &["seg"])] }]);
    req.global_metadata_json = Some(json!({"customs": {"data_stream": {"logs": definition}}}));
    req.data_streams = vec!["logs".into()];
    let src = svc.create_snapshot(req).await.unwrap();
    assert_eq!(src.state, Some(SnapshotState::Success));
    let meta = location.join(format!("meta-{}.dat", src.uuid));
    let hidden = location.join("hidden-meta");
    // The source's root metadata cannot be read: the clone fails.
    std::fs::rename(&meta, &hidden).unwrap();
    let e = svc.clone_snapshot("r", "src", "copy", &json!({"indices": "*"})).await.unwrap_err();
    assert!(e.reason.contains("metadata"), "{e}");
    assert_eq!(snapshot_names(&svc, "r").await, vec!["src"], "no clone published");
    // Also with a plain index only (no data stream): independent of it.
    write_file(&data, "plain", &random_bytes(1_000));
    let plain = svc.create_snapshot(request("r", "plain", one_index(&data, &["plain"]))).await.unwrap();
    let plain_meta = location.join(format!("meta-{}.dat", plain.uuid));
    std::fs::rename(&plain_meta, location.join("hidden-plain")).unwrap();
    svc.clone_snapshot("r", "plain", "plain-copy", &json!({"indices": "*"})).await.unwrap();
    std::fs::rename(location.join("hidden-plain"), &plain_meta).unwrap();
    // Readable again: the clone carries the stream's definition.
    std::fs::rename(&hidden, &meta).unwrap();
    svc.clone_snapshot("r", "src", "copy", &json!({"indices": "*"})).await.unwrap();
    let repo = svc.repository("r").unwrap();
    let catalog = repo.repository_data().await.unwrap();
    let copy = catalog.snapshot_by_name("copy").unwrap();
    let g = repo.read_global_metadata(&copy.uuid).await.unwrap();
    assert_eq!(g["customs"]["data_stream"]["logs"], definition, "{g}");
    let info = repo.read_snapshot_info(&copy.uuid).await.unwrap();
    assert_eq!(info.data_streams, vec!["logs".to_string()]);
}

/// Production review #132: two attempts of one snapshot (a deposed master's
/// and its successor's, each with its own repository instance and its own
/// image of the shard) never leave a descriptor describing another image
/// than the one the published generation lists: the later attempt adopts
/// the published descriptor. After the deposed master finalizes and a
/// cleanup runs, the SUCCESS snapshot restores the image it published.
#[tokio::test]
async fn a_second_attempt_of_a_snapshot_adopts_the_published_shard_image() {
    use elasticsearch_snapshots::repository::shard::ShardSnapshotTask;
    use elasticsearch_snapshots::repository::{FinalizeIndex, FinalizeRequest};
    use elasticsearch_snapshots::snapshot::{ShardStage, ShardStatus};
    let tmp = tempfile::tempdir().unwrap();
    let location = tmp.path().join("r");
    async fn open(location: &Path) -> Arc<Repository> {
        Repository::with_store(
            RepositoryMetadata::new("r", "fs", json!({"location": location.to_string_lossy()})),
            Arc::new(FsBlobStore::open(location.to_path_buf(), false).await.unwrap()),
            RepositorySettings::default(),
        )
    }
    let (master_a, master_b) = (open(&location).await, open(&location).await);
    let (dir_a, dir_b) = (tmp.path().join("a"), tmp.path().join("b"));
    write_file(&dir_a, "seg", &random_bytes(20_000));
    write_file(&dir_a, "segments_1", b"commit a");
    // Later writes: another image of the same shard.
    write_file(&dir_b, "seg", &random_bytes(20_000));
    write_file(&dir_b, "segments_1", b"commit b");
    let files = ["seg", "segments_1"];
    let uuid = "same-snapshot-uuid".to_string();
    let index_id = "index-id-i".to_string();
    let task = |dir: &Path| ShardSnapshotTask {
        snapshot_name: "s".into(),
        snapshot_uuid: uuid.clone(),
        index_id: index_id.clone(),
        shard: 0,
        dir: dir.to_path_buf(),
        files: files.iter().map(|f| f.to_string()).collect::<Vec<_>>().into(),
        status: Arc::new(parking_lot::Mutex::new(ShardStatus::new(ShardStage::Init))),
        cancel: CancelToken::new(),
    };
    let data_a = master_a.repository_data().await.unwrap();
    let ra = master_a.snapshot_shard(&data_a, task(&dir_a)).await.unwrap();
    // The successor resumes the same snapshot before the deposed master finalizes.
    let data_b = master_b.repository_data().await.unwrap();
    let rb = master_b.snapshot_shard(&data_b, task(&dir_b)).await.unwrap();
    master_b.verify_shard_snapshot(&index_id, 0, &uuid, &rb.new_gen).await.unwrap();
    master_a.verify_shard_snapshot(&index_id, 0, &uuid, &ra.new_gen).await.expect("the published generation still matches the descriptor");
    let request = || FinalizeRequest {
        info: elasticsearch_snapshots::SnapshotInfo {
            repository: "r".into(),
            snapshot: "s".into(),
            uuid: uuid.clone(),
            version_id: None,
            version: None,
            indices: vec!["i".into()],
            data_streams: vec![],
            feature_states: vec![],
            include_global_state: Some(false),
            metadata: None,
            state: Some(SnapshotState::Success),
            reason: None,
            start_time: 1,
            end_time: 2,
            total_shards: 1,
            successful_shards: 1,
            failures: vec![],
            index_details: Default::default(),
        },
        indices: vec![FinalizeIndex { name: "i".into(), index_id: index_id.clone(), metadata: index_meta("i", 1), shard_count: 1 }],
        global_metadata: None,
        obsolete_shard_gens: vec![],
    };
    // The deposed master publishes first; the successor finds it published.
    master_a.finalize_snapshot(request()).await.unwrap();
    assert!(master_b.finalize_snapshot(request()).await.is_err());
    master_a.cleanup().await.unwrap();
    let out = tmp.path().join("out");
    master_a.restore_shard(&uuid, &index_id, 0, &out, &CancelToken::new()).await.expect("the SUCCESS snapshot restores");
    assert_same_dir_files(&dir_a, &files, &out);
}

/// Production review #131: a snapshot is finalized with the index metadata
/// current once every shard image is captured (ES `metadataForSnapshot`),
/// so a field mapped while the snapshot ran (and indexed into the frozen
/// image) is in the snapshot's mapping. An index re-created meanwhile (a
/// new UUID) fails its shards instead of pairing images and metadata of
/// different incarnations.
#[tokio::test]
async fn a_snapshot_publishes_the_mapping_current_after_its_images() {
    use elasticsearch_snapshots::service::IndexMetadataSource;
    let tmp = tempfile::tempdir().unwrap();
    let location = tmp.path().join("r");
    let svc = SnapshotsService::new(RepositoryContext::default());
    svc.register_repository(Repository::with_store(
        RepositoryMetadata::new("r", "fs", json!({"location": location.to_string_lossy()})),
        Arc::new(FsBlobStore::open(location.clone(), false).await.unwrap()),
        RepositorySettings::default(),
    ));
    let data = tmp.path().join("data");
    write_file(&data, "seg", &random_bytes(2_000));
    // Captured at the start: `f` only. Mapped meanwhile: `late`.
    let mut grown = index_meta("i", 1);
    grown["mappings"]["properties"]["late"] = json!({"type": "date"});
    let current = grown.clone();
    let mut req = request("r", "s", one_index(&data, &["seg"]));
    req.current_index_metadata = Some(IndexMetadataSource(Arc::new(move |name: &str| (name == "i").then(|| current.clone()))));
    let info = svc.create_snapshot(req).await.unwrap();
    assert_eq!(info.state, Some(SnapshotState::Success));
    let repo = svc.repository("r").unwrap();
    let catalog = repo.repository_data().await.unwrap();
    let stored = repo.read_index_metadata(&catalog, &info.uuid, "i").await.unwrap();
    assert_eq!(stored["mappings"]["properties"]["late"], json!({"type": "date"}), "{stored}");
    // Re-created under the name (another UUID): the shards fail.
    let mut other = index_meta("i", 1);
    other["settings"]["index"]["uuid"] = json!("another-incarnation");
    let mut req = request("r", "s2", one_index(&data, &["seg"]));
    req.current_index_metadata = Some(IndexMetadataSource(Arc::new(move |_: &str| Some(other.clone()))));
    match svc.create_snapshot(req).await {
        Ok(info) => {
            assert_ne!(info.state, Some(SnapshotState::Success), "{info:?}");
            assert_eq!(info.successful_shards, 0);
            assert!(info.failures[0].reason.contains("re-created"), "{:?}", info.failures);
        }
        Err(e) => assert!(e.reason.contains("re-created"), "{e}"),
    }
}
