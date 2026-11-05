//! Shared snapshot/restore cycle, run against any repository type.
#![allow(dead_code)]

use bytes::Bytes;
use elasticsearch_snapshots::blobstore::BlobContainer;
use elasticsearch_snapshots::{
    CancelToken, CreateSnapshotRequest, ExistingIndex, GetSnapshotsRequest, IndexSnapshotInput, RestoreRequest,
    ShardSnapshotInput, SnapshotState, SnapshotsService, SortBy,
};
use rand::RngCore;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub fn random_bytes(n: usize) -> Vec<u8> {
    let mut v = vec![0u8; n];
    rand::thread_rng().fill_bytes(&mut v);
    v
}

pub fn write_file(dir: &Path, name: &str, data: &[u8]) {
    let p = dir.join(name);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, data).unwrap();
}

pub fn shard(dir: &Path, shard: u32, files: &[&str]) -> ShardSnapshotInput {
    ShardSnapshotInput { shard, dir: dir.to_path_buf(), files: files.iter().map(|s| s.to_string()).collect::<Vec<_>>().into(), unavailable: None, remote: None }
}

pub fn index_meta(name: &str, shards: u32) -> Value {
    json!({
        "settings": {"index": {"number_of_shards": shards.to_string(), "number_of_replicas": "1", "uuid": format!("uuid-{name}")}},
        "mappings": {"properties": {"f": {"type": "keyword"}}},
        "aliases": {format!("{name}-alias"): {}}
    })
}

pub fn request(repo: &str, snap: &str, indices: Vec<IndexSnapshotInput>) -> CreateSnapshotRequest {
    CreateSnapshotRequest {
        repository: repo.into(),
        snapshot: snap.into(),
        indices,
        global_metadata_json: Some(json!({"persistent_settings": {"a": "b"}, "templates": {}})),
        include_global_state: true,
        node_id: Some("node-1".into()),
        ..Default::default()
    }
}

/// Count data blobs (`__*`) under `indices/`.
pub async fn data_blobs(svc: &SnapshotsService, repo: &str) -> BTreeMap<String, u64> {
    let r = svc.repository(repo).unwrap();
    let indices = r.indices_container();
    let mut out = BTreeMap::new();
    for id in indices.children().await.unwrap() {
        let ic = indices.child(&id);
        for shard in ic.children().await.unwrap() {
            let sc = ic.child(&shard);
            for (n, s) in sc.list_blobs_by_prefix("__").await.unwrap() {
                out.insert(format!("{id}/{shard}/{n}"), s);
            }
        }
    }
    out
}

pub fn assert_same_dir_files(expected_dir: &Path, files: &[&str], restored: &Path) {
    for f in files {
        let a = std::fs::read(expected_dir.join(f)).unwrap();
        let b = std::fs::read(restored.join(f)).unwrap_or_else(|_| panic!("missing restored file {f}"));
        assert!(a == b, "file {f} differs after restore");
    }
    let mut n = 0;
    for e in walk(restored) {
        let _ = e;
        n += 1;
    }
    assert_eq!(n, files.len(), "restored dir has unexpected files");
}

fn walk(p: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(p).unwrap() {
        let e = e.unwrap();
        if e.file_type().unwrap().is_dir() {
            out.extend(walk(&e.path()));
        } else {
            out.push(e.path());
        }
    }
    out
}

fn get_req(snapshots: &[&str]) -> GetSnapshotsRequest {
    GetSnapshotsRequest { snapshots: snapshots.iter().map(|s| s.to_string()).collect(), ..Default::default() }
}

/// The full cycle. `chunk` = repository chunk_size in bytes (to size files so they are chunked).
pub async fn full_cycle(svc: &Arc<SnapshotsService>, repo: &str, work: &Path, chunk: usize) {
    // ---------- source data: idx1 (2 shards), idx2 (1 shard) ----------
    let s10 = work.join("data/idx1/0");
    let s11 = work.join("data/idx1/1");
    let s20 = work.join("data/idx2/0");
    write_file(&s10, "_0.cfs", &random_bytes(chunk * 3 + 17)); // 4 parts
    write_file(&s10, "_0.si", b"segment info 0");
    write_file(&s10, "segments_1", b"commit 1");
    write_file(&s11, "a.dat", &random_bytes(1000));
    write_file(&s11, "b.dat", &random_bytes(2000));
    write_file(&s20, "x/nested.bin", &random_bytes(500));
    write_file(&s20, "empty", b"");

    let idx1 = |files0: &[&str], files1: &[&str]| IndexSnapshotInput {
        index_name: "idx1".into(),
        index_metadata_json: index_meta("idx1", 2),
        shards: vec![shard(&s10, 0, files0), shard(&s11, 1, files1)],
    };
    let idx2 = || IndexSnapshotInput {
        index_name: "idx2".into(),
        index_metadata_json: index_meta("idx2", 1),
        shards: vec![shard(&s20, 0, &["x/nested.bin", "empty"])],
    };

    // ---------- snapshot 1 ----------
    let mut req = request(repo, "snap-1", vec![idx1(&["_0.cfs", "_0.si", "segments_1"], &["a.dat", "b.dat"]), idx2()]);
    req.metadata = Some(json!({"taken_by": "test"}).as_object().unwrap().clone());
    let info1 = svc.create_snapshot(req).await.unwrap();
    assert_eq!(info1.state, Some(SnapshotState::Success), "shard failures: {:?}", info1.failures);
    assert_eq!(info1.indices, vec!["idx1", "idx2"]);
    assert_eq!((info1.total_shards, info1.successful_shards), (3, 3));
    let blobs1 = data_blobs(svc, repo).await;
    // 4 parts for _0.cfs + 2 small + 2 + 2 (incl. empty file) = 10
    assert_eq!(blobs1.len(), 10, "{blobs1:?}");
    assert_eq!(blobs1.keys().filter(|k| k.contains(".part")).count(), 4);

    // Same name again → ES error.
    let e = svc.create_snapshot(request(repo, "snap-1", vec![idx2()])).await.unwrap_err();
    assert_eq!(e.kind, "invalid_snapshot_name_exception");
    assert_eq!(e.reason, format!("[{repo}:snap-1] Invalid snapshot name [snap-1], snapshot with the same name already exists"));

    // ---------- snapshot 2: incremental ----------
    write_file(&s10, "_1.cfs", &random_bytes(777));
    write_file(&s10, "segments_2", b"commit 2");
    write_file(&s11, "c.dat", &random_bytes(3000));
    let info2 = svc
        .create_snapshot(request(
            repo,
            "snap-2",
            vec![idx1(&["_0.cfs", "_0.si", "_1.cfs", "segments_2"], &["a.dat", "c.dat"]), idx2()],
        ))
        .await
        .unwrap();
    assert_eq!(info2.state, Some(SnapshotState::Success));
    let blobs2 = data_blobs(svc, repo).await;
    // Only _1.cfs, segments_2 and c.dat were uploaded.
    assert_eq!(blobs2.len(), blobs1.len() + 3, "incremental snapshot uploaded too much");
    for k in blobs1.keys() {
        assert!(blobs2.contains_key(k));
    }

    // ---------- status shape ----------
    let st = svc.snapshot_status(Some(repo), &["snap-2".into()], false, false).await.unwrap();
    let s = &st["snapshots"][0];
    let keys: Vec<&String> = s.as_object().unwrap().keys().collect();
    assert_eq!(
        keys,
        vec!["snapshot", "repository", "uuid", "state", "include_global_state", "shards_stats", "stats", "indices"]
    );
    assert_eq!(s["state"], "SUCCESS");
    assert_eq!(s["shards_stats"], json!({"initializing":0,"started":0,"finalizing":0,"done":3,"failed":0,"total":3}));
    assert_eq!(s["stats"]["incremental"]["file_count"], 3);
    assert_eq!(s["stats"]["incremental"]["size_in_bytes"], 777 + 8 + 3000);
    assert_eq!(s["stats"]["total"]["file_count"], 8);
    assert!(s["stats"].get("processed").is_none());
    assert_eq!(s["stats"]["start_time_in_millis"], info2.start_time);
    let sh = &s["indices"]["idx1"]["shards"]["0"];
    assert_eq!(sh["stage"], "DONE");
    assert_eq!(sh["stats"]["incremental"]["file_count"], 2);
    assert_eq!(sh["stats"]["total"]["file_count"], 4);
    assert_eq!(s["indices"]["idx2"]["stats"]["incremental"]["file_count"], 0);
    let e = svc.snapshot_status(Some(repo), &["nope".into()], false, false).await.unwrap_err();
    assert_eq!(e.reason, format!("[{repo}:nope] is missing"));
    assert_eq!(svc.snapshot_status(Some(repo), &["nope".into()], true, false).await.unwrap(), json!({"snapshots":[]}));

    // ---------- get shapes ----------
    let all = svc.get_snapshots(&get_req(&["_all"])).await.unwrap();
    assert_eq!(all["total"], 2);
    assert_eq!(all["remaining"], 0);
    let first = &all["snapshots"][0];
    assert_eq!(first["snapshot"], "snap-1");
    assert_eq!(first["repository"], repo);
    assert_eq!(first["metadata"], json!({"taken_by":"test"}));
    assert_eq!(first["shards"], json!({"total":3,"failed":0,"successful":3}));
    assert_eq!(first["include_global_state"], true);
    assert_eq!(first["failures"], json!([]));
    assert_eq!(first["feature_states"], json!([]));
    assert!(first["version"].is_string());
    let star = svc.get_snapshots(&get_req(&["snap-*"])).await.unwrap();
    assert_eq!(star["total"], 2);
    let mut r = get_req(&["*"]);
    r.sort = SortBy::Name;
    r.desc = true;
    r.size = Some(1);
    let page1 = svc.get_snapshots(&r).await.unwrap();
    assert_eq!(page1["snapshots"][0]["snapshot"], "snap-2");
    assert_eq!((page1["total"].clone(), page1["remaining"].clone()), (json!(2), json!(1)));
    let next = page1["next"].as_str().unwrap().to_string();
    r.after = Some(next);
    let page2 = svc.get_snapshots(&r).await.unwrap();
    assert_eq!(page2["snapshots"][0]["snapshot"], "snap-1");
    assert_eq!(page2["remaining"], 0);
    assert!(page2.get("next").is_none());
    let mut r = get_req(&["_all"]);
    r.offset = 1;
    assert_eq!(svc.get_snapshots(&r).await.unwrap()["snapshots"][0]["snapshot"], "snap-2");
    let mut r = get_req(&["_all"]);
    r.verbose = false;
    let nv = svc.get_snapshots(&r).await.unwrap();
    assert_eq!(
        nv["snapshots"][0],
        json!({"snapshot":"snap-1","uuid":info1.uuid,"repository":repo,"indices":["idx1","idx2"],"state":"SUCCESS"})
    );
    let mut r = get_req(&["snap-1"]);
    r.include_repository = false;
    r.index_names = false;
    let g = svc.get_snapshots(&r).await.unwrap();
    assert!(g["snapshots"][0].get("repository").is_none());
    assert!(g["snapshots"][0].get("indices").is_none());
    let e = svc.get_snapshots(&get_req(&["missing"])).await.unwrap_err();
    assert_eq!((e.kind, e.status), ("snapshot_missing_exception", 404));
    let mut r = get_req(&["missing", "snap-1"]);
    r.ignore_unavailable = true;
    assert_eq!(svc.get_snapshots(&r).await.unwrap()["total"], 1);
    let mut r = get_req(&["_all"]);
    r.sort = SortBy::ShardCount;
    r.desc = true;
    assert_eq!(svc.get_snapshots(&r).await.unwrap()["total"], 2);

    // ---------- restore ----------
    let rdir = work.join("restore/snap1-idx1-0");
    let res = svc.restore_shard(repo, "snap-1", "idx1", 0, &rdir, &CancelToken::new()).await.unwrap();
    assert_eq!(res.files, 3);
    // Every file is listed with the CRC-32 of the bytes written.
    assert_eq!(res.restored.len(), 3);
    for f in &res.restored {
        let bytes = std::fs::read(rdir.join(&f.physical_name)).unwrap();
        assert_eq!((f.length, f.crc32), (bytes.len() as u64, elasticsearch_snapshots::util::crc32(&bytes)), "{}", f.physical_name);
    }
    // A second restore into the same directory keeps the verified files
    // and still reports their checksums.
    let again = svc.restore_shard(repo, "snap-1", "idx1", 0, &rdir, &CancelToken::new()).await.unwrap();
    assert_eq!(again.reused_files, 3);
    assert_eq!(again.restored, res.restored);
    // _0.cfs + _0.si + segments_1 from the original dir (segments_1 still exists there).
    assert_same_dir_files(&s10, &["_0.cfs", "_0.si", "segments_1"], &rdir);
    let rdir2 = work.join("restore/snap2-idx2-0");
    svc.restore_shard(repo, "snap-2", "idx2", 0, &rdir2, &CancelToken::new()).await.unwrap();
    assert_same_dir_files(&s20, &["x/nested.bin", "empty"], &rdir2);

    // Metadata helpers.
    let im = svc.read_index_metadata(repo, "snap-1", "idx1").await.unwrap();
    assert_eq!(im, index_meta("idx1", 2));
    let gm = svc.read_global_metadata(repo, "snap-1").await.unwrap();
    assert_eq!(gm["persistent_settings"]["a"], "b");

    // Restore planning.
    let mut cluster = BTreeMap::new();
    cluster.insert("idx1".to_string(), ExistingIndex { open: true, number_of_shards: 2 });
    let e = svc.plan_restore(repo, "snap-1", &RestoreRequest::from_json(None).unwrap(), &cluster).await.unwrap_err();
    assert_eq!(e.kind, "snapshot_restore_exception");
    assert_eq!(
        e.reason,
        format!(
            "[{repo}:snap-1/{}] cannot restore index [idx1] because an open index with same name already exists in the cluster. Either close or delete the existing index or restore the index under a different name by providing a rename pattern and replacement name",
            info1.uuid
        )
    );
    let rr = RestoreRequest::from_json(Some(&json!({
        "indices": "idx*",
        "rename_pattern": "idx(\\d)",
        "rename_replacement": "restored_$1",
        "include_aliases": false,
        "index_settings": {"index.number_of_replicas": 0},
        "include_global_state": true
    })))
    .unwrap();
    let plan = svc.plan_restore(repo, "snap-1", &rr, &cluster).await.unwrap();
    let targets: Vec<&str> = plan.indices.iter().map(|i| i.target_index.as_str()).collect();
    assert_eq!(targets, vec!["restored_1", "restored_2"]);
    assert_eq!(plan.indices[0].shards, vec![0, 1]);
    assert_eq!(plan.indices[0].metadata["settings"]["index.number_of_replicas"], 0);
    assert!(plan.indices[0].metadata.get("aliases").is_none());
    assert!(plan.global_metadata.is_some());
    assert_eq!(plan.response_json(3)["snapshot"]["shards"], json!({"total":3,"failed":0,"successful":3}));
    let rr = RestoreRequest::from_json(Some(&json!({"rename_pattern": "idx\\d", "rename_replacement": "same"}))).unwrap();
    let e = svc.plan_restore(repo, "snap-1", &rr, &BTreeMap::new()).await.unwrap_err();
    assert!(e.reason.ends_with("indices [idx2] and [idx1] are renamed into the same index [same]"), "{}", e.reason);
    let rr = RestoreRequest::from_json(Some(&json!({"indices": "nope"}))).unwrap();
    let e = svc.plan_restore(repo, "snap-1", &rr, &BTreeMap::new()).await.unwrap_err();
    assert_eq!((e.kind, e.reason.as_str()), ("index_not_found_exception", "no such index [nope]"));

    // ---------- clone ----------
    svc.clone_snapshot(repo, "snap-2", "snap-3", &json!({"indices": "idx2"})).await.unwrap();
    let c = svc.snapshot_info(repo, "snap-3").await.unwrap();
    assert_eq!(c.indices, vec!["idx2"]);
    assert_eq!(data_blobs(svc, repo).await.len(), blobs2.len(), "clone must not copy data");
    let rdir3 = work.join("restore/snap3-idx2-0");
    svc.restore_shard(repo, "snap-3", "idx2", 0, &rdir3, &CancelToken::new()).await.unwrap();
    assert_same_dir_files(&s20, &["x/nested.bin", "empty"], &rdir3);
    let e = svc.clone_snapshot(repo, "snap-2", "snap-3", &json!({"indices": "idx2"})).await.unwrap_err();
    assert_eq!(e.kind, "invalid_snapshot_name_exception");

    // ---------- delete snap-1: shared blobs stay, b.dat + segments_1 go ----------
    svc.delete_snapshots(repo, &["snap-1".into()]).await.unwrap();
    let blobs3 = data_blobs(svc, repo).await;
    assert_eq!(blobs3.len(), blobs2.len() - 2, "only b.dat and segments_1 blobs should be deleted");
    let rdir4 = work.join("restore/snap2-idx1-0");
    svc.restore_shard(repo, "snap-2", "idx1", 0, &rdir4, &CancelToken::new()).await.unwrap();
    assert_same_dir_files(&s10, &["_0.cfs", "_0.si", "_1.cfs", "segments_2"], &rdir4);
    let rdir5 = work.join("restore/snap2-idx1-1");
    svc.restore_shard(repo, "snap-2", "idx1", 1, &rdir5, &CancelToken::new()).await.unwrap();
    assert_same_dir_files(&s11, &["a.dat", "c.dat"], &rdir5);
    let e = svc.delete_snapshots(repo, &["snap-1".into()]).await.unwrap_err();
    assert_eq!(e.reason, format!("[{repo}:snap-1] is missing"));

    // ---------- cleanup ----------
    let repository = svc.repository(repo).unwrap();
    let root: &BlobContainer = repository.root();
    root.write_blob("snap-dangling.dat", Bytes::from_static(b"12345"), false).await.unwrap();
    root.child("tests-xyz").write_blob("master.dat", Bytes::from_static(b"123"), false).await.unwrap();
    repository.indices_container().child("ghost").child("0").write_blob("__ghost", Bytes::from_static(b"1234567"), false).await.unwrap();
    let some_shard = blobs3.keys().next().unwrap().rsplitn(2, '/').nth(1).unwrap().to_string();
    let (iid, sh) = some_shard.split_once('/').unwrap();
    repository.indices_container().child(iid).child(sh).write_blob("__dangling", Bytes::from_static(b"12"), false).await.unwrap();
    let cr = svc.cleanup_repository(repo).await.unwrap();
    assert_eq!(cr, json!({"results": {"deleted_bytes": 5 + 3 + 7 + 2, "deleted_blobs": 4}}));
    assert_eq!(svc.cleanup_repository(repo).await.unwrap(), json!({"results": {"deleted_bytes": 0, "deleted_blobs": 0}}));
    assert_eq!(data_blobs(svc, repo).await, blobs3);

    // ---------- verify ----------
    let v = svc
        .verify_repository(repo, &[("n1".into(), "node-1".into()), ("n2".into(), "node-2".into())])
        .await
        .unwrap();
    assert_eq!(v, json!({"nodes": {"n1": {"name": "node-1"}, "n2": {"name": "node-2"}}}));

    // ---------- delete everything ----------
    svc.delete_snapshots(repo, &["snap-*".into()]).await.unwrap();
    assert!(data_blobs(svc, repo).await.is_empty());
    let all = svc.get_snapshots(&get_req(&["_all"])).await.unwrap();
    assert_eq!(all["total"], 0);
    let root_blobs = repository.root().list_blobs().await.unwrap();
    let names: Vec<&String> = root_blobs.keys().collect();
    assert_eq!(names.len(), 2, "only index-N and index.latest should remain: {names:?}");
    assert!(repository.indices_container().children().await.unwrap().is_empty());
}
