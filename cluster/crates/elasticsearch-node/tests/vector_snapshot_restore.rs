//! Snapshot and restore of `dense_vector` shards (quantized HNSW with its
//! shared mutable graph, checkpoint and code log, plus DiskBBQ with its
//! shared IVF files) on three node processes ([`common::procs`]), a 2-shard
//! index. The snapshot carries only the files a shard's commit references
//! (a stray file in a primary's core directory is in no repository blob).
//! Restore adopts the snapshot's files on each primary's node (ES
//! `RecoverySource.SNAPSHOT`): the restored index answers every kNN query,
//! unrescored, with exactly the hits and scores of the original, so the
//! exact neighbours it found are the same ones, and no document was
//! re-indexed.
mod common;
use common::procs::{start_procs, Procs};
use reqwest::Method;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const SNAP: usize = 3;
const DIMS: usize = 64;
const DOCS: usize = 1500;
const K: usize = 10;
/// A file name the engine could own (a text segment) that no commit lists.
const STRAY: &str = "seg_7ffffff0.rstc";

fn vector(i: usize, seed: u64) -> Vec<f32> {
    let mut x = (i as u64 ^ seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let centre = (i % 24) as f32 / 24.0;
    (0..DIMS).map(|d| {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        let noise = (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        ((d as f32 * 0.37 + centre * 6.0).sin()) + 0.4 * noise
    }).collect()
}

fn l2_score(q: &[f32], v: &[f32]) -> f64 {
    1.0 / (1.0 + q.iter().zip(v).map(|(a, b)| ((*a - *b) as f64).powi(2)).sum::<f64>())
}

/// Every directory named `core-*` below `root`.
fn core_dirs(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(root) else { return out };
    for entry in entries.flatten() {
        let path = entry.path();
        if entry.file_type().is_ok_and(|t| t.is_dir()) {
            if entry.file_name().to_string_lossy().starts_with("core-") {
                out.push(path.clone());
            }
            out.extend(core_dirs(&path));
        }
    }
    out
}

/// `(id, score)` of an unrescored kNN search on `field` for query `q`:
/// scores are the quantized estimates, so they are identical only if the
/// graph and codes are.
async fn knn(pl: &Procs, node: usize, field: &str, q: &[f32]) -> Vec<(String, f64)> {
    let body = json!({"knn": {"field": field, "query_vector": q, "k": K, "num_candidates": 100}, "size": K, "_source": false});
    let (st, v) = pl.call(node, Method::POST, "/vec/_search", Some(body)).await;
    assert!(st == 200 && v["_shards"]["failed"] == 0, "{st} {v}");
    v["hits"]["hits"].as_array().unwrap().iter()
        .map(|h| (h["_id"].as_str().unwrap().to_string(), h["_score"].as_f64().unwrap())).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dense_vector_shards_restore_with_identical_knn_results() {
    let repo = common::scratch_dir("vector-snapshot-repo");
    let repo_path = repo.to_string_lossy().to_string();
    // Every snapshot freeze saves the graphs with its flush (as a file
    // recovery source does with this setting): the snapshotted checkpoint
    // is exact, so the restored graph is the original's.
    let pl = start_procs(SNAP, &[("path.repo", repo_path.as_str()), ("indices.recovery.graph_checkpoint_lag_ops", "0")]).await;
    let master = pl.master(0).await;
    let (st, v) = pl.call(master, Method::PUT, "/vec", Some(json!({
        "settings": {"number_of_shards": 2, "number_of_replicas": 0, "refresh_interval": "-1"},
        "mappings": {"properties": {
            "hnsw": {"type": "dense_vector", "dims": DIMS, "index": true, "similarity": "l2_norm", "index_options": {"type": "int8_hnsw"}},
            "disk": {"type": "dense_vector", "dims": DIMS, "index": true, "similarity": "l2_norm", "index_options": {"type": "bbq_disk"}}}}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(master, "vec", "green", 90).await;
    // Two rounds with a flush each (several segments, checkpoints and code
    // log chunks per shard), deletes in between.
    for round in 0..2 {
        let range = round * DOCS / 2..(round + 1) * DOCS / 2;
        let lines: String = range.map(|i| format!("{}\n{}\n", json!({"index": {"_id": format!("d{i}")}}),
            json!({"hnsw": vector(i, 1), "disk": vector(i, 2)}))).collect();
        let (st, v) = pl.ndjson(master, "/vec/_bulk?refresh=true", lines).await;
        assert!(st == 200 && v["errors"] == false, "{v}");
        let (st, v) = pl.call(master, Method::POST, "/vec/_flush", None).await;
        assert_eq!(st, 200, "{v}");
    }
    let deleted: BTreeSet<usize> = (0..DOCS).step_by(9).collect();
    let lines: String = deleted.iter().map(|i| format!("{}\n", json!({"delete": {"_id": format!("d{i}")}}))).collect();
    let (st, v) = pl.ndjson(master, "/vec/_bulk?refresh=true", lines).await;
    assert!(st == 200 && v["errors"] == false, "{v}");
    // DiskBBQ indexes every row once merged; then a flushed commit.
    let (st, v) = pl.call(master, Method::POST, "/vec/_forcemerge?max_num_segments=1", None).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = pl.call(master, Method::POST, "/vec/_flush", None).await;
    assert_eq!(st, 200, "{v}");

    let queries: Vec<usize> = (0..12).map(|i| i * 113 + 5).collect();
    let mut before = Vec::new();
    let mut exact_found = Vec::new();
    for (field, seed) in [("hnsw", 1), ("disk", 2)] {
        for &i in &queries {
            let q = vector(i, seed + 10);
            let hits = knn(&pl, master, field, &q).await;
            assert_eq!(hits.len(), K, "{field} query {i}");
            assert!(hits.iter().all(|(id, _)| !deleted.contains(&id[1..].parse::<usize>().unwrap())), "{field}: a deleted doc");
            // The exact top-k over the live docs: which of them were found.
            let mut exact: Vec<(usize, f64)> = (0..DOCS).filter(|d| !deleted.contains(d)).map(|d| (d, l2_score(&q, &vector(d, seed)))).collect();
            exact.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            let truth: BTreeSet<String> = exact[..K].iter().map(|(d, _)| format!("d{d}")).collect();
            let found: BTreeSet<String> = hits.iter().map(|h| h.0.clone()).filter(|id| truth.contains(id)).collect();
            assert!(found.len() * 2 >= K, "{field} query {i}: recall {}/{K}", found.len());
            exact_found.push(found);
            before.push(hits);
        }
    }

    // A stray engine-named file in every core directory: no commit lists
    // it, so the snapshot must not carry it.
    let cores = core_dirs(&pl.dir);
    assert!(!cores.is_empty(), "no core directory under {}", pl.dir.display());
    for dir in &cores {
        std::fs::write(dir.join(STRAY), b"not part of any commit").unwrap();
    }

    let (st, v) = pl.call(master, Method::PUT, "/_snapshot/repo", Some(json!({"type": "fs", "settings": {"location": repo_path}}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = pl.call(master, Method::PUT, "/_snapshot/repo/snap1?wait_for_completion=true", Some(json!({"indices": "vec"}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["snapshot"]["state"], "SUCCESS", "{v}");
    assert_eq!(v["snapshot"]["shards"]["failed"], 0, "{v}");
    let (st, v) = pl.call(master, Method::DELETE, "/vec", None).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = pl.call(master, Method::POST, "/_snapshot/repo/snap1/_restore?wait_for_completion=true", Some(json!({"indices": "vec"}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(master, "vec", "green", 120).await;

    // The snapshot lists only committed files: the stray name is in none of
    // the repository's blobs (shard snapshot metadata names every file).
    fn blobs(dir: &Path, out: &mut Vec<PathBuf>) {
        for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            if e.file_type().is_ok_and(|t| t.is_dir()) { blobs(&e.path(), out) } else { out.push(e.path()) }
        }
    }
    let mut files = Vec::new();
    blobs(&repo, &mut files);
    assert!(files.iter().any(|f| f.file_name().is_some_and(|n| n.to_string_lossy().starts_with("snap-"))), "no snapshot metadata in {}", repo.display());
    for f in &files {
        let bytes = std::fs::read(f).unwrap();
        assert!(!bytes.windows(STRAY.len()).any(|w| w == STRAY.as_bytes()), "the unreferenced {STRAY} was snapshotted ({})", f.display());
    }
    // The restored copies hold only committed files (the deleted index's
    // own directories, should any remain, are not the restored copies).
    for dir in core_dirs(&pl.dir).into_iter().filter(|dir| !cores.contains(dir)) {
        assert!(!dir.join(STRAY).exists(), "an unreferenced file was snapshotted and restored into {}", dir.display());
    }
    let (st, v) = pl.call(master, Method::GET, "/vec/_count", None).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["count"], (DOCS - deleted.len()) as u64, "{v}");
    // Exactly the live documents, from every node.
    let live: BTreeSet<String> = (0..DOCS).filter(|d| !deleted.contains(d)).map(|d| format!("d{d}")).collect();
    for node in 0..SNAP {
        let (st, v) = pl.call(node, Method::POST, "/vec/_search", Some(json!({"size": 10000, "_source": false, "query": {"match_all": {}}}))).await;
        assert!(st == 200 && v["_shards"]["failed"] == 0, "{st} {v}");
        let ids: BTreeSet<String> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
        assert_eq!(ids, live, "n{node}: restored ids differ from the snapshotted live docs");
    }
    // The primaries adopted the snapshot's files: `_recovery` reports a
    // SNAPSHOT recovery per shard and no document went through indexing.
    let (st, v) = pl.call(master, Method::GET, "/vec/_recovery", None).await;
    assert_eq!(st, 200, "{v}");
    let primaries: Vec<&Value> = v["vec"]["shards"].as_array().unwrap().iter().filter(|s| s["primary"] == true).collect();
    assert_eq!(primaries.len(), 2, "{v}");
    for p in &primaries {
        assert_eq!((p["type"].as_str(), p["stage"].as_str()), (Some("SNAPSHOT"), Some("DONE")), "{p}");
        assert_eq!((p["source"]["repository"].as_str(), p["source"]["snapshot"].as_str(), p["source"]["index"].as_str()), (Some("repo"), Some("snap1"), Some("vec")), "{p}");
        assert!(p["index"]["size"]["recovered_in_bytes"].as_u64().unwrap() > 0, "{p}");
    }
    let (st, v) = pl.call(master, Method::GET, "/vec/_stats/indexing", None).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["indices"]["vec"]["total"]["indexing"]["index_total"], 0, "restored without re-indexing: {v}");
    // Unrescored kNN: the snapshotted graphs and codes, so the very hits
    // and scores of the original, from every node.
    let mut n = 0;
    for (field, seed) in [("hnsw", 1), ("disk", 2)] {
        for &i in &queries {
            let q = vector(i, seed + 10);
            for node in 0..SNAP {
                let after = knn(&pl, node, field, &q).await;
                assert_eq!(after, before[n], "{field} query {i} via n{node}: restored results differ");
                let found: BTreeSet<String> = after.iter().map(|h| h.0.clone()).filter(|id| exact_found[n].contains(id)).collect();
                assert_eq!(found, exact_found[n], "{field} query {i}: exact neighbours found differ after restore");
                assert!(after.iter().all(|(id, _)| live.contains(id)), "{field} query {i} via n{node}: a deleted doc");
            }
            n += 1;
        }
    }
}
