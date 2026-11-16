//! Restart recovery of local shard data whose host-side bookkeeping is
//! damaged or could not be written.
//!
//! * The index uuid maps (`_cluster/uuids*.json`) are a cache of which
//!   cluster index incarnation each local engine index is. A missing,
//!   truncated, garbage or stale map must never cost a shard copy: the
//!   identity is reconstructed from the copy itself. An unreadable map fails
//!   the start (it is not an empty map), and the data waits for the operator.
//! * A primary whose shard copy record (`cluster-copy.json`) cannot be
//!   persisted does not start (nothing is acknowledged on it), and starts
//!   normally, with every document, once the disk is repaired.

mod common;
use common::procs::{start_procs, Procs};
use reqwest::Method;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Every file under `dir` whose name satisfies `f`.
fn files_where(dir: &Path, f: &dyn Fn(&str) -> bool) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if f(&e.file_name().to_string_lossy()) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

fn uuid_maps(data: &Path) -> Vec<PathBuf> {
    files_where(&data.join("_cluster"), &|n| n.starts_with("uuids") && n.ends_with(".json"))
}

async fn count(pl: &Procs, index: &str) -> u64 {
    let (st, v) = pl.call(0, Method::GET, &format!("/{index}/_count"), None).await;
    assert_eq!(st, 200, "{v}");
    v["count"].as_u64().unwrap()
}

async fn index_docs(pl: &Procs, index: &str, n: usize) {
    let mut body = String::new();
    for i in 0..n {
        body.push_str(&format!("{{\"index\":{{\"_id\":\"d{i}\"}}}}\n{{\"n\":{i}}}\n"));
    }
    let (st, v) = pl.ndjson(0, &format!("/{index}/_bulk?refresh=true"), body).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["errors"], false, "{v}");
    let (st, v) = pl.call(0, Method::POST, &format!("/{index}/_flush"), None).await;
    assert_eq!(st, 200, "{v}");
}

async fn restart(pl: &mut Procs) {
    pl.spawn(0, &[]).await;
    pl.wait_nodes(0, 1, 120).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn damaged_uuid_maps_never_delete_shard_data() {
    let mut pl = start_procs(1, &[]).await;
    let data = pl.dir.join("n0");
    let (st, v) = pl.call(0, Method::PUT, "/keep", Some(json!({"settings": {"number_of_shards": 3, "number_of_replicas": 0}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(0, "keep", "green", 120).await;
    index_docs(&pl, "keep", 60).await;
    assert_eq!(count(&pl, "keep").await, 60);
    let records = files_where(&data, &|n| n == "cluster-copy.json");
    assert_eq!(records.len(), 3, "one copy record per shard copy: {records:?}");

    for variant in ["missing", "truncated", "garbage", "stale", "empty"] {
        pl.stop(0, 30).await;
        let maps = uuid_maps(&data);
        assert!(!maps.is_empty(), "{variant}: no uuid map under {}", data.display());
        for m in &maps {
            let good = std::fs::read(m).unwrap();
            match variant {
                "missing" => std::fs::remove_file(m).unwrap(),
                "truncated" => std::fs::write(m, &good[..good.len() / 2]).unwrap(),
                "garbage" => std::fs::write(m, b"\x00\xffnot json").unwrap(),
                // Every entry names a local index that does not exist.
                "stale" => {
                    let mut v: Value = serde_json::from_slice(&good).unwrap();
                    if let Some(o) = v["indices"].as_object_mut() {
                        for e in o.values_mut() {
                            e[1] = json!("staleLocalUuid00000000");
                        }
                    }
                    std::fs::write(m, serde_json::to_vec(&v).unwrap()).unwrap();
                }
                _ => std::fs::write(m, b"{\"indices\":{}}").unwrap(),
            }
        }
        restart(&mut pl).await;
        pl.wait_health(0, "keep", "green", 120).await;
        assert_eq!(count(&pl, "keep").await, 60, "{variant}: documents lost");
        assert_eq!(files_where(&data, &|n| n == "cluster-copy.json"), records, "{variant}: a shard copy was removed or moved");
        // Still writable, and the map is whole again.
        let (st, v) = pl.call(0, Method::PUT, &format!("/keep/_doc/after-{variant}?refresh=true"), Some(json!({"n": -1}))).await;
        assert!(st == 201 || st == 200, "{variant}: {v}");
        let (st, v) = pl.call(0, Method::DELETE, &format!("/keep/_doc/after-{variant}?refresh=true"), None).await;
        assert_eq!(st, 200, "{variant}: {v}");
        for m in uuid_maps(&data) {
            serde_json::from_slice::<Value>(&std::fs::read(&m).unwrap()).unwrap_or_else(|e| panic!("{variant}: {} unreadable: {e}", m.display()));
        }
    }

    // Unreadable (a read error, not "not found"): the node refuses to start
    // and touches nothing.
    pl.stop(0, 30).await;
    let map = data.join("_cluster").join("uuids.json");
    std::fs::remove_file(&map).unwrap();
    std::fs::create_dir(&map).unwrap();
    pl.spawn(0, &[]).await;
    let status = pl.wait_exit(0, 120).await;
    assert!(!status.success(), "a node with an unreadable uuid map must not start:\n{}", pl.log_tail(0));
    assert_eq!(files_where(&data, &|n| n == "cluster-copy.json"), records, "data touched by a refused start");
    std::fs::remove_dir(&map).unwrap();
    restart(&mut pl).await;
    pl.wait_health(0, "keep", "green", 120).await;
    assert_eq!(count(&pl, "keep").await, 60);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn primary_start_fails_until_its_copy_record_can_be_persisted() {
    let mut pl = start_procs(1, &[]).await;
    let data = pl.dir.join("n0");
    let (st, v) = pl.call(0, Method::PUT, "/rec", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(0, "rec", "green", 120).await;
    index_docs(&pl, "rec", 25).await;
    pl.stop(0, 30).await;

    // The record's temporary file cannot be created (as with a full or
    // failing disk at that step): the existing-store primary must not start.
    let records = files_where(&data, &|n| n == "cluster-copy.json");
    assert_eq!(records.len(), 1, "{records:?}");
    let before = std::fs::read(&records[0]).unwrap();
    let blocker = records[0].with_extension("tmp");
    std::fs::create_dir(&blocker).unwrap();
    restart(&mut pl).await;
    tokio::time::sleep(Duration::from_secs(5)).await;
    let (_, h) = pl.call(0, Method::GET, "/_cluster/health/rec?wait_for_status=yellow&timeout=5s", None).await;
    assert_eq!(h["status"], "red", "a primary started without its copy record: {h}");
    assert_eq!(h["active_primary_shards"], 0, "{h}");
    // No write is acknowledged meanwhile.
    let (st, v) = pl.call(0, Method::PUT, "/rec/_doc/during?timeout=2s", Some(json!({"n": -1}))).await;
    assert!(st != 200 && st != 201, "a write was acknowledged by a primary without its copy record: {v}");
    let (_, explain) = pl.call(0, Method::GET, "/_cluster/allocation/explain", Some(json!({"index": "rec", "shard": 0, "primary": true}))).await;
    assert!(explain.to_string().contains("copy record"), "the failure is explained: {explain}");
    assert_eq!(std::fs::read(&records[0]).unwrap(), before, "the previous record stays whole");

    // Repaired: the same copy starts, with every document.
    std::fs::remove_dir(&blocker).unwrap();
    let (st, v) = pl.call(0, Method::POST, "/_cluster/reroute?retry_failed=true", None).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(0, "rec", "green", 120).await;
    assert_eq!(count(&pl, "rec").await, 25);
    let (st, v) = pl.call(0, Method::PUT, "/rec/_doc/after?refresh=true", Some(json!({"n": -1}))).await;
    assert!(st == 201 || st == 200, "{v}");
    // And again after another restart.
    pl.stop(0, 30).await;
    restart(&mut pl).await;
    pl.wait_health(0, "rec", "green", 120).await;
    assert_eq!(count(&pl, "rec").await, 26);
}

async fn index_uuid(pl: &Procs, via: usize, index: &str) -> String {
    let (st, v) = pl.call(via, Method::GET, &format!("/{index}/_settings"), None).await;
    assert_eq!(st, 200, "{v}");
    v[index]["settings"]["index"]["uuid"].as_str().unwrap().to_string()
}

/// Shard copy records under `dir` naming index uuid `uuid`.
fn copies_of(dir: &Path, uuid: &str) -> Vec<PathBuf> {
    files_where(dir, &|n| n == "cluster-copy.json").into_iter()
        .filter(|p| std::fs::read_to_string(p).is_ok_and(|s| s.contains(&format!("\"{uuid}\"")))).collect()
}

/// ES `IndexGraveyard`: an index deleted while a node holding its only copy
/// was away is tombstoned, and the node deletes its copy when it rejoins
/// (it is not left dangling).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_node_that_missed_a_deletion_deletes_the_tombstoned_copy() {
    let mut pl = start_procs(3, &[]).await;
    let (st, v) = pl.call(0, Method::PUT, "/gone", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0,
        "index.routing.allocation.require._name": "n2"}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(0, "gone", "green", 120).await;
    index_docs(&pl, "gone", 10).await;
    let uuid = index_uuid(&pl, 0, "gone").await;
    let n2_dir = pl.dir.join("n2");
    assert_eq!(copies_of(&n2_dir, &uuid).len(), 1);

    pl.stop(2, 30).await;
    pl.wait_nodes(0, 2, 120).await;
    let (st, v) = pl.call(0, Method::DELETE, "/gone", None).await;
    assert_eq!(st, 200, "{v}");
    let (_, v) = pl.call(0, Method::GET, "/_cluster/state/metadata", None).await;
    let stones = v["metadata"]["index-graveyard"]["tombstones"].as_array().cloned().unwrap_or_default();
    assert!(stones.iter().any(|t| t["index"]["index_uuid"] == uuid.as_str() && t["index"]["index_name"] == "gone"), "{v}");
    assert_eq!(copies_of(&n2_dir, &uuid).len(), 1, "the absent node still holds its copy");

    pl.spawn(2, &[]).await;
    pl.wait_nodes(2, 3, 120).await;
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    while !copies_of(&n2_dir, &uuid).is_empty() {
        assert!(std::time::Instant::now() < deadline, "the tombstoned copy was not deleted:\n{}", pl.log_tail(2));
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    let (st, v) = pl.call(0, Method::GET, "/_dangling", None).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["dangling_indices"], json!([]), "a tombstoned index is never dangling: {v}");
}

/// ES `DanglingIndicesState`: a node whose cluster metadata lacks an index it
/// holds (here: its cluster state was lost and it formed a new cluster) keeps
/// the index as dangling, never deletes it; the dangling index can be
/// imported (every document back, under its own uuid) or deleted.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_index_the_cluster_does_not_know_is_kept_dangling_importable_and_deletable() {
    let mut pl = start_procs(1, &[]).await;
    let data = pl.dir.join("n0");
    for (name, shards, docs) in [("imp", 2, 20), ("del", 1, 7)] {
        let (st, v) = pl.call(0, Method::PUT, &format!("/{name}"), Some(json!({"settings": {"number_of_shards": shards, "number_of_replicas": 0}}))).await;
        assert_eq!(st, 200, "{v}");
        pl.wait_health(0, name, "green", 120).await;
        index_docs(&pl, name, docs).await;
    }
    let (imp, del) = (index_uuid(&pl, 0, "imp").await, index_uuid(&pl, 0, "del").await);
    assert_eq!(copies_of(&data, &imp).len(), 2);
    pl.stop(0, 30).await;
    // The node loses its cluster state (not its index data) and bootstraps a
    // new cluster whose metadata knows neither index.
    std::fs::remove_dir_all(data.join("_cluster")).unwrap();
    restart(&mut pl).await;
    let (_, v) = pl.call(0, Method::GET, "/_cluster/state/metadata", None).await;
    assert!(v["metadata"]["indices"].get("imp").is_none() && v["metadata"]["indices"].get("del").is_none(), "{v}");
    assert_eq!(copies_of(&data, &imp).len(), 2, "dangling data is kept");
    assert_eq!(copies_of(&data, &del).len(), 1, "dangling data is kept");
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    let listed = loop {
        let (st, v) = pl.call(0, Method::GET, "/_dangling", None).await;
        assert_eq!(st, 200, "{v}");
        let list = v["dangling_indices"].as_array().cloned().unwrap_or_default();
        if list.len() == 2 {
            break list;
        }
        assert!(std::time::Instant::now() < deadline, "dangling indices not listed: {v}");
        tokio::time::sleep(Duration::from_millis(250)).await;
    };
    let by_name = |n: &str| listed.iter().find(|d| d["index_name"] == n).cloned().unwrap_or_else(|| panic!("{n} not listed: {listed:?}"));
    assert_eq!(by_name("imp")["index_uuid"], imp.as_str());
    assert_eq!(by_name("del")["index_uuid"], del.as_str());
    assert_eq!(by_name("imp")["node_ids"].as_array().map(Vec::len), Some(1), "{listed:?}");

    // Import needs accept_data_loss (ES), then brings every document back.
    let (st, v) = pl.call(0, Method::POST, &format!("/_dangling/{imp}"), None).await;
    assert_eq!(st, 400, "{v}");
    let (st, v) = pl.call(0, Method::POST, &format!("/_dangling/{imp}?accept_data_loss=true"), None).await;
    assert_eq!((st, v.clone()), (202, json!({"acknowledged": true})), "{v}");
    pl.wait_health(0, "imp", "green", 120).await;
    assert_eq!(index_uuid(&pl, 0, "imp").await, imp, "imported under its own uuid");
    assert_eq!(count(&pl, "imp").await, 20, "every document of every shard is back");

    // Delete removes the other one, data included.
    let (st, v) = pl.call(0, Method::DELETE, &format!("/_dangling/{del}?accept_data_loss=true"), None).await;
    assert_eq!((st, v.clone()), (202, json!({"acknowledged": true})), "{v}");
    let (_, v) = pl.call(0, Method::GET, "/_dangling", None).await;
    assert_eq!(v["dangling_indices"], json!([]), "{v}");
    assert!(copies_of(&data, &del).is_empty(), "the deleted dangling index's data is gone");

    // Both outcomes survive a restart.
    pl.stop(0, 30).await;
    restart(&mut pl).await;
    pl.wait_health(0, "imp", "green", 120).await;
    assert_eq!(count(&pl, "imp").await, 20);
    let (_, v) = pl.call(0, Method::GET, "/_dangling", None).await;
    assert_eq!(v["dangling_indices"], json!([]), "{v}");
}

/// Every shard's copy of a dangling index is listed, including copies of
/// shards above 0 in shard engines that are not open on their node, and
/// `DELETE _dangling` tombstones the index (ES): a holder that is away
/// during the delete deletes its copy when it comes back.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_shard_copy_is_listed_and_a_dangling_delete_reaches_offline_holders() {
    let mut pl = start_procs(3, &[]).await;
    // One shard per node: shards 1 and 2 live only in shard engines.
    let (st, v) = pl.call(0, Method::PUT, "/split", Some(json!({"settings": {"number_of_shards": 3, "number_of_replicas": 0,
        "index.routing.allocation.total_shards_per_node": 1}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(0, "split", "green", 120).await;
    index_docs(&pl, "split", 30).await;
    let uuid = index_uuid(&pl, 0, "split").await;
    let dirs: Vec<PathBuf> = (0..3).map(|i| pl.dir.join(format!("n{i}"))).collect();
    for d in &dirs {
        assert_eq!(copies_of(d, &uuid).len(), 1, "one shard copy per node: {}", d.display());
    }
    // Every node loses its cluster state (not its data); a new cluster forms.
    for i in (0..3).rev() {
        pl.stop(i, 30).await;
    }
    for d in &dirs {
        std::fs::remove_dir_all(d.join("_cluster")).unwrap();
    }
    for i in 0..3 {
        pl.spawn(i, &[]).await;
    }
    for i in 0..3 {
        pl.wait_nodes(i, 3, 120).await;
    }
    let ids = pl.node_ids(0).await;
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    loop {
        let (st, v) = pl.call(0, Method::GET, "/_dangling", None).await;
        assert_eq!(st, 200, "{v}");
        let entry = v["dangling_indices"].as_array().and_then(|a| a.iter().find(|d| d["index_uuid"] == uuid.as_str()).cloned());
        let mut holders: Vec<String> = entry.as_ref().and_then(|e| e["node_ids"].as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
        holders.sort();
        let mut want: Vec<String> = ids.values().cloned().collect();
        want.sort();
        if holders == want {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "every node holds a shard copy and must be listed: {v}");
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    for d in &dirs {
        assert_eq!(copies_of(d, &uuid).len(), 1, "listing keeps every copy: {}", d.display());
    }

    // n2 is away while the dangling index is deleted.
    pl.stop(2, 30).await;
    pl.wait_nodes(0, 2, 120).await;
    let (st, v) = pl.call(0, Method::DELETE, &format!("/_dangling/{uuid}?accept_data_loss=true"), None).await;
    assert_eq!((st, v.clone()), (202, json!({"acknowledged": true})), "{v}");
    let (_, v) = pl.call(0, Method::GET, "/_cluster/state/metadata", None).await;
    let stones = v["metadata"]["index-graveyard"]["tombstones"].as_array().cloned().unwrap_or_default();
    assert!(stones.iter().any(|t| t["index"]["index_uuid"] == uuid.as_str()), "the delete is a tombstone: {v}");
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    while !(copies_of(&dirs[0], &uuid).is_empty() && copies_of(&dirs[1], &uuid).is_empty()) {
        assert!(std::time::Instant::now() < deadline, "the present holders did not delete their copies");
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    assert_eq!(copies_of(&dirs[2], &uuid).len(), 1, "the absent holder still has its copy");

    // n2 returns: the tombstone deletes its copy.
    pl.spawn(2, &[]).await;
    pl.wait_nodes(2, 3, 120).await;
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    while !copies_of(&dirs[2], &uuid).is_empty() {
        assert!(std::time::Instant::now() < deadline, "the returning holder kept a tombstoned copy:\n{}", pl.log_tail(2));
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    let (_, v) = pl.call(0, Method::GET, "/_dangling", None).await;
    assert_eq!(v["dangling_indices"], json!([]), "{v}");
}
