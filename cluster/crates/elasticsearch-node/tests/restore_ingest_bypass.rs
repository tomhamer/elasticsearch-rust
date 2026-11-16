//! A restore never runs an ingest pipeline (`snapshot::restore`): every
//! path that re-indexes the snapshot's documents instead of adopting its
//! files writes them exactly as the snapshot holds them, although the
//! restored index has a default and a final pipeline (both non-idempotent
//! here: each run appends to an array). Restored `_source`, `_version` and
//! `_routing` match the snapshotted index exactly, and the pipelines apply
//! again to writes after the restore.
//!
//! Fallback paths: a source-only repository, a restore that removes a
//! routing setting, files a member cannot adopt (seam), and
//! `snapshot.restore.file_based: false`.

mod common;

use common::Cluster;
use reqwest::Method;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::atomic::Ordering;

async fn ok(c: &Cluster, method: Method, path: &str, body: Option<Value>) -> Value {
    let (st, v) = c.call(c.master, method.clone(), path, body).await;
    assert!((200..300).contains(&st), "{method} {path}: {st} {v}");
    v
}

async fn bulk(c: &Cluster, nd: String) {
    let r = c.c.post(format!("http://127.0.0.1:{}/_bulk?refresh=true", c.p.http[c.master])).header("content-type", "application/x-ndjson").body(nd).send().await.unwrap();
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["errors"], false, "{v}");
}

fn routing(i: usize) -> Option<String> {
    (i % 2 == 0).then(|| format!("r{}", i % 3))
}

fn docs(ids: impl Iterator<Item = usize>, v: &str) -> String {
    ids.map(|i| {
        let mut meta = json!({"_index": "orig", "_id": format!("d{i}")});
        if let Some(r) = routing(i) {
            meta["routing"] = json!(r);
        }
        format!("{}\n{}\n", json!({"index": meta}), json!({"n": i, "v": v}))
    }).collect()
}

/// Every document of `index`: id → (_source, _version, _routing).
async fn contents(c: &Cluster, index: &str) -> BTreeMap<String, (Value, u64, Option<String>)> {
    let v = ok(c, Method::POST, &format!("/{index}/_search"), Some(json!({"size": 1000, "version": true, "query": {"match_all": {}}}))).await;
    v["hits"]["hits"].as_array().unwrap_or_else(|| panic!("{v}")).iter().map(|h| {
        (h["_id"].as_str().unwrap().to_string(), (h["_source"].clone(), h["_version"].as_u64().unwrap(), h["_routing"].as_str().map(str::to_string)))
    }).collect()
}

/// Pipelines, an index with both, documents (some routed, some updated),
/// and its snapshots in an fs and a source-only repository. Returns the
/// snapshotted contents.
async fn prepare(c: &Cluster, repo: &str) -> BTreeMap<String, (Value, u64, Option<String>)> {
    ok(c, Method::PUT, "/_ingest/pipeline/dflt", Some(json!({"processors": [{"append": {"field": "dflt", "value": ["y"]}}]}))).await;
    ok(c, Method::PUT, "/_ingest/pipeline/fin", Some(json!({"processors": [{"append": {"field": "fin", "value": ["x"]}}]}))).await;
    ok(c, Method::PUT, "/orig", Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0, "index.default_pipeline": "dflt", "index.final_pipeline": "fin"}}))).await;
    bulk(c, docs(0..40, "first")).await;
    bulk(c, docs(1..2, "second")).await;
    bulk(c, docs(1..3, "third")).await;
    let before = contents(c, "orig").await;
    assert_eq!(before.len(), 40);
    assert_eq!(before["d1"].1, 3);
    assert_eq!(before["d1"].0, json!({"n": 1, "v": "third", "dflt": ["y"], "fin": ["x"]}), "each write ran each pipeline once");
    assert_eq!(before["d0"].2.as_deref(), Some("r0"));
    ok(c, Method::POST, "/orig/_flush", None).await;
    ok(c, Method::PUT, "/_snapshot/fs", Some(json!({"type": "fs", "settings": {"location": format!("{repo}/fs")}}))).await;
    ok(c, Method::PUT, "/_snapshot/src", Some(json!({"type": "source", "settings": {"delegate_type": "fs", "location": format!("{repo}/src")}}))).await;
    for r in ["fs", "src"] {
        let v = ok(c, Method::PUT, &format!("/_snapshot/{r}/snap?wait_for_completion=true"), Some(json!({"indices": "orig"}))).await;
        assert_eq!(v["snapshot"]["state"], "SUCCESS", "{v}");
    }
    before
}

/// Restore `orig` from `repo` as `target` (with `extra` request fields)
/// and check it was re-indexed with every document exactly as before.
async fn restore_exactly(c: &Cluster, repo: &str, target: &str, extra: Value, before: &BTreeMap<String, (Value, u64, Option<String>)>) {
    let mut body = json!({"indices": "orig", "rename_pattern": "orig", "rename_replacement": target});
    for (k, v) in extra.as_object().unwrap() {
        body[k] = v.clone();
    }
    let v = ok(c, Method::POST, &format!("/_snapshot/{repo}/snap/_restore?wait_for_completion=true"), Some(body)).await;
    assert_eq!(v["snapshot"]["shards"]["failed"], 0, "{v}");
    ok(c, Method::GET, &format!("/_cluster/health/{target}?wait_for_status=green&timeout=60s"), None).await;
    let stats = ok(c, Method::GET, &format!("/{target}/_stats/indexing"), None).await;
    assert!(stats["indices"][target]["primaries"]["indexing"]["index_total"].as_u64().unwrap_or(0) > 0, "{target} was not re-indexed: {stats}");
    let after = contents(c, target).await;
    assert_eq!(&after, before, "{target}: restored documents differ (a pipeline ran on them, or versions / routing changed)");
    // The pipelines are restored and apply to new writes, once.
    if repo == "src" {
        return; // a source-only restore's settings follow ES's source-only rules
    }
    let settings = ok(c, Method::GET, &format!("/{target}/_settings?flat_settings=true"), None).await;
    assert_eq!(settings[target]["settings"]["index.default_pipeline"], "dflt", "{settings}");
    assert_eq!(settings[target]["settings"]["index.final_pipeline"], "fin", "{settings}");
    ok(c, Method::PUT, &format!("/{target}/_doc/new?refresh=true"), Some(json!({"v": "new"}))).await;
    let d = ok(c, Method::GET, &format!("/{target}/_doc/new"), None).await;
    assert_eq!(d["_source"], json!({"v": "new", "dflt": ["y"], "fin": ["x"]}), "{d}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reindexing_restore_paths_never_run_pipelines() {
    let repo = common::scratch_dir("restore-ingest-repo");
    let repo_s = repo.to_string_lossy().to_string();
    let c = common::start(&[("path.repo", repo_s.as_str())]).await;
    let before = prepare(&c, &repo_s).await;
    // Source-only repository.
    restore_exactly(&c, "src", "from-source-repo", json!({}), &before).await;
    // A restore that removes a routing setting re-indexes.
    restore_exactly(&c, "fs", "routing-changed", json!({"ignore_index_settings": ["index.routing_path"]}), &before).await;
    // Files a member cannot adopt: the master deletes the file-restored
    // incarnation and re-indexes.
    elasticsearch_node::snapshot_api::file_restore::FORCE_REINDEX.store(1, Ordering::SeqCst);
    restore_exactly(&c, "fs", "files-refused", json!({}), &before).await;
    assert_eq!(elasticsearch_node::snapshot_api::file_restore::FORCE_REINDEX.load(Ordering::SeqCst), 0, "the seam was not reached");
    // Over the original name (deleted first), with the snapshot's pipelines.
    ok(&c, Method::DELETE, "/orig", None).await;
    let v = ok(&c, Method::POST, "/_snapshot/src/snap/_restore?wait_for_completion=true", Some(json!({"indices": "orig"}))).await;
    assert_eq!(v["snapshot"]["shards"]["failed"], 0, "{v}");
    assert_eq!(contents(&c, "orig").await, before);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn restore_with_file_restore_disabled_never_runs_pipelines() {
    let repo = common::scratch_dir("restore-ingest-nofiles-repo");
    let repo_s = repo.to_string_lossy().to_string();
    let c = common::start(&[("path.repo", repo_s.as_str()), ("snapshot.restore.file_based", "false")]).await;
    let before = prepare(&c, &repo_s).await;
    restore_exactly(&c, "fs", "file-based-off", json!({}), &before).await;
}
