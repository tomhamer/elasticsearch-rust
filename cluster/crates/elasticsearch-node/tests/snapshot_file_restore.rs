//! File-based snapshot restore on a three-node in-process cluster
//! (`snapshot_api::file_restore`):
//!
//! * a multi-shard index restored with replicas: every copy holds exactly
//!   its shard's documents, the primaries adopted the snapshot's files and
//!   the replicas recovered from them by files, nothing was re-indexed, and
//!   kNN answers as before from every node;
//! * a rename restore adopts the files under the new name (aliases dropped,
//!   settings changed as asked);
//! * a dynamically grown mapping (fields numbered out of name order): the
//!   restored index and its replica take the snapshot's field ids, so the
//!   files are adopted and answer as before;
//! * the fallback: a restore that changes a routing setting re-indexes;
//! * corrupt snapshot files fail the restore and no copy ever starts;
//! * a restore over a closed index replaces it with the snapshot;
//! * a snapshot staged without a flush replays its translog above the
//!   commit;
//! * `_recovery` / `_cat/recovery` report the restore (`SNAPSHOT`).

#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use elasticsearch_node::config::{NodeConfig, Settings};
use elasticsearch_node::node::ClusterNode;
use reqwest::Method;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The tests share the process-wide snapshot freeze seam.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

const DIMS: usize = 16;

struct Three {
    nodes: Vec<Arc<ClusterNode>>,
    http: [u16; 3],
    c: reqwest::Client,
    repo: PathBuf,
}

fn settings(cluster: &str, i: usize, seeds: &[u16], repo: &Path) -> Settings {
    let dir = evidence::scratch_dir(&format!("{cluster}-n{i}"));
    let mut s = Settings::new();
    let mut put = |k: &str, v: Value| {
        s.insert(k.to_string(), v);
    };
    put("cluster.name", json!(cluster));
    put("node.name", json!(format!("n{i}")));
    put("path.data", json!(dir.join("data").to_string_lossy()));
    put("path.logs", json!(dir.join("logs").to_string_lossy()));
    put("path.repo", json!(repo.to_string_lossy()));
    put("network.host", json!("127.0.0.1"));
    // Not a security test (security defaults to on, as in ES).
    put("xpack.security.enabled", json!("false"));
    put("http.port", json!("0"));
    put("transport.port", json!("0"));
    put("discovery.seed_hosts", json!(ports::seeds(seeds)));
    put("cluster.initial_master_nodes", json!(["n0", "n1", "n2"]));
    // Snapshot freezes and file recovery sources save the graphs first:
    // shipped checkpoints are exact, so restored graphs are the originals.
    put(elasticsearch_node::replication::file_recovery::GRAPH_LAG_OPS, json!("0"));
    for w in ["low", "high", "flood_stage"] {
        put(&format!("cluster.routing.allocation.disk.watermark.{w}"), json!("1b"));
    }
    s
}

async fn start(cluster: &str) -> Three {
    let repo = evidence::scratch_dir(&format!("{cluster}-repo"));
    let mut http = [0u16; 3];
    let mut transport: Vec<u16> = Vec::new();
    let mut nodes = Vec::new();
    for i in 0..3 {
        let cfg = NodeConfig::from_settings(settings(cluster, i, &transport, &repo)).expect("config");
        let node = tokio::spawn(async move {
            let (node, listener) = ClusterNode::start(cfg).await.expect("node starts");
            elasticsearch_node::rest::register(&node);
            elasticsearch_node::snapshot_api::register(&node);
            let n = node.clone();
            tokio::spawn(async move {
                let _ = elasticsearch_node::front::serve(n, listener).await;
            });
            node
        }).await.unwrap();
        http[i] = ports::http_port(&node);
        transport.push(ports::transport_port(&node));
        nodes.push(node);
    }
    let c = reqwest::Client::new();
    ports::wait_formed(&c, &http, None, 3, Duration::from_secs(60)).await.unwrap_or_else(|e| panic!("cluster did not form: {e}"));
    let three = Three { nodes, http, c, repo };
    three.ok(0, Method::PUT, "/_snapshot/repo", Some(json!({"type": "fs", "settings": {"location": three.repo.to_string_lossy()}}))).await;
    three
}

impl Three {
    async fn call(&self, i: usize, method: Method, path: &str, body: Option<Value>) -> (u16, Value) {
        let mut r = self.c.request(method, format!("http://127.0.0.1:{}{path}", self.http[i]));
        if let Some(b) = body {
            r = r.json(&b);
        }
        let Ok(resp) = r.send().await else { return (0, Value::Null) };
        let status = resp.status().as_u16();
        (status, resp.json().await.unwrap_or(Value::Null))
    }

    async fn ok(&self, i: usize, method: Method, path: &str, body: Option<Value>) -> Value {
        let (st, v) = self.call(i, method.clone(), path, body).await;
        assert!((200..300).contains(&st), "{method} {path}: {st} {v}");
        v
    }

    /// Green with `copies` active copies, as every node sees it.
    async fn green(&self, index: &str, copies: u32) {
        for i in 0..3 {
            let v = self.ok(i, Method::GET, &format!("/_cluster/health/{index}?wait_for_status=green&wait_for_active_shards={copies}&wait_for_no_relocating_shards=true&wait_for_no_initializing_shards=true&timeout=120s"), None).await;
            assert!(v["status"] == "green" && v["active_shards"] == copies && v["timed_out"] == false, "n{i}: {v}");
        }
    }

    async fn vector_index(&self, index: &str, shards: u32, replicas: u32) {
        self.ok(0, Method::PUT, &format!("/{index}"), Some(json!({
            "settings": {"number_of_shards": shards, "number_of_replicas": replicas},
            "mappings": {"properties": {"emb": {"type": "dense_vector", "dims": DIMS, "similarity": "l2_norm", "index_options": {"type": "int8_hnsw", "m": 8, "ef_construction": 50}},
                "tag": {"type": "keyword"}, "n": {"type": "integer"}}}
        }))).await;
        self.green(index, shards * (replicas + 1)).await;
    }

    async fn ndjson(&self, nd: String) {
        let r = self.c.post(format!("http://127.0.0.1:{}/_bulk?refresh=true", self.http[0])).header("content-type", "application/x-ndjson").body(nd).send().await.unwrap();
        let v: Value = r.json().await.unwrap();
        assert_eq!(v["errors"], false, "{v}");
    }

    async fn bulk(&self, index: &str, ids: std::ops::Range<usize>) {
        let mut nd = String::new();
        for i in ids {
            nd.push_str(&format!("{}\n{}\n", json!({"index": {"_index": index, "_id": format!("d{i}")}}), json!({"emb": vector(i), "tag": format!("t{}", i % 7), "n": i})));
        }
        self.ndjson(nd).await;
    }

    async fn delete_docs(&self, index: &str, ids: impl Iterator<Item = usize>) {
        let nd: String = ids.map(|i| format!("{}\n", json!({"delete": {"_index": index, "_id": format!("d{i}")}}))).collect();
        self.ndjson(nd).await;
    }

    async fn snapshot(&self, snap: &str, indices: &str) {
        let v = self.ok(0, Method::PUT, &format!("/_snapshot/repo/{snap}?wait_for_completion=true"), Some(json!({"indices": indices}))).await;
        assert_eq!(v["snapshot"]["state"], "SUCCESS", "{v}");
        assert_eq!(v["snapshot"]["shards"]["failed"], 0, "{v}");
    }

    async fn restore(&self, snap: &str, body: Value) -> (u16, Value) {
        self.call(0, Method::POST, &format!("/_snapshot/repo/{snap}/_restore?wait_for_completion=true"), Some(body)).await
    }

    /// The ids in shard `k` of `index`, from the copy on node `node` (any
    /// copy when `None`).
    async fn shard_ids(&self, index: &str, k: u32, node: Option<&str>) -> BTreeSet<String> {
        let pref = match node {
            Some(n) => format!("_shards:{k}|_only_nodes:{n}"),
            None => format!("_shards:{k}"),
        };
        let v = self.ok(0, Method::POST, &format!("/{index}/_search?preference={pref}"), Some(json!({"size": 10000, "_source": false, "query": {"match_all": {}}}))).await;
        assert!(v["_shards"]["failed"] == 0 && v["_shards"]["successful"] == 1, "{pref}: {v}");
        v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect()
    }

    /// `(shard, primary, node name)` of every assigned copy of `index`.
    async fn copies(&self, index: &str) -> Vec<(u32, bool, String)> {
        let v = self.ok(0, Method::GET, &format!("/_cat/shards/{index}?format=json"), None).await;
        v.as_array().unwrap().iter().filter(|r| r["state"] == "STARTED")
            .map(|r| (r["shard"].as_str().unwrap().parse().unwrap(), r["prirep"] == "p", r["node"].as_str().unwrap().to_string())).collect()
    }

    /// kNN hits (id, score), unrescored, preferring node `n{node}`'s copies.
    async fn knn(&self, index: &str, node: usize, q: usize) -> Vec<(String, f64)> {
        let v = self.ok(0, Method::POST, &format!("/{index}/_search?preference=_prefer_nodes:n{node}"), Some(json!({
            "knn": {"field": "emb", "query_vector": vector(1_000_000 + q), "k": 10, "num_candidates": 50}, "_source": false, "size": 10
        }))).await;
        assert_eq!(v["_shards"]["failed"], 0, "{v}");
        v["hits"]["hits"].as_array().unwrap().iter().map(|h| (h["_id"].as_str().unwrap().to_string(), h["_score"].as_f64().unwrap())).collect()
    }

    /// Documents indexed into `index` (all copies) since its creation.
    async fn index_total(&self, index: &str) -> u64 {
        let v = self.ok(0, Method::GET, &format!("/{index}/_stats/indexing"), None).await;
        v["indices"][index]["total"]["indexing"]["index_total"].as_u64().unwrap_or_else(|| panic!("{v}"))
    }

    async fn recovery(&self, index: &str) -> Vec<Value> {
        let v = self.ok(0, Method::GET, &format!("/{index}/_recovery"), None).await;
        v[index]["shards"].as_array().unwrap_or_else(|| panic!("{v}")).clone()
    }
}

fn vector(i: usize) -> Vec<f32> {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..DIMS).map(|_| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % 2000) as f32 / 1000.0 - 1.0
    }).collect()
}

/// Every file below `dir`.
fn files_under(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        if e.file_type().is_ok_and(|t| t.is_dir()) { files_under(&e.path(), out) } else { out.push(e.path()) }
    }
}

/// A three-shard index restored with a replica per shard: the primaries
/// adopt the snapshot's files, the replicas recover from them by files
/// (not operations), every copy holds exactly its shard's documents, no
/// document is re-indexed anywhere, and kNN answers exactly as before.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn multi_shard_restore_with_replicas_copies_files_to_every_copy() {
    let _serial = SERIAL.lock().await;
    let three = start("sfr-multi").await;
    three.vector_index("docs", 3, 0).await;
    three.bulk("docs", 0..3000).await;
    three.ok(0, Method::POST, "/docs/_flush", None).await;
    three.bulk("docs", 3000..3600).await;
    three.delete_docs("docs", (0..3600).step_by(17)).await;
    three.ok(0, Method::POST, "/docs/_flush", None).await;
    let mut before = Vec::new();
    for k in 0..3 {
        before.push(three.shard_ids("docs", k, None).await);
    }
    let live: usize = before.iter().map(BTreeSet::len).sum();
    assert_eq!(live, 3600 - (0..3600).step_by(17).count());
    let knn_before: Vec<_> = futures::future::join_all((0..5).map(|q| three.knn("docs", 0, q))).await;
    three.snapshot("snap", "docs").await;
    three.ok(0, Method::DELETE, "/docs", None).await;
    let (st, v) = three.restore("snap", json!({"indices": "docs", "index_settings": {"index.number_of_replicas": 1}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["snapshot"]["shards"], json!({"total": 3, "failed": 0, "successful": 3}), "{v}");
    three.green("docs", 6).await;
    let copies = three.copies("docs").await;
    assert_eq!(copies.len(), 6, "{copies:?}");
    for (k, primary, node) in &copies {
        assert_eq!(&three.shard_ids("docs", *k, Some(node)).await, &before[*k as usize], "shard {k} ({}) on {node}", if *primary { "primary" } else { "replica" });
    }
    let rec = three.recovery("docs").await;
    assert_eq!(rec.len(), 6, "{rec:?}");
    for r in &rec {
        if r["primary"] == true {
            assert_eq!((r["type"].as_str(), r["stage"].as_str()), (Some("SNAPSHOT"), Some("DONE")), "{r}");
        } else {
            // File-based peer recovery: the primary's commit was copied.
            assert_eq!((r["type"].as_str(), r["stage"].as_str()), (Some("PEER"), Some("DONE")), "{r}");
            let files = &r["index"]["files"];
            assert!(files["total"].as_u64().unwrap() > 0, "the replica copied no files: {r}");
            assert_eq!(files["recovered"], files["total"], "{r}");
            assert_eq!(r["translog"]["recovered"], 0, "nothing written since the restore: {r}");
        }
    }
    let v = three.ok(0, Method::GET, "/docs/_stats?level=shards", None).await;
    for k in 0..3u32 {
        let shard = v["indices"]["docs"]["shards"][k.to_string()].as_array().unwrap_or_else(|| panic!("{v}"));
        assert_eq!(shard.len(), 2, "{v}");
        for c in shard {
            assert_eq!(c["indexing"]["index_total"], 0, "shard {k}: a copy re-indexed documents: {c}");
            assert_eq!(c["docs"]["count"].as_u64(), Some(before[k as usize].len() as u64), "shard {k}: {c}");
        }
    }
    for node in 0..3 {
        for (q, expected) in knn_before.iter().enumerate() {
            assert_eq!(&three.knn("docs", node, q).await, expected, "query {q} preferring n{node}");
        }
    }
    // The restore is over: no entry (or armed file restore) is left.
    let v = three.ok(0, Method::GET, "/_cluster/state", None).await;
    assert!(v["restore"]["snapshots"].as_array().is_none_or(|a| a.is_empty()), "{}", v["restore"]);
}

/// A rename restore adopts the snapshot's files under the new name, while
/// the original index stays; `include_aliases: false` and `index_settings`
/// apply to the restored index.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rename_restore_adopts_the_files_under_the_new_name() {
    let _serial = SERIAL.lock().await;
    let three = start("sfr-rename").await;
    three.vector_index("logs-a", 2, 0).await;
    three.ok(0, Method::POST, "/_aliases", Some(json!({"actions": [{"add": {"index": "logs-a", "alias": "logs"}}]}))).await;
    three.bulk("logs-a", 0..1200).await;
    three.ok(0, Method::POST, "/logs-a/_flush", None).await;
    let before = [three.shard_ids("logs-a", 0, None).await, three.shard_ids("logs-a", 1, None).await];
    three.snapshot("snap", "logs-a").await;
    let (st, v) = three.restore("snap", json!({
        "indices": "logs-a", "rename_pattern": "logs-(.+)", "rename_replacement": "restored-$1",
        "include_aliases": false, "index_settings": {"index.refresh_interval": "7s"}
    })).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["snapshot"]["indices"], json!(["restored-a"]), "{v}");
    three.green("restored-a", 2).await;
    for k in 0..2u32 {
        assert_eq!(three.shard_ids("restored-a", k, None).await, before[k as usize], "shard {k}");
        assert_eq!(three.shard_ids("logs-a", k, None).await, before[k as usize], "the original's shard {k}");
    }
    assert_eq!(three.index_total("restored-a").await, 0, "restored without re-indexing");
    for r in three.recovery("restored-a").await {
        assert_eq!((r["type"].as_str(), r["source"]["index"].as_str()), (Some("SNAPSHOT"), Some("logs-a")), "{r}");
    }
    let v = three.ok(0, Method::GET, "/restored-a/_alias", None).await;
    assert_eq!(v["restored-a"]["aliases"], json!({}), "{v}");
    let v = three.ok(0, Method::GET, "/logs-a/_alias", None).await;
    assert!(v["logs-a"]["aliases"].get("logs").is_some(), "{v}");
    let v = three.ok(0, Method::GET, "/restored-a/_settings", None).await;
    assert_eq!(v["restored-a"]["settings"]["index"]["refresh_interval"], "7s", "{v}");
    // Writes to the restored index work as usual.
    three.bulk("restored-a", 5000..5010).await;
    let v = three.ok(0, Method::GET, "/restored-a/_count", None).await;
    assert_eq!(v["count"], 1210, "{v}");
}

/// The core field ids `index`'s copy of shard 0 on node `i` reads with.
fn core_ids(three: &Three, i: usize, index: &str) -> Option<std::collections::BTreeMap<String, u16>> {
    let slot = three.nodes[i].shards.get(index, 0)?;
    let ids = slot.lock().svc.meta().mappings.core_ids;
    Some(ids.into_iter().filter(|(_, id)| *id >= elasticsearch_engine::mapping::FIRST_USER_FIELD).collect())
}

/// A search answered by the copy on node `node`: hits (id, score, sort),
/// total and aggregations.
async fn search_on(three: &Three, index: &str, node: &str, body: Value) -> Value {
    let v = three.ok(0, Method::POST, &format!("/{index}/_search?preference=_only_nodes:{node}"), Some(body)).await;
    assert_eq!(v["_shards"]["failed"], 0, "{v}");
    json!({"hits": v["hits"]["hits"].as_array().unwrap().iter().map(|h| json!([h["_id"], h["_score"], h["sort"]])).collect::<Vec<_>>(),
        "total": v["hits"]["total"], "aggs": v["aggregations"]})
}

/// Fields mapped dynamically in another order than their names' (`z`
/// before `a`) are numbered in that order; the restored index would number
/// them in sorted order. It takes the snapshot's ids instead, so the files
/// are adopted (nothing re-indexed), and the replica, on a node whose shell
/// numbered them its own way too, takes the primary's: every query answers
/// as before the snapshot on both copies.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn restore_adopts_the_files_of_a_dynamically_grown_mapping() {
    let _serial = SERIAL.lock().await;
    let three = start("sfr-grown").await;
    three.ok(0, Method::PUT, "/dyn", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}}))).await;
    three.green("dyn", 1).await;
    // Separate requests: `z` is mapped (and numbered) before `a`.
    fn docs(f: impl Fn(usize) -> Value) -> String {
        (0..300).map(|i| format!("{}\n{}\n", json!({"index": {"_index": "dyn", "_id": format!("d{i}")}}), f(i))).collect()
    }
    three.ndjson(docs(|i| json!({"z": i}))).await;
    three.ndjson(docs(|i| json!({"z": i, "a": format!("alpha w{}", i % 4)}))).await;
    three.ok(0, Method::POST, "/dyn/_flush", None).await;
    let holder = (0..3).find(|i| core_ids(&three, *i, "dyn").is_some()).expect("a copy of dyn");
    let original = core_ids(&three, holder, "dyn").unwrap();
    assert!(original["z"] < original["a"], "numbered as the mapping grew: {original:?}");
    let queries = [
        json!({"query": {"term": {"z": 7}}}),
        json!({"query": {"match": {"a": "w1"}}, "sort": [{"z": "asc"}], "size": 10}),
        json!({"query": {"range": {"z": {"gte": 290}}}, "size": 20, "sort": [{"z": "desc"}]}),
        json!({"size": 0, "aggs": {"words": {"terms": {"field": "a.keyword"}}, "max": {"max": {"field": "z"}}}}),
    ];
    let mut baseline = Vec::new();
    for q in &queries {
        baseline.push(search_on(&three, "dyn", &format!("n{holder}"), q.clone()).await);
    }
    three.snapshot("snap-dyn", "dyn").await;
    three.ok(0, Method::DELETE, "/dyn", None).await;
    let (st, v) = three.restore("snap-dyn", json!({"indices": "dyn", "index_settings": {"index.number_of_replicas": 1}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["snapshot"]["shards"]["successful"], 1, "{v}");
    three.green("dyn", 2).await;
    assert_eq!(three.index_total("dyn").await, 0, "the files were adopted, not re-indexed");
    let copies = three.copies("dyn").await;
    assert_eq!(copies.len(), 2, "{copies:?}");
    for (_, primary, node) in &copies {
        let i: usize = node[1..].parse().unwrap();
        let ids = core_ids(&three, i, "dyn").unwrap();
        for (field, id) in &original {
            assert_eq!(ids.get(field), Some(id), "{node} ({}) reads [{field}] under the snapshot's id: {ids:?}", if *primary { "primary" } else { "replica" });
        }
        for (q, want) in queries.iter().zip(&baseline) {
            assert_eq!(&search_on(&three, "dyn", node, q.clone()).await, want, "{node} answers {q}");
        }
    }
    for r in three.recovery("dyn").await {
        let want = if r["primary"] == true { "SNAPSHOT" } else { "PEER" };
        assert_eq!(r["type"], want, "{r}");
    }
}

/// The fallback: a restore that changes (here: removes) a routing setting
/// re-indexes from the start, and the documents route to their shards as
/// the restored index's routing says.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn restore_falls_back_to_reindexing_when_it_changes_routing() {
    let _serial = SERIAL.lock().await;
    let three = start("sfr-fallback").await;
    three.vector_index("plain", 2, 0).await;
    three.bulk("plain", 0..400).await;
    three.ok(0, Method::POST, "/plain/_flush", None).await;
    let before = [three.shard_ids("plain", 0, None).await, three.shard_ids("plain", 1, None).await];
    three.snapshot("snap-plain", "plain").await;
    three.ok(0, Method::DELETE, "/plain", None).await;
    let (st, v) = three.restore("snap-plain", json!({"indices": "plain", "ignore_index_settings": ["index.routing_partition_size"]})).await;
    assert_eq!(st, 200, "{v}");
    three.green("plain", 2).await;
    for k in 0..2u32 {
        assert_eq!(three.shard_ids("plain", k, None).await, before[k as usize], "re-indexed documents route to their shard {k}");
    }
    assert_eq!(three.index_total("plain").await, 400, "re-indexed");
    // `_recovery` still reports the restore.
    let rec = three.recovery("plain").await;
    assert!(rec.iter().filter(|r| r["primary"] == true).all(|r| r["type"] == "SNAPSHOT"), "{rec:?}");
    let v = three.ok(0, Method::GET, "/_cluster/state", None).await;
    assert!(v["restore"]["snapshots"].as_array().is_none_or(|a| a.is_empty()), "{}", v["restore"]);
}

/// A corrupt data blob in the repository: the download's SHA-256 check
/// fails, the primary's restore fails and is never retried (ES
/// `RestoreInProgressAllocationDecider`), so no copy of the index ever
/// starts and nothing half-restored is left on disk.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn corrupt_snapshot_files_fail_the_restore_and_no_copy_starts() {
    let _serial = SERIAL.lock().await;
    let three = start("sfr-corrupt").await;
    three.vector_index("c", 1, 0).await;
    three.bulk("c", 0..1500).await;
    three.ok(0, Method::POST, "/c/_flush", None).await;
    three.snapshot("snap", "c").await;
    three.ok(0, Method::DELETE, "/c", None).await;
    // The largest data blob of the shard (`indices/<id>/0/__<uuid>`).
    let mut blobs = Vec::new();
    files_under(&three.repo.join("indices"), &mut blobs);
    let blob = blobs.into_iter().filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("__")))
        .max_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)).expect("a data blob");
    let mut bytes = std::fs::read(&blob).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x5a;
    std::fs::write(&blob, bytes).unwrap();
    let (st, v) = three.restore("snap", json!({"indices": "c"})).await;
    assert!(st >= 400, "{st} {v}");
    assert!(v.to_string().contains("checksum failed"), "{v}");
    let v = three.ok(0, Method::GET, "/_cluster/health/c?timeout=1s", None).await;
    assert_eq!(v["status"], "red", "{v}");
    assert_eq!(v["active_shards"], 0, "{v}");
    // Not retried: still unassigned a while later, with the reason.
    tokio::time::sleep(Duration::from_secs(2)).await;
    let v = three.ok(0, Method::GET, "/_cat/shards/c?format=json", None).await;
    assert!(v.as_array().unwrap().iter().all(|r| r["state"] == "UNASSIGNED"), "{v}");
    let v = three.ok(0, Method::POST, "/_cluster/allocation/explain", Some(json!({"index": "c", "shard": 0, "primary": true}))).await;
    assert!(v.to_string().contains("shard has failed to be restored from the snapshot"), "{v}");
    for (i, n) in three.nodes.iter().enumerate() {
        assert!(n.shards.get("c", 0).is_none(), "n{i} holds a copy of the corrupt shard");
        let mut left = Vec::new();
        files_under(&n.config.data_path, &mut left);
        assert!(!left.iter().any(|p| p.to_string_lossy().contains("shard-0.recovery")), "n{i}: restore staging left behind: {left:?}");
    }
    // A search finds no copy to answer from (never an empty answer).
    let (st, v) = three.call(0, Method::POST, "/c/_search", Some(json!({"query": {"match_all": {}}}))).await;
    assert!(st != 200 || v["_shards"]["successful"] == 0, "{st} {v}");
    // Deleting the failed index ends it (ES).
    three.ok(0, Method::DELETE, "/c", None).await;
}

/// Production review #130: a restore planned over a closed index replaces
/// only that incarnation, still closed. Opened and written after the plan
/// was validated, or deleted and re-created under the name and written,
/// the live index and its acknowledged writes stay; the restore fails.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn restore_never_replaces_an_index_that_changed_after_its_plan() {
    let _serial = SERIAL.lock().await;
    let three = start("sfr-closed-race").await;
    three.vector_index("cr", 1, 0).await;
    three.bulk("cr", 0..100).await;
    three.snapshot("snap", "cr").await;
    for case in ["reopened", "recreated"] {
        three.ok(0, Method::POST, "/cr/_close", None).await;
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let (e, r) = (entered.clone(), release.clone());
        *elasticsearch_node::snapshot_api::restore::BEFORE_CLOSED_REPLACEMENT.lock() = Some(Arc::new(move || {
            let (e, r) = (e.clone(), r.clone());
            Box::pin(async move {
                e.notify_one();
                r.notified().await;
            }) as std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
        }));
        let interfere = async {
            tokio::time::timeout(Duration::from_secs(30), entered.notified()).await.expect("the restore reaches its replacement");
            if case == "reopened" {
                three.ok(0, Method::POST, "/cr/_open", None).await;
            } else {
                three.ok(0, Method::DELETE, "/cr", None).await;
                three.vector_index("cr", 1, 0).await;
            }
            three.green("cr", 1).await;
            three.bulk("cr", 1000..1050).await;
            three.ok(0, Method::POST, "/cr/_refresh", None).await;
            release.notify_one();
        };
        let ((st, v), ()) = tokio::join!(three.restore("snap", json!({"indices": "cr"})), interfere);
        *elasticsearch_node::snapshot_api::restore::BEFORE_CLOSED_REPLACEMENT.lock() = None;
        assert!(st >= 400 || v["snapshot"]["shards"]["failed"].as_u64().unwrap_or(0) > 0, "{case}: the restore must fail: {st} {v}");
        assert!(v.to_string().contains("cr"), "{case}: {v}");
        let v = three.ok(0, Method::GET, "/cr/_count", None).await;
        let want = if case == "reopened" { 150 } else { 50 };
        assert_eq!(v["count"], want, "{case}: the live index and its writes survive: {v}");
        let v = three.ok(0, Method::GET, "/cr/_doc/d1049", None).await;
        assert_eq!(v["found"], true, "{case}: {v}");
    }
}

/// Restoring over a closed index replaces it with the snapshot's data: the
/// documents written after the snapshot are gone, the index is open.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn restore_over_a_closed_index_replaces_it_with_the_snapshot() {
    let _serial = SERIAL.lock().await;
    let three = start("sfr-closed").await;
    three.vector_index("cl", 2, 0).await;
    three.bulk("cl", 0..800).await;
    three.ok(0, Method::POST, "/cl/_flush", None).await;
    let before = [three.shard_ids("cl", 0, None).await, three.shard_ids("cl", 1, None).await];
    three.snapshot("snap", "cl").await;
    three.bulk("cl", 800..1000).await;
    three.ok(0, Method::POST, "/cl/_close", None).await;
    let (st, v) = three.restore("snap", json!({"indices": "cl"})).await;
    assert_eq!(st, 200, "{v}");
    three.green("cl", 2).await;
    let v = three.ok(0, Method::GET, "/_cluster/state/metadata/cl", None).await;
    assert_eq!(v["metadata"]["indices"]["cl"]["state"], "open", "{v}");
    for k in 0..2u32 {
        assert_eq!(three.shard_ids("cl", k, None).await, before[k as usize], "shard {k}");
    }
    let v = three.ok(0, Method::GET, "/cl/_count", None).await;
    assert_eq!(v["count"], 800, "{v}");
    assert_eq!(three.index_total("cl").await, 0, "restored without re-indexing");
    for r in three.recovery("cl").await {
        assert_eq!(r["type"], "SNAPSHOT", "{r}");
    }
}

/// A snapshot staged without a flush (a copy was still recovering) holds
/// the commit plus the translog above it; the restored primary replays it:
/// every document and delete is there, and `_recovery` counts the ops.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unflushed_snapshot_replays_its_translog_on_restore() {
    let _serial = SERIAL.lock().await;
    let three = start("sfr-translog").await;
    three.vector_index("tl", 1, 0).await;
    three.bulk("tl", 0..400).await;
    three.ok(0, Method::POST, "/tl/_flush", None).await;
    // Above the commit: 250 documents and 10 deletes.
    three.bulk("tl", 400..650).await;
    three.delete_docs("tl", (0..50).step_by(5)).await;
    let live: BTreeSet<String> = (0..650).filter(|i| !(i % 5 == 0 && *i < 50)).map(|i| format!("d{i}")).collect();
    elasticsearch_node::snapshot_api::shard::SKIP_FLUSHES.store(1, Ordering::SeqCst);
    three.snapshot("snap", "tl").await;
    assert_eq!(elasticsearch_node::snapshot_api::shard::SKIP_FLUSHES.load(Ordering::SeqCst), 0, "the freeze staged an unflushed image");
    three.ok(0, Method::DELETE, "/tl", None).await;
    let (st, v) = three.restore("snap", json!({"indices": "tl"})).await;
    assert_eq!(st, 200, "{v}");
    three.green("tl", 1).await;
    assert_eq!(three.shard_ids("tl", 0, None).await, live);
    let rec = three.recovery("tl").await;
    let p = rec.iter().find(|r| r["primary"] == true).unwrap_or_else(|| panic!("{rec:?}"));
    assert_eq!(p["type"], "SNAPSHOT", "{p}");
    assert_eq!(p["translog"]["recovered"], 260, "the ops above the commit were replayed: {p}");
    assert_eq!(p["translog"]["total"], 260, "{p}");
    // A write after the restore continues the history.
    three.bulk("tl", 700..701).await;
    let v = three.ok(0, Method::GET, "/tl/_count", None).await;
    assert_eq!(v["count"], live.len() + 1, "{v}");
}

/// `_recovery` and `_cat/recovery` of a restored index: one `SNAPSHOT`
/// entry per primary with the snapshot as its source, the adopted bytes,
/// nothing reused or replayed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn recovery_apis_report_the_snapshot_restore_of_every_shard() {
    let _serial = SERIAL.lock().await;
    let three = start("sfr-recovery").await;
    three.vector_index("r", 2, 0).await;
    three.bulk("r", 0..600).await;
    three.ok(0, Method::POST, "/r/_flush", None).await;
    three.snapshot("snap", "r").await;
    three.ok(0, Method::DELETE, "/r", None).await;
    let started = Instant::now();
    let (st, v) = three.restore("snap", json!({"indices": "r"})).await;
    assert_eq!(st, 200, "{v}");
    three.green("r", 2).await;
    let rec = three.recovery("r").await;
    let mut ids: Vec<u64> = rec.iter().map(|r| r["id"].as_u64().unwrap()).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![0, 1], "{rec:?}");
    let names: BTreeSet<String> = (0..3).map(|i| format!("n{i}")).collect();
    for r in &rec {
        assert_eq!((r["type"].as_str(), r["stage"].as_str(), r["primary"].as_bool()), (Some("SNAPSHOT"), Some("DONE"), Some(true)), "{r}");
        let src = &r["source"];
        assert_eq!((src["repository"].as_str(), src["snapshot"].as_str(), src["index"].as_str()), (Some("repo"), Some("snap"), Some("r")), "{r}");
        assert!(src["restoreUUID"].as_str().is_some_and(|u| !u.is_empty()), "{r}");
        assert!(src.get("version").is_some(), "{r}");
        assert!(names.contains(r["target"]["name"].as_str().unwrap_or_default()), "{r}");
        let (size, files) = (&r["index"]["size"], &r["index"]["files"]);
        assert!(size["total_in_bytes"].as_u64().unwrap() > 0, "{r}");
        assert_eq!(size["recovered_in_bytes"], size["total_in_bytes"], "{r}");
        assert_eq!((size["reused_in_bytes"].as_u64(), size["recovered_from_snapshot_in_bytes"].as_u64()), (Some(0), Some(0)), "{r}");
        assert_eq!((files["total"].as_u64(), files["recovered"].as_u64(), files["reused"].as_u64()), (Some(4), Some(4), Some(0)), "a Lucene shard's commit plus one segment: {r}");
        assert_eq!((files["percent"].as_str(), size["percent"].as_str()), (Some("100.0%"), Some("100.0%")), "{r}");
        assert_eq!((r["translog"]["recovered"].as_u64(), r["translog"]["total"].as_u64(), r["translog"]["total_on_start"].as_u64()), (Some(0), Some(0), Some(0)), "{r}");
        let took = r["total_time_in_millis"].as_u64().unwrap();
        assert!(took as u128 <= started.elapsed().as_millis() + 1000, "{r}");
    }
    let v = three.ok(0, Method::GET, "/r/_recovery?detailed=true", None).await;
    assert!(v["r"]["shards"].as_array().unwrap().iter().all(|s| s["index"]["files"]["details"].is_array()), "{v}");
    let v = three.ok(0, Method::GET, "/r/_recovery?active_only=true", None).await;
    assert!(v.as_object().is_some_and(|o| o.is_empty()), "nothing in flight: {v}");
    let v = three.ok(0, Method::GET, "/_cat/recovery/r?format=json&h=index,shard,type,stage,repository,snapshot,files_percent,bytes_percent", None).await;
    let rows = v.as_array().unwrap();
    assert_eq!(rows.len(), 2, "{v}");
    for row in rows {
        assert_eq!(row["type"], "snapshot", "{row}");
        assert_eq!(row["stage"], "done", "{row}");
        assert_eq!((row["repository"].as_str(), row["snapshot"].as_str()), (Some("repo"), Some("snap")), "{row}");
        assert_eq!((row["files_percent"].as_str(), row["bytes_percent"].as_str()), (Some("100.0%"), Some("100.0%")), "{row}");
    }
}

/// Production review #377, #378, #379 on a cluster:
/// - a global-state restore replaces composable templates (a template
///   created after the snapshot is gone afterwards);
/// - a restored index is created from the snapshot's metadata only: a
///   current template's alias and pipeline are not added;
/// - the snapshot's ILM policies are installed before restored indices
///   resume their lifecycle (a destination policy under the same name that
///   would delete at once never acts on them).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn restore_uses_the_snapshot_templates_and_policies() {
    let _serial = SERIAL.lock().await;
    let three = start("sfr-global").await;
    three.ok(0, Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"indices.lifecycle.poll_interval": "1s"}}))).await;
    three.ok(0, Method::PUT, "/_ilm/policy/keep", Some(json!({"policy": {"phases": {"hot": {"actions": {}}, "delete": {"min_age": "30d", "actions": {"delete": {}}}}}}))).await;
    three.ok(0, Method::PUT, "/_index_template/tpl-a", Some(json!({"index_patterns": ["gs-*"], "priority": 10, "template": {"settings": {"number_of_replicas": 0}}}))).await;
    three.ok(0, Method::PUT, "/orig", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.lifecycle.name": "keep"}}))).await;
    three.green("orig", 1).await;
    three.ndjson(format!("{}\n{}\n", json!({"index": {"_index": "orig", "_id": "1"}}), json!({"n": 1}))).await;
    let v = three.ok(0, Method::PUT, "/_snapshot/repo/gsnap?wait_for_completion=true", Some(json!({"indices": "orig", "include_global_state": true}))).await;
    assert_eq!(v["snapshot"]["state"], "SUCCESS", "{v}");
    // After the snapshot: a newer template, a template for the restored
    // name with an alias and a pipeline, and a destructive same-name policy.
    three.ok(0, Method::PUT, "/_index_template/tpl-new", Some(json!({"index_patterns": ["gs-*"], "priority": 100, "template": {"settings": {"number_of_replicas": 0}}}))).await;
    three.ok(0, Method::PUT, "/_ingest/pipeline/tag", Some(json!({"processors": [{"set": {"field": "tagged", "value": true}}]}))).await;
    three.ok(0, Method::PUT, "/_index_template/live", Some(json!({"index_patterns": ["arch-*"], "priority": 50, "template": {"aliases": {"live-read": {}}, "settings": {"index.default_pipeline": "tag"}}}))).await;
    three.ok(0, Method::PUT, "/_ilm/policy/keep", Some(json!({"policy": {"phases": {"hot": {"actions": {}}, "delete": {"min_age": "0ms", "actions": {"delete": {}}}}}}))).await;
    three.ok(0, Method::DELETE, "/orig", None).await;
    // #378: a renamed restore without aliases, no global state. Without
    // the snapshot's policies the restored copy would follow the
    // destination's `keep` (delete at 0ms, polled every second) and could be
    // deleted while this part waits for it (a race, as in ES): its
    // lifecycle is left out here; the global-state restore below covers it.
    let (st, v) = three.restore("gsnap", json!({"indices": "orig", "rename_pattern": "orig", "rename_replacement": "arch-orig", "include_aliases": false, "ignore_index_settings": ["index.lifecycle.name"]})).await;
    assert_eq!(st, 200, "{v}");
    three.green("arch-orig", 1).await;
    let v = three.ok(0, Method::GET, "/arch-orig", None).await;
    assert_eq!(v["arch-orig"]["aliases"], json!({}), "no current template alias: {v}");
    assert!(v["arch-orig"]["settings"]["index"].get("default_pipeline").is_none(), "no current template pipeline: {v}");
    three.ok(0, Method::DELETE, "/arch-orig", None).await;
    // #377 + #379: global-state restore.
    let (st, v) = three.restore("gsnap", json!({"indices": "orig", "include_global_state": true})).await;
    assert_eq!(st, 200, "{v}");
    let (st, _) = three.call(0, Method::GET, "/_index_template/tpl-new", None).await;
    assert_eq!(st, 404, "a template created after the snapshot is replaced away");
    three.ok(0, Method::GET, "/_index_template/tpl-a", None).await;
    let v = three.ok(0, Method::GET, "/_ilm/policy/keep", None).await;
    assert_eq!(v["keep"]["policy"]["phases"]["delete"]["min_age"], json!("30d"), "{v}");
    tokio::time::sleep(Duration::from_secs(4)).await;
    let v = three.ok(0, Method::GET, "/orig/_count", None).await;
    assert_eq!(v["count"], 1, "the restored index was not deleted by the destination's policy: {v}");
}

/// Production review #381: a snapshot an active restore still reads cannot
/// be deleted (SLM retention included); once the restore finished it can.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_deletion_waits_for_an_active_restore() {
    let _serial = SERIAL.lock().await;
    let three = start("sfr-del-restore").await;
    three.vector_index("rs", 1, 0).await;
    three.bulk("rs", 0..50).await;
    three.snapshot("snap", "rs").await;
    three.ok(0, Method::DELETE, "/rs", None).await;
    let entered = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let (e, r) = (entered.clone(), release.clone());
    *elasticsearch_node::snapshot_api::restore::BEFORE_RESTORE_INDEX.lock() = Some(Arc::new(move || {
        let (e, r) = (e.clone(), r.clone());
        Box::pin(async move {
            e.notify_one();
            r.notified().await;
        }) as std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
    }));
    let interfere = async {
        tokio::time::timeout(Duration::from_secs(30), entered.notified()).await.expect("the restore starts");
        let (st, v) = three.call(1, Method::DELETE, "/_snapshot/repo/snap", None).await;
        release.notify_one();
        (st, v)
    };
    let ((st, v), (dst, dv)) = tokio::join!(three.restore("snap", json!({"indices": "rs"})), interfere);
    *elasticsearch_node::snapshot_api::restore::BEFORE_RESTORE_INDEX.lock() = None;
    assert_eq!((dst, dv["error"]["type"].as_str()), (503, Some("concurrent_snapshot_execution_exception")), "{dv}");
    assert_eq!(st, 200, "the restore completes: {v}");
    three.green("rs", 1).await;
    let v = three.ok(0, Method::GET, "/rs/_count", None).await;
    assert_eq!(v["count"], 50, "{v}");
    three.ok(0, Method::DELETE, "/_snapshot/repo/snap", None).await;
}

/// Production review #393: `feature_states` selects the feature's system
/// indices into the snapshot and the restore brings them back, replacing
/// the feature's current (open) indices as ES does; an unknown feature is
/// rejected (400 on create, ES's restore error on restore).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn feature_state_snapshots_carry_the_system_indices() {
    let _serial = SERIAL.lock().await;
    let three = start("sfr-features").await;
    three.ok(0, Method::PUT, "/.kibana_7", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.hidden": true}}))).await;
    three.green(".kibana_7", 1).await;
    three.ndjson(format!("{}\n{}\n", json!({"index": {"_index": ".kibana_7", "_id": "cfg"}}), json!({"n": 7}))).await;
    three.ok(0, Method::PUT, "/plain", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}}))).await;
    three.green("plain", 1).await;
    let (st, v) = three.call(0, Method::PUT, "/_snapshot/repo/nope?wait_for_completion=true", Some(json!({"indices": "plain", "feature_states": ["no_such_feature"]}))).await;
    assert_eq!((st, v["error"]["type"].as_str()), (400, Some("illegal_argument_exception")), "{v}");
    let v = three.ok(0, Method::PUT, "/_snapshot/repo/fs?wait_for_completion=true", Some(json!({"indices": "plain", "include_global_state": false, "feature_states": ["kibana"]}))).await;
    assert_eq!(v["snapshot"]["state"], "SUCCESS", "{v}");
    let idx: Vec<&str> = v["snapshot"]["indices"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
    assert!(idx.contains(&".kibana_7") && idx.contains(&"plain"), "the feature's index is in the snapshot: {v}");
    let fs = &v["snapshot"]["feature_states"];
    assert_eq!(fs[0]["feature_name"], "kibana", "{v}");
    assert_eq!(fs[0]["indices"], json!([".kibana_7"]), "{v}");
    // The feature's indices stay open and change after the snapshot; a
    // newer one appears. The feature-state restore replaces them all (ES).
    three.ndjson(format!("{}\n{}\n", json!({"index": {"_index": ".kibana_7", "_id": "later"}}), json!({"n": 8}))).await;
    three.ok(0, Method::PUT, "/.kibana_8", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.hidden": true}}))).await;
    three.green(".kibana_8", 1).await;
    three.ok(0, Method::DELETE, "/plain", None).await;
    let (st, v) = three.restore("fs", json!({"indices": "plain", "include_global_state": false})).await;
    assert_eq!(st, 200, "no feature state requested: {v}");
    three.ok(0, Method::DELETE, "/plain", None).await;
    let (st, _) = three.call(0, Method::GET, "/.kibana_8", None).await;
    assert_eq!(st, 200, "without feature states the live system indices stay");
    let (st, v) = three.restore("fs", json!({"indices": "plain", "feature_states": ["security"]})).await;
    assert_eq!((st, v["error"]["type"].as_str()), (500, Some("snapshot_restore_exception")), "{v}");
    let (st, v) = three.restore("fs", json!({"indices": "plain", "feature_states": ["kibana"]})).await;
    assert_eq!(st, 200, "{v}");
    three.green(".kibana_7", 1).await;
    three.ok(0, Method::POST, "/.kibana_7/_refresh", None).await;
    let v = three.ok(0, Method::GET, "/.kibana_7/_count", None).await;
    assert_eq!(v["count"], 1, "the feature state is restored over the open index: {v}");
    let (st, _) = three.call(0, Method::GET, "/.kibana_7/_doc/later", None).await;
    assert_eq!(st, 404, "the post-snapshot document is gone");
    let (st, _) = three.call(0, Method::GET, "/.kibana_8", None).await;
    assert_eq!(st, 404, "the feature's other current index is deleted");
}
