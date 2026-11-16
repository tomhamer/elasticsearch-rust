//! `_clone` / `_split` / `_shrink` answer success only with a complete
//! target (`resize`): every source document with the same `_source`,
//! `_version` and `_routing`, each on its shard, no pipeline run on the
//! copies. A target primary that does not start in time (allocation
//! delay), rejected bulk items, or a lost node fail the resize instead, and
//! the target is left explicitly incomplete (read and write blocked).

mod common;

use common::procs::start_procs;
use common::Cluster;
use reqwest::Method;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::time::Duration;

/// The tests share the process-wide copy seam.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

type Contents = BTreeMap<String, (Value, u64, Option<String>)>;

async fn ok(c: &Cluster, method: Method, path: &str, body: Option<Value>) -> Value {
    let (st, v) = c.call(c.master, method.clone(), path, body).await;
    assert!((200..300).contains(&st), "{method} {path}: {st} {v}");
    v
}

fn docs(index: &str, ids: std::ops::Range<usize>, v: &str) -> String {
    ids.map(|i| {
        let mut meta = json!({"_index": index, "_id": format!("d{i}")});
        if i % 3 == 0 {
            meta["routing"] = json!(format!("r{}", i % 7));
        }
        format!("{}\n{}\n", json!({"index": meta}), json!({"n": i, "v": v}))
    }).collect()
}

async fn bulk(c: &Cluster, nd: String) {
    let r = c.c.post(format!("http://127.0.0.1:{}/_bulk?refresh=true", c.p.http[c.master])).header("content-type", "application/x-ndjson").body(nd).send().await.unwrap();
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["errors"], false, "{v}");
}

/// Every document of `index` (id → source, version, routing), read
/// through a search of all shards.
async fn contents(c: &Cluster, index: &str) -> Contents {
    let v = ok(c, Method::POST, &format!("/{index}/_search"), Some(json!({"size": 10000, "version": true, "query": {"match_all": {}}}))).await;
    assert_eq!(v["_shards"]["failed"], 0, "{v}");
    v["hits"]["hits"].as_array().unwrap_or_else(|| panic!("{v}")).iter()
        .map(|h| (h["_id"].as_str().unwrap().to_string(), (h["_source"].clone(), h["_version"].as_u64().unwrap(), h["_routing"].as_str().map(str::to_string))))
        .collect()
}

/// A write-blocked source with a default and a final pipeline (each
/// appends: running again would change the documents), routed and
/// updated documents.
async fn source(c: &Cluster, name: &str, shards: u32) -> Contents {
    ok(c, Method::PUT, "/_ingest/pipeline/dflt", Some(json!({"processors": [{"append": {"field": "dflt", "value": ["y"]}}]}))).await;
    ok(c, Method::PUT, "/_ingest/pipeline/fin", Some(json!({"processors": [{"append": {"field": "fin", "value": ["x"]}}]}))).await;
    ok(c, Method::PUT, &format!("/{name}"), Some(json!({"settings": {"number_of_shards": shards, "number_of_replicas": 0, "index.default_pipeline": "dflt", "index.final_pipeline": "fin"}}))).await;
    bulk(c, docs(name, 0..1200, "first")).await;
    bulk(c, docs(name, 0..30, "second")).await;
    ok(c, Method::PUT, &format!("/{name}/_settings"), Some(json!({"index.blocks.write": true}))).await;
    let before = contents(c, name).await;
    assert_eq!(before.len(), 1200);
    assert_eq!(before["d0"].1, 2);
    before
}

async fn assert_incomplete(c: &Cluster, target: &str) {
    let v = ok(c, Method::GET, &format!("/{target}/_settings?flat_settings=true"), None).await;
    let s = &v[target]["settings"];
    assert_eq!((s["index.blocks.write"].as_str(), s["index.blocks.read"].as_str()), (Some("true"), Some("true")), "the incomplete target is not blocked: {v}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn clone_split_and_shrink_answer_only_with_an_exact_copy() {
    let _serial = SERIAL.lock().await;
    let c = common::start(&[]).await;
    let before = source(&c, "src", 2).await;
    for (kind, target, shards) in [("_clone", "cloned", 2), ("_split", "split", 4), ("_shrink", "shrunk", 1)] {
        let v = ok(&c, Method::POST, &format!("/src/{kind}/{target}?wait_for_active_shards=1"), Some(json!({"settings": {"index.number_of_shards": shards, "index.number_of_replicas": 0}}))).await;
        assert_eq!((v["acknowledged"].as_bool(), v["shards_acknowledged"].as_bool(), v["index"].as_str()), (Some(true), Some(true), Some(target)), "{v}");
        assert_eq!(contents(&c, target).await, before, "{kind}: the target differs from the source (or a pipeline ran on the copies)");
        // Every document is found by id (its routing selects its shard).
        for (id, (_, _, routing)) in before.iter().take(50) {
            let path = match routing { Some(r) => format!("/{target}/_doc/{id}?routing={r}"), None => format!("/{target}/_doc/{id}") };
            let d = ok(&c, Method::GET, &path, None).await;
            assert_eq!(d["found"], true, "{kind}: {d}");
        }
        let s = ok(&c, Method::GET, &format!("/{target}/_settings?flat_settings=true"), None).await;
        assert_eq!(s[target]["settings"]["index.blocks.write"], "true", "{kind}: the target keeps the source's write block: {s}");
        assert!(s[target]["settings"].get("index.blocks.read").is_none(), "{kind}: {s}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_target_primary_that_does_not_start_fails_the_resize() {
    let _serial = SERIAL.lock().await;
    let c = common::start(&[]).await;
    source(&c, "src", 2).await;
    let (st, v) = c.call(c.master, Method::POST, "/src/_split/stuck?timeout=2s", Some(json!({"settings": {"index.number_of_shards": 4, "index.routing.allocation.require._name": "nowhere"}}))).await;
    assert_eq!(st, 500, "an unallocated target reported success: {v}");
    assert!(v["error"]["reason"].as_str().is_some_and(|r| r.contains("did not complete") && r.contains("stuck")), "{v}");
    assert_incomplete(&c, "stuck").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rejected_copy_items_fail_the_resize_and_a_retry_succeeds() {
    let _serial = SERIAL.lock().await;
    let c = common::start(&[]).await;
    let before = source(&c, "src", 2).await;
    elasticsearch_node::resize::REJECT_COPY_ITEMS.store(3, Ordering::SeqCst);
    let (st, v) = c.call(c.master, Method::POST, "/src/_clone/partial", None).await;
    elasticsearch_node::resize::REJECT_COPY_ITEMS.store(0, Ordering::SeqCst);
    assert_eq!(st, 500, "a partial copy reported success: {v}");
    assert!(v["error"]["reason"].as_str().is_some_and(|r| r.contains("bulk item")), "{v}");
    assert_incomplete(&c, "partial").await;
    ok(&c, Method::DELETE, "/partial", None).await;
    let v = ok(&c, Method::POST, "/src/_clone/partial", None).await;
    assert_eq!(v["shards_acknowledged"], true, "{v}");
    assert_eq!(contents(&c, "partial").await, before);
}

/// A node holding source and target copies is lost while a split copies:
/// the resize either fails (its target left incomplete or gone) or
/// answers with a complete target, never with a partial one.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn node_loss_during_the_copy_never_yields_a_partial_success() {
    let mut pl = start_procs(3, &[]).await;
    let m = pl.master(0).await;
    let victim = (0..3).find(|i| *i != m).unwrap();
    let (st, v) = pl.call(m, Method::PUT, "/src", Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 1, "index.max_result_window": 30000}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(m, "src", "green", 90).await;
    for chunk in (0..20_000).collect::<Vec<usize>>().chunks(2000) {
        let (st, v) = pl.ndjson(m, "/_bulk", docs("src", chunk[0]..chunk[chunk.len() - 1] + 1, "first")).await;
        assert!(st == 200 && v["errors"] == false, "{st} {v}");
    }
    let (st, v) = pl.call(m, Method::POST, "/src/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = pl.call(m, Method::PUT, "/src/_settings", Some(json!({"index.blocks.write": true}))).await;
    assert_eq!(st, 200, "{v}");
    let all = |pl: &common::procs::Procs, index: &'static str| {
        let (c, port) = (pl.c.clone(), pl.p.http[m]);
        async move {
            let v: Value = c.post(format!("http://127.0.0.1:{port}/{index}/_search")).json(&json!({"size": 20000, "version": true, "query": {"match_all": {}}}))
                .send().await.unwrap().json().await.unwrap();
            assert_eq!(v["_shards"]["failed"], 0, "{v}");
            v["hits"]["hits"].as_array().unwrap().iter()
                .map(|h| (h["_id"].as_str().unwrap().to_string(), (h["_source"].clone(), h["_version"].as_u64().unwrap(), h["_routing"].as_str().map(str::to_string))))
                .collect::<Contents>()
        }
    };
    let before = all(&pl, "src").await;
    assert_eq!(before.len(), 20_000);
    let (c, port) = (pl.c.clone(), pl.p.http[m]);
    let resize = tokio::spawn(async move {
        let r = c.post(format!("http://127.0.0.1:{port}/src/_split/target?timeout=60s")).timeout(Duration::from_secs(300))
            .json(&json!({"settings": {"index.number_of_shards": 4, "index.number_of_replicas": 1}})).send().await.unwrap();
        let st = r.status().as_u16();
        (st, r.json::<Value>().await.unwrap_or(Value::Null))
    });
    tokio::time::sleep(Duration::from_millis(700)).await;
    pl.kill(victim);
    let (st, v) = resize.await.unwrap();
    eprintln!("resize after node loss answered {st}: {v}");
    pl.wait_nodes(m, 2, 120).await;
    if st == 200 {
        pl.wait_health(m, "target", "yellow", 120).await;
        assert_eq!(all(&pl, "target").await, before, "the resize reported success with a target that differs from the source: {v}");
    } else {
        assert!(st >= 500, "{st} {v}");
        let (exists, _) = pl.call(m, Method::HEAD, "/target", None).await;
        if exists == 200 {
            let (_, s) = pl.call(m, Method::GET, "/target/_settings?flat_settings=true", None).await;
            assert_eq!(s["target"]["settings"]["index.blocks.read"], "true", "a failed resize left an unmarked target: {s}");
        }
    }
}
