//! `_update_by_query`, `_delete_by_query` and `_reindex` on three node
//! processes ([`common::procs`]), as ES's BulkByScroll actions behave:
//! - a filtered alias restricts the operation to the alias's documents and
//!   alias `search_routing` to the shards it selects, through every
//!   coordinator fan-out (multi-shard by-query, single-shard by-query
//!   through an alias, the background by-query task, distributed reindex);
//! - a source read that misses a shard never reports a complete run;
//! - a background by-query completes only once its writes are on the
//!   in-sync replicas: delayed replication keeps it pending, and every
//!   change it reports survives the loss of the primary right after.
mod common;
use common::procs::{start_procs, Procs};
use reqwest::Method;
use serde_json::{json, Value};
use std::time::{Duration, Instant};

async fn create(pl: &Procs, via: usize, index: &str, settings: Value) {
    let (st, v) = pl.call(via, Method::PUT, &format!("/{index}"), Some(json!({"settings": settings, "mappings": {"properties": {"tag": {"type": "keyword"}, "v": {"type": "long"}}}}))).await;
    assert_eq!(st, 200, "create {index}: {v}");
}

async fn bulk(pl: &Procs, via: usize, lines: String) {
    let (st, v) = pl.ndjson(via, "/_bulk?refresh=true", lines).await;
    assert!(st == 200 && v["errors"] == false, "bulk: {st} {v}");
}

/// Matching documents of `target` (every shard answered).
async fn count(pl: &Procs, via: usize, target: &str, query: Value) -> u64 {
    let (st, v) = pl.call(via, Method::POST, &format!("/{target}/_search?request_cache=false"), Some(json!({"size": 0, "track_total_hits": true, "query": query}))).await;
    assert!(st == 200 && v["_shards"]["failed"] == 0, "count {target}: {st} {v}");
    v["hits"]["total"]["value"].as_u64().unwrap()
}

fn tag(t: &str) -> Value {
    json!({"term": {"tag": t}})
}

/// A 200 BulkByScroll response without failures.
async fn ok(pl: &Procs, via: usize, path: &str, body: Value) -> Value {
    let (st, v) = pl.call(via, Method::POST, path, Some(body)).await;
    assert!(st == 200 && v["failures"] == json!([]), "{path}: {st} {v}");
    v
}

/// The task's `GET _tasks/{id}` once completed.
async fn completed(pl: &Procs, via: usize, task: &str, secs: u64) -> Value {
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        let (st, v) = pl.call(via, Method::GET, &format!("/_tasks/{task}"), None).await;
        if st == 200 && v["completed"] == true {
            return v;
        }
        assert!(Instant::now() < deadline, "task {task} did not complete: {st} {v}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// The nodes holding a started copy of shard `k` of `index`, primary first.
async fn holders(pl: &Procs, via: usize, index: &str, k: u64) -> Vec<usize> {
    let mut rows: Vec<Value> = pl.shards(via, index).await.into_iter().filter(|r| r["shard"] == k.to_string() && r["state"] == "STARTED").collect();
    rows.sort_by_key(|r| r["prirep"] != "p");
    rows.iter().filter_map(|r| r["node"].as_str()?.trim_start_matches('n').parse().ok()).collect()
}

async fn wait_green(pl: &Procs, via: usize, index: &str) {
    pl.wait_health(via, index, "green", 90).await;
}

/// Two groups of documents behind two filtered aliases: update, delete
/// and reindex through one leave the other untouched, on a multi-shard
/// index (per-shard fan-out), a single-shard index (fan-out over one
/// target) and in a background task. A routed alias over a multi-shard
/// index: by-query and reindex reach only the shard its routing selects.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn alias_filters_and_routing_hold_through_every_fan_out() {
    let pl = start_procs(3, &[]).await;
    let m = pl.master(0).await;
    create(&pl, m, "grp", json!({"number_of_shards": 3, "number_of_replicas": 0})).await;
    create(&pl, m, "one", json!({"number_of_shards": 1, "number_of_replicas": 0})).await;
    create(&pl, m, "rt", json!({"number_of_shards": 3, "number_of_replicas": 0})).await;
    for i in ["grp", "one", "rt"] {
        wait_green(&pl, m, i).await;
    }
    let mut lines = String::new();
    for i in 0..60 {
        lines += &format!("{}\n{}\n", json!({"index": {"_index": "grp", "_id": format!("g{i}")}}), json!({"tag": if i % 2 == 0 { "a" } else { "b" }, "v": i}));
    }
    for i in 0..20 {
        lines += &format!("{}\n{}\n", json!({"index": {"_index": "one", "_id": format!("o{i}")}}), json!({"tag": if i % 2 == 0 { "a" } else { "b" }, "v": i}));
    }
    for i in 0..90 {
        lines += &format!("{}\n{}\n", json!({"index": {"_index": "rt", "_id": format!("r{i}"), "routing": format!("k{}", i % 9)}}), json!({"tag": "x", "v": i}));
    }
    bulk(&pl, m, lines).await;
    let (st, v) = pl.call(m, Method::POST, "/_aliases", Some(json!({"actions": [
        {"add": {"index": "grp", "alias": "ga", "filter": tag("a")}},
        {"add": {"index": "grp", "alias": "gb", "filter": tag("b")}},
        {"add": {"index": "one", "alias": "oa", "filter": tag("a")}},
        {"add": {"index": "one", "alias": "ob", "filter": tag("b")}},
        {"add": {"index": "rt", "alias": "rr", "routing": "k1"}},
    ]}))).await;
    assert_eq!(st, 200, "{v}");
    let via = (m + 1) % 3;

    // Synchronous update through `gb` (three shards).
    let v = ok(&pl, via, "/gb/_update_by_query?refresh=true", json!({"script": {"source": "ctx._source.touched = true"}})).await;
    assert_eq!((v["total"].as_u64(), v["updated"].as_u64()), (Some(30), Some(30)), "{v}");
    assert_eq!(count(&pl, m, "grp", json!({"bool": {"filter": [{"exists": {"field": "touched"}}, tag("a")]}})).await, 0, "update through gb touched group a");
    assert_eq!(count(&pl, m, "grp", json!({"exists": {"field": "touched"}})).await, 30);

    // Background update through `ga` (one cluster task over three shards).
    let (st, t) = pl.call(via, Method::POST, "/ga/_update_by_query?wait_for_completion=false&refresh=true", Some(json!({"script": {"source": "ctx._source.bg = true"}}))).await;
    assert_eq!(st, 200, "{t}");
    let done = completed(&pl, via, t["task"].as_str().unwrap(), 60).await;
    assert_eq!(done["response"]["updated"].as_u64(), Some(30), "{done}");
    let (st, v) = pl.call(m, Method::POST, "/grp/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(count(&pl, m, "grp", json!({"bool": {"filter": [{"exists": {"field": "bg"}}, tag("b")]}})).await, 0, "background update through ga touched group b");

    // Distributed reindex from `gb`: group b only.
    let v = ok(&pl, via, "/_reindex?refresh=true", json!({"source": {"index": "gb"}, "dest": {"index": "gb_copy"}})).await;
    assert_eq!(v["created"].as_u64(), Some(30), "{v}");
    assert_eq!(count(&pl, m, "gb_copy", json!({"match_all": {}})).await, 30);
    assert_eq!(count(&pl, m, "gb_copy", tag("a")).await, 0, "reindex from gb copied group a");

    // match_all delete through `ga`: group b stays.
    let v = ok(&pl, via, "/ga/_delete_by_query?refresh=true", json!({"query": {"match_all": {}}})).await;
    assert_eq!(v["deleted"].as_u64(), Some(30), "{v}");
    assert_eq!((count(&pl, m, "grp", tag("a")).await, count(&pl, m, "grp", tag("b")).await), (0, 30), "delete through ga");

    // The single-shard index through its filtered alias: `q` and no body.
    let v = ok(&pl, via, "/oa/_delete_by_query?refresh=true&q=v:%5B0%20TO%20100%5D", json!({})).await;
    assert_eq!(v["deleted"].as_u64(), Some(10), "{v}");
    let v = ok(&pl, via, "/ob/_update_by_query?refresh=true", json!({})).await;
    assert_eq!(v["updated"].as_u64(), Some(10), "{v}");
    assert_eq!((count(&pl, m, "one", tag("a")).await, count(&pl, m, "one", tag("b")).await), (0, 10), "delete through oa");

    // A routed alias over three shards: only the shard `k1` selects.
    let routed = count(&pl, m, "rr", json!({"match_all": {}})).await;
    assert!(routed > 0 && routed < 90, "k1 selects one shard of three: {routed}");
    let (st, v) = pl.call(m, Method::POST, "/rt/_search?routing=k1&request_cache=false", Some(json!({"size": 0, "track_total_hits": true}))).await;
    assert_eq!((st, v["hits"]["total"]["value"].as_u64()), (200, Some(routed)), "{v}");
    let v = ok(&pl, via, "/_reindex?refresh=true", json!({"source": {"index": "rr"}, "dest": {"index": "rr_copy"}})).await;
    assert_eq!(v["created"].as_u64(), Some(routed), "{v}");
    assert_eq!(count(&pl, m, "rr_copy", json!({"match_all": {}})).await, routed);
    let v = ok(&pl, via, "/rr/_update_by_query?refresh=true", json!({"script": {"source": "ctx._source.r = true"}})).await;
    assert_eq!(v["updated"].as_u64(), Some(routed), "{v}");
    assert_eq!(count(&pl, m, "rt", json!({"exists": {"field": "r"}})).await, routed);
    let v = ok(&pl, via, "/rr/_delete_by_query?refresh=true", json!({"query": {"match_all": {}}})).await;
    assert_eq!(v["deleted"].as_u64(), Some(routed), "{v}");
    assert_eq!(count(&pl, m, "rt", json!({"match_all": {}})).await, 90 - routed, "delete through rr");
}

/// One source shard is lost while the other answers: reindex and by-query
/// report the failure instead of a complete run.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_missing_source_shard_is_never_a_complete_run() {
    let mut pl = start_procs(3, &[]).await;
    let m = pl.master(0).await;
    create(&pl, m, "part", json!({"number_of_shards": 2, "number_of_replicas": 0, "index.routing.allocation.exclude._name": format!("n{m}"), "index.routing.allocation.total_shards_per_node": 1})).await;
    wait_green(&pl, m, "part").await;
    let lines: String = (0..200).map(|i| format!("{}\n{}\n", json!({"index": {"_index": "part", "_id": format!("p{i}")}}), json!({"tag": "x", "v": i}))).collect();
    bulk(&pl, m, lines).await;
    let on0 = count(&pl, m, "part", json!({"match_all": {}})).await;
    assert_eq!(on0, 200);
    let victim = holders(&pl, m, "part", 1).await[0];
    let survivor = holders(&pl, m, "part", 0).await[0];
    assert!(victim != m && survivor != m && victim != survivor);
    pl.stop(victim, 30).await;
    pl.wait_nodes(m, 2, 60).await;

    // Distributed reindex (a two-shard source).
    let (st, v) = pl.call(m, Method::POST, "/_reindex?refresh=true", Some(json!({"source": {"index": "part"}, "dest": {"index": "part_copy"}}))).await;
    let failures = v["failures"].as_array().map(Vec::len).unwrap_or(0);
    assert!(st >= 400 && (failures > 0 || v["error"].is_object()), "reindex over a missing shard claimed success: {st} {v}");
    assert!(v["created"].as_u64().unwrap_or(0) < 200, "{v}");

    // By-query over both shards.
    let (st, v) = pl.call(m, Method::POST, "/part/_update_by_query?refresh=true&timeout=2s", Some(json!({"script": {"source": "ctx._source.u = true"}}))).await;
    let failures = v["failures"].as_array().map(Vec::len).unwrap_or(0);
    assert!(st >= 400 && (failures > 0 || v["error"].is_object()), "update_by_query over a missing shard claimed success: {st} {v}");
    let (st, v) = pl.call(m, Method::POST, "/part/_delete_by_query?refresh=true&timeout=2s", Some(json!({"query": {"match_all": {}}}))).await;
    let failures = v["failures"].as_array().map(Vec::len).unwrap_or(0);
    assert!(st >= 400 && (failures > 0 || v["error"].is_object()), "delete_by_query over a missing shard claimed success: {st} {v}");
}

/// Wait until shard 0 of `index` has a started primary off `dead`.
async fn wait_promoted(pl: &Procs, via: usize, index: &str, dead: usize) {
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        if holders(pl, via, index, 0).await.first().is_some_and(|p| *p != dead) && pl.shards(via, index).await.iter().any(|r| r["shard"] == "0" && r["prirep"] == "p" && r["state"] == "STARTED") {
            return;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    panic!("{index}: no primary off n{dead}: {}", json!(pl.shards(via, index).await));
}

/// A background update-by-query whose replica is hung stays pending (its
/// batches wait for the replica); once it completes, the primary's node is
/// killed at once and the promoted replica holds every change the task
/// reported.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_background_by_query_completes_only_once_replicated() {
    let mut pl = start_procs(3, &[]).await;
    let m = pl.master(0).await;
    create(&pl, m, "acked", json!({"number_of_shards": 1, "number_of_replicas": 1, "index.routing.allocation.exclude._name": format!("n{m}"), "index.unassigned.node_left.delayed_timeout": "0"})).await;
    wait_green(&pl, m, "acked").await;
    const DOCS: u64 = 2000;
    let lines: String = (0..DOCS).map(|i| format!("{}\n{}\n", json!({"index": {"_index": "acked", "_id": format!("d{i}")}}), json!({"tag": "x", "v": 0}))).collect();
    bulk(&pl, m, lines).await;
    let copies = holders(&pl, m, "acked", 0).await;
    let (primary, replica) = (copies[0], copies[1]);
    assert!(primary != m && replica != m);

    // About half a second of throttled batches.
    let (st, t) = pl.call(primary, Method::POST, "/acked/_update_by_query?wait_for_completion=false&scroll_size=100&requests_per_second=4000", Some(json!({"script": {"source": "ctx._source.v = 1"}}))).await;
    assert_eq!(st, 200, "{t}");
    let task = t["task"].as_str().unwrap().to_string();
    pl.pause(replica);
    tokio::time::sleep(Duration::from_millis(2500)).await;
    let (st, v) = pl.call(primary, Method::GET, &format!("/_tasks/{task}"), None).await;
    pl.resume(replica);
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["completed"], false, "completed while its replica could not have the writes: {v}");

    let done = completed(&pl, primary, &task, 90).await;
    let r = &done["response"];
    assert_eq!((r["updated"].as_u64(), r["failures"].clone()), (Some(DOCS), json!([])), "{done}");

    // Lose the primary right after the acknowledgement.
    pl.kill(primary);
    pl.wait_nodes(m, 2, 60).await;
    wait_promoted(&pl, m, "acked", primary).await;
    let (st, v) = pl.call(m, Method::POST, "/acked/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(count(&pl, m, "acked", json!({"term": {"v": 1}})).await, DOCS, "changes the task reported were lost with the primary");
}

/// A synchronous reindex whose script sends every document to index B
/// (another index than its `dest`) acknowledges only once B's replica has
/// them: with B's replica hung the request stays unanswered, and after the
/// acknowledgement B's primary is lost and its promoted replica holds every
/// document.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_redirected_reindex_acknowledges_the_index_it_wrote() {
    let mut pl = start_procs(3, &[]).await;
    let m = pl.master(0).await;
    create(&pl, m, "rdb", json!({"number_of_shards": 1, "number_of_replicas": 1, "index.routing.allocation.exclude._name": format!("n{m}"), "index.unassigned.node_left.delayed_timeout": "0"})).await;
    wait_green(&pl, m, "rdb").await;
    let copies = holders(&pl, m, "rdb", 0).await;
    let (p, r) = (copies[0], copies[1]);
    for i in ["rsrc", "rda"] {
        create(&pl, m, i, json!({"number_of_shards": 1, "number_of_replicas": 0, "index.routing.allocation.require._name": format!("n{p}")})).await;
        wait_green(&pl, m, i).await;
    }
    const DOCS: u64 = 300;
    let lines: String = (0..DOCS).map(|i| format!("{}\n{}\n", json!({"index": {"_index": "rsrc", "_id": format!("s{i}")}}), json!({"tag": "x", "v": i}))).collect();
    bulk(&pl, m, lines).await;

    pl.pause(r);
    let (c, url) = (pl.c.clone(), format!("http://127.0.0.1:{}/_reindex", pl.p.http[m]));
    let call = tokio::spawn(async move {
        let resp = c.post(url).json(&json!({"source": {"index": "rsrc"}, "dest": {"index": "rda"}, "script": {"source": "ctx._index = 'rdb'"}})).timeout(Duration::from_secs(120)).send().await.unwrap();
        let st = resp.status().as_u16();
        (st, resp.json::<Value>().await.unwrap_or(Value::Null))
    });
    tokio::time::sleep(Duration::from_millis(2500)).await;
    let early = call.is_finished();
    pl.resume(r);
    assert!(!early, "acknowledged while B's replica could not have the documents");
    let (st, v) = call.await.unwrap();
    assert!(st == 200 && v["failures"] == json!([]) && v["created"].as_u64() == Some(DOCS), "{st} {v}");

    pl.kill(p);
    pl.wait_nodes(m, 2, 60).await;
    wait_promoted(&pl, m, "rdb", p).await;
    let (st, v) = pl.call(m, Method::POST, "/rdb/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(count(&pl, m, "rdb", json!({"match_all": {}})).await, DOCS, "documents reindexed into B were lost with its primary");
}

/// Two background by-query tasks whose children start together on one
/// engine: each child keeps its own parent; cancelling and rethrottling A
/// leaves B's child at its own rate and B runs to completion.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_parents_keep_their_own_children() {
    let pl = start_procs(3, &[]).await;
    let m = pl.master(0).await;
    let home = (m + 1) % 3;
    for i in ["ia1", "ia2", "ib1", "ib2"] {
        create(&pl, m, i, json!({"number_of_shards": 1, "number_of_replicas": 0, "index.routing.allocation.require._name": format!("n{home}")})).await;
        wait_green(&pl, m, i).await;
    }
    let mut lines = String::new();
    for i in ["ia1", "ia2", "ib1", "ib2"] {
        for d in 0..100 {
            lines += &format!("{}\n{}\n", json!({"index": {"_index": i, "_id": format!("{i}-{d}")}}), json!({"tag": "x", "v": d}));
        }
    }
    bulk(&pl, m, lines).await;
    let body = json!({"script": {"source": "ctx._source.v += 1000"}});
    let (a, b) = tokio::join!(
        pl.call(m, Method::POST, "/ia1,ia2/_update_by_query?wait_for_completion=false&scroll_size=10&requests_per_second=20", Some(body.clone())),
        pl.call(m, Method::POST, "/ib1,ib2/_update_by_query?wait_for_completion=false&scroll_size=10&requests_per_second=40", Some(body.clone())),
    );
    assert!(a.0 == 200 && b.0 == 200, "{a:?} {b:?}");
    let (ta, tb) = (a.1["task"].as_str().unwrap().to_string(), b.1["task"].as_str().unwrap().to_string());
    // The children of both, by the index they work on.
    let children = |v: &Value| -> Vec<Value> {
        v["nodes"].as_object().into_iter().flatten().flat_map(|(_, n)| n["tasks"].as_object().into_iter().flatten().map(|(_, t)| t.clone()))
            .filter(|t| t["parent_task_id"].is_string()).collect()
    };
    let deadline = Instant::now() + Duration::from_secs(20);
    let seen = loop {
        let (st, v) = pl.call(m, Method::GET, "/_tasks?detailed=true&actions=*byquery", None).await;
        assert_eq!(st, 200, "{v}");
        let c = children(&v);
        let has = |p: &str| c.iter().any(|t| t["description"].as_str().unwrap_or("").contains(p));
        if has("[ia") && has("[ib") {
            break c;
        }
        assert!(Instant::now() < deadline, "children of both parents not seen: {v}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    for t in &seen {
        let d = t["description"].as_str().unwrap_or("");
        let want = if d.contains("[ia") { &ta } else { &tb };
        assert_eq!(t["parent_task_id"].as_str(), Some(want.as_str()), "child {t} has another parent");
    }
    let (st, v) = pl.call(m, Method::POST, &format!("/_update_by_query/{ta}/_rethrottle?requests_per_second=1000"), None).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = pl.call(m, Method::POST, &format!("/_tasks/{ta}/_cancel"), None).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = pl.call(m, Method::GET, "/_tasks?detailed=true&actions=*byquery", None).await;
    assert_eq!(st, 200, "{v}");
    for t in children(&v).iter().filter(|t| t["description"].as_str().unwrap_or("").contains("[ib")) {
        assert_eq!(t["parent_task_id"].as_str(), Some(tb.as_str()), "{t}");
        assert_eq!(t["status"]["requests_per_second"].as_f64(), Some(40.0), "B's child rethrottled with A: {t}");
        assert!(t["status"]["canceled"].is_null() && t["cancelled"] != true, "B's child cancelled with A: {t}");
    }
    let done = completed(&pl, m, &tb, 60).await;
    assert_eq!(done["response"]["updated"].as_u64(), Some(200), "{done}");
    assert!(done["response"]["canceled"].is_null(), "{done}");
    let done = completed(&pl, m, &ta, 60).await;
    assert!(done["response"]["updated"].as_u64().unwrap_or(0) < 200, "A ran to completion despite its cancellation: {done}");
}

/// Distributed reindex (a two-shard source) honours `requests_per_second`
/// between batches and a rethrottle while it runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn distributed_reindex_is_throttled_and_rethrottled() {
    let pl = start_procs(3, &[]).await;
    let m = pl.master(0).await;
    create(&pl, m, "thr", json!({"number_of_shards": 2, "number_of_replicas": 0})).await;
    wait_green(&pl, m, "thr").await;
    let lines: String = (0..60).map(|i| format!("{}\n{}\n", json!({"index": {"_index": "thr", "_id": format!("t{i}")}}), json!({"tag": "x", "v": i}))).collect();
    bulk(&pl, m, lines).await;

    // 60 documents in batches of 10 at 20/s: half a second per batch.
    let started = Instant::now();
    let v = ok(&pl, m, "/_reindex?requests_per_second=20", json!({"source": {"index": "thr", "size": 10}, "dest": {"index": "thr_sync"}})).await;
    let elapsed = started.elapsed();
    assert_eq!(v["created"].as_u64(), Some(60), "{v}");
    assert!(elapsed >= Duration::from_millis(2400), "not throttled: {elapsed:?} {v}");
    assert!(v["throttled_millis"].as_u64().unwrap_or(0) >= 1500, "{v}");

    // At 5/s the run would take 12 s; rethrottled to unlimited after one
    // second it finishes promptly.
    let (st, t) = pl.call(m, Method::POST, "/_reindex?wait_for_completion=false&requests_per_second=5", Some(json!({"source": {"index": "thr", "size": 10}, "dest": {"index": "thr_async"}}))).await;
    assert_eq!(st, 200, "{t}");
    let task = t["task"].as_str().unwrap().to_string();
    tokio::time::sleep(Duration::from_millis(1000)).await;
    let (_, v) = pl.call(m, Method::GET, &format!("/_tasks/{task}"), None).await;
    assert_eq!(v["completed"], false, "not throttled: {v}");
    let (st, v) = pl.call(m, Method::POST, &format!("/_reindex/{task}/_rethrottle?requests_per_second=-1"), None).await;
    assert_eq!(st, 200, "{v}");
    let rethrottled = Instant::now();
    let done = completed(&pl, m, &task, 30).await;
    assert!(rethrottled.elapsed() < Duration::from_secs(5), "rethrottle ignored: {:?} {done}", rethrottled.elapsed());
    assert_eq!(done["response"]["created"].as_u64(), Some(60), "{done}");
}

/// `conflicts: proceed` with `max_docs`: conflicts do not use up the quota
/// (distributed path: a two-shard destination).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn conflicts_do_not_use_up_max_docs_when_distributed() {
    let pl = start_procs(3, &[]).await;
    let m = pl.master(0).await;
    create(&pl, m, "qsrc", json!({"number_of_shards": 1, "number_of_replicas": 0})).await;
    create(&pl, m, "qdst", json!({"number_of_shards": 2, "number_of_replicas": 0})).await;
    wait_green(&pl, m, "qsrc").await;
    wait_green(&pl, m, "qdst").await;
    let mut lines = String::new();
    for i in 0..10 {
        lines += &format!("{}\n{}\n", json!({"index": {"_index": "qsrc", "_id": format!("q{i}")}}), json!({"tag": "x", "v": i}));
    }
    for i in 0..5 {
        lines += &format!("{}\n{}\n", json!({"index": {"_index": "qdst", "_id": format!("q{i}")}}), json!({"tag": "old", "v": i}));
    }
    bulk(&pl, m, lines).await;
    let (st, v) = pl.call(m, Method::POST, "/_reindex?refresh=true", Some(json!({"conflicts": "proceed", "max_docs": 5, "source": {"index": "qsrc"}, "dest": {"index": "qdst", "op_type": "create"}}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!((v["created"].as_u64(), v["version_conflicts"].as_u64()), (Some(5), Some(5)), "{v}");
    assert_eq!(count(&pl, m, "qdst", json!({"match_all": {}})).await, 10);
}

/// Every `_id` of `index`.
async fn ids(pl: &Procs, via: usize, index: &str) -> std::collections::BTreeSet<String> {
    let (st, v) = pl.call(via, Method::POST, &format!("/{index}/_search"), Some(json!({"size": 1000, "_source": false}))).await;
    assert_eq!(st, 200, "{v}");
    v["hits"]["hits"].as_array().into_iter().flatten().filter_map(|h| h["_id"].as_str().map(str::to_string)).collect()
}

/// A remote reindex whose source has a namesake here, run distributed (a
/// two-shard destination): it copies the remote documents only.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_distributed_remote_reindex_reads_the_remote_source() {
    let remote = start_procs(1, &[]).await;
    create(&remote, 0, "sent", json!({"number_of_shards": 1, "number_of_replicas": 0})).await;
    let lines: String = (0..5).map(|i| format!("{}\n{}\n", json!({"index": {"_index": "sent", "_id": format!("remote{i}")}}), json!({"tag": "remote", "v": i}))).collect();
    bulk(&remote, 0, lines).await;
    let host = format!("127.0.0.1:{}", remote.p.http[0]);
    let pl = start_procs(3, &[("reindex.remote.whitelist", host.as_str())]).await;
    let m = pl.master(0).await;
    create(&pl, m, "sent", json!({"number_of_shards": 2, "number_of_replicas": 0})).await;
    create(&pl, m, "sent_copy", json!({"number_of_shards": 2, "number_of_replicas": 0})).await;
    wait_green(&pl, m, "sent").await;
    wait_green(&pl, m, "sent_copy").await;
    let lines: String = (0..7).map(|i| format!("{}\n{}\n", json!({"index": {"_index": "sent", "_id": format!("local{i}")}}), json!({"tag": "local", "v": i}))).collect();
    bulk(&pl, m, lines).await;
    let v = ok(&pl, m, "/_reindex?refresh=true", json!({"source": {"remote": {"host": format!("http://{host}")}, "index": "sent"}, "dest": {"index": "sent_copy"}})).await;
    assert_eq!(v["created"].as_u64(), Some(5), "{v}");
    assert_eq!(ids(&pl, m, "sent_copy").await, (0..5).map(|i| format!("remote{i}")).collect(), "copied other documents than the remote source's");
}

/// A distributed (two-shard destination) remote reindex of hits without
/// `_source` fails like ES (`Reindexer`: "didn't store _source") with
/// nothing written: a source with `_source` disabled, and a `"_source":
/// false` request, synchronously and as a task. It never indexes them as
/// `{}` (silent data loss). A filter that matches no field returns
/// `"_source": {}`, which ES copies, and so does this.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_distributed_remote_reindex_never_writes_a_missing_source() {
    let remote = start_procs(1, &[]).await;
    let (st, v) = remote.call(0, Method::PUT, "/nosrc", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"_source": {"enabled": false}, "properties": {"tag": {"type": "keyword"}}}}))).await;
    assert_eq!(st, 200, "{v}");
    create(&remote, 0, "withsrc", json!({"number_of_shards": 1, "number_of_replicas": 0})).await;
    for index in ["nosrc", "withsrc"] {
        let lines: String = (0..5).map(|i| format!("{}\n{}\n", json!({"index": {"_index": index, "_id": format!("d{i}")}}), json!({"tag": "remote", "v": i}))).collect();
        bulk(&remote, 0, lines).await;
    }
    let host = format!("127.0.0.1:{}", remote.p.http[0]);
    let pl = start_procs(2, &[("reindex.remote.whitelist", host.as_str())]).await;
    let m = pl.master(0).await;
    let from = |index: &str, filter: Option<Value>| {
        let mut source = json!({"remote": {"host": format!("http://{host}")}, "index": index});
        if let Some(f) = filter {
            source["_source"] = f;
        }
        source
    };
    for (dest, source) in [("d_disabled", from("nosrc", None)), ("d_unfetched", from("withsrc", Some(json!(false))))] {
        create(&pl, m, dest, json!({"number_of_shards": 2, "number_of_replicas": 0})).await;
        wait_green(&pl, m, dest).await;
        let (st, v) = pl.call(m, Method::POST, "/_reindex?refresh=true", Some(json!({"source": source, "dest": {"index": dest}}))).await;
        assert_eq!(st, 400, "{dest}: {v}");
        assert_eq!(v["error"]["type"], "illegal_argument_exception", "{dest}: {v}");
        let reason = v["error"]["reason"].as_str().unwrap_or_default();
        assert!(reason.starts_with(&format!("[{}][d", source["index"].as_str().unwrap())) && reason.ends_with("] didn't store _source"), "{dest}: {v}");
        assert_eq!(count(&pl, m, dest, json!({"match_all": {}})).await, 0, "{dest}: documents were written");
        // As a task: the task fails with the same error.
        let (st, v) = pl.call(m, Method::POST, "/_reindex?wait_for_completion=false", Some(json!({"source": source, "dest": {"index": dest}}))).await;
        assert_eq!(st, 200, "{dest}: {v}");
        let done = completed(&pl, m, v["task"].as_str().unwrap(), 60).await;
        assert_eq!(done["error"]["type"], "illegal_argument_exception", "{dest}: {done}");
        assert!(done["error"]["reason"].as_str().unwrap_or_default().ends_with("didn't store _source"), "{dest}: {done}");
        let (_, _) = pl.call(m, Method::POST, &format!("/{dest}/_refresh"), None).await;
        assert_eq!(count(&pl, m, dest, json!({"match_all": {}})).await, 0, "{dest}: the task wrote documents");
    }
    create(&pl, m, "d_filtered", json!({"number_of_shards": 2, "number_of_replicas": 0})).await;
    wait_green(&pl, m, "d_filtered").await;
    let v = ok(&pl, m, "/_reindex?refresh=true", json!({"source": from("withsrc", Some(json!(["absent_field"]))), "dest": {"index": "d_filtered"}})).await;
    assert_eq!(v["created"].as_u64(), Some(5), "{v}");
    assert_eq!(ids(&pl, m, "d_filtered").await, (0..5).map(|i| format!("d{i}")).collect());
}

/// Reindex copies dense vectors from an index excluding them from
/// `_source`, on one- and multi-shard topologies; the copies answer kNN.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reindex_copies_vectors_on_every_topology() {
    let pl = start_procs(3, &[]).await;
    let m = pl.master(0).await;
    let mappings = json!({"properties": {"emb": {"type": "dense_vector", "dims": 3, "index": true, "similarity": "l2_norm"}, "v": {"type": "long"}}});
    for (index, shards) in [("vec1", 1), ("vec1_copy", 1), ("vec2", 2), ("vec2_copy", 2)] {
        let (st, v) = pl.call(m, Method::PUT, &format!("/{index}"), Some(json!({"settings": {"number_of_shards": shards, "number_of_replicas": 0, "index.mapping.exclude_source_vectors": true}, "mappings": mappings}))).await;
        assert_eq!(st, 200, "{v}");
        wait_green(&pl, m, index).await;
    }
    let mut lines = String::new();
    for index in ["vec1", "vec2"] {
        for i in 0..6 {
            lines += &format!("{}\n{}\n", json!({"index": {"_index": index, "_id": format!("e{i}")}}), json!({"emb": [i as f64, 1.0, 0.0], "v": i}));
        }
    }
    bulk(&pl, m, lines).await;
    for (src, dst) in [("vec1", "vec1_copy"), ("vec2", "vec2_copy")] {
        let v = ok(&pl, m, "/_reindex?refresh=true", json!({"source": {"index": src}, "dest": {"index": dst}})).await;
        assert_eq!(v["created"].as_u64(), Some(6), "{v}");
        let (st, v) = pl.call(m, Method::POST, &format!("/{dst}/_search"), Some(json!({"size": 10, "sort": [{"v": "asc"}], "_source": {"exclude_vectors": false}}))).await;
        assert_eq!(st, 200, "{v}");
        let got: Vec<Value> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_source"]["emb"].clone()).collect();
        let want: Vec<Value> = (0..6).map(|i| json!([i as f64, 1.0, 0.0])).collect();
        assert_eq!(got.len(), 6, "{v}");
        for (g, w) in got.iter().zip(&want) {
            let g: Vec<f64> = g.as_array().unwrap_or_else(|| panic!("{dst}: no vector copied: {v}")).iter().filter_map(Value::as_f64).collect();
            let w: Vec<f64> = w.as_array().unwrap().iter().filter_map(Value::as_f64).collect();
            assert_eq!(g, w, "{dst}");
        }
        let (st, v) = pl.call(m, Method::POST, &format!("/{dst}/_search"), Some(json!({"knn": {"field": "emb", "query_vector": [4.0, 1.0, 0.0], "k": 1, "num_candidates": 10}, "_source": false}))).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(v["hits"]["hits"][0]["_id"], "e4", "{dst}: {v}");
    }
}

/// Manual slices of a distributed reindex (a three-shard source) copy
/// disjoint parts whose union is the whole source.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn manual_slices_partition_a_distributed_reindex() {
    let pl = start_procs(3, &[]).await;
    let m = pl.master(0).await;
    create(&pl, m, "sl", json!({"number_of_shards": 3, "number_of_replicas": 0})).await;
    wait_green(&pl, m, "sl").await;
    let lines: String = (0..50).map(|i| format!("{}\n{}\n", json!({"index": {"_index": "sl", "_id": format!("s{i}")}}), json!({"tag": "x", "v": i}))).collect();
    bulk(&pl, m, lines).await;
    let mut parts = Vec::new();
    for id in 0..2 {
        let dst = format!("sl{id}");
        let v = ok(&pl, m, "/_reindex?refresh=true", json!({"source": {"index": "sl", "slice": {"id": id, "max": 2}}, "dest": {"index": dst}})).await;
        let got = ids(&pl, m, &dst).await;
        assert_eq!(v["created"].as_u64(), Some(got.len() as u64), "{v}");
        assert!(!got.is_empty() && got.len() < 50, "slice {id} copied {} documents", got.len());
        parts.push(got);
    }
    assert!(parts[0].is_disjoint(&parts[1]), "slices overlap");
    let all: std::collections::BTreeSet<String> = parts[0].union(&parts[1]).cloned().collect();
    assert_eq!(all, (0..50).map(|i| format!("s{i}")).collect());
}
