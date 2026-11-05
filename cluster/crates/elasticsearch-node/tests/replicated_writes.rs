//! Writes with replicas, read back from every copy, on three node processes
//! ([`common::procs`]). A copy is read on its own through the node holding
//! it: `_mget` / `GET` with `preference=_only_local` (realtime) and searches
//! with `preference=_shards:k|_only_local`.
//! - P0-3: `_seq_no` / `_primary_term` / `_version` are the same on every
//!   copy after failovers (ES replicas apply each op with its own primary
//!   term, `IndexShard.applyIndexOperationOnReplica`), and compare-and-set
//!   with values read from any copy succeeds on the primary.
//! - P0-5: single-doc writes into a data stream retried across rollovers
//!   never create a doc twice.
//! - P0-6: `refresh=true` / `wait_for` writes are visible on every started
//!   copy when acked, also while a copy recovers.
//! - P1-10: realtime GET via a replica, `retry_on_conflict`, external
//!   versions (also after a failover), TSDB, nested blocks + join, and
//!   conflicting dynamic types with replicas; search-idle replicas.
mod common;
use common::procs::{start_procs, Procs};
use reqwest::Method;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const TERMS: usize = 3;
const STREAMS: usize = 3;
const STREAM_RETRY: usize = 3;
const REFRESH: usize = 3;
const REALTIME: usize = 3;
const TSDB_JOIN: usize = 3;
const DYNAMIC: usize = 3;
const BASE_MS: i64 = 1_700_000_000_000;

async fn create(pl: &Procs, via: usize, index: &str, body: Value) {
    let (st, v) = pl.call(via, Method::PUT, &format!("/{index}"), Some(body)).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(via, index, "green", 90).await;
}

async fn bulk(pl: &Procs, via: usize, path: &str, lines: String) -> Value {
    let (st, v) = pl.ndjson(via, path, lines).await;
    assert_eq!(st, 200, "{path}: {v}");
    assert_eq!(v["errors"], false, "{path}: {v}");
    v
}

/// A 200 where every shard answered.
async fn search(pl: &Procs, node: usize, path: &str, body: Value) -> Value {
    let (st, v) = pl.call(node, Method::POST, path, Some(body.clone())).await;
    let s = &v["_shards"];
    assert!(st == 200 && s["failed"] == 0 && s["successful"] == s["total"], "n{node} {path} {body}: {st} {v}");
    v
}

fn hits(v: &Value) -> Vec<Value> {
    v["hits"]["hits"].as_array().cloned().unwrap_or_default()
}

/// The nodes holding a started copy of shard `k` of `index`, primary first.
async fn holders(pl: &Procs, via: usize, index: &str, k: u64) -> Vec<usize> {
    let mut rows: Vec<Value> = pl.shards(via, index).await.into_iter().filter(|r| r["shard"] == k.to_string() && r["state"] == "STARTED").collect();
    rows.sort_by_key(|r| r["prirep"] != "p");
    rows.iter().filter_map(|r| r["node"].as_str()?.trim_start_matches('n').parse().ok()).collect()
}

/// Wait until shard `k` of `index` has a started primary off `dead`.
async fn wait_promoted(pl: &Procs, via: usize, index: &str, k: u64, dead: usize) {
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        if holders(pl, via, index, k).await.first().is_some_and(|p| *p != dead) && pl.shards(via, index).await.iter().any(|r| r["shard"] == k.to_string() && r["prirep"] == "p" && r["state"] == "STARTED") {
            return;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    panic!("{index}[{k}]: no primary off n{dead}: {}\n{}", json!(pl.shards(via, index).await), json!(pl.call(via, Method::GET, "/_cluster/allocation/explain", Some(json!({"index": index, "shard": k, "primary": true}))).await.1));
}

/// `[found, _seq_no, _primary_term, _version]` of `ids` read realtime from
/// node `n`'s own copies.
async fn copy_versions(pl: &Procs, n: usize, index: &str, ids: &[String]) -> BTreeMap<String, Value> {
    let (st, v) = pl.call(n, Method::POST, &format!("/{index}/_mget?preference=_only_local"), Some(json!({"ids": ids}))).await;
    assert_eq!(st, 200, "n{n}: {v}");
    v["docs"].as_array().into_iter().flatten()
        .map(|d| (d["_id"].as_str().unwrap_or_default().to_string(), json!([d["found"], d["_seq_no"], d["_primary_term"], d["_version"]])))
        .collect()
}

/// The same from search hits (`seq_no_primary_term`, `version`) of shard
/// `k`'s copy on node `n`.
async fn searched_versions(pl: &Procs, n: usize, index: &str, k: u64) -> BTreeMap<String, Value> {
    let v = search(pl, n, &format!("/{index}/_search?preference=_shards:{k}%7C_only_local"),
        json!({"size": 10_000, "seq_no_primary_term": true, "version": true, "_source": false, "query": {"match_all": {}}})).await;
    hits(&v).iter().map(|h| (h["_id"].as_str().unwrap_or_default().to_string(), json!([true, h["_seq_no"], h["_primary_term"], h["_version"]]))).collect()
}

/// Every copy of every shard of `index` (`shards` of them) reports the
/// primary's `_seq_no` / `_primary_term` / `_version` for every doc, by
/// search and by realtime `_mget`. Returns the primary's view per shard.
async fn assert_version_parity(pl: &Procs, via: usize, index: &str, shards: u64, phase: &str) -> Vec<BTreeMap<String, Value>> {
    let mut errs = Vec::new();
    let mut views = Vec::new();
    for k in 0..shards {
        let copies = holders(pl, via, index, k).await;
        let want = searched_versions(pl, copies[0], index, k).await;
        let ids: Vec<String> = want.keys().cloned().collect();
        for &n in &copies {
            let searched = searched_versions(pl, n, index, k).await;
            let got = copy_versions(pl, n, index, &ids).await;
            for id in &ids {
                if searched.get(id) != want.get(id) {
                    errs.push(format!("{index}[{k}] {id}: searched on n{n} {:?}, primary n{} {}", searched.get(id), copies[0], want[id]));
                }
                if got.get(id) != want.get(id) {
                    errs.push(format!("{index}[{k}] {id}: get on n{n} {:?}, primary n{} {}", got.get(id), copies[0], want[id]));
                }
            }
            if searched.len() != want.len() {
                errs.push(format!("{index}[{k}]: n{n} has {} docs, the primary {}", searched.len(), want.len()));
            }
        }
        views.push(want);
    }
    errs.truncate(40);
    assert!(errs.is_empty(), "{phase}:\n{}", errs.join("\n"));
    views
}

/// P0-3: two rounds of (write, stop the node with shard 0's primary, write
/// and overwrite, restart): every copy agrees with the primary on
/// `_seq_no` / `_primary_term` / `_version`, docs written after a
/// promotion carry the new term, and compare-and-set with values read from
/// any copy succeeds.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn primary_terms_match_on_every_copy_after_failovers() {
    let mut pl = start_procs(TERMS, &[]).await;
    let first = pl.master(0).await;
    create(&pl, first, "terms", json!({"settings": {"number_of_shards": 2, "number_of_replicas": 2}, "mappings": {"properties": {"v": {"type": "keyword"}}}})).await;
    let docs = |prefix: &str, n: usize| -> String { (0..n).map(|i| format!("{}\n{}\n", json!({"index": {"_id": format!("{prefix}-{i}")}}), json!({"v": prefix}))).collect() };
    let mut term = 1;
    for round in 0..2 {
        let via = pl.live()[0];
        bulk(&pl, via, "/terms/_bulk", docs(&format!("before{round}"), 200)).await;
        let victim = holders(&pl, via, "terms", 0).await[0];
        pl.stop(victim, 30).await;
        let via = pl.live()[0];
        pl.wait_nodes(via, 2, 60).await;
        wait_promoted(&pl, via, "terms", 0, victim).await;
        term += 1;
        bulk(&pl, via, "/terms/_bulk", docs(&format!("after{round}"), 200)).await;
        // Overwrites of docs from before the promotion.
        bulk(&pl, via, "/terms/_bulk", docs(&format!("before{round}"), 50)).await;
        pl.spawn(victim, &[]).await;
        // The restarted node itself answers (and has joined) before it is asked.
        pl.wait_nodes(victim, 3, 90).await;
        pl.wait_nodes(via, 3, 90).await;
        pl.wait_health(via, "terms", "green", 120).await;
        let (st, v) = pl.call(via, Method::POST, "/terms/_refresh", None).await;
        assert_eq!(st, 200, "{v}");
        let views = assert_version_parity(&pl, via, "terms", 2, &format!("round {round}")).await;
        for (id, ver) in &views[0] {
            if id.starts_with(&format!("after{round}-")) {
                assert!(ver[2].as_u64().unwrap_or(0) >= term, "{id} written after promotion {round}: {ver}");
            }
        }
        // Compare-and-set with what each copy says.
        for i in 0..5 {
            let id = format!("after{round}-{i}");
            for n in pl.live() {
                let got = copy_versions(&pl, n, "terms", std::slice::from_ref(&id)).await;
                let (seq, primary_term) = (&got[&id][1], &got[&id][2]);
                let (st, v) = pl.call(n, Method::PUT, &format!("/terms/_doc/{id}?if_seq_no={seq}&if_primary_term={primary_term}"), Some(json!({"v": format!("cas-n{n}")}))).await;
                assert_eq!(st, 200, "CAS with n{n}'s copy of {id} ({seq}, {primary_term}): {v}");
            }
        }
    }
}

/// Every doc of `index` (any number, past 10k) by scroll: the total and the
/// hits, `_source` `seq` only.
async fn scroll_all(pl: &Procs, via: usize, index: &str) -> (u64, Vec<Value>) {
    let v = search(pl, via, &format!("/{index}/_search?scroll=1m"), json!({"size": 2000, "sort": ["_doc"], "track_total_hits": true, "_source": ["seq"]})).await;
    let total = v["hits"]["total"]["value"].as_u64().unwrap_or(0);
    let mut all = hits(&v);
    let mut id = v["_scroll_id"].clone();
    loop {
        let v = search(pl, via, "/_search/scroll", json!({"scroll": "1m", "scroll_id": id})).await;
        let page = hits(&v);
        if page.is_empty() {
            break;
        }
        all.extend(page);
        id = v["_scroll_id"].clone();
    }
    let (st, v) = pl.call(via, Method::DELETE, "/_search/scroll", Some(json!({"scroll_id": [id]}))).await;
    assert_eq!(st, 200, "{v}");
    (total, all)
}

/// P0-5: writers on every node send single docs (generated ids) into a data
/// stream while it rolls over five times and the node holding the write
/// index's primary (master or not) is killed and restarted. No doc exists
/// twice; every acked one exists (read back by scroll, past 10k).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn data_stream_retries_never_duplicate_across_rollovers() {
    let mut pl = start_procs(STREAMS, &[]).await;
    let master = pl.master(0).await;
    let (st, v) = pl.call(master, Method::PUT, "/_index_template/logs-t", Some(json!({"index_patterns": ["logs-t*"], "data_stream": {},
        "template": {"settings": {"number_of_shards": 1, "number_of_replicas": 1}, "mappings": {"properties": {"@timestamp": {"type": "date"}, "seq": {"type": "keyword"}}}}}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = pl.call(master, Method::PUT, "/_data_stream/logs-t", None).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(master, "logs-t", "green", 60).await;
    let stop = Arc::new(AtomicBool::new(false));
    let writers: Vec<_> = (0..pl.len()).map(|n| {
        let (c, port, stop) = (pl.c.clone(), pl.p.http[n], stop.clone());
        tokio::spawn(async move {
            let mut acked = Vec::new();
            let mut i = 0;
            while !stop.load(Ordering::Relaxed) {
                let seq = format!("n{n}-{i}");
                i += 1;
                let r = c.post(format!("http://127.0.0.1:{port}/logs-t/_doc")).timeout(Duration::from_secs(60))
                    .json(&json!({"@timestamp": BASE_MS + i, "seq": seq})).send().await;
                match r {
                    Ok(r) if r.status().as_u16() == 201 => acked.push(seq),
                    Ok(_) => {}
                    Err(_) => tokio::time::sleep(Duration::from_millis(100)).await,
                }
            }
            acked
        })
    }).collect();
    let mut via = master;
    let mut killed = false;
    for round in 0..5 {
        tokio::time::sleep(Duration::from_millis(700)).await;
        let (st, v) = pl.call(via, Method::POST, "/logs-t/_rollover", None).await;
        assert_eq!(st, 200, "rollover {round}: {v}");
        // ES's rollover waits for the new write index's active shards
        // (`wait_for_active_shards`, default 1: the primary) before answering.
        assert_eq!(v["shards_acknowledged"], true, "rollover {round}: {v}");
        let write = v["new_index"].as_str().unwrap_or_else(|| panic!("rollover {round}: no new_index: {v}")).to_string();
        let copies = holders(&pl, via, &write, 0).await;
        assert!(!copies.is_empty(), "rollover {round} answered before {write}'s primary started: {}", json!(pl.shards(via, &write).await));
        if round == 2 {
            // The write index's primary goes, master or not.
            // Only an in-sync copy can be promoted (ES `in_sync_allocations`):
            // the new write index's replica is still recovering right after
            // the rollover, and with it unstarted the shard goes red, not over.
            pl.wait_health(via, &write, "green", 60).await;
            let copies = holders(&pl, via, &write, 0).await;
            let primary = copies[0];
            eprintln!("killing n{primary}, holder of {write}'s primary; shards: {}", json!(pl.shards(via, &write).await));
            pl.kill(primary);
            killed = true;
            via = pl.live()[0];
            pl.wait_nodes(via, pl.len() - 1, 60).await;
            wait_promoted(&pl, via, &write, 0, primary).await;
            pl.spawn(primary, &[]).await;
        }
    }
    assert!(killed);
    tokio::time::sleep(Duration::from_millis(700)).await;
    stop.store(true, Ordering::Relaxed);
    let mut acked = Vec::new();
    for w in writers {
        acked.extend(w.await.unwrap());
    }
    pl.wait_nodes(via, pl.len(), 90).await;
    pl.wait_health(via, "logs-t", "green", 120).await;
    let (st, v) = pl.call(via, Method::POST, "/logs-t/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
    let (total, all) = scroll_all(&pl, via, "logs-t").await;
    assert_eq!(total, all.len() as u64, "scrolled every doc");
    let (mut ids, mut seqs) = (BTreeSet::new(), BTreeSet::new());
    let mut dups = Vec::new();
    for h in &all {
        let (id, seq) = (h["_id"].as_str().unwrap_or_default().to_string(), h["_source"]["seq"].as_str().unwrap_or_default().to_string());
        if !ids.insert(id.clone()) || !seqs.insert(seq.clone()) {
            dups.push(format!("{id} ({seq}) in {}", h["_index"]));
        }
    }
    assert!(dups.is_empty(), "docs written twice: {dups:?}");
    let missing: Vec<&String> = acked.iter().filter(|s| !seqs.contains(*s)).collect();
    assert!(missing.is_empty(), "{} of {} acked docs missing: {:?}", missing.len(), acked.len(), &missing[..missing.len().min(10)]);
    assert!(acked.len() > 50, "only {} writes acked", acked.len());
}

/// P0-5, forced: a single-doc write into a data stream whose answer is
/// lost. The replica's node is paused, so the primary applies the write but
/// cannot finish replicating it; the stream rolls over; the primary's node
/// dies; the replica comes back. ES's `TransportReplicationAction` retries
/// the shard request it resolved (backing index 1, now on the promoted
/// replica): one doc, in the first backing index, answered 201 or (the
/// replica had applied it) 409. A retry that resolved the write index again
/// would write it a second time into backing index 2.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_indeterminate_data_stream_write_is_retried_where_it_was_resolved() {
    // The replica's node is paused for a few seconds: the rollover's
    // publication must not wait 30s for it, and fault detection must not drop
    // it meanwhile (that would fail its copy, leaving nothing to promote when
    // the primary dies; ES alike). A killed node still drops at once (its
    // connections close).
    let mut pl = start_procs(STREAM_RETRY, &[("cluster.publish.timeout", "2s"), ("cluster.fault_detection.follower_check.timeout", "30s"),
        ("cluster.follower_lag.timeout", "120s")]).await;
    let a = pl.master(0).await;
    let (b, c) = ((a + 1) % 3, (a + 2) % 3);
    let template = |settings: Value| json!({"index_patterns": ["logs-u*"], "data_stream": {}, "template": {"settings": settings,
        "mappings": {"properties": {"@timestamp": {"type": "date"}, "seq": {"type": "keyword"}}}}});
    // Backing index 1: primary on b, replica on c.
    let (st, v) = pl.call(a, Method::PUT, "/_index_template/logs-u", Some(template(json!({"number_of_shards": 1, "number_of_replicas": 0, "index.routing.allocation.require._name": format!("n{b}")})))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = pl.call(a, Method::PUT, "/_data_stream/logs-u", None).await;
    assert_eq!(st, 200, "{v}");
    let (_, ds) = pl.call(a, Method::GET, "/_data_stream/logs-u", None).await;
    let first = ds["data_streams"][0]["indices"][0]["index_name"].as_str().unwrap_or_else(|| panic!("{ds}")).to_string();
    pl.wait_health(a, &first, "green", 60).await;
    let (st, v) = pl.call(a, Method::PUT, &format!("/{first}/_settings"), Some(json!({"index.routing.allocation.require._name": null,
        "index.routing.allocation.exclude._name": format!("n{a}"), "index.number_of_replicas": 1}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(a, &first, "green", 60).await;
    assert_eq!(holders(&pl, a, &first, 0).await, vec![b, c]);
    // Later backing indices stay off b.
    let (st, v) = pl.call(a, Method::PUT, "/_index_template/logs-u", Some(template(json!({"number_of_shards": 1, "number_of_replicas": 1, "index.routing.allocation.exclude._name": format!("n{b}")})))).await;
    assert_eq!(st, 200, "{v}");
    let lines: String = (0..20).map(|i| format!("{}\n{}\n", json!({"create": {}}), json!({"@timestamp": BASE_MS + i, "seq": format!("s{i}")}))).collect();
    bulk(&pl, a, "/logs-u/_bulk?refresh=true", lines).await;

    pl.pause(c);
    let inflight = {
        let (cl, url) = (pl.c.clone(), format!("http://127.0.0.1:{}/logs-u/_doc", pl.p.http[a]));
        tokio::spawn(async move {
            let r = cl.post(url).timeout(Duration::from_secs(120)).json(&json!({"@timestamp": BASE_MS + 100, "seq": "inflight"})).send().await;
            match r {
                Ok(r) => (r.status().as_u16(), r.json::<Value>().await.unwrap_or(Value::Null)),
                Err(e) => (0, json!(e.to_string())),
            }
        })
    };
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(!inflight.is_finished(), "the write waits for the paused replica");
    let (st, v) = pl.call(a, Method::POST, "/logs-u/_rollover?timeout=2s", None).await;
    assert_eq!(st, 200, "rollover with the replica's node paused: {v}");
    let second = v["new_index"].as_str().unwrap_or_default().to_string();
    assert_ne!(second, first, "{v}");
    // Still the set-up the test needs: three nodes, and c's copy of the
    // first backing index started (to be promoted).
    let (_, h) = pl.call(a, Method::GET, "/_cluster/health", None).await;
    assert_eq!(h["number_of_nodes"], 3, "the paused node was dropped: {h}");
    assert_eq!(holders(&pl, a, &first, 0).await, vec![b, c], "{}", json!(pl.shards(a, &first).await));
    pl.kill(b);
    tokio::time::sleep(Duration::from_millis(500)).await;
    pl.resume(c);
    let (st, ack) = inflight.await.unwrap();
    pl.wait_nodes(a, 2, 60).await;
    wait_promoted(&pl, a, &first, 0, b).await;
    let (st2, v) = pl.call(a, Method::POST, "/logs-u/_refresh", None).await;
    assert_eq!(st2, 200, "{v}");
    let (total, all) = scroll_all(&pl, a, "logs-u").await;
    let ids: BTreeSet<&str> = all.iter().filter_map(|h| h["_id"].as_str()).collect();
    assert_eq!(ids.len(), all.len(), "an _id twice across backing indices: {}", json!(all));
    let copies: Vec<&Value> = all.iter().filter(|h| h["_source"]["seq"] == "inflight").collect();
    assert_eq!(copies.len(), 1, "the in-flight write exists once: {}", json!(copies));
    // The retry is the same shard request (`ReroutePhase.retry` after
    // `onRetry(possiblyExecuted)`): if the promoted copy already applied the
    // op, the retried auto-id create plans against it (`isRetry`, so not
    // append-only) and conflicts, 409 "document already exists"; if not, it
    // is created, 201. Either way in the index it resolved.
    match st {
        201 => assert_eq!(ack["_index"], first.as_str(), "retried into the index it resolved: {ack}"),
        409 => {
            assert_eq!(ack["error"]["type"], "version_conflict_engine_exception", "{ack}");
            assert_eq!(ack["error"]["index"], first.as_str(), "the conflict is in the index it resolved: {ack}");
        }
        _ => panic!("ES answers 201 or 409: {st} {ack}"),
    }
    assert_eq!(copies[0]["_index"], first.as_str(), "{}", json!(copies));
    assert_eq!(total, 21);
}

/// Every node whose copies of all `shards` shards of `index` are started:
/// the hits of `query` on its own copies.
async fn per_copy_totals(pl: &Procs, via: usize, index: &str, shards: u64, query: &Value) -> BTreeMap<usize, u64> {
    let rows = pl.shards(via, index).await;
    let mut out = BTreeMap::new();
    for n in pl.live() {
        let mine: Vec<&Value> = rows.iter().filter(|r| r["node"] == format!("n{n}")).collect();
        if mine.len() as u64 != shards || mine.iter().any(|r| r["state"] != "STARTED") {
            continue;
        }
        let list = (0..shards).map(|k| k.to_string()).collect::<Vec<_>>().join(",");
        let v = search(pl, n, &format!("/{index}/_search?preference=_shards:{list}%7C_only_local"), json!({"size": 0, "track_total_hits": true, "query": query})).await;
        out.insert(n, v["hits"]["total"]["value"].as_u64().unwrap_or(0));
    }
    out
}

/// P0-6: each refreshing write (`_doc`, `_update`, `_delete`, `_bulk` with
/// `true` and `wait_for`, `_delete_by_query` with `true`) is visible on
/// every copy the moment it is acked; again while a replica recovers.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refreshing_writes_are_visible_on_every_copy_when_acked() {
    let pl = start_procs(REFRESH, &[]).await;
    let master = pl.master(0).await;
    // `true` refreshes itself; `wait_for` waits for a scheduled refresh (ES
    // never refreshes for it: with refresh off it would wait until
    // `index.max_refresh_listeners` forces one), so its index refreshes.
    create(&pl, master, "rf", json!({"settings": {"number_of_shards": 3, "number_of_replicas": 2, "refresh_interval": "-1"}, "mappings": {"properties": {"v": {"type": "long"}}}})).await;
    create(&pl, master, "rfw", json!({"settings": {"number_of_shards": 3, "number_of_replicas": 2, "refresh_interval": "1s"}, "mappings": {"properties": {"v": {"type": "long"}}}})).await;
    let mut errs = Vec::new();
    let mut check = |what: String, totals: BTreeMap<usize, u64>, want: u64| {
        for (n, t) in totals {
            if t != want {
                errs.push(format!("{what}: n{n}'s copies show {t}, want {want}"));
            }
        }
    };
    let mut step = 0usize;
    for (index, refresh) in [("rf", "true"), ("rfw", "wait_for")] {
        for _ in 0..3 {
            step += 1;
            let via = step % pl.len();
            let id = format!("a{step}");
            let ids = json!({"ids": {"values": [id]}});
            let (st, v) = pl.call(via, Method::PUT, &format!("/{index}/_doc/{id}?refresh={refresh}"), Some(json!({"v": 1}))).await;
            assert!(st == 201, "{v}");
            check(format!("index {id} refresh={refresh}"), per_copy_totals(&pl, master, index, 3, &json!({"bool": {"filter": [ids, {"term": {"v": 1}}]}})).await, 1);
            let (st, v) = pl.call(via, Method::POST, &format!("/{index}/_update/{id}?refresh={refresh}"), Some(json!({"doc": {"v": 2}}))).await;
            assert_eq!(st, 200, "{v}");
            check(format!("update {id} refresh={refresh}"), per_copy_totals(&pl, master, index, 3, &json!({"bool": {"filter": [ids, {"term": {"v": 2}}]}})).await, 1);
            let (st, v) = pl.call(via, Method::DELETE, &format!("/{index}/_doc/{id}?refresh={refresh}"), None).await;
            assert_eq!(st, 200, "{v}");
            check(format!("delete {id} refresh={refresh}"), per_copy_totals(&pl, master, index, 3, &ids).await, 0);
            let lines: String = (0..20).map(|i| format!("{}\n{}\n", json!({"index": {"_id": format!("b{step}-{i}")}}), json!({"v": step}))).collect();
            bulk(&pl, via, &format!("/{index}/_bulk?refresh={refresh}"), lines).await;
            check(format!("bulk {step} refresh={refresh}"), per_copy_totals(&pl, master, index, 3, &json!({"term": {"v": step}})).await, 20);
            let (st, v) = pl.call(via, Method::POST, &format!("/{index}/_delete_by_query?refresh=true"), Some(json!({"query": {"term": {"v": step}}}))).await;
            assert_eq!(st, 200, "{v}");
            check(format!("delete_by_query {step}"), per_copy_totals(&pl, master, index, 3, &json!({"term": {"v": step}})).await, 0);
        }
    }
    // A replica recovering in place (`_cluster/reroute` `cancel`) while
    // refreshing writes go on: whichever copies are started when a write is
    // acked show it.
    let rows = pl.shards(master, "rf").await;
    let replica = rows.iter().find(|r| r["shard"] == "0" && r["prirep"] == "r").unwrap().clone();
    let (st, v) = pl.call(master, Method::POST, "/_cluster/reroute", Some(json!({"commands": [{"cancel": {"index": "rf", "shard": 0, "node": replica["node"], "allow_primary": false}}]}))).await;
    assert_eq!(st, 200, "{v}");
    for i in 0..30 {
        let (st, v) = pl.call(master, Method::PUT, &format!("/rf/_doc/r{i}?refresh=true"), Some(json!({"v": 1000 + i}))).await;
        assert_eq!(st, 201, "{v}");
        check(format!("r{i} during recovery"), per_copy_totals(&pl, master, "rf", 3, &json!({"term": {"v": 1000 + i}})).await, 1);
    }
    pl.wait_health(master, "rf", "green", 60).await;
    check("after recovery".into(), per_copy_totals(&pl, master, "rf", 3, &json!({"range": {"v": {"gte": 1000}}})).await, 30);
    drop(check);
    assert!(errs.is_empty(), "{} stale copies:\n{}", errs.len(), errs.join("\n"));
}

/// Realtime GET of `id` through node `n`'s own copy.
async fn local_get(pl: &Procs, n: usize, index: &str, id: &str) -> (u16, Value) {
    pl.call(n, Method::GET, &format!("/{index}/_doc/{id}?preference=_only_local"), None).await
}

/// P1-10: realtime GET via every copy right after each ack, concurrent
/// `retry_on_conflict` updates, and external versions before and after a
/// failover.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn realtime_reads_conflicts_and_external_versions_per_copy() {
    let mut pl = start_procs(REALTIME, &[]).await;
    let master = pl.master(0).await;
    create(&pl, master, "rt", json!({"settings": {"number_of_shards": 3, "number_of_replicas": 2, "refresh_interval": "-1"}})).await;
    // Realtime GET on every copy, with the ack's version and seq_no; a
    // delete is found:false everywhere.
    for i in 0..20 {
        let id = format!("g{i}");
        let (st, ack) = pl.call(i % 3, Method::PUT, &format!("/rt/_doc/{id}"), Some(json!({"v": i}))).await;
        assert_eq!(st, 201, "{ack}");
        for n in pl.live() {
            let (st, v) = local_get(&pl, n, "rt", &id).await;
            assert_eq!(st, 200, "n{n} {id}: {v}");
            assert_eq!((&v["_version"], &v["_seq_no"], &v["_primary_term"], &v["_source"]["v"]), (&ack["_version"], &ack["_seq_no"], &ack["_primary_term"], &json!(i)), "n{n} {id}: {v}");
        }
        let (st, ack) = pl.call((i + 1) % 3, Method::DELETE, &format!("/rt/_doc/{id}"), None).await;
        assert_eq!(st, 200, "{ack}");
        for n in pl.live() {
            let (st, v) = local_get(&pl, n, "rt", &id).await;
            assert_eq!((st, &v["found"]), (404, &json!(false)), "n{n} {id} after delete: {v}");
        }
    }
    // 3 coordinators x 50 concurrent increments with retry_on_conflict.
    let (st, v) = pl.call(master, Method::PUT, "/rt/_doc/counter", Some(json!({"c": 0}))).await;
    assert_eq!(st, 201, "{v}");
    let updates = (0..150).map(|i| {
        let (c, port) = (pl.c.clone(), pl.p.http[i % 3]);
        async move {
            let r = c.post(format!("http://127.0.0.1:{port}/rt/_update/counter?retry_on_conflict=10")).timeout(Duration::from_secs(60))
                .json(&json!({"script": {"source": "ctx._source.c += 1"}})).send().await;
            r.ok().is_some_and(|r| r.status().as_u16() == 200)
        }
    });
    let ok = futures::future::join_all(updates).await.into_iter().filter(|ok| *ok).count();
    assert!(ok > 0);
    for n in pl.live() {
        let (st, v) = local_get(&pl, n, "rt", "counter").await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(v["_source"]["c"], ok, "n{n}: {ok} updates succeeded: {v}");
        assert_eq!(v["_version"], ok + 1, "n{n}: {v}");
    }
    // External versions: v5, delete at v6, then v4 conflicts.
    let ext = |v: u64| format!("/rt/_doc/e?version={v}&version_type=external");
    let (st, v) = pl.call(master, Method::PUT, &ext(5), Some(json!({"x": 5}))).await;
    assert_eq!((st, &v["_version"]), (201, &json!(5)), "{v}");
    let (st, v) = pl.call(master, Method::DELETE, &ext(6), None).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = pl.call(master, Method::PUT, &ext(4), Some(json!({"x": 4}))).await;
    assert_eq!((st, &v["error"]["type"]), (409, &json!("version_conflict_engine_exception")), "{v}");
    // The same after the primary's node goes: the promoted copy keeps the
    // delete's version (within index.gc_deletes, 60s).
    // `e`'s shard: where a doc routed by `e` (its id's routing) lands.
    let (st, v) = pl.call(master, Method::PUT, "/rt/_doc/probe?routing=e&refresh=true", Some(json!({}))).await;
    assert_eq!(st, 201, "{v}");
    let mut k = None;
    for shard in 0..3u64 {
        let v = search(&pl, master, &format!("/rt/_search?preference=_shards:{shard}"), json!({"size": 0, "track_total_hits": true, "query": {"ids": {"values": ["probe"]}}})).await;
        if v["hits"]["total"]["value"] == 1 {
            k = Some(shard);
        }
    }
    let k = k.expect("the probe's shard");
    let victim = holders(&pl, master, "rt", k).await[0];
    let via = if victim == master { (master + 1) % 3 } else { master };
    pl.kill(victim);
    pl.wait_nodes(via, 2, 60).await;
    wait_promoted(&pl, via, "rt", k, victim).await;
    let (st, v) = pl.call(via, Method::PUT, &ext(4), Some(json!({"x": 4}))).await;
    assert_eq!((st, &v["error"]["type"]), (409, &json!("version_conflict_engine_exception")), "after failover: {v}");
    let (st, v) = pl.call(via, Method::PUT, &ext(7), Some(json!({"x": 7}))).await;
    assert_eq!((st, &v["_version"]), (201, &json!(7)), "{v}");
    pl.spawn(victim, &[]).await;
    // The restarted node itself answers (and has joined) before it is asked.
    pl.wait_nodes(victim, 3, 90).await;
    pl.wait_nodes(via, 3, 90).await;
    pl.wait_health(via, "rt", "green", 120).await;
    for n in pl.live() {
        let (st, v) = local_get(&pl, n, "rt", "e").await;
        assert_eq!((st, &v["_version"], &v["_source"]["x"]), (200, &json!(7), &json!(7)), "n{n}: {v}");
    }
}

/// Shard `k`'s copies, each on its own: the normalized answer to `body`.
async fn copies_answer(pl: &Procs, via: usize, index: &str, k: u64, body: &Value) -> Vec<(usize, Value)> {
    let mut out = Vec::new();
    for n in holders(pl, via, index, k).await {
        let mut v = search(pl, n, &format!("/{index}/_search?preference=_shards:{k}%7C_only_local"), body.clone()).await;
        for key in ["took", "_shards", "timed_out"] {
            v.as_object_mut().map(|m| m.remove(key));
        }
        out.push((n, v));
    }
    out
}

fn assert_copies_agree(what: &str, answers: &[(usize, Value)]) {
    for (n, v) in answers.iter().skip(1) {
        assert_eq!(v, &answers[0].1, "{what}: n{n} vs n{}", answers[0].0);
    }
}

/// P1-10: a TSDB index (3x1, routing_path) and nested blocks with a join
/// field (3x1) under concurrent whole-block updates, compared copy by copy.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tsdb_and_nested_join_agree_per_copy() {
    let pl = start_procs(TSDB_JOIN, &[]).await;
    let master = pl.master(0).await;
    create(&pl, master, "ts", json!({"settings": {"number_of_shards": 3, "number_of_replicas": 1, "index.mode": "time_series", "index.routing_path": ["host"],
        "index.time_series.start_time": "2023-11-14T00:00:00Z", "index.time_series.end_time": "2023-11-16T00:00:00Z"},
        "mappings": {"properties": {"@timestamp": {"type": "date"}, "host": {"type": "keyword", "time_series_dimension": true}, "cpu": {"type": "double", "time_series_metric": "gauge"}}}})).await;
    let point = |i: i64| json!({"@timestamp": BASE_MS + i * 1000, "host": format!("h{}", i % 10), "cpu": (i % 97) as f64 / 10.0});
    let lines: String = (0..300).map(|i| format!("{}\n{}\n", json!({"create": {}}), point(i))).collect();
    bulk(&pl, master, "/ts/_bulk?refresh=true", lines).await;
    // The same dimensions and timestamp again: a conflict.
    let (st, v) = pl.ndjson(master, "/ts/_bulk", format!("{}\n{}\n", json!({"create": {}}), point(7))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["items"][0]["create"]["status"], 409, "{v}");
    let body = json!({"size": 100, "sort": [{"@timestamp": "asc"}, {"host": "asc"}], "_source": true, "track_total_hits": true,
        "aggs": {"h": {"terms": {"field": "host", "size": 20}, "aggs": {"m": {"max": {"field": "cpu"}}}}, "series": {"time_series": {}}}});
    let mut total = 0;
    for k in 0..3 {
        let answers = copies_answer(&pl, master, "ts", k, &body).await;
        assert_eq!(answers.len(), 2, "two copies of ts[{k}]");
        assert_copies_agree(&format!("ts[{k}]"), &answers);
        total += answers[0].1["hits"]["total"]["value"].as_u64().unwrap();
        // GET by the generated id on each copy.
        for h in hits(&answers[0].1).iter().take(5) {
            let id = h["_id"].as_str().unwrap();
            for (n, _) in &answers {
                let (st, v) = local_get(&pl, *n, "ts", id).await;
                assert_eq!((st, &v["_source"]), (200, &h["_source"]), "n{n} ts {id}: {v}");
            }
        }
    }
    assert_eq!(total, 300);

    // Parents p0..p29 with three nested items each, a child per parent.
    create(&pl, master, "nj", json!({"settings": {"number_of_shards": 3, "number_of_replicas": 1, "refresh_interval": "50ms"},
        "mappings": {"properties": {"items": {"type": "nested", "properties": {"q": {"type": "long"}}}, "rel": {"type": "join", "relations": {"parent": "child"}}, "round": {"type": "long"}}}})).await;
    let parent = |_i: usize, round: u64| json!({"rel": "parent", "round": round, "items": [{"q": round}, {"q": round + 1}, {"q": round + 2}]});
    let mut lines = String::new();
    for i in 0..30 {
        lines.push_str(&format!("{}\n{}\n", json!({"index": {"_id": format!("p{i}"), "routing": format!("p{i}")}}), parent(i, 0)));
        lines.push_str(&format!("{}\n{}\n", json!({"index": {"_id": format!("c{i}"), "routing": format!("p{i}")}}), json!({"rel": {"name": "child", "parent": format!("p{i}")}})));
    }
    bulk(&pl, master, "/nj/_bulk?refresh=true", lines).await;
    // A child without routing is refused.
    let (st, v) = pl.call(master, Method::PUT, "/nj/_doc/orphan", Some(json!({"rel": {"name": "child", "parent": "p1"}}))).await;
    // ES: DocumentParsingException caused by "[routing] is missing for join field".
    assert_eq!((st, &v["error"]["type"]), (400, &json!("document_parsing_exception")), "{v}");
    assert_eq!(v["error"]["caused_by"]["type"], json!("illegal_argument_exception"), "{v}");
    assert_eq!(v["error"]["caused_by"]["reason"], json!("[routing] is missing for join field [rel]"), "{v}");
    // Whole-block rewrites on every node while copies are read.
    let stop = Arc::new(AtomicBool::new(false));
    let writers: Vec<_> = (0..pl.len()).map(|n| {
        let (c, port, stop) = (pl.c.clone(), pl.p.http[n], stop.clone());
        tokio::spawn(async move {
            let mut round = 1;
            while !stop.load(Ordering::Relaxed) {
                let lines: String = (0..30).filter(|i| i % 3 == n).map(|i| format!("{}\n{}\n", json!({"index": {"_id": format!("p{i}"), "routing": format!("p{i}")}}), parent(i, round))).collect();
                let _ = c.post(format!("http://127.0.0.1:{port}/nj/_bulk")).header("content-type", "application/x-ndjson").body(lines).timeout(Duration::from_secs(60)).send().await;
                round += 1;
            }
        })
    }).collect();
    let nested = json!({"size": 0, "aggs": {"items": {"nested": {"path": "items"}}}, "track_total_hits": true, "query": {"term": {"rel": "parent"}}});
    let mut errs = Vec::new();
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(8) {
        for k in 0..3 {
            for n in holders(&pl, master, "nj", k).await {
                let v = search(&pl, n, &format!("/nj/_search?preference=_shards:{k}%7C_only_local"), nested.clone()).await;
                let (parents, items) = (v["hits"]["total"]["value"].as_u64().unwrap_or(0), v["aggregations"]["items"]["doc_count"].as_u64().unwrap_or(0));
                if items != parents * 3 {
                    errs.push(format!("nj[{k}] on n{n}: {items} nested docs under {parents} parents"));
                }
            }
        }
    }
    stop.store(true, Ordering::Relaxed);
    for w in writers {
        w.await.unwrap();
    }
    assert!(errs.is_empty(), "torn nested blocks:\n{}", errs.join("\n"));
    let (st, v) = pl.call(master, Method::POST, "/nj/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
    let joined = json!({"size": 50, "sort": [{"_id": "asc"}], "_source": false, "track_total_hits": true,
        "query": {"has_child": {"type": "child", "query": {"match_all": {}}, "inner_hits": {"_source": false}}}});
    let mut parents = 0;
    for k in 0..3 {
        let answers = copies_answer(&pl, master, "nj", k, &joined).await;
        assert_copies_agree(&format!("nj[{k}] has_child"), &answers);
        parents += answers[0].1["hits"]["total"]["value"].as_u64().unwrap();
        for h in hits(&answers[0].1) {
            let p = h["_id"].as_str().unwrap();
            assert_eq!(h["inner_hits"]["child"]["hits"]["hits"][0]["_id"], format!("c{}", &p[1..]), "{h}");
        }
    }
    assert_eq!(parents, 30);
}

/// P1-10: conflicting dynamic types with replicas (4x2, three coordinators):
/// one committed type everywhere, no copy failed or re-recovered, the
/// copies of each shard agree (copy_to and runtime fields included). Then
/// a search-idle replica shows a new doc after the refresh interval.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn conflicting_dynamic_types_converge_with_replicas() {
    let pl = start_procs(DYNAMIC, &[]).await;
    let master = pl.master(0).await;
    create(&pl, master, "conflict", json!({"settings": {"number_of_shards": 4, "number_of_replicas": 2},
        "mappings": {"properties": {"a": {"type": "keyword", "copy_to": "all"}, "all": {"type": "text"}},
            "runtime": {"ar": {"type": "keyword", "script": {"source": "emit(doc['a'].value + '!')"}}}}})).await;
    let started_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
    let writes = (0..60).map(|i| {
        let (c, port) = (pl.c.clone(), pl.p.http[i % 3]);
        let value = if i % 2 == 0 { json!(7) } else { json!("seven") };
        async move {
            let r = c.put(format!("http://127.0.0.1:{port}/conflict/_doc/{i}")).timeout(Duration::from_secs(60)).json(&json!({"x": value, "a": format!("a{i}")})).send().await;
            (i, r.map(|r| r.status().as_u16()).unwrap_or(0))
        }
    });
    let results = futures::future::join_all(writes).await;
    let acked: BTreeSet<String> = results.iter().filter(|(_, st)| (200..300).contains(st)).map(|(i, _)| i.to_string()).collect();
    let (st, v) = pl.call(master, Method::POST, "/conflict/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
    let mut mapping = None;
    for n in pl.live() {
        let (st, v) = pl.call(n, Method::GET, "/conflict/_mapping", None).await;
        assert_eq!(st, 200, "{v}");
        let m = v["conflict"]["mappings"].clone();
        assert!(m["properties"]["x"]["type"].is_string(), "x is mapped: {m}");
        assert!(mapping.as_ref().is_none_or(|w| *w == m), "n{n}'s mapping differs: {m} vs {mapping:?}");
        mapping = Some(m);
    }
    // No copy failed and recovered again after the writes started.
    let (st, v) = pl.call(master, Method::GET, "/conflict/_recovery", None).await;
    assert_eq!(st, 200, "{v}");
    for r in v["conflict"]["shards"].as_array().into_iter().flatten() {
        assert!(r["start_time_in_millis"].as_u64().unwrap_or(0) < started_ms, "a copy recovered during the writes: {r}");
    }
    // Whichever dynamic type reached the master first is the committed one
    // (ES `MapperService.merge`; the other docs fail to parse): `long`, or a
    // string's `text` with a `keyword` subfield. A `text` field has no
    // doc values to count (ES: 400 "Fielddata is disabled"), its subfield has.
    let x_type = mapping.as_ref().and_then(|m| m["properties"]["x"]["type"].as_str().map(str::to_string)).unwrap_or_default();
    let x_field = if x_type == "text" { "x.keyword" } else { "x" };
    let body = json!({"size": 100, "sort": [{"a": "asc"}], "_source": true, "fields": ["ar"], "track_total_hits": true,
        "aggs": {"x": {"value_count": {"field": x_field}}, "all": {"filter": {"match": {"all": "a1"}}}}});
    let mut ids = BTreeSet::new();
    for k in 0..4 {
        let answers = copies_answer(&pl, master, "conflict", k, &body).await;
        assert_eq!(answers.len(), 3, "three copies of conflict[{k}]");
        assert_copies_agree(&format!("conflict[{k}]"), &answers);
        for h in hits(&answers[0].1) {
            assert_eq!(h["fields"]["ar"][0], format!("{}!", h["_source"]["a"].as_str().unwrap()), "{h}");
            // "seven" cannot be a long (a number can be text).
            assert!(x_type != "long" || h["_source"]["x"].is_number(), "x is long: {h}");
            ids.insert(h["_id"].as_str().unwrap().to_string());
        }
    }
    assert_eq!(ids, acked, "the acked docs, each once");

    // A search-idle replica: no search for 2s, then a write; every copy
    // shows it to the next search.
    create(&pl, master, "idle", json!({"settings": {"number_of_shards": 1, "number_of_replicas": 2, "index.search.idle.after": "1s"}})).await;
    tokio::time::sleep(Duration::from_secs(2)).await;
    let (st, v) = pl.call(master, Method::PUT, "/idle/_doc/1", Some(json!({"v": 1}))).await;
    assert_eq!(st, 201, "{v}");
    // At once: a search-idle copy refreshes for the search and waits for it
    // (`IndexShard.awaitShardSearchActive`), before any scheduled refresh.
    for n in holders(&pl, master, "idle", 0).await {
        let v = search(&pl, n, "/idle/_search?preference=_shards:0%7C_only_local", json!({"size": 0, "track_total_hits": true})).await;
        assert_eq!(v["hits"]["total"]["value"], 1, "n{n}'s copy: {v}");
    }
}
