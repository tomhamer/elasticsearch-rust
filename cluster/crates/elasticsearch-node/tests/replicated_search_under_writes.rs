//! P0-4: searches whose phases run separately (kNN's DFS phase then the
//! query, a hybrid kNN + match search, collapse then its inner_hits expand)
//! while writes and deletes go on and copies refresh at their own pace, on
//! three node processes ([`common::procs`]), a 3x2 index.
//!
//! ES runs the kNN DFS and query phases against the copies the first phase
//! chose; the collapse expand is a separate search. So:
//! - a kNN search returns exactly `k` hits (total `k`, relation `eq`), each
//!   a live doc; two docs sharing an `_id` under different routings are two
//!   hits, each once;
//! - no hit is a doc whose delete was acked (`refresh=wait_for`: visible on
//!   every copy) before the search started;
//! - a collapsed hit's inner_hits all belong to its group (ES's
//!   `ExpandSearchPhase` re-searches each group, so the head itself may be
//!   gone by then).
mod common;
use common::procs::{start_procs, Procs};
use reqwest::Method;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const KNN: usize = 3;
const DOCS: usize = 2000;
const WORDS: [&str; 6] = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"];

fn vector(i: usize) -> Value {
    let x = i as f64;
    json!([(x * 0.37).sin(), (x * 0.71).cos(), (i % 13) as f64 / 13.0, (i % 7) as f64 / 7.0])
}

fn doc(i: usize, round: usize) -> Value {
    json!({"uid": format!("d{i:05}"), "g": format!("g{}", i % 20), "t": format!("{} {}", WORDS[i % 6], WORDS[(i / 6) % 6]), "v": vector(i + round * 7919)})
}

/// Ids whose delete was acked, with when.
type Deleted = Arc<Mutex<BTreeMap<String, Instant>>>;

/// Hits of `v` deleted (acked) before `started`.
fn deleted_hits(v: &Value, deleted: &Deleted, started: Instant) -> Vec<String> {
    let d = deleted.lock().unwrap();
    v["hits"]["hits"].as_array().into_iter().flatten()
        .filter_map(|h| h["_id"].as_str())
        .filter(|id| d.get(*id).is_some_and(|at| *at < started))
        .map(String::from).collect()
}

async fn search(pl: &Procs, n: usize, path: &str, body: &Value) -> Result<Value, String> {
    let (st, v) = pl.call(n, Method::POST, path, Some(body.clone())).await;
    let s = &v["_shards"];
    if st != 200 || s["failed"] != 0 || s["successful"] != s["total"] {
        return Err(format!("n{n} {path}: {st} {v}"));
    }
    Ok(v)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn knn_hybrid_and_collapse_stay_consistent_under_writes() {
    let pl = start_procs(KNN, &[]).await;
    let master = pl.master(0).await;
    let (st, v) = pl.call(master, Method::PUT, "/vec", Some(json!({"settings": {"number_of_shards": 3, "number_of_replicas": 2, "refresh_interval": "200ms"},
        "mappings": {"properties": {"uid": {"type": "keyword"}, "g": {"type": "keyword"}, "t": {"type": "text"},
            "v": {"type": "dense_vector", "dims": 4, "index": true, "similarity": "l2_norm"}}}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(master, "vec", "green", 90).await;
    let lines: String = (0..DOCS).map(|i| format!("{}\n{}\n", json!({"index": {"_id": format!("d{i:05}")}}), doc(i, 0))).collect();
    let (st, v) = pl.ndjson(master, "/vec/_bulk?refresh=true", lines).await;
    assert!(st == 200 && v["errors"] == false, "{v}");
    // Two docs with `_id` `dup` on different shards (different routings),
    // both right at the query vector.
    let q0 = json!([0.5, 0.5, 0.5, 0.5]);
    let dup = |r: &str| json!({"uid": format!("dup-{r}"), "g": "gdup", "t": "dup", "v": q0});
    let (st, v) = pl.call(master, Method::PUT, "/vec/_doc/dup?routing=ra&refresh=true", Some(dup("ra"))).await;
    assert_eq!(st, 201, "{v}");
    let mut placed = false;
    for j in 0..50 {
        let r = format!("rb{j}");
        let (st, v) = pl.call(master, Method::PUT, &format!("/vec/_doc/dup?routing={r}&refresh=true"), Some(dup(&r))).await;
        assert!(st == 201 || st == 200, "{v}");
        let v = search(&pl, master, "/vec/_search", &json!({"size": 0, "track_total_hits": true, "query": {"ids": {"values": ["dup"]}}})).await.unwrap();
        if v["hits"]["total"]["value"] == 2 {
            placed = true;
            break;
        }
        let (st, v) = pl.call(master, Method::DELETE, &format!("/vec/_doc/dup?routing={r}&refresh=true"), None).await;
        assert_eq!(st, 200, "{v}");
    }
    assert!(placed, "no routing put the second dup on another shard");

    // Two writers: rewrites and deletes (never re-created), refresh=wait_for.
    let deleted: Deleted = Arc::default();
    let stop = Arc::new(AtomicBool::new(false));
    let writers: Vec<_> = (0..2usize).map(|w| {
        let (c, port, stop, deleted) = (pl.c.clone(), pl.p.http[(master + 1 + w) % 3], stop.clone(), deleted.clone());
        tokio::spawn(async move {
            let mut round = 1;
            // This writer's half of the ids in a fixed shuffled order, so
            // deletes go on for the whole run (37 is coprime with DOCS / 2).
            let mut k = 0usize;
            while !stop.load(Ordering::Relaxed) {
                let mut lines = String::new();
                let mut gone = Vec::new();
                for j in 0..20 {
                    let i = (round * 40 + j * 2 + w) % DOCS;
                    if j % 4 == 0 && k < DOCS / 2 {
                        let id = format!("d{:05}", w + 2 * ((k * 37) % (DOCS / 2)));
                        lines.push_str(&format!("{}\n", json!({"delete": {"_id": id}})));
                        gone.push(id);
                        k += 1;
                    } else if !deleted.lock().unwrap().contains_key(&format!("d{i:05}")) && !gone.contains(&format!("d{i:05}")) {
                        lines.push_str(&format!("{}\n{}\n", json!({"index": {"_id": format!("d{i:05}")}}), doc(i, round)));
                    }
                }
                let r = c.post(format!("http://127.0.0.1:{port}/vec/_bulk?refresh=wait_for")).header("content-type", "application/x-ndjson")
                    .body(lines).timeout(Duration::from_secs(60)).send().await;
                if let Ok(r) = r {
                    let v: Value = r.json().await.unwrap_or(Value::Null);
                    let at = Instant::now();
                    for it in v["items"].as_array().into_iter().flatten() {
                        if it["delete"]["status"] == 200 {
                            deleted.lock().unwrap().insert(it["delete"]["_id"].as_str().unwrap_or_default().to_string(), at);
                        }
                    }
                }
                round += 1;
            }
        })
    }).collect();

    let mut errs = Vec::new();
    let begun = Instant::now();
    let mut runs = 0;
    while begun.elapsed() < Duration::from_secs(12) {
        for n in pl.live() {
            runs += 1;
            // kNN alone, at the dup docs.
            let started = Instant::now();
            let knn = json!({"knn": {"field": "v", "query_vector": q0, "k": 10, "num_candidates": 50}, "size": 10, "_source": ["uid"]});
            match search(&pl, n, "/vec/_search", &knn).await {
                Ok(v) => {
                    let hs = v["hits"]["hits"].as_array().cloned().unwrap_or_default();
                    if hs.len() != 10 || v["hits"]["total"]["value"] != 10 || v["hits"]["total"]["relation"] != "eq" {
                        errs.push(format!("n{n} knn: {} hits, total {}", hs.len(), v["hits"]["total"]));
                    }
                    let pairs: BTreeSet<(String, String)> = hs.iter().map(|h| (h["_id"].to_string(), h["_routing"].to_string())).collect();
                    if pairs.len() != hs.len() {
                        errs.push(format!("n{n} knn: a doc twice: {}", json!(hs)));
                    }
                    if hs.iter().filter(|h| h["_id"] == "dup").count() != 2 {
                        errs.push(format!("n{n} knn: the two dup docs, once each: {}", json!(hs)));
                    }
                    let d = deleted_hits(&v, &deleted, started);
                    if !d.is_empty() {
                        errs.push(format!("n{n} knn: deleted docs {d:?}"));
                    }
                }
                Err(e) => errs.push(e),
            }
            // Hybrid: kNN folded into a match query (DFS).
            let started = Instant::now();
            let qv = vector(runs * 31);
            let hybrid = json!({"knn": {"field": "v", "query_vector": qv, "k": 10, "num_candidates": 50}, "query": {"match": {"t": "alpha"}}, "size": 50, "_source": ["uid"], "track_total_hits": true});
            match search(&pl, n, "/vec/_search", &hybrid).await {
                Ok(v) => {
                    let d = deleted_hits(&v, &deleted, started);
                    if !d.is_empty() {
                        errs.push(format!("n{n} hybrid: deleted docs {d:?}"));
                    }
                    if v["hits"]["total"]["value"].as_u64().unwrap_or(0) < 10 {
                        errs.push(format!("n{n} hybrid: total {}", v["hits"]["total"]));
                    }
                }
                Err(e) => errs.push(e),
            }
            // Collapse + inner_hits, rotating copies and this node's own.
            for pref in ["", "?preference=_shards:0,1,2%7C_only_local"] {
                let started = Instant::now();
                let body = json!({"size": 20, "query": {"match_all": {}}, "sort": [{"uid": "asc"}], "_source": ["uid", "g"],
                    "collapse": {"field": "g", "inner_hits": {"name": "top", "size": 3, "sort": [{"uid": "asc"}], "_source": ["uid", "g"]}}});
                match search(&pl, n, &format!("/vec/_search{pref}"), &body).await {
                    Ok(v) => {
                        for h in v["hits"]["hits"].as_array().into_iter().flatten() {
                            let inner = h["inner_hits"]["top"]["hits"]["hits"].as_array().cloned().unwrap_or_default();
                            // ES's `ExpandSearchPhase` runs a new search per group
                            // (only preference and routing carried over), so under
                            // writes the head need not be among its inner hits; the
                            // group key always holds.
                            if inner.iter().any(|i| i["_source"]["g"] != h["_source"]["g"]) {
                                errs.push(format!("n{n} collapse{pref}: inner hits of {} ({}) from another group: {}", h["_id"], h["_source"]["g"], json!(inner)));
                            }
                            let d = deleted_hits(&json!({"hits": {"hits": inner}}), &deleted, started);
                            if !d.is_empty() {
                                errs.push(format!("n{n} collapse{pref}: deleted inner hits {d:?}"));
                            }
                        }
                        let d = deleted_hits(&v, &deleted, started);
                        if !d.is_empty() {
                            errs.push(format!("n{n} collapse{pref}: deleted docs {d:?}"));
                        }
                    }
                    Err(e) => errs.push(e),
                }
            }
        }
    }
    stop.store(true, Ordering::Relaxed);
    for w in writers {
        w.await.unwrap();
    }
    assert!(!deleted.lock().unwrap().is_empty(), "the writers deleted docs");
    errs.truncate(40);
    assert!(errs.is_empty(), "{} inconsistent searches of {runs} rounds:\n{}", errs.len(), errs.join("\n"));
}
