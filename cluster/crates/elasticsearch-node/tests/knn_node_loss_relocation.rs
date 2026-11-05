//! kNN on a 3-shard, 1-replica `dense_vector` index across three node
//! processes ([`common::procs`]) while writes and deletes go on, a node is
//! killed and comes back, and shards relocate off a node and back:
//! - every kNN answer during the disruption holds at most `k` hits, no
//!   `_id` twice, and no doc whose delete was acked (`refresh=wait_for`)
//!   before the search started;
//! - afterwards no acknowledged write is lost and no deleted doc is back:
//!   the live ids are exactly the acked ones (writes whose outcome the
//!   client never learned may be either);
//! - kNN converges: every node, under every preference (so every copy),
//!   returns `k` live hits with the exact neighbours' recall, and the same
//!   answer each time it is asked.
mod common;
use common::procs::{start_procs, Procs};
use reqwest::Method;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const NODES: usize = 3;
const DIMS: usize = 8;
const INITIAL: usize = 1500;
const K: usize = 10;

fn vector(i: usize) -> Vec<f32> {
    let x = i as f32;
    (0..DIMS).map(|d| ((x * (0.13 + d as f32 * 0.071)).sin() + (i % (7 + d)) as f32 / (7 + d) as f32)).collect()
}

fn l2_score(q: &[f32], v: &[f32]) -> f64 {
    1.0 / (1.0 + q.iter().zip(v).map(|(a, b)| ((*a - *b) as f64).powi(2)).sum::<f64>())
}

/// What the writers learned: acked live docs, acked deletes (with when),
/// and ids whose last request got no answer (either outcome is allowed).
#[derive(Default)]
struct Ledger {
    live: BTreeSet<usize>,
    deleted: BTreeMap<usize, Instant>,
    uncertain: BTreeSet<usize>,
}

fn id_num(id: &str) -> usize {
    id[1..].parse().unwrap()
}

async fn knn(pl: &Procs, node: usize, q: &[f32], preference: Option<&str>) -> Result<Vec<String>, String> {
    let path = match preference {
        Some(p) => format!("/vec/_search?preference={p}"),
        None => "/vec/_search".to_string(),
    };
    let body = json!({"knn": {"field": "v", "query_vector": q, "k": K, "num_candidates": 100}, "size": K, "_source": false});
    let (st, v) = pl.call(node, Method::POST, &path, Some(body)).await;
    if st != 200 || v["_shards"]["failed"] != 0 {
        return Err(format!("n{node} {path}: {st} {v}"));
    }
    Ok(v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect())
}

async fn wait_green_on(pl: &Procs, via: usize, nodes: usize, secs: u64) {
    pl.wait_nodes(via, nodes, secs).await;
    pl.wait_health(via, "vec", "green", secs).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn knn_survives_node_loss_and_relocation_without_lost_or_duplicate_hits() {
    let mut pl = start_procs(NODES, &[]).await;
    let master = pl.master(0).await;
    let (st, v) = pl.call(master, Method::PUT, "/vec", Some(json!({
        "settings": {"number_of_shards": 3, "number_of_replicas": 1, "refresh_interval": "200ms"},
        "mappings": {"properties": {"n": {"type": "long"},
            "v": {"type": "dense_vector", "dims": DIMS, "index": true, "similarity": "l2_norm", "index_options": {"type": "hnsw"}}}}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(master, "vec", "green", 90).await;
    let lines: String = (0..INITIAL).map(|i| format!("{}\n{}\n", json!({"index": {"_id": format!("d{i}")}}), json!({"n": i, "v": vector(i)}))).collect();
    let (st, v) = pl.ndjson(master, "/vec/_bulk?refresh=true", lines).await;
    assert!(st == 200 && v["errors"] == false, "{v}");
    let ledger = Arc::new(Mutex::new(Ledger { live: (0..INITIAL).collect(), ..Default::default() }));

    // One writer: new docs (ids never reused) and deletes of older ones,
    // sent to whichever node answers; refresh=wait_for.
    let stop = Arc::new(AtomicBool::new(false));
    let writer = {
        let (c, ports, stop, ledger) = (pl.c.clone(), pl.p.http.clone(), stop.clone(), ledger.clone());
        tokio::spawn(async move {
            let mut next = INITIAL;
            let mut round = 0usize;
            while !stop.load(Ordering::Relaxed) {
                round += 1;
                let mut lines = String::new();
                let mut sent: Vec<(usize, bool)> = Vec::new();
                for _ in 0..10 {
                    lines.push_str(&format!("{}\n{}\n", json!({"index": {"_id": format!("d{next}")}}), json!({"n": next, "v": vector(next)})));
                    sent.push((next, true));
                    next += 1;
                }
                for j in 0..3 {
                    let victim = (round * 53 + j * 307) % next;
                    let l = ledger.lock().unwrap();
                    if l.live.contains(&victim) && !sent.iter().any(|s| s.0 == victim) {
                        lines.push_str(&format!("{}\n", json!({"delete": {"_id": format!("d{victim}")}})));
                        sent.push((victim, false));
                    }
                }
                let port = ports[round % ports.len()];
                let r = c.post(format!("http://127.0.0.1:{port}/vec/_bulk?refresh=wait_for")).header("content-type", "application/x-ndjson")
                    .body(lines).timeout(Duration::from_secs(60)).send().await;
                let v: Value = match r { Ok(r) => r.json().await.unwrap_or(Value::Null), Err(_) => Value::Null };
                let at = Instant::now();
                {
                let mut l = ledger.lock().unwrap();
                let items = v["items"].as_array().cloned().unwrap_or_default();
                for (n, index) in &sent {
                    let item = items.iter().find(|it| {
                        let op = if *index { &it["index"] } else { &it["delete"] };
                        op["_id"].as_str() == Some(format!("d{n}").as_str())
                    });
                    let op = item.map(|it| if *index { it["index"].clone() } else { it["delete"].clone() });
                    match (op.as_ref().and_then(|o| o["status"].as_u64()), *index) {
                        (Some(200 | 201), true) => { l.live.insert(*n); l.uncertain.remove(n); }
                        (Some(200), false) => { l.live.remove(n); l.uncertain.remove(n); l.deleted.insert(*n, at); }
                        // A 5xx item may have executed (ES: a node closing or a
                        // connection lost after the primary applied it, then
                        // reported with the lost attempt's error instead of the
                        // retry's own-echo 404/409): its outcome is unknown, as
                        // when no answer came. A delete of d1462 reported this
                        // way had applied (seq_no 2280 in the primary's
                        // translog) while this ledger still counted it live.
                        (Some(status), _) if status >= 500 => { l.uncertain.insert(*n); if !index { l.live.remove(n); } }
                        // Rejected outright (a 4xx item): nothing changed for a
                        // create; a delete that failed leaves the doc as it was.
                        (Some(_), _) => {}
                        // No answer: the op may or may not have applied.
                        (None, _) => { l.uncertain.insert(*n); if !index { l.live.remove(n); } }
                    }
                }
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
    };

    // Searches throughout: invariants on every answer that has no shard
    // failure (copies are expected to fail while a node is gone).
    let searching = Arc::new(AtomicBool::new(true));
    let errors = Arc::new(Mutex::new(Vec::<String>::new()));
    let pl = Arc::new(tokio::sync::Mutex::new(pl));
    let searcher = {
        let (pl, searching, errors, ledger) = (pl.clone(), searching.clone(), errors.clone(), ledger.clone());
        tokio::spawn(async move {
            let mut i = 0usize;
            let mut answered = 0usize;
            while searching.load(Ordering::Relaxed) {
                i += 1;
                let q = vector(i * 7919 % INITIAL);
                let started = Instant::now();
                let guard = pl.lock().await;
                let live = guard.live();
                let node = live[i % live.len()];
                let r = knn(&guard, node, &q, None).await;
                drop(guard);
                let Ok(hits) = r else { continue };
                answered += 1;
                let ids: BTreeSet<&String> = hits.iter().collect();
                let mut e = Vec::new();
                if hits.len() > K { e.push(format!("{} hits", hits.len())); }
                if ids.len() != hits.len() { e.push(format!("an _id twice: {hits:?}")); }
                {
                    let l = ledger.lock().unwrap();
                    let gone: Vec<&String> = hits.iter().filter(|h| l.deleted.get(&id_num(h)).is_some_and(|at| *at < started)).collect();
                    if !gone.is_empty() { e.push(format!("deleted docs {gone:?}")); }
                }
                if !e.is_empty() { errors.lock().unwrap().push(format!("n{node} search {i}: {}", e.join("; "))); }
                tokio::time::sleep(Duration::from_millis(30)).await;
            }
            answered
        })
    };

    tokio::time::sleep(Duration::from_secs(3)).await;
    // Node loss: a non-master node is killed; its copies are rebuilt on
    // the other two, then it rejoins.
    let victim = (master + 1) % 3;
    {
        let mut g = pl.lock().await;
        g.kill(victim);
    }
    {
        let g = pl.lock().await;
        wait_green_on(&g, master, 2, 180).await;
    }
    tokio::time::sleep(Duration::from_secs(2)).await;
    {
        let mut g = pl.lock().await;
        g.spawn(victim, &[]).await;
    }
    {
        let g = pl.lock().await;
        wait_green_on(&g, master, 3, 180).await;
    }
    // Relocation: every copy moves off another node, then may come back.
    let moved = (master + 2) % 3;
    {
        let g = pl.lock().await;
        g.settings(master, "persistent", json!({"cluster.routing.allocation.exclude._name": format!("n{moved}")})).await;
        let deadline = Instant::now() + Duration::from_secs(180);
        loop {
            let shards = g.shards(master, "vec").await;
            let settled = !shards.is_empty() && shards.iter().all(|s| s["state"] == "STARTED" && s["node"] != format!("n{moved}").as_str());
            if settled { break; }
            assert!(Instant::now() < deadline, "copies did not relocate off n{moved}: {shards:?}");
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        g.settings(master, "persistent", json!({"cluster.routing.allocation.exclude._name": null})).await;
        wait_green_on(&g, master, 3, 180).await;
    }
    tokio::time::sleep(Duration::from_secs(2)).await;
    stop.store(true, Ordering::Relaxed);
    writer.await.unwrap();
    searching.store(false, Ordering::Relaxed);
    let answered = searcher.await.unwrap();
    assert!(answered > 0, "no search was answered during the run");
    let errors = errors.lock().unwrap().clone();
    assert!(errors.is_empty(), "{} bad kNN answers, first: {:?}", errors.len(), &errors[..errors.len().min(5)]);

    let g = pl.lock().await;
    let (st, v) = g.call(master, Method::POST, "/vec/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
    let l = ledger.lock().unwrap();
    // No lost acked write, no deleted doc back, from every node.
    for node in 0..3 {
        let (st, v) = g.call(node, Method::POST, "/vec/_search", Some(json!({"size": 10000, "_source": false, "query": {"match_all": {}}, "track_total_hits": true}))).await;
        assert!(st == 200 && v["_shards"]["failed"] == 0, "{st} {v}");
        let hits = v["hits"]["hits"].as_array().unwrap();
        assert_eq!(v["hits"]["total"]["value"].as_u64(), Some(hits.len() as u64), "n{node}: more docs than one page holds");
        let seen: Vec<usize> = hits.iter().map(|h| id_num(h["_id"].as_str().unwrap())).collect();
        let set: BTreeSet<usize> = seen.iter().copied().collect();
        assert_eq!(set.len(), seen.len(), "n{node}: an _id twice");
        let lost: Vec<&usize> = l.live.difference(&set).filter(|n| !l.uncertain.contains(n)).collect();
        assert!(lost.is_empty(), "n{node}: acked docs lost: {lost:?}");
        let back: Vec<&usize> = set.difference(&l.live).filter(|n| !l.uncertain.contains(n)).collect();
        assert!(back.is_empty(), "n{node}: deleted or never-acked docs present: {back:?}");
    }
    // Converged kNN: every copy (preference strings route to different
    // copies) returns k live hits with the exact neighbours' recall, and
    // the same answer when asked again.
    let present: BTreeSet<usize> = {
        let (_, v) = g.call(master, Method::POST, "/vec/_search", Some(json!({"size": 10000, "_source": false, "query": {"match_all": {}}}))).await;
        v["hits"]["hits"].as_array().unwrap().iter().map(|h| id_num(h["_id"].as_str().unwrap())).collect()
    };
    for qi in 0..6 {
        let q = vector(qi * 211 + 3);
        let mut exact: Vec<(usize, f64)> = present.iter().map(|&d| (d, l2_score(&q, &vector(d)))).collect();
        exact.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let truth: BTreeSet<String> = exact[..K].iter().map(|(d, _)| format!("d{d}")).collect();
        for node in 0..3 {
            for pref in ["copy-a", "copy-b", "copy-c", "copy-d"] {
                let first = knn(&g, node, &q, Some(pref)).await.unwrap();
                assert_eq!(first.len(), K, "n{node} {pref} query {qi}");
                assert!(first.iter().all(|h| present.contains(&id_num(h))), "n{node} {pref} query {qi}: a hit not live: {first:?}");
                let found = first.iter().filter(|h| truth.contains(*h)).count();
                assert!(found * 10 >= K * 9, "n{node} {pref} query {qi}: recall {found}/{K}: {first:?} vs {truth:?}");
                let again = knn(&g, node, &q, Some(pref)).await.unwrap();
                assert_eq!(first, again, "n{node} {pref} query {qi}: the same copy answered differently");
            }
        }
    }
}
