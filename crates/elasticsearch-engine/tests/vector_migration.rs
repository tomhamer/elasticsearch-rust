//! A `dense_vector` `index_options` update (`hnsw` → `int8_hnsw`) on a
//! field holding data, with bulks and kNN searches running throughout: the
//! mapping update only schedules the new graph, the background vector
//! maintenance builds it off the shard locks and swaps it in, writes never
//! wait for the build, no row is lost (written before, during or after the
//! build, sealed or not) and every search finds its document throughout.

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const DIMS: usize = 16;
const INITIAL: usize = 6_000;
const BATCH: usize = 50;

fn vector(i: usize) -> Vec<f32> {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..DIMS).map(|_| {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        (x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }).collect()
}

fn bulk(ids: std::ops::Range<usize>) -> Vec<WriteReq> {
    ids.map(|i| WriteReq::Index { id: Some(format!("d{i}")), source: json!({"v": vector(i), "n": i}).to_string(),
        routing: None, cond: Default::default() }).collect()
}

/// The ids of a k=10 kNN search for document `i`'s own vector (no refresh).
fn knn_ids(node: &Node, i: usize) -> Vec<String> {
    let body = json!({"knn": {"field": "v", "query_vector": vector(i), "k": 10, "num_candidates": 100}, "size": 10, "_source": false});
    let req = search::parse_request(&body, &HashMap::new()).unwrap();
    let targets = node.resolve("vec", Default::default()).unwrap();
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let out: Value = serde_json::from_str(&search::execute(&targets, &req, &opts).unwrap().json).unwrap();
    out["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect()
}

#[test]
fn index_options_migration_runs_in_the_background_under_concurrent_bulks_and_searches() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    node.create_index("vec", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"properties": {
        "v": {"type": "dense_vector", "dims": DIMS, "index": true, "similarity": "l2_norm", "index_options": {"type": "hnsw"}},
        "n": {"type": "long"}}}})).unwrap();
    let svc = node.get_index("vec").unwrap();
    for start in (0..INITIAL).step_by(500) {
        assert!(svc.write(bulk(start..start + 500), false).iter().all(Result::is_ok));
        if start % 2_000 == 0 { svc.refresh().unwrap(); }
    }
    // Some rows sealed, some still in the writing memtable.
    let shard = svc.local_shards().into_iter().next().unwrap();
    let field = shard.shard.core.read().field_id("v").unwrap();
    let store = || shard.shard.core.read().vector_store().clone();
    assert_eq!(store().mutable_active_nodes(field), Some(INITIAL as u32));

    // The mapping update and its shard sync do no graph work.
    let started = Instant::now();
    svc.put_mapping(&json!({"properties": {"v": {"type": "dense_vector", "dims": DIMS, "index": true, "similarity": "l2_norm",
        "index_options": {"type": "int8_hnsw"}}}})).unwrap();
    shard.sync_schema().unwrap();
    let update = started.elapsed();
    assert_eq!(store().migrations_pending(), 1, "scheduled, not built under the shard lock");

    let done = Arc::new(AtomicBool::new(false));
    let written = Arc::new(AtomicUsize::new(INITIAL));
    let writer = {
        let (node, done, written) = (node.clone(), done.clone(), written.clone());
        std::thread::spawn(move || {
            let svc = node.get_index("vec").unwrap();
            let mut slowest = Duration::ZERO;
            let mut batches = 0;
            while !done.load(Ordering::Acquire) || batches < 20 {
                let start = written.load(Ordering::Acquire);
                let began = Instant::now();
                for r in svc.write(bulk(start..start + BATCH), false) { r.unwrap(); }
                slowest = slowest.max(began.elapsed());
                written.store(start + BATCH, Ordering::Release);
                batches += 1;
                // Seals during the build: rows reserved in the old graph,
                // in the new one, and pooled ones.
                if batches % 8 == 0 { svc.refresh().unwrap(); }
            }
            slowest
        })
    };
    let searcher = {
        let (node, done) = (node.clone(), done.clone());
        std::thread::spawn(move || {
            let (mut searches, mut misses) = (0usize, Vec::new());
            let mut i = 7usize;
            while !done.load(Ordering::Acquire) || searches < 50 {
                i = (i * 7_919 + 13) % INITIAL;
                let ids = knn_ids(&node, i);
                assert_eq!(ids.len(), 10, "search {searches}");
                if !ids.contains(&format!("d{i}")) { misses.push(i); }
                searches += 1;
            }
            (searches, misses)
        })
    };

    // Background maintenance passes until the new graph is swapped in.
    let started = Instant::now();
    while store().migrations_pending() > 0 {
        node.optimize_vectors_all();
        assert!(started.elapsed() < Duration::from_secs(600), "migration never completed");
    }
    let migration = started.elapsed();
    done.store(true, Ordering::Release);
    let slowest = writer.join().unwrap();
    let (searches, misses) = searcher.join().unwrap();

    assert!(update < migration / 4, "the mapping update took {update:?} of a {migration:?} migration");
    assert!(slowest < migration / 2 || slowest < Duration::from_millis(250),
        "a bulk waited {slowest:?} during a {migration:?} migration: the build held the shard");
    assert!(misses.len() * 100 <= searches, "{} of {searches} searches missed their own document: {misses:?}", misses.len());

    // No lost row: every document is in the new graph, and every one written
    // during the build is found by its own vector.
    let total = written.load(Ordering::Acquire);
    svc.refresh().unwrap();
    node.optimize_vectors_all();
    assert_eq!(store().migrations_pending(), 0);
    assert_eq!(store().mutable_active_nodes(field), Some(total as u32), "every row written before, during and after the build");
    let late: Vec<usize> = (INITIAL..total).step_by(37).filter(|&i| !knn_ids(&node, i).contains(&format!("d{i}"))).collect();
    assert!(late.len() * 100 <= (total - INITIAL) / 37 + 1, "documents written during the build not found: {late:?}");
    // The update took effect.
    let mapping = svc.meta().mappings.to_json();
    assert_eq!(mapping["properties"]["v"]["index_options"]["type"], "int8_hnsw", "{mapping}");
}

/// `(id, _score)` of a k=10 kNN search of `index` for document `i`'s own
/// vector, without rescoring (`oversample: 0`: the quantized estimates).
fn knn_estimates(node: &Node, index: &str, i: usize) -> Vec<(String, f64)> {
    let body = json!({"knn": {"field": "v", "query_vector": vector(i), "k": 10, "num_candidates": 100,
        "rescore_vector": {"oversample": 0}}, "size": 10, "_source": false});
    let req = search::parse_request(&body, &HashMap::new()).unwrap();
    let targets = node.resolve(index, Default::default()).unwrap();
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let out: Value = serde_json::from_str(&search::execute(&targets, &req, &opts).unwrap().json).unwrap();
    out["hits"]["hits"].as_array().unwrap().iter()
        .map(|h| (h["_id"].as_str().unwrap().to_string(), h["_score"].as_f64().unwrap())).collect()
}

/// `flat` → `int8_flat` → `int4_flat` updates of a field whose rows are
/// already sealed (some with the old encoding's stored codes): searches
/// score with the new encoding at once, exactly as an index created with
/// it (each segment's codes are fitted per encoding from its f32 rows,
/// never read from the stored copy of another), and keep doing so after
/// the optimizer rewrote the segments.
#[test]
fn flat_index_options_upgrades_score_with_the_new_encoding() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    let mapping = |ty: &str| json!({"properties": {
        "v": {"type": "dense_vector", "dims": DIMS, "index": true, "similarity": "l2_norm", "index_options": {"type": ty}},
        "n": {"type": "long"}}});
    for (index, ty) in [("up", "flat"), ("int8", "int8_flat"), ("int4", "int4_flat")] {
        node.create_index(index, &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
            "mappings": mapping(ty)})).unwrap();
        let svc = node.get_index(index).unwrap();
        assert!(svc.write(bulk(0..600), false).iter().all(Result::is_ok));
        svc.refresh().unwrap();
        svc.flush().unwrap();
    }
    node.optimize_vectors_all();
    let up = node.get_index("up").unwrap();
    for (ty, oracle) in [("int8_flat", "int8"), ("int4_flat", "int4")] {
        up.put_mapping(&mapping(ty)).unwrap();
        for shard in up.local_shards() {
            shard.sync_schema().unwrap();
        }
        for round in 0..2 {
            for i in [3, 77, 401, 599] {
                let (got, want) = (knn_estimates(&node, "up", i), knn_estimates(&node, oracle, i));
                assert_eq!(got.len(), 10, "{ty} query {i}");
                let ids = |hits: &[(String, f64)]| hits.iter().map(|h| h.0.clone()).collect::<Vec<_>>();
                assert_eq!(ids(&got), ids(&want), "{ty} query {i} (round {round}): ids differ from an index created as {ty}");
                for ((_, a), (_, b)) in got.iter().zip(&want) {
                    assert!((a - b).abs() <= 1e-6, "{ty} query {i} (round {round}): score {a} != {b}");
                }
            }
            // The optimizer rewrites the segments with the new encoding's
            // stored codes; scores do not move.
            node.optimize_vectors_all();
        }
        let m = up.meta().mappings.to_json();
        assert_eq!(m["properties"]["v"]["index_options"]["type"], ty, "{m}");
    }
}

/// 180_update_dense_vector_type "bulk indexing": rows written while the
/// field had no graph (`flat`) and not sealed yet (a refresh publishes the
/// memtable without storing its rows) hold no point and are in no graph. A
/// graph built for a later `hnsw` mapping from the stored rows lacks them
/// until their memtable seals, so a kNN right after found only the rows
/// written since. They are scored exactly meanwhile.
#[test]
fn rows_written_before_a_graph_exists_are_found_before_their_memtable_seals() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    node.create_index("flatmig", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
        "mappings": {"properties": {"v": {"type": "dense_vector", "dims": DIMS, "index": true, "similarity": "l2_norm",
            "index_options": {"type": "flat"}}}}})).unwrap();
    let svc = node.get_index("flatmig").unwrap();
    let docs = |ids: std::ops::Range<usize>| -> Vec<WriteReq> { ids.map(|i| WriteReq::Index { id: Some(format!("d{i}")),
        source: json!({"v": vector(i)}).to_string(), routing: None, cond: Default::default() }).collect() };
    assert!(svc.write(docs(0..30), false).iter().all(Result::is_ok));
    svc.refresh().unwrap();
    let shard = svc.local_shards().into_iter().next().unwrap();
    svc.put_mapping(&json!({"properties": {"v": {"type": "dense_vector", "dims": DIMS, "index": true, "similarity": "l2_norm",
        "index_options": {"type": "hnsw"}}}})).unwrap();
    shard.sync_schema().unwrap();
    let store = || shard.shard.core.read().vector_store().clone();
    let started = Instant::now();
    while store().migrations_pending() > 0 {
        node.optimize_vectors_all();
        assert!(started.elapsed() < Duration::from_secs(60), "migration never completed");
    }
    assert!(svc.write(docs(30..40), false).iter().all(Result::is_ok));
    svc.refresh().unwrap();
    let all = |node: &Node| -> Vec<String> {
        let body = json!({"knn": {"field": "v", "query_vector": vector(0), "k": 40, "num_candidates": 40}, "size": 40, "_source": false});
        let req = search::parse_request(&body, &HashMap::new()).unwrap();
        let targets = node.resolve("flatmig", Default::default()).unwrap();
        let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
        let out: Value = serde_json::from_str(&search::execute(&targets, &req, &opts).unwrap().json).unwrap();
        let mut ids: Vec<String> = out["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
        ids.sort();
        ids
    };
    let mut expected: Vec<String> = (0..40).map(|i| format!("d{i}")).collect();
    expected.sort();
    assert_eq!(all(&node), expected, "every row, graphless memtable rows included");
    // Once stored (the seal inserts them into the graph), still every row once.
    svc.flush().unwrap();
    node.optimize_vectors_all();
    assert_eq!(all(&node), expected);
}
