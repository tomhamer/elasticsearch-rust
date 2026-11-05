//! Many-segment counts (Rally `geopointshape` shape): a `geo_shape` field of
//! points over 30+ segments of one shard, an ES|QL `STATS COUNT(*)` under a
//! spatial predicate and `_count` of the same `geo_shape` query, with the
//! shard's leaves counted concurrently (the default) and one at a time (in
//! a one-thread pool, the path before leaves were counted concurrently).
//!
//! `cargo test --release -p elasticsearch-engine --lib par_count_microbench -- --ignored --nocapture`
//! (`PAR_COUNT_POINTS`, default 2,000,000; `PAR_COUNT_SEGMENTS`, default 32;
//! `PAR_COUNT_RUNS`, default 9, best of).

use crate::index::WriteReq;
use crate::node::Node;
use serde_json::json;
use std::time::Instant;

#[test]
#[ignore]
fn par_count_microbench() {
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (n, segments, runs) = (env("PAR_COUNT_POINTS", 2_000_000), env("PAR_COUNT_SEGMENTS", 32).max(1), env("PAR_COUNT_RUNS", 9).max(1));
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    // A wide tier keeps the flushed segments apart (no background merge).
    node.create_index("osmgeoshapes", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1",
        "index.merge.policy.segments_per_tier": 500, "index.merge.policy.max_merge_at_once": 500},
        "mappings": {"properties": {"location": {"type": "geo_shape"}}}})).unwrap();
    let svc = node.get_index("osmgeoshapes").unwrap();
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; (seed >> 11) as f64 / (1u64 << 53) as f64 };
    let per = n.div_ceil(segments);
    let started = Instant::now();
    let mut batch = Vec::with_capacity(20_000);
    for i in 0..n {
        // Europe-heavy, like the OSM points the Rally polygon covers.
        let (lon, lat) = (-12.0 + rnd() * 45.0, 34.0 + rnd() * 34.0);
        batch.push(WriteReq::Index { id: Some(i.to_string()), source: json!({"location": format!("POINT ({lon} {lat})")}).to_string(), routing: None, cond: Default::default() });
        if batch.len() == 20_000 || (i + 1) % per == 0 || i + 1 == n {
            for r in svc.write(std::mem::take(&mut batch), false) { r.unwrap(); }
            if (i + 1) % per == 0 || i + 1 == n { svc.flush().unwrap(); }
        }
    }
    svc.refresh().unwrap();
    let leaves = svc.local_shards()[0].shard.reader_view().num_leaves();
    eprintln!("{n} points in {leaves} segments, indexed in {:.1} s", started.elapsed().as_secs_f64());
    let poly = "POLYGON((-0.1 49.0, 5.0 48.0, 15.0 49.0, 14.0 60.0, -0.1 61.0, -0.1 49.0))";
    let esql = format!("FROM osmgeoshapes | WHERE ST_Intersects(location, TO_GEOSHAPE(\"{poly}\")) | STATS count=COUNT(*)");
    let query = json!({"geo_shape": {"location": {"shape": "POLYGON((-0.1 49.0, 5.0 48.0, 15.0 49.0, 14.0 60.0, -0.1 61.0, -0.1 49.0))", "relation": "intersects"}}});
    let targets = node.resolve("osmgeoshapes", Default::default()).unwrap();
    let esql_count = || -> u64 {
        let r = crate::esql::run(&node, &esql, crate::esql::EsqlOptions { default_limit: 1000, max_limit: 10_000, ..Default::default() }).unwrap();
        match r.table.rows.first().and_then(|row| row.first()) {
            Some(crate::esql::Val::Long(x)) | Some(crate::esql::Val::Int(x)) => *x as u64,
            other => panic!("count: {other:?}"),
        }
    };
    let count = || crate::search::par_count(&targets, &query).unwrap();
    let best = |f: &dyn Fn() -> u64| -> (f64, u64) {
        let mut out = 0;
        let ms = (0..runs).map(|_| { let t = Instant::now(); out = std::hint::black_box(f()); t.elapsed().as_secs_f64() * 1e3 }).fold(f64::INFINITY, f64::min);
        (ms, out)
    };
    let one_thread = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    for (label, f) in [("ES|QL COUNT(*) WHERE ST_Intersects", &esql_count as &(dyn Fn() -> u64 + Sync)), ("_count geo_shape intersects", &count)] {
        let (par, a) = best(f);
        let (seq, b) = one_thread.install(|| best(f));
        assert_eq!(a, b, "{label}: the same count either way");
        eprintln!("{label:<38} concurrent {par:>8.2} ms   one at a time {seq:>8.2} ms   ({a} matches)");
    }
}
