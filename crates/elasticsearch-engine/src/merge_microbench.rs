//! Merge cost of a nested-like (Rally `nested`: questions with nested
//! answers) and a geonames-like (`geo_point`, text, keywords, numbers)
//! corpus, indexed in refreshed batches with background merges running, as
//! a Rally bulk phase does: merges run, IO-limited merges, input bytes
//! merged (write amplification against the final store), summed merge wall
//! time (what `merges.total_time_in_millis` reports) and its stages, and the
//! throttle and pacing sleeps; then the same with the IO limit off, which
//! shows how much of the merge time is the throttle.
//!
//! `cargo test --release -p elasticsearch-engine --lib merge_microbench -- --ignored --nocapture`
//! (`MERGE_BENCH_DOCS`, default 400,000 top-level docs per corpus;
//! `MERGE_BENCH_REFRESH_DOCS`, default 20,000: docs between refreshes).

use crate::index::WriteReq;
use crate::node::Node;
use rustic_core::merge::{merge_sleep_ns, set_merge_io_adaptive, set_merge_io_limit, MERGE_TOTALS};
use serde_json::{json, Value};
use std::sync::atomic::Ordering::Relaxed;
use std::time::{Duration, Instant};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn below(&mut self, n: u64) -> u64 { self.next() % n }
}

const WORDS: &[&str] = &["rust", "index", "merge", "segment", "query", "shard", "node", "cluster", "vector", "nested", "answer", "question", "thread", "lock", "memory", "disk", "search", "score", "term", "field"];

fn words(n: u64, r: &mut Rng) -> String {
    (0..n).map(|_| WORDS[r.below(WORDS.len() as u64) as usize]).collect::<Vec<_>>().join(" ")
}

fn nested_doc(i: u64, r: &mut Rng) -> Value {
    let answers: Vec<Value> = (0..r.below(8)).map(|k| json!({
        "user": format!("u{}", r.below(50_000)),
        "date": 1_300_000_000_000i64 + (i * 60_000 + k * 1_000) as i64,
        "text": words(10 + r.below(30), r),
    })).collect();
    json!({
        "title": words(6 + r.below(6), r),
        "user": format!("u{}", r.below(50_000)),
        "tags": (0..1 + r.below(4)).map(|_| format!("tag{}", r.below(2_000))).collect::<Vec<_>>(),
        "date": 1_300_000_000_000i64 + (i * 60_000) as i64,
        "answers": answers,
    })
}

fn geonames_doc(i: u64, r: &mut Rng) -> Value {
    json!({
        "geonameid": i,
        "name": words(1 + r.below(3), r),
        "asciiname": words(1 + r.below(3), r),
        "feature_class": (["P", "A", "H", "S", "T"][r.below(5) as usize]),
        "feature_code": format!("F{}", r.below(600)),
        "country_code": format!("C{}", r.below(250)),
        "population": if r.below(4) == 0 { r.below(5_000_000) } else { 0 },
        "elevation": r.below(4_000),
        "timezone": format!("Zone/{}", r.below(400)),
        "location": [-180.0 + r.below(3_600_000) as f64 / 10_000.0, -90.0 + r.below(1_800_000) as f64 / 10_000.0],
    })
}

struct Snap { merges: u64, io: u64, input: u64, docs: u64, wall_ns: u64, stages: [u64; 7], sleeps: (u64, u64) }

fn snap() -> Snap {
    let t = &MERGE_TOTALS;
    Snap { merges: t.merges.load(Relaxed), io: t.io_limited.load(Relaxed), input: t.input_bytes.load(Relaxed), docs: t.docs.load(Relaxed), wall_ns: t.wall_ns.load(Relaxed),
        stages: std::array::from_fn(|k| t.stage_ms[k].load(Relaxed)), sleeps: merge_sleep_ns() }
}

fn run(label: &str, mapping: Value, n: u64, refresh_every: u64, make: fn(u64, &mut Rng) -> Value) {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("bench", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}, "mappings": mapping})).unwrap();
    let svc = node.get_index("bench").unwrap();
    let sh = svc.local_shards().remove(0);
    let before = snap();
    let es_time0 = sh.shard.counters.merge_time_nanos.load(Relaxed);
    let mut r = Rng(0x2545_f491_4f6c_dd1d);
    let t0 = Instant::now();
    let mut batch = Vec::with_capacity(5_000);
    for i in 0..n {
        batch.push(WriteReq::Index { id: None, source: make(i, &mut r).to_string(), routing: None, cond: Default::default() });
        if batch.len() == 5_000 || i + 1 == n {
            for x in svc.write(std::mem::take(&mut batch), false) { x.unwrap(); }
        }
        if (i + 1) % refresh_every == 0 { svc.refresh().unwrap(); }
    }
    svc.refresh().unwrap();
    let indexed = t0.elapsed();
    // Background merges settle: nothing running for two seconds.
    let mut quiet = Instant::now();
    let deadline = Instant::now() + Duration::from_secs(1800);
    while quiet.elapsed() < Duration::from_secs(2) && Instant::now() < deadline {
        if sh.shard.counters.merge_current.load(Relaxed) > 0 { quiet = Instant::now(); }
        std::thread::sleep(Duration::from_millis(50));
    }
    let settled = t0.elapsed();
    let after = snap();
    let es_ms = (sh.shard.counters.merge_time_nanos.load(Relaxed) - es_time0) / 1_000_000;
    let core = sh.shard.core.read();
    let (segs, store): (usize, u64) = (core.segments().len(), core.segments().iter().map(|s| s.segment_bytes()).sum());
    drop(core);
    let mb = |b: u64| b as f64 / (1 << 20) as f64;
    let stage = |k: usize| after.stages[k] - before.stages[k];
    eprintln!("{label}: {n} docs indexed in {:.1} s (merges settled at {:.1} s); {segs} segments, {:.1} MB store", indexed.as_secs_f64(), settled.as_secs_f64(), mb(store));
    eprintln!("  merges {} ({} IO-limited), {:.1} MB in ({:.2}x the store), {} docs; merge time (stats) {} ms, summed wall {} ms",
        after.merges - before.merges, after.io - before.io, mb(after.input - before.input), (after.input - before.input) as f64 / store.max(1) as f64,
        after.docs - before.docs, es_ms, (after.wall_ns - before.wall_ns) / 1_000_000);
    eprintln!("  stage ms (overlapping): norms {} postings {} store {} fast {} points {} extra {} finish {}; throttle sleep {} ms, pace sleep {} ms (node-wide)",
        stage(0), stage(1), stage(2), stage(3), stage(4), stage(5), stage(6), (after.sleeps.0 - before.sleeps.0) / 1_000_000, (after.sleeps.1 - before.sleeps.1) / 1_000_000);
}

#[test]
#[ignore]
fn merge_microbench() {
    let n: u64 = std::env::var("MERGE_BENCH_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(400_000);
    let every: u64 = std::env::var("MERGE_BENCH_REFRESH_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(20_000).max(1);
    let nested = json!({"properties": {"title": {"type": "text"}, "user": {"type": "keyword"}, "tags": {"type": "keyword"}, "date": {"type": "date"},
        "answers": {"type": "nested", "properties": {"user": {"type": "keyword"}, "date": {"type": "date"}, "text": {"type": "text"}}}}});
    let geonames = json!({"properties": {"geonameid": {"type": "long"}, "name": {"type": "text"}, "asciiname": {"type": "text"}, "feature_class": {"type": "keyword"},
        "feature_code": {"type": "keyword"}, "country_code": {"type": "keyword"}, "population": {"type": "long"}, "elevation": {"type": "integer"},
        "timezone": {"type": "keyword"}, "location": {"type": "geo_point"}}});
    for (limit, tag) in [(None, "adaptive IO limit"), (Some(0u64), "IO limit off")] {
        match limit { Some(l) => set_merge_io_limit(l), None => set_merge_io_adaptive() }
        run(&format!("nested-like, {tag}"), nested.clone(), n, every, nested_doc);
        run(&format!("geonames-like, {tag}"), geonames.clone(), n, every, geonames_doc);
    }
    set_merge_io_adaptive();
}
