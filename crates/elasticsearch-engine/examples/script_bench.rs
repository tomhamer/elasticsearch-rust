//! End-to-end script benchmark on a geonames-like index (Rally
//! `geonames` `painless_static` / `painless_dynamic`, plus script_fields,
//! a runtime-field sort and a script filter).
//!
//! ```text
//! SB_DOCS=3000000 SB_DIR=/tmp/sb cargo run --profile fast -p elasticsearch-engine --example script_bench
//! ```
//! The index is built once per `SB_DIR` (5 shards) and reused by later runs.
//! Prints, per query: median wall ms over `SB_RUNS` runs, CPU ms per run
//! (all threads) and ns of CPU per document.

use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::shard::WriteCond;
use serde_json::{json, Value};
use std::time::Instant;

fn cpu_ms() -> f64 {
    // utime + stime of the whole process, in clock ticks (Linux).
    let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let rest = s.rsplit_once(')').map(|x| x.1).unwrap_or("");
    let f: Vec<&str> = rest.split_whitespace().collect();
    let t = |i: usize| f.get(i).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0);
    (t(11) + t(12)) * 10.0
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn main() {
    let n: u64 = std::env::var("SB_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(3_000_000);
    let runs: usize = std::env::var("SB_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let dir = std::env::var("SB_DIR").unwrap_or_else(|_| format!("/tmp/script-bench-{n}"));
    let only = std::env::var("SB_ONLY").ok();
    std::fs::create_dir_all(&dir).unwrap();
    let node = Node::open(std::path::Path::new(&dir)).unwrap();
    let svc = match node.get_index("geonames") {
        Some(s) => s,
        None => {
            let t0 = Instant::now();
            let svc = node
                .create_index(
                    "geonames",
                    &json!({
                        "settings": {"number_of_shards": 5, "number_of_replicas": 0, "refresh_interval": "-1"},
                        "mappings": {"properties": {
                            "name": {"type": "keyword"},
                            "feature_class": {"type": "keyword"},
                            "population": {"type": "long"},
                            "elevation": {"type": "integer"},
                            "location": {"type": "geo_point"}
                        }}
                    }),
                )
                .unwrap();
            let mut rng = Rng(0x9E3779B97F4A7C15);
            let classes = ["A", "H", "L", "P", "R", "S", "T", "U", "V"];
            let mut i = 0u64;
            while i < n {
                let end = (i + 20_000).min(n);
                let reqs: Vec<WriteReq> = (i..end)
                    .map(|d| {
                        // ~70% of geonames entries have population 0.
                        let pop = if rng.f() < 0.7 { 0 } else { (rng.f() * rng.f() * 1e7) as i64 };
                        let lat = rng.f() * 180.0 - 90.0;
                        let lon = rng.f() * 360.0 - 180.0;
                        let src = json!({
                            "name": format!("place{}", d % 200_000),
                            "feature_class": classes[(rng.next() % 9) as usize],
                            "population": pop,
                            "elevation": (rng.next() % 3000) as i64,
                            "location": [lon, lat]
                        });
                        WriteReq::Index { id: Some(d.to_string()), source: src.to_string(), routing: None, cond: WriteCond::default() }
                    })
                    .collect();
                for r in svc.write(reqs, false) {
                    r.unwrap();
                }
                i = end;
            }
            svc.refresh().unwrap();
            svc.flush().unwrap();
            eprintln!("indexed {n} docs in {:.1}s", t0.elapsed().as_secs_f64());
            svc
        }
    };
    svc.refresh().unwrap();

    let fs = |src: &str| json!({"query": {"function_score": {"query": {"match_all": {}}, "functions": [{"script_score": {"script": {"source": src, "lang": "painless"}}}]}}});
    let queries: Vec<(&str, Value)> = vec![
        ("painless_static", fs("Math.abs(Math.log(Math.abs((int)((List)doc.population).get(0)) + 1) + (double)(doc.location.lon) * (double)(doc.location.lat))/_score")),
        ("painless_dynamic", fs("Math.abs(Math.log(Math.abs(doc['population'].value) + 1) + doc['location'].lon * doc['location'].lat)/_score")),
        ("script_score_query", json!({"query": {"script_score": {"query": {"match_all": {}}, "script": {"source": "Math.log(2 + doc['population'].value) * params.f", "params": {"f": 1.5}}}}})),
        ("script_filter", json!({"query": {"bool": {"filter": {"script": {"script": {"source": "doc['population'].value > params.min", "params": {"min": 1000000}}}}}}})),
        (
            "runtime_sort",
            json!({"runtime_mappings": {"pop2": {"type": "long", "script": "emit(doc['population'].value * 2 + doc['elevation'].value)"}}, "query": {"match_all": {}}, "sort": [{"pop2": "desc"}]}),
        ),
        ("script_sort", json!({"query": {"match_all": {}}, "sort": [{"_script": {"type": "number", "order": "desc", "script": {"source": "doc['population'].value + doc['location'].lat"}}}]})),
        (
            "runtime_aggs",
            json!({"size": 0, "runtime_mappings": {"pop2": {"type": "long", "script": "emit(doc['population'].value * 2)"}, "fc": {"type": "keyword", "script": "emit(doc['feature_class'].value)"}}, "aggs": {"avg_pop2": {"avg": {"field": "pop2"}}, "by_fc": {"terms": {"field": "fc"}}}}),
        ),
        ("runtime_range", json!({"runtime_mappings": {"pop2": {"type": "long", "script": "emit(doc['population'].value * 2)"}}, "query": {"range": {"pop2": {"gte": 1000000}}}})),
        ("script_agg", json!({"size": 0, "aggs": {"s": {"sum": {"script": {"source": "doc['population'].value * params.f", "params": {"f": 0.5}}}}}})),
        (
            "script_fields_1000",
            json!({"size": 1000, "query": {"match_all": {}}, "script_fields": {"p": {"script": "doc['population'].value * 2"}, "g": {"script": "doc['location'].lat + doc['location'].lon"}, "n": {"script": "doc['name'].value"}}}),
        ),
    ];
    for (name, body) in &queries {
        if only.as_deref().is_some_and(|o| !o.split(',').any(|x| x == *name)) {
            continue;
        }
        let first = elasticsearch_engine::ml::search(&node, "geonames", body);
        match &first {
            Ok(v) => {
                let total = v.pointer("/hits/total/value").cloned().unwrap_or(Value::Null);
                let top = v.pointer("/hits/hits/0").map(|h| json!([h.get("_id"), h.get("_score"), h.get("sort")])).or_else(|| v.get("aggregations").cloned()).unwrap_or(Value::Null);
                eprintln!("{name}: total={total} top={top}");
            }
            Err(e) => {
                eprintln!("{name}: ERROR {e:?}");
                continue;
            }
        }
        let mut walls = Vec::new();
        let mut cpus = Vec::new();
        for _ in 0..runs {
            let c0 = cpu_ms();
            let t0 = Instant::now();
            let r = elasticsearch_engine::ml::search(&node, "geonames", body).unwrap();
            std::hint::black_box(r);
            walls.push(t0.elapsed().as_secs_f64() * 1000.0);
            cpus.push(cpu_ms() - c0);
        }
        walls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        cpus.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (w, c) = (walls[runs / 2], cpus[runs / 2]);
        println!("{name:20} wall_ms={w:9.1} cpu_ms={c:9.1} cpu_ns_per_doc={:7.1} (min wall {:.1})", c * 1e6 / n as f64, walls[0]);
    }
}
