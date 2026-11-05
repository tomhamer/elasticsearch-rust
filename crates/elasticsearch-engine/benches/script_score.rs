//! Per-doc cost of a typical `script_score` script:
//! `doc['price'].value * params.factor + _score` over a sealed segment and
//! the memtable. Criterion reports throughput in docs/s (target: < ~50 ns/doc).

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::script::{self, ScriptContext};
use elasticsearch_engine::shard::WriteCond;
use serde_json::json;
use std::hint::black_box;

const N: u32 = 100_000;

fn bench(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    let svc = node.create_index("bench", &json!({"mappings": {"properties": {"price": {"type": "double"}, "qty": {"type": "long"}}}})).unwrap();
    let reqs: Vec<WriteReq> = (0..N)
        .map(|i| WriteReq::Index { id: Some(i.to_string()), source: json!({"price": (i % 1000) as f64 * 0.5 + 1.0, "qty": i % 7}).to_string(), routing: None, cond: WriteCond::default() })
        .collect();
    for chunk in reqs.chunks(10_000) {
        let v: Vec<WriteReq> = chunk.iter().map(|r| match r {
            WriteReq::Index { id, source, .. } => WriteReq::Index { id: id.clone(), source: source.clone(), routing: None, cond: WriteCond::default() },
            WriteReq::Delete { .. } => unreachable!(),
        }).collect();
        for r in svc.write(v, false) {
            r.unwrap();
        }
    }
    let lookup = svc.lookup();
    let cs = script::compile("doc['price'].value * params.factor + _score", &json!({"factor": 1.5}), ScriptContext::Score).unwrap();

    let mut g = c.benchmark_group("script_score");
    g.throughput(Throughput::Elements(N as u64));
    for (label, seal) in [("memtable", false), ("segment", true)] {
        if seal {
            svc.shard(0).unwrap().shard.flush().unwrap();
        }
        let shard = svc.shard(0).unwrap();
        let core = shard.shard.core.read();
        let leaves = core.leaves();
        let leaf = leaves.iter().max_by_key(|l| l.num_docs()).copied().unwrap();
        let mut exec = cs.executor(Some(lookup.clone()), None);
        // SAFETY: `leaf` outlives `exec` within this scope.
        unsafe { exec.bind_leaf(&leaf) };
        let n = leaf.num_docs();
        g.bench_function(format!("price_times_factor_plus_score/{label}"), |b| {
            b.iter(|| {
                let mut acc = 0.0f64;
                for d in 0..n {
                    acc += script::execute_score(&mut exec, d, 1.0).unwrap();
                }
                black_box(acc)
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
