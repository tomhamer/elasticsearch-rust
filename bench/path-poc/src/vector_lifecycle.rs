//! EC2-only library POC on a disposable COPY of an existing persisted shard.
//! Measures optimizer backlog, query planner work and process memory together.
use rustic_core::{Index, index::KnnRequest, query::Query, vector::store::PlannerConfig};
use serde_json::json;
use std::{path::Path, sync::Arc, time::{Duration,Instant}};
fn memory() -> String { std::fs::read_to_string("/proc/self/status").unwrap_or_default().lines().filter(|l|l.starts_with("VmRSS:")||l.starts_with("VmHWM:")).collect::<Vec<_>>().join("; ") }
fn batch(index:&Index, queries:&[Vec<f32>], phase:&str, started:Instant) {
    let field=index.field_id("v").expect("vector field");let tag=index.field_id("tag").expect("tag field");
    let segments=index.vector_store().segments(field);
    println!("{}",json!({"event":"progress","phase":phase,"elapsed_s":started.elapsed().as_secs_f64(),"segments":segments.len(),"persisted_rows":segments.iter().map(|s|s.len() as u64).sum::<u64>(),"indexed_rows":segments.iter().filter(|s|s.has_graph()).map(|s|s.len() as u64).sum::<u64>(),"graphless_rows":segments.iter().filter(|s|!s.has_graph()).map(|s|s.len() as u64).sum::<u64>(),"memory":memory()}));
    for label in ["all","ten","half"] {for (qi,q) in queries.iter().enumerate() {
        let mut req=KnnRequest::new(field,q.clone(),10);req.num_candidates=500;
        req.filter=match label {"ten"=>Some(Query::term(tag,"t0")),"half"=>Some(Query::term_set(tag,&["t0","t1","t2","t3","t4"])),_=>None};
        let t=Instant::now();let (hits,stats)=index.knn_search_with(&req,&PlannerConfig::default()).unwrap();
        println!("{}",json!({"event":"query","phase":phase,"elapsed_s":started.elapsed().as_secs_f64(),"filter":label,"query":qi,"latency_ms":t.elapsed().as_secs_f64()*1000.,"exact_segments":stats.exact_segments,"graph_segments":stats.graph_segments,"fallbacks":stats.fallbacks,"vector_ops":stats.vector_ops,"hits":hits.iter().map(|h|json!({"leaf":h.leaf,"doc":h.doc,"score":h.score})).collect::<Vec<_>>(),"memory":memory()}));
    }}
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let args:Vec<_>=std::env::args().collect();if args.len()!=3{return Err("usage: rustic-vector-lifecycle DISPOSABLE_COPY queries.json".into())}
    let root=Path::new(&args[1]);if !root.join(".allow-lifecycle-poc").is_file(){return Err("requires disposable-copy marker .allow-lifecycle-poc; never run on production fixture".into())}
    let queries:Vec<Vec<f32>>=serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let begun=Instant::now();let index=Arc::new(Index::open(root)?);
    println!("{}",json!({"event":"open","elapsed_s":begun.elapsed().as_secs_f64(),"memory":memory()}));
    for _ in 0..3 {batch(&index,&queries,"before-indexing",begun)}
    // Same detached library optimizer called by the production maintenance path.
    let job=index.vector_optimizer();let worker=std::thread::spawn(move||job.run());
    while !worker.is_finished(){batch(&index,&queries,"during-indexing",begun);std::thread::sleep(Duration::from_secs(5));}
    let result=worker.join().map_err(|_|"optimizer thread panicked")??;
    println!("{}",json!({"event":"optimizer_pass_complete","elapsed_s":begun.elapsed().as_secs_f64(),"report":format!("{result:?}"),"memory":memory()}));
    // One optimizer pass may leave additional compaction work: report actual
    // coverage rather than calling this fully indexed steady state.
    for _ in 0..5 {batch(&index,&queries,"after-one-optimizer-pass",begun)}
    Ok(())
}
