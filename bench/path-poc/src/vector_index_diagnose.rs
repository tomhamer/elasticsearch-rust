//! Read-only library profiling of full index state across persisted shard copies.
//! No optimizer, commit, writes or fixture recovery. Not an ES API benchmark.
use rustic_core::{Index,index::KnnRequest,query::Query,vector::store::PlannerConfig};
use serde_json::json;
use std::{path::Path,time::Instant};
fn memory()->String {std::fs::read_to_string("/proc/self/status").unwrap_or_default().lines().filter(|s|s.starts_with("VmRSS:")||s.starts_with("VmHWM:")||s.starts_with("RssAnon:")||s.starts_with("RssFile:")).collect::<Vec<_>>().join("; ")}
fn main()->Result<(),Box<dyn std::error::Error>> {
 let a:Vec<_>=std::env::args().collect();if a.len()!=3{return Err("usage: rustic-vector-index-diagnose core-directories.json queries.json".into())}
 let paths:Vec<String>=serde_json::from_slice(&std::fs::read(&a[1])?)?;
 let queries:Vec<Vec<f32>>=serde_json::from_slice(&std::fs::read(&a[2])?)?;
 let mut indexes=Vec::new();
 for path in paths {let t=Instant::now();let index=Index::open(Path::new(&path))?;println!("{}",json!({"event":"open","path":path,"elapsed_ms":t.elapsed().as_secs_f64()*1000.,"memory":memory()}));indexes.push(index);}
 for round in 0..3 {for (copy,index) in indexes.iter().enumerate() {
  let field=index.field_id("v").ok_or("no vector field")?;let tag=index.field_id("tag").ok_or("no tag")?;
  for label in ["all","ten","half"] {for(qi,q)in queries.iter().enumerate(){
   let mut req=KnnRequest::new(field,q.clone(),10);req.num_candidates=500;req.filter=match label{"ten"=>Some(Query::term(tag,"t0")),"half"=>Some(Query::term_set(tag,&["t0","t1","t2","t3","t4"])),_=>None};
   let t=Instant::now();let(hits,stats)=index.knn_search_with(&req,&PlannerConfig::default()).unwrap();
   println!("{}",json!({"event":"query","round":round,"copy":copy,"filter":label,"query":qi,"latency_ms":t.elapsed().as_secs_f64()*1000.,"hits":hits.len(),"graph_segments":stats.graph_segments,"exact_segments":stats.exact_segments,"fallbacks":stats.fallbacks,"vector_ops":stats.vector_ops,"memory":memory()}));
  }}
 }}
 Ok(())
}
