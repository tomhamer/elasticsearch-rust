//! Immutable-snapshot filter-reuse POC. No production cache or invalidation claim.
use rustic_core::{Index,index::KnnRequest,query::{Query,LeafReader,SearchContext,Bm25Params},vector::store::{DocFilter,OrdFilter,PointFilter,PlannerConfig}};
use serde_json::json;
use std::{path::Path,time::Instant,collections::HashMap};
fn prepare(index:&Index,field:u16,filter:&Query)->OrdFilter {
 let leaves=index.leaves();let ctx=SearchContext::for_query(filter,&leaves,Bm25Params::default());let mut df=DocFilter::new();
 for leaf in &leaves {
  let mut words=filter.filter_bits(leaf,&ctx);
  if let Some(d)=leaf.deleted_bits(){for(w,d)in words.iter_mut().zip(d.words()){*w &= !d;}}
  match leaf{LeafReader::Segment(s)=>df.add_segment(s.id,words),_=>panic!("POC requires persisted immutable leaves")}
 }
 index.vector_store().ord_filter(field,df,|sid|leaves.iter().find_map(|l|match l{LeafReader::Segment(s)if s.id==sid=>Some((s.point_map().column(field)?,s.point_identity_base(field))),_=>None}))
}
fn main()->Result<(),Box<dyn std::error::Error>> {
 let a:Vec<_>=std::env::args().collect();if a.len()!=3{return Err("usage: rustic-vector-filter-poc CORE queries.json".into())}
 let index=Index::open(Path::new(&a[1]))?;let queries:Vec<Vec<f32>>=serde_json::from_slice(&std::fs::read(&a[2])?)?;
 let field=index.field_id("v").ok_or("no vector field")?;let tag=index.field_id("tag").ok_or("no tag")?;
 let leaves=index.leaves();let leaf_of:HashMap<u32,usize>=leaves.iter().enumerate().filter_map(|(i,l)|match l{LeafReader::Segment(s)=>Some((s.id,i)),_=>None}).collect();
 for (label,filter)in [("ten",Query::term(tag,"t0")),("half",Query::term_set(tag,&["t0","t1","t2","t3","t4"]))] {
  let t=Instant::now();let ords=prepare(&index,field,&filter);println!("{}",json!({"event":"prepare","filter":label,"elapsed_ms":t.elapsed().as_secs_f64()*1000.,"matches":ords.len()}));
  for round in 0..6 {for(qi,q)in queries.iter().enumerate(){
   let mut req=KnnRequest::new(field,q.clone(),10);req.num_candidates=500;req.filter=Some(filter.clone());
   let baseline=||{let t=Instant::now();let(hits,_)=index.knn_search_with(&req,&PlannerConfig::default()).unwrap();(t.elapsed().as_secs_f64()*1000.,hits)};
   let cached=||{let t=Instant::now();let(hits,stats)=index.vector_store().search_filtered(field,q,10,500,Some(PointFilter::Ords(&ords)),&PlannerConfig::default());(t.elapsed().as_secs_f64()*1000.,hits,stats)};
   let((base_ms,base),(prepared_ms,prepared,stats))=if round%2==0{(baseline(),cached())}else{let c=cached();(baseline(),c)};
   let mut got:Vec<_>=prepared.iter().map(|h|(leaf_of[&h.segment],h.doc,h.score)).collect();got.sort_by(|a,b|b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
   assert_eq!(base.len(),got.len());for(a,b)in base.iter().zip(&got){assert_eq!((a.leaf,a.doc),(b.0,b.1));assert!((a.score-b.2).abs()<2e-5);}
   println!("{}",json!({"event":"query","filter":label,"round":round,"query":qi,"baseline_ms":base_ms,"prepared_ms":prepared_ms,"ids_and_scores_match":true,"exact_segments":stats.exact_segments,"graph_segments":stats.graph_segments,"vector_ops":stats.vector_ops}));
  }}
 }
 Ok(())
}
