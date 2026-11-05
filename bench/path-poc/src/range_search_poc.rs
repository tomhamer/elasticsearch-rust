//! EC2-only scaling check through the real Searcher and BM25 boolean scorers.
use rustic_core::{codec::directory::RamDirectory,fieldtypes::{filter::FilterQuery,range::{self,Relation},range_query::RangeRelationQuery},query::{BoolBuilder,LeafReader,Query,Searcher,SearchOptions,TopDocs,TrackTotalHits},segment::SegmentWriter,types::{Document,Field,FieldType}};
use std::time::Instant;
fn range_query(fast:bool,small:bool)->Query {
 let bounds=if small {(range::key_i64(-20),range::key_i64(-10))} else {(range::key_i64(-800),range::key_i64(800))};
 if fast {RangeRelationQuery {field:0,bounds,relation:Relation::Intersects}.into_query()}
 else {FilterQuery::constant("range",Some(Query::Exists {field:0}),move |l,d|l.keyword_values(0,d).iter().filter_map(|s|range::decode(s)).any(|v|range::matches(v,bounds,Relation::Intersects))).into_query()}
}
fn make(fast:bool,small:bool,mode:&str)->Query {
 // Match the engine range parser: preserve its constant-score wrapper.
 let r=Query::ConstScore {query:Box::new(range_query(fast,small)),score:1.0, point_set: false};
 match mode {
  "and-rare"=>BoolBuilder::default().must(r).must(Query::term(1,"rare")).build(),
  "and-broad"=>BoolBuilder::default().must(r).must(Query::term(2,"broad")).build(),
  "or-rare"=>BoolBuilder::default().should(r).should(Query::term(1,"rare")).build(),
  _=>r,
 }
}
fn signature(t:&TopDocs)->(u64,String,Vec<(usize,u32,u32)>) {
 (t.total_hits,format!("{:?}",t.total_hits_relation),t.hits.iter().map(|h|(h.leaf,h.doc,h.score.to_bits())).collect())
}
fn main() {
 for (n,clustered) in [(1_000_000u32,false),(8_000_000,false),(8_000_000,true)] {
  let mut schema=vec![Field::new(0,"range",FieldType::Keyword),Field::new(1,"rare",FieldType::Keyword),Field::new(2,"broad",FieldType::Keyword)];
  for f in &mut schema {f.stored=false;f.norms=false;}schema[0].indexed=false;
  let mut writer=SegmentWriter::new(0);
  for d in 0..n {
   let mut doc=Document::new();let lo=if clustered {if d<n/2 {100} else {-15}} else {(d.wrapping_mul(7919)%2000) as i64-1000};
   if d%97!=0 {doc.add_keyword(0,&range::encode(range::key_i64(lo),range::key_i64(lo+20)));}
   if d%1000==0 {doc.add_keyword(1,"rare");}
   if d%3!=0 {doc.add_keyword(2,"broad");}
   writer.add_document(&doc,&schema);
  }
  let dir=RamDirectory::new();let segment=writer.write_to(&dir).unwrap();
  let searcher=Searcher::new(vec![LeafReader::Segment(&segment)]);
  for small in [false,true] {for mode in ["range","and-rare","and-broad","or-rare"] {
   for exact in [false,true] {
    let opts=SearchOptions {track_total_hits:if exact {TrackTotalHits::True} else {TrackTotalHits::UpTo(10000)},..Default::default()};
    // Initialize metadata separately so warm execution is reported explicitly.
    let candidate=make(true,small,mode);let oracle=make(false,small,mode);
    let start=Instant::now();let warm=searcher.search(&candidate,&opts);let warm_ms=start.elapsed().as_secs_f64()*1000.0;
    let start=Instant::now();let base=searcher.search(&oracle,&opts);let base_ms=start.elapsed().as_secs_f64()*1000.0;
    let start=Instant::now();let fast=searcher.search(&candidate,&opts);let fast_ms=start.elapsed().as_secs_f64()*1000.0;
    assert!(signature(&warm)==signature(&base) && signature(&fast)==signature(&base),"Mismatch {n}/{small}/{mode}/{exact}");
    println!("{}",serde_json::json!({"documents":n,"clustered":clustered,"small":small,"mode":mode,"exact_count":exact,"baseline_ms":base_ms,"candidate_ms":fast_ms,"metadata_warmup_ms":warm_ms,"speedup":base_ms/fast_ms,"exact_same_hits_scores_counts":true}));
   }
  }}
 }
}
