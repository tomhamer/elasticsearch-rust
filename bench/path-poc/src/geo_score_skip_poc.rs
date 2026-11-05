//! Exact-score spatial pruning experiment, independent of the server.
use rustic_core::geo::{GeoIndex,GeoPointEntry,encode_lat,encode_lon,decode_lat,decode_lon,sloppy_haversin_meters};
use rustic_core::query::DecayKind;
use std::{cell::Cell,cmp::Ordering,collections::BinaryHeap,time::Instant};
#[derive(Clone,Copy,Debug,PartialEq)]
struct Hit{doc:u32,score:f32}
impl Eq for Hit{}
impl PartialOrd for Hit{fn partial_cmp(&self,b:&Self)->Option<Ordering>{Some(self.cmp(b))}}
impl Ord for Hit{fn cmp(&self,b:&Self)->Ordering{b.score.total_cmp(&self.score).then(self.doc.cmp(&b.doc))}}
fn offer(heap:&mut BinaryHeap<Hit>,h:Hit,k:usize){
 if heap.len()<k{heap.push(h)}else if let Some(w)=heap.peek(){if h.score>w.score || (h.score==w.score&&h.doc<w.doc){*heap.peek_mut().unwrap()=h;}}
}
fn sorted(heap:BinaryHeap<Hit>)->Vec<Hit>{let mut v=heap.into_vec();v.sort_by(|a,b|b.score.total_cmp(&a.score).then(a.doc.cmp(&b.doc)));v}
fn main(){
 let points:Vec<_>=(0..1_000_000u64).map(|i|{
  let a=i.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
  let b=a.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
  GeoPointEntry{doc:i as u32,lat:encode_lat((a>>11)as f64/(1u64<<53)as f64*180.0-90.0),lon:encode_lon((b>>11)as f64/(1u64<<53)as f64*360.0-180.0)}
 }).collect();
 let index=GeoIndex::build(points.clone());assert!(index.single_valued());
 for (lat,lon) in [(52.37,4.8951),(89.9,179.9),(-89.9,-179.9),(0.0,0.0)]{
 for (scale,offset,k) in [(500_000.0,0.0,10usize),(10_000.0,0.0,100),(20_000_000.0,1_000_000.0,10)]{
  let score=|d:f64|DecayKind::Gauss.eval((d-offset).max(0.0),scale,0.1)as f32;
  let point_score=|p:&GeoPointEntry|score(sloppy_haversin_meters(lat,lon,decode_lat(p.lat),decode_lon(p.lon)));
  // Deterministic deleted documents. Missing/multivalued fields deliberately
  // remain outside this initial proof-of-concept and must fall back in a server.
  let live=|doc:u32|doc%7!=0;
  let start=Instant::now();let mut heap=BinaryHeap::new();
  for p in &points{if live(p.doc){offer(&mut heap,Hit{doc:p.doc,score:point_score(p)},k)}}
  let expected=sorted(heap);let full=start.elapsed().as_secs_f64();
  let start=Instant::now();let mut heap=BinaryHeap::new();let threshold=Cell::new(f32::NEG_INFINITY);let mut scored=0;
  index.visit_by_distance(lat,lon,|bound,points|{
   if score(bound)<threshold.get(){return false;}
   for p in points {
    if live(p.doc){scored+=1;offer(&mut heap,Hit{doc:p.doc,score:point_score(p)},k);if heap.len()==k{threshold.set(heap.peek().unwrap().score)}}
   }
   true
  });
  let found=sorted(heap);let pruned=start.elapsed().as_secs_f64();assert_eq!(expected,found,"origin={lat},{lon} scale={scale} offset={offset} k={k}");
  println!("{}",serde_json::json!({"lat":lat,"lon":lon,"scale":scale,"offset":offset,"k":k,"full_seconds":full,"pruned_seconds":pruned,"speedup":full/pruned,"scored":scored,"exact_hits_and_scores":true}));
 }}
}
