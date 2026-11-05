//! Isolated EC2 component comparison, not Rally or an ES9 speedup claim.
use rustic_core::{codec::directory::RamDirectory, fieldtypes::{filter::FilterQuery, range::{self, Relation}, range_query::RangeRelationQuery}, query::{ConstScorer, EmptyDocSet, IntersectionN, LeafReader, Query, Scorer, SearchContext, Union, VecDocSet, TERMINATED}, segment::SegmentWriter, types::{Document, Field, FieldType}};
use std::time::Instant;
fn query(bounds: (u128,u128), relation: Relation, optimized: bool) -> Query {
    if optimized { RangeRelationQuery { field:0, bounds, relation }.into_query() }
    else { FilterQuery::constant("range", Some(Query::Exists {field:0}), move |l,d| l.keyword_values(0,d).iter().filter_map(|s|range::decode(s)).any(|v|range::matches(v,bounds,relation))).into_query() }
}
fn run(leaf: &LeafReader<'_>, bounds:(u128,u128), relation:Relation, optimized:bool, mode:&str, step:u32) -> Vec<(u32,u32)> {
    let q=query(bounds,relation,optimized);
    let ctx=SearchContext::default();
    let mut scorer=q.scorer(leaf,&ctx).unwrap_or_else(||Box::new(ConstScorer::new(EmptyDocSet,1.0)));
    if mode!="range" {
        let terms=(0..leaf.num_docs()).step_by(step as usize).filter(|&d|!leaf.is_deleted(d)).collect();
        let term:Box<dyn Scorer>=Box::new(ConstScorer::new(VecDocSet::new(terms),1.0));
        scorer=if mode=="and" {Box::new(IntersectionN::new(vec![term,scorer]))} else {Box::new(Union::new(vec![term,scorer]))};
    }
    let mut out=Vec::new();
    loop {let d=scorer.advance(); if d==TERMINATED {break} if !leaf.is_deleted(d) {out.push((d,scorer.score().to_bits()));}}
    out
}
fn main() {
    let n:u32=std::env::args().nth(1).unwrap_or("1000000".into()).parse().unwrap();
    // Real core writers/readers with repeated and unique encoded ranges,
    // missing values, deletes, and separate multi-valued fallback fixtures.
    for (layout,multi,unique) in [("repeated",false,false),("unique",false,true),("multi",true,false)] {
        let docs=if multi {n.min(20000)} else {n};
        let mut writer=SegmentWriter::new(0);
        let mut field=Field::new(0,"range",FieldType::Keyword);field.indexed=false;field.stored=false;
        let schema=vec![field];
        for d in 0..docs {
            let mut doc=Document::new();
            if d%97!=0 {
                let low=if unique {d as i64 - docs as i64/2} else {(d.wrapping_mul(7919)%2000) as i64-1000};
                doc.add_keyword(0,&range::encode(range::key_i64(low),range::key_i64(low+20)));
                if multi && d%3==0 {doc.add_keyword(0,&range::encode(range::key_i64(-5),range::key_i64(5)));}
            }
            writer.add_document(&doc,&schema);
        }
        for d in (0..docs).step_by(101) {writer.delete_doc(d);}
        if multi { check(&LeafReader::Memtable(&writer),layout,"memtable",docs); }
        let dir=RamDirectory::new();let segment=writer.write_to(&dir).unwrap();
        check(&LeafReader::Segment(&segment),layout,"sealed",docs);
    }
}
fn check(leaf:&LeafReader<'_>,layout:&str,leaf_name:&str,docs:u32) {
            for (range_name,lo,hi) in [("large",-800,800),("small",-20,-10),("empty",2000000,3000000)] {
                let bounds=(range::key_i64(lo),range::key_i64(hi));
                for relation in [Relation::Intersects,Relation::Contains,Relation::Within,Relation::Disjoint] {
                    for (mode,step) in [("range",1),("and",1000),("and",2),("or",1000),("or",2)] {
                        let mut times=[0.0;2];let mut expected:Option<Vec<(u32,u32)>>=None;
                        for optimized in [false,true,true,false] {
                            let start=Instant::now();let got=run(&leaf,bounds,relation,optimized,mode,step);
                            times[usize::from(optimized)]+=start.elapsed().as_secs_f64();
                            if let Some(ref expected)=expected {assert!(&got==expected,"{layout}/{leaf_name}/{range_name}/{relation:?}/{mode}/{step}: candidate={} baseline={} first_diff={:?}",got.len(),expected.len(),got.iter().zip(expected).position(|(a,b)|a!=b));} else {expected=Some(got);}
                        }
                        println!("{}",serde_json::json!({"layout":layout,"leaf":leaf_name,"documents":docs,"range":range_name,"relation":format!("{relation:?}"),"mode":mode,"term_step":step,"matches":expected.unwrap().len(),"baseline_ms":times[0]*500.0,"candidate_ms":times[1]*500.0,"speedup":times[0]/times[1],"exact_docs_scores":true}));
                    }
                }
            }
}
