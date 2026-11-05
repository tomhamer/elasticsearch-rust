//! Ordinal collection must preserve nested series counts, sums and size/order.
use elasticsearch_engine::{index::WriteReq, node::Node, search::{self, ExecOpts}};
use serde_json::{json, Value};
use std::collections::HashMap;

#[test]
fn nested_series_match_before_flush_after_flush_and_restart() {
    let dir = tempfile::tempdir().unwrap();
    let mut node = Node::open(dir.path()).unwrap();
    let definition=json!({"settings":{"index.mode":"time_series","index.routing_path":["host"],
        "index.time_series.start_time":"2024-01-01T00:00:00Z","index.time_series.end_time":"2025-01-01T00:00:00Z",
        "number_of_shards":1,"number_of_replicas":0},"mappings":{"properties":{
        "host":{"type":"keyword","time_series_dimension":true},"@timestamp":{"type":"date"},
        "usage":{"type":"long","time_series_metric":"gauge"}}}});
    let svc=node.create_index("metrics",&definition).unwrap();
    for hour in 0..3 {
        let docs=(0..40).map(|n|WriteReq::Index{id:None,routing:None,cond:Default::default(),source:json!({
            "host":if n%2==0{"alpha"}else{"beta"},"@timestamp":format!("2024-06-01T{hour:02}:00:{n:02}Z"),"usage":n+1}).to_string()}).collect();
        for result in svc.write(docs,true){result.unwrap();}
        if hour==0 {svc.flush().unwrap();}
    }
    svc.refresh().unwrap();
    drop(svc);
    let query=json!({"size":0,"aggs":{"hours":{"date_histogram":{"field":"@timestamp","fixed_interval":"1h"},"aggs":{
        "series":{"time_series":{"keyed":false},"aggs":{"usage":{"sum":{"field":"usage"}}}},
        "minimum":{"min_bucket":{"buckets_path":"series>usage"}}}}}});
    let mut baseline=None;
    for phase in 0..3 {
        if phase==1 {node.get_index("metrics").unwrap().flush().unwrap();}
        if phase==2 {drop(node); node=Node::open(dir.path()).unwrap();}
        let request=search::parse_request(&query,&HashMap::new()).unwrap();
        let targets=node.resolve("metrics",Default::default()).unwrap();
        let options=ExecOpts{terms_lookup:None,cursors:None,max_buckets:65536,scroll_id:None,pit_id:None};
        let result=search::execute(&targets,&request,&options).unwrap();
        let aggs=serde_json::from_str::<Value>(&result.json).unwrap()["aggregations"].clone();
        let hours=aggs["hours"]["buckets"].as_array().unwrap();assert_eq!(hours.len(),3);
        for h in hours {
            assert_eq!(h["doc_count"],40);
            let series=h["series"]["buckets"].as_array().unwrap();assert_eq!(series.len(),2);
            for s in series {assert_eq!(s["doc_count"],20);let expected=if s["key"]["host"]=="alpha"{400.0}else{420.0};assert_eq!(s["usage"]["value"].as_f64(),Some(expected));}
            assert_eq!(h["minimum"]["value"].as_f64(),Some(400.0));
        }
        if let Some(ref original)=baseline {assert_eq!(&aggs,original);}else{baseline=Some(aggs);}
    }
}
