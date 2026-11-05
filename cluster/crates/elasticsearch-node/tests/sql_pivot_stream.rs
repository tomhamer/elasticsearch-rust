//! SQL PIVOT must fold all pages even when no shard copy is local.
mod common;
use common::start;
use reqwest::Method;
use serde_json::{json,Value};
#[tokio::test(flavor="multi_thread",worker_threads=4)]
async fn pivot_remote_batches_match_local() {
    let cl = start(&[]).await;
    let ports = cl.p;
    cl.exclude_master().await;
    let (st,b)=cl.call(0,Method::PUT,"/pivotremote",Some(json!({"settings":{"number_of_shards":2,"number_of_replicas":0},"mappings":{"properties":{"g":{"type":"keyword"},"c":{"type":"keyword"},"v":{"type":"integer"}}}}))).await;
    assert_eq!(st,200,"{b}");
    let (_,h)=cl.call(0,Method::GET,"/_cluster/health/pivotremote?wait_for_status=green&timeout=60s",None).await;
    assert_eq!(h["status"],"green","{h}");
    assert!(cl.primary_off_master("pivotremote").await);
    for offset in (0..5000).step_by(500) {
        let mut bulk=String::new();
        for i in offset..offset+500 {
            bulk.push_str(&json!({"index":{"_id":i.to_string()}}).to_string());bulk.push('\n');
            bulk.push_str(&json!({"g":format!("g{}",i%5),"c":if i%2==0 {"A"}else{"B"},"v":i}).to_string());bulk.push('\n');
        }
        let b:Value=cl.c.post(format!("http://127.0.0.1:{}/pivotremote/_bulk?refresh=true",ports.http[cl.other])).header("content-type","application/x-ndjson").body(bulk).send().await.unwrap().json().await.unwrap();
        assert_eq!(b["errors"],false,"{b}");
    }
    for agg in ["AVG(v)","COUNT(v)","SUM(v)","MIN(v)","MAX(v)"] {
        let query=format!("SELECT * FROM (SELECT g,c,v FROM pivotremote) PIVOT ({agg} FOR c IN ('A','B','Z')) ORDER BY g");
        let mut outputs=Vec::new();
        for node in [cl.other,cl.master] {
            let (st,mut b)=cl.call(node,Method::POST,"/_sql",Some(json!({"query":query,"fetch_size":2}))).await;
            assert_eq!(st,200,"{agg}: {b}");
            let mut rows=Vec::new();
            loop {
                rows.extend(b["rows"].as_array().unwrap().iter().cloned());
                let Some(cursor)=b.get("cursor").cloned() else {break};
                let (st,next)=cl.call(node,Method::POST,"/_sql",Some(json!({"cursor":cursor}))).await;
                assert_eq!(st,200,"{next}");b=next;
            }
            assert_eq!(rows.len(),5);
            if agg=="COUNT(v)" { for (i,row) in rows.iter().enumerate() { assert_eq!(row,&json!([format!("g{i}"),500,500,null])); } }
            outputs.push(rows);
        }
        assert_eq!(outputs[0],outputs[1],"{agg}");
    }
}
