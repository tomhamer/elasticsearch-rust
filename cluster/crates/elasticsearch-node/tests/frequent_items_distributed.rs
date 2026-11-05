//! Real node fanout: retain transactions until the final item-set reduce.
mod common;
use common::start;
use reqwest::Method;
use serde_json::{json, Value};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn frequent_items_across_nodes_matches_single_shard() {
    let cl = start(&[]).await;
    let ports = cl.p;
    for (name, shards) in [("whole",1), ("split",4)] {
        let (status, body) = cl.call(0, Method::PUT, &format!("/{name}"), Some(json!({
            "settings":{"number_of_shards":shards,"number_of_replicas":1},
            "mappings":{"properties":{"tags":{"type":"keyword"},"active":{"type":"boolean"},"type":{"type":"keyword"}}}
        }))).await;
        assert_eq!(status,200,"{body}");
        let (status, body) = cl.call(0,Method::GET,&format!("/_cluster/health/{name}?wait_for_status=green&timeout=60s"),None).await;
        assert_eq!(status,200,"{body}");
        assert_eq!(body["status"],"green","{body}");
        let mut bulk=String::new();
        for i in 0..100 {
            bulk.push_str(&format!("{{\"index\":{{\"_id\":\"{i}\"}}}}\n"));
            bulk.push_str(&json!({"tags":if i<60 {vec!["a","a","b","python"]} else {vec!["b","c","java"]},"active":i%3==0,"type":"question"}).to_string());
            bulk.push('\n');
        }
        let body: Value=cl.c.post(format!("http://127.0.0.1:{}/{name}/_bulk?refresh=true",ports.http[0]))
            .header("content-type","application/x-ndjson").body(bulk).send().await.unwrap().json().await.unwrap();
        assert_eq!(body["errors"],false,"{body}");
    }
    for alias in ["frequent_item_sets","frequent_items"] {
        for filtered in [false,true] {
            let mut conf=json!({"fields":[{"field":"tags"},{"field":"active"}],"minimum_support":0.05,"minimum_set_size":2,"size":20});
            if filtered { conf["filter"]=json!({"term":{"active":true}}); }
            let query=json!({"size":0,"aggs":{"all":{"filter":{"match_all":{}},"aggs":{"sets":{alias:conf}}}}});
            let (status, expected)=cl.call(0,Method::POST,"/whole/_search",Some(query.clone())).await;
            assert_eq!(status,200,"{expected}");
            for node in 0..2 {
                let (status,got)=cl.call(node,Method::POST,"/split/_search",Some(query.clone())).await;
                assert_eq!(status,200,"{got}");
                assert_eq!(got["_shards"]["failed"],0,"{got}");
                assert_eq!(got["aggregations"],expected["aggregations"]);
                assert!(!got["aggregations"].to_string().contains("__p"));
                assert!(!got["aggregations"]["all"]["sets"]["buckets"].as_array().unwrap().is_empty());
            }
        }
    }
    // Exact Stack Overflow query shapes, including its script/query filters.
    for (size, support, tag) in [(100,0.01,None),(10,0.01,Some("python")),(1000,0.005,None)] {
        let mut must=vec![json!({"exists":{"field":"tags"}})];
        if let Some(tag)=tag { must.push(json!({"term":{"tags":{"value":tag}}})); }
        let query=json!({"size":0,"query":{"bool":{"must":must,"filter":[
            {"script":{"script":"doc['tags'].length > 2"}},{"term":{"type":"question"}}
        ]}},"aggs":{"fi":{"frequent_item_sets":{"minimum_set_size":2,"minimum_support":support,"size":size,"fields":[{"field":"tags"}]}}}});
        let (status,want)=cl.call(0,Method::POST,"/whole/_search?typed_keys=true",Some(query.clone())).await;
        assert_eq!(status,200,"{want}");
        let (status,got)=cl.call(1,Method::POST,"/split/_search?typed_keys=true",Some(query)).await;
        assert_eq!(status,200,"{got}");
        assert_eq!(got["_shards"]["failed"],0,"{got}");
        assert_eq!(got["aggregations"],want["aggregations"]);
        assert!(!got["aggregations"].to_string().contains("__p"));
    }

}
