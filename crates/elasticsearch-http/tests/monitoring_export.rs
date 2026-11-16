//! Finding 388: legacy monitoring export indexes the component payload, so
//! Stack Monitoring readers find the reported nodes.

use elasticsearch_engine::node::Node;
use reqwest::{Client, Method};
use serde_json::{json, Value};

#[tokio::test]
async fn exported_payload_fields_are_indexed() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    let c = Client::new();
    let base = format!("http://{addr}");
    let call = |m: Method, p: String, body: Option<String>, ct: &'static str| {
        let c = c.clone();
        let url = format!("{base}{p}");
        async move {
            let mut r = c.request(m, url);
            if let Some(b) = body {
                r = r.header("content-type", ct).body(b);
            }
            let resp = r.send().await.unwrap();
            let st = resp.status().as_u16();
            let t = resp.text().await.unwrap();
            (st, serde_json::from_str::<Value>(&t).unwrap_or(Value::String(t)))
        }
    };
    let (st, v) = call(Method::PUT, "/_cluster/settings".into(), Some(json!({"persistent": {"xpack.monitoring.collection.enabled": true}}).to_string()), "application/json").await;
    assert_eq!(st, 200, "{v}");
    let mut body = String::new();
    for (uuid, events) in [("ls-1", 10), ("ls-2", 3)] {
        body.push_str(&json!({"index": {"_type": "logstash_stats"}}).to_string());
        body.push('\n');
        body.push_str(&json!({"logstash": {"uuid": uuid, "name": "host", "version": "9.5.5"}, "events": {"in": events, "out": 2}}).to_string());
        body.push('\n');
    }
    let (st, v) = call(Method::POST, "/_monitoring/bulk?system_id=logstash&system_api_version=7&interval=10s".into(), Some(body), "application/x-ndjson").await;
    assert_eq!(st, 200, "{v}");
    assert_eq!((v["errors"].clone(), v["ignored"].clone()), (json!(false), json!(false)), "{v}");
    let _ = call(Method::POST, "/.monitoring-logstash-*/_refresh".into(), None, "application/json").await;
    let q = json!({
        "size": 0,
        "query": {"bool": {"filter": [{"exists": {"field": "logstash_stats.logstash.uuid"}}, {"range": {"logstash_stats.events.in": {"gte": 5}}}]}},
        "aggs": {"nodes": {"terms": {"field": "logstash_stats.logstash.uuid"}}}
    });
    let (st, v) = call(Method::POST, "/.monitoring-logstash-*/_search".into(), Some(q.to_string()), "application/json").await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["total"]["value"], 1, "{v}");
    assert_eq!(v["aggregations"]["nodes"]["buckets"][0]["key"], "ls-1", "{v}");
    // ES's monitoring-logstash template types, not inferred ones.
    let (st, v) = call(Method::GET, "/.monitoring-logstash-*/_mapping/field/logstash_stats.events.in,logstash_stats.logstash.uuid,logstash_stats.os.cpu.load_average.1m".into(), None, "application/json").await;
    assert_eq!(st, 200, "{v}");
    let types: Vec<(String, String)> = v.as_object().unwrap().values().flat_map(|i| i["mappings"].as_object().unwrap().iter().map(|(k, m)| (k.clone(), m["mapping"].as_object().unwrap().values().next().unwrap()["type"].as_str().unwrap().to_string())).collect::<Vec<_>>()).collect();
    let mut types = types;
    types.sort();
    assert_eq!(types, vec![
        ("logstash_stats.events.in".to_string(), "long".to_string()),
        ("logstash_stats.logstash.uuid".to_string(), "keyword".to_string()),
        ("logstash_stats.os.cpu.load_average.1m".to_string(), "half_float".to_string()),
    ], "{v}");
    // The Kibana and Beats templates are accepted as well.
    for (system, ty, doc) in [("kibana", "kibana_stats", json!({"kibana": {"uuid": "k-1"}, "requests": {"total": 4}})), ("beats", "beats_stats", json!({"beat": {"uuid": "b-1", "type": "filebeat"}, "metrics": {"libbeat": {"output": {"events": {"acked": 7}}}}}))] {
        let body = format!("{}\n{}\n", json!({"index": {"_type": ty}}), doc);
        let (st, v) = call(Method::POST, format!("/_monitoring/bulk?system_id={system}&system_api_version=7&interval=10s"), Some(body), "application/x-ndjson").await;
        assert_eq!(st, 200, "{system}: {v}");
        assert_eq!(v["errors"], false, "{system}: {v}");
    }
}
