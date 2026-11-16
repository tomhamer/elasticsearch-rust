//! Runtime `lookup` fields over a two-node cluster: the coordinator runs the
//! term lookups on the target index (whose documents live on both nodes)
//! and fills every hit, from either node, as one node would.
mod common;
use common::start;
use reqwest::Method;
use serde_json::{json, Value};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lookup_fields_resolve_through_the_coordinator() {
    let cl = start(&[]).await;
    for (name, shards, mapping) in [
        ("ips", 2, json!({"properties": {"ip": {"type": "ip"}, "name": {"type": "keyword"}}})),
        ("logs", 2, json!({"properties": {"client": {"type": "ip"}, "host": {"type": "keyword"}, "msg": {"type": "keyword"}}})),
        ("hosts", 2, json!({"properties": {"host": {"type": "keyword"}, "owner": {"type": "keyword"}}})),
    ] {
        let (status, body) = cl.call(0, Method::PUT, &format!("/{name}"), Some(json!({
            "settings": {"number_of_shards": shards, "number_of_replicas": 0, "index.routing.allocation.total_shards_per_node": 1},
            "mappings": mapping
        }))).await;
        assert_eq!(status, 200, "{body}");
        let (_, health) = cl.call(0, Method::GET, &format!("/_cluster/health/{name}?wait_for_status=green&timeout=60s"), None).await;
        assert_eq!(health["status"], "green", "{health}");
    }
    let docs: &[(&str, &str, Value)] = &[
        ("ips", "i1", json!({"ip": "10.0.0.1", "name": "alpha"})),
        ("ips", "i2", json!({"ip": "10.0.0.2", "name": "beta"})),
        ("ips", "i3", json!({"ip": "10.0.0.3", "name": "gamma"})),
        ("hosts", "web", json!({"host": "web", "owner": "ops"})),
        ("logs", "l1", json!({"client": "10.0.0.1", "host": "web", "msg": "a"})),
        ("logs", "l2", json!({"client": "10.0.0.2", "host": "db", "msg": "b"})),
        ("logs", "l3", json!({"client": "10.0.0.3", "host": "web", "msg": "c"})),
        ("logs", "l4", json!({"client": "10.0.0.9", "host": "web", "msg": "d"})),
    ];
    for (index, id, doc) in docs {
        let (status, body) = cl.call(0, Method::PUT, &format!("/{index}/_doc/{id}?refresh=true"), Some(doc.clone())).await;
        assert!(status == 200 || status == 201, "{body}");
    }
    let body = json!({"size": 10, "sort": [{"msg": "asc"}], "_source": false, "fields": ["who", "owner"], "runtime_mappings": {
        "who": {"type": "lookup", "target_index": "ips", "input_field": "client", "target_field": "ip", "fetch_fields": ["name"]},
        "owner": {"type": "lookup", "target_index": "hosts", "input_field": "host", "target_field": "_id", "fetch_fields": ["owner"]}
    }});
    for node in 0..2 {
        let (status, v) = cl.call(node, Method::POST, "/logs/_search", Some(body.clone())).await;
        assert_eq!(status, 200, "{v}");
        let hits = v["hits"]["hits"].as_array().unwrap();
        let fields: Vec<Value> = hits.iter().map(|h| h["fields"].clone()).collect();
        assert_eq!(fields, vec![
            json!({"who": [{"name": ["alpha"]}], "owner": [{"owner": ["ops"]}]}),
            json!({"who": [{"name": ["beta"]}]}),
            json!({"who": [{"name": ["gamma"]}], "owner": [{"owner": ["ops"]}]}),
            // No match: no field (and the input field is not returned unasked).
            json!({"owner": [{"owner": ["ops"]}]}),
        ], "via n{node}: {v}");
    }
    // A lookup field mapped on the searched index resolves the same way,
    // asked by name or through a `fields` pattern.
    let (status, body) = cl.call(0, Method::PUT, "/logs/_mapping", Some(json!({"runtime": {
        "who_m": {"type": "lookup", "target_index": "ips", "input_field": "client", "target_field": "ip", "fetch_fields": ["name"]}
    }}))).await;
    assert_eq!(status, 200, "{body}");
    for fields in [json!(["who_m"]), json!(["who_*"])] {
        let body = json!({"size": 10, "sort": [{"msg": "asc"}], "_source": false, "fields": fields});
        for node in 0..2 {
            let (status, v) = cl.call(node, Method::POST, "/logs/_search", Some(body.clone())).await;
            assert_eq!(status, 200, "{v}");
            let got: Vec<Value> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["fields"]["who_m"].clone()).collect();
            assert_eq!(got, vec![json!([{"name": ["alpha"]}]), json!([{"name": ["beta"]}]), json!([{"name": ["gamma"]}]), Value::Null],
                "{fields} via n{node}: {v}");
        }
    }
}
