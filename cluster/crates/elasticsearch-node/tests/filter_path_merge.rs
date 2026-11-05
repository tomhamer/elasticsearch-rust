//! `filter_path` on coordinator-merged responses: ES filters the final
//! response once, after the reduce. The shards' parts must reach the merge
//! unfiltered (sort values, `_shards`, totals), and nothing the merge
//! renders (`took`, `_shards`, `max_score`, `profile`) may survive a filter
//! that excludes it.

#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use elasticsearch_node::config::{NodeConfig, Settings};
use elasticsearch_node::node::ClusterNode;
use serde_json::{json, Value};
use std::time::Duration;

/// The nodes' bound HTTP ports (OS-assigned), set once the test has
/// started them (one cluster per test binary).
static HTTP_PORTS: std::sync::OnceLock<[u16; 2]> = std::sync::OnceLock::new();

fn http() -> [u16; 2] {
    *HTTP_PORTS.get().expect("nodes started")
}

/// Node `i` on port 0; `seeds`: the transport ports of the nodes started
/// before it.
fn settings(i: usize, dir: &std::path::Path, seeds: &[u16]) -> Settings {
    let mut s = Settings::new();
    let mut put = |k: &str, v: Value| {
        s.insert(k.to_string(), v);
    };
    put("cluster.name", json!("filter-path-merge"));
    put("node.name", json!(format!("n{i}")));
    put("path.data", json!(dir.join(format!("n{i}")).to_string_lossy()));
    put("path.logs", json!(dir.join(format!("logs{i}")).to_string_lossy()));
    put("network.host", json!("127.0.0.1"));
    put("xpack.security.enabled", json!("false"));
    put("http.port", json!("0"));
    put("transport.port", json!("0"));
    put("discovery.seed_hosts", json!(ports::seeds(seeds)));
    put("cluster.initial_master_nodes", json!(["n0", "n1"]));
    for w in ["low", "high", "flood_stage"] {
        put(&format!("cluster.routing.allocation.disk.watermark.{w}"), json!("1b"));
    }
    s
}

async fn call(c: &reqwest::Client, node: usize, method: reqwest::Method, path: &str, body: Option<Value>) -> (u16, String) {
    let mut r = c.request(method, format!("http://127.0.0.1:{}{path}", http()[node]));
    if let Some(b) = body {
        r = r.json(&b);
    }
    let resp = r.send().await.expect("request");
    let status = resp.status().as_u16();
    (status, resp.text().await.unwrap_or_default())
}

async fn json_call(c: &reqwest::Client, node: usize, method: reqwest::Method, path: &str, body: Option<Value>) -> Value {
    let (st, text) = call(c, node, method, path, body).await;
    assert!((200..300).contains(&st), "{path}: {st} {text}");
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path}: {e}: {text}"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn filter_path_applies_once_to_the_merged_response() {
    let dir = evidence::retained_dir("filter_path_merge");
    let mut http_ports = [0u16; 2];
    let mut transport_ports: Vec<u16> = Vec::new();
    for i in 0..2 {
        let cfg = NodeConfig::from_settings(settings(i, dir.path(), &transport_ports)).expect("config");
        let (node, listener) = ClusterNode::start(cfg).await.unwrap_or_else(|e| panic!("node {i}: {e}"));
        elasticsearch_node::rest::register(&node);
        elasticsearch_node::snapshot_api::register(&node);
        http_ports[i] = ports::http_port(&node);
        transport_ports.push(ports::transport_port(&node));
        tokio::spawn(async move {
            if let Err(e) = elasticsearch_node::front::serve(node, listener).await {
                panic!("node {i}: {e}");
            }
        });
    }
    HTTP_PORTS.set(http_ports).expect("one cluster per test binary");
    let c = reqwest::Client::new();
    ports::wait_formed(&c, &http(), None, 2, Duration::from_secs(60)).await
        .unwrap_or_else(|e| panic!("cluster did not form: {e}"));

    use reqwest::Method as M;
    json_call(&c, 0, M::PUT, "/f", Some(json!({"settings": {"number_of_shards": 4, "number_of_replicas": 0}, "mappings": {"properties": {"n": {"type": "integer"}, "k": {"type": "keyword"}}}}))).await;
    json_call(&c, 0, M::GET, "/_cluster/health/f?wait_for_status=green&timeout=60s", None).await;
    let mut bulk = String::new();
    for n in 0..20 {
        bulk.push_str(&format!("{{\"index\":{{\"_id\":\"d{n}\"}}}}\n{{\"n\":{n},\"k\":\"{}\"}}\n", if n % 2 == 0 { "even" } else { "odd" }));
    }
    let (st, text) = {
        let r = c.post(format!("http://127.0.0.1:{}/f/_bulk?refresh=true", http()[0])).header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap();
        (r.status().as_u16(), r.text().await.unwrap())
    };
    assert_eq!(st, 200, "{text}");

    for node in 0..2 {
        // Unfiltered: the merge renders ES's full shape, and no `profile`
        // unless one was asked for.
        let v = json_call(&c, node, M::POST, "/f/_search", Some(json!({"size": 1, "sort": ["n"]}))).await;
        assert_eq!(v["_shards"], json!({"total": 4, "successful": 4, "skipped": 0, "failed": 0}), "node {node}");
        assert!(v.get("profile").is_none(), "node {node}: {v}");
        assert_eq!(v["hits"]["hits"][0]["sort"], json!([0]), "node {node}");
        let v = json_call(&c, node, M::POST, "/f/_search", Some(json!({"size": 1, "profile": true}))).await;
        assert_eq!(v["profile"]["shards"].as_array().map(Vec::len), Some(4), "node {node}");

        // Only the requested paths survive, and the hits are merged in sort
        // order although `sort` itself is filtered out of the response.
        let v = json_call(&c, node, M::POST, "/f/_search?filter_path=hits.hits._id", Some(json!({"size": 5, "sort": [{"n": "desc"}]}))).await;
        assert_eq!(v, json!({"hits": {"hits": [{"_id": "d19"}, {"_id": "d18"}, {"_id": "d17"}, {"_id": "d16"}, {"_id": "d15"}]}}), "node {node}");

        // `_shards` comes from the shards' real headers.
        let v = json_call(&c, node, M::GET, "/f/_search?filter_path=_shards,hits.total", None).await;
        assert_eq!(v, json!({"_shards": {"total": 4, "successful": 4, "skipped": 0, "failed": 0}, "hits": {"total": {"value": 20, "relation": "eq"}}}), "node {node}");

        // Exclusions, wildcards and aggregations.
        let v = json_call(&c, node, M::POST, "/f/_search?filter_path=-took,-timed_out,-_shards,-hits", Some(json!({"size": 0, "aggs": {"t": {"terms": {"field": "k"}}}}))).await;
        assert_eq!(v, json!({"aggregations": {"t": {"doc_count_error_upper_bound": 0, "sum_other_doc_count": 0, "buckets": [{"key": "even", "doc_count": 10}, {"key": "odd", "doc_count": 10}]}}}), "node {node}");
        let v = json_call(&c, node, M::POST, "/f/_search?filter_path=aggregations.*.buckets.key", Some(json!({"size": 0, "aggs": {"t": {"terms": {"field": "k"}}}}))).await;
        assert_eq!(v, json!({"aggregations": {"t": {"buckets": [{"key": "even"}, {"key": "odd"}]}}}), "node {node}");

        // `_count`.
        let v = json_call(&c, node, M::GET, "/f/_count?filter_path=count", None).await;
        assert_eq!(v, json!({"count": 20}), "node {node}");

        // `_msearch`: the whole multi-response is filtered.
        let body = "{\"index\":\"f\"}\n{\"size\":2,\"sort\":[{\"n\":\"asc\"}]}\n{\"index\":\"f\"}\n{\"size\":1,\"sort\":[{\"n\":\"desc\"}]}\n";
        let r = c.post(format!("http://127.0.0.1:{}/_msearch?filter_path=responses.hits.hits._id", http()[node])).header("content-type", "application/x-ndjson").body(body).send().await.unwrap();
        let v: Value = r.json().await.unwrap();
        assert_eq!(v, json!({"responses": [{"hits": {"hits": [{"_id": "d0"}, {"_id": "d1"}]}}, {"hits": {"hits": [{"_id": "d19"}]}}]}), "node {node}");

        // Scroll: the first page and the continuation.
        let v = json_call(&c, node, M::POST, "/f/_search?scroll=1m&filter_path=_scroll_id,hits.hits._id", Some(json!({"size": 3, "sort": ["n"]}))).await;
        let id = v["_scroll_id"].as_str().expect("scroll id").to_string();
        assert_eq!(v["hits"], json!({"hits": [{"_id": "d0"}, {"_id": "d1"}, {"_id": "d2"}]}), "node {node}");
        assert_eq!(v.as_object().unwrap().len(), 2, "node {node}: {v}");
        let v = json_call(&c, node, M::POST, "/_search/scroll?filter_path=hits.hits._id", Some(json!({"scroll": "1m", "scroll_id": id}))).await;
        assert_eq!(v, json!({"hits": {"hits": [{"_id": "d3"}, {"_id": "d4"}, {"_id": "d5"}]}}), "node {node}");
        json_call(&c, node, M::DELETE, "/_search/scroll", Some(json!({"scroll_id": id}))).await;

        // A filter matching nothing leaves an empty object, as in ES.
        let v = json_call(&c, node, M::GET, "/f/_search?filter_path=nothing.here", None).await;
        assert_eq!(v, json!({}), "node {node}");

        // `pretty` still pretty-prints the filtered merge.
        let (st, text) = call(&c, node, M::GET, "/f/_count?filter_path=count&pretty", None).await;
        assert_eq!(st, 200);
        assert!(text.contains('\n'), "node {node}: not pretty: {text}");
        assert_eq!(serde_json::from_str::<Value>(&text).unwrap(), json!({"count": 20}), "node {node}");
    }
}
