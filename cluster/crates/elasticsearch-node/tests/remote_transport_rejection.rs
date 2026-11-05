//! Unsupported-remote contract against four elasticsearch-node subprocesses,
//! two nodes in each independent cluster. No in-process adapter is installed.
mod common;
use common::procs::start_procs;
use serde_json::{json, Value};

async fn call(c: &reqwest::Client, http: u16, method: reqwest::Method, path: &str, body: Option<Value>) -> (u16, Value) {
    let mut request = c.request(method, format!("http://127.0.0.1:{http}{path}"));
    if let Some(body) = body { request = request.json(&body); }
    let response = request.send().await.expect("HTTP response");
    let status = response.status().as_u16();
    (status, response.json().await.expect("JSON response"))
}

async fn call_ndjson(c: &reqwest::Client, http: u16, path: &str, body: &str) -> (u16, Value) {
    let response = c.post(format!("http://127.0.0.1:{http}{path}")).header("content-type", "application/x-ndjson")
        .body(body.to_string()).send().await.expect("HTTP response");
    let status = response.status().as_u16();
    (status, response.json().await.expect("JSON response"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn separate_process_clusters_reject_remote_transport_without_local_substitution() {
    let license = [("xpack.license.self_generated.type", "trial")];
    let a_nodes = start_procs(2, &license).await;
    let b_nodes = start_procs(2, &license).await;
    let (http_a, transport_a) = (a_nodes.p.http.clone(), a_nodes.p.transport.clone());
    let (http_b, transport_b) = (b_nodes.p.http.clone(), b_nodes.p.transport.clone());
    a_nodes.wait_nodes(0, 2, 60).await;
    b_nodes.wait_nodes(0, 2, 60).await;
    let client = reqwest::Client::new();
    let (_, a) = call(&client, http_a[0], reqwest::Method::GET, "/", None).await;
    let (_, b) = call(&client, http_b[0], reqwest::Method::GET, "/", None).await;
    assert!(a["cluster_uuid"].as_str().is_some_and(|s| !s.is_empty() && s != "_na_"), "{a}");
    assert!(b["cluster_uuid"].as_str().is_some_and(|s| !s.is_empty() && s != "_na_"), "{b}");
    assert_ne!(a["cluster_uuid"], b["cluster_uuid"]);
    for (http, marker) in [(http_a[0], "local-only"), (http_b[0], "remote-only")] {
        let (status, body) = call(&client, http, reqwest::Method::PUT, "/remote-contract", Some(json!({
            "settings": {"number_of_shards": 1, "number_of_replicas": 1}
        }))).await;
        assert_eq!(status, 200, "{body}");
        let (status, body) = call(&client, http, reqwest::Method::PUT, "/remote-contract/_doc/1?refresh=true&wait_for_active_shards=all", Some(json!({"marker": marker}))).await;
        assert_eq!(status, 201, "{body}");
    }
    // Multi-shard local index: reads take the cluster fan-out path, which
    // rebuilds shard URIs from local names and must not drop remote parts.
    let (status, body) = call(&client, http_a[0], reqwest::Method::PUT, "/sharded", Some(json!({
        "settings": {"number_of_shards": 2, "number_of_replicas": 1}
    }))).await;
    assert_eq!(status, 200, "{body}");
    for id in 0..6 {
        let (status, body) = call(&client, http_a[0], reqwest::Method::PUT, &format!("/sharded/_doc/{id}?refresh=true&wait_for_active_shards=all"), Some(json!({"marker": "local-only"}))).await;
        assert_eq!(status, 201, "{body}");
    }
    let (status, body) = call(&client, http_a[0], reqwest::Method::PUT, "/_cluster/settings", Some(json!({"persistent": {
        "cluster.remote.actual.seeds": [format!("127.0.0.1:{}", transport_b[0])],
        "cluster.remote.actual.skip_unavailable": false,
        "cluster.remote.self.seeds": [format!("127.0.0.1:{}", transport_a[0])],
        "cluster.remote.self.skip_unavailable": false,
        "cluster.remote.optional.seeds": [format!("127.0.0.1:{}", transport_b[1])],
        "cluster.remote.optional.skip_unavailable": true
    }}))).await;
    assert_eq!(status, 200, "{body}");
    // Probe both local coordinators, including the node that did not accept the
    // settings update. Neither may infer connectivity from cluster membership.
    for http in http_a {
        let (status, info) = call(&client, http, reqwest::Method::GET, "/_remote/info", None).await;
        assert_eq!(status, 200, "{info}");
        assert_eq!(info["actual"]["seeds"], json!([format!("127.0.0.1:{}", transport_b[0])]), "{info}");
        assert_eq!(info["optional"]["skip_unavailable"], true, "{info}");
        for alias in ["actual", "self", "optional"] {
            assert_eq!(info[alias]["connected"], false, "{info}");
            assert_eq!(info[alias]["num_nodes_connected"], 0, "{info}");
        }
        for alias in ["actual", "self"] {
            let (status, body) = call(&client, http, reqwest::Method::POST, &format!("/remote-contract,{alias}:remote-contract/_search"), Some(json!({"query": {"match_all": {}}}))).await;
            assert_eq!(status, 502, "{body}");
            assert_eq!(body["error"]["type"], "connect_transport_exception", "{body}");
            assert!(body.get("hits").is_none(), "must not return local partial success: {body}");
            for follower in ["new-follower", "remote-contract"] {
                let (status, body) = call(&client, http, reqwest::Method::PUT, &format!("/{follower}/_ccr/follow"), Some(json!({"remote_cluster": alias, "leader_index": "remote-contract"}))).await;
                assert_eq!(status, 502, "{body}");
                assert_eq!(body["error"]["type"], "connect_transport_exception", "{body}");
            }
        }
        let (status, body) = call(&client, http, reqwest::Method::POST, "/remote-contract,optional:remote-contract/_search", Some(json!({"query": {"match_all": {}}}))).await;
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["hits"]["total"]["value"], 1, "{body}");
        assert_eq!(body["hits"]["hits"][0]["_source"]["marker"], "local-only", "{body}");
        assert_eq!(body["_clusters"]["successful"], 1, "{body}");
        assert_eq!(body["_clusters"]["skipped"], 1, "{body}");
        assert_eq!(body["_clusters"]["details"]["optional"]["status"], "skipped", "{body}");
        // Two-shard local index: same contract on the cluster fan-out path.
        for expr in ["sharded,optional:sharded", "sharded,opt*:sharded*"] {
            let (status, body) = call(&client, http, reqwest::Method::POST, &format!("/{expr}/_search"), Some(json!({"query": {"match_all": {}}}))).await;
            assert_eq!(status, 200, "{expr}: {body}");
            assert_eq!(body["hits"]["total"]["value"], 6, "{expr}: {body}");
            assert_eq!(body["_clusters"]["total"], 2, "{expr}: {body}");
            assert_eq!(body["_clusters"]["skipped"], 1, "{expr}: {body}");
            assert_eq!(body["_clusters"]["details"]["(local)"]["_shards"]["total"], 2, "{expr}: {body}");
            assert_eq!(body["_clusters"]["details"]["optional"]["status"], "skipped", "{expr}: {body}");
        }
        let (status, body) = call(&client, http, reqwest::Method::POST, "/sharded,optional:sharded/_count", None).await;
        assert_eq!((status, body["count"].clone()), (200, json!(6)), "{body}");
        let (status, body) = call(&client, http, reqwest::Method::POST, "/optional:sharded/_search?ignore_unavailable=true", Some(json!({}))).await;
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["hits"]["total"]["value"], 0, "a remote-only search must not read local indices: {body}");
        assert_eq!(body["_clusters"]["skipped"], 1, "{body}");
        let (status, body) = call(&client, http, reqwest::Method::POST, "/sharded,optional:sharded/_async_search?wait_for_completion_timeout=30s", Some(json!({}))).await;
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["response"]["hits"]["total"]["value"], 6, "{body}");
        assert_eq!(body["response"]["_clusters"]["skipped"], 1, "{body}");
        for (method, path) in [(reqwest::Method::POST, "/sharded,actual:sharded/_search"), (reqwest::Method::POST, "/sharded,actual:sharded/_count"),
            (reqwest::Method::POST, "/sharded,actual:sharded/_pit?keep_alive=1m"), (reqwest::Method::POST, "/sharded,actual:sharded/_async_search")] {
            let (status, body) = call(&client, http, method, path, Some(json!({}))).await;
            assert_eq!(status, 502, "{path}: {body}");
            assert_eq!(body["error"]["type"], "connect_transport_exception", "{path}: {body}");
            assert!(body.get("hits").is_none() && body.get("response").is_none(), "{path}: must not return local partial success: {body}");
        }
        // _msearch items with remote parts, including ignore_unavailable.
        let (status, body) = call_ndjson(&client, http, "/_msearch",
            "{\"index\":\"sharded,optional:sharded\",\"ignore_unavailable\":true}\n{}\n{\"index\":\"sharded,actual:sharded\"}\n{}\n{\"index\":\"remote-contract,optional:remote-contract\"}\n{}\n").await;
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["responses"][0]["hits"]["total"]["value"], 6, "{body}");
        assert_eq!(body["responses"][0]["_clusters"]["skipped"], 1, "{body}");
        assert_eq!(body["responses"][1]["status"], 502, "{body}");
        assert_eq!(body["responses"][1]["error"]["type"], "connect_transport_exception", "{body}");
        assert_eq!(body["responses"][2]["hits"]["total"]["value"], 1, "{body}");
        assert_eq!(body["responses"][2]["_clusters"]["skipped"], 1, "{body}");
        let (status, body) = call(&client, http, reqwest::Method::GET, "/_resolve/cluster/sharded,optional:sharded", None).await;
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["optional"], json!({"connected": false, "skip_unavailable": true}), "{body}");
        assert_eq!(body["(local)"]["connected"], true, "{body}");
        assert_eq!(call(&client, http, reqwest::Method::GET, "/new-follower", None).await.0, 404);
        let (status, body) = call(&client, http, reqwest::Method::GET, "/remote-contract/_doc/1", None).await;
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["_source"]["marker"], "local-only", "{body}");
        let (_, mapping) = call(&client, http, reqwest::Method::GET, "/remote-contract/_settings", None).await;
        assert_ne!(mapping["remote-contract"]["settings"]["index"]["xpack"]["ccr"]["following_index"], "true", "{mapping}");
    }
    let (_, remote) = call(&client, http_b[0], reqwest::Method::GET, "/remote-contract/_doc/1", None).await;
    assert_eq!(remote["_source"]["marker"], "remote-only", "{remote}");
}
