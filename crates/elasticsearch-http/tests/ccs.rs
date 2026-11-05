use elasticsearch_engine::node::Node;
use reqwest::{Client, Method};
use serde_json::{json, Value};

async fn call(client: &Client, base: &str, method: Method, path: &str, body: Value) -> (u16, Value) {
    let mut request = client.request(method, format!("{base}{path}"));
    if !body.is_null() { request = request.json(&body); }
    let response = request.send().await.unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap())
}

#[tokio::test]
async fn unsupported_self_aliases_reject_or_explicitly_skip_without_fabricated_hits() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(elasticsearch_http::serve_listener(node.clone(), listener));
    let client = Client::new();
    assert_eq!(call(&client, &base, Method::PUT, "/i", json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"properties": {"tag": {"type": "keyword"}}}})).await.0, 200);
    assert_eq!(call(&client, &base, Method::PUT, "/i/_doc/1?refresh=true", json!({"tag": "yes"})).await.0, 201);
    assert_eq!(call(&client, &base, Method::PUT, "/ev", json!({"mappings": {"properties": {"@timestamp": {"type": "date"}, "event": {"properties": {"category": {"type": "keyword"}}}}}})).await.0, 200);
    assert_eq!(call(&client, &base, Method::PUT, "/ev/_doc/1?refresh=true", json!({"@timestamp": "2026-01-01T00:00:00Z", "event": {"category": "process"}})).await.0, 201);
    assert_eq!(call(&client, &base, Method::PUT, "/_cluster/settings", json!({"persistent": {
        "cluster.remote.a.seeds": ["127.0.0.1:9300"], "cluster.remote.b.seeds": ["127.0.0.1:9300"],
        "cluster.remote.down.seeds": ["127.0.0.1:1"], "cluster.remote.down.skip_unavailable": true
    }})).await.0, 200);
    // Required self aliases fail instead of duplicating the local document.
    assert_eq!(call(&client, &base, Method::PUT, "/_cluster/settings", json!({"persistent": {
        "cluster.remote.a.skip_unavailable": false, "cluster.remote.b.skip_unavailable": false
    }})).await.0, 200);
    let (status, failed) = call(&client, &base, Method::POST, "/i,a:i,b:i/_search", json!({"query": {"match_all": {}}})).await;
    assert_eq!(status, 502, "{failed}");
    assert_eq!(failed["error"]["type"], "connect_transport_exception");
    assert!(failed.get("hits").is_none());
    let (_, info) = call(&client, &base, Method::GET, "/_remote/info", Value::Null).await;
    for alias in ["a", "b", "down"] {
        assert_eq!(info[alias]["connected"], false);
        assert_eq!(info[alias]["num_nodes_connected"], 0);
    }
    let (status, skipped) = call(&client, &base, Method::POST, "/i,down:i/_search", json!({"query": {"term": {"tag": "yes"}}})).await;
    assert_eq!(status, 200, "{skipped}");
    assert_eq!(skipped["hits"]["total"]["value"], 1);
    assert_eq!(skipped["hits"]["hits"][0]["_index"], "i");
    assert_eq!(skipped["_clusters"]["successful"], 1);
    assert_eq!(skipped["_clusters"]["skipped"], 1);
    // ES 8.10+ per-cluster details.
    assert_eq!(skipped["_clusters"]["details"]["(local)"]["status"], "successful", "{skipped}");
    assert_eq!(skipped["_clusters"]["details"]["(local)"]["indices"], "i");
    assert_eq!(skipped["_clusters"]["details"]["(local)"]["_shards"]["total"], 1);
    assert_eq!(skipped["_clusters"]["details"]["down"]["status"], "skipped");
    assert_eq!(skipped["_clusters"]["details"]["down"]["indices"], "i");
    assert_eq!(skipped["_clusters"]["details"]["down"]["failures"][0]["shard"], -1);
    assert_eq!(skipped["_clusters"]["details"]["down"]["failures"][0]["reason"]["type"], "connect_transport_exception");
    assert_eq!(skipped["_clusters"]["details"]["down"]["failures"][0]["reason"]["reason"], "[][127.0.0.1:1] connect_exception");
    // A purely remote, optional expression reads nothing local.
    let (status, only_remote) = call(&client, &base, Method::POST, "/down:i/_search", json!({})).await;
    assert_eq!(status, 200, "{only_remote}");
    assert_eq!(only_remote["hits"]["total"]["value"], 0, "{only_remote}");
    assert_eq!(only_remote["_clusters"]["total"], 1);
    assert!(only_remote["_clusters"]["details"].get("(local)").is_none());
    // Every entry point that accepts cluster:index plans it the same way.
    let (status, count) = call(&client, &base, Method::POST, "/i,down:i/_count", Value::Null).await;
    assert_eq!((status, count["count"].clone()), (200, json!(1)), "{count}");
    let (status, count) = call(&client, &base, Method::POST, "/down:i/_count", Value::Null).await;
    assert_eq!((status, count["count"].clone()), (200, json!(0)), "a remote-only count is not every local index: {count}");
    let (status, pit) = call(&client, &base, Method::POST, "/i,down:i/_pit?keep_alive=1m", Value::Null).await;
    assert_eq!(status, 200, "{pit}");
    assert_eq!(pit["_shards"]["total"], 1, "{pit}");
    let (status, caps) = call(&client, &base, Method::GET, "/i,down:i/_field_caps?fields=tag", Value::Null).await;
    assert_eq!(status, 200, "{caps}");
    assert_eq!(caps["indices"], json!(["i"]), "{caps}");
    let (status, eql) = call(&client, &base, Method::POST, "/ev,down:ev/_eql/search", json!({"query": "any where true"})).await;
    assert_eq!(status, 200, "{eql}");
    assert_eq!(eql["hits"]["events"].as_array().map(Vec::len), Some(1), "{eql}");
    let (status, esql) = call(&client, &base, Method::POST, "/_query", json!({"query": "FROM i,down:i | STATS c = COUNT(*)"})).await;
    assert_eq!(status, 200, "{esql}");
    assert_eq!(esql["values"], json!([[1]]), "{esql}");
    // _resolve/cluster reports the remote, unconnected, and its policy.
    let (status, resolved) = call(&client, &base, Method::GET, "/_resolve/cluster/i,down:i*", Value::Null).await;
    assert_eq!(status, 200, "{resolved}");
    assert_eq!(resolved["(local)"]["connected"], true, "{resolved}");
    assert_eq!(resolved["down"], json!({"connected": false, "skip_unavailable": true}), "{resolved}");
    let (_, resolved) = call(&client, &base, Method::GET, "/_resolve/cluster/down:i*", Value::Null).await;
    assert!(resolved.get("(local)").is_none(), "no local part, no local entry: {resolved}");
    let (_, resolved) = call(&client, &base, Method::GET, "/_resolve/cluster", Value::Null).await;
    assert_eq!(resolved["a"]["skip_unavailable"], false, "{resolved}");
    assert_eq!(resolved["down"]["connected"], false, "{resolved}");
    let (status, resolved) = call(&client, &base, Method::GET, "/_resolve/cluster/missing:i", Value::Null).await;
    assert_eq!(status, 404, "{resolved}");
    assert_eq!(resolved["error"]["type"], "no_such_remote_cluster_exception");
    let (_, stats) = call(&client, &base, Method::GET, "/_cluster/stats?include_remotes=true", Value::Null).await;
    // Both cross-cluster searches above (local+remote, remote-only).
    assert_eq!(stats["ccs"]["_search"]["total"], 2, "{stats}");
    assert_eq!(stats["ccs"]["_search"]["skipped"], 2, "{stats}");
    assert!(stats["ccs"].get("clusters").is_none(), "must not fabricate remote cluster stats: {stats}");
    assert_eq!(call(&client, &base, Method::PUT, "/_cluster/settings", json!({"persistent": {"cluster.remote.down.skip_unavailable": false}})).await.0, 200);
    let (status, failed) = call(&client, &base, Method::POST, "/down:i/_search", json!({"query": {"match_all": {}}})).await;
    assert_eq!(status, 502, "{failed}");
    // Required remotes fail every entry point instead of being dropped.
    for (method, path, body) in [
        (Method::POST, "/i,down:i/_count", Value::Null),
        (Method::POST, "/i,down:i/_pit?keep_alive=1m", Value::Null),
        (Method::GET, "/i,down:i/_field_caps?fields=tag", Value::Null),
        (Method::POST, "/ev,down:ev/_eql/search", json!({"query": "any where true"})),
        (Method::POST, "/_query", json!({"query": "FROM i,down:i | STATS c = COUNT(*)"})),
    ] {
        let (status, failed) = call(&client, &base, method, path, body).await;
        assert_eq!(status, 502, "{path}: {failed}");
        assert_eq!(failed["error"]["type"], "connect_transport_exception", "{path}: {failed}");
    }
    let (status, failed) = call(&client, &base, Method::POST, "/missing:i/_count", Value::Null).await;
    assert_eq!(status, 404, "{failed}");
    let (status, failed) = call(&client, &base, Method::POST, "/missing:i/_search", json!({"query": {"match_all": {}}})).await;
    assert_eq!(status, 404, "{failed}");
    assert_eq!(failed["error"]["type"], "no_such_remote_cluster_exception");
    server.abort();
}
