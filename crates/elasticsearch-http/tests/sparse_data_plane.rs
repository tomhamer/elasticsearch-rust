//! A sparse (cluster-managed) index serves the HTTP data plane over the
//! shard copies open on this node: document ops routed to an open copy
//! apply there, ops routed elsewhere fail with `shard_not_found_exception`,
//! a search reads the open copies its routing selects, and a search one
//! of whose shards is not open here fails (`shard_not_found_exception`)
//! rather than answering for the others.

use axum::body::Body;
use axum::http::Request;
use elasticsearch_engine::index::{RecoverySource, ShardOpenSpec};
use elasticsearch_engine::node::Node;
use serde_json::{json, Value};
use tower::ServiceExt;

async fn call(router: &axum::Router, method: &str, uri: &str, body: Option<Value>) -> (u16, Value) {
    let b = body.map(|v| Body::from(v.to_string())).unwrap_or_else(Body::empty);
    let req = Request::builder().method(method).uri(uri).header("content-type", "application/json").body(b).unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    let st = resp.status().as_u16();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (st, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

#[tokio::test]
async fn sparse_index_serves_its_open_copies() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    let meta = json!({"settings": {"index.number_of_shards": "2", "index.number_of_replicas": "0"}, "mappings": {"properties": {"t": {"type": "keyword"}}}});
    let svc = node.create_index_sparse("i", "iUuidSparse1", &meta).unwrap();
    let r = elasticsearch_http::router(node.clone());

    // No copy open yet: search / count fail, never a 200 that silently
    // covers zero shards (the cluster then reads another copy).
    let (st, v) = call(&r, "POST", "/i/_search", Some(json!({}))).await;
    assert_eq!((st, v["error"]["type"].as_str()), (404, Some("shard_not_found_exception")), "{v}");
    let (st, v) = call(&r, "GET", "/i/_count", None).await;
    assert_eq!((st, v["error"]["type"].as_str()), (404, Some("shard_not_found_exception")), "{v}");

    svc.open_shard(1, ShardOpenSpec::local_primary(RecoverySource::Empty)).unwrap();
    let to = |n: u32| (0..1000).map(|i| format!("r{i}")).find(|x| svc.route("x", Some(x)).unwrap() == n).unwrap();
    let (r0, r1) = (to(0), to(1));

    // Routed to the open copy: created.
    let (st, v) = call(&r, "POST", &format!("/i/_doc?routing={r1}&refresh=true"), Some(json!({"t": "one"}))).await;
    assert_eq!(st, 201, "{v}");
    let id = v["_id"].as_str().unwrap().to_string();
    let (st, v) = call(&r, "PUT", &format!("/i/_doc/k?routing={r1}&refresh=true"), Some(json!({"t": "two"}))).await;
    assert_eq!(st, 201, "{v}");
    // Get / update on the open copy.
    let (st, v) = call(&r, "GET", &format!("/i/_doc/k?routing={r1}"), None).await;
    assert_eq!((st, v["found"].as_bool()), (200, Some(true)), "{v}");
    let (st, v) = call(&r, "POST", &format!("/i/_update/k?routing={r1}&refresh=true"), Some(json!({"doc": {"t": "three"}}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["result"], "updated");

    // A whole-index search with one of its shards not open here fails: it
    // would answer for some shards and report none failed.
    let (st, v) = call(&r, "POST", "/i/_search", Some(json!({"query": {"match_all": {}}}))).await;
    assert_eq!((st, v["error"]["type"].as_str()), (404, Some("shard_not_found_exception")), "{v}");
    // Routed to the open copy: 200 with exactly its docs.
    let (st, v) = call(&r, "POST", &format!("/i/_search?routing={r1}"), Some(json!({"query": {"match_all": {}}}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["_shards"]["total"], 1);
    let mut ids: Vec<&str> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap()).collect();
    ids.sort();
    let mut want = vec![id.as_str(), "k"];
    want.sort();
    assert_eq!(ids, want);

    // Routed to the shard with no local copy: shard_not_found (404).
    let (st, v) = call(&r, "POST", &format!("/i/_doc?routing={r0}"), Some(json!({"t": "x"}))).await;
    assert_eq!(st, 404, "{v}");
    assert_eq!(v["error"]["type"], "shard_not_found_exception");
    let (st, v) = call(&r, "GET", &format!("/i/_doc/k?routing={r0}"), None).await;
    assert_eq!((st, v["error"]["type"].as_str()), (404, Some("shard_not_found_exception")), "{v}");

    // A search routed only to the shard with no local copy fails too.
    let (st, v) = call(&r, "POST", &format!("/i/_search?routing={r0}"), Some(json!({}))).await;
    assert_eq!((st, v["error"]["type"].as_str()), (404, Some("shard_not_found_exception")), "{v}");

    // Count: the whole index fails as the search does; routed to the open copy, 200.
    let (st, v) = call(&r, "GET", "/i/_count", None).await;
    assert_eq!((st, v["error"]["type"].as_str()), (404, Some("shard_not_found_exception")), "{v}");
    let (st, v) = call(&r, "GET", &format!("/i/_count?routing={r1}"), None).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!((v["count"].as_u64(), v["_shards"]["total"].as_u64()), (Some(2), Some(1)));

    // Bulk: per-item results.
    let body = format!("{{\"index\":{{\"_id\":\"b1\",\"routing\":\"{r1}\"}}}}\n{{\"t\":\"a\"}}\n{{\"index\":{{\"_id\":\"b0\",\"routing\":\"{r0}\"}}}}\n{{\"t\":\"b\"}}\n");
    let req = Request::builder().method("POST").uri("/i/_bulk?refresh=true").header("content-type", "application/x-ndjson").body(Body::from(body)).unwrap();
    let resp = r.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let v: Value = serde_json::from_slice(&axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(v["errors"], true);
    assert_eq!(v["items"][0]["index"]["status"], 201, "{v}");
    assert_eq!(v["items"][1]["index"]["status"], 404, "{v}");
    assert_eq!(v["items"][1]["index"]["error"]["type"], "shard_not_found_exception");

    // Scroll and PIT work over the open copy (routed to it).
    let (st, v) = call(&r, "POST", &format!("/i/_search?scroll=1m&routing={r1}"), Some(json!({"size": 1}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["total"]["value"], 3);
    let (st, v) = call(&r, "POST", &format!("/i/_pit?keep_alive=1m&routing={r1}"), None).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["_shards"]["total"], 1);

    // Opening shard 0 too: writes there succeed, search covers both.
    svc.open_shard(0, ShardOpenSpec::local_primary(RecoverySource::Empty)).unwrap();
    let (st, v) = call(&r, "POST", &format!("/i/_doc?routing={r0}&refresh=true"), Some(json!({"t": "zero"}))).await;
    assert_eq!(st, 201, "{v}");
    let (st, v) = call(&r, "GET", "/i/_count", None).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!((v["count"].as_u64(), v["_shards"]["total"].as_u64()), (Some(4), Some(2)));
}
