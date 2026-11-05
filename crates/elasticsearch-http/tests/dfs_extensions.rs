//! In-process DFS hooks for a cluster coordinator: `_search` through
//! `router().oneshot()` with the [`DfsPhaseOnly`] / `Arc<DfsStats>` http
//! request extensions (unreachable from the network).

use axum::body::Body;
use axum::http::Request;
use elasticsearch_engine::node::Node;
use elasticsearch_http::{merge_dfs, DfsPhaseOnly, DfsStats};
use serde_json::{json, Value};
use std::sync::Arc;
use tower::ServiceExt;

async fn call(router: &axum::Router, method: &str, uri: &str, body: Value, ext: impl FnOnce(&mut axum::http::Extensions)) -> (u16, Value) {
    let mut req = Request::builder().method(method).uri(uri).header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
    ext(req.extensions_mut());
    let resp = router.clone().oneshot(req).await.unwrap();
    let st = resp.status().as_u16();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (st, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// A 1-shard engine holding `docs`.
async fn engine(dir: &std::path::Path, docs: &[&str]) -> axum::Router {
    let r = elasticsearch_http::router(Node::open(dir).unwrap());
    let (st, v) = call(&r, "PUT", "/i", json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}}), |_| {}).await;
    assert_eq!(st, 200, "{v}");
    for (n, d) in docs.iter().enumerate() {
        let (st, v) = call(&r, "PUT", &format!("/i/_doc/{n}?refresh=true"), json!({"title": d}), |_| {}).await;
        assert!(st == 200 || st == 201, "{v}");
    }
    r
}

fn scores(v: &Value) -> Vec<(String, f64)> {
    v["hits"]["hits"].as_array().unwrap().iter().map(|h| (h["_id"].as_str().unwrap().to_string(), h["_score"].as_f64().unwrap())).collect()
}

#[tokio::test]
async fn dfs_stats_round_and_supplied_stats_through_the_router() {
    let (a, b, all) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let docs_a = ["quick brown fox", "lazy dog", "quick quick fox jumps"];
    let docs_b = ["the dog barks", "a fox", "dog dog dog", "nothing here"];
    let (ea, eb) = (engine(a.path(), &docs_a).await, engine(b.path(), &docs_b).await);
    // Reference: every doc in one shard.
    let one: Vec<&str> = docs_a.iter().chain(docs_b.iter()).copied().collect();
    let ref_engine = engine(all.path(), &one).await;
    let q = json!({"query": {"match": {"title": "quick fox dog"}}, "size": 20});

    // Stats round: 200 with the shard's DfsStats, after full validation.
    let mut parts = Vec::new();
    for e in [&ea, &eb] {
        let (st, v) = call(e, "POST", "/i/_search", q.clone(), |x| {
            x.insert(DfsPhaseOnly);
        })
        .await;
        assert_eq!(st, 200, "{v}");
        parts.push(serde_json::from_value::<DfsStats>(v).unwrap());
    }
    let global = merge_dfs(&parts);
    let (st, v) = call(&ref_engine, "POST", "/i/_search", q.clone(), |x| {
        x.insert(DfsPhaseOnly);
    })
    .await;
    assert_eq!(st, 200);
    assert_eq!(serde_json::from_value::<DfsStats>(v).unwrap(), global);

    // Validation still runs before the DFS phase (ES-exact error).
    let (st, v) = call(&ea, "POST", "/i/_search", json!({"nope": 1}), |x| {
        x.insert(DfsPhaseOnly);
    })
    .await;
    assert_eq!(st, 400, "{v}");

    // Supplied stats: each shard engine scores like the single-shard reference.
    let (_, rv) = call(&ref_engine, "POST", "/i/_search", q.clone(), |_| {}).await;
    let want: std::collections::HashMap<String, f64> = scores(&rv).into_iter().collect();
    let shared = Arc::new(global);
    let mut seen = 0;
    for (e, offset) in [(&ea, 0), (&eb, docs_a.len())] {
        let s = shared.clone();
        let (st, v) = call(e, "POST", "/i/_search", q.clone(), move |x| {
            x.insert(s);
        })
        .await;
        assert_eq!(st, 200, "{v}");
        for (id, score) in scores(&v) {
            let ref_id = (id.parse::<usize>().unwrap() + offset).to_string();
            assert_eq!(Some(&score), want.get(&ref_id), "doc {id}");
            seen += 1;
        }
        // Without supplied stats the shard scores with its own statistics.
        let (_, plain) = call(e, "POST", "/i/_search", q.clone(), |_| {}).await;
        assert_eq!(scores(&plain).len(), scores(&v).len());
    }
    assert_eq!(seen, want.len());
}
