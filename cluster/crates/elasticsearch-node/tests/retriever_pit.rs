//! A compound retriever search on a two-node cluster reads one point in
//! time: its child searches and its final fetch see the same documents
//! (ES opens an implicit point in time shared by the sub-retrievers;
//! production review #262).
mod common;
use common::start;
use reqwest::Method;
use serde_json::{json, Value};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compound_retrieval_reads_one_point_in_time() {
    let cl = start(&[]).await;
    let (st, v) = cl.call(0, Method::PUT, "/rp", Some(json!({
        "settings": {"number_of_shards": 2, "number_of_replicas": 0},
        "mappings": {"properties": {"t": {"type": "text"}}}
    }))).await;
    assert_eq!(st, 200, "{v}");
    cl.call(0, Method::GET, "/_cluster/health/rp?wait_for_status=green&timeout=60s", None).await;
    for (id, t) in [("1", "alpha beta"), ("2", "alpha"), ("3", "beta")] {
        let (st, v) = cl.call(0, Method::PUT, &format!("/rp/_doc/{id}?refresh=true"), Some(json!({"t": t}))).await;
        assert!(st == 200 || st == 201, "{v}");
    }
    let rrf = json!({"retriever": {"rrf": {"retrievers": [{"standard": {"query": {"match": {"t": "alpha"}}}}, {"standard": {"query": {"match": {"t": "beta"}}}}]}}});
    // Control: no change between the phases.
    let (st, v) = cl.call(0, Method::POST, "/rp/_search", Some(rrf.clone())).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["hits"].as_array().unwrap().len(), 3, "{v}");
    assert!(v.get("pit_id").is_none(), "the caller asked for no point in time: {v}");
    // Document 1 is replaced while the search waits between its phases.
    *elasticsearch_node::front::RANK_PHASE_PAUSE.lock().unwrap() = Some(("rp".to_string(), 2000));
    let search = {
        let (c, port, body) = (cl.c.clone(), cl.p.http[0], rrf.clone());
        tokio::spawn(async move {
            let r = c.post(format!("http://127.0.0.1:{port}/rp/_search")).json(&body).send().await.unwrap();
            (r.status().as_u16(), r.json::<Value>().await.unwrap())
        })
    };
    tokio::time::sleep(std::time::Duration::from_millis(700)).await;
    let (st, v) = cl.call(1, Method::PUT, "/rp/_doc/1?refresh=true", Some(json!({"t": "gamma"}))).await;
    assert!(st == 200 || st == 201, "{v}");
    let (st, v) = search.await.unwrap();
    *elasticsearch_node::front::RANK_PHASE_PAUSE.lock().unwrap() = None;
    assert_eq!(st, 200, "{v}");
    let hit = v["hits"]["hits"].as_array().unwrap().iter().find(|h| h["_id"] == "1").cloned().unwrap_or_else(|| panic!("the ranked doc: {v}"));
    assert_eq!(hit["_source"]["t"], json!("alpha beta"), "fetched as it was ranked: {v}");
    assert!(v.get("pit_id").is_none(), "{v}");
    // The implicit point in time is closed.
    let (_, stats) = cl.call(0, Method::GET, "/rp/_stats/search", None).await;
    assert_eq!(stats["_all"]["total"]["search"]["open_contexts"], json!(0), "{stats}");
}

/// kNN retrievers read the point in time too (a kNN child is a DFS kNN
/// search of the point in time), and a point-in-time search combining kNN
/// with a query works across shards.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn knn_compound_retrieval_reads_one_point_in_time() {
    let cl = start(&[]).await;
    let (st, v) = cl.call(0, Method::PUT, "/rk", Some(json!({
        "settings": {"number_of_shards": 2, "number_of_replicas": 0},
        "mappings": {"properties": {"t": {"type": "text"}, "v": {"type": "dense_vector", "dims": 1, "index": true, "similarity": "l2_norm"}}}
    }))).await;
    assert_eq!(st, 200, "{v}");
    cl.call(0, Method::GET, "/_cluster/health/rk?wait_for_status=green&timeout=60s", None).await;
    for (id, t, x) in [("1", "alpha", 0.0), ("2", "alpha", 5.0), ("3", "beta", 1.0)] {
        let (st, v) = cl.call(0, Method::PUT, &format!("/rk/_doc/{id}?refresh=true"), Some(json!({"t": t, "v": [x]}))).await;
        assert!(st == 200 || st == 201, "{v}");
    }
    let rrf = json!({"retriever": {"rrf": {"retrievers": [{"standard": {"query": {"match": {"t": "alpha"}}}}, {"knn": {"field": "v", "query_vector": [0.0], "k": 2, "num_candidates": 3}}]}}});
    *elasticsearch_node::front::RANK_PHASE_PAUSE.lock().unwrap() = Some(("rk".to_string(), 2000));
    let search = {
        let (c, port, body) = (cl.c.clone(), cl.p.http[0], rrf.clone());
        tokio::spawn(async move {
            let r = c.post(format!("http://127.0.0.1:{port}/rk/_search")).json(&body).send().await.unwrap();
            (r.status().as_u16(), r.json::<Value>().await.unwrap())
        })
    };
    tokio::time::sleep(std::time::Duration::from_millis(700)).await;
    let (st, v) = cl.call(1, Method::PUT, "/rk/_doc/1?refresh=true", Some(json!({"t": "gamma", "v": [9.0]}))).await;
    assert!(st == 200 || st == 201, "{v}");
    let (st, v) = search.await.unwrap();
    *elasticsearch_node::front::RANK_PHASE_PAUSE.lock().unwrap() = None;
    assert_eq!(st, 200, "{v}");
    let hit = v["hits"]["hits"].as_array().unwrap().iter().find(|h| h["_id"] == "1").cloned().unwrap_or_else(|| panic!("the ranked doc: {v}"));
    assert_eq!(hit["_source"]["t"], json!("alpha"), "fetched as it was ranked: {v}");
    // A point-in-time search with kNN and a query over both shards.
    let (st, p) = cl.call(0, Method::POST, "/rk/_pit?keep_alive=1m", None).await;
    assert_eq!(st, 200, "{p}");
    let (st, v) = cl.call(0, Method::POST, "/_search", Some(json!({"pit": {"id": p["id"]}, "query": {"match": {"t": "alpha"}},
        "knn": {"field": "v", "query_vector": [0.0], "k": 1, "num_candidates": 3}}))).await;
    assert_eq!(st, 200, "{v}");
    let mut ids: Vec<&str> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap()).collect();
    ids.sort();
    // Doc 1 is now `gamma` at 9.0: the query matches doc 2, the kNN's top 1
    // is doc 3 (1.0).
    assert_eq!(ids, ["2", "3"], "{v}");
}
