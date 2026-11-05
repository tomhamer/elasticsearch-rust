//! A sub-request some executor thread waits for (on this node or another)
//! never queues behind the bounded executors (B9: two nodes' search
//! threads each waiting for fetches queued behind the other's).
use axum::{body::Body, http::Request};
use elasticsearch_engine::node::Node;
use elasticsearch_http::pools::{self, Pool};
use std::time::Duration;
use tower::ServiceExt;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn nested_requests_complete_while_every_search_thread_is_busy() {
    let dir = tempfile::tempdir().unwrap();
    let router = elasticsearch_http::router(Node::open(dir.path()).unwrap());
    // Hold every search thread, as fan-outs waiting on a peer would.
    let threads = pools::executor(Pool::Search).stats().threads;
    let (release, held) = std::sync::mpsc::channel::<()>();
    let held = std::sync::Arc::new(std::sync::Mutex::new(held));
    let (busy, all_busy) = std::sync::mpsc::channel::<()>();
    let mut holders = Vec::new();
    for _ in 0..threads {
        let (held, busy) = (held.clone(), busy.clone());
        holders.push(tokio::spawn(pools::run(Pool::Search, move || {
            busy.send(()).unwrap();
            let _ = held.lock().unwrap().recv();
        })));
    }
    for _ in 0..threads {
        all_busy.recv_timeout(Duration::from_secs(10)).unwrap();
    }
    let search = |nested: bool| {
        let mut r = Request::builder().method("POST").uri("/_search").header("content-type", "application/json").body(Body::from("{}")).unwrap();
        if nested { r.extensions_mut().insert(elasticsearch_http::NestedRequest); }
        router.clone().oneshot(r)
    };
    // A plain search queues behind the held threads.
    assert!(tokio::time::timeout(Duration::from_millis(300), search(false)).await.is_err(), "a plain search queues");
    // The nested one does not.
    let response = tokio::time::timeout(Duration::from_secs(10), search(true)).await.expect("a nested search completes").unwrap();
    assert_eq!(response.status().as_u16(), 200);
    for _ in 0..threads { release.send(()).unwrap(); }
    for h in holders { h.await.unwrap().unwrap(); }
}
