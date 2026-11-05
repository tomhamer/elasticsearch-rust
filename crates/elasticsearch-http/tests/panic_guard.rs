//! A panicking request handler answers 500 with an ES error body instead of
//! dropping the connection or taking the process down.

use axum::body::Body;
use axum::http::{Response, StatusCode};

#[tokio::test]
async fn panic_in_handler_becomes_500() {
    let response = elasticsearch_http::panic_guard(async {
        if std::hint::black_box(true) {
            panic!("boom in handler");
        }
        Response::new(Body::empty())
    })
    .await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(response.headers()["x-elastic-product"], "Elasticsearch");
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["status"], 500);
    assert_eq!(body["error"]["type"], "exception");
    assert!(body["error"]["reason"].as_str().unwrap().contains("boom in handler"), "{body}");
}

#[tokio::test]
async fn non_panicking_handler_passes_through() {
    let response = elasticsearch_http::panic_guard(async {
        let mut r = Response::new(Body::from("ok"));
        *r.status_mut() = StatusCode::CREATED;
        r
    })
    .await;
    assert_eq!(response.status(), StatusCode::CREATED);
}

/// The release profile must unwind: every panic guard in the node (request
/// pools, this HTTP guard, shard seal/tragic handling, the cluster-state
/// applier, the master service) is dead code under `panic = "abort"`.
#[test]
fn release_profile_unwinds() {
    let manifest = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.toml")).unwrap();
    let release = manifest.split("[profile.release]").nth(1).expect("[profile.release]");
    let release = release.split("\n[").next().unwrap();
    let panic = release.lines().map(str::trim).find(|l| l.starts_with("panic")).expect("panic setting in [profile.release]");
    assert!(panic.contains("\"unwind\""), "release profile must unwind, found `{panic}`");
}
