//! Optional exact engine/HTTP oracle, compiled once to localize cluster overhead.
//! The default POC deliberately does not depend on either crate.
use axum::{
    Router,
    body::Bytes,
    extract::{Query as Params, State},
    http::StatusCode,
    routing::post,
};
use elasticsearch_engine::{
    node::{Node, ResolveOpts},
    search::{self, ExecOpts},
};
use std::{collections::HashMap, sync::Arc};

async fn fused_search(
    State(node): State<Arc<Node>>,
    Params(params): Params<HashMap<String, String>>,
    bytes: Bytes,
) -> Result<([(axum::http::HeaderName, &'static str); 1], String), (StatusCode, String)> {
    tokio::task::spawn_blocking(move || {
        let body = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        let req = search::parse_request(&body, &params).map_err(|e| e.reason)?;
        let targets = node
            .resolve("path-poc", ResolveOpts::from_params(&params))
            .map_err(|e| e.reason)?;
        let out = search::execute(
            &targets,
            &req,
            &ExecOpts {
                terms_lookup: None,
                cursors: None,
                max_buckets: node.max_buckets(),
                scroll_id: None,
                pit_id: None,
            },
        )
        .map_err(|e| e.reason)?;
        Ok::<_, String>((
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            out.json,
        ))
    })
    .await
    .unwrap()
    .map_err(|e| (StatusCode::BAD_REQUEST, e))
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let node = elasticsearch_engine::node::Node::open(std::path::Path::new(&args[1])).unwrap();
    let addr = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("127.0.0.1:19214")
        .parse()
        .unwrap();
    if args.get(3).map(String::as_str) == Some("fused") {
        // Benchmark subset only: identical engine parsing/collection/fetch,
        // one CPU job, without compatibility/observability/cluster wrappers.
        // Index loading and all other endpoints still use the exact HTTP layer.
        let router = Router::new()
            .route("/path-poc/_search", post(fused_search))
            .with_state(node.clone())
            .fallback_service(elasticsearch_http::router(node));
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
        axum::serve(listener, router).await.unwrap();
    } else {
        elasticsearch_http::serve(node, addr).await.unwrap();
    }
}
