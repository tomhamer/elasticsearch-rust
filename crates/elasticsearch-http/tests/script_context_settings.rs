//! Public cross-setting validation; budget exhaustion itself uses isolated engine
//! admission instances so parallel HTTP fixtures cannot exhaust each other.
use serde_json::{json,Value};

#[tokio::test]
async fn context_cache_updates_are_atomic_and_reset_to_general_mode() {
    let dir=tempfile::tempdir().unwrap();
    let node=elasticsearch_engine::node::Node::open(dir.path()).unwrap();
    let listener=tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base=format!("http://{}",listener.local_addr().unwrap());
    tokio::spawn(async move { let _=elasticsearch_http::serve_listener(node,listener).await; });
    let client=reqwest::Client::new();
    let url=format!("{base}/_cluster/settings?flat_settings=true");
    let response=client.put(&url).json(&json!({"persistent":{
        "script.max_compilations_rate":"use-context",
        "script.context.score.max_compilations_rate":"100000/1m",
        "script.context.score.cache_max_size":2
    }})).send().await.unwrap();
    assert_eq!(response.status().as_u16(),200,"{}",response.text().await.unwrap());
    let before:Value=client.get(&url).send().await.unwrap().json().await.unwrap();
    let response=client.put(&url).json(&json!({"persistent":{
        "script.max_compilations_rate":"150/5m",
        "script.context.filter.cache_max_size":3
    }})).send().await.unwrap();
    assert_eq!(response.status().as_u16(),400);
    let after:Value=client.get(&url).send().await.unwrap().json().await.unwrap();
    assert_eq!(before,after,"invalid combined update must not alter either setting");
    let response=client.put(&url).json(&json!({"transient":{"script.context.missing.max_compilations_rate":"1/1m"}})).send().await.unwrap();
    assert_eq!(response.status().as_u16(),400);
    let response=client.put(&url).json(&json!({"persistent":{
        "script.context.*":null,"script.max_compilations_rate":null
    }})).send().await.unwrap();
    assert_eq!(response.status().as_u16(),200,"{}",response.text().await.unwrap());
    let after:Value=client.get(&url).send().await.unwrap().json().await.unwrap();
    assert!(after["persistent"].as_object().unwrap().keys().all(|key|!key.starts_with("script.")));
}
