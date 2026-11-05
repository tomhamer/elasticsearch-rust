//! Persisted settings compatibility with the real baseline executable.
//! This is a full cluster restart, separate from rolling-under-load coverage.
mod common;
use common::procs::start_procs_with_binaries;
use reqwest::Method;
use serde_json::json;
use std::path::PathBuf;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn old_binary_invalid_index_setting_is_archived_after_upgrade() {
    let old = PathBuf::from(std::env::var_os("PR3_OLD_BINARY")
        .expect("PR3_OLD_BINARY must name baseline64ea4834; old/new settings coverage cannot be skipped"));
    let new = PathBuf::from(env!("CARGO_BIN_EXE_elasticsearch-node"));
    assert_ne!(std::fs::canonicalize(&old).unwrap(), std::fs::canonicalize(&new).unwrap());
    assert!(std::fs::read(&old).unwrap() != std::fs::read(&new).unwrap());
    let mut p = start_procs_with_binaries(&[], &[old.clone(), old]).await;
    // Baseline64ea4834 accepted this value before the PR3 regex limit validator.
    let (status, body) = p.call(0, Method::PUT, "/legacy", Some(json!({
        "settings":{"number_of_shards":1,"number_of_replicas":1,"max_regex_length":"broken", "query.default_field.0":"tag"},
        "mappings":{"properties":{"tag":{"type":"keyword"}}}
    }))).await;
    assert_eq!(status, 200, "baseline must actually persist the legacy setting: {body}");
    let (status, body) = p.call(0, Method::PUT, "/legacy/_doc/1?refresh=true&wait_for_active_shards=all", Some(json!({"tag":"kept"}))).await;
    assert_eq!(status, 201, "{body}");
    let (status, body) = p.call(0, Method::GET, "/_cluster/health/legacy?wait_for_status=green&timeout=30s", None).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["status"], "green");
    for node in 0..2 { p.stop(node, 10).await; p.set_binary(node, &new); }
    for node in 0..2 { p.spawn(node, &[]).await; }
    p.wait_nodes(0, 2, 90).await;
    let (status, body) = p.call(0, Method::GET, "/_cluster/health/legacy?wait_for_status=green&timeout=90s", None).await;
    assert_eq!(status, 200, "upgraded recovery: {body}");
    assert_eq!(body["status"], "green", "both retained copies must recover: {body}");
    for node in 0..2 {
        let (status, body) = p.call(node, Method::GET, "/legacy/_settings?flat_settings=true", None).await;
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["legacy"]["settings"]["archived.index.max_regex_length"], "broken", "{body}");
        assert!(body["legacy"]["settings"].get("index.max_regex_length").is_none(), "{body}");
        assert_eq!(body["legacy"]["settings"]["index.query.default_field"], json!(["tag"]));
        assert!(body["legacy"]["settings"].get("index.query.default_field.0").is_none(), "{body}");
        let log = std::fs::read_to_string(p.log_path(node)).unwrap();
        assert!(log.contains("deprecated old-format indexed-list setting"), "n{node}: {log}");
        assert!(log.contains("archiving invalid persisted index setting") && log.contains("index.max_regex_length"), "n{node}: {log}");
    }
    let (status, body) = p.call(0, Method::GET, "/legacy/_doc/1", None).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["_source"]["tag"], "kept");
    let (status, body) = p.call(0, Method::PUT, "/legacy/_settings", Some(json!({"index.max_regex_length":"broken"}))).await;
    assert_eq!(status, 400, "new invalid requests are rejected, not archived: {body}");
    let (status, body) = p.call(0, Method::PUT, "/legacy/_settings", Some(json!({"archived.*":null}))).await;
    assert_eq!(status, 200, "open-index archived reset: {body}");
    let (status, body) = p.call(0, Method::GET, "/legacy/_settings?flat_settings=true", None).await;
    assert_eq!(status, 200, "{body}");
    assert!(body["legacy"]["settings"].as_object().unwrap().keys().all(|key| !key.starts_with("archived.")));
}
