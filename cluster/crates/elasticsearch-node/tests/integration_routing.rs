//! Integrations served by nodes without the data they touch (findings 386,
//! 391, 394): a master-only elected master, one data node and a
//! coordinating-only node.
//! - Logstash pipelines stored on the data node are read through the
//!   coordinating node.
//! - Monitoring export records the cluster's UUID.
//! - Behavioral analytics events posted to the master reach the
//!   collection's data stream on the data node.

mod common;

use common::procs::start_procs_per_node;
use reqwest::Method;
use serde_json::{json, Value};
use std::time::{Duration, Instant};

const MASTER: usize = 0;
const DATA: usize = 1;
const COORD: usize = 2;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn integrations_reach_remote_shards() {
    let p = start_procs_per_node(
        &[("cluster.initial_master_nodes", "n0")],
        &[&[("node.roles", "master")], &[("node.roles", "data,ingest")], &[("node.roles", "remote_cluster_client")]],
    )
    .await;
    assert_eq!(p.master(COORD).await, MASTER);

    // 386: pipelines live in `.logstash` on the data node.
    let pipeline = |n: u32| json!({"description": format!("p{n}"), "last_modified": "2021-01-02T02:50:51.250Z", "pipeline_metadata": {"type": "logstash_pipeline", "version": "1"}, "username": "elastic", "pipeline": format!("input {{}} # {n}"), "pipeline_settings": {"pipeline.workers": 1}});
    for (via, id) in [(MASTER, "a"), (COORD, "b")] {
        let (st, v) = p.call(via, Method::PUT, &format!("/_logstash/pipeline/{id}"), Some(pipeline(1))).await;
        assert!(st == 201 || st == 200, "n{via} {id}: {st} {v}");
    }
    let rows = p.shards(MASTER, ".logstash").await;
    assert!(!rows.is_empty() && rows.iter().all(|r| r["node"] == "n1"), "{rows:?}");
    for via in [MASTER, DATA, COORD] {
        let (st, v) = p.call(via, Method::GET, "/_logstash/pipeline/a", None).await;
        assert_eq!(st, 200, "n{via}: {v}");
        assert_eq!(v["a"]["pipeline"], "input {} # 1", "n{via}: {v}");
        let (st, v) = p.call(via, Method::GET, "/_logstash/pipeline", None).await;
        assert_eq!(st, 200, "n{via}: {v}");
        assert_eq!(v.as_object().map(|o| o.len()), Some(2), "n{via}: {v}");
        let (st, _) = p.call(via, Method::GET, "/_logstash/pipeline/missing", None).await;
        assert_eq!(st, 404, "n{via}");
    }
    let (st, _) = p.call(COORD, Method::DELETE, "/_logstash/pipeline/b", None).await;
    assert_eq!(st, 200);
    let (_, v) = p.call(DATA, Method::GET, "/_logstash/pipeline", None).await;
    assert_eq!(v.as_object().map(|o| o.len()), Some(1), "{v}");

    // 391: the monitoring envelope carries the cluster UUID.
    p.settings(MASTER, "persistent", json!({"xpack.monitoring.collection.enabled": true})).await;
    let body = format!("{}\n{}\n", json!({"index": {"_type": "logstash_stats"}}), json!({"logstash": {"uuid": "ls-1"}}));
    let (st, v) = p.ndjson(COORD, "/_monitoring/bulk?system_id=logstash&system_api_version=7&interval=10s", body).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["errors"], false, "{v}");
    let (_, root) = p.call(COORD, Method::GET, "/", None).await;
    let uuid = root["cluster_uuid"].as_str().unwrap().to_string();
    let _ = p.call(COORD, Method::POST, "/.monitoring-logstash-*/_refresh", None).await;
    let (st, v) = p.call(COORD, Method::POST, "/.monitoring-logstash-*/_search", Some(json!({"query": {"term": {"cluster_uuid": uuid}}}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["total"]["value"], 1, "cluster {uuid}: {v}");

    // 394: events posted through the master-only master reach the data node.
    let (st, v) = p.call(COORD, Method::PUT, "/_application/analytics/shop", None).await;
    assert!(st == 200 || st == 201, "{st} {v}");
    for i in 0..3 {
        let (st, v) = p.call(COORD, Method::POST, "/_application/analytics/shop/event/page_view", Some(json!({"session": {"id": format!("s{i}")}, "user": {"id": "u1"}, "page": {"url": "https://shop.example/a"}}))).await;
        assert_eq!(st, 202, "{v}");
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let _ = p.call(COORD, Method::POST, "/behavioral_analytics-events-shop/_refresh", None).await;
        let (_, v): (u16, Value) = p.call(COORD, Method::GET, "/behavioral_analytics-events-shop/_count", None).await;
        if v["count"] == 3 {
            break;
        }
        assert!(Instant::now() < deadline, "events not stored: {v}");
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}
