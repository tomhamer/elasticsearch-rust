//! The two-node harness keeps the nodes' data directory for the life of the
//! test process. The nodes run on the test's runtime, which (with the
//! coordinator persisting cluster state) outlives the `Cluster` value: a
//! directory deleted with it made the next persist fail, and a node that
//! cannot persist its coordination state exits the process, so the test
//! binary died with status 1 and no output (`async_two_nodes` flaked so).

mod common;

use common::{call, start};
use reqwest::Method;
use serde_json::json;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn nodes_keep_persisting_after_the_cluster_value_is_dropped() {
    let (c, p) = {
        let cl = start(&[]).await;
        (cl.c.clone(), cl.p)
    };
    // The nodes are still running: every settings update is a new cluster
    // state each of them persists.
    for i in 0..5 {
        let (st, v) = call(&c, p, i % 2, Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"cluster.routing.allocation.node_concurrent_recoveries": i + 2}}))).await;
        assert_eq!(st, 200, "{v}");
    }
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}
