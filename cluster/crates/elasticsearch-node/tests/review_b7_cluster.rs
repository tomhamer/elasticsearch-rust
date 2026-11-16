//! Production review batch 7 on node processes ([`common::procs`]):
//! - SIDE 003: a shard engine that fails to open while its index is
//!   created is retried; once storage recovers the copy starts without an
//!   unrelated cluster change;
//! - SIDE 015: shard-started reports refused by the master more than the
//!   old retry budget (40 sends) are still delivered once it accepts them.
mod common;
use common::procs::start_procs_env;
use reqwest::Method;
use serde_json::json;
use std::time::{Duration, Instant};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_shard_engine_that_failed_to_open_recovers_by_itself() {
    let trigger = std::env::temp_dir().join(format!("rustic-b7-fail-shard-engine-{}", std::process::id()));
    std::fs::write(&trigger, b"1").unwrap();
    let pl = start_procs_env(1, &[], &[("RUSTIC_TEST_FAIL_SHARD_ENGINE_OPEN", trigger.to_str().unwrap())]).await;
    let via = 0;
    let (st, v) = pl.call(via, Method::PUT, "/two", Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0}}))).await;
    assert_eq!(st, 200, "{v}");
    // Shard 1's engine cannot open: the index does not go green.
    tokio::time::sleep(Duration::from_secs(3)).await;
    let (_, h) = pl.call(via, Method::GET, "/_cluster/health/two", None).await;
    assert_ne!(h["status"], "green", "{h}");
    // Storage recovers; nothing else changes in the cluster.
    std::fs::remove_file(&trigger).unwrap();
    pl.wait_health(via, "two", "green", 60).await;
    let (st, v) = pl.call(via, Method::PUT, "/two/_doc/a?refresh=true&routing=x", Some(json!({"n": 1}))).await;
    assert_eq!(st, 201, "{v}");
    for r in 0..8 {
        let (st, v) = pl.call(via, Method::PUT, &format!("/two/_doc/r{r}?refresh=true"), Some(json!({"n": r}))).await;
        assert_eq!(st, 201, "{v}");
    }
    let (_, c) = pl.call(via, Method::GET, "/two/_count", None).await;
    assert_eq!(c["count"], 9, "{c}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shard_started_reports_outlast_master_refusals() {
    // Every node refuses its first 45 shard-started sends (more than the
    // 40 the report used to give up after).
    let pl = start_procs_env(2, &[], &[("RUSTIC_TEST_REFUSE_SHARD_STARTED", "45")]).await;
    let via = 0;
    let t0 = Instant::now();
    let (st, v) = pl.call(via, Method::PUT, "/late", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 1}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(via, "late", "green", 150).await;
    assert!(t0.elapsed() > Duration::from_secs(15), "the refusals were exercised ({:?})", t0.elapsed());
    let rows = pl.shards(via, "late").await;
    assert!(rows.iter().all(|r| r["state"] == "STARTED"), "{rows:?}");
}

const PW: &str = "test-password";

fn keystore_conf(tag: &str) -> String {
    let conf = std::env::temp_dir().join(format!("rustic-b7-ks-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&conf).unwrap();
    let mut ks = elasticsearch_node::keystore::KeyStore::create().unwrap();
    ks.set_string("bootstrap.password", PW).unwrap();
    ks.save(&conf, "").unwrap();
    conf.to_string_lossy().to_string()
}

/// The first `PUT _security/user` right after a 2-node secured cluster
/// starts creates `.security-7` and answers (it timed out about every
/// second start with "security index creation was not applied
/// everywhere"): repeated over fresh clusters.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn first_put_user_after_start_creates_the_security_index() {
    let secured = [("xpack.license.self_generated.type", "trial"), ("xpack.security.enabled", "true"), ("xpack.security.authc.password_hashing.algorithm", "pbkdf2_1000")];
    for round in 0..6 {
        let conf = keystore_conf(&format!("r{round}"));
        let pl = common::procs::start_procs_secured_env(2, &secured, &[("ES_PATH_CONF", conf.as_str())], "elastic", PW).await;
        let t0 = Instant::now();
        let (st, v) = pl.call(0, Method::POST, "/_security/user/joe", Some(json!({"password": "joe-password", "roles": ["superuser"]}))).await;
        assert_eq!(st, 200, "round {round} after {:?}: {v}", t0.elapsed());
        assert!(t0.elapsed() < Duration::from_secs(20), "round {round}: {:?}", t0.elapsed());
        let (st, v) = pl.call(1, Method::POST, "/_security/user/jane", Some(json!({"password": "jane-password", "roles": ["superuser"]}))).await;
        assert_eq!(st, 200, "round {round}: {v}");
    }
}
