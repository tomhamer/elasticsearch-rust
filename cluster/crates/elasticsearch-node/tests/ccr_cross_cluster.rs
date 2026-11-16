//! Cross-cluster replication between two separate rustic clusters, each two
//! `elasticsearch-node` processes: a leader cluster and a follower cluster
//! whose `cluster.remote.leader` seeds point at the leader's transport.
//! No self alias: every leader read, lease and restore goes over the
//! follower's remote-cluster connection to the other cluster.
mod common;
use common::procs::{start_procs, Procs};
use reqwest::Method;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

const TRIAL: [(&str, &str); 1] = [("xpack.license.self_generated.type", "trial")];

async fn clusters(follower_extra: &[(&str, &str)]) -> (Procs, Procs) {
    let leader = start_procs(2, &TRIAL).await;
    let mut extra: Vec<(&str, &str)> = TRIAL.to_vec();
    extra.extend_from_slice(follower_extra);
    let follower = start_procs(2, &extra).await;
    leader.wait_nodes(0, 2, 60).await;
    follower.wait_nodes(0, 2, 60).await;
    let seeds: Vec<String> = leader.p.transport.iter().map(|p| format!("127.0.0.1:{p}")).collect();
    follower.settings(0, "persistent", json!({"cluster.remote.leader.seeds": seeds})).await;
    // Connected before anything follows.
    let f = &follower;
    wait_until(60, "remote cluster [leader] connected", || async move {
        let (_, v) = f.call(0, Method::GET, "/_remote/info", None).await;
        v["leader"]["connected"] == true
    })
    .await;
    (leader, follower)
}

async fn wait_until<F, Fut>(secs: u64, what: &str, mut f: F)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let deadline = Instant::now() + Duration::from_secs(secs);
    while Instant::now() < deadline {
        if f().await {
            return;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    panic!("timed out waiting for {what}");
}

async fn ok(p: &Procs, node: usize, method: Method, path: &str, body: Option<Value>) -> Value {
    let (st, v) = p.call(node, method, path, body).await;
    assert!((200..300).contains(&st), "{path}: {st} {v}");
    v
}

/// `_id` → (`_seq_no`, `_version`, `_source`) of every document.
async fn docs(p: &Procs, node: usize, index: &str) -> BTreeMap<String, (i64, i64, Value)> {
    let _ = p.call(node, Method::POST, &format!("/{index}/_refresh"), None).await;
    let (st, v) = p.call(node, Method::POST, &format!("/{index}/_search"), Some(json!({"size": 10000, "seq_no_primary_term": true, "version": true, "query": {"match_all": {}}}))).await;
    if st != 200 {
        return BTreeMap::new();
    }
    v["hits"]["hits"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|h| (h["_id"].as_str().unwrap_or_default().to_string(), (h["_seq_no"].as_i64().unwrap_or(-1), h["_version"].as_i64().unwrap_or(-1), h["_source"].clone())))
        .collect()
}

async fn converged(leader: &Procs, follower: &Procs, l: &str, f: &str, secs: u64) {
    let deadline = Instant::now() + Duration::from_secs(secs);
    let mut last = (BTreeMap::new(), BTreeMap::new());
    while Instant::now() < deadline {
        // Read from nodes that are up (a test may have one of each down).
        let (ln, fnode) = (leader.live()[0], *follower.live().last().unwrap());
        let (a, b) = (docs(leader, ln, l).await, docs(follower, fnode, f).await);
        if a == b && !a.is_empty() {
            return;
        }
        last = (a, b);
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    panic!("follower [{f}] did not converge on leader [{l}]:\nleader {:?}\nfollower {:?}", last.0, last.1);
}

async fn create_leader(leader: &Procs, index: &str) {
    ok(leader, 0, Method::PUT, &format!("/{index}"), Some(json!({
        "settings": {"number_of_shards": 2, "number_of_replicas": 1, "index.soft_deletes.enabled": true},
        "mappings": {"properties": {"n": {"type": "long"}, "tag": {"type": "keyword"}}}
    })))
    .await;
    leader.wait_health(0, index, "green", 60).await;
}

async fn bulk(p: &Procs, index: &str, ids: std::ops::Range<u32>, tag: &str) {
    bulk_on(p, 0, index, ids, tag).await
}

async fn bulk_on(p: &Procs, node: usize, index: &str, ids: std::ops::Range<u32>, tag: &str) {
    let mut body = String::new();
    for i in ids {
        body.push_str(&format!("{{\"index\":{{\"_index\":\"{index}\",\"_id\":\"{i}\"}}}}\n{{\"n\":{i},\"tag\":\"{tag}\"}}\n"));
    }
    // Right after a leader node goes away, writes routed to its primaries
    // fail until the master promotes their replicas: retry (same ids).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(90);
    loop {
        let (st, v) = p.ndjson(node, "/_bulk?refresh=true", body.clone()).await;
        if st == 200 && v["errors"] == false {
            return;
        }
        assert!(std::time::Instant::now() < deadline, "{st} {v}");
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
}

async fn follow(follower: &Procs, f: &str, l: &str) {
    let v = ok(follower, 0, Method::PUT, &format!("/{f}/_ccr/follow?wait_for_active_shards=1"), Some(json!({"remote_cluster": "leader", "leader_index": l}))).await;
    assert_eq!(v, json!({"follow_index_created": true, "follow_index_shards_acked": true, "index_following_started": true}));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn follower_converges_then_pauses_resumes_and_unfollows() {
    let (leader, follower) = clusters(&[]).await;
    create_leader(&leader, "logs").await;
    bulk(&leader, "logs", 0..50, "a").await;
    follow(&follower, "logs-copy", "logs").await;
    follower.wait_health(0, "logs-copy", "green", 60).await;
    // The restore copied the leader's documents with their seq_nos and versions.
    converged(&leader, &follower, "logs", "logs-copy", 60).await;

    // Index, update, delete on the leader: the follower follows.
    bulk(&leader, "logs", 40..80, "b").await;
    for id in [1, 2, 3] {
        ok(&leader, 1, Method::DELETE, &format!("/logs/_doc/{id}?refresh=true"), None).await;
    }
    // A mapping update on the leader reaches the follower.
    ok(&leader, 0, Method::PUT, "/logs/_mapping", Some(json!({"properties": {"extra": {"type": "keyword"}}}))).await;
    ok(&leader, 0, Method::PUT, "/logs/_doc/extra?refresh=true", Some(json!({"n": 1000, "extra": "x"}))).await;
    converged(&leader, &follower, "logs", "logs-copy", 90).await;
    let m = ok(&follower, 1, Method::GET, "/logs-copy/_mapping", None).await;
    assert_eq!(m["logs-copy"]["mappings"]["properties"]["extra"]["type"], "keyword", "{m}");

    // Read-only for clients.
    let (st, v) = follower.call(0, Method::PUT, "/logs-copy/_doc/direct", Some(json!({"n": 1}))).await;
    assert_eq!(st, 403, "{v}");
    assert_eq!(v["error"]["type"], "status_exception", "{v}");

    // Info and stats.
    let info = ok(&follower, 1, Method::GET, "/logs-copy/_ccr/info", None).await;
    assert_eq!(info["follower_indices"][0]["status"], "active", "{info}");
    assert_eq!(info["follower_indices"][0]["remote_cluster"], "leader", "{info}");
    let stats = ok(&follower, 1, Method::GET, "/logs-copy/_ccr/stats", None).await;
    assert_eq!(stats["indices"][0]["shards"].as_array().map(Vec::len), Some(2), "{stats}");
    let read: i64 = stats["indices"][0]["shards"].as_array().unwrap().iter().map(|s| s["operations_written"].as_i64().unwrap_or(0)).sum();
    assert!(read > 0, "{stats}");

    // Pause: the leader moves on alone; resume: the follower catches up.
    ok(&follower, 0, Method::POST, "/logs-copy/_ccr/pause_follow", None).await;
    let info = ok(&follower, 0, Method::GET, "/logs-copy/_ccr/info", None).await;
    assert_eq!(info["follower_indices"][0]["status"], "paused", "{info}");
    bulk(&leader, "logs", 100..120, "c").await;
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert_eq!(docs(&follower, 0, "logs-copy").await.get("110"), None, "a paused follower must not read");
    ok(&follower, 0, Method::POST, "/logs-copy/_ccr/resume_follow", Some(json!({}))).await;
    converged(&leader, &follower, "logs", "logs-copy", 90).await;

    // Unfollow needs a paused, closed follower, then the index is writable.
    let (st, v) = follower.call(0, Method::POST, "/logs-copy/_ccr/unfollow", None).await;
    assert_eq!(st, 400, "{v}");
    ok(&follower, 0, Method::POST, "/logs-copy/_ccr/pause_follow", None).await;
    ok(&follower, 0, Method::POST, "/logs-copy/_close", None).await;
    ok(&follower, 0, Method::POST, "/logs-copy/_ccr/unfollow", None).await;
    ok(&follower, 0, Method::POST, "/logs-copy/_open?wait_for_active_shards=all", None).await;
    ok(&follower, 0, Method::PUT, "/logs-copy/_doc/direct?refresh=true", Some(json!({"n": 1}))).await;
    let info = ok(&follower, 0, Method::GET, "/logs-copy/_ccr/info", None).await;
    assert_eq!(info["follower_indices"], json!([]), "{info}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn auto_follow_follows_new_matching_indices_only() {
    let (leader, follower) = clusters(&[]).await;
    create_leader(&leader, "metrics-old").await;
    ok(&follower, 0, Method::PUT, "/_ccr/auto_follow/metrics", Some(json!({
        "remote_cluster": "leader", "leader_index_patterns": ["metrics-*"], "leader_index_exclusion_patterns": ["metrics-excluded"],
        "follow_index_pattern": "{{leader_index}}-copy"
    })))
    .await;
    let p = ok(&follower, 1, Method::GET, "/_ccr/auto_follow/metrics", None).await;
    assert_eq!(p["patterns"][0]["pattern"]["leader_index_exclusion_patterns"], json!(["metrics-excluded"]), "{p}");
    create_leader(&leader, "metrics-new").await;
    create_leader(&leader, "metrics-excluded").await;
    bulk(&leader, "metrics-new", 0..20, "x").await;
    let f = &follower;
    wait_until(90, "metrics-new-copy followed", || async move {
        let (st, v) = f.call(0, Method::GET, "/metrics-new-copy/_ccr/info", None).await;
        st == 200 && v["follower_indices"][0]["status"] == "active"
    })
    .await;
    converged(&leader, &follower, "metrics-new", "metrics-new-copy", 90).await;
    // Existing (when the pattern was put) and excluded indices are not followed.
    for absent in ["metrics-old-copy", "metrics-excluded-copy"] {
        let (st, _) = follower.call(0, Method::GET, &format!("/{absent}"), None).await;
        assert_eq!(st, 404, "{absent} must not be followed");
    }
    let stats = ok(&follower, 0, Method::GET, "/_ccr/stats", None).await;
    assert!(stats["auto_follow_stats"]["number_of_successful_follow_indices"].as_u64().unwrap_or(0) >= 1, "{stats}");
    // Pause, resume, delete the pattern.
    ok(&follower, 0, Method::POST, "/_ccr/auto_follow/metrics/pause", None).await;
    let p = ok(&follower, 0, Method::GET, "/_ccr/auto_follow/metrics", None).await;
    assert_eq!(p["patterns"][0]["pattern"]["active"], false, "{p}");
    ok(&follower, 0, Method::POST, "/_ccr/auto_follow/metrics/resume", None).await;
    ok(&follower, 0, Method::DELETE, "/_ccr/auto_follow/metrics", None).await;
    let (st, _) = follower.call(0, Method::GET, "/_ccr/auto_follow/metrics", None).await;
    assert_eq!(st, 404);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn leader_and_follower_node_restarts_lose_no_operations() {
    let (mut leader, mut follower) = clusters(&[]).await;
    create_leader(&leader, "events").await;
    bulk(&leader, "events", 0..30, "a").await;
    follow(&follower, "events-copy", "events").await;
    follower.wait_health(0, "events-copy", "green", 60).await;
    converged(&leader, &follower, "events", "events-copy", 60).await;

    // A leader node goes away while the leader takes writes (its replicas
    // are promoted, keeping the follower's lease), then comes back.
    // A two-node cluster keeps its master only while the master stays: the
    // node that goes away is the other one.
    let victim = 1 - leader.master(0).await;
    let survivor = 1 - victim;
    leader.kill(victim);
    bulk_on(&leader, survivor, "events", 30..60, "b").await;
    converged(&leader, &follower, "events", "events-copy", 120).await;
    leader.spawn(victim, &[]).await;
    leader.wait_nodes(survivor, 2, 120).await;
    leader.wait_health(survivor, "events", "green", 120).await;
    bulk(&leader, "events", 60..90, "c").await;
    converged(&leader, &follower, "events", "events-copy", 120).await;

    // A follower node goes away (its primaries' tasks move to the promoted
    // copies on the other node), then comes back.
    let fvictim = 1 - follower.master(0).await;
    let fsurvivor = 1 - fvictim;
    follower.kill(fvictim);
    bulk(&leader, "events", 90..120, "d").await;
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        if docs(&leader, 0, "events").await == docs(&follower, fsurvivor, "events-copy").await {
            break;
        }
        assert!(Instant::now() < deadline, "follower did not converge with a node down");
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    follower.spawn(fvictim, &[]).await;
    follower.wait_nodes(fsurvivor, 2, 120).await;
    follower.wait_health(fsurvivor, "events-copy", "green", 120).await;
    bulk(&leader, "events", 120..150, "e").await;
    converged(&leader, &follower, "events", "events-copy", 120).await;
    assert_eq!(docs(&follower, 1, "events-copy").await.len(), 150);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn retention_lease_keeps_history_while_the_follower_lags() {
    let (leader, follower) = clusters(&[]).await;
    create_leader(&leader, "audit").await;
    bulk(&leader, "audit", 0..10, "a").await;
    follow(&follower, "audit-copy", "audit").await;
    converged(&leader, &follower, "audit", "audit-copy", 60).await;
    // The follower lags: the leader writes, updates and deletes, and flushes
    // (trimming every translog generation no lease retains).
    ok(&follower, 0, Method::POST, "/audit-copy/_ccr/pause_follow", None).await;
    bulk(&leader, "audit", 10..200, "b").await;
    bulk(&leader, "audit", 0..50, "updated").await;
    for id in 50..60 {
        ok(&leader, 0, Method::DELETE, &format!("/audit/_doc/{id}?refresh=true"), None).await;
    }
    for _ in 0..3 {
        ok(&leader, 0, Method::POST, "/audit/_flush?force=true", None).await;
    }
    ok(&leader, 0, Method::POST, "/audit/_forcemerge?max_num_segments=1", None).await;
    // The follower's lease on the leader kept the operations it needs.
    ok(&follower, 0, Method::POST, "/audit-copy/_ccr/resume_follow", Some(json!({}))).await;
    converged(&leader, &follower, "audit", "audit-copy", 120).await;
    let stats = ok(&follower, 0, Method::GET, "/audit-copy/_ccr/stats", None).await;
    for s in stats["indices"][0]["shards"].as_array().unwrap() {
        assert!(s.get("fatal_exception").is_none(), "{stats}");
    }
    // Forget the follower on the leader: its leases go.
    let st = ok(&follower, 0, Method::GET, "/_cluster/state/metadata/audit-copy", None).await;
    let (follower_name, follower_uuid) = (st["cluster_name"].as_str().unwrap().to_string(), st["metadata"]["indices"]["audit-copy"]["settings"]["index"]["uuid"].as_str().unwrap_or_default().to_string());
    let r = ok(&leader, 1, Method::POST, "/audit/_ccr/forget_follower", Some(json!({
        "follower_cluster": follower_name, "follower_index": "audit-copy", "follower_index_uuid": follower_uuid, "leader_remote_cluster": "leader"
    })))
    .await;
    assert_eq!(r["_shards"]["total"], 2, "{r}");
}

/// ILM `unfollow` (ES `UnfollowAction`): a follower waits while its leader
/// still accepts writes, then — once the leader marks indexing complete and
/// every follow task has caught up — is paused, closed, converted to a
/// regular index and reopened.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn ilm_unfollow_waits_for_the_leader_then_converts_the_follower() {
    let (leader, follower) = clusters(&[]).await;
    follower.settings(0, "persistent", json!({"indices.lifecycle.poll_interval": "1s"})).await;
    ok(&follower, 0, Method::PUT, "/_ilm/policy/unfollow-now", Some(json!({"policy": {"phases": {"hot": {"actions": {"unfollow": {}}}}}}))).await;
    create_leader(&leader, "ilm-leader").await;
    bulk(&leader, "ilm-leader", 0..20, "a").await;
    let v = ok(&follower, 0, Method::PUT, "/ilm-follower/_ccr/follow?wait_for_active_shards=1", Some(json!({
        "remote_cluster": "leader", "leader_index": "ilm-leader", "settings": {"index.lifecycle.name": "unfollow-now"}
    }))).await;
    assert_eq!(v["index_following_started"], true, "{v}");
    follower.wait_health(0, "ilm-follower", "green", 60).await;
    converged(&leader, &follower, "ilm-leader", "ilm-follower", 60).await;

    // The leader still accepts writes: ILM waits, the index keeps following.
    let f = &follower;
    let step = || async move {
        let (_, v) = f.call(0, Method::GET, "/ilm-follower/_ilm/explain", None).await;
        v["indices"]["ilm-follower"]["step"].as_str().unwrap_or_default().to_string()
    };
    wait_until(60, "ILM waiting for the leader's indexing to complete", || async { step().await == "wait-for-indexing-complete" }).await;
    bulk(&leader, "ilm-leader", 20..40, "b").await;
    converged(&leader, &follower, "ilm-leader", "ilm-follower", 60).await;
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert_eq!(step().await, "wait-for-indexing-complete");
    let info = ok(&follower, 0, Method::GET, "/ilm-follower/_ccr/info", None).await;
    assert_eq!(info["follower_indices"][0]["status"], "active", "{info}");

    // Last writes, then the leader is done: the follower catches up first,
    // then stops being a follower.
    bulk(&leader, "ilm-leader", 40..60, "c").await;
    ok(&leader, 0, Method::PUT, "/ilm-leader/_settings", Some(json!({"index.lifecycle.indexing_complete": true}))).await;
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let (_, info) = f.call(0, Method::GET, "/ilm-follower/_ccr/info", None).await;
        // Explained throughout, the closed follower included (ES explains a
        // closed index named explicitly).
        let (st, explain) = f.call(0, Method::GET, "/ilm-follower/_ilm/explain", None).await;
        assert_eq!(st, 200, "{explain}");
        if info["follower_indices"] == json!([]) && explain["indices"]["ilm-follower"]["step"] == "complete" {
            break;
        }
        assert!(Instant::now() < deadline, "ILM unfollow did not complete: {explain} {info}");
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    follower.wait_health(0, "ilm-follower", "green", 60).await;
    let (l, fo) = (docs(&leader, 0, "ilm-leader").await, docs(&follower, 0, "ilm-follower").await);
    assert_eq!(fo.len(), 60, "every leader operation was replicated before unfollowing");
    assert_eq!(l, fo);
    let s = ok(&follower, 0, Method::GET, "/ilm-follower/_settings?flat_settings=true", None).await;
    assert!(s["ilm-follower"]["settings"].get("index.xpack.ccr.following_index").is_none(), "{s}");
    // A regular index now.
    ok(&follower, 0, Method::PUT, "/ilm-follower/_doc/direct?refresh=true", Some(json!({"n": 1}))).await;
    // `_ilm/explain` explains a closed index named explicitly (ES).
    ok(&follower, 0, Method::POST, "/ilm-follower/_close", None).await;
    let v = ok(&follower, 0, Method::GET, "/ilm-follower/_ilm/explain", None).await;
    assert_eq!(v["indices"]["ilm-follower"]["managed"], true, "{v}");
    assert_eq!(v["indices"]["ilm-follower"]["policy"], "unfollow-now", "{v}");
    ok(&follower, 0, Method::POST, "/ilm-follower/_open", None).await;
}

/// A follower all the way through ILM: the implicit `unfollow` before
/// `rollover` (#575) converts it once the leader is done, `rollover`
/// passes (indexing complete), the cold `searchable_snapshot` snapshots
/// the converted index and mounts it, and the frozen phase's implicit
/// `unfollow` is skipped (the mounted index is no follower).
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn follower_through_ilm_unfollow_rollover_and_searchable_snapshot() {
    let repo = common::scratch_dir("ccr-ilm-e2e-repo");
    std::fs::create_dir_all(&repo).unwrap();
    let repo_path = repo.to_string_lossy().into_owned();
    let (leader, follower) = clusters(&[("path.repo", repo_path.as_str())]).await;
    follower.settings(0, "persistent", json!({"indices.lifecycle.poll_interval": "1s"})).await;
    ok(&follower, 0, Method::PUT, "/_snapshot/repo", Some(json!({"type": "fs", "settings": {"location": repo_path}}))).await;
    ok(&follower, 0, Method::PUT, "/_ilm/policy/e2e", Some(json!({"policy": {"phases": {
        "hot": {"actions": {"rollover": {"max_docs": 1}}},
        "cold": {"min_age": "0ms", "actions": {"searchable_snapshot": {"snapshot_repository": "repo", "force_merge_index": false}}},
        "frozen": {"min_age": "0ms", "actions": {"searchable_snapshot": {"snapshot_repository": "repo"}}}
    }}}))).await;
    // The leader: written through its alias, the rollover alias replicated.
    ok(&leader, 0, Method::PUT, "/e2e-000001", Some(json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 1, "index.soft_deletes.enabled": true, "index.lifecycle.rollover_alias": "e2e"},
        "aliases": {"e2e": {"is_write_index": true}},
        "mappings": {"properties": {"n": {"type": "long"}, "tag": {"type": "keyword"}}}
    }))).await;
    leader.wait_health(0, "e2e-000001", "green", 60).await;
    bulk(&leader, "e2e-000001", 0..30, "a").await;
    let v = ok(&follower, 0, Method::PUT, "/e2e-follower/_ccr/follow?wait_for_active_shards=1", Some(json!({
        "remote_cluster": "leader", "leader_index": "e2e-000001", "settings": {"index.lifecycle.name": "e2e"}
    }))).await;
    assert_eq!(v["index_following_started"], true, "{v}");
    converged(&leader, &follower, "e2e-000001", "e2e-follower", 60).await;
    let f = &follower;
    let explain = |index: &'static str| async move {
        let (st, v) = f.call(0, Method::GET, &format!("/{index}/_ilm/explain"), None).await;
        (st, v["indices"][index].clone())
    };
    // The implicit unfollow comes first and waits for the leader.
    wait_until(60, "the implicit unfollow waiting for the leader", || async {
        let (_, e) = explain("e2e-follower").await;
        e["action"] == "unfollow" && e["step"] == "wait-for-indexing-complete"
    })
    .await;
    // The leader is done (its own rollover would set this).
    bulk(&leader, "e2e-000001", 30..40, "b").await;
    ok(&leader, 0, Method::PUT, "/e2e-000001/_settings", Some(json!({"index.lifecycle.indexing_complete": true}))).await;

    // Converted, rolled over (indexing complete), snapshotted and mounted.
    let deadline = Instant::now() + Duration::from_secs(240);
    let mounted = loop {
        let (_, cat) = follower.call(0, Method::GET, "/_cat/indices/restored-*,partial-*?format=json&h=index", None).await;
        let names: Vec<String> = cat.as_array().into_iter().flatten().filter_map(|r| r["index"].as_str().map(str::to_string)).collect();
        if let Some(n) = names.iter().find(|n| n.starts_with("partial-")) {
            break n.clone();
        }
        let (_, e) = explain("e2e-follower").await;
        assert!(Instant::now() < deadline, "the follower did not reach the frozen mount: {names:?} {e}");
        assert_ne!(e["step"], "ERROR", "{e}");
        tokio::time::sleep(Duration::from_millis(500)).await;
    };
    assert!(mounted.ends_with("e2e-follower"), "{mounted}");
    wait_until(120, "the frozen phase to complete", || async {
        let (_, v) = f.call(0, Method::GET, &format!("/{mounted}/_ilm/explain"), None).await;
        let e = &v["indices"][mounted.as_str()];
        e["phase"] == "frozen" && e["action"] == "complete"
    })
    .await;
    follower.wait_health(0, &mounted, "green", 60).await;

    // No follower state on the mounted index, which holds every operation.
    let s = ok(&follower, 0, Method::GET, &format!("/{mounted}/_settings?flat_settings=true"), None).await;
    let settings = &s[mounted.as_str()]["settings"];
    assert!(settings.get("index.xpack.ccr.following_index").is_none(), "{s}");
    assert_eq!(settings["index.store.type"], "snapshot", "{s}");
    let info = ok(&follower, 0, Method::GET, "/_all/_ccr/info", None).await;
    assert_eq!(info["follower_indices"], json!([]), "{info}");
    let (st, v) = follower.call(0, Method::POST, &format!("/{mounted}/_count"), None).await;
    assert_eq!((st, v["count"].clone()), (200, json!(40)), "{v}");
    // The snapshot is of the converted index (no clone: force_merge_index false).
    assert_eq!(settings["index.store.snapshot.index_name"], "e2e-follower", "{s}");
    let snapshot = settings["index.store.snapshot.snapshot_name"].as_str().unwrap_or_default().to_string();
    let snap = ok(&follower, 0, Method::GET, &format!("/_snapshot/repo/{snapshot}"), None).await;
    assert_eq!(snap["snapshots"][0]["indices"], json!(["e2e-follower"]), "{snap}");
    assert_eq!(snap["snapshots"][0]["state"], "SUCCESS", "{snap}");

    // The lifecycle history: the hot phase's implicit unfollow converted the
    // follower; the frozen phase's was skipped at its branch.
    let _ = follower.call(0, Method::POST, "/ilm-history-*/_refresh?expand_wildcards=all", None).await;
    let deadline = Instant::now() + Duration::from_secs(60);
    let steps = loop {
        let (_, h) = follower.call(0, Method::POST, "/ilm-history-*/_search?expand_wildcards=all", Some(json!({"size": 1000, "query": {"term": {"policy": "e2e"}}}))).await;
        let steps: Vec<(String, String, String)> = h["hits"]["hits"].as_array().into_iter().flatten().map(|d| {
            let s = &d["_source"]["state"];
            (s["phase"].as_str().unwrap_or_default().to_string(), s["action"].as_str().unwrap_or_default().to_string(), s["step"].as_str().unwrap_or_default().to_string())
        }).collect();
        if steps.iter().any(|(p, a, _)| p == "frozen" && a != "unfollow") {
            break steps;
        }
        assert!(Instant::now() < deadline, "no frozen phase history: {steps:?}");
        tokio::time::sleep(Duration::from_millis(500)).await;
        let _ = follower.call(0, Method::POST, "/ilm-history-*/_refresh?expand_wildcards=all", None).await;
    };
    let has = |p: &str, a: &str, s: &str| steps.iter().any(|(x, y, z)| x == p && y == a && z == s);
    assert!(has("hot", "unfollow", "unfollow-follower-index") && has("hot", "unfollow", "open-follower-index"), "{steps:?}");
    let frozen_unfollow: Vec<&String> = steps.iter().filter(|(p, a, _)| p == "frozen" && a == "unfollow").map(|(_, _, s)| s).collect();
    assert!(frozen_unfollow.iter().all(|s| *s == "branch-check-unfollow-prerequisites"), "the frozen unfollow ran past its branch: {steps:?}");
    // The frozen mount came from the cold mount's snapshot (no new snapshot of a non-follower copy).
    let fs = ok(&follower, 0, Method::GET, "/_snapshot/repo/_all", None).await;
    assert!(fs["snapshots"].as_array().is_some_and(|a| a.iter().all(|s| s["indices"] == json!(["e2e-follower"]))), "{fs}");
}
