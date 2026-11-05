//! Internal CCR state-machine tests using an explicitly injected local adapter.
//! They do not exercise remote transport or establish CCS/CCR API parity.
use super::*;
use crate::hooks::changes::ChangeOp;
use crate::index::WriteReq;
use crate::shard::WriteCond;
use std::time::{Duration, Instant};

fn install_internal_local_adapter(node: &Arc<Node>) {
    *service(node).connector.write() = |node, alias| {
        if alias == "local" { Ok(LeaderCluster::Local(node.clone())) }
        else { Err(remote::no_such_remote(alias)) }
    };
}

fn node() -> (tempfile::TempDir, Arc<Node>) {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.node_settings.write().insert("xpack.license.self_generated.type".into(), json!("trial"));
    crate::license::start(&node);
    install_internal_local_adapter(&node);
    node.cluster_settings.write().0.insert("cluster.remote.local.seeds".into(), json!(remote::TRANSPORT_ADDRESS));
    (tmp, node)
}

fn put(svc: &IndexService, id: &str, body: &str) {
    svc.write(vec![WriteReq::Index { id: Some(id.into()), source: body.into(), routing: None, cond: WriteCond::default() }], true).remove(0).unwrap();
}

fn wait_for(mut f: impl FnMut() -> bool) {
    let t0 = Instant::now();
    while !f() {
        assert!(t0.elapsed() < Duration::from_secs(20), "timed out");
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn follow(node: &Arc<Node>, follower: &str, leader: &str) {
    let req = PutFollowRequest::parse(follower, &json!({"remote_cluster": "local", "leader_index": leader})).unwrap();
    let req = PutFollowRequest { wait_for_active_shards: true, ..req };
    assert_eq!(put_follow(node, &req).unwrap(), (true, true, true));
}

#[test]
fn defaults_resolve_like_es() {
    let mut m = Map::new();
    FollowParams::default().resolve().fragment(&mut m);
    assert_eq!(
        Value::Object(m),
        json!({
            "max_read_request_operation_count": 5120, "max_write_request_operation_count": 5120,
            "max_outstanding_read_requests": 12, "max_outstanding_write_requests": 9,
            "max_read_request_size": "32mb", "max_write_request_size": "9223372036854775807b",
            "max_write_buffer_count": 2147483647, "max_write_buffer_size": "512mb",
            "max_retry_delay": "500ms", "read_poll_timeout": "1m"
        })
    );
}

#[test]
fn validation_messages() {
    let r = PutFollowRequest::parse("f", &json!({"max_read_request_operation_count": 0, "max_retry_delay": "10m"})).unwrap();
    let e = r.validate().unwrap_err();
    assert_eq!(e.kind, "action_request_validation_exception");
    assert_eq!(
        e.reason,
        "Validation Failed: 1: max_read_request_operation_count must be larger than 0;2: [max_retry_delay] must be less than [5m] but was [10m];3: remote_cluster is missing;4: leader_index is missing;"
    );
}

#[test]
fn internal_adapter_follower_replicates_ops_with_leader_seq_nos_and_rejects_direct_writes() {
    let (_t, node) = node();
    let leader = node.create_index("leader", &json!({"settings": {"number_of_shards": 2}})).unwrap();
    for i in 0..10 {
        put(&leader, &format!("d{i}"), &format!("{{\"n\":{i}}}"));
    }
    follow(&node, "follower", "leader");
    let follower = node.get_index("follower").unwrap();
    assert_eq!(follower.setting(FOLLOWING_INDEX_SETTING).as_deref(), Some("true"));
    assert_eq!(follower.doc_count_realtime().0, 10);
    // Direct writes are refused.
    let e = follower.write(vec![WriteReq::Index { id: None, source: "{}".into(), routing: None, cond: WriteCond::default() }], true).remove(0).unwrap_err();
    assert_eq!((e.status, e.kind.as_str()), (403, "status_exception"));
    // New leader ops (index, update, delete) replicate with the same _seq_no / _version.
    for i in 10..20 {
        put(&leader, &format!("d{i}"), &format!("{{\"n\":{i},\"extra\":\"x\"}}"));
    }
    put(&leader, "d0", r#"{"n":100}"#);
    leader.write(vec![WriteReq::Delete { id: "d1".into(), routing: None, cond: WriteCond::default() }], true).remove(0).unwrap();
    wait_for(|| follower.doc_count_realtime().0 == 19 && follower.get("d0").ok().flatten().is_some_and(|g| g.version == 2));
    for id in ["d0", "d5", "d15"] {
        let (l, f) = (leader.get(id).unwrap().unwrap(), follower.get(id).unwrap().unwrap());
        assert_eq!((l.seq_no, l.version, l.source.clone()), (f.seq_no, f.version, f.source.clone()), "{id}");
    }
    assert!(follower.get("d1").unwrap().is_none());
    // Dynamic mapping of the leader reached the follower.
    assert!(follower.lookup().get("extra").is_some());
    // Stats.
    let s = follow_stats(&node, "follower").unwrap();
    assert_eq!(s["indices"][0]["shards"].as_array().unwrap().len(), 2);
    assert_eq!(s["indices"][0]["total_global_checkpoint_lag"], json!(0));
    // Pause stops replication; resume catches up.
    pause_follow(&node, "follower").unwrap();
    put(&leader, "late", "{}");
    std::thread::sleep(Duration::from_millis(100));
    assert!(follower.get("late").unwrap().is_none());
    resume_follow(&node, "follower", &FollowParams::default()).unwrap();
    wait_for(|| follower.get("late").ok().flatten().is_some());
    // Leases on the leader name the follower.
    let leases = leader.shard(0).unwrap().shard.retention_leases();
    assert!(leases.iter().any(|l| l.id.contains("/follower/") && l.source == "ccr"), "{leases:?}");
    // Unfollow requires paused + closed, removes the setting and the leases.
    assert!(unfollow(&node, "follower").is_err());
    pause_follow(&node, "follower").unwrap();
    follower.set_open(false).unwrap();
    unfollow(&node, "follower").unwrap();
    assert!(follower.setting(FOLLOWING_INDEX_SETTING).is_none());
    assert!(!leader.shard(0).unwrap().shard.retention_leases().iter().any(|l| l.source == "ccr"));
}

#[test]
fn internal_adapter_auto_follow_follows_new_matching_indices() {
    let (_t, node) = node();
    node.create_index("logs-old", &json!({})).unwrap();
    put_auto_follow_pattern(
        &node,
        "p",
        &json!({"remote_cluster": "local", "leader_index_patterns": ["logs-*"], "leader_index_exclusion_patterns": ["logs-excluded"], "follow_index_pattern": "copy-{{leader_index}}"}),
    )
    .unwrap();
    node.create_index("logs-new", &json!({})).unwrap();
    node.create_index("logs-excluded", &json!({})).unwrap();
    wait_for(|| node.get_index("copy-logs-new").is_some());
    std::thread::sleep(Duration::from_millis(200));
    assert!(node.get_index("copy-logs-old").is_none(), "existing indices are not auto-followed");
    assert!(node.get_index("copy-logs-excluded").is_none());
    let s = ccr_stats(&node).unwrap();
    assert_eq!(s["auto_follow_stats"]["number_of_successful_follow_indices"], json!(1));
    assert_eq!(s["follow_stats"]["indices"][0]["index"], json!("copy-logs-new"));
}

#[test]
fn follower_index_names_for_patterns() {
    let p = |fp: Option<&str>| AutoFollowPattern {
        remote_cluster: "r".into(),
        leader_index_patterns: vec![],
        leader_index_exclusion_patterns: vec![],
        follow_index_pattern: fp.map(str::to_string),
        settings: FlatSettings::new(),
        active: true,
        params: FollowParams::default(),
    };
    assert_eq!(follower_index_name(&p(None), "foo"), "foo");
    assert_eq!(follower_index_name(&p(Some("{{leader_index}}_copy")), "foo"), "foo_copy");
    assert_eq!(follower_index_name(&p(Some("{{leader_index}}_copy")), ".ds-logs-foo-bar-2022.02.02-000001"), ".ds-logs-foo-bar_copy-2022.02.02-000001");
}

#[test]
fn cursor_reads_match_changes_range() {
    let (_t, node) = node();
    let svc = node.create_index("c", &json!({})).unwrap();
    let sh = svc.shard(0).unwrap();
    sh.shard.add_retention_lease("t", 0, "test").unwrap();
    for i in 0..50 {
        put(&svc, &format!("d{}", i % 7), &format!("{{\"i\":{i}}}"));
        if i == 20 || i == 35 {
            svc.flush().unwrap();
        }
    }
    let all: Vec<_> = sh.shard.changes_range(0, 49, true).unwrap().collect();
    assert_eq!(all.len(), 50);
    let mut cursor = None;
    let mut got = Vec::new();
    let mut from = 0;
    loop {
        let ops = sh.shard.changes_from_cursor(&mut cursor, from, 49, 8).unwrap();
        if ops.is_empty() {
            break;
        }
        assert!(ops.len() <= 8);
        from = ops.remaining_last().unwrap().seq_no() + 1;
        got.extend(ops);
    }
    assert_eq!(got, all);
    // Bounded by `to`; a cursor that does not sit at `from` falls back.
    let mut c = cursor;
    assert_eq!(sh.shard.changes_from_cursor(&mut c, 10, 12, 100).unwrap(), all[10..=12].to_vec());
    let mut none = None;
    assert_eq!(sh.shard.changes_from_cursor(&mut none, 40, 49, 100).unwrap(), all[40..].to_vec());
}

/// Replication throughput (run with `--ignored --nocapture`).
#[test]
#[ignore]
fn bench_follow_throughput() {
    let (_t, node) = node();
    let leader = node.create_index("bl", &json!({"settings": {"number_of_shards": 1}})).unwrap();
    follow(&node, "bf", "bl");
    let follower = node.get_index("bf").unwrap();
    let n = 200_000usize;
    let t0 = Instant::now();
    for chunk in (0..n).collect::<Vec<_>>().chunks(1000) {
        let reqs = chunk
            .iter()
            .map(|i| WriteReq::Index { id: Some(format!("d{i}")), source: format!("{{\"n\":{i},\"s\":\"some text value {i}\"}}"), routing: None, cond: WriteCond::default() })
            .collect();
        for r in leader.write(reqs, false) {
            r.unwrap();
        }
    }
    let wrote = t0.elapsed();
    wait_for(|| follower.doc_count_realtime().0 as usize == n);
    let total = t0.elapsed();
    eprintln!("leader write {n} docs: {wrote:?}; follower caught up after {total:?} ({:.0} docs/s end-to-end)", n as f64 / total.as_secs_f64());
}

#[test]
fn internal_adapter_following_resumes_after_restart() {
    let tmp = tempfile::tempdir().unwrap();
    let open = || {
        let node = Node::open(tmp.path()).unwrap();
        node.node_settings.write().insert("xpack.license.self_generated.type".into(), json!("trial"));
        crate::license::start(&node);
        install_internal_local_adapter(&node);
        node
    };
    {
        let node = open();
        node.cluster_settings.write().0.insert("cluster.remote.local.seeds".into(), json!(remote::TRANSPORT_ADDRESS));
        node.persist_state().unwrap();
        let leader = node.create_index("rl", &json!({})).unwrap();
        put(&leader, "a", "{}");
        follow(&node, "rf", "rl");
        node.flush_all();
    }
    // The old node (and its worker) go away; reopen on the same path.
    std::thread::sleep(Duration::from_millis(300));
    let node = open();
    start(&node);
    let leader = node.get_index("rl").unwrap();
    let follower = node.get_index("rf").unwrap();
    assert_eq!(follower.doc_count_realtime().0, 1);
    put(&leader, "b", "{}");
    wait_for(|| follower.get("b").ok().flatten().is_some());
    assert_eq!(follow_info(&node, "rf").unwrap()["follower_indices"][0]["status"], json!("active"));
}

#[test]
fn cursor_reads_respect_byte_budget() {
    let (_t, node) = node();
    let svc = node.create_index("cb", &json!({})).unwrap();
    let sh = svc.shard(0).unwrap();
    for i in 0..20 {
        put(&svc, &format!("d{i}"), "{\"x\":\"0123456789\"}");
    }
    let size = |o: &ChangeOp| -> u64 { if let ChangeOp::Index { source_json, .. } = o { source_json.len() as u64 } else { 0 } };
    let mut cursor = None;
    let mut from = 0;
    let mut got = 0;
    while from <= 19 {
        let ops = sh.shard.changes_from_cursor_sized(&mut cursor, from, 19, 100, 50, size).unwrap();
        assert!(!ops.is_empty() && ops.len() <= 2, "{}", ops.len());
        assert_eq!(ops[0].seq_no(), from);
        from = ops.remaining_last().unwrap().seq_no() + 1;
        got += ops.len();
    }
    assert_eq!(got, 20);
}

#[test]
fn internal_adapter_does_not_change_public_or_other_node_connection_policy() {
    let (_tmp, injected) = node();
    assert!(service(&injected).connect(&injected, "local").is_ok());
    assert!(LeaderCluster::connect(&injected, "local").is_err());
    let other_dir = tempfile::tempdir().unwrap();
    let other = Node::open(other_dir.path()).unwrap();
    other.cluster_settings.write().0.insert("cluster.remote.local.seeds".into(), json!(remote::TRANSPORT_ADDRESS));
    let err = match service(&other).connect(&other, "local") {
        Ok(_) => panic!("test adapter escaped its node"),
        Err(error) => error,
    };
    assert_eq!(err.kind, "connect_transport_exception");
    assert_eq!(err.status, 502);
}


#[test]
fn internal_adapter_pending_mapping_does_not_reread_operations() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let (_tmp, node) = node();
    let leader = node.create_index("leader", &json!({"settings":{"number_of_shards":1}})).unwrap();
    follow(&node, "follower", "leader");
    let follower = node.get_index("follower").unwrap();
    let pending_calls = Arc::new(AtomicUsize::new(0));
    let calls = pending_calls.clone();
    set_follower_metadata_hook(&node, Some(Arc::new(move |_, change| {
        assert!(change.mapping.is_some());
        calls.fetch_add(1, Ordering::SeqCst);
        Err(EsError::new(503, FOLLOWER_METADATA_PENDING, "test mapping commit is pending"))
    })));
    put(&leader, "new", r#"{"new_field":42}"#);
    wait_for(|| pending_calls.load(Ordering::SeqCst) > 0);
    // Let the first observed change finish its current poll. Later polls must
    // neither reread nor count the same range while the mapping stays pending.
    std::thread::sleep(Duration::from_millis(150));
    let first = follow_stats(&node, "follower").unwrap();
    std::thread::sleep(Duration::from_millis(550));
    let second = follow_stats(&node, "follower").unwrap();
    for key in ["operations_read", "bytes_read", "successful_read_requests"] {
        assert_eq!(first["indices"][0]["shards"][0][key].as_u64().expect("read counter"), second["indices"][0]["shards"][0][key].as_u64().expect("read counter"), "{key}");
    }
    assert_eq!(follower.doc_count_realtime().0, 0);
    set_follower_metadata_hook(&node, None);
    wait_for(|| follower.doc_count_realtime().0 == 1);
    // The op is realtime-visible once its replay installs it, before
    // `apply_replica_ops` returns (its translog sync and inversion follow)
    // and the task counts it: wait for the count, then check it is exactly
    // one (the pending polls reread nothing).
    wait_for(|| follow_stats(&node, "follower").unwrap()["indices"][0]["shards"][0]["operations_written"].as_u64() >= Some(1));
    std::thread::sleep(Duration::from_millis(200));
    let done = follow_stats(&node, "follower").unwrap();
    assert_eq!(done["indices"][0]["shards"][0]["operations_written"], 1);
}
