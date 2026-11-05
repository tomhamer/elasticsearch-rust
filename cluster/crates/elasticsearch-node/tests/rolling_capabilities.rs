//! Real two-executable rolling upgrade. The validation script must build
//! PR3_OLD_BINARY from current master (pinned by PR3_OLD_REVISION; validated
//! against ffe4b591, and historically 64ea4834) and record both executable hashes.
//! Missing old executable is a failure, never a silent coverage exclusion.
//! The segment format floor: while an old node is a member, nothing above
//! v11 is written; once the last one is replaced the floor rises to this
//! build's format, new segments are written in it, and an old executable
//! is refused at join (and by the gateway fence when it has upgraded data).
mod common;
use common::procs::{start_procs_with_binaries, Procs};
use reqwest::Method;
use serde_json::{json, Value};
use std::{path::PathBuf, time::{Duration, Instant}};

async fn append(p: &Procs, from: usize, to: usize) {
    for id in from..to {
        let (status, body) = p.call(id % 2, Method::PUT, &format!("/rolling/_doc/{id}"), Some(json!({"n":id,"tag":"known"}))).await;
        assert!(status == 200 || status == 201, "known-mapping write {id}: {status} {body}");
    }
}

async fn assert_count(p: &Procs, expected: usize) {
    let (status, body) = p.call(0, Method::POST, "/rolling/_refresh", None).await;
    assert_eq!(status, 200, "{body}");
    for node in 0..2 {
        let (status, body) = p.call(node, Method::GET, "/rolling/_count", None).await;
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["count"], expected, "n{node}: {body}");
    }
}

/// Reads each node's gateway files without writing: the master baseline's
/// format1 generations already carry the legacy marker, so open is read-only.
async fn wait_lifecycle_persisted(p: &Procs, index: &str) {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let ready = (0..2).all(|i| {
            elasticsearch_node::gateway::Gateway::open(&p.dir.join(format!("n{i}")), &p.cluster)
                .ok().is_some_and(|g| g.persisted().last_accepted_state.metadata.indices.get(index)
                    .is_some_and(|m| m.lifecycle.is_some()))
        });
        if ready { return; }
        assert!(Instant::now() < deadline, "lifecycle of [{index}] was not durably published to both nodes");
        let _ = p.call(0, Method::GET, &format!("/{index}/_ilm/explain"), None).await;
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

/// The segment format floor node `i` last accepted (read-only gateway open).
fn persisted_floor(p: &Procs, i: usize) -> Option<u32> {
    elasticsearch_node::gateway::Gateway::open(&p.dir.join(format!("n{i}")), &p.cluster)
        .ok().and_then(|g| elasticsearch_cluster::compatibility::segment_format_floor(&g.persisted().last_accepted_state.metadata))
}

/// The committed segment format floor node `i` recorded on applying it
/// (what it writes in at its next start).
fn committed_floor(p: &Procs, i: usize) -> Option<u32> {
    elasticsearch_node::gateway::load_committed_segment_format(&p.dir.join(format!("n{i}"))).ok().flatten().map(|(_, floor)| floor)
}

/// Every node accepted `floor` and applied it (its log says so: only the
/// new binary logs this, and only once a master raised the floor).
async fn wait_floor_applied(p: &Procs, floor: u32) {
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let ready = (0..2).all(|i| persisted_floor(p, i) == Some(floor)
            && committed_floor(p, i) == Some(floor)
            && std::fs::read_to_string(p.log_path(i)).is_ok_and(|log| log.lines()
                .any(|l| l.contains("applied the cluster's segment format") && l.contains(&format!("floor={floor}")))));
        if ready { return; }
        assert!(Instant::now() < deadline, "segment format floor v{floor} was not applied on both nodes: {:?}", (0..2).map(|i| persisted_floor(p, i)).collect::<Vec<_>>());
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

/// Segment file header versions (`RSTC` + u32) in an engine image.
fn segment_versions(files: &std::collections::BTreeMap<PathBuf, Vec<u8>>) -> std::collections::BTreeSet<u32> {
    files.iter().filter(|(path, bytes)| path.extension().is_some_and(|ext| ext == "rstc") && bytes.len() >= 8 && &bytes[..4] == b"RSTC")
        .map(|(_, bytes)| u32::from_le_bytes(bytes[4..8].try_into().unwrap())).collect()
}

/// Exact bytes, not timestamps: attempts to open an engine may update its
/// manifests, WAL or segment files before an older gateway rejects format2.
/// Only the node's engine* trees are compared; process logs and coordination
/// runtime files are excluded (accepted term/version are checked separately).
fn engine_image(root: &std::path::Path) -> std::collections::BTreeMap<PathBuf, Vec<u8>> {
    fn walk(root: &std::path::Path, path: &std::path::Path, out: &mut std::collections::BTreeMap<PathBuf, Vec<u8>>) {
        for entry in std::fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() { walk(root, &entry.path(), out); }
            else if entry.file_type().unwrap().is_file() {
                out.insert(entry.path().strip_prefix(root).unwrap().to_path_buf(), std::fs::read(entry.path()).unwrap());
            }
        }
    }
    let mut files = std::collections::BTreeMap::new();
    for entry in std::fs::read_dir(root).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name().to_string_lossy().starts_with("engine") && entry.file_type().unwrap().is_dir() {
            walk(root, &entry.path(), &mut files);
        }
    }
    assert!(!files.is_empty(), "downgrade fixture must contain persisted engine data");
    files
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn old_new_known_mapping_load_then_lifecycle_and_gateway_downgrade_fence() {
    let old = PathBuf::from(std::env::var_os("PR3_OLD_BINARY").expect("PR3_OLD_BINARY is required: build the pinned master baseline on EC2; this compatibility test cannot be skipped"));
    let new = PathBuf::from(env!("CARGO_BIN_EXE_elasticsearch-node"));
    assert_ne!(std::fs::canonicalize(&old).unwrap(), std::fs::canonicalize(&new).unwrap(), "old/new binaries must be distinct");
    assert!(std::fs::read(&old).unwrap() != std::fs::read(&new).unwrap(), "different paths must not disguise the same executable");
    let mut p = start_procs_with_binaries(&[], &[old.clone(), old.clone()]).await;
    let (status, body) = p.call(0, Method::PUT, "/rolling", Some(json!({
        "settings":{"number_of_shards":2,"number_of_replicas":1},
        "mappings":{"dynamic":"strict","properties":{"n":{"type":"long"},"tag":{"type":"keyword"}}}
    }))).await;
    assert_eq!(status, 200, "{body}");
    p.wait_health(0, "rolling", "green", 120).await;
    append(&p, 0, 40).await;
    // ILM state exists BEFORE any node is upgraded: master already persists
    // IndexMetadata.lifecycle, so the upgraded nodes must accept the old
    // master's publications and the old members after a new master is elected.
    let (status, body) = p.call(0, Method::PUT, "/_ilm/policy/rolling-policy", Some(json!({"policy":{"phases":{"hot":{"min_age":"1000s","actions":{}}}}}))).await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = p.call(0, Method::PUT, "/managed-before", Some(json!({"settings":{"number_of_shards":1,"number_of_replicas":1,"index.lifecycle.name":"rolling-policy"}}))).await;
    assert_eq!(status, 200, "{body}");
    p.wait_health(0, "managed-before", "green", 120).await;
    wait_lifecycle_persisted(&p, "managed-before").await;
    let master = p.master(0).await;
    let follower = 1 - master;
    p.stop(follower, 20).await;
    p.set_binary(follower, &new);
    p.spawn(follower, &[]).await;
    p.wait_nodes(master, 2, 120).await;
    p.wait_health(master, "rolling", "green", 120).await;
    assert_eq!(p.master(master).await, master, "old master must remain exercised during mixed-version phase");
    append(&p, 40, 80).await;
    assert_count(&p, 80).await;
    // The old master can still mutate metadata (and publish lifecycle state)
    // to the upgraded follower.
    let (status, body) = p.call(master, Method::PUT, "/mixed-phase", Some(json!({"settings":{"number_of_shards":1,"number_of_replicas":1}}))).await;
    assert_eq!(status, 200, "old master publication with lifecycle state present: {body}");
    p.wait_health(master, "mixed-phase", "green", 120).await;
    // A bbq_disk copy holds DiskBBQ IVF files whatever the segment format
    // floor, so the upgraded primary refuses to recover it onto the old
    // node: before any flush, tagged as a segment format refusal. The old
    // master allocates blindly and counts those toward max_retry; it never
    // places the copy, and the old node keeps no partial copy.
    let (status, body) = p.call(master, Method::PUT, "/disk-bbq", Some(json!({
        "settings":{"number_of_shards":1,"number_of_replicas":0,"index.routing.allocation.include._name":format!("n{follower}")},
        "mappings":{"properties":{"v":{"type":"dense_vector","dims":64,"index":true,"similarity":"l2_norm","index_options":{"type":"bbq_disk"}}}}
    }))).await;
    assert_eq!(status, 200, "{body}");
    p.wait_health(master, "disk-bbq", "green", 120).await;
    for id in 0..8 {
        let v: Vec<f32> = (0..64).map(|d| ((id * 64 + d) % 17) as f32).collect();
        let (status, body) = p.call(follower, Method::PUT, &format!("/disk-bbq/_doc/{id}?refresh=true"), Some(json!({"v": v}))).await;
        assert!(status == 200 || status == 201, "{body}");
    }
    let (status, body) = p.call(master, Method::PUT, "/disk-bbq/_settings", Some(json!({"index":{"number_of_replicas":1,"routing.allocation.include._name":"n0,n1"}}))).await;
    assert_eq!(status, 200, "{body}");
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let (_, explain) = p.call(master, Method::POST, "/_cluster/allocation/explain", Some(json!({"index":"disk-bbq","shard":0,"primary":false}))).await;
        if explain.to_string().contains("segment_format_refusal[v13]") { break; }
        assert!(Instant::now() < deadline, "the upgraded primary never refused the old node's recovery: {explain}\n{}", p.log_tail(follower));
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    let (status, health) = p.call(master, Method::GET, "/_cluster/health/disk-bbq", None).await;
    assert_eq!(status, 200, "{health}");
    assert_eq!(health["status"], "yellow", "the old node must not hold a copy: {health}");
    let (status, nodes) = p.call(master, Method::GET, "/_nodes", None).await;
    assert_eq!(status, 200, "{nodes}");
    for node in nodes["nodes"].as_object().unwrap().values() {
        let advertised = node["attributes"]["rustic.internal.capabilities_schema"] == "1";
        assert_eq!(advertised, node["name"] == format!("n{follower}"), "{node}");
    }
    // An old member holds the segment format at the v11 baseline: nothing
    // raised it, so the upgraded follower writes what the old node reads.
    assert_eq!(persisted_floor(&p, follower), None, "the floor must not rise while an old member is present");
    // Keep idempotent known-mapping load in flight while the old master is
    // replaced. Transient quorum/transport unavailability is retried, but every
    // document must eventually be acknowledged and counted exactly once.
    let client = p.c.clone();
    let load_ports = p.p.http.clone();
    let old_master_port = load_ports[master];
    let load = tokio::spawn(async move {
        for id in 120..160 {
            let deadline = Instant::now() + Duration::from_secs(120);
            loop {
                let port = load_ports[id % 2];
                let attempt = client.put(format!("http://127.0.0.1:{port}/rolling/_doc/{id}"))
                    .timeout(Duration::from_secs(5)).json(&json!({"n":id,"tag":"known"})).send().await;
                if let Ok(response) = attempt {
                    let status = response.status().as_u16();
                    if status == 200 || status == 201 { break; }
                    let body = response.text().await.unwrap_or_default();
                    // The old release, stopping while it is master, answers a
                    // write whose local auto-create its runtime aborted with a
                    // 500 `exception`, "local handler for [internal:cluster/
                    // auto_create] failed: task N was cancelled". That is that
                    // binary's answer, not this build's: this build reports the
                    // same abort as a node disconnect (`send_local`) and turns
                    // the old answer into a retryable `node_closed_exception`
                    // when it forwards to an old node (`closing_peer_answer`).
                    // A client talking to the stopping old node directly sees
                    // it and retries, as for a node closing under it.
                    let old_shutdown_abort = port == old_master_port && status == 500
                        && body.contains("local handler for [internal:cluster/") && body.contains("was cancelled");
                    assert!(old_shutdown_abort || [409, 429, 502, 503, 504].contains(&status), "unexpected rolling-load error {status} from port {port}: {body}");
                }
                assert!(Instant::now() < deadline, "rolling-load document {id} never acknowledged");
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    });
    p.stop(master, 20).await;
    p.set_binary(master, &new);
    p.spawn(master, &[]).await;
    p.wait_nodes(follower, 2, 120).await;
    p.wait_health(follower, "rolling", "green", 120).await;
    // The upgraded master does not treat the old master's counted refusals
    // as failures: the replica is placed without reroute?retry_failed.
    p.wait_health(follower, "disk-bbq", "green", 120).await;
    load.await.expect("rolling load task");
    append(&p, 80, 120).await;
    assert_count(&p, 160).await;
    // The new master's tasks succeed with pre-upgrade lifecycle state present.
    let (status, body) = p.call(0, Method::PUT, "/_ilm/policy/rolling-policy", Some(json!({"policy":{"phases":{"hot":{"min_age":"1000s","actions":{}}}}}))).await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = p.call(0, Method::GET, "/managed-before/_ilm/explain", None).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["indices"]["managed-before"]["managed"], true, "{body}");
    let (status, body) = p.call(0, Method::PUT, "/managed", Some(json!({"settings":{"number_of_shards":1,"number_of_replicas":1,"index.lifecycle.name":"rolling-policy"}}))).await;
    assert_eq!(status, 200, "{body}");
    wait_lifecycle_persisted(&p, "managed").await;
    wait_lifecycle_persisted(&p, "managed-before").await;
    // With the last old node replaced, the master raises the segment format
    // to what every member reads, automatically, and new files use it.
    let current = elasticsearch_engine::rustic_core::codec::VERSION;
    wait_floor_applied(&p, current).await;
    append(&p, 160, 200).await;
    assert_count(&p, 200).await;
    let (status, body) = p.call(0, Method::POST, "/rolling/_flush", None).await;
    assert_eq!(status, 200, "{body}");
    // An old executable can no longer join, even with fresh data (no gateway
    // fence to stop it first): the upgraded node refuses it at join/vote.
    let (remaining, fresh) = (master, 1 - master);
    p.stop(fresh, 20).await;
    std::fs::rename(p.dir.join(format!("n{fresh}")), p.dir.join(format!("n{fresh}-upgraded"))).unwrap();
    p.set_binary(fresh, &old);
    p.spawn(fresh, &[]).await;
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let log = std::fs::read_to_string(p.log_path(remaining)).unwrap_or_default();
        if log.lines().any(|l| l.contains("refusing incompatible coordination operation") && l.contains("below the cluster's segment format")) { break; }
        assert!(Instant::now() < deadline, "upgraded node never refused the old executable: {}\n{}", p.log_tail(remaining), p.log_tail(fresh));
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let (status, nodes) = p.call(fresh, Method::GET, "/_nodes", None).await;
    assert!(status != 200 || nodes["_nodes"]["total"].as_u64() != Some(2), "old executable joined the upgraded cluster: {nodes}");
    p.stop(fresh, 20).await;
    std::fs::remove_dir_all(p.dir.join(format!("n{fresh}"))).unwrap();
    std::fs::rename(p.dir.join(format!("n{fresh}-upgraded")), p.dir.join(format!("n{fresh}"))).unwrap();
    p.set_binary(fresh, &new);
    // Every metadata mutation above is persisted in checked format2. A real
    // old executable must refuse those same files, not form an empty cluster.
    p.stop(0, 20).await;
    p.stop(1, 20).await;
    for i in 0..2 {
        let versions = segment_versions(&engine_image(&p.dir.join(format!("n{i}"))));
        assert!(versions.contains(&current), "n{i} wrote no v{current} segment after the full upgrade: {versions:?}");
    }
    let before: Value = {
        let gateway = elasticsearch_node::gateway::Gateway::open(&p.dir.join("n0"), &p.cluster).unwrap();
        json!({"term":gateway.persisted().current_term,"version":gateway.persisted().last_accepted_state.version})
    };
    let engine_before = engine_image(&p.dir.join("n0"));
    p.set_binary(0, &old);
    p.spawn(0, &[]).await;
    let status = p.wait_exit(0, 30).await;
    assert!(!status.success(), "old executable must refuse checked gateway format");
    let log = p.log_tail(0);
    assert!(log.contains("_cluster/state-") && log.contains("downgrading this node is not supported"), "refusal must identify the downgrade explicitly: {log}");
    let engine_after = engine_image(&p.dir.join("n0"));
    assert_eq!(engine_before.keys().collect::<Vec<_>>(), engine_after.keys().collect::<Vec<_>>(), "old startup changed engine file inventory before gateway refusal");
    for (path, bytes) in &engine_before {
        assert!(engine_after.get(path) == Some(bytes), "old startup mutated engine file {:?} before gateway refusal", path);
    }
    let gateway = elasticsearch_node::gateway::Gateway::open(&p.dir.join("n0"), &p.cluster).unwrap();
    assert_eq!(json!({"term":gateway.persisted().current_term,"version":gateway.persisted().last_accepted_state.version}), before, "failed downgrade must not rewrite accepted coordination state");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn old_binary_reserved_attributes_cannot_authorize_a_join_to_new_binary() {
    use elasticsearch_cluster::{coordination::Message, state::DiscoveryNode};
    use elasticsearch_transport::{TransportConfig, TransportError, TransportService};
    let old = PathBuf::from(std::env::var_os("PR3_OLD_BINARY").expect("PR3_OLD_BINARY required; mixed-version coverage cannot be skipped"));
    let new = PathBuf::from(env!("CARGO_BIN_EXE_elasticsearch-node"));
    let attrs = [
        ("node.attr.rustic.internal.capabilities_schema","1"),
        ("node.attr.rustic.internal.lifecycle_state","1"),
        ("node.attr.rustic.internal.segment_read_max","12"),
        ("node.attr.rustic.internal.segment_write","11"),
    ];
    let mut p = start_procs_with_binaries(&attrs, &[old.clone(),old]).await;
    let master = p.master(0).await;
    let follower = 1-master;
    let identity = DiscoveryNode { id:"probe-client".into(), name:"probe-client".into(), ephemeral_id:"probe-client-process".into(),
        transport_address:"127.0.0.1:0".into(), http_address:String::new(), host:"localhost".into(), ip:"127.0.0.1".into(),
        roles:Default::default(), attributes:Default::default(), version:"9.5.5".into() };
    let client = TransportService::bind(TransportConfig::new(p.cluster.clone(), identity, "127.0.0.1:0".parse().unwrap())).await.unwrap();
    let old_addr = format!("127.0.0.1:{}",p.p.transport[master]).parse().unwrap();
    let old_identity = client.connect(old_addr).await.unwrap();
    let old_peer = &old_identity.node;
    assert_eq!(old_peer.attributes["rustic.internal.lifecycle_state"], "1", "old binary echoes user attributes");
    let proof: Result<DiscoveryNode, _> = client.send_request(old_addr, "internal:node/capabilities/v1", &(), Duration::from_secs(2)).await;
    assert!(matches!(&proof, Err(TransportError::ActionNotFound { .. })), "old executable must lack the real capability action: {proof:?}");
    p.stop(follower, 20).await;
    p.set_binary(follower, &new);
    p.spawn(follower, &[]).await;
    let new_addr = format!("127.0.0.1:{}",p.p.transport[follower]).parse().unwrap();
    let body = bytes::Bytes::from(serde_json::to_vec(&json!({"from":old_peer.transport_address.clone(),"msg":Message::JoinRequest {
        request_id: 99001, cluster_name:p.cluster.clone(), source:(*old_peer).clone(), min_term:0,
        join:None, cluster_uuid:String::new(), cluster_uuid_committed:false,
    }})).unwrap());
    let deadline = Instant::now()+Duration::from_secs(60);
    loop {
        let result = client.send_request_bytes(new_addr, elasticsearch_node::coord::ACTION, body.clone(), Duration::from_secs(2)).await;
        if let Err(TransportError::Remote(error)) = &result {
            if error.error_type == "unsupported_operation_exception" && error.reason.contains("unverified internal capability") { break; }
        }
        assert!(Instant::now()<deadline, "new node never explicitly rejected old forged attributes: {result:?}; {}",p.log_tail(follower));
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    client.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compatible_v12_full_restart_negotiates_new_process_ids_before_writing() {
    let new = PathBuf::from(env!("CARGO_BIN_EXE_elasticsearch-node"));
    let mut p = start_procs_with_binaries(&[], &[new.clone(),new]).await;
    let (status, body) = p.call(0, Method::PUT, "/rolling", Some(json!({
        "settings":{"number_of_shards":2,"number_of_replicas":1},
        "mappings":{"dynamic":"strict","properties":{"n":{"type":"long"},"tag":{"type":"keyword"}}}
    }))).await;
    assert_eq!(status,200,"{body}");
    p.wait_health(0,"rolling","green",120).await;
    append(&p,0,20).await;
    let ids = p.node_ids(0).await;
    p.stop(0,20).await;
    p.stop(1,20).await;
    // Both persisted peers have old ephemeral IDs. Start both probe listeners
    // concurrently through the real process harness, with v12 explicitly set.
    p.spawn_with_env(0,&[],&[("RUSTIC_SEGMENT_WRITE_VERSION","12")]).await;
    p.spawn_with_env(1,&[],&[("RUSTIC_SEGMENT_WRITE_VERSION","12")]).await;
    for i in 0..2 { p.wait_nodes(i,2,120).await; }
    p.wait_health(0,"rolling","green",120).await;
    assert_eq!(p.node_ids(0).await,ids,"stable node IDs survive full restart");
    append(&p,20,40).await;
    assert_count(&p,40).await;
    let (status,body) = p.call(0,Method::POST,"/rolling/_flush",None).await;
    assert_eq!(status,200,"{body}");
    p.stop(0,20).await;
    p.stop(1,20).await;
    for i in 0..2 {
        let files = engine_image(&p.dir.join(format!("n{i}")));
        assert!(files.iter().any(|(path,bytes)| path.extension().is_some_and(|ext| ext == "rstc")
            && bytes.len() >= 8 && &bytes[..4] == b"RSTC" && u32::from_le_bytes(bytes[4..8].try_into().unwrap()) == 12),
            "n{i} never wrote a v12 segment; must not silently downgrade configured writer");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn v12_quorum_restart_with_offline_peer_and_old_return_refusal() {
    let old = PathBuf::from(std::env::var_os("PR3_OLD_BINARY").expect("PR3_OLD_BINARY required for old-return coverage"));
    let new = PathBuf::from(env!("CARGO_BIN_EXE_elasticsearch-node"));
    let mut p = start_procs_with_binaries(&[], &[new.clone(),new.clone(),new.clone()]).await;
    let (status,body) = p.call(0,Method::PUT,"/rolling",Some(json!({
        "settings":{"number_of_shards":2,"number_of_replicas":1},
        "mappings":{"dynamic":"strict","properties":{"n":{"type":"long"},"tag":{"type":"keyword"}}}
    }))).await;
    assert_eq!(status,200,"{body}");
    p.wait_health(0,"rolling","green",120).await;
    append(&p,0,20).await;
    assert_count(&p,20).await;
    // Kill together: each gateway retains the complete membership, including
    // n2. Restart only a voting quorum, without rewriting gateway membership.
    for i in 0..3 { p.kill(i); }
    for i in 0..2 { p.spawn_with_env(i,&[],&[("RUSTIC_SEGMENT_WRITE_VERSION","12")]).await; }
    for i in 0..2 { p.wait_nodes(i,2,120).await; }
    p.wait_health(0,"rolling","green",120).await;
    append(&p,20,40).await;
    assert_count(&p,40).await;
    // A returning old process cannot acquire feature authority or rejoin.
    p.set_binary(2,&old);
    let before = engine_image(&p.dir.join("n2"));
    p.spawn(2,&[]).await;
    let exit = p.wait_exit(2,60).await;
    assert!(!exit.success(),"old executable unexpectedly accepted format2 gateway");
    assert_eq!(before,engine_image(&p.dir.join("n2")),"old return changed engine data before refusal");
    p.wait_nodes(0,2,30).await;
    append(&p,40,50).await;
    assert_count(&p,50).await;
    p.set_binary(2,&new);
    p.spawn_with_env(2,&[],&[("RUSTIC_SEGMENT_WRITE_VERSION","12")]).await;
    p.wait_nodes(0,3,120).await;
    p.wait_health(0,"rolling","green",120).await;
    assert_count(&p,50).await;
}
