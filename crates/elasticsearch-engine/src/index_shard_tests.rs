//! Logical shards: lifecycle guarantees (A1–A6, FromFiles) and N-shard
//! search / aggregation / kNN equivalence with a single-shard reference.

use super::*;
use crate::hooks::agg_partial::{AggPartial, AggRequest};
use crate::hooks::changes::ApplyOutcome;
use crate::node::Node;
use crate::search::phases::{self, merge_dfs, merge_knn_candidates, merge_query_phases, fetch_page, QueryPhaseOpts, QueryPhaseResult, SearcherHandle};
use crate::search::{ExecOpts, SearchRequest, Target};

fn idx(id: &str, body: Value) -> WriteReq {
    WriteReq::Index { id: Some(id.into()), source: body.to_string(), routing: None, cond: WriteCond::default() }
}

fn node() -> (tempfile::TempDir, Arc<Node>) {
    let tmp = tempfile::tempdir().unwrap();
    let n = Node::open(tmp.path()).unwrap();
    (tmp, n)
}

fn spec(primary: bool, term: u64, recovery: RecoverySource) -> ShardOpenSpec {
    ShardOpenSpec { allocation_id: new_uuid(), primary, primary_term: term, recovery }
}

#[test]
fn n_shards_route_by_murmur3_and_layout() {
    let (_t, node) = node();
    let svc = node.create_index("m", &json!({"settings": {"number_of_shards": 3, "number_of_replicas": 0}})).unwrap();
    assert_eq!(svc.number_of_shards(), 3);
    assert_eq!(svc.local_shards().len(), 3);
    for n in 0..3 {
        assert!(svc.path().join(format!("shard-{n}")).join("shard_state.json").exists());
    }
    let r = svc.write((0..60).map(|i| idx(&format!("d{i}"), json!({"n": i}))).collect(), true);
    assert!(r.iter().all(|x| x.is_ok()));
    // A 9.x-created index: `floorMod(hash, number_of_shards)`.
    let routing = elasticsearch_routing::IndexRouting::new(3, elasticsearch_routing::calculate_num_routing_shards(3), 1).with_modulo(true);
    let mut per = [0u64; 3];
    for i in 0..60 {
        let id = format!("d{i}");
        let want = routing.shard_id(&id, None).unwrap();
        assert_eq!(svc.route(&id, None).unwrap(), want);
        per[want as usize] += 1;
        // The document is on exactly that shard.
        for s in svc.local_shards() {
            assert_eq!(s.shard.peek(&id).unwrap().is_some(), s.id == want, "{id} on shard {}", s.id);
        }
        assert!(svc.get(&id).unwrap().is_some());
    }
    for s in svc.local_shards() {
        assert_eq!(s.doc_count_realtime().0, per[s.id as usize]);
    }
    assert_eq!(svc.doc_count_realtime().0, 60);
    // Custom routing: a GET with another routing value misses (other shard).
    svc.write(vec![WriteReq::Index { id: Some("r1".into()), source: "{}".into(), routing: Some("abc".into()), cond: WriteCond::default() }], true).remove(0).unwrap();
    assert!(svc.get_routed("r1", Some("abc"), true).unwrap().is_some());
    let other = (0..50).map(|i| format!("x{i}")).find(|r| svc.route("r1", Some(r)).unwrap() != svc.route("r1", Some("abc")).unwrap()).unwrap();
    assert!(svc.get_routed("r1", Some(&other), true).unwrap().is_none());
    // Reopen: same shards, same docs.
    drop(svc);
    drop(node);
    let node = Node::open(_t.path()).unwrap();
    let svc = node.get_index("m").unwrap();
    assert_eq!(svc.doc_count_realtime().0, 61);
    for s in svc.local_shards() {
        let extra = u64::from(s.id == svc.route("r1", Some("abc")).unwrap());
        assert_eq!(s.doc_count_realtime().0, per[s.id as usize] + extra);
    }
}

#[test]
fn one_shard_layout_unchanged() {
    let (_t, node) = node();
    let svc = node.create_index("one", &json!({})).unwrap();
    assert_eq!(svc.number_of_shards(), 1);
    svc.write(vec![idx("a", json!({"x": 1}))], true).remove(0).unwrap();
    let names: Vec<String> = std::fs::read_dir(svc.path()).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| !n.ends_with(".lock")).collect();
    let mut names = names;
    names.sort();
    assert_eq!(names, vec!["index_meta.json", "shard-0"]);
    let meta: Value = serde_json::from_slice(&std::fs::read(svc.path().join("index_meta.json")).unwrap()).unwrap();
    assert!(meta.get("sparse").is_none() && meta.get("metadata_version").is_none());
}

#[test]
fn closed_handle_is_fenced_with_already_closed() {
    let (_t, node) = node();
    let svc = node.create_index("c", &json!({"settings": {"number_of_shards": 2}})).unwrap();
    let s0 = svc.shard(0).unwrap();
    s0.write(vec![idx("k", json!({"a": 1}))], true).remove(0).unwrap();
    svc.close_shard(0).unwrap();
    assert!(s0.is_closed());
    let e = s0.write(vec![idx("k2", json!({"a": 2}))], true).remove(0).unwrap_err();
    assert!(e.is_already_closed(), "{e}");
    assert_eq!(e.kind, "index_shard_closed_exception");
    assert!(s0.update("k", &json!({"doc": {"a": 3}}), None, 0, WriteCond::default()).unwrap_err().is_already_closed());
    assert!(s0.apply_replica_ops(1, vec![], Durability::Request).unwrap_err().is_already_closed());
    assert!(s0.snapshot_files(false).unwrap_err().is_already_closed());
    assert!(s0.open_searcher(None).unwrap_err().is_already_closed());
    assert!(s0.get("k", None, true).unwrap_err().is_already_closed());
    // Other errors are not mistaken for it.
    assert!(!EsError::illegal_argument("x").is_already_closed());
    // The slot is free again: reopen from the store.
    let s0b = svc.open_shard(0, ShardOpenSpec::local_primary(RecoverySource::ExistingStore)).unwrap();
    assert!(s0b.shard.peek("k").unwrap().is_some());
}

#[test]
fn stale_close_cannot_retire_a_replacement_physical_copy() {
    let (_t, node) = node();
    let svc = node.create_index("conditional-close", &json!({})).unwrap();
    let original = svc.shard(0).unwrap();
    original.write(vec![idx("kept", json!({"n": 1}))], true).remove(0).unwrap();
    assert!(svc.close_shard_if_same(&original).unwrap());
    assert!(original.is_closed());
    assert!(!svc.close_shard_if_same(&original).unwrap());
    let replacement = svc.open_shard(0, ShardOpenSpec::local_primary(RecoverySource::ExistingStore)).unwrap();
    assert!(!svc.close_shard_if_same(&original).unwrap());
    assert!(Arc::ptr_eq(&svc.shard(0).unwrap(), &replacement));
    assert!(!replacement.is_closed());
    assert!(svc.get("kept").unwrap().is_some());
    replacement.write(vec![idx("new", json!({"n": 2}))], true).remove(0).unwrap();
    // Equal shard numbers in another index are not physical-copy identity.
    let other = node.create_index("other-copy", &json!({})).unwrap();
    assert!(!other.close_shard_if_same(&replacement).unwrap());
    assert!(!other.shard(0).unwrap().is_closed());
    assert!(svc.close_shard_if_same(&replacement).unwrap());
}

#[test]
fn prepared_shard_is_hidden_and_cancelled_before_installation() {
    let (_t, node) = node();
    let svc = node.create_index("prepared", &json!({})).unwrap();
    svc.write(vec![idx("kept", json!({"n": 1}))], true).remove(0).unwrap();
    svc.close_shard(0).unwrap();
    let prepared = svc.prepare_shard(0, ShardOpenSpec::local_primary(RecoverySource::ExistingStore)).unwrap();
    assert!(svc.shard(0).is_none());
    assert!(svc.local_shards().is_empty());
    // Cancellation during an asynchronous open must also fence its later install.
    svc.invalidate_shard_preparation(0).unwrap();
    assert!(svc.install_prepared_shard(prepared).is_err());
    assert!(svc.shard(0).is_none());
    // Rejecting the stale owner released its OS lock; a new generation can open.
    let prepared = svc.prepare_shard(0, ShardOpenSpec::local_primary(RecoverySource::ExistingStore)).unwrap();
    let copy = svc.install_prepared_shard(prepared).unwrap();
    assert!(Arc::ptr_eq(&svc.shard(0).unwrap(), &copy));
    assert!(svc.get("kept").unwrap().is_some());
    svc.close_shard(0).unwrap();
    let prepared = svc.prepare_shard(0, ShardOpenSpec::local_primary(RecoverySource::ExistingStore)).unwrap();
    let other = node.create_index("other-prepared", &json!({})).unwrap();
    assert!(other.install_prepared_shard(prepared).is_err());
    assert!(!other.shard(0).unwrap().is_closed());
    // Abandoning a prepared result never publishes it or leaves its lock held.
    drop(svc.prepare_shard(0, ShardOpenSpec::local_primary(RecoverySource::ExistingStore)).unwrap());
    svc.open_shard(0, ShardOpenSpec::local_primary(RecoverySource::ExistingStore)).unwrap();
    assert!(svc.get("kept").unwrap().is_some());
}

#[test]
fn detached_copy_is_fenced_before_drain_and_keeps_store_exclusive() {
    let (_t, node) = node();
    let svc = node.create_index("detach", &json!({})).unwrap();
    let copy = svc.shard(0).unwrap();
    let permit = copy.operation_permit().unwrap();
    // This must finish while the old permit is still held, not wait for drain.
    let detached = svc.detach_shard_if_same(&copy).unwrap();
    assert!(svc.shard(0).is_none());
    assert!(copy.operation_permit().is_err());
    assert!(svc.prepare_shard(0, ShardOpenSpec::local_primary(RecoverySource::ExistingStore)).is_err());
    drop(permit);
    detached.close().unwrap();
    let mut prepared = svc.prepare_shard(0, ShardOpenSpec::local_primary(RecoverySource::ExistingStore)).unwrap();
    svc.invalidate_shard_preparation(0).unwrap();
    assert!(svc.try_install_prepared_shard(&mut prepared).is_err());
    // Rejection leaves ownership with the caller, until outside ordered locks.
    assert!(svc.prepare_shard(0, ShardOpenSpec::local_primary(RecoverySource::ExistingStore)).is_err());
    drop(prepared);
    let mut prepared = svc.prepare_shard(0, ShardOpenSpec::local_primary(RecoverySource::ExistingStore)).unwrap();
    let replacement = svc.try_install_prepared_shard(&mut prepared).unwrap();
    assert!(svc.try_install_prepared_shard(&mut prepared).is_err());
    drop(prepared);
    assert!(!replacement.is_closed());
    assert!(svc.detach_shard_if_same(&copy).is_none());
    assert!(Arc::ptr_eq(&svc.shard(0).unwrap(), &replacement));
}

#[test]
fn close_waits_for_in_flight_permits() {
    let (_t, node) = node();
    let svc = node.create_index("w", &json!({})).unwrap();
    let s = svc.shard(0).unwrap();
    let permit = s.permit().unwrap();
    let s2 = s.clone();
    let closer = std::thread::spawn(move || s2.close());
    std::thread::sleep(std::time::Duration::from_millis(100));
    assert!(!closer.is_finished(), "close must wait for the in-flight op");
    // New ops are already fenced.
    assert!(s.permit().is_err());
    drop(permit);
    closer.join().unwrap().unwrap();
}

#[test]
fn double_open_fails_with_shard_lock_error() {
    let (_t, node) = node();
    let svc = node.create_index("d", &json!({"settings": {"number_of_shards": 2}})).unwrap();
    let e = svc.open_shard(1, ShardOpenSpec::local_primary(RecoverySource::ExistingStore)).unwrap_err();
    assert_eq!(e.kind, "shard_lock_obtain_failed_exception");
    // A second service over the same directory cannot take the lock either.
    let e = IndexService::open(svc.path()).unwrap_err();
    assert_eq!(e.kind, "shard_lock_obtain_failed_exception");
    // remove_shard(n, true) needs the lock: refused while another holder has it.
    let other = IndexService::sparse_from(IndexCore::new_sparse("d", &svc.uuid, 0, svc.path(), svc.meta().settings, svc.meta().mappings, (*svc.lookup()).clone(), svc.registry(), BTreeMap::new()));
    assert_eq!(other.remove_shard(1, true).unwrap_err().kind, "shard_lock_obtain_failed_exception");
    assert!(svc.path().join("shard-1").exists());
    // After closing, the store can be deleted.
    svc.close_shard(1).unwrap();
    other.remove_shard(1, true).unwrap();
    assert!(!svc.path().join("shard-1").exists());
}

/// A replica batch large enough to parse in parallel applies while every
/// global rayon worker is blocked: its parsing runs on a pool of its own,
/// before the operation permit is taken (a parse on the global pool under
/// the permit could steal, or queue behind, work that waits for it).
#[test]
fn replica_batch_parses_off_the_global_rayon_pool() {
    let (_t, node) = node();
    let svc = node.create_index("rp", &json!({"settings": {"number_of_shards": 1}})).unwrap();
    svc.close_shard(0).unwrap();
    let r = svc.open_shard(0, spec(false, 3, RecoverySource::ExistingStore)).unwrap();
    let ops: Vec<ChangeOp> = (0..200).map(|i| ChangeOp::Index { id: format!("d{i}"), source_json: format!(r#"{{"n":{i},"t":"word {i}"}}"#), routing: None, seq_no: i, version: 1, primary_term: 3 }).collect();
    // Every global worker blocked on a lock this test holds.
    let lock = Arc::new(parking_lot::Mutex::new(()));
    let guard = lock.lock();
    let busy = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let n = rayon::current_num_threads();
    for _ in 0..n {
        let (lock, busy) = (lock.clone(), busy.clone());
        rayon::spawn(move || {
            busy.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            drop(lock.lock());
        });
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while busy.load(std::sync::atomic::Ordering::SeqCst) < n {
        if std::time::Instant::now() >= deadline {
            drop(guard);
            panic!("global Rayon blockers did not start");
        }
        std::thread::yield_now();
    }
    let (tx, rx) = std::sync::mpsc::channel();
    let replica = r.clone();
    std::thread::spawn(move || { let _ = tx.send(replica.apply_replica_ops(3, ops, Durability::Request).map(|res| res.outcomes.len())); });
    let got = rx.recv_timeout(std::time::Duration::from_secs(20));
    drop(guard);
    assert_eq!(got.expect("replica parsing waited on the global rayon pool").unwrap(), 200);
    assert_eq!(r.shard.max_seq_no(), 199);
}

#[test]
fn promote_rejects_lower_or_equal_term_and_replicas_refuse_writes() {
    let (_t, node) = node();
    let svc = node.create_index("p", &json!({"settings": {"number_of_shards": 1}})).unwrap();
    svc.close_shard(0).unwrap();
    let r = svc.open_shard(0, spec(false, 3, RecoverySource::ExistingStore)).unwrap();
    assert!(!r.is_primary());
    assert_eq!(r.shard.primary_term(), 3);
    // A2: writes / updates on a replica are refused.
    let e = r.write(vec![idx("a", json!({}))], true).remove(0).unwrap_err();
    assert_eq!((e.kind.as_str(), e.status), ("shard_not_in_primary_mode_exception", 500));
    assert_eq!(r.update("a", &json!({"doc": {"x": 1}, "doc_as_upsert": true}), None, 0, WriteCond::default()).unwrap_err().kind, "shard_not_in_primary_mode_exception");
    assert_eq!(svc.write(vec![idx("a", json!({}))], true).remove(0).unwrap_err().kind, "shard_not_in_primary_mode_exception");
    // Replica apply works (term-fenced as before).
    let op = |seq: i64, id: &str, term: u64| ChangeOp::Index { id: id.into(), source_json: r#"{"n":1}"#.into(), routing: None, seq_no: seq, version: 1, primary_term: term };
    assert_eq!(r.apply_replica_ops(3, vec![op(0, "x", 3)], Durability::Request).unwrap().outcomes, vec![ApplyOutcome::Applied]);
    assert_eq!(r.apply_replica_ops(2, vec![op(1, "y", 2)], Durability::Request).unwrap_err().kind, "illegal_state_exception");
    // Promotion needs a higher term.
    assert_eq!(r.promote(3).unwrap_err().kind, "illegal_state_exception");
    assert_eq!(r.promote(2).unwrap_err().kind, "illegal_state_exception");
    assert!(!r.is_primary());
    r.promote(4).unwrap();
    assert!(r.is_primary());
    assert_eq!(r.shard.primary_term(), 4);
    let w = r.write(vec![idx("z", json!({}))], true).remove(0).unwrap();
    assert_eq!((w.primary_term, w.seq_no), (4, 1));
    // The term survives a restart.
    svc.close_shard(0).unwrap();
    let again = svc.open_shard(0, spec(true, 1, RecoverySource::ExistingStore)).unwrap();
    assert_eq!(again.shard.primary_term(), 4);
}

#[test]
fn close_then_existing_store_keeps_every_acked_op() {
    let (_t, node) = node();
    let svc = node.create_index("dur", &json!({"settings": {"number_of_shards": 2, "refresh_interval": "-1"}})).unwrap();
    // Acked with sync = false (no per-request fsync): close must still make them durable.
    for i in 0..200 {
        svc.write(vec![idx(&format!("d{i}"), json!({"n": i, "t": "dynamic field"}))], false).remove(0).unwrap();
    }
    svc.write(vec![WriteReq::Delete { id: "d7".into(), routing: None, cond: WriteCond::default() }], false).remove(0).unwrap();
    let before: Vec<(u32, i64, i64, u64)> = svc.local_shards().iter().map(|s| (s.id, s.shard.max_seq_no(), s.shard.local_checkpoint(), s.doc_count_realtime().0)).collect();
    for n in 0..2 {
        svc.close_shard(n).unwrap();
    }
    for n in 0..2 {
        svc.open_shard(n, ShardOpenSpec::local_primary(RecoverySource::ExistingStore)).unwrap();
    }
    let after: Vec<(u32, i64, i64, u64)> = svc.local_shards().iter().map(|s| (s.id, s.shard.max_seq_no(), s.shard.local_checkpoint(), s.doc_count_realtime().0)).collect();
    assert_eq!(before, after);
    assert!(svc.get("d7").unwrap().is_none());
    for i in (0..200).filter(|i| *i != 7) {
        let g = svc.get(&format!("d{i}")).unwrap().unwrap_or_else(|| panic!("d{i} lost"));
        assert!(g.source.unwrap().contains(&format!("\"n\":{i}")));
    }
    // Writes continue the sequence.
    let s = svc.route("next", None).unwrap();
    let w = svc.write(vec![idx("next", json!({}))], true).remove(0).unwrap();
    assert_eq!(w.seq_no, before[s as usize].1 + 1);
}

#[test]
fn from_files_round_trip_and_corrupt_manifest_leaves_nothing() {
    let (_t, node) = node();
    let src = node.create_index("src", &json!({"settings": {"number_of_shards": 2}})).unwrap();
    src.write((0..40).map(|i| idx(&format!("d{i}"), json!({"n": i}))).collect(), true);
    let s1 = src.shard(1).unwrap();
    s1.shard.flush().unwrap();
    src.write((40..50).map(|i| idx(&format!("d{i}"), json!({"n": i}))).collect(), true);
    let set = s1.snapshot_files(true).unwrap();
    let docs1 = s1.doc_count_realtime().0;
    let lcp = s1.shard.local_checkpoint();
    assert_eq!(set.manifest.local_checkpoint, lcp);
    // A sparse copy of the index elsewhere on the node (cluster-style).
    let meta = json!({"settings": settings::as_flat_json(&src.meta().settings), "mappings": src.meta().mappings.to_json()});
    let copy = node.create_index_sparse("copy", "copyUuid1", &meta).unwrap();
    assert!(copy.local_shards().is_empty());
    // Corrupt manifest: nothing under shard-1, staged left alone.
    let staged = node.new_staging_dir().unwrap();
    set.write_to_dir(&staged, false).unwrap();
    let mut bad = set.manifest.clone();
    bad.files[0].crc32 ^= 1;
    let e = copy.open_shard(1, spec(false, 1, RecoverySource::FromFiles { staged: staged.clone(), manifest: bad })).unwrap_err();
    assert_eq!(e.kind, "corrupt_index_exception");
    assert!(!copy.path().join("shard-1").exists());
    assert!(copy.shard(1).is_none());
    // A file missing from the staged copy: same.
    let missing = set.manifest.files.iter().find(|f| f.path.contains("core-")).unwrap().path.clone();
    let staged2 = node.new_staging_dir().unwrap();
    set.write_to_dir(&staged2, false).unwrap();
    std::fs::remove_file(staged2.join(&missing)).unwrap();
    assert_eq!(copy.open_shard(1, spec(false, 1, RecoverySource::FromFiles { staged: staged2, manifest: set.manifest.clone() })).unwrap_err().kind, "corrupt_index_exception");
    assert!(!copy.path().join("shard-1").exists());
    // The intact set recovers: same docs, checkpoints from the manifest.
    let r = copy.open_shard(1, spec(false, 1, RecoverySource::FromFiles { staged: staged.clone(), manifest: set.manifest.clone() })).unwrap();
    assert!(!staged.exists(), "staged directory moved into place");
    assert_eq!(r.doc_count_realtime().0, docs1);
    assert_eq!(r.shard.local_checkpoint(), set.manifest.local_checkpoint);
    assert_eq!(r.global_checkpoint(), set.manifest.local_checkpoint);
    assert!(!copy.path().join("shard-1").join(crate::hooks::files::MANIFEST_FILE).exists());
    // Refused when shard-1/ exists.
    let staged3 = node.new_staging_dir().unwrap();
    set.write_to_dir(&staged3, false).unwrap();
    copy.close_shard(1).unwrap();
    let e = copy.open_shard(1, spec(false, 1, RecoverySource::FromFiles { staged: staged3, manifest: set.manifest.clone() })).unwrap_err();
    assert_eq!(e.kind, "illegal_state_exception");
    assert!(copy.path().join("shard-1").join("shard_state.json").exists(), "existing shard untouched");
}

fn routed_idx(id: &str, routing: &str, body: Value) -> WriteReq {
    WriteReq::Index { id: Some(id.into()), source: body.to_string(), routing: Some(routing.into()), cond: WriteCond::default() }
}

/// A routing value that `svc` routes to shard `n`.
fn routing_to(svc: &IndexService, n: u32) -> String {
    (0..1000).map(|i| format!("r{i}")).find(|r| svc.route("x", Some(r)).unwrap() == n).unwrap()
}

fn search_ids(svc: &Arc<IndexService>) -> (Vec<String>, u64) {
    let req = crate::search::parse_request(&json!({"size": 100}), &std::collections::HashMap::new()).unwrap();
    let out = crate::search::execute(&[Target::new(svc.clone())], &req, &ExecOpts { terms_lookup: None, cursors: None, max_buckets: 100, scroll_id: None, pit_id: None }).unwrap();
    let v: Value = serde_json::from_str(&out.json).unwrap();
    let mut ids: Vec<String> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
    ids.sort();
    (ids, v["_shards"]["total"].as_u64().unwrap())
}

#[test]
fn sparse_service_serves_the_open_copies() {
    let (_t, node) = node();
    let meta = json!({"settings": {"index.number_of_shards": "2", "index.number_of_replicas": "0"}, "mappings": {"properties": {"n": {"type": "long"}}}});
    let svc = node.create_index_sparse("sp", "spUuid", &meta).unwrap();
    // Verbatim metadata: no defaults added.
    assert!(svc.setting("index.creation_date").is_none() || svc.setting("index.version.created").is_none());
    assert_eq!(svc.setting("index.uuid").as_deref(), Some("spUuid"));
    // No copy open: the search fails (shard_not_found_exception), never a
    // 200 that covers zero shards of the index.
    let req = crate::search::parse_request(&json!({"size": 100}), &std::collections::HashMap::new()).unwrap();
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 100, scroll_id: None, pit_id: None };
    let e = crate::search::execute(&[Target::new(svc.clone())], &req, &opts).err().expect("no shard of the index is open here");
    assert_eq!((e.status, e.kind.as_str()), (404, "shard_not_found_exception"));
    svc.open_shard(1, ShardOpenSpec::local_primary(RecoverySource::Empty)).unwrap();
    let (r0, r1) = (routing_to(&svc, 0), routing_to(&svc, 1));
    // Writes routed to the open copy apply there.
    let w = svc.write(vec![routed_idx("a", &r1, json!({"n": 1}))], true).remove(0).unwrap();
    assert_eq!(w.result, "created");
    assert!(svc.shard(1).unwrap().shard.peek("a").unwrap().is_some());
    // Writes routed to a shard with no local copy: shard_not_found.
    let e = svc.write(vec![routed_idx("b", &r0, json!({"n": 2}))], true).remove(0).unwrap_err();
    assert_eq!((e.kind.as_str(), e.status), ("shard_not_found_exception", 404));
    // A mixed batch: per-item results, in request order.
    let rs = svc.write(vec![routed_idx("c", &r1, json!({"n": 3})), routed_idx("d", &r0, json!({"n": 4})), routed_idx("e", &r1, json!({"n": 5})), WriteReq::Delete { id: "f".into(), routing: Some(r0.clone()), cond: WriteCond::default() }], true);
    assert_eq!(rs.len(), 4);
    assert_eq!(rs[0].as_ref().unwrap().id, "c");
    assert_eq!(rs[1].as_ref().unwrap_err().kind, "shard_not_found_exception");
    assert_eq!(rs[2].as_ref().unwrap().id, "e");
    assert_eq!(rs[3].as_ref().unwrap_err().kind, "shard_not_found_exception");
    // Get / update / peek on the open copy work; on the missing one: shard_not_found.
    assert!(svc.get_routed("a", Some(&r1), true).unwrap().is_some());
    assert!(svc.peek("a", Some(&r1)).unwrap().is_some());
    let (u, src) = svc.update("a", &json!({"doc": {"n": 10}}), Some(r1.clone()), 0, WriteCond::default()).unwrap();
    assert_eq!(u.result, "updated");
    assert_eq!(serde_json::from_str::<Value>(&src.unwrap()).unwrap()["n"], 10);
    assert_eq!(svc.get_routed("a", Some(&r0), true).unwrap_err().kind, "shard_not_found_exception");
    assert_eq!(svc.update("a", &json!({"doc": {"n": 1}}), Some(r0.clone()), 0, WriteCond::default()).unwrap_err().kind, "shard_not_found_exception");
    assert_eq!(svc.peek("a", Some(&r0)).unwrap_err().kind, "shard_not_found_exception");
    // Whole-index maintenance covers the open copies.
    svc.refresh().unwrap();
    svc.flush().unwrap();
    assert_eq!(svc.doc_count_realtime().0, 3);
    // Search: exactly the docs of the open copy, one shard searched.
    assert_eq!(search_ids(&svc), (vec!["a".to_string(), "c".to_string(), "e".to_string()], 1));
    // Search routing selecting only the missing shard: not zero shards
    // silently, but ES's shard_not_found_exception.
    let mut t = Target::new(svc.clone());
    t.routing = Some(r0.clone());
    let e = crate::search::expand_targets(&[t]).err().expect("a search of a shard with no local copy fails");
    assert_eq!((e.status, e.kind.as_str()), (404, "shard_not_found_exception"));
    // PIT / scroll snapshots cover the open copy.
    assert_eq!(node.snapshot(&[Target::new(svc.clone())]).unwrap().len(), 1);
    // Opening shard 0 too: writes there succeed and search covers both.
    svc.open_shard(0, ShardOpenSpec::local_primary(RecoverySource::Empty)).unwrap();
    svc.write(vec![routed_idx("d", &r0, json!({"n": 4}))], true).remove(0).unwrap();
    svc.refresh().unwrap();
    assert_eq!(search_ids(&svc), (["a", "c", "d", "e"].iter().map(|s| s.to_string()).collect(), 2));
    assert_eq!(svc.doc_count_realtime().0, 4);
    svc.close_shard(0).unwrap();
    // A single-shard view still refuses whole-index operations.
    let view = IndexService::view_of(&svc.shard(1).unwrap());
    assert_eq!(view.write(vec![routed_idx("z", &r1, json!({}))], true).remove(0).unwrap_err().kind, "illegal_state_exception");
    assert_eq!(view.get_routed("a", Some(&r1), true).unwrap_err().kind, "illegal_state_exception");
    drop(view);
    // Per-shard operations work.
    let req = crate::search::parse_request(&json!({}), &std::collections::HashMap::new()).unwrap();
    let s1 = svc.shard(1).unwrap();
    s1.write(vec![idx("a", json!({"n": 1}))], true).remove(0).unwrap();
    let h = s1.open_searcher(None).unwrap();
    assert_eq!(s1.query_phase(&h, &req, &QueryPhaseOpts::default()).unwrap().total.value, 3);
    // Restart: still sparse, no copy opened implicitly.
    drop((h, s1, svc));
    drop(node);
    let node = Node::open(_t.path()).unwrap();
    let svc = node.get_index("sp").unwrap();
    assert!(svc.is_sparse() && svc.local_shards().is_empty());
    let s1 = svc.open_shard(1, ShardOpenSpec::local_primary(RecoverySource::ExistingStore)).unwrap();
    assert!(s1.shard.peek("a").unwrap().is_some());
}

#[test]
fn committed_archived_settings_are_inert_but_public_writes_remain_rejected() {
    let (_t, node) = node();
    let svc = node.create_index_sparse("archive", "archiveUuid", &json!({})).unwrap();
    let archived: FlatSettings = [("archived.index.max_regex_length".into(), json!("broken"))].into();
    assert!(svc.put_settings(archived.clone(), false).is_err());
    svc.apply_settings_delta(archived.clone()).unwrap();
    assert_eq!(svc.meta().settings["archived.index.max_regex_length"], "broken");
    assert!(svc.put_settings(archived, false).is_err());
    // Trusted metadata still validates active settings and applies atomically.
    let invalid: FlatSettings = [("index.max_regex_length".into(), json!("broken")),
        ("archived.test".into(), json!("retained"))].into();
    assert!(svc.apply_settings_delta(invalid).is_err());
    assert!(!svc.meta().settings.contains_key("archived.test"));
    svc.put_settings([("archived.*".into(), Value::Null)].into(), false).unwrap();
    assert!(!svc.meta().settings.contains_key("archived.index.max_regex_length"));
}

#[test]
fn apply_metadata_is_monotonic() {
    let (_t, node) = node();
    let svc = node.create_index_sparse("am", "amUuid", &json!({"settings": {"index.number_of_shards": "1"}, "mappings": {}})).unwrap();
    let flat = |r: &str| settings::normalize_index_settings(&json!({"number_of_shards": 1, "refresh_interval": r}));
    svc.apply_metadata(&json!({"properties": {"a": {"type": "keyword"}}}), &flat("5s"), 2).unwrap();
    assert!(svc.lookup().get("a").is_some());
    assert_eq!(svc.setting("index.refresh_interval").as_deref(), Some("5s"));
    // Stale / duplicate versions are no-ops.
    svc.apply_metadata(&json!({"properties": {"b": {"type": "keyword"}}}), &flat("9s"), 1).unwrap();
    svc.apply_metadata(&json!({"properties": {"b": {"type": "keyword"}}}), &flat("9s"), 2).unwrap();
    assert!(svc.lookup().get("b").is_none());
    assert_eq!(svc.setting("index.refresh_interval").as_deref(), Some("5s"));
    svc.apply_metadata(&json!({"properties": {"b": {"type": "long"}}}), &flat("1s"), 3).unwrap();
    assert!(svc.lookup().get("b").is_some() && svc.lookup().get("a").is_some());
    assert_eq!(svc.metadata_version(), 3);
}

/// A cluster-style hook that records proposals and applies them back
/// through `apply_metadata` (as the master's publication would).
struct Master {
    svc: std::sync::Mutex<Option<std::sync::Weak<IndexService>>>,
    calls: std::sync::atomic::AtomicU64,
    version: AtomicU64,
}

impl MappingUpdateHook for Master {
    fn mapping_update(&self, _index: &str, _uuid: &str, mappings: &Value) -> Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if mappings.to_string().contains("forbidden") {
            return Err(EsError::illegal_argument("rejected by master"));
        }
        if let Some(svc) = self.svc.lock().unwrap().as_ref().and_then(|w| w.upgrade()) {
            let v = self.version.fetch_add(1, Ordering::SeqCst) + 1;
            svc.apply_metadata(mappings, &svc.meta().settings, v)?;
        }
        Ok(())
    }
}

#[test]
fn dynamic_mapping_goes_through_the_node_hook_once() {
    let (_t, node) = node();
    let svc = node.create_index("dm", &json!({"settings": {"number_of_shards": 3}})).unwrap();
    let m = Arc::new(Master { svc: std::sync::Mutex::new(Some(Arc::downgrade(&svc))), calls: Default::default(), version: AtomicU64::new(0) });
    node.set_mapping_update_hook(Some(m.clone() as Arc<dyn MappingUpdateHook>));
    // Docs on every shard introduce the same new field: the core learns it once.
    let r = svc.write((0..30).map(|i| idx(&format!("d{i}"), json!({"newf": "x"}))).collect(), true);
    assert!(r.iter().all(|x| x.is_ok()));
    assert_eq!(m.calls.load(Ordering::SeqCst), 1, "one proposal, later docs see the applied mapping");
    assert!(svc.lookup().get("newf").is_some());
    assert_eq!(svc.metadata_version(), 1);
    // The master can refuse: the op fails, nothing is applied.
    let e = svc.write(vec![idx("bad", json!({"forbidden": 1}))], true).remove(0).unwrap_err();
    assert_eq!(e.kind, "illegal_argument_exception");
    assert!(svc.lookup().get("forbidden").is_none());
    assert!(svc.get("bad").unwrap().is_none());
    // Without a hook the update is applied locally.
    node.set_mapping_update_hook(None);
    svc.write(vec![idx("loc", json!({"local_only": 1}))], true).remove(0).unwrap();
    assert!(svc.lookup().get("local_only").is_some());
}

#[test]
fn emulated_layout_is_migrated_on_open() {
    let (_t, node) = node();
    let svc = node.create_index("leg", &json!({"settings": {"number_of_shards": 3}})).unwrap();
    svc.write((0..30).map(|i| idx(&format!("d{i}"), json!({"n": i}))).collect(), true);
    let path = svc.path().to_path_buf();
    // Fold everything into shard-0 (what the single-shard emulation wrote).
    let s0 = svc.shard(0).unwrap();
    for s in svc.local_shards().into_iter().skip(1) {
        s0.copy_docs_from(&s, &|_, _| true).unwrap();
    }
    s0.shard.flush().unwrap();
    drop((s0, svc));
    drop(node);
    for n in 1..3 {
        std::fs::remove_dir_all(path.join(format!("shard-{n}"))).unwrap();
    }
    let node = Node::open(_t.path()).unwrap();
    let svc = node.get_index("leg").unwrap();
    assert_eq!(svc.doc_count_realtime().0, 30);
    for i in 0..30 {
        let id = format!("d{i}");
        let want = svc.route(&id, None).unwrap();
        for s in svc.local_shards() {
            assert_eq!(s.shard.peek(&id).unwrap().is_some(), s.id == want, "{id}");
        }
    }
    assert!(!path.join(MIGRATION_MARKER).exists());
}

// --------------------------------------------------------- equivalence

// `hnsw`: exact scores (int8 scores depend on the segment layout, as in ES).
const MAPPING: &str = r#"{"properties": {
    "title": {"type": "text"}, "tag": {"type": "keyword"}, "n": {"type": "long"}, "p": {"type": "double"},
    "v": {"type": "dense_vector", "dims": 4, "index": true, "similarity": "cosine", "index_options": {"type": "hnsw"}}
}}"#;

fn vec_of(i: usize) -> Vec<f64> {
    let a = i as f64 * 0.37;
    vec![a.cos(), a.sin(), (i % 7) as f64 / 7.0, 1.0 + (i % 3) as f64]
}

fn doc(i: usize) -> Value {
    let words = ["quick", "brown", "fox", "lazy", "dog", "jumps"];
    let title: Vec<&str> = (0..1 + i % 5).map(|k| words[(i * 7 + k * 3) % words.len()]).collect();
    json!({"title": title.join(" "), "tag": format!("t{}", i % 6), "n": i, "p": (i * 3 % 20) as f64 + 0.5, "v": vec_of(i)})
}

/// The same 120 docs in a 3-shard index and in a 1-shard reference.
fn pair() -> (tempfile::TempDir, Arc<Node>, Arc<IndexService>, Arc<IndexService>) {
    let (t, node) = node();
    let mapping: Value = serde_json::from_str(MAPPING).unwrap();
    let multi = node.create_index("multi", &json!({"settings": {"number_of_shards": 3, "number_of_replicas": 0}, "mappings": mapping})).unwrap();
    let single = node.create_index("single", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": mapping})).unwrap();
    for chunk in (0..120).collect::<Vec<_>>().chunks(40) {
        let reqs = |_: ()| chunk.iter().map(|i| idx(&format!("d{i}"), doc(*i))).collect::<Vec<_>>();
        assert!(multi.write(reqs(()), true).iter().all(|r| r.is_ok()));
        assert!(single.write(reqs(()), true).iter().all(|r| r.is_ok()));
        multi.refresh().unwrap();
        single.refresh().unwrap();
    }
    assert!(multi.local_shards().iter().all(|s| s.doc_count_realtime().0 > 0), "every shard holds documents");
    (t, node, multi, single)
}

/// A page large enough to render its hits in parallel equals the same
/// hits fetched in small (sequentially rendered) pages.
#[test]
fn large_fetch_page_renders_like_small_pages() {
    let (_t, node) = node();
    let mapping: Value = serde_json::from_str(MAPPING).unwrap();
    let svc = node.create_index("big", &json!({"settings": {"number_of_shards": 3, "number_of_replicas": 0}, "mappings": mapping})).unwrap();
    for chunk in (0..700).collect::<Vec<_>>().chunks(100) {
        assert!(svc.write(chunk.iter().map(|i| idx(&format!("d{i}"), doc(*i))).collect(), true).iter().all(|r| r.is_ok()));
        svc.refresh().unwrap();
    }
    for body in [
        json!({"query": {"match_all": {}}, "sort": [{"n": "asc"}, {"_id": "asc"}]}),
        json!({"query": {"match_all": {}}}),
        json!({"query": {"bool": {"should": [{"match": {"title": "fox dog"}}, {"range": {"n": {"gte": 0}}}]}}, "_source": ["n", "tag"]}),
    ] {
        let mut big = body.clone();
        big["size"] = json!(600);
        let all = exec(&svc, &parse(big, &[]));
        let mut pages: Vec<Value> = Vec::new();
        for from in (0..600).step_by(100) {
            let mut small = body.clone();
            small["size"] = json!(100);
            small["from"] = json!(from);
            pages.extend(exec(&svc, &parse(small, &[]))["hits"]["hits"].as_array().unwrap().iter().cloned());
        }
        assert!(pages.len() > 256, "{body}");
        assert_eq!(all["hits"]["hits"].as_array().unwrap(), &pages, "{body}");
    }
}

fn parse(body: Value, params: &[(&str, &str)]) -> SearchRequest {
    let p: std::collections::HashMap<String, String> = params.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    crate::search::parse_request(&body, &p).unwrap()
}

fn exec(svc: &Arc<IndexService>, req: &SearchRequest) -> Value {
    let out = crate::search::execute(&[Target::new(svc.clone())], req, &ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None }).unwrap();
    serde_json::from_str(&out.json).unwrap()
}

fn ids(v: &Value) -> Vec<String> {
    v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect()
}

/// (id, score) sorted by score desc then id (shard tie order differs).
fn scored(v: &Value) -> Vec<(String, String)> {
    let mut x: Vec<(String, String)> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| (h["_id"].as_str().unwrap().to_string(), h["_score"].to_string())).collect();
    x.sort_by(|a, b| b.1.parse::<f64>().unwrap_or(0.0).total_cmp(&a.1.parse::<f64>().unwrap_or(0.0)).then(a.0.cmp(&b.0)));
    x
}

/// The phases API over every shard copy: DFS (kNN + term stats when asked),
/// query phase, merge, fetch, aggregation reduce.
fn phased(svc: &Arc<IndexService>, req: &SearchRequest, dfs: bool) -> (Value, Option<Value>) {
    let handles: Vec<SearcherHandle> = svc.local_shards().iter().map(|s| s.open_searcher(None).unwrap()).collect();
    let stats = dfs.then(|| merge_dfs(&handles.iter().map(|h| h.shard().dfs_stats(h, req).unwrap()).collect::<Vec<_>>()));
    let cands: Vec<_> = handles.iter().map(|h| h.shard().knn_candidates(h, req, &QueryPhaseOpts::default()).unwrap()).collect();
    let fixed = merge_knn_candidates(req, &cands);
    let opts = QueryPhaseOpts { now: Some(0), knn_fixed: (!req.knn.is_empty()).then_some(&fixed[..]), dfs: stats.as_ref(), ..Default::default() };
    let qps: Vec<QueryPhaseResult> = handles.iter().map(|h| h.shard().query_phase(h, req, &opts).unwrap()).collect();
    let merged = merge_query_phases(req, &qps);
    let hits = fetch_page(&handles.iter().collect::<Vec<_>>(), &merged, req).unwrap();
    let hits: Vec<Value> = hits.iter().map(|x| serde_json::from_str(x.get()).unwrap()).collect();
    let total = merged.total.map(|t| json!({"value": t.value, "relation": if t.relation == phases::TotalRelation::Gte { "gte" } else { "eq" }}));
    let aggs = req.aggs.as_ref().map(|_| AggPartial::reduce(qps.iter().map(|q| q.aggs.clone().unwrap()).collect(), &AggRequest::from_search(req, 65_536, 0).unwrap()).unwrap());
    (json!({"hits": {"total": total, "hits": hits}}), aggs)
}

#[test]
fn n_shard_search_equals_single_shard_reference() {
    let (_t, _n, multi, single) = pair();
    let field_sorted = [
        json!({"query": {"match_all": {}}, "sort": [{"n": "desc"}], "size": 15, "track_total_hits": true}),
        json!({"query": {"range": {"n": {"gte": 17}}}, "sort": [{"tag": "asc"}, {"n": "asc"}], "from": 7, "size": 20, "track_total_hits": true}),
        json!({"query": {"match": {"title": "fox"}}, "sort": [{"p": "asc"}, {"n": "desc"}], "size": 25, "track_total_hits": true}),
        json!({"query": {"term": {"tag": "t3"}}, "sort": [{"n": "asc"}], "size": 100, "_source": ["n"], "track_total_hits": true}),
        json!({"query": {"match_all": {}}, "collapse": {"field": "tag"}, "sort": [{"n": "desc"}], "size": 10}),
    ];
    for body in field_sorted {
        let req = parse(body.clone(), &[]);
        let (m, s) = (exec(&multi, &req), exec(&single, &req));
        assert_eq!(ids(&m), ids(&s), "{body}");
        assert_eq!(m["hits"]["total"], s["hits"]["total"], "{body}");
        assert_eq!(m["_shards"]["total"], json!(3));
        // The phases API (what the cluster coordinator runs) agrees.
        let (p, _) = phased(&multi, &req, false);
        assert_eq!(ids(&p), ids(&s), "phased {body}");
        assert_eq!(p["hits"]["total"], s["hits"]["total"], "phased {body}");
    }
    // Scored: DFS statistics reproduce the single-shard scores exactly.
    for body in [
        // Every hit (score ties cut by `size` break by shard / doc order).
        json!({"query": {"match": {"title": "quick fox"}}, "size": 200, "track_total_hits": true}),
        json!({"query": {"bool": {"must": {"match": {"title": "dog"}}, "filter": {"range": {"n": {"lt": 90}}}}}, "size": 200}),
    ] {
        let req = parse(body.clone(), &[("search_type", "dfs_query_then_fetch")]);
        let (m, s) = (exec(&multi, &req), exec(&single, &req));
        assert_eq!(scored(&m), scored(&s), "{body}");
        assert_eq!(m["hits"]["total"], s["hits"]["total"]);
        assert_eq!(m["hits"]["max_score"], s["hits"]["max_score"]);
        let (p, _) = phased(&multi, &req, true);
        assert_eq!(scored(&p), scored(&s), "phased dfs {body}");
        // Without DFS each shard scores with its own statistics (ES
        // query_then_fetch): same hits matched, scores differ.
        let plain = exec(&multi, &parse(body.clone(), &[]));
        assert_eq!(plain["hits"]["total"], s["hits"]["total"]);
    }
}

/// Supplied DFS statistics (a cluster coordinator's merged stats round):
/// every shard copy, searched on its own, scores like the single-shard
/// reference.
#[test]
fn supplied_dfs_stats_score_like_the_single_shard_reference() {
    let (_t, _n, multi, single) = pair();
    let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    for body in [
        json!({"query": {"match": {"title": "quick fox"}}, "size": 200, "track_total_hits": true}),
        json!({"query": {"bool": {"must": {"match": {"title": "dog"}}, "filter": {"range": {"n": {"lt": 90}}}}}, "size": 200}),
    ] {
        let plain = parse(body.clone(), &[]);
        let reference = exec(&single, &plain);
        // The stats round over every shard (summed) equals one shard's view of all docs.
        let global = crate::search::dfs_phase(&[Target::new(multi.clone())], &plain, &opts).unwrap();
        assert_eq!(global, crate::search::dfs_phase(&[Target::new(single.clone())], &plain, &opts).unwrap(), "{body}");
        assert!(!global.terms.is_empty() && !global.fields.is_empty(), "{body}");
        // Serializable (it crosses the cluster transport).
        let wire: phases::DfsStats = serde_json::from_str(&serde_json::to_string(&global).unwrap()).unwrap();
        assert_eq!(wire, global);
        let mut with = plain.clone();
        with.dfs_stats = Some(Arc::new(wire));
        // Whole index, no `search_type`: supplied stats are used.
        assert_eq!(scored(&exec(&multi, &with)), scored(&reference), "{body}");
        // One shard copy at a time (how the cluster calls each 1-shard engine).
        let want: std::collections::HashMap<String, String> = scored(&reference).into_iter().collect();
        let mut seen = 0;
        for s in multi.local_shards() {
            let t = Target { shard: Some(s.clone()), ..Target::new(multi.clone()) };
            let out: Value = serde_json::from_str(&crate::search::execute(&[t], &with, &opts).unwrap().json).unwrap();
            for (id, score) in scored(&out) {
                assert_eq!(Some(&score), want.get(&id), "{body} shard {} doc {id}", s.id);
                seen += 1;
            }
        }
        assert_eq!(seen, want.len(), "{body}");
    }
}

#[test]
fn n_shard_aggregations_reduce_to_the_single_shard_result() {
    let (_t, _n, multi, single) = pair();
    let aggs = json!({
        "tags": {"terms": {"field": "tag", "size": 10}, "aggs": {"avg_p": {"avg": {"field": "p"}}, "mx": {"max": {"field": "n"}}}},
        "h": {"histogram": {"field": "n", "interval": 25}, "aggs": {"s": {"sum": {"field": "p"}}, "cs": {"cumulative_sum": {"buckets_path": "s"}}}},
        "st": {"stats": {"field": "p"}}, "c": {"cardinality": {"field": "tag"}}, "vc": {"value_count": {"field": "n"}},
        "r": {"range": {"field": "n", "ranges": [{"to": 30}, {"from": 30, "to": 80}, {"from": 80}]}},
        "f": {"filter": {"term": {"tag": "t1"}}, "aggs": {"mn": {"min": {"field": "p"}}}},
        "sb": {"sum_bucket": {"buckets_path": "tags>_count"}}
    });
    for q in [json!({"match_all": {}}), json!({"range": {"n": {"gte": 13}}}), json!({"match": {"title": "fox"}})] {
        let req = parse(json!({"size": 0, "query": q, "aggs": aggs, "track_total_hits": true}), &[]);
        let (m, s) = (exec(&multi, &req), exec(&single, &req));
        // Every bucket count / metric here is exact in ES across shards
        // (terms: fewer terms than shard_size, so no count error).
        let mut ma = m["aggregations"].clone();
        let sa = s["aggregations"].clone();
        assert_eq!(ma["tags"]["doc_count_error_upper_bound"], json!(0));
        ma["tags"]["doc_count_error_upper_bound"] = sa["tags"]["doc_count_error_upper_bound"].clone();
        assert_eq!(ma, sa, "{q}");
        assert_eq!(m["hits"]["total"], s["hits"]["total"]);
        let (_, pa) = phased(&multi, &req, false);
        let mut pa = pa.unwrap();
        pa["tags"]["doc_count_error_upper_bound"] = sa["tags"]["doc_count_error_upper_bound"].clone();
        assert_eq!(pa, sa, "phased {q}");
    }
}

#[test]
fn n_shard_knn_uses_dfs_and_equals_the_single_shard_result() {
    let (_t, _n, multi, single) = pair();
    let qv = vec_of(10);
    for body in [
        json!({"knn": {"field": "v", "query_vector": qv, "k": 5, "num_candidates": 120}, "size": 5}),
        json!({"knn": {"field": "v", "query_vector": qv, "k": 4, "num_candidates": 120, "filter": {"term": {"tag": "t1"}}}, "size": 10}),
        json!({"knn": [{"field": "v", "query_vector": qv, "k": 3, "num_candidates": 120}, {"field": "v", "query_vector": vec_of(33), "k": 3, "num_candidates": 120, "boost": 2.0}], "size": 10}),
        json!({"knn": {"field": "v", "query_vector": qv, "k": 6, "num_candidates": 120}, "size": 3, "from": 2, "aggs": {"t": {"terms": {"field": "tag"}}}}),
    ] {
        let req = parse(body.clone(), &[]);
        let (m, s) = (exec(&multi, &req), exec(&single, &req));
        // The global top k, not k per shard.
        assert_eq!(scored(&m), scored(&s), "{body}");
        assert_eq!(m["hits"]["total"], s["hits"]["total"], "{body}");
        if req.aggs.is_some() {
            let mut a = m["aggregations"].clone();
            a["t"]["doc_count_error_upper_bound"] = s["aggregations"]["t"]["doc_count_error_upper_bound"].clone();
            assert_eq!(a, s["aggregations"], "{body}");
        }
        let (p, _) = phased(&multi, &req, false);
        assert_eq!(scored(&p), scored(&s), "phased {body}");
    }
}

#[test]
fn search_routing_and_slices_follow_shards() {
    let (_t, _n, multi, _single) = pair();
    // ?routing=x searches only the shard x routes to.
    let mut t = Target::new(multi.clone());
    t.routing = Some("d5".into());
    let shard = multi.route("d5", None).unwrap();
    let req = parse(json!({"size": 200, "track_total_hits": true}), &[]);
    let out = crate::search::execute(&[t], &req, &ExecOpts { terms_lookup: None, cursors: None, max_buckets: 100, scroll_id: None, pit_id: None }).unwrap();
    let v: Value = serde_json::from_str(&out.json).unwrap();
    assert_eq!(v["_shards"]["total"], json!(1));
    assert_eq!(v["hits"]["total"]["value"].as_u64().unwrap(), multi.shard(shard).unwrap().doc_count_realtime().0);
    // Slices partition the documents (max < shards: whole shards; max > shards: split).
    for max in [2u32, 3, 5] {
        let mut seen = std::collections::BTreeSet::new();
        let mut total = 0;
        for id in 0..max {
            let r = parse(json!({"size": 200, "slice": {"id": id, "max": max}}), &[]);
            let v = exec(&multi, &r);
            for x in ids(&v) {
                assert!(seen.insert(x));
                total += 1;
            }
        }
        assert_eq!(total, 120, "max {max}");
    }
}

#[test]
fn split_shrink_clone_copy_segments_and_route() {
    let (_t, node) = node();
    let src = node.create_index("src", &json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0}})).unwrap();
    src.write((0..80).map(|i| idx(&format!("d{i}"), json!({"n": i}))).collect(), true);
    src.put_settings(settings::normalize_index_settings(&json!({"index.blocks.write": true})), false).unwrap();
    use crate::ilm::resize::{resize, ResizeType};
    resize(&node, "src", "cl", ResizeType::Clone, &json!({})).unwrap();
    resize(&node, "src", "sp", ResizeType::Split, &json!({"settings": {"index.number_of_shards": 4}})).unwrap();
    resize(&node, "src", "sh", ResizeType::Shrink, &json!({"settings": {"index.number_of_shards": 1}})).unwrap();
    for (name, n) in [("cl", 2), ("sp", 4), ("sh", 1)] {
        let t = node.get_index(name).unwrap();
        assert_eq!(t.number_of_shards(), n);
        assert_eq!(t.doc_count_realtime().0, 80, "{name}");
        // Searchable right away, like ES's freshly recovered target shards.
        let targets = node.resolve(name, Default::default()).unwrap();
        assert_eq!(crate::search::count(&targets, None, None).unwrap(), 80, "{name}: searchable");
        for i in 0..80 {
            let id = format!("d{i}");
            let want = t.route(&id, None).unwrap();
            for s in t.local_shards() {
                assert_eq!(s.shard.peek(&id).unwrap().is_some(), s.id == want, "{name} {id}");
            }
            assert_eq!(t.get(&id).unwrap().unwrap().version, 1);
        }
    }
}

#[test]
fn pit_retains_vector_scores_sources_and_deletion_view_after_updates_and_merge() {
    let (_temporary, node) = node();
    let svc = node.create_index("pit-vectors", &json!({
        "settings": {"number_of_shards": 2, "number_of_replicas": 0},
        "mappings": {"properties": {
            "vector": {"type":"dense_vector", "dims":2, "similarity":"l2_norm", "index_options":{"type":"hnsw"}},
            "version_tag": {"type":"keyword"}
        }}
    })).unwrap();
    let writes = svc.write((0..20).map(|i| idx(&format!("d{i}"), json!({"vector":[i as f32,1.0],"version_tag":"old"}))).collect(), true);
    assert!(writes.iter().all(|result| result.is_ok()));
    // `write(..., true)` syncs the translog; refresh publishes search visibility.
    svc.refresh().unwrap();
    let targets = [Target::new(svc.clone())];
    let captured = node.snapshot(&targets).unwrap();
    let request = parse(json!({"size":10,"knn":{"field":"vector","query_vector":[0.0,1.0],"k":10,"num_candidates":20}}), &[]);
    let execute = |targets: &[Target]| {
        let output = crate::search::execute(targets, &request, &ExecOpts {terms_lookup:None,cursors:None,max_buckets:100,scroll_id:None,pit_id:None}).unwrap();
        let value: Value = serde_json::from_str(&output.json).unwrap();
        value["hits"]["hits"].clone()
    };
    let before = execute(&captured);
    assert_eq!(before.as_array().unwrap().len(), 10);
    let writes = svc.write(vec![
        idx("d0", json!({"vector":[1000.0,1.0],"version_tag":"new"})),
        WriteReq::Delete {id:"d1".into(),routing:None,cond:WriteCond::default()},
    ], true);
    assert!(writes.iter().all(|result| result.is_ok()));
    svc.refresh().unwrap();
    svc.force_merge(1, false).unwrap();
    assert_eq!(execute(&captured), before, "old PIT ids, scores and sources must remain stable");
    let current = execute(&node.snapshot(&targets).unwrap());
    assert_ne!(current, before);
    assert!(current.as_array().unwrap().iter().all(|hit| hit["_id"] != "d0" && hit["_id"] != "d1"));
}

#[test]
fn pit_shard_doc_and_scroll_paginate_across_shards() {
    let (_t, node, multi, _single) = pair();
    let snap = node.snapshot(&[Target::new(multi.clone())]).unwrap();
    assert_eq!(snap.len(), 3, "one pinned target per shard");
    // search_after over `_shard_doc` visits every document once.
    let mut seen = std::collections::BTreeSet::new();
    let mut after: Option<Value> = None;
    loop {
        let mut body = json!({"size": 7, "sort": [{"_shard_doc": "asc"}]});
        if let Some(a) = &after {
            body["search_after"] = a.clone();
        }
        let out = crate::search::execute(&snap, &parse(body, &[]), &ExecOpts { terms_lookup: None, cursors: None, max_buckets: 100, scroll_id: None, pit_id: None }).unwrap();
        let v: Value = serde_json::from_str(&out.json).unwrap();
        let hits = v["hits"]["hits"].as_array().unwrap().clone();
        if hits.is_empty() {
            break;
        }
        for h in &hits {
            assert!(seen.insert(h["_id"].as_str().unwrap().to_string()), "duplicate {}", h["_id"]);
        }
        after = Some(hits.last().unwrap()["sort"].clone());
    }
    assert_eq!(seen.len(), 120);
    // Scroll: per-shard cursors.
    let req = parse(json!({"size": 9, "sort": [{"n": "asc"}]}), &[]);
    let mut cursors: Option<Vec<crate::search::Cursor>> = None;
    let mut got = Vec::new();
    loop {
        let out = crate::search::execute(&snap, &req, &ExecOpts { terms_lookup: None, cursors: cursors.as_deref(), max_buckets: 100, scroll_id: None, pit_id: None }).unwrap();
        let v: Value = serde_json::from_str(&out.json).unwrap();
        let page = ids(&v);
        if page.is_empty() {
            break;
        }
        got.extend(page);
        cursors = Some(out.cursors);
    }
    let want: Vec<String> = (0..120).map(|i| format!("d{i}")).collect();
    assert_eq!(got, want);
}

/// search_after over a sort with ties and missing values (rendered as the
/// long sentinel and read back as missing) plus `_shard_doc` visits every
/// document once, in sort order.
#[test]
fn pit_search_after_with_ties_and_missing_values_is_exact() {
    let (_t, node) = node();
    let svc = node.create_index("pm", &json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0}, "mappings": {"properties": {"n": {"type": "long"}}}})).unwrap();
    // Two segments per shard; a fifth of the docs have no `n`, some hold a
    // real Long.MAX / Long.MIN (equal to a missing value's sentinel), the
    // rest few values.
    let value = |i: i64| if i % 5 == 0 { json!({}) } else if i % 7 == 0 { json!({"n": i64::MAX}) } else if i % 11 == 0 { json!({"n": i64::MIN}) } else { json!({"n": i % 3}) };
    for half in [0..40, 40..80] {
        let docs = half.map(|i| idx(&format!("d{i}"), value(i))).collect();
        assert!(svc.write(docs, true).iter().all(|r| r.is_ok()));
        svc.refresh().unwrap();
    }
    let snap = node.snapshot(&[Target::new(svc.clone())]).unwrap();
    let missing = (0..80).filter(|i| i % 5 == 0).count();
    for (order, sentinel) in [("asc", i64::MAX), ("desc", i64::MIN)] {
        let (mut seen, mut keys) = (std::collections::BTreeSet::new(), Vec::new());
        let mut after: Option<Value> = None;
        loop {
            let mut body = json!({"size": 7, "sort": [{"n": order}, {"_shard_doc": "asc"}]});
            if let Some(a) = &after {
                body["search_after"] = a.clone();
            }
            let out = crate::search::execute(&snap, &parse(body, &[]), &ExecOpts { terms_lookup: None, cursors: None, max_buckets: 100, scroll_id: None, pit_id: None }).unwrap();
            let v: Value = serde_json::from_str(&out.json).unwrap();
            let hits = v["hits"]["hits"].as_array().unwrap().clone();
            if hits.is_empty() {
                break;
            }
            for h in &hits {
                assert!(seen.insert(h["_id"].as_str().unwrap().to_string()), "{order}: duplicate {h}");
                keys.push(h["sort"][0].as_i64().unwrap());
            }
            after = Some(hits.last().unwrap()["sort"].clone());
        }
        assert_eq!(seen.len(), 80, "{order}");
        let sorted = if order == "asc" { keys.windows(2).all(|w| w[0] <= w[1]) } else { keys.windows(2).all(|w| w[0] >= w[1]) };
        assert!(sorted, "{order}: {keys:?}");
        // ES renders a missing value as its sentinel (Long.MAX ascending,
        // Long.MIN descending); real values equal to it come out with them.
        let real = (0..80).filter(|i| i % 5 != 0 && value(*i)["n"].as_i64() == Some(sentinel)).count();
        assert_eq!(keys.iter().filter(|k| **k == sentinel).count(), missing + real, "{order}");
    }
}

#[test]
fn alias_routing_selects_shards() {
    let (_t, node) = node();
    let svc = node.create_index("ar", &json!({"settings": {"number_of_shards": 4}, "aliases": {"routed": {"routing": "k1"}, "plain": {}}})).unwrap();
    assert_eq!(node.index_routing_for("routed", None).unwrap().as_deref(), Some("k1"));
    assert_eq!(node.index_routing_for("routed", Some("k1")).unwrap().as_deref(), Some("k1"));
    assert_eq!(node.index_routing_for("routed", Some("other")).unwrap_err().kind, "illegal_argument_exception");
    assert_eq!(node.index_routing_for("plain", Some("x")).unwrap().as_deref(), Some("x"));
    assert_eq!(node.index_routing_for("ar", None).unwrap(), None);
    // A search through the routed alias visits only k1's shard.
    let t = node.resolve("routed", Default::default()).unwrap();
    assert_eq!(t[0].routing.as_deref(), Some("k1"));
    assert_eq!(crate::search::expand_targets(&t).unwrap().len(), 1);
    assert_eq!(crate::search::expand_targets(&node.resolve("ar", Default::default()).unwrap()).unwrap().len(), 4);
    assert_eq!(crate::search::expand_targets(&node.resolve("routed,ar", Default::default()).unwrap()).unwrap().len(), 4, "a direct reference searches every shard");
    let _ = svc;
}

/// ES `flushOnIdle`: the maintenance tick commits a shard that holds
/// uncommitted ops once no write came for `indices.memory.shard_inactive_time`,
/// and leaves a shard that is still being written to alone.
#[test]
fn maintenance_flushes_idle_shards() {
    let (_t, node) = node();
    node.node_settings.write().insert("indices.memory.shard_inactive_time".into(), json!("200ms"));
    let svc = node.create_index("idle", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}})).unwrap();
    svc.write((0..5).map(|i| idx(&format!("d{i}"), json!({"n": i}))).collect(), true);
    let sh = svc.local_shards().remove(0);
    let pending = || sh.shard.max_seq_no() > sh.shard.committed_seq_no();
    assert!(pending());
    node.maintenance(); // first sighting: not idle yet
    assert!(pending(), "flushed before the inactive time");
    std::thread::sleep(std::time::Duration::from_millis(300));
    node.maintenance();
    assert!(!pending(), "idle shard was not flushed");
    // A write resets the clock.
    svc.write(vec![idx("late", json!({"n": 9}))], true).remove(0).unwrap();
    node.maintenance();
    assert!(pending(), "a fresh write is not idle");
}

/// ES `index.translog.flush_threshold_age`: a copy that keeps being
/// written to (never idle) is still committed once its last flush is older
/// than the age, and not before.
#[test]
fn maintenance_flushes_busy_shards_by_age() {
    let (_t, node) = node();
    node.node_settings.write().insert("indices.memory.shard_inactive_time".into(), json!("10m"));
    let svc = node.create_index("busy", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.translog.flush_threshold_age": "1s"}})).unwrap();
    let sh = svc.local_shards().remove(0);
    sh.shard.flush().unwrap();
    let pending = || sh.shard.max_seq_no() > sh.shard.committed_seq_no();
    svc.write(vec![idx("a", json!({"n": 1}))], true).remove(0).unwrap();
    node.maintenance();
    assert!(pending(), "flushed before the age");
    for i in 0..6 {
        std::thread::sleep(std::time::Duration::from_millis(250));
        svc.write(vec![idx(&format!("b{i}"), json!({"n": i}))], true).remove(0).unwrap();
    }
    node.maintenance();
    assert!(!pending(), "a busy shard past the flush age was not flushed");
}

/// A policy-only force merge flushes first and merges after, then no write
/// comes: as ES (`maybeFlushAfterMerge` / `flushOnIdle`), the idle tick
/// commits the merge output, the sources leave the commit, `_cat/segments`
/// and the disk; before the inactive time they stay committed, unsearched.
#[test]
fn idle_tick_commits_policy_merges_and_frees_their_sources() {
    let (_t, node) = node();
    node.node_settings.write().insert("indices.memory.shard_inactive_time".into(), json!("200ms"));
    // A wide tier: the node's background merges leave the 12 flushed
    // segments alone, so the one merge below is the only one, and it runs
    // after the last commit (as a policy merge after a force merge's flush
    // does; `flush: false` keeps its output uncommitted).
    let svc = node.create_index("merged", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.merge.policy.segments_per_tier": 50}})).unwrap();
    let sh = svc.local_shards().remove(0);
    for round in 0..12 {
        for r in svc.write(vec![idx(&format!("d{round}"), json!({"n": round}))], true) { r.unwrap(); }
        sh.shard.flush().unwrap();
    }
    svc.refresh().unwrap();
    let committed_before: Vec<u32> = sh.shard.reader_view().segment_stats().iter().filter(|s| s.committed && s.search).map(|s| s.id).collect();
    assert!(committed_before.len() >= 12, "{:?}", sh.shard.reader_view().segment_stats());
    sh.shard.force_merge_with(1, false, false).unwrap();
    svc.refresh().unwrap();
    let on_disk = |id: u32| sh.shard.core.read().directory().exists(&rustic_core::codec::segment_file_name(id));
    let stats = sh.shard.reader_view().segment_stats();
    let sources: Vec<u32> = stats.iter().filter(|s| s.committed && !s.search).map(|s| s.id).collect();
    assert!(!sources.is_empty(), "the policy merged nothing: {stats:?}");
    assert!(stats.iter().any(|s| s.search && !s.committed), "merge output not yet committed");
    assert!(sh.shard.has_uncommitted_merges() && sh.shard.merged_away_committed_bytes() > 0);
    // Within the inactive time: sources stay (the commit on disk lists them).
    node.maintenance();
    assert!(sources.iter().all(|&id| on_disk(id)));
    assert!(sh.shard.reader_view().segment_stats().iter().any(|s| s.committed && !s.search));
    std::thread::sleep(std::time::Duration::from_millis(300));
    node.maintenance();
    let after = sh.shard.reader_view().segment_stats();
    assert!(after.iter().all(|s| s.committed && s.search), "{after:?}");
    assert!(sources.iter().all(|&id| !on_disk(id)), "merged-away sources still on disk");
    assert!(after.iter().all(|s| s.bytes > 0), "segment sizes reported: {after:?}");
    assert_eq!(after.iter().map(|s| s.docs - s.deleted).sum::<u64>(), 12);
    assert!(!sh.shard.has_uncommitted_merges());
    // The store size follows: only live segment files (and the translog) remain.
    let live: u64 = after.iter().map(|s| s.bytes).sum();
    assert!(sh.shard.store_size() >= live);
    drop(sh);
    drop(svc);
}

#[test]
fn scoped_read_targets_search_one_copy_and_restore_worker_state() {
    let (_tmp, node) = node();
    let svc = node.create_index("scoped", &json!({"settings":{"number_of_shards":3,"number_of_replicas":0}})).unwrap();
    for result in svc.write((0..30).map(|i| idx(&format!("scope-{i}"), json!({"n":i}))).collect(), true) { result.unwrap(); }
    svc.refresh().unwrap();
    let resolve = || node.resolve("scoped", crate::node::ResolveOpts::default()).unwrap();
    let total = crate::search::count(&resolve(), None, None).unwrap();
    assert_eq!(total, 30);
    let mut sum = 0;
    for k in 0..3 {
        let _guard = crate::search::ReadShardScope::enter(Some(k));
        let targets = resolve();
        assert_eq!(targets[0].shard.as_ref().unwrap().id, k);
        sum += crate::search::count(&targets, None, None).unwrap();
        {
            let _nested = crate::search::ReadShardScope::enter(Some((k+1)%3));
            assert_eq!(resolve()[0].shard.as_ref().unwrap().id, (k+1)%3);
        }
        assert_eq!(resolve()[0].shard.as_ref().unwrap().id, k);
    }
    assert_eq!(sum, total);
    assert!(resolve()[0].shard.is_none(), "selection leaked into a later request on this worker");
}

/// Background flush / merge (`Shard::run_background`) under load: writers
/// (indexing, overwriting and deleting), scheduled + explicit refreshes,
/// flushes, force merges, point-in-time snapshots and realtime gets run
/// concurrently on a multi-shard index with a tiny seal threshold, so
/// memtables are sealed and segments merged on the workers all the time.
/// Nothing may deadlock (the bulk / refresh / flush hangs this guards
/// against), and the final contents match a model of the writes.
#[test]
fn concurrent_writes_refresh_flush_and_background_merges() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let (_t, node) = node();
    let svc = node.create_index("bg", &json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0, "refresh_interval": "30ms"}})).unwrap();
    for s in svc.local_shards() {
        s.shard.bg.threshold.store(300, Ordering::Relaxed);
    }
    const WRITERS: usize = 4;
    const ROUNDS: usize = 40;
    const IDS: usize = 1500;
    let stop = Arc::new(AtomicBool::new(false));
    let forced = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let (tx, rx) = std::sync::mpsc::channel();
    let (svc2, node2, stop2, forced2) = (svc.clone(), node.clone(), stop.clone(), forced.clone());
    std::thread::spawn(move || {
        let svc = svc2;
        let models: Vec<std::collections::BTreeMap<String, i64>> = std::thread::scope(|sc| {
            let writers: Vec<_> = (0..WRITERS)
                .map(|t| {
                    let svc = &svc;
                    sc.spawn(move || {
                        let mut model = std::collections::BTreeMap::new();
                        for r in 0..ROUNDS {
                            let reqs: Vec<WriteReq> = (0..150)
                                .map(|i| {
                                    let n = ((r * 150 + i) * 7) % IDS;
                                    let id = format!("w{t}-{n}");
                                    model.insert(id.clone(), (r * 1000 + i) as i64);
                                    idx(&id, json!({"n": r * 1000 + i, "w": t, "body": format!("round {r} item {i} writer {t}")}))
                                })
                                .collect();
                            for x in svc.write(reqs, r % 4 == 0) {
                                x.expect("write failed");
                            }
                            if r % 3 == 2 {
                                let dels: Vec<WriteReq> = (0..40)
                                    .map(|i| {
                                        let id = format!("w{t}-{}", (r * 31 + i * 13) % IDS);
                                        model.remove(&id);
                                        WriteReq::Delete { id, routing: None, cond: WriteCond::default() }
                                    })
                                    .collect();
                                let _ = svc.write(dels, false);
                            }
                        }
                        model
                    })
                })
                .collect();
            // Auto-id writers: append-only batches indexed in parallel.
            let auto: Vec<_> = (0..2)
                .map(|t| {
                    let svc = &svc;
                    sc.spawn(move || {
                        for r in 0..ROUNDS {
                            let reqs: Vec<WriteReq> = (0..120).map(|i| WriteReq::Index { id: None, source: json!({"a": t, "r": r, "i": i}).to_string(), routing: None, cond: WriteCond::default() }).collect();
                            for x in svc.write(reqs, r % 3 == 0) {
                                x.expect("auto-id write failed");
                            }
                        }
                    })
                })
                .collect();
            let svc = &svc;
            let node = &node2;
            let stop = &stop2;
            let forced = &forced2;
            sc.spawn(move || {
                let mut k = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    k += 1;
                    match k % 5 {
                        0 => svc.flush().unwrap(),
                        1 => svc.refresh().unwrap(),
                        2 => {
                            if k % 25 == 2 {
                                svc.force_merge(2, false).unwrap();
                                forced.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        3 => {
                            let t = node.resolve("bg", Default::default()).unwrap();
                            let snap = node.snapshot(&t).unwrap();
                            drop(snap);
                        }
                        _ => {
                            let _ = svc.get(&format!("w0-{}", k as usize % IDS)).unwrap();
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_millis(3));
                }
            });
            let models = writers.into_iter().map(|h| h.join().unwrap()).collect();
            for h in auto {
                h.join().unwrap();
            }
            stop2.store(true, Ordering::Relaxed);
            models
        });
        let _ = tx.send(models);
    });
    let models = rx.recv_timeout(std::time::Duration::from_secs(300)).expect("concurrent writes / refresh / flush / merges deadlocked");
    stop.store(true, Ordering::Relaxed);
    svc.refresh().unwrap();
    let expected: usize = models.iter().map(|m| m.len()).sum::<usize>() + 2 * ROUNDS * 120;
    assert_eq!(svc.doc_count_realtime().0 as usize, expected, "live docs after the run");
    for m in &models {
        for (id, n) in m.iter().step_by(37) {
            let g = svc.get(id).unwrap().unwrap_or_else(|| panic!("{id} lost"));
            let v: Value = serde_json::from_str(g.source.as_deref().unwrap()).unwrap();
            assert_eq!(v["n"].as_i64(), Some(*n), "{id}: latest version");
        }
    }
    // The workers sealed and merged in the background (force merges count
    // one merge per shard that needed one).
    // A merge waits for a node-wide permit other tests' merges may hold:
    // give the workers time rather than reading the count once.
    let merged = || svc.local_shards().iter().map(|s| s.shard.counters.merge_total.load(Ordering::Relaxed)).sum::<u64>();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while merged() <= 2 * forced.load(Ordering::Relaxed) && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let merges = merged();
    assert!(merges > 2 * forced.load(Ordering::Relaxed), "no background merge ran ({merges} merges)");
    svc.flush().unwrap();
    assert_eq!(svc.doc_count_realtime().0 as usize, expected);
}


#[cfg(feature = "mutable-hnsw")]
#[test]
fn mutable_vectors_follow_es_writes_filters_pit_and_reopen() {
    let (tmp, node) = node();
    let mapping: Value = serde_json::from_str(MAPPING).unwrap();
    let create = |name| node.create_index(name, &json!({"settings": {
        "number_of_shards": 2, "number_of_replicas": 0}, "mappings": mapping})).unwrap();
    let mutable = create("mutable");
    let baseline = create("baseline");
    for shard in mutable.local_shards() { shard.shard.core.write().enable_mutable_vectors().unwrap(); }
    for svc in [&mutable, &baseline] {
        let results = svc.write((0..120).map(|i| idx(&format!("d{i}"), doc(i))).collect(), true);
        assert!(results.iter().all(|r| r.is_ok()));
        svc.refresh().unwrap();
    }
    let requests: Vec<_> = [None, Some(json!({"term": {"tag": "t1"}})),
        Some(json!({"range": {"n": {"lt": 3}}}))].into_iter().map(|filter| {
        let mut knn = json!({"field": "v", "query_vector": vec_of(10), "k": 5, "num_candidates": 120});
        if let Some(filter) = filter { knn["filter"] = filter; }
        parse(json!({"knn": knn, "size": 5}), &[])
    }).collect();
    let check = |a: &Arc<IndexService>, b: &Arc<IndexService>| {
        for request in &requests {
            let (a,b) = (exec(a, request), exec(b, request));
            assert_eq!(ids(&a), ids(&b));
            for (a,b) in a["hits"]["hits"].as_array().unwrap().iter().zip(b["hits"]["hits"].as_array().unwrap()) {
                assert!((a["_score"].as_f64().unwrap()-b["_score"].as_f64().unwrap()).abs() < 1e-5);
            }
        }
    };
    check(&mutable, &baseline);
    let retained = node.snapshot(&[Target::new(mutable.clone())]).unwrap();
    let before = exec(&mutable, &requests[0]);
    for svc in [&mutable, &baseline] {
        svc.write(vec![idx("d10", doc(501)), WriteReq::Delete {
            id: "d11".into(), routing: None, cond: WriteCond::default()}], true)
            .into_iter().for_each(|r| { r.unwrap(); });
        svc.refresh().unwrap();
        svc.flush().unwrap();
    }
    check(&mutable, &baseline);
    let old = crate::search::execute(&retained, &requests[0], &ExecOpts {
        terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None }).unwrap();
    let old: Value = serde_json::from_str(&old.json).unwrap();
    assert_eq!(ids(&old), ids(&before));
    drop(retained); drop(mutable); drop(baseline); drop(node);
    let reopened = Node::open(tmp.path()).unwrap();
    check(&reopened.get_index("mutable").unwrap(), &reopened.get_index("baseline").unwrap());
}

#[test]
fn unpaired_source_surrogates_index_without_rewriting_stored_source() {
    let (_t, node) = node();
    let svc = node.create_index("surrogate", &json!({"mappings":{"properties":{"body":{"type":"text"}}}})).unwrap();
    let raw = r#"{"body":"before \ud83d after \ude02 pair \ud83d\ude02"}"#;
    let result = svc.write(vec![WriteReq::Index { id: Some("one".into()), source: raw.into(), routing: None, cond: WriteCond::default() }], true);
    assert!(result[0].is_ok(), "{result:?}");
    assert_eq!(svc.get("one").unwrap().unwrap().source.as_deref(), Some(raw));
    assert!(svc.parse_for_replica(raw, "one").is_ok());
    assert!(svc.prepare_replica_sources(&vec![("one", raw); 32]).is_some());
    svc.flush().unwrap();
    assert_eq!(svc.get("one").unwrap().unwrap().source.as_deref(), Some(raw));
}

#[test]
fn deleting_index_drains_unpublished_copy_and_rejects_old_preparation() {
    let (_t, node) = node();
    let svc = node.create_index("pending-delete", &json!({})).unwrap();
    svc.close_shard(0).unwrap();
    let mut prepared = svc.prepare_shard(0, ShardOpenSpec::local_primary(RecoverySource::ExistingStore)).unwrap();
    let copy = svc.prepared_shard_copy(&prepared).unwrap();
    let permit = copy.operation_permit().unwrap();
    let path = svc.path().to_path_buf();
    let deleting = node.clone();
    let thread = std::thread::spawn(move || deleting.delete_index("pending-delete"));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !copy.is_closed() && std::time::Instant::now() < deadline {
        std::thread::yield_now();
    }
    let fenced = copy.is_closed();
    let existed_while_draining = path.exists();
    drop(permit);
    thread.join().unwrap().unwrap();
    assert!(fenced, "deletion must fence a copy absent from local_shards");
    assert!(existed_while_draining, "cannot remove files under an active WAL operation");
    assert!(!path.exists());
    assert!(svc.try_install_prepared_shard(&mut prepared).is_err());
    assert!(svc.prepare_shard(0, ShardOpenSpec::local_primary(RecoverySource::ExistingStore)).is_err());
    assert!(!path.exists(), "stale preparation cannot recreate a deleted directory");
    let replacement = node.create_index("pending-delete", &json!({})).unwrap();
    replacement.write(vec![idx("new", json!({"v": 7}))], true).remove(0).unwrap();
    drop(prepared);
    assert!(replacement.get("new").unwrap().is_some());
    assert!(!replacement.shard(0).unwrap().is_closed());
}

/// Append-only (auto-id) bulks index into checked-out memtables without the
/// core lock. A memtable holding any document must never be checked out:
/// while it is out, updates and deletes by id cannot find its documents and
/// would leave a duplicate live copy or a surviving deleted doc. Concurrent
/// auto-id bulks, explicit-id writes (updates and deletes), and updates and
/// deletes by id of previously auto-id-indexed documents; after a refresh
/// every acknowledged state is exact: no duplicate live ids, no surviving
/// deleted docs, nothing missing.
#[test]
fn append_only_checkout_keeps_every_indexed_doc_addressable() {
    use std::collections::{BTreeMap, HashMap, HashSet};
    use std::sync::atomic::{AtomicBool, Ordering};
    let (_t, node) = node();
    let svc = node.create_index("checkout-vis", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "20ms"}})).unwrap();
    for s in svc.local_shards() {
        s.shard.bg.threshold.store(500, Ordering::Relaxed);
    }
    const ROUNDS: usize = 25;
    fn lcg(x: &mut u64) -> u64 {
        *x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x >> 33
    }
    let auto_ids: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let done = Arc::new(AtomicBool::new(false));
    let (tx, rx) = std::sync::mpsc::channel();
    let (svc2, ids2, done2) = (svc.clone(), auto_ids.clone(), done.clone());
    std::thread::spawn(move || {
        let svc = svc2;
        let out = std::thread::scope(|sc| {
            let (svc, auto_ids, done) = (&svc, &ids2, &done2);
            // Auto-id bulks (append-only path).
            let bulks: Vec<_> = (0..3).map(|t| sc.spawn(move || {
                for r in 0..ROUNDS {
                    let reqs: Vec<WriteReq> = (0..40).map(|i| WriteReq::Index { id: None, source: json!({"t": t, "r": r, "i": i, "v": 0}).to_string(), routing: None, cond: WriteCond::default() }).collect();
                    let acked: Vec<String> = svc.write(reqs, r % 3 == 0).into_iter().map(|x| x.expect("auto-id write failed").id).collect();
                    auto_ids.lock().extend(acked);
                }
            })).collect();
            // Explicit ids: index, overwrite, delete.
            let explicit = sc.spawn(move || {
                let mut live: BTreeMap<String, i64> = BTreeMap::new();
                let mut rng = 5u64;
                for r in 0..ROUNDS * 4 {
                    let reqs: Vec<WriteReq> = (0..20).map(|i| {
                        let id = format!("e-{}", lcg(&mut rng) % 200);
                        live.insert(id.clone(), (r * 100 + i) as i64);
                        idx(&id, json!({"v": r * 100 + i}))
                    }).collect();
                    for x in svc.write(reqs, false) { x.expect("explicit write failed"); }
                    let dels: Vec<WriteReq> = (0..5).map(|_| {
                        let id = format!("e-{}", lcg(&mut rng) % 200);
                        live.remove(&id);
                        WriteReq::Delete { id, routing: None, cond: WriteCond::default() }
                    }).collect();
                    for x in svc.write(dels, false) { x.expect("explicit delete failed"); }
                }
                live
            });
            // Updates and deletes by id of acknowledged auto-id documents,
            // racing later auto-id bulks.
            let by_id = sc.spawn(move || {
                let mut updated: HashSet<String> = HashSet::new();
                let mut deleted: HashSet<String> = HashSet::new();
                let mut rng = 9u64;
                while !done.load(Ordering::Relaxed) {
                    let pick: Vec<String> = {
                        let ids = auto_ids.lock();
                        if ids.is_empty() { Vec::new() } else { (0..8).map(|_| ids[lcg(&mut rng) as usize % ids.len()].clone()).collect() }
                    };
                    if pick.is_empty() {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                        continue;
                    }
                    let mut reqs = Vec::new();
                    for (k, id) in pick.into_iter().enumerate() {
                        if deleted.contains(&id) {
                            continue;
                        }
                        if k % 3 == 0 {
                            deleted.insert(id.clone());
                            reqs.push(WriteReq::Delete { id, routing: None, cond: WriteCond::default() });
                        } else {
                            updated.insert(id.clone());
                            reqs.push(idx(&id, json!({"v": 1})));
                        }
                    }
                    for x in svc.write(reqs, false) { x.expect("update / delete by id failed"); }
                }
                (updated, deleted)
            });
            for b in bulks { b.join().unwrap(); }
            let live = explicit.join().unwrap();
            done.store(true, Ordering::Relaxed);
            let (updated, deleted) = by_id.join().unwrap();
            (live, updated, deleted)
        });
        let _ = tx.send(out);
    });
    let (explicit_live, _updated, deleted) = rx.recv_timeout(std::time::Duration::from_secs(300)).expect("checkout visibility test deadlocked");
    svc.refresh().unwrap();
    let auto: Vec<String> = auto_ids.lock().clone();
    let mut want: HashSet<String> = explicit_live.keys().cloned().collect();
    want.extend(auto.iter().filter(|id| !deleted.contains(*id)).cloned());
    let out = crate::search::execute(&[Target::new(svc.clone())], &parse(json!({"size": 10_000, "track_total_hits": true, "_source": false}), &[]),
        &ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None }).unwrap();
    let v: Value = serde_json::from_str(&out.json).unwrap();
    let mut seen: HashMap<String, usize> = HashMap::new();
    for h in v["hits"]["hits"].as_array().unwrap() {
        *seen.entry(h["_id"].as_str().unwrap().to_string()).or_default() += 1;
    }
    let duplicated: Vec<(&String, &usize)> = seen.iter().filter(|(_, n)| **n > 1).collect();
    let survivors: Vec<&String> = seen.keys().filter(|id| !want.contains(*id)).collect();
    let missing: Vec<&String> = want.iter().filter(|id| !seen.contains_key(*id)).collect();
    assert!(duplicated.is_empty() && survivors.is_empty() && missing.is_empty(),
        "after refresh: total {} want {}; duplicated live ids {duplicated:?}; deleted/unknown ids still live {survivors:?}; missing {missing:?}",
        v["hits"]["total"]["value"], want.len());
    assert_eq!(svc.doc_count_realtime().0 as usize, want.len());
}

// ---------------------------------------------------------------- reader views

fn rv_opts() -> ExecOpts<'static> {
    ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None }
}

fn rv_total(targets: &[Target], req: &SearchRequest) -> u64 {
    let out = crate::search::execute(targets, req, &rv_opts()).unwrap();
    let v: Value = serde_json::from_str(&out.json).unwrap();
    v["hits"]["total"]["value"].as_u64().unwrap()
}

/// The multi-index lock-order deadlock: search A over (a, b) and search B
/// over (b, a), with a write queued on each shard. Searches now read
/// published reader views and take no shard lock, so they (and count, PIT
/// open and PIT searches) complete while both cores are write-locked and a
/// write waits on each.
#[test]
fn searches_in_either_index_order_complete_while_writers_hold_both_shards() {
    let (_t, node) = node();
    let settings = json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}});
    let a = node.create_index("rv-a", &settings).unwrap();
    let b = node.create_index("rv-b", &settings).unwrap();
    for svc in [&a, &b] {
        let w = svc.write((0..30).map(|i| idx(&format!("d{i}"), json!({"n": i, "tag": format!("t{}", i % 3)}))).collect(), true);
        assert!(w.iter().all(|r| r.is_ok()));
        svc.refresh().unwrap();
    }
    let body = json!({"size": 100, "track_total_hits": true, "aggs": {"tags": {"terms": {"field": "tag.keyword"}}}});
    let req = parse(body.clone(), &[]);
    let ab = vec![Target::new(a.clone()), Target::new(b.clone())];
    let ba = vec![Target::new(b.clone()), Target::new(a.clone())];
    // Warm up: schemas synced, shards search-active.
    assert_eq!(rv_total(&ab, &req), 60);
    assert_eq!(rv_total(&ba, &req), 60);
    let shards: Vec<Arc<IndexShard>> = [&a, &b].iter().map(|svc| svc.local_shards()[0].clone()).collect();
    let (held_tx, held_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let holder = std::thread::spawn(move || {
        let guards: Vec<_> = shards.iter().map(|s| s.shard.core.write()).collect();
        held_tx.send(()).unwrap();
        let _ = release_rx.recv_timeout(std::time::Duration::from_secs(120));
        drop(guards);
    });
    held_rx.recv().unwrap();
    // A write queued on each shard.
    let writers: Vec<_> = [&a, &b].iter().map(|svc| {
        let svc = (*svc).clone();
        std::thread::spawn(move || svc.write(vec![idx("queued", json!({"n": 99, "tag": "q"}))], true).remove(0).map(|_| ()))
    }).collect();
    std::thread::sleep(std::time::Duration::from_millis(50));
    let (tx, rx) = std::sync::mpsc::channel();
    let mut jobs = 0;
    for targets in [ab.clone(), ba.clone()] {
        let (tx, body, node) = (tx.clone(), body.clone(), node.clone());
        jobs += 1;
        std::thread::spawn(move || {
            let req = parse(body, &[]);
            let total = rv_total(&targets, &req);
            let count = crate::search::count(&targets, None, None).unwrap();
            let pit = node.snapshot(&targets).unwrap();
            let pit_total = rv_total(&pit, &req);
            let _ = tx.send((total, count, pit_total));
        });
    }
    for _ in 0..jobs {
        let got = rx.recv_timeout(std::time::Duration::from_secs(30)).expect("a search waited on a shard lock (reader views must not lock the core)");
        assert_eq!(got, (60, 60, 60));
    }
    release_tx.send(()).unwrap();
    holder.join().unwrap();
    for w in writers {
        w.join().unwrap().unwrap();
    }
    for svc in [&a, &b] {
        svc.refresh().unwrap();
    }
    assert_eq!(rv_total(&ab, &req), 62);
}

/// A mapping update moves the schema generation; the next search must not
/// wait on the core lock to sync it (here a flush-like holder keeps the core
/// write-locked): it searches its view, where the new field has no data.
#[test]
fn search_after_mapping_update_does_not_wait_for_a_held_core() {
    let (_t, node) = node();
    let svc = node.create_index("rv-map", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}})).unwrap();
    assert!(svc.write((0..10).map(|i| idx(&format!("d{i}"), json!({"n": i}))).collect(), true).iter().all(|r| r.is_ok()));
    svc.refresh().unwrap();
    let all = json!({"size": 20, "track_total_hits": true});
    assert_eq!(rv_total(&[Target::new(svc.clone())], &parse(all.clone(), &[])), 10);
    let shard = svc.local_shards()[0].clone();
    let (held_tx, held_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let holder = std::thread::spawn(move || {
        let guard = shard.shard.core.write();
        held_tx.send(()).unwrap();
        let _ = release_rx.recv_timeout(std::time::Duration::from_secs(120));
        drop(guard);
    });
    held_rx.recv().unwrap();
    svc.put_mapping(&json!({"properties": {"extra": {"type": "keyword"}}})).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let svc2 = svc.clone();
    let all2 = all.clone();
    std::thread::spawn(move || {
        let t = [Target::new(svc2.clone())];
        let total = rv_total(&t, &parse(all2, &[]));
        let on_new = rv_total(&t, &parse(json!({"track_total_hits": true, "query": {"term": {"extra": "x"}}}), &[]));
        let _ = tx.send((total, on_new));
    });
    let got = rx.recv_timeout(std::time::Duration::from_secs(30)).expect("search after a mapping update waited on the core lock");
    assert_eq!(got, (10, 0));
    release_tx.send(()).unwrap();
    holder.join().unwrap();
    // Once the core is free, writes and searches use the new field.
    svc.write(vec![idx("e", json!({"n": 99, "extra": "x"}))], true).remove(0).unwrap();
    svc.refresh().unwrap();
    assert_eq!(rv_total(&[Target::new(svc.clone())], &parse(json!({"track_total_hits": true, "query": {"term": {"extra": "x"}}}), &[])), 1);
}

/// File lifetime: a point in time keeps reading every document it saw
/// (sealed segments and a published memtable) across updates, deletes,
/// refreshes, force merges and flushes (commits); the merged-away segment
/// files stay until the point in time is released, then the next commit
/// deletes them.
#[test]
fn pit_reads_every_doc_across_merges_and_commits_and_pins_its_files() {
    let (_t, node) = node();
    let svc = node.create_index("rv-pit", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}})).unwrap();
    const N: usize = 80;
    for chunk in 0..4 {
        let w = svc.write((0..20).map(|i| { let n = chunk * 20 + i; idx(&format!("d{n}"), json!({"n": n, "body": format!("doc {n}")})) }).collect(), true);
        assert!(w.iter().all(|r| r.is_ok()));
        svc.refresh().unwrap();
        // The last chunk stays a published memtable.
        if chunk < 3 {
            svc.flush().unwrap();
        }
    }
    let sh = svc.local_shards()[0].clone();
    let pit = node.snapshot(&[Target::new(svc.clone())]).unwrap();
    assert_eq!(pit.len(), 1);
    let view = pit[0].snapshot.clone().unwrap();
    assert!(!view.segments().is_empty() && view.num_memtables() > 0, "segments {} memtables {}", view.segments().len(), view.num_memtables());
    let pinned: Vec<u32> = view.segments().iter().map(|s| s.id).collect();
    drop(view);
    let dir = sh.shard.core.read().directory().clone();
    // Churn: update and delete docs of every segment and of the memtable,
    // then merge everything away and commit, twice.
    for round in 0..2 {
        let mut reqs: Vec<WriteReq> = (0..N).step_by(3).map(|n| idx(&format!("d{n}"), json!({"n": 1000 + n, "body": format!("new {n} {round}")}))).collect();
        reqs.extend((1..N).step_by(7).map(|n| WriteReq::Delete { id: format!("d{n}"), routing: None, cond: WriteCond::default() }));
        let _ = svc.write(reqs, true);
        svc.refresh().unwrap();
        svc.force_merge(1, false).unwrap();
        svc.flush().unwrap();
    }
    for id in &pinned {
        assert!(rustic_core::codec::directory::Directory::exists(&*dir, &rustic_core::codec::segment_file_name(*id)), "segment {id} deleted while a PIT holds it");
    }
    // Every document of the point in time reads as it was.
    let req = parse(json!({"size": 1000, "track_total_hits": true, "sort": [{"n": "asc"}]}), &[]);
    let out = crate::search::execute(&pit, &req, &rv_opts()).unwrap();
    let v: Value = serde_json::from_str(&out.json).unwrap();
    let hits = v["hits"]["hits"].as_array().unwrap();
    assert_eq!(hits.len(), N);
    for (n, hit) in hits.iter().enumerate() {
        assert_eq!(hit["_id"].as_str(), Some(format!("d{n}").as_str()));
        assert_eq!(hit["_source"]["n"].as_u64(), Some(n as u64));
    }
    assert_eq!(crate::search::count(&pit, None, None).unwrap(), N as u64);
    // The live index saw the churn.
    let deleted = (1..N).step_by(7).count() as u64;
    assert_eq!(rv_total(&[Target::new(svc.clone())], &req), N as u64 - deleted);
    // Released: the next commit deletes the merged-away files.
    drop(pit);
    svc.write(vec![idx("after", json!({"n": 5000}))], true).remove(0).unwrap();
    svc.flush().unwrap();
    for id in &pinned {
        assert!(!rustic_core::codec::directory::Directory::exists(&*dir, &rustic_core::codec::segment_file_name(*id)), "segment {id} kept after the PIT closed");
    }
}

/// One shard under randomized concurrent search, explicit-id and auto-id
/// bulk, refresh, flush, force merge, PIT open / search / close and realtime
/// get. Nothing may deadlock (watchdog), every op's seq_no increases per
/// writer, and every acknowledged document is searchable after a refresh.
#[test]
fn randomized_reader_view_stress_on_one_shard() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let (_t, node) = node();
    let svc = node.create_index("rv-mix", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "20ms"}})).unwrap();
    for s in svc.local_shards() {
        s.shard.bg.threshold.store(200, Ordering::Relaxed);
    }
    const ROUNDS: usize = 30;
    const IDS: usize = 400;
    fn lcg(x: &mut u64) -> u64 {
        *x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x >> 33
    }
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = std::sync::mpsc::channel();
    let (svc2, node2, stop2) = (svc.clone(), node.clone(), stop.clone());
    std::thread::spawn(move || {
        let svc = svc2;
        let result = std::thread::scope(|sc| {
            let svc = &svc;
            let node = &node2;
            let stop = &stop2;
            let writers: Vec<_> = (0..2).map(|t| sc.spawn(move || {
                let mut model = std::collections::BTreeMap::new();
                let mut last_seq = -1i64;
                let mut rng = 17 + t as u64;
                for r in 0..ROUNDS {
                    let reqs: Vec<WriteReq> = (0..50).map(|i| {
                        let id = format!("w{t}-{}", lcg(&mut rng) as usize % IDS);
                        model.insert(id.clone(), (r * 100 + i) as i64);
                        idx(&id, json!({"n": r * 100 + i, "w": t}))
                    }).collect();
                    for x in svc.write(reqs, r % 4 == 0) {
                        let x = x.expect("write failed");
                        assert!(x.seq_no > last_seq, "seq_no went back: {} after {last_seq}", x.seq_no);
                        last_seq = x.seq_no;
                    }
                    if r % 5 == 4 {
                        let dels: Vec<WriteReq> = (0..10).map(|_| {
                            let id = format!("w{t}-{}", lcg(&mut rng) as usize % IDS);
                            model.remove(&id);
                            WriteReq::Delete { id, routing: None, cond: WriteCond::default() }
                        }).collect();
                        // Deletes of absent ids may report no new seq_no.
                        let _ = svc.write(dels, false);
                    }
                }
                model
            })).collect();
            let auto = sc.spawn(move || {
                let mut ids = Vec::new();
                let mut last_seq = -1i64;
                for r in 0..ROUNDS {
                    let reqs: Vec<WriteReq> = (0..40).map(|i| WriteReq::Index { id: None, source: json!({"a": 1, "r": r, "i": i}).to_string(), routing: None, cond: WriteCond::default() }).collect();
                    for x in svc.write(reqs, r % 3 == 0) {
                        let x = x.expect("auto-id write failed");
                        assert!(x.seq_no > last_seq);
                        last_seq = x.seq_no;
                        ids.push(x.id);
                    }
                }
                ids
            });
            for s in 0..2u64 {
                sc.spawn(move || {
                    let mut rng = 101 + s;
                    let bodies = [
                        json!({"size": 10, "query": {"match_all": {}}}),
                        json!({"size": 0, "aggs": {"w": {"terms": {"field": "w"}}}}),
                        json!({"size": 5, "sort": [{"n": {"order": "desc", "unmapped_type": "long"}}], "query": {"range": {"n": {"gte": 100}}}}),
                    ];
                    while !stop.load(Ordering::Relaxed) {
                        let body = &bodies[lcg(&mut rng) as usize % bodies.len()];
                        crate::search::execute(&[Target::new(svc.clone())], &parse(body.clone(), &[]), &rv_opts()).expect("search failed");
                        crate::search::count(&[Target::new(svc.clone())], None, None).expect("count failed");
                    }
                });
            }
            sc.spawn(move || {
                let mut rng = 7u64;
                let mut held: Option<Vec<Target>> = None;
                while !stop.load(Ordering::Relaxed) {
                    match lcg(&mut rng) % 8 {
                        0 => svc.flush().unwrap(),
                        1 | 2 => svc.refresh().unwrap(),
                        3 => svc.force_merge(1, false).unwrap(),
                        4 => held = Some(node.snapshot(&[Target::new(svc.clone())]).unwrap()),
                        5 => {
                            if let Some(pit) = &held {
                                crate::search::execute(pit, &parse(json!({"size": 20, "sort": [{"n": {"order": "asc", "unmapped_type": "long"}}]}), &[]), &rv_opts()).expect("PIT search failed");
                            }
                        }
                        6 => held = None,
                        _ => {
                            let _ = svc.get(&format!("w0-{}", lcg(&mut rng) as usize % IDS)).unwrap();
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
            });
            let models: Vec<_> = writers.into_iter().map(|h| h.join().unwrap()).collect();
            let auto = auto.join().unwrap();
            stop.store(true, Ordering::Relaxed);
            (models, auto)
        });
        let _ = tx.send(result);
    });
    let (models, auto) = rx.recv_timeout(std::time::Duration::from_secs(300)).expect("reader-view stress deadlocked");
    stop.store(true, Ordering::Relaxed);
    svc.refresh().unwrap();
    let expected: usize = models.iter().map(|m| m.len()).sum::<usize>() + auto.len();
    let req = parse(json!({"size": 10_000, "track_total_hits": true, "_source": false}), &[]);
    let out = crate::search::execute(&[Target::new(svc.clone())], &req, &rv_opts()).unwrap();
    let v: Value = serde_json::from_str(&out.json).unwrap();
    let hits: Vec<String> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
    let mut seen: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for id in &hits {
        *seen.entry(id.as_str()).or_default() += 1;
    }
    let want: std::collections::HashSet<&str> = models.iter().flat_map(|m| m.keys()).chain(auto.iter()).map(String::as_str).collect();
    let duplicated: Vec<(&str, usize)> = seen.iter().filter(|(_, n)| **n > 1).map(|(id, n)| (*id, *n)).collect();
    let extra: Vec<&str> = seen.keys().filter(|id| !want.contains(*id)).copied().collect();
    let missing: Vec<&str> = want.iter().filter(|id| !seen.contains_key(*id)).copied().collect();
    assert!(duplicated.is_empty() && extra.is_empty() && missing.is_empty() && v["hits"]["total"]["value"].as_u64() == Some(expected as u64),
        "acked docs searchable after refresh: total {} expected {expected}; duplicated {duplicated:?}; extra {extra:?}; missing {missing:?}",
        v["hits"]["total"]["value"]);
    assert_eq!(svc.doc_count_realtime().0 as usize, expected);
}

/// A refresh while a search is still running on the previous view: the old
/// view stays exact (no new docs, deletes made visible afterwards do not
/// reach it, published memtables included); only views taken after the
/// refresh see the changes. The refresh holds the core write lock only for
/// the short publication (its hold is recorded).
#[test]
fn refresh_under_a_held_old_view_keeps_it_exact() {
    let (_t, node) = node();
    let svc = node.create_index("rv-refresh", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}})).unwrap();
    assert!(svc.write((0..20).map(|i| idx(&format!("d{i}"), json!({"n": i}))).collect(), true).iter().all(|r| r.is_ok()));
    svc.refresh().unwrap();
    let sh = svc.local_shards()[0].clone();
    // The "in-flight search": a target pinned to the current view.
    let old = sh.shard.reader_view();
    let old_target = Target { snapshot: Some(old.clone()), shard: Some(sh.clone()), ..Target::new(svc.clone()) };
    let all = parse(json!({"size": 100, "track_total_hits": true}), &[]);
    assert_eq!(rv_total(std::slice::from_ref(&old_target), &all), 20);
    let (holds_before, _, _) = crate::shard::refresh_lock_hold();
    // New docs, an update and a delete of docs in the published memtable.
    let mut reqs: Vec<WriteReq> = (20..30).map(|i| idx(&format!("d{i}"), json!({"n": i}))).collect();
    reqs.push(idx("d0", json!({"n": 1000})));
    reqs.push(WriteReq::Delete { id: "d1".into(), routing: None, cond: WriteCond::default() });
    assert!(svc.write(reqs, true).iter().all(|r| r.is_ok()));
    svc.refresh().unwrap();
    let (holds_after, _, max_hold) = crate::shard::refresh_lock_hold();
    assert!(holds_after > holds_before, "refresh publication hold not recorded");
    assert!(max_hold > std::time::Duration::ZERO);
    // Old view: exactly as before.
    assert_eq!(old.num_live_docs(), 20);
    let out = crate::search::execute(std::slice::from_ref(&old_target), &parse(json!({"size": 100, "sort": [{"n": "asc"}]}), &[]), &rv_opts()).unwrap();
    let v: Value = serde_json::from_str(&out.json).unwrap();
    let ns: Vec<u64> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_source"]["n"].as_u64().unwrap()).collect();
    assert_eq!(ns, (0..20).collect::<Vec<u64>>(), "old view changed under a refresh");
    // New views see the refresh.
    assert_eq!(rv_total(&[Target::new(svc.clone())], &all), 29);
    assert!(sh.shard.reader_view().generation() > old.generation());
}

/// Scheduled refreshes of other shards keep running while one shard's core
/// is write-locked (the scheduler only try-locks; the busy shard is handed
/// to a background worker and refreshed once its core is free).
#[test]
fn scheduled_refresh_is_not_stalled_by_one_locked_shard() {
    let (_t, node) = node();
    let settings = json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "50ms"}});
    let busy = node.create_index("rv-busy", &settings).unwrap();
    let other = node.create_index("rv-other", &settings).unwrap();
    // A write on the busy shard, still unpublished when its core is locked.
    busy.write(vec![idx("b", json!({"n": 1}))], true).remove(0).unwrap();
    let shard = busy.local_shards()[0].clone();
    let (held_tx, held_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let holder = std::thread::spawn(move || {
        let guard = shard.shard.core.write();
        held_tx.send(()).unwrap();
        let _ = release_rx.recv_timeout(std::time::Duration::from_secs(120));
        drop(guard);
    });
    held_rx.recv().unwrap();
    // Let the scheduler meet the locked shard at least once.
    std::thread::sleep(std::time::Duration::from_millis(200));
    other.write(vec![idx("o", json!({"n": 2}))], true).remove(0).unwrap();
    let all = parse(json!({"track_total_hits": true}), &[]);
    let visible = |svc: &Arc<IndexService>, want: u64| {
        let t0 = std::time::Instant::now();
        while t0.elapsed() < std::time::Duration::from_secs(20) {
            if rv_total(&[Target::new(svc.clone())], &all) == want {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        false
    };
    let other_ok = visible(&other, 1);
    release_tx.send(()).unwrap();
    holder.join().unwrap();
    assert!(other_ok, "a write-locked shard stalled scheduled refresh of another shard");
    assert!(visible(&busy, 1), "the busy shard was not refreshed once its core was free");
}

/// Mutable-graph delete repair runs in bounded background batches, outside
/// the refresh critical section; a refresh still makes deletes invisible to
/// kNN at once (the reader view masks them), before any physical repair.
#[cfg(feature = "mutable-hnsw")]
#[test]
fn knn_deletes_are_invisible_at_refresh_before_graph_repair() {
    let (_t, node) = node();
    let svc = node.create_index("rv-knn-del", &json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
        "mappings": {"properties": {"vector": {"type": "dense_vector", "dims": 2, "similarity": "l2_norm", "index_options": {"type": "hnsw"}}}}
    })).unwrap();
    assert!(svc.write((0..30).map(|i| idx(&format!("d{i}"), json!({"vector": [i as f32, 1.0]}))).collect(), true).iter().all(|r| r.is_ok()));
    svc.refresh().unwrap();
    svc.flush().unwrap();
    let knn = parse(json!({"size": 30, "knn": {"field": "vector", "query_vector": [0.0, 1.0], "k": 30, "num_candidates": 60}}), &[]);
    let ids_of = |v: &Value| -> Vec<String> { v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect() };
    assert_eq!(ids_of(&exec(&svc, &knn)).len(), 30);
    let gone: Vec<String> = (0..30).step_by(3).map(|i| format!("d{i}")).collect();
    let dels = gone.iter().map(|id| WriteReq::Delete { id: id.clone(), routing: None, cond: WriteCond::default() }).collect();
    assert!(svc.write(dels, true).iter().all(|r| r.is_ok()));
    let sh = svc.local_shards()[0].clone();
    // The refresh critical section only: repair is left to the workers.
    assert!(sh.shard.publish().unwrap());
    let hits = ids_of(&exec(&svc, &knn));
    assert_eq!(hits.len(), 20, "{hits:?}");
    assert!(hits.iter().all(|id| !gone.contains(id)), "a deleted doc was returned before repair: {hits:?}");
    // After the repair results are unchanged, whether or not a background
    // merge expunged the deletes meanwhile (merged segments keep their
    // points). The repair runs here to completion: `repair_vectors` is the
    // background pass (it skips a busy core and re-arms instead of waiting,
    // and a worker may already hold the batch), so calling it once left it
    // to timing whether the search below ran before or after the repair.
    let store = sh.shard.core.read().vector_store().clone();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while store.drain_mutable_deletes_batch(usize::MAX).unwrap() {
        // Only a live older view (a background merge's) defers a delete.
        assert!(std::time::Instant::now() < deadline, "graph repair never became eligible");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(store.mutable_pressure().pending_repairs, 0);
    let describe = || {
        let view = sh.shard.reader_view();
        let segments: Vec<_> = view.segments().iter().map(|s| (s.id, s.num_docs, s.num_live_docs())).collect();
        let pressure = sh.shard.core.read().vector_store().mutable_pressure();
        format!("segments (id, docs, live) {segments:?}, memtables {}, pending repairs {}, retired slots {}",
            view.num_memtables(), pressure.pending_repairs, pressure.retired_slots)
    };
    let after = ids_of(&exec(&svc, &knn));
    assert_eq!(after, hits, "{}", describe());
}

/// Flushes that let writes continue (stage 3) keep every acknowledged doc
/// across a reopen: writes race flushes, then the node is dropped without a
/// final flush and reopened (commit + translog replay above it).
#[test]
fn acked_docs_survive_reopen_after_flushes_concurrent_with_writes() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let svc = node.create_index("rv-crash", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}})).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let flusher = {
        let (svc, stop) = (svc.clone(), stop.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                svc.flush().unwrap();
                std::thread::sleep(std::time::Duration::from_millis(3));
            }
        })
    };
    let mut live = std::collections::BTreeSet::new();
    for round in 0..60 {
        let ids: Vec<String> = (0..10).map(|i| format!("d{}", (round * 7 + i) % 300)).collect();
        let reqs: Vec<WriteReq> = ids.iter().map(|id| idx(id, json!({"round": round}))).collect();
        for r in svc.write(reqs, round % 2 == 0) {
            r.expect("write failed");
        }
        live.extend(ids);
        if round % 9 == 8 {
            let id = format!("d{}", round % 300);
            svc.write(vec![WriteReq::Delete { id: id.clone(), routing: None, cond: WriteCond::default() }], true).remove(0).expect("delete failed");
            live.remove(&id);
        }
    }
    svc.sync_translog().unwrap();
    stop.store(true, Ordering::Relaxed);
    flusher.join().unwrap();
    drop(svc);
    drop(node);
    let node = Node::open(tmp.path()).unwrap();
    let svc = node.get_index("rv-crash").unwrap();
    svc.refresh().unwrap();
    let v = exec(&svc, &parse(json!({"size": 1000, "track_total_hits": true, "_source": false}), &[]));
    let seen: std::collections::BTreeSet<String> = ids(&v).into_iter().collect();
    assert_eq!(seen, live, "acked docs after reopen");
}

#[cfg(feature = "mutable-hnsw")]
fn knn_index(node: &Arc<Node>, name: &str) -> Arc<IndexService> {
    node.create_index(name, &json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
        "mappings": {"properties": {"vector": {"type": "dense_vector", "dims": 2, "similarity": "l2_norm", "index_options": {"type": "hnsw"}}}}
    })).unwrap()
}

/// `dense_vector` off-heap stats report the shared mutable graph of a
/// default HNSW field (no `flat_index_threshold: 0`, whose payload
/// segments never carry a graph): `vex` used to be 0 for it.
#[cfg(feature = "mutable-hnsw")]
#[test]
fn dense_vector_off_heap_reports_the_shared_mutable_graph() {
    let (_t, node) = node();
    let svc = knn_index(&node, "vs-offheap");
    assert!(svc.write((0..20).map(|i| idx(&format!("d{i}"), json!({"vector": [i as f32, 1.0]}))).collect(), true).iter().all(|r| r.is_ok()));
    svc.refresh().unwrap();
    svc.flush().unwrap();
    let v = crate::vector_stats::index_stats(&svc, true);
    assert_eq!(v["value_count"], 20, "{v}");
    let field = &v["off_heap"]["fielddata"]["vector"];
    assert!(field["vex_size_bytes"].as_u64().unwrap() > 0, "{v}");
    assert_eq!(v["off_heap"]["total_vex_size_bytes"], field["vex_size_bytes"], "{v}");
    assert_eq!(field["vec_size_bytes"], 20 * 2 * 4, "{v}");
}

#[cfg(feature = "mutable-hnsw")]
fn knn_ids(targets: &[Target], k: usize) -> Vec<String> {
    let req = parse(json!({"size": k, "knn": {"field": "vector", "query_vector": [0.0, 1.0], "k": k, "num_candidates": 2 * k}}), &[]);
    let out = crate::search::execute(targets, &req, &rv_opts()).unwrap();
    let v: Value = serde_json::from_str(&out.json).unwrap();
    ids(&v)
}

/// Deferred physical removal: a view taken before deletes keeps finding
/// every document it counts live by kNN, even after the deletes were made
/// visible and repair ran, while new views never return a deleted doc.
/// Once the old view is gone, repair removes the nodes and results stay
/// exact.
#[cfg(feature = "mutable-hnsw")]
#[test]
fn older_view_keeps_live_knn_hits_across_deletes_and_repair() {
    let (_t, node) = node();
    let svc = knn_index(&node, "rv-knn-old");
    assert!(svc.write((0..30).map(|i| idx(&format!("d{i}"), json!({"vector": [i as f32, 1.0]}))).collect(), true).iter().all(|r| r.is_ok()));
    svc.refresh().unwrap();
    let sh = svc.local_shards()[0].clone();
    let old = sh.shard.reader_view();
    let old_target = [Target { snapshot: Some(old.clone()), shard: Some(sh.clone()), ..Target::new(svc.clone()) }];
    assert_eq!(knn_ids(&old_target, 30).len(), 30);
    let gone: Vec<String> = (0..30).step_by(3).map(|i| format!("d{i}")).collect();
    let dels = gone.iter().map(|id| WriteReq::Delete { id: id.clone(), routing: None, cond: WriteCond::default() }).collect();
    assert!(svc.write(dels, true).iter().all(|r| r.is_ok()));
    svc.refresh().unwrap();
    sh.shard.repair_vectors();
    // The old view still counts all 30 live and finds them all.
    assert_eq!(knn_ids(&old_target, 30).len(), 30, "an older view lost live kNN hits to graph repair");
    // New views never return a deleted doc.
    let fresh = knn_ids(&[Target::new(svc.clone())], 30);
    assert_eq!(fresh.len(), 20);
    assert!(fresh.iter().all(|id| !gone.contains(id)));
    drop(old_target);
    drop(old);
    // Repair to completion (the background pass skips a busy core).
    let store = sh.shard.core.read().vector_store().clone();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while store.drain_mutable_deletes_batch(usize::MAX).unwrap() {
        assert!(std::time::Instant::now() < deadline, "graph repair never became eligible");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let field = sh.shard.core.read().field_id("vector").unwrap();
    let after = knn_ids(&[Target::new(svc.clone())], 30);
    let pressure = store.mutable_pressure();
    assert_eq!(after, fresh, "live kNN hits lost after the deleted nodes were removed: graph nodes {:?}, pending repairs {}, \
        retired slots {}, payload segments {:?}", store.mutable_active_nodes(field), pressure.pending_repairs, pressure.retired_slots,
        store.segments(field).iter().map(|s| (s.id, s.len())).collect::<Vec<_>>());
}

/// The vector auto-id bulk path used to hold `batches.read()` while its
/// preparation ran on the shared rayon pool: a worker could steal another
/// vector bulk of the same shard and re-take `batches.read()` behind a
/// queued flush (fair lock), deadlocking itself. Many concurrent auto-id
/// vector bulks issued from global-pool workers, with flushes and kNN
/// searches running throughout, must all complete (watchdog).
#[cfg(feature = "mutable-hnsw")]
#[test]
fn vector_append_only_bulks_from_pool_workers_with_concurrent_flush_complete() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let (_t, node) = node();
    let svc = knn_index(&node, "rv-knn-reentry");
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = std::sync::mpsc::channel();
    let (svc2, stop2) = (svc.clone(), stop.clone());
    std::thread::spawn(move || {
        let svc = svc2;
        let flusher = {
            let (svc, stop) = (svc.clone(), stop2.clone());
            std::thread::spawn(move || while !stop.load(Ordering::Relaxed) {
                svc.flush().unwrap();
                svc.refresh().unwrap();
                let _ = knn_ids(&[Target::new(svc.clone())], 5);
            })
        };
        let acked = std::sync::atomic::AtomicUsize::new(0);
        rayon::scope(|scope| {
            for t in 0..16 {
                let (svc, acked) = (&svc, &acked);
                scope.spawn(move |_| {
                    for r in 0..6 {
                        let reqs: Vec<WriteReq> = (0..64).map(|i| WriteReq::Index { id: None,
                            source: json!({"vector": [(t * 1000 + r * 64 + i) as f32, 1.0]}).to_string(), routing: None, cond: WriteCond::default() }).collect();
                        for x in svc.write(reqs, false) {
                            x.expect("vector bulk failed");
                            acked.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                });
            }
        });
        stop2.store(true, Ordering::Relaxed);
        flusher.join().unwrap();
        let _ = tx.send(acked.load(Ordering::Relaxed));
    });
    let acked = rx.recv_timeout(std::time::Duration::from_secs(300)).expect("vector bulks deadlocked with concurrent flush");
    stop.store(true, Ordering::Relaxed);
    assert_eq!(acked, 16 * 6 * 64);
    svc.refresh().unwrap();
    assert_eq!(svc.doc_count_realtime().0 as usize, acked);
}

/// kNN searches keep running while a force merge encodes (they read their
/// view; the merge holds no core lock across its work).
#[cfg(feature = "mutable-hnsw")]
#[test]
fn knn_search_runs_during_force_merge() {
    let (_t, node) = node();
    let svc = knn_index(&node, "rv-knn-fm");
    for round in 0..6 {
        assert!(svc.write((0..40).map(|i| idx(&format!("d{round}-{i}"), json!({"vector": [(round * 40 + i) as f32, 1.0]}))).collect(), true).iter().all(|r| r.is_ok()));
        svc.refresh().unwrap();
        svc.flush().unwrap();
    }
    let merging = svc.clone();
    let merge = std::thread::spawn(move || merging.force_merge(1, false));
    let (tx, rx) = std::sync::mpsc::channel();
    let searcher = svc.clone();
    std::thread::spawn(move || {
        let mut n = 0;
        for _ in 0..50 {
            n += knn_ids(&[Target::new(searcher.clone())], 10).len();
        }
        let _ = tx.send(n);
    });
    let n = rx.recv_timeout(std::time::Duration::from_secs(120)).expect("kNN searches blocked during force merge");
    assert_eq!(n, 500);
    merge.join().unwrap().unwrap();
    assert_eq!(knn_ids(&[Target::new(svc.clone())], 10).len(), 10);
}

/// Index-sorted shard under concurrent bulk, search, refresh, flush and force
/// merge: sorted refreshes seal detached (writes continue, searches read their
/// view), so nothing deadlocks (watchdog), no search waits long (latency
/// bound), and every acked doc is searchable, in sort order, after a refresh.
#[test]
fn sorted_index_concurrent_bulk_search_refresh_flush_and_force_merge() {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    let (_t, node) = node();
    let svc = node.create_index("rv-sorted", &json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "30ms", "index.sort.field": "n", "index.sort.order": "asc"},
        "mappings": {"properties": {"n": {"type": "long"}}}
    })).unwrap();
    const WRITERS: usize = 3;
    const ROUNDS: usize = 20;
    let stop = Arc::new(AtomicBool::new(false));
    let max_search_ms = Arc::new(AtomicU64::new(0));
    let (tx, rx) = std::sync::mpsc::channel();
    let (svc2, stop2, max2) = (svc.clone(), stop.clone(), max_search_ms.clone());
    std::thread::spawn(move || {
        let svc = svc2;
        std::thread::scope(|sc| {
            let (svc, stop, max_ms) = (&svc, &stop2, &max2);
            let writers: Vec<_> = (0..WRITERS).map(|t| sc.spawn(move || {
                for r in 0..ROUNDS {
                    let reqs: Vec<WriteReq> = (0..50).map(|i| {
                        let n = (t * ROUNDS + r) * 50 + i;
                        idx(&format!("d{n}"), json!({"n": (n * 7919) % 100_003}))
                    }).collect();
                    for x in svc.write(reqs, r % 3 == 0) { x.expect("sorted write failed"); }
                }
            })).collect();
            sc.spawn(move || {
                let req = parse(json!({"size": 10, "sort": [{"n": "asc"}], "track_total_hits": true}), &[]);
                while !stop.load(Ordering::Relaxed) {
                    let t0 = std::time::Instant::now();
                    crate::search::execute(&[Target::new(svc.clone())], &req, &rv_opts()).expect("sorted search failed");
                    max_ms.fetch_max(t0.elapsed().as_millis() as u64, Ordering::Relaxed);
                }
            });
            sc.spawn(move || {
                let mut k = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    k += 1;
                    match k % 4 {
                        0 => svc.flush().unwrap(),
                        1 | 2 => svc.refresh().unwrap(),
                        _ => svc.force_merge(1, false).unwrap(),
                    }
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            });
            for w in writers { w.join().unwrap(); }
            stop.store(true, Ordering::Relaxed);
        });
        let _ = tx.send(());
    });
    rx.recv_timeout(std::time::Duration::from_secs(300)).expect("sorted index load deadlocked");
    stop.store(true, Ordering::Relaxed);
    assert!(max_search_ms.load(Ordering::Relaxed) < 10_000, "a search waited {} ms", max_search_ms.load(Ordering::Relaxed));
    svc.refresh().unwrap();
    let expected = WRITERS * ROUNDS * 50;
    let v = exec(&svc, &parse(json!({"size": 10_000, "track_total_hits": true, "sort": [{"n": "asc"}]}), &[]));
    assert_eq!(v["hits"]["total"]["value"].as_u64(), Some(expected as u64));
    let ns: Vec<i64> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_source"]["n"].as_i64().unwrap()).collect();
    assert!(ns.windows(2).all(|w| w[0] <= w[1]), "not in sort order");
    assert_eq!(svc.doc_count_realtime().0 as usize, expected);
}

/// R2: a file copy (commit + translog) captured while bulks are in flight
/// holds every op acknowledged before the capture: documents still in a
/// pooled memtable at capture time come back through the translog above
/// the commit's local checkpoint.
#[test]
fn file_copy_taken_while_bulks_are_in_flight_keeps_every_acked_op() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let (_t, node) = node();
    let src = node.create_index("inflight", &json!({"settings": {"number_of_shards": 1}})).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let acked = Arc::new(parking_lot::Mutex::new(Vec::<String>::new()));
    let writers: Vec<_> = (0..3).map(|thread| {
        let (src, stop, acked) = (src.clone(), stop.clone(), acked.clone());
        std::thread::spawn(move || {
            let mut round = 0;
            while !stop.load(Ordering::Acquire) {
                let ids: Vec<String> = (0..300).map(|i| format!("f{thread}-{round}-{i}")).collect();
                let results = src.write(ids.iter().map(|id| idx(id, json!({"t": "text to analyze for a while"}))).collect(), false);
                let mut acked = acked.lock();
                for (id, r) in ids.into_iter().zip(results) { if r.is_ok() { acked.push(id); } }
                round += 1;
            }
        })
    }).collect();
    std::thread::sleep(std::time::Duration::from_millis(60));
    src.shard(0).unwrap().shard.flush().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(40));
    let before = acked.lock().clone();
    let set = src.shard(0).unwrap().snapshot_files(true).unwrap();
    stop.store(true, Ordering::Release);
    for w in writers { w.join().unwrap(); }
    assert!(!before.is_empty());
    let meta = json!({"settings": settings::as_flat_json(&src.meta().settings), "mappings": src.meta().mappings.to_json()});
    let copy = node.create_index_sparse("inflight-copy", "inflightCopy1", &meta).unwrap();
    let staged = node.new_staging_dir().unwrap();
    set.write_to_dir(&staged, false).unwrap();
    let r = copy.open_shard(0, spec(false, 1, RecoverySource::FromFiles { staged, manifest: set.manifest.clone() })).unwrap();
    for id in &before {
        assert!(r.shard.get(id).unwrap().is_some(), "acked {id} missing from the copy");
    }
}

/// D1/D2 through a file copy: a batch held between sequencing and
/// inversion keeps the commit point below completed auto-id writes and a
/// completed delete; a commit + translog copy then replays everything above
/// the commit point: each auto-id document once, the delete wins.
#[test]
fn file_copy_replay_above_the_commit_point_keeps_docs_once_and_deletes_win() {
    let (_t, node) = node();
    let src = node.create_index("floorcopy", &json!({"settings": {"number_of_shards": 1}})).unwrap();
    src.write(vec![idx("x", json!({"t": "first"}))], false).remove(0).unwrap();
    let shard = src.shard(0).unwrap();
    let (sequenced_tx, sequenced) = std::sync::mpsc::channel();
    let (release, release_rx) = std::sync::mpsc::channel::<()>();
    shard.shard.pause_next_staged_write(move || {
        let _ = sequenced_tx.send(());
        let _ = release_rx.recv_timeout(std::time::Duration::from_secs(30));
    });
    let held = {
        let src = src.clone();
        std::thread::spawn(move || src.write(vec![idx("x", json!({"t": "held"})), idx("held1", json!({"t": "h"}))], false))
    };
    sequenced.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
    let mut auto_ids = Vec::new();
    for _ in 0..10 {
        let reqs = (0..3).map(|_| WriteReq::Index { id: None, source: json!({"t": "auto"}).to_string(), routing: None, cond: WriteCond::default() }).collect();
        for r in src.write(reqs, false) { auto_ids.push(r.unwrap().id); }
    }
    src.write(vec![WriteReq::Delete { id: "x".into(), routing: None, cond: WriteCond::default() }], false).remove(0).unwrap();
    shard.shard.flush().unwrap();
    release.send(()).unwrap();
    for r in held.join().unwrap() { r.unwrap(); }
    let set = shard.snapshot_files(true).unwrap();
    let meta = json!({"settings": settings::as_flat_json(&src.meta().settings), "mappings": src.meta().mappings.to_json()});
    let copy = node.create_index_sparse("floorcopy-copy", "floorCopy1", &meta).unwrap();
    let staged = node.new_staging_dir().unwrap();
    set.write_to_dir(&staged, false).unwrap();
    let r = copy.open_shard(0, spec(false, 1, RecoverySource::FromFiles { staged, manifest: set.manifest.clone() })).unwrap();
    r.shard.refresh().unwrap();
    let live_copies = |id: &str| -> usize {
        let core = r.shard.core.read();
        let term = rustic_core::types::Term::new_text(crate::mapping::ID_FIELD, id);
        core.all_leaves().iter().map(|leaf| {
            let mut n = 0;
            if let Some(mut postings) = leaf.postings(&term) {
                while let Some(d) = postings.next() { if !leaf.is_deleted_latest(d) { n += 1; } }
            }
            n
        }).sum()
    };
    for id in &auto_ids {
        assert_eq!(live_copies(id), 1, "auto-id {id} duplicated or lost in the copy");
    }
    assert_eq!(live_copies("x"), 0);
    assert!(r.shard.get("x").unwrap().is_none(), "the delete lost to an older replayed index op");
    assert!(r.shard.get("held1").unwrap().is_some());
}

/// B3: batches append to the translog out of seq_no order, so a crash can
/// leave a later batch durable (seq_nos 4-5) and an earlier one never
/// written (3). Local recovery of a primary fills the hole with a no-op:
/// the checkpoint reaches the max seq_no and keeps following it.
#[test]
fn a_seq_no_hole_left_by_a_crash_is_filled_at_local_recovery() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let svc = node.create_index("gap", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}})).unwrap();
    for r in svc.write((0..3).map(|i| idx(&format!("a{i}"), json!({"n": i}))).collect(), true) { r.unwrap(); }
    let dir = svc.shard(0).unwrap().shard.path().to_path_buf();
    drop(svc);
    drop(node);
    {
        let mut log = crate::translog::Translog::open(&dir.join("translog"), 1).unwrap();
        for seq_no in [4u64, 5] {
            log.append(&crate::translog::TranslogOp::Index { id: format!("b{seq_no}"), source_json: "{\"n\":1}".into(), seq_no, version: 1, routing: None }).unwrap();
        }
        log.fsync().unwrap();
    }
    let node = Node::open(tmp.path()).unwrap();
    let svc = node.get_index("gap").unwrap();
    let copy = svc.shard(0).unwrap();
    assert_eq!(copy.shard.max_seq_no(), 5);
    assert_eq!(copy.shard.local_checkpoint(), 5, "the hole at 3 is filled");
    assert!(copy.shard.peek("b5").unwrap().is_some());
    // The checkpoint keeps following new writes and flushes, across a restart.
    for round in 0..3 {
        svc.write(vec![idx(&format!("c{round}"), json!({"n": round}))], true).remove(0).unwrap();
        svc.flush().unwrap();
        assert_eq!(copy.shard.local_checkpoint(), copy.shard.max_seq_no());
    }
    drop((copy, svc));
    drop(node);
    let node = Node::open(tmp.path()).unwrap();
    let copy = node.get_index("gap").unwrap().shard(0).unwrap();
    assert_eq!(copy.shard.local_checkpoint(), copy.shard.max_seq_no());
    assert_eq!(copy.shard.max_seq_no(), 8);
}

/// B7: an index-sorted refresh while a flush encodes its seal completes
/// (it waits for the flush) instead of failing on the flush's seal in
/// flight; afterwards every document, updated ones included, is visible
/// exactly once.
#[test]
fn a_sorted_refresh_during_a_flush_completes_and_keeps_one_copy() {
    let (_t, node) = node();
    let svc = node.create_index("sorted-flush", &json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1", "index.sort.field": "n", "index.sort.order": "asc"},
        "mappings": {"properties": {"n": {"type": "long"}}}
    })).unwrap();
    for r in svc.write((0..10).map(|i| idx(&format!("a{i}"), json!({"n": i}))).collect(), false) { r.unwrap(); }
    svc.refresh().unwrap();
    // Updates of half of them, and new documents, before the flush.
    for r in svc.write((0..5).map(|i| idx(&format!("a{i}"), json!({"n": 100 + i}))).collect(), false) { r.unwrap(); }
    for r in svc.write((0..10).map(|i| idx(&format!("b{i}"), json!({"n": 50 + i}))).collect(), false) { r.unwrap(); }
    let copy = svc.shard(0).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let during = svc.clone();
    copy.shard.pause_next_flush_commit(move || {
        let refreshing = during.clone();
        tx.send(std::thread::spawn(move || refreshing.refresh())).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
    });
    svc.flush().unwrap();
    let refreshed = rx.recv().unwrap().join().unwrap();
    assert!(refreshed.is_ok(), "{refreshed:?}");
    svc.refresh().unwrap();
    let v = exec(&svc, &parse(json!({"size": 100, "track_total_hits": true, "_source": false}), &[]));
    let mut seen = ids(&v);
    seen.sort();
    let mut expected: Vec<String> = (0..10).map(|i| format!("a{i}")).chain((0..10).map(|i| format!("b{i}"))).collect();
    expected.sort();
    assert_eq!(seen, expected, "every document exactly once");
}

/// M10: a standalone copy failed by a tragic event (its translog closed)
/// is recovered from its store by the node's maintenance, as ES
/// reallocates and store-recovers a failed shard: acked writes survive and
/// writes are accepted again.
#[test]
fn a_standalone_copy_failed_by_a_tragic_event_recovers_from_its_store() {
    let (_t, node) = node();
    let svc = node.create_index("tragic", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}})).unwrap();
    svc.write(vec![idx("a", json!({"n": 1}))], true).remove(0).unwrap();
    svc.shard(0).unwrap().shard.fail_next_translog_appends(1);
    assert!(svc.write(vec![idx("b", json!({"n": 2}))], true).remove(0).is_err());
    assert!(svc.shard(0).unwrap().shard.tragic_error().is_some());
    assert!(svc.write(vec![idx("c", json!({"n": 3}))], true).remove(0).is_err(), "a failed engine refuses writes");
    node.maintenance();
    let copy = svc.shard(0).expect("the copy is open again");
    assert!(copy.shard.tragic_error().is_none());
    assert!(copy.shard.peek("a").unwrap().is_some(), "the acked write survived");
    svc.write(vec![idx("d", json!({"n": 4}))], true).remove(0).unwrap();
    assert!(svc.failed_shards().is_empty());
}

/// M10: a copy whose tragedy recurs after each recovery is retried a
/// bounded number of times (ES `index.allocation.max_retries`), then left
/// failed with the reason; a translog tragedy is reported as one, any other
/// as an engine failure.
#[test]
fn a_copy_failing_on_every_recovery_is_retried_a_bounded_number_of_times() {
    let (_t, node) = node();
    let svc = node.create_index("recurring", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}})).unwrap();
    svc.write(vec![idx("a", json!({"n": 1}))], true).remove(0).unwrap();
    let fail = |svc: &Arc<IndexService>| {
        let copy = svc.shard(0).unwrap();
        copy.shard.fail_next_translog_appends(1);
        assert!(svc.write(vec![idx("b", json!({"n": 2}))], true).remove(0).is_err());
        assert_eq!(copy.shard.tragic_failure().unwrap().kind, "translog_exception");
        copy
    };
    for attempt in 0..5 {
        let failed = fail(&svc);
        node.maintenance();
        let reopened = svc.shard(0).expect("recovered");
        assert!(!Arc::ptr_eq(&failed, &reopened), "attempt {attempt} did not recover the copy");
    }
    let failed = fail(&svc);
    node.maintenance();
    let still = svc.shard(0).unwrap();
    assert!(Arc::ptr_eq(&failed, &still), "recovered past the retry limit");
    let reason = svc.failed_shards().remove(&0).expect("left failed with a reason");
    assert!(reason.to_string().contains("not retrying"), "{reason}");
    // Sticky: however long ago the last attempt was, no further recovery.
    if let Some(a) = crate::index::recovery_attempts().lock().get_mut(&(svc.uuid.clone(), 0)) {
        a.next = std::time::Instant::now() - std::time::Duration::from_secs(3600);
    }
    node.maintenance();
    assert!(Arc::ptr_eq(&failed, &svc.shard(0).unwrap()), "recovered again after the limit");
    // Deleting the index forgets its attempts.
    let uuid = svc.uuid.clone();
    node.delete_index("recurring").unwrap();
    assert!(!crate::index::recovery_attempts().lock().keys().any(|(u, _)| *u == uuid));
}

/// M10: the attempt count restarts only after the copy stayed healthy for
/// a while after a successful recovery, never merely because time passed
/// since the last attempt.
#[test]
fn recovery_attempts_restart_only_after_a_healthy_period() {
    use crate::index::RecoveryAttempts;
    let now = std::time::Instant::now();
    let long_ago = now - std::time::Duration::from_secs(3600);
    let mut a = RecoveryAttempts { count: 5, next: long_ago, healthy_since: None };
    assert!(!crate::index::restart_count(&mut a, now), "restarted with no healthy period");
    assert_eq!(a.count, 5);
    a.healthy_since = Some(long_ago);
    assert!(crate::index::restart_count(&mut a, now));
    assert_eq!(a.count, 0);
}

/// M10: the failed copy's detached background jobs never change the files
/// its recovery opens: once fenced, a seal still encoding writes nothing
/// into the directory (it could otherwise rename a segment over one the
/// reopened copy numbers the same).
#[test]
fn a_fenced_copys_background_seal_writes_nothing_into_its_directory() {
    let (_t, node) = node();
    let svc = node.create_index("fenced", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}})).unwrap();
    for i in 0..20 {
        svc.write(vec![idx(&format!("d{i}"), json!({"n": i}))], false).remove(0).unwrap();
    }
    let copy = svc.shard(0).unwrap();
    let dir = svc.path().to_path_buf();
    // The scheduler leaves this copy's seals to the test.
    copy.shard.bg.queued.store(true, std::sync::atomic::Ordering::Release);
    let files = |dir: &std::path::Path| -> Vec<String> {
        let mut v: Vec<String> = walk(dir).into_iter().filter(|n| n.ends_with(".rstc")).collect();
        v.sort();
        v
    };
    let before = files(&dir);
    // A background seal detached before the fence, encoding after it.
    copy.shard.write_indexing_buffer();
    let task = copy.shard.start_seal().expect("a seal job");
    copy.fence();
    let done = task.run();
    drop(done);
    assert_eq!(files(&dir), before, "a fenced copy's seal wrote a segment file");
}

fn walk(dir: &std::path::Path) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() { out.extend(walk(&p)); } else { out.push(e.file_name().to_string_lossy().into_owned()); }
        }
    }
    out
}

/// Coalescing (c): a crash after a coalesced vector seal is installed and
/// before the next commit. The reopened copy replays the uncommitted
/// documents from the translog: each is live exactly once, no point is
/// alive without a live owner segment or twice, and kNN is exact.
#[test]
fn a_crash_after_a_coalesced_vector_seal_replays_each_document_once() {
    let tmp = tempfile::tempdir().unwrap();
    let mapping = json!({"properties": {"v": {"type": "dense_vector", "dims": 2, "similarity": "l2_norm", "index_options": {"type": "flat"}}}});
    {
        let node = Node::open(tmp.path()).unwrap();
        let svc = node.create_index("crash", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}, "mappings": mapping})).unwrap();
        svc.write(vec![idx("base", json!({"v": [100.0, 1.0]}))], true).remove(0).unwrap();
        svc.flush().unwrap();
        for round in 0..5 {
            let reqs = (0..6).map(|k| { let i = round * 6 + k; idx(&format!("d{i}"), json!({"v": [i as f64, 1.0]})) }).collect();
            assert!(svc.write(reqs, true).iter().all(|r| r.is_ok()));
            svc.refresh().unwrap();
        }
        let copy = svc.shard(0).unwrap();
        copy.shard.write_indexing_buffer();
        for _ in 0..100 {
            if !copy.shard.background_due() { break; }
            copy.shard.run_background();
        }
        assert_eq!(copy.shard.core.read().num_published_memtables(), 0, "the memtables were not sealed");
        // Dropped without a flush or close: a crash.
    }
    let mut reopened = None;
    for _ in 0..50 {
        if let Ok(node) = Node::open(tmp.path()) {
            if node.get_index("crash").and_then(|svc| svc.shard(0)).is_some() { reopened = Some(node); break; }
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let node = reopened.expect("the copy reopened");
    let svc = node.get_index("crash").unwrap();
    svc.refresh().unwrap();
    let (found, _) = search_ids(&svc);
    let mut unique = found.clone();
    unique.dedup();
    assert_eq!(found.len(), 31, "documents after the replay: {found:?}");
    assert_eq!(unique.len(), found.len(), "a document is live twice");
    for (q, want) in [(0.2, "d0"), (7.6, "d8"), (29.0, "d29")] {
        let v = exec(&svc, &parse(json!({"knn": {"field": "v", "query_vector": [q, 1.0], "k": 1, "num_candidates": 100}, "size": 1}), &[]));
        assert_eq!(ids(&v), vec![want.to_string()], "query {q}");
    }
    let copy = svc.shard(0).unwrap();
    // The replayed documents are in memtables (their rows have no store
    // points until sealed): seal them, then every live doc has one point.
    copy.shard.seal_all().unwrap();
    let core = copy.shard.core.read();
    let live: std::collections::HashSet<u32> = core.segments().iter().chain(core.hidden_segments()).map(|s| s.id).collect();
    let store = core.vector_store();
    let mut alive = 0;
    for p in 0..store.next_point_id() {
        if store.is_alive(p) {
            alive += 1;
            let owner = store.owner(p);
            assert!(owner.is_some_and(|(seg, _)| live.contains(&seg)), "point {p} alive with a dangling owner {owner:?}");
        }
    }
    assert_eq!(alive, 31, "alive points differ from the live documents");
}

/// B8: a replica apply whose translog append fails installs nothing (no
/// document, no version-map entry, so nothing is committed without a
/// translog record); a failed append or fsync fails the engine, which
/// refuses every later replica apply.
#[test]
fn replica_apply_is_durable_before_it_installs_and_fails_the_engine_on_translog_errors() {
    let op = |seq: i64, id: &str| ChangeOp::Index { id: id.into(), source_json: r#"{"n":1}"#.into(), routing: None, seq_no: seq, version: 1, primary_term: 3 };
    for fsync in [false, true] {
        let (_t, node) = node();
        let svc = node.create_index("ra", &json!({"settings": {"number_of_shards": 1}})).unwrap();
        svc.close_shard(0).unwrap();
        let r = svc.open_shard(0, spec(false, 3, RecoverySource::ExistingStore)).unwrap();
        r.apply_replica_ops(3, vec![op(0, "a")], Durability::Request).unwrap();
        if fsync { r.shard.fail_next_translog_syncs(1); } else { r.shard.fail_next_translog_appends(1); }
        assert!(r.apply_replica_ops(3, vec![op(1, "b")], Durability::Request).is_err(), "fsync {fsync}");
        if !fsync {
            assert!(r.shard.get_mode("b", true).unwrap().is_none(), "an op with no translog record was installed");
            assert_eq!(r.shard.local_checkpoint(), 0);
        }
        assert!(r.shard.tragic_error().is_some(), "fsync {fsync}: the engine was not failed");
        assert!(r.apply_replica_ops(3, vec![op(2, "c")], Durability::Request).is_err(), "fsync {fsync}: a failed engine applied");
        assert!(r.shard.get_mode("c", true).unwrap().is_none());
    }
}

/// A replica apply whose history side-log append fails (no-op and term
/// records) fails the engine like a translog failure.
#[test]
fn replica_side_log_failure_fails_the_engine() {
    let (_t, node) = node();
    let svc = node.create_index("rs", &json!({"settings": {"number_of_shards": 1}})).unwrap();
    svc.close_shard(0).unwrap();
    let r = svc.open_shard(0, spec(false, 3, RecoverySource::ExistingStore)).unwrap();
    let index = ChangeOp::Index { id: "a".into(), source_json: r#"{"n":1}"#.into(), routing: None, seq_no: 0, version: 1, primary_term: 3 };
    r.apply_replica_ops(3, vec![index], Durability::Request).unwrap();
    r.shard.fail_next_side_log_appends(1);
    let noop = ChangeOp::NoOp { seq_no: 1, primary_term: 3, reason: "gap".into() };
    assert!(r.apply_replica_ops(3, vec![noop], Durability::Request).is_err());
    assert!(r.shard.tragic_error().is_some(), "a failed side-log append left the engine running");
    let next = ChangeOp::Index { id: "b".into(), source_json: r#"{"n":2}"#.into(), routing: None, seq_no: 2, version: 1, primary_term: 3 };
    assert!(r.apply_replica_ops(3, vec![next], Durability::Request).is_err(), "a failed engine applied");
}

#[test]
fn retained_search_orders_timestamp_segments_like_live_search() {
    let (_tmp, node) = node();
    let svc = node.create_index("timestamps", &json!({"settings":{"number_of_shards":1,"number_of_replicas":0},
        "mappings":{"properties":{"@timestamp":{"type":"date"}}}})).unwrap();
    for (id, body) in [("old", json!({"@timestamp":"2021-08-01"})),
                       ("new", json!({"@timestamp":"2021-08-02"})), ("missing", json!({}))] {
        assert!(svc.write(vec![idx(id, body)], true)[0].is_ok());
        svc.refresh().unwrap();
    }
    // A single flush must not coalesce refresh-visible timestamp leaves.
    svc.flush().unwrap();
    let live = vec![Target::new(svc.clone())];
    let retained = node.snapshot(&live).unwrap();
    let req = parse(json!({"size":10}), &[]);
    for targets in [&live, &retained] {
        let out = crate::search::execute(targets, &req, &rv_opts()).unwrap();
        let value: Value = serde_json::from_str(&out.json).unwrap();
        let ids: Vec<_> = value["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap()).collect();
        assert_eq!(ids, vec!["new", "old", "missing"]);
    }
}

#[test]
fn retained_snapshot_reads_its_generation_after_store_replacement() {
    let (_temporary, node) = node();
    let svc = node.create_index("retained-generation", &json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"properties": {
            "vector": {"type":"dense_vector", "dims":2, "similarity":"l2_norm", "index_options":{"type":"hnsw"}},
            "tag": {"type":"keyword"}, "rank": {"type":"integer"}
        }}
    })).unwrap();
    assert!(svc.write((0..20).map(|i| idx(&format!("old-{i}"), json!({"vector":[i as f32,1.0],"tag":"old","rank":i}))).collect(), true).iter().all(Result::is_ok));
    svc.flush().unwrap();
    svc.refresh().unwrap();
    let captured = node.snapshot(&[Target::new(svc.clone())]).unwrap();
    // No search has decoded stored values/docvalues in these retained readers.
    let retired = svc.path().join("retired-generation");
    svc.set_aside_shard_store(0, &retired).unwrap();
    svc.open_shard(0, ShardOpenSpec::local_primary(RecoverySource::Empty)).unwrap();
    svc.write(vec![idx("replacement", json!({"vector":[0.0,1.0],"tag":"new","rank":-1}))], true).remove(0).unwrap();
    svc.flush().unwrap();
    svc.refresh().unwrap();
    std::fs::remove_dir_all(retired).unwrap();
    let execute = |body| {
        let req = parse(body, &[]);
        let out = crate::search::execute(&captured, &req, &ExecOpts { terms_lookup:None,cursors:None,max_buckets:100,scroll_id:None,pit_id:None }).unwrap();
        serde_json::from_str::<Value>(&out.json).unwrap()
    };
    let old = execute(json!({"size":20,"sort":[{"rank":"asc"}],"docvalue_fields":["tag","rank"],"query":{"term":{"tag":"old"}}}));
    assert_eq!(old["hits"]["hits"].as_array().unwrap().len(),20);
    for (i, hit) in old["hits"]["hits"].as_array().unwrap().iter().enumerate() {
        assert_eq!(hit["_id"], format!("old-{i}"));
        assert_eq!(hit["_source"]["tag"], "old");
        assert_eq!(hit["fields"]["rank"][0], i);
        assert_eq!(hit["fields"]["tag"][0], "old");
    }
    let knn = execute(json!({"size":1,"knn":{"field":"vector","query_vector":[0.0,1.0],"k":1,"num_candidates":20}}));
    assert_eq!(knn["hits"]["hits"][0]["_id"], "old-0");
    assert_eq!(ids(&exec(&svc, &parse(json!({"query":{"match_all":{}}}), &[]))), vec!["replacement"]);
}

/// Mapping publication runs while this write owns an operation permit. It
/// must be able to quarantine without draining that same permit, and neither
/// the in-flight request nor subsequent known-field writes may acknowledge.
#[test]
fn mapping_quarantine_fences_inflight_and_known_field_writes_without_draining() {
    struct Quarantine(std::sync::Weak<IndexService>);
    impl MappingUpdateHook for Quarantine {
        fn mapping_update(&self, _: &str, _: &str, _: &Value) -> Result<()> {
            self.0.upgrade().unwrap().quarantine_mapping();
            Ok(())
        }
    }
    let (_tmp, node) = node();
    let svc = node.create_index("quarantine", &json!({"settings":{"number_of_shards":1},
        "mappings":{"properties":{"n":{"type":"long"}}}})).unwrap();
    svc.write(vec![idx("before", json!({"n":1}))], true).remove(0).unwrap();
    let shard = svc.shard(0).unwrap();
    svc.set_mapping_update_hook(Some(Arc::new(Quarantine(Arc::downgrade(&svc)))));
    let error = shard.write(vec![idx("inflight", json!({"new_field":2}))], true).remove(0).unwrap_err();
    assert_eq!(error.kind, crate::index::COPY_REPLACED);
    assert!(shard.is_closed(), "new operation permits are fenced without draining");
    let error = shard.write(vec![idx("after", json!({"n":3}))], true).remove(0).unwrap_err();
    assert_eq!(error.kind, crate::index::COPY_REPLACED);
    assert_eq!(shard.update("before", &json!({"doc":{"n":1}}), None, 0, WriteCond::default()).unwrap_err().kind,
        crate::index::COPY_REPLACED);
    // A write that already completed its local apply before quarantine must
    // also fail the shared acknowledgement check (not just error results).
    assert_eq!(svc.attached_or(Ok(())).unwrap_err().kind, crate::index::COPY_REPLACED);
    svc.set_mapping_update_hook(None);
    svc.detach();
}

/// Replay of acknowledged ops is never refused or delayed by a circuit
/// breaker (ES `recoverFromTranslog` applies `LOCAL_TRANSLOG_RECOVERY` ops
/// with none): after a crash, a translog op larger than the whole request
/// breaker is validated, replayed and made searchable on restart, and the
/// shard comes up with it. The memory is accounted, not refused: no trip.
#[test]
fn an_acked_op_larger_than_the_request_breaker_replays_on_restart() {
    use crate::memory::{scoped, Breakers, RequestScope};
    let (_t, node) = node();
    let svc = node.create_index("big-replay", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}})).unwrap();
    let big = "x".repeat(1024 * 1024);
    svc.write(vec![idx("small", json!({"n": 1}))], true).remove(0).unwrap();
    svc.write(vec![idx("big", json!({"t": big}))], true).remove(0).unwrap();
    let copy = svc.shard(0).unwrap();
    let (checkpoint, max) = (copy.shard.local_checkpoint(), copy.shard.max_seq_no());
    // Crash: fence the copy without a commit (its translog holds both ops).
    let removed = svc.detach_shard_if_same(&copy).unwrap();
    removed.fence();
    let spec = ShardOpenSpec { allocation_id: removed.allocation_id.clone(), primary: true, primary_term: removed.shard.primary_term(), recovery: RecoverySource::ExistingStore };
    drop(removed);
    drop(copy);
    let tiny = Breakers::leak_for_tests(1 << 30);
    tiny.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
    // Far below the one op (a 1mb frame).
    tiny.request.set_limit(64 * 1024);
    let tripped = tiny.request.tripped();
    let scope = RequestScope::new("restart past the request breaker", tiny);
    let started = std::time::Instant::now();
    let reopened = scoped(Some(scope.clone()), || svc.open_shard(0, spec)).expect("the shard comes up");
    assert!(started.elapsed() < std::time::Duration::from_secs(30), "the replay waited for memory");
    assert_eq!((reopened.shard.local_checkpoint(), reopened.shard.max_seq_no()), (checkpoint, max));
    assert_eq!(tiny.request.tripped(), tripped, "nothing was refused");
    assert!(reopened.recovery_over_limit() > 0, "the replay ran under the small breaker and went past its limit");
    let g = svc.get("big").unwrap().expect("the big op replayed");
    assert!(g.source.unwrap().len() > 1024 * 1024);
    assert!(svc.get("small").unwrap().is_some());
    // The copy keeps what it charged while it lives, not this test's request.
    std::mem::forget(scope);
}

/// `cargo test -p elasticsearch-engine --release --lib metricbeat_bulk_microbench -- --ignored --nocapture`
/// (`MB_BULK_DOCS`, default 100,000; `MB_FIELDS`, default 2,000):
/// metricbeat-shaped bulks of 1,000 docs into a ~2,000-field mapping.
/// Each doc sets ~70 mapped fields over nested objects and two dynamic
/// `labels.*` keys, new keys arriving throughout the load. Prints docs/s
/// and how many field lookups the writes built.
#[test]
#[ignore]
fn metricbeat_bulk_microbench() {
    use std::sync::atomic::Ordering::Relaxed;
    let docs_n: usize = std::env::var("MB_BULK_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(100_000);
    let fields_n: usize = std::env::var("MB_FIELDS").ok().and_then(|v| v.parse().ok()).unwrap_or(2_000);
    let (_t, node) = node();
    // `m{g}.{s}.f{k}`: 40 groups of 5 sub-objects, leaves of mixed types.
    let mut props = serde_json::Map::new();
    let per_group = fields_n / 40;
    for g in 0..40 {
        let mut subs = serde_json::Map::new();
        for s in 0..5 {
            let mut leaves = serde_json::Map::new();
            for k in 0..per_group / 5 {
                let ty = match k % 4 { 0 => json!({"type": "long"}), 1 => json!({"type": "keyword", "ignore_above": 1024}), 2 => json!({"type": "scaled_float", "scaling_factor": 1000}), _ => json!({"type": "date"}) };
                leaves.insert(format!("f{k}"), ty);
            }
            subs.insert(format!("s{s}"), json!({"properties": leaves}));
        }
        props.insert(format!("m{g}"), json!({"properties": subs}));
    }
    props.insert("@timestamp".into(), json!({"type": "date"}));
    let body = json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.mapping.total_fields.limit": 20000, "refresh_interval": "-1"},
        "mappings": {"date_detection": false, "dynamic_templates": [{"labels": {"path_match": "labels.*", "match_mapping_type": "string", "mapping": {"type": "keyword"}}}], "properties": props}});
    let svc = node.create_index("mb", &body).unwrap();
    let doc = |i: usize| {
        let g = i % 40;
        let mut m = serde_json::Map::new();
        for s in 0..5 {
            let mut leaves = serde_json::Map::new();
            for k in (0..per_group / 5).step_by(per_group / 5 / 14 + 1) {
                let v = match k % 4 { 0 => json!(i as i64 * 7 + k as i64), 1 => json!(format!("v{}", (i + k) % 97)), 2 => json!((i % 1000) as f64 / 1000.0), _ => json!(1_550_000_000_000i64 + i as i64 * 1000) };
                leaves.insert(format!("f{k}"), v);
            }
            m.insert(format!("s{s}"), Value::Object(leaves));
        }
        let mut labels = serde_json::Map::new();
        labels.insert(format!("k{}", i % 1500), json!("a"));
        labels.insert(format!("pod{}", i / 700), json!("b"));
        let mut d = serde_json::Map::new();
        d.insert("@timestamp".into(), json!(1_550_000_000_000i64 + i as i64 * 1000));
        d.insert(format!("m{g}"), Value::Object(m));
        d.insert("labels".into(), Value::Object(labels));
        Value::Object(d)
    };
    let before = crate::mapping::LOOKUP_BUILDS.load(Relaxed);
    let t = std::time::Instant::now();
    let mut i = 0;
    while i < docs_n {
        let n = 1000.min(docs_n - i);
        for r in svc.write((i..i + n).map(|j| idx(&format!("d{j}"), doc(j))).collect(), false) { r.unwrap(); }
        i += n;
    }
    let secs = t.elapsed().as_secs_f64();
    let builds = crate::mapping::LOOKUP_BUILDS.load(Relaxed) - before;
    eprintln!("{docs_n} docs in {:.2} s: {:.0} docs/s; {builds} field lookups built ({} mapped fields at the end)",
        secs, docs_n as f64 / secs, svc.lookup().fields.len());
}

/// The fault matrix's search invariant on one node: after overwrites and
/// deletes, flushed segments, a merge with no commit after it
/// (merged-away sources retained, committed but not searchable), and a
/// refresh, a complete search's `hits.total` (match_all, exact), the
/// `stats.count` of a field every doc has, `_count`, the shard doc count
/// and ES|QL-visible live docs all agree, over two indices at once
/// (`jepsen-*`). Also once the idle tick has committed the merges.
#[test]
fn totals_and_counts_agree_with_merged_away_sources_retained() {
    let (_t, node) = node();
    node.node_settings.write().insert("indices.memory.shard_inactive_time".into(), json!("200ms"));
    let mapping = json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
        "mappings": {"properties": {"k": {"type": "keyword"}, "v": {"type": "long"}}}});
    let mut live = std::collections::BTreeMap::new();
    for name in ["jepsen-a", "jepsen-b"] {
        let svc = node.create_index(name, &mapping).unwrap();
        let sh = svc.local_shards().remove(0);
        for round in 0..12u32 {
            // New keys, overwrites of earlier ones, and a delete per round.
            let reqs: Vec<WriteReq> = (0..40u32).map(|i| { let key = (round * 25 + i) % 300; idx(&format!("{name}-{key}"), json!({"k": key.to_string(), "v": round})) }).collect();
            for r in svc.write(reqs, false) { r.unwrap(); }
            svc.write(vec![WriteReq::Delete { id: format!("{name}-{}", round * 7), cond: Default::default(), routing: None }], false);
            sh.shard.flush().unwrap();
        }
        // A merge after the last commit, with no commit after it (`flush`
        // false): its committed inputs stay referenced by the last commit.
        // The policy's own merges may all have run (and been committed)
        // during the rounds, so merge down to one segment.
        sh.shard.force_merge_with(1, false, false).unwrap();
        // Until the policy's merges ran and installed. Each needs a node-wide
        // merge permit, shared by every copy in the process: under a parallel
        // test run other tests' merges can hold them all, and a pass then
        // starts nothing (the request stays, as a node's scheduler retries
        // it); and the process-wide merge workers may be running this copy's
        // merge (`merge_current`) while this thread's pass finds no slot. So
        // wait on time, not on a count of passes.
        sh.shard.settle_background();
        svc.refresh().unwrap();
        assert!(sh.shard.reader_view().segment_stats().iter().any(|s| s.committed && !s.search), "{name}: the merge retained no committed source");
        live.insert(name, sh.doc_count().0);
    }
    let expect: u64 = live.values().sum();
    let check = |when: &str| {
        let targets = node.resolve("jepsen-*", Default::default()).unwrap();
        let req = parse(json!({"size": 0, "track_total_hits": true, "aggs": {"values": {"stats": {"field": "v"}}}}), &[("allow_partial_search_results", "false")]);
        let out: Value = serde_json::from_str(&crate::search::execute(&targets, &req, &ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None }).unwrap().json).unwrap();
        assert_eq!(out["hits"]["total"], json!({"value": expect, "relation": "eq"}), "{when}");
        assert_eq!(out["aggregations"]["values"]["count"], json!(expect), "{when}");
        assert_eq!(crate::search::count(&targets, None, None).unwrap(), expect, "{when}: _count");
        let views: u64 = ["jepsen-a", "jepsen-b"].iter().map(|n| node.resolve(n, Default::default()).unwrap()[0].svc.local_shards()[0].shard.reader_view().num_live_docs()).sum();
        assert_eq!(views, expect, "{when}: reader views");
    };
    check("merge sources retained");
    node.maintenance();
    std::thread::sleep(std::time::Duration::from_millis(300));
    node.maintenance();
    check("after the idle commit");
}

/// New dynamic fields of one bulk become one mapping update (one lookup
/// build, not one per document): every document is indexed against it,
/// the first document to introduce a field decides its type, and a later
/// document of the same bulk whose value does not fit fails as in ES.
#[test]
fn a_bulks_new_dynamic_fields_are_one_mapping_update() {
    let (_t, node) = node();
    let svc = node.create_index("dyn", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}})).unwrap();
    let mut reqs: Vec<WriteReq> = (0..300).map(|i| {
        let mut labels = serde_json::Map::new();
        labels.insert(format!("k{}", i % 120), json!(format!("v{i}")));
        idx(&format!("d{i}"), json!({"n": i, "labels": labels}))
    }).collect();
    reqs.push(idx("first", json!({"x": 5})));
    reqs.push(idx("second", json!({"x": "not a number"})));
    let generation = svc.schema_generation();
    let out = svc.write(reqs, true);
    assert!(out[..301].iter().all(|r| r.is_ok()), "{:?}", out.iter().find(|r| r.is_err()));
    let e = out[301].as_ref().err().expect("a value that does not fit the field the bulk introduced fails");
    assert_eq!(e.status, 400, "{e:?}");
    assert_eq!(svc.schema_generation() - generation, 1, "one mapping update for the bulk");
    let lk = svc.lookup();
    assert!((0..120).all(|k| lk.get(&format!("labels.k{k}")).is_some()));
    assert_eq!(lk.get("x").map(|f| f.ty), Some(crate::mapping::EsType::Long));
    svc.refresh().unwrap();
    let targets = node.resolve("dyn", Default::default()).unwrap();
    assert_eq!(crate::search::count(&targets, None, None).unwrap(), 301);
}

#[test]
fn adopting_files_takes_the_stored_block_field_id() {
    // The block's field is numbered when the first mapping is (before `s`
    // here); a shell made from the final mapping numbers it last. The files'
    // schema names it by its `core_ids` key, so the copy takes its id, and
    // a commit that named it `_stored_fields` (older builds) is read alike.
    let (_t, node) = node();
    let src = node.create_index("st", &json!({"settings": {"number_of_shards": 1}, "mappings": {"properties": {"z": {"type": "long"}}}})).unwrap();
    assert!(src.write((0..5).map(|i| idx(&format!("z{i}"), json!({"z": i}))).collect(), true).iter().all(|r| r.is_ok()));
    src.put_mapping(&json!({"properties": {"s": {"type": "keyword", "store": true}}})).unwrap();
    assert!(src.write((0..20).map(|i| idx(&format!("s{i}"), json!({"z": i, "s": format!("kept {i}")}))).collect(), true).iter().all(|r| r.is_ok()));
    let s0 = src.shard(0).unwrap();
    s0.shard.flush().unwrap();
    src.refresh().unwrap();
    let ids = src.meta().mappings.core_ids;
    let schema: std::collections::BTreeMap<String, u16> = s0.shard.core_schema().iter().filter(|f| f.id >= crate::mapping::FIRST_USER_FIELD).map(|f| (f.name.clone(), f.id)).collect();
    assert_eq!(schema.get(crate::mapping::STORED_BLOCK_KEY), ids.get(crate::mapping::STORED_BLOCK_KEY), "the commit names the block by its key: {schema:?}");
    let set = s0.snapshot_files(false).unwrap();
    let meta = json!({"settings": settings::as_flat_json(&src.meta().settings), "mappings": src.meta().mappings.to_json()});
    for (name, legacy) in [("copy-new", false), ("copy-old", true)] {
        let files_ids: std::collections::BTreeMap<String, u16> = schema.iter().map(|(k, id)| (if legacy && k == crate::mapping::STORED_BLOCK_KEY { crate::mapping::LEGACY_STORED_BLOCK_NAME.to_string() } else { k.clone() }, *id)).collect();
        let copy = node.create_index_sparse(name, &format!("{name}Uuid"), &meta).unwrap();
        copy.adopt_core_ids(&files_ids).unwrap();
        let got = copy.meta().mappings.core_ids;
        assert_eq!(got.get(crate::mapping::STORED_BLOCK_KEY), ids.get(crate::mapping::STORED_BLOCK_KEY), "{name}: {got:?}");
        assert!(!got.contains_key(crate::mapping::LEGACY_STORED_BLOCK_NAME), "{name}: {got:?}");
        let staged = node.new_staging_dir().unwrap();
        set.write_to_dir(&staged, false).unwrap();
        copy.open_shard(0, spec(true, 1, RecoverySource::FromFiles { staged, manifest: set.manifest.clone() })).unwrap();
        let req = parse(json!({"query": {"term": {"z": 7}}, "stored_fields": ["s"], "_source": true}), &[]);
        let (want, have) = (exec(&src, &req), exec(&copy, &req));
        assert_eq!(have["hits"]["hits"][0]["fields"], want["hits"]["hits"][0]["fields"], "{name}: {have}");
        assert_eq!(have["hits"]["hits"][0]["fields"]["s"], json!(["kept 7"]), "{name}: {have}");
        assert_eq!(have["hits"]["hits"][0]["_source"], want["hits"]["hits"][0]["_source"], "{name}");
    }
}

/// Field ids follow the order fields first appear: a copy whose mapping grew
/// field by field numbers them differently from a shell made from the final
/// mapping (sorted). Files adopted into such a shell are read under the
/// wrong fields unless it takes the files' ids first (`adopt_core_ids`);
/// then every field answers as on the source, a copy holding operations
/// refuses another change, fields mapped afterwards get fresh ids, and the
/// empty open copies of a full-open index are reopened under the new ids.
#[test]
fn adopting_files_takes_the_source_field_ids() {
    let (_t, node) = node();
    let src = node.create_index("grown", &json!({"settings": {"number_of_shards": 1}})).unwrap();
    // `z` is mapped (and numbered) before `a`.
    assert!(src.write((0..20).map(|i| idx(&format!("z{i}"), json!({"z": i}))).collect(), true).iter().all(|r| r.is_ok()));
    assert!(src.write((0..20).map(|i| idx(&format!("a{i}"), json!({"a": format!("w{}", i % 3)}))).collect(), true).iter().all(|r| r.is_ok()));
    let s0 = src.shard(0).unwrap();
    s0.shard.flush().unwrap();
    // Searchable on the source as on the adopted copy (published at open).
    src.refresh().unwrap();
    let ids = src.meta().mappings.core_ids;
    assert!(ids["z"] < ids["a"], "{ids:?}");
    let set = s0.snapshot_files(false).unwrap();
    let meta = json!({"settings": settings::as_flat_json(&src.meta().settings), "mappings": src.meta().mappings.to_json()});
    let copy = node.create_index_sparse("copy", "copyUuid3", &meta).unwrap();
    assert_ne!(copy.meta().mappings.core_ids, ids, "a shell made from the final mapping numbers its fields in sorted order");
    let user: std::collections::BTreeMap<String, u16> = ids.iter().filter(|(_, id)| **id >= crate::mapping::FIRST_USER_FIELD).map(|(k, id)| (k.clone(), *id)).collect();
    assert!(copy.adopt_core_ids(&user).unwrap());
    assert!(!copy.adopt_core_ids(&user).unwrap(), "already the files' ids");
    for (k, id) in &user {
        assert_eq!(copy.meta().mappings.core_ids.get(k), Some(id), "{k}");
    }
    let staged = node.new_staging_dir().unwrap();
    set.write_to_dir(&staged, false).unwrap();
    copy.open_shard(0, spec(true, 1, RecoverySource::FromFiles { staged, manifest: set.manifest.clone() })).unwrap();
    for body in [
        json!({"query": {"term": {"z": 7}}}),
        json!({"query": {"match": {"a": "w1"}}, "size": 50}),
        json!({"query": {"range": {"z": {"gte": 15}}}, "size": 50}),
        json!({"size": 0, "aggs": {"t": {"terms": {"field": "a.keyword"}}, "m": {"max": {"field": "z"}}}}),
    ] {
        let req = parse(body.clone(), &[]);
        let (want, got) = (exec(&src, &req), exec(&copy, &req));
        let (mut w, mut g) = (ids_of(&want), ids_of(&got));
        w.sort();
        g.sort();
        assert_eq!(g, w, "{body}");
        assert_eq!(got["hits"]["total"], want["hits"]["total"], "{body}");
        assert_eq!(got["aggregations"], want["aggregations"], "{body}");
    }
    let other: std::collections::BTreeMap<String, u16> = user.iter().map(|(k, id)| (k.clone(), *id + 100)).collect();
    assert_eq!(copy.adopt_core_ids(&other).unwrap_err().kind, "illegal_state_exception");
    copy.write(vec![idx("n1", json!({"new_field": "x"}))], true).remove(0).unwrap();
    let new_id = copy.meta().mappings.core_ids["new_field"];
    assert!(user.values().all(|id| *id != new_id), "{new_id}");
    // A full-open index: shard 0's slot is free, shard 1's empty copy is
    // reopened with a core schema under the adopted ids.
    let full = node.create_index("full", &json!({"settings": {"number_of_shards": 2}, "mappings": src.meta().mappings.to_json()})).unwrap();
    full.remove_shard(0, true).unwrap();
    let before = full.shard(1).unwrap();
    assert!(full.adopt_core_ids(&user).unwrap());
    let after = full.shard(1).unwrap();
    assert!(!Arc::ptr_eq(&before, &after) && before.is_closed());
    let schema = after.shard.core_schema();
    assert_eq!(schema.iter().find(|f| f.name == "z").map(|f| f.id), Some(user["z"]));
}

fn ids_of(v: &Value) -> Vec<String> {
    v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect()
}

/// `terms` on `_index` (ES `IndexFieldMapper`): one bucket per concrete
/// index with its own documents, over one index and over several; the
/// sub-aggregation sees only that index's documents.
#[test]
fn terms_on_index_names_each_index() {
    let (_t, node) = node();
    let mut svcs = Vec::new();
    for (name, n) in [("ix-a", 3usize), ("ix-b", 2usize)] {
        let svc = node.create_index(name, &json!({"settings": {"number_of_shards": 1}, "mappings": {"properties": {"v": {"type": "long"}}}})).unwrap();
        assert!(svc.write((0..n).map(|i| idx(&format!("d{i}"), json!({"v": i}))).collect(), true).iter().all(|r| r.is_ok()));
        svc.refresh().unwrap();
        svcs.push(svc);
    }
    let req = parse(json!({"size": 0, "track_total_hits": true, "aggs": {"by_index": {"terms": {"field": "_index", "size": 10}, "aggs": {"vc": {"value_count": {"field": "v"}}}}}}), &[]);
    let buckets = |v: &Value| -> Vec<(String, u64, u64)> {
        v["aggregations"]["by_index"]["buckets"].as_array().unwrap_or_else(|| panic!("{v}")).iter()
            .map(|b| (b["key"].as_str().unwrap().to_string(), b["doc_count"].as_u64().unwrap(), b["vc"]["value"].as_u64().unwrap())).collect()
    };
    assert_eq!(buckets(&exec(&svcs[0], &req)), vec![("ix-a".to_string(), 3, 3)]);
    let targets: Vec<Target> = svcs.iter().map(|s| Target::new(s.clone())).collect();
    let out = crate::search::execute(&targets, &req, &ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None }).unwrap();
    let both: Value = serde_json::from_str(&out.json).unwrap();
    assert_eq!(buckets(&both), vec![("ix-a".to_string(), 3, 3), ("ix-b".to_string(), 2, 2)], "{both}");
}

/// Index-sorted index with a `sparse_vector` field: its feature doc values
/// (the source of the vectors kept out of the stored `_source`) survive the
/// sorted refreshes' seals and the sorted force merge, value for value.
#[test]
fn sorted_index_keeps_sparse_vectors_through_refresh_and_merge() {
    use rustic_core::fieldtypes::features::quantize;
    let (_t, node) = node();
    let svc = node.create_index("sv-sorted", &json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1", "index.sort.field": "n", "index.sort.order": "desc"},
        "mappings": {"properties": {"n": {"type": "long"}, "sv": {"type": "sparse_vector"}, "kw": {"type": "keyword"}}}
    })).unwrap();
    let vector = |i: usize| -> serde_json::Map<String, Value> {
        (0..(3 + i % 11)).map(|k| (format!("t{}", (i * 31 + k * 7) % 97), json!(0.01 + ((i * 13 + k * 5) % 400) as f64 / 37.0))).collect()
    };
    for batch in 0..4 {
        let reqs: Vec<WriteReq> = (0..60).map(|j| {
            let i = batch * 60 + j;
            idx(&format!("d{i}"), json!({"n": (i * 7919) % 1009, "kw": format!("k{}", i % 5), "sv": vector(i)}))
        }).collect();
        for r in svc.write(reqs, false) { r.unwrap(); }
        svc.refresh().unwrap();
    }
    let read = || -> Vec<Value> {
        let v = exec(&svc, &parse(json!({"size": 1000, "sort": [{"n": "desc"}, "_doc"], "_source": {"exclude_vectors": false}}), &[]));
        v["hits"]["hits"].as_array().unwrap().iter().map(|h| json!([h["_id"], h["_source"]])).collect()
    };
    let before = read();
    assert_eq!(before.len(), 240);
    // Each vector comes back in its indexed form: tokens in index order,
    // weights at feature precision as Java floats.
    for h in &before {
        let i: usize = h[0].as_str().unwrap()[1..].parse().unwrap();
        let mut want: Vec<(String, f32)> = vector(i).into_iter().map(|(t, w)| (t, quantize(w.as_f64().unwrap() as f32))).collect();
        want.sort_by(|a, b| a.0.cmp(&b.0));
        let want: serde_json::Map<String, Value> = want.into_iter().map(|(t, w)| (t, crate::values::java_float(w))).collect();
        assert_eq!(h[1]["sv"], Value::Object(want), "doc {i}");
    }
    svc.force_merge(1, false).unwrap();
    svc.refresh().unwrap();
    assert_eq!(read(), before, "sorted force merge changed the sparse vectors");
    // Merging again (a single sorted segment) keeps them too.
    for r in svc.write(vec![idx("extra", json!({"n": 5, "sv": {"a": 1.5}}))], false) { r.unwrap(); }
    svc.refresh().unwrap();
    svc.force_merge(1, false).unwrap();
    svc.refresh().unwrap();
    let after: Vec<Value> = read().into_iter().filter(|h| h[0] != "extra").collect();
    assert_eq!(after, before);
}

/// Committed small segments for the merge-commit tests: `batches` flushes
/// of 50 docs each, under a merge policy that merges any two of them.
fn committed_small_segments(node: &Arc<Node>, name: &str, batches: usize) -> Arc<IndexService> {
    let svc = node.create_index(name, &json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1",
            "index.merge.policy.segments_per_tier": 2, "index.merge.policy.max_merge_at_once": 2, "index.merge.policy.floor_segment": "1kb"},
        "mappings": {"properties": {"k": {"type": "keyword"}, "n": {"type": "long"}}}
    })).unwrap();
    for b in 0..batches {
        let reqs: Vec<WriteReq> = (0..50).map(|i| idx(&format!("d{b}-{i}"), json!({"k": format!("v{}", i % 7), "n": b * 50 + i}))).collect();
        for r in svc.write(reqs, false) { r.unwrap(); }
        svc.refresh().unwrap();
        svc.flush().unwrap();
    }
    svc
}

/// A force merge without a segment count (Rally's `force-merge` with no
/// `max_num_segments`) returns once the policy's merges ran and are
/// committed and published: the inputs' files are gone and the doc count
/// (`_cat/indices docs.count`) is the live docs.
#[test]
fn force_merge_without_a_count_commits_and_publishes_its_merges() {
    let (_t, node) = node();
    let svc = committed_small_segments(&node, "fm-policy", 12);
    svc.force_merge(0, false).unwrap();
    let sh = svc.local_shards().remove(0);
    // Background merges may already have run some of the policy's merges;
    // either way every segment left is committed and searchable and holds
    // only live docs (no retained inputs).
    let stats = sh.shard.reader_view().segment_stats();
    assert!(stats.iter().all(|s| s.committed && s.search), "{stats:?}");
    assert_eq!(stats.iter().map(|s| s.docs).sum::<u64>(), 600, "segment docs = live docs");
    assert_eq!(sh.shard.merged_away_committed_bytes(), 0, "merged-away inputs are still committed");
    assert!(!sh.shard.has_uncommitted_changes(), "the merges were not committed");
    assert_eq!(svc.doc_count().0, 600, "docs.count after the merges");
    assert_eq!(sh.shard.queued_merges(), 0);
}

/// After the last write of an index, a merge that finishes later still
/// leaves something to commit (the periodic flush must not skip it on its
/// op count alone), and that commit frees the merged-away inputs.
#[test]
fn a_merge_after_the_last_write_is_an_uncommitted_change_until_flushed() {
    let (_t, node) = node();
    let svc = committed_small_segments(&node, "fm-idle", 8);
    let sh = svc.local_shards().remove(0);
    // Every op is committed; a merge runs now, after the last write, and is
    // left uncommitted (as a background merge is).
    assert!(sh.shard.max_seq_no() <= sh.shard.committed_seq_no());
    sh.shard.force_merge_with(1, false, false).unwrap();
    assert!(sh.shard.merged_away_committed_bytes() > 0, "no merge replaced a committed segment");
    assert!(sh.shard.max_seq_no() <= sh.shard.committed_seq_no(), "no new op");
    assert!(sh.shard.has_uncommitted_changes(), "a finished merge is an uncommitted change");
    let store = sh.shard.store_size();
    sh.shard.flush().unwrap();
    sh.shard.sweep_retired();
    assert_eq!(sh.shard.merged_away_committed_bytes(), 0);
    assert!(!sh.shard.has_uncommitted_changes());
    assert!(sh.shard.store_size() < store, "the merged-away inputs' files were not deleted: {} vs {store}", sh.shard.store_size());
    svc.refresh().unwrap();
    assert_eq!(svc.doc_count().0, 400);
}

/// `_cat/indices docs.count` counts the sealed docs of a search-idle copy
/// (ES counts over its internal reader): docs a flush sealed into hidden
/// segments count before any refresh publishes them.
#[test]
fn doc_count_counts_sealed_docs_before_a_refresh() {
    let (_t, node) = node();
    let svc = node.create_index("dc-sealed", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}})).unwrap();
    let reqs: Vec<WriteReq> = (0..300).map(|i| idx(&format!("d{i}"), json!({"n": i}))).collect();
    for r in svc.write(reqs, false) { r.unwrap(); }
    svc.flush().unwrap();
    assert_eq!(svc.doc_count().0, 300);
}

/// A synthetic-source index (Rally `cohere_vector`: `dynamic: false`,
/// unmapped text, an indexed float `dense_vector`) stores each vector
/// once, in the vector store: the stored `_source` holds the `null`
/// marker, not the vector as JSON text again. Gets, searches (vectors
/// excluded by default, included on request) and a force merge return
/// what the index storing its synthetic source whole returns
/// (`exclude_source_vectors: false`, which keeps no vector out of it):
/// exactly, but for a cosine vector, which comes back from the vector
/// store denormalized (as ES returns it), within float rounding.
#[test]
fn synthetic_source_keeps_indexed_vectors_out_of_the_stored_source() {
    let (_t, node) = node();
    let mapping = |cosine: bool| json!({"dynamic": false, "properties": {
        "emb": {"type": "dense_vector", "dims": 8, "index": true, "similarity": if cosine { "cosine" } else { "max_inner_product" }, "index_options": {"type": "hnsw"}},
        "obj": {"properties": {"b": {"type": "dense_vector", "element_type": "byte", "dims": 4, "index": true, "similarity": "l2_norm"}}},
        "n": {"type": "long"}}});
    let make = |name: &str, excludes: bool, cosine: bool| node.create_index(name, &json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1", "index.mapping.source.mode": "synthetic", "index.mapping.exclude_source_vectors": excludes},
        "mappings": mapping(cosine)})).unwrap();
    let doc = |i: usize| -> Value {
        let emb: Vec<Value> = (0..8).map(|k| json!(((i * 37 + k * 11) % 200) as f64 / 97.0 - 1.03 + 1e-7 * k as f64)).collect();
        let mut d = json!({"docid": format!("{i}#0"), "title": format!("title {i}"), "text": "some unmapped text", "n": i, "emb": emb});
        if i % 3 == 0 { d["obj"] = json!({"b": [i as i64 % 100, -3, 7, -(i as i64 % 50)], "x": "y"}); }
        if i % 7 == 0 { d.as_object_mut().unwrap().remove("emb"); }
        d
    };
    for cosine in [false, true] {
        let (a, c) = (make(&format!("syn-marked-{cosine}"), true, cosine), make(&format!("syn-whole-{cosine}"), false, cosine));
        for svc in [&a, &c] {
            for batch in 0..3 {
                for r in svc.write((batch * 20..batch * 20 + 20).map(|i| idx(&format!("d{i}"), doc(i))).collect(), false) { r.unwrap(); }
                svc.refresh().unwrap();
            }
        }
        // The stored `_source`: the marker for every indexed vector.
        for i in 0..60 {
            let id = format!("d{i}");
            let raw = |svc: &Arc<IndexService>| svc.get(&id).unwrap().unwrap().doc.get(crate::mapping::SOURCE_FIELD).and_then(|v| v.as_text()).map(str::to_string).unwrap();
            let (ra, rc) = (raw(&a), raw(&c));
            let want_doc = doc(i);
            if want_doc.get("emb").is_some() {
                assert!(rc.contains("\"emb\":["), "{rc}");
                assert!(ra.contains("\"emb\":null") && !ra.contains("\"emb\":["), "d{i}: {ra}");
            }
            if want_doc.get("obj").is_some() { assert!(ra.contains("\"b\":null"), "d{i}: {ra}"); }
            assert!(ra.contains("some unmapped text"));
            // Stored once: folded into the marked source, not also as
            // `_ignored_source` values (the whole source keeps both).
            let ignored = |svc: &Arc<IndexService>| svc.get(&id).unwrap().unwrap().doc.get_all(crate::mapping::IGNORED_SOURCE_FIELD).count();
            assert!(ignored(&c) > 0);
            if ra.contains("null") { assert_eq!(ignored(&a), 0, "d{i}: {ra}"); }
        }
        let read = |svc: &Arc<IndexService>, body: Value| -> Vec<Value> {
            let v = exec(svc, &parse(body, &[]));
            v["hits"]["hits"].as_array().unwrap_or_else(|| panic!("{v}")).iter().map(|h| json!([h["_id"], h["_source"], h["fields"]])).collect()
        };
        // Equal, a cosine index's vector components within float rounding.
        fn same(x: &Value, y: &Value, cosine: bool) -> bool {
            match (x, y) {
                (Value::Number(a), Value::Number(b)) if cosine => { let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap()); (a - b).abs() <= 1e-6 * a.abs().max(1.0) }
                (Value::Array(a), Value::Array(b)) => a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same(a, b, cosine)),
                (Value::Object(a), Value::Object(b)) => a.len() == b.len() && a.iter().zip(b).all(|((ka, a), (kb, b))| ka == kb && same(a, b, cosine)),
                _ => x == y,
            }
        }
        let check = |stage: &str| {
            let with = json!({"size": 100, "sort": [{"n": "asc"}], "_source": {"exclude_vectors": false}});
            let full_c = read(&c, with.clone());
            assert_eq!(full_c.len(), 60);
            let full_a = read(&a, with);
            assert!(same(&json!(full_a), &json!(full_c), cosine), "{stage} {cosine}: sources with vectors\n{full_a:?}\n{full_c:?}");
            // Default: vectors left out; the rest byte for byte.
            let stripped: Vec<Value> = full_c.iter().map(|h| {
                let mut h = h.clone();
                if let Some(s) = h[1].as_object_mut() { s.remove("emb"); if let Some(o) = s.get_mut("obj").and_then(Value::as_object_mut) { o.remove("b"); } }
                h
            }).collect();
            assert_eq!(read(&a, json!({"size": 100, "sort": [{"n": "asc"}]})), stripped, "{stage} {cosine}: default sources");
            let inc = json!({"size": 100, "sort": [{"n": "asc"}], "_source": {"includes": ["emb", "obj.b", "docid"]}, "fields": ["docid", "emb"]});
            assert!(same(&json!(read(&a, inc.clone())), &json!(read(&c, inc)), cosine), "{stage} {cosine}: filtered sources");
            // The unmapped strings' `_ignored_source` values, folded into
            // the marked source's leaves, come back as stored separately.
            let ign = json!({"size": 100, "sort": [{"n": "asc"}], "stored_fields": ["_ignored_source"], "_source": false});
            let ignored_hits = |svc: &Arc<IndexService>| -> Vec<Value> { exec(svc, &parse(ign.clone(), &[]))["hits"]["hits"].as_array().unwrap().iter().map(|h| { let mut h = h.clone(); h.as_object_mut().unwrap().remove("_index"); h }).collect() };
            let (ia, ic) = (ignored_hits(&a), ignored_hits(&c));
            assert!(ic.iter().all(|h| h.get("_ignored_source").is_some()), "{ic:?}");
            assert_eq!(ia, ic, "{stage} {cosine}: _ignored_source");
            for i in [0usize, 3, 7, 21, 59] {
                let id = format!("d{i}");
                let g = |svc: &Arc<IndexService>| -> Value { serde_json::from_str(&svc.get(&id).unwrap().unwrap().source.unwrap()).unwrap() };
                assert!(same(&g(&a), &g(&c), cosine), "{stage} {cosine}: get d{i}");
            }
        };
        check("refreshed");
        for svc in [&a, &c] { svc.force_merge(1, false).unwrap(); svc.refresh().unwrap(); }
        check("merged");
        // Realtime: a pending document (not yet refreshed) reads the same.
        for svc in [&a, &c] { for r in svc.write(vec![idx("rt", doc(61))], false) { r.unwrap(); } }
        let g = |svc: &Arc<IndexService>| -> Value { serde_json::from_str(&svc.get("rt").unwrap().unwrap().source.unwrap()).unwrap() };
        assert!(same(&g(&a), &g(&c), cosine));
    }
}

/// Synthetic `_source` renders float vector components as ES does (Java
/// `Float.toString`: the shortest digits, ties to even), so a component
/// halfway between two shortest decimals (0.369140625 = 0.36914062|3) reads
/// back as ES's `0.36914062`, in synthetic and stored mode alike, whether
/// it comes from the stored source or the vector store (realtime, after a
/// refresh and with vectors kept in `_source` by the setting). ES 9.5.5:
/// `{"emb":[0.36914062,0.25,1.0]}` for `[0.369140625,0.25,1]` in both modes.
#[test]
fn synthetic_vector_components_render_like_es() {
    let (_t, node) = node();
    let es = "[0.36914062,0.25,1.0,0.36914062,-0.25097656,0.3]";
    let vector = json!([0.369140625, 0.25, 1, 0.3691406250, -0.250976563, 0.3]);
    for (mode, excludes) in [("synthetic", true), ("synthetic", false), ("stored", true)] {
        let name = format!("tie-{mode}-{excludes}");
        let svc = node.create_index(&name, &json!({
            "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1", "index.mapping.source.mode": mode, "index.mapping.exclude_source_vectors": excludes},
            "mappings": {"properties": {"emb": {"type": "dense_vector", "dims": 6, "index": true, "similarity": "max_inner_product"}}}})).unwrap();
        for r in svc.write(vec![idx("1", json!({"emb": vector}))], false) { r.unwrap(); }
        let realtime: Value = serde_json::from_str(&svc.get("1").unwrap().unwrap().source.unwrap()).unwrap();
        assert_eq!(serde_json::to_string(&realtime["emb"]).unwrap(), es, "{name}: realtime get");
        svc.refresh().unwrap();
        let v = exec(&svc, &parse(json!({"_source": {"exclude_vectors": false}}), &[]));
        assert_eq!(serde_json::to_string(&v["hits"]["hits"][0]["_source"]["emb"]).unwrap(), es, "{name}: search");
        let got: Value = serde_json::from_str(&svc.get("1").unwrap().unwrap().source.unwrap()).unwrap();
        assert_eq!(serde_json::to_string(&got["emb"]).unwrap(), es, "{name}: get");
    }
}

/// logsdb keeps `_seq_no` as doc values only (as ES 9 does) and a dense
/// constant date column (`event.ingested`) without a points index: range,
/// term and exists queries and sorts on them return what a standard index
/// returns, refreshed and merged.
/// Script scoring over a large segment split into doc-range parts
/// (collected on the search pool) returns what one pass returns: the same
/// hits, scores and totals, ties broken by doc.
#[test]
fn script_scoring_in_doc_range_parts_matches_one_pass() {
    let (_t, node) = node();
    let svc = node.create_index("parts", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
        "mappings": {"properties": {"n": {"type": "long"}, "k": {"type": "keyword"}, "population": {"type": "long"}, "location": {"type": "geo_point"}}}})).unwrap();
    let docs = (0..3000i64).map(|i| idx(&format!("d{i}"), json!({"n": (i * 7919) % 1013, "k": format!("k{}", i % 4), "population": (i * 104729) % 100_000,
        "location": {"lat": ((i * 37) % 170) as f64 - 85.0 + 0.25, "lon": ((i * 53) % 350) as f64 - 175.0 + 0.5}}))).collect();
    for r in svc.write(docs, false) { r.unwrap(); }
    svc.flush().unwrap();
    svc.refresh().unwrap();
    let queries = [
        json!({"function_score": {"query": {"match_all": {}}, "functions": [{"script_score": {"script": {"source": "doc['n'].value % 97", "lang": "expression"}}}]}}),
        json!({"function_score": {"query": {"term": {"k": "k1"}}, "functions": [{"script_score": {"script": {"source": "Math.log(doc['n'].value + 2)"}}}]}}),
        json!({"script_score": {"query": {"match_all": {}}, "script": {"source": "doc['n'].value % 13"}}}),
        // geonames' expression: a compound top-k plan over the geo index,
        // each part's plan over its doc range.
        json!({"function_score": {"query": {"match_all": {}}, "functions": [{"script_score": {"script": {"source": "abs(ln(abs(doc['population']) + 1) + doc['location'].lon + doc['location'].lat) * _score", "lang": "expression"}}}]}}),
    ];
    let run = |q: &Value, tth: Value| exec(&svc, &parse(json!({"size": 25, "track_total_hits": tth, "query": q}), &[]));
    let hits = |v: &Value| v["hits"]["hits"].as_array().unwrap_or_else(|| panic!("{v}")).iter().map(|h| (h["_id"].as_str().unwrap().to_string(), h["_score"].as_f64().unwrap())).collect::<Vec<_>>();
    let prev = rustic_core::query::scorer::PART_SPLIT_MIN_DOCS.load(std::sync::atomic::Ordering::Relaxed);
    for q in &queries {
        for tth in [json!(true), json!(100)] {
            rustic_core::query::scorer::PART_SPLIT_MIN_DOCS.store(u32::MAX, std::sync::atomic::Ordering::Relaxed);
            let one = run(q, tth.clone());
            rustic_core::query::scorer::PART_SPLIT_MIN_DOCS.store(256, std::sync::atomic::Ordering::Relaxed);
            let parts = run(q, tth.clone());
            assert_eq!(hits(&parts), hits(&one), "{q} {tth}");
            assert_eq!(parts["hits"]["total"], one["hits"]["total"], "{q} {tth}");
        }
    }
    // Histograms of metrics and plain stats fold doc-range parts too: the
    // same buckets and counts, sums combined from the parts' compensated
    // sums (equal to a last-bit rounding).
    let aggs = [
        json!({"h": {"histogram": {"field": "n", "interval": 50}, "aggs": {"s": {"stats": {"field": "n"}}}}}),
        json!({"s": {"stats": {"field": "n"}}, "a": {"avg": {"field": "n"}}}),
    ];
    fn close(a: &Value, b: &Value, path: &str) {
        match (a, b) {
            (Value::Object(x), Value::Object(y)) => { assert_eq!(x.len(), y.len(), "{path}"); for (k, v) in x { close(v, &y[k], &format!("{path}.{k}")); } }
            (Value::Array(x), Value::Array(y)) => { assert_eq!(x.len(), y.len(), "{path}"); for (i, (u, v)) in x.iter().zip(y).enumerate() { close(u, v, &format!("{path}[{i}]")); } }
            (Value::Number(x), Value::Number(y)) if x.is_f64() || y.is_f64() => { let (u, v) = (x.as_f64().unwrap(), y.as_f64().unwrap()); assert!((u - v).abs() <= 1e-12 * u.abs().max(1.0), "{path}: {u} vs {v}"); }
            _ => assert_eq!(a, b, "{path}"),
        }
    }
    for a in &aggs {
        for q in [json!({"match_all": {}}), json!({"range": {"n": {"gte": 10, "lt": 900}}})] {
            rustic_core::query::scorer::PART_SPLIT_MIN_DOCS.store(u32::MAX, std::sync::atomic::Ordering::Relaxed);
            let one = exec(&svc, &parse(json!({"size": 0, "query": q, "aggs": a}), &[]));
            rustic_core::query::scorer::PART_SPLIT_MIN_DOCS.store(256, std::sync::atomic::Ordering::Relaxed);
            let parts = exec(&svc, &parse(json!({"size": 0, "query": q, "aggs": a}), &[]));
            close(&parts["aggregations"], &one["aggregations"], "aggregations");
        }
    }
    rustic_core::query::scorer::PART_SPLIT_MIN_DOCS.store(prev, std::sync::atomic::Ordering::Relaxed);
}

/// Aggregations over two single-shard indices (a data stream's backing
/// indices) reduce per-shard partials as ES 9.5.5 does: each shard's top
/// `shard_size` terms, doc_count_error_upper_bound and sum_other_doc_count
/// from them, significant_terms' doubled shard queue, ties ordered as ES's
/// priority queues and reduce map order them. The expected responses were
/// recorded from ES 9.5.5 on the same documents
/// (`aggs/test_data/es955_shard_terms.json`).
#[test]
fn shard_reduced_terms_match_recorded_es_responses() {
    let (_t, node) = node();
    let doc = |n: i64, i: i64| {
        let p = ["a", "b", "c", "d", "e"][((i * 3 + n) % 5) as usize];
        json!({"k": format!("k{:02}", (i * i + 3 * i * n + 11 * n) % 41), "p": p, "v": i})
    };
    let mut svcs = Vec::new();
    for n in 0..2i64 {
        let svc = node.create_index(&format!("parity{n}"), &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
            "mappings": {"properties": {"k": {"type": "keyword"}, "p": {"type": "keyword"}, "v": {"type": "long"}}}})).unwrap();
        for r in svc.write((0..400i64).map(|i| idx(&format!("{n}-{i}"), doc(n, i))).collect(), false) { r.unwrap(); }
        svc.flush().unwrap();
        svc.refresh().unwrap();
        svcs.push(svc);
    }
    let recorded: Value = serde_json::from_str(include_str!("aggs/test_data/es955_shard_terms.json")).unwrap();
    for (name, case) in recorded.as_object().unwrap() {
        let req = parse(case["request"].clone(), &[("request_cache", "false")]);
        let targets: Vec<Target> = svcs.iter().map(|s| Target::new(s.clone())).collect();
        let out = crate::search::execute(&targets, &req, &ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None }).unwrap();
        let mut got: Value = serde_json::from_str(&out.json).unwrap();
        let mut want = case["aggregations"].clone();
        if name == "terms_sig" {
            // Under a parent bucket, which of several significant terms
            // tied on score ES keeps follows its collection internals
            // (not reproduced): those buckets compare by score and counts.
            fn untie(v: &mut Value) {
                for outer in v["t"]["buckets"].as_array_mut().unwrap() {
                    let inner = outer["s"]["buckets"].as_array_mut().unwrap();
                    let scores: Vec<f64> = inner.iter().map(|b| b["score"].as_f64().unwrap()).collect();
                    for b in inner.iter_mut() {
                        let sc = b["score"].as_f64().unwrap();
                        if scores.iter().filter(|&&x| x == sc).count() > 1 { b.as_object_mut().unwrap().remove("key"); }
                    }
                }
            }
            untie(&mut got["aggregations"]);
            untie(&mut want);
        }
        assert_eq!(got["aggregations"], want, "{name}");
    }
}

/// A range over a doc-values-only field (no points: logsdb's `@timestamp`,
/// big5's `metrics.size`) in a search that stops after its first matches
/// (`size` 5, `track_total_hits` 100) finds them lazily in doc order: the
/// same hits as the bitset path an exact total takes, every leaf order and
/// deleted doc respected, and the bounded total reported as `gte`.
#[test]
fn early_stopping_ranges_on_doc_values_match_the_full_bitset() {
    let (_t, node) = node();
    let svc = node.create_index("dv-range", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
        "mappings": {"properties": {"n": {"type": "long", "index": false}, "t": {"type": "date", "index": false}, "k": {"type": "keyword"}}}})).unwrap();
    for batch in 0..3 {
        let docs = (batch * 700..batch * 700 + 700).map(|i: i64| idx(&format!("d{i}"), json!({"n": (i * 37) % 1000, "t": 1_700_000_000_000i64 + i * 1000, "k": format!("k{}", i % 3)}))).collect();
        for r in svc.write(docs, false) { r.unwrap(); }
        svc.flush().unwrap();
        svc.refresh().unwrap();
    }
    for r in svc.write((0..40).map(|i| WriteReq::Delete { id: format!("d{}", i * 13), routing: None, cond: Default::default() }).collect(), false) { r.unwrap(); }
    svc.flush().unwrap();
    svc.refresh().unwrap();
    let ids = |v: &Value| v["hits"]["hits"].as_array().unwrap_or_else(|| panic!("{v}")).iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect::<Vec<_>>();
    for q in [json!({"range": {"n": {"gte": 100, "lte": 400}}}), json!({"range": {"n": {"gt": 990}}}), json!({"range": {"t": {"gte": 1_700_000_500_000i64, "lt": 1_700_001_900_000i64}}}),
              json!({"range": {"t": {"lt": 1_600_000_000_000i64}}}), json!({"bool": {"filter": [{"range": {"n": {"lte": 30}}}]}}), json!({"constant_score": {"filter": {"range": {"n": {"gte": 0}}}}}),
              // big5's range_field_* shapes: a term AND / OR a scored range.
              json!({"bool": {"must": [{"term": {"k": "k1"}}, {"range": {"n": {"gte": 1, "lte": 300}}}]}}), json!({"bool": {"should": [{"term": {"k": "k2"}}, {"range": {"n": {"gte": 10, "lte": 20}}}]}}),
              json!({"bool": {"should": [{"term": {"k": "k0"}}, {"range": {"t": {"gte": 1_700_000_100_000i64, "lte": 1_700_001_000_000i64}}}]}})] {
        let lazy = exec(&svc, &parse(json!({"size": 5, "track_total_hits": 100, "query": q}), &[]));
        let full = exec(&svc, &parse(json!({"size": 5, "track_total_hits": true, "query": q}), &[]));
        assert_eq!(ids(&lazy), ids(&full), "{q}");
        let exact = full["hits"]["total"]["value"].as_u64().unwrap();
        let (value, relation) = (lazy["hits"]["total"]["value"].as_u64().unwrap(), lazy["hits"]["total"]["relation"].as_str().unwrap().to_string());
        if exact > 100 { assert_eq!((value, relation.as_str()), (100, "gte"), "{q}"); } else { assert_eq!((value, relation.as_str()), (exact, "eq"), "{q}"); }
    }
}

#[test]
fn seq_no_and_constant_dates_without_points_answer_like_standard() {
    let (_t, node) = node();
    let mapping = json!({"properties": {"@timestamp": {"type": "date"}, "event": {"properties": {"ingested": {"type": "date"}}}, "host": {"properties": {"name": {"type": "keyword"}}}, "n": {"type": "long"}}});
    let mk = |name: &str, mode: Option<&str>| {
        let mut settings = json!({"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"});
        if let Some(m) = mode { settings["index.mode"] = json!(m); }
        node.create_index(name, &json!({"settings": settings, "mappings": mapping.clone()})).unwrap()
    };
    let (l, s) = (mk("pts-logsdb", Some("logsdb")), mk("pts-std", None));
    assert!(l.lookup().core_schema().iter().any(|f| f.name == "_seq_no" && !f.points));
    assert!(s.lookup().core_schema().iter().any(|f| f.name == "_seq_no" && f.points));
    let doc = |i: usize| json!({"@timestamp": 1_672_531_200_000i64 + (i as i64 % 97) * 1000, "event": {"ingested": "2023-07-20T04:20:34.181Z"}, "host": {"name": format!("h{}", i % 5)}, "n": i});
    for svc in [&l, &s] {
        for batch in 0..3 {
            for r in svc.write((batch * 400..batch * 400 + 400).map(|i| idx(&format!("d{i}"), doc(i))).collect(), false) { r.unwrap(); }
            svc.refresh().unwrap();
        }
    }
    let queries = vec![
        json!({"range": {"_seq_no": {"gte": 100, "lt": 350}}}),
        json!({"range": {"_seq_no": {"gt": 1100}}}),
        json!({"term": {"_seq_no": 7}}),
        json!({"range": {"event.ingested": {"gte": "2023-07-20T00:00:00Z", "lte": "2023-07-21T00:00:00Z"}}}),
        json!({"range": {"event.ingested": {"gt": "2023-07-20T04:20:34.181Z"}}}),
        json!({"range": {"event.ingested": {"lt": "2023-07-20T04:20:34.181Z"}}}),
        json!({"term": {"event.ingested": "2023-07-20T04:20:34.181Z"}}),
        json!({"bool": {"filter": [{"range": {"event.ingested": {"lte": "now"}}}, {"range": {"_seq_no": {"lt": 50}}}]}}),
        json!({"exists": {"field": "event.ingested"}}),
    ];
    let ids = |svc: &Arc<IndexService>, q: &Value, sort: Value| -> (u64, Vec<String>) {
        let v = exec(svc, &parse(json!({"size": 2000, "track_total_hits": true, "query": q, "sort": sort}), &[]));
        (v["hits"]["total"]["value"].as_u64().unwrap(), {
            let mut ids: Vec<String> = v["hits"]["hits"].as_array().unwrap_or_else(|| panic!("{v}")).iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
            ids.sort();
            ids
        })
    };
    let check = |stage: &str| {
        for q in &queries {
            let (a, b) = (ids(&l, q, json!([{"n": "asc"}])), ids(&s, q, json!([{"n": "asc"}])));
            assert_eq!(a, b, "{stage}: {q}");
        }
        // Sorting on the constant column ties every doc.
        let (a, b) = (ids(&l, &json!({"match_all": {}}), json!([{"event.ingested": "desc"}, {"n": "asc"}])), ids(&s, &json!({"match_all": {}}), json!([{"event.ingested": "desc"}, {"n": "asc"}])));
        assert_eq!(a, b, "{stage}: sort");
        assert_eq!(a.0, 1200);
    };
    check("refreshed");
    for svc in [&l, &s] { svc.force_merge(1, false).unwrap(); svc.refresh().unwrap(); }
    check("merged");
}
