//! End-to-end tests on real engine indices in temp dirs (no networking):
//! an in-process primary + replicas harness.

use super::flush_gate::{safe_flush, FlushOutcome, WriteGate};
use super::recovery::{RecoveryOptions, RecoverySource, RecoveryStage, RecoveryTarget};
use super::*;
use elasticsearch_engine::index::{IndexService, WriteReq};
use elasticsearch_engine::mapping::{ID_FIELD, ROUTING_FIELD, SEQ_NO_FIELD, SOURCE_FIELD, VERSION_FIELD};
use elasticsearch_engine::search::{count, Target};
use elasticsearch_engine::settings;
use elasticsearch_engine::shard::WriteCond;
use elasticsearch_engine::translog::TranslogOp;
use parking_lot::Mutex;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const TERM: u64 = 1;

#[test]
fn replica_size_flush_waits_for_reclaimable_history() {
    let tmp = tempfile::tempdir().unwrap();
    let (svc, r) = replica(&tmp.path().join("replica"), &json!({}));
    let op = |seq| ReplicaOp::Index { id: format!("doc-{seq}"), source_json: "{}".into(),
        routing: None, seq_no: seq, version: 1, primary_term: TERM };
    r.apply(&[op(0)], None).unwrap();
    assert!(r.stats().wal_size_bytes > 1);
    assert!(!r.needs_size_flush(1), "unacknowledged recovery history cannot be trimmed");
    assert!(!r.needs_periodic_flush(u64::MAX, true), "age must not bypass the recovery fence");
    // An explicit flush remains legal but must not trigger a checkpoint loop.
    r.flush().unwrap();
    assert!(!r.needs_size_flush(1));
    r.update_global_checkpoint(0).unwrap();
    assert!(r.needs_size_flush(1));
    r.flush().unwrap();
    assert!(!r.needs_size_flush(1));
    assert_eq!(r.reusable_prefix().unwrap().checkpoint, 0);
    r.apply(&[op(1)], None).unwrap();
    assert!(!r.needs_size_flush(1), "new writes alone do not make the retained WAL reclaimable");
    r.update_global_checkpoint(1).unwrap();
    assert!(r.needs_size_flush(1));
    assert!(!r.needs_size_flush(u64::MAX));
    assert!(!r.needs_periodic_flush(u64::MAX, false));
    assert!(r.needs_periodic_flush(u64::MAX, true), "acknowledged history can age-flush below the size threshold");
    r.flush().unwrap();
    assert!(!r.needs_periodic_flush(u64::MAX, true), "already trimmed history must not flush again");
    assert!(!r.needs_size_flush(1));
    drop(r); drop(svc);
    let svc = Arc::new(IndexService::open(&tmp.path().join("replica")).unwrap());
    let reopened = ReplicaApplier::open(svc, 0).unwrap();
    assert_eq!(reopened.reusable_prefix().unwrap().checkpoint, 1);
    assert!(!reopened.needs_size_flush(1));
}

/// The reusable prefix stops at the global checkpoint: ops above it (an
/// unacknowledged suffix, committed or not) are re-sent by the recovery's
/// translog phase and, under a newer term, verified one by one.
#[test]
fn reusable_prefix_stops_at_the_global_checkpoint() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("replica");
    let (svc, r) = replica(&path, &json!({}));
    let op = ReplicaOp::Index { id: "kept".into(), source_json: "{}".into(), routing: None, seq_no: 0, version: 1, primary_term: TERM };
    r.apply(&[op], None).unwrap();
    assert!(r.reusable_prefix().is_none(), "no global checkpoint yet: nothing is known common");
    r.flush().unwrap();
    assert!(r.reusable_prefix().is_none(), "flushing does not turn an unacknowledged suffix into common history");
    r.update_global_checkpoint(0).unwrap();
    assert_eq!(r.reusable_prefix().unwrap().checkpoint, 0, "no commit needed: the WAL is durable");
    r.flush().unwrap();
    assert_eq!(r.reusable_prefix().unwrap().checkpoint, 0);
    drop(r); drop(svc);
    let svc = Arc::new(IndexService::open(&path).unwrap());
    let r = ReplicaApplier::open(svc.clone(), 0).unwrap();
    assert_eq!(r.reusable_prefix().unwrap().checkpoint, 0, "proof survives a real reopen");
    r.apply(&[ReplicaOp::Delete { id: "kept".into(), seq_no: 1, version: 2, primary_term: TERM }], None).unwrap();
    r.flush().unwrap();
    assert_eq!(r.reusable_prefix().unwrap().checkpoint, 0, "a committed delete above the global checkpoint is re-sent, not trusted");
    r.bump_term(TERM + 1, 0).unwrap();
    assert!(r.resync_pending());
    let p = r.reusable_prefix().unwrap();
    assert_eq!((p.checkpoint, p.primary_term), (0, TERM + 1), "the suspect above 0 is verified by the recovery");
    assert_eq!(r.unverified_suspects(), vec![1]);
    // The new primary's op at seq_no 1 is the same delete (with the term
    // that assigned it), sent by the term-2 recovery: verified.
    r.apply_under(&[ReplicaOp::Delete { id: "kept".into(), seq_no: 1, version: 2, primary_term: TERM }], Some(TERM + 1), None, 0).unwrap();
    assert!(!r.resync_pending(), "every suspect verified: the resync is resolved");
    assert_eq!(r.local_checkpoint(), 1);
    r.fail_unverified_suspects().unwrap();
}

/// A suspect the primary's history does not hold fails the copy once the
/// recovery's translog phase caught up (it then recovers from files).
#[test]
fn unverified_suspects_fail_the_copy_after_the_translog_phase() {
    let tmp = tempfile::tempdir().unwrap();
    let (_svc, r) = replica(&tmp.path().join("replica"), &json!({}));
    let op = |seq: i64| ReplicaOp::Index { id: format!("d{seq}"), source_json: "{}".into(), routing: None, seq_no: seq, version: 1, primary_term: TERM };
    r.apply(&[op(0), op(1), op(2)], None).unwrap();
    r.update_global_checkpoint(0).unwrap();
    r.bump_term(TERM + 1, 0).unwrap();
    assert_eq!(r.local_checkpoint(), 0, "capped at the resync checkpoint");
    r.apply_under(&[op(1)], Some(TERM + 1), None, 0).unwrap();
    assert_eq!(r.unverified_suspects(), vec![2]);
    assert!(matches!(r.fail_unverified_suspects(), Err(ReplicationError::NeedsFullRecovery { .. })));
    assert!(r.needs_full_recovery().is_some(), "sticky");
    assert!(r.reusable_prefix().is_none());
    // A different op at a suspect's seq_no fails at once.
    let tmp2 = tempfile::tempdir().unwrap();
    let (_svc2, r2) = replica(&tmp2.path().join("replica"), &json!({}));
    r2.apply(&[op(0), op(1)], None).unwrap();
    r2.update_global_checkpoint(0).unwrap();
    r2.bump_term(TERM + 1, 0).unwrap();
    let other = ReplicaOp::Index { id: "other".into(), source_json: "{}".into(), routing: None, seq_no: 1, version: 1, primary_term: TERM + 1 };
    assert!(matches!(r2.apply(&[other], None), Err(ReplicationError::NeedsFullRecovery { .. })));
}

#[test]
fn replica_committed_outside_the_applier_stays_reusable() {
    // `_forcemerge` and REST `_flush` commit a replica's engine directly. The
    // copy must still offer its committed (merged) segments for reuse after a
    // restart, instead of being wiped and rebuilt from the primary's history.
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("replica");
    let (svc, r) = replica(&path, &json!({}));
    let op = |seq: i64| ReplicaOp::Index { id: format!("doc-{seq}"), source_json: "{}".into(), routing: None, seq_no: seq, version: 1, primary_term: TERM };
    for seq in 0..10 {
        r.apply(&[op(seq)], None).unwrap();
        crate::shards::copy_of(&svc, 0).shard.refresh().unwrap();
    }
    r.update_global_checkpoint(9).unwrap();
    svc.force_merge(1, false).unwrap();
    assert_eq!(r.reusable_prefix().unwrap().checkpoint, 9, "the WAL floor may lag the engine commit");
    r.adopt_engine_commit().unwrap();
    assert_eq!(r.reusable_prefix().unwrap().checkpoint, 9);
    drop(r); drop(svc);
    let svc = Arc::new(IndexService::open(&path).unwrap());
    let r = ReplicaApplier::open(svc.clone(), 0).unwrap();
    assert_eq!(r.stats().wal_ops_replayed, 0, "the commit's ops are not replayed on reopen");
    assert_eq!(r.reusable_prefix().unwrap().checkpoint, 9);
    assert_eq!(svc.num_segments(), 1, "the merged segment is kept");
    // A suffix the commit does not hold, or an unacknowledged one, is never
    // adopted: the prefix stays at the global checkpoint.
    r.apply(&[op(10)], None).unwrap();
    r.adopt_engine_commit().unwrap();
    assert_eq!(r.reusable_prefix().unwrap().checkpoint, 9);
    svc.flush().unwrap();
    r.adopt_engine_commit().unwrap();
    assert_eq!(r.reusable_prefix().unwrap().checkpoint, 9, "committed but above the global checkpoint");
}

#[test]
fn reusable_prefix_rejects_wal_only_crash_suffix() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("replica");
    let (svc, r) = replica(&path, &json!({}));
    let op = |seq| ReplicaOp::Index { id: format!("doc-{seq}"), source_json: "{}".into(), routing: None, seq_no: seq, version: 1, primary_term: TERM };
    r.apply(&[op(0)], None).unwrap();
    r.update_global_checkpoint(0).unwrap(); r.flush().unwrap();
    assert!(r.reusable_prefix().is_some());
    r.wal_append_only(&[op(1)]).unwrap();
    drop(r); drop(svc);
    let svc = Arc::new(IndexService::open(&path).unwrap());
    let r = ReplicaApplier::open(svc.clone(), 0).unwrap();
    assert_eq!(r.local_checkpoint(), 1);
    assert!(svc.get("doc-1").unwrap().is_some());
    assert_eq!(r.reusable_prefix().unwrap().checkpoint, 0, "the replayed suffix is re-sent from above the global checkpoint");
}

/// B5: a replica-prefix recovery accepts retained history whose frames are
/// not in seq_no order (seq_no 3 appended before 2): it checks coverage,
/// not frame order (frame order forced a needless full recovery).
#[test]
fn replica_prefix_resume_accepts_reordered_frames() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = create(&tmp.path().join("primary"), &json!({}));
    index(&primary, "a", json!({"n": 1}));
    index(&primary, "b", json!({"n": 2}));
    let copy = crate::shards::copy_of(&primary, 0);
    let dir = copy.shard.path().join("translog");
    let generation = std::fs::read_dir(&dir).unwrap()
        .filter_map(|e| e.ok()?.file_name().to_str()?.strip_prefix("translog-")?.strip_suffix(".tlog")?.parse::<u64>().ok())
        .max().unwrap();
    let mut log = elasticsearch_engine::translog::Translog::open(&dir, generation).unwrap();
    for seq_no in [3u64, 2] {
        log.append_many(&[TranslogOp::Index { id: format!("x{seq_no}"), source_json: "{}".into(), seq_no, version: 1, routing: None }]).unwrap();
    }
    log.fsync().unwrap();
    // The engine has sequenced through 3.
    copy.shard.fill_seq_no_gaps(&[2, 3], TERM, "sequenced by the test").unwrap();
    assert_eq!(copy.shard.max_seq_no(), 3);
    let source = RecoverySource::resume(primary.clone(), 0, TERM, 1, RecoveryOptions::default())
        .expect("reordered retained frames forced a full recovery");
    assert_eq!(source.info().translog_ops_on_start, 2);
    // A real gap still falls back.
    copy.shard.fill_seq_no_gaps(&[4], TERM, "sequenced, never logged").unwrap();
    assert!(matches!(RecoverySource::resume(primary.clone(), 0, TERM, 1, RecoveryOptions::default()), Err(ReplicationError::OpsTrimmed { .. })));
}

#[test]
fn replica_prefix_recovery_replays_deletes_and_primary_advancement_without_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = create(&tmp.path().join("primary"), &json!({}));
    index(&primary, "keep", json!({"n":1}));
    index(&primary, "delete", json!({"n":2}));
    let (svc, r) = replica(&tmp.path().join("replica"), &json!({}));
    let ops = ReplicaOp::from_translog_batch(TranslogTailer::open(crate::shards::copy_of(&primary, 0).shard.path()).unwrap().poll().unwrap(), TERM).unwrap();
    r.apply(&ops, Some(&primary.mapping_json())).unwrap();
    r.update_global_checkpoint(1).unwrap();
    r.flush().unwrap();
    let prefix = r.reusable_prefix().unwrap();
    // Retained operations before negotiation: overwrites and deletes must
    // be replayed, not inferred from equal live-document counts.
    delete(&primary, "delete");
    index(&primary, "keep", json!({"n":3}));
    let mut source = RecoverySource::resume(primary.clone(), 0, TERM + 1, prefix.checkpoint, RecoveryOptions::default()).unwrap();
    assert_eq!(source.info().snapshot_docs, 0);
    assert_eq!(source.info().snapshot_max_seq_no, 1);
    assert_eq!(source.info().translog_ops_on_start, 2);
    assert!(source.next_snapshot_batch().unwrap().is_none());
    let mut target = RecoveryTarget::with_applier(r.clone(), source.info().clone()).unwrap();
    assert_eq!(target.snapshot_done().unwrap(), 1, "source advancement must not fast-forward the target");
    let before = source.next_translog_batch().unwrap();
    target.apply_translog_batch(&before, Some(&primary.mapping_json())).unwrap();
    assert_eq!(r.local_checkpoint(), 3);
    // New writes after negotiation remain in the same tailer and can
    // overlap ordinary replication without double application.
    index(&primary, "late", json!({"n":4}));
    let late = source.next_translog_batch().unwrap();
    r.apply_under(&late, Some(TERM + 1), Some(&primary.mapping_json()), 0).unwrap();
    target.apply_translog_batch(&late, Some(&primary.mapping_json())).unwrap();
    assert_eq!(target.finalize(4).unwrap(), 4);
    assert_eq!(target.state().snapshot_docs_recovered, 0);
    assert!(svc.get("delete").unwrap().is_none());
    assert_identical(&primary, &svc, "replica prefix + concurrent advancement");
}

#[test]
fn snapshot_admission_failure_rewinds_the_entire_unreturned_batch() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = create(&tmp.path().join("primary"), &json!({}));
    for id in ["a", "b", "c"] { index(&primary, id, json!({"n": 1})); }
    let mut source = RecoverySource::start(primary, 0, TERM, RecoveryOptions::default()).unwrap();
    let mut calls = 0;
    let denied = source.next_snapshot_batch_guarded(|_| {
        calls += 1;
        if calls >= 4 { return Err(ReplicationError::IllegalState("injected memory denial".into())); }
        Ok(())
    });
    assert!(denied.is_err());
    assert_eq!(source.docs_sent(), 0);
    let batch = source.next_snapshot_batch_guarded(|_| Ok(())).unwrap().unwrap();
    assert_eq!(batch.len(), 3, "retry must include documents visited before memory denial");
    assert_eq!(source.docs_sent(), 3);
    assert!(source.next_snapshot_batch().unwrap().is_none());
}

#[test]
fn snapshot_many_tiny_documents_budget_denial_rewinds_all_progress() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = create(&tmp.path().join("primary"), &json!({}));
    for id in 0..256 { index(&primary, &id.to_string(), json!({})); }
    let mut source = RecoverySource::start(primary, 0, TERM, RecoveryOptions {
        batch_docs: 256, batch_bytes: usize::MAX, ..RecoveryOptions::default()
    }).unwrap();
    // Discover the first document's decode bound, then allow only a small
    // fixed amount beyond it. Later retained operation arrays cannot fit.
    let mut budget = None;
    let mut attempts = 0;
    let result = source.next_snapshot_batch_guarded(|peak| {
        attempts += 1;
        let limit = *budget.get_or_insert(peak.saturating_add(4096));
        if peak > limit { return Err(ReplicationError::IllegalState("low source budget".into())); }
        Ok(())
    });
    assert!(result.is_err());
    assert!(attempts > 1);
    assert_eq!(source.docs_sent(), 0);
    let retry = source.next_snapshot_batch_guarded(|_| Ok(())).unwrap().unwrap();
    assert_eq!(retry.len(), 256);
    let ids: std::collections::BTreeSet<_> = retry.iter().map(|op| match op {
        ReplicaOp::Index { id, .. } => id.clone(), _ => panic!("snapshot must contain index operations"),
    }).collect();
    assert_eq!(ids.len(), 256, "no omissions or duplicate progress after denial");
}

#[test]
fn snapshot_rehydrated_vector_capacity_is_retained_across_the_batch() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = Arc::new(IndexService::create(
        &tmp.path().join("primary"), "idx",
        settings::normalize_index_settings(&json!({"index.mapping.exclude_source_vectors": true})),
        &json!({"properties": {"v": {"type":"dense_vector", "dims":2, "similarity":"l2_norm"}}}),
        BTreeMap::new(),
    ).unwrap());
    for id in 0..256 { index(&primary, &format!("{id:04}"), json!({"v":[0.5,0.25]})); }
    let opts = RecoveryOptions { batch_docs:256, batch_bytes:usize::MAX, ..RecoveryOptions::default() };
    let mut source = RecoverySource::start(primary.clone(), 0, TERM, opts.clone()).unwrap();
    let mut admitted_peak = 0;
    let batch = source.next_snapshot_batch_guarded(|peak| { admitted_peak = admitted_peak.max(peak); Ok(()) }).unwrap().unwrap();
    let (retained_strings, logical_strings) = batch.iter().fold((0u64, 0u64), |(capacity, len), op| match op {
        ReplicaOp::Index { id, source_json, routing, .. } => (
            capacity + id.capacity() as u64 + source_json.capacity() as u64 + routing.as_ref().map_or(0, |s| s.capacity() as u64),
            len + id.len() as u64 + source_json.len() as u64 + routing.as_ref().map_or(0, |s| s.len() as u64)),
        _ => panic!("snapshot must contain index operations"),
    });
    assert!(retained_strings > logical_strings * 4, "fixture must retain reconstructed output slack");
    assert!(admitted_peak >= retained_strings + batch.capacity() as u64 * std::mem::size_of::<ReplicaOp>() as u64);
    // This budget fits logical bytes but not the accumulated retained buffers.
    let mut limited = RecoverySource::start(primary, 0, TERM, opts).unwrap();
    assert!(limited.next_snapshot_batch_guarded(|peak| {
        if peak > retained_strings / 2 { Err(ReplicationError::IllegalState("retained capacity budget".into())) } else { Ok(()) }
    }).is_err());
    assert_eq!(limited.docs_sent(), 0);
    assert_eq!(limited.next_snapshot_batch_guarded(|_| Ok(())).unwrap().unwrap().len(), 256);
}

#[test]
fn recovery_target_retains_archived_settings_without_enabling_public_writes() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = create(&tmp.path().join("primary"), &json!({}));
    primary.apply_settings_delta([("archived.index.max_regex_length".into(), json!("broken"))].into()).unwrap();
    let source = RecoverySource::start(primary, 0, TERM, RecoveryOptions::default()).unwrap();
    let target = RecoveryTarget::create_index(&tmp.path().join("target"), source.info()).unwrap();
    assert_eq!(target.meta().settings["archived.index.max_regex_length"], "broken");
    assert!(target.put_settings([("archived.index.max_regex_length".into(), json!("changed"))].into(), false).is_err());
}

/// B6: a recovery translog batch that overwrites a document the snapshot
/// installed, arriving before any live request taught the target an MSU
/// covering it, must replace that copy: the target never keeps two live
/// copies of one id.
#[test]
fn recovery_translog_overwrite_of_a_snapshot_document_keeps_one_live_copy() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = create(&tmp.path().join("primary"), &json!({}));
    index(&primary, "x", json!({"n": 1}));
    index(&primary, "y", json!({"n": 2}));
    let mut source = RecoverySource::start(primary.clone(), 0, TERM, RecoveryOptions::default()).unwrap();
    let info = source.info().clone();
    let tsvc = RecoveryTarget::create_index(&tmp.path().join("t"), &info).unwrap();
    let mut target = RecoveryTarget::new(tsvc.clone(), info.clone()).unwrap();
    while let Some(batch) = source.next_snapshot_batch().unwrap() {
        target.apply_snapshot_batch(&batch).unwrap();
    }
    target.snapshot_done().unwrap();
    // Refreshed: the version map no longer remembers the snapshot's ids.
    tsvc.refresh().unwrap();
    // A live request reaches the registered target with the primary's MSU
    // as of now (no update or delete yet).
    let msu = crate::shards::copy_of(&primary, 0).shard.max_seq_no_of_updates();
    let mut live = req(TERM, -1, vec![]);
    live.max_seq_no_of_updates = msu;
    target.applier().handle(&live).unwrap();
    // x is overwritten; the recovery's translog batch carries it first.
    index(&primary, "x", json!({"n": 3}));
    let batch = source.next_translog_batch().unwrap();
    assert!(batch.iter().any(|op| matches!(op, ReplicaOp::Index { id, .. } if id == "x")));
    target.apply_translog_batch(&batch, Some(&primary.mapping_json())).unwrap();
    let max = crate::shards::copy_of(&primary, 0).shard.max_seq_no();
    assert_eq!(target.finalize(max).unwrap(), max);
    assert_identical(&primary, &tsvc, "recovery overwrite of a snapshot document");
}

/// B6: raising the target's MSU over a recovery batch must not
/// switch the first-copy fast path on (it is not the primary's MSU). A live
/// update above the batch, applied to a target that never received the
/// primary's MSU, must replace the snapshot's copy, not sit next to it
/// (the concurrent recovery test saw a duplicate live document).
#[test]
fn recovery_batch_msu_does_not_turn_live_updates_into_first_copies() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = create(&tmp.path().join("primary"), &json!({}));
    index(&primary, "x", json!({"n": 1}));
    index(&primary, "y", json!({"n": 2}));
    let mut source = RecoverySource::start(primary.clone(), 0, TERM, RecoveryOptions::default()).unwrap();
    let info = source.info().clone();
    let tsvc = RecoveryTarget::create_index(&tmp.path().join("t"), &info).unwrap();
    let mut target = RecoveryTarget::new(tsvc.clone(), info.clone()).unwrap();
    while let Some(batch) = source.next_snapshot_batch().unwrap() {
        target.apply_snapshot_batch(&batch).unwrap();
    }
    target.snapshot_done().unwrap();
    tsvc.refresh().unwrap();
    // A recovery translog batch below the live ops.
    index(&primary, "z", json!({"n": 3}));
    let batch = source.next_translog_batch().unwrap();
    target.apply_translog_batch(&batch, Some(&primary.mapping_json())).unwrap();
    // x is updated; the live op (no MSU attached, as the in-process pump
    // ships it) is above the batch.
    index(&primary, "x", json!({"n": 4}));
    let live = source.next_translog_batch().unwrap();
    assert!(live.iter().any(|op| matches!(op, ReplicaOp::Index { id, .. } if id == "x")));
    target.applier().apply(&live, Some(&primary.mapping_json())).unwrap();
    let max = crate::shards::copy_of(&primary, 0).shard.max_seq_no();
    assert_eq!(target.finalize(max).unwrap(), max);
    assert_identical(&primary, &tsvc, "live update after a recovery batch");
}

/// B1: after a failover to term 2, a recovery ships the documents term 1
/// wrote with term 1. The target fences on the recovery's term, so the
/// recovery completes, and every document keeps its own term.
#[test]
fn recovery_after_a_failover_ships_older_terms_and_completes() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = create(&tmp.path().join("primary"), &json!({}));
    index(&primary, "old", json!({"n": 1}));
    index(&primary, "older", json!({"n": 2}));
    crate::shards::copy_of(&primary, 0).shard.set_primary_term(2).unwrap();
    index(&primary, "new", json!({"n": 3}));
    let mut source = RecoverySource::start(primary.clone(), 0, 2, RecoveryOptions::default()).unwrap();
    let info = source.info().clone();
    let tsvc = RecoveryTarget::create_index(&tmp.path().join("t"), &info).unwrap();
    let mut target = RecoveryTarget::new(tsvc.clone(), info).unwrap();
    while let Some(batch) = source.next_snapshot_batch().unwrap() {
        target.apply_snapshot_batch(&batch).expect("a term-1 batch under a term-2 recovery is not stale");
    }
    target.snapshot_done().unwrap();
    // Translog ops written in term 1 only (a new op of the old term range).
    let batch = source.next_translog_batch().unwrap();
    target.apply_translog_batch(&batch, Some(&primary.mapping_json())).unwrap();
    let max = crate::shards::copy_of(&primary, 0).shard.max_seq_no();
    assert_eq!(target.finalize(max).unwrap(), max);
    let t = crate::shards::copy_of(&tsvc, 0);
    assert_eq!(t.shard.doc_primary_term("old"), Some(1));
    assert_eq!(t.shard.doc_primary_term("new"), Some(2));
    // A live request of an older sender is still fenced.
    let stale = vec![ReplicaOp::Index { id: "late".into(), source_json: "{}".into(), routing: None, seq_no: max + 1, version: 1, primary_term: 1 }];
    assert!(matches!(target.applier().apply(&stale, None), Err(ReplicationError::StalePrimaryTerm { .. })));
}

#[test]
fn replica_prefix_recovery_requires_retained_history_and_preserves_snapshot_fallback() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = create(&tmp.path().join("primary"), &json!({}));
    index(&primary, "deleted", json!({"n":1}));
    index(&primary, "kept", json!({"n":2}));
    delete(&primary, "deleted");
    crate::shards::copy_of(&primary, 0).shard.flush().unwrap();
    assert!(matches!(RecoverySource::resume(primary.clone(), 0, TERM, 1, RecoveryOptions::default()), Err(ReplicationError::OpsTrimmed { .. })));
    assert!(matches!(RecoverySource::resume(primary.clone(), 0, TERM, 3, RecoveryOptions::default()), Err(ReplicationError::IllegalState(_))));
    let mut source = RecoverySource::start(primary.clone(), 0, TERM, RecoveryOptions::default()).unwrap();
    let svc = RecoveryTarget::create_index(&tmp.path().join("fresh-target"), source.info()).unwrap();
    let mut target = RecoveryTarget::new(svc.clone(), source.info().clone()).unwrap();
    while let Some(batch) = source.next_snapshot_batch().unwrap() { target.apply_snapshot_batch(&batch).unwrap(); }
    target.snapshot_done().unwrap();
    target.finalize(2).unwrap();
    assert!(svc.get("deleted").unwrap().is_none());
    assert_identical(&primary, &svc, "trimmed-history snapshot fallback");
}

#[test]
fn replica_prefix_recovery_keeps_term_fences() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = create(&tmp.path().join("primary"), &json!({}));
    index(&primary, "a", json!({}));
    let (svc, r) = replica(&tmp.path().join("replica"), &json!({}));
    let ops = ReplicaOp::from_translog_batch(TranslogTailer::open(crate::shards::copy_of(&primary, 0).shard.path()).unwrap().poll().unwrap(), TERM).unwrap();
    r.apply(&ops, Some(&primary.mapping_json())).unwrap();
    r.update_global_checkpoint(0).unwrap(); r.flush().unwrap();
    let source = RecoverySource::resume(primary, 0, TERM + 1, 0, RecoveryOptions::default()).unwrap();
    let mut target = RecoveryTarget::with_applier(r.clone(), source.info().clone()).unwrap();
    target.snapshot_done().unwrap();
    r.bump_term(TERM + 2, 0).unwrap();
    assert!(matches!(target.finalize(0), Err(ReplicationError::RecoveryAborted { .. })));
    assert!(matches!(RecoveryTarget::with_applier(r, source.info().clone()), Err(ReplicationError::RecoveryAborted { .. })));
    assert!(svc.get("a").unwrap().is_some(), "a rejected old recovery must preserve the store");
}

#[test]
fn new_replica_batches_preserve_checkpoints_retries_and_recovery() {
    let tmp = tempfile::tempdir().unwrap();
    let mapping = json!({"properties":{"n":{"type":"long"}}});
    let path = tmp.path().join("batch");
    let (batched, batch) = replica(&path,&mapping);
    let (serial, single) = replica(&tmp.path().join("serial"),&mapping);
    let ops: Vec<_> = (0..600).map(|i| ReplicaOp::Index {
        id:i.to_string(),source_json:json!({"n":i}).to_string(),routing:Some("route".into()),
        seq_no:i,version:3,primary_term:TERM,
    }).collect();
    assert_eq!(batch.apply(&ops,None).unwrap(),599);
    for op in &ops { single.apply(std::slice::from_ref(op),None).unwrap(); }
    assert_eq!(docs(&batched),docs(&serial));
    assert_eq!(batch.apply(&ops,None).unwrap(),599);
    drop(batch); drop(batched);
    let recovered = Arc::new(IndexService::open(&path).unwrap());
    let batch = ReplicaApplier::open(recovered.clone(),0).unwrap();
    assert_eq!(docs(&recovered),docs(&serial));
    assert_eq!(batch.apply(&ops,None).unwrap(),599);
    batch.flush().unwrap();
    assert_eq!(docs(&recovered),docs(&serial));
}

#[test]
fn prepared_replica_chunks_preserve_order_versions_and_recovery() {
    let tmp = tempfile::tempdir().unwrap();
    let mapping = json!({"properties":{"n":{"type":"long"}}});
    let path = tmp.path().join("batch");
    let (batched, batch) = replica(&path, &mapping);
    let (serial, single) = replica(&tmp.path().join("serial"), &mapping);
    let ops: Vec<_> = (0..600).map(|i| ReplicaOp::Index {
        id:(i%60).to_string(), source_json:json!({"n":i}).to_string(), routing:None,
        seq_no:i, version:(i/60+1) as u64, primary_term:TERM,
    }).collect();
    assert_eq!(batch.apply(&ops, None).unwrap(), 599);
    for op in &ops { single.apply(std::slice::from_ref(op), None).unwrap(); }
    assert_eq!(docs(&batched), docs(&serial));
    assert_eq!(batch.apply(&ops, None).unwrap(), 599);
    drop(batch); drop(batched);
    let recovered = Arc::new(IndexService::open(&path).unwrap());
    let recovered_batch = ReplicaApplier::open(recovered.clone(), 0).unwrap();
    assert_eq!(recovered_batch.apply(&ops, None).unwrap(), 599);
    assert_eq!(docs(&recovered), docs(&serial));
}

#[test]
fn speculative_replica_errors_do_not_change_partial_apply_order() {
    let tmp = tempfile::tempdir().unwrap();
    let (svc, replica) = replica(&tmp.path().join("r"), &json!({"properties":{"n":{"type":"long"}}}));
    let ops: Vec<_> = (0..64).map(|i| ReplicaOp::Index {
        id:i.to_string(), source_json:match i {1=>json!({"n":"invalid"}),2=>json!({"future_field":1}),_=>json!({"n":i})}.to_string(),
        routing:None, seq_no:i, version:1, primary_term:TERM,
    }).collect();
    assert!(replica.apply(&ops, None).is_err());
    assert_eq!(docs(&svc).len(), 1);
    assert!(svc.get("0").unwrap().is_some());
    assert!(svc.lookup().get("future_field").is_none());
}

#[test]
fn stale_invalid_source_remains_ignored_in_prepared_replica_chunk() {
    let tmp = tempfile::tempdir().unwrap();
    let (svc, replica) = replica(&tmp.path().join("r"), &json!({"properties":{"n":{"type":"long"}}}));
    replica.apply(&[ReplicaOp::Index {id:"0".into(),source_json:json!({"n":100}).to_string(),routing:None,seq_no:100,version:2,primary_term:TERM}],None).unwrap();
    let ops: Vec<_> = (0..64).map(|i| ReplicaOp::Index {
        id:i.to_string(), source_json:if i==0 {"{invalid".into()} else {json!({"n":i}).to_string()},
        routing:None,seq_no:i,version:1,primary_term:TERM,
    }).collect();
    assert_eq!(replica.apply(&ops,None).unwrap(),63);
    assert_eq!(svc.get("0").unwrap().unwrap().seq_no,100);
    assert_eq!(docs(&svc).len(),64);
}

fn create(path: &Path, mapping: &Value) -> Arc<IndexService> {
    Arc::new(IndexService::create(path, "idx", settings::normalize_index_settings(&json!({})), mapping, BTreeMap::new()).unwrap())
}

fn index(svc: &IndexService, id: &str, src: Value) {
    let r = svc.write(vec![WriteReq::Index { id: Some(id.into()), source: src.to_string(), routing: None, cond: WriteCond::default() }], false);
    r[0].as_ref().unwrap();
}

fn delete(svc: &IndexService, id: &str) {
    let r = svc.write(vec![WriteReq::Delete { id: id.into(), routing: None, cond: WriteCond::default() }], false);
    r[0].as_ref().unwrap();
}

fn update(svc: &IndexService, id: &str, doc: Value) {
    let _ = svc.update(id, &json!({ "doc": doc, "doc_as_upsert": true }), None, 3, WriteCond::default());
}

type DocState = (u64, i64, String, Option<String>);

/// Every live doc: id → (version, seq_no, source, routing).
fn docs(svc: &IndexService) -> BTreeMap<String, DocState> {
    let copy = crate::shards::copy_of(svc, 0);
    // Pooled memtables (concurrent DWPT-style applies) join the index first.
    copy.shard.checkin_pool(None);
    let core = copy.shard.core.read();
    let mut out = BTreeMap::new();
    // Every write applied so far, refreshed or not.
    for leaf in core.all_leaves() {
        for d in 0..leaf.num_docs() {
            if leaf.is_deleted_latest(d) || !leaf.is_root(d) {
                continue;
            }
            let doc = leaf.get_doc(d).expect("stored document decode").expect("live root document");
            let t = |f: u16| doc.get(f).and_then(|v| v.as_text()).map(str::to_string);
            let id = t(ID_FIELD).unwrap();
            let st = (leaf.i64_value(VERSION_FIELD, d).unwrap() as u64, leaf.i64_value(SEQ_NO_FIELD, d).unwrap(), t(SOURCE_FIELD).unwrap_or_default(), t(ROUTING_FIELD));
            assert!(out.insert(id.clone(), st).is_none(), "duplicate live doc {id}");
        }
    }
    out
}

fn hits(svc: &Arc<IndexService>, q: &Value) -> u64 {
    count(&[Target::new(svc.clone())], Some(q), None).unwrap()
}

fn queries() -> Vec<Value> {
    vec![
        json!({"match_all": {}}),
        json!({"match": {"title": "hello"}}),
        json!({"term": {"tag.keyword": "t1"}}),
        json!({"range": {"n": {"gte": 10}}}),
    ]
}

/// Replica == primary: doc sets, versions, seq_nos, sources, routing,
/// realtime GET and search hit counts.
fn assert_identical(primary: &Arc<IndexService>, replica: &Arc<IndexService>, label: &str) {
    let p = docs(primary);
    let r = docs(replica);
    if p != r {
        let only_p: Vec<_> = p.iter().filter(|(k, v)| r.get(*k) != Some(*v)).take(5).collect();
        let only_r: Vec<_> = r.iter().filter(|(k, v)| p.get(*k) != Some(*v)).take(5).collect();
        panic!("{label}: doc sets differ ({} vs {} docs); primary-only/different {only_p:?}; replica-only/different {only_r:?}", p.len(), r.len());
    }
    for (id, (v, s, src, routing)) in &p {
        let g = replica.get(id).unwrap().unwrap_or_else(|| panic!("{label}: realtime get {id} missing"));
        assert_eq!((g.version, g.seq_no, g.source.as_deref(), g.routing.as_ref()), (*v, *s, Some(src.as_str()), routing.as_ref()), "{label}: get {id}");
    }
    // Searches compare after a refresh of both copies.
    primary.refresh().unwrap();
    replica.refresh().unwrap();
    for q in queries() {
        if primary.lookup().get("title").is_none() && q.get("match").is_some() {
            continue;
        }
        assert_eq!(hits(primary, &q), hits(replica, &q), "{label}: hit count for {q}");
    }
}

/// The primary's replication pump (what the node crate runs per shard).
struct Pump {
    primary: Arc<IndexService>,
    tailer: TranslogTailer,
    tracker: Arc<Mutex<ReplicationTracker>>,
    primary_cp: LocalCheckpointTracker,
    targets: BTreeMap<String, Arc<ReplicaApplier>>,
    last_shipped: i64,
    send_mapping: bool,
}

fn ship(
    primary: &IndexService,
    tracker: &Mutex<ReplicationTracker>,
    primary_cp: &mut LocalCheckpointTracker,
    targets: &BTreeMap<String, Arc<ReplicaApplier>>,
    last_shipped: &mut i64,
    send_mapping: bool,
    ops: elasticsearch_engine::translog::AdmittedBatch<TranslogOp>,
) {
    if ops.is_empty() {
        return;
    }
    for op in &ops {
        primary_cp.mark_processed(op.seq_no() as i64);
        *last_shipped = (*last_shipped).max(op.seq_no() as i64);
    }
    tracker.lock().update_local_checkpoint("p", primary_cp.checkpoint());
    let rops = ReplicaOp::from_translog_batch(ops, TERM).unwrap();
    let mapping = primary.meta().mappings.to_json();
    for (aid, app) in targets {
        let cp = app.apply(&rops, send_mapping.then_some(&mapping)).unwrap();
        tracker.lock().update_local_checkpoint(aid, cp);
    }
}

impl Pump {
    fn new(primary: Arc<IndexService>, replicas: &[(&str, Arc<ReplicaApplier>)]) -> Pump {
        let mut tracker = ReplicationTracker::new("p", TERM, -1);
        let ids: Vec<&str> = std::iter::once("p").chain(replicas.iter().map(|(a, _)| *a)).collect();
        tracker.update_from_cluster_state(ids.iter().copied(), []);
        for (a, _) in replicas {
            tracker.update_local_checkpoint(a, -1);
        }
        Pump {
            tailer: TranslogTailer::open(crate::shards::copy_of(&primary, 0).shard.path()).unwrap(),
            primary,
            tracker: Arc::new(Mutex::new(tracker)),
            primary_cp: LocalCheckpointTracker::default(),
            targets: replicas.iter().map(|(a, r)| (a.to_string(), r.clone())).collect(),
            last_shipped: -1,
            send_mapping: true,
        }
    }

    fn pump(&mut self) -> usize {
        let ops = self.tailer.poll().unwrap();
        let n = ops.len();
        ship(&self.primary, &self.tracker, &mut self.primary_cp, &self.targets, &mut self.last_shipped, self.send_mapping, ops);
        n
    }

    /// Start shipping live ops to a recovering copy; returns the last
    /// seq_no shipped before it joined (the translog phase must cover it).
    fn register(&mut self, aid: &str, app: Arc<ReplicaApplier>) -> i64 {
        self.targets.insert(aid.to_string(), app);
        self.last_shipped
    }

    fn safe_flush(&mut self, gate: &WriteGate) -> FlushOutcome {
        let Pump { primary, tailer, tracker, primary_cp, targets, last_shipped, send_mapping } = self;
        let p = primary.clone();
        let mut f = |ops: elasticsearch_engine::translog::AdmittedBatch<TranslogOp>| -> Result<()> {
            ship(&p, tracker, primary_cp, targets, last_shipped, *send_mapping, ops);
            Ok(())
        };
        safe_flush(&crate::shards::copy_of(primary, 0), gate, tailer, tracker, None, &mut f).unwrap()
    }
}

fn replica(path: &Path, mapping: &Value) -> (Arc<IndexService>, Arc<ReplicaApplier>) {
    let svc = create(path, mapping);
    let app = Arc::new(ReplicaApplier::open(svc.clone(), 0).unwrap());
    (svc, app)
}

fn doc(rng: &mut impl Rng, i: usize) -> Value {
    json!({"title": format!("hello world {i}"), "n": rng.gen_range(0..50), "tag": format!("t{}", rng.gen_range(0..4))})
}

#[test]
fn primary_and_two_replicas_mixed_ops_out_of_order_and_duplicated() {
    let tmp = tempfile::tempdir().unwrap();
    let mapping = json!({"properties": {"title": {"type": "text"}}});
    let primary = create(&tmp.path().join("p"), &mapping);
    let (r1svc, r1) = replica(&tmp.path().join("r1"), &mapping);
    let (r2svc, r2) = replica(&tmp.path().join("r2"), &mapping);
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    let mut tailer = TranslogTailer::open(crate::shards::copy_of(&primary, 0).shard.path()).unwrap();
    let mut tracker = ReplicationTracker::new("p", TERM, -1);
    tracker.update_from_cluster_state(["p", "r1", "r2"], []);
    let mut primary_cp = LocalCheckpointTracker::default();
    let mut r2_backlog: Vec<Vec<ReplicaOp>> = Vec::new();
    for round in 0..12 {
        // Mixed writes through the engine's public write APIs.
        for i in 0..20 {
            let id = format!("d{}", rng.gen_range(0..60));
            match rng.gen_range(0..10) {
                0..=4 => index(&primary, &id, doc(&mut rng, i)),
                5..=6 => update(&primary, &id, json!({"n": rng.gen_range(0..50), "tag": "t1"})),
                7..=8 => {
                    let _ = primary.write(vec![WriteReq::Delete { id: id.clone(), routing: None, cond: WriteCond::default() }], false);
                }
                _ => {
                    // bulk with routing and an auto id
                    let reqs = vec![
                        WriteReq::Index { id: Some(format!("b{round}-{i}")), source: doc(&mut rng, i).to_string(), routing: Some("r7".into()), cond: WriteCond::default() },
                        WriteReq::Index { id: None, source: doc(&mut rng, i).to_string(), routing: None, cond: WriteCond::default() },
                        WriteReq::Delete { id: format!("d{}", rng.gen_range(0..60)), routing: None, cond: WriteCond::default() },
                    ];
                    let _ = primary.write(reqs, true);
                }
            }
        }
        if round == 5 {
            // A primary flush between rounds (all copies are caught up).
            crate::shards::copy_of(&primary, 0).shard.flush().unwrap();
        }
        let ops = ReplicaOp::from_translog_batch(tailer.poll().unwrap(), TERM).unwrap();
        for op in &ops {
            primary_cp.mark_processed(op.seq_no());
        }
        tracker.update_local_checkpoint("p", primary_cp.checkpoint());
        let mapping = primary.meta().mappings.to_json();
        // r1: in order, split into small batches.
        for chunk in ops.chunks(7) {
            let cp = r1.apply(chunk, Some(&mapping)).unwrap();
            tracker.update_local_checkpoint("r1", cp);
        }
        // r2: shuffled small batches, some duplicated, delivered late.
        let mut chunks: Vec<Vec<ReplicaOp>> = ops.chunks(5).map(<[ReplicaOp]>::to_vec).collect();
        let dups: Vec<Vec<ReplicaOp>> = chunks.iter().filter(|_| rng.gen_bool(0.3)).cloned().collect();
        chunks.extend(dups);
        r2_backlog.extend(chunks);
        r2_backlog.shuffle(&mut rng);
        let deliver = r2_backlog.len() / 2;
        for b in r2_backlog.drain(..deliver) {
            let cp = r2.apply(&b, Some(&mapping)).unwrap();
            tracker.update_local_checkpoint("r2", cp);
        }
    }
    for b in std::mem::take(&mut r2_backlog) {
        let cp = r2.apply(&b, None).unwrap();
        tracker.update_local_checkpoint("r2", cp);
    }
    let max = crate::shards::copy_of(&primary, 0).shard.max_seq_no();
    assert_eq!(primary_cp.checkpoint(), max);
    assert_eq!(r1.local_checkpoint(), max);
    assert_eq!(r2.local_checkpoint(), max);
    assert_eq!(tracker.global_checkpoint(), max);
    assert!(r2.stats().ops_duplicate > 0);
    assert!(r2.stats().ops_stale > 0, "out-of-order delivery produced stale ops: {:?}", r2.stats());
    assert!(docs(&primary).len() > 30);
    assert_identical(&primary, &r1svc, "r1");
    assert_identical(&primary, &r2svc, "r2");
    // Routing survived.
    assert!(docs(&r1svc).values().any(|d| d.3.as_deref() == Some("r7")));
}

#[test]
fn replica_crash_mid_stream_recovers_from_wal() {
    let tmp = tempfile::tempdir().unwrap();
    let mapping = json!({});
    let primary = create(&tmp.path().join("p"), &mapping);
    let rpath = tmp.path().join("r");
    let (mut rsvc, mut r) = replica(&rpath, &mapping);
    let mut rng = rand::rngs::StdRng::seed_from_u64(9);
    let mut tailer = TranslogTailer::open(crate::shards::copy_of(&primary, 0).shard.path()).unwrap();
    let mut all_ops = Vec::new();
    for round in 0..6 {
        for i in 0..40 {
            let id = format!("d{}", rng.gen_range(0..50));
            if rng.gen_bool(0.25) {
                let _ = primary.write(vec![WriteReq::Delete { id, routing: None, cond: WriteCond::default() }], false);
            } else {
                index(&primary, &id, doc(&mut rng, i));
            }
        }
        let ops = ReplicaOp::from_translog_batch(tailer.poll().unwrap(), TERM).unwrap();
        all_ops.extend(ops.iter().cloned());
        // Deliver out of order: second half first, leaving a gap.
        let (a, b) = ops.split_at(ops.len() / 2);
        r.apply(b, None).unwrap();
        if round == 2 {
            // Flush with a gap pending: the WAL keeps everything above the
            // checkpoint captured before the flush.
            let cp = r.flush().unwrap();
            assert!(cp < r.max_seq_no());
        }
        if round == 1 || round == 3 || round == 5 {
            // Crash: drop the engine without flushing, reopen, replay WAL.
            let before = (r.local_checkpoint(), r.max_seq_no());
            drop(r);
            drop(rsvc);
            rsvc = Arc::new(IndexService::open(&rpath).unwrap());
            r = Arc::new(ReplicaApplier::open(rsvc.clone(), 0).unwrap());
            assert_eq!((r.local_checkpoint(), r.max_seq_no()), before, "checkpoints survive the crash");
            assert!(r.stats().wal_ops_replayed > 0 || round == 1);
        }
        r.apply(a, None).unwrap();
        // Redelivery of everything after the crash is harmless.
        r.apply(&ops, None).unwrap();
        assert_eq!(r.local_checkpoint(), crate::shards::copy_of(&primary, 0).shard.max_seq_no());
        assert_identical(&primary, &rsvc, &format!("round {round}"));
    }
    // Replaying the full history again changes nothing.
    r.apply(&all_ops, None).unwrap();
    assert_identical(&primary, &rsvc, "full redelivery");
    // Flush trims the WAL (below min(checkpoint, global checkpoint)).
    r.update_global_checkpoint(crate::shards::copy_of(&primary, 0).shard.max_seq_no()).unwrap();
    r.flush().unwrap();
    assert!(r.stats().wal_generations <= 2, "{:?}", r.stats());
    drop(r);
    drop(rsvc);
    let rsvc = Arc::new(IndexService::open(&rpath).unwrap());
    let r = ReplicaApplier::open(rsvc.clone(), 0).unwrap();
    assert_eq!(r.local_checkpoint(), crate::shards::copy_of(&primary, 0).shard.max_seq_no());
    assert_identical(&primary, &rsvc, "after flush + reopen");
}

/// Resync's ordered apply and ordinary transport apply must publish identical
/// engine checkpoints, including stale document operations and explicit no-ops.
#[test]
fn replica_stale_ops_and_noops_remain_file_recoverable_after_promotion() {
    for transport in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        let (svc, r) = replica(&tmp.path().join("r"), &json!({}));
        let ix = |seq| ReplicaOp::Index { id: "same".into(), source_json: "{}".into(), routing: None,
            seq_no: seq, version: seq as u64 + 1, primary_term: TERM };
        let apply = |ops: Vec<ReplicaOp>| {
            if transport { r.handle(&req(TERM, -1, ops)).map(|v| v.local_checkpoint) }
            else { r.apply(&ops, None) }
        };
        apply(vec![ix(2)]).unwrap();
        apply(vec![ix(1)]).unwrap(); // stale index must still mark seq_no 1
        apply(vec![ReplicaOp::NoOp { seq_no: 0, primary_term: TERM, reason: "real failed operation".into() }]).unwrap();
        apply(vec![ix(4)]).unwrap();
        apply(vec![ReplicaOp::Delete { id: "same".into(), seq_no: 3, version: 4, primary_term: TERM }]).unwrap();
        let copy = crate::shards::copy_of(&svc, 0);
        assert_eq!(r.local_checkpoint(), 4);
        assert_eq!(copy.shard.local_checkpoint(), 4, "replica tracker must not outrun engine history");
        assert_eq!(svc.get("same").unwrap().unwrap().seq_no, 4);
        r.update_global_checkpoint(4).unwrap();
        r.flush().unwrap();
        let promoted = r.promote_to_primary(TERM + 1).unwrap();
        assert_eq!(promoted.max_seq_no, 4);
        let files = copy.snapshot_files(false).unwrap();
        assert_eq!(files.manifest.local_checkpoint, 4);
        let mut source = RecoverySource::start_from(svc.clone(), 0, TERM + 1, RecoveryOptions::default(), files.manifest.local_checkpoint)
            .expect("full file recovery must not request trimmed history below a completed no-op/stale op");
        assert!(source.next_translog_batch().unwrap().is_empty());
    }
}

#[test]
fn tombstones_reject_stale_index_ops_across_flush_and_crash() {
    let tmp = tempfile::tempdir().unwrap();
    let rpath = tmp.path().join("r");
    let (rsvc, r) = replica(&rpath, &json!({}));
    let ix = |id: &str, seq: i64, v: u64| ReplicaOp::Index { id: id.into(), source_json: format!("{{\"seq\":{seq}}}"), routing: None, seq_no: seq, version: v, primary_term: TERM };
    let del = |id: &str, seq: i64, v: u64| ReplicaOp::Delete { id: id.into(), seq_no: seq, version: v, primary_term: TERM };
    assert_eq!(r.apply(&[ix("a", 0, 1), ix("b", 1, 1), ix("c", 2, 1)], None).unwrap(), 2);
    // Delete of x at 5 arrives before its index at 3 (gap at 3, 4).
    r.apply(&[del("x", 5, 2)], None).unwrap();
    r.apply(&[ix("x", 3, 1)], None).unwrap();
    assert!(rsvc.get("x").unwrap().is_none(), "stale index after delete must not resurrect");
    assert_eq!(r.local_checkpoint(), 3);
    // Delete older than the live doc is stale too.
    r.apply(&[ix("y", 7, 3)], None).unwrap();
    r.apply(&[del("y", 6, 2)], None).unwrap();
    assert_eq!(rsvc.get("y").unwrap().unwrap().seq_no, 7);
    // The engine forgets deletes on flush; our tombstone (5 > checkpoint 3) stays.
    r.flush().unwrap();
    r.apply(&[ix("x", 4, 1)], None).unwrap();
    assert!(rsvc.get("x").unwrap().is_none());
    assert_eq!(r.local_checkpoint(), 7);
    // Crash before the next flush: tombstones are rebuilt from the WAL.
    r.apply(&[del("z", 10, 2)], None).unwrap();
    drop(r);
    drop(rsvc);
    let rsvc = Arc::new(IndexService::open(&rpath).unwrap());
    let r = ReplicaApplier::open(rsvc.clone(), 0).unwrap();
    assert_eq!(r.local_checkpoint(), 7);
    r.apply(&[ix("z", 9, 1)], None).unwrap();
    assert!(rsvc.get("z").unwrap().is_none());
    assert_eq!(r.stats().tombstones, 1);
    // Re-index after delete with a newer seq_no works; NoOp fills gap 8.
    assert_eq!(r.apply(&[ix("x", 11, 3), ReplicaOp::NoOp { seq_no: 8, primary_term: TERM, reason: "gap".into() }], None).unwrap(), 11);
    let g = rsvc.get("x").unwrap().unwrap();
    assert_eq!((g.version, g.seq_no), (3, 11));
    assert_eq!(r.stats().tombstones, 0, "tombstones at/below the checkpoint are pruned");
    // Stale primary term is rejected; newer is adopted (everything up to
    // the gcp is known in sync, so no resync is needed).
    r.update_global_checkpoint(11).unwrap();
    r.apply(&[ReplicaOp::NoOp { seq_no: 12, primary_term: 2, reason: "x".into() }], None).unwrap();
    let e = r.apply(&[ix("q", 13, 1)], None).unwrap_err();
    assert!(matches!(e, ReplicationError::StalePrimaryTerm { ours: 2, theirs: 1 }));
    drop(r);
    let r = ReplicaApplier::open(rsvc.clone(), 0).unwrap();
    assert_eq!(r.primary_term(), 2, "term is durable");
    let resp = r
        .handle(&ReplicationRequest::try_new("idx", 0, "r", 2, 12, 11, None,
            super::ops::admitted_ops_for_test(vec![ReplicaOp::Index { id: "q".into(), source_json: "{}".into(), routing: None, seq_no: 13, version: 1, primary_term: 2 }]), 0).unwrap())
        .unwrap();
    assert_eq!(resp, ReplicationResponse { local_checkpoint: 13, max_seq_no: 13, applied: 1 });
    assert_eq!(r.global_checkpoint(), 12);
}

#[test]
fn dynamic_mapping_updates_arrive_via_ops() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = create(&tmp.path().join("p"), &json!({}));
    let rpath = tmp.path().join("r");
    let (rsvc, r) = replica(&rpath, &json!({}));
    let (r2svc, r2) = replica(&tmp.path().join("r2"), &json!({}));
    let mut pump = Pump::new(primary.clone(), &[("r", r.clone())]);
    pump.send_mapping = false; // r learns fields only by parsing ops
    index(&primary, "1", json!({"title": "hello world", "n": 12, "obj": {"inner": "deep value"}, "when": "2024-01-02"}));
    index(&primary, "2", json!({"title": "hello again", "n": 3, "tag": "t1"}));
    pump.pump();
    let pm = primary.meta().mappings.to_json();
    assert_eq!(rsvc.meta().mappings.to_json(), pm, "replica derived the same dynamic mapping");
    assert!(rsvc.lookup().get("obj.inner").is_some());
    assert_identical(&primary, &rsvc, "dynamic");
    assert_eq!(hits(&rsvc, &json!({"range": {"when": {"gte": "2024-01-01"}}})), 1);
    // Explicit put-mapping on the primary reaches a replica through the
    // mapping shipped with the batch, before the ops that use it.
    primary.put_mapping(&json!({"properties": {"code": {"type": "keyword"}, "score": {"type": "float"}}})).unwrap();
    index(&primary, "3", json!({"code": "ABC-1", "score": 1.5}));
    let ops = ReplicaOp::from_translog_batch(TranslogTailer::open(crate::shards::copy_of(&primary, 0).shard.path()).unwrap().poll().unwrap(), TERM).unwrap();
    r2.apply(&ops, Some(&primary.meta().mappings.to_json())).unwrap();
    assert_eq!(r2svc.meta().mappings.to_json(), primary.meta().mappings.to_json());
    assert_eq!(r2svc.lookup().get("code").unwrap().ty.name(), "keyword");
    r2svc.refresh().unwrap();
    assert_eq!(hits(&r2svc, &json!({"term": {"code": "ABC-1"}})), 1);
    assert_identical(&primary, &r2svc, "explicit mapping");
    // Mapping updates are persisted: survive a crash of the replica.
    drop(pump);
    drop(r);
    drop(rsvc);
    let rsvc = Arc::new(IndexService::open(&rpath).unwrap());
    let _r = ReplicaApplier::open(rsvc.clone(), 0).unwrap();
    assert!(rsvc.lookup().get("obj.inner").is_some());
    assert_eq!(hits(&rsvc, &json!({"match": {"obj.inner": "deep"}})), 1);
}

#[test]
fn snapshot_recovery_jumps_checkpoint_over_delete_gaps() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = create(&tmp.path().join("p"), &json!({}));
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    for i in 0..120 {
        index(&primary, &format!("d{}", i % 70), doc(&mut rng, i));
    }
    for i in 0..20 {
        delete(&primary, &format!("d{i}"));
    }
    crate::shards::copy_of(&primary, 0).shard.flush().unwrap();
    for i in 0..15 {
        update(&primary, &format!("d{}", 30 + i), json!({"n": 99}));
    }
    delete(&primary, "d69");
    let max = crate::shards::copy_of(&primary, 0).shard.max_seq_no();
    let mut source = RecoverySource::start(primary.clone(), 0, TERM, RecoveryOptions { batch_docs: 13, ..Default::default() }).unwrap();
    let info = source.info().clone();
    assert_eq!(info.snapshot_max_seq_no, max);
    assert_eq!(info.snapshot_docs, 49);
    // The start info survives the wire.
    let info: RecoveryStartInfo = bincode::deserialize(&bincode::serialize(&info).unwrap()).unwrap();
    let tsvc = RecoveryTarget::create_index(&tmp.path().join("t"), &info).unwrap();
    let mut target = RecoveryTarget::new(tsvc.clone(), info.clone()).unwrap();
    let mut batches = 0;
    while let Some(b) = source.next_snapshot_batch().unwrap() {
        assert!(b.len() <= 13);
        target.apply_snapshot_batch(&b).unwrap();
        batches += 1;
    }
    assert_eq!(batches, 4);
    assert!(target.applier().local_checkpoint() < max, "gaps from deletes/overwrites");
    assert_eq!(target.snapshot_done().unwrap(), max);
    assert!(source.next_translog_batch().unwrap().is_empty());
    assert_eq!(target.state().stage, RecoveryStage::Translog);
    // More writes during the translog phase.
    index(&primary, "late", json!({"title": "hello late"}));
    delete(&primary, "d50");
    let b = source.next_translog_batch().unwrap();
    assert_eq!(b.len(), 2);
    target.apply_translog_batch(&b, Some(&primary.meta().mappings.to_json())).unwrap();
    assert!(target.finalize(max + 5).is_err(), "cannot finalize below the global checkpoint");
    let cp = target.finalize(max + 2).unwrap();
    assert_eq!(cp, crate::shards::copy_of(&primary, 0).shard.max_seq_no());
    let st = target.state();
    assert_eq!(st.stage, RecoveryStage::Done);
    assert_eq!((st.snapshot_docs_recovered, st.translog_ops_recovered, st.ops_recovered), (49, 2, 51));
    let j = st.to_recovery_json();
    assert_eq!(j["stage"], "DONE");
    assert_eq!(j["type"], "PEER");
    assert_eq!(j["translog"]["recovered"], 51);
    assert_eq!(j["translog"]["percent"], "100.0%");
    assert_eq!(st.cat_row()["stage"], "done");
    assert_identical(&primary, &tsvc, "recovered");
    // The target is now a normal replica: crash + reopen keeps the jump.
    let applier = target.into_applier();
    drop(applier);
    let tpath = tsvc.path().to_path_buf();
    drop(tsvc);
    let tsvc = Arc::new(IndexService::open(&tpath).unwrap());
    let r = ReplicaApplier::open(tsvc.clone(), 0).unwrap();
    assert_eq!(r.local_checkpoint(), crate::shards::copy_of(&primary, 0).shard.max_seq_no());
    assert_identical(&primary, &tsvc, "recovered + reopened");
    // Create-index body for Node::create_index carries settings + mapping.
    let body = RecoveryTarget::create_index_body(&info).unwrap();
    assert!(body["settings"]["index"].get("uuid").is_none());
    assert_eq!(body["settings"]["index"]["number_of_shards"], "1");
}

#[test]
fn recovery_of_new_replica_while_writes_continue() {
    let tmp = tempfile::tempdir().unwrap();
    let mapping = json!({"properties": {"title": {"type": "text"}}});
    let primary = create(&tmp.path().join("p"), &mapping);
    let (r1svc, r1) = replica(&tmp.path().join("r1"), &mapping);
    let gate = Arc::new(WriteGate::new());
    let pump = Arc::new(Mutex::new(Pump::new(primary.clone(), &[("r1", r1.clone())])));
    let mut rng = rand::rngs::StdRng::seed_from_u64(5);
    for i in 0..300 {
        index(&primary, &format!("d{}", i % 200), doc(&mut rng, i));
    }
    for i in 0..40 {
        delete(&primary, &format!("d{}", i * 3));
    }
    pump.lock().pump();
    assert!(matches!(pump.lock().safe_flush(&gate), FlushOutcome::Flushed { .. }));
    for i in 0..50 {
        update(&primary, &format!("d{}", i + 100), json!({"tag": "t1"}));
    }

    // Concurrent writers and the replication pump.
    let stop = Arc::new(AtomicBool::new(false));
    let mut writers = Vec::new();
    for w in 0..3u64 {
        let (primary, gate) = (primary.clone(), gate.clone());
        writers.push(std::thread::spawn(move || {
            let mut rng = rand::rngs::StdRng::seed_from_u64(100 + w);
            for i in 0..400 {
                let _permit = gate.write_permit();
                let id = format!("d{}", rng.gen_range(0..250));
                match rng.gen_range(0..10) {
                    0..=4 => index(&primary, &id, json!({"title": format!("hello w{w} {i}"), "n": i % 50, "tag": format!("t{}", i % 4), format!("extra{}", i % 3): true})),
                    5..=6 => update(&primary, &id, json!({"n": rng.gen_range(0..50)})),
                    _ => {
                        let _ = primary.write(vec![WriteReq::Delete { id, routing: None, cond: WriteCond::default() }], false);
                    }
                }
            }
        }));
    }
    let pumper = {
        let (pump, stop) = (pump.clone(), stop.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                if pump.lock().pump() == 0 {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        })
    };

    // --- recovery of copy "t" ---
    let tracker = pump.lock().tracker.clone();
    tracker.lock().init_tracking("t");
    // While "t" is tracked but behind, flushing is not safe.
    match pump.lock().safe_flush(&gate) {
        FlushOutcome::Skipped { lagging, .. } => assert!(lagging.iter().any(|(a, _)| a == "t")),
        other => panic!("flush must be skipped during recovery: {other:?}"),
    }
    let mut source = RecoverySource::start(primary.clone(), 0, TERM, RecoveryOptions { batch_docs: 25, ..Default::default() }).unwrap();
    let info = source.info().clone();
    let tsvc = RecoveryTarget::create_index(&tmp.path().join("t"), &info).unwrap();
    let mut target = RecoveryTarget::new(tsvc.clone(), info).unwrap();
    let live_from = pump.lock().register("t", target.applier().clone());
    while let Some(b) = source.next_snapshot_batch().unwrap() {
        target.apply_snapshot_batch(&b).unwrap();
    }
    target.snapshot_done().unwrap();
    while source.last_seq_no_sent() < live_from {
        let b = source.next_translog_batch().unwrap();
        target.apply_translog_batch(&b, Some(&primary.meta().mappings.to_json())).unwrap();
    }
    let cp = loop {
        let gcp = tracker.lock().global_checkpoint();
        match target.finalize(gcp) {
            Ok(cp) => break cp,
            Err(ReplicationError::IllegalState(_)) => std::thread::sleep(std::time::Duration::from_millis(2)),
            Err(e) => panic!("{e}"),
        }
    };
    tracker.lock().update_local_checkpoint("t", cp);
    while tracker.lock().mark_in_sync("t") != Ok(true) {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(target.state().stage, RecoveryStage::Done);

    for w in writers {
        w.join().unwrap();
    }
    stop.store(true, Ordering::Relaxed);
    pumper.join().unwrap();
    while pump.lock().pump() > 0 {}

    let max = crate::shards::copy_of(&primary, 0).shard.max_seq_no();
    let t = tracker.lock().clone();
    assert!(t.is_in_sync("t"));
    assert_eq!(t.global_checkpoint(), max);
    assert_identical(&primary, &r1svc, "r1");
    assert_identical(&primary, &tsvc, "recovered t");
    assert_eq!(target.applier().local_checkpoint(), max);
    // Everyone acked: the flush goes through now and later ops still tail.
    assert!(matches!(pump.lock().safe_flush(&gate), FlushOutcome::Flushed { .. }));
    index(&primary, "after", json!({"title": "hello after"}));
    pump.lock().pump();
    assert_identical(&primary, &tsvc, "after flush");
    assert_identical(&primary, &r1svc, "r1 after flush");
}

// ---------------------------------------------------------------------------
// Fencing, failover resync, durability
// ---------------------------------------------------------------------------

fn req(term: u64, gcp: i64, ops: Vec<ReplicaOp>) -> ReplicationRequest {
    ReplicationRequest::try_new("idx", 0, "x", term, gcp, -1, None, super::ops::admitted_ops_for_test(ops), 0).unwrap()
}

/// A replica's engine copy knows the global checkpoint the replication
/// requests (and op-free checkpoint syncs) carried, as ES's
/// `getLastKnownGlobalCheckpoint` (a fleet search's wait on a replica reads
/// it).
#[test]
fn a_replica_publishes_its_known_global_checkpoint_to_its_copy() {
    let tmp = tempfile::tempdir().unwrap();
    let (svc, r) = replica(&tmp.path().join("replica"), &json!({}));
    let copy = crate::shards::copy_of(&svc, 0);
    let op = |seq: i64| ReplicaOp::Index { id: format!("g{seq}"), source_json: "{}".into(), routing: None, seq_no: seq, version: 1, primary_term: TERM };
    r.handle(&req(TERM, -1, vec![op(0), op(1)])).unwrap();
    assert_eq!(copy.global_checkpoint(), -1);
    r.handle(&req(TERM, 1, vec![op(2)])).unwrap();
    assert_eq!(copy.global_checkpoint(), 1, "carried by a batch");
    r.handle(&req(TERM, 2, vec![])).unwrap();
    assert_eq!(copy.global_checkpoint(), 2, "an op-free checkpoint sync");
}

fn tail_all(svc: &IndexService, from: i64, term: u64) -> Vec<ReplicaOp> {
    ReplicaOp::from_translog_batch(TranslogTailer::open_at(crate::shards::copy_of(svc, 0).shard.path(), from).unwrap().poll().unwrap(), term).unwrap().iter().cloned().collect()
}

fn durable_state(svc: &IndexService) -> Value {
    let v: Value = serde_json::from_slice(&std::fs::read(crate::shards::copy_of(svc, 0).shard.path().join("cluster-replica-wal/state.json")).unwrap()).unwrap();
    v.get("state").unwrap_or(&v).clone()
}

#[test]
fn deposed_primary_is_fenced_and_term_is_persisted_before_ack() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = create(&tmp.path().join("p"), &json!({}));
    let rpath = tmp.path().join("r");
    let (rsvc, r) = replica(&rpath, &json!({}));
    for i in 0..5 {
        index(&primary, &format!("d{i}"), json!({"n": i}));
    }
    let ops = tail_all(&primary, 0, 1);
    r.handle(&req(1, 2, ops[..3].to_vec())).unwrap();
    // The master promoted another copy (term 2): the replica learns it.
    r.bump_term(2, 2).unwrap();
    assert_eq!(durable_state(&rsvc)["primary_term"], 2, "term fsynced before bump_term returns");
    // The deposed primary (term 1) keeps sending: never acked.
    let e = r.handle(&req(1, 2, ops[3..].to_vec())).unwrap_err();
    assert!(matches!(e, ReplicationError::StalePrimaryTerm { ours: 2, theirs: 1 }), "{e}");
    assert_eq!(e.es_type(), "illegal_state_exception");
    let e = r.apply(&ops[3..], None).unwrap_err();
    assert!(matches!(e, ReplicationError::StalePrimaryTerm { ours: 2, theirs: 1 }));
    assert!(rsvc.get("d3").unwrap().is_none(), "nothing from the deposed primary was applied");
    // A request with a higher term is persisted before it is acked.
    let resp = r.handle(&req(3, 2, vec![])).unwrap();
    assert_eq!(resp.local_checkpoint, 2);
    assert_eq!(durable_state(&rsvc)["primary_term"], 3);
    // Survives a crash.
    drop(r);
    let r = ReplicaApplier::open(rsvc.clone(), 0).unwrap();
    assert_eq!(r.primary_term(), 3);
    assert!(matches!(r.handle(&req(2, 2, vec![])).unwrap_err(), ReplicationError::StalePrimaryTerm { ours: 3, theirs: 2 }));
}

#[test]
fn global_checkpoint_never_goes_backwards_across_restarts() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = create(&tmp.path().join("p"), &json!({}));
    let (rsvc, r) = replica(&tmp.path().join("r"), &json!({}));
    for i in 0..6 {
        index(&primary, &format!("d{i}"), json!({"n": i}));
    }
    let ops = tail_all(&primary, 0, 1);
    r.handle(&req(1, 3, ops.clone())).unwrap();
    assert_eq!(r.global_checkpoint(), 3);
    r.handle(&req(1, 1, vec![])).unwrap();
    assert_eq!(r.global_checkpoint(), 3, "lower gcp from a delayed request is ignored");
    r.handle(&req(1, 5, vec![])).unwrap(); // gcp-only advance is durable too
    drop(r);
    let r = ReplicaApplier::open(rsvc.clone(), 0).unwrap();
    assert_eq!(r.global_checkpoint(), 5);
    r.flush().unwrap();
    drop(r);
    let r = ReplicaApplier::open(rsvc.clone(), 0).unwrap();
    assert_eq!((r.global_checkpoint(), r.local_checkpoint()), (5, 5));
}

#[test]
fn crash_between_wal_append_and_apply_restores_same_checkpoint_and_docs() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = create(&tmp.path().join("p"), &json!({}));
    let rpath = tmp.path().join("r");
    let (rsvc, r) = replica(&rpath, &json!({}));
    for i in 0..20 {
        index(&primary, &format!("d{}", i % 12), json!({"title": format!("hello {i}"), "n": i}));
    }
    delete(&primary, "d3");
    let ops = tail_all(&primary, 0, 1);
    let (applied, pending) = ops.split_at(10);
    assert_eq!(r.apply(applied, None).unwrap(), 9);
    // Fsynced into the WAL, then the process dies before the engine apply.
    r.wal_append_only(pending).unwrap();
    assert_eq!(r.local_checkpoint(), 9, "not applied, not acked");
    drop(r);
    drop(rsvc);
    let rsvc = Arc::new(IndexService::open(&rpath).unwrap());
    let r = ReplicaApplier::open(rsvc.clone(), 0).unwrap();
    assert_eq!(r.local_checkpoint(), crate::shards::copy_of(&primary, 0).shard.max_seq_no());
    assert_identical(&primary, &rsvc, "after crash between append and apply");
    // And again (idempotent replay), including a redelivery.
    drop(r);
    let r = ReplicaApplier::open(rsvc.clone(), 0).unwrap();
    r.apply(&ops, None).unwrap();
    assert_eq!(r.local_checkpoint(), crate::shards::copy_of(&primary, 0).shard.max_seq_no());
    assert_identical(&primary, &rsvc, "second reopen");
}

#[test]
fn prepared_replica_replays_wal_before_publication_and_rejects_stale_installation() {
    use elasticsearch_engine::index::{RecoverySource as EngineRecovery, ShardOpenSpec};
    let tmp = tempfile::tempdir().unwrap();
    let (svc, old) = replica(&tmp.path().join("prepared"), &json!({}));
    let op = ReplicaOp::Index { id: "pending".into(), source_json: "{}".into(),
        routing: None, seq_no: 0, version: 1, primary_term: TERM };
    old.wal_append_only(&[op]).unwrap();
    svc.close_shard(0).unwrap();
    drop(old);
    let prepared = svc.prepare_shard(0, ShardOpenSpec::local_primary(EngineRecovery::ExistingStore)).unwrap();
    let other = create(&tmp.path().join("other"), &json!({}));
    assert!(ReplicaApplier::open_prepared(other, &prepared).is_err());
    let pending = ReplicaApplier::open_prepared(svc.clone(), &prepared).unwrap();
    assert_eq!(pending.local_checkpoint(), 0);
    assert!(svc.shard(0).is_none(), "WAL replay must not publish the slot");
    svc.install_prepared_shard(prepared).unwrap();
    assert_eq!(svc.doc_count_realtime().0, 1);
    svc.close_shard(0).unwrap();
    drop(pending);

    let mut prepared = svc.prepare_shard(0, ShardOpenSpec::local_primary(EngineRecovery::ExistingStore)).unwrap();
    let pending = ReplicaApplier::open_prepared(svc.clone(), &prepared).unwrap();
    svc.invalidate_shard_preparation(0).unwrap();
    assert!(ReplicaApplier::open_prepared(svc.clone(), &prepared).is_err());
    assert!(svc.try_install_prepared_shard(&mut prepared).is_err());
    assert!(svc.shard(0).is_none());
    drop(prepared);
    assert!(pending.copy().is_closed(), "discarded preparation fences retained applier handles");
    assert!(pending.recover_from_wal().is_err());
}

/// Replay of acknowledged ops is never refused or delayed by a circuit
/// breaker (ES translog recovery): a replica WAL op larger than the whole
/// request breaker replays on restart, while other work holds all of the
/// breaker, and the copy comes up with it. The refusal seam (standing in
/// for the breaker) does not refuse a replay either.
#[test]
fn a_replica_op_larger_than_the_request_breaker_replays_on_restart() {
    use elasticsearch_engine::memory::{scoped, Breakers, RequestScope};
    let tmp = tempfile::tempdir().unwrap();
    let rpath = tmp.path().join("replica");
    let (rsvc, r) = replica(&rpath, &json!({}));
    let big = "x".repeat(512 * 1024);
    let op = |seq: i64, source_json: String| ReplicaOp::Index { id: format!("t{seq}"), source_json, routing: None, seq_no: seq, version: 1, primary_term: TERM };
    r.handle(&req(TERM, -1, vec![op(0, json!({"n": 0}).to_string())])).unwrap();
    r.handle(&req(TERM, -1, vec![op(1, json!({"big": big}).to_string())])).unwrap();
    assert_eq!(r.local_checkpoint(), 1);
    // Crash: drop the engine without flushing; the WAL holds both ops.
    drop(r);
    drop(rsvc);
    let rsvc = Arc::new(IndexService::open(&rpath).unwrap());
    let tiny = Breakers::leak_for_tests(1 << 30);
    tiny.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
    tiny.request.set_limit(64 * 1024);
    let competing = tiny.reserve_request(64 * 1024, "competing request").unwrap();
    let tripped = tiny.request.tripped();
    let scope = RequestScope::new("replica restart past the breaker", tiny);
    let started = std::time::Instant::now();
    let r = scoped(Some(scope.clone()), || ReplicaApplier::open(rsvc.clone(), 0)).expect("the copy comes up");
    assert!(started.elapsed() < std::time::Duration::from_secs(30), "the replay waited for memory");
    assert_eq!(r.local_checkpoint(), 1);
    assert!(r.stats().wal_ops_replayed >= 1);
    assert_eq!(tiny.request.tripped(), tripped, "nothing was refused");
    // A replay again (explicit WAL recovery) with the refusal seam armed:
    // it is not reached as a refusal.
    *r.refuse_admission.lock() = Some("replica term staging");
    let stats = scoped(Some(scope.clone()), || r.recover_from_wal()).expect("a WAL replay is never refused");
    assert_eq!(stats.local_checkpoint, 1);
    assert_eq!(*r.refuse_admission.lock(), Some("replica term staging"), "the replay was not refused");
    *r.refuse_admission.lock() = None;
    let source = rsvc.get("t1").unwrap().expect("the big op replayed").source.unwrap();
    assert!(source.len() > 512 * 1024);
    drop(competing);
    // The copy keeps what it charged while it lives, not this test's request.
    std::mem::forget(scope);
}

/// Tom's replay policy: a live replicated batch is ordinary replication
/// (charged without refusal only past its WAL append): it does not force
/// the copy's background seals. A replay of the WAL (at open, or an
/// explicit recovery) does.
#[test]
fn only_a_wal_replay_forces_the_copys_seals_not_a_live_batch() {
    let tmp = tempfile::tempdir().unwrap();
    let (_svc, r) = replica(&tmp.path().join("forced"), &json!({}));
    let op = |seq| ReplicaOp::Index { id: format!("doc-{seq}"), source_json: "{}".into(),
        routing: None, seq_no: seq, version: 1, primary_term: TERM };
    r.handle(&req(TERM, -1, vec![op(0), op(1)])).unwrap();
    assert_eq!(r.local_checkpoint(), 1);
    assert!(!r.copy().shard.replay_seals_forced(), "a live replicated batch forced the copy's seals");
    r.wal_append_only(&[op(2)]).unwrap();
    assert_eq!(r.recover_from_wal().unwrap().local_checkpoint, 2);
    assert!(r.copy().shard.replay_seals_forced(), "the WAL replay did not force the copy's seals");
}

#[test]
fn explicit_recovery_rereads_records_appended_after_open() {
    let tmp = tempfile::tempdir().unwrap();
    let (svc, replica) = replica(&tmp.path().join("reread"), &json!({}));
    let op = |seq| ReplicaOp::Index { id: format!("doc-{seq}"), source_json: "{}".into(),
        routing: None, seq_no: seq, version: 1, primary_term: TERM };
    replica.wal_append_only(&[op(0)]).unwrap();
    assert_eq!(replica.local_checkpoint(), -1);
    assert_eq!(replica.recover_from_wal().unwrap().local_checkpoint, 0);
    replica.wal_append_only(&[op(1)]).unwrap();
    assert_eq!(replica.recover_from_wal().unwrap().local_checkpoint, 1);
    assert_eq!(svc.doc_count_realtime().0, 2);
    assert_eq!(replica.recover_from_wal().unwrap().local_checkpoint, 1);
    assert_eq!(svc.doc_count_realtime().0, 2);
}

/// Primary P (term 1) with replicas A and B. P replicates ops above the
/// global checkpoint to A only, then P is partitioned away and B is
/// promoted (term 2). A must detect that it holds ops B never had.
#[test]
fn divergent_replica_detected_needs_full_recovery() {
    let tmp = tempfile::tempdir().unwrap();
    let p = create(&tmp.path().join("p"), &json!({}));
    let (asvc, a) = replica(&tmp.path().join("a"), &json!({}));
    let (bsvc, b) = replica(&tmp.path().join("b"), &json!({}));
    let (csvc, c) = replica(&tmp.path().join("c"), &json!({}));
    for i in 0..10 {
        index(&p, &format!("d{i}"), json!({"n": i}));
    }
    let common = tail_all(&p, 0, 1);
    for r in [&a, &b, &c] {
        r.handle(&req(1, 9, common.clone())).unwrap();
    }
    // Only A and C get these (B is partitioned from P); gcp stays at 9.
    index(&p, "x1", json!({"n": 100}));
    delete(&p, "d1");
    let extra = tail_all(&p, 10, 1);
    a.handle(&req(1, 9, extra.clone())).unwrap();
    c.handle(&req(1, 9, extra.clone())).unwrap();
    assert_eq!(a.ops_above(9).unwrap(), extra);

    // B is promoted: its resync only covers up to its own max (9).
    let resync = b.promote_to_primary(2).unwrap();
    assert_eq!((resync.global_checkpoint, resync.max_seq_no), (9, 9));
    assert!(resync.ops.is_empty());
    let e = a.resync(&resync.to_request("idx", 0, "a").unwrap()).unwrap_err();
    assert!(matches!(e, ReplicationError::NeedsFullRecovery { .. }), "{e}");
    // Sticky, also across a restart: nothing is ever acked again.
    assert!(matches!(a.handle(&req(2, 9, vec![])).unwrap_err(), ReplicationError::NeedsFullRecovery { .. }));
    drop(a);
    let a = ReplicaApplier::open(asvc.clone(), 0).unwrap();
    assert!(a.needs_full_recovery().is_some());
    assert!(matches!(a.apply(&[], None).unwrap_err(), ReplicationError::NeedsFullRecovery { .. }));

    // C learns the new term from B's first ordinary request, which carries
    // a *different* op at seq 10 than the one C holds: detected too.
    index(&bsvc, "new", json!({"n": 7}));
    let b_ops = tail_all(&bsvc, 10, 2);
    assert_eq!(b_ops[0].seq_no(), 10);
    let e = c.handle(&req(2, 9, b_ops)).unwrap_err();
    assert!(matches!(e, ReplicationError::NeedsFullRecovery { .. }), "{e}");
    assert!(csvc.get("new").unwrap().is_none(), "divergent op not applied");
}

#[test]
fn identical_history_replica_resyncs_without_full_recovery() {
    let tmp = tempfile::tempdir().unwrap();
    let p = create(&tmp.path().join("p"), &json!({}));
    let (asvc, a) = replica(&tmp.path().join("a"), &json!({}));
    let (bsvc, b) = replica(&tmp.path().join("b"), &json!({}));
    let mut rng = rand::rngs::StdRng::seed_from_u64(11);
    for i in 0..30 {
        index(&p, &format!("d{}", i % 17), doc(&mut rng, i));
    }
    delete(&p, "d2");
    let ops = tail_all(&p, 0, 1);
    // Both copies have everything, but the gcp they know lags (20): ops
    // 21.. are above it and must be verified on failover.
    for r in [&a, &b] {
        r.handle(&req(1, 20, ops.clone())).unwrap();
    }
    // B misses nothing; A got one op twice and out of order (harmless).
    a.handle(&req(1, 20, vec![ops[25].clone()])).unwrap();

    // A hears of term 2 before the resync arrives: its reported checkpoint
    // is capped at the gcp until the suspects are verified.
    a.bump_term(2, 20).unwrap();
    assert!(a.resync_pending());
    assert_eq!(a.local_checkpoint(), 20);
    assert_eq!(durable_state(&asvc)["resync_pending_gcp"], 20);
    // A crash here keeps the pending resync.
    drop(a);
    let a = ReplicaApplier::open(asvc.clone(), 0).unwrap();
    assert!(a.resync_pending());
    assert_eq!(a.local_checkpoint(), 20);

    let resync = b.promote_to_primary(2).unwrap();
    assert_eq!(resync.ops.len() as i64, resync.max_seq_no - 20);
    let resp = a.resync(&resync.to_request("idx", 0, "a").unwrap()).unwrap();
    assert_eq!(resp.local_checkpoint, resync.max_seq_no);
    assert!(!a.resync_pending());
    assert!(a.needs_full_recovery().is_none());
    assert_eq!(durable_state(&asvc)["resync_pending_gcp"], Value::Null);

    // B now writes as primary under term 2; A follows.
    let start = crate::shards::copy_of(&bsvc, 0).shard.max_seq_no() + 1;
    index(&bsvc, "d5", json!({"title": "hello from b", "n": 1}));
    delete(&bsvc, "d6");
    let mut tailer = TranslogTailer::open_at(crate::shards::copy_of(&bsvc, 0).shard.path(), start).unwrap();
    let b_ops = ReplicaOp::from_translog_batch(tailer.poll().unwrap(), 2).unwrap();
    assert_eq!(b_ops.len(), 2);
    // Move the actual batch owner into the envelope without fixture re-admission.
    let request = ReplicationRequest::try_new("idx", 0, "x", 2, resync.max_seq_no, -1, None, b_ops, 0).unwrap();
    a.handle(&request).unwrap();
    assert_identical(&bsvc, &asvc, "a follows promoted b");
    // The deposed P can no longer reach A.
    index(&p, "late", json!({}));
    let late = tail_all(&p, 31, 1);
    assert!(matches!(a.handle(&req(1, 20, late)).unwrap_err(), ReplicationError::StalePrimaryTerm { ours: 2, theirs: 1 }));
}

#[test]
fn promotion_fills_gaps_with_noops() {
    let tmp = tempfile::tempdir().unwrap();
    let p = create(&tmp.path().join("p"), &json!({}));
    let (_bsvc, b) = replica(&tmp.path().join("b"), &json!({}));
    let (_asvc, a) = replica(&tmp.path().join("a"), &json!({}));
    for i in 0..6 {
        index(&p, &format!("d{i}"), json!({"n": i}));
    }
    let ops = tail_all(&p, 0, 1);
    // B lacks seq 4 (in flight when P died); A has 0..=3 only.
    b.handle(&req(1, 2, [&ops[..4], &ops[5..]].concat())).unwrap();
    a.handle(&req(1, 2, ops[..4].to_vec())).unwrap();
    let resync = b.promote_to_primary(2).unwrap();
    assert_eq!(resync.max_seq_no, 5);
    assert!(matches!(&resync.ops[1], ReplicaOp::NoOp { seq_no: 4, primary_term: 2, .. }));
    // A had nothing B lacks: resync succeeds and fills A up to 5.
    assert_eq!(a.resync(&resync.to_request("idx", 0, "a").unwrap()).unwrap().local_checkpoint, 5);
    assert_eq!(b.local_checkpoint(), 5);
}

/// Promotion drains a replicated batch still applying with the apply lock
/// released (ES: the term bump acquires all operation permits): the
/// resync and the new primary's sequence generation cover its op, which
/// must never land in the engine behind the promotion.
#[test]
fn promotion_drains_in_flight_replica_batches() {
    let tmp = tempfile::tempdir().unwrap();
    let (bsvc, b) = replica(&tmp.path().join("b"), &json!({}));
    let op = |seq| ReplicaOp::Index { id: format!("doc-{seq}"), source_json: "{}".into(),
        routing: None, seq_no: seq, version: 1, primary_term: TERM };
    b.handle(&req(TERM, 0, vec![op(0), op(1), op(2)])).unwrap();
    let pause = Arc::new(std::sync::Barrier::new(2));
    *b.apply_pause.lock() = Some(pause.clone());
    let applier = b.clone();
    let in_flight = std::thread::spawn(move || applier.handle(&req(TERM, 0, vec![op(3)])));
    pause.wait(); // seq_no 3 is in the WAL, its engine apply is held
    let promoter = b.clone();
    let promotion = std::thread::spawn(move || promoter.promote_to_primary(TERM + 1));
    // The promotion has adopted the new term (it published it) before the
    // held batch is released.
    while b.primary_term() != TERM + 1 {
        std::thread::yield_now();
    }
    pause.wait();
    in_flight.join().unwrap().unwrap();
    let resync = promotion.join().unwrap().unwrap();
    assert_eq!(resync.max_seq_no, 3, "the resync covers the in-flight op: {resync:?}");
    assert!(resync.ops.iter().any(|o| matches!(o, ReplicaOp::Index { seq_no: 3, .. })), "{resync:?}");
    assert_eq!(crate::shards::copy_of(&bsvc, 0).shard.max_seq_no(), 3, "new seq_nos start above every op the copy took");
    assert!(matches!(b.handle(&req(TERM, 0, vec![op(4)])).unwrap_err(), ReplicationError::StalePrimaryTerm { .. }));
}

/// A promoted copy whose highest seq_no is a no-op the engine never saw
/// must not reissue it: the new primary's first op would share a seq_no
/// with the resync's no-op.
#[test]
fn promotion_continues_sequencing_above_trailing_noops() {
    let tmp = tempfile::tempdir().unwrap();
    let (bsvc, b) = replica(&tmp.path().join("b"), &json!({}));
    let op = |seq| ReplicaOp::Index { id: format!("doc-{seq}"), source_json: "{}".into(),
        routing: None, seq_no: seq, version: 1, primary_term: TERM };
    let noop = ReplicaOp::NoOp { seq_no: 2, primary_term: TERM, reason: "failed write".into() };
    // seq_no 4 never arrived: the promotion fills it with a no-op too.
    let noop5 = ReplicaOp::NoOp { seq_no: 5, primary_term: TERM, reason: "failed write".into() };
    b.handle(&req(TERM, 0, vec![op(0), op(1), noop, op(3), noop5])).unwrap();
    let resync = b.promote_to_primary(TERM + 1).unwrap();
    assert_eq!(resync.max_seq_no, 5);
    assert!(matches!(&resync.ops[..], [_, ReplicaOp::NoOp { seq_no: 2, .. }, _, ReplicaOp::NoOp { seq_no: 4, .. }, ReplicaOp::NoOp { seq_no: 5, .. }]), "{resync:?}");
    let engine = crate::shards::copy_of(&bsvc, 0);
    assert_eq!(engine.shard.max_seq_no(), 5, "sequencing continues above the no-ops");
    index(&bsvc, "new", json!({}));
    assert_eq!(bsvc.get("new").unwrap().unwrap().seq_no, 6);
}

#[test]
fn recovery_is_fenced_by_newer_primary_term() {
    let tmp = tempfile::tempdir().unwrap();
    let p = create(&tmp.path().join("p"), &json!({}));
    for i in 0..30 {
        index(&p, &format!("d{i}"), json!({"n": i}));
    }
    let mut source = RecoverySource::start(p.clone(), 0, 1, RecoveryOptions { batch_docs: 10, ..Default::default() }).unwrap();
    let info = source.info().clone();
    let tsvc = RecoveryTarget::create_index(&tmp.path().join("t"), &info).unwrap();
    let mut target = RecoveryTarget::new(tsvc.clone(), info.clone()).unwrap();
    target.apply_snapshot_batch(&source.next_snapshot_batch().unwrap().unwrap()).unwrap();
    // A new primary (term 2) contacts the target: the term-1 recovery dies.
    target.applier().bump_term(2, -1).unwrap();
    let e = target.apply_snapshot_batch(&source.next_snapshot_batch().unwrap().unwrap()).unwrap_err();
    assert!(matches!(e, ReplicationError::RecoveryAborted { recovery_term: 1, current_term: 2 }), "{e}");
    assert!(matches!(target.finalize(0).unwrap_err(), ReplicationError::RecoveryAborted { .. }));
    // Restarting a term-1 recovery on a copy that has seen term 2 is refused.
    drop(target);
    let e = RecoveryTarget::new(tsvc, info).err().unwrap();
    assert!(matches!(e, ReplicationError::RecoveryAborted { recovery_term: 1, current_term: 2 }), "{e}");
}

/// ES `IndexShard.postRecovery` refreshes: a recovered replica serves
/// searches over everything it recovered without waiting for a periodic
/// refresh (a search routed to a just-started replica must not miss
/// documents the primary already made visible).
#[test]
fn recovered_replica_is_searchable_without_refresh() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = create(&tmp.path().join("p"), &json!({}));
    for i in 0..5 {
        index(&primary, &format!("d{i}"), json!({"title": "hello", "n": i}));
    }
    primary.refresh().unwrap();
    let mut source = RecoverySource::start(primary.clone(), 0, TERM, RecoveryOptions::default()).unwrap();
    let info = source.info().clone();
    let tsvc = RecoveryTarget::create_index(&tmp.path().join("t"), &info).unwrap();
    let mut target = RecoveryTarget::new(tsvc.clone(), info.clone()).unwrap();
    while let Some(b) = source.next_snapshot_batch().unwrap() {
        target.apply_snapshot_batch(&b).unwrap();
    }
    let max = target.snapshot_done().unwrap();
    // A write during the translog phase is recovered too.
    index(&primary, "late", json!({"title": "hello", "n": 9}));
    let b = source.next_translog_batch().unwrap();
    target.apply_translog_batch(&b, Some(&primary.meta().mappings.to_json())).unwrap();
    target.finalize(max + 1).unwrap();
    assert_eq!(hits(&tsvc, &json!({"match_all": {}})), 6);
}

/// A replica of a `semantic_text` index on a node that is not the master
/// recovers only once that node knows the field's inference endpoint (the
/// published `_inference_endpoints` cluster state): replaying the primary's
/// `_source` resolves the endpoint like an index-time parse does.
#[test]
fn semantic_text_replica_recovers_with_published_endpoints() {
    use elasticsearch_engine::inference::{self, InferenceService};
    inference::mock::enable();
    let tmp = tempfile::tempdir().unwrap();
    let master = Arc::new(InferenceService::open(&tmp.path().join("master")));
    master.put("sparse_embedding", "sparse-ep", &json!({"service": "test_service", "service_settings": {"model": "m", "api_key": "k"}, "task_settings": {}})).unwrap();
    let mapping = json!({"properties": {"body": {"type": "semantic_text", "inference_id": "sparse-ep"}}});
    let primary = inference::scoped(Some(master.clone()), || {
        let p = create(&tmp.path().join("p"), &mapping);
        index(&p, "d1", json!({"body": "hello world"}));
        p.refresh().unwrap();
        p
    });
    let recover = |other: Arc<InferenceService>, dir: &str| {
        inference::scoped(Some(other), || -> std::result::Result<Arc<IndexService>, String> {
            let mut source = RecoverySource::start(primary.clone(), 0, TERM, RecoveryOptions::default()).map_err(|e| e.to_string())?;
            let info = source.info().clone();
            let tsvc = RecoveryTarget::create_index(&tmp.path().join(dir), &info).map_err(|e| e.to_string())?;
            let mut target = RecoveryTarget::new(tsvc.clone(), info).map_err(|e| e.to_string())?;
            while let Some(b) = source.next_snapshot_batch().map_err(|e| e.to_string())? {
                target.apply_snapshot_batch(&b).map_err(|e| e.to_string())?;
            }
            let max = target.snapshot_done().map_err(|e| e.to_string())?;
            target.finalize(max).map_err(|e| e.to_string())?;
            Ok(tsvc)
        })
    };
    // Without the published endpoints the replay fails like ES's replica would.
    let bare = Arc::new(InferenceService::open(&tmp.path().join("bare")));
    let err = recover(bare, "t-bare").unwrap_err();
    assert!(err.contains("sparse-ep"), "{err}");
    // With them (as `meta_extra::apply_inference` restores) it recovers.
    let other = Arc::new(InferenceService::open(&tmp.path().join("other")));
    other.restore_cluster_state(&master.cluster_state()).unwrap();
    let tsvc = recover(other, "t").unwrap();
    assert_eq!(hits(&tsvc, &json!({"match_all": {}})), 1);
}

#[test]
fn segment_batches_stop_at_each_segment() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = create(&tmp.path().join("p"), &json!({}));
    for i in 0..3 {
        index(&primary, &format!("a{i}"), json!({"n": i}));
    }
    crate::shards::copy_of(&primary, 0).shard.flush().unwrap();
    for i in 0..2 {
        index(&primary, &format!("b{i}"), json!({"n": i}));
    }
    crate::shards::copy_of(&primary, 0).shard.flush().unwrap();
    let mut source = RecoverySource::start(primary.clone(), 0, TERM, RecoveryOptions { segment_batches: true, ..Default::default() }).unwrap();
    let mut sizes = Vec::new();
    while let Some(b) = source.next_snapshot_batch().unwrap() {
        assert!(source.at_segment_boundary());
        sizes.push(b.len());
    }
    sizes.sort();
    assert_eq!(sizes, vec![2, 3], "one batch per segment");
}

#[test]
fn recovery_reuses_registered_applier_and_observes_term_fences() {
    let tmp = tempfile::tempdir().unwrap();
    let p = create(&tmp.path().join("p"), &json!({}));
    index(&p, "doc", json!({"n": 1}));
    let mut source = RecoverySource::start(p, 0, 1, RecoveryOptions::default()).unwrap();
    let info = source.info().clone();
    let svc = RecoveryTarget::create_index(&tmp.path().join("r"), &info).unwrap();
    let registered = Arc::new(ReplicaApplier::open(svc, 0).unwrap());
    let mut target = RecoveryTarget::with_applier(registered.clone(), info.clone()).unwrap();
    assert!(Arc::ptr_eq(target.applier(), &registered));
    target.apply_snapshot_batch(&source.next_snapshot_batch().unwrap().unwrap()).unwrap();
    assert_eq!(registered.local_checkpoint(), target.applier().local_checkpoint());
    // Cluster-state term fencing must affect the already-running recovery,
    // not a second applier with an independent lock and stale term snapshot.
    registered.bump_term(2, -1).unwrap();
    assert!(matches!(target.snapshot_done(), Err(ReplicationError::RecoveryAborted { recovery_term: 1, current_term: 2 })));
    assert!(matches!(RecoveryTarget::with_applier(registered, info), Err(ReplicationError::RecoveryAborted { recovery_term: 1, current_term: 2 })));
}

/// A replica keeps applying replicated batches while it flushes:
/// `ReplicaApplier::flush` does not hold `inner` (which `apply` / `handle`
/// need) across the engine flush. Flushes and applies interleave on two
/// threads under a watchdog; the replica ends identical to the primary.
#[test]
fn replica_applies_batches_while_flushing() {
    let tmp = tempfile::tempdir().unwrap();
    let mapping = json!({"properties": {"title": {"type": "text"}}});
    let primary = create(&tmp.path().join("p"), &mapping);
    let (r1svc, r1) = replica(&tmp.path().join("r1"), &mapping);
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let mut tailer = TranslogTailer::open(crate::shards::copy_of(&primary, 0).shard.path()).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let flusher = {
        let (r1, stop) = (r1.clone(), stop.clone());
        std::thread::spawn(move || {
            let mut n = 0;
            while !stop.load(Ordering::Relaxed) {
                r1.flush().unwrap();
                n += 1;
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            n
        })
    };
    let (tx, rx) = std::sync::mpsc::channel();
    let applier = {
        let (primary, r1) = (primary.clone(), r1.clone());
        std::thread::spawn(move || {
            for round in 0..40 {
                for i in 0..25 {
                    let id = format!("d{}", rng.gen_range(0..200));
                    if rng.gen_range(0..5) == 0 {
                        delete_if_present(&primary, &id);
                    } else {
                        index(&primary, &id, doc(&mut rng, round * 100 + i));
                    }
                }
                let ops = ReplicaOp::from_translog_batch(tailer.poll().unwrap(), TERM).unwrap();
                let mapping = primary.meta().mappings.to_json();
                for chunk in ops.chunks(9) {
                    r1.apply(chunk, Some(&mapping)).unwrap();
                }
            }
            let _ = tx.send(());
        })
    };
    rx.recv_timeout(std::time::Duration::from_secs(120)).expect("replica apply stalled behind its flush");
    applier.join().unwrap();
    stop.store(true, Ordering::Relaxed);
    assert!(flusher.join().unwrap() > 0);
    r1.flush().unwrap();
    assert_eq!(r1.local_checkpoint(), crate::shards::copy_of(&primary, 0).shard.max_seq_no());
    assert_identical(&primary, &r1svc, "r1");
}

fn delete_if_present(svc: &IndexService, id: &str) {
    let _ = svc.write(vec![WriteReq::Delete { id: id.into(), routing: None, cond: WriteCond::default() }], false);
}

/// R7: replicated batches delivered shuffled and applied concurrently, the
/// last few only reaching the WAL before a crash: the reopened replica
/// holds exactly the primary's last op per id (deletes included), each
/// once, and its checkpoint covers every op.
#[test]
fn shuffled_concurrent_replica_batches_and_crash_match_the_primary() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("shuffled");
    let (svc, r) = replica(&path, &json!({"properties": {"n": {"type": "long"}}}));
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x5EED);
    let mut versions: BTreeMap<String, u64> = BTreeMap::new();
    let mut last: BTreeMap<String, Option<(u64, i64)>> = BTreeMap::new();
    let mut ops = Vec::new();
    for seq_no in 0..600i64 {
        let id = format!("d{}", rng.gen_range(0..25));
        let version = versions.get(&id).map_or(1, |v| v + 1);
        versions.insert(id.clone(), version);
        if rng.gen_range(0..4) == 0 {
            ops.push(ReplicaOp::Delete { id: id.clone(), seq_no, version, primary_term: TERM });
            last.insert(id, None);
        } else {
            ops.push(ReplicaOp::Index { id: id.clone(), source_json: json!({"n": seq_no}).to_string(), routing: None, seq_no, version, primary_term: TERM });
            last.insert(id, Some((version, seq_no)));
        }
    }
    ops.shuffle(&mut rng);
    let batches: Vec<Vec<ReplicaOp>> = ops.chunks(12).map(<[ReplicaOp]>::to_vec).collect();
    let (applied, crashed) = batches.split_at(batches.len() - 3);
    std::thread::scope(|scope| {
        for lane in 0..4 {
            let r = r.clone();
            scope.spawn(move || {
                for batch in applied.iter().skip(lane).step_by(4) {
                    r.apply(batch, None).unwrap();
                }
            });
        }
    });
    // Durable in the WAL, never applied: a crash between the two.
    for batch in crashed { r.wal_append_only(batch).unwrap(); }
    drop(r); drop(svc);
    let svc = Arc::new(IndexService::open(&path).unwrap());
    let r = ReplicaApplier::open(svc.clone(), 0).unwrap();
    assert_eq!(r.local_checkpoint(), 599, "every op is processed after the WAL replay");
    let expected: BTreeMap<String, (u64, i64)> = last.into_iter().filter_map(|(id, state)| state.map(|s| (id, s))).collect();
    let got: BTreeMap<String, (u64, i64)> = docs(&svc).into_iter().map(|(id, (version, seq_no, _, _))| (id, (version, seq_no))).collect();
    assert_eq!(got, expected);
}

/// D1/D2 through ops-based peer recovery: the primary commits auto-id
/// writes and a delete above a held batch's commit point; the replica
/// already holds those ops (but not the held batch), so its prefix stops
/// below them; recovery from that prefix replays them again with the held
/// batch: each auto-id document once, the delete wins, both copies equal.
#[test]
fn peer_recovery_replay_above_the_commit_point_keeps_docs_once_and_deletes_win() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = create(&tmp.path().join("primary"), &json!({}));
    index(&primary, "x", json!({"n": 1}));
    let shard = crate::shards::copy_of(&primary, 0);
    let (sequenced_tx, sequenced) = std::sync::mpsc::channel();
    let (release, release_rx) = std::sync::mpsc::channel::<()>();
    shard.shard.pause_next_staged_write(move || {
        let _ = sequenced_tx.send(());
        let _ = release_rx.recv_timeout(std::time::Duration::from_secs(30));
    });
    let held = {
        let primary = primary.clone();
        std::thread::spawn(move || primary.write(vec![
            WriteReq::Index { id: Some("x".into()), source: json!({"n": 2}).to_string(), routing: None, cond: WriteCond::default() },
            WriteReq::Index { id: Some("held1".into()), source: json!({"n": 3}).to_string(), routing: None, cond: WriteCond::default() },
        ], false))
    };
    sequenced.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
    for _ in 0..10 {
        let reqs = (0..3).map(|_| WriteReq::Index { id: None, source: json!({"n": 9}).to_string(), routing: None, cond: WriteCond::default() }).collect();
        for r in primary.write(reqs, false) { r.unwrap(); }
    }
    delete(&primary, "x");
    shard.shard.flush().unwrap();
    let committed = shard.shard.committed_seq_no();
    release.send(()).unwrap();
    let held = held.join().unwrap();
    let held_first = held[0].as_ref().unwrap().seq_no;
    assert!(committed < held_first, "the held batch holds the primary's commit point");
    // The replica got every op but the held batch's: its prefix stops below them.
    let (svc, r) = replica(&tmp.path().join("replica"), &json!({}));
    let ops = ReplicaOp::from_translog_batch(TranslogTailer::open(shard.shard.path()).unwrap().poll().unwrap(), TERM).unwrap();
    let (held_ops, others): (Vec<ReplicaOp>, Vec<ReplicaOp>) = ops.iter().cloned().partition(|op| op.seq_no() == held_first || op.seq_no() == held_first + 1);
    assert_eq!(held_ops.len(), 2);
    r.apply(&others, Some(&primary.mapping_json())).unwrap();
    let prefix = r.local_checkpoint();
    assert_eq!(prefix, held_first - 1);
    r.update_global_checkpoint(prefix).unwrap();
    r.flush().unwrap();
    let mut source = RecoverySource::resume(primary.clone(), 0, TERM, prefix, RecoveryOptions::default()).unwrap();
    assert!(source.next_snapshot_batch().unwrap().is_none());
    let mut target = RecoveryTarget::with_applier(r.clone(), source.info().clone()).unwrap();
    target.snapshot_done().unwrap();
    loop {
        let batch = source.next_translog_batch().unwrap();
        if batch.is_empty() { break; }
        target.apply_translog_batch(&batch, Some(&primary.mapping_json())).unwrap();
    }
    let max = shard.shard.max_seq_no();
    assert_eq!(target.finalize(max).unwrap(), max);
    assert!(svc.get("x").unwrap().is_none(), "the delete lost to the replayed older index op");
    assert_identical(&primary, &svc, "ops-based recovery above the commit point");
}

/// B4: a replica's engine flush is detached and can commit below the
/// checkpoint it was started at (here: below memtables a background seal
/// still encodes). The WAL must keep those ops: after a crash before the
/// seal installs, the replica recovers every one of them.
#[test]
fn replica_flush_never_trims_the_wal_past_the_engine_commit() {
    let tmp = tempfile::tempdir().unwrap();
    let mapping = json!({});
    let primary = create(&tmp.path().join("p"), &mapping);
    let rpath = tmp.path().join("r");
    let (rsvc, r) = replica(&rpath, &mapping);
    let mut tailer = TranslogTailer::open(crate::shards::copy_of(&primary, 0).shard.path()).unwrap();
    for i in 0..20 {
        index(&primary, &format!("d{i}"), json!({"n": i}));
    }
    let ops = ReplicaOp::from_translog_batch(tailer.poll().unwrap(), TERM).unwrap();
    r.apply(&ops, None).unwrap();
    let max = crate::shards::copy_of(&primary, 0).shard.max_seq_no();
    r.update_global_checkpoint(max).unwrap();
    // A background seal claims the replica's (published) memtables and is
    // still encoding when the flush runs.
    let copy = crate::shards::copy_of(&rsvc, 0);
    copy.shard.refresh().unwrap();
    let in_flight = copy.shard.core.write().begin_seal_part(false, usize::MAX).unwrap().expect("published memtables to seal");
    r.flush().unwrap();
    assert!(copy.shard.committed_seq_no() < max, "the detached flush committed below the in-flight seal");
    // Crash before the seal installs.
    drop(in_flight);
    drop(copy);
    drop(r);
    drop(rsvc);
    let rsvc = Arc::new(IndexService::open(&rpath).unwrap());
    let r = ReplicaApplier::open(rsvc.clone(), 0).unwrap();
    assert_eq!(r.local_checkpoint(), max);
    assert_identical(&primary, &rsvc, "after a crash beside an in-flight seal");
}

/// A peer-recovery snapshot batch carries the source's mapping.
/// A target whose local mapping lags (its node has not applied the state
/// that mapped `code`) installs it before the batch's documents, instead of
/// mapping the field dynamically with another type.
#[test]
fn recovery_snapshot_batches_install_the_sources_mapping() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = create(&tmp.path().join("primary"), &json!({"properties": {"code": {"type": "keyword"}}}));
    index(&primary, "a", json!({"code": "ABC-1"}));
    let mut source = RecoverySource::start(primary.clone(), 0, TERM, RecoveryOptions::default()).unwrap();
    let info = source.info().clone();
    // The target's mapping predates `code`.
    let tsvc = create(&tmp.path().join("t"), &json!({}));
    let mut target = RecoveryTarget::new(tsvc.clone(), info).unwrap();
    while let Some(batch) = source.next_snapshot_batch().unwrap() {
        target.apply_snapshot_batch_mapped(&batch, Some(&primary.mapping_json()), 0).unwrap();
    }
    assert_eq!(tsvc.lookup().get("code").map(|f| f.ty.name().to_string()).as_deref(), Some("keyword"), "the snapshot's field took the source's type");
    target.snapshot_done().unwrap();
    let max = crate::shards::copy_of(&primary, 0).shard.max_seq_no();
    assert_eq!(target.finalize(max).unwrap(), max);
    assert_identical(&primary, &tsvc, "snapshot batch with the source's mapping");
    // The batch's mapping version survives the wire, and an older source's
    // batch (without it) still decodes.
    let b = crate::shards::RecoveryBatch::try_new(true, super::ops::admitted_ops_for_test(vec![]), false, "{}", 3, 7).unwrap();
    let back: crate::shards::RecoveryBatch = bincode::deserialize(&bincode::serialize(&b).unwrap()).unwrap();
    assert_eq!((back.mapping_version, back.global_checkpoint), (7, 3));
    let old = bincode::serialize(&(true, Vec::<ReplicaOp>::new(), false, "{}".to_string(), 3i64)).unwrap();
    let back: crate::shards::RecoveryBatch = bincode::deserialize(&old).unwrap();
    assert_eq!((back.mapping_version, back.global_checkpoint, back.snapshot), (0, 3, true));
    // A newer source's batch on an older target: the appended version is
    // ignored (review P2-3, both directions).
    let (snapshot, ops, phase_done, mapping_json, global_checkpoint): (bool, Vec<ReplicaOp>, bool, String, i64) = bincode::deserialize(&bincode::serialize(&b).unwrap()).unwrap();
    assert_eq!((snapshot, ops.len(), phase_done, mapping_json.as_str(), global_checkpoint), (true, 0, false, "{}", 3));
}

/// A replica whose node already applied a newer committed
/// mapping keeps it when a replication request carries an older version;
/// a newer version still installs, and an unknown version (an older
/// primary) behaves as before.
#[test]
fn a_replica_never_reverts_to_an_older_mapping_version() {
    let tmp = tempfile::tempdir().unwrap();
    let older = json!({"properties": {"a": {"type": "keyword", "ignore_above": 10}}});
    let (rsvc, r) = replica(&tmp.path().join("r"), &older);
    // The replica's node applies mapping version 3 (a raised ignore_above).
    rsvc.put_mapping(&json!({"properties": {"a": {"type": "keyword", "ignore_above": 512}}})).unwrap();
    rsvc.advance_mapping_version(3);
    let op = |seq: i64| ReplicaOp::Index { id: format!("d{seq}"), source_json: "{\"a\":\"v\"}".into(), routing: None, seq_no: seq, version: 1, primary_term: TERM };
    let ignore_above = || rsvc.mapping_json()["properties"]["a"]["ignore_above"].clone();
    // A request built on version 2.
    r.apply_versioned(&[op(0)], Some(&older), 2).unwrap();
    assert_eq!(ignore_above(), json!(512), "an older mapping version reverted the replica's mapping");
    // Version 4 adds a field.
    let newer = json!({"properties": {"a": {"type": "keyword", "ignore_above": 512}, "b": {"type": "long"}}});
    r.apply_versioned(&[op(1)], Some(&newer), 4).unwrap();
    assert!(rsvc.lookup().get("b").is_some(), "a newer mapping version installs");
    assert_eq!(rsvc.mapping_version(), 4);
    // The same version again (another request of version 4) is a no-op.
    let updates = r.stats().mapping_updates;
    r.apply_versioned(&[op(2)], Some(&newer), 4).unwrap();
    assert_eq!(r.stats().mapping_updates, updates);
    // Newer content read under the same version (review round 3): its
    // additions install, its older parameters do not revert.
    let racing = json!({"properties": {"a": {"type": "keyword", "ignore_above": 10}, "b": {"type": "long"}, "d": {"type": "keyword"}}});
    r.apply_versioned(&[op(4)], Some(&racing), 4).unwrap();
    assert!(rsvc.lookup().get("d").is_some(), "a same-version addition installs");
    assert_eq!(ignore_above(), json!(512), "a same-version request never reverts a parameter");
    // Version unknown (an older primary): installed when it differs, as before.
    let unversioned = json!({"properties": {"a": {"type": "keyword", "ignore_above": 512}, "b": {"type": "long"}, "c": {"type": "keyword"}}});
    r.apply(&[op(3)], Some(&unversioned)).unwrap();
    assert!(rsvc.lookup().get("c").is_some());
}

// ---------------------------------------------------------------------------
// Restart reuse (B4): operations-based catch-up of a copy with local data
// ---------------------------------------------------------------------------

/// A replica that restarts with intact local data catches up by operations
/// from its global checkpoint, although its WAL tail is neither committed
/// nor acknowledged (the 6.35M fixture replayed ~85K such ops and was then
/// fully re-indexed): no snapshot docs, and the ops it holds above the
/// checkpoint are verified against the primary's under the new term.
#[test]
fn restarted_replica_with_an_unflushed_wal_catches_up_by_operations() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = create(&tmp.path().join("p"), &json!({}));
    let rpath = tmp.path().join("r");
    let (rsvc, r) = replica(&rpath, &json!({}));
    for i in 0..60 {
        index(&primary, &format!("d{i}"), json!({"title": format!("hello {i}"), "n": i}));
    }
    r.apply(&tail_all(&primary, 0, TERM), Some(&primary.mapping_json())).unwrap();
    r.update_global_checkpoint(39).unwrap();
    // Crash: nothing flushed, the WAL holds all 60 ops.
    drop(r);
    drop(rsvc);
    let rsvc = Arc::new(IndexService::open(&rpath).unwrap());
    let r = Arc::new(ReplicaApplier::open(rsvc.clone(), 0).unwrap());
    assert_eq!(r.stats().wal_ops_replayed, 60);
    let prefix = r.reusable_prefix().expect("an unflushed, partly acknowledged WAL is reusable");
    assert_eq!(prefix.checkpoint, 39);
    // Meanwhile the primary moved on (a delete and an overwrite included).
    delete(&primary, "d5");
    index(&primary, "d6", json!({"n": 600}));
    index(&primary, "new", json!({"n": 1}));
    // The primary was re-allocated too: a new term.
    let mut source = RecoverySource::resume(primary.clone(), 0, TERM + 1, prefix.checkpoint, RecoveryOptions::default()).unwrap();
    assert_eq!(source.info().snapshot_docs, 0);
    let mut target = RecoveryTarget::with_applier(r.clone(), source.info().clone()).unwrap();
    assert!(r.resync_pending(), "ops 40..59 are verified against the primary's");
    target.snapshot_done().unwrap();
    let mapping = primary.mapping_json();
    let mut sent = 0;
    loop {
        let b = source.next_translog_batch().unwrap();
        if b.is_empty() {
            break;
        }
        sent += b.len();
        target.apply_translog_batch(&b, Some(&mapping)).unwrap();
    }
    assert_eq!(sent, 20 + 3, "only the ops above the global checkpoint");
    assert!(!r.resync_pending(), "every suspect verified");
    assert!(r.stats().ops_duplicate >= 20, "what it held was not applied again");
    let max = crate::shards::copy_of(&primary, 0).shard.max_seq_no();
    assert_eq!(target.finalize(max).unwrap(), max);
    assert_identical(&primary, &rsvc, "restarted replica");
}

/// A deposed primary that comes back catches up by operations from the
/// global checkpoint it recorded when its history matches the new
/// primary's, and fails (to recover from files) when it held an op the
/// new primary never got.
#[test]
fn deposed_primary_catches_up_by_operations_or_fails_on_divergence() {
    for diverged in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        let old = create(&tmp.path().join("old-primary"), &json!({}));
        let (nsvc, newp) = replica(&tmp.path().join("new-primary"), &json!({}));
        for i in 0..10 {
            index(&old, &format!("d{i}"), json!({"n": i}));
        }
        newp.apply(&tail_all(&old, 0, TERM), Some(&old.mapping_json())).unwrap();
        newp.update_global_checkpoint(9).unwrap();
        if diverged {
            // seq_no 10, never replicated.
            index(&old, "lost", json!({"n": -1}));
        }
        // Failover: the replica is promoted under term 2 and writes on.
        newp.promote_to_primary(TERM + 1).unwrap();
        crate::shards::copy_of(&nsvc, 0).shard.set_primary_term(TERM + 1).unwrap();
        for i in 10..13 {
            index(&nsvc, &format!("n{i}"), json!({"n": i}));
        }
        let demoted = Arc::new(ReplicaApplier::open_demoted(old.clone(), 0, 9, TERM).unwrap());
        let prefix = demoted.reusable_prefix().unwrap();
        assert_eq!((prefix.checkpoint, prefix.primary_term), (9, TERM));
        let mut source = RecoverySource::resume(nsvc.clone(), 0, TERM + 1, prefix.checkpoint, RecoveryOptions::default()).unwrap();
        let mut target = RecoveryTarget::with_applier(demoted.clone(), source.info().clone()).unwrap();
        assert_eq!(demoted.resync_pending(), diverged);
        target.snapshot_done().unwrap();
        let mapping = nsvc.mapping_json();
        let mut result = Ok(0);
        loop {
            let b = source.next_translog_batch().unwrap();
            if b.is_empty() {
                break;
            }
            result = target.apply_translog_batch(&b, Some(&mapping));
            if result.is_err() {
                break;
            }
        }
        if diverged {
            assert!(matches!(result, Err(ReplicationError::NeedsFullRecovery { .. })), "{result:?}");
            assert!(demoted.needs_full_recovery().is_some(), "sticky: the copy recovers from files");
        } else {
            result.unwrap();
            let max = crate::shards::copy_of(&nsvc, 0).shard.max_seq_no();
            assert_eq!(target.finalize(max).unwrap(), max);
            assert_identical(&nsvc, &old, "deposed primary caught up");
        }
    }
}

/// `_recovery` / `_cat/recovery` of a file-based recovery: ES's field
/// shapes and percent rules (of the files not reused).
#[test]
fn file_phase_progress_is_reported_in_es_shapes() {
    use super::recovery::{FileDetails, FileRecoveryStats};
    let info = RecoveryStartInfo::try_new("idx","u",0,1,"{}","{}",10,9,0,0,4).unwrap();
    let files = FileRecoveryStats { files_total: 4, files_reused: 1, files_recovered: 1, bytes_total: 1000, bytes_reused: 100, bytes_recovered: 450,
        source_throttle_millis: 7, target_throttle_millis: 9, time_millis: 30,
        details: { let mut details=FileDetails::default(); details.push("core-1/seg_00000001.rstc",100,0,true).unwrap(); details } };
    let st = RecoveryState::file_phase(&info, files, Some(super::recovery::RecoveryText::try_new("n0").unwrap()), Some(super::recovery::RecoveryText::try_new("n1").unwrap())).unwrap();
    let j = st.to_recovery_json();
    assert_eq!(j["stage"], "INDEX");
    assert_eq!(j["index"]["size"], json!({"total_in_bytes": 1000, "reused_in_bytes": 100, "recovered_in_bytes": 450, "recovered_from_snapshot_in_bytes": 0, "percent": "50.0%"}));
    assert_eq!(j["index"]["files"]["total"], 4);
    assert_eq!(j["index"]["files"]["percent"], "33.3%");
    assert_eq!(j["index"]["source_throttle_time_in_millis"], 7);
    assert_eq!(j["index"]["target_throttle_time_in_millis"], 9);
    assert_eq!(j["index"]["files"]["details"][0]["reused"], true);
    let row = st.cat_row();
    assert_eq!((row["files"].as_str(), row["files_total"].as_str(), row["bytes"].as_str(), row["bytes_percent"].as_str()), ("3", "4", "900", "50.0%"));
}

/// Every admission a replicated batch needs between its WAL
/// append and marking its ops processed is made before the append. A
/// refused batch leaves the WAL untouched, so the primary's retry is not a
/// second durable copy of the same ops, and the retry applies.
#[test]
fn a_refused_replica_batch_appends_nothing_and_its_retry_applies() {
    let tmp = tempfile::tempdir().unwrap();
    let (svc, r) = replica(&tmp.path().join("replica"), &json!({}));
    let op = |seq: i64| ReplicaOp::Index { id: format!("t{seq}"), source_json: format!("{{\"n\":{seq}}}"), routing: None, seq_no: seq, version: 1, primary_term: TERM };
    r.handle(&req(TERM, -1, vec![op(0)])).unwrap();
    let mut next = 1;
    for label in ["replica term staging", "replica apply result slots"] {
        let records = r.wal_record_count();
        let batch = vec![op(next), op(next + 1)];
        *r.refuse_admission.lock() = Some(label);
        let e = r.handle(&req(TERM, -1, batch.clone())).unwrap_err();
        assert_eq!(e.es_type(), "circuit_breaking_exception", "{label}: {e}");
        assert!(r.refuse_admission.lock().is_none(), "{label}: the injected refusal was reached");
        assert_eq!(r.wal_record_count(), records, "{label}: a refused batch appends nothing");
        assert_eq!(r.local_checkpoint(), next - 1, "{label}");
        let resp = r.handle(&req(TERM, -1, batch)).unwrap();
        assert_eq!(resp.local_checkpoint, next + 1, "{label}: the retry applies");
        assert_eq!(resp.applied, 2, "{label}");
        assert_eq!(r.wal_record_count(), records + 1, "{label}: one durable copy of the batch");
        next += 2;
    }
    for seq in 0..next {
        assert!(svc.get(&format!("t{seq}")).unwrap().is_some(), "t{seq}");
    }
}

/// Once a replicated batch is durable in the WAL,
/// nothing until its ops are marked processed is refused (the engine's
/// staging and apply reservations are forced, as a WAL replay's): the
/// request breaker filled between the append and the apply neither fails
/// the batch nor counts a trip, and the batch applies exactly once.
#[test]
fn a_replicated_batch_is_never_refused_after_its_wal_append() {
    use elasticsearch_engine::memory::{scoped, Breakers, Kind, RequestScope};
    let tmp = tempfile::tempdir().unwrap();
    let (svc, r) = replica(&tmp.path().join("replica"), &json!({}));
    let op = |seq: i64| ReplicaOp::Index { id: format!("a{seq}"), source_json: format!("{{\"n\":{seq}}}"), routing: None, seq_no: seq, version: 1, primary_term: TERM };
    r.handle(&req(TERM, -1, vec![op(0)])).unwrap();
    let tiny = Breakers::leak_for_tests(1 << 20);
    tiny.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
    let limit = tiny.get(Kind::Request).limit() as u64;
    let records = r.wal_record_count();
    let pause = Arc::new(std::sync::Barrier::new(2));
    *r.apply_pause.lock() = Some(pause.clone());
    let applier = r.clone();
    let worker = std::thread::spawn(move || {
        scoped(Some(RequestScope::new("replicated batch", tiny)), || applier.handle(&req(TERM, -1, vec![op(1), op(2)])))
    });
    pause.wait(); // durable in the WAL, its engine apply held
    assert_eq!(r.wal_record_count(), records + 1);
    let used = tiny.get(Kind::Request).used();
    let held = tiny.reserve_request(limit.saturating_sub(used), "competing request").unwrap();
    let tripped = tiny.get(Kind::Request).tripped();
    pause.wait();
    let resp = worker.join().unwrap().expect("a batch in the WAL is applied, never refused");
    assert_eq!((resp.local_checkpoint, resp.applied), (2, 2));
    assert_eq!(tiny.get(Kind::Request).tripped(), tripped, "nothing was refused after the append");
    assert_eq!(r.wal_record_count(), records + 1, "one durable copy of the batch");
    drop(held);
    for seq in 0..3 {
        assert!(svc.get(&format!("a{seq}")).unwrap().is_some(), "a{seq}");
    }
}

/// A promotion interrupted after it saved the new term is
/// not skipped as already done. The durable marker makes the next promotion
/// under the same term run in full (gap fill, resync, WAL trim), also when
/// the copy restarted in between.
#[test]
fn a_promotion_refused_after_adopting_its_term_completes_on_retry_and_after_restart() {
    for restart in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        let p = create(&tmp.path().join("p"), &json!({}));
        let (bsvc, b) = replica(&tmp.path().join("b"), &json!({}));
        for i in 0..8 {
            index(&p, &format!("d{i}"), json!({"n": i}));
        }
        let ops = tail_all(&p, 0, 1);
        // Seq 5 never reached B: a gap above the global checkpoint (3).
        let held: Vec<ReplicaOp> = ops.iter().filter(|o| o.seq_no() != 5).cloned().collect();
        b.handle(&req(1, 3, held)).unwrap();
        assert_eq!((b.local_checkpoint(), b.max_seq_no()), (4, 7));

        // A promotion is never refused by a breaker (it is a replay); an
        // I/O failure after the term was saved is what interrupts it.
        b.interrupt_promotion.store(true, Ordering::Release);
        let e = b.promote_to_primary(2).unwrap_err();
        assert!(e.to_string().contains("injected promotion interruption"), "{e}");
        assert!(!b.interrupt_promotion.load(Ordering::Acquire), "the injected interruption was reached");
        assert_eq!(b.primary_term(), 2, "the term was adopted before the refusal");
        assert_eq!(b.promotion_pending(), Some(2));
        assert_eq!(durable_state(&bsvc)["promotion_pending"], 2);

        let b = if restart {
            drop(b);
            let b = Arc::new(ReplicaApplier::open(bsvc.clone(), 0).unwrap());
            assert_eq!((b.primary_term(), b.promotion_pending()), (2, Some(2)), "the marker survives a restart");
            b
        } else { b };
        let resync = b.promote_to_primary(2).expect("the unfinished promotion runs again under its term");
        assert_eq!((resync.primary_term, resync.global_checkpoint, resync.max_seq_no), (2, 3, 7));
        assert_eq!(resync.ops.iter().map(ReplicaOp::seq_no).collect::<Vec<_>>(), vec![4, 5, 6, 7]);
        assert!(matches!(&resync.ops[1], ReplicaOp::NoOp { primary_term: 2, .. }), "the gap is filled: {:?}", resync.ops[1]);
        assert_eq!(b.local_checkpoint(), 7);
        assert_eq!(b.promotion_pending(), None);
        assert_eq!(durable_state(&bsvc).get("promotion_pending"), None);
        // Done exactly once: the same term is now stale.
        assert!(matches!(b.promote_to_primary(2).unwrap_err(), ReplicationError::StalePrimaryTerm { ours: 2, theirs: 2 }));
        assert_eq!(crate::shards::copy_of(&bsvc, 0).shard.committed_seq_no(), 7, "restart={restart}");
    }
}

/// A promotion is a replay. With the request breaker full,
/// the last in-sync copy still applies its held ops, fills its gaps and
/// takes the primary role (a refusal would fail it and leave the shard red),
/// and nothing is counted as a trip.
#[test]
fn a_promotion_is_never_refused_by_the_breaker() {
    use elasticsearch_engine::memory::{scoped, Breakers, Kind, RequestScope};
    let tmp = tempfile::tempdir().unwrap();
    let p = create(&tmp.path().join("p"), &json!({}));
    let (bsvc, b) = replica(&tmp.path().join("b"), &json!({}));
    for i in 0..8 {
        index(&p, &format!("d{i}"), json!({"n": i}));
    }
    let ops = tail_all(&p, 0, 1);
    // Seq 5 never reached B (a gap); seq 6 is held in the WAL, not applied.
    let applied: Vec<ReplicaOp> = ops.iter().filter(|o| o.seq_no() != 5 && o.seq_no() != 6).cloned().collect();
    b.handle(&req(1, 3, applied)).unwrap();
    b.wal_append_only(&[ops[6].clone()]).unwrap();
    let tiny = Breakers::leak_for_tests(1 << 20);
    tiny.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
    let held = tiny.reserve_request(tiny.get(Kind::Request).limit() as u64, "competing request").unwrap();
    let tripped = tiny.get(Kind::Request).tripped();
    *b.refuse_admission.lock() = Some("promotion grouped no-op slots");
    let resync = scoped(Some(RequestScope::new("promotion", tiny)), || b.promote_to_primary(2))
        .expect("a promotion is never refused by the breaker");
    assert_eq!(*b.refuse_admission.lock(), Some("promotion grouped no-op slots"), "nothing is refused during the replay");
    *b.refuse_admission.lock() = None;
    assert_eq!(tiny.get(Kind::Request).tripped(), tripped);
    assert_eq!(resync.ops.iter().map(ReplicaOp::seq_no).collect::<Vec<_>>(), vec![4, 5, 6, 7]);
    assert!(matches!(&resync.ops[1], ReplicaOp::NoOp { .. }));
    assert!(!matches!(&resync.ops[2], ReplicaOp::NoOp { .. }), "the held op is resynced as itself: {:?}", resync.ops[2]);
    assert!(bsvc.get("d6").unwrap().is_some(), "the held op was applied");
    assert_eq!((b.local_checkpoint(), b.promotion_pending()), (7, None));
    drop(held);
}

/// A held op that cannot be applied fails the promotion: it
/// may have been acknowledged, so it is never replaced by a gap no-op (which
/// would drop it here and make every copy holding it diverge). The pending
/// marker keeps the promotion due.
#[test]
fn a_held_op_that_cannot_apply_fails_the_promotion_instead_of_becoming_a_noop() {
    let tmp = tempfile::tempdir().unwrap();
    let mapping = json!({"properties": {"n": {"type": "integer"}}});
    let (bsvc, b) = replica(&tmp.path().join("b"), &mapping);
    let op = |seq: i64, source: &str| ReplicaOp::Index { id: format!("h{seq}"), source_json: source.into(), routing: None, seq_no: seq, version: 1, primary_term: TERM };
    b.handle(&req(TERM, -1, vec![op(0, r#"{"n":0}"#), op(2, r#"{"n":2}"#)])).unwrap();
    // Held in the WAL, unprocessed, and not parseable under the mapping.
    b.wal_append_only(&[op(1, r#"{"n":"not a number"}"#)]).unwrap();
    let e = b.promote_to_primary(TERM + 1).expect_err("an unappliable held op fails the promotion");
    assert!(!matches!(e, ReplicationError::StalePrimaryTerm { .. }), "{e}");
    assert_eq!(b.promotion_pending(), Some(TERM + 1), "the promotion stays due");
    assert_eq!(durable_state(&bsvc)["promotion_pending"], TERM + 1);
    assert!(b.local_checkpoint() < 2, "seq 1 was not marked processed as a no-op");
    assert!(b.ops_above(-1).unwrap().iter().any(|o| o.seq_no() == 1 && !matches!(o, ReplicaOp::NoOp { .. })), "the WAL still holds the op");
}

/// A restart replays the replica WAL one record at a time,
/// so an unflushed WAL far bigger than the request breaker still opens with
/// memory bounded by a record, and the breaker never refuses or delays the
/// replay of these acknowledged ops (ES translog recovery has no breaker):
/// it completes while other work holds most of the breaker.
#[test]
fn replaying_a_wal_bigger_than_the_request_breaker_streams_and_is_never_delayed() {
    use elasticsearch_engine::memory::{scoped, Breakers, Kind, RequestScope};
    let tmp = tempfile::tempdir().unwrap();
    let (svc, r) = replica(&tmp.path().join("replica"), &json!({}));
    // Incompressible sources: the WAL on disk is as large as the ops.
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let (batches, per) = (48i64, 25i64);
    for b in 0..batches {
        let ops = (0..per).map(|i| {
            let seq = b * per + i;
            let pad: String = (&mut rng).sample_iter(&rand::distributions::Alphanumeric).take(1024).map(char::from).collect();
            ReplicaOp::Index { id: format!("w{seq}"), source_json: json!({"n": seq, "pad": pad}).to_string(), routing: None, seq_no: seq, version: 1, primary_term: TERM }
        }).collect();
        r.handle(&req(TERM, -1, ops)).unwrap();
    }
    let total = batches * per;
    assert_eq!(r.wal_record_count() as i64, batches);
    drop(r);

    let tiny = Breakers::leak_for_tests(1 << 20);
    tiny.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
    let limit = tiny.get(Kind::Request).limit();
    let wal_bytes: u64 = std::fs::read_dir(crate::shards::copy_of(&svc, 0).shard.path().join("cluster-replica-wal")).unwrap()
        .map(|e| e.unwrap()).filter(|e| e.file_name().to_string_lossy().ends_with(".log")).map(|e| e.metadata().unwrap().len()).sum();
    assert!(wal_bytes as i64 > limit, "the WAL ({wal_bytes} bytes) must exceed the request breaker ({limit})");

    // Most of the breaker is taken by other work for the whole replay: the
    // replay neither waits for it nor fails (its memory is accounted, past
    // the limit, never refused), and counts no trip.
    let held = tiny.reserve_request((limit - 2048) as u64, "competing request").unwrap();
    let tripped = tiny.get(Kind::Request).tripped();
    let scope = RequestScope::new("replica WAL replay", tiny);
    let r = scoped(Some(scope), || ReplicaApplier::open(svc.clone(), 0)).expect("the replay is never refused by the breaker");
    assert_eq!(r.local_checkpoint(), total - 1);
    assert_eq!(r.stats().wal_ops_replayed as i64, total);
    assert_eq!(tiny.get(Kind::Request).tripped(), tripped, "nothing was refused");
    assert_eq!(tiny.get(Kind::Request).used(), (limit - 2048) as u64, "the replay completed with the competing reservation still held, and keeps nothing");
    drop(held);
    assert_eq!(tiny.get(Kind::Request).used(), 0, "nothing of the replay stays reserved");
    for seq in [0, total / 2, total - 1] {
        assert!(svc.get(&format!("w{seq}")).unwrap().is_some(), "w{seq}");
    }
    // The replayed copy keeps applying.
    let next = ReplicaOp::Index { id: "after".into(), source_json: "{}".into(), routing: None, seq_no: total, version: 1, primary_term: TERM };
    assert_eq!(r.handle(&req(TERM, -1, vec![next])).unwrap().local_checkpoint, total);
}

/// A recovery source serves the copy it started from. The node detaches a
/// primary's copy from its index service (relocated away, closed for a
/// reset) while recoveries from it may still be starting or in their
/// translog phase; with the copy captured, the translog phase reads the
/// pinned copy's history, never an empty slot or a later incarnation's
/// translog.
#[test]
fn a_recovery_source_serves_its_own_copy_after_the_service_detached_it() {
    let tmp = tempfile::tempdir().unwrap();
    let svc = create(&tmp.path().join("p"), &json!({}));
    for i in 0..4 {
        index(&svc, &format!("d{i}"), json!({"n": i}));
    }
    let copy = crate::shards::copy_of(&svc, 0);
    let mut source = RecoverySource::start_from_copy(svc.clone(), copy.clone(), 0, TERM, RecoveryOptions::default(), 1).unwrap();
    // The node detaches the copy (as `remove_copy` does) mid-recovery.
    let detached = svc.detach_shard_if_same(&copy).expect("the copy was attached");
    assert!(svc.shard(0).is_none(), "the service slot is empty now");
    let ops = source.next_translog_batch().expect("the translog phase reads its own copy");
    assert_eq!(ops.iter().map(ReplicaOp::seq_no).collect::<Vec<_>>(), vec![2, 3]);
    // Starting one from the detached copy works too (no service lookup).
    let mut late = RecoverySource::start_from_copy(svc.clone(), detached, 0, TERM, RecoveryOptions::default(), 2).unwrap();
    assert_eq!(late.next_translog_batch().unwrap().iter().map(ReplicaOp::seq_no).collect::<Vec<_>>(), vec![3]);
}
