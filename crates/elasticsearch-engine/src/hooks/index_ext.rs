//! Hooks 2, 3 and 8 on [`IndexService`]: replica apply (parses sources
//! against the copy's mapping, then applies through the shard), public
//! source parsing, and file sets for file-based recovery / snapshots.
//! Compiled as a private child module of `crate::index`.

use super::*;
use crate::document::ParsedDoc;
use crate::hooks::changes::{ChangeOp, Durability, ReplicaApplyResult};
use crate::hooks::files::IndexFileSet;

impl IndexCore {
    /// Prepare a bounded replica chunk without changing mappings or invoking
    /// inference/scripts. None means replay the original chunk sequentially,
    /// including its original stale-op and error ordering. Never surface a
    /// speculative parse error before the replication layer checks staleness.
    pub fn prepare_replica_sources(&self, sources: &[(&str, &str)]) -> Option<Vec<ParsedDoc>> {
        use rayon::prelude::*;
        let limits = crate::fieldtypes::limits::Limits::from_settings(|k| self.setting(k));
        let lookup = self.lookup();
        // Vector graph preparation is expensive even for small recovery batches.
        // The replay layer now supports it; do not reject all vector mappings.
        if sources.len() < (if lookup.may_have_vector_fields { 2 } else { 32 }) { return None; }
        let guard = self.meta.read();
        let meta = &*guard;
        let source_mode = crate::synthetic::source_mode(settings::get_str(&meta.settings, "index.mapping.source.mode"),
            &meta.mappings, settings::get_str(&meta.settings, "index.mode"));
        if lookup.may_have_semantic_fields || lookup.has_index_scripts
            || crate::tsdb::is_time_series(&meta.settings)
            || source_mode == crate::synthetic::SourceMode::Synthetic {
            return None;
        }
        static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
        let pool = POOL.get_or_init(|| rayon::ThreadPoolBuilder::new().num_threads(4)
            .thread_name(|i| format!("replica-parse-{i}")).build().expect("replica parse pool"));
        pool.install(|| sources.par_iter().map(|&(id, source)| {
            let value = if source.trim().is_empty() { Value::Object(Map::new()) }
                else { crate::jsonpos::parse_source_value(source).ok()? };
            let (mut parsed, update) = crate::document::parse_with_limits(&value, id, &meta.mappings, &lookup, &limits).ok()?;
            if update.is_some() { return None; }
            finish_parsed(&mut parsed, &value, &meta.mappings, &lookup, &meta.settings).ok()?;
            // The parallel replay path must honor the same source-storage
            // policy as parse_for_replica, while retaining original WAL data.
            if source_mode == crate::synthetic::SourceMode::Disabled {
                parsed.source = None;
                if !parsed.doc.fields.iter().any(|(f, _)| *f == crate::mapping::SOURCE_FIELD) {
                    parsed.doc.fields.push((crate::mapping::SOURCE_FIELD, rustic_core::FieldValue::Keyword(String::new())));
                }
            }
            Some(parsed)
        }).collect())
    }

    /// Hook 8: parse a `_source` against this index's mapping, applying
    /// (and persisting) dynamic mapping updates like the primary write path
    /// does. An empty source (`_source` disabled on the primary) parses as `{}`.
    pub fn parse_for_replica(&self, source_json: &str, id: &str) -> Result<ParsedDoc> {
        let v: Value = if source_json.trim().is_empty() {
            Value::Object(Map::new())
        } else {
            crate::jsonpos::parse_source_value(source_json).map_err(|e| EsError::document_parsing(format!("[{}:{}] failed to parse", e.line(), e.column())).caused_by(EsError::json_parse(&e)))?
        };
        let mut parsed = self.parse_source(&v, id)?.0;
        // `_source` disabled: the operation carries the original source
        // (to rebuild the documents) but it is not stored.
        let disabled = {
            let m = self.meta.read();
            crate::synthetic::source_mode(crate::settings::get_str(&m.settings, "index.mapping.source.mode"), &m.mappings, crate::settings::get_str(&m.settings, "index.mode")) == crate::synthetic::SourceMode::Disabled
        };
        if disabled && !parsed.doc.fields.iter().any(|(f, _)| *f == crate::mapping::SOURCE_FIELD) {
            parsed.source = None;
            parsed.doc.fields.push((crate::mapping::SOURCE_FIELD, rustic_core::FieldValue::Keyword(String::new())));
        }
        // The primary adds this hidden field after parsing the source. Every
        // replica/recovery path must reconstruct it too; it is not in _source.
        if !source_json.trim().is_empty() && crate::tsdb::is_time_series(&self.meta.read().settings) {
            let ids = {
                let meta = self.meta.read();
                self.lookup().tsdb_fields.build_ids(&meta.mappings, &meta.settings, &v)?
            };
            crate::tsdb::push_hidden_fields(&self.lookup(), &ids, &mut parsed.doc);
        }
        Ok(parsed)
    }
}

/// Deprecated whole-index shims of hooks 2/3/12 (the cluster crate pinned
/// to the single-shard engine): they act on the local copy of shard 0.
/// Use the [`IndexShard`] methods instead.
impl IndexService {
    fn shard0(&self) -> Result<Arc<IndexShard>> {
        self.shard(0).ok_or_else(|| EsError::shard_not_found(&self.name, &self.uuid, 0))
    }

    /// Hook 3 on shard 0 (see [`IndexShard::apply_replica_ops`]).
    ///
    /// * `request_term` is the sending primary's term: lower than this
    ///   copy's term → `illegal_state_exception` (409), nothing applied;
    ///   higher → the copy moves to it durably first (hook 4). Each op's
    ///   own term must be `1..=request_term`.
    /// * Idempotent: an op whose seq_no was already processed is a
    ///   `Duplicate`; an index/delete older than what the copy holds for its
    ///   `_id` is `Stale`; no-ops are recorded in the history sidecar.
    /// * Durable per `durability` before returning.
    #[deprecated(note = "use IndexShard::apply_replica_ops on IndexService::shard(n)")]
    pub fn apply_replica_ops(&self, request_term: u64, ops: Vec<ChangeOp>, durability: Durability) -> Result<ReplicaApplyResult> {
        self.shard0()?.apply_replica_ops(request_term, ops, durability)
    }

    /// Hook 12 replica apply on shard 0 (see [`IndexShard::apply_replica_ops_marked`]).
    #[deprecated(note = "use IndexShard::apply_replica_ops_marked on IndexService::shard(n)")]
    pub fn apply_replica_ops_marked(&self, request_term: u64, ops: Vec<ChangeOp>, markers: Vec<Option<String>>, durability: Durability) -> Result<ReplicaApplyResult> {
        self.shard0()?.apply_replica_ops_marked(request_term, ops, markers, durability)
    }

    /// Hook 12: the original result of the last op on `id` that carried
    /// `coordinator_id` (on the shard `id` routes to).
    pub fn retried_result(&self, id: &str, coordinator_id: &str) -> Option<crate::shard::WriteResult> {
        self.route(id, None).ok().and_then(|n| self.shard(n)).and_then(|s| s.retried_result(id, coordinator_id))
    }

    /// Hook 2 (source side) on shard 0: `index_meta.json` + `shard-0/...`
    /// (see [`IndexShard::snapshot_files`] for the per-shard set).
    #[deprecated(note = "use IndexShard::snapshot_files on IndexService::shard(n)")]
    pub fn snapshot_files(&self, include_translog: bool) -> Result<IndexFileSet> {
        self.shard0()?.snapshot_index_files(include_translog)
    }
}

#[cfg(test)]
mod retry_tests {
    #![allow(deprecated)]
    use super::*;
    use crate::hooks::changes::ApplyOutcome;
    use crate::node::Node;
    use crate::index::WriteReq;
    use crate::shard::WriteCond;

    fn svc() -> (tempfile::TempDir, Arc<Node>, Arc<IndexService>) {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let s = node.create_index("r", &json!({})).unwrap();
        (tmp, node, s)
    }

    #[cfg(feature = "mutable-hnsw")]
    #[test]
    fn vector_replica_parser_reaches_parallel_replay_for_small_batches() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let s = node.create_index("vectors", &json!({"mappings":{"properties":{
            "v":{"type":"dense_vector","dims":2,"similarity":"l2_norm",
                 "index_options":{"type":"hnsw"}}
        }}})).unwrap();
        let copy = s.shard(0).unwrap();
        let sources = [("a", r#"{"v":[1,2]}"#), ("b", r#"{"v":[3,4]}"#)];
        let mut parsed = copy.prepare_replica_sources(&sources).expect("vector batches must reach parallel replay");
        for (pd, (id, source)) in parsed.iter().zip(sources) {
            let serial = copy.parse_for_replica(source, id).unwrap();
            assert_eq!(pd.source, serial.source);
            assert_eq!(format!("{:?}", pd.doc), format!("{:?}", serial.doc));
        }
        let ops: Vec<_> = sources.iter().enumerate().map(|(i, (id, source))| crate::shard::ReplayIndex {
            id, source, routing: None, seq_no: i as i64, version: 1,
        }).collect();
        assert!(copy.try_replay_new_batch(&ops, &mut parsed).unwrap());
        assert!(parsed.is_empty());
        assert!(s.get("a").unwrap().is_some());
        assert!(s.get("b").unwrap().is_some());
        let invalid = [("c", r#"{"v":[1,2,3]}"#), sources[1]];
        assert!(copy.prepare_replica_sources(&invalid).is_none(), "parse failures retain ordered fallback");
        let dynamic = [("c", r#"{"v":[1,2],"new_field":1}"#), sources[1]];
        assert!(copy.prepare_replica_sources(&dynamic).is_none(), "mapping changes retain ordered fallback");
    }

    fn cond(m: &str, retry: bool) -> WriteCond {
        WriteCond { coordinator_id: Some(m.into()), is_retry: retry, ..Default::default() }
    }

    fn put(s: &IndexService, id: &str, body: &str, c: WriteCond) -> Result<crate::shard::WriteResult> {
        s.write(vec![WriteReq::Index { id: Some(id.into()), source: body.into(), routing: None, cond: c }], true).remove(0)
    }

    #[test]
    fn retried_create_returns_original_instead_of_conflict() {
        let (_t, _n, s) = svc();
        let first = put(&s, "a", r#"{"n":1}"#, WriteCond { create: true, ..cond("c1", false) }).unwrap();
        assert_eq!((first.result, first.version, first.seq_no), ("created", 1, 0));
        let again = put(&s, "a", r#"{"n":1}"#, WriteCond { create: true, ..cond("c1", true) }).unwrap();
        assert_eq!(again, first);
        assert_eq!(s.shard(0).unwrap().shard.max_seq_no(), 0, "nothing re-applied");
        // Different marker, or no retry bit: a genuine conflict.
        assert_eq!(put(&s, "a", "{}", WriteCond { create: true, ..cond("c2", true) }).unwrap_err().status, 409);
        assert_eq!(put(&s, "a", "{}", WriteCond { create: true, ..cond("c1", false) }).unwrap_err().status, 409);
        assert_eq!(s.retried_result("a", "c1"), Some(first));
    }

    #[test]
    fn retried_cas_delete_and_update_return_applied_result() {
        let (_t, _n, s) = svc();
        put(&s, "a", r#"{"n":1}"#, WriteCond::default()).unwrap();
        let cas = |m: &str, retry: bool| WriteCond { if_seq_no: Some(0), if_primary_term: Some(1), ..cond(m, retry) };
        let upd = put(&s, "a", r#"{"n":2}"#, cas("u1", false)).unwrap();
        assert_eq!((upd.result, upd.seq_no, upd.version), ("updated", 1, 2));
        // The CAS would now fail (seq 0 is gone); the retry returns the applied result.
        assert_eq!(put(&s, "a", r#"{"n":2}"#, cas("u1", true)).unwrap(), upd);
        assert_eq!(put(&s, "a", r#"{"n":3}"#, cas("u2", true)).unwrap_err().status, 409);

        let dcas = |m: &str, retry: bool| WriteCond { if_seq_no: Some(1), if_primary_term: Some(1), ..cond(m, retry) };
        let del = |c: WriteCond| s.write(vec![WriteReq::Delete { id: "a".into(), routing: None, cond: c }], true).remove(0);
        let d = del(dcas("d1", false)).unwrap();
        assert_eq!(d.result, "deleted");
        assert_eq!(del(dcas("d1", true)).unwrap(), d);
        assert_eq!(s.shard(0).unwrap().shard.max_seq_no(), d.seq_no);
        assert_eq!(del(dcas("d2", true)).unwrap_err().status, 409);
    }

    #[test]
    fn retried_scripted_update_is_not_reapplied() {
        let (_t, _n, s) = svc();
        put(&s, "a", r#"{"n":1}"#, WriteCond::default()).unwrap();
        let body = json!({"script": {"source": "ctx._source.n += 1"}});
        let (w1, _) = s.update("a", &body, None, 0, cond("s1", false)).unwrap();
        assert_eq!(s.get("a").unwrap().unwrap().version, 2);
        let (w2, src) = s.update("a", &body, None, 0, cond("s1", true)).unwrap();
        assert_eq!(w2, w1);
        assert_eq!(serde_json::from_str::<Value>(&src.unwrap()).unwrap()["n"], 2);
        assert_eq!(s.get("a").unwrap().unwrap().version, w1.version, "script ran once");
        // A fresh (non-retry) request applies again.
        let (w3, _) = s.update("a", &body, None, 0, cond("s2", false)).unwrap();
        assert_eq!(w3.version, w1.version + 1);
    }

    #[test]
    fn replica_keeps_markers_and_dedups_retries() {
        let (_t, _n, s) = svc();
        let op = |seq: i64| ChangeOp::Index { id: "a".into(), source_json: r#"{"n":1}"#.into(), routing: None, seq_no: seq, version: 1, primary_term: 1 };
        let r = s.apply_replica_ops_marked(1, vec![op(0)], vec![Some("m".into())], Durability::Request).unwrap();
        assert_eq!(r.outcomes, vec![ApplyOutcome::Applied]);
        // The same logical op under another seq_no is a no-op success.
        let r = s.apply_replica_ops_marked(1, vec![op(5)], vec![Some("m".into())], Durability::Request).unwrap();
        assert_eq!(r.outcomes, vec![ApplyOutcome::Duplicate]);
        assert_eq!(s.shard(0).unwrap().shard.retry_markers(&[0, 1]), vec![Some("m".into()), None]);
        // Promoted: a retried create returns the original seq_no / version.
        let w = put(&s, "a", "{}", WriteCond { create: true, ..cond("m", true) }).unwrap();
        assert_eq!((w.seq_no, w.version, w.result), (0, 1, "created"));
        let e = s.apply_replica_ops_marked(1, vec![op(6)], vec![], Durability::Request).unwrap_err();
        assert_eq!(e.status, 400);
    }
}
