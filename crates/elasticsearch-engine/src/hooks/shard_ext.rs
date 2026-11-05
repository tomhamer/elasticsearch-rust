//! Hooks 3, 4 and 5 on [`Shard`]: primary terms, durable replica apply,
//! operation history (`changes_since`) and history retention. Compiled as
//! a private child module of `crate::shard` (see `hooks/mod.rs`).

use super::*;
use crate::translog::AdmittedBatch;
use crate::hooks::changes::{ApplyOutcome, ChangeOp, ChangesSnapshot, Durability, ReplicaApplyResult};
use crate::hooks::files::FileSource;
use crate::hooks::history::{SeqNoState, ShardHistory, SideLog, SideRec, TermHistory, TERMS_FILE};
use crate::hooks::retention::{Leases, RetentionLease, RetentionLock};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Where a change feed stopped reading a shard's translog (see
/// [`Shard::changes_from_cursor`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChangesCursor {
    generation: u64,
    offset: u64,
    next_seq_no: i64,
}

/// Hook state that is read without the shard write lock.
pub(crate) struct ShardExt {
    term: AtomicU64,
    terms: Mutex<TermHistory>,
    leases: Arc<Mutex<Leases>>,
    /// Hook 12: last marked op per `_id` (in memory, bounded).
    retries: Mutex<RetryTable>,
    /// Terms of replicated ops noted before their replay
    /// ([`Shard::note_op_terms`]), until they are processed: the side log
    /// record of the same term can be trimmed by a flush in between.
    noted: Mutex<HashMap<i64, u64>>,
    /// The processed checkpoint the last complete refresh made visible
    /// (ES's refreshed checkpoint, `RefreshListeners`).
    refreshed: std::sync::atomic::AtomicI64,
}

/// Marked ops remembered per shard before the oldest are forgotten.
const RETRY_TABLE_CAP: usize = 65_536;

#[derive(Clone, Debug)]
struct RetryRec {
    marker: String,
    seq_no: i64,
    version: u64,
    term: u64,
    result: &'static str,
    deleted: bool,
}

#[derive(Default)]
struct RetryTable {
    map: HashMap<String, RetryRec>,
    order: std::collections::VecDeque<(String, i64)>,
}

impl RetryTable {
    fn put(&mut self, id: &str, rec: RetryRec) {
        self.order.push_back((id.to_string(), rec.seq_no));
        self.map.insert(id.to_string(), rec);
        while self.map.len() > RETRY_TABLE_CAP {
            let Some((i, s)) = self.order.pop_front() else { break };
            if self.map.get(&i).is_some_and(|r| r.seq_no == s) {
                self.map.remove(&i);
            }
        }
        if self.order.len() > RETRY_TABLE_CAP * 2 {
            self.order.retain(|(i, s)| self.map.get(i).is_some_and(|r| r.seq_no == *s));
        }
    }
}

impl ShardExt {
    pub(crate) fn open(path: &Path) -> Result<ShardExt> {
        let t = TermHistory::load(path)?;
        Ok(ShardExt { term: AtomicU64::new(t.current), terms: Mutex::new(t), leases: Arc::new(Mutex::new(Leases::open(path)?)), retries: Mutex::new(RetryTable::default()), noted: Mutex::new(HashMap::new()), refreshed: std::sync::atomic::AtomicI64::new(-1) })
    }
}

pub(crate) fn new_history(path: &Path, state: Option<&SeqNoState>, committed_seq_no: i64) -> Result<ShardHistory> {
    Ok(ShardHistory::new(SideLog::open(path)?, state, committed_seq_no))
}

/// Translog ops not reflected in the commit: above its local checkpoint
/// and not among the seq_nos it had already processed, read lazily (the
/// translog may hold far more than memory allows at once).
pub(crate) fn ops_to_replay(translog: &Translog, state: Option<&SeqNoState>, committed_seq_no: i64) -> Result<crate::translog::TranslogReplay> {
    let (cp, above): (i64, HashSet<i64>) = match state {
        Some(s) => (s.local_checkpoint, s.processed_above.iter().copied().collect()),
        None => (committed_seq_no, HashSet::new()),
    };
    translog.replay_stream((cp + 1).max(0) as u64, above)
}

/// The primary term stored in a live document.
pub(crate) fn doc_term(core: &CoreIndex, addr: &DocAddress) -> u64 {
    core.all_leaves().get(addr.segment_id as usize).and_then(|l| l.i64_value(PRIMARY_TERM_FIELD, addr.doc_id)).map(|t| t.max(1) as u64).unwrap_or(1)
}

/// Highest seq_no this copy knows for `id` (live doc, version map entry,
/// or delete tombstone).
pub(crate) fn known_seq(ws: &WriteState, live_seq: Option<i64>, id: &str) -> Option<i64> {
    let v = ws.versions.get(id).map(|e| e.seq_no);
    let t = ws.hist.tombstones.get(id).map(|t| t.0);
    [live_seq, v, t].into_iter().flatten().max()
}

fn stale_term(term: u64, current: u64) -> EsError {
    EsError::new(409, "illegal_state_exception", format!("operation primary term [{term}] is too old (current [{current}])"))
}

/// One replicated op with its parsed document (index ops).
pub(crate) struct PreparedReplicaOp {
    pub op: ChangeOp,
    pub parsed: Option<(Document, Vec<String>)>,
    /// Hook 12 coordinator marker of the op (when the sender supplied one).
    pub marker: Option<String>,
}

#[allow(clippy::too_many_arguments)]
fn add_meta_fields(doc: &mut Document, id: &str, source: String, version: u64, seq_no: i64, term: u64, routing: Option<String>, ignored: Vec<String>) {
    doc.fields.push((ID_FIELD, FieldValue::Keyword(id.to_string())));
    // The parse may have set the stored form (`vector_source::stored_form`).
    if !doc.fields.iter().any(|(f, _)| *f == SOURCE_FIELD) {
        doc.fields.push((SOURCE_FIELD, FieldValue::Keyword(source)));
    }
    doc.fields.push((VERSION_FIELD, FieldValue::I64(version as i64)));
    doc.fields.push((SEQ_NO_FIELD, FieldValue::I64(seq_no)));
    doc.fields.push((PRIMARY_TERM_FIELD, FieldValue::I64(term as i64)));
    if let Some(r) = routing {
        doc.fields.push((ROUTING_FIELD, FieldValue::Keyword(r)));
    }
    for f in ignored {
        doc.fields.push((IGNORED_FIELD, FieldValue::Keyword(f)));
    }
}

impl Shard {
    // ------------------------------------------------------------ hook 4

    /// The copy's operating primary term (1 until bumped).
    pub fn primary_term(&self) -> u64 {
        self.ext.term.load(Ordering::Acquire)
    }

    /// Move to a higher primary term (failover / promotion). Durable
    /// (fsynced) before it returns; ops assigned from now on carry `term`.
    /// Lower terms are rejected, equal is a no-op.
    pub fn set_primary_term(&self, term: u64) -> Result<()> {
        let ws = self.write.lock();
        self.bump_term_locked(&ws, term)
    }

    fn bump_term_locked(&self, ws: &WriteState, term: u64) -> Result<()> {
        let cur = self.primary_term();
        if term < cur {
            return Err(stale_term(term, cur));
        }
        if term == cur {
            return Ok(());
        }
        let mut t = self.ext.terms.lock();
        let mut next = t.clone();
        next.bump(term, ws.seq_no + 1);
        next.persist(&self.path)?;
        *t = next;
        self.ext.term.store(term, Ordering::Release);
        Ok(())
    }

    /// The term history (which term assigned which seq_nos).
    pub fn term_history(&self) -> TermHistory {
        self.ext.terms.lock().clone()
    }

    /// Primary term of the op with `seq_no` (exact for ops still in the
    /// history; otherwise from the term ranges).
    pub fn primary_term_of(&self, seq_no: i64) -> u64 {
        let ws = self.write.lock();
        self.term_of_locked(&ws, seq_no)
    }

    /// Record the primary terms of replicated ops before they are replayed
    /// ([`Shard::replay_index`], [`Shard::replay_delete`],
    /// [`Shard::replay_staged`] and the new-batch path stamp the term this
    /// copy knows for a seq_no): an op whose term is not the term history's
    /// goes into the history side log, so the copy stores the op's own
    /// `_primary_term`, as the primary did (ES applies a replica op with
    /// `op.primaryTerm()`). The replication layer owns durability: its log
    /// replays the ops, and their terms, after a crash.
    pub fn note_op_terms(&self, ops: &[(i64, u64)]) -> Result<()> {
        let mut ws = self.write.lock();
        let side: Vec<SideRec> = {
            let terms = self.ext.terms.lock();
            ops.iter()
                .filter(|(seq_no, term)| ws.hist.side.terms.get(seq_no).map_or(terms.term_of(*seq_no), |(t, _)| *t) != *term)
                .map(|&(seq_no, term)| SideRec { seq_no, term, noop: None })
                .collect()
        };
        {
            // Held in memory until the op is processed (a flush may trim
            // the side log generation before the replay stamps the doc);
            // entries of processed ops are dropped here.
            let mut noted = self.ext.noted.lock();
            let checkpoint = ws.hist.checkpoint;
            noted.retain(|seq_no, _| *seq_no > checkpoint && !ws.hist.above.contains(seq_no));
            noted.extend(side.iter().map(|r| (r.seq_no, r.term)));
        }
        if side.is_empty() {
            return Ok(());
        }
        let generation = ws.translog.generation();
        ws.hist.side.append(generation, &side, false)
    }

    /// Every operation up to this seq_no is visible to searches (the
    /// processed checkpoint as of the last refresh that published every
    /// completed write).
    pub fn refreshed_checkpoint(&self) -> i64 {
        self.ext.refreshed.load(Ordering::Acquire)
    }

    /// A search waits on this copy (a fleet search's checkpoints): it stays
    /// search-active, so its scheduled refresh runs.
    pub fn note_search_wait(&self) {
        self.sched.note_search();
    }

    pub(super) fn note_refreshed_checkpoint(&self, checkpoint: i64) {
        self.ext.refreshed.fetch_max(checkpoint, Ordering::AcqRel);
    }

    pub(super) fn term_of_locked(&self, ws: &WriteState, seq_no: i64) -> u64 {
        if let Some(t) = self.ext.noted.lock().get(&seq_no) {
            return *t;
        }
        if let Some((t, _)) = ws.hist.side.terms.get(&seq_no) {
            return *t;
        }
        if let Some((t, _, _)) = ws.hist.side.noops.get(&seq_no) {
            return *t;
        }
        self.ext.terms.lock().term_of(seq_no)
    }

    /// `_primary_term` of the live document `id` (what `if_primary_term`
    /// is compared with).
    pub fn doc_primary_term(&self, id: &str) -> Option<u64> {
        // A document in a pooled memtable (not checked in yet).
        if let Some(pending) = self.pending_doc(id) {
            return Some(pending.primary_term);
        }
        let core = self.core.read();
        Self::lookup(&core, id).map(|(a, _, _)| doc_term(&core, &a))
    }

    // ----------------------------------------------------------- hook 12

    /// The original result of the last op for `id` that carried
    /// `coordinator_id`, if it is still remembered (bounded, in memory).
    pub fn retry_lookup(&self, id: &str, coordinator_id: &str) -> Option<WriteResult> {
        let t = self.ext.retries.lock();
        t.map.get(id).filter(|r| r.marker == coordinator_id).map(|r| WriteResult { id: id.to_string(), version: r.version, seq_no: r.seq_no, primary_term: r.term, result: r.result })
    }

    /// Markers of the ops with these seq_nos (to ship to replicas next to
    /// the ops; `None` where the op carried no marker or was forgotten).
    pub fn retry_markers(&self, seq_nos: &[i64]) -> Vec<Option<String>> {
        let t = self.ext.retries.lock();
        let by_seq: HashMap<i64, &str> = t.map.values().map(|r| (r.seq_no, r.marker.as_str())).collect();
        seq_nos.iter().map(|s| by_seq.get(s).map(|m| m.to_string())).collect()
    }

    /// Remember `res` as the result of the op marked `coordinator_id`.
    pub fn retry_record(&self, coordinator_id: &str, res: &WriteResult, deleted: bool) {
        self.ext.retries.lock().put(&res.id, RetryRec { marker: coordinator_id.to_string(), seq_no: res.seq_no, version: res.version, term: res.primary_term, result: res.result, deleted });
    }

    /// Write path, under the write lock: a retry whose effect is already the
    /// current state of the `_id` returns the original result.
    pub(super) fn retry_hit(&self, cond: &WriteCond, id: &str, cur: Option<VersionEntry>) -> Option<WriteResult> {
        if !cond.is_retry {
            return None;
        }
        let m = cond.coordinator_id.as_deref()?;
        let t = self.ext.retries.lock();
        let r = t.map.get(id).filter(|r| r.marker == m)?;
        let c = cur?;
        (c.seq_no == r.seq_no && c.deleted == r.deleted).then(|| WriteResult { id: id.to_string(), version: r.version, seq_no: r.seq_no, primary_term: r.term, result: r.result })
    }

    pub(super) fn retry_note(&self, cond: &WriteCond, res: &WriteResult, deleted: bool) {
        if let Some(m) = &cond.coordinator_id {
            self.retry_record(m, res, deleted);
        }
    }

    // ------------------------------------------------------- checkpoints

    /// Highest seq_no such that every seq_no at or below it is processed.
    pub fn local_checkpoint(&self) -> i64 {
        self.write.lock().hist.checkpoint
    }

    /// Seq_no of the last commit (flush).
    pub fn committed_seq_no(&self) -> i64 {
        self.write.lock().committed_seq_no
    }

    /// Current seq_no bookkeeping (what the next commit will persist).
    pub fn seq_no_state(&self) -> SeqNoState {
        self.write.lock().hist.state()
    }

    /// Bytes of the current translog generation (what a flush would
    /// commit); unlike [`Shard::translog_size`] it excludes generations
    /// kept only for history retention.
    pub fn uncommitted_translog_size(&self) -> u64 {
        let ws = self.write.lock();
        std::fs::metadata(self.path.join(TRANSLOG_DIR).join(crate::translog::file_name(ws.translog.generation()))).map(|m| m.len()).unwrap_or(0)
    }

    // ------------------------------------------------------- retention

    /// Retain history `>= from_seq_no` until the lock is dropped.
    pub fn acquire_retention_lock(&self, from_seq_no: i64, holder: &str) -> RetentionLock {
        // Under the write lock: serialises with a flush deciding what to trim.
        let _ws = self.write.lock();
        Leases::lock(&self.ext.leases, from_seq_no, holder.to_string())
    }

    /// Add a durable retention lease (fails if `id` exists).
    pub fn add_retention_lease(&self, id: &str, retaining_seq_no: i64, source: &str) -> Result<RetentionLease> {
        let _ws = self.write.lock();
        let l = RetentionLease { id: id.to_string(), retaining_seq_no, timestamp_millis: crate::date::now_millis(), source: source.to_string() };
        self.ext.leases.lock().upsert(l.clone(), true)?;
        Ok(l)
    }

    /// Create or advance a durable retention lease (never moves backwards).
    pub fn renew_retention_lease(&self, id: &str, retaining_seq_no: i64, source: &str) -> Result<RetentionLease> {
        let _ws = self.write.lock();
        let l = RetentionLease { id: id.to_string(), retaining_seq_no, timestamp_millis: crate::date::now_millis(), source: source.to_string() };
        self.ext.leases.lock().upsert(l.clone(), false)?;
        Ok(l)
    }

    pub fn remove_retention_lease(&self, id: &str) -> Result<bool> {
        self.ext.leases.lock().remove(id)
    }

    /// Expire durable leases not renewed within `period_millis` (see
    /// [`Leases::expire`]): an abandoned CCR follower would otherwise pin
    /// this copy's translog forever. Returns the removed ids.
    pub fn expire_retention_leases(&self, period_millis: i64) -> Result<Vec<String>> {
        let now = crate::date::now_millis();
        // Cheap check first: the write lock only when something expires.
        if !self.ext.leases.lock().any_expired(now, period_millis) {
            return Ok(Vec::new());
        }
        let _ws = self.write.lock();
        self.ext.leases.lock().expire(now, period_millis)
    }

    pub fn retention_leases(&self) -> Vec<RetentionLease> {
        self.ext.leases.lock().list()
    }

    /// In-memory retention locks: (retained seq_no, holder).
    pub fn retention_locks(&self) -> Vec<(i64, String)> {
        self.ext.leases.lock().locks()
    }

    /// Lowest seq_no retained by any lease or lock.
    pub fn min_retained_seq_no(&self) -> Option<i64> {
        self.ext.leases.lock().min_retained()
    }

    /// Before writing a commit's `shard_state.json` (flush / force merge):
    /// note the rolled generation's max seq_no and return the state to persist.
    pub(super) fn commit_seqno_state(&self, ws: &mut WriteState, next_gen: u64) -> Option<SeqNoState> {
        let max = ws.seq_no;
        ws.hist.gen_max_seq.insert(next_gen.saturating_sub(1), max);
        ws.hist.prune_tombstones();
        Some(ws.hist.state())
    }

    /// After a commit: delete translog generations (and sidecars) below
    /// `next_gen`, except those that may hold retained ops.
    pub(super) fn trim_translog_retained(&self, ws: &mut WriteState, next_gen: u64) -> Result<()> {
        let keep_from = match self.ext.leases.lock().min_retained() {
            None => next_gen,
            Some(r) => {
                let mut keep = next_gen;
                for (g, _) in crate::translog::list_generations(&self.path.join(TRANSLOG_DIR))? {
                    if g >= next_gen {
                        break;
                    }
                    match ws.hist.gen_max_seq.get(&g) {
                        Some(&m) if m < r => continue,
                        _ => {
                            keep = g;
                            break;
                        }
                    }
                }
                keep
            }
        };
        ws.translog.trim_to(keep_from)?;
        ws.hist.side.trim_below(keep_from)?;
        ws.hist.gen_max_seq.retain(|g, _| *g >= keep_from);
        Ok(())
    }

    // ------------------------------------------------------------ hook 5

    /// Every op with seq_no `>= from_seq_no` (see [`Shard::changes_range`]),
    /// failing if any seq_no in `[from_seq_no, local checkpoint]` is no
    /// longer in the history.
    pub fn changes_since(&self, from_seq_no: i64) -> Result<ChangesSnapshot> {
        self.changes_range(from_seq_no, i64::MAX, true)
    }

    /// Ops with `from <= seq_no <= to`, ascending, each seq_no once, read
    /// from the translog (index/delete) and the history sidecar (no-ops,
    /// per-op terms). With `require_full_range`, a gap in
    /// `[from, min(to, local checkpoint)]` (history trimmed: hold a
    /// retention lock/lease) is a `missing_history_operations_exception`.
    /// Ops above the local checkpoint are included when present.
    pub fn changes_range(&self, from_seq_no: i64, to_seq_no: i64, require_full_range: bool) -> Result<ChangesSnapshot> {
        let from = from_seq_no.max(0);
        let to = to_seq_no;
        let (_metadata_memory, files, mut ops, overrides, terms, cp, max) = {
            let ws = self.write.lock();
            let files: Vec<PathBuf> = crate::translog::list_generations(&self.path.join(TRANSLOG_DIR))?
                .into_iter()
                .filter(|(g, _)| ws.hist.gen_max_seq.get(g).is_none_or(|m| *m >= from))
                .map(|(_, p)| p)
                .collect();
            let mut noops=AdmittedBatch::new(crate::memory::breakers())?;
            if from <= to {
                for (seq,(term,reason,_)) in ws.hist.side.noops.range(from..=to) {
                    let memory=crate::memory::breakers().reserve_request(reason.len() as u64,"change no-op reason")?;
                    let value=ChangeOp::NoOp{seq_no:*seq,primary_term:*term,reason:reason.clone()};
                    noops.push_reserved(value,memory).map_err(|(e,_,_)|e)?;
                }
            }
            let terms=self.ext.terms.lock();
            let metadata_bytes=(ws.hist.side.terms.len() as u64).saturating_mul(128)
                .saturating_add((terms.ranges.len() as u64).saturating_mul(32));
            let metadata_memory=crate::memory::breakers().reserve_request(metadata_bytes,"change history term snapshot")?;
            let overrides: HashMap<i64, u64> = ws.hist.side.terms.iter().filter(|(s, _)| **s >= from && **s <= to).map(|(s, (t, _))| (*s, *t)).collect();
            (metadata_memory, files, noops, overrides, terms.clone(), ws.hist.checkpoint, ws.seq_no)
        };
        for p in files {
            crate::translog::scan_from_admitted(&p,0,|op,_| {
                let seq=op.seq_no() as i64;
                if seq>=from && seq<=to {
                    let term=overrides.get(&seq).copied().unwrap_or_else(||terms.term_of(seq));
                    op.move_into(&mut ops,|op|ChangeOp::from_translog(op,term)).map_err(|(e,_)|e)?;
                }
                Ok(true)
            })?;
        }
        ops.try_stable_sort_by_key(crate::memory::breakers(), |o| o.seq_no())?;
        ops.dedup_by_key(|o| o.seq_no());
        if require_full_range {
            let upto = to.min(cp);
            let mut expect = from;
            for o in &ops {
                if o.seq_no() > upto || o.seq_no() != expect {
                    break;
                }
                expect += 1;
            }
            if expect <= upto {
                return Err(EsError::new(
                    500,
                    "missing_history_operations_exception",
                    format!("Not all operations between from_seqno [{from}] and to_seqno [{upto}] found; expected seqno [{expect}]; history is retained only above a retention lock or lease"),
                ));
            }
        }
        ChangesSnapshot::new(ops, from, to, cp, max)
    }

    /// Up to `max_ops` ops with `from <= seq_no <= to`, ascending and
    /// contiguous, for a change feed that reads a shard repeatedly (CCR
    /// shard follow tasks). The translog is decoded incrementally from
    /// `cursor` (where the previous call stopped, valid when it ended just
    /// below `from`), so a feed that keeps up reads every record once
    /// instead of re-reading whole generations. No-ops come from the
    /// history sidecar. When the history at `from` is not contiguous in the
    /// translog (out-of-order records) it falls back to
    /// [`Shard::changes_range`] (which fails on trimmed history).
    pub fn changes_from_cursor(&self, cursor: &mut Option<ChangesCursor>, from: i64, to: i64, max_ops: usize) -> Result<ChangesSnapshot> {
        self.changes_from_cursor_sized(cursor, from, to, max_ops, u64::MAX, |_| 0)
    }

    /// [`Shard::changes_from_cursor`] that also stops before the batch's
    /// estimated size (`size_of` per op) would exceed `max_bytes` (the
    /// first op is always returned); the cursor stays on the first op not
    /// returned.
    pub fn changes_from_cursor_sized(&self, cursor: &mut Option<ChangesCursor>, from: i64, to: i64, max_ops: usize, max_bytes: u64, size_of: impl Fn(&ChangeOp) -> u64) -> Result<ChangesSnapshot> {
        let from = from.max(0);
        if to < from || max_ops == 0 {
            return ChangesSnapshot::new(AdmittedBatch::new(crate::memory::breakers())?,from,to,-1,-1);
        }
        let (_snapshot_memory, gens, noops, overrides, terms, cp, max) = {
            let ws = self.write.lock();
            let gens: Vec<(u64, PathBuf, bool)> = crate::translog::list_generations(&self.path.join(TRANSLOG_DIR))?
                .into_iter()
                .map(|(g, p)| {
                    let relevant = ws.hist.gen_max_seq.get(&g).is_none_or(|m| *m >= from);
                    (g, p, relevant)
                })
                .collect();
            let terms=self.ext.terms.lock();
            let bytes=ws.hist.side.noops.range(from..=to).fold(0u64,|n,(_,(_,reason,_))|n.saturating_add(reason.len() as u64).saturating_add(256))
                .saturating_add((ws.hist.side.terms.len() as u64).saturating_mul(128))
                .saturating_add((terms.ranges.len() as u64).saturating_mul(32));
            let snapshot_memory=crate::memory::breakers().reserve_request(bytes,"change feed sidecar snapshot")?;
            let noops: BTreeMap<i64, ChangeOp> =
                ws.hist.side.noops.range(from..=to).map(|(s, (t, r, _))| (*s, ChangeOp::NoOp { seq_no: *s, primary_term: *t, reason: r.clone() })).collect();
            let overrides: HashMap<i64, u64> = ws.hist.side.terms.iter().filter(|(s, _)| **s >= from && **s <= to).map(|(s, (t, _))| (*s, *t)).collect();
            (snapshot_memory, gens, noops, overrides, terms.clone(), ws.hist.checkpoint, ws.seq_no)
        };
        let start = match *cursor {
            Some(c) if c.next_seq_no == from && gens.iter().any(|(g, _, _)| *g == c.generation) => Some((c.generation, c.offset)),
            _ => gens.iter().find(|(_, _, relevant)| *relevant).map(|(g, _, _)| (*g, 0)),
        };
        let mut out = AdmittedBatch::new(crate::memory::breakers())?;
        let mut expect = from;
        let mut pos = start.unwrap_or((0, 0));
        let mut stop = false;
        let bytes = std::cell::Cell::new(0u64);
        // Room for one more op of `size` in the batch?
        let fits = |out: &AdmittedBatch<ChangeOp>, size: u64| out.is_empty() || bytes.get().saturating_add(size) <= max_bytes;
        let fill = |expect: &mut i64, upto: i64, out: &mut AdmittedBatch<ChangeOp>| -> Result<bool> {
            while *expect < upto && out.len() < max_ops {
                match noops.get(expect) {
                    Some(n) => {
                        let size = size_of(n);
                        if !fits(out, size) {
                            return Ok(false);
                        }
                        bytes.set(bytes.get() + size);
                        let (value,memory)=n.try_copy()?;
                        out.push_reserved(value,memory).map_err(|(e,_,_)|e)?;
                        *expect += 1;
                    }
                    None => break,
                }
            }
            Ok(true)
        };
        if let Some((g0, off0)) = start {
            for (g, path, _) in gens.iter().filter(|(g, _, _)| *g >= g0) {
                if stop {
                    break;
                }
                let off = if *g == g0 { off0 } else { 0 };
                pos = (*g, off);
                crate::translog::scan_from_admitted(path, off, |op, end| {
                    let s = op.seq_no() as i64;
                    if s < expect {
                        pos = (*g, end);
                        return Ok(true);
                    }
                    if s > to {
                        stop = true;
                        return Ok(false);
                    }
                    if !fill(&mut expect, s, &mut out)? || s != expect || out.len() >= max_ops {
                        stop = true;
                        return Ok(false);
                    }
                    let term = overrides.get(&s).copied().unwrap_or_else(|| terms.term_of(s));
                    let mut one=AdmittedBatch::new(crate::memory::breakers())?;
                    op.move_into(&mut one,|op|ChangeOp::from_translog(op,term)).map_err(|(e,_)|e)?;
                    let size = size_of(&one[0]);
                    if !fits(&out, size) {
                        stop = true;
                        return Ok(false);
                    }
                    bytes.set(bytes.get() + size);
                    out.try_append(one).map_err(|(e,_)|e)?;
                    expect += 1;
                    pos = (*g, end);
                    Ok(true)
                })?;
            }
        }
        if !stop {
            fill(&mut expect, to.saturating_add(1), &mut out)?;
        }
        if out.is_empty() && from <= to.min(cp) {
            // History at `from` is not where the cursor expects it.
            *cursor = None;
            let mut result=self.changes_range(from,to,true)?;
            result.limit(max_ops,max_bytes,size_of);
            return Ok(result);
        }
        let result=ChangesSnapshot::new(out,from,to,cp,max)?;
        *cursor = Some(ChangesCursor { generation: pos.0, offset: pos.1, next_seq_no: expect });
        Ok(result)
    }

    // ------------------------------------------------------------ hook 3

    /// Apply replicated ops (see [`IndexService::apply_replica_ops`](crate::index::IndexService::apply_replica_ops)).
    pub(crate) fn apply_replica(&self, request_term: u64, ops: Vec<PreparedReplicaOp>, durability: Durability) -> Result<ReplicaApplyResult> {
        for p in &ops {
            let t = p.op.primary_term();
            if p.op.seq_no() < 0 || t == 0 || t > request_term {
                return Err(EsError::illegal_argument(format!("invalid replicated op: seq_no [{}], primary term [{t}] (request term [{request_term}])", p.op.seq_no())));
            }
        }
        // A failed engine applies nothing (ES `ensureOpen`).
        if let Some(error) = self.tragic_error() {
            return Err(Self::tragic_exception(&error));
        }
        let t0 = std::time::Instant::now();
        self.files_changed();
        let _version_slots = super::VersionMapSlots::new(self, ops.len())?;
        let version_keys = super::registry::Keys::prepare(ops.iter().map(|p| match &p.op {
            ChangeOp::Index { id, .. } | ChangeOp::Delete { id, .. } => id.as_str(),
            ChangeOp::NoOp { .. } => "",
        }))?;
        let mut ws = self.write.lock();
        self.bump_term_locked(&ws, request_term)?;
        let mut core = self.core.write();
        let terms = self.ext.terms.lock().clone();
        let mut log: Vec<TranslogOp> = Vec::new();
        let mut side: Vec<SideRec> = Vec::new();
        let mut outcomes = Vec::with_capacity(ops.len());
        let mut done: Vec<i64> = Vec::new();
        let mut seen: HashSet<i64> = HashSet::new();
        let (mut n_index, mut n_delete) = (0u64, 0u64);
        // Decide every op first, against this copy's state plus the ops of
        // this batch decided before it; install nothing until the batch is
        // in the translog. An op installed before a failed append would stay
        // in the index (and its version map) with no translog record, be
        // committed by the next flush and leave a checkpoint hole.
        enum Install {
            Index { version_key: super::registry::Key, id: String, source_json: String, routing: Option<String>, version: u64, seq_no: i64, term: u64, parsed: (Document, Vec<String>), marker: Option<String> },
            Delete { version_key: super::registry::Key, id: String, version: u64, seq_no: i64, term: u64, marker: Option<String> },
        }
        let mut installs: Vec<Install> = Vec::new();
        let mut decided: HashMap<String, i64> = HashMap::new();
        let mut max_seq_no = ws.seq_no;
        for (p, version_key) in ops.into_iter().zip(version_keys) {
            let s = p.op.seq_no();
            let term = p.op.primary_term();
            if ws.hist.is_processed(s) || !seen.insert(s) {
                outcomes.push(ApplyOutcome::Duplicate);
                continue;
            }
            if let Some(m) = &p.marker {
                // A marked op this copy already holds under another seq_no.
                let t = self.ext.retries.lock();
                if t.map.get(match &p.op { ChangeOp::Index { id, .. } | ChangeOp::Delete { id, .. } => id.as_str(), _ => "" }).is_some_and(|r| &r.marker == m && r.seq_no != s) {
                    outcomes.push(ApplyOutcome::Duplicate);
                    continue;
                }
            }
            let marker = p.marker.clone();
            let side_term = |side: &mut Vec<SideRec>| {
                if term != terms.term_of(s) {
                    side.push(SideRec { seq_no: s, term, noop: None });
                }
            };
            // The seq_no this copy holds for `id`, counting the ops of this
            // batch decided so far.
            let current = |core: &CoreIndex, ws: &WriteState, decided: &HashMap<String, i64>, id: &str| match decided.get(id) {
                Some(&seq_no) => Some(seq_no),
                None => known_seq(ws, Self::lookup(core, id).map(|f| f.2), id),
            };
            let outcome = match p.op {
                ChangeOp::NoOp { reason, .. } => {
                    side.push(SideRec { seq_no: s, term, noop: Some(reason) });
                    ApplyOutcome::NoOp
                }
                ChangeOp::Index { id, source_json, routing, version, .. } => {
                    self.resolve_buffered_delete(&core, &id);
                    let outcome = match current(&core, &ws, &decided, &id) {
                        Some(c) if c == s => ApplyOutcome::Duplicate,
                        Some(c) if c > s => ApplyOutcome::Stale,
                        _ => {
                            let parsed = p.parsed.ok_or_else(|| EsError::internal("replica index op without a parsed document"))?;
                            decided.insert(id.clone(), s);
                            installs.push(Install::Index { version_key, id: id.clone(), source_json: source_json.clone(), routing: routing.clone(), version, seq_no: s, term, parsed, marker });
                            ApplyOutcome::Applied
                        }
                    };
                    if outcome != ApplyOutcome::Duplicate {
                        log.push(TranslogOp::Index { id, source_json, seq_no: s as u64, version, routing });
                        side_term(&mut side);
                    }
                    outcome
                }
                ChangeOp::Delete { id, version, .. } => {
                    self.resolve_buffered_delete(&core, &id);
                    let outcome = match current(&core, &ws, &decided, &id) {
                        Some(c) if c == s => ApplyOutcome::Duplicate,
                        Some(c) if c > s => ApplyOutcome::Stale,
                        _ => {
                            decided.insert(id.clone(), s);
                            installs.push(Install::Delete { version_key, id: id.clone(), version, seq_no: s, term, marker });
                            ApplyOutcome::Applied
                        }
                    };
                    if outcome != ApplyOutcome::Duplicate {
                        log.push(TranslogOp::Delete { id, seq_no: s as u64, version });
                        side_term(&mut side);
                    }
                    outcome
                }
            };
            max_seq_no = max_seq_no.max(s);
            done.push(s);
            outcomes.push(outcome);
        }
        // Durable before anything is installed or reported processed.
        let generation = ws.translog.generation();
        if !log.is_empty() {
            if let Err(error) = ws.translog.append_many(&log) {
                // Nothing of the batch is installed; the engine fails itself
                // (translog tragic event) and the copy is failed.
                self.mark_tragic(&format!("translog append failed: {error}"));
                return Err(error);
            }
            // Let `sync_translog` (group commit) cover these appends too.
            *self.tail.lock() = Some(ws.translog.sync_point());
        }
        ws.seq_no = max_seq_no;
        for install in installs {
            let installed = match install {
                Install::Index { version_key, id, source_json, routing, version, seq_no, term, parsed: (mut doc, ignored), marker } => {
                    let found = Self::lookup(&core, &id);
                    add_meta_fields(&mut doc, &id, source_json, version, seq_no, term, routing, ignored);
                    core.replace_document(found.as_ref().map(|found| &found.0), &doc).map(|_| {
                        ws.versions.insert(version_key, VersionEntry { version, seq_no, deleted: false, primary_term: term });
                        ws.hist.tombstones.remove(&id);
                        if let Some(m) = &marker {
                            self.ext.retries.lock().put(&id, RetryRec { marker: m.clone(), seq_no, version, term, result: if found.is_some() { "updated" } else { "created" }, deleted: false });
                        }
                        n_index += 1;
                    })
                }
                Install::Delete { version_key, id, version, seq_no, term, marker } => {
                    let found = Self::lookup(&core, &id);
                    let deleted = match found { Some((a, _, _)) => core.delete_doc(&a).map(|_| ()), None => Ok(()) };
                    deleted.map(|_| {
                        ws.versions.insert(version_key, VersionEntry { version, seq_no, deleted: true, primary_term: term });
                        ws.hist.tombstones.insert(id.clone(), (seq_no, version));
                        if let Some(m) = &marker {
                            self.ext.retries.lock().put(&id, RetryRec { marker: m.clone(), seq_no, version, term, result: if found.is_some() { "deleted" } else { "not_found" }, deleted: true });
                        }
                        n_delete += 1;
                    })
                }
            };
            if let Err(error) = installed {
                // In the translog but not in the index: the copy is failed.
                self.mark_tragic(&format!("replica install failed after its translog append: {error}"));
                return Err(error.into());
            }
        }
        // The history side log (no-ops, terms) is part of the durable
        // record: its failure fails the engine as a translog failure does.
        if let Err(error) = ws.hist.side.append(generation, &side, false) {
            self.mark_tragic(&format!("history side log append failed: {error}"));
            return Err(error);
        }
        if durability == Durability::Request {
            if !log.is_empty() {
                #[cfg(test)]
                let injected = self.fail_syncs.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1)).is_ok();
                #[cfg(not(test))]
                let injected = false;
                let synced = if injected { Err(EsError::internal("injected translog fsync failure")) } else { ws.translog.fsync() };
                if let Err(error) = synced {
                    self.mark_tragic(&format!("translog fsync failed: {error}"));
                    return Err(error);
                }
            }
            if !side.is_empty() {
                if let Err(error) = ws.hist.side.sync() {
                    self.mark_tragic(&format!("history side log fsync failed: {error}"));
                    return Err(error);
                }
            }
        }
        for s in done {
            ws.hist.mark(s);
        }
        ws.hist.prune_tombstones();
        if core.num_writing_docs() >= self.seal_threshold() {
            if self.bg.served.load(Ordering::Acquire) {
                self.bg.want_seal.store(true, Ordering::Release);
            } else if let Err(e) = core.seal() {
                tracing::error!("seal failed: {e}");
            }
        }
        if n_index + n_delete > 0 {
            self.sched.note_write();
        }
        self.counters.index_total.fetch_add(n_index, Ordering::Relaxed);
        self.counters.delete_total.fetch_add(n_delete, Ordering::Relaxed);
        Counters::add_time(&self.counters.index_time_nanos, t0.elapsed());
        Ok(ReplicaApplyResult { outcomes, local_checkpoint: ws.hist.checkpoint, max_seq_no: ws.seq_no, primary_term: self.primary_term() })
    }

    // ------------------------------------------------------------ hook 2

    /// Pin this shard's files for a transfer: the last core commit (mapped),
    /// `shard_state.json`, the term history and optionally the translog +
    /// sidecar files (their current length). Keys are prefixed with
    /// `prefix`. Also returns [committed seq_no, local checkpoint of the
    /// commit, current local checkpoint, current max seq_no], the primary
    /// term and a retention lock on history above the commit.
    ///
    /// Under the write state and the core read lock only the commit's files
    /// are mapped (no read of their contents): their checksums are computed
    /// by [`IndexFileSet::build`](crate::hooks::files::IndexFileSet) once
    /// the caller dropped the locks (a mutable graph checkpoint can be tens
    /// of GB). The mappings pin the files: a later commit deleting them does
    /// not change what is read.
    pub(crate) fn capture_files(&self, include_translog: bool, prefix: &str) -> Result<(crate::hooks::files::CapturedSources, [i64; 4], u64, RetentionLock)> {
        // Metadata path joins and the formatted core prefix are transient;
        // their independent owner lasts through source registry construction.
        let _paths=rustic_core::admission::shared().reserve((self.path.as_os_str().as_encoded_bytes().len() as u64).saturating_mul(8).saturating_add((prefix.len() as u64).saturating_mul(2)).saturating_add(1024),"capture path construction")?;
        let mut ws = self.write.lock();
        let core = self.core.read();
        let state_bytes = crate::hooks::files::read_metadata_file(&self.path.join(STATE_FILE))?;
        let _state_decode_memory=rustic_core::admission::shared().reserve((state_bytes.len() as u64).saturating_mul(16).saturating_add(4096),"captured shard state decode")?;
        let st: ShardState = serde_json::from_slice(&state_bytes).map_err(|e| EsError::internal(format!("corrupt shard state: {e}")))?;
        let commit_cp = st.seqno.as_ref().map(|s| s.local_checkpoint).unwrap_or(st.committed_seq_no);
        let lock = Leases::lock(&self.ext.leases, commit_cp + 1, "file-transfer".to_string());
        let pinned = core.pin_commit()?;
        drop(core);
        let mut m = crate::hooks::files::CapturedSources::new()?;
        m.reserve_entry(prefix.len()+STATE_FILE.len(),0)?;
        m.values.insert(format!("{prefix}{STATE_FILE}"), FileSource::Bytes(state_bytes));
        if let Some(bytes)=crate::hooks::files::read_optional_metadata_file(&self.path.join(TERMS_FILE))? {
            m.reserve_entry(prefix.len()+TERMS_FILE.len(),0)?;
            m.values.insert(format!("{prefix}{TERMS_FILE}"),FileSource::Bytes(bytes));
        }
        let core_prefix = format!("{prefix}core-{}/", st.core_generation);
        for f in pinned.iter() {
            m.reserve_entry(core_prefix.len()+f.name.len(),self.path.as_os_str().as_encoded_bytes().len()+f.name.len()+128)?;
            let bytes = (self.path.as_os_str().len() as u64).saturating_add(core_prefix.len() as u64)
                .saturating_add((f.name.len() as u64).saturating_mul(2)).saturating_add(1024);
            let memory = rustic_core::admission::shared().reserve(bytes, "transfer mapped source path")?;
            let on_disk = self.path.join(format!("core-{}", st.core_generation)).join(&f.name);
            // The identity is the mapped file's, taken when it was pinned:
            // `seg_N.del` can be replaced (same name and length) by a commit
            // before the checksums are computed off the locks.
            m.values.insert(format!("{core_prefix}{}", f.name), FileSource::PinnedAt(f.clone(), on_disk, memory));
        }
        if include_translog {
            let generations = crate::translog::list_generations_admitted(&self.path.join(TRANSLOG_DIR))?;
            for (g, p) in generations.iter() {
                let len = std::fs::metadata(&p)?.len();
                m.reserve_entry(prefix.len()+TRANSLOG_DIR.len()+64,p.as_os_str().as_encoded_bytes().len())?;
                m.values.insert(format!("{prefix}{TRANSLOG_DIR}/{}", crate::translog::file_name(*g)), FileSource::Disk(p.clone(), len));
            }
            ws.hist.side.sync()?;
            let histories = ws.hist.side.files_admitted()?;
            for (g, p) in histories.iter() {
                let len = std::fs::metadata(&p)?.len();
                m.reserve_entry(prefix.len()+crate::hooks::history::HISTORY_DIR.len()+64,p.as_os_str().as_encoded_bytes().len())?;
                m.values.insert(format!("{prefix}{}/{}", crate::hooks::history::HISTORY_DIR, crate::hooks::history::side_file_name(*g)), FileSource::Disk(p.clone(), len));
            }
        }
        Ok((m, [st.committed_seq_no, commit_cp, ws.hist.checkpoint, ws.seq_no], self.primary_term(), lock))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::reserved_core_fields;

    fn schema() -> Vec<CoreField> {
        let mut s = reserved_core_fields();
        s.push(CoreField::new(8, "title", rustic_core::types::FieldType::Text));
        s
    }

    fn doc(title: &str) -> Document {
        let mut d = Document::new();
        d.add_text(8, title);
        d
    }

    fn idx(id: &str) -> WriteOp {
        idx_cond(id, WriteCond::default())
    }

    fn idx_cond(id: &str, cond: WriteCond) -> WriteOp {
        WriteOp::Index { id: id.into(), source: format!("{{\"title\":\"{id}\"}}"), doc: doc(id), ignored: vec![], routing: None, cond, auto_id: false }
    }

    fn rindex(id: &str, seq: i64, term: u64) -> PreparedReplicaOp {
        PreparedReplicaOp {
            op: ChangeOp::Index { id: id.into(), source_json: format!("{{\"title\":\"{id}\"}}"), routing: None, seq_no: seq, version: seq as u64 + 1, primary_term: term },
            parsed: Some((doc(id), vec![])),
            marker: None,
        }
    }

    fn rdelete(id: &str, seq: i64, term: u64) -> PreparedReplicaOp {
        PreparedReplicaOp { op: ChangeOp::Delete { id: id.into(), seq_no: seq, version: seq as u64 + 1, primary_term: term }, parsed: None, marker: None }
    }

    /// Reopen like `IndexService::open` does (replaying the returned ops).
    fn reopen(path: &Path) -> Shard {
        let (s, ops) = Shard::open(path, schema()).unwrap();
        // Borrow the replay batch so its allocation owner remains alive for
        // every op. Only the owned replay API inputs need fixture copies.
        for op in ops.iter() {
            match op {
                TranslogOp::Index { id, source_json, seq_no, version, routing } => s.replay_index(id.clone(), source_json.clone(), doc(id), vec![], routing.clone(), *seq_no as i64, *version).unwrap(),
                TranslogOp::Delete { id, seq_no, version } => s.replay_delete(id, *seq_no as i64, *version).unwrap(),
            }
        }
        s
    }

    #[test]
    fn replayed_ops_keep_their_own_primary_term() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        // A replica that stayed a replica through a failover: its engine
        // still says term 1, the ops say 2 (and an older one, re-sent, 1).
        s.note_op_terms(&[(0, 1), (1, 2), (2, 2)]).unwrap();
        s.replay_index("a".into(), "{}".into(), doc("a"), vec![], None, 0, 1).unwrap();
        s.replay_index("b".into(), "{}".into(), doc("b"), vec![], None, 1, 1).unwrap();
        s.replay_delete("a", 2, 2).unwrap();
        s.replay_index("c".into(), "{}".into(), doc("c"), vec![], None, 3, 1).unwrap();
        assert_eq!(s.doc_primary_term("b"), Some(2), "the op's term, not the copy's");
        assert_eq!(s.doc_primary_term("c"), Some(1), "no recorded term: the term history's");
        assert_eq!(s.primary_term_of(2), 2);
        // The staged (concurrent) replay path too.
        s.note_op_terms(&[(4, 5)]).unwrap();
        s.replay_staged(vec![crate::shard::StagedReplay::Index { id: "d".into(), source: "{}".into(), doc: doc("d"), ignored: vec![], routing: None, seq_no: 4, version: 1 }]).unwrap();
        assert_eq!(s.doc_primary_term("d"), Some(5));
        // A flush between the note and the replay (it may trim the side
        // log generation the note went to) keeps the noted term.
        s.note_op_terms(&[(5, 6)]).unwrap();
        s.flush().unwrap();
        s.replay_index("e".into(), "{}".into(), doc("e"), vec![], None, 5, 1).unwrap();
        assert_eq!(s.doc_primary_term("e"), Some(6));
    }

    #[test]
    fn terms_are_per_op_durable_and_fenced() {
        let tmp = tempfile::tempdir().unwrap();
        {
            let s = Shard::create(tmp.path(), "i", schema()).unwrap();
            s.apply("i", "u", vec![idx("a"), idx("b")], true);
            assert_eq!(s.primary_term(), 1);
            s.set_primary_term(3).unwrap();
            assert_eq!(s.set_primary_term(2).unwrap_err().kind, "illegal_state_exception");
            s.apply("i", "u", vec![idx("c")], true);
            assert_eq!(s.doc_primary_term("a"), Some(1));
            assert_eq!(s.doc_primary_term("c"), Some(3));
            // CAS compares with the document's own term.
            let r = s.apply("i", "u", vec![idx_cond("a", WriteCond { if_seq_no: Some(0), if_primary_term: Some(1), ..Default::default() })], true);
            assert!(r[0].is_ok(), "{:?}", r[0]);
            let e = s.apply("i", "u", vec![idx_cond("c", WriteCond { if_seq_no: Some(2), if_primary_term: Some(1), ..Default::default() })], true).remove(0).unwrap_err();
            assert!(e.reason.contains("primary term [3]"), "{}", e.reason);
        }
        // Crash (no flush): terms and per-op terms survive.
        let s = reopen(tmp.path());
        assert_eq!(s.primary_term(), 3);
        let ch: Vec<(i64, u64)> = s.changes_since(0).unwrap().map(|o| (o.seq_no(), o.primary_term())).collect();
        assert_eq!(ch, vec![(0, 1), (1, 1), (2, 3), (3, 3)]);
        assert_eq!(s.doc_primary_term("c"), Some(3));
        assert_eq!(s.local_checkpoint(), 3);
    }

    #[test]
    fn retention_lock_survives_flush_and_force_merge() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        s.apply("i", "u", (0..5).map(|i| idx(&format!("d{i}"))).collect(), true);
        let lock = s.acquire_retention_lock(2, "test");
        s.flush().unwrap();
        s.apply("i", "u", vec![idx("d5"), WriteOp::Delete { id: "d0".into(), cond: WriteCond::default() }], true);
        s.force_merge(1, false).unwrap();
        s.flush().unwrap();
        let ch: Vec<i64> = s.changes_since(2).unwrap().map(|o| o.seq_no()).collect();
        assert_eq!(ch, vec![2, 3, 4, 5, 6]);
        assert!(matches!(s.changes_since(2).unwrap().last(), Some(op) if matches!(&*op, ChangeOp::Delete { .. })));
        // Advancing releases older generations at the next flush.
        lock.advance_to(5);
        s.apply("i", "u", vec![idx("d7")], true);
        s.flush().unwrap();
        assert_eq!(s.changes_since(5).unwrap().map(|o| o.seq_no()).collect::<Vec<_>>(), vec![5, 6, 7]);
        assert_eq!(s.changes_since(2).unwrap_err().kind, "missing_history_operations_exception");
        drop(lock);
        s.apply("i", "u", vec![idx("d8")], true);
        s.flush().unwrap();
        assert_eq!(s.changes_since(0).unwrap_err().kind, "missing_history_operations_exception");
        assert_eq!(s.changes_since(9).unwrap().count(), 0);
        // Durable leases survive a restart.
        s.add_retention_lease("replica-1", 9, "peer recovery").unwrap();
        assert!(s.add_retention_lease("replica-1", 9, "x").is_err());
        assert!(s.renew_retention_lease("replica-1", 3, "x").is_err(), "leases never go backwards");
        drop(s);
        let s = reopen(tmp.path());
        assert_eq!(s.min_retained_seq_no(), Some(9));
        s.apply("i", "u", vec![idx("d9"), idx("d10")], true);
        s.flush().unwrap();
        assert_eq!(s.changes_since(9).unwrap().map(|o| o.seq_no()).collect::<Vec<_>>(), vec![9, 10]);
        assert!(s.remove_retention_lease("replica-1").unwrap());
    }

    #[test]
    fn stale_retention_leases_expire_except_peer_recovery() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Shard::create(tmp.path(), "i", schema()).unwrap();
        s.add_retention_lease("follower", 0, "ccr").unwrap();
        s.add_retention_lease("replica", 0, "peer recovery").unwrap();
        // Renewed recently: kept.
        assert!(s.expire_retention_leases(60_000).unwrap().is_empty());
        assert_eq!(s.min_retained_seq_no(), Some(0));
        // Older than the period: the ccr lease goes, the peer-recovery one stays.
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert_eq!(s.expire_retention_leases(1).unwrap(), vec!["follower".to_string()]);
        assert_eq!(s.retention_leases().iter().map(|l| l.id.as_str()).collect::<Vec<_>>(), vec!["replica"]);
        // Durable: the expiry survives a restart.
        drop(s);
        let s = reopen(tmp.path());
        assert_eq!(s.retention_leases().len(), 1);
    }

    #[test]
    fn replica_apply_is_idempotent_ordered_and_durable() {
        let tmp = tempfile::tempdir().unwrap();
        {
            let s = Shard::create(tmp.path(), "i", schema()).unwrap();
            // Out of order: delete of x (seq 3) before its index (seq 1).
            let r = s.apply_replica(2, vec![rindex("a", 0, 1), rdelete("x", 3, 2)], Durability::Request).unwrap();
            assert_eq!(r.outcomes, vec![ApplyOutcome::Applied, ApplyOutcome::Applied]);
            assert_eq!((r.local_checkpoint, r.max_seq_no, r.primary_term), (0, 3, 2));
            s.add_retention_lease("peer", 0, "test").unwrap(); // keep the history for changes_since below
            s.flush().unwrap(); // tombstone of x must survive the flush
            let r = s.apply_replica(2, vec![rindex("x", 1, 1), rindex("a", 0, 1), PreparedReplicaOp { op: ChangeOp::NoOp { seq_no: 2, primary_term: 2, reason: "gap".into() }, parsed: None, marker: None }], Durability::Request).unwrap();
            assert_eq!(r.outcomes, vec![ApplyOutcome::Stale, ApplyOutcome::Duplicate, ApplyOutcome::NoOp]);
            assert_eq!(r.local_checkpoint, 3);
            assert!(s.get("x").unwrap().is_none(), "stale index must not resurrect a deleted doc");
            // Stale primary terms are fenced; ops above the request term are invalid.
            assert_eq!(s.apply_replica(1, vec![rindex("b", 4, 1)], Durability::Request).unwrap_err().kind, "illegal_state_exception");
            assert!(s.apply_replica(2, vec![rindex("b", 4, 3)], Durability::Request).is_err());
            let r = s.apply_replica(2, vec![rindex("b", 4, 2), rindex("b", 4, 2)], Durability::Async).unwrap();
            assert_eq!(r.outcomes, vec![ApplyOutcome::Applied, ApplyOutcome::Duplicate]);
            s.sync_translog().unwrap();
            // Crash without flush.
        }
        let s = reopen(tmp.path());
        assert_eq!(s.primary_term(), 2);
        assert_eq!(s.local_checkpoint(), 4);
        assert!(s.get("x").unwrap().is_none());
        assert_eq!(s.get("b").unwrap().unwrap().seq_no, 4);
        assert_eq!(s.doc_primary_term("b"), Some(2));
        assert_eq!(s.doc_primary_term("a"), Some(1));
        // Retried batch after restart: everything is a duplicate.
        let r = s.apply_replica(2, vec![rindex("a", 0, 1), rindex("b", 4, 2)], Durability::Request).unwrap();
        assert_eq!(r.outcomes, vec![ApplyOutcome::Duplicate, ApplyOutcome::Duplicate]);
        // The copy's history (for when it is promoted) has every op with its term.
        let ch: Vec<(i64, u64, &'static str)> = s
            .changes_since(1)
            .unwrap()
            .map(|o| (o.seq_no(), o.primary_term(), match &*o { ChangeOp::Index { .. } => "i", ChangeOp::Delete { .. } => "d", ChangeOp::NoOp { .. } => "n" }))
            .collect();
        assert_eq!(ch, vec![(1, 1, "i"), (2, 2, "n"), (3, 2, "d"), (4, 2, "i")]);
        // Promotion: new primary writes continue the sequence in a new term.
        s.set_primary_term(3).unwrap();
        let w = s.apply("i", "u", vec![idx("c")], true).remove(0).unwrap();
        assert_eq!(w.seq_no, 5);
        assert_eq!(s.doc_primary_term("c"), Some(3));
    }
}
