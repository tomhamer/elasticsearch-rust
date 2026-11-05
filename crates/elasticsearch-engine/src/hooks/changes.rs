//! Operation history types (hook 5) and replica-apply types (hook 3).

use crate::translog::{TranslogOp, AdmittedBatch};
use std::sync::Arc;
use serde::{Deserialize, Serialize};

/// One operation of a shard's history (ES `Translog.Operation`), as
/// returned by [`Shard::changes_since`](crate::shard::Shard::changes_since)
/// and accepted by
/// [`IndexService::apply_replica_ops`](crate::index::IndexService::apply_replica_ops).
/// Same shape as the cluster crate's `ReplicaOp`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeOp {
    Index {
        id: String,
        /// Raw `_source` JSON (empty when `_source` is disabled).
        source_json: String,
        routing: Option<String>,
        seq_no: i64,
        version: u64,
        primary_term: u64,
    },
    Delete {
        id: String,
        seq_no: i64,
        version: u64,
        primary_term: u64,
    },
    /// Fills a seq_no without touching any document.
    NoOp {
        seq_no: i64,
        primary_term: u64,
        reason: String,
    },
}

impl ChangeOp {
    /// Reserve every copied payload before cloning an operation for a consumer.
    pub(crate) fn try_copy(&self)->crate::error::Result<(Self,crate::memory::Reservation)> {
        let bytes=match self {
            Self::Index{id,source_json,routing,..} => id.len().saturating_add(source_json.len()).saturating_add(routing.as_ref().map_or(0,String::len)),
            Self::Delete{id,..}=>id.len(), Self::NoOp{reason,..}=>reason.len(),
        };
        let memory=crate::memory::breakers().reserve_request(bytes as u64,"change operation copy")?;
        fn text(s:&str)->crate::error::Result<String> {
            let mut out=String::new();out.try_reserve_exact(s.len()).map_err(|e|crate::error::EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
            out.push_str(s);Ok(out)
        }
        let value=match self {
            Self::Index{id,source_json,routing,seq_no,version,primary_term}=>Self::Index{id:text(id)?,source_json:text(source_json)?,routing:routing.as_deref().map(text).transpose()?,seq_no:*seq_no,version:*version,primary_term:*primary_term},
            Self::Delete{id,seq_no,version,primary_term}=>Self::Delete{id:text(id)?,seq_no:*seq_no,version:*version,primary_term:*primary_term},
            Self::NoOp{seq_no,primary_term,reason}=>Self::NoOp{seq_no:*seq_no,primary_term:*primary_term,reason:text(reason)?},
        };
        Ok((value,memory))
    }
    pub fn seq_no(&self) -> i64 {
        match self {
            ChangeOp::Index { seq_no, .. } | ChangeOp::Delete { seq_no, .. } | ChangeOp::NoOp { seq_no, .. } => *seq_no,
        }
    }

    pub fn primary_term(&self) -> u64 {
        match self {
            ChangeOp::Index { primary_term, .. } | ChangeOp::Delete { primary_term, .. } | ChangeOp::NoOp { primary_term, .. } => *primary_term,
        }
    }

    /// A translog record with its primary term.
    pub fn from_translog(op: TranslogOp, primary_term: u64) -> ChangeOp {
        match op {
            TranslogOp::Index { id, source_json, seq_no, version, routing } => {
                ChangeOp::Index { id, source_json, routing, seq_no: seq_no as i64, version, primary_term }
            }
            TranslogOp::Delete { id, seq_no, version } => ChangeOp::Delete { id, seq_no: seq_no as i64, version, primary_term },
        }
    }

    /// The translog record of an index/delete (`None` for a no-op, which is
    /// kept in the history sidecar instead).
    pub fn to_translog(&self) -> Option<TranslogOp> {
        match self {
            ChangeOp::Index { id, source_json, routing, seq_no, version, .. } => {
                Some(TranslogOp::Index { id: id.clone(), source_json: source_json.clone(), seq_no: *seq_no as u64, version: *version, routing: routing.clone() })
            }
            ChangeOp::Delete { id, seq_no, version, .. } => Some(TranslogOp::Delete { id: id.clone(), seq_no: *seq_no as u64, version: *version }),
            ChangeOp::NoOp { .. } => None,
        }
    }
}

/// Immutable operation reference that keeps its decoded allocation admitted.
#[derive(Clone)]
pub struct ChangeItem { batch: Arc<ChangeStorage>, index: usize }
struct ChangeStorage { ops: AdmittedBatch<ChangeOp>, _memory: crate::memory::Reservation }
impl std::ops::Deref for ChangeItem {
    type Target = ChangeOp;
    fn deref(&self) -> &ChangeOp { &self.batch.ops[self.index] }
}
impl std::fmt::Debug for ChangeItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { std::fmt::Debug::fmt(&**self,f) }
}
impl PartialEq for ChangeItem { fn eq(&self, other:&Self)->bool { **self == **other } }
impl Eq for ChangeItem {}
impl PartialEq<ChangeOp> for ChangeItem { fn eq(&self, other:&ChangeOp)->bool { &**self == other } }

/// A point-in-time, seq_no-ordered copy of a range of history. Each yielded
/// item retains the allocation owner even after the iterator is dropped.
pub struct ChangesSnapshot {
    storage: Arc<ChangeStorage>,
    next: usize,
    end: usize,
    pub from_seq_no: i64,
    pub to_seq_no: i64,
    pub local_checkpoint: i64,
    pub max_seq_no: i64,
    total: usize,
}
impl std::fmt::Debug for ChangesSnapshot {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result { f.debug_list().entries(self.storage.ops.iter()).finish() }
}
impl ChangesSnapshot {
    pub(crate) fn new(ops: AdmittedBatch<ChangeOp>, from_seq_no:i64, to_seq_no:i64, local_checkpoint:i64,max_seq_no:i64)->crate::error::Result<Self> {
        let memory=crate::memory::breakers().reserve_request((std::mem::size_of::<ChangeStorage>()+2*std::mem::size_of::<usize>()) as u64,"change snapshot owner")?;
        let total=ops.len();
        Ok(Self {storage:Arc::new(ChangeStorage{ops,_memory:memory}),next:0,end:total,from_seq_no,to_seq_no,local_checkpoint,max_seq_no,total})
    }
    pub fn remaining_last(&self)->Option<&ChangeOp> {self.storage.ops[self.next..self.end].last()}
    pub fn total_ops(&self)->usize {self.total}
    pub fn len(&self)->usize {self.end-self.next}
    pub fn is_empty(&self)->bool {self.len()==0}
    pub(crate) fn limit(&mut self, max_ops:usize,max_bytes:u64,size:impl Fn(&ChangeOp)->u64) {
        let mut bytes=0u64; let mut end=self.next;
        for op in self.storage.ops[self.next..self.end].iter().take(max_ops) {
            let n=size(op); if end>self.next && bytes.saturating_add(n)>max_bytes {break;}
            bytes=bytes.saturating_add(n);end+=1;
        }
        self.end=end;
    }
}
impl Iterator for ChangesSnapshot {
    type Item=ChangeItem;
    fn next(&mut self)->Option<ChangeItem> {
        if self.next==self.end{return None;}
        let index=self.next;self.next+=1;Some(ChangeItem{batch:self.storage.clone(),index})
    }
    fn size_hint(&self)->(usize,Option<usize>){(self.len(),Some(self.len()))}
}
impl ExactSizeIterator for ChangesSnapshot {}
impl std::ops::Index<usize> for ChangesSnapshot {
    type Output=ChangeOp;
    fn index(&self,index:usize)->&ChangeOp {&self.storage.ops[self.next..self.end][index]}
}
impl PartialEq<Vec<ChangeItem>> for ChangesSnapshot {
    fn eq(&self, other:&Vec<ChangeItem>)->bool { self.len()==other.len() && self.storage.ops[self.next..self.end].iter().zip(other).all(|(a,b)|a==&**b) }
}


/// Durability of a replica apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Durability {
    /// fsync the translog (and history sidecar) before returning (ES
    /// `index.translog.durability: request`, the default).
    #[default]
    Request,
    /// `write(2)` only: survives a process crash, not a power loss (ES
    /// `async`).
    Async,
}

/// What happened to one replicated op.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApplyOutcome {
    /// Applied to the index and recorded in the translog.
    Applied,
    /// Its seq_no was already processed (retry / overlap): nothing done.
    Duplicate,
    /// A newer op for the same `_id` was already applied (out-of-order
    /// delivery): recorded in the history only, the document is untouched.
    Stale,
    /// A no-op: recorded in the history only.
    NoOp,
}

/// Result of [`IndexService::apply_replica_ops`](crate::index::IndexService::apply_replica_ops).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplicaApplyResult {
    /// One per input op, in input order.
    pub outcomes: Vec<ApplyOutcome>,
    /// Highest seq_no such that every seq_no at or below it was processed.
    pub local_checkpoint: i64,
    pub max_seq_no: i64,
    /// The copy's primary term after the call.
    pub primary_term: u64,
}

#[cfg(test)]
mod admitted_history_tests {
    use super::*;
    #[test]
    fn yielded_change_keeps_payload_owner_after_snapshot_drop() {
        let budget=crate::memory::Breakers::leak_for_tests(1<<20);
        budget.apply_settings(&|key|(key=="indices.breaker.total.use_real_memory").then(||"false".into()));
        let memory=budget.reserve_request(4096,"history payload").unwrap();
        let mut ops=AdmittedBatch::new(budget).unwrap();
        ops.push_reserved(ChangeOp::NoOp{seq_no:4,primary_term:2,reason:"x".repeat(4096)},memory).unwrap_or_else(|_|panic!("roomy admission"));
        let mut snapshot=ChangesSnapshot::new(ops,4,4,4,4).unwrap();
        let item=snapshot.next().unwrap();
        let retained=item.clone();
        drop(snapshot);drop(item);
        assert_eq!(retained.seq_no(),4);
        assert!(budget.get(crate::memory::Kind::Request).used()>=4096);
        drop(retained);
        assert_eq!(budget.get(crate::memory::Kind::Request).used(),0);
    }
}
