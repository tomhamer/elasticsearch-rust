//! The wire form of replicated operations.

use elasticsearch_engine::translog::TranslogOp;
use serde::{Deserialize, Serialize};

/// One operation as shipped from the primary to a replica (ES
/// `Translog.Operation`). Carries the primary's `_seq_no`/`_version` so the
/// replica reproduces them exactly.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicaOp {
    Index {
        id: String,
        /// Raw `_source` JSON (empty when the primary has `_source` disabled).
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
    /// Fills a seq_no without touching any document (ES no-op: failed
    /// writes that consumed a seq_no, primary-replica resync gaps).
    NoOp {
        seq_no: i64,
        primary_term: u64,
        reason: String,
    },
}

impl ReplicaOp {
    /// Convert an engine translog record; the engine does not record the
    /// primary term per op, so the caller supplies the current one.
    pub fn from_translog(op: TranslogOp, primary_term: u64) -> ReplicaOp {
        match op {
            TranslogOp::Index { id, source_json, seq_no, version, routing } => {
                ReplicaOp::Index { id, source_json, routing, seq_no: seq_no as i64, version, primary_term }
            }
            TranslogOp::Delete { id, seq_no, version } => ReplicaOp::Delete { id, seq_no: seq_no as i64, version, primary_term },
        }
    }

    /// Convert a batch of translog records.
    pub fn from_translog_batch(ops: elasticsearch_engine::translog::AdmittedBatch<TranslogOp>, primary_term: u64)
        -> std::result::Result<elasticsearch_engine::translog::AdmittedBatch<ReplicaOp>, (elasticsearch_engine::error::EsError, elasticsearch_engine::translog::AdmittedBatch<TranslogOp>)> {
        ops.try_map(elasticsearch_engine::memory::breakers(), |o| ReplicaOp::from_translog(o, primary_term))
    }

    pub fn seq_no(&self) -> i64 {
        match self {
            ReplicaOp::Index { seq_no, .. } | ReplicaOp::Delete { seq_no, .. } | ReplicaOp::NoOp { seq_no, .. } => *seq_no,
        }
    }

    pub fn primary_term(&self) -> u64 {
        match self {
            ReplicaOp::Index { primary_term, .. } | ReplicaOp::Delete { primary_term, .. } | ReplicaOp::NoOp { primary_term, .. } => *primary_term,
        }
    }

    /// The document id (`None` for no-ops).
    pub fn id(&self) -> Option<&str> {
        match self {
            ReplicaOp::Index { id, .. } | ReplicaOp::Delete { id, .. } => Some(id),
            ReplicaOp::NoOp { .. } => None,
        }
    }

    /// Rough payload size, for batching by bytes.
    pub fn estimated_bytes(&self) -> usize {
        32 + match self {
            ReplicaOp::Index { id, source_json, routing, .. } => id.len() + source_json.len() + routing.as_ref().map_or(0, String::len),
            ReplicaOp::Delete { id, .. } => id.len(),
            ReplicaOp::NoOp { reason, .. } => reason.len(),
        }
    }
}

/// A replication request (`indices:data/write/replica` equivalent).
#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct ReplicationRequest {
    pub index: String,
    pub shard: u32,
    /// Allocation id of the target copy (the replica rejects requests for
    /// another allocation, e.g. after it was re-created).
    pub target_allocation_id: String,
    /// Term of the primary sending the request.
    pub primary_term: u64,
    /// The primary's global checkpoint (piggybacked).
    pub global_checkpoint: i64,
    /// ES `max_seq_no_of_updates_or_deletes`: highest seq_no of an op that
    /// overwrote or deleted a doc (recorded by the replica; -1 = none).
    pub max_seq_no_of_updates: i64,
    /// The primary's mapping (`GET _mapping` body form) as JSON text, sent
    /// when it changed since the last request to this copy.
    pub mapping_json: Option<String>,
    pub ops: elasticsearch_engine::translog::AdmittedBatch<ReplicaOp>,
    /// The cluster `mapping_version` of `mapping_json` (0 = unknown): a
    /// replica that already holds a newer committed mapping keeps it.
    /// Appended: absent from an older sender's request (bincode ignores the
    /// trailing bytes on an older receiver).
    pub mapping_version: u64,
    #[serde(skip)]
    pub memory: crate::replication::decode::DecodeMemory,
}

impl ReplicationRequest {
    /// Admit scalar copies before constructing the outgoing envelope. Operations
    /// arrive with their own owner, which is moved without copying.
    pub fn try_new(index: &str, shard: u32, target_allocation_id: &str, primary_term: u64,
        global_checkpoint: i64, max_seq_no_of_updates: i64, mapping_json: Option<&str>,
        ops: elasticsearch_engine::translog::AdmittedBatch<ReplicaOp>, mapping_version: u64)
        -> elasticsearch_engine::error::Result<Self> {
        let memory = crate::replication::decode::DecodeMemory::reserve_texts(
            elasticsearch_engine::memory::breakers(), &[index, target_allocation_id, mapping_json.unwrap_or("")])?;
        fn copy(value: &str) -> elasticsearch_engine::error::Result<String> {
            let mut result = String::new();
            result.try_reserve_exact(value.len()).map_err(|e| elasticsearch_engine::error::EsError::new(
                429, "circuit_breaking_exception", format!("replication envelope text allocation: {e}")))?;
            result.push_str(value);
            Ok(result)
        }
        Ok(Self { index: copy(index)?, shard, target_allocation_id: copy(target_allocation_id)?, primary_term,
            global_checkpoint, max_seq_no_of_updates, mapping_json: mapping_json.map(copy).transpose()?,
            ops, mapping_version, memory })
    }

    pub fn try_clone(&self) -> elasticsearch_engine::error::Result<Self> {
        Self::try_new(&self.index, self.shard, &self.target_allocation_id, self.primary_term,
            self.global_checkpoint, self.max_seq_no_of_updates, self.mapping_json.as_deref(),
            ReplicaOp::try_clone_batch(&self.ops)?, self.mapping_version)
    }
}

/// Only adapts already constructed test fixtures; production must admit before
/// constructing values and use the owning batch APIs.
#[cfg(test)]
pub(crate) fn admitted_ops_for_test(values: Vec<ReplicaOp>) -> elasticsearch_engine::translog::AdmittedBatch<ReplicaOp> {
    let bytes = values.capacity() as u64 * std::mem::size_of::<ReplicaOp>() as u64 + values.iter().map(|op| match op {
        ReplicaOp::Index { id, source_json, routing, .. } => id.capacity() as u64 + source_json.capacity() as u64 + routing.as_ref().map_or(0, |s| s.capacity() as u64),
        ReplicaOp::Delete { id, .. } => id.capacity() as u64,
        ReplicaOp::NoOp { reason, .. } => reason.capacity() as u64,
    }).sum::<u64>();
    let memory = elasticsearch_engine::memory::Breakers::leak_for_tests(u64::MAX / 4).reserve_request(bytes, "explicit replication test fixture").unwrap();
    elasticsearch_engine::translog::AdmittedBatch::from_reserved(values, memory)
}

impl<'de> Deserialize<'de> for ReplicationRequest {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::{Error, Visitor, SeqAccess, MapAccess};
        use crate::replication::decode::{Text, OptionalText, Operations, DecodeMemory};
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = ReplicationRequest;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("ReplicationRequest") }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let breakers = elasticsearch_engine::memory::breakers();
                let mut memory = breakers.reserve_request(0, "decoded envelope scalars").map_err(|e| A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let index = seq.next_element_seed(Text(&mut memory))?.ok_or_else(|| A::Error::invalid_length(0, &"ReplicationRequest"))?;
                let shard = seq.next_element::<u32>()?.ok_or_else(|| A::Error::invalid_length(1, &"ReplicationRequest"))?;
                let target_allocation_id = seq.next_element_seed(Text(&mut memory))?.ok_or_else(|| A::Error::invalid_length(2, &"ReplicationRequest"))?;
                let primary_term = seq.next_element::<u64>()?.ok_or_else(|| A::Error::invalid_length(3, &"ReplicationRequest"))?;
                let global_checkpoint = seq.next_element::<i64>()?.ok_or_else(|| A::Error::invalid_length(4, &"ReplicationRequest"))?;
                let max_seq_no_of_updates = seq.next_element::<i64>()?.ok_or_else(|| A::Error::invalid_length(5, &"ReplicationRequest"))?;
                let mapping_json = seq.next_element_seed(OptionalText(&mut memory))?.ok_or_else(|| A::Error::invalid_length(6, &"ReplicationRequest"))?;
                let ops = seq.next_element_seed(Operations(breakers))?.ok_or_else(|| A::Error::invalid_length(7, &"ReplicationRequest"))?;
                let mapping_version = seq.next_element::<u64>().ok().flatten().unwrap_or(0);
                Ok(ReplicationRequest { index, shard, target_allocation_id, primary_term, global_checkpoint, max_seq_no_of_updates, mapping_json, ops, mapping_version, memory: DecodeMemory::new(memory) })
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                #[derive(serde::Deserialize)]
                #[serde(field_identifier, rename_all = "snake_case")]
                enum Field { Index, Shard, TargetAllocationId, PrimaryTerm, GlobalCheckpoint, MaxSeqNoOfUpdates, MappingJson, Ops, MappingVersion }
                let breakers = elasticsearch_engine::memory::breakers();
                let mut memory = breakers.reserve_request(0, "decoded envelope scalars").map_err(|e| A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let mut index = None;
                let mut shard = None;
                let mut target_allocation_id = None;
                let mut primary_term = None;
                let mut global_checkpoint = None;
                let mut max_seq_no_of_updates = None;
                let mut mapping_json = None;
                let mut ops = None;
                let mut mapping_version = None;
                while let Some(field) = map.next_key::<Field>()? {
                    match field {
                        Field::Index => {
                            if index.is_some() { return Err(A::Error::duplicate_field("index")); }
                            index = Some(map.next_value_seed(Text(&mut memory))?);
                        }
                        Field::Shard => {
                            if shard.is_some() { return Err(A::Error::duplicate_field("shard")); }
                            shard = Some(map.next_value::<u32>()?);
                        }
                        Field::TargetAllocationId => {
                            if target_allocation_id.is_some() { return Err(A::Error::duplicate_field("target_allocation_id")); }
                            target_allocation_id = Some(map.next_value_seed(Text(&mut memory))?);
                        }
                        Field::PrimaryTerm => {
                            if primary_term.is_some() { return Err(A::Error::duplicate_field("primary_term")); }
                            primary_term = Some(map.next_value::<u64>()?);
                        }
                        Field::GlobalCheckpoint => {
                            if global_checkpoint.is_some() { return Err(A::Error::duplicate_field("global_checkpoint")); }
                            global_checkpoint = Some(map.next_value::<i64>()?);
                        }
                        Field::MaxSeqNoOfUpdates => {
                            if max_seq_no_of_updates.is_some() { return Err(A::Error::duplicate_field("max_seq_no_of_updates")); }
                            max_seq_no_of_updates = Some(map.next_value::<i64>()?);
                        }
                        Field::MappingJson => {
                            if mapping_json.is_some() { return Err(A::Error::duplicate_field("mapping_json")); }
                            mapping_json = Some(map.next_value_seed(OptionalText(&mut memory))?);
                        }
                        Field::Ops => {
                            if ops.is_some() { return Err(A::Error::duplicate_field("ops")); }
                            ops = Some(map.next_value_seed(Operations(breakers))?);
                        }
                        Field::MappingVersion => {
                            if mapping_version.is_some() { return Err(A::Error::duplicate_field("mapping_version")); }
                            mapping_version = Some(map.next_value::<u64>()?);
                        }
                    }
                }
                Ok(ReplicationRequest {
                    index: index.ok_or_else(|| A::Error::missing_field("index"))?,
                    shard: shard.ok_or_else(|| A::Error::missing_field("shard"))?,
                    target_allocation_id: target_allocation_id.ok_or_else(|| A::Error::missing_field("target_allocation_id"))?,
                    primary_term: primary_term.ok_or_else(|| A::Error::missing_field("primary_term"))?,
                    global_checkpoint: global_checkpoint.ok_or_else(|| A::Error::missing_field("global_checkpoint"))?,
                    max_seq_no_of_updates: max_seq_no_of_updates.ok_or_else(|| A::Error::missing_field("max_seq_no_of_updates"))?,
                    mapping_json: mapping_json.unwrap_or(None),
                    ops: ops.ok_or_else(|| A::Error::missing_field("ops"))?,
                    mapping_version: mapping_version.unwrap_or(0),
                    memory: DecodeMemory::new(memory),
                })
            }
        }
        d.deserialize_struct("ReplicationRequest", &["index", "shard", "target_allocation_id", "primary_term", "global_checkpoint", "max_seq_no_of_updates", "mapping_json", "ops", "mapping_version"], V)
    }
}

/// Primary-replica resync after a promotion (ES `ResyncReplicationRequest`):
/// every op the new primary holds in `(global_checkpoint, max_seq_no]`,
/// with gaps filled by `NoOp`s. Ops keep their original primary terms;
/// fencing uses the request's `primary_term`.
#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct ResyncRequest {
    pub index: String,
    pub shard: u32,
    pub target_allocation_id: String,
    pub primary_term: u64,
    pub global_checkpoint: i64,
    pub max_seq_no: i64,
    pub ops: elasticsearch_engine::translog::AdmittedBatch<ReplicaOp>,
    #[serde(skip)]
    pub memory: crate::replication::decode::DecodeMemory,
}

impl ResyncRequest {
    pub fn try_new(index: &str, shard: u32, target_allocation_id: &str, primary_term: u64,
        global_checkpoint: i64, max_seq_no: i64, ops: elasticsearch_engine::translog::AdmittedBatch<ReplicaOp>)
        -> elasticsearch_engine::error::Result<Self> {
        let request = ReplicationRequest::try_new(index, shard, target_allocation_id, primary_term,
            global_checkpoint, -1, None, ops, 0)?;
        Ok(Self { index: request.index, shard, target_allocation_id: request.target_allocation_id,
            primary_term, global_checkpoint, max_seq_no, ops: request.ops, memory: request.memory })
    }
}

impl<'de> Deserialize<'de> for ResyncRequest {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::{Error, Visitor, SeqAccess, MapAccess};
        use crate::replication::decode::{Text, Operations, DecodeMemory};
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = ResyncRequest;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("ResyncRequest") }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let breakers = elasticsearch_engine::memory::breakers();
                let mut memory = breakers.reserve_request(0, "decoded envelope scalars").map_err(|e| A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let index = seq.next_element_seed(Text(&mut memory))?.ok_or_else(|| A::Error::invalid_length(0, &"ResyncRequest"))?;
                let shard = seq.next_element::<u32>()?.ok_or_else(|| A::Error::invalid_length(1, &"ResyncRequest"))?;
                let target_allocation_id = seq.next_element_seed(Text(&mut memory))?.ok_or_else(|| A::Error::invalid_length(2, &"ResyncRequest"))?;
                let primary_term = seq.next_element::<u64>()?.ok_or_else(|| A::Error::invalid_length(3, &"ResyncRequest"))?;
                let global_checkpoint = seq.next_element::<i64>()?.ok_or_else(|| A::Error::invalid_length(4, &"ResyncRequest"))?;
                let max_seq_no = seq.next_element::<i64>()?.ok_or_else(|| A::Error::invalid_length(5, &"ResyncRequest"))?;
                let ops = seq.next_element_seed(Operations(breakers))?.ok_or_else(|| A::Error::invalid_length(6, &"ResyncRequest"))?;
                Ok(ResyncRequest { index, shard, target_allocation_id, primary_term, global_checkpoint, max_seq_no, ops, memory: DecodeMemory::new(memory) })
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                #[derive(serde::Deserialize)]
                #[serde(field_identifier, rename_all = "snake_case")]
                enum Field { Index, Shard, TargetAllocationId, PrimaryTerm, GlobalCheckpoint, MaxSeqNo, Ops }
                let breakers = elasticsearch_engine::memory::breakers();
                let mut memory = breakers.reserve_request(0, "decoded envelope scalars").map_err(|e| A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let mut index = None;
                let mut shard = None;
                let mut target_allocation_id = None;
                let mut primary_term = None;
                let mut global_checkpoint = None;
                let mut max_seq_no = None;
                let mut ops = None;
                while let Some(field) = map.next_key::<Field>()? {
                    match field {
                        Field::Index => {
                            if index.is_some() { return Err(A::Error::duplicate_field("index")); }
                            index = Some(map.next_value_seed(Text(&mut memory))?);
                        }
                        Field::Shard => {
                            if shard.is_some() { return Err(A::Error::duplicate_field("shard")); }
                            shard = Some(map.next_value::<u32>()?);
                        }
                        Field::TargetAllocationId => {
                            if target_allocation_id.is_some() { return Err(A::Error::duplicate_field("target_allocation_id")); }
                            target_allocation_id = Some(map.next_value_seed(Text(&mut memory))?);
                        }
                        Field::PrimaryTerm => {
                            if primary_term.is_some() { return Err(A::Error::duplicate_field("primary_term")); }
                            primary_term = Some(map.next_value::<u64>()?);
                        }
                        Field::GlobalCheckpoint => {
                            if global_checkpoint.is_some() { return Err(A::Error::duplicate_field("global_checkpoint")); }
                            global_checkpoint = Some(map.next_value::<i64>()?);
                        }
                        Field::MaxSeqNo => {
                            if max_seq_no.is_some() { return Err(A::Error::duplicate_field("max_seq_no")); }
                            max_seq_no = Some(map.next_value::<i64>()?);
                        }
                        Field::Ops => {
                            if ops.is_some() { return Err(A::Error::duplicate_field("ops")); }
                            ops = Some(map.next_value_seed(Operations(breakers))?);
                        }
                    }
                }
                Ok(ResyncRequest {
                    index: index.ok_or_else(|| A::Error::missing_field("index"))?,
                    shard: shard.ok_or_else(|| A::Error::missing_field("shard"))?,
                    target_allocation_id: target_allocation_id.ok_or_else(|| A::Error::missing_field("target_allocation_id"))?,
                    primary_term: primary_term.ok_or_else(|| A::Error::missing_field("primary_term"))?,
                    global_checkpoint: global_checkpoint.ok_or_else(|| A::Error::missing_field("global_checkpoint"))?,
                    max_seq_no: max_seq_no.ok_or_else(|| A::Error::missing_field("max_seq_no"))?,
                    ops: ops.ok_or_else(|| A::Error::missing_field("ops"))?,
                    memory: DecodeMemory::new(memory),
                })
            }
        }
        d.deserialize_struct("ResyncRequest", &["index", "shard", "target_allocation_id", "primary_term", "global_checkpoint", "max_seq_no", "ops"], V)
    }
}

impl ReplicaOp {
    /// Same operation: kind, seq_no, primary term, id and version.
    pub fn same_op(&self, other: &ReplicaOp) -> bool {
        match (self, other) {
            (ReplicaOp::Index { id: a, seq_no: s1, version: v1, primary_term: t1, .. }, ReplicaOp::Index { id: b, seq_no: s2, version: v2, primary_term: t2, .. })
            | (ReplicaOp::Delete { id: a, seq_no: s1, version: v1, primary_term: t1 }, ReplicaOp::Delete { id: b, seq_no: s2, version: v2, primary_term: t2 }) => {
                a == b && s1 == s2 && v1 == v2 && t1 == t2
            }
            (ReplicaOp::NoOp { seq_no: s1, primary_term: t1, .. }, ReplicaOp::NoOp { seq_no: s2, primary_term: t2, .. }) => s1 == s2 && t1 == t2,
            _ => false,
        }
    }
}

/// The replica's answer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicationResponse {
    pub local_checkpoint: i64,
    pub max_seq_no: i64,
    /// Ops of the request that were applied (not duplicates / stale).
    pub applied: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resync_admitted_envelope_preserves_legacy_wire_layout() {
        #[derive(Serialize)]
        struct Legacy<'a> { index: &'a str, shard: u32, target_allocation_id: &'a str,
            primary_term: u64, global_checkpoint: i64, max_seq_no: i64, ops: &'a [ReplicaOp] }
        let values = vec![ReplicaOp::NoOp { seq_no: 7, primary_term: 3, reason: "gap\nλ".into() }];
        let legacy = Legacy { index: "i", shard: 2, target_allocation_id: "a", primary_term: 3,
            global_checkpoint: 5, max_seq_no: 7, ops: &values };
        let bytes = bincode::serialize(&legacy).unwrap();
        let decoded: ResyncRequest = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.ops, values);
        assert_eq!(bincode::serialize(&decoded).unwrap(), bytes);
        let json = serde_json::to_string(&decoded).unwrap();
        assert_eq!(serde_json::from_str::<ResyncRequest>(&json).unwrap(), decoded);
        let produced = ResyncRequest::try_new("i", 2, "a", 3, 5, 7, admitted_ops_for_test(values)).unwrap();
        assert_eq!(bincode::serialize(&produced).unwrap(), bytes);
    }

    #[test]
    fn translog_conversion_and_bincode_roundtrip() {
        let op = ReplicaOp::from_translog(
            TranslogOp::Index { id: "a".into(), source_json: "{}".into(), seq_no: 7, version: 2, routing: Some("r".into()) },
            3,
        );
        assert_eq!(op, ReplicaOp::Index { id: "a".into(), source_json: "{}".into(), routing: Some("r".into()), seq_no: 7, version: 2, primary_term: 3 });
        assert_eq!(op.seq_no(), 7);
        assert_eq!(op.primary_term(), 3);
        let d = ReplicaOp::from_translog(TranslogOp::Delete { id: "a".into(), seq_no: 8, version: 3 }, 3);
        assert_eq!(d.id(), Some("a"));
        let req = ReplicationRequest::try_new("i", 0, "x", 3, 6, 8, Some("{\"properties\":{}}"),
            admitted_ops_for_test(vec![op, d, ReplicaOp::NoOp { seq_no: 9, primary_term: 3, reason: "gap".into() }]), 4).unwrap();
        let bytes = bincode::serialize(&req).unwrap();
        let back: ReplicationRequest = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back, req);
        let json: ReplicationRequest = serde_json::from_str(&serde_json::to_string(&req).unwrap()).unwrap();
        assert_eq!(json, req);
        // An older sender's request (no mapping_version) still decodes...
        #[derive(Serialize)]
        struct Old<'a> {
            index: &'a str,
            shard: u32,
            target_allocation_id: &'a str,
            primary_term: u64,
            global_checkpoint: i64,
            max_seq_no_of_updates: i64,
            mapping_json: Option<&'a str>,
            ops: &'a [ReplicaOp],
        }
        let old = Old { index: "i", shard: 0, target_allocation_id: "x", primary_term: 3, global_checkpoint: 6, max_seq_no_of_updates: 8, mapping_json: req.mapping_json.as_deref(), ops: &req.ops };
        let back: ReplicationRequest = bincode::deserialize(&bincode::serialize(&old).unwrap()).unwrap();
        assert_eq!(back, ReplicationRequest { mapping_version: 0, ..req.try_clone().unwrap() });
        // ...and an older receiver ignores the appended field.
        #[derive(Deserialize, Debug, PartialEq)]
        struct OldOwned {
            index: String,
            shard: u32,
            target_allocation_id: String,
            primary_term: u64,
            global_checkpoint: i64,
            max_seq_no_of_updates: i64,
            mapping_json: Option<String>,
            ops: Vec<ReplicaOp>,
        }
        let seen: OldOwned = bincode::deserialize(&bytes).unwrap();
        assert_eq!(seen.ops.as_slice(), &*req.ops);
    }
}
