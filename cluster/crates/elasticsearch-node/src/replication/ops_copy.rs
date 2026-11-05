//! Explicit fallible deep copies; never share the source guard for new strings.
use super::ops::ReplicaOp;
use elasticsearch_engine::{error::{EsError, Result}, memory::{breakers, Reservation}, translog::AdmittedBatch};
fn copy(text: &str, memory: &mut Reservation) -> Result<String> {
    memory.grow(text.len() as u64, "replication copied text")?;
    let mut out = String::new();
    out.try_reserve_exact(text.len()).map_err(|e| EsError::new(429, "circuit_breaking_exception", format!("replication copy refused: {e}")))?;
    out.push_str(text);
    Ok(out)
}
impl ReplicaOp {
    pub fn try_noop(seq_no: i64, primary_term: u64, reason: &str) -> Result<AdmittedBatch<Self>> {
        let mut out = AdmittedBatch::new(breakers())?;
        out.reserve_one()?;
        let mut memory = breakers().reserve_request(0, "replication no-op")?;
        let value = Self::NoOp { seq_no, primary_term, reason: copy(reason, &mut memory)? };
        out.push_reserved(value, memory).map_err(|(e, _, _)| e)?;
        Ok(out)
    }
    pub fn try_copy_change(op: &elasticsearch_engine::hooks::ChangeOp) -> Result<(Self, Reservation)> {
        use elasticsearch_engine::hooks::ChangeOp;
        let mut memory = breakers().reserve_request(0, "replication copied change")?;
        let value = match op {
            ChangeOp::Index { id, source_json, routing, seq_no, version, primary_term } => Self::Index {
                id: copy(id, &mut memory)?, source_json: copy(source_json, &mut memory)?,
                routing: routing.as_deref().map(|r| copy(r, &mut memory)).transpose()?,
                seq_no: *seq_no, version: *version, primary_term: *primary_term,
            },
            ChangeOp::Delete { id, seq_no, version, primary_term } => Self::Delete {
                id: copy(id, &mut memory)?, seq_no: *seq_no, version: *version, primary_term: *primary_term,
            },
            ChangeOp::NoOp { seq_no, primary_term, reason } => Self::NoOp {
                seq_no: *seq_no, primary_term: *primary_term, reason: copy(reason, &mut memory)?,
            },
        };
        Ok((value, memory))
    }
    pub fn try_clone_batch(ops: &AdmittedBatch<Self>) -> Result<AdmittedBatch<Self>> {
        Self::try_copy_selected(ops, |_| true)
    }
    pub fn try_copy_selected(ops: &[Self], mut select: impl FnMut(&Self) -> bool) -> Result<AdmittedBatch<Self>> {
        // Reserve the input bound; predicates need not be stable across calls.
        let count = ops.len();
        let bytes = count.checked_mul(std::mem::size_of::<Self>()).ok_or_else(|| EsError::new(429, "circuit_breaking_exception", "replication copy size overflow"))?;
        let mut memory = breakers().reserve_request(bytes as u64, "replication copied slots")?;
        let mut values = Vec::new();
        values.try_reserve_exact(count).map_err(|e| EsError::new(429, "circuit_breaking_exception", format!("replication copy refused: {e}")))?;
        for op in ops.iter().filter(|op| select(op)) {
            values.push(match op {
                Self::Index { id, source_json, routing, seq_no, version, primary_term } => Self::Index {
                    id: copy(id, &mut memory)?, source_json: copy(source_json, &mut memory)?,
                    routing: routing.as_deref().map(|r| copy(r, &mut memory)).transpose()?,
                    seq_no: *seq_no, version: *version, primary_term: *primary_term,
                },
                Self::Delete { id, seq_no, version, primary_term } => Self::Delete {
                    id: copy(id, &mut memory)?, seq_no: *seq_no, version: *version, primary_term: *primary_term,
                },
                Self::NoOp { seq_no, primary_term, reason } => Self::NoOp {
                    seq_no: *seq_no, primary_term: *primary_term, reason: copy(reason, &mut memory)?,
                },
            });
        }
        Ok(AdmittedBatch::from_reserved(values, memory))
    }
}
