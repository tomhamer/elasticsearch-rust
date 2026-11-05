//! Operation history: one record per client operation (one per item for
//! bulk), with invocation and completion timestamps relative to run start.

use parking_lot::Mutex;
use serde::Serialize;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// Setup write (initial register values); always acknowledged.
    Init,
    /// `PUT /idx/_create/id` of a unique id.
    Create,
    /// One item of a `_bulk` of `create` actions with unique ids.
    BulkCreate,
    /// `PUT /idx/_doc/id?if_seq_no=&if_primary_term=` on a register key.
    Cas,
    /// `DELETE /idx/_doc/id`.
    Delete,
    /// Realtime `GET /idx/_doc/id`.
    Read,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// Acknowledged (2xx, or 404 not_found for delete / read).
    Ok,
    /// Definitely not applied: connection refused before the request was
    /// sent, 409 conflict, 503 cluster_block_exception (checked before
    /// execution), 400.
    Fail,
    /// Unknown: timeout after sending, connection reset, other 5xx.
    Info,
    /// Not yet completed (only while running).
    Pending,
}

#[derive(Clone, Debug, Serialize)]
pub struct Op {
    pub id: u64,
    pub worker: usize,
    /// Node the client sent the request to.
    pub node: usize,
    pub kind: Kind,
    pub index: String,
    pub key: String,
    /// Value written (writes) — unique per write.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub if_seq_no: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub if_primary_term: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bulk_id: Option<u64>,
    pub t_invoke: f64,
    pub t_complete: f64,
    pub outcome: Outcome,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seq_no: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_term: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<i64>,
    /// Reads: `Some(None)` = not found; deletes: `Some(None)` = not_found.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub read: Option<Option<i64>>,
}

pub struct History {
    pub t0: Instant,
    ops: Mutex<Vec<Op>>,
}

impl History {
    pub fn new(t0: Instant) -> History {
        History { t0, ops: Mutex::new(Vec::new()) }
    }
    pub fn now(&self) -> f64 {
        self.t0.elapsed().as_secs_f64()
    }
    /// Record an invocation; returns the op id.
    pub fn invoke(&self, mut op: Op) -> u64 {
        let mut ops = self.ops.lock();
        op.id = ops.len() as u64;
        op.t_invoke = self.now();
        op.outcome = Outcome::Pending;
        ops.push(op);
        (ops.len() - 1) as u64
    }
    pub fn complete(&self, id: u64, f: impl FnOnce(&mut Op)) {
        let t = self.now();
        let mut ops = self.ops.lock();
        let op = &mut ops[id as usize];
        op.t_complete = t;
        f(op);
    }
    pub fn snapshot(&self) -> Vec<Op> {
        self.ops.lock().clone()
    }
    pub fn len(&self) -> usize {
        self.ops.lock().len()
    }
}

pub fn blank(worker: usize, node: usize, kind: Kind, index: &str, key: &str) -> Op {
    Op {
        id: 0,
        worker,
        node,
        kind,
        index: index.to_string(),
        key: key.to_string(),
        value: None,
        if_seq_no: None,
        if_primary_term: None,
        bulk_id: None,
        t_invoke: 0.0,
        t_complete: 0.0,
        outcome: Outcome::Pending,
        status: None,
        error: None,
        seq_no: None,
        primary_term: None,
        version: None,
        read: None,
    }
}
