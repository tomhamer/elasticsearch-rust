//! History retention (hook 5): while any lease or lock retains seq_no `R`,
//! flush and force merge still commit, but never delete a translog
//! generation (or its history sidecar) that may hold an op `>= R`.
//!
//! * [`RetentionLock`]: in-memory, RAII (released on drop). For a
//!   recovery or a change-feed reader that lives in this process.
//! * [`RetentionLease`]: durable (`<shard>/retention_leases.json`), named,
//!   explicitly renewed/removed (ES retention leases: one per replica
//!   copy). Not copied by file transfers: leases are per copy.

use crate::error::{EsError, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const LEASES_FILE: &str = "retention_leases.json";

/// ES `ReplicationTracker.PEER_RECOVERY_RETENTION_LEASE_SOURCE`.
pub const PEER_RECOVERY_SOURCE: &str = "peer recovery";

/// `index.soft_deletes.retention_lease.period` default (12h).
pub const DEFAULT_LEASE_PERIOD_MILLIS: i64 = 12 * 3600 * 1000;

/// A durable retention lease (ES `RetentionLease`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RetentionLease {
    pub id: String,
    /// Ops with seq_no `>=` this are retained.
    pub retaining_seq_no: i64,
    pub timestamp_millis: i64,
    pub source: String,
}

pub(crate) struct Leases {
    path: PathBuf,
    durable: BTreeMap<String, RetentionLease>,
    /// lock id → (retained seq_no, holder)
    locks: BTreeMap<u64, (i64, String)>,
    next_lock: u64,
}

impl Leases {
    pub fn open(shard_path: &Path) -> Result<Leases> {
        let path = shard_path.join(LEASES_FILE);
        let durable: Vec<RetentionLease> = match std::fs::read(&path) {
            Ok(b) => serde_json::from_slice(&b).map_err(|e| EsError::internal(format!("corrupt {LEASES_FILE}: {e}")))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e.into()),
        };
        Ok(Leases { path, durable: durable.into_iter().map(|l| (l.id.clone(), l)).collect(), locks: BTreeMap::new(), next_lock: 1 })
    }

    fn persist(&self) -> Result<()> {
        let v: Vec<&RetentionLease> = self.durable.values().collect();
        super::write_durable(&self.path, &serde_json::to_vec_pretty(&v)?)?;
        Ok(())
    }

    /// Lowest retained seq_no over every lease and lock.
    pub fn min_retained(&self) -> Option<i64> {
        self.durable.values().map(|l| l.retaining_seq_no).chain(self.locks.values().map(|l| l.0)).min()
    }

    pub fn upsert(&mut self, lease: RetentionLease, create_only: bool) -> Result<()> {
        if create_only && self.durable.contains_key(&lease.id) {
            return Err(EsError::new(409, "retention_lease_already_exists_exception", format!("retention lease with ID [{}] already exists", lease.id)));
        }
        if let Some(cur) = self.durable.get(&lease.id) {
            if lease.retaining_seq_no < cur.retaining_seq_no {
                return Err(EsError::illegal_argument(format!(
                    "the current retention lease with [{}] is retaining a higher sequence number [{}] than the new retaining sequence number [{}] from [{}]",
                    lease.id, cur.retaining_seq_no, lease.retaining_seq_no, lease.source
                )));
            }
        }
        self.durable.insert(lease.id.clone(), lease);
        self.persist()
    }

    pub fn remove(&mut self, id: &str) -> Result<bool> {
        let had = self.durable.remove(id).is_some();
        if had {
            self.persist()?;
        }
        Ok(had)
    }

    /// Remove the durable leases not renewed within `period_millis` of
    /// `now_millis` (ES `ReplicationTracker` expiry of
    /// `index.soft_deletes.retention_lease.period`). Peer-recovery leases
    /// never expire this way (ES removes them only when the copy goes).
    /// Returns the removed ids.
    pub fn expire(&mut self, now_millis: i64, period_millis: i64) -> Result<Vec<String>> {
        let expired: Vec<String> = self.durable.values()
            .filter(|l| Self::is_expired(l, now_millis, period_millis))
            .map(|l| l.id.clone())
            .collect();
        if !expired.is_empty() {
            for id in &expired {
                self.durable.remove(id);
            }
            self.persist()?;
        }
        Ok(expired)
    }

    /// Whether [`Leases::expire`] would remove anything.
    pub fn any_expired(&self, now_millis: i64, period_millis: i64) -> bool {
        self.durable.values().any(|l| Self::is_expired(l, now_millis, period_millis))
    }

    fn is_expired(l: &RetentionLease, now_millis: i64, period_millis: i64) -> bool {
        l.source != PEER_RECOVERY_SOURCE && now_millis.saturating_sub(l.timestamp_millis) > period_millis
    }

    pub fn list(&self) -> Vec<RetentionLease> {
        self.durable.values().cloned().collect()
    }

    pub fn lock(this: &Arc<Mutex<Leases>>, from_seq_no: i64, holder: String) -> RetentionLock {
        let mut l = this.lock();
        let id = l.next_lock;
        l.next_lock += 1;
        l.locks.insert(id, (from_seq_no, holder));
        RetentionLock { leases: this.clone(), id }
    }

    pub fn locks(&self) -> Vec<(i64, String)> {
        self.locks.values().cloned().collect()
    }
}

/// Retains history `>= retained_seq_no()` until dropped.
pub struct RetentionLock {
    leases: Arc<Mutex<Leases>>,
    id: u64,
}

impl RetentionLock {
    /// The lowest seq_no this lock retains.
    pub fn retained_seq_no(&self) -> i64 {
        self.leases.lock().locks.get(&self.id).map(|l| l.0).unwrap_or(i64::MAX)
    }

    /// Release history below `seq_no` (only moves forward), e.g. as a
    /// change-feed reader consumes ops.
    pub fn advance_to(&self, seq_no: i64) {
        if let Some(l) = self.leases.lock().locks.get_mut(&self.id) {
            l.0 = l.0.max(seq_no);
        }
    }
}

impl Drop for RetentionLock {
    fn drop(&mut self) {
        self.leases.lock().locks.remove(&self.id);
    }
}

impl std::fmt::Debug for RetentionLock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RetentionLock(#{} >= {})", self.id, self.retained_seq_no())
    }
}
