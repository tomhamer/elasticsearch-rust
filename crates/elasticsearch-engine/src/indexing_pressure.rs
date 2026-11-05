//! Byte-based indexing pressure (ES `IndexingPressure`): document writes
//! hold the bytes of their requests while they run; past
//! `indexing_pressure.memory.limit` (default 10% of the memory budget, ES's
//! heap) new coordinating / primary work is rejected with 429
//! `es_rejected_execution_exception`, and replica work past 1.5x the limit.
//! `_nodes/stats` reports the live and cumulative figures.

use crate::error::EsError;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Node-wide indexing pressure.
#[derive(Default)]
pub struct IndexingPressure {
    limit: AtomicU64,
    coordinating: AtomicU64,
    primary: AtomicU64,
    /// Coordinating plus primary bytes, a primary op local to its
    /// coordinating request counted once (ES `currentCombinedCoordinatingAndPrimaryBytes`).
    combined: AtomicU64,
    replica: AtomicU64,
    total_coordinating: AtomicU64,
    total_primary: AtomicU64,
    total_combined: AtomicU64,
    total_replica: AtomicU64,
    coordinating_rejections: AtomicU64,
    primary_rejections: AtomicU64,
    replica_rejections: AtomicU64,
}

/// Bytes held by one running write; released on drop.
#[derive(Debug)]
#[must_use = "the bytes are released when the guard drops"]
pub struct PressureGuard {
    counters: Vec<&'static AtomicU64>,
    bytes: u64,
}

impl Drop for PressureGuard {
    fn drop(&mut self) {
        for counter in &self.counters {
            counter.fetch_sub(self.bytes, Ordering::AcqRel);
        }
    }
}

/// Live and cumulative figures (`_nodes/stats` `indexing_pressure`).
#[derive(Clone, Copy, Debug, Default)]
pub struct PressureStats {
    pub coordinating_bytes: u64,
    pub primary_bytes: u64,
    pub combined_bytes: u64,
    pub replica_bytes: u64,
    pub total_coordinating_bytes: u64,
    pub total_primary_bytes: u64,
    pub total_combined_bytes: u64,
    pub total_replica_bytes: u64,
    pub coordinating_rejections: u64,
    pub primary_rejections: u64,
    pub replica_rejections: u64,
    pub limit_bytes: u64,
}

/// The node's indexing pressure.
pub fn global() -> &'static IndexingPressure {
    static P: OnceLock<IndexingPressure> = OnceLock::new();
    P.get_or_init(|| {
        let p = IndexingPressure::default();
        let limit = std::env::var("ES_INDEXING_PRESSURE_MEMORY_LIMIT").ok().and_then(|v| v.trim().parse().ok())
            .unwrap_or_else(|| crate::memory::es_heap() / 10);
        p.limit.store(limit, Ordering::Relaxed);
        p
    })
}

/// Validate and apply the node setting `indexing_pressure.memory.limit` at
/// startup: an invalid value fails the node's start (as ES fails on an
/// unparsable setting) instead of being ignored.
pub fn configure(settings: &serde_json::Map<String, serde_json::Value>) -> Result<(), EsError> {
    let Some(value) = settings.get("indexing_pressure.memory.limit") else { return Ok(()) };
    let bytes = parse_limit(value).ok_or_else(|| EsError::illegal_argument(format!(
        "failed to parse setting [indexing_pressure.memory.limit] with value [{}] as a size in bytes or a percentage",
        value.as_str().map(str::to_string).unwrap_or_else(|| value.to_string()))))?;
    global().set_limit(bytes);
    Ok(())
}

/// Apply the node setting `indexing_pressure.memory.limit` (a byte size or
/// a percentage of ES's heap, [`crate::memory::es_heap`], like its 10%
/// default), once per process: the
/// first node that has it configures the limit.
pub fn configure_from(node: &crate::node::Node) {
    static DONE: OnceLock<()> = OnceLock::new();
    if DONE.get().is_some() {
        return;
    }
    let value = node.node_settings.read().get("indexing_pressure.memory.limit").cloned();
    let Some(value) = value else { return };
    if let Some(bytes) = parse_limit(&value) {
        global().set_limit(bytes);
        let _ = DONE.set(());
    }
}

/// `indexing_pressure.memory.limit`: `"512mb"`, `"10%"`, or bytes.
fn parse_limit(value: &serde_json::Value) -> Option<u64> {
    let text = match value {
        serde_json::Value::String(s) => s.trim().to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        _ => return None,
    };
    if let Some(pct) = text.strip_suffix('%') {
        let pct: f64 = pct.trim().parse().ok()?;
        return (0.0..=100.0).contains(&pct).then(|| (crate::memory::es_heap() as f64 * pct / 100.0) as u64);
    }
    crate::ingest::util::parse_bytes(&text).and_then(|b| u64::try_from(b).ok())
}

impl IndexingPressure {
    /// The limit in bytes (`indexing_pressure.memory.limit`).
    pub fn limit(&self) -> u64 {
        self.limit.load(Ordering::Relaxed)
    }

    /// Set the limit in bytes (`0`: unlimited).
    pub fn set_limit(&self, bytes: u64) {
        self.limit.store(bytes, Ordering::Relaxed);
    }

    /// Whether `bytes` more of coordinating / primary work would pass the
    /// limit. A request alone larger than the limit is admitted when
    /// nothing else is in flight (as ES: the request is already read).
    ///
    /// ES order: the bytes are added first, then the total is checked and
    /// the addition rolled back on a rejection, so concurrent writers can
    /// never all pass a check made before any of them counted.
    fn reserve_combined(&self, bytes: u64) -> bool {
        let limit = self.limit();
        let combined = self.combined.fetch_add(bytes, Ordering::AcqRel) + bytes;
        let others = combined - bytes;
        if limit > 0 && combined + self.replica.load(Ordering::Acquire) > limit && others > 0 {
            self.combined.fetch_sub(bytes, Ordering::AcqRel);
            return false;
        }
        true
    }

    /// Hold `bytes` of coordinating work (a write request received by this
    /// node), or reject (429) past the limit.
    pub fn coordinating(&'static self, bytes: u64) -> Result<PressureGuard, EsError> {
        if !self.reserve_combined(bytes) {
            self.coordinating_rejections.fetch_add(1, Ordering::Relaxed);
            return Err(self.rejection("coordinating", bytes));
        }
        self.coordinating.fetch_add(bytes, Ordering::AcqRel);
        self.total_coordinating.fetch_add(bytes, Ordering::Relaxed);
        self.total_combined.fetch_add(bytes, Ordering::Relaxed);
        Ok(PressureGuard { counters: vec![&self.coordinating, &self.combined], bytes })
    }

    /// Hold `bytes` of primary work (a shard-level write executed here).
    /// `local`: its coordinating request on this node already holds the
    /// bytes (counted as primary, not again in the combined figure, never
    /// rejected); otherwise rejected (429) past the limit.
    pub fn primary(&'static self, bytes: u64, local: bool) -> Result<PressureGuard, EsError> {
        if local {
            self.primary.fetch_add(bytes, Ordering::AcqRel);
            self.total_primary.fetch_add(bytes, Ordering::Relaxed);
            return Ok(PressureGuard { counters: vec![&self.primary], bytes });
        }
        if !self.reserve_combined(bytes) {
            self.primary_rejections.fetch_add(1, Ordering::Relaxed);
            return Err(self.rejection("primary", bytes));
        }
        self.primary.fetch_add(bytes, Ordering::AcqRel);
        self.total_primary.fetch_add(bytes, Ordering::Relaxed);
        self.total_combined.fetch_add(bytes, Ordering::Relaxed);
        Ok(PressureGuard { counters: vec![&self.primary, &self.combined], bytes })
    }

    fn rejection(&self, stage: &str, bytes: u64) -> EsError {
        let combined = self.combined.load(Ordering::Acquire);
        let replica = self.replica.load(Ordering::Acquire);
        EsError::new(429, "es_rejected_execution_exception", format!(
            "rejected execution of {stage} operation [coordinating_and_primary_bytes={combined}, replica_bytes={replica}, all_bytes={}, {stage}_operation_bytes={bytes}, max_coordinating_and_primary_bytes={}]",
            combined + replica, self.limit()))
    }

    /// Hold `bytes` of replica work, or reject past 1.5x the limit.
    pub fn replica(&'static self, bytes: u64) -> Result<PressureGuard, EsError> {
        let limit = self.limit().saturating_mul(3) / 2;
        let current = self.replica.fetch_add(bytes, Ordering::AcqRel) + bytes;
        let others = current - bytes;
        if limit > 0 && current > limit && others > 0 {
            self.replica.fetch_sub(bytes, Ordering::AcqRel);
            self.replica_rejections.fetch_add(1, Ordering::Relaxed);
            return Err(EsError::new(429, "es_rejected_execution_exception", format!(
                "rejected execution of replica operation [replica_bytes={others}, replica_operation_bytes={bytes}, max_replica_bytes={limit}]")));
        }
        self.total_replica.fetch_add(bytes, Ordering::Relaxed);
        Ok(PressureGuard { counters: vec![&self.replica], bytes })
    }

    /// Account `bytes` of replica work without rejecting it (ES forces
    /// replica execution: replication must not fail a copy because the
    /// node is busy with user writes).
    pub fn track_replica(&'static self, bytes: u64) -> PressureGuard {
        self.replica.fetch_add(bytes, Ordering::AcqRel);
        self.total_replica.fetch_add(bytes, Ordering::Relaxed);
        PressureGuard { counters: vec![&self.replica], bytes }
    }

    pub fn stats(&self) -> PressureStats {
        PressureStats {
            coordinating_bytes: self.coordinating.load(Ordering::Relaxed),
            primary_bytes: self.primary.load(Ordering::Relaxed),
            combined_bytes: self.combined.load(Ordering::Relaxed),
            replica_bytes: self.replica.load(Ordering::Relaxed),
            total_coordinating_bytes: self.total_coordinating.load(Ordering::Relaxed),
            total_primary_bytes: self.total_primary.load(Ordering::Relaxed),
            total_combined_bytes: self.total_combined.load(Ordering::Relaxed),
            total_replica_bytes: self.total_replica.load(Ordering::Relaxed),
            coordinating_rejections: self.coordinating_rejections.load(Ordering::Relaxed),
            primary_rejections: self.primary_rejections.load(Ordering::Relaxed),
            replica_rejections: self.replica_rejections.load(Ordering::Relaxed),
            limit_bytes: self.limit(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_past_the_limit_and_releases_on_drop() {
        static P: OnceLock<IndexingPressure> = OnceLock::new();
        let p = P.get_or_init(IndexingPressure::default);
        p.set_limit(100);
        let first = p.coordinating(80).unwrap();
        let err = p.coordinating(30).unwrap_err();
        assert_eq!((err.status, err.kind.as_str()), (429, "es_rejected_execution_exception"));
        assert_eq!(p.stats().coordinating_rejections, 1);
        drop(first);
        assert_eq!(p.stats().coordinating_bytes, 0);
        let _alone = p.coordinating(500).unwrap();
    }

    /// M13: primary work is its own stage: local to its coordinating
    /// request it counts as primary only (never rejected, not twice in the
    /// combined figure); a primary op from another node is limited too.
    #[test]
    fn primary_stage_is_counted_and_limited() {
        static P: OnceLock<IndexingPressure> = OnceLock::new();
        let p = P.get_or_init(IndexingPressure::default);
        p.set_limit(100);
        let coordinating = p.coordinating(60).unwrap();
        let local = p.primary(60, true).unwrap();
        let st = p.stats();
        assert_eq!((st.coordinating_bytes, st.primary_bytes, st.combined_bytes), (60, 60, 60));
        let remote = p.primary(50, false).unwrap_err();
        assert_eq!(remote.status, 429);
        assert_eq!(p.stats().primary_rejections, 1);
        drop((coordinating, local));
        let remote = p.primary(50, false).unwrap();
        assert_eq!(p.stats().combined_bytes, 50);
        drop(remote);
        assert_eq!(p.stats().combined_bytes, 0);
    }

    /// M13: concurrent writers can never all pass the limit (the bytes are
    /// counted before the check, ES order).
    #[test]
    fn concurrent_writers_never_overshoot_the_limit() {
        static P: OnceLock<IndexingPressure> = OnceLock::new();
        let p: &'static IndexingPressure = P.get_or_init(IndexingPressure::default);
        p.set_limit(1000);
        let _first = p.coordinating(10).unwrap();
        let barrier = std::sync::Barrier::new(16);
        let held: Vec<PressureGuard> = std::thread::scope(|scope| {
            let racers: Vec<_> = (0..16).map(|_| scope.spawn(|| { barrier.wait(); p.coordinating(300).ok() })).collect();
            racers.into_iter().filter_map(|r| r.join().unwrap()).collect()
        });
        assert!(10 + 300 * held.len() as u64 <= 1000, "{} writers of 300 bytes passed a 1000-byte limit", held.len());
        assert_eq!(p.stats().combined_bytes, 10 + 300 * held.len() as u64, "a rejection left its bytes counted");
    }

    #[test]
    fn an_invalid_limit_fails_the_configuration() {
        let mut settings = serde_json::Map::new();
        settings.insert("indexing_pressure.memory.limit".into(), serde_json::json!("lots"));
        assert_eq!(configure(&settings).unwrap_err().status, 400);
    }

    #[test]
    fn the_limit_setting_takes_bytes_or_a_percentage() {
        assert_eq!(parse_limit(&serde_json::json!("512mb")), Some(512 << 20));
        assert_eq!(parse_limit(&serde_json::json!(1024)), Some(1024));
        assert_eq!(parse_limit(&serde_json::json!("50%")), Some(crate::memory::es_heap() / 2));
        assert_eq!(parse_limit(&serde_json::json!("bogus")), None);
    }
}
