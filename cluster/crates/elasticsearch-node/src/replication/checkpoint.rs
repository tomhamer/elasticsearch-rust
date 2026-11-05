//! Sequence-number checkpoints (ES `LocalCheckpointTracker` and
//! `ReplicationTracker`).
//!
//! * The **local checkpoint** of a copy is the highest seq_no such that
//!   every op with a seq_no `<=` it has been processed by that copy.
//! * The **global checkpoint** (computed on the primary) is the minimum
//!   local checkpoint over the in-sync copies; every op at or below it is
//!   on every in-sync copy. It never goes backwards.
//! * The **retained seq_no** is the lowest seq_no some tracked copy (in
//!   sync or being recovered) may still need: the primary must keep every
//!   op `>=` it in its translog (ES retention leases).

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// No operation has been performed (ES `SequenceNumbers.NO_OPS_PERFORMED`).
pub const NO_OPS_PERFORMED: i64 = -1;
/// Unknown (ES `SequenceNumbers.UNASSIGNED_SEQ_NO`).
pub const UNASSIGNED_SEQ_NO: i64 = -2;

/// Tracks which seq_nos a copy has processed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalCheckpointTracker {
    checkpoint: i64,
    max_seq_no: i64,
    /// Processed seq_nos above `checkpoint` (never contains `checkpoint + 1`).
    pending: BTreeSet<i64>,
}

impl Default for LocalCheckpointTracker {
    fn default() -> Self {
        Self::new(NO_OPS_PERFORMED)
    }
}

impl LocalCheckpointTracker {
    /// Start at `checkpoint` (everything `<=` it already processed).
    pub fn new(checkpoint: i64) -> Self {
        let checkpoint = checkpoint.max(NO_OPS_PERFORMED);
        Self { checkpoint, max_seq_no: checkpoint, pending: BTreeSet::new() }
    }

    pub fn checkpoint(&self) -> i64 {
        self.checkpoint
    }

    /// Highest seq_no seen (processed or fast-forwarded to).
    pub fn max_seq_no(&self) -> i64 {
        self.max_seq_no
    }

    /// Number of processed seq_nos above the checkpoint (gaps pending).
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    pub fn has_processed(&self, seq_no: i64) -> bool {
        seq_no <= self.checkpoint || self.pending.contains(&seq_no)
    }

    /// Mark `seq_no` processed; returns the (possibly advanced) checkpoint.
    pub fn mark_processed(&mut self, seq_no: i64) -> i64 {
        if seq_no < 0 {
            return self.checkpoint;
        }
        self.max_seq_no = self.max_seq_no.max(seq_no);
        if seq_no <= self.checkpoint {
            return self.checkpoint;
        }
        if seq_no == self.checkpoint + 1 {
            self.checkpoint = seq_no;
            self.advance();
        } else {
            self.pending.insert(seq_no);
        }
        self.checkpoint
    }

    /// Declare every seq_no `<= to` processed (after a snapshot recovery
    /// whose segments reflect all those ops). Never moves backwards.
    pub fn fast_forward(&mut self, to: i64) -> i64 {
        if to > self.checkpoint {
            self.checkpoint = to;
            self.pending = self.pending.split_off(&(to + 1));
            self.max_seq_no = self.max_seq_no.max(to);
            self.advance();
        }
        self.checkpoint
    }

    fn advance(&mut self) {
        while let Some(&first) = self.pending.first() {
            if first != self.checkpoint + 1 {
                break;
            }
            self.pending.pop_first();
            self.checkpoint = first;
        }
    }
}

/// Checkpoint state of one shard copy as seen by the primary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CopyCheckpointState {
    pub local_checkpoint: i64,
    /// Counts towards the global checkpoint (and must ack every write).
    pub in_sync: bool,
    /// Receives replicated ops and holds back translog trimming.
    pub tracked: bool,
}

/// Serializable view of a [`ReplicationTracker`] (for `_stats`/debug).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicationTrackerStats {
    pub primary_allocation_id: String,
    pub primary_term: u64,
    pub global_checkpoint: i64,
    pub min_retained_seq_no: i64,
    pub copies: BTreeMap<String, CopyCheckpointState>,
}

/// Primary-side tracking of every copy's local checkpoint.
#[derive(Clone, Debug)]
pub struct ReplicationTracker {
    primary_allocation_id: String,
    primary_term: u64,
    copies: BTreeMap<String, CopyCheckpointState>,
    global_checkpoint: i64,
    /// The allocation ids the routing table assigns to this shard (ES
    /// `IndexShardRoutingTable` promotable copies), with their node ids,
    /// and how many copies it lists, unassigned and relocation targets
    /// included, as last applied (`None`: no cluster state applied yet).
    routing: Option<(BTreeMap<String, String>, u32)>,
}

impl ReplicationTracker {
    /// A new primary. `primary_local_checkpoint` is its own local
    /// checkpoint; it starts as the only in-sync copy until the cluster
    /// state says otherwise.
    pub fn new(primary_allocation_id: &str, primary_term: u64, primary_local_checkpoint: i64) -> Self {
        let mut copies = BTreeMap::new();
        copies.insert(primary_allocation_id.to_string(), CopyCheckpointState { local_checkpoint: primary_local_checkpoint, in_sync: true, tracked: true });
        let mut t = Self { primary_allocation_id: primary_allocation_id.to_string(), primary_term, copies, global_checkpoint: UNASSIGNED_SEQ_NO, routing: None };
        t.recompute();
        t
    }

    pub fn primary_allocation_id(&self) -> &str {
        &self.primary_allocation_id
    }

    pub fn primary_term(&self) -> u64 {
        self.primary_term
    }

    pub fn set_primary_term(&mut self, term: u64) {
        self.primary_term = self.primary_term.max(term);
    }

    /// Apply the cluster state's view: `in_sync` = the index metadata's
    /// in-sync allocation ids, `initializing` = copies being recovered
    /// (routing table INITIALIZING/RELOCATING targets). Copies absent from
    /// both are dropped (failed or removed). In-sync copies we have not
    /// heard from start at [`UNASSIGNED_SEQ_NO`], holding the global
    /// checkpoint back until they report.
    pub fn update_from_cluster_state<'a>(&mut self, in_sync: impl IntoIterator<Item = &'a str>, initializing: impl IntoIterator<Item = &'a str>) {
        let in_sync: BTreeSet<&str> = in_sync.into_iter().collect();
        let init: BTreeSet<&str> = initializing.into_iter().collect();
        let primary = self.primary_allocation_id.clone();
        self.copies.retain(|aid, _| *aid == primary || in_sync.contains(aid.as_str()) || init.contains(aid.as_str()));
        for aid in &in_sync {
            let e = self.copies.entry((*aid).to_string()).or_insert(CopyCheckpointState { local_checkpoint: UNASSIGNED_SEQ_NO, in_sync: true, tracked: true });
            e.in_sync = true;
            e.tracked = true;
        }
        for aid in &init {
            self.copies.entry((*aid).to_string()).or_insert(CopyCheckpointState { local_checkpoint: UNASSIGNED_SEQ_NO, in_sync: false, tracked: true });
        }
        self.recompute();
    }

    /// Apply the routing table's view of the shard: `assigned` = the
    /// (allocation id, node id) of its assigned copies, `copies` = every
    /// copy it lists (ES `ReplicationGroup`: replication targets plus
    /// skipped shards, the `_shards.total` of a write).
    pub fn set_routing<'a>(&mut self, assigned: impl IntoIterator<Item = (&'a str, &'a str)>, copies: u32) {
        self.routing = Some((assigned.into_iter().map(|(a, n)| (a.to_string(), n.to_string())).collect(), copies));
    }

    /// The node the routing table assigns `allocation_id` to.
    pub fn routed_node(&self, allocation_id: &str) -> Option<String> {
        self.routing.as_ref().and_then(|(assigned, _)| assigned.get(allocation_id).cloned())
    }

    /// Copies that have processed everything through `seq_no` and that the
    /// routing table still lists: the primary and every replication target
    /// that acknowledged it (a write's `_shards.successful`).
    pub fn acked_copies(&self, seq_no: i64) -> usize {
        self.copies.iter().filter(|(aid, c)| c.tracked && c.local_checkpoint >= seq_no
            && (**aid == self.primary_allocation_id || self.routing.as_ref().is_none_or(|(assigned, _)| assigned.contains_key(aid.as_str())))).count()
    }

    /// The number of copies the routing table lists (see [`Self::set_routing`]).
    pub fn routing_copies(&self) -> Option<u32> {
        self.routing.as_ref().map(|(_, n)| *n)
    }

    /// An in-sync copy the routing table no longer has (ES
    /// `ReplicationGroup.getUnavailableInSyncShards`): its node left or it
    /// was failed, and nothing can reach it. The primary marks it stale on
    /// its next write, without reporting a failure.
    pub fn is_unavailable_in_sync(&self, allocation_id: &str) -> bool {
        allocation_id != self.primary_allocation_id && self.is_in_sync(allocation_id)
            && self.routing.as_ref().is_some_and(|(assigned, _)| !assigned.contains_key(allocation_id))
    }

    /// Start tracking a copy that is about to be recovered (not in sync).
    pub fn init_tracking(&mut self, allocation_id: &str) {
        self.copies.entry(allocation_id.to_string()).or_insert(CopyCheckpointState { local_checkpoint: UNASSIGNED_SEQ_NO, in_sync: false, tracked: true });
    }

    /// Stop tracking a copy (failed / removed).
    pub fn remove(&mut self, allocation_id: &str) {
        if allocation_id != self.primary_allocation_id {
            self.copies.remove(allocation_id);
            self.recompute();
        }
    }

    /// Record a copy's local checkpoint (monotonic; unknown ids ignored).
    /// Returns whether the global checkpoint advanced.
    pub fn update_local_checkpoint(&mut self, allocation_id: &str, checkpoint: i64) -> bool {
        let Some(c) = self.copies.get_mut(allocation_id) else { return false };
        if checkpoint <= c.local_checkpoint {
            return false;
        }
        c.local_checkpoint = checkpoint;
        let before = self.global_checkpoint;
        self.recompute();
        self.global_checkpoint != before
    }

    /// Mark a recovering copy in sync. Succeeds (`Ok(true)`) only once its
    /// local checkpoint has reached the global checkpoint (ES
    /// `markAllocationIdAsInSync`); `Ok(false)` = not yet, retry after
    /// more acks.
    pub fn mark_in_sync(&mut self, allocation_id: &str) -> Result<bool, String> {
        let gcp = self.global_checkpoint;
        let c = self.copies.get_mut(allocation_id).ok_or_else(|| format!("allocation id [{allocation_id}] is not tracked"))?;
        if c.in_sync {
            return Ok(true);
        }
        if c.local_checkpoint < gcp || c.local_checkpoint == UNASSIGNED_SEQ_NO {
            return Ok(false);
        }
        c.in_sync = true;
        self.recompute();
        Ok(true)
    }

    pub fn global_checkpoint(&self) -> i64 {
        self.global_checkpoint
    }

    pub fn local_checkpoint(&self, allocation_id: &str) -> Option<i64> {
        self.copies.get(allocation_id).map(|c| c.local_checkpoint)
    }

    pub fn is_in_sync(&self, allocation_id: &str) -> bool {
        self.copies.get(allocation_id).is_some_and(|c| c.in_sync)
    }

    pub fn in_sync_ids(&self) -> Vec<String> {
        self.copies.iter().filter(|(_, c)| c.in_sync).map(|(a, _)| a.clone()).collect()
    }

    /// Every tracked copy except the primary: where replication requests go.
    pub fn replication_targets(&self) -> Vec<String> {
        self.copies.iter().filter(|(a, c)| c.tracked && **a != self.primary_allocation_id).map(|(a, _)| a.clone()).collect()
    }

    /// Lowest seq_no some tracked copy may still need (`min(cp) + 1`);
    /// translog ops at or above it must be retained.
    pub fn min_retained_seq_no(&self) -> i64 {
        self.copies.values().filter(|c| c.tracked).map(|c| c.local_checkpoint.max(NO_OPS_PERFORMED) + 1).min().unwrap_or(0)
    }

    /// Whether every tracked copy (in sync or recovering) has processed
    /// everything up to `seq_no` — the precondition of a safe flush.
    pub fn all_tracked_acked(&self, seq_no: i64) -> bool {
        self.copies.values().filter(|c| c.tracked).all(|c| c.local_checkpoint >= seq_no)
    }

    /// Whether every in-sync copy has processed up to `seq_no` (write ack).
    pub fn all_in_sync_acked(&self, seq_no: i64) -> bool {
        self.copies.values().filter(|c| c.in_sync).all(|c| c.local_checkpoint >= seq_no)
    }

    /// Tracked copies whose checkpoint is below `seq_no`.
    pub fn lagging(&self, seq_no: i64) -> Vec<(String, i64)> {
        self.copies.iter().filter(|(_, c)| c.tracked && c.local_checkpoint < seq_no).map(|(a, c)| (a.clone(), c.local_checkpoint)).collect()
    }

    pub fn stats(&self) -> ReplicationTrackerStats {
        ReplicationTrackerStats {
            primary_allocation_id: self.primary_allocation_id.clone(),
            primary_term: self.primary_term,
            global_checkpoint: self.global_checkpoint,
            min_retained_seq_no: self.min_retained_seq_no(),
            copies: self.copies.clone(),
        }
    }

    fn recompute(&mut self) {
        if let Some(min) = self.copies.values().filter(|c| c.in_sync).map(|c| c.local_checkpoint).min() {
            if min > self.global_checkpoint {
                self.global_checkpoint = min;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::seq::SliceRandom;
    use rand::{Rng, SeedableRng};

    #[test]
    fn local_checkpoint_out_of_order() {
        let mut t = LocalCheckpointTracker::default();
        assert_eq!(t.checkpoint(), -1);
        assert_eq!(t.mark_processed(1), -1);
        assert_eq!(t.mark_processed(2), -1);
        assert!(t.has_processed(2));
        assert!(!t.has_processed(0));
        assert_eq!(t.mark_processed(0), 2);
        assert_eq!(t.max_seq_no(), 2);
        assert_eq!(t.mark_processed(5), 2);
        assert_eq!(t.fast_forward(3), 3);
        assert_eq!(t.mark_processed(4), 5);
        assert_eq!(t.fast_forward(1), 5);
        assert_eq!(t.pending_count(), 0);
        assert_eq!(t.fast_forward(9), 9);
        assert_eq!(t.max_seq_no(), 9);
    }

    /// Property: for random permutations / duplicates of 0..n with a random
    /// subset processed, the checkpoint is the length of the processed
    /// prefix minus one, max_seq_no the max processed.
    #[test]
    fn local_checkpoint_property() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..300 {
            let n = rng.gen_range(0..200i64);
            let mut seqs: Vec<i64> = (0..n).filter(|_| rng.gen_bool(0.9)).collect();
            let dups: Vec<i64> = seqs.iter().copied().filter(|_| rng.gen_bool(0.2)).collect();
            seqs.extend(dups);
            seqs.shuffle(&mut rng);
            let mut t = LocalCheckpointTracker::default();
            let mut seen = BTreeSet::new();
            let mut last_cp = -1;
            for s in &seqs {
                let cp = t.mark_processed(*s);
                assert!(cp >= last_cp, "checkpoint never goes backwards");
                last_cp = cp;
                seen.insert(*s);
                let expected = (0..).take_while(|i| seen.contains(i)).count() as i64 - 1;
                assert_eq!(cp, expected);
                assert!(t.has_processed(*s));
            }
            assert_eq!(t.max_seq_no(), seen.iter().next_back().copied().unwrap_or(-1));
            for s in 0..n {
                assert_eq!(t.has_processed(s), seen.contains(&s));
            }
        }
    }

    #[test]
    fn global_checkpoint_rules() {
        let mut t = ReplicationTracker::new("p", 1, 10);
        assert_eq!(t.global_checkpoint(), 10);
        t.update_from_cluster_state(["p", "r1"], ["r2"]);
        // r1 unknown -> global checkpoint held at 10 (never goes back).
        assert_eq!(t.global_checkpoint(), 10);
        assert_eq!(t.replication_targets(), vec!["r1".to_string(), "r2".to_string()]);
        assert_eq!(t.min_retained_seq_no(), -1 + 1);
        t.update_local_checkpoint("r1", 7);
        assert_eq!(t.global_checkpoint(), 10);
        t.update_local_checkpoint("p", 15);
        t.update_local_checkpoint("r1", 12);
        assert_eq!(t.global_checkpoint(), 12);
        assert!(t.all_in_sync_acked(12));
        assert!(!t.all_tracked_acked(12), "recovering r2 still behind");
        // r2 cannot become in-sync below the global checkpoint.
        t.update_local_checkpoint("r2", 11);
        assert_eq!(t.mark_in_sync("r2"), Ok(false));
        t.update_local_checkpoint("r2", 12);
        assert_eq!(t.mark_in_sync("r2"), Ok(true));
        assert_eq!(t.in_sync_ids(), vec!["p".to_string(), "r1".to_string(), "r2".to_string()]);
        assert_eq!(t.min_retained_seq_no(), 13);
        // Checkpoints are monotonic per copy.
        assert!(!t.update_local_checkpoint("r1", 3));
        assert_eq!(t.local_checkpoint("r1"), Some(12));
        // Failing r1: removed via cluster state; gcp does not go back.
        t.update_from_cluster_state(["p", "r2"], []);
        assert_eq!(t.replication_targets(), vec!["r2".to_string()]);
        assert_eq!(t.global_checkpoint(), 12);
        t.update_local_checkpoint("r2", 15);
        assert_eq!(t.global_checkpoint(), 15);
        assert!(t.mark_in_sync("zz").is_err());
    }

    /// Property: random interleavings of checkpoint updates keep the global
    /// checkpoint = max over time of min(in-sync cps), monotone, and
    /// min_retained <= gcp + 1.
    #[test]
    fn global_checkpoint_property() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for _ in 0..200 {
            let ids = ["p", "a", "b", "c"];
            let mut t = ReplicationTracker::new("p", 1, -1);
            t.update_from_cluster_state(["p", "a", "b"], ["c"]);
            let mut cps: BTreeMap<&str, i64> = ids.iter().map(|i| (*i, -2)).collect();
            cps.insert("p", -1);
            let mut last = t.global_checkpoint();
            for _ in 0..100 {
                let id = ids[rng.gen_range(0..ids.len())];
                let cp = rng.gen_range(-1..100);
                t.update_local_checkpoint(id, cp);
                let e = cps.get_mut(id).unwrap();
                *e = (*e).max(cp);
                if rng.gen_bool(0.05) {
                    let _ = t.mark_in_sync("c");
                }
                let gcp = t.global_checkpoint();
                assert!(gcp >= last);
                last = gcp;
                let min_in_sync = t.in_sync_ids().iter().map(|i| cps[i.as_str()]).min().unwrap();
                // Copies only join the in-sync set at/above the gcp, so the
                // gcp is exactly the in-sync minimum (floored at the start).
                assert_eq!(gcp, min_in_sync.max(-1));
                for i in t.in_sync_ids() {
                    assert!(cps[i.as_str()].max(-1) >= gcp);
                }
                assert!(t.min_retained_seq_no() <= cps.values().copied().min().unwrap().max(-1) + 1);
            }
        }
    }
}
