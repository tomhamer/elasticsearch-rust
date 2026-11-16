//! Watcher scheduling on a cluster (ES `WatcherLifeCycleService` /
//! `WatcherIndexingListener`): a node triggers the watches whose `.watches`
//! documents it holds a copy of. Every active copy (primary or replica) of a
//! `.watches` shard takes a share: the copies are ordered by allocation id
//! and a watch belongs to the copy at `floorMod(murmur3(id), copies)`
//! (ES `WatcherIndexingListener.Configuration.shouldBeTriggered`). So each
//! watch fires on exactly one node per trigger, and moves with its copy when
//! the shard relocates or a node fails (a replica is promoted, the next
//! state assigns the survivors).
//!
//! The scheduler ticks every 250 ms. It reloads the watch set when this
//! node's assignment or a local `.watches` copy changed (its max sequence
//! number moves on every put, delete or status update), then fires the due
//! watches it owns. Executions run as the subject stored with each watch
//! (`elasticsearch_engine::watcher::exec::stored_identity`), their searches
//! and writes through this node's coordinator.

use crate::node::ClusterNode;
use elasticsearch_cluster::state::ClusterState;
use elasticsearch_engine::watcher;
use parking_lot::Mutex;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

/// A watch a node starts owning is scheduled from `now + GRACE_MS`: the
/// previous owner applies the same cluster state within it, so a trigger
/// both could see is fired by one of them only.
pub const GRACE_MS: i64 = 1000;

/// One node's view of the `.watches` copies: per shard, the active copies'
/// allocation ids in order and the positions of this node's copies.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Assignment {
    /// shard -> (sorted active allocation ids, indices of the local ones)
    pub shards: BTreeMap<u32, (Vec<String>, Vec<usize>)>,
}

impl Assignment {
    pub fn of(state: &ClusterState, local: &str) -> Assignment {
        let mut shards = BTreeMap::new();
        if let Some(irt) = state.routing_table.indices.get(watcher::WATCHES_INDEX) {
            for t in &irt.shards {
                let mut copies: Vec<(&str, bool)> = t
                    .active_copies()
                    .filter_map(|c| c.allocation_id().map(|a| (a, c.node_id.as_deref() == Some(local))))
                    .collect();
                copies.sort_by(|a, b| a.0.cmp(b.0));
                let mine: Vec<usize> = copies.iter().enumerate().filter(|(_, c)| c.1).map(|(i, _)| i).collect();
                shards.insert(t.shard, (copies.iter().map(|c| c.0.to_string()).collect(), mine));
            }
        }
        Assignment { shards }
    }

    /// Does this node hold any active `.watches` copy?
    pub fn any_local(&self) -> bool {
        self.shards.values().any(|(_, mine)| !mine.is_empty())
    }

    /// Does this node trigger the watch `id` stored in `shard`?
    pub fn owns(&self, shard: u32, id: &str) -> bool {
        let Some((copies, mine)) = self.shards.get(&shard) else { return false };
        if copies.is_empty() {
            return false;
        }
        let pick = (elasticsearch_cluster::routing::murmur3_hash(id) as i64).rem_euclid(copies.len() as i64) as usize;
        mine.contains(&pick)
    }
}

#[derive(Default)]
struct NodeState {
    running: AtomicBool,
    /// What the last reload saw: the assignment and the local copies' max
    /// sequence numbers.
    seen: Mutex<Option<(Assignment, Vec<(u32, i64)>)>>,
}

fn states() -> &'static Mutex<HashMap<String, Arc<NodeState>>> {
    static S: OnceLock<Mutex<HashMap<String, Arc<NodeState>>>> = OnceLock::new();
    S.get_or_init(Default::default)
}

fn node_state(node: &ClusterNode) -> Arc<NodeState> {
    states().lock().entry(node.local.id.clone()).or_default().clone()
}

/// Install the coordinator call factory the watcher's background work uses
/// (once, when the scheduler starts; must run on the node's runtime).
pub fn install(node: &Arc<ClusterNode>) {
    watcher::set_rest_factory(Some(crate::front::watcher_rest_factory(node)));
}

/// One scheduler tick (off the runtime: reloads and executions block).
/// Skipped while the previous tick still runs.
pub fn tick(node: &Arc<ClusterNode>) {
    let st = node_state(node);
    if st.running.swap(true, Ordering::SeqCst) {
        return;
    }
    let _done = scopeguard(|| st.running.store(false, Ordering::SeqCst));
    let engine = node.engine.clone();
    if !watcher::watcher_enabled(&engine) {
        return;
    }
    let svc = watcher::service(&engine);
    let state = node.state();
    let assignment = Assignment::of(&state, &node.local.id);
    let now = elasticsearch_engine::date::now_millis();
    if !assignment.any_local() {
        // Nothing to trigger here: forget any schedule (a later
        // assignment starts from its own grace period).
        let _ = watcher::due_watches_where(&svc, now, GRACE_MS, &|_| false);
        *st.seen.lock() = None;
        return;
    }
    let seq: Vec<(u32, i64)> = assignment
        .shards
        .iter()
        .filter(|(_, (_, mine))| !mine.is_empty())
        .map(|(shard, _)| {
            let max = node.shards.get(watcher::WATCHES_INDEX, *shard).and_then(|c| c.lock().svc.shard(*shard)).map(|s| s.shard.max_seq_no()).unwrap_or(-2);
            (*shard, max)
        })
        .collect();
    let changed = st.seen.lock().as_ref() != Some(&(assignment.clone(), seq.clone()));
    if changed {
        match watcher::load_from_cluster() {
            Ok(watches) => {
                svc.replace_all(watches);
                *st.seen.lock() = Some((assignment.clone(), seq));
            }
            Err(e) => {
                tracing::debug!("watcher: cannot load .watches yet: {e}");
                return;
            }
        }
    }
    if svc.manually_stopped() {
        return;
    }
    let md = &state.metadata;
    let owned = |id: &str| crate::shard_engines::doc_shard(md, watcher::WATCHES_INDEX, id, None).is_ok_and(|shard| assignment.owns(shard, id));
    // Triggers a previous run of this node left pending (persisted but
    // never finished) run once, for the watches this node owns (ES
    // `WatcherService.reloadInner` executes triggered watches on start).
    if !svc.recovered.load(std::sync::atomic::Ordering::SeqCst) {
        watcher::recover_triggered(&engine, &svc, &owned);
    }
    let due = watcher::due_watches_where(&svc, now, GRACE_MS, &owned);
    if !due.is_empty() {
        watcher::fire(&engine, &svc, due, now);
    }
}

/// Run `f` when dropped.
fn scopeguard<F: FnOnce()>(f: F) -> impl Drop {
    struct G<F: FnOnce()>(Option<F>);
    impl<F: FnOnce()> Drop for G<F> {
        fn drop(&mut self) {
            if let Some(f) = self.0.take() {
                f();
            }
        }
    }
    G(Some(f))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assignment(copies: &[&str], mine: &[usize]) -> Assignment {
        let mut shards = BTreeMap::new();
        shards.insert(0, (copies.iter().map(|c| c.to_string()).collect(), mine.to_vec()));
        Assignment { shards }
    }

    #[test]
    fn every_watch_has_exactly_one_owning_copy() {
        let copies = ["a1", "b2", "c3"];
        let nodes: Vec<Assignment> = (0..3).map(|i| assignment(&copies, &[i])).collect();
        let mut per_node = [0usize; 3];
        for n in 0..200 {
            let id = format!("watch-{n}");
            let owners: Vec<usize> = (0..3).filter(|i| nodes[*i].owns(0, &id)).collect();
            assert_eq!(owners.len(), 1, "{id}: {owners:?}");
            per_node[owners[0]] += 1;
        }
        assert!(per_node.iter().all(|c| *c > 30), "watches spread over the copies: {per_node:?}");
        // A node without a copy owns nothing; the only copy owns everything.
        assert!(!assignment(&copies, &[]).owns(0, "w"));
        assert!(assignment(&["a1"], &[0]).owns(0, "w"));
        assert!(!assignment(&["a1"], &[0]).owns(1, "w"));
    }
}
