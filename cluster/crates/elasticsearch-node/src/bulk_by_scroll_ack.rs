//! Replicated acknowledgement of BulkByScroll write batches.
//!
//! The engine runs `_update_by_query`, `_delete_by_query` and colocated
//! `_reindex` against its local primary copy, also in the background
//! (`wait_for_completion=false`), long after the request that started it
//! was acknowledged. ES's `BulkByScrollTask` sends every batch through the
//! ordinary bulk action, which answers only once the in-sync copies have
//! the writes: a batch counts, and the task completes, only once it would
//! survive the loss of the primary. [`ClusterReplicationBarrier`] gives the
//! engine that acknowledgement: after each batch (and once more before the
//! task completes) it ships the copy's operations to every in-sync copy and
//! waits for them, as [`crate::front::execute_write`] does for a write.

use crate::node::{ClusterNode, TransportNet};
use elasticsearch_engine::error::{EsError, Result};
use elasticsearch_engine::index::IndexService;
use std::sync::{Arc, Weak};

pub struct ClusterReplicationBarrier {
    node: Weak<ClusterNode>,
    runtime: tokio::runtime::Handle,
}

impl ClusterReplicationBarrier {
    /// Created on the node's runtime (the replication runs there).
    pub fn new(node: Weak<ClusterNode>) -> ClusterReplicationBarrier {
        ClusterReplicationBarrier { node, runtime: tokio::runtime::Handle::current() }
    }
}

/// ES `RetryOnPrimaryException`: executed on the primary, not acknowledged.
fn not_acked(index: &str, shard: Option<u32>, why: &str) -> EsError {
    let shard = shard.map(|k| k.to_string()).unwrap_or_else(|| "_na".into());
    EsError::new(500, "retry_on_primary_exception", format!("[{index}][{shard}] {why}"))
}

impl elasticsearch_engine::reindex::ReplicationBarrier for ClusterReplicationBarrier {
    fn await_replicated(&self, svc: &IndexService) -> Result<()> {
        let Some(node) = self.node.upgrade() else { return Ok(()) };
        // An engine-local index (not in the cluster metadata) has no other
        // copies to wait for.
        if !node.state().metadata.indices.contains_key(&svc.name) {
            return Ok(());
        }
        // Which shard engine holds this copy.
        let mut shard = None;
        let _ = node.try_visit_shard_engines(|k, engine| {
            if engine.get_index(&svc.name).is_some_and(|s| std::ptr::eq::<IndexService>(&*s, svc)) {
                shard = Some(k);
                return Err(());
            }
            Ok(())
        });
        let Some(k) = shard else {
            return Err(not_acked(&svc.name, None, "the written copy was removed before its writes were acknowledged"));
        };
        // Written as this node's primary: it must still be that primary.
        let Some(p) = node.shards.primary(&svc.name, k).filter(|p| std::ptr::eq::<IndexService>(&*p.svc, svc)) else {
            return Err(not_acked(&svc.name, Some(k), "primary shard copy was removed from this node while the operation was in flight"));
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let n = node.clone();
        self.runtime.spawn(async move {
            let seq = p.max_seq_no_async().await;
            let out = match p.replicate_through(seq, &TransportNet(n.clone())).await {
                Ok(_) if n.shards.primary(&p.index, p.shard).is_some_and(|cur| Arc::ptr_eq(&cur, &p)) => Ok(()),
                Ok(_) => Err("primary shard copy was removed from this node while the operation was in flight".to_string()),
                Err(e) => Err(format!("replication failed and the failed copy could not be removed: {e}")),
            };
            let _ = tx.send(out);
        });
        match rx.recv() {
            Ok(Ok(())) => Ok(()),
            Ok(Err(why)) => {
                tracing::warn!(index = %svc.name, shard = k, "bulk by scroll batch not acknowledged: {why}");
                Err(not_acked(&svc.name, Some(k), &why))
            }
            Err(_) => Err(not_acked(&svc.name, Some(k), "replication stopped before acknowledging the writes")),
        }
    }
}
