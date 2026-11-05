//! Engine hooks for the cluster layer (`cluster/docs/ENGINE_HOOKS.md`) and
//! sharding phase 1 (`cluster/docs/SHARDING_PLAN.md`). Documented in
//! `docs/ENGINE_HOOKS_IMPL.md`.
//!
//! | hook | API |
//! |---|---|
//! | 1 create with uuid | [`Node::create_index_with_uuid`](crate::node::Node::create_index_with_uuid) |
//! | 2 adopt from files | [`Node::adopt_index`](crate::node::Node::adopt_index), [`Node::reopen_index`](crate::node::Node::reopen_index), [`IndexService::snapshot_files`](crate::index::IndexService::snapshot_files) |
//! | 3 replica apply | [`IndexService::apply_replica_ops`](crate::index::IndexService::apply_replica_ops) |
//! | 4 primary term | [`Shard::set_primary_term`](crate::shard::Shard::set_primary_term), [`Shard::primary_term`](crate::shard::Shard::primary_term) |
//! | 5 history | [`Shard::changes_since`](crate::shard::Shard::changes_since), [`Shard::acquire_retention_lock`](crate::shard::Shard::acquire_retention_lock), retention leases |
//! | 9 query / fetch phase | [`search_ext`] (`IndexService::open_searcher` / `query_phase` / `fetch`) |
//! | 10 agg partials | [`AggPartial`], [`AggRequest`] |
//! | 8 parse | [`IndexService::parse_for_replica`](crate::index::IndexService::parse_for_replica) |
//!
//! The `impl` blocks live in `shard_ext.rs`, `index_ext.rs` and
//! `node_ext.rs`, compiled as private child modules of `shard`, `index` and
//! `node` so they can reach those types' internals without widening them.

pub mod changes;
pub mod files;
pub mod history;
pub mod retention;
pub mod tablets;

pub use crate::search::phases as search_ext;
pub use crate::aggs::partial as agg_partial;
pub use agg_partial::{AggPartial, AggRequest};
pub use changes::{ApplyOutcome, ChangeOp, ChangesSnapshot, Durability, ReplicaApplyResult};
pub use files::{IndexFileEntry, IndexFileSet, IndexManifest};
pub use history::{SeqNoState, TermHistory};
pub use retention::{RetentionLease, RetentionLock};
pub use tablets::{Tablet, TabletMap, TabletRange, TabletState};

static VOLATILE_ROOTS: parking_lot::RwLock<Vec<std::path::PathBuf>> = parking_lot::RwLock::new(Vec::new());

/// Mark everything under `root` volatile: [`write_durable`] still writes
/// atomically there but skips the fsyncs (a host's throwaway copy that is
/// rebuilt from elsewhere whenever it is opened).
pub fn set_volatile_root(root: &std::path::Path) {
    let mut roots = VOLATILE_ROOTS.write();
    if !roots.iter().any(|r| r == root) {
        roots.push(root.to_path_buf());
    }
}

fn volatile(path: &std::path::Path) -> bool {
    let roots = VOLATILE_ROOTS.read();
    !roots.is_empty() && roots.iter().any(|r| path.starts_with(r))
}

/// Write `bytes` to `path` atomically and durably: temp file, fsync,
/// rename, fsync of the parent directory (no fsync under a volatile root).
pub(crate) fn write_durable(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let durable = !volatile(path);
    // Unique per write: two concurrent writers of one file never rename (or
    // truncate) each other's temp file.
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = path.with_extension(format!("{}.{n}.tmp~", std::process::id()));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        if durable {
            f.sync_all()?;
        }
    }
    std::fs::rename(&tmp, path)?;
    if durable {
        sync_dir(path.parent());
    }
    Ok(())
}

/// fsync a directory (makes renames/creates in it durable). Best effort.
pub(crate) fn sync_dir(dir: Option<&std::path::Path>) {
    if let Some(d) = dir {
        if let Ok(f) = std::fs::File::open(d) {
            let _ = f.sync_all();
        }
    }
}
