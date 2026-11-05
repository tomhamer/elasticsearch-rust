//! ES-compatible document routing. The implementation (Murmur3 over UTF-16
//! code units, `routing_num_shards`, partitioned routing) lives in the
//! shared `elasticsearch-routing` crate so the engine and the cluster layer
//! use one source of truth; this module re-exports it and adds the bridge
//! from cluster metadata.

pub use elasticsearch_routing::*;

use crate::state::IndexMetadata;

/// The routing function of an index, from its metadata.
pub fn index_routing(meta: &IndexMetadata) -> IndexRouting {
    let partition = meta
        .setting("index.routing_partition_size")
        .and_then(|s| s.parse().ok())
        .or_else(|| meta.settings.get("index.routing_partition_size").and_then(|v| v.as_u64()).map(|v| v as u32))
        .unwrap_or(1);
    // `IndexRouting.shouldUseShardCountModRouting`: keyed on the index's
    // creation version (absent = an index from an older, 8.x-targeting build).
    let created = meta
        .settings
        .get("index.version.created")
        .and_then(|v| v.as_str().and_then(|s| s.trim().parse::<u32>().ok()).or_else(|| v.as_u64().map(|n| n as u32)))
        .unwrap_or(0);
    IndexRouting::new(meta.number_of_shards, meta.routing_num_shards, partition).with_modulo(should_use_shard_count_mod_routing(created))
}
