//! Native graph storage accounting; source and query ownership are separate.
use rustic_core::vector::hnsw::MutableHnsw;
use serde_json::{Value, json};

pub(super) fn report(graph: &MutableHnsw) -> Value {
    let memory = graph.storage_memory();
    macro_rules! arrays {
        ($store:expr) => {
            json!({
                "allocated_payload_bytes":$store.allocated_payload_bytes,
                "used_payload_bytes":$store.used_payload_bytes,
                "dead_payload_bytes":$store.dead_payload_bytes,
                "held_payload_bytes":$store.held_payload_bytes,
                "metadata_bytes":$store.metadata_bytes,
                "accounted_bytes":$store.accounted_bytes
            })
        };
    }
    let nodes = json!({"current_payload_bytes":memory.node_current_payload_bytes,
        "held_payload_bytes":memory.node_held_payload_bytes,"accounted_bytes":memory.node_accounted_bytes});
    let generation = json!({"accounted_bytes":memory.generation_accounted_bytes,
        "cells":memory.generation_cells,"guard_references":memory.guard_references});
    let lifecycle = json!({"free_slot_capacity":memory.free_slot_capacity,
        "retirement_capacity":memory.retirement_capacity,
        "pending_reservation_capacity":memory.pending_reservation_capacity,
        "visibility_retirement_capacity":memory.visibility_retirement_capacity,
        "free_slot_payload_bytes":memory.free_slot_payload_bytes,
        "retirement_payload_bytes":memory.retirement_payload_bytes,
        "hash_capacity_payload_bytes_lower_bound":memory.hash_capacity_payload_bytes_lower_bound,
        "outstanding_ack_tokens":memory.outstanding_ack_tokens,
        "outstanding_ack_payload_bytes":memory.outstanding_ack_payload_bytes});
    json!({"levels":arrays!(memory.levels),"links":arrays!(memory.links),
        "nodes":nodes,"generation":generation,"lifecycle":lifecycle,
        "graph_struct_bytes":memory.graph_struct_bytes,
        "accounted_bytes_lower_bound":memory.accounted_bytes_lower_bound,
        "scope":"graph storage and outstanding lifecycle-token capacity lower bound, not process RSS; array payload includes buffer allocation headers and large-array descriptors; excludes source rows, prepared operations, query/snapshot payload, allocator/Arc overhead and unexposed HashMap bucket/control allocation"})
}
