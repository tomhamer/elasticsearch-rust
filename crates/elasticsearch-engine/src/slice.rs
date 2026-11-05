//! Sliced scroll / search (`slice: {id, max, field}`): each slice is a
//! disjoint subset of the matching documents.
//!
//! As ES `SliceBuilder.toFilter`: with at least as many shards searched as
//! slices, a slice is a set of whole shards (`shard % max == id`);
//! otherwise slice `id` belongs to shard `id % shards` and that shard's
//! documents are split by a hash of the `_id` over the slices it holds.

use crate::mapping::ID_FIELD;
use crate::tsdb::murmur3_32;
use rustic_core::query::query::{CustomQuery, ScoredDocs, SearchContext};
use rustic_core::query::{LeafReader, Scorer};
use rustic_core::types::DocId;

#[derive(Debug)]
pub struct SliceQuery {
    pub id: u32,
    pub max: u32,
    /// Position of the searched shard among the request's shards and their
    /// number (ES `shardRequestIndex` / `numberOfShards`).
    pub shard_index: u32,
    pub num_shards: u32,
}

/// ES default `number_of_routing_shards` for an index without the setting.
pub fn default_routing_shards(shards: u32) -> u32 {
    let shards = shards.max(1);
    let log2_shards = 32 - (shards - 1).leading_zeros();
    let splits = 10u32.saturating_sub(log2_shards).max(1);
    shards << splits
}

/// ES `OperationRouting`: shard of a document id.
pub fn shard_of(id: &str, shards: u32, routing_shards: u32) -> u32 {
    let mut bytes = Vec::with_capacity(id.len() * 2);
    for u in id.encode_utf16() {
        bytes.push(u as u8);
        bytes.push((u >> 8) as u8);
    }
    let h = murmur3_32(&bytes, 0) as i32;
    let factor = (routing_shards / shards.max(1)).max(1) as i64;
    ((h as i64).rem_euclid(routing_shards as i64) / factor) as u32
}

impl SliceQuery {
    /// `Some(all)`: the whole shard is in (true) or out (false) of the
    /// slice; `None`: documents are split within the shard.
    fn whole_shard(&self) -> Option<bool> {
        let n = self.num_shards.max(1);
        if n >= self.max {
            return Some(self.shard_index % self.max.max(1) == self.id);
        }
        (self.id % n != self.shard_index).then_some(false)
    }

    fn keep(&self, doc_id: &str) -> bool {
        let n = self.num_shards.max(1);
        let target = self.id % n;
        let mut in_shard = self.max / n;
        if self.max % n > target {
            in_shard += 1;
        }
        let shard_slice = self.id / n;
        (murmur3_32(doc_id.as_bytes(), 7919) as i32 as i64).rem_euclid(in_shard.max(1) as i64) as u32 == shard_slice
    }
}

impl CustomQuery for SliceQuery {
    fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        let live = leaf.live_docs();
        match self.whole_shard() {
            Some(false) => return None,
            Some(true) => {
                let mut all = Vec::new();
                for d in 0..leaf.num_docs() {
                    if !live(d) { continue; }
                    let doc = match leaf.get_doc(d) { Ok(doc) => doc, Err(e) => { record_decode_error(ctx, e); return None; } };
                    if doc.is_some_and(|doc| doc.get(ID_FIELD).is_some()) { all.push((d, 0.0)); }
                }
                return if all.is_empty() { None } else { Some(Box::new(ScoredDocs::new(all))) };
            }
            None => {}
        }
        let mut out: Vec<(DocId, f32)> = Vec::new();
        for d in 0..leaf.num_docs() {
            if !live(d) {
                continue;
            }
            let doc = match leaf.get_doc(d) { Ok(doc) => doc.unwrap_or_default(), Err(e) => { record_decode_error(ctx, e); return None; } };
            if let Some(id) = doc.get(ID_FIELD).and_then(|v| v.as_text()) {
                if self.keep(id) {
                    out.push((d, 0.0));
                }
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(Box::new(ScoredDocs::new(out)))
        }
    }
    fn name(&self) -> &'static str {
        "slice"
    }
}

fn record_decode_error(ctx: &SearchContext, error: rustic_core::RusticError) {
    match error {
        rustic_core::RusticError::Memory(reason) => ctx.record_memory_error(reason),
        other => { *ctx.execution_error.lock() = Some(other.to_string()); },
    }
}
