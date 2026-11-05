//! Segment ordering by `@timestamp` (ES `LeafSorter`): indices that had a
//! `@timestamp` date field when they were opened and no `index.sort` search
//! their segments by max `@timestamp` descending, so score ties (and
//! `sort: @timestamp desc` early termination) see the newest segment first.

use crate::mapping::FieldLookup;
use parking_lot::Mutex;
use rustic_core::query::LeafReader;
use std::collections::HashMap;
use std::sync::OnceLock;

type Cache = Mutex<HashMap<(u64, u32, u16), Option<i64>>>;

fn cache() -> &'static Cache {
    static C: OnceLock<Cache> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Max value of the i64 column `field` over all docs of the leaf (`None`: no values).
pub fn leaf_max(leaf: &LeafReader<'_>, field: u16) -> Option<i64> {
    let scan = || (0..leaf.num_docs()).filter_map(|d| leaf.i64_values(field, d).into_iter().max()).max();
    match leaf {
        // Memtables change under us: never cached.
        LeafReader::Memtable(_, ..) => scan(),
        LeafReader::Segment(segment) => {
            let key = (segment.immutable_cache_identity(), leaf.num_docs(), field);
            if let Some(v) = cache().lock().get(&key) {
                return *v;
            }
            // A single-valued column's max from its block maxima, not one
            // column open and read per doc (a 10M-doc merged segment took
            // seconds, in the first search after its merge).
            let v = match segment.fast_fields().get_i64(field) {
                Some(column) => column.max(),
                None => scan(),
            };
            let mut c = cache().lock();
            if c.len() > 8192 {
                c.clear();
            }
            c.insert(key, v);
            v
        }
    }
}

/// Order `leaves` by max `@timestamp` descending (leaves without values last; stable).
pub fn order_leaves(leaves: &mut Vec<LeafReader<'_>>, lookup: &FieldLookup) {
    if leaves.len() < 2 {
        return;
    }
    let Some(id) = lookup.get("@timestamp").and_then(|f| f.core_id) else { return };
    let mut keyed: Vec<(Option<i64>, LeafReader<'_>)> = leaves.drain(..).map(|l| (leaf_max(&l, id), l)).collect();
    keyed.sort_by(|a, b| b.0.cmp(&a.0));
    leaves.extend(keyed.into_iter().map(|(_, l)| l));
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustic_core::codec::directory::RamDirectory;
    use rustic_core::segment::{Segment, SegmentWriter};
    use rustic_core::types::{Document, Field, FieldType};

    fn segment(value: i64) -> Segment {
        let mut writer = SegmentWriter::new(0);
        let mut doc = Document::new();
        doc.add_i64(0, value);
        writer.add_document(&doc, &[Field::new(0, "@timestamp", FieldType::I64)]);
        writer.write_to(&RamDirectory::new()).unwrap()
    }

    #[test]
    fn timestamp_max_cache_does_not_reuse_replaced_segment_address() {
        // Reuse one allocation deterministically, as allocator reuse can do
        // after closing an index. Field, segment ID and document count match.
        let mut slot = Box::new(segment(100));
        let address = LeafReader::Segment(&slot).id();
        assert_eq!(leaf_max(&LeafReader::Segment(&slot), 0), Some(100));
        *slot = segment(200);
        assert_eq!(LeafReader::Segment(&slot).id(), address);
        assert_eq!(leaf_max(&LeafReader::Segment(&slot), 0), Some(200));
    }
}
