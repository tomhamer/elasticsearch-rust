//! EC2-only read-only diagnostic. Never opens or mutates the live shard engine.
use rustic_core::{codec::directory::MmapDirectory, segment::Segment, query::reader::LeafReader};
use std::{path::PathBuf, time::Instant};
fn main() {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().expect("segment path"));
    let field: u16 = args.next().expect("field id").parse().unwrap();
    let id = path.file_stem().unwrap().to_str().unwrap().strip_prefix("seg_").unwrap().parse().unwrap();
    let dir = MmapDirectory::open(path.parent().unwrap()).unwrap();
    let segment = Segment::open(&dir, id).unwrap();
    let leaf = LeafReader::Segment(&segment);
    let start = Instant::now();
    let max = leaf.max_i64_value(field);
    let optimized_ms = start.elapsed().as_secs_f64()*1000.0;
    // Bound the old quadratic path; compare precisely the same prefix values.
    let prefix = leaf.num_docs().min(1000);
    let start = Instant::now();
    let old_prefix = (0..prefix).filter_map(|d| leaf.i64_value(field,d)).max();
    let old_prefix_ms = start.elapsed().as_secs_f64()*1000.0;
    let ff = segment.fast_fields();
    let col = ff.get_i64(field).expect("column present");
    assert_eq!(old_prefix, (0..prefix).filter_map(|d| col.get(d)).max());
    assert_eq!(max, (0..leaf.num_docs()).filter_map(|d| col.get(d)).max());
    println!("{}", serde_json::json!({"documents":leaf.num_docs(), "maximum":max, "optimized_full_ms":optimized_ms, "old_prefix_documents":prefix, "old_prefix_ms":old_prefix_ms, "exact_full_scan_matches":true}));
}
