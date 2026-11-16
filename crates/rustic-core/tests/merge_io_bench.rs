//! Bytes written (and wall time) by an index-sorted flush and by merges:
//! a sorted merge of inputs that do not interleave, one of inputs that
//! do, and an unsorted merge of many doc-value columns.
//!
//! `cargo test --release -p rustic-core --test merge_io_bench -- --ignored --nocapture`
//! with `MERGE_IO_DIR` on the volume to measure (default: the temp dir)
//! and `MERGE_IO_DEV` its block device (e.g. `nvme1n1`) for the device's
//! sectors written. Linux only (reads `/proc/self/io`, `/proc/diskstats`).

use rustic_core::codec::directory::{Directory, MmapDirectory};
use rustic_core::index_sort::{IndexSort, IndexSortField, IndexSortKind};
use rustic_core::segment::{Segment, SegmentWriter};
use rustic_core::types::{DocId, Document, Field, FieldType, IndexSettings};
use std::sync::Arc;

#[derive(Clone, Copy, Default)]
struct Io {
    write_bytes: u64,
    cancelled: u64,
    device: u64,
}

fn io() -> Io {
    let text = std::fs::read_to_string("/proc/self/io").unwrap_or_default();
    let field = |name: &str| text.lines().find_map(|l| l.strip_prefix(name)).and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0);
    let device = std::env::var("MERGE_IO_DEV").ok().and_then(|dev| {
        let stats = std::fs::read_to_string("/proc/diskstats").ok()?;
        stats.lines().find_map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            (f.get(2) == Some(&dev.as_str())).then(|| f.get(9).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0) * 512)
        })
    }).unwrap_or(0);
    Io { write_bytes: field("write_bytes:"), cancelled: field("cancelled_write_bytes:"), device }
}

fn settle() {
    // SAFETY: sync(2) takes no arguments and cannot fail.
    unsafe { libc::sync() };
    std::thread::sleep(std::time::Duration::from_millis(500));
}

/// Run `f`, then sync: its bytes written and wall time (sync excluded).
fn measure<R>(name: &str, f: impl FnOnce() -> R) -> R {
    settle();
    let before = io();
    let started = std::time::Instant::now();
    let r = f();
    let wall = started.elapsed();
    settle();
    let after = io();
    let mb = |b: u64| b as f64 / (1 << 20) as f64;
    println!(
        "BENCH {name}: wall {:.2}s write_bytes {:.1} MB cancelled {:.1} MB net {:.1} MB device {:.1} MB",
        wall.as_secs_f64(),
        mb(after.write_bytes - before.write_bytes),
        mb(after.cancelled - before.cancelled),
        mb((after.write_bytes - before.write_bytes).saturating_sub(after.cancelled - before.cancelled)),
        mb(after.device - before.device),
    );
    r
}

fn bench_dir(name: &str) -> (tempfile::TempDir, MmapDirectory) {
    let root = std::env::var("MERGE_IO_DIR").unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
    let tmp = tempfile::Builder::new().prefix(name).tempdir_in(root).unwrap();
    let dir = MmapDirectory::open(tmp.path()).unwrap();
    (tmp, dir)
}

fn docs() -> u32 {
    std::env::var("MERGE_IO_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(250_000)
}

fn log_schema() -> Vec<Field> {
    vec![
        Field::new(0, "@timestamp", FieldType::I64),
        Field::new(1, "host", FieldType::Keyword),
        Field::new(2, "message", FieldType::Text),
        Field::new(3, "status", FieldType::I64),
        Field::new(4, "id", FieldType::Keyword),
        Field::new(5, "bytes", FieldType::I64),
    ]
}

fn log_doc(ts: i64, n: u64) -> Document {
    let mut d = Document::new();
    let h = n.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    d.add_i64(0, ts);
    d.add_keyword(1, &format!("host-{:02}", h % 50));
    d.add_text(2, &format!("GET /api/v{}/items/{} HTTP/1.1 user{} session {:x} took {}ms", h % 3, h % 10_000, h % 977, h >> 40, h % 1500));
    d.add_i64(3, [200, 200, 200, 404, 500, 301][(h % 6) as usize]);
    d.add_keyword(4, &format!("{:016x}{:04x}", h, n % 65536));
    d.add_i64(5, (h % 100_000) as i64);
    d
}

/// Segments of `per` docs each, every one in timestamp order; `interleave`:
/// segment `s` holds timestamps `t * segs + s` (their ranges interleave),
/// else `s * per + t` (disjoint ranges). Returns the merged order by
/// timestamp.
fn sorted_segments(dir: &MmapDirectory, segs: u32, per: u32, interleave: bool) -> (Vec<Arc<Segment>>, Vec<(usize, DocId)>) {
    let schema = log_schema();
    let segments: Vec<Arc<Segment>> = (0..segs).map(|s| {
        let mut w = SegmentWriter::new(s);
        for t in 0..per {
            let ts = if interleave { (t * segs + s) as i64 } else { (s * per + t) as i64 };
            w.add_document(&log_doc(ts, (s * per + t) as u64), &schema);
        }
        Arc::new(w.write_to(dir).unwrap())
    }).collect();
    let mut order: Vec<(i64, usize, DocId)> = (0..segs).flat_map(|s| (0..per).map(move |t| {
        let ts = if interleave { (t * segs + s) as i64 } else { (s * per + t) as i64 };
        (ts, s as usize, t)
    })).collect();
    order.sort_unstable();
    (segments, order.into_iter().map(|(_, s, d)| (s, d)).collect())
}

fn file_mb(dir: &MmapDirectory, id: u32) -> f64 {
    dir.open_read(&rustic_core::codec::segment_file_name(id)).unwrap().len() as f64 / (1 << 20) as f64
}

fn setup() {
    let v = rustic_core::codec::apply_cluster_segment_format(rustic_core::codec::VERSION);
    rustic_core::merge::set_merge_io_limit(0);
    // A node's memory budget (the test default is 256 MB).
    let gb: u64 = std::env::var("MERGE_IO_BUDGET_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    rustic_core::admission::configure_shared(gb << 30);
    println!("BENCH format v{v}");
}

#[test]
#[ignore]
fn merge_io_sorted_flush() {
    setup();
    let (tmp, _) = bench_dir("flush");
    let mut index = rustic_core::Index::create(tmp.path(), "bench", log_schema(), IndexSettings::default()).unwrap();
    index.set_index_sort(Some(IndexSort::new(vec![IndexSortField { field: 0, kind: IndexSortKind::Long, reverse: true, max_mode: false, missing_first: false }])));
    let n = docs() * 2;
    for i in 0..n {
        // Arrival order unlike the sort (a random-ish timestamp).
        let ts = ((i as u64).wrapping_mul(2_654_435_761) % 1_000_000_000) as i64;
        index.add_document(&log_doc(ts, i as u64)).unwrap();
    }
    measure(&format!("sorted flush ({n} docs)"), || index.refresh().unwrap());
    let segs = index.segments();
    println!("BENCH sorted flush: {} segment(s), {:.1} MB", segs.len(),
        segs.iter().map(|s| s.segment_bytes()).sum::<u64>() as f64 / (1 << 20) as f64);
}

#[test]
#[ignore]
fn merge_io_sorted_merges() {
    setup();
    for interleave in [false, true] {
        let (_tmp, dir) = bench_dir("sorted");
        let (segments, order) = sorted_segments(&dir, 4, docs(), interleave);
        let what = if interleave { "interleaved" } else { "non-interleaved" };
        let merged = measure(&format!("sorted merge {what} (4 x {} docs)", docs()), || rustic_core::merge::merge_segments_with(&dir, &segments, 100, Some(order.clone())).unwrap());
        println!("BENCH sorted merge {what}: output {:.1} MB, {} docs", file_mb(&dir, 100), merged.num_docs);
    }
}

#[test]
#[ignore]
fn merge_io_unsorted_many_fast_fields() {
    setup();
    let (_tmp, dir) = bench_dir("fast");
    let mut schema: Vec<Field> = (0..100u16).map(|f| Field::new(f, &format!("m{f}"), FieldType::I64)).collect();
    schema.extend((100..120u16).map(|f| Field::new(f, &format!("k{f}"), FieldType::Keyword)));
    let per = docs() / 2;
    let segments: Vec<Arc<Segment>> = (0..4u32).map(|s| {
        let mut w = SegmentWriter::new(s);
        for i in 0..per {
            let h = ((s * per + i) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut d = Document::new();
            for f in 0..100u16 { d.add_i64(f, ((h >> (f % 48)) % (1000 + f as u64 * 977)) as i64); }
            for f in 100..120u16 { d.add_keyword(f, &format!("v{}", (h >> (f % 40)) % (10 + f as u64 * 13))); }
            w.add_document(&d, &schema);
        }
        Arc::new(w.write_to(&dir).unwrap())
    }).collect();
    let merged = measure(&format!("unsorted merge, 120 columns (4 x {per} docs)"), || rustic_core::merge::merge_segments(&dir, &segments, 100).unwrap());
    println!("BENCH unsorted merge: output {:.1} MB, {} docs", file_mb(&dir, 100), merged.num_docs);
}
