//! Merge memory admission against what a merge really allocates: the peak
//! of the merge's admitted reservations (its `admission::Owner`) next to the
//! peak of heap bytes allocated meanwhile (a counting global allocator) and,
//! on Linux, the resident set growth.
//!
//! `big_df_term_merge_reserves_about_twice_its_allocation` runs by default.
//! The wiki-like measurement is ignored; run it with
//! `cargo test --release -p rustic-core --test merge_memory -- --ignored --nocapture`
//! (`MERGE_DOCS` documents, default 300000, in `MERGE_SEGMENTS` segments).

use rustic_core::admission::Owner;
use rustic_core::codec::directory::MmapDirectory;
use rustic_core::segment::{Segment, SegmentWriter};
use rustic_core::types::{Document, Field, FieldType, FieldValue};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

struct Counting;
static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(bytes: usize) {
    let now = CURRENT.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) };
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, layout, new) };
        if !q.is_null() {
            if new >= layout.size() {
                grew(new - layout.size());
            } else {
                CURRENT.fetch_sub(layout.size() - new, Ordering::Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Measurements share the global counters: one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn schema() -> Vec<Field> {
    let mut fields = vec![
        Field::new(0, "title", FieldType::Text),
        Field::new(1, "body", FieldType::Text),
        Field::new(2, "page_id", FieldType::I64),
        Field::new(3, "namespace", FieldType::Keyword),
    ];
    for f in fields.iter_mut() {
        f.fast = !matches!(f.field_type, FieldType::Text);
    }
    fields
}

/// xorshift64*: deterministic documents without a dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Zipf(1) ranks over `vocab` words, like natural-language text: rank 0
/// ("the") is in nearly every article, many times.
struct Zipf(Vec<f64>);
impl Zipf {
    fn new(vocab: usize) -> Self {
        let mut acc = 0.0;
        let mut cdf: Vec<f64> = (1..=vocab).map(|r| { acc += 1.0 / r as f64; acc }).collect();
        for c in cdf.iter_mut() {
            *c /= acc;
        }
        Self(cdf)
    }
    fn word(&self, rng: &mut Rng) -> usize {
        let u = rng.unit();
        self.0.partition_point(|&c| c < u).min(self.0.len() - 1)
    }
}

fn word(rank: usize) -> String {
    if rank == 0 { "the".into() } else { format!("w{rank}") }
}

fn wiki_segments(dir: &MmapDirectory, docs: usize, segments: usize, body_words: usize) -> Vec<Arc<Segment>> {
    let schema = schema();
    let zipf = Zipf::new(200_000);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let per = docs.div_ceil(segments);
    (0..segments)
        .map(|s| {
            let mut writer = SegmentWriter::new(s as u32);
            for i in 0..per.min(docs - s * per) {
                let id = s * per + i;
                let mut doc = Document::new();
                let title: Vec<String> = (0..4).map(|_| word(zipf.word(&mut rng))).collect();
                let words = body_words / 2 + (rng.next() % body_words as u64) as usize;
                let body: Vec<String> = (0..words).map(|_| word(zipf.word(&mut rng))).collect();
                doc.add_text(0, &title.join(" "));
                doc.add_text(1, &body.join(" "));
                doc.fields.push((2, FieldValue::I64(id as i64)));
                doc.add_keyword(3, &format!("ns{}", id % 8));
                writer.add_document(&doc, &schema);
            }
            Arc::new(writer.write_to(dir).unwrap())
        })
        .collect()
}

/// One text field whose single term is in every document three times,
/// beside a unique term per document: the big-df term dominates the merge.
fn big_df_segments(dir: &MmapDirectory, docs: usize, segments: usize) -> Vec<Arc<Segment>> {
    let schema = vec![Field::new(0, "body", FieldType::Text)];
    let per = docs / segments;
    (0..segments)
        .map(|s| {
            let mut writer = SegmentWriter::new(s as u32);
            for i in 0..per {
                let mut doc = Document::new();
                doc.add_text(0, &format!("the the the u{}", s * per + i));
                writer.add_document(&doc, &schema);
            }
            Arc::new(writer.write_to(dir).unwrap())
        })
        .collect()
}

fn resident_bytes() -> Option<usize> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: usize = statm.split_whitespace().nth(1)?.parse().ok()?;
    Some(pages * 4096)
}

struct Measured {
    reserved_peak: u64,
    allocated_peak: u64,
    resident_growth: Option<u64>,
    seconds: f64,
}

fn measure_merge(dir: &MmapDirectory, segments: &[Arc<Segment>], new_id: u32) -> Measured {
    // Room for any merge: the measurement compares, it must not refuse.
    rustic_core::admission::configure_shared(1 << 40);
    let owner = Owner::new();
    let stop = AtomicBool::new(false);
    let rss_base = resident_bytes();
    let rss_peak = AtomicUsize::new(rss_base.unwrap_or(0));
    let base = CURRENT.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let started = std::time::Instant::now();
    std::thread::scope(|s| {
        s.spawn(|| {
            while !stop.load(Ordering::Relaxed) {
                if let Some(r) = resident_bytes() {
                    rss_peak.fetch_max(r, Ordering::Relaxed);
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        });
        let merged = {
            let _scope = owner.enter();
            rustic_core::merge::merge_segments(dir, segments, new_id).unwrap()
        };
        assert!(merged.num_docs > 0);
        stop.store(true, Ordering::Relaxed);
    });
    Measured {
        reserved_peak: owner.bytes().peak,
        allocated_peak: (PEAK.load(Ordering::Relaxed) - base) as u64,
        resident_growth: rss_base.map(|b| rss_peak.load(Ordering::Relaxed).saturating_sub(b) as u64),
        seconds: started.elapsed().as_secs_f64(),
    }
}

fn report(label: &str, m: &Measured) {
    let mib = |b: u64| b as f64 / (1 << 20) as f64;
    println!(
        "{label}: reserved peak {:.1} MiB, allocated peak {:.1} MiB (ratio {:.2}), resident growth {}, {:.2} s",
        mib(m.reserved_peak),
        mib(m.allocated_peak),
        m.reserved_peak as f64 / m.allocated_peak.max(1) as f64,
        m.resident_growth.map_or("n/a".into(), |r| format!("{:.1} MiB", mib(r))),
        m.seconds
    );
}

#[test]
fn big_df_term_merge_reserves_about_twice_its_allocation() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::open(tmp.path()).unwrap();
    let docs: usize = std::env::var("MERGE_BIG_DF_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(200_000);
    let segments = big_df_segments(&dir, docs, 2);
    let m = measure_merge(&dir, &segments, 100);
    report(&format!("big-df term ({docs} docs, tf 3)"), &m);
    // Measured 1.6x (debug) to 1.95x (release): beside the big term, the
    // per-term dictionary and list-header bounds of the 200k unique terms.
    // The old charges reserved 5.7x here. Allocation peaks vary a little
    // with rayon scheduling, hence the margin over 2x.
    assert!(
        m.reserved_peak <= 5 * m.allocated_peak / 2 + (1 << 20),
        "a big-df term merge reserved {} bytes for {} allocated",
        m.reserved_peak,
        m.allocated_peak
    );
}

#[test]
#[ignore]
fn wiki_like_merge_memory() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let docs: usize = std::env::var("MERGE_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(300_000);
    let nseg: usize = std::env::var("MERGE_SEGMENTS").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
    let words: usize = std::env::var("MERGE_BODY_WORDS").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
    let tmp = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::open(tmp.path()).unwrap();
    let built = std::time::Instant::now();
    let segments = wiki_segments(&dir, docs, nseg, words);
    println!("built {nseg} segments of {docs} wiki-like docs ({words} body words avg) in {:.1} s", built.elapsed().as_secs_f64());
    let m = measure_merge(&dir, &segments, 1000);
    report(&format!("wiki-like merge ({docs} docs, {nseg} segments)"), &m);
}
