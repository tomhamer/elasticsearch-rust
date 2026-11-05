//! Query execution counters for benchmarks and diagnosis
//! (`wiki_search_bench` `WB_SLOW`): a handful of relaxed atomic adds per
//! leaf, never per document. Compiled in only with the `query-stats`
//! feature (off by default): without it every function here is an empty
//! inline no-op and [`Start`] is a zero-sized type, so production builds
//! and authoritative timings carry no counters and no extra clock reads.
//! Meaningful per query only when one query runs at a time; [`reset`]
//! before it, [`snapshot`] after.

macro_rules! counters {
    ($($name:ident: $doc:literal,)*) => {
        /// Counter ids.
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug)]
        pub enum C { $($name,)* }
        const NAMES: &[&str] = &[$(stringify!($name),)*];
        const DOCS: &[&str] = &[$($doc,)*];
    };
}

counters! {
    context_ns: "term statistics pass (dictionary lookups, before any leaf runs)",
    term_lookups: "term-by-leaf dictionary lookups of the statistics pass",
    search_ns: "leaves stage wall time (Searcher::search_with_context)",
    leaves: "leaves run",
    leaf_ns: "sum of per-leaf run times",
    leaf_max_ns: "slowest leaf",
    leaf_bulk: "leaves collected by a bulk (block-at-a-time) path",
    leaf_per_doc: "leaves collected doc by doc (no bulk path applied)",
    leaves_skipped: "leaves not visited: earlier ones already decided the hits and total",
    or_calls: "block OR (or_groups) leaf runs",
    or_ns: "time in block OR",
    or_terms: "term clauses over all block OR runs",
    or_groups: "dis_max groups over all block OR runs",
    or_postings: "postings (sum of doc freqs) of those clauses",
    or_scored: "candidate docs fully scored",
    or_counted: "matches counted",
    or_count_windows: "4096-doc counting windows (every clause walked)",
    or_partitions: "pruning partitions (outer windows / threshold renewals)",
    or_essential: "essential groups summed over partitions",
    or_clause_partitions: "partitions split at the clause level",
    or_essential_clauses: "walked clauses summed over partitions",
    or_blocks_skipped: "walked clauses' postings blocks skipped by their block bound",
    or_required: "partitions with a required lead group (conjunction mode)",
    or_skipped_windows: "outer windows skipped whole (nothing could compete)",
    or_inner_windows: "pruning inner windows walked",
    or_blocks: "postings blocks decoded by walks",
    or_probes: "non-walked clause probes (seek) on candidates",
    fetch_ns: "fetch phase (stored fields, highlight, hit JSON)",
    store_docs: "stored documents read (Segment::get_doc)",
    store_decompress_ns: "stored-field chunk decompression",
    store_decompressed_bytes: "bytes decompressed",
    store_decode_ns: "stored document decoding (bincode)",
    store_cache_hits: "reads served by the decompressed-chunk cache",
    fetch_hits: "hits fetched",
}

/// Whether the counters are compiled in.
pub const ENABLED: bool = cfg!(feature = "query-stats");

#[cfg(feature = "query-stats")]
static VALUES: [AtomicU64; NAMES.len()] = [const { AtomicU64::new(0) }; NAMES.len()];

/// A timing start: an `Instant` with the feature, nothing without it.
#[cfg(feature = "query-stats")]
pub type Start = std::time::Instant;
/// A timing start: an `Instant` with the feature, nothing without it.
#[cfg(not(feature = "query-stats"))]
pub type Start = ();

/// Start timing (no clock read without the feature).
#[inline(always)]
pub fn start() -> Start {
    #[cfg(feature = "query-stats")]
    return std::time::Instant::now();
}

/// Add the nanoseconds since `s` to counter `c`.
#[inline(always)]
pub fn add_since(c: C, s: Start) {
    #[cfg(feature = "query-stats")]
    add(c, s.elapsed().as_nanos() as u64);
    #[cfg(not(feature = "query-stats"))]
    let _ = (c, s);
}

/// Raise counter `c` to at least the nanoseconds since `s`.
#[inline(always)]
pub fn max_since(c: C, s: Start) {
    #[cfg(feature = "query-stats")]
    max(c, s.elapsed().as_nanos() as u64);
    #[cfg(not(feature = "query-stats"))]
    let _ = (c, s);
}

/// Add `v` to counter `c`.
#[inline(always)]
pub fn add(c: C, v: u64) {
    #[cfg(feature = "query-stats")]
    VALUES[c as usize].fetch_add(v, Relaxed);
    #[cfg(not(feature = "query-stats"))]
    let _ = (c, v);
}

/// Raise counter `c` to at least `v`.
#[inline(always)]
pub fn max(c: C, v: u64) {
    #[cfg(feature = "query-stats")]
    VALUES[c as usize].fetch_max(v, Relaxed);
    #[cfg(not(feature = "query-stats"))]
    let _ = (c, v);
}

/// Zero every counter.
pub fn reset() {
    #[cfg(feature = "query-stats")]
    for v in &VALUES {
        v.store(0, Relaxed);
    }
}

/// `(name, value, description)` of every counter (all zero without the
/// feature).
pub fn snapshot() -> Vec<(&'static str, u64, &'static str)> {
    #[cfg(feature = "query-stats")]
    let value = |i: usize| VALUES[i].load(Relaxed);
    #[cfg(not(feature = "query-stats"))]
    let value = |_: usize| 0u64;
    NAMES.iter().zip(DOCS).enumerate().map(|(i, (n, d))| (*n, value(i), *d)).collect()
}
