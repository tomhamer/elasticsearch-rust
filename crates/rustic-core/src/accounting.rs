//! Process-wide accounting of long-lived memory (what Elasticsearch's
//! `accounting` and `fielddata` circuit breakers report).
//!
//! Each owner of such memory holds a [`Charge`] for the bytes it keeps; the
//! charge is released when the owner is dropped, so a figure falls exactly
//! when the memory is freed (a memtable kept by a PIT counts until the PIT's
//! view, its last reader, drops it). Charging is two relaxed atomic adds and
//! never takes a lock.

use std::sync::atomic::{AtomicU64, Ordering};

/// What a [`Charge`] accounts for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Category {
    /// Closed memtables (parked, frozen, published), until the last view or
    /// seal job holding them drops them. Memtables still being written are
    /// the indexing buffer's.
    Memtables = 0,
    /// Frozen deletion copies of reader views (published, PIT, scroll).
    ReaderViews = 1,
    /// Point tables copied by retained vector captures (PIT, scroll, ES|QL).
    RetainedCaptures = 2,
    /// Mutable HNSW graph arrays (levels, links), retired generations
    /// included until they are reclaimed.
    VectorGraphs = 3,
    /// Mutable vector source rows (f32 and quantized codes).
    VectorRows = 4,
    /// Uninverted text fielddata.
    Fielddata = 5,
    /// Live vector point tables (owners, locations, liveness, payload rows).
    VectorPoints = 6,
    /// Pooled memtables outside the index: idle, frozen, or filled by a
    /// bulk, until checked in (then the indexing buffer's).
    PooledMemtables = 7,
    /// Read caches a memtable builds after the fact, closed or not (ES
    /// quantized codes of its vector rows, encoded postings of long terms).
    ReadCaches = 8,
}

impl Category {
    pub const ALL: [Category; 9] = [
        Category::Memtables,
        Category::ReaderViews,
        Category::RetainedCaptures,
        Category::VectorGraphs,
        Category::VectorRows,
        Category::Fielddata,
        Category::VectorPoints,
        Category::PooledMemtables,
        Category::ReadCaches,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Category::Memtables => "memtables",
            Category::ReaderViews => "reader_views",
            Category::RetainedCaptures => "retained_captures",
            Category::VectorGraphs => "vector_graphs",
            Category::VectorRows => "vector_rows",
            Category::Fielddata => "fielddata",
            Category::VectorPoints => "vector_points",
            Category::PooledMemtables => "pooled_memtables",
            Category::ReadCaches => "read_caches",
        }
    }
}

/// One counter per category, each on its own cache line: charges from
/// different owners (vector inserts, memtable closes, views) never share
/// one.
#[repr(align(64))]
struct Slot(AtomicU64);

static BYTES: [Slot; 9] = [const { Slot(AtomicU64::new(0)) }; 9];

/// Bytes currently charged to `category`.
/// Mutable code log bytes (`vector::hnsw` code logs): anonymous chunk
/// buffers, and mapped (page-cache backed) chunk files. Diagnostics only:
/// the anonymous buffers are admitted separately; the mapped ones are not
/// resident memory owned by the process.
pub static CODE_LOG_ANON_BYTES: AtomicU64 = AtomicU64::new(0);
pub static CODE_LOG_MAPPED_BYTES: AtomicU64 = AtomicU64::new(0);

/// One line of every category, the shared admission budget and the code
/// logs, for diagnostic memory reports.
pub fn report() -> String {
    let mut out: Vec<String> = Category::ALL.iter().map(|c| format!("{}={}", c.name(), bytes(*c))).collect();
    let budget = crate::admission::shared();
    out.push(format!("admission_used={} admission_limit={}", budget.used(), budget.limit()));
    out.push(format!("code_log_anon={} code_log_mapped={}", CODE_LOG_ANON_BYTES.load(Ordering::Relaxed), CODE_LOG_MAPPED_BYTES.load(Ordering::Relaxed)));
    out.join(" ")
}

pub fn bytes(category: Category) -> u64 {
    BYTES[category as usize].0.load(Ordering::Relaxed)
}

/// Charge `n` bytes to `category` without a [`Charge`] (an owner that
/// releases them itself with [`sub`] when freed).
pub fn add(category: Category, n: u64) {
    if n != 0 {
        BYTES[category as usize].0.fetch_add(n, Ordering::Relaxed);
    }
}

/// Release `n` bytes charged with [`add`]. Never wraps: a release of more
/// than is charged (a bug) stops at zero, and debug builds panic.
pub fn sub(category: Category, n: u64) {
    if n == 0 {
        return;
    }
    let before = BYTES[category as usize].0.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |b| Some(b.saturating_sub(n))).unwrap_or(0);
    debug_assert!(before >= n, "accounting [{}] released {n} bytes of {before}", category.name());
}

/// Long-lived bytes outside fielddata: every category but
/// [`Category::Fielddata`], plus the optional vector column copies.
pub fn accounting_bytes() -> u64 {
    Category::ALL.iter().filter(|c| **c != Category::Fielddata).map(|c| bytes(*c)).sum::<u64>() + crate::vector::column_cache_bytes() as u64
}

/// Retired mutable-graph bytes pinned by retained views (PIT, scroll,
/// ES|QL captures): part of [`Category::VectorGraphs`], reported apart.
pub fn retained_graph_bytes() -> u64 {
    #[cfg(feature = "mutable-hnsw-poc")]
    {
        crate::vector::mutable_ingest::retained_graph_bytes() as u64
    }
    #[cfg(not(feature = "mutable-hnsw-poc"))]
    {
        0
    }
}

/// Bytes charged to one category, released on drop.
#[derive(Debug, Default)]
pub struct Charge {
    category: Option<Category>,
    bytes: u64,
}

impl Charge {
    pub fn new(category: Category, bytes: u64) -> Charge {
        add(category, bytes);
        Charge { category: Some(category), bytes }
    }

    /// Change the charged amount to `bytes`.
    pub fn resize(&mut self, bytes: u64) {
        let Some(c) = self.category else { return };
        if bytes > self.bytes {
            add(c, bytes - self.bytes);
        } else if bytes < self.bytes {
            sub(c, self.bytes - bytes);
        }
        self.bytes = bytes;
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        if let Some(c) = self.category {
            sub(c, self.bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charges_release_on_drop_and_resize() {
        // Other tests charge concurrently: compare deltas of a category no
        // other unit test of this module's crate builds in this test.
        let before = bytes(Category::ReaderViews);
        let mut a = Charge::new(Category::ReaderViews, 100);
        let b = Charge::new(Category::ReaderViews, 50);
        assert!(bytes(Category::ReaderViews) >= before + 150);
        a.resize(10);
        assert_eq!(a.bytes(), 10);
        drop(b);
        drop(a);
        let empty = Charge::default();
        drop(empty);
        assert!(bytes(Category::ReaderViews) <= before + 150);
    }
}
