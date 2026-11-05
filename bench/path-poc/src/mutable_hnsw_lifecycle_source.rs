//! Safe source-ownership adapter for the standalone lifecycle experiment.
//! A query captures the graph epoch/publication first, then pins this root.
//! Direct lookups borrow through immutable Arc-owned pages; no unsafe code.
use rustic_core::vector::hnsw::{
    MutableHnsw, MutableHnswSnapshot, MutableVectorSource, NodeHandle, ReclaimedNode,
};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering::Relaxed},
    Arc, Mutex, RwLock,
};

pub const PAGE_ROWS: usize = 128;

#[derive(Default)]
struct Counters {
    rows: AtomicUsize,
    row_bytes: AtomicUsize,
    peak_rows: AtomicUsize,
    roots: AtomicUsize,
    pages: AtomicUsize,
    root_capacity_bytes: AtomicUsize,
    root_clones: AtomicUsize,
    root_copy_bytes: AtomicUsize,
    page_clones: AtomicUsize,
    page_copy_bytes: AtomicUsize,
    relocation_owners: AtomicUsize,
}
pub struct Row {
    values: Box<[f32]>,
    counters: Arc<Counters>,
}
impl Drop for Row {
    fn drop(&mut self) {
        self.counters.rows.fetch_sub(1, Relaxed);
        self.counters
            .row_bytes
            .fetch_sub(self.values.len() * 4, Relaxed);
    }
}
struct RelocatedRow {
    row: Arc<Row>,
    storage_identity: u64,
    counters: Arc<Counters>,
}
impl Drop for RelocatedRow {
    fn drop(&mut self) {
        self.counters.relocation_owners.fetch_sub(1, Relaxed);
    }
}
#[derive(Clone)]
enum RowRef {
    Incoming(Arc<Row>),
    Relocated(Arc<RelocatedRow>),
}
impl RowRef {
    fn row(&self) -> &Arc<Row> {
        match self {
            Self::Incoming(row) => row,
            Self::Relocated(owner) => &owner.row,
        }
    }
}
#[derive(Clone)]
struct Entry {
    handle: NodeHandle,
    key: u64,
    row: RowRef,
}
struct Page {
    entries: [Option<Entry>; PAGE_ROWS],
    counters: Arc<Counters>,
}
impl Page {
    fn new(counters: &Arc<Counters>) -> Self {
        counters.pages.fetch_add(1, Relaxed);
        Self {
            entries: std::array::from_fn(|_| None),
            counters: counters.clone(),
        }
    }
}
impl Clone for Page {
    fn clone(&self) -> Self {
        self.counters.pages.fetch_add(1, Relaxed);
        self.counters.page_clones.fetch_add(1, Relaxed);
        self.counters
            .page_copy_bytes
            .fetch_add(std::mem::size_of::<Self>(), Relaxed);
        Self {
            entries: self.entries.clone(),
            counters: self.counters.clone(),
        }
    }
}
impl Drop for Page {
    fn drop(&mut self) {
        self.counters.pages.fetch_sub(1, Relaxed);
    }
}
struct Root {
    pages: Vec<Arc<Page>>,
    counters: Arc<Counters>,
}
impl Root {
    fn new(counters: &Arc<Counters>) -> Self {
        counters.roots.fetch_add(1, Relaxed);
        Self {
            pages: Vec::new(),
            counters: counters.clone(),
        }
    }
    fn ensure_page(&mut self, index: usize) {
        let before = self.pages.capacity() * std::mem::size_of::<Arc<Page>>();
        while self.pages.len() <= index {
            self.pages.push(Arc::new(Page::new(&self.counters)));
        }
        let after = self.pages.capacity() * std::mem::size_of::<Arc<Page>>();
        self.counters
            .root_capacity_bytes
            .fetch_add(after - before, Relaxed);
    }
    fn entry(&self, slot: u32) -> Option<&Entry> {
        self.pages.get(slot as usize / PAGE_ROWS)?.entries[slot as usize % PAGE_ROWS].as_ref()
    }
}
impl Clone for Root {
    fn clone(&self) -> Self {
        let pages = self.pages.clone();
        self.counters.roots.fetch_add(1, Relaxed);
        self.counters.root_clones.fetch_add(1, Relaxed);
        self.counters
            .root_copy_bytes
            .fetch_add(pages.len() * std::mem::size_of::<Arc<Page>>(), Relaxed);
        self.counters
            .root_capacity_bytes
            .fetch_add(pages.capacity() * std::mem::size_of::<Arc<Page>>(), Relaxed);
        Self {
            pages,
            counters: self.counters.clone(),
        }
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        self.counters.roots.fetch_sub(1, Relaxed);
        self.counters.root_capacity_bytes.fetch_sub(
            self.pages.capacity() * std::mem::size_of::<Arc<Page>>(),
            Relaxed,
        );
    }
}

pub struct SourceView {
    root: Arc<Root>,
}
impl MutableVectorSource for SourceView {
    // Compatibility/diagnostic path only. Production-like navigation always
    // supplies the reusable handle to the direct get_node implementation.
    fn get(&self, key: u64) -> Option<&[f32]> {
        self.root
            .pages
            .iter()
            .flat_map(|p| p.entries.iter())
            .flatten()
            .find(|entry| entry.key == key)
            .map(|entry| entry.row.row().values.as_ref())
    }
    fn get_node(&self, handle: NodeHandle, key: u64) -> Option<&[f32]> {
        let entry = self.root.entry(handle.slot())?;
        if entry.handle != handle || entry.key != key {
            return None;
        }
        Some(&entry.row.row().values)
    }
}
impl SourceView {
    pub fn payload_address(&self, handle: NodeHandle, key: u64) -> Option<usize> {
        self.get_node(handle, key).map(|row| row.as_ptr() as usize)
    }
    pub fn rows(&self) -> impl Iterator<Item = (NodeHandle, u64, &[f32])> {
        self.root
            .pages
            .iter()
            .flat_map(|p| p.entries.iter())
            .flatten()
            .map(|entry| (entry.handle, entry.key, entry.row.row().values.as_ref()))
    }
    pub fn storage_identity(&self, handle: NodeHandle) -> Option<u64> {
        let entry = self.root.entry(handle.slot())?;
        if entry.handle != handle {
            return None;
        }
        match &entry.row {
            RowRef::Incoming(_) => None,
            RowRef::Relocated(owner) => Some(owner.storage_identity),
        }
    }
}

pub struct PagedSource {
    graph: Arc<MutableHnsw>,
    root: RwLock<Arc<Root>>,
    counters: Arc<Counters>,
    dims: usize,
    max_rows: usize,
    max_slots: usize,
    // An invariant violation must NOT return a slot to the allocator before
    // its descriptor is acknowledged. Quarantine tokens and fail the trial.
    quarantine: Mutex<Vec<ReclaimedNode>>,
}
impl PagedSource {
    pub fn new(graph: Arc<MutableHnsw>, dims: usize, max_rows: usize, max_slots: usize) -> Self {
        let counters = Arc::new(Counters::default());
        Self {
            graph,
            root: RwLock::new(Arc::new(Root::new(&counters))),
            counters,
            dims,
            max_rows,
            max_slots,
            quarantine: Mutex::new(Vec::new()),
        }
    }
    pub fn pin(&self) -> SourceView {
        SourceView {
            root: self.root.read().unwrap().clone(),
        }
    }
    pub fn read(&self) -> (MutableHnswSnapshot, SourceView) {
        let graph = self.graph.snapshot(); // Epoch/publication first; source second.
        let source = self.pin();
        (graph, source)
    }
    pub fn allocate(&self, values: Vec<f32>) -> Result<Arc<Row>, String> {
        if values.len() != self.dims || values.iter().any(|v| !v.is_finite()) {
            return Err("invalid row dimension/value".into());
        }
        let count = self
            .counters
            .rows
            .fetch_update(Relaxed, Relaxed, |n| (n < self.max_rows).then_some(n + 1))
            .map_err(|_| "source row budget exhausted (held readers or pending writes)")?
            + 1;
        self.counters.peak_rows.fetch_max(count, Relaxed);
        self.counters.row_bytes.fetch_add(values.len() * 4, Relaxed);
        Ok(Arc::new(Row {
            values: values.into_boxed_slice(),
            counters: self.counters.clone(),
        }))
    }
    /// Publish an entire batch before any of its graph proposals can commit.
    pub fn install(&self, rows: Vec<(NodeHandle, u64, Arc<Row>)>) -> Result<(), String> {
        let mut current = self.root.write().unwrap();
        for (handle, _, row) in &rows {
            if handle.slot() as usize >= self.max_slots {
                return Err("source slot capacity exceeded".into());
            }
            if !Arc::ptr_eq(&row.counters, &self.counters) {
                return Err("row owner belongs to another source directory".into());
            }
            if current.entry(handle.slot()).is_some() {
                return Err("source slot still owns an earlier row".into());
            }
        }
        let mut seen = std::collections::HashSet::new();
        if rows.iter().any(|r| !seen.insert(r.0.slot())) {
            return Err("duplicate source slot in batch".into());
        }
        let root = Arc::make_mut(&mut current);
        for (handle, key, row) in rows {
            let page = handle.slot() as usize / PAGE_ROWS;
            root.ensure_page(page);
            Arc::make_mut(&mut root.pages[page]).entries[handle.slot() as usize % PAGE_ROWS] =
                Some(Entry {
                    handle,
                    key,
                    row: RowRef::Incoming(row),
                });
        }
        Ok(())
    }
    /// Models changing storage ownership without copying canonical vector bytes.
    /// This is not a real VectorSegment codec, mmap relocation or durable flush.
    pub fn relocate(
        &self,
        handle: NodeHandle,
        key: u64,
        storage_identity: u64,
    ) -> Result<(), String> {
        let mut current = self.root.write().unwrap();
        let entry = current
            .entry(handle.slot())
            .ok_or("relocation source missing")?;
        if entry.handle != handle || entry.key != key {
            return Err("stale relocation handle/key".into());
        }
        let row = entry.row.row().clone();
        self.counters.relocation_owners.fetch_add(1, Relaxed);
        let owner = Arc::new(RelocatedRow {
            row,
            storage_identity,
            counters: self.counters.clone(),
        });
        let root = Arc::make_mut(&mut current);
        let page = Arc::make_mut(&mut root.pages[handle.slot() as usize / PAGE_ROWS]);
        page.entries[handle.slot() as usize % PAGE_ROWS]
            .as_mut()
            .unwrap()
            .row = RowRef::Relocated(owner);
        Ok(())
    }
    /// Erase current source ownership BEFORE dropping tokens permits slot reuse.
    /// Source-only old roots independently retain their immutable rows safely.
    pub fn acknowledge(&self, tokens: Vec<ReclaimedNode>) -> Result<usize, String> {
        if tokens.is_empty() {
            return Ok(0);
        }
        let count = tokens.len();
        let mut current = self.root.write().unwrap();
        if tokens.iter().any(|token| {
            !self.graph.owns_reclaimed(token)
                || current
                    .entry(token.handle().slot())
                    .is_some_and(|entry| entry.handle != token.handle() || entry.key != token.key())
        }) {
            self.quarantine.lock().unwrap().extend(tokens);
            return Err("reclamation token mismatches source descriptor; token quarantined".into());
        }
        let root = Arc::make_mut(&mut current);
        for token in &tokens {
            let page = token.handle().slot() as usize / PAGE_ROWS;
            if let Some(page) = root.pages.get_mut(page) {
                Arc::make_mut(page).entries[token.handle().slot() as usize % PAGE_ROWS] = None;
            }
        }
        drop(current); // Publish the cleared current root before allocator reuse.
        drop(tokens);
        Ok(count)
    }
    pub fn stats(&self) -> Value {
        let view = self.pin();
        let rows: Vec<_> = view.rows().collect();
        let current_rows = rows.len();
        let resident_rows = self.counters.rows.load(Relaxed);
        let current_pages = view.root.pages.len();
        let resident_pages = self.counters.pages.load(Relaxed);
        json!({"current_rows":current_rows,"resident_rows":resident_rows,"retained_or_pending_rows":resident_rows.saturating_sub(current_rows),"current_payload_bytes":current_rows*self.dims*4,"resident_payload_bytes":self.counters.row_bytes.load(Relaxed),"peak_resident_rows":self.counters.peak_rows.load(Relaxed),"row_budget":self.max_rows,"slot_budget":self.max_slots,"directory_slot_capacity":current_pages*PAGE_ROWS,"current_pages":current_pages,"resident_pages":resident_pages,"retained_pages":resident_pages.saturating_sub(current_pages),"resident_page_bytes":resident_pages*std::mem::size_of::<Page>(),"resident_roots":self.counters.roots.load(Relaxed),"resident_root_capacity_bytes":self.counters.root_capacity_bytes.load(Relaxed),"root_clones":self.counters.root_clones.load(Relaxed),"root_pointer_copy_bytes":self.counters.root_copy_bytes.load(Relaxed),"page_clones":self.counters.page_clones.load(Relaxed),"page_descriptor_copy_bytes":self.counters.page_copy_bytes.load(Relaxed),"relocation_owners":self.counters.relocation_owners.load(Relaxed),"quarantined_tokens":self.quarantine.lock().unwrap().len(),"memory_scope":"distinct resident row payload admission only; retained root/page/relocation metadata has no admission bound; quiescent drain is checked, arbitrary concurrent reader memory is not bounded", "copy_complexity":"outer root clones O(page count); modified pages clone O(PAGE_ROWS); standalone ownership proof, not production directory approval","accounting":"owner/drop counters and logical allocation sizes exclude allocator overhead and RSS page residency"})
    }
}
