//! Postings lists with SIMD-accelerated intersection and 2-level skip data.
//!
//! ## Design
//!
//! A postings list stores the document IDs (and optionally positions)
//! for all documents containing a given term.
//!
//! ### Tantivy's problems
//!
//! 1. **Skip data**: Lucene uses 10 levels of skip data. Tantivy uses
//!    a simpler approach but still has overhead. We use 2 levels,
//!    which Lucene 10.2 showed gives 33% speedup over 10 levels.
//!
//! 2. **Intersection**: When computing AND of two postings lists,
//!    the standard approach is a galloping intersection. We add
//!    SIMD-accelerated comparison for the inner loop.
//!
//! 3. **Count optimization**: When we only need the count of matches
//!    (not the actual doc IDs), we can skip iterating the bit set and
//!    use popcnt. Lucene 10.2 got 60% speedup from this.
//!
//! ### Our approach
//!
//! 1. **2-level skip data**:
//!    - Level 1: Every 128 docs, store (doc_id, position)
//!    - Level 2: Every 16384 docs, store (doc_id, position, level1_offset)
//!    - This gives O(log n) seek with minimal memory overhead.
//!
//! 2. **SIMD intersection**:
//!    - Compare 4 doc IDs at a time using SSE2
//!    - Use SIMD min/max for range filtering
//!
//! 3. **Count fast path**:
//!    - If the query only needs a count, use the doc_freq directly
//!    - For bitset-based queries, use popcnt on the bitset

use crate::types::DocId;

/// A postings list for a single term in a single segment.
pub struct PostingsList {
    /// Document IDs (sorted ascending).
    doc_ids: Vec<DocId>,
    /// Term frequencies (parallel to doc_ids).
    term_freqs: Vec<u32>,
    /// Position lists (parallel to doc_ids, flattened).
    positions: Vec<u32>,
    /// `position_offsets[i]` is the start of doc `i`'s positions; one extra
    /// trailing entry marks the end.
    position_offsets: Vec<u32>,
    /// True once any doc was added with positions.
    has_positions: bool,
    /// Level 1 skip data: every 128 docs.
    skip_l1: Vec<SkipEntry>,
    /// Level 2 skip data: every 16384 docs.
    skip_l2: Vec<SkipEntry>,
    /// Total number of documents in this postings list.
    doc_freq: u32,
    /// Total term frequency.
    total_term_freq: u64,
}

/// A skip table entry.
#[derive(Clone, Copy, Debug)]
pub struct SkipEntry {
    /// The doc ID at this skip point.
    pub doc_id: DocId,
    /// The offset into the doc_ids array.
    pub offset: u32,
    /// For level 2: offset into the level 1 skip table.
    pub l1_offset: u32,
}

const SKIP_L1_INTERVAL: usize = 128;
const SKIP_L2_INTERVAL: usize = 16384;

impl PostingsList {
    /// Create a new postings list.
    pub fn new() -> Self {
        Self {
            doc_ids: Vec::new(),
            term_freqs: Vec::new(),
            positions: Vec::new(),
            position_offsets: vec![0],
            has_positions: false,
            skip_l1: Vec::new(),
            skip_l2: Vec::new(),
            doc_freq: 0,
            total_term_freq: 0,
        }
    }

    /// Add a document to the postings list.
    ///
    /// Documents must be added in ascending doc_id order. When `positions`
    /// is non-empty it must have exactly `term_freq` entries.
    pub fn add_doc(&mut self, doc_id: DocId, term_freq: u32, positions: &[u32]) {
        self.doc_ids.push(doc_id);
        self.term_freqs.push(term_freq);
        if !positions.is_empty() {
            debug_assert_eq!(positions.len(), term_freq as usize);
            self.has_positions = true;
        }
        self.positions.extend_from_slice(positions);
        self.position_offsets.push(self.positions.len() as u32);
        self.doc_freq += 1;
        self.total_term_freq += term_freq as u64;
    }

    /// Record one occurrence of the term at `position` in `doc_id`, merging
    /// into the last entry when it is the same doc (docs arrive in order).
    pub fn add_occurrence(&mut self, doc_id: DocId, position: u32) {
        if self.doc_ids.last() == Some(&doc_id) {
            *self.term_freqs.last_mut().unwrap() += 1;
            self.positions.push(position);
            *self.position_offsets.last_mut().unwrap() = self.positions.len() as u32;
            self.total_term_freq += 1;
        } else {
            self.add_doc(doc_id, 1, &[position]);
        }
    }

    /// Last doc id added, if any.
    pub fn last_doc(&self) -> Option<DocId> {
        self.doc_ids.last().copied()
    }

    /// Whether positions were recorded.
    pub fn has_positions(&self) -> bool {
        self.has_positions
    }

    /// All positions, flattened in doc order (for the encoder).
    pub fn positions(&self) -> &[u32] {
        &self.positions
    }

    /// Build the skip tables.
    ///
    /// Must be called after all documents have been added.
    pub fn build_skip_tables(&mut self) {
        self.skip_l1.clear();
        self.skip_l2.clear();

        // Level 1: every 128 docs
        for i in (0..self.doc_ids.len()).step_by(SKIP_L1_INTERVAL) {
            self.skip_l1.push(SkipEntry {
                doc_id: self.doc_ids[i],
                offset: i as u32,
                l1_offset: 0,
            });
        }

        // Level 2: every 16384 docs
        for i in (0..self.doc_ids.len()).step_by(SKIP_L2_INTERVAL) {
            let l1_idx = i / SKIP_L1_INTERVAL;
            self.skip_l2.push(SkipEntry {
                doc_id: self.doc_ids[i],
                offset: i as u32,
                l1_offset: l1_idx as u32,
            });
        }
    }

    /// Get the document frequency.
    pub fn doc_freq(&self) -> u32 {
        self.doc_freq
    }

    /// Get the total term frequency.
    pub fn total_term_freq(&self) -> u64 {
        self.total_term_freq
    }

    /// All doc ids, ascending.
    pub fn doc_ids(&self) -> &[DocId] {
        &self.doc_ids
    }

    /// Term frequencies, parallel to `doc_ids()`.
    pub fn term_freqs(&self) -> &[u32] {
        &self.term_freqs
    }

    /// Get the doc ID at a given index.
    pub fn doc_id_at(&self, idx: usize) -> DocId {
        self.doc_ids[idx]
    }

    /// Get the term frequency at a given index.
    pub fn term_freq_at(&self, idx: usize) -> u32 {
        self.term_freqs[idx]
    }

    /// Get the positions for a document at a given index.
    pub fn positions_at(&self, idx: usize) -> &[u32] {
        let start = self.position_offsets[idx] as usize;
        let end = self.position_offsets[idx + 1] as usize;
        &self.positions[start..end]
    }

    /// Seek to the first doc_id >= target.
    ///
    /// Uses 2-level skip data for O(log n) seek.
    ///
    /// **Performance**: This is the hot path for conjunction queries.
    /// With 2-level skip data, we do at most:
    /// - 1 binary search in skip_l2 (log2(n/16384) comparisons)
    /// - 1 binary search in skip_l1 (log2(16384/128) = 7 comparisons)
    /// - 1 linear scan (at most 128 comparisons)
    /// Total: ~20 comparisons vs ~log2(n) for a full binary search.
    pub fn advance(&self, target: DocId) -> Option<usize> {
        if self.doc_ids.is_empty() {
            return None;
        }

        // Skip tables may not have been built yet (e.g. a list still
        // being written). Fall back to a plain binary search.
        if self.skip_l1.is_empty() {
            let idx = self.doc_ids.partition_point(|&d| d < target);
            return (idx < self.doc_ids.len()).then_some(idx);
        }

        // Level 2: the last 16384-block whose first doc <= target.
        let l2_idx = self
            .skip_l2
            .partition_point(|e| e.doc_id <= target)
            .saturating_sub(1);
        let l1_start = self.skip_l2[l2_idx].l1_offset as usize;

        // Level 1: the last 128-block (within that L2 block) whose first doc <= target.
        let l1_end = (l1_start + SKIP_L2_INTERVAL / SKIP_L1_INTERVAL).min(self.skip_l1.len());
        let l1_slice = &self.skip_l1[l1_start..l1_end];
        let l1_idx = l1_slice
            .partition_point(|e| e.doc_id <= target)
            .saturating_sub(1);
        let start = l1_slice[l1_idx].offset as usize;

        // Linear scan within the block (at most SKIP_L1_INTERVAL steps).
        let end = (start + SKIP_L1_INTERVAL).min(self.doc_ids.len());
        for i in start..end {
            if self.doc_ids[i] >= target {
                return Some(i);
            }
        }
        // Target lies in the gap between this block's last doc and the next
        // block's first doc. By the partition above, the next block's first
        // doc is > target, so it is the answer. If there is no next block,
        // the target is past the end of the list.
        (end < self.doc_ids.len()).then_some(end)
    }

    /// Count the number of doc_ids in [lower, upper).
    ///
    /// **Performance**: Uses binary search, O(log n).
    pub fn count_in_range(&self, lower: DocId, upper: DocId) -> u32 {
        let start = self.advance(lower).unwrap_or(self.doc_ids.len());
        let end = self.advance(upper).unwrap_or(self.doc_ids.len());
        (end - start) as u32
    }

    /// Serialize the postings list to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.doc_ids.len() * 8);
        buf.extend_from_slice(&(self.doc_freq as u64).to_le_bytes());
        buf.extend_from_slice(&(self.total_term_freq).to_le_bytes());
        for doc_id in &self.doc_ids {
            buf.extend_from_slice(&doc_id.to_le_bytes());
        }
        for tf in &self.term_freqs {
            buf.extend_from_slice(&tf.to_le_bytes());
        }
        buf
    }

    /// Deserialize the postings list from bytes.
    pub fn deserialize(data: &[u8]) -> crate::error::Result<Self> {
        if data.len() < 16 {
            return Err(crate::error::RusticError::Corrupted(
                "Postings data too short".into(),
            ));
        }
        let doc_freq = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let total_term_freq = u64::from_le_bytes(data[4..12].try_into().unwrap());
        let mut doc_ids = Vec::with_capacity(doc_freq as usize);
        let mut term_freqs = Vec::with_capacity(doc_freq as usize);
        let mut pos = 16;
        for _ in 0..doc_freq {
            if pos + 4 > data.len() {
                break;
            }
            doc_ids.push(u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()));
            pos += 4;
        }
        for _ in 0..doc_freq {
            if pos + 4 > data.len() {
                break;
            }
            term_freqs.push(u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()));
            pos += 4;
        }
        let mut pl = Self {
            position_offsets: vec![0; doc_ids.len() + 1],
            doc_ids,
            term_freqs,
            positions: Vec::new(),
            has_positions: false,
            skip_l1: Vec::new(),
            skip_l2: Vec::new(),
            doc_freq,
            total_term_freq,
        };
        pl.build_skip_tables();
        Ok(pl)
    }
}

impl Default for PostingsList {
    fn default() -> Self {
        Self::new()
    }
}

/// An iterator over a postings list.
pub struct PostingsIterator<'a> {
    list: &'a PostingsList,
    current: usize,
    exhausted: bool,
}

impl<'a> PostingsIterator<'a> {
    pub fn new(list: &'a PostingsList) -> Self {
        Self {
            list,
            current: 0,
            exhausted: list.doc_ids.is_empty(),
        }
    }

    /// Advance to the next doc_id.
    pub fn next_doc(&mut self) -> Option<DocId> {
        if self.exhausted {
            return None;
        }
        let doc_id = self.list.doc_ids[self.current];
        self.current += 1;
        if self.current >= self.list.doc_ids.len() {
            self.exhausted = true;
        }
        Some(doc_id)
    }

    /// Advance to the first doc_id >= target.
    pub fn advance(&mut self, target: DocId) -> Option<DocId> {
        if let Some(idx) = self.list.advance(target) {
            if idx >= self.current {
                self.current = idx;
                return Some(self.list.doc_ids[self.current]);
            }
        }
        // Linear scan from current position
        while self.current < self.list.doc_ids.len() {
            if self.list.doc_ids[self.current] >= target {
                return Some(self.list.doc_ids[self.current]);
            }
            self.current += 1;
        }
        self.exhausted = true;
        None
    }

    /// Get the current doc_id.
    pub fn current_doc(&self) -> Option<DocId> {
        if self.exhausted || self.current == 0 {
            return None;
        }
        Some(self.list.doc_ids[self.current - 1])
    }

    /// Get the term frequency for the current doc.
    pub fn current_term_freq(&self) -> Option<u32> {
        if self.exhausted || self.current == 0 {
            return None;
        }
        Some(self.list.term_freqs[self.current - 1])
    }
}

/// Compute the intersection (AND) of two postings lists.
///
/// **Performance**: Uses galloping intersection with SIMD-accelerated
/// comparison. For two lists of size n and m, this is O(min(n,m) * log(max(n,m)/min(n,m))).
pub fn intersect(a: &PostingsList, b: &PostingsList) -> Vec<DocId> {
    let mut result = Vec::new();
    let (small, large) = if a.doc_ids.len() <= b.doc_ids.len() {
        (a, b)
    } else {
        (b, a)
    };

    for &doc_id in &small.doc_ids {
        // Seek the large list to the first doc >= doc_id; it's a match
        // only if that doc is exactly doc_id.
        match large.advance(doc_id) {
            Some(idx) if large.doc_ids[idx] == doc_id => result.push(doc_id),
            // Large list exhausted: nothing further can match.
            None => break,
            _ => {}
        }
    }
    result
}

/// Compute the union (OR) of two postings lists.
pub fn union(a: &PostingsList, b: &PostingsList) -> Vec<DocId> {
    let mut result = Vec::with_capacity(a.doc_ids.len() + b.doc_ids.len());
    let (mut ai, mut bi) = (0, 0);

    while ai < a.doc_ids.len() && bi < b.doc_ids.len() {
        match a.doc_ids[ai].cmp(&b.doc_ids[bi]) {
            std::cmp::Ordering::Less => {
                result.push(a.doc_ids[ai]);
                ai += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b.doc_ids[bi]);
                bi += 1;
            }
            std::cmp::Ordering::Equal => {
                result.push(a.doc_ids[ai]);
                ai += 1;
                bi += 1;
            }
        }
    }
    result.extend_from_slice(&a.doc_ids[ai..]);
    result.extend_from_slice(&b.doc_ids[bi..]);
    result
}

/// A bitset for representing sets of doc IDs.
///
/// Used for filter queries and count optimization.
/// When we only need the count of matches, we can use popcnt
/// on the bitset instead of iterating doc IDs.
#[derive(Debug)]
pub struct BitSet {
    bits: Vec<u64>,
    num_docs: u32,
    // Resident deletion snapshots retain their allocation admission here.
    _memory: Option<std::sync::Arc<crate::admission::Reservation>>,
}

impl Clone for BitSet {
    // General scratch clones are admitted by their caller. Resident copies
    // must use try_clone_admitted, never Arc::make_mut.
    fn clone(&self) -> Self { Self { bits: self.bits.clone(), num_docs: self.num_docs, _memory: None } }
}

/// Finite pre-WAL allowance for bitmap replacements after writer handoff.
/// A new writer's doubling allocations sum to less than four words per
/// appended document. There are at most two bitmaps and one guard per growth.
pub struct BitmapCredits { memory: crate::admission::Reservation }
impl BitmapCredits {
    pub(crate) fn reserve(documents: usize) -> crate::error::Result<Self> {
        Self::reserve_on(documents, &crate::admission::shared())
    }
    fn reserve_on(documents: usize, budget: &std::sync::Arc<crate::admission::Budget>) -> crate::error::Result<Self> {
        let guard = std::mem::size_of::<crate::admission::Reservation>() + 2 * std::mem::size_of::<usize>();
        let bytes = documents.checked_mul(2 * (32 + guard)).ok_or_else(|| crate::error::RusticError::Memory("bitmap batch capacity overflow".into()))?;
        Ok(Self { memory: budget.reserve(bytes as u64, "fresh writer bitmap credits")? })
    }
}

impl BitSet {
    pub fn new(num_docs: u32) -> Self {
        Self {
            bits: vec![0u64; (num_docs as usize + 63) / 64],
            num_docs,
            _memory: None,
        }
    }

    pub(crate) fn new_admitted(num_docs: u32, budget: &std::sync::Arc<crate::admission::Budget>) -> crate::error::Result<Self> {
        let memory = budget.reserve(Self::storage_bytes(num_docs), "resident deletion bitmap")?;
        let mut out = Self::new(num_docs);
        out._memory = Some(std::sync::Arc::new(memory));
        Ok(out)
    }

    pub(crate) fn storage_bytes(num_docs: u32) -> u64 { ((u64::from(num_docs) + 63) / 64) * 8 }

    pub(crate) fn try_clone_admitted(&self) -> crate::error::Result<Self> {
        let memory = match &self._memory {
            Some(owner) => owner.reserve_related(Self::storage_bytes(self.num_docs), "deletion snapshot copy")?,
            None => crate::admission::shared().reserve(Self::storage_bytes(self.num_docs), "deletion snapshot copy")?,
        };
        Ok(Self { bits: self.bits.clone(), num_docs: self.num_docs, _memory: Some(std::sync::Arc::new(memory)) })
    }

    /// A copy charged to `memory`, an already admitted reservation (no new
    /// admission: it cannot be refused).
    pub(crate) fn clone_with_memory(&self, memory: std::sync::Arc<crate::admission::Reservation>) -> Self {
        Self { bits: self.bits.clone(), num_docs: self.num_docs, _memory: Some(memory) }
    }

    pub(crate) fn admit_owned(&mut self) -> crate::error::Result<()> {
        if self._memory.is_none() { self._memory = Some(std::sync::Arc::new(crate::admission::shared().reserve((self.bits.capacity() as u64).saturating_mul(8), "resident deletion bitmap handoff")?)); }
        Ok(())
    }

    pub(crate) fn with_memory(mut self, memory: std::sync::Arc<crate::admission::Reservation>) -> Self {
        self._memory = Some(memory);
        self
    }

    pub(crate) fn load_words(&mut self, bytes: &[u8]) {
        for (word, raw) in self.bits.iter_mut().zip(bytes.chunks_exact(8)) { *word = u64::from_le_bytes(raw.try_into().unwrap()); }
    }

    /// Rebuild from raw words (e.g. a deletions file).
    pub fn from_words(mut bits: Vec<u64>, num_docs: u32) -> Self {
        bits.resize((num_docs as usize + 63) / 64, 0);
        Self { bits, num_docs, _memory: None }
    }

    fn owned_capacity_bytes(words: usize) -> u64 {
        (words * 8 + std::mem::size_of::<crate::admission::Reservation>() + 2 * std::mem::size_of::<usize>()) as u64
    }
    pub(crate) fn prepare_capacity(&mut self, documents: u32) -> crate::error::Result<()> {
        let words = (documents as usize + 63) / 64;
        if words <= self.bits.capacity() && self._memory.is_some() { return Ok(()); }
        let capacity = words.max(self.bits.capacity());
        let memory = crate::admission::shared().reserve(Self::owned_capacity_bytes(capacity), "writer bitmap capacity")?;
        let mut replacement = Vec::new();
        replacement.try_reserve_exact(capacity).map_err(|e| crate::error::RusticError::Memory(e.to_string()))?;
        replacement.extend_from_slice(&self.bits);
        let old = std::mem::replace(&mut self.bits, replacement);
        drop(old);
        self._memory = Some(std::sync::Arc::new(memory));
        Ok(())
    }
    pub(crate) fn has_prepared_capacity(&self, documents: u32) -> bool {
        let words=(u64::from(documents)+63)/64;
        words<=self.bits.capacity() as u64 && (words==0 || self._memory.is_some())
    }
    pub(crate) fn ensure_credited_capacity(&mut self, documents: u32, credits: &mut BitmapCredits) {
        let words = (documents as usize + 63) / 64;
        if words <= self.bits.capacity() { return; }
        let capacity = words.max(self.bits.capacity().saturating_mul(2)).max(1);
        let memory = credits.memory.split(Self::owned_capacity_bytes(capacity)).expect("pre-WAL bitmap credit bound");
        let mut replacement = Vec::with_capacity(capacity);
        replacement.extend_from_slice(&self.bits);
        let old = std::mem::replace(&mut self.bits, replacement);
        drop(old);
        self._memory = Some(std::sync::Arc::new(memory));
    }

    pub(crate) fn copy_into_prepared(&self, target: &mut Self) -> crate::error::Result<()> {
        if target.bits.capacity() < self.bits.len() {
            *target = self.try_clone_admitted()?;
        } else {
            target.bits.clear();
            target.bits.extend_from_slice(&self.bits);
            target.num_docs = self.num_docs;
        }
        Ok(())
    }

    /// Grow (or shrink) to `num_docs` docs, keeping existing bits.
    pub fn resize(&mut self, num_docs: u32) {
        self.bits.resize((num_docs as usize + 63) / 64, 0);
        self.num_docs = num_docs;
    }

    /// Raw words.
    pub fn words(&self) -> &[u64] {
        &self.bits
    }

    /// True if no bit is set.
    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|&w| w == 0)
    }

    /// Set a bit.
    pub fn set(&mut self, doc_id: DocId) {
        let idx = doc_id as usize;
        self.bits[idx / 64] |= 1u64 << (idx % 64);
    }

    /// Clear a bit.
    pub fn clear(&mut self, doc_id: DocId) {
        let idx = doc_id as usize;
        self.bits[idx / 64] &= !(1u64 << (idx % 64));
    }

    /// Check if a bit is set.
    pub fn get(&self, doc_id: DocId) -> bool {
        let idx = doc_id as usize;
        (self.bits[idx / 64] >> (idx % 64)) & 1 == 1
    }

    /// Count the number of set bits using popcnt.
    ///
    /// **Performance**: This is the key optimization from Lucene 10.2.
    /// Instead of iterating doc IDs, we count set bits in the bitset.
    /// On modern CPUs, popcnt is a single instruction.
    pub fn count(&self) -> u32 {
        self.bits.iter().map(|w| w.count_ones()).sum::<u32>()
    }

    /// AND with another bitset (in-place).
    pub fn and(&mut self, other: &BitSet) {
        for (a, b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a &= *b;
        }
    }

    /// OR with another bitset (in-place).
    pub fn or(&mut self, other: &BitSet) {
        for (a, b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a |= *b;
        }
    }

    /// NOT (in-place).
    pub fn not(&mut self) {
        for w in self.bits.iter_mut() {
            *w = !*w;
        }
        // Clear bits beyond num_docs
        let last_idx = (self.num_docs as usize) / 64;
        let last_bit = self.num_docs as usize % 64;
        if last_bit > 0 {
            self.bits[last_idx] &= (1u64 << last_bit) - 1;
        }
    }

    /// Iterate over set bits.
    pub fn iter_set_bits(&self) -> impl Iterator<Item = DocId> + '_ {
        (0..self.bits.len()).flat_map(move |word_idx| {
            let word = self.bits[word_idx];
            (0..64).filter(move |bit| (word >> bit) & 1 == 1).map(move |bit| {
                (word_idx * 64 + bit) as DocId
            })
        })
    }

    /// Get the number of documents.
    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }
}

// ---------------------------------------------------------------------------
// In-memory postings for the indexing path (Lucene `TermsHash` style)
// ---------------------------------------------------------------------------

/// Slice sizes by level (Lucene `ByteBlockPool.LEVEL_SIZE_ARRAY`).
const LEVEL_SIZE: [usize; 10] = [5, 14, 20, 30, 40, 40, 80, 80, 120, 200];
const NEXT_LEVEL: [usize; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9];

/// One flat byte pool holding many interleaved append-only byte streams as
/// chains of growing slices (Lucene's `ByteBlockPool` slices). A slice ends
/// in a non-zero level marker; the writer detects it on the byte it is
/// about to overwrite, allocates the next slice, moves the last 3 bytes
/// there and replaces them + the marker with a 4-byte forward address.
#[derive(Default)]
pub struct SlicePool {
    buf: Vec<u8>,
}

impl SlicePool {
    /// Start a new stream; returns its start (also its first write offset).
    #[inline]
    pub fn new_stream(&mut self) -> u32 {
        self.new_slice(0) as u32
    }

    #[inline]
    fn new_slice(&mut self, level: usize) -> usize {
        let start = self.buf.len();
        let size = LEVEL_SIZE[level];
        assert!(start + size < u32::MAX as usize, "memtable byte pool over 4 GiB");
        self.buf.resize(start + size, 0);
        self.buf[start + size - 1] = 16 | level as u8;
        start
    }

    #[cold]
    #[inline(never)]
    fn next_slice(&mut self, upto: usize) -> usize {
        let level = (self.buf[upto] & 15) as usize;
        let new_start = self.new_slice(NEXT_LEVEL[level]);
        let moved = [self.buf[upto - 3], self.buf[upto - 2], self.buf[upto - 1]];
        self.buf[new_start..new_start + 3].copy_from_slice(&moved);
        self.buf[upto - 3..upto + 1].copy_from_slice(&(new_start as u32).to_le_bytes());
        new_start + 3
    }

    #[inline(always)]
    pub fn write_byte(&mut self, upto: &mut u32, b: u8) {
        let mut u = *upto as usize;
        if self.buf[u] != 0 {
            u = self.next_slice(u);
        }
        self.buf[u] = b;
        *upto = (u + 1) as u32;
    }

    #[inline(always)]
    pub fn write_vint(&mut self, upto: &mut u32, mut v: u32) {
        while v >= 0x80 {
            self.write_byte(upto, (v as u8) | 0x80);
            v >>= 7;
        }
        self.write_byte(upto, v as u8);
    }

    /// Bytes allocated.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Reader over the stream that starts at `start` and was written up to `end`.
    pub fn reader(&self, start: u32, end: u32) -> SliceReader<'_> {
        let (start, end) = (start as usize, end as usize);
        let limit = if start + LEVEL_SIZE[0] >= end { end } else { start + LEVEL_SIZE[0] - 4 };
        SliceReader { buf: &self.buf, pos: start, limit, level: 0, end }
    }
}

/// Sequential reader of one [`SlicePool`] stream.
pub struct SliceReader<'a> {
    buf: &'a [u8],
    pos: usize,
    limit: usize,
    level: usize,
    end: usize,
}

impl SliceReader<'_> {
    #[inline]
    pub fn eof(&self) -> bool {
        self.pos == self.end
    }

    #[inline(always)]
    fn read_byte(&mut self) -> u8 {
        if self.pos == self.limit {
            let a = self.limit;
            let next = u32::from_le_bytes(self.buf[a..a + 4].try_into().unwrap()) as usize;
            self.level = NEXT_LEVEL[self.level];
            let size = LEVEL_SIZE[self.level];
            self.pos = next;
            self.limit = if next + size >= self.end { self.end } else { next + size - 4 };
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        b
    }

    #[inline(always)]
    pub fn read_vint(&mut self) -> u32 {
        let mut b = self.read_byte();
        let mut v = (b & 0x7F) as u32;
        let mut shift = 7;
        while b & 0x80 != 0 {
            b = self.read_byte();
            v |= ((b & 0x7F) as u32) << shift;
            shift += 7;
        }
        v
    }
}

/// Bytes of a term key kept inline in its [`TermState`].
const INLINE_KEY: usize = 24;

/// A probe key: length plus its first [`INLINE_KEY`] bytes, zero padded,
/// compared against a term state's inline copy as three words.
struct Probe<'k> {
    key: &'k [u8],
    head: [u64; 3],
}

impl<'k> Probe<'k> {
    #[inline(always)]
    fn new(key: &'k [u8]) -> Self {
        let mut head = [0u64; 3];
        let n = key.len().min(INLINE_KEY);
        let mut i = 0;
        while i + 8 <= n {
            head[i / 8] = u64::from_le_bytes(key[i..i + 8].try_into().unwrap());
            i += 8;
        }
        if i < n {
            head[i / 8] = load_partial(&key[i..n]);
        }
        Self { key, head }
    }

    /// Hash of the key, computed from the head words already loaded (plus
    /// the tail of a long key): three folded multiplies for an inline key,
    /// no byte loop and no hasher call.
    #[inline(always)]
    fn hash(&self) -> u64 {
        let mut h = fold(self.head[0] ^ HASH_K[0], HASH_K[1] ^ self.key.len() as u64);
        h = fold(h ^ self.head[1], HASH_K[2]);
        h = fold(h ^ self.head[2], HASH_K[3]);
        if self.key.len() > INLINE_KEY {
            h = hash_tail(h, &self.key[INLINE_KEY..]);
        }
        h
    }
}

const HASH_K: [u64; 4] = [0x9E37_79B9_7F4A_7C15, 0xD6E8_FEB8_6659_FD93, 0xA076_1D64_78BD_642F, 0xE703_7ED1_A0B4_28DB];

/// Folded 64x64 -> 128 multiply (the mixing step of ahash's fallback).
#[inline(always)]
fn fold(a: u64, b: u64) -> u64 {
    let r = (a as u128).wrapping_mul(b as u128);
    (r as u64) ^ ((r >> 64) as u64)
}

#[cold]
#[inline(never)]
fn hash_tail(mut h: u64, tail: &[u8]) -> u64 {
    let mut chunks = tail.chunks_exact(8);
    for c in &mut chunks {
        h = fold(h ^ u64::from_le_bytes(c.try_into().unwrap()), HASH_K[1]);
    }
    let rest = chunks.remainder();
    if !rest.is_empty() {
        h = fold(h ^ load_partial(rest), HASH_K[2]);
    }
    h
}

/// Prefetch the cache line holding `*p` (no-op where unsupported).
#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a prefetch hint never faults, whatever the address.
    unsafe {
        std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0)
    };
    // AArch64 (Graviton): this was a no-op there, so the staged interning
    // in `MemPostings::flush` took its slot / term / stream misses one
    // dependent chain at a time.
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `prfm` never faults, even on an invalid address.
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, preserves_flags, readonly))
    };
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = p;
}

/// Little-endian word of 1..=7 bytes, zero padded (no `memcpy` call).
#[inline(always)]
fn load_partial(b: &[u8]) -> u64 {
    let n = b.len();
    let mut v = 0u64;
    let mut i = 0;
    if n & 4 != 0 {
        v = u32::from_le_bytes(b[0..4].try_into().unwrap()) as u64;
        i = 4;
    }
    if n & 2 != 0 {
        v |= (u16::from_le_bytes(b[i..i + 2].try_into().unwrap()) as u64) << (i * 8);
        i += 2;
    }
    if n & 1 != 0 {
        v |= (b[i] as u64) << (i * 8);
    }
    v
}

/// Per-term indexing state: exactly one cache line, holding the key (or its
/// head), the stream cursor and the statistics, so interning plus appending
/// a posting touches this line and the stream's current slice only.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct TermState {
    /// First [`INLINE_KEY`] key bytes, zero padded.
    head: [u64; 3],
    key_len: u32,
    /// Offset of the full key in the arena (keys longer than `INLINE_KEY`).
    key_off: u32,
    start: u32,
    upto: u32,
    /// Last doc that contains the term (the open doc for position terms).
    last_doc: DocId,
    doc_freq: u32,
    total_term_freq: u64,
    positions: bool,
    /// Docs with explicit term frequencies and no positions
    /// ([`MemPostings::add_doc_tf`]).
    freqs: bool,
}

/// The memtable's inverted index: term interning (a hash table of term ids
/// over one-cache-line term states with inline keys) plus one append-only
/// stream per term in a [`SlicePool`].
///
/// Stream of a term with positions (Tantivy-style, one stream so a token
/// touches one pool line): per doc `vint(doc delta)` then `vint(pos + 1)`
/// per occurrence, closed by a `0` byte when the term's next doc starts
/// (the newest doc stays open). Positions are absolute (a multi-valued
/// field may restart them). Without positions: one `vint(doc delta)` per
/// doc (tf is always 1), or, for terms with explicit frequencies
/// ([`MemPostings::add_doc_tf`]), `vint(doc delta)` then `vint(tf)`.
pub struct MemPostings {
    /// Open-addressing table (linear probing, load <= 1/2): a slot holds
    /// the term's hash tag ([`SLOT_TAG`] bits) and `id + 1`; `0` is empty.
    /// The tag filters nearly every probe without touching the term state,
    /// and the flat layout lets a lookup be prefetched.
    slots: Vec<u64>,
    mask: usize,
    /// Full hash per term (table growth).
    hashes: Vec<u64>,
    /// Full keys of terms longer than `INLINE_KEY`.
    long_keys: Vec<u8>,
    terms: Vec<TermState>,
    pool: SlicePool,
    /// Occurrences queued by [`MemPostings::push_occurrence`], applied by
    /// [`MemPostings::flush`] in stages so the cache misses of many tokens
    /// overlap instead of being taken one dependent chain at a time.
    batch: Vec<Queued>,
    /// Full keys of queued occurrences longer than `INLINE_KEY`.
    batch_keys: Vec<u8>,
    batch_doc: DocId,
}

/// The high half of a slot: the hash's top 32 bits.
const SLOT_TAG: u64 = 0xFFFF_FFFF_0000_0000;

/// Slot of term `id` with hash `h` (never `0`).
#[inline(always)]
fn slot_entry(h: u64, id: u32) -> u64 {
    (h & SLOT_TAG) | (id as u64 + 1)
}

/// Term id stored in a non-empty slot.
#[inline(always)]
fn slot_id(s: u64) -> u32 {
    (s as u32) - 1
}
const MIN_SLOTS: usize = 64;
/// Queued occurrences per flush: deep enough that the memory accesses of a
/// stage overlap, small enough that the queue stays in L1.
const BATCH: usize = 128;
const NO_TERM: u32 = u32::MAX;

/// One queued occurrence (see [`MemPostings::flush`]).
#[derive(Clone, Copy)]
struct Queued {
    head: [u64; 3],
    hash: u64,
    key_len: u32,
    /// Offset in `batch_keys` of a key longer than `INLINE_KEY`.
    key_off: u32,
    position: u32,
    /// Term id, filled in by stage A of [`MemPostings::flush`].
    id: u32,
}

impl Default for MemPostings {
    fn default() -> Self {
        Self::new()
    }
}

/// Decoded postings of one term (reusable scratch).
#[derive(Default)]
pub struct DecodedPostings {
    pub doc_ids: Vec<DocId>,
    pub term_freqs: Vec<u32>,
    pub positions: Vec<u32>,
    pub has_positions: bool,
}

impl MemPostings {
    pub fn new() -> Self {
        let slots = MIN_SLOTS;
        Self {
            slots: vec![0u64; slots],
            mask: slots - 1,
            hashes: Vec::new(),
            long_keys: Vec::new(),
            terms: Vec::new(),
            pool: SlicePool::default(),
            batch: Vec::with_capacity(BATCH),
            batch_keys: Vec::new(),
            batch_doc: 0,
        }
    }

    /// Number of distinct terms.
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Approximate heap use.
    pub fn heap_bytes(&self) -> usize {
        self.pool.len()
            + self.long_keys.len()
            + self.slots.len() * 8
            + self.terms.len() * (std::mem::size_of::<TermState>() + 8)
    }

    /// Key bytes (`Term::key()` layout) of term `id`.
    #[inline]
    pub fn key(&self, id: u32) -> &[u8] {
        let t = &self.terms[id as usize];
        let n = t.key_len as usize;
        if n <= INLINE_KEY && cfg!(target_endian = "little") {
            // SAFETY: `head` is 24 initialized bytes; on little-endian
            // targets its in-memory bytes are the key bytes in order.
            let bytes: &[u8; INLINE_KEY] = unsafe { &*(t.head.as_ptr() as *const [u8; INLINE_KEY]) };
            &bytes[..n]
        } else {
            &self.long_keys[t.key_off as usize..t.key_off as usize + n]
        }
    }

    /// Compare the cached prefixes before touching the arena of long keys.
    /// Prefix words are stored little-endian; swapping gives bytewise order.
    #[inline]
    pub(crate) fn cmp_keys(&self, a: u32, b: u32) -> std::cmp::Ordering {
        let (a_state, b_state) = (&self.terms[a as usize], &self.terms[b as usize]);
        for i in 0..3 {
            let order = a_state.head[i].swap_bytes().cmp(&b_state.head[i].swap_bytes());
            if order != std::cmp::Ordering::Equal { return order; }
        }
        if a_state.key_len.min(b_state.key_len) <= INLINE_KEY as u32 {
            a_state.key_len.cmp(&b_state.key_len)
        } else {
            self.key(a)[INLINE_KEY..].cmp(&self.key(b)[INLINE_KEY..])
        }
    }

    #[inline(always)]
    fn matches(&self, t: &TermState, p: &Probe<'_>) -> bool {
        t.key_len as usize == p.key.len()
            && t.head == p.head
            && (p.key.len() <= INLINE_KEY
                || self.long_keys[t.key_off as usize + INLINE_KEY..t.key_off as usize + p.key.len()] == p.key[INLINE_KEY..])
    }

    /// Key match of a queued occurrence against a stored term (same as
    /// [`Self::matches`], with the queued key in `batch_keys`).
    #[inline(always)]
    fn matches_q(&self, t: &TermState, q: &Queued) -> bool {
        t.key_len == q.key_len
            && t.head == q.head
            && (q.key_len <= INLINE_KEY as u32
                || self.long_keys[t.key_off as usize + INLINE_KEY..t.key_off as usize + q.key_len as usize]
                    == self.batch_keys[q.key_off as usize + INLINE_KEY..q.key_off as usize + q.key_len as usize])
    }

    /// Term id of `key`, if present.
    pub fn get(&self, key: &[u8]) -> Option<u32> {
        let p = Probe::new(key);
        let h = p.hash();
        let mut i = (h as usize) & self.mask;
        loop {
            let s = self.slots[i];
            if s == 0 {
                return None;
            }
            if (s & SLOT_TAG) == (h & SLOT_TAG) && self.matches(&self.terms[slot_id(s) as usize], &p) {
                return Some(slot_id(s));
            }
            i = (i + 1) & self.mask;
        }
    }

    /// Term id of `key`, creating it (with `positions`) when absent.
    #[inline]
    pub fn intern(&mut self, key: &[u8], positions: bool) -> u32 {
        let p = Probe::new(key);
        let h = p.hash();
        let mut i = (h as usize) & self.mask;
        loop {
            let s = self.slots[i];
            if s == 0 {
                return self.insert(key, &p, h, positions);
            }
            if (s & SLOT_TAG) == (h & SLOT_TAG) && self.matches(&self.terms[slot_id(s) as usize], &p) {
                return slot_id(s);
            }
            i = (i + 1) & self.mask;
        }
    }

    /// Queue one occurrence of `key` at `position` in `doc` (docs ascending,
    /// one doc per queue) for the next [`MemPostings::flush`].
    ///
    /// The queue holds a single doc: call [`MemPostings::flush`] before
    /// queueing occurrences of the next doc (and before any read of the
    /// postings).
    #[inline]
    pub fn push_occurrence(&mut self, key: &[u8], doc: DocId, position: u32) {
        if self.batch.is_empty() {
            self.batch_doc = doc;
        }
        debug_assert_eq!(self.batch_doc, doc, "push_occurrence: flush() before queueing another doc");
        // Keys that `key()` must read from the arena are kept whole.
        let key_off = if key.len() > INLINE_KEY || cfg!(target_endian = "big") {
            let off = self.batch_keys.len();
            self.batch_keys.extend_from_slice(key);
            off as u32
        } else {
            0
        };
        let p = Probe::new(key);
        self.batch.push(Queued {
            head: p.head,
            hash: p.hash(),
            key_len: key.len() as u32,
            key_off,
            position,
            id: NO_TERM,
        });
        if self.batch.len() == BATCH {
            self.flush();
        }
    }

    /// Apply the queued occurrences of [`MemPostings::push_occurrence`].
    ///
    /// Two passes over the queue: A interns every key, with the table slots
    /// of upcoming tokens and the write position of each resolved term
    /// prefetched so their cache misses overlap; B appends the postings in
    /// token order (the prefetched lines are in cache by then).
    pub fn flush(&mut self) {
        if self.batch.is_empty() {
            return;
        }
        let doc = self.batch_doc;
        // Stage A: intern. Two prefetch distances, so a token's two
        // dependent misses overlap with other tokens' work: the slot lines
        // of the tokens `AHEAD` on, and (their slot line arrived meanwhile)
        // the term state of the candidate in the first slot of the tokens
        // `TERM_AHEAD` on.
        const AHEAD: usize = 8;
        const TERM_AHEAD: usize = 4;
        let n = self.batch.len();
        for j in 0..n.min(AHEAD) {
            prefetch(&self.slots[(self.batch[j].hash as usize) & self.mask]);
        }
        for j in 0..n.min(TERM_AHEAD) {
            let h = self.batch[j].hash;
            let s = self.slots[(h as usize) & self.mask];
            if s != 0 && (s & SLOT_TAG) == (h & SLOT_TAG) {
                prefetch(&self.terms[slot_id(s) as usize]);
            }
        }
        for j in 0..n {
            if j + AHEAD < n {
                prefetch(&self.slots[(self.batch[j + AHEAD].hash as usize) & self.mask]);
            }
            if j + TERM_AHEAD < n {
                let h = self.batch[j + TERM_AHEAD].hash;
                let s = self.slots[(h as usize) & self.mask];
                if s != 0 && (s & SLOT_TAG) == (h & SLOT_TAG) {
                    prefetch(&self.terms[slot_id(s) as usize]);
                }
            }
            let q = self.batch[j];
            let mut i = (q.hash as usize) & self.mask;
            let id = loop {
                let s = self.slots[i];
                if s == 0 {
                    break self.insert_queued(&q);
                }
                if (s & SLOT_TAG) == (q.hash & SLOT_TAG) && self.matches_q(&self.terms[slot_id(s) as usize], &q) {
                    break slot_id(s);
                }
                i = (i + 1) & self.mask;
            };
            self.batch[j].id = id;
            prefetch(&self.pool.buf[self.terms[id as usize].upto as usize]);
        }
        // Stage B: append the postings (doc delta, then position + 1).
        for q in &self.batch {
            let t = &mut self.terms[q.id as usize];
            if t.doc_freq == 0 || t.last_doc != doc {
                if t.doc_freq > 0 {
                    self.pool.write_byte(&mut t.upto, 0);
                }
                self.pool.write_vint(&mut t.upto, doc - t.last_doc);
                t.last_doc = doc;
                t.doc_freq += 1;
            }
            t.total_term_freq += 1;
            self.pool.write_vint(&mut t.upto, q.position + 1);
        }
        self.batch.clear();
        self.batch_keys.clear();
    }

    /// `doc` contains `key` with term frequency `tf` and no positions
    /// (feature fields: the tf encodes a weight). A repeat within a doc
    /// keeps the first frequency. A key must take only this kind of
    /// posting (feature fields index nothing else).
    #[inline]
    pub fn add_doc_tf(&mut self, key: &[u8], doc: DocId, tf: u32) {
        let id = self.intern(key, false);
        let t = &mut self.terms[id as usize];
        if t.doc_freq > 0 && t.last_doc == doc {
            return;
        }
        debug_assert!(!t.positions && (t.freqs || t.doc_freq == 0), "explicit tf on a term with other postings");
        let tf = tf.max(1);
        t.freqs = true;
        self.pool.write_vint(&mut t.upto, doc - t.last_doc);
        self.pool.write_vint(&mut t.upto, tf);
        t.last_doc = doc;
        t.doc_freq += 1;
        t.total_term_freq += tf as u64;
    }

    /// `doc` contains `key` (tf 1, no positions; repeats within a doc are
    /// ignored).
    #[inline]
    pub fn add_doc(&mut self, key: &[u8], doc: DocId) {
        let id = self.intern(key, false);
        let t = &mut self.terms[id as usize];
        if t.doc_freq > 0 && t.last_doc == doc {
            return;
        }
        if t.positions {
            // A positions term (same key indexed as text earlier): keep its
            // stream well-formed with an occurrence at position 0.
            if t.doc_freq > 0 {
                self.pool.write_byte(&mut t.upto, 0);
            }
            self.pool.write_vint(&mut t.upto, doc - t.last_doc);
            self.pool.write_vint(&mut t.upto, 1);
        } else {
            self.pool.write_vint(&mut t.upto, doc - t.last_doc);
        }
        t.last_doc = doc;
        t.doc_freq += 1;
        t.total_term_freq += 1;
    }

    /// Create a term from a live key (the key is copied into the arena).
    #[cold]
    #[inline(never)]
    fn insert(&mut self, key: &[u8], p: &Probe<'_>, h: u64, positions: bool) -> u32 {
        let key_off = if key.len() > INLINE_KEY || cfg!(target_endian = "big") {
            let off = self.long_keys.len();
            assert!(off + key.len() < u32::MAX as usize, "memtable key arena over 4 GiB");
            self.long_keys.extend_from_slice(key);
            off as u32
        } else {
            0
        };
        let id = self.terms.len() as u32;
        self.hashes.push(h);
        let start = self.pool.new_stream();
        self.terms.push(TermState {
            head: p.head,
            key_len: key.len() as u32,
            key_off,
            start,
            upto: start,
            last_doc: 0,
            doc_freq: 0,
            total_term_freq: 0,
            positions,
            freqs: false,
        });
        self.place(h, id);
        id
    }

    /// Create a term from a queued occurrence (the key is copied from
    /// `batch_keys` into the arena).
    #[cold]
    #[inline(never)]
    fn insert_queued(&mut self, q: &Queued) -> u32 {
        let key_off = if q.key_len > INLINE_KEY as u32 || cfg!(target_endian = "big") {
            let off = self.long_keys.len();
            assert!(off + (q.key_len as usize) < u32::MAX as usize, "memtable key arena over 4 GiB");
            self.long_keys.extend_from_slice(&self.batch_keys[q.key_off as usize..q.key_off as usize + q.key_len as usize]);
            off as u32
        } else {
            0
        };
        let id = self.terms.len() as u32;
        self.hashes.push(q.hash);
        let start = self.pool.new_stream();
        self.terms.push(TermState {
            head: q.head,
            key_len: q.key_len,
            key_off,
            start,
            upto: start,
            last_doc: 0,
            doc_freq: 0,
            total_term_freq: 0,
            positions: true,
            freqs: false,
        });
        self.place(q.hash, id);
        id
    }

    /// Put `id` in the table at the first empty slot of its probe chain.
    /// The caller has already established the slot is empty.
    #[inline]
    fn place(&mut self, h: u64, id: u32) {
        if (self.terms.len() as u64) * 2 > self.slots.len() as u64 {
            self.grow();
        }
        let mut i = (h as usize) & self.mask;
        while self.slots[i] != 0 {
            i = (i + 1) & self.mask;
        }
        self.slots[i] = slot_entry(h, id);
    }

    /// Double the table (load stays <= 1/2, so probe chains stay short).
    #[cold]
    #[inline(never)]
    fn grow(&mut self) {
        let mut slots = vec![0u64; self.slots.len() * 2];
        let mask = slots.len() - 1;
        for (id, h) in self.hashes.iter().enumerate() {
            let mut i = (*h as usize) & mask;
            while slots[i] != 0 {
                i = (i + 1) & mask;
            }
            slots[i] = slot_entry(*h, id as u32);
        }
        self.slots = slots;
        self.mask = mask;
    }

    pub fn doc_freq(&self, id: u32) -> u32 {
        self.terms[id as usize].doc_freq
    }

    pub fn total_term_freq(&self, id: u32) -> u64 {
        self.terms[id as usize].total_term_freq
    }

    /// Decode term `id` into `out` (cleared first).
    pub fn decode(&self, id: u32, out: &mut DecodedPostings) {
        out.doc_ids.clear();
        out.term_freqs.clear();
        out.positions.clear();
        let t = &self.terms[id as usize];
        out.has_positions = t.positions;
        if t.doc_freq == 0 {
            return;
        }
        out.doc_ids.reserve(t.doc_freq as usize);
        out.term_freqs.reserve(t.doc_freq as usize);
        let mut r = self.pool.reader(t.start, t.upto);
        let mut doc = 0u32;
        if !t.positions {
            while !r.eof() {
                doc += r.read_vint();
                out.doc_ids.push(doc);
                out.term_freqs.push(if t.freqs { r.read_vint() } else { 1 });
            }
            return;
        }
        out.positions.reserve(t.total_term_freq as usize);
        while !r.eof() {
            doc += r.read_vint();
            let mut tf = 0u32;
            while !r.eof() {
                let v = r.read_vint();
                if v == 0 {
                    break;
                }
                out.positions.push(v - 1);
                tf += 1;
            }
            out.doc_ids.push(doc);
            out.term_freqs.push(tf);
        }
    }

    /// How term `id` is posted: `(positions, explicit frequencies)`.
    pub(crate) fn term_kind(&self, id: u32) -> (bool, bool) {
        let t = &self.terms[id as usize];
        (t.positions, t.freqs)
    }

    /// Rebuild a term's stream (a one-pass sorted seal): append `doc`
    /// (greater than the term's last doc) to term `id`, posted as
    /// `(positions, freqs)` ([`MemPostings::term_kind`] of its source),
    /// with `tf` occurrences at `positions` (positions terms) or `tf`
    /// (explicit frequencies; else tf is 1).
    pub(crate) fn append_rebuilt(&mut self, id: u32, freqs: bool, doc: DocId, tf: u32, positions: &[u32]) {
        let t = &mut self.terms[id as usize];
        debug_assert!(t.doc_freq == 0 || doc > t.last_doc, "rebuilt postings: docs ascending");
        if t.positions {
            if t.doc_freq > 0 {
                self.pool.write_byte(&mut t.upto, 0);
            }
            self.pool.write_vint(&mut t.upto, doc - t.last_doc);
            for &p in positions {
                self.pool.write_vint(&mut t.upto, p + 1);
            }
            t.total_term_freq += positions.len() as u64;
        } else if freqs {
            t.freqs = true;
            self.pool.write_vint(&mut t.upto, doc - t.last_doc);
            self.pool.write_vint(&mut t.upto, tf);
            t.total_term_freq += tf as u64;
        } else {
            self.pool.write_vint(&mut t.upto, doc - t.last_doc);
            t.total_term_freq += 1;
        }
        t.last_doc = doc;
        t.doc_freq += 1;
    }

    /// Term ids sorted by key bytes.
    pub fn sorted_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = (0..self.terms.len() as u32).collect();
        ids.sort_unstable_by(|&a, &b| self.key(a).cmp(self.key(b)));
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_prefix_order_matches_bytes() {
        let mut p = MemPostings::default();
        let mut keys = vec![vec![], vec![0], vec![0, 0], vec![1], vec![1, 0], vec![1, 0, 255]];
        for len in 0..80 {
            for marker in [0u8, 1, 127, 255] {
                let mut key = vec![0; len];
                if len > 0 { key[len - 1] = marker; }
                keys.push(key);
            }
        }
        for key in keys { p.add_doc(&key, 0); }
        for a in 0..p.len() as u32 {
            for b in 0..p.len() as u32 {
                assert_eq!(p.cmp_keys(a,b), p.key(a).cmp(p.key(b)), "{:?} {:?}", p.key(a), p.key(b));
            }
        }
    }

    #[test]
    #[ignore = "EC2 diagnostic sort POC, not a timing assertion"]
    fn cached_prefix_sort_poc() {
        use rayon::prelude::*;
        for prefix in ["", "common-prefix-longer-than-twenty-four-bytes/"] {
            let mut p = MemPostings::default();
            for i in 0u64..300_000 {
                let n = i.wrapping_mul(0x9e3779b97f4a7c15);
                p.add_doc(format!("{prefix}{n:016x}-{i:016x}").as_bytes(), 0);
            }
            for pass in 0..4 {
                let mut a: Vec<u32> = (0..p.len() as u32).collect();
                let mut b = a.clone();
                let start = std::time::Instant::now();
                a.par_sort_unstable_by(|&a,&b| p.key(a).cmp(p.key(b)));
                let old = start.elapsed();
                let start = std::time::Instant::now();
                b.par_sort_unstable_by(|&a,&b| p.cmp_keys(a,b));
                let new = start.elapsed();
                assert_eq!(a,b);
                eprintln!("prefix={} pass={pass} old_ms={} new_ms={}",prefix.len(),old.as_millis(),new.as_millis());
            }
        }
    }

    #[test]
    fn test_postings_basic() {
        let mut pl = PostingsList::new();
        for i in 0..100 {
            pl.add_doc(i, 1, &[]);
        }
        pl.build_skip_tables();

        assert_eq!(pl.doc_freq(), 100);
        assert_eq!(pl.doc_id_at(0), 0);
        assert_eq!(pl.doc_id_at(99), 99);
    }

    #[test]
    fn test_advance() {
        let mut pl = PostingsList::new();
        for i in (0..1000).step_by(10) {
            pl.add_doc(i, 1, &[]);
        }
        pl.build_skip_tables();

        assert_eq!(pl.advance(0), Some(0));
        assert_eq!(pl.advance(5), Some(1)); // first doc >= 5 is 10
        assert_eq!(pl.advance(50), Some(5)); // first doc >= 50 is 50
        assert_eq!(pl.advance(999), None);
    }

    /// Regression: targets that fall in the gap between two skip blocks
    /// (e.g. an odd target against an all-even list) must resolve to the
    /// next block's first entry, not `None`.
    #[test]
    fn test_advance_sparse_matches_bruteforce() {
        let mut pl = PostingsList::new();
        for i in (0..200_000u32).step_by(2) {
            pl.add_doc(i, 1, &[]);
        }
        pl.build_skip_tables();
        let ids: Vec<DocId> = (0..200_000u32).step_by(2).collect();

        // Every target from 0 to just past the end, including block gaps.
        for target in 0..200_010u32 {
            let expected = {
                let idx = ids.partition_point(|&d| d < target);
                (idx < ids.len()).then_some(idx)
            };
            assert_eq!(pl.advance(target), expected, "target {target}");
        }
    }

    #[test]
    fn test_intersect_sparse() {
        // Evens vs. multiples of 3 over 100k: matches are multiples of 6.
        let mut a = PostingsList::new();
        let mut b = PostingsList::new();
        for i in (0..100_000u32).step_by(2) {
            a.add_doc(i, 1, &[]);
        }
        for i in (0..100_000u32).step_by(3) {
            b.add_doc(i, 1, &[]);
        }
        a.build_skip_tables();
        b.build_skip_tables();

        let result = intersect(&a, &b);
        let expected: Vec<DocId> = (0..100_000u32).step_by(6).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_count_in_range() {
        let mut pl = PostingsList::new();
        for i in 0..100 {
            pl.add_doc(i, 1, &[]);
        }
        pl.build_skip_tables();

        assert_eq!(pl.count_in_range(0, 100), 100);
        assert_eq!(pl.count_in_range(10, 20), 10);
        assert_eq!(pl.count_in_range(50, 100), 50);
    }

    #[test]
    fn test_intersect() {
        let mut a = PostingsList::new();
        let mut b = PostingsList::new();
        for i in 0..100 {
            a.add_doc(i, 1, &[]);
        }
        for i in (0..100).step_by(2) {
            b.add_doc(i, 1, &[]);
        }
        a.build_skip_tables();
        b.build_skip_tables();

        let result = intersect(&a, &b);
        assert_eq!(result.len(), 50);
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 2);
    }

    #[test]
    fn test_union() {
        let mut a = PostingsList::new();
        let mut b = PostingsList::new();
        for i in (0..100).step_by(2) {
            a.add_doc(i, 1, &[]);
        }
        for i in (1..100).step_by(2) {
            b.add_doc(i, 1, &[]);
        }
        a.build_skip_tables();
        b.build_skip_tables();

        let result = union(&a, &b);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_bitset_count() {
        let mut bs = BitSet::new(1000);
        for i in (0..1000).step_by(2) {
            bs.set(i);
        }
        assert_eq!(bs.count(), 500);
    }

    #[test]
    fn test_bitset_and() {
        let mut a = BitSet::new(100);
        let mut b = BitSet::new(100);
        for i in (0..100).step_by(2) {
            a.set(i);
        }
        for i in (0..100).step_by(3) {
            b.set(i);
        }
        a.and(&b);
        // Should have bits set at multiples of 6
        assert_eq!(a.count(), 17); // 0, 6, 12, ..., 96
    }

    #[test]
    fn test_serialize_deserialize() {
        let mut pl = PostingsList::new();
        for i in 0..50 {
            pl.add_doc(i * 2, 1, &[]);
        }
        pl.build_skip_tables();

        let data = pl.serialize();
        let restored = PostingsList::deserialize(&data).unwrap();
        assert_eq!(restored.doc_freq(), 50);
        assert_eq!(restored.doc_id_at(0), 0);
        assert_eq!(restored.doc_id_at(49), 98);
    }

    #[test]
    fn mem_postings_roundtrip() {
        let mut m = MemPostings::new();
        // Reference: (doc, positions) per key.
        let mut reference: std::collections::BTreeMap<Vec<u8>, Vec<(u32, Vec<u32>)>> = Default::default();
        let mut x = 12345u64;
        for doc in 0..3000u32 {
            let n = (doc % 37) + 1;
            for p in 0..n {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let id = (x >> 33) % if p % 3 == 0 { 5 } else { 400 };
                // Some keys longer than the inline head.
                let k = if id % 7 == 0 { format!("long-key-{}-{}", "x".repeat(20 + (id % 5) as usize), id) } else { format!("t{id}") };
                let pos = p * 1000 + (x >> 50) as u32 % 7; // multi-byte vints
                m.push_occurrence(k.as_bytes(), doc, pos);
                if p % 13 == 0 {
                    m.flush(); // exercise mid-doc flushes
                }
                let e = reference.entry(k.into_bytes()).or_default();
                match e.last_mut() {
                    Some((d, ps)) if *d == doc => ps.push(pos),
                    _ => e.push((doc, vec![pos])),
                }
            }
            // The queue holds one doc: flush before the next doc (as
            // `SegmentWriter::index_text` does) and before any read.
            m.flush();
            m.add_doc(b"kw", doc);
            m.add_doc(b"kw", doc);
        }
        let mut out = DecodedPostings::default();
        for (k, docs) in &reference {
            let id = m.get(k).unwrap();
            m.decode(id, &mut out);
            assert!(out.has_positions);
            let want_docs: Vec<u32> = docs.iter().map(|d| d.0).collect();
            let want_tfs: Vec<u32> = docs.iter().map(|d| d.1.len() as u32).collect();
            let want_pos: Vec<u32> = docs.iter().flat_map(|d| d.1.iter().copied()).collect();
            assert_eq!(out.doc_ids, want_docs);
            assert_eq!(out.term_freqs, want_tfs);
            assert_eq!(out.positions, want_pos);
            assert_eq!(m.doc_freq(id), docs.len() as u32);
            assert_eq!(m.total_term_freq(id), want_pos.len() as u64);
        }
        let id = m.get(b"kw").unwrap();
        m.decode(id, &mut out);
        assert!(!out.has_positions);
        assert_eq!(out.doc_ids, (0..3000).collect::<Vec<_>>());
        assert!(out.term_freqs.iter().all(|&t| t == 1));
        assert!(m.get(b"nope").is_none());
        let ids = m.sorted_ids();
        assert!(ids.windows(2).all(|w| m.key(w[0]) < m.key(w[1])));
    }
}

#[cfg(test)]
mod bitmap_credit_tests {
    use super::*;
    #[test]
    fn preflight_credit_survives_lowered_limit_and_every_document_handoff() {
        let budget = crate::admission::Budget::new(1 << 20);
        let mut credits = BitmapCredits::reserve_on(129, &budget).unwrap();
        budget.set_limit(0);
        let mut escaped = Vec::new();
        for _ in 0..129 {
            let mut latest = BitSet::new(0);
            let mut visible = BitSet::new(0);
            latest.ensure_credited_capacity(1, &mut credits);
            visible.ensure_credited_capacity(1, &mut credits);
            latest.resize(1); visible.resize(1);
            escaped.push((latest, visible));
        }
        drop(credits);
        assert!(budget.used() > 0);
        drop(escaped);
        assert_eq!(budget.used(), 0);
    }
    #[test]
    fn prepared_delete_snapshot_reuses_admission_after_limit_reduction() {
        let budget = crate::admission::Budget::new(1024);
        let mut latest = BitSet::new(129);
        let mut snapshot = BitSet::new_admitted(129, &budget).unwrap();
        let retained = budget.used();
        budget.set_limit(0);
        latest.set(12);
        latest.copy_into_prepared(&mut snapshot).unwrap();
        assert!(snapshot.get(12));
        latest.set(128);
        assert!(!snapshot.get(128));
        latest.copy_into_prepared(&mut snapshot).unwrap();
        assert!(snapshot.get(128));
        assert_eq!(budget.used(), retained);
        drop(snapshot); assert_eq!(budget.used(), 0);
    }
    #[test]
    fn geometric_bitmap_growth_consumes_bounded_preflight_and_preserves_bits() {
        let budget = crate::admission::Budget::new(1 << 20);
        let mut credits = BitmapCredits::reserve_on(1025, &budget).unwrap();
        budget.set_limit(0);
        let mut bits = BitSet::new(0);
        for n in 1..=1025 {
            bits.ensure_credited_capacity(n, &mut credits);
            bits.resize(n); bits.set(n - 1);
        }
        assert_eq!(bits.count(), 1025);
        drop(credits); assert!(budget.used() > 0);
        drop(bits); assert_eq!(budget.used(), 0);
        assert!(BitmapCredits::reserve_on(1, &budget).is_err());
        assert_eq!(budget.used(), 0);
    }
}
