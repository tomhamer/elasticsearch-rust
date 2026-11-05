//! Forward-only iterators over sorted doc ids (Lucene's `DocIdSetIterator`).

use crate::DocId;

/// Sentinel returned by [`DocSet::advance`] / [`DocSet::seek`] when the set
/// is exhausted.
pub const TERMINATED: DocId = u32::MAX;

/// A forward-only iterator over a sorted set of doc ids.
///
/// Iteration is strictly increasing: `advance` moves to the next doc, `seek`
/// jumps to the first doc `>= target`. Before the first `advance` (or after
/// exhaustion) [`DocSet::doc`] returns [`TERMINATED`].
///
/// `Send` is a supertrait so boxed composites (`Box<dyn DocSet>`,
/// `Box<dyn Scorer>`) can cross thread boundaries.
pub trait DocSet: Send {
    /// Advance to the next doc; returns [`TERMINATED`] when exhausted.
    fn advance(&mut self) -> DocId;

    /// Seek to the first doc `>= target` (target is expected to be greater
    /// than the current doc). The default implementation loops `advance`;
    /// implementors with skipping structure should override.
    fn seek(&mut self, target: DocId) -> DocId {
        let mut d = self.doc();
        while d < target {
            d = self.advance();
            if d == TERMINATED {
                return TERMINATED;
            }
        }
        d
    }

    /// Whether exact seeks can test one candidate without scanning intervening
    /// documents. Only opt in for pure membership checks without observable
    /// evaluation errors; conjunctions may omit calls for other documents.
    fn has_fast_exact_seek(&self) -> bool { false }

    /// Whether [`DocSet::seek_exact`] reads a value per candidate (a doc
    /// values range) rather than testing a bit: a conjunction then prefers
    /// leading with it over a clause of about its size.
    fn exact_seek_reads_value(&self) -> bool { false }

    /// Test a monotonically increasing candidate. On success `doc()` and the
    /// scorer are positioned on `target`. On failure the position is unspecified;
    /// a later exact seek/ordinary seek/advance must still work forward.
    fn seek_exact(&mut self, target: DocId) -> bool { self.seek(target) == target && target != TERMINATED }

    /// Called by a conjunction on its exact-seek verifiers, before any
    /// iteration, with the cost (`size_hint`) of the clause that leads it:
    /// a verifier may then pick its strategy (Lucene `ScorerSupplier.get(
    /// leadCost)`, as `IndexOrDocValuesQuery` does). A hint: results must
    /// not depend on it.
    fn set_lead_cost(&mut self, _lead_cost: u32) {}

    /// Current doc, or [`TERMINATED`] before the first advance or after
    /// exhaustion.
    fn doc(&self) -> DocId;

    /// The set's docs as shared words (bit `d` = doc `d`), when it holds or
    /// can cheaply get them and has not started. `None` otherwise.
    fn filter_words(&mut self) -> Option<std::sync::Arc<crate::admission::Allocation<Vec<u64>>>> {
        None
    }

    /// Upper bound on the number of docs in the set (0 if unknown).
    fn size_hint(&self) -> u32;

    /// Bulk entry point for uniformly scored sets: the first `k` live docs
    /// and the exact number of live docs of the whole, not yet advanced,
    /// set. `deleted` = deletion bitset words. `None` = not supported (the
    /// caller iterates).
    fn collect_first_k_bulk(&mut self, _k: usize, _deleted: Option<&[u64]>) -> Option<(Vec<DocId>, u64)> {
        None
    }

    /// Append every remaining doc not set in `deleted` (deletion bitset
    /// words) to `out`, ascending, and exhaust the set: the docs a loop of
    /// [`DocSet::advance`] filtered by the live docs would give. `false`:
    /// not supported, nothing consumed (the caller iterates).
    fn fill_all(&mut self, _deleted: Option<&[u64]>, _out: &mut Vec<DocId>) -> bool {
        false
    }
}

/// A doc set over a sorted `Vec<DocId>`.
pub struct VecDocSet {
    docs: Vec<DocId>,
    /// Index of the next doc to emit.
    pos: usize,
    current: DocId,
}

impl VecDocSet {
    /// Wrap a sorted, deduplicated doc-id vector.
    pub fn new(docs: Vec<DocId>) -> Self {
        Self {
            docs,
            pos: 0,
            current: TERMINATED,
        }
    }

    /// Number of docs in the set.
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }
}

impl DocSet for VecDocSet {
    fn fill_all(&mut self, deleted: Option<&[u64]>, out: &mut Vec<DocId>) -> bool {
        let rest = &self.docs[self.pos.min(self.docs.len())..];
        match deleted {
            None => out.extend_from_slice(rest),
            Some(d) => out.extend(rest.iter().copied().filter(|&x| d.get(x as usize / 64).is_none_or(|w| w & (1u64 << (x % 64)) == 0))),
        }
        self.pos = self.docs.len();
        self.current = TERMINATED;
        true
    }

    fn advance(&mut self) -> DocId {
        if self.pos >= self.docs.len() {
            self.current = TERMINATED;
        } else {
            self.current = self.docs[self.pos];
            self.pos += 1;
        }
        self.current
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.current != TERMINATED && self.current >= target {
            return self.current;
        }
        // Gallop from the next position, then binary-search the window.
        let n = self.docs.len();
        let mut lo = self.pos;
        if lo >= n {
            self.current = TERMINATED;
            return TERMINATED;
        }
        if self.docs[lo] >= target {
            self.current = self.docs[lo];
            self.pos = lo + 1;
            return self.current;
        }
        // docs[lo] < target; find hi with docs[hi] >= target (or hi == n).
        let mut step = 1usize;
        let hi = loop {
            let j = lo + step;
            if j >= n {
                break n;
            }
            if self.docs[j] >= target {
                break j;
            }
            lo = j;
            step *= 2;
        };
        if hi >= n && self.docs[n - 1] < target {
            self.pos = n;
            self.current = TERMINATED;
            return TERMINATED;
        }
        let j = lo + self.docs[lo..hi.min(n)].partition_point(|&d| d < target);
        let j = if j < n && self.docs[j] >= target { j } else { hi };
        self.current = self.docs[j];
        self.pos = j + 1;
        self.current
    }

    fn doc(&self) -> DocId {
        self.current
    }

    fn size_hint(&self) -> u32 {
        self.docs.len() as u32
    }
}

/// A doc set over a bit set of doc ids, stored as 64-bit words.
enum BitSetWords {
    Owned(Vec<u64>),
    Shared(std::sync::Arc<crate::admission::Allocation<Vec<u64>>>),
}
impl std::ops::Deref for BitSetWords {
    type Target=[u64];
    fn deref(&self)->&[u64] { match self {Self::Owned(v)=>v,Self::Shared(v)=>v} }
}
pub struct BitSetDocSet {
    words: BitSetWords,
    /// Index of the word being consumed.
    word_idx: usize,
    /// Remaining (not yet emitted) bits of `words[word_idx]`.
    cur: u64,
    current: DocId,
    /// Number of set bits.
    count: u32,
}

impl BitSetDocSet {
    /// Build a bit set from the given words (bit `i` of word `w` is doc
    /// `w * 64 + i`).
    pub fn new(words: Vec<u64>) -> Self {
        let cur = words.first().copied().unwrap_or(0);
        let count = words.iter().map(|w| w.count_ones()).sum();
        Self {
            words:BitSetWords::Owned(words),
            word_idx: 0,
            cur,
            current: TERMINATED,
            count,
        }
    }

    pub fn shared(words:std::sync::Arc<crate::admission::Allocation<Vec<u64>>>)->Self {
        let count=words.iter().map(|w|w.count_ones()).sum();
        Self::shared_counted(words,count)
    }

    /// [`Self::shared`] with its set-bit count already known (a segment's
    /// cached filter bitset, [`crate::segment::Segment::filter_bits_count`]).
    pub fn shared_counted(words:std::sync::Arc<crate::admission::Allocation<Vec<u64>>>,count:u32)->Self {
        let cur=words.first().copied().unwrap_or(0);
        Self {words:BitSetWords::Shared(words),word_idx:0,cur,current:TERMINATED,count}
    }

    /// Build a bit set from a doc-id slice (any order, duplicates allowed).
    pub fn from_docs(docs: &[DocId]) -> Self {
        let max = docs.iter().copied().max();
        let nwords = max.map_or(0, |m| m as usize / 64 + 1);
        let mut words = vec![0u64; nwords];
        for &d in docs {
            words[d as usize / 64] |= 1u64 << (d % 64);
        }
        Self::new(words)
    }

    /// Number of set bits (docs in the set).
    pub fn len(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }
}

impl DocSet for BitSetDocSet {
    fn filter_words(&mut self) -> Option<std::sync::Arc<crate::admission::Allocation<Vec<u64>>>> {
        match &self.words {
            BitSetWords::Shared(w) if self.current == TERMINATED && self.word_idx == 0 => Some(w.clone()),
            _ => None,
        }
    }
    fn advance(&mut self) -> DocId {
        let n = self.words.len();
        loop {
            if self.cur != 0 {
                let bit = self.cur.trailing_zeros();
                self.cur &= self.cur - 1;
                self.current = (self.word_idx as u32) * 64 + bit;
                return self.current;
            }
            self.word_idx += 1;
            if self.word_idx >= n {
                self.word_idx = n;
                self.current = TERMINATED;
                return TERMINATED;
            }
            self.cur = self.words[self.word_idx];
        }
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.current != TERMINATED && self.current >= target {
            return self.current;
        }
        let n = self.words.len();
        let tw = target as usize / 64;
        let mask = u64::MAX << (target % 64);
        if tw > self.word_idx {
            if tw >= n {
                self.word_idx = n;
                self.cur = 0;
                self.current = TERMINATED;
                return TERMINATED;
            }
            self.word_idx = tw;
            self.cur = self.words[tw] & mask;
        } else if tw == self.word_idx {
            self.cur &= mask;
        }
        self.advance()
    }

    fn doc(&self) -> DocId {
        self.current
    }

    fn size_hint(&self) -> u32 {
        self.count
    }

    fn fill_all(&mut self, deleted: Option<&[u64]>, out: &mut Vec<DocId>) -> bool {
        let n = self.words.len();
        if self.word_idx >= n {
            return true;
        }
        out.reserve(self.count as usize);
        let (mut wi, mut w) = (self.word_idx, self.cur);
        loop {
            let dead = deleted.and_then(|d| d.get(wi)).copied().unwrap_or(0);
            let mut x = w & !dead;
            while x != 0 {
                out.push((wi as u32) * 64 + x.trailing_zeros());
                x &= x - 1;
            }
            wi += 1;
            if wi >= n {
                break;
            }
            w = self.words[wi];
        }
        self.word_idx = n;
        self.cur = 0;
        self.current = TERMINATED;
        true
    }

    fn collect_first_k_bulk(&mut self, k: usize, deleted: Option<&[u64]>) -> Option<(Vec<DocId>, u64)> {
        if self.current != TERMINATED || self.word_idx != 0 || self.words.first().copied().unwrap_or(0) != self.cur {
            return None;
        }
        let mut first = Vec::with_capacity(k.min(self.count as usize));
        let mut total = 0u64;
        for (wi, &w0) in self.words.iter().enumerate() {
            let dead = deleted.and_then(|d| d.get(wi)).copied().unwrap_or(0);
            let mut w = w0 & !dead;
            total += w.count_ones() as u64;
            while first.len() < k && w != 0 {
                first.push((wi as u32) * 64 + w.trailing_zeros());
                w &= w - 1;
            }
        }
        self.word_idx = self.words.len();
        self.cur = 0;
        Some((first, total))
    }
}

/// An empty doc set.
pub struct EmptyDocSet;

impl DocSet for EmptyDocSet {
    fn advance(&mut self) -> DocId {
        TERMINATED
    }

    fn fill_all(&mut self, _deleted: Option<&[u64]>, _out: &mut Vec<DocId>) -> bool {
        true
    }

    fn doc(&self) -> DocId {
        TERMINATED
    }

    fn size_hint(&self) -> u32 {
        0
    }
}

/// The doc set `0..num_docs`.
pub struct AllDocSet {
    num_docs: u32,
    /// Next doc to emit; `num_docs` once exhausted.
    next: u32,
    current: DocId,
}

impl AllDocSet {
    /// All doc ids in `0..num_docs`.
    pub fn new(num_docs: u32) -> Self {
        Self {
            num_docs,
            next: 0,
            current: TERMINATED,
        }
    }
}

impl DocSet for AllDocSet {
    fn advance(&mut self) -> DocId {
        if self.next >= self.num_docs {
            self.current = TERMINATED;
        } else {
            self.current = self.next;
            self.next += 1;
        }
        self.current
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.current != TERMINATED && self.current >= target {
            return self.current;
        }
        self.next = self.next.max(target.min(self.num_docs));
        self.advance()
    }

    fn doc(&self) -> DocId {
        self.current
    }

    fn size_hint(&self) -> u32 {
        self.num_docs
    }

    fn fill_all(&mut self, deleted: Option<&[u64]>, out: &mut Vec<DocId>) -> bool {
        let start = if self.current == TERMINATED && self.next == 0 { 0 } else { self.next };
        let end = self.num_docs;
        if start < end {
            out.reserve((end - start) as usize);
            match deleted {
                None => out.extend(start..end),
                Some(d) => {
                    // Word at a time: the set bits of the complement.
                    let mut w = start as usize / 64;
                    while (w as u64) * 64 < end as u64 {
                        let base = (w * 64) as u32;
                        let mut live = !d.get(w).copied().unwrap_or(0);
                        if base < start { live &= u64::MAX << (start - base); }
                        if end - base < 64 { live &= (1u64 << (end - base)) - 1; }
                        while live != 0 {
                            out.push(base + live.trailing_zeros());
                            live &= live - 1;
                        }
                        w += 1;
                    }
                }
            }
        }
        self.next = end;
        self.current = TERMINATED;
        true
    }
}

impl<T: DocSet + ?Sized> DocSet for Box<T> {
    fn has_fast_exact_seek(&self) -> bool { (**self).has_fast_exact_seek() }

    fn exact_seek_reads_value(&self) -> bool { (**self).exact_seek_reads_value() }

    fn set_lead_cost(&mut self, lead_cost: u32) { (**self).set_lead_cost(lead_cost) }

    fn seek_exact(&mut self, target: DocId) -> bool { (**self).seek_exact(target) }

    fn advance(&mut self) -> DocId {
        (**self).advance()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        (**self).seek(target)
    }

    fn doc(&self) -> DocId {
        (**self).doc()
    }

    fn size_hint(&self) -> u32 {
        (**self).size_hint()
    }

    fn collect_first_k_bulk(&mut self, k: usize, deleted: Option<&[u64]>) -> Option<(Vec<DocId>, u64)> {
        (**self).collect_first_k_bulk(k, deleted)
    }

    fn fill_all(&mut self, deleted: Option<&[u64]>, out: &mut Vec<DocId>) -> bool {
        (**self).fill_all(deleted, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn multiples_of(n: u32, limit: u32) -> Vec<DocId> {
        (0..limit).step_by(n as usize).collect()
    }

    #[test]
    fn vec_docset_advance_and_seek() {
        let docs = multiples_of(3, 100);
        let mut s = VecDocSet::new(docs.clone());
        assert_eq!(s.doc(), TERMINATED);
        assert_eq!(s.size_hint(), docs.len() as u32);
        assert_eq!(s.advance(), 0);
        assert_eq!(s.doc(), 0);
        assert_eq!(s.advance(), 3);
        assert_eq!(s.seek(10), 12);
        assert_eq!(s.doc(), 12);
        assert_eq!(s.seek(12), 12);
        assert_eq!(s.seek(99), 99);
        assert_eq!(s.advance(), TERMINATED);
        assert_eq!(s.doc(), TERMINATED);
        assert_eq!(s.seek(5), TERMINATED);
        assert_eq!(s.advance(), TERMINATED);
    }

    #[test]
    fn vec_docset_seek_matches_brute_force() {
        let docs: Vec<DocId> = (0..10_000)
            .filter(|d| d % 7 == 0 || d % 11 == 0)
            .collect();
        let mut s = VecDocSet::new(docs.clone());
        for target in [0, 1, 5, 100, 4999, 9999, 10_000, 10_001] {
            let got = s.seek(target);
            let want = docs
                .iter()
                .copied()
                .find(|&d| d >= target)
                .unwrap_or(TERMINATED);
            assert_eq!(got, want, "seek({target})");
        }
        // Interleaved advance/seek.
        let mut s = VecDocSet::new(docs.clone());
        let mut i = 0usize;
        for target in [0, 3, 10, 50, 51, 200, 201, 9998, 9999, 10_000] {
            while i < docs.len() && docs[i] < target {
                i += 1;
            }
            let want = docs.get(i).copied().unwrap_or(TERMINATED);
            assert_eq!(s.seek(target), want, "seek({target})");
            if want == TERMINATED {
                assert_eq!(s.advance(), TERMINATED);
                break;
            }
            for _ in 0..3 {
                i += 1;
                let want = docs.get(i).copied().unwrap_or(TERMINATED);
                assert_eq!(s.advance(), want, "advance after seek({target})");
            }
        }
        // Seek to a target at or below the current doc is a no-op.
        let mut s = VecDocSet::new(docs.clone());
        assert_eq!(s.seek(14), 14);
        assert_eq!(s.seek(14), 14);
        assert_eq!(s.seek(3), 14);
        assert_eq!(s.advance(), 21);
    }

    #[test]
    fn bitset_docset_matches_vec() {
        let docs = multiples_of(5, 1000);
        let mut b = BitSetDocSet::from_docs(&docs);
        assert_eq!(b.len(), docs.len());
        assert_eq!(b.size_hint(), docs.len() as u32);
        assert_eq!(b.doc(), TERMINATED);
        for &want in &docs {
            assert_eq!(b.advance(), want);
        }
        assert_eq!(b.advance(), TERMINATED);
        assert_eq!(b.doc(), TERMINATED);

        let mut b = BitSetDocSet::from_docs(&docs);
        for target in [0, 1, 4, 5, 6, 63, 64, 65, 999, 1000] {
            let got = b.seek(target);
            let want = docs
                .iter()
                .copied()
                .find(|&d| d >= target)
                .unwrap_or(TERMINATED);
            assert_eq!(got, want, "seek({target})");
        }
        assert_eq!(b.advance(), TERMINATED);
        assert_eq!(b.seek(2000), TERMINATED);

        // Dense bits within one word, plus seek within the same word.
        let docs: Vec<DocId> = (0..200).filter(|d| d % 3 != 0).collect();
        let mut b = BitSetDocSet::from_docs(&docs);
        let mut i = 0usize;
        for target in [0u32, 1, 2, 10, 63, 64, 65, 127, 128, 190, 199] {
            while i < docs.len() && docs[i] < target {
                i += 1;
            }
            let want = docs.get(i).copied().unwrap_or(TERMINATED);
            assert_eq!(b.seek(target), want, "seek({target})");
            if want == TERMINATED {
                break;
            }
            i += 1;
            let want = docs.get(i).copied().unwrap_or(TERMINATED);
            assert_eq!(b.advance(), want, "advance after seek({target})");
        }

        let mut b = BitSetDocSet::from_docs(&[]);
        assert_eq!(b.advance(), TERMINATED);
        assert_eq!(b.seek(0), TERMINATED);
        assert!(b.is_empty());
    }

    #[test]
    fn empty_and_all() {
        let mut e = EmptyDocSet;
        assert_eq!(e.advance(), TERMINATED);
        assert_eq!(e.doc(), TERMINATED);
        assert_eq!(e.size_hint(), 0);

        let mut a = AllDocSet::new(5);
        assert_eq!(a.doc(), TERMINATED);
        for d in 0..5 {
            assert_eq!(a.advance(), d);
        }
        assert_eq!(a.advance(), TERMINATED);
        assert_eq!(a.seek(3), TERMINATED);

        let mut a = AllDocSet::new(10);
        assert_eq!(a.seek(7), 7);
        assert_eq!(a.advance(), 8);
        assert_eq!(a.seek(10), TERMINATED);
        assert_eq!(a.doc(), TERMINATED);

        let mut a = AllDocSet::new(0);
        assert_eq!(a.advance(), TERMINATED);
        assert_eq!(a.seek(0), TERMINATED);
    }

    #[test]
    fn box_docset_delegates() {
        let mut s: Box<dyn DocSet> = Box::new(VecDocSet::new(multiples_of(2, 20)));
        assert_eq!(s.advance(), 0);
        assert_eq!(s.seek(5), 6);
        assert_eq!(s.doc(), 6);
        assert_eq!(s.size_hint(), 10);
    }
}
