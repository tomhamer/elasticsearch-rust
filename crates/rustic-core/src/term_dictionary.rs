//! Term dictionary.
//!
//! The term dictionary maps terms to their postings list locations.
//! This is the hottest data structure in a search engine: every query
//! starts with a term lookup.
//!
//! Two representations:
//! - [`TermDictionary`]: an in-memory `BTreeMap` builder used while a
//!   segment is being written (and in benchmarks).
//! - [`crate::codec::terms::TermDictReader`]: the on-disk SSTable that
//!   sealed segments use. Creating an empty one is O(1), unlike Tantivy's
//!   `TermDictionary::empty()` (~5ms in debug mode, issue #766).
//!
//! [`MultiSegmentDictionary`] answers "which segments contain this term?"
//! across all segments of an index in one lookup.

use crate::codec::terms::{TermDictReader, TermDictWriter};
use crate::error::Result;
use crate::types::Term;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Build the dictionary key for a term: 2-byte field id + term bytes.
///
/// Measured note: a `SmallVec<[u8; 32]>` here was ~18% slower than `Vec`
/// on the lookup benchmark (the inline/spilled branch costs more than the
/// small allocation).
#[inline]
fn term_key(term: &Term) -> Vec<u8> {
    term.key()
}

/// BM25 IDF: ln(1 + (N - df + 0.5) / (df + 0.5)). Zero when df == 0.
#[inline]
pub fn bm25_idf(doc_freq: u32, num_docs: u32) -> f32 {
    if doc_freq == 0 {
        return 0.0;
    }
    let n = num_docs as f32;
    let df = doc_freq as f32;
    (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
}

/// [`PostingsLocation::offset`] bit of a single-doc list kept in the term
/// dictionary itself (format v15+, see
/// [`crate::codec::INLINE_POSTINGS_SINCE`]): the low bits are the doc id;
/// its one posting has term frequency 1 and no positions.
pub const INLINE_POSTINGS: u64 = 1 << 63;

/// [`PostingsLocation::offset`] bit of an inline list that also carries its
/// one position (format [`crate::codec::INLINE_POSITIONS_SINCE`]+): bits
/// 32..62 hold it ([`inline_position`]), the low 32 bits the doc.
pub const INLINE_POSITION: u64 = 1 << 62;

/// Largest position an inline list carries.
pub const INLINE_MAX_POSITION: u32 = (1 << 30) - 1;

/// The doc of an inline list's offset.
#[inline]
pub fn inline_doc(offset: u64) -> crate::types::DocId {
    offset as u32
}

/// The position an inline list's offset carries, if any.
#[inline]
pub fn inline_position(offset: u64) -> Option<u32> {
    (offset & INLINE_POSITION != 0).then(|| ((offset >> 32) as u32) & INLINE_MAX_POSITION)
}

/// The inline offset of doc `doc` (with `position`, for a positional field).
#[inline]
pub fn inline_offset(doc: crate::types::DocId, position: Option<u32>) -> u64 {
    match position {
        Some(p) => INLINE_POSTINGS | INLINE_POSITION | ((p as u64) << 32) | doc as u64,
        None => INLINE_POSTINGS | doc as u64,
    }
}

/// A postings list location within a segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PostingsLocation {
    /// Offset into the postings file.
    pub offset: u64,
    /// Number of documents containing this term.
    pub doc_freq: u32,
    /// Total term frequency across all documents.
    pub total_term_freq: u64,
}

/// A term dictionary entry.
#[derive(Clone, Debug)]
pub struct TermEntry {
    pub term: Term,
    pub location: PostingsLocation,
}

/// The term dictionary for a single segment.
///
/// Uses a BTreeMap internally (ordered, supports range iteration).
pub struct TermDictionary {
    /// Terms sorted lexicographically.
    terms: BTreeMap<Vec<u8>, PostingsLocation>,
    /// Number of terms in the dictionary.
    num_terms: usize,
    /// Total number of documents in the segment.
    num_docs: u32,
}

impl TermDictionary {
    /// Create an empty term dictionary.
    ///
    /// **Performance note**: This is O(1), unlike Tantivy's
    /// `TermDictionary::empty()` which takes ~5ms in debug mode.
    pub fn new(num_docs: u32) -> Self {
        Self {
            terms: BTreeMap::new(),
            num_terms: 0,
            num_docs,
        }
    }

    /// Insert a term into the dictionary.
    pub fn insert(&mut self, term: &Term, location: PostingsLocation) {
        if self.terms.insert(term_key(term), location).is_none() {
            self.num_terms += 1;
        }
    }

    /// Look up a term in the dictionary.
    ///
    /// **Target**: <100ns per lookup (Tantivy: ~5ms cold, ~100ns hot).
    ///
    /// The BTreeMap gives us O(log n) lookup. For a segment with
    /// 1M terms, that's ~20 comparisons. With SIMD-accelerated
    /// byte comparison, each comparison processes 16 bytes at a time.
    pub fn get(&self, term: &Term) -> Option<&PostingsLocation> {
        self.terms.get(term_key(term).as_slice())
    }

    /// Check if a term exists in the dictionary.
    pub fn contains(&self, term: &Term) -> bool {
        self.terms.contains_key(term_key(term).as_slice())
    }

    /// Iterate over all terms in the dictionary.
    pub fn iter(&self) -> impl Iterator<Item = (&Vec<u8>, &PostingsLocation)> {
        self.terms.iter()
    }

    /// Iterate over terms of `field` whose text is in [lower, upper).
    ///
    /// `None` for a bound means "unbounded within this field".
    /// This is used for range queries on keyword fields.
    /// The BTreeMap gives us O(log n + k) where k is the number
    /// of terms in the range.
    pub fn range_iter(
        &self,
        field: u16,
        lower: Option<&[u8]>,
        upper: Option<&[u8]>,
    ) -> impl Iterator<Item = (&Vec<u8>, &PostingsLocation)> {
        use std::ops::Bound;

        let prefix = field.to_le_bytes();
        let mut lo = prefix.to_vec();
        if let Some(l) = lower {
            lo.extend_from_slice(l);
        }
        let lo_bound = Bound::Included(lo);

        let hi_bound = match upper {
            Some(u) => {
                let mut hi = prefix.to_vec();
                hi.extend_from_slice(u);
                Bound::Excluded(hi)
            }
            // Everything in this field: stop at the next field's prefix.
            None => match crate::types::Term::field_key_end(field) {
                Some(end) => Bound::Excluded(end),
                None => Bound::Unbounded,
            },
        };

        self.terms.range::<Vec<u8>, _>((lo_bound, hi_bound))
    }

    /// Get the number of terms in the dictionary.
    pub fn num_terms(&self) -> usize {
        self.num_terms
    }

    /// Get the number of documents in the segment.
    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// Get the document frequency for a term.
    pub fn doc_freq(&self, term: &Term) -> u32 {
        self.get(term).map(|l| l.doc_freq).unwrap_or(0)
    }

    /// Get the total term frequency for a term.
    pub fn total_term_freq(&self, term: &Term) -> u64 {
        self.get(term).map(|l| l.total_term_freq).unwrap_or(0)
    }

    /// Compute the IDF (inverse document frequency) for a term.
    ///
    /// Uses the standard BM25 IDF formula:
    /// idf = ln(1 + (N - df + 0.5) / (df + 0.5))
    pub fn idf(&self, term: &Term) -> f32 {
        bm25_idf(self.doc_freq(term), self.num_docs)
    }

    /// Encode as an on-disk SSTable (see `codec::terms`).
    pub fn encode(&self) -> Vec<u8> {
        let mut w = TermDictWriter::new();
        for (key, loc) in &self.terms {
            w.add(key, loc);
        }
        w.finish(self.num_docs)
    }

    /// Encode and reopen as a reader.
    pub fn into_reader(&self) -> Result<TermDictReader> {
        TermDictReader::open(crate::codec::directory::FileSlice::from_vec(self.encode()))
    }
}

impl Default for TermDictionary {
    fn default() -> Self {
        Self::new(0)
    }
}

/// A multi-segment term dictionary that can search across segments.
///
/// This is the key to fixing Tantivy's compound query performance:
/// instead of N separate dictionary lookups (one per segment), we
/// maintain a merged view that can answer "which segments contain
/// this term?" in a single pass.
#[derive(Clone)]
pub struct MultiSegmentDictionary {
    /// Per-segment dictionaries.
    segments: Vec<Arc<TermDictReader>>,
    /// Merged term → segment_ids mapping (built lazily).
    merged: Option<BTreeMap<Vec<u8>, Vec<u32>>>,
}

impl MultiSegmentDictionary {
    pub fn new(segments: Vec<Arc<TermDictReader>>) -> Self {
        Self {
            segments,
            merged: None,
        }
    }

    /// Get the number of segments.
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Look up a term across all segments.
    ///
    /// Returns the list of (segment_id, location) pairs.
    pub fn get(&self, term: &Term) -> Vec<(u32, PostingsLocation)> {
        let mut results = Vec::new();
        for (seg_id, dict) in self.segments.iter().enumerate() {
            if let Some(loc) = dict.get(term) {
                results.push((seg_id as u32, loc));
            }
        }
        results
    }

    /// Check if a term exists in any segment.
    ///
    /// Uses the merged index when it has been built (one lookup), and
    /// falls back to probing each segment otherwise.
    pub fn contains(&self, term: &Term) -> bool {
        match &self.merged {
            Some(merged) => merged.contains_key(term_key(term).as_slice()),
            None => self.segments.iter().any(|d| d.contains(term)),
        }
    }

    /// Get the total document frequency across all segments.
    pub fn total_doc_freq(&self, term: &Term) -> u32 {
        self.segments
            .iter()
            .map(|d| d.doc_freq(term))
            .sum()
    }

    /// Get the total number of documents across all segments.
    pub fn total_num_docs(&self) -> u32 {
        self.segments.iter().map(|d| d.num_docs()).sum()
    }

    /// Build the merged term → segment_ids mapping.
    ///
    /// This is a one-time O(total_terms) cost that enables fast
    /// "which segments have this term?" queries.
    pub fn build_merged(&mut self) {
        if self.merged.is_some() {
            return;
        }
        let mut merged: BTreeMap<Vec<u8>, Vec<u32>> = BTreeMap::new();
        for (seg_id, dict) in self.segments.iter().enumerate() {
            for (key, _) in dict.iter_all().unwrap_or_default() {
                merged.entry(key).or_default().push(seg_id as u32);
            }
        }
        self.merged = Some(merged);
    }

    /// Get the segment IDs that contain a term (using merged index).
    pub fn segments_for_term(&self, term: &Term) -> Option<&Vec<u32>> {
        self.merged.as_ref()?.get(term_key(term).as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_dictionary_is_fast() {
        // This is the key test: creating an empty dictionary must be O(1).
        // Tantivy's takes ~5ms in debug mode.
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = TermDictionary::new(0);
        }
        let elapsed = start.elapsed();
        // Should be well under 1ms total for 1000 empty dicts.
        assert!(
            elapsed.as_micros() < 1000,
            "Creating 1000 empty dictionaries took {:?} (should be < 1ms)",
            elapsed
        );
    }

    #[test]
    fn test_insert_and_lookup() {
        let mut dict = TermDictionary::new(1000);
        let term = Term::new_text(0, "hello");
        dict.insert(
            &term,
            PostingsLocation {
                offset: 0,
                doc_freq: 10,
                total_term_freq: 25,
            },
        );

        assert!(dict.contains(&term));
        let loc = dict.get(&term).unwrap();
        assert_eq!(loc.doc_freq, 10);
        assert_eq!(loc.total_term_freq, 25);
    }

    #[test]
    fn test_range_iter() {
        let mut dict = TermDictionary::new(100);
        for word in ["apple", "banana", "cherry", "date", "elderberry"] {
            let term = Term::new_text(0, word);
            dict.insert(
                &term,
                PostingsLocation {
                    offset: 0,
                    doc_freq: 1,
                    total_term_freq: 1,
                },
            );
        }

        // Range [banana, date)
        let results: Vec<_> = dict
            .range_iter(0, Some(b"banana"), Some(b"date"))
            .map(|(k, _)| k)
            .collect();
        assert_eq!(results.len(), 2); // banana, cherry

        // Whole field
        assert_eq!(dict.range_iter(0, None, None).count(), 5);
        // Different field: nothing
        assert_eq!(dict.range_iter(1, None, None).count(), 0);
    }

    #[test]
    fn test_idf() {
        let mut dict = TermDictionary::new(1000);
        let term = Term::new_text(0, "common");
        dict.insert(
            &term,
            PostingsLocation {
                offset: 0,
                doc_freq: 500,
                total_term_freq: 1000,
            },
        );

        let idf = dict.idf(&term);
        // idf = ln(1 + (1000 - 500 + 0.5) / (500 + 0.5))
        //     = ln(1 + 500.5/500.5)
        //     = ln(2) ≈ 0.693
        assert!((idf - 0.693).abs() < 0.01);
    }

    #[test]
    fn test_encode_and_reopen() {
        let mut dict = TermDictionary::new(100);
        for word in ["hello", "world", "rust"] {
            let term = Term::new_text(0, word);
            dict.insert(
                &term,
                PostingsLocation {
                    offset: word.len() as u64,
                    doc_freq: 1,
                    total_term_freq: 1,
                },
            );
        }

        let restored = dict.into_reader().unwrap();
        assert_eq!(restored.num_terms(), 3);
        assert_eq!(restored.num_docs(), 100);
        assert!(restored.contains(&Term::new_text(0, "hello")));
        assert_eq!(restored.get(&Term::new_text(0, "world")).unwrap().offset, 5);
    }

    #[test]
    fn test_multi_segment_lookup() {
        let mut d1 = TermDictionary::new(100);
        let mut d2 = TermDictionary::new(200);
        let term = Term::new_text(0, "test");
        d1.insert(
            &term,
            PostingsLocation {
                offset: 0,
                doc_freq: 5,
                total_term_freq: 10,
            },
        );
        d2.insert(
            &term,
            PostingsLocation {
                offset: 100,
                doc_freq: 15,
                total_term_freq: 30,
            },
        );

        let mut multi = MultiSegmentDictionary::new(vec![
            Arc::new(d1.into_reader().unwrap()),
            Arc::new(d2.into_reader().unwrap()),
        ]);
        let results = multi.get(&term);
        assert_eq!(results.len(), 2);
        assert_eq!(multi.total_doc_freq(&term), 20);
        assert_eq!(multi.total_num_docs(), 300);
        assert!(multi.contains(&term));
        multi.build_merged();
        assert!(multi.contains(&term));
        assert!(!multi.contains(&Term::new_text(0, "nope")));
        assert_eq!(multi.segments_for_term(&term), Some(&vec![0, 1]));
    }
}
