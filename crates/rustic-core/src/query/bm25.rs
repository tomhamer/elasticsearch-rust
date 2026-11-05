//! BM25 scoring (Lucene's `BM25Similarity`).

use std::sync::atomic::{AtomicBool, Ordering};

static LEGACY_K1_BOOST: AtomicBool = AtomicBool::new(false);

/// Multiply every BM25 weight by `(k1 + 1)` (Lucene's `LegacyBM25Similarity`,
/// which Elasticsearch's index similarity uses: a one-term, average-length,
/// `tf = 1` match scores exactly its idf). Off by default for modern Lucene;
/// the Elasticsearch layer turns it on. Percolation MemoryIndex searchers
/// explicitly select modern Lucene weights instead.
pub fn set_legacy_k1_boost(on: bool) {
    LEGACY_K1_BOOST.store(on, Ordering::Relaxed);
}

#[inline(always)]
fn k1_factor(k1: f32) -> f32 {
    if LEGACY_K1_BOOST.load(Ordering::Relaxed) {
        k1 + 1.0
    } else {
        1.0
    }
}

/// BM25 parameters.
#[derive(Debug, Clone, Copy)]
pub struct Bm25Params {
    /// Term-frequency saturation parameter.
    pub k1: f32,
    /// Field-length normalization parameter.
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// BM25 inverse document frequency: `ln(1 + (N - df + 0.5) / (df + 0.5))`,
/// evaluated in `f64` and rounded to `f32` as Lucene's `BM25Similarity.idf`.
pub fn idf(doc_freq: u64, num_docs: u64) -> f32 {
    let df = doc_freq as f64;
    let n = num_docs as f64;
    (1.0 + (n - df + 0.5) / (df + 0.5)).ln() as f32
}

// ---------------------------------------------------------------------------
// Norms: Lucene `SmallFloat.intToByte4` / `byte4ToInt` (identical to
// Tantivy's `FIELD_NORMS_TABLE`). Lengths 0..=39 are exact, then 4
// significant bits. Field lengths are quantized to one byte at index time,
// as Lucene/ES/Tantivy do, so scores match them bit for bit.
// ---------------------------------------------------------------------------

fn long_to_int4(i: u64) -> u32 {
    let num_bits = 64 - i.leading_zeros();
    if num_bits < 4 {
        return i as u32;
    }
    let shift = num_bits - 4;
    let mut encoded = ((i >> shift) as u32) & 0x07;
    encoded |= (shift + 1) << 3;
    encoded
}

fn int4_to_long(i: u32) -> u64 {
    let bits = (i & 0x07) as u64;
    let shift = (i >> 3) as i32 - 1;
    if shift == -1 { bits } else { (bits | 0x08) << shift }
}

/// `255 - long_to_int4(i32::MAX)`.
const NUM_FREE_VALUES: u32 = 24;

/// Encode a field length into one norm byte (monotone, lossy above 39).
pub fn int_to_byte4(i: u32) -> u8 {
    if i < NUM_FREE_VALUES {
        i as u8
    } else {
        (NUM_FREE_VALUES + long_to_int4((i - NUM_FREE_VALUES) as u64)) as u8
    }
}

/// Decode a norm byte to the (lower bound of the) field length.
pub fn byte4_to_int(b: u8) -> u32 {
    let i = b as u32;
    if i < NUM_FREE_VALUES {
        i
    } else {
        (NUM_FREE_VALUES as u64 + int4_to_long(i - NUM_FREE_VALUES)).min(u32::MAX as u64) as u32
    }
}

/// Decoded length of every norm byte.
pub fn norm_table() -> &'static [u32; 256] {
    static T: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, v) in t.iter_mut().enumerate() {
            *v = byte4_to_int(i as u8);
        }
        t
    })
}

/// The per-norm-byte inverse BM25 length factor
/// `1 / (k1 * (1 - b + b * len / avg))` (Lucene's `BM25Similarity` cache).
/// Depends only on `(k1, b, avg)`, so tables are shared through a small
/// per-thread cache instead of being rebuilt for every term of every query.
fn length_factors(k1: f32, b: f32, avg: f32) -> Arc<[f32; 256]> {
    thread_local! {
        static CACHE: std::cell::RefCell<Vec<((u32, u32, u32), Arc<[f32; 256]>)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    let key = (k1.to_bits(), b.to_bits(), avg.to_bits());
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if let Some(i) = c.iter().position(|(k, _)| *k == key) {
            return c[i].1.clone();
        }
        let lens = norm_table();
        let mut t = [0f32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            *slot = 1.0 / (k1 * ((1.0 - b) + b * (lens[i] as f32) / avg));
        }
        let t = Arc::new(t);
        if c.len() >= 16 {
            c.remove(0);
        }
        c.push((key, t.clone()));
        t
    })
}

use std::sync::Arc;

/// Precomputed BM25 weight for one term in one field (Lucene's
/// `BM25Scorer`, as used by Elasticsearch >= 7: no legacy `(k1 + 1)`
/// numerator factor).
///
/// `score = weight - weight / (1 + tf * inv)` with `weight = boost * idf`
/// and `inv = 1 / (k1 * (1 - b + b * len / avg_len))`, i.e.
/// `weight * tf / (tf + k1 * (1 - b + b * len / avg_len))`; `inv` is cached
/// per norm byte. The evaluation order is Lucene's, so scores match
/// Lucene/ES bit for bit.
#[derive(Clone)]
pub struct Bm25Weight {
    /// `idf * boost`, the factor applied on the hot path.
    weight: f32,
    k1_boost: f32,
    boost: f32,
    idf: f32,
    doc_freq: u64,
    num_docs: u64,
    k1: f32,
    b: f32,
    avg_field_len: f32,
    cache: Arc<[f32; 256]>,
}

impl Bm25Weight {
    /// Build a weight for a term with the given document frequency over
    /// `num_docs` docs, using the given BM25 parameters and average field
    /// length.
    pub fn new(doc_freq: u64, num_docs: u64, params: Bm25Params, avg_field_len: f32) -> Self {
        let avg = if avg_field_len > 0.0 { avg_field_len } else { 1.0 };
        let idf_v = idf(doc_freq, num_docs);
        Self {
            weight: k1_factor(params.k1) * idf_v,
            k1_boost: k1_factor(params.k1),
            boost: 1.0,
            idf: idf_v,
            doc_freq,
            num_docs,
            k1: params.k1,
            b: params.b,
            avg_field_len: avg,
            cache: length_factors(params.k1, params.b, avg),
        }
    }

    /// Lucene's modern BM25, independent of the process-wide legacy default.
    /// MemoryIndex searchers (including percolation) use this similarity even
    /// when the surrounding index retains legacy Elasticsearch scoring.
    pub fn new_lucene(doc_freq: u64, num_docs: u64, params: Bm25Params, avg_field_len: f32) -> Self {
        let mut weight = Self::new(doc_freq, num_docs, params, avg_field_len);
        weight.k1_boost = 1.0;
        weight.weight = weight.idf;
        weight
    }

    /// BM25 score for a term frequency `tf` in a field of (decoded) length
    /// `field_len`. Equal to [`score_norm`](Self::score_norm) when
    /// `field_len == byte4_to_int(norm)`.
    #[inline]
    pub fn score(&self, tf: u32, field_len: u32) -> f32 {
        let inv = 1.0 / (self.k1 * ((1.0 - self.b) + self.b * (field_len as f32) / self.avg_field_len));
        Self::combine(self.weight, tf as f32, inv)
    }

    /// Lucene's `BM25Scorer.doScore`: `weight - weight / (1 + tf * inv)`.
    #[inline(always)]
    pub fn combine(weight: f32, tf: f32, inv: f32) -> f32 {
        weight - weight / (1.0 + tf * inv)
    }

    /// BM25 score for `tf` in a doc with norm byte `norm`.
    #[inline(always)]
    pub fn score_norm(&self, tf: u32, norm: u8) -> f32 {
        Self::combine(self.weight, tf as f32, self.cache[norm as usize])
    }

    /// Upper bound of any score in a block with `max_tf` and `min_norm`
    /// (the score grows with tf and shrinks with the norm).
    #[inline]
    pub fn bound(&self, max_tf: u32, min_norm: u8) -> f32 {
        if max_tf == u32::MAX {
            return self.weight;
        }
        self.score_norm(max_tf, min_norm)
    }

    /// Upper bound of any score in a postings block: the best corner of
    /// its impact staircase when it has one, else `bound(max_tf, min_norm)`.
    #[inline]
    pub fn block_bound(&self, s: &crate::codec::postings::SkipEntry) -> f32 {
        let [n1, t1, n2, t2] = s.impacts;
        if n1 == 0 || s.max_tf == u32::MAX {
            return self.bound(s.max_tf, s.min_norm);
        }
        let t = |x: u8| if x == 255 { s.max_tf } else { (x as u32).min(s.max_tf) };
        self.score_norm(t(t1), s.min_norm).max(self.score_norm(t(t2), n1)).max(self.score_norm(s.max_tf, n2))
    }

    /// Upper bound over all docs (`tf -> inf`).
    #[inline]
    pub fn max_score(&self) -> f32 {
        self.weight
    }

    /// Whether scores grow with tf and shrink with length, as every block,
    /// window and staircase bound assumes: Lucene's `BM25Similarity` ranges
    /// (finite `k1 >= 0`, `b` in `[0, 1]`), which index settings enforce.
    pub fn monotonic(&self) -> bool {
        self.k1.is_finite() && self.k1 >= 0.0 && (0.0..=1.0).contains(&self.b)
    }

    /// `idf * boost`.
    #[inline(always)]
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Inverse length factor `1 / (k1 * (1 - b + b * len / avg))` per norm byte.
    #[inline(always)]
    pub fn norm_cache(&self) -> &[f32; 256] {
        &self.cache
    }

    /// Multiply every score by `boost`.
    pub fn with_boost(mut self, boost: f32) -> Self {
        // Lucene scores with `weight = (boost * (k1 + 1)) * idf` (the legacy
        // factor rides on the boost), so the products associate that way.
        self.weight = (boost * self.k1_boost) * self.idf;
        self.boost = boost;
        self
    }

    /// Replace the idf by `idf` (a multi-term construct's summed idf).
    pub fn with_idf(mut self, idf: f32) -> Self {
        self.idf = idf;
        self.weight = (self.boost * self.k1_boost) * idf;
        self
    }

    /// The idf folded into this weight.
    pub fn idf_value(&self) -> f32 {
        self.idf
    }

    /// Record the term statistics this weight was built from (for
    /// explanations; scoring is unaffected).
    pub fn with_stats(mut self, stats: crate::query::reader::TermStats) -> Self {
        self.doc_freq = stats.doc_freq;
        self.num_docs = stats.num_docs;
        self
    }

    /// The `(k1 + 1)` factor folded into this weight (1 unless
    /// [`set_legacy_k1_boost`] is on).
    pub fn k1_boost(&self) -> f32 {
        self.k1_boost
    }

    /// The query boost folded into this weight.
    pub fn boost(&self) -> f32 {
        self.boost
    }

    /// `(doc_freq, num_docs)` this weight's IDF was computed from.
    pub fn stats(&self) -> (u64, u64) {
        (self.doc_freq, self.num_docs)
    }

    /// `(k1, b, avg_field_len)`.
    pub fn params_avg(&self) -> (f32, f32, f32) {
        (self.k1, self.b, self.avg_field_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idf_values() {
        // df == N: ln(1 + 0.5/(N + 0.5))
        assert!((idf(10, 10) - (1.0f32 + 0.5 / 10.5).ln()).abs() < 1e-6);
        // df == 0: ln(1 + (N + 0.5)/0.5)
        assert!((idf(0, 10) - (1.0f32 + 10.5 / 0.5).ln()).abs() < 1e-6);
        // Rarer terms score higher.
        assert!(idf(1, 1000) > idf(100, 1000));
    }

    #[test]
    fn score_at_average_length() {
        let w = Bm25Weight::new(10, 1000, Bm25Params::default(), 10.0);
        let idf = idf(10, 1000);
        for tf in [0u32, 1, 2, 5, 100] {
            let want = idf * (tf as f32) / (tf as f32 + 1.2);
            let got = w.score(tf, 10);
            assert!((got - want).abs() < 1e-5, "tf={tf}: {got} vs {want}");
        }
    }

    #[test]
    fn score_length_normalization() {
        let w = Bm25Weight::new(10, 1000, Bm25Params::default(), 10.0);
        // Longer-than-average fields score lower, shorter score higher.
        assert!(w.score(5, 20) < w.score(5, 10));
        assert!(w.score(5, 10) < w.score(5, 1));
        // Cache and direct path agree for long fields.
        let direct = |len: u32| {
            let norm = 1.2 * (1.0 - 0.75 + 0.75 * (len as f32) / 10.0);
            idf(10, 1000) * 5.0 / (5.0 + norm)
        };
        for len in [255u32, 256, 257, 1000] {
            assert!((w.score(5, len) - direct(len)).abs() < 1e-5, "len={len}");
        }
    }

    #[test]
    fn smallfloat_matches_lucene_table() {
        // Lucene/Tantivy: 0..=40 exact, then 42, 44, ...
        for i in 0..=40u32 {
            assert_eq!(byte4_to_int(int_to_byte4(i)), i);
        }
        assert_eq!(byte4_to_int(41), 42);
        for l in 0..100_000u32 {
            let b = int_to_byte4(l);
            assert!(byte4_to_int(b) <= l);
            if b < 255 {
                assert!(byte4_to_int(b + 1) > l);
            }
        }
    }

    #[test]
    fn norm_byte_score_equals_decoded_length_score() {
        let w = Bm25Weight::new(10, 1000, Bm25Params::default(), 37.5).with_boost(1.7);
        for nb in 0..=255u8 {
            for tf in [1u32, 2, 9] {
                assert_eq!(w.score_norm(tf, nb).to_bits(), w.score(tf, byte4_to_int(nb)).to_bits());
            }
        }
    }

    #[test]
    fn matches_lucene_bm25_scorer() {
        // Lucene BM25Similarity (k1 1.2, b 0.75): weight = boost * idf,
        // score = weight - weight / (1 + freq / (k1 * (1 - b + b * dl / avgdl))).
        let w = Bm25Weight::new(3, 50, Bm25Params::default(), 7.25).with_boost(2.0);
        let idf = (1.0f64 + (50.0 - 3.0 + 0.5) / (3.0 + 0.5)).ln() as f32;
        let weight = 2.0f32 * idf;
        for nb in [0u8, 1, 7, 40, 120, 255] {
            let dl = byte4_to_int(nb) as f32;
            let inv = 1.0f32 / (1.2 * ((1.0 - 0.75) + 0.75 * dl / 7.25));
            for tf in [1u32, 3, 17] {
                let want = weight - weight / (1.0 + tf as f32 * inv);
                assert_eq!(w.score_norm(tf, nb).to_bits(), want.to_bits(), "nb={nb} tf={tf}");
            }
        }
        // No legacy (k1 + 1) factor: the score saturates at boost * idf.
        assert!(w.score_norm(1_000_000, 1) <= weight);
        assert_eq!(w.max_score(), weight);
    }

    #[test]
    fn zero_tf_scores_zero() {
        let w = Bm25Weight::new(1, 100, Bm25Params::default(), 5.0);
        assert_eq!(w.score(0, 5), 0.0);
    }
}
