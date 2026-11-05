//! Scoring models (Lucene's `Similarity` hierarchy).
//!
//! A [`Similarity`] turns index-wide statistics for one term into a
//! [`SimWeight`], which scores a `(term frequency, field length)` pair for
//! each matching doc. Implementations:
//!
//! - [`Bm25`]: Okapi BM25 (Lucene's `BM25Similarity`, the default);
//! - [`Classic`]: TF-IDF (Lucene's `ClassicSimilarity`);
//! - [`BooleanSim`]: every match scores the query boost;
//! - [`DfrSimilarity`]: divergence from randomness (basic model, after
//!   effect, normalization);
//! - [`IbSimilarity`]: information-based (distribution, lambda, normalization);
//! - [`LmDirichlet`], [`LmJelinekMercer`]: language models.
//!
//! Multi-term constructs (phrases, spans) sum the per-term weights, as
//! Lucene's `SimilarityBase.scorer` does for several `TermStatistics`.

use crate::query::bm25::{idf as bm25_idf, Bm25Params, Bm25Weight};
use crate::query::explain::Explanation;
use crate::query::reader::{FieldStats, TermStats};
use std::fmt;
use std::sync::Arc;

/// A per-term scoring function (Lucene's `SimScorer`).
pub trait SimWeight: Send + Sync {
    /// Score of a doc with term frequency `tf` and field length `field_len`
    /// (the decoded 1-byte norm, as Lucene scores).
    fn score(&self, tf: u32, field_len: u32) -> f32;

    /// Score of a doc with term frequency `tf` and norm byte `norm`
    /// (`SmallFloat`-encoded length). BM25 overrides this with a 256-entry
    /// table lookup.
    #[inline]
    fn score_norm(&self, tf: u32, norm: u8) -> f32 {
        self.score(tf, crate::query::bm25::byte4_to_int(norm))
    }

    /// Explanation tree for [`SimWeight::score`] on the same inputs.
    fn explain(&self, tf: u32, field_len: u32) -> Explanation;
}

impl SimWeight for Box<dyn SimWeight> {
    #[inline]
    fn score(&self, tf: u32, field_len: u32) -> f32 {
        (**self).score(tf, field_len)
    }
    #[inline]
    fn score_norm(&self, tf: u32, norm: u8) -> f32 {
        (**self).score_norm(tf, norm)
    }
    fn explain(&self, tf: u32, field_len: u32) -> Explanation {
        (**self).explain(tf, field_len)
    }
}

/// A scoring model (Lucene's `Similarity`).
pub trait Similarity: Send + Sync + fmt::Debug {
    /// Weight for one term with index-wide `stats`, in a field with `field`
    /// statistics, multiplied by `boost`.
    fn weight(&self, stats: TermStats, field: FieldStats, boost: f32) -> Box<dyn SimWeight>;

    /// Weight for a multi-term construct (phrase, span): the sum of the
    /// per-term weights, each scored with the construct's frequency.
    fn phrase_weight(&self, stats: &[TermStats], field: FieldStats, boost: f32) -> Box<dyn SimWeight> {
        match stats.len() {
            0 => self.weight(TermStats::default(), field, boost),
            1 => self.weight(stats[0], field, boost),
            _ => Box::new(SumWeight {
                parts: stats.iter().map(|s| self.weight(*s, field, boost)).collect(),
            }),
        }
    }

    /// Short name used in explanations.
    fn name(&self) -> String {
        format!("{self:?}")
    }

    /// `Some` when this is plain BM25, letting term scorers use the
    /// statically dispatched [`Bm25Weight`] on the hot path.
    fn as_bm25(&self) -> Option<Bm25> {
        None
    }
}

impl<T: Similarity + ?Sized> Similarity for Arc<T> {
    fn weight(&self, stats: TermStats, field: FieldStats, boost: f32) -> Box<dyn SimWeight> {
        (**self).weight(stats, field, boost)
    }
    fn phrase_weight(&self, stats: &[TermStats], field: FieldStats, boost: f32) -> Box<dyn SimWeight> {
        (**self).phrase_weight(stats, field, boost)
    }
    fn name(&self) -> String {
        (**self).name()
    }
    fn as_bm25(&self) -> Option<Bm25> {
        (**self).as_bm25()
    }
}

/// The default similarity: BM25 with Lucene's parameters.
pub fn default_similarity() -> Arc<dyn Similarity> {
    Arc::new(Bm25::default())
}

/// Sum of several weights (multi-term constructs).
pub struct SumWeight {
    parts: Vec<Box<dyn SimWeight>>,
}

impl SimWeight for SumWeight {
    fn score(&self, tf: u32, field_len: u32) -> f32 {
        self.parts.iter().map(|p| p.score(tf, field_len)).sum()
    }
    fn score_norm(&self, tf: u32, norm: u8) -> f32 {
        self.parts.iter().map(|p| p.score_norm(tf, norm)).sum()
    }
    fn explain(&self, tf: u32, field_len: u32) -> Explanation {
        let details: Vec<Explanation> = self.parts.iter().map(|p| p.explain(tf, field_len)).collect();
        let value = details.iter().map(|d| d.value).sum();
        Explanation::new(value, "sum of per-term scores", details)
    }
}

/// Average field length from `FieldStats`, or `fallback` when unknown.
fn avg_len(field: FieldStats, fallback: f32) -> f32 {
    if field.num_docs > 0 {
        field.avg_field_len()
    } else {
        fallback.max(1.0)
    }
}

fn log2(x: f64) -> f64 {
    x.ln() / std::f64::consts::LN_2
}

// ---------------------------------------------------------------------------
// BM25
// ---------------------------------------------------------------------------

/// Okapi BM25 (Lucene's `BM25Similarity`).
///
/// `score = boost * idf * tf / (tf + k1 * (1 - b + b * len / avg_len))`
/// with `idf = ln(1 + (N - df + 0.5) / (df + 0.5))` (Lucene >= 8 and
/// Elasticsearch >= 7: no legacy `(k1 + 1)` factor).
///
/// `discount_overlaps` is Lucene's indexing-time option (overlap tokens do
/// not count towards the field length); norms here are produced by the
/// indexer, so the flag is carried for configuration fidelity only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bm25 {
    /// Term-frequency saturation.
    pub k1: f32,
    /// Length normalization.
    pub b: f32,
    /// Whether overlap tokens are discounted from the field length.
    pub discount_overlaps: bool,
}

impl Default for Bm25 {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75, discount_overlaps: true }
    }
}

impl Bm25 {
    /// BM25 with explicit `k1`/`b`.
    pub fn new(k1: f32, b: f32) -> Self {
        Self { k1, b, discount_overlaps: true }
    }

    /// From the legacy parameter struct.
    pub fn from_params(p: Bm25Params) -> Self {
        Self::new(p.k1, p.b)
    }

    /// As the legacy parameter struct.
    pub fn params(&self) -> Bm25Params {
        Bm25Params { k1: self.k1, b: self.b }
    }
}

/// Modern Lucene BM25 for isolated MemoryIndex searchers. Intentionally does
/// not expose the legacy BM25 fast path, which uses the process-wide default.
#[derive(Clone, Copy, Debug, Default)]
pub struct LuceneBm25;

impl Similarity for LuceneBm25 {
    fn weight(&self, stats: TermStats, field: FieldStats, boost: f32) -> Box<dyn SimWeight> {
        Box::new(Bm25Weight::new_lucene(stats.doc_freq, stats.num_docs.max(1), Bm25Params::default(), avg_len(field, 1.0))
            .with_boost(boost).with_stats(stats))
    }
}

impl Similarity for Bm25 {
    fn weight(&self, stats: TermStats, field: FieldStats, boost: f32) -> Box<dyn SimWeight> {
        let avg = avg_len(field, 1.0);
        Box::new(Bm25Weight::new(stats.doc_freq, stats.num_docs.max(1), self.params(), avg).with_boost(boost).with_stats(stats))
    }
    fn phrase_weight(&self, stats: &[TermStats], field: FieldStats, boost: f32) -> Box<dyn SimWeight> {
        match stats.len() {
            0 => self.weight(TermStats::default(), field, boost),
            1 => self.weight(stats[0], field, boost),
            _ => Box::new(Bm25PhraseWeight::new(self, stats, field, boost)),
        }
    }
    fn name(&self) -> String {
        "BM25".into()
    }
    fn as_bm25(&self) -> Option<Bm25> {
        Some(*self)
    }
}

impl SimWeight for Bm25Weight {
    #[inline]
    fn score(&self, tf: u32, field_len: u32) -> f32 {
        Bm25Weight::score(self, tf, field_len)
    }

    #[inline]
    fn score_norm(&self, tf: u32, norm: u8) -> f32 {
        Bm25Weight::score_norm(self, tf, norm)
    }

    fn explain(&self, tf: u32, field_len: u32) -> Explanation {
        let (df, n) = self.stats();
        let idf = bm25_idf(df, n.max(1));
        let idf_node = Explanation::new(
            idf,
            "idf, computed as ln(1 + (N - n + 0.5) / (n + 0.5))",
            vec![Explanation::leaf(df as f32, "n, number of documents containing term"), Explanation::leaf(n as f32, "N, total number of documents")],
        );
        self.explain_with_idf(tf, field_len, idf_node, format!("freq, occurrences of term within document"))
    }
}

impl Bm25Weight {
    /// Lucene's `BM25Scorer.explain`: `score(freq=..)` = boost * idf * tf,
    /// with `tf = 1 - 1 / (1 + freq * norm_inverse)` (the form the scorer
    /// evaluates) and `idf` given as an explanation node (one term, or the
    /// sum over a phrase's terms).
    pub fn explain_with_idf(&self, tf: u32, field_len: u32, idf_node: Explanation, freq_desc: String) -> Explanation {
        self.explain_freq(tf as f32, field_len, idf_node, freq_desc)
    }

    /// [`Bm25Weight::explain_with_idf`] for a fractional frequency
    /// (`combined_fields` weights its fields' frequencies).
    pub fn explain_freq(&self, freq: f32, field_len: u32, idf_node: Explanation, freq_desc: String) -> Explanation {
        let (k1, b, avg) = self.params_avg();
        let inv = 1.0 / (k1 * ((1.0 - b) + b * field_len as f32 / avg));
        let tf_part = 1.0 - 1.0 / (1.0 + freq * inv);
        // Lucene flags lengths it can only approximate (norm byte > 39).
        let dl = if field_len > 39 { "dl, length of field (approximate)" } else { "dl, length of field" };
        Explanation::new(
            Bm25Weight::combine(self.weight(), freq, inv),
            format!("score(freq={freq:?}), computed as boost * idf * tf"),
            vec![
                Explanation::leaf(self.boost() * self.k1_boost(), "boost"),
                idf_node,
                Explanation::new(
                    tf_part,
                    "tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl)) from:",
                    vec![
                        Explanation::leaf(freq, freq_desc),
                        Explanation::leaf(k1, "k1, term saturation parameter"),
                        Explanation::leaf(b, "b, length normalization parameter"),
                        Explanation::leaf(field_len as f32, dl),
                        Explanation::leaf(avg, "avgdl, average length of field"),
                    ],
                ),
            ],
        )
    }
}

/// BM25 over a multi-term construct (phrase, span): one weight whose idf is
/// the sum of the terms' idfs (Lucene `BM25Similarity.scorer` with several
/// `TermStatistics`), scored with the construct's frequency.
pub struct Bm25PhraseWeight {
    inner: Bm25Weight,
    terms: Vec<(u64, u64)>,
}

impl Bm25PhraseWeight {
    fn new(sim: &Bm25, stats: &[TermStats], field: FieldStats, boost: f32) -> Self {
        let avg = avg_len(field, 1.0);
        // idf, sum of: accumulated in double from the float idfs, rounded once.
        let mut sum = 0f64;
        for s in stats {
            sum += bm25_idf(s.doc_freq, s.num_docs.max(1)) as f64;
        }
        let first = stats[0];
        let inner = Bm25Weight::new(first.doc_freq, first.num_docs.max(1), sim.params(), avg).with_boost(boost).with_idf(sum as f32);
        Self { inner, terms: stats.iter().map(|s| (s.doc_freq, s.num_docs)).collect() }
    }
}

impl SimWeight for Bm25PhraseWeight {
    #[inline]
    fn score(&self, tf: u32, field_len: u32) -> f32 {
        self.inner.score(tf, field_len)
    }

    #[inline]
    fn score_norm(&self, tf: u32, norm: u8) -> f32 {
        self.inner.score_norm(tf, norm)
    }

    fn explain(&self, tf: u32, field_len: u32) -> Explanation {
        let parts: Vec<Explanation> = self
            .terms
            .iter()
            .map(|&(df, n)| {
                Explanation::new(
                    bm25_idf(df, n.max(1)),
                    "idf, computed as ln(1 + (N - n + 0.5) / (n + 0.5))",
                    vec![Explanation::leaf(df as f32, "n, number of documents containing term"), Explanation::leaf(n as f32, "N, total number of documents")],
                )
            })
            .collect();
        let idf_node = Explanation::new(self.inner.idf_value(), "idf, sum of:", parts);
        self.inner.explain_with_idf(tf, field_len, idf_node, format!("phraseFreq={:?}", tf as f32))
    }
}

// ---------------------------------------------------------------------------
// Classic TF-IDF
// ---------------------------------------------------------------------------

/// TF-IDF (Lucene's `ClassicSimilarity` / `TFIDFSimilarity`, Lucene >= 7,
/// no query norm):
///
/// `score = sqrt(tf) * (boost * idf) * lengthNorm` with
/// `idf = ln((N + 1) / (df + 1)) + 1` and `lengthNorm = 1 / sqrt(len)`
/// (a zero length uses `1 / lengthNorm(byte4ToInt(255))`, as Lucene's
/// norm table does).
#[derive(Clone, Copy, Debug, Default)]
pub struct Classic;

/// Weight for [`Classic`].
pub struct ClassicWeight {
    idf: f32,
    boost: f32,
    df: u64,
    n: u64,
}

impl Similarity for Classic {
    fn weight(&self, stats: TermStats, _field: FieldStats, boost: f32) -> Box<dyn SimWeight> {
        let idf = ((stats.num_docs as f64 + 1.0) / (stats.doc_freq as f64 + 1.0)).ln() + 1.0;
        Box::new(ClassicWeight { idf: idf as f32, boost, df: stats.doc_freq, n: stats.num_docs })
    }
    fn name(&self) -> String {
        "Classic".into()
    }
}

/// Lucene `ClassicSimilarity.lengthNorm` as stored in `TFIDFSimilarity`'s
/// norm table (length 0 maps to `1 / normTable[255]`).
fn classic_length_norm(field_len: u32) -> f32 {
    if field_len == 0 {
        let max_len = crate::query::bm25::byte4_to_int(255);
        return 1.0 / (1.0 / (max_len as f64).sqrt()) as f32;
    }
    (1.0 / (field_len as f64).sqrt()) as f32
}

impl SimWeight for ClassicWeight {
    fn score(&self, tf: u32, field_len: u32) -> f32 {
        if tf == 0 {
            return 0.0;
        }
        // TFIDFScorer: tf(freq) * (boost * idf) * normTable[norm].
        let raw = ((tf as f64).sqrt() as f32) * (self.boost * self.idf);
        raw * classic_length_norm(field_len)
    }

    fn explain(&self, tf: u32, field_len: u32) -> Explanation {
        let len_norm = classic_length_norm(field_len);
        Explanation::new(
            self.score(tf, field_len),
            format!("score(freq={tf}), computed as boost * sqrt(tf) * idf * lengthNorm"),
            vec![
                Explanation::leaf(self.boost, "boost"),
                Explanation::leaf((tf as f32).sqrt(), "sqrt(tf)"),
                Explanation::new(
                    self.idf,
                    "idf, computed as log((docCount+1)/(docFreq+1)) + 1",
                    vec![Explanation::leaf(self.df as f32, "n, number of documents containing term"), Explanation::leaf(self.n as f32, "N, total number of documents")],
                ),
                Explanation::leaf(len_norm, format!("lengthNorm, computed as 1 / sqrt(dl) with dl={field_len}")),
            ],
        )
    }
}

// ---------------------------------------------------------------------------
// Boolean
// ---------------------------------------------------------------------------

/// Every match scores the query boost (Lucene's `BooleanSimilarity`).
#[derive(Clone, Copy, Debug, Default)]
pub struct BooleanSim;

/// Weight for [`BooleanSim`].
pub struct BooleanWeight {
    boost: f32,
}

impl Similarity for BooleanSim {
    fn weight(&self, _stats: TermStats, _field: FieldStats, boost: f32) -> Box<dyn SimWeight> {
        Box::new(BooleanWeight { boost })
    }
    /// A phrase scores its boost once, however many terms it has.
    fn phrase_weight(&self, _stats: &[TermStats], _field: FieldStats, boost: f32) -> Box<dyn SimWeight> {
        Box::new(BooleanWeight { boost })
    }
    fn name(&self) -> String {
        "Boolean".into()
    }
}

impl SimWeight for BooleanWeight {
    fn score(&self, _tf: u32, _field_len: u32) -> f32 {
        self.boost
    }
    fn explain(&self, _tf: u32, _field_len: u32) -> Explanation {
        Explanation::new(self.boost, "score(BooleanWeight), computed from:", vec![Explanation::leaf(self.boost, "boost, query boost")])
    }
}

// ---------------------------------------------------------------------------
// Shared pieces for DFR / IB: normalizations
// ---------------------------------------------------------------------------

/// Term-frequency normalization by document length (Lucene's
/// `Normalization` classes, shared by DFR and IB).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Normalization {
    /// `tfn = tf` (no normalization).
    None,
    /// `tfn = tf * c * avg_len / len`.
    H1 { c: f32 },
    /// `tfn = tf * log2(1 + c * avg_len / len)`.
    H2 { c: f32 },
    /// Dirichlet priors: `tfn = (tf + mu * ((F + 1) / (T + 1))) / (len + mu) * mu`,
    /// where `F` is the total term frequency and `T` the number of tokens
    /// in the field.
    H3 { mu: f32 },
    /// Pareto-Zipf: `tfn = tf * (avg_len / len)^z`.
    Z { z: f32 },
}

impl Normalization {
    /// Lucene defaults: `H1(c=1)`, `H2(c=1)`, `H3(mu=800)`, `Z(z=0.3)`.
    pub fn h1() -> Self {
        Normalization::H1 { c: 1.0 }
    }
    /// `H2` with `c = 1`.
    pub fn h2() -> Self {
        Normalization::H2 { c: 1.0 }
    }
    /// `H3` with `mu = 800`.
    pub fn h3() -> Self {
        Normalization::H3 { mu: 800.0 }
    }
    /// `Z` with `z = 0.3`.
    pub fn z() -> Self {
        Normalization::Z { z: 0.3 }
    }

    fn tfn(&self, tf: f64, len: f64, avg_len: f64, total_term_freq: f64, field_tokens: f64) -> f64 {
        match *self {
            Normalization::None => tf,
            Normalization::H1 { c } => tf * c as f64 * avg_len / len,
            Normalization::H2 { c } => tf * log2(1.0 + c as f64 * avg_len / len),
            Normalization::H3 { mu } => {
                let mu = mu as f64;
                (tf + mu * ((total_term_freq + 1.0) / (field_tokens + 1.0))) / (len + mu) * mu
            }
            Normalization::Z { z } => tf * (avg_len / len).powf(z as f64),
        }
    }
}

/// Statistics shared by the DFR / IB / LM weights (Lucene's `BasicStats`).
#[derive(Clone, Copy, Debug)]
struct BasicStats {
    /// Number of docs in the index.
    n: f64,
    /// Doc frequency of the term.
    df: f64,
    /// Total term frequency of the term.
    ttf: f64,
    /// Number of tokens in the field across the index.
    field_tokens: f64,
    /// Average field length.
    avg_len: f64,
    boost: f32,
}

impl BasicStats {
    fn new(stats: TermStats, field: FieldStats, boost: f32) -> Self {
        let avg = avg_len(field, 1.0) as f64;
        let n = stats.num_docs.max(1) as f64;
        let field_tokens = if field.sum_field_len > 0 { field.sum_field_len as f64 } else { avg * n };
        Self { n, df: stats.doc_freq as f64, ttf: stats.total_term_freq as f64, field_tokens, avg_len: avg, boost }
    }
}

// ---------------------------------------------------------------------------
// DFR
// ---------------------------------------------------------------------------

/// DFR basic models (Lucene's `BasicModel*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BasicModel {
    /// Geometric approximation of the Bose-Einstein model.
    G,
    /// Inverse term frequency: `tfn * log2(1 + (N + 1) / (F + 0.5))`.
    IF,
    /// Inverse document frequency: `tfn * log2((N + 1) / (n + 0.5))`.
    In,
    /// Inverse expected document frequency: `tfn * log2((N + 1) / (ne + 0.5))`
    /// with `ne = N * (1 - ((N - 1) / N)^F)`.
    Ine,
}

impl BasicModel {
    fn score(&self, s: &BasicStats, tfn: f64) -> f64 {
        let n_docs = s.n;
        match self {
            BasicModel::G => {
                let f = s.ttf + 1.0;
                let lambda = f / (n_docs + f);
                let a = log2(lambda + 1.0);
                let b = log2((1.0 + lambda) / lambda);
                a + b * tfn
            }
            BasicModel::IF => tfn * log2(1.0 + (n_docs + 1.0) / (s.ttf + 0.5)),
            BasicModel::In => tfn * log2((n_docs + 1.0) / (s.df + 0.5)),
            BasicModel::Ine => {
                let ne = n_docs * (1.0 - ((n_docs - 1.0) / n_docs).powf(s.ttf));
                tfn * log2((n_docs + 1.0) / (ne + 0.5))
            }
        }
    }
}

/// DFR after effects (first normalization; Lucene's `AfterEffect*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AfterEffect {
    /// No after effect.
    None,
    /// Bernoulli: `(F + 1) / (n * (tfn + 1))` with `F = ttf + 1` and
    /// `n = df + 1` (Lucene's `AfterEffectB`).
    B,
    /// Laplace: `1 / (tfn + 1)`.
    L,
}

impl AfterEffect {
    fn score(&self, s: &BasicStats, tfn: f64) -> f64 {
        match self {
            AfterEffect::None => 1.0,
            AfterEffect::B => (s.ttf + 2.0) / ((s.df + 1.0) * (tfn + 1.0)),
            AfterEffect::L => 1.0 / (tfn + 1.0),
        }
    }
}

/// Divergence from randomness (Lucene's `DFRSimilarity`):
/// `score = boost * basic_model(tfn) * after_effect(tfn)`.
#[derive(Clone, Copy, Debug)]
pub struct DfrSimilarity {
    /// Basic model of the term's randomness.
    pub basic_model: BasicModel,
    /// First normalization (information gain).
    pub after_effect: AfterEffect,
    /// Second normalization (term frequency by length).
    pub normalization: Normalization,
}

impl DfrSimilarity {
    /// Build a DFR similarity.
    pub fn new(basic_model: BasicModel, after_effect: AfterEffect, normalization: Normalization) -> Self {
        Self { basic_model, after_effect, normalization }
    }
}

/// Weight for [`DfrSimilarity`].
pub struct DfrWeight {
    sim: DfrSimilarity,
    stats: BasicStats,
}

impl Similarity for DfrSimilarity {
    fn weight(&self, stats: TermStats, field: FieldStats, boost: f32) -> Box<dyn SimWeight> {
        Box::new(DfrWeight { sim: *self, stats: BasicStats::new(stats, field, boost) })
    }
    fn name(&self) -> String {
        format!("DFR {:?}{:?}{:?}", self.basic_model, self.after_effect, self.normalization)
    }
}

impl DfrWeight {
    /// Lucene >= 8 `DFRSimilarity.score`: the basic model is evaluated with
    /// the after effect folded in as `aeTimes1pTfn` (same value as
    /// `basic(tfn) * after_effect(tfn)`, monotone evaluation order).
    fn lucene_score(&self, tfn: f64) -> f64 {
        let s = &self.stats;
        let ae1p = match self.sim.after_effect {
            AfterEffect::B => (s.ttf + 2.0) / (s.df + 1.0),
            AfterEffect::L => 1.0,
            // Not in Lucene >= 8 (pre-8 `NoAfterEffect`): basic(tfn) * 1.
            AfterEffect::None => return self.sim.basic_model.score(s, tfn),
        };
        let n = s.n;
        match self.sim.basic_model {
            BasicModel::G => {
                let f = s.ttf + 1.0;
                let lambda = f / (n + f);
                let a = log2(lambda + 1.0);
                let b = log2((1.0 + lambda) / lambda);
                ae1p * (b - (b - a) / (1.0 + tfn))
            }
            BasicModel::IF => {
                let a = log2(1.0 + (n + 1.0) / (s.ttf + 0.5));
                a * ae1p * (1.0 - 1.0 / (1.0 + tfn))
            }
            BasicModel::In => {
                let a = log2((n + 1.0) / (s.df + 0.5));
                a * ae1p * (1.0 - 1.0 / (1.0 + tfn))
            }
            BasicModel::Ine => {
                let ne = n * (1.0 - ((n - 1.0) / n).powf(s.ttf));
                let a = log2((n + 1.0) / (ne + 0.5));
                a * ae1p * (1.0 - 1.0 / (1.0 + tfn))
            }
        }
    }

    fn parts(&self, tf: u32, field_len: u32) -> (f64, f64, f64) {
        let s = &self.stats;
        let tfn = self.sim.normalization.tfn(tf as f64, field_len.max(1) as f64, s.avg_len, s.ttf, s.field_tokens);
        let basic = self.sim.basic_model.score(s, tfn);
        let ae = self.sim.after_effect.score(s, tfn);
        (tfn, basic, ae)
    }
}

impl SimWeight for DfrWeight {
    fn score(&self, tf: u32, field_len: u32) -> f32 {
        if tf == 0 {
            return 0.0;
        }
        let (tfn, _, _) = self.parts(tf, field_len);
        (self.stats.boost as f64 * self.lucene_score(tfn)).max(0.0) as f32
    }

    fn explain(&self, tf: u32, field_len: u32) -> Explanation {
        let (tfn, basic, ae) = self.parts(tf, field_len);
        Explanation::new(
            self.score(tf, field_len),
            format!("score(freq={tf}), computed as boost * basic_model * after_effect"),
            vec![
                Explanation::leaf(self.stats.boost, "boost"),
                Explanation::leaf(tfn as f32, format!("tfn, normalized term frequency ({:?})", self.sim.normalization)),
                Explanation::leaf(basic as f32, format!("basic model {:?}", self.sim.basic_model)),
                Explanation::leaf(ae as f32, format!("after effect {:?}", self.sim.after_effect)),
            ],
        )
    }
}

// ---------------------------------------------------------------------------
// IB
// ---------------------------------------------------------------------------

/// IB distributions (Lucene's `Distribution*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Distribution {
    /// Log-logistic: `-ln(lambda / (tfn + lambda))`.
    LL,
    /// Smoothed power-law: `-ln((lambda^(tfn / (tfn + 1)) - lambda) / (1 - lambda))`.
    SPL,
}

impl Distribution {
    fn score(&self, tfn: f64, lambda: f64) -> f64 {
        match self {
            // Lucene `DistributionLL` / `DistributionSPL` (natural log).
            Distribution::LL => -(lambda / (tfn + lambda)).ln(),
            Distribution::SPL => {
                let mut q = 1.0 - 1.0 / (tfn + 1.0);
                if q == 1.0 {
                    q = 1.0f64.next_down();
                }
                let mut pow = lambda.powf(q);
                if pow == lambda {
                    pow = if lambda < 1.0 { lambda.next_up() } else { lambda.next_down() };
                }
                (-((pow - lambda) / (1.0 - lambda)).ln()).max(0.0)
            }
        }
    }
}

/// IB lambdas (Lucene's `Lambda*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lambda {
    /// `(n + 1) / (N + 1)` (document frequency).
    DF,
    /// `(F + 1) / (N + 1)` (total term frequency).
    TTF,
}

impl Lambda {
    /// Lucene `LambdaDF` / `LambdaTTF`: computed in double, rounded to
    /// float, nudged below 1.
    fn value(&self, s: &BasicStats) -> f64 {
        let l = match self {
            Lambda::DF => ((s.df + 1.0) / (s.n + 1.0)) as f32,
            Lambda::TTF => ((s.ttf + 1.0) / (s.n + 1.0)) as f32,
        };
        (if l == 1.0 { l.next_down() } else { l }) as f64
    }
}

/// Information-based model (Lucene's `IBSimilarity`):
/// `score = boost * distribution(tfn, lambda)`.
#[derive(Clone, Copy, Debug)]
pub struct IbSimilarity {
    /// Probabilistic distribution.
    pub distribution: Distribution,
    /// Lambda estimate.
    pub lambda: Lambda,
    /// Term-frequency normalization.
    pub normalization: Normalization,
}

impl IbSimilarity {
    /// Build an IB similarity.
    pub fn new(distribution: Distribution, lambda: Lambda, normalization: Normalization) -> Self {
        Self { distribution, lambda, normalization }
    }
}

/// Weight for [`IbSimilarity`].
pub struct IbWeight {
    sim: IbSimilarity,
    stats: BasicStats,
}

impl Similarity for IbSimilarity {
    fn weight(&self, stats: TermStats, field: FieldStats, boost: f32) -> Box<dyn SimWeight> {
        Box::new(IbWeight { sim: *self, stats: BasicStats::new(stats, field, boost) })
    }
    fn name(&self) -> String {
        format!("IB {:?}{:?}{:?}", self.distribution, self.lambda, self.normalization)
    }
}

impl SimWeight for IbWeight {
    fn score(&self, tf: u32, field_len: u32) -> f32 {
        if tf == 0 {
            return 0.0;
        }
        let s = &self.stats;
        let tfn = self.sim.normalization.tfn(tf as f64, field_len.max(1) as f64, s.avg_len, s.ttf, s.field_tokens);
        let lambda = self.sim.lambda.value(s);
        (s.boost as f64 * self.sim.distribution.score(tfn, lambda)).max(0.0) as f32
    }

    fn explain(&self, tf: u32, field_len: u32) -> Explanation {
        let s = &self.stats;
        let tfn = self.sim.normalization.tfn(tf as f64, field_len.max(1) as f64, s.avg_len, s.ttf, s.field_tokens);
        let lambda = self.sim.lambda.value(s);
        Explanation::new(
            self.score(tf, field_len),
            format!("score(freq={tf}), computed as boost * distribution(tfn, lambda)"),
            vec![
                Explanation::leaf(s.boost, "boost"),
                Explanation::leaf(tfn as f32, format!("tfn, normalized term frequency ({:?})", self.sim.normalization)),
                Explanation::leaf(lambda as f32, format!("lambda ({:?})", self.sim.lambda)),
                Explanation::leaf(self.sim.distribution.score(tfn, lambda) as f32, format!("distribution {:?}", self.sim.distribution)),
            ],
        )
    }
}

// ---------------------------------------------------------------------------
// Language models
// ---------------------------------------------------------------------------

/// Collection probability of a term: `(F + 1) / (T + 1)`.
fn collection_probability(s: &BasicStats) -> f64 {
    (s.ttf + 1.0) / (s.field_tokens + 1.0)
}

/// Language model with Dirichlet smoothing (Lucene's `LMDirichletSimilarity`):
/// `score = max(0, boost * (ln(1 + tf / (mu * p)) + ln(mu / (len + mu))))`
/// with `p` the collection probability of the term.
#[derive(Clone, Copy, Debug)]
pub struct LmDirichlet {
    /// Smoothing parameter (default 2000).
    pub mu: f32,
}

impl Default for LmDirichlet {
    fn default() -> Self {
        Self { mu: 2000.0 }
    }
}

impl LmDirichlet {
    /// Dirichlet smoothing with `mu`.
    pub fn new(mu: f32) -> Self {
        Self { mu }
    }
}

/// Weight for [`LmDirichlet`].
pub struct LmDirichletWeight {
    mu: f64,
    stats: BasicStats,
}

impl Similarity for LmDirichlet {
    fn weight(&self, stats: TermStats, field: FieldStats, boost: f32) -> Box<dyn SimWeight> {
        Box::new(LmDirichletWeight { mu: self.mu as f64, stats: BasicStats::new(stats, field, boost) })
    }
    fn name(&self) -> String {
        format!("LM Dirichlet({})", self.mu)
    }
}

impl SimWeight for LmDirichletWeight {
    fn score(&self, tf: u32, field_len: u32) -> f32 {
        let p = collection_probability(&self.stats);
        let len = field_len.max(1) as f64;
        let raw = (1.0 + tf as f64 / (self.mu * p)).ln() + (self.mu / (len + self.mu)).ln();
        (self.stats.boost as f64 * raw).max(0.0) as f32
    }

    fn explain(&self, tf: u32, field_len: u32) -> Explanation {
        let p = collection_probability(&self.stats);
        let len = field_len.max(1) as f64;
        Explanation::new(
            self.score(tf, field_len),
            format!("score(freq={tf}), computed as max(0, boost * (ln(1 + freq / (mu * p)) + ln(mu / (dl + mu))))"),
            vec![
                Explanation::leaf(self.stats.boost, "boost"),
                Explanation::leaf(self.mu as f32, "mu"),
                Explanation::leaf(p as f32, "p, collection probability (F + 1) / (T + 1)"),
                Explanation::leaf(len as f32, "dl, length of field"),
            ],
        )
    }
}

/// Language model with Jelinek-Mercer smoothing (Lucene's
/// `LMJelinekMercerSimilarity`):
/// `score = boost * ln(1 + ((1 - lambda) * tf / len) / (lambda * p))`.
#[derive(Clone, Copy, Debug)]
pub struct LmJelinekMercer {
    /// Mixing weight of the collection model (default 0.1).
    pub lambda: f32,
}

impl Default for LmJelinekMercer {
    fn default() -> Self {
        Self { lambda: 0.1 }
    }
}

impl LmJelinekMercer {
    /// Jelinek-Mercer smoothing with `lambda`.
    pub fn new(lambda: f32) -> Self {
        Self { lambda }
    }
}

/// Weight for [`LmJelinekMercer`].
pub struct LmJelinekMercerWeight {
    lambda: f64,
    stats: BasicStats,
}

impl Similarity for LmJelinekMercer {
    fn weight(&self, stats: TermStats, field: FieldStats, boost: f32) -> Box<dyn SimWeight> {
        Box::new(LmJelinekMercerWeight { lambda: self.lambda as f64, stats: BasicStats::new(stats, field, boost) })
    }
    fn name(&self) -> String {
        format!("LM Jelinek-Mercer({})", self.lambda)
    }
}

impl SimWeight for LmJelinekMercerWeight {
    fn score(&self, tf: u32, field_len: u32) -> f32 {
        if tf == 0 {
            return 0.0;
        }
        let p = collection_probability(&self.stats);
        let len = field_len.max(1) as f64;
        let raw = (1.0 + ((1.0 - self.lambda) * tf as f64 / len) / (self.lambda * p)).ln();
        (self.stats.boost as f64 * raw).max(0.0) as f32
    }

    fn explain(&self, tf: u32, field_len: u32) -> Explanation {
        let p = collection_probability(&self.stats);
        Explanation::new(
            self.score(tf, field_len),
            format!("score(freq={tf}), computed as boost * ln(1 + ((1 - lambda) * freq / dl) / (lambda * p))"),
            vec![
                Explanation::leaf(self.stats.boost, "boost"),
                Explanation::leaf(self.lambda as f32, "lambda"),
                Explanation::leaf(p as f32, "p, collection probability (F + 1) / (T + 1)"),
                Explanation::leaf(field_len.max(1) as f32, "dl, length of field"),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn stats() -> (TermStats, FieldStats) {
        // 100 docs, term in 10 of them 25 times total, field has 1000 tokens (avg len 10).
        (TermStats { doc_freq: 10, total_term_freq: 25, num_docs: 100 }, FieldStats { num_docs: 100, sum_field_len: 1000 })
    }

    fn close(a: f32, b: f32, what: &str) {
        assert!((a - b).abs() < EPS, "{what}: {a} vs {b}");
    }

    #[test]
    fn bm25_hand_computed() {
        let (ts, fs) = stats();
        let w = Bm25::default().weight(ts, fs, 1.0);
        // idf = ln(1 + (100 - 10 + 0.5)/(10 + 0.5)) = ln(1 + 90.5/10.5)
        let idf = (1.0f32 + 90.5 / 10.5).ln();
        // tf=2, len=5: norm = 1.2 * (1 - 0.75 + 0.75 * 5/10) = 1.2 * 0.625 = 0.75
        let want = idf * 2.0 / (2.0 + 0.75);
        close(w.score(2, 5), want, "bm25");
        close(Bm25::default().weight(ts, fs, 2.0).score(2, 5), 2.0 * want, "bm25 boost");
        let e = w.explain(2, 5);
        close(e.value, want, "bm25 explain");
        assert_eq!(e.details.len(), 3);
    }

    #[test]
    fn classic_hand_computed() {
        let (ts, fs) = stats();
        let w = Classic.weight(ts, fs, 1.0);
        // Lucene: idf = ln((N + 1) / (df + 1)) + 1 = ln(101 / 11) + 1
        let idf = (101.0f64 / 11.0).ln() as f32 + 1.0;
        // tf=4, len=16: sqrt(4) * idf / sqrt(16) = 2 * idf / 4
        close(w.score(4, 16), 0.5 * idf, "classic");
        assert_eq!(w.score(0, 16), 0.0);
        close(Classic.weight(ts, fs, 3.0).score(4, 16), 1.5 * idf, "classic boost");
    }

    #[test]
    fn boolean_is_boost() {
        let (ts, fs) = stats();
        let w = BooleanSim.weight(ts, fs, 2.5);
        assert_eq!(w.score(1, 1), 2.5);
        assert_eq!(w.score(100, 1000), 2.5);
    }

    #[test]
    fn dfr_hand_computed() {
        let (ts, fs) = stats();
        // I(n) + L + H2, tf=3, len=20, avg=10: tfn = 3 * log2(1 + 10/20) = 3 * log2(1.5)
        let tfn = 3.0 * 1.5f64.log2();
        let basic = tfn * (101.0f64 / 10.5).log2();
        let ae = 1.0 / (tfn + 1.0);
        let w = DfrSimilarity::new(BasicModel::In, AfterEffect::L, Normalization::h2()).weight(ts, fs, 1.0);
        close(w.score(3, 20), (basic * ae) as f32, "dfr InL2");

        // I(F) + B + H1, tf=2, len=5: tfn = 2 * 10/5 = 4
        let tfn = 4.0f64;
        let basic = tfn * (1.0 + 101.0 / 25.5f64).log2();
        // Lucene AfterEffectB: (F + 1) / (n * (tfn + 1)), F = ttf + 1, n = df + 1
        let ae = 27.0 / (11.0 * 5.0);
        let w = DfrSimilarity::new(BasicModel::IF, AfterEffect::B, Normalization::h1()).weight(ts, fs, 1.0);
        close(w.score(2, 5), (basic * ae) as f32, "dfr IFB1");

        // G + None + None, tf=1: lambda = 26/126; A + B*1
        let lambda = 26.0f64 / 126.0;
        let g = (lambda + 1.0).log2() + ((1.0 + lambda) / lambda).log2();
        let w = DfrSimilarity::new(BasicModel::G, AfterEffect::None, Normalization::None).weight(ts, fs, 1.0);
        close(w.score(1, 10), g as f32, "dfr G");

        // I(ne) + L + Z, tf=1, len=10 (tfn = 1): ne = 100 * (1 - 0.99^25)
        let ne = 100.0 * (1.0 - 0.99f64.powi(25));
        let basic = (101.0 / (ne + 0.5)).log2();
        let w = DfrSimilarity::new(BasicModel::Ine, AfterEffect::L, Normalization::z()).weight(ts, fs, 1.0);
        close(w.score(1, 10), (basic * 0.5) as f32, "dfr IneLZ");

        // H3: tfn = (tf + mu * 26/1001) / (len + mu) * mu
        let mu = 800.0f64;
        let tfn = (2.0 + mu * 26.0 / 1001.0) / (5.0 + mu) * mu;
        let basic = tfn * (101.0f64 / 10.5).log2();
        let w = DfrSimilarity::new(BasicModel::In, AfterEffect::None, Normalization::h3()).weight(ts, fs, 1.0);
        close(w.score(2, 5), basic as f32, "dfr InH3");
        assert_eq!(w.score(0, 5), 0.0);
    }

    #[test]
    fn ib_hand_computed() {
        let (ts, fs) = stats();
        // LL + DF + H1: tf=2, len=5 → tfn=4; lambda = 11/101
        let lambda = (11.0f64 / 101.0) as f32 as f64;
        let want = -(lambda / (4.0 + lambda)).ln();
        let w = IbSimilarity::new(Distribution::LL, Lambda::DF, Normalization::h1()).weight(ts, fs, 1.0);
        close(w.score(2, 5), want as f32, "ib LL-DF-H1");
        // SPL + TTF + None: tf=1 → tfn=1; lambda = 26/101
        let lambda = (26.0f64 / 101.0) as f32 as f64;
        let want = -((lambda.powf(0.5) - lambda) / (1.0 - lambda)).ln();
        let w = IbSimilarity::new(Distribution::SPL, Lambda::TTF, Normalization::None).weight(ts, fs, 1.0);
        close(w.score(1, 10), want as f32, "ib SPL-TTF");
    }

    #[test]
    fn lm_hand_computed() {
        let (ts, fs) = stats();
        let p = 26.0f64 / 1001.0;
        // Dirichlet mu=2000, tf=3, len=10
        let want = (1.0 + 3.0 / (2000.0 * p)).ln() + (2000.0 / 2010.0f64).ln();
        let w = LmDirichlet::default().weight(ts, fs, 1.0);
        close(w.score(3, 10), want.max(0.0) as f32, "lm dirichlet");
        // tf=0 in a very long doc → negative raw → clamped to 0
        assert_eq!(w.score(0, 100_000), 0.0);
        // Jelinek-Mercer lambda=0.1, tf=2, len=8
        let want = (1.0 + (0.9 * 2.0 / 8.0) / (0.1 * p)).ln();
        let w = LmJelinekMercer::default().weight(ts, fs, 1.0);
        close(w.score(2, 8), want as f32, "lm jm");
        close(LmJelinekMercer::new(0.5).weight(ts, fs, 2.0).score(2, 8), 2.0 * (1.0 + (0.5 * 2.0 / 8.0) / (0.5 * p)).ln() as f32, "lm jm boost");
    }

    #[test]
    fn phrase_weight_sums_terms() {
        let (ts, fs) = stats();
        let other = TermStats { doc_freq: 50, total_term_freq: 80, num_docs: 100 };
        // BM25 scores a phrase with one weight over the summed idf
        // ("idf, sum of:"), not the sum of per-term scores.
        let sim = Bm25::default();
        let w = sim.phrase_weight(&[ts, other], fs, 1.0);
        let idf_sum = bm25_idf(10, 100) + bm25_idf(50, 100);
        let want = idf_sum * 2.0 / (2.0 + 1.2 * (1.0 - 0.75 + 0.75 * 7.0 / 10.0));
        close(w.score(2, 7), want, "bm25 phrase");
        let e = w.explain(2, 7);
        assert_eq!(e.details.len(), 3); // boost, idf (sum of), tf
        assert_eq!(e.details[1].description, "idf, sum of:");
        assert_eq!(e.details[1].details.len(), 2);
        assert!(e.details[2].details[0].description.starts_with("phraseFreq=2.0"));
        close(e.value, want, "bm25 phrase explain");
        // Other similarities sum the per-term weights.
        let sim = Classic;
        let w = sim.phrase_weight(&[ts, other], fs, 1.0);
        let want = sim.weight(ts, fs, 1.0).score(2, 7) + sim.weight(other, fs, 1.0).score(2, 7);
        close(w.score(2, 7), want, "sum");
        let e = w.explain(2, 7);
        assert_eq!(e.details.len(), 2);
        close(e.value, want, "sum explain");
    }
}
