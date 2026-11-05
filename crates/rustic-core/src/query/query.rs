//! The query tree (Lucene's `Query` classes) and per-leaf scorer building
//! (Lucene's `Weight.scorer(LeafReaderContext)`).
//!
//! A [`Query`] is a plain value. The searcher first collects every term it
//! references to compute index-wide statistics ([`SearchContext`]), then
//! asks the query for a [`Scorer`] on each leaf. Multi-term queries
//! (prefix / wildcard / regexp / fuzzy / term range) enumerate the leaf's
//! dictionary through [`LeafReader::term_range`], so they work on the
//! memtable as well as sealed segments; their scoring rewrites
//! ([`MultiTermRewrite`]) are expanded once over all leaves when the
//! context is built.
//!
//! Scores go through a [`Similarity`] (BM25 by default), chosen per field
//! by [`SearchContext::sim`].

use crate::automaton::levenshtein::{edit_distance, LevenshteinMatcher};
use crate::automaton::regexp::{RegexpFlags, RegexpMatcher};
use crate::automaton::wildcard::WildcardMatcher;
use crate::automaton::TermMatcher;
use crate::query::bm25::Bm25Params;
use crate::query::boolean::{BooleanScorer, MinShouldMatch, MinShouldMatchUnion, Union};
use crate::query::docset::{AllDocSet, BitSetDocSet, DocSet, EmptyDocSet, VecDocSet, TERMINATED};
use crate::query::explain::Explanation;
use crate::query::function_score::{
    BoostMode, FunctionScoreParams, FunctionScoreScorer, PreparedFunctions, ScoreFunction, ScoreMode,
};
use crate::query::multi_term::{
    const_union, const_union_sorted, enumerate_terms, expand_global_bounded, scoring_union, ExpandedTerm, Fuzziness, MultiTermRewrite,
};
use crate::codec::postings::PostingsReader;
use crate::query::reader::{field_stats, term_stats_with_infos, FieldStats, LeafReader, TermStats};
use crate::term_dictionary::PostingsLocation;
use crate::query::scorer::{ConstScorer, Scorer};
use crate::query::segment_scorer::Scaled;
use crate::query::similarity::{Bm25, Similarity};
use crate::query::spans::{IntervalsRule, SpanQuery, SpanScorer};
use crate::geo::GeoShape;
use crate::nested::{NestedHit, NestedScoreMode};
use crate::types::{DocId, Term};
use std::collections::HashMap;
use std::sync::Arc;

/// Inclusive/exclusive numeric bound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Bound<T> {
    /// `>=` / `<=`.
    Included(T),
    /// `>` / `<`.
    Excluded(T),
    /// No bound.
    Unbounded,
}

/// Date range bounds: i64 milliseconds since the epoch ([`Query::RangeDate`]
/// behaves exactly like [`Query::RangeI64`]).
pub type RangeDate = Bound<i64>;

/// A search query.
#[derive(Clone, Debug)]
pub enum Query {
    /// Single term, scored by the field's similarity. For a
    /// case-insensitive match build it with [`Query::term_case_insensitive`].
    Term(Term),
    /// Any of the terms (ES `terms`): constant score `boost` per matching doc.
    TermSet { field: u16, terms: Vec<Vec<u8>>, boost: f32 },
    /// Consecutive terms in one field with `slop` (ES `match_phrase`).
    Phrase { terms: Vec<Term>, slop: u32 },
    /// A phrase whose every position accepts any of several terms (Lucene
    /// `MultiPhraseQuery`; `match_phrase` with synonyms).
    MultiPhrase { field: u16, positions: Vec<Vec<Term>>, slop: u32 },
    /// Terms starting with `prefix`. With `case_insensitive` the prefix is
    /// lowercased (the field's analyzer must have lowercased the indexed
    /// terms for this to be a true case-insensitive match).
    Prefix { field: u16, prefix: Vec<u8>, max_expansions: usize, case_insensitive: bool, rewrite: MultiTermRewrite },
    /// `*`/`?` pattern over term text. `case_insensitive` lowercases the
    /// pattern (see [`Query::Prefix`]).
    Wildcard { field: u16, pattern: String, max_expansions: usize, case_insensitive: bool, rewrite: MultiTermRewrite },
    /// Whole-term regular expression; `case_insensitive` matches terms
    /// regardless of case.
    Regexp { field: u16, pattern: String, case_insensitive: bool, max_expansions: usize, rewrite: MultiTermRewrite },
    /// Edit-distance match. Each term's score is scaled by
    /// `1 - distance / term_len` as Lucene does; the default rewrite is
    /// [`MultiTermRewrite::TopTermsBlendedFreqs`]`(max_expansions)`.
    Fuzzy {
        field: u16,
        text: String,
        fuzziness: Fuzziness,
        prefix_length: usize,
        transpositions: bool,
        max_expansions: usize,
        rewrite: MultiTermRewrite,
    },
    /// Dictionary range on term bytes (keyword/text range).
    TermRange { field: u16, lower: Bound<Vec<u8>>, upper: Bound<Vec<u8>>, rewrite: MultiTermRewrite },
    /// i64 fast-field range.
    RangeI64 { field: u16, lower: Bound<i64>, upper: Bound<i64> },
    /// u64 fast-field range.
    RangeU64 { field: u16, lower: Bound<u64>, upper: Bound<u64> },
    /// f64 fast-field range.
    RangeF64 { field: u16, lower: Bound<f64>, upper: Bound<f64> },
    /// Date range (i64 millis); identical to [`Query::RangeI64`].
    RangeDate { field: u16, lower: RangeDate, upper: RangeDate },
    /// Docs that have a value in `field` (ES `exists`): fast-field presence
    /// when the leaf can tell, else any indexed term.
    Exists { field: u16 },
    /// Boolean combination. Shoulds only add score when there is a must or
    /// filter clause, unless `minimum_should_match` requires some; give a
    /// clause its own boost by wrapping it in [`Query::Boost`].
    Bool {
        must: Vec<Query>,
        should: Vec<Query>,
        must_not: Vec<Query>,
        filter: Vec<Query>,
        minimum_should_match: MinShouldMatch,
    },
    /// Every match of `query` scores `score`. `point_set` marks the single
    /// set query ES builds for `terms` over point/term-set fields (numbers,
    /// ip, boolean, version, ...): it counts as one clause against the max
    /// clause count however many values it holds. Only the query builder
    /// that produced the set sets it; a user `constant_score` never does.
    ConstScore { query: Box<Query>, score: f32, point_set: bool },
    /// `query`'s scores multiplied by `boost`.
    Boost { query: Box<Query>, boost: f32 },
    /// Max of the sub-scores plus `tie_breaker` times the rest (ES `dis_max`).
    DisMax { queries: Vec<Query>, tie_breaker: f32 },
    /// Approximate k-NN (ES `knn`), optionally pre-filtered. Hits whose
    /// similarity score is below `similarity_threshold` are dropped; scores
    /// are multiplied by `boost`.
    Knn {
        field: u16,
        vector: Vec<f32>,
        k: usize,
        num_candidates: usize,
        filter: Option<Box<Query>>,
        similarity_threshold: Option<f32>,
        boost: f32,
        /// Rescore oversample override (ES `rescore_vector.oversample` on
        /// the query): `Some(x)` rescores the best `k * x` quantized
        /// candidates with `f32` (`0` = none, quantized scores are
        /// returned); `None` = the field's configured default.
        oversample: Option<f32>,
        /// DiskBBQ (`bbq_disk`) `visit_percentage` of each segment's vectors
        /// (`0`: ES's dynamic ratio; `None`: the field's
        /// `default_visit_percentage`). Other index types ignore it.
        visit_percentage: Option<f32>,
    },
    /// ES `function_score`: combine `query`'s score with functions of the
    /// doc's values.
    FunctionScore {
        query: Box<Query>,
        functions: Vec<ScoreFunction>,
        score_mode: ScoreMode,
        boost_mode: BoostMode,
        /// Cap on the combined function value (`f32::MAX` = none).
        max_boost: f32,
        /// Docs whose final score is below this are dropped.
        min_score: Option<f32>,
    },
    /// A Lucene span query (ES `span_*`).
    Span(SpanQuery),
    /// ES `intervals` on `field`, executed through the span machinery.
    Intervals { field: u16, rule: IntervalsRule },
    /// Every doc, score 1.
    MatchAll,
    /// No doc.
    MatchNone,
    /// ES `nested` (Lucene `ToParentBlockJoinQuery`): `query` runs over the
    /// nested docs of `path`; each parent matches when any of its children
    /// does, scored by `score_mode` over the matching children. As in ES the
    /// parent level is the query's context: root documents at top level,
    /// the enclosing nested path's docs inside another `Nested` (so a
    /// top-level `nested` on `a.b` joins straight to the root documents).
    /// See [`Query::nested_inner_hits`] for ES `inner_hits`.
    Nested { path: String, query: Box<Query>, score_mode: NestedScoreMode, filter: bool },
    /// Docs with a geo point of `field` in the box (`left > right` crosses
    /// the dateline). Constant score.
    GeoBoundingBox { field: u16, top: f64, left: f64, bottom: f64, right: f64 },
    /// Docs with a geo point of `field` within `meters` (haversine / ES
    /// `arc`) of `(lat, lon)`. Constant score.
    GeoDistance { field: u16, lat: f64, lon: f64, meters: f64 },
    /// Docs with a geo point of `field` inside the polygon (`(lat, lon)`
    /// vertices). Constant score.
    GeoPolygon { field: u16, points: Vec<(f64, f64)> },
    /// An engine-supplied node (scripts, ...) that builds its own per-leaf
    /// scorer; see [`CustomQuery`].
    Custom(Arc<dyn CustomQuery>),
}

/// Extension point for query nodes implemented outside this crate (the
/// Elasticsearch layer's Painless `script` / `script_score` queries).
///
/// The scorer returned by [`CustomQuery::scorer`] is only ever driven while
/// `leaf` is still borrowed by the search that built it (scorers never
/// outlive a search), which implementors may rely on.
pub trait CustomQuery: Send + Sync + std::fmt::Debug {
    /// Per-leaf scorer (`None` = no match on this leaf).
    fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>>;
    /// Scorer enumerating the nested children for `inner_hits` (defaults to
    /// [`CustomQuery::scorer`]; nested kNN matches only the best child per
    /// parent but lists every candidate child).
    fn inner_hits_scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        self.scorer(leaf, ctx)
    }
    /// Wrapped sub-queries (their terms feed the statistics pass).
    fn children(&self) -> Vec<&Query> {
        Vec::new()
    }
    /// The number of docs of `leaf` matching, when known without a scan
    /// (callers only ask for leaves without deletes or nested docs).
    fn exact_count(&self, _leaf: &LeafReader<'_>) -> Option<u64> {
        None
    }
    /// The number of docs of `leaf` matching, capped at `limit`
    /// (`min(count, limit)`, exact), when the query finds it without
    /// producing its matches in doc order (a bounded total next to a few
    /// hits). Callers only ask for leaves without deletes or nested docs.
    fn count_at_least(&self, _leaf: &LeafReader<'_>, _limit: u64) -> Option<u64> {
        None
    }
    /// The query as a point predicate on one geo field (see
    /// [`Query::geo_pred`]).
    fn geo_pred(&self) -> Option<(u16, crate::geo::GeoPred)> {
        None
    }
    /// Whether the scorer is a geo predicate scorer that honours
    /// [`crate::query::geo_scorer::with_wanted`] without having a
    /// [`CustomQuery::geo_pred`] (e.g. a `geo_shape` field's point tree).
    fn takes_geo_hint(&self) -> bool {
        false
    }
    /// True if every match scores the same constant.
    fn is_constant_score(&self) -> bool {
        false
    }
    /// Every match has the same score and enumeration may safely stop
    /// after the hit/count threshold. Kept opt-in for custom scripts whose
    /// later documents may still produce errors.
    fn scores_uniformly(&self) -> bool { false }
    /// Short name (explanations).
    fn name(&self) -> &'static str {
        "custom"
    }
    /// The concrete query, for callers that inspect specific query types
    /// (highlighters). `None` by default.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }
    /// Explanation of a matching doc's `score`.
    fn explain(&self, _leaf: &LeafReader<'_>, _ctx: &SearchContext, _doc: DocId, score: f32) -> Explanation {
        Explanation::leaf(score, format!("{} query match", self.name()))
    }
    /// Whether [`CustomQuery::scorer_b`] / [`CustomQuery::explain_b`] fold an
    /// enclosing `BoostQuery`'s boost into the term weights (as Lucene does)
    /// rather than scaling the finished scores.
    fn absorbs_boost(&self) -> bool {
        false
    }
    /// [`CustomQuery::scorer`] under an enclosing `boost`.
    fn scorer_b(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, boost: f32) -> Option<Box<dyn Scorer>> {
        let s = self.scorer(leaf, ctx)?;
        Some(if boost == 1.0 { s } else { Box::new(Scaled::new(s, boost)) })
    }
    /// [`CustomQuery::explain`] under an enclosing `boost` (only called when
    /// [`CustomQuery::absorbs_boost`]); `score` includes the boost.
    fn explain_b(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, doc: DocId, score: f32, _boost: f32) -> Explanation {
        self.explain(leaf, ctx, doc, score)
    }
}

/// Builder for [`Query::Bool`] (see [`Query::bool_query`]).
#[derive(Clone, Debug, Default)]
pub struct BoolBuilder {
    must: Vec<Query>,
    should: Vec<Query>,
    must_not: Vec<Query>,
    filter: Vec<Query>,
    msm: MinShouldMatch,
}

impl BoolBuilder {
    /// Add a required, scoring clause.
    pub fn must(mut self, q: Query) -> Self {
        self.must.push(q);
        self
    }
    /// Add an optional, scoring clause.
    pub fn should(mut self, q: Query) -> Self {
        self.should.push(q);
        self
    }
    /// Add an optional clause with its own boost.
    pub fn should_boosted(self, q: Query, boost: f32) -> Self {
        self.should(q.boost(boost))
    }
    /// Add a required, non-scoring clause.
    pub fn filter(mut self, q: Query) -> Self {
        self.filter.push(q);
        self
    }
    /// Add an excluding clause.
    pub fn must_not(mut self, q: Query) -> Self {
        self.must_not.push(q);
        self
    }
    /// Set `minimum_should_match`.
    pub fn minimum_should_match(mut self, m: impl Into<MinShouldMatch>) -> Self {
        self.msm = m.into();
        self
    }
    /// The query.
    pub fn build(self) -> Query {
        Query::Bool { must: self.must, should: self.should, must_not: self.must_not, filter: self.filter, minimum_should_match: self.msm }
    }
}

impl Query {
    /// A text term query.
    pub fn term(field: u16, text: &str) -> Self {
        Query::Term(Term::new_text(field, text))
    }

    /// A term query with the text lowercased (ES `case_insensitive`); only
    /// a true case-insensitive match when the field's analyzer lowercases.
    pub fn term_case_insensitive(field: u16, text: &str) -> Self {
        Query::Term(Term::new_text(field, &text.to_lowercase()))
    }

    /// An exact phrase.
    pub fn phrase(field: u16, words: &[&str]) -> Self {
        Query::Phrase {
            terms: words.iter().map(|w| Term::new_text(field, w)).collect(),
            slop: 0,
        }
    }

    /// A multi-phrase: `positions[i]` lists the words accepted at position `i`.
    pub fn multi_phrase(field: u16, positions: &[&[&str]], slop: u32) -> Self {
        Query::MultiPhrase {
            field,
            positions: positions.iter().map(|alts| alts.iter().map(|w| Term::new_text(field, w)).collect()).collect(),
            slop,
        }
    }

    /// Any of `terms` (ES `terms`), score 1.
    pub fn term_set(field: u16, terms: &[&str]) -> Self {
        Query::TermSet { field, terms: terms.iter().map(|t| t.as_bytes().to_vec()).collect(), boost: 1.0 }
    }

    /// Prefix query with constant-score rewrite and no expansion limit.
    pub fn prefix(field: u16, prefix: &str) -> Self {
        Query::Prefix { field, prefix: prefix.as_bytes().to_vec(), max_expansions: usize::MAX, case_insensitive: false, rewrite: MultiTermRewrite::ConstantScore }
    }

    /// Wildcard query with constant-score rewrite and no expansion limit.
    pub fn wildcard(field: u16, pattern: &str) -> Self {
        Query::Wildcard { field, pattern: pattern.into(), max_expansions: usize::MAX, case_insensitive: false, rewrite: MultiTermRewrite::ConstantScore }
    }

    /// Regexp query with constant-score rewrite and no expansion limit.
    pub fn regexp(field: u16, pattern: &str) -> Self {
        Query::Regexp { field, pattern: pattern.into(), case_insensitive: false, max_expansions: usize::MAX, rewrite: MultiTermRewrite::ConstantScore }
    }

    /// Fuzzy query with ES defaults (`AUTO`, prefix 0, 50 expansions,
    /// transpositions, top-terms-blended rewrite).
    pub fn fuzzy(field: u16, text: &str) -> Self {
        Query::Fuzzy {
            field,
            text: text.into(),
            fuzziness: Fuzziness::Auto,
            prefix_length: 0,
            transpositions: true,
            max_expansions: 50,
            rewrite: MultiTermRewrite::TopTermsBlendedFreqs(50),
        }
    }

    /// Term range with constant-score rewrite.
    pub fn term_range(field: u16, lower: Bound<Vec<u8>>, upper: Bound<Vec<u8>>) -> Self {
        Query::TermRange { field, lower, upper, rewrite: MultiTermRewrite::ConstantScore }
    }

    /// k-NN with `num_candidates = 10 * k`, no filter, threshold or boost.
    pub fn knn(field: u16, vector: Vec<f32>, k: usize) -> Self {
        Query::Knn { field, vector, k, num_candidates: k.saturating_mul(10), filter: None, similarity_threshold: None, boost: 1.0, oversample: None, visit_percentage: None }
    }

    /// `function_score` with default modes (multiply / multiply), no cap.
    pub fn function_score(query: Query, functions: Vec<ScoreFunction>) -> Self {
        Query::FunctionScore {
            query: Box::new(query),
            functions,
            score_mode: ScoreMode::Multiply,
            boost_mode: BoostMode::Multiply,
            max_boost: f32::MAX,
            min_score: None,
        }
    }

    /// Start a boolean query.
    pub fn bool_query() -> BoolBuilder {
        BoolBuilder::default()
    }

    /// Wrap in a [`Query::Boost`].
    pub fn boost(self, boost: f32) -> Self {
        Query::Boost {
            query: Box::new(self),
            boost,
        }
    }

    /// Set the rewrite method of a multi-term query (no-op on others).
    pub fn with_rewrite(mut self, r: MultiTermRewrite) -> Self {
        match &mut self {
            Query::Prefix { rewrite, .. }
            | Query::Wildcard { rewrite, .. }
            | Query::Regexp { rewrite, .. }
            | Query::Fuzzy { rewrite, .. }
            | Query::TermRange { rewrite, .. } => *rewrite = r,
            _ => {}
        }
        self
    }

    /// Set `max_expansions` of a multi-term query (no-op on others).
    pub fn with_max_expansions(mut self, n: usize) -> Self {
        match &mut self {
            Query::Prefix { max_expansions, .. }
            | Query::Wildcard { max_expansions, .. }
            | Query::Regexp { max_expansions, .. }
            | Query::Fuzzy { max_expansions, .. } => *max_expansions = n,
            _ => {}
        }
        self
    }

    /// Turn on `case_insensitive` for prefix / wildcard / regexp, or
    /// lowercase a term query's text.
    pub fn case_insensitive(mut self) -> Self {
        match &mut self {
            Query::Prefix { case_insensitive, .. } | Query::Wildcard { case_insensitive, .. } | Query::Regexp { case_insensitive, .. } => {
                *case_insensitive = true
            }
            Query::Term(t) => {
                if let Ok(s) = std::str::from_utf8(&t.text) {
                    t.text = s.to_lowercase().into_bytes();
                }
            }
            _ => {}
        }
        self
    }

    /// Direct sub-queries (bool clauses, wrapped queries, filters).
    pub fn children(&self) -> Vec<&Query> {
        match self {
            Query::Bool { must, should, must_not, filter, .. } => must.iter().chain(should).chain(must_not).chain(filter).collect(),
            Query::ConstScore { query, .. } | Query::Boost { query, .. } => vec![&**query],
            Query::DisMax { queries, .. } => queries.iter().collect(),
            Query::Knn { filter: Some(f), .. } => vec![&**f],
            Query::FunctionScore { query, functions, .. } => {
                let mut v: Vec<&Query> = vec![&**query];
                fn walk<'q>(f: &'q ScoreFunction, v: &mut Vec<&'q Query>) {
                    match f {
                        ScoreFunction::Filtered { filter, function } => {
                            v.push(&**filter);
                            walk(function, v);
                        }
                        ScoreFunction::Weighted { function, .. } => walk(function, v),
                        _ => {}
                    }
                }
                functions.iter().for_each(|f| walk(f, &mut v));
                v
            }
            Query::Custom(c) => c.children(),
            Query::Nested { query, .. } => vec![&**query],
            _ => Vec::new(),
        }
    }

    /// Clauses this query counts against the max clause count, as ES's
    /// `MaxClauseCountQueryVisitor` counts them: each leaf once (or
    /// `leaf_weight`, for rewritten expansions), a phrase once per term, a
    /// term or point set once, and `function_score` / `knn` filters not at all.
    pub fn clause_count(&self, leaf_weight: &dyn Fn(&Query) -> usize) -> usize {
        let mut pending = vec![self];
        let mut clauses = 0usize;
        while let Some(q) = pending.pop() {
            let add = match q {
                Query::Bool { must, should, must_not, filter, .. } => { pending.extend(must.iter().chain(should).chain(must_not).chain(filter)); 0 }
                Query::ConstScore { point_set: true, .. } => 1,
                Query::ConstScore { query, .. } | Query::Boost { query, .. } | Query::Nested { query, .. } | Query::FunctionScore { query, .. } => { pending.push(query); 0 }
                Query::DisMax { queries, .. } => { pending.extend(queries); 0 }
                Query::Phrase { terms, .. } => terms.len().max(1),
                Query::MultiPhrase { positions, .. } => positions.iter().map(Vec::len).sum::<usize>().max(1),
                Query::Custom(c) => {
                    let children = c.children();
                    if children.is_empty() { leaf_weight(q) } else { pending.extend(children); 0 }
                }
                _ => leaf_weight(q),
            };
            clauses = clauses.saturating_add(add);
        }
        clauses
    }

    /// Visit this node and every descendant.
    pub fn visit<'q>(&'q self, f: &mut impl FnMut(&'q Query)) {
        f(self);
        for c in self.children() {
            c.visit(f);
        }
    }

    /// Terms whose index-wide statistics scoring needs.
    pub fn collect_terms(&self, out: &mut Vec<Term>) {
        self.visit(&mut |q| match q {
            Query::Term(t) => out.push(t.clone()),
            Query::Phrase { terms, .. } => out.extend(terms.iter().cloned()),
            Query::MultiPhrase { positions, .. } => positions.iter().for_each(|p| out.extend(p.iter().cloned())),
            Query::Span(sq) => sq.collect_terms(out),
            Query::Intervals { field, rule } => rule.to_span(*field).collect_terms(out),
            _ => {}
        });
    }

    /// Fields whose statistics (average length) scoring needs.
    pub fn collect_fields(&self, out: &mut Vec<u16>) {
        let mut terms = Vec::new();
        self.collect_terms(&mut terms);
        for f in self.fields_with_terms(&terms) {
            if !out.contains(&f) {
                out.push(f);
            }
        }
    }

    /// [`Query::collect_fields`] given this query's already collected terms.
    fn fields_with_terms(&self, terms: &[Term]) -> Vec<u16> {
        let mut out = Vec::new();
        let mut fields: Vec<u16> = terms.iter().map(|t| t.field).collect();
        self.visit(&mut |q| match q {
            Query::MultiPhrase { field, .. } | Query::Intervals { field, .. } => fields.push(*field),
            Query::Span(sq) => fields.extend(sq.field()),
            _ => {
                if let Some(spec) = q.multi_term_spec() {
                    if spec.rewrite.is_scoring() {
                        fields.push(spec.field);
                    }
                }
            }
        });
        for f in fields {
            if !out.contains(&f) {
                out.push(f);
            }
        }
        out
    }

    /// True if this query only ever produces constant scores (so sort by
    /// score is meaningless and the searcher can skip scoring).
    pub fn is_constant_score(&self) -> bool {
        match self {
            Query::Term(_)
            | Query::Phrase { .. }
            | Query::MultiPhrase { .. }
            | Query::Knn { .. }
            | Query::FunctionScore { .. }
            | Query::Span(_)
            | Query::Intervals { .. } => false,
            Query::Prefix { rewrite, .. }
            | Query::Wildcard { rewrite, .. }
            | Query::Regexp { rewrite, .. }
            | Query::Fuzzy { rewrite, .. }
            | Query::TermRange { rewrite, .. } => !rewrite.is_scoring(),
            Query::Bool { must, should, .. } => must.iter().chain(should).all(|q| q.is_constant_score()),
            Query::Boost { query, .. } => query.is_constant_score(),
            Query::DisMax { queries, .. } => queries.iter().all(|q| q.is_constant_score()),
            Query::Custom(c) => c.is_constant_score(),
            Query::Nested { query, score_mode, .. } => {
                *score_mode == NestedScoreMode::None || (*score_mode != NestedScoreMode::Sum && query.is_constant_score())
            }
            _ => true,
        }
    }

    /// Stricter than [`Query::is_constant_score`]: every matching doc gets
    /// the *same* score, so the top hits are simply the first ones in doc
    /// order and collection can stop early.
    pub fn scores_uniformly(&self) -> bool {
        match self {
            Query::MatchAll
            | Query::MatchNone
            | Query::TermSet { .. }
            | Query::RangeI64 { .. }
            | Query::RangeU64 { .. }
            | Query::RangeF64 { .. }
            | Query::RangeDate { .. }
            | Query::Exists { .. }
            | Query::GeoBoundingBox { .. }
            | Query::GeoDistance { .. }
            | Query::GeoPolygon { .. }
            | Query::ConstScore { .. } => true,
            Query::Prefix { rewrite, .. }
            | Query::Wildcard { rewrite, .. }
            | Query::Regexp { rewrite, .. }
            | Query::Fuzzy { rewrite, .. }
            | Query::TermRange { rewrite, .. } => !rewrite.is_scoring(),
            Query::Boost { query, .. } => query.scores_uniformly(),
            // Without `should` clauses every match matches every `must`
            // clause: two uniform clauses sum to the same score for all
            // (a two-term float sum does not depend on the order; longer
            // sums might, were a scorer to reorder its clauses).
            Query::Bool { must, should, .. } => (must.len() + should.len() <= 1 || (should.is_empty() && must.len() == 2)) && must.iter().chain(should).all(|q| q.scores_uniformly()),
            Query::Custom(c) => c.scores_uniformly(),
            _ => false,
        }
    }

    /// Short name of the variant (explanations).
    pub fn kind_name(&self) -> &'static str {
        match self {
            Query::Term(_) => "term",
            Query::TermSet { .. } => "terms",
            Query::Phrase { .. } => "phrase",
            Query::MultiPhrase { .. } => "multi_phrase",
            Query::Prefix { .. } => "prefix",
            Query::Wildcard { .. } => "wildcard",
            Query::Regexp { .. } => "regexp",
            Query::Fuzzy { .. } => "fuzzy",
            Query::TermRange { .. } => "term_range",
            Query::RangeI64 { .. } | Query::RangeU64 { .. } | Query::RangeF64 { .. } | Query::RangeDate { .. } => "range",
            Query::Exists { .. } => "exists",
            Query::Bool { .. } => "bool",
            Query::ConstScore { .. } => "constant_score",
            Query::Boost { .. } => "boost",
            Query::DisMax { .. } => "dis_max",
            Query::Knn { .. } => "knn",
            Query::FunctionScore { .. } => "function_score",
            Query::Span(_) => "span",
            Query::Intervals { .. } => "intervals",
            Query::MatchAll => "match_all",
            Query::MatchNone => "match_none",
            Query::Custom(c) => c.name(),
            Query::Nested { .. } => "nested",
            Query::GeoBoundingBox { .. } => "geo_bounding_box",
            Query::GeoDistance { .. } => "geo_distance",
            Query::GeoPolygon { .. } => "geo_polygon",
        }
    }

    /// Identity of this node within one search (scoring-rewrite expansions
    /// are keyed by it).
    fn node_key(&self) -> usize {
        self as *const Query as usize
    }

    /// The enumeration this node performs, for multi-term queries.
    fn multi_term_spec(&self) -> Option<MultiTermSpec> {
        let lower_ci = |b: &[u8]| String::from_utf8_lossy(b).to_lowercase().into_bytes();
        Some(match self {
            Query::Prefix { field, prefix, max_expansions, case_insensitive, rewrite } => {
                let p = if *case_insensitive { lower_ci(prefix) } else { prefix.clone() };
                MultiTermSpec::new(*field, Box::new(PrefixMatcher(p)), *max_expansions, *rewrite)
            }
            Query::Wildcard { field, pattern, max_expansions, case_insensitive, rewrite } => {
                let p = if *case_insensitive { pattern.to_lowercase() } else { pattern.clone() };
                MultiTermSpec::new(*field, Box::new(WildcardMatcher::new(&p)), *max_expansions, *rewrite)
            }
            Query::Regexp { field, pattern, case_insensitive, max_expansions, rewrite } => {
                let m = shared_regexp(pattern, *case_insensitive)?;
                MultiTermSpec::new(*field, Box::new(SharedRegexp(m)), *max_expansions, *rewrite)
            }
            Query::Fuzzy { field, text, fuzziness, prefix_length, transpositions, max_expansions, rewrite } => {
                let edits = fuzziness.edits(text);
                let m = LevenshteinMatcher::new(text, edits, *prefix_length, *transpositions);
                let mut spec = MultiTermSpec::new(*field, Box::new(m), *max_expansions, *rewrite);
                spec.fuzzy = Some((text.clone(), *transpositions, edits));
                spec
            }
            Query::TermRange { field, lower, upper, rewrite } => {
                let mut spec = MultiTermSpec::new(*field, Box::new(RangeMatcher { lower: lower.clone(), upper: upper.clone() }), usize::MAX, *rewrite);
                spec.lower = match lower {
                    Bound::Included(l) | Bound::Excluded(l) => Some(l.clone()),
                    Bound::Unbounded => None,
                };
                spec.upper = match upper {
                    Bound::Excluded(u) => Some(u.clone()),
                    Bound::Included(u) => {
                        let mut v = u.clone();
                        v.push(0);
                        Some(v)
                    }
                    Bound::Unbounded => None,
                };
                spec
            }
            _ => return None,
        })
    }
}

/// What a multi-term query enumerates and how it scores.
/// A compiled [`RegexpMatcher`] shared by every leaf and search using the
/// same pattern ([`shared_regexp`]).
struct SharedRegexp(Arc<RegexpMatcher>);

impl TermMatcher for SharedRegexp {
    fn matches(&self, term: &[u8]) -> bool {
        self.0.matches(term)
    }
    fn literal_prefix(&self) -> &[u8] {
        self.0.literal_prefix()
    }
}

/// The compiled matcher of (`pattern`, `case_insensitive`), from a small
/// process-wide cache: a regexp (a case-insensitive `term` or `prefix` on a
/// keyword is one) was compiled to an automaton for every leaf of every
/// search (EQL's `startsWith` stages: ~5% of a sequence query). `None`: the
/// pattern does not compile (not cached).
fn shared_regexp(pattern: &str, case_insensitive: bool) -> Option<Arc<RegexpMatcher>> {
    const ENTRIES: usize = 256;
    static CACHE: std::sync::LazyLock<parking_lot::Mutex<HashMap<(String, bool), Arc<RegexpMatcher>>>> = std::sync::LazyLock::new(Default::default);
    if let Some(m) = CACHE.lock().get(&(pattern.to_string(), case_insensitive)) {
        return Some(m.clone());
    }
    let m = Arc::new(RegexpMatcher::new(pattern, RegexpFlags { case_insensitive }).ok()?);
    let mut cache = CACHE.lock();
    if cache.len() >= ENTRIES {
        cache.clear();
    }
    cache.insert((pattern.to_string(), case_insensitive), m.clone());
    Some(m)
}

struct MultiTermSpec {
    field: u16,
    matcher: Box<dyn TermMatcher>,
    lower: Option<Vec<u8>>,
    upper: Option<Vec<u8>>,
    max_expansions: usize,
    rewrite: MultiTermRewrite,
    /// Fuzzy: (query text, transpositions, max edits).
    fuzzy: Option<(String, bool, u8)>,
}

impl MultiTermSpec {
    fn new(field: u16, matcher: Box<dyn TermMatcher>, max_expansions: usize, rewrite: MultiTermRewrite) -> Self {
        Self { field, matcher, lower: None, upper: None, max_expansions, rewrite, fuzzy: None }
    }

    fn expand(&self, leaves: &[LeafReader<'_>], clause_limit: Option<usize>) -> Result<Vec<ExpandedTerm>, ()> {
        let dist = self.fuzzy.as_ref().map(|(text, tr, max)| {
            let (text, tr, max) = (text.clone(), *tr, *max);
            move |b: &[u8]| std::str::from_utf8(b).map(|t| edit_distance(&text, t, tr) as u8).unwrap_or(max)
        });
        let qlen = self.fuzzy.as_ref().map(|(t, _, _)| t.chars().count()).unwrap_or(1);
        let bounds = (self.lower.as_deref(), self.upper.as_deref());
        let d: Option<&dyn Fn(&[u8]) -> u8> = dist.as_ref().map(|f| f as &dyn Fn(&[u8]) -> u8);
        expand_global_bounded(leaves, self.field, &*self.matcher, bounds, self.max_expansions, self.rewrite, d, qlen, clause_limit)
    }
}

struct PrefixMatcher(Vec<u8>);
impl TermMatcher for PrefixMatcher {
    fn matches(&self, term: &[u8]) -> bool {
        term.starts_with(&self.0)
    }
    fn literal_prefix(&self) -> &[u8] {
        &self.0
    }
}

struct AllTerms;
impl TermMatcher for AllTerms {
    fn matches(&self, _: &[u8]) -> bool {
        true
    }
    fn literal_prefix(&self) -> &[u8] {
        &[]
    }
}

struct RangeMatcher {
    lower: Bound<Vec<u8>>,
    upper: Bound<Vec<u8>>,
}
impl TermMatcher for RangeMatcher {
    fn matches(&self, t: &[u8]) -> bool {
        (match &self.lower {
            Bound::Included(l) => t >= l.as_slice(),
            Bound::Excluded(l) => t > l.as_slice(),
            Bound::Unbounded => true,
        }) && (match &self.upper {
            Bound::Included(u) => t <= u.as_slice(),
            Bound::Excluded(u) => t < u.as_slice(),
            Bound::Unbounded => true,
        })
    }
    fn literal_prefix(&self) -> &[u8] {
        &[]
    }
}

// Node policy changes publish here; each SearchContext captures it once.
static MAX_CLAUSE_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);
pub fn set_max_clause_count(max: usize) { MAX_CLAUSE_COUNT.store(max, std::sync::atomic::Ordering::Relaxed); }
fn max_clause_count() -> usize { MAX_CLAUSE_COUNT.load(std::sync::atomic::Ordering::Relaxed) }

/// Index-wide statistics computed once per search, plus the similarity
/// configuration and global multi-term expansions.
pub struct SearchContext {
    /// Legacy BM25 parameters (used when built through [`SearchContext::for_query`]).
    pub params: Bm25Params,
    /// Default similarity.
    pub similarity: Arc<dyn Similarity>,
    /// Per-field similarity overrides.
    pub field_similarity: HashMap<u16, Arc<dyn Similarity>>,
    /// Index-wide term statistics.
    pub term_stats: HashMap<Term, TermStats>,
    /// Index-wide field statistics.
    pub field_stats: HashMap<u16, FieldStats>,
    /// Scoring-rewrite expansions, keyed by query node identity.
    pub expansions: HashMap<usize, Arc<Vec<ExpandedTerm>>>,
    span_expansions: HashMap<usize, Arc<SpanQuery>>,
    interval_expansions: HashMap<(usize,usize), Arc<SpanQuery>>,
    interval_error: parking_lot::Mutex<Option<String>>,
    /// Global k-NN results: `(knn node, leaf id)` -> that leaf's share of
    /// the index-wide top-k, sorted by doc.
    pub knn_hits: HashMap<(usize, usize), Vec<(DocId, f32)>>,
    /// Vector distance / similarity computations made by the global k-NN
    /// searches (ES profile `vector_operations_count`).
    pub knn_vector_ops: u64,
    /// Fallible vector execution must fail the shard, not become an empty hit set.
    pub execution_error: parking_lot::Mutex<Option<String>>,
    memory_error: parking_lot::Mutex<Option<String>>,
    pub max_clause_count: usize,
    clause_error: parking_lot::Mutex<Option<bool>>,
    pub deadline: Option<std::time::Instant>,
    /// The search task's cancellation (the thread's search token when the
    /// context was built, see [`crate::cancel::search_scope`]).
    pub cancel: Option<Arc<crate::cancel::Token>>,
    pub vector_timed_out: std::sync::atomic::AtomicBool,
    /// Per term (sorted by field, then text): its index-wide statistics and
    /// the per-leaf dictionary lookups made by the stats pass (indexed like
    /// `leaf_ids`), reused by scorers instead of looking each term up again.
    pub term_infos: Vec<(Term, TermStats, Vec<Option<PostingsLocation>>)>,
    /// [`LeafReader::id`] of the leaves the context was built over.
    pub leaf_ids: Vec<usize>,
    /// Numeric range bitsets built during this request, per leaf (see
    /// [`Query::numeric_range_bits_cached`]).
    range_cache: parking_lot::Mutex<std::collections::HashMap<RangeKey, Arc<crate::admission::Allocation<Vec<u64>>>>>,
    /// Dense filter conjunctions kept while a leaf's query builds several
    /// scorers (see [`SearchContext::dense_scope`]), by leaf id.
    dense_scopes: parking_lot::Mutex<std::collections::HashMap<usize, DenseScope>>,
    /// First doc a `function_score` gave a negative or NaN score (ES fails
    /// the shard: "function score query returned an invalid score").
    pub invalid_score: Arc<parking_lot::Mutex<Option<(DocId, f32)>>>,
    /// Rescore steps: the previous pass's window, per leaf id, as
    /// `(doc, score)` sorted by doc (ES `KnnScoreDocQuery` over the top N).
    pub prev_pass: HashMap<usize, Vec<(DocId, f32)>>,
}

impl Default for SearchContext {
    fn default() -> Self {
        Self {
            params: Bm25Params::default(),
            similarity: Arc::new(Bm25::default()),
            field_similarity: HashMap::new(),
            term_stats: HashMap::new(),
            field_stats: HashMap::new(),
            expansions: HashMap::new(),
            span_expansions: HashMap::new(),
            interval_expansions: HashMap::new(),
            interval_error: parking_lot::Mutex::new(None),
            knn_hits: HashMap::new(),
            knn_vector_ops: 0,
            execution_error: parking_lot::Mutex::new(None),
            memory_error: parking_lot::Mutex::new(None),
            max_clause_count: max_clause_count(),
            clause_error: parking_lot::Mutex::new(None),
            deadline: None,
            cancel: None,
            vector_timed_out: std::sync::atomic::AtomicBool::new(false),
            term_infos: Vec::new(),
            leaf_ids: Vec::new(),
            range_cache: parking_lot::Mutex::new(std::collections::HashMap::new()),
            dense_scopes: parking_lot::Mutex::new(std::collections::HashMap::new()),
            invalid_score: Arc::default(),
            prev_pass: HashMap::new(),
        }
    }
}

impl SearchContext {
    /// Whether the search must stop: its deadline passed or its task was
    /// cancelled (for amortised checks, not per document).
    #[inline]
    pub fn stopped(&self) -> bool {
        self.deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) || self.cancel.as_ref().is_some_and(|c| c.is_cancelled())
    }

    /// Preserve admission failures across infallible scorer interfaces. The
    /// outer search must call execution_result before returning any hits.
    pub fn record_memory_error(&self, reason: impl Into<String>) {
        self.memory_error.lock().get_or_insert(reason.into());
    }
    pub(crate) fn inherit_errors(&self, other: &Self) {
        if let Some(error)=other.memory_error.lock().clone() { self.record_memory_error(error); }
        if let Some(error) = other.interval_error.lock().clone() { self.interval_error.lock().get_or_insert(error); }
        if let Some(error) = *other.clause_error.lock() { *self.clause_error.lock()=Some(error); }
        if let Some(error) = other.execution_error.lock().clone() { self.execution_error.lock().get_or_insert(error); }
    }
    pub fn execution_result(&self) -> crate::error::Result<()> {
        if let Some(error)=self.memory_error.lock().clone() { return Err(crate::error::RusticError::Memory(error)); }
        if let Some(error) = self.interval_error.lock().clone() { return Err(crate::error::RusticError::IntervalExpansion(error)); }
        if let Some(nested) = *self.clause_error.lock() { return Err(crate::error::RusticError::TooManyClauses { max:self.max_clause_count,nested }); }
        match self.execution_error.lock().clone() {
            Some(error) => Err(crate::error::RusticError::InvalidQuery(error)),
            None => Ok(()),
        }
    }

    /// Gather every statistic `query` needs from `leaves`, scoring with
    /// BM25(`params`).
    pub fn for_query(query: &Query, leaves: &[LeafReader<'_>], params: Bm25Params) -> Self {
        let mut ctx = Self::with_similarity(query, leaves, Arc::new(Bm25::from_params(params)), HashMap::new());
        ctx.params = params;
        ctx
    }

    pub fn for_query_deadline(query: &Query, leaves: &[LeafReader<'_>], params: Bm25Params,
        deadline: Option<std::time::Instant>) -> Self {
        let mut ctx = Self::with_similarity_deadline(query, leaves, Arc::new(Bm25::from_params(params)), HashMap::new(), deadline);
        ctx.params = params;
        ctx
    }

    /// Gather every statistic `query` needs from `leaves`, scoring with
    /// `similarity` (overridden per field by `field_similarity`).
    pub fn with_similarity(
        query: &Query,
        leaves: &[LeafReader<'_>],
        similarity: Arc<dyn Similarity>,
        field_similarity: HashMap<u16, Arc<dyn Similarity>>,
    ) -> Self {
        Self::with_similarity_deadline(query, leaves, similarity, field_similarity, None)
    }

    /// Term-by-leaf dictionary lookups worth running in parallel.
    const PARALLEL_LOOKUPS: usize = 128;

    pub fn with_similarity_deadline(query: &Query, leaves: &[LeafReader<'_>],
        similarity: Arc<dyn Similarity>, field_similarity: HashMap<u16, Arc<dyn Similarity>>,
        deadline: Option<std::time::Instant>) -> Self {
        Self::with_similarity_deadline_limit(query, leaves, similarity, field_similarity, deadline, max_clause_count())
    }

    pub fn with_similarity_deadline_limit(query: &Query, leaves: &[LeafReader<'_>],
        similarity: Arc<dyn Similarity>, field_similarity: HashMap<u16, Arc<dyn Similarity>>,
        deadline: Option<std::time::Instant>, max_clauses: usize) -> Self {
        let started = crate::query::stats::start();
        let mut terms = Vec::new();
        query.collect_terms(&mut terms);
        let fields = query.fields_with_terms(&terms);
        let mut ctx = SearchContext {
            params: similarity.as_bm25().map(|b| b.params()).unwrap_or_default(),
            similarity,
            field_similarity,
            term_stats: HashMap::with_capacity(terms.len()),
            field_stats: HashMap::with_capacity(fields.len()),
            expansions: HashMap::new(),
            span_expansions: HashMap::new(),
            interval_expansions: HashMap::new(),
            interval_error: parking_lot::Mutex::new(None),
            knn_hits: HashMap::new(),
            knn_vector_ops: 0,
            execution_error: parking_lot::Mutex::new(None),
            memory_error: parking_lot::Mutex::new(None),
            max_clause_count: max_clauses,
            clause_error: parking_lot::Mutex::new(None),
            deadline,
            cancel: crate::cancel::current(),
            vector_timed_out: std::sync::atomic::AtomicBool::new(false),
            term_infos: Vec::with_capacity(terms.len()),
            leaf_ids: leaves.iter().map(|l| l.id()).collect(),
            range_cache: parking_lot::Mutex::new(std::collections::HashMap::new()),
            dense_scopes: parking_lot::Mutex::new(std::collections::HashMap::new()),
            invalid_score: Arc::default(),
            prev_pass: HashMap::new(),
        };
        terms.sort_unstable_by(|a, b| a.field.cmp(&b.field).then_with(|| a.text.cmp(&b.text)));
        terms.dedup();
        // One dictionary lookup per term and leaf, before any leaf runs: a
        // multi-field query over a few dozen segments makes hundreds, so
        // they spread over the search pool.
        let stats: Vec<(TermStats, Vec<Option<PostingsLocation>>)> = if terms.len() > 1 && terms.len() * leaves.len() >= Self::PARALLEL_LOOKUPS {
            use rayon::prelude::*;
            crate::query::searcher::in_search_pool(|| terms.par_iter().map(|t| term_stats_with_infos(leaves, t)).collect())
        } else {
            terms.iter().map(|t| term_stats_with_infos(leaves, t)).collect()
        };
        crate::query::stats::add(crate::query::stats::C::term_lookups, (terms.len() * leaves.len()) as u64);
        for (t, (s, infos)) in terms.into_iter().zip(stats) {
            ctx.term_stats.insert(t.clone(), s);
            ctx.term_infos.push((t, s, infos));
        }
        for f in fields {
            ctx.field_stats.insert(f, field_stats(leaves, f));
        }
        let mut knn_nodes = Vec::new();
        let mut multi_term = false;
        query.visit(&mut |q| match q {
            Query::Knn { .. } => knn_nodes.push(q),
            Query::Prefix { .. } | Query::Wildcard { .. } | Query::Regexp { .. } | Query::Fuzzy { .. } | Query::TermRange { .. } => multi_term = true,
            _ => {}
        });
        if multi_term {
            let mut expansions = HashMap::new();
            query.visit(&mut |q| {
                if ctx.clause_error.lock().is_some() { return; }
                if let Some(spec) = q.multi_term_spec() {
                    if spec.rewrite != MultiTermRewrite::ConstantScore {
                        match spec.expand(leaves, Some(ctx.max_clause_count)) {
                            Ok(terms) => { expansions.insert(q.node_key(), Arc::new(terms)); },
                            Err(()) => { *ctx.clause_error.lock()=Some(false); },
                        }
                    }
                }
            });
            ctx.expansions = expansions;
        }
        query.visit(&mut |q| {
            if let Query::Span(span)=q {
                match span.rewrite_bounded(leaves,ctx.max_clause_count) {
                    Ok(span)=> {
                        let mut terms=Vec::new();
                        span.collect_terms(&mut terms);
                        for term in terms { ctx.term_stats.entry(term.clone()).or_insert_with(||crate::query::reader::term_stats(leaves,&term)); }
                        ctx.span_expansions.insert(q.node_key(),Arc::new(span));
                    },
                    Err(())=> { *ctx.clause_error.lock()=Some(false); },
                }
            }
        });
        query.visit(&mut |q| {
            if let Query::Intervals { field,rule } = q {
                for leaf in leaves {
                    match rule.rewrite_bounded(*field,leaf,ctx.max_clause_count) {
                        Ok(span)=> { ctx.interval_expansions.insert((q.node_key(),leaf.id()),Arc::new(span)); },
                        Err(error)=> { ctx.interval_error.lock().get_or_insert(error); break; },
                    }
                }
            }
        });
        let clauses=query.clause_count(&|q| ctx.span_expansions.get(&q.node_key()).map(|span|span.clause_count()).or_else(||ctx.expansions.get(&q.node_key()).map(|terms|terms.len().max(1))).unwrap_or(1));
        if clauses > ctx.max_clause_count && ctx.clause_error.lock().is_none() { *ctx.clause_error.lock()=Some(true); }
        if ctx.execution_result().is_err() { return ctx; }
        for q in knn_nodes {
            let (hits, ops) = q.global_knn(leaves, &ctx);
            ctx.knn_hits.extend(hits);
            ctx.knn_vector_ops += ops;
        }
        crate::query::stats::add_since(crate::query::stats::C::context_ns, started);
        ctx
    }

    /// Postings of `term` on `leaf`, reusing the stats pass's dictionary
    /// lookup when the context saw this term on this leaf.
    #[inline]
    pub fn postings(&self, leaf: &LeafReader<'_>, term: &Term) -> Option<PostingsReader> {
        let id = leaf.id();
        if let Some(li) = self.leaf_ids.iter().position(|&x| x == id) {
            if let Some((_, _, infos)) = self.term_info(term) {
                return match &infos[li] {
                    Some(info) => leaf.postings_with(term, info),
                    None => None,
                };
            }
        }
        leaf.postings(term)
    }

    #[inline]
    fn term_info(&self, term: &Term) -> Option<&(Term, TermStats, Vec<Option<PostingsLocation>>)> {
        let i = self
            .term_infos
            .binary_search_by(|(t, _, _)| t.field.cmp(&term.field).then_with(|| t.text.as_slice().cmp(term.text.as_slice())))
            .ok()?;
        Some(&self.term_infos[i])
    }

    /// The similarity scoring `field`.
    pub fn sim(&self, field: u16) -> &dyn Similarity {
        match self.field_similarity.get(&field) {
            Some(s) => &**s,
            None => &*self.similarity,
        }
    }

    /// Index-wide statistics of `term` (zero if not gathered).
    pub fn stats(&self, term: &Term) -> TermStats {
        match self.term_info(term) {
            Some((_, s, _)) => *s,
            None => self.term_stats.get(term).copied().unwrap_or_default(),
        }
    }

    /// Index-wide statistics of `field` (zero if not gathered).
    pub fn fstats(&self, field: u16) -> FieldStats {
        self.field_stats.get(&field).copied().unwrap_or_default()
    }
}

/// A scorer over an explicit `(doc, score)` list sorted by doc (knn results,
/// expanded multi-term unions with per-term boosts).
pub struct ScoredDocs {
    docs: Vec<(DocId, f32)>,
    pos: usize,
    current: DocId,
}

impl ScoredDocs {
    /// `docs` must be sorted by doc id, unique.
    pub fn new(mut docs: Vec<(DocId, f32)>) -> Self {
        docs.sort_by_key(|(d, _)| *d);
        docs.dedup_by_key(|(d, _)| *d);
        Self {
            docs,
            pos: 0,
            current: TERMINATED,
        }
    }
}

impl DocSet for ScoredDocs {
    fn advance(&mut self) -> DocId {
        self.current = match self.docs.get(self.pos) {
            Some((d, _)) => {
                self.pos += 1;
                *d
            }
            None => TERMINATED,
        };
        self.current
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.current != TERMINATED && self.current >= target {
            return self.current;
        }
        let idx = self.pos + self.docs[self.pos..].partition_point(|(d, _)| *d < target);
        self.pos = idx;
        self.advance()
    }

    fn doc(&self) -> DocId {
        self.current
    }

    fn size_hint(&self) -> u32 {
        self.docs.len() as u32
    }
}

impl Scorer for ScoredDocs {
    fn score(&mut self) -> f32 {
        if self.pos == 0 {
            return 0.0;
        }
        self.docs[self.pos - 1].1
    }
}

/// `max(sub scores) + tie_breaker * sum(other sub scores)`.
pub struct DisMaxScorer {
    inner: Union,
    children_scores: Vec<f32>,
    tie_breaker: f32,
}

impl DocSet for DisMaxScorer {
    fn advance(&mut self) -> DocId {
        self.inner.advance()
    }
    fn seek(&mut self, target: DocId) -> DocId {
        self.inner.seek(target)
    }
    fn doc(&self) -> DocId {
        self.inner.doc()
    }
    fn size_hint(&self) -> u32 {
        self.inner.size_hint()
    }
}

impl DisMaxScorer {
    /// `max(sub scores) + tie_breaker * sum(other sub scores)` of `children`.
    pub fn new(children: Vec<Box<dyn Scorer>>, tie_breaker: f32) -> Self {
        Self { inner: Union::new(children), children_scores: Vec::new(), tie_breaker }
    }

    /// [`Scorer::max_score`] from the clauses' bounds.
    fn combine_bounds(&self, b: &[f32]) -> Option<f32> {
        let t = self.tie_breaker;
        if !(t.is_finite() && t >= 0.0) {
            return None;
        }
        let max = b.iter().copied().fold(0.0f32, f32::max);
        let sum: f32 = b.iter().sum();
        // max(s) + t * (sum(s) - max(s)) = (1 - t) * max(s) + t * sum(s)
        // grows with every clause for t <= 1, so the same expression over
        // the bounds bounds it; past 1, sum(s) - max(s) <= sum(bounds).
        let m = if t <= 1.0 { max + t * (sum - max) } else { max + t * sum };
        m.is_finite().then_some(m)
    }
}

impl Scorer for DisMaxScorer {
    fn max_score(&self) -> Option<f32> {
        let b = self.inner.child_max_scores()?;
        self.combine_bounds(&b)
    }

    fn max_score_up_to(&mut self, up_to: DocId) -> Option<f32> {
        let b = self.inner.child_max_scores_up_to(up_to)?;
        self.combine_bounds(&b)
    }

    fn as_term_groups(&mut self) -> Option<Vec<(f32, Vec<&mut crate::query::bulk::Bm25TermScorer>)>> {
        let tie = self.tie_breaker;
        let terms = self.inner.children_mut()?.iter_mut().map(|c| c.as_bm25_term()).collect::<Option<Vec<_>>>()?;
        Some(vec![(tie, terms)])
    }

    fn collect_top_k_count_limited(&mut self, k: usize, deleted: Option<&[u64]>, count_limit: u64, floor: &crate::query::bulk::Floor<'_>) -> Option<(Vec<crate::query::collector::ScoredDoc>, u64)> {
        crate::query::bulk::try_term_groups_lim(std::iter::once(self), k, deleted, count_limit, floor)
    }

    fn score(&mut self) -> f32 {
        self.children_scores.clear();
        self.inner.child_scores(&mut self.children_scores);
        let max = self.children_scores.iter().copied().fold(0.0f32, f32::max);
        let sum: f32 = self.children_scores.iter().sum();
        max + self.tie_breaker * (sum - max)
    }
}

/// Wrap a scorer as a plain doc set (for `must_not`).
struct AsDocSet(Box<dyn Scorer>);

impl DocSet for AsDocSet {
    fn has_fast_exact_seek(&self) -> bool { self.0.has_fast_exact_seek() }
    fn exact_seek_reads_value(&self) -> bool { self.0.exact_seek_reads_value() }
    fn set_lead_cost(&mut self, lead_cost: u32) { self.0.set_lead_cost(lead_cost) }
    fn seek_exact(&mut self, target: DocId) -> bool { self.0.seek_exact(target) }
    fn advance(&mut self) -> DocId {
        self.0.advance()
    }
    fn seek(&mut self, target: DocId) -> DocId {
        self.0.seek(target)
    }
    fn doc(&self) -> DocId {
        self.0.doc()
    }
    fn size_hint(&self) -> u32 {
        self.0.size_hint()
    }
    fn collect_first_k_bulk(&mut self, k: usize, deleted: Option<&[u64]>) -> Option<(Vec<DocId>, u64)> {
        self.0.collect_first_k_bulk(k, deleted)
    }
    fn fill_all(&mut self, deleted: Option<&[u64]>, out: &mut Vec<DocId>) -> bool {
        self.0.fill_all(deleted, out)
    }
}

fn empty() -> Box<dyn Scorer> {
    Box::new(ConstScorer::new(EmptyDocSet, 0.0))
}

/// A constant scorer over geo query hits.
pub fn hits_scorer(hits: crate::geo::DocHits, score: f32) -> Box<dyn Scorer> {
    match hits {
        crate::geo::DocHits::Sorted(v) => const_docs(v, score),
        h if h.is_empty() => empty(),
        crate::geo::DocHits::Bits(w) => Box::new(ConstScorer::new(super::docset::BitSetDocSet::new(w), score)),
    }
}

/// A constant-score scorer over shared bits (a segment's cached filter
/// matches), read in place.
pub fn shared_bits_scorer(words: Arc<crate::admission::Allocation<Vec<u64>>>, score: f32) -> Box<dyn Scorer> {
    if words.iter().all(|w| *w == 0) {
        return empty();
    }
    Box::new(ConstScorer::new(super::docset::BitSetDocSet::shared(words), score))
}

fn const_docs(docs: Vec<DocId>, score: f32) -> Box<dyn Scorer> {
    if docs.is_empty() {
        empty()
    } else {
        Box::new(ConstScorer::new(VecDocSet::new(docs), score))
    }
}

/// Materialize a scorer's docs as a bitset over `num_docs`.
fn doc_bits(s: &mut dyn Scorer, num_docs: u32) -> Vec<u64> {
    let mut bits = vec![0u64; (num_docs as usize).div_ceil(64)];
    loop {
        let d = s.advance();
        if d == TERMINATED {
            break;
        }
        if let Some(w) = bits.get_mut(d as usize / 64) {
            *w |= 1 << (d % 64);
        }
    }
    bits
}

/// A numeric range query with inclusive bounds (exclusive ones already
/// moved to the adjacent value).
#[derive(Clone, Copy, Debug)]
enum NumericRange {
    I64(u16, i64, i64),
    U64(u16, u64, u64),
    F64(u16, f64, f64),
    /// An empty range (e.g. `> i64::MAX`).
    Empty,
}

/// [`Query::lazy_range_scorer`]'s docs: the range read lazily from doc
/// values, which can still hand a conjunction the segment's cached bitset
/// of the range before it starts ([`DocSet::filter_words`]), so a scored
/// text clause beside it takes its block-max path over those docs.
struct LazyRangeDocs {
    docs: crate::codec::fastfields::RangeDocs<'static, i64>,
    seg: &'static crate::segment::Segment,
    key: crate::segment::FilterCacheKey,
    spec: NumericRange,
    started: bool,
}

impl DocSet for LazyRangeDocs {
    fn advance(&mut self) -> DocId {
        self.started = true;
        self.docs.advance()
    }
    fn seek(&mut self, target: DocId) -> DocId {
        self.started = true;
        self.docs.seek(target)
    }
    fn doc(&self) -> DocId {
        self.docs.doc()
    }
    fn size_hint(&self) -> u32 {
        self.docs.size_hint()
    }
    fn has_fast_exact_seek(&self) -> bool {
        self.docs.has_fast_exact_seek()
    }
    fn exact_seek_reads_value(&self) -> bool {
        self.docs.exact_seek_reads_value()
    }
    fn seek_exact(&mut self, target: DocId) -> bool {
        self.started = true;
        self.docs.seek_exact(target)
    }
    fn filter_words(&mut self) -> Option<Arc<crate::admission::Allocation<Vec<u64>>>> {
        if self.started {
            return None;
        }
        let (seg, spec) = (self.seg, self.spec);
        seg.cached_filter_bits(self.key, || spec.bits(&LeafReader::Segment(seg))).ok()
    }
}

/// Cache key of a range bitset: `(leaf id, kind, field, lo bits, hi bits)`.
type RangeKey = (usize, u8, u16, u64, u64);
/// Segment filter-cache kind of an `exists` presence bitset (numeric range
/// kinds are 0-3).
const EXISTS_FILTER_KIND: u8 = 9;

impl NumericRange {
    fn bits(&self, leaf: &LeafReader<'_>) -> Vec<u64> {
        let empty = || vec![0u64; (leaf.num_docs() as usize).div_ceil(64)];
        match *self {
            NumericRange::I64(f, lo, hi) => leaf.range_i64_bits(f, lo, hi).unwrap_or_else(empty),
            NumericRange::U64(f, lo, hi) => leaf.range_u64_bits(f, lo, hi).unwrap_or_else(empty),
            NumericRange::F64(f, lo, hi) => leaf.range_f64_bits(f, lo, hi).unwrap_or_else(empty),
            NumericRange::Empty => empty(),
        }
    }

    fn key(&self, leaf: usize) -> RangeKey {
        match *self {
            NumericRange::I64(f, lo, hi) => (leaf, 1, f, lo as u64, hi as u64),
            NumericRange::U64(f, lo, hi) => (leaf, 2, f, lo, hi),
            NumericRange::F64(f, lo, hi) => (leaf, 3, f, lo.to_bits(), hi.to_bits()),
            NumericRange::Empty => (leaf, 0, 0, 0, 0),
        }
    }
}

/// Set bits of `bits`, `leaf`'s filter bitset of segment cache key `key`
/// ([`Query::numeric_range_bits_cached`], an `exists` presence bitset):
/// remembered by a sealed segment, whose cached bitsets are immutable;
/// counted again over a memtable, rebuilt per request anyway.
fn filter_count(leaf: &LeafReader<'_>, key: crate::segment::FilterCacheKey, bits: &[u64]) -> u64 {
    match leaf {
        LeafReader::Segment(seg) => seg.filter_bits_count(key, bits),
        LeafReader::Memtable(..) => bits.iter().map(|w| u64::from(w.count_ones())).sum(),
    }
}

/// [`filter_count`] as a doc set's size (a leaf has under 2^32 docs).
fn filter_count_u32(leaf: &LeafReader<'_>, key: crate::segment::FilterCacheKey, bits: &[u64]) -> u32 {
    u32::try_from(filter_count(leaf, key, bits)).unwrap_or(u32::MAX)
}

fn and_words(acc: &mut [u64], other: &[u64]) {
    for (a, b) in acc.iter_mut().zip(other) {
        *a &= *b;
    }
}

/// A scorer restricted to the docs of a bitset (a bool query's numeric
/// range filters). Per-doc iteration checks one bit per candidate; the
/// bulk top-k paths of the inner scorer run unchanged with the bitset's
/// complement folded into the deleted docs, so block-max skipping, the
/// block AND and MaxScore OR keep working under a filter.
pub struct FilteredScorer {
    inner: Box<dyn Scorer>,
    bits: Arc<crate::admission::Allocation<Vec<u64>>>,
    count: u32,
    /// The bitset is far smaller than the inner scorer: iteration leapfrogs
    /// from set bit to set bit (seeking the inner scorer to each) instead
    /// of stepping through every inner candidate.
    drive_bits: bool,
    /// The bitset has no doc at or after the inner scorer's position.
    exhausted: bool,
}

/// [`FilteredScorer`] drives iteration from its bitset once the inner
/// scorer has this many times more candidates than the bitset has docs.
const FILTER_DRIVE_RATIO: u64 = 8;

impl FilteredScorer {
    /// `bits`: bit `d` set = doc `d` passes the filter.
    pub fn new(inner: Box<dyn Scorer>, bits: Arc<crate::admission::Allocation<Vec<u64>>>) -> Self {
        let count: u32 = bits.iter().map(|w| w.count_ones()).sum();
        Self::new_counted(inner, bits, count)
    }

    /// [`Self::new`] with the bitset's set-bit count already known.
    pub fn new_counted(inner: Box<dyn Scorer>, bits: Arc<crate::admission::Allocation<Vec<u64>>>, count: u32) -> Self {
        let drive_bits = (count as u64).saturating_mul(FILTER_DRIVE_RATIO) < inner.size_hint() as u64;
        Self { inner, bits, count, drive_bits, exhausted: false }
    }

    #[inline(always)]
    fn accepts(&self, d: DocId) -> bool {
        self.bits.get(d as usize >> 6).is_some_and(|w| (w >> (d & 63)) & 1 == 1)
    }

    /// The first doc at or after `from` whose bit is set.
    #[inline]
    fn next_bit(&self, from: DocId) -> DocId {
        if from == TERMINATED {
            return TERMINATED;
        }
        let mut w = from as usize >> 6;
        let Some(&first) = self.bits.get(w) else { return TERMINATED };
        let mut word = first & (u64::MAX << (from & 63));
        loop {
            if word != 0 {
                return ((w as u32) << 6) | word.trailing_zeros();
            }
            w += 1;
            match self.bits.get(w) {
                Some(&next) => word = next,
                None => return TERMINATED,
            }
        }
    }

    #[inline]
    fn settle(&mut self, mut d: DocId) -> DocId {
        if self.drive_bits {
            // Leapfrog: the inner scorer is at `d`; agree on a doc both match.
            loop {
                if d == TERMINATED {
                    return TERMINATED;
                }
                let b = self.next_bit(d);
                if b == TERMINATED {
                    self.exhausted = true;
                    return TERMINATED;
                }
                if b == d {
                    return d;
                }
                d = self.inner.seek(b);
            }
        }
        while d != TERMINATED && !self.accepts(d) {
            d = self.inner.advance();
        }
        d
    }

    /// Deleted-or-filtered-out words for the bulk paths.
    fn excluded(&self, deleted: Option<&[u64]>) -> Option<crate::admission::Allocation<Vec<u64>>> {
        self.excluded_admitted(deleted,&crate::admission::shared())
    }
    fn excluded_admitted(&self,deleted:Option<&[u64]>,budget:&Arc<crate::admission::Budget>)->Option<crate::admission::Allocation<Vec<u64>>> {
        // Bulk collection is optional. Refuse this scratch before allocating
        // and let the ordinary scorer retain the same filtered semantics.
        let memory=budget.reserve((self.bits.len() as u64).saturating_mul(8).saturating_add(128),"bulk filter complement").ok()?;
        let mut ex: Vec<u64> = self.bits.iter().map(|w| !w).collect();
        if let Some(del) = deleted {
            for (e, d) in ex.iter_mut().zip(del) {
                *e |= *d;
            }
        }
        Some(crate::admission::Allocation::new(ex,memory))
    }
}

impl DocSet for FilteredScorer {
    fn advance(&mut self) -> DocId {
        if self.exhausted {
            return TERMINATED;
        }
        let d = self.inner.advance();
        self.settle(d)
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.exhausted {
            return TERMINATED;
        }
        let cur = self.inner.doc();
        if cur != TERMINATED && cur >= target {
            return cur;
        }
        let d = if self.drive_bits {
            match self.next_bit(target) {
                TERMINATED => {
                    self.exhausted = true;
                    return TERMINATED;
                }
                b => self.inner.seek(b),
            }
        } else {
            self.inner.seek(target)
        };
        self.settle(d)
    }

    fn doc(&self) -> DocId {
        if self.exhausted {
            return TERMINATED;
        }
        self.inner.doc()
    }

    fn size_hint(&self) -> u32 {
        self.inner.size_hint().min(self.count)
    }

    fn collect_first_k_bulk(&mut self, k: usize, deleted: Option<&[u64]>) -> Option<(Vec<DocId>, u64)> {
        let ex = self.excluded(deleted)?;
        self.inner.collect_first_k_bulk(k, Some(&ex))
    }
}

impl Scorer for FilteredScorer {
    fn score(&mut self) -> f32 {
        self.inner.score()
    }

    /// The inner scorer's bulk collection with the filtered-out docs
    /// masked as deleted (an aggregation over a bool of constant-score
    /// clauses and a range: the range's points bitset, every match added
    /// a word at a time).
    fn collect_all(&mut self, deleted: Option<&[u64]>, out: &mut Vec<DocId>) -> Option<f32> {
        if self.exhausted || self.inner.doc() != TERMINATED {
            return None;
        }
        let ex = self.excluded(deleted)?;
        self.inner.collect_all(Some(&ex), out)
    }

    fn collect_top_k_bulk(&mut self, k: usize, deleted: Option<&[u64]>) -> Option<(Vec<crate::query::collector::ScoredDoc>, u64)> {
        let ex = self.excluded(deleted)?;
        self.inner.collect_top_k_bulk(k, Some(&ex))
    }

    fn collect_top_k_bulk_min(&mut self, k: usize, deleted: Option<&[u64]>, floor: f32) -> Option<(Vec<crate::query::collector::ScoredDoc>, u64)> {
        let ex = self.excluded(deleted)?;
        self.inner.collect_top_k_bulk_min(k, Some(&ex), floor)
    }

    fn collect_top_k_count_limited(&mut self, k: usize, deleted: Option<&[u64]>, count_limit: u64, floor: &crate::query::bulk::Floor<'_>) -> Option<(Vec<crate::query::collector::ScoredDoc>, u64)> {
        let ex = self.excluded(deleted)?;
        self.inner.collect_top_k_count_limited(k, Some(&ex), count_limit, floor)
    }

    fn max_score(&self) -> Option<f32> {
        // A bound of the inner scores bounds those of any subset.
        self.inner.max_score()
    }

    fn max_score_up_to(&mut self, up_to: DocId) -> Option<f32> {
        self.inner.max_score_up_to(up_to)
    }
}

impl Query {
    /// Doc bitset of a numeric range query (`RangeI64` / `RangeDate` /
    /// `RangeU64` / `RangeF64`) on `leaf`, built from the points index;
    /// `None` when `self` is not a numeric range. Exclusive bounds are
    /// turned into the adjacent inclusive value (exact for integers and,
    /// via `next_up`/`next_down`, for doubles). A field without a numeric
    /// column matches nothing.
    pub fn numeric_range_bits(&self, leaf: &LeafReader<'_>) -> Option<Vec<u64>> {
        self.numeric_range_spec().map(|r| r.bits(leaf))
    }

    /// A nested query's child that [`range_block_join`] verifies per
    /// parent block: an i64 / date range, possibly boosted, as `(field, lo,
    /// hi, child score)` with inclusive bounds; the score is what the
    /// range's scorer gives every match (1, times the boost).
    fn nested_range_child(&self) -> Option<(u16, i64, i64, f32)> {
        match self {
            Query::RangeI64 { .. } | Query::RangeDate { .. } => match self.numeric_range_spec()? {
                NumericRange::I64(f, lo, hi) => Some((f, lo, hi, 1.0)),
                _ => None,
            },
            Query::Boost { query, boost } if matches!(**query, Query::RangeI64 { .. } | Query::RangeDate { .. }) && boost.is_finite() && *boost >= 0.0 => {
                let (f, lo, hi, s) = query.nested_range_child()?;
                Some((f, lo, hi, s * *boost))
            }
            _ => None,
        }
    }

    /// An i64 / date range's scorer over a sealed segment's single-valued
    /// column that finds its docs in doc order as asked for
    /// ([`RangeDocs`](crate::codec::fastfields::RangeDocs)), when the search
    /// stops after its first matches (the [`geo_scorer::with_wanted`] hint:
    /// a uniformly scored `size: 10` search counting up to 10,000) and the
    /// segment holds no bitset of this range yet. The bitset would read the
    /// whole column (big5's `range` over `@timestamp`, a doc-values-only
    /// field in logsdb, read every doc of every backing index on every
    /// search); the same docs come from the first blocks instead.
    ///
    /// [`geo_scorer::with_wanted`]: crate::query::geo_scorer::with_wanted
    fn lazy_range_scorer(&self, leaf: &LeafReader<'_>) -> Option<Box<dyn Scorer>> {
        let LeafReader::Segment(seg) = leaf else { return None };
        let spec = self.numeric_range_spec()?;
        let NumericRange::I64(field, lo, hi) = spec else { return None };
        if crate::query::geo_scorer::take_hint() == 0 {
            return None;
        }
        let (_, kind, f, klo, khi) = spec.key(leaf.id());
        if seg.peek_filter_bits((kind, f, klo, khi)).is_some() {
            return None;
        }
        let docs = seg.fast_fields().get_i64(field)?.lazy_range(lo, hi)?;
        // SAFETY: the scorer never outlives the search holding the segment
        // (the contract every leaf-bound scorer relies on, as the geo
        // doc-order scorer's).
        let docs: crate::codec::fastfields::RangeDocs<'static, i64> = unsafe { std::mem::transmute(docs) };
        // SAFETY: as above (the segment outlives the search's scorers).
        let seg: &'static crate::segment::Segment = unsafe { std::mem::transmute::<&crate::segment::Segment, &'static crate::segment::Segment>(seg) };
        Some(Box::new(ConstScorer::new(LazyRangeDocs { docs, seg, key: (kind, f, klo, khi), spec, started: false }, 1.0).with_safe_bound(true)))
    }

    /// An i64 / date range, possibly boosted or constant-scored (the
    /// clauses [`Query::lazy_range_scorer`] serves).
    fn is_scored_i64_range(&self) -> bool {
        match self {
            Query::RangeI64 { .. } | Query::RangeDate { .. } => true,
            Query::Boost { query, .. } | Query::ConstScore { query, .. } => query.is_scored_i64_range(),
            _ => false,
        }
    }

    /// The segment filter-cache key of this numeric range's bitset
    /// ([`Query::numeric_range_bits_cached`]); `None` when `self` is not a
    /// numeric range.
    fn numeric_range_filter_key(&self) -> Option<crate::segment::FilterCacheKey> {
        let (_, kind, field, lo, hi) = self.numeric_range_spec()?.key(0);
        Some((kind, field, lo, hi))
    }

    /// [`Query::numeric_range_bits`] through the request's
    /// [`SearchContext`] cache: the same range on the same leaf is
    /// computed once per context (e.g. a hybrid search's text top-k and its
    /// scoring of the k-NN-only docs).
    pub fn numeric_range_bits_cached(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Arc<crate::admission::Allocation<Vec<u64>>>> {
        let spec = self.numeric_range_spec()?;
        let key = spec.key(leaf.id());
        if let Some(b) = ctx.range_cache.lock().get(&key) {
            return Some(b.clone());
        }
        // Sealed segments are immutable: the bitset is remembered on the
        // segment across requests (Lucene's query cache). Memtables change
        // between requests and are rebuilt every time.
        let bits = match leaf {
            LeafReader::Segment(seg) => {
                let (_, kind, field, lo, hi) = key;
                seg.cached_filter_bits((kind, field, lo, hi), || spec.bits(leaf))
            }
            LeafReader::Memtable(_, ..) => crate::admission::shared().reserve((leaf.num_docs() as u64).div_ceil(64).saturating_mul(8).saturating_add(512),"request numeric filter").map(|memory|Arc::new(crate::admission::Allocation::new(spec.bits(leaf),memory))),
        };
        let bits=match bits { Ok(bits)=>bits,Err(error)=>{ctx.record_memory_error(error.to_string());return None;} };
        ctx.range_cache.lock().insert(key, bits.clone());
        Some(bits)
    }

    /// `exists` on a field whose fast fields know presence: its docs as
    /// words, built once per sealed segment and kept with it like a numeric
    /// range's (an `exists` kNN filter on a keyword with a value per
    /// document listed every doc, decoding each one's term, on every query).
    fn presence_bits_cached(field: u16, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Arc<crate::admission::Allocation<Vec<u64>>>> {
        let bits = match leaf {
            LeafReader::Segment(seg) => {
                // Probe first: only fields whose fast fields tell presence.
                if leaf.numeric_presence(field).is_none() && seg.fast_fields().get_keyword(field).is_none() { return None; }
                seg.cached_filter_bits((EXISTS_FILTER_KIND, field, 0, 0), || leaf.presence_words(field).unwrap_or_default())
            }
            LeafReader::Memtable(..) => {
                let words = leaf.presence_words(field)?;
                crate::admission::shared().reserve((words.len() as u64).saturating_mul(8).saturating_add(128), "request exists filter")
                    .map(|memory| Arc::new(crate::admission::Allocation::new(words, memory)))
            }
        };
        match bits {
            Ok(bits) => Some(bits),
            Err(error) => { ctx.record_memory_error(error.to_string()); None }
        }
    }

    /// A disjunction of numeric clauses on one integer (or date) field and
    /// nothing else, at most one required (a numeric `terms` query): its
    /// docs over `leaf` as one bitset, each clause OR-ed in, instead of a
    /// bitset per clause. `None` for any other query.
    fn numeric_union_bits(&self, leaf: &LeafReader<'_>) -> Option<Vec<u64>> {
        let Query::Bool { must, should, must_not, filter, minimum_should_match } = self else { return None };
        if !must.is_empty() || !must_not.is_empty() || !filter.is_empty() || should.len() < 2 || minimum_should_match.resolve(should.len()) > 1 {
            return None;
        }
        let mut field = None;
        let mut ranges = Vec::with_capacity(should.len());
        for q in should {
            match q.numeric_range_spec()? {
                NumericRange::I64(f, lo, hi) => {
                    if field.is_some_and(|x| x != f) {
                        return None;
                    }
                    field = Some(f);
                    ranges.push((lo, hi));
                }
                NumericRange::Empty => {}
                _ => return None,
            }
        }
        let field = field?;
        let mut words = vec![0u64; (leaf.num_docs() as usize).div_ceil(64)];
        for (lo, hi) in ranges {
            leaf.or_range_i64_bits(field, lo, hi, &mut words);
        }
        Some(words)
    }

    /// Whether this is a plain numeric range query (`RangeI64` /
    /// `RangeDate` / `RangeU64` / `RangeF64`), whose bitset needs no term
    /// statistics from a [`SearchContext`].
    pub fn is_numeric_range(&self) -> bool {
        self.numeric_range_spec().is_some()
    }

    /// Normalised inclusive bounds of a numeric range query.
    /// A long / date range every match of this query satisfies: the query
    /// itself, or a required clause (`must` / `filter`, through constant
    /// score and boost wrappers). `(field, lo, hi)` inclusive. Aggregations
    /// over the query's matches use it to count from the points index.
    pub fn required_i64_range(&self) -> Option<(u16, i64, i64)> {
        match self {
            Query::RangeI64 { .. } | Query::RangeDate { .. } => match self.numeric_range_spec()? {
                NumericRange::I64(f, lo, hi) => Some((f, lo, hi)),
                _ => None,
            },
            Query::ConstScore { query, .. } | Query::Boost { query, .. } => query.required_i64_range(),
            Query::Bool { must, filter, .. } => must.iter().chain(filter.iter()).find_map(|q| q.required_i64_range()),
            _ => None,
        }
    }

    /// Every long / date range a match of this query satisfies (as
    /// [`Self::required_i64_range`], all of them, nested required bools
    /// included), pushed to `out` as `(field, lo, hi)` inclusive. A leaf
    /// none of whose values lies in one of them matches nothing.
    pub fn required_i64_ranges(&self, out: &mut Vec<(u16, i64, i64)>) {
        match self {
            Query::RangeI64 { .. } | Query::RangeDate { .. } => {
                if let Some(NumericRange::I64(f, lo, hi)) = self.numeric_range_spec() {
                    out.push((f, lo, hi));
                }
            }
            Query::ConstScore { query, .. } | Query::Boost { query, .. } => query.required_i64_ranges(out),
            Query::Bool { must, filter, .. } => must.iter().chain(filter.iter()).for_each(|q| q.required_i64_ranges(out)),
            _ => {}
        }
    }

    /// [`Self::required_i64_range`] of a query that is that range alone
    /// (through constant score and boost wrappers and a one-clause `bool`
    /// with no `should` / `must_not`): its matches are exactly the docs with
    /// a value of `field` in `lo..=hi`.
    pub fn sole_i64_range(&self) -> Option<(u16, i64, i64)> {
        match self {
            Query::RangeI64 { .. } | Query::RangeDate { .. } => self.required_i64_range(),
            Query::ConstScore { query, .. } | Query::Boost { query, .. } => query.sole_i64_range(),
            Query::Bool { must, should, must_not, filter, .. } if should.is_empty() && must_not.is_empty() && must.len() + filter.len() == 1 => must.iter().chain(filter).next()?.sole_i64_range(),
            _ => None,
        }
    }

    /// A `double` range query alone (through constant score and boost
    /// wrappers and a one-clause `bool` with no `should` / `must_not`): its
    /// matches are exactly the docs whose value of `field` lies in
    /// `lo..=hi` in the total order, NaN excluded (as the points index
    /// answers it). `(field, lo, hi)` inclusive.
    pub fn sole_f64_range(&self) -> Option<(u16, f64, f64)> {
        match self {
            Query::RangeF64 { .. } => match self.numeric_range_spec()? {
                NumericRange::F64(f, lo, hi) => Some((f, lo, hi)),
                _ => None,
            },
            Query::ConstScore { query, .. } | Query::Boost { query, .. } => query.sole_f64_range(),
            Query::Bool { must, should, must_not, filter, .. } if should.is_empty() && must_not.is_empty() && must.len() + filter.len() == 1 => must.iter().chain(filter).next()?.sole_f64_range(),
            _ => None,
        }
    }

    fn numeric_range_spec(&self) -> Option<NumericRange> {
        Some(match self {
            Query::RangeI64 { field, lower, upper } | Query::RangeDate { field, lower, upper } => {
                let lo = match *lower {
                    Bound::Included(l) => Some(l),
                    Bound::Excluded(l) => l.checked_add(1),
                    Bound::Unbounded => Some(i64::MIN),
                };
                let hi = match *upper {
                    Bound::Included(u) => Some(u),
                    Bound::Excluded(u) => u.checked_sub(1),
                    Bound::Unbounded => Some(i64::MAX),
                };
                match (lo, hi) {
                    (Some(lo), Some(hi)) => NumericRange::I64(*field, lo, hi),
                    _ => NumericRange::Empty,
                }
            }
            Query::RangeU64 { field, lower, upper } => {
                let lo = match *lower {
                    Bound::Included(l) => Some(l),
                    Bound::Excluded(l) => l.checked_add(1),
                    Bound::Unbounded => Some(0),
                };
                let hi = match *upper {
                    Bound::Included(u) => Some(u),
                    Bound::Excluded(u) => u.checked_sub(1),
                    Bound::Unbounded => Some(u64::MAX),
                };
                match (lo, hi) {
                    (Some(lo), Some(hi)) => NumericRange::U64(*field, lo, hi),
                    _ => NumericRange::Empty,
                }
            }
            Query::RangeF64 { field, lower, upper } => {
                let lo = match *lower {
                    Bound::Included(l) => Some(l),
                    Bound::Excluded(l) if l == f64::INFINITY => None,
                    Bound::Excluded(l) => Some(l.next_up()),
                    Bound::Unbounded => Some(f64::NEG_INFINITY),
                };
                let hi = match *upper {
                    Bound::Included(u) => Some(u),
                    Bound::Excluded(u) if u == f64::NEG_INFINITY => None,
                    Bound::Excluded(u) => Some(u.next_down()),
                    Bound::Unbounded => Some(f64::INFINITY),
                };
                match (lo, hi) {
                    (Some(lo), Some(hi)) => NumericRange::F64(*field, lo, hi),
                    _ => NumericRange::Empty,
                }
            }
            _ => return None,
        })
    }

    /// The docs of `leaf` matching this query, as a bitset (deleted docs
    /// are not cleared). Numeric ranges come straight from the points
    /// index, bool queries without `should` clauses are the AND (and
    /// AND-NOT) of their clauses' bitsets, constant-score/boost wrappers
    /// are transparent; anything else walks its scorer.
    pub fn filter_bits(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> crate::error::Result<crate::admission::Allocation<Vec<u64>>> {
        self.filter_bits_admitted(leaf,ctx,&crate::admission::shared())
    }
    fn filter_bits_admitted(&self,leaf:&LeafReader<'_>,ctx:&SearchContext,budget:&Arc<crate::admission::Budget>)->crate::error::Result<crate::admission::Allocation<Vec<u64>>> {
        let memory=budget.reserve((leaf.num_docs() as u64).div_ceil(64).saturating_mul(8).saturating_add(128),"query filter words")?;
        let words=self.filter_bits_inner(leaf,ctx)?;
        ctx.execution_result()?;
        Ok(crate::admission::Allocation::new(words,memory))
    }
    fn filter_bits_inner(&self, leaf:&LeafReader<'_>,ctx:&SearchContext)->crate::error::Result<Vec<u64>> {
        if let Some(b) = self.numeric_range_bits_cached(leaf, ctx) {
            return Ok(b.to_vec());
        }
        ctx.execution_result()?;
        let n = leaf.num_docs();
        match self {
            Query::Term(term) => {
                // Filters need membership only, not term frequencies or BM25.
                return Ok(match ctx.postings(leaf, term) {
                    Some(postings) => postings_words(&postings, n),
                    None => vec![0u64; (n as usize).div_ceil(64)],
                });
            }
            Query::ConstScore { query, .. } | Query::Boost { query, .. } => return query.filter_bits_inner(leaf, ctx),
            Query::Exists { field } => {
                if let Some(bits) = Self::presence_bits_cached(*field, leaf, ctx) {
                    return Ok(bits.to_vec());
                }
            }
            Query::MatchAll => {
                let mut w = vec![0u64; (n as usize).div_ceil(64)];
                crate::codec::fastfields::set_bit_range(&mut w, 0, n as usize);
                return Ok(w);
            }
            Query::Bool { must, should, must_not, filter, minimum_should_match }
                if must.is_empty() && filter.is_empty() && !should.is_empty() && minimum_should_match.resolve(should.len()) <= 1 => {
                // A filter disjunction is a word-wise union, rather than a
                // scored heap merge for every matching document.
                let mut bits = vec![0u64; (n as usize).div_ceil(64)];
                for clause in should {
                    let child = clause.filter_bits(leaf, ctx)?;
                    for (dst, src) in bits.iter_mut().zip(child.iter()) { *dst |= src; }
                }
                for clause in must_not {
                    let child = clause.filter_bits(leaf, ctx)?;
                    for (dst, src) in bits.iter_mut().zip(child.iter()) { *dst &= !src; }
                }
                return Ok(bits);
            }
            Query::Bool { must, should, must_not, filter, .. } if should.is_empty() && !(must.is_empty() && filter.is_empty()) => {
                let mut acc=vec![u64::MAX;(n as usize).div_ceil(64)];
                for q in must.iter().chain(filter) {
                    let b = q.filter_bits(leaf, ctx)?;
                    and_words(&mut acc,&b);
                    if acc.iter().all(|&w|w==0) {
                        break;
                    }
                }
                for q in must_not {
                    let b = q.filter_bits(leaf, ctx)?;
                    for (a, x) in acc.iter_mut().zip(b.iter()) {
                        *a &= !x;
                    }
                }
                return Ok(acc);
            }
            _ => {}
        }
        Ok(match self.scorer(leaf, ctx) {
            Some(mut s) => doc_bits(&mut *s, n),
            None => vec![0u64; (n as usize).div_ceil(64)],
        })
    }

    /// [`Query::scorer`] under `boost` (the product of the enclosing
    /// `BoostQuery`s): folded into the term weights below, as Lucene does,
    /// instead of scaling the finished scores.
    pub fn scorer_b(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, boost: f32) -> Option<Box<dyn Scorer>> {
        if boost == 1.0 {
            return self.scorer(leaf, ctx);
        }
        match self {
            Query::Term(t) => self_term_boosted(leaf, ctx, t, boost),
            Query::Phrase { terms, .. } if terms.len() == 1 => self_term_boosted(leaf, ctx, &terms[0], boost),
            Query::Phrase { terms, slop } => {
                let field = terms.first()?.field;
                let stats: Vec<TermStats> = terms.iter().map(|t| ctx.stats(t)).collect();
                let positions: Vec<Vec<Term>> = terms.iter().map(|t| vec![t.clone()]).collect();
                let s = leaf.multi_phrase_raw_boosted(&positions, *slop, ctx.sim(field), &stats, ctx.fstats(field), boost, &|t| ctx.postings(leaf, t))?;
                Some(Box::new(s))
            }
            Query::MultiPhrase { field, positions, slop } if positions.len() > 1 => {
                let stats: Vec<TermStats> = positions.iter().flatten().map(|t| ctx.stats(t)).collect();
                let s = leaf.multi_phrase_raw_boosted(positions, *slop, ctx.sim(*field), &stats, ctx.fstats(*field), boost, &|t| ctx.postings(leaf, t))?;
                Some(Box::new(s))
            }
            Query::Boost { query, boost: b } => query.scorer_b(leaf, ctx, boost * *b),
            Query::Custom(c) => c.scorer_b(leaf, ctx, boost),
            Query::Bool { must, should, must_not, filter, minimum_should_match } => bool_scorer(must, should, must_not, filter, minimum_should_match, leaf, ctx, boost),
            Query::DisMax { queries, tie_breaker } => dis_max_scorer(queries, *tie_breaker, leaf, ctx, Some(boost)),
            _ => Some(Box::new(Scaled::new(self.scorer(leaf, ctx)?, boost))),
        }
    }

    /// Build this query's scorer for one leaf. `None` means "matches nothing
    /// here" (a missing term, an unknown field).
    pub fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        match self {
            Query::Term(t) => self_term(leaf, ctx, t),
            Query::TermSet { field, terms, boost } => {
                // Parsed `terms` queries arrive sorted; sort a copy otherwise.
                let mut texts: Vec<&[u8]> = terms.iter().map(|t| t.as_slice()).collect();
                if !texts.windows(2).all(|w| w[0] <= w[1]) {
                    texts.sort_unstable();
                }
                Some(const_union_sorted(leaf, *field, &texts, *boost).unwrap_or_else(empty))
            }
            Query::Phrase { terms, slop } => {
                if terms.len() == 1 {
                    return self_term(leaf, ctx, &terms[0]);
                }
                let field = terms.first()?.field;
                let stats: Vec<TermStats> = terms.iter().map(|t| ctx.stats(t)).collect();
                let positions: Vec<Vec<Term>> = terms.iter().map(|t| vec![t.clone()]).collect();
                let s = leaf.multi_phrase_raw_with(&positions, *slop, ctx.sim(field), &stats, ctx.fstats(field), &|t| ctx.postings(leaf, t))?;
                Some(Box::new(s))
            }
            Query::MultiPhrase { field, positions, slop } => {
                if positions.is_empty() {
                    return None;
                }
                if positions.len() == 1 {
                    let children: Vec<Box<dyn Scorer>> = positions[0].iter().filter_map(|t| self_term(leaf, ctx, t)).collect();
                    return if children.is_empty() { None } else { Some(crate::query::boolean::union_of(children)) };
                }
                let stats: Vec<TermStats> = positions.iter().flatten().map(|t| ctx.stats(t)).collect();
                let s = leaf.multi_phrase_raw_with(positions, *slop, ctx.sim(*field), &stats, ctx.fstats(*field), &|t| ctx.postings(leaf, t))?;
                Some(Box::new(s))
            }
            Query::Prefix { .. } | Query::Wildcard { .. } | Query::Regexp { .. } | Query::Fuzzy { .. } | Query::TermRange { .. } => {
                let spec = self.multi_term_spec()?;
                if spec.rewrite == MultiTermRewrite::ConstantScore {
                    let terms = enumerate_terms(leaf, spec.field, &*spec.matcher, spec.lower.as_deref(), spec.upper.as_deref(), spec.max_expansions);
                    // Constant-score rewrites (plain and boolean) score the boost, 1.
                    return Some(const_union(leaf, &terms, 1.0).unwrap_or_else(empty));
                }
                let expanded = match ctx.expansions.get(&self.node_key()) {
                    Some(e) => e.clone(),
                    None => match spec.expand(std::slice::from_ref(leaf), Some(ctx.max_clause_count)) {
                        Ok(terms) => Arc::new(terms),
                        Err(()) => { *ctx.clause_error.lock()=Some(false); return None; },
                    },
                };
                if spec.rewrite == MultiTermRewrite::ConstantScoreBoolean {
                    let terms:Vec<_>=expanded.iter().map(|expanded|expanded.term.clone()).collect();
                    return Some(const_union(leaf, &terms, 1.0).unwrap_or_else(empty));
                }
                scoring_union(leaf, &expanded, spec.rewrite, ctx.sim(spec.field), ctx.fstats(spec.field))
            }
            Query::RangeI64 { .. } | Query::RangeDate { .. } | Query::RangeU64 { .. } | Query::RangeF64 { .. } => {
                // Outside the segment's values (kept with the segment, as
                // the can-match phase reads them): nothing matches, without
                // reading a block's bounds (a billion-doc leaf has
                // millions) or building a bitset.
                if let (LeafReader::Segment(seg), Some(NumericRange::I64(field, lo, hi))) = (leaf, self.numeric_range_spec()) {
                    if seg.min_max_i64(field).is_some_and(|(min, max)| max < lo || min > hi) {
                        return Some(empty());
                    }
                }
                if let Some(lazy) = self.lazy_range_scorer(leaf) {
                    return Some(lazy);
                }
                // Points index -> bitset; no per-match Vec.
                let bits = self.numeric_range_bits_cached(leaf, ctx)?;
                let count = filter_count_u32(leaf, self.numeric_range_filter_key()?, &bits);
                // A finished bitset: skipping the rest of it can hide no
                // evaluation error, so its constant is a safe score bound
                // (a scored range in `must` then lets top-k stop early).
                Some(Box::new(ConstScorer::new(BitSetDocSet::shared_counted(bits, count), 1.0).with_safe_bound(true)))
            }
            Query::Exists { field } => {
                // A numeric column answers from its presence bitmap (or
                // matches every doc) instead of listing each doc.
                if let Some(present) = leaf.numeric_presence(*field) {
                    return Some(match present {
                        None if leaf.num_docs() == 0 => empty(),
                        None => Box::new(ConstScorer::new(AllDocSet::new(leaf.num_docs()), 1.0).with_safe_bound(true)) as Box<dyn Scorer>,
                        Some(words) if words.iter().all(|&w| w == 0) => empty(),
                        Some(words) => Box::new(ConstScorer::new(BitSetDocSet::new(words), 1.0).with_safe_bound(true)) as Box<dyn Scorer>,
                    });
                }
                // A keyword column: its presence words (cached per segment),
                // not one decoded term per doc.
                if let Some(bits) = Self::presence_bits_cached(*field, leaf, ctx) {
                    let count = filter_count_u32(leaf, (EXISTS_FILTER_KIND, *field, 0, 0), &bits);
                    if count == 0 { return Some(empty()); }
                    return Some(Box::new(ConstScorer::new(BitSetDocSet::shared_counted(bits, count), 1.0).with_safe_bound(true)));
                }
                if let Some(docs) = leaf.value_presence(*field) {
                    return Some(const_docs(docs, 1.0));
                }
                let terms = enumerate_terms(leaf, *field, &AllTerms, None, None, usize::MAX);
                Some(const_union(leaf, &terms, 1.0).unwrap_or_else(empty))
            }
            Query::Bool { must, should, must_not, filter, minimum_should_match } => bool_scorer(must, should, must_not, filter, minimum_should_match, leaf, ctx, 1.0),
            Query::ConstScore { query, score, .. } => {
                // A numeric `terms` query: one bitset for all its values.
                if let Some(bits) = query.numeric_union_bits(leaf) {
                    if bits.iter().all(|&w| w == 0) {
                        return None;
                    }
                    let memory = match crate::admission::shared().reserve((bits.len() as u64).saturating_mul(8).saturating_add(128), "numeric terms filter") {
                        Ok(memory) => memory,
                        Err(error) => {
                            ctx.record_memory_error(error.to_string());
                            return None;
                        }
                    };
                    return Some(Box::new(ConstScorer::new(BitSetDocSet::shared(Arc::new(crate::admission::Allocation::new(bits, memory))), *score).with_safe_bound(true)));
                }
                let mut s = query.scorer(leaf, ctx)?;
                // A postings walk cannot fail later, so skipping it is safe
                // even when its own scores vary (a `match_only_text` term
                // keeps frequencies): `query_string` words on such a field
                // then take the MaxScore path.
                let safe_bound = s.score_upper_bound().is_some() || s.as_bm25_term().is_some();
                Some(Box::new(ConstScorer::new(AsDocSet(s), *score).with_safe_bound(safe_bound)))
            }
            Query::Boost { query, boost } => query.scorer_b(leaf, ctx, *boost),
            Query::DisMax { queries, tie_breaker } => dis_max_scorer(queries, *tie_breaker, leaf, ctx, None),
            Query::Knn { boost, .. } => {
                // The context holds this leaf's share of the global top-k;
                // without it (a context built for another query) fall back
                // to this leaf's own top-k.
                let hits = match ctx.knn_hits.get(&(self.node_key(), leaf.id())) {
                    Some(h) => h.clone(),
                    None => self.leaf_knn(leaf, ctx),
                };
                let hits: Vec<(DocId, f32)> = hits.into_iter().map(|(d, s)| (d, s * boost)).collect();
                if hits.is_empty() {
                    return None;
                }
                Some(Box::new(ScoredDocs::new(hits)))
            }
            Query::FunctionScore { query, functions, score_mode, boost_mode, max_boost, min_score } => {
                let inner = query.scorer(leaf, ctx)?;
                let params = FunctionScoreParams { score_mode: *score_mode, boost_mode: *boost_mode, max_boost: *max_boost, min_score: *min_score };
                let funcs = PreparedFunctions::prepare(functions, params, leaf, ctx);
                let scorer = FunctionScoreScorer::new(inner, funcs);
                Some(Box::new(if matches!(query.as_ref(), Query::MatchAll) { scorer.with_match_all(leaf.num_docs()) } else { scorer }))
            }
            Query::Span(sq) => {
                let expanded = match ctx.span_expansions.get(&self.node_key()) {
                    Some(span)=>span.clone(),
                    None=>match sq.rewrite_bounded(std::slice::from_ref(leaf),ctx.max_clause_count) {
                        Ok(span)=>Arc::new(span),
                        Err(())=>{ *ctx.clause_error.lock()=Some(false); return None; },
                    },
                };
                span_scorer(&expanded, leaf, ctx).map(|s| Box::new(s) as Box<dyn Scorer>)
            },
            Query::Intervals { field, rule } => {
                let expanded = match ctx.interval_expansions.get(&(self.node_key(),leaf.id())) {
                    Some(span)=>span.clone(),
                    None=>match rule.rewrite_bounded(*field,leaf,ctx.max_clause_count) {
                        Ok(span)=>Arc::new(span),
                        Err(error)=>{ ctx.interval_error.lock().get_or_insert(error); return None; },
                    },
                };
                interval_scorer(rule, &expanded, leaf, ctx).map(|s| Box::new(s) as Box<dyn Scorer>)
            },
            Query::MatchAll => Some(Box::new(ConstScorer::new(AllDocSet::new(leaf.num_docs()), 1.0))),
            Query::MatchNone => None,
            Query::Custom(c) => c.scorer(leaf, ctx),
            Query::Nested { path, query, score_mode, filter } => {
                if *filter {
                    nested_filter_scorer(path, query, leaf, ctx)
                } else {
                    nested_scorer(path, query, *score_mode, leaf, ctx)
                }
            }
            Query::GeoBoundingBox { .. } | Query::GeoDistance { .. } | Query::GeoPolygon { .. } => {
                // Sealed segments: the KD walk with the cell-grid point
                // test, over a few threads for large trees (or the
                // doc-order scan when the search stops early). A doc
                // matches when any of its points is in the shape.
                if let LeafReader::Segment(s) = leaf {
                    let (field, pred) = self.geo_pred()?;
                    return match s.geo_index(field) {
                        Some(g) => crate::query::geo_scorer::geo_pred_scorer(g, pred, true, s.num_docs, 1.0),
                        None => Some(empty()),
                    };
                }
                let (field, shape) = self.geo_shape()?;
                Some(hits_scorer(leaf.geo_hits(field, &shape), 1.0))
            }
        }
    }

    /// The number of docs of `leaf` this query matches when it is known
    /// without visiting them (KD leaf counts, all docs), else `None`.
    /// Only for leaves without deletes or nested docs.
    pub fn exact_count(&self, leaf: &LeafReader<'_>) -> Option<u64> {
        // Deleted or nested child docs: counts would include them.
        if leaf.deleted_bits().is_some() {
            return None;
        }
        match self {
            Query::MatchAll => Some(leaf.num_docs() as u64),
            Query::MatchNone => Some(0),
            Query::ConstScore { query, .. } | Query::Boost { query, .. } => query.exact_count(leaf),
            Query::Bool { must, should, must_not, filter, .. } if should.is_empty() && must_not.is_empty() && must.len() + filter.len() == 1 => must.iter().chain(filter).next()?.exact_count(leaf),
            Query::Custom(c) if c.geo_pred().is_none() => c.exact_count(leaf),
            _ => {
                let (field, pred) = self.geo_pred()?;
                match leaf {
                    LeafReader::Segment(s) => match s.geo_index(field) {
                        Some(g) => g.pred_count(&pred),
                        None => Some(0),
                    },
                    LeafReader::Memtable(_, ..) => None,
                }
            }
        }
    }

    /// A geo point predicate's matches on `leaf`, capped at `limit`
    /// ([`crate::geo::GeoIndex::pred_count_at_least`]).
    fn geo_count_at_least(&self, leaf: &LeafReader<'_>, limit: u64) -> Option<u64> {
        let (field, pred) = self.geo_pred()?;
        match leaf {
            LeafReader::Segment(s) => match s.geo_index(field) {
                Some(g) => g.pred_count_at_least(&pred, limit),
                None => Some(0),
            },
            LeafReader::Memtable(_, ..) => None,
        }
    }

    /// [`CustomQuery::count_at_least`] through constant-score wrappers.
    pub fn count_at_least(&self, leaf: &LeafReader<'_>, limit: u64) -> Option<u64> {
        if leaf.deleted_bits().is_some() {
            return None;
        }
        match self {
            Query::ConstScore { query, .. } | Query::Boost { query, .. } => query.count_at_least(leaf, limit),
            Query::Bool { must, should, must_not, filter, .. } if should.is_empty() && must_not.is_empty() && must.len() + filter.len() == 1 => must.iter().chain(filter).next()?.count_at_least(leaf, limit),
            Query::Custom(c) => c.count_at_least(leaf, limit).or_else(|| c.geo_pred().and_then(|_| self.geo_count_at_least(leaf, limit))),
            // Geo point predicates: the KD tree's leaf counts, stopping at
            // the limit.
            _ if self.geo_pred().is_some() => self.geo_count_at_least(leaf, limit),
            _ => None,
        }
    }

    /// The query as a boolean combination of point predicates on one geo
    /// field (geo queries, `bool` of them with no `should`, constant-score
    /// wrappers): it holds for a doc of a single-valued field when its
    /// point passes. `None` for anything else.
    pub fn geo_pred(&self) -> Option<(u16, crate::geo::GeoPred)> {
        use crate::geo::GeoPred;
        match self {
            Query::GeoBoundingBox { .. } | Query::GeoDistance { .. } | Query::GeoPolygon { .. } => {
                let (f, shape) = self.geo_shape()?;
                // Distances and polygons test points through a cell grid.
                let pred: std::sync::Arc<dyn crate::geo::PointPredicate> = match (&shape, shape.plain_bbox()) {
                    (GeoShape::BoundingBox { .. }, _) | (_, None) => std::sync::Arc::new(shape),
                    // Grids shared by every leaf and repeat of the shape.
                    (GeoShape::Distance { lat, lon, meters }, Some(b)) => std::sync::Arc::new(crate::geo::GridPred::keyed(crate::geo::DistancePred::new(*lat, *lon, *meters), b, format!("{shape:?}"))),
                    // (A huge polygon is not worth its key.)
                    (GeoShape::Polygon { points }, Some(b)) if points.len() > crate::geo::GRID_KEY_MAX_VERTICES => std::sync::Arc::new(crate::geo::GridPred::new(shape, b)),
                    (_, Some(b)) => {
                        let key = format!("{shape:?}");
                        std::sync::Arc::new(crate::geo::GridPred::keyed(shape, b, key))
                    }
                };
                Some((f, GeoPred::Leaf(pred)))
            }
            Query::ConstScore { query, .. } | Query::Boost { query, .. } => query.geo_pred(),
            Query::Custom(c) => c.geo_pred(),
            Query::Bool { must, should, must_not, filter, .. } if should.is_empty() => bool_geo_pred(must, must_not, filter),
            _ => None,
        }
    }

    /// Whether the query is one geo predicate whose scorer may stop
    /// after the first matches (see [`crate::query::geo_scorer`]).
    pub fn takes_geo_hint(&self) -> bool {
        match self {
            Query::ConstScore { query, .. } | Query::Boost { query, .. } => query.takes_geo_hint(),
            // A bool of one required clause (ES|QL's pushed-down WHERE is
            // `bool.filter: [q]`): its scorer is that clause's.
            Query::Bool { must, should, must_not, filter, .. } if should.is_empty() && must_not.is_empty() && must.len() + filter.len() == 1 => {
                must.iter().chain(filter).all(|q| q.takes_geo_hint())
            }
            // A bool with a scored range clause (big5's `range_field_*`: a
            // term AND / OR a range): the range's scorer takes the hint and
            // finds its docs lazily (`lazy_range_scorer`); filter clauses
            // read cached bitsets and never take it.
            Query::Bool { must, should, .. } if must.iter().chain(should).any(|q| q.is_scored_i64_range()) => true,
            Query::Custom(c) if c.takes_geo_hint() => true,
            // Its scorer may find its first docs lazily (see
            // `lazy_range_scorer`).
            Query::RangeI64 { .. } | Query::RangeDate { .. } => true,
            _ => self.geo_pred().is_some(),
        }
    }

    /// [`Query::geo_pred`] of a clause inside a `bool`: a nested `bool`'s
    /// predicate may hold for docs without a point (see `bool_geo_parts`).
    fn geo_pred_part(&self) -> Option<(u16, crate::geo::GeoPred)> {
        match self {
            Query::ConstScore { query, .. } | Query::Boost { query, .. } => query.geo_pred_part(),
            Query::Bool { must, should, must_not, filter, .. } if should.is_empty() => bool_geo_parts(must, must_not, filter),
            _ => self.geo_pred(),
        }
    }

    /// `match_all` (possibly wrapped): every live doc.
    pub fn is_match_all(&self) -> bool {
        match self {
            Query::MatchAll => true,
            Query::ConstScore { query, .. } | Query::Boost { query, .. } => query.is_match_all(),
            Query::Bool { must, should, must_not, filter, .. } => should.is_empty() && must_not.is_empty() && must.len() + filter.len() == 1 && must.iter().chain(filter).all(|q| q.is_match_all()),
            _ => false,
        }
    }

    /// The field and shape of a geo query.
    pub fn geo_shape(&self) -> Option<(u16, GeoShape)> {
        Some(match self {
            Query::GeoBoundingBox { field, top, left, bottom, right } => {
                (*field, GeoShape::BoundingBox { top: *top, left: *left, bottom: *bottom, right: *right })
            }
            Query::GeoDistance { field, lat, lon, meters } => (*field, GeoShape::Distance { lat: *lat, lon: *lon, meters: *meters }),
            Query::GeoPolygon { field, points } => (*field, GeoShape::Polygon { points: points.clone() }),
            _ => return None,
        })
    }

    /// ES `inner_hits` of a [`Query::Nested`] node for one matching
    /// `parent` on `leaf`: the parent's live children of the node's path
    /// that match the inner query, in doc order, with their
    /// `_nested.offset` and score. Empty for other queries.
    pub fn nested_inner_hits(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, parent: DocId) -> Vec<NestedHit> {
        self.nested_inner_hits_with(&mut None, leaf, ctx, parent)
    }

    /// [`Query::nested_inner_hits`] reusing the inner scorer of `cursor`
    /// when it is on the same leaf and `parent` comes after the previous
    /// call's parent (hits rendered in doc order then share one scorer,
    /// built once per leaf instead of once per hit).
    pub fn nested_inner_hits_with(&self, cursor: &mut Option<InnerHitsCursor>, leaf: &LeafReader<'_>, ctx: &SearchContext, parent: DocId) -> Vec<NestedHit> {
        let Query::Nested { path, query, .. } = self else {
            return Vec::new();
        };
        let Some(info) = leaf.nested_info() else { return Vec::new() };
        let Some(ord) = info.path_ord(path) else { return Vec::new() };
        let start = info.block_start(parent);
        // A numeric range child (Rally `nested`'s `range answers.date`):
        // check the parent's block directly, through the range's cached
        // segment bitset or else the doc values, instead of building the
        // range's docs over the whole segment for a few parents.
        if let Some((field, lo, hi, score)) = query.nested_range_child() {
            if let (LeafReader::Segment(seg), Some(values)) = (*leaf, I64Values::open(leaf, field)) {
                let (_, kind, kf, klo, khi) = NumericRange::I64(field, lo, hi).key(leaf.id());
                let bits = if lo <= hi { seg.peek_filter_bits((kind, kf, klo, khi)) } else { None };
                let dels = leaf.deletions_only();
                let mut out = Vec::new();
                for d in start..parent {
                    if info.ord(d) == ord
                        && !dels.as_ref().is_some_and(|b| b.get(d))
                        && lo <= hi
                        && match &bits {
                            Some(b) => word_bit(b, d),
                            None => values.any_in(d, lo, hi),
                        }
                    {
                        out.push(NestedHit { doc: d, offset: info.offset(d), score });
                    }
                }
                return out;
            }
        }
        let reuse = cursor.as_ref().is_some_and(|c| c.leaf == leaf.id() && c.last < start);
        if !reuse {
            // Inner joins inside `query` are relative to this node's path.
            let prev = NESTED_PARENT_DEPTH.with(|c| c.replace(info.ord_depth(ord)));
            let s = match &**query {
                Query::Custom(c) => c.inner_hits_scorer(leaf, ctx),
                q => q.scorer(leaf, ctx),
            };
            NESTED_PARENT_DEPTH.with(|c| c.set(prev));
            *cursor = s.map(|s| InnerHitsCursor { scorer: s, leaf: leaf.id(), last: 0 });
        }
        let Some(cur) = cursor.as_mut() else { return Vec::new() };
        cur.last = parent;
        let s = &mut cur.scorer;
        let dels = leaf.deletions_only();
        let mut out = Vec::new();
        // A reused scorer sits on the first doc the previous call did not consume.
        let mut d = if reuse && s.doc() >= start { s.doc() } else { s.seek(start) };
        while d != TERMINATED && d < parent {
            if info.ord(d) == ord && !dels.as_ref().is_some_and(|b| b.get(d)) {
                out.push(NestedHit { doc: d, offset: info.offset(d), score: s.score() });
            }
            d = s.advance();
        }
        out
    }

    /// This k-NN node's top-k on one leaf: live, filtered, above the
    /// similarity threshold; unboosted scores.
    fn leaf_knn(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Vec<(DocId, f32)> {
        let Query::Knn { field, vector, k, num_candidates, filter, similarity_threshold, oversample, visit_percentage, .. } = self else {
            return Vec::new();
        };
        // Deleted docs only: a vector inside a `nested` object lives on a
        // child doc, which `live_docs` would hide (nested kNN joins the
        // children to their parents afterwards).
        let live = leaf.undeleted_docs();
        let ef = (*num_candidates).max(*k);
        let cfg = crate::vector::store::PlannerConfig { oversample: *oversample, visit_ratio: visit_percentage.map(|p| p / 100.0), ..Default::default() };
        let hits = match filter {
            None => {
                // Fast path: no filter and no deletions → skip the O(n)
                // point-map scan entirely.
                if leaf.deletions_only().is_none() {
                    leaf.knn_unfiltered(*field, vector, *k, ef, &cfg)
                } else {
                    leaf.knn_with(*field, vector, *k, ef, live, &cfg)
                }
            }
            Some(f) => {
                // Pre-filter: the filter's doc set as a bitset.
                let bits = match f.filter_bits(leaf,ctx) {Ok(bits)=>bits,Err(error)=>{if let crate::error::RusticError::Memory(reason)=error {ctx.record_memory_error(reason);} return Vec::new();}};
                if bits.iter().all(|&w| w == 0) {
                    return Vec::new();
                }
                leaf.knn_with(*field, vector, *k, ef, move |d| live(d) && (bits[d as usize / 64] >> (d % 64)) & 1 == 1, &cfg)
            }
        };
        let hits=match hits { Ok(hits)=>hits, Err(error)=>{
            match error { crate::error::RusticError::Memory(reason)=>ctx.record_memory_error(reason), other=>{*ctx.execution_error.lock()=Some(other.to_string());} }
            return Vec::new();
        }};
        hits.into_iter().filter(|(_, s)| similarity_threshold.is_none_or(|t| *s >= t)).collect()
    }

    /// This k-NN node's top-k over every segment leaf with ONE vector-store
    /// search, as `(score, leaf index, doc)`, filtered and thresholded like
    /// [`Query::leaf_knn`]. The store spans every segment of the index, so
    /// searching it once per leaf (as `leaf_knn` does) repeats the whole
    /// search per segment. As in [`crate::index::Index::knn_search`], the
    /// filter (and deletions) become one doc bitset per segment, resolved
    /// to point ordinals when selective. `None` (search per leaf) when no
    /// leaf is a segment or the segments do not share one store; memtable
    /// leaves are never covered.
    fn store_knn(&self, leaves: &[LeafReader<'_>], ctx: &SearchContext) -> Option<(Vec<(f32, usize, DocId)>, u64)> {
        use crate::vector::store::{DocFilter, OrdFilter, PlannerConfig, PointFilter};
        let Query::Knn { field, vector, k, num_candidates, filter, similarity_threshold, oversample, visit_percentage, .. } = self else {
            return None;
        };
        let segs: Vec<(usize, &crate::segment::Segment)> = leaves
            .iter()
            .enumerate()
            .filter_map(|(i, l)| match l {
                LeafReader::Segment(s) => Some((i, *s)),
                LeafReader::Memtable(_, ..) => None,
            })
            .collect();
        let store = segs.first()?.1.vector_store()?;
        if segs.iter().any(|(_, s)| !s.vector_store().is_some_and(|v| Arc::ptr_eq(v, store))) {
            return None;
        }
        let ef = (*num_candidates).max(*k);
        let cfg = PlannerConfig { oversample: *oversample, visit_ratio: visit_percentage.map(|p| p / 100.0), ..Default::default() };
        // Deleted docs only, as `leaf_knn`: a vector inside a `nested`
        // object lives on a child doc, which `live_docs` would hide.
        let restricted = filter.is_some() || segs.iter().any(|(i, _)| leaves[*i].deletions_only().is_some());
        let mut docs = DocFilter::new();
        let mut filter_memory=Vec::new();
        if restricted {
            for (i, seg) in &segs {
                let leaf = &leaves[*i];
                let mut words = match filter {
                    Some(f) => match f.filter_bits(leaf,ctx) {Ok(bits)=>{let(words,memory)=bits.into_parts();filter_memory.push(memory);words},Err(error)=>{if let crate::error::RusticError::Memory(reason)=error {ctx.record_memory_error(reason);} return Some((Vec::new(),0));}},
                    None => {
                        let n = seg.num_docs as usize;
                        match crate::admission::shared().reserve((n as u64).div_ceil(64).saturating_mul(8).saturating_add(128),"vector visibility words") {
                            Ok(memory)=>filter_memory.push(memory),Err(error)=>{ctx.record_memory_error(error.to_string());return Some((Vec::new(),0));}
                        }
                        let mut w = vec![u64::MAX; n.div_ceil(64)];
                        if let (Some(last), r @ 1..) = (w.last_mut(), n % 64) {
                            *last = (1u64 << r) - 1;
                        }
                        w
                    }
                };
                if let Some(dels) = leaf.deletions_only() {
                    for (w, d) in words.iter_mut().zip(dels.words()) {
                        *w &= !d;
                    }
                }
                docs.add_segment(seg.id, words);
            }
            if docs.is_empty() {
                return Some((Vec::new(), 0));
            }
        }
        // Selective filters: per-segment point ordinals, so exact searches
        // enumerate just the matches and graph checks are one bit test.
        let selective = filter.is_some() && {
            let total: usize = store.segments(*field).iter().map(|s| s.len() as usize).sum();
            docs.len() <= cfg.full_scan_threshold || (docs.len() as f32) < cfg.two_hop_fraction * total as f32
                || ((docs.len() as f32) < 0.6 * total as f32 && segs.iter().all(|(_, s)| s.point_identity_base(*field).is_some()))
        };
        // `bbq_disk`: the matches as one dense point set, so the posting
        // scan tests each entry with one bit (ES's accept-docs bits).
        let ivf_field = store.field_options(*field).ivf.is_some();
        // The dense set is one bit a point id, held for the search.
        let _point_bits_memory = if ivf_field && restricted {
            match crate::admission::shared().reserve(store.next_point_id().div_ceil(64).saturating_mul(8).saturating_add(128), "kNN filter point bits") {
                Ok(memory) => Some(memory),
                Err(crate::error::RusticError::Memory(reason)) => { ctx.record_memory_error(reason); return Some((Vec::new(), 0)); }
                Err(other) => { *ctx.execution_error.lock() = Some(other.to_string()); return Some((Vec::new(), 0)); }
            }
        } else {
            None
        };
        let point_bits: Option<crate::vector::store::PointBits> = if ivf_field && restricted {
            if filter.is_none() {
                // Deletions only: each segment's live points, cached on the
                // segment until its deletions change.
                let mut bits = crate::vector::store::PointBits::with_capacity(store.next_point_id());
                let words: std::collections::HashMap<u32, &[u64]> = docs.segments().collect();
                for (i, seg) in &segs {
                    let (Some(col), Some(w)) = (seg.point_map().column(*field), words.get(&seg.id)) else { continue };
                    let deleted = leaves[*i].deletions_only().map_or(0, |d| d.count());
                    match seg.cached_live_points(*field, deleted, || store.segment_point_words(w, &col, seg.point_runs(*field).as_deref())) {
                        Ok(lp) => bits.or_words(lp.first_word, &lp.words),
                        Err(crate::error::RusticError::Memory(reason)) => { ctx.record_memory_error(reason); return Some((Vec::new(), 0)); }
                        Err(other) => { *ctx.execution_error.lock() = Some(other.to_string()); return Some((Vec::new(), 0)); }
                    }
                }
                Some(bits)
            } else {
                Some(store.point_bits(&docs, |sid| {
                    let seg = segs.iter().find(|(_, s)| s.id == sid)?.1;
                    Some((seg.point_map().column(*field)?, seg.point_runs(*field)))
                }))
            }
        } else {
            None
        };
        let ords: Option<OrdFilter> = (selective && !ivf_field).then(|| {
            store.ord_filter(*field, std::mem::take(&mut docs), |sid| {
                let seg = segs.iter().find(|(_, s)| s.id == sid)?.1;
                Some((seg.point_map().column(*field)?, seg.point_identity_base(*field)))
            })
        });
        let pf = match (&ords, restricted) {
            _ if point_bits.is_some() => point_bits.as_ref().map(PointFilter::Points),
            (Some(o), _) => Some(PointFilter::Ords(o)),
            (None, true) => Some(PointFilter::Docs(&docs)),
            (None, false) => None,
        };
        let (hits, stats) = match store.search_filtered(*field, vector, *k, ef, pf, &cfg) {
            Ok(result)=>result, Err(error)=>{
                match error { crate::error::RusticError::Memory(reason)=>ctx.record_memory_error(reason), other=>{*ctx.execution_error.lock()=Some(other.to_string());} }
                return Some((Vec::new(),0));
            }
        };
        let hits = hits
            .into_iter()
            .filter(|h| similarity_threshold.is_none_or(|t| h.score >= t))
            .filter_map(|h| Some((h.score, segs.iter().find(|(_, s)| s.id == h.segment)?.0, h.doc)))
            .collect();
        Some((hits, stats.vector_ops))
    }

    /// The index-wide top-k of this k-NN node (ES semantics: `k` hits in
    /// total, not per leaf), split back into per-leaf shares keyed by
    /// `(node, leaf id)`, and the vector operations it took. Ties break on
    /// leaf order, then doc.
    fn global_knn(&self, leaves: &[LeafReader<'_>], ctx: &SearchContext) -> (Vec<((usize, usize), Vec<(DocId, f32)>)>, u64) {
        let Query::Knn { k, .. } = self else {
            return (Vec::new(), 0);
        };
        #[cfg(feature = "mutable-hnsw-poc")]
        if let Query::Knn { field, vector, num_candidates, filter, similarity_threshold, oversample, .. } = self {
            match super::mutable_knn::search(leaves, *field, vector, *k, (*num_candidates).max(*k), filter.as_deref(), ctx, false, *oversample) {
                Ok(Some((hits, stats))) => {
                    let mut per_leaf = vec![Vec::new(); leaves.len()];
                    for (score, leaf, doc) in hits {
                        if similarity_threshold.is_none_or(|limit| score >= limit) { per_leaf[leaf].push((doc, score)); }
                    }
                    return (leaves.iter().zip(per_leaf).map(|(leaf, mut hits)| {
                        hits.sort_by_key(|hit| hit.0); ((self.node_key(), leaf.id()), hits)
                    }).collect(), stats.vector_ops);
                }
                Err(error) => {
                    // A deadline or a cancellation stops the kNN as a timeout
                    // (the search's caller reports a cancellation).
                    if ctx.stopped() {
                        ctx.vector_timed_out.store(true, std::sync::atomic::Ordering::Relaxed);
                    } else {
                        *ctx.execution_error.lock() = Some(error.to_string());
                    }
                    return (Vec::new(), 0);
                }
                Ok(None) => {}
            }
        }
        let mut all: Vec<(f32, usize, DocId)> = Vec::new();
        let (from_store, mut ops) = match self.store_knn(leaves, ctx) {
            Some((hits, ops)) => (Some(hits), ops),
            None => (None, 0),
        };
        let store_covered = from_store.is_some();
        all.extend(from_store.into_iter().flatten());
        // The rest (memtables, segments without a shared store) runs here;
        // its operations are counted on this thread (a per-leaf store
        // search fanned out to other threads is not).
        let ops0 = crate::vector::hnsw::vector_ops();
        for (i, leaf) in leaves.iter().enumerate() {
            if store_covered && matches!(leaf, LeafReader::Segment(_)) {
                continue;
            }
            all.extend(self.leaf_knn(leaf, ctx).into_iter().map(|(d, s)| (s, i, d)));
        }
        ops += crate::vector::hnsw::vector_ops().wrapping_sub(ops0);
        all.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        all.truncate(*k);
        let mut per_leaf: Vec<Vec<(DocId, f32)>> = vec![Vec::new(); leaves.len()];
        for (s, i, d) in all {
            per_leaf[i].push((d, s));
        }
        let shares = leaves
            .iter()
            .zip(per_leaf)
            .map(|(leaf, mut v)| {
                v.sort_by_key(|(d, _)| *d);
                ((self.node_key(), leaf.id()), v)
            })
            .collect();
        (shares, ops)
    }

    /// Score of `doc` under this query on `leaf`, or `None` if it does not
    /// match (live-doc filtering is the caller's job).
    pub fn score_at(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, doc: DocId) -> Option<f32> {
        let mut s = self.scorer(leaf, ctx)?;
        if s.seek(doc) == doc {
            Some(s.score())
        } else {
            None
        }
    }

    /// Explain how `doc` on `leaf` was scored (Lucene `Weight.explain`).
    /// Term, phrase, span, bool, boost, constant-score, dis-max and
    /// function-score queries return a full tree; other queries a one-line
    /// node. A non-matching doc yields a 0-valued "no match" node.
    pub fn explain(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, doc: DocId) -> Explanation {
        self.explain_b(leaf, ctx, doc, 1.0)
    }

    /// [`Query::explain`] under the `boost` of the enclosing `BoostQuery`s,
    /// folded into the term weights (see [`Query::scorer_b`]).
    pub fn explain_b(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, doc: DocId, boost: f32) -> Explanation {
        let score = self.scorer_b(leaf, ctx, boost).and_then(|mut s| (s.seek(doc) == doc).then(|| s.score()));
        let Some(score) = score else {
            return Explanation::no_match(format!("{} query", self.kind_name()));
        };
        // Queries that cannot take the boost into their weights are scaled afterwards.
        let absorbs = match self {
            Query::Term(_) | Query::Phrase { .. } | Query::Boost { .. } | Query::Bool { .. } | Query::DisMax { .. } => true,
            Query::MultiPhrase { positions, .. } => positions.len() > 1,
            Query::Custom(c) => c.absorbs_boost(),
            _ => false,
        };
        if boost != 1.0 && !absorbs {
            return Explanation::new(score, "product of:", vec![self.explain_b(leaf, ctx, doc, 1.0), Explanation::leaf(boost, "boost")]);
        }
        match self {
            Query::Term(t) => leaf
                .explain_term(t, ctx.sim(t.field), boost, ctx.stats(t), ctx.fstats(t.field), doc)
                .unwrap_or_else(|| Explanation::leaf(score, format!("term {t}"))),
            Query::Phrase { terms, .. } if terms.len() == 1 => Query::Term(terms[0].clone()).explain_b(leaf, ctx, doc, boost),
            Query::Phrase { terms, slop } => {
                let positions: Vec<Vec<Term>> = terms.iter().map(|t| vec![t.clone()]).collect();
                explain_phrase(&positions, *slop, leaf, ctx, doc, score, boost)
            }
            Query::MultiPhrase { positions, slop, .. } if positions.len() > 1 => explain_phrase(positions, *slop, leaf, ctx, doc, score, boost),
            Query::Span(sq) => explain_span(ctx.span_expansions.get(&self.node_key()).map(Arc::as_ref).unwrap_or(sq), leaf, ctx, doc, score),
            Query::Intervals { field, rule } => {
                let expanded = ctx.interval_expansions.get(&(self.node_key(),leaf.id()));
                let fallback = rule.to_span(*field);
                explain_span_scorer(interval_scorer(rule,expanded.map(Arc::as_ref).unwrap_or(&fallback),leaf,ctx),doc,score)
            },
            Query::Bool { must, should, filter, .. } => {
                let mut details = Vec::new();
                for q in must.iter().chain(should) {
                    if q.scorer_b(leaf, ctx, boost).is_some_and(|mut s| s.seek(doc) == doc) {
                        details.push(q.explain_b(leaf, ctx, doc, boost));
                    }
                }
                for q in filter {
                    details.push(Explanation::new(0.0, "match on required clause, product of: (filter, score ignored)", vec![q.explain(leaf, ctx, doc)]));
                }
                Explanation::new(score, "sum of:", details)
            }
            Query::Boost { query, boost: b } => query.explain_b(leaf, ctx, doc, boost * *b),
            Query::ConstScore { query, .. } => {
                // Lucene `ConstantScoreWeight.explain`: `ConstantScore(..)^score`.
                let sc = if score == 1.0 {
                    String::new()
                } else if score.fract() == 0.0 && score.abs() < 1e7 {
                    format!("^{score:.1}")
                } else {
                    format!("^{score}")
                };
                let inner = match query.as_ref() {
                    // Renamed to the mapped field by the ES layer.
                    Query::Term(t) => format!("field{}:{}", t.field, String::from_utf8_lossy(&t.text)),
                    q => q.kind_name().to_string(),
                };
                Explanation::leaf(score, format!("ConstantScore({inner}){sc}"))
            }
            Query::DisMax { queries, tie_breaker } => {
                let details = queries.iter().filter(|q| q.scorer_b(leaf, ctx, boost).is_some_and(|mut s| s.seek(doc) == doc)).map(|q| q.explain_b(leaf, ctx, doc, boost)).collect();
                Explanation::new(score, format!("max plus {tie_breaker} times others of:"), details)
            }
            Query::FunctionScore { query, functions, score_mode, boost_mode, max_boost, min_score } => {
                let params = FunctionScoreParams { score_mode: *score_mode, boost_mode: *boost_mode, max_boost: *max_boost, min_score: *min_score };
                let funcs = PreparedFunctions::prepare(functions, params, leaf, ctx);
                funcs.explain(doc, query.explain(leaf, ctx, doc))
            }
            Query::Custom(c) => c.explain_b(leaf, ctx, doc, score, boost),
            // ES `KnnScoreDocQuery`: the kNN winners as fixed docs.
            Query::Knn { .. } => Explanation::leaf(score, "within top k documents"),
            _ => Explanation::leaf(score, format!("{} query match", self.kind_name())),
        }
    }
}

fn explain_phrase(positions: &[Vec<Term>], slop: u32, leaf: &LeafReader<'_>, ctx: &SearchContext, doc: DocId, score: f32, boost: f32) -> Explanation {
    let field = positions[0][0].field;
    let stats: Vec<TermStats> = positions.iter().flatten().map(|t| ctx.stats(t)).collect();
    if let Some(mut p) = leaf.multi_phrase_raw_boosted(positions, slop, ctx.sim(field), &stats, ctx.fstats(field), boost, &|t| leaf.postings(t)) {
        if p.seek(doc) == doc {
            let e = p.weight().explain(p.freq(), p.field_len());
            return Explanation::new(e.value, format!("weight({} in {doc}) [{}]", phrase_text(positions, slop), ctx.sim(field).name()), vec![e]);
        }
    }
    Explanation::leaf(score, "phrase match")
}

/// Lucene's `PhraseQuery` / `MultiPhraseQuery` `toString`: `field:"a b"` with
/// `(x y)` for alternatives at one position and `~slop` when sloppy.
fn phrase_text(positions: &[Vec<Term>], slop: u32) -> String {
    let field = positions[0][0].field;
    let word = |t: &Term| String::from_utf8_lossy(&t.text).into_owned();
    let parts: Vec<String> = positions
        .iter()
        .map(|alts| if alts.len() == 1 { word(&alts[0]) } else { format!("({})", alts.iter().map(word).collect::<Vec<_>>().join(" ")) })
        .collect();
    let mut s = format!("field{field}:\"{}\"", parts.join(" "));
    if slop > 0 {
        s.push_str(&format!("~{slop}"));
    }
    s
}

fn explain_span(sq: &SpanQuery, leaf: &LeafReader<'_>, ctx: &SearchContext, doc: DocId, score: f32) -> Explanation {
    explain_span_scorer(span_scorer(sq,leaf,ctx),doc,score)
}
fn explain_span_scorer(scorer:Option<SpanScorer>, doc:DocId, score:f32) -> Explanation {
    if let Some(mut s) = scorer {
        if s.seek(doc) == doc {
            let e = s.weight().explain(s.freq(), s.field_len());
            return Explanation::new(e.value, format!("weight(span in {doc}), spanFreq={}", s.freq()), vec![e]);
        }
    }
    Explanation::leaf(score, "span match")
}

/// Spans of `sq` on `leaf`, scored through the field's similarity with the
/// summed statistics of its terms.
fn span_scorer(sq: &SpanQuery, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<SpanScorer> {
    let mut terms = Vec::new();
    sq.collect_terms(&mut terms);
    let mut stats: Vec<TermStats> = terms.iter().map(|t| ctx.stats(t)).collect();
    sq.expanded_stats(leaf, &mut stats);
    span_scorer_with_stats(sq,leaf,ctx,stats)
}
fn interval_scorer(rule:&IntervalsRule, expanded:&SpanQuery, leaf:&LeafReader<'_>, ctx:&SearchContext) -> Option<SpanScorer> {
    let mut terms = Vec::new();
    rule.to_span(expanded.field()?).collect_terms(&mut terms);
    let mut stats:Vec<TermStats> = terms.iter().map(|term|ctx.stats(term)).collect();
    terms.clear();
    rule.expanded_terms(expanded,&mut terms);
    stats.extend(terms.iter().map(|term|TermStats {
        doc_freq:leaf.doc_freq(term) as u64,
        total_term_freq:leaf.total_term_freq(term), num_docs:leaf.num_docs() as u64,
    }));
    span_scorer_with_stats(expanded,leaf,ctx,stats)
}
fn span_scorer_with_stats(sq:&SpanQuery, leaf:&LeafReader<'_>, ctx:&SearchContext, mut stats:Vec<TermStats>) -> Option<SpanScorer> {
    let field = sq.field()?;
    let spans = sq.spans(leaf)?;
    let mut fs = ctx.fstats(field);
    if fs.num_docs == 0 {
        fs = leaf.field_stats(field);
    }
    if stats.is_empty() {
        stats.push(TermStats { doc_freq: 1, total_term_freq: 1, num_docs: leaf.num_docs() as u64 });
    }
    let weight = ctx.sim(field).phrase_weight(&stats, fs, 1.0);
    Some(SpanScorer::new(spans, weight, Arc::new(leaf.norms(field))))
}

thread_local! {
    /// Depth of the parent level of the `Nested` node whose inner scorer is
    /// being built (0 = root documents): nested joins are relative to their
    /// enclosing join, as ES parent filters are.
    static NESTED_PARENT_DEPTH: std::cell::Cell<u16> = const { std::cell::Cell::new(0) };
}

/// Block join: run `query` over the leaf, keep the live docs of nested
/// `path`, and fold each run of children into their parent's score.
fn nested_scorer(path: &str, query: &Query, mode: NestedScoreMode, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
    let info = leaf.nested_info()?;
    let ord = info.path_ord(path)?;
    let target = NESTED_PARENT_DEPTH.with(|c| c.get());
    let depth = info.ord_depth(ord);
    if depth <= target {
        return None;
    }
    // Root parents of a numeric range over the children (Rally `nested`'s
    // `range answers.date`): verify each candidate parent's block lazily
    // instead of collecting the range's docs over the whole segment.
    if target == 0 {
        if let Some(s) = range_block_join(info, ord, query, mode, leaf) {
            return s;
        }
    }
    let prev = NESTED_PARENT_DEPTH.with(|c| c.replace(depth));
    let s = query.scorer(leaf, ctx);
    NESTED_PARENT_DEPTH.with(|c| c.set(prev));
    let s = s?;
    let bound = nested_bound(mode, s.score_upper_bound());
    // SAFETY: the scorer never outlives the search holding the leaf (the
    // contract every leaf-bound scorer relies on); the nested ordinals are
    // owned by the leaf's segment / memtable.
    let info: &'static crate::nested::NestedInfo = unsafe { std::mem::transmute::<&crate::nested::NestedInfo, &'static crate::nested::NestedInfo>(info) };
    Some(Box::new(BlockJoinScorer { child: s, info, ord, target, mode, dels: leaf.deletions_only(), pending: None, doc: TERMINATED, score: 0.0, bound }))
}

/// The score bound of a block join whose child scores are at most
/// `child` (`None`: unknown, or the child may not be skipped): `avg`,
/// `max` and `min` of child scores never exceed the largest one (the
/// average is folded in `f64` and rounds back to at most the bound),
/// `none` scores 0, `sum` is unbounded.
fn nested_bound(mode: NestedScoreMode, child: Option<f32>) -> Option<f32> {
    let child = child.filter(|b| b.is_finite() && *b >= 0.0)?;
    match mode {
        NestedScoreMode::None => Some(0.0),
        NestedScoreMode::Sum => None,
        NestedScoreMode::Avg | NestedScoreMode::Max | NestedScoreMode::Min => Some(child),
    }
}

/// The child scores of one parent folded as Lucene's block join scorer
/// does: in `double`, cast to `float` at the end.
#[derive(Clone, Copy)]
struct NestedAcc {
    sum: f64,
    min: f64,
    max: f64,
    n: u32,
}

impl NestedAcc {
    #[inline]
    fn new() -> Self {
        Self { sum: 0.0, min: f64::INFINITY, max: f64::NEG_INFINITY, n: 0 }
    }

    #[inline]
    fn add(&mut self, score: f32) {
        let s = score as f64;
        self.sum += s;
        self.min = self.min.min(s);
        self.max = self.max.max(s);
        self.n += 1;
    }

    #[inline]
    fn finish(&self, mode: NestedScoreMode) -> f32 {
        match mode {
            NestedScoreMode::Avg => (self.sum / self.n.max(1) as f64) as f32,
            NestedScoreMode::Max => self.max as f32,
            NestedScoreMode::Min => self.min as f32,
            NestedScoreMode::Sum => self.sum as f32,
            NestedScoreMode::None => 0.0,
        }
    }
}

/// A nested query's inner scorer kept across the inner hits of
/// consecutive parents (see [`Query::nested_inner_hits_with`]).
pub struct InnerHitsCursor {
    scorer: Box<dyn Scorer>,
    leaf: usize,
    last: DocId,
}

/// Lucene `ToParentBlockJoinQuery`'s scorer: parents (at nested depth
/// `target`) of the live children of path `ord` the child scorer matches,
/// scored with `mode` over those children. Lazy: a conjunction seeking it
/// to candidate parents only visits the children of those parents.
struct BlockJoinScorer {
    child: Box<dyn Scorer>,
    info: &'static crate::nested::NestedInfo,
    ord: u16,
    target: u16,
    mode: NestedScoreMode,
    dels: Option<Arc<crate::postings::BitSet>>,
    /// The child scorer's current doc when it is ahead of the last group.
    pending: Option<DocId>,
    /// Current parent (`TERMINATED` before the first advance).
    doc: DocId,
    score: f32,
    /// See [`nested_bound`].
    bound: Option<f32>,
}

impl BlockJoinScorer {
    #[inline]
    fn valid(&self, d: DocId) -> bool {
        self.info.ord(d) == self.ord && !self.dels.as_ref().is_some_and(|b| b.get(d))
    }

    /// The next group starting at child `d` (the child scorer is on `d`).
    fn group_from(&mut self, mut d: DocId) -> DocId {
        loop {
            if d == TERMINATED {
                self.pending = Some(TERMINATED);
                self.doc = TERMINATED;
                return TERMINATED;
            }
            if self.valid(d) {
                if let Some(p) = self.info.ancestor_at(d, self.target) {
                    let mut acc = NestedAcc::new();
                    acc.add(if self.mode == NestedScoreMode::None { 0.0 } else { self.child.score() });
                    let mut c = self.child.advance();
                    while c != TERMINATED && c < p {
                        if self.valid(c) {
                            acc.add(if self.mode == NestedScoreMode::None { 0.0 } else { self.child.score() });
                        }
                        c = self.child.advance();
                    }
                    self.pending = Some(c);
                    self.doc = p;
                    self.score = acc.finish(self.mode);
                    return p;
                }
            }
            d = self.child.advance();
        }
    }
}

impl DocSet for BlockJoinScorer {
    fn advance(&mut self) -> DocId {
        let d = match self.pending {
            Some(d) => d,
            None => self.child.advance(),
        };
        self.group_from(d)
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.pending.is_some() && self.doc >= target {
            return self.doc;
        }
        if self.target != 0 {
            let mut d = self.advance();
            while d < target {
                d = self.advance();
            }
            return d;
        }
        // Root parents: the first root at or after `target` owns the block
        // its children start.
        if target >= self.info.num_docs() {
            self.pending = Some(TERMINATED);
            self.doc = TERMINATED;
            return TERMINATED;
        }
        let root = if self.info.is_root(target) { target } else { self.info.root(target) };
        let start = self.info.block_start(root);
        let d = match self.pending {
            Some(d) if d >= start => d,
            _ => self.child.seek(start),
        };
        let mut p = self.group_from(d);
        while p != TERMINATED && p < target {
            p = self.advance();
        }
        p
    }

    fn doc(&self) -> DocId {
        self.doc
    }

    fn size_hint(&self) -> u32 {
        self.child.size_hint()
    }
}

impl Scorer for BlockJoinScorer {
    fn score(&mut self) -> f32 {
        self.score
    }

    fn score_upper_bound(&self) -> Option<f32> {
        self.bound
    }
}

/// The i64 doc values of one field on a sealed segment, matched exactly as
/// [`LeafReader::range_i64_bits`] matches them (a single-valued column
/// first, else a multi-valued one; no column: no doc matches).
enum I64Values<'a> {
    Single(crate::codec::fastfields::ColumnReader<'a, i64>),
    Multi(crate::codec::fastfields::MultiColumnReader<'a, i64>),
    Missing,
}

impl<'a> I64Values<'a> {
    /// `None` for memtable leaves.
    fn open(leaf: &LeafReader<'a>, field: u16) -> Option<I64Values<'a>> {
        let LeafReader::Segment(seg) = *leaf else { return None };
        let ff = seg.fast_fields();
        Some(if let Some(c) = ff.get_i64(field) {
            I64Values::Single(c)
        } else if let Some(c) = ff.get_multi_i64(field) {
            I64Values::Multi(c)
        } else {
            I64Values::Missing
        })
    }

    /// Whether `doc` has a value in `[lo, hi]`.
    #[inline]
    fn any_in(&self, doc: DocId, lo: i64, hi: i64) -> bool {
        match self {
            I64Values::Single(c) => matches!(c.get(doc), Some(v) if v >= lo && v <= hi),
            I64Values::Multi(c) => doc < c.num_docs() && c.values(doc).any(|v| v >= lo && v <= hi),
            I64Values::Missing => false,
        }
    }
}

#[inline]
fn word_bit(words: &[u64], doc: DocId) -> bool {
    words.get(doc as usize >> 6).is_some_and(|w| (w >> (doc & 63)) & 1 == 1)
}

/// The first set bit at or after `from` (`TERMINATED` when none).
#[inline]
fn next_word_bit(words: &[u64], from: DocId) -> DocId {
    let mut w = from as usize >> 6;
    if w >= words.len() {
        return TERMINATED;
    }
    let mut word = words[w] & (u64::MAX << (from & 63));
    loop {
        if word != 0 {
            return (w * 64) as DocId + word.trailing_zeros();
        }
        w += 1;
        if w >= words.len() {
            return TERMINATED;
        }
        word = words[w];
    }
}

/// [`range_block_join`]'s scorer: root parents with a live child of path
/// `ord` holding a value of `field` in `[lo, hi]` (Lucene's
/// `ToParentBlockJoinQuery` over an `IndexOrDocValuesQuery` range: as a
/// conjunction's verifier it reads the doc values of the candidate
/// parent's block only, ES's doc-values side). Every matching child scores
/// `child_score`, folded with `mode` like [`BlockJoinScorer`]. Iterated on
/// its own (`advance` / `seek`) it walks the range's segment bitset, the
/// cached one ([`crate::segment::Segment::cached_filter_bits`]) as the plain range query
/// builds it.
struct RangeBlockJoin {
    info: &'static crate::nested::NestedInfo,
    ord: u16,
    mode: NestedScoreMode,
    child_score: f32,
    field: u16,
    lo: i64,
    hi: i64,
    values: I64Values<'static>,
    /// The range's docs over the segment, when cached or once built.
    bits: Option<Arc<crate::admission::Allocation<Vec<u64>>>>,
    /// Building `bits` was refused (memory budget): verify by values.
    refused: bool,
    leaf: LeafReader<'static>,
    /// The segment filter-cache key of the range's bitset.
    key: crate::segment::FilterCacheKey,
    dels: Option<Arc<crate::postings::BitSet>>,
    num_docs: u32,
    /// Estimated docs in the range (size hint).
    est: u32,
    /// A broad range verifies candidates (`seek_exact`) rather than lead.
    fast_exact: bool,
    /// Where the next `advance` resumes (no matching parent before it is
    /// still to come).
    pos: DocId,
    doc: DocId,
    done: bool,
    score: f32,
}

/// The block join of a numeric (i64 / date) range child on a sealed
/// segment with root parents: `None` when `query` is not such a range or
/// the leaf is not a sealed segment (the generic block join then runs),
/// `Some(None)` when nothing can match.
fn range_block_join(info: &crate::nested::NestedInfo, ord: u16, query: &Query, mode: NestedScoreMode, leaf: &LeafReader<'_>) -> Option<Option<Box<dyn Scorer>>> {
    let (field, lo, hi, child_score) = query.nested_range_child()?;
    let LeafReader::Segment(seg) = *leaf else { return None };
    let values = I64Values::open(leaf, field)?;
    if matches!(values, I64Values::Missing) || lo > hi {
        return Some(None);
    }
    let (_, kind, kf, klo, khi) = NumericRange::I64(field, lo, hi).key(leaf.id());
    let key = (kind, kf, klo, khi);
    let bits = seg.peek_filter_bits(key);
    let num_docs = leaf.num_docs();
    let est: u64 = match (&values, &bits) {
        (I64Values::Single(c), _) if c.counts_from_points() => c.count_range(lo, hi),
        (_, Some(b)) => b.iter().map(|w| w.count_ones() as u64).sum(),
        _ => num_docs as u64,
    };
    if est == 0 {
        return Some(None);
    }
    // SAFETY: the scorer never outlives the search holding the leaf (the
    // contract every leaf-bound scorer relies on); the nested ordinals and
    // the doc-value columns are owned by the leaf's segment.
    let info: &'static crate::nested::NestedInfo = unsafe { std::mem::transmute::<&crate::nested::NestedInfo, &'static crate::nested::NestedInfo>(info) };
    let values: I64Values<'static> = unsafe { std::mem::transmute::<I64Values<'_>, I64Values<'static>>(values) };
    let leaf_s: LeafReader<'static> = unsafe { std::mem::transmute::<LeafReader<'_>, LeafReader<'static>>(*leaf) };
    Some(Some(Box::new(RangeBlockJoin {
        info,
        ord,
        mode,
        child_score,
        field,
        lo,
        hi,
        values,
        bits,
        refused: false,
        leaf: leaf_s,
        key,
        dels: leaf.deletions_only(),
        num_docs,
        est: est.min(num_docs as u64) as u32,
        // More than 1/16 of the segment: collecting the range's docs would
        // cost more than checking the blocks of a selective lead.
        fast_exact: est.saturating_mul(16) > num_docs as u64,
        pos: 0,
        doc: TERMINATED,
        done: false,
        score: 0.0,
    })))
}

impl RangeBlockJoin {
    #[inline]
    fn live(&self, d: DocId) -> bool {
        !self.dels.as_ref().is_some_and(|b| b.get(d))
    }

    #[inline]
    fn child_matches(&self, d: DocId) -> bool {
        self.info.ord(d) == self.ord
            && self.live(d)
            && match &self.bits {
                Some(b) => word_bit(b, d),
                None => self.values.any_in(d, self.lo, self.hi),
            }
    }

    /// Root `p`'s score over its block's matching children (`None` when
    /// none matches).
    fn score_parent(&self, p: DocId) -> Option<f32> {
        let sc = if self.mode == NestedScoreMode::None { 0.0 } else { self.child_score };
        let mut acc = NestedAcc::new();
        for d in self.info.block_start(p)..p {
            if self.child_matches(d) {
                acc.add(sc);
            }
        }
        (acc.n > 0).then(|| acc.finish(self.mode))
    }

    fn ensure_bits(&mut self) {
        if self.bits.is_some() || self.refused {
            return;
        }
        let leaf = self.leaf;
        let (field, lo, hi) = (self.field, self.lo, self.hi);
        match leaf {
            LeafReader::Segment(seg) => match seg.cached_filter_bits(self.key, || NumericRange::I64(field, lo, hi).bits(&leaf)) {
                Ok(b) => self.bits = Some(b),
                Err(_) => self.refused = true,
            },
            LeafReader::Memtable(..) => self.refused = true,
        }
    }

    fn finish(&mut self) -> DocId {
        self.done = true;
        self.doc = TERMINATED;
        self.pos = self.num_docs;
        TERMINATED
    }

    fn found(&mut self, p: DocId, score: f32) -> DocId {
        self.doc = p;
        self.score = score;
        self.pos = p + 1;
        p
    }

    /// The first matching parent at or after `pos`.
    fn next_from_pos(&mut self) -> DocId {
        if self.done || self.pos >= self.num_docs {
            return self.finish();
        }
        self.ensure_bits();
        let root = if self.info.is_root(self.pos) { self.pos } else { self.info.root(self.pos) };
        if !self.info.is_root(root) {
            return self.finish();
        }
        if let Some(bits) = self.bits.clone() {
            let mut d = self.info.block_start(root);
            loop {
                d = next_word_bit(&bits, d);
                if d == TERMINATED || d >= self.num_docs {
                    return self.finish();
                }
                if self.info.ord(d) == self.ord && self.live(d) {
                    let p = self.info.root(d);
                    if !self.info.is_root(p) {
                        return self.finish();
                    }
                    if let Some(s) = self.score_parent(p) {
                        return self.found(p, s);
                    }
                    d = p + 1;
                } else {
                    d += 1;
                }
            }
        }
        // No bitset (refused): each root from `root` on, by its values.
        let mut p = root;
        while p < self.num_docs {
            if self.info.is_root(p) {
                if let Some(s) = self.score_parent(p) {
                    return self.found(p, s);
                }
            }
            p += 1;
        }
        self.finish()
    }
}

impl DocSet for RangeBlockJoin {
    fn has_fast_exact_seek(&self) -> bool {
        self.fast_exact
    }

    fn seek_exact(&mut self, target: DocId) -> bool {
        self.pos = self.pos.max(target.saturating_add(1));
        if self.done || target >= self.num_docs || !self.info.is_root(target) {
            self.doc = TERMINATED;
            return false;
        }
        match self.score_parent(target) {
            Some(s) => {
                self.doc = target;
                self.score = s;
                true
            }
            None => {
                self.doc = TERMINATED;
                false
            }
        }
    }

    fn advance(&mut self) -> DocId {
        if self.doc != TERMINATED {
            self.pos = self.pos.max(self.doc + 1);
        }
        self.next_from_pos()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.done {
            return TERMINATED;
        }
        if self.doc != TERMINATED && self.doc >= target {
            return self.doc;
        }
        self.pos = self.pos.max(target);
        self.next_from_pos()
    }

    fn doc(&self) -> DocId {
        self.doc
    }

    fn size_hint(&self) -> u32 {
        self.est
    }
}

impl Scorer for RangeBlockJoin {
    fn score(&mut self) -> f32 {
        self.score
    }

    fn score_upper_bound(&self) -> Option<f32> {
        // A numeric range never fails on a later doc: skipping is safe.
        nested_bound(self.mode, Some(self.child_score))
    }
}

/// Return a copy of `q` with `filter: true` set on any `Query::Nested`
/// (possibly wrapped in `Boost`/`ConstScore`) so its scorer skips per-child
/// scoring. Non-nested queries are returned unchanged.
fn with_nested_filter(q: Query) -> Query {
    match q {
        Query::Nested { path, query, score_mode, .. } => {
            Query::Nested { path, query, score_mode, filter: true }
        }
        Query::Boost { query, boost } => {
            Query::Boost { query: Box::new(with_nested_filter(*query)), boost }
        }
        Query::ConstScore { query, score, point_set } => {
            Query::ConstScore { query: Box::new(with_nested_filter(*query)), score, point_set }
        }
        _ => q,
    }
}

/// Filter-context nested scorer: the nested query is used only for matching
/// (e.g. inside a `bool.filter` or as a `must` clause whose score is
/// irrelevant), so child scores are never computed. Walks the inner scorer
/// once, collects the matching parents into a bitset, and returns a
/// zero-score `ConstScorer` over it. Skips the per-child `s.score()` calls
/// (expensive BM25) and the `Vec<(DocId, f32)>` materialization of
/// [`nested_scorer`].
fn nested_filter_scorer(path: &str, query: &Query, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
    let info = leaf.nested_info()?;
    let ord = info.path_ord(path)?;
    let target = NESTED_PARENT_DEPTH.with(|c| c.get());
    let depth = info.ord_depth(ord);
    if depth <= target {
        return None;
    }
    // Root parents of a numeric child range: matched lazily per parent
    // block (score 0, as below) instead of collecting every parent.
    if target == 0 {
        if let Some(s) = range_block_join(info, ord, query, NestedScoreMode::None, leaf) {
            return s;
        }
    }
    let prev = NESTED_PARENT_DEPTH.with(|c| c.replace(depth));
    let s = query.scorer(leaf, ctx);
    NESTED_PARENT_DEPTH.with(|c| c.set(prev));
    let mut s = s?;
    let nwords = (leaf.num_docs() as usize / 64) + 1;
    let mut words = vec![0u64; nwords];
    loop {
        let d = s.advance();
        if d == TERMINATED {
            break;
        }
        if info.ord(d) != ord || leaf.is_deleted(d) {
            continue;
        }
        let Some(p) = info.ancestor_at(d, target) else { continue };
        words[p as usize / 64] |= 1u64 << (p % 64);
    }
    if words.iter().all(|&w| w == 0) {
        return None;
    }
    Some(Box::new(ConstScorer::new(BitSetDocSet::new(words), 0.0)))
}

/// [`Query::geo_pred`] of a `bool` without `should` clauses.
fn bool_geo_pred(must: &[Query], must_not: &[Query], filter: &[Query]) -> Option<(u16, crate::geo::GeoPred)> {
    if must.is_empty() && filter.is_empty() {
        return None;
    }
    let (field, pred) = bool_geo_parts(must, must_not, filter)?;
    pred.requires_point().then_some((field, pred))
}

/// The point predicate of a `bool` without `should` clauses, also when it
/// holds for docs without a point (a nested `bool` of `match_all` and
/// `must_not`, ES's pure-negative form: Rally `distanceFilterCount200x300-alt`'s
/// ring): only valid as a part of a predicate that requires a point.
/// `match_all` clauses in `must` / `filter` restrict nothing and are left
/// out. `None` when a clause is not a predicate on the same geo field, or
/// nothing but `match_all` is left.
fn bool_geo_parts(must: &[Query], must_not: &[Query], filter: &[Query]) -> Option<(u16, crate::geo::GeoPred)> {
    use crate::geo::GeoPred;
    let mut field = None;
    let mut parts = Vec::with_capacity(must.len() + filter.len() + must_not.len());
    for (q, neg) in must.iter().chain(filter).map(|q| (q, false)).chain(must_not.iter().map(|q| (q, true))) {
        if !neg && q.is_match_all() {
            continue;
        }
        let (f, p) = q.geo_pred_part()?;
        if field.is_some_and(|x| x != f) {
            return None;
        }
        field = Some(f);
        parts.push(if neg { GeoPred::Not(Box::new(p)) } else { p });
    }
    Some((field?, GeoPred::And(parts)))
}

/// A boolean query's scorer; `boost` (from enclosing `BoostQuery`s) is passed
/// down to the scoring clauses, as Lucene's `BooleanWeight` does.
#[allow(clippy::too_many_arguments)]
fn bool_scorer(
    must: &[Query],
    should: &[Query],
    must_not: &[Query],
    filter: &[Query],
    minimum_should_match: &MinShouldMatch,
    leaf: &LeafReader<'_>,
    ctx: &SearchContext,
    boost: f32,
) -> Option<Box<dyn Scorer>> {
    // A filter-only combination of geo predicates on one single-valued
    // field (a distance ring, `>=` distances): one KD-tree pass instead of
    // a conjunction / exclusion of per-clause doc sets. Scores 0 like any
    // filter-only bool.
    // `must` clauses that score a constant +0 (`geo_shape` without a
    // boost, e.g. Rally's distance rings `must: [intersects R2, disjoint
    // R1]`) add nothing to the score either, so they join the same pass.
    let zero_musts = must.iter().all(|q| matches!(q, Query::ConstScore { score, .. } if *score == 0.0 && score.is_sign_positive()))
        && (must.is_empty() || minimum_should_match.resolve(0) == 0);
    if should.is_empty() && zero_musts && !(must.is_empty() && filter.is_empty()) && must.len() + filter.len() + must_not.len() > 1 {
        if let LeafReader::Segment(seg) = leaf {
            if let Some((field, pred)) = bool_geo_pred(must, must_not, filter) {
                if let Some(g) = seg.geo_index(field) {
                    if let Some(s) = crate::query::geo_scorer::geo_pred_scorer(g, pred, false, seg.num_docs, 0.0) {
                        return Some(s);
                    }
                } else {
                    return None;
                }
            }
        }
    }
    // A filter-only conjunction of cacheable clauses: one bitset, AND-ed a
    // word at a time (its score is 0, as the filters' scaled scores are).
    if must.is_empty() && should.is_empty() {
        if let Some(bits) = dense_filter_words(filter, must_not, leaf, ctx) {
            return bits.map(|b| Box::new(ConstScorer::new(b, 0.0)) as Box<dyn Scorer>);
        }
    }
    let resolved = minimum_should_match.resolve(should.len());
    // `must` clauses contribute scores, including a bool containing only
    // nested queries. Only explicit `filter` clauses may discard them.
    let mut m: Vec<Box<dyn Scorer>> = Vec::new();
    for q in must {
        m.push(q.scorer_b(leaf, ctx, boost)?);
    }
    // Numeric range filters become one doc bitset (Lucene's
    // points-based filter): the scoring clauses run their bulk
    // paths with the complement folded into the deleted docs,
    // and per-doc paths check one bit per candidate.
    let mut filter_bits: Option<Arc<crate::admission::Allocation<Vec<u64>>>> = None;
    // Its set bits, when it is one cached range bitset (as the segment
    // remembers them, not counted per request).
    let mut filter_count: Option<u32> = None;
    for q in filter {
        // Nested queries in a filter clause only need to match, not score:
        // mark them so their scorer skips per-child BM25 scoring.
        let q = with_nested_filter(q.clone());
        if let Some(b) = q.numeric_range_bits_cached(leaf, ctx) {
            match &mut filter_bits {
                Some(acc) => {
                    let memory=match crate::admission::shared().reserve((acc.len() as u64).saturating_mul(8).saturating_add(128),"combined numeric filter") {
                        Ok(memory)=>memory,Err(error)=>{ctx.record_memory_error(error.to_string());return None;}
                    };
                    let mut combined=acc.to_vec();and_words(&mut combined,&b);
                    *acc=Arc::new(crate::admission::Allocation::new(combined,memory));
                    filter_count = None;
                },
                None => {
                    filter_count = q.numeric_range_filter_key().map(|key| filter_count_u32(leaf, key, &b));
                    filter_bits = Some(b);
                }
            }
            continue;
        }
        // Filters participate in matching, not scoring.
        let s = q.scorer(leaf, ctx)?;
        m.push(Box::new(Scaled::new(s, 0.0)));
    }
    if let Some(b) = &filter_bits {
        if filter_count == Some(0) || (filter_count.is_none() && b.iter().all(|&w| w == 0)) {
            return None;
        }
    }
    // Wrap the scoring tree when there is one; otherwise the
    // bitset itself is a required (score 0) clause.
    let mut wrap_bits = if m.is_empty() {
        if let Some(b) = filter_bits.take() {
            let docs = match filter_count {
                Some(count) => BitSetDocSet::shared_counted(b, count),
                None => BitSetDocSet::shared(b),
            };
            m.push(Box::new(ConstScorer::new(docs, 0.0)));
        }
        None
    } else {
        filter_bits
    };
    let mut s: Vec<Box<dyn Scorer>> = Vec::with_capacity(should.len());
    let mut s_query: Vec<usize> = Vec::with_capacity(should.len());
    for (i, q) in should.iter().enumerate() {
        if let Some(x) = q.scorer_b(leaf, ctx, boost) {
            s.push(x);
            s_query.push(i);
        }
    }
    let msm = if m.is_empty() && resolved == 0 && !should.is_empty() { 1 } else { resolved };
    if msm > s.len() && !should.is_empty() {
        return None;
    }
    if !m.is_empty() && resolved > 0 {
        // must + minimum_should_match: fold the shoulds into a
        // required min-match union.
        let group = MinShouldMatchUnion::new(std::mem::take(&mut s), msm);
        m.push(Box::new(group));
    }
    let mut mn: Vec<Box<dyn DocSet>> = must_not
        .iter()
        .filter_map(|q| q.scorer(leaf, ctx))
        .map(|s| Box::new(AsDocSet(s)) as Box<dyn DocSet>)
        .collect();
    if m.is_empty() && s.is_empty() {
        return None;
    }
    // A few excluded docs (query rules' `exclude` ids, an `ids` must_not):
    // cleared from a pass bitset instead of an Exclusion, which has only a
    // per-doc path, so the scoring tree keeps its bulk (MaxScore) paths.
    let excluded_hint: u64 = mn.iter().map(|d| d.size_hint() as u64).sum();
    if !mn.is_empty() && excluded_hint <= SMALL_EXCLUSION_DOCS {
        let n = leaf.num_docs() as usize;
        let words = n.div_ceil(64);
        if let Ok(memory) = crate::admission::shared().reserve((words as u64).saturating_mul(8).saturating_add(128), "excluded docs pass bitset") {
            let mut bits: Vec<u64> = match &wrap_bits {
                Some(b) => b.to_vec(),
                None => {
                    let mut v = vec![u64::MAX; words];
                    if n % 64 != 0 {
                        v[words - 1] = (1u64 << (n % 64)) - 1;
                    }
                    v
                }
            };
            bits.resize(words, 0);
            for mut d in mn.drain(..) {
                loop {
                    let x = d.advance();
                    if x == TERMINATED {
                        break;
                    }
                    if (x as usize) < n {
                        bits[x as usize / 64] &= !(1u64 << (x % 64));
                    }
                }
            }
            wrap_bits = Some(Arc::new(crate::admission::Allocation::new(bits, memory)));
            filter_count = None;
        }
    }
    // Pinned docs (query rules' `pinned` ids, a `pinned` query) OR-ed with
    // one organic clause: the organic clause keeps its bulk top-k paths and
    // the few pinned docs are scored on the side (see [`PinnedUnion`]).
    if m.is_empty() && msm <= 1 && s.len() >= 2 {
        if let Some(big) = pinned_shape(&s, &s_query, should) {
            if let Some(probe) = should[s_query[big]].scorer_b(leaf, ctx, boost) {
                let pinned: Box<dyn Scorer> = Box::new(crate::query::boolean::PinnedUnion::new(std::mem::take(&mut s), big, probe, leaf.num_docs(), None));
                s.push(pinned);
            }
        }
    }
    let base = BooleanScorer::build(m, s, mn, msm);
    Some(match (wrap_bits, filter_count) {
        (Some(bits), Some(count)) => Box::new(FilteredScorer::new_counted(base, bits, count)),
        (Some(bits), None) => Box::new(FilteredScorer::new(base, bits)),
        (None, _) => base,
    })

}

/// The organic clause of `scorers` (built from `queries[at[i]]`) when every
/// other clause is a constant-score clause matching few docs (pinned ids:
/// a `pinned` query's pins, query rules' boosted `ids`); see
/// [`crate::query::boolean::PinnedUnion`].
fn pinned_shape(scorers: &[Box<dyn Scorer>], at: &[usize], queries: &[Query]) -> Option<usize> {
    let big = crate::query::boolean::PinnedUnion::organic_clause(scorers)?;
    at.iter().enumerate().all(|(i, &q)| i == big || pin_like(&queries[q])).then_some(big)
}

/// A pinned-docs clause: constant scores on a few listed docs.
fn pin_like(q: &Query) -> bool {
    match q {
        Query::ConstScore { .. } => true,
        Query::Boost { query, .. } => pin_like(query),
        Query::Custom(c) => c.name() == "pinned",
        _ => false,
    }
}

/// A dis_max scorer of `queries` (clause order kept); a pinned one
/// ([`pinned_shape`]) bulk-scores its organic clause apart from the pins.
fn dis_max_scorer(queries: &[Query], tie_breaker: f32, leaf: &LeafReader<'_>, ctx: &SearchContext, boost: Option<f32>) -> Option<Box<dyn Scorer>> {
    let build = |q: &Query| match boost {
        Some(b) => q.scorer_b(leaf, ctx, b),
        None => q.scorer(leaf, ctx),
    };
    let mut children: Vec<Box<dyn Scorer>> = Vec::with_capacity(queries.len());
    let mut at: Vec<usize> = Vec::with_capacity(queries.len());
    for (i, q) in queries.iter().enumerate() {
        if let Some(s) = build(q) {
            children.push(s);
            at.push(i);
        }
    }
    if children.is_empty() {
        return None;
    }
    // A `pinned` query on a leaf without any pinned doc: its organic clause
    // alone, which keeps its bulk paths (a dis_max of one nonnegative
    // score is that score: `max + tie * (s - max)` with `max = s`).
    if children.len() == 1 && queries.len() >= 2 && !pin_like(&queries[at[0]]) && queries.iter().any(pin_like) {
        return children.pop();
    }
    if children.len() >= 2 {
        if let Some(big) = pinned_shape(&children, &at, queries) {
            if let Some(probe) = build(&queries[at[big]]) {
                return Some(Box::new(crate::query::boolean::PinnedUnion::new(children, big, probe, leaf.num_docs(), Some(tie_breaker))));
            }
        }
    }
    Some(Box::new(DisMaxScorer { inner: Union::new(children), children_scores: Vec::new(), tie_breaker }))
}

/// Most excluded docs ([`bool_scorer`]'s `must_not` size hints) folded
/// into a pass bitset rather than run as an [`crate::query::boolean::Exclusion`].
const SMALL_EXCLUSION_DOCS: u64 = 4096;

/// The docs of `postings` over a leaf of `n` docs as words (bit `d` =
/// doc `d`): membership only, no frequencies.
fn postings_words(postings: &PostingsReader, n: u32) -> Vec<u64> {
    let mut bits = vec![0u64; (n as usize).div_ceil(64)];
    let mut docs = [0u32; crate::codec::postings::BLOCK_DOCS];
    for block in 0..postings.num_full_blocks() {
        postings.decode_docs(block, &mut docs);
        for &doc in &docs { bits[doc as usize / 64] |= 1 << (doc % 64); }
    }
    let mut frequencies = [0u32; crate::codec::postings::BLOCK_DOCS];
    let count = postings.decode_tail(&mut docs, &mut frequencies);
    for &doc in &docs[..count] { bits[doc as usize / 64] |= 1 << (doc % 64); }
    bits
}

/// Segment filter-cache kind of a term's doc bitset ([`dense_filter_words`]);
/// numeric ranges are 0-3, `exists` 9, `geo_shape` 10.
const TERM_FILTER_KIND: u8 = 11;

/// Uses of a term filter on a segment (of its last 256 filter uses) before
/// its bitset is kept (Lucene's `UsageTrackingQueryCachingPolicy` asks 2
/// for costly filters, 5 for others; a dense term is the costly kind here).
const TERM_FILTER_CACHE_MIN_USES: usize = 2;

/// Leaves smaller than this keep the doc-at-a-time conjunction.
const DENSE_FILTER_MIN_DOCS: u32 = 4096;

#[cfg(test)]
pub(crate) static DENSE_FILTER_CONJUNCTIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// One clause of a [`dense_filter_words`] conjunction.
enum DenseClause {
    /// Matches every doc (`match_all`): adds nothing.
    All,
    /// Matches no doc: neither does the conjunction.
    None,
    /// Its docs, and how many there are.
    Words(Arc<crate::admission::Allocation<Vec<u64>>>, u64),
    /// A term's postings and doc count, read into words only once the
    /// conjunction is taken ([`term_words`]).
    Term(Term, PostingsReader, u64),
    /// A union's postings (a terms set, single terms in a should, a
    /// multi-term query's terms) and their summed doc counts, OR-ed into
    /// words only once the conjunction is taken: a sparse clause beside it
    /// sends the conjunction to the iterators without reading them.
    Union(Vec<PostingsReader>, u64),
}

/// The words of a cacheable filter leaf: an `exists` whose fast fields know
/// presence, a numeric range, a term on a sealed segment (a term of under
/// one doc per 64 is left to the iterators), `match_all`, and their
/// constant-score / boost wrappers. `None`: not one of these.
fn dense_clause(q: &Query, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<DenseClause> {
    let words = |key: crate::segment::FilterCacheKey, b: Arc<crate::admission::Allocation<Vec<u64>>>| {
        let count = filter_count(leaf, key, &b);
        if count == 0 { DenseClause::None } else { DenseClause::Words(b, count) }
    };
    if let (Some(key), Some(b)) = (q.numeric_range_filter_key(), q.numeric_range_bits_cached(leaf, ctx)) {
        return Some(words(key, b));
    }
    match q {
        Query::MatchAll => Some(DenseClause::All),
        Query::ConstScore { query, .. } | Query::Boost { query, .. } => dense_clause(query, leaf, ctx),
        Query::Exists { field } => Query::presence_bits_cached(*field, leaf, ctx).map(|b| words((EXISTS_FILTER_KIND, *field, 0, 0), b)),
        // A `terms` filter (keyword `TermSet`, or ip / other single-term
        // clauses in a `should`): the OR of its terms' docs, read from the
        // postings (Lucene's `TermInSetQuery` bitset); under one doc per 64
        // is left to the iterators.
        Query::TermSet { field, terms, .. } => {
            let n = leaf.num_docs();
            let mut texts: Vec<&[u8]> = terms.iter().map(|t| t.as_slice()).collect();
            if !texts.windows(2).all(|w| w[0] <= w[1]) {
                texts.sort_unstable();
            }
            let mut postings = Vec::new();
            leaf.for_each_postings_sorted(*field, &texts, |p| postings.push(p));
            union_clause(postings, n)
        }
        // A constant-score multi-term query (prefix / wildcard / regexp /
        // case-insensitive term ...): the OR of its expanded terms' docs,
        // the terms its scorer unions.
        Query::Prefix { .. } | Query::Wildcard { .. } | Query::Regexp { .. } | Query::Fuzzy { .. } | Query::TermRange { .. } => {
            let spec = q.multi_term_spec()?;
            if spec.rewrite != MultiTermRewrite::ConstantScore {
                return None;
            }
            let n = leaf.num_docs();
            let terms = enumerate_terms(leaf, spec.field, &*spec.matcher, spec.lower.as_deref(), spec.upper.as_deref(), spec.max_expansions);
            let postings = terms.iter().filter_map(|t| leaf.postings(t)).collect();
            union_clause(postings, n)
        }
        Query::Bool { must, should, must_not, filter, minimum_should_match }
            if must.is_empty() && must_not.is_empty() && filter.is_empty() && !should.is_empty() && minimum_should_match.resolve(should.len()) <= 1 && should.iter().all(|q| matches!(q, Query::Term(_))) =>
        {
            let n = leaf.num_docs();
            let postings = should.iter().filter_map(|q| match q { Query::Term(t) => ctx.postings(leaf, t), _ => None }).collect();
            union_clause(postings, n)
        }
        Query::Term(term) => {
            let Some(postings) = ctx.postings(leaf, term) else { return Some(DenseClause::None) };
            let df = u64::from(postings.doc_freq());
            if df == 0 {
                return Some(DenseClause::None);
            }
            if df.saturating_mul(64) < u64::from(leaf.num_docs()) {
                return None;
            }
            Some(DenseClause::Term(term.clone(), postings, df))
        }
        _ => None,
    }
}

/// OR the docs of `postings` into `words`.
fn or_postings_words(postings: &PostingsReader, words: &mut [u64]) {
    let mut docs = [0u32; crate::codec::postings::BLOCK_DOCS];
    let mut set = |d: u32| {
        if let Some(w) = words.get_mut(d as usize / 64) {
            *w |= 1 << (d % 64);
        }
    };
    for block in 0..postings.num_full_blocks() {
        postings.decode_docs(block, &mut docs);
        docs.iter().for_each(|&d| set(d));
    }
    let mut frequencies = [0u32; crate::codec::postings::BLOCK_DOCS];
    let count = postings.decode_tail(&mut docs, &mut frequencies);
    docs[..count].iter().for_each(|&d| set(d));
}

/// A union clause over `postings`: `None` (left to the iterators) when
/// sparse, under one doc per 64 of a leaf of `n` docs.
fn union_clause(postings: Vec<PostingsReader>, n: u32) -> Option<DenseClause> {
    let df: u64 = postings.iter().map(|p| u64::from(p.doc_freq())).sum();
    if df == 0 {
        return Some(DenseClause::None);
    }
    if df.saturating_mul(64) < u64::from(n) {
        return None;
    }
    Some(DenseClause::Union(postings, df))
}

/// A [`DenseClause::Union`]'s words. `None`: over the memory budget.
fn union_words(postings: &[PostingsReader], n: u32) -> Option<Arc<crate::admission::Allocation<Vec<u64>>>> {
    let words = (n as usize).div_ceil(64);
    let memory = crate::admission::shared().reserve((words as u64).saturating_mul(8).saturating_add(128), "terms filter words").ok()?;
    let mut acc = vec![0u64; words];
    for p in postings {
        or_postings_words(p, &mut acc);
    }
    Some(Arc::new(crate::admission::Allocation::new(acc, memory)))
}

/// A [`DenseClause::Term`]'s words from the segment's filter cache, built
/// and kept once the term was used [`TERM_FILTER_CACHE_MIN_USES`] times
/// there (terms of up to 15 bytes, which key the cache exactly). `None`:
/// not (yet) cached; the iterators then run, as before.
fn term_words(seg: &crate::segment::Segment, term: &Term, postings: &PostingsReader, n: u32) -> Option<Arc<crate::admission::Allocation<Vec<u64>>>> {
    if term.text.len() >= 16 {
        return None;
    }
    let mut packed = [0u8; 16];
    packed[..term.text.len()].copy_from_slice(&term.text);
    packed[15] = term.text.len() as u8;
    let key = (TERM_FILTER_KIND, term.field, u64::from_le_bytes(packed[..8].try_into().expect("8 bytes")), u64::from_le_bytes(packed[8..].try_into().expect("8 bytes")));
    if let Some(b) = seg.peek_filter_bits(key) {
        return Some(b);
    }
    if seg.note_filter_use(key) < TERM_FILTER_CACHE_MIN_USES {
        return None;
    }
    seg.cached_filter_bits(key, || postings_words(postings, n)).ok()
}

/// Words per window of a [`DenseAnd`].
const DENSE_WINDOW_WORDS: usize = 64;

/// The AND (and AND-NOT) of finished doc bitsets, computed a window of
/// [`DENSE_WINDOW_WORDS`] words at a time as iteration reaches it (Lucene's
/// `DenseConjunctionBulkScorer` windows): a seek past whole windows skips
/// their words, so a sorted search that stops early (block skipping, a
/// full page) pays only for the windows it visits.
struct DenseAnd {
    pos: Vec<Arc<crate::admission::Allocation<Vec<u64>>>>,
    neg: Vec<Arc<crate::admission::Allocation<Vec<u64>>>>,
    /// Words of the leaf (`num_docs` / 64, rounded up).
    words: usize,
    win: [u64; DENSE_WINDOW_WORDS],
    /// First word of the window, and its words.
    win_lo: usize,
    win_len: usize,
    /// Index in the window of the word being consumed, and its remaining bits.
    wi: usize,
    cur: u64,
    current: DocId,
    hint: u32,
}

impl DenseAnd {
    fn new(pos: Vec<Arc<crate::admission::Allocation<Vec<u64>>>>, neg: Vec<Arc<crate::admission::Allocation<Vec<u64>>>>, num_docs: u32, hint: u32) -> Self {
        let mut s = DenseAnd { pos, neg, words: (num_docs as usize).div_ceil(64), win: [0; DENSE_WINDOW_WORDS], win_lo: 0, win_len: 0, wi: 0, cur: 0, current: TERMINATED, hint };
        s.load(0);
        s.cur = s.win[0];
        s
    }

    /// Compute the window starting at word `lo` (< `words`).
    fn load(&mut self, lo: usize) {
        let hi = (lo + DENSE_WINDOW_WORDS).min(self.words);
        let len = hi.saturating_sub(lo);
        let win = &mut self.win[..len];
        win.fill(0);
        let first = &self.pos[0];
        if lo < first.len() {
            let end = hi.min(first.len());
            win[..end - lo].copy_from_slice(&first[lo..end]);
        }
        for b in &self.pos[1..] {
            // Words past a clause's end (a column over fewer docs) match none of it.
            let end = hi.min(b.len()).max(lo);
            for (w, x) in win.iter_mut().zip(&b[lo.min(end)..end]) {
                *w &= *x;
            }
            win[end - lo..].fill(0);
        }
        for b in &self.neg {
            let end = hi.min(b.len()).max(lo);
            for (w, x) in win.iter_mut().zip(&b[lo.min(end)..end]) {
                *w &= !*x;
            }
        }
        self.win_lo = lo;
        self.win_len = len;
        self.wi = 0;
    }

    /// The next set bit from the current word on.
    fn next_set(&mut self) -> DocId {
        loop {
            if self.cur != 0 {
                let bit = self.cur.trailing_zeros();
                self.cur &= self.cur - 1;
                self.current = ((self.win_lo + self.wi) as u32) * 64 + bit;
                return self.current;
            }
            self.wi += 1;
            if self.wi >= self.win_len {
                let next = self.win_lo + self.win_len;
                if next >= self.words || self.win_len == 0 {
                    self.win_len = 0;
                    self.wi = 0;
                    self.cur = 0;
                    self.current = TERMINATED;
                    return TERMINATED;
                }
                self.load(next);
            }
            self.cur = self.win[self.wi];
        }
    }
}

impl DocSet for DenseAnd {
    fn advance(&mut self) -> DocId {
        if self.win_len == 0 {
            self.current = TERMINATED;
            return TERMINATED;
        }
        self.next_set()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.current != TERMINATED && self.current >= target {
            return self.current;
        }
        if self.win_len == 0 {
            return TERMINATED;
        }
        let tw = target as usize / 64;
        if tw >= self.words {
            self.win_len = 0;
            self.cur = 0;
            self.current = TERMINATED;
            return TERMINATED;
        }
        let at = self.win_lo + self.wi;
        let mask = u64::MAX << (target % 64);
        if tw > at {
            if tw >= self.win_lo + self.win_len {
                self.load(tw);
            }
            self.wi = tw - self.win_lo;
            self.cur = self.win[self.wi] & mask;
        } else if tw == at {
            self.cur &= mask;
        }
        self.next_set()
    }

    fn doc(&self) -> DocId {
        self.current
    }

    fn size_hint(&self) -> u32 {
        self.hint
    }
}

/// A filter-only conjunction (no `must`, no `should`) on a sealed segment
/// as one bitset, AND-ed (and AND-NOT-ed) a word at a time: ES serves such
/// filters from its query cache as bitsets and Lucene's
/// `DenseConjunctionBulkScorer` intersects them by window. Leapfrogging
/// them one doc at a time costs a seek per candidate of the sparsest
/// clause instead: an EQL stage search (`event.category`, `exists` on the
/// timestamp and every join key) visited every event holding one of its
/// join keys (millions) for a few hundred matches, on each search.
///
/// Only when every clause (nested filter-only bools included) is a
/// [`dense_clause`], at least two of them hold docs, and the sparsest of
/// those leaves a candidate per 64 docs or more (a selective clause leads
/// the iterators for less). `None`: not taken; `Some(None)`: nothing
/// matches. The docs are exactly the iterator conjunction's.
/// The dense filter conjunctions of one leaf's query, kept while the
/// query builds several scorers over the leaf. Only the bools of that query
/// tree (`allowed`: their `filter` clause vectors' addresses, live and so
/// unique while the tree is borrowed) are cached, by that address.
pub(crate) struct DenseScope {
    allowed: std::collections::HashSet<usize>,
    parts: std::collections::HashMap<usize, Option<Option<DenseParts>>>,
}

/// Closes a [`SearchContext::dense_scope`], dropping its words.
pub(crate) struct DenseScopeGuard<'a> {
    ctx: &'a SearchContext,
    leaf: usize,
}

impl Drop for DenseScopeGuard<'_> {
    fn drop(&mut self) {
        self.ctx.dense_scopes.lock().remove(&self.leaf);
    }
}

impl SearchContext {
    /// Keep the dense filter conjunctions `query`'s bools build over leaf
    /// `leaf` until the guard drops, so each scorer of `query` after the
    /// first reuses them (no postings union or bitset read again). A sort's
    /// best-block walk builds a scorer per batch of blocks.
    pub(crate) fn dense_scope<'a>(&'a self, query: &Query, leaf: usize) -> DenseScopeGuard<'a> {
        fn walk(q: &Query, out: &mut std::collections::HashSet<usize>) {
            match q {
                Query::Bool { must, should, must_not, filter, .. } => {
                    if !filter.is_empty() {
                        out.insert(filter.as_ptr() as usize);
                    }
                    for c in must.iter().chain(should).chain(must_not).chain(filter) {
                        walk(c, out);
                    }
                }
                Query::ConstScore { query, .. } | Query::Boost { query, .. } => walk(query, out),
                Query::DisMax { queries, .. } => queries.iter().for_each(|c| walk(c, out)),
                _ => {}
            }
        }
        let mut allowed = std::collections::HashSet::new();
        walk(query, &mut allowed);
        self.dense_scopes.lock().insert(leaf, DenseScope { allowed, parts: std::collections::HashMap::new() });
        DenseScopeGuard { ctx: self, leaf }
    }
}

/// A dense filter conjunction's words and its first-word hint.
#[derive(Clone)]
struct DenseParts {
    pos: Vec<Arc<crate::admission::Allocation<Vec<u64>>>>,
    neg: Vec<Arc<crate::admission::Allocation<Vec<u64>>>>,
    num_docs: u32,
    hint: u32,
}

/// [`dense_filter_words_uncached`], taken from the leaf's open
/// [`SearchContext::dense_scope`] when the bool belongs to its query.
fn dense_filter_words(filter: &[Query], must_not: &[Query], leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Option<DenseAnd>> {
    let build = |p: Option<Option<DenseParts>>| p.map(|p| p.map(|p| DenseAnd::new(p.pos, p.neg, p.num_docs, p.hint)));
    let key = filter.as_ptr() as usize;
    let (lid, scoped) = (leaf.id(), !filter.is_empty());
    let cached = if scoped {
        match ctx.dense_scopes.lock().get(&lid) {
            Some(sc) if sc.allowed.contains(&key) => Some(sc.parts.get(&key).cloned()),
            _ => None,
        }
    } else {
        None
    };
    match cached {
        Some(Some(parts)) => build(parts),
        Some(None) => {
            let parts = dense_filter_words_uncached(filter, must_not, leaf, ctx);
            if let Some(sc) = ctx.dense_scopes.lock().get_mut(&lid) {
                sc.parts.insert(key, parts.clone());
            }
            build(parts)
        }
        None => build(dense_filter_words_uncached(filter, must_not, leaf, ctx)),
    }
}

fn dense_filter_words_uncached(filter: &[Query], must_not: &[Query], leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Option<DenseParts>> {
    let LeafReader::Segment(seg) = leaf else { return None };
    let n = leaf.num_docs();
    if n < DENSE_FILTER_MIN_DOCS || filter.len() < 2 {
        return None;
    }
    struct Gathered {
        pos: Vec<DenseClause>,
        neg: Vec<DenseClause>,
        empty: bool,
    }
    fn gather(filter: &[Query], must_not: &[Query], negated: bool, leaf: &LeafReader<'_>, ctx: &SearchContext, g: &mut Gathered) -> Option<()> {
        for q in filter {
            if let Query::Bool { must, should, must_not: inner_not, filter: inner, .. } = q {
                if should.is_empty() {
                    // A nested filter-only bool (scores are not used here):
                    // its `must` and `filter` clauses join this conjunction.
                    if negated {
                        return None;
                    }
                    gather(must, &[], false, leaf, ctx, g)?;
                    gather(inner, inner_not, false, leaf, ctx, g)?;
                    continue;
                }
                // A disjunction: one clause ([`dense_clause`]), if it can be.
            }
            match dense_clause(q, leaf, ctx)? {
                DenseClause::All => g.empty |= negated,
                DenseClause::None => g.empty |= !negated,
                c if negated => g.neg.push(c),
                c => g.pos.push(c),
            }
        }
        if !must_not.is_empty() {
            gather(must_not, &[], true, leaf, ctx, g)?;
        }
        Some(())
    }
    let mut g = Gathered { pos: Vec::new(), neg: Vec::new(), empty: false };
    gather(filter, must_not, false, leaf, ctx, &mut g)?;
    if g.empty {
        return Some(None);
    }
    if g.pos.len() < 2 {
        return None;
    }
    let count = |c: &DenseClause| match c {
        DenseClause::Words(_, n) | DenseClause::Term(_, _, n) | DenseClause::Union(_, n) => *n,
        _ => 0,
    };
    g.pos.sort_by_key(count);
    if count(&g.pos[0]).saturating_mul(64) < u64::from(n) {
        return None;
    }
    let materialize = |c: &DenseClause| match c {
        DenseClause::Words(b, _) => Some(b.clone()),
        DenseClause::Term(t, p, _) => term_words(seg, t, p, n),
        DenseClause::Union(postings, _) => union_words(postings, n),
        _ => None,
    };
    // Every term notes its use (a missing one does not stop the others').
    let pos: Vec<_> = g.pos.iter().map(materialize).collect();
    let neg: Vec<_> = g.neg.iter().map(materialize).collect();
    let pos: Vec<_> = pos.into_iter().collect::<Option<_>>()?;
    let neg: Vec<_> = neg.into_iter().collect::<Option<_>>()?;
    #[cfg(test)]
    DENSE_FILTER_CONJUNCTIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let hint = u32::try_from(count(&g.pos[0])).unwrap_or(u32::MAX);
    Some(Some(DenseParts { pos, neg, num_docs: n, hint }))
}

fn self_term(leaf: &LeafReader<'_>, ctx: &SearchContext, t: &Term) -> Option<Box<dyn Scorer>> {
    self_term_boosted(leaf, ctx, t, 1.0)
}

/// A term scorer with `boost` folded into the BM25 weight, as Lucene passes a
/// `BoostQuery`'s boost down to the term weight.
fn self_term_boosted(leaf: &LeafReader<'_>, ctx: &SearchContext, t: &Term, boost: f32) -> Option<Box<dyn Scorer>> {
    let postings = ctx.postings(leaf, t)?;
    leaf.term_scorer_postings(postings, t.field, ctx.sim(t.field), boost, ctx.stats(t), ctx.fstats(t.field))
}

#[cfg(test)]
mod filter_bit_tests {
    use super::*;
    use crate::{Document, Field, FieldType, Index, IndexSettings};

    #[test]
    fn filter_words_refuse_before_evaluation_and_follow_returned_owner() {
        let mut index=Index::new("filter-memory",vec![Field::new(0,"tag",FieldType::Keyword)],IndexSettings::default());
        for _ in 0..130 {let mut doc=Document::new();doc.add_keyword(0,"a");index.add_document(&doc).unwrap();}
        index.refresh().unwrap();let leaves=index.leaves();let leaf=&leaves[0];
        let query=Query::MatchAll;let ctx=SearchContext::default();
        let small=crate::admission::Budget::new(1);
        assert!(matches!(query.filter_bits_admitted(leaf,&ctx,&small),Err(crate::error::RusticError::Memory(_))));
        assert_eq!(small.used(),0);
        let budget=crate::admission::Budget::new(4096);
        let bits=query.filter_bits_admitted(leaf,&ctx,&budget).unwrap();
        assert_eq!(bits.iter().map(|w|w.count_ones()).sum::<u32>(),leaf.num_docs());
        drop(ctx);assert!(budget.used()>0);drop(bits);assert_eq!(budget.used(),0);
    }

    /// A numeric `terms` query (constant score over a disjunction of point
    /// clauses) answers from one OR-ed bitset: the same docs as one bitset
    /// per value, on segments and memtables, single- and multi-valued.
    #[test]
    fn numeric_terms_union_matches_per_value_bitsets() {
        let mut index = Index::new("numeric-terms", vec![Field::new(0, "n", FieldType::I64)], IndexSettings::default());
        for i in 0..500i64 {
            let mut document = Document::new();
            if i % 7 != 0 {
                document.add_i64(0, i % 50);
            }
            if i % 11 == 0 {
                document.add_i64(0, 1000 + i % 3);
            }
            index.add_document(&document).unwrap();
            if i == 300 {
                index.refresh().unwrap();
            }
        }
        let leaves = index.leaves();
        let ctx = SearchContext::default();
        for values in [vec![3i64, 5], vec![0, 49, 1001], (0..50).collect::<Vec<_>>(), vec![999, -4], vec![7, 7]] {
            let should: Vec<Query> = values.iter().map(|&v| Query::RangeI64 { field: 0, lower: Bound::Included(v), upper: Bound::Included(v) }).collect();
            let inner = Query::Bool { must: vec![], should: should.clone(), must_not: vec![], filter: vec![], minimum_should_match: MinShouldMatch::Count(1) };
            let union = Query::ConstScore { query: Box::new(inner.clone()), score: 1.0, point_set: true };
            for leaf in &leaves {
                assert!(inner.numeric_union_bits(leaf).is_some());
                let n = leaf.num_docs();
                let mut expected = vec![0u64; (n as usize).div_ceil(64)];
                for q in &should {
                    if let Some(mut s) = q.scorer(leaf, &ctx) {
                        for (e, w) in expected.iter_mut().zip(doc_bits(&mut *s, n)) {
                            *e |= w;
                        }
                    }
                }
                let got = union.scorer(leaf, &ctx).map(|mut s| doc_bits(&mut *s, n)).unwrap_or_else(|| vec![0; (n as usize).div_ceil(64)]);
                assert_eq!(got, expected, "{values:?}");
            }
        }
    }

    /// Driving iteration from a small bitset (leapfrog) visits exactly the
    /// docs, in the same order, as stepping through the inner scorer and
    /// checking each candidate's bit, for mixed `advance` / `seek` calls.
    #[test]
    fn filtered_scorer_leapfrog_matches_per_candidate_checks() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |n: u64| { state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (state >> 33) % n.max(1) };
        for case in 0..400u64 {
            let num_docs = 1 + next(3000) as u32;
            let inner_every = 1 + next(4);
            let inner: Vec<DocId> = (0..num_docs).filter(|d| (*d as u64) % inner_every == 0 || next(5) == 0).collect();
            let mut words = vec![0u64; (num_docs as usize).div_ceil(64)];
            for _ in 0..next(40) {
                let d = next(num_docs as u64) as usize;
                words[d / 64] |= 1 << (d % 64);
            }
            let expected: Vec<DocId> = inner.iter().copied().filter(|d| (words[*d as usize / 64] >> (d % 64)) & 1 == 1).collect();
            let seeks: Vec<u64> = (0..64).map(|_| next(4)).collect();
            let mut runs = Vec::new();
            for drive in [false, true] {
                let memory = crate::admission::shared().reserve(8, "test filter").unwrap();
                let bits = Arc::new(crate::admission::Allocation::new(words.clone(), memory));
                let mut f = FilteredScorer::new(Box::new(ConstScorer::new(VecDocSet::new(inner.clone()), 1.0)), bits);
                f.drive_bits = drive;
                let mut got = Vec::new();
                let mut d = f.advance();
                let mut i = 0;
                while d != TERMINATED {
                    got.push(d);
                    assert_eq!(f.doc(), d);
                    // Every fourth step seeks a few docs ahead.
                    d = if seeks[i % seeks.len()] == 0 { f.seek(d + 1 + (i as u32 % 7)) } else { f.advance() };
                    i += 1;
                }
                assert_eq!(f.doc(), TERMINATED, "case {case} drive {drive}");
                assert_eq!(f.advance(), TERMINATED);
                runs.push(got);
            }
            assert_eq!(runs[0], runs[1], "case {case}");
            if seeks.iter().all(|&s| s != 0) {
                assert_eq!(runs[0], expected, "case {case}");
            }
            for d in &runs[0] {
                assert!(expected.binary_search(d).is_ok(), "case {case}: {d} is not a match");
            }
        }
        // A bitset this much smaller than its scorer drives iteration.
        let memory = crate::admission::shared().reserve(8, "test filter").unwrap();
        let mut words = vec![0u64; 1000];
        words[500] = 1;
        let f = FilteredScorer::new(Box::new(ConstScorer::new(AllDocSet::new(64_000), 1.0)), Arc::new(crate::admission::Allocation::new(words, memory)));
        assert!(f.drive_bits);
    }

    /// An aggregation's bulk collection through a boxed, scaled filtered
    /// scorer over all docs (a bool of a match-all clause and a range):
    /// the same docs and score as advancing it, deleted docs left out.
    #[test]
    fn filtered_all_docs_collect_all_matches_iteration() {
        use crate::query::segment_scorer::Scaled;
        for n in [1u32, 63, 64, 65, 1000, 4099] {
            let words: Vec<u64> = (0..n.div_ceil(64)).map(|w| 0x9E37_79B9_7F4A_7C15u64.rotate_left(w * 7) ^ (w as u64)).collect();
            let deleted: Vec<u64> = (0..n.div_ceil(64)).map(|w| 0x0F0F_00FF_1234_8001u64.rotate_left(w * 3)).collect();
            let make = || -> Box<dyn Scorer> {
                let memory = crate::admission::shared().reserve(8 * words.len() as u64 + 8, "test filter").unwrap();
                let filtered: Box<dyn Scorer> = Box::new(FilteredScorer::new(Box::new(ConstScorer::new(AllDocSet::new(n), 1.0)), Arc::new(crate::admission::Allocation::new(words.clone(), memory))));
                Box::new(Scaled::new(filtered, 2.5))
            };
            for del in [None, Some(&deleted[..])] {
                let mut want = Vec::new();
                let mut s = make();
                let mut score = None;
                loop {
                    let d = s.advance();
                    if d == TERMINATED { break; }
                    if del.is_some_and(|m| m[d as usize / 64] >> (d % 64) & 1 == 1) { continue; }
                    score = Some(s.score());
                    want.push(d);
                }
                let mut got = Vec::new();
                let collected = make().collect_all(del, &mut got);
                assert_eq!(got, want, "n={n} deleted={}", del.is_some());
                assert_eq!(collected, Some(2.5));
                if let Some(score) = score { assert_eq!(score, 2.5); }
            }
        }
    }

    #[test]
    fn membership_only_term_and_disjunction_match_scorers() {
        let mut index = Index::new("filters", vec![Field::new(0, "tag", FieldType::Keyword)], IndexSettings::default());
        for i in 0..410 {
            let mut document = Document::new();
            if i % 2 == 0 { document.add_keyword(0, "a"); }
            if i % 3 == 0 { document.add_keyword(0, "b"); }
            index.add_document(&document).unwrap();
        }
        index.refresh().unwrap();
        for tag in ["a", "b", "missing"] {
            let mut document = Document::new();
            document.add_keyword(0, tag);
            index.add_document(&document).unwrap();
        }
        let a = Query::term(0, "a");
        let b = Query::term(0, "b");
        let queries = [a.clone(), Query::term(0, "missing"),
            Query::bool_query().should(a.clone()).should(b.clone()).build(),
            Query::bool_query().should(a.clone()).should(b.clone()).must_not(Query::term(0, "missing")).build(),
            Query::bool_query().should(a.clone()).should(b.clone()).minimum_should_match(2usize).build(),
            Query::bool_query().must(a.clone()).should(b.clone()).build(),
            Query::bool_query().filter(a.clone()).must_not(b.clone()).build(),
            Query::ConstScore { query: Box::new(a), score: 0.0, point_set: false }];
        let leaves = index.leaves();
        for query in queries {
            let ctx = SearchContext::for_query(&query, &leaves, Bm25Params::default());
            for leaf in &leaves {
                let expected = query.scorer(leaf, &ctx).map(|mut scorer| doc_bits(&mut *scorer, leaf.num_docs()))
                    .unwrap_or_else(|| vec![0; (leaf.num_docs() as usize).div_ceil(64)]);
                assert_eq!(*query.filter_bits(leaf, &ctx).unwrap(), expected, "{query:?}");
            }
        }
    }
}


#[cfg(test)]
mod filter_count_tests {
    use super::*;
    use crate::{Document, Field, FieldType, Index, IndexSettings};

    /// Scorers over a segment's cached range / `exists` bitsets take the
    /// set-bit count the segment remembers for the bitset's key instead of
    /// counting the bitset per request: the same size and the same docs on
    /// every build, on sealed segments and on a memtable, for a scored range
    /// in `must`, a range `filter` beside a scored clause, a filter-only
    /// range and a range matching nothing.
    #[test]
    fn cached_filter_counts_match_the_bitsets() {
        let fields = vec![Field::new(0, "w", FieldType::Keyword), Field::new(1, "ts", FieldType::I64)];
        let mut index = Index::new("filter-counts", fields, IndexSettings::default());
        for i in 0..9_000i64 {
            let mut d = Document::new();
            d.add_keyword(0, if i % 3 == 0 { "a" } else { "b" });
            if i % 5 != 0 {
                d.add_i64(1, (i * 7919) % 10_000);
            }
            index.add_document(&d).unwrap();
            if i == 4_999 || i == 8_499 {
                index.refresh().unwrap();
            }
        }
        let leaves = index.leaves();
        assert!(leaves.iter().any(|l| matches!(l, LeafReader::Segment(_))) && leaves.iter().any(|l| matches!(l, LeafReader::Memtable(..))));
        let range = |lo: i64, hi: i64| Query::RangeI64 { field: 1, lower: Bound::Included(lo), upper: Bound::Excluded(hi) };
        let brute = |leaf: &LeafReader<'_>, q: &Query, ctx: &SearchContext| -> Vec<DocId> {
            let Some(mut s) = q.scorer(leaf, ctx) else { return Vec::new() };
            let mut out = Vec::new();
            loop {
                let d = s.advance();
                if d == TERMINATED { break; }
                out.push(d);
            }
            out
        };
        for (lo, hi) in [(1_000, 4_000), (0, 10_000), (20_000, 30_000), (5, 6)] {
            let r = range(lo, hi);
            let cases = [
                r.clone(),
                Query::bool_query().must(r.clone()).must(Query::term(0, "a")).build(),
                Query::bool_query().filter(r.clone()).must(Query::term(0, "a")).build(),
                Query::bool_query().filter(r.clone()).build(),
                Query::Exists { field: 1 },
            ];
            for q in &cases {
                let ctx = SearchContext::for_query(q, &leaves, Bm25Params::default());
                for leaf in &leaves {
                    let first = brute(leaf, q, &ctx);
                    // A second build reads the remembered count.
                    let again = brute(leaf, q, &ctx);
                    assert_eq!(first, again, "{q:?}");
                    let want: Vec<DocId> = (0..leaf.num_docs()).filter(|&d| {
                        let v = leaf.i64_value(1, d);
                        let in_range = v.is_some_and(|v| v >= lo && v < hi);
                        let is_a = leaf.keyword_value(0, d).as_deref() == Some("a");
                        match q {
                            Query::RangeI64 { .. } => in_range,
                            Query::Exists { .. } => v.is_some(),
                            Query::Bool { must, .. } if !must.is_empty() => in_range && is_a,
                            _ => in_range,
                        }
                    }).collect();
                    assert_eq!(first, want, "{q:?} over {lo}..{hi}");
                }
            }
            for leaf in &leaves {
                let ctx = SearchContext::for_query(&r, &leaves, Bm25Params::default());
                let s = r.scorer(leaf, &ctx);
                let n = brute(leaf, &r, &ctx).len() as u32;
                if let Some(s) = s {
                    assert!(s.size_hint() >= n, "a range's size bounds its docs");
                }
                if let LeafReader::Segment(seg) = leaf {
                    if seg.peek_filter_bits(r.numeric_range_filter_key().unwrap()).is_some() {
                        // Remembered: a lookup with no words returns the count.
                        assert_eq!(seg.filter_bits_count(r.numeric_range_filter_key().unwrap(), &[]), u64::from(n));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod dense_filter_tests {
    use super::*;
    use crate::{Document, Field, FieldType, Index, IndexSettings};

    /// Filter-only conjunctions of cacheable clauses (terms, `exists`,
    /// numeric ranges, nested filter-only bools, `must_not`) taken a word
    /// at a time match exactly the docs of the iterator conjunction (the
    /// same bool with a `must: match_all`, which never takes it), score 0,
    /// on sealed segments (once their term words are cached) and memtables;
    /// iteration is checked with `advance` and with `seek`s across windows.
    #[test]
    fn dense_filter_conjunction_matches_iterator_conjunction() {
        let fields = vec![Field::new(0, "cat", FieldType::Keyword), Field::new(1, "ip", FieldType::Keyword), Field::new(2, "n", FieldType::I64), Field::new(3, "ts", FieldType::I64)];
        let mut index = Index::new("dense-filters", fields, IndexSettings::default());
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = |m: u64| { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state % m };
        for i in 0..12_700u32 {
            let mut d = Document::new();
            d.add_keyword(0, ["process", "network", "file"][next(3) as usize]);
            // ip: dense on about a third of the docs, rare value "x".
            if next(3) == 0 {
                d.add_keyword(1, if next(500) == 0 { "x" } else { ["a", "b", "c"][next(3) as usize] });
            }
            if next(10) != 0 {
                d.add_i64(2, next(100) as i64);
            }
            // The first segment's ts column ends early (a shorter column).
            if !(i < 6000 && i % 6000 >= 5800) {
                d.add_i64(3, i as i64);
            }
            index.add_document(&d).unwrap();
            if i == 5999 || i == 11_999 {
                index.refresh().unwrap();
            }
        }
        let t = |f: u16, v: &str| Query::term(f, v);
        let ex = |f: u16| Query::Exists { field: f };
        let range = |lo: i64, hi: i64| Query::RangeI64 { field: 2, lower: Bound::Included(lo), upper: Bound::Included(hi) };
        let filter_only = |filter: Vec<Query>, must_not: Vec<Query>| Query::Bool { must: vec![], should: vec![], must_not, filter, minimum_should_match: MinShouldMatch::Count(0) };
        let not_ip_a = Query::Bool { must: vec![Query::MatchAll], should: vec![], must_not: vec![t(1, "a")], filter: vec![], minimum_should_match: MinShouldMatch::Count(0) };
        // (query, whether the sealed segments take the word conjunction)
        let cases = vec![
            (filter_only(vec![ex(3), Query::MatchAll, t(0, "process"), ex(1), ex(2)], vec![]), true),
            (filter_only(vec![t(0, "process"), ex(1)], vec![t(1, "b")]), true),
            (filter_only(vec![t(0, "network"), not_ip_a.clone(), ex(1)], vec![]), true),
            (filter_only(vec![t(0, "file"), range(10, 60), Query::ConstScore { query: Box::new(ex(3)), score: 1.0, point_set: false }], vec![ex(1)]), true),
            (filter_only(vec![t(0, "process"), filter_only(vec![ex(1), ex(2)], vec![])], vec![]), true),
            (filter_only(vec![t(0, "nothing"), ex(1)], vec![]), true),
            (filter_only(vec![t(0, "process"), ex(1)], vec![Query::MatchAll]), true),
            // `terms` filters: a keyword term set, and single terms in a should.
            (filter_only(vec![t(0, "process"), Query::TermSet { field: 1, terms: vec![b"a".to_vec(), b"c".to_vec(), b"zz".to_vec()], boost: 1.0 }, ex(2)], vec![]), true),
            (filter_only(vec![t(0, "network"), Query::ConstScore { query: Box::new(Query::bool_query().should(t(1, "b")).should(t(1, "c")).build()), score: 1.0, point_set: true }], vec![t(1, "c")]), true),
            (filter_only(vec![t(0, "file"), Query::bool_query().should(t(1, "a")).should(t(1, "x")).build()], vec![]), true),
            // Multi-term queries (constant score): a prefix, a negated
            // case-insensitive term (a regexp) in a nested bool.
            (filter_only(vec![t(0, "process"), Query::Prefix { field: 1, prefix: b"".to_vec(), case_insensitive: false, max_expansions: usize::MAX, rewrite: MultiTermRewrite::ConstantScore }], vec![]), true),
            (filter_only(vec![t(0, "network"), Query::Bool { must: vec![Query::MatchAll], should: vec![], must_not: vec![Query::Regexp { field: 1, pattern: "A".into(), case_insensitive: true, max_expansions: usize::MAX, rewrite: MultiTermRewrite::ConstantScore }], filter: vec![], minimum_should_match: MinShouldMatch::Count(0) }, ex(1)], vec![]), true),
            // A selective clause leads the iterators instead.
            (filter_only(vec![t(1, "x"), t(0, "process"), ex(2)], vec![]), false),
            (filter_only(vec![t(0, "process"), range(7, 7), ex(1)], vec![]), false),
        ];
        let leaves = index.leaves();
        assert!(leaves.iter().any(|l| matches!(l, LeafReader::Segment(_))));
        for (case, (query, dense)) in cases.into_iter().enumerate() {
            let Query::Bool { must_not, filter, .. } = &query else { unreachable!() };
            let reference = Query::Bool { must: vec![Query::MatchAll], should: vec![], must_not: must_not.clone(), filter: filter.clone(), minimum_should_match: MinShouldMatch::Count(0) };
            // The first use of a term runs the iterators; its words are
            // cached (and the conjunction taken) from the second.
            for round in 0..3 {
                let ctx = SearchContext::for_query(&query, &leaves, Bm25Params::default());
                for leaf in &leaves {
                    let n = leaf.num_docs();
                    let expected = reference.scorer(leaf, &ctx).map(|mut s| doc_bits(&mut *s, n)).unwrap_or_else(|| vec![0; (n as usize).div_ceil(64)]);
                    let before = DENSE_FILTER_CONJUNCTIONS.load(std::sync::atomic::Ordering::Relaxed);
                    let got = match query.scorer(leaf, &ctx) {
                        Some(mut s) => {
                            let mut bits = vec![0u64; (n as usize).div_ceil(64)];
                            let mut d = s.advance();
                            while d != TERMINATED {
                                assert_eq!(s.score(), 0.0, "case {case}");
                                bits[d as usize / 64] |= 1 << (d % 64);
                                d = s.advance();
                            }
                            bits
                        }
                        None => vec![0; (n as usize).div_ceil(64)],
                    };
                    assert_eq!(got, expected, "case {case} round {round}");
                    // Seeks within a word, across words and across windows.
                    let docs: Vec<DocId> = (0..n).filter(|&d| (expected[d as usize / 64] >> (d % 64)) & 1 == 1).collect();
                    if let Some(mut s) = query.scorer(leaf, &ctx) {
                        let steps = [1u32, 37, 900, 4500, 64];
                        let (mut target, mut i) = (3u32, 0usize);
                        loop {
                            let d = s.seek(target);
                            let want = docs.iter().copied().find(|&x| x >= target).unwrap_or(TERMINATED);
                            assert_eq!(d, want, "case {case} round {round} seek {target}");
                            if d == TERMINATED {
                                break;
                            }
                            if i % 3 == 2 {
                                let a = s.advance();
                                assert_eq!(a, docs.iter().copied().find(|&x| x > d).unwrap_or(TERMINATED), "case {case} advance after {d}");
                                if a == TERMINATED {
                                    break;
                                }
                                target = a + steps[i % steps.len()];
                            } else {
                                target = d + steps[i % steps.len()];
                            }
                            i += 1;
                        }
                    }
                    if dense && round > 0 && matches!(leaf, LeafReader::Segment(_)) {
                        assert!(DENSE_FILTER_CONJUNCTIONS.load(std::sync::atomic::Ordering::Relaxed) > before || got.iter().all(|&w| w == 0), "case {case}: word conjunction not taken");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod interval_rewrite_tests {
    use super::*;
    use crate::{segment::SegmentWriter, types::{Document, Field, FieldType}};

    #[test]
    fn bounded_interval_rewrite_preserves_leaf_statistics_and_scores() {
        let schema = vec![Field::new(0,"body",FieldType::Text)];
        let mut first = SegmentWriter::new(1);
        let mut second = SegmentWriter::new(2);
        for (writer,texts) in [(&mut first,vec!["aa aa fixed","ab fixed"]),(&mut second,vec!["aa fixed","aa aa aa fixed","ab"])] {
            for text in texts {
                let mut doc=Document::new();
                doc.add_text(0,text);
                writer.add_document(&doc,&schema);
            }
        }
        let leaves=vec![LeafReader::Memtable(&first,None),LeafReader::Memtable(&second,None)];
        let rule=IntervalsRule::AnyOf(vec![IntervalsRule::Prefix("a".into()),
            IntervalsRule::Match { terms:vec!["fixed".into()],max_gaps:None,ordered:true }]);
        let original=rule.to_span(0);
        let query=Query::Intervals { field:0,rule };
        let context=SearchContext::with_similarity_deadline_limit(&query,&leaves,
            Arc::new(Bm25::default()),HashMap::new(),None,10);
        context.execution_result().unwrap();
        for leaf in &leaves {
            let mut before=span_scorer(&original,leaf,&context).unwrap();
            let mut after=query.scorer(leaf,&context).unwrap();
            loop {
                let doc=before.advance();
                assert_eq!(doc,after.advance());
                if doc==TERMINATED { break; }
                assert_eq!(before.score(),after.score());
            }
        }
    }
}

#[cfg(test)]
mod memory_failure_handoff_tests {
    use super::SearchContext;
    #[test]
    fn scorer_memory_refusal_stays_typed_after_context_handoff() {
        let child=SearchContext::default();
        child.record_memory_error("quantized cache refused");
        let parent=SearchContext::default();
        parent.inherit_errors(&child);
        assert!(matches!(parent.execution_result(),Err(crate::error::RusticError::Memory(reason)) if reason=="quantized cache refused"));
    }
}

#[cfg(test)]
mod filtered_scratch_admission_tests {
    use super::*;
    #[test]
    fn optional_bulk_refusal_keeps_filtered_iteration_available() {
        let budget=crate::admission::Budget::new(4096);
        let memory=budget.reserve(136,"filter").unwrap();
        let bits=Arc::new(crate::admission::Allocation::new(vec![0b1010],memory));
        let inner=Box::new(ConstScorer::new(AllDocSet::new(4),1.0));
        let mut scorer=FilteredScorer::new(inner,bits);
        let refused=crate::admission::Budget::new(1);
        assert!(scorer.excluded_admitted(None,&refused).is_none());
        assert_eq!(refused.used(),0);
        assert_eq!(scorer.advance(),1);assert_eq!(scorer.advance(),3);
        assert_eq!(scorer.advance(),TERMINATED);
        let complement=scorer.excluded_admitted(Some(&[2]),&budget).unwrap();
        assert_eq!(complement[0],!0b1010u64|2);drop(complement);drop(scorer);
        assert_eq!(budget.used(),0);
    }
}

#[cfg(test)]
mod geo_pure_negative_tests {
    use super::*;
    use crate::codec::directory::RamDirectory;
    use crate::geo::GeoShape;
    use crate::segment::SegmentWriter;
    use crate::types::{Document, Field, FieldType};

    /// ES's pure-negative ring (`must: [circle R2, bool { must: [match_all],
    /// must_not: [circle R1] }]`, Rally `distanceFilterCount200x300-alt`)
    /// is one point predicate: counted from the KD tree, equal to its
    /// definition and to the scorer; `match_all` alone or a bare negation
    /// is not.
    #[test]
    fn pure_negative_ring_is_a_point_predicate() {
        let schema = vec![Field::new(0, "g", FieldType::GeoPoint)];
        let mut w = SegmentWriter::new(0);
        let mut pts = Vec::new();
        for i in 0..20_000u32 {
            let mut d = Document::new();
            if i % 37 != 0 {
                let (lat, lon) = (51.0 + (i % 101) as f64 * 0.08, 1.0 + (i % 103) as f64 * 0.12);
                d.add_geo_point(0, lat, lon);
                pts.push(Some(crate::geo::quantize(lat, lon)));
            } else {
                pts.push(None);
            }
            w.add_document(&d, &schema);
        }
        let seg = w.write_to(&RamDirectory::new()).unwrap();
        let leaf = LeafReader::Segment(&seg);
        let dist = |m: f64| Query::ConstScore { query: Box::new(Query::GeoDistance { field: 0, lat: 55.0, lon: 7.0, meters: m }), score: 0.0, point_set: false };
        let neg = Query::Bool { must: vec![Query::MatchAll], should: vec![], minimum_should_match: MinShouldMatch::Count(0), filter: vec![], must_not: vec![dist(200_000.0)] };
        let ring = Query::Bool { must: vec![dist(300_000.0), neg.clone()], should: vec![], minimum_should_match: MinShouldMatch::Count(0), filter: vec![], must_not: vec![] };
        assert!(ring.geo_pred().is_some());
        assert!(neg.geo_pred().is_none());
        assert!(Query::MatchAll.geo_pred().is_none());
        let within = |p: (f64, f64), m: f64| GeoShape::Distance { lat: 55.0, lon: 7.0, meters: m }.contains(p.0, p.1);
        let want = pts.iter().filter(|p| p.is_some_and(|p| within(p, 300_000.0) && !within(p, 200_000.0))).count() as u64;
        assert!(want > 0);
        assert_eq!(ring.exact_count(&leaf), Some(want));
        let ctx = SearchContext::default();
        let mut s = ring.scorer(&leaf, &ctx).unwrap();
        let mut n = 0;
        while s.advance() != TERMINATED {
            n += 1;
        }
        assert_eq!(n, want);
    }
}
