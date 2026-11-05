//! Multi-term query expansion and rewrite methods (Lucene's
//! `MultiTermQuery.RewriteMethod`), shared by prefix / wildcard / regexp /
//! fuzzy / term-range queries.
//!
//! Constant-score rewrites expand per leaf and never need statistics.
//! Scoring rewrites expand once over every leaf when the
//! [`SearchContext`](crate::query::query::SearchContext) is built, so every
//! leaf scores the same term set with the same index-wide statistics.

use crate::automaton::TermMatcher;
use crate::query::boolean::union_of;
use crate::query::docset::VecDocSet;
use crate::types::DocId;
use crate::query::reader::{term_stats, LeafReader, TermStats};
use crate::query::scorer::{ConstScorer, Scorer};
use crate::query::segment_scorer::Scaled;
use crate::query::similarity::Similarity;
use crate::query::reader::FieldStats;
use crate::types::Term;
use std::collections::BTreeMap;

/// How a multi-term query turns its expanded terms into a scorer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MultiTermRewrite {
    /// Union of the matching docs, every doc scoring the query boost
    /// (Lucene `CONSTANT_SCORE_REWRITE`, the ES default).
    #[default]
    ConstantScore,
    /// Sum of real term scores over every expanded term (Lucene
    /// `SCORING_BOOLEAN_REWRITE`), bounded by `max_expansions`.
    ScoringBoolean,
    /// A boolean of the terms, then constant-scored as a whole (Lucene
    /// `CONSTANT_SCORE_BOOLEAN_REWRITE`); same docs and scores as
    /// [`MultiTermRewrite::ConstantScore`].
    ConstantScoreBoolean,
    /// Keep the `n` best terms (highest doc freq; lowest edit distance for
    /// fuzzy) and sum their real scores.
    TopTermsScoringBoolean(usize),
    /// Like [`MultiTermRewrite::TopTermsScoringBoolean`], but every term is
    /// scored with the maximum doc freq / total term freq among the kept
    /// terms (Lucene `BlendedTermQuery`), so rare misspellings do not
    /// outrank the intended term. The ES fuzzy default.
    TopTermsBlendedFreqs(usize),
    /// Keep the `n` best terms; each scores its constant boost (1, or the
    /// fuzzy similarity boost) and the boosts are summed.
    TopTermsBoost(usize),
}

impl MultiTermRewrite {
    /// Whether this rewrite produces real (non-constant) scores and so
    /// needs global expansion and statistics.
    pub fn is_scoring(&self) -> bool {
        !matches!(self, MultiTermRewrite::ConstantScore | MultiTermRewrite::ConstantScoreBoolean)
    }

    /// The top-terms bound, if any.
    pub fn top_n(&self) -> Option<usize> {
        match self {
            MultiTermRewrite::TopTermsScoringBoolean(n)
            | MultiTermRewrite::TopTermsBlendedFreqs(n)
            | MultiTermRewrite::TopTermsBoost(n) => Some(*n),
            _ => None,
        }
    }
}

/// Edit distance allowed by a fuzzy query (ES `fuzziness`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fuzziness {
    /// A fixed number of edits (0, 1 or 2; larger values are clamped to 2).
    Edits(u8),
    /// ES `AUTO` = `AUTO:3,6`.
    Auto,
    /// ES `AUTO:low,high`: 0 edits below `low` chars, 1 below `high`, else 2.
    AutoLow(usize, usize),
}

impl Default for Fuzziness {
    fn default() -> Self {
        Fuzziness::Auto
    }
}

impl Fuzziness {
    /// Edits allowed for a query term of `text`.
    pub fn edits(&self, text: &str) -> u8 {
        let len = text.chars().count();
        let (low, high) = match *self {
            Fuzziness::Edits(e) => return e.min(2),
            Fuzziness::Auto => (3, 6),
            Fuzziness::AutoLow(l, h) => (l, h),
        };
        if len < low {
            0
        } else if len < high {
            1
        } else {
            2
        }
    }
}

/// One term a multi-term query expanded to.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpandedTerm {
    /// The term.
    pub term: Term,
    /// Statistics used to score it (index-wide, or leaf-local as a fallback).
    pub stats: TermStats,
    /// Per-term boost (fuzzy similarity `1 - d / len`; 1 otherwise).
    pub boost: f32,
    /// Edit distance (fuzzy only; 0 otherwise).
    pub distance: u8,
}

/// Terms of `field` matched by `matcher` within `[lower, upper)` (in
/// addition to the matcher's literal-prefix bounds), in term order, at
/// most `limit`.
pub fn enumerate_terms(
    leaf: &LeafReader<'_>,
    field: u16,
    matcher: &dyn TermMatcher,
    lower: Option<&[u8]>,
    upper: Option<&[u8]>,
    limit: usize,
) -> Vec<Term> {
    let prefix = matcher.literal_prefix();
    let pre_upper = prefix_upper(prefix);
    let lo: Option<&[u8]> = match (lower, prefix.is_empty()) {
        (Some(l), false) => Some(if l > prefix { l } else { prefix }),
        (Some(l), true) => Some(l),
        (None, false) => Some(prefix),
        (None, true) => None,
    };
    let hi: Option<&[u8]> = match (upper, pre_upper.as_deref()) {
        (Some(u), Some(p)) => Some(if u < p { u } else { p }),
        (Some(u), None) => Some(u),
        (None, p) => p,
    };
    let mut out = Vec::new();
    for (key, _) in leaf.term_range(field, lo, hi) {
        let term = Term::from_key(&key);
        if !matcher.matches(&term.text) {
            continue;
        }
        if out.len() >= limit {
            break;
        }
        out.push(term);
    }
    out
}

/// Upper bound for a prefix walk: the prefix with its last byte bumped.
pub fn prefix_upper(prefix: &[u8]) -> Option<Vec<u8>> {
    if prefix.is_empty() {
        return None;
    }
    let mut up = prefix.to_vec();
    while let Some(last) = up.pop() {
        if last < 0xFF {
            up.push(last + 1);
            return Some(up);
        }
    }
    None
}

/// A rule to compute a fuzzy term's distance and boost, `None` for
/// non-fuzzy queries.
pub type DistanceFn<'a> = Option<&'a dyn Fn(&[u8]) -> u8>;

/// Expand over every leaf, attach index-wide statistics, then apply the
/// rewrite's term selection (top-n or `max_expansions`).
pub fn expand_global(
    leaves: &[LeafReader<'_>],
    field: u16,
    matcher: &dyn TermMatcher,
    bounds: (Option<&[u8]>, Option<&[u8]>),
    max_expansions: usize,
    rewrite: MultiTermRewrite,
    distance: DistanceFn<'_>,
    query_len: usize,
) -> Vec<ExpandedTerm> {
    expand_global_bounded(leaves, field, matcher, bounds, max_expansions, rewrite, distance, query_len, None).expect("unbounded expansion")
}

pub fn expand_global_bounded(
    leaves: &[LeafReader<'_>],
    field: u16,
    matcher: &dyn TermMatcher,
    bounds: (Option<&[u8]>, Option<&[u8]>),
    max_expansions: usize,
    rewrite: MultiTermRewrite,
    distance: DistanceFn<'_>,
    query_len: usize,
    clause_limit: Option<usize>,
) -> Result<Vec<ExpandedTerm>, ()> {
    // Top-terms rewrites consider every match and keep the best n.
    let rewrite = match (rewrite, clause_limit) {
        (MultiTermRewrite::TopTermsScoringBoolean(n),Some(max)) => MultiTermRewrite::TopTermsScoringBoolean(n.min(max)),
        (MultiTermRewrite::TopTermsBlendedFreqs(n),Some(max)) => MultiTermRewrite::TopTermsBlendedFreqs(n.min(max)),
        (MultiTermRewrite::TopTermsBoost(n),Some(max)) => MultiTermRewrite::TopTermsBoost(n.min(max)),
        (rewrite,_) => rewrite,
    };
    let enum_limit = if rewrite.top_n().is_some() { usize::MAX } else { max_expansions.min(clause_limit.map(|n|n.saturating_add(1)).unwrap_or(usize::MAX)) };
    let mut all: BTreeMap<Vec<u8>, Term> = BTreeMap::new();
    for leaf in leaves {
        for t in enumerate_terms(leaf, field, matcher, bounds.0, bounds.1, enum_limit) {
            all.entry(t.text.clone()).or_insert(t);
            if rewrite.top_n().is_none() && clause_limit.is_some_and(|max|max_expansions>max && all.len()>max) { return Err(()); }
        }
    }
    let terms: Vec<Term> = all.into_values().take(enum_limit).collect();
    let expanded = terms
        .into_iter()
        .map(|t| {
            let stats = term_stats(leaves, &t);
            make_expanded(t, stats, distance, query_len)
        })
        .collect();
    Ok(select(expanded, rewrite))
}

/// Leaf-local fallback for [`expand_global`].
pub fn expand_leaf(
    leaf: &LeafReader<'_>,
    field: u16,
    matcher: &dyn TermMatcher,
    bounds: (Option<&[u8]>, Option<&[u8]>),
    max_expansions: usize,
    rewrite: MultiTermRewrite,
    distance: DistanceFn<'_>,
    query_len: usize,
) -> Vec<ExpandedTerm> {
    expand_global(std::slice::from_ref(leaf), field, matcher, bounds, max_expansions, rewrite, distance, query_len)
}

fn make_expanded(term: Term, stats: TermStats, distance: DistanceFn<'_>, query_len: usize) -> ExpandedTerm {
    let (d, boost) = match distance {
        Some(f) => {
            let d = f(&term.text);
            (d, (1.0 - d as f32 / query_len.max(1) as f32).max(0.0))
        }
        None => (0, 1.0),
    };
    ExpandedTerm { term, stats, boost, distance: d }
}

/// Apply top-n selection and blending.
fn select(mut terms: Vec<ExpandedTerm>, rewrite: MultiTermRewrite) -> Vec<ExpandedTerm> {
    if let Some(n) = rewrite.top_n() {
        terms.sort_by(|a, b| {
            a.distance
                .cmp(&b.distance)
                .then(b.stats.doc_freq.cmp(&a.stats.doc_freq))
                .then(a.term.text.cmp(&b.term.text))
        });
        terms.truncate(n);
        terms.sort_by(|a, b| a.term.text.cmp(&b.term.text));
    }
    if let MultiTermRewrite::TopTermsBlendedFreqs(_) = rewrite {
        let max_df = terms.iter().map(|t| t.stats.doc_freq).max().unwrap_or(0);
        let max_ttf = terms.iter().map(|t| t.stats.total_term_freq).max().unwrap_or(0);
        for t in &mut terms {
            t.stats.doc_freq = max_df;
            t.stats.total_term_freq = max_ttf;
        }
    }
    terms
}

/// Constant-score union of the postings of `terms` on `leaf`.
pub fn const_union(leaf: &LeafReader<'_>, terms: &[Term], score: f32) -> Option<Box<dyn Scorer>> {
    let mut docs: Vec<DocId> = Vec::new();
    for t in terms {
        if let Some(mut p) = leaf.postings(t) {
            while let Some(d) = p.next() {
                docs.push(d);
            }
        }
    }
    docs_scorer(docs, score)
}

/// [`const_union`] of the ascending `texts` of `field` (a `terms` query),
/// looked up in one pass over the leaf's dictionary.
pub fn const_union_sorted(leaf: &LeafReader<'_>, field: u16, texts: &[&[u8]], score: f32) -> Option<Box<dyn Scorer>> {
    // A few terms: a lazy union of their postings (Lucene rewrites a
    // `TermInSetQuery` of up to 16 terms to a disjunction), advanced only
    // as far as the conjunction around it asks, instead of every doc of
    // every term listed (and sorted) up front: an EQL join-key filter on
    // a field of three values listed every doc of the leaf per search.
    if texts.len() <= LAZY_UNION_TERMS {
        let mut parts: Vec<Box<dyn Scorer>> = Vec::new();
        leaf.for_each_postings_sorted(field, texts, |p| parts.push(Box::new(ConstScorer::new(crate::query::segment_scorer::PostingsDocSet::new(p), score))));
        return match parts.len() {
            0 => None,
            1 => parts.pop(),
            _ => Some(Box::new(ConstScorer::new(union_of(parts), score))),
        };
    }
    let mut docs: Vec<DocId> = Vec::new();
    leaf.for_each_postings_sorted(field, texts, |mut p| {
        while let Some(d) = p.next() {
            docs.push(d);
        }
    });
    // Many docs (a `terms` filter on a low-cardinality field): a bitset of
    // them instead of sorting (and deduplicating) the list.
    let n = leaf.num_docs() as usize;
    if docs.len().saturating_mul(16) > n && !docs.is_empty() {
        let mut words = vec![0u64; n.div_ceil(64)];
        for d in docs {
            if let Some(w) = words.get_mut(d as usize / 64) {
                *w |= 1 << (d % 64);
            }
        }
        return Some(Box::new(ConstScorer::new(crate::query::docset::BitSetDocSet::new(words), score)));
    }
    docs_scorer(docs, score)
}

/// Most terms [`const_union_sorted`] unions lazily.
const LAZY_UNION_TERMS: usize = 16;

fn docs_scorer(mut docs: Vec<DocId>, score: f32) -> Option<Box<dyn Scorer>> {
    if docs.is_empty() {
        return None;
    }
    docs.sort_unstable();
    docs.dedup();
    Some(Box::new(ConstScorer::new(VecDocSet::new(docs), score)))
}

/// Build the per-leaf scorer of a scoring rewrite over `terms`.
pub fn scoring_union(
    leaf: &LeafReader<'_>,
    terms: &[ExpandedTerm],
    rewrite: MultiTermRewrite,
    sim: &dyn Similarity,
    field_stats: FieldStats,
) -> Option<Box<dyn Scorer>> {
    let mut children: Vec<Box<dyn Scorer>> = Vec::new();
    for t in terms {
        match rewrite {
            MultiTermRewrite::TopTermsBoost(_) => {
                if let Some(s) = const_union(leaf, std::slice::from_ref(&t.term), t.boost) {
                    children.push(s);
                }
            }
            _ => {
                if let Some(s) = leaf.term_scorer_sim(&t.term, sim, 1.0, t.stats, field_stats) {
                    if t.boost == 1.0 {
                        children.push(s);
                    } else {
                        children.push(Box::new(Scaled::new(s, t.boost)));
                    }
                }
            }
        }
    }
    if children.is_empty() {
        None
    } else {
        Some(union_of(children))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzziness_auto() {
        assert_eq!(Fuzziness::Auto.edits("ab"), 0);
        assert_eq!(Fuzziness::Auto.edits("abc"), 1);
        assert_eq!(Fuzziness::Auto.edits("abcde"), 1);
        assert_eq!(Fuzziness::Auto.edits("abcdef"), 2);
        assert_eq!(Fuzziness::AutoLow(2, 4).edits("ab"), 1);
        assert_eq!(Fuzziness::AutoLow(2, 4).edits("abcd"), 2);
        assert_eq!(Fuzziness::Edits(5).edits("x"), 2);
    }
}
