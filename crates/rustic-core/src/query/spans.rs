//! Span queries (Lucene's `spans` package) and ES `intervals` on top of them.
//!
//! A [`Spans`] is a [`DocSet`] that, on every doc it lands on, also exposes
//! the non-empty list of `[start, end)` position ranges that match there.
//! Composites ([`SpanQuery::Near`], [`SpanQuery::Or`], [`SpanQuery::First`],
//! [`SpanQuery::Not`]) only ever land on docs where they have at least one
//! span, so `doc != TERMINATED` implies `!spans().is_empty()`.
//!
//! [`SpanScorer`] counts the spans of a doc as its frequency and scores it
//! through a [`SimWeight`].

use crate::automaton::wildcard::WildcardMatcher;
use crate::automaton::TermMatcher;
use crate::query::docset::{DocSet, TERMINATED};
use crate::query::reader::{LeafReader, TermStats};
use crate::query::scorer::Scorer;
use crate::query::segment_scorer::{Norms, PostingsDocSet};
use crate::query::similarity::SimWeight;
use crate::types::{DocId, Term};
use std::sync::Arc;

/// A position range `[start, end)`.
pub type Span = (u32, u32);

/// Maximum number of terms a multi-term span ([`SpanQuery::Prefix`],
/// [`SpanQuery::Wildcard`]) expands to per leaf (ES `intervals` uses 128;
/// `span_multi` is bounded by the boolean clause limit).
pub const SPAN_MAX_EXPANSIONS: usize = 1024;

/// A doc set that also exposes the matching spans of its current doc.
pub trait Spans: DocSet {
    /// Matching spans of the current doc, sorted by `(start, end)`; never
    /// empty while positioned on a doc.
    fn spans(&self) -> &[Span];
}

/// A span query tree.
#[derive(Clone, Debug, PartialEq)]
pub enum SpanQuery {
    /// Every occurrence of a term (Lucene `SpanTermQuery`).
    Term(Term),
    /// Clauses occurring within `slop` positions of each other (Lucene
    /// `SpanNearQuery`). Ordered: each clause starts at or after the end of
    /// the previous one and the summed gaps are `<= slop`. Unordered: the
    /// window from the first start to the last end, minus the clauses'
    /// lengths, is `<= slop`.
    Near { clauses: Vec<SpanQuery>, slop: u32, in_order: bool },
    /// Union of the clauses' spans (Lucene `SpanOrQuery`).
    Or(Vec<SpanQuery>),
    /// Spans of `inner` ending at or before position `end` (Lucene
    /// `SpanFirstQuery`).
    First { inner: Box<SpanQuery>, end: u32 },
    /// Spans of `include` that do not overlap a span of `exclude` extended
    /// by `pre` positions before and `post` after (Lucene `SpanNotQuery`).
    Not { include: Box<SpanQuery>, exclude: Box<SpanQuery>, pre: u32, post: u32 },
    /// Every occurrence of every term starting with `prefix` (ES
    /// `span_multi` / intervals `prefix`).
    Prefix { field: u16, prefix: Vec<u8> },
    /// Every occurrence of every term matching a `*`/`?` pattern (ES
    /// `span_multi` / intervals `wildcard`).
    Wildcard { field: u16, pattern: String },
}

impl SpanQuery {
    /// Rewrite span_multi against the whole shard dictionary, without silently
    /// truncating to the old fixed per-leaf limit.
    pub(crate) fn rewrite_bounded(&self, leaves: &[LeafReader<'_>], max: usize) -> Result<Self, ()> {
        Ok(match self {
            Self::Prefix { field, .. } | Self::Wildcard { field, .. } => {
                let matcher:Box<dyn TermMatcher>=match self {
                    Self::Prefix { prefix, .. } => Box::new(PrefixBytes(prefix.clone())),
                    Self::Wildcard { pattern, .. } => Box::new(WildcardMatcher::new(pattern)),
                    _ => unreachable!(),
                };
                let mut terms=std::collections::BTreeMap::new();
                for leaf in leaves {
                    for term in crate::query::multi_term::enumerate_terms(leaf,*field,&*matcher,None,None,max.saturating_add(1)) {
                        terms.entry(term.text.clone()).or_insert(term);
                        if terms.len()>max { return Err(()); }
                    }
                }
                Self::Or(terms.into_values().map(Self::Term).collect())
            },
            Self::Near { clauses,slop,in_order } => Self::Near { clauses:clauses.iter().map(|q|q.rewrite_bounded(leaves,max)).collect::<Result<_,_>>()?,slop:*slop,in_order:*in_order },
            Self::Or(clauses) => Self::Or(clauses.iter().map(|q|q.rewrite_bounded(leaves,max)).collect::<Result<_,_>>()?),
            Self::First { inner,end } => Self::First { inner:Box::new(inner.rewrite_bounded(leaves,max)?),end:*end },
            Self::Not { include,exclude,pre,post } => Self::Not { include:Box::new(include.rewrite_bounded(leaves,max)?),exclude:Box::new(exclude.rewrite_bounded(leaves,max)?),pre:*pre,post:*post },
            Self::Term(_) => self.clone(),
        })
    }
    pub(crate) fn clause_count(&self) -> usize {
        match self {
            Self::Near { clauses,.. } | Self::Or(clauses) => clauses.iter().fold(0usize,|n,q|n.saturating_add(q.clause_count())).max(1),
            Self::First { inner,.. } => inner.clause_count(),
            Self::Not { include,exclude,.. } => include.clause_count().saturating_add(exclude.clause_count()),
            _ => 1,
        }
    }
    /// A span over one text term.
    pub fn term(field: u16, text: &str) -> Self {
        SpanQuery::Term(Term::new_text(field, text))
    }

    /// The field this span query targets (`None` for an empty clause list).
    pub fn field(&self) -> Option<u16> {
        match self {
            SpanQuery::Term(t) => Some(t.field),
            SpanQuery::Near { clauses, .. } | SpanQuery::Or(clauses) => clauses.iter().find_map(|c| c.field()),
            SpanQuery::First { inner, .. } => inner.field(),
            SpanQuery::Not { include, .. } => include.field(),
            SpanQuery::Prefix { field, .. } | SpanQuery::Wildcard { field, .. } => Some(*field),
        }
    }

    /// Fixed terms whose index statistics scoring uses (multi-term leaves
    /// are expanded per leaf and contribute leaf-local statistics).
    pub fn collect_terms(&self, out: &mut Vec<Term>) {
        match self {
            SpanQuery::Term(t) => out.push(t.clone()),
            SpanQuery::Near { clauses, .. } | SpanQuery::Or(clauses) => clauses.iter().for_each(|c| c.collect_terms(out)),
            SpanQuery::First { inner, .. } => inner.collect_terms(out),
            SpanQuery::Not { include, .. } => include.collect_terms(out),
            SpanQuery::Prefix { .. } | SpanQuery::Wildcard { .. } => {}
        }
    }

    /// Leaf-local statistics of the terms multi-term leaves expand to here.
    pub fn expanded_stats(&self, leaf: &LeafReader<'_>, out: &mut Vec<TermStats>) {
        match self {
            SpanQuery::Term(_) => {}
            SpanQuery::Near { clauses, .. } | SpanQuery::Or(clauses) => clauses.iter().for_each(|c| c.expanded_stats(leaf, out)),
            SpanQuery::First { inner, .. } => inner.expanded_stats(leaf, out),
            SpanQuery::Not { include, .. } => include.expanded_stats(leaf, out),
            SpanQuery::Prefix { .. } | SpanQuery::Wildcard { .. } => {
                for t in self.expand(leaf) {
                    out.push(TermStats {
                        doc_freq: leaf.doc_freq(&t) as u64,
                        total_term_freq: leaf.total_term_freq(&t),
                        num_docs: leaf.num_docs() as u64,
                    });
                }
            }
        }
    }

    fn expand(&self, leaf: &LeafReader<'_>) -> Vec<Term> {
        let (field, matcher): (u16, Box<dyn TermMatcher>) = match self {
            SpanQuery::Prefix { field, prefix } => (*field, Box::new(PrefixBytes(prefix.clone()))),
            SpanQuery::Wildcard { field, pattern } => (*field, Box::new(WildcardMatcher::new(pattern))),
            _ => return Vec::new(),
        };
        crate::query::multi_term::enumerate_terms(leaf, field, &*matcher, None, None, SPAN_MAX_EXPANSIONS)
    }

    /// Build this query's spans on one leaf; `None` if it cannot match here.
    pub fn spans(&self, leaf: &LeafReader<'_>) -> Option<Box<dyn Spans>> {
        match self {
            SpanQuery::Term(t) => Some(Box::new(TermSpans::new(PostingsDocSet::new(leaf.postings(t)?)))),
            SpanQuery::Near { clauses, slop, in_order } => {
                if clauses.is_empty() {
                    return None;
                }
                let mut children = Vec::with_capacity(clauses.len());
                for c in clauses {
                    children.push(c.spans(leaf)?);
                }
                if children.len() == 1 {
                    return children.pop();
                }
                Some(Box::new(NearSpans::new(children, *slop, *in_order)))
            }
            SpanQuery::Or(clauses) => {
                let mut children: Vec<Box<dyn Spans>> = clauses.iter().filter_map(|c| c.spans(leaf)).collect();
                match children.len() {
                    0 => None,
                    1 => children.pop(),
                    _ => Some(Box::new(OrSpans::new(children))),
                }
            }
            SpanQuery::First { inner, end } => {
                let end = *end;
                Some(Box::new(FilterSpans::new(inner.spans(leaf)?, move |spans: &[Span], _doc| {
                    spans.iter().copied().filter(|s| s.1 <= end).collect()
                })))
            }
            SpanQuery::Not { include, exclude, pre, post } => {
                let inc = include.spans(leaf)?;
                let Some(mut exc) = exclude.spans(leaf) else {
                    return Some(inc);
                };
                let (pre, post) = (*pre, *post);
                let mut primed = false;
                Some(Box::new(FilterSpans::new(inc, move |spans: &[Span], doc| {
                    if !primed || (exc.doc() != TERMINATED && exc.doc() < doc) {
                        exc.seek(doc);
                        primed = true;
                    }
                    if exc.doc() != doc {
                        return spans.to_vec();
                    }
                    let ex = exc.spans();
                    spans
                        .iter()
                        .copied()
                        .filter(|&(s, e)| {
                            !ex.iter().any(|&(xs, xe)| (xs as u64) < e as u64 + post as u64 && xe as u64 + pre as u64 > s as u64)
                        })
                        .collect()
                })))
            }
            SpanQuery::Prefix { .. } | SpanQuery::Wildcard { .. } => {
                let mut children: Vec<Box<dyn Spans>> = self
                    .expand(leaf)
                    .iter()
                    .filter_map(|t| leaf.postings(t))
                    .map(|p| Box::new(TermSpans::new(PostingsDocSet::new(p))) as Box<dyn Spans>)
                    .collect();
                match children.len() {
                    0 => None,
                    1 => children.pop(),
                    _ => Some(Box::new(OrSpans::new(children))),
                }
            }
        }
    }
}

struct PrefixBytes(Vec<u8>);
impl TermMatcher for PrefixBytes {
    fn matches(&self, term: &[u8]) -> bool {
        term.starts_with(&self.0)
    }
    fn literal_prefix(&self) -> &[u8] {
        &self.0
    }
}

/// Spans of one term: `[p, p + 1)` for every position `p`.
pub struct TermSpans {
    postings: PostingsDocSet,
    spans: Vec<Span>,
}

impl TermSpans {
    /// Spans over a positional postings list.
    pub fn new(postings: PostingsDocSet) -> Self {
        Self { postings, spans: Vec::new() }
    }

    fn load(&mut self) -> bool {
        self.spans.clear();
        self.spans.extend(self.postings.positions().iter().map(|&p| (p, p + 1)));
        !self.spans.is_empty()
    }
}

impl DocSet for TermSpans {
    fn advance(&mut self) -> DocId {
        loop {
            let d = self.postings.advance();
            if d == TERMINATED || self.load() {
                return d;
            }
        }
    }
    fn seek(&mut self, target: DocId) -> DocId {
        if self.postings.doc() != TERMINATED && self.postings.doc() >= target {
            return self.postings.doc();
        }
        let d = self.postings.seek(target);
        if d == TERMINATED || self.load() {
            return d;
        }
        self.advance()
    }
    fn doc(&self) -> DocId {
        self.postings.doc()
    }
    fn size_hint(&self) -> u32 {
        self.postings.size_hint()
    }
}

impl Spans for TermSpans {
    fn spans(&self) -> &[Span] {
        &self.spans
    }
}

/// Union of several spans.
pub struct OrSpans {
    children: Vec<Box<dyn Spans>>,
    current: DocId,
    primed: bool,
    spans: Vec<Span>,
}

impl OrSpans {
    fn new(children: Vec<Box<dyn Spans>>) -> Self {
        Self { children, current: TERMINATED, primed: false, spans: Vec::new() }
    }

    fn land(&mut self) -> DocId {
        self.current = self.children.iter().map(|c| c.doc()).min().unwrap_or(TERMINATED);
        self.spans.clear();
        if self.current != TERMINATED {
            for c in &self.children {
                if c.doc() == self.current {
                    self.spans.extend_from_slice(c.spans());
                }
            }
            self.spans.sort_unstable();
            self.spans.dedup();
        }
        self.current
    }
}

impl DocSet for OrSpans {
    fn advance(&mut self) -> DocId {
        if !self.primed {
            self.primed = true;
            for c in &mut self.children {
                c.advance();
            }
        } else if self.current != TERMINATED {
            let d = self.current;
            for c in &mut self.children {
                if c.doc() == d {
                    c.advance();
                }
            }
        } else {
            return TERMINATED;
        }
        self.land()
    }
    fn seek(&mut self, target: DocId) -> DocId {
        if self.current != TERMINATED && self.current >= target {
            return self.current;
        }
        for c in &mut self.children {
            if !self.primed || (c.doc() != TERMINATED && c.doc() < target) {
                c.seek(target);
            }
        }
        self.primed = true;
        self.land()
    }
    fn doc(&self) -> DocId {
        self.current
    }
    fn size_hint(&self) -> u32 {
        self.children.iter().map(|c| c.size_hint()).fold(0u32, |a, b| a.saturating_add(b))
    }
}

impl Spans for OrSpans {
    fn spans(&self) -> &[Span] {
        &self.spans
    }
}

/// Spans of an inner spans filtered per doc by a closure (first / not).
pub struct FilterSpans<F> {
    inner: Box<dyn Spans>,
    filter: F,
    current: DocId,
    spans: Vec<Span>,
}

impl<F: FnMut(&[Span], DocId) -> Vec<Span> + Send> FilterSpans<F> {
    fn new(inner: Box<dyn Spans>, filter: F) -> Self {
        Self { inner, filter, current: TERMINATED, spans: Vec::new() }
    }

    fn confirm_from(&mut self, mut d: DocId) -> DocId {
        loop {
            if d == TERMINATED {
                self.current = TERMINATED;
                return d;
            }
            self.spans = (self.filter)(self.inner.spans(), d);
            if !self.spans.is_empty() {
                self.current = d;
                return d;
            }
            d = self.inner.advance();
        }
    }
}

impl<F: FnMut(&[Span], DocId) -> Vec<Span> + Send> DocSet for FilterSpans<F> {
    fn advance(&mut self) -> DocId {
        let d = self.inner.advance();
        self.confirm_from(d)
    }
    fn seek(&mut self, target: DocId) -> DocId {
        if self.current != TERMINATED && self.current >= target {
            return self.current;
        }
        let d = self.inner.seek(target);
        self.confirm_from(d)
    }
    fn doc(&self) -> DocId {
        self.current
    }
    fn size_hint(&self) -> u32 {
        self.inner.size_hint()
    }
}

impl<F: FnMut(&[Span], DocId) -> Vec<Span> + Send> Spans for FilterSpans<F> {
    fn spans(&self) -> &[Span] {
        &self.spans
    }
}

/// Ordered or unordered proximity of several spans.
pub struct NearSpans {
    children: Vec<Box<dyn Spans>>,
    slop: u32,
    in_order: bool,
    current: DocId,
    primed: bool,
    spans: Vec<Span>,
}

impl NearSpans {
    fn new(children: Vec<Box<dyn Spans>>, slop: u32, in_order: bool) -> Self {
        Self { children, slop, in_order, current: TERMINATED, primed: false, spans: Vec::new() }
    }

    /// Leapfrog the children onto a common doc.
    fn intersect(&mut self) -> DocId {
        loop {
            let mut target = 0;
            for c in &self.children {
                let d = c.doc();
                if d == TERMINATED {
                    return TERMINATED;
                }
                target = target.max(d);
            }
            let mut ok = true;
            for c in &mut self.children {
                if c.doc() < target {
                    let d = c.seek(target);
                    if d == TERMINATED {
                        return TERMINATED;
                    }
                    if d != target {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                return target;
            }
        }
    }

    fn confirm_from(&mut self, mut d: DocId) -> DocId {
        loop {
            if d == TERMINATED {
                self.current = TERMINATED;
                return d;
            }
            let lists: Vec<&[Span]> = self.children.iter().map(|c| c.spans()).collect();
            self.spans = if self.in_order {
                near_ordered(&lists, self.slop)
            } else {
                near_unordered(&lists, self.slop)
            };
            if !self.spans.is_empty() {
                self.current = d;
                return d;
            }
            self.children[0].advance();
            d = self.intersect();
        }
    }
}

/// Ordered proximity matches: for each span of the first clause, take the
/// earliest span of every following clause starting at or after the
/// previous end; keep the combination when the summed gaps are `<= slop`.
pub fn near_ordered(lists: &[&[Span]], slop: u32) -> Vec<Span> {
    let mut out = Vec::new();
    'first: for &(s0, e0) in lists[0] {
        let mut prev_end = e0;
        let mut gaps = 0u64;
        for l in &lists[1..] {
            let i = l.partition_point(|&(s, _)| s < prev_end);
            // Earliest-ending span among those starting at or after prev_end.
            let Some(&(s, e)) = l[i..].iter().min_by_key(|&&(s, e)| (e, s)) else {
                continue 'first;
            };
            gaps += (s - prev_end) as u64;
            prev_end = e;
        }
        if gaps <= slop as u64 {
            out.push((s0, prev_end));
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Unordered proximity matches: sweep the clause with the smallest start,
/// emitting `(min start, max end)` whenever the window minus the clauses'
/// lengths is `<= slop` (Lucene's `NearSpansUnordered`).
pub fn near_unordered(lists: &[&[Span]], slop: u32) -> Vec<Span> {
    let mut idx = vec![0usize; lists.len()];
    let mut out = Vec::new();
    loop {
        let mut min_i = 0;
        let mut min_start = u32::MAX;
        let mut max_end = 0u32;
        let mut total = 0u64;
        for (i, l) in lists.iter().enumerate() {
            let (s, e) = l[idx[i]];
            if s < min_start {
                min_start = s;
                min_i = i;
            }
            max_end = max_end.max(e);
            total += (e - s) as u64;
        }
        let width = (max_end - min_start) as u64;
        if width.saturating_sub(total) <= slop as u64 {
            out.push((min_start, max_end));
        }
        idx[min_i] += 1;
        if idx[min_i] >= lists[min_i].len() {
            break;
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

impl DocSet for NearSpans {
    fn advance(&mut self) -> DocId {
        if !self.primed {
            self.primed = true;
            for c in &mut self.children {
                c.advance();
            }
        } else if self.current == TERMINATED {
            return TERMINATED;
        } else {
            self.children[0].advance();
        }
        let d = self.intersect();
        self.confirm_from(d)
    }
    fn seek(&mut self, target: DocId) -> DocId {
        if self.current != TERMINATED && self.current >= target {
            return self.current;
        }
        for c in &mut self.children {
            if !self.primed || (c.doc() != TERMINATED && c.doc() < target) {
                c.seek(target);
            }
        }
        self.primed = true;
        let d = self.intersect();
        self.confirm_from(d)
    }
    fn doc(&self) -> DocId {
        self.current
    }
    fn size_hint(&self) -> u32 {
        self.children.iter().map(|c| c.size_hint()).min().unwrap_or(0)
    }
}

impl Spans for NearSpans {
    fn spans(&self) -> &[Span] {
        &self.spans
    }
}

/// Scores span matches: the number of spans in a doc is its frequency.
pub struct SpanScorer {
    spans: Box<dyn Spans>,
    weight: Box<dyn SimWeight>,
    norms: Arc<Norms>,
}

impl SpanScorer {
    /// Score `spans` through `weight` with field lengths from `norms`.
    pub fn new(spans: Box<dyn Spans>, weight: Box<dyn SimWeight>, norms: Arc<Norms>) -> Self {
        Self { spans, weight, norms }
    }

    /// Span frequency of the current doc.
    pub fn freq(&self) -> u32 {
        self.spans.spans().len() as u32
    }

    /// Field length of the current doc.
    pub fn field_len(&self) -> u32 {
        self.norms.len(self.spans.doc())
    }

    /// The weight scores go through.
    pub fn weight(&self) -> &dyn SimWeight {
        &*self.weight
    }
}

impl DocSet for SpanScorer {
    fn advance(&mut self) -> DocId {
        self.spans.advance()
    }
    fn seek(&mut self, target: DocId) -> DocId {
        self.spans.seek(target)
    }
    fn doc(&self) -> DocId {
        self.spans.doc()
    }
    fn size_hint(&self) -> u32 {
        self.spans.size_hint()
    }
}

impl Scorer for SpanScorer {
    fn score(&mut self) -> f32 {
        self.weight.score(self.freq(), self.field_len())
    }
}

/// An ES `intervals` rule, mapped onto span queries
/// ([`IntervalsRule::to_span`]). Terms are already analyzed.
#[derive(Clone, Debug, PartialEq)]
pub enum IntervalsRule {
    /// The terms within `max_gaps` positions (`None` = unlimited), in order
    /// if `ordered`.
    Match { terms: Vec<String>, max_gaps: Option<u32>, ordered: bool },
    /// Any of the rules.
    AnyOf(Vec<IntervalsRule>),
    /// All of the rules within `max_gaps`, in order if `ordered`.
    AllOf { rules: Vec<IntervalsRule>, max_gaps: Option<u32>, ordered: bool },
    /// Terms starting with a prefix.
    Prefix(String),
    /// Terms matching a `*`/`?` pattern.
    Wildcard(String),
}

impl IntervalsRule {
    /// ES supplies the Boolean cap to Lucene's per-leaf interval expansion.
    /// Each automaton has its own bound; unlike span_multi, there is no global
    /// Boolean union of terms across leaves.
    pub(crate) fn rewrite_bounded(&self, field:u16, leaf:&LeafReader<'_>, max:usize) -> Result<SpanQuery,String> {
        let gaps = |value:&Option<u32>| value.unwrap_or(u32::MAX);
        match self {
            Self::Prefix(pattern) | Self::Wildcard(pattern) => self.to_span(field)
                .rewrite_bounded(std::slice::from_ref(leaf),max)
                .map_err(|()|format!("Automaton [{pattern}] expanded to too many terms (limit {max})")),
            Self::AnyOf(rules) => Ok(SpanQuery::Or(rules.iter().map(|r|r.rewrite_bounded(field,leaf,max)).collect::<Result<_,_>>()?)),
            Self::AllOf { rules,max_gaps,ordered } => Ok(SpanQuery::Near {
                clauses:rules.iter().map(|r|r.rewrite_bounded(field,leaf,max)).collect::<Result<_,_>>()?,
                slop:gaps(max_gaps),in_order:*ordered }),
            Self::Match { .. } => Ok(self.to_span(field)),
        }
    }
    /// Terms originating from multi-term interval rules, preserving their
    /// leaf-local statistics separately from fixed terms in match rules.
    pub(crate) fn expanded_terms(&self, rewritten:&SpanQuery, out:&mut Vec<Term>) {
        match (self,rewritten) {
            (Self::Prefix(_) | Self::Wildcard(_),span) => span.collect_terms(out),
            (Self::AnyOf(rules),SpanQuery::Or(spans)) |
            (Self::AllOf { rules,.. },SpanQuery::Near { clauses:spans,.. }) => {
                for (rule,span) in rules.iter().zip(spans) { rule.expanded_terms(span,out); }
            },
            _=>{},
        }
    }
    /// The equivalent span query over `field`.
    pub fn to_span(&self, field: u16) -> SpanQuery {
        let gaps = |g: &Option<u32>| g.unwrap_or(u32::MAX);
        match self {
            IntervalsRule::Match { terms, max_gaps, ordered } => {
                let clauses: Vec<SpanQuery> = terms.iter().map(|t| SpanQuery::term(field, t)).collect();
                if clauses.len() == 1 {
                    clauses.into_iter().next().unwrap()
                } else {
                    SpanQuery::Near { clauses, slop: gaps(max_gaps), in_order: *ordered }
                }
            }
            IntervalsRule::AnyOf(rules) => SpanQuery::Or(rules.iter().map(|r| r.to_span(field)).collect()),
            IntervalsRule::AllOf { rules, max_gaps, ordered } => SpanQuery::Near {
                clauses: rules.iter().map(|r| r.to_span(field)).collect(),
                slop: gaps(max_gaps),
                in_order: *ordered,
            },
            IntervalsRule::Prefix(p) => SpanQuery::Prefix { field, prefix: p.as_bytes().to_vec() },
            IntervalsRule::Wildcard(p) => SpanQuery::Wildcard { field, pattern: p.clone() },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_and_unordered_on_lists() {
        // a at 0, 5; b at 2, 9
        let a: &[Span] = &[(0, 1), (5, 6)];
        let b: &[Span] = &[(2, 3), (9, 10)];
        assert_eq!(near_ordered(&[a, b], 0), vec![]);
        assert_eq!(near_ordered(&[a, b], 1), vec![(0, 3)]);
        assert_eq!(near_ordered(&[a, b], 3), vec![(0, 3), (5, 10)]);
        assert_eq!(near_ordered(&[b, a], 2), vec![(2, 6)]);
        // unordered: b(2) a(5) window 4 - 2 = 2
        assert_eq!(near_unordered(&[b, a], 1), vec![(0, 3)]);
        assert_eq!(near_unordered(&[b, a], 2), vec![(0, 3), (2, 6)]);
    }
}
