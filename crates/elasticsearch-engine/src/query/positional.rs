//! Positional queries: ES `intervals` and the `span_*` family.
//!
//! Both compile to a small tree of [`Node`]s over `rustic_core` span
//! machinery. Trees without positional filters (`span_containing`,
//! `span_within`, intervals `filter`) and without intervals scoring run as a
//! core [`Query::Span`]; everything else runs as a [`CustomQuery`] that
//! materializes the matching intervals per leaf.
//!
//! Intervals score like Lucene's `IntervalQuery`: `boost * f / (f + 1)` with
//! `f = sum(1 / (1 + gaps))` over the doc's minimal intervals.

use super::dsl::{parse_fuzziness, scalar_string, QueryContext};
use crate::error::{EsError, Result};
use crate::mapping::{EsType, ResolvedField};
use rustic_core::analysis::Analyzer;
use rustic_core::automaton::levenshtein::LevenshteinMatcher;
use rustic_core::automaton::regexp::{RegexpFlags, RegexpMatcher};
use rustic_core::automaton::wildcard::WildcardMatcher;
use rustic_core::automaton::TermMatcher;
use rustic_core::query::explain::Explanation;
use rustic_core::query::spans::{near_ordered, near_unordered, Span};
use rustic_core::query::{CustomQuery, DocSet, Fuzziness, LeafReader, Query, Scorer, SearchContext, SpanQuery, TERMINATED};
use rustic_core::types::{DocId, Term};
use serde_json::{Map, Value};
use std::sync::Arc;

/// Interval filter relations (ES intervals `filter`, `span_containing`, `span_within`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rel {
    Containing,
    NotContaining,
    ContainedBy,
    NotContainedBy,
    Overlapping,
    NotOverlapping,
    Before,
    After,
}

impl Rel {
    fn parse(s: &str) -> Option<Rel> {
        Some(match s {
            "containing" => Rel::Containing,
            "not_containing" => Rel::NotContaining,
            "contained_by" => Rel::ContainedBy,
            "not_contained_by" => Rel::NotContainedBy,
            "overlapping" => Rel::Overlapping,
            "not_overlapping" => Rel::NotOverlapping,
            "before" => Rel::Before,
            "after" => Rel::After,
            _ => return None,
        })
    }

    /// Does source interval `s` satisfy the relation against `fs`?
    fn keep(self, s: Span, fs: &[Span]) -> bool {
        let contains = |a: Span, b: Span| a.0 <= b.0 && b.1 <= a.1;
        let overlaps = |a: Span, b: Span| a.0 < b.1 && b.0 < a.1;
        match self {
            Rel::Containing => fs.iter().any(|&f| contains(s, f)),
            Rel::NotContaining => !fs.iter().any(|&f| contains(s, f)),
            Rel::ContainedBy => fs.iter().any(|&f| contains(f, s)),
            Rel::NotContainedBy => !fs.iter().any(|&f| contains(f, s)),
            Rel::Overlapping => fs.iter().any(|&f| overlaps(s, f)),
            Rel::NotOverlapping => !fs.iter().any(|&f| overlaps(s, f)),
            Rel::Before => fs.iter().any(|&f| s.1 <= f.0),
            Rel::After => fs.iter().any(|&f| s.0 >= f.1),
        }
    }

    /// Whether a doc without filter intervals can still match.
    fn negative(self) -> bool {
        matches!(self, Rel::NotContaining | Rel::NotContainedBy | Rel::NotOverlapping)
    }
}

/// A positional query tree.
#[derive(Clone, Debug)]
enum Node {
    /// Matches nothing.
    Empty,
    /// A plain core span query.
    Span(SpanQuery),
    Near { children: Vec<Node>, slop: u32, ordered: bool },
    Or(Vec<Node>),
    Filter { src: Box<Node>, rel: Rel, filter: Box<Node> },
    /// `span_first` over a child with a filter: its spans ending by `end`.
    First { inner: Box<Node>, end: u32 },
    /// `span_not` where a side has a filter: `include`'s spans no `exclude`
    /// span overlaps, widened by `pre` / `post` (Lucene `SpanNotQuery`).
    Not { include: Box<Node>, exclude: Box<Node>, pre: u32, post: u32 },
}

impl Node {
    fn or(mut v: Vec<Node>) -> Node {
        v.retain(|n| !matches!(n, Node::Empty));
        match v.len() {
            0 => Node::Empty,
            1 => v.pop().unwrap(),
            _ => Node::Or(v),
        }
    }

    fn near(children: Vec<Node>, slop: u32, ordered: bool) -> Node {
        if children.iter().any(|c| matches!(c, Node::Empty)) || children.is_empty() {
            return Node::Empty;
        }
        if children.len() == 1 {
            return children.into_iter().next().unwrap();
        }
        Node::Near { children, slop, ordered }
    }

    /// Minimal interval length (positions) of a match.
    fn min_len(&self) -> u32 {
        match self {
            Node::Empty => 0,
            Node::Span(q) => span_min_len(q),
            Node::Near { children, .. } => children.iter().map(|c| c.min_len()).sum(),
            Node::Or(v) => v.iter().map(|c| c.min_len()).min().unwrap_or(0),
            Node::Filter { src, .. } => src.min_len(),
            Node::First { inner, .. } => inner.min_len(),
            Node::Not { include, .. } => include.min_len(),
        }
    }

    /// The equivalent core span query, if there is no filter.
    fn to_span(&self) -> Option<SpanQuery> {
        match self {
            Node::Empty => None,
            Node::Span(q) => Some(q.clone()),
            Node::Near { children, slop, ordered } => {
                Some(SpanQuery::Near { clauses: children.iter().map(|c| c.to_span()).collect::<Option<Vec<_>>>()?, slop: *slop, in_order: *ordered })
            }
            Node::Or(v) => Some(SpanQuery::Or(v.iter().map(|c| c.to_span()).collect::<Option<Vec<_>>>()?)),
            Node::Filter { .. } => None,
            Node::First { inner, end } => Some(SpanQuery::First { inner: Box::new(inner.to_span()?), end: *end }),
            Node::Not { include, exclude, pre, post } => {
                Some(SpanQuery::Not { include: Box::new(include.to_span()?), exclude: Box::new(exclude.to_span()?), pre: *pre, post: *post })
            }
        }
    }

    fn collect_terms(&self, out: &mut Vec<Term>) {
        match self {
            Node::Empty => {}
            Node::Span(q) => q.collect_terms(out),
            Node::Near { children: v, .. } | Node::Or(v) => v.iter().for_each(|c| c.collect_terms(out)),
            Node::First { inner, .. } => inner.collect_terms(out),
            // Lucene `SpanNotQuery` weighs the include side's terms only.
            Node::Not { include, .. } => include.collect_terms(out),
            Node::Filter { src, filter, .. } => {
                src.collect_terms(out);
                filter.collect_terms(out);
            }
        }
    }

    /// Lucene's disjunction rewrite: an ordered/gap-limited near over a
    /// disjunction of differently sized alternatives becomes a disjunction
    /// of nears (so `the (big | big bad) wolf` matches both phrasings).
    fn rewrite_disjunctions(self, budget: &mut usize) -> Result<Node> {
        Ok(match self {
            Node::Near { children, slop, ordered } => {
                let children: Vec<Node> = children.into_iter().map(|c| c.rewrite_disjunctions(budget)).collect::<Result<_>>()?;
                let pos = children.iter().position(|c| match c {
                    Node::Or(v) => v.iter().map(|x| x.min_len()).collect::<std::collections::BTreeSet<_>>().len() > 1,
                    _ => false,
                });
                match pos {
                    Some(i) if children.len() > 1 => {
                        let Node::Or(alts) = children[i].clone() else { unreachable!() };
                        let mut out = Vec::new();
                        // Every alternative, each expansion counted against
                        // the clause limit before it is built (Lucene
                        // `Disjunctions.pullUp`): the product of several
                        // groups is refused, never silently cut.
                        for a in alts {
                            if *budget == 0 {
                                return Err(EsError::illegal_argument("Too many disjunctions to expand"));
                            }
                            *budget -= 1;
                            let mut c = children.clone();
                            c[i] = a;
                            out.push(Node::near(c, slop, ordered).rewrite_disjunctions(budget)?);
                        }
                        Node::or(out)
                    }
                    _ => Node::near(children, slop, ordered),
                }
            }
            Node::Or(v) => Node::or(v.into_iter().map(|c| c.rewrite_disjunctions(budget)).collect::<Result<_>>()?),
            Node::Filter { src, rel, filter } => Node::Filter { src: Box::new(src.rewrite_disjunctions(budget)?), rel, filter: Box::new(filter.rewrite_disjunctions(budget)?) },
            Node::First { inner, end } => Node::First { inner: Box::new(inner.rewrite_disjunctions(budget)?), end },
            Node::Not { include, exclude, pre, post } => {
                Node::Not { include: Box::new(include.rewrite_disjunctions(budget)?), exclude: Box::new(exclude.rewrite_disjunctions(budget)?), pre, post }
            }
            other => other,
        })
    }

    /// Every matching doc of the leaf with its sorted intervals; the
    /// materialized occurrences are charged to the request breaker and the
    /// search's deadline / cancellation polled as they are read (`b`: on a
    /// refusal or a stop, evaluation ends and `b.failed` is set).
    fn eval(&self, leaf: &LeafReader<'_>, b: &mut EvalBudget<'_>) -> Vec<(DocId, Vec<Span>)> {
        match self {
            Node::Empty => Vec::new(),
            Node::Span(q) => {
                let Some(mut s) = q.spans(leaf) else { return Vec::new() };
                let mut out = Vec::new();
                let mut d = s.advance();
                while d != TERMINATED {
                    let spans = s.spans();
                    if !b.admit(48 + 8 * spans.len() as u64) {
                        return Vec::new();
                    }
                    out.push((d, spans.to_vec()));
                    d = s.advance();
                }
                out
            }
            Node::Near { children, slop, ordered } => {
                let mut lists: Vec<Vec<(DocId, Vec<Span>)>> = Vec::with_capacity(children.len());
                for c in children {
                    let l = c.eval(leaf, b);
                    if b.failed || l.is_empty() {
                        return Vec::new();
                    }
                    lists.push(l);
                }
                let mut out = Vec::new();
                'docs: for (doc, first) in &lists[0] {
                    let mut per: Vec<&[Span]> = vec![first.as_slice()];
                    for l in &lists[1..] {
                        match l.binary_search_by_key(doc, |(d, _)| *d) {
                            Ok(i) => per.push(l[i].1.as_slice()),
                            Err(_) => continue 'docs,
                        }
                    }
                    let spans = if *ordered { near_ordered(&per, *slop) } else { near_unordered(&per, *slop) };
                    if !spans.is_empty() {
                        if !b.admit(48 + 8 * spans.len() as u64) {
                            return Vec::new();
                        }
                        out.push((*doc, spans));
                    }
                }
                out
            }
            Node::Or(v) => {
                let mut all: std::collections::BTreeMap<DocId, Vec<Span>> = std::collections::BTreeMap::new();
                for c in v {
                    for (d, s) in c.eval(leaf, b) {
                        all.entry(d).or_default().extend(s);
                    }
                    if b.failed {
                        return Vec::new();
                    }
                }
                all.into_iter()
                    .map(|(d, mut s)| {
                        s.sort_unstable();
                        s.dedup();
                        (d, s)
                    })
                    .collect()
            }
            Node::First { inner, end } => inner
                .eval(leaf, b)
                .into_iter()
                .filter_map(|(d, s)| {
                    let kept: Vec<Span> = s.into_iter().filter(|sp| sp.1 <= *end).collect();
                    (!kept.is_empty()).then_some((d, kept))
                })
                .collect(),
            Node::Not { include, exclude, pre, post } => {
                // Both sides materialize under the same budget.
                let ex = exclude.eval(leaf, b);
                if b.failed {
                    return Vec::new();
                }
                let inc = include.eval(leaf, b);
                if b.failed {
                    return Vec::new();
                }
                inc
                    .into_iter()
                    .filter_map(|(d, s)| {
                        let es: &[Span] = match ex.binary_search_by_key(&d, |(x, _)| *x) {
                            Ok(i) => &ex[i].1,
                            Err(_) => &[],
                        };
                        let kept: Vec<Span> = s
                            .into_iter()
                            .filter(|i| !es.iter().any(|e| e.1 + pre > i.0 && e.0 < i.1 + post))
                            .collect();
                        (!kept.is_empty()).then_some((d, kept))
                    })
                    .collect()
            }
            Node::Filter { src, rel, filter } => {
                let fl = filter.eval(leaf, b);
                if b.failed {
                    return Vec::new();
                }
                let src_docs = src.eval(leaf, b);
                if b.failed {
                    return Vec::new();
                }
                let mut out = Vec::new();
                for (doc, spans) in src_docs {
                    let fs: &[Span] = match fl.binary_search_by_key(&doc, |(d, _)| *d) {
                        Ok(i) => &fl[i].1,
                        Err(_) => &[],
                    };
                    let kept: Vec<Span> = if fs.is_empty() {
                        if rel.negative() { spans } else { Vec::new() }
                    } else {
                        spans.into_iter().filter(|s| rel.keep(*s, fs)).collect()
                    };
                    if !kept.is_empty() {
                        out.push((doc, kept));
                    }
                }
                out
            }
        }
    }
}

/// The request-breaker charge and deadline / cancellation polling of one
/// positional evaluation (see [`Node::eval`]).
struct EvalBudget<'c> {
    mem: crate::memory::RequestMemory,
    ctx: &'c SearchContext,
    seen: u64,
    failed: bool,
}

impl<'c> EvalBudget<'c> {
    fn new(ctx: &'c SearchContext) -> Self {
        EvalBudget { mem: crate::memory::RequestMemory::new(), ctx, seen: 0, failed: false }
    }

    /// Admit `bytes` more of materialized matches; false (and `failed`)
    /// once refused or once the search was stopped.
    fn admit(&mut self, bytes: u64) -> bool {
        if self.failed {
            return false;
        }
        self.seen += 1;
        if self.seen % 1024 == 0 && self.ctx.stopped() {
            self.failed = true;
            return false;
        }
        if let Err(e) = self.mem.add(bytes, "<positional query matches>") {
            self.ctx.record_memory_error(e.reason);
            self.failed = true;
            return false;
        }
        true
    }
}

fn span_min_len(q: &SpanQuery) -> u32 {
    match q {
        SpanQuery::Term(_) | SpanQuery::Prefix { .. } | SpanQuery::Wildcard { .. } => 1,
        SpanQuery::Near { clauses, .. } => clauses.iter().map(span_min_len).sum(),
        SpanQuery::Or(v) => v.iter().map(span_min_len).min().unwrap_or(0),
        SpanQuery::First { inner, .. } => span_min_len(inner),
        SpanQuery::Not { include, .. } => span_min_len(include),
    }
}

/// How a materialized positional query scores.
#[derive(Clone, Copy, Debug)]
enum Scoring {
    /// Lucene `IntervalQuery`: `boost * f / (f + 1)`.
    Intervals,
    /// Lucene `SpanScorer`: the field similarity over the summed term
    /// statistics, with frequency `sum(1 / (1 + width))` (filtered span
    /// queries, which Lucene scores as ordinary span weights).
    Span,
}

#[derive(Debug)]
struct PositionalQuery {
    node: Node,
    min_len: u32,
    scoring: Scoring,
    boost: f32,
    name: &'static str,
}

impl PositionalQuery {
    fn score(&self, spans: &[Span]) -> f32 {
        match self.scoring {
            Scoring::Span => self.boost,
            Scoring::Intervals => {
                let f: f32 = spans.iter().map(|(s, e)| 1.0 / (1.0 + (e - s).saturating_sub(self.min_len) as f32)).sum();
                self.boost * f / (f + 1.0)
            }
        }
    }
}

impl CustomQuery for PositionalQuery {
    fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        // Every path (span- or interval-scored) materializes its matches
        // under the request budget and the search's deadline/cancellation.
        let mut budget = EvalBudget::new(ctx);
        let matched = self.node.eval(leaf, &mut budget);
        // Refused (recorded on the context) or stopped (the search reports
        // its timeout / cancellation): no partial matches as complete ones.
        if budget.failed {
            return None;
        }
        let docs: Vec<(DocId, f32)> = match self.scoring {
            Scoring::Span => {
                let mut terms = Vec::new();
                self.node.collect_terms(&mut terms);
                let field = terms.first()?.field;
                let mut stats: Vec<rustic_core::query::TermStats> = terms.iter().map(|t| ctx.stats(t)).collect();
                if stats.is_empty() {
                    stats.push(rustic_core::query::TermStats { doc_freq: 1, total_term_freq: 1, num_docs: leaf.num_docs() as u64 });
                }
                let mut fs = ctx.fstats(field);
                if fs.num_docs == 0 {
                    fs = leaf.field_stats(field);
                }
                let weight = ctx.sim(field).phrase_weight(&stats, fs, self.boost);
                let norms = leaf.norms(field);
                matched.into_iter()
                    .map(|(d, s)| (d, weight.score_freq(rustic_core::query::spans::sloppy_freq(&s, self.min_len), norms.len(d))))
                    .collect()
            }
            _ => matched.into_iter().map(|(d, s)| (d, self.score(&s))).collect(),
        };
        if docs.is_empty() {
            return None;
        }
        Some(Box::new(rustic_core::query::query::ScoredDocs::new(docs)))
    }
    fn name(&self) -> &'static str {
        self.name
    }
    fn explain(&self, _leaf: &LeafReader<'_>, _ctx: &SearchContext, _doc: DocId, score: f32) -> Explanation {
        Explanation::leaf(score, format!("{} query match", self.name))
    }
}

fn custom(node: Node, scoring: Scoring, boost: f32, name: &'static str) -> Query {
    if matches!(node, Node::Empty) {
        return Query::MatchNone;
    }
    let min_len = node.min_len();
    Query::Custom(Arc::new(PositionalQuery { node, min_len, scoring, boost, name }))
}

// ---------------------------------------------------------------------------
// Term expansion over the searched leaves.

struct PrefixMatcher(Vec<u8>);
impl TermMatcher for PrefixMatcher {
    fn matches(&self, term: &[u8]) -> bool {
        term.starts_with(&self.0)
    }
    fn literal_prefix(&self) -> &[u8] {
        &self.0
    }
}


/// Terms between two (analysed) bounds, for the `range` interval rule.
struct TermRange {
    lower: std::ops::Bound<Vec<u8>>,
    upper: std::ops::Bound<Vec<u8>>,
}

impl TermMatcher for TermRange {
    fn matches(&self, t: &[u8]) -> bool {
        use std::ops::Bound::*;
        (match &self.lower {
            Included(l) => t >= l.as_slice(),
            Excluded(l) => t > l.as_slice(),
            Unbounded => true,
        }) && (match &self.upper {
            Included(u) => t <= u.as_slice(),
            Excluded(u) => t < u.as_slice(),
            Unbounded => true,
        })
    }
    fn literal_prefix(&self) -> &[u8] {
        &[]
    }
}

/// An intervals multi-term rule's expansion (Lucene
/// `MultiTermIntervalsSource`): every matching term up to the clause limit;
/// more fail ("expanded to too many terms") instead of being cut.
fn expand_rule(field: u16, m: &dyn TermMatcher, ctx: &QueryContext, pattern: &str) -> Result<Node> {
    let limit = crate::query::limits::clause_limit();
    let node = expand(field, m, ctx, limit.saturating_add(1));
    if let Node::Or(v) = &node {
        if v.len() > limit {
            return Err(EsError::new(500, "illegal_state_exception", format!("Automaton [{pattern}] expanded to too many terms (limit {limit})")));
        }
    }
    Ok(node)
}

fn expand(field: u16, m: &dyn TermMatcher, ctx: &QueryContext, limit: usize) -> Node {
    let mut texts: Vec<Vec<u8>> = Vec::new();
    for leaf in ctx.leaves {
        for t in rustic_core::query::multi_term::enumerate_terms(leaf, field, m, None, None, limit) {
            if !texts.contains(&t.text) {
                texts.push(t.text);
            }
        }
    }
    texts.sort();
    texts.truncate(limit);
    Node::or(texts.into_iter().map(|t| Node::Span(SpanQuery::Term(Term::new(field, t)))).collect())
}

fn fuzzy_edits(f: &Fuzziness, len: usize) -> u8 {
    match f {
        Fuzziness::Edits(n) => (*n).min(2),
        Fuzziness::Auto => {
            if len < 3 {
                0
            } else if len < 6 {
                1
            } else {
                2
            }
        }
        Fuzziness::AutoLow(lo, hi) => {
            if len < *lo {
                0
            } else if len < *hi {
                1
            } else {
                2
            }
        }
    }
}

fn parsing(reason: impl Into<String>) -> EsError {
    EsError::parsing(reason)
}

fn analyze_positions(a: &Analyzer, text: &str) -> Vec<Vec<String>> {
    let mut out: Vec<(u32, Vec<String>)> = Vec::new();
    for t in a.analyze(text) {
        match out.last_mut() {
            Some((p, v)) if *p == t.position => {
                if !v.contains(&t.text) {
                    v.push(t.text)
                }
            }
            _ => out.push((t.position, vec![t.text])),
        }
    }
    out.into_iter().map(|(_, v)| v).collect()
}

fn normalize(a: &Analyzer, s: &str) -> String {
    let toks = a.analyze(s);
    if toks.len() == 1 {
        toks.into_iter().next().unwrap().text
    } else {
        s.to_lowercase()
    }
}

// ---------------------------------------------------------------------------
// intervals.

struct IvCtx<'c, 'a> {
    ctx: &'c QueryContext<'a>,
    field: Arc<ResolvedField>,
    id: u16,
}

impl IvCtx<'_, '_> {
    fn analyzer(&self, o: &Map<String, Value>) -> Result<Arc<Analyzer>> {
        if let Some(a) = o.get("analyzer").and_then(Value::as_str) {
            return self.ctx.lookup.registry.analyzer(a).ok_or_else(|| EsError::illegal_argument(format!("Analyzer [{a}] not found")));
        }
        Ok(self.field.search_analyzer.clone().unwrap_or_else(|| Arc::new(Analyzer::standard())))
    }

    /// `use_field`: run the rule on another field (ES masks it as this one).
    fn target(&self, o: &Map<String, Value>) -> Result<(u16, Arc<Analyzer>)> {
        if let Some(uf) = o.get("use_field").and_then(Value::as_str) {
            let f = self.ctx.lookup.get(uf).ok_or_else(|| EsError::illegal_argument(format!("Cannot use field [{uf}] in intervals query: field does not exist")))?;
            let id = f.core_id.ok_or_else(|| EsError::illegal_argument(format!("Cannot use field [{uf}] with no positions")))?;
            let a = match o.get("analyzer").and_then(Value::as_str) {
                Some(a) => self.ctx.lookup.registry.analyzer(a).ok_or_else(|| EsError::illegal_argument(format!("Analyzer [{a}] not found")))?,
                None => f.search_analyzer.clone().unwrap_or_else(|| Arc::new(Analyzer::standard())),
            };
            return Ok((id, a));
        }
        Ok((self.id, self.analyzer(o)?))
    }

    fn rule(&self, v: &Value) -> Result<Node> {
        let o = v.as_object().ok_or_else(|| parsing("Expected [START_OBJECT] but found [VALUE_STRING]"))?;
        let mut it = o.iter();
        let (name, body) = it.next().ok_or_else(|| parsing("Expected [FIELD_NAME] but got [END_OBJECT]"))?;
        if it.next().is_some() {
            return Err(parsing(format!("Only one interval rule can be specified, found [{}] and [{}]", name, o.keys().nth(1).unwrap())));
        }
        let b = body.as_object().ok_or_else(|| parsing(format!("Expected [START_OBJECT] but found [{name}] value")))?;
        let node = match name.as_str() {
            "match" => {
                let q = b.get("query").ok_or_else(|| parsing("[match] requires [query]"))?;
                let (id, a) = self.target(b)?;
                let ordered = b.get("ordered").and_then(Value::as_bool).unwrap_or(false);
                let gaps = max_gaps(b);
                let pos = analyze_positions(&a, &scalar_string(q));
                let children: Vec<Node> = pos.iter().map(|alts| Node::or(alts.iter().map(|t| Node::Span(SpanQuery::term(id, t))).collect())).collect();
                if children.is_empty() {
                    Node::Empty
                } else {
                    Node::near(children, gaps, ordered)
                }
            }
            "any_of" => {
                let rules = b.get("intervals").and_then(Value::as_array).ok_or_else(|| parsing("[any_of] requires [intervals]"))?;
                Node::or(rules.iter().map(|r| self.rule(r)).collect::<Result<Vec<_>>>()?)
            }
            "all_of" => {
                let rules = b.get("intervals").and_then(Value::as_array).ok_or_else(|| parsing("[all_of] requires [intervals]"))?;
                let ordered = b.get("ordered").and_then(Value::as_bool).unwrap_or(false);
                Node::near(rules.iter().map(|r| self.rule(r)).collect::<Result<Vec<_>>>()?, max_gaps(b), ordered)
            }
            "prefix" => {
                let p = b.get("prefix").map(scalar_string).ok_or_else(|| parsing("[prefix] requires [prefix]"))?;
                let (id, a) = self.target(b)?;
                let p = normalize(&a, &p);
                expand_rule(id, &PrefixMatcher(p.clone().into_bytes()), self.ctx, &format!("{p}*"))?
            }
            "wildcard" => {
                let p = b.get("pattern").map(scalar_string).ok_or_else(|| parsing("[wildcard] requires [pattern]"))?;
                let (id, a) = self.target(b)?;
                let p = if b.contains_key("analyzer") { normalize(&a, &p) } else { p.to_lowercase() };
                expand_rule(id, &WildcardMatcher::new(&p), self.ctx, &p)?
            }
            "regexp" => {
                let p = b.get("pattern").map(scalar_string).ok_or_else(|| parsing("[regexp] requires [pattern]"))?;
                let (id, _) = self.target(b)?;
                let m = RegexpMatcher::new(&p, RegexpFlags::NONE).map_err(|e| parsing(e))?;
                expand_rule(id, &m, self.ctx, &p)?
            }
            "range" => {
                let (id, a) = self.target(b)?;
                let bound = |inc: &str, exc: &str| -> Result<std::ops::Bound<Vec<u8>>> {
                    Ok(match (b.get(inc), b.get(exc)) {
                        (Some(_), Some(_)) => return Err(parsing(format!("Only one of [{inc}] and [{exc}] can be specified"))),
                        (Some(v), None) => std::ops::Bound::Included(normalize(&a, &scalar_string(v)).into_bytes()),
                        (None, Some(v)) => std::ops::Bound::Excluded(normalize(&a, &scalar_string(v)).into_bytes()),
                        (None, None) => std::ops::Bound::Unbounded,
                    })
                };
                let (lower, upper) = (bound("gte", "gt")?, bound("lte", "lt")?);
                if matches!((&lower, &upper), (std::ops::Bound::Unbounded, std::ops::Bound::Unbounded)) {
                    return Err(parsing("[range] requires at least one of [gte], [gt], [lte], [lt]"));
                }
                expand_rule(id, &TermRange { lower, upper }, self.ctx, "range")?
            }
            "fuzzy" => {
                let t = b.get("term").map(scalar_string).ok_or_else(|| parsing("[fuzzy] requires [term]"))?;
                let (id, a) = self.target(b)?;
                let t = normalize(&a, &t);
                let fz = match b.get("fuzziness") {
                    Some(v) => parse_fuzziness(v)?,
                    None => Fuzziness::Auto,
                };
                let edits = fuzzy_edits(&fz, t.chars().count());
                let pl = b.get("prefix_length").and_then(Value::as_u64).unwrap_or(0) as usize;
                let tr = b.get("transpositions").and_then(Value::as_bool).unwrap_or(true);
                expand_rule(id, &LevenshteinMatcher::new(&t, edits, pl, tr), self.ctx, &t)?
            }
            other => return Err(parsing(format!("Unknown interval type [{other}], expecting one of [match, any_of, all_of, prefix, wildcard, fuzzy, regexp, range]"))),
        };
        match b.get("filter") {
            None => Ok(node),
            Some(fv) => {
                let fo = fv.as_object().ok_or_else(|| parsing("Expected [START_OBJECT] but found [filter] value"))?;
                let (k, rv) = fo.iter().next().ok_or_else(|| parsing("Expected [FIELD_NAME] but got [END_OBJECT]"))?;
                if k == "script" {
                    return Err(EsError::illegal_argument("intervals script filters are not supported"));
                }
                let rel = Rel::parse(k).ok_or_else(|| parsing(format!("Unknown filter type [{k}]")))?;
                let filter = self.rule(rv)?;
                Ok(Node::Filter { src: Box::new(node), rel, filter: Box::new(filter) })
            }
        }
    }
}

fn max_gaps(b: &Map<String, Value>) -> u32 {
    match b.get("max_gaps").and_then(Value::as_i64) {
        Some(n) if n >= 0 => n as u32,
        _ => u32::MAX,
    }
}

/// `{"intervals": {"<field>": {<rule>}}}`.
pub fn intervals_query(body: &Value, ctx: &QueryContext) -> Result<Query> {
    let o = body.as_object().ok_or_else(|| parsing("[intervals] query malformed, no start_object after query name"))?;
    let (field, spec) = o.iter().find(|(k, _)| !matches!(k.as_str(), "boost" | "_name")).ok_or_else(|| parsing("[intervals] query requires a field"))?;
    let so = spec.as_object().ok_or_else(|| parsing(format!("Expected [START_OBJECT] but got [{spec}]")))?;
    let boost = so.get("boost").and_then(Value::as_f64).unwrap_or(1.0) as f32;
    let _name = so.get("_name");
    let mut rule = Map::new();
    for (k, v) in so {
        if !matches!(k.as_str(), "boost" | "_name") {
            rule.insert(k.clone(), v.clone());
        }
    }
    if rule.is_empty() {
        return Err(parsing("Expected [FIELD_NAME] but got [END_OBJECT]"));
    }
    let Some(f) = ctx.lookup.get(field).cloned() else {
        return Ok(Query::MatchNone);
    };
    if !f.ty.is_text() || f.ty == EsType::MatchOnlyText && f.core_id.is_none() {
        return Err(ctx_err(ctx, format!("failed to create query: Can only use interval queries on text fields - not on [{}] which is of type [{}]", f.name, f.declared.name())));
    }
    let Some(id) = f.core_id else { return Ok(Query::MatchNone) };
    let iv = IvCtx { ctx, field: f, id };
    let node = iv.rule(&Value::Object(rule))?.rewrite_disjunctions(&mut crate::query::limits::clause_limit())?;
    Ok(custom(node, Scoring::Intervals, boost, "intervals"))
}

fn ctx_err(ctx: &QueryContext, reason: String) -> EsError {
    EsError::query_shard(ctx.index_name, ctx.index_uuid, reason)
}

// ---------------------------------------------------------------------------
// span_*.

fn span_obj<'v>(name: &str, v: &'v Value) -> Result<&'v Map<String, Value>> {
    v.as_object().ok_or_else(|| parsing(format!("[{name}] query malformed, no start_object after query name")))
}

fn span_clause(v: &Value, ctx: &QueryContext) -> Result<Node> {
    let o = v.as_object().ok_or_else(|| parsing("spanNear [clauses] must be of type span query"))?;
    let (name, body) = o.iter().next().ok_or_else(|| parsing("query malformed, empty clause found"))?;
    if !name.starts_with("span_") && name != "field_masking_span" {
        return Err(parsing(format!("span query must be of type span query, found [{name}]")));
    }
    span_node(name, body, ctx)
}

fn span_node(name: &str, body: &Value, ctx: &QueryContext) -> Result<Node> {
    let o = span_obj(name, body)?;
    Ok(match name {
        "span_term" => {
            let (field, spec) = o.iter().find(|(k, _)| !matches!(k.as_str(), "boost" | "_name")).ok_or_else(|| parsing("[span_term] requires a field"))?;
            let v = match spec {
                Value::Object(so) => so.get("value").or(so.get("term")).cloned().ok_or_else(|| parsing("[span_term] requires [value]"))?,
                other => other.clone(),
            };
            match ctx.lookup.get(field).and_then(|f| f.core_id.map(|id| (id, f.clone()))) {
                // `pattern_text` with `index_options: docs` has no positions.
                Some((_, f)) if f.declared == EsType::PatternText && f.ext.as_ref().and_then(|e| e.params.get("index_options")).and_then(Value::as_str) != Some("positions") => {
                    return Err(EsError::query_shard(ctx.index_name, ctx.index_uuid, format!("field:[{}] was indexed without position data; cannot run SpanTermQuery (term={})", f.name, scalar_string(&v))));
                }
                Some((id, f)) => {
                    let s = scalar_string(&v);
                    let s = if f.ty.is_keyword_like() { f.normalizer.as_ref().map(|n| n.analyze(&s).into_iter().map(|t| t.text).collect::<String>()).unwrap_or(s) } else { s };
                    Node::Span(SpanQuery::term(id, &s))
                }
                None => Node::Empty,
            }
        }
        "span_multi" => {
            let m = o.get("match").and_then(Value::as_object).ok_or_else(|| parsing("spanMultiTerm must have [match] multi term query clause"))?;
            let (qn, qb) = m.iter().next().ok_or_else(|| parsing("spanMultiTerm must have [match] multi term query clause"))?;
            let (field, spec) = qb.as_object().and_then(|x| x.iter().find(|(k, _)| !matches!(k.as_str(), "boost" | "_name"))).ok_or_else(|| parsing(format!("[{qn}] requires a field")))?;
            let Some((id, f)) = ctx.lookup.get(field).and_then(|f| f.core_id.map(|id| (id, f.clone()))) else { return Ok(Node::Empty) };
            let (val, so) = match spec {
                Value::Object(so) => (so.get("value").or(so.get(qn.as_str())).map(scalar_string).unwrap_or_default(), so.clone()),
                other => (scalar_string(other), Map::new()),
            };
            let limit = 1024;
            match qn.as_str() {
                "prefix" => expand(id, &PrefixMatcher(val.into_bytes()), ctx, limit),
                "wildcard" => expand(id, &WildcardMatcher::new(&val), ctx, limit),
                "regexp" => expand(id, &RegexpMatcher::new(&val, RegexpFlags::NONE).map_err(|e| parsing(e))?, ctx, limit),
                "fuzzy" => {
                    let fz = match so.get("fuzziness") {
                        Some(v) => parse_fuzziness(v)?,
                        None => Fuzziness::Auto,
                    };
                    let pl = so.get("prefix_length").and_then(Value::as_u64).unwrap_or(0) as usize;
                    let tr = so.get("transpositions").and_then(Value::as_bool).unwrap_or(true);
                    let me = so.get("max_expansions").and_then(Value::as_u64).unwrap_or(50) as usize;
                    expand(id, &LevenshteinMatcher::new(&val, fuzzy_edits(&fz, val.chars().count()), pl, tr), ctx, me)
                }
                "range" => {
                    let _ = f;
                    return Err(ctx_err(ctx, "[span_multi] range queries are not supported".into()));
                }
                other => return Err(parsing(format!("spanMultiTerm [match] must be of type multi term query, found [{other}]"))),
            }
        }
        "span_near" => {
            let cl = o.get("clauses").and_then(Value::as_array).ok_or_else(|| parsing("span_near must include [clauses]"))?;
            if cl.is_empty() {
                return Err(parsing("span_near must include [clauses]"));
            }
            let slop = o.get("slop").and_then(Value::as_u64).unwrap_or(0) as u32;
            let ordered = o.get("in_order").and_then(Value::as_bool).unwrap_or(true);
            Node::near(cl.iter().map(|c| span_clause(c, ctx)).collect::<Result<Vec<_>>>()?, slop, ordered)
        }
        "span_or" => {
            let cl = o.get("clauses").and_then(Value::as_array).ok_or_else(|| parsing("spanOr must include [clauses]"))?;
            Node::or(cl.iter().map(|c| span_clause(c, ctx)).collect::<Result<Vec<_>>>()?)
        }
        "span_first" => {
            let m = o.get("match").ok_or_else(|| parsing("spanFirst must have [match] span query clause"))?;
            let end = o.get("end").and_then(Value::as_u64).ok_or_else(|| parsing("spanFirst must have [end] set for it"))? as u32;
            let inner = span_clause(m, ctx)?;
            match inner.to_span() {
                Some(s) => Node::Span(SpanQuery::First { inner: Box::new(s), end }),
                None if matches!(inner, Node::Empty) => Node::Empty,
                // A filtered child (containing / within) is kept and
                // evaluated, not dropped as matching nothing.
                None => Node::First { inner: Box::new(inner), end },
            }
        }
        "span_not" => {
            let inc = o.get("include").ok_or_else(|| parsing("span_not must have [include] span query clause"))?;
            let exc = o.get("exclude").ok_or_else(|| parsing("span_not must have [exclude] span query clause"))?;
            let dist = o.get("dist").and_then(Value::as_u64);
            if dist.is_some() && (o.contains_key("pre") || o.contains_key("post")) {
                return Err(parsing("spanNot can either use [dist] or [pre] & [post] (or none)"));
            }
            let pre = dist.or(o.get("pre").and_then(Value::as_u64)).unwrap_or(0) as u32;
            let post = dist.or(o.get("post").and_then(Value::as_u64)).unwrap_or(0) as u32;
            let i = span_clause(inc, ctx)?;
            let e = span_clause(exc, ctx)?;
            match (&i, &e, i.to_span(), e.to_span()) {
                (_, _, Some(is), Some(es)) => Node::Span(SpanQuery::Not { include: Box::new(is), exclude: Box::new(es), pre, post }),
                (Node::Empty, ..) => Node::Empty,
                (_, Node::Empty, ..) => i,
                // Either side filtered: evaluated with the exclusion kept.
                _ => Node::Not { include: Box::new(i), exclude: Box::new(e), pre, post },
            }
        }
        "span_containing" | "span_within" => {
            let big = span_clause(o.get("big").ok_or_else(|| parsing(format!("{name} must include [big]")))?, ctx)?;
            let little = span_clause(o.get("little").ok_or_else(|| parsing(format!("{name} must include [little]")))?, ctx)?;
            if name == "span_containing" {
                Node::Filter { src: Box::new(big), rel: Rel::Containing, filter: Box::new(little) }
            } else {
                Node::Filter { src: Box::new(little), rel: Rel::ContainedBy, filter: Box::new(big) }
            }
        }
        "field_masking_span" => {
            let q = o.get("query").ok_or_else(|| parsing("field_masking_span must have [query] span query clause"))?;
            span_clause(q, ctx)?
        }
        other => return Err(parsing(format!("unknown query [{other}]"))),
    })
}

/// Is `name` a span query handled here?
pub fn is_span(name: &str) -> bool {
    matches!(name, "span_term" | "span_multi" | "span_near" | "span_or" | "span_first" | "span_not" | "span_containing" | "span_within" | "field_masking_span")
}

/// A top-level `span_*` query.
pub fn span_query(name: &str, body: &Value, ctx: &QueryContext) -> Result<Query> {
    let boost = body.get("boost").and_then(Value::as_f64).unwrap_or(1.0) as f32;
    let node = span_node(name, body, ctx)?;
    let q = match node.to_span() {
        Some(s) => Query::Span(s),
        None if matches!(node, Node::Empty) => Query::MatchNone,
        None => return Ok(custom(node, Scoring::Span, boost, "span")),
    };
    Ok(if (boost - 1.0).abs() > f32::EPSILON { Query::Boost { query: Box::new(q), boost } } else { q })
}

#[cfg(test)]
mod review_tests {
    use serde_json::{json, Value};

    fn node_with(docs: &[Value]) -> (tempfile::TempDir, std::sync::Arc<crate::node::Node>) {
        let tmp = tempfile::tempdir().unwrap();
        let node = crate::node::Node::open(tmp.path()).unwrap();
        let svc = node.create_index("iv", &json!({"settings": {"number_of_shards": 1}, "mappings": {"properties": {"t": {"type": "text"}}}})).unwrap();
        for (c, chunk) in docs.chunks(5000).enumerate() {
            let reqs = chunk.iter().enumerate().map(|(i, d)| crate::index::WriteReq::Index { id: Some(format!("{}", c * 5000 + i)), source: d.to_string(), routing: None, cond: Default::default() }).collect::<Vec<_>>();
            for r in svc.write(reqs, false) {
                r.unwrap();
            }
        }
        svc.refresh().unwrap();
        (tmp, node)
    }

    fn total(node: &crate::node::Node, q: Value) -> crate::error::Result<u64> {
        let r = crate::ml::search(node, "iv", &json!({"query": q, "track_total_hits": true, "size": 0}))?;
        Ok(r["hits"]["total"]["value"].as_u64().unwrap())
    }

    /// #337: a prefix with more than 128 terms (below the clause limit)
    /// matches every one of them.
    #[test]
    fn prefix_intervals_keep_every_expansion() {
        let docs: Vec<Value> = (0..200).map(|i| json!({"t": format!("pre{i:03} tail")})).collect();
        let (_t, node) = node_with(&docs);
        assert_eq!(total(&node, json!({"intervals": {"t": {"prefix": {"prefix": "pre"}}}})).unwrap(), 200);
        assert_eq!(total(&node, json!({"intervals": {"t": {"wildcard": {"pattern": "pre1*"}}}})).unwrap(), 100);
    }

    /// #335: every alternative is kept (33 at one position), and a product
    /// of groups past the clause limit is refused before it is built.
    #[test]
    fn disjunction_rewrite_is_bounded_and_complete() {
        let (_t, node) = node_with(&[json!({"t": "start w31 end"}), json!({"t": "start x y end"})]);
        let mut alts: Vec<Value> = (0..32).map(|i| json!({"match": {"query": format!("w{i}")}})).collect();
        alts.insert(0, json!({"match": {"query": "x y", "ordered": true}}));
        let q = json!({"intervals": {"t": {"all_of": {"ordered": true, "max_gaps": 0, "intervals": [
            {"match": {"query": "start"}}, {"any_of": {"intervals": alts}}, {"match": {"query": "end"}}]}}}});
        assert_eq!(total(&node, q).unwrap(), 2, "the 33rd alternative and the two-word one both match");
        let group = json!({"any_of": {"intervals": [{"match": {"query": "a"}}, {"match": {"query": "b c", "ordered": true}}]}});
        let q = json!({"intervals": {"t": {"all_of": {"ordered": true, "intervals": vec![group; 24]}}}});
        let started = std::time::Instant::now();
        let e = total(&node, q).unwrap_err();
        assert!(e.reason.contains("Too many disjunctions to expand") || format!("{e:?}").contains("Too many disjunctions to expand"), "{e:?}");
        assert!(started.elapsed() < std::time::Duration::from_secs(30));
    }

    /// #336: the occurrences an intervals query materializes are charged
    /// to the request breaker.
    #[test]
    fn intervals_materialization_is_admitted() {
        let docs: Vec<Value> = (0..20_000).map(|i| json!({"t": format!("common word{i} common")})).collect();
        let (_t, node) = node_with(&docs);
        let q = json!({"intervals": {"t": {"match": {"query": "common"}}}});
        let b = crate::memory::Breakers::leak_for_tests(1 << 31);
        for (limit, ok) in [("256kb", false), ("256mb", true)] {
            b.apply_settings(&|key| match key {
                "indices.breaker.request.limit" => Some(limit.into()),
                "indices.breaker.total.limit" => Some("-1".into()),
                _ => None,
            });
            let scope = crate::memory::RequestScope::new("intervals admission", b);
            let r = crate::memory::scoped(Some(scope), || total(&node, q.clone()));
            match (ok, r) {
                (true, Ok(n)) => assert_eq!(n, 20_000),
                (false, Err(e)) => assert!(crate::memory::is_breaking(&e) || format!("{e:?}").contains("circuit_breaking"), "{e:?}"),
                (ok, r) => panic!("{limit}: expected ok={ok}, got {r:?}"),
            }
        }
    }
}
