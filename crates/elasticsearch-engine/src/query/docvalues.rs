//! Queries on fields that are not indexed but have doc values
//! (`index: false`): ES answers term-level queries on such keyword, ip and
//! boolean fields by scanning doc values (Lucene's slow doc-values queries).
//! [`rewrite`] turns the core leaves built for the field into such scans.

use super::dsl::QueryContext;
use crate::mapping::EsType;
use rustic_core::automaton::levenshtein::LevenshteinMatcher;
use rustic_core::automaton::regexp::{RegexpFlags, RegexpMatcher};
use rustic_core::automaton::wildcard::WildcardMatcher;
use rustic_core::automaton::TermMatcher;
use rustic_core::query::explain::Explanation;
use rustic_core::query::{Bound, CustomQuery, Fuzziness, LeafReader, Query, Scorer, SearchContext};
use rustic_core::types::DocId;
use serde_json::Value;
use std::sync::Arc;

/// The field a leaf query's body targets.
fn field_of<'v>(name: &str, body: &'v Value) -> Option<&'v str> {
    let o = body.as_object()?;
    if name == "exists" {
        return o.get("field").and_then(Value::as_str);
    }
    o.keys().find(|k| !matches!(k.as_str(), "boost" | "_name")).map(String::as_str)
}

/// Doc-values fallback for leaf query `name` on a non-indexed field.
pub fn rewrite(name: &str, body: &Value, q: Query, ctx: &QueryContext) -> Query {
    // Query parsers building term-level leaves over many fields
    // (`query_string`'s `+dim:value`): each doc-values-only keyword, ip or
    // boolean field they target is scanned. A time series index's
    // dimensions are doc values only (ES 9.5 `index: false` with skippers).
    if matches!(name, "query_string" | "simple_query_string" | "multi_match" | "combined_fields" | "kql") {
        return rewrite_fields(q, ctx);
    }
    if !matches!(name, "term" | "terms" | "prefix" | "wildcard" | "regexp" | "fuzzy" | "range" | "match" | "match_phrase" | "match_bool_prefix" | "match_phrase_prefix") {
        return q;
    }
    let Some(field) = field_of(name, body) else { return q };
    let Some(f) = ctx.lookup.get(field) else { return q };
    if f.indexed || !f.doc_values {
        return q;
    }
    let Some(id) = f.core_id else { return q };
    let (kind, scan) = match f.declared {
        // Text doc values hold whole values (`#dv`); only the multi-term
        // queries run on them.
        EsType::Text | EsType::MatchOnlyText => {
            if !matches!(name, "prefix" | "wildcard" | "regexp") {
                return q;
            }
            match f.aux_id("dv") {
                Some(dv) => (Kind::Keyword, dv),
                None => return q,
            }
        }
        _ => match f.ty {
            EsType::Keyword | EsType::Ip | EsType::Wildcard => (Kind::Keyword, id),
            EsType::Boolean => (Kind::Bool, id),
            _ => return q,
        },
    };
    convert(q, id, scan, kind)
}

/// [`rewrite`] for every doc-values-only keyword, ip or boolean field the
/// term-level leaves of `q` target.
fn rewrite_fields(mut q: Query, ctx: &QueryContext) -> Query {
    fn fields(q: &Query, out: &mut Vec<u16>) {
        match q {
            Query::Term(t) => out.push(t.field),
            Query::TermSet { field, .. } | Query::Prefix { field, .. } | Query::Wildcard { field, .. } | Query::Regexp { field, .. }
            | Query::Fuzzy { field, .. } | Query::TermRange { field, .. } => out.push(*field),
            Query::ConstScore { query, .. } | Query::Boost { query, .. } => fields(query, out),
            Query::Bool { must, should, must_not, filter, .. } => must.iter().chain(should).chain(must_not).chain(filter).for_each(|x| fields(x, out)),
            Query::DisMax { queries, .. } => queries.iter().for_each(|x| fields(x, out)),
            _ => {}
        }
    }
    let mut ids = Vec::new();
    fields(&q, &mut ids);
    ids.sort_unstable();
    ids.dedup();
    if ids.is_empty() {
        return q;
    }
    for f in ctx.lookup.fields.values() {
        let Some(id) = f.core_id.filter(|id| ids.binary_search(id).is_ok()) else { continue };
        if f.indexed || !f.doc_values || matches!(f.declared, EsType::Text | EsType::MatchOnlyText) {
            continue;
        }
        let kind = match f.ty {
            EsType::Keyword | EsType::Ip | EsType::Wildcard => Kind::Keyword,
            EsType::Boolean => Kind::Bool,
            _ => continue,
        };
        q = convert(q, id, id, kind);
    }
    q
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Keyword,
    Bool,
}

fn convert(q: Query, id: u16, scan: u16, kind: Kind) -> Query {
    let pred: Pred = match q {
        Query::Term(t) if t.field == id => Pred::Exact(vec![t.text]),
        Query::TermSet { field, terms, boost } if field == id => return wrap(Pred::Exact(terms), scan, kind, boost),
        Query::Prefix { field, prefix, case_insensitive, .. } if field == id => Pred::Prefix(prefix, case_insensitive),
        Query::Wildcard { field, pattern, case_insensitive, .. } if field == id => {
            Pred::Matcher(Arc::new(WildcardMatcher::new(&if case_insensitive { pattern.to_lowercase() } else { pattern })), case_insensitive)
        }
        Query::Regexp { field, pattern, case_insensitive, .. } if field == id => match RegexpMatcher::new(&pattern, RegexpFlags { case_insensitive }) {
            Ok(m) => Pred::Matcher(Arc::new(m), false),
            Err(_) => return Query::MatchNone,
        },
        Query::Fuzzy { field, text, fuzziness, prefix_length, transpositions, .. } if field == id => {
            let n = text.chars().count();
            let edits = match fuzziness {
                Fuzziness::Edits(e) => e.min(2),
                Fuzziness::Auto => if n < 3 { 0 } else if n < 6 { 1 } else { 2 },
                Fuzziness::AutoLow(a, b) => if n < a { 0 } else if n < b { 1 } else { 2 },
            };
            Pred::Matcher(Arc::new(LevenshteinMatcher::new(&text, edits, prefix_length, transpositions)), false)
        }
        Query::TermRange { field, lower, upper, .. } if field == id => Pred::Range(lower, upper),
        Query::ConstScore { query, score, point_set } => return Query::ConstScore { query: Box::new(convert(*query, id, scan, kind)), score, point_set },
        Query::Boost { query, boost } => return Query::Boost { query: Box::new(convert(*query, id, scan, kind)), boost },
        Query::Bool { must, should, must_not, filter, minimum_should_match } => {
            let c = |v: Vec<Query>| v.into_iter().map(|x| convert(x, id, scan, kind)).collect();
            return Query::Bool { must: c(must), should: c(should), must_not: c(must_not), filter: c(filter), minimum_should_match };
        }
        Query::DisMax { queries, tie_breaker } => return Query::DisMax { queries: queries.into_iter().map(|x| convert(x, id, scan, kind)).collect(), tie_breaker },
        other => return other,
    };
    wrap(pred, scan, kind, 1.0)
}

fn wrap(pred: Pred, field: u16, kind: Kind, score: f32) -> Query {
    Query::Custom(Arc::new(DocValuesQuery { field, kind, pred, score }))
}

#[derive(Clone)]
enum Pred {
    Exact(Vec<Vec<u8>>),
    Prefix(Vec<u8>, bool),
    Matcher(Arc<dyn TermMatcher>, bool),
    Range(Bound<Vec<u8>>, Bound<Vec<u8>>),
}

impl std::fmt::Debug for Pred {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Pred::Exact(v) => write!(f, "Exact({v:?})"),
            Pred::Prefix(p, ci) => write!(f, "Prefix({p:?}, {ci})"),
            Pred::Matcher(_, _) => write!(f, "Matcher"),
            Pred::Range(a, b) => write!(f, "Range({a:?}, {b:?})"),
        }
    }
}

impl Pred {
    fn test(&self, v: &[u8]) -> bool {
        match self {
            Pred::Exact(ts) => ts.iter().any(|t| t.as_slice() == v),
            Pred::Prefix(p, ci) => {
                if *ci {
                    String::from_utf8_lossy(v).to_lowercase().starts_with(&String::from_utf8_lossy(p).to_lowercase())
                } else {
                    v.starts_with(p)
                }
            }
            Pred::Matcher(m, lower) => {
                if *lower {
                    m.matches(String::from_utf8_lossy(v).to_lowercase().as_bytes())
                } else {
                    m.matches(v)
                }
            }
            Pred::Range(lo, hi) => {
                let ok_lo = match lo {
                    Bound::Included(b) => v >= b.as_slice(),
                    Bound::Excluded(b) => v > b.as_slice(),
                    Bound::Unbounded => true,
                };
                let ok_hi = match hi {
                    Bound::Included(b) => v <= b.as_slice(),
                    Bound::Excluded(b) => v < b.as_slice(),
                    Bound::Unbounded => true,
                };
                ok_lo && ok_hi
            }
        }
    }
}

#[derive(Debug)]
struct DocValuesQuery {
    field: u16,
    kind: Kind,
    pred: Pred,
    score: f32,
}

impl CustomQuery for DocValuesQuery {
    fn scorer(&self, leaf: &LeafReader<'_>, _ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        let mut docs = Vec::new();
        for d in 0..leaf.num_docs() {
            if leaf.is_deleted(d) {
                continue;
            }
            let hit = match self.kind {
                Kind::Keyword => leaf.keyword_values(self.field, d).iter().any(|v| self.pred.test(v.as_bytes())),
                Kind::Bool => leaf.bool_value(self.field, d).map(|b| self.pred.test(&[b as u8])).unwrap_or(false),
            };
            if hit {
                docs.push((d, self.score));
            }
        }
        if docs.is_empty() {
            return None;
        }
        Some(Box::new(rustic_core::query::query::ScoredDocs::new(docs)))
    }
    fn is_constant_score(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "doc_values"
    }
    fn explain(&self, _leaf: &LeafReader<'_>, _ctx: &SearchContext, _doc: DocId, score: f32) -> Explanation {
        Explanation::leaf(score, "doc values query match".to_string())
    }
}
