//! `multi_match` `type: cross_fields` (ES `MultiMatchQueryParser` cross
//! fields mode): fields are grouped by search analyzer; within a group every
//! analyzed term becomes a Lucene `BlendedTermQuery` (dis_max blended): one
//! term query per field whose document frequency is blended across the
//! group's fields, so a term rare in one field but common in another is not
//! over-weighted. Term queries of a group are combined with
//! `operator` / `minimum_should_match`; several groups are combined with
//! `dis_max(tie_breaker)`.

use super::dsl::{analyze_positions, match_field, scalar_string, MatchOpts, QueryContext};
use crate::error::{EsError, Result};
use crate::mapping::EsType;
use rustic_core::analysis::Analyzer;
use rustic_core::query::{CustomQuery, Explanation, LeafReader, MinShouldMatch, Query, Scorer, SearchContext, TermStats};
use rustic_core::types::{DocId, Term};
use serde_json::Value;
use std::sync::Arc;

/// The per-field term queries of one blended term (shared by its fields).
#[derive(Debug)]
struct BlendGroup {
    terms: Vec<Query>,
}

impl BlendGroup {
    fn term(&self, i: usize) -> &Term {
        match &self.terms[i] {
            Query::Term(t) => t,
            _ => unreachable!(),
        }
    }

    /// Lucene `BlendedTermQuery.blend`: the statistics field `idx` scores with.
    fn blended_stats(&self, ctx: &SearchContext, idx: usize) -> TermStats {
        let own = ctx.stats(self.term(idx));
        let n = self.terms.len();
        if n <= 1 {
            return own;
        }
        let stats: Vec<TermStats> = (0..n).map(|i| ctx.stats(self.term(i))).collect();
        let mut max_df = 0u64;
        let mut min_sum_ttf = u64::MAX;
        for (i, s) in stats.iter().enumerate() {
            max_df = max_df.max(s.doc_freq);
            min_sum_ttf = min_sum_ttf.min(ctx.fstats(self.term(i).field).sum_field_len);
        }
        let mut max_doc = stats.iter().map(|s| s.num_docs).max().unwrap_or(0);
        if min_sum_ttf != u64::MAX && max_doc > min_sum_ttf {
            max_doc = min_sum_ttf;
        }
        if max_df == 0 || own.doc_freq == 0 {
            return own;
        }
        // Descending doc freq (stable); bias the rarer fields by +1 per step.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|a, b| stats[*b].doc_freq.cmp(&stats[*a].doc_freq));
        let mut prev = stats[order[0]].doc_freq;
        let mut actual = max_doc.min(max_df);
        for i in order {
            let cur = stats[i].doc_freq;
            if cur == 0 {
                break;
            }
            if prev > cur {
                actual += 1;
            }
            if i == idx {
                let mut s = own;
                s.doc_freq = max_doc.min(actual);
                s.total_term_freq = s.total_term_freq.max(s.doc_freq);
                return s;
            }
            prev = cur;
        }
        own
    }
}

/// One field's term of a blended term: a term query scored with the
/// blended statistics.
#[derive(Debug)]
struct BlendedTerm {
    group: Arc<BlendGroup>,
    idx: usize,
}

impl CustomQuery for BlendedTerm {
    fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        self.scorer_b(leaf, ctx, 1.0)
    }

    fn absorbs_boost(&self) -> bool {
        true
    }

    fn scorer_b(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, boost: f32) -> Option<Box<dyn Scorer>> {
        let t = self.group.term(self.idx);
        let postings = ctx.postings(leaf, t)?;
        leaf.term_scorer_postings(postings, t.field, ctx.sim(t.field), boost, self.group.blended_stats(ctx, self.idx), ctx.fstats(t.field))
    }

    fn children(&self) -> Vec<&Query> {
        self.group.terms.iter().collect()
    }

    fn name(&self) -> &'static str {
        "blended_term"
    }

    fn explain(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, doc: DocId, score: f32) -> Explanation {
        self.explain_b(leaf, ctx, doc, score, 1.0)
    }

    fn explain_b(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, doc: DocId, score: f32, boost: f32) -> Explanation {
        let t = self.group.term(self.idx);
        leaf.explain_term(t, ctx.sim(t.field), boost, self.group.blended_stats(ctx, self.idx), ctx.fstats(t.field), doc)
            .unwrap_or_else(|| Explanation::leaf(score, format!("term {t}")))
    }
}

/// `blended(terms)` over `(field id, boost)`: dis_max of boosted per-field
/// blended term queries.
fn blended(fields: &[(u16, f32)], tokens: &[String], tie: f32) -> Query {
    let mut terms = Vec::new();
    let mut boosts = Vec::new();
    for tok in tokens {
        for (id, b) in fields {
            terms.push(Query::Term(Term::new_text(*id, tok)));
            boosts.push(*b);
        }
    }
    let group = Arc::new(BlendGroup { terms });
    let mut qs: Vec<Query> = (0..boosts.len())
        .map(|i| {
            let q = Query::Custom(Arc::new(BlendedTerm { group: group.clone(), idx: i }));
            if boosts[i] == 1.0 {
                q
            } else {
                Query::Boost { query: Box::new(q), boost: boosts[i] }
            }
        })
        .collect();
    if qs.len() == 1 {
        qs.pop().unwrap()
    } else {
        Query::DisMax { queries: qs, tie_breaker: tie }
    }
}

fn combine(clauses: Vec<Query>, opts: &MatchOpts) -> Query {
    if clauses.len() == 1 {
        return clauses.into_iter().next().unwrap();
    }
    if opts.operator_and {
        Query::Bool { must: clauses, should: vec![], must_not: vec![], filter: vec![], minimum_should_match: MinShouldMatch::Count(0) }
    } else {
        Query::Bool { must: vec![], should: clauses, must_not: vec![], filter: vec![], minimum_should_match: opts.msm.clone().unwrap_or(MinShouldMatch::Count(1)) }
    }
}

/// Build the cross_fields query for `text` over `(field, boost)`.
pub fn build(text: &Value, fields: &[(String, f32)], opts: &MatchOpts, tie: f32, ctx: &QueryContext) -> Result<Query> {
    if opts.fuzziness.is_some() {
        return Err(EsError::parsing("Fuzziness not allowed for type [cross_fields]"));
    }
    // Groups: (key, analyzer, [(core id, boost)]) for analyzed fields; other
    // fields are queried on their own.
    let mut groups: Vec<(String, Arc<Analyzer>, Vec<(u16, f32)>)> = Vec::new();
    let mut singles: Vec<Query> = Vec::new();
    for (name, boost) in fields {
        let Some(f) = ctx.lookup.get(name) else { continue };
        let Some(id) = f.core_id else { continue };
        let (key, an) = if f.ty.is_text() {
            match &opts.analyzer {
                Some(a) => ("_query_analyzer".to_string(), a.clone()),
                None => (
                    format!("text:{}", f.index_analyzer_name.clone().unwrap_or_default()),
                    f.search_analyzer.clone().unwrap_or_else(|| Arc::new(Analyzer::standard())),
                ),
            }
        } else if f.ty == EsType::Keyword && f.normalizer.is_none() && f.declared == EsType::Keyword {
            ("keyword".to_string(), Arc::new(Analyzer::keyword()))
        } else {
            let mut o2 = opts.clone();
            o2.lenient = true;
            match match_field(name, text, &o2, ctx) {
                Ok(Query::MatchNone) | Err(_) => {}
                Ok(q) => singles.push(if *boost == 1.0 { q } else { Query::Boost { query: Box::new(q), boost: *boost } }),
            }
            continue;
        };
        match groups.iter_mut().find(|g| g.0 == key) {
            Some(g) => g.2.push((id, *boost)),
            None => groups.push((key, an, vec![(id, *boost)])),
        }
    }
    let s = scalar_string(text);
    let mut out: Vec<Query> = Vec::new();
    let mut any_terms = false;
    for (_, an, ids) in &groups {
        let positions = analyze_positions(an, &s);
        if positions.is_empty() {
            continue;
        }
        any_terms = true;
        let clauses: Vec<Query> = positions.iter().map(|alts| blended(ids, alts, tie)).collect();
        out.push(combine(clauses, opts));
    }
    out.extend(singles);
    if out.is_empty() {
        return Ok(if !any_terms && opts.zero_terms_all { Query::MatchAll } else { Query::MatchNone });
    }
    Ok(if out.len() == 1 { out.pop().unwrap() } else { Query::DisMax { queries: out, tie_breaker: tie } })
}
