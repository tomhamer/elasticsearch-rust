//! `combined_fields` (Lucene `CombinedFieldQuery`, BM25F): the query text is
//! analyzed once; every term is scored as if all the fields were one big
//! field.
//!
//! Per term (and per synonym alternative at one position):
//! * a doc matches when any field has the term;
//! * `freq = sum(field weight * term freq in that field)`;
//! * the doc length is `intToByte4(round(sum(weight * decoded field length)))`
//!   over the fields that have norms (norm byte 1 when none has);
//! * `n` is the largest document frequency over the fields, `N` the largest
//!   field `docCount`, and `avgdl = sum(weight * sumTotalTermFreq) / N`;
//! * scored with the index's default BM25 over those pseudo statistics.
//!
//! Terms combine like `match`: `should` clauses (or `must` for
//! `operator: and`), honouring `minimum_should_match`.

use super::dsl::{analyze_positions, expand_fields, match_opts_from, MatchKind, QueryContext};
use super::synonym::SynonymDocSet;
use crate::error::{EsError, Result};
use rustic_core::query::bm25::{self, Bm25Params, Bm25Weight};
use rustic_core::query::segment_scorer::PostingsDocSet;
use rustic_core::query::{CustomQuery, DocSet, Explanation, LeafReader, MinShouldMatch, Query, Scorer, SearchContext, TERMINATED};
use rustic_core::query::segment_scorer::Norms;
use rustic_core::types::{DocId, Term};
use serde_json::{Map, Value};
use std::sync::Arc;

/// One `CombinedFieldQuery`: the alternatives of one query position over a
/// set of weighted fields.
#[derive(Debug)]
pub struct CombinedFieldQuery {
    /// `(field name, core id, weight)`, sorted by name (Lucene's `TreeMap`).
    fields: Vec<(String, u16, f32)>,
    /// Alternative terms (a single one unless the position has synonyms).
    terms: Vec<String>,
    boost: f32,
    /// A `Query::Term` per (field, term), for the statistics pass.
    pairs: Vec<Query>,
}

/// The pseudo statistics a combined query scores with.
struct Pseudo {
    doc_freq: u64,
    doc_count: u64,
    avg: f32,
}

fn java_float(x: f32) -> String {
    format!("{x:?}")
}

impl CombinedFieldQuery {
    fn pseudo(&self, ctx: &SearchContext) -> Option<Pseudo> {
        let mut doc_freq = 0u64;
        for q in &self.pairs {
            if let Query::Term(t) = q {
                doc_freq = doc_freq.max(ctx.stats(t).doc_freq);
            }
        }
        if doc_freq == 0 {
            return None;
        }
        let mut doc_count = 0u64;
        let mut sum_ttf = 0u64;
        for (_, id, w) in &self.fields {
            let fs = ctx.fstats(*id);
            doc_count = doc_count.max(fs.num_docs);
            // `(long) (acc + (double) weight * sumTotalTermFreq)`, per field.
            sum_ttf = (sum_ttf as f64 + *w as f64 * fs.sum_field_len as f64) as u64;
        }
        let avg = (sum_ttf as f64 / doc_count.max(1) as f64) as f32;
        Some(Pseudo { doc_freq, doc_count, avg })
    }

    fn params(ctx: &SearchContext) -> Bm25Params {
        ctx.similarity.as_bm25().map(|b| b.params()).unwrap_or_default()
    }

    fn weight(&self, ctx: &SearchContext) -> Option<Bm25Weight> {
        let p = self.pseudo(ctx)?;
        Some(Bm25Weight::new(p.doc_freq, p.doc_count, Self::params(ctx), p.avg).with_boost(self.boost))
    }

    /// Norm byte of `doc`: `intToByte4(round(sum(weight * length)))` over the
    /// fields with norms, 1 if no field has norms.
    fn norm_byte(norms: &[(Arc<Norms>, f32)], doc: DocId) -> u8 {
        if norms.is_empty() {
            return 1;
        }
        let mut sum = 0f32;
        for (n, w) in norms {
            sum += *w * bm25::byte4_to_int(n.byte(doc)) as f32;
        }
        // Java `Math.round(float)`: floor(x + 0.5).
        bm25::int_to_byte4((sum + 0.5).floor().max(0.0) as u32)
    }

    fn leaf_norms(&self, leaf: &LeafReader<'_>) -> Vec<(Arc<Norms>, f32)> {
        self.fields
            .iter()
            .filter_map(|(_, id, w)| {
                let n = leaf.norms_arc(*id);
                (!n.is_unit()).then_some((n, *w))
            })
            .collect()
    }

    fn describe(&self) -> String {
        let fields: Vec<String> = self.fields.iter().map(|(n, _, w)| if *w == 1.0 { n.clone() } else { format!("{n}^{}", java_float(*w)) }).collect();
        format!("CombinedFieldQuery(({})({}))", fields.join(" "), self.terms.join(" "))
    }
}

struct CombinedScorer {
    docs: SynonymDocSet,
    weight: Bm25Weight,
    norms: Vec<(Arc<Norms>, f32)>,
}

impl DocSet for CombinedScorer {
    fn advance(&mut self) -> DocId {
        self.docs.advance()
    }
    fn seek(&mut self, target: DocId) -> DocId {
        self.docs.seek(target)
    }
    fn doc(&self) -> DocId {
        self.docs.doc()
    }
    fn size_hint(&self) -> u32 {
        self.docs.size_hint()
    }
}

impl Scorer for CombinedScorer {
    fn score(&mut self) -> f32 {
        let d = self.docs.doc();
        if d == TERMINATED {
            return 0.0;
        }
        let norm = CombinedFieldQuery::norm_byte(&self.norms, d);
        Bm25Weight::combine(self.weight.weight(), self.docs.weighted_freq(), self.weight.norm_cache()[norm as usize])
    }
}

impl CustomQuery for CombinedFieldQuery {
    fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        let weight = self.weight(ctx)?;
        let mut members = Vec::new();
        for (_, id, w) in &self.fields {
            for t in &self.terms {
                if let Some(p) = ctx.postings(leaf, &Term::new_text(*id, t)) {
                    members.push((PostingsDocSet::new(p), *w));
                }
            }
        }
        if members.is_empty() {
            return None;
        }
        Some(Box::new(CombinedScorer { docs: SynonymDocSet::new(members), weight, norms: self.leaf_norms(leaf) }))
    }

    fn children(&self) -> Vec<&Query> {
        self.pairs.iter().collect()
    }

    fn name(&self) -> &'static str {
        "combined_fields"
    }

    fn explain(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, doc: DocId, score: f32) -> Explanation {
        let (Some(p), Some(weight)) = (self.pseudo(ctx), self.weight(ctx)) else { return Explanation::leaf(score, self.describe()) };
        let mut freq = 0f32;
        for (_, id, w) in &self.fields {
            for t in &self.terms {
                if let Some((tf, _)) = leaf.term_freq_at(&Term::new_text(*id, t), doc) {
                    freq += *w * tf as f32;
                }
            }
        }
        if freq == 0.0 {
            return Explanation::leaf(score, self.describe());
        }
        let norm = Self::norm_byte(&self.leaf_norms(leaf), doc);
        let idf_node = Explanation::new(
            weight.idf_value(),
            "idf, computed as ln(1 + (N - n + 0.5) / (n + 0.5))",
            vec![Explanation::leaf(p.doc_freq as f32, "n, number of documents containing term"), Explanation::leaf(p.doc_count as f32, "N, total number of documents")],
        );
        let e = weight.explain_freq(freq, bm25::byte4_to_int(norm), idf_node, format!("termFreq={}", java_float(freq)));
        Explanation::new(e.value, format!("weight({} in {doc}), result of:", self.describe()), vec![e])
    }
}

fn str_list(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
        Some(Value::String(s)) => vec![s.clone()],
        _ => Vec::new(),
    }
}

/// Parse a `combined_fields` query body.
pub fn parse(o: &Map<String, Value>, ctx: &QueryContext) -> Result<Query> {
    for k in o.keys() {
        if !matches!(k.as_str(), "query" | "fields" | "operator" | "minimum_should_match" | "auto_generate_synonyms_phrase_query" | "zero_terms_query" | "boost" | "_name") {
            return Err(EsError::parsing(format!("[combined_fields] query does not support [{k}]")));
        }
    }
    let text = o.get("query").ok_or_else(|| EsError::parsing("Missing [query] in [combined_fields]"))?;
    let specs = str_list(o.get("fields"));
    if specs.is_empty() {
        return Err(EsError::parsing("In [combined_fields] query, at least one field must be provided"));
    }
    let boost = o.get("boost").and_then(Value::as_f64).unwrap_or(1.0) as f32;
    let opts = match_opts_from(o, MatchKind::Bool, ctx)?;
    // (name, core id, weight, search analyzer)
    let mut fields: Vec<(String, u16, f32)> = Vec::new();
    let mut analyzer = opts.analyzer.clone();
    for (name, w) in expand_fields(&specs, ctx) {
        if w < 1.0 {
            return Err(EsError::illegal_argument("In [combined_fields] query, field weights must be greater than or equal to 1.0"));
        }
        let Some(f) = ctx.lookup.get(&name) else { continue };
        if !f.ty.is_text() {
            if specs.iter().any(|s| s.split('^').next() == Some(name.as_str())) {
                return Err(EsError::illegal_argument(format!("Field [{name}] of type [{}] does not support [combined_fields] queries", f.declared.name())));
            }
            continue;
        }
        let Some(id) = f.core_id else { continue };
        if f.similarity != "cosine" {
            return Err(ctx.shard_err_pub("failed to create query: [combined_fields] queries cannot be used with per-field similarities"));
        }
        if analyzer.is_none() {
            analyzer = f.search_analyzer.clone();
        }
        fields.push((name, id, w));
    }
    if fields.is_empty() {
        return Ok(Query::MatchNone);
    }
    if ctx.lookup.registry.sims.default.sim.as_bm25().is_none() {
        return Err(ctx.shard_err_pub("failed to create query: [combined_fields] queries can only be used with the [BM25] similarity"));
    }
    fields.sort_by(|a, b| a.0.cmp(&b.0));
    let analyzer = analyzer.unwrap_or_else(|| Arc::new(rustic_core::analysis::Analyzer::standard()));
    let s = match text {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    let positions = analyze_positions(&analyzer, &s);
    if positions.is_empty() {
        return Ok(if opts.zero_terms_all { Query::MatchAll } else { Query::MatchNone });
    }
    let mut clauses: Vec<Query> = Vec::new();
    for alts in &positions {
        let mut terms = alts.clone();
        terms.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        terms.dedup();
        let mut pairs = Vec::new();
        for (_, id, _) in &fields {
            for t in &terms {
                pairs.push(Query::Term(Term::new_text(*id, t)));
            }
        }
        clauses.push(Query::Custom(Arc::new(CombinedFieldQuery { fields: fields.clone(), terms, boost, pairs })));
    }
    if clauses.len() == 1 {
        return Ok(clauses.pop().unwrap());
    }
    Ok(if opts.operator_and {
        Query::Bool { must: clauses, should: vec![], must_not: vec![], filter: vec![], minimum_should_match: MinShouldMatch::Count(0) }
    } else {
        Query::Bool { must: vec![], should: clauses, must_not: vec![], filter: vec![], minimum_should_match: opts.msm.clone().unwrap_or(MinShouldMatch::Count(1)) }
    })
}
