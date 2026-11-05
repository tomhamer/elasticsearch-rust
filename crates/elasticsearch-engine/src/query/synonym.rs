//! Lucene `SynonymQuery`: several terms of one field that stand for the same
//! position (synonyms, stacked shingles, `word_delimiter_graph` parts)
//! matched and scored as ONE pseudo term.
//!
//! * a doc matches when it has any of the terms;
//! * the term frequency is the sum of the terms' frequencies in the doc;
//! * the statistics are `docFreq = max(docFreq)` and
//!   `totalTermFreq = sum(totalTermFreq)` over the terms, with the field's
//!   collection statistics;
//! * explanations read `weight(Synonym(f:a f:b) in doc) [...]` and label the
//!   frequency `termFreq=<n>`.

use rustic_core::query::segment_scorer::PostingsDocSet;
use rustic_core::query::{CustomQuery, DocSet, Explanation, LeafReader, Query, Scorer, SearchContext, TermFreqSource, TermScorer, TermStats, TERMINATED};
use rustic_core::types::{DocId, Term};
use std::sync::Arc;

/// Docs of any of several postings lists, each with a weight; the frequency
/// of a doc is the weighted sum of the lists' frequencies (all weights 1 for
/// a synonym query; field weights for `combined_fields`).
pub(crate) struct SynonymDocSet {
    members: Vec<(PostingsDocSet, f32)>,
    current: DocId,
    primed: bool,
}

impl SynonymDocSet {
    pub(crate) fn new(members: Vec<(PostingsDocSet, f32)>) -> Self {
        Self { members, current: TERMINATED, primed: false }
    }

    fn min_doc(&self) -> DocId {
        self.members.iter().map(|m| m.0.doc()).min().unwrap_or(TERMINATED)
    }

    /// `sum(weight * freq)` over the lists positioned on the current doc.
    pub(crate) fn weighted_freq(&self) -> f32 {
        let mut f = 0f32;
        for (m, w) in &self.members {
            if m.doc() == self.current {
                f += *w * TermFreqSource::term_freq(m) as f32;
            }
        }
        f
    }
}

impl DocSet for SynonymDocSet {
    fn advance(&mut self) -> DocId {
        if !self.primed {
            self.primed = true;
            for m in &mut self.members {
                m.0.advance();
            }
        } else {
            let d = self.current;
            for m in &mut self.members {
                if m.0.doc() == d {
                    m.0.advance();
                }
            }
        }
        self.current = self.min_doc();
        self.current
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.primed && self.current != TERMINATED && self.current >= target {
            return self.current;
        }
        self.primed = true;
        for m in &mut self.members {
            m.0.seek(target);
        }
        self.current = self.min_doc();
        self.current
    }

    fn doc(&self) -> DocId {
        self.current
    }

    fn size_hint(&self) -> u32 {
        self.members.iter().map(|m| m.0.size_hint()).sum()
    }
}

impl TermFreqSource for SynonymDocSet {
    fn term_freq(&self) -> u32 {
        self.weighted_freq() as u32
    }
}

/// `Synonym(field:a field:b ...)`.
#[derive(Debug)]
pub struct SynonymQuery {
    field: u16,
    /// One `Query::Term` per alternative, sorted by term bytes (Lucene sorts).
    terms: Vec<Query>,
}

impl SynonymQuery {
    fn term(&self, i: usize) -> &Term {
        match &self.terms[i] {
            Query::Term(t) => t,
            _ => unreachable!(),
        }
    }

    /// The pseudo term's statistics: max docFreq, summed totalTermFreq, the
    /// field's docCount as `N`.
    fn stats(&self, ctx: &SearchContext) -> TermStats {
        let mut s = TermStats::default();
        for i in 0..self.terms.len() {
            let t = ctx.stats(self.term(i));
            if t.doc_freq > 0 {
                s.doc_freq = s.doc_freq.max(t.doc_freq);
                s.total_term_freq += t.total_term_freq;
            }
            s.num_docs = s.num_docs.max(t.num_docs);
        }
        s
    }

    fn describe(&self) -> String {
        let parts: Vec<String> = (0..self.terms.len()).map(|i| self.term(i).to_string()).collect();
        format!("Synonym({})", parts.join(" "))
    }
}

/// A query matching any of `alts` (terms of field `field`) as one pseudo term.
pub fn query(field: u16, alts: &[String]) -> Query {
    let mut texts: Vec<&String> = alts.iter().collect();
    texts.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    texts.dedup();
    if texts.len() == 1 {
        return Query::Term(Term::new_text(field, texts[0]));
    }
    let terms = texts.into_iter().map(|t| Query::Term(Term::new_text(field, t))).collect();
    Query::Custom(Arc::new(SynonymQuery { field, terms }))
}

impl CustomQuery for SynonymQuery {
    fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        self.scorer_b(leaf, ctx, 1.0)
    }

    fn absorbs_boost(&self) -> bool {
        true
    }

    fn scorer_b(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, boost: f32) -> Option<Box<dyn Scorer>> {
        let mut members = Vec::new();
        for i in 0..self.terms.len() {
            if let Some(p) = ctx.postings(leaf, self.term(i)) {
                members.push((PostingsDocSet::new(p), 1.0));
            }
        }
        if members.is_empty() {
            return None;
        }
        let mut fs = ctx.fstats(self.field);
        if fs.num_docs == 0 {
            fs = leaf.field_stats(self.field);
        }
        let weight = ctx.sim(self.field).weight(self.stats(ctx), fs, boost);
        Some(Box::new(TermScorer::with_norms(SynonymDocSet::new(members), weight, leaf.norms_arc(self.field))))
    }

    fn children(&self) -> Vec<&Query> {
        self.terms.iter().collect()
    }

    fn name(&self) -> &'static str {
        "synonym"
    }

    fn explain(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, doc: DocId, score: f32) -> Explanation {
        self.explain_b(leaf, ctx, doc, score, 1.0)
    }

    fn explain_b(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, doc: DocId, score: f32, boost: f32) -> Explanation {
        let mut freq = 0u32;
        for i in 0..self.terms.len() {
            if let Some((tf, _)) = leaf.term_freq_at(self.term(i), doc) {
                freq += tf;
            }
        }
        if freq == 0 {
            return Explanation::leaf(score, self.describe());
        }
        let mut fs = ctx.fstats(self.field);
        if fs.num_docs == 0 {
            fs = leaf.field_stats(self.field);
        }
        let len = leaf.norms(self.field).len(doc);
        let sim = ctx.sim(self.field);
        let mut e = sim.weight(self.stats(ctx), fs, boost).explain(freq, len);
        relabel_freq(&mut e, &format!("termFreq={:?}", freq as f32));
        Explanation::new(e.value, format!("weight({} in {doc}) [{}]", self.describe(), sim.name()), vec![e])
    }
}

/// Replace the `freq, occurrences of term within document` leaf's wording.
pub(crate) fn relabel_freq(e: &mut Explanation, label: &str) {
    if e.description.starts_with("freq, occurrences of term within document") {
        e.description = label.to_string();
    }
    for d in &mut e.details {
        relabel_freq(d, label);
    }
}
