//! [`FilterQuery`]: a query node that takes the docs of an optional
//! candidate query (an approximation, e.g. the 3-gram terms of a wildcard
//! pattern) and keeps those a per-doc check accepts (verification against
//! doc values or stored fields). The check may also produce the doc's
//! score (feature queries, `has_child` score modes).
//!
//! Lucene equivalent: a `TwoPhaseIterator` over the approximation's
//! `DocIdSetIterator`.

use crate::query::query::ScoredDocs;
use crate::query::{CustomQuery, LeafReader, Query, Scorer, SearchContext, TERMINATED};
use crate::types::DocId;
use std::sync::Arc;

/// Per-doc check: `None` = no match, `Some(score)` = match.
pub type DocCheck = Arc<dyn Fn(&LeafReader<'_>, DocId) -> Option<f32> + Send + Sync>;

/// Candidates + verification.
pub struct FilterQuery {
    name: &'static str,
    candidates: Option<Query>,
    check: DocCheck,
    constant: bool,
}

impl std::fmt::Debug for FilterQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilterQuery").field("name", &self.name).field("candidates", &self.candidates).finish()
    }
}

impl FilterQuery {
    /// Docs of `candidates` (every doc of the leaf when `None`) accepted by
    /// `check`, scored by what `check` returns.
    pub fn new(name: &'static str, candidates: Option<Query>, check: DocCheck) -> Self {
        Self { name, candidates, check, constant: false }
    }

    /// A constant-score filter: `pred` accepts, every match scores 1.
    pub fn constant(name: &'static str, candidates: Option<Query>, pred: impl Fn(&LeafReader<'_>, DocId) -> bool + Send + Sync + 'static) -> Self {
        Self { name, candidates, check: Arc::new(move |l, d| pred(l, d).then_some(1.0)), constant: true }
    }

    /// Wrap as a [`Query`].
    pub fn into_query(self) -> Query {
        Query::Custom(Arc::new(self))
    }
}

impl CustomQuery for FilterQuery {
    fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        let mut out: Vec<(DocId, f32)> = Vec::new();
        match &self.candidates {
            Some(q) => {
                let mut s = q.scorer(leaf, ctx)?;
                loop {
                    let d = s.advance();
                    if d == TERMINATED {
                        break;
                    }
                    if let Some(score) = (self.check)(leaf, d) {
                        out.push((d, score));
                    }
                }
            }
            None => {
                for d in 0..leaf.num_docs() {
                    if leaf.is_deleted(d) {
                        continue;
                    }
                    if let Some(score) = (self.check)(leaf, d) {
                        out.push((d, score));
                    }
                }
            }
        }
        if out.is_empty() {
            return None;
        }
        Some(Box::new(ScoredDocs::new(out)))
    }

    fn children(&self) -> Vec<&Query> {
        self.candidates.iter().collect()
    }

    fn is_constant_score(&self) -> bool {
        self.constant
    }

    fn name(&self) -> &'static str {
        self.name
    }
}
