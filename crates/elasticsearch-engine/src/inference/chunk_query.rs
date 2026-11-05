//! Queries over a `semantic_text` field's chunk embeddings (ES runs them
//! as `nested` queries over the `<field>.inference.chunks` child docs:
//! `sparse_vector` / `knn` on `<field>.inference.chunks.embeddings`).
//!
//! The chunk embeddings live in the field's inference blob
//! ([`super::semantic`]); a doc matches when one of its chunks does, scored
//! by the chunk scores combined with the nested `score_mode`. Scoring is
//! brute force over the leaf's docs.

use rustic_core::nested::NestedScoreMode;
use rustic_core::query::query::{CustomQuery, ScoredDocs, SearchContext};
use rustic_core::query::{Explanation, LeafReader, Scorer};
use rustic_core::types::DocId;
use rustic_core::vector::Similarity;
use serde_json::Value;
use std::sync::Arc;

/// What a chunk is scored against.
#[derive(Clone, Debug)]
pub enum ChunkVec {
    /// kNN: the query vector, the field's similarity and an optional
    /// `similarity` threshold in ES score space.
    Dense { v: Arc<Vec<f32>>, sim: Similarity, threshold: Option<f32> },
    /// `sparse_vector`: weighted tokens (Lucene `FeatureField` linear scoring).
    Sparse(Arc<Vec<(String, f32)>>),
    /// Every chunk matches with score 1 (`match_all` in a nested query).
    All,
}

#[derive(Debug)]
pub struct ChunkQuery {
    /// The `semantic_text` field.
    pub field: String,
    /// Its inference blob doc-values id.
    pub aux: u16,
    pub vec: ChunkVec,
    pub mode: NestedScoreMode,
    /// kNN: only the `k` best docs (per leaf) match.
    pub k: Option<usize>,
    /// kNN pre-filter on the parent docs (applied before the top `k`).
    pub filter: Option<Box<rustic_core::query::Query>>,
    /// The shard-wide top `k` (kNN), computed up front: leaf id -> hits.
    pub fixed: Option<std::collections::HashMap<usize, Vec<(DocId, f32)>>>,
}

/// Lucene `FeatureField`: a stored weight keeps 9 significant bits.
fn feature_value(w: f64) -> f32 {
    f32::from_bits(((w as f32).to_bits() >> 15) << 15)
}

impl ChunkQuery {
    /// Per chunk of `doc` (blob order): its score when the chunk matches.
    pub fn chunk_scores(&self, leaf: &LeafReader<'_>, doc: DocId) -> Vec<Option<f32>> {
        let Some(b) = super::semantic::blob_of(leaf, self.aux, doc) else { return Vec::new() };
        self.blob_scores(&b)
    }

    fn blob_scores(&self, b: &Value) -> Vec<Option<f32>> {
        let vecs = b["v"].as_array().map(Vec::as_slice).unwrap_or(&[]);
        match &self.vec {
            ChunkVec::All => vecs.iter().map(|_| Some(1.0)).collect(),
            ChunkVec::Dense { v, sim, threshold } => {
                if b["k"] != "d" {
                    return vecs.iter().map(|_| None).collect();
                }
                vecs.iter()
                    .map(|c| {
                        let d: Vec<f32> = c.as_array().map(|a| a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect()).unwrap_or_default();
                        if d.len() != v.len() {
                            return None;
                        }
                        let s = sim.score(v, &d);
                        threshold.is_none_or(|t| s >= t).then_some(s)
                    })
                    .collect()
            }
            ChunkVec::Sparse(q) => {
                if b["k"] != "s" {
                    return vecs.iter().map(|_| None).collect();
                }
                vecs.iter()
                    .map(|c| {
                        // Lucene sums the clause scores in a double.
                        let mut s = 0f64;
                        let mut hit = false;
                        for (t, qw) in q.iter() {
                            if let Some(w) = c.get(t).and_then(Value::as_f64) {
                                s += (qw * feature_value(w)) as f64;
                                hit = true;
                            }
                        }
                        hit.then_some(s as f32)
                    })
                    .collect()
            }
        }
    }

    fn combine(&self, scores: &[Option<f32>]) -> Option<f32> {
        let hits: Vec<f32> = scores.iter().flatten().copied().collect();
        if hits.is_empty() {
            return None;
        }
        // kNN in a nested query is diversified: one (the best) child per parent.
        if matches!(self.vec, ChunkVec::Dense { .. }) {
            return hits.iter().copied().reduce(f32::max);
        }
        let sum: f32 = hits.iter().sum();
        let min = hits.iter().copied().fold(f32::INFINITY, f32::min);
        let max = hits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        Some(self.mode.combine(sum, min, max, hits.len() as u32))
    }

    /// The doc's score (`None` = no matching chunk).
    pub fn doc_score(&self, leaf: &LeafReader<'_>, doc: DocId) -> Option<f32> {
        self.combine(&self.chunk_scores(leaf, doc))
    }
}

impl CustomQuery for ChunkQuery {
    fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        if let Some(f) = &self.fixed {
            let v = f.get(&leaf.id()).filter(|v| !v.is_empty())?;
            return Some(Box::new(ScoredDocs::new(v.clone())));
        }
        let hits = self.leaf_hits(leaf, ctx);
        if hits.is_empty() {
            return None;
        }
        Some(Box::new(ScoredDocs::new(hits)))
    }
    fn name(&self) -> &'static str {
        "semantic"
    }
    fn children(&self) -> Vec<&rustic_core::query::Query> {
        self.filter.iter().map(|b| b.as_ref()).collect()
    }
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn explain(&self, _leaf: &LeafReader<'_>, _ctx: &SearchContext, _doc: DocId, score: f32) -> Explanation {
        let mode = match self.mode {
            NestedScoreMode::Avg => "Avg",
            NestedScoreMode::Max => "Max",
            NestedScoreMode::Min => "Min",
            NestedScoreMode::Sum => "Total",
            NestedScoreMode::None => "None",
        };
        Explanation::leaf(score, format!("Score based on {} child docs in range from 0 to 0, using score mode {mode}", 1))
    }
}

impl ChunkQuery {
    /// Matching docs of one leaf (pre-filter applied, before any top `k`).
    fn leaf_hits(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Vec<(DocId, f32)> {
        let live = leaf.live_docs();
        let allowed: Option<std::collections::HashSet<DocId>> = self.filter.as_ref().map(|f| {
            let mut set = std::collections::HashSet::new();
            if let Some(mut s) = f.scorer(leaf, ctx) {
                loop {
                    let d = s.advance();
                    if d == rustic_core::query::TERMINATED {
                        break;
                    }
                    set.insert(d);
                }
            }
            set
        });
        let mut hits: Vec<(DocId, f32)> = Vec::new();
        for d in 0..leaf.num_docs() {
            if !leaf.is_root(d) || !live(d) || allowed.as_ref().is_some_and(|a| !a.contains(&d)) {
                continue;
            }
            if let Some(s) = self.doc_score(leaf, d) {
                hits.push((d, s));
            }
        }
        hits
    }

    /// Resolve the shard-wide top `k` over `leaves` (kNN semantics).
    pub fn with_top_k(mut self, leaves: &[LeafReader<'_>]) -> crate::error::Result<Self> {
        let Some(k) = self.k else { return Ok(self) };
        let ctx = match &self.filter {
            Some(f) => SearchContext::for_query(f, leaves, rustic_core::query::Bm25Params::default()),
            None => SearchContext::for_query(&rustic_core::query::Query::MatchAll, leaves, rustic_core::query::Bm25Params::default()),
        };
        ctx.execution_result()?;
        let mut all: Vec<(usize, DocId, f32)> = Vec::new();
        for (li, leaf) in leaves.iter().enumerate() {
            for (d, s) in self.leaf_hits(leaf, &ctx) {
                all.push((li, d, s));
            }
        }
        ctx.execution_result()?;
        all.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
        all.truncate(k);
        let mut fixed: std::collections::HashMap<usize, Vec<(DocId, f32)>> = std::collections::HashMap::new();
        for (li, d, s) in all {
            fixed.entry(leaves[li].id()).or_default().push((d, s));
        }
        for v in fixed.values_mut() {
            v.sort_by_key(|h| h.0);
        }
        self.fixed = Some(fixed);
        Ok(self)
    }
}

/// The chunk queries of `q` on `field` (highlighting).
pub fn find<'a>(q: &'a rustic_core::query::Query, field: &str, out: &mut Vec<&'a ChunkQuery>) {
    use rustic_core::query::Query;
    match q {
        Query::Custom(c) => {
            if let Some(cq) = c.as_any().and_then(|a| a.downcast_ref::<ChunkQuery>()) {
                if cq.field == field {
                    out.push(cq);
                }
            }
            for ch in c.children() {
                find(ch, field, out);
            }
        }
        Query::Bool { must, should, must_not: _, filter, .. } => {
            for x in must.iter().chain(should).chain(filter) {
                find(x, field, out);
            }
        }
        Query::Boost { query, .. } | Query::ConstScore { query, .. } | Query::FunctionScore { query, .. } | Query::Nested { query, .. } => find(query, field, out),
        Query::DisMax { queries, .. } => {
            for x in queries {
                find(x, field, out);
            }
        }
        _ => {}
    }
}

/// `WeightedTokensUtils.queryBuilderWithPrunedTokens` over a field's chunk
/// embeddings (the `<field>.inference.chunks.embeddings` sparse_vector):
/// chunk-level doc counts and frequencies, read from the inference blobs.
/// `None` when no chunk holds a token.
pub fn prune_tokens(leaves: &[LeafReader<'_>], aux: u16, tokens: Vec<(String, f32)>, freq_ratio: f32, weight: f32, only_pruned: bool) -> Option<Vec<(String, f32)>> {
    let mut doc_count = 0u64;
    let mut sum_doc_freq = 0u64;
    let mut max_unique = 0usize;
    let mut dfs: std::collections::HashMap<&str, u64> = tokens.iter().map(|(t, _)| (t.as_str(), 0)).collect();
    for leaf in leaves {
        let mut unique: std::collections::HashSet<String> = std::collections::HashSet::new();
        for d in 0..leaf.num_docs() {
            if !leaf.is_root(d) {
                continue;
            }
            let Some(b) = super::semantic::blob_of(leaf, aux, d) else { continue };
            if b["k"] != "s" {
                continue;
            }
            for c in b["v"].as_array().into_iter().flatten() {
                let Some(o) = c.as_object() else { continue };
                if o.is_empty() {
                    continue;
                }
                doc_count += 1;
                sum_doc_freq += o.len() as u64;
                for k in o.keys() {
                    if let Some(df) = dfs.get_mut(k.as_str()) {
                        *df += 1;
                    }
                    if !unique.contains(k) {
                        unique.insert(k.clone());
                    }
                }
            }
        }
        max_unique = max_unique.max(unique.len());
    }
    if max_unique == 0 || doc_count == 0 {
        return None;
    }
    let avg = sum_doc_freq as f32 / doc_count as f32 / max_unique as f32;
    let best = tokens.iter().map(|(_, w)| *w).fold(0f32, f32::max);
    Some(
        tokens
            .iter()
            .filter(|(t, w)| {
                let df = dfs.get(t.as_str()).copied().unwrap_or(0);
                let keep = df > 0 && ((df as f32 / doc_count as f32) < freq_ratio * avg || *w > weight * best);
                keep ^ only_pruned
            })
            .cloned()
            .collect(),
    )
}
