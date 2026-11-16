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
    /// kNN: the query vector, the field's similarity, the model's element
    /// type and an optional `similarity` threshold in ES score space.
    Dense { v: Arc<Vec<f32>>, sim: Similarity, elem: ChunkElem, threshold: Option<f32> },
    /// `sparse_vector`: weighted tokens (Lucene `FeatureField` linear scoring).
    Sparse(Arc<Vec<(String, f32)>>),
    /// Every chunk matches with score 1 (`match_all` in a nested query).
    All,
}

/// The element type of a dense model's embeddings (its `model_settings`
/// `element_type`): `byte` and `bit` embeddings are stored as their byte
/// values and score with ES's byte / Hamming formulas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkElem {
    Float,
    Byte,
    /// Packed bits, one stored value (a signed byte) per 8 dimensions.
    Bit,
}

impl ChunkElem {
    pub fn of(element_type: Option<&str>) -> Self {
        match element_type {
            Some("byte") => ChunkElem::Byte,
            Some("bit") => ChunkElem::Bit,
            _ => ChunkElem::Float,
        }
    }

    /// The `ElementType` the ES score conversions take.
    pub fn es(self) -> crate::vectors::ElementType {
        match self {
            ChunkElem::Float => crate::vectors::ElementType::Float,
            ChunkElem::Byte => crate::vectors::ElementType::Byte,
            ChunkElem::Bit => crate::vectors::ElementType::Bit,
        }
    }

    /// ES's score of a chunk `d` against query `v` (same lengths): `byte`
    /// `dot_product` is `0.5 + dot / (dims * 2^15)`, `bit` is
    /// `(bits - hamming) / bits`; the others are the float formulas.
    pub fn score(self, sim: Similarity, v: &[f32], d: &[f32]) -> f32 {
        match self {
            ChunkElem::Bit => {
                let byte = |x: f32| x as i32 as u8;
                let h: u32 = v.iter().zip(d).map(|(a, b)| (byte(*a) ^ byte(*b)).count_ones()).sum();
                let bits = (v.len() * 8) as f32;
                (bits - h as f32) / bits
            }
            ChunkElem::Byte if sim == Similarity::DotProduct => {
                let dot: f32 = v.iter().zip(d).map(|(a, b)| a * b).sum();
                0.5 + dot / (v.len() as f32 * 32768.0)
            }
            _ => sim.score(v, d),
        }
    }
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

/// A probe called with each root document the scan scores (tests count
/// the documents a stopped scan still visits).
static SCAN_PROBE: parking_lot::RwLock<Option<Arc<dyn Fn() + Send + Sync>>> = parking_lot::RwLock::new(None);
static PROBE_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[doc(hidden)]
pub fn set_scan_probe(probe: Option<Arc<dyn Fn() + Send + Sync>>) {
    PROBE_ON.store(probe.is_some(), std::sync::atomic::Ordering::Relaxed);
    *SCAN_PROBE.write() = probe;
}

/// Chunks scored between deadline / cancellation checks within one doc.
const CHUNK_POLL: usize = 256;

/// One candidate of the shard-wide top `k`, ordered worst first (a lower
/// score, then a later leaf, then a later doc is worse), so a max-heap of
/// them keeps the `k` best at the cost of `k` entries.
struct Candidate {
    score: f32,
    leaf: usize,
    doc: DocId,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.score.total_cmp(&self.score).then(self.leaf.cmp(&other.leaf)).then(self.doc.cmp(&other.doc))
    }
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
        self.blob_scores_until(b, &|| false).unwrap_or_default()
    }

    /// [`Self::blob_scores`], checking `stopped` every [`CHUNK_POLL`]
    /// chunks (`None`: stopped before the doc was scored).
    fn blob_scores_until(&self, b: &Value, stopped: &dyn Fn() -> bool) -> Option<Vec<Option<f32>>> {
        let vecs = b["v"].as_array().map(Vec::as_slice).unwrap_or(&[]);
        let mut out = Vec::with_capacity(vecs.len());
        for (i, part) in vecs.chunks(CHUNK_POLL).enumerate() {
            if i > 0 && stopped() {
                return None;
            }
            out.extend(self.scores_of(&b["k"], part));
        }
        Some(out)
    }

    /// Scores of chunk embeddings `vecs` of blob kind `kind` (`d` / `s`).
    fn scores_of(&self, kind: &Value, vecs: &[Value]) -> Vec<Option<f32>> {
        let b = kind;
        match &self.vec {
            ChunkVec::All => vecs.iter().map(|_| Some(1.0)).collect(),
            ChunkVec::Dense { v, sim, elem, threshold } => {
                if *b != "d" {
                    return vecs.iter().map(|_| None).collect();
                }
                vecs.iter()
                    .map(|c| {
                        let d: Vec<f32> = c.as_array().map(|a| a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect()).unwrap_or_default();
                        if d.len() != v.len() {
                            return None;
                        }
                        let s = elem.score(*sim, v, &d);
                        threshold.is_none_or(|t| s >= t).then_some(s)
                    })
                    .collect()
            }
            ChunkVec::Sparse(q) => {
                if *b != "s" {
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
    /// Matching docs of one leaf (pre-filter applied, before any top `k`);
    /// a stopped search (deadline, cancellation) keeps what it scored.
    fn leaf_hits(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Vec<(DocId, f32)> {
        let mut hits: Vec<(DocId, f32)> = Vec::new();
        self.scan_leaf(leaf, ctx, &mut |d, s| hits.push((d, s)));
        hits
    }

    /// Score the matching docs of one leaf into `emit`, polling the search's
    /// deadline and cancellation while filtering, before every document and
    /// within a document's chunks. `false`: stopped before the end.
    fn scan_leaf(&self, leaf: &LeafReader<'_>, ctx: &SearchContext, emit: &mut dyn FnMut(DocId, f32)) -> bool {
        if ctx.stopped() {
            return false;
        }
        let live = leaf.live_docs();
        let allowed: Option<std::collections::HashSet<DocId>> = match self.filter.as_ref() {
            None => None,
            Some(f) => {
                let mut set = std::collections::HashSet::new();
                if let Some(mut s) = f.scorer(leaf, ctx) {
                    loop {
                        let d = s.advance();
                        if d == rustic_core::query::TERMINATED {
                            break;
                        }
                        set.insert(d);
                        if set.len() % 1024 == 0 && ctx.stopped() {
                            return false;
                        }
                    }
                }
                Some(set)
            }
        };
        let stopped = || ctx.stopped();
        for d in 0..leaf.num_docs() {
            if !leaf.is_root(d) || !live(d) || allowed.as_ref().is_some_and(|a| !a.contains(&d)) {
                continue;
            }
            if ctx.stopped() {
                return false;
            }
            if PROBE_ON.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(p) = SCAN_PROBE.read().clone() {
                    p();
                }
            }
            let Some(b) = super::semantic::blob_of(leaf, self.aux, d) else { continue };
            let Some(scores) = self.blob_scores_until(&b, &stopped) else { return false };
            if let Some(s) = self.combine(&scores) {
                emit(d, s);
            }
        }
        true
    }

    /// Resolve the shard-wide top `k` over `leaves` (kNN semantics), within
    /// the search's `deadline` and cancellation (finding 123): the scan
    /// polls both and keeps only the `k` best candidates. A cancelled search
    /// fails with ES's `task_cancelled_exception`; at the deadline the scan
    /// stops with the candidates found so far (the search times out).
    pub fn with_top_k(mut self, leaves: &[LeafReader<'_>], deadline: Option<std::time::Instant>) -> crate::error::Result<Self> {
        let Some(k) = self.k else { return Ok(self) };
        let ctx = match &self.filter {
            Some(f) => SearchContext::for_query_deadline(f, leaves, rustic_core::query::Bm25Params::default(), deadline),
            None => SearchContext::for_query_deadline(&rustic_core::query::Query::MatchAll, leaves, rustic_core::query::Bm25Params::default(), deadline),
        };
        ctx.execution_result()?;
        let mut best: std::collections::BinaryHeap<Candidate> = std::collections::BinaryHeap::with_capacity(k.min(1 << 16) + 1);
        for (li, leaf) in leaves.iter().enumerate() {
            let complete = self.scan_leaf(leaf, &ctx, &mut |doc, score| {
                best.push(Candidate { score, leaf: li, doc });
                if best.len() > k {
                    best.pop();
                }
            });
            if !complete {
                break;
            }
        }
        if ctx.cancel.as_ref().is_some_and(|c| c.is_cancelled()) {
            let reason = ctx.cancel.as_ref().and_then(|c| c.reason()).unwrap_or_else(|| "by user request".to_string());
            return Err(crate::error::EsError::new(400, "task_cancelled_exception", format!("task cancelled [{reason}]")));
        }
        ctx.execution_result()?;
        // Best first: the order a full sort truncated to `k` gives.
        let all = best.into_sorted_vec();
        let mut fixed: std::collections::HashMap<usize, Vec<(DocId, f32)>> = std::collections::HashMap::new();
        for Candidate { score, leaf, doc } in all {
            fixed.entry(leaves[leaf].id()).or_default().push((doc, score));
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

#[cfg(test)]
mod element_type_tests {
    use super::*;

    /// #545: byte `dot_product` chunks score `0.5 + dot / (dims * 2^15)`;
    /// bit chunks by Hamming similarity, so query byte 0 ranks byte -128 (one
    /// bit away) above byte 3 (two bits away); floats keep their formula.
    #[test]
    fn chunk_scores_follow_the_element_type() {
        let byte = ChunkElem::Byte.score(Similarity::DotProduct, &[100.0, 0.0], &[100.0, 0.0]);
        assert_eq!(byte, 0.652587890625);
        let bit = |d: f32| ChunkElem::Bit.score(Similarity::L2, &[0.0], &[d]);
        assert_eq!(bit(-128.0), 7.0 / 8.0);
        assert_eq!(bit(3.0), 6.0 / 8.0);
        assert!(bit(-128.0) > bit(3.0));
        assert_eq!(ChunkElem::Float.score(Similarity::DotProduct, &[0.6, 0.8], &[0.6, 0.8]), Similarity::DotProduct.score(&[0.6, 0.8], &[0.6, 0.8]));
        assert_eq!(ChunkElem::of(Some("bit")), ChunkElem::Bit);
        assert_eq!(ChunkElem::of(None), ChunkElem::Float);
    }
}
