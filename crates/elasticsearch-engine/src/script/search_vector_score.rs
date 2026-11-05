//! Bind a query vector once for a pure `cosineSimilarity` / `dotProduct`
//! score expression. Effectful scripts keep the general executor; document
//! errors fall back to it too. A bound leaf scans its rows in parallel on
//! the search pool (ES runs query-phase slices concurrently), one top-k per
//! slice merged by score then doc id, so results equal a sequential scan.
use super::{ast::{BinOp, ExprKind, StmtKind}, search_geo_topk::constant, CompiledScript, Executor};
use rustic_core::query::collector::{ScoredDoc, TopKCollector};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Function { Cosine, Dot }

#[derive(Clone, Debug)]
pub(super) struct Plan {
    field: String,
    query: Vec<f32>,
    norm: f32,
    offset: f64,
    function: Function,
}

/// Rows per parallel slice of a bound scan (below it one slice).
const SLICE_ROWS: usize = 16_384;

impl Plan {
    pub(super) fn parse(script: &CompiledScript) -> Option<Self> {
        if script.program.expression.is_some() { return None; }
        let (ast, _) = super::parser::parse(script.source()).ok()?;
        if !ast.functions.is_empty() || ast.body.len() != 1 { return None; }
        let expr = match &ast.body[0].kind { StmtKind::Expr(e) | StmtKind::Return(Some(e)) => e, _ => return None };
        let (expr, offset) = match &expr.kind {
            ExprKind::Binary(BinOp::Add, lhs, rhs) => (lhs.as_ref(), constant(rhs, &script.params)?.as_f64()?),
            _ => (expr, 0.0),
        };
        if !offset.is_finite() { return None; }
        let ExprKind::FnCall { name, args } = &expr.kind else { return None };
        let function = match name.as_str() { "cosineSimilarity" => Function::Cosine, "dotProduct" => Function::Dot, _ => return None };
        if args.len() != 2 { return None; }
        let field = constant(&args[1], &script.params)?.as_str()?.to_owned();
        let value = constant(&args[0], &script.params)?;
        let query: Vec<f32> = value.as_array()?.iter().map(|v| v.as_f64().map(|n| n as f32)).collect::<Option<_>>()?;
        if query.is_empty() || query.iter().any(|n| !n.is_finite()) { return None; }
        let norm = rustic_core::vector::norm(&query);
        if !norm.is_finite() || (function == Function::Cosine && norm == 0.0) { return None; }
        Some(Self { field, query, norm, offset, function })
    }

    pub(super) fn score(&self, exec: &mut Executor) -> Option<f64> {
        exec.rt.doc.with_vector(&self.field, |vector| self.score_row(vector))?
    }

    fn score_row(&self, vector: &[f32]) -> Option<f64> {
        if vector.len() != self.query.len() { return None; }
        match self.function {
            Function::Dot => self.score_row_with_norm(vector, 1.0),
            Function::Cosine => self.score_row_with_norm(vector, rustic_core::vector::norm(vector)),
        }
    }

    fn score_row_with_norm(&self, vector: &[f32], norm: f32) -> Option<f64> {
        if vector.len() != self.query.len() { return None; }
        match self.function {
            // ES `dotProduct` on float vectors: the plain f32 dot product
            // (Lucene `VectorUtil.dotProduct`), whatever the norms.
            Function::Dot => Some(rustic_core::vector::dot(&self.query, vector) as f64 + self.offset),
            Function::Cosine => {
                if norm == 0.0 { return None; }
                Some((rustic_core::vector::dot(&self.query, vector) / (self.norm * norm)) as f64 + self.offset)
            }
        }
    }

    pub(super) fn bind(&self, lookup: &crate::mapping::FieldLookup, leaf: &rustic_core::query::LeafReader<'_>) -> Option<Bound> {
        if leaf.nested_info().is_some() { return None; }
        let resolved = lookup.get(&self.field)?;
        let field = resolved.core_id?;
        let packing = crate::vectors::Packing::of(&resolved);
        // SAFETY: the bound visitor lives in the scorer borrowing this leaf.
        let leaf = unsafe { std::mem::transmute::<rustic_core::query::LeafReader<'_>, rustic_core::query::LeafReader<'static>>(*leaf) };
        Some(Bound { plan: self.clone(), field, packing, leaf })
    }
}

pub(super) struct Bound {
    plan: Plan,
    field: u16,
    /// A `byte` / `bit` field: rows are unpacked (into one reused buffer
    /// per slice) and scored exactly as the per-doc path scores them.
    packing: Option<crate::vectors::Packing>,
    leaf: rustic_core::query::LeafReader<'static>,
}

impl Bound {
    /// The top `k` of the leaf's docs (`docs`: the inner query's matches in
    /// doc order; `None`: every doc), skipping `deleted`. `None` when a doc
    /// lacks a valid value: the ordinary scorer then reports its error.
    pub(super) fn collect(&self, docs: Option<&[u32]>, k: usize, deleted: Option<&[u64]>, boost: f32) -> Option<(Vec<ScoredDoc>, u64)> {
        let live = |d: u32| deleted.and_then(|b| b.get(d as usize / 64)).copied().unwrap_or(0) & (1u64 << (d % 64)) == 0;
        let count = docs.map_or(self.leaf.num_docs() as usize, <[u32]>::len);
        let field = self.field;
        let plan = &self.plan;
        let packing = self.packing;
        // Sealed segments scan concurrently in slices; memtables, small
        // leaves and callers already on another Rayon pool sequentially.
        if let rustic_core::query::LeafReader::Segment(segment) = self.leaf {
            let slices = if count >= 2 * SLICE_ROWS {
                count.div_ceil(SLICE_ROWS).min(4 * rustic_core::query::searcher::search_pool_threads().max(1))
            } else { 1 };
            let rows = |docs: &mut dyn Iterator<Item = u32>, visit: &mut dyn FnMut(u32, &[f32], f32) -> bool| segment.visit_vectors_with_norm(field, docs, visit);
            if slices == 1 { return scan(plan, packing, docs, 0..count, k, boost, &live, &rows); }
            let per = count.div_ceil(slices);
            let parts = rustic_core::query::searcher::par_map_search(slices, |i| scan(plan, packing, docs, i * per..((i + 1) * per).min(count), k, boost, &live, &rows));
            let mut merged: Vec<ScoredDoc> = Vec::with_capacity(slices * k);
            let mut total = 0;
            for part in parts {
                let (hits, n) = part?;
                merged.extend(hits);
                total += n;
            }
            // TopKCollector's order: score descending, then doc ascending.
            merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal).then(a.doc.cmp(&b.doc)));
            merged.truncate(k);
            return Some((merged, total));
        }
        let leaf = self.leaf;
        let rows = |docs: &mut dyn Iterator<Item = u32>, visit: &mut dyn FnMut(u32, &[f32], f32) -> bool| leaf.visit_vectors_with_norm(field, docs, visit);
        scan(plan, packing, docs, 0..count, k, boost, &live, &rows)
    }
}

/// One slice (`range` of `docs`, or of doc ids) into its own top `k`.
#[allow(clippy::too_many_arguments)]
fn scan(plan: &Plan, packing: Option<crate::vectors::Packing>, docs: Option<&[u32]>, range: std::ops::Range<usize>, k: usize, boost: f32, live: &dyn Fn(u32) -> bool,
    rows: &dyn Fn(&mut dyn Iterator<Item = u32>, &mut dyn FnMut(u32, &[f32], f32) -> bool) -> bool) -> Option<(Vec<ScoredDoc>, u64)> {
    let mut top = TopKCollector::new(k);
    let mut total = 0;
    let mut unpacked: Vec<f32> = Vec::new();
    let mut visit = |doc: u32, vector: &[f32], norm: f32| {
        // A packed row's stored norm is not its values' norm.
        let score = match packing {
            Some(packing) => {
                packing.unpack_into(vector, &mut unpacked);
                plan.score_row(&unpacked)
            }
            None => plan.score_row_with_norm(vector, norm),
        };
        let Some(score) = score else { return false };
        if score < 0.0 || score.is_nan() { return false; }
        top.collect(doc, score as f32 * boost);
        total += 1;
        true
    };
    let valid = match docs {
        Some(docs) => rows(&mut docs[range].iter().copied().filter(|&d| live(d)), &mut visit),
        None => rows(&mut (range.start as u32..range.end as u32).filter(|&d| live(d)), &mut visit),
    };
    // A missing/invalid value must still execute the ordinary scorer and
    // report its original error, even when it cannot enter the top k.
    valid.then(|| (top.into_sorted_vec(), total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn only_pure_parameter_vector_expressions_are_bound() {
        for (source, expected) in [
            ("cosineSimilarity(params.q, 'v') + 1.0", true),
            ("return cosineSimilarity(params['q'], params.field);", true),
            ("params.q[0] = 2; cosineSimilarity(params.q, 'v') + 1.0", false),
            ("cosineSimilarity(params.q, 'v') + _score", false),
            ("cosineSimilarity(params.q, 'v') + Math.random()", false),
        ] {
            let script = super::super::compile(source, &json!({"q":[1.0,2.0],"field":"v"}), super::super::ScriptContext::Score).unwrap();
            assert_eq!(Plan::parse(&script).is_some(), expected, "{source}");
        }
    }
}
