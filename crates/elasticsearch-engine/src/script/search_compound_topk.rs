//! Spatial upper bounds for one pure expression over an integer and a point.
//! The original compiled expression evaluates every selected document.
use super::{CompiledScript, ast::{Expr, ExprKind, StmtKind, BinOp, Ty}};
use crate::mapping::{EsType, FieldLookup};
use rustic_core::{geo::{GeoIndex, Relation}, query::{LeafReader, collector::{ScoredDoc, TopKCollector}}};
use std::cell::Cell;

#[derive(Clone, Debug)]
pub(super) struct Plan { geo: u16, numeric: u16, formula: Formula }
#[derive(Clone, Copy, Debug)]
enum Formula { Expression, Dynamic, Static }
impl Formula {
    fn valid(self, value: i64) -> bool {
        match self {
            Self::Expression => true,
            Self::Dynamic => value.wrapping_abs().wrapping_add(1) > 0,
            Self::Static => (value as i32).wrapping_abs().wrapping_add(1) > 0,
        }
    }
    fn may_be_invalid(self, low: i64, high: i64) -> bool {
        match self {
            Self::Expression => false,
            Self::Dynamic => [i64::MIN, -i64::MAX, i64::MAX].into_iter().any(|x| low <= x && x <= high),
            Self::Static => [i32::MIN, -i32::MAX, i32::MAX].into_iter().any(|x| {
                let first = low as i128 + (x as i128 - low as i128).rem_euclid(1i128 << 32);
                first <= high as i128
            }),
        }
    }
}

fn math<'a>(expr: &'a Expr, method: &str) -> Option<&'a Expr> {
    match &expr.kind {
        ExprKind::Call { obj, name, args, null_safe: false }
            if matches!(&obj.kind, ExprKind::Var(v) if v == "Math") && name == method && args.len() == 1 => Some(&args[0]),
        _ => None,
    }
}
fn doc_field(expr: &Expr) -> Option<String> {
    match &expr.kind {
        ExprKind::Index { obj, idx } if matches!(&obj.kind, ExprKind::Var(v) if v == "doc") => {
            let ExprKind::Str(field) = &idx.kind else { return None };
            Some(field.clone())
        }
        ExprKind::Field { obj, name, null_safe: false } if matches!(&obj.kind, ExprKind::Var(v) if v == "doc") => Some(name.clone()),
        _ => None,
    }
}
fn painless_fields(source: &str) -> Option<(String, String, Formula)> {
    let (ast, _) = super::parser::parse(source).ok()?;
    if !ast.functions.is_empty() || ast.body.len() != 1 { return None; }
    let expr = match &ast.body[0].kind { StmtKind::Expr(e) | StmtKind::Return(Some(e)) => e, _ => return None };
    let expr = match &expr.kind {
        ExprKind::Binary(BinOp::Div, body, score) if matches!(&score.kind, ExprKind::Var(v) if v == "_score") => body.as_ref(),
        _ => expr,
    };
    let ExprKind::Binary(BinOp::Add, log, product) = &math(expr, "abs")?.kind else { return None };
    let ExprKind::Binary(BinOp::Add, absolute, one) = &math(log, "log")?.kind else { return None };
    if !matches!(one.kind, ExprKind::Int(1)) { return None; }
    let value = math(absolute, "abs")?;
    let (numeric, formula) = match &value.kind {
        ExprKind::Field { obj, name, null_safe: false } if name == "value" => (doc_field(obj)?, Formula::Dynamic),
        ExprKind::Cast { ty: Ty::Int, expr } => {
            let ExprKind::Call { obj, name, args, null_safe: false } = &expr.kind else { return None };
            if name != "get" || args.len() != 1 || !matches!(args[0].kind, ExprKind::Int(0)) { return None; }
            let ExprKind::Cast { ty: Ty::Ref(ty), expr } = &obj.kind else { return None };
            if ty != "List" { return None; }
            (doc_field(expr)?, Formula::Static)
        }
        _ => return None,
    };
    let ExprKind::Binary(BinOp::Mul, lon, lat) = &product.kind else { return None };
    let coordinate = |expr: &Expr, coord: &str| -> Option<String> {
        let expr = match &expr.kind { ExprKind::Cast { ty: Ty::Double, expr } => expr.as_ref(), _ => expr };
        let ExprKind::Field { obj, name, null_safe: false } = &expr.kind else { return None };
        if name != coord { return None; }
        doc_field(obj)
    };
    let geo = coordinate(lon, "lon")?;
    if coordinate(lat, "lat")? != geo { return None; }
    Some((numeric, geo, formula))
}
impl Plan {
    pub(super) fn parse(script: &CompiledScript, lookup: &FieldLookup) -> Option<Self> {
        // Expression parameters can shadow a doc-variable name at link time.
        if script.params.as_object().is_some_and(|params| !params.is_empty()) { return None; }
        let (numeric, geo, formula) = if script.program.expression.is_some() {
            let (numeric, geo) = super::expression::compound_geo_fields(script.source())?;
            (numeric, geo, Formula::Expression)
        } else { painless_fields(script.source())? };
        // Integer doc values (including a missing value's expression zero)
        // guarantee 0 <= ln(abs(value) + 1) <= 64 ln(2). Other mappings,
        // runtime values and arbitrary expression operations do not qualify.
        let numeric = lookup.get(&numeric)?;
        let geo = lookup.get(&geo)?;
        if !numeric.doc_values || !geo.doc_values || !matches!(numeric.declared, EsType::Long | EsType::Integer | EsType::Short | EsType::Byte) { return None; }
        Some(Self { geo: geo.geo_id?, numeric: numeric.core_id?, formula })
    }
    pub(super) fn bind(&self, leaf: &LeafReader<'_>) -> Option<Bound> {
        if leaf.nested_info().is_some() { return None; }
        let LeafReader::Segment(segment) = leaf else { return None };
        if !matches!(self.formula, Formula::Expression) {
            // Missing values or invalid logarithms must still raise their
            // ordinary script errors, even if their geo blocks cannot win.
            let column = segment.fast_fields().get_i64(self.numeric)?;
            if column.is_sparse() || column.num_docs() != leaf.num_docs() { return None; }
            for block in 0..column.block_count() {
                let (low, high) = column.block_min_max(block);
                if self.formula.may_be_invalid(low, high) {
                    let start = block * rustic_core::codec::fastfields::BLOCK_SIZE;
                    let end = (start + rustic_core::codec::fastfields::BLOCK_SIZE).min(leaf.num_docs() as usize);
                    for doc in start..end {
                        if !self.formula.valid(column.get(doc as u32)?) { return None; }
                    }
                }
            }
        }
        let index = segment.geo_index(self.geo)?;
        if index.docs_with_points() != Some(leaf.num_docs() as u64) { return None; }
        // SAFETY: same leaf/scorer lifetime contract as Binding::executor.
        let index = unsafe { std::mem::transmute::<&GeoIndex, &'static GeoIndex>(index) };
        Some(Bound { index, num_docs: leaf.num_docs(), product: !matches!(self.formula, Formula::Expression) })
    }
}

pub(super) struct Bound { index: &'static GeoIndex, num_docs: u32, product: bool }
impl Bound {
    /// `floor`: the k-th best score other leaves already hold (hits must
    /// beat it to matter; `-inf` without one). Blocks whose ceiling is below
    /// it are not evaluated; hits returned may still be below it.
    /// The top `k` of the docs in `range` (a part of the leaf, see
    /// `PartedScorer`; `(0, u32::MAX)`: every doc) and their live count.
    pub(super) fn collect(&self, k: usize, deleted: Option<&[u64]>, floor: f32, range: (u32, u32), mut score: impl FnMut(u32) -> Option<f32>) -> (Vec<ScoredDoc>, u64) {
        let (lo, hi) = (range.0.min(self.num_docs), range.1.min(self.num_docs));
        let deleted_bit = |doc: u32| deleted.is_some_and(|words| words.get(doc as usize / 64).is_some_and(|w| w & (1 << (doc % 64)) != 0));
        let dead = |doc: u32| doc < lo || doc >= hi || deleted_bit(doc);
        let removed: u64 = if deleted.is_some() { (lo..hi).filter(|&d| deleted_bit(d)).count() as u64 } else { 0 };
        let threshold = Cell::new(if floor.is_nan() { f32::NEG_INFINITY } else { floor });
        let rejected = Cell::new(0usize);
        let fallback = Cell::new(false);
        let failed = Cell::new(false);
        let mut seeds = TopKCollector::new(k);
        let n = (hi - lo).min(256);
        for i in 0..n {
            let doc = lo + (i as u64 * (hi - lo) as u64 / n as u64) as u32;
            if dead(doc) { continue; }
            let Some(value) = score(doc) else { return (Vec::new(), 0) };
            seeds.collect(doc, value);
        }
        if let Some(minimum) = seeds.min_score() { threshold.set(threshold.get().max(minimum)); }
        let mut top = TopKCollector::new(k);
        let mut evaluated = 0;
        self.index.visit(|min_lat, max_lat, min_lon, max_lon| {
            if failed.get() || fallback.get() { return Relation::Outside; }
            let (low, upper) = if self.product {
                let corners = [min_lon*min_lat, min_lon*max_lat, max_lon*min_lat, max_lon*max_lat];
                (corners.into_iter().fold(f64::INFINITY, f64::min), corners.into_iter().fold(f64::NEG_INFINITY, f64::max))
            } else { (min_lon + min_lat, max_lon + max_lat) };
            let high = 64.0 * std::f64::consts::LN_2 + upper;
            // Outward allowance exceeds accumulated f64 error for this
            // bounded operation chain; f32 conversion is monotonic.
            let ceiling = (low.abs().max(high.abs()) + 1e-8) as f32;
            if ceiling < threshold.get() {
                rejected.set(rejected.get() + 1);
                Relation::Outside
            } else { Relation::Crosses }
        }, |point, _| {
            if failed.get() || fallback.get() || dead(point.doc) { return; }
            let Some(value) = score(point.doc) else { failed.set(true); return };
            evaluated += 1;
            top.collect(point.doc, value);
            if let Some(minimum) = top.min_score() { threshold.set(threshold.get().max(minimum)); }
            if evaluated >= 8192 && rejected.get() == 0 { fallback.set(true); }
        });
        if failed.get() { return (Vec::new(), 0); }
        if fallback.get() {
            top = TopKCollector::new(k);
            for doc in lo..hi {
                if dead(doc) { continue; }
                let Some(value) = score(doc) else { return (Vec::new(), 0) };
                top.collect(doc, value);
            }
        }
        (top.into_sorted_vec(), (hi - lo) as u64 - removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn painless_domain_checks_cover_wrapping_boundaries() {
        for formula in [Formula::Dynamic, Formula::Static] {
            for center in [i64::MIN, -i64::MAX, i64::MAX, i32::MIN as i64, i32::MAX as i64,
                           -(i32::MAX as i64), (1i64<<32)+(i32::MAX as i64), -(1i64<<32)+(i32::MIN as i64)] {
                let low = center.saturating_sub(8);
                let high = center.saturating_add(8);
                let invalid = (low..=high).any(|value| !formula.valid(value));
                assert_eq!(formula.may_be_invalid(low, high), invalid, "{formula:?} {center}");
            }
            assert!(!formula.may_be_invalid(0, 100000));
            assert!(!formula.may_be_invalid(-100000, 100000));
        }
        assert!(Formula::Static.valid(6_814_400_000));
        assert!(!Formula::Static.valid(2_147_483_647));
        assert!(!Formula::Dynamic.valid(i64::MAX));
    }

    #[test]
    fn painless_plan_accepts_only_exact_safe_formula_shapes() {
        let registry = crate::analysis::AnalysisRegistry::default();
        let mut mappings = crate::mapping::Mappings::parse(&json!({"properties":{
            "population":{"type":"long"},"location":{"type":"geo_point"}
        }}), &registry).unwrap();
        mappings.assign_core_ids();
        let lookup = FieldLookup::build(&mappings, &registry);
        for source in [
            "Math.abs(Math.log(Math.abs(doc['population'].value) + 1) + doc['location'].lon * doc['location'].lat)/_score",
            "Math.abs(Math.log(Math.abs((int)((List)doc.population).get(0)) + 1) + (double)(doc.location.lon) * (double)(doc.location.lat))/_score",
        ] {
            let compile = |source: &str| super::super::compile(source, &json!({}), super::super::ScriptContext::Score).unwrap();
            assert!(Plan::parse(&compile(source), &lookup).is_some());
            assert!(Plan::parse(&compile(&format!("({source})+1")), &lookup).is_none());
            assert!(Plan::parse(&compile(&source.replace("+ 1", "+ 0")), &lookup).is_none());
        }
    }

    #[test]
    fn compound_bounds_match_exhaustive_topk_and_reduce_evaluations() {
        use rustic_core::geo::{GeoPointEntry, encode_lat, encode_lon, decode_lat, decode_lon};
        for tied in [false, true] {
            let n = 20003u32;
            let points: Vec<_> = (0..n).map(|doc| GeoPointEntry { doc,
                lat: encode_lat(if tied { 45.0 } else { (doc % 179) as f64 - 89.0 }),
                lon: encode_lon(if tied { 90.0 } else { (doc.wrapping_mul(1237) % 359) as f64 - 179.0 }),
            }).collect();
            // Bound normally borrows its segment; the test owns that lifetime.
            let index = Box::leak(Box::new(GeoIndex::build(points.clone())));
            let bound = Bound { index, num_docs: n, product: false };
            let mut deleted = vec![0u64; (n as usize).div_ceil(64)];
            for doc in (0..n).step_by(7) { deleted[doc as usize / 64] |= 1 << (doc % 64); }
            let score = |doc: u32| {
                let p = &points[doc as usize];
                let number = if tied { 1234.0 } else if doc % 11 == 0 { i64::MIN as f64 } else { doc as f64 };
                ((number.abs()+1.0).ln()+decode_lon(p.lon)+decode_lat(p.lat)).abs() as f32
            };
            for k in [1,10,100] {
                let mut expected = TopKCollector::new(k);
                let mut total = 0;
                for doc in 0..n {
                    if deleted[doc as usize / 64] & (1 << (doc % 64)) != 0 { continue; }
                    expected.collect(doc, score(doc)); total += 1;
                }
                let mut evaluations = 0;
                let expected = expected.into_sorted_vec();
                let actual = bound.collect(k, Some(&deleted), f32::NEG_INFINITY, (0, u32::MAX), |doc| { evaluations += 1; Some(score(doc)) });
                assert_eq!(actual, (expected.clone(), total));
                // Another leaf's k-th best score: every hit above it is
                // still found, with the same count, and fewer evaluations.
                if let Some(cut) = expected.get(k / 2).map(|h| h.score) {
                    let mut fewer = 0;
                    let (hits, n) = bound.collect(k, Some(&deleted), cut, (0, u32::MAX), |doc| { fewer += 1; Some(score(doc)) });
                    assert_eq!(n, total);
                    let above = |v: &[ScoredDoc]| v.iter().filter(|h| h.score > cut).cloned().collect::<Vec<_>>();
                    assert_eq!(above(&hits), above(&expected), "k={k} tied={tied}");
                    assert!(fewer <= evaluations, "{fewer} vs {evaluations}");
                }
                if !tied { assert!(evaluations < total / 2, "{evaluations} vs {total}"); }
            }
        }
    }

    #[test]
    fn compound_plan_requires_exact_bounded_expression() {
        let registry = crate::analysis::AnalysisRegistry::default();
        let mut mappings = crate::mapping::Mappings::parse(&json!({"properties":{
            "p":{"type":"long"},"g":{"type":"geo_point"},"other":{"type":"geo_point"},"floating":{"type":"double"},"disabled":{"type":"long","doc_values":false}
        }}), &registry).unwrap();
        mappings.assign_core_ids();
        let lookup = FieldLookup::build(&mappings, &registry);
        let base = "abs(ln(abs(doc['p']) + 1) + doc['g'].lon + doc['g'].lat)";
        for (source, accepted) in [
            (base.to_owned(), true), (format!("{base} * _score"), true),
            (format!("{base} + 0"), false), (format!("{base} / _score"), false),
            (base.replace("doc['p']", "doc['floating']"), false),
            (base.replace("doc['p']", "doc['disabled']"), false),
            (base.replace("doc['g'].lat", "doc['other'].lat"), false),
            (base.replace("+ 1", "- 1"), false),
        ] {
            let script = super::super::compile_expression(&source, &json!({}), super::super::ScriptContext::Score).unwrap();
            assert_eq!(Plan::parse(&script, &lookup).is_some(), accepted, "{source}");
        }
        let shadowed = super::super::compile_expression(base, &json!({"doc['p']":1e300}), super::super::ScriptContext::Score).unwrap();
        assert!(Plan::parse(&shadowed, &lookup).is_none());
    }
}
