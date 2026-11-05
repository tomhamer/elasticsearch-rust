//! Restricted spatial pruning for a pure built-in geo decay score script.
//! Selected docs still execute the original compiled script. Unsupported
//! syntax, missing/multivalued points and runtime fields use the usual scorer.
use super::{ast::{Expr, ExprKind, StmtKind}, compile::score_fns::{self, Shape}, CompiledScript};
use crate::mapping::FieldLookup;
use rustic_core::{geo::GeoIndex, query::{LeafReader, collector::{ScoredDoc, TopKCollector}}};
use serde_json::Value;

#[derive(Clone, Debug)]
pub(super) struct Plan {
    field: u16,
    lat: f64,
    lon: f64,
    offset: f64,
    scaling: f64,
    shape: Shape,
}

pub(super) fn constant(expr: &Expr, params: &Value) -> Option<Value> {
    match &expr.kind {
        ExprKind::Str(s) => Some(Value::String(s.clone())),
        ExprKind::Double(n) => serde_json::Number::from_f64(*n).map(Value::Number),
        ExprKind::Int(n) => Some(Value::from(*n)),
        ExprKind::Long(n) => Some(Value::from(*n)),
        ExprKind::Field { obj, name, null_safe: false } if matches!(&obj.kind, ExprKind::Var(v) if v == "params") => params.get(name).cloned(),
        ExprKind::Index { obj, idx } if matches!(&obj.kind, ExprKind::Var(v) if v == "params") => {
            let ExprKind::Str(key) = &idx.kind else { return None };
            params.get(key).cloned()
        }
        _ => None,
    }
}

impl Plan {
    pub(super) fn parse(script: &CompiledScript, lookup: &FieldLookup) -> Option<Self> {
        if script.program.expression.is_some() { return None; }
        let (ast, _) = super::parser::parse(script.source()).ok()?;
        if !ast.functions.is_empty() || ast.body.len() != 1 { return None; }
        let expr = match &ast.body[0].kind { StmtKind::Expr(e) | StmtKind::Return(Some(e)) => e, _ => return None };
        let ExprKind::FnCall { name, args } = &expr.kind else { return None };
        let shape = match name.as_str() {
            "decayGeoGauss" => Shape::Gauss, "decayGeoExp" => Shape::Exp, "decayGeoLinear" => Shape::Linear, _ => return None,
        };
        if args.len() != 5 { return None; }
        let ExprKind::Field { obj, name, null_safe: false } = &args[4].kind else { return None };
        if name != "value" { return None; }
        let ExprKind::Index { obj, idx } = &obj.kind else { return None };
        if !matches!(&obj.kind, ExprKind::Var(v) if v == "doc") { return None; }
        let ExprKind::Str(field) = &idx.kind else { return None };
        let field = lookup.get(field)?.geo_id?;
        let origin = constant(&args[0], &script.params)?;
        let scale = constant(&args[1], &script.params)?;
        let offset = constant(&args[2], &script.params)?;
        // The built-in expects strings for these constructor arguments.
        origin.as_str()?; scale.as_str()?; offset.as_str()?;
        let (lat, lon) = crate::values::parse_geo_point(&origin)?;
        let scale = crate::values::parse_distance(&scale)?;
        let offset = crate::values::parse_distance(&offset)?;
        let decay = constant(&args[3], &script.params)?.as_f64()?;
        if !scale.is_finite() || scale <= 0.0 || !offset.is_finite() || offset < 0.0 || !(decay > 0.0 && decay < 1.0) { return None; }
        let scaling = score_fns::scaling(shape, scale, decay);
        if !scaling.is_finite() || scaling == 0.0 { return None; }
        Some(Self { field, lat, lon, offset, scaling, shape })
    }

    pub(super) fn bind(&self, leaf: &LeafReader<'_>) -> Option<Bound<'static>> {
        if leaf.nested_info().is_some() { return None; }
        let LeafReader::Segment(segment) = leaf else { return None };
        let index = segment.geo_index(self.field)?;
        if index.docs_with_points() != Some(leaf.num_docs() as u64) { return None; }
        // SAFETY: same scorer/leaf borrowing contract as Binding::executor.
        let index = unsafe { std::mem::transmute::<&GeoIndex, &'static GeoIndex>(index) };
        Some(Bound { plan: self.clone(), index, num_docs: leaf.num_docs() })
    }
}

pub(super) struct Bound<'a> {
    plan: Plan,
    index: &'a GeoIndex,
    num_docs: u32,
}

impl Bound<'_> {
    pub(super) fn collect(&self, k: usize, deleted: Option<&[u64]>, boost: f32, mut score: impl FnMut(u32) -> Option<f32>) -> (Vec<ScoredDoc>, u64) {
        let deleted_doc = |doc: u32| deleted.is_some_and(|words| words.get(doc as usize / 64).is_some_and(|w| w & (1 << (doc % 64)) != 0));
        let removed: u64 = deleted.map(|words| words.iter().take((self.num_docs as usize).div_ceil(64)).enumerate().map(|(i, &word)| {
            let valid = (self.num_docs as usize - i * 64).min(64);
            (word & if valid == 64 { u64::MAX } else { (1u64 << valid) - 1 }).count_ones() as u64
        }).sum()).unwrap_or(0);
        let mut top = TopKCollector::new(k);
        let plan = &self.plan;
        self.index.visit_by_distance(plan.lat, plan.lon, |distance, points| {
            let ceiling = score_fns::apply(plan.shape, plan.scaling, (distance - plan.offset).max(0.0)) as f32 * boost;
            if top.min_score().is_some_and(|minimum| ceiling < minimum) { return false; }
            for point in points {
                if deleted_doc(point.doc) { continue; }
                let Some(value) = score(point.doc) else { return false };
                top.collect(point.doc, value);
            }
            true
        });
        (top.into_sorted_vec(), self.num_docs as u64 - removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn geo_script_plan_accepts_only_pure_constant_binding_calls() {
        let registry = crate::analysis::AnalysisRegistry::default();
        let mut mappings = crate::mapping::Mappings::parse(&json!({"properties":{"location":{"type":"geo_point"}}}), &registry).unwrap();
        mappings.assign_core_ids();
        let lookup = FieldLookup::build(&mappings, &registry);
        let params = json!({"origin":"52.37, 4.8951","scale":"500km","offset":"0km","decay":0.1});
        for source in [
            "decayGeoGauss(params.origin, params.scale, params.offset, params.decay, doc['location'].value)",
            "return decayGeoExp(params['origin'], '500km', '0km', 0.1, doc['location'].value);",
        ] {
            let script = super::super::compile(source, &params, super::super::ScriptContext::Score).unwrap();
            assert!(Plan::parse(&script, &lookup).is_some());
        }
        for source in [
            "decayGeoGauss(params.origin, params.scale, params.offset, params.decay, doc['location'].value) + 1",
            "double x = 1; return decayGeoGauss(params.origin, params.scale, params.offset, params.decay, doc['location'].value);",
            "decayGeoGauss(params.origin, params.scale, params.offset, params.decay, doc['missing'].value)",
            "decayGeoGauss(params.origin, params.scale, params.offset, _score, doc['location'].value)",
        ] {
            let script = super::super::compile(source, &params, super::super::ScriptContext::Score).unwrap();
            assert!(Plan::parse(&script, &lookup).is_none(), "{source}");
        }
    }

    #[test]
    fn geo_script_bound_matches_exhaustive_scores_and_ties() {
        use rustic_core::geo::{GeoPointEntry, encode_lat, encode_lon, decode_lat, decode_lon};
        let n = 10003u32;
        let points: Vec<_> = (0..n).map(|doc| GeoPointEntry { doc,
            lat: encode_lat((doc % 179) as f64 - 89.0), lon: encode_lon((doc.wrapping_mul(1237) % 359) as f64 - 179.0) }).collect();
        let index = GeoIndex::build(points.clone());
        let mut deleted = vec![0u64; (n as usize).div_ceil(64)];
        for doc in (0..n).step_by(7) { deleted[doc as usize / 64] |= 1 << (doc % 64); }
        for shape in [Shape::Gauss, Shape::Exp, Shape::Linear] {
            for (lat, lon) in [(52.37, 4.8951), (89.9, 179.9), (-89.9, -179.9)] {
                for (scale, offset) in [(500000.0, 0.0), (10000.0, 0.0), (20000000.0, 1000000.0)] {
                    let plan = Plan { field: 0, lat, lon, offset, scaling: score_fns::scaling(shape, scale, 0.1), shape };
                    let score = |doc: u32| {
                        let point = &points[doc as usize];
                        let distance = crate::sloppy_math::haversin_meters(lat, lon, decode_lat(point.lat), decode_lon(point.lon));
                        score_fns::apply(shape, plan.scaling, (distance - offset).max(0.0)) as f32 * 1.2
                    };
                    for k in [1, 10, 100] {
                        let mut expected = TopKCollector::new(k);
                        let mut total = 0;
                        for doc in 0..n {
                            if deleted[doc as usize / 64] & (1 << (doc % 64)) != 0 { continue; }
                            expected.collect(doc, score(doc)); total += 1;
                        }
                        let bound = Bound { plan: plan.clone(), index: &index, num_docs: n };
                        assert_eq!(bound.collect(k, Some(&deleted), 1.2, |doc| Some(score(doc))), (expected.into_sorted_vec(), total));
                    }
                }
            }
        }
    }
}
