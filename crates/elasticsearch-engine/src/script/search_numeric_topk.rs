//! Bounds for pure `Math.log[10](doc[field].value * double + offset)`.
//! All selected docs retain the compiled script's arithmetic and errors.
use super::{ast::{BinOp, ExprKind, StmtKind}, CompiledScript, search_geo_topk::constant};
use crate::mapping::{EsType, FieldLookup};
use rustic_core::query::{LeafReader, NumericDv};

#[derive(Clone, Debug)]
pub(super) struct Plan { field: u16 }

impl Plan {
    pub(super) fn parse(script: &CompiledScript, lookup: &FieldLookup) -> Option<Self> {
        if script.program.expression.is_some() { return None; }
        let (ast, _) = super::parser::parse(script.source()).ok()?;
        if !ast.functions.is_empty() || ast.body.len() != 1 { return None; }
        let expr = match &ast.body[0].kind { StmtKind::Expr(e) | StmtKind::Return(Some(e)) => e, _ => return None };
        let ExprKind::Call { obj, name, args, null_safe: false } = &expr.kind else { return None };
        if !matches!(&obj.kind, ExprKind::Var(v) if v == "Math") || !matches!(name.as_str(), "log" | "log10") || args.len() != 1 { return None; }
        let ExprKind::Binary(BinOp::Add, product, offset) = &args[0].kind else { return None };
        let ExprKind::Binary(BinOp::Mul, doc_value, factor) = &product.kind else { return None };
        let factor = constant(factor, &script.params)?;
        // Integer multiplication could overflow before conversion to double.
        if !factor.is_f64() { return None; }
        let factor = factor.as_f64()?;
        let offset = constant(offset, &script.params)?.as_f64()?;
        if !factor.is_finite() || factor <= 0.0 || !offset.is_finite() || offset < 1.0 { return None; }
        let ExprKind::Field { obj, name, null_safe: false } = &doc_value.kind else { return None };
        if name != "value" { return None; }
        let ExprKind::Index { obj, idx } = &obj.kind else { return None };
        if !matches!(&obj.kind, ExprKind::Var(v) if v == "doc") { return None; }
        let ExprKind::Str(field) = &idx.kind else { return None };
        let field = lookup.get(field)?;
        if !matches!(field.declared, EsType::Long | EsType::Integer | EsType::Short | EsType::Byte | EsType::Double | EsType::Float | EsType::HalfFloat) { return None; }
        Some(Self { field: field.core_id? })
    }

    pub(super) fn bind(&self, leaf: &LeafReader<'_>) -> Option<Bound> {
        if leaf.nested_info().is_some() { return None; }
        let values = NumericDv::open(leaf, self.field);
        // SAFETY: bound values live only inside the scorer borrowing the leaf.
        let values = unsafe { std::mem::transmute::<NumericDv<'_>, NumericDv<'static>>(values) };
        Some(Bound { values, num_docs: leaf.num_docs() })
    }
}

pub(super) struct Bound {
    pub(super) values: NumericDv<'static>,
    pub(super) num_docs: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn numeric_plan_rejects_overflow_and_nonmonotone_or_effectful_scripts() {
        let registry = crate::analysis::AnalysisRegistry::default();
        let mut mappings = crate::mapping::Mappings::parse(&json!({"properties":{"population":{"type":"long"}}}), &registry).unwrap();
        mappings.assign_core_ids();
        let lookup = FieldLookup::build(&mappings, &registry);
        for (source, eligible) in [
            ("Math.log10(doc['population'].value * 1.2 + 2)", true),
            ("return Math.log(doc['population'].value * params.factor + 1);", true),
            ("Math.log10(doc['population'].value * 2 + 2)", false),
            ("Math.log10(doc['population'].value * -1.2 + 2)", false),
            ("Math.log10(doc['population'].value * 1.2 + 0)", false),
            ("Math.log10(doc['population'].value * 1.2 + 2) + _score", false),
        ] {
            let script = super::super::compile(source, &json!({"factor":1.2}), super::super::ScriptContext::Score).unwrap();
            assert_eq!(Plan::parse(&script, &lookup).is_some(), eligible, "{source}");
        }
    }
}
