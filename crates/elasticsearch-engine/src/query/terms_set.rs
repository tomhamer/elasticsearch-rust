//! `terms_set`: documents matching at least N of the given terms, N taken
//! from a numeric field of the document (`minimum_should_match_field`).
//! Score = sum of the matching terms' scores (Lucene `CoveringQuery`).

use super::dsl::{term_query, QueryContext};
use crate::error::{EsError, Result};
use rustic_core::query::{CustomQuery, LeafReader, Query, Scorer, SearchContext, TERMINATED};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug)]
struct TermsSet {
    terms: Vec<Query>,
    msm_field: u16,
    boost: f32,
}

impl CustomQuery for TermsSet {
    fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        let mut acc: BTreeMap<u32, (u32, f32)> = BTreeMap::new();
        for q in &self.terms {
            let Some(mut s) = q.scorer(leaf, ctx) else { continue };
            loop {
                let d = s.advance();
                if d == TERMINATED {
                    break;
                }
                let e = acc.entry(d).or_insert((0, 0.0));
                e.0 += 1;
                e.1 += s.score();
            }
        }
        let out: Vec<(u32, f32)> = acc
            .into_iter()
            .filter(|(d, (n, _))| {
                let need = leaf.numeric_values(self.msm_field, *d).into_iter().fold(None, |a: Option<f64>, v| Some(a.map_or(v, |a| a.min(v))));
                matches!(need, Some(m) if (*n as f64) >= m)
            })
            .map(|(d, (_, s))| (d, s * self.boost))
            .collect();
        if out.is_empty() {
            return None;
        }
        Some(Box::new(rustic_core::query::query::ScoredDocs::new(out)))
    }
    fn children(&self) -> Vec<&Query> {
        self.terms.iter().collect()
    }
    fn name(&self) -> &'static str {
        "terms_set"
    }
}

pub fn terms_set(body: &Value, ctx: &QueryContext) -> Result<Query> {
    let o = body.as_object().ok_or_else(|| EsError::parsing("[terms_set] query malformed, no start_object after query name"))?;
    let (field, spec) = o.iter().next().ok_or_else(|| EsError::parsing("[terms_set] query malformed, no field specified"))?;
    if o.len() > 1 {
        let k2 = o.keys().nth(1).unwrap();
        return Err(EsError::parsing(format!("[terms_set] query doesn't support multiple fields, found [{field}] and [{k2}]")));
    }
    let so: &Map<String, Value> = spec.as_object().ok_or_else(|| EsError::parsing("[terms_set] query malformed, no start_object after field name"))?;
    for k in so.keys() {
        if !matches!(k.as_str(), "terms" | "minimum_should_match_field" | "minimum_should_match_script" | "boost" | "_name") {
            return Err(EsError::parsing(format!("[terms_set] query does not support [{k}]")));
        }
    }
    // ES `doToQuery`: no terms is a match-none query, checked before the
    // minimum-should-match source.
    let empty = Vec::new();
    let terms = match so.get("terms") {
        None => &empty,
        Some(v) => v.as_array().ok_or_else(|| EsError::parsing("[terms_set] query requires a terms array"))?,
    };
    if terms.is_empty() {
        return Ok(Query::MatchNone);
    }
    let boost = so.get("boost").and_then(Value::as_f64).unwrap_or(1.0) as f32;
    let mf = match (so.get("minimum_should_match_field").and_then(Value::as_str), so.get("minimum_should_match_script")) {
        (Some(f), None) => f,
        (None, Some(_)) => return Err(EsError::illegal_argument("[terms_set] minimum_should_match_script is not supported")),
        (Some(_), Some(_)) => return Err(EsError::illegal_argument("[terms_set] query requires exactly one of [minimum_should_match_field] or [minimum_should_match_script]")),
        (None, None) => return Err(EsError::query_shard(ctx.index_name, ctx.index_uuid, "failed to create query: No minimum should match has been specified")),
    };
    let Some(f) = ctx.lookup.get(field) else { return Ok(Query::MatchNone) };
    let msm = ctx
        .lookup
        .get(mf)
        .and_then(|m| m.core_id)
        .ok_or_else(|| EsError::query_shard(ctx.index_name, ctx.index_uuid, format!("failed to find minimum_should_match field [{mf}]")))?;
    let qs = terms.iter().map(|t| term_query(f, t, false, ctx)).collect::<Result<Vec<_>>>()?;
    let _ = Arc::new(());
    Ok(Query::Custom(Arc::new(TermsSet { terms: qs, msm_field: msm, boost })))
}
