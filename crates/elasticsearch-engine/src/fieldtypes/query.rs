//! Query DSL for the field types of this module. [`intercept`] runs before
//! the generic Query DSL translation and handles the queries whose
//! semantics depend on the field type (range relations, wildcard
//! verification, `constant_keyword` rewrites, date nanos, feature
//! scoring, shapes, joins, percolation, ...).

use super::{is_histogram, is_shape, range_kind};
use crate::date::{parse_date_math, DateFormatter, Tz};
use crate::error::{EsError, Result};
use crate::mapping::{EsType, ResolvedField};
use crate::query::dsl::{parse_query, scalar_string, QueryContext};
use crate::values;
use rustic_core::fieldtypes::features::{self, FeatureFunction};
use rustic_core::fieldtypes::filter::FilterQuery;
use rustic_core::fieldtypes::{ngram, range as rng, shape, version};
use rustic_core::query::{Bound, LeafReader, MinShouldMatch, MultiTermRewrite, Query};
use rustic_core::types::{DocId, Term};
use serde_json::{json, Map, Value};
use std::cell::Cell;
use std::sync::Arc;

thread_local! {
    static BYPASS: Cell<bool> = const { Cell::new(false) };
}

/// Run `f` with interception disabled (to delegate to the generic DSL).
fn bypass<T>(f: impl FnOnce() -> T) -> T {
    let prev = BYPASS.with(|b| b.replace(true));
    let r = f();
    BYPASS.with(|b| b.set(prev));
    r
}

fn shard_err(ctx: &QueryContext, reason: impl Into<String>) -> EsError {
    EsError::query_shard(ctx.index_name, ctx.index_uuid, reason)
}

fn boost_of(o: &Map<String, Value>) -> f32 {
    o.get("boost").and_then(Value::as_f64).unwrap_or(1.0) as f32
}

fn with_boost(q: Query, boost: f32) -> Query {
    if (boost - 1.0).abs() > f32::EPSILON {
        Query::Boost { query: Box::new(q), boost }
    } else {
        q
    }
}

fn filter_all(qs: Vec<Query>) -> Option<Query> {
    match qs.len() {
        0 => None,
        1 => qs.into_iter().next(),
        _ => Some(Query::Bool { must: vec![], should: vec![], must_not: vec![], filter: qs, minimum_should_match: MinShouldMatch::Count(0) }),
    }
}

fn any_of(qs: Vec<Query>) -> Query {
    match qs.len() {
        0 => Query::MatchNone,
        1 => qs.into_iter().next().unwrap(),
        _ => Query::Bool { must: vec![], should: qs, must_not: vec![], filter: vec![], minimum_should_match: MinShouldMatch::Count(1) },
    }
}

/// The single `field: spec` entry of a field-level query body.
fn field_entry<'v>(body: &'v Value) -> Option<(&'v String, &'v Value)> {
    body.as_object()?.iter().find(|(k, _)| k.as_str() != "boost" && k.as_str() != "_name")
}

/// Handle `name` (with `body`) when it needs field-type specific logic.
pub fn intercept(name: &str, body: &Value, ctx: &QueryContext) -> Result<Option<Query>> {
    if BYPASS.with(|b| b.get()) {
        return Ok(None);
    }
    let obj = || body.as_object().ok_or_else(|| EsError::parsing(format!("[{name}] query malformed, no start_object after query name")));
    match name {
        "has_child" => return super::join::has_child(obj()?, ctx).map(Some),
        "has_parent" => return super::join::has_parent(obj()?, ctx).map(Some),
        "parent_id" => return super::join::parent_id(obj()?, ctx).map(Some),
        "rank_feature" => return rank_feature(obj()?, ctx).map(Some),
        "sparse_vector" | "text_expansion" | "weighted_tokens" => return sparse_vector_query(name, body, ctx).map(Some),
        "semantic" => return semantic(obj()?, ctx).map(Some),
        "percolate" => return super::percolate::query(obj()?, ctx).map(Some),
        "geo_shape" | "shape" => return shape_query(name, obj()?, ctx).map(Some),
        _ => {}
    }
    if name == "match" {
        // ES 8.18+: `match` on a semantic_text field runs the semantic query.
        if let Some((field, spec)) = field_entry(body) {
            if let Some(f) = ctx.lookup.get(field) {
                if f.declared.is_semantic() {
                    let (q, boost) = match spec {
                        Value::Object(o) => (o.get("query").cloned().unwrap_or(Value::Null), boost_of(o)),
                        other => (other.clone(), 1.0),
                    };
                    if let Some(sq) = semantic_text_query(&f, &q, boost, ctx).map_err(|e| semantic_shard_err(ctx, e))? {
                        return Ok(Some(sq));
                    }
                }
            }
        }
    }
    if !matches!(name, "term" | "terms" | "range" | "exists" | "prefix" | "wildcard" | "regexp" | "fuzzy" | "match" | "match_phrase" | "match_phrase_prefix" | "match_bool_prefix") {
        return Ok(None);
    }
    // Field of the query.
    let field = if name == "exists" {
        match body.get("field").and_then(Value::as_str) {
            Some(f) => f.to_string(),
            None => return Ok(None),
        }
    } else if name == "terms" {
        match body.as_object().and_then(|o| o.keys().find(|k| !matches!(k.as_str(), "boost" | "_name"))) {
            Some(k) => k.clone(),
            None => return Ok(None),
        }
    } else {
        match field_entry(body) {
            Some((k, _)) => k.clone(),
            None => return Ok(None),
        }
    };
    let Some(f) = ctx.lookup.get(&field).cloned() else {
        return rank_features_sub(name, &field, body, ctx);
    };
    if f.flattened_key().is_some() {
        return flattened_sub(name, &f, body, ctx);
    }
    let decl = f.declared;
    if matches!(decl, EsType::MatchOnlyText | EsType::PatternText) && name != "exists" {
        let q = bypass(|| parse_query(&json!({ name: body }), ctx))?;
        let boost = match field_entry(body) {
            Some((_, Value::Object(o))) => boost_of(o),
            _ => 1.0,
        };
        return Ok(Some(Query::ConstScore { query: Box::new(q), score: boost, point_set: false }));
    }
    if name == "exists" {
        return Ok(exists(&f, body, ctx));
    }
    match name {
        "term" => {
            let (_, spec) = field_entry(body).unwrap();
            let (v, boost, ci) = match spec {
                Value::Object(o) => (o.get("value").cloned().unwrap_or(Value::Null), boost_of(o), o.get("case_insensitive").and_then(Value::as_bool).unwrap_or(false)),
                other => (other.clone(), 1.0, false),
            };
            Ok(term(&f, &v, ci, ctx)?.map(|q| with_boost(q, boost)))
        }
        "terms" => {
            let vals = match body.get(&field) {
                Some(Value::Array(a)) => a.clone(),
                _ => return Ok(None),
            };
            let mut qs = Vec::new();
            for v in &vals {
                match term(&f, v, false, ctx)? {
                    Some(q) => qs.push(q),
                    None => return Ok(None),
                }
            }
            let boost = body.get("boost").and_then(Value::as_f64).unwrap_or(1.0) as f32;
            // A single set query in ES for these types (one clause); range,
            // date_nanos and wildcard fields use a disjunction counted per value.
            let point_set = matches!(f.declared, EsType::UnsignedLong | EsType::AggregateMetricDouble | EsType::Join | EsType::AnnotatedText);
            Ok(Some(Query::ConstScore { query: Box::new(any_of(qs)), score: boost, point_set }))
        }
        "match" | "match_phrase" | "match_phrase_prefix" | "match_bool_prefix" => {
            if decl.is_text() || matches!(decl, EsType::Keyword | EsType::Wildcard | EsType::Version | EsType::Flattened) || !handles_term(&f) {
                return Ok(None);
            }
            let (_, spec) = field_entry(body).unwrap();
            let v = match spec {
                Value::Object(o) => o.get("query").cloned().unwrap_or(Value::Null),
                other => other.clone(),
            };
            term(&f, &v, false, ctx)
        }
        "range" => range(&f, body, ctx),
        _ => multi_term(name, &f, body, ctx),
    }
}

fn handles_term(f: &ResolvedField) -> bool {
    range_kind(f.declared).is_some()
        || is_shape(f.declared)
        || is_histogram(f.declared)
        || matches!(
            f.declared,
            EsType::ConstantKeyword
                | EsType::DateNanos
                | EsType::UnsignedLong
                | EsType::AggregateMetricDouble
                | EsType::SparseVector
                | EsType::RankFeatures
                | EsType::RankFeature
                | EsType::Murmur3
                | EsType::Binary
                | EsType::Percolator
                | EsType::Completion
                | EsType::RankVectors
                | EsType::Wildcard
        )
}

fn constant_str(f: &ResolvedField) -> Option<String> {
    f.constant_value.as_ref().map(|v| match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    })
}

fn all_or_none(b: bool) -> Query {
    if b {
        Query::MatchAll
    } else {
        Query::MatchNone
    }
}

/// Exact-value semantics for the field types here (`None` = generic).
pub fn term(f: &ResolvedField, v: &Value, ci: bool, ctx: &QueryContext) -> Result<Option<Query>> {
    let s = scalar_string(v);
    if let Some(kind) = range_kind(f.declared) {
        let Some(id) = f.core_id else { return Ok(Some(Query::MatchNone)) };
        let k = super::index::range_key(f, kind, v, false, ctx.now).map_err(|e| shard_err(ctx, format!("failed to create query: {}", e.reason)))?;
        let k2 = if kind == rng::RangeKind::Date { super::index::range_key(f, kind, v, true, ctx.now).map_err(|e| shard_err(ctx, e.reason))? } else { k };
        return Ok(Some(range_relation_query(id, (k, k2), rng::Relation::Intersects)));
    }
    if is_shape(f.declared) {
        return Err(shard_err(ctx, format!("Geometry fields do not support exact searching, use dedicated geometry queries instead: [{}]", f.name)));
    }
    if is_histogram(f.declared) {
        return Err(shard_err(ctx, format!("[{}] field do not support searching, use dedicated aggregations instead: [{}]", f.declared.name(), f.name)));
    }
    Ok(Some(match f.declared {
        EsType::ConstantKeyword => match constant_str(f) {
            Some(c) => all_or_none(if ci { c.to_lowercase() == s.to_lowercase() } else { c == s }),
            None => Query::MatchNone,
        },
        EsType::DateNanos => {
            let Some(id) = f.core_id else { return Ok(Some(Query::MatchNone)) };
            let lo = nanos_bound(f, v, None, None, false, ctx)?;
            let hi = nanos_bound(f, v, None, None, true, ctx)?;
            Query::RangeI64 { field: id, lower: Bound::Included(lo), upper: Bound::Included(hi) }
        }
        EsType::UnsignedLong => {
            let Some(id) = f.core_id else { return Ok(Some(Query::MatchNone)) };
            match parse_u64_exact(v) {
                Some(Ok(x)) => Query::RangeU64 { field: id, lower: Bound::Included(x), upper: Bound::Included(x) },
                Some(Err(())) => Query::MatchNone,
                None => return Err(shard_err(ctx, format!("failed to create query: For input string: \"{s}\""))),
            }
        }
        EsType::AggregateMetricDouble => {
            let Some(id) = super::aggs_default_metric(f).and_then(|dm| f.aux_id(&dm)) else { return Ok(Some(Query::MatchNone)) };
            let x: f64 = s.trim().parse().map_err(|_| shard_err(ctx, format!("failed to create query: For input string: \"{s}\"")))?;
            Query::RangeF64 { field: id, lower: Bound::Included(x), upper: Bound::Included(x) }
        }
        EsType::SparseVector | EsType::RankFeatures => {
            let Some(id) = f.core_id else { return Ok(Some(Query::MatchNone)) };
            feature_scoring(id, vec![(s, 1.0)], "term")
        }
        EsType::RankFeature => return Err(shard_err(ctx, format!("[rank_feature] fields do not support term queries"))),
        EsType::Join => {
            let Some(id) = f.core_id else { return Ok(Some(Query::MatchNone)) };
            Query::Term(Term::new_text(id, &s))
        }
        EsType::AnnotatedText => {
            let Some(id) = f.core_id else { return Ok(Some(Query::MatchNone)) };
            // Annotation values are indexed verbatim; plain words analyzed.
            let analyzed = f.search_analyzer.as_ref().map(|a| a.analyze(&s).into_iter().map(|t| t.text).collect::<Vec<_>>()).unwrap_or_default();
            let mut qs = vec![Query::Term(Term::new_text(id, &s.replace(' ', super::TOKEN_JOIN)))];
            if analyzed.len() == 1 && analyzed[0] != s {
                qs.push(Query::Term(Term::new_text(id, &analyzed[0])));
            }
            any_of(qs)
        }
        EsType::Murmur3 => return Err(shard_err(ctx, format!("Murmur3 fields are not searchable: [{}]", f.name))),
        EsType::Binary => return Err(shard_err(ctx, format!("Binary fields do not support searching"))),
        EsType::Percolator => return Err(shard_err(ctx, format!("Percolator fields are not searchable directly, use a percolate query instead"))),
        EsType::RankVectors => return Err(shard_err(ctx, format!("[rank_vectors] fields do not support term queries"))),
        EsType::Completion => return Err(shard_err(ctx, format!("Field [{}] of type [completion] does not support term queries", f.name))),
        EsType::Wildcard => {
            let Some(id) = f.core_id else { return Ok(Some(Query::MatchNone)) };
            if ci {
                let pat = escape_glob(&s);
                wildcard_query(f, id, &pat, true)
            } else {
                Query::ConstScore { query: Box::new(Query::Term(Term::new_text(id, &s))), score: 1.0, point_set: false }
            }
        }
        _ => return Ok(None),
    }))
}

fn escape_glob(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        if matches!(c, '*' | '?' | '\\') {
            o.push('\\');
        }
        o.push(c);
    }
    o
}

/// `Some(Ok(x))` exact u64, `Some(Err)` a valid number outside u64 /
/// fractional, `None` not a number.
fn parse_u64_exact(v: &Value) -> Option<std::result::Result<u64, ()>> {
    match v {
        // `u64::MAX as f64` is 2^64, one past the largest unsigned long.
        Value::Number(n) => match n.as_u64() {
            Some(x) => Some(Ok(x)),
            None => n.as_f64().map(|x| if x.fract() == 0.0 && x >= 0.0 && x < u64::MAX as f64 { Ok(x as u64) } else { Err(()) }),
        },
        Value::String(s) => {
            let t = s.trim();
            match (t.parse::<u64>(), u64_decimal(t)) {
                (Ok(x), _) => Some(Ok(x)),
                (Err(_), Some((x, false))) => Some(Ok(x)),
                (Err(_), Some((_, true))) => Some(Err(())),
                (Err(_), None) => t.parse::<f64>().ok().map(|x| if x.fract() == 0.0 && x >= 0.0 && x < u64::MAX as f64 { Ok(x as u64) } else { Err(()) }),
            }
        }
        _ => None,
    }
}

/// A plain non-negative decimal string (`123`, `18446744073709551614.5`)
/// read exactly, like ES's BigDecimal: its integer part (when it fits an
/// unsigned long) and whether it has a non-zero fraction.
fn u64_decimal(t: &str) -> Option<(u64, bool)> {
    let (int, frac) = t.split_once('.').unwrap_or((t, ""));
    if int.is_empty() && frac.is_empty() || !int.bytes().all(|b| b.is_ascii_digit()) || !frac.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let i = if int.is_empty() { 0 } else { int.parse::<u64>().ok()? };
    Some((i, frac.bytes().any(|b| b != b'0')))
}

/// Epoch nanos for a date_nanos query bound.
/// A `date_nanos` value (origin of `distance_feature`) in nanos.
pub(crate) fn nanos_value(f: &ResolvedField, v: &Value, ctx: &QueryContext) -> Result<i64> {
    nanos_bound(f, v, None, None, false, ctx)
}

fn nanos_bound(f: &ResolvedField, v: &Value, format: Option<&str>, tz: Option<Tz>, round_up: bool, ctx: &QueryContext) -> Result<i64> {
    if let (Value::String(s), None) = (v, format) {
        if !s.contains("||") && !s.starts_with("now") {
            if let Some(_frac) = rustic_core::fieldtypes::numeric::fraction_nanos(s) {
                if let Ok(n) = super::index::parse_nanos(f, v) {
                    return Ok(n);
                }
            }
        }
    }
    let fmt = match format {
        Some(spec) => DateFormatter::new(spec)?,
        None => f.date_format.clone(),
    };
    let ms = match v {
        Value::Number(n) => {
            let x = n.as_f64().unwrap_or(0.0);
            if fmt.accepts_numbers() {
                fmt.parse_number(x)?
            } else {
                x as i64
            }
        }
        Value::String(s) => parse_date_math(s, &fmt, tz.unwrap_or(Tz::UTC), ctx.now, round_up)
            .map_err(|e| shard_err(ctx, format!("failed to create query: {}", e.reason)).caused_by(EsError::new(400, "parse_exception", e.reason.clone())))?,
        _ => return Err(shard_err(ctx, "failed to create query: date value must be a string or a number")),
    };
    let ms = ms.max(0);
    Ok(ms.saturating_mul(1_000_000).saturating_add(if round_up { 999_999 } else { 0 }))
}

fn range_relation_query(id: u16, q: (u128, u128), rel: rng::Relation) -> Query {
    rustic_core::fieldtypes::range_query::RangeRelationQuery { field: id, bounds: q, relation: rel }.into_query()
}

fn exists(f: &ResolvedField, body: &Value, ctx: &QueryContext) -> Option<Query> {
    let _ = (body, ctx);
    match f.declared {
        EsType::ConstantKeyword => Some(all_or_none(f.constant_value.is_some())),
        EsType::AggregateMetricDouble => {
            // `existsQuery`: the first configured metric's field.
            let first = super::amd_metrics(f).into_iter().next();
            Some(first.and_then(|m| f.aux_id(&m)).map(|id| Query::Exists { field: id }).unwrap_or(Query::MatchNone))
        }
        EsType::SparseVector | EsType::RankFeatures => {
            // `{}` stores an empty marker so exists still matches it.
            Some(f.core_id.map(|id| Query::Exists { field: id }).unwrap_or(Query::MatchNone))
        }
        // No keyword column: docs with shape data.
        EsType::GeoShape => Some(f.core_id.map(|id| Query::Custom(Arc::new(super::geo_shape_query::ShapeExistsQuery { field: id }))).unwrap_or(Query::MatchNone)),
        _ => None,
    }
}

fn bounds_of(o: &Map<String, Value>) -> (Option<(Value, bool)>, Option<(Value, bool)>) {
    let mut lo = None;
    let mut hi = None;
    let inc_lo = o.get("include_lower").and_then(Value::as_bool).unwrap_or(true);
    let inc_hi = o.get("include_upper").and_then(Value::as_bool).unwrap_or(true);
    for (k, v) in o {
        if v.is_null() {
            continue;
        }
        match k.as_str() {
            "gte" => lo = Some((v.clone(), true)),
            "gt" => lo = Some((v.clone(), false)),
            "from" => lo = lo.or(Some((v.clone(), inc_lo))),
            "lte" => hi = Some((v.clone(), true)),
            "lt" => hi = Some((v.clone(), false)),
            "to" => hi = hi.or(Some((v.clone(), inc_hi))),
            _ => {}
        }
    }
    (lo, hi)
}

fn range(f: &ResolvedField, body: &Value, ctx: &QueryContext) -> Result<Option<Query>> {
    let Some((_, spec)) = field_entry(body) else { return Ok(None) };
    let Some(o) = spec.as_object() else { return Ok(None) };
    let boost = boost_of(o);
    let (lo, hi) = bounds_of(o);
    let format = o.get("format").and_then(Value::as_str);
    let tz = match o.get("time_zone").and_then(Value::as_str) {
        Some(z) => Some(Tz::parse_query_option(z)?),
        None => None,
    };
    if let Some(kind) = range_kind(f.declared) {
        let Some(id) = f.core_id else { return Ok(Some(Query::MatchNone)) };
        let rel = match o.get("relation").and_then(Value::as_str) {
            Some(r) => rng::Relation::parse(r).ok_or_else(|| EsError::parsing(format!("[range] query does not support relation [{r}]")))?,
            None => rng::Relation::Intersects,
        };
        let key = |v: &Value, round_up: bool| -> Result<u128> {
            let mut ff = f.clone();
            if let Some(fm) = format {
                ff.date_format = DateFormatter::new(fm)?;
            }
            if kind == rng::RangeKind::Date {
                if let Value::String(s) = v {
                    let ms = parse_date_math(s, &ff.date_format, tz.unwrap_or(Tz::UTC), ctx.now, round_up).map_err(|e| shard_err(ctx, format!("failed to create query: {}", e.reason)))?;
                    return Ok(rng::key_i64(ms));
                }
            }
            super::index::range_key(&ff, kind, v, round_up, ctx.now).map_err(|e| shard_err(ctx, format!("failed to create query: {}", e.reason)))
        };
        let qlo = match &lo {
            Some((v, true)) => key(v, false)?,
            Some((v, false)) => kind.next_up(key(v, true)?),
            None => kind.min_key(),
        };
        let qhi = match &hi {
            Some((v, true)) => key(v, true)?,
            Some((v, false)) => kind.next_down(key(v, false)?),
            None => kind.max_key(),
        };
        if qlo > qhi {
            return Ok(Some(Query::MatchNone));
        }
        return Ok(Some(with_boost(Query::ConstScore { query: Box::new(range_relation_query(id, (qlo, qhi), rel)), score: 1.0, point_set: false }, boost)));
    }
    Ok(Some(match f.declared {
        EsType::ConstantKeyword => {
            let Some(c) = constant_str(f) else { return Ok(Some(Query::MatchNone)) };
            let ok_lo = lo.as_ref().map(|(v, inc)| {
                let b = scalar_string(v);
                if *inc { c >= b } else { c > b }
            });
            let ok_hi = hi.as_ref().map(|(v, inc)| {
                let b = scalar_string(v);
                if *inc { c <= b } else { c < b }
            });
            all_or_none(ok_lo.unwrap_or(true) && ok_hi.unwrap_or(true))
        }
        EsType::DateNanos => {
            let Some(id) = f.core_id else { return Ok(Some(Query::MatchNone)) };
            let lower = match &lo {
                Some((v, true)) => Bound::Included(nanos_bound(f, v, format, tz, false, ctx)?),
                Some((v, false)) => Bound::Excluded(nanos_bound(f, v, format, tz, true, ctx)?),
                None => Bound::Unbounded,
            };
            let upper = match &hi {
                Some((v, true)) => Bound::Included(nanos_bound(f, v, format, tz, true, ctx)?),
                Some((v, false)) => Bound::Excluded(nanos_bound(f, v, format, tz, false, ctx)?),
                None => Bound::Unbounded,
            };
            Query::ConstScore { query: Box::new(Query::RangeI64 { field: id, lower, upper }), score: boost, point_set: false }
        }
        EsType::UnsignedLong => {
            let Some(id) = f.core_id else { return Ok(Some(Query::MatchNone)) };
            let num = |v: &Value| -> Result<f64> {
                match v {
                    Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0)),
                    Value::String(s) => s.trim().parse::<f64>().map_err(|_| shard_err(ctx, format!("failed to create query: For input string: \"{s}\""))),
                    _ => Err(shard_err(ctx, "failed to create query: range bound must be a number")),
                }
            };
            // Exact decimal strings: gte rounds up, gt goes past the integer part.
            let dec = |v: &Value| v.as_str().and_then(|s| u64_decimal(s.trim()));
            let lower = match &lo {
                Some((v, inc)) => match (parse_u64_exact(v), dec(v)) {
                    (Some(Ok(x)), _) => if *inc { Bound::Included(x) } else { Bound::Excluded(x) },
                    (_, Some((i, _))) if i == u64::MAX => return Ok(Some(Query::MatchNone)),
                    (_, Some((i, _))) => Bound::Included(i + 1),
                    _ => {
                        let x = num(v)?;
                        if x < 0.0 {
                            Bound::Unbounded
                        } else if x > u64::MAX as f64 {
                            return Ok(Some(Query::MatchNone));
                        } else {
                            Bound::Included(if *inc { x.ceil() as u64 } else { x.floor() as u64 + 1 })
                        }
                    }
                },
                None => Bound::Unbounded,
            };
            let upper = match &hi {
                Some((v, inc)) => match (parse_u64_exact(v), dec(v)) {
                    (Some(Ok(x)), _) => if *inc { Bound::Included(x) } else { Bound::Excluded(x) },
                    // A fractional bound: lte/lt both stop at the integer part.
                    (_, Some((i, true))) => Bound::Included(i),
                    _ => {
                        let x = num(v)?;
                        if x < 0.0 {
                            return Ok(Some(Query::MatchNone));
                        } else if x >= u64::MAX as f64 {
                            Bound::Unbounded
                        } else {
                            Bound::Included(if *inc { x.floor() as u64 } else { (x.ceil() as u64).saturating_sub(1) })
                        }
                    }
                },
                None => Bound::Unbounded,
            };
            Query::ConstScore { query: Box::new(Query::RangeU64 { field: id, lower, upper }), score: boost, point_set: false }
        }
        EsType::Version => {
            let Some(sid) = f.aux_id("sort") else { return Ok(Some(Query::MatchNone)) };
            let enc = |v: &Value| version::encode(&scalar_string(v)).into_bytes();
            let lower = match &lo {
                Some((v, true)) => Bound::Included(enc(v)),
                Some((v, false)) => Bound::Excluded(enc(v)),
                None => Bound::Unbounded,
            };
            let upper = match &hi {
                Some((v, true)) => Bound::Included(enc(v)),
                Some((v, false)) => Bound::Excluded(enc(v)),
                None => Bound::Unbounded,
            };
            Query::ConstScore { query: Box::new(Query::TermRange { field: sid, lower, upper, rewrite: MultiTermRewrite::ConstantScore }), score: boost, point_set: false }
        }
        EsType::AggregateMetricDouble => {
            let Some(id) = super::aggs_default_metric(f).and_then(|dm| f.aux_id(&dm)) else { return Ok(Some(Query::MatchNone)) };
            let n = |v: &Value| scalar_string(v).trim().parse::<f64>().map_err(|_| shard_err(ctx, "failed to create query: range bound must be a number"));
            let lower = match &lo {
                Some((v, true)) => Bound::Included(n(v)?),
                Some((v, false)) => Bound::Excluded(n(v)?),
                None => Bound::Unbounded,
            };
            let upper = match &hi {
                Some((v, true)) => Bound::Included(n(v)?),
                Some((v, false)) => Bound::Excluded(n(v)?),
                None => Bound::Unbounded,
            };
            Query::ConstScore { query: Box::new(Query::RangeF64 { field: id, lower, upper }), score: boost, point_set: false }
        }
        t if is_shape(t) || is_histogram(t) || matches!(t, EsType::Murmur3 | EsType::Binary | EsType::SparseVector | EsType::RankFeature | EsType::RankFeatures | EsType::Percolator | EsType::Completion | EsType::RankVectors) => {
            return Err(shard_err(ctx, format!("Field [{}] of type [{}] does not support range queries", f.name, t.name())));
        }
        _ => return Ok(None),
    }))
}

fn wildcard_query(f: &ResolvedField, id: u16, pattern: &str, ci: bool) -> Query {
    let grams = ngram::required_ngrams(pattern);
    let cand = match f.aux_id("ngram") {
        Some(g) if !grams.is_empty() => filter_all(grams.iter().map(|t| Query::Term(Term::new_text(g, t))).collect()),
        _ => Some(Query::Exists { field: id }),
    };
    let p = pattern.to_string();
    FilterQuery::constant("wildcard", cand, move |leaf: &LeafReader<'_>, d: DocId| leaf.keyword_values(id, d).iter().any(|v| ngram::wildcard_match(&p, v, ci))).into_query()
}

fn multi_term(qname: &str, f: &ResolvedField, body: &Value, ctx: &QueryContext) -> Result<Option<Query>> {
    let Some((_, spec)) = field_entry(body) else { return Ok(None) };
    let (value, o): (String, Map<String, Value>) = match spec {
        Value::Object(o) => (scalar_string(o.get("value").or_else(|| o.get("wildcard")).unwrap_or(&Value::Null)), o.clone()),
        other => (scalar_string(other), Map::new()),
    };
    let ci = o.get("case_insensitive").and_then(Value::as_bool).unwrap_or(false);
    let boost = boost_of(&o);
    match f.declared {
        EsType::ConstantKeyword => {
            let Some(c) = constant_str(f) else { return Ok(Some(Query::MatchNone)) };
            let (cv, pv) = if ci { (c.to_lowercase(), value.to_lowercase()) } else { (c.clone(), value.clone()) };
            let ok = match qname {
                "prefix" => cv.starts_with(&pv),
                "wildcard" => ngram::wildcard_match(&pv, &cv, false),
                "regexp" => regex::RegexBuilder::new(&format!("^(?:{value})$")).case_insensitive(ci).build().map(|r| r.is_match(&c)).unwrap_or(false),
                _ => {
                    let fz = o.get("fuzziness").map(|v| scalar_string(v)).unwrap_or_else(|| "AUTO".into());
                    let max = match fz.trim().to_uppercase().as_str() {
                        "0" => 0,
                        "1" => 1,
                        "2" => 2,
                        _ => match value.chars().count() {
                            0..=2 => 0,
                            3..=5 => 1,
                            _ => 2,
                        },
                    };
                    edit_distance(&cv, &pv) <= max
                }
            };
            Ok(Some(with_boost(all_or_none(ok), boost)))
        }
        EsType::Wildcard => {
            let Some(id) = f.core_id else { return Ok(Some(Query::MatchNone)) };
            let q = match qname {
                "wildcard" => wildcard_query(f, id, &value, ci),
                "prefix" => wildcard_query(f, id, &format!("{}*", escape_glob(&value)), ci),
                "regexp" => {
                    let re = regex::RegexBuilder::new(&format!("^(?:{value})$"))
                        .case_insensitive(ci)
                        .build()
                        .map_err(|e| shard_err(ctx, format!("failed to create query: {e}")))?;
                    let grams = if ci { Vec::new() } else { ngram::regex_required_ngrams(&value) };
                    let cand = match f.aux_id("ngram") {
                        Some(g) if !grams.is_empty() => filter_all(grams.iter().map(|t| Query::Term(Term::new_text(g, t))).collect()),
                        _ => Some(Query::Exists { field: id }),
                    };
                    FilterQuery::constant("regexp", cand, move |leaf: &LeafReader<'_>, d: DocId| leaf.keyword_values(id, d).iter().any(|v| re.is_match(v))).into_query()
                }
                _ => return Ok(None),
            };
            Ok(Some(Query::ConstScore { query: Box::new(q), score: boost, point_set: false }))
        }
        t if range_kind(t).is_some() || is_shape(t) || is_histogram(t) || matches!(t, EsType::Murmur3 | EsType::Binary | EsType::SparseVector | EsType::RankFeature | EsType::RankFeatures | EsType::DateNanos | EsType::AggregateMetricDouble) => {
            if qname == "fuzzy" && t == EsType::DateNanos {
                return Ok(None);
            }
            Err(shard_err(ctx, format!("Field [{}] of type [{}] does not support {qname} queries", f.name, t.name())))
        }
        _ => Ok(None),
    }
}

fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + usize::from(a[i - 1] != b[j - 1]));
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Queries on `<flattened>.<key>` that the normalizer trick cannot express.
fn flattened_sub(name: &str, f: &ResolvedField, body: &Value, ctx: &QueryContext) -> Result<Option<Query>> {
    let key = f.flattened_key().unwrap_or("");
    let Some(id) = f.core_id else { return Ok(Some(Query::MatchNone)) };
    let prefix = rustic_core::fieldtypes::flattened::key_prefix(key);
    match name {
        "exists" => Ok(Some(Query::Prefix { field: id, prefix: prefix.into_bytes(), max_expansions: usize::MAX, case_insensitive: false, rewrite: MultiTermRewrite::ConstantScore })),
        "range" => {
            let Some((_, Value::Object(o))) = field_entry(body) else { return Ok(None) };
            let (lo, hi) = bounds_of(o);
            let b = |v: &Value| format!("{prefix}{}", scalar_string(v)).into_bytes();
            let lower = match &lo {
                Some((v, true)) => Bound::Included(b(v)),
                Some((v, false)) => Bound::Excluded(b(v)),
                None => Bound::Included(prefix.clone().into_bytes()),
            };
            let upper = match &hi {
                Some((v, true)) => Bound::Included(b(v)),
                Some((v, false)) => Bound::Excluded(b(v)),
                None => Bound::Excluded(format!("{key}\u{1}").into_bytes()),
            };
            Ok(Some(with_boost(Query::ConstScore { query: Box::new(Query::TermRange { field: id, lower, upper, rewrite: MultiTermRewrite::ConstantScore }), score: 1.0, point_set: false }, boost_of(o))))
        }
        "regexp" => {
            let Some((_, spec)) = field_entry(body) else { return Ok(None) };
            let (value, o) = match spec {
                Value::Object(o) => (scalar_string(o.get("value").unwrap_or(&Value::Null)), o.clone()),
                other => (scalar_string(other), Map::new()),
            };
            let ci = o.get("case_insensitive").and_then(Value::as_bool).unwrap_or(false);
            let re = regex::RegexBuilder::new(&format!("^(?:{value})$")).case_insensitive(ci).build().map_err(|e| shard_err(ctx, format!("failed to create query: {e}")))?;
            let p = prefix.clone();
            Ok(Some(
                FilterQuery::constant("regexp", Some(Query::Prefix { field: id, prefix: p.clone().into_bytes(), max_expansions: usize::MAX, case_insensitive: false, rewrite: MultiTermRewrite::ConstantScore }), move |leaf: &LeafReader<'_>, d: DocId| {
                    leaf.keyword_values(id, d).iter().filter_map(|v| v.strip_prefix(&p)).any(|v| re.is_match(v))
                })
                .into_query(),
            ))
        }
        _ => Ok(None),
    }
}

/// A doc scores `Σ query_weight × doc_weight` over shared tokens: an OR
/// of weighted feature terms run as MaxScore top-k (Lucene `FeatureQuery`
/// SHOULD clauses; see [`rustic_core::query::feature`]).
fn feature_scoring(id: u16, tokens: Vec<(String, f32)>, name: &'static str) -> Query {
    Query::Custom(Arc::new(rustic_core::query::feature::FeatureQuery::new(id, tokens, name)))
}

/// Docs that have the feature `token` (constant score).
fn feature_present(id: u16, token: &str) -> Query {
    Query::TermSet { field: id, terms: vec![token.as_bytes().to_vec()], boost: 1.0 }
}

fn sparse_vector_query(name: &str, body: &Value, ctx: &QueryContext) -> Result<Query> {
    let o = body.as_object().ok_or_else(|| EsError::parsing(format!("[{name}] query malformed")))?;
    // text_expansion / weighted_tokens: {"<field>": {...}}.
    let (field, spec) = if name == "sparse_vector" {
        (o.get("field").and_then(Value::as_str).ok_or_else(|| EsError::parsing("[sparse_vector] requires a [field]"))?.to_string(), o.clone())
    } else {
        let (k, v) = field_entry(body).ok_or_else(|| EsError::parsing(format!("[{name}] query requires a field")))?;
        (k.clone(), v.as_object().cloned().unwrap_or_default())
    };
    let boost = boost_of(&spec);
    let vector = spec.get("query_vector").or_else(|| spec.get("tokens"));
    // `SparseVectorQueryBuilder.doRewrite`: query text needs an inference id
    // (a semantic_text field supplies its own).
    let semantic = ctx.lookup.get(&field).is_some_and(|f| f.declared.is_semantic());
    if name == "sparse_vector" && vector.is_none() && spec.get("inference_id").is_none() && !semantic {
        return Err(EsError::illegal_argument("inference_id required to perform vector search on query string"));
    }
    // `TextExpansionQueryBuilder.doRewrite`: the model runs before the field
    // is looked at; a dense model's results are refused.
    if name == "text_expansion" && vector.is_none() {
        if let Some(id) = spec.get("model_id").and_then(Value::as_str) {
            let dense = crate::inference::current().and_then(|s| s.get_endpoint(id)).is_some_and(|ep| ep.task_type.is_dense());
            if dense {
                return Err(EsError::illegal_argument(format!("expected a result of type [text_expansion_result] received [text_embedding_result]. Is [{id}] a compatible model?")));
            }
        }
    }
    let Some(f) = ctx.lookup.get(&field).cloned() else { return Ok(Query::MatchNone) };
    if f.declared.is_semantic() && name == "sparse_vector" && crate::inference::current().is_some() {
        return Ok(with_boost(crate::inference::semantic_query::sparse_vector(&f, &spec, ctx)?, boost));
    }
    if f.declared.is_semantic() {
        let text = spec.get("query").or_else(|| spec.get("model_text")).cloned().unwrap_or(Value::Null);
        return bypass(|| parse_query(&json!({"match": {field.clone(): {"query": text, "boost": boost}}}), ctx));
    }
    if !matches!(f.declared, EsType::SparseVector | EsType::RankFeatures) {
        if name == "text_expansion" || name == "weighted_tokens" {
            return Err(shard_err(ctx, format!("[{}] is not an appropriate field type for this query. Allowed field types are [rank_features, sparse_vector].", f.declared.name())));
        }
        return Err(shard_err(ctx, format!("field [{field}] must be type [sparse_vector] but is type [{}]", f.declared.name())));
    }
    // `{"token": weight, ..}` or `[{"token": weight}, ..]` (weighted_tokens).
    let vector_map: Option<Map<String, Value>> = match vector {
        Some(Value::Object(m)) => Some(m.clone()),
        Some(Value::Array(a)) => Some(a.iter().filter_map(Value::as_object).flat_map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone()))).collect()),
        _ => None,
    };
    let tokens: Vec<(String, f32)> = match vector_map.as_ref() {
        Some(vector) => vector.iter().filter_map(|(k, v)| v.as_f64().map(|w| (k.clone(), w as f32))).collect(),
        None => {
            // The query text goes through the `inference_id` endpoint (or
            // `_ml` trained model; `model_id` for `text_expansion`).
            let id = spec.get("inference_id").or_else(|| spec.get("model_id")).and_then(Value::as_str).unwrap_or_default().to_string();
            let text = spec.get("query").or_else(|| spec.get("model_text")).map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())).unwrap_or_default();
            let svc = crate::inference::current().ok_or_else(|| EsError::resource_not_found(format!("Inference endpoint not found [{id}]")))?;
            let emb = if svc.get_endpoint(&id).is_some() {
                svc.embed_typed(&id, &[text], "search")?
            } else {
                match svc.embed_trained_model(&id, &text)? {
                    Some(e) => e,
                    None => return Err(EsError::resource_not_found(format!("Inference endpoint not found [{id}]"))),
                }
            };
            match emb {
                crate::inference::Embedded::Sparse(mut v) => v.pop().unwrap_or_default().into_iter().filter_map(|(k, w)| w.as_f64().map(|w| (k, w as f32))).collect(),
                crate::inference::Embedded::Dense(_) => return Err(EsError::illegal_argument(format!("Inference endpoint [{id}] is not a sparse_embedding endpoint"))),
            }
        }
    };
    let Some(id) = f.core_id else { return Ok(Query::MatchNone) };
    // Token pruning: the query's `prune` / `pruning_config`, else (for the
    // `sparse_vector` query) the field's `index_options`, which default to
    // pruning with the default config.
    let query_config = spec.get("pruning_config").filter(|v| !v.is_null()).map(pruning_config).transpose()?;
    let (prune, config) = if name == "sparse_vector" {
        let mut prune = spec.get("prune").and_then(Value::as_bool);
        let mut config = query_config;
        if f.declared == EsType::SparseVector {
            let io = f.ext.as_ref().and_then(|e| e.params.get("index_options").cloned());
            let (ip, ic) = match io.as_ref().and_then(Value::as_object) {
                None => (Some(true), Some(PruningConfig::default())),
                Some(o) => (o.get("prune").and_then(Value::as_bool), o.get("pruning_config").map(pruning_config).transpose()?),
            };
            if prune.is_none() {
                prune = ip;
            }
            if config.is_none() {
                config = ic;
            }
        }
        (prune == Some(true), config.unwrap_or_default())
    } else {
        (query_config.is_some(), query_config.unwrap_or_default())
    };
    if prune {
        return match prune_tokens(ctx, id, &tokens, &config) {
            Some(kept) => Ok(with_boost(feature_scoring(id, kept, "sparse_vector"), boost)),
            None => Ok(Query::MatchNone),
        };
    }
    Ok(with_boost(feature_scoring(id, tokens, "sparse_vector"), boost))
}

/// `TokenPruningConfig`.
pub(crate) struct PruningConfig {
    pub(crate) freq_ratio: f32,
    pub(crate) weight: f32,
    pub(crate) only_pruned: bool,
}

impl Default for PruningConfig {
    fn default() -> Self {
        PruningConfig { freq_ratio: 5.0, weight: 0.4, only_pruned: false }
    }
}

pub(crate) fn pruning_config(v: &Value) -> Result<PruningConfig> {
    let num = |k: &str, d: f32| v.get(k).and_then(Value::as_f64).map(|x| x as f32).unwrap_or(d);
    let c = PruningConfig {
        freq_ratio: num("tokens_freq_ratio_threshold", 5.0),
        weight: num("tokens_weight_threshold", 0.4),
        only_pruned: v.get("only_score_pruned_tokens").and_then(Value::as_bool).unwrap_or(false),
    };
    if !(1.0..=100.0).contains(&c.freq_ratio) {
        return Err(EsError::illegal_argument(format!("[tokens_freq_ratio_threshold] must be between [1] and [100], got {}", crate::fieldtypes::params::jdouble(c.freq_ratio as f64))));
    }
    if !(0.0..=1.0).contains(&c.weight) {
        return Err(EsError::illegal_argument("[tokens_weight_threshold] must be between 0 and 1"));
    }
    Ok(c)
}

/// `WeightedTokensUtils.queryBuilderWithPrunedTokens`: drop the tokens that
/// are frequent in the field (relative to the average token) and light in
/// the query; `None` when the field holds no tokens.
fn prune_tokens(ctx: &QueryContext, id: u16, tokens: &[(String, f32)], c: &PruningConfig) -> Option<Vec<(String, f32)>> {
    // Field statistics come from each leaf's cached term stats and the
    // query tokens' doc freqs from term lookups: no walk of the field's
    // vocabulary per query (it made pruned queries ~10x slower than
    // unpruned ones, and a pruned rescore paid it twice).
    let mut doc_count = 0u64;
    let mut sum_doc_freq = 0u64;
    let mut max_unique = 0u64;
    let mut texts: Vec<&str> = tokens.iter().map(|(t, _)| t.as_str()).collect();
    texts.sort_unstable();
    texts.dedup();
    let keys: Vec<&[u8]> = texts.iter().map(|t| t.as_bytes()).collect();
    let mut freqs = vec![0u64; texts.len()];
    for leaf in ctx.leaves {
        let (unique, sum_df) = leaf.field_term_stats(id);
        if unique == 0 {
            continue;
        }
        doc_count += leaf.field_doc_count(id);
        sum_doc_freq += sum_df;
        max_unique = max_unique.max(unique);
        leaf.doc_freqs_sorted(id, &keys, |i, df| freqs[i] += df as u64);
    }
    let dfs: std::collections::HashMap<&str, u64> = texts.iter().copied().zip(freqs).collect();
    if max_unique == 0 || doc_count == 0 {
        return None;
    }
    let avg = sum_doc_freq as f32 / doc_count as f32 / max_unique as f32;
    let best = tokens.iter().map(|(_, w)| *w).fold(0f32, f32::max);
    let kept = tokens
        .iter()
        .filter(|(t, w)| {
            let df = dfs.get(t.as_str()).copied().unwrap_or(0);
            let keep = df > 0 && ((df as f32 / doc_count as f32) < c.freq_ratio * avg || *w > c.weight * best);
            keep ^ c.only_pruned
        })
        .cloned()
        .collect();
    Some(kept)
}

/// Real semantic scoring for a `semantic_text` field (`None` = no inference
/// service attached or the field has no inference companion: lexical fallback).
fn json_kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn semantic_text_query(f: &ResolvedField, q: &Value, boost: f32, ctx: &QueryContext) -> Result<Option<Query>> {
    if crate::inference::current().is_none() || f.aux_id("inference").is_none() {
        return Ok(None);
    }
    let text = match q {
        Value::String(s) => s.clone(),
        Value::Null => return Ok(None),
        other => {
            return Err(EsError::new(
                400,
                "status_exception",
                format!("Invalid format for field [{}]. Expected a string but got a {}.", f.name, json_kind(other)),
            ))
        }
    };
    Ok(Some(with_boost(crate::inference::semantic_query::semantic_query(f, &text, Some(ctx))?, boost)))
}

/// Semantic query failures are shard failures (`SearchPhaseExecutionException`
/// with the `IllegalArgumentException` as its cause).
fn semantic_shard_err(ctx: &QueryContext, e: EsError) -> EsError {
    if e.kind == "illegal_argument_exception" {
        EsError::search_phase(ctx.index_name, e)
    } else {
        e
    }
}

fn semantic(o: &Map<String, Value>, ctx: &QueryContext) -> Result<Query> {
    semantic_inner(o, ctx).map_err(|e| semantic_shard_err(ctx, e))
}

fn semantic_inner(o: &Map<String, Value>, ctx: &QueryContext) -> Result<Query> {
    let field = o.get("field").and_then(Value::as_str).ok_or_else(|| EsError::parsing("[semantic] query requires a [field]"))?;
    let q = o.get("query").cloned().ok_or_else(|| EsError::parsing("[semantic] query requires a [query]"))?;
    let lenient = o.get("lenient").and_then(Value::as_bool).unwrap_or(false);
    match ctx.lookup.get(field) {
        None => Ok(Query::MatchNone),
        Some(f) if f.declared.is_semantic() => {
            let boost = boost_of(o);
            if let Some(sq) = semantic_text_query(&f, &q, boost, ctx)? {
                return Ok(sq);
            }
            bypass(|| parse_query(&json!({"match": {field: {"query": q, "boost": boost}}}), ctx))
        }
        Some(_) if lenient => Ok(Query::MatchNone),
        Some(f) => Err(EsError::illegal_argument(format!("Field [{field}] of type [{}] does not support semantic queries", f.declared.name()))),
    }
}

/// `rank_feature` on `<rank_features>.<feature>`.
fn rank_features_sub(name: &str, field: &str, body: &Value, ctx: &QueryContext) -> Result<Option<Query>> {
    let Some((root, tok)) = field.rsplit_once('.') else { return Ok(None) };
    let Some(f) = ctx.lookup.fields.get(root) else { return Ok(None) };
    if !matches!(f.declared, EsType::RankFeatures | EsType::SparseVector) {
        return Ok(None);
    }
    let Some(id) = f.core_id else { return Ok(Some(Query::MatchNone)) };
    match name {
        "exists" => Ok(Some(feature_present(id, tok))),
        "term" => {
            let _ = body;
            Ok(Some(feature_scoring(id, vec![(tok.to_string(), 1.0)], "term")))
        }
        _ => Ok(None),
    }
}

fn rank_feature(o: &Map<String, Value>, ctx: &QueryContext) -> Result<Query> {
    let field = o.get("field").and_then(Value::as_str).ok_or_else(|| EsError::illegal_argument("Required [field]"))?;
    let boost = boost_of(o);
    // (core id, Some(token) for rank_features)
    let (id, tok): (u16, Option<String>) = match ctx.lookup.get(field) {
        Some(f) if f.declared == EsType::RankFeature => match f.core_id {
            Some(id) => (id, None),
            None => return Ok(Query::MatchNone),
        },
        Some(f) => return Err(shard_err(ctx, format!("failed to create query: [rank_feature] query only works on [rank_feature] fields and features of [rank_features] fields, not [{}]", f.declared.name()))),
        None => match field.rsplit_once('.').and_then(|(r, t)| ctx.lookup.fields.get(r).map(|f| (f.clone(), t))) {
            Some((f, t)) if matches!(f.declared, EsType::RankFeatures) => match f.core_id {
                Some(id) => (id, Some(t.to_string())),
                None => return Ok(Query::MatchNone),
            },
            _ => return Ok(Query::MatchNone),
        },
    };
    let read = {
        let tok = tok.clone();
        move |leaf: &LeafReader<'_>, d: DocId| -> Option<f32> {
            match &tok {
                None => leaf.f64_values(id, d).first().map(|x| *x as f32),
                Some(t) => leaf.keyword_values(id, d).iter().filter_map(|v| features::decode(v)).find(|(k, _)| k == t).map(|(_, w)| w),
            }
        }
    };
    let func_keys: Vec<&str> = ["saturation", "log", "sigmoid", "linear"].into_iter().filter(|k| o.contains_key(*k)).collect();
    if func_keys.len() > 1 {
        return Err(EsError::x_content_parse("Failed to build [feature] after last required field arrived"));
    }
    let pivot_default = || -> f32 {
        // ES: approximate geometric mean of the feature over the index.
        let mut sum_ln = 0.0f64;
        let mut n = 0u64;
        for leaf in ctx.leaves {
            let live = leaf.live_docs();
            for d in 0..leaf.num_docs() {
                if live(d) {
                    if let Some(v) = read(leaf, d) {
                        sum_ln += (v as f64).ln();
                        n += 1;
                    }
                }
            }
        }
        if n == 0 {
            1.0
        } else {
            (sum_ln / n as f64).exp() as f32
        }
    };
    let num = |v: Option<&Value>, k: &str| -> Result<Option<f32>> {
        match v {
            None => Ok(None),
            Some(x) => x.as_f64().map(|y| Some(y as f32)).ok_or_else(|| EsError::parsing(format!("[{k}] must be a number"))),
        }
    };
    let func = match func_keys.first().copied() {
        None => FeatureFunction::Saturation { pivot: pivot_default() },
        Some("saturation") => FeatureFunction::Saturation { pivot: match num(o["saturation"].get("pivot"), "pivot")? {
            Some(p) => p,
            None => pivot_default(),
        } },
        Some("log") => FeatureFunction::Log { scaling_factor: num(o["log"].get("scaling_factor"), "scaling_factor")?.ok_or_else(|| EsError::parsing("[log] requires [scaling_factor]"))? },
        Some("sigmoid") => FeatureFunction::Sigmoid {
            pivot: num(o["sigmoid"].get("pivot"), "pivot")?.ok_or_else(|| EsError::parsing("[sigmoid] requires [pivot]"))?,
            exponent: num(o["sigmoid"].get("exponent"), "exponent")?.ok_or_else(|| EsError::parsing("[sigmoid] requires [exponent]"))?,
        },
        _ => FeatureFunction::Linear,
    };
    let cand = match &tok {
        None => Query::MatchAll,
        Some(t) => feature_present(id, t),
    };
    let q = FilterQuery::new("rank_feature", Some(cand), Arc::new(move |leaf: &LeafReader<'_>, d: DocId| read(leaf, d).map(|v| func.score(v) * boost))).into_query();
    Ok(q)
}

/// ES-shaped error for a geometry that failed to parse.
fn shape_parse_error(msg: &str, wkt: bool) -> EsError {
    if wkt {
        return EsError::illegal_argument(msg.to_string());
    }
    let build = || EsError::x_content_parse("Failed to build [geojson] after last required field arrived");
    let parse_exc = |m: &str| EsError::new(400, "parse_exception", m.to_string());
    if msg == "shape type not included" {
        return EsError::illegal_argument("Required [type]");
    }
    if matches!(msg, "coordinates not included" | "radius is not specified" | "geometries not included") {
        return build().caused_by(parse_exc(msg));
    }
    if msg.starts_with("invalid number of points [") {
        return build().caused_by(parse_exc(msg));
    }
    if msg.starts_with("invalid number of points in LineString") {
        return build().caused_by(EsError::new(400, "exception", "not enough points to build a line"));
    }
    if msg.contains("numeric value expected") || msg.contains("geo coordinates must be numbers") {
        return EsError::x_content_parse("[1:1] [geojson] failed to parse field [coordinates]").caused_by(parse_exc("geo coordinates must be numbers"));
    }
    build()
}

fn shape_query(qname: &str, o: &Map<String, Value>, ctx: &QueryContext) -> Result<Query> {
    let ignore_unmapped = o.get("ignore_unmapped").and_then(Value::as_bool).unwrap_or(false);
    // Unlike most queries a geo_shape scores 0 unless a `boost` is given.
    let score = o.get("boost").and_then(Value::as_f64).map(|b| b as f32).unwrap_or(0.0);
    let (field, spec) = o
        .iter()
        .filter(|(k, _)| !matches!(k.as_str(), "boost" | "_name" | "ignore_unmapped"))
        .last()
        .ok_or_else(|| EsError::parsing(format!("[{qname}] query requires a field")))?;
    let spec = spec.as_object().ok_or_else(|| EsError::parsing(format!("[{qname}] query malformed")))?;
    for k in spec.keys() {
        if !matches!(k.as_str(), "shape" | "indexed_shape" | "relation") {
            return Err(EsError::parsing(format!("query does not support [{k}]")));
        }
    }
    let rel = match spec.get("relation") {
        None => shape::SpatialRelation::Intersects,
        Some(Value::String(r)) => shape::SpatialRelation::parse(r).ok_or_else(|| EsError::parsing(format!("Unknown shape operation [{r} ]")))?,
        Some(_) => return Err(EsError::illegal_argument("Expected text but found VALUE_NULL")),
    };
    let geo = qname == "geo_shape";
    let shape_v = match (spec.get("shape"), spec.get("indexed_shape")) {
        (Some(s), _) => s.clone(),
        (None, Some(Value::Object(is))) => {
            let Some(id) = is.get("id").map(scalar_string) else {
                return Err(EsError::illegal_argument("either shape or indexedShapeId is required"));
            };
            let index = is.get("index").and_then(Value::as_str).unwrap_or(ctx.index_name);
            let explicit_path = is.get("path").and_then(Value::as_str);
            let path = explicit_path.unwrap_or("shape");
            let routing = is.get("routing").and_then(Value::as_str);
            let lk = ctx.terms_lookup.ok_or_else(|| shard_err(ctx, "indexed shapes are not available here"))?;
            let vals = lk(index, &id, path, routing)?;
            match vals.into_iter().next() {
                Some(v) => v,
                None if explicit_path.is_some() => return Err(EsError::illegal_argument(format!("Shape with ID [{id}] not found"))),
                None => return Err(EsError::new(500, "illegal_state_exception", format!("Shape with name [{id}] found but missing {path} field"))),
            }
        }
        _ => return Err(EsError::illegal_argument("either shape or indexedShapeId is required")),
    };
    let wkt = shape_v.is_string();
    let qg = if geo {
        shape::parse_query_shape(&shape_v)
    } else {
        shape::parse(&shape_v, &shape::ParseOpts { geo: false, coerce: true, ignore_z: true })
    }
    .map_err(|e| shape_parse_error(&e, wkt))?;
    // A `geo_shape` runtime field: its values related as one shape doc value.
    if geo {
        if let Some(q) = crate::script::search::runtime_shape_query(field, &qg, super::geo_shape_query::circle_of(&shape_v), rel, ctx)? {
            return Ok(Query::ConstScore { query: Box::new(q), score, point_set: false });
        }
    }
    let Some(f) = ctx.lookup.get(field).cloned() else {
        if ignore_unmapped {
            return Ok(Query::MatchNone);
        }
        return Err(shard_err(ctx, format!("failed to find type for field [{field}]")));
    };
    let ok_type = if geo { matches!(f.declared, EsType::GeoShape | EsType::GeoPoint) } else { matches!(f.declared, EsType::Shape | EsType::Point) };
    if !ok_type {
        return Err(shard_err(ctx, format!("Field [{field}] is of unsupported type [{}] for [{qname}] query", f.declared.name())));
    }
    // Lucene tests a circle against indexed points by spherical distance.
    // Polygonizing it creates an inscribed approximation and loses points near
    // the boundary (including centimetre-wide distance rings).
    if geo && f.declared == EsType::GeoPoint && rel != shape::SpatialRelation::Contains
        && shape_v.get("type").and_then(Value::as_str).is_some_and(|t| t.eq_ignore_ascii_case("circle"))
    {
        if let (Some(gid), Some((lat, lon)), Some(meters)) = (f.geo_id,
            shape_v.get("coordinates").and_then(values::parse_geo_point),
            shape_v.get("radius").and_then(values::parse_distance))
        {
            // ES LuceneGeometriesUtils uses LATLON_QUANTIZER for geo_shape
            // query coordinates. geo_distance intentionally keeps its raw centre.
            let (lat, lon) = rustic_core::geo::quantize(lat, lon);
            let distance = Query::GeoDistance { field: gid, lat, lon, meters };
            let q = match rel {
                shape::SpatialRelation::Intersects => distance,
                shape::SpatialRelation::Within => {
                    let circle = rustic_core::geo::GeoShape::Distance { lat, lon, meters };
                    FilterQuery::constant("geo_shape", Some(distance), move |leaf: &LeafReader<'_>, doc| {
                        let points = leaf.geo_values(gid, doc);
                        !points.is_empty() && points.iter().all(|&(lat, lon)| circle.contains(lat, lon))
                    }).into_query()
                }
                _ => Query::Bool {
                    must: vec![], should: vec![], minimum_should_match: MinShouldMatch::Count(0),
                    filter: vec![Query::GeoBoundingBox { field: gid, top: 90.0, bottom: -90.0, left: -180.0, right: 180.0 }],
                    must_not: vec![distance],
                },
            };
            return Ok(Query::ConstScore { query: Box::new(q), score, point_set: false });
        }
    }
    // Geographic shapes are cut at the dateline and snapped to the Lucene grid
    // (as indexed) before relating.
    let qn = if geo { shape::normalize_geo(&qg) } else { qg.clone() };
    let qp = Arc::new(if geo { shape::Prims::of(&qn).quantized() } else { shape::Prims::of(&qn) });
    let qbox = qn.bbox();
    let q = if f.declared == EsType::GeoPoint && geo && rel != shape::SpatialRelation::Contains {
        // Points relate one at a time: walk the KD tree (leaves outside the
        // query's box skipped, leaves its area covers taken whole).
        let Some(gid) = f.geo_id else { return Ok(Query::MatchNone) };
        Query::Custom(Arc::new(GeoPointShapeQuery { field: gid, query: qp.clone(), rel }))
    } else if f.declared == EsType::GeoPoint && geo && qp.lines.iter().chain(qp.polys.iter().flatten()).any(|l| l.windows(2).any(|w| w[0] != w[1])) {
        // Points never contain a line or an area: an edge's midpoint is
        // never one of finitely many points (see `shape::within`).
        Query::MatchNone
    } else if f.declared == EsType::GeoPoint {
        let Some(gid) = f.geo_id else { return Ok(Query::MatchNone) };
        FilterQuery::constant("geo_shape", Some(Query::Exists { field: gid }), move |leaf: &LeafReader<'_>, d: DocId| {
            let pts = leaf.geo_values(gid, d);
            if pts.is_empty() {
                return false;
            }
            let dp = shape::Prims { points: pts.iter().map(|(lat, lon)| (*lon, *lat)).collect(), ..Default::default() }.quantized();
            shape::relate(&dp, &qp, rel)
        })
        .into_query()
    } else if geo {
        // `geo_shape` fields: ES's `LatLonShape` query over the segments'
        // triangle trees (`GeoShapeQueryable.geoShapeQuery`: the geometry
        // normalized and quantized to the index's grid).
        let Some(id) = f.core_id else { return Ok(Query::MatchNone) };
        // A circle stays a circle (Lucene `Circle2D`, its centre quantized).
        let geoms = match super::geo_shape_query::circle_of(&shape_v) {
            Some(c) => vec![c],
            None => rustic_core::fieldtypes::geo2d::query_geometries(&qn, rel == shape::SpatialRelation::Within)
                .map_err(|e| shard_err(ctx, format!("Exception creating query on Field [{field}] {e}")))?,
        };
        if geoms.is_empty() {
            return Ok(Query::ConstScore { query: Box::new(Query::MatchNone), score, point_set: false });
        }
        let sq = rustic_core::fieldtypes::geo2d::ShapeQuery::new(&geoms, super::geo_shape_query::query_relation(rel));
        Query::Custom(Arc::new(super::geo_shape_query::ShapeTreeQuery { field: id, query: Arc::new(sq), full_hits: Default::default(), cache_key: Default::default(), noted: Default::default() }))
    } else {
        // Cartesian `shape` fields: each doc's encoded values related.
        let Some(id) = f.core_id else { return Ok(Query::MatchNone) };
        let exact_qp = qp.clone();
        FilterQuery::constant("shape", Some(Query::Exists { field: id }), move |leaf: &LeafReader<'_>, d: DocId| {
            let owned = leaf.keyword_values(id, d);
            let mut doc_shapes: Vec<shape::Geometry> = Vec::with_capacity(owned.len());
            for v in &owned {
                if let Some(g) = shape::decode(v) {
                    doc_shapes.push(g);
                }
            }
            if owned.is_empty() {
                return false;
            }
            if doc_shapes.is_empty() {
                return rel == shape::SpatialRelation::Disjoint;
            }
            let dp = shape::Prims::of(&shape::Geometry::Collection(doc_shapes));
            shape::relate(&dp, &exact_qp, rel)
        })
        .into_query()
    };
    let _ = values::ipv4;
    Ok(Query::ConstScore { query: Box::new(q), score, point_set: false })
}


/// `geo_shape` (intersects / within / disjoint) over a `geo_point` field:
/// per-point relations evaluated over the segment's KD tree. A leaf whose
/// box misses the query's bounding box is skipped; one the query's area
/// covers matches whole; the others test their points. Same results as
/// relating each document's points ([`shape::relate`]).
#[derive(Debug)]
pub(crate) struct GeoPointShapeQuery {
    field: u16,
    query: Arc<shape::Prims>,
    rel: shape::SpatialRelation,
}

impl GeoPointShapeQuery {
    /// Docs with at least one point passing the per-point test
    /// (`point_within` for `within`, else `point_intersects`).
    fn scan(&self, leaf: &LeafReader<'_>) -> rustic_core::geo::DocHits {
        use rustic_core::geo::{decode_lat, decode_lon, DocHits, GeoPointEntry, HitCollector, Relation};
        let q = &*self.query;
        let Some(qbox) = shape::prims_bbox(q) else { return DocHits::Sorted(Vec::new()) };
        let within = self.rel == shape::SpatialRelation::Within;
        let mut hits = HitCollector::new(leaf.num_docs());
        let mut last = DocId::MAX;
        let mut test = |p: &GeoPointEntry, inside: bool| {
            if p.doc == last {
                return;
            }
            let pt = (decode_lon(p.lon), decode_lat(p.lat));
            if inside || if within { shape::point_within(pt, q) } else { shape::point_intersects(pt, q) } {
                last = p.doc;
                hits.push(p.doc);
            }
        };
        let eps = 1e-9;
        match leaf {
            LeafReader::Segment(seg) => {
                let Some(g) = seg.geo_index(self.field) else { return DocHits::Sorted(Vec::new()) };
                g.visit(
                    |min_lat, max_lat, min_lon, max_lon| {
                        if min_lon > qbox.2 + eps || max_lon < qbox.0 - eps || min_lat > qbox.3 + eps || max_lat < qbox.1 - eps {
                            Relation::Outside
                        } else if shape::box_covered(q, min_lon, min_lat, max_lon, max_lat) {
                            Relation::Inside
                        } else {
                            Relation::Crosses
                        }
                    },
                    &mut test,
                );
            }
            LeafReader::Memtable(_, ..) => {
                for p in rustic_core::query::GeoDv::open(leaf, self.field).iter() {
                    test(&p, false);
                }
            }
        }
        hits.finish()
    }
}

/// A shape relation of single points ([`GeoPointShapeQuery`]'s per-point
/// test with its KD-leaf classification).
#[derive(Debug)]
struct PointShapePred {
    query: Arc<shape::Prims>,
    qbox: (f64, f64, f64, f64),
    within: bool,
    /// Every vertex of `query`, for [`shape::point_intersects`]'s
    /// vertex check as one loop.
    vertices: Vec<shape::Pt>,
}

impl rustic_core::geo::PointPredicate for PointShapePred {
    fn classify(&self, min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> rustic_core::geo::Relation {
        use rustic_core::geo::Relation;
        let (q, b, eps) = (&*self.query, self.qbox, 1e-9);
        if min_lon > b.2 + eps || max_lon < b.0 - eps || min_lat > b.3 + eps || max_lat < b.1 - eps {
            Relation::Outside
        } else if shape::box_covered(q, min_lon, min_lat, max_lon, max_lat) {
            Relation::Inside
        } else if shape::box_disjoint(q, min_lon, min_lat, max_lon, max_lat) {
            // Inside the shape's box but away from the shape (a grid cell
            // in a concave or slanted polygon's corner).
            Relation::Outside
        } else {
            Relation::Crosses
        }
    }

    fn test(&self, lat: f64, lon: f64) -> bool {
        let p = (lon, lat);
        if self.within {
            return shape::point_within(p, &self.query);
        }
        // `shape::point_intersects`, its vertices flattened once.
        shape::point_in(p, &self.query) || self.vertices.iter().any(|v| (v.0 - p.0).abs() <= shape::EPS && (v.1 - p.1).abs() <= shape::EPS)
    }
}

/// A shape relation of single points as a point predicate (for a doc of
/// one point: its relation), points tested through a cell grid over the
/// shape's box. `None` for `contains` and shapes without a box.
fn point_shape_pred(query: &Arc<shape::Prims>, rel: shape::SpatialRelation) -> Option<rustic_core::geo::GeoPred> {
    use rustic_core::geo::GeoPred;
    let qbox = shape::prims_bbox(query)?;
    // Grids shared by every leaf and repeat of the query shape (unless it
    // is huge).
    let vertices = query.points.len() + query.lines.iter().map(Vec::len).sum::<usize>() + query.polys.iter().flatten().map(Vec::len).sum::<usize>();
    let leaf = |within| {
        let verts = query.points.iter().chain(query.lines.iter().flatten()).chain(query.polys.iter().flatten().flatten()).copied().collect();
        let (p, b) = (PointShapePred { query: query.clone(), qbox, within, vertices: verts }, (qbox.1, qbox.3, qbox.0, qbox.2));
        GeoPred::Leaf(Arc::new(if vertices <= rustic_core::geo::GRID_KEY_MAX_VERTICES {
            rustic_core::geo::GridPred::keyed(p, b, format!("point shape within={within} {query:?}"))
        } else {
            rustic_core::geo::GridPred::new(p, b)
        }))
    };
    Some(match rel {
        shape::SpatialRelation::Intersects => leaf(false),
        shape::SpatialRelation::Within => leaf(true),
        // A point, none intersecting.
        shape::SpatialRelation::Disjoint => GeoPred::And(vec![GeoPred::All, GeoPred::Not(Box::new(leaf(false)))]),
        _ => return None,
    })
}

impl rustic_core::query::CustomQuery for GeoPointShapeQuery {
    fn geo_pred(&self) -> Option<(u16, rustic_core::geo::GeoPred)> {
        Some((self.field, point_shape_pred(&self.query, self.rel)?))
    }

    fn scorer(&self, leaf: &LeafReader<'_>, _ctx: &rustic_core::query::SearchContext) -> Option<Box<dyn rustic_core::query::Scorer>> {
        use rustic_core::geo::{decode_lat, decode_lon, DocHits};
        // Sealed segments: one predicate walk (cell grid, threads, or the
        // doc-order scan when the search stops early). `intersects` means
        // "some point intersects" on any field; `within` and `disjoint`
        // are per-point predicates only when no doc has several points.
        if let LeafReader::Segment(seg) = leaf {
            let g = seg.geo_index(self.field)?;
            let any_point = self.rel == shape::SpatialRelation::Intersects;
            if any_point || g.single_valued() {
                if let Some((_, pred)) = rustic_core::query::CustomQuery::geo_pred(self) {
                    if let Some(s) = rustic_core::query::geo_pred_scorer(g, pred, any_point, seg.num_docs, 1.0) {
                        return Some(s);
                    }
                }
            }
        }
        let mut hits = self.scan(leaf);
        let dv = rustic_core::query::GeoDv::open(leaf, self.field);
        match self.rel {
            shape::SpatialRelation::Intersects => {}
            // A doc with several points is within when all of them are.
            shape::SpatialRelation::Within => {
                let q = &*self.query;
                hits.retain(|d| dv.entries(d).iter().all(|p| shape::point_within((decode_lon(p.lon), decode_lat(p.lat)), q)));
            }
            // Docs with a point, none intersecting.
            _ => {
                let mut with = vec![0u64; (leaf.num_docs() as usize).div_ceil(64)];
                for p in dv.iter() {
                    if let Some(w) = with.get_mut(p.doc as usize / 64) {
                        *w |= 1u64 << (p.doc % 64);
                    }
                }
                match &hits {
                    DocHits::Sorted(v) => v.iter().for_each(|&d| with[d as usize / 64] &= !(1u64 << (d % 64))),
                    DocHits::Bits(b) => with.iter_mut().zip(b).for_each(|(w, h)| *w &= !h),
                }
                hits = DocHits::Bits(with);
            }
        }
        if hits.is_empty() {
            return None;
        }
        Some(rustic_core::query::hits_scorer(hits, 1.0))
    }

    fn is_constant_score(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "geo_shape"
    }
}

