//! ES Query DSL JSON → `rustic_core::query::Query`.
//!
//! Field-aware: text fields are analyzed with the field's search analyzer,
//! keyword values go through the normalizer, numbers/dates/ips/booleans
//! become exact or range queries on their core encodings, unmapped fields
//! match nothing (as in ES).

use crate::date::{parse_date_math, DateFormatter, Tz};
use crate::error::{EsError, Result};
use crate::mapping::{EsType, FieldLookup, ResolvedField, FIELD_NAMES_FIELD, ID_FIELD, IGNORED_FIELD, ROUTING_FIELD, SEQ_NO_FIELD};
use crate::values;
use rustic_core::analysis::Analyzer;
use rustic_core::query::{
    BoostMode, Bound, DecayKind, FieldValueModifier, Fuzziness, LeafReader, MinShouldMatch, MultiTermRewrite, Query, ScoreFunction, ScoreMode,
};
use rustic_core::types::Term;
use serde_json::{Map, Value};
use std::sync::Arc;

/// Resolves a terms-lookup (`{"index","id","path"}`) to values.
pub type TermsLookupFn<'a> = &'a (dyn Fn(&str, &str, &str, Option<&str>) -> Result<Vec<Value>> + Sync);

/// Everything query translation needs.
#[derive(Clone)]
pub struct QueryContext<'a> {
    pub deadline: Option<std::time::Instant>,
    pub lookup: &'a FieldLookup,
    pub index_name: &'a str,
    pub index_uuid: &'a str,
    pub aliases: Vec<String>,
    pub now: i64,
    /// Leaves of the searched snapshot (prefix expansion, geo distance).
    pub leaves: &'a [LeafReader<'a>],
    pub default_fields: Vec<String>,
    pub terms_lookup: Option<TermsLookupFn<'a>>,
    pub max_terms_count: usize,
    pub max_regex_length: usize,
    /// `index.max_result_window`: the `ids` query takes at most this many ids.
    pub max_result_window: usize,
    /// `index.disable_sequence_numbers`: `_seq_no` cannot be queried.
    pub seq_no_disabled: bool,
    /// `index.query.parse.allow_unmapped_fields` (default true): when false
    /// a query on an unmapped field is an error instead of matching nothing.
    pub allow_unmapped_fields: bool,
    /// Filter ANDed into every `knn` pre-filter (the searched alias's
    /// filter, which ES applies inside the kNN search).
    pub knn_prefilter: Option<Value>,
    /// Full mapping for trusted alias/role filters only. User queries continue
    /// to resolve against `lookup`, which may be restricted by FLS.
    pub knn_prefilter_lookup: Option<Arc<FieldLookup>>,
    /// `index.search.allow_expensive_queries` (default true).
    pub allow_expensive_queries: bool,
    /// The index, for its mapping (percolate builds its in-memory index
    /// with it): read when needed, not cloned for every search.
    pub mappings: Option<&'a crate::index::IndexCore>,
}

impl<'a> QueryContext<'a> {
    pub(crate) fn knn_prefilter_query(&self) -> Result<Option<Query>> {
        self.knn_prefilter.as_ref().map(|filter| {
            let trusted = QueryContext {
                lookup: self.knn_prefilter_lookup.as_deref().unwrap_or(self.lookup),
                knn_prefilter: None,
                ..self.clone()
            };
            parse_query(filter, &trusted)
        }).transpose()
    }
    pub(crate) fn shard_err_pub(&self, reason: impl Into<String>) -> EsError {
        self.shard_err(reason)
    }
    fn shard_err(&self, reason: impl Into<String>) -> EsError {
        EsError::query_shard(self.index_name, self.index_uuid, reason)
    }
}

fn parsing(reason: impl Into<String>) -> EsError {
    EsError::parsing(reason)
}

fn f32_of(v: &Value) -> Option<f32> {
    match v {
        Value::Number(n) => n.as_f64().map(|x| x as f32),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn boost_of(o: &Map<String, Value>) -> Result<f32> {
    match o.get("boost") {
        None => Ok(1.0),
        // ES `parser.floatValue()`: strings go through `Float.parseFloat`, other tokens are a Jackson error.
        Some(Value::String(s)) => s.trim().parse::<f32>().map_err(|_| EsError::number_format(s)).and_then(valid_boost),
        Some(Value::Number(n)) => valid_boost(n.as_f64().unwrap_or(1.0) as f32),
        Some(v) => Err(EsError::jackson_type("boost", None, v, "not numeric, can not use numeric value accessors")),
    }
}

/// ES `AbstractQueryBuilder.checkNegativeBoost` (`Float.compare(boost, 0f) < 0`,
/// so `-0.0` too), then Lucene `BoostQuery`'s finite check. Every pruning
/// bound assumes a finite weight >= 0.
pub(crate) fn valid_boost(b: f32) -> Result<f32> {
    if b.is_finite() && b.total_cmp(&0.0).is_ge() {
        return Ok(b);
    }
    if b.total_cmp(&0.0).is_lt() && !b.is_nan() {
        return Err(EsError::illegal_argument("negative [boost] are not allowed."));
    }
    let shown = if b.is_nan() { "NaN" } else { "Infinity" };
    Err(EsError::illegal_argument(format!("boost must be a positive float, got {shown}")))
}

/// Java `Float.parseFloat` (what ES reads a `field^weight` with): spaces
/// around, an `f` / `d` suffix, `Infinity` / `NaN` spelled as Java does.
pub(crate) fn java_parse_float(s: &str) -> Option<f32> {
    let t = s.trim();
    let (neg, u) = match t.strip_prefix('-') {
        Some(u) => (true, u),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    // `Infinity` / `NaN` take no suffix.
    match u {
        "Infinity" => return Some(if neg { f32::NEG_INFINITY } else { f32::INFINITY }),
        "NaN" => return Some(f32::NAN),
        _ => {}
    }
    let u = u.strip_suffix(['f', 'F', 'd', 'D']).unwrap_or(u);
    // Hexadecimal: `0x<hex>[.<hex>]p<exp>` (the binary exponent is required).
    if let Some(h) = u.strip_prefix("0x").or_else(|| u.strip_prefix("0X")) {
        let (m, e) = h.split_once(['p', 'P'])?;
        let exp: i32 = e.parse().ok()?;
        let (ip, fp) = m.split_once('.').unwrap_or((m, ""));
        if ip.is_empty() && fp.is_empty() {
            return None;
        }
        let mut v = 0f64;
        for c in ip.chars().chain(fp.chars()) {
            v = v * 16.0 + c.to_digit(16)? as f64;
        }
        let v = v * 2f64.powi(exp - 4 * fp.len() as i32);
        return Some(if neg { -v as f32 } else { v as f32 });
    }
    if u.is_empty() || u.chars().any(|c| !(c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-'))) {
        return None;
    }
    let v: f32 = u.parse().ok()?;
    Some(if neg { -v } else { v })
}

/// The weight of a `field^weight` spec as ES applies it
/// (`QueryParserHelper.parseFieldsAndWeights`, then Lucene `BoostQuery`):
/// unparsable is a `number_format_exception`, negative or not finite fails
/// [`valid_boost`].
pub(crate) fn field_weight(spec: &str) -> Result<Option<f32>> {
    let Some((_, w)) = spec.rsplit_once('^') else { return Ok(None) };
    // Java's message for an empty string differs.
    if w.trim().is_empty() {
        return Err(EsError::new(400, "number_format_exception", "empty String"));
    }
    let b = java_parse_float(w).ok_or_else(|| EsError::number_format(w))?;
    valid_boost(b).map(Some)
}

/// `index.query.default_field` (the query context's default fields) when a
/// query names none: its `^` weights checked as ES checks them where it
/// applies them.
pub(crate) fn default_specs(ctx: &QueryContext) -> Result<Vec<String>> {
    for f in &ctx.default_fields {
        field_weight(f)?;
    }
    Ok(ctx.default_fields.clone())
}

/// Queries whose body is keyed by field (`{"match": {"title": {...}}}`):
/// a member's `boost` is that field's, and a lone top-level `boost` is a
/// field named `boost` (ES's short form, `{"term": {"boost": -1}}`).
const FIELD_KEYED: &[&str] = &[
    "term", "range", "prefix", "wildcard", "regexp", "fuzzy", "match", "match_phrase", "match_phrase_prefix", "match_bool_prefix", "span_term", "intervals",
    "text_expansion", "weighted_tokens", "terms_set",
];

/// [`valid_boost`] for every boost a query body carries: its own `boost`,
/// a field-keyed one (`{"match": {"title": {"boost": 2}}}`) and the `^`
/// weights of its `fields` (`multi_match`, `query_string`, ...; ES reads a
/// query's `default_field` as one field name, no weight). Bodies a query parser later rejects are left to it
/// (a non-numeric boost), as is everything that is data (a rule's match
/// criteria, sparse vector tokens) and `combined_fields`' weights (it
/// rejects those below 1 itself). ObjectParser-built queries wrap the error
/// as ES does (`[match_all] failed to parse field [boost]`). A short-form
/// query with more top-level keys than its field never gets here
/// (`strict::check`: "doesn't support multiple fields").
fn check_boosts(name: &str, body: &Value) -> Result<()> {
    let Some(o) = body.as_object() else { return Ok(()) };
    let num = |v: &Value| match v {
        Value::Number(n) => n.as_f64().map(|x| x as f32),
        Value::String(s) => s.trim().parse::<f32>().ok(),
        _ => None,
    };
    let keyed = FIELD_KEYED.contains(&name);
    let wrap = |e: EsError| {
        let w = || EsError::x_content_parse(format!("[{name}] failed to parse field [boost]")).at_key("boost", Some(name)).prefixed("cursor");
        match name {
            // ObjectParser over a `{name: {...}}` pair: a parsing_exception
            // around the field failure.
            "match_all" | "ids" => EsError::parsing(format!("[{name}] failed to parse field [boost]")).at_key("boost", Some(name)).prefixed("cursor").with_lc().caused_by(w().caused_by(e)),
            // `KnnVectorQueryBuilder`'s ConstructingObjectParser.
            "knn" => w().caused_by(e),
            _ => e,
        }
    };
    let own = o.get("boost").filter(|_| !keyed || o.keys().any(|k| k != "boost" && k != "_name"));
    if let Some(b) = own.and_then(num) {
        valid_boost(b).map_err(wrap)?;
    }
    for b in o.values().filter(|_| keyed).filter_map(Value::as_object).filter_map(|f| f.get("boost")).filter_map(num) {
        valid_boost(b)?;
    }
    // ES `ScoreFunctionBuilder.setWeight`.
    if name == "function_score" {
        let functions = o.get("functions").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_object);
        if std::iter::once(o).chain(functions).filter_map(|f| f.get("weight")).filter_map(num).any(|w| w.total_cmp(&0.0).is_lt()) {
            return Err(EsError::illegal_argument("[weight] cannot be negative for a filtering function"));
        }
    }
    if name != "combined_fields" {
        let fields = match o.get("fields") {
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).collect(),
            Some(Value::String(f)) => vec![f.as_str()],
            _ => Vec::new(),
        };
        for f in fields {
            field_weight(f)?;
        }
    }
    Ok(())
}

fn with_boost(q: Query, boost: f32) -> Query {
    if (boost - 1.0).abs() < f32::EPSILON {
        q
    } else {
        q.boost(boost)
    }
}

fn single_entry<'v>(qname: &str, v: &'v Value) -> Result<(&'v String, &'v Value)> {
    let o = v.as_object().ok_or_else(|| parsing(format!("[{qname}] query malformed, no start_object after query name")))?;
    let mut it = o.iter().filter(|(k, _)| k.as_str() != "boost" && k.as_str() != "_name");
    // ES's short form keys by field only: a lone `boost` is a field.
    let first = it.next().or_else(|| o.get_key_value("boost")).ok_or_else(|| parsing(format!("[{qname}] query malformed, no field specified")))?;
    if let Some(second) = it.next() {
        return Err(parsing(format!("[{qname}] query doesn't support multiple fields, found [{}] and [{}]", first.0, second.0)));
    }
    Ok(first)
}

/// Parse `minimum_should_match` (int or string spec).
pub fn parse_msm(v: &Value) -> Option<MinShouldMatch> {
    match v {
        Value::Number(n) => {
            let i = n.as_i64()?;
            if i >= 0 {
                Some(MinShouldMatch::Count(i as usize))
            } else {
                Some(MinShouldMatch::Combination(i.to_string()))
            }
        }
        Value::String(s) => Some(match s.trim().parse::<usize>() {
            Ok(n) => MinShouldMatch::Count(n),
            Err(_) => MinShouldMatch::Combination(s.trim().to_string()),
        }),
        _ => None,
    }
}

pub fn parse_fuzziness(v: &Value) -> Result<Fuzziness> {
    match v {
        Value::Number(n) => {
            let x = n.as_f64().unwrap_or(0.0);
            if !(0.0..=2.0).contains(&x) {
                return Err(EsError::illegal_argument(format!("Valid edit distances are [0, 1, 2] but was [{}]", x as i64)));
            }
            Ok(Fuzziness::Edits(x as u8))
        }
        Value::String(s) => {
            let s = s.trim();
            if s.eq_ignore_ascii_case("auto") {
                return Ok(Fuzziness::Auto);
            }
            if let Some(rest) = s.strip_prefix("AUTO:").or_else(|| s.strip_prefix("auto:")) {
                let (a, b) = rest.split_once(',').ok_or_else(|| parsing(format!("failed to find low and high distance values")))?;
                return Ok(Fuzziness::AutoLow(a.trim().parse().unwrap_or(3), b.trim().parse().unwrap_or(6)));
            }
            match s.parse::<f64>() {
                Ok(f) if (0.0..=2.0).contains(&f) => Ok(Fuzziness::Edits(f as u8)),
                Ok(f) => Err(EsError::illegal_argument(format!("Valid edit distances are [0, 1, 2] but was [{}]", f as i64))),
                Err(_) => Err(EsError::illegal_argument(format!("fuzziness cannot be [{s}].")).caused_by(EsError::number_format(&s.to_uppercase()))),
            }
        }
        _ => Err(parsing("fuzziness must be a string or a number")),
    }
}

fn rewrite_of(o: &Map<String, Value>) -> MultiTermRewrite {
    match o.get("rewrite").and_then(Value::as_str) {
        Some("scoring_boolean") => MultiTermRewrite::ScoringBoolean,
        Some("constant_score_boolean") => MultiTermRewrite::ConstantScoreBoolean,
        Some(s) if s.starts_with("top_terms_blended_freqs_") => MultiTermRewrite::TopTermsBlendedFreqs(s[24..].parse().unwrap_or(50)),
        Some(s) if s.starts_with("top_terms_boost_") => MultiTermRewrite::TopTermsBoost(s[16..].parse().unwrap_or(50)),
        Some(s) if s.starts_with("top_terms_") => MultiTermRewrite::TopTermsScoringBoolean(s[10..].parse().unwrap_or(50)),
        _ => MultiTermRewrite::ConstantScore,
    }
}

/// Translate a query object (`{"match": {...}}`).
pub fn parse_query(v: &Value, ctx: &QueryContext) -> Result<Query> {
    let o = match v {
        Value::Object(o) => o,
        // The enclosing parser adds the position (`default_key` / `in_clause`).
        _ => return Err(parsing("[_na] query malformed, must start with start_object").with_lc()),
    };
    if o.is_empty() {
        return Err(EsError::illegal_argument("query malformed, empty clause found at [@POS@]").cursor("end"));
    }
    if o.len() > 1 {
        let keys: Vec<&String> = o.keys().collect();
        return Err(parsing(format!("[{}] malformed query, expected [END_OBJECT] but found [FIELD_NAME]", keys[0])).at_key(keys[1], None).cursor("key").with_lc());
    }
    let (name, body) = o.iter().next().unwrap();
    // `AbstractQueryBuilder.parseTopLevelQuery`'s depth limit, counted
    // before the nested query is parsed.
    let depth = super::limits::Depth::enter(&super::limits::PARSE_DEPTH)?;
    if !body.is_object() {
        if !matches!(name.as_str(), "match_all" | "match_none") || !body.is_object() {
            return Err(parsing(format!("[{name}] query malformed, no start_object after query name")).at_key(name, None).with_lc());
        }
    }
    super::strict::check(name, body)?;
    check_boosts(name, body)?;
    let q = parse_named(name, body, ctx)?;
    let q = super::docvalues::rewrite(name, body, q, ctx);
    // Once, over the whole query (Lucene checks the rewritten tree).
    if depth.is_root() { super::limits::check_clauses(&q)?; }
    Ok(q)
}

fn obj<'v>(qname: &str, v: &'v Value) -> Result<&'v Map<String, Value>> {
    v.as_object().ok_or_else(|| parsing(format!("[{qname}] query malformed, no start_object after query name")))
}

fn parse_named(name: &str, body: &Value, ctx: &QueryContext) -> Result<Query> {
    if matches!(name, "regexp" | "prefix") {
        super::limits::check_regex_body(name, body, ctx.max_regex_length)?;
    }
    if name == "match" && body.get(crate::tsdb::TSID_FIELD).is_some() {
        return Err(ctx.shard_err("Field [_tsid] of type [_tsid] does not support match queries"));
    }
    if matches!(name, "term" | "terms" | "match" | "match_phrase" | "range" | "prefix" | "wildcard" | "regexp" | "fuzzy" | "exists") && body.get(crate::tsdb::TSID_FIELD).is_some() {
        return Err(ctx.shard_err("[_tsid] is not searchable"));
    }
    if let Some(q) = crate::fieldtypes::query::intercept(name, body, ctx)? {
        return Ok(q);
    }
    // Queries on runtime fields (search `runtime_mappings` / mapping `runtime`).
    if let Some(q) = crate::script::search::runtime_field_query(name, body, ctx)? {
        return Ok(q);
    }
    match name {
        "match_all" => Ok(with_boost(Query::MatchAll, boost_of(obj(name, body)?)?)),
        "match_none" => Ok(Query::MatchNone),
        "match" => parse_match(body, ctx, MatchKind::Bool),
        "match_phrase" => parse_match(body, ctx, MatchKind::Phrase),
        "match_phrase_prefix" => parse_match(body, ctx, MatchKind::PhrasePrefix),
        "match_bool_prefix" => parse_match(body, ctx, MatchKind::BoolPrefix),
        "multi_match" => parse_multi_match(body, ctx),
        "combined_fields" => {
            let o = obj(name, body)?;
            if let Some(Value::Array(fs)) = o.get("fields") {
                let specs: Vec<String> = fs.iter().filter_map(|x| x.as_str().map(str::to_string)).collect();
                for (fname, _) in expand_fields(&specs, ctx) {
                    if let Some(f) = ctx.lookup.get(&fname) {
                        if !f.ty.is_text() {
                            return Err(ctx.shard_err(format!("failed to create query: Field [{}] of type [{}] does not support [combined_fields] queries", f.name, f.declared.name())));
                        }
                    }
                }
            }
            super::combined_fields::parse(o, ctx)
        }
        "query_string" => super::query_string::parse_query_string(obj(name, body)?, ctx),
        "simple_query_string" => super::query_string::parse_simple_query_string(obj(name, body)?, ctx),
        "term" => parse_term(body, ctx),
        "terms" => parse_terms(body, ctx),
        "range" => parse_range(body, ctx),
        "exists" => {
            let o = obj(name, body)?;
            let f_owned = o.get("field").filter(|v| !v.is_null()).map(scalar_string).ok_or_else(|| parsing("[exists] must be provided with a [field]").at_key("exists", None).cursor("end").with_lc())?;
            let f = f_owned.as_str();
            if f == "_source" {
                return Err(ctx.shard_err("The _source field is not searchable"));
            }
            Ok(with_boost(exists_query(f, ctx), boost_of(o)?))
        }
        "prefix" => parse_multi_term(name, body, ctx),
        "wildcard" => parse_multi_term(name, body, ctx),
        "regexp" => parse_multi_term(name, body, ctx),
        "fuzzy" => parse_multi_term(name, body, ctx),
        "ids" => {
            let o = obj(name, body)?;
            let vals = match o.get("values") {
                Some(Value::Array(a)) => a.clone(),
                Some(Value::Null) | None => Vec::new(),
                Some(v) => vec![v.clone()],
            };
            let terms: Vec<Vec<u8>> = vals.iter().map(|v| scalar_string(v).into_bytes()).collect();
            // `IdsQueryBuilder.doToQuery` (ids are a set).
            if terms.len() > ctx.max_result_window && terms.iter().collect::<std::collections::HashSet<_>>().len() > ctx.max_result_window {
                return Err(EsError::illegal_argument(format!("Too many ids specified, allowed max result window is [{}]", ctx.max_result_window)));
            }
            Ok(Query::TermSet { field: ID_FIELD, terms, boost: boost_of(o)? })
        }
        "bool" => parse_bool(body, ctx),
        "boosting" => {
            let o = obj(name, body)?;
            let pos = o.get("positive").ok_or_else(|| parsing("[boosting] query requires 'positive' query to be set'").at_key("boosting", None).cursor("end").with_lc())?;
            let neg = o.get("negative").ok_or_else(|| parsing("[boosting] query requires 'negative' query to be set'").at_key("boosting", None).cursor("end").with_lc())?;
            if let Some(Value::String(s)) = o.get("negative_boost") {
                if s.trim().parse::<f64>().is_err() {
                    return Err(EsError::number_format(s));
                }
            }
            let nb = o.get("negative_boost").and_then(f32_of).filter(|b| *b >= 0.0).ok_or_else(|| parsing("[boosting] query requires 'negative_boost' to be set to be a positive value'").at_key("boosting", None).cursor("end").with_lc())?;
            let q = Query::FunctionScore {
                query: Box::new(parse_query(pos, ctx).map_err(|e| e.default_key("positive", Some("boosting")))?),
                functions: vec![ScoreFunction::Filtered { filter: Box::new(parse_query(neg, ctx).map_err(|e| e.default_key("negative", Some("boosting")))?), function: Box::new(ScoreFunction::Weight(nb)) }],
                score_mode: ScoreMode::Multiply,
                boost_mode: BoostMode::Multiply,
                max_boost: f32::MAX,
                min_score: None,
            };
            Ok(with_boost(q, boost_of(o)?))
        }
        "constant_score" => {
            let o = obj(name, body)?;
            let f = o.get("filter").ok_or_else(|| parsing("[constant_score] requires a 'filter' element").at_key("constant_score", None).cursor("end").with_lc())?;
            Ok(Query::ConstScore { query: Box::new(parse_query(f, ctx).map_err(|e| e.default_key("filter", Some("constant_score")))?), score: boost_of(o)?, point_set: false })
        }
        "dis_max" => {
            let o = obj(name, body)?;
            let qs = o.get("queries").and_then(Value::as_array).ok_or_else(|| parsing("[dis_max] requires 'queries' field with at least one clause"))?;
            if qs.is_empty() {
                return Err(parsing("[_na] query malformed, must start with start_object").at_key("queries", Some("dis_max")).cursor("end").with_lc());
            }
            let queries = qs.iter().enumerate().map(|(i, q)| parse_query(q, ctx).map_err(|e| if e.extra.iter().any(|(k, _)| k == "@k") { e } else { e.at_key("queries", Some("dis_max")).elem(i) })).collect::<Result<Vec<_>>>()?;
            let tb = o.get("tie_breaker").and_then(f32_of).unwrap_or(0.0);
            Ok(with_boost(Query::DisMax { queries, tie_breaker: tb }, boost_of(o)?))
        }
        "function_score" => parse_function_score(body, ctx),
        "knn" => parse_knn_query(obj(name, body)?, ctx, None),
        "terms_set" => super::terms_set::terms_set(body, ctx),
        "pinned" => super::pinned::pinned(body, ctx),
        "geo_grid" => super::geo::geo_grid(body, ctx),
        "geo_bounding_box" => super::geo::geo_bounding_box(body, ctx),
        "geo_distance" => super::geo::geo_distance(body, ctx),
        "wrapper" => {
            let o = obj(name, body)?;
            let q = o.get("query").and_then(Value::as_str).ok_or_else(|| parsing("[wrapper] query malformed").at_key("wrapper", None).cursor("end").with_lc())?;
            use base64::Engine;
            let bytes = base64::engine::general_purpose::STANDARD.decode(q).map_err(|_| {
                let bad = q.chars().find(|c| !(c.is_ascii_alphanumeric() || *c == '+' || *c == '/' || *c == '=')).unwrap_or('=');
                let msg = format!("Failed to decode VALUE_STRING as base64 (MIME-NO-LINEFEEDS): Illegal character '{bad}' (code 0x{:x}) in base64 content", bad as u32);
                EsError::x_content_parse(msg.clone()).at_key("query", Some("wrapper")).cursor("after").prefixed("cursor").with_source().caused_by(EsError::new(400, "json_parse_exception", msg).at_key("query", Some("wrapper")).cursor("after").with_source())
            })?;
            let v: Value = match crate::jsonpos::validate(&bytes) {
                Err(e) => return Err(e.to_error(&bytes)),
                Ok(_) => serde_json::from_slice(&bytes)?,
            };
            // Positions inside the wrapped query are relative to the decoded query.
            parse_query(&v, ctx).map_err(|e| crate::jsonpos::resolve(e, &bytes))
        }
        "nested" => parse_nested(obj(name, body)?, ctx),
        "geo_polygon" => super::geo::geo_polygon(body, ctx),
        "has_child" | "has_parent" | "parent_id" => Err(ctx.shard_err(format!("[{name}] queries are not supported (no join field support)"))),
        "kql" => parse_query(&super::kql::rewrite(body, ctx)?, ctx),
        "script" => crate::script::search::script_query(body, ctx),
        "script_score" => crate::script::search::script_score_query(body, ctx, &parse_query),
        "intervals" => super::positional::intervals_query(body, ctx),
        n if super::positional::is_span(n) => super::positional::span_query(n, body, ctx),
        "distance_feature" => super::extra::distance_feature(body, ctx),
        "more_like_this" | "mlt" => super::extra::more_like_this(body, ctx),
        // The `x-pack-test-deprecated-query` test module's query: match_all
        // plus a deprecation warning.
        "deprecated" if crate::warnings::has_test_module("x-pack-test-deprecated-query") => {
            if let Some(k) = obj(name, body)?.keys().next() {
                return Err(EsError::parsing(format!("[deprecated] unknown field [{k}]")));
            }
            crate::warnings::add("[deprecated] query");
            Ok(Query::MatchAll)
        }
        other => Err(EsError::unknown_named("unknown query", other, None, super::strict::QUERY_NAMES)),
    }
}

pub fn scalar_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Field-level helpers.

fn unmapped(_ctx: &QueryContext) -> Query {
    Query::MatchNone
}

/// Presence of a mapped field in a doc: doc values / postings (`Exists`),
/// plus a `_field_names` term for types that can hold a value without
/// either (binary, text with empty values, unindexed fields).
fn field_presence(f: &ResolvedField) -> Option<Query> {
    let mut qs = Vec::new();
    if let Some(id) = f.core_id.or(f.geo_id).or(f.lat_id) {
        qs.push(Query::Exists { field: id });
    }
    if f.core_id.is_none() || f.ty.is_text() || matches!(f.ty, EsType::Binary | EsType::DenseVector | EsType::SparseVector) || (!f.indexed && !f.doc_values) {
        qs.push(Query::Term(Term::new_text(FIELD_NAMES_FIELD, &f.name)));
    }
    match qs.len() {
        0 => None,
        1 => qs.pop(),
        _ => Some(mk_bool(vec![], qs, vec![], vec![], MinShouldMatch::Count(1))),
    }
}

fn exists_query(field: &str, ctx: &QueryContext) -> Query {
    if let Some(f) = ctx.lookup.get(field) {
        return field_presence(f).unwrap_or(Query::MatchNone);
    }
    if field == "_seq_no" && ctx.seq_no_disabled {
        return Query::MatchNone;
    }
    if matches!(field, "_id" | "_index" | "_seq_no" | "_version" | "_primary_term") {
        return Query::MatchAll;
    }
    if field == "_routing" {
        return Query::Exists { field: ROUTING_FIELD };
    }
    if field == "_ignored" {
        return Query::Exists { field: IGNORED_FIELD };
    }
    // Object or wildcard: any sub-field.
    let pattern = if field.contains('*') { field.to_string() } else { format!("{field}.*") };
    let subs: Vec<Query> = ctx
        .lookup
        .matching(&pattern)
        .iter()
        .filter_map(|n| ctx.lookup.get(n))
        .filter(|f| !f.is_multi_field)
        .filter_map(|f| field_presence(f))
        .collect();
    match subs.len() {
        0 => Query::MatchNone,
        1 => subs.into_iter().next().unwrap(),
        _ => mk_bool(vec![], subs, vec![], vec![], MinShouldMatch::Count(1)),
    }
}

fn mk_bool(must: Vec<Query>, should: Vec<Query>, must_not: Vec<Query>, filter: Vec<Query>, msm: MinShouldMatch) -> Query {
    Query::Bool { must, should, must_not, filter, minimum_should_match: msm }
}

fn normalize_keyword(f: &ResolvedField, s: &str) -> String {
    match &f.normalizer {
        Some(n) => n.analyze(s).into_iter().map(|t| t.text).collect::<Vec<_>>().join(""),
        None => s.to_string(),
    }
}

/// Exact-value query on a field (term semantics).
pub fn term_query(f: &ResolvedField, v: &Value, case_insensitive: bool, ctx: &QueryContext) -> Result<Query> {
    if let Some(q) = crate::fieldtypes::query::term(f, v, case_insensitive, ctx)? {
        return Ok(q);
    }
    let Some(id) = f.core_id else {
        if f.ty == EsType::GeoPoint {
            return Err(ctx.shard_err(format!("failed to create query: Geometry fields do not support exact searching, use dedicated geometry queries instead: [{}]", f.name)));
        }
        return Ok(Query::MatchNone);
    };
    let s = scalar_string(v);
    let num_err = |ty: &str| ctx.shard_err(format!("failed to create query: For input string: \"{s}\"")).caused_by(EsError::new(400, "number_format_exception", format!("For input string: \"{s}\""))).with("type_hint", ty.to_string());
    Ok(match f.ty {
        EsType::Text | EsType::MatchOnlyText => {
            if case_insensitive {
                Query::term_case_insensitive(id, &s)
            } else {
                Query::Term(Term::new_text(id, &s))
            }
        }
        EsType::Keyword | EsType::Wildcard | EsType::Flattened | EsType::ConstantKeyword => {
            let n = normalize_keyword(f, &s);
            if case_insensitive {
                Query::Regexp { field: id, pattern: regex_escape(&n), case_insensitive: true, max_expansions: usize::MAX, rewrite: MultiTermRewrite::ConstantScore }
            } else {
                Query::Term(Term::new_text(id, &n))
            }
        }
        EsType::Long | EsType::Integer | EsType::Short | EsType::Byte => {
            let x: f64 = match v {
                Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => s.trim().parse().map_err(|_| num_err("long"))?,
            };
            if x.fract() != 0.0 {
                return Ok(Query::MatchNone);
            }
            Query::RangeI64 { field: id, lower: Bound::Included(x as i64), upper: Bound::Included(x as i64) }
        }
        EsType::UnsignedLong => {
            let x: u64 = match v {
                Value::Number(n) => n.as_u64().ok_or_else(|| num_err("unsigned_long"))?,
                _ => s.trim().parse().map_err(|_| num_err("unsigned_long"))?,
            };
            Query::RangeU64 { field: id, lower: Bound::Included(x), upper: Bound::Included(x) }
        }
        EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat => {
            let x: f64 = match v {
                Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => s.trim().parse().map_err(|_| num_err("double"))?,
            };
            let x = match f.ty {
                EsType::Float => values::to_float(x),
                EsType::HalfFloat => values::to_half_float(x),
                EsType::ScaledFloat => (x * f.scaling_factor).round() / f.scaling_factor,
                _ => x,
            };
            Query::RangeF64 { field: id, lower: Bound::Included(x), upper: Bound::Included(x) }
        }
        EsType::Boolean => {
            let b = match v {
                Value::Bool(b) => *b,
                _ => match s.as_str() {
                    "true" => true,
                    "false" | "" => false,
                    _ => return Err(ctx.shard_err(format!("failed to create query: Can't parse boolean value [{s}], expected [true] or [false]"))),
                },
            };
            Query::Term(Term::new_bool(id, b))
        }
        EsType::Date | EsType::DateNanos => {
            let lo = date_value(f, v, None, None, false, ctx)?;
            let hi = date_value(f, v, None, None, true, ctx)?;
            Query::RangeI64 { field: id, lower: Bound::Included(lo), upper: Bound::Included(hi) }
        }
        EsType::Ip => {
            if s.contains('/') {
                let (lo, hi) = values::cidr_bounds(&s).ok_or_else(|| ctx.shard_err(format!("failed to create query: Expected [ip/prefix] but was [{s}]")))?;
                Query::term_range(id, Bound::Included(lo.into_bytes()), Bound::Included(hi.into_bytes()))
            } else {
                let b = values::parse_ip(&s).ok_or_else(|| ctx.shard_err(format!("failed to create query: '{s}' is not an IP string literal.")))?;
                Query::Term(Term::new_text(id, &values::ip_to_term(&b)))
            }
        }
        EsType::DenseVector => return Err(ctx.shard_err(format!("[dense_vector] fields do not support term queries"))),
        _ => Query::MatchNone,
    })
}

fn regex_escape(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if ".?+*|{}[]()\"\\#@&<>~".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// The unit a custom date pattern's (`dd/MM/yyyy`) rounded-up bound
/// ends: `Some(None)` when the pattern reaches milliseconds (nothing to
/// fill), `None` when `spec` is a named format (`strict_date_optional_time`,
/// `year`) or several (`a||b`). ES's round-up parser
/// (`JavaDateFormatter.defaultRoundUp`) defaults a missing month and day
/// of month (or day of year) to 1, not their maximum, and only the missing
/// time fields to theirs; so a date-only pattern, `uuuu` and `yyyy-MM`
/// included, ends its first day (`lte: 1500` is 1500-01-01T23:59:59.999),
/// and a pattern with a time ends its finest time field.
fn finest_pattern_unit(spec: &str) -> Option<Option<crate::date::CalUnit>> {
    use crate::date::CalUnit;
    let named = spec.contains("||") || spec.contains('_') || (spec.chars().all(|c| c.is_ascii_lowercase()) && !spec.chars().all(|c| c == 'y' || c == 'u'));
    if named {
        return None;
    }
    let mut rank = 0u8;
    let mut quoted = false;
    for c in spec.chars() {
        if c == '\'' {
            quoted = !quoted;
            continue;
        }
        if quoted {
            continue;
        }
        // Date letters (years, quarters, months, weeks, days) all end
        // the day; hours, minutes and seconds end themselves.
        rank = rank.max(match c {
            'H' | 'h' | 'k' | 'K' => 4,
            'm' => 5,
            's' => 6,
            'S' | 'n' | 'A' | 'N' => 7,
            _ => 0,
        });
    }
    Some(match rank {
        0 => Some(CalUnit::Day),
        4 => Some(CalUnit::Hour),
        5 => Some(CalUnit::Minute),
        6 => Some(CalUnit::Second),
        _ => None,
    })
}

/// A range bound given as text in an explicit non-epoch `format` (`spec`):
/// `round_up` (`gt`, `lte`) fills the fields the format lacks as ES's
/// round-up parser does; `gte` and `lt` take the value as parsed.
fn text_date_bound(s: &str, spec: &str, fmt: &DateFormatter, tz: Tz, now: i64, round_up: bool) -> Result<i64> {
    let unit = finest_pattern_unit(spec);
    let ms = parse_date_math(s, fmt, tz, now, round_up && unit.is_none())?;
    if !round_up {
        return Ok(ms);
    }
    if let Some(unit) = unit {
        return Ok(match unit {
            Some(u) => crate::date::add_units(ms, u, 1, tz).saturating_sub(1),
            None => ms,
        });
    }
    // A named format: `parse_date_math` rounded a `yyyy-MM-dd` date
    // up already; another value without a time takes its day.
    let b = s.as_bytes();
    let rounded = b.len() == 10 && b[4] == b'-' && b[7] == b'-';
    let has_time = s.contains(['T', ':']) || rounded;
    Ok(if !has_time { ms + 86_400_000 - 1 } else { ms })
}

fn date_value(f: &ResolvedField, v: &Value, format: Option<&str>, tz: Option<Tz>, round_up: bool, ctx: &QueryContext) -> Result<i64> {
    let fmt = match format {
        Some(spec) => DateFormatter::new(spec)?,
        None => f.date_format.clone(),
    };
    let tz = tz.unwrap_or(Tz::UTC);
    // An explicit non-epoch format parses numbers as text (`gte: 1500` with
    // `uuuu`); rounding up a value fills its missing time fields with their
    // max (`lte: "23/02/2019"` with `dd/MM/yyyy`, `lte: 1500` with `uuuu`
    // both end that day: a missing month and day default to 1).
    if let (Some(spec), false, Value::Number(_) | Value::String(_)) = (format, fmt.accepts_numbers(), v) {
        let s = scalar_string(v);
        if !s.contains("||") && !s.starts_with("now") {
            return text_date_bound(&s, spec, &fmt, tz, ctx.now, round_up).map_err(|e| {
                let cause = if e.reason.starts_with("failed to parse date field") { EsError::illegal_argument(e.reason.clone()) } else { EsError::new(400, "parse_exception", e.reason.clone()) };
                ctx.shard_err(format!("failed to create query: {}", e.reason)).caused_by(cause)
            });
        }
    }
    match v {
        // A JSON number without an explicit `format` is epoch milliseconds
        // whatever the field's format (ES 9: `term: 894225602000` and
        // `gte: 894225602000` match an `epoch_second` field's 894225602,
        // `gte: 1500` a `yyyy` field's year 1970; a fraction fails).
        Value::Number(n) if format.is_none() && f.ty == EsType::Date => match n.as_i64() {
            Some(ms) => Ok(ms),
            None => {
                let reason = format!("failed to parse date field [{n}] with format [epoch_millis]");
                Err(ctx.shard_err(format!("failed to create query: {reason}")).caused_by(EsError::new(400, "parse_exception", reason)))
            }
        },
        Value::Number(n) => {
            let x = n.as_f64().unwrap_or(0.0);
            if fmt.accepts_numbers() {
                fmt.parse_number(x)
            } else {
                Ok(x as i64)
            }
        }
        Value::String(s) => parse_date_math(s, &fmt, tz, ctx.now, round_up).map_err(|e| {
            ctx.shard_err(format!("failed to create query: {}", e.reason)).caused_by(EsError::new(400, "parse_exception", e.reason.clone()))
        }),
        _ => Err(ctx.shard_err("failed to create query: date value must be a string or a number")),
    }
}

// ---------------------------------------------------------------------------
// term / terms / range / multi-term.

fn parse_term(body: &Value, ctx: &QueryContext) -> Result<Query> {
    let (field, spec) = single_entry("term", body)?;
    if spec.is_null() {
        return Err(EsError::illegal_argument("field name is null or empty"));
    }
    let (value, boost, ci) = match spec {
        Value::Object(o) => {
            // `value` (or its legacy alias `term`); any other key is rejected
            // by name, as ES does.
            let v = match o.get("value").or_else(|| o.get("term")) {
                Some(v) => v,
                None => {
                    let bad = o.keys().find(|k| !matches!(k.as_str(), "boost" | "_name" | "case_insensitive")).cloned().unwrap_or_else(|| "query".into());
                    return Err(parsing(format!("[term] query does not support [{bad}]")));
                }
            };
            (v.clone(), boost_of(o)?, o.get("case_insensitive").and_then(Value::as_bool).unwrap_or(false))
        }
        Value::Array(_) => return Err(parsing("[term] query does not support array of values")),
        other => (other.clone(), 1.0, false),
    };
    let q = match field.as_str() {
        "_id" => Query::TermSet { field: ID_FIELD, terms: vec![scalar_string(&value).into_bytes()], boost: 1.0 },
        "_index" => {
            let s = scalar_string(&value);
            if s == ctx.index_name || ctx.aliases.contains(&s) { Query::MatchAll } else { Query::MatchNone }
        }
        "_routing" => Query::Term(Term::new_text(ROUTING_FIELD, &scalar_string(&value))),
        "_ignored" => Query::Term(Term::new_text(IGNORED_FIELD, &scalar_string(&value))),
        "_seq_no" if ctx.seq_no_disabled => return Err(ctx.shard_err(format!("failed to create query: {}", crate::search::SEQ_NO_DISABLED))),
        "_seq_no" => {
            let n = value.as_i64().unwrap_or(-1);
            Query::RangeI64 { field: SEQ_NO_FIELD, lower: Bound::Included(n), upper: Bound::Included(n) }
        }
        _ => match ctx.lookup.get(field) {
            Some(f) => term_query(f, &value, ci, ctx)?,
            None => unmapped(ctx),
        },
    };
    Ok(with_boost(q, boost))
}

fn parse_terms(body: &Value, ctx: &QueryContext) -> Result<Query> {
    let o = obj("terms", body)?;
    let boost = boost_of(o)?;
    let mut entries = o.iter().filter(|(k, _)| k.as_str() != "boost" && k.as_str() != "_name");
    let (field, spec) = entries.next().ok_or_else(|| parsing("[terms] query requires a field"))?;
    if let Some((other, v)) = entries.next() {
        if v.is_array() || v.is_object() {
            return Err(parsing("[terms] query does not support multiple fields"));
        }
        return Err(parsing(format!("[terms] query does not support [{other}]")));
    }
    // Borrowed, not cloned: a `terms` query may list tens of thousands of
    // values (Rally geonames `large_terms`: 45,586).
    let values: std::borrow::Cow<'_, [Value]> = match spec {
        Value::Array(a) => std::borrow::Cow::Borrowed(a.as_slice()),
        Value::Object(l) => std::borrow::Cow::Owned({
            let idx = l.get("index").and_then(Value::as_str).ok_or_else(|| parsing("[terms] query lookup element requires specifying the index."))?;
            let id = l.get("id").map(scalar_string).ok_or_else(|| parsing("[terms] query lookup element requires specifying the id."))?;
            let path = l.get("path").and_then(Value::as_str).ok_or_else(|| parsing("[terms] query lookup element requires specifying the path."))?;
            let routing = l.get("routing").and_then(Value::as_str);
            match ctx.terms_lookup {
                Some(f) => f(idx, &id, path, routing)?,
                None => return Err(ctx.shard_err("terms lookup is not available here")),
            }
        }),
        _ => return Err(parsing(format!("[terms] query does not support [{field}]"))),
    };
    if values.len() > ctx.max_terms_count {
        return Err(ctx.shard_err(format!(
            "The number of terms [{}] used in the Terms Query request has exceeded the allowed maximum of [{}]. This maximum can be set by changing the [index.max_terms_count] index level setting.",
            values.len(),
            ctx.max_terms_count
        )));
    }
    let q = match field.as_str() {
        "_id" => Query::TermSet { field: ID_FIELD, terms: values.iter().map(|v| scalar_string(v).into_bytes()).collect(), boost: 1.0 },
        "_index" => {
            if values.iter().any(|v| { let s = scalar_string(v); s == ctx.index_name || ctx.aliases.contains(&s) }) { Query::MatchAll } else { Query::MatchNone }
        }
        "_routing" => Query::TermSet { field: ROUTING_FIELD, terms: values.iter().map(|v| scalar_string(v).into_bytes()).collect(), boost: 1.0 },
        "_ignored" => Query::TermSet { field: IGNORED_FIELD, terms: values.iter().map(|v| scalar_string(v).into_bytes()).collect(), boost: 1.0 },
        _ => match ctx.lookup.get(field) {
            None => Query::MatchNone,
            Some(f) => match (f.ty, f.core_id) {
                (EsType::Keyword | EsType::Wildcard | EsType::Flattened | EsType::ConstantKeyword | EsType::Text | EsType::MatchOnlyText, Some(id)) => {
                    let mut terms: Vec<Vec<u8>> = values.iter().map(|v| {
                        let s = scalar_string(v);
                        if f.ty.is_text() { s.into_bytes() } else { normalize_keyword(f, &s).into_bytes() }
                    }).collect();
                    // Sorted once here so each segment looks them up in one pass.
                    terms.sort_unstable();
                    terms.dedup();
                    Query::TermSet { field: id, terms, boost: 1.0 }
                }
                _ => {
                    let qs = values.iter().map(|v| term_query(f, v, false, ctx)).collect::<Result<Vec<_>>>()?;
                    // ES builds one point/term set query for these types (one
                    // clause); `date` uses the default disjunction of term
                    // queries, one clause per value.
                    let point_set = matches!(f.ty, EsType::Long | EsType::Integer | EsType::Short | EsType::Byte | EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat | EsType::UnsignedLong | EsType::TokenCount | EsType::Ip | EsType::Boolean | EsType::Version);
                    Query::ConstScore { query: Box::new(mk_bool(vec![], qs, vec![], vec![], MinShouldMatch::Count(1))), score: 1.0, point_set }
                }
            },
        },
    };
    Ok(with_boost(q, boost))
}

fn parse_range(body: &Value, ctx: &QueryContext) -> Result<Query> {
    let (field, spec) = single_entry("range", body)?;
    let o = spec.as_object().ok_or_else(|| parsing(format!("[range] query malformed, after field missing start object")))?;
    let mut gt = None;
    let mut gte = None;
    let mut lt = None;
    let mut lte = None;
    let mut include_lower = true;
    let mut include_upper = true;
    let mut from = None;
    let mut to = None;
    for (k, v) in o {
        match k.as_str() {
            "gt" => gt = Some(v.clone()),
            "gte" => gte = Some(v.clone()),
            "lt" => lt = Some(v.clone()),
            "lte" => lte = Some(v.clone()),
            "from" => from = Some(v.clone()),
            "to" => to = Some(v.clone()),
            "include_lower" => include_lower = v.as_bool().unwrap_or(true),
            "include_upper" => include_upper = v.as_bool().unwrap_or(true),
            "boost" | "format" | "time_zone" | "relation" | "_name" => {}
            other => return Err(parsing(format!("[range] query does not support [{other}]"))),
        }
    }
    if let Some(f) = from.filter(|v| !v.is_null()) {
        if include_lower { gte = Some(f) } else { gt = Some(f) }
    }
    if let Some(t) = to.filter(|v| !v.is_null()) {
        if include_upper { lte = Some(t) } else { lt = Some(t) }
    }
    let gt = gt.filter(|v| !v.is_null());
    let gte = gte.filter(|v| !v.is_null());
    let lt = lt.filter(|v| !v.is_null());
    let lte = lte.filter(|v| !v.is_null());
    let boost = boost_of(o)?;
    let format = o.get("format").and_then(Value::as_str);
    let tz = match o.get("time_zone").and_then(Value::as_str) {
        Some(z) => Some(Tz::parse_query_option(z)?),
        None => None,
    };
    if field == "_seq_no" && ctx.seq_no_disabled {
        return Err(ctx.shard_err(format!("failed to create query: {}", crate::search::SEQ_NO_DISABLED)));
    }
    if field == "_seq_no" {
        let lo = gte.as_ref().and_then(Value::as_i64).map(Bound::Included).or(gt.as_ref().and_then(Value::as_i64).map(Bound::Excluded)).unwrap_or(Bound::Unbounded);
        let hi = lte.as_ref().and_then(Value::as_i64).map(Bound::Included).or(lt.as_ref().and_then(Value::as_i64).map(Bound::Excluded)).unwrap_or(Bound::Unbounded);
        return Ok(with_boost(Query::RangeI64 { field: SEQ_NO_FIELD, lower: lo, upper: hi }, boost));
    }
    let Some(f) = ctx.lookup.get(field) else { return Ok(Query::MatchNone) };
    if matches!(f.declared, EsType::GeoPoint | EsType::GeoShape) {
        return Err(ctx.shard_err(format!("failed to create query: Field [{}] of type [{}] does not support range queries", f.name, f.declared.name())));
    }
    let Some(id) = f.core_id else { return Ok(Query::MatchNone) };
    let num = |v: &Value| -> Result<f64> {
        match v {
            Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0)),
            Value::String(s) => s.trim().parse::<f64>().map_err(|_| ctx.shard_err(format!("failed to create query: For input string: \"{s}\"")).caused_by(EsError::new(400, "number_format_exception", format!("For input string: \"{s}\"")))),
            _ => Err(ctx.shard_err("failed to create query: range bound must be a number")),
        }
    };
    let q = match f.ty {
        EsType::Long | EsType::Integer | EsType::Short | EsType::Byte => {
            // Integer fields: round fractional bounds inward.
            let lower = if let Some(v) = &gte { Bound::Included(num(v)?.ceil() as i64) } else if let Some(v) = &gt { let x = num(v)?; Bound::Included(if x.fract() == 0.0 { x as i64 + 1 } else { x.ceil() as i64 }) } else { Bound::Unbounded };
            let upper = if let Some(v) = &lte { Bound::Included(num(v)?.floor() as i64) } else if let Some(v) = &lt { let x = num(v)?; Bound::Included(if x.fract() == 0.0 { x as i64 - 1 } else { x.floor() as i64 }) } else { Bound::Unbounded };
            Query::RangeI64 { field: id, lower, upper }
        }
        EsType::UnsignedLong => {
            let lower = if let Some(v) = &gte { Bound::Included(num(v)?.ceil().max(0.0) as u64) } else if let Some(v) = &gt { Bound::Excluded(num(v)?.max(0.0) as u64) } else { Bound::Unbounded };
            let upper = if let Some(v) = &lte { Bound::Included(num(v)?.floor().max(0.0) as u64) } else if let Some(v) = &lt { Bound::Excluded(num(v)?.max(0.0) as u64) } else { Bound::Unbounded };
            Query::RangeU64 { field: id, lower, upper }
        }
        EsType::ScaledFloat => {
            // ES `ScaledFloatFieldType.rangeQuery`: the bounds become scaled
            // longs (`ceil` of the scaled lower bound, nudged up when
            // exclusive; `floor` of the upper, nudged down) and the query is
            // a long range. Stored values are `long / factor`, which orders
            // as the longs do, so the same longs as stored bounds match the
            // same docs (a raw double bound could include or drop a value one
            // rounding away from it).
            let factor = f.scaling_factor;
            let lower = match (&gte, &gt) {
                (Some(v), _) => Some(((num(v)? * factor).ceil()) as i64),
                (None, Some(v)) => Some(((num(v)? * factor).next_up().ceil()) as i64),
                _ => None,
            };
            let upper = match (&lte, &lt) {
                (Some(v), _) => Some(((num(v)? * factor).floor()) as i64),
                (None, Some(v)) => Some(((num(v)? * factor).next_down().floor()) as i64),
                _ => None,
            };
            let stored = |l: i64| rustic_core::fieldtypes::numeric::scaled_decode(l, factor);
            Query::RangeF64 { field: id, lower: lower.map_or(Bound::Unbounded, |l| Bound::Included(stored(l))), upper: upper.map_or(Bound::Unbounded, |h| Bound::Included(stored(h))) }
        }
        EsType::Double | EsType::Float | EsType::HalfFloat => {
            let conv = |x: f64| match f.ty {
                EsType::Float => values::to_float(x),
                EsType::HalfFloat => values::to_half_float(x),
                _ => x,
            };
            let lower = if let Some(v) = &gte { Bound::Included(conv(num(v)?)) } else if let Some(v) = &gt { Bound::Excluded(conv(num(v)?)) } else { Bound::Unbounded };
            let upper = if let Some(v) = &lte { Bound::Included(conv(num(v)?)) } else if let Some(v) = &lt { Bound::Excluded(conv(num(v)?)) } else { Bound::Unbounded };
            Query::RangeF64 { field: id, lower, upper }
        }
        EsType::Date | EsType::DateNanos => {
            // gt rounds up, gte rounds down, lt rounds down, lte rounds up.
            let lower = if let Some(v) = &gte {
                Bound::Included(date_value(f, v, format, tz, false, ctx)?)
            } else if let Some(v) = &gt {
                Bound::Excluded(date_value(f, v, format, tz, true, ctx)?)
            } else {
                Bound::Unbounded
            };
            let upper = if let Some(v) = &lte {
                Bound::Included(date_value(f, v, format, tz, true, ctx)?)
            } else if let Some(v) = &lt {
                Bound::Excluded(date_value(f, v, format, tz, false, ctx)?)
            } else {
                Bound::Unbounded
            };
            Query::RangeI64 { field: id, lower, upper }
        }
        EsType::Ip => {
            let ip = |v: &Value| -> Result<Vec<u8>> {
                let s = scalar_string(v);
                values::parse_ip(&s).map(|b| values::ip_to_term(&b).into_bytes()).ok_or_else(|| ctx.shard_err(format!("'{s}' is not an IP string literal.")))
            };
            let lower = if let Some(v) = &gte { Bound::Included(ip(v)?) } else if let Some(v) = &gt { Bound::Excluded(ip(v)?) } else { Bound::Unbounded };
            let upper = if let Some(v) = &lte { Bound::Included(ip(v)?) } else if let Some(v) = &lt { Bound::Excluded(ip(v)?) } else { Bound::Unbounded };
            Query::term_range(id, lower, upper)
        }
        EsType::Keyword | EsType::Text | EsType::Wildcard | EsType::MatchOnlyText | EsType::Flattened | EsType::ConstantKeyword => {
            if !ctx.allow_expensive_queries {
                return Err(ctx.shard_err("[range] queries on [text] or [keyword] fields cannot be executed when 'search.allow_expensive_queries' is set to false."));
            }
            let t = |v: &Value| normalize_keyword(f, &scalar_string(v)).into_bytes();
            let lower = if let Some(v) = &gte { Bound::Included(t(v)) } else if let Some(v) = &gt { Bound::Excluded(t(v)) } else { Bound::Unbounded };
            let upper = if let Some(v) = &lte { Bound::Included(t(v)) } else if let Some(v) = &lt { Bound::Excluded(t(v)) } else { Bound::Unbounded };
            Query::term_range(id, lower, upper)
        }
        EsType::Boolean => {
            // false < true, as ES's boolean term range.
            let b = |v: &Value| -> Result<u8> {
                match v {
                    Value::Bool(b) => Ok(*b as u8),
                    _ => match scalar_string(v).as_str() {
                        "true" => Ok(1),
                        "false" | "" => Ok(0),
                        s => Err(ctx.shard_err(format!("Can't parse boolean value [{s}], expected [true] or [false]"))),
                    },
                }
            };
            let lo: i16 = if let Some(v) = &gte { b(v)? as i16 } else if let Some(v) = &gt { b(v)? as i16 + 1 } else { 0 };
            let hi: i16 = if let Some(v) = &lte { b(v)? as i16 } else if let Some(v) = &lt { b(v)? as i16 - 1 } else { 1 };
            let qs: Vec<Query> = (lo.max(0)..=hi.min(1)).map(|x| Query::Term(Term::new_bool(id, x == 1))).collect();
            // One term range query in ES, so one clause.
            Query::ConstScore { query: Box::new(mk_bool(vec![], qs, vec![], vec![], MinShouldMatch::Count(1))), score: 1.0, point_set: true }
        }
        _ => return Err(ctx.shard_err(format!("Field [{}] of type [{}] does not support range queries", f.name, f.ty.name()))),
    };
    Ok(with_boost(q, boost))
}

fn parse_multi_term(qname: &str, body: &Value, ctx: &QueryContext) -> Result<Query> {
    let (field, spec) = single_entry(qname, body)?;
    let (value, o): (String, Map<String, Value>) = match spec {
        Value::Object(o) => {
            let v = o.get("value").or_else(|| o.get(if qname == "wildcard" { "wildcard" } else { "value" })).ok_or_else(|| parsing(format!("[{qname}] query requires a value")))?;
            (scalar_string(v), o.clone())
        }
        other => (scalar_string(other), Map::new()),
    };
    let boost = boost_of(&o)?;
    let ci = o.get("case_insensitive").and_then(Value::as_bool).unwrap_or(false);
    let rewrite = rewrite_of(&o);
    let (id, f) = match field.as_str() {
        "_id" if qname == "prefix" => {
            return Err(ctx.shard_err("Can only use prefix queries on keyword, text and wildcard fields - not on [_id] which is of type [_id]"));
        }
        "_id" => (ID_FIELD, None),
        _ => match ctx.lookup.get(field) {
            None => return Ok(Query::MatchNone),
            Some(f) => match f.core_id {
                Some(id) => (id, Some(f.clone())),
                None => return Ok(Query::MatchNone),
            },
        },
    };
    if let Some(f) = &f {
        if !(f.ty.is_text() || f.ty.is_keyword_like() || f.ty == EsType::Ip) {
            let what = match qname {
                "prefix" => "prefix",
                "wildcard" => "wildcard",
                "regexp" => "regexp",
                _ => "fuzzy",
            };
            return Err(if qname == "fuzzy" {
                ctx.shard_err(format!("failed to create query: Can only use fuzzy queries on keyword and text fields - not on [{}] which is of type [{}]", f.name, f.ty.name()))
            } else {
                ctx.shard_err(format!("Can only use {what} queries on keyword, text and wildcard fields - not on [{}] which is of type [{}]", f.name, f.ty.name()))
            });
        }
    }
    let norm = |s: &str| -> String {
        match &f {
            Some(f) if f.ty.is_keyword_like() => normalize_keyword(f, s),
            _ => s.to_string(),
        }
    };
    if !ctx.allow_expensive_queries {
        return Err(ctx.shard_err(format!("[{qname}] queries cannot be executed when 'search.allow_expensive_queries' is set to false.{}", if qname == "prefix" { " For optimised prefix queries on text fields please enable [index_prefixes]." } else { "" })));
    }
    let q = match qname {
        // Keyword terms keep their case: match case-insensitively via a regexp.
        "prefix" if ci && f.as_ref().is_some_and(|f| f.ty.is_keyword_like()) => {
            Query::Regexp { field: id, pattern: format!("{}.*", regex_escape(&norm(&value))), case_insensitive: true, max_expansions: usize::MAX, rewrite }
        }
        "wildcard" if ci && f.as_ref().is_some_and(|f| f.ty.is_keyword_like()) => {
            let mut re = String::new();
            let mut it = norm(&value).chars().collect::<Vec<_>>().into_iter();
            while let Some(c) = it.next() {
                match c {
                    '\\' => {
                        if let Some(n) = it.next() {
                            re.push_str(&regex_escape(&n.to_string()));
                        }
                    }
                    '*' => re.push_str(".*"),
                    '?' => re.push('.'),
                    other => re.push_str(&regex_escape(&other.to_string())),
                }
            }
            Query::Regexp { field: id, pattern: re, case_insensitive: true, max_expansions: usize::MAX, rewrite }
        }
        "prefix" => Query::Prefix { field: id, prefix: norm(&value).into_bytes(), max_expansions: usize::MAX, case_insensitive: ci, rewrite },
        "wildcard" => Query::Wildcard { field: id, pattern: norm(&value), max_expansions: usize::MAX, case_insensitive: ci, rewrite },
        "regexp" => super::regexp::regexp_query(id, &value, o.get("flags").and_then(Value::as_str), ci, rewrite, ctx)?,
        _ => Query::Fuzzy {
            field: id,
            text: norm(&value),
            fuzziness: match o.get("fuzziness") {
                Some(v) => parse_fuzziness(v)?,
                None => Fuzziness::Auto,
            },
            prefix_length: o.get("prefix_length").and_then(Value::as_u64).unwrap_or(0) as usize,
            transpositions: o.get("transpositions").and_then(Value::as_bool).unwrap_or(true),
            max_expansions: o.get("max_expansions").and_then(Value::as_u64).unwrap_or(50) as usize,
            rewrite: if o.contains_key("rewrite") { rewrite } else { MultiTermRewrite::TopTermsBlendedFreqs(o.get("max_expansions").and_then(Value::as_u64).unwrap_or(50) as usize) },
        },
    };
    Ok(with_boost(q, boost))
}

// ---------------------------------------------------------------------------
// bool.

fn clauses(v: Option<&Value>, ctx: &QueryContext) -> Result<Vec<Query>> {
    match v {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(a)) => a.iter().map(|q| parse_query(q, ctx)).collect(),
        Some(q @ Value::Object(_)) => Ok(vec![parse_query(q, ctx)?]),
        Some(_) => Err(parsing("[bool] query malformed, clause must be an object or array")),
    }
}

thread_local! {
    /// ES 9 `AutoPrefilteringScope`: per enclosing `bool` clause, the
    /// sibling `must` / `filter` / inverted `must_not` clauses (JSON), which
    /// auto-prefiltering kNN queries (dense `semantic_text` matches) apply.
    static AUTO_PREFILTERS: std::cell::RefCell<Vec<Vec<Value>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The prefilters in scope (outermost first).
pub fn auto_prefilters() -> Vec<Value> {
    AUTO_PREFILTERS.with(|s| s.borrow().iter().flatten().cloned().collect())
}

/// `BoolQueryBuilder.collectPrefilters`: the clauses a clause's kNN queries
/// are prefiltered by.
fn collect_prefilters(o: &Map<String, Value>, excluded: &Value) -> Vec<Value> {
    let list = |k: &str| -> Vec<&Value> {
        match o.get(k) {
            Some(Value::Array(a)) => a.iter().collect(),
            Some(Value::Null) | None => Vec::new(),
            Some(q) => vec![q],
        }
    };
    let mut out: Vec<Value> = Vec::new();
    for k in ["must", "filter"] {
        out.extend(list(k).into_iter().filter(|q| !std::ptr::eq(*q, excluded)).cloned());
    }
    out.extend(list("must_not").into_iter().filter(|q| !std::ptr::eq(*q, excluded)).map(|q| serde_json::json!({"bool": {"must_not": [q]}})));
    out
}

fn parse_bool(body: &Value, ctx: &QueryContext) -> Result<Query> {
    let o = obj("bool", body)?;
    // Auto-prefiltering only matters to dense semantic queries (ES 9).
    let scoped = !crate::version::current_rest_api_version().is_v8() && ctx.lookup.fields.values().any(|f| f.declared.is_semantic());
    let parse_clause = |q: &Value| -> Result<Query> {
        if !scoped {
            return parse_query(q, ctx);
        }
        AUTO_PREFILTERS.with(|s| s.borrow_mut().push(collect_prefilters(o, q)));
        let r = parse_query(q, ctx);
        AUTO_PREFILTERS.with(|s| s.borrow_mut().pop());
        r
    };
    // ES wraps clause failures in `[bool] failed to parse field [<clause>]`.
    let clause = |name: &str| -> Result<Vec<Query>> {
        match o.get(name) {
            Some(Value::Array(a)) => a.iter().enumerate().map(|(i, q)| parse_clause(q).map_err(|e| e.in_clause(name, Some(i)))).collect(),
            Some(Value::Null) | None => Ok(Vec::new()),
            Some(q) => Ok(vec![parse_clause(q).map_err(|e| e.in_clause(name, None))?]),
        }
    };
    let must = clause("must")?;
    let should = clause("should")?;
    let must_not = clause("must_not")?;
    let filter = clause("filter")?;
    let msm = match o.get("minimum_should_match") {
        Some(v) => {
            let m = parse_msm(v).ok_or_else(|| parsing("[bool] minimum_should_match must be an integer or a string"))?;
            // ES `doRewrite` turns a clause-less bool into `match_all` before
            // `minimum_should_match` is ever parsed.
            let no_clauses = must.is_empty() && should.is_empty() && must_not.is_empty() && filter.is_empty();
            if let MinShouldMatch::Combination(c) = &m {
                if !no_clauses && !msm_spec_valid(c) {
                    return Err(ctx.shard_err(format!("failed to create query: For input string: \"{}\"", c.split(['<', ' ']).next().unwrap_or(c).trim_end_matches('%'))));
                }
            }
            m
        }
        None => MinShouldMatch::Count(0),
    };
    let adjust_pure_negative = o.get("adjust_pure_negative").and_then(Value::as_bool).unwrap_or(true);
    // A bool with only must_not matches all docs except those (ES adds match_all).
    let (must, filter) = if must.is_empty() && should.is_empty() && filter.is_empty() && !must_not.is_empty() && !adjust_pure_negative {
        return Ok(with_boost(Query::MatchNone, boost_of(o)?));
    } else if must.is_empty() && should.is_empty() && filter.is_empty() && !must_not.is_empty() {
        (vec![Query::MatchAll], filter)
    } else if must.is_empty() && should.is_empty() && filter.is_empty() {
        (vec![Query::MatchAll], filter)
    } else {
        (must, filter)
    };
    Ok(with_boost(mk_bool(must, should, must_not, filter, msm), boost_of(o)?))
}

// ---------------------------------------------------------------------------
// match family.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MatchKind {
    Bool,
    Phrase,
    PhrasePrefix,
    BoolPrefix,
}

#[derive(Clone)]
pub struct MatchOpts {
    pub kind: MatchKind,
    pub operator_and: bool,
    pub msm: Option<MinShouldMatch>,
    pub fuzziness: Option<Fuzziness>,
    pub prefix_length: usize,
    pub max_expansions: usize,
    pub transpositions: bool,
    pub analyzer: Option<Arc<Analyzer>>,
    pub slop: u32,
    pub zero_terms_all: bool,
    pub lenient: bool,
    pub quoted: bool,
}

impl Default for MatchOpts {
    fn default() -> Self {
        Self {
            kind: MatchKind::Bool,
            operator_and: false,
            msm: None,
            fuzziness: None,
            prefix_length: 0,
            max_expansions: 50,
            transpositions: true,
            analyzer: None,
            slop: 0,
            zero_terms_all: false,
            lenient: false,
            quoted: false,
        }
    }
}

pub fn match_opts_from(o: &Map<String, Value>, kind: MatchKind, ctx: &QueryContext) -> Result<MatchOpts> {
    let mut m = MatchOpts { kind, ..Default::default() };
    if let Some(op) = o.get("operator").and_then(Value::as_str) {
        m.operator_and = match op.to_ascii_lowercase().as_str() {
            "and" => true,
            "or" => false,
            _ => return Err(EsError::illegal_argument(format!("No enum constant org.elasticsearch.index.query.Operator.{}", op.to_ascii_uppercase()))),
        };
    }
    if let Some(v) = o.get("minimum_should_match") {
        m.msm = parse_msm(v);
    }
    if let Some(v) = o.get("fuzziness") {
        m.fuzziness = Some(parse_fuzziness(v)?);
    }
    m.prefix_length = o.get("prefix_length").and_then(Value::as_u64).unwrap_or(0) as usize;
    m.max_expansions = o.get("max_expansions").and_then(Value::as_u64).unwrap_or(50) as usize;
    m.transpositions = o.get("fuzzy_transpositions").and_then(Value::as_bool).unwrap_or(true);
    if let Some(a) = o.get("analyzer").and_then(Value::as_str) {
        m.analyzer = Some(ctx.lookup.registry.analyzer(a).ok_or_else(|| ctx.shard_err(format!("[match] analyzer [{a}] not found")))?);
    }
    m.slop = o.get("slop").or(o.get("phrase_slop")).and_then(Value::as_u64).unwrap_or(0) as u32;
    m.zero_terms_all = o.get("zero_terms_query").and_then(Value::as_str).map(|s| s.eq_ignore_ascii_case("all")).unwrap_or(false);
    m.lenient = o.get("lenient").and_then(Value::as_bool).unwrap_or(false);
    Ok(m)
}

fn parse_match(body: &Value, ctx: &QueryContext, kind: MatchKind) -> Result<Query> {
    let qname = match kind {
        MatchKind::Bool => "match",
        MatchKind::Phrase => "match_phrase",
        MatchKind::PhrasePrefix => "match_phrase_prefix",
        MatchKind::BoolPrefix => "match_bool_prefix",
    };
    let (field, spec) = single_entry(qname, body)?;
    let (text, opts, boost) = match spec {
        Value::Object(o) => {
            let q = o.get("query").ok_or_else(|| parsing(format!("No text specified for text query")))?;
            for k in o.keys() {
                if !matches!(
                    k.as_str(),
                    "query" | "operator" | "minimum_should_match" | "fuzziness" | "prefix_length" | "max_expansions" | "fuzzy_transpositions" | "fuzzy_rewrite"
                        | "analyzer" | "slop" | "zero_terms_query" | "lenient" | "boost" | "_name" | "auto_generate_synonyms_phrase_query" | "cutoff_frequency"
                ) {
                    return Err(parsing(format!("[{qname}] query does not support [{k}]")));
                }
            }
            (q.clone(), match_opts_from(o, kind, ctx)?, boost_of(o)?)
        }
        other => (other.clone(), MatchOpts { kind, ..Default::default() }, 1.0),
    };
    let q = match_field(field, &text, &opts, ctx)?;
    Ok(with_boost(q, boost))
}

/// Tokens grouped by position: `positions[i]` = terms at position i.
pub(crate) fn analyze_positions(a: &Analyzer, text: &str) -> Vec<Vec<String>> {
    let toks = a.analyze(text);
    let mut out: Vec<(u32, Vec<String>)> = Vec::new();
    for t in toks {
        match out.last_mut() {
            Some((p, v)) if *p == t.position => {
                if !v.contains(&t.text) {
                    v.push(t.text)
                }
            }
            _ => out.push((t.position, vec![t.text])),
        }
    }
    out.into_iter().map(|(_, v)| v).collect()
}

/// Expand `prefix` against the term dictionaries of the searched leaves.
fn expand_prefix(field: u16, prefix: &str, max: usize, ctx: &QueryContext) -> Vec<String> {
    let lower = prefix.as_bytes().to_vec();
    let mut upper = lower.clone();
    let upper = loop {
        match upper.pop() {
            Some(b) if b < 0xff => {
                upper.push(b + 1);
                break Some(upper);
            }
            Some(_) => continue,
            None => break None,
        }
    };
    let mut out: Vec<String> = Vec::new();
    for leaf in ctx.leaves {
        for (key, _) in leaf.term_range(field, Some(&lower), upper.as_deref()) {
            let t = Term::from_key(&key);
            if let Ok(s) = String::from_utf8(t.text) {
                if !out.contains(&s) {
                    out.push(s);
                }
            }
        }
    }
    out.sort();
    out.truncate(max);
    out
}

/// The query for analyzed `text` on one field (match family semantics).
pub fn match_field(field: &str, text: &Value, opts: &MatchOpts, ctx: &QueryContext) -> Result<Query> {
    // `match_only_text` / `pattern_text` queries score constantly.
    if ctx.lookup.get(field).is_some_and(|f| matches!(f.declared, EsType::MatchOnlyText | EsType::PatternText)) {
        let q = match_field_inner(field, text, opts, ctx)?;
        return Ok(match q {
            Query::MatchNone => q,
            q => Query::ConstScore { query: Box::new(q), score: 1.0, point_set: false },
        });
    }
    match_field_inner(field, text, opts, ctx)
}

fn match_field_inner(field: &str, text: &Value, opts: &MatchOpts, ctx: &QueryContext) -> Result<Query> {
    if field == "_index" {
        let s = scalar_string(text);
        return Ok(if s == ctx.index_name || ctx.aliases.contains(&s) { Query::MatchAll } else { Query::MatchNone });
    }
    let Some(f) = ctx.lookup.get(field) else {
        return Ok(Query::MatchNone);
    };
    let f = f.clone();
    // `MatchQueryParser`: fields without text search info (`TextSearchInfo.NONE`).
    if matches!(f.declared, EsType::GeoPoint | EsType::GeoShape | EsType::DenseVector) {
        return Err(ctx.shard_err(format!("failed to create query: Field [{}] of type [{}] does not support match queries", f.name, f.declared.name())));
    }
    if !f.ty.is_text() {
        if matches!(opts.kind, MatchKind::PhrasePrefix | MatchKind::BoolPrefix) && f.ty.is_keyword_like() {
            if let Some(id) = f.core_id {
                return Ok(Query::Prefix { field: id, prefix: normalize_keyword(&f, &scalar_string(text)).into_bytes(), max_expansions: opts.max_expansions, case_insensitive: false, rewrite: MultiTermRewrite::ConstantScore });
            }
        }
        if let (Some(fz), true, Some(id)) = (&opts.fuzziness, f.ty.is_keyword_like(), f.core_id) {
            return Ok(Query::Fuzzy {
                field: id,
                text: normalize_keyword(&f, &scalar_string(text)),
                fuzziness: *fz,
                prefix_length: opts.prefix_length,
                transpositions: opts.transpositions,
                max_expansions: opts.max_expansions,
                rewrite: MultiTermRewrite::TopTermsBlendedFreqs(opts.max_expansions),
            });
        }
        return match term_query(&f, text, false, ctx) {
            Ok(q) => Ok(q),
            Err(_) if opts.lenient => Ok(Query::MatchNone),
            Err(e) => Err(e),
        };
    }
    let Some(id) = f.core_id else { return Ok(Query::MatchNone) };
    let analyzer = opts
        .analyzer
        .clone()
        .or_else(|| if opts.quoted { f.search_quote_analyzer.clone() } else { f.search_analyzer.clone() })
        .unwrap_or_else(|| Arc::new(Analyzer::standard()));
    let s = scalar_string(text);
    let positions = analyze_positions(&analyzer, &s);
    if positions.is_empty() {
        return Ok(if opts.zero_terms_all { Query::MatchAll } else { Query::MatchNone });
    }
    let term = |t: &str| Query::Term(Term::new_text(id, t));
    let fuzzy_or_term = |t: &str| match &opts.fuzziness {
        Some(fz) => Query::Fuzzy {
            field: id,
            text: t.to_string(),
            fuzziness: *fz,
            prefix_length: opts.prefix_length,
            transpositions: opts.transpositions,
            max_expansions: opts.max_expansions,
            rewrite: MultiTermRewrite::TopTermsBlendedFreqs(opts.max_expansions),
        },
        None => term(t),
    };
    let position_query = |alts: &Vec<String>, last_prefix: bool| -> Query {
        let mk = |t: &String| {
            if last_prefix {
                Query::Prefix { field: id, prefix: t.as_bytes().to_vec(), max_expansions: opts.max_expansions, case_insensitive: false, rewrite: MultiTermRewrite::ConstantScore }
            } else {
                fuzzy_or_term(t)
            }
        };
        if alts.len() == 1 {
            mk(&alts[0])
        } else if !last_prefix && opts.fuzziness.is_none() {
            // Terms stacked on one position score as one pseudo term (Lucene `SynonymQuery`).
            super::synonym::query(id, alts)
        } else {
            mk_bool(vec![], alts.iter().map(mk).collect(), vec![], vec![], MinShouldMatch::Count(1))
        }
    };
    match opts.kind {
        MatchKind::Bool | MatchKind::BoolPrefix => {
            let n = positions.len();
            let clauses: Vec<Query> = positions.iter().enumerate().map(|(i, alts)| position_query(alts, opts.kind == MatchKind::BoolPrefix && i + 1 == n)).collect();
            if clauses.len() == 1 && opts.msm.is_none() {
                return Ok(clauses.into_iter().next().unwrap());
            }
            if opts.operator_and {
                Ok(mk_bool(clauses, vec![], vec![], vec![], MinShouldMatch::Count(0)))
            } else {
                let msm = match &opts.msm {
                    Some(m) => m.clone(),
                    None => MinShouldMatch::Count(1),
                };
                Ok(mk_bool(vec![], clauses, vec![], vec![], msm))
            }
        }
        MatchKind::Phrase => {
            if positions.len() == 1 {
                return Ok(position_query(&positions[0], false));
            }
            if positions.iter().all(|p| p.len() == 1) {
                Ok(Query::Phrase { terms: positions.iter().map(|p| Term::new_text(id, &p[0])).collect(), slop: opts.slop })
            } else {
                Ok(Query::MultiPhrase { field: id, positions: positions.iter().map(|p| p.iter().map(|t| Term::new_text(id, t)).collect()).collect(), slop: opts.slop })
            }
        }
        MatchKind::PhrasePrefix => {
            let last = positions.last().unwrap();
            let mut expanded: Vec<String> = Vec::new();
            for p in last {
                for t in expand_prefix(id, p, opts.max_expansions, ctx) {
                    if !expanded.contains(&t) {
                        expanded.push(t);
                    }
                }
            }
            if positions.len() == 1 {
                return Ok(Query::Prefix { field: id, prefix: last[0].as_bytes().to_vec(), max_expansions: opts.max_expansions, case_insensitive: false, rewrite: MultiTermRewrite::ConstantScore });
            }
            if expanded.is_empty() {
                return Ok(Query::MatchNone);
            }
            let mut pos: Vec<Vec<Term>> = positions[..positions.len() - 1].iter().map(|p| p.iter().map(|t| Term::new_text(id, t)).collect()).collect();
            pos.push(expanded.iter().map(|t| Term::new_text(id, t)).collect());
            Ok(Query::MultiPhrase { field: id, positions: pos, slop: opts.slop })
        }
    }
}

// ---------------------------------------------------------------------------
// multi_match.

/// `["title^2", "body", "*_name"]` → expanded `(field, boost)`.
pub fn expand_fields(specs: &[String], ctx: &QueryContext) -> Vec<(String, f32)> {
    let mut out: Vec<(String, f32)> = Vec::new();
    for s in specs {
        let (name, boost) = match s.rsplit_once('^') {
            // Validated where the specs come from (`check_boosts`,
            // `default_specs`).
            Some((n, b)) => (n.to_string(), java_parse_float(b).unwrap_or(1.0)),
            None => (s.clone(), 1.0),
        };
        if name.contains('*') {
            for m in ctx.lookup.matching(&name) {
                if let Some(f) = ctx.lookup.get(&m) {
                    let eligible = f.ty.is_text() || f.ty.is_keyword_like() || f.ty.is_numeric() || f.ty.is_date() || f.ty == EsType::Boolean || f.ty == EsType::Ip;
                    // Strict columnar: `*` resolves to the indexed fields only.
                    let eligible = eligible && (name != "*" || !ctx.lookup.registry.strict_columnar() || f.index_declared);
                    if eligible && !out.iter().any(|(n, _)| n == &m) {
                        out.push((m, boost));
                    }
                }
            }
        } else if !out.iter().any(|(n, _)| n == &name) {
            out.push((name, boost));
        }
    }
    out
}

fn parse_multi_match(body: &Value, ctx: &QueryContext) -> Result<Query> {
    let o = obj("multi_match", body)?;
    let text = o.get("query").ok_or_else(|| parsing("No text specified for multi_match query"))?;
    let ty = o.get("type").and_then(Value::as_str).unwrap_or("best_fields");
    let kind = match ty {
        "best_fields" | "most_fields" | "cross_fields" => MatchKind::Bool,
        "phrase" => MatchKind::Phrase,
        "phrase_prefix" => MatchKind::PhrasePrefix,
        "bool_prefix" => MatchKind::BoolPrefix,
        other => return Err(EsError::new(400, "parse_exception", format!("failed to parse [multi_match] query type [{other}]. unknown type."))),
    };
    if ty == "bool_prefix" && o.contains_key("slop") {
        return Err(parsing("[slop] not allowed for type [bool_prefix]"));
    }
    let mut opts = match_opts_from(o, kind, ctx)?;
    let specs: Vec<String> = match o.get("fields") {
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
        Some(Value::String(s)) => vec![s.clone()],
        _ => default_specs(ctx)?,
    };
    let lenient_default = o.get("fields").is_none() || specs.iter().all(|s| s == "*");
    if lenient_default && o.get("lenient").is_none() {
        opts.lenient = true;
    }
    let fields = expand_fields(&specs, ctx);
    let tie = o.get("tie_breaker").and_then(f32_of);
    let boost = boost_of(o)?;
    if fields.is_empty() {
        return Ok(Query::MatchNone);
    }
    let q = match ty {
        "cross_fields" => super::cross_fields::build(text, &fields, &opts, tie.unwrap_or(0.0), ctx)?,
        _ => {
            let mut qs = Vec::new();
            for (fname, b) in &fields {
                match match_field(fname, text, &opts, ctx) {
                    Ok(Query::MatchNone) => {}
                    Ok(q) => qs.push(with_boost(q, *b)),
                    Err(_) if opts.lenient => {}
                    Err(e) => return Err(e),
                }
            }
            if qs.is_empty() {
                Query::MatchNone
            } else if ty == "most_fields" {
                mk_bool(vec![], qs, vec![], vec![], MinShouldMatch::Count(1))
            } else if qs.len() == 1 {
                qs.into_iter().next().unwrap()
            } else {
                Query::DisMax { queries: qs, tie_breaker: tie.unwrap_or(0.0) }
            }
        }
    };
    Ok(with_boost(q, boost))
}

// ---------------------------------------------------------------------------
// function_score.

pub(crate) fn parse_function(o: &Map<String, Value>, ctx: &QueryContext) -> Result<Option<ScoreFunction>> {
    let mut func: Option<ScoreFunction> = None;
    let mut found: Option<&str> = None;
    for (k, v) in o {
        if matches!(k.as_str(), "field_value_factor" | "random_score" | "gauss" | "exp" | "linear" | "script_score") {
            if let Some(prev) = found {
                return Err(parsing(format!("failed to parse function_score functions. already found [{prev}], now encountering [{k}].")));
            }
            found = Some(k.as_str());
        }
        match k.as_str() {
            "field_value_factor" => {
                let fo = obj(k, v)?;
                let field = fo.get("field").and_then(Value::as_str).ok_or_else(|| parsing("[field_value_factor] requires [field]"))?;
                let factor = fo.get("factor").and_then(f32_of).unwrap_or(1.0);
                let modifier = match fo.get("modifier").and_then(Value::as_str).unwrap_or("none") {
                    "none" => FieldValueModifier::None,
                    "log" => FieldValueModifier::Log,
                    "log1p" => FieldValueModifier::Log1p,
                    "log2p" => FieldValueModifier::Log2p,
                    "ln" => FieldValueModifier::Ln,
                    "ln1p" => FieldValueModifier::Ln1p,
                    "ln2p" => FieldValueModifier::Ln2p,
                    "square" => FieldValueModifier::Square,
                    "sqrt" => FieldValueModifier::Sqrt,
                    "reciprocal" => FieldValueModifier::Reciprocal,
                    other => return Err(EsError::illegal_argument(format!("No enum constant org.elasticsearch.common.lucene.search.function.FieldValueFactorFunction.Modifier.{}", other.to_ascii_uppercase()))),
                };
                let missing = fo.get("missing").and_then(Value::as_f64);
                let f = ctx.lookup.get(field);
                match f.and_then(|f| f.core_id) {
                    Some(id) => func = Some(ScoreFunction::FieldValueFactor { field: id, factor, modifier, missing }),
                    None => {
                        if missing.is_none() {
                            return Err(EsError::new(400, "exception", format!("Unable to find a field mapper for field [{field}]. No 'missing' value defined.")));
                        }
                        func = Some(ScoreFunction::Weight((missing.unwrap() * factor as f64) as f32));
                    }
                }
            }
            "random_score" => {
                let ro = v.as_object().cloned().unwrap_or_default();
                // `RandomScoreFunctionBuilder.doToFunction`: a seeded score reads
                // its randomness from `field` (ES 9 default `_seq_no`).
                if ro.get("seed").is_some_and(|s| !s.is_null()) {
                    let index = format!("[{}/{}]", ctx.index_name, ctx.index_uuid);
                    let field = match ro.get("field").and_then(Value::as_str) {
                        Some(f) => f.to_string(),
                        None if ctx.seq_no_disabled => {
                            return Err(ctx.shard_err(format!("failed to create query: random_score requires a [field] parameter when [index.disable_sequence_numbers] is [true] on index [{index}]")));
                        }
                        None => "_seq_no".to_string(),
                    };
                    let mapped = matches!(field.as_str(), "_seq_no" | "_id" | "_index" | "_routing" | "_version" | "_primary_term") || ctx.lookup.get(&field).is_some();
                    if !mapped && !ctx.lookup.fields.is_empty() {
                        return Err(ctx.shard_err(format!("failed to create query: Field [{field}] is not mapped on [{index}] and cannot be used as a source of random numbers.")));
                    }
                }
                let seed = ro.get("seed").map(|s| match s {
                    Value::Number(n) => n.as_u64().unwrap_or(0),
                    Value::String(s) => s.bytes().fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64)),
                    _ => 0,
                }).unwrap_or_else(|| crate::date::now_millis() as u64);
                func = Some(ScoreFunction::RandomScore { seed });
            }
            "gauss" | "exp" | "linear" => {
                let kind = match k.as_str() {
                    "gauss" => DecayKind::Gauss,
                    "exp" => DecayKind::Exp,
                    _ => DecayKind::Linear,
                };
                let fo = obj(k, v)?;
                let (field, spec) = fo.iter().find(|(n, _)| n.as_str() != "multi_value_mode").ok_or_else(|| parsing(format!("[{k}] requires a field")))?;
                let so = spec.as_object().ok_or_else(|| parsing(format!("[{k}] field must be an object")))?;
                let Some(rf) = ctx.lookup.get(field) else {
                    return Err(parsing(format!("unknown field [{field}]")));
                };
                if rf.ty == EsType::GeoPoint {
                    return Err(ctx.shard_err("geo decay functions are not supported yet"));
                }
                let id = rf.core_id.ok_or_else(|| ctx.shard_err(format!("unknown field [{field}]")))?;
                let decay = so.get("decay").and_then(Value::as_f64).unwrap_or(0.5);
                if !(decay > 0.0 && decay < 1.0) {
                    return Err(ctx.shard_err("failed to create query: function_score : decay must be in the range [0..1]."));
                }
                let (origin, scale, offset) = if rf.ty.is_date() {
                    let origin = match so.get("origin") {
                        Some(v) => date_value(rf, v, None, None, false, ctx)? as f64,
                        None => ctx.now as f64,
                    };
                    let tv = |key: &str| -> Result<f64> {
                        match so.get(key) {
                            Some(Value::String(s)) => crate::date::parse_time_value(s).map(|x| x as f64).ok_or_else(|| parsing(format!("failed to parse [{key}]"))),
                            Some(Value::Number(n)) => Ok(n.as_f64().unwrap_or(0.0)),
                            _ => Ok(0.0),
                        }
                    };
                    (origin, tv("scale")?, tv("offset")?)
                } else {
                    let n = |key: &str| so.get(key).and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())));
                    (
                        n("origin").zip(n("scale")).map(|(o, _)| o).ok_or_else(|| EsError::new(400, "parse_exception", "both [scale] and [origin] must be set for numeric fields."))?,
                        n("scale").ok_or_else(|| EsError::new(400, "parse_exception", "both [scale] and [origin] must be set for numeric fields."))?,
                        n("offset").unwrap_or(0.0),
                    )
                };
                func = Some(ScoreFunction::Decay { field: id, origin, scale, offset, decay, kind });
            }
            "script_score" => return Err(ctx.shard_err("[script_score] functions are not supported (no scripting engine)")),
            "weight" | "filter" | "_name" => {}
            _ => return Err(parsing("unknown query [function_score] did you mean any of [random_score, script_score]?")),
        }
    }
    let weight = o.get("weight").and_then(f32_of);
    let func = match (func, weight) {
        (Some(f), Some(w)) => Some(ScoreFunction::Weighted { function: Box::new(f), weight: w }),
        (Some(f), None) => Some(f),
        (None, Some(w)) => Some(ScoreFunction::Weight(w)),
        (None, None) => None,
    };
    let func = match (func, o.get("filter")) {
        (Some(f), Some(filter)) => Some(ScoreFunction::Filtered { filter: Box::new(parse_query(filter, ctx).map_err(|e| e.default_key("filter", Some("functions")))?), function: Box::new(f) }),
        (f, _) => f,
    };
    Ok(func)
}

fn parse_function_score(body: &Value, ctx: &QueryContext) -> Result<Query> {
    let o = obj("function_score", body)?;
    for k in o.keys() {
        if !matches!(
            k.as_str(),
            "query" | "functions" | "score_mode" | "boost_mode" | "max_boost" | "min_score" | "boost" | "_name" | "field_value_factor" | "random_score" | "gauss" | "exp" | "linear" | "script_score" | "weight" | "filter"
        ) {
            return Err(parsing(format!("failed to parse [function_score] query. field [{k}] is not supported")));
        }
    }
    if crate::script::search::function_score_has_script(o) {
        return crate::script::search::function_score_query(o, ctx, &parse_query, &parse_function);
    }
    if super::geo_decay::has_geo_decay(o, ctx) {
        return super::geo_decay::function_score_query(o, ctx);
    }
    let query = match o.get("query") {
        Some(q) => parse_query(q, ctx).map_err(|e| e.default_key("query", Some("function_score")))?,
        None => Query::MatchAll,
    };
    let mut functions = Vec::new();
    if let Some(fs) = o.get("functions").and_then(Value::as_array) {
        for f in fs {
            let fo = obj("functions", f)?;
            if let Some(func) = parse_function(fo, ctx)? {
                functions.push(func);
            }
        }
    }
    // Single-function shorthand.
    let mut top = Map::new();
    for k in ["field_value_factor", "random_score", "gauss", "exp", "linear", "script_score", "weight"] {
        if let Some(v) = o.get(k) {
            top.insert(k.to_string(), v.clone());
        }
    }
    if !top.is_empty() {
        if let Some(func) = parse_function(&top, ctx)? {
            functions.push(func);
        }
    }
    let score_mode = match o.get("score_mode").and_then(Value::as_str).unwrap_or("multiply") {
        "multiply" => ScoreMode::Multiply,
        "sum" => ScoreMode::Sum,
        "avg" => ScoreMode::Avg,
        "first" => ScoreMode::First,
        "max" => ScoreMode::Max,
        "min" => ScoreMode::Min,
        other => return Err(EsError::illegal_argument(format!("No enum constant org.elasticsearch.common.lucene.search.function.FunctionScoreQuery.ScoreMode.{}", other.to_ascii_uppercase()))),
    };
    let boost_mode = match o.get("boost_mode").and_then(Value::as_str).unwrap_or("multiply") {
        "multiply" => BoostMode::Multiply,
        "replace" => BoostMode::Replace,
        "sum" => BoostMode::Sum,
        "avg" => BoostMode::Avg,
        "max" => BoostMode::Max,
        "min" => BoostMode::Min,
        other => return Err(EsError::illegal_argument(format!("No enum constant org.elasticsearch.common.lucene.search.function.CombineFunction.{}", other.to_ascii_uppercase()))),
    };
    let q = Query::FunctionScore {
        query: Box::new(query),
        functions,
        score_mode,
        boost_mode,
        max_boost: o.get("max_boost").and_then(f32_of).unwrap_or(f32::MAX),
        min_score: o.get("min_score").and_then(f32_of),
    };
    Ok(with_boost(q, boost_of(o)?))
}

// ---------------------------------------------------------------------------
// knn.

/// `knn` (query or top-level section entry). `default_k` = request size for
/// the top-level section.
pub fn parse_knn_query(o: &Map<String, Value>, ctx: &QueryContext, default_k: Option<usize>) -> Result<Query> {
    // element types, index options, ES errors, similarity thresholds,
    // nested vector fields: see `crate::vectors`.
    crate::vectors::parse_knn(o, ctx, default_k)
}

// ---------------------------------------------------------------------------
// nested.

/// ES `nested` query → core block join.
fn parse_nested(o: &Map<String, Value>, ctx: &QueryContext) -> Result<Query> {
    for k in o.keys() {
        if !matches!(k.as_str(), "path" | "query" | "score_mode" | "ignore_unmapped" | "inner_hits" | "boost" | "_name") {
            return Err(parsing(format!("[nested] query does not support [{k}]")));
        }
    }
    let path = o.get("path").and_then(Value::as_str).ok_or_else(|| parsing("[nested] requires 'path' field"))?;
    if !ctx.allow_expensive_queries {
        return Err(ctx.shard_err("[joining] queries cannot be executed when 'search.allow_expensive_queries' is set to false."));
    }
    let inner = o.get("query").ok_or_else(|| EsError::illegal_argument("[nested] requires 'query' field"))?;
    let ignore_unmapped = o.get("ignore_unmapped").and_then(Value::as_bool).unwrap_or(false);
    let score_mode = match o.get("score_mode").and_then(Value::as_str) {
        None => rustic_core::nested::NestedScoreMode::Avg,
        Some(m) => rustic_core::nested::NestedScoreMode::parse(m).ok_or_else(|| EsError::illegal_argument(format!("No score mode for child query [{m}] found")))?,
    };
    if let Some(f) = crate::inference::semantic_query::chunks_field(ctx, path) {
        let f = f.clone();
        let q = crate::inference::semantic_query::nested(&f, inner, score_mode, ctx)?;
        return Ok(with_boost(q, boost_of(o)?));
    }
    match ctx.lookup.objects.get(path) {
        Some(info) if info.nested => {}
        _ if ignore_unmapped => return Ok(Query::MatchNone),
        _ => return Err(ctx.shard_err(format!("failed to create query: [nested] failed to find nested object under path [{path}]"))),
    }
    let q = parse_query(inner, ctx).map_err(|e| e.default_key("query", Some("nested")))?;
    Ok(with_boost(Query::Nested { path: path.to_string(), query: Box::new(q), score_mode, filter: false }, boost_of(o)?))
}

/// Is `spec` a valid `minimum_should_match` combination (`2<75%`, `-25%`, `3`)?
fn msm_spec_valid(spec: &str) -> bool {
    let num = |t: &str| {
        let t = t.trim().trim_end_matches('%');
        !t.is_empty() && t.trim_start_matches('-').chars().all(|c| c.is_ascii_digit())
    };
    !spec.trim().is_empty()
        && spec.split_whitespace().all(|part| match part.split_once('<') {
            Some((a, b)) => num(a) && num(b),
            None => num(part),
        })
}

#[cfg(test)]
mod date_round_up_tests {
    use super::*;
    use crate::date::CalUnit;

    /// The unit a custom pattern's bound rounds up to the end of: ES's
    /// round-up parser defaults a missing month and day to 1 and only the
    /// missing time fields to their maximum, so every date-only pattern
    /// ends its (first) day. Named formats and alternatives keep their
    /// own rules.
    #[test]
    fn custom_patterns_round_up_to_their_finest_unit() {
        let unit = |s: &str| finest_pattern_unit(s);
        for day in ["dd/MM/yyyy", "yyyy-MM", "yyyy", "uuuu", "YYYY", "yyyy-DDD", "YYYY-'W'ww", "YYYY-ww-e", "MM/dd", "uuuu-MM-dd", "G yyyy", "yyyy QQQ"] {
            assert_eq!(unit(day), Some(Some(CalUnit::Day)), "{day}");
        }
        assert_eq!(unit("yyyyMMddHH"), Some(Some(CalUnit::Hour)));
        assert_eq!(unit("dd/MM/yyyy hh a"), Some(Some(CalUnit::Hour)));
        assert_eq!(unit("yyyy-MM-dd'T'HH:mm"), Some(Some(CalUnit::Minute)));
        assert_eq!(unit("HH:mm"), Some(Some(CalUnit::Minute)));
        assert_eq!(unit("yyyy-MM-dd HH:mm:ss"), Some(Some(CalUnit::Second)));
        assert_eq!(unit("yyyy-MM-dd HH:mm:ss.SSS"), Some(None));
        assert_eq!(unit("'day' dd/MM/yyyy"), Some(Some(CalUnit::Day)), "quoted letters are literals");
        assert_eq!(unit("'at' HH 'h'"), Some(Some(CalUnit::Hour)), "quoted letters are literals");
        for named in ["strict_date_optional_time", "date", "year", "year_month", "basic_date", "yyyy-MM-dd||epoch_millis", "epoch_millis", "epoch_second"] {
            assert_eq!(unit(named), None, "{named}");
        }
    }

    const DAY: i64 = 86_400_000;

    fn bound(value: &str, spec: &str, round_up: bool) -> i64 {
        let fmt = DateFormatter::new(spec).unwrap();
        text_date_bound(value, spec, &fmt, Tz::UTC, 0, round_up).unwrap()
    }

    fn utc(s: &str) -> i64 {
        DateFormatter::new("strict_date_optional_time").unwrap().parse(s).unwrap()
    }

    /// `gte` and `lt` (no round-up) take the value as parsed; `gt` and
    /// `lte` (round-up) end its first day for a date-only pattern.
    #[test]
    fn only_gt_and_lte_round_up() {
        for spec in ["uuuu", "yyyy"] {
            assert_eq!(bound("1500", spec, false), utc("1500-01-01T00:00:00Z"), "{spec}");
            assert_eq!(bound("1500", spec, true), utc("1500-01-01T23:59:59.999Z"), "{spec}");
        }
        assert_eq!(bound("2019-02", "yyyy-MM", false), utc("2019-02-01T00:00:00Z"));
        assert_eq!(bound("2019-02", "yyyy-MM", true), utc("2019-02-01T23:59:59.999Z"));
        assert_eq!(bound("23/02/2019", "dd/MM/yyyy", false), utc("2019-02-23T00:00:00Z"));
        assert_eq!(bound("23/02/2019", "dd/MM/yyyy", true), utc("2019-02-23T23:59:59.999Z"));
        assert_eq!(bound("2019-02-23 10:30", "yyyy-MM-dd HH:mm", true), utc("2019-02-23T10:30:59.999Z"));
        assert_eq!(bound("2019-02-23 10:30", "yyyy-MM-dd HH:mm", false), utc("2019-02-23T10:30:00Z"));
        assert_eq!(bound("2019-02-23 10:30:15.250", "yyyy-MM-dd HH:mm:ss.SSS", true), utc("2019-02-23T10:30:15.250Z"));
        // 2019-02-23 in dd/MM/yyyy, gte and lte the same day: the whole day.
        let day = 1_550_880_000_000i64;
        assert_eq!((bound("23/02/2019", "dd/MM/yyyy", false), bound("23/02/2019", "dd/MM/yyyy", true)), (day, day + DAY - 1));
    }

    /// Each range scenario of ES's `search/500_date_range.yml` (docs at
    /// 12:12:12.123 on 1900-01-01, 2022-01-01, 2022-01-03, 1500-01-01 and
    /// 1500-01-05), counted against the bounds `format: uuuu` yields.
    #[test]
    fn yaml_500_date_range_scenarios() {
        let docs = ["1900-01-01", "2022-01-01", "2022-01-03", "1500-01-01", "1500-01-05"].map(|d| utc(&format!("{d}T12:12:12.123Z")));
        let hits = |gte: &str, lte: &str| -> Vec<usize> {
            let (lo, hi) = (bound(gte, "uuuu", false), bound(lte, "uuuu", true));
            (0..docs.len()).filter(|&i| lo <= docs[i] && docs[i] <= hi).map(|i| i + 1).collect()
        };
        // "test range query for all docs with year uuuu".
        assert_eq!(hits("1000", "2023"), vec![1, 2, 3, 4, 5]);
        // "test match query gte and lt for single result with year uuuu":
        // lte 1500 ends 1500-01-01, so 1500-01-05 is out.
        assert_eq!(hits("1500", "1500"), vec![4]);
        // "test match query gte and lte with year uuuu".
        assert_eq!(hits("1500", "2000"), vec![1, 4, 5]);
        // "test match query with year uuuu": a term on a `uuuu` field
        // matches 1500 only, against the field's 1500, 1900, 2022, 2022.
        let field = ["1900", "2022", "2022", "1500"].map(|y| bound(y, "uuuu", false));
        let (lo, hi) = (bound("1500", "uuuu", false), bound("1500", "uuuu", true));
        assert_eq!((0..4).filter(|&i| lo <= field[i] && field[i] <= hi).map(|i| i + 1).collect::<Vec<_>>(), vec![4]);
    }
}

#[cfg(test)]
mod boost_tests {
    use super::*;
    use serde_json::json;

    /// ES rejects negative boosts (`-0.0` included) with a 400; NaN and
    /// infinite ones fail Lucene's `BoostQuery` check.
    #[test]
    fn negative_and_non_finite_boosts_are_rejected() {
        for b in [0.0f32, 1.0, 2.5, f32::MAX] {
            assert_eq!(valid_boost(b).unwrap(), b);
        }
        for b in [-2.0f32, -0.0, f32::NEG_INFINITY] {
            let e = valid_boost(b).unwrap_err();
            assert_eq!((e.status, e.kind.as_str(), e.reason.as_str()), (400, "illegal_argument_exception", "negative [boost] are not allowed."), "{b}");
        }
        for (b, shown) in [(f32::NAN, "NaN"), (f32::INFINITY, "Infinity")] {
            let e = valid_boost(b).unwrap_err();
            assert_eq!((e.status, e.reason), (400, format!("boost must be a positive float, got {shown}")));
        }
        for (name, body) in [
            ("bool", json!({"should": [], "boost": -1})),
            ("match", json!({"title": {"query": "foo", "boost": -2}})),
            ("term", json!({"k": {"value": "x", "boost": "-0.5"}})),
            ("dis_max", json!({"queries": [], "boost": "NaN"})),
            ("multi_match", json!({"query": "foo", "fields": ["title^-2", "body"]})),
            ("query_string", json!({"query": "foo", "fields": ["title^2", "body^1e39"]})),
            ("multi_match", json!({"query": "foo", "fields": "title^-2"})),
            ("query_string", json!({"query": "foo", "fields": [format!("b^1{}", "0".repeat(39))]})),
            ("term", json!({"boost": -1, "k": {"value": "x"}})),
            ("function_score", json!({"query": {"match_all": {}}, "functions": [{"weight": -2}]})),
            ("function_score", json!({"weight": -1})),
            ("multi_match", json!({"query": "foo", "fields": ["x^abc"]})),
            ("multi_match", json!({"query": "foo", "fields": ["x^NaN"]})),
        ] {
            assert_eq!(check_boosts(name, &body).unwrap_err().status, 400, "{name} {body}");
        }
        // `indices_boost` fails Lucene's `BoostQuery` check.
        for (b, shown) in [(json!(-1), "-1.0"), (json!("NaN"), "NaN")] {
            let e = crate::search_strict::check_indices_boost(&json!([{"i": b}])).unwrap_err();
            assert_eq!(e.reason, format!("boost must be a positive float, got {shown}"));
        }
        crate::search_strict::check_indices_boost(&json!([{"i": 2}, {"j": 0}])).unwrap();
        // ObjectParser builders wrap it.
        let e = check_boosts("match_all", &json!({"boost": -1})).unwrap_err();
        assert_eq!((e.kind.as_str(), e.reason.as_str()), ("parsing_exception", "[match_all] failed to parse field [boost]"));
        let e = check_boosts("knn", &json!({"field": "v", "query_vector": [1.0], "boost": -1})).unwrap_err();
        assert_eq!((e.kind.as_str(), e.reason.as_str()), ("x_content_parse_exception", "[knn] failed to parse field [boost]"));
        assert!(check_boosts("terms_set", &json!({"f": {"terms": ["a"], "boost": -1}})).is_err());
        // Float.parseFloat semantics.
        assert_eq!(field_weight("x^abc").unwrap_err().reason, "For input string: \"abc\"");
        assert_eq!(field_weight("x^2.5d").unwrap(), Some(2.5));
        assert_eq!(field_weight("x").unwrap(), None);
        assert!(field_weight("x^Infinity").is_err() && field_weight("x^-1").is_err() && java_parse_float("inf").is_none());
        assert_eq!(java_parse_float("0x1.8p1"), Some(3.0));
        assert_eq!(java_parse_float("-0X10p-4f"), Some(-1.0));
        assert!(java_parse_float("0x1.8").is_none() && java_parse_float("Infinityf").is_none() && java_parse_float("1.5x").is_none());
        assert_eq!(java_parse_float("-Infinity"), Some(f32::NEG_INFINITY));
        assert_eq!(field_weight("x^").unwrap_err().reason, "empty String");
        for (name, body) in [
            ("bool", json!({"should": [], "boost": 0})),
            ("match", json!({"title": {"query": "foo", "boost": 2}})),
            ("multi_match", json!({"query": "foo", "fields": ["title^3", "body", "x^2f", "y^ 1.5 "]})),
            // A percolated document's fields, a rule's criteria and sparse
            // vector tokens are data; a lone short-form `boost` is a field.
            ("percolate", json!({"field": "q", "document": {"boost": -1}})),
            ("rule", json!({"match_criteria": {"boost": -1}, "ruleset_ids": ["r"], "organic": {"match_all": {}}})),
            ("sparse_vector", json!({"field": "v", "query_vector": {"boost": -1.0}})),
            ("term", json!({"boost": -1})),
            ("combined_fields", json!({"query": "foo", "fields": ["title^0.5"]})),
            // A query's `default_field` is a field name.
            ("query_string", json!({"query": "foo", "default_field": "title^-1"})),
        ] {
            check_boosts(name, &body).unwrap();
        }
    }
}
