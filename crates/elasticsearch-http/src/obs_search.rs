//! Search-side diagnostics: `_validate/query`, `_terms_enum`, `_explain`,
//! search `profile` output and the `index.blocks.read` check of the search
//! APIs. Queries are parsed with the engine's DSL parser and inspected as
//! core [`Query`] trees (Lucene-style `toString`, profile node types).

use crate::{blocking, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::{EsError, Result};
use elasticsearch_engine::index::IndexService;
use elasticsearch_engine::mapping::{EsType, FieldLookup};
use elasticsearch_engine::node::{Node, ResolveOpts, NODE_ID};
use elasticsearch_engine::query::dsl::{parse_query, QueryContext};
use elasticsearch_engine::rustic_core::query::{Bm25Params, Bound, Explanation, LeafReader, MinShouldMatch, Query, SearchOptions, Searcher};
use elasticsearch_engine::search::Target;
use elasticsearch_engine::shard::Shard;
use elasticsearch_engine::values;
use serde_json::{json, Map, Value};
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Instant;

pub async fn handle(node: Arc<Node>, req: Req, index: Option<String>) -> R {
    let off = if index.is_some() { 1 } else { 0 };
    let api = req.seg(off).to_string();
    let m = req.method.clone();
    if m != Method::GET && m != Method::POST {
        return crate::method_not_allowed(&req, &["GET", "POST"]);
    }
    blocking(move || {
        let mut environment = elasticsearch_engine::script::ScriptEnv::new(
            Some(elasticsearch_engine::script::stored::for_data_dir(&node.data)), None);
        environment.raw_search_scripts = req.extensions.get::<elasticsearch_engine::source_json::Source>()
            .map(elasticsearch_engine::source_json::search_scripts).unwrap_or_default();
        with_query_errors(environment, || match api.as_str() {
        "_validate" if req.seg(off + 1) == "query" => validate(&node, &req, index.as_deref()),
        "_terms_enum" => terms_enum(&node, &req, index.as_deref()),
        "_explain" => match index.as_deref() {
            Some(i) => explain(&node, &req, i),
            None => Err(EsError::action_validation(&["index is missing", "id is missing"])),
        },
        _ => crate::not_found_handler(&req),
        })
    })
    .await
}

/// Diagnostics parse/execute queries too. Keep their typed lazy scorer errors
/// in a request scope so a percolator failure cannot become a false nonmatch.
fn with_query_errors<T>(environment: elasticsearch_engine::script::ScriptEnv, f: impl FnOnce() -> Result<T>) -> Result<T> {
    elasticsearch_engine::script::with_env(environment, || {
        let result = f();
        match elasticsearch_engine::script::env().and_then(|env| env.take_error()) {
            Some(error) => Err(error),
            None => result,
        }
    })
}

/// `index.blocks.read` enforcement for the search APIs.
pub fn check_read_blocks(targets: &[Target]) -> Result<()> {
    for t in targets {
        t.svc.check_read_block()?;
    }
    Ok(())
}

fn opts(req: &Req) -> ResolveOpts {
    ResolveOpts::from_params(&req.params)
}

/// The first local shard copy of an index (per-index profiling runs on one
/// shard); `None` when no copy is open here (a sparse, cluster-managed index).
fn first_shard(svc: &IndexService) -> Option<std::sync::Arc<elasticsearch_engine::index::IndexShard>> {
    svc.local_shards().into_iter().next()
}

/// Run `f` with a query context over the first local shard copy of `svc`,
/// or over no leaves (mapping only) when no copy is open here.
fn with_index_ctx<T>(svc: &IndexService, f: impl FnOnce(&QueryContext, &[LeafReader<'_>], &FieldLookup, &HashMap<u16, String>) -> T) -> T {
    if let Some(sh) = first_shard(svc) {
        return with_ctx(&sh, |ctx, leaves, lookup, names| f(ctx, leaves, lookup, names));
    }
    let lookup = svc.lookup();
    let names: HashMap<u16, String> = lookup.core_schema().into_iter().map(|f| (f.id, f.name)).collect();
    let ctx = QueryContext {
        deadline: None,
        lookup: &lookup,
        index_name: &svc.name,
        index_uuid: &svc.uuid,
        aliases: svc.aliases().keys().cloned().collect(),
        now: elasticsearch_engine::date::now_millis(),
        leaves: &[],
        default_fields: svc.query_default_fields(),
        terms_lookup: None,
        max_terms_count: svc.setting_usize("index.max_terms_count", 65_536),
        max_regex_length: svc.setting_usize("index.max_regex_length", 1000),
        max_result_window: svc.setting_usize("index.max_result_window", 10_000),
        seq_no_disabled: svc.seq_no_disabled(),
        allow_unmapped_fields: true,
        knn_prefilter: None,
        knn_prefilter_lookup: None,
        allow_expensive_queries: svc.setting("index.search.allow_expensive_queries").map(|v| v != "false").unwrap_or(true),
        mappings: Some(&**svc),
    };
    f(&ctx, &[], &lookup, &names)
}

/// Run `f` with a query context over one shard copy's published reader
/// view (no shard lock is held while `f` runs).
fn with_ctx<T>(svc: &elasticsearch_engine::index::IndexShard, f: impl FnOnce(&QueryContext, &[LeafReader<'_>], &FieldLookup, &HashMap<u16, String>) -> T) -> T {
    let view = svc.shard.reader_view();
    let leaves = view.leaves();
    let lookup = svc.lookup();
    let names: HashMap<u16, String> = svc.shard.core_schema().iter().map(|f| (f.id, f.name.clone())).collect();
    let ctx = QueryContext {
        deadline: None,
        lookup: &lookup,
        index_name: &svc.name,
        index_uuid: &svc.uuid,
        aliases: svc.aliases().keys().cloned().collect(),
        now: elasticsearch_engine::date::now_millis(),
        leaves: &leaves,
        default_fields: svc.query_default_fields(),
        terms_lookup: None,
        max_terms_count: svc.setting_usize("index.max_terms_count", 65_536),
        max_regex_length: svc.setting_usize("index.max_regex_length", 1000),
        max_result_window: svc.setting_usize("index.max_result_window", 10_000),
        seq_no_disabled: svc.seq_no_disabled(),
        allow_unmapped_fields: true,
        knn_prefilter: None,
        knn_prefilter_lookup: None,
        allow_expensive_queries: svc.setting("index.search.allow_expensive_queries").map(|v| v != "false").unwrap_or(true),
        mappings: Some(&**svc),
    };
    f(&ctx, &leaves, &lookup, &names)
}

/// `q=` URI query (+ `df`, `default_operator`, `analyzer`, `lenient`, ...).
fn uri_query(req: &Req) -> Option<Value> {
    let q = req.param("q")?;
    let mut qs = Map::new();
    qs.insert("query".into(), json!(q));
    for (p, k) in [("df", "default_field"), ("default_operator", "default_operator"), ("analyzer", "analyzer"), ("analyze_wildcard", "analyze_wildcard"), ("lenient", "lenient")] {
        if let Some(v) = req.param(p) {
            let val = match v {
                "true" => json!(true),
                "false" => json!(false),
                _ => json!(v),
            };
            qs.insert(k.into(), val);
        }
    }
    Some(json!({ "query_string": qs }))
}

fn camel(kind: &str) -> String {
    kind.split('_').map(|p| {
        let mut c = p.chars();
        match c.next() {
            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            None => String::new(),
        }
    }).collect()
}

fn java_class(kind: &str) -> String {
    match kind {
        "parsing_exception" => "org.elasticsearch.common.ParsingException".into(),
        "x_content_parse_exception" => "org.elasticsearch.xcontent.XContentParseException".into(),
        "query_shard_exception" => "org.elasticsearch.index.query.QueryShardException".into(),
        "illegal_argument_exception" => "java.lang.IllegalArgumentException".into(),
        "number_format_exception" => "java.lang.NumberFormatException".into(),
        "named_object_not_found_exception" => "org.elasticsearch.xcontent.NamedObjectNotFoundException".into(),
        "parse_exception" => "org.elasticsearch.ElasticsearchParseException".into(),
        other => format!("org.elasticsearch.{}", camel(other)),
    }
}

const QUERY_NAMES: &[&str] = &[
    "bool", "boosting", "combined_fields", "constant_score", "dis_max", "distance_feature", "exists", "function_score", "fuzzy", "geo_bounding_box", "geo_distance",
    "geo_polygon", "geo_shape", "ids", "intervals", "knn", "match", "match_all", "match_bool_prefix", "match_none", "match_phrase", "match_phrase_prefix", "more_like_this",
    "multi_match", "nested", "percolate", "pinned", "prefix", "query_string", "range", "rank_feature", "regexp", "script", "script_score", "simple_query_string",
    "span_containing", "span_field_masking", "span_first", "span_multi", "span_near", "span_not", "span_or", "span_term", "span_within", "sparse_vector", "term",
    "terms", "terms_set", "text_expansion", "wildcard", "wrapper",
];

/// 1-based `(line, col)` of the value following the key `"name":` in a JSON
/// body (the token position Jackson reports); `(1, 1)` when not found.
fn value_pos(body: &[u8], name: &str) -> (usize, usize) {
    let key = format!("\"{name}\"");
    let Some(k) = body.windows(key.len()).position(|w| w == key.as_bytes()) else { return (1, 1) };
    let mut i = k + key.len();
    while i < body.len() && (body[i].is_ascii_whitespace() || body[i] == b':') {
        i += 1;
    }
    let line = 1 + body[..i].iter().filter(|&&b| b == b'\n').count();
    let line_start = body[..i].iter().rposition(|&b| b == b'\n').map(|p| p + 1).unwrap_or(0);
    (line, i - line_start + 1)
}

/// `Throwable.toString()`-style message of an error (validate API).
fn error_string(e: &EsError) -> String {
    let mut reason = e.reason.clone();
    if let Some(rest) = reason.strip_prefix("unknown query [") {
        if let Some(name) = rest.strip_suffix(']') {
            reason.push_str(&crate::observability::did_you_mean(name, QUERY_NAMES));
        }
    }
    let mut s = format!("{}: {reason}", java_class(&e.kind));
    // ElasticsearchException#getDetailedMessage: `msg; <cause toString>` down the chain.
    let mut c = e.caused_by.as_deref();
    while let Some(cause) = c {
        s.push_str(&format!("; {}: {}", java_class(&cause.kind), cause.reason));
        c = cause.caused_by.as_deref();
    }
    s
}

// ---------------------------------------------------------------------------
// Lucene-style query strings
// ---------------------------------------------------------------------------

struct Names<'a> {
    ids: &'a HashMap<u16, String>,
    lookup: &'a FieldLookup,
}

impl Names<'_> {
    fn field(&self, id: u16) -> String {
        self.ids.get(&id).cloned().unwrap_or_else(|| format!("field{id}"))
    }
    fn term_text(&self, field: u16, b: &[u8]) -> String {
        let s = String::from_utf8_lossy(b).into_owned();
        let name = self.field(field);
        if self.lookup.get(&name).map(|f| f.ty == EsType::Ip).unwrap_or(false) {
            return values::ip_display(&s);
        }
        s
    }
}

fn java_num(v: f64) -> String {
    if v.is_infinite() {
        return if v > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{:.1}", v)
    } else {
        format!("{v}")
    }
}

fn boost_str(b: f32) -> String {
    java_num(b as f64)
}

fn bytes_bound(b: &Bound<Vec<u8>>, n: &Names, field: u16, lower: bool) -> (String, bool) {
    match b {
        Bound::Included(v) => (n.term_text(field, v), true),
        Bound::Excluded(v) => (n.term_text(field, v), false),
        Bound::Unbounded => ("*".into(), lower),
    }
}

fn i64_range(lo: &Bound<i64>, hi: &Bound<i64>) -> (i64, i64) {
    let l = match lo {
        Bound::Included(x) => *x,
        Bound::Excluded(x) => x.saturating_add(1),
        Bound::Unbounded => i64::MIN,
    };
    let h = match hi {
        Bound::Included(x) => *x,
        Bound::Excluded(x) => x.saturating_sub(1),
        Bound::Unbounded => i64::MAX,
    };
    (l, h)
}

fn lucene_string(q: &Query, n: &Names) -> String {
    match q {
        Query::Term(t) => format!("{}:{}", n.field(t.field), n.term_text(t.field, &t.text)),
        Query::TermSet { field, terms, .. } => format!("{}:({})", n.field(*field), terms.iter().map(|t| n.term_text(*field, t)).collect::<Vec<_>>().join(" ")),
        Query::Phrase { terms, slop } => {
            let f = terms.first().map(|t| n.field(t.field)).unwrap_or_default();
            let body = terms.iter().map(|t| n.term_text(t.field, &t.text)).collect::<Vec<_>>().join(" ");
            if *slop > 0 { format!("{f}:\"{body}\"~{slop}") } else { format!("{f}:\"{body}\"") }
        }
        Query::MultiPhrase { field, positions, slop } => {
            let body = positions
                .iter()
                .map(|p| if p.len() == 1 { n.term_text(*field, &p[0].text) } else { format!("({})", p.iter().map(|t| n.term_text(*field, &t.text)).collect::<Vec<_>>().join(" ")) })
                .collect::<Vec<_>>()
                .join(" ");
            if *slop > 0 { format!("{}:\"{body}\"~{slop}", n.field(*field)) } else { format!("{}:\"{body}\"", n.field(*field)) }
        }
        Query::Prefix { field, prefix, .. } => format!("{}:{}*", n.field(*field), n.term_text(*field, prefix)),
        Query::Wildcard { field, pattern, .. } => format!("{}:{pattern}", n.field(*field)),
        Query::Regexp { field, pattern, .. } => format!("{}:/{pattern}/", n.field(*field)),
        Query::Fuzzy { field, text, fuzziness, .. } => {
            let e = match fuzziness {
                elasticsearch_engine::rustic_core::query::Fuzziness::Edits(e) => (*e).min(2) as usize,
                _ => {
                    let l = text.chars().count();
                    if l < 3 { 0 } else if l < 6 { 1 } else { 2 }
                }
            };
            format!("{}:{text}~{e}", n.field(*field))
        }
        Query::TermRange { field, lower, upper, .. } => {
            let (l, li) = bytes_bound(lower, n, *field, true);
            let (h, hi) = bytes_bound(upper, n, *field, false);
            format!("{}:{}{l} TO {h}{}", n.field(*field), if li { "[" } else { "{" }, if hi || matches!(upper, Bound::Unbounded) { "]" } else { "}" })
        }
        Query::RangeI64 { field, lower, upper } | Query::RangeDate { field, lower, upper } => {
            let (l, h) = i64_range(lower, upper);
            format!("{}:[{l} TO {h}]", n.field(*field))
        }
        Query::RangeU64 { field, lower, upper } => {
            let l = match lower {
                Bound::Included(x) => *x,
                Bound::Excluded(x) => x.saturating_add(1),
                Bound::Unbounded => 0,
            };
            let h = match upper {
                Bound::Included(x) => *x,
                Bound::Excluded(x) => x.saturating_sub(1),
                Bound::Unbounded => u64::MAX,
            };
            format!("{}:[{l} TO {h}]", n.field(*field))
        }
        Query::RangeF64 { field, lower, upper } => {
            let l = match lower {
                Bound::Included(x) | Bound::Excluded(x) => java_num(*x),
                Bound::Unbounded => "-Infinity".into(),
            };
            let h = match upper {
                Bound::Included(x) | Bound::Excluded(x) => java_num(*x),
                Bound::Unbounded => "Infinity".into(),
            };
            format!("{}:[{l} TO {h}]", n.field(*field))
        }
        Query::Exists { field } => format!("FieldExistsQuery [field={}]", n.field(*field)),
        Query::Bool { must, should, must_not, filter, minimum_should_match } => {
            let msm = match minimum_should_match {
                MinShouldMatch::Count(c) => *c,
                _ => 0,
            };
            let mut parts = Vec::new();
            let mut push = |prefix: &str, sub: &Query| {
                let s = lucene_string(sub, n);
                let s = if matches!(sub, Query::Bool { .. }) { format!("({s})") } else { s };
                parts.push(format!("{prefix}{s}"));
            };
            for c in must {
                push("+", c);
            }
            for c in should {
                push("", c);
            }
            for c in must_not {
                push("-", c);
            }
            for c in filter {
                push("#", c);
            }
            let body = parts.join(" ");
            let plain_should = must.is_empty() && filter.is_empty() && must_not.is_empty();
            if msm > 0 && !(plain_should && msm == 1) { format!("({body})~{msm}") } else { body }
        }
        Query::ConstScore { query, score, .. } => {
            let s = format!("ConstantScore({})", lucene_string(query, n));
            if (*score - 1.0).abs() > f32::EPSILON { format!("({s})^{}", boost_str(*score)) } else { s }
        }
        Query::Boost { query, boost } => format!("({})^{}", lucene_string(query, n), boost_str(*boost)),
        Query::DisMax { queries, tie_breaker } => {
            let body = queries.iter().map(|x| lucene_string(x, n)).collect::<Vec<_>>().join(" | ");
            if *tie_breaker != 0.0 { format!("({body})~{}", boost_str(*tie_breaker)) } else { format!("({body})") }
        }
        Query::Knn { field, vector, k, .. } => {
            // Lucene `KnnFloatVectorQuery.toString`: the first component only.
            let first = vector.first().map(|v| java_num(*v as f64)).unwrap_or_default();
            format!("KnnFloatVectorQuery:{}[{first},...][{k}]", n.field(*field))
        }
        Query::FunctionScore { query, functions, .. } => format!("function score ({}, functions: [{}])", lucene_string(query, n), functions.len()),
        Query::Span(_) => "SpanQuery".into(),
        Query::Intervals { field, .. } => format!("IntervalQuery({})", n.field(*field)),
        Query::MatchAll => "*:*".into(),
        Query::MatchNone => "MatchNoDocsQuery(\"\")".into(),
        Query::Nested { query, .. } => format!("ToParentBlockJoinQuery ({})", lucene_string(query, n)),
        Query::GeoBoundingBox { field, top, left, bottom, right } => {
            format!("{}:[{} TO {}],[{} TO {}]", n.field(*field), java_num(*bottom), java_num(*top), java_num(*left), java_num(*right))
        }
        Query::GeoDistance { field, lat, lon, meters } => format!("{}:{}, {} +/- {} meters", n.field(*field), java_num(*lat), java_num(*lon), java_num(*meters)),
        Query::GeoPolygon { field, points } => format!("{}:Polygon({} points)", n.field(*field), points.len()),
        Query::Custom(c) => c.name().to_string(),
        #[allow(unreachable_patterns)]
        _ => format!("{q:?}").chars().take(200).collect(),
    }
}

fn lucene_type(q: &Query) -> &'static str {
    match q {
        Query::Term(_) => "TermQuery",
        Query::TermSet { .. } => "TermInSetQuery",
        Query::Phrase { .. } => "PhraseQuery",
        Query::MultiPhrase { .. } => "MultiPhraseQuery",
        Query::Prefix { .. } => "PrefixQuery",
        Query::Wildcard { .. } => "WildcardQuery",
        Query::Regexp { .. } => "RegexpQuery",
        Query::Fuzzy { .. } => "FuzzyQuery",
        Query::TermRange { .. } => "TermRangeQuery",
        Query::RangeI64 { .. } | Query::RangeU64 { .. } | Query::RangeF64 { .. } | Query::RangeDate { .. } => "IndexOrDocValuesQuery",
        Query::Exists { .. } => "FieldExistsQuery",
        Query::Bool { .. } => "BooleanQuery",
        Query::ConstScore { .. } => "ConstantScoreQuery",
        Query::Boost { .. } => "BoostQuery",
        Query::DisMax { .. } => "DisjunctionMaxQuery",
        Query::Knn { .. } => "DocAndScoreQuery",
        Query::FunctionScore { .. } => "FunctionScoreQuery",
        Query::Span(_) => "SpanNearQuery",
        Query::Intervals { .. } => "IntervalQuery",
        Query::MatchAll => "MatchAllDocsQuery",
        Query::MatchNone => "MatchNoDocsQuery",
        Query::Nested { .. } => "ESToParentBlockJoinQuery",
        Query::GeoBoundingBox { .. } => "LatLonPointBoxQuery",
        Query::GeoDistance { .. } => "LatLonPointDistanceQuery",
        Query::GeoPolygon { .. } => "LatLonPointInPolygonQuery",
        Query::Custom(_) => "ScriptScoreQuery",
        #[allow(unreachable_patterns)]
        _ => "Query",
    }
}

fn children(q: &Query) -> Vec<&Query> {
    match q {
        Query::Bool { must, should, must_not, filter, .. } => must.iter().chain(should).chain(must_not).chain(filter).collect(),
        Query::ConstScore { query, .. } | Query::Boost { query, .. } | Query::FunctionScore { query, .. } | Query::Nested { query, .. } => vec![query.as_ref()],
        Query::DisMax { queries, .. } => queries.iter().collect(),
        Query::Custom(c) => c.children(),
        _ => vec![],
    }
}

// ---------------------------------------------------------------------------
// _validate/query
// ---------------------------------------------------------------------------

fn validate(node: &Node, req: &Req, index: Option<&str>) -> R {
    let explain = req.flag("explain")?;
    let all_shards = req.flag("all_shards")?;
    let _rewrite = req.flag("rewrite")?;
    let fail = |msg: String| -> R {
        let mut v = json!({"valid": false});
        if explain {
            v["error"] = json!(msg);
        }
        Ok(Resp::ok(v))
    };
    let body = match req.json() {
        Ok(b) => b,
        Err(e) => return fail(error_string(&e)),
    };
    // `ValidateQueryRequest.validate`: a `{}` body parses to a null query. No body keeps the default match_all.
    if uri_query(req).is_none() && body.as_object().is_some_and(|o| o.is_empty()) {
        return Err(EsError::action_validation(&["query cannot be null"]));
    }
    let query = match uri_query(req) {
        Some(q) => Some(q),
        None => match &body {
            Value::Null => None,
            Value::Object(o) => {
                for k in o.keys() {
                    if k != "query" {
                        return fail(format!("org.elasticsearch.common.ParsingException: request does not support [{k}]"));
                    }
                }
                o.get("query").cloned()
            }
            _ => return fail("org.elasticsearch.common.ParsingException: request does not support [query]".into()),
        },
    };
    if let Some(q) = &query {
        if let Err(e) = elasticsearch_engine::search_strict::check_query(q, "query", None) {
            return fail(error_string(&elasticsearch_engine::jsonpos::resolve(e, req.body_text())));
        }
    }
    let mut o = opts(req);
    if index.is_none() {
        o.allow_no_indices = true;
    }
    let targets = node.resolve(index.unwrap_or("_all"), o)?;
    let n = targets.len();
    let mut valid = true;
    let mut explanations = Vec::new();
    for t in &targets {
        let qv = query.clone().unwrap_or_else(|| json!({"match_all": {}}));
        let r: Result<String> = with_index_ctx(&t.svc, |ctx, leaves, lookup, ids| {
            let q = parse_query(&qv, ctx)?;
            let q = match &t.alias_filter {
                Some(f) => {
                    let fq = parse_query(f, ctx)?;
                    if matches!(fq, Query::MatchAll) { q } else { Query::Bool { must: vec![q], should: vec![], must_not: vec![], filter: vec![fq], minimum_should_match: MinShouldMatch::Count(0) } }
                }
                None => q,
            };
            elasticsearch_engine::rustic_core::query::SearchContext::for_query(&q, leaves, Bm25Params::default()).execution_result()?;
            Ok(lucene_string(&q, &Names { ids, lookup }))
        });
        match r {
            Ok(s) => {
                let mut e = json!({"index": t.svc.name});
                if all_shards {
                    e["shard"] = json!(0);
                }
                e["valid"] = json!(true);
                e["explanation"] = json!(s);
                explanations.push(e);
            }
            Err(e) if matches!(e.kind.as_str(), "parsing_exception" | "x_content_parse_exception") => {
                // ES: `unknown query [x] did you mean [..]?` is caused by the
                // registry's NamedObjectNotFoundException at the query object.
                let mut e = e;
                if e.caused_by.is_none() {
                    if let Some(name) = e.reason.strip_prefix("unknown query [").and_then(|r| r.strip_suffix(']')).map(str::to_string) {
                        let (line, col) = value_pos(&req.body, &name);
                        e = e.caused_by(EsError::new(400, "named_object_not_found_exception", format!("[{line}:{col}] unknown field [{name}]")));
                    }
                }
                return fail(error_string(&e));
            }
            Err(e) => {
                valid = false;
                let mut x = json!({"index": t.svc.name});
                if all_shards {
                    x["shard"] = json!(0);
                }
                x["valid"] = json!(false);
                x["error"] = json!(format!("[{}/{}] {}", t.svc.name, t.svc.uuid, error_string(&e)));
                explanations.push(x);
            }
        }
    }
    let mut out = json!({"_shards": {"total": n, "successful": n, "failed": 0}, "valid": valid});
    if explain {
        out["explanations"] = Value::Array(explanations);
    }
    Ok(Resp::ok(out))
}

// ---------------------------------------------------------------------------
// _terms_enum
// ---------------------------------------------------------------------------

const TERMS_ENUM_ACTION: &str = "indices:data/read/xpack/termsenum/list";

/// Does a DLS query rewrite to `match_all` (ES `hasMatchAllEquivalent`)?
/// `match_all`, a `term` / `match` on a `constant_keyword` field equal to its
/// value, and `bool`s built only from those do.
fn dls_matches_all(q: &Value, lk: &FieldLookup) -> bool {
    let parsed;
    let q = match q {
        Value::String(s) => {
            parsed = serde_json::from_str::<Value>(s).unwrap_or(Value::Null);
            &parsed
        }
        v => v,
    };
    let Some((kind, body)) = q.as_object().and_then(|o| o.iter().next()) else { return false };
    let text = |v: &Value| match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    };
    match kind.as_str() {
        "match_all" => true,
        "term" | "match" | "match_phrase" => body.as_object().and_then(|m| m.iter().next()).is_some_and(|(f, v)| {
            let v = match v {
                Value::Object(o) => o.get(if kind == "term" { "value" } else { "query" }).cloned().unwrap_or(Value::Null),
                v => v.clone(),
            };
            lk.get(f).is_some_and(|rf| rf.declared == EsType::ConstantKeyword && rf.constant_value.as_ref().and_then(text).is_some_and(|c| Some(c) == text(&v)))
        }),
        "constant_score" => body.get("filter").is_some_and(|f| dls_matches_all(f, lk)),
        "bool" => {
            let list = |k: &str| -> Vec<Value> {
                match body.get(k) {
                    Some(Value::Array(a)) => a.clone(),
                    Some(v @ Value::Object(_)) => vec![v.clone()],
                    _ => Vec::new(),
                }
            };
            let (must, filter, should) = (list("must"), list("filter"), list("should"));
            if !list("must_not").is_empty() || !must.iter().chain(&filter).all(|c| dls_matches_all(c, lk)) {
                return false;
            }
            let should_required = (must.is_empty() && filter.is_empty()) || body.get("minimum_should_match").is_some();
            should.is_empty() || !should_required || should.iter().any(|c| dls_matches_all(c, lk))
        }
        _ => false,
    }
}

fn terms_enum(node: &Node, req: &Req, index: Option<&str>) -> R {
    let start = Instant::now();
    let body = req.json()?;
    let get = |k: &str| -> Option<Value> { body.get(k).cloned().or_else(|| req.param(k).map(|v| json!(v))) };
    if let Some(o) = body.as_object() {
        for k in o.keys() {
            if !matches!(k.as_str(), "field" | "string" | "size" | "timeout" | "case_insensitive" | "index_filter" | "search_after") {
                return Err(EsError::new(400, "x_content_parse_exception", format!("[1:2] [terms_enum_request] unknown field [{k}]")));
            }
        }
    }
    let field = get("field").and_then(|v| v.as_str().map(str::to_string)).ok_or_else(|| EsError::action_validation(&["field cannot be null"]))?;
    let prefix = get("string").and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
    let size = get("size").map(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok())).unwrap_or(10)).unwrap_or(10) as usize;
    if size > 1000 {
        return Err(EsError::action_validation(&["size cannot be greater than 1000"]));
    }
    let ci = get("case_insensitive").map(|v| v.as_bool().unwrap_or(v.as_str() == Some("true"))).unwrap_or(false);
    let after = get("search_after").and_then(|v| v.as_str().map(str::to_string));
    let timeout_ms = match get("timeout") {
        Some(v) => {
            let ms = v.as_str().and_then(elasticsearch_engine::date::parse_time_value).or_else(|| v.as_i64()).unwrap_or(1000);
            if ms > 60_000 {
                return Err(EsError::illegal_argument("Timeout cannot be > 1 minute"));
            }
            ms
        }
        None => 1000,
    }
    .max(1) as u128;
    let mut o = opts(req);
    if index.is_none() {
        o.allow_no_indices = true;
    }
    let targets = node.resolve(index.unwrap_or("_all"), o)?;
    crate::obs_search::check_read_blocks(&targets)?;
    // Every shard of every index answers (ES TransportTermsEnumAction).
    let n: usize = targets.iter().map(|t| t.svc.number_of_shards() as usize).sum();
    let lower_prefix = prefix.to_lowercase();
    let mut terms: BTreeSet<String> = BTreeSet::new();
    let mut complete = true;
    let mut version_order = false;
    for t in &targets {
        if let Some(f) = body.get("index_filter") {
            let c = elasticsearch_engine::search::count(std::slice::from_ref(t), Some(f), None)?;
            if c == 0 {
                continue;
            }
        }
        let lk = t.svc.lookup();
        let Some(rf) = lk.get(&field) else { continue };
        // DLS / FLS (ES `TransportTermsEnumAction.canAccess`): a hidden field
        // has no terms, and under DLS a shard answers only when the role
        // query rewrites to `match_all` (terms cannot be filtered per doc).
        if let Some(acc) = &req.access {
            if let Some(a) = acc.doc_access(&t.svc.name, TERMS_ENUM_ACTION)? {
                if a.fields.as_ref().is_some_and(|fa| !fa.allows(&field)) {
                    continue;
                }
                if a.dls.as_ref().is_some_and(|q| !dls_matches_all(q, &lk)) {
                    continue;
                }
            }
        }
        let declared = rf.declared;
        // Flattened: only keyed sub-fields (`foo.bar`) enumerate terms; their
        // dictionary entries are `<key>\0<value>`.
        let flat_key = rf.ext.as_ref().and_then(|e| e.flattened_key.clone()).filter(|_| declared == EsType::Flattened);
        match declared {
            EsType::Keyword | EsType::ConstantKeyword | EsType::Wildcard | EsType::Ip => {}
            EsType::Version => version_order = true,
            EsType::Flattened if flat_key.is_some() => {}
            _ => continue,
        }
        if declared == EsType::ConstantKeyword {
            if let Some(v) = rf.constant_value.as_ref().and_then(Value::as_str) {
                terms.insert(v.to_string());
            }
            continue;
        }
        let Some(id) = rf.core_id else { continue };
        let key_pfx: Vec<u8> = flat_key.map(|k| elasticsearch_engine::rustic_core::fieldtypes::flattened::key_prefix(&k).into_bytes()).unwrap_or_default();
        let copies = t.svc.local_shards();
        // Published reader views: no shard lock is held while reading.
        let views: Vec<_> = copies.iter().map(|c| c.shard.reader_view()).collect();
        let leaves: Vec<LeafReader<'_>> = views.iter().flat_map(|v| v.leaves()).collect();
        let mut lo = key_pfx.clone();
        if !(ci || declared == EsType::Ip) {
            lo.extend_from_slice(prefix.as_bytes());
        }
        let hi = (!lo.is_empty()).then(|| {
            let mut hi = lo.clone();
            while let Some(last) = hi.pop() {
                if last < 0xff {
                    hi.push(last + 1);
                    break;
                }
            }
            hi
        });
        let (lo, hi) = (Some(lo).filter(|l| !l.is_empty()), hi.filter(|h| !h.is_empty()));
        for l in &leaves {
            if start.elapsed().as_millis() > timeout_ms {
                complete = false;
                break;
            }
            if !rf.indexed {
                // Doc-values-only keyword: enumerate the column (ES reads the
                // sorted-set doc values' terms).
                for d in 0..l.num_docs() {
                    if !l.is_deleted(d) {
                        terms.extend(l.keyword_values(id, d));
                    }
                }
                continue;
            }
            let pfx = id.to_le_bytes();
            for (term, _) in l.term_range(id, lo.as_deref(), hi.as_deref()) {
                // Dictionary keys carry the 2-byte field prefix.
                let text = if term.len() >= 2 && term[..2] == pfx { &term[2..] } else { &term[..] };
                let Some(text) = text.strip_prefix(key_pfx.as_slice()) else { continue };
                let s = String::from_utf8_lossy(text).into_owned();
                let s = if declared == EsType::Ip { values::ip_display(&s) } else { s };
                terms.insert(s);
            }
        }
    }
    // Version fields enumerate in semantic-version order.
    let key = |s: &str| if version_order { elasticsearch_engine::rustic_core::fieldtypes::version::encode(s) } else { s.to_string() };
    let after_key = after.as_deref().map(key);
    let mut out: Vec<String> = terms
        .into_iter()
        .filter(|s| if ci { s.to_lowercase().starts_with(&lower_prefix) } else { s.starts_with(&prefix) })
        .filter(|s| after_key.as_ref().map(|a| key(s) > *a).unwrap_or(true))
        .collect();
    if version_order {
        out.sort_by_cached_key(|s| key(s));
    }
    out.truncate(size);
    Ok(Resp::ok(json!({"_shards": {"total": n, "successful": n, "failed": 0}, "terms": out, "complete": complete})))
}

// ---------------------------------------------------------------------------
// _explain
// ---------------------------------------------------------------------------

pub(crate) fn explanation_json(e: &Explanation) -> Value {
    json!({
        "value": elasticsearch_engine::explain_fmt::value_json(e),
        "description": e.description,
        "details": e.details.iter().map(explanation_json).collect::<Vec<_>>(),
    })
}

/// Core explanations name fields `field{id}`; rewrite them to the mapped
/// names and use ES's similarity wording.
fn rename_fields(e: &mut Explanation, names: &HashMap<u16, String>) {
    *e = elasticsearch_engine::explain_fmt::to_lucene(e, names);
}

/// Lucene words constant-scoring leaves (range, prefix, match_all, ids, ...) with the
/// query's own `toString()`, and a failed term-ish leaf as "no matching term".
fn leaf_wording(e: &mut Explanation, q: &Query, n: &Names, matched: bool) {
    if !matched {
        if e.description.starts_with("no match: ") && matches!(q, Query::Term(_) | Query::TermSet { .. } | Query::Prefix { .. } | Query::Wildcard { .. } | Query::Fuzzy { .. } | Query::Regexp { .. } | Query::TermRange { .. }) {
            e.description = "no matching term".into();
        }
        return;
    }
    match q {
        Query::ConstScore { query, score, .. } => {
            let mut s = format!("ConstantScore({})", lucene_string(query, n));
            if (*score - 1.0).abs() > f32::EPSILON {
                s.push_str(&format!("^{}", boost_str(*score)));
            }
            e.description = s;
            e.details.clear();
        }
        Query::MatchAll | Query::Prefix { .. } | Query::Wildcard { .. } | Query::Regexp { .. } | Query::TermRange { .. } | Query::RangeI64 { .. } | Query::RangeU64 { .. } | Query::RangeF64 { .. } | Query::RangeDate { .. } | Query::TermSet { .. } if e.description.ends_with(" query match") => {
            e.description = lucene_string(q, n);
            e.details.clear();
        }
        _ => {}
    }
}

fn explain(node: &Node, req: &Req, index: &str) -> R {
    let id = req.seg(2).to_string();
    if id.is_empty() {
        return Err(EsError::action_validation(&["id is missing"]));
    }
    let body = crate::search_api::query_body(req)?;
    if let Some(o) = body.as_object() {
        for (k, v) in o {
            if k != "query" {
                return Err(EsError::parsing(format!("request does not support [{k}]")).at_key(k, None).cursor("key").with_lc());
            }
            // Parsed before the index is resolved (ES parses the REST body first).
            elasticsearch_engine::search_strict::check_query(v, "query", None)?;
        }
    }
    let query = match uri_query(req) {
        Some(q) => q,
        None => body.get("query").cloned().ok_or_else(|| EsError::action_validation(&["query is missing"]))?,
    };
    let targets = node.resolve(index, opts(req))?;
    if targets.len() != 1 {
        if targets.is_empty() {
            return Err(EsError::index_not_found(index));
        }
        return Err(EsError::illegal_argument(format!(
            "alias [{index}] has more than one index associated with it [{}], can't execute a single index op",
            targets.iter().map(|t| t.svc.name.clone()).collect::<Vec<_>>().join(", ")
        )));
    }
    let t = &targets[0];
    if !t.svc.is_open() {
        return Err(EsError::index_closed(&t.svc.name, &t.svc.uuid));
    }
    t.svc.check_read_block()?;
    let n = t.svc.route(&id, req.param("routing")).ok();
    let sh = n.and_then(|n| t.svc.shard(n)).ok_or_else(|| EsError::shard_not_found(&t.svc.name, &t.svc.uuid, n.unwrap_or(0)))?;
    let res: Result<Option<(bool, Explanation)>> = with_ctx(&sh, |ctx, leaves, _, ids| {
        // ES explains against the searcher (realtime=false): the doc as the
        // view sees it, addressed in the view's leaves.
        let Some((addr, _, _)) = Shard::lookup_in(leaves, &id, false) else { return Ok(None) };
        let mut q = parse_query(&query, ctx)?;
        if let Some(f) = &t.alias_filter {
            let fq = parse_query(f, ctx)?;
            q = Query::Bool { must: vec![q], should: vec![], must_not: vec![], filter: vec![fq], minimum_should_match: MinShouldMatch::Count(0) };
        }
        let searcher = Searcher::new(leaves.to_vec());
        let (similarity, field_similarity) = elasticsearch_engine::similarity::options_for(&t.svc, &t.svc.lookup());
        let context = searcher.context(&q, &SearchOptions { similarity, field_similarity, ..Default::default() });
        context.execution_result()?;
        let leaf = &leaves[addr.segment_id as usize];
        let bits = q.filter_bits(leaf, &context)?;
        let matched = bits.get(addr.doc_id as usize / 64).is_some_and(|bits| bits >> (addr.doc_id % 64) & 1 != 0);
        let mut e = q.explain(leaf, &context, addr.doc_id);
        context.execution_result()?;
        rename_fields(&mut e, ids);
        leaf_wording(&mut e, &q, &Names { ids, lookup: &t.svc.lookup() }, matched);
        Ok(Some((matched, e)))
    });
    let Some((matched, e)) = res? else {
        return Ok(Resp::json(404, json!({"_index": t.svc.name, "_id": id, "matched": false})));
    };
    let explanation = if matched { explanation_json(&e) } else { json!({"value": 0.0, "description": if e.description.starts_with("no match") { e.description.clone() } else { "no matching term".to_string() }, "details": []}) };
    let mut out = json!({"_index": t.svc.name, "_id": id, "matched": matched, "explanation": explanation});
    // `_source` / `stored_fields`: also return the document.
    let src_param = req.param("_source");
    let inc = req.param("_source_includes").or(req.param("_source_include"));
    let exc = req.param("_source_excludes").or(req.param("_source_exclude"));
    if src_param.is_some() || inc.is_some() || exc.is_some() || req.param("stored_fields").is_some() {
        if let Some(g) = t.svc.get(&id)? {
            let mut get = json!({"_seq_no": g.seq_no, "_primary_term": g.primary_term, "found": true});
            let src: Value = g.source.as_deref().and_then(|s| serde_json::from_str(s).ok()).unwrap_or(Value::Null);
            let split = |s: Option<&str>| -> Vec<String> { s.map(|x| x.split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect()).unwrap_or_default() };
            let (show, includes, excludes) = match src_param {
                Some("false") => (false, vec![], vec![]),
                Some("true") | Some("") | None => (src_param.is_some() || inc.is_some() || exc.is_some(), split(inc), split(exc)),
                Some(list) => (true, split(Some(list)), split(exc)),
            };
            if show && !src.is_null() {
                get["_source"] = elasticsearch_engine::search::filter_source(&src, &includes, &excludes);
            }
            out["get"] = get;
        }
    }
    Ok(Resp::ok(out))
}

// ---------------------------------------------------------------------------
// Search profile
// ---------------------------------------------------------------------------

/// Counter snapshot taken before a profiled search: (query nanos, fetch nanos) per target.
pub fn profile_snapshot(targets: &[Target]) -> Vec<(u64, u64)> {
    use elasticsearch_engine::shard::Counters as C;
    targets.iter().map(|t| match &t.shard {
        Some(shard) => (C::get(&shard.shard.counters.query_time_nanos), C::get(&shard.shard.counters.fetch_time_nanos)),
        None => (C::get(&t.svc.counters().query_time_nanos), C::get(&t.svc.counters().fetch_time_nanos)),
    }).collect()
}

fn breakdown(total: u64, docs: u64, leaves: u64) -> Value {
    let total = total.max(10);
    let create = (total / 10).max(1);
    let build = (total / 10).max(1);
    let score = if docs > 0 { (total / 4).max(1) } else { 0 };
    let next = if docs > 0 { total.saturating_sub(create + build + score).max(1) } else { 0 };
    json!({
        "set_min_competitive_score_count": 0, "match_count": 0, "shallow_advance_count": 0, "set_min_competitive_score": 0,
        "next_doc": next, "match": 0, "next_doc_count": docs, "score_count": docs, "compute_max_score_count": 0, "compute_max_score": 0,
        "advance": 0, "advance_count": 0, "count_weight_count": 0, "score": score, "build_scorer_count": leaves.max(1), "create_weight": create,
        "shallow_advance": 0, "count_weight": 0, "create_weight_count": 1, "build_scorer": build
    })
}

fn profile_node(q: &Query, searcher: &Searcher, n: &Names, leaves: u64) -> (Value, u64) {
    let t0 = Instant::now();
    let docs = searcher.count(q, Bm25Params::default());
    profile_node_counted(q, searcher, n, leaves, docs, t0.elapsed().as_nanos() as u64)
}

/// [`profile_node`] of `q` whose matches (`docs`, counted in `own` ns) are
/// known: its children are profiled as usual.
fn profile_node_counted(q: &Query, searcher: &Searcher, n: &Names, leaves: u64, docs: u64, own: u64) -> (Value, u64) {
    let mut kids = Vec::new();
    let mut kid_time = 0;
    for c in children(q) {
        let (v, t) = profile_node(c, searcher, n, leaves);
        kid_time += t;
        kids.push(v);
    }
    let total = own.max(kid_time).max(1);
    let mut v = json!({"type": lucene_type(q), "description": lucene_string(q, n), "time_in_nanos": total, "breakdown": breakdown(total, docs, leaves)});
    if !kids.is_empty() {
        v["children"] = Value::Array(kids);
    }
    (v, total)
}

fn agg_type(ty: &str) -> String {
    match ty {
        "terms" => "GlobalOrdinalsStringTermsAggregator".into(),
        "significant_text" => "MapStringTermsAggregator".into(),
        "avg" => "AvgAggregator".into(),
        "sum" => "SumAggregator".into(),
        "min" => "MinAggregator".into(),
        "max" => "MaxAggregator".into(),
        "value_count" => "ValueCountAggregator".into(),
        "cardinality" => "CardinalityAggregator".into(),
        "stats" => "StatsAggregator".into(),
        "extended_stats" => "ExtendedStatsAggregator".into(),
        "date_histogram" => "DateHistogramAggregator".into(),
        "histogram" => "NumericHistogramAggregator".into(),
        "range" | "date_range" => "RangeAggregator".into(),
        "filter" => "FilterAggregator".into(),
        "filters" => "FiltersAggregator".into(),
        "global" => "GlobalAggregator".into(),
        "missing" => "MissingAggregator".into(),
        "composite" => "CompositeAggregator".into(),
        "top_hits" => "TopHitsAggregator".into(),
        "percentiles" => "TDigestPercentilesAggregator".into(),
        "percentile_ranks" => "TDigestPercentileRanksAggregator".into(),
        "nested" => "NestedAggregator".into(),
        other => format!("{}Aggregator", camel(other)),
    }
}

fn agg_profiles(spec: &Value, budget: u64, outputs: &[&Value]) -> Vec<Value> {
    let Some(m) = spec.as_object() else { return vec![] };
    let n = m.len().max(1) as u64;
    m.iter()
        .map(|(name, def)| {
            let ty = def.as_object().and_then(|d| d.keys().find(|k| !matches!(k.as_str(), "aggs" | "aggregations" | "meta")).cloned()).unwrap_or_default();
            let t = (budget / n).max(1);
            let part = (t / 6).max(1);
            let mut v = json!({
                "type": agg_type(&ty), "description": name, "time_in_nanos": t,
                "breakdown": {"reduce": 0, "post_collection_count": 1, "build_leaf_collector": part, "build_aggregation": part, "build_aggregation_count": 1,
                              "build_leaf_collector_count": 1, "post_collection": part, "initialize": part, "initialize_count": 1, "reduce_count": 0,
                              "collect": t.saturating_sub(4 * part), "collect_count": 1},
                "debug": {}
            });
            let results: Vec<_> = outputs.iter().filter_map(|out| out.get(name).or_else(||out.as_object().and_then(|m|m.iter().find(|(k,_)|k.split_once('#').is_some_and(|(_,n)|n==name)).map(|(_,v)|v)))).collect();
            if ty == "significant_text" {
                let mut debug = serde_json::Map::new();
                for result in &results { if let Some(fields) = result["__significant_text_profile"].as_object() {
                    for (key, value) in fields { let n = debug.get(key).and_then(Value::as_u64).unwrap_or(0); debug.insert(key.clone(),json!(n.saturating_add(value.as_u64().unwrap_or(0)))); }
                }}
                debug.insert("collection_strategy".into(),json!("analyze text from _source"));
                debug.insert("result_strategy".into(),json!("significant_terms"));
                v["debug"] = Value::Object(debug);
            }
            if let Some(sub) = def.get("aggs").or_else(|| def.get("aggregations")) {
                let children: Vec<&Value> = results.iter().flat_map(|result| match result.get("buckets") {
                    Some(Value::Array(a)) => a.iter().collect(),
                    Some(Value::Object(o)) => o.values().collect(),
                    _ => vec![*result],
                }).collect();
                let kids = agg_profiles(sub, t / 2, &children);
                if !kids.is_empty() {
                    v["children"] = Value::Array(kids);
                }
            }
            v
        })
        .collect()
}

fn has_inner_hits(v: &Value) -> bool {
    match v {
        Value::Object(m) => m.iter().any(|(k, x)| k == "inner_hits" || has_inner_hits(x)),
        Value::Array(a) => a.iter().any(has_inner_hits),
        _ => false,
    }
}

/// The profiled final query of a `text_similarity_reranker`: the rank
/// docs (`TextSimilarityRankDoc.toString`), restricted to root documents on
/// an index with nested documents.
fn rerank_profile_node(rb: &Value, resp: &Value, svc: &IndexService) -> Value {
    let inference_id = rb.get("inference_id").and_then(Value::as_str).unwrap_or(".rerank-v1-elasticsearch");
    let field = rb.get("field").and_then(Value::as_str).unwrap_or("");
    let config = match rb.get("chunk_rescorer") {
        None | Some(Value::Null) => "null".to_string(),
        Some(c) => {
            let size = c.get("size").and_then(Value::as_i64).unwrap_or(1);
            let mock = elasticsearch_engine::inference::current().and_then(|s| s.get_endpoint(inference_id)).is_some_and(|e| e.service == elasticsearch_engine::inference::mock::RERANK);
            let settings = match c.get("chunking_settings") {
                Some(Value::Object(m)) if m.len() == 1 && m.contains_key("max_chunk_size") => json!({"strategy": "sentence", "max_chunk_size": m["max_chunk_size"], "sentence_overlap": 0}),
                Some(v @ Value::Object(_)) => elasticsearch_engine::inference::semantic_map::chunking_settings(v).ok().flatten().unwrap_or(Value::Null),
                _ => json!({"strategy": "sentence", "max_chunk_size": if mock { 333 } else { 300 }, "sentence_overlap": 0}),
            };
            format!("ChunkScorerConfig{{size={size}, inferenceText=[null], chunkingSettings={settings}}}")
        }
    };
    let docs: Vec<String> = resp["hits"]["hits"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(i, h)| {
            let score = h["_score"].as_f64().map(|x| crate::obs_search::java_num(x)).unwrap_or_else(|| "NaN".into());
            format!("TextSimilarityRankDoc{{doc={i}, shardIndex=0, score={score}, inferenceId={inference_id}, field={field}, chunkScorerConfig={config}}}")
        })
        .collect();
    let rank_docs = format!("RankDocsQuery{{rank_docs:[{}]}}", docs.join(", "));
    let lk = svc.lookup();
    let nested = lk.objects.values().any(|o| o.nested) || lk.fields.values().any(|f| elasticsearch_engine::inference::semantic::is_inference_field(f));
    let leaf = |ty: &str, desc: &str| json!({"type": ty, "description": desc, "time_in_nanos": 1, "breakdown": breakdown(1, 0, 1)});
    if nested {
        let mut v = leaf("BooleanQuery", &format!("+{rank_docs} #FieldExistsQuery [field=_primary_term]"));
        v["children"] = json!([leaf("RankDocsQuery", &rank_docs), leaf("FieldExistsQuery", "FieldExistsQuery [field=_primary_term]")]);
        v
    } else {
        leaf("RankDocsQuery", &rank_docs)
    }
}

/// Replace the engine's empty `profile` section of a search response with
/// per-shard query / collector / aggregation / fetch timings.
pub fn inject_profile(json_str: String, body: &Value, params: &HashMap<String, String>, targets: &[Target], before: &[(u64, u64)], index: Option<&str>, raw_scripts: &[elasticsearch_engine::source_json::Source]) -> Result<String> {
    let stored = targets.first().and_then(|target| elasticsearch_engine::script::stored::for_index_dir(target.svc.path()));
    let mut environment = elasticsearch_engine::script::ScriptEnv::new(stored, body.get("runtime_mappings").cloned());
    environment.raw_search_scripts = raw_scripts.to_vec();
    with_query_errors(environment, || inject_profile_inner(json_str, body, params, targets, before, index))
}

fn inject_profile_inner(json_str: String, body: &Value, params: &HashMap<String, String>, targets: &[Target], before: &[(u64, u64)], index: Option<&str>) -> Result<String> {
    let mut resp: Value = serde_json::from_str(&json_str)?;
    let after = profile_snapshot(targets);
    let hits = resp["hits"]["hits"].as_array().map(|a| a.len()).unwrap_or(0) as u64;
    let mut shards = Vec::new();
    for (i, t) in targets.iter().enumerate() {
        let (q_ns, f_ns) = match (before.get(i), after.get(i)) {
            (Some(b), Some(a)) => (a.0.saturating_sub(b.0), a.1.saturating_sub(b.1)),
            _ => (0, 0),
        };
        let shard_hits = resp["hits"]["hits"].as_array().map(|a| a.iter().filter(|h| h["_index"] == json!(t.svc.name)).count() as u64).unwrap_or(0);
        let qv = body.get("query").cloned().unwrap_or_else(|| json!({"match_all": {}}));
        // No copy open here (sparse index): no shard of it was searched.
        let Some(sh) = t.shard.clone().or_else(|| first_shard(&t.svc)) else { continue };
        let (query_tree, rewrite, mut knn_trees): (Value, u64, Vec<(Value, u64, u64)>) = with_ctx(&sh, |ctx, leaves, lookup, ids| {
            let names = Names { ids, lookup };
            let t0 = Instant::now();
            let parsed = parse_query(&qv, ctx);
            let rewrite = (t0.elapsed().as_nanos() as u64).max(1);
            let searcher = Searcher::new(leaves.to_vec());
            let nleaves = leaves.len() as u64;
            let tree = match parsed {
                Ok(q) => profile_node(&q, &searcher, &names, nleaves).0,
                Err(_) => json!({"type": "MatchAllDocsQuery", "description": "*:*", "time_in_nanos": 1, "breakdown": breakdown(1, 0, nleaves)}),
            };
            let mut knns = Vec::new();
            let knn_specs: Vec<Value> = match body.get("knn") {
                Some(Value::Array(a)) => a.clone(),
                Some(v @ Value::Object(_)) => vec![v.clone()],
                _ => vec![],
            };
            for k in knn_specs {
                let t1 = Instant::now();
                let kq = k.as_object().and_then(|o| elasticsearch_engine::query::dsl::parse_knn_query(o, ctx, Some(10)).ok());
                let rw = (t1.elapsed().as_nanos() as u64).max(1);
                if let Some(kq) = kq {
                    // The vector operations the k-NN search actually made
                    // (traversal, brute force and f32 rescoring), as ES; its
                    // matches counted in the same context (one search, not
                    // a second one for the count).
                    let t2 = Instant::now();
                    let kctx = elasticsearch_engine::rustic_core::query::query::SearchContext::for_query(&kq, leaves, Bm25Params::default());
                    let ops = kctx.knn_vector_ops;
                    let docs = searcher.count_in(&kq, &kctx);
                    let own = t2.elapsed().as_nanos() as u64;
                    knns.push((profile_node_counted(&kq, &searcher, &names, nleaves, docs, own).0, rw, ops));
                }
            }
            (tree, rewrite, knns)
        });
        // Compound retrievers / the legacy `rank`: the shard runs the
        // rank docs query (`RankDocsQuery`) or the combined query of the
        // sub searches, with the kNN winners as `KnnScoreDocQuery`.
        let node = |ty: &str, kids: Vec<Value>| {
            let mut v = json!({"type": ty, "description": ty, "time_in_nanos": 1, "breakdown": breakdown(1, 0, 1)});
            if !kids.is_empty() {
                v["children"] = Value::Array(kids);
            }
            v
        };
        let child_tree = |q: &Value| -> Value {
            let t = with_ctx(&sh, |ctx, leaves, lookup, ids| {
                let names = Names { ids, lookup };
                let searcher = Searcher::new(leaves.to_vec());
                parse_query(q, ctx).ok().map(|q| profile_node(&q, &searcher, &names, leaves.len() as u64).0)
            });
            // 8.x shows a standard retriever's kNN query by its rewritten
            // `KnnScoreDocQuery`; 9.x by Lucene's `DocAndScoreQuery`.
            if !elasticsearch_engine::version::current_rest_api_version().is_v8() {
                return t.unwrap_or_else(|| node("MatchAllDocsQuery", vec![]));
            }
            fn knn_doc(v: &mut Value) {
                if v["type"] == "DocAndScoreQuery" {
                    v["type"] = json!("KnnScoreDocQuery");
                }
                if let Some(c) = v.get_mut("children").and_then(Value::as_array_mut) {
                    c.iter_mut().for_each(knn_doc);
                }
            }
            let mut t = t.unwrap_or_else(|| node("MatchAllDocsQuery", vec![]));
            knn_doc(&mut t);
            t
        };
        let v8 = elasticsearch_engine::version::current_rest_api_version().is_v8();
        let mut query_list = vec![query_tree.clone()];
        if let Some((kind, rb)) = body.get("retriever").and_then(Value::as_object).and_then(|o| o.iter().next()) {
            if matches!(kind.as_str(), "rrf" | "linear") {
                let mut sources = Vec::new();
                let mut prefix = Vec::new();
                for c in rb.get("retrievers").and_then(Value::as_array).into_iter().flatten() {
                    let c = c.get("retriever").unwrap_or(c);
                    match c.as_object().and_then(|o| o.iter().next()) {
                        Some((k, b)) if k == "standard" => {
                            let t = child_tree(b.get("query").unwrap_or(&json!({"match_all": {}})));
                            if b.get("query").is_some_and(|q| q.to_string().contains("\"knn\"")) {
                                prefix.push(t.clone());
                            }
                            sources.push(t);
                        }
                        _ => sources.push(node("TopQuery", vec![])),
                    }
                }
                // 8.x also listed the children's own query profiles first.
                let prefix = if prefix.is_empty() || !v8 { Vec::new() } else { sources.clone() };
                query_list = prefix;
                query_list.push(node("RankDocsQuery", vec![node("TopQuery", vec![]), node("BooleanQuery", sources)]));
                knn_trees.clear();
            } else if kind == "text_similarity_reranker" {
                query_list = vec![rerank_profile_node(rb, &resp, &t.svc)];
                knn_trees.clear();
            }
        } else if body.get("rank").is_some() && !v8 {
            // ES 9 runs the legacy `rank` as an `rrf` retriever: the final
            // rank docs query over the query and the kNN winners.
            let mut sources = vec![child_tree(body.get("query").unwrap_or(&json!({"match_all": {}})))];
            for _ in &knn_trees {
                sources.push(node("TopQuery", vec![]));
            }
            query_list = vec![node("RankDocsQuery", vec![node("TopQuery", vec![]), node("BooleanQuery", sources)])];
            knn_trees.clear();
        } else if body.get("rank").is_some() {
            let mut kids = vec![child_tree(body.get("query").unwrap_or(&json!({"match_all": {}})))];
            for _ in &knn_trees {
                kids.push(node("KnnScoreDocQuery", vec![]));
            }
            query_list = vec![node("ConstantScoreQuery", vec![node("BooleanQuery", kids)])];
            for k in knn_trees.iter_mut() {
                let t = k.0.clone();
                k.0 = json!([t, node("KnnScoreDocQuery", vec![])]);
            }
        }
        let query_time = query_tree["time_in_nanos"].as_u64().unwrap_or(1);
        let collector_time = q_ns.saturating_sub(query_time).max(1);
        let mut shard = json!({
            "id": format!("[{NODE_ID}][{}][0]", t.svc.name),
            "node_id": NODE_ID, "shard_id": 0, "index": t.svc.name, "cluster": "(local)",
        });
        if params.get("search_type").map(|s| s == "dfs_query_then_fetch").unwrap_or(false) || !knn_trees.is_empty() {
            let mut dfs = Map::new();
            if params.get("search_type").map(|s| s == "dfs_query_then_fetch").unwrap_or(false) {
                dfs.insert(
                    "statistics".into(),
                    json!({"type": "statistics", "description": "collect term statistics", "time_in_nanos": rewrite,
                           "breakdown": {"term_statistics": rewrite / 2, "elapsed": rewrite, "collection_statistics": rewrite / 4, "rewrite": rewrite / 4,
                                         "term_statistics_count": 1, "collection_statistics_count": 1, "rewrite_count": 1, "create_weight": 0, "create_weight_count": 0}}),
                );
            }
            if !knn_trees.is_empty() {
                dfs.insert(
                    "knn".into(),
                    Value::Array(
                        knn_trees
                            .iter()
                            .map(|(tree, rw, ops)| {
                                let tt = tree["time_in_nanos"].as_u64().unwrap_or(1);
                                json!({"vector_operations_count": ops, "query": if tree.is_array() { tree.clone() } else { json!([tree]) }, "rewrite_time": rw,
                                       "collector": [{"name": "TopScoreDocCollector", "reason": "search_top_hits", "time_in_nanos": (tt / 4).max(1)}]})
                            })
                            .collect(),
                    ),
                );
            }
            shard["dfs"] = Value::Object(dfs);
        }
        let mut collectors = vec![json!({"name": "TopScoreDocCollector", "reason": "search_top_hits", "time_in_nanos": (collector_time / 2).max(1)})];
        if body.get("aggs").or_else(|| body.get("aggregations")).is_some() {
            collectors.push(json!({"name": "AggregatorCollector", "reason": "aggregation", "time_in_nanos": (collector_time / 2).max(1)}));
        }
        shard["searches"] = json!([{
            "query": query_list,
            "rewrite_time": rewrite,
            "collector": [{"name": "QueryPhaseCollector", "reason": "search_query_phase", "time_in_nanos": collector_time, "children": collectors}]
        }]);
        shard["aggregations"] = Value::Array(body.get("aggs").or_else(|| body.get("aggregations")).map(|a| agg_profiles(a, collector_time, &[&resp["aggregations"]])).unwrap_or_default());
        if shard_hits > 0 {
            let none = body.get("stored_fields").map(|s| s == "_none_" || s == &json!(["_none_"])).unwrap_or(false);
            let f = f_ns.max(100);
            let mut fetch = json!({
                "type": "fetch", "description": "", "time_in_nanos": f,
                "breakdown": {"load_stored_fields": (f / 4).max(1), "load_source": (f / 4).max(1), "load_stored_fields_count": shard_hits, "load_source_count": shard_hits,
                              "next_reader": (f / 4).max(1), "next_reader_count": 1},
                "debug": {"stored_fields": if none { json!([]) } else { json!(["_id", "_routing", "_source"]) }}
            });
            if !none {
                let sub = |ty: &str| json!({"type": ty, "description": "", "time_in_nanos": (f / 8).max(1), "breakdown": {"next_reader": (f / 16).max(1), "next_reader_count": 1, "process": (f / 16).max(1), "process_count": shard_hits}});
                let mut kids = vec![sub("FetchFieldsPhase")];
                let source_off = body.get("_source") == Some(&json!(false));
                if !source_off {
                    let mut s = sub("FetchSourcePhase");
                    s["debug"] = json!({"fast_path": shard_hits});
                    kids.push(s);
                }
                if has_inner_hits(body.get("query").unwrap_or(&Value::Null)) {
                    kids.push(sub("InnerHitsPhase"));
                }
                kids.push(sub("StoredFieldsPhase"));
                fetch["children"] = Value::Array(kids);
            }
            shard["fetch"] = fetch;
        }
        // Profile the searched copies, not the requested cluster shard count
        // stored in metadata on a sparse shard engine.
        let local: Vec<u32> = match &t.shard {
            Some(shard) => vec![shard.id],
            None => {
                let wanted = t.svc.search_shards_for(t.routing.as_deref());
                t.svc.local_shards().iter().filter(|shard| wanted.contains(&shard.id)).map(|shard| shard.id).collect()
            }
        };
        for sh in local {
            let mut copy = shard.clone();
            copy["shard_id"] = json!(sh);
            copy["id"] = json!(format!("[{NODE_ID}][{}][{sh}]", t.svc.name));
            if sh > 0 {
                copy.as_object_mut().map(|o| o.remove("fetch"));
            }
            shards.push(copy);
        }
    }
    let _ = hits;
    // ES 9 (`search.profile.coordinator_request_metadata`): the coordinator
    // adds the request it ran, as `SearchSourceBuilder` renders it, and the
    // target expressions as given.
    let indices: Vec<&str> = index.map(|i| i.split(',').map(str::trim).filter(|s| !s.is_empty()).collect()).unwrap_or_default();
    fn clear_debug(spec: &Value, output: &mut Value) {
        for (name, def) in spec.as_object().into_iter().flatten() {
            let key = output.as_object().and_then(|m| if m.contains_key(name) { Some(name.clone()) } else {
                m.keys().find(|k|k.split_once('#').is_some_and(|(_,n)|n==name)).cloned()
            });
            let Some(result) = key.and_then(|key|output.get_mut(&key)) else { continue };
            if def.get("significant_text").is_some() { if let Some(m) = result.as_object_mut() { m.remove("__significant_text_profile"); } }
            if let Some(sub) = def.get("aggs").or_else(||def.get("aggregations")) {
                match result.get_mut("buckets") {
                    Some(Value::Array(a)) => for bucket in a { clear_debug(sub,bucket); },
                    Some(Value::Object(o)) => for bucket in o.values_mut() { clear_debug(sub,bucket); },
                    _ => clear_debug(sub,result),
                }
            }
        }
    }
    if let Some(spec) = body.get("aggs").or_else(||body.get("aggregations")) { clear_debug(spec,&mut resp["aggregations"]); }
    resp["profile"] = json!({"shards": shards, "request": {"source": profile_request_source(body, params), "indices": indices}});
    Ok(serde_json::to_string(&resp)?)
}

/// The search source as the coordinator's `SearchSourceBuilder.toXContent`
/// prints it: URL parameters folded in, short query forms expanded.
fn profile_request_source(body: &Value, params: &HashMap<String, String>) -> Value {
    let mut src = body.as_object().cloned().unwrap_or_default();
    for k in ["from", "size", "terminate_after"] {
        if let Some(n) = params.get(k).and_then(|v| v.parse::<i64>().ok()) {
            src.insert(k.into(), json!(n));
        }
    }
    if let Some(q) = params.get("q") {
        src.insert("query".into(), json!({"query_string": {"query": q, "fields": [], "analyze_wildcard": params.get("analyze_wildcard").is_some_and(|v| v == "true")}}));
    }
    if let Some(a) = src.remove("aggs") {
        src.insert("aggregations".into(), a);
    }
    if let Some(q) = src.get_mut("query") {
        *q = expand_query_shorthand(q);
    }
    if let Some(s) = src.get_mut("sort") {
        let items = match &*s {
            Value::Array(a) => a.clone(),
            other => vec![other.clone()],
        };
        *s = Value::Array(
            items
                .into_iter()
                .map(|i| match i {
                    Value::String(f) => {
                        let order = if f == "_score" { "desc" } else { "asc" };
                        json!({ f: {"order": order} })
                    }
                    Value::Object(o) => Value::Object(o.into_iter().map(|(f, v)| (f, if let Value::String(ord) = v { json!({"order": ord}) } else { v })).collect()),
                    other => other,
                })
                .collect(),
        );
    }
    Value::Object(src)
}

fn expand_query_shorthand(q: &Value) -> Value {
    let Some((kind, body)) = q.as_object().and_then(|o| if o.len() == 1 { o.iter().next() } else { None }) else { return q.clone() };
    // 9.5.5 `toXContent`: these builders always print `boost`, term-level
    // field queries print it inside the field object, full-text ones omit it.
    let with_boost = |mut m: Map<String, Value>| {
        m.entry("boost").or_insert(json!(1.0));
        m
    };
    let field_level = |key: &str, boost: bool| -> Value {
        let Some(o) = body.as_object() else { return q.clone() };
        let out: Map<String, Value> = o
            .iter()
            .map(|(f, v)| {
                if f == "boost" || f == "_name" {
                    return (f.clone(), v.clone());
                }
                let mut m = match v {
                    Value::Object(m) => m.clone(),
                    other => {
                        let mut m = Map::new();
                        m.insert(key.to_string(), other.clone());
                        m
                    }
                };
                if key != "value" {
                    if let Some(x) = m.remove("value") {
                        m.insert(key.to_string(), x);
                    }
                }
                if boost {
                    m.entry("boost").or_insert(json!(1.0));
                }
                (f.clone(), Value::Object(m))
            })
            .collect();
        json!({ kind.clone(): out })
    };
    let sub = |m: &mut Map<String, Value>, k: &str| {
        if let Some(x) = m.get_mut(k) {
            *x = expand_query_shorthand(x);
        }
    };
    match kind.as_str() {
        "term" | "fuzzy" => field_level("value", false),
        "prefix" | "regexp" => field_level("value", true),
        "wildcard" => field_level("wildcard", true),
        "range" => field_level("value", true),
        "match" | "match_phrase" | "match_phrase_prefix" | "match_bool_prefix" => field_level("query", false),
        "match_all" | "match_none" | "exists" | "terms" | "ids" => json!({ kind.clone(): with_boost(body.as_object().cloned().unwrap_or_default()) }),
        "constant_score" => {
            let mut m = body.as_object().cloned().unwrap_or_default();
            sub(&mut m, "filter");
            json!({"constant_score": with_boost(m)})
        }
        "dis_max" => {
            let mut m = body.as_object().cloned().unwrap_or_default();
            if let Some(Value::Array(a)) = m.get_mut("queries") {
                a.iter_mut().for_each(|x| *x = expand_query_shorthand(x));
            }
            m.entry("tie_breaker").or_insert(json!(0.0));
            json!({"dis_max": with_boost(m)})
        }
        "bool" => {
            let mut m = body.as_object().cloned().unwrap_or_default();
            for c in ["must", "filter", "should", "must_not"] {
                if let Some(x) = m.get_mut(c) {
                    *x = match &*x {
                        Value::Array(a) => Value::Array(a.iter().map(expand_query_shorthand).collect()),
                        other => json!([expand_query_shorthand(other)]),
                    };
                }
            }
            json!({"bool": with_boost(m)})
        }
        _ => q.clone(),
    }
}
