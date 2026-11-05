//! Strict structure checks for the sections of a search body that ES parses
//! with `ObjectParser`s or hand-written loops (sort, highlight, suggest,
//! knn, retriever, collapse, rescore, `_source`, field lists, ...), with
//! ES's exception types, messages and token positions (`jsonpos` hints).
//! Called from [`crate::search::parse_request`] before the body is read.

use crate::error::{EsError, Result};
use crate::jsonpos::token_name;
use crate::query::strict::QUERY_NAMES;
use serde_json::{Map, Value};

const COMMON_SORT: &[&str] = &["order", "missing", "mode", "unmapped_type", "nested", "numeric_type", "format"];

/// Non-numeric string for a numeric ObjectParser field: `[parser] failed to parse field [f]` → `number_format_exception`.
fn numeric(parser: &str, field: &str, v: &Value, within: Option<&str>) -> Result<()> {
    match v {
        Value::String(s) if s.trim().parse::<f64>().is_err() => Err(EsError::field_failure(parser, field, within, EsError::number_format(s))),
        Value::Object(_) | Value::Array(_) | Value::Null | Value::Bool(_) => Err(EsError::unsupported_type_xc(parser, field, v, within)),
        _ => Ok(()),
    }
}

fn boolean(parser: &str, field: &str, v: &Value, within: Option<&str>) -> Result<()> {
    match v {
        Value::String(s) if !matches!(s.as_str(), "true" | "false") => Err(EsError::field_failure(parser, field, within, EsError::illegal_argument(format!("Failed to parse value [{s}] as only [true] or [false] are allowed.")))),
        Value::Object(_) | Value::Array(_) | Value::Null | Value::Number(_) => Err(EsError::unsupported_type_xc(parser, field, v, within)),
        _ => Ok(()),
    }
}

fn string(parser: &str, field: &str, v: &Value, within: Option<&str>) -> Result<()> {
    if v.is_string() {
        Ok(())
    } else {
        Err(EsError::unsupported_type_xc(parser, field, v, within))
    }
}

fn object_keys(parser: &str, o: &Map<String, Value>, allowed: &[&str], within: Option<&str>) -> Result<()> {
    for (k, _) in o {
        if !allowed.contains(&k.as_str()) {
            return Err(EsError::unknown_field_xc(parser, k, within, allowed));
        }
    }
    Ok(())
}

/// Validate a query object (`{"match_all": {...}}`) structurally — unknown
/// names and unknown fields — without an index context.
pub fn check_query(v: &Value, key: &str, within: Option<&str>) -> Result<()> {
    let Some(o) = v.as_object() else {
        return Err(EsError::parsing("[_na] query malformed, must start with start_object").at_key(key, within).with_lc());
    };
    if o.is_empty() {
        return Err(EsError::illegal_argument("query malformed, empty clause found at [@POS@]").at_key(key, within).cursor("end"));
    }
    if o.len() > 1 {
        let keys: Vec<&String> = o.keys().collect();
        return Err(EsError::parsing(format!("[{}] malformed query, expected [END_OBJECT] but found [FIELD_NAME]", keys[0])).at_key(keys[1], None).cursor("key").with_lc());
    }
    let (name, body) = o.iter().next().unwrap();
    // ES counts the depth as it meets each named query, before resolving it.
    let _depth = crate::query::limits::Depth::enter(&crate::query::limits::CHECK_DEPTH)?;
    if !QUERY_NAMES.contains(&name.as_str()) && !matches!(name.as_str(), "mlt" | "rule_query") && !(name == "deprecated" && crate::warnings::has_test_module("x-pack-test-deprecated-query")) {
        return Err(EsError::unknown_named("unknown query", name, None, QUERY_NAMES));
    }
    if !body.is_object() {
        return Err(EsError::parsing(format!("[{name}] query malformed, no start_object after query name")).at_key(name, None).with_lc());
    }
    crate::query::strict::check(name, body)?;
    // Recurse into the clauses whose failures ES wraps.
    let bo = body.as_object().unwrap();
    match name.as_str() {
        "bool" => {
            for c in ["must", "filter", "must_not", "should"] {
                match bo.get(c) {
                    Some(Value::Array(a)) => {
                        for (i, q) in a.iter().enumerate() {
                            check_query(q, c, Some("bool")).map_err(|e| e.in_clause(c, Some(i)))?;
                        }
                    }
                    Some(q) if !q.is_null() => check_query(q, c, Some("bool")).map_err(|e| e.in_clause(c, None))?,
                    _ => {}
                }
            }
        }
        "constant_score" => {
            if let Some(f) = bo.get("filter") {
                check_query(f, "filter", Some("constant_score"))?;
            }
        }
        "boosting" => {
            for c in ["positive", "negative"] {
                if let Some(q) = bo.get(c) {
                    check_query(q, c, Some("boosting"))?;
                }
            }
        }
        "dis_max" => {
            if let Some(Value::Array(a)) = bo.get("queries") {
                for (i, q) in a.iter().enumerate() {
                    check_query(q, "queries", Some("dis_max")).map_err(|e| if e.extra.iter().any(|(k, _)| k == "@k") { e } else { e.at_key("queries", Some("dis_max")).elem(i) })?;
                }
            }
        }
        "nested" | "has_child" | "has_parent" | "script_score" => {
            if let Some(q) = bo.get("query") {
                check_query(q, "query", Some(name))?;
            }
        }
        "function_score" => {
            if let Some(q) = bo.get("query") {
                check_query(q, "query", Some("function_score"))?;
            }
            if let Some(Value::Array(fns)) = bo.get("functions") {
                for f in fns {
                    if let Some(flt) = f.get("filter") {
                        check_query(flt, "filter", Some("functions"))?;
                    }
                }
            }
        }
        "knn" => {
            if let Some(f) = bo.get("filter") {
                let r = match f {
                    Value::Array(a) => a.iter().try_for_each(|q| check_query(q, "filter", Some("knn"))),
                    q => check_query(q, "filter", Some("knn")),
                };
                r.map_err(|e| EsError::field_failure("knn", "filter", Some("knn"), e))?;
            }
        }
        "pinned" => {
            if let Some(q) = bo.get("organic") {
                check_query(q, "organic", Some("pinned")).map_err(|e| {
                    let w = EsError::field_failure("pinned", "organic", Some("pinned"), e);
                    EsError::parsing("[pinned] failed to parse field [organic]").at_key("organic", Some("pinned")).cursor("inherit").prefixed("cursor").with_lc().caused_by(w)
                })?;
            }
        }
        "rule" | "rule_query" => {
            if let Some(q) = bo.get("organic") {
                check_query(q, "organic", Some("rule"))?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// `sort` section (`SortBuilder.fromXContent`).
pub fn check_sort(v: &Value) -> Result<()> {
    match v {
        Value::Array(a) => {
            for item in a {
                match item {
                    Value::String(_) => {}
                    Value::Object(o) => check_sort_object(o)?,
                    _ => return Err(EsError::illegal_argument("malformed sort format, within the sort array, an object, or an actual string are allowed")),
                }
            }
            Ok(())
        }
        Value::Object(o) => check_sort_object(o),
        _ => Ok(()),
    }
}

fn check_sort_object(o: &Map<String, Value>) -> Result<()> {
    for (field, spec) in o {
        match field.as_str() {
            "_geo_distance" => {
                if let Some(so) = spec.as_object() {
                    for (k, v) in so {
                        if matches!(k.as_str(), "order" | "unit" | "mode" | "distance_type" | "ignore_unmapped" | "validation_method" | "nested" | "missing") {
                            if k == "unit" {
                                if let Some(u) = v.as_str() {
                                    if crate::values::unit_meters(u).is_none() {
                                        return Err(EsError::illegal_argument(format!("No distance unit match [{u}]")));
                                    }
                                }
                            }
                            if k == "order" {
                                if let Some(x) = v.as_str() {
                                    if !matches!(x.to_ascii_lowercase().as_str(), "asc" | "desc") {
                                        return Err(EsError::illegal_argument(format!("No enum constant org.elasticsearch.search.sort.SortOrder.{}", x.to_ascii_uppercase())));
                                    }
                                }
                            }
                            continue;
                        }
                        if v.is_object() || v.is_array() || v.is_string() {
                            continue; // the geo point(s)
                        }
                        return Err(EsError::parsing_at_key(format!("[_geo_distance] does not support [{k}]"), k, Some("_geo_distance")));
                    }
                }
            }
            "_script" => {
                if let Some(so) = spec.as_object() {
                    let allowed = ["script", "type", "order", "mode", "nested"];
                    object_keys("_script", so, &allowed, Some("_script"))?;
                    if so.get("type").is_none() {
                        return Err(EsError::illegal_argument("Required [type]"));
                    }
                    if let Some(t) = so.get("type").and_then(Value::as_str) {
                        if !matches!(t, "string" | "number" | "version") {
                            return Err(EsError::field_failure("_script", "type", Some("_script"), EsError::illegal_argument(format!("Unknown ScriptSortType [{t}]"))));
                        }
                    }
                }
            }
            "_score" => {
                if let Some(so) = spec.as_object() {
                    object_keys("_score", so, &["order"], Some("_score"))?;
                }
            }
            _ => match spec {
                Value::String(ord) => {
                    if !matches!(ord.to_ascii_lowercase().as_str(), "asc" | "desc") {
                        return Err(EsError::illegal_argument(format!("No enum constant org.elasticsearch.search.sort.SortOrder.{}", ord.to_ascii_uppercase())));
                    }
                }
                Value::Object(so) => {
                    object_keys("field_sort", so, COMMON_SORT, Some(field))?;
                    if let Some(v) = so.get("order") {
                        string("field_sort", "order", v, Some(field))?;
                        let x = v.as_str().unwrap_or("");
                        if !matches!(x.to_ascii_lowercase().as_str(), "asc" | "desc") {
                            return Err(EsError::field_failure("field_sort", "order", Some(field), EsError::illegal_argument(format!("No enum constant org.elasticsearch.search.sort.SortOrder.{}", x.to_ascii_uppercase()))));
                        }
                    }
                    if let Some(v) = so.get("mode") {
                        string("field_sort", "mode", v, Some(field))?;
                        let x = v.as_str().unwrap_or("");
                        if !matches!(x.to_ascii_lowercase().as_str(), "min" | "max" | "sum" | "avg" | "median") {
                            return Err(EsError::field_failure("field_sort", "mode", Some(field), EsError::illegal_argument(format!("Unknown SortMode [{x}]"))));
                        }
                    }
                    if let Some(v) = so.get("missing") {
                        if v.is_object() || v.is_array() {
                            return Err(EsError::unsupported_type_xc("field_sort", "missing", v, Some(field)));
                        }
                    }
                    for k in ["unmapped_type", "format"] {
                        if let Some(v) = so.get(k) {
                            string("field_sort", k, v, Some(field))?;
                        }
                    }
                    if let Some(v) = so.get("numeric_type") {
                        string("field_sort", "numeric_type", v, Some(field))?;
                        let x = v.as_str().unwrap_or("");
                        if !matches!(x, "long" | "double" | "date" | "date_nanos") {
                            return Err(EsError::field_failure("field_sort", "numeric_type", Some(field), EsError::illegal_argument(format!("invalid value for [numeric_type], must be [long, double, date, date_nanos], got {x}"))));
                        }
                    }
                    if let Some(n) = so.get("nested") {
                        let Some(no) = n.as_object() else {
                            return Err(EsError::unsupported_type_xc("field_sort", "nested", n, Some(field)));
                        };
                        for (k, v) in no {
                            if !matches!(k.as_str(), "path" | "filter" | "max_children" | "nested") {
                                return Err(EsError::field_failure("field_sort", "nested", Some(field), EsError::illegal_argument(format!("malformed nested sort format, unknown field name [{k}]")).at_key(k, Some("nested"))));
                            }
                            if k == "filter" {
                                check_query(v, "filter", Some("nested")).map_err(|e| EsError::field_failure("field_sort", "nested", Some(field), EsError::parsing("Expected filter element.").cursor("inherit").with_lc().caused_by(e)))?;
                            }
                        }
                    }
                }
                _ => return Err(EsError::illegal_argument("malformed sort format, within the sort array, an object, or an actual string are allowed")),
            },
        }
    }
    Ok(())
}

const HIGHLIGHT_COMMON: &[&str] = &[
    "pre_tags", "post_tags", "fields", "order", "highlight_filter", "fragment_size", "number_of_fragments", "encoder", "require_field_match", "boundary_scanner", "boundary_max_scan",
    "boundary_chars", "boundary_scanner_locale", "type", "fragmenter", "no_match_size", "force_source", "phrase_limit", "options", "highlight_query", "max_analyzed_offset", "tags_schema",
];
const HIGHLIGHT_FIELD: &[&str] = &[
    "pre_tags", "post_tags", "order", "highlight_filter", "fragment_size", "number_of_fragments", "encoder", "require_field_match", "boundary_scanner", "boundary_max_scan", "boundary_chars",
    "boundary_scanner_locale", "type", "fragmenter", "no_match_size", "force_source", "phrase_limit", "options", "highlight_query", "max_analyzed_offset", "fragment_offset", "matched_fields",
];

/// `highlight` section (`HighlightBuilder.fromXContent`).
pub fn check_highlight(o: &Map<String, Value>) -> Result<()> {
    check_highlight_common("highlight", o, HIGHLIGHT_COMMON, None)?;
    if let Some(f) = o.get("fields") {
        let entries: Vec<(&String, &Value)> = match f {
            Value::Object(fo) => fo.iter().collect(),
            Value::Array(a) => a.iter().filter_map(|x| x.as_object()).flat_map(|m| m.iter()).collect(),
            other => return Err(EsError::unsupported_type_xc("highlight", "fields", other, Some("highlight"))),
        };
        for (name, spec) in entries {
            let inner = match spec {
                Value::Object(so) => check_highlight_common("highlight_field", so, HIGHLIGHT_FIELD, Some(name)),
                other => Err(EsError::x_content_parse(format!("[highlight_field] Expected START_OBJECT but was: {}", token_name(other))).at_key(name, Some("fields")).prefixed("cursor")),
            };
            inner.map_err(|e| EsError::field_failure("highlight", "fields", Some("highlight"), EsError::field_failure("fields", name, Some("fields"), e)))?;
        }
    }
    Ok(())
}

fn check_highlight_common(parser: &str, o: &Map<String, Value>, allowed: &[&str], within: Option<&str>) -> Result<()> {
    object_keys(parser, o, allowed, within)?;
    if let Some(t) = o.get("type").and_then(Value::as_str) {
        if !matches!(t, "unified" | "plain" | "fvh" | "highlighter" | "fast-vector-highlighter" | "postings" | "postings-highlighter" | "semantic") {
            // A top-level `type` fails on the first highlighted field.
            let first = || match o.get("fields") {
                Some(Value::Object(f)) => f.keys().next().map(String::as_str),
                Some(Value::Array(a)) => a.iter().find_map(|e| e.as_object().and_then(|m| m.keys().next())).map(String::as_str),
                _ => None,
            };
            return Err(shard_failure(&format!("unknown highlighter type [{t}] for the field [{}]", within.or_else(first).unwrap_or("*"))));
        }
    }
    for k in ["fragment_size", "number_of_fragments", "boundary_max_scan", "no_match_size", "phrase_limit", "max_analyzed_offset", "fragment_offset"] {
        if let Some(v) = o.get(k) {
            numeric(parser, k, v, within)?;
        }
    }
    for k in ["require_field_match", "force_source", "highlight_filter"] {
        if let Some(v) = o.get(k) {
            boolean(parser, k, v, within)?;
        }
    }
    for k in ["pre_tags", "post_tags", "matched_fields"] {
        if let Some(v) = o.get(k) {
            if !v.is_array() {
                return Err(EsError::unsupported_type_xc(parser, k, v, within));
            }
        }
    }
    if let Some(q) = o.get("highlight_query") {
        check_query(q, "highlight_query", within).map_err(|e| EsError::field_failure(parser, "highlight_query", within, e))?;
    }
    Ok(())
}

/// `suggest` section (`SuggestBuilder.fromXContent`).
pub fn check_suggest(o: &Map<String, Value>) -> Result<()> {
    for (name, spec) in o {
        if name == "text" {
            continue;
        }
        let Some(so) = spec.as_object() else {
            return Err(EsError::illegal_argument(format!("[suggest] does not support [{name}]")));
        };
        let mut suggester: Option<(&String, &Value)> = None;
        for (k, v) in so {
            match k.as_str() {
                "text" | "prefix" | "regex" => {}
                "term" | "phrase" | "completion" => {
                    if !v.is_object() {
                        return Err(EsError::parsing_at_key(format!("suggestion does not support [{k}]"), k, Some(name)));
                    }
                    suggester = Some((k, v));
                }
                other => {
                    if v.is_object() {
                        return Err(EsError::new(400, "named_object_not_found_exception", format!("unknown field [{other}]")).at_key(other, Some(name)).prefixed("cursor"));
                    }
                    return Err(EsError::parsing_at_key(format!("suggestion does not support [{other}]"), other, Some(name)));
                }
            }
        }
        let Some((kind, body)) = suggester else {
            return Err(EsError::new(400, "parse_exception", "missing suggestion object"));
        };
        let bo = body.as_object().unwrap();
        match kind.as_str() {
            "term" => {
                let allowed = ["field", "analyzer", "size", "shard_size", "suggest_mode", "accuracy", "sort", "string_distance", "max_edits", "max_inspections", "max_term_freq", "prefix_length", "min_word_length", "min_doc_freq", "text"];
                for (k, v) in bo {
                    if !allowed.contains(&k.as_str()) {
                        if v.is_object() || v.is_array() {
                            return Err(EsError::parsing_at_key(format!("suggester[term] parsing failed on [{k}]"), k, Some("term")));
                        }
                        return Err(EsError::parsing_at_key(format!("suggester[term] doesn't support field [{k}]"), k, Some("term")));
                    }
                }
            }
            "phrase" => {
                let allowed = ["field", "analyzer", "size", "shard_size", "text", "real_word_error_likelihood", "confidence", "separator", "max_errors", "gram_size", "force_unigrams", "token_limit", "pre_tag", "post_tag", "collate", "smoothing", "direct_generator", "highlight"];
                for (k, v) in bo {
                    if !allowed.contains(&k.as_str()) {
                        if v.is_array() || v.is_object() {
                            return Err(EsError::parsing_at_key(format!("suggester[phrase]  doesn't support array field [{k}]"), k, Some("phrase")));
                        }
                        return Err(EsError::parsing_at_key(format!("suggester[phrase] doesn't support field [{k}]"), k, Some("phrase")));
                    }
                }
                if let Some(Value::Object(c)) = bo.get("collate") {
                    for (k, _) in c {
                        if !matches!(k.as_str(), "query" | "params" | "prune") {
                            return Err(EsError::parsing_at_key(format!("suggester[phrase][collate] doesn't support field [{k}]"), k, Some("collate")));
                        }
                    }
                }
                if let Some(Value::Object(s)) = bo.get("smoothing") {
                    for (k, _) in s {
                        if !matches!(k.as_str(), "stupid_backoff" | "laplace" | "linear_interpolation" | "linear") {
                            return Err(EsError::illegal_argument(format!("suggester[phrase] doesn't support object field [{k}]")));
                        }
                    }
                }
                if let Some(gens) = bo.get("direct_generator") {
                    let items: Vec<&Value> = match gens {
                        Value::Array(a) => a.iter().collect(),
                        other => vec![other],
                    };
                    let allowed = ["field", "size", "suggest_mode", "max_edits", "max_inspections", "max_term_freq", "prefix_length", "min_word_length", "min_doc_freq", "pre_filter", "post_filter"];
                    for g in items {
                        if let Some(go) = g.as_object() {
                            object_keys("direct_generator", go, &allowed, Some("direct_generator"))?;
                        }
                    }
                }
            }
            "completion" => {
                let allowed = ["field", "analyzer", "size", "shard_size", "prefix", "regex", "fuzzy", "contexts", "skip_duplicates", "text"];
                object_keys("completion", bo, &allowed, Some("completion"))?;
                if let Some(c) = bo.get("contexts") {
                    if !c.is_object() {
                        return Err(EsError::unsupported_type_xc("completion", "contexts", c, Some("completion")));
                    }
                }
                if let Some(Value::Object(f)) = bo.get("fuzzy") {
                    object_keys("fuzzy", f, &["fuzziness", "transpositions", "min_length", "prefix_length", "unicode_aware", "max_determinized_states"], Some("fuzzy")).map_err(|e| EsError::field_failure("completion", "fuzzy", Some("completion"), e))?;
                }
                if let Some(v) = bo.get("skip_duplicates") {
                    boolean("completion", "skip_duplicates", v, Some("completion"))?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

const KNN_FIELDS: &[&str] = &["field", "k", "num_candidates", "visit_percentage", "query_vector", "query_vector_builder", "filter", "boost", "_name", "similarity", "inner_hits", "rescore_vector"];

/// One `knn` section entry (`KnnSearchBuilder` ObjectParser).
pub fn check_knn_entry(o: &Map<String, Value>) -> Result<()> {
    object_keys("knn", o, KNN_FIELDS, Some("knn"))?;
    for k in ["k", "num_candidates", "visit_percentage", "boost", "similarity"] {
        if let Some(v) = o.get(k) {
            numeric("knn", k, v, Some("knn"))?;
        }
    }
    // `KnnSearchBuilder.toQueryBuilder` hands it to the query's `boost()`.
    if let Some(b) = o.get("boost").and_then(|v| v.as_f64().or_else(|| v.as_str()?.trim().parse().ok())) {
        crate::query::dsl::valid_boost(b as f32)?;
    }
    if let Some(v) = o.get("query_vector") {
        match v {
            Value::Array(a) => {
                for (i, x) in a.iter().enumerate() {
                    if let Some(s) = x.as_str() {
                        if s.parse::<f64>().is_err() {
                            return Err(EsError::field_failure("knn", "query_vector", Some("knn"), EsError::number_format(s).at_key("query_vector", Some("knn")).elem(i)));
                        }
                    }
                }
            }
            // ES 9: a hex or base64 string, decoded against the field.
            Value::String(_) | Value::Number(_) => {}
            other => return Err(EsError::field_failure("knn", "query_vector", Some("knn"), EsError::parsing(format!("Unknown type [{}] for parsing vector", token_name(other))).cursor("inherit").with_lc())),
        }
    }
    if let Some(f) = o.get("filter") {
        let r = match f {
            Value::Array(a) => a.iter().try_for_each(|q| check_query(q, "filter", Some("knn"))),
            q => check_query(q, "filter", Some("knn")),
        };
        r.map_err(|e| EsError::field_failure("knn", "filter", Some("knn"), e))?;
    }
    if let Some(Value::Object(qvb)) = o.get("query_vector_builder") {
        for (k, v) in qvb {
            if k == "lookup" {
                continue;
            }
            if k == "embedding" {
                crate::inference::semantic::parse_embedding_builder(v)?;
                continue;
            }
            if k != "text_embedding" {
                return Err(EsError::field_failure("knn", "query_vector_builder", Some("knn"), EsError::field_failure("query_vector_builder", k, Some("query_vector_builder"), EsError::new(400, "named_object_not_found_exception", format!("unknown field [{k}]")).at_key(k, Some("query_vector_builder")).cursor("key").prefixed("cursor"))));
            }
            if let Some(te) = v.as_object() {
                object_keys("text_embedding", te, &["model_id", "model_text"], Some("text_embedding")).map_err(|e| EsError::field_failure("knn", "query_vector_builder", Some("knn"), EsError::field_failure("query_vector_builder", "text_embedding", Some("query_vector_builder"), e)))?;
            }
        }
    }
    if let Some(Value::Object(ih)) = o.get("inner_hits") {
        check_inner_hits(ih).map_err(|e| EsError::field_failure("knn", "inner_hits", Some("knn"), e))?;
    }
    if let Some(Value::Object(rv)) = o.get("rescore_vector") {
        object_keys("rescore_vector", rv, &["oversample"], Some("rescore_vector")).map_err(|e| EsError::field_failure("knn", "rescore_vector", Some("knn"), e))?;
    }
    if o.get("query_vector").is_none() && o.get("query_vector_builder").is_none() {
        return Err(EsError::illegal_argument("either [query_vector_builder] or [query_vector] must be provided"));
    }
    if o.get("field").is_none() {
        return Err(EsError::illegal_argument("Required [field]"));
    }
    Ok(())
}

pub fn check_knn(v: &Value) -> Result<()> {
    match v {
        Value::Array(a) => {
            for x in a {
                let Some(o) = x.as_object() else {
                    return Err(EsError::x_content_parse(format!("malformed knn format, within the knn search array only objects are allowed; found {}", token_name(x))).at_key("knn", None).elem(0).prefixed("cursor"));
                };
                check_knn_entry(o)?;
            }
            Ok(())
        }
        Value::Object(o) => check_knn_entry(o),
        _ => Ok(()),
    }
}

const INNER_HITS: &[&str] = &["name", "ignore_unmapped", "from", "size", "version", "seq_no_primary_term", "explain", "track_scores", "stored_fields", "docvalue_fields", "fields", "script_fields", "sort", "_source", "highlight", "collapse"];

pub fn check_inner_hits(o: &Map<String, Value>) -> Result<()> {
    object_keys("inner_hits", o, INNER_HITS, Some("inner_hits"))
}

/// A failure ES raises on every shard: `search_phase_execution_exception` over the cause (twice, as ES renders it).
pub fn shard_failure(msg: &str) -> EsError {
    EsError::search_phase("", EsError::illegal_argument(msg).caused_by(EsError::illegal_argument(msg)))
}

/// `collapse` section (`CollapseBuilder` ObjectParser).
pub fn check_collapse(v: &Value) -> Result<()> {
    let Some(o) = v.as_object() else { return Ok(()) };
    object_keys("collapse", o, &["field", "inner_hits", "max_concurrent_group_searches"], Some("collapse"))?;
    if o.get("field").is_none() {
        return Err(shard_failure("no collapse field specified"));
    }
    if let Some(f) = o.get("field") {
        string("collapse", "field", f, Some("collapse"))?;
    }
    if let Some(n) = o.get("max_concurrent_group_searches") {
        numeric("collapse", "max_concurrent_group_searches", n, Some("collapse"))?;
    }
    if let Some(ih) = o.get("inner_hits") {
        let items: Vec<&Value> = match ih {
            Value::Array(a) => a.iter().collect(),
            other => vec![other],
        };
        for i in items {
            if let Some(io) = i.as_object() {
                check_inner_hits(io).map_err(|e| EsError::field_failure("collapse", "inner_hits", Some("collapse"), e))?;
            }
        }
    }
    Ok(())
}

/// `rescore` section (`RescorerBuilder.parseFromXContent`).
pub fn check_rescore(v: &Value) -> Result<()> {
    let items: Vec<&Value> = match v {
        Value::Array(a) => a.iter().collect(),
        other => vec![other],
    };
    for item in items {
        let Some(o) = item.as_object() else { continue };
        let mut rescorer: Option<&str> = None;
        for (k, val) in o {
            match (k.as_str(), val) {
                ("window_size", Value::String(s)) if s.trim().parse::<i64>().is_err() => return Err(EsError::number_format(s)),
                ("window_size", _) => {}
                ("query", Value::Object(q)) => {
                    rescorer = Some("query");
                    object_keys("query", q, &["rescore_query", "query_weight", "rescore_query_weight", "score_mode"], Some("rescore"))?;
                    for f in ["query_weight", "rescore_query_weight"] {
                        if let Some(x) = q.get(f) {
                            numeric("query", f, x, Some("rescore"))?;
                        }
                    }
                    if let Some(Value::String(m)) = q.get("score_mode") {
                        if !matches!(m.as_str(), "avg" | "max" | "min" | "total" | "multiply") {
                            return Err(EsError::field_failure("query", "score_mode", Some("rescore"), EsError::illegal_argument(format!("illegal score_mode [{m}]"))));
                        }
                    }
                    if let Some(rq) = q.get("rescore_query") {
                        check_query(rq, "rescore_query", Some("query")).map_err(|e| EsError::field_failure("query", "rescore_query", Some("rescore"), e))?;
                    }
                }
                ("script", Value::Object(q)) => {
                    rescorer = Some("script");
                    object_keys("script", q, &["script"], Some("rescore"))?;
                }
                ("learning_to_rank", Value::Object(q)) => {
                    rescorer = Some("learning_to_rank");
                    object_keys("learning_to_rank", q, &["model_id", "params"], Some("rescore"))?;
                }
                (other, Value::Object(_)) => return Err(EsError::new(400, "named_object_not_found_exception", format!("unknown field [{other}]")).at_key(other, Some("rescore")).prefixed("cursor")),
                (other, _) => return Err(EsError::parsing_at_key(format!("rescore doesn't support [{other}]"), other, Some("rescore"))),
            }
        }
        if rescorer.is_none() {
            return Err(EsError::parsing("missing rescore type").at_key("rescore", None).cursor("end").with_lc());
        }
    }
    Ok(())
}

/// `retriever` section (`RetrieverBuilder.parseTopLevelRetrieverBuilder`).
pub fn check_retriever(v: &Value) -> Result<()> {
    const NAMES: &[&str] = &["standard", "knn", "rrf", "text_similarity_reranker", "rule", "linear", "rescorer", "pinned", "random_reranker", "diversify"];
    let Some(o) = v.as_object() else { return Ok(()) };
    let mut it = o.iter();
    let Some((name, body)) = it.next() else { return Ok(()) };
    if let Some((second, _)) = it.next() {
        return Err(EsError::parsing(format!("[{name}] malformed retriever, expected [END_OBJECT] but found [FIELD_NAME]")).at_key(second, None).cursor("key").with_lc());
    }
    if !NAMES.contains(&name.as_str()) {
        return Err(EsError::unknown_named("unknown retriever", name, None, NAMES));
    }
    let Some(b) = body.as_object() else { return Ok(()) };
    let common = ["filter", "_name", "min_score"];
    let own: &[&str] = match name.as_str() {
        "standard" => &["query", "sort", "search_after", "terminate_after", "collapse"],
        "knn" => &["field", "k", "num_candidates", "visit_percentage", "query_vector", "query_vector_builder", "similarity", "rescore_vector"],
        "rrf" => &["retrievers", "rank_window_size", "rank_constant", "query", "fields"],
        "text_similarity_reranker" => &["retriever", "field", "inference_id", "inference_text", "rank_window_size", "failure_allowed", "allow_rerank_failures", "chunk_rescorer"],
        // `_resolved`: the matched rules, filled in by the search preprocessing (`retriever::resolve_rules`).
        "rule" => &["retriever", "ruleset_ids", "match_criteria", "rank_window_size", "_resolved"],
        "linear" => &["retrievers", "rank_window_size", "query", "fields", "normalizer"],
        "rescorer" => &["retriever", "rescore"],
        "pinned" => &["retriever", "ids", "docs", "rank_window_size"],
        "random_reranker" => &["retriever", "field", "rank_window_size", "seed"],
        "diversify" => &["retriever", "type", "field", "rank_window_size", "query_vector", "query_vector_builder", "lambda", "size"],
        _ => &[],
    };
    let allowed: Vec<&str> = own.iter().chain(common.iter()).copied().collect();
    let pair = matches!(name.as_str(), "rule" | "rescorer" | "pinned");
    for (k, val) in b {
        if !allowed.contains(&k.as_str()) {
            return Err(if pair { EsError::unknown_field_wrapped(name, k, Some(name), &allowed) } else { EsError::unknown_field_xc(name, k, Some(name), &allowed) });
        }
        match k.as_str() {
            "_name" => string(name, k, val, Some(name))?,
            "min_score" | "k" | "num_candidates" | "rank_window_size" | "rank_constant" | "terminate_after" => numeric(name, k, val, Some(name))?,
            "query" if name == "standard" => check_query(val, "query", Some(name)).map_err(|e| EsError::field_failure(name, "query", Some(name), e))?,
            "filter" => {
                let r = match val {
                    Value::Array(a) => a.iter().try_for_each(|q| check_query(q, "filter", Some(name))),
                    q => check_query(q, "filter", Some(name)),
                };
                r.map_err(|e| EsError::field_failure(name, "filter", Some(name), e))?;
            }
            "sort" => check_sort(val).map_err(|e| EsError::field_failure(name, "sort", Some(name), e))?,
            "collapse" => check_collapse(val).map_err(|e| EsError::field_failure(name, "collapse", Some(name), e))?,
            "retrievers" => {
                if let Value::Array(a) = val {
                    for r in a {
                        let Some(ro) = r.as_object() else {
                            return Err(EsError::field_failure(name, "retrievers", Some(name), EsError::new(400, "named_object_not_found_exception", "unknown field [retrievers]").at_key("retrievers", Some(name)).elem(0).cursor("after").prefixed("cursor")));
                        };
                        // Weighted entries (`linear`, weighted `rrf`): `{retriever, weight, normalizer}`.
                        if let Some(inner) = ro.get("retriever") {
                            check_retriever(inner).map_err(|e| EsError::field_failure(name, "retrievers", Some(name), e))?;
                            continue;
                        }
                        if let Some((rn, _)) = ro.iter().next() {
                            if !NAMES.contains(&rn.as_str()) {
                                return Err(EsError::field_failure(name, "retrievers", Some(name), EsError::new(400, "named_object_not_found_exception", format!("unknown field [{rn}]")).at_key(rn, None).cursor("key").prefixed("cursor")));
                            }
                        }
                        check_retriever(r).map_err(|e| EsError::field_failure(name, "retrievers", Some(name), e))?;
                    }
                }
            }
            "retriever" => check_retriever(val).map_err(|e| EsError::field_failure(name, "retriever", Some(name), e))?,
            _ => {}
        }
    }
    Ok(())
}

/// `_source` (`FetchSourceContext.fromXContent`).
pub fn check_source(v: &Value) -> Result<()> {
    match v {
        Value::Bool(_) | Value::String(_) | Value::Array(_) => Ok(()),
        Value::Object(o) => {
            for (k, x) in o {
                if !matches!(k.as_str(), "includes" | "include" | "excludes" | "exclude" | "exclude_vectors") {
                    return Err(EsError::unknown_key(k, x, Some("_source")));
                }
            }
            Ok(())
        }
        other => Err(EsError::parsing_at_key(format!("Expected one of [VALUE_BOOLEAN, VALUE_STRING, START_ARRAY, START_OBJECT] but found [{}]", token_name(other)), "_source", None)),
    }
}

/// `fields` / `docvalue_fields` entries (`FieldAndFormat` ObjectParser).
pub fn check_field_list(key: &str, v: &Value) -> Result<()> {
    if let Value::Array(a) = v {
        for x in a {
            if let Some(o) = x.as_object() {
                object_keys("fetch_field_and_format", o, &["field", "format", "include_unmapped"], Some(key))?;
            }
        }
    }
    Ok(())
}

/// `indices_boost` (array of single-entry objects).
pub fn check_indices_boost(v: &Value) -> Result<()> {
    if let Value::Array(a) = v {
        for x in a {
            if let Some(o) = x.as_object() {
                if o.len() > 1 {
                    let second = o.keys().nth(1).unwrap();
                    return Err(EsError::parsing("Expected [END_OBJECT] in [indices_boost] but found [FIELD_NAME]").at_key(second, Some("indices_boost")).cursor("key").with_lc());
                }
                // The shard wraps its query in a Lucene `BoostQuery`.
                if let Some(b) = o.values().next().and_then(|v| v.as_f64().or_else(|| v.as_str()?.trim().parse().ok())).map(|b| b as f32) {
                    if !b.is_finite() || b.total_cmp(&0.0).is_lt() {
                        return Err(EsError::illegal_argument(format!("boost must be a positive float, got {}", crate::similarity::java_float(b))));
                    }
                }
            }
        }
    }
    Ok(())
}

/// `pit` (`PointInTimeBuilder` ObjectParser).
pub fn check_pit(o: &Map<String, Value>) -> Result<()> {
    object_keys("pit", o, &["id", "keep_alive"], Some("pit"))?;
    match o.get("id") {
        None => return Err(EsError::illegal_argument("point in time id is not provided")),
        Some(Value::String(id)) => {
            if let Err(e) = crate::esql::async_id::java_b64_decode(id, true) {
                return Err(EsError::field_failure("pit", "id", Some("pit"), e));
            }
        }
        Some(other) => return Err(EsError::unsupported_type_xc("pit", "id", other, Some("pit"))),
    }
    if let Some(Value::String(ka)) = o.get("keep_alive") {
        if crate::date::parse_time_value(ka).is_none() {
            return Err(EsError::field_failure("pit", "keep_alive", Some("pit"), EsError::illegal_argument(format!("failed to parse setting [keep_alive] with value [{ka}] as a time value: unit is missing or unrecognized"))));
        }
    }
    Ok(())
}

/// `script_fields` entries.
pub fn check_script_fields(o: &Map<String, Value>) -> Result<()> {
    for (name, spec) in o {
        if let Some(so) = spec.as_object() {
            for (k, v) in so {
                if !matches!(k.as_str(), "script" | "ignore_failure") {
                    return Err(EsError::unknown_key(k, v, Some(name)));
                }
            }
        }
    }
    Ok(())
}
