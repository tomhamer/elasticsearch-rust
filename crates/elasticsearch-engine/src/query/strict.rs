//! Strict query-DSL structure checks: the field-level validation each ES
//! `QueryBuilder.fromXContent` performs before building the query, with
//! ES's exact exception types, messages and token positions (as
//! `jsonpos` hints). Called once per query object from `parse_named`;
//! nested queries are checked when they are parsed in turn.

use crate::error::{EsError, Result};
use crate::jsonpos::{jackson_token, token_name};
use serde_json::{Map, Value};

/// Every query name ES 8.19 registers (server + x-pack), the candidates of
/// `unknown query [x] did you mean ...?`.
pub const QUERY_NAMES: &[&str] = &[
    "bool", "boosting", "combined_fields", "constant_score", "dis_max", "distance_feature", "exists", "field_masking_span", "function_score", "fuzzy",
    "geo_bounding_box", "geo_distance", "geo_grid", "geo_polygon", "geo_shape", "has_child", "has_parent", "ids", "intervals", "knn", "kql", "match",
    "match_all", "match_bool_prefix", "match_none", "match_phrase", "match_phrase_prefix", "more_like_this", "multi_match", "nested", "parent_id",
    "percolate", "pinned", "prefix", "query_string", "random_sampling", "range", "rank_feature", "regexp", "rule", "script", "script_score", "semantic",
    "shape", "simple_query_string", "span_containing", "span_field_masking", "span_first", "span_gap", "span_multi", "span_near", "span_not", "span_or",
    "span_term", "span_within", "sparse_vector", "term", "terms", "terms_set", "text_expansion", "weighted_tokens", "wildcard", "wrapper",
];

/// Score-function names (`function_score` `functions` entries).
pub const SCORE_FUNCTIONS: &[&str] = &["script_score", "random_score", "field_value_factor", "gauss", "linear", "exp"];

const COMMON: &[&str] = &["boost", "_name"];

/// How a query parser reports an unknown field.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Style {
    /// `[name] query does not support [f]` (ParsingException at the value).
    DoesNotSupport,
    /// Like `DoesNotSupport`, but an object / array value is
    /// `[name] unknown token [START_OBJECT] after [f]`.
    DoesNotSupportToken,
    /// `[simple_query_string] unsupported field [f]` (+ unknown token for objects).
    Unsupported,
    /// `failed to parse [name] query. field [f] is not supported`.
    FailedNotSupported,
    /// `failed to parse [name] query. unexpected field [f]`.
    FailedUnexpected,
    /// `query does not support [f]` (geo_shape / shape: no query name).
    Bare,
    /// `unknown field [f]` (weighted_tokens).
    UnknownField,
    /// ObjectParser whose `fromXContent` wraps in a ParsingException (match_all, ids, ...).
    ObjectPair,
    /// ObjectParser reported bare (bool, knn, ...).
    ObjectBare,
}

// ---------------------------------------------------------------------------
// Value typing (what `parser.intValue()` / `booleanValue()` / enum parsing do)
// ---------------------------------------------------------------------------

const NUM_FIELDS: &[&str] = &[
    "boost", "slop", "prefix_length", "max_expansions", "tie_breaker", "negative_boost", "max_boost", "min_score", "weight", "max_determinized_states", "phrase_slop",
    "fuzzy_prefix_length", "fuzzy_max_expansions", "k", "num_candidates", "similarity", "min_term_freq", "min_doc_freq", "max_doc_freq", "max_query_terms", "min_word_length",
    "max_word_length", "boost_terms", "end", "dist", "pre", "post", "min_children", "max_children", "scaling_factor", "pivot", "exponent", "factor",
];
const BOOL_FIELDS: &[&str] = &[
    "lenient", "case_insensitive", "fuzzy_transpositions", "transpositions", "auto_generate_synonyms_phrase_query", "analyze_wildcard", "allow_leading_wildcard",
    "enable_position_increments", "escape", "in_order", "ignore_unmapped", "include", "fail_on_unsupported_field", "prune", "include_lower", "include_upper", "collect_payloads",
];
/// Option keys a value-shaped token is expected for (an object / array there is an "unknown token").
const VALUE_FIELDS: &[&str] = &[
    "query", "analyzer", "operator", "zero_terms_query", "minimum_should_match", "fuzziness", "type", "rewrite", "fuzzy_rewrite", "default_operator", "time_zone", "format",
    "relation", "flags", "value", "term", "wildcard", "prefix", "gte", "gt", "lte", "lt", "from", "to", "quote_analyzer", "quote_field_suffix", "default_field", "score_mode",
    "boost_mode", "distance", "distance_type", "validation_method", "unit", "path", "parent_type", "id", "model_id", "model_text", "field", "inference_id", "like", "unlike",
    "stop_words", "fields", "clauses", "match", "little", "big", "include", "exclude", "positive", "negative", "filter", "organic", "script", "origin", "like_text",
];

fn upper(s: &str) -> String {
    s.to_ascii_uppercase()
}

fn bad_bool(s: &str) -> EsError {
    EsError::illegal_argument(format!("Failed to parse value [{s}] as only [true] or [false] are allowed."))
}

/// Type / enum validation of one option `k: v` of query `name` (manual parsers).
fn typed(name: &str, k: &str, v: &Value, style: Style) -> Result<()> {
    let within = Some(name);
    match v {
        Value::String(s) => {
            if NUM_FIELDS.contains(&k) && !(k == "fuzziness") && s.trim().parse::<f64>().is_err() {
                return Err(EsError::number_format(s));
            }
            if BOOL_FIELDS.contains(&k) && !matches!(s.as_str(), "true" | "false") {
                return Err(bad_bool(s));
            }
            match k {
                "operator" | "default_operator" if !matches!(upper(s).as_str(), "AND" | "OR") => {
                    let e = EsError::illegal_argument(format!("No enum constant org.elasticsearch.index.query.Operator.{}", upper(s)));
                    return Err(if name == "combined_fields" { EsError::field_failure(name, k, within, e) } else { e });
                }
                "zero_terms_query" if !matches!(upper(s).as_str(), "NONE" | "ALL") => {
                    return Err(EsError::parsing_at_key(format!("Unsupported zero_terms_query value [{s}]"), k, within));
                }
                // `ShapeRelation.getRelationByName`; `range` allows all but `disjoint`
                // (`RangeQueryBuilder.relation`).
                "relation" if !matches!(upper(s).as_str(), "WITHIN" | "CONTAINS" | "INTERSECTS" | "DISJOINT") => {
                    return Err(EsError::illegal_argument(format!("{s} is not a valid relation")));
                }
                "relation" if name == "range" && upper(s) == "DISJOINT" => {
                    return Err(EsError::illegal_argument(format!("[range] query does not support relation [{s}]")));
                }
                "score_mode" if matches!(name, "nested" | "has_child") && !matches!(s.as_str(), "none" | "avg" | "max" | "sum" | "min") => {
                    return Err(EsError::illegal_argument(format!("No score mode for child query [{s}] found")));
                }
                "score_mode" if name == "function_score" && !matches!(s.as_str(), "multiply" | "sum" | "avg" | "first" | "max" | "min") => {
                    return Err(EsError::illegal_argument(format!("No enum constant org.elasticsearch.common.lucene.search.function.FunctionScoreQuery.ScoreMode.{}", upper(s))));
                }
                "boost_mode" if name == "function_score" && !matches!(s.as_str(), "multiply" | "replace" | "sum" | "avg" | "max" | "min") => {
                    return Err(EsError::illegal_argument(format!("No enum constant org.elasticsearch.common.lucene.search.function.CombineFunction.{}", upper(s))));
                }
                "type" if matches!(name, "multi_match" | "query_string") && !matches!(s.as_str(), "best_fields" | "most_fields" | "cross_fields" | "phrase" | "phrase_prefix" | "bool_prefix") => {
                    return Err(EsError::new(400, "parse_exception", format!("failed to parse [multi_match] query type [{s}]. unknown type.")));
                }
                "flags" if name == "simple_query_string" => {
                    for f in s.split('|').map(str::trim).filter(|f| !f.is_empty()) {
                        if !matches!(upper(f).as_str(), "ALL" | "NONE" | "AND" | "OR" | "NOT" | "PREFIX" | "PHRASE" | "PRECEDENCE" | "ESCAPE" | "WHITESPACE" | "FUZZY" | "NEAR" | "SLOP") {
                            return Err(EsError::illegal_argument(format!("Unknown simple_query_string flag [{}]", upper(f))));
                        }
                    }
                }
                "flags" if name == "regexp" => {
                    for f in s.split('|').map(str::trim).filter(|f| !f.is_empty()) {
                        if !matches!(upper(f).as_str(), "ALL" | "NONE" | "INTERSECTION" | "COMPLEMENT" | "EMPTY" | "ANYSTRING" | "INTERVAL") {
                            return Err(EsError::illegal_argument(format!("Unknown regexp flag [{}]", upper(f))));
                        }
                    }
                }
                "rewrite" | "fuzzy_rewrite" => {
                    let ok = matches!(s.as_str(), "constant_score" | "constant_score_blended" | "constant_score_boolean" | "scoring_boolean")
                        || ["top_terms_blended_freqs_", "top_terms_boost_", "top_terms_"].iter().any(|p| s.strip_prefix(p).is_some_and(|n| n.parse::<u32>().is_ok()));
                    if !ok {
                        return Err(EsError::search_phase("", EsError::query_shard("", "", format!("failed to create query: Failed to parse rewrite_method [{s}]"))));
                    }
                }
                "distance_type" if name == "geo_distance" && !matches!(s.as_str(), "arc" | "plane") => {
                    return Err(EsError::illegal_argument(format!("No geo distance for [{s}]")));
                }
                "validation_method" if !matches!(upper(s).as_str(), "COERCE" | "IGNORE_MALFORMED" | "STRICT") => {
                    return Err(EsError::illegal_argument(format!("operator needs to be either [COERCE, IGNORE_MALFORMED, STRICT], but not [{s}]")));
                }
                _ => {}
            }
            Ok(())
        }
        Value::Number(n) => {
            if BOOL_FIELDS.contains(&k) {
                return Err(EsError::jackson_type(k, within, v, "not of boolean type"));
            }
            if k == "type" && matches!(name, "multi_match" | "query_string") {
                return Err(EsError::new(400, "parse_exception", format!("failed to parse [multi_match] query type [{n}]. unknown type.")));
            }
            if k == "default_operator" || (k == "operator" && !NUM_FIELDS.contains(&k)) {
                return Err(EsError::illegal_argument(format!("No enum constant org.elasticsearch.index.query.Operator.{n}")));
            }
            Ok(())
        }
        Value::Bool(_) => {
            if NUM_FIELDS.contains(&k) {
                return Err(EsError::jackson_type(k, within, v, "not numeric, can not use numeric value accessors"));
            }
            Ok(())
        }
        Value::Object(_) | Value::Array(_) | Value::Null => {
            if NUM_FIELDS.contains(&k) {
                return Err(EsError::jackson_type(k, within, v, "not numeric, can not use numeric value accessors"));
            }
            if BOOL_FIELDS.contains(&k) {
                return Err(EsError::jackson_type(k, within, v, "not of boolean type"));
            }
            match (name, k) {
                ("term" | "span_term" | "range", "value" | "term" | "gte" | "gt" | "lte" | "lt" | "from" | "to") if !v.is_null() => {
                    return Err(EsError::parsing("Unexpected token [END_OBJECT] found after the main object.").cursor("rootend").with_lc());
                }
                ("wildcard" | "regexp" | "prefix", "value" | "wildcard" | "prefix") if !v.is_null() => {
                    return Err(EsError::illegal_argument(format!("Expected text at @POS@ but found {}", token_name(v))).at_key(k, within));
                }
                ("fuzzy", "value" | "term") => {
                    return Err(EsError::parsing_at_key(format!("[fuzzy] unexpected token [{}] after [{k}]", token_name(v)), k, within));
                }
                _ => {}
            }
            if VALUE_FIELDS.contains(&k) && matches!(style, Style::DoesNotSupportToken | Style::Unsupported) && !matches!(k, "fields" | "filter" | "query" if matches!(name, "nested" | "has_child" | "has_parent" | "function_score" | "script_score" | "boosting" | "constant_score" | "span_multi" | "span_first" | "span_not" | "span_containing" | "span_within" | "span_field_masking" | "field_masking_span")) {
                if !(k == "fields" && v.is_array()) && !(k == "query" && name == "query_string" && false) {
                    return Err(EsError::parsing_at_key(format!("[{name}] unknown token [{}] after [{k}]", token_name(v)), k, within));
                }
            }
            Ok(())
        }
    }
}

/// ES `ParsingException` positioned at the end of query `name`'s body.
fn at_end(name: &str, msg: impl Into<String>) -> EsError {
    EsError::parsing(msg).at_key(name, None).cursor("end").with_lc()
}

fn unknown(style: Style, name: &str, parser: &str, field: &str, v: &Value, allowed: &[&str]) -> EsError {
    let within = Some(name);
    let is_container = v.is_object() || v.is_array();
    match style {
        Style::DoesNotSupport => EsError::query_unsupported(name, field),
        Style::DoesNotSupportToken if is_container || v.is_null() => EsError::parsing_at_key(format!("[{name}] unknown token [{}] after [{field}]", token_name(v)), field, within),
        Style::DoesNotSupportToken => EsError::query_unsupported(name, field),
        Style::Unsupported if is_container => EsError::parsing_at_key(format!("[{name}] unknown token [{}] after [{field}]", token_name(v)), field, within),
        Style::Unsupported => EsError::parsing_at_key(format!("[{name}] unsupported field [{field}]"), field, within),
        Style::FailedNotSupported => EsError::parsing_at_key(format!("failed to parse [{name}] query. field [{field}] is not supported"), field, within),
        Style::FailedUnexpected => EsError::parsing_at_key(format!("failed to parse [{name}] query. unexpected field [{field}]"), field, within),
        Style::Bare => EsError::parsing_at_key(format!("query does not support [{field}]"), field, within),
        Style::UnknownField => EsError::parsing_at_key(format!("unknown field [{field}]"), field, within),
        Style::ObjectPair => EsError::unknown_field_wrapped(parser, field, within, allowed),
        Style::ObjectBare => EsError::unknown_field_xc(parser, field, within, allowed),
    }
}

fn check_keys(style: Style, name: &str, parser: &str, o: &Map<String, Value>, allowed: &[&str]) -> Result<()> {
    for (k, v) in o {
        if !allowed.contains(&k.as_str()) && !COMMON.contains(&k.as_str()) {
            return Err(unknown(style, name, parser, k, v, allowed));
        }
        if matches!(style, Style::ObjectPair | Style::ObjectBare) {
            object_parser_value(style == Style::ObjectPair, parser, name, k, v)?;
        } else {
            typed(name, k, v, style)?;
        }
    }
    Ok(())
}

/// ObjectParser-declared field typing: numbers via `failed to parse field`
/// (number_format), booleans via `Failed to parse value`, declared objects
/// / arrays via `doesn't support values of type`.
fn object_parser_value(pair: bool, parser: &str, name: &str, k: &str, v: &Value) -> Result<()> {
    let resolved = if k == "script" { Some(crate::script::search_script_source(v)?.0) } else { None };
    let v = resolved.as_ref().unwrap_or(v);
    let within = Some(name);
    let wrap_pair = |e: EsError| -> EsError {
        if pair {
            let msg = e.reason.clone();
            EsError::parsing(msg).at_key(k, within).cursor("inherit").prefixed("cursor").with_lc().caused_by(e)
        } else {
            e
        }
    };
    if NUM_FIELDS.contains(&k) {
        return match v {
            Value::String(s) if s.trim().parse::<f64>().is_err() => Err(wrap_pair(EsError::field_failure(parser, k, within, EsError::number_format(s)))),
            Value::Object(_) | Value::Array(_) | Value::Null | Value::Bool(_) => Err(wrap_pair(EsError::unsupported_type_xc(parser, k, v, within))),
            _ => Ok(()),
        };
    }
    if BOOL_FIELDS.contains(&k) {
        return match v {
            Value::String(s) if !matches!(s.as_str(), "true" | "false") => Err(wrap_pair(EsError::field_failure(parser, k, within, bad_bool(s)))),
            Value::Object(_) | Value::Array(_) | Value::Null | Value::Number(_) => Err(wrap_pair(EsError::unsupported_type_xc(parser, k, v, within))),
            _ => Ok(()),
        };
    }
    let string_only = matches!(k, "_name" | "field" | "inference_id" | "model_id" | "model_text" | "query" if name != "script_score" && !(k == "query" && matches!(name, "knn" | "rule" | "rule_query" | "pinned")));
    if string_only && !v.is_string() && !(k == "query" && name == "semantic" && v.is_null()) {
        return Err(wrap_pair(EsError::unsupported_type_xc(parser, k, v, within)));
    }
    // `Script.parse` (`OBJECT_OR_STRING`): an inline source string is a script too.
    if (name, k) == ("script_score", "script") && !v.is_object() && !v.is_string() {
        return Err(wrap_pair(EsError::unsupported_type_xc(parser, k, v, within)));
    }
    let object_only = matches!((name, k), ("knn", "query_vector_builder") | ("knn", "rescore_vector") | ("percolate", "document") | ("rule" | "rule_query", "match_criteria") | ("rule" | "rule_query" | "pinned", "organic") | ("knn", "filter") if !v.is_array());
    if object_only && !v.is_object() {
        return Err(wrap_pair(EsError::unsupported_type_xc(parser, k, v, within)));
    }
    let array_only = matches!((name, k), ("pinned", "ids" | "docs") | ("knn", "query_vector") | ("combined_fields", "fields") | ("rule" | "rule_query", "ruleset_ids") | ("ids", "values") | ("percolate", "documents"));
    if array_only && !v.is_array() && !(k == "query_vector" && v.is_string()) && !(k == "values" && v.is_string()) && !(k == "ruleset_ids" && v.is_string()) {
        return Err(wrap_pair(EsError::unsupported_type_xc(parser, k, v, within)));
    }
    if (name, k) == ("ids", "values") {
        if let Value::Array(a) = v {
            for (i, x) in a.iter().enumerate() {
                let cause = match x {
                    Value::Object(_) | Value::Null => EsError::illegal_argument(format!("Expected text at @POS@ but found {}", token_name(x))).at_key("values", within).elem(i),
                    Value::Array(_) => EsError::new(400, "illegal_state_exception", "expected value but got [START_ARRAY]").at_key("values", within).elem(i),
                    _ => continue,
                };
                return Err(wrap_pair(EsError::field_failure(parser, "values", within, cause)));
            }
        }
    }
    if (name, k) == ("pinned", "docs") {
        if let Value::Array(a) = v {
            for d in a {
                if let Some(o) = d.as_object() {
                    for dk in o.keys() {
                        if !matches!(dk.as_str(), "_index" | "_id") {
                            return Err(wrap_pair(EsError::field_failure(parser, "docs", within, EsError::unknown_field_xc("specified_document", dk, Some("docs"), &["_index", "_id"]))));
                        }
                    }
                }
            }
        }
    }
    if (name, k) == ("sparse_vector", "pruning_config") {
        if let Some(o) = v.as_object() {
            for pk in o.keys() {
                if !matches!(pk.as_str(), "tokens_freq_ratio_threshold" | "tokens_weight_threshold" | "only_score_pruned_tokens") {
                    return Err(wrap_pair(EsError::field_failure(parser, "pruning_config", within, EsError::parsing_at_key(format!("[pruning_config] unknown token [{pk}]"), pk, Some("pruning_config")))));
                }
            }
        }
    }
    if name == "rank_feature" && matches!(k, "saturation" | "log" | "sigmoid" | "linear") {
        if let Some(o) = v.as_object() {
            let inner: &[&str] = match k {
                "saturation" => &["pivot"],
                "log" => &["scaling_factor"],
                "sigmoid" => &["pivot", "exponent"],
                _ => &[],
            };
            for ik in o.keys() {
                if !inner.contains(&ik.as_str()) {
                    return Err(EsError::field_failure("feature", k, within, EsError::unknown_field_xc(k, ik, Some(k), inner)));
                }
            }
        }
    }
    Ok(())
}

/// `Failed to build [parser] after last required field arrived` over `cause` (ObjectParser build failure).
fn build_failure(pair: bool, parser: &str, name: &str, cause: EsError) -> EsError {
    let e = EsError::x_content_parse(format!("Failed to build [{parser}] after last required field arrived")).caused_by(cause);
    if pair {
        EsError::parsing(format!("Failed to build [{parser}] after last required field arrived")).at_key(name, None).cursor("end").with_lc().caused_by(e)
    } else {
        e
    }
}

/// `Required [f]` for an ObjectParser (bare iae, or the pair form).
fn required(pair: bool, name: &str, field: &str) -> EsError {
    let e = EsError::illegal_argument(format!("Required [{field}]"));
    if pair {
        EsError::parsing(format!("Required [{field}]")).at_key(name, None).cursor("end").with_lc().caused_by(e)
    } else {
        e
    }
}

/// `[name] query doesn't support multiple fields, found [a] and [b]` at the second field's value.
fn multiple_fields(name: &str, first: &str, second: &str) -> EsError {
    EsError::parsing_at_key(format!("[{name}] query doesn't support multiple fields, found [{first}] and [{second}]"), second, Some(name))
}

fn obj<'v>(name: &str, v: &'v Value) -> Result<&'v Map<String, Value>> {
    v.as_object().ok_or_else(|| EsError::parsing_at_key(format!("[{name}] query malformed, no start_object after query name"), name, None))
}

/// A field-keyed query (`{"match": {"<field>": {...}}}`): exactly one key,
/// whose object value may only hold `inner`.
fn field_keyed(name: &str, o: &Map<String, Value>, inner: &[&str], style: Style, inner_style: Style) -> Result<()> {
    let mut it = o.iter();
    let Some((field, spec)) = it.next() else {
        return Err(match name {
            "match" => at_end(name, "No text specified for text query"),
            "match_phrase" | "match_phrase_prefix" | "match_bool_prefix" => EsError::illegal_argument(format!("[{name}] requires query value")),
            _ => EsError::illegal_argument("field name is null or empty"),
        });
    };
    if let Some((second, _)) = it.next() {
        return Err(multiple_fields(name, field, second));
    }
    let _ = style;
    match spec {
        Value::Object(so) => {
            for (k, v) in so {
                if !inner.contains(&k.as_str()) && !COMMON.contains(&k.as_str()) {
                    return Err(unknown(inner_style, name, name, k, v, inner));
                }
                // `match` only parses `fuzzy_rewrite` when it builds a fuzzy query.
                if name == "match" && k == "fuzzy_rewrite" && so.get("fuzziness").is_none_or(Value::is_null) {
                    continue;
                }
                typed(name, k, v, inner_style)?;
            }
            let has = |keys: &[&str]| keys.iter().any(|k| so.get(*k).is_some_and(|v| !v.is_null()));
            match name {
                "match" if !has(&["query"]) => return Err(at_end(name, "No text specified for text query")),
                "match_phrase" | "match_phrase_prefix" | "match_bool_prefix" if !has(&["query"]) => return Err(EsError::illegal_argument(format!("[{name}] requires query value"))),
                "term" | "span_term" | "wildcard" | "regexp" | "prefix" if !has(&["value", "term", "wildcard", "prefix"]) => return Err(EsError::illegal_argument("value cannot be null")),
                "fuzzy" if !has(&["value", "term"]) => return Err(EsError::illegal_argument("query value cannot be null")),
                _ => {}
            }
            if name == "match" {
                if let Some(q) = so.get("query") {
                    if q.is_null() {
                        return Err(EsError::parsing_at_key("[match] unknown token [VALUE_NULL] after [query]", "query", Some(name)));
                    }
                }
            }
        }
        Value::Array(_) if matches!(name, "match" | "match_phrase" | "match_phrase_prefix" | "match_bool_prefix" | "wildcard" | "regexp" | "prefix") => {
            return Err(EsError::illegal_argument(format!("Expected text at @POS@ but found START_ARRAY")).at_key(field, Some(name)));
        }
        Value::Array(_) if name == "term" => return Err(EsError::parsing_at_key("[term] query does not support array of values", field, Some(name))),
        Value::Null if name == "match" => return Err(at_end(name, "No text specified for text query")),
        Value::Null if matches!(name, "term" | "range" | "prefix" | "wildcard" | "regexp" | "fuzzy" | "span_term") => return Err(EsError::illegal_argument("field name is null or empty")),
        _ => {}
    }
    Ok(())
}

/// ObjectParser-style type check for a declared field.
fn object_parser_type(pair: bool, parser: &str, field: &str, v: &Value, ok: impl Fn(&Value) -> bool) -> Result<()> {
    if ok(v) {
        return Ok(());
    }
    let e = EsError::unsupported_type_xc(parser, field, v, Some(parser));
    if pair {
        let msg = e.reason.clone();
        Err(EsError::parsing(msg).at_key(field, Some(parser)).prefixed("cursor").with_lc().caused_by(e))
    } else {
        Err(e)
    }
}

/// Validate the top-level structure of query `name` with body `body`.
pub fn check(name: &str, body: &Value) -> Result<()> {
    use Style::*;
    match name {
        "match_all" => {
            let o = obj(name, body)?;
            check_keys(ObjectPair, name, name, o, &[])?;
            if let Some(v) = o.get("_name") {
                object_parser_type(true, name, "_name", v, |v| v.is_string())?;
            }
            if let Some(v) = o.get("boost") {
                object_parser_type(true, name, "boost", v, |v| v.is_number() || v.is_string())?;
                if let Some(s) = v.as_str() {
                    if s.trim().parse::<f64>().is_err() {
                        let w = EsError::x_content_parse(format!("[{name}] failed to parse field [boost]")).at_key("boost", Some(name)).prefixed("cursor").caused_by(EsError::number_format(s));
                        return Err(EsError::parsing(format!("[{name}] failed to parse field [boost]")).at_key("boost", Some(name)).prefixed("cursor").with_lc().caused_by(w));
                    }
                }
            }
            Ok(())
        }
        "match_none" => check_keys(DoesNotSupport, name, name, obj(name, body)?, &[]),
        "ids" => {
            let o = obj(name, body)?;
            check_keys(ObjectPair, name, name, o, &["values"])?;
            if let Some(v) = o.get("values") {
                object_parser_type(true, name, "values", v, |v| v.is_array() || v.is_string())?;
            }
            Ok(())
        }
        "match" => field_keyed(name, obj(name, body)?, &["query", "analyzer", "operator", "fuzziness", "prefix_length", "max_expansions", "fuzzy_transpositions", "fuzzy_rewrite", "lenient", "zero_terms_query", "minimum_should_match", "auto_generate_synonyms_phrase_query"], DoesNotSupportToken, DoesNotSupportToken),
        "match_phrase" => field_keyed(name, obj(name, body)?, &["query", "analyzer", "slop", "zero_terms_query"], DoesNotSupportToken, DoesNotSupportToken),
        "match_phrase_prefix" => field_keyed(name, obj(name, body)?, &["query", "analyzer", "slop", "max_expansions", "zero_terms_query"], DoesNotSupportToken, DoesNotSupportToken),
        "match_bool_prefix" => field_keyed(name, obj(name, body)?, &["query", "analyzer", "operator", "minimum_should_match", "fuzziness", "prefix_length", "max_expansions", "fuzzy_transpositions", "fuzzy_rewrite"], DoesNotSupportToken, DoesNotSupportToken),
        "term" => {
            let o = obj(name, body)?;
            if let Some((_, Value::Object(so))) = o.iter().next() {
                if so.get("value").is_some_and(Value::is_array) {
                    return Err(EsError::parsing_at_key("[term] query does not support arrays for value - use a bool query with multiple term clauses in the should section or use a Terms query if scoring is not required", "value", Some(name)));
                }
            }
            field_keyed(name, o, &["value", "term", "case_insensitive"], DoesNotSupport, DoesNotSupport)
        }
        "span_term" => field_keyed(name, obj(name, body)?, &["value", "term"], DoesNotSupport, DoesNotSupport),
        "range" => {
            let o = obj(name, body)?;
            let mut field: Option<&str> = None;
            for (k, v) in o {
                if COMMON.contains(&k.as_str()) {
                    typed(name, k, v, DoesNotSupport)?;
                    continue;
                }
                if !v.is_object() {
                    if v.is_null() || v.is_array() {
                        return Err(EsError::illegal_argument("field name is null or empty"));
                    }
                    return Err(EsError::query_unsupported(name, k));
                }
                if let Some(f) = field {
                    return Err(multiple_fields(name, f, k));
                }
                field = Some(k);
                for (ik, iv) in v.as_object().unwrap() {
                    // ES 9 removed `from` / `to` / `include_lower` / `include_upper`
                    // (in every REST API version).
                    if !matches!(ik.as_str(), "gt" | "gte" | "lt" | "lte" | "time_zone" | "format" | "relation" | "boost" | "_name") {
                        return Err(EsError::query_unsupported(name, ik));
                    }
                    typed(name, ik, iv, DoesNotSupport)?;
                }
            }
            if field.is_none() {
                return Err(EsError::illegal_argument("field name is null or empty"));
            }
            Ok(())
        }
        "prefix" => field_keyed(name, obj(name, body)?, &["value", "prefix", "rewrite", "case_insensitive"], DoesNotSupport, DoesNotSupport),
        "wildcard" => field_keyed(name, obj(name, body)?, &["value", "wildcard", "rewrite", "case_insensitive"], DoesNotSupport, DoesNotSupport),
        "regexp" => field_keyed(name, obj(name, body)?, &["value", "flags", "flags_value", "max_determinized_states", "rewrite", "case_insensitive"], DoesNotSupport, DoesNotSupport),
        "fuzzy" => field_keyed(name, obj(name, body)?, &["value", "term", "fuzziness", "prefix_length", "max_expansions", "transpositions", "rewrite"], DoesNotSupport, DoesNotSupport),
        "terms" => {
            let o = obj(name, body)?;
            if o.is_empty() {
                return Err(at_end(name, "[terms] query requires a field name, followed by array of terms or a document lookup specification"));
            }
            let mut field: Option<&str> = None;
            for (k, v) in o {
                if COMMON.contains(&k.as_str()) {
                    typed(name, k, v, DoesNotSupport)?;
                    continue;
                }
                if field.is_some() {
                    return Err(if v.is_array() || v.is_object() {
                        EsError::parsing_at_key("[terms] query does not support multiple fields", k, Some(name))
                    } else {
                        EsError::query_unsupported(name, k)
                    });
                }
                match v {
                    Value::Array(_) => {}
                    Value::Object(lookup) => {
                        for (lk, _) in lookup {
                            if !matches!(lk.as_str(), "index" | "id" | "path" | "routing") {
                                return Err(EsError::unknown_field_xc("terms_lookup", lk, Some(name), &["index", "id", "path", "routing"]));
                            }
                        }
                        for req in ["index", "id", "path"] {
                            if !lookup.contains_key(req) {
                                return Err(EsError::illegal_argument(format!("Required [{req}]")));
                            }
                        }
                    }
                    _ => return Err(EsError::query_unsupported(name, k)),
                }
                field = Some(k);
            }
            Ok(())
        }
        "terms_set" => {
            let o = obj(name, body)?;
            let mut it = o.iter().filter(|(k, _)| !COMMON.contains(&k.as_str()));
            if let Some((_, spec)) = it.next() {
                if let Some((second, _)) = it.next() {
                    return Err(EsError::parsing("[terms_set] unknown token [FIELD_NAME]").at_key(second, Some(name)).cursor("key").with_lc());
                }
                if let Value::Object(so) = spec {
                    let mut msm: Option<&str> = None;
                    for (k, v) in so {
                        if !matches!(k.as_str(), "terms" | "minimum_should_match_field" | "minimum_should_match_script" | "minimum_should_match" | "boost" | "_name") {
                            return Err(EsError::query_unsupported(name, k));
                        }
                        if k == "terms" && !v.is_array() {
                            return Err(EsError::query_unsupported(name, k));
                        }
                        if matches!(k.as_str(), "minimum_should_match_field" | "minimum_should_match_script") {
                            if msm.is_some() {
                                return Err(EsError::query_unsupported(name, k));
                            }
                            msm = Some(k);
                        }
                        typed(name, k, v, DoesNotSupport)?;
                    }
                }
            }
            Ok(())
        }
        "exists" => {
            let o = obj(name, body)?;
            for (k, v) in o {
                if k == "field" {
                    if v.is_object() || v.is_array() || v.is_null() {
                        return Err(EsError::parsing_at_key(format!("[exists] unknown token [{}] after [field]", token_name(v)), k, Some(name)));
                    }
                    continue;
                }
                if !COMMON.contains(&k.as_str()) {
                    return Err(unknown(DoesNotSupportToken, name, name, k, v, &["field"]));
                }
                typed(name, k, v, DoesNotSupport)?;
            }
            Ok(())
        }
        "multi_match" => {
            let o = obj(name, body)?;
            check_keys(DoesNotSupportToken, name, name, o, &["query", "fields", "type", "analyzer", "slop", "fuzziness", "prefix_length", "max_expansions", "operator", "minimum_should_match", "fuzzy_rewrite", "tie_breaker", "lenient", "zero_terms_query", "auto_generate_synonyms_phrase_query", "fuzzy_transpositions"])?;
            if !o.contains_key("query") {
                return Err(at_end(name, "No text specified for multi_match query"));
            }
            Ok(())
        }
        "query_string" => {
            let o = obj(name, body)?;
            for (k, v) in o {
                if k == "fields" && !v.is_array() {
                    return Err(EsError::query_unsupported(name, k));
                }
            }
            check_keys(DoesNotSupportToken, name, name, o, &["query", "fields", "default_field", "default_operator", "analyzer", "quote_analyzer", "allow_leading_wildcard", "max_determinized_states", "enable_position_increments", "escape", "fuzziness", "fuzzy_prefix_length", "fuzzy_max_expansions", "fuzzy_rewrite", "phrase_slop", "type", "tie_breaker", "rewrite", "minimum_should_match", "quote_field_suffix", "lenient", "analyze_wildcard", "time_zone", "fuzzy_transpositions", "auto_generate_synonyms_phrase_query"])?;
            if !o.contains_key("query") {
                return Err(at_end(name, "[query_string] must be provided with a [query]"));
            }
            Ok(())
        }
        "simple_query_string" => {
            let o = obj(name, body)?;
            check_keys(Unsupported, name, name, o, &["query", "fields", "default_operator", "analyzer", "flags", "lenient", "analyze_wildcard", "quote_field_suffix", "minimum_should_match", "auto_generate_synonyms_phrase_query", "fuzzy_prefix_length", "fuzzy_max_expansions", "fuzzy_transpositions"])?;
            if !o.contains_key("query") {
                return Err(at_end(name, "[simple_query_string] query text missing"));
            }
            Ok(())
        }
        "combined_fields" => {
            let o = obj(name, body)?;
            let allowed = ["query", "fields", "operator", "minimum_should_match", "auto_generate_synonyms_phrase_query", "zero_terms_query"];
            check_keys(ObjectBare, name, name, o, &allowed)?;
            if let Some(Value::String(op)) = o.get("operator") {
                if !matches!(upper(op).as_str(), "AND" | "OR") {
                    return Err(EsError::field_failure(name, "operator", Some(name), EsError::illegal_argument(format!("No enum constant org.elasticsearch.index.query.Operator.{}", upper(op)))));
                }
            }
            if o.get("fields").and_then(Value::as_array).map_or(true, |a| a.is_empty()) {
                return Err(EsError::search_phase("", EsError::query_shard("", "", "failed to create query: In [combined_fields] query, at least one field must be provided")));
            }
            Ok(())
        }
        "bool" => {
            let o = obj(name, body)?;
            let allowed = ["must", "filter", "must_not", "should", "minimum_should_match", "adjust_pure_negative"];
            check_keys(ObjectBare, name, name, o, &allowed)?;
            for c in ["must", "filter", "must_not", "should"] {
                if let Some(v) = o.get(c) {
                    object_parser_type(false, name, c, v, |v| v.is_object() || v.is_array() || v.is_null())?;
                }
            }
            if let Some(v) = o.get("minimum_should_match") {
                object_parser_type(false, name, "minimum_should_match", v, |v| v.is_string() || v.is_number() || v.is_boolean())?;
            }
            if let Some(v) = o.get("_name") {
                object_parser_type(false, name, "_name", v, |v| v.is_string())?;
            }
            if let Some(v) = o.get("boost") {
                object_parser_type(false, name, "boost", v, |v| v.is_number() || v.is_string())?;
                if let Some(s) = v.as_str() {
                    if s.trim().parse::<f64>().is_err() {
                        return Err(EsError::field_failure(name, "boost", Some(name), EsError::number_format(s)));
                    }
                }
            }
            if let Some(v) = o.get("adjust_pure_negative") {
                if let Some(s) = v.as_str() {
                    if !matches!(s, "true" | "false") {
                        return Err(EsError::field_failure(name, "adjust_pure_negative", Some(name), EsError::illegal_argument(format!("Failed to parse value [{s}] as only [true] or [false] are allowed."))));
                    }
                }
            }
            Ok(())
        }
        "boosting" => check_keys(DoesNotSupport, name, name, obj(name, body)?, &["positive", "negative", "negative_boost"]),
        "constant_score" => {
            let o = obj(name, body)?;
            for (k, v) in o {
                if k == "filter" {
                    if v.is_array() {
                        return Err(EsError::parsing_at_key("unexpected token [START_ARRAY]", k, Some(name)));
                    }
                    if !v.is_object() {
                        return Err(EsError::query_unsupported(name, k));
                    }
                } else if !COMMON.contains(&k.as_str()) {
                    if v.is_array() {
                        return Err(EsError::parsing_at_key("unexpected token [START_ARRAY]", k, Some(name)));
                    }
                    return Err(EsError::query_unsupported(name, k));
                } else {
                    typed(name, k, v, DoesNotSupport)?;
                }
            }
            if !o.contains_key("filter") {
                return Err(at_end(name, "[constant_score] requires a 'filter' element"));
            }
            Ok(())
        }
        "dis_max" => {
            let o = obj(name, body)?;
            check_keys(DoesNotSupport, name, name, o, &["queries", "tie_breaker"])?;
            match o.get("queries") {
                Some(Value::Array(a)) if a.is_empty() => Err(EsError::parsing("[_na] query malformed, must start with start_object").at_key("queries", Some(name)).cursor("end").with_lc()),
                Some(Value::Object(m)) if m.is_empty() => Err(EsError::illegal_argument("query malformed, empty clause found at [@POS@]").at_key("queries", Some(name)).cursor("end")),
                Some(Value::Array(a)) => {
                    if let Some((i, _)) = a.iter().enumerate().find(|(_, q)| !q.is_object()) {
                        return Err(EsError::parsing("[_na] query malformed, must start with start_object").at_key("queries", Some(name)).elem(i).cursor("after").with_lc());
                    }
                    Ok(())
                }
                _ => Ok(()),
            }
        }
        "function_score" => {
            let o = obj(name, body)?;
            let allowed = ["query", "functions", "score_mode", "boost_mode", "max_boost", "min_score", "weight", "filter"];
            for (k, v) in o {
                if allowed.contains(&k.as_str()) || COMMON.contains(&k.as_str()) || SCORE_FUNCTIONS.contains(&k.as_str()) {
                    if k == "query" && !v.is_object() {
                        return Err(EsError::parsing_at_key("failed to parse [function_score] query. field [query] is not supported", k, Some(name)));
                    }
                    continue;
                }
                if v.is_object() {
                    return Err(EsError::unknown_named("unknown query", name, None, SCORE_FUNCTIONS).map_key(k, Some(name)));
                }
                return Err(EsError::parsing_at_key(format!("failed to parse [function_score] query. field [{k}] is not supported"), k, Some(name)));
            }
            let single_fn = o.keys().find(|k| k.as_str() == "weight" || SCORE_FUNCTIONS.contains(&k.as_str()));
            if let (Some(sf), true) = (single_fn, o.contains_key("functions")) {
                return Err(EsError::parsing_at_key(format!("failed to parse [function_score] query. [you can either define [functions] array or a single function, not both. already found [{sf}], now encountering [functions].]"), "functions", Some(name)));
            }
            for (k, v) in o {
                if matches!(k.as_str(), "score_mode" | "boost_mode" | "max_boost" | "min_score" | "boost" | "weight") {
                    typed(name, k, v, DoesNotSupport)?;
                }
            }
            if let Some(Value::Array(fns)) = o.get("functions") {
                for (i, f) in fns.iter().enumerate() {
                    let Some(fo) = f.as_object() else {
                        return Err(EsError::parsing(format!("failed to parse [START_OBJECT]. malformed query, expected a [{}] while parsing functions but got a [function_score] instead", token_name(f))).at_key("functions", Some(name)).elem(i).cursor("after").with_lc());
                    };
                    let mut seen: Option<&str> = None;
                    for (k, v) in fo {
                        if !matches!(k.as_str(), "filter" | "weight" | "_name") && !SCORE_FUNCTIONS.contains(&k.as_str()) {
                            if v.is_object() {
                                return Err(EsError::unknown_named("unknown query", name, None, SCORE_FUNCTIONS).map_key(k, Some("functions")));
                            }
                            return Err(EsError::parsing_at_key(format!("failed to parse [function_score] query. field [{k}] is not supported"), k, Some("functions")));
                        }
                        if SCORE_FUNCTIONS.contains(&k.as_str()) {
                            if let Some(prev) = seen {
                                return Err(EsError::parsing_at_key(format!("failed to parse function_score functions. already found [{prev}], now encountering [{k}]."), k, Some("functions")));
                            }
                            seen = Some(k);
                        }
                        if k == "weight" {
                            typed(name, k, v, DoesNotSupport)?;
                        }
                        if matches!(k.as_str(), "gauss" | "linear" | "exp") {
                            if let Some(decay) = v.as_object() {
                                for (_, fv) in decay {
                                    if let Some(fd) = fv.as_object() {
                                        for pk in fd.keys() {
                                            if !matches!(pk.as_str(), "origin" | "scale" | "offset" | "decay") {
                                                return Err(EsError::search_phase("", EsError::new(400, "parse_exception", format!("parameter [{pk}] not supported!"))));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    for fname in SCORE_FUNCTIONS {
                        if let Some(Value::Object(fb)) = fo.get(*fname) {
                            let inner: &[&str] = match *fname {
                                "random_score" => &["seed", "field"],
                                "field_value_factor" => &["field", "factor", "modifier", "missing"],
                                "script_score" => &["script"],
                                _ => continue,
                            };
                            for (k, _) in fb {
                                if !inner.contains(&k.as_str()) {
                                    return Err(EsError::parsing_at_key(format!("{fname} query does not support [{k}]"), k, Some(fname)));
                                }
                            }
                        }
                    }
                }
            } else if let Some(v) = o.get("functions") {
                if v.is_object() {
                    return Err(EsError::unknown_named("unknown query", name, None, SCORE_FUNCTIONS).map_key("functions", Some(name)));
                }
            }
            Ok(())
        }
        "nested" => {
            let o = obj(name, body)?;
            check_keys(DoesNotSupport, name, name, o, &["path", "query", "score_mode", "ignore_unmapped", "inner_hits"])?;
            if !o.contains_key("path") {
                return Err(EsError::illegal_argument("[nested] requires 'path' field"));
            }
            if !o.contains_key("query") {
                return Err(EsError::illegal_argument("[nested] requires 'query' field"));
            }
            if let Some(Value::Object(ih)) = o.get("inner_hits") {
                crate::search_strict::check_inner_hits(ih)?;
            }
            Ok(())
        }
        "has_child" => {
            let o = obj(name, body)?;
            check_keys(DoesNotSupport, name, name, o, &["type", "query", "score_mode", "min_children", "max_children", "ignore_unmapped", "inner_hits"])?;
            if !o.contains_key("type") {
                return Err(EsError::illegal_argument("[has_child] requires 'type' field"));
            }
            if !o.contains_key("query") {
                return Err(EsError::illegal_argument("[has_child] requires 'query' field"));
            }
            Ok(())
        }
        "has_parent" => {
            let o = obj(name, body)?;
            check_keys(DoesNotSupport, name, name, o, &["parent_type", "query", "score", "ignore_unmapped", "inner_hits"])?;
            if !o.contains_key("parent_type") {
                return Err(EsError::illegal_argument("[has_parent] requires 'parent_type' field"));
            }
            if !o.contains_key("query") {
                return Err(EsError::illegal_argument("[has_parent] requires 'query' field"));
            }
            Ok(())
        }
        "parent_id" => {
            let o = obj(name, body)?;
            check_keys(DoesNotSupport, name, name, o, &["type", "id", "ignore_unmapped"])?;
            if !o.contains_key("type") {
                return Err(EsError::illegal_argument("[parent_id] requires 'type' field"));
            }
            if !o.contains_key("id") {
                return Err(EsError::illegal_argument("[parent_id] requires 'id' field"));
            }
            Ok(())
        }
        "script" => {
            let o = obj(name, body)?;
            check_keys(DoesNotSupport, name, name, o, &["script"])?;
            // Internal slots have request-scoped originals; validate that
            // original's metadata, never accept an arbitrary numeric script.
            let resolved = o.get("script").map(crate::script::search_script_source).transpose()?;
            match resolved.as_ref().map(|(metadata, _)| metadata) {
                None => return Err(at_end(name, "script must be provided with a [script] filter")),
                Some(Value::Object(so)) => {
                    for sk in so.keys() {
                        if !matches!(sk.as_str(), "source" | "id" | "lang" | "params" | "options" | "inline") {
                            return Err(EsError::unknown_field_xc("script", sk, Some("script"), &["source", "id", "lang", "params", "options"]));
                        }
                    }
                }
                Some(Value::String(_)) => {}
                Some(_) => return Err(EsError::x_content_parse("[script] Expected START_OBJECT but was: END_OBJECT").at_key("script", Some(name)).cursor("after").prefixed("cursor")),
            }
            Ok(())
        }
        "script_score" => {
            let o = obj(name, body)?;
            check_keys(ObjectBare, name, name, o, &["query", "script", "min_score"])?;
            if !o.contains_key("script") {
                return Err(required(false, name, "script"));
            }
            Ok(())
        }
        "wrapper" => {
            let o = obj(name, body)?;
            if let Some((second, _)) = o.iter().filter(|(k, _)| k.as_str() != "query").next() {
                if o.contains_key("query") {
                    return Err(EsError::parsing("[wrapper] malformed query, expected [END_OBJECT] but found [FIELD_NAME]").at_key(second, Some(name)).cursor("key").with_lc());
                }
            }
            Ok(())
        }
        "more_like_this" | "mlt" => {
            let o = obj(name, body)?;
            check_keys(DoesNotSupport, "mlt", "mlt", o, &["fields", "like", "unlike", "like_text", "ids", "docs", "max_query_terms", "min_term_freq", "min_doc_freq", "max_doc_freq", "min_word_length", "max_word_length", "stop_words", "analyzer", "minimum_should_match", "boost_terms", "include", "fail_on_unsupported_field"])?;
            if let Some(Value::Number(_)) = o.get("fields") {
                return Err(EsError::query_unsupported("mlt", "fields"));
            }
            for key in ["like", "unlike", "docs"] {
                let items: Vec<&Value> = match o.get(key) {
                    Some(Value::Array(a)) => a.iter().collect(),
                    Some(v) => vec![v],
                    None => vec![],
                };
                for it in items {
                    if let Some(io) = it.as_object() {
                        for ik in io.keys() {
                            if !matches!(ik.as_str(), "_index" | "_id" | "doc" | "fields" | "per_field_analyzer" | "_routing" | "_version" | "_version_type" | "routing" | "version" | "version_type") {
                                return Err(EsError::new(400, "parse_exception", format!("failed to parse More Like This item. unknown field [{ik}]")));
                            }
                        }
                    }
                }
            }
            if !o.contains_key("like") && !o.contains_key("like_text") && !o.contains_key("ids") && !o.contains_key("docs") {
                return Err(at_end("more_like_this", "more_like_this requires 'like' to be specified"));
            }
            Ok(())
        }
        "span_near" | "span_or" => {
            let o = obj(name, body)?;
            let allowed: &[&str] = if name == "span_near" { &["clauses", "slop", "in_order", "collect_payloads"] } else { &["clauses"] };
            for (k, v) in o {
                if !allowed.contains(&k.as_str()) && !COMMON.contains(&k.as_str()) {
                    return Err(EsError::query_unsupported(name, k));
                }
                if k == "clauses" {
                    let Value::Array(a) = v else { return Err(EsError::query_unsupported(name, k)) };
                    for c in a {
                        if let Some((cn, _)) = c.as_object().and_then(|m| m.iter().next()) {
                            if !cn.starts_with("span_") && cn != "field_masking_span" {
                                return Err(at_end(name, format!("{name} [clauses] must be of type span query")));
                            }
                        }
                    }
                } else {
                    typed(name, k, v, DoesNotSupport)?;
                }
            }
            if !o.contains_key("clauses") {
                return Err(at_end(name, format!("{name} must include [clauses]")));
            }
            Ok(())
        }
        "span_not" => {
            let o = obj(name, body)?;
            check_keys(DoesNotSupport, name, name, o, &["include", "exclude", "dist", "pre", "post"])?;
            if !o.contains_key("include") {
                return Err(at_end(name, "span_not must have [include] span query clause"));
            }
            if !o.contains_key("exclude") {
                return Err(at_end(name, "span_not must have [exclude] span query clause"));
            }
            if o.contains_key("dist") && (o.contains_key("pre") || o.contains_key("post")) {
                return Err(at_end(name, "span_not can either use [dist] or [pre] & [post] (or none)"));
            }
            Ok(())
        }
        "span_first" => {
            let o = obj(name, body)?;
            check_keys(DoesNotSupport, name, name, o, &["match", "end"])?;
            if !o.contains_key("match") {
                return Err(at_end(name, "span_first must have [match] span query clause"));
            }
            if !o.contains_key("end") {
                return Err(at_end(name, "span_first must have [end] set for it"));
            }
            Ok(())
        }
        "span_multi" => {
            let o = obj(name, body)?;
            check_keys(DoesNotSupport, name, name, o, &["match"])?;
            match o.get("match") {
                None => return Err(at_end(name, "[span_multi] must have [match] multi term query clause")),
                Some(m) => {
                    if let Some((mn, _)) = m.as_object().and_then(|x| x.iter().next()) {
                        if !matches!(mn.as_str(), "prefix" | "wildcard" | "regexp" | "fuzzy" | "range") {
                            return Err(at_end(name, "[span_multi] [match] must be of type multi term query"));
                        }
                    }
                }
            }
            Ok(())
        }
        "span_containing" | "span_within" => check_keys(DoesNotSupport, name, name, obj(name, body)?, &["little", "big"]),
        "span_field_masking" | "field_masking_span" => check_keys(DoesNotSupport, name, name, obj(name, body)?, &["query", "field"]),
        "geo_distance" => {
            let o = obj(name, body)?;
            let mut field: Option<&str> = None;
            for (k, v) in o {
                if matches!(k.as_str(), "distance" | "unit" | "distance_type" | "validation_method" | "ignore_unmapped" | "boost" | "_name") {
                    if k == "distance" {
                        if let Value::String(d) = v {
                            if crate::values::parse_distance(&Value::String(d.clone())).is_none() {
                                // ES `DistanceUnit.parse`: strip a known unit suffix, `Double.parseDouble` the rest.
                                return Err(super::geo::parse_distance_es(d).err().unwrap_or_else(|| EsError::number_format(d)));
                            }
                        }
                    }
                    typed(name, k, v, DoesNotSupport)?;
                    continue;
                }
                if v.is_object() || v.is_array() || v.is_string() {
                    if let Some(f) = field {
                        return Err(multiple_fields(name, f, k));
                    }
                    field = Some(k);
                    continue;
                }
                return Err(EsError::parsing_at_key(format!("failed to parse [geo_distance] query. unexpected field [{k}]"), k, Some(name)));
            }
            if !o.contains_key("distance") {
                return Err(at_end(name, "geo_distance requires 'distance' to be specified"));
            }
            if field.is_none() {
                return Err(at_end(name, "geo_distance requires 'field' to be specified"));
            }
            Ok(())
        }
        "geo_bounding_box" => {
            let o = obj(name, body)?;
            let mut field = false;
            for (k, v) in o {
                if matches!(k.as_str(), "validation_method" | "ignore_unmapped" | "boost" | "_name") {
                    typed(name, k, v, DoesNotSupport)?;
                    continue;
                }
                if let Some(bo) = v.as_object() {
                    field = true;
                    for ck in bo.keys() {
                        if !matches!(ck.as_str(), "top_left" | "bottom_right" | "top_right" | "bottom_left" | "top" | "left" | "bottom" | "right" | "wkt") {
                            return Err(EsError::new(400, "parse_exception", format!("failed to parse [geo_bounding_box] query. [failed to parse bounding box. unexpected field [{ck}]]")));
                        }
                    }
                    continue;
                }
                return Err(EsError::parsing_at_key(format!("failed to parse [geo_bounding_box] query. unexpected field [{k}]"), k, Some(name)));
            }
            if !field {
                return Err(EsError::new(400, "parse_exception", "failed to parse [geo_bounding_box] query. bounding box not provided"));
            }
            Ok(())
        }
        "geo_grid" => {
            let o = obj(name, body)?;
            for (k, v) in o {
                if COMMON.contains(&k.as_str()) || v.is_object() {
                    continue;
                }
                return Err(EsError::parsing_at_key(format!("failed to parse [geo_grid] query. unexpected field [{k}]"), k, Some(name)));
            }
            Ok(())
        }
        "geo_polygon" => check_keys_fielded(DoesNotSupport, name, obj(name, body)?, &["validation_method", "ignore_unmapped"], &["points"], DoesNotSupport),
        "geo_shape" | "shape" => {
            let o = obj(name, body)?;
            check_keys_fielded(Bare, name, o, &["ignore_unmapped"], &["shape", "indexed_shape", "relation", "strategy"], Bare)?;
            for (_, v) in o.iter().filter(|(_, v)| v.is_object()) {
                let so = v.as_object().unwrap();
                if let Some(Value::String(r)) = so.get("relation") {
                    if !matches!(r.as_str(), "intersects" | "disjoint" | "within" | "contains") {
                        return Err(EsError::parsing_at_key(format!("Unknown shape operation [{r} ]"), "relation", Some(name)));
                    }
                }
                if !so.contains_key("shape") && !so.contains_key("indexed_shape") {
                    return Err(EsError::illegal_argument("either shape or indexedShapeId is required"));
                }
            }
            Ok(())
        }
        "text_expansion" => {
            let o = obj(name, body)?;
            check_keys_fielded(DoesNotSupport, name, o, &[], &["model_id", "model_text", "pruning_config"], DoesNotSupport)?;
            for (_, v) in o.iter().filter(|(_, v)| v.is_object()) {
                let so = v.as_object().unwrap();
                if !so.contains_key("model_id") {
                    return Err(EsError::illegal_argument("[text_expansion] requires a model_id value"));
                }
                if !so.contains_key("model_text") {
                    return Err(EsError::illegal_argument("[text_expansion] requires a model_text value"));
                }
            }
            Ok(())
        }
        "weighted_tokens" => check_keys_fielded(UnknownField, name, obj(name, body)?, &[], &["tokens", "pruning_config"], UnknownField),
        "intervals" => {
            let o = obj(name, body)?;
            let mut it = o.iter();
            if let Some((_, spec)) = it.next() {
                if let Some((second, _)) = it.next() {
                    return Err(EsError::parsing("Expected [END_OBJECT] but got [FIELD_NAME]").at_key(second, Some(name)).cursor("key").with_lc());
                }
                if let Value::Object(so) = spec {
                    let rules: Vec<&String> = so.keys().filter(|k| !COMMON.contains(&k.as_str())).collect();
                    if rules.len() > 1 {
                        return Err(EsError::parsing_at_key(format!("Only one interval rule can be specified, found [{}] and [{}]", rules[0], rules[1]), rules[1], None));
                    }
                    if let Some(r) = rules.first() {
                        if !matches!(r.as_str(), "match" | "any_of" | "all_of" | "prefix" | "wildcard" | "regexp" | "range" | "fuzzy") {
                            return Err(EsError::parsing(format!("Unknown interval type [{r}], expecting one of [match, any_of, all_of, prefix, wildcard, regexp, range]")).at_key(r, None).cursor("key").with_lc());
                        }
                        check_interval_rule(r, &so[r.as_str()])?;
                    }
                }
            }
            Ok(())
        }
        "knn" => {
            let o = obj(name, body)?;
            // `_inner_hits_children`: internal marker on the nested knn query built for top-level knn `inner_hits` (vectors.rs).
            let allowed = ["field", "query_vector", "query_vector_builder", "k", "num_candidates", "visit_percentage", "filter", "similarity", "rescore_vector", "_inner_hits_children"];
            check_keys(ObjectBare, name, name, o, &allowed)?;
            if let Some(Value::String(s)) = o.get("query_vector") {
                if s.len() % 2 == 1 {
                    return Err(EsError::field_failure(name, "query_vector", Some(name), EsError::illegal_argument(format!("string length not even: {}", s.len()))));
                }
            }
            if !o.contains_key("field") {
                return Err(required(false, name, "field"));
            }
            if !o.contains_key("query_vector") && !o.contains_key("query_vector_builder") {
                return Err(build_failure(false, name, name, EsError::illegal_argument("either [query_vector] or [query_vector_builder] must be provided")));
            }
            Ok(())
        }
        "pinned" => {
            let o = obj(name, body)?;
            check_keys(ObjectPair, name, name, o, &["ids", "docs", "organic"])?;
            if !o.contains_key("organic") {
                return Err(required(true, name, "organic"));
            }
            if o.contains_key("ids") && o.contains_key("docs") {
                let inner = build_failure(false, name, name, EsError::illegal_argument("[pinned] ids and docs cannot both be used"));
                let w = EsError::field_failure(name, "organic", Some(name), inner);
                return Err(EsError::parsing("[pinned] failed to parse field [organic]").at_key("organic", Some(name)).prefixed("cursor").with_lc().caused_by(w));
            }
            Ok(())
        }
        "rank_feature" => {
            let o = obj(name, body)?;
            check_keys(ObjectBare, name, "feature", o, &["field", "saturation", "log", "sigmoid", "linear"])?;
            if !o.contains_key("field") {
                return Err(required(false, name, "field"));
            }
            if ["saturation", "log", "sigmoid", "linear"].iter().filter(|k| o.contains_key(**k)).count() > 1 {
                return Err(build_failure(false, "feature", name, EsError::illegal_argument("Can only specify one of [log], [saturation], [sigmoid] and [linear]")));
            }
            Ok(())
        }
        "percolate" => {
            let o = obj(name, body)?;
            check_keys(ObjectBare, name, name, o, &["field", "document", "documents", "index", "id", "routing", "preference", "version", "name"])?;
            if !o.contains_key("field") {
                return Err(required(false, name, "field"));
            }
            Ok(())
        }
        "sparse_vector" => {
            let o = obj(name, body)?;
            check_keys(ObjectPair, name, name, o, &["field", "query_vector", "inference_id", "query", "prune", "pruning_config"])?;
            if !o.contains_key("field") {
                return Err(required(true, name, "field"));
            }
            let qv = o.contains_key("query_vector");
            let inf = o.contains_key("inference_id") || o.contains_key("query");
            if qv == inf {
                return Err(build_failure(true, name, name, EsError::illegal_argument("[sparse_vector] requires one of [query_vector] or [inference_id] for sparse_vector fields")));
            }
            Ok(())
        }
        "semantic" => {
            let o = obj(name, body)?;
            check_keys(ObjectBare, name, name, o, &["field", "query", "lenient"])?;
            for r in ["field", "query"] {
                if !o.contains_key(r) {
                    return Err(required(false, name, r));
                }
            }
            Ok(())
        }
        "rule" | "rule_query" => {
            let o = obj(name, body)?;
            check_keys(ObjectPair, "rule", "rule", o, &["organic", "ruleset_ids", "ruleset_id", "match_criteria"])?;
            if !o.contains_key("organic") {
                return Err(required(true, "rule", "organic"));
            }
            if !o.contains_key("match_criteria") {
                return Err(required(true, "rule", "match_criteria"));
            }
            Ok(())
        }
        "distance_feature" => {
            let o = obj(name, body)?;
            // `pivot` / `origin` are distances, durations, dates or points: not the numeric `pivot` of rank_feature.
            const ALLOWED: &[&str] = &["field", "origin", "pivot"];
            for (k, v) in o {
                if !ALLOWED.contains(&k.as_str()) && !COMMON.contains(&k.as_str()) {
                    return Err(unknown(ObjectBare, name, name, k, v, ALLOWED));
                }
                if !matches!(k.as_str(), "origin" | "pivot") {
                    object_parser_value(false, name, name, k, v)?;
                }
            }
            for r in ["field", "origin", "pivot"] {
                if !o.contains_key(r) {
                    return Err(required(false, name, r));
                }
            }
            Ok(())
        }
        "kql" => check_keys(ObjectBare, name, name, obj(name, body)?, &["query", "case_insensitive", "time_zone", "default_field"]),
        _ => Ok(()),
    }
}

/// Field-keyed queries that also take options next to the field
/// (`geo_polygon`, `geo_shape`, `text_expansion`, `weighted_tokens`):
/// object values are the field, anything else an option.
fn check_keys_fielded(style: Style, name: &str, o: &Map<String, Value>, options: &[&str], inner: &[&str], inner_style: Style) -> Result<()> {
    let mut field: Option<&str> = None;
    for (k, v) in o {
        if options.contains(&k.as_str()) || COMMON.contains(&k.as_str()) {
            continue;
        }
        if let Value::Object(so) = v {
            if let Some(f) = field {
                return Err(multiple_fields(name, f, k));
            }
            field = Some(k);
            for (ik, iv) in so {
                if !inner.contains(&ik.as_str()) && !COMMON.contains(&ik.as_str()) {
                    return Err(unknown(inner_style, name, name, ik, iv, inner));
                }
            }
            continue;
        }
        return Err(unknown(style, name, name, k, v, options));
    }
    Ok(())
}

impl EsError {
    /// Re-point an error (and its cause) built for name `n` at key `k`.
    fn map_key(mut self, k: &str, within: Option<&str>) -> Self {
        self.extra.retain(|(key, _)| key != "@k" && key != "@in");
        self = self.at_key(k, within);
        if let Some(c) = self.caused_by.take() {
            let mut c = *c;
            c.extra.retain(|(key, _)| key != "@k" && key != "@in");
            c.reason = format!("unknown field [{k}]");
            self.caused_by = Some(Box::new(c.at_key(k, within)));
        }
        self
    }

    /// Array element `i` of the hinted key (positions of `[_na]`-style errors).
    pub fn elem(mut self, i: usize) -> Self {
        self.extra.push(("@idx".into(), Value::from(i as u64)));
        self
    }

    /// Add a key hint only when the error has none yet (the enclosing parser
    /// knows the key, the inner one does not).
    pub fn default_key(self, key: &str, within: Option<&str>) -> Self {
        if self.extra.iter().any(|(k, _)| k == "@k") {
            return self;
        }
        self.at_key(key, within)
    }

    /// Wrap a failure inside bool clause `clause` element `i`.
    pub fn in_clause(self, clause: &str, i: Option<usize>) -> Self {
        let na = self.reason.starts_with("[_na]");
        let inner = match i {
            Some(i) if !self.extra.iter().any(|(k, _)| k == "@k") => {
                let e = self.at_key(clause, Some("bool")).elem(i);
                if na { e.cursor("after") } else { e }
            }
            _ => self.default_key(clause, Some("bool")),
        };
        EsError::field_failure("bool", clause, Some("bool"), inner)
    }
}

/// `Unexpected token [END_OBJECT] found after the main object.`-style
/// Jackson token name helper re-exported for parsers.
pub fn jackson_name(v: &Value) -> &'static str {
    jackson_token(v)
}


/// Field checks of one interval rule (`IntervalsSourceProvider` ObjectParsers).
fn check_interval_rule(rule: &str, v: &Value) -> Result<()> {
    let Some(o) = v.as_object() else { return Ok(()) };
    let allowed: &[&str] = match rule {
        "match" => &["query", "max_gaps", "ordered", "analyzer", "filter", "use_field"],
        "prefix" => &["prefix", "analyzer", "use_field"],
        "wildcard" => &["pattern", "analyzer", "use_field"],
        "fuzzy" => &["term", "prefix_length", "transpositions", "fuzziness", "analyzer", "use_field"],
        "range" => &["gte", "gt", "lte", "lt", "analyzer", "use_field"],
        "all_of" => &["intervals", "max_gaps", "ordered", "filter"],
        "any_of" => &["intervals", "filter"],
        _ => return Ok(()),
    };
    for (k, fv) in o {
        if !allowed.contains(&k.as_str()) {
            return Err(EsError::unknown_field_xc(rule, k, Some(rule), allowed));
        }
        if k == "filter" {
            if let Some(fo) = fv.as_object() {
                for fk in fo.keys() {
                    if !matches!(fk.as_str(), "before" | "after" | "contained_by" | "containing" | "not_contained_by" | "not_containing" | "not_overlapping" | "overlapping" | "script") {
                        return Err(EsError::field_failure(rule, "filter", Some(rule), EsError::parsing("Expected [FIELD_NAME] but got [END_OBJECT]").cursor("inherit").with_lc()));
                    }
                }
            }
        }
    }
    Ok(())
}
