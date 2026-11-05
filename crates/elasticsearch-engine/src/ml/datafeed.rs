//! Datafeed configs (`DatafeedConfig`, `DatafeedUpdate`): parsing with ES's
//! validation, defaults and rendering.

use super::xc::{self, bad_request, fmt, unknown, wrap, xparse, Pos};
use crate::error::{EsError, Result};
use serde_json::{json, Map, Value};

const FIELDS: &[&str] = &[
    "datafeed_id", "job_id", "indices", "indexes", "query", "query_delay", "frequency", "scroll_size", "aggregations", "aggs", "script_fields", "runtime_mappings", "chunking_config",
    "delayed_data_check_config", "max_empty_searches", "indices_options",
];

/// Java `String.hashCode`.
fn java_hash(s: &str) -> i32 {
    s.encode_utf16().fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(c as i32))
}

/// `new java.util.Random(seed).longs(origin, bound).findFirst()`
/// (`RandomSupport.boundedNextLong`).
fn java_bounded_next_long(seed: i64, origin: i64, bound: i64) -> i64 {
    const MUL: i64 = 0x5DEECE66D;
    const MASK: i64 = (1 << 48) - 1;
    let mut s = (seed ^ MUL) & MASK;
    let mut next = |bits: u32| -> i32 {
        s = (s.wrapping_mul(MUL).wrapping_add(0xB)) & MASK;
        (s >> (48 - bits)) as i32
    };
    let mut next_long = || -> i64 { ((next(32) as i64) << 32).wrapping_add(next(32) as i64) };
    let n = bound - origin;
    let m = n - 1;
    let r = next_long();
    if n & m == 0 {
        return (r & m) + origin;
    }
    let mut u = ((r as u64) >> 1) as i64;
    loop {
        let r = u % n;
        if u.wrapping_add(m).wrapping_sub(r) >= 0 {
            return r + origin;
        }
        u = ((next_long() as u64) >> 1) as i64;
    }
}

/// `DatafeedConfig.defaultRandomQueryDelay(jobId)`: 60s..120s seeded by the
/// job id's `hashCode`.
pub fn default_query_delay(job_id: &str) -> String {
    format!("{}ms", java_bounded_next_long(java_hash(job_id) as i64, 60_000, 120_000))
}

fn indices_options_default() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("ignore_unavailable".into(), json!(false));
    m.insert("expand_wildcards".into(), json!(["open"]));
    m.insert("allow_no_indices".into(), json!(true));
    m.insert("ignore_throttled".into(), json!(true));
    m
}

/// Merge `indices_options` from the body / URL params into `base`.
fn merge_indices_options(base: &mut Map<String, Value>, o: &Map<String, Value>) -> Result<()> {
    for (k, v) in o {
        match k.as_str() {
            "ignore_unavailable" | "allow_no_indices" | "ignore_throttled" => {
                let b = match v {
                    Value::Bool(b) => *b,
                    Value::String(s) if s == "true" || s == "false" => s == "true",
                    _ => return Err(EsError::illegal_argument(format!("Failed to parse value [{v}] as only [true] or [false] are allowed."))),
                };
                base.insert(k.clone(), json!(b));
            }
            "expand_wildcards" => {
                let list: Vec<String> = match v {
                    Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
                    Value::String(s) => s.split(',').map(|x| x.trim().to_string()).collect(),
                    _ => vec![],
                };
                let mut out: Vec<&str> = Vec::new();
                for w in &list {
                    match w.as_str() {
                        "all" => out.extend(["open", "closed", "hidden"]),
                        "open" | "closed" | "hidden" => out.push(w.as_str()),
                        "none" => {}
                        other => return Err(EsError::illegal_argument(format!("No enum constant org.elasticsearch.action.support.IndicesOptions.WildcardStates.{}", other.to_ascii_uppercase()))),
                    }
                }
                let mut ordered: Vec<&str> = ["open", "closed", "hidden"].into_iter().filter(|x| out.contains(x)).collect();
                ordered.dedup();
                base.insert(k.clone(), json!(ordered));
            }
            _ => {}
        }
    }
    Ok(())
}

/// The `indices_options` URL params of put/update datafeed.
pub fn indices_options_from_params(params: &std::collections::HashMap<String, String>) -> Map<String, Value> {
    let mut m = Map::new();
    for k in ["ignore_unavailable", "allow_no_indices", "ignore_throttled", "expand_wildcards"] {
        if let Some(v) = params.get(k) {
            m.insert(k.into(), json!(v));
        }
    }
    m
}

fn normalize_script_fields(v: &Value) -> Value {
    let mut out = Map::new();
    for (k, f) in v.as_object().into_iter().flatten() {
        let script = match &f["script"] {
            Value::String(s) => json!({"source": s, "lang": "painless"}),
            Value::Object(o) => {
                let mut m = Map::new();
                m.insert("source".into(), o.get("source").or_else(|| o.get("inline")).cloned().unwrap_or(json!("")));
                m.insert("lang".into(), o.get("lang").cloned().unwrap_or(json!("painless")));
                if let Some(p) = o.get("params") {
                    m.insert("params".into(), p.clone());
                }
                Value::Object(m)
            }
            other => other.clone(),
        };
        out.insert(k.clone(), json!({"script": script, "ignore_failure": f.get("ignore_failure").cloned().unwrap_or(json!(false))}));
    }
    Value::Object(out)
}

/// Query names of the query DSL (`SearchModule` registrations).
const QUERY_NAMES: &[&str] = &[
    "match_all", "match_none", "match", "match_phrase", "match_phrase_prefix", "match_bool_prefix", "multi_match", "combined_fields", "query_string", "simple_query_string", "term",
    "terms", "terms_set", "range", "exists", "prefix", "wildcard", "regexp", "fuzzy", "ids", "bool", "boosting", "constant_score", "dis_max", "function_score", "script_score", "script",
    "nested", "has_child", "has_parent", "parent_id", "geo_bounding_box", "geo_distance", "geo_polygon", "geo_shape", "geo_grid", "shape", "more_like_this", "percolate", "rank_feature",
    "distance_feature", "wrapper", "pinned", "span_term", "span_first", "span_near", "span_or", "span_not", "span_containing", "span_within", "span_multi", "field_masking_span",
    "intervals", "knn", "sparse_vector", "text_expansion", "weighted_tokens", "semantic", "rule", "type", "match_bool", "rank_features",
];

/// Names of every query object in a query tree that is not a known query.
fn unknown_query(q: &Value) -> Option<String> {
    let o = q.as_object()?;
    for (name, body) in o {
        if !QUERY_NAMES.contains(&name.as_str()) {
            return Some(name.clone());
        }
        match name.as_str() {
            "bool" => {
                for k in ["must", "should", "filter", "must_not"] {
                    match &body[k] {
                        Value::Array(a) => {
                            if let Some(n) = a.iter().find_map(unknown_query) {
                                return Some(n);
                            }
                        }
                        v @ Value::Object(_) => {
                            if let Some(n) = unknown_query(v) {
                                return Some(n);
                            }
                        }
                        _ => {}
                    }
                }
            }
            "constant_score" => {
                if let Some(n) = unknown_query(&body["filter"]) {
                    return Some(n);
                }
            }
            _ => {}
        }
    }
    None
}

/// Check that a query parses (`Datafeed query is not parsable`).
fn check_query(q: &Value, at: xc::Loc) -> Result<()> {
    let cause = if let Some(name) = unknown_query(q) {
        Some(EsError::new(400, "named_object_not_found_exception", format!("[1:{}] unknown field [{name}]", name.len() + 5)))
    } else {
        crate::search::parse_request(&json!({"query": q}), &Default::default()).err()
    };
    if let Some(cause) = cause {
        return Err(xc::wrap(at, "datafeed_config", "query", bad_request("Datafeed query is not parsable").caused_by(cause)));
    }
    Ok(())
}

/// Parsed fields of a datafeed body (put or update).
#[derive(Default)]
pub struct Parsed {
    pub fields: Map<String, Value>,
}

fn parse_fields(body: &Map<String, Value>, p: &Pos, parser: &str, for_update: bool) -> Result<Parsed> {
    let mut out = Map::new();
    for (k, v) in body {
        let vp = p.get(k);
        let tv = |name: &str| -> Result<String> {
            let s = v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string());
            super::job::parse_time(&s, name).map(|t| t.rep).map_err(|e| wrap(vp.start, parser, name, e))
        };
        match k.as_str() {
            "datafeed_id" | "job_id" => {
                out.insert(k.clone(), json!(v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())));
            }
            "indices" | "indexes" => {
                let list: Vec<Value> = match v {
                    Value::Array(a) => a.clone(),
                    Value::String(s) => vec![json!(s)],
                    other => return Err(xparse(format!("{} [{parser}] {k} doesn't support values of type: {}", fmt(vp.start), xc::token_name(other)))),
                };
                out.insert("indices".into(), Value::Array(list));
            }
            "query" => {
                check_query(v, vp.end)?;
                out.insert("query".into(), v.clone());
            }
            "query_delay" | "frequency" => {
                let r = tv(k)?;
                out.insert(k.clone(), json!(r));
            }
            "scroll_size" => {
                let n = v.as_i64().ok_or_else(|| xparse(format!("{} [{parser}] scroll_size doesn't support values of type: {}", fmt(vp.start), xc::token_name(v))))?;
                out.insert(k.clone(), json!(n));
            }
            "aggregations" | "aggs" => {
                out.insert("aggregations".into(), v.clone());
            }
            "script_fields" => {
                out.insert(k.clone(), normalize_script_fields(v));
            }
            "runtime_mappings" => {
                out.insert(k.clone(), v.clone());
            }
            "chunking_config" => {
                let o = v.as_object().cloned().unwrap_or_default();
                let mode = o.get("mode").and_then(Value::as_str).unwrap_or("auto").to_ascii_lowercase();
                if !matches!(mode.as_str(), "auto" | "manual" | "off") {
                    return Err(wrap(vp.start, parser, k, EsError::illegal_argument(format!("No enum constant org.elasticsearch.xpack.core.ml.datafeed.ChunkingConfig.Mode.{}", mode.to_ascii_uppercase()))));
                }
                let mut m = Map::new();
                m.insert("mode".into(), json!(mode));
                if let Some(ts) = o.get("time_span") {
                    let s = ts.as_str().map(str::to_string).unwrap_or_else(|| ts.to_string());
                    let t = super::job::parse_time(&s, "time_span").map_err(|e| wrap(vp.start, parser, k, e))?;
                    m.insert("time_span".into(), json!(t.rep));
                }
                if mode == "manual" && !m.contains_key("time_span") {
                    return Err(bad_request("When chunking_config mode is manual, time_span must be set"));
                }
                out.insert(k.clone(), Value::Object(m));
            }
            "delayed_data_check_config" => {
                let o = v.as_object().cloned().unwrap_or_default();
                let mut m = Map::new();
                m.insert("enabled".into(), o.get("enabled").cloned().unwrap_or(json!(true)));
                if let Some(cw) = o.get("check_window") {
                    let s = cw.as_str().map(str::to_string).unwrap_or_else(|| cw.to_string());
                    let t = super::job::parse_time(&s, "check_window").map_err(|e| wrap(vp.start, parser, k, e))?;
                    m.insert("check_window".into(), json!(t.rep));
                }
                out.insert(k.clone(), Value::Object(m));
            }
            "max_empty_searches" => {
                let n = v.as_i64().ok_or_else(|| xparse(format!("{} [{parser}] max_empty_searches doesn't support values of type: {}", fmt(vp.start), xc::token_name(v))))?;
                if n == 0 || n < -1 {
                    return Err(bad_request(format!("Invalid max_empty_searches value [{n}]; must be greater than zero or equal to -1")));
                }
                out.insert(k.clone(), json!(n));
            }
            "indices_options" => {
                out.insert(k.clone(), v.clone());
            }
            other => {
                let _ = for_update;
                return Err(unknown(p.key(other), parser, other, FIELDS));
            }
        }
    }
    Ok(Parsed { fields: out })
}

/// Parse a `PUT _ml/datafeeds/{id}` body (also a job's `datafeed_config`).
/// `param_opts` are indices options given as URL params.
pub fn parse_put(id: &str, body: &Value, raw: &str, param_opts: &Map<String, Value>) -> Result<Value> {
    let pos = xc::positions(raw);
    let o = match body {
        Value::Object(o) => o.clone(),
        Value::Null => return Err(EsError::new(400, "parse_exception", "request body is required")),
        other => return Err(xparse(format!("[1:1] [datafeed_config] Expected START_OBJECT but was: {}", xc::token_name(other)))),
    };
    build(id, &o, &pos, param_opts)
}

/// Build a datafeed config from a body object.
pub fn build(id: &str, o: &Map<String, Value>, pos: &Pos, param_opts: &Map<String, Value>) -> Result<Value> {
    let parsed = parse_fields(o, pos, "datafeed_config", false)?.fields;
    if let Some(b) = parsed.get("datafeed_id").and_then(Value::as_str) {
        if b != id {
            return Err(EsError::illegal_argument(format!("Inconsistent datafeed_id; '{b}' specified in the body differs from '{id}' specified as a URL argument")));
        }
    }
    if !super::is_valid_id(id) {
        return Err(bad_request(format!(
            "Invalid datafeed_id; '{id}' can contain lowercase alphanumeric (a-z and 0-9), hyphens or underscores; must start and end with alphanumeric"
        )));
    }
    if id.len() > super::ID_LENGTH_LIMIT {
        return Err(bad_request(format!("The datafeed_id cannot contain more than {} characters.", super::ID_LENGTH_LIMIT)));
    }
    let Some(job_id) = parsed.get("job_id").and_then(Value::as_str).map(str::to_string) else {
        return Err(EsError::illegal_argument("[job_id] must not be null."));
    };
    let indices = parsed.get("indices").cloned().unwrap_or(json!([]));
    if indices.as_array().map(|a| a.is_empty() || a.iter().any(|x| x.as_str().map(str::is_empty).unwrap_or(true))).unwrap_or(true) {
        return Err(bad_request(format!("Invalid indices value '{}' in datafeed configuration", render_list(&indices))));
    }
    if parsed.contains_key("aggregations") && parsed.contains_key("script_fields") {
        return Err(bad_request("script_fields cannot be used in combination with aggregations"));
    }
    let mut opts = indices_options_default();
    if let Some(io) = parsed.get("indices_options").and_then(Value::as_object) {
        merge_indices_options(&mut opts, io)?;
    }
    merge_indices_options(&mut opts, param_opts)?;
    let mut f = parsed;
    f.insert("datafeed_id".into(), json!(id));
    f.insert("job_id".into(), json!(job_id));
    f.insert("indices_options".into(), Value::Object(opts));
    if !f.contains_key("query_delay") {
        f.insert("query_delay".into(), json!(default_query_delay(&job_id)));
    }
    if !f.contains_key("chunking_config") {
        f.insert("chunking_config".into(), json!({"mode": "auto"}));
    }
    if !f.contains_key("query") {
        f.insert("query".into(), json!({"match_all": {}}));
    }
    if !f.contains_key("scroll_size") {
        f.insert("scroll_size".into(), json!(1000));
    }
    if !f.contains_key("delayed_data_check_config") {
        f.insert("delayed_data_check_config".into(), json!({"enabled": true}));
    }
    if f.get("max_empty_searches").and_then(Value::as_i64) == Some(-1) {
        f.remove("max_empty_searches");
    }
    f.insert("authorization".into(), json!({"roles": ["superuser"]}));
    Ok(render(&Value::Object(f), false))
}

fn render_list(v: &Value) -> String {
    format!("[{}]", v.as_array().map(|a| a.iter().map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string())).collect::<Vec<_>>().join(", ")).unwrap_or_default())
}

/// Render in `DatafeedConfig.toXContent` order; `exclude_generated` drops
/// generated fields (`query_delay` default, `chunking_config`,
/// `authorization`...).
pub fn render(d: &Value, exclude_generated: bool) -> Value {
    let src = d.as_object().cloned().unwrap_or_default();
    let mut o = Map::new();
    let order: &[&str] = if exclude_generated {
        &["datafeed_id", "job_id", "query_delay", "indices_options", "chunking_config", "query", "frequency", "indices", "aggregations", "script_fields", "scroll_size", "delayed_data_check_config", "max_empty_searches", "runtime_mappings"]
    } else {
        &[
            "datafeed_id", "job_id", "authorization", "query_delay", "chunking_config", "indices_options", "query", "frequency", "indices", "aggregations", "script_fields", "scroll_size", "delayed_data_check_config",
            "max_empty_searches", "runtime_mappings",
        ]
    };
    let job_id = src.get("job_id").and_then(Value::as_str).unwrap_or("");
    for k in order {
        if let Some(v) = src.get(*k) {
            if exclude_generated {
                // Values equal to the generated defaults are left out.
                let default = match *k {
                    "query_delay" => Some(json!(default_query_delay(job_id))),
                    "chunking_config" => Some(json!({"mode": "auto"})),
                    "indices_options" => Some(Value::Object(indices_options_default())),
                    _ => None,
                };
                if default.as_ref() == Some(v) {
                    continue;
                }
            }
            o.insert((*k).to_string(), v.clone());
        }
    }
    Value::Object(o)
}

/// Mark whether `query_delay` was defaulted (for `exclude_generated`).
pub fn mark_generated(stored: &mut Value, body: &Value) {
    let generated = body.get("query_delay").is_none();
    if let Some(o) = stored.as_object_mut() {
        o.insert("query_delay_generated".into(), json!(generated));
    }
}

/// Apply `POST _ml/datafeeds/{id}/_update`.
pub fn apply_update(stored: &Value, body: &Value, raw: &str, param_opts: &Map<String, Value>) -> Result<Value> {
    let pos = xc::positions(raw);
    let empty = Map::new();
    let o = match body {
        Value::Object(o) => o,
        Value::Null => &empty,
        other => return Err(xparse(format!("[1:1] [datafeed_update] Expected START_OBJECT but was: {}", xc::token_name(other)))),
    };
    let parsed = parse_fields(o, &pos, "datafeed_update", true)?.fields;
    let mut cur = stored.as_object().cloned().unwrap_or_default();
    if let Some(j) = parsed.get("job_id") {
        if Some(j) != cur.get("job_id") {
            return Err(bad_request("Datafeed's job_id cannot be changed."));
        }
    }
    for (k, v) in &parsed {
        match k.as_str() {
            "datafeed_id" | "job_id" => {}
            "indices_options" => {
                let mut opts = cur.get("indices_options").and_then(Value::as_object).cloned().unwrap_or_else(indices_options_default);
                if let Some(io) = v.as_object() {
                    merge_indices_options(&mut opts, io)?;
                }
                cur.insert(k.clone(), Value::Object(opts));
            }
            "max_empty_searches" => {
                if v.as_i64() == Some(-1) {
                    cur.remove(k);
                } else {
                    cur.insert(k.clone(), v.clone());
                }
            }
            "query_delay" => {
                cur.insert(k.clone(), v.clone());
                cur.insert("query_delay_generated".into(), json!(false));
            }
            _ => {
                cur.insert(k.clone(), v.clone());
            }
        }
    }
    if !param_opts.is_empty() {
        let mut opts = cur.get("indices_options").and_then(Value::as_object).cloned().unwrap_or_else(indices_options_default);
        merge_indices_options(&mut opts, param_opts)?;
        cur.insert("indices_options".into(), Value::Object(opts));
    }
    let indices = cur.get("indices").cloned().unwrap_or(json!([]));
    if indices.as_array().map(|a| a.is_empty()).unwrap_or(true) {
        return Err(bad_request(format!("Invalid indices value '{}' in datafeed configuration", render_list(&indices))));
    }
    let mut v = Value::Object(cur);
    let generated = v.get("query_delay_generated").cloned();
    v = render(&v, false);
    if let (Some(g), Some(o)) = (generated, v.as_object_mut()) {
        o.insert("query_delay_generated".into(), g);
    }
    Ok(v)
}

/// The stored datafeed for API output (internal markers removed). ES
/// records the creating user's roles (`authorization`) only when security
/// is enabled.
pub fn public(stored: &Value, exclude_generated: bool, security: bool) -> Value {
    let mut v = render(stored, exclude_generated);
    if exclude_generated || !security {
        if let Some(o) = v.as_object_mut() {
            o.remove("authorization");
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_delay_matches_es() {
        // ES 8.19: job "j1" -> 68582ms, "y1" -> 107752ms
        assert_eq!(default_query_delay("j1"), "68582ms");
        assert_eq!(default_query_delay("y1"), "107752ms");
        assert_eq!(default_query_delay("p7aspot-j1"), "102822ms");
    }
}
