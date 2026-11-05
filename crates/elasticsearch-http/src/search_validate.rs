//! Search request checks ES performs before (or while) running a search:
//! scroll-context restrictions (`SearchRequest#validate`), keep-alive
//! limits and the per-index request limits (`index.max_rescore_window`,
//! `index.max_docvalue_fields_search`, `index.max_regex_length`,
//! `index.max_slices_per_scroll`, scroll batch size).

use elasticsearch_engine::error::{EsError, Result};
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{SearchRequest, Target};
use elasticsearch_engine::rustic_core::query::TrackTotalHits;
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;

/// `SearchRequest#validate` (scroll restrictions).
pub(crate) fn validate_request(body: &Value, params: &HashMap<String, String>, req: &SearchRequest) -> Result<()> {
    if let Some(v) = params.get("pre_filter_shard_size") {
        if v.trim().parse::<i64>().map(|n| n < 1).unwrap_or(false) {
            return Err(EsError::illegal_argument("preFilterShardSize must be >= 1"));
        }
    }
    if params.get("search_type").is_some_and(|v| !v.is_empty()) && !req.knn.is_empty() {
        return Err(EsError::illegal_argument("cannot set [search_type] when using [knn] search, since the search type is determined automatically"));
    }
    if !params.contains_key("scroll") {
        return Ok(());
    }
    let mut errs: Vec<String> = Vec::new();
    // RestSearchAction applies URL parameters after the body. Validate that
    // effective value, rather than rejecting an overridden body setting or
    // overlooking a URL-only limit. An absent value is made exact at execution.
    let explicit_tracking = params.contains_key("track_total_hits") || body.get("track_total_hits").is_some();
    let tth_limited = explicit_tracking && !matches!(req.track_total_hits,
        Some(TrackTotalHits::True) | Some(TrackTotalHits::UpTo(2_147_483_647)));
    if tth_limited {
        errs.push("disabling [track_total_hits] is not allowed in a scroll context".into());
    }
    if req.from > 0 && (body.get("from").is_some() || params.contains_key("from")) {
        errs.push("using [from] is not allowed in a scroll context".into());
    }
    if req.size == 0 && (body.get("size").is_some() || params.contains_key("size")) {
        errs.push("[size] cannot be [0] in a scroll context".into());
    }
    if body.get("rescore").map(|r| !r.is_null() && r.as_array().map(|a| !a.is_empty()).unwrap_or(true)).unwrap_or(false) {
        errs.push("using [rescore] is not allowed in a scroll context".into());
    }
    if body.get("collapse").is_some() {
        errs.push("cannot use `collapse` in a scroll context".into());
    }
    if params.get("request_cache").map(|v| v == "true").unwrap_or(false) {
        errs.push("[request_cache] cannot be used in a scroll context".into());
    }
    if errs.is_empty() {
        return Ok(());
    }
    let refs: Vec<&str> = errs.iter().map(String::as_str).collect();
    Err(EsError::action_validation(&refs))
}

/// `search.max_open_scroll_context` (cluster setting, default 500): ES
/// counts one context per shard a scroll searches, per node.
pub(crate) fn check_scroll_contexts(node: &Node, adding: usize) -> Result<()> {
    let max = {
        let cs = node.cluster_settings.read();
        cs.1.get("search.max_open_scroll_context")
            .or(cs.0.get("search.max_open_scroll_context"))
            .and_then(|v| v.as_u64().or_else(|| v.as_str()?.parse().ok()))
            .unwrap_or(500) as usize
    };
    let now = std::time::Instant::now();
    let open: usize = node.scrolls.lock().values().filter(|s| s.expires > now).map(|s| s.targets.len().max(1)).sum();
    if open + adding > max {
        return Err(EsError::new(429, "too_many_scroll_contexts_exception", format!(
            "Trying to create too many scroll contexts. Must be less than or equal to: [{max}]. This limit can be set by changing the [search.max_open_scroll_context] setting."
        )));
    }
    Ok(())
}

fn cluster_time_setting(node: &Node, key: &str, default_ms: i64) -> i64 {
    let cs = node.cluster_settings.read();
    cs.1.get(key)
        .or(cs.0.get(key))
        .and_then(|v| v.as_str().map(str::to_string).or_else(|| v.as_i64().map(|n| n.to_string())))
        .and_then(|s| elasticsearch_engine::date::parse_time_value(&s))
        .unwrap_or(default_ms)
}

#[cfg(test)]
mod scroll_total_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn open_scroll_contexts_are_capped() {
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        node.cluster_settings.write().0.insert("search.max_open_scroll_context".into(), json!("2"));
        let ctx = |expires| elasticsearch_engine::node::ScrollCtx {
            targets: Vec::new(),
            req: elasticsearch_engine::search::parse_request(&json!({}), &HashMap::new()).unwrap(),
            cursors: Vec::new(),
            keep_alive: std::time::Duration::from_secs(60),
            expires,
            accounting: None,
            lost: Vec::new(),
        };
        let later = std::time::Instant::now() + std::time::Duration::from_secs(60);
        check_scroll_contexts(&node, 2).unwrap();
        node.scrolls.lock().insert("a".into(), ctx(later));
        node.scrolls.lock().insert("old".into(), ctx(std::time::Instant::now()));
        check_scroll_contexts(&node, 1).unwrap();
        node.scrolls.lock().insert("b".into(), ctx(later));
        let e = check_scroll_contexts(&node, 1).unwrap_err();
        assert_eq!((e.status, e.kind.as_str()), (429, "too_many_scroll_contexts_exception"));
        assert!(e.reason.contains("less than or equal to: [2]"), "{}", e.reason);
    }

    #[test]
    fn scroll_total_validation_uses_url_precedence_and_exact_sentinel() {
        for (body, parameter, accepted) in [
            (json!({}), None, true),
            (json!({"track_total_hits": true}), None, true),
            (json!({"track_total_hits": 2_147_483_647i64}), None, true),
            (json!({"track_total_hits": false}), None, false),
            (json!({"track_total_hits": 10_000}), None, false),
            (json!({"track_total_hits": -1}), None, false),
            (json!({"track_total_hits": "10000"}), None, false),
            (json!({}), Some("false"), false),
            (json!({}), Some("0"), false),
            (json!({}), Some("10000"), false),
            (json!({}), Some("-1"), false),
            (json!({}), Some("2147483647"), true),
            (json!({"track_total_hits": false}), Some("true"), true),
            (json!({"track_total_hits": 1}), Some("2147483647"), true),
            (json!({"track_total_hits": true}), Some("false"), false),
            (json!({"track_total_hits": true}), Some("1"), false),
        ] {
            let mut params = HashMap::from([("scroll".to_string(), "1m".to_string())]);
            if let Some(parameter) = parameter { params.insert("track_total_hits".into(), parameter.into()); }
            let req = elasticsearch_engine::search::parse_request(&body, &params).unwrap();
            let result = validate_request(&body, &params, &req);
            assert_eq!(result.is_ok(), accepted, "body={body}, params={params:?}, result={result:?}");
            if let Err(error) = result {
                assert_eq!(error.kind, "action_request_validation_exception");
                assert!(error.reason.contains("disabling [track_total_hits] is not allowed in a scroll context"));
            }
            // The same valid search setting is accepted outside scroll.
            params.remove("scroll");
            let req = elasticsearch_engine::search::parse_request(&body, &params).unwrap();
            validate_request(&body, &params, &req).unwrap();
        }
    }

    #[test]
    fn scroll_total_numeric_sentinels_match_boolean_tracking() {
        for params in [false, true] {
            let parse = |value: Value| {
                let (body, query) = if params {
                    (json!({}), HashMap::from([("track_total_hits".into(), value.to_string())]))
                } else { (json!({"track_total_hits": value}), HashMap::new()) };
                elasticsearch_engine::search::parse_request(&body, &query).unwrap().track_total_hits
            };
            assert_eq!(parse(json!(-1)), parse(json!(false)));
            assert_eq!(parse(json!(2_147_483_647i64)), parse(json!(true)));
            assert_eq!(parse(json!(10_000)), Some(TrackTotalHits::UpTo(10_000)));
        }
    }
}

fn fmt_time(ms: i64) -> String {
    let units = [(86_400_000, "d"), (3_600_000, "h"), (60_000, "m"), (1000, "s")];
    for (u, s) in units {
        if ms >= u && ms % u == 0 {
            return format!("{}{s}", ms / u);
        }
    }
    for (u, s) in units {
        if ms >= u {
            return format!("{:.1}{s}", ms as f64 / u as f64);
        }
    }
    format!("{ms}ms")
}

/// `search.max_keep_alive` (cluster setting, default 24h).
pub(crate) fn check_keep_alive(node: &Node, ka: Duration) -> Result<()> {
    let max = cluster_time_setting(node, "search.max_keep_alive", 86_400_000);
    let ms = ka.as_millis() as i64;
    if ms > max {
        return Err(EsError::illegal_argument(format!(
            "Keep alive for request ({}) is too large. It must be less than ({}). This limit can be set by changing the [search.max_keep_alive] cluster level setting.",
            fmt_time(ms),
            fmt_time(max)
        )));
    }
    Ok(())
}

fn shard_failure(t: &Target, e: EsError) -> EsError {
    EsError::search_phase(&t.svc.name, e)
}

/// Per-index limits checked by `SearchService` when it creates the context.
pub(crate) fn check_limits(body: &Value, req: &SearchRequest, targets: &[Target], scroll: bool) -> Result<()> {
    for t in targets {
        let svc = &t.svc;
        let max_window = svc.setting_usize("index.max_result_window", 10_000);
        if scroll && req.from + req.size > max_window {
            return Err(shard_failure(
                t,
                EsError::illegal_argument(format!(
                    "Batch size is too large, size must be less than or equal to: [{max_window}] but was [{}]. Scroll batch sizes cost as much memory as result windows so they are controlled by the [index.max_result_window] index level setting.",
                    req.from + req.size
                )),
            ));
        }
        let max_rescore = svc.setting_usize("index.max_rescore_window", max_window);
        let rescores: Vec<&Value> = match body.get("rescore") {
            Some(Value::Array(a)) => a.iter().collect(),
            Some(o @ Value::Object(_)) => vec![o],
            _ => vec![],
        };
        for r in rescores {
            let w = r.get("window_size").and_then(Value::as_u64).unwrap_or(10) as usize;
            if w > max_rescore {
                return Err(shard_failure(
                    t,
                    EsError::illegal_argument(format!(
                        "Rescore window [{w}] is too large. It must be less than [{max_rescore}]. This prevents allocating massive heaps for storing the results to be rescored. This limit can be set by changing the [index.max_rescore_window] index level setting."
                    )),
                ));
            }
        }
        let max_dv = svc.setting_usize("index.max_docvalue_fields_search", 100);
        if req.docvalue_fields.len() > max_dv {
            return Err(shard_failure(
                t,
                EsError::illegal_argument(format!(
                    "Trying to retrieve too many docvalue_fields. Must be less than or equal to: [{max_dv}] but was [{}]. This limit can be set by changing the [index.max_docvalue_fields_search] index level setting.",
                    req.docvalue_fields.len()
                )),
            ));
        }
        if let Some(slice) = body.get("slice") {
            let max_slices = svc.setting_usize("index.max_slices_per_scroll", 1024);
            let n = slice.get("max").and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))).unwrap_or(1) as usize;
            if n > max_slices {
                return Err(shard_failure(
                    t,
                    EsError::illegal_argument(format!(
                        "The number of slices [{n}] is too large. It must be less than [{max_slices}]. This limit can be set by changing the [index.max_slices_per_scroll] index level setting."
                    )),
                ));
            }
        }
        if let Some(q) = &req.query {
            let max_re = svc.setting_usize("index.max_regex_length", 1000);
            if let Some(e) = regex_lengths(q, max_re) {
                return Err(shard_failure(t, EsError::query_shard(&svc.name, &svc.uuid, e)));
            }
        }
    }
    Ok(())
}

/// The first `regexp` / `prefix` (or query_string `/regex/`) longer than `max`.
fn regex_lengths(v: &Value, max: usize) -> Option<String> {
    let too_long = |kind: &str, what: &str, len: usize| {
        format!(
            "The length of {what} [{len}] used in the {kind} Query request has exceeded the allowed maximum of [{max}]. This maximum can be set by changing the [index.max_regex_length] index level setting."
        )
    };
    match v {
        Value::Object(o) => {
            for (k, x) in o {
                match k.as_str() {
                    "regexp" | "prefix" => {
                        if let Some(xo) = x.as_object() {
                            for (f, spec) in xo {
                                if f == "boost" || f == "_name" {
                                    continue;
                                }
                                let s = match spec {
                                    Value::String(s) => Some(s.as_str()),
                                    Value::Object(so) => so.get("value").and_then(Value::as_str),
                                    _ => None,
                                };
                                if let Some(s) = s {
                                    let len = s.encode_utf16().count();
                                    if len > max {
                                        return Some(if k == "regexp" { too_long("Regexp", "regex", len) } else { too_long("Prefix", "prefix", len) });
                                    }
                                }
                            }
                        }
                    }
                    "query_string" => {
                        if let Some(q) = x.get("query").and_then(Value::as_str) {
                            let t = q.trim();
                            if t.len() > 1 && t.starts_with('/') {
                                let inner = t[1..].rsplit_once('/').map(|(a, _)| a).unwrap_or(&t[1..]);
                                let len = inner.encode_utf16().count();
                                if len > max {
                                    return Some(too_long("Regexp", "regex", len));
                                }
                            }
                        }
                    }
                    _ => {
                        if let Some(e) = regex_lengths(x, max) {
                            return Some(e);
                        }
                    }
                }
            }
            None
        }
        Value::Array(a) => a.iter().find_map(|x| regex_lengths(x, max)),
        _ => None,
    }
}

/// `indices_boost` (`[{"name": boost}, ...]` or `{"name": boost}`) resolved
/// to concrete indices; unknown names fail unless `ignore_unavailable`.
pub(crate) fn index_boosts(node: &Node, body: &Value, params: &HashMap<String, String>) -> Result<Vec<(String, f32)>> {
    let entries: Vec<(String, f32)> = match body.get("indices_boost") {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(Value::as_object)
            .flat_map(|o| o.iter().map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(1.0) as f32)).collect::<Vec<_>>())
            .collect(),
        Some(Value::Object(o)) => o.iter().map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(1.0) as f32)).collect(),
        Some(_) => return Err(EsError::parsing("Expected [START_ARRAY] in [indices_boost] but found [VALUE_STRING]")),
    };
    let ignore = params.get("ignore_unavailable").map(|v| v == "true").unwrap_or(false);
    let mut out: Vec<(String, f32)> = Vec::new();
    for (name, boost) in entries {
        reject_remote(&name)?;
        let opts = elasticsearch_engine::node::ResolveOpts { ignore_unavailable: ignore, allow_no_indices: true, ..Default::default() };
        for t in node.resolve(&name, opts)? {
            if !out.iter().any(|(n, _)| n == &t.svc.name) {
                out.push((t.svc.name.clone(), boost));
            }
        }
    }
    Ok(out)
}

/// `IndexNameExpressionResolver.checkForRemoteIndexExpression`: APIs that
/// do not allow remote indices reject `cluster:index` expressions.
pub(crate) fn reject_remote(expr: &str) -> Result<()> {
    let is_remote = |e: &str| {
        let e = e.strip_prefix('-').unwrap_or(e);
        !e.starts_with('<') && e.find(':').is_some_and(|i| i > 0 && !e[i..].starts_with("::"))
    };
    let parts: Vec<&str> = expr.split(',').map(str::trim).collect();
    if parts.iter().any(|p| is_remote(p)) {
        let remote: Vec<&str> = parts.into_iter().filter(|p| is_remote(p)).collect();
        return Err(EsError::illegal_argument(format!("Cross-cluster calls are not supported in this context but remote indices were requested: [{}]", remote.join(", "))));
    }
    Ok(())
}
