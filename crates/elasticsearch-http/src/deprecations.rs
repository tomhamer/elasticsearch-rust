//! `Warning` headers for deprecated request forms, derived from the request
//! alone (ES emits them from the REST handlers / query parsers).

use crate::Req;
use axum::http::Method;
use serde_json::Value;

/// Request forms ES 9 rejects that 8.x (and `compatible-with=8`) only warned about.
pub fn reject(req: &Req) -> Option<elasticsearch_engine::error::EsError> {
    // `RestCloseIndexAction`.
    if !req.compat_v8()
        && req.segs.last().map(String::as_str) == Some("_close")
        && req.param("wait_for_active_shards").is_some_and(|v| v.eq_ignore_ascii_case("index-setting"))
    {
        return Some(elasticsearch_engine::error::EsError::illegal_argument(
            "The 'index-setting' value for parameter 'wait_for_active_shards' is the default behaviour and this configuration value is not supported anymore. Please remove 'wait_for_active_shards=index-setting'.",
        ));
    }
    None
}

/// The APIs whose handler calls `RestUtils.consumeDeprecatedLocalParameter` (ES 9.5.5).
fn local_is_deprecated(req: &Req) -> bool {
    let (a, b) = (req.seg(0), req.seg(1));
    if req.segs.iter().any(|s| s == "field") {
        return false; // `RestGetFieldMappingAction`: `local` was removed
    }
    match a {
        "_cat" => matches!(b, "templates" | "segments" | "plugins" | "component_templates" | "allocation" | "thread_pool" | "master" | "nodeattrs"),
        "_component_template" | "_template" | "_index_template" | "_mapping" | "_mappings" | "_settings" => true,
        "_cluster" => matches!(b, "state" | "settings"),
        _ if a.starts_with('_') && a != "_all" => false,
        // `GET /{index}`, `/{index}/_mapping`, `/{index}/_settings[/{name}]`
        _ => req.segs.len() == 1 || matches!(b, "_mapping" | "_mappings" | "_settings"),
    }
}

/// Messages of the deprecation warnings this request earns.
pub fn for_request(req: &Req, memory:&mut elasticsearch_engine::memory::Reservation) -> Result<Vec<String>,elasticsearch_engine::error::EsError> {
    let mut w = Vec::new();
    macro_rules! warn {($($arg:tt)*)=>{{let message=crate::response_tree::formatted(memory,format_args!($($arg)*))?;crate::response_tree::push(memory,&mut w,message)?;}}}
    fn parsed(req:&Req,memory:&mut elasticsearch_engine::memory::Reservation)->Result<Option<Value>,elasticsearch_engine::error::EsError>{
        let Ok(text)=std::str::from_utf8(&req.body) else{return Ok(None)};
        match crate::response_tree::parse(text,memory){Ok(value)=>Ok(Some(value)),Err(error) if error.status==429=>Err(error),Err(_)=>Ok(None)}
    }
    // `RestFlushJobAction`: a `deprecateAndKeep` route (both REST versions).
    if req.method == Method::POST && req.segs.len() == 4 && req.seg(0) == "_ml" && req.seg(1) == "anomaly_detectors" && req.seg(3) == "_flush" {
        warn!("Forcing any buffered data to be processed is deprecated, in a future major version it will be compulsory to use a datafeed");
    }
    let has_seg = |s: &str| req.segs.iter().any(|x| x == s);
    // ES 9 removed `?local` from the alias APIs (the route table rejects it);
    // `RestGetAliasesAction` still consumes it under `compatible-with=8`.
    if req.compat_v8() && matches!(req.method, Method::GET | Method::HEAD) && req.seg(0) != "_cat" && (has_seg("_alias") || has_seg("_aliases")) {
        if let Some(v) = req.param("local") {
            let v = if v.is_empty() { "true" } else { v };
            warn!("the [?local={v}] query parameter to get-aliases requests has no effect and will be removed in a future version");
        }
    }
    // `RestUtils.consumeDeprecatedLocalParameter`: natively, `?local` on these
    // APIs is accepted with a deprecation (silently under `compatible-with=8`).
    if !req.compat_v8() && req.params.contains_key("local") && matches!(req.method, Method::GET | Method::HEAD) && local_is_deprecated(req) {
        warn!("the [?local] query parameter to this API has no effect, is now deprecated, and will be removed in a future version");
    }
    if req.compat_v8()
        && req.segs.last().map(String::as_str) == Some("_close")
        && req.param("wait_for_active_shards").is_some_and(|v| v.eq_ignore_ascii_case("index-setting"))
    {
        warn!("?wait_for_active_shards=index-setting is now the default behaviour; the 'index-setting' value for this parameter should no longer be used since it will become unsupported in version 9");
    }
    // `SearchSourceBuilder.parseXContent`: the legacy RRF forms.
    if has_seg("_search") && !req.body.is_empty() && (has_key_text(&req.body, b"\"rank\"") || has_key_text(&req.body, b"\"sub_searches\"")) {
        if let Some(Value::Object(body)) = parsed(req,memory)? {
            for k in body.keys() {
                if k == "rank" || k == "sub_searches" {
                    warn!("Deprecated field [{k}] used, replaced by [retriever]");
                }
            }
            // `RRFRankBuilder`: `window_size` is the deprecated name of `rank_window_size`.
            if body.get("rank").and_then(|r| r.get("rrf")).and_then(|r| r.get("window_size")).is_some() {
                warn!("Deprecated field [window_size] used, expected [rank_window_size] instead");
            }
        }
    }
    // Multi-fields on mappers that ignore them (index creation, put mapping).
    let mapping_write = (req.method == Method::PUT && req.segs.len() == 1 && !req.seg(0).starts_with('_')) || has_seg("_mapping") || has_seg("_mappings");
    if mapping_write && !req.body.is_empty() && has_key_text(&req.body, b"\"fields\"") {
        if let Some(body) = parsed(req,memory)? {
            let props = if req.segs.len() == 1 { body.get("mappings").and_then(|m| m.get("properties")) } else { body.get("properties") };
            let mut types = Vec::new();
            if let Some(p) = props {
                ignored_multi_fields(p, &mut types,memory)?;
            }
            for t in types {
                let m = crate::response_tree::formatted(memory,format_args!("Adding multifields to [{t}] mappers has no effect and will be forbidden in future"))?;
                if !w.contains(&m) {
                    crate::response_tree::push(memory,&mut w,m)?;
                }
            }
        }
    }
    // `AggregateMetricDoubleFieldMapper.defaultMetric` is a deprecated
    // parameter (9.4+): parsing a mapping that sets it warns (index
    // creation, put mapping, index / component templates).
    let template_write = matches!(req.method, Method::PUT | Method::POST) && matches!(req.seg(0), "_index_template" | "_component_template" | "_template") && !req.seg(1).is_empty() && !req.seg(1).starts_with('_');
    if (mapping_write || template_write) && !req.body.is_empty() && has_key_text(&req.body, b"\"default_metric\"") {
        if let Some(body) = parsed(req,memory)? {
            if sets_default_metric(&body) {
                warn!("Parameter [default_metric] is deprecated and will be removed in a future version");
            }
        }
    }
    // `MappingParser`: `_source.mode` in a mapping merged as an index
    // template (index creation, template PUTs) on 9.x-created indices.
    let index_create = req.method == Method::PUT && req.segs.len() == 1 && !req.seg(0).starts_with('_');
    if (index_create || template_write) && !req.body.is_empty() && has_key_text(&req.body, b"\"mode\"") {
        if let Some(body) = parsed(req,memory)? {
            let mappings = [body.get("mappings"), body.pointer("/template/mappings")];
            let has_mode = mappings.iter().flatten().any(|m| {
                let m = m.get("_doc").filter(|d| d.is_object()).unwrap_or(m);
                m.get("_source").and_then(|s| s.get("mode")).is_some()
            });
            if has_mode {
                warn!("Configuring source mode in mappings is deprecated and will be removed in future versions. Use [index.mapping.source.mode] index setting instead.");
            }
        }
    }
    Ok(w)
}

/// Whether `body` contains `quoted` (a `"`-delimited token) anywhere: the
/// same test as a scan of every window, visiting only the positions of a
/// `"` (a kNN body of thousands of digits holds a handful).
fn has_key_text(body: &[u8], quoted: &[u8]) -> bool {
    debug_assert!(quoted.first() == Some(&b'"'));
    // The same test as any `"`-position starting with `quoted` (which
    // starts with `"`), by a vectorized substring search: every `_search`
    // passes here, and a kNN body is ~10 KB of floats a byte loop walked
    // twice per request.
    memchr::memmem::find(body, quoted).is_some()
}

/// Whether a mapping (anywhere in `v`) declares an `aggregate_metric_double`
/// field with `default_metric`.
fn sets_default_metric(v: &Value) -> bool {
    match v {
        Value::Object(o) => {
            (o.get("type").and_then(Value::as_str) == Some("aggregate_metric_double") && o.contains_key("default_metric")) || o.values().any(sets_default_metric)
        }
        Value::Array(a) => a.iter().any(sets_default_metric),
        _ => false,
    }
}

/// Field types whose mappers accept but ignore `fields` (ES warns).
fn ignored_multi_fields<'a>(props: &'a Value, out: &mut Vec<&'a str>,memory:&mut elasticsearch_engine::memory::Reservation)->Result<(),elasticsearch_engine::error::EsError> {
    let Some(o) = props.as_object() else { return Ok(()) };
    for spec in o.values() {
        let ty = spec.get("type").and_then(Value::as_str).unwrap_or("object");
        if matches!(ty, "constant_keyword" | "aggregate_metric_double" | "geo_shape" | "shape" | "point") && spec.get("fields").is_some() {
            crate::response_tree::push(memory,out,ty)?;
        }
        if let Some(p) = spec.get("properties") {
            ignored_multi_fields(p, out,memory)?;
        }
    }
    Ok(())
}

/// Warnings of the routes ES registers with `Route.deprecatedForRemoval` for
/// the current REST API version (`RestMonitoringBulkAction`,
/// `RestMonitoringMigrateAlertsAction`: removal after V_9, so every request).
pub fn for_route(method: &Method, path: &str) -> Vec<String> {
    match path {
        "/_template/*" if matches!(*method, Method::PUT | Method::POST) =>
            vec!["Legacy index templates are deprecated in favor of composable templates.".to_string()],
        "/_monitoring/bulk" => vec!["Legacy monitoring API _monitoring/bulk is deprecated for removal in Elasticsearch 10.0.".to_string()],
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_text_matches_the_window_scan() {
        let window = |body: &[u8], k: &[u8]| body.windows(k.len()).any(|w| w == k);
        for body in [&b""[..], b"\"", b"\"rank\"", b"{\"rank\":1}", b"{\"ranks\":1}", b"[\"\"rank\"\"]", b"{\"a\":\"x\\\"rank\\\"\"}", b"\"ran", b"{\"sub_searches\":[]}"] {
            for k in [&b"\"rank\""[..], b"\"sub_searches\"", b"\"mode\""] {
                assert_eq!(has_key_text(body, k), window(body, k), "{:?} {:?}", String::from_utf8_lossy(body), String::from_utf8_lossy(k));
            }
        }
    }

    #[test]
    fn legacy_template_write_routes_warn_without_deprecating_reads_or_deletes() {
        for method in [Method::PUT, Method::POST] {
            assert_eq!(for_route(&method, "/_template/*"),
                vec!["Legacy index templates are deprecated in favor of composable templates."]);
            assert!(for_route(&method, "/_index_template/*").is_empty());
            assert!(for_route(&method, "/_component_template/*").is_empty());
        }
        for method in [Method::GET, Method::HEAD, Method::DELETE] {
            assert!(for_route(&method, "/_template/*").is_empty());
        }
    }
}
