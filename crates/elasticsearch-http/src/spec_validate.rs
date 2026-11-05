//! Spec-driven request validation: the one prologue every request passes
//! before a handler runs (ES `RestController` + `BaseRestHandler`).
//!
//! The table and its semantics match ES 9.5.5, per request REST API version
//! (`Req::api_version`: native v9, or v8 for `compatible-with=8` requests).
//!
//! * Route lookup is a port of ES's `PathTrie` (four matching modes, explicit
//!   segments shadow wildcards) over [`spec_routes::ROUTES`]: the REST spec
//!   routes plus the routes ES registers outside the spec, including the
//!   routes registered only for REST API v8 (`RestApiVersion.V_8`, e.g.
//!   `/{index}/_knn_search`).
//! * A path no route matches is left to the router (`no handler found`).
//! * A matched path with a method no route has: `405 Incorrect HTTP method`
//!   + `Allow` (`OPTIONS` answers `200` + `Allow`). The `Allow` set counts
//!   v8-only routes for every request (`MethodHandlers.getValidMethods`).
//! * A v8-only route dispatches only v8 requests; for a native request whose
//!   method only a v8-only route has, ES finds no handler
//!   (`MethodHandlers.getHandler(method, V_9)` is null): `400 no handler found`.
//! * Then, in ES order: body required (`parse_exception`), unrecognized
//!   query parameters (with Lucene-Levenshtein "did you mean"), body not
//!   supported.
//!
//! Accepted parameters come from probing ES itself (spec/probe.py), natively
//! and with the compat-8 headers, so they include params ES consumes outside
//! the spec, the ones it only consumes with / without a body (`_count?q=` is
//! rejected when a body is sent) and the ones it consumes only for v8
//! requests (`?local` on `GET /_alias`).

use crate::spec_routes::{ROUTES, SETS, V8_RULES};
use crate::{Req, Resp, RespBody};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::version::RestApiVersion;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Body rule of one route.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Body {
    /// `request [M /p] does not support having a body`.
    None,
    Optional,
    /// `request body is required`.
    Required,
    /// `request body or source parameter is required` (`?source=` satisfies it).
    RequiredOrSource,
    /// The handler rejects a body with its own message (`illegal_argument_exception`).
    Reject(&'static str),
    /// The handler rejects a missing body with its own `400` (error type, reason).
    RequiredWith(&'static str, &'static str),
}

/// `RouteDef::v8` when the v8 rules are the native ones.
pub(crate) const SAME: u16 = u16::MAX;

pub(crate) struct RouteDef {
    pub method: &'static str,
    pub path: &'static str,
    #[allow(dead_code)]
    pub api: &'static str,
    /// Registered only for REST API v8 (`RestApiVersion.V_8`): dispatches
    /// only `compatible-with=8` requests; for native ones it only counts for
    /// `Allow`.
    pub v8_only: bool,
    /// Registered only with security enabled.
    pub security: bool,
    /// Index into [`V8_RULES`] of the rules for v8 requests, or [`SAME`].
    pub v8: u16,
    /// Rules for native requests (for a v8-only route: its v8 rules).
    pub rules: Rules,
}

impl RouteDef {
    /// Whether a request of REST API version `v` dispatches to this route.
    fn dispatches(&self, v: RestApiVersion) -> bool {
        !self.v8_only || v == RestApiVersion::V8
    }

    /// The validation rules for a request of REST API version `v`.
    pub(crate) fn rules(&'static self, v: RestApiVersion) -> &'static Rules {
        if v == RestApiVersion::V8 && self.v8 != SAME { &V8_RULES[self.v8 as usize] } else { &self.rules }
    }
}

/// The `BaseRestHandler` validation rules of one route for one REST API version.
pub(crate) struct Rules {
    pub body: Body,
    /// Accepted query params `[without body, with body]` (indexes into `SETS`).
    pub accepted: [u16; 2],
    /// "did you mean" candidates `[without body, with body]`.
    pub candidates: [u16; 2],
    /// Params accepted only together with `q` (`[without body, with body]`).
    pub q_only: [u16; 2],
    /// `RestActions.urlParamsToQueryBuilder`: `df`/`analyzer`/... without `q`
    /// is an error of its own (`[without body, with body]`).
    pub q_check: [bool; 2],
}

/// `RestActions.urlParamsToQueryBuilder` parameters, in ES's order.
const Q_PARAMS: [&str; 5] = ["df", "analyzer", "analyze_wildcard", "lenient", "default_operator"];

/// ES `RestRequest.Method` enum order (the order of `Allow` / `allowed: [..]`).
const METHOD_ORDER: [&str; 9] = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH", "TRACE", "CONNECT"];

#[derive(Default)]
struct Node {
    value: Vec<u16>,
    children: HashMap<&'static str, Node>,
}

#[derive(Default)]
struct Trie {
    root: Node,
    root_value: Vec<u16>,
}

const WILD: &str = "*";

/// Java `String.split("/")`: trailing empty strings are dropped.
fn jsplit(path: &str) -> Vec<&str> {
    let mut v: Vec<&str> = path.split('/').collect();
    while v.last() == Some(&"") {
        v.pop();
    }
    v
}

impl Node {
    fn insert(&mut self, path: &[&'static str], i: usize, value: u16) {
        if i >= path.len() {
            return;
        }
        let tok = path[i];
        let node = self.children.entry(tok).or_default();
        if i == path.len() - 1 {
            node.value.push(value);
        }
        node.insert(path, i + 1, value);
    }

    /// `PathTrie.TrieNode.retrieve` (mode: 0 explicit only, 1 root wildcard,
    /// 2 leaf wildcard, 3 any wildcard).
    fn retrieve<'a>(&'a self, path: &[String], i: usize, mode: u8) -> Option<&'a Vec<u16>> {
        if i >= path.len() {
            return None;
        }
        let tok = path[i].as_str();
        let wild = self.children.get(WILD);
        let (node, used_wild) = match self.children.get(tok) {
            None => {
                let allowed = mode == 3 || (mode == 1 && i == 1) || (mode == 2 && i + 1 == path.len());
                if !allowed {
                    return None;
                }
                (wild?, true)
            }
            Some(n) => {
                if i + 1 == path.len() && n.value.is_empty() && wild.is_some() && mode >= 2 {
                    (wild.unwrap(), true)
                } else if i == 1 && n.value.is_empty() && wild.is_some() && mode == 1 {
                    (wild.unwrap(), true)
                } else {
                    (n, tok == WILD)
                }
            }
        };
        if i == path.len() - 1 {
            return if node.value.is_empty() { None } else { Some(&node.value) };
        }
        let mut v = node.retrieve(path, i + 1, mode);
        if v.is_none() && !used_wild && mode != 0 {
            if let Some(w) = wild {
                v = w.retrieve(path, i + 1, mode);
            }
        }
        v
    }
}

impl Trie {
    fn build(security: bool) -> Trie {
        let mut t = Trie::default();
        for (i, r) in ROUTES.iter().enumerate() {
            if r.security && !security {
                continue;
            }
            let parts = jsplit(r.path);
            if parts.is_empty() {
                t.root_value.push(i as u16);
                continue;
            }
            let start = usize::from(parts[0].is_empty());
            t.root.insert(&parts, start, i as u16);
        }
        t
    }

    /// `PathTrie.retrieveAll`: the values found in each of the four modes.
    fn retrieve_all(&self, raw_path: &str) -> [Option<&Vec<u16>>; 4] {
        let parts = jsplit(raw_path);
        if parts.is_empty() {
            let v = if self.root_value.is_empty() { None } else { Some(&self.root_value) };
            return [v; 4];
        }
        let toks: Vec<String> = parts.iter().map(|p| crate::decode_path_segment(p)).collect();
        let start = usize::from(toks[0].is_empty());
        [0u8, 1, 2, 3].map(|m| self.root.retrieve(&toks, start, m))
    }
}

fn trie(security: bool) -> &'static Trie {
    static PLAIN: OnceLock<Trie> = OnceLock::new();
    static SECURE: OnceLock<Trie> = OnceLock::new();
    if security { SECURE.get_or_init(|| Trie::build(true)) } else { PLAIN.get_or_init(|| Trie::build(false)) }
}

pub(crate) enum Lookup {
    /// No registered route matches the path: the router answers.
    Unmatched,
    /// The route the request dispatches to.
    Route(&'static RouteDef),
    /// 405 / OPTIONS / v8-only-route "no handler" answer.
    Early(Resp),
}

/// Route lookup (`RestController.tryAllHandlers`) for a request of REST API
/// version `version`. `uri` is the raw request URI (path + query) ES prints
/// in 405 / no-handler messages.
pub(crate) fn lookup(method: &Method, raw_path: &str, uri: &str, security: bool, version: RestApiVersion) -> Lookup {
    let found = trie(security).retrieve_all(raw_path);
    if found.iter().all(Option::is_none) {
        return Lookup::Unmatched;
    }
    let m = method.as_str();
    for v in found.iter().flatten() {
        if let Some(&i) = v.iter().find(|&&i| ROUTES[i as usize].method == m && ROUTES[i as usize].dispatches(version)) {
            return Lookup::Route(&ROUTES[i as usize]);
        }
    }
    let mut allowed: Vec<&str> = Vec::new();
    for v in found.iter().flatten() {
        for &i in v.iter() {
            let rm = ROUTES[i as usize].method;
            if !allowed.contains(&rm) {
                allowed.push(rm);
            }
        }
    }
    allowed.sort_by_key(|m| METHOD_ORDER.iter().position(|x| x == m).unwrap_or(usize::MAX));
    if allowed.contains(&m) {
        // Only a v8-only route has this method and this is a native request:
        // ES finds no handler.
        return Lookup::Early(Resp::json(400, serde_json::json!({"error": format!("no handler found for uri [{uri}] and method [{m}]")})));
    }
    if *method == Method::OPTIONS {
        return Lookup::Early(Resp { status: 200, body: RespBody::Text(String::new()), headers: vec![("Allow", allowed.join(","))] });
    }
    Lookup::Early(Resp {
        status: 405,
        body: RespBody::Json(serde_json::json!({
            "error": format!("Incorrect HTTP method for uri [{uri}] and method [{m}], allowed: [{}]", allowed.join(", ")),
            "status": 405
        })),
        headers: vec![("Allow", allowed.join(","))],
    })
}

/// Lucene `LevenshteinDistance.getDistance` (a similarity in `[0, 1]`).
fn similarity(target: &str, other: &str) -> f32 {
    let sa: Vec<u16> = target.encode_utf16().collect();
    let ot: Vec<u16> = other.encode_utf16().collect();
    let (n, m) = (sa.len(), ot.len());
    if n == 0 || m == 0 {
        return if n == m { 1.0 } else { 0.0 };
    }
    let mut p: Vec<usize> = (0..=n).collect();
    let mut d = vec![0usize; n + 1];
    for j in 1..=m {
        let t_j = ot[j - 1];
        d[0] = j;
        for i in 1..=n {
            let cost = usize::from(sa[i - 1] != t_j);
            d[i] = (d[i - 1] + 1).min(p[i] + 1).min(p[i - 1] + cost);
        }
        std::mem::swap(&mut p, &mut d);
    }
    1.0f32 - (p[n] as f32 / n.max(m) as f32)
}

/// `BaseRestHandler.unrecognized`.
pub(crate) fn unrecognized(path: &str, invalid: &[&str], candidates: &[&str]) -> String {
    let mut msg = format!("request [{path}] contains unrecognized parameter{}: ", if invalid.len() > 1 { "s" } else { "" });
    for (n, bad) in invalid.iter().enumerate() {
        let mut scored: Vec<(f32, &str)> = candidates.iter().map(|c| (similarity(bad, c), *c)).filter(|(d, _)| *d > 0.5).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.1.cmp(b.1)));
        if n > 0 {
            msg.push_str(", ");
        }
        msg.push('[');
        msg.push_str(bad);
        msg.push(']');
        match scored.len() {
            0 => {}
            1 => msg.push_str(&format!(" -> did you mean [{}]?", scored[0].1)),
            _ => msg.push_str(&format!(" -> did you mean any of [{}]?", scored.iter().map(|s| s.1).collect::<Vec<_>>().join(", "))),
        }
    }
    msg
}

/// `RestRequest.path()`: the decoded path.
fn decoded_path(req: &Req) -> String {
    crate::decode_component(&req.path)
}

/// The `BaseRestHandler` checks for the dispatched route, in ES order, with
/// the rules of the request's REST API version.
pub(crate) fn check(req: &Req, route: &'static RouteDef) -> Result<(), EsError> {
    let route = route.rules(req.api_version);
    let has_source = req.params.contains_key("source");
    // A coordinator-split bulk carries its items instead of a body.
    let body_empty = req.body.is_empty() && req.extensions.get::<crate::PreparedBulk>().is_none();
    match route.body {
        Body::Required if body_empty => return Err(EsError::new(400, "parse_exception", "request body is required")),
        Body::RequiredOrSource if body_empty && !has_source => {
            return Err(EsError::new(400, "parse_exception", "request body or source parameter is required"))
        }
        Body::RequiredWith(kind, reason) if body_empty && !has_source => return Err(EsError::new(400, kind, reason)),
        _ => {}
    }
    if !req.params.is_empty() {
        let source_is_body = body_empty && has_source && SETS[route.accepted[0] as usize].binary_search(&"source").is_ok();
        if source_is_body {
            // `RestRequest.contentOrSourceParam`.
            match req.params.get("source_content_type") {
                None => return Err(EsError::new(400, "validation_exception", "Validation Failed: 1: source and source_content_type parameters are required;")),
                Some(ct) if !crate::body_type_supported(ct) => {
                    return Err(EsError::new(400, "validation_exception", format!("Validation Failed: 1: Unknown value for source_content_type [{ct}];")));
                }
                _ => {}
            }
        }
        let variant = usize::from(!body_empty || source_is_body);
        let has_q = req.params.contains_key("q");
        if route.q_check[variant] && !has_q {
            let present: Vec<&str> = Q_PARAMS.iter().copied().filter(|p| req.params.contains_key(*p)).collect();
            if !present.is_empty() {
                return Err(EsError::illegal_argument(format!(
                    "request [{}] contains parameters [{}] but missing query string parameter 'q'.",
                    decoded_path(req),
                    present.join(", ")
                )));
            }
        }
        let accepted = SETS[route.accepted[variant] as usize];
        let q_only = SETS[route.q_only[variant] as usize];
        let mut bad: Vec<&str> = req
            .params
            .keys()
            .map(String::as_str)
            .filter(|k| {
                let ok = accepted.binary_search(k).is_ok() && (has_q || q_only.binary_search(k).is_err());
                !ok && !(source_is_body && (*k == "source" || *k == "source_content_type"))
            })
            .collect();
        if !bad.is_empty() {
            bad.sort_unstable();
            let cands = SETS[route.candidates[variant] as usize];
            return Err(EsError::illegal_argument(unrecognized(&decoded_path(req), &bad, cands)));
        }
    }
    if !body_empty {
        match route.body {
            Body::None => {
                return Err(EsError::illegal_argument(format!("request [{} {}] does not support having a body", req.method, decoded_path(req))));
            }
            Body::Reject(msg) => return Err(EsError::illegal_argument(msg)),
            _ => {}
        }
    }
    Ok(())
}

/// The `BaseRestHandler` error for a request with a body on the route
/// `method path` dispatches to, when that route takes none (`request [M /p]
/// does not support having a body`, or the handler's own rejection);
/// `None` when it takes a body or no route matches.
pub(crate) fn body_rejection(method: &Method, path: &str, version: RestApiVersion) -> Option<EsError> {
    let Lookup::Route(route) = lookup(method, path, path, false, version) else { return None };
    match route.rules(version).body {
        Body::None => Some(EsError::illegal_argument(format!("request [{} {}] does not support having a body", method, crate::decode_component(path)))),
        Body::Reject(msg) => Some(EsError::illegal_argument(msg)),
        _ => None,
    }
}

/// Route lookup + checks, for callers that already handled the 405 answer
/// (the security layer validates after authentication, as ES does).
pub(crate) fn validate(req: &Req, security: bool) -> Result<(), EsError> {
    match lookup(&req.method, &req.path, &req.path, security, req.api_version) {
        Lookup::Route(r) => check(req, r),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The no-body broadcast APIs reject a body as ES does; body-taking
    /// routes and unknown paths are not refused here.
    #[test]
    fn body_rejection_names_es_message_for_no_body_routes() {
        let v = RestApiVersion::CURRENT;
        for (method, path) in [(Method::POST, "/_refresh"), (Method::POST, "/logs/_refresh"), (Method::POST, "/_flush"), (Method::POST, "/logs/_flush"),
            (Method::POST, "/logs/_forcemerge")] {
            let e = body_rejection(&method, path, v).unwrap_or_else(|| panic!("{method} {path} accepted a body"));
            assert_eq!((e.status, e.kind.as_str()), (400, "illegal_argument_exception"), "{method} {path}");
            assert_eq!(e.reason, format!("request [{method} {path}] does not support having a body"));
        }
        assert!(body_rejection(&Method::POST, "/logs/_search", v).is_none());
        assert!(body_rejection(&Method::POST, "/logs/_bulk", v).is_none());
        assert!(body_rejection(&Method::POST, "/_no_such_api_here/x/y/z", v).is_none());
    }

    fn allow_v(method: Method, path: &str, v: RestApiVersion) -> Option<String> {
        match lookup(&method, path, path, false, v) {
            Lookup::Early(r) => r.headers.iter().find(|(k, _)| *k == "Allow").map(|(_, v)| v.clone()).or_else(|| match &r.body {
                RespBody::Json(j) => Some(format!("{} {}", r.status, j["error"].as_str().unwrap_or_default())),
                _ => None,
            }),
            Lookup::Route(r) => Some(format!("route {} {}", r.method, r.path)),
            Lookup::Unmatched => None,
        }
    }

    fn allow(method: Method, path: &str) -> Option<String> {
        allow_v(method, path, RestApiVersion::V9)
    }

    fn route(method: &str, path: &str) -> &'static RouteDef {
        ROUTES.iter().find(|r| r.method == method && r.path == path).unwrap()
    }

    #[test]
    fn trie_matches_es() {
        // Verified against ES 9.5.5 (native requests).
        assert_eq!(allow(Method::DELETE, "/_refresh").as_deref(), Some("GET,POST"));
        assert_eq!(allow(Method::DELETE, "/_search").as_deref(), Some("GET,POST"));
        assert_eq!(allow(Method::POST, "/x1").as_deref(), Some("GET,PUT,DELETE,HEAD"));
        assert_eq!(allow(Method::HEAD, "/_cluster/health").as_deref(), Some("GET"));
        assert_eq!(allow(Method::OPTIONS, "/x1/_doc/1").as_deref(), Some("GET,POST,PUT,DELETE,HEAD"));
        assert_eq!(allow(Method::GET, "/x1/_search").as_deref(), Some("route GET /*/_search"));
        assert_eq!(allow(Method::GET, "/").as_deref(), Some("route GET /"));
        // The v7-compat `/{index}/{type}` routes are gone: no handler.
        assert!(allow(Method::GET, "/aa/bb").is_none());
        assert!(allow(Method::POST, "/aa/bb").is_none());
        assert!(allow(Method::GET, "/aa/bb/cc/dd/ee/ff").is_none());
        // Removed in 9.0.
        assert!(allow(Method::POST, "/x1/_unfreeze").is_none());
        // `/{index}/_knn_search` is registered for REST API v8 only: a native
        // request finds no handler, but the methods still count for `Allow`.
        assert_eq!(
            allow(Method::GET, "/x1/_knn_search").as_deref(),
            Some("400 no handler found for uri [/x1/_knn_search] and method [GET]")
        );
        assert_eq!(allow(Method::PUT, "/x1/_knn_search").as_deref(), Some("GET,POST"));
        assert_eq!(allow(Method::OPTIONS, "/x1/_knn_search").as_deref(), Some("GET,POST"));
    }

    #[test]
    fn compat8_matches_es() {
        // Verified against ES 9.5.5 with `compatible-with=8` requests.
        let v8 = RestApiVersion::V8;
        assert_eq!(allow_v(Method::GET, "/x1/_knn_search", v8).as_deref(), Some("route GET /*/_knn_search"));
        assert_eq!(allow_v(Method::POST, "/x1/_knn_search", v8).as_deref(), Some("route POST /*/_knn_search"));
        assert_eq!(allow_v(Method::PUT, "/x1/_knn_search", v8).as_deref(), Some("GET,POST"));
        assert_eq!(allow_v(Method::DELETE, "/_refresh", v8).as_deref(), Some("GET,POST"));
        assert!(allow_v(Method::POST, "/x1/_unfreeze", v8).is_none());
        // `?local` on get-aliases: consumed only for v8 requests.
        for (m, p) in [("GET", "/_alias"), ("GET", "/_alias/*"), ("GET", "/*/_alias"), ("HEAD", "/_alias/*"), ("HEAD", "/*/_alias/*")] {
            let r = route(m, p);
            let has_local = |v| SETS[r.rules(v).accepted[0] as usize].contains(&"local");
            assert!(!has_local(RestApiVersion::V9), "{m} {p}");
            assert!(has_local(v8), "{m} {p}");
        }
        assert!(!SETS[route("GET", "/_cat/aliases").rules(v8).accepted[0] as usize].contains(&"local"));
    }

    #[test]
    fn did_you_mean() {
        assert_eq!(unrecognized("/_search", &["foo"], &["from", "size"]), "request [/_search] contains unrecognized parameter: [foo]");
        assert_eq!(
            unrecognized("/i/_refresh", &["ignore_unavailabl", "x"], &["ignore_unavailable", "allow_no_indices"]),
            "request [/i/_refresh] contains unrecognized parameters: [ignore_unavailabl] -> did you mean [ignore_unavailable]?, [x]"
        );
        assert!((similarity("sise", "size") - 0.75).abs() < 1e-6);
    }
}
