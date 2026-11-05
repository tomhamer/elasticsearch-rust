//! Monitoring / observability / admin APIs answered for a single node:
//! `_nodes/stats`, `_nodes/hot_threads`, `_nodes/usage`, `_cluster/stats`,
//! `_health_report`, `_migration/deprecations`, `_cluster/allocation/explain`,
//! `_cluster/reroute`, `_recovery`, `_shard_stores`, `_disk_usage`,
//! `_field_usage_stats`, `_block`, `_reload_search_analyzers`,
//! `_resolve/index`, `_search_shards`, plus the REST / HTTP usage counters
//! those APIs report. Search-side APIs (`_validate/query`, `_terms_enum`,
//! `_explain`, profile) live in [`crate::obs_search`].

use crate::sysinfo::{self, Sys};
use crate::util::bytes_fmt;
use crate::{blocking, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::index::IndexService;
use elasticsearch_engine::node::{Node, ResolveOpts, ES_VERSION, NODE_ID};
use elasticsearch_engine::settings;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

pub const ROLES: [&str; 11] = ["data", "data_cold", "data_content", "data_frozen", "data_hot", "data_warm", "ingest", "master", "ml", "remote_cluster_client", "transform"];
pub const ROLE_CHARS: &str = "cdfhilmrstw";
pub const EPHEMERAL_ID: &str = "rUst1cEph3m3ra1Id00000";
pub const TRANSPORT_ADDRESS: &str = "127.0.0.1:9300";

/// Node attributes (`_nodes`, `_cluster/state`, `_cat/nodeattrs`): the ones ES
/// registers on a default node (ML / transform / x-pack) plus the
/// `node.attr.testattr` the REST test cluster sets.
pub fn node_attributes() -> Value {
    static ATTRS: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    ATTRS.get_or_init(compute_node_attributes).clone()
}

fn compute_node_attributes() -> Value {
    let s = sysinfo::sys();
    let (_, heap_max) = sysinfo::heap(&s);
    json!({
        "ml.allocated_processors": s.cpus.to_string(),
        "ml.allocated_processors_double": format!("{:.1}", s.cpus as f64),
        "ml.config_version": "12.0.0",
        "ml.machine_memory": s.total_mem.to_string(),
        "ml.max_jvm_size": heap_max.to_string(),
        "testattr": "test",
        "transform.config_version": "10.0.0",
        "xpack.installed": "true"
    })
}

/// `component_versions` of `_nodes` info.
pub fn component_versions() -> Value {
    use elasticsearch_engine::version as v;
    json!({"api_key_version": v::API_KEY_VERSION, "ml_config_version": v::ML_CONFIG_VERSION, "transform_config_version": v::TRANSFORM_CONFIG_VERSION})
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// Usage counters (`_nodes/usage`, `_nodes/stats` http section)
// ---------------------------------------------------------------------------

static REST_ACTIONS: LazyLock<Mutex<BTreeMap<String, u64>>> = LazyLock::new(Default::default);
static AGG_USAGE: LazyLock<Mutex<BTreeMap<String, BTreeMap<String, u64>>>> = LazyLock::new(Default::default);
static USAGE_SINCE: LazyLock<i64> = LazyLock::new(elasticsearch_engine::date::now_millis);
static HTTP_TOTAL: AtomicU64 = AtomicU64::new(0);
static HTTP_CURRENT: AtomicU64 = AtomicU64::new(0);

/// HTTP connection accounting. Open records are sharded; completed records
/// have bounded FIFO retention. Direct in-process router calls use a temporary
/// channel; trusted cluster-internal dispatches do not create channels.
#[derive(Default, Clone)]
struct HttpClient {
    id: u64,
    opened: i64,
    closed: Option<i64>,
    last_request: i64,
    last_uri: String,
    agent: Option<String>,
    x_opaque_id: Option<String>,
    request_count: u64,
    request_size: u64,
    local_address: String,
    remote_address: String,
}

const CLIENT_SHARDS: usize = 64;
#[derive(Default)]
struct ClientShard {
    clients: HashMap<u64, HttpClient>,
}
static HTTP_CLIENTS: LazyLock<[Mutex<ClientShard>; CLIENT_SHARDS]> =
    LazyLock::new(|| std::array::from_fn(|_| Mutex::new(ClientShard::default())));
// HttpTransportSettings.SETTING_HTTP_CLIENT_STATS_MAX_CLOSED_CHANNEL_COUNT.
const MAX_CLOSED_CLIENTS: usize = 10000;
const CLOSED_CLIENT_MAX_AGE_MS: i64 = 5 * 60 * 1000;
#[derive(Default)]
struct ClosedClients {
    clients: VecDeque<HttpClient>,
}
impl ClosedClients {
    fn push(&mut self, client: HttpClient) {
        // O(1) at completion, never scan a request-rate-dependent history.
        if self.clients.len() == MAX_CLOSED_CLIENTS {
            self.clients.pop_front();
        }
        self.clients.push_back(client);
    }
    fn snapshot(&mut self, now: i64) -> Vec<HttpClient> {
        // Retain rather than stop at the first recent record: clock rollback
        // or out-of-order completions need not be ordered by wall time.
        self.clients.retain(|c| c.closed.is_some_and(|t| now.saturating_sub(t) <= CLOSED_CLIENT_MAX_AGE_MS));
        self.clients.iter().cloned().collect()
    }
}
static CLOSED_CLIENTS: LazyLock<Mutex<ClosedClients>> = LazyLock::new(Default::default);

/// Cloneable connection lifetime attached by Axum's per-connection service.
#[derive(Clone, Debug)]
pub struct HttpConnection(Arc<HttpConnectionState>);
#[derive(Debug)]
struct HttpConnectionState { id: u64 }
impl Drop for HttpConnectionState {
    fn drop(&mut self) { close_client(self.id); }
}
impl axum::extract::connect_info::Connected<axum::serve::IncomingStream<'_, tokio::net::TcpListener>> for HttpConnection {
    fn connect_info(stream: axum::serve::IncomingStream<'_, tokio::net::TcpListener>) -> Self {
        // ES `http.tcp.no_delay` (default true, from `network.tcp.no_delay`):
        // axum's accept loop leaves Nagle on, so a response written in more
        // than one segment can wait for the client's delayed ACK. Done here,
        // where each accepted connection passes once before it is served.
        let _ = stream.io().set_nodelay(true);
        Self::open(stream.io().local_addr().map(|a| a.to_string()).unwrap_or_default(), stream.remote_addr().to_string())
    }
}
/// HTTPS connections (`xpack.security.http.ssl.enabled`): the TCP socket
/// under the TLS session (no-delay was set before the handshake).
impl axum::extract::connect_info::Connected<axum::serve::IncomingStream<'_, crate::tls::TlsListener>> for HttpConnection {
    fn connect_info(stream: axum::serve::IncomingStream<'_, crate::tls::TlsListener>) -> Self {
        let local = stream.io().get_ref().0.local_addr().map(|a| a.to_string()).unwrap_or_default();
        Self::open(local, stream.remote_addr().to_string())
    }
}
impl HttpConnection {
    fn open(local_address: String, remote_address: String) -> Self {
        let id = HTTP_TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
        HTTP_CURRENT.fetch_add(1, Ordering::Relaxed);
        lock(&HTTP_CLIENTS[id as usize % CLIENT_SHARDS]).clients.insert(id, HttpClient {
            id, opened: elasticsearch_engine::date::now_millis(), local_address, remote_address,
            ..Default::default()
        });
        Self(Arc::new(HttpConnectionState { id }))
    }
}
fn close_client(id: u64) {
    HTTP_CURRENT.fetch_sub(1, Ordering::Relaxed);
    let client = lock(&HTTP_CLIENTS[id as usize % CLIENT_SHARDS]).clients.remove(&id);
    if let Some(mut client) = client {
        client.closed = Some(elasticsearch_engine::date::now_millis());
        lock(&CLOSED_CLIENTS).push(client);
    }
}

/// ES `HandlingTimeTracker` bucket upper bounds (millis).
const TIME_BOUNDS: [u64; 17] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536];
/// ES `HttpRouteStatsTracker` size histogram bucket upper bounds (bytes).
const SIZE_BOUNDS: [u64; 31] = {
    let mut b = [0u64; 31];
    let mut i = 0;
    while i < 31 {
        b[i] = 1u64 << i;
        i += 1;
    }
    b
};

#[derive(Clone)]
struct RouteStats {
    req_count: u64,
    req_bytes: u64,
    req_hist: [u64; 32],
    resp_count: u64,
    resp_bytes: u64,
    resp_hist: [u64; 32],
    time_hist: [u64; 18],
}

impl Default for RouteStats {
    fn default() -> Self {
        RouteStats { req_count: 0, req_bytes: 0, req_hist: [0; 32], resp_count: 0, resp_bytes: 0, resp_hist: [0; 32], time_hist: [0; 18] }
    }
}

static ROUTE_STATS: LazyLock<Mutex<BTreeMap<String, RouteStats>>> = LazyLock::new(Default::default);
static TRANSPORT_HIST: Mutex<[u64; 18]> = Mutex::new([0; 18]);

fn bucket(bounds: &[u64], v: u64) -> usize {
    bounds.iter().position(|b| v < *b).unwrap_or(bounds.len())
}

/// ES histogram rendering: the buckets from the first to the last non-empty one.
fn histogram_json(bounds: &[u64], counts: &[u64], ge: &str, lt: &str) -> Value {
    let first = counts.iter().position(|c| *c > 0);
    let last = counts.iter().rposition(|c| *c > 0);
    let (Some(first), Some(last)) = (first, last) else { return json!([]) };
    let mut out = Vec::new();
    for i in first..=last {
        let mut m = Map::new();
        if i > 0 {
            m.insert(ge.into(), json!(bounds[i - 1]));
        }
        if i < bounds.len() {
            m.insert(lt.into(), json!(bounds[i]));
        }
        m.insert("count".into(), json!(counts[i]));
        out.push(Value::Object(m));
    }
    Value::Array(out)
}

/// ES REST handlers: `(method, path template, handler name)`, extracted from
/// the 8.19 REST action classes.
static HANDLERS: LazyLock<Vec<(String, Vec<String>, String, String)>> = LazyLock::new(|| {
    let v: Vec<(String, String, String)> = serde_json::from_str(include_str!("data/rest_handlers.json")).unwrap_or_default();
    v.into_iter().map(|(m, p, n)| (m, p.split('/').filter(|s| !s.is_empty()).map(str::to_string).collect(), p, n)).collect()
});

/// The ES handler a request dispatches to: `(path template, handler name)`.
/// Explicit segments win over `{param}` ones (ES `PathTrie`); `HEAD` falls
/// back to the `GET` handler.
pub fn rest_handler(method: &Method, segs: &[String]) -> Option<(&'static str, &'static str)> {
    // The handlers by method and segment count, each with its explicit
    // segments as a bit mask (first segment highest): the lexicographic
    // comparison of the explicit-segment flags, without a scan of every
    // handler and an allocation per candidate on every request.
    static BY_SHAPE: LazyLock<HashMap<(String, usize), Vec<(usize, u64)>>> = LazyLock::new(|| {
        let mut m: HashMap<(String, usize), Vec<(usize, u64)>> = HashMap::new();
        for (i, (hm, hs, _, _)) in HANDLERS.iter().enumerate() {
            if hs.len() > 64 { continue; }
            let mask = hs.iter().enumerate().fold(0u64, |acc, (p, s)| if s.starts_with('{') { acc } else { acc | 1u64 << (63 - p) });
            m.entry((hm.clone(), hs.len())).or_default().push((i, mask));
        }
        m
    });
    let find = |m: &str| -> Option<(&'static str, &'static str)> {
        let candidates = BY_SHAPE.get(&(m.to_string(), segs.len()))?;
        let mut best: Option<(u64, usize)> = None;
        for &(i, mask) in candidates {
            let ok = HANDLERS[i].1.iter().zip(segs).all(|(p, s)| p.starts_with('{') || p == s);
            if ok && best.is_none_or(|(k, _)| mask > k) {
                best = Some((mask, i));
            }
        }
        best.map(|(_, i)| (HANDLERS[i].2.as_str(), HANDLERS[i].3.as_str()))
    };
    find(method.as_str()).or_else(|| if *method == Method::HEAD { find("GET") } else { None })
}

/// Tracks one request: the open-request gauge, its client channel and its
/// route's handling time / response size.
pub struct HttpGuard {
    _connection: HttpConnection,
    route: Option<&'static str>,
    start: std::time::Instant,
}

impl HttpGuard {
    /// Record the response of the request (size and handling time).
    pub fn done(&self, resp: &axum::response::Response) {
        use axum::body::HttpBody;
        let size = resp.body().size_hint().exact().unwrap_or(0);
        let ms = self.start.elapsed().as_millis() as u64;
        let tb = bucket(&TIME_BOUNDS, ms);
        lock(&TRANSPORT_HIST)[tb] += 1;
        if let Some(r) = self.route {
            let mut st = lock(&ROUTE_STATS);
            let e = st.entry(r.to_string()).or_default();
            e.resp_count += 1;
            e.resp_bytes += size;
            e.resp_hist[bucket(&SIZE_BOUNDS, size)] += 1;
            e.time_hist[tb] += 1;
        }
    }
}

/// Record one HTTP request (called for every request).
pub fn http_request(method: &Method, path: &str, uri: &str, headers: &axum::http::HeaderMap, size: u64, connection: Option<&HttpConnection>) -> HttpGuard {
    LazyLock::force(&USAGE_SINCE);
    let connection = connection.cloned().unwrap_or_else(|| HttpConnection::open("127.0.0.1:9200".into(), "127.0.0.1".into()));
    let id = connection.0.id;
    let now = elasticsearch_engine::date::now_millis();
    let segs: Vec<String> = path.split('/').filter(|s| !s.is_empty()).map(crate::decode_path_segment).collect();
    let route = rest_handler(method, &segs).map(|(p, _)| p);
    if let Some(r) = route {
        let mut st = lock(&ROUTE_STATS);
        let e = st.entry(r.to_string()).or_default();
        e.req_count += 1;
        e.req_bytes += size;
        e.req_hist[bucket(&SIZE_BOUNDS, size)] += 1;
    }
    {
        let shard = &HTTP_CLIENTS[(id as usize) % CLIENT_SHARDS];
        let mut cs = lock(shard);
        if let Some(client) = cs.clients.get_mut(&id) {
            client.last_request = now;
            client.last_uri = uri.to_string();
            client.agent = headers.get("user-agent").and_then(|v| v.to_str().ok()).map(str::to_string);
            client.x_opaque_id = headers.get("x-opaque-id").and_then(|v| v.to_str().ok()).map(str::to_string);
            client.request_count += 1;
            client.request_size += size;
        }
    }
    HttpGuard { _connection: connection, route, start: std::time::Instant::now() }
}

/// Count the request under its ES REST handler name (`_nodes/usage`).
pub fn record_rest(req: &Req) {
    if let Some((_, name)) = rest_handler(&req.method, &req.segs) {
        *lock(&REST_ACTIONS).entry(name.to_string()).or_insert(0) += 1;
    }
}

/// Search request usage (`_cluster/stats` `indices.search`): searches, query
/// types, body sections, rescorers and retrievers, each counted once per request.
#[derive(Default)]
struct SearchUsage {
    total: u64,
    queries: BTreeMap<String, u64>,
    sections: BTreeMap<String, u64>,
    rescorers: BTreeMap<String, u64>,
    retrievers: BTreeMap<String, u64>,
    /// `ExtendedSearchUsageStats`: category -> name -> field -> count.
    extended: BTreeMap<String, BTreeMap<String, BTreeMap<String, u64>>>,
}

/// `RetrieverBuilder.getExtendedUsageFields` over a retriever tree (each
/// field counted once per request).
fn retriever_extended(r: &Value, out: &mut BTreeSet<(String, String)>) {
    let Some(o) = r.as_object() else { return };
    for (kind, body) in o {
        let Some(b) = body.as_object() else { continue };
        if kind == "text_similarity_reranker" && b.get("chunk_rescorer").is_some_and(|c| !c.is_null()) {
            out.insert((kind.clone(), "chunk_rescorer".into()));
        }
        if let Some(c) = b.get("retriever") {
            retriever_extended(c, out);
        }
        for c in b.get("retrievers").and_then(Value::as_array).into_iter().flatten() {
            retriever_extended(c.get("retriever").unwrap_or(c), out);
        }
    }
}

static SEARCH_USAGE: LazyLock<Mutex<SearchUsage>> = LazyLock::new(Default::default);

const SEARCH_SECTIONS: &[&str] = &[
    "query", "post_filter", "knn", "rank", "highlight", "suggest", "rescore", "sort", "collapse", "search_after", "pit", "runtime_mappings", "script_fields",
    "fields", "docvalue_fields", "stored_fields", "_source", "slice", "retriever", "indices_boost", "stats", "ext", "min_score",
];

/// Query types used in a query tree.
fn query_types(q: &Value, out: &mut BTreeSet<String>) {
    let Some(o) = q.as_object() else { return };
    for (kind, body) in o {
        out.insert(kind.clone());
        let Some(b) = body.as_object() else { continue };
        for (k, v) in b {
            match (k.as_str(), v) {
                ("must" | "should" | "filter" | "must_not" | "queries", Value::Array(a)) => a.iter().for_each(|x| query_types(x, out)),
                ("must" | "should" | "filter" | "must_not" | "query" | "positive" | "negative" | "organic", Value::Object(_)) => query_types(v, out),
                _ => {}
            }
        }
    }
}

fn record_search_usage(body: &Value) {
    let mut u = lock(&SEARCH_USAGE);
    u.total += 1;
    let Some(o) = body.as_object() else { return };
    let mut qs = BTreeSet::new();
    for (k, v) in o {
        let sec = if k == "aggregations" { "aggs" } else { k.as_str() };
        if sec == "aggs" || SEARCH_SECTIONS.contains(&sec) {
            *u.sections.entry(sec.to_string()).or_default() += 1;
        }
        match k.as_str() {
            "query" | "post_filter" => query_types(v, &mut qs),
            "rescore" => {
                let list: Vec<&Value> = match v {
                    Value::Array(a) => a.iter().collect(),
                    other => vec![other],
                };
                for r in list {
                    if let Some(ro) = r.as_object() {
                        for rk in ro.keys().filter(|k| *k != "window_size") {
                            *u.rescorers.entry(rk.clone()).or_default() += 1;
                        }
                        if let Some(q) = ro.get("query").and_then(|q| q.get("rescore_query")) {
                            query_types(q, &mut qs);
                        }
                    }
                }
            }
            "retriever" => {
                if let Some(ro) = v.as_object() {
                    for rk in ro.keys() {
                        *u.retrievers.entry(rk.clone()).or_default() += 1;
                    }
                }
                let mut ext = BTreeSet::new();
                retriever_extended(v, &mut ext);
                for (name, field) in ext {
                    *u.extended.entry("retrievers".into()).or_default().entry(name).or_default().entry(field).or_default() += 1;
                }
            }
            _ => {}
        }
    }
    for q in qs {
        *u.queries.entry(q).or_default() += 1;
    }
}

pub fn search_usage_json() -> Value {
    let u = lock(&SEARCH_USAGE);
    json!({"total": u.total, "queries": u.queries, "rescorers": u.rescorers, "sections": u.sections, "retrievers": u.retrievers, "extended": u.extended})
}

/// ES `ValuesSourceType` name of a mapped field type (`_nodes/usage`).
fn values_source_type(ty: elasticsearch_engine::mapping::EsType) -> &'static str {
    use elasticsearch_engine::mapping::EsType as T;
    match ty {
        _ if ty.is_numeric() => "numeric",
        T::Date | T::DateNanos => "date",
        T::Boolean => "boolean",
        T::Ip => "ip",
        T::GeoPoint => "geopoint",
        T::GeoShape => "geoshape",
        T::Point => "point",
        T::Shape => "shape",
        T::Histogram => "histogram",
        T::AggregateMetricDouble => "aggregate_metric",
        T::IntegerRange | T::LongRange | T::FloatRange | T::DoubleRange | T::DateRange | T::IpRange => "range",
        _ if ty.is_keyword_like() => "keyword",
        _ => "keyword",
    }
}

/// Count aggregation usage of a search body (`_nodes/usage` `aggregations`),
/// keyed by aggregation type and values-source type of its field.
pub fn record_aggs(node: &Node, index: Option<&str>, body: &Value) {
    record_search_usage(body);
    let spec = body.get("aggs").or_else(|| body.get("aggregations"));
    let Some(spec) = spec else { return };
    let lookups: Vec<_> = {
        let mut o = ResolveOpts::default();
        o.ignore_unavailable = true;
        o.allow_no_indices = true;
        node.resolve(index.unwrap_or("_all"), o).map(|t| t.iter().map(|t| t.svc.lookup()).collect()).unwrap_or_default()
    };
    let field_vs = |f: &str| -> &'static str {
        lookups.iter().find_map(|l| l.get(f).map(|rf| values_source_type(rf.ty))).unwrap_or("other")
    };
    fn walk(spec: &Value, out: &mut BTreeMap<String, BTreeMap<String, u64>>, field_vs: &dyn Fn(&str) -> &'static str) {
        let Some(m) = spec.as_object() else { return };
        for (_, def) in m {
            let Some(d) = def.as_object() else { continue };
            for (k, v) in d {
                match k.as_str() {
                    "aggs" | "aggregations" => walk(v, out, field_vs),
                    "meta" => {}
                    ty => {
                        let vs = match v.get("field").and_then(Value::as_str) {
                            Some(f) => field_vs(f),
                            None => "other",
                        };
                        *out.entry(ty.to_string()).or_default().entry(vs.to_string()).or_insert(0) += 1;
                    }
                }
            }
        }
    }
    walk(spec, &mut lock(&AGG_USAGE), &field_vs);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Levenshtein distance.
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Candidates close to `word`, best first.
pub fn suggestions<'a>(word: &str, candidates: &[&'a str]) -> Vec<&'a str> {
    let mut scored: Vec<(f64, &str)> = candidates
        .iter()
        .map(|c| (1.0 - edit_distance(word, c) as f64 / word.len().max(c.len()).max(1) as f64, *c))
        .filter(|(s, _)| *s >= 0.5)
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(b.1)));
    scored.into_iter().map(|x| x.1).collect()
}

/// ES `SuggestingErrorOnUnknown`-style suffix: ` did you mean [x]?`.
pub fn did_you_mean(word: &str, candidates: &[&str]) -> String {
    let mut scored: Vec<(f64, &str)> = candidates
        .iter()
        .map(|c| (1.0 - edit_distance(word, c) as f64 / word.len().max(c.len()).max(1) as f64, *c))
        .filter(|(s, _)| *s >= 0.5)
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(b.1)));
    match scored.len() {
        0 => String::new(),
        1 => format!(" did you mean [{}]?", scored[0].1),
        _ => format!(" did you mean any of [{}]?", scored.iter().map(|x| x.1).collect::<Vec<_>>().join(", ")),
    }
}

/// ES `TimeValue` rendering (`1.2s`, `15ms`, `3.5m`).
pub fn time_fmt(ms: u64) -> String {
    let units = [("d", 86_400_000u64), ("h", 3_600_000), ("m", 60_000), ("s", 1000)];
    for (u, n) in units {
        if ms >= n {
            let v = ms as f64 / n as f64;
            let t = format!("{:.1}", v);
            return format!("{}{u}", t.strip_suffix(".0").unwrap_or(&t));
        }
    }
    format!("{ms}ms")
}

/// `?human` is applied centrally (`util::humanize` in `finish`).
fn finish_json(_req: &Req, v: Value) -> R {
    Ok(Resp::ok(v))
}

/// Does a `_nodes/{node_id}` spec select this node?
pub fn node_matches(node: &Node, spec: &str) -> bool {
    spec.split(',').map(str::trim).any(|p| {
        if p.is_empty() || matches!(p, "_all" | "_local" | "_master" | "*") {
            return true;
        }
        if let Some((k, v)) = p.split_once(':') {
            let truthy = v == "true" || v == "*";
            return match k {
                "master" | "data" | "ingest" | "ml" | "transform" | "remote_cluster_client" => truthy,
                "voting_only" | "coordinating_only" => !truthy,
                _ => ROLES.contains(&k) && truthy || node_attributes().get(k).and_then(Value::as_str).map(|x| elasticsearch_engine::mapping::wildcard_match(v, x)).unwrap_or(false),
            };
        }
        [NODE_ID, node.name.as_str(), "127.0.0.1", TRANSPORT_ADDRESS].iter().any(|c| elasticsearch_engine::mapping::wildcard_match(p, c))
    })
}

fn nodes_header(n: usize) -> Value {
    json!({"total": n, "successful": n, "failed": 0})
}

fn opts(req: &Req) -> ResolveOpts {
    ResolveOpts::from_params(&req.params)
}

pub(crate) fn health_of_svc(svc: &IndexService) -> &'static str {
    if svc.setting_usize("index.number_of_replicas", 1) > 0 { "yellow" } else { "green" }
}

fn node_ident(node: &Node) -> Value {
    json!({
        "name": node.name,
        "ephemeral_id": EPHEMERAL_ID,
        "transport_address": TRANSPORT_ADDRESS,
        "external_id": node.name,
        "attributes": node_attributes(),
        "roles": ROLES,
        "version": ES_VERSION,
        "min_index_version": elasticsearch_engine::version::IndexVersion::MINIMUM_COMPATIBLE.id(),
        "max_index_version": elasticsearch_engine::version::IndexVersion::CURRENT.id()
    })
}

fn check_get(req: &Req, allowed: &[&str]) -> Option<R> {
    let m = req.method.as_str();
    if allowed.contains(&m) {
        None
    } else {
        Some(crate::method_not_allowed(req, allowed))
    }
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

/// `_nodes/...` monitoring endpoints (`stats`, `hot_threads`, `usage`).
pub async fn handle_nodes(node: Arc<Node>, req: Req) -> R {
    if let Some(r) = check_get(&req, &["GET"]) {
        return r;
    }
    let kind = req.segs.iter().skip(1).take(2).find(|s| matches!(s.as_str(), "stats" | "hot_threads" | "hotthreads" | "usage")).cloned().unwrap_or_default();
    blocking(move || match kind.as_str() {
        "stats" => nodes_stats(&node, &req),
        "usage" => nodes_usage(&node, &req),
        _ => hot_threads(&node, &req),
    })
    .await
}

/// `_cluster/...` endpoints owned here.
pub async fn handle_cluster(node: Arc<Node>, req: Req) -> R {
    let b = req.seg(1).to_string();
    blocking(move || match b.as_str() {
        "stats" => match check_get(&req, &["GET"]) {
            Some(r) => r,
            None => cluster_stats(&node, &req),
        },
        "allocation" if req.seg(2) == "explain" => match check_get(&req, &["GET", "POST"]) {
            Some(r) => r,
            None => allocation_explain(&node, &req),
        },
        "reroute" => match check_get(&req, &["POST"]) {
            Some(r) => r,
            None => reroute(&node, &req),
        },
        _ => crate::not_found_handler(&req),
    })
    .await
}

/// Top-level `/_xxx` endpoints owned here.
pub async fn handle_global(node: Arc<Node>, req: Req) -> R {
    let a = req.seg(0).to_string();
    match a.as_str() {
        "_validate" | "_terms_enum" | "_explain" => crate::obs_search::handle(node, req, None).await,
        _ => blocking(move || match a.as_str() {
            "_migration" if req.seg(1) == "deprecations" => match check_get(&req, &["GET"]) {
                Some(r) => r,
                None => deprecations(&node, &req, None),
            },
            "_health_report" => match check_get(&req, &["GET"]) {
                Some(r) => r,
                None => health_report(&node, &req),
            },
            "_recovery" => match check_get(&req, &["GET"]) {
                Some(r) => r,
                None => recovery(&node, &req, None),
            },
            "_shard_stores" => match check_get(&req, &["GET"]) {
                Some(r) => r,
                None => shard_stores(&node, &req, None),
            },
            "_search_shards" => match check_get(&req, &["GET", "POST"]) {
                Some(r) => r,
                None => search_shards(&node, &req, None),
            },
            "_resolve" if req.seg(1) == "index" => match check_get(&req, &["GET"]) {
                Some(r) => r,
                None => resolve_index(&node, &req),
            },
            "_resolve" if req.seg(1) == "cluster" => match check_get(&req, &["GET"]) {
                Some(r) => r,
                None => resolve_cluster(&node, &req),
            },
            "_cache" if req.seg(1) == "clear" => match check_get(&req, &["POST"]) {
                Some(r) => r,
                None => clear_cache(&node, &req, None),
            },
            "_segments" => match check_get(&req, &["GET"]) {
                Some(r) => r,
                None => segments_global(&node, &req, None),
            },
            "_internal" if req.seg(1) == "desired_balance" && req.method == Method::DELETE => Ok(Resp::text(String::new())),
            "_internal" if req.seg(1) == "desired_balance" => match check_get(&req, &["GET"]) {
                Some(r) => r,
                None => desired_balance(&node, &req),
            },
            "_internal" if req.seg(1) == "desired_nodes" => crate::desired_nodes::handle(&node, &req),
            "_internal" if req.seg(1) == "prevalidate_node_removal" => prevalidate_node_removal(&node, &req),
            "_remote" if req.seg(1) == "info" => match check_get(&req, &["GET"]) {
                Some(r) => r,
                None => remote_info(&node, &req),
            },
            _ => crate::not_found_handler(&req),
        })
        .await,
    }
}

/// `/{index}/_xxx` endpoints owned here.
pub async fn handle_index(node: Arc<Node>, req: Req, index: String) -> R {
    let api = req.seg(1).to_string();
    match api.as_str() {
        "_validate" | "_terms_enum" | "_explain" => crate::obs_search::handle(node, req, Some(index)).await,
        _ => blocking(move || {
            let idx = Some(index.as_str());
            match api.as_str() {
                "_migration" if req.seg(2) == "deprecations" => check_get(&req, &["GET"]).unwrap_or_else(|| deprecations(&node, &req, idx)),
                "_recovery" => check_get(&req, &["GET"]).unwrap_or_else(|| recovery(&node, &req, idx)),
                "_segments" => check_get(&req, &["GET"]).unwrap_or_else(|| segments_global(&node, &req, idx)),
                "_shard_stores" => check_get(&req, &["GET"]).unwrap_or_else(|| shard_stores(&node, &req, idx)),
                "_search_shards" => check_get(&req, &["GET", "POST"]).unwrap_or_else(|| search_shards(&node, &req, idx)),
                "_disk_usage" => check_get(&req, &["POST"]).unwrap_or_else(|| disk_usage(&node, &req, &index)),
                "_field_usage_stats" => check_get(&req, &["GET"]).unwrap_or_else(|| field_usage_stats(&node, &req, &index)),
                "_block" => check_get(&req, &["PUT", "DELETE"]).unwrap_or_else(|| add_block(&node, &req, &index)),
                "_reload_search_analyzers" => check_get(&req, &["GET", "POST"]).unwrap_or_else(|| reload_search_analyzers(&node, &req, &index)),
                "_cache" if req.seg(2) == "clear" => check_get(&req, &["POST"]).unwrap_or_else(|| clear_cache(&node, &req, idx)),
                _ => crate::not_found_handler(&req),
            }
        })
        .await,
    }
}

// ---------------------------------------------------------------------------
// _nodes/stats
// ---------------------------------------------------------------------------

const NODE_METRICS: &[&str] = &[
    "indices", "os", "process", "jvm", "thread_pool", "fs", "transport", "http", "breaker", "script", "discovery", "ingest", "adaptive_selection", "script_cache",
    "indexing_pressure", "repositories", "allocations",
];
const INDEX_METRICS: &[&str] = &[
    "docs", "store", "indexing", "get", "search", "merge", "refresh", "flush", "warmer", "query_cache", "fielddata", "completion", "segments", "translog", "request_cache",
    "recovery", "bulk", "shard_stats", "dense_vector", "sparse_vector", "mappings",
];

fn index_metric_key(m: &str) -> &str {
    if m == "merge" { "merges" } else { m }
}

/// Parse and validate a comma-separated metric list against `known`.
fn parse_metrics(path: &str, spec: Option<&str>, known: &[&str], what: &str) -> Result<Option<Vec<String>>, EsError> {
    let Some(spec) = spec else { return Ok(None) };
    let list: Vec<String> = spec.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect();
    if list.iter().any(|m| m == "_all") {
        if list.len() > 1 {
            return Err(EsError::illegal_argument(format!("request [{path}] contains _all and individual {what}s [{}]", list.join(","))));
        }
        return Ok(None);
    }
    let bad: Vec<&String> = list.iter().filter(|m| !known.contains(&m.as_str())).collect();
    if !bad.is_empty() {
        let mut msg = if bad.len() == 1 {
            format!("request [{path}] contains unrecognized {what}: [{}]", bad[0])
        } else {
            format!("request [{path}] contains unrecognized {what}s: [{}]", bad.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))
        };
        for b in &bad {
            match suggestions(b, known).as_slice() {
                [] => {}
                [one] => msg.push_str(&format!(" -> did you mean [{one}]?")),
                many => msg.push_str(&format!(" -> did you mean any of [{}]?", many.join(", "))),
            }
        }
        return Err(EsError::illegal_argument(msg));
    }
    Ok(Some(list))
}

/// Zero-valued index stats (every section `_nodes/stats` reports, ES order).
fn zero_index_section() -> Value {
    json!({
        "docs": {"count": 0, "deleted": 0, "total_size_in_bytes": 0},
        "shard_stats": {"total_count": 0},
        "store": {"size_in_bytes": 0, "total_data_set_size_in_bytes": 0, "reserved_in_bytes": 0},
        "indexing": {"index_total": 0, "index_time_in_millis": 0, "index_current": 0, "index_failed": 0, "index_failed_due_to_version_conflict": 0, "delete_total": 0,
                     "delete_time_in_millis": 0, "delete_current": 0, "noop_update_total": 0, "is_throttled": false, "throttle_time_in_millis": 0, "write_load": 0.0},
        "get": {"total": 0, "time_in_millis": 0, "exists_total": 0, "exists_time_in_millis": 0, "missing_total": 0, "missing_time_in_millis": 0, "current": 0},
        "search": {"open_contexts": 0, "query_total": 0, "query_time_in_millis": 0, "query_current": 0, "query_failure": 0, "fetch_total": 0, "fetch_time_in_millis": 0, "fetch_current": 0,
                   "fetch_failure": 0, "scroll_total": 0, "scroll_time_in_millis": 0, "scroll_current": 0, "suggest_total": 0, "suggest_time_in_millis": 0, "suggest_current": 0,
                   "recent_search_load": 0.0},
        "merges": {"current": 0, "current_docs": 0, "current_size_in_bytes": 0, "total": 0, "total_time_in_millis": 0, "total_docs": 0, "total_size_in_bytes": 0,
                   "total_stopped_time_in_millis": 0, "total_throttled_time_in_millis": 0, "total_auto_throttle_in_bytes": 0},
        "refresh": {"total": 0, "total_time_in_millis": 0, "external_total": 0, "external_total_time_in_millis": 0, "listeners": 0},
        "flush": {"total": 0, "periodic": 0, "total_time_in_millis": 0, "total_time_excluding_waiting_on_lock_in_millis": 0},
        "warmer": {"current": 0, "total": 0, "total_time_in_millis": 0},
        "query_cache": {"memory_size_in_bytes": 0, "total_count": 0, "hit_count": 0, "miss_count": 0, "cache_size": 0, "cache_count": 0, "evictions": 0},
        "fielddata": {"memory_size_in_bytes": 0, "evictions": 0, "global_ordinals": {"build_time_in_millis": 0}},
        "completion": {"size_in_bytes": 0},
        "segments": {"count": 0, "memory_in_bytes": 0, "terms_memory_in_bytes": 0, "stored_fields_memory_in_bytes": 0, "term_vectors_memory_in_bytes": 0,
                     "norms_memory_in_bytes": 0, "points_memory_in_bytes": 0, "doc_values_memory_in_bytes": 0, "index_writer_memory_in_bytes": 0,
                     "version_map_memory_in_bytes": 0, "fixed_bit_set_memory_in_bytes": 0, "max_unsafe_auto_id_timestamp": -1, "file_sizes": {}},
        "translog": {"operations": 0, "size_in_bytes": 0, "uncommitted_operations": 0, "uncommitted_size_in_bytes": 0, "earliest_last_modified_age": 0},
        "request_cache": {"memory_size_in_bytes": 0, "evictions": 0, "hit_count": 0, "miss_count": 0},
        "recovery": {"current_as_source": 0, "current_as_target": 0, "throttle_time_in_millis": 0},
        "bulk": {"total_operations": 0, "total_time_in_millis": 0, "total_size_in_bytes": 0, "avg_time_in_millis": 0, "avg_size_in_bytes": 0},
        "mappings": {"total_count": 0, "total_estimated_overhead_in_bytes": 0, "total_segments": 0, "total_segment_fields": 0, "average_fields_per_segment": 0},
        "dense_vector": elasticsearch_engine::vector_stats::zero(),
        "sparse_vector": {"value_count": 0}
    })
}

/// Segment file sizes by extension (`include_segment_file_sizes`).
pub(crate) fn segment_file_sizes(svc: &IndexService) -> Value {
    fn walk(p: &std::path::Path, out: &mut BTreeMap<String, Vec<u64>>) {
        if let Ok(rd) = std::fs::read_dir(p) {
            for e in rd.flatten() {
                let path = e.path();
                if path.is_dir() {
                    if path.file_name().map(|n| n != "translog").unwrap_or(true) {
                        walk(&path, out);
                    }
                } else if let Ok(md) = e.metadata() {
                    let ext = path.extension().and_then(|x| x.to_str()).unwrap_or("other").to_string();
                    out.entry(ext).or_default().push(md.len());
                }
            }
        }
    }
    let mut sizes = BTreeMap::new();
    for sh in svc.local_shards() {
        walk(sh.shard.path(), &mut sizes);
    }
    let mut m = Map::new();
    for (ext, files) in sizes {
        // Lucene names the segment info file extension `si`.
        let (key, desc) = match ext.as_str() {
            "json" => ("si", "Segment Info"),
            "tim" | "terms" => ("tim", "Term Dictionary"),
            "doc" | "post" | "postings" => ("doc", "Frequencies"),
            "pos" => ("pos", "Positions"),
            "fdt" | "store" => ("fdt", "Field Data"),
            "dvd" | "fast" => ("dvd", "DocValues"),
            "nvd" | "norms" => ("nvd", "Norms"),
            "vec" | "hnsw" => ("vec", "Vector Data"),
            _ => (ext.as_str(), "Others"),
        };
        let total: u64 = files.iter().sum();
        let n = files.len() as u64;
        let e = m.entry(key.to_string()).or_insert_with(|| json!({"description": desc, "size_in_bytes": 0, "min_size_in_bytes": u64::MAX, "max_size_in_bytes": 0, "average_size_in_bytes": 0, "count": 0}));
        let (cur_total, cur_n) = (e["size_in_bytes"].as_u64().unwrap_or(0) + total, e["count"].as_u64().unwrap_or(0) + n);
        e["size_in_bytes"] = json!(cur_total);
        e["count"] = json!(cur_n);
        e["min_size_in_bytes"] = json!(e["min_size_in_bytes"].as_u64().unwrap_or(u64::MAX).min(files.iter().copied().min().unwrap_or(0)));
        e["max_size_in_bytes"] = json!(e["max_size_in_bytes"].as_u64().unwrap_or(0).max(files.iter().copied().max().unwrap_or(0)));
        e["average_size_in_bytes"] = json!(if cur_n > 0 { cur_total / cur_n } else { 0 });
    }
    Value::Object(m)
}

/// One index's stats: `_stats` section plus the sections node stats adds.
pub(crate) fn full_index_section(node: &Node, svc: &IndexService, file_sizes_on: bool, shard_level: bool) -> Value {
    selected_index_section(node, svc, file_sizes_on, shard_level, false, true)
}

// Keep the complete path for default/mixed metrics and per-shard extras. A
// counter-only node/index request must not enter core or translog locks.
fn selected_index_section(node: &Node, svc: &IndexService, file_sizes_on: bool, shard_level: bool, counters_only: bool, translog: bool) -> Value {
    let mut s = zero_index_section();
    if !counters_only {
        s["dense_vector"] = elasticsearch_engine::vector_stats::index_stats(svc, false);
    }
    let base = if counters_only { crate::indices::counter_index_section(svc) } else { crate::indices::index_stats_selected(svc, translog) };
    if let (Value::Object(dst), Value::Object(src)) = (&mut s, base) {
        for (k, v) in src {
            match (dst.get_mut(&k), v) {
                (Some(Value::Object(d)), Value::Object(sv)) => {
                    for (kk, vv) in sv {
                        d.insert(kk, vv);
                    }
                }
                (_, v) => {
                    dst.insert(k, v);
                }
            }
        }
    }
    for k in ["fielddata", "completion"] {
        if let Some(o) = s[k].as_object_mut() {
            o.remove("fields");
        }
    }
    let open_scrolls = node.scrolls.lock().values().filter(|sc| sc.targets.iter().any(|t| t.svc.name == svc.name)).count();
    let open_pits = node.pits.lock().values().filter(|p| p.targets.iter().any(|t| t.svc.name == svc.name)).count();
    s["search"]["open_contexts"] = json!(open_scrolls + open_pits);
    s["search"]["scroll_current"] = json!(open_scrolls);
    s["shard_stats"] = json!({"total_count": svc.local_shards().len().max(1)});
    if file_sizes_on && !counters_only {
        s["segments"]["file_sizes"] = segment_file_sizes(svc);
    }
    if shard_level {
        if let Some(o) = s.as_object_mut() {
            o.remove("mappings");
        }
    } else if !counters_only {
        let n = svc.lookup().fields.len() as u64;
        let segs = svc.num_segments() as u64;
        s["mappings"] = json!({"total_count": n, "total_estimated_overhead_in_bytes": n * 1024, "total_segments": segs, "total_segment_fields": n * segs,
                               "average_fields_per_segment": if segs > 0 { n } else { 0 }});
    }
    s
}

fn filter_sections(v: &mut Value, keep: &Option<Vec<String>>) {
    if let (Some(k), Some(o)) = (keep, v.as_object_mut()) {
        let keys: Vec<&str> = k.iter().map(|m| index_metric_key(m)).collect();
        o.retain(|key, _| keys.contains(&key.as_str()));
    }
}

/// Thread pool definition (ES 8 defaults for `cpus` processors).
#[allow(dead_code)]
pub struct Pool {
    pub name: &'static str,
    pub ty: &'static str,
    pub core: usize,
    pub max: usize,
    pub queue_size: i64,
    pub keep_alive: Option<&'static str>,
}

/// The node's thread pools with ES 9.5.5's sizing (no `search_throttled` since 9.0) (`ThreadPool` built-ins plus
/// the ML, watcher, CCR, rollup, downsample, searchable-snapshot, Azure and
/// profiling plugin pools), sorted by name.
pub fn pools() -> Vec<Pool> {
    let n = crate::sysinfo::processors();
    let half = n.div_ceil(2).max(1);
    let half5 = half.min(5);
    let half10 = half.min(10);
    let search = n * 3 / 2 + 1;
    let eighth = (n / 8).max(1);
    // `sysinfo::heap(..).1`, without sampling the process (`sysinfo::sys`
    // reads /proc and lists every open descriptor) for a configured figure.
    let heap_max = elasticsearch_engine::memory::es_heap();
    let snapshot_max = if heap_max >= 750 << 20 { 10 } else { n.min(5) };
    let fixed = |name, size: usize, q| Pool { name, ty: "fixed", core: size.max(1), max: size.max(1), queue_size: q, keep_alive: None };
    let scaling = |name, core: usize, max: usize, ka| Pool { name, ty: "scaling", core, max: max.max(1), queue_size: -1, keep_alive: Some(ka) };
    let mut v = vec![
        fixed("analyze", 1, 16),
        fixed("auto_complete", (n / 4).max(1), 100),
        scaling("azure_event_loop", 0, (n / 4).max(1), "30s"),
        fixed("ccr", 32, 100),
        fixed("cluster_coordination", 1, -1),
        // `DLMFrozenTransitionPlugin`: a scale-down fixed pool (keep_alive 10m).
        Pool { name: "dlm_frozen_transition", ty: "fixed", core: (2 * n).min(100), max: (2 * n).min(100), queue_size: (20 * n).min(1000) as i64, keep_alive: Some("10m") },
        fixed("downsample_indexing", eighth, 256),
        // `EsqlPlugin.workerQueueSize`: max((heap MB + 400 * processors) / 2, 1000).
        fixed("esql_worker", search, (((heap_max >> 20) as i64 + 400 * n as i64) / 2).max(1000)),
        scaling("esql_external_io", 0, (3 * n).clamp(16, 100), "30s"),
        scaling("fetch_shard_started", 1, 2 * n, "5m"),
        scaling("fetch_shard_store", 1, 2 * n, "5m"),
        scaling("flush", 1, half5, "5m"),
        fixed("force_merge", eighth, -1),
        scaling("generic", 4, (4 * n).clamp(128, 512), "30s"),
        fixed("get", search, 1000),
        scaling("inference_response", 0, 10, "10m"),
        scaling("inference_utility", 0, 10, "10m"),
        scaling("management", 1, n.clamp(1, 5), "5m"),
        scaling("merge", 1, n, "5m"),
        scaling("ml_datafeed", 1, 512, "1m"),
        scaling("ml_job_comms", 4, 2048, "1m"),
        scaling("ml_native_inference_comms", 3, 348, "1m"),
        scaling("ml_utility", 1, 2048, "10m"),
        fixed("model_download", 5, -1),
        scaling("profiling", 0, 1, "30m"),
        scaling("refresh", 1, half10, "5m"),
        scaling("repository_azure", 0, 5, "30s"),
        fixed("rollup_indexing", 1, -1),
        fixed("search", search, (search * 1000) as i64),
        fixed("search_coordination", half, 1000),
        scaling("searchable_snapshots_cache_fetch_async", 0, 3 * n, "30s"),
        scaling("searchable_snapshots_cache_prewarming", 0, n, "30s"),
        scaling("snapshot", 1, snapshot_max, "5m"),
        scaling("snapshot_meta", 1, (3 * n).min(50), "30s"),
        fixed("system_critical_read", half5, 2000),
        fixed("system_critical_write", half5, 1500),
        fixed("system_read", half5, 2000),
        fixed("system_write", half5, 1000),
        fixed("system_write_coordination", half5, 1000),
        scaling("warmer", 1, half5, "5m"),
        fixed("watcher", (5 * n).min(50), 1000),
        fixed("write", n, (n * 750).max(10000) as i64),
        fixed("write_coordination", n, 10000),
    ];
    v.sort_by_key(|p| p.name);
    v
}

/// `_nodes` info `thread_pool` section.
pub fn thread_pool_info() -> Value {
    let mut m = Map::new();
    for p in pools() {
        let v = if p.ty == "fixed" {
            match p.keep_alive {
                Some(ka) => json!({"type": "fixed", "size": p.max, "keep_alive": ka, "queue_size": p.queue_size}),
                None => json!({"type": "fixed", "size": p.max, "queue_size": p.queue_size}),
            }
        } else {
            json!({"type": "scaling", "core": p.core, "max": p.max, "keep_alive": p.keep_alive.unwrap_or("5m"), "queue_size": p.queue_size})
        };
        m.insert(p.name.to_string(), v);
    }
    Value::Object(m)
}

/// The shard counters a pool's (active, completed) figures come from,
/// summed in one pass over the node's indices (the thread pool section asks
/// for every pool; it walked every index per pool before).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PoolCounters {
    /// (active, completed) of `search`, `write` and `get`.
    search: (u64, u64),
    write: (u64, u64),
    get: (u64, u64),
    refresh: u64,
    flush: u64,
    force_merge: u64,
}

impl PoolCounters {
    /// (active, completed) of pool `name`.
    fn activity(&self, name: &str) -> (u64, u64) {
        match name {
            "search" => self.search,
            "write" => self.write,
            "get" => self.get,
            "refresh" => (0, self.refresh),
            "flush" => (0, self.flush),
            "force_merge" => (0, self.force_merge),
            "management" => (0, HTTP_TOTAL.load(Ordering::Relaxed)),
            _ => (0, 0),
        }
    }
}

fn pool_counters(node: &Node) -> PoolCounters {
    use elasticsearch_engine::shard::Counters as C;
    let mut p = PoolCounters::default();
    for svc in node.all_indices() {
        let c = &svc.counters();
        p.search.0 += C::get(&c.query_current);
        p.search.1 += C::get(&c.search_total);
        p.write.0 += C::get(&c.index_current) + C::get(&c.delete_current);
        p.write.1 += C::get(&c.index_total) + C::get(&c.delete_total);
        p.get.0 += C::get(&c.get_current);
        p.get.1 += C::get(&c.get_total);
        p.refresh += C::get(&c.refresh_total);
        p.flush += C::get(&c.flush_total);
        p.force_merge += C::get(&c.merge_total);
    }
    p
}

/// Pool thread count as ES reports it (fixed: size; scaling: core, grown).
pub fn pool_threads(p: &Pool, completed: u64) -> usize {
    // ES pools start threads lazily: a fixed pool grows one thread per task
    // up to its size; a scaling pool keeps its core threads.
    if p.ty == "fixed" {
        (completed as usize).min(p.max)
    } else if completed > 0 {
        p.core.max(1).min(p.max)
    } else {
        p.core
    }
}

pub(crate) fn thread_pool_section(node: &Node) -> Value {
    let mut m = Map::new();
    let counters = pool_counters(node);
    for p in pools() {
        let (active, completed) = counters.activity(p.name);
        let threads = pool_threads(&p, completed);
        // Queue depth and rejections of the executors engine work runs on.
        let live = crate::pools::Pool::ALL.iter().find(|q| q.name() == p.name).map(|q| crate::pools::executor(*q).stats());
        // The executors engine work runs on report their real figures.
        let (threads, queue, active, rejected, completed) = match live {
            Some(s) => (s.threads, s.queue, s.active as u64, s.rejected, s.completed),
            None => (threads, 0, active, 0, completed),
        };
        m.insert(p.name.into(), json!({"threads": threads, "queue": queue, "active": active, "rejected": rejected, "largest": threads, "completed": completed}));
    }
    Value::Object(m)
}

/// ES's byte-size rendering of a breaker limit (`-1b` when unlimited).
fn limit_fmt(limit: i64) -> String {
    if limit < 0 { format!("{limit}b") } else { bytes_fmt(limit as u64, None) }
}

/// Circuit breakers (`_nodes/stats` `breakers`): the live limits, estimates
/// (reserved plus measured bytes; `parent`: the real memory it checks),
/// overheads and trip counts, keyed and ordered as ES 9.
fn breakers(node: &Node) -> Value {
    let mut m = Map::new();
    for b in node.breakers.stats() {
        m.insert(
            b.name.into(),
            json!({"limit_size_in_bytes": b.limit, "limit_size": limit_fmt(b.limit), "estimated_size_in_bytes": b.estimated, "estimated_size": bytes_fmt(b.estimated, None), "overhead": b.overhead, "tripped": b.tripped}),
        );
    }
    Value::Object(m)
}

/// `_cat/circuit_breaker` rows: `(name, limit, estimated, tripped, overhead)`.
pub(crate) fn breaker_rows(node: &Node) -> Vec<(&'static str, i64, u64, u64, f64)> {
    node.breakers.stats().into_iter().map(|b| (b.name, b.limit, b.estimated, b.tripped, b.overhead)).collect()
}

/// Painless compilation counters (`_nodes/stats` `script`).
fn script_section() -> Value {
    elasticsearch_engine::script::telemetry::stats_json()
}

fn indexing_pressure(heap_max: u64) -> Value {
    let _ = heap_max;
    // Live figures of the node's byte-based indexing pressure (document
    // writes hold their request bytes as coordinating and primary work).
    let p = elasticsearch_engine::indexing_pressure::global().stats();
    let all = p.combined_bytes + p.replica_bytes;
    let cur = json!({"combined_coordinating_and_primary_in_bytes": p.combined_bytes, "coordinating_in_bytes": p.coordinating_bytes,
        "primary_in_bytes": p.primary_bytes, "replica_in_bytes": p.replica_bytes, "all_in_bytes": all});
    let total_all = p.total_combined_bytes + p.total_replica_bytes;
    let total = json!({"combined_coordinating_and_primary_in_bytes": p.total_combined_bytes, "coordinating_in_bytes": p.total_coordinating_bytes,
        "primary_in_bytes": p.total_primary_bytes, "replica_in_bytes": p.total_replica_bytes, "all_in_bytes": total_all,
        "coordinating_rejections": p.coordinating_rejections, "primary_rejections": p.primary_rejections, "replica_rejections": p.replica_rejections,
        "primary_document_rejections": 0});
    json!({"memory": {"current": cur, "total": total, "limit_in_bytes": p.limit_bytes}})
}

/// Per-route request / response statistics (`http.routes`).
fn http_routes() -> Value {
    let st = lock(&ROUTE_STATS).clone();
    let mut m = Map::new();
    for (route, s) in st {
        m.insert(
            route,
            json!({
                "requests": {"count": s.req_count, "total_size_in_bytes": s.req_bytes, "size_histogram": histogram_json(&SIZE_BOUNDS, &s.req_hist, "ge_bytes", "lt_bytes")},
                "responses": {
                    "count": s.resp_count, "total_size_in_bytes": s.resp_bytes,
                    "size_histogram": histogram_json(&SIZE_BOUNDS, &s.resp_hist, "ge_bytes", "lt_bytes"),
                    "handling_time_histogram": histogram_json(&TIME_BOUNDS, &s.time_hist, "ge_millis", "lt_millis")
                }
            }),
        );
    }
    Value::Object(m)
}

pub(crate) fn http_section() -> Value {
    let now = elasticsearch_engine::date::now_millis();
    let mut all = lock(&CLOSED_CLIENTS).snapshot(now);
    for shard in HTTP_CLIENTS.iter() {
        let cs = lock(shard);
        all.extend(cs.clients.values().cloned());
    }
    let clients: Vec<Value> = all
        .iter()
        .map(|c| {
            let mut m = Map::new();
            m.insert("id".into(), json!(c.id));
            m.insert("agent".into(), json!(c.agent.as_deref().unwrap_or("")));
            m.insert("local_address".into(), json!(c.local_address));
            m.insert("remote_address".into(), json!(c.remote_address));
            m.insert("last_uri".into(), json!(c.last_uri));
            if let Some(x) = &c.x_opaque_id {
                m.insert("x_opaque_id".into(), json!(x));
            }
            m.insert("opened_time_millis".into(), json!(c.opened));
            if let Some(t) = c.closed {
                m.insert("closed_time_millis".into(), json!(t));
            }
            m.insert("last_request_time_millis".into(), json!(c.last_request));
            m.insert("request_count".into(), json!(c.request_count));
            m.insert("request_size_bytes".into(), json!(c.request_size));
            Value::Object(m)
        })
        .collect();
    json!({"current_open": HTTP_CURRENT.load(Ordering::Relaxed), "total_opened": HTTP_TOTAL.load(Ordering::Relaxed).max(1), "clients": clients, "routes": http_routes()})
}

fn fs_section(node: &Node) -> Value {
    // Probed at most once a second (ES `FsService`, `monitor.fs.refresh_interval`).
    let (f, io) = sysinfo::fs_cached(&node.data);
    // Free space each disk watermark leaves (ES 85% / 90% / 95% defaults).
    let free_at = |pct: f64| (f.total as f64 * (100.0 - pct) / 100.0) as u64;
    let mount = match &f.device {
        Some(d) => format!("{} ({d})", f.mount),
        None => format!("{} ({})", f.mount, f.mount),
    };
    let mut out = json!({
        "timestamp": elasticsearch_engine::date::now_millis(),
        "total": {"total_in_bytes": f.total, "free_in_bytes": f.free, "available_in_bytes": f.available},
        "data": [{
            "path": node.data.to_string_lossy(), "mount": mount, "type": f.fs_type,
            "total_in_bytes": f.total, "free_in_bytes": f.free, "available_in_bytes": f.available,
            "low_watermark_free_space_in_bytes": free_at(85.0), "high_watermark_free_space_in_bytes": free_at(90.0), "flood_stage_free_space_in_bytes": free_at(95.0)
        }]
    });
    if let Some((name, io)) = io {
        let d = json!({"operations": io.read_ops + io.write_ops, "read_operations": io.read_ops, "write_operations": io.write_ops,
                       "read_kilobytes": io.read_kb, "write_kilobytes": io.write_kb, "io_time_in_millis": io.io_time_ms});
        let mut dev = json!({"device_name": name});
        if let (Some(o), Some(src)) = (dev.as_object_mut(), d.as_object()) {
            o.extend(src.clone());
        }
        out["io_stats"] = json!({"devices": [dev], "total": d});
    }
    out
}

fn os_section(s: &Sys) -> Value {
    let used = s.total_mem.saturating_sub(s.free_mem);
    let pct = |a: u64| if s.total_mem == 0 { 0 } else { (a * 100 + s.total_mem / 2) / s.total_mem };
    let mut v = json!({
        "timestamp": elasticsearch_engine::date::now_millis(),
        "cpu": {"percent": s.os_cpu_percent, "load_average": {"1m": s.load[0], "5m": s.load[1], "15m": s.load[2]}},
        "mem": {"total_in_bytes": s.total_mem, "adjusted_total_in_bytes": s.total_mem, "free_in_bytes": s.free_mem, "used_in_bytes": used, "free_percent": pct(s.free_mem), "used_percent": pct(used)},
        "swap": {"total_in_bytes": s.swap_total, "free_in_bytes": s.swap_free, "used_in_bytes": s.swap_total.saturating_sub(s.swap_free)}
    });
    if let Some(c) = sysinfo::cgroup_cached() {
        v["cgroup"] = json!({
            "cpuacct": {"control_group": c.cpuacct_group, "usage_nanos": c.usage_nanos},
            "cpu": {"control_group": c.cpu_group, "cfs_period_micros": c.cfs_period_micros, "cfs_quota_micros": c.cfs_quota_micros,
                    "stat": {"number_of_elapsed_periods": c.elapsed_periods, "number_of_times_throttled": c.times_throttled, "time_throttled_nanos": c.throttled_nanos}},
            "memory": {"control_group": c.memory_group, "limit_in_bytes": c.memory_limit, "usage_in_bytes": c.memory_usage}
        });
    }
    v
}

fn process_section(s: &Sys) -> Value {
    json!({
        "timestamp": elasticsearch_engine::date::now_millis(),
        "open_file_descriptors": s.open_fds,
        "max_file_descriptors": s.max_fds,
        "cpu": {"percent": s.process_cpu_percent, "total_in_millis": s.cpu_millis},
        "mem": {"total_virtual_in_bytes": s.virt, "resident_in_bytes": s.rss,
            "anonymous_resident_in_bytes": s.rss_anon, "file_resident_in_bytes": s.rss_file,
            "shared_resident_in_bytes": s.rss_shmem,
            "native_budget_in_bytes": elasticsearch_engine::memory::budget(),
            "measurement": "native_process_resident_memory"}
    })
}

fn jvm_section(node: &Node, s: &Sys) -> Value {
    native_runtime_section(node.uptime_millis(), s)
}

fn native_runtime_section(uptime: u64, s: &Sys) -> Value {
    let (used, max) = sysinfo::heap(s);
    // These compatibility fields expose the labelled native measurement.
    // A Rust process has no JVM generations, code heaps or GC collectors;
    // empty maps mark measurements that do not apply, not measured zero usage.
    // ES's `direct` buffer pool is the off-heap network buffers its
    // `in_flight_requests` breaker accounts; here each in-flight request
    // body holds one reservation on that breaker, so the pool is measured
    // from it (buffers held and their bytes), not invented.
    let inflight = elasticsearch_engine::memory::breakers().get(elasticsearch_engine::memory::Kind::InflightRequests);
    let direct_bytes = inflight.reserved();
    json!({
        "timestamp": elasticsearch_engine::date::now_millis(),
        "uptime_in_millis": uptime,
        "runtime": "native",
        "jvm_measurements_available": false,
        "mem": {
            "heap_measurement": if s.rss_anon.is_some() { "native_anonymous_resident" } else { "native_accounted_allocations" },
            "heap_used_in_bytes": used, "heap_used_percent": if max == 0 { 0 } else { ((used as u128 * 100) / max as u128).min(u64::MAX as u128) as u64 },
            "heap_committed_in_bytes": max, "heap_max_in_bytes": max,
            "heap_capacity_measurement": "configured_compatibility_budget",
            "pools": {}
        },
        "threads": {"count": s.threads},
        "gc": {"collectors": {}},
        "buffer_pools": {
            "direct": {"count": inflight.live(), "used_in_bytes": direct_bytes, "total_capacity_in_bytes": direct_bytes,
                "measurement": "in_flight_request_buffers"}
        }
    })
}

#[cfg(test)]
mod native_runtime_tests {
    use super::*;
    #[test]
    fn runtime_reports_native_usage_without_invented_jvm_pools() {
        for anonymous in [123u64, 1024 * 1024 * 1024] {
            let sys = Sys { rss_anon: Some(anonymous), rss: anonymous + 4096,
                virt: 1 << 40, threads: 7, ..Sys::default() };
            let stats = native_runtime_section(42, &sys);
            assert_eq!(stats["mem"]["heap_used_in_bytes"], anonymous);
            assert_eq!(stats["mem"]["heap_measurement"], "native_anonymous_resident");
            assert_eq!(stats["mem"]["pools"], json!({}));
            assert_eq!(stats["gc"]["collectors"], json!({}));
            assert!(stats["buffer_pools"]["direct"]["count"].is_u64());
            assert!(stats["buffer_pools"]["direct"]["used_in_bytes"].is_u64());
            assert_eq!(stats["buffer_pools"]["direct"]["measurement"], "in_flight_request_buffers");
            assert_eq!(stats["jvm_measurements_available"], false);
            assert!(stats["mem"].get("non_heap_used_in_bytes").is_none());
            assert!(stats["threads"].get("peak_count").is_none());
        }
    }
}

fn transport_section() -> Value {
    // A single node handles its transport actions locally: one inbound
    // action per REST request.
    let inbound = histogram_json(&TIME_BOUNDS, &*lock(&TRANSPORT_HIST), "ge_millis", "lt_millis");
    json!({
        "server_open": 0, "total_outbound_connections": 0, "rx_count": 0, "rx_size_in_bytes": 0, "tx_count": 0, "tx_size_in_bytes": 0,
        "inbound_handling_time_histogram": inbound, "outbound_handling_time_histogram": []
    })
}

fn discovery_section(node: &Node, human: bool) -> Value {
    let up = node.uptime_millis().max(1);
    let timing = |fields: &[&str]| -> Value {
        let mut m = Map::new();
        m.insert("count".into(), json!(0));
        for f in fields {
            if human {
                m.insert(format!("{f}_time"), json!("0s"));
            }
            m.insert(format!("{f}_time_millis"), json!(0));
        }
        Value::Object(m)
    };
    let success = ["computation", "publication", "context_construction", "commit", "completion", "master_apply", "notification"];
    let ser = json!({"count": 0, "uncompressed_size_in_bytes": 0, "compressed_size_in_bytes": 0});
    let mut rec = json!({"name": "org.elasticsearch.indices.cluster.IndicesClusterStateService", "cumulative_execution_count": 1, "cumulative_execution_time_millis": up.min(1)});
    if human {
        rec["cumulative_execution_time"] = json!(time_fmt(1));
    }
    json!({
        "cluster_state_queue": {"total": 0, "pending": 0, "committed": 0},
        "published_cluster_states": {"full_states": 1, "incompatible_diffs": 0, "compatible_diffs": 0},
        "cluster_state_update": {"unchanged": timing(&["computation", "notification"]), "success": timing(&success), "failure": timing(&success)},
        "serialized_cluster_states": {"full_states": ser, "diffs": ser},
        "cluster_applier_stats": {"recordings": [rec]}
    })
}

fn allocations_section(node: &Node) -> Value {
    let all = node.all_indices();
    let store: u64 = all.iter().map(|s| s.store_size()).sum();
    json!({ NODE_ID: {"shards": all.len(), "undesired_shards": 0, "forecasted_ingest_load": 0.0, "forecasted_disk_usage_in_bytes": store, "current_disk_usage_in_bytes": store} })
}

fn nodes_stats(node: &Node, req: &Req) -> R {
    let p = req.segs.iter().position(|s| s == "stats").unwrap_or(1);
    let node_spec = if p >= 2 { req.seg(1).to_string() } else { "_all".into() };
    let metric_spec = req.segs.get(p + 1).cloned().or_else(|| req.param("metric").map(str::to_string));
    let index_spec = req.segs.get(p + 2).cloned().or_else(|| req.param("index_metric").map(str::to_string));
    let path_metrics = metric_spec.clone().unwrap_or_default();
    let path = format!("/_nodes/stats/{path_metrics}");
    let metrics = parse_metrics(&path, metric_spec.as_deref(), NODE_METRICS, "metric")?;
    let index_metrics = parse_metrics(&format!("{path}/{}", index_spec.clone().unwrap_or_default()), index_spec.as_deref(), INDEX_METRICS, "index metric")?;
    if index_spec.is_some() && metrics.as_ref().map(|m| !m.iter().any(|x| x == "indices")).unwrap_or(false) {
        return Err(EsError::illegal_argument(format!("request [{}] contains index metrics [{}] but indices stats not requested", req.path, index_spec.unwrap_or_default())));
    }
    let level = req.param("level").unwrap_or("node");
    if !matches!(level, "node" | "indices" | "shards") {
        return Err(EsError::illegal_argument(format!("level parameter must be one of [node] or [indices] or [shards] but was [{level}]")));
    }
    let want = |m: &str| metrics.as_ref().map(|l| l.iter().any(|x| x == m)).unwrap_or(true);
    let mut nodes = Map::new();
    if node_matches(node, &node_spec) {
        let s = sysinfo::sys_cached();
        let (_, heap_max) = sysinfo::heap(&s);
        let now = elasticsearch_engine::date::now_millis();
        let mut n = Map::new();
        n.insert("timestamp".into(), json!(now));
        n.insert("name".into(), json!(node.name));
        n.insert("transport_address".into(), json!(TRANSPORT_ADDRESS));
        n.insert("host".into(), json!("127.0.0.1"));
        n.insert("ip".into(), json!(TRANSPORT_ADDRESS));
        n.insert("roles".into(), json!(ROLES));
        n.insert("attributes".into(), node_attributes());
        if want("indices") {
            let response_metrics = index_metrics.as_ref().map(|ms| ms.iter().map(|m| index_metric_key(m).to_string()).collect::<Vec<_>>());
            let translog = index_metrics.as_ref().is_none_or(|ms| ms.iter().any(|m| m == "translog"));
            let counters_only = level != "shards" && crate::indices::counter_metrics_only(response_metrics.as_deref());
            let file_sizes_on = req.flag("include_segment_file_sizes")? && !counters_only;
            let mut agg = zero_index_section();
            let mut per_index = Map::new();
            let mut per_shard = Map::new();
            for svc in node.all_indices() {
                if !svc.is_open() {
                    continue;
                }
                let sec = selected_index_section(node, &svc, file_sizes_on, false, counters_only, translog);
                crate::indices::add_stats(&mut agg, &sec);
                if level == "indices" {
                    let mut v = sec.clone();
                    filter_sections(&mut v, &index_metrics);
                    per_index.insert(svc.name.clone(), v);
                } else if level == "shards" {
                    let mut v = selected_index_section(node, &svc, file_sizes_on, true, false, translog);
                    filter_sections(&mut v, &index_metrics);
                    v["routing"] = json!({"state": "STARTED", "primary": true, "node": NODE_ID, "relocating_node": null});
                    let docs = svc.doc_count().0 as u64;
                    for (k, x) in crate::indices::shard_copy_extras(node, &svc, docs) {
                        v[k.as_str()] = x;
                    }
                    per_shard.insert(svc.name.clone(), json!([{ "0": v }]));
                }
            }
            if file_sizes_on {
                // file sizes are objects of objects: re-sum by extension.
                let mut fs = Map::new();
                for svc in node.all_indices() {
                    if let Value::Object(m) = segment_file_sizes(&svc) {
                        for (k, v) in m {
                            match fs.get_mut(&k) {
                                Some(e) => crate::indices::add_stats(e, &json!({"size_in_bytes": v["size_in_bytes"]})),
                                None => {
                                    fs.insert(k, v);
                                }
                            }
                        }
                    }
                }
                agg["segments"]["file_sizes"] = Value::Object(fs);
            } else {
                agg["segments"]["file_sizes"] = json!({});
            }
            agg["segments"]["max_unsafe_auto_id_timestamp"] = json!(-1);
            if req.param("fielddata_fields").is_some() || req.param("fields").is_some() {
                agg["fielddata"]["fields"] = json!({});
            }
            if req.param("completion_fields").is_some() || req.param("fields").is_some() {
                agg["completion"]["fields"] = json!({});
            }
            if let Some(g) = req.param("groups") {
                let groups: Map<String, Value> = g.split(',').filter(|x| !x.is_empty() && *x != "_all").map(|x| (x.to_string(), json!({"query_total": 0, "query_time_in_millis": 0, "query_current": 0, "fetch_total": 0, "fetch_time_in_millis": 0, "fetch_current": 0, "scroll_total": 0, "scroll_time_in_millis": 0, "scroll_current": 0, "suggest_total": 0, "suggest_time_in_millis": 0, "suggest_current": 0}))).collect();
                agg["search"]["groups"] = Value::Object(groups);
            }
            if agg["refresh"].is_object() {
                // Engine lock instrumentation for benchmarks (extra fields;
                // ES clients ignore them): refresh's core write-lock holds
                // and mutable kNN point-map rebuilds, node-wide.
                let (holds, total, max) = elasticsearch_engine::shard::refresh_lock_hold();
                agg["refresh"]["rustic_lock_hold_count"] = json!(holds);
                agg["refresh"]["rustic_lock_hold_total_time_in_millis"] = json!(total.as_millis() as u64);
                agg["refresh"]["rustic_lock_hold_max_time_in_millis"] = json!(max.as_millis() as u64);
                agg["refresh"]["rustic_lock_hold_max_time_in_micros"] = json!(max.as_micros() as u64);
                agg["refresh"]["rustic_moved_owner_rebuilds"] = json!(elasticsearch_engine::rustic_core::reader_view::moved_owner_rebuilds());
                agg["refresh"]["rustic_point_inverse_size_in_bytes"] = json!(elasticsearch_engine::rustic_core::segment::point_inverse_bytes());
                // Reader pressure (gauges, node-wide): the oldest superseded
                // view still searched, queued mutable-graph delete repairs
                // and the retired graph memory old readers (PITs) hold.
                let mut oldest = std::time::Duration::ZERO;
                let (mut repairs, mut retired_slots, mut retired_bytes) = (0u64, 0u64, 0u64);
                for svc in node.all_indices() {
                    if !svc.is_open() { continue; }
                    for copy in svc.local_shards() {
                        let p = copy.shard.reader_pressure();
                        oldest = oldest.max(p.oldest_live_view_age);
                        repairs += p.vectors.pending_repairs;
                        retired_slots += p.vectors.retired_slots;
                        retired_bytes += p.vectors.retired_adjacency_bytes;
                    }
                }
                agg["refresh"]["rustic_oldest_live_view_age_in_millis"] = json!(oldest.as_millis() as u64);
                agg["refresh"]["rustic_pending_vector_repairs"] = json!(repairs);
                agg["refresh"]["rustic_retired_vector_slots"] = json!(retired_slots);
                agg["refresh"]["rustic_retired_vector_adjacency_bytes"] = json!(retired_bytes);
                agg["refresh"]["rustic_degraded_retained_views"] = json!(elasticsearch_engine::rustic_core::vector::degraded_retained_views());
            }
            filter_sections(&mut agg, &index_metrics);
            if level == "indices" {
                agg["indices"] = Value::Object(per_index);
            } else if level == "shards" {
                agg["shards"] = Value::Object(per_shard);
            }
            n.insert("indices".into(), agg);
        }
        if want("os") {
            n.insert("os".into(), os_section(&s));
        }
        if want("process") {
            n.insert("process".into(), process_section(&s));
        }
        if want("jvm") {
            n.insert("jvm".into(), jvm_section(node, &s));
        }
        if want("thread_pool") {
            n.insert("thread_pool".into(), thread_pool_section(node));
        }
        if want("fs") {
            n.insert("fs".into(), fs_section(node));
        }
        if want("transport") {
            n.insert("transport".into(), transport_section());
        }
        if want("http") {
            n.insert("http".into(), http_section());
        }
        if want("breaker") {
            n.insert("breakers".into(), breakers(node));
        }
        // Rustic-only: geo structures the memory budget refused (their
        // paths fell back) and point-tree uses; only with every metric.
        if metrics.is_none() {
            let g: Map<String, Value> = elasticsearch_engine::rustic_core::geo::structures::snapshot().into_iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
            n.insert("rustic_geo".into(), Value::Object(g));
            // Rustic-only: the shared core memory budget (owned caches,
            // resident vector rows and graphs, request workspaces) and the
            // share background work (merges) is admitted against.
            let budget = elasticsearch_engine::rustic_core::admission::shared();
            n.insert("rustic_memory_admission".into(), json!({
                "limit_in_bytes": budget.limit(),
                "used_in_bytes": budget.used(),
                "background_used_in_bytes": budget.background_used(),
                "background_share_in_bytes": budget.background_limit(),
            }));
        }
        if want("script") {
            n.insert("script".into(), script_section());
        }
        if want("discovery") {
            n.insert("discovery".into(), discovery_section(node, req.flag("human")?));
        }
        if want("ingest") {
            n.insert("ingest".into(), node.ingest().stats_json());
        }
        if want("adaptive_selection") {
            // The single-engine router owns no coordinator replica feedback.
            // The cluster layer replaces this with its measured peer samples.
            n.insert("adaptive_selection".into(), json!({}));
        }
        if want("script_cache") {
            let stats = script_section();
            n.insert("script_cache".into(), json!({"sum": {"compilations": stats["compilations"], "cache_evictions": stats["cache_evictions"], "compilation_limit_triggered": stats["compilation_limit_triggered"]}}));
        }
        if want("indexing_pressure") {
            n.insert("indexing_pressure".into(), indexing_pressure(heap_max));
        }
        if want("indices") {
            // Internal fold contexts (remote SQL PIVOT pages), apart from
            // user PITs: a node figure outside ES's index stats.
            n.insert("rustic_fold_contexts".into(), json!(node.fold_contexts.lock().len()));
        }
        if want("repositories") {
            let mut repos = Map::new();
            for r in node.snapshots.repository_names() {
                repos.insert(r, json!({"total_read_throttled_time_nanos": 0, "total_write_throttled_time_nanos": 0}));
            }
            n.insert("repositories".into(), Value::Object(repos));
        }
        if want("allocations") {
            n.insert("allocations".into(), allocations_section(node)[NODE_ID].clone());
        }
        nodes.insert(NODE_ID.into(), Value::Object(n));
    }
    let k = nodes.len();
    finish_json(req, json!({"_nodes": nodes_header(k), "cluster_name": node.cluster_name, "nodes": nodes}))
}

/// `GET {path}` of node stats (`/_nodes/_local/stats[/{metrics}[/{index_metrics}]]`,
/// `params`: the decoded query string) run in process on this engine,
/// without the HTTP dispatch: the JSON body the route answers, `?human`
/// applied as the route's response shaping would (`filter_path` / `pretty`
/// are left to the caller). A cluster node adds its shard engines' `indices`
/// sections to a node stats request engine 0's route already validated (see
/// `index_stats_in_process`).
pub fn nodes_stats_in_process(node: &Node, path: &str, params: HashMap<String, String>) -> Result<Value, EsError> {
    let human = matches!(params.get("human").map(String::as_str), Some("") | Some("true"));
    let segs: Vec<String> = path.split('/').filter(|s| !s.is_empty()).map(str::to_string).collect();
    let req = Req { method: Method::GET, path: path.to_string(), segs, params, body: Default::default(), headers: Default::default(),
        access: None, extensions: Default::default(), api_version: crate::RestApiVersion::V9 };
    let mut value = match nodes_stats(node, &req)?.body {
        crate::RespBody::Json(v) => v,
        crate::RespBody::RetainedJson { value, .. } => value,
        _ => return Err(EsError::internal("node stats rendered no JSON body")),
    };
    if human {
        crate::humanize(&mut value);
    }
    Ok(value)
}

// ---------------------------------------------------------------------------
// _nodes/usage, _nodes/hot_threads
// ---------------------------------------------------------------------------

fn nodes_usage(node: &Node, req: &Req) -> R {
    let p = req.segs.iter().position(|s| s == "usage").unwrap_or(1);
    let node_spec = if p >= 2 { req.seg(1).to_string() } else { "_all".into() };
    let metrics = parse_metrics(&req.path, req.segs.get(p + 1).map(String::as_str), &["rest_actions", "aggregations"], "metric")?;
    let want = |m: &str| metrics.as_ref().map(|l| l.iter().any(|x| x == m)).unwrap_or(true);
    let mut nodes = Map::new();
    if node_matches(node, &node_spec) {
        let mut n = json!({"timestamp": elasticsearch_engine::date::now_millis(), "since": *USAGE_SINCE});
        if want("rest_actions") {
            n["rest_actions"] = json!(*lock(&REST_ACTIONS));
        }
        if want("aggregations") {
            n["aggregations"] = json!(*lock(&AGG_USAGE));
        }
        nodes.insert(NODE_ID.into(), n);
    }
    Ok(Resp::ok(json!({"_nodes": nodes_header(nodes.len()), "cluster_name": node.cluster_name, "nodes": nodes})))
}

/// `(thread name, CPU ms)` of every thread of this process over `interval`,
/// busiest first (ties by name), from `/proc/self/task/*/stat`. `None` where
/// `/proc` has no task list (not Linux).
fn busiest_threads(interval: std::time::Duration) -> Option<Vec<(String, u64)>> {
    fn sample() -> Option<std::collections::HashMap<u32, (String, u64)>> {
        let mut out = std::collections::HashMap::new();
        for entry in std::fs::read_dir("/proc/self/task").ok()?.flatten() {
            let Ok(tid) = entry.file_name().to_string_lossy().parse::<u32>() else { continue };
            let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else { continue };
            if let Some(parsed) = parse_task_stat(&stat) { out.insert(tid, parsed); }
        }
        Some(out)
    }
    let before = sample()?;
    std::thread::sleep(interval);
    let after = sample()?;
    // SAFETY: sysconf has no preconditions.
    let ticks = match unsafe { libc::sysconf(libc::_SC_CLK_TCK) } { t if t > 0 => t as u64, _ => 100 };
    let mut busiest: Vec<(String, u64)> = after.into_iter().map(|(tid, (name, total))| {
        let start = before.get(&tid).map_or(0, |(_, t)| *t);
        (name, total.saturating_sub(start) * 1000 / ticks)
    }).collect();
    busiest.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Some(busiest)
}

/// `(comm, utime + stime ticks)` of one `/proc/<pid>/task/<tid>/stat` line.
/// The name is in parentheses and may itself hold spaces or parentheses:
/// the fields after it start past the last `)`.
fn parse_task_stat(stat: &str) -> Option<(String, u64)> {
    let open = stat.find('(')?;
    let close = stat.rfind(')')?;
    let name = stat.get(open + 1..close)?.to_string();
    // After the name: state (field 3), ..., utime (14), stime (15).
    let fields: Vec<&str> = stat.get(close + 1..)?.split_whitespace().collect();
    let utime: u64 = fields.get(11)?.parse().ok()?;
    let stime: u64 = fields.get(12)?.parse().ok()?;
    Some((name, utime + stime))
}

fn hot_threads(node: &Node, req: &Req) -> R {
    let ty = req.param("type").unwrap_or("cpu");
    if !matches!(ty, "cpu" | "wait" | "block" | "mem") {
        return Err(EsError::illegal_argument(format!("type not supported [{ty}]")));
    }
    let sort = req.param("sort").unwrap_or("total");
    if !matches!(sort, "cpu" | "total") {
        return Err(EsError::illegal_argument(format!("sort order not supported [{sort}]")));
    }
    if sort == "cpu" && ty != "cpu" {
        return Err(EsError::illegal_argument("sort order [cpu] can only be used with type [cpu]"));
    }
    let threads: usize = req.param("threads").and_then(|s| s.parse().ok()).unwrap_or(3);
    let interval = req.param("interval").unwrap_or("500ms").to_string();
    let snapshots: usize = req.param("snapshots").and_then(|s| s.parse().ok()).unwrap_or(10);
    let ignore_idle = req.bool_param("ignore_idle_threads")?.unwrap_or(true);
    let p = req.segs.iter().position(|s| s == "hot_threads" || s == "hotthreads").unwrap_or(1);
    let node_spec = if p >= 2 { req.seg(1).to_string() } else { "_all".into() };
    if !node_matches(node, &node_spec) {
        return Ok(Resp::text(String::new()));
    }
    let now = elasticsearch_engine::date::from_millis(elasticsearch_engine::date::now_millis()).format("%Y-%m-%dT%H:%M:%S%.3fZ");
    let mut out = format!(
        "::: {{{}}}{{{NODE_ID}}}{{{EPHEMERAL_ID}}}{{127.0.0.1}}{{{TRANSPORT_ADDRESS}}}{{{ROLE_CHARS}}}{{{ES_VERSION}}}{{{}-{}}}{{transform.config_version=10.0.0, xpack.installed=true}}\n   Hot threads at {now}, interval={interval}, busiestThreads={threads}, ignoreIdleThreads={ignore_idle}:\n   \n",
        node.name,
        elasticsearch_engine::version::IndexVersion::MINIMUM_COMPATIBLE,
        elasticsearch_engine::version::IndexVersion::CURRENT
    );
    // Per-thread CPU over the interval from /proc (Linux): every thread of
    // the process (indexing, merge, seal, vector optimizer and search pool
    // workers included), busiest first. Stacks of other threads are not
    // sampled; each entry names the thread and its CPU share.
    let interval_ms = elasticsearch_engine::date::parse_time_value(&interval).filter(|ms| *ms > 0).unwrap_or(500).min(60_000) as u64;
    if let Some(busiest) = busiest_threads(std::time::Duration::from_millis(interval_ms)) {
        let mut shown = 0;
        for (name, cpu_ms) in busiest {
            if shown >= threads || (ignore_idle && cpu_ms == 0) { break; }
            let pct = cpu_ms as f64 * 100.0 / interval_ms as f64;
            out.push_str(&format!(
                "   {pct:.1}% [cpu={pct:.1}%, other=0.0%] ({cpu_ms}ms out of {interval}) {ty} usage by thread 'elasticsearch[{}][{name}]'\n     unique snapshot\n       (per-thread CPU from /proc; stack not sampled)\n   \n",
                node.name
            ));
            shown += 1;
        }
        return Ok(Resp::text(out));
    }
    let s = sysinfo::sys();
    let thread = std::thread::current();
    let tname = thread.name().unwrap_or("worker");
    let bt = std::backtrace::Backtrace::force_capture().to_string();
    let frames: Vec<String> = bt
        .lines()
        .map(str::trim)
        .filter(|l| !l.starts_with("at ") && !l.is_empty())
        .map(|l| l.split_once(": ").map(|x| x.1).unwrap_or(l).to_string())
        .take(24)
        .collect();
    let pct = s.process_cpu_percent as f64;
    out.push_str(&format!(
        "   {pct:.1}% [cpu={pct:.1}%, other=0.0%] (0s out of {interval}) {ty} usage by thread 'elasticsearch[{}][{tname}]'\n     {snapshots}/{snapshots} snapshots sharing following {} elements\n",
        node.name,
        frames.len()
    ));
    for f in frames {
        out.push_str(&format!("       {f}\n"));
    }
    Ok(Resp::text(out))
}

// ---------------------------------------------------------------------------
// _cluster/stats
// ---------------------------------------------------------------------------

#[derive(Default)]
struct TypeCount {
    count: u64,
    indices: BTreeSet<String>,
    /// dense_vector: indexed fields, their dims (min, max), element types.
    indexed_vectors: u64,
    dim_min: Option<u64>,
    dim_max: Option<u64>,
    element_types: BTreeMap<String, u64>,
    index_types: BTreeMap<String, u64>,
    similarity_types: BTreeMap<String, u64>,
}

fn walk_props(props: &Map<String, Value>, idx: &str, out: &mut BTreeMap<String, TypeCount>, total: &mut u64, prefix: &str, paths: &mut BTreeSet<String>) {
    for (name, def) in props {
        let path = if prefix.is_empty() { name.clone() } else { format!("{prefix}.{name}") };
        paths.insert(path.clone());
        let ty = def.get("type").and_then(Value::as_str).unwrap_or("object").to_string();
        let e = out.entry(ty.clone()).or_default();
        e.count += 1;
        e.indices.insert(idx.to_string());
        *total += 1;
        if ty == "dense_vector" {
            let et = def.get("element_type").and_then(Value::as_str).unwrap_or("float").to_string();
            let indexed = def.get("index").and_then(Value::as_bool).unwrap_or(true);
            let index_type = if !indexed {
                "not_indexed".to_string()
            } else {
                match def.get("index_options").and_then(|o| o.get("type")).and_then(Value::as_str) {
                    Some(t) => t.to_string(),
                    None => match et.as_str() {
                        "byte" => "hnsw".to_string(),
                        "bit" => "flat".to_string(),
                        _ => "int8_hnsw".to_string(),
                    },
                }
            };
            *e.index_types.entry(index_type).or_default() += 1;
            if indexed {
                let sim = def.get("similarity").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| if et == "bit" { "l2_norm".into() } else { "cosine".into() });
                *e.similarity_types.entry(sim).or_default() += 1;
            }
            *e.element_types.entry(et).or_default() += 1;
            if indexed {
                e.indexed_vectors += 1;
                if let Some(d) = def.get("dims").and_then(Value::as_u64) {
                    e.dim_min = Some(e.dim_min.map_or(d, |m| m.min(d)));
                    e.dim_max = Some(e.dim_max.map_or(d, |m| m.max(d)));
                }
            }
        }
        if let Some(p) = def.get("properties").and_then(Value::as_object) {
            walk_props(p, idx, out, total, &path, paths);
        }
        if let Some(f) = def.get("fields").and_then(Value::as_object) {
            walk_props(f, idx, out, total, &path, paths);
        }
    }
}

#[derive(Default)]
struct RuntimeTypeStats {
    count: u64,
    indices: BTreeSet<String>,
    scriptless: u64,
    shadowed: u64,
    lang: BTreeSet<String>,
    lines_max: u64,
    lines_total: u64,
    chars_max: u64,
    chars_total: u64,
    source_max: u64,
    source_total: u64,
    doc_max: u64,
    doc_total: u64,
}

/// `_cluster/stats` `nodes.ingest`.
fn ingest_cluster_stats(node: &Node) -> Value {
    let store = node.ingest();
    let mut ps = Map::new();
    for (k, [count, time, current, failed]) in store.processor_type_stats() {
        ps.insert(k, json!({"count": count, "failed": failed, "current": current, "time_in_millis": time}));
    }
    json!({"number_of_pipelines": store.count(), "processor_stats": ps})
}

/// `_cluster/stats` `repositories`: repository count per type.
fn repository_type_usage(node: &Node) -> Value {
    let repos = node.snapshots.get_repositories(None).unwrap_or_default();
    let mut by: BTreeMap<String, u64> = BTreeMap::new();
    if let Some(m) = repos.as_object() {
        for v in m.values() {
            *by.entry(v["type"].as_str().unwrap_or("fs").to_string()).or_default() += 1;
        }
    }
    let m: Map<String, Value> = by.into_iter().map(|(k, n)| (k, json!({"read_write": n, "count": n}))).collect();
    Value::Object(m)
}

fn cluster_stats(node: &Node, req: &Req) -> R {
    let all: Vec<Arc<IndexService>> = node.all_indices();
    let s = sysinfo::sys();
    let (heap_used, heap_max) = sysinfo::heap(&s);
    // Index-level sums.
    let mut agg = zero_index_section();
    let mut primaries = 0u64;
    let mut total_shards = 0u64;
    let mut field_types: BTreeMap<String, TypeCount> = BTreeMap::new();
    let mut runtime_types: BTreeMap<String, RuntimeTypeStats> = BTreeMap::new();
    let mut built_in: BTreeMap<&'static str, BTreeMap<String, (u64, BTreeSet<String>)>> = BTreeMap::new();
    let mut total_fields = 0u64;
    let mut dedup: BTreeSet<String> = BTreeSet::new();
    let mut dedup_size = 0u64;
    let mut analysis: BTreeMap<&'static str, BTreeMap<String, (u64, BTreeSet<String>)>> = BTreeMap::new();
    let mut per_index_shards: Vec<f64> = Vec::new();
    let mut per_index_reps: Vec<f64> = Vec::new();
    for svc in &all {
        let reps = svc.setting_usize("index.number_of_replicas", 1) as u64;
        primaries += 1;
        total_shards += 1;
        per_index_shards.push(1.0);
        per_index_reps.push(reps as f64);
        if svc.is_open() {
            crate::indices::add_stats(&mut agg, &full_index_section(node, svc, false, false));
            let sv = crate::indices::sparse_vector_count(svc);
            crate::indices::add_stats(&mut agg, &json!({"sparse_vector": {"value_count": sv}}));
        }
        let meta = svc.meta();
        let mj = meta.mappings.to_json();
        let ser = mj.to_string();
        if dedup.insert(ser.clone()) {
            dedup_size += ser.len() as u64;
        }
        let mut paths = BTreeSet::new();
        if let Some(p) = mj.get("properties").and_then(Value::as_object) {
            walk_props(p, &svc.name, &mut field_types, &mut total_fields, "", &mut paths);
        }
        if let Some(rt) = mj.get("runtime").and_then(Value::as_object) {
            for (name, def) in rt {
                let ty = def.get("type").and_then(Value::as_str).unwrap_or("keyword").to_string();
                let e = runtime_types.entry(ty).or_default();
                e.count += 1;
                e.indices.insert(svc.name.clone());
                match def.get("script") {
                    None => e.scriptless += 1,
                    Some(sc) => {
                        let (src, lang) = match sc {
                            Value::String(x) => (x.clone(), "painless".to_string()),
                            o => (o.get("source").and_then(Value::as_str).unwrap_or("").to_string(), o.get("lang").and_then(Value::as_str).unwrap_or("painless").to_string()),
                        };
                        e.lang.insert(lang);
                        let lines = src.lines().count().max(1) as u64;
                        let chars = src.chars().count() as u64;
                        let source = src.matches("params._source").count() as u64;
                        let doc = src.matches("doc[").count() as u64 + src.matches("doc.get(").count() as u64;
                        e.lines_max = e.lines_max.max(lines);
                        e.lines_total += lines;
                        e.chars_max = e.chars_max.max(chars);
                        e.chars_total += chars;
                        e.source_max = e.source_max.max(source);
                        e.source_total += source;
                        e.doc_max = e.doc_max.max(doc);
                        e.doc_total += doc;
                    }
                }
                if paths.contains(name) {
                    e.shadowed += 1;
                }
            }
        }
        // Built-in analysis components named by custom analyzers / mappings.
        {
            let st = &meta.settings;
            let custom = |kind: &str, name: &str| st.keys().any(|k| k.starts_with(&format!("index.analysis.{kind}.{name}.")));
            let mut note = |bucket: &'static str, name: &str| {
                let e = built_in.entry(bucket).or_default().entry(name.to_string()).or_default();
                e.0 += 1;
                e.1.insert(svc.name.clone());
            };
            for (k, v) in st {
                let Some(rest) = k.strip_prefix("index.analysis.analyzer.").or_else(|| k.strip_prefix("index.analysis.normalizer.")) else { continue };
                let names: Vec<String> = match v {
                    Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
                    Value::String(x) => vec![x.clone()],
                    _ => vec![],
                };
                if rest.ends_with(".tokenizer") {
                    for n in &names {
                        if !custom("tokenizer", n) {
                            note("built_in_tokenizers", n);
                        }
                    }
                } else if rest.ends_with(".filter") {
                    for n in &names {
                        if !custom("filter", n) {
                            note("built_in_filters", n);
                        }
                    }
                } else if rest.ends_with(".char_filter") {
                    for n in &names {
                        if !custom("char_filter", n) {
                            note("built_in_char_filters", n);
                        }
                    }
                }
            }
            fn analyzers(props: &Map<String, Value>, out: &mut Vec<String>) {
                for def in props.values() {
                    for k in ["analyzer", "search_analyzer", "search_quote_analyzer"] {
                        if let Some(a) = def.get(k).and_then(Value::as_str) {
                            out.push(a.to_string());
                        }
                    }
                    for k in ["properties", "fields"] {
                        if let Some(p) = def.get(k).and_then(Value::as_object) {
                            analyzers(p, out);
                        }
                    }
                }
            }
            let mut used = Vec::new();
            if let Some(p) = mj.get("properties").and_then(Value::as_object) {
                analyzers(p, &mut used);
            }
            used.sort();
            used.dedup();
            for a in used {
                if !custom("analyzer", &a) {
                    note("built_in_analyzers", &a);
                }
            }
        }
        for (k, v) in &meta.settings {
            let Some(rest) = k.strip_prefix("index.analysis.") else { continue };
            let parts: Vec<&str> = rest.split('.').collect();
            if parts.len() == 3 && parts[2] == "type" {
                let kind = match parts[0] {
                    "analyzer" => "analyzer_types",
                    "tokenizer" => "tokenizer_types",
                    "filter" => "filter_types",
                    "char_filter" => "char_filter_types",
                    "normalizer" => "normalizer_types",
                    _ => continue,
                };
                let e = analysis.entry(kind).or_default().entry(v.as_str().unwrap_or("custom").to_string()).or_default();
                e.0 += 1;
                e.1.insert(svc.name.clone());
            }
        }
    }
    let stat = |v: &[f64]| -> Value {
        if v.is_empty() {
            return json!({});
        }
        let min = v.iter().cloned().fold(f64::MAX, f64::min);
        let max = v.iter().cloned().fold(f64::MIN, f64::max);
        json!({"min": min, "max": max, "avg": v.iter().sum::<f64>() / v.len() as f64})
    };
    let shards = if all.is_empty() {
        json!({})
    } else {
        json!({"total": total_shards, "primaries": primaries, "replication": 0.0,
               "index": {"shards": stat(&per_index_shards), "primaries": stat(&per_index_shards), "replication": stat(&vec![0.0; per_index_reps.len()])}})
    };
    let ft: Vec<Value> = field_types
        .into_iter()
                .map(|(k, v)| {
            let mut e = json!({"name": k, "count": v.count, "index_count": v.indices.len(), "script_count": 0});
            if k == "dense_vector" {
                e["indexed_vector_count"] = json!(v.indexed_vectors);
                e["indexed_vector_dim_min"] = json!(v.dim_min.map(|d| d as i64).unwrap_or(-1));
                e["indexed_vector_dim_max"] = json!(v.dim_max.map(|d| d as i64).unwrap_or(-1));
                e["vector_element_type_count"] = json!(v.element_types);
                e["vector_index_type_count"] = json!(v.index_types);
                e["vector_similarity_type_count"] = json!(v.similarity_types);
            }
            e
        })
        .collect();
    let rt: Vec<Value> = runtime_types
        .into_iter()
        .map(|(k, e)| {
            json!({"name": k, "count": e.count, "index_count": e.indices.len(), "scriptless_count": e.scriptless, "shadowed_count": e.shadowed, "lang": e.lang,
                   "lines_max": e.lines_max, "lines_total": e.lines_total, "chars_max": e.chars_max, "chars_total": e.chars_total,
                   "source_max": e.source_max, "source_total": e.source_total, "doc_max": e.doc_max, "doc_total": e.doc_total})
        })
        .collect();
    let bi = |k: &str| -> Value {
        Value::Array(built_in.get(k).map(|m| m.iter().map(|(n, (c, i))| json!({"name": n, "count": c, "index_count": i.len()})).collect()).unwrap_or_default())
    };
    let an = |k: &str| -> Value {
        Value::Array(analysis.get(k).map(|m| m.iter().map(|(n, (c, i))| json!({"name": n, "count": c, "index_count": i.len()})).collect()).unwrap_or_default())
    };
    let docs = agg["docs"].clone();
    let store = agg["store"].clone();
    let total_primary_bytes = store["size_in_bytes"].as_u64().unwrap_or(0);
    let mut segments = agg["segments"].clone();
    segments["max_unsafe_auto_id_timestamp"] = json!(-1);
    if let Some(o) = segments.as_object_mut() {
        o.insert("file_sizes".into(), json!({}));
    }
    let versions = if all.is_empty() {
        json!([])
    } else {
        // One row per release range of the indices' creation versions
        // (`IndexVersion.toReleaseVersion`); the shard / byte totals are not
        // split per range (a single range is the common case).
        let mut rows: Vec<(String, usize)> = Vec::new();
        for svc in &all {
            let rv = svc.created_version().release_version();
            match rows.iter_mut().find(|r| r.0 == rv) {
                Some(r) => r.1 += 1,
                None => rows.push((rv, 1)),
            }
        }
        let single = rows.len() == 1;
        Value::Array(
            rows.into_iter()
                .map(|(v, c)| {
                    let (p, b) = if single { (primaries, total_primary_bytes) } else { (c as u64, 0) };
                    json!({"version": v, "index_count": c, "primary_shard_count": p, "total_primary_bytes": b})
                })
                .collect(),
        )
    };
    let mut source_modes = serde_json::Map::new();
    for svc in &all {
        let meta = svc.meta();
        let mode = elasticsearch_engine::synthetic::source_mode(meta.settings.get("index.mapping.source.mode").and_then(Value::as_str), &meta.mappings, meta.settings.get("index.mode").and_then(Value::as_str));
        let key = match mode {
            elasticsearch_engine::synthetic::SourceMode::Stored => "stored",
            elasticsearch_engine::synthetic::SourceMode::Synthetic => "synthetic",
            elasticsearch_engine::synthetic::SourceMode::Disabled => "disabled",
        };
        let n = source_modes.get(key).and_then(Value::as_u64).unwrap_or(0);
        source_modes.insert(key.to_string(), json!(n + 1));
    }
    let (status, _) = cluster_status(node);
    let used_mem = s.total_mem.saturating_sub(s.free_mem);
    let pct = |a: u64| if s.total_mem == 0 { 0 } else { (a * 100 + s.total_mem / 2) / s.total_mem };
    let f = sysinfo::fs(&node.data);
    let selected = req.segs.get(3).map(|n| node_matches(node, n)).unwrap_or(true);
    let nn = if selected { 1 } else { 0 };
    let mut body = json!({
        "_nodes": nodes_header(nn),
        "cluster_name": node.cluster_name,
        "cluster_uuid": node.cluster_uuid,
        "timestamp": elasticsearch_engine::date::now_millis(),
        "status": status,
        "indices": {
            "count": all.len(),
            "shards": shards,
            "docs": docs,
            "store": store,
            "fielddata": agg["fielddata"],
            "query_cache": agg["query_cache"],
            "completion": agg["completion"],
            "segments": segments,
            "mappings": {
                "total_field_count": total_fields,
                "total_deduplicated_field_count": total_fields,
                "total_deduplicated_mapping_size_in_bytes": dedup_size,
                "field_types": ft,
                "runtime_field_types": rt,
                "source_modes": source_modes
            },
            "analysis": {
                "char_filter_types": an("char_filter_types"), "tokenizer_types": an("tokenizer_types"), "filter_types": an("filter_types"),
                "analyzer_types": an("analyzer_types"), "built_in_char_filters": bi("built_in_char_filters"), "built_in_tokenizers": bi("built_in_tokenizers"),
                "built_in_filters": bi("built_in_filters"), "built_in_analyzers": bi("built_in_analyzers"),
                "synonyms": {}
            },
            "versions": versions,
            "search": search_usage_json(),
            "dense_vector": {"value_count": agg["dense_vector"]["value_count"]},
            "sparse_vector": agg["sparse_vector"]
        },
        "nodes": {
            "count": {"total": nn, "coordinating_only": 0, "data": nn, "data_cold": nn, "data_content": nn, "data_frozen": nn, "data_hot": nn, "data_warm": nn,
                      "index": 0, "ingest": nn, "master": nn, "ml": nn, "remote_cluster_client": nn, "search": 0, "transform": nn, "voting_only": 0},
            "versions": [ES_VERSION],
            "os": {
                "available_processors": s.cpus, "allocated_processors": s.cpus,
                "names": [{"name": s.os_name, "count": nn}],
                "pretty_names": [{"pretty_name": s.os_pretty_name, "count": nn}],
                "architectures": [{"arch": std::env::consts::ARCH, "count": nn}],
                "mem": {"total_in_bytes": s.total_mem, "adjusted_total_in_bytes": s.total_mem, "free_in_bytes": s.free_mem, "used_in_bytes": used_mem,
                        "free_percent": pct(s.free_mem), "used_percent": pct(used_mem)}
            },
            "process": {"cpu": {"percent": s.process_cpu_percent}, "open_file_descriptors": {"min": s.open_fds, "max": s.open_fds, "avg": s.open_fds}},
            "jvm": {
                "max_uptime_in_millis": node.uptime_millis(),
                "versions": [{"version": "21.0.2", "vm_name": "OpenJDK 64-Bit Server VM", "vm_version": "21.0.2+13-58", "vm_vendor": "Oracle Corporation",
                              "bundled_jdk": true, "using_bundled_jdk": true, "count": nn}],
                "mem": {"heap_used_in_bytes": heap_used, "heap_max_in_bytes": heap_max},
                "threads": s.threads
            },
            "fs": {"total_in_bytes": f.total, "free_in_bytes": f.free, "available_in_bytes": f.available},
            "plugins": [],
            "network_types": {"transport_types": {"netty4": nn}, "http_types": {"netty4": nn}},
            "discovery_types": {"single-node": nn},
            "packaging_types": [{"flavor": "default", "type": "tar", "count": nn}],
            "ingest": ingest_cluster_stats(node),
            "indexing_pressure": indexing_pressure(heap_max)
        },
        "snapshots": {
            "current_counts": {"snapshots": 0, "shard_snapshots": 0, "snapshot_deletions": 0, "concurrent_operations": 0, "cleanups": 0},
            "repositories": snapshot_repository_stats(node, req.flag("human")?)
        },
        "repositories": repository_type_usage(node),
        // ES omits the `_esql` CCS section when no remote cluster is configured.
        "ccs": {"_search": {"total": 0, "success": 0, "skipped": 0, "took": {"max": 0, "avg": 0, "p90": 0}, "took_mrt_true": {"max": 0, "avg": 0, "p90": 0},
                            "took_mrt_false": {"max": 0, "avg": 0, "p90": 0}, "remotes_per_search_max": 0, "remotes_per_search_avg": 0.0,
                            "failure_reasons": {}, "features": {}, "clients": {}, "clusters": {}}}
    });
    body["ccs"] = crate::ccs::stats(node, req.flag("include_remotes")?, &body);
    finish_json(req, body)
}

/// (status, unassigned replica count) for every index.
fn cluster_status(node: &Node) -> (&'static str, usize) {
    let unassigned: usize = node.all_indices().iter().map(|s| s.setting_usize("index.number_of_replicas", 1)).sum();
    (if unassigned > 0 { "yellow" } else { "green" }, unassigned)
}

// ---------------------------------------------------------------------------
// _health_report
// ---------------------------------------------------------------------------

const INDICATORS: &[&str] = &["master_is_stable", "repository_integrity", "disk", "shards_capacity", "shards_availability", "data_stream_lifecycle", "slm", "ilm"];

fn indicator(node: &Node, name: &str, verbose: bool) -> Value {
    let all = node.all_indices();
    let mut v = match name {
        "master_is_stable" => json!({"status": "green", "symptom": "The cluster has a stable master node",
            "details": {"current_master": {"node_id": NODE_ID, "name": node.name}, "recent_masters": [{"node_id": NODE_ID, "name": node.name}]}}),
        "repository_integrity" => match node.snapshots.repository_names().len() {
            0 => json!({"status": "green", "symptom": "No snapshot repositories configured."}),
            n => json!({"status": "green", "symptom": "All repositories are healthy.", "details": {"total_repositories": n}}),
        },
        "shards_capacity" => json!({"status": "green", "symptom": "The cluster has enough room to add new shards.",
            "details": {"data": {"max_shards_in_cluster": 1000}, "frozen": {"max_shards_in_cluster": 3000}}}),
        "shards_availability" => {
            let primaries = all.iter().filter(|s| s.is_open()).count();
            let replicas: usize = all.iter().filter(|s| s.is_open()).map(|s| s.setting_usize("index.number_of_replicas", 1)).sum();
            let details = json!({"unassigned_replicas": replicas, "initializing_replicas": 0, "creating_replicas": 0, "restarting_replicas": 0, "started_replicas": 0,
                                 "unassigned_primaries": 0, "initializing_primaries": 0, "creating_primaries": 0, "restarting_primaries": 0, "started_primaries": primaries});
            // Indices whose primaries may not be allocated (`index.routing.allocation.enable` none / replicas).
            let blocked: Vec<&std::sync::Arc<IndexService>> =
                all.iter().filter(|s| s.is_open() && matches!(s.setting("index.routing.allocation.enable").as_deref(), Some("none") | Some("replicas"))).collect();
            if !blocked.is_empty() {
                let unassigned: usize = blocked.iter().map(|s| s.setting_usize("index.number_of_shards", 1)).sum();
                let mut details = details;
                details["unassigned_primaries"] = json!(unassigned);
                details["started_primaries"] = json!(primaries.saturating_sub(blocked.len()));
                let affected: Vec<String> = blocked.iter().map(|s| s.name.clone()).collect();
                json!({"status": "red",
                       "symptom": format!("This cluster has {unassigned} unavailable primary shard{}.", if unassigned == 1 { "" } else { "s" }),
                       "details": details,
                       "impacts": [{"id": "elasticsearch:health:shards_availability:impact:primary_unassigned", "severity": 1,
                                    "description": format!("Cannot add data to {} {}. Searches might return incomplete results.", affected.len(), if affected.len() == 1 { "index" } else { "indices" }),
                                    "impact_areas": ["ingest", "search"]}],
                       "diagnosis": [{"id": "elasticsearch:health:shards_availability:diagnosis:enable_index_allocations",
                                      "cause": "Elasticsearch isn't allowed to allocate some shards from these indices because allocation for those shards has been disabled at the index level.",
                                      "action": "Check that the [index.routing.allocation.enable] index setting is set to [all].",
                                      "help_url": "https://ela.st/fix-index-allocation",
                                      "affected_resources": {"indices": affected}}]})
            } else if replicas == 0 {
                json!({"status": "green", "symptom": "This cluster has all shards available.", "details": details})
            } else {
                let affected: Vec<String> = all.iter().filter(|s| s.setting_usize("index.number_of_replicas", 1) > 0).map(|s| s.name.clone()).collect();
                json!({"status": "yellow",
                       "symptom": format!("This cluster has {replicas} unavailable replica shard{}.", if replicas == 1 { "" } else { "s" }),
                       "details": details,
                       "impacts": [{"id": "elasticsearch:health:shards_availability:impact:replica_unassigned", "severity": 2,
                                    "description": format!("Searches might be slower than usual. Fewer redundant copies of the data exist on {} {}. Repair the {} quickly to prevent data loss.",
                                                           affected.len(), if affected.len() == 1 { "index" } else { "indices" }, if affected.len() == 1 { "index" } else { "indices" }),
                                    "impact_areas": ["search"]}],
                       "diagnosis": [{"id": "elasticsearch:health:shards_availability:diagnosis:increase_tier_capacity_for_allocations:tier:data_content",
                                      "cause": "Elasticsearch isn't allowed to allocate some shards from these indices to any of the nodes in the desired data tier because there are not enough nodes in the [data_content] tier to allocate each shard copy on a different node.",
                                      "action": "Increase the number of nodes in this tier or decrease the number of replica shards in the affected indices.",
                                      "help_url": "https://ela.st/tier-capacity",
                                      "affected_resources": {"indices": affected}}]})
            }
        }
        "disk" => json!({"status": "green", "symptom": "The cluster has enough available disk space.",
            "details": {"indices_with_readonly_block": all.iter().filter(|s| s.setting("index.blocks.read_only_allow_delete").as_deref() == Some("true")).count(),
                        "nodes_with_enough_disk_space": 1, "nodes_with_unknown_disk_status": 0, "nodes_over_high_watermark": 0, "nodes_over_flood_stage_watermark": 0}}),
        "data_stream_lifecycle" => json!({"status": "green", "symptom": "Data streams are executing their lifecycles without issues",
            "details": {"stagnating_backing_indices_count": 0, "total_backing_indices_in_error": 0}}),
        "ilm" | "slm" => elasticsearch_engine::ilm::health_indicator(node, name),
        _ => json!({"status": "green", "symptom": "No file-based setting errors found.", "details": {"failure_streak": 0, "most_recent_failure": null}}),
    };
    if !verbose {
        if let Some(o) = v.as_object_mut() {
            o.remove("details");
            o.remove("diagnosis");
        }
    }
    v
}

fn health_report(node: &Node, req: &Req) -> R {
    let verbose = req.bool_param("verbose")?.unwrap_or(true);
    if let Some(sz) = req.param("size") {
        match sz.parse::<i64>() {
            Ok(n) if n >= 0 => {}
            _ => return Err(EsError::illegal_argument("The size parameter must be a positive integer")),
        }
    }
    let one = req.segs.get(1).cloned();
    let mut indicators = Map::new();
    match &one {
        Some(i) => {
            if !INDICATORS.contains(&i.as_str()) {
                return Err(EsError::resource_not_found(format!("Did not find indicator {i}")));
            }
            indicators.insert(i.clone(), indicator(node, i, verbose));
        }
        None => {
            for i in INDICATORS {
                indicators.insert((*i).into(), indicator(node, i, verbose));
            }
        }
    }
    let mut body = json!({"cluster_name": node.cluster_name});
    let status = if one.is_none() {
        let worst = indicators.values().filter_map(|v| v["status"].as_str()).max_by_key(|s| match *s {
            "green" => 0,
            "unknown" => 1,
            "yellow" => 2,
            _ => 3,
        });
        worst.unwrap_or("green").to_string()
    } else {
        // Single-indicator (drill-down) report: no top-level status in
        // the body; the usage stats count that indicator's status
        // (ES `GetHealthAction.Response` / `HealthApiStats.track`).
        indicators.values().next().and_then(|v| v["status"].as_str()).unwrap_or("green").to_string()
    };
    if one.is_none() {
        body["status"] = json!(status);
    }
    crate::usage::record_health(verbose, Some(&status), &indicators);
    body["indicators"] = Value::Object(indicators);
    Ok(Resp::ok(body))
}

// ---------------------------------------------------------------------------
// _migration/deprecations
// ---------------------------------------------------------------------------

fn deprecations(node: &Node, req: &Req, index: Option<&str>) -> R {
    let mut index_settings = Map::new();
    if let Some(i) = index {
        let _ = node.resolve(i, opts(req))?;
    }
    let _ = &mut index_settings;
    Ok(Resp::ok(json!({
        "cluster_settings": [],
        "node_settings": [],
        "ml_settings": [],
        "index_settings": index_settings,
        "data_streams": {},
        "templates": {},
        "ilm_policies": {}
    })))
}

// ---------------------------------------------------------------------------
// _cluster/allocation/explain, _cluster/reroute
// ---------------------------------------------------------------------------

fn current_node_json(node: &Node) -> Value {
    json!({"id": NODE_ID, "name": node.name, "transport_address": TRANSPORT_ADDRESS, "attributes": node_attributes(), "roles": ROLES, "weight_ranking": 1})
}

fn disk_info(node: &Node) -> Value {
    let f = sysinfo::fs(&node.data);
    let used = f.total.saturating_sub(f.available);
    let usage = |n: u64| {
        json!({"path": node.data.to_string_lossy(), "total_bytes": f.total, "used_bytes": used, "free_bytes": n,
               "free_disk_percent": if f.total == 0 { 0.0 } else { n as f64 * 100.0 / f.total as f64 },
               "used_disk_percent": if f.total == 0 { 0.0 } else { used as f64 * 100.0 / f.total as f64 }})
    };
    let mut sizes = Map::new();
    for svc in node.all_indices() {
        for sh in svc.local_shards() {
            sizes.insert(format!("[{}][{}][p]", svc.name, sh.id), json!(sh.shard.store_size()));
        }
    }
    json!({"nodes": { NODE_ID: {"node_name": node.name, "least_available": usage(f.available), "most_available": usage(f.available)} }, "shard_sizes": sizes,
           "shard_data_set_sizes": {}, "shard_paths": {}, "reserved_sizes": []})
}

fn allocation_explain(node: &Node, req: &Req) -> R {
    let has_body = !req.body.is_empty() || req.param("source").is_some();
    let body = req.json()?;
    let include_disk = req.flag("include_disk_info")?;
    let include_yes = req.flag("include_yes_decisions")?;
    if let Some(o) = body.as_object() {
        for k in o.keys() {
            if !matches!(k.as_str(), "index" | "shard" | "primary" | "current_node") {
                return Err(EsError::new(400, "x_content_parse_exception", format!("[1:2] [cluster/allocation/explain] unknown field [{k}]")));
            }
        }
    }
    // `RestClusterAllocationExplainAction` (9.2+): the shard comes from the
    // body or, without one, from the query parameters (never both).
    const URL_PARAMS: [&str; 4] = ["current_node", "index", "primary", "shard"];
    let (index, shard, primary, current_node) = if has_body {
        let given: Vec<&str> = URL_PARAMS.iter().copied().filter(|p| req.param(p).is_some()).collect();
        if !given.is_empty() {
            let list = given.iter().map(|p| format!("[{p}]")).collect::<Vec<_>>().join(", ");
            let noun = if given.len() == 1 { "parameter" } else { "parameters" };
            return Err(EsError::illegal_argument(format!("request [{}] contains unrecognized {noun}: {list}", req.path)));
        }
        (
            body.get("index").and_then(Value::as_str).map(str::to_string),
            body.get("shard").and_then(Value::as_i64),
            body.get("primary").and_then(Value::as_bool),
            body.get("current_node").and_then(Value::as_str).map(str::to_string),
        )
    } else {
        let shard = match req.param("shard") {
            Some(v) => Some(v.parse::<i64>().map_err(|_| EsError::illegal_argument(format!("Failed to parse int parameter [shard] with value [{v}]")))?),
            None => None,
        };
        (req.param("index").map(str::to_string), shard, req.bool_param("primary")?, req.param("current_node").map(str::to_string))
    };
    // `ClusterAllocationExplainRequest.validate`.
    if index.is_some() || shard.is_some() || primary.is_some() || current_node.is_some() {
        let mut missing = Vec::new();
        if index.is_none() {
            missing.push("index must be specified");
        }
        if shard.is_none() {
            missing.push("shard must be specified");
        }
        if primary.is_none() {
            missing.push("primary must be specified");
        }
        if !missing.is_empty() {
            return Err(EsError::action_validation(&missing));
        }
    }
    let (svc, primary) = match &index {
        None => {
            // First unassigned shard.
            match node.all_indices().into_iter().find(|s| s.setting_usize("index.number_of_replicas", 1) > 0) {
                Some(s) => (s, false),
                None => {
                    return Err(EsError::illegal_argument(format!(
                        "No shard was specified in the explain API request, so this request was interpreted as a request to explain the first unassigned shard that the cluster finds, but there are no unassigned shards in this cluster. Specify an index and shard number to explain an assigned shard instead.{}",
                        ""
                    )))
                }
            }
        }
        Some(i) => {
            let (shard, primary) = (shard.unwrap_or(0), primary.unwrap_or(false));
            let svc = node.get_index(i).ok_or_else(|| EsError::index_not_found(i))?;
            if shard < 0 || shard >= svc.number_of_shards() as i64 {
                return Err(EsError::new(404, "shard_not_found_exception", "no such shard").with("index_uuid", svc.uuid.clone()).with("shard", shard.to_string()).with("index", svc.name.clone()));
            }
            if !primary && svc.setting_usize("index.number_of_replicas", 1) == 0 {
                return Err(EsError::illegal_argument(format!("unable to find any replica shards [{i}][{shard}] to explain")));
            }
            (svc, primary)
        }
    };
    let mut out = json!({"index": svc.name, "shard": if index.is_some() { shard.unwrap_or(0) } else { 0 }, "primary": primary});
    if primary && svc.failed_shards().contains_key(&0) {
        // An unassigned primary: its recovery failed `max_retry` times.
        let at = elasticsearch_engine::date::from_millis(svc.creation_date).format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();
        let details = svc.allocation_failure_details(0).unwrap_or_default();
        out["current_state"] = json!("unassigned");
        out["unassigned_info"] = json!({"reason": "ALLOCATION_FAILED", "at": at, "failed_allocation_attempts": 5, "details": details, "last_allocation_status": "no"});
        out["can_allocate"] = json!("no");
        out["allocate_explanation"] = json!("Elasticsearch isn't allowed to allocate this shard to any of the nodes in the cluster. Choose a node to which you expect this shard to be allocated, find this node in the node-by-node explanation, and address the reasons which prevent Elasticsearch from allocating this shard there.");
        out["node_allocation_decisions"] = json!([{
            "node_id": NODE_ID, "node_name": node.name, "transport_address": TRANSPORT_ADDRESS, "node_attributes": node_attributes(), "roles": ROLES,
            "node_decision": "no", "weight_ranking": 1,
            "deciders": [{"decider": "max_retry", "decision": "NO",
                "explanation": format!("shard has exceeded the maximum number of retries [5] on failed allocation attempts - manually call [POST /_cluster/reroute?retry_failed&metric=none] to retry, and for more information, see [https://www.elastic.co/guide/en/elasticsearch/reference/8.19/cluster-allocation-explain.html#maximum-number-of-retries-exceeded] [unassigned_info[[reason=ALLOCATION_FAILED], at[{at}], failed_attempts[5], failed_nodes[[{NODE_ID}]], delayed=false, last_node[{NODE_ID}], details[{details}], allocation_status[deciders_no]]]")}]
        }]);
    } else if primary {
        out["current_state"] = json!("started");
        out["current_node"] = current_node_json(node);
        out["can_remain_on_current_node"] = json!("yes");
        out["can_rebalance_cluster"] = json!("yes");
        out["can_rebalance_to_other_node"] = json!("no");
        out["rebalance_explanation"] = json!("Elasticsearch cannot rebalance this shard to another node since there is no node to which allocation is permitted which would improve the cluster balance. If you expect this shard to be rebalanced to another node, find this node in the node-by-node explanation and address the reasons which prevent Elasticsearch from rebalancing this shard there.");
        if include_yes {
            out["can_remain_decisions"] = json!([]);
        }
    } else {
        let at = elasticsearch_engine::date::from_millis(svc.creation_date).format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();
        out["current_state"] = json!("unassigned");
        out["unassigned_info"] = json!({"reason": "INDEX_CREATED", "at": at, "last_allocation_status": "no_attempt"});
        out["can_allocate"] = json!("no");
        out["allocate_explanation"] = json!("Elasticsearch isn't allowed to allocate this shard to any of the nodes in the cluster. Choose a node to which you expect this shard to be allocated, find this node in the node-by-node explanation, and address the reasons which prevent Elasticsearch from allocating this shard there.");
        let mut deciders = Vec::new();
        if include_yes {
            for d in ["max_retry", "replica_after_primary_active", "enable", "node_version", "snapshot_in_progress", "restore_in_progress", "node_shutdown", "node_replacement", "filter", "throttling"] {
                deciders.push(json!({"decider": d, "decision": "YES", "explanation": "the decider returned YES"}));
            }
        }
        deciders.push(json!({"decider": "same_shard", "decision": "NO",
            "explanation": format!("a copy of this shard is already allocated to this node [[{}][0], node[{NODE_ID}], [P], s[STARTED], a[id={}]]", svc.name, svc.uuid)}));
        out["node_allocation_decisions"] = json!([{
            "node_id": NODE_ID, "node_name": node.name, "transport_address": TRANSPORT_ADDRESS, "node_attributes": node_attributes(), "roles": ROLES,
            "node_decision": "no", "weight_ranking": 1, "deciders": deciders
        }]);
    }
    if include_disk {
        out["cluster_info"] = disk_info(node);
    }
    Ok(Resp::ok(out))
}

fn reroute(node: &Node, req: &Req) -> R {
    let body = req.json()?;
    let explain = req.flag("explain")?;
    let dry_run = req.flag("dry_run")?;
    let retry = req.flag("retry_failed")?;
    if retry && !dry_run {
        // Retry the primaries whose allocation failed (`MaxRetryAllocationDecider` reset).
        for svc in node.all_indices() {
            if svc.is_open() && !svc.failed_shards().is_empty() {
                svc.try_allocate();
            }
        }
    }
    let commands = body.get("commands").and_then(Value::as_array).cloned().unwrap_or_default();
    let mut explanations = Vec::new();
    for c in &commands {
        let o = c.as_object().ok_or_else(|| EsError::parsing("[commands] must be an array of objects"))?;
        let (name, params) = o.iter().next().ok_or_else(|| EsError::parsing("empty allocation command"))?;
        if !matches!(name.as_str(), "move" | "cancel" | "allocate_replica" | "allocate_stale_primary" | "allocate_empty_primary") {
            return Err(EsError::parsing(format!("[cluster_reroute] unknown field [{name}]")).caused_by(EsError::new(400, "named_object_not_found_exception", format!("[1:1] unknown field [{name}]"))));
        }
        let index = params.get("index").and_then(Value::as_str).unwrap_or("").to_string();
        let shard = params.get("shard").and_then(Value::as_u64).unwrap_or(0);
        let node_param = params.get("node").or_else(|| params.get("from_node")).and_then(Value::as_str).unwrap_or("").to_string();
        let mut p = Map::new();
        p.insert("index".into(), json!(index));
        p.insert("shard".into(), json!(shard));
        let svc = node.get_index(&index);
        let (decider, explanation) = match name.as_str() {
            "cancel" => {
                p.insert("node".into(), json!(node_param));
                let ap = params.get("allow_primary").and_then(Value::as_bool).unwrap_or(false);
                p.insert("allow_primary".into(), json!(ap));
                let msg = match &svc {
                    Some(_) if shard == 0 && !ap => format!("can't cancel [{index}][0] on node {{{}}}{{{NODE_ID}}}, shard is primary and started", node.name),
                    Some(_) if shard == 0 => format!("can't cancel [{index}][0] on node {{{}}}: cancelling a started primary is not allowed on a single node", node.name),
                    _ => format!("can't cancel [{index}][{shard}], failed to find it on node {{{}}}{{{NODE_ID}}}", node.name),
                };
                ("cancel_allocation_command", msg)
            }
            "move" => {
                p.insert("from_node".into(), json!(node_param));
                let to = params.get("to_node").and_then(Value::as_str).unwrap_or("").to_string();
                p.insert("to_node".into(), json!(to));
                ("move_allocation_command", format!("[move_allocation] can't move [{index}][{shard}], from {{{}}}, to {to}, since its not allowed, reason: [NO(a copy of this shard is already allocated to this node)]", node.name))
            }
            other => {
                p.insert("node".into(), json!(node_param));
                if other != "allocate_replica" {
                    p.insert("accept_data_loss".into(), json!(params.get("accept_data_loss").and_then(Value::as_bool).unwrap_or(false)));
                }
                let cmd = match other {
                    "allocate_replica" => "allocate_replica_allocation_command",
                    "allocate_stale_primary" => "allocate_stale_primary_allocation_command",
                    _ => "allocate_empty_primary_allocation_command",
                };
                (cmd, format!("[{other}] all copies of [{index}][{shard}] are already assigned. Use the move allocation command instead"))
            }
        };
        if !explain && svc.is_none() {
            return Err(EsError::index_not_found(&index));
        }
        if !explain {
            return Err(EsError::illegal_argument(explanation));
        }
        explanations.push(json!({"command": name, "parameters": p, "decisions": [{"decider": decider, "decision": "NO", "explanation": explanation}]}));
    }
    let mut out = json!({"acknowledged": true});
    let metric = req.param("metric").unwrap_or("");
    let mut headers = Vec::new();
    // ES 9 (`cluster.reroute.ignores_metric_param`): no `state` in the
    // response and `?metric` is ignored with a warning; compatible-with=8
    // keeps the 8.x response.
    if !req.compat_v8() {
        if req.param("metric").is_some() {
            headers.push(crate::search_api::warning_header("the [?metric] query parameter to the [POST /_cluster/reroute] API has no effect; its use will be forbidden in a future version"));
        }
    } else if metric != "none" {
        let state_req = Req { method: Method::GET, path: "/_cluster/state".into(), segs: vec!["_cluster".into(), "state".into()], params: Default::default(), body: Default::default(), headers: Default::default(), access: None, extensions: Default::default(), api_version: req.api_version };
        let full = match crate::cluster::state(node, &state_req)?.body {
            crate::RespBody::Json(v) => v,
            _ => json!({}),
        };
        let mut st = Map::new();
        let keep: Vec<&str> = if metric.is_empty() || metric == "_all" {
            vec!["cluster_uuid", "version", "state_uuid", "master_node", "blocks", "nodes", "routing_table", "routing_nodes"]
        } else {
            metric.split(',').collect()
        };
        if let Some(o) = full.as_object() {
            for (k, v) in o {
                if k == "cluster_uuid" || keep.contains(&k.as_str()) {
                    st.insert(k.clone(), v.clone());
                }
            }
        }
        if keep.contains(&"routing_nodes") {
            st.insert("routing_nodes".into(), json!({"unassigned": [], "nodes": { NODE_ID: [] }}));
        }
        out["state"] = Value::Object(st);
        headers.push(("Warning", format!("299 Elasticsearch-{ES_VERSION}-{} \"The [state] field in the response to the reroute API is deprecated and will be removed in a future version. Specify ?metric=none to adopt the future behaviour.\"", crate::cluster::BUILD_HASH)));
    }
    if explain {
        out["explanations"] = Value::Array(explanations);
    }
    let mut r = Resp::ok(out);
    r.headers = headers;
    Ok(r)
}

// ---------------------------------------------------------------------------
// _recovery, _shard_stores
// ---------------------------------------------------------------------------

/// (type, start millis, stop millis) of an index's (only) recovery.
pub fn recovery_info(node: &Node, svc: &IndexService) -> (&'static str, i64, i64) {
    let node_start = elasticsearch_engine::date::now_millis() - node.uptime_millis() as i64;
    if svc.creation_date < node_start || !svc.is_open() {
        (if svc.is_open() || svc.creation_date < node_start { "EXISTING_STORE" } else { "EXISTING_STORE" }, node_start.max(svc.creation_date), node_start.max(svc.creation_date) + 5)
    } else {
        ("EMPTY_STORE", svc.creation_date, svc.creation_date + 3)
    }
}

fn iso(ms: i64) -> String {
    elasticsearch_engine::date::from_millis(ms).format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()
}

fn recovery(node: &Node, req: &Req, index: Option<&str>) -> R {
    let mut o = opts(req);
    o.expand_closed = true;
    let targets = node.resolve(index.unwrap_or("_all"), o)?;
    let active_only = req.flag("active_only")?;
    let detailed = req.flag("detailed")?;
    let human = req.flag("human")?;
    let mut out = Map::new();
    if !active_only {
        for t in &targets {
            let svc = &t.svc;
            let (mut ty, mut start, mut stop) = recovery_info(node, svc);
            let ops = 0;
            let mut files = json!({"total": 0, "reused": 0, "recovered": 0, "percent": "0.0%"});
            let mut rec_bytes = 0u64;
            if let Some((s0, s1, bytes, nfiles)) = node.snapshots.restored_info(&svc.name) {
                ty = "SNAPSHOT";
                start = s0;
                stop = s1.max(s0);
                rec_bytes = bytes;
                files = json!({"total": nfiles, "reused": 0, "recovered": nfiles, "percent": "100.0%"});
            }
            if detailed {
                files["details"] = json!([]);
            }
            let mut shard = json!({
                "id": 0, "type": ty, "stage": "DONE", "primary": true,
                "start_time_in_millis": start, "stop_time_in_millis": stop, "total_time_in_millis": stop - start,
                "source": if ty == "EXISTING_STORE" { json!({"bootstrap_new_history_uuid": false}) } else if ty == "SNAPSHOT" { json!({"repository": "", "snapshot": "", "version": "8.19.0", "index": svc.name, "restoreUUID": svc.uuid}) } else { json!({}) },
                "target": {"id": NODE_ID, "host": "127.0.0.1", "transport_address": TRANSPORT_ADDRESS, "ip": "127.0.0.1", "name": node.name},
                "index": {"size": {"total_in_bytes": rec_bytes, "reused_in_bytes": 0, "recovered_in_bytes": rec_bytes, "recovered_from_snapshot_in_bytes": 0, "percent": if rec_bytes > 0 { "100.0%" } else { "0.0%" }},
                          "files": files, "total_time_in_millis": 0, "source_throttle_time_in_millis": 0, "target_throttle_time_in_millis": 0},
                "translog": {"recovered": ops, "total": ops, "percent": "100.0%", "total_on_start": ops, "total_time_in_millis": 0},
                "verify_index": {"check_index_time_in_millis": 0, "total_time_in_millis": 0}
            });
            if human {
                shard["start_time"] = json!(iso(start));
                shard["stop_time"] = json!(iso(stop));
            }
            // One recovery per shard copy.
            let copies: Vec<Value> = (0..svc.number_of_shards().max(1))
                .map(|i| {
                    let mut c = shard.clone();
                    c["id"] = json!(i);
                    c
                })
                .collect();
            out.insert(svc.name.clone(), json!({ "shards": copies }));
        }
    }
    Ok(Resp::ok(Value::Object(out)))
}

fn shard_stores(node: &Node, req: &Req, index: Option<&str>) -> R {
    let mut o = opts(req);
    o.expand_closed = true;
    let targets = node.resolve(index.unwrap_or("_all"), o)?;
    let status = req.param("status").unwrap_or("yellow,red");
    let wanted: Vec<&str> = status.split(',').map(str::trim).collect();
    for w in &wanted {
        if !matches!(*w, "green" | "yellow" | "red" | "all") {
            return Err(EsError::illegal_argument(format!("unknown shard status [{w}]")));
        }
    }
    let mut indices = Map::new();
    for t in &targets {
        let h = health_of_svc(&t.svc);
        if !(wanted.contains(&"all") || wanted.contains(&h)) {
            continue;
        }
        let mut n = Map::new();
        n.insert(NODE_ID.into(), node_ident(node));
        let shards: Map<String, Value> = (0..t.svc.number_of_shards())
            .map(|i| {
                let alloc = t.svc.shard(i).map(|c| c.allocation_id.clone()).unwrap_or_else(|| t.svc.uuid.clone());
                (i.to_string(), json!({"stores": [{ NODE_ID: n[NODE_ID], "allocation_id": alloc, "allocation": "primary" }]}))
            })
            .collect();
        indices.insert(t.svc.name.clone(), json!({ "shards": shards }));
    }
    Ok(Resp::ok(json!({"indices": indices})))
}

// ---------------------------------------------------------------------------
// _disk_usage, _field_usage_stats
// ---------------------------------------------------------------------------

#[derive(Default, Clone, Copy)]
struct Usage {
    inverted: u64,
    stored: u64,
    doc_values: u64,
    points: u64,
    norms: u64,
    knn: u64,
}

impl Usage {
    fn total(&self) -> u64 {
        self.inverted + self.stored + self.doc_values + self.points + self.norms + self.knn
    }
    fn json(&self) -> Value {
        let b = |n: u64| json!(bytes_fmt(n, None));
        json!({
            "total": b(self.total()), "total_in_bytes": self.total(),
            "inverted_index": {"total": b(self.inverted), "total_in_bytes": self.inverted},
            "stored_fields": b(self.stored), "stored_fields_in_bytes": self.stored,
            "doc_values": b(self.doc_values), "doc_values_in_bytes": self.doc_values,
            "points": b(self.points), "points_in_bytes": self.points,
            "norms": b(self.norms), "norms_in_bytes": self.norms,
            "term_vectors": "0b", "term_vectors_in_bytes": 0,
            "knn_vectors": b(self.knn), "knn_vectors_in_bytes": self.knn
        })
    }
    fn add(&mut self, o: &Usage) {
        self.inverted += o.inverted;
        self.stored += o.stored;
        self.doc_values += o.doc_values;
        self.points += o.points;
        self.norms += o.norms;
        self.knn += o.knn;
    }
}

/// Per ES field, the bytes `leaves` hold: a sealed segment's real on-disk
/// bytes per field and structure ([`rustic_core::segment::Segment::disk_usage_sampled`]:
/// exact but for the per-field split of large stores, sampled from 64
/// chunks), in ES's categories (terms and postings: inverted index;
/// columns: doc values, or points for a field mapped without doc values,
/// whose column serves its range queries; geo and shape indexes: points;
/// per-field statistics: inverted index);
/// a memtable's (ES analyzes the last commit: only documents not flushed)
/// estimated from its terms and stored documents. Bytes of no field
/// (section framing) are returned apart.
fn leaf_disk_usage(svc: &Arc<elasticsearch_engine::index::IndexService>, lookup: &elasticsearch_engine::mapping::FieldLookup,
    leaves: &[elasticsearch_engine::rustic_core::query::LeafReader<'_>], synthetic: bool, live: u64) -> Result<(BTreeMap<String, Usage>, Usage), EsError> {
    use elasticsearch_engine::mapping as m;
    use elasticsearch_engine::rustic_core::query::LeafReader;
    let source_name = if synthetic { "_ignored_source" } else { "_source" };
    // Core field id -> ES field name (an auxiliary column `<field>#<x>`
    // counts for its field).
    let mut names: BTreeMap<u16, String> = BTreeMap::new();
    for (id, name) in [(m::ID_FIELD, "_id"), (m::SOURCE_FIELD, source_name), (m::VERSION_FIELD, "_version"), (m::SEQ_NO_FIELD, "_seq_no"), (m::ROUTING_FIELD, "_routing"), (m::IGNORED_FIELD, "_ignored"), (m::PRIMARY_TERM_FIELD, "_primary_term"), (m::FIELD_NAMES_FIELD, "_field_names"), (m::IGNORED_SOURCE_FIELD, "_ignored_source")] {
        names.insert(id, name.to_string());
    }
    for (key, id) in &svc.meta().mappings.core_ids {
        let base = key.split_once('#').map_or(key.as_str(), |(b, _)| b);
        names.entry(*id).or_insert_with(|| base.to_string());
    }
    let mut fields: BTreeMap<String, Usage> = BTreeMap::new();
    let mut framing = Usage::default();
    let mut memtables = Vec::new();
    for l in leaves {
        let LeafReader::Segment(seg) = l else {
            memtables.push(*l);
            continue;
        };
        let du = seg.disk_usage_sampled(64).map_err(|e| EsError::internal(e.to_string()))?;
        for (id, u) in du.fields {
            let (e, no_doc_values) = if id == u16::MAX {
                (&mut framing, false)
            } else {
                let name = names.get(&id).cloned().unwrap_or_else(|| format!("_core_{id}"));
                let no_dv = lookup.fields.get(&name).is_some_and(|f| !f.doc_values && f.core_id == Some(id));
                (fields.entry(name).or_default(), no_dv)
            };
            let column = u.doc_values + u.doc_values_terms;
            e.inverted += u.terms + u.postings;
            if matches!(id, m::SEQ_NO_FIELD | m::VERSION_FIELD | m::PRIMARY_TERM_FIELD) {
                // Stored copies of a column (formats before v15; ES keeps
                // these as doc values only).
                e.doc_values += u.stored;
            } else {
                e.stored += u.stored;
            }
            if id == m::ID_FIELD {
                // `_id` has no doc values in ES; a column of it (formats
                // before v15) is derivable from its postings.
                e.inverted += column;
            } else if no_doc_values { e.points += column } else { e.doc_values += column }
            e.points += u.points + u.geo + u.shape;
            e.norms += u.norms;
            // The extra section's per-field entries (collection statistics,
            // nested ordinals): inverted-index metadata in Lucene.
            e.inverted += u.other_extra;
            e.knn += u.vectors;
        }
    }
    // Memtables: their terms (key bytes and a doc each) and stored values;
    // columns as their docs' share of the segment columns' bytes a doc.
    let mem_docs: u64 = memtables.iter().map(|l| l.num_docs() as u64).sum();
    if mem_docs > 0 {
        let seg_docs = live.saturating_sub(mem_docs).max(1);
        for (name, f) in lookup.fields.iter() {
            let Some(id) = f.core_id else { continue };
            let mut u = Usage::default();
            if f.indexed {
                for l in &memtables {
                    for (term, _) in l.term_range(id, None, None) {
                        u.inverted += term.len() as u64 + 4;
                    }
                }
            }
            if let Some(seg) = fields.get(name) {
                u.doc_values += seg.doc_values * mem_docs / seg_docs;
                u.points += seg.points * mem_docs / seg_docs;
                u.norms += seg.norms * mem_docs / seg_docs;
            } else if f.doc_values && !f.ty.is_text() {
                u.doc_values += mem_docs * if f.ty.is_keyword_like() { 2 } else { 8 };
            }
            fields.entry(name.clone()).or_default().add(&u);
        }
        let mut idu = Usage::default();
        for l in &memtables {
            for (term, _) in l.term_range(m::ID_FIELD, None, None) {
                idu.inverted += term.len() as u64 + 4;
            }
            let n = l.num_docs();
            for start in (0..n).step_by(1024) {
                let docs: Vec<_> = (start..start.saturating_add(1024).min(n)).collect();
                l.for_each_doc(&docs, |_, doc| {
                    for (fid, v) in &doc.fields {
                        let bytes = v.as_text().map_or(8, |t| t.len() as u64);
                        let name = names.get(fid).cloned().unwrap_or_else(|| format!("_core_{fid}"));
                        fields.entry(name).or_default().stored += bytes;
                    }
                })?;
            }
        }
        fields.entry("_id".into()).or_default().add(&idu);
    }
    Ok((fields, framing))
}

fn disk_usage(node: &Node, req: &Req, index: &str) -> R {
    if !req.flag("run_expensive_tasks")? {
        return Err(EsError::illegal_argument(
            "analyzing the disk usage of an index is expensive and resource-intensive, the parameter [run_expensive_tasks] must be set to [true] in order for the task to be performed.",
        ));
    }
    let targets = node.resolve(index, opts(req))?;
    let mut out = Map::new();
    let n = targets.len();
    out.insert("_shards".into(), json!({"total": n, "successful": n, "failed": 0}));
    for t in &targets {
        let svc = &t.svc;
        let store = svc.store_size();
        let lookup = svc.lookup();
        let copies = svc.local_shards();
        // ES analyzes the last commit (flushing first unless `flush=false`).
        let flush = req.bool_param("flush")?.unwrap_or(true);
        if flush {
            for c in &copies {
                c.shard.flush()?;
            }
        }
        // The published reader views (published memtables included, no seal)
        // plus each copy's sealed-but-unpublished segments (after a flush:
        // the documents not yet refreshed), taken under a short read; the
        // analysis below holds no shard lock. Documents still in memtables
        // being written are not analyzed (ES analyzes the last commit).
        let views: Vec<_> = copies.iter().map(|c| c.shard.reader_view()).collect();
        let hidden: Vec<_> = copies.iter().flat_map(|c| c.shard.core.read().hidden_segments().to_vec()).collect();
        let mut leaves: Vec<elasticsearch_engine::rustic_core::query::LeafReader<'_>> = views.iter().flat_map(|v| v.leaves()).collect();
        leaves.extend(hidden.iter().map(|s| elasticsearch_engine::rustic_core::query::LeafReader::Segment(s)));
        let live: u64 = leaves.iter().map(|l| l.num_docs() as u64).sum();
        // A synthetic-source index stores no `_source` field: what it keeps
        // (the residual the columns cannot rebuild) is ES's `_ignored_source`.
        let synthetic = {
            let m = svc.meta();
            elasticsearch_engine::synthetic::source_mode(
                elasticsearch_engine::settings::get_str(&m.settings, "index.mapping.source.mode"),
                &m.mappings,
                elasticsearch_engine::settings::get_str(&m.settings, "index.mode"),
            ) == elasticsearch_engine::synthetic::SourceMode::Synthetic
        };
        let (mut fields, framing) = leaf_disk_usage(svc, &lookup, &leaves, synthetic, live)?;
        drop(leaves);
        drop(hidden);
        drop(views);
        // Raw vectors live in the vector store's files: rows x row bytes.
        for (name, f) in lookup.fields.iter() {
            if f.ty == elasticsearch_engine::mapping::EsType::DenseVector && f.core_id.is_some() {
                fields.entry(name.clone()).or_default().knn += live * elasticsearch_engine::vectors::stored_row_bytes(f);
            }
        }
        fields.retain(|_, u| u.total() > 0);
        let mut all = framing;
        for u in fields.values() {
            all.add(u);
        }
        let fm: Map<String, Value> = fields.iter().map(|(k, u)| (k.clone(), u.json())).collect();
        out.insert(svc.name.clone(), json!({"store_size": bytes_fmt(store, None), "store_size_in_bytes": store, "all_fields": all.json(), "fields": fm}));
    }
    Ok(Resp::ok(Value::Object(out)))
}

fn field_usage_stats(node: &Node, req: &Req, index: &str) -> R {
    let targets = node.resolve(index, opts(req))?;
    let field_filter: Option<Vec<String>> = req.param("fields").map(|f| f.split(',').map(str::to_string).collect());
    let render = |u: &elasticsearch_engine::field_usage::FieldUse| {
        json!({"any": u.any(), "inverted_index": {"terms": u.terms, "postings": u.postings, "proximity": u.proximity, "positions": u.positions, "term_frequencies": u.term_frequencies, "offsets": u.offsets, "payloads": 0},
               "stored_fields": u.stored_fields, "doc_values": u.doc_values, "points": u.points, "norms": u.norms, "term_vectors": u.term_vectors, "knn_vectors": u.knn_vectors})
    };
    let mut out = Map::new();
    let n = targets.len();
    out.insert("_shards".into(), json!({"total": n, "successful": n, "failed": 0}));
    for t in &targets {
        let mut fields = Map::new();
        let usage = elasticsearch_engine::field_usage::snapshot(&t.svc.uuid);
        let mut total = elasticsearch_engine::field_usage::FieldUse::default();
        for u in usage.values() {
            total.terms += u.terms;
            total.postings += u.postings;
            total.stored_fields += u.stored_fields;
            total.doc_values += u.doc_values;
            total.knn_vectors += u.knn_vectors;
            total.proximity += u.proximity;
            total.positions += u.positions;
            total.term_frequencies += u.term_frequencies;
            total.points += u.points;
            total.norms += u.norms;
            total.term_vectors += u.term_vectors;
            total.offsets += u.offsets;
        }
        let mut all_any = 0;
        for (name, u) in &usage {
            if field_filter.as_ref().map_or(true, |ff| ff.iter().any(|p| elasticsearch_engine::mapping::wildcard_match(p, name))) {
                fields.insert(name.clone(), render(u));
            }
            all_any += u.any();
        }
        if let Some(ff) = &field_filter {
            let lk = t.svc.lookup();
            let zero = elasticsearch_engine::field_usage::FieldUse::default();
            for name in lk.fields.keys() {
                if !usage.contains_key(name) && ff.iter().any(|p| elasticsearch_engine::mapping::wildcard_match(p, name)) {
                    fields.insert(name.clone(), render(&zero));
                }
            }
        }
        let mut all = render(&total);
        all["any"] = json!(all_any);
        out.insert(
            t.svc.name.clone(),
            json!({"shards": [{
                "tracking_id": t.svc.uuid, "tracking_started_at_millis": t.svc.creation_date,
                "routing": {"state": "STARTED", "primary": true, "node": NODE_ID, "relocating_node": null},
                "stats": {"all_fields": all, "fields": fields}
            }]}),
        );
    }
    Ok(Resp::ok(Value::Object(out)))
}

// ---------------------------------------------------------------------------
// _block, _reload_search_analyzers
// ---------------------------------------------------------------------------

fn add_block(node: &Node, req: &Req, index: &str) -> R {
    let block = req.seg(2).to_string();
    if !matches!(block.as_str(), "metadata" | "read" | "read_only" | "read_only_allow_delete" | "write") {
        return Err(EsError::illegal_argument(format!("No block found with name [{block}]")));
    }
    if index == "_all" || index.split(',').any(|p| p.contains('*') || p == "_all") {
        if node.destructive_requires_name() {
            return Err(EsError::illegal_argument("Wildcard expressions or all indices are not allowed"));
        }
    }
    let add = req.method == Method::PUT;
    let mut o = opts(req);
    o.expand_closed = true;
    let targets = node.resolve(index, o)?;
    let key = format!("index.blocks.{block}");
    let mut list = Vec::new();
    for t in &targets {
        let mut flat = settings::FlatSettings::new();
        flat.insert(key.clone(), if add { json!("true") } else { Value::Null });
        t.svc.put_settings(flat, false)?;
        list.push(if add { json!({"name": t.svc.name, "blocked": true}) } else { json!({"name": t.svc.name, "unblocked": true}) });
    }
    if add {
        Ok(Resp::ok(json!({"acknowledged": true, "shards_acknowledged": true, "indices": list})))
    } else {
        Ok(Resp::ok(json!({"acknowledged": true, "indices": list})))
    }
}

fn reload_search_analyzers(node: &Node, req: &Req, index: &str) -> R {
    let targets = node.resolve(index, opts(req))?;
    // `?resource=`: only filters loading that resource (synonyms set / file) reload.
    let resource = req.param("resource").filter(|r| !r.is_empty());
    let mut details = Vec::new();
    for t in &targets {
        let s = t.svc.meta().settings;
        let uses_resource = |f: &str| match resource {
            None => true,
            Some(r) => ["synonyms_set", "synonyms_path"].iter().any(|k| s.get(&format!("index.analysis.filter.{f}.{k}")).and_then(Value::as_str) == Some(r)),
        };
        let updateable: BTreeSet<String> = s
            .iter()
            .filter(|(k, v)| k.starts_with("index.analysis.filter.") && k.ends_with(".updateable") && v.as_str() == Some("true"))
            .filter_map(|(k, _)| k.strip_prefix("index.analysis.filter.").and_then(|r| r.strip_suffix(".updateable")).map(str::to_string))
            .filter(|f| uses_resource(f))
            .collect();
        let mut reloaded = Vec::new();
        for (k, v) in &s {
            let Some(rest) = k.strip_prefix("index.analysis.analyzer.") else { continue };
            let Some(name) = rest.strip_suffix(".filter") else { continue };
            let uses = match v {
                Value::Array(a) => a.iter().any(|f| f.as_str().map(|x| updateable.contains(x)).unwrap_or(false)),
                Value::String(x) => updateable.contains(x),
                _ => false,
            };
            if uses {
                reloaded.push(name.to_string());
            }
        }
        details.push(json!({"index": t.svc.name, "reloaded_analyzers": reloaded, "reloaded_node_ids": if reloaded.is_empty() { json!([]) } else { json!([NODE_ID]) }}));
    }
    let n = targets.len();
    Ok(Resp::ok(json!({"_shards": {"total": n, "successful": n, "failed": 0}, "reload_details": details})))
}

// ---------------------------------------------------------------------------
// _resolve/index, _search_shards
// ---------------------------------------------------------------------------

fn resolve_index(node: &Node, req: &Req) -> R {
    Ok(Resp::ok(resolve_index_value(node, req)?))
}

/// `IndexMode` names (`IndexMode.fromString`), in declaration order.
const INDEX_MODES: &[&str] = &["standard", "time_series", "logsdb", "lookup", "columnar", "logsdb_columnar", "vectordb_document"];

fn index_mode_of(s: &IndexService) -> String {
    s.setting("index.mode").map(|m| m.to_ascii_lowercase()).filter(|m| !m.is_empty()).unwrap_or_else(|| "standard".into())
}

fn resolve_index_value(node: &Node, req: &Req) -> Result<Value, EsError> {
    resolve_index_value_for(node, req, req.seg(2))
}

fn resolve_index_value_for(node: &Node, req: &Req, expr: &str) -> Result<Value, EsError> {
    let expr = expr.to_string();
    let o = opts(req);
    // `RestResolveIndexAction`: `?mode=` filters on the index mode.
    let modes: Vec<String> = match req.param("mode").filter(|m| !m.is_empty()) {
        None => Vec::new(),
        Some(m) => {
            let mut out = Vec::new();
            for v in m.split(',').map(str::trim) {
                let v = v.to_ascii_lowercase();
                if !INDEX_MODES.contains(&v.as_str()) {
                    return Err(EsError::illegal_argument(format!("[{v}] is an invalid index mode, valid modes are: [{}]", INDEX_MODES.join(","))));
                }
                out.push(v);
            }
            out
        }
    };
    let mode_ok = |s: &IndexService| modes.is_empty() || modes.contains(&index_mode_of(s));
    let all = node.all_indices();
    let is_hidden = |s: &IndexService| s.setting("index.hidden").as_deref() == Some("true") || s.name.starts_with('.');
    let ds_all = node.data_streams_matching("*", true);
    let mut idx_sel: BTreeSet<String> = BTreeSet::new();
    let mut alias_sel: BTreeSet<String> = BTreeSet::new();
    let mut ds_sel: BTreeSet<String> = BTreeSet::new();
    let parts: Vec<String> = if expr.is_empty() { vec!["*".into()] } else { expr.split(',').map(|s| s.trim().to_string()).collect() };
    for (pi, raw) in parts.iter().enumerate() {
        let (exclude, p) = match raw.strip_prefix('-') {
            Some(r) if pi > 0 => (true, r.to_string()),
            _ => (false, raw.clone()),
        };
        let p = elasticsearch_engine::node::resolve_date_math(&p)?;
        let pat = if p == "_all" { "*".to_string() } else { p.clone() };
        let wildcard = pat.contains('*');
        let mut mi = BTreeSet::new();
        let mut ma = BTreeSet::new();
        let mut md = BTreeSet::new();
        for s in &all {
            if !elasticsearch_engine::mapping::wildcard_match(&pat, &s.name) {
                continue;
            }
            if wildcard {
                let hidden = is_hidden(s);
                if hidden && !o.expand_hidden && !(pat.starts_with('.')) {
                    continue;
                }
                if (s.is_open() && !o.expand_open) || (!s.is_open() && !o.expand_closed) {
                    continue;
                }
            }
            mi.insert(s.name.clone());
        }
        for s in &all {
            for (a, def) in s.aliases() {
                if elasticsearch_engine::mapping::wildcard_match(&pat, &a) {
                    let hidden = def.get("is_hidden").and_then(Value::as_bool).unwrap_or(false);
                    if wildcard && hidden && !o.expand_hidden {
                        continue;
                    }
                    ma.insert(a);
                }
            }
        }
        for d in &ds_all {
            if elasticsearch_engine::mapping::wildcard_match(&pat, &d.name) && (!wildcard || !d.hidden || o.expand_hidden) {
                md.insert(d.name.clone());
            }
        }
        if !wildcard && !exclude && mi.is_empty() && ma.is_empty() && md.is_empty() && !o.ignore_unavailable {
            return Err(EsError::index_not_found(&p));
        }
        if exclude {
            idx_sel.retain(|x| !mi.contains(x));
            alias_sel.retain(|x| !ma.contains(x));
            ds_sel.retain(|x| !md.contains(x));
        } else {
            idx_sel.extend(mi);
            alias_sel.extend(ma);
            ds_sel.extend(md);
        }
    }
    let mut indices = Vec::new();
    for name in &idx_sel {
        let Some(s) = node.get_index(name) else { continue };
        if !mode_ok(&s) {
            continue;
        }
        let al: Vec<String> = s.aliases().keys().cloned().collect();
        let mut attrs = vec![if s.is_open() { "open" } else { "closed" }];
        if is_hidden(&s) {
            attrs.push("hidden");
        }
        if s.name.starts_with(".security") || s.name.starts_with(".kibana") || s.name.starts_with(".tasks") {
            attrs.push("system");
        }
        attrs.sort();
        let mut e = json!({"name": s.name});
        if !al.is_empty() {
            e["aliases"] = json!(al);
        }
        e["attributes"] = json!(attrs);
        if let Some(ds) = node.data_stream_of_index(&s.name) {
            e["data_stream"] = json!(ds);
        }
        e["mode"] = json!(index_mode_of(&s));
        indices.push(e);
    }
    let aliases: Vec<Value> = alias_sel
        .iter()
        .filter_map(|a| {
            let mut ix: Vec<String> = all.iter().filter(|s| s.aliases().contains_key(a) && mode_ok(s)).map(|s| s.name.clone()).collect();
            if !modes.is_empty() && ix.is_empty() {
                return None;
            }
            ix.sort();
            Some(json!({"name": a, "indices": ix}))
        })
        .collect();
    let dss: Vec<Value> = ds_all
        .iter()
        .filter(|d| ds_sel.contains(&d.name))
        .filter_map(|d| {
            let backing: Vec<&String> = d.indices.iter().filter(|i| modes.is_empty() || node.get_index(i).is_some_and(|s| mode_ok(&s))).collect();
            if !modes.is_empty() && backing.is_empty() {
                return None;
            }
            Some(json!({"name": d.name, "backing_indices": backing, "timestamp_field": "@timestamp"}))
        })
        .collect();
    Ok(json!({"indices": indices, "aliases": aliases, "data_streams": dss}))
}

/// `cluster.stats` `snapshots.repositories`: type and (idle) operation counts per repository.
fn snapshot_repository_stats(node: &Node, human: bool) -> Value {
    let now = elasticsearch_engine::date::now_millis();
    let repos = node.snapshots.get_repositories(None).unwrap_or_default();
    let mut out = Map::new();
    if let Some(m) = repos.as_object() {
        for (name, v) in m {
            out.insert(name.clone(), json!({
                "type": v["type"].as_str().unwrap_or("fs"),
                "oldest_start_time_millis": now,
                "current_counts": {"snapshots": 0, "clones": 0, "finalizations": 0, "deletions": 0, "snapshot_deletions": 0,
                                   "active_deletions": 0,
                                   "shards": {"total": 0, "complete": 0, "incomplete": 0,
                                              "states": {"INIT": 0, "SUCCESS": 0, "FAILED": 0, "ABORTED": 0, "MISSING": 0, "WAITING": 0, "QUEUED": 0, "PAUSED_FOR_NODE_REMOVAL": 0}}}
            }));
            if human {
                out[name.as_str()]["oldest_start_time"] = json!(iso(now));
            }
        }
    }
    Value::Object(out)
}

/// `GET /_resolve/cluster[/{expr}]` (`TransportResolveClusterAction`): the
/// local cluster only when the expression has local parts, and an entry per
/// named remote cluster (every configured one without an expression). This
/// engine has no remote transport, so remotes report `connected: false`.
fn resolve_cluster(node: &Node, req: &Req) -> R {
    let expr = req.seg(2);
    let (local_expr, mut out) = elasticsearch_engine::ccr::remote::resolve_cluster_remotes(node, (!expr.is_empty()).then_some(expr))?;
    if let Some(local_expr) = local_expr {
        use elasticsearch_engine::version as v;
        let version = json!({"number": v::ES_VERSION, "build_flavor": v::BUILD_FLAVOR, "minimum_wire_compatibility_version": v::MIN_WIRE_COMPAT, "minimum_index_compatibility_version": v::MIN_INDEX_COMPAT});
        let local = match resolve_index_value_for(node, req, &local_expr) {
            Ok(v) => {
                let open_index = v["indices"].as_array().map(|a| a.iter().any(|i| i["attributes"].as_array().map(|at| at.iter().any(|x| x == "open")).unwrap_or(false))).unwrap_or(false);
                let any = |k: &str| v[k].as_array().map(|a| !a.is_empty()).unwrap_or(false);
                json!({"connected": true, "skip_unavailable": false, "matching_indices": open_index || any("aliases") || any("data_streams"), "version": version})
            }
            Err(e) => json!({"connected": true, "skip_unavailable": false, "error": e.reason}),
        };
        out.insert("(local)".into(), local);
    }
    Ok(Resp::ok(Value::Object(out)))
}

/// `{total, min, max, average, std_dev}` over a set of values.
fn balance_summary(v: &[f64]) -> Value {
    let n = v.len().max(1) as f64;
    let total: f64 = v.iter().sum();
    let avg = total / n;
    let var = v.iter().map(|x| (x - avg).powi(2)).sum::<f64>() / n;
    json!({
        "total": total,
        "min": v.iter().cloned().fold(f64::INFINITY, f64::min).min(f64::MAX).max(0.0),
        "max": v.iter().cloned().fold(0.0, f64::max),
        "average": avg,
        "std_dev": var.sqrt()
    })
}

fn desired_balance(node: &Node, req: &Req) -> R {
    use elasticsearch_engine::node::NODE_ID;
    let targets = node.resolve("_all", { let mut o = opts(req); o.expand_closed = false; o })?;
    let mut routing = Map::new();
    let (mut shards, mut unassigned, mut disk) = (0u64, 0u64, 0u64);
    for t in &targets {
        let n = t.svc.setting_usize("index.number_of_shards", 1);
        let replicas = t.svc.setting_usize("index.number_of_replicas", 1) as u64;
        let size = t.svc.store_size();
        let mut per = Map::new();
        for i in 0..n {
            per.insert(i.to_string(), json!({
                "current": [{
                    "state": "STARTED", "primary": true, "node": NODE_ID, "node_is_desired": true,
                    "relocating_node": null, "relocating_node_is_desired": false,
                    "shard_id": i, "index": t.svc.name, "tier_preference": ["data_content"]
                }],
                "desired": {"node_ids": [NODE_ID], "total": 1, "unassigned": replicas, "ignored": 0}
            }));
        }
        shards += n as u64;
        unassigned += replicas * n as u64;
        disk += size;
        routing.insert(t.svc.name.clone(), Value::Object(per));
    }
    let sh = [shards as f64];
    let wl = [0.0];
    let du = [disk as f64];
    let ud = [0.0];
    Ok(Resp::ok(json!({
        "stats": {
            "computation_submitted": 0, "computation_executed": 0, "computation_converged": 0,
            "computation_iterations": 0, "computation_converged_index": 0, "computation_time_in_millis": 0,
            "reconciliation_time_in_millis": 0, "unassigned_shards": unassigned, "total_allocations": shards,
            "undesired_allocations": 0, "undesired_allocations_ratio": 0.0, "computed_shard_movements": 0
        },
        "cluster_balance_stats": {
            "shard_count": shards,
            "undesired_shard_allocation_count": 0,
            "tiers": {"data_content": {
                "shard_count": balance_summary(&sh),
                "undesired_shard_allocation_count": balance_summary(&ud),
                "forecast_write_load": balance_summary(&wl),
                "forecast_disk_usage": balance_summary(&du),
                "actual_disk_usage": balance_summary(&du)
            }},
            "nodes": {node.name.clone(): {
                "node_id": NODE_ID, "roles": ["data_content", "data", "ingest", "master"],
                "shard_count": shards, "undesired_shard_allocation_count": 0,
                "forecast_write_load": 0.0, "forecast_disk_usage_bytes": disk, "actual_disk_usage_bytes": disk,
                // `WeightFunction.calculateNodeWeight`: a single node sits at every average.
                "node_weight": 0.0
            }}
        },
        "cluster_info": {"nodes": {}, "shard_sizes": {}, "shard_data_set_sizes": {}, "shard_paths": {}, "reserved_sizes": []},
        "routing_table": routing
    })))
}

fn prevalidate_node_removal(node: &Node, req: &Req) -> R {
    use elasticsearch_engine::node::NODE_ID;
    let present: Vec<&str> = ["names", "ids", "external_ids"].into_iter().filter(|k| req.param(k).is_some()).collect();
    if present.is_empty() {
        return Err(EsError::action_validation(&["request must contain one of the parameters 'names', 'ids', or 'external_ids'"]));
    }
    if present.len() > 1 {
        return Err(EsError::action_validation(&["request must contain only one of the parameters 'names', 'ids', or 'external_ids'"]));
    }
    let kind = present[0];
    let wanted: Vec<String> = req.param(kind).unwrap_or_default().split(',').filter(|s| !s.is_empty()).map(str::to_string).collect();
    let mine = match kind {
        "ids" => NODE_ID,
        _ => node.name.as_str(),
    };
    let missing: Vec<&String> = wanted.iter().filter(|w| w.as_str() != mine).collect();
    if !missing.is_empty() {
        let what = match kind { "ids" => "ids", "names" => "names", _ => "external IDs" };
        return Err(EsError::resource_not_found(format!("could not resolve node {what} [{}]", missing.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))));
    }
    Ok(Resp::ok(json!({
        "is_safe": true,
        "message": "",
        "nodes": [{
            "id": NODE_ID, "name": node.name, "external_id": node.name,
            "result": {"is_safe": true, "reason": "no_problems", "message": ""}
        }]
    })))
}

/// `POST [/{index}]/_cache/clear`: clears the request cache of the target
/// indices (the query / fielddata caches hold nothing between requests) and
/// reports the shard copies it reached.
fn clear_cache(node: &Node, req: &Req, index: Option<&str>) -> R {
    for flag in ["fielddata", "query", "request"] {
        req.bool_param(flag)?;
    }
    let targets = node.resolve(index.unwrap_or("_all"), opts(req))?;
    let mut total = 0usize;
    let mut ok = 0usize;
    let mut seen = BTreeSet::new();
    for t in &targets {
        if !seen.insert(t.svc.name.clone()) || !t.svc.is_open() {
            continue;
        }
        let n = t.svc.number_of_shards() as usize;
        total += n * (1 + t.svc.setting_usize("index.number_of_replicas", 1));
        ok += n;
    }
    crate::request_cache::clear(&targets.iter().map(|t| t.svc.uuid.clone()).collect::<Vec<_>>());
    Ok(Resp::ok(json!({"_shards": {"total": total, "successful": ok, "failed": 0}})))
}

fn remote_info(node: &Node, _req: &Req) -> R {
    Ok(Resp::ok(elasticsearch_engine::ccr::remote::remote_info(node)))
}

fn segments_global(node: &Node, req: &Req, index: Option<&str>) -> R {
    use elasticsearch_engine::node::NODE_ID;
    let targets = node.resolve(index.unwrap_or("_all"), opts(req))?;
    let mut indices = Map::new();
    let mut total = 0usize;
    for t in &targets {
        let n_shards = t.svc.setting_usize("index.number_of_shards", 1);
        let replicas = t.svc.setting_usize("index.number_of_replicas", 1);
        total += n_shards * (replicas + 1);
        let mut shards = Map::new();
        for sh in t.svc.local_shards() {
            // The published reader view, published memtables included (as
            // the segments they seal into): no seal, no shard lock.
            let view = sh.shard.reader_view();
            let leaves = view.leaves();
            let mut seg_map = Map::new();
            let details = view.segment_stats();
            for entry in &details {
                let i = entry.id;
                // ES `num_docs`: live documents (maxDoc minus deletions).
                let docs = entry.docs.saturating_sub(entry.deleted);
                let legacy = view.segments().iter().any(|seg| seg.id == i && seg.legacy_presence());
                seg_map.insert(format!("_{i}"), json!({
                    "generation": i, "num_docs": docs, "deleted_docs": entry.deleted, "size_in_bytes": entry.bytes, "memory_in_bytes": 0,
                    "committed": entry.committed, "search": entry.search, "version": elasticsearch_engine::version::LUCENE_VERSION, "compound": false,
                    "attributes": if legacy { json!({"legacy_presence": "true"}) } else { json!({}) }
                }));
            }
            let copy = json!({
                "routing": {"state": "STARTED", "primary": true, "node": NODE_ID},
                "num_committed_segments": details.iter().filter(|s| s.committed).count(), "num_search_segments": leaves.len(),
                "segments": seg_map
            });
            shards.insert(sh.id.to_string(), json!([copy]));
        }
        indices.insert(t.svc.name.clone(), json!({"shards": shards}));
    }
    Ok(Resp::ok(json!({"_shards": {"total": total, "successful": total, "failed": 0}, "indices": indices})))
}

/// The common query shapes as ES re-renders a parsed query (`toXContent`):
/// explicit `value` / `boost`, `adjust_pure_negative` on `bool`.
fn normalize_query(v: &Value) -> Value {
    let Some((kind, body)) = v.as_object().and_then(|o| if o.len() == 1 { o.iter().next() } else { None }) else { return v.clone() };
    match kind.as_str() {
        "term" | "prefix" | "wildcard" | "regexp" | "fuzzy" => {
            let mut out = Map::new();
            if let Some(o) = body.as_object() {
                for (f, x) in o {
                    let mut e = match x {
                        Value::Object(m) => m.clone(),
                        other => {
                            let mut m = Map::new();
                            m.insert("value".into(), other.clone());
                            m
                        }
                    };
                    e.entry("boost").or_insert(json!(1.0));
                    out.insert(f.clone(), Value::Object(e));
                }
            }
            json!({ kind.clone(): out })
        }
        "match_all" | "match_none" | "exists" | "ids" | "terms" | "range" => {
            let mut m = body.as_object().cloned().unwrap_or_default();
            if kind == "range" {
                for (_, r) in m.iter_mut() {
                    if let Some(ro) = r.as_object_mut() {
                        ro.entry("boost").or_insert(json!(1.0));
                    }
                }
            } else {
                m.entry("boost").or_insert(json!(1.0));
            }
            json!({ kind.clone(): m })
        }
        "bool" => {
            let mut m = Map::new();
            if let Some(o) = body.as_object() {
                for (k, x) in o {
                    match (k.as_str(), x) {
                        ("must" | "filter" | "should" | "must_not", Value::Array(a)) => {
                            m.insert(k.clone(), Value::Array(a.iter().map(normalize_query).collect()));
                        }
                        ("must" | "filter" | "should" | "must_not", q) => {
                            m.insert(k.clone(), json!([normalize_query(q)]));
                        }
                        _ => {
                            m.insert(k.clone(), x.clone());
                        }
                    }
                }
            }
            m.entry("adjust_pure_negative").or_insert(json!(true));
            m.entry("boost").or_insert(json!(1.0));
            json!({"bool": m})
        }
        _ => v.clone(),
    }
}

fn search_shards(node: &Node, req: &Req, index: Option<&str>) -> R {
    let mut o = opts(req);
    // ES answers an unknown index with an empty shard list rather than an error.
    o.allow_no_indices = true;
    o.ignore_unavailable = true;
    let expr = index.unwrap_or("_all");
    let targets = node.resolve(expr, o)?;
    // Which aliases of each index were named by the expression.
    let parts: Vec<&str> = expr.split(',').map(str::trim).collect();
    let mut indices = Map::new();
    let mut shards = Vec::new();
    let mut seen = BTreeSet::new();
    for t in &targets {
        if !seen.insert(t.svc.name.clone()) {
            continue;
        }
        let aliases = t.svc.aliases();
        let mut named: Vec<String> = aliases.keys().filter(|a| parts.iter().any(|p| !p.starts_with('-') && elasticsearch_engine::mapping::wildcard_match(p, a))).cloned().collect();
        named.sort();
        let direct = parts.iter().any(|p| *p == "_all" || elasticsearch_engine::mapping::wildcard_match(p, &t.svc.name));
        let mut e = Map::new();
        if !named.is_empty() {
            e.insert("aliases".into(), json!(named));
            if !direct {
                let filters: Vec<Value> = named.iter().filter_map(|a| aliases.get(a).and_then(|d| d.get("filter")).cloned()).collect();
                let unfiltered = named.iter().any(|a| aliases.get(a).and_then(|d| d.get("filter")).is_none());
                if !unfiltered && !filters.is_empty() {
                    let filters: Vec<Value> = filters.iter().map(normalize_query).collect();
                    let f = if filters.len() == 1 { filters[0].clone() } else { json!({"bool": {"should": filters, "adjust_pure_negative": true, "boost": 1.0}}) };
                    e.insert("filter".into(), f);
                }
            }
        }
        indices.insert(t.svc.name.clone(), Value::Object(e));
        // The shards the request's routing (intersected with the aliases'
        // search routing) selects, as the search would visit them.
        let routed = crate::search_api::apply_search_routing(vec![t.clone()], req.param("routing"));
        let wanted: Vec<u32> = routed.first().map(|rt| t.svc.search_shards_for(rt.routing.as_deref())).unwrap_or_default();
        for sh in wanted {
            let alloc = t.svc.shard(sh).map(|c| c.allocation_id.clone()).unwrap_or_else(|| t.svc.uuid.clone());
            shards.push(json!([{"state": "STARTED", "primary": true, "node": NODE_ID, "relocating_node": null, "shard": sh, "index": t.svc.name, "allocation_id": {"id": alloc}, "relocation_failure_info": {"failed_attempts": 0}}]));
        }
    }
    let mut nodes = Map::new();
    if !shards.is_empty() {
        nodes.insert(NODE_ID.into(), node_ident(node));
    }
    Ok(Resp::ok(json!({"nodes": nodes, "indices": indices, "shards": shards})))
}

#[allow(dead_code)]
fn _unused(_: HashMap<String, String>) {}


#[cfg(test)]
mod client_pruning_tests {

    /// A thread's name may hold spaces and parentheses; utime and stime are
    /// fields 14 and 15 of its stat line, counted past the name.
    #[test]
    fn task_stat_lines_parse_past_the_name() {
        let line = "4242 (tokio-runtime-w) S 1 4241 4241 0 -1 4194368 120 0 3 0 250 50 0 0 20 0 30 0 100 0 0";
        assert_eq!(super::parse_task_stat(line), Some(("tokio-runtime-w".to_string(), 300)));
        let odd = "7 (a (b) c) R 1 1 1 0 -1 0 0 0 0 0 7 5 0 0 20 0 1 0 1 0 0";
        assert_eq!(super::parse_task_stat(odd), Some(("a (b) c".to_string(), 12)));
        assert_eq!(super::parse_task_stat("7 (short) R 1"), None);
    }
    use super::*;
    #[test]
    fn keep_alive_requests_share_a_channel_until_last_owner_closes() {
        let connection = HttpConnection::open("127.0.0.1:9200".into(), "127.0.0.1:54321".into());
        let id = connection.0.id;
        let copy = connection.clone();
        for _ in 0..100 {
            let guard = http_request(&Method::GET, "/", "/", &Default::default(), 12, Some(&connection));
            drop(guard);
        }
        {
            let shard = lock(&HTTP_CLIENTS[id as usize % CLIENT_SHARDS]);
            let client = shard.clients.get(&id).unwrap();
            assert_eq!(client.request_count, 100);
            assert_eq!(client.request_size, 1200);
            assert_eq!(client.remote_address, "127.0.0.1:54321");
            assert!(client.closed.is_none());
        }
        drop(connection);
        assert!(lock(&HTTP_CLIENTS[id as usize % CLIENT_SHARDS]).clients.contains_key(&id));
        drop(copy);
        assert!(!lock(&HTTP_CLIENTS[id as usize % CLIENT_SHARDS]).clients.contains_key(&id));
        assert!(lock(&CLOSED_CLIENTS).clients.iter().any(|c| c.id == id && c.closed.is_some()));
    }

    #[test]
    fn completed_history_is_bounded_and_keeps_latest_completions() {
        let mut closed = ClosedClients::default();
        for id in 0..100_000 {
            closed.push(HttpClient { id, closed: Some(100), ..Default::default() });
        }
        let rows = closed.snapshot(100);
        assert_eq!(rows.len(), MAX_CLOSED_CLIENTS);
        assert_eq!(rows.first().unwrap().id, 90_000);
        assert_eq!(rows.last().unwrap().id, 99_999);
    }
    #[test]
    fn expiry_handles_out_of_order_timestamps_and_clock_rollback() {
        let mut closed = ClosedClients::default();
        let now = CLOSED_CLIENT_MAX_AGE_MS + 100;
        for (id, time) in [(1, now), (2, 0), (3, 100), (4, now + 100)] {
            closed.push(HttpClient { id, closed: Some(time), ..Default::default() });
        }
        assert_eq!(closed.snapshot(now).iter().map(|c| c.id).collect::<Vec<_>>(), vec![1, 3, 4]);
        assert_eq!(closed.snapshot(now + 1).iter().map(|c| c.id).collect::<Vec<_>>(), vec![1, 4]);
        assert_eq!(closed.snapshot(0).len(), 2);
    }
    #[test]
    fn closing_removes_open_record_without_evicting_other_open_requests() {
        let mut open = ClientShard::default();
        for id in 0..100 { open.clients.insert(id, HttpClient { id, ..Default::default() }); }
        let mut client = open.clients.remove(&7).unwrap();
        client.closed = Some(0);
        let mut closed = ClosedClients::default();
        closed.push(client);
        assert_eq!(open.clients.len(), 99);
        assert_eq!(closed.snapshot(0)[0].id, 7);
        assert!(open.clients.values().all(|c| c.closed.is_none()));
    }
}

#[cfg(test)]
mod node_stats_cost_tests {
    use super::*;
    use elasticsearch_engine::index::WriteReq;
    use elasticsearch_engine::shard::WriteCond;

    /// The per-pool walk over every index the thread pool section did before.
    fn activity_walk(node: &Node, name: &str) -> (u64, u64) {
        use elasticsearch_engine::shard::Counters as C;
        let (mut active, mut completed) = (0u64, 0u64);
        for svc in node.all_indices() {
            let c = &svc.counters();
            match name {
                "search" => { active += C::get(&c.query_current); completed += C::get(&c.search_total); }
                "write" => { active += C::get(&c.index_current) + C::get(&c.delete_current); completed += C::get(&c.index_total) + C::get(&c.delete_total); }
                "get" => { active += C::get(&c.get_current); completed += C::get(&c.get_total); }
                "refresh" => completed += C::get(&c.refresh_total),
                "flush" => completed += C::get(&c.flush_total),
                "force_merge" => completed += C::get(&c.merge_total),
                _ => {}
            }
        }
        if name == "management" {
            completed = HTTP_TOTAL.load(Ordering::Relaxed);
        }
        (active, completed)
    }

    /// Drop the process-wide figures other tests running in parallel move
    /// (engine lock instrumentation, the merge auto-throttle rate).
    pub(super) fn stable(v: &mut Value) {
        match v {
            Value::Object(m) => {
                m.retain(|k, _| !k.starts_with("rustic_") && !k.starts_with("total_auto_throttle"));
                m.values_mut().for_each(stable);
            }
            Value::Array(a) => a.iter_mut().for_each(stable),
            _ => {}
        }
    }

    fn node_with_indices() -> (tempfile::TempDir, Arc<Node>) {
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        for (name, n) in [("pc-a", 3), ("pc-b", 5)] {
            let svc = node.create_index(name, &json!({"settings": {"refresh_interval": "-1"}})).unwrap();
            let docs: Vec<WriteReq> = (0..n).map(|i| WriteReq::Index { id: Some(i.to_string()), source: json!({"n": i}).to_string(), routing: None, cond: WriteCond::default() }).collect();
            assert!(svc.write(docs, true).iter().all(|r| r.is_ok()));
            svc.refresh().unwrap();
            svc.flush().unwrap();
        }
        (dir, node)
    }

    #[test]
    fn one_pass_pool_counters_match_the_per_pool_walk() {
        let (_dir, node) = node_with_indices();
        let counters = pool_counters(&node);
        for p in pools() {
            assert_eq!(counters.activity(p.name), activity_walk(&node, p.name), "{}", p.name);
        }
        for name in ["search", "write", "get", "refresh", "flush", "force_merge", "management", "unknown"] {
            assert_eq!(counters.activity(name), activity_walk(&node, name), "{name}");
        }
        assert!(counters.write.1 >= 8, "{counters:?}");
    }

    #[test]
    fn pools_report_the_configured_heap_figure() {
        // `pools()` sizes the snapshot pool from the configured heap, which
        // is what `sysinfo::heap(..).1` reported for any sample.
        assert_eq!(sysinfo::heap(&sysinfo::sys()).1, elasticsearch_engine::memory::es_heap());
        let snapshot = pools().into_iter().find(|p| p.name == "snapshot").unwrap();
        let n = crate::sysinfo::processors();
        let expected = if elasticsearch_engine::memory::es_heap() >= 750 << 20 { 10 } else { n.min(5) };
        assert_eq!(snapshot.max, expected.max(1));
    }

    /// In-process node stats render the handler's body (indices section and
    /// header), with `?human` applied as the route applies it.
    #[test]
    fn in_process_node_stats_match_the_handler() {
        let (_dir, node) = node_with_indices();
        for (path, params) in [
            ("/_nodes/_local/stats/indices", vec![]),
            ("/_nodes/_local/stats/indices/docs,store,translog", vec![]),
            ("/_nodes/_local/stats/indices", vec![("level", "indices")]),
            ("/_nodes/_local/stats/indices", vec![("level", "shards")]),
            ("/_nodes/_local/stats/indices", vec![("human", "true")]),
        ] {
            let req = Req { method: Method::GET, path: path.into(), segs: path.split('/').filter(|s| !s.is_empty()).map(str::to_string).collect(),
                params: params.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), body: Default::default(), headers: Default::default(),
                access: None, extensions: Default::default(), api_version: crate::RestApiVersion::V9 };
            let mut expected = match nodes_stats(&node, &req).unwrap().body { crate::RespBody::Json(v) => v, _ => panic!("expected JSON") };
            if params.iter().any(|(k, _)| *k == "human") {
                crate::humanize(&mut expected);
            }
            let mut actual = nodes_stats_in_process(&node, path, params.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()).unwrap();
            stable(&mut actual);
            stable(&mut expected);
            assert_eq!(actual["_nodes"], expected["_nodes"], "{path}");
            assert_eq!(actual["nodes"][NODE_ID]["indices"], expected["nodes"][NODE_ID]["indices"], "{path} {params:?}");
            assert!(actual["nodes"][NODE_ID]["indices"]["docs"]["count"].as_u64().unwrap_or(8) >= 8, "{path}");
        }
    }
}
