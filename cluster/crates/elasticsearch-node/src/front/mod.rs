//! The cluster-aware HTTP front. See [`classify`] for the routing table.
//!
//! Requests that must run elsewhere are forwarded over the transport
//! (`internal:rest/forward`) and executed there by the same code path, so
//! a request is always finally executed by some node's engine REST layer
//! (exact ES behaviour), plus the cluster semantics added here:
//! replication of writes, metadata publication, routing, id wrapping.

pub mod bulk;
mod admitted;
mod decode;
mod response;
pub use response::RestResp;
mod failure_store;
mod ingest;
mod native_security;
pub mod classify;
pub mod fanout;
mod agg_reduce;
pub(crate) mod bg_task;
mod contexts;
mod disk_usage;
mod write_path;
mod connector_store;
mod fleet_checkpoints;
mod fleet_search;
mod ccr_follow;
mod snapshot_cache_stats;
mod query_rewrite;
mod app_search;
mod esql_dist;
mod xpack_query;
mod stateless_read;
mod copies;
pub(crate) mod adaptive;
mod qtf;
mod search_task;
mod merge_memo;
pub use qtf::{counts as qtf_counts, reruns as qtf_reruns, single_rounds as qtf_single_rounds, SINGLE_ROUND_DISABLED as QTF_SINGLE_ROUND_DISABLED, LIGHT_DISABLED as QTF_LIGHT_DISABLED};

use crate::node::{ClusterNode, TransportNet};
use crate::shards::{PrimaryCtx, ShardsInfo};
use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, Method, Request, Response, StatusCode, Uri};
use classify::{classify, BroadcastOp, ContextKind, Route};
use elasticsearch_cluster::state::ShardState;
use elasticsearch_engine::node::ResolveOpts;
use elasticsearch_engine::error::Result as EsResult;
use elasticsearch_transport::RemoteError;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

pub const ACTION_FORWARD: &str = "internal:rest/forward";
pub const ACTION_AUTO_CREATE: &str = "internal:cluster/auto_create";
const DATA_STREAM_BACKING_WRITE_HEADER: &str = "x-rustic-internal-datastream-backing-write";
/// A `retriever` / `rank` search's phase (`plan` | `apply`), see
/// [`rank_search`]; never accepted from clients.
const RANK_PLAN_HEADER: &str = "x-rustic-internal-rank-plan";
/// Set on an ES|QL response that could not read every shard it needed from
/// started local copies: the `index#shard` list it needed.
const ESQL_SHARDS_HEADER: &str = "x-rustic-internal-esql-shards";
/// Cluster-internal work running as ES's `_system` user (see
/// [`system_headers`]). Stripped from client HTTP requests at the edge, so
/// only node-to-node requests built by this code can carry it.
pub(crate) const SYSTEM_HEADER: &str = "x-rustic-internal-system";
/// Internal request header: a sub-request some executor thread waits for
/// (see `forward_addr`); its engine work runs unpooled on the receiving node.
pub(crate) const NESTED_HEADER: &str = "x-rustic-internal-nested";
/// Internal request header on every forward this version sends: its
/// sender reads a 429 circuit breaker trip as a shard failure and tries
/// the next copy. A forward without it comes from an older coordinator,
/// which moves to another copy on 502/503 only (see
/// [`for_older_sender`]).
const TRIPS_HEADER: &str = "x-rustic-internal-trips";

/// Headers that run an internal request as `_system`: background work
/// with no caller (a restore resumed after a master failover, ...).
pub(crate) fn system_headers() -> Vec<(String, String)> {
    vec![(SYSTEM_HEADER.into(), "true".into())]
}

fn is_system(headers: &[(String, String)]) -> bool {
    headers.iter().any(|(k, v)| k.eq_ignore_ascii_case(SYSTEM_HEADER) && v == "true")
}

/// Marks the inner dispatch of a synonyms update, whose reload details the
/// outer call merges across nodes.
const SYNONYMS_INNER_HEADER: &str = "x-rustic-internal-synonyms-inner";
/// Master action: create the backing index of a time series data stream
/// covering a past `@timestamp` (`TransportPastTimeSeriesIndexCreationAction`).
pub const ACTION_PAST_TSDB_INDEX: &str = "internal:cluster/past_tsdb_index";
const MAX_HOPS: u8 = 3;
/// `http.max_content_length` default.
const MAX_CONTENT_LENGTH: usize = 100 * 1024 * 1024;

/// A REST request in transit between nodes.
///
/// Wire compatibility (rolling upgrades): fields are only ever appended,
/// and every appended field is optional in both directions:
/// - an older node decoding a newer node's request ignores the trailing
///   bytes (bincode's default options allow trailing bytes);
/// - a newer node decoding an older node's request sees the input end
///   before the appended field. bincode ignores `#[serde(default)]`, so
///   [`Deserialize`] is hand-written (see `rest_req_de`): a field missing
///   at the end of a sequence decodes as its default.
mod request;
mod request_shared;
pub use request::RestReq;
pub use request_shared::{AppSearchPlan, SharedDfsStats, SharedHeaders, SharedText};
pub(crate) use request_shared::{UriComponent,UriJoin,FilteredQuery};

/// `Vec<u8>` fields as one byte string. bincode writes the same length and
/// bytes as for a sequence of `u8` (so nodes on either side of this change
/// interoperate), but reads and writes them in one copy instead of visiting
/// every byte. Self-describing formats still read arrays of numbers.
pub(crate) mod wire_bytes {
    use serde::de::{Deserializer, SeqAccess, Visitor};
    use serde::Serializer;
    use std::fmt;

    pub fn serialize<S: Serializer>(v: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(v)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        d.deserialize_byte_buf(BytesVisitor)
    }

    /// A [`wire_bytes`](self) value for positional (`SeqAccess`) decoding.
    pub struct Bytes(pub Vec<u8>);

    impl<'de> serde::Deserialize<'de> for Bytes {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Bytes, D::Error> {
            deserialize(d).map(Bytes)
        }
    }

    struct BytesVisitor;

    impl<'de> Visitor<'de> for BytesVisitor {
        type Value = Vec<u8>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("bytes")
        }
        fn visit_bytes<E>(self, v: &[u8]) -> Result<Vec<u8>, E> {
            Ok(v.to_vec())
        }
        fn visit_byte_buf<E>(self, v: Vec<u8>) -> Result<Vec<u8>, E> {
            Ok(v)
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<u8>, A::Error> {
            let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(1 << 20));
            while let Some(b) = seq.next_element()? {
                out.push(b);
            }
            Ok(out)
        }
    }
}

impl<'de> serde::Deserialize<'de> for RestReq {
    fn deserialize<D:serde::Deserializer<'de>>(d:D)->Result<Self,D::Error>{
        admitted::Request::deserialize(d).map(|request|request.value)
    }
}

/// Cross-shard DFS (`dfs_query_then_fetch`) role of a shard search, applied
/// in [`local_engine_once`] as an engine request extension.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum Dfs {
    /// An ordinary request.
    #[default]
    None,
    /// The DFS stats round: return the shard's term/field statistics
    /// ([`elasticsearch_http::DfsStats`] JSON) instead of hits.
    Collect,
    /// The query round: score with these merged (global) statistics.
    Use(SharedDfsStats),
}

/// Preserve ES error details through the admitted renderer and owned bytes.
fn err(error: elasticsearch_engine::error::EsError) -> RestResp {
    RestResp::from_es_error(&error)
}

impl RestResp {
    /// ES error body (`{"error":{root_cause,type,reason,...},"status"}`).
    pub fn error(status: u16, kind: &str, reason: &str) -> RestResp {
        let e = json!({"type": kind, "reason": reason});
        RestResp::json(status, &json!({"error": {"root_cause": [e.clone()], "type": kind, "reason": reason}, "status": status}))
    }

    pub fn from_remote(e: &RemoteError) -> RestResp {
        #[derive(serde::Serialize)]
        struct Body<T:serde::Serialize>{error:T,status:u16}
        elasticsearch_engine::memory::native_control(|| Self::json_serialized(e.status,&Body{error:e.json_with_root(),status:e.status}))
    }

    pub fn body_json(&self) -> Option<Value> {
        serde_json::from_slice(&self.body).ok()
    }

    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }
}

impl RestReq {
    pub fn path(&self) -> &str {
        self.uri.split('?').next().unwrap_or("/")
    }

    pub fn query(&self) -> &str {
        self.uri.split_once('?').map(|(_, q)| q).unwrap_or("")
    }

    pub fn params(&self) -> HashMap<String, String> {
        self.query()
            .split('&')
            .filter(|p| !p.is_empty())
            .map(|p| match p.split_once('=') {
                Some((k, v)) => (decode(k), decode(v)),
                None => (decode(p), String::new()),
            })
            .collect()
    }

    pub fn segments(&self) -> Vec<String> {
        self.path().split('/').filter(|s| !s.is_empty()).map(decode_path).collect()
    }

    pub fn method(&self) -> Method {
        Method::from_bytes(self.method.as_bytes()).unwrap_or(Method::GET)
    }

    pub fn json(&self) -> Option<Value> {
        self.json_ref().cloned()
    }

    /// The parsed body, parsed at most once per request body.
    pub fn json_ref(&self) -> Option<&Value> {
        self.parsed.get_or_init(|| if self.body.is_empty() { None } else {
            // Plain JSON parses to the same value either way; recording it
            // lets the engine take this value (`PreparsedBody`).
            elasticsearch_engine::jsonpos::parse_plain(&self.body).filter(|v| !v.is_null()).map(|v| (Arc::new(v), true)).or_else(|| serde_json::from_slice(&self.body).ok().or_else(|| {
                if self.path().ends_with("/_search") {
                    std::str::from_utf8(&self.body).ok().and_then(|s|elasticsearch_engine::source_json::search_request(s).ok()).map(|p|p.0)
                } else {None}
            }).map(|v| (Arc::new(v), false)))
        }).as_ref().map(|(v, _)| &**v)
    }

    /// The body as [`json_ref`](Self::json_ref) already parsed it, when that
    /// was a plain-JSON parse (never parses).
    fn plain_parsed(&self) -> Option<&Arc<Value>> {
        self.parsed.get().and_then(Option::as_ref).filter(|(_, plain)| *plain).map(|(v, _)| v)
    }

    /// Internal JSON bodies must use the client's negotiated API version.
    /// A bodyless request may only carry Accept; rebuilding it must add a
    /// matching Content-Type instead of implicitly switching to the current API.
    pub(crate) fn json_content_type(&self) -> String {
        // Preserve an existing body's version (including a mismatch the
        // engine must reject); only a bodyless request negotiates from Accept.
        let header = if self.body.is_empty() { "accept" } else { "content-type" };
        let version = self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(header))
            .and_then(|(_, value)| value.to_ascii_lowercase().split(';')
                .find_map(|part| part.trim().strip_prefix("compatible-with=").map(|v| v.trim_matches('"').to_owned())));
        match version {
            Some(version) => format!("application/vnd.elasticsearch+json; compatible-with={version}"),
            None => "application/json".into(),
        }
    }

    /// This request as a JSON `POST` of `body`, keeping the query string
    /// parameters `keep` accepts.
    fn try_rebuilt(&self, body: &Value, keep: impl Fn(&str) -> bool) -> elasticsearch_engine::error::Result<RestReq> {
        let mut query_memory=elasticsearch_engine::memory::breakers().reserve_request(0,"rebuilt request query slots")?;
        let mut q=admitted::slots::<&str>(&mut query_memory,self.query().split('&').count())?;
        q.extend(self.query().split('&').filter(|p| !p.is_empty() && keep(&decode(p.split('=').next().unwrap_or("")))));
        let mut r = self.try_without_body()?;
        r.set_json(body)?;
        r.set_uri_fmt(format_args!("{}{}{}", self.path(), if q.is_empty() { "" } else { "?" }, UriJoin(q.iter(),"&")))?;
        r.set_method("POST")?;
        r.retain_headers(|(h, _)| !h.eq_ignore_ascii_case("content-type") && !h.eq_ignore_ascii_case("content-length"))?;
        r.push_header("content-type", &self.json_content_type())?;
        Ok(r)
    }
}

/// Percent-encode a path segment or query value.
pub(crate) fn encode_component(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn decode_hex(b: &[u8], plus_is_space: bool) -> String {
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() + 0 && i + 2 <= b.len() - 1 {
            if let Ok(v) = u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or("zz"), 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(if plus_is_space && b[i] == b'+' { b' ' } else { b[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn decode(s: &str) -> String {
    decode_hex(s.as_bytes(), true)
}

fn decode_path(s: &str) -> String {
    decode_hex(s.as_bytes(), false)
}

// ------------------------------------------------------------------ engine

/// Is an index of the applied cluster state missing from the local engine?
/// Peer recovery drops and recreates the target's index shell
/// (`wipe_local_copy`), so for a moment the engine does not know an index
/// that exists in the cluster.
fn mirror_incomplete(node: &ClusterNode) -> bool {
    let st = node.state();
    let have: BTreeSet<String> = node.engine.all_indices().iter().map(|s| s.name.clone()).collect();
    if st.metadata.indices.keys().any(|k| !have.contains(k)) {
        return true;
    }
    let expected_streams: BTreeSet<String> = st.metadata.customs.get("data_stream")
        .and_then(Value::as_object).map(|streams| streams.keys().cloned().collect()).unwrap_or_default();
    let local_streams: BTreeSet<String> = node.engine.data_streams.read().keys().cloned().collect();
    expected_streams != local_streams
}

/// Wait (bounded) for the local metadata mirror to hold every index of the
/// applied cluster state. Returns whether anything was waited for.
pub(crate) async fn await_mirror(node: &ClusterNode) -> bool {
    if !mirror_incomplete(node) {
        return false;
    }
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    while mirror_incomplete(node) && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    true
}

/// Run a request against this node's engine REST layer. A 404 produced
/// while the mirror was momentarily missing an index is not a real answer:
/// the request is re-run once the mirror is whole again.
pub async fn local_engine(node: &ClusterNode, req: &RestReq) -> RestResp {
    let r = local_engine_once(node, req).await;
    if r.status == 404 && await_mirror(node).await {
        return local_engine_once(node, req).await;
    }
    r
}

async fn local_engine_once(node: &ClusterNode, req: &RestReq) -> RestResp {
    local_engine_request(node, req, false, false, None, None).await.0
}

pub(crate) async fn local_node_monitor(node: &ClusterNode, req: &RestReq) -> RestResp {
    local_engine_request(node, req, true, false, None, None).await.0
}

/// The search task a request's synchronous cluster calls run under: the
/// cluster parent task id of a search shard request (a retriever's plan
/// phase) and its cancellation token. A search such a call coordinates is
/// that task's child (ES: a retriever's child searches are children of the
/// client's search task): cancelled with it, also when its client goes away.
type LinkedParent = Option<(String, Option<Arc<elasticsearch_engine::rustic_core::cancel::Token>>)>;

/// Run `sub` through this node's cluster front for a synchronous caller
/// (engine work waiting for the answer on its thread), as the child of
/// `parent`'s search task. A `search_coordination` worker parks for the
/// wait, so a spare runs the pool's next job meanwhile.
fn dispatch_sync(coordinator: &Arc<ClusterNode>, runtime: &tokio::runtime::Handle, sub: RestReq, parent: &LinkedParent) -> RestResp {
    let call = || {
        // Callable from a blocking-pool thread, an engine worker thread
        // or (synchronous engine code) a runtime worker.
        tokio::task::block_in_place(|| runtime.block_on(search_task::linked(parent.clone(), dispatch(coordinator, sub))))
    };
    if elasticsearch_http::pools::on_coordination() { elasticsearch_http::pools::blocking(call) } else { call() }
}

/// A [`elasticsearch_http::ClusterRest`] call through this node's cluster
/// front, sending `credentials` with every sub-request. A call made where
/// waiting could deadlock a bounded executor
/// ([`elasticsearch_http::pools::sub_requests_nested`]) is marked nested,
/// so its fan-out runs off the executors on every node it reaches.
fn cluster_rest(coordinator: Arc<ClusterNode>, runtime: tokio::runtime::Handle, credentials: SharedHeaders, parent: LinkedParent)
    -> Arc<dyn Fn(&str, &str, Option<Value>) -> Result<Value, elasticsearch_engine::error::EsError> + Send + Sync> {
    Arc::new(move |method, uri, body| {
            let nested = elasticsearch_http::pools::sub_requests_nested();
            let admitted = admitted::Request::cluster_call(method, uri, &credentials, body.as_ref(), nested, coordinator.engine.breakers)?;
            // Declare the owner first: reverse local destruction drops payload
            // before admission on errors/unwind, and retains it through dispatch.
            let memory = admitted.memory;
            let sub = admitted.value;
            let response = dispatch_sync(&coordinator, &runtime, sub, &parent);
            drop(memory);
            cluster_rest_answer(response)
        })
}

/// A [`cluster_rest`] call's answer: the body, or its error.
fn cluster_rest_answer(response: RestResp) -> Result<Value, elasticsearch_engine::error::EsError> {
    let value = response.body_json().unwrap_or(Value::Null);
    // A response that is not an error body is the API's answer even
    // with a failure status (`_reindex` reports its worst failure's
    // status alongside `failures`).
    if response.is_success() || (value.is_object() && value.get("error").is_none()) {
        return Ok(value);
    }
    Err(elasticsearch_engine::error::EsError::from_error_json(response.status, &value["error"]))
}

/// [`cluster_rest`] for several calls at once ([`elasticsearch_http::ClusterRestMany`]):
/// dispatched together on this node's runtime while the caller waits once,
/// each as `parent`'s child as a single call is.
fn cluster_rest_many(coordinator: Arc<ClusterNode>, runtime: tokio::runtime::Handle, credentials: SharedHeaders, parent: LinkedParent)
    -> Arc<dyn Fn(Vec<(String, String, Option<Value>)>) -> Vec<Result<Value, elasticsearch_engine::error::EsError>> + Send + Sync> {
    Arc::new(move |calls| {
        let nested = elasticsearch_http::pools::sub_requests_nested();
        let mut out: Vec<Option<Result<Value, elasticsearch_engine::error::EsError>>> = Vec::with_capacity(calls.len());
        let mut admitted_calls = Vec::with_capacity(calls.len());
        for (i, (method, uri, body)) in calls.iter().enumerate() {
            match admitted::Request::cluster_call(method, uri, &credentials, body.as_ref(), nested, coordinator.engine.breakers) {
                Ok(admitted) => { admitted_calls.push((i, admitted.memory, admitted.value)); out.push(None); }
                Err(e) => out.push(Some(Err(e))),
            }
        }
        let (memory, subs): (Vec<_>, Vec<_>) = admitted_calls.into_iter().map(|(i, m, sub)| ((i, m), sub)).unzip();
        let wait = || {
            tokio::task::block_in_place(|| runtime.block_on(futures::future::join_all(subs.into_iter().map(|sub| search_task::linked(parent.clone(), dispatch(&coordinator, sub))))))
        };
        let responses = if elasticsearch_http::pools::on_coordination() { elasticsearch_http::pools::blocking(wait) } else { wait() };
        for ((i, m), response) in memory.into_iter().zip(responses) {
            drop(m);
            out[i] = Some(cluster_rest_answer(response));
        }
        out.into_iter().map(|r| r.unwrap_or_else(|| Err(elasticsearch_engine::error::EsError::internal("cluster call lost")))).collect()
    })
}

fn cluster_rest_raw(coordinator: Arc<ClusterNode>, runtime: tokio::runtime::Handle, credentials: SharedHeaders, parent: LinkedParent)
    -> Arc<dyn Fn(&str, &str, Option<Value>) -> Result<(u16, String), elasticsearch_engine::error::EsError> + Send + Sync> {
    Arc::new(move |method, uri, body| {
            let nested = elasticsearch_http::pools::sub_requests_nested();
            let admitted = admitted::Request::cluster_call(method, uri, &credentials, body.as_ref(), nested, coordinator.engine.breakers)?;
            // Declare the owner first: reverse local destruction drops payload
            // before admission on errors/unwind, and retains it through dispatch.
            let memory = admitted.memory;
            let sub = admitted.value;
            let response = dispatch_sync(&coordinator, &runtime, sub, &parent);
            drop(memory);
            raw_cluster_response(response)
        })
}

fn raw_cluster_response(response: RestResp) -> Result<(u16, String), elasticsearch_engine::error::EsError> {
    use elasticsearch_engine::{error::EsError, source_json::raw_members};
    let text = std::str::from_utf8(&response.body).map_err(|_| EsError::internal("unreadable coordinated response"))?;
    raw_members(text).ok_or_else(|| EsError::internal("unreadable coordinated response"))?;
    // Preserve failures too: reconstructing EsError loses raw error fields,
    // and a non-success response must never become a successful Fleet item.
    Ok((response.status, text.to_owned()))
}

/// The first phase of a coordinated `retriever` / `rank` search (`rank_search`).
fn is_rank_plan_phase(req: &RestReq) -> bool {
    req.headers.iter().any(|(name, value)| name.eq_ignore_ascii_case(RANK_PLAN_HEADER) && value == "plan")
}

/// [`local_engine`], with a plain JSON response body handed over parsed
/// (`Some`, and an empty `body`) instead of serialized: for callers that
/// would parse it right away.
pub async fn local_engine_json(node: &ClusterNode, req: &RestReq, append_only: Option<Arc<Vec<bool>>>) -> (RestResp, Option<Value>) {
    local_engine_json_with(node, req, append_only, None).await
}

/// [`local_engine_json`] of a bulk subgroup whose items the engine may
/// take pre-parsed (`prepared`) instead of parsing `req`'s body.
async fn local_engine_json_with(node: &ClusterNode, req: &RestReq, append_only: Option<Arc<Vec<bool>>>, prepared: Option<&elasticsearch_http::PreparedBulk>) -> (RestResp, Option<Value>) {
    let r = local_engine_request(node, req, false, true, append_only.clone(), prepared).await;
    if r.0.status == 404 && await_mirror(node).await {
        return local_engine_request(node, req, false, true, append_only, prepared).await;
    }
    r
}

async fn local_engine_request(node: &ClusterNode, req: &RestReq, internal_monitor: bool, want_json: bool, append_only: Option<Arc<Vec<bool>>>, prepared: Option<&elasticsearch_http::PreparedBulk>) -> (RestResp, Option<Value>) {
    local_engine_request_on(node, None, req, internal_monitor, want_json, append_only, prepared).await
}

/// See [`write_path::cluster_rollover_stats`].
pub(crate) async fn cluster_rollover_stats(node: &Arc<ClusterNode>, index: &str) -> Option<elasticsearch_engine::datastream::RolloverStats> {
    write_path::cluster_rollover_stats(node, index).await
}

/// Run a metadata request against the master service's scratch engine
/// (`crate::scratch`): the engine REST layer as is, on a copy.
pub(crate) async fn scratch_engine_once(node: &ClusterNode, router: axum::Router, req: &RestReq) -> RestResp {
    local_engine_request_on(node, Some(router), req, false, false, None, None).await.0
}

/// Where [`local_engine_request_on`] sends a request.
enum EngineTarget {
    /// One of this node's engines.
    Engine(Arc<elasticsearch_engine::node::Node>),
    /// Another router (the master service's scratch engine).
    Router(axum::Router),
}

async fn local_engine_request_on(node: &ClusterNode, router_override: Option<axum::Router>, req: &RestReq, internal_monitor: bool, want_json: bool, append_only: Option<Arc<Vec<bool>>>, prepared: Option<&elasticsearch_http::PreparedBulk>) -> (RestResp, Option<Value>) {
    let call_started = std::time::Instant::now();
    // The internal shard header picks the engine holding that shard.
    let shard = crate::shard_engines::shard_of(&req.headers).unwrap_or(0);
    // This node's engines are called directly (their router routes every
    // request to the same handler), and hand a rendered answer over typed.
    let target = match router_override {
        Some(router) => EngineTarget::Router(router),
        None => match node.engine_for(shard) {
            Some(engine) => EngineTarget::Engine(engine),
            None => return (RestResp::error(503, "shard_not_found_exception", &format!("no engine for shard [{shard}] on this node")), None),
        },
    };
    let b = Request::builder().method(req.method()).uri(req.uri.parse::<Uri>().unwrap_or_else(|_| Uri::from_static("/")));
    let headers=req.headers.iter().filter(|(k,_)|!(k.eq_ignore_ascii_case(crate::shard_engines::SHARD_HEADER) || k.eq_ignore_ascii_case(esql_dist::FETCH_HEADER) || k.eq_ignore_ascii_case(esql_dist::COORDINATOR_HEADER) || k.eq_ignore_ascii_case(SYSTEM_HEADER) || k.eq_ignore_ascii_case(NESTED_HEADER) || k.eq_ignore_ascii_case(copies::READ_COPIES_HEADER) || k.eq_ignore_ascii_case(fleet_search::WAIT_HEADER) || k.eq_ignore_ascii_case(TRIPS_HEADER))).map(|(k,v)|(k.as_str(),v.as_str()));
    let headers=match elasticsearch_http::outgoing_headers::collect_valid(headers) {
        Ok(headers)=>headers,Err(error)=>return (err(error),None),
    };
    // Items the engine takes pre-parsed replace the body; once taken (a
    // retry), the body is sent again.
    let prepared = prepared.filter(|p| p.is_pending());
    let body = if prepared.is_some() { axum::body::Bytes::new() } else {
        match admitted::http_body(&req.body, node.engine.breakers) {
            Ok(body) => body,
            Err(error) => return (RestResp::json(error.status, &error.to_json()), None),
        }
    };
    // The body this front end already parsed as plain JSON goes along
    // parsed: the engine takes it instead of parsing the same bytes again.
    let preparsed = if prepared.is_none() && !body.is_empty() { req.plain_parsed().map(|v| elasticsearch_http::PreparsedBody::new(body.clone(), v.clone())) } else { None };
    // The body goes to an engine of this node as held (not as a stream it
    // collects again): see `elasticsearch_http::handle_in_process_ready`.
    let (ready, stream) = match &target { EngineTarget::Engine(_) => (Some(body), Body::empty()), EngineTarget::Router(_) => (None, Body::from(body)) };
    let Ok(mut r) = b.body(stream) else {
        return (RestResp::error(400, "illegal_argument_exception", "invalid request"), None);
    };
    *r.headers_mut()=headers;
    // A search shard read (not a retriever's plan phase): none of the
    // write, fleet or failure-store hooks below apply to it, and building
    // them (a metadata copy of the request among them) for every shard
    // request of every search cost more than the hooks are worth.
    let search_read = search_task::shard_route(req) && !is_rank_plan_phase(req);
    r.extensions_mut().insert(elasticsearch_http::InternalHttpDispatch);
    if let Some(preparsed) = preparsed { r.extensions_mut().insert(preparsed); }
    if is_nested(&req.headers) {
        r.extensions_mut().insert(elasticsearch_http::NestedRequest);
    }
    if req.hops == 0 && COORDINATING_HELD.try_with(|held| *held).unwrap_or(false) {
        // Not forwarded, and part of a request whose coordinating stage this
        // node holds (a bulk): the primary stage counts its bytes once. A
        // single-document write takes both stages in the engine (limited).
        r.extensions_mut().insert(elasticsearch_http::CoordinatorLocal);
    }
    if internal_monitor { r.extensions_mut().insert(elasticsearch_http::InternalNodeMonitor); }
    // The search task this request's own cluster searches run under.
    let mut linked_parent: LinkedParent = None;
    // A copy-local msearch batch stops at its next item once its coordinator
    // task is banned (cancelled, or its client went away).
    if let Some((_, parent)) = req.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(crate::rest::tasks::PARENT_HEADER)) {
        // A search shard request of a coordinating search task: the engine
        // runs it as that task's child (see `search_task`).
        if search_task::shard_route(req) {
            let token = search_task::parent_token(node, parent);
            if is_rank_plan_phase(req) {
                linked_parent = Some((parent.clone(), token.clone()));
            }
            r.extensions_mut().insert(elasticsearch_http::SearchParent { id: parent.clone(), token });
        }
        if req.segments().last().map(String::as_str) == Some("_msearch") {
            let parent = match SharedText::try_new(parent) { Ok(v) => v, Err(e) => return (err(e), None) };
            let seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
            r.extensions_mut().insert(elasticsearch_http::ParentCancelled(Arc::new(move || {
                // Latched: the ban outlives its cache entry for this batch.
                seen.load(std::sync::atomic::Ordering::Relaxed) || (crate::rest::tasks::banned(&parent) && { seen.store(true, std::sync::atomic::Ordering::Relaxed); true })
            })));
        }
    }
    if is_system(&req.headers) { r.extensions_mut().insert(elasticsearch_http::SystemIdentity); }
    r.extensions_mut().insert(elasticsearch_http::CoordinatorUsage);
    let internal_backing_write = req.headers.iter().any(|(name, value)| name.eq_ignore_ascii_case(DATA_STREAM_BACKING_WRITE_HEADER) && value == "true");
    if internal_backing_write {
        r.extensions_mut().insert(elasticsearch_http::InternalDataStreamBackingWrite);
    }
    if req.headers.iter().any(|(name, _)| name.eq_ignore_ascii_case(contexts::SCROLL_TOUCH_HEADER)) {
        r.extensions_mut().insert(elasticsearch_http::InternalScrollTouch);
    }
    match req.headers.iter().find(|(name, _)| name.eq_ignore_ascii_case(qtf::HEADER)).map(|(_, v)| v.as_str()) {
        Some("query") => {
            r.extensions_mut().insert(elasticsearch_http::InternalQtf::Query);
            if let Some((_, id)) = req.headers.iter().find(|(name, v)| name.eq_ignore_ascii_case(qtf::CTX_HEADER) && !v.is_empty()) {
                r.extensions_mut().insert(elasticsearch_http::InternalQtfCtx(id.clone()));
            }
            if req.headers.iter().any(|(name, _)| name.eq_ignore_ascii_case(qtf::PINNED_HEADER)) {
                r.extensions_mut().insert(elasticsearch_http::InternalQtfPinned);
            }
        }
        Some("fetch") => {
            r.extensions_mut().insert(elasticsearch_http::InternalQtf::Fetch);
        }
        _ => {}
    }
    if req.headers.iter().any(|(name, _)| name.eq_ignore_ascii_case(qtf::LIGHT_HEADER)) {
        r.extensions_mut().insert(elasticsearch_http::InternalQtfLight);
    }
    match req.headers.iter().find(|(name, _)| name.eq_ignore_ascii_case(qtf::HEADER)).map(|(_, v)| v.as_str()) {
        Some("rerun") => {
            r.extensions_mut().insert(elasticsearch_http::InternalQtf::Rerun);
        }
        Some(qtf::SINGLE_ROUND) => {
            r.extensions_mut().insert(elasticsearch_http::InternalQtf::QueryFetch);
        }
        _ => {}
    }
    match req.headers.iter().find(|(name, _)| name.eq_ignore_ascii_case(RANK_PLAN_HEADER)).map(|(_, v)| v.as_str()) {
        Some("plan") => {
            r.extensions_mut().insert(elasticsearch_http::InternalRankPlan::Plan);
        }
        Some("apply") => {
            r.extensions_mut().insert(elasticsearch_http::InternalRankPlan::Apply);
        }
        _ => {}
    }
    let coordinator = node.me();
    let runtime = tokio::runtime::Handle::current();
    {
        // Engine work that creates or fills indices (enrich policy
        // execution) runs its index operations through this coordinator.
        let coordinator = node.me();
        let runtime = runtime.clone();
        // Sub-requests are authorized as the caller (the engine checks every
        // request it serves); the internal variant runs as `_system`, for
        // work ES runs under an internal origin (enrich, downsample).
        let credentials = match crate::reindex_multi::credentials(req) {Ok(v)=>v,Err(e)=>return (err(e),None)};
        if !search_read {
            r.extensions_mut().insert(elasticsearch_http::ClusterRestRaw(cluster_rest_raw(coordinator.clone(), runtime.clone(), credentials.clone(), linked_parent.clone())));
        }
        if is_rank_plan_phase(req) {
            r.extensions_mut().insert(elasticsearch_http::ClusterRestMany(cluster_rest_many(coordinator.clone(), runtime.clone(), credentials.clone(), linked_parent.clone())));
        }
        r.extensions_mut().insert(elasticsearch_http::ClusterRest(cluster_rest(coordinator.clone(), runtime.clone(), credentials, linked_parent.clone())));
        r.extensions_mut().insert(elasticsearch_http::InternalClusterRest(cluster_rest(coordinator, runtime, match SharedHeaders::try_pairs(&[(SYSTEM_HEADER,"true")]) {Ok(v)=>v,Err(e)=>return (err(e),None)}, linked_parent)));
    }
    let failure_positions = Arc::new(std::sync::Mutex::new(BTreeSet::<usize>::new()));
    let recorded_positions = failure_positions.clone();
    if !search_read {
    // The failure store writer reads the request's parameters and headers
    // only: no copy of the body.
    let template = match req.try_without_body() {Ok(value)=>value,Err(error)=>return (err(error),None)};
    r.extensions_mut().insert(elasticsearch_http::FailureStoreWriter(Arc::new(move |stream, document, position| {
        let token = failure_store::token(&template, position);
        let response = runtime.block_on(failure_store::store(&coordinator, &template, stream.to_string(), document, token));
        let body = response.body_json().unwrap_or(Value::Null);
        if response.is_success() {
            if let Some(position) = position { recorded_positions.lock().unwrap().insert(position); }
            return Ok(body);
        }
        Err(elasticsearch_engine::error::EsError::new(response.status,
            body["error"]["type"].as_str().unwrap_or("exception"),
            body["error"]["reason"].as_str().unwrap_or("failure store write failed")))
    })));
    }
    if req.segments().first().map(String::as_str) == Some("_security") {
        let coordinator = node.me();
        let runtime = tokio::runtime::Handle::current();
        r.extensions_mut().insert(elasticsearch_http::NativeSecurityMirror(Arc::new(move |id, document| {
            runtime.block_on(native_security::mirror(&coordinator, id.to_string(), document))
        })));
        if req.segments().get(1).map(String::as_str) == Some("enroll") {
            r.extensions_mut().insert(crate::enrollment::request_extension(&node.me()));
        }
    }
    let prepared_ingest = ingest::prepared(req);
    if !prepared_ingest.0.is_empty() { r.extensions_mut().insert(prepared_ingest); }
    if req.segments().get(0).map(String::as_str) == Some("_cat") && req.segments().get(1).map(String::as_str) == Some("aliases") {
        let state = node.state();
        let aliases = state.metadata.indices.iter().map(|(name, im)| {
            let hidden = name.starts_with('.') || im.settings.get("index.hidden").is_some_and(|v| v == &Value::Bool(true) || v.as_str() == Some("true"));
            (name.clone(), hidden, im.aliases.clone())
        }).collect();
        r.extensions_mut().insert(elasticsearch_http::AliasCatalog(aliases));
    }

    // `_rank_eval` runs its rated searches as cluster searches: the engine
    // only holds the shards on this node.
    if req.segments().iter().any(|s| s == "_rank_eval") {
        let coordinator = node.me();
        let template = req.clone();
        let runtime = tokio::runtime::Handle::current();
        r.extensions_mut().insert(elasticsearch_http::ClusterSearchExecutor(Arc::new(move |index, body| {
            let params = template.params();
            let query=contexts::SelectedParams(&params,&["ignore_unavailable","allow_no_indices","expand_wildcards","search_type"]);
            let prefix=if index.is_some(){"/"}else{""};let target=index.unwrap_or("");
            let mut sub = contexts::json_req_formatted(&template, "POST", format_args!("{prefix}{target}/_search{query}"), body)?;
            sub.hops = 0;
            sub.set_dfs(Dfs::None)?;
            sub.set_app_plan(None)?;
            sub.retain_headers(|(k, _)| !k.eq_ignore_ascii_case(EXEC_LOCAL) && !k.eq_ignore_ascii_case(RANK_PLAN_HEADER)
                && !k.eq_ignore_ascii_case(qtf::HEADER) && !k.eq_ignore_ascii_case(qtf::CTX_HEADER) && !k.eq_ignore_ascii_case(qtf::PINNED_HEADER) && !k.eq_ignore_ascii_case(qtf::LIGHT_HEADER) && !k.eq_ignore_ascii_case(crate::shard_engines::PREFILTER_SHARDS_HEADER))?;
            // A `search` thread waits here: the rated search's shard
            // requests (spawned per shard, run on other nodes) must not
            // queue behind it.
            if elasticsearch_http::pools::sub_requests_nested() && !is_nested(&sub.headers) {
                sub.push_header(NESTED_HEADER, "1")?;
            }
            let response = runtime.block_on(dispatch(&coordinator, sub));
            if response.is_success() {
                return Ok(String::from_utf8_lossy(&response.body).into_owned());
            }
            let value = response.body_json().unwrap_or(Value::Null);
            Err(elasticsearch_engine::error::EsError::new(response.status,
                value["error"]["type"].as_str().unwrap_or("exception"),
                value["error"]["reason"].as_str().unwrap_or("rank_eval search failed")))
        })));
    }
    if req.method() == Method::POST && req.path().trim_end_matches('/') == "/_connector/_sync_job" {
        let coordinator = node.me();
        let template = req.clone();
        let runtime = tokio::runtime::Handle::current();
        r.extensions_mut().insert(elasticsearch_http::ConnectorSourceLookup(Arc::new(move |id| {
            let mut lookup = template.clone();
            lookup.set_method("GET")?;
            lookup.set_uri_fmt(format_args!("/_connector/{}?include_deleted=true", UriComponent(id)))?;
            lookup.set_body(&[])?;
            let response = runtime.block_on(dispatch(&coordinator, lookup));
            if response.status == 404 { return Ok(None); }
            let mut value = response.body_json().unwrap_or(Value::Null);
            if !response.is_success() {
                return Err(elasticsearch_engine::error::EsError::new(response.status,
                    value["error"]["type"].as_str().unwrap_or("exception"),
                    value["error"]["reason"].as_str().unwrap_or("connector lookup failed")));
            }
            if let Some(object) = value.as_object_mut() { object.remove("id"); }
            Ok(Some(value))
        })));
    }
    if app_search::is_search(req) {
        if let Some(plan) = &req.app_plan {
            match serde_json::from_slice(&plan.query_json) {
                Ok(query) => { r.extensions_mut().insert(elasticsearch_http::PreparedSearchApplication { indices: plan.indices.clone(), query }); }
                Err(_) => return (RestResp::error(400, "parse_exception", "invalid internal application query"), None),
            }
        } else {
            let coordinator = node.me();
            let template = req.clone();
            let runtime = tokio::runtime::Handle::current();
            r.extensions_mut().insert(elasticsearch_http::SearchApplicationExecutor(Arc::new(move |indices, query, params| {
                let response = runtime.block_on(app_search::execute(&coordinator, &template, indices, query, params));
                if response.is_success() {
                    return Ok(String::from_utf8_lossy(&response.body).into_owned());
                }
                let value = response.body_json().unwrap_or(Value::Null);
                Err(elasticsearch_engine::error::EsError::new(response.status,
                    value["error"]["type"].as_str().unwrap_or("exception"),
                    value["error"]["reason"].as_str().unwrap_or("application query failed")))
            })));
        }
        let count = req.headers.iter().find(|(name, _)| name.eq_ignore_ascii_case(crate::shard_engines::PREFILTER_SHARDS_HEADER))
            .and_then(|(_, value)| value.parse().ok());
        r.extensions_mut().insert(elasticsearch_http::ReadShardSelection(shard, count));
    }
    let segments = req.segments();
    let api = if segments.first().is_some_and(|s| s.starts_with('_') && s != "_all") { segments.first() } else { segments.get(1) };
    // A rank plan phase (`rank_search`) reads no shard: it runs its child
    // retrievers as cluster searches and only needs the targets' metadata.
    // Pinning it to shard 0 failed with "[index][0] no active shard copy"
    // whenever the coordinating node does not hold that shard.
    if api.is_some_and(|s| matches!(s.as_str(), "_search" | "_count" | "_pit" | "_knn_search" | "_terms_enum" | "_validate" | "_rank_eval")) && !is_rank_plan_phase(req) {
        let count = req.headers.iter().find(|(name, _)| name.eq_ignore_ascii_case(crate::shard_engines::PREFILTER_SHARDS_HEADER))
            .and_then(|(_, value)| value.parse().ok());
        r.extensions_mut().insert(elasticsearch_http::ReadShardSelection(shard, count));
    }
    // ES|QL reads every wanted shard from a started copy: a local one, else
    // one on another node (`esql_dist`). Shards with no started copy
    // anywhere are reported so the coordinator retries (`esql_on_copies`).
    let esql_shards = Arc::new(std::sync::Mutex::new(esql_dist::Seen::default()));
    if req.segments().first().is_some_and(|s| s == "_query" || s == "_sql") {
        // SQL runs on the ES|QL executor: its remote shard reads are
        // ES|QL fetches (`POST /_query` with the fetch header).
        let mut template = req.clone();
        if req.segments().first().map(String::as_str) == Some("_sql") {
            if let Err(error)=template.set_uri("/_query"){return (err(error),None);}
        }
        // The coordinating run may move the whole query to a node holding
        // every shard, which then reads them all locally. SQL too: its
        // cursors and async results stay on the node that ran it, and the
        // follow-ups look for them on every node (`xpack_query`).
        let may_forward = req.segments().first().is_some_and(|s| s == "_query" || s == "_sql")
            && req.headers.iter().any(|(name, _)| name.eq_ignore_ascii_case(esql_dist::COORDINATOR_HEADER));
        let source = esql_dist::EsqlShards { node: node.me(), state: node.state(), engines: node.all_shard_engines(), template,
            runtime: tokio::runtime::Handle::current(), seen: esql_shards.clone(), may_forward, pit_nodes: Default::default() };
        r.extensions_mut().insert(elasticsearch_engine::esql::remote::ShardSourceExt(Arc::new(source)));
        if let Some(spec) = esql_dist::fetch_spec(req) {
            r.extensions_mut().insert(elasticsearch_engine::esql::remote::FetchRequest(spec));
        }
    }
    if want_json {
        r.extensions_mut().insert(elasticsearch_http::InProcessJson);
    }
    if matches!(target, EngineTarget::Engine(_)) {
        r.extensions_mut().insert(elasticsearch_http::InProcessTyped);
    }
    if let Some(a) = append_only {
        r.extensions_mut().insert(elasticsearch_http::AppendOnlyItems(a));
    }
    if let Some(p) = prepared {
        r.extensions_mut().insert(p.clone());
    }
    match &req.dfs {
        Dfs::None => {}
        Dfs::Collect => {
            r.extensions_mut().insert(elasticsearch_http::DfsPhaseOnly);
        }
        Dfs::Use(s) => {
            r.extensions_mut().insert(s.engine_owner());
        }
    }
    let built_us = call_started.elapsed().as_micros();
    let mut resp = match target {
        // Boxed: the engine's request future is large, and would otherwise
        // be moved with every shard call's future.
        EngineTarget::Engine(engine) => {
            let (parts, _) = r.into_parts();
            Box::pin(elasticsearch_http::handle_in_process_ready(engine, parts, ready.unwrap_or_default())).await
        }
        EngineTarget::Router(router) => match router.oneshot(r).await {
            Ok(r) => r,
            Err(e) => match e {},
        },
    };
    let routed_us = call_started.elapsed().as_micros();
    let typed = resp.extensions_mut().remove::<elasticsearch_http::TypedResponse>();
    let was_typed = typed.is_some();
    let mut result=match typed {
        Some(typed) => RestResp::typed(typed, req.method=="HEAD"),
        None => match RestResp::http_metadata(&resp,req.method=="HEAD") {Ok(result)=>result,Err(error)=>return (err(error),None)},
    };
    let extra_headers=||->Result<(),elasticsearch_engine::error::EsError>{
        let positions = failure_positions.lock().unwrap();
        if !positions.is_empty() { result.append_json_header(failure_store::POSITIONS_HEADER,&*positions)?; }
        let shards = esql_shards.lock().unwrap();
        if let Some(target) = &shards.forward {
            result.append_header(esql_dist::FORWARD_HEADER,target)?;
        } else if !shards.unread.is_empty() {
            struct Wanted<'a>(&'a std::collections::BTreeSet<(String,u32)>);
            impl serde::Serialize for Wanted<'_>{fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{
                use serde::ser::SerializeSeq;
                struct Item<'a>(&'a str,u32);
                impl std::fmt::Display for Item<'_>{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{write!(f,"{}#{}",self.0,self.1)}}
                impl serde::Serialize for Item<'_>{fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{s.collect_str(self)}}
                let mut seq=s.serialize_seq(Some(self.0.len()))?;
                for (index,shard) in self.0{seq.serialize_element(&Item(index,*shard))?;}seq.end()
            }}
            result.append_json_header(ESQL_SHARDS_HEADER,&Wanted(&shards.unread))?;
        }
        Ok(())
    };
    let mut extra_headers=extra_headers;
    if let Err(error)=extra_headers(){return (if resp.status().as_u16()==401{RestResp::fallback(401,req.method=="HEAD")}else{err(error)},None);}
    let json = resp.extensions_mut().remove::<elasticsearch_http::JsonResponse>().map(|j| j.0);
    let collected = if was_typed { result } else { result.collect(resp.into_body(),req.method=="HEAD").await };
    if stats_phases::ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
        let total_us = call_started.elapsed().as_micros();
        // An in-process engine call: building the request and its
        // extensions, the engine's router (admission, security, the
        // executor hop, the handler, rendering) and collecting the body.
        stats_phases::record("engine_call", vec![("build", built_us), ("router", routed_us - built_us), ("collect", total_us - routed_us), ("total", total_us)]);
    }
    (collected, json)
}

fn is_gzip(r: &RestResp) -> bool {
    r.headers.iter().any(|(k, v)| k.eq_ignore_ascii_case("content-encoding") && v.contains("gzip"))
}

/// Run on the local engine with cluster-uuid canonicalisation of the body.
/// A shard-level read naming its copies (`copies::READ_COPIES_HEADER`) runs
/// only on copies this node holds readable, and fails with
/// `shard_not_found_exception` if one was replaced while it ran.
async fn local_canonical(node: &ClusterNode, req: &RestReq) -> RestResp {
    let guard = match copies::ReadGuard::check(node, req) {
        Ok(g) => g,
        Err(r) => return r,
    };
    // A fleet search's copy waits for its checkpoints before it reads.
    if guard.is_some() {
        if let Some(r) = fleet_search::wait_in_read(node, req).await {
            return r;
        }
    }
    let mut resp = local_canonical_unchecked(node, req).await;
    if let Some(Err(r)) = guard.map(|g| g.held(node)) {
        free_opened_contexts(node, req, &resp);
        return r;
    }
    // A coordinator's query phase: report this node's search queue and
    // service time with a successful answer (ES `QuerySearchResult`).
    if resp.is_success() && req.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(adaptive::QUERY_HEADER)) {
        // Without room for the header the answer goes as is: it reports nothing.
        let _ = resp.append_header(adaptive::STATS_HEADER, &adaptive::encode_stats(adaptive::local_stats()));
    }
    resp
}

/// A read refused after it ran (its copy was replaced meanwhile) may have
/// opened a reader context on the engine: a query_then_fetch query phase's,
/// a scroll's or a point in time's. The answer naming it is dropped, so it
/// is freed here, or nothing would free it before its keep-alive.
fn free_opened_contexts(node: &ClusterNode, req: &RestReq, resp: &RestResp) {
    if !resp.is_success() {
        return;
    }
    let Some(engine) = node.engine_for(crate::shard_engines::shard_of(&req.headers).unwrap_or(0)) else { return };
    if let Some(id) = qtf::ctx_of(&resp.body) {
        engine.drop_fetch_context(&id);
        return;
    }
    let Some(v) = resp.body_json() else { return };
    if let Some(id) = v.get("_scroll_id").and_then(Value::as_str) {
        let dead = engine.scrolls.lock().remove(id);
        drop(dead);
    }
    if req.path().trim_end_matches('/').ends_with("/_pit") {
        if let Some(id) = v.get("id").and_then(Value::as_str) {
            let dead = engine.pits.lock().remove(id);
            drop(dead);
        }
    }
}

async fn local_canonical_unchecked(node: &ClusterNode, req: &RestReq) -> RestResp {
    // Ask the engine for an uncompressed body so it can be rewritten.
    // Internal requests never carry the header (`dispatch` strips it):
    // rebuilding every shard sub-request's metadata for nothing is skipped.
    let stripped;
    let r2 = if req.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("accept-encoding")) {
        let mut r = req.clone();
        if let Err(error)=r.retain_headers(|(k, _)| !k.eq_ignore_ascii_case("accept-encoding")){return err(error);}
        stripped = r;
        &stripped
    } else {
        req
    };
    let mut resp = local_engine(node, r2).await;
    if !is_gzip(&resp) {
        let shard = crate::shard_engines::shard_of(&req.headers).unwrap_or(0);
        // Scanned outside the uuid map's lock; a body without local uuids
        // (search hits) is not copied.
        let renames = node.with_uuids(shard, |u| u.renames()).unwrap_or_default();
        if let Some(c) = canonical_body(&renames, &resp.body) {
            resp.replace_body(c.into_bytes());
            resp.retain_headers(|(k, _)| !k.eq_ignore_ascii_case("content-length"));
        }
    }
    resp
}

/// `body` with its local index uuids renamed to the cluster's
/// (`mirror::canonicalize_with`); `None` when it holds none or is not UTF-8.
/// Uuids are ASCII, so a byte match in the UTF-8 body is a text match: a
/// body holding none of them (a page of hits, megabytes of `_source`) is
/// neither validated as UTF-8 nor scanned as text once per index, only
/// searched once per uuid with memchr's memmem.
fn canonical_body(renames: &[(String, String)], body: &[u8]) -> Option<String> {
    if !renames.iter().any(|(local, _)| memchr::memmem::find(body, local.as_bytes()).is_some()) {
        return None;
    }
    match crate::mirror::canonicalize_with(renames, std::str::from_utf8(body).ok()?) {
        std::borrow::Cow::Owned(c) => Some(c),
        std::borrow::Cow::Borrowed(_) => None,
    }
}

// ---------------------------------------------------------------- forward

pub async fn forward(node: &ClusterNode, node_id: &str, req: &RestReq) -> RestResp {
    let Some(addr) = node.address_of(node_id) else {
        return RestResp::error(502, "node_not_connected_exception", &format!("[{node_id}] node not connected"));
    };
    forward_addr(node, addr, req).await
}

pub async fn forward_addr(node: &ClusterNode, addr: SocketAddr, req: &RestReq) -> RestResp {
    if req.hops >= MAX_HOPS {
        return RestResp::error(500, "illegal_state_exception", "request was forwarded too many times");
    }
    let admitted = match admitted::Request::forwarding(req, elasticsearch_http::pools::on_executor(), elasticsearch_engine::memory::breakers()) {
        Ok(request) => request,
        Err(error) => return RestResp::error(error.status, &error.kind, &error.reason),
    };
    let fwd = &admitted.value;
    // A forwarded write carries indexing data (ES compresses bulk shard
    // requests by default): compressed as such, and so is its response.
    let sent = if is_indexing_forward(req) {
        node.transport.send_indexing_request::<_, RestResp>(addr, ACTION_FORWARD, fwd, forwarding_timeout(req)).await
    } else {
        node.transport.send_request::<_, RestResp>(addr, ACTION_FORWARD, fwd, forwarding_timeout(req)).await
    };
    match sent {
        Ok(r) => closing_peer_answer(r),
        Err(e) => RestResp::from_remote(&e.to_remote_error()),
    }
}

/// A forwarded request the target answered while it shut down: the previous
/// release reports a local master-node handler (its auto-create of the
/// write's index, say) that its stopping runtime aborted as a 500
/// `exception`, "local handler for [<action>] failed: task <n> was
/// cancelled". This build answers the same abort as a node disconnect.
/// It is the target closing under the request (ES `NodeClosedException`),
/// which the reroute phase of ES's replication and master node actions
/// retries on the next cluster state, so it becomes a retryable,
/// indeterminate `node_closed_exception` here rather than a final 500.
/// Any other answer is returned unchanged.
fn closing_peer_answer(r: RestResp) -> RestResp {
    if r.status != 500 {
        return r;
    }
    let Some(v) = r.body_json() else { return r };
    let reason = v["error"]["reason"].as_str().unwrap_or_default();
    let aborted = v["error"]["type"] == "exception" && reason.starts_with("local handler for [")
        && reason.contains("] failed: task ") && reason.ends_with(" was cancelled");
    if !aborted {
        return r;
    }
    RestResp::error(500, "node_closed_exception", &format!("node closed while handling the forwarded request: {reason}"))
}

/// A document write or bulk request (`_bulk`, `_doc`, `_create`, `_update`
/// other than a GET): its body is indexing data.
fn is_indexing_forward(req: &RestReq) -> bool {
    req.method != "GET" && req.method != "HEAD" && req.segments().iter().any(|s| matches!(s.as_str(), "_bulk" | "_doc" | "_create" | "_update"))
}

/// A forward sent from engine work holding an executor thread (a fan-out
/// waiting for this answer) is marked nested: the peer runs it off its
/// bounded executors, so two nodes' search threads can never each wait for
/// work queued behind the other's.
fn mark_nested(fwd: &mut RestReq) -> elasticsearch_engine::error::Result<()> {
    if elasticsearch_http::pools::on_executor() && !is_nested(&fwd.headers) {
        fwd.push_header(NESTED_HEADER,"1")?;
    }
    Ok(())
}

/// Every forward this version sends says it handles 429 trips itself.
fn mark_trips(fwd: &mut RestReq) -> elasticsearch_engine::error::Result<()> {
    if !fwd.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(TRIPS_HEADER)) {
        fwd.push_header(TRIPS_HEADER,"1")?;
    }
    Ok(())
}

/// The answer to a forwarded request as its sender understands it (see
/// [`older_copy_read`] and [`as_retryable`]).
fn for_older_sender(req: &RestReq, resp: RestResp) -> RestResp {
    if older_copy_read(req) { as_retryable(resp) } else { resp }
}

/// Whether `req` is a shard copy read an older coordinator (no
/// [`TRIPS_HEADER`]) would try on its other copies after a 502/503: an
/// [`EXEC_LOCAL`] read of a shard copy, not a write, and not bound to a
/// context only this copy holds (a scroll continuation, a point in time).
/// An older coordinator charges no breakers and retries only on 502/503,
/// so a 429 trip here would end its search on this copy where ES tries
/// the next one. Anything else it forwards it answers as is, so the 429
/// stays (ES's status).
fn older_copy_read(req: &RestReq) -> bool {
    let has = |name: &str| req.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(name));
    if has(TRIPS_HEADER) || !has(EXEC_LOCAL) {
        return false;
    }
    let segments = req.segments();
    if elasticsearch_http::pools::pool_for(&req.method(), &segments) == elasticsearch_http::pools::Pool::Write {
        return false;
    }
    if segments.iter().any(|s| s == "scroll" || s == "_pit") || req.body.windows(5).any(|w| w == b"\"pit\"") {
        return false;
    }
    true
}

/// A 429 `circuit_breaking_exception` (top level or root cause) as the
/// same error body with status 503, which an older coordinator moves to
/// its next copy on; when all of them fail it still fails the request with
/// the breaker's error. Other answers (`es_rejected_execution_exception`
/// 429s, which older nodes already sent, included) are unchanged.
fn as_retryable(mut resp: RestResp) -> RestResp {
    if resp.status != 429 {
        return resp;
    }
    let Some(mut v) = resp.body_json() else { return resp };
    let tripped = v["error"]["root_cause"].as_array().into_iter().flatten().chain(std::iter::once(&v["error"])).any(|e| e["type"] == "circuit_breaking_exception");
    if !tripped {
        return resp;
    }
    resp.status = 503;
    if v.get("status").is_some() {
        v["status"] = json!(503);
        resp.replace_json(&v);
        resp.retain_headers(|(k, _)| !k.eq_ignore_ascii_case("content-length"));
    }
    resp
}

/// Whether an internal request carries the nested marker (stripped from
/// client requests at the HTTP edge).
fn is_nested(headers: &[(String, String)]) -> bool {
    headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(NESTED_HEADER))
}

// Synchronous force-merge has no response deadline: large shard merges can
// legitimately take hours. Connection establishment and disconnect handling
// remain bounded by the transport. Async submission keeps the usual deadline;
// its background task forwards a wait_for_completion=true request.
fn forwarding_timeout(req: &RestReq) -> Option<Duration> {
    let segments = req.segments();
    let refs: Vec<&str> = segments.iter().map(String::as_str).collect();
    // A copy-local msearch batch can contain slow healthy items. There is no
    // remote cancellation handshake, so a transport deadline would rerun
    // still-active work. Retain item deadlines and disconnect detection.
    if refs.last() == Some(&"_msearch") && copies::read_copies(&req.headers).is_some() {
        return None;
    }
    if matches!(classify(&req.method(), &refs), Route::Broadcast { op: BroadcastOp::ForceMerge, .. })
        && req.params().get("wait_for_completion").map(String::as_str) != Some("false") {
        None
    } else {
        Some(Duration::from_secs(300))
    }
}

pub fn register_forward_handler(node: &Arc<ClusterNode>) {
    failure_store::register(node);
    native_security::register(node);
    fleet_search::register(node);
    let n = node.clone();
    node.transport.register_handler(ACTION_FORWARD, move |admitted: admitted::Request| {
        let n = n.clone();
        async move {
            let memory = admitted.memory;
            let req = admitted.value;
            // The payload is held on `inflight_requests` by the transport's
            // inbound gate, without tripping. A forwarded request may trip
            // here as it could at a client edge: only when its REST policy
            // trips (cluster settings, health, stats never do) and it is not
            // a write (indexing pressure governs those, wherever their
            // primary is).
            if forward_trips(&req) {
                use elasticsearch_engine::memory::{process_breakers, Kind};
                if let Err(e) = process_breakers().add_estimate_bytes_and_maybe_break(Kind::InflightRequests, 0, "<transport_request>") {
                    return Ok(for_older_sender(&req, RestResp::json(e.status, &e.to_json())));
                }
            }
            let older = older_copy_read(&req);
            let resp = dispatch(&n, req).await;
            drop(memory);
            Ok::<_, RemoteError>(if older { as_retryable(resp) } else { resp })
        }
    });
    let n = node.clone();
    node.transport.register_handler(ACTION_AUTO_CREATE, move |name: SharedText| {
        let n = n.clone();
        async move {
            if !n.is_master() {
                return Err(RemoteError::new("not_master_exception", "not the elected master", 503));
            }
            let (concrete, version) = n.mutate_engine_metadata(move |engine| engine.write_index(&name, true).and_then(|service|SharedText::try_new(&service.name)))
                .await.map_err(|error| RemoteError::new(&error.kind, error.reason, error.status))?;
            n.wait_applied_everywhere(version, Duration::from_secs(30)).await;
            Ok(concrete)
        }
    });
}

/// Whether a forwarded REST request may trip `inflight_requests` on the
/// node it was forwarded to (see the `ACTION_FORWARD` handler).
fn forward_trips(req: &RestReq) -> bool {
    let write = elasticsearch_http::pools::pool_for(&req.method(), &req.segments()) == elasticsearch_http::pools::Pool::Write;
    !write && elasticsearch_http::in_flight_policy(&req.method, req.path(), true) == elasticsearch_http::InFlight::Trips
}

pub fn register_past_tsdb_handler(node: &Arc<ClusterNode>) {
    let n = node.clone();
    node.transport.register_handler(ACTION_PAST_TSDB_INDEX, move |(ds, ms): (SharedText, i64)| {
        let n = n.clone();
        async move {
            if !n.is_master() {
                return Err(RemoteError::new("not_master_exception", "not the elected master", 503));
            }
            let (name, version) = n.mutate_engine_metadata(move |engine| engine.create_past_time_series_service(&ds, ms).and_then(|service|SharedText::try_new(&service.name)))
                .await.map_err(|error| RemoteError::new(&error.kind, error.reason, error.status))?;
            n.wait_applied_everywhere(version, Duration::from_secs(30)).await;
            Ok(name)
        }
    });
}

/// The backing index of time series data stream `ds` covering the
/// document's `@timestamp` (`DataStream.selectTimeSeriesWriteIndex`); with
/// `data_stream.past_tsdb_index_creation_enabled`, a past timestamp no
/// backing index covers first gets one from the master. `None`: not a time
/// series data stream (or nothing covers it: the engine reports that).
async fn covering_index(node: &Arc<ClusterNode>, ds: &str, src: &str, scan: &elasticsearch_engine::datastream::TimestampScan) -> Result<Option<SharedText>,RestResp> {
    if let Some(s) = node.engine.time_series_write_index_probed(ds, scan) {
        return SharedText::try_new(&s.name).map(Some).map_err(err);
    }
    let Some(ms) = node.engine.past_time_series_timestamp(ds, src) else{return Ok(None)};
    let name: SharedText = match node.to_master(ACTION_PAST_TSDB_INDEX, &(ds, ms), Duration::from_secs(30)).await {
        Ok(name)=>name,Err(error) if error.status==429=>return Err(RestResp::from_remote(&error)),Err(_)=>return Ok(None),
    };
    wait_for_primary(node, &name, Duration::from_secs(60)).await;
    Ok(Some(name))
}

/// The data stream checks of a write to time series data stream `ds` that
/// no backing index covers (`timestamp_error` & co.); `Ok` for anything
/// else (other targets are checked by the shard).
fn data_stream_write_check(node: &Arc<ClusterNode>, ds: &str, create: bool, src: &str, scan: &elasticsearch_engine::datastream::TimestampScan) -> Result<(), elasticsearch_engine::error::EsError> {
    let ts = node.engine.data_stream(ds).and_then(|d| node.engine.get_index(d.write_index())).is_some_and(|w| w.setting("index.mode").as_deref() == Some("time_series"));
    if !ts {
        return Ok(());
    }
    node.engine.check_data_stream_write_only_probed(ds, create, Some((src, scan)))
}

/// Point a single-document write at backing index `to` instead of data
/// stream `from` (shard engines resolve concrete indices only). Only
/// create-semantics writes: an `index` op on a backing index is an error
/// of its own, which the data stream path must keep reporting.
fn retarget_write(req: &RestReq, from: &str, to: &str) -> elasticsearch_engine::error::Result<Option<RestReq>> {
    let segs = req.segments();
    let params = req.params();
    let create = match (req.method.as_str(), segs.get(1).map(String::as_str), segs.len()) {
        ("POST", Some("_doc"), 2) | (_, Some("_create"), _) => true,
        _ => params.get("op_type").map(String::as_str) == Some("create"),
    };
    if !create || segs.first().map(String::as_str) != Some(from) {
        return Ok(None);
    }
    let path = req.path();
    let Some(position)=path.match_indices('/').map(|(i,_)|i).find(|i|path[*i+1..].starts_with(&segs[1])) else{return Ok(None)};
    let rest=&path[position..];
    struct WriteQuery<'a>{raw:&'a str,create:bool}
    impl std::fmt::Display for WriteQuery<'_>{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{
        let mut sep="?";
        for part in self.raw.split('&').filter(|p|!p.is_empty()&&!p.starts_with("require_data_stream")){write!(f,"{sep}{part}")?;sep="&";}
        if self.create{write!(f,"{sep}op_type=create")?;}Ok(())
    }}
    // The data stream made an id-less write a `create`; spell it out for the backing index.
    let query=WriteQuery{raw:req.query(),create:!params.contains_key("op_type")};
    let mut out = req.clone();
    out.set_uri_fmt(format_args!("/{}{rest}{query}", UriComponent(to)))?;
    Ok(Some(out))
}

/// Borrowed document route used by bulk failure redirection.
struct DocumentUri<'a>{index:&'a str,id:Option<&'a str>,create:bool,refresh:Option<&'a str>,routing:Option<&'a str>}
impl std::fmt::Display for DocumentUri<'_>{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{
    write!(f,"/{}/_doc",UriComponent(self.index))?;
    if let Some(id)=self.id{write!(f,"/{}",UriComponent(id))?;}
    let mut sep="?";
    if self.create{f.write_str("?op_type=create")?;sep="&";}
    for (key,value) in [("refresh",self.refresh),("routing",self.routing)]{if let Some(value)=value{write!(f,"{sep}{key}={}",UriComponent(value))?;sep="&";}}
    Ok(())
}}

// ---------------------------------------------------------------- serving

/// Serve the REST API on `listener`: HTTPS when `xpack.security.http.ssl`
/// is enabled ([`ClusterNode::http_tls`]; a failed handshake drops only that
/// connection), plain HTTP otherwise.
pub async fn serve(node: Arc<ClusterNode>, listener: tokio::net::TcpListener) -> std::io::Result<()> {
    let tls = node.http_tls.clone();
    let app = axum::Router::new().fallback(move |req: Request<Body>| {
        let node = node.clone();
        elasticsearch_http::panic_guard(async move { handle_http(node, req).await })
    });
    let app = app.into_make_service_with_connect_info::<elasticsearch_http::HttpConnection>();
    match tls {
        Some(config) => axum::serve(elasticsearch_http::TlsListener::new(listener, config)?, app).await,
        None => axum::serve(listener, app).await,
    }
}

async fn handle_http(node:Arc<ClusterNode>,req:Request<Body>)->Response<Body>{
    let control=elasticsearch_http::in_flight_policy(req.method().as_str(),req.uri().path(),elasticsearch_http::incremental_bulk(&node.engine))==elasticsearch_http::InFlight::Holds;
    let started=std::time::Instant::now();
    let r=elasticsearch_engine::memory::request_policy_future(control,handle_http_inner(node,req)).await;
    stats_phases::record("front_http", vec![("total", started.elapsed().as_micros())]);
    r
}
async fn handle_http_inner(node: Arc<ClusterNode>, req: Request<Body>) -> Response<Body> {
    let (parts, body) = req.into_parts();
    // ES `RestController`: the client's body is held on `inflight_requests`
    // here, at the edge, until the response is built (the engine dispatches
    // it makes below hold their copies without tripping).
    let policy = elasticsearch_http::in_flight_policy(parts.method.as_str(), parts.uri.path(), elasticsearch_http::incremental_bulk(&node.engine));
    let authorization = parts.headers.get("authorization").and_then(|v| v.to_str().ok());
    let authenticated=elasticsearch_http::security::authenticated(&node.engine, authorization);
    let mut _inflight = match elasticsearch_http::InFlightBody::reserve(node.engine.breakers, policy, 0, || authenticated) {
        Ok(r) => r,
        Err(e) => return to_http(err(e)),
    };
    let body=match elasticsearch_http::collect_request_body(body,MAX_CONTENT_LENGTH,&mut _inflight,authenticated).await {
        Ok(body)=>body,Err(error)=>return to_http(err(error)),
    };
    let http = elasticsearch_http::http_request(&parts.method, parts.uri.path(),
        parts.uri.path_and_query().map(|p| p.as_str()).unwrap_or("/"), &parts.headers,
        body.len() as u64, parts.extensions.get::<axum::extract::ConnectInfo<elasticsearch_http::HttpConnection>>().map(|c| &c.0));
    let mut edge_memory=match node.engine.breakers.reserve_native(0,"HTTP edge metadata"){Ok(memory)=>memory,Err(error)=>return to_http(err(error))};
    let mut headers = match header_vec(&parts.headers,&mut edge_memory){Ok(headers)=>headers,Err(error)=>return to_http(err(error))};
    headers.retain(|(name, _)| !name.eq_ignore_ascii_case(crate::shard_engines::PREFILTER_SHARDS_HEADER)
        && !name.eq_ignore_ascii_case(ingest::PREPARED_HEADER)
        && !name.eq_ignore_ascii_case(failure_store::TOKEN_HEADER)
        && !name.eq_ignore_ascii_case(failure_store::TOKENS_HEADER)
        && !name.eq_ignore_ascii_case(failure_store::ACK_HEADER)
        && !name.eq_ignore_ascii_case(failure_store::POSITIONS_HEADER)
        && !name.eq_ignore_ascii_case(DATA_STREAM_BACKING_WRITE_HEADER)
        && !name.eq_ignore_ascii_case(esql_dist::FETCH_HEADER)
        && !name.eq_ignore_ascii_case(SYNONYMS_INNER_HEADER)
        && !name.eq_ignore_ascii_case(SYSTEM_HEADER)
        && !name.eq_ignore_ascii_case(RANK_PLAN_HEADER)
        && !name.eq_ignore_ascii_case(qtf::HEADER)
        && !name.eq_ignore_ascii_case(qtf::CTX_HEADER)
        && !name.eq_ignore_ascii_case(qtf::PINNED_HEADER)
        && !name.eq_ignore_ascii_case(qtf::LIGHT_HEADER)
        && !name.eq_ignore_ascii_case(elasticsearch_http::HIT_SPANS_HEADER)
        && !name.eq_ignore_ascii_case(contexts::SCROLL_TOUCH_HEADER)
        && !name.eq_ignore_ascii_case(esql_dist::COORDINATOR_HEADER)
        && !name.eq_ignore_ascii_case(copies::READ_COPIES_HEADER)
        && !name.eq_ignore_ascii_case(fleet_search::WAIT_HEADER)
        && !name.eq_ignore_ascii_case(NESTED_HEADER)
        && !name.eq_ignore_ascii_case(adaptive::QUERY_HEADER)
        && !name.eq_ignore_ascii_case(TRIPS_HEADER));
    // Content codings are negotiated here, at the client edge, only: every
    // internal sub-request (per-shard bulk, forwards, fan-outs) must see
    // plain bodies, or the coordinator can't parse the replies it merges.
    let accepts_gzip = header_has(&headers, "accept-encoding", "gzip");
    let gzipped = header_has(&headers, "content-encoding", "gzip");
    let body=match elasticsearch_http::decode_request_body_guarded(body,gzipped,MAX_CONTENT_LENGTH,&mut _inflight,authenticated) {
        Ok(body)=>body,Err(error)=>return to_http(err(error)),
    };
    if gzipped {headers.retain(|(k,_)|!k.eq_ignore_ascii_case("content-encoding")&&!k.eq_ignore_ascii_case("content-length"));}
    headers.retain(|(k, _)| !k.eq_ignore_ascii_case("accept-encoding"));
    let is_head = parts.method == Method::HEAD;
    let mut uri = match admitted::text(&mut edge_memory,parts.uri.path_and_query().map(|p|p.as_str()).unwrap_or("/")){Ok(uri)=>uri,Err(error)=>return to_http(err(error))};
    // YAML / CBOR / Smile are decoded and encoded here, at the client edge:
    // every internal request and reply between nodes is JSON.
    let (body, wire, map) = match xcontent_edge(parts.method.as_str(), &parts.headers, &mut headers, &mut uri, body, &mut edge_memory) {
        Ok(x) => x,
        Err((r, wire)) => return to_http(encode_wire(r, wire.as_ref(), is_head)),
    };
    let json_body = map.as_ref().map(|_| body.clone());
    let mut req = match RestReq::try_new(parts.method.as_str(),&uri,&headers,&body,0,Dfs::None,None){Ok(value)=>value,Err(error)=>return to_http(err(error))};
    // ES applies `filter_path` once, to the final response. Coordinated APIs
    // merge shard / node parts that must arrive whole (sort values, `_shards`,
    // totals), so the parameter stops here and filters the answer below.
    let filter_path = match take_filter_path(&mut req){Ok(value)=>value,Err(error)=>return to_http(err(error))};
    record_search_usage(&node, &req);
    let mut resp = if node.engine.security.enabled() {
        authorized_frontend(&node, &req).await
    } else {
        match ingest::prepare(&node, req, None).await {
            Ok(req) => dispatch(&node, req).await,
            Err(error) => error,
        }
    };
    // Preserve the already-decided authentication result through optional
    // response remapping, rendering and encoding under memory pressure.
    let unauthorized=resp.status==401;
    // Errors about the decoded JSON point into the original body.
    if let (Some(m), Some(jb)) = (map.as_ref(), json_body.as_ref()) {
        if resp.status >= 400 {
            if let Ok(text)=std::str::from_utf8(&resp.body) {
                match crate::owned_json::OwnedJson::parse(text) {
                    Ok(mut value)=>match value.transform(|v,memory|elasticsearch_http::xcontent_in::remap_error_positions(v,jb,m,memory)) {
                        Ok(())=>resp.replace_json(&value), Err(error)=>resp=err(error),
                    },
                    Err(error) if error.status==429=>resp=err(error),
                    Err(_)=>{}, // Preserve non-JSON error bodies unchanged.
                }
            }
        }
    }
    resp.retain_headers(|(name, _)| !name.eq_ignore_ascii_case(failure_store::ACK_HEADER) && !name.eq_ignore_ascii_case(failure_store::POSITIONS_HEADER)
        && !name.eq_ignore_ascii_case(ESQL_SHARDS_HEADER) && !name.eq_ignore_ascii_case(esql_dist::FORWARD_HEADER));
    if let Some((fp, pretty)) = filter_path.filter(|_| !is_head) {
        resp = crate::rest::util::render_json(resp, Some(&fp), pretty, false);
    }
    let resp = encode_wire(resp, wire.as_ref(), is_head);
    let mut response = if accepts_gzip && !is_head { gzip_response(resp) } else { to_http(resp) };
    if unauthorized && matches!(response.status().as_u16(),429|500){response=authentication_required_http(is_head);}
    if unauthorized && is_head {*response.body_mut()=Body::empty();}
    http.done(&response);
    response
}

/// Remove a client request's non-empty `filter_path` (to apply at the
/// client edge, see `handle_http`); returns it with the `pretty` flag the
/// re-rendered body must keep.
fn take_filter_path(req: &mut RestReq) -> elasticsearch_engine::error::Result<Option<(String, bool)>> {
    let params = req.params();
    let Some(fp)=params.get("filter_path").filter(|f|!f.is_empty()) else {return Ok(None)};
    let fp=fp.clone();
    let pretty = matches!(params.get("pretty").map(String::as_str), Some("" | "true"));
    let original=req.clone();
    req.set_uri_fmt(format_args!("{}{}",original.path(),FilteredQuery{query:original.query(),drop:&["filter_path"],append:""}))?;
    Ok(Some((fp, pretty)))
}

/// Count a client search's analytics aggregations once, on the node that
/// received it (ES parses the request once, on the coordinating node).
fn record_search_usage(node: &ClusterNode, req: &RestReq) {
    let segs = req.segments();
    let Some(last) = segs.iter().rev().find(|s| s.starts_with('_')) else { return };
    match last.as_str() {
        "_search" | "_async_search" if segs.get(1).map(String::as_str) != Some("scroll") => {
            // Borrowed: the parse is kept for the rest of the request.
            if let Some(body) = req.json_ref() {
                elasticsearch_http::record_analytics(&node.engine, body);
            }
        }
        "_msearch" => {
            let text = String::from_utf8_lossy(&req.body);
            for line in text.lines().filter(|l| !l.trim().is_empty()).skip(1).step_by(2) {
                if let Ok(body) = serde_json::from_str::<Value>(line) {
                    elasticsearch_http::record_analytics(&node.engine, &body);
                }
            }
        }
        _ => {}
    }
}

/// Run the existing HTTP validation/security pipeline before the cluster
/// coordinator. The continuation is an in-process extension, never a header.
async fn authorized_frontend(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    // Authorization distinguishes an existing index from auto-creation. The
    // applied cluster state can precede its engine mirror; wait before checking
    // privileges so a newly created index does not require create_index again.
    await_mirror(node).await;
    let builder = Request::builder().method(req.method()).uri(&req.uri);
    let headers=match elasticsearch_http::outgoing_headers::collect_valid(req.headers.iter().map(|(k,v)|(k.as_str(),v.as_str()))) {
        Ok(headers)=>headers,Err(error)=>return err(error),
    };
    let body = match admitted::http_body(&req.body, node.engine.breakers) {
        Ok(body) => body,
        Err(error) => return RestResp::json(error.status, &error.to_json()),
    };
    let Ok(mut request) = builder.body(Body::from(body)) else {
        return RestResp::error(400, "illegal_argument_exception", "invalid request");
    };
    *request.headers_mut()=headers;
    let state = node.state();
    let mut existing: BTreeSet<String> = state.metadata.indices.keys().cloned().collect();
    for index in state.metadata.indices.values() { existing.extend(index.aliases.keys().cloned()); }
    existing.extend(node.engine.data_streams.read().keys().cloned());
    request.extensions_mut().insert(elasticsearch_http::InternalHttpDispatch);
    request.extensions_mut().insert(elasticsearch_http::AuthorizationIndexCatalog(existing));
    let node_for_hook = node.clone();
    request.extensions_mut().insert(elasticsearch_http::AuthorizedDispatch(Arc::new(move |authorized| {
        let node = node_for_hook.clone();
        Box::pin(async move {
            // Keep empty identifier segments for the endpoint's validation.
            // Reconstructing from normalized segments turns /_connector//_name
            // into a request for a connector named _name.
            let original_segments: Vec<_> = authorized.path.split('/').filter(|s| !s.is_empty()).map(decode_path).collect();
            let mut uri = if original_segments == authorized.segs {
                authorized.path.clone()
            } else {
                // Authorization may replace a wildcard with the permitted
                // indices or insert the permitted targets into /_search.
                format!("/{}", authorized.segs.iter().map(|s| encode_component(s)).collect::<Vec<_>>().join("/"))
            };
            if !authorized.params.is_empty() {
                let mut params: Vec<_> = authorized.params.iter().map(|(k, v)| format!("{}={}", encode_component(k), encode_component(v))).collect();
                params.sort();
                uri.push('?');
                uri.push_str(&params.join("&"));
            }
            // Retain the body until the outer security layer has redacted
            // mappings; its HTTP finish step applies HEAD afterwards.
            let admitted = admitted::Request::authorized(
                if authorized.method == Method::HEAD { "GET" } else { authorized.method.as_str() },
                &uri, &authorized.headers, &authorized.body, node.engine.breakers,
            )?;
            let request_memory = admitted.memory;
            let rest = admitted.value;
            let user = authorized.access.as_deref().map(|access| access.user().clone());
            let response = match ingest::prepare(&node, rest, user).await {
                Ok(rest) => dispatch(&node, rest).await,
                Err(error) => error,
            };
            drop(request_memory);
            let mut header_memory = elasticsearch_engine::memory::breakers().reserve_request(response::header_owner_bytes() as u64, "coordinator HTTP response owner")?;
            let mut headers = admitted::slots::<(&'static str, String)>(&mut header_memory, response.headers.len())?;
            for (key, value) in &response.headers {
                let name = match key.to_ascii_lowercase().as_str() {
                    "content-type" => "content-type", "content-length" => "content-length",
                    "warning" => "warning", "allow" => "allow", "location" => "location",
                    "etag" => "etag", "cache-control" => "cache-control", "retry-after" => "retry-after",
                    "www-authenticate" => "www-authenticate", "x-elastic-product" => "x-elastic-product",
                    "x-opaque-id" => "x-opaque-id", "content-encoding" => "content-encoding",
                    // SQL text formats carry the cursor and async state in headers.
                    "cursor" => "cursor", "async-id" => "async-id", "async-partial" => "async-partial", "async-running" => "async-running",
                    _ => return Err(elasticsearch_engine::error::EsError::new(500, "illegal_state_exception", format!("unsupported coordinator response header [{key}]"))),
                };
                headers.push((name, admitted::text(&mut header_memory, value)?));
            }
            let json = headers.iter().any(|(key, value)| *key == "content-type" && value.contains("json"));
            if json {std::str::from_utf8(&response.body).map_err(|_|elasticsearch_engine::error::EsError::new(500,"illegal_state_exception","coordinator returned invalid JSON encoding"))?;}
            let body = elasticsearch_http::RespBody::Owned { bytes: response::with_header_owner(response.body.clone(),header_memory), json };
            Ok(elasticsearch_http::Resp { status: response.status, body, headers })
        })
    })));
    let response = match node.router.clone().oneshot(request).await {
        Ok(response) => response,
        Err(error) => match error {},
    };
    let result=match RestResp::http_metadata(&response,req.method=="HEAD"){Ok(result)=>result,Err(error)=>return err(error)};
    result.collect(response.into_body(),req.method=="HEAD").await
}

type Wire = (elasticsearch_http::wire::Wire, String);

/// Decode a YAML / CBOR / Smile request body into JSON (`elasticsearch_http::xcontent_in`)
/// and pick the response encoding (`?format`, `Accept`, else the body's
/// type). The request then travels as JSON: its `Accept` and `?format` ask
/// for JSON, its `Content-Type` is JSON (`x-ndjson` for bulk-style bodies).
#[allow(clippy::type_complexity)]
fn xcontent_edge(method: &str, hm: &HeaderMap, headers: &mut Vec<(String, String)>, uri: &mut String, body: Vec<u8>, edge_memory: &mut elasticsearch_engine::memory::Reservation) -> Result<(bytes::Bytes, Option<Wire>, Option<elasticsearch_engine::jsonpos::SourceMap>), (RestResp, Option<Wire>)> {
    use elasticsearch_http::xcontent_in as xc;
    use elasticsearch_engine::{error::EsError, memory::Reservation};
    let (path, query) = uri.split_once('?').unwrap_or((uri.as_str(), ""));
    let format = query.split('&').find_map(|kv| kv.strip_prefix("format="));
    let wire = elasticsearch_http::negotiate_wire_admitted(hm, format, !body.is_empty(), edge_memory).map_err(|error|(err(error),None))?;
    macro_rules! admitted_try { ($value:expr) => { match $value { Ok(value) => value, Err(error) => return Err((err(error), wire)) } }; }
    // These exact copies allow URI replacement without retaining borrowed slices
    // into the old String. Their owner is held by the dispatch caller.
    let path = admitted_try!(admitted::text(edge_memory, path));
    let query = admitted_try!(admitted::text(edge_memory, query));
    let rewrite_format = matches!(format, Some("yaml" | "cbor" | "smile"));
    fn compatibility(value: &str) -> Option<&str> {
        value.split(';').find_map(|part| {
            let part=part.trim(); let prefix="compatible-with=";
            part.get(..prefix.len()).filter(|p|p.eq_ignore_ascii_case(prefix)).map(|_|&part[prefix.len()..])
        })
    }
    fn json_type(memory: &mut Reservation, compat: Option<&str>, fallback: &str) -> Result<String,EsError> {
        let Some(compat)=compat else {return admitted::text(memory,fallback)};
        let prefix="application/vnd.elasticsearch+json;compatible-with=";
        let length=prefix.len().checked_add(compat.len()).ok_or_else(||EsError::new(429,"circuit_breaking_exception","content type length overflow"))?;
        let mut bytes=admitted::slots::<u8>(memory,length)?;
        bytes.extend_from_slice(prefix.as_bytes());bytes.extend(compat.bytes().map(|b|b.to_ascii_lowercase()));
        // ASCII case folding preserves the UTF-8 validity of the borrowed text.
        Ok(String::from_utf8(bytes).expect("ASCII folding preserves UTF-8"))
    }
    fn room(memory:&mut Reservation,headers:&mut Vec<(String,String)>,additional:usize)->Result<(),EsError>{
        let needed=headers.len().checked_add(additional).ok_or_else(||EsError::new(429,"circuit_breaking_exception","header capacity overflow"))?;
        if needed>headers.capacity(){
            // Admit the complete replacement while the original backing remains
            // live, then move existing entries without cloning their strings.
            let mut replacement=admitted::slots(memory,needed)?;
            replacement.append(headers);*headers=replacement;
        }
        Ok(())
    }
    if wire.is_some() {
        let value=admitted_try!(json_type(edge_memory,headers.iter().find(|(n,_)|n.eq_ignore_ascii_case("accept")).and_then(|(_,v)|compatibility(v)),"application/json"));
        admitted_try!(room(edge_memory,headers,1));
        let name=admitted_try!(admitted::text(edge_memory,"accept"));
        headers.retain(|(k,_)|!k.eq_ignore_ascii_case("accept"));headers.push((name,value));
        if rewrite_format {
            let mut length=path.len().checked_add(1).ok_or_else(||EsError::new(429,"circuit_breaking_exception","URI length overflow"));
            for (i,part) in query.split('&').enumerate(){
                let part=if part.starts_with("format="){"format=json"}else{part};
                length=length.and_then(|n|n.checked_add(part.len()).and_then(|n|n.checked_add(usize::from(i>0))).ok_or_else(||EsError::new(429,"circuit_breaking_exception","URI length overflow")));
            }
            let mut bytes=admitted_try!(admitted::slots::<u8>(edge_memory,admitted_try!(length)));
            bytes.extend_from_slice(path.as_bytes());bytes.push(b'?');
            for (i,part) in query.split('&').enumerate(){if i>0{bytes.push(b'&');}bytes.extend_from_slice(if part.starts_with("format="){b"format=json"}else{part.as_bytes()});}
            *uri=String::from_utf8(bytes).expect("URI slices preserve UTF-8");
        }
    }
    let Some(ct)=headers.iter().find(|(n,_)|n.eq_ignore_ascii_case("content-type")).map(|(_,v)|v.as_str()) else {return Ok((bytes::Bytes::from(body),wire,None))};
    let Some(f)=xc::in_format(ct).filter(|_|!body.is_empty()) else {return Ok((bytes::Bytes::from(body),wire,None))};
    let delimited=xc::is_delimited_endpoint(&path);
    if delimited && !xc::supports_delimited(f){
        let reason=admitted_try!(SharedText::try_format(format_args!("Content-Type header [{ct}] is not supported")));
        #[derive(serde::Serialize)] struct Unsupported<'a>{error:&'a str,status:u16}
        let encoded=admitted_try!(elasticsearch_http::wire::json_admitted(&Unsupported{error:&reason,status:406},false));
        return Err((RestResp::with_payload(406,std::iter::once(("content-type","application/json")),encoded).unwrap_or_else(err),wire));
    }
    let d=admitted_try!(if delimited{xc::decode_stream(f,&body)}else{xc::decode(f,&body)});
    match admitted_try!(xc::decode_failure(method,&path,&d)) {
        Some(body)=>return Err((RestResp::with_payload(400,std::iter::once(("content-type","application/json")),body).unwrap_or_else(err),wire)),
        None=>{}
    }
    let json_ct=admitted_try!(json_type(edge_memory,compatibility(ct),if delimited{"application/x-ndjson"}else{"application/json"}));
    let ct=admitted_try!(admitted::text(edge_memory,ct));
    admitted_try!(room(edge_memory,headers,2));
    let json_name=admitted_try!(admitted::text(edge_memory,"content-type"));
    let original_name=admitted_try!(admitted::text(edge_memory,xc::ORIGINAL_CONTENT_TYPE));
    headers.retain(|(k,_)|!k.eq_ignore_ascii_case("content-type")&&!k.eq_ignore_ascii_case("content-length")&&!k.eq_ignore_ascii_case(xc::ORIGINAL_CONTENT_TYPE));
    headers.push((json_name,json_ct));headers.push((original_name,ct));
    Ok((d.json,wire,d.map))
}

/// Re-encode a JSON reply in the negotiated YAML / CBOR / Smile.
fn encode_wire(mut r: RestResp, wire: Option<&Wire>, is_head: bool) -> RestResp {
    let Some((w, ct)) = wire else { return r };
    let is_json = r.headers.iter().any(|(k, v)| k.eq_ignore_ascii_case("content-type") && (v.starts_with("application/json") || v.starts_with("application/vnd.elasticsearch+json")));
    if !is_json || is_gzip(&r) {
        return r;
    }
    if !is_head && !r.body.is_empty() {
        let Ok(text)=std::str::from_utf8(&r.body) else{return r};
        let v=match crate::owned_json::OwnedJson::parse(text){Ok(v)=>v,Err(error)if error.status==429=>return err(error),Err(_)=>return r};
        match elasticsearch_http::wire::encode_admitted(*w, &v) {
            Ok(body) => r.replace_body(body),
            Err(error) => return RestResp::error(error.status, &error.kind, &error.reason),
        }
    }
    r.retain_headers(|(k, _)| !k.eq_ignore_ascii_case("content-type") && !k.eq_ignore_ascii_case("content-length"));
    if let Err(error)=r.append_header("content-type",ct){return err(error);}
    r
}

fn header_has(h: &[(String, String)], name: &str, needle: &str) -> bool {
    h.iter().any(|(k, v)| k.eq_ignore_ascii_case(name) && v.to_ascii_lowercase().contains(needle))
}

/// Periodic cleanup of coordinator-held contexts (expired async searches).
pub(crate) fn reap_expired_contexts() {
    contexts::reap_asyncs();
}

/// Inflate a `Content-Encoding: gzip` request body (as the engine does) and
/// drop the header so forwarded sub-requests carry plain bodies. The
/// inflated size is bounded by `limit` like the transport size: a small
/// gzip body can expand ~1000x.
#[cfg(test)]
fn decode_request_body(headers: &mut Vec<(String, String)>, body: Vec<u8>, limit: usize) -> Result<Vec<u8>, RestResp> {
    if !header_has(headers, "content-encoding", "gzip") {
        return Ok(body);
    }
    headers.retain(|(k, _)| !k.eq_ignore_ascii_case("content-encoding") && !k.eq_ignore_ascii_case("content-length"));
    let mut budget=elasticsearch_http::InFlightBody::reserve(elasticsearch_engine::memory::breakers(),elasticsearch_http::InFlight::Holds,0,||true)
        .map_err(|error|RestResp::json(error.status,&error.to_json()))?;
    elasticsearch_http::decode_request_body_guarded(body,true,limit,&mut budget,true)
        .map_err(|error|RestResp::json(error.status,&error.to_json()))
}

/// Gzip the final client response (bodies over 64 bytes, like the engine)
/// when the client sent `Accept-Encoding: gzip`.
fn gzip_response(mut r: RestResp) -> Response<Body> {
    if is_gzip(&r) || r.body.len() <= 64 {
        return to_http(r);
    }
    let Ok(body) = elasticsearch_http::compression::gzip_body(&r.body) else { return to_http(r) };
    r.replace_body([]);
    r.retain_headers(|(k, _)| !k.eq_ignore_ascii_case("content-length") && !k.eq_ignore_ascii_case("vary"));
    r.push_header(("content-encoding".into(), "gzip".into()));
    r.push_header(("vary".into(), "Accept-Encoding".into()));
    if r.is_admission_refused(){return to_http(r);}
    let mut response = to_http(r);
    *response.body_mut() = body;
    response
}

fn header_vec(h: &HeaderMap,memory:&mut elasticsearch_engine::memory::Reservation) -> Result<Vec<(String,String)>,elasticsearch_engine::error::EsError> {
    let count=h.iter().filter(|(_,value)|value.to_str().is_ok()).count();
    let mut out=admitted::slots(memory,count)?;
    for (key,value) in h.iter(){if let Ok(value)=value.to_str(){out.push((admitted::text(memory,key.as_str())?,admitted::text(memory,value)?));}}
    Ok(out)
}

fn authentication_required_http(head:bool)->Response<Body> {
    Response::builder().status(401).header("x-elastic-product","Elasticsearch")
        .header("content-type","application/json").header("www-authenticate","Basic realm=\"security\", charset=\"UTF-8\"")
        .body(if head{Body::empty()}else{Body::from(Bytes::from_static(br#"{"error":{"type":"security_exception","reason":"authentication required"},"status":401}"#))}).unwrap()
}

fn to_http(r: RestResp) -> Response<Body> {
    let headers=r.headers.iter().filter(|(k,_)|!k.eq_ignore_ascii_case("content-length")&&!k.eq_ignore_ascii_case("transfer-encoding")&&!k.eq_ignore_ascii_case("x-elastic-product")).map(|(k,v)|(k.as_str(),v.as_str()));
    match elasticsearch_http::outgoing_headers::response(r.status,headers.chain(std::iter::once(("x-elastic-product","Elasticsearch"))),Body::from(r.body.clone())) {
        Ok(response)=>response,
        Err(error)=>{
            // Authentication is already decided; header admission must not
            // replace its challenge. Never copy an arbitrary failed body.
            if r.status==401 {return authentication_required_http(r.body.is_empty());}
            let (status,body):(u16,&'static [u8])=if error.status==429 {(429,br#"{"error":{"type":"circuit_breaking_exception","reason":"outgoing headers refused"},"status":429}"#)}
                else {(500,br#"{"error":{"type":"illegal_state_exception","reason":"invalid outgoing headers"},"status":500}"#)};
            Response::builder().status(status).header("content-type","application/json").body(Body::from(bytes::Bytes::from_static(body))).unwrap()
        },
    }
}

// --------------------------------------------------------------- dispatch

/// Marks a forwarded request that must run on the receiving node's engine
/// as-is (its context ids were already unwrapped by the sender).
const EXEC_LOCAL: &str = "x-es-cluster-exec-local";

/// `uri` without the query parameter `name`.
#[cfg(test)]
fn without_query_param(uri: &str, name: &str) -> String {
    let Some((path, query)) = uri.split_once('?') else { return uri.to_string() };
    let kept: Vec<&str> = query.split('&').filter(|p| !p.is_empty() && p.split('=').next() != Some(name)).collect();
    if kept.is_empty() { path.to_string() } else { format!("{path}?{}", kept.join("&")) }
}

#[cfg(test)]
mod query_param_tests {
    use serde_json::json;
    use super::{bulk, bulk_resolution_failure, copy_replaced, without_query_param, RestResp};


    #[test]
    fn bulk_resolution_refusal_preserves_prior_success() {
        let items=bulk::parse(b"{\"index\":{\"_index\":\"i\",\"_id\":\"1\"}}\n{}\n{\"index\":{\"_index\":\"i\",\"_id\":\"2\"}}\n{}\n",None).unwrap();
        let refusal=RestResp::error(429,"circuit_breaking_exception","covering index admission");
        let response=bulk::merge(&items,vec![
            (vec![0],bulk::PartResult::Response(json!({"items":[{"index":{"_id":"1","status":201,"result":"created"}}]}))),
            (vec![1],bulk_resolution_failure(&refusal)),
        ]);
        assert_eq!(response["items"][0]["index"]["status"],201);
        assert_eq!(response["items"][0]["index"]["result"],"created");
        assert_eq!(response["items"][1]["index"]["status"],429);
        assert_eq!(response["items"][1]["index"]["_id"],"2");
        assert_eq!(response["items"][1]["index"]["error"]["type"],"circuit_breaking_exception");
        assert_eq!(response["items"].as_array().unwrap().len(),2);
    }

    #[test]
    fn replaced_copy_errors_are_retried_and_others_are_not() {
        let err = |status: u16, kind: &str| RestResp::json(status, &serde_json::json!({"error": {"type": kind, "reason": "x"}, "status": status}));
        assert!(copy_replaced(&err(503, elasticsearch_engine::index::COPY_REPLACED)));
        assert!(!copy_replaced(&err(503, "unavailable_shards_exception")));
        assert!(!copy_replaced(&err(500, "exception")));
    }

    #[test]
    fn drops_only_the_named_parameter() {
        assert_eq!(without_query_param("/_bulk?refresh=true&require_data_stream=true", "require_data_stream"), "/_bulk?refresh=true");
        assert_eq!(without_query_param("/_bulk?require_data_stream=true", "require_data_stream"), "/_bulk");
        assert_eq!(without_query_param("/_bulk?require_data_stream_x=1&a=b", "require_data_stream"), "/_bulk?require_data_stream_x=1&a=b");
        assert_eq!(without_query_param("/_bulk", "require_data_stream"), "/_bulk");
    }
}

/// `POST /_query` runs on the elected master (the owner of ES|QL views),
/// reading every shard from a started copy, local or on another node
/// (`esql_dist`). When some shards are not readable here but another node
/// holds started copies of all of them, the whole query runs there.
/// Shards with no started copy anywhere yet are waited for.
async fn esql_on_copies(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    if !node.is_master() {
        return match node.state().master_node_id.clone() {
            Some(master) => forward(node, &master, req).await,
            None => RestResp::error(503, "master_not_discovered_exception", ""),
        };
    }
    let started = tokio::time::Instant::now();
    let deadline = started + Duration::from_secs(30);
    // Run as the coordinator: a node holding every shard may take the query.
    let mut coordinating = req.clone();
    if let Err(error)=coordinating.push_header(esql_dist::COORDINATOR_HEADER,"1"){return err(error)};
    loop {
        let mut resp = local_canonical(node, &coordinating).await;
        if let Some(i) = resp.headers.iter().position(|(name, _)| name.eq_ignore_ascii_case(esql_dist::FORWARD_HEADER)) {
            let target = &resp.headers[i].1;
            // Bounded by the node leaving the cluster; a node that failed the
            // query as a node leaves it to this one, reading its shards
            // remotely (no second forward).
            let mut remote = copy_request(node, req, &target, None).await;
            if copies::node_failure(&remote) {
                if let Err(error)=coordinating.retain_headers(|(name,_)|!name.eq_ignore_ascii_case(esql_dist::COORDINATOR_HEADER)){return err(error)};
                continue;
            }
            remote.retain_headers(|(name, _)| !name.eq_ignore_ascii_case(ESQL_SHARDS_HEADER) && !name.eq_ignore_ascii_case(esql_dist::FORWARD_HEADER));
            return remote;
        }
        let Some(i) = resp.headers.iter().position(|(name, _)| name.eq_ignore_ascii_case(ESQL_SHARDS_HEADER)) else { return resp };
        let unread: Vec<(String, u32)> = serde_json::from_str::<Vec<String>>(&resp.headers[i].1).unwrap_or_default().iter()
            .filter_map(|key| key.rsplit_once('#').and_then(|(index, shard)| Some((index.to_string(), shard.parse().ok()?)))).collect();
        resp.retain_headers(|(name, _)| !name.eq_ignore_ascii_case(ESQL_SHARDS_HEADER));
        let grace = started.elapsed() >= Duration::from_secs(3);
        match esql_placement(&node.state(), &unread, grace) {
            EsqlPlacement::Partial => return resp,
            EsqlPlacement::Wait if tokio::time::Instant::now() >= deadline => return resp,
            EsqlPlacement::Wait => tokio::time::sleep(Duration::from_millis(50)).await,
        }
    }
}

/// What an ES|QL query that found no started copy of the shards in
/// `unread` does next.
#[derive(Debug, PartialEq)]
enum EsqlPlacement {
    /// Copies are being allocated, recovering, or started since: retry.
    Wait,
    /// Every copy of those shards is unassigned (after a short grace, as an
    /// unassigned copy may never be allocated): read the other shards only.
    Partial,
}

fn esql_placement(state: &elasticsearch_cluster::state::ClusterState, unread: &[(String, u32)], grace: bool) -> EsqlPlacement {
    let settled = unread.iter().all(|(index, shard)| state.routing_table.shard_table(index, *shard)
        .is_some_and(|t| t.copies.iter().all(|c| grace && c.node_id.is_none())));
    if settled { EsqlPlacement::Partial } else { EsqlPlacement::Wait }
}

/// Whether ES|QL may read the local copy of `index`'s `shard` on `me`: only
/// a started copy (a recovering one may be missing documents). Indices the
/// cluster state does not know (node-local) are always readable.
fn esql_copy_readable(state: &elasticsearch_cluster::state::ClusterState, me: &str, index: &str, shard: u32) -> bool {
    !state.metadata.indices.contains_key(index)
        || state.routing_table.shard_table(index, shard).is_some_and(|t| t.searchable_copies(&state.nodes).any(|c| c.node_id.as_deref() == Some(me)))
}

#[cfg(test)]
mod esql_placement_tests {
    use super::*;
    use elasticsearch_cluster::state::{AllocationId, ClusterState, DiscoveryNode, IndexMetadata, IndexRoutingTable, IndexShardRoutingTable, IndexState, Role, ShardRouting, ShardState};
    use std::collections::BTreeMap;

    fn copy(shard: u32, primary: bool, state: ShardState, node: Option<&str>) -> ShardRouting {
        ShardRouting { index: "i".into(), shard, primary, state, node_id: node.map(str::to_string), relocating_node_id: None,
            allocation_id: node.map(|n| AllocationId { id: format!("a-{n}-{shard}"), relocation_id: None }), recovery_source: None,
            unassigned_info: None, expected_shard_size_in_bytes: None }
    }

    fn cluster(shards: Vec<Vec<ShardRouting>>) -> ClusterState {
        let mut s = ClusterState::empty("test");
        for n in ["n1", "n2"] {
            s.nodes.insert(n.into(), DiscoveryNode { id: n.into(), name: n.into(), ephemeral_id: "e".into(), transport_address: "127.0.0.1:9300".into(),
                http_address: "127.0.0.1:9200".into(), host: "h".into(), ip: "127.0.0.1".into(), roles: Role::defaults(), attributes: BTreeMap::new(), version: "9.5.5".into() });
        }
        let n = shards.len() as u32;
        s.metadata.indices.insert("i".into(), IndexMetadata { name: "i".into(), uuid: "u".into(), version: 1, mapping_version: 1, settings_version: 1,
            aliases_version: 1, state: IndexState::Open, settings: Default::default(), mappings: json!({}).into(), aliases: BTreeMap::new(), number_of_shards: n,
            number_of_replicas: 1, routing_num_shards: 1024, primary_terms: vec![1; n as usize], in_sync_allocation_ids: BTreeMap::new(), creation_date: 0, lifecycle: None });
        s.routing_table.indices.insert("i".into(), IndexRoutingTable { index_uuid: "u".into(),
            shards: shards.into_iter().enumerate().map(|(k, copies)| IndexShardRoutingTable { shard: k as u32, copies }).collect() });
        s
    }

    fn wanted(n: u32) -> Vec<(String, u32)> {
        (0..n).map(|k| ("i".to_string(), k)).collect()
    }

    use ShardState::{Initializing, Started, Unassigned};

    #[test]
    fn a_recovering_copy_is_not_readable() {
        let s = cluster(vec![vec![copy(0, true, Started, Some("n2")), copy(0, false, Initializing, Some("n1"))]]);
        assert!(!esql_copy_readable(&s, "n1", "i", 0));
        assert!(esql_copy_readable(&s, "n2", "i", 0));
    }

    #[test]
    fn an_initializing_primary_is_waited_for() {
        let s = cluster(vec![vec![copy(0, true, Initializing, Some("n2"))]]);
        assert_eq!(esql_placement(&s, &wanted(1), false), EsqlPlacement::Wait);
        assert_eq!(esql_placement(&s, &wanted(1), true), EsqlPlacement::Wait);
    }

    #[test]
    fn a_copy_started_since_the_read_is_retried() {
        let s = cluster(vec![vec![copy(0, true, Started, Some("n2"))]]);
        assert_eq!(esql_placement(&s, &wanted(1), true), EsqlPlacement::Wait);
    }

    #[test]
    fn unassigned_copies_are_waited_for_until_the_grace_ends() {
        let s = cluster(vec![vec![copy(0, true, Unassigned, None), copy(0, false, Unassigned, None)]]);
        assert_eq!(esql_placement(&s, &wanted(1), false), EsqlPlacement::Wait);
        assert_eq!(esql_placement(&s, &wanted(1), true), EsqlPlacement::Partial);
    }

    #[test]
    fn node_local_indices_are_readable() {
        let s = cluster(vec![vec![copy(0, true, Started, Some("n2"))]]);
        assert!(esql_copy_readable(&s, "n1", ".local-only", 0));
    }
}

/// Run `req` on the engine holding shard `shard` on node `target` (this
/// node's engine directly, else forwarded for local execution there).
pub(crate) async fn exec_shard(node: &ClusterNode, target: &str, req: &RestReq, shard: u32) -> RestResp {
    let mut sub = req.clone();
    if let Err(error)=sub.set_shard(shard){return err(error)};
    if target == node.local.id {
        return local_canonical(node, &sub).await;
    }
    if let Err(error)=sub.push_header(EXEC_LOCAL,"1"){return err(error)};
    forward(node, target, &sub).await
}

/// Run a context request (scroll / PIT / async search) on the node that
/// owns the context; ids in the response are wrapped by that node.
async fn exec_on(node: &Arc<ClusterNode>, owner: &str, req: RestReq) -> RestResp {
    if owner == node.local.id {
        return wrap_context_ids(node, &req, local_canonical(node, &req).await);
    }
    let mut r = req;
    if let Err(error)=r.push_header(EXEC_LOCAL,"1"){return err(error)};
    forward(node, owner, &r).await
}

pub async fn dispatch(node:&Arc<ClusterNode>,req:RestReq)->RestResp{
    let control=elasticsearch_http::in_flight_policy(&req.method,req.path(),true)==elasticsearch_http::InFlight::Holds;
    let started=std::time::Instant::now();
    let r=elasticsearch_engine::memory::request_policy_future(control,dispatch_inner(node,req)).await;
    stats_phases::record("front_dispatch", vec![("total", started.elapsed().as_micros())]);
    r
}
async fn dispatch_inner(node: &Arc<ClusterNode>, mut req: RestReq) -> RestResp {
    // Replies inside the cluster are always plain (see `handle_http`).
    if let Err(error)=req.retain_headers(|(k,_)|!k.eq_ignore_ascii_case("accept-encoding")){return err(error)};
    if req.hops == 0 {
        if let Err(error)=req.retain_headers(|(k,_)|!k.eq_ignore_ascii_case(crate::shard_engines::SHARD_HEADER)){return err(error)};
        if let Err(error)=req.retain_headers(|(k,_)|!k.eq_ignore_ascii_case(crate::rest::tasks::PARENT_HEADER)){return err(error)};
    }
    if req.hops > 0 && req.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(EXEC_LOCAL)) {
        let mut r = req;
        if let Err(error)=r.retain_headers(|(k,_)|!k.eq_ignore_ascii_case(EXEC_LOCAL)){return err(error)};
        // A bulk part naming backing indices whose primaries are here: run
        // it through the primaries' write path (gate, replication before
        // the ack, replica refresh, dynamic mapping publication), as for a
        // part the coordinator executes on its own node.
        if r.segments().last().map(String::as_str) == Some("_bulk")
            && r.headers.iter().any(|(k, v)| k.eq_ignore_ascii_case(DATA_STREAM_BACKING_WRITE_HEADER) && v == "true")
        {
            return local_bulk(node, &r).await;
        }
        if let Some(spec) = r.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(ACK_THROUGH_HEADER)).map(|(_, v)| v.clone()) {
            let segs = r.segments();
            let parts: Vec<&str> = spec.split(':').collect();
            if let (3, [shard, seq_no, term]) = (segs.len(), parts.as_slice()) {
                if segs[1] == "_fleet" && segs[2] == "global_checkpoints" {
                    let (Ok(shard), Ok(seq_no), Ok(term)) = (shard.parse::<u32>(), seq_no.parse::<i64>(), term.parse::<u64>()) else {
                        return RestResp::error(400, "illegal_argument_exception", "malformed acknowledgement request");
                    };
                    return ack_reply_json(&local_ack_through(node, &segs[0], shard, seq_no, term).await);
                }
            }
        }
        if r.method() == Method::DELETE && r.path().trim_end_matches('/') == "/_search/scroll/_all" {
            let freed = match contexts::clear_all_local(node, &r).await { Ok(n) => n, Err(e) => return err(e) };
            return RestResp::json(200, &json!({"succeeded": true, "num_freed": freed}));
        }
        return wrap_context_ids(node, &r, local_canonical(node, &r).await);
    }
    let segs = req.segments();
    let seg_refs: Vec<&str> = segs.iter().map(String::as_str).collect();
    if let Some(response) = connector_store::handle(node, &req).await {
        return response;
    }
    // Fleet secrets perform internal document/metadata requests through
    // ClusterRest. Do not hold the master's metadata mutation lock around
    // this handler: its index-creation sub-request needs that same lock.
    if seg_refs.starts_with(&["_fleet", "secret"]) {
        return local_canonical(node, &req).await;
    }
    if let Some(response) = fleet_checkpoints::handle(node, &req).await {
        return response;
    }
    if let Some(response) = fleet_search::handle(node, &req).await {
        return response;
    }
    ccr_follow::await_leader_copies(node, &req).await;
    if let Some(response) = snapshot_cache_stats::handle(node, &req).await {
        return response;
    }
    if let Some(response) = snapshot_cache_stats::handle_stats(node, &req).await {
        return response;
    }
    if let Some((source, target)) = crate::resize::resize_target(&req) {
        return crate::resize::handle(node, &req, &source, &target).await;
    }
    if req.method() == Method::POST && seg_refs == ["_query"] {
        return esql_on_copies(node, &req).await;
    }
    if let Some(kind) = xpack_query::kind(req.method.as_str(), &segs, &req.body) {
        return xpack_query::handle(node, &req, kind).await;
    }
    if req.method() == Method::GET && seg_refs == ["_xpack", "usage"] {
        return crate::rest::nodes::xpack_usage(node, &req).await;
    }
    // A synonyms set update reloads dependent analyzers on every node.
    if req.hops == 0 && matches!(req.method(), Method::PUT | Method::DELETE) && seg_refs.first() == Some(&"_synonyms") && seg_refs.len() >= 2
        && !req.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(SYNONYMS_INNER_HEADER))
    {
        let set=match SharedText::try_new(seg_refs[1]){Ok(v)=>v,Err(e)=>return err(e)};
        let mut inner = req.clone();
        if let Err(error)=inner.push_header(SYNONYMS_INNER_HEADER,"1"){return err(error)};
        let mut resp = Box::pin(dispatch(node, inner)).await;
        if resp.is_success(){
            let text=match std::str::from_utf8(&resp.body){Ok(v)=>v,Err(_)=>return resp};
            let mut body=match crate::owned_json::OwnedJson::parse(text){Ok(v)=>v,Err(e)=>return err(e)};
            if body["reload_analyzers_details"].is_object(){
                let mut reload=match crate::rest::nodes::cluster_synonyms_reload(node,&set).await{Ok(v)=>v,Err(e)=>return err(e)};
                if let Err(e)=body.prepare_reply_owner(&reload){return err(e)}
                let mut value=Value::Null;
                if let Err(e)=reload.transform(|v,_|{value=std::mem::take(v);Ok(())}){return err(e)}
                if let Err(e)=body.transform(|body,_|{body["reload_analyzers_details"]=std::mem::take(&mut value);Ok(())}){return err(e)}
                body.retain_reply_owner(reload);
                resp.replace_json(&body);
                resp.retain_headers(|(k, _)| !k.eq_ignore_ascii_case("content-length"));
            }
        }
        return resp;
    }
    let route = classify(&req.method(), &seg_refs);
    // These metadata APIs expand wildcards locally. A node with an
    // incompletely applied index mirror can otherwise return a successful
    // but partial response, which a 404 retry cannot detect.
    if req.method() == Method::GET
        && (segs.first().is_some_and(|s| matches!(s.as_str(), "_alias" | "_aliases"))
            || segs.get(1).is_some_and(|s| s == "_alias")
            || write_path::is_data_stream_get(&req))
    {
        await_mirror(node).await;
    }
    match route {
        Route::Cluster(api) => {
            let before = node.state().version;
            let resp = crate::rest::handle(node, api, &req).await;
            acknowledged(node, &req, before, resp).await
        }
        Route::Local if req.method() == Method::GET && (
            segs.first().is_some_and(|s| matches!(s.as_str(), "_connector" | "_connector_sync_job" | "_connector_secret" | "_inference"))
            || (segs.first().map(String::as_str) == Some("_ml") && segs.get(1).map(String::as_str) != Some("memory"))
            || (segs.first().map(String::as_str) == Some("_cat") && segs.get(1).map(String::as_str) == Some("ml"))
            || (segs.first().map(String::as_str) == Some("_rollup") && segs.get(1).map(String::as_str) == Some("job"))
            // Migration reindex tasks and their status live on the master.
            || (segs.first().map(String::as_str) == Some("_migration") && segs.get(1).map(String::as_str) == Some("reindex"))
            || (segs.first().map(String::as_str) == Some("_query") && segs.get(1).map(String::as_str) == Some("view"))) => {
            // These resource writes are owned by the elected master. A
            // read must use the same owner until its store is replicated.
            if node.is_master() { local_canonical(node, &req).await }
            else { crate::rest::forward_to_master(node, &req).await }
        }
        Route::Local if write_path::is_data_stream_get(&req) => write_path::data_stream_get(node, &req).await,
        Route::Local => {
            let response = local_canonical(node, &req).await;
            let segments = req.segments();
            if segments.first().map(String::as_str) == Some("_data_stream") && segments.last().map(String::as_str) == Some("_stats") {
                // Data stream stats broadcast to every active backing-index
                // copy. The engine's local header only counts local shards.
                let names: Vec<String> = response.body_json().into_iter().flat_map(|body|
                    body["data_streams"].as_array().cloned().unwrap_or_default()).filter_map(|stream|
                    stream["data_stream"].as_str().map(str::to_string)).collect();
                if !names.is_empty() {
                    let response = write_path::data_stream_stats(node, &req, response).await;
                    return crate::stats_shards::adjust(node, &req, &names.join(","), response).await;
                }
            }
            response
        },
        Route::Master => {
            let before = node.state().version;
            let resp = on_master(node, &req).await;
            acknowledged(node, &req, before, resp).await
        }
        Route::Write { index } if write_path::is_by_query(&req) => write_path::by_query(node, &req, &index).await,
        Route::Write { index } => write(node, &req, &index).await,
        Route::Bulk { default_index } => bulk_route(node, &req, default_index.as_deref()).await,
        // A client search runs as its coordinating search task.
        Route::Read { targets } if search_task::coordinates(&req) => {
            // A retriever's child search runs as the client task's child.
            let (parent_id, parent_token) = match search_task::take_linked() { Some((id, token)) => (Some(id), token), None => (None, None) };
            search_task::run(node, &req, search_task::SEARCH_ACTION, parent_id.as_deref(), parent_token, |r| async move { read(node, &r, &targets).await }).await
        }
        Route::Read { targets } => read(node, &req, &targets).await,
        Route::Broadcast { targets, op } => broadcast(node, &req, &targets, op).await,
        Route::Context(ContextKind::Scroll) if search_task::coordinates_scroll(&req) => {
            search_task::run(node, &req, search_task::SCROLL_ACTION, None, None, |r| async move { context(node, &r, ContextKind::Scroll).await }).await
        }
        Route::Context(kind) => context(node, &req, kind).await,
    }
}

/// Internal response header: the cluster state version a master-run
/// request's change was published at (for the coordinating node).
const PUBLISHED_VERSION_HEADER: &str = "x-rustic-internal-published-version";

/// ES's acknowledgement of a request that changed the cluster state: on
/// the master, the reply waits until every node applied the new state (or
/// the request's `timeout`, default 30s: then `acknowledged: false`); a
/// coordinating node that forwarded it waits until it applied that state
/// itself, so a read right after on the same node sees the change.
async fn acknowledged(node: &Arc<ClusterNode>, req: &RestReq, before: u64, mut resp: RestResp) -> RestResp {
    if matches!(req.method.as_str(), "GET" | "HEAD") || !resp.is_success() {
        resp.retain_headers(|(k, _)| !k.eq_ignore_ascii_case(PUBLISHED_VERSION_HEADER));
        return resp;
    }
    let timeout = req.params().get("timeout").and_then(|t| crate::rest::parse_timeout(t)).unwrap_or(Duration::from_secs(30));
    let published = resp.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(PUBLISHED_VERSION_HEADER)).and_then(|(_, v)| v.parse::<u64>().ok());
    resp.retain_headers(|(k, _)| !k.eq_ignore_ascii_case(PUBLISHED_VERSION_HEADER));
    match published {
        // Forwarded: the master acknowledged it; this node catches up.
        Some(version) => {
            node.wait_applied_here(version, timeout).await;
        }
        None => {
            let version = node.state().version;
            if version > before && node.is_master() {
                if !node.wait_applied_everywhere(version, timeout).await {
                    if let Some(mut v) = resp.body_json() {
                        if v.get("acknowledged").is_some() {
                            v["acknowledged"] = json!(false);
                            resp.replace_json(&v);
                        }
                    }
                }
                // For the node that forwarded it (stripped there).
                if req.hops > 0 {
                    resp.push_header((PUBLISHED_VERSION_HEADER.to_string(), version.to_string()));
                }
            }
        }
    }
    resp
}

/// These index-level metadata APIs (explicit mappings included) run as dry
/// runs on the master service's scratch copy (`crate::scratch`): the copy
/// produces the engine's own response (errors, warnings, decoded bodies)
/// and the metadata to publish; the live engines change only when the
/// applier applies the committed state.
/// `GET /_security/enroll/kibana`: answered by the master's engine, whose
/// security state gains a service account token.
pub(crate) fn is_kibana_enrollment<S: AsRef<str>>(segments: &[S]) -> bool {
    segments.len() == 3 && segments[0].as_ref() == "_security" && segments[1].as_ref() == "enroll" && segments[2].as_ref() == "kibana"
}

fn serialized_engine_metadata(req: &RestReq) -> bool {
    let segments = req.segments();
    if !matches!(req.method.as_str(), "PUT" | "POST" | "DELETE") { return false; }
    match segments.first().map(String::as_str) {
        Some("_cluster") if segments.get(1).is_some_and(|api| api == "settings") => true,
        Some("_aliases" | "_index_template" | "_component_template" | "_template" | "_data_stream") => true,
        // Analytics collections are data streams, not search-application
        // customs. Publish their templates, backing indices and stream entry
        // atomically; event POSTs write documents and must never use scratch.
        Some("_application") => segments.len() == 3
            && segments.get(1).is_some_and(|api| api == "analytics")
            && matches!(req.method.as_str(), "PUT" | "DELETE"),
        // Lifecycle steps run by the API: on the copy, like the master's passes.
        Some("_ilm") => segments.get(1).is_some_and(|api| api == "move"),
        Some(index) if !index.starts_with('_') || index == "_all" => {
            // `_rollover` judges its conditions on the cluster-wide stats
            // (gathered before the batch); `_block` is a settings change.
            segments.len() == 1
                || segments.get(1).is_some_and(|api| matches!(api.as_str(), "_alias" | "_aliases" | "_mapping" | "_settings" | "_close" | "_open" | "_rollover" | "_block"))
                || (segments.get(1).is_some_and(|api| api == "_ilm") && segments.get(2).is_some_and(|op| matches!(op.as_str(), "retry" | "remove")))
        }
        _ => false,
    }
}

/// Run a metadata request against the master's engine in the metadata
/// service (never on the coordination actor) and publish its result with
/// the rest of the batch. A failed publication rolls the engine back.
/// Only this outcome authorizes resubmission. A response/publication failure
/// after the scratch closure started is deliberately terminal: its commit may
/// be indeterminate, particularly for non-idempotent rollover requests.
enum MetadataAdmission {
    Finished(RestResp, Result<u64, String>),
    NotExecuted(RestResp),
}

const METADATA_PENDING: u8 = 0;
const METADATA_STARTED: u8 = 1;
const METADATA_ABANDONED: u8 = 2;

fn start_metadata_attempt(phase: &std::sync::atomic::AtomicU8) -> bool {
    phase.compare_exchange(METADATA_PENDING, METADATA_STARTED,
        std::sync::atomic::Ordering::AcqRel, std::sync::atomic::Ordering::Acquire).is_ok()
}

fn finish_metadata_attempt(phase: &std::sync::atomic::AtomicU8, response: RestResp,
    publication: Result<u64, String>) -> MetadataAdmission {
    // A successful scratch response is not success until its publication commits.
    let response = match &publication {
        Err(error) if response.is_success() => RestResp::error(503, "failed_to_commit_cluster_state_exception", error),
        _ => response,
    };
    if publication.is_err() && phase.compare_exchange(METADATA_PENDING, METADATA_ABANDONED,
        std::sync::atomic::Ordering::AcqRel, std::sync::atomic::Ordering::Acquire).is_ok() {
        MetadataAdmission::NotExecuted(response)
    } else {
        MetadataAdmission::Finished(response, publication)
    }
}

fn finish_scratch_attempt(phase: &std::sync::atomic::AtomicU8, response: RestResp,
    publication: Result<u64, String>, unpublished: bool) -> MetadataAdmission {
    if publication.is_err() && unpublished {
        MetadataAdmission::NotExecuted(response)
    } else {
        finish_metadata_attempt(phase, response, publication)
    }
}

fn master_admission_limit(req: &RestReq) -> Duration {
    req.params().get("master_timeout").and_then(|t| crate::rest::parse_timeout(t))
        .unwrap_or(Duration::from_secs(30)).min(Duration::from_secs(365 * 86_400))
}

/// A master task's error as ES answers it: a refusal before the task was
/// queued (`CoordinationError::is_not_master`) is a retryable 503
/// `not_master_exception`, anything else a 500.
pub(crate) fn task_error(reason: &str) -> RestResp {
    if elasticsearch_cluster::coordination::CoordinationError::is_not_master(reason) {
        RestResp::error(503, "not_master_exception", reason)
    } else {
        RestResp::error(500, "illegal_state_exception", reason)
    }
}

fn refused_not_master(r: &RestResp) -> bool {
    r.status == 503 && r.body_json().is_some_and(|v| v["error"]["type"] == "not_master_exception")
}

/// ES's `TransportMasterNodeAction` retry: `run` executes a master action
/// here; a `not_master_exception` it answers (refused before anything ran,
/// see [`task_error`]) waits for the next cluster state, then runs again
/// here or goes to the new master with what is left of `master_timeout`.
pub(crate) async fn retry_on_new_master<F, Fut>(node: &Arc<ClusterNode>, req: &RestReq, mut run: F) -> RestResp
where F: FnMut() -> Fut, Fut: std::future::Future<Output = RestResp> {
    let limit = master_admission_limit(req);
    let deadline = tokio::time::Instant::now() + limit;
    let mut resp = run().await;
    while refused_not_master(&resp) {
        if !wait_state_change(node, deadline).await {
            return RestResp::error(503, "master_not_discovered_exception", &format!("no stable master for [{}] within {}ms", req.path(), limit.as_millis()));
        }
        if node.is_master() {
            resp = run().await;
        } else if let Some(master) = node.state().master_node_id.clone().filter(|id| id != &node.local.id) {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            return match remaining_master_request(req, remaining) { Ok(bounded) => forward(node, &master, &bounded).await, Err(error) => err(error) };
        }
    }
    resp
}

/// Forward only an unexecuted request, carrying the remaining admission
/// budget rather than granting another full master_timeout on the next node.
fn remaining_master_request(req: &RestReq, remaining: Duration) -> elasticsearch_engine::error::Result<RestReq> {
    let mut request = req.clone();
    // Compare a percent-decoded key to an ASCII control key without allocating
    // decoded UTF-8 strings. Any non-ASCII byte cannot match this control key.
    fn timeout_key(raw:&str)->bool {
        let mut bytes=raw.bytes();
        for expected in b"master_timeout" {
            let Some(mut byte)=bytes.next() else{return false};
            if byte==b'%' {
                let mut probe=bytes.clone();
                let hex=|c:u8|match c {b'0'..=b'9'=>Some(c-b'0'),b'a'..=b'f'=>Some(c-b'a'+10),b'A'..=b'F'=>Some(c-b'A'+10),_=>None};
                if let (Some(a),Some(b))=(probe.next().and_then(hex),probe.next().and_then(hex)){byte=(a<<4)|b;bytes=probe;}
            }else if byte==b'+'{byte=b' ';}
            if byte!=*expected{return false;}
        }
        bytes.next().is_none()
    }
    struct RemainingQuery<'a>{raw:&'a str,millis:u128}
    impl std::fmt::Display for RemainingQuery<'_>{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{
        for part in self.raw.split('&').filter(|p|!p.is_empty()&&!timeout_key(p.split('=').next().unwrap_or(""))){write!(f,"{part}&")?;}
        write!(f,"master_timeout={}ms",self.millis)
    }}
    request.set_uri_fmt(format_args!("{}?{}", req.path(),RemainingQuery{raw:req.query(),millis:remaining.as_millis()}))?;
    Ok(request)
}

async fn commit_engine_request(node: &Arc<ClusterNode>, req: &RestReq) -> (RestResp, Result<u64, String>) {
    let deadline = tokio::time::Instant::now() + master_admission_limit(req);
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        match commit_engine_request_once(node, req, remaining).await {
            MetadataAdmission::Finished(response, publication) => return (response, publication),
            MetadataAdmission::NotExecuted(response) => {
                // Membership can already name this node while its coordinator
                // is collecting a fresh term's votes. Wait for actual authority;
                // never replay a closure that reached STARTED.
                loop {
                    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                    if remaining.is_zero() {
                        return (response, Err("master admission deadline expired before execution".into()));
                    }
                    if node.is_master() {
                        tokio::time::sleep(remaining.min(Duration::from_millis(10))).await;
                        if tokio::time::Instant::now() >= deadline {
                            return (response, Err("master admission deadline expired before execution".into()));
                        }
                        break;
                    }
                    if let Some(master) = node.state().master_node_id.clone().filter(|id| id != &node.local.id) {
                        let bounded=match remaining_master_request(req,remaining){Ok(value)=>value,Err(error)=>return (err(error), Err("request preparation refused before execution".into()))};
                        let response = forward(node, &master, &bounded).await;
                        return (response, Err("no change".into()));
                    }
                    tokio::time::sleep(remaining.min(Duration::from_millis(10))).await;
                }
            }
        }
    }
}

async fn commit_engine_request_once(node: &Arc<ClusterNode>, req: &RestReq, master_timeout: Duration) -> MetadataAdmission {

    // The scratch copy shares the live security service: only index-level
    // APIs (which never write security state) run there.
    if req.segments().first().is_some_and(|s| s.starts_with("_security")) {
        debug_assert!(false, "security APIs never run as metadata dry runs");
        return MetadataAdmission::Finished(RestResp::error(500, "illegal_state_exception", "security APIs cannot run as metadata dry runs"), Err("refused".into()));
    }
    let result = Arc::new(parking_lot::Mutex::new(None));
    let response_slot = result.clone();
    let mut request = req.clone();
    if let Err(error)=request.retain_headers(|(name,_)|!name.eq_ignore_ascii_case("accept-encoding")){return MetadataAdmission::Finished(err(error), Err("request preparation refused before execution".into()))};
    let runtime = tokio::runtime::Handle::current();
    // `master_timeout` (ES default 30s) bounds the wait for the job to be
    // picked up: a job not started by then never runs (ES drops a timed-out
    // task from the pending queue); one already running is waited out.
    let phase = Arc::new(std::sync::atomic::AtomicU8::new(METADATA_PENDING));
    let job_phase = phase.clone();
    let run: Box<dyn FnOnce(&ClusterNode, crate::master_service::On) -> bool + Send> = Box::new(move |node: &ClusterNode, on: crate::master_service::On| {
        if !start_metadata_attempt(&job_phase) {
            return false;
        }
        let router = node.router_on(on);
        // Recovery uses the live map's lock while replacing index shells.
        let uuids = node.uuids_on(on).lock();
        // The service runs batches off the async workers (`block_in_place`).
        let mut response = runtime.block_on(scratch_engine_once(node, router, &request));
        if !is_gzip(&response) {
            if let Ok(body) = std::str::from_utf8(&response.body) {
                let rewritten = match uuids.canonicalize(body) {
                    std::borrow::Cow::Owned(c) => Some(c),
                    std::borrow::Cow::Borrowed(_) => None,
                };
                if let Some(canonical) = rewritten {
                    response.replace_body(canonical.into_bytes());
                    response.retain_headers(|(name, _)| !name.eq_ignore_ascii_case("content-length"));
                }
            }
        }
        drop(uuids);
        let success = response.is_success();
        *response_slot.lock() = Some(response);
        success
    });
    // A dry run on the scratch copy; applied here once committed.
    // `_rollover` judges its conditions inside the batch on stats gathered here.
    let segs = req.segments();
    if segs.get(1).is_some_and(|api| api == "_rollover") {
        if let Some(write) = segs.first().and_then(|target| crate::node::rollover_write_index(&node.state(), target)) {
            node.prefetch_rollover_stats(vec![write]).await;
        }
    }
    let unpublished = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let job = node.metadata_service().run_with_proof(run, true, Some(unpublished.clone()));
    tokio::pin!(job);
    let publication = tokio::select! {
        publication = &mut job => publication,
        _ = tokio::time::sleep(master_timeout) => {
            use std::sync::atomic::Ordering;
            if phase.compare_exchange(METADATA_PENDING, METADATA_ABANDONED, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                let reason = format!("failed to process cluster event ({}) within {}ms", req.path(), master_timeout.as_millis());
                return MetadataAdmission::Finished(RestResp::error(503, "process_cluster_event_timeout_exception", &reason), Err(reason));
            }
            job.await
        }
    };
    let response = result.lock().take().unwrap_or_else(|| RestResp::error(503, "failed_to_commit_cluster_state_exception", publication.as_ref().err().map(String::as_str).unwrap_or("metadata task did not execute")));
    // Scratch execution alone is not a commit: an explicitly uninvoked
    // coordinator state task cannot have published any part of the delta.
    // The service has also invalidated the failed scratch state before reply.
    finish_scratch_attempt(&phase, response, publication, unpublished.load(std::sync::atomic::Ordering::Acquire))
}

/// Whether an engine API routed to the master can change the master's
/// engine (a kept job publishes it): not a read, nor one of the POST APIs
/// that only read (searches, queries, privilege checks, inference, ...).
fn engine_api_mutates(req: &RestReq) -> bool {
    // Kibana enrollment creates an `elastic/kibana` service account token.
    if is_kibana_enrollment(req.segments().as_slice()) {
        return true;
    }
    if matches!(req.method.as_str(), "GET" | "HEAD") {
        return false;
    }
    const READS: &[&str] = &["_search", "_query", "_has_privileges", "_infer", "_inference_query", "_explain", "_validate", "_msearch", "_count", "_field_caps", "_terms_enum", "_stats", "_status", "_preview", "_eql", "_render", "_simulate", "_analyze", "_authenticate"];
    let segs = req.segments();
    if segs.first().is_some_and(|s| s == "_eql" || s == "_profiling") {
        return false;
    }
    // `PUT|DELETE /_query/view/{name}` defines ES|QL views (cluster
    // metadata), unlike `POST /_query`, which only reads. Unpublished, a
    // view exists only on the master: a query forwarded to a full shard
    // holder, and the copy that fences index and alias names, miss it.
    if segs.first().is_some_and(|s| s == "_query") && segs.get(1).is_some_and(|s| s == "view") {
        return true;
    }
    if segs.first().is_some_and(|s| s == "_inference") && segs.len() > 2 && req.method.as_str() == "POST" {
        // POST /_inference/{task}/{id}: running a model, not defining one.
        return false;
    }
    !segs.iter().any(|s| READS.contains(&s.as_str()))
}

/// Metadata changes run on the elected master's engine, then the master
/// publishes the resulting metadata and waits for the commit.
async fn on_master(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    // `_reindex` / by-query without an index in the path move documents:
    // they run where the data is, not on the master.
    let segs = req.segments();
    if segs.first().map(|s| s == "_reindex").unwrap_or(false) {
        return reindex(node, req).await;
    }
    // ES's `TransportMasterNodeAction`: a retryable global block (state not
    // recovered, no master) or a master that is not known (or not yet
    // applied) here is waited out until `master_timeout` (default 30s)
    // instead of failing at once. A just-elected master answers forwarded
    // requests while its own applier still holds the pre-election state,
    // with the "state not recovered" block, although its followers have
    // already applied the recovered state.
    let deadline = tokio::time::Instant::now() + master_admission_limit(req);
    let mut waited = false;
    loop {
        let state = node.state();
        let refusal = if let Some(b) = state.blocks.global_blocked("metadata_write") {
            let r = RestResp::error(503, "cluster_block_exception", &format!("blocked by: [SERVICE_UNAVAILABLE/{}/{}];", b.id, b.description));
            if !b.retryable {
                return r;
            }
            r
        } else if node.is_master() {
            break;
        } else {
            match state.master_node_id.clone() {
                // A re-elected master whose applier still holds an older-term
                // state naming it: not forwarded to itself (that would loop
                // to the hop limit); retryable.
                Some(m) if m == node.local.id => RestResp::error(503, "master_not_discovered_exception", "this node was elected master and has not applied its own cluster state yet"),
                Some(m) => {
                    if !waited {
                        return forward(node, &m, req).await;
                    }
                    // After a wait the master gets what is left of
                    // `master_timeout`, not a fresh one.
                    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                    return match remaining_master_request(req, remaining) {
                        Ok(bounded) => forward(node, &m, &bounded).await,
                        Err(error) => err(error),
                    };
                }
                None => RestResp::error(503, "master_not_discovered_exception", ""),
            }
        };
        if !wait_state_change(node, deadline).await {
            return refusal;
        }
        waited = true;
    }
    // Wildcards resolve against the master's engine: wait out a peer
    // recovery that has momentarily dropped an index shell from it.
    await_mirror(node).await;
    if req.method() == Method::GET && segs.as_slice() == ["_profiling", "status"] {
        if let Err(error) = elasticsearch_http::validate_profiling_status(req.path(), &req.params()) {
            return err(error);
        }
        // Resource installation changes templates, indices and data streams.
        // Commit those through the metadata service before status searches the
        // cluster; otherwise locally created shells have no published routing
        // and has_data waits for shards which cannot be allocated.
        if !elasticsearch_engine::profiling::resources_created(&node.engine) {
            if let Err(error) = node.mutate_engine_metadata(|engine| elasticsearch_engine::profiling::ensure_resources(engine)).await {
                return err(error);
            }
        }
    }
    if req.method() == Method::POST && segs.as_slice() == ["_features", "_reset"] {
        if let Err(response) = delete_feature_indices(node, req).await {
            return response;
        }
    }
    // An engine API's own change is in engine 0 before it is published:
    // its acknowledgement never waits for a custom deferred for another API
    // in flight (see `ClusterNode::applied_here`).
    let own_engine_change = !serialized_engine_metadata(req) && engine_api_mutates(req);
    let (mut resp, publication) = if serialized_engine_metadata(req) {
        commit_engine_request(node, req).await
    } else if !engine_api_mutates(req) {
        // A read (or a long read-like call) routed to the master: no
        // snapshot, no deferral of engine 0's customs, no kept job.
        let response = master_engine_request(node, req).await;
        (response, Err("no change".to_string()))
    } else {
        // An engine API that changes the master's engine itself; the change
        // (engine 0 before vs after the API) is published by a kept job after
        // it. The API runs outside the service: it may itself wait for
        // metadata batches (a native user's security index). The API, its
        // kept job and the in-flight guard run in their own task: a client
        // that goes away does not leave the change unpublished.
        let (n, r) = (node.clone(), req.clone());
        let api = tokio::spawn(async move {
            // Only the custom family the API changes, when it changes one
            // (only that family's application is deferred meanwhile).
            let family = r.segments().first().and_then(|s| crate::meta_extra::custom_family_of_api(s));
            let in_flight = match enter_engine_api_bounded(&n, &r, family).await {
                Ok(guard) => guard,
                Err(response) => return (response, Err("engine rollback admission timed out".into())),
            };
            let (response, publication) = crate::node::IN_ENGINE_API.scope((), async {
                // Its index incarnations are held until the guard drops: an
                // application meanwhile never re-creates one the API deletes.
                let before = n.engine_api_snapshot(&in_flight).await;
                let response = master_engine_request(&n, &r).await;
                let publication = match (response.is_success(), before) {
                    (true, Some(before)) => n.publish_engine_metadata_since(before).await,
                    (true, None) => n.publish_engine_metadata().await,
                    (false, _) => Err("engine metadata request failed".into()),
                };
                (response, publication)
            }).await;
            // What was deferred while it ran (a concurrent change of its
            // custom family) is applied before it answers.
            drop(in_flight);
            n.settle_deferred_customs().await;
            (response, publication)
        });
        match api.await {
            Ok(done) => done,
            Err(e) => (RestResp::error(500, "exception", &format!("engine API task failed: {e}")), Err("engine API task failed".into())),
        }
    };
    if resp.is_success() {
        match publication {
            Ok(version) => {
                let ack_timeout = req.params().get("timeout").and_then(|t| crate::rest::parse_timeout(t)).unwrap_or(Duration::from_secs(30));
                let applied = if own_engine_change {
                    crate::node::IN_ENGINE_API.scope((), node.wait_applied_everywhere(version, ack_timeout)).await
                } else {
                    node.wait_applied_everywhere(version, ack_timeout).await
                };
                if !applied {
                    if let Some(mut v) = resp.body_json() {
                        if v.get("acknowledged").is_some() {
                            v["acknowledged"] = json!(false);
                            resp.replace_json(&v);
                        }
                    }
                }
            }
            Err(e) if e == "no change" => {}
            Err(e) if elasticsearch_cluster::coordination::CoordinationError::is_not_master(&e) => return RestResp::error(503, "not_master_exception", &e),
            Err(e) => return RestResp::error(500, "failed_to_commit_cluster_state_exception", &e),
        }
        // An API that creates indices (create, rollover, shrink / split /
        // clone) waits for their active shards (`wait_for_active_shards`,
        // default 1) and says whether they came up in time
        // (`shards_acknowledged`), as ES's `ActiveShardsObserver`.
        let created = created_indices(req, &resp);
        if !created.is_empty() {
            let timeout = req.params().get("timeout").and_then(|t| crate::rest::parse_timeout(t)).unwrap_or(Duration::from_secs(30));
            let mut all = true;
            let mut waited = false;
            for index in &created {
                if let Some(wanted) = wanted_active_shards(node, req, index) {
                    waited = true;
                    all &= wait_for_active_shards(node, index, wanted, timeout).await;
                }
            }
            if waited {
                if let Some(mut v) = resp.body_json() {
                    if v.is_object() {
                        v["shards_acknowledged"] = json!(all);
                        resp.replace_json(&v);
                    }
                }
            }
        }
        if req.method() == Method::PUT && segs.len() == 2 && segs[0] == "_data_stream" {
            write_path::wait_data_stream_primaries(node, &segs[1]).await;
        }
        // `EnrichPolicyRunner` waits for the new enrich index to be green
        // (a copy on every node) before the policy can be used.
        if segs.len() == 4 && segs[0] == "_enrich" && segs[1] == "policy" && segs[3] == "_execute" {
            wait_enrich_green(node, &segs[2], Duration::from_secs(30)).await;
        }
    }
    resp
}

/// `POST /_features/_reset` first deletes every feature's system indices
/// as a published metadata change (a dry run on the scratch copy, applied by
/// every node's applier), as ES's reset deletes them through cluster-state
/// tasks on the master. Only then does the reset run on the master's engine
/// for the feature state kept outside indices: a delete is never made in
/// engine 0 first and published after the fact (an application in between
/// could re-create the shell, and the delete would never be published).
async fn delete_feature_indices(node: &Arc<ClusterNode>, req: &RestReq) -> Result<(), RestResp> {
    let (deleted, version) = node.mutate_engine_metadata(|engine| {
        let names = elasticsearch_http::feature_system_indices(engine);
        for name in &names {
            match engine.delete_index(name) {
                Ok(()) => {}
                Err(e) if e.status == 404 => {}
                Err(e) => return Err(e),
            }
        }
        Ok(!names.is_empty())
    }).await.map_err(err)?;
    if deleted {
        // The reset then finds them gone from engine 0 too.
        let timeout = req.params().get("timeout").and_then(|t| crate::rest::parse_timeout(t)).unwrap_or(Duration::from_secs(30));
        node.wait_applied_here(version, timeout).await;
    }
    Ok(())
}

/// Run a metadata request on the master's engine. Peer recovery can
/// replace the engine's copy of an index while the request holds it
/// (`index_copy_replaced_exception`): run it again on the current copy.
async fn master_engine_request(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let resp = local_canonical(node, req).await;
        if !copy_replaced(&resp) || tokio::time::Instant::now() >= deadline {
            return resp;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

fn copy_replaced(resp: &RestResp) -> bool {
    resp.status == 503 && resp.body_json().is_some_and(|v| v["error"]["type"] == elasticsearch_engine::index::COPY_REPLACED)
}

async fn wait_enrich_green(node: &ClusterNode, policy: &str, timeout: Duration) -> bool {
    let mut rx = node.subscribe();
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if enrich_index_green(&rx.borrow(), policy) {
            return true;
        }
        if tokio::time::timeout_at(deadline, rx.changed()).await.is_err() {
            return false;
        }
    }
}

/// Every enrich index of `policy` is green: with `auto_expand_replicas:
/// 0-all`, every data node holds a started copy of each shard (a replica
/// still recovering is open but missing documents).
fn enrich_index_green(st: &elasticsearch_cluster::state::ClusterState, policy: &str) -> bool {
    let data_nodes: Vec<&String> = st.nodes.iter().filter(|(_, n)| n.is_data()).map(|(id, _)| id).collect();
    st.metadata.indices.iter().filter(|(name, _)| elasticsearch_engine::enrich::is_policy_index(policy, name)).all(|(name, md)| {
        (0..md.number_of_shards.max(1)).all(|k| st.routing_table.shard_table(name, k).is_some_and(|t| {
            t.copies.iter().all(|c| c.active() || c.node_id.is_none() && data_nodes.len() <= 1)
                && data_nodes.iter().all(|id| t.copies.iter().any(|c| c.active() && c.node_id.as_deref() == Some(id.as_str())))
        }))
    })
}

#[cfg(test)]
mod enrich_green_tests {
    use super::*;
    use elasticsearch_cluster::state::{AllocationId, ClusterState, DiscoveryNode, IndexMetadata, IndexRoutingTable, IndexShardRoutingTable, IndexState, Role, ShardRouting, ShardState};
    use std::collections::BTreeMap;

    fn copy(primary: bool, state: ShardState, node: &str) -> ShardRouting {
        ShardRouting { index: ".enrich-p-1".into(), shard: 0, primary, state, node_id: Some(node.into()), relocating_node_id: None,
            allocation_id: Some(AllocationId { id: format!("a-{node}"), relocation_id: None }), recovery_source: None, unassigned_info: None, expected_shard_size_in_bytes: None }
    }

    fn cluster(copies: Vec<ShardRouting>) -> ClusterState {
        let mut s = ClusterState::empty("test");
        for n in ["n1", "n2"] {
            s.nodes.insert(n.into(), DiscoveryNode { id: n.into(), name: n.into(), ephemeral_id: "e".into(), transport_address: "127.0.0.1:9300".into(),
                http_address: "127.0.0.1:9200".into(), host: "h".into(), ip: "127.0.0.1".into(), roles: Role::defaults(), attributes: BTreeMap::new(), version: "9.5.5".into() });
        }
        s.metadata.indices.insert(".enrich-p-1".into(), IndexMetadata { name: ".enrich-p-1".into(), uuid: "u".into(), version: 1, mapping_version: 1,
            settings_version: 1, aliases_version: 1, state: IndexState::Open, settings: Default::default(), mappings: json!({}).into(), aliases: BTreeMap::new(),
            number_of_shards: 1, number_of_replicas: 0, routing_num_shards: 1, primary_terms: vec![1], in_sync_allocation_ids: BTreeMap::new(), creation_date: 0, lifecycle: None });
        s.routing_table.indices.insert(".enrich-p-1".into(), IndexRoutingTable { index_uuid: "u".into(), shards: vec![IndexShardRoutingTable { shard: 0, copies }] });
        s
    }

    #[test]
    fn green_needs_a_started_copy_on_every_data_node() {
        // Just created: only the primary is listed (the auto-expanded
        // replica is added by a later reroute).
        assert!(!enrich_index_green(&cluster(vec![copy(true, ShardState::Started, "n1")]), "p"));
        // The replica is recovering: open on n2, but still missing documents.
        assert!(!enrich_index_green(&cluster(vec![copy(true, ShardState::Started, "n1"), copy(false, ShardState::Initializing, "n2")]), "p"));
        assert!(enrich_index_green(&cluster(vec![copy(true, ShardState::Started, "n1"), copy(false, ShardState::Started, "n2")]), "p"));
        // Other policies' indices are not waited for.
        assert!(enrich_index_green(&cluster(vec![copy(true, ShardState::Started, "n1")]), "other"));
    }
}

/// The indices a successful index-creating request made: `PUT /{index}`,
/// a rollover's new index, a resize's target.
fn created_indices(req: &RestReq, resp: &RestResp) -> Vec<String> {
    if !resp.is_success() {
        return Vec::new();
    }
    let segs = req.segments();
    match (req.method.as_str(), segs.len()) {
        // The created name from the response (date math resolved).
        ("PUT", 1) if !segs[0].starts_with('_') => vec![resp.body_json().and_then(|v| v["index"].as_str().map(str::to_string)).unwrap_or_else(|| segs[0].clone())],
        ("POST", 2 | 3) if segs[1] == "_rollover" => resp
            .body_json()
            .filter(|v| v["rolled_over"].as_bool() == Some(true) && v["dry_run"].as_bool() != Some(true))
            .and_then(|v| v["new_index"].as_str().map(str::to_string))
            .into_iter()
            .collect(),
        ("PUT" | "POST", 3) if matches!(segs[1].as_str(), "_shrink" | "_split" | "_clone") => {
            vec![resp.body_json().and_then(|v| v["index"].as_str().map(str::to_string)).unwrap_or_else(|| segs[2].clone())]
        }
        _ => Vec::new(),
    }
}

/// ES `wait_for_active_shards` of a request (`None`: do not wait).
#[derive(Clone, Copy)]
enum ActiveShards {
    All,
    Count(usize),
}

/// `wait_for_active_shards` of the request for `index`: the parameter, else
/// the index's `index.write.wait_for_active_shards` (default 1), as ES.
fn wanted_active_shards(node: &ClusterNode, req: &RestReq, index: &str) -> Option<ActiveShards> {
    let parse = |v: &str| match v {
        "" => Some(ActiveShards::Count(1)),
        "all" => Some(ActiveShards::All),
        n => match n.parse::<usize>() {
            Ok(0) => None,
            Ok(n) => Some(ActiveShards::Count(n)),
            Err(_) => Some(ActiveShards::Count(1)),
        },
    };
    if let Some(v) = req.params().get("wait_for_active_shards") {
        return parse(v);
    }
    let st = node.state();
    match st.metadata.indices.get(index).and_then(|im| im.setting("index.write.wait_for_active_shards").map(str::to_string)) {
        Some(v) => parse(&v),
        None => Some(ActiveShards::Count(1)),
    }
}

/// Wait until every shard of `index` has the wanted active copies.
async fn wait_for_active_shards(node: &ClusterNode, index: &str, wanted: ActiveShards, timeout: Duration) -> bool {
    let mut rx = node.subscribe();
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let ok = {
            let st = rx.borrow();
            match st.metadata.indices.get(index) {
                None => false,
                Some(m) => (0..m.number_of_shards.max(1)).all(|k| {
                    st.routing_table.shard_table(index, k).is_some_and(|t| match wanted {
                        ActiveShards::Count(1) => t.one_shard_active(&st.nodes),
                        ActiveShards::Count(n) => t.copies.iter().filter(|c| c.active()).count() >= n,
                        ActiveShards::All => t.copies.iter().all(|c| c.active()) && t.copies.len() > m.number_of_replicas as usize,
                    })
                }),
            }
        };
        if ok {
            return true;
        }
        if tokio::time::timeout_at(deadline, rx.changed()).await.is_err() {
            return false;
        }
    }
}

async fn wait_for_primary(node: &ClusterNode, index: &str, timeout: Duration) -> bool {
    let mut rx = node.subscribe();
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let ok = {
            let st = rx.borrow();
            let n = st.metadata.indices.get(index).map(|m| m.number_of_shards).unwrap_or(1);
            // `wait_for_active_shards` 1 (ES: a search shard on stateless).
            (0..n).all(|k| st.routing_table.shard_table(index, k).is_some_and(|t| t.one_shard_active(&st.nodes)))
        };
        if ok {
            return true;
        }
        if tokio::time::timeout_at(deadline, rx.changed()).await.is_err() {
            return false;
        }
    }
}

/// Resolve a write target to its concrete index, auto-creating through the
/// master when needed.
fn bulk_resolution_failure(response:&RestResp)->bulk::PartResult {
    let error=response.body_json().map(|mut value|value.get_mut("error").map(std::mem::take).unwrap_or(Value::Null)).unwrap_or(Value::Null);
    bulk::PartResult::Failed{status:response.status,error}
}

async fn concrete_write_index(node: &Arc<ClusterNode>, name: &str, auto_create: bool) -> Result<SharedText, RestResp> {
    // A write can arrive on a node while peer recovery has temporarily
    // removed a local index shell that is still present in the applied
    // cluster state. Resolve after the mirror catches up so a real data
    // stream is not mistaken for an auto-create candidate (or a missing
    // backing index).
    await_mirror(node).await;
    let engine = node.engine.clone();
    let n = SharedText::try_new(name).map_err(err)?;
    let local = tokio::task::spawn_blocking(move || engine.write_index(&n, false).and_then(|s|SharedText::try_new(&s.name))).await.ok();
    match local {
        Some(Ok(c)) => Ok(c),
        Some(Err(e)) if e.kind != "index_not_found_exception" => Err(err(e)),
        _ if !auto_create => {
            let reason = format!("no such index [{name}] and [require_alias] request flag is [true] and [{name}] is not an alias");
            let cause = json!({"type": "index_not_found_exception", "reason": reason, "resource.type": "index_expression", "resource.id": name, "index_uuid": "_na_", "index": name});
            let mut error = cause.clone();
            error["root_cause"] = json!([cause]);
            Err(RestResp::json(404, &json!({"error": error, "status": 404})))
        }
        _ => {
            let r: Result<SharedText, RemoteError> = node.to_master(ACTION_AUTO_CREATE, &name, Duration::from_secs(30)).await;
            match r {
                Ok(c) => {
                    wait_for_primary(node, &c, Duration::from_secs(60)).await;
                    // Auto-creation publishes the backing index and data
                    // stream together. A different coordinator can observe
                    // the active shard before its local data-stream mirror is
                    // applied, so wait before deciding whether this is an
                    // ordinary index or a stream write.
                    await_mirror(node).await;
                    Ok(c)
                }
                Err(e) => Err(RestResp::from_remote(&e)),
            }
        }
    }
}

fn primary_node(node: &ClusterNode, index: &str) -> Option<String> {
    primary_node_of(node, index, 0)
}

fn primary_node_of(node: &ClusterNode, index: &str, shard: u32) -> Option<String> {
    node.state().routing_table.shard_table(index, shard).and_then(|t| t.active_primary()).and_then(|p| p.node_id.clone())
}

fn unavailable(index: &str) -> RestResp {
    unavailable_shard(index, 0)
}

fn unavailable_shard(index: &str, shard: u32) -> RestResp {
    RestResp::error(503, "unavailable_shards_exception", &format!("[{index}][{shard}] primary shard is not active Timeout: [1m]"))
}

/// Errors that prove the request was not executed, so it can be retried
/// once the cluster state changes (ES retries these until `timeout`).
fn retryable(r: &RestResp) -> bool {
    if r.is_success() {
        return false;
    }
    let v = r.body_json().unwrap_or(Value::Null);
    let ty = v["error"]["type"].as_str().unwrap_or_default().to_string();
    if ty == "cluster_block_exception" {
        // Retryable blocks: no master (2) and state not recovered (1).
        let reason = v["error"]["reason"].as_str().unwrap_or_default();
        return reason.contains("/2/no master") || reason.contains("/1/state not recovered");
    }
    matches!(
        ty.as_str(),
        "connect_transport_exception" | "node_not_connected_exception" | "node_disconnected_exception" | "transport_exception" | "receive_timeout_transport_exception" | "unavailable_shards_exception" | "retry_on_primary_exception" | "node_closed_exception" | "shard_not_found_exception" | "no_shard_available_action_exception" | "master_not_discovered_exception" | "not_master_exception" | "index_shard_closed_exception"
    )
}

/// The request's `timeout` (default 1m), as ES's replication actions use it.
fn request_timeout(req: &RestReq) -> Duration {
    req.params().get("timeout").and_then(|t| crate::rest::parse_timeout(t)).unwrap_or(Duration::from_secs(60))
}

/// Wait for the next applied cluster state (or a short poll interval).
async fn wait_state_change(node: &ClusterNode, deadline: tokio::time::Instant) -> bool {
    let mut rx = node.subscribe();
    let until = deadline.min(tokio::time::Instant::now() + Duration::from_millis(500));
    let _ = tokio::time::timeout_at(until, rx.changed()).await;
    tokio::time::Instant::now() < deadline
}

/// Retryable errors after which the request may nevertheless have executed
/// on the primary (the connection broke or timed out after it was sent, or
/// the primary executed it but could not complete replication: see
/// [`not_acked`]).
fn is_indeterminate(r: &RestResp) -> bool {
    if r.is_success() {
        return false;
    }
    let v = r.body_json().unwrap_or(Value::Null);
    matches!(
        v["error"]["type"].as_str().unwrap_or_default(),
        "node_disconnected_exception" | "transport_exception" | "receive_timeout_transport_exception" | "retry_on_primary_exception" | "node_closed_exception"
    )
}

/// Executed on the primary but not acknowledged (replication could not be
/// completed, or this copy stopped being the primary meanwhile). ES answers
/// this with a `RetryOnPrimaryException` that the coordinator's reroute
/// phase retries on the next cluster state until `timeout`; it is never a
/// 2xx. Retryable and indeterminate here: the coordinator's write loop
/// re-runs the request on the (possibly new) primary and acknowledges only
/// a fully replicated attempt, while a retry answer that may reflect this
/// attempt's own effect (`own_echo`) is never reported as a definite
/// failure. Status 500, as ES's `RetryOnPrimaryException` without a cause.
fn not_acked(index: &str, shard: u32, why: &str) -> RestResp {
    RestResp::error(500, "retry_on_primary_exception", &format!("[{index}][{shard}] {why}"))
}

/// A document id as ES auto-generates them: 20 URL-safe characters
/// (15 bytes: 6 of millisecond time, 9 random; base64url without padding).
fn new_doc_id() -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let millis = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
    let mut bytes = [0u8; 15];
    bytes[..6].copy_from_slice(&millis.to_be_bytes()[2..]);
    let random: [u8; 9] = rand::random();
    bytes[6..].copy_from_slice(&random);
    let mut out = String::with_capacity(20);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16 | (c[1] as u32) << 8 | c[2] as u32;
        for shift in [18, 12, 6, 0] {
            out.push(ALPHABET[(n >> shift) as usize & 63] as char);
        }
    }
    out
}

/// A single-document write with create semantics.
struct CreateOp {
    /// The id as it appears in the request path (still percent-encoded).
    raw_id: String,
    /// The coordinator generated the id (the client sent an auto-id index).
    generated: bool,
}

/// Recognise a create and give an auto-id index its id here, once, so every
/// retry carries the same id (ES generates it on the coordinator and its
/// REST layer makes an auto-id index a create). Returns the request to send.
fn prepare_write(req: &RestReq, new_id: impl FnOnce() -> String) -> elasticsearch_engine::error::Result<(Option<RestReq>, Option<CreateOp>)> {
    let raw: Vec<&str> = req.path().split('/').filter(|s| !s.is_empty()).collect();
    let params = req.params();
    let op_type = params.get("op_type").map(String::as_str);
    Ok(match (req.method.as_str(), raw.as_slice()) {
        ("POST", [_, "_doc"]) if matches!(op_type, None | Some("create")) && !["if_seq_no", "if_primary_term", "version", "version_type"].iter().any(|k| params.contains_key(*k)) => {
            let id = new_id();
            let mut rewritten=req.clone();
            rewritten.set_uri_fmt(format_args!("{}/{id}?{}{}{}",req.path().trim_end_matches('/'),req.query(),if op_type.is_none()&&!req.query().is_empty(){"&"}else{""},if op_type.is_none(){"op_type=create"}else{""}))?;
            (Some(rewritten),Some(CreateOp {raw_id:id,generated:true}))
        }
        ("PUT" | "POST", [_, "_create", id]) => (None, Some(CreateOp { raw_id: id.to_string(), generated: false })),
        ("PUT" | "POST", [_, "_doc", id]) if op_type == Some("create") => (None, Some(CreateOp { raw_id: id.to_string(), generated: false })),
        _ => (None, None),
    })
}

/// A create answered 409 on a retry: decide whether the document there is
/// the one this very request wrote on an earlier attempt whose response was
/// lost, and if so return the success that attempt would have returned.
///
/// For a coordinator-generated id the document's existence is the proof
/// (nobody else knows the id). For a client-supplied id (`expect_source`)
/// it is a heuristic: the document must be at version 1 with exactly the
/// source this request sent.
async fn confirm_created(node: &Arc<ClusterNode>, conflict: &Value, index: &str, raw_id: &str, routing: Option<&str>, expect_source: Option<&Value>, deadline: tokio::time::Instant) -> elasticsearch_engine::error::Result<Option<Value>> {
    let index = conflict["index"].as_str().unwrap_or(index);
    let Some((doc, at)) = primary_doc(node, index, raw_id, routing).await? else { return Ok(None) };
    if doc["found"] != json!(true) {
        return Ok(None);
    }
    if let Some(src) = expect_source {
        if doc["_version"] != json!(1) || &doc["_source"] != src {
            return Ok(None);
        }
    }
    if !acked_on_primary(node, &doc, &at, deadline).await {
        return Ok(None);
    }
    Ok(Some(write_result(node, &doc, index, "created")))
}

/// The primary a confirming read was served by: its node, the shard's
/// primary term in this node's state at the time, and the shard.
#[derive(Clone, Debug, PartialEq)]
struct PrimaryAt {
    index: String,
    shard: u32,
    node: String,
    term: u64,
}

/// The GET preference that reads shard `shard` of `index` on its active
/// primary in `st` (`local` is this node's id), with that primary.
fn primary_read(st: &elasticsearch_cluster::state::ClusterState, local: &str, index: &str, shard: u32) -> Option<(String, PrimaryAt)> {
    let node = st.routing_table.shard_table(index, shard)?.active_primary()?.node_id.clone()?;
    let term = st.metadata.indices.get(index)?.primary_term(shard);
    let preference = if node == local { "_only_local".to_string() } else { format!("_only_nodes:{node}") };
    Some((preference, PrimaryAt { index: index.to_string(), shard, node, term }))
}

/// Read back the document a retried write may have left, on the shard's
/// primary: never on another copy, which can be one the master already
/// failed, holding an operation of an earlier primary term that the
/// promoted primary never received (flapping seed 1: n4's own
/// failed copy showed its CAS at seq 4337 term 1, which the term-2 history
/// does not contain, and n4 acknowledged it). A first read on any copy
/// finds the concrete index and shard (`_index`, `_id`, `_routing`).
async fn primary_doc(node: &Arc<ClusterNode>, index: &str, raw_id: &str, routing: Option<&str>) -> elasticsearch_engine::error::Result<Option<(Value, PrimaryAt)>> {
    let route = |r: Option<&str>, sep: char| r.map(|r| format!("{sep}routing={r}")).unwrap_or_default();
    let probe = RestReq::try_new_uri("GET", format_args!("/{index}/_doc/{raw_id}{}", route(routing, '?')), &[], &[], 0, Dfs::None, None)?;
    let Some(found) = read(node, &probe, index).await.body_json() else { return Ok(None) };
    let (Some(name), Some(id)) = (found["_index"].as_str(), found["_id"].as_str()) else { return Ok(None) };
    // The shard needs the routing value itself; the probe reports it only
    // for a found document.
    let doc_routing = match (routing, found["_routing"].as_str()) {
        (None, _) => None,
        (Some(_), Some(r)) => Some(r.to_string()),
        (Some(_), None) => return Ok(None),
    };
    let st = node.state();
    let Ok(shard) = crate::shard_engines::doc_shard(&st.metadata, name, id, doc_routing.as_deref()) else { return Ok(None) };
    let Some((preference, at)) = primary_read(&st, &node.local.id, name, shard) else { return Ok(None) };
    let get = RestReq::try_new_uri("GET", format_args!("/{}/_doc/{raw_id}?preference={}{}", encode_component(name), encode_component(&preference), route(routing, '&')), &[], &[], 0, Dfs::None, None)?;
    let r = read(node, &get, name).await;
    if !r.is_success() && r.status != 404 {
        return Ok(None);
    }
    Ok(r.body_json().map(|doc| (doc, at)))
}

/// Whether the operation that left `doc`, read on the primary `at`, is
/// acknowledged: the primary itself runs what its write path runs before an
/// ack (`replicate_through` the document's seq_no: every in-sync copy holds
/// it, or was failed through the master, an unreachable one at once) and is
/// still that primary afterwards. That is what an acknowledgement of the
/// lost attempt would have promised; its term does not matter (an operation
/// of an earlier term that the promoted primary holds is part of the history
/// it resyncs). The primary must still be `at` (same node, same term).
/// Retried until `deadline`; otherwise the outcome stays unknown.
async fn acked_on_primary(node: &Arc<ClusterNode>, doc: &Value, at: &PrimaryAt, deadline: tokio::time::Instant) -> bool {
    let Some(seq_no) = doc["_seq_no"].as_i64() else { return false };
    loop {
        let reply = ack_through_on_primary(node, at, seq_no).await;
        match acked_decision(at, &reply) {
            Durability::Durable => return true,
            Durability::NotCurrent => return false,
            Durability::NotYet => {}
        }
        if tokio::time::Instant::now() + Duration::from_millis(100) >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[derive(Debug, PartialEq)]
enum Durability {
    /// Acknowledged by the primary that served the read.
    Durable,
    /// That node no longer holds the shard's primary in that term: unknown.
    NotCurrent,
    /// Not acknowledged yet (replication failed, or no answer): retry.
    NotYet,
}

/// The primary's answer to [`ACK_THROUGH_HEADER`].
#[derive(Debug, PartialEq)]
enum AckReply {
    /// Acknowledged through the seq_no by the primary of this term.
    Acked { term: u64 },
    /// The node holds no primary context of the shard, or one of this term.
    NotPrimary { term: Option<u64> },
    /// Replication or transport failed.
    Failed,
}

/// The decision of [`acked_on_primary`] on the primary's answer.
fn acked_decision(at: &PrimaryAt, reply: &AckReply) -> Durability {
    match reply {
        AckReply::Acked { term } if *term == at.term => Durability::Durable,
        AckReply::Acked { .. } | AckReply::NotPrimary { .. } => Durability::NotCurrent,
        AckReply::Failed => Durability::NotYet,
    }
}

/// Internal header on a forwarded `/{index}/_fleet/global_checkpoints`
/// (`EXEC_LOCAL`): `<shard>:<seq_no>:<primary term>` asks that node's primary
/// context of the shard, if of that term, to acknowledge through `seq_no` as
/// its write path does (see [`acked_on_primary`]).
const ACK_THROUGH_HEADER: &str = "x-es-cluster-ack-through";

async fn ack_through_on_primary(node: &Arc<ClusterNode>, at: &PrimaryAt, seq_no: i64) -> AckReply {
    if at.node == node.local.id {
        return local_ack_through(node, &at.index, at.shard, seq_no, at.term).await;
    }
    let Ok(mut sub) = RestReq::try_new_uri("GET", format_args!("/{}/_fleet/global_checkpoints", encode_component(&at.index)), &[], &[], 0, Dfs::None, None) else { return AckReply::Failed };
    if sub.push_header(EXEC_LOCAL, "1").is_err() || sub.push_header(ACK_THROUGH_HEADER, &format!("{}:{seq_no}:{}", at.shard, at.term)).is_err() {
        return AckReply::Failed;
    }
    let r = forward(node, &at.node, &sub).await;
    let v = r.body_json().unwrap_or(Value::Null);
    match (r.status, v["ack_through"].as_str()) {
        (200, Some("acked")) => v["primary_term"].as_u64().map_or(AckReply::Failed, |term| AckReply::Acked { term }),
        (200, Some("not_primary")) => AckReply::NotPrimary { term: v["primary_term"].as_u64() },
        // An older node answers the plain checkpoints request: no proof.
        _ => AckReply::Failed,
    }
}

/// [`ACK_THROUGH_HEADER`] on the node holding the primary.
async fn local_ack_through(node: &Arc<ClusterNode>, index: &str, shard: u32, seq_no: i64, term: u64) -> AckReply {
    let Some(p) = node.shards.primary(index, shard) else { return AckReply::NotPrimary { term: None } };
    if p.term != term {
        return AckReply::NotPrimary { term: Some(p.term) };
    }
    // As a write: under a gate permit (flush, recovery start and snapshot
    // freezes take the gate exclusively).
    let _permit = p.gate.read().await;
    if p.replicate_through(seq_no, &TransportNet(node.clone())).await.is_err() {
        return AckReply::Failed;
    }
    if !still_primary(node, &p) {
        return AckReply::NotPrimary { term: None };
    }
    AckReply::Acked { term: p.term }
}

fn ack_reply_json(reply: &AckReply) -> RestResp {
    match reply {
        AckReply::Acked { term } => RestResp::json(200, &json!({"ack_through": "acked", "primary_term": term})),
        AckReply::NotPrimary { term } => RestResp::json(200, &json!({"ack_through": "not_primary", "primary_term": term})),
        AckReply::Failed => RestResp::error(503, "unavailable_shards_exception", "the primary could not acknowledge the operation"),
    }
}

/// The response of the write that left `doc` (a GET response) in place.
fn write_result(node: &ClusterNode, doc: &Value, index: &str, result: &str) -> Value {
    let st = node.state();
    let name = doc["_index"].as_str().unwrap_or(index);
    let shard = doc["_id"].as_str().and_then(|id| crate::shard_engines::doc_shard(&st.metadata, name, id, doc["_routing"].as_str()).ok()).unwrap_or(0);
    let table = st.routing_table.shard_table(name, shard);
    let total = table.map(|t| t.copies.len()).unwrap_or(1);
    let successful = table.map(|t| t.active_copies().count()).unwrap_or(1);
    json!({
        "_index": doc["_index"],
        "_id": doc["_id"],
        "_version": doc["_version"],
        "result": result,
        "_shards": {"total": total, "successful": successful, "failed": 0},
        "_seq_no": doc["_seq_no"],
        "_primary_term": doc["_primary_term"],
    })
}

/// After an attempt that may have executed, an answer to the retry that the
/// earlier attempt itself can have caused: a version conflict (a create or
/// a conditional write now conflicts with its own first application) or a
/// delete finding nothing (the first attempt deleted the document). Such an
/// answer is not a definite outcome of the request: unless the first
/// attempt's effect can be confirmed, the client gets the lost attempt's
/// error, which says the outcome is unknown, and never a definite failure
/// for a write that was applied. (ES retries the same way but reports the
/// retry's answer as is; this is deliberately stricter.)
fn own_echo(delete: bool, status: u16, body: &Value) -> bool {
    status == 409 || (delete && status == 404 && body["result"] == "not_found")
}

/// The condition a conditional index request carried.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Condition {
    if_seq_no: Option<i64>,
    version: Option<i64>,
}

/// Whether the document now stored (`doc`, a GET response) is what a
/// conditional index request with this `source` wrote: same source, and a
/// seq_no above the one it required (`if_seq_no`) or exactly its external
/// `version`. A heuristic, as for creates with a client-supplied id: another
/// request sending the identical source under the same condition is
/// indistinguishable (and then equally "this" write).
fn is_own_write(doc: &Value, source: &Value, cond: Condition) -> bool {
    if doc["found"] != json!(true) || &doc["_source"] != source {
        return false;
    }
    match (cond.if_seq_no, cond.version) {
        (Some(required), _) => doc["_seq_no"].as_i64().is_some_and(|s| s > required),
        (None, Some(v)) => doc["_version"].as_i64() == Some(v),
        (None, None) => false,
    }
}

/// A conditional index answered 409 on a retry: if the stored document is
/// the one this request wrote on its lost attempt, the success to return.
async fn confirm_written(node: &Arc<ClusterNode>, conflict: &Value, index: &str, raw_id: &str, routing: Option<&str>, source: &Value, cond: Condition, deadline: tokio::time::Instant) -> elasticsearch_engine::error::Result<Option<Value>> {
    let index = conflict["index"].as_str().unwrap_or(index);
    let Some((doc, at)) = primary_doc(node, index, raw_id, routing).await? else { return Ok(None) };
    if !is_own_write(&doc, source, cond) || !acked_on_primary(node, &doc, &at, deadline).await {
        return Ok(None);
    }
    let result = if doc["_version"] == json!(1) { "created" } else { "updated" };
    Ok(Some(write_result(node, &doc, index, result)))
}

/// The raw id of an index request (`PUT|POST /{index}/_doc/{id}`).
fn indexed_id(req: &RestReq) -> Option<String> {
    let raw: Vec<&str> = req.path().split('/').filter(|s| !s.is_empty()).collect();
    match (req.method.as_str(), raw.as_slice()) {
        ("PUT" | "POST", [_, "_doc", id]) => Some(id.to_string()),
        _ => None,
    }
}

async fn write(node: &Arc<ClusterNode>, req: &RestReq, index: &str) -> RestResp {
    let mut stable = req.clone();
    if let Err(e) = failure_store::prepare_single(&mut stable) { return err(e); }
    let req = &stable;
    // Validate before either id preparation or routing can auto-create an index.
    if let Err(e) = node.engine.require_data_stream(index, req.params().get("require_data_stream").is_some_and(|v| v == "true")) {
        return RestResp::json(e.status, &e.to_json_with_trace(false));
    }
    let deadline = tokio::time::Instant::now() + request_timeout(req);
    // Time series indices generate the `_id` on the primary (from the
    // document's dimensions and timestamp): no coordinator id.
    // ES's id layouts (as for bulk items): k-ordered for logsdb / columnar.
    let target_mode = node.engine.get_index(index).or_else(|| node.engine.data_stream(index).and_then(|d| node.engine.get_index(d.write_index()))).and_then(|svc| svc.setting("index.mode"));
    let (mut rewritten,mut create)=match prepare_write(req,|| elasticsearch_engine::index::auto_id_for_mode(target_mode.as_deref())){Ok(value)=>value,Err(error)=>return err(error)};
    if rewritten.is_some() {
        let auto_create = req.params().get("require_alias").map(|v| v != "true").unwrap_or(true);
        if let Ok(c) = concrete_write_index(node, index, auto_create).await {
            if crate::shard_engines::is_time_series(&node.state().metadata, &c) {
                (rewritten, create) = (None, None);
            }
        }
    }
    let req = rewritten.as_ref().unwrap_or(req);
    // An earlier attempt may have executed although its response was lost.
    let mut lost: Option<RestResp> = None;
    // The data stream write index the first attempt resolved: a retry after
    // a rollover writes there again, not a second copy of the document into
    // the new write index (as bulk resolves an item once).
    let mut pinned: Option<SharedText> = None;
    loop {
        let r = write_once(node, req, index, &mut pinned).await;
        if let Some(lost) = lost.as_ref().filter(|_| own_echo(req.method == "DELETE", r.status, &r.body_json().unwrap_or(Value::Null))) {
            let conflict = r.body_json().map(|v| v["error"].clone()).unwrap_or(Value::Null);
            let params = req.params();
            let routing = params.get("routing").map(|r| encode_component(r));
            if let Some(op) = &create {
                let source = if op.generated { None } else { Some(req.json().unwrap_or(Value::Null)) };
                if let Some(created) = match confirm_created(node, &conflict, index, &op.raw_id, routing.as_deref(), source.as_ref(), deadline).await {Ok(value)=>value,Err(error)=>return err(error)} {
                    tracing::info!(index, id = %op.raw_id, "retried create already applied by an earlier attempt: returning its result");
                    return RestResp::json(201, &created);
                }
            } else if let (Some(id), Some(source)) = (indexed_id(req), req.json()) {
                let cond = Condition { if_seq_no: params.get("if_seq_no").and_then(|v| v.parse().ok()), version: params.get("version").and_then(|v| v.parse().ok()) };
                if let Some(written) = match confirm_written(node, &conflict, index, &id, routing.as_deref(), &source, cond, deadline).await {Ok(value)=>value,Err(error)=>return err(error)} {
                    tracing::info!(index, id = %id, "retried conditional write already applied by an earlier attempt: returning its result");
                    let status = if written["result"] == "created" { 201 } else { 200 };
                    return RestResp::json(status, &written);
                }
            }
            // The earlier attempt may be what this answer reflects and
            // that cannot be told: the outcome is unknown, not a failure.
            tracing::info!(index, status = r.status, "retry answer may reflect the request's own earlier attempt: reporting the outcome as unknown");
            return lost.clone();
        }
        if !retryable(&r) {
            return r;
        }
        if is_indeterminate(&r) {
            lost = Some(r.clone());
        }
        if !wait_state_change(node, deadline).await {
            return r;
        }
    }
}

async fn write_once(node: &Arc<ClusterNode>, req: &RestReq, index: &str, pinned: &mut Option<SharedText>) -> RestResp {
    if let Some(b) = node.state().blocks.global_blocked("write") {
        return RestResp::error(503, "cluster_block_exception", &format!("blocked by: [SERVICE_UNAVAILABLE/{}/{}];", b.id, b.description));
    }
    let create = req.segments().get(1).map(String::as_str) == Some("_create") || req.params().get("op_type").map(String::as_str) == Some("create") || (req.method == "POST" && req.segments().len() == 2);
    let concrete = match pinned.clone() {
        // The pinned backing index was deleted since: ES's immediate 404,
        // not a wait for a primary that will never come.
        Some(c) if !node.state().metadata.indices.contains_key(c.as_str()) => {
            let e = elasticsearch_engine::error::EsError::index_not_found(&c);
            return RestResp::json(e.status, &e.to_json_with_trace(false));
        }
        Some(c) => c,
        None => {
            let auto_create = req.params().get("require_alias").map(|v| v != "true").unwrap_or(true);
            let concrete = match concrete_write_index(node, index, auto_create).await {
                Ok(c) => c,
                Err(r) => return r,
            };
            // A time series data stream writes into the backing index covering the
            // document's `@timestamp`.
            let covering = match (req.method.as_str(), std::str::from_utf8(&req.body)) {
                ("POST" | "PUT", Ok(src)) => match covering_index(node,index,src,&elasticsearch_engine::datastream::TimestampScan::of(src)).await{Ok(value)=>value,Err(response)=>return response},
                _ => None,
            };
            if let (None, "POST" | "PUT", Ok(src)) = (&covering, req.method.as_str(), std::str::from_utf8(&req.body)) {
                if let Err(e) = data_stream_write_check(node, index, create, src, &elasticsearch_engine::datastream::TimestampScan::of(src)) {
                    return failure_store::timestamp_failure(node, req, index, e).await;
                }
            }
            covering.unwrap_or(concrete)
        }
    };
    // A data stream's write, by its name or a data stream alias: the
    // resolved index is one of its backing indices.
    let stream = if concrete != index { node.engine.data_stream_of_index(&concrete) } else { None };
    let data_stream = stream.is_some();
    if data_stream && matches!(req.method.as_str(), "POST" | "PUT") {
        *pinned = Some(concrete.clone());
    }
    let segs = req.segments();
    let op = match (req.method.as_str(), segs.get(1).map(String::as_str)) {
        (_, Some("_update")) => "update",
        ("DELETE", _) => "delete",
        _ => "index",
    };
    if let Some(e) = crate::shard_engines::tsdb_doc_error(&node.state().metadata, &concrete, op, req.params().get("routing").map(String::as_str)) {
        return RestResp::json(e.status, &e.to_json_with_trace(false));
    }
    let shard = match crate::shard_engines::doc_id_of(&segs) {
        Some(id) => match crate::shard_engines::doc_shard(&node.state().metadata, &concrete, id, req.params().get("routing").map(String::as_str)) {
            Ok(k) => k,
            Err(_) => return crate::shard_engines::routing_missing(&concrete, id),
        },
        // An id-less time series write: routed by its dimensions.
        None if crate::shard_engines::shard_of(&req.headers).is_none() && crate::shard_engines::is_time_series(&node.state().metadata, &concrete) && matches!(req.method.as_str(), "POST" | "PUT") => {
            let src = req.json().unwrap_or(Value::Null);
            crate::shard_engines::tsdb_source_shard(&node.state().metadata, &concrete, &src)
        }
        // An internal per-shard sub-request (multi-shard by-query fan-out
        // in `write_path::by_query`); client requests have it stripped.
        None => crate::shard_engines::shard_of(&req.headers).unwrap_or(0),
    };
    // A data stream write goes to the backing index this coordinator
    // resolved, by name, once it has run the data stream's own checks: the
    // primary's node never resolves the write index again (it may have
    // rolled over meanwhile), and shard engines past the first hold no data
    // streams.
    let mut sreq = match (data_stream, req.method.as_str()) {
        (true, "POST" | "PUT") => {
            if let Err(e) = node.engine.check_data_stream_write_only(stream.as_deref().unwrap_or(index), create, std::str::from_utf8(&req.body).ok()) {
                return RestResp::json(e.status, &e.to_json_with_trace(false));
            }
            match retarget_write(req, index, &concrete) {
                Ok(Some(mut retargeted)) => {
                    if let Err(error)=retargeted.push_header(DATA_STREAM_BACKING_WRITE_HEADER,"true"){return err(error)};
                    retargeted
                }
                Ok(None) => req.clone(),
                Err(error)=>return err(error),
            }
        }
        _ => req.clone(),
    };
    if let Err(error)=sreq.set_shard(shard){return err(error)};
    match node.shards.primary(&concrete, shard) {
        Some(p) => execute_write(node, &p, &sreq).await,
        None => match primary_node_of(node, &concrete, shard) {
            Some(n) if n != node.local.id => forward(node, &n, &sreq).await,
            _ => unavailable_shard(&concrete, shard),
        },
    }
}

/// Execute a write on the local primary and replicate it.
pub async fn execute_write(node: &Arc<ClusterNode>, p: &Arc<PrimaryCtx>, req: &RestReq) -> RestResp {
    if let Err(reason) = crate::internal_capabilities::require_local_writer(&node.state(), &node.local) {
        return RestResp::error(503, "unsupported_operation_exception", &reason);
    }
    let _permit = p.gate.read().await;
    if p.relocated.load(std::sync::atomic::Ordering::SeqCst) {
        drop(_permit);
        return reroute_after_relocation(node, p, req).await;
    }
    // The REST layer resolves the index by name: it must be this primary's
    // engine copy, or the op would be sequenced by some other instance
    // (a wiped/recreated shell) and acknowledged under this primary.
    let Some(engine) = node.engine_for(p.shard) else {
        drop(_permit);
        return unavailable_shard(&p.index, p.shard);
    };
    if !engine_is(&engine, p) {
        drop(_permit);
        fail_orphaned_primary(node, p);
        return unavailable_shard(&p.index, p.shard);
    }
    let mut req = req.clone();
    if let Err(e) = req.set_shard(p.shard) { return err(e); }
    let req = &req;
    let before = p.max_seq_no_async().await;
    let mut resp = crate::rest::tasks::as_child(node, p.shard, req).await;
    // The trusted failure-store writer already acknowledged the destination
    // primary and replicas. Its sequence number belongs to that other index.
    let redirected = resp.is_success() && resp.headers.iter().any(|(name, value)| name.eq_ignore_ascii_case(failure_store::ACK_HEADER) && value == "true");
    resp.retain_headers(|(name, _)| !name.eq_ignore_ascii_case(failure_store::ACK_HEADER));
    if redirected { return resp; }
    let seq = p.max_seq_no_async().await;
    if let Some(why) = foreign_seq_no(&engine, p, before, seq, &resp) {
        tracing::warn!(index = %p.index, "write not acknowledged: {why}");
        return not_acked(&p.index, p.shard, &format!("operation was not sequenced by the primary copy: {why}"));
    }
    let mut info = match p.replicate_through(seq, &TransportNet(node.clone())).await {
        Ok(i) => i,
        Err(e) => {
            tracing::warn!(index = %p.index, "write not acknowledged: replication failed: {e}");
            // Executed on the primary but not acknowledged: indeterminate,
            // retried by the coordinator (see `not_acked`).
            return not_acked(&p.index, p.shard, &format!("replication failed and the failed copy could not be removed: {e}"));
        }
    };
    if !still_primary(node, p) {
        return no_longer_primary(p);
    }
    // `refresh=true|wait_for`: searchable on every copy before the ack.
    if matches!(req.params().get("refresh").map(String::as_str), Some("" | "true" | "wait_for")) {
        if let Err(e) = p.refresh_replicas(&TransportNet(node.clone()), &mut info).await {
            tracing::warn!(index = %p.index, "write not acknowledged: {e}");
            return not_acked(&p.index, p.shard, &e);
        }
    }
    // Dynamic mapping updates were committed by the master before the
    // operations that needed them were indexed (`mapping_update::ClusterMappingHook`).
    patch_shards(&mut resp, &info);
    crate::rest::tasks::rewrite_task_ref(node, p.shard, &mut resp);
    resp
}

/// Is `p` still the primary context this node holds for its shard? A write
/// that started on a primary context which was removed meanwhile (the copy
/// was failed, or the shard moved away) wrote into a copy the cluster no
/// longer knows: every replication target it remembers is stale, the master
/// "fails" them trivially, and nothing else would stop the acknowledgement.
fn still_primary(node: &Arc<ClusterNode>, p: &Arc<PrimaryCtx>) -> bool {
    node.shards.primary(&p.index, p.shard).map(|cur| Arc::ptr_eq(&cur, p)).unwrap_or(false)
}

/// Is the engine index registered under `p.index` this primary's own copy?
fn engine_is(engine: &elasticsearch_engine::node::Node, p: &PrimaryCtx) -> bool {
    engine.get_index(&p.index).map(|s| Arc::ptr_eq(&s, &p.svc)).unwrap_or(false)
}

/// A primary this node still holds whose engine index was replaced or
/// dropped beneath it (engine 0 rolled back or caught up to a state that
/// re-created the shell) can never sequence another write: the context
/// keeps its old service by design, so every retry would see the same
/// mismatch. Fail the copy, as ES fails a shard whose engine failed, so the
/// master promotes an in-sync replica or reallocates, and a retried write
/// finds the new primary instead of timing out against this one.
fn fail_orphaned_primary(node: &Arc<ClusterNode>, p: &Arc<PrimaryCtx>) {
    if still_primary(node, p) && !node.engine_for(p.shard).is_some_and(|engine| engine_is(&engine, p)) {
        tracing::warn!(index = %p.index, shard = p.shard, "failing a primary whose engine copy was replaced beneath it");
        node.report_failed(&p.index, p.shard, &p.allocation_id, None, "the primary's engine copy was replaced", false);
    }
}

/// A primary assigns every new op a seq_no above all it assigned before
/// (ES `LocalCheckpointTracker`). An op answered with a seq_no at or below
/// the primary's max from before it ran, or by an engine copy that is no
/// longer the primary's, was sequenced elsewhere and must never be acked
/// under this primary (FAULT_RESULTS.md §10, G3).
fn foreign_seq_no(engine: &elasticsearch_engine::node::Node, p: &PrimaryCtx, before: i64, after: i64, resp: &RestResp) -> Option<String> {
    if !resp.is_success() || is_gzip(resp) {
        return None;
    }
    if !engine_is(engine, p) {
        return Some("the local engine copy was replaced while the operation ran".into());
    }
    let v = resp.body_json()?;
    let seq = v["_seq_no"].as_i64()?;
    if v["result"] == "noop" || seq > before {
        return None;
    }
    // `index.disable_sequence_numbers`: the response hides the seq_no
    // (`-2`); the primary's own sequencing is what counts.
    if seq == -2 && after > before {
        return None;
    }
    Some(format!("seq_no [{seq}] is not above the primary's max_seq_no [{before}]"))
}

/// Executed locally but must not be acknowledged: indeterminate for the
/// client (same class as a failed replication round), never a 2xx.
fn no_longer_primary(p: &PrimaryCtx) -> RestResp {
    tracing::warn!(index = %p.index, "write not acknowledged: this copy is no longer the primary");
    not_acked(&p.index, p.shard, "primary shard copy was removed from this node while the operation was in flight")
}

/// This copy handed primary authority to a relocation target: wait until
/// the cluster state names the new primary, then route there.
async fn reroute_after_relocation(node: &Arc<ClusterNode>, p: &PrimaryCtx, req: &RestReq) -> RestResp {
    let mut rx = node.subscribe();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    loop {
        let moved = rx.borrow().routing_table.shard_table(&p.index, p.shard).and_then(|t| t.active_primary()).map(|c| c.allocation_id() != Some(p.allocation_id.as_str())).unwrap_or(false);
        if moved {
            let mut again = req.clone();
            again.hops = again.hops.saturating_add(0);
            return Box::pin(dispatch(node, again)).await;
        }
        if tokio::time::timeout_at(deadline, rx.changed()).await.is_err() {
            return RestResp::error(503, "retry_on_primary_exception", &format!("[{}][{}] primary relocation did not complete in time", p.index, p.shard));
        }
    }
}

/// ES `RestStatus` constant name of the statuses a replica failure has.
fn rest_status_name(status: u16) -> &'static str {
    match status {
        400 => "BAD_REQUEST",
        404 => "NOT_FOUND",
        409 => "CONFLICT",
        429 => "TOO_MANY_REQUESTS",
        502 => "BAD_GATEWAY",
        503 => "SERVICE_UNAVAILABLE",
        _ => "INTERNAL_SERVER_ERROR",
    }
}

fn shards_json(info: &ShardsInfo) -> Value {
    let mut v = json!({"total": info.total, "successful": info.successful, "failed": info.failures.len()});
    if !info.failures.is_empty() {
        // ES `ReplicationResponse.ShardInfo.Failure.toXContent`.
        v["failures"] = Value::Array(info.failures.iter().map(|f| json!({"_index": f.index, "_shard": f.shard, "_node": f.node,
            "reason": {"type": f.error_type, "reason": f.reason}, "status": rest_status_name(f.status), "primary": false})).collect());
    }
    v
}

/// Rewrite `_shards` (single-doc responses and each bulk item) with the
/// real replication outcome.
fn patch_shards(resp: &mut RestResp, info: &ShardsInfo) {
    let failure_positions: BTreeSet<usize> = resp.headers.iter().find(|(name, _)| name.eq_ignore_ascii_case(failure_store::POSITIONS_HEADER))
        .and_then(|(_, value)| serde_json::from_str(value).ok()).unwrap_or_default();
    resp.retain_headers(|(name, _)| !name.eq_ignore_ascii_case(failure_store::POSITIONS_HEADER));
    if is_gzip(resp) {
        return;
    }
    let Some(mut v) = resp.body_json() else { return };
    if patch_shards_value(&mut v, info, &failure_positions) {
        resp.replace_json(&v);
    }
}

/// [`patch_shards`] on a parsed body; whether anything changed.
fn patch_shards_value(v: &mut Value, info: &ShardsInfo, failure_positions: &BTreeSet<usize>) -> bool {
    let sj = shards_json(info);
    let mut changed = false;
    if v.get("_shards").is_some() && v.get("hits").is_none() {
        v["_shards"] = sj.clone();
        changed = true;
    }
    if let Some(items) = v.get_mut("items").and_then(Value::as_array_mut) {
        for (position, it) in items.iter_mut().enumerate() {
            if let Some(inner) = it.as_object_mut().and_then(|o| o.values_mut().next()) {
                if inner.get("_shards").is_some() && inner["failure_store"] != "used" && !failure_positions.contains(&position) {
                    inner["_shards"] = sj.clone();
                    changed = true;
                }
            }
        }
    }
    changed
}

tokio::task_local! {
    /// Set while a request whose coordinating stage of indexing pressure
    /// this node holds runs (its local shard-level writes count once).
    static COORDINATING_HELD: bool;
}

/// Turn off typed shard-group results (`elasticsearch_http::typed_bulk`):
/// every local group answers JSON, as remote ones do. Equality tests and
/// A/B measurement (`RUSTIC_UNTYPED_BULK` at start sets it).
#[doc(hidden)]
pub static UNTYPED_BULK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn typed_bulk_enabled() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    !*ENV.get_or_init(|| std::env::var_os("RUSTIC_UNTYPED_BULK").is_some()) && !UNTYPED_BULK.load(std::sync::atomic::Ordering::Relaxed)
}

async fn bulk_route(node: &Arc<ClusterNode>, req: &RestReq, default_index: Option<&str>) -> RestResp {
    COORDINATING_HELD.scope(true, bulk_route_held(node, req, default_index)).await
}

async fn bulk_route_held(node: &Arc<ClusterNode>, req: &RestReq, default_index: Option<&str>) -> RestResp {
    if let Some(b) = node.state().blocks.global_blocked("write") {
        return RestResp::error(503, "cluster_block_exception", &format!("blocked by: [SERVICE_UNAVAILABLE/{}/{}];", b.id, b.description));
    }
    // The coordinating stage of indexing pressure (ES): the whole request,
    // until every group answered.
    elasticsearch_engine::indexing_pressure::configure_from(&node.engine);
    let _coordinating = match elasticsearch_engine::indexing_pressure::global().coordinating(req.body.len() as u64) {
        Ok(guard) => guard,
        Err(e) => return RestResp::json(e.status, &e.to_json_with_trace(false)),
    };
    // A body the splitter rejects (malformed or unknown action-line
    // parameter, missing source, no items) fails as a whole in the engine
    // before anything executes, with ES's exact error.
    let mut items = match bulk::parse(&req.body, default_index) {
        Some(items) if !items.is_empty() => items,
        _ => return local_engine(node, req).await,
    };
    // Auto ids are generated here, once: a retried group carries the same
    // ids, so a retry cannot index a document a second time.
    // ES's id layouts: k-ordered for logsdb / columnar targets (the index,
    // or a data stream's write index), else time-based (`IndexRouting`).
    let mut k_ordered: HashMap<Option<String>, bool> = HashMap::new();
    bulk::assign_ids(&mut items, |it| {
        let k = *k_ordered.entry(it.index.clone()).or_insert_with(|| {
            it.index.as_deref().and_then(|name| {
                let engine = &node.engine;
                engine.get_index(name).or_else(|| engine.data_stream(name).and_then(|d| engine.get_index(d.write_index())))
            }).is_some_and(|svc| elasticsearch_engine::index::k_ordered_ids(svc.setting("index.mode").as_deref()))
        });
        if k { elasticsearch_engine::index::auto_id_k_ordered() } else { elasticsearch_engine::index::auto_id() }
    });
    // Every subgroup is built from `items`: nothing below reads the body.
    let mut stable = match req.try_without_body() { Ok(req) => req, Err(e) => return err(e) };
    if let Err(e) = failure_store::prepare_bulk(&mut stable, items.len()) { return err(e); }
    let req = &stable;
    // Items whose group failed in a way that may have executed it.
    let mut indeterminate: std::collections::BTreeMap<usize, (u16, Value)> = Default::default();
    let prepared_ingest = ingest::prepared(req);
    let url_require_data_stream = req.params().get("require_data_stream").is_some_and(|v| v == "true");
    let url_require_alias = req.params().get("require_alias").map(|v| v == "true").unwrap_or(false);
    let deadline = tokio::time::Instant::now() + request_timeout(req);
    let mut parts: Vec<(Vec<usize>, bulk::PartResult)> = Vec::new();
    let mut raw_sources = elasticsearch_engine::source_json::ResponseSources::default();
    let mut resolved: HashMap<(String, bool), Result<SharedText, RestResp>> = HashMap::new();
    let mut tsdb_routing = crate::shard_engines::TsdbRoutingCache::default();
    let mut internal_backing_writes = BTreeSet::new();
    let mut pending: Vec<usize> = (0..items.len()).collect();
    let mut first_attempt = true;
    loop {
        // Group pending positions by the node holding each item's primary
        // shard (murmur3 over `_id`/`routing` on multi-shard indices).
        let mut by_node: std::collections::BTreeMap<(String, u32), Vec<usize>> = Default::default();
        let url_routing = req.params().get("routing").cloned();
        let mut waiting: Vec<(usize, bulk::PartResult)> = Vec::new();
        let st = node.state();
        for &i in &pending {
            let it = &items[i];
            let Some(name) = it.index.clone() else {
                parts.push((vec![i], bulk::PartResult::Failed { status: 400, error: json!({"type": "action_request_validation_exception", "reason": "Validation Failed: 1: index is missing;"}) }));
                continue;
            };
            // Preserve per-item validation before primary routing auto-creates.
            if it.op != "delete" {
                // Parsed only when the action line can carry the flag (most
                // don't): this runs per item on the async runtime.
                let required = if it.action_line.contains("require_data_stream") {
                    let meta: Value = serde_json::from_str(&it.action_line).ok().and_then(|v: Value| v.as_object().and_then(|o| o.values().next().cloned())).unwrap_or(Value::Null);
                    meta.get("require_data_stream").and_then(|v| v.as_bool().or_else(|| v.as_str().map(|s| s == "true"))).unwrap_or(url_require_data_stream)
                } else { url_require_data_stream };
                if let Err(e) = node.engine.require_data_stream(&name, required) {
                    parts.push((vec![i], bulk::PartResult::Failed { status: e.status, error: e.to_json_with_trace(false)["error"].clone() }));
                    continue;
                }
            }
            // `require_alias` (item, else URL) forbids auto-creation.
            if let Some(elasticsearch_http::PreparedBulkIngest::Failed(failure)) = prepared_ingest.0.get(&i) {
                // Ingest failed before an index existed. Store against the
                // actual ingest destination without auto-creating the original
                // target or advancing the first failure-index generation.
                let metadata: Value = serde_json::from_str(&it.action_line).ok().and_then(|line: Value| line.as_object().and_then(|object| object.values().next().cloned())).unwrap_or(Value::Null);
                let mut document = req.clone();
                if let Err(error) = document.set_method("POST") { return err(error); }
                let params = req.params();
                let uri=DocumentUri{index:&name,id:it.id.as_deref(),create:false,refresh:params.get("refresh").map(String::as_str),routing:metadata.get("routing").and_then(Value::as_str).or_else(||params.get("routing").map(String::as_str))};
                if let Err(error) = document.set_uri_fmt(format_args!("{uri}")).and_then(|_| document.set_body(it.source_line.as_deref().unwrap_or_default().as_bytes())) { return err(error); }
                if let Err(error) = document.replace_header_json(failure_store::TOKEN_HEADER, &failure_store::token(req, Some(i))) { return err(error); }
                let response = failure_store::ingest_failure(node, &document, &name, failure.clone()).await;
                let mut result = response.body_json().unwrap_or(Value::Null);
                let item = if response.is_success() {
                    result["status"] = json!(response.status);
                    json!({it.op.clone(): result})
                } else {
                    let mut item = bulk::error_item(it, response.status, &result["error"]);
                    if let Some(status) = result["error"]["failure_store"].as_str() {
                        if let Some(inner) = item.as_object_mut().and_then(|object| object.values_mut().next()) { inner["failure_store"] = json!(status); }
                    }
                    item
                };
                parts.push((vec![i], bulk::PartResult::Response(json!({"items":[item]}))));
                continue;
            }
            let key = (name, it.op != "delete" && it.require_alias.unwrap_or(url_require_alias));
            if !resolved.contains_key(&key) {
                let r = concrete_write_index(node, &key.0, !key.1).await;
                resolved.insert(key.clone(), r);
            }
            match &resolved[&key] {
                Ok(concrete) => {
                    // Validate the operation as supplied by the client before
                    // auto-generated IDs turn an index operation into create.
                    if matches!(it.op.as_str(), "index" | "create") {
                        // Parsed only when the action line can carry a CAS condition.
                        let cas = (it.action_line.contains("if_seq_no") || it.action_line.contains("if_primary_term")) && {
                            let meta: Value = serde_json::from_str(&it.action_line).ok().and_then(|v: Value| v.as_object().and_then(|o| o.values().next().cloned())).unwrap_or(Value::Null);
                            meta.get("if_seq_no").is_some() || meta.get("if_primary_term").is_some()
                        };
                        if let Err(e) = node.engine.check_backing_write(&key.0, it.original.as_ref().map(|(_, create)| *create).unwrap_or(it.create), cas) {
                            parts.push((vec![i], bulk::PartResult::Failed { status: e.status, error: e.to_json_with_trace(false)["error"].clone() }));
                            continue;
                        }
                    }
                    // A time series data stream writes into the backing
                    // index covering the document's `@timestamp`.
                    // A data stream item's source, scanned once for its
                    // `@timestamp` (routing and the data stream checks).
                    let scan = (matches!(items[i].op.as_str(), "index" | "create") && node.engine.is_data_stream(&key.0))
                        .then(|| items[i].source_line.as_deref().map(elasticsearch_engine::datastream::TimestampScan::of)).flatten();
                    let covering = match (items[i].source_line.as_deref(), items[i].op.as_str(), scan.as_ref()) {
                        // Not a data stream: no backing index to choose.
                        (Some(_), "index" | "create", None) => None,
                        (Some(src), "index" | "create", Some(scan)) => match covering_index(node,&key.0,src,scan).await {
                            Ok(value)=>value,
                            Err(response)=>{
                                // This is an item resolution failure. Preserve already
                                // completed items and never replay them as a whole bulk.
                                parts.push((vec![i],bulk_resolution_failure(&response)));
                                continue;
                            }
                        },
                        _ => None,
                    };
                    // A time series data stream write no backing index covers
                    // fails here, with the data stream's own error.
                    if let (None, Some(src), "index" | "create", Some(scan)) = (&covering, items[i].source_line.as_deref(), items[i].op.as_str(), scan.as_ref()) {
                        if let Err(e) = data_stream_write_check(node, &key.0, items[i].create, src, scan) {
                            let meta: Value = serde_json::from_str(&items[i].action_line).ok().and_then(|line: Value| line.as_object().and_then(|object| object.values().next().cloned())).unwrap_or(Value::Null);
                            let mut document = req.clone();
                            if let Err(error) = document.set_method("POST") { return err(error); }
                            let params = req.params();
                            let uri=DocumentUri{index:&key.0,id:if items[i].generated{None}else{items[i].id.as_deref()},create:true,refresh:params.get("refresh").map(String::as_str),routing:meta.get("routing").and_then(Value::as_str).or_else(||params.get("routing").map(String::as_str))};
                            if let Err(error) = document.set_uri_fmt(format_args!("{uri}")).and_then(|_| document.set_body(src.as_bytes())) { return err(error); }
                            if let Err(error) = document.replace_header_json(failure_store::TOKEN_HEADER, &failure_store::token(req, Some(i))) { return err(error); }
                            let response = failure_store::timestamp_failure(node, &document, &key.0, e).await;
                            let mut result = response.body_json().unwrap_or(Value::Null);
                            if response.is_success() {
                                result["status"] = json!(response.status);
                                parts.push((vec![i], bulk::PartResult::Response(json!({"items":[{"create":result}]}))));
                            } else {
                                let mut item = bulk::error_item(&items[i], response.status, &result["error"]);
                                if let Some(status) = result["error"]["failure_store"].as_str() {
                                    if let Some(inner) = item.as_object_mut().and_then(|object| object.values_mut().next()) { inner["failure_store"] = json!(status); }
                                }
                                parts.push((vec![i], bulk::PartResult::Response(json!({"items":[item]}))));
                            }
                            continue;
                        }
                    }
                    let concrete = covering.as_ref().unwrap_or(concrete);
                    // Time series: the primary generates the `_id` from the
                    // document; route an id-less write by its dimensions.
                    // Keep the published state alive instead of cloning the
                    // entire cluster metadata for every bulk item. Each item
                    // still sees the current state, including auto-creation.
                    let state = node.state();
                    let md = &state.metadata;
                    let ts = crate::shard_engines::is_time_series(md, concrete);
                    if ts && items[i].generated {
                        items[i].unassign_id();
                    }
                    let it = &items[i];
                    // The action line's routing, parsed once by `bulk::parse`.
                    let routing = match prepared_ingest.0.get(&i) {
                        Some(elasticsearch_http::PreparedBulkIngest::Write { routing, .. }) => routing.clone(),
                        _ => it.routing.clone().or_else(|| url_routing.clone()),
                    };
                    let id = it.id.clone().unwrap_or_default();
                    if let Some(e) = crate::shard_engines::tsdb_doc_error(&md, concrete, &it.op, routing.as_deref()) {
                        parts.push((vec![i], bulk::PartResult::Failed { status: e.status, error: e.to_json_with_trace(false)["error"].clone() }));
                        continue;
                    }
                    // Only a multi-shard index routes on the source (one shard:
                    // shard 0 whatever it holds): the parse of every touch
                    // document was ~6% of a TSDB bulk's CPU.
                    let multi_shard = md.indices.get(concrete.as_str()).is_some_and(|im| im.number_of_shards > 1);
                    let ts_source_shard = (ts && it.id.is_none() && it.op != "delete")
                        .then(|| if multi_shard { it.source_line.as_deref().and_then(|l| serde_json::from_str::<Value>(l).ok()) } else { Some(Value::Null) })
                        .flatten()
                        .map(|src| tsdb_routing.source_shard(md, concrete, &src));
                    let shard = match ts_source_shard.map(Ok).unwrap_or_else(|| crate::shard_engines::doc_shard(md, concrete, &id, routing.as_deref())) {
                        Ok(k) => k,
                        Err(_) => {
                            parts.push((vec![i], bulk::PartResult::Failed { status: 400, error: json!({"type": "routing_missing_exception", "reason": format!("routing is required for [{concrete}]/[{id}]"), "index": concrete}) }));
                            continue;
                        }
                    };
                    // Send the validated item directly to its backing index.
                    // The internal backing-write marker lets every shard's
                    // REST layer distinguish this coordinator write from a
                    // client attempt to address a backing index.
                    let stream_is_known = node.engine.data_stream(&key.0).is_some()
                        || node.state().metadata.customs.get("data_stream").and_then(Value::as_object)
                            .is_some_and(|streams| streams.contains_key(&key.0));
                    if matches!(items[i].op.as_str(), "index" | "create") && stream_is_known {
                        let checked = match (items[i].source_line.as_deref(), scan.as_ref()) {
                            (Some(src), Some(scan)) => node.engine.check_data_stream_write_only_probed(&key.0, items[i].create, Some((src, scan))),
                            (src, _) => node.engine.check_data_stream_write_only(&key.0, items[i].create, src),
                        };
                        if let Err(e) = checked {
                            let meta: Value = serde_json::from_str(&items[i].action_line).ok()
                                .and_then(|line: Value| line.as_object().and_then(|object| object.values().next().cloned())).unwrap_or(Value::Null);
                            let routing = meta["routing"].as_str().map(str::to_string).or_else(|| url_routing.clone());
                            let response = failure_store::bulk_prewrite_failure(node, req, &key.0, items[i].id.as_deref(),
                                routing.as_deref(), items[i].source_line.as_deref().unwrap_or_default(), e, i).await;
                            let mut result = response.body_json().unwrap_or(Value::Null);
                            let item = if response.is_success() {
                                result["status"] = json!(response.status);
                                json!({items[i].op.clone(): result})
                            } else {
                                let mut item = bulk::error_item(&items[i], response.status, &result["error"]);
                                if let Some(inner) = item.as_object_mut().and_then(|object| object.values_mut().next()) {
                                    inner["_index"] = json!(concrete);
                                    if let Some(status) = result["error"]["failure_store"].as_str() {
                                        inner["failure_store"] = json!(status);
                                    }
                                }
                                item
                            };
                            parts.push((vec![i], bulk::PartResult::Response(json!({"items":[item]}))));
                            continue;
                        }
                        items[i].retarget(concrete);
                        internal_backing_writes.insert(i);
                    }
                    match primary_node_of(node, concrete, shard) {
                        Some(n) => by_node.entry((n, shard)).or_default().push(i),
                        None => waiting.push((i, bulk::PartResult::Failed { status: 503, error: json!({"type": "unavailable_shards_exception", "reason": format!("[{concrete}][{shard}] primary shard is not active Timeout: [1m]")}) })),
                    }
                }
                Err(r) => {
                    parts.push((vec![i], bulk_resolution_failure(r)));
                }
            }
        }
        drop(st);
        let mut futs = Vec::new();
        for ((n, shard), positions) in by_node {
            let mut sub = match req.try_with_encoded_body(|out| {
                for &i in &positions {
                    out.write_all(items[i].action_line.as_bytes())?; out.write_all(b"\n")?;
                    if let Some(source)=&items[i].source_line { out.write_all(source.as_bytes())?; out.write_all(b"\n")?; }
                }
                Ok(())
            }) { Ok(sub) => sub, Err(e) => return err(e) };
            // Every item's data stream requirement was checked above, against
            // the name it was sent to; items may now name backing indices.
            let original=sub.clone();
            if let Err(e) = sub.set_uri_fmt(format_args!("{}{}",original.path(),FilteredQuery{query:original.query(),drop:&["require_data_stream"],append:""})) { return err(e); }
            if let Err(e) = ingest::subgroup(req, &mut sub, &positions) { return err(e); }
            if let Err(e) = failure_store::subgroup(req, &mut sub, &positions) { return err(e); }
            if positions.iter().any(|i| internal_backing_writes.contains(i)) {
                if let Err(e) = sub.push_header(DATA_STREAM_BACKING_WRITE_HEADER, "true") { return err(e); }
                // This subgroup already passed the coordinator's data-stream
                // checks and names a backing index. Do not feed it through
                // bulk_route again on the remote primary, where it would be
                // mistaken for a client attempt to write to `.ds-*`.
                if let Err(e) = sub.push_header(EXEC_LOCAL, "true") { return err(e); }
            }
            if let Err(error)=sub.set_shard(shard){return err(error)};
            let node = node.clone();
            // A shard engine may not contain the logical data-stream metadata.
            // Match the concrete target serialized for remote primaries, so
            // local backing-index writes cannot bypass primary replication.
            let names: BTreeSet<String> = positions.iter().filter_map(|i| items[*i].target_index.as_ref().or(items[*i].index.as_ref()).cloned()).collect();
            // Ids generated here, executed in process on the first attempt:
            // no earlier copy can exist (a forwarded group may still be in
            // flight when its retry runs, so only local groups qualify).
            let append_only = (first_attempt && positions.iter().any(|i| items[*i].generated)).then(|| Arc::new(positions.iter().map(|i| items[*i].generated).collect::<Vec<bool>>()));
            // A group executed here hands its items to the engine already
            // parsed (security authorizes them item by item as they are).
            let local = n == node.local.id;
            // It answers typed (`elasticsearch_http::typed_bulk`): this
            // coordinator renders its items once, below.
            let prepared = local.then(|| elasticsearch_http::PreparedBulk::new(positions.iter().map(|i| {
                let it = &items[*i];
                elasticsearch_http::PreparedBulkItem { action: it.wire_op().to_string(), meta: it.meta.clone(), source: it.source_line.clone() }
            }).collect(), sub.body.len() as u64)).map(|p| if typed_bulk_enabled() { p.typed() } else { p });
            futs.push(async move {
                let r = if local { local_bulk_json(&node, &sub, names, append_only, prepared).await } else { (forward(&node, &n, &sub).await, None, None) };
                (positions, r)
            });
        }
        let mut retry: Vec<(usize, bulk::PartResult)> = waiting;
        for (positions, (r, json, typed)) in futures::future::join_all(futs).await {
            if let (true, Some(typed)) = (r.is_success(), typed) {
                parts.push((positions, bulk::PartResult::Typed(typed)));
                continue;
            }
            let json = json.or_else(|| raw_sources.parse(&r.body).ok());
            if r.is_success() {
                let part = match json {
                    Some(v) => bulk::PartResult::Response(v),
                    None => bulk::PartResult::Failed { status: 500, error: json!({"type": "exception", "reason": "unreadable bulk response"}) },
                };
                parts.push((positions, part));
            } else {
                let err = json.map(|v| v["error"].clone()).unwrap_or(Value::Null);
                if retryable(&r) {
                    // Try the group's items again. After a broken or timed
                    // out connection the group may have executed: remember
                    // that, so a create that then conflicts with its own
                    // first attempt is recognised below.
                    if is_indeterminate(&r) {
                        indeterminate.extend(positions.iter().map(|i| (*i, (r.status, err.clone()))));
                    }
                    for i in positions {
                        retry.push((i, bulk::PartResult::Failed { status: r.status, error: err.clone() }));
                    }
                } else {
                    parts.push((positions, bulk::PartResult::Failed { status: r.status, error: err }));
                }
            }
        }
        if retry.is_empty() {
            break;
        }
        if !wait_state_change(node, deadline).await {
            for (i, f) in retry {
                parts.push((vec![i], f));
            }
            break;
        }
        pending = retry.into_iter().map(|(i, _)| i).collect();
        first_attempt = false;
        // Primaries may have moved; resolve write indices afresh.
        resolved.clear();
    }
    if !indeterminate.is_empty() {
        // The checks below read and replace item values.
        for (_, part) in parts.iter_mut() {
            if let bulk::PartResult::Typed(_) = part {
                let bulk::PartResult::Typed(typed) = std::mem::replace(part, bulk::PartResult::Response(Value::Null)) else { unreachable!() };
                *part = bulk::PartResult::Response(typed.into_response());
            }
        }
        for (positions, part) in parts.iter_mut() {
            let bulk::PartResult::Response(v) = part else { continue };
            for (k, pos) in positions.iter().enumerate() {
                let it = &items[*pos];
                let Some((lost_status, lost_error)) = indeterminate.get(pos) else { continue };
                let Some(inner) = v["items"][k].as_object().and_then(|o| o.values().next()).cloned() else { continue };
                // Only an answer the item's own earlier attempt can have
                // caused is in doubt (see `own_echo`).
                if !own_echo(it.op == "delete", inner["status"].as_u64().unwrap_or(0) as u16, &inner) {
                    continue;
                }
                let meta: Value = serde_json::from_str(&it.action_line).ok().and_then(|l: Value| l.as_object().and_then(|o| o.values().next().cloned())).unwrap_or(Value::Null);
                let routing = meta["routing"].as_str().map(encode_component);
                let sent: Option<Value> = it.source_line.as_deref().and_then(|s| serde_json::from_str::<Value>(s).ok());
                let index = it.index.clone().unwrap_or_default();
                let mut confirmed: Option<Value> = None;
                if let Some(id) = it.id.as_deref() {
                    if it.create {
                        let source = if it.generated { None } else { Some(sent.clone().unwrap_or(Value::Null)) };
                        confirmed = match confirm_created(node, &inner["error"], &index, &encode_component(id), routing.as_deref(), source.as_ref(), deadline).await {Ok(value)=>value,Err(error)=>return err(error)};
                    } else if let (true, Some(source)) = (it.op == "index", &sent) {
                        let cond = Condition { if_seq_no: meta["if_seq_no"].as_i64(), version: meta["version"].as_i64() };
                        confirmed = match confirm_written(node, &inner["error"], &index, &encode_component(id), routing.as_deref(), source, cond, deadline).await {Ok(value)=>value,Err(error)=>return err(error)};
                    }
                }
                let replacement = match confirmed {
                    Some(mut done) => {
                        done["status"] = json!(if done["result"] == "created" { 201 } else { 200 });
                        done
                    }
                    // Unknown outcome: the lost attempt's error, never a
                    // definite failure for an item that may have applied.
                    None => {
                        let mut e = json!({"_index": inner["_index"], "status": lost_status, "error": lost_error});
                        if !inner["_id"].is_null() {
                            e["_id"] = inner["_id"].clone();
                        }
                        e
                    }
                };
                if let Some(slot) = v["items"][k].as_object_mut().and_then(|o| o.values_mut().next()) {
                    *slot = replacement;
                }
            }
        }
    }
    let mut response = RestResp::json(200, &Value::Null);
    if !parts.is_empty() && parts.iter().all(|(_, part)| matches!(part, bulk::PartResult::Typed(_))) {
        // Every group answered typed: the response written directly.
        let typed = parts.into_iter().filter_map(|(positions, part)| match part { bulk::PartResult::Typed(t) => Some((positions, t)), _ => None }).collect();
        response.replace_body_vec(bulk::render_typed(&items, typed));
        return response;
    }
    let merged = bulk::merge(&items, parts);
    response.replace_body_vec(raw_sources.render(&merged));
    response
}

/// A bulk whose items all have their primaries here (the item index
/// names are `names`): execute under every involved primary's gate, then
/// replicate each. A plain JSON response comes back parsed (`Some`, empty
/// body), see [`local_engine_json`].
async fn local_bulk(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let items = bulk::parse(&req.body, req.segments().first().filter(|s| !s.starts_with('_')).map(String::as_str)).unwrap_or_default();
    let names = items.iter().filter_map(|item| item.index.clone()).collect();
    let (mut response, json, _) = local_bulk_json(node, req, names, None, None).await;
    if let Some(value) = json { response.replace_json(&value); }
    response
}

async fn local_bulk_json(node: &Arc<ClusterNode>, req: &RestReq, names: BTreeSet<String>, append_only: Option<Arc<Vec<bool>>>, prepared: Option<elasticsearch_http::PreparedBulk>) -> (RestResp, Option<Value>, Option<bulk::TypedPart>) {
    if let Err(reason) = crate::internal_capabilities::require_local_writer(&node.state(), &node.local) {
        return (RestResp::error(503, "unsupported_operation_exception", &reason), None, None);
    }
    let profile = tracing::enabled!(target: "bulk_profile", tracing::Level::DEBUG).then(std::time::Instant::now);
    let shard = crate::shard_engines::shard_of(&req.headers).unwrap_or(0);
    let mut prims: Vec<Arc<PrimaryCtx>> = Vec::new();
    if !names.is_empty() {
        let Some(engine) = node.engine_for(shard) else { return (unavailable_shard(names.first().map(String::as_str).unwrap_or_default(), shard), None, None) };
        // One blocking hop resolves every distinct target.
        let concrete: Vec<String> = tokio::task::spawn_blocking(move || names.iter().filter_map(|n| engine.write_index(n, false).ok().map(|s| s.name.clone())).collect::<BTreeSet<_>>().into_iter().collect()).await.unwrap_or_default();
        for c in concrete {
            match node.shards.primary(&c, shard) {
                Some(p) => prims.push(p),
                // The index is routed, but this node holds no primary
                // for it (stale routing on the sender, a copy that is
                // still recovering, or one just removed). Writing into
                // the local engine would acknowledge ops no primary
                // ever sequenced or replicated; answer like ES does so
                // the coordinator retries on the next cluster state.
                None if node.state().routing_table.shard_table(&c, shard).is_some() => return (unavailable_shard(&c, shard), None, None),
                None => {}
            }
        }
    }
    let mut guards = Vec::new();
    for p in &prims {
        guards.push(p.gate.read().await);
    }
    if let Some(p) = prims.iter().find(|p| p.relocated.load(std::sync::atomic::Ordering::SeqCst)) {
        drop(guards);
        return (reroute_after_relocation(node, p, req).await, None, None);
    }
    let setup_us = profile.map(|t|t.elapsed().as_micros() as u64).unwrap_or(0);
    let (mut resp, mut json) = local_engine_json_with(node, req, append_only, prepared.as_ref()).await;
    // Typed results the engine left (`elasticsearch_http::typed_bulk`):
    // its JSON body is then a placeholder.
    let typed = prepared.as_ref().and_then(elasticsearch_http::PreparedBulk::take_typed);
    let engine_done_us = profile.map(|t|t.elapsed().as_micros() as u64).unwrap_or(0);
    let refresh = matches!(req.params().get("refresh").map(String::as_str), Some("" | "true" | "wait_for"));
    let mut total = ShardsInfo::default();
    for p in &prims {
        let mut info = match p.replicate_through(p.max_seq_no_async().await, &TransportNet(node.clone())).await {
            Ok(info) => info,
            Err(e) => return (not_acked(&p.index, p.shard, &format!("replication failed: {e}")), None, None),
        };
        if !still_primary(node, p) {
            return (no_longer_primary(p), None, None);
        }
        if refresh {
            if let Err(e) = p.refresh_replicas(&TransportNet(node.clone()), &mut info).await {
                return (not_acked(&p.index, p.shard, &e), None, None);
            }
        }
        total.total = total.total.max(info.total);
        total.successful = total.successful.max(info.successful);
        total.failures.extend(info.failures);
    }
    drop(guards);
    if let Some(t) = profile {
        tracing::debug!(target: "bulk_profile", shard, bytes=req.body.len(), setup_us,
            engine_us=engine_done_us.saturating_sub(setup_us),
            replication_and_mapping_us=(t.elapsed().as_micros() as u64).saturating_sub(engine_done_us), "local bulk stages");
    }
    if let Some(typed) = typed {
        // The JSON path's `patch_shards_value`, on the typed results.
        let mut part = bulk::TypedPart { took: typed.took, items: typed.items, shards: None, keep_own: BTreeSet::new() };
        if !prims.is_empty() {
            let failure_positions: BTreeSet<usize> = resp.headers.iter().find(|(name, _)| name.eq_ignore_ascii_case(failure_store::POSITIONS_HEADER))
                .and_then(|(_, value)| serde_json::from_str(value).ok()).unwrap_or_default();
            resp.retain_headers(|(name, _)| !name.eq_ignore_ascii_case(failure_store::POSITIONS_HEADER));
            let sj = shards_json(&total);
            for (position, item) in part.items.iter_mut().enumerate() {
                if let elasticsearch_http::typed_bulk::TypedItem::Json(v) = item {
                    if let Some(inner) = v.as_object_mut().and_then(|o| o.values_mut().next()) {
                        if inner.get("_shards").is_some() && inner["failure_store"] != "used" && !failure_positions.contains(&position) {
                            inner["_shards"] = sj.clone();
                        }
                    }
                }
            }
            part.keep_own = failure_positions;
            part.shards = Some(sj);
        }
        return (resp, None, Some(part));
    }
    if !prims.is_empty() {
        match &mut json {
            Some(v) => {
                let failure_positions: BTreeSet<usize> = resp.headers.iter().find(|(name, _)| name.eq_ignore_ascii_case(failure_store::POSITIONS_HEADER))
                    .and_then(|(_, value)| serde_json::from_str(value).ok()).unwrap_or_default();
                resp.retain_headers(|(name, _)| !name.eq_ignore_ascii_case(failure_store::POSITIONS_HEADER));
                patch_shards_value(v, &total, &failure_positions);
            }
            None => patch_shards(&mut resp, &total),
        }
    }
    (resp, json, None)
}

/// Reads whose index expression may name remote clusters (ES CCS).
fn ccs_endpoint(req: &RestReq) -> bool {
    let segs = req.segments();
    segs.len() == 2 && match segs[1].as_str() {
        "_search" | "_count" | "_pit" => true,
        "_async_search" => req.method() == Method::POST,
        _ => false,
    }
}

/// Percent-encode an index expression for a path segment, keeping the
/// separators and wildcards that `segments()` hands back verbatim.
fn encode_targets(targets: &str) -> String {
    targets.split(',').map(|part| encode_component(part).replace("%2A", "*")).collect::<Vec<_>>().join(",")
}

/// A local+remote read: the shared CCS plan fails a required remote, skips an
/// optional one, and the local part runs as an ordinary cluster read (every
/// shard, on whichever nodes hold it). Searches report `_clusters`.
async fn ccs_read(node: &Arc<ClusterNode>, req: &RestReq, targets: &str) -> RestResp {
    let started = std::time::Instant::now();
    let plan = match elasticsearch_engine::ccr::remote::ccs_plan(&node.engine, targets) {
        Ok(plan) => plan,
        Err(error) => {
            node.engine.ccs_stats.lock().record(started.elapsed().as_millis() as u64, &[], &[], false, true, targets.contains(['*', '?']));
            return err(error);
        }
    };
    let search = matches!(req.segments().last().map(String::as_str), Some("_search" | "_async_search"));
    let Some(local) = plan.local_expr() else {
        // Nothing of this cluster is read: the engine renders the empty
        // (skipped) answer through the same plan, with no local shard.
        return wrap_context_ids(node, req, local_canonical(node, req).await);
    };
    let rest = req.path().trim_start_matches('/').split_once('/').map(|(_, rest)| rest).unwrap_or("");
    let query = req.query();
    let mut local_req = req.clone();
    if let Err(error) = local_req.set_uri_fmt(format_args!("/{}/{rest}{}{query}", encode_targets(&local), if query.is_empty() { "" } else { "?" })) {
        return err(error);
    }
    // The response is patched below: ask for it uncompressed.
    if search {
        if let Err(error) = local_req.retain_headers(|(k, _)| !k.eq_ignore_ascii_case("accept-encoding")) { return err(error); }
    }
    // A submitted async search keeps the plan with its stored result, so a
    // later `GET _async_search/{id}` still reports the skipped remotes. It
    // runs as the cluster async search (the local part as an ordinary
    // cluster `_search`), wherever the local shards are.
    let submit = req.method() == Method::POST && req.segments().last().map(String::as_str) == Some("_async_search");
    let mut resp = if submit {
        let holders = Default::default();
        async_search_ccs(node, &local_req, &holders, plan.clone()).await
    } else {
        Box::pin(read(node, &local_req, &local)).await
    };
    if search {
        let aliases = plan.aliases();
        node.engine.ccs_stats.lock().record(started.elapsed().as_millis() as u64, &aliases, &aliases, resp.is_success(),
            req.params().get("ccs_minimize_roundtrips").is_none_or(|v| v != "false"), targets.contains(['*', '?']));
        if resp.is_success() && !is_gzip(&resp) {
            // Members are patched losslessly (hits keep their exact text).
            use elasticsearch_engine::source_json::{raw_members, with_member};
            let patched = std::str::from_utf8(&resp.body).ok().and_then(|text| {
                // `POST _async_search` wraps the search response.
                let wrapped = raw_members(text)?.into_iter().find(|(k, _)| k == "response").map(|(_, v)| v.get().to_string());
                let inner = wrapped.as_deref().unwrap_or(text);
                let summary: Value = serde_json::from_str(inner).ok()?;
                let clusters = plan.clusters(Some(&summary)).to_string();
                let inner = with_member(inner, "_clusters", &clusters, false)?;
                match wrapped { Some(_) => with_member(text, "response", &inner, false), None => Some(inner) }
            });
            if let Some(body) = patched {
                resp.replace_body(body.into_bytes());
                resp.retain_headers(|(k, _)| !k.eq_ignore_ascii_case("content-length"));
            }
        }
    }
    resp
}

/// Run a read on a node holding an active copy of every target index.
async fn read(node: &Arc<ClusterNode>, req: &RestReq, targets: &str) -> RestResp {
    // Parse once for validation, PIT routing and rewrite. Vector query bodies
    // contain hundreds of floats; reparsing each guard is measurable CPU work.
    // Borrowed: a `large_terms` body is ~1 MB, and a copy costs as much as a
    // parse. Only a body the rewrite changes is copied.
    let request_body = req.json_ref();
    if let Some(body) = request_body.filter(|body| {
        matches!(req.segments().last().map(String::as_str), Some("_search" | "_async_search" | "_knn_search"))
            && (body.get("rank").is_some() || body.get("retriever").is_some())
    }) {
        // Match the engine's parse-before-license ordering, before the
        // coordinator can reject an unsupported distributed plan.
        if let Err(error) = elasticsearch_http::validate_search_request(&node.engine, body, &req.params()) {
            return RestResp::json(error.status, &error.to_json_with_trace(false));
        }
        if let Err(error) = elasticsearch_http::check_search_license(&node.engine, body) {
            return RestResp::json(error.status, &error.to_json_with_trace(false));
        }
    }
    if req.segments().last().map(String::as_str) == Some("_async_search") && request_body.and_then(|v| v.get("pit")).is_some() {
        return async_search(node, req, &Default::default()).await;
    }
    // A PIT search runs where the PIT lives.
    if let Some(pit) = request_body.and_then(|v| v["pit"]["id"].as_str().map(str::to_string)) {
        if let Some((_, inner)) = unwrap_id(&pit) {
            if inner.starts_with(contexts::PIT_PREFIX) {
                if let Some(r) = contexts::search_pit(node, req, &pit, inner).await {
                    return r;
                }
            }
        }
        return context_forward(node, req, &pit, |body, inner| {
            let mut v: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
            v["pit"]["id"] = json!(inner);
            serde_json::to_vec(&v).unwrap_or_default()
        })
        .await;
    }
    let st = node.state();
    if let Some(b) = st.blocks.global_blocked("read") {
        return RestResp::error(503, "cluster_block_exception", &format!("blocked by: [SERVICE_UNAVAILABLE/{}/{}];", b.id, b.description));
    }
    if req.segments().last().map(String::as_str) == Some("_search_shards") {
        await_mirror(node).await;
        return search_shards(node, req).await;
    }
    // Cross-cluster expressions: the remote part never resolves against this
    // cluster's indices (a pattern would silently match nothing), so plan it
    // first and read only the local part. `_msearch` items arrive here as
    // their own `_search`.
    if ccs_endpoint(req) && elasticsearch_engine::ccr::remote::has_remote_expression(targets) {
        return ccs_read(node, req, targets).await;
    }
    let opts = ResolveOpts::from_params(&req.params());
    // Resolve against a whole mirror (see `await_mirror`), else a
    // recovering node would answer for an index it momentarily lacks.
    await_mirror(node).await;
    let engine = node.engine.clone();
    let expr = targets.to_string();
    // A concrete existing name is a bounded metadata lookup. Avoid waking
    // another CPU worker before the actual search's blocking job. Expressions
    // and aliases retain the general resolver path.
    let resolved = if !expr.contains([',', '*', '?', '<', '>']) && engine.get_index(&expr).is_some() {
        Some(engine.resolve(&expr, opts).map(|ts| ts.iter().map(|t| t.svc.name.clone()).collect::<BTreeSet<_>>()))
    } else {
        tokio::task::spawn_blocking(move || engine.resolve(&expr, opts).map(|ts| ts.iter().map(|t| t.svc.name.clone()).collect::<BTreeSet<_>>())).await.ok()
    };
    let names = match resolved {
        Some(Ok(n)) if !n.is_empty() => n,
        // Errors (index_not_found, ...) and empty matches render exactly
        // as the engine renders them.
        _ => return wrap_context_ids(node, req, local_canonical(node, req).await),
    };
    if matches!(req.segments().last().map(String::as_str), Some("_search" | "_count" | "_async_search")) {
        if let Some(body) = request_body.filter(|body| query_rewrite::needed(body)) {
            match query_rewrite::rewrite(node, req, body.clone()).await {
                Ok(Some(body)) => {
                    let mut rewritten = req.clone();
                    if let Err(e) = rewritten.set_json(&body) { return err(e); }
                    return Box::pin(read(node, &rewritten, targets)).await;
                }
                Ok(None) => {}
                Err(response) => return response,
            }
        }
    }
    let segs = req.segments();
    let multi_shard = names.iter().any(|n| crate::shard_engines::shards_of(&node.state().metadata, n) > 1);
    if segs.len() >= 2 && segs[segs.len() - 1] == "template" {
        // Each `_msearch/template` item names its own targets.
        match segs[segs.len() - 2].as_str() {
            "_search" if multi_shard => return search_template(node, req, targets).await,
            "_msearch" => return msearch_template(node, req, targets).await,
            _ => {}
        }
    }
    // `preference=_shards:` selects shards, which the multi-shard read does
    // for single-shard indices too.
    let shards_preference = req.params().get("preference").is_some_and(|p| p.starts_with("_shards:"));
    if multi_shard || shards_preference {
        if let Some(r) = multi_shard_read(node, req, &names).await {
            return r;
        }
    }
    // Each `_msearch` item is a search of its own (its copies, its
    // preference): a stale copy fails only the items that read it.
    if segs.last().map(String::as_str) == Some("_msearch") {
        if let Some(r) = msearch(node, req).await {
            return r;
        }
    }
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let st = node.state();
        // Active copies per index; our own copy only counts if this node
        // really holds it (right after a restart the routing table may
        // still name us before the master noticed).
        let mut holders: std::collections::BTreeMap<String, BTreeSet<String>> = Default::default();
        let mut initializing = false;
        for name in &names {
            let Some(t) = st.routing_table.shard_table(name, 0) else { continue };
            initializing |= t.copies.iter().any(|c| c.state == ShardState::Initializing);
            // Searchable copies only: on a stateless cluster the indexing
            // copy serves no reads (ES `ShardRouting.isSearchable`).
            let h: BTreeSet<String> = t
                .searchable_copies(&st.nodes)
                .filter(|c| c.node_id.as_deref() != Some(node.local.id.as_str()) || local_holds(node, c))
                .filter_map(|c| c.node_id.clone())
                .collect();
            holders.insert(name.clone(), h);
        }
        if let Some(name) = names.iter().find(|n| holders.get(*n).map(|h| h.is_empty()).unwrap_or(true)) {
            // Only a copy of a shard that has none active yet is waited for.
            let waiting = names.iter().any(|n| holders.get(n).is_none_or(|h| h.is_empty())
                && st.routing_table.shard_table(n, 0).is_some_and(|t| t.copies.iter().any(|c| c.state == ShardState::Initializing)));
            if waiting && wait_state_change(node, deadline).await {
                continue;
            }
            let missing: Vec<stateless_read::Missing> = names
                .iter()
                .filter(|n| holders.get(*n).is_none_or(|h| h.is_empty()) && st.routing_table.shard_table(n, 0).is_some_and(|t| t.stateless_without_search_copy(&st.nodes)))
                .map(|n| (n.clone(), 0))
                .collect();
            if !missing.is_empty() && missing.len() == names.iter().filter(|n| holders.get(*n).is_none_or(|h| h.is_empty())).count() {
                return stateless_unavailable(node, req, &names, &holders, missing, &st).await;
            }
            // ES `failOnUnavailable`: a search or count runs on the shards
            // that have a copy and reports the others failed (or fails as
            // a whole when every shard is missing or partial results are
            // not allowed), instead of failing as soon as one is missing.
            let op = req.segments().last().cloned().unwrap_or_default();
            if matches!(op.as_str(), "_search" | "_count") && !req.params().contains_key("scroll") {
                let by_shard: std::collections::BTreeMap<String, BTreeSet<String>> = names.iter().map(|n| (format!("{n}#0"), holders.get(n).cloned().unwrap_or_default())).collect();
                return multi_shard_fan_out(node, req, &op, &by_shard).await;
            }
            return RestResp::error(503, "no_shard_available_action_exception", &format!("[{name}][0] no active shard copy"));
        }
        // Stateless realtime document reads need the indexing shard.
        let read_shards: Vec<stateless_read::Missing> = names.iter().map(|n| (n.clone(), 0)).collect();
        if let Some(r) = stateless_read::realtime_without_primary(req, &st, &read_shards) {
            return r;
        }
        let mut common: Option<BTreeSet<String>> = None;
        for h in holders.values() {
            common = Some(match common {
                None => h.clone(),
                Some(c) => c.intersection(h).cloned().collect(),
            });
        }
        let common = common.unwrap_or_default();
        if !common.is_empty() {
            match single_node_order(node, &st, req, &names, &holders, &common) {
                // One node holds a copy of every target: it runs the whole
                // request, the next such node on any failure but a request
                // error (ES retries every shard on its next copy).
                Ok(Some(cands)) => {
                    let mut local = req.clone();
                    if let Err(e) = local.set_read_copies(&names.iter().cloned().collect::<Vec<_>>()) { return err(e); }
                    let bound = copies::copy_bound(&req.params(), &Value::Null);
                    let mut last = None;
                    // Nodes the per-shard fallback does not try again: those
                    // that failed as nodes, and, for one index, every node
                    // that failed it (a hung node is not waited for twice).
                    let mut tried: BTreeSet<String> = BTreeSet::new();
                    for c in cands {
                        let r = if c == node.local.id {
                            wrap_context_ids(node, req, local_canonical(node, &local).await)
                        } else {
                            // The executing node wraps context ids itself.
                            copy_request(node, &local, &c, bound).await
                        };
                        if copies::answered(&r) || copies::is_request_error(&r) {
                            return r;
                        }
                        if names.len() == 1 || copies::node_failure(&r) {
                            tried.insert(c.clone());
                        }
                        if last.as_ref().map_or(true, |prev| copies::replaces(prev, &r)) {
                            last = Some(r);
                        }
                    }
                    // No such node answered. A search, count, scroll or PIT
                    // goes per shard (each index on its own copies, partial
                    // results, ES's phase failure when all fail); other reads
                    // (a document, ...) fail as ES's single-shard actions do.
                    let op = req.segments().last().cloned().unwrap_or_default();
                    match last {
                        Some(r) if matches!(op.as_str(), "_search" | "_count" | "_pit") => {
                            // An index whose every holder failed keeps them (it
                            // reports its own failure, not "no copy"), unless it
                            // is the only one: then that failure is the answer.
                            let mut rest: std::collections::BTreeMap<String, BTreeSet<String>> = Default::default();
                            for (n, h) in &holders {
                                let left: BTreeSet<String> = h.iter().filter(|c| !tried.contains(*c)).cloned().collect();
                                if left.is_empty() && names.len() == 1 {
                                    let failures = copies::ShardFailure::from_response(&r, std::slice::from_ref(n), 0, "");
                                    return copies::phase_failure("query", "all shards failed", &failures);
                                }
                                rest.insert(n.clone(), if left.is_empty() { h.clone() } else { left });
                            }
                            return fan_out(node, req, &names, &rest).await;
                        }
                        Some(r) => return r,
                        None => {}
                    }
                }
                // The preference starts the indices at different nodes: per
                // shard at once (waiting for copies to initialize would not
                // change that).
                Ok(None) => return fan_out(node, req, &names, &holders).await,
                Err(r) => return r,
            }
        }
        // Copies still initializing may soon give one node everything.
        if initializing && wait_state_change(node, deadline).await {
            continue;
        }
        return fan_out(node, req, &names, &holders).await;
    }
}

/// The nodes, in order, that may run a read of single-shard `names` whole
/// (`common`: the nodes holding a copy of every one). Without a preference
/// this node first, then the others in a random rotation. With one, every
/// index must start at the same node under it (else `None`: the read fans
/// out per shard), and only nodes it allows for every index follow.
fn single_node_order(node: &ClusterNode, st: &elasticsearch_cluster::state::ClusterState, req: &RestReq, names: &BTreeSet<String>,
    holders: &std::collections::BTreeMap<String, BTreeSet<String>>, common: &BTreeSet<String>) -> Result<Option<Vec<String>>, RestResp> {
    let pref = copies::Preference::parse(req.params().get("preference").map(String::as_str))?;
    if !pref.pins() {
        let mut rest: Vec<String> = common.iter().filter(|n| **n != node.local.id).cloned().collect();
        if !rest.is_empty() {
            let start = rand::random::<usize>() % rest.len();
            rest.rotate_left(start);
        }
        let mut order: Vec<String> = common.iter().filter(|n| **n == node.local.id).cloned().collect();
        order.extend(rest);
        return Ok(Some(order));
    }
    let mut lists = Vec::new();
    for n in names {
        let order = copies::order_copies(node, st, &pref, n, 0, holders.get(n).unwrap_or(&BTreeSet::new()), 0, &mut Default::default())?;
        if order.is_empty() {
            return Ok(None);
        }
        lists.push(order);
    }
    let Some(first) = lists.first() else { return Ok(None) };
    if lists.iter().any(|l| l[0] != first[0]) {
        return Ok(None);
    }
    Ok(Some(first.iter().filter(|c| common.contains(*c) && lists.iter().all(|l| l.contains(*c))).cloned().collect()))
}

/// A read of single-shard indices some of which are stateless shards
/// without a search copy (see [`stateless_read`]): ES fails those shards
/// like unassigned ones. A search / count over other indices too runs on
/// them and reports the missing shards as failed (unless
/// `allow_partial_search_results=false`); everything else gets ES's
/// unavailable-shard response.
async fn stateless_unavailable(
    node: &Arc<ClusterNode>,
    req: &RestReq,
    names: &BTreeSet<String>,
    holders: &std::collections::BTreeMap<String, BTreeSet<String>>,
    missing: Vec<stateless_read::Missing>,
    st: &elasticsearch_cluster::state::ClusterState,
) -> RestResp {
    let available: BTreeSet<String> = names.iter().filter(|n| holders.get(*n).is_some_and(|h| !h.is_empty())).cloned().collect();
    let search = stateless_read::classify(req) == stateless_read::ReadKind::Search;
    let partial_ok = copies::allow_partial(node, &req.params());
    if search && partial_ok && !available.is_empty() && req.segments().last().map(String::as_str) != Some("_async_search") {
        let subset: std::collections::BTreeMap<String, BTreeSet<String>> = holders.iter().filter(|(n, _)| available.contains(*n)).map(|(n, h)| (n.clone(), h.clone())).collect();
        let r = fan_out(node, req, &available, &subset).await;
        return match r.body_json().filter(|_| r.is_success()) {
            Some(v) => RestResp::json(r.status, &stateless_read::add_missing(v, &missing)),
            None => r,
        };
    }
    stateless_read::all_missing_response(req, st, &missing).unwrap_or_else(|| {
        let (name, k) = &missing[0];
        RestResp::error(503, "no_shard_available_action_exception", &format!("[{name}][{k}] no active shard copy"))
    })
}

/// Reads touching an index with more than one shard. A document read
/// runs on a copy of the shard its `_id`/`routing` hashes to; `_search` and
/// `_count` run once per (node, shard) over every shard and are reduced like
/// a cross-node search. `None` leaves the request to the single-shard path
/// (index-level reads that shard 0's engine answers).
/// How many `_msearch` items run at once: `max_concurrent_searches`, else
/// ES's default (`TransportMultiSearchAction`): data nodes times the search
/// pool size, at most 10 per node.
fn msearch_concurrency(node: &ClusterNode, req: &RestReq) -> usize {
    if let Some(n) = req.params().get("max_concurrent_searches").and_then(|v| v.parse::<usize>().ok()).filter(|n| *n > 0) {
        return n;
    }
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let pool = (cores * 3 / 2 + 1).min(10);
    node.state().nodes.values().filter(|n| n.is_data()).count().max(1) * pool
}

/// The `template_output` of a search template body, rendered by this node's
/// engine (stored scripts are cluster metadata, mirrored to every node).
async fn render_search_template(node: &Arc<ClusterNode>, req: &RestReq, body: &Value) -> Result<Value, RestResp> {
    let sub = contexts::json_req(req, "POST", "/_render/template", body).map_err(err)?;
    let r = local_canonical(node, &sub).await;
    if !r.is_success() {
        return Err(r);
    }
    r.body_json().and_then(|mut v| v.get_mut("template_output").map(Value::take))
        .ok_or_else(|| RestResp::error(500, "illegal_state_exception", "unreadable rendered search template"))
}

/// `_search/template` over several shards: rendered on the coordinating node
/// (as ES does), then run as the plain search, which fans out.
async fn search_template(node: &Arc<ClusterNode>, req: &RestReq, targets: &str) -> RestResp {
    let rendered = match render_search_template(node, req, &req.json().unwrap_or(json!({}))).await {
        Ok(v) => v,
        Err(r) => return r,
    };
    let prefix=if targets.is_empty(){""}else{"/"};
    let query = match contexts::query_without(req.query(), &["explain", "profile"]) {Ok(query)=>query,Err(error)=>return err(error)};
    let sub = match contexts::json_req_formatted(req, "POST", format_args!("{prefix}{targets}/_search{query}"), &rendered) { Ok(sub) => sub, Err(e) => return err(e) };
    Box::pin(read(node, &sub, targets)).await
}

/// `_msearch/template`: every item rendered here and run as its own cluster
/// search (an item's failure is that item's response, as in ES).
async fn msearch_template(node: &Arc<ClusterNode>, req: &RestReq, targets: &str) -> RestResp {
    let started = std::time::Instant::now();
    let text = String::from_utf8_lossy(&req.body).into_owned();
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.len() % 2 == 1 {
        return RestResp::error(400, "illegal_argument_exception", "The msearch request must be terminated by a newline [\n]");
    }
    let params = req.params();
    let query=contexts::SelectedParams(&params,&["rest_total_hits_as_int","typed_keys"]);
    let mut items = Vec::new();
    for pair in lines.chunks(2) {
        let (header, body) = match (serde_json::from_str::<Value>(pair[0]), serde_json::from_str::<Value>(pair[1])) {
            (Ok(h), Ok(b)) => (h, b),
            (Err(e), _) | (_, Err(e)) => return RestResp::error(400, "parse_exception", &e.to_string()),
        };
        let index = match header.get("index") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(","),
            _ => targets.to_string(),
        };
        items.push((index, body));
    }
    let query = &query;
    let futs: Vec<_> = items.into_iter().map(|(index, body)| Box::pin(async move {
        let response = match render_search_template(node, req, &body).await {
            Ok(rendered) => {
                let prefix=if index.is_empty(){""}else{"/"};
                let prepared = contexts::json_req_formatted(req, "POST", format_args!("{prefix}{index}/_search{query}"), &rendered).and_then(|mut sub| { sub.replace_header("content-type", "application/json")?; Ok(sub) });
                match prepared { Ok(sub) => Box::pin(read(node, &sub, &index)).await, Err(e) => err(e) }
            }
            Err(r) => r,
        };
        // Lossless and in order (sources with a lone surrogate, keyed
        // buckets): only `status` is set.
        std::str::from_utf8(&response.body).ok()
            .and_then(|s| elasticsearch_engine::source_json::with_member(s, "status", &response.status.to_string(), false))
            .unwrap_or_else(|| format!("{{\"status\":{}}}", response.status))
    })).collect();
    use futures::StreamExt;
    let responses: Vec<String> = futures::stream::iter(futs).buffered(msearch_concurrency(node, req)).collect().await;
    RestResp::new(200,vec![("content-type".into(), "application/json".into())],format!("{{\"took\":{},\"responses\":[{}]}}", started.elapsed().as_millis(), responses.join(",")).into_bytes())
}

/// Resolve alias routing before assigning explicit shard headers. Once a shard
/// header is set, shard-local execution cannot undo an over-broad fan-out.
/// Concrete-index and all-index requests retain the cheap metadata-only path.
fn alias_search_routes(node: &ClusterNode, req: &RestReq) -> Result<Option<HashMap<String, Option<String>>>, RestResp> {
    let segs = req.segments();
    let Some(expr) = segs.first().filter(|_| segs.len() > 1) else { return Ok(None) };
    if expr == "_all" || expr == "*" || node.engine.get_index(expr).is_some() { return Ok(None) }
    let params = req.params();
    let targets = node.engine.resolve(expr, ResolveOpts::from_params(&params))
        .map_err(|e| RestResp::json(e.status, &e.to_json_with_trace(false)))?;
    let targets = elasticsearch_http::apply_search_routing(targets, params.get("routing").map(String::as_str));
    Ok(Some(targets.into_iter().map(|t| (t.svc.name.clone(), t.routing)).collect()))
}

async fn multi_shard_read(node: &Arc<ClusterNode>, req: &RestReq, names: &BTreeSet<String>) -> Option<RestResp> {
    use crate::shard_engines as se;
    let segs = req.segments();
    let params = req.params();
    if segs.iter().any(|s| s == "_mget" || s == "_mtermvectors") {
        return fanout::multi_doc(node, req).await;
    }
    let last = segs.last().map(String::as_str).unwrap_or("");
    if last == "_msearch" {
        return msearch(node, req).await;
    }
    // `POST _async_search` runs the cluster search (DFS round included) in
    // the background; get / status / delete are context routes.
    if last == "_async_search" && req.method() != Method::POST {
        return None;
    }
    let single = if names.len() == 1 { se::doc_id_of(&segs).map(str::to_string) } else { None };
    let op = matches!(last, "_search" | "_count" | "_pit" | "_terms_enum" | "_async_search").then(|| last.to_string());
    if single.is_none() && op.is_none() {
        return None;
    }
    let alias_routes = if single.is_none() {
        match alias_search_routes(node, req) { Ok(routes) => routes, Err(e) => return Some(e) }
    } else { None };
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let st = node.state();
        let wanted: Vec<(String, u32)> = match &single {
            Some(id) => {
                let name = names.iter().next()?.clone();
                if let Some(e) = se::tsdb_doc_error(&st.metadata, &name, "get", params.get("routing").map(String::as_str)) {
                    return Some(RestResp::json(e.status, &e.to_json_with_trace(false)));
                }
                match se::doc_shard(&st.metadata, &name, id, params.get("routing").map(String::as_str)) {
                    Ok(k) => vec![(name, k)],
                    Err(_) => return Some(se::routing_missing(&name, id)),
                }
            }
            None => {
                let only = params.get("preference").and_then(|p| shards_preference(p));
                names.iter().flat_map(|n| {
                    let routing = match &alias_routes {
                        Some(routes) => match routes.get(n) { Some(r) => r.as_deref(), None => return Vec::new() },
                        None => params.get("routing").map(String::as_str),
                    };
                    se::search_shards(&st.metadata, n, routing).into_iter().map(|k| (n.clone(), k)).collect::<Vec<_>>()
                }).filter(|(_, k)| only.as_ref().map_or(true, |s| s.contains(k))).collect()
            }
        };
        if wanted.is_empty() && matches!(op.as_deref(), Some("_count") | Some("_search")) && !params.contains_key("scroll") {
            let merged = if op.as_deref() == Some("_count") { fanout::merge_count(&[]) } else { fanout::merge_search(&[], &req.json().unwrap_or(json!({})), &params) };
            return Some(RestResp::json(200, &merged));
        }
        let mut holders: std::collections::BTreeMap<String, BTreeSet<String>> = Default::default();
        let mut initializing = false;
        let mut missing: Option<(String, u32)> = None;
        for (name, k) in &wanted {
            let h: BTreeSet<String> = match st.routing_table.shard_table(name, *k) {
                Some(t) => {
                    initializing |= t.copies.iter().any(|c| c.state == ShardState::Initializing);
                    t.searchable_copies(&st.nodes).filter(|c| c.node_id.as_deref() != Some(node.local.id.as_str()) || local_holds(node, c)).filter_map(|c| c.node_id.clone()).collect()
                }
                None => BTreeSet::new(),
            };
            if h.is_empty() && missing.is_none() {
                missing = Some((name.clone(), *k));
            }
            holders.insert(format!("{name}#{k}"), h);
        }
        if let Some((name, k)) = missing {
            // Only a copy of a shard that has none active yet is waited for.
            let waiting = initializing && wanted.iter().any(|(n, k)| holders.get(&format!("{n}#{k}")).is_none_or(|h| h.is_empty())
                && st.routing_table.shard_table(n, *k).is_some_and(|t| t.copies.iter().any(|c| c.state == ShardState::Initializing)));
            if waiting && wait_state_change(node, deadline).await {
                continue;
            }
            // Stateless shards without a search copy fail like unassigned
            // shards: searches / counts run on the other shards and report
            // them as failed, single-document reads fail.
            let empty: Vec<(String, u32)> = wanted.iter().filter(|(n, k)| holders.get(&format!("{n}#{k}")).is_none_or(|h| h.is_empty())).cloned().collect();
            let stateless: Vec<(String, u32)> = empty.iter().filter(|(n, k)| st.routing_table.shard_table(n, *k).is_some_and(|t| t.stateless_without_search_copy(&st.nodes))).cloned().collect();
            if !stateless.is_empty() && stateless.len() == empty.len() {
                let partial = matches!(op.as_deref(), Some("_search") | Some("_count")) && !params.contains_key("scroll") && stateless.len() < wanted.len();
                if !partial || !copies::allow_partial(node, &params) {
                    return stateless_read::all_missing_response(req, &st, &stateless).or_else(|| Some(stateless_read::all_shards_failed(&stateless)));
                }
                holders.retain(|_, h| !h.is_empty());
                let r = multi_shard_fan_out(node, req, op.as_deref().unwrap_or("_search"), &holders).await;
                return Some(match r.body_json().filter(|_| r.is_success()) {
                    Some(v) => RestResp::json(r.status, &stateless_read::add_missing(v, &stateless)),
                    None => r,
                });
            }
            // ES `failOnUnavailable`: a search or count runs on the shards
            // that have a copy and reports the others failed (see
            // `shard_fan_out`); other reads fail.
            if !matches!(op.as_deref(), Some("_search") | Some("_count")) {
                return Some(RestResp::error(503, "no_shard_available_action_exception", &format!("[{name}][{k}] no active shard copy")));
            }
        }
        if single.is_some() {
            if let Some(r) = stateless_read::realtime_without_primary(req, &st, &wanted) {
                return Some(r);
            }
        }
        if single.is_some() {
            // ES `TransportSingleShardAction`: the shard's copies in
            // preference order, the next one on any failure.
            let (name, k) = &wanted[0];
            let key = format!("{name}#{k}");
            let mut sub = req.clone();
            if let Err(e) = sub.set_shard(*k).and_then(|_| sub.set_read_copies(std::slice::from_ref(name))) { return Some(err(e)); }
            let plan = match copies::Plan::new(node, req, &std::collections::BTreeMap::from([(key.clone(), holders[&key].clone())])) {
                Ok(p) => p,
                Err(r) => return Some(r),
            };
            let mut cands = plan.candidates(&key);
            // Without a preference the local copy answers when there is one.
            if !plan.pinned {
                if let Some(i) = cands.iter().position(|c| *c == node.local.id) {
                    let me = cands.remove(i);
                    cands.insert(0, me);
                }
            }
            if cands.is_empty() {
                return Some(RestResp::error(503, "no_shard_available_action_exception", &format!("[{name}][{k}] no active shard copy")));
            }
            return Some(on_shard_copies(node, &sub, cands, None).await.0);
        }
        let op = op.as_deref().unwrap_or("_search");
        if op == "_pit" {
            let plan = match copies::Plan::new(node, req, &holders) { Ok(p) => p, Err(r) => return Some(r) };
            return Some(contexts::open_pit(node, req, contexts::plan_groups(&plan), true, Some(&plan)).await);
        }
        if op == "_search" && params.contains_key("scroll") {
            let body = req.json().unwrap_or(json!({}));
            let mut p = params.clone();
            p.remove("scroll");
            // A sliced scroll is split per shard below (`SliceBuilder`).
            let mut checked = body.clone();
            if let Some(o) = checked.as_object_mut() {
                o.remove("slice");
            }
            if let Some(f) = fanout::unsupported_feature(&checked, &p) {
                return Some(RestResp::error(500, "illegal_state_exception", &format!("{f} cannot be combined across shards yet")));
            }
            if fanout::knn_needs_dfs(&checked, &p) {
                return Some(RestResp::error(500, "illegal_state_exception", "knn with a query or several knn clauses in a scroll cannot be combined across shards yet"));
            }
            // Over `index.max_slices_per_scroll` the shards get the request
            // as sent so the engine raises ES's error itself.
            let too_many = fanout::slice_spec(&body).is_some_and(|(_, max)| {
                names.iter().any(|n| {
                    let lim = st.metadata.indices.get(n).and_then(|im| im.setting("index.max_slices_per_scroll")).and_then(|v| v.parse::<u64>().ok()).unwrap_or(1024);
                    max > lim
                })
            });
            let counts: HashMap<String, u32> = if too_many { HashMap::new() } else { names.iter().map(|n| (n.clone(), se::shards_of(&st.metadata, n))).collect() };
            let plan = match copies::Plan::new(node, req, &holders) { Ok(p) => p, Err(r) => return Some(r) };
            return Some(contexts::start_scroll(node, req, contexts::plan_groups(&plan), true, &body, &params, &counts, Some(&plan)).await);
        }
        return Some(multi_shard_fan_out(node, req, op, &holders).await);
    }
}

/// `_search_shards`: the engine renders the target indices (aliases,
/// filters, routing → shard numbers); the copies and nodes come from the
/// cluster routing table.
async fn search_shards(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let r = local_canonical(node, req).await;
    if !r.is_success() {
        return r;
    }
    let Some(mut v) = r.body_json() else { return r };
    let st = node.state();
    let mut used: BTreeSet<String> = BTreeSet::new();
    let mut shards = Vec::new();
    // Engine 0 only knows shard 0 of each index: a multi-shard index is
    // re-expanded to every shard its `routing` covers (ES order: index,
    // then shard id).
    let routing = req.params().get("routing").cloned();
    let alias_routes = match alias_search_routes(node, req) { Ok(routes) => routes, Err(e) => return e };
    let mut targets: Vec<(Option<Value>, String, u32)> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for entry in v["shards"].as_array().cloned().unwrap_or_default() {
        let Some(index) = entry[0]["index"].as_str().map(str::to_string) else {
            targets.push((Some(entry), String::new(), 0));
            continue;
        };
        if crate::shard_engines::shards_of(&st.metadata, &index) > 1 {
            if seen.insert(index.clone()) {
                let selected_routing = match &alias_routes {
                    Some(routes) => match routes.get(&index) { Some(r) => r.as_deref(), None => continue },
                    None => routing.as_deref(),
                };
                for k in crate::shard_engines::search_shards(&st.metadata, &index, selected_routing) {
                    targets.push((None, index.clone(), k));
                }
            }
            continue;
        }
        let k = entry[0]["shard"].as_u64().unwrap_or(0) as u32;
        targets.push((Some(entry), index, k));
    }
    for (entry, index, k) in targets {
        let Some(t) = st.routing_table.shard_table(&index, k) else {
            if let Some(e) = entry {
                shards.push(e);
            }
            continue;
        };
        let mut copies: Vec<&elasticsearch_cluster::state::ShardRouting> = t.copies.iter().filter(|c| matches!(c.state, ShardState::Started | ShardState::Relocating | ShardState::Initializing) && c.node_id.is_some() && elasticsearch_cluster::state::copy_searchable(c, &st.nodes)).collect();
        copies.sort_by_key(|c| (!c.primary, c.state == ShardState::Initializing));
        let out: Vec<Value> = copies
            .into_iter()
            .map(|c| {
                if let Some(n) = &c.node_id {
                    used.insert(n.clone());
                }
                let mut j = crate::rest::cluster_state::shard_routing_json(c);
                j["relocation_failure_info"] = json!({"failed_attempts": 0});
                j
            })
            .collect();
        shards.push(Value::Array(out));
    }
    v["shards"] = Value::Array(shards);
    let nodes: serde_json::Map<String, Value> = st
        .nodes
        .values()
        .filter(|n| used.contains(&n.id))
        .map(|n| {
            (
                n.id.clone(),
                json!({
                    "name": n.name,
                    "ephemeral_id": n.ephemeral_id,
                    "transport_address": n.transport_address,
                    "external_id": n.name,
                    "attributes": n.attributes,
                    "roles": n.roles.iter().map(|r| r.name()).collect::<Vec<_>>(),
                    "version": n.version,
                    "min_index_version": elasticsearch_engine::version::IndexVersion::MINIMUM_COMPATIBLE.id(),
                    "max_index_version": elasticsearch_engine::version::IndexVersion::CURRENT.id(),
                }),
            )
        })
        .collect();
    v["nodes"] = Value::Object(nodes);
    RestResp::json(200, &v)
}

/// The `_search` a header line of `_msearch` stands for.
const MSEARCH_HEADER_PARAMS: &[&str] = &[
    "search_type", "preference", "routing", "request_cache", "allow_no_indices", "expand_wildcards", "ignore_unavailable", "ignore_throttled", "allow_partial_search_results", "ccs_minimize_roundtrips",
];

/// `_msearch` over indices with several shards or on several nodes: every
/// (header, body) pair runs as its own `_search` through `read`, so each
/// gets the multi-shard / cross-node treatment. `None` when the body is
/// not something to split (the engine then renders its error).
// Engine msearch failures that belong to the complete request must survive
// the batching optimization. They are not stale-copy retry signals. A breaker
// trip is request-wide only from this node's own batch (the coordinator's
// response accounting, as in ES). From another node it is a data-node
// failure, which ES retries on the next copy: those items run one by one.
fn msearch_request_error(response: &RestResp, batch: bool) -> bool {
    if response.status != 400 && !(batch && response.status == 429) { return false; }
    let Some(value) = response.body_json() else { return false };
    let error = &value["error"];
    (batch && response.status == 429 && error["type"] == "circuit_breaking_exception")
        || (response.status == 400 && error["type"] == "illegal_argument_exception"
            && error["reason"].as_str().is_some_and(|reason| reason.starts_with("[rest_total_hits_as_int] cannot be used")))
}

fn msearch<'a>(node: &'a Arc<ClusterNode>, req: &'a RestReq) -> futures::future::BoxFuture<'a, Option<RestResp>> {
    Box::pin(msearch_inner(node, req))
}

async fn msearch_inner(node: &Arc<ClusterNode>, req: &RestReq) -> Option<RestResp> {
    let started = std::time::Instant::now();
    let text = std::str::from_utf8(&req.body).ok()?;
    let mut lines: Vec<&str> = text.split('\n').map(|l| l.trim_end_matches('\r')).collect();
    while lines.last().map(|l| l.trim().is_empty()).unwrap_or(false) {
        lines.pop();
    }
    if lines.is_empty() || lines.len() % 2 != 0 {
        return None;
    }
    let segs = req.segments();
    let path_index = if segs.len() >= 2 && segs[1] == "_msearch" { Some(segs[0].clone()) } else { None };
    let base=||req.query().split('&').filter(|p|!p.is_empty()).filter_map(|p|{
        let (key,value)=p.split_once('=').unwrap_or((p,""));
        (!matches!(key,"max_concurrent_searches"|"index")).then_some((key,value))
    });
    let mut query_memory=match elasticsearch_engine::memory::breakers().reserve_request(0,"msearch query metadata"){Ok(memory)=>memory,Err(error)=>return Some(err(error))};
    let mut subs = Vec::new();
    for pair in lines.chunks(2) {
        let header: Value = if pair[0].trim().is_empty() { json!({}) } else { serde_json::from_str(pair[0]).ok()? };
        let header = header.as_object()?;
        let body: Value = serde_json::from_str(pair[1]).ok().or_else(||elasticsearch_engine::source_json::search_request(pair[1]).ok().map(|p|p.0))?;
        if !body.is_object() {
            return None;
        }
        if body.get("pit").is_some() && (header.contains_key("index") || header.contains_key("indices")) {
            let error = elasticsearch_engine::error::EsError::action_validation(&["[indices] cannot be used with point in time. Do not specify any index with point in time."]);
            return Some(RestResp::json(error.status, &error.to_json_with_trace(false)));
        }
        let prepared=(|| -> EsResult<_> {
            let count=base().count().checked_add(header.len()).ok_or_else(||elasticsearch_engine::error::EsError::new(429,"circuit_breaking_exception","msearch query capacity overflow"))?;
            let mut q=admitted::slots(&mut query_memory,count)?;
            for (key,value) in base(){q.push((admitted::text(&mut query_memory,key)?,admitted::text(&mut query_memory,value)?));}
            let mut index=path_index.as_deref().map(|s|admitted::text(&mut query_memory,s)).transpose()?;
            for (key,value) in header {
                let value=match value {
                    Value::String(value)=>admitted::text(&mut query_memory,value)?,
                    Value::Array(values)=>{
                        if !values.iter().all(Value::is_string){return Ok(None);}
                        let value=SharedText::try_format(format_args!("{}",UriJoin(values.iter().filter_map(Value::as_str),",")))?;
                        admitted::text(&mut query_memory,&value)?
                    },
                    Value::Bool(_) | Value::Number(_)=>{
                        let value=SharedText::try_format(format_args!("{value}"))?;
                        admitted::text(&mut query_memory,&value)?
                    },
                    _=>return Ok(None),
                };
                match key.as_str(){
                    "index"|"indices"=>index=Some(value),
                    key if MSEARCH_HEADER_PARAMS.contains(&key)=>{
                        q.retain(|(qk,_)|qk!=key);
                        q.push((admitted::text(&mut query_memory,key)?,value));
                    },
                    _=>return Ok(None),
                }
            }
            Ok(Some((q,index)))
        })();
        let (q,index)=match prepared {Ok(Some(value))=>value,Ok(None)=>return None,Err(error)=>return Some(err(error))};
        let query=contexts::QueryPairs(&q);
        let targets = index.unwrap_or_default();
        let prefix=if targets.is_empty() { "" } else { "/" };
        let separator=if q.is_empty() { "" } else { "?" };
        let batchable = !q.iter().any(|(k, _)| k == "preference") && body.get("pit").is_none();
        let mut sub=match contexts::json_req_formatted(req,"POST",format_args!("{prefix}{targets}/_search{separator}{query}"),&body) { Ok(sub)=>sub, Err(e)=>return Some(err(e)) };
        if let Err(e)=sub.set_body(pair[1].as_bytes()) { return Some(err(e)); }
        subs.push((sub,targets,batchable,(pair[0],pair[1])));
    }
    let render = |r: &RestResp| -> String {
        // Lossless and in order; an error keeps its own `status`.
        let status = r.status.to_string();
        std::str::from_utf8(&r.body).ok()
            .and_then(|s| elasticsearch_engine::source_json::with_member(s, "status", &status, !r.is_success()))
            .unwrap_or_else(|| format!(r#"{{"error":{{"type":"illegal_state_exception","reason":"unreadable response"}},"status":{status}}}"#))
    };
    let mut responses: Vec<Option<String>> = vec![None; subs.len()];
    let mut response_breaker = elasticsearch_http::MsearchBreaker::default();
    // Items over single-shard indices that one node holds readable copies of
    // run there in one engine `_msearch` (as a single-node cluster does);
    // the engine fails items one by one. If the batch fails as a whole (a
    // copy turned stale, the node left) its items run one by one below.
    // Batches run concurrently. A batch is only a faster way to read the
    // same copies. A failed batch/copy can retry, but elapsed time alone
    // must not launch duplicates of healthy items still executing remotely.
    // Item search timeouts remain enforced by the engine; node departure
    // remains bounded by fault detection.

    let batches = msearch_batches(node, &subs.iter().map(|(_, t, b, _)| (t.as_str(), *b)).collect::<Vec<_>>());
    // The msearch task (ES `indices:data/read/msearch`): its items run as
    // its child search tasks.
    let task = match MsearchTask::register(node, req) { Ok(task) => Some(task), Err(error) => return Some(err(error)) };
    let task = task.as_ref();
    let batch_runs = batches.into_iter().map(|(target, items)| {
        let subs = &subs;
        async move {
            let mut names: BTreeSet<String> = BTreeSet::new();
            for (_, item_names) in &items { names.extend(item_names.iter().cloned()); }
            let mut batch = req.try_with_encoded_body(|out| {
                for (i, _) in &items {
                    out.write_all(subs[*i].3 .0.as_bytes())?; out.write_all(b"\n")?;
                    out.write_all(subs[*i].3 .1.as_bytes())?; out.write_all(b"\n")?;
                }
                Ok(())
            }).map_err(err)?;
            batch.retain_headers(|(h, _)| !h.eq_ignore_ascii_case("content-length")).map_err(err)?;
            batch.set_read_copies(&names.into_iter().collect::<Vec<_>>()).map_err(err)?;
            if let Some(task) = task {
                batch.push_header(crate::rest::tasks::PARENT_HEADER, &task.id).map_err(err)?;
                task.running(&target, true);
            }
            let r = copy_request(node, &batch, &target, None).await;
            if let Some(task) = task { task.running(&target, false); }
            if msearch_request_error(&r, target == node.local.id) { return Err(r); }
            // Lossless, as `render` is: an item's sources may hold a lone
            // surrogate.
            use elasticsearch_engine::source_json::{raw_elements, raw_members};
            let responses = std::str::from_utf8(&r.body).ok().filter(|_| r.is_success()).and_then(raw_members)
                .and_then(|m| m.into_iter().find(|(k, _)| k == "responses")).and_then(|(_, a)| raw_elements(a.get()));
            let parts = responses.filter(|a| a.len() == items.len()).map(|a| a.iter().map(|part| {
                // Each item exactly as the engine wrote it (order, number
                // text and lone surrogates kept).
                let text = part.get();
                let status = raw_members(text).and_then(|m| m.into_iter().find(|(k, _)| k == "status")).and_then(|(_, v)| v.get().parse::<u16>().ok()).unwrap_or(200);
                // A copy failure (rejection, server error, missing copy)
                // is retried; a request error is the item's answer.
                let copy_failure = status == 429 || status >= 500 || (status == 404 && text.contains("shard_not_found_exception"));
                (!copy_failure).then(|| text.to_string())
            }).collect::<Vec<_>>());
            Ok(match parts {
                Some(parts) => items.iter().map(|(i, _)| *i).zip(parts).collect::<Vec<_>>(),
                None => Vec::new(),
            })
        }
    });
    let Some(batch_results) = until_cancelled(task, futures::future::join_all(batch_runs)).await else { return Some(msearch_cancelled()) };
    for result in batch_results {
        match result {
            Err(response) => return Some(response),
            Ok(parts) => for (i, part) in parts {
                if let Some(text) = part.as_deref() {
                    if let Err(error) = response_breaker.add(node.engine.breakers, text, responses.iter().filter_map(Option::as_deref)) {
                        return Some(RestResp::json(error.status, &error.to_json_with_trace(false)));
                    }
                }
                responses[i] = part;
            },
        }
    }
    use futures::StreamExt;
    let parent = task.map(|t| (t.id.as_str(), t.task.cancel_token().clone()));
    let futs: Vec<_> = subs.iter().enumerate().filter(|(i, _)| responses[*i].is_none())
        .map(|(i, (sub, targets, _, _))| {
            let (parent_id, parent_token) = (parent.as_ref().map(|p| p.0), parent.as_ref().map(|p| p.1.clone()));
            Box::pin(async move {
                (i, search_task::run(node, sub, search_task::SEARCH_ACTION, parent_id, parent_token, |r| async move { read(node, &r, targets).await }).await)
            })
        }).collect();
    let Some(item_results) = until_cancelled(task, futures::stream::iter(futs).buffered(msearch_concurrency(node, req)).collect::<Vec<_>>()).await else { return Some(msearch_cancelled()) };
    for (i, r) in item_results {
        if msearch_request_error(&r, false) { return Some(r); }
        let rendered = render(&r);
        if let Err(error) = response_breaker.add(node.engine.breakers, &rendered, responses.iter().filter_map(Option::as_deref)) {
            return Some(RestResp::json(error.status, &error.to_json_with_trace(false)));
        }
        responses[i] = Some(rendered);
    }
    let responses: Vec<String> = responses.into_iter().map(Option::unwrap_or_default).collect();
    Some(RestResp::new(200,vec![("content-type".into(),"application/json".into())],format!("{{\"took\":{},\"responses\":[{}]}}",started.elapsed().as_millis(),responses.join(",")).into_bytes()))
}

/// The coordinator's cancellable `indices:data/read/msearch` task while its
/// copy-local batches run (listed by `_tasks`, cancelled by `_tasks/_cancel`).
/// Each batch carries it as the parent task, and the node running a batch
/// stops at its next item once that parent is banned. When the coordinator
/// stops waiting (the client disconnected and this future was dropped, or
/// the task was cancelled) the nodes still running a batch are banned, as
/// ES bans the child tasks of a closed channel or a cancelled task. There is
/// no deadline: a slow healthy batch is never run a second time.
struct MsearchTask {
    node: Arc<ClusterNode>,
    task: Arc<elasticsearch_engine::tasks::Task>,
    id: String,
    running: parking_lot::Mutex<HashMap<String, usize>>,
}
impl MsearchTask {
    fn register(node: &Arc<ClusterNode>, req: &RestReq) -> EsResult<Self> {
        let opaque: Vec<(String, String)> = req.headers.iter().filter(|(k, _)| k.eq_ignore_ascii_case("x-opaque-id")).map(|(_, v)| ("X-Opaque-Id".to_string(), v.clone())).collect();
        let task = node.engine.tasks.try_register_borrowed("indices:data/read/msearch", None, true, false, &opaque, None)?;
        // Engine 0's task number is its cluster task id on this node.
        let id = format!("{}:{}", node.local.id, task.id);
        Ok(Self { node: node.clone(), task, id, running: parking_lot::Mutex::new(HashMap::new()) })
    }
    /// Track the batches outstanding on `target`.
    fn running(&self, target: &str, started: bool) {
        let mut running = self.running.lock();
        if started { *running.entry(target.to_string()).or_default() += 1; }
        else if let Some(n) = running.get_mut(target) {
            *n -= 1;
            if *n == 0 { running.remove(target); }
        }
    }
}
impl Drop for MsearchTask {
    fn drop(&mut self) {
        let outstanding: Vec<String> = std::mem::take(&mut *self.running.lock()).into_keys().collect();
        if !outstanding.is_empty() { crate::rest::tasks::ban_on(&self.node, outstanding, &self.id); }
        self.node.engine.tasks.finish(&self.task, Ok(Value::Null));
    }
}

/// `work`'s output, or `None` once the msearch task is cancelled.
async fn until_cancelled<F: std::future::Future>(task: Option<&MsearchTask>, work: F) -> Option<F::Output> {
    let Some(task) = task else { return Some(work.await) };
    tokio::pin!(work);
    let mut tick = tokio::time::interval(Duration::from_millis(50));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            output = &mut work => return Some(output),
            _ = tick.tick() => if task.task.is_cancelled() { return None },
        }
    }
}

fn msearch_cancelled() -> RestResp {
    RestResp::json(400, &json!({"error": {"root_cause": [{"type": "task_cancelled_exception", "reason": "task cancelled [by user request]"}], "type": "task_cancelled_exception", "reason": "task cancelled [by user request]"}, "status": 400}))
}

/// `_msearch` items (targets, batchable) that can run together on one node:
/// node → `(item, its indices)`, two items or more per node. An item is
/// batched when it names no preference or PIT, its targets are single-shard
/// indices and some node holds a readable copy of each (this node first).
fn msearch_batches(node: &ClusterNode, items: &[(&str, bool)]) -> Vec<(String, Vec<(usize, Vec<String>)>)> {
    let st = node.state();
    let mut by_node: std::collections::BTreeMap<String, Vec<(usize, Vec<String>)>> = Default::default();
    for (i, (targets, batchable)) in items.iter().enumerate() {
        // A cross-cluster item runs through `read`, which plans its remotes.
        if !batchable || elasticsearch_engine::ccr::remote::has_remote_expression(targets) {
            continue;
        }
        let Ok(resolved) = node.engine.resolve(targets, ResolveOpts::default()) else { continue };
        let names: Vec<String> = resolved.iter().map(|t| t.svc.name.clone()).collect::<BTreeSet<_>>().into_iter().collect();
        if names.is_empty() || names.iter().any(|n| crate::shard_engines::shards_of(&st.metadata, n) > 1) {
            continue;
        }
        let mut common: Option<BTreeSet<String>> = None;
        for n in &names {
            let h: BTreeSet<String> = st.routing_table.shard_table(n, 0).into_iter().flat_map(|t| t.searchable_copies(&st.nodes))
                .filter(|c| c.node_id.as_deref() != Some(node.local.id.as_str()) || local_holds(node, c))
                .filter_map(|c| c.node_id.clone()).collect();
            common = Some(match common { None => h, Some(c) => c.intersection(&h).cloned().collect() });
        }
        let common = common.unwrap_or_default();
        let target = if common.contains(&node.local.id) { Some(node.local.id.clone()) } else { common.into_iter().next() };
        if let Some(t) = target {
            by_node.entry(t).or_default().push((i, names));
        }
    }
    by_node.into_iter().filter(|(_, v)| v.len() > 1).collect()
}

/// A `retriever` / legacy `rank` search the coordinator runs in two phases.
fn is_rank_search(body: &Value, params: &HashMap<String, String>) -> bool {
    (body.get("retriever").is_some() || body.get("rank").is_some()) && body.get("pit").is_none() && !params.contains_key("scroll")
}

/// `retriever` / `rank` over several shards (ES runs compound retrievers on
/// the coordinator): this node's engine runs every child retriever as a
/// cluster search and fuses them into the body of the final search (a rank
/// plan keyed by `_id`, or a plain `query` / `knn` rewrite), which then runs
/// on every shard and merges like any search. Warnings of the first phase
/// (deprecated `rank`) belong to the response.
fn rank_search<'a>(node: &'a Arc<ClusterNode>, req: &'a RestReq) -> futures::future::BoxFuture<'a, RestResp> {
    Box::pin(rank_search_inner(node, req))
}

async fn rank_search_inner(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let mut plan_req = req.clone();
    if let Err(e) = plan_req.replace_header(RANK_PLAN_HEADER, "plan") { return err(e); }
    let planned = local_canonical(node, &plan_req).await;
    if !planned.is_success() {
        return planned;
    }
    let Some(final_body) = planned.body_json().and_then(|mut v| v.get_mut("_rustic_final_body").map(Value::take)) else {
        return planned;
    };
    let mut final_req = req.clone();
    if let Err(e) = final_req.set_json(&final_body) { return err(e); }
    if let Err(e) = final_req.retain_headers(|(k, _)| !k.eq_ignore_ascii_case(RANK_PLAN_HEADER) && !k.eq_ignore_ascii_case("content-length")) { return err(e); }
    if let Err(e) = final_req.push_header(RANK_PLAN_HEADER, "apply") { return err(e); }
    if !final_req.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("content-type")) {
        if let Err(e) = final_req.push_header("content-type", &req.json_content_type()) { return err(e); }
    }
    let mut resp = dispatch(node, final_req).await;
    for (k, v) in planned.headers.iter().filter(|(k, _)| k.eq_ignore_ascii_case("warning")) {
        if !resp.headers.iter().any(|(rk, rv)| rk.eq_ignore_ascii_case(k) && rv == v) {
            resp.push_header((k.clone(), v.clone()));
        }
    }
    resp
}

/// The number of shards the whole search covers, for the engines' can-match
/// pre-filter (`pre_filter_shard_size` compares against the search's shard
/// count, not the part one node runs); a forwarded value is replaced.
fn set_prefilter_count(headers: &mut Vec<(String, String)>, shards: usize) {
    headers.retain(|(k, _)| !k.eq_ignore_ascii_case(crate::shard_engines::PREFILTER_SHARDS_HEADER));
    headers.push((crate::shard_engines::PREFILTER_SHARDS_HEADER.into(), shards.to_string()));
}

/// `_search` / `_count` over every shard of the targets: one request per
/// (node, shard), each restricted to that shard's concrete indices, reduced
/// by `fanout` (each part's `_shards` is normalized to the shards it
/// covers, so totals add up). `holders` maps `"{index}#{shard}"` to the
/// nodes of its searchable copies; a shard with none is reported failed.
async fn multi_shard_fan_out(node: &Arc<ClusterNode>, req: &RestReq, op: &str, holders: &std::collections::BTreeMap<String, BTreeSet<String>>) -> RestResp {
    if op == "_async_search" {
        return async_search(node, req, holders).await;
    }
    let plan = match copies::Plan::new(node, req, holders) {
        Ok(p) => Arc::new(p),
        Err(r) => return r,
    };
    shard_fan_out(node, req, op, &plan).await
}

/// One shard request of a fan-out: shard `k` of `idx`, tried on `cands`.
struct ShardUnit {
    k: u32,
    idx: Vec<String>,
    cands: Vec<String>,
}

/// What became of a [`ShardUnit`].
enum UnitResult {
    /// `parsed`: the answer's body as [`parse_search_part`] reads it, when
    /// the task that received it parsed it already.
    Answer { k: u32, idx: Vec<String>, node: String, resp: RestResp, parsed: Option<Box<ParsedPart>> },
    Failed(Vec<copies::ShardFailure>),
    /// A request error: the search's answer as it is.
    Request(RestResp),
}

/// ES `AbstractSearchAsyncAction` over `plan` (see [`multi_shard_fan_out`]):
/// every shard request goes to its first copy and, on any failure but a
/// request error, to the next one. A shard no copy answered is a shard
/// failure: the response is partial (`_shards.failures`), or ES's
/// `search_phase_execution_exception` when every shard failed or
/// `allow_partial_search_results=false`. Every phase of one search (DFS,
/// kNN, expand) runs on the same `plan`, so each shard is read from one copy.
/// Tests only: answer large hits in a single round as before
/// ([`large_hits`]), to compare the two.
#[doc(hidden)]
pub static LARGE_HITS_SINGLE_ROUND: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Hits whose `_source` averages at least this many bytes make a page
/// large enough to fetch only its winners. Measured (c30, bench box): at
/// ~10 KB a single round still wins (pmc_tail_latency, 2 shards: 1.47 vs
/// 1.68 ms p50 for term), at 50 KB fetching the winners halves the
/// coordinator's fan-in alone (coordinator_merge_microbench: 955 vs 503 us)
/// before the shards' saved rendering; the difference changes sign near
/// 24 KB.
const LARGE_HIT_SOURCE_BYTES: u64 = 24 << 10;

/// Whether a search of several shards returns large hits (a target copy's
/// sampled `_source` averages [`LARGE_HIT_SOURCE_BYTES`], `_source` not
/// turned off): it then runs as query_then_fetch, as ES does, fetching only
/// the merged page's winners (Rally pmc: tens of KB of `_source` per hit,
/// two shards rendering ten each for a page of ten).
fn large_hits(node: &Arc<ClusterNode>, body: &Value, units: &[ShardUnit]) -> bool {
    if units.len() < 2 || body.get("_source") == Some(&Value::Bool(false)) || LARGE_HITS_SINGLE_ROUND.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    units.iter().any(|u| u.idx.iter().any(|i| {
        node.engine_for(u.k).and_then(|e| e.get_index(i)).and_then(|svc| svc.shard(u.k))
            .and_then(|copy| copy.shard.sampled_source_bytes()).is_some_and(|b| b >= LARGE_HIT_SOURCE_BYTES)
    }))
}

async fn shard_fan_out(node: &Arc<ClusterNode>, req: &RestReq, op: &str, plan: &Arc<copies::Plan>) -> RestResp {
    let started = std::time::Instant::now();
    let params = req.params();
    // Borrowed: copying a large body (`large_terms`, ~1 MB) costs as much
    // as parsing it, and nothing here changes it.
    let empty = json!({});
    let body: &Value = req.json_ref().unwrap_or(&empty);
    if op == "_search" && req.dfs != Dfs::Collect && is_rank_search(body, &params) {
        return rank_search(node, req).await;
    }
    // A stats round (`Dfs::Collect`) only needs the shards' statistics.
    if op == "_search" && req.dfs != Dfs::Collect {
        if let Some(f) = fanout::unsupported_feature(body, &params) {
            return RestResp::error(500, "illegal_state_exception", &format!("{f} cannot be combined across shards yet"));
        }
        if let Some(r) = collapse_expand(req, |sub| shard_search(node, sub, plan)).await {
            return r;
        }
        match knn_dfs(req, |sub| shard_search(node, sub, plan)).await {
            Err(r) => return r,
            Ok(Some(rewritten)) => return match force_dfs(rewritten) { Ok(rewritten) => shard_search(node, rewritten, plan).await, Err(e) => err(e) },
            Ok(None) => {}
        }
    }
    let partial_allowed = copies::allow_partial(node, &params);
    let bound = copies::copy_bound(&params, body);
    let (from, size) = fanout::from_size(body, &params);
    // Borrow filtered query components; final joining is admitted before growth.
    // Preserve empty components for terms-enum, as the old first pass did.
    let parts=req.query().split('&').filter(|p| (op=="_terms_enum"||(!p.is_empty()&&!p.starts_with("from=")&&!p.starts_with("size=")&&!is_shards_preference(p)))
        &&!RENDER_PARAMS.contains(&p.split('=').next().unwrap_or("")));
    // A kNN search whose page comes from the URL (`?size=`, as Rally sends
    // it) and whose body names no `from`, `size` or aggregations: the
    // shards get the window as `size=` on their URL and the original body
    // bytes, instead of the body re-serialized with `"from":0,"size":N`
    // (the engine reads a URL `size` as the body's, and with no `from` the
    // shard's is 0). Such a body never runs query_then_fetch (`knn` is not
    // one of its body keys), whose fetch phase sets `size` in the body.
    let url_window = (op == "_search" && body.get("knn").is_some() && body.is_object() && req.json_ref().is_some()
        && ["from", "size", "aggs", "aggregations", "retriever", "rank"].iter().all(|k| body.get(*k).is_none()))
        .then(|| from.saturating_add(size));
    let query=match match url_window {
        Some(w) => SharedText::try_format(format_args!("{}{}size={w}",UriJoin(parts.clone(),"&"),if parts.clone().next().is_some(){"&"}else{""})),
        None => SharedText::try_format(format_args!("{}",UriJoin(parts,"&"))),
    }{Ok(query)=>query,Err(error)=>return err(error)};
    let n_shards = plan.copies.len() + plan.unavailable.len();
    // Every shard sub-request carries the same body: build it once. When
    // the per-node rewrite would not change it (no `from`, `size` already
    // the window, no aggregations), forward the original bytes: a kNN body
    // is thousands of floats, and re-parsing and re-serializing it per
    // shard cost more than the shard's graph search.
    let shard_body: Option<Vec<u8>> = if op == "_search" {
        let window = from.saturating_add(size) as u64;
        let unchanged = url_window.is_some() || body.get("from").is_none_or(|f| f.as_u64() == Some(0))
            && match body.get("size") {
                Some(s) => s.as_u64() == Some(window),
                // No `size` anywhere: the shard's default page is the
                // window (`fanout::node_body` would only add `"from":0,
                // "size":10`). Not for a body the coordinator could not
                // read (it searched `{}` then) nor for retrievers (their
                // validation reads whether `size` was given).
                None => from == 0 && window == 10 && body.is_object() && req.json_ref().is_some()
                    && body.get("retriever").is_none() && body.get("rank").is_none(),
            }
            && body.get("aggs").is_none() && body.get("aggregations").is_none();
        if unchanged { None } else { Some(elasticsearch_engine::source_json::search_body(&req.body, &fanout::node_body(body, from, size))) }
    } else { None };
    // What every shard sub-request shares, set once (each change rebuilds
    // the request's metadata, and setting the body copies it): only the
    // URI, shard and copies differ per shard. The original body is shared,
    // not copied.
    let template = if op == "_search" {
        let built = (|| -> EsResult<RestReq> {
            let mut t = match &shard_body { Some(b) => req.try_with_body(b)?, None => req.clone() };
            t.replace_header(crate::shard_engines::PREFILTER_SHARDS_HEADER, &n_shards.to_string())?;
            t.set_method("POST")?;
            if !t.headers.iter().any(|(h, _)| h.eq_ignore_ascii_case("content-type")) {
                t.push_header("content-type", &req.json_content_type())?;
            }
            Ok(t)
        })();
        match built { Ok(t) => t, Err(e) => return err(e) }
    } else { req.clone() };
    let make = |idx: &[String], k: u32| -> EsResult<RestReq> {
        let mut sub = template.clone();
        sub.set_shard_target(&shard_read_uri(node, req, idx, op, &query)?, k, idx)?;
        Ok(sub)
    };
    // One request per (first copy, shard); its fallbacks are the other
    // nodes, in the first key's order, holding a copy of every key.
    let mut groups: std::collections::BTreeMap<(String, u32), Vec<String>> = Default::default();
    for key in plan.copies.keys() {
        let (Some((idx, k)), Some(first)) = (copies::split_key(key), plan.candidates(key).first().cloned()) else { continue };
        groups.entry((first, k)).or_default().push(idx);
    }
    let mut units: Vec<ShardUnit> = groups.into_iter().map(|((n, k), idx)| {
        let lists: Vec<Vec<String>> = idx.iter().map(|i| plan.candidates(&format!("{i}#{k}"))).collect();
        let mut cands = vec![n.clone()];
        cands.extend(lists[0].iter().filter(|c| **c != n && lists.iter().all(|l| l.contains(*c))).cloned());
        ShardUnit { k, idx, cands }
    }).collect();
    let mut failures: Vec<copies::ShardFailure> = plan.unavailable.iter().map(|(i, k)| copies::ShardFailure::unavailable(i, *k)).collect();
    // ES's DFS phase: collect every shard's term statistics, merge them, and
    // score each shard with the global ones, on the copy that gave them.
    let mut dfs = Dfs::None;
    if op == "_search" && (req.dfs == Dfs::Collect || dfs_wanted(req, body, &params, units.len())) {
        let collect = |idx: &[String], k: u32| dfs_collect_req(&make(idx, k)?);
        let round = futures::future::join_all(units.iter().map(|u| run_unit(node, plan, u, &collect, bound))).await;
        let mut stats = Vec::new();
        let mut answered = Vec::new();
        for r in round.into_iter().flatten() {
            match r {
                UnitResult::Request(r) => return r,
                UnitResult::Failed(f) => failures.extend(f),
                UnitResult::Answer { k, idx, node: n, resp, .. } if !resp.is_success() => failures.extend(copies::ShardFailure::from_response(&resp, &idx, k, &n)),
                UnitResult::Answer { k, idx, node: n, resp, .. } => match serde_json::from_slice::<elasticsearch_http::DfsStats>(&resp.body) {
                    Ok(s) => {
                        stats.push(s);
                        answered.push(ShardUnit { k, idx, cands: vec![n] });
                    }
                    Err(e) => return RestResp::error(500, "illegal_state_exception", &format!("unreadable dfs phase result: {e}")),
                },
            }
        }
        if answered.is_empty() && !failures.is_empty() {
            return copies::phase_failure("dfs", "all shards failed", &failures);
        }
        let merged = elasticsearch_http::merge_dfs(&stats);
        if req.dfs == Dfs::Collect {
            return RestResp::json(200, &serde_json::to_value(&merged).unwrap_or(Value::Null));
        }
        dfs = match SharedDfsStats::try_new(&merged) { Ok(stats) => Dfs::Use(stats), Err(e) => return err(e) };
        units = answered;
    }
    let run = |idx: &[String], k: u32| -> EsResult<RestReq> {
        let mut sub = make(idx, k)?;
        // (Rebuilding the request is not free: only a DFS round sets it.)
        if sub.dfs != dfs { sub.set_dfs(dfs.clone())?; }
        Ok(sub)
    };
    // ES query_then_fetch over the same plan and shard requests (after a
    // DFS round, each on the copy that answered it). A small page whose
    // every shard copy is on this node answers in one round instead
    // (`qtf::single_round`): the same response without a fetch round.
    let mut single_round = false;
    if op == "_search" && qtf::eligible(req, body, &params, units.len()) {
        let all_local = units.iter().all(|u| u.idx.iter().all(|i| plan.candidates(&format!("{i}#{}", u.k)).iter().all(|c| *c == node.local.id)));
        single_round = qtf::single_round(body, &params, all_local) && !large_hits(node, body, &units);
        if !single_round {
            let pins = plan.pins();
            if let Some(r) = qtf::run(node, req, body, &params, plan, &units, &run, &failures, started).await {
                return r;
            }
            // The single-round fallback starts from the pins before qtf ran.
            plan.restore_pins(pins);
        }
    }
    let round = |idx: &[String], k: u32| -> EsResult<RestReq> {
        let mut sub = run(idx, k)?;
        if single_round {
            sub.replace_header(qtf::HEADER, qtf::SINGLE_ROUND)?;
        }
        Ok(sub)
    };
    // Each answer is parsed on the task that received it, in parallel,
    // unless the merge may be memoized from the raw answers (below).
    let parse = !(op == "_search" && !single_round && matches!(dfs, Dfs::None) && size == 0 && failures.is_empty() && body.get("profile").is_none());
    let setup_us = started.elapsed().as_micros();
    let results: Vec<UnitResult> = run_units_concurrently(node, plan, &units, &round, bound, parse).await;
    let shards_us = started.elapsed().as_micros();
    // A `size: 0` search whose every shard answered as before (e.g. from
    // its request cache): the merge memoized for exactly these answers.
    let memo_key = if op == "_search" && !single_round && matches!(dfs, Dfs::None) && size == 0 && failures.is_empty() && body.get("profile").is_none() {
        // Each answer is labelled with its node and the indices it answered
        // for: the merge counts a unit's shards by its indices
        // (`normalize_part_shards`), so two plans whose answers are the same
        // bytes but cover different indices must not share a merge.
        let labels: Vec<String> = results.iter().map(|r| match r {
            UnitResult::Answer { node: n, idx, .. } => format!("{n}/{}", idx.join(",")),
            _ => String::new(),
        }).collect();
        let mut inputs: Vec<(u32, &str, &[u8])> = Vec::with_capacity(results.len());
        let mut complete = !results.is_empty();
        for (r, label) in results.iter().zip(&labels) {
            match r {
                UnitResult::Answer { k, resp, .. } if resp.is_success() && !resp.headers.iter().any(|(h, _)| h.eq_ignore_ascii_case("warning")) => inputs.push((*k, label.as_str(), &resp.body[..])),
                _ => { complete = false; break; }
            }
        }
        if complete { merge_memo::key(req.query(), &req.body[..], node.engine.max_buckets(), &mut inputs) } else { None }
    } else { None };
    if let Some(key) = &memo_key {
        if let Some(bytes) = merge_memo::get(key, started.elapsed().as_millis() as u64) {
            let mut raw = RestResp::json(200, &Value::Null);
            raw.replace_body_vec(bytes);
            return crate::rest::util::postprocess(req, raw);
        }
    }
    let mut sources=elasticsearch_engine::source_json::ResponseSources::default();
    let partial_memory=elasticsearch_engine::memory::RequestMemory::new();
    let mut parts = Vec::new();
    let mut warnings = BTreeSet::new();
    // A single round's answers, by position (its hits' tag): shard, the
    // copy that answered, the fetch time the shard left to count.
    let mut rounds: Vec<qtf::RoundAnswer> = Vec::new();
    // Tables and charges of the answers parsed on their own tasks.
    let mut part_memory = Vec::new();
    for r in results {
        let (k, idx, answered, r, parsed) = match r {
            UnitResult::Request(r) => return r,
            UnitResult::Failed(f) => {
                failures.extend(f);
                continue;
            }
            UnitResult::Answer { k, idx, node, resp, parsed } if resp.is_success() => (k, idx, node, resp, parsed),
            UnitResult::Answer { k, idx, node, resp, .. } => {
                failures.extend(copies::ShardFailure::from_response(&resp, &idx, k, &node));
                continue;
            }
        };
        let parsed = match parsed {
            Some(p) => {
                let ParsedPart { value, sources: table, memory } = *p;
                sources.absorb(table);
                part_memory.push(memory);
                value
            }
            None => parse_search_part(&mut sources, &r.body, &partial_memory),
        };
        let mut part=match parsed{Ok(v)=>v,Err(error)=>return RestResp::json(error.status,&error.to_json())};
        if single_round {
            rounds.push(qtf::round_answer(&mut part, rounds.len(), k, &answered));
        }
        fanout::normalize_part_shards(&mut part, idx.len());
        label_part_failures(&mut part, k, &idx, &answered);
        label_explained_hits(&mut part, k, &answered);
        warnings.extend(r.headers.iter().filter(|(name, _)| name.eq_ignore_ascii_case("warning")).map(|(_, value)| value.clone()));
        // `get_mut`, not `part["profile"]`: indexing mutably would insert a
        // `profile` into every part (and the merged response).
        for shard in part.get_mut("profile").and_then(|p| p.get_mut("shards")).and_then(Value::as_array_mut).into_iter().flatten() {
            shard["node_id"] = json!(answered);
            shard["shard_id"] = json!(k);
            if let Some(index) = shard["index"].as_str() { shard["id"] = json!(format!("[{answered}][{index}][{k}]")); }
        }
        parts.push((k, part));
    }
    if !failures.is_empty() {
        let only_missing = failures.len() == plan.unavailable.len();
        if parts.is_empty() {
            return copies::phase_failure("query", "all shards failed", &failures);
        }
        if !partial_allowed {
            if only_missing {
                return stateless_read::missing_shards_rejected(&plan.unavailable);
            }
            return copies::phase_failure("query", "Partial shards failure", &failures);
        }
    }
    // Shard order, not node order: the merge's stable sort then breaks
    // ties by shard as ES does.
    let parsed_us = started.elapsed().as_micros();
    parts.sort_by_key(|(k, _)| *k);
    let parts: Vec<Value> = parts.into_iter().map(|(_, p)| p).collect();
    let merged = match op {
        "_count" => fanout::merge_count(&parts),
        "_terms_enum" => merge_terms_enum(&parts, body, &params),
        _ => fanout::merge_search_with_source(&parts, body, &params, node.engine.max_buckets(), &req.body),
    };
    if let Some(e) = fanout::take_agg_breaking() {
        return RestResp::json(e.status, &e.to_json());
    }
    if let Some(e) = fanout::take_agg_error() {
        return RestResp::json(e.status, &e.to_json());
    }
    let mut merged = merged;
    if single_round {
        // Untag the page; count the fetch phase on the shards it came from.
        qtf::finish_single_round(node, &mut merged, &rounds);
    }
    copies::add_failures(&mut merged, &failures);
    // `took`: the coordinating node's time for the whole request (ES
    // `SearchResponse.took`), not the slowest shard's query time.
    if merged.get("took").is_some() {
        merged["took"] = json!(started.elapsed().as_millis() as u64);
    }
    let merged_us = started.elapsed().as_micros();
    let rendered = sources.render(&merged);
    if let (Some(key), true) = (memo_key, failures.is_empty() && warnings.is_empty()) {
        merge_memo::put(key, &rendered);
    }
    let mut raw=RestResp::json(200,&Value::Null);raw.replace_body_vec(rendered);
    let mut response = crate::rest::util::postprocess(req, raw);
    response.extend_headers(warnings.into_iter().map(|value| ("Warning".into(), value)));
    let total_us = started.elapsed().as_micros();
    // Coordinator phases of a fanned-out search (`fixed_cost_latency`):
    // planning and the single-round choice, the shards' answers, their
    // parse, the merge and the render.
    stats_phases::record(if single_round { "search_single_round" } else { "search_fan_out" }, vec![("setup", setup_us), ("shards", shards_us - setup_us),
        ("parse", parsed_us - shards_us), ("merge", merged_us - parsed_us), ("render", total_us - merged_us), ("total", total_us)]);
    response
}

/// Keep admitted decode/scratch storage alive through fan-out reduction.
/// Raw source/script payloads still go through the lossless response parser.
fn parse_search_part(sources:&mut elasticsearch_engine::source_json::ResponseSources,bytes:&[u8],memory:&elasticsearch_engine::memory::RequestMemory)->elasticsearch_engine::error::Result<Value> {
    memory.charge((bytes.len() as u64).saturating_mul(128).saturating_add(4096),"<search_partial_decode>")?;
    sources.parse_lazy(bytes).map_err(|error|elasticsearch_engine::error::EsError::internal(format!("invalid shard response: {error}")))
}

/// A part's own shard failures (one engine searching several indices)
/// name the cluster shard and the answering node, not the engine's.
fn label_part_failures(part: &mut Value, k: u32, idx: &[String], node: &str) {
    for f in part.get_mut("_shards").and_then(|s| s.get_mut("failures")).and_then(Value::as_array_mut).into_iter().flatten() {
        f["shard"] = json!(k);
        f["node"] = json!(node);
        if f["index"].is_null() {
            f["index"] = json!(idx.first());
        }
    }
}

/// Run one shard request (`make` builds it for some indices and a shard)
/// on its copies in order. A request over several indices that no
/// candidate answered is split: each index then goes to its own copies
/// (ES addresses every shard on its own, so no one node needs to hold all
/// of them), skipping nodes that failed as nodes.
/// [`run_unit`] of every unit, each first try on a task of its own (with
/// the request's memory scope, admission policy and cluster task): one
/// task polling them all ran their synchronous parts (building each
/// sub-request, the engine's request handling up to its executor) one
/// after another, so a shard whose search had long finished waited for the
/// others' preparation before its answer was taken (fixed_cost_latency:
/// ~120 us of a 5-shard search's ~400 us shard phase). The rarer retry of
/// a failed multi-index unit index by index runs here, as before.
async fn run_units_concurrently(node: &Arc<ClusterNode>, plan: &Arc<copies::Plan>, units: &[ShardUnit], make: &(dyn Fn(&[String], u32) -> EsResult<RestReq> + Sync), bound: Option<Duration>, parse: bool) -> Vec<UnitResult> {
    if units.len() <= 1 {
        return futures::future::join_all(units.iter().map(|u| run_unit(node, plan, u, make, bound))).await.into_iter().flatten().collect();
    }
    let mut handles = Vec::with_capacity(units.len());
    for (i, unit) in units.iter().enumerate() {
        let request = match make(unit.idx.as_slice(), unit.k) { Ok(req) => req, Err(e) => return vec![UnitResult::Request(err(e))] };
        let (node, cands) = (node.clone(), unit.cands.clone());
        let spawned = std::time::Instant::now();
        handles.push(spawn_in_request(async move {
            let start = spawned.elapsed().as_micros();
            let tried = try_copies(&node, &request, &cands, bound).await;
            let ran = spawned.elapsed().as_micros();
            let parsed = match &tried {
                Tried::Answered(r, _) if parse && r.is_success() => Some(Box::new(ParsedPart::parse(i as u32, &r.body))),
                _ => None,
            };
            let done = spawned.elapsed().as_micros();
            if stats_phases::ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
                stats_phases::record("unit_task", vec![("start", start), ("call", ran - start), ("parse", done - ran), ("end", done)]);
            }
            (tried, parsed)
        }));
    }
    let join_started = std::time::Instant::now();
    let tried = futures::future::join_all(handles).await;
    if stats_phases::ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
        stats_phases::record("unit_join", vec![("join", join_started.elapsed().as_micros())]);
    }
    let mut out = Vec::with_capacity(units.len());
    for (unit, t) in units.iter().zip(tried) {
        let (tried, parsed) = match t {
            Ok(t) => t,
            Err(e) => return vec![UnitResult::Request(RestResp::error(500, "exception", &format!("shard request task failed: {e}")))],
        };
        out.extend(unit_result(node, plan, unit, make, bound, tried, parsed).await);
    }
    out
}

/// A shard's search answer parsed on the task that received it
/// ([`parse_search_part`] with a table of its own, joined to the merge's).
struct ParsedPart {
    value: EsResult<Value>,
    sources: elasticsearch_engine::source_json::ResponseSources,
    /// The parse's charge, held until the merge is done.
    memory: elasticsearch_engine::memory::RequestMemory,
}

impl ParsedPart {
    fn parse(part: u32, body: &[u8]) -> Self {
        let (mut sources, memory) = (elasticsearch_engine::source_json::ResponseSources::for_part(part), elasticsearch_engine::memory::RequestMemory::new());
        let value = parse_search_part(&mut sources, body, &memory);
        ParsedPart { value, sources, memory }
    }
}

/// Spawn `work` (part of the calling request: a shard call) on a task of
/// its own, carrying the request's memory scope, admission policy and
/// cluster task, so a fan-out's calls run their synchronous parts in
/// parallel instead of one after another in the coordinator's task.
pub(super) fn spawn_in_request<F>(work: F) -> tokio::task::JoinHandle<F::Output>
where
    F: std::future::Future + Send + 'static,
    F::Output: Send + 'static,
{
    let (scope, native, task) = (elasticsearch_engine::memory::current_scope(), elasticsearch_engine::memory::current_native_policy(), bg_task::current());
    tokio::spawn(bg_task::under(task, elasticsearch_engine::memory::request_context_future(scope, native, work)))
}

/// What [`run_unit`] makes of the first try of `unit` on its copies.
async fn unit_result(node: &Arc<ClusterNode>, plan: &copies::Plan, unit: &ShardUnit, make: &(dyn Fn(&[String], u32) -> EsResult<RestReq> + Sync), bound: Option<Duration>, tried: Tried, parsed: Option<Box<ParsedPart>>) -> Vec<UnitResult> {
    let keys = |idx: &[String]| idx.iter().map(|i| format!("{i}#{}", unit.k)).collect::<Vec<_>>();
    let answer = |idx: Vec<String>, n: String, resp: RestResp| {
        for key in keys(idx.as_slice()) {
            plan.pin(&key, &n);
        }
        UnitResult::Answer { k: unit.k, idx, node: n, resp, parsed: None }
    };
    let answer = &answer;
    match tried {
        Tried::Answered(r, n) => {
            let mut a = answer(unit.idx.clone(), n, r);
            if let UnitResult::Answer { parsed: slot, .. } = &mut a {
                *slot = parsed;
            }
            vec![a]
        }
        Tried::Request(r) => vec![UnitResult::Request(r)],
        Tried::Failed { resp, node: n, dead } if unit.idx.len() > 1 => {
            let singles = unit.idx.iter().map(|i| {
                let cands: Vec<String> = plan.candidates(&format!("{i}#{}", unit.k)).into_iter().filter(|c| !dead.contains(c)).collect();
                let (resp, n) = (&resp, &n);
                async move {
                    let one = std::slice::from_ref(i);
                    if cands.is_empty() {
                        return UnitResult::Failed(copies::ShardFailure::from_response(resp, one, unit.k, n));
                    }
                    let request = match make(one, unit.k) { Ok(req) => req, Err(e) => return UnitResult::Request(err(e)) };
                    match try_copies(node, &request, &cands, bound).await {
                        Tried::Answered(r, n) => answer(vec![i.clone()], n, r),
                        Tried::Request(r) => UnitResult::Request(r),
                        Tried::Failed { resp, node: n, .. } => UnitResult::Failed(copies::ShardFailure::from_response(&resp, one, unit.k, &n)),
                    }
                }
            });
            futures::future::join_all(singles).await
        }
        Tried::Failed { resp, node: n, .. } => vec![UnitResult::Failed(copies::ShardFailure::from_response(&resp, &unit.idx, unit.k, &n))],
    }
}

async fn run_unit(node: &Arc<ClusterNode>, plan: &copies::Plan, unit: &ShardUnit, make: &(dyn Fn(&[String], u32) -> EsResult<RestReq> + Sync), bound: Option<Duration>) -> Vec<UnitResult> {
    let keys = |idx: &[String]| idx.iter().map(|i| format!("{i}#{}", unit.k)).collect::<Vec<_>>();
    let answer = |idx: Vec<String>, n: String, resp: RestResp| {
        for key in keys(idx.as_slice()) {
            plan.pin(&key, &n);
        }
        UnitResult::Answer { k: unit.k, idx, node: n, resp, parsed: None }
    };
    let answer = &answer;
    let request = match make(unit.idx.as_slice(), unit.k) { Ok(req) => req, Err(e) => return vec![UnitResult::Request(err(e))] };
    match try_copies(node, &request, &unit.cands, bound).await {
        Tried::Answered(r, n) => vec![answer(unit.idx.clone(), n, r)],
        Tried::Request(r) => vec![UnitResult::Request(r)],
        Tried::Failed { resp, node: n, dead } if unit.idx.len() > 1 => {
            let singles = unit.idx.iter().map(|i| {
                let cands: Vec<String> = plan.candidates(&format!("{i}#{}", unit.k)).into_iter().filter(|c| !dead.contains(c)).collect();
                let (resp, n) = (&resp, &n);
                async move {
                    let one = std::slice::from_ref(i);
                    if cands.is_empty() {
                        return UnitResult::Failed(copies::ShardFailure::from_response(resp, one, unit.k, n));
                    }
                    let request = match make(one, unit.k) { Ok(req) => req, Err(e) => return UnitResult::Request(err(e)) };
                    match try_copies(node, &request, &cands, bound).await {
                        Tried::Answered(r, n) => answer(vec![i.clone()], n, r),
                        Tried::Request(r) => UnitResult::Request(r),
                        Tried::Failed { resp, node: n, .. } => UnitResult::Failed(copies::ShardFailure::from_response(&resp, one, unit.k, &n)),
                    }
                }
            });
            futures::future::join_all(singles).await
        }
        Tried::Failed { resp, node: n, .. } => vec![UnitResult::Failed(copies::ShardFailure::from_response(&resp, &unit.idx, unit.k, &n))],
    }
}

/// The outcome of trying a shard request on its copies.
enum Tried {
    Answered(RestResp, String),
    Request(RestResp),
    /// No copy answered: the most meaningful failure, the node that gave
    /// it, and the nodes that failed as nodes (transport failures).
    Failed { resp: RestResp, node: String, dead: Vec<String> },
}

/// `sub` on each of `cands` in turn until one answers, or fails with a
/// request error (which another copy would only repeat).
async fn try_copies(node: &Arc<ClusterNode>, sub: &RestReq, cands: &[String], bound: Option<Duration>) -> Tried {
    let mut best: Option<(RestResp, String)> = None;
    let mut dead = Vec::new();
    let query = is_query_phase(sub);
    for c in cands {
        let r = if query { copy_query_request(node, sub, c, bound).await } else { copy_request(node, sub, c, bound).await };
        if copies::answered(&r) {
            return Tried::Answered(r, c.clone());
        }
        if copies::is_request_error(&r) {
            return Tried::Request(r);
        }
        if copies::node_failure(&r) {
            dead.push(c.clone());
        }
        if best.as_ref().map_or(true, |(prev, _)| copies::replaces(prev, &r)) {
            best = Some((r, c.clone()));
        }
    }
    let (resp, node) = best.unwrap_or_else(|| (RestResp::error(503, "no_shard_available_action_exception", "no active shard copy"), String::new()));
    Tried::Failed { resp, node, dead }
}

/// A shard `_search` / `_count` request's query phase (not a DFS stats
/// round): its answers feed adaptive replica selection.
fn is_query_phase(sub: &RestReq) -> bool {
    sub.dfs != Dfs::Collect && sub.segments().last().is_some_and(|op| op == "_search" || op == "_count")
}

/// `sub` on copy holder `c`: this node's engine, else forwarded for local
/// execution there. A forwarded request gives up when `c` leaves the
/// cluster (fault detection) or after `bound`, so a hung node delays a
/// search by those, not by the transport's forwarding deadline.
/// Counted as an outstanding request to `c` while it runs (adaptive
/// replica selection); only a query phase ([`copy_query_request`]) feeds
/// its statistics.
async fn copy_request(node: &Arc<ClusterNode>, sub: &RestReq, c: &str, bound: Option<Duration>) -> RestResp {
    let attempt = node.adaptive_selection.begin(&node.state(), c);
    let response = copy_request_unmeasured(node, sub, c, bound).await;
    drop(attempt);
    response
}

/// A query-phase shard request ([`copy_request`]): a successful answer
/// records `c`'s response time with the search queue size and service time
/// it reports ([`adaptive::STATS_HEADER`]; read directly for this node).
async fn copy_query_request(node: &Arc<ClusterNode>, sub: &RestReq, c: &str, bound: Option<Duration>) -> RestResp {
    let Some(attempt) = node.adaptive_selection.begin(&node.state(), c) else {
        return copy_request_unmeasured(node, sub, c, bound).await;
    };
    if c == node.local.id {
        let response = copy_request_unmeasured(node, sub, c, bound).await;
        attempt.finish_query(response.is_success(), Some(adaptive::local_stats()));
        return response;
    }
    let mut marked = sub.clone();
    if let Err(error) = marked.push_header(adaptive::QUERY_HEADER, "1") { return err(error); }
    let mut response = copy_request_unmeasured(node, &marked, c, bound).await;
    let reported = response.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(adaptive::STATS_HEADER)).map(|(_, v)| adaptive::decode_stats(v));
    if reported.is_some() {
        response.retain_headers(|(k, _)| !k.eq_ignore_ascii_case(adaptive::STATS_HEADER));
    }
    attempt.finish_query(response.is_success(), reported.flatten());
    response
}

async fn copy_request_unmeasured(node: &Arc<ClusterNode>, sub: &RestReq, c: &str, bound: Option<Duration>) -> RestResp {
    if c == node.local.id {
        return local_canonical(node, sub).await;
    }
    let mut fwd = sub.clone();
    if let Err(error)=fwd.push_header(EXEC_LOCAL,"1"){return err(error)};
    let bounded = async {
        match bound {
            Some(d) => tokio::time::sleep(d).await,
            None => std::future::pending::<()>().await,
        }
    };
    tokio::select! {
        r = forward(node, c, &fwd) => r,
        _ = copies::node_left(node, c) => copies::node_disconnected(c),
        _ = bounded => RestResp::error(500, "receive_timeout_transport_exception", &format!("[{c}] request timed out after [{}ms]", bound.unwrap_or_default().as_millis())),
    }
}

/// Retain alias resolution when reducing a read to one shard group. Replacing
/// an alias by its concrete name discards its filter (including the filter on
/// global aggregations and the KNN prefilter). Exclude other resolved indices
/// instead; those absent from this shard engine were validated by the front.
fn shard_read_uri(node: &ClusterNode, req: &RestReq, indices: &[String], op: &str, query: &str) -> elasticsearch_engine::error::Result<SharedText> {
    let concrete = || SharedText::try_format(format_args!("/{}/{}{}{}",UriJoin(indices.iter(),","),op,if query.is_empty(){""}else{"?"},query));
    let segs = req.segments();
    let Some(expr) = segs.first().filter(|_| segs.len() > 1) else { return concrete() };
    if expr == "_all" || expr == "*" || node.engine.get_index(expr).is_some() { return concrete() }
    let Ok(targets) = node.engine.resolve(expr, ResolveOpts::from_params(&req.params())) else { return concrete() };
    if !targets.iter().any(|t| t.alias_filter.is_some() || t.routing.is_some()) { return concrete() }
    let scoped=req.path().trim_start_matches('/').split('/').next().unwrap_or(expr);
    let excluded=targets.iter().filter(|t|!indices.contains(&t.svc.name));
    let has_excluded=excluded.clone().next().is_some();
    struct Query<'a>(&'a str,bool);
    impl std::fmt::Display for Query<'_>{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{
        if !self.1 {return if self.0.is_empty(){Ok(())}else{write!(f,"?{}",self.0)};}
        let mut sep="?";
        for part in self.0.split('&').filter(|p|!p.is_empty()&&!p.starts_with("ignore_unavailable=")){write!(f,"{sep}{part}")?;sep="&";}
        write!(f,"{sep}ignore_unavailable=true")
    }}
    SharedText::try_format(format_args!("/{scoped}{}{}/{op}{}",if has_excluded{",-"}else{""},UriJoin(excluded.map(|t|UriComponent(&t.svc.name)),",-"),Query(query,has_excluded)))
}

/// Run `sub` on the first of `cands` that answers: any failure but a
/// request error moves on to the next copy, as ES retries a shard on its
/// other copies. Returns the response (the most meaningful failure when no
/// copy answered) and the node that gave it.
async fn on_shard_copies(node: &Arc<ClusterNode>, sub: &RestReq, cands: Vec<String>, bound: Option<Duration>) -> (RestResp, String) {
    match try_copies(node, sub, &cands, bound).await {
        Tried::Answered(r, n) => (r, n),
        Tried::Request(r) => (r, String::new()),
        Tried::Failed { resp, node, .. } => (resp, node),
    }
}

/// Does this coordinated `_search` over `n_subs` shard requests need ES's
/// DFS phase? ES runs one for `search_type=dfs_query_then_fetch` and for any
/// search whose kNN is folded into a query (`knn_needs_dfs`: the kNN forces
/// DFS, and the folded query's text scoring needs the global statistics); a
/// lone kNN search scores no text, and the knn-dfs rewrite marks its query
/// with the search type (`force_dfs`). With a single sub request there is
/// nothing to combine: ES turns a one-shard DFS search into
/// query_then_fetch, and one engine computes a DFS search over its own
/// shards itself. A request already in a DFS round (`req.dfs` set) never
/// starts another.
fn dfs_wanted(req: &RestReq, body: &Value, params: &HashMap<String, String>, n_subs: usize) -> bool {
    if n_subs < 2 || req.dfs != Dfs::None {
        return false;
    }
    if fanout::knn_needs_dfs(body, params) {
        return true;
    }
    // A lone kNN (also each knn-dfs phase) scores no text.
    let lone_knn = body.get("knn").is_some_and(|k| !k.is_null());
    !lone_knn && params.get("search_type").map(String::as_str) == Some("dfs_query_then_fetch")
}

/// The stats-round twin of a shard query request: same target, query and
/// body, answered by the engine with its `DfsStats` (rendering params
/// dropped: the answer is internal).
fn dfs_collect_req(sub: &RestReq) -> elasticsearch_engine::error::Result<RestReq> {
    let mut r = sub.clone();
    let (path, q) = sub.uri.split_once('?').unwrap_or((sub.uri.as_str(), ""));
    r.set_uri_fmt(format_args!("{path}{}", FilteredQuery{query:q,drop:&["filter_path", "pretty", "human", "error_trace"],append:""}))?;
    r.set_dfs(Dfs::Collect)?;
    Ok(r)
}

/// Merge the stats round's answers (`merge_dfs`); the first failure is the
/// search's answer, as a failing ES DFS phase fails the search.
async fn dfs_round<F: std::future::Future<Output = RestResp>>(round: impl IntoIterator<Item = F>) -> Result<SharedDfsStats, RestResp> {
    let mut stats = Vec::new();
    for r in futures::future::join_all(round).await {
        if !r.is_success() {
            return Err(r);
        }
        match serde_json::from_slice::<elasticsearch_http::DfsStats>(&r.body) {
            Ok(s) => stats.push(s),
            Err(e) => return Err(RestResp::error(500, "illegal_state_exception", &format!("unreadable dfs phase result: {e}"))),
        }
    }
    SharedDfsStats::try_new(&elasticsearch_http::merge_dfs(&stats)).map_err(err)
}

/// `req` with `search_type=dfs_query_then_fetch`: the knn-dfs rewrite's
/// query keeps ES's forced DFS (the kNN clauses are gone from its body).
fn force_dfs(mut req: RestReq) -> elasticsearch_engine::error::Result<RestReq> {
    // Hold the immutable source payload independently while replacing the URI.
    // Formatting arguments borrow their inputs until the setter has encoded them.
    let source = req.clone();
    let (path, q) = source.uri.split_once('?').unwrap_or((source.uri.as_str(), ""));
    req.set_uri_fmt(format_args!("{path}{}",FilteredQuery{query:q,drop:&["search_type"],append:"search_type=dfs_query_then_fetch"}))?;
    Ok(req)
}

/// One phase of a search over `plan` (see [`shard_fan_out`]).
fn shard_search<'a>(node: &'a Arc<ClusterNode>, req: RestReq, plan: &'a Arc<copies::Plan>) -> futures::future::BoxFuture<'a, RestResp> {
    Box::pin(async move { shard_fan_out(node, &req, "_search", plan).await })
}

/// ES's DFS kNN phase (`DfsQueryPhase`): the global top k of each `knn`
/// clause, each found by `run` as a lone kNN search (which merges exactly),
/// folded into the request's query (`fanout::knn_rewrite`). `None` when the
/// search needs no DFS phase.
async fn knn_dfs<'a>(req: &RestReq, run: impl Fn(RestReq) -> futures::future::BoxFuture<'a, RestResp>) -> Result<Option<RestReq>, RestResp> {
    let params = req.params();
    let empty = json!({});
    let body = req.json_ref().unwrap_or(&empty);
    if !fanout::knn_needs_dfs(&body, &params) {
        return Ok(None);
    }
    let (_, size) = fanout::from_size(&body, &params);
    let requests = fanout::knn_clauses(&body).iter().map(|c| req.try_rebuilt(&fanout::knn_dfs_body(c, size), |p| !["from", "size", "sort"].contains(&p) && !fanout::Q_PARAMS.contains(&p))).collect::<EsResult<Vec<_>>>().map_err(err)?;
    let phases: Vec<_> = requests.into_iter().map(&run).collect();
    let mut top = Vec::new();
    for r in futures::future::join_all(phases).await {
        if !r.is_success() {
            return Err(r);
        }
        top.push(fanout::scored_hits(&r.body_json().unwrap_or(Value::Null)));
    }
    req.try_rebuilt(&fanout::knn_rewrite(&body, &params, &top), |p| !fanout::Q_PARAMS.contains(&p)).map(Some).map_err(err)
}

/// ES's expand phase (`ExpandSearchPhase`) for `collapse.inner_hits`: the
/// search runs (and merges) without `inner_hits`, then every returned hit
/// gets one search per inner_hits entry over the same targets, restricted
/// to its collapse value (`fanout::expand_body`). `None` when the request
/// has no `collapse.inner_hits`, or when they are invalid: the caller then
/// fans the request out unchanged so the engine raises ES's own error.
async fn collapse_expand<'a>(req: &RestReq, run: impl Fn(RestReq) -> futures::future::BoxFuture<'a, RestResp>) -> Option<RestResp> {
    // Checked on the borrowed body: every search passes here, and copying a
    // large one (`large_terms`) costs as much as parsing it.
    if req.json_ref().and_then(|b| b.get("collapse")).and_then(|c| c.get("inner_hits")).is_none() {
        return None;
    }
    let body = req.json().unwrap_or(json!({}));
    // Malformed entries: the plain fan-out sends the request unchanged and
    // every shard answers with the engine's own validation error.
    let entries = fanout::collapse_inner_hits(&body);
    if !fanout::collapse_inner_hits_valid(&body) || entries.is_empty() {
        return None;
    }
    let params = req.params();
    // Rendering params are applied once to the final response (`rest::util::postprocess`).
    let first_req = match req.try_rebuilt(&fanout::strip_inner_hits(&body), |p| !["filter_path", "pretty", "human"].contains(&p)) { Ok(req) => req, Err(e) => return Some(err(e)) };
    let first = run(first_req).await;
    if !first.is_success() {
        return Some(first);
    }
    // Lossless: the hits' and inner hits' sources pass through raw.
    let mut sources = elasticsearch_engine::source_json::ResponseSources::default();
    let Ok(mut resp) = sources.parse_lazy(&first.body) else { return Some(first) };
    let field = body["collapse"]["field"].as_str().unwrap_or_default().to_string();
    // The group request copies the original's targets, routing, preference,
    // search type and index options; everything the URL sets on the source
    // (or on rendering) is replaced by the expand body.
    let keep = |p: &str| {
        !fanout::Q_PARAMS.contains(&p)
            && ![
                "from", "size", "sort", "_source", "_source_includes", "_source_excludes", "stored_fields", "docvalue_fields", "track_total_hits", "track_scores", "explain", "version",
                "seq_no_primary_term", "timeout", "terminate_after", "scroll", "stats", "filter_path", "pretty", "human", "typed_keys",
            ]
            .contains(&p)
            && !p.starts_with("suggest")
    };
    let hits: Vec<Value> = resp["hits"]["hits"].as_array().cloned().unwrap_or_default();
    let subs = hits
        .iter()
        .flat_map(|h| entries.iter().map(move |e| (h, e)))
        .map(|(h, e)| req.try_rebuilt(&fanout::expand_body(&body, &params, &field, fanout::collapse_value(h, &field), e), keep))
        .collect::<EsResult<Vec<_>>>();
    let subs = match subs { Ok(subs) => subs, Err(e) => return Some(err(e)) };
    let subs: Vec<_> = subs.into_iter().map(&run).collect();
    let mut results = Vec::with_capacity(subs.len());
    for r in futures::future::join_all(subs).await {
        if !r.is_success() {
            return Some(r);
        }
        match sources.parse_lazy(&r.body) {
            Ok(v) => results.push(v),
            Err(e) => return Some(RestResp::error(500, "illegal_state_exception", &e.to_string())),
        }
    }
    fanout::attach_inner_hits(&mut resp, &entries, &results);
    let mut out = RestResp::json(200, &Value::Null);
    out.replace_body_vec(sources.render(&resp));
    Some(out)
}

/// `explain=true` hits name the shard copy that scored them: the shard
/// engine only knows itself as shard 0 of a local node.
fn label_explained_hits(part: &mut Value, shard: u32, node: &str) {
    for h in part.get_mut("hits").and_then(|h| h.get_mut("hits")).and_then(Value::as_array_mut).into_iter().flatten() {
        if h.get("_shard").is_some() {
            let index = h["_index"].as_str().unwrap_or_default().to_string();
            // ES renders `_shard` and `_node` first.
            let mut m = serde_json::Map::new();
            m.insert("_shard".into(), json!(format!("[{index}][{shard}]")));
            m.insert("_node".into(), json!(node));
            if let Value::Object(o) = std::mem::take(h) {
                m.extend(o.into_iter().filter(|(k, _)| k != "_shard" && k != "_node"));
            }
            *h = Value::Object(m);
        }
    }
    // `profile=true`: likewise each shard's profile names its real shard.
    // (`get_mut`: indexing a missing key would insert it.)
    for p in part.get_mut("profile").and_then(|p| p.get_mut("shards")).and_then(Value::as_array_mut).into_iter().flatten() {
        let index = p["index"].as_str().unwrap_or_default().to_string();
        p["id"] = json!(format!("[{node}][{index}][{shard}]"));
        p["node_id"] = json!(node);
        p["shard_id"] = json!(shard);
    }
}

/// `_terms_enum` parts: the union of the terms, sorted, cut to `size`;
/// `complete` only when every part was.
fn merge_terms_enum(parts: &[Value], body: &Value, params: &HashMap<String, String>) -> Value {
    let size = body.get("size").and_then(Value::as_u64).or_else(|| params.get("size").and_then(|s| s.parse().ok())).unwrap_or(10) as usize;
    let mut terms: BTreeSet<String> = BTreeSet::new();
    let mut complete = true;
    for p in parts {
        for t in p["terms"].as_array().into_iter().flatten() {
            if let Some(t) = t.as_str() {
                terms.insert(t.to_string());
            }
        }
        complete &= p["complete"].as_bool().unwrap_or(true);
    }
    let terms: Vec<String> = terms.into_iter().take(size).collect();
    json!({"_shards": fanout::merge_count(parts)["_shards"], "terms": terms, "complete": complete})
}

/// `POST /{index}/_async_search` that is not kept: run the search to
/// completion across the shards and answer as a finished async search.
fn async_search<'a>(node: &'a Arc<ClusterNode>, req: &'a RestReq, holders: &'a std::collections::BTreeMap<String, BTreeSet<String>>) -> futures::future::BoxFuture<'a, RestResp> {
    Box::pin(async_search_inner(node, req, holders, None))
}

/// The cross-cluster submit's async search, boxed as a `Send` future: it is
/// awaited from `ccs_read`, which `read` reaches, and `async_search_inner`
/// spawns `read`, so an unboxed call would make the future types cyclic.
fn async_search_ccs<'a>(node: &'a Arc<ClusterNode>, req: &'a RestReq, holders: &'a std::collections::BTreeMap<String, BTreeSet<String>>, plan: elasticsearch_engine::ccr::remote::CcsPlan) -> futures::future::BoxFuture<'a, RestResp> {
    Box::pin(async_search_inner(node, req, holders, Some(plan)))
}

/// A finished search response with the CCS plan's `_clusters` (the local
/// part's `_shards` / `took` and every skipped remote), patched losslessly.
fn with_ccs_clusters(text: &str, plan: &elasticsearch_engine::ccr::remote::CcsPlan) -> Option<String> {
    let summary: Value = serde_json::from_str(text).ok()?;
    let clusters = plan.clusters(Some(&summary)).to_string();
    elasticsearch_engine::source_json::with_member(text, "_clusters", &clusters, false)
}

/// `ccs`: the plan of a cross-cluster submit whose local part this search
/// reads; the stored result carries its `_clusters`.
async fn async_search_inner(node: &Arc<ClusterNode>, req: &RestReq, holders: &std::collections::BTreeMap<String, BTreeSet<String>>, ccs: Option<elasticsearch_engine::ccr::remote::CcsPlan>) -> RestResp {
    use elasticsearch_engine::error::EsError;
    let err = |e: EsError| RestResp::json(e.status, &e.to_json());
    let params = req.params();
    // ES `SubmitAsyncSearchRequest` validation (as the engine's handler).
    if params.contains_key("scroll") {
        return err(EsError::action_validation(&["[scroll] queries are not supported"]));
    }
    let (wait, keep_alive) = match (async_time_param(&params, "wait_for_completion_timeout", "1s"), async_time_param(&params, "keep_alive", "5d")) {
        (Ok(w), Ok(k)) => (w, k),
        (Err(e), _) | (_, Err(e)) => return err(e),
    };
    if keep_alive < Duration::from_secs(1) {
        return err(EsError::action_validation(&["[keep_alive] must be greater or equal to [1s]"]));
    }
    let keep_on_completion = match params.get("keep_on_completion").map(String::as_str) {
        None | Some("false") => false,
        Some("") | Some("true") => true,
        Some(v) => return err(EsError::illegal_argument(format!("Failed to parse value [{v}] as only [true] or [false] are allowed."))),
    };
    const ASYNC_ONLY: [&str; 5] = ["wait_for_completion_timeout", "keep_alive", "keep_on_completion", "batched_reduce_size", "ccs_minimize_roundtrips"];
    // Request errors come back as plain ES errors, before anything runs.
    let mut search_params = params.clone();
    for k in ASYNC_ONLY {
        search_params.remove(k);
    }
    let mut validation_body = req.json().unwrap_or(json!({}));
    if let Some(id) = validation_body["pit"]["id"].as_str().and_then(contexts::pit_validation_id) {
        validation_body["pit"]["id"] = json!(id);
    }
    if let Err(e) = elasticsearch_http::validate_search_request(&node.engine, &validation_body, &search_params) {
        return err(e);
    }
    // The search itself: the ordinary cluster `_search` (DFS / knn rounds
    // included). Response shaping applies to the async wrapper.
    let mut sub = req.clone();
    let base = req.path().trim_end_matches('/').rsplit_once('/').map(|(b, _)| b).unwrap_or("");
    let uri = FilteredQuery { query:req.query(),drop:&ASYNC_ONLY,append:"" };
    // The first formatting stage owns its filtered output; the second stage
    // removes rendering options without unguarded joined URI temporaries.
    let filtered=match SharedText::try_format(format_args!("{uri}")){Ok(v)=>v,Err(e)=>return err(e)};
    if let Err(e) = sub.set_uri_fmt(format_args!("{base}/_search{}",FilteredQuery{query:filtered.strip_prefix('?').unwrap_or(&filtered),drop:&["filter_path","pretty","human","typed_keys"],append:"typed_keys=true"})) { return err(e); }
    let s = elasticsearch_engine::async_search::AsyncSearch::new(holders.len(), keep_alive);
    let c = Arc::new(contexts::ClusterAsync { s: s.clone(), error: parking_lot::Mutex::new(None), warnings: parking_lot::Mutex::new(Vec::new()) });
    // The search reads the request's own targets (`/{targets}/_async_search`);
    // an empty expression would read every index.
    let segs = req.segments();
    let targets = if segs.len() == 2 { segs[0].clone() } else { String::new() };
    let (n2, c2) = (node.clone(), c.clone());
    // The submission runs as its task (`DELETE _async_search/<id>` cancels it).
    let async_key = format!("{}{}", contexts::ASYNC_PREFIX, s.id);
    let async_token = search_task::async_started(&async_key);
    let mut task = tokio::spawn(async move {
        let node_ref = &n2;
        let r = search_task::run(node_ref, &sub, search_task::ASYNC_SUBMIT_ACTION, None, Some(async_token), |r| async move { read(node_ref, &r, &targets).await }).await;
        search_task::async_finished(&async_key);
        *c2.warnings.lock() = r.headers.iter().filter(|(name, _)| name.eq_ignore_ascii_case("warning")).cloned().collect();
        if r.is_success() {
            match std::str::from_utf8(&r.body) {
                Ok(text) => match ccs.as_ref().and_then(|plan| with_ccs_clusters(text, plan)) {
                    Some(patched) => c2.s.complete_borrowed(&patched),
                    None => c2.s.complete_borrowed(text),
                },
                Err(_) => c2.s.complete(Err(EsError::internal("async search response is not UTF-8"))),
            }
        } else {
            let e = r.body_json().map(|v| v["error"].clone()).filter(Value::is_object).unwrap_or_else(|| json!({"type": "exception", "reason": String::from_utf8_lossy(&r.body)}));
            let kind = e["type"].as_str().unwrap_or("exception").to_string();
            let reason = e["reason"].as_str().unwrap_or_default().to_string();
            *c2.error.lock() = Some(e);
            c2.s.complete(Err(EsError::new(r.status, &kind, reason)));
        }
    });
    let _ = tokio::time::timeout(wait, &mut task).await;
    let done = !s.is_running();
    let store = !done || keep_on_completion;
    let key = format!("{}{}", contexts::ASYNC_PREFIX, s.id);
    let wrapped = wrap_id(node, &key);
    if store {
        contexts::async_put(key, c.clone());
    }
    crate::rest::util::postprocess(req, contexts::async_render(&c, store, &wrapped, params.get("typed_keys").map(|v| v.is_empty() || v == "true").unwrap_or(false)))
}

/// A time parameter of an async-search request (ES `TimeValue` syntax).
fn async_time_param(params: &HashMap<String, String>, name: &str, default: &str) -> Result<Duration, elasticsearch_engine::error::EsError> {
    let s = params.get(name).map(String::as_str).unwrap_or(default);
    elasticsearch_engine::date::parse_time_value(s)
        .map(|ms| Duration::from_millis(ms.max(0) as u64))
        .ok_or_else(|| elasticsearch_engine::error::EsError::illegal_argument(format!("failed to parse setting [{name}] with value [{s}] as a time value: unit is missing or unrecognized")))
}

/// `GET|DELETE /_async_search[/status]/{node}.~a...`: a cluster async
/// search kept by this (coordinating) node.
async fn cluster_async_context(req: &RestReq, wrapped: &str, key: &str) -> RestResp {
    let missing = || RestResp::error(404, "resource_not_found_exception", wrapped);
    let segs = req.segments();
    let is_status = segs.get(1).map(String::as_str) == Some("status");
    let params = req.params();
    match req.method() {
        Method::DELETE => match contexts::async_remove(key) {
            Some(_) => {
                search_task::async_cancel(key);
                RestResp::json(200, &json!({"acknowledged": true}))
            }
            None => missing(),
        },
        _ => {
            let Some(c) = contexts::async_get(key) else { return missing() };
            if is_status {
                return crate::rest::util::postprocess(req, contexts::async_status(&c, wrapped));
            }
            if params.contains_key("keep_alive") {
                match async_time_param(&params, "keep_alive", "5d") {
                    Ok(k) => c.s.extend(k),
                    Err(e) => return RestResp::json(e.status, &e.to_json()),
                }
            }
            if params.contains_key("wait_for_completion_timeout") {
                match async_time_param(&params, "wait_for_completion_timeout", "0s") {
                    Ok(w) => {
                        let s = c.s.clone();
                        let _ = tokio::task::spawn_blocking(move || s.wait(w)).await;
                    }
                    Err(e) => return RestResp::json(e.status, &e.to_json()),
                }
            }
            crate::rest::util::postprocess(req, contexts::async_render(&c, true, wrapped, params.get("typed_keys").map(|v| v.is_empty() || v == "true").unwrap_or(false)))
        }
    }
}

/// Is this routing entry's copy really live on this node?
/// Does this node hold copy `c` readable (`copies::readable_copy`)? Right
/// after a restart the routing table may still name a copy here that the
/// node does not have, and a copy being removed, reset or still recovering
/// must not answer reads.
fn local_holds(node: &ClusterNode, c: &elasticsearch_cluster::state::ShardRouting) -> bool {
    node.shards.get(&c.index, c.shard).is_some_and(|s| Some(s.lock().allocation_id.as_str()) == c.allocation_id())
        && copies::readable_copy(node, &c.index, c.shard).is_ok()
}

fn fan_out_search<'a>(node: &'a Arc<ClusterNode>, req: RestReq, names: &'a BTreeSet<String>, holders: &'a std::collections::BTreeMap<String, BTreeSet<String>>) -> futures::future::BoxFuture<'a, RestResp> {
    Box::pin(async move { fan_out(node, &req, names, holders).await })
}

/// Search / count across nodes (see `fanout`).
async fn fan_out(node: &Arc<ClusterNode>, req: &RestReq, names: &BTreeSet<String>, holders: &std::collections::BTreeMap<String, BTreeSet<String>>) -> RestResp {
    let segs = req.segments();
    let op = segs.iter().rev().find(|s| *s == "_search" || *s == "_count").cloned();
    let params = req.params();
    let body = req.json().unwrap_or(json!({}));
    let fail = |why: &str| RestResp::error(500, "illegal_state_exception", &format!("no single node holds copies of all of {names:?} and {why} cannot be combined across nodes yet (needs multi-shard partial results)"));
    if let Some(r) = fanout::multi_doc(node, req).await {
        return r;
    }
    // APIs reduced per shard: run them as single-shard indices.
    let last = segs.last().map(String::as_str).unwrap_or("");
    if last == "_msearch" {
        return match msearch(node, req).await {
            Some(r) => r,
            None => local_canonical(node, req).await,
        };
    }
    if last == "_terms_enum" || (last == "_async_search" && req.method() == Method::POST) {
        let by_shard: std::collections::BTreeMap<String, BTreeSet<String>> = holders.iter().map(|(n, h)| (format!("{n}#0"), h.clone())).collect();
        return multi_shard_fan_out(node, req, last, &by_shard).await;
    }
    // Scroll and PIT contexts spanning nodes (see `contexts`).
    // Each index's shard 0 on its copies in preference order, the next copy
    // when one fails, as a search.
    let by_shard: std::collections::BTreeMap<String, BTreeSet<String>> = holders.iter().map(|(n, h)| (format!("{n}#0"), h.clone())).collect();
    if segs.last().map(String::as_str) == Some("_pit") {
        let plan = match copies::Plan::new(node, req, &by_shard) { Ok(p) => p, Err(r) => return r };
        return contexts::open_pit(node, req, contexts::plan_groups(&plan), true, Some(&plan)).await;
    }
    if op.as_deref() == Some("_search") && params.contains_key("scroll") {
        let mut p = params.clone();
        p.remove("scroll");
        if let Some(f) = fanout::unsupported_feature(&body, &p) {
            return fail(f);
        }
        if fanout::knn_needs_dfs(&body, &p) {
            return fail("knn with a query or several knn clauses in a scroll");
        }
        let plan = match copies::Plan::new(node, req, &by_shard) { Ok(p) => p, Err(r) => return r };
        return contexts::start_scroll(node, req, contexts::plan_groups(&plan), true, &body, &params, &HashMap::new(), Some(&plan)).await;
    }
    if let Some(at) = segs.iter().position(|s| s == "_stats") {
        // Index stats: each node reports the whole indices it covers.
        let Some(groups) = fanout::cover(names, holders, &node.local.id) else { return fail("index stats") };
        let rest = UriJoin(segs[at..].iter().map(|s|UriComponent(s)),"/");
        let query = req.query();
        let futs = groups.into_iter().map(|(n, idx)| {
            let mut sub = req.clone();
            // Response shaping happens once, on the merged response.
            let prepared = sub.set_uri_fmt(format_args!("/{}/{rest}{}",UriJoin(idx.iter().map(|i|UriComponent(i)),","),FilteredQuery{query,drop:&["filter_path","pretty"],append:""}));
            let node = node.clone();
            async move { if let Err(e) = prepared { return err(e); } if n == node.local.id { local_canonical(&node, &sub).await } else { forward(&node, &n, &sub).await } }
        });
        let mut parts = Vec::new();
        for r in futures::future::join_all(futs).await {
            if !r.is_success() {
                return r;
            }
            parts.push(r.body_json().unwrap_or(Value::Null));
        }
        return crate::rest::util::postprocess(req, RestResp::json(200, &fanout::merge_index_stats(&parts)));
    }
    let Some(op) = op else { return fail("this API") };
    // Search / count: every index's shard 0 on its own copies, as a
    // multi-shard search (preference, next copy on failure, partial
    // results); no node needs to hold all of them.
    let by_shard: std::collections::BTreeMap<String, BTreeSet<String>> = holders.iter().map(|(n, h)| (format!("{n}#0"), h.clone())).collect();
    multi_shard_fan_out(node, req, &op, &by_shard).await
}

/// Response-rendering params: applied once to a coordinated response,
/// never forwarded to the shard requests it merges.
const RENDER_PARAMS: [&str; 3] = ["filter_path", "pretty", "human"];

/// `preference=_shards:0,2[|rest]`: the shard numbers to search. Shard
/// engines are single-shard, so the coordinator applies the filter and the
/// parameter is not forwarded (`is_shards_preference`).
fn shards_preference(p: &str) -> Option<BTreeSet<u32>> {
    let list = p.strip_prefix("_shards:")?;
    let list = list.split('|').next().unwrap_or("");
    Some(list.split(',').filter_map(|s| s.trim().parse().ok()).collect())
}

/// Is this raw query pair a `preference=_shards:...` parameter?
fn is_shards_preference(pair: &str) -> bool {
    pair.strip_prefix("preference=").is_some_and(|v| v.starts_with("_shards:") || v.starts_with("_shards%3A") || v.starts_with("_shards%3a"))
}

/// Spread reads over copies (`preference` keeps a session on one node).
fn pick(candidates: &BTreeSet<String>, req: &RestReq) -> Option<String> {
    if candidates.is_empty() {
        return None;
    }
    let v: Vec<&String> = candidates.iter().collect();
    let i = match req.params().get("preference") {
        Some(p) => p.bytes().fold(0usize, |a, b| a.wrapping_mul(31).wrapping_add(b as usize)),
        None => rand::random::<usize>(),
    };
    Some(v[i % v.len()].clone())
}

// ------------------------------------------------------ context id wrapping

const ID_SEP: char = '.';

fn wrap_id(node: &ClusterNode, id: &str) -> String {
    format!("{}{ID_SEP}{id}", node.local.id)
}

fn unwrap_id(id: &str) -> Option<(&str, &str)> {
    let (n, inner) = id.split_once(ID_SEP)?;
    (n.len() == 22).then_some((n, inner))
}

/// Prefix scroll / PIT / async-search ids with the node that owns them.
fn wrap_context_ids(node: &ClusterNode, req: &RestReq, mut resp: RestResp) -> RestResp {
    if !resp.is_success() || is_gzip(&resp) {
        return resp;
    }
    // A plain search (every shard sub-request of a fan-out) opens no scroll
    // or point in time: skip scanning its body, megabytes with shard state.
    if req.segments().last().is_some_and(|s| s == "_search") && !req.params().contains_key("scroll")
        && !req.body.windows(5).any(|w| w == b"\"pit\"") {
        return resp;
    }
    // Scroll pages: the engine writes `_scroll_id` first. Swap that token
    // alone instead of parsing (and re-serializing) a page of hits.
    const SCROLL_HEAD: &[u8] = b"{\"_scroll_id\":\"";
    if resp.body.starts_with(SCROLL_HEAD) {
        let start = SCROLL_HEAD.len();
        let mut end = start;
        while end < resp.body.len() && resp.body[end] != b'"' {
            end += if resp.body[end] == b'\\' { 2 } else { 1 };
        }
        if resp.body.get(start - 1..=end).is_some_and(|token|serde_json::from_slice::<&serde_json::value::RawValue>(token).is_ok()) {
            // Preserve the existing JSON string token verbatim. Only the node
            // prefix needs encoding, and both encodings are admitted first.
            let prefix=match RestReq::encode_body(|out|serde_json::to_writer(out,&node.local.id).map_err(std::io::Error::other)){Ok(v)=>v,Err(e)=>return err(e)};
            return resp.with_encoded_body(|out| {
                out.write_all(&resp.body[..start])?;
                out.write_all(&prefix[1..prefix.len()-1])?;
                out.write_all(b".")?;
                out.write_all(&resp.body[start..])
            });
        }
    }
    // Engine-produced context keys are emitted literally. Most searches have
    // none; preserve their raw hit bodies without another full JSON parse.
    if ![b"\"_scroll_id\"".as_slice(), b"\"pit_id\"".as_slice(), b"\"id\"".as_slice()]
        .iter().any(|key|resp.body.windows(key.len()).any(|window|window == *key)) {
        return resp;
    }
    let Some(mut v) = resp.body_json() else { return resp };
    let mut changed = false;
    for key in ["_scroll_id", "pit_id"] {
        if let Some(s) = v.get(key).and_then(Value::as_str) {
            v[key] = json!(wrap_id(node, s));
            changed = true;
        }
    }
    // `POST /{index}/_pit` → {"id": ...}; async search → {"id": ..., "is_running": ...}
    if v.get("id").and_then(Value::as_str).is_some() && (v.as_object().map(|o| o.len() == 1).unwrap_or(false) || v.get("is_running").is_some() || v.get("_shards").is_some() && v.get("hits").is_none()) {
        let s = v["id"].as_str().unwrap_or_default().to_string();
        v["id"] = json!(wrap_id(node, &s));
        changed = true;
    }
    if changed {
        resp.replace_json(&v);
    }
    resp
}

/// A clear-context 404 with a valid count is ES's zero-freed result, not
/// a transport/admission failure. Never turn a rejected cleanup into success.
fn freed_contexts(response: RestResp) -> EsResult<u64> {
    let body = response.body_json().unwrap_or(Value::Null);
    if !response.is_success() && !(response.status == 404 && body["succeeded"] == true && body["num_freed"].as_u64() == Some(0)) {
        return Err(elasticsearch_engine::error::EsError::from_error_json(response.status, &body["error"]));
    }
    body["num_freed"].as_u64().ok_or_else(|| elasticsearch_engine::error::EsError::internal("clear context response omitted num_freed"))
}

async fn context_forward(node: &Arc<ClusterNode>, req: &RestReq, wrapped: &str, rewrite: impl Fn(&[u8], &str) -> Vec<u8>) -> RestResp {
    let Some((owner, inner)) = unwrap_id(wrapped) else {
        return RestResp::error(400, "illegal_argument_exception", &format!("Cannot parse scroll id [{wrapped}]"));
    };
    let r = match req.try_with_body(&rewrite(&req.body, inner)) { Ok(r) => r, Err(e) => return err(e) };
    exec_on(node, owner, r).await
}

async fn context(node: &Arc<ClusterNode>, req: &RestReq, kind: ContextKind) -> RestResp {
    let segs = req.segments();
    match kind {
        ContextKind::Scroll | ContextKind::ClearScroll => {
            // ES: ids in the body win over the path and `scroll_id` param.
            let params = req.params();
            let body = req.json();
            let list = |s: &str| s.split(',').filter(|x| !x.is_empty()).map(str::to_string).collect::<Vec<_>>();
            let from_body = body.as_ref().filter(|v| v.is_object()).and_then(|v| match &v["scroll_id"] {
                Value::String(s) => Some(vec![s.clone()]),
                Value::Array(a) => Some(a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()),
                _ => None,
            });
            let raw = String::from_utf8_lossy(&req.body).trim().to_string();
            let ids: Vec<String> = from_body
                .or_else(|| segs.get(2).map(|s| list(s)))
                .or_else(|| params.get("scroll_id").map(|s| list(s)))
                .or_else(|| (kind == ContextKind::Scroll && body.is_none() && !raw.is_empty()).then(|| vec![raw]))
                .unwrap_or_default();
            if ids.is_empty() {
                // The engine renders the validation error.
                return local_engine(node, req).await;
            }
            let bad = |id: &str| RestResp::error(400, "illegal_argument_exception", &format!("Cannot parse scroll id [{id}]"));
            if kind == ContextKind::Scroll {
                let id = &ids[0];
                let Some((owner, inner)) = unwrap_id(id) else { return bad(id) };
                let scroll = body.as_ref().and_then(|v| v["scroll"].as_str().map(str::to_string)).or_else(|| params.get("scroll").cloned());
                if inner.starts_with(contexts::SCROLL_PREFIX) {
                    return contexts::continue_scroll(node, req, owner, inner, scroll).await;
                }
                let mut b = body.filter(Value::is_object).unwrap_or(json!({}));
                b["scroll_id"] = json!(inner);
                let query=match contexts::query_without(req.query(), &["scroll_id"]){Ok(query)=>query,Err(error)=>return err(error)};
                let r = match contexts::json_req_formatted(req, "POST", format_args!("/_search/scroll{query}"), &b) { Ok(r) => r, Err(e) => return err(e) };
                return exec_on(node, owner, r).await;
            }
            if ids.len() == 1 && ids[0] == "_all" {
                // Every node's contexts (all of its shard engines), each
                // node asked exactly once.
                let st = node.state();
                let futs = st.nodes.keys().map(|n| {
                    let r = contexts::json_req(req, "DELETE", "/_search/scroll/_all", &Value::Null);
                    let (node, n) = (node.clone(), n.clone());
                    async move {
                        let r = r?;
                        if n == node.local.id {
                            return contexts::clear_all_local(&node, &r).await;
                        }
                        freed_contexts(exec_on(&node, &n, r).await)
                    }
                });
                let freed = match futures::future::join_all(futs).await.into_iter().collect::<EsResult<Vec<u64>>>() { Ok(counts) => counts.into_iter().sum::<u64>(), Err(e) => return err(e) };
                // ES ClearScrollResponse.status(): NOT_FOUND when nothing
                // was freed, `_all` included.
                return RestResp::json(if freed == 0 { 404 } else { 200 }, &json!({"succeeded": true, "num_freed": freed}));
            }
            let mut parsed = Vec::new();
            for id in &ids {
                match unwrap_id(id) {
                    Some((owner, inner)) => parsed.push((id.clone(), owner.to_string(), inner.to_string())),
                    None => return bad(id),
                }
            }
            let futs = parsed.iter().map(|(id, owner, inner)| {
                let (node, req) = (node.clone(), req.clone());
                async move {
                    if inner.starts_with(contexts::SCROLL_PREFIX) {
                        return contexts::clear_scroll(&node, &req, id, owner, inner).await;
                    }
                    let r = contexts::json_req(&req, "DELETE", "/_search/scroll", &json!({"scroll_id": [inner]}))?;
                    freed_contexts(exec_on(&node, owner, r).await)
                }
            });
            let freed = match futures::future::join_all(futs).await.into_iter().collect::<EsResult<Vec<u64>>>() { Ok(counts) => counts.into_iter().sum::<u64>(), Err(e) => return err(e) };
            // ES: 404 when nothing was freed, `succeeded` stays true.
            RestResp::json(if freed == 0 { 404 } else { 200 }, &json!({"succeeded": true, "num_freed": freed}))
        }
        ContextKind::ClosePit => {
            let Some(id) = req.json().and_then(|v| v["id"].as_str().map(str::to_string)) else {
                return local_engine(node, req).await;
            };
            if let Some((_, inner)) = unwrap_id(&id) {
                if inner.starts_with(contexts::PIT_PREFIX) {
                    if let Some(r) = contexts::close_pit(node, req, inner).await {
                        return r;
                    }
                }
            }
            context_forward(node, req, &id, |body, inner| {
                let mut v: Value = serde_json::from_slice(body).unwrap_or(json!({}));
                v["id"] = json!(inner);
                serde_json::to_vec(&v).unwrap_or_default()
            })
            .await
        }
        ContextKind::AsyncSearch => {
            let id = segs.last().cloned().unwrap_or_default();
            let Some((owner, inner)) = unwrap_id(&id) else {
                return RestResp::error(404, "resource_not_found_exception", &id);
            };
            // A cluster async search (`~a`) lives in its coordinating
            // node's front; GET / DELETE on it (status: GET only) are
            // answered there, anything else falls through to the engine's
            // method / route errors.
            let status = segs.get(1).map(String::as_str) == Some("status");
            let m = req.method();
            if inner.starts_with(contexts::ASYNC_PREFIX) && (m == Method::GET || (m == Method::DELETE && !status)) {
                if owner != node.local.id {
                    return forward(node, owner, req).await;
                }
                return cluster_async_context(req, &id, inner).await;
            }
            let mut r = req.clone();
            let (base,replacement,suffix)=match req.uri.split_once(id.as_str()){Some((base,suffix))=>(base,inner,suffix),None=>(req.uri.as_str(),"","")};
            if let Err(e) = r.set_uri_fmt(format_args!("{base}{replacement}{suffix}")) { return err(e); }
            exec_on(node, owner, r).await
        }
    }
}

// --------------------------------------------------------------- broadcast

/// Phase times of the stats APIs for benchmarks (`stats_api_latency`):
/// while [`stats_phases::ENABLED`], each `/_stats` and `/_nodes/stats`
/// request records its phases in microseconds; [`stats_phases::take`]
/// drains them. Off (one relaxed load) otherwise; the `stats_timing` debug
/// log stays for nodes with a tracing subscriber.
pub mod stats_phases {
    use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
    pub static ENABLED: AtomicBool = AtomicBool::new(false);
    static RECORDED: parking_lot::Mutex<Vec<(&'static str, Vec<(&'static str, u128)>)>> = parking_lot::const_mutex(Vec::new());

    /// Record one request's `phases` under `kind` (when enabled).
    pub fn record(kind: &'static str, phases: Vec<(&'static str, u128)>) {
        if ENABLED.load(Relaxed) {
            RECORDED.lock().push((kind, phases));
        }
    }

    /// Every record so far, oldest first; the buffer is emptied.
    pub fn take() -> Vec<(&'static str, Vec<(&'static str, u128)>)> {
        std::mem::take(&mut *RECORDED.lock())
    }
}

/// The node holding the primary of every (single-shard) index `targets`
/// names, if there is one; `None` for forwarded requests, multi-shard
/// targets, unresolvable expressions or primaries spread over nodes.
async fn stats_primary_node(node: &Arc<ClusterNode>, req: &RestReq, names: Option<&BTreeSet<String>>) -> Option<String> {
    if req.hops > 0 {
        return None;
    }
    let names = names?;
    let st = node.state();
    let mut owner: Option<String> = None;
    for name in names {
        if crate::shard_engines::shards_of(&st.metadata, name) > 1 {
            return None;
        }
        let p = st.routing_table.shard_table(name, 0)?.active_copies().find(|c| c.primary)?.node_id.clone()?;
        match &owner {
            Some(o) if *o != p => return None,
            _ => owner = Some(p),
        }
    }
    owner
}

/// Refresh / flush / force-merge / cache-clear on every node holding copies;
/// stats-style reads go to one node that holds all copies.
async fn broadcast(node: &Arc<ClusterNode>, req: &RestReq, targets: &str, op: BroadcastOp) -> RestResp {
    // Validate the coordinating request before fan-out: a request error must
    // not become a successful response containing failed shard operations.
    if let Some(error) = broadcast_body_rejection(req) {
        return err(error);
    }
    let params = req.params();
    if matches!(op, BroadcastOp::Flush) && params.get("force").map(String::as_str) == Some("true")
        && params.get("wait_if_ongoing").map(String::as_str) == Some("false") {
        return RestResp::error(400, "action_request_validation_exception", "Validation Failed: 1: wait_if_ongoing must be true for a force flush;");
    }
    if matches!(op, BroadcastOp::ForceMerge) && params.get("only_expunge_deletes").map(String::as_str) == Some("true")
        && params.get("max_num_segments").map(|v| v != "-1").unwrap_or(false) {
        return RestResp::error(400, "action_request_validation_exception", "Validation Failed: 1: cannot set only_expunge_deletes and max_num_segments at the same time, those two parameters are mutually exclusive;");
    }
    if matches!(op, BroadcastOp::ForceMerge) && req.hops == 0
        && params.get("wait_for_completion").map(String::as_str) == Some("false") {
        let mut work = req.clone();
        struct WaitQuery<'a>(&'a str);
        impl std::fmt::Display for WaitQuery<'_>{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{
            for part in self.0.split('&').filter(|p|!p.is_empty()&&!p.starts_with("wait_for_completion=")){write!(f,"{part}&")?;}
            f.write_str("wait_for_completion=true")
        }}
        if let Err(e) = work.set_uri_fmt(format_args!("{}?{}", req.path(),WaitQuery(req.query()))) { return err(e); }
        let task = node.engine.tasks.register("indices:admin/forcemerge", Some(format!("Force merge [{targets}]")), false, true, serde_json::Map::new());
        let task_id = format!("{}:{}", node.local.id, task.id);
        let node = node.clone();
        let targets = targets.to_owned();
        tokio::spawn(async move {
            let response = broadcast_sync(&node, &work, &targets, op).await;
            let value = response.body_json().unwrap_or(Value::Null);
            node.engine.tasks.finish(&task, if response.is_success() { Ok(value) } else { Err(value["error"].clone()) });
        });
        return RestResp::json(200, &json!({"task": task_id}));
    }
    broadcast_sync(node, req, targets, op).await
}

/// A body on a broadcast API that takes none (`_refresh`, `_flush`,
/// `_forcemerge`, `_cache/clear`, `_stats`, `_segments`, ...): ES's REST
/// layer answers 400 `request [M /p] does not support having a body` before
/// any shard runs. The shards' own rejection made it a 200 with
/// `_shards.failed` and no refresh or flush.
pub(crate) fn broadcast_body_rejection(req: &RestReq) -> Option<elasticsearch_engine::error::EsError> {
    if req.body.is_empty() {
        return None;
    }
    elasticsearch_http::body_not_supported(&req.method(), req.path(), elasticsearch_engine::version::current_rest_api_version())
}

async fn broadcast_sync(node: &Arc<ClusterNode>, req: &RestReq, targets: &str, op: BroadcastOp) -> RestResp {
    match op {
        BroadcastOp::Stats => {
            // Phase times of an index stats request (`stats_timing` debug
            // log): where a request's time goes without a profiler.
            let started = std::time::Instant::now();
            // `primaries` sections must come from primary copies (a replica
            // engine's translog, for one, holds no bytes): when one node
            // holds the primaries of every target, it answers.
            // The target expression, resolved once for the primary check and
            // the `_shards` header (it was resolved for each).
            await_mirror(node).await;
            let names: Option<BTreeSet<String>> = {
                let (engine, expr, opts) = (node.engine.clone(), targets.to_string(), ResolveOpts::from_params(&req.params()));
                tokio::task::spawn_blocking(move || engine.resolve(&expr, opts).map(|ts| ts.iter().map(|t| t.svc.name.clone()).collect::<BTreeSet<_>>())).await.ok().and_then(Result::ok)
            };
            if let Some(p) = stats_primary_node(node, req, names.as_ref()).await {
                if p != node.local.id {
                    return forward(node, &p, req).await;
                }
            }
            let primary_us = started.elapsed().as_micros();
            let mut response = read(node, req, targets).await;
            // Stats are a broadcast aggregation: an unallocated index has
            // no responding copies, rather than failing the whole request.
            if response.status == 503 && response.body_json().is_some_and(|value| value["error"]["type"] == "no_shard_available_action_exception") {
                response = local_canonical(node, req).await;
            }
            let read_us = started.elapsed().as_micros();
            let response = crate::stats_shards::adjust_resolved(node, req, targets, names, response).await;
            tracing::debug!(target: "stats_timing", path = %req.path(), primary_us, read_us = read_us - primary_us,
                adjust_us = started.elapsed().as_micros() - read_us, total_us = started.elapsed().as_micros(), "index stats phases");
            let total_us = started.elapsed().as_micros();
            stats_phases::record("index_stats", vec![("primary", primary_us), ("read", read_us - primary_us), ("adjust", total_us - read_us), ("total", total_us)]);
            return response;
        }
        BroadcastOp::Segments | BroadcastOp::ShardStores => return read(node, req, targets).await,
        BroadcastOp::DiskUsage => return disk_usage::disk_usage(node, req, targets).await,
        BroadcastOp::FieldUsage => return disk_usage::field_usage(node, req, targets).await,
        _ => {}
    }
    if req.hops > 0 {
        // Already fanned out: run locally (flush through the gate).
        return local_maintenance(node, req, op).await;
    }
    // Resolve the expression once, here; errors (index_not_found, ...) and
    // empty matches render exactly as the engine renders them.
    await_mirror(node).await;
    let opts = ResolveOpts::from_params(&req.params());
    let engine = node.engine.clone();
    let expr = targets.to_string();
    let resolved = tokio::task::spawn_blocking(move || engine.resolve(&expr, opts).map(|ts| ts.iter().map(|t| t.svc.name.clone()).collect::<BTreeSet<_>>())).await.ok();
    let names = match resolved {
        Some(Ok(n)) if !n.is_empty() => n,
        _ => return local_maintenance(node, req, op).await,
    };
    // ES counts shard copies from the routing table: every copy is in
    // `total`, the active ones on nodes that performed the op succeeded.
    let st = node.state();
    let mut total = 0u64;
    let mut active_on: HashMap<String, u64> = HashMap::new();
    for name in &names {
        for k in 0..crate::shard_engines::shards_of(&st.metadata, name) {
            let Some(t) = st.routing_table.shard_table(name, k) else { continue };
            total += t.copies.len() as u64;
            for c in t.copies.iter().filter(|c| c.state != ShardState::Unassigned) {
                if let Some(n) = &c.node_id {
                    *active_on.entry(n.clone()).or_default() += u64::from(active(c.state));
                }
            }
        }
    }
    let mut ok = 0u64;
    let mut failed = 0u64;
    let mut failures = Vec::new();
    // Force merge used to finish every copy on one node before even asking
    // the next node. Keep local engine merges/gates unchanged, but overlap
    // at most two nodes so co-located test/development nodes do not create
    // unbounded simultaneous merge working sets. Preserve response order.
    use futures::StreamExt;
    let concurrency = if matches!(op, BroadcastOp::ForceMerge) { 2 } else { 1 };
    let recipients: Vec<String> = st.nodes.values().filter(|n| n.is_data() && active_on.contains_key(&n.id))
        .map(|n| n.id.clone()).collect();
    let requests = recipients.into_iter().map(|id| {
        let op = op.clone();
        let node = node.clone();
        let req = req.clone();
        async move {
            let response = if id == node.local.id { local_maintenance(&node, &req, op).await } else { forward(&node, &id, &req).await };
            (id, response)
        }
    });
    for (id, r) in futures::stream::iter(requests).buffered(concurrency).collect::<Vec<_>>().await {
        if r.is_success() {
            ok += active_on[&id];
        } else if r.status != 404 {
            // A 404 from a node for names that resolved here is a copy
            // being rebuilt by peer recovery: not a failure, not counted.
            failed += active_on[&id];
            let reason = r.body_json().and_then(|body| body.get("error").cloned())
                .unwrap_or_else(|| json!({"reason": String::from_utf8_lossy(&r.body)}));
            failures.push(json!({"node": id, "status": r.status, "reason": reason}));
        }
    }
    let mut body = json!({"_shards": {"total": total, "successful": ok, "failed": failed}});
    if !failures.is_empty() {
        body["_shards"]["failures"] = Value::Array(failures);
    }
    RestResp::json(200, &body)
}

async fn local_maintenance(node: &Arc<ClusterNode>, req: &RestReq, op: BroadcastOp) -> RestResp {
    if matches!(op, BroadcastOp::ForceMerge) {
        // ES: merge, then (`flush`, default true) flush. Only the flush must
        // exclude writes, so the translog is trimmed below what every copy
        // acknowledged; the merge itself runs while indexing continues.
        let params = req.params();
        let mut merge = req.clone();
        if params.get("flush").map(String::as_str) != Some("false") {
            if let Err(e) = merge.set_uri_fmt(format_args!("{}{}flush=false", req.uri, if req.uri.contains('?') { "&" } else { "?" })) { return err(e); }
        }
        let merged = maintenance_all_engines(node, &merge).await;
        if !merged.is_success() || params.get("flush").map(String::as_str) == Some("false") {
            return merged;
        }
        let mut flush = req.clone();
        struct FlushQuery<'a>(&'a HashMap<String,String>);
        impl std::fmt::Display for FlushQuery<'_>{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{
            let mut sep="?";
            for key in ["ignore_unavailable","allow_no_indices","expand_wildcards"] {
                if let Some(value)=self.0.get(key){write!(f,"{sep}{key}={value}")?;sep="&";}
            }Ok(())
        }}
        let path=req.path();
        let (base,replacement,suffix)=match path.split_once("/_forcemerge"){Some((base,suffix))=>(base,"/_flush",suffix),None=>(path,"","")};
        if let Err(e) = flush.set_uri_fmt(format_args!("{base}{replacement}{suffix}{}",FlushQuery(&params))) { return err(e); }
        let flushed = Box::pin(local_maintenance(node, &flush, BroadcastOp::Flush)).await;
        return if flushed.is_success() { merged } else { flushed };
    }
    if matches!(op, BroadcastOp::Flush) {
        // No write fence (ES flushes without blocking indexing): each
        // primary's commit point stays below its in-flight writes' claims,
        // and the replication retention `PrimaryCtx::new` installs keeps
        // every translog op some copy (or the tailer) still needs. Ship what
        // is pending first so the replicas' commits cover as much as ours.
        let names = node.engine.resolve(req.segments().first().filter(|s| !s.starts_with('_')).map(String::as_str).unwrap_or("_all"), ResolveOpts::from_params(&req.params()))
            .map(|targets| targets.into_iter().map(|t| t.svc.name.clone()).collect::<BTreeSet<_>>()).unwrap_or_default();
        let prims: Vec<Arc<PrimaryCtx>> = node.shards.all().iter().filter_map(|s| match &s.lock().role {
            crate::shards::Role::Primary(p) if names.contains(&p.index) => Some(p.clone()),
            _ => None,
        }).collect();
        if let Some(p) = prims.iter().find(|p| !still_primary(node, p) || !node.engine_for(p.shard).map(|engine| engine_is(&engine, p)).unwrap_or(false)) {
            return no_longer_primary(p);
        }
        let net = TransportNet(node.clone());
        for p in &prims {
            if let Err(e) = p.replicate_all(&net).await {
                tracing::debug!(index = %p.index, shard = p.shard, "flush: shipping pending operations failed: {e}");
            }
        }
    }
    maintenance_all_engines(node, req).await
}

/// Run every local shard engine, preserving any failure rather than hiding
/// secondary-shard errors behind engine 0's success. Missing local indices are
/// expected on engines hosting other shards and do not override a real result.
async fn maintenance_all_engines(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let mut response = local_engine(node, req).await;
    for (k, _) in node.shard_engines.all() {
        let mut sub = req.clone();
        if let Err(e) = sub.set_shard(k) { return err(e); }
        if let Err(e) = sub.set_uri_fmt(format_args!("{}{}",req.path(),FilteredQuery{query:req.query(),drop:&["ignore_unavailable","allow_no_indices"],append:"ignore_unavailable=true&allow_no_indices=true"})) { return err(e); }
        response = maintenance_response(response, local_engine(node, &sub).await);
    }
    response
}

fn maintenance_response(current: RestResp, next: RestResp) -> RestResp {
    if current.status != 404 && !current.is_success() {
        current
    } else if next.status != 404 && (!next.is_success() || current.status == 404) {
        next
    } else {
        current
    }
}

/// `_reindex` runs on the node holding the destination primary, provided it
/// also holds copies of every source index; it then replicates like any
/// other write.
async fn reindex(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    // Multi-shard source / destination: scroll + routed bulk (cluster-side).
    if let Some(r) = crate::reindex_multi::maybe(node, req).await {
        return r;
    }
    // A freshly auto-created destination can be active in the routing table
    // before its primary context is up here: retry like any write until
    // `timeout` (these failures happen before anything executed).
    let deadline = tokio::time::Instant::now() + request_timeout(req);
    loop {
        let r = reindex_once(node, req).await;
        if !retryable(&r) || is_indeterminate(&r) || !wait_state_change(node, deadline).await {
            return r;
        }
    }
}

async fn reindex_once(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let Some(body) = req.json() else { return local_engine(node, req).await };
    let Some(dest) = body["dest"]["index"].as_str() else { return local_engine(node, req).await };
    let concrete = match concrete_write_index(node, dest, true).await {
        Ok(c) => c,
        Err(r) => return r,
    };
    let Some(pn) = primary_node(node, &concrete) else { return unavailable(&concrete) };
    if pn != node.local.id {
        return forward(node, &pn, req).await;
    }
    let Some(p) = node.shards.primary(&concrete, 0) else { return unavailable(&concrete) };
    execute_write(node, &p, req).await
}

#[allow(dead_code)]
fn active(state: ShardState) -> bool {
    matches!(state, ShardState::Started | ShardState::Relocating)
}

/// Entry points used by the cluster REST module.
pub async fn read_route(node: &Arc<ClusterNode>, req: &RestReq, targets: &str) -> RestResp {
    read(node, req, targets).await
}

pub async fn on_master_route(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    on_master(node, req).await
}

#[cfg(test)]
mod tests {
    #[test]
    fn bulk_shard_counts_preserve_failure_destinations_after_filtering() {
        let mut response = super::RestResp::json(200, &serde_json::json!({"items":[
            {"create":{"_shards":{"total":1,"successful":1,"failed":0}}},
            {"create":{"_shards":{"total":0,"successful":0,"failed":0}}}
        ]}));
        response.push_header((super::failure_store::POSITIONS_HEADER.into(), "[0]".into()));
        super::patch_shards(&mut response, &crate::shards::ShardsInfo { total:3, successful:3, failures:vec![] });
        let body = response.body_json().unwrap();
        assert_eq!(body["items"][0]["create"]["_shards"]["total"], 1);
        assert_eq!(body["items"][1]["create"]["_shards"]["total"], 3);
        assert!(!response.headers.iter().any(|(name, _)| name == super::failure_store::POSITIONS_HEADER));
    }
    use super::*;

    #[test]
    fn stored_ccs_async_result_reports_the_skipped_remote() {
        use elasticsearch_engine::ccr::remote::{CcsPlan, SkippedRemote};
        let plan = CcsPlan { local: vec!["logs".into()], skipped: vec![SkippedRemote { alias: "far".into(), indices: "logs".into(), error_type: "connect_transport_exception".into(), reason: "unreachable".into() }] };
        let text = r#"{"took":3,"timed_out":false,"_shards":{"total":2,"successful":2,"skipped":0,"failed":0},"hits":{"total":{"value":1,"relation":"eq"},"hits":[{"_id":"1","_score":1.0}]}}"#;
        let patched: Value = serde_json::from_str(&with_ccs_clusters(text, &plan).unwrap()).unwrap();
        assert_eq!(patched["hits"]["hits"][0]["_id"], "1");
        assert_eq!(patched["_clusters"]["total"], 2);
        assert_eq!(patched["_clusters"]["skipped"], 1);
        assert_eq!(patched["_clusters"]["details"]["far"]["status"], "skipped");
        assert_eq!(patched["_clusters"]["details"]["(local)"]["status"], "successful");
    }

    #[test]
    fn prefilter_count_replaces_a_forwarded_one() {
        let h = crate::shard_engines::PREFILTER_SHARDS_HEADER;
        let mut headers = vec![(h.to_string(), "1".to_string()), ("authorization".to_string(), "x".to_string())];
        set_prefilter_count(&mut headers, 5);
        assert_eq!(headers.iter().filter(|(k, _)| k == h).map(|(_, v)| v.as_str()).collect::<Vec<_>>(), ["5"]);
        assert!(headers.iter().any(|(k, _)| k == "authorization"));
    }

    #[test]
    fn analytics_collection_metadata_is_serialized_but_events_are_not() {
        for method in ["PUT", "DELETE"] {
            assert!(serialized_engine_metadata(&req(method, "/_application/analytics/shop")));
        }
        for (method, path) in [
            ("GET", "/_application/analytics/shop"),
            ("POST", "/_application/analytics/shop/event/page_view"),
            ("PUT", "/_application/search_application/shop"),
        ] {
            assert!(!serialized_engine_metadata(&req(method, path)));
        }
    }

    #[test]
    fn clear_context_counts_preserve_rejection_and_valid_not_found() {
        let refused = freed_contexts(RestResp::error(429, "circuit_breaking_exception", "request admission refused")).unwrap_err();
        assert_eq!(refused.status, 429);
        assert_eq!(freed_contexts(RestResp::json(404, &json!({"succeeded":true,"num_freed":0}))).unwrap(), 0);
        assert_eq!(freed_contexts(RestResp::json(200, &json!({"succeeded":true,"num_freed":7}))).unwrap(), 7);
        assert_eq!(freed_contexts(RestResp::error(404, "resource_not_found_exception", "node missing")).unwrap_err().status, 404);
        assert_eq!(freed_contexts(RestResp::json(200, &json!({"succeeded":true}))).unwrap_err().status, 500);
    }

    fn req(method: &str, uri: &str) -> RestReq {
        RestReq::try_new(method,uri,&[],&[],0,Dfs::None,None).unwrap()
    }

    #[test]
    fn esql_view_definitions_are_published_but_queries_are_not() {
        for (method, path) in [("PUT", "/_query/view/dogs"), ("DELETE", "/_query/view/dogs,cats")] {
            assert!(engine_api_mutates(&req(method, path)), "{method} {path}");
            assert_eq!(crate::meta_extra::custom_family_of_api("_query"), Some(crate::meta_extra::ESQL_VIEWS_CUSTOM));
        }
        for (method, path) in [("POST", "/_query"), ("POST", "/_query/async"), ("GET", "/_query/view/dogs"), ("GET", "/_query/view")] {
            assert!(!engine_api_mutates(&req(method, path)), "{method} {path}");
        }
    }

    /// B9: a forward sent from engine work on an executor carries the
    /// nested marker (once, also across further hops); the receiving node
    /// reads it into `NestedRequest` (which runs the work unpooled, see the
    /// http crate's nested_requests tests). A forward from outside engine
    /// work is not marked.
    #[test]
    fn forwards_from_engine_work_are_marked_nested_for_the_peer() {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let mut plain = req("POST", "/i/_search");
        mark_nested(&mut plain).unwrap();
        assert!(!is_nested(&plain.headers), "a forward from outside engine work was marked");
        let marked = rt.block_on(elasticsearch_http::pools::run_unpooled(|| {
            let mut fwd = req("POST", "/i/_search");
            mark_nested(&mut fwd).unwrap();
            // Forwarded again by the peer's engine work: still one marker.
            mark_nested(&mut fwd).unwrap();
            fwd
        })).unwrap();
        assert_eq!(marked.headers.iter().filter(|(k, _)| k.eq_ignore_ascii_case(NESTED_HEADER)).count(), 1);
        assert!(is_nested(&marked.headers), "the peer does not see the forward as nested");
    }

    /// Coverage audit P2-5: a shard copy read an older coordinator (no
    /// trips marker) forwarded here and this node's breakers tripped
    /// answers 503 with the breaker's error, so the older coordinator tries
    /// the next copy as ES does. This version's forwards carry the marker
    /// and keep the 429; writes, other 429s, forwards that are not copy
    /// reads, and scroll / PIT reads bound to this copy's context are never
    /// rewritten (combined review LOW).
    #[test]
    fn a_trip_answers_an_older_coordinator_with_a_status_it_retries() {
        let trip = RestResp::json(429, &json!({
            "error": {"root_cause": [{"type": "circuit_breaking_exception", "reason": "[parent] Data too large"}], "type": "circuit_breaking_exception", "reason": "[parent] Data too large", "durability": "TRANSIENT"},
            "status": 429,
        }));
        let copy_read = |method: &str, uri: &str| {
            let mut r = req(method, uri);
            r.push_header(EXEC_LOCAL, "1").unwrap();
            r
        };
        let old = copy_read("POST", "/i/_search");
        let r = for_older_sender(&old, trip.clone());
        assert_eq!(r.status, 503);
        let v = r.body_json().unwrap();
        assert_eq!(v["status"], 503);
        assert_eq!(v["error"]["type"], "circuit_breaking_exception", "the breaker's error is kept");
        assert_eq!(v["error"]["durability"], "TRANSIENT");
        // Wrapped (all shards failed): the root cause decides.
        let wrapped = RestResp::json(429, &json!({
            "error": {"root_cause": [{"type": "circuit_breaking_exception", "reason": "x"}], "type": "search_phase_execution_exception", "reason": "all shards failed"},
            "status": 429,
        }));
        assert_eq!(for_older_sender(&old, wrapped).status, 503);
        let mut new = copy_read("POST", "/i/_search");
        mark_trips(&mut new).unwrap();
        mark_trips(&mut new).unwrap();
        assert_eq!(new.headers.iter().filter(|(k, _)| k.eq_ignore_ascii_case(TRIPS_HEADER)).count(), 1);
        assert_eq!(for_older_sender(&new, trip.clone()).status, 429, "this version's coordinator retries a 429 itself");
        assert_eq!(for_older_sender(&copy_read("POST", "/i/_bulk"), trip.clone()).status, 429, "a write is never retried on another copy");
        assert_eq!(for_older_sender(&req("POST", "/i/_search"), trip.clone()).status, 429, "a whole-request forward keeps ES's status");
        assert_eq!(for_older_sender(&copy_read("POST", "/_search/scroll"), trip.clone()).status, 429, "a scroll continuation has its context on this copy only");
        let mut pit = copy_read("POST", "/_search");
        pit.set_body(br#"{"pit":{"id":"abc"}}"#).unwrap();
        assert_eq!(for_older_sender(&pit, trip.clone()).status, 429, "a PIT read has its context on this copy only");
        let rejected = RestResp::error(429, "es_rejected_execution_exception", "rejected execution");
        assert_eq!(for_older_sender(&old, rejected).status, 429);
    }

    #[test]
    fn canonical_body_renames_like_text_scan() {
        // The former path: validate the whole body, then scan it per uuid.
        let old = |renames: &[(String, String)], body: &[u8]| -> Option<String> {
            match crate::mirror::canonicalize_with(renames, std::str::from_utf8(body).ok()?) {
                std::borrow::Cow::Owned(c) => Some(c),
                std::borrow::Cow::Borrowed(_) => None,
            }
        };
        let renames = vec![("LoCaLuUiD0000000000001".to_string(), "ClUsTeRuUiD00000000001".to_string()), ("LoCaLuUiD0000000000002".to_string(), "ClUsTeRuUiD00000000002".to_string())];
        let big = format!(r#"{{"hits":[{{"_source":{{"t":"{}"}}}}]}}"#, "é x \\n ".repeat(50_000));
        let bodies: Vec<Vec<u8>> = vec![
            br#"{"index":{"uuid":"LoCaLuUiD0000000000001"}}"#.to_vec(),
            "{\"a\":\"LoCaLuUiD0000000000002 and LoCaLuUiD0000000000001\",\"b\":\"é\"}".as_bytes().to_vec(),
            big.clone().into_bytes(),
            format!("{big}LoCaLuUiD0000000000002").into_bytes(),
            b"\xff\xfe LoCaLuUiD0000000000001".to_vec(),
            Vec::new(),
        ];
        for b in &bodies {
            assert_eq!(canonical_body(&renames, b), old(&renames, b), "{}", String::from_utf8_lossy(&b[..b.len().min(80)]));
            assert_eq!(canonical_body(&[], b), None);
        }
        assert_eq!(canonical_body(&renames, &bodies[0]).as_deref(), Some(r#"{"index":{"uuid":"ClUsTeRuUiD00000000001"}}"#));
    }

    #[test]
    fn raw_cluster_search_response_preserves_source_and_error_status() {
        let text = r#"{"hits":{"hits":[{"_source":{"z":"\ud800","n":1.50,"a":2}}]}}"#;
        let response = RestResp::new(200,vec![],text.as_bytes().to_vec());
        assert_eq!(raw_cluster_response(response).unwrap(), (200, text.to_owned()));
        for (status, text) in [
            (429, r#"{"error":{"type":"exception","reason":"\ud800","root_cause":[{"reason":"original"}],"stack_trace":"kept"},"status":429}"#),
            (503, r#"{"unexpected":"failure without error member"}"#),
        ] {
            let response = RestResp::new(status,vec![],text.as_bytes().to_vec());
            assert_eq!(raw_cluster_response(response).unwrap(), (status, text.to_owned()));
        }
        assert!(raw_cluster_response(RestResp::new(200,vec![],b"null".to_vec())).is_err());
    }

    #[test]
    fn copy_local_msearch_has_no_transport_deadline_that_duplicates_items() {
        for path in ["/_msearch", "/i/_msearch?max_concurrent_searches=1"] {
            let mut batch = req("POST", path);
            // The ordinary outer request keeps its transport deadline;
            // only the coordinator's local-copy optimization is unbounded.
            assert_eq!(forwarding_timeout(&batch), Some(Duration::from_secs(300)));
            batch.set_read_copies(&["i".to_string()]).unwrap();
            assert_eq!(forwarding_timeout(&batch), None);
        }
        let mut search = req("POST", "/i/_search?timeout=1s");
        search.set_read_copies(&["i".to_string()]).unwrap();
        assert_eq!(forwarding_timeout(&search), Some(Duration::from_secs(300)));
    }

    #[test]
    fn force_merge_forwarding_waits_for_completion_without_response_deadline() {
        for path in ["/_forcemerge", "/i/_forcemerge?max_num_segments=1", "/i/_forcemerge?wait_for_completion=true"] {
            assert_eq!(forwarding_timeout(&req("POST", path)), None);
        }
        for path in ["/i/_forcemerge?wait_for_completion=false", "/i/_flush", "/i/_search", "/i/_doc/1"] {
            assert_eq!(forwarding_timeout(&req("POST", path)), Some(Duration::from_secs(300)));
        }
    }

    #[test]
    fn maintenance_preserves_secondary_errors_and_ignores_missing_engines() {
        let ok = || RestResp::json(200, &json!({"_shards": {"successful": 1}}));
        let missing = || RestResp::error(404, "index_not_found_exception", "absent locally");
        let failed = || RestResp::error(500, "merge_exception", "secondary merge failed");
        let response = maintenance_response(ok(), failed());
        assert_eq!(response.status, 500);
        assert_eq!(response.body_json().unwrap()["error"]["reason"], "secondary merge failed");
        assert_eq!(maintenance_response(response, ok()).status, 500);
        assert_eq!(maintenance_response(missing(), ok()).status, 200);
        assert_eq!(maintenance_response(ok(), missing()).status, 200);
        assert_eq!(maintenance_response(missing(), failed()).status, 500);
        assert_eq!(maintenance_response(failed(), missing()).status, 500);
        assert_eq!(maintenance_response(missing(), missing()).status, 404);
    }

    #[test]
    fn client_filter_path_is_taken_off_the_request() {
        let mut r = req("GET", "/i/_search?size=2&filter_path=hits.hits._id,-took&pretty");
        assert_eq!(take_filter_path(&mut r).unwrap(), Some(("hits.hits._id,-took".to_string(), true)));
        assert_eq!(r.uri, "/i/_search?size=2&pretty");
        let mut r = req("GET", "/i/_search?filter_path=hits.total");
        assert_eq!(take_filter_path(&mut r).unwrap(), Some(("hits.total".to_string(), false)));
        assert_eq!(r.uri, "/i/_search");
        // Absent or empty: nothing to filter, the request is untouched.
        for uri in ["/i/_search?pretty", "/i/_search?filter_path="] {
            let mut r = req("GET", uri);
            assert_eq!(take_filter_path(&mut r).unwrap(), None);
            assert_eq!(r.uri, uri);
        }
    }

    /// The merge needs whole parts: filtering each shard's part first loses
    /// the sort values (wrong order) and the `_shards` header, and the merge
    /// re-adds `took` / `_shards` / `max_score` / `profile`. Filtering the
    /// merged body once gives ES's answer.
    #[test]
    fn filter_path_applies_after_the_merge() {
        let part = |shard: &[(&str, i64)]| {
            json!({"took": 3, "timed_out": false, "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
                   "hits": {"total": {"value": shard.len(), "relation": "eq"}, "max_score": null,
                            "hits": shard.iter().map(|(id, n)| json!({"_index": "i", "_id": id, "_score": null, "sort": [n]})).collect::<Vec<_>>()},
                   "profile": {"shards": [{"id": "x"}]}})
        };
        let parts = [part(&[("a", 9), ("b", 1)]), part(&[("c", 5)])];
        let body = json!({"sort": [{"n": "desc"}]});
        let merged = fanout::merge_search(&parts, &body, &HashMap::new());
        let out = crate::rest::util::render_json(RestResp::json(200, &merged), Some("hits.hits._id"), false, false);
        assert_eq!(out.body_json().unwrap(), json!({"hits": {"hits": [{"_id": "a"}, {"_id": "c"}, {"_id": "b"}]}}));
        let out = crate::rest::util::render_json(RestResp::json(200, &merged), Some("_shards,hits.total"), false, false);
        assert_eq!(out.body_json().unwrap(), json!({"_shards": {"total": 2, "successful": 2, "skipped": 0, "failed": 0}, "hits": {"total": {"value": 3, "relation": "eq"}}}));
        // What per-part filtering produced: no sort values to merge by, zero
        // shard counts and the merge's own fields back in the response.
        let filtered: Vec<Value> = parts.iter().map(|p| elasticsearch_http::filter_path(p, "hits.hits._id")).collect();
        let wrong = fanout::merge_search(&filtered, &body, &HashMap::new());
        assert_eq!(wrong["_shards"]["total"], 0);
        assert!(wrong.get("took").is_some());
    }

    #[test]
    fn rebuilt_json_preserves_compatibility_version() {
        for version in [None, Some("8"), Some("9")] {
            let mut base = req("GET", "/i/_search?scroll=1m");
            base.push_header("authorization", "test").unwrap();
            if let Some(version) = version {
                base.push_header("Accept", &format!("application/vnd.elasticsearch+json; compatible-with={version}")).unwrap();
            }
            let expected = version.map(|v| format!("application/vnd.elasticsearch+json; compatible-with={v}")).unwrap_or_else(|| "application/json".into());
            let body = json!({"scroll_id": "context", "scroll": "1m"});
            for derived in [base.try_rebuilt(&body, |_| true).unwrap(), contexts::json_req(&base, "POST", "/_search/scroll", &body).unwrap()] {
                assert_eq!(derived.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case("content-type")).unwrap().1, expected);
                assert!(derived.headers.contains(&("authorization".into(), "test".into())));
                assert_eq!(derived.json(), Some(body.clone()));
            }
            let empty = contexts::json_req(&base, "DELETE", "/_search/scroll/_all", &Value::Null).unwrap();
            assert!(!empty.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("content-type")));
        }
        let mut invalid = req("POST", "/_search/scroll");
        invalid.set_body(b"{}").unwrap();
        invalid.set_headers(&[("accept".into(), "application/vnd.elasticsearch+json; compatible-with=8".into()), ("content-type".into(), "application/json".into())]).unwrap();
        // Internal rebuilding must not repair an invalid external header pair.
        assert_eq!(invalid.try_rebuilt(&json!({"scroll_id": "context"}), |_| true).unwrap().json_content_type(), "application/json");
    }

    fn gunzip(b: &[u8]) -> Vec<u8> {
        use std::io::Read;
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(b).read_to_end(&mut out).unwrap();
        out
    }

    #[test]
    fn product_header_is_singleton_without_collapsing_warnings() {
        let mut reply = RestResp::json(200, &json!({"ok":true}));
        reply.extend_headers([
            ("X-Elastic-Product".into(), "Elasticsearch, Elasticsearch".into()),
            ("x-elastic-product".into(), "Elasticsearch".into()),
            ("warning".into(), "first".into()),
            ("warning".into(), "second".into()),
        ]);
        let response = to_http(reply);
        assert_eq!(response.headers().get_all("x-elastic-product").iter().count(), 1);
        assert_eq!(response.headers()["x-elastic-product"], "Elasticsearch");
        assert_eq!(response.headers().get_all("warning").iter().count(), 2);
    }

    /// Regression: a client `Accept-Encoding: gzip` used to reach the
    /// per-shard bulk sub-requests, whose gzipped replies the coordinator
    /// could not parse ("unreadable bulk response" for every item).
    #[tokio::test]
    async fn gzip_is_negotiated_only_at_the_client_edge() {
        let body = json!({"took": 1, "errors": false, "items": [{"index": {"_index": "i", "_id": "1", "result": "created", "status": 201}}]});
        let plain = RestResp::json(200, &body);
        assert!(plain.body.len() > 64);
        let gz = gzip_response(plain.clone());
        assert_eq!(gz.headers()["content-encoding"], "gzip");
        assert_eq!(gz.headers()["vary"], "Accept-Encoding");
        let bytes = axum::body::to_bytes(gz.into_body(), usize::MAX).await.unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&gunzip(&bytes)).unwrap(), body);
        // Already-gzipped and tiny bodies are left alone.
        let mut encoded = plain.clone();
        encoded.replace_body(bytes.to_vec());
        encoded.push_header(("content-encoding".into(), "gzip".into()));
        let twice = gzip_response(encoded);
        assert_eq!(axum::body::to_bytes(twice.into_body(), usize::MAX).await.unwrap(), bytes);
        let tiny = RestResp::json(200, &json!({"a": 1}));
        assert!(!gzip_response(tiny).headers().contains_key("content-encoding"));

        // Gzipped request bodies are inflated and the coding headers dropped.
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        std::io::Write::write_all(&mut enc, b"{\"index\":{}}\n{\"f\":1}\n").unwrap();
        let mut h = vec![("Content-Encoding".to_string(), "gzip".to_string()), ("content-length".into(), "9".into()), ("Accept-Encoding".into(), "gzip, deflate".into())];
        let out = decode_request_body(&mut h, enc.finish().unwrap(), MAX_CONTENT_LENGTH).unwrap();
        assert_eq!(out, b"{\"index\":{}}\n{\"f\":1}\n");
        assert!(!header_has(&h, "content-encoding", "gzip"));
        assert!(header_has(&h, "accept-encoding", "gzip"));
        let mut bad = vec![("content-encoding".to_string(), "gzip".to_string())];
        assert_eq!(decode_request_body(&mut bad, b"nope".to_vec(), MAX_CONTENT_LENGTH).unwrap_err().status, 400);
        // A gzip bomb is cut off at the limit: 413, not an unbounded inflate.
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
        std::io::Write::write_all(&mut enc, &vec![b' '; 4 << 20]).unwrap();
        let bomb = enc.finish().unwrap();
        assert!(bomb.len() < 64 << 10);
        let mut h = vec![("content-encoding".to_string(), "gzip".to_string())];
        assert_eq!(decode_request_body(&mut h, bomb.clone(), 1 << 20).unwrap_err().status, 413);
        let mut h = vec![("content-encoding".to_string(), "gzip".to_string())];
        assert_eq!(decode_request_body(&mut h, bomb, 4 << 20).unwrap().len(), 4 << 20);
    }

    #[test]
    fn incoming_search_part_rejects_before_lossless_decode() {
        let breakers=elasticsearch_engine::memory::Breakers::leak_for_tests(1<<30);
        breakers.apply_settings(&|key|match key {
            "indices.breaker.request.limit"=>Some("64kb".into()),
            "indices.breaker.total.limit"=>Some("-1".into()),_=>None,
        });
        let mut sources=elasticsearch_engine::source_json::ResponseSources::default();
        let memory=elasticsearch_engine::memory::RequestMemory::on(breakers);
        let raw=format!("{{\"aggregations\":{{\"s\":{{\"value\":\"{}\"}}}}}}","x".repeat(1000));
        assert_eq!(parse_search_part(&mut sources,raw.as_bytes(),&memory).unwrap_err().status,429);
        assert_eq!(breakers.request.reserved(),0);
        let memory=elasticsearch_engine::memory::RequestMemory::on(breakers);
        let raw=br#"{"hits":{"hits":[{"_index":"i","_id":"1","_source":{"v":"\ud800"}}]}}"#;
        let parsed=parse_search_part(&mut sources,raw,&memory).unwrap();
        assert!(breakers.request.reserved()>0);
        assert!(String::from_utf8(sources.render(&parsed)).unwrap().contains("\\ud800"));
        drop(parsed);drop(sources);drop(memory);
        assert_eq!(breakers.request.reserved(),0);
    }

    #[test]
    fn shard_closed_is_retryable() {
        let r = RestResp::error(500, "index_shard_closed_exception", "CurrentState[CLOSED] operation only allowed when not closed");
        assert!(retryable(&r));
    }

    fn some_stats() -> elasticsearch_http::DfsStats {
        elasticsearch_http::DfsStats {
            terms: vec![elasticsearch_http::DfsTerm { field: "t".into(), term: b"fox".to_vec(), doc_freq: 3, total_term_freq: 5, field_doc_count: 9 }],
            fields: vec![elasticsearch_http::DfsField { field: "t".into(), doc_count: 9, sum_total_term_freq: 40 }],
        }
    }

    #[test]
    fn dfs_rides_the_transport() {
        for dfs in [Dfs::None, Dfs::Collect, Dfs::Use(SharedDfsStats::try_new(&some_stats()).unwrap())] {
            let mut r = req("POST", "/i/_search");
            r.set_dfs(dfs).unwrap();
            let back: RestReq = bincode::deserialize(&bincode::serialize(&r).unwrap()).unwrap();
            assert_eq!(back.dfs, r.dfs);
            assert_eq!(back.uri, r.uri);
        }
        // A request from a node without the field reads as no DFS.
        let old = json!({"method": "GET", "uri": "/i/_search", "headers": [], "body": [], "hops": 1});
        let r: RestReq = serde_json::from_value(old).unwrap();
        assert_eq!(r.dfs, Dfs::None);
    }

    /// Cluster async searches: stored under the `~a` key, rendered with the
    /// wrapped id, a failure keeps its original `error` object and status,
    /// and delete / expiry drop them.
    #[test]
    fn cluster_async_store_and_render() {
        use elasticsearch_engine::async_search::AsyncSearch;
        use elasticsearch_engine::error::EsError;
        let s = AsyncSearch::new(2, Duration::from_secs(60));
        let c = Arc::new(contexts::ClusterAsync { s: s.clone(), error: parking_lot::Mutex::new(None), warnings: parking_lot::Mutex::new(Vec::new()) });
        let key = format!("{}{}", contexts::ASYNC_PREFIX, s.id);
        let wrapped = format!("{}.{key}", "n".repeat(22));
        assert_eq!(unwrap_id(&wrapped), Some(("n".repeat(22).as_str(), key.as_str())));
        contexts::async_put(key.clone(), c.clone());
        let v = contexts::async_render(&contexts::async_get(&key).unwrap(), true, &wrapped, false).body_json().unwrap();
        assert_eq!((v["id"].as_str(), v["is_running"].as_bool()), (Some(wrapped.as_str()), Some(true)));
        s.complete(Ok(r#"{"hits":{"total":{"value":3}}}"#.into()));
        let v = contexts::async_render(&c, true, &wrapped, false).body_json().unwrap();
        assert_eq!((v["is_running"].as_bool(), v["response"]["hits"]["total"]["value"].as_u64()), (Some(false), Some(3)));
        assert!(contexts::async_render(&c, false, &wrapped, false).body_json().unwrap().get("id").is_none());
        let st = contexts::async_status(&c, &wrapped).body_json().unwrap();
        assert_eq!((st["id"].as_str(), st["completion_status"].as_u64()), (Some(wrapped.as_str()), Some(200)));
        assert!(contexts::async_remove(&key).is_some() && contexts::async_get(&key).is_none());

        let f = AsyncSearch::new(1, Duration::from_secs(60));
        let e = json!({"root_cause": [{"type": "query_shard_exception", "reason": "boom"}], "type": "search_phase_execution_exception", "reason": "all shards failed", "phase": "dfs"});
        let c = contexts::ClusterAsync { s: f.clone(), error: parking_lot::Mutex::new(Some(e.clone())), warnings: parking_lot::Mutex::new(Vec::new()) };
        f.complete(Err(EsError::new(400, "search_phase_execution_exception", "all shards failed")));
        let r = contexts::async_render(&c, true, &wrapped, false);
        assert_eq!((r.status, r.body_json().unwrap()["error"].clone()), (400, e));

        let gone = AsyncSearch::new(1, Duration::from_millis(0));
        contexts::async_put("~agone".into(), Arc::new(contexts::ClusterAsync { s: gone, error: parking_lot::Mutex::new(None), warnings: parking_lot::Mutex::new(Vec::new()) }));
        assert!(contexts::async_get("~agone").is_none());
    }

    #[test]
    fn async_time_params() {
        let p: HashMap<String, String> = [("keep_alive".to_string(), "5d".to_string()), ("wait_for_completion_timeout".to_string(), "10".to_string())].into();
        assert_eq!(async_time_param(&p, "keep_alive", "1s").unwrap(), Duration::from_secs(5 * 86_400));
        assert_eq!(async_time_param(&p, "missing", "1s").unwrap(), Duration::from_secs(1));
        let e = async_time_param(&p, "wait_for_completion_timeout", "1s").unwrap_err();
        assert_eq!((e.status, e.reason.as_str()), (400, "failed to parse setting [wait_for_completion_timeout] with value [10] as a time value: unit is missing or unrecognized"));
    }

    #[test]
    fn prepared_application_plan_roundtrips_over_transport() {
        let mut request = req("POST", "/_application/search_application/app/_search");
        request.set_app_plan(Some(AppSearchPlan::try_new(&["docs".into()], br#"{"query":{"match_all":{}}}"#).unwrap())).unwrap();
        let decoded: RestReq = bincode::deserialize(&bincode::serialize(&request).unwrap()).unwrap();
        let plan = decoded.app_plan.as_ref().unwrap();
        assert_eq!(plan.indices, vec!["docs"]);
        assert_eq!(serde_json::from_slice::<Value>(&plan.query_json).unwrap()["query"], json!({"match_all":{}}));
    }

    /// Byte bodies keep the wire format of a plain `Vec<u8>` field.
    #[test]
    fn byte_bodies_keep_their_wire_format() {
        #[derive(Serialize, Deserialize)]
        struct PlainResp {
            status: u16,
            headers: Vec<(String, String)>,
            body: Vec<u8>,
        }
        let body: Vec<u8> = (0..=255u8).chain(br#"{"hits":[]}"#.iter().copied()).collect();
        let resp = RestResp::new(201,vec![("a".into(), "b".into())],body.clone());
        let plain = PlainResp { status: 201, headers: vec![("a".into(), "b".into())], body: body.clone() };
        assert_eq!(bincode::serialize(&resp).unwrap(), bincode::serialize(&plain).unwrap());
        let back: RestResp = bincode::deserialize(&bincode::serialize(&plain).unwrap()).unwrap();
        assert_eq!(back.body, body);
        let old: PlainResp = bincode::deserialize(&bincode::serialize(&resp).unwrap()).unwrap();
        assert_eq!(old.body, body);
        // JSON stays an array of numbers.
        let j = serde_json::to_value(&resp).unwrap();
        assert_eq!(j["body"], serde_json::to_value(&plain).unwrap()["body"]);
        assert_eq!(serde_json::from_value::<RestResp>(j).unwrap().body, body);
        let r = req("POST", "/i/_search");
        let jr: RestReq = serde_json::from_value(serde_json::to_value(&r).unwrap()).unwrap();
        assert_eq!(jr.body, r.body);
    }

    /// Rolling upgrade: a pre-6b2a119 node sends `RestReq` without `dfs`.
    #[test]
    fn old_rest_req_decodes_on_new_code() {
        #[derive(Serialize)]
        struct OldRestReq {
            method: String,
            uri: String,
            headers: Vec<(String, String)>,
            body: Vec<u8>,
            hops: u8,
        }
        let old = OldRestReq { method: "POST".into(), uri: "/i/_search?q=a".into(), headers: vec![("content-type".into(), "application/json".into())], body: br#"{"size":1}"#.to_vec(), hops: 2 };
        let bytes = bincode::serialize(&old).unwrap();
        let r: RestReq = bincode::deserialize(&bytes).expect("old shape decodes");
        assert_eq!((r.method.as_str(), r.uri.as_str(), r.hops), ("POST", "/i/_search?q=a", 2));
        assert_eq!(r.headers, old.headers);
        assert_eq!(r.body, old.body);
        assert_eq!(r.dfs, Dfs::None);
        // And the other direction: an old node decodes a new request
        // (trailing `dfs` bytes ignored).
        let mut new = req("POST", "/i/_search");
        new.set_dfs(Dfs::Use(SharedDfsStats::try_new(&some_stats()).unwrap())).unwrap();
        #[derive(Deserialize)]
        struct OldReq {
            method: String,
            uri: String,
            headers: Vec<(String, String)>,
            body: Vec<u8>,
            hops: u8,
        }
        let o: OldReq = bincode::deserialize(&bincode::serialize(&new).unwrap()).expect("new shape decodes on old code");
        assert_eq!((o.method.as_str(), o.uri.as_str(), &o.headers, o.body.as_slice(), o.hops),
            (new.method.as_str(), new.uri.as_str(), &new.headers, new.body.as_ref(), new.hops));
        // A truncated required field is still an error.
        assert!(bincode::deserialize::<RestReq>(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn dfs_round_is_wanted_like_es() {
        let p = |q: &str| req("POST", &format!("/i/_search?{q}")).params();
        let text = json!({"query": {"match": {"t": "fox"}}});
        let r = req("POST", "/i/_search");
        let dfs = p("search_type=dfs_query_then_fetch");
        assert!(dfs_wanted(&r, &text, &dfs, 3));
        // query_then_fetch (the default) and a single shard: no round.
        assert!(!dfs_wanted(&r, &text, &p(""), 3));
        assert!(!dfs_wanted(&r, &text, &dfs, 1));
        // kNN folded into a query forces it; a lone kNN scores no text.
        let knn = json!({"field": "v", "query_vector": [1.0], "k": 3, "num_candidates": 10});
        assert!(dfs_wanted(&r, &json!({"query": text["query"], "knn": knn}), &p(""), 3));
        assert!(!dfs_wanted(&r, &json!({"knn": knn}), &dfs, 3));
        // Never a nested round.
        let mut inner = r.clone();
        inner.set_dfs(Dfs::Collect).unwrap();
        assert!(!dfs_wanted(&inner, &text, &dfs, 3));
        inner.set_dfs(Dfs::Use(SharedDfsStats::try_new(&some_stats()).unwrap())).unwrap();
        assert!(!dfs_wanted(&inner, &text, &dfs, 3));
    }

    #[test]
    fn dfs_request_rewrites() {
        let c = dfs_collect_req(&req("POST", "/i/_search?filter_path=hits&typed_keys=true&pretty")).unwrap();
        assert_eq!(c.uri, "/i/_search?typed_keys=true");
        assert_eq!(c.dfs, Dfs::Collect);
        assert_eq!(force_dfs(req("POST", "/i/_search")).unwrap().uri, "/i/_search?search_type=dfs_query_then_fetch");
        assert_eq!(force_dfs(req("POST", "/i/_search?search_type=query_then_fetch&size=2")).unwrap().uri, "/i/_search?size=2&search_type=dfs_query_then_fetch");
    }

    #[test]
    fn shards_preference_parses_list_and_tail() {
        assert_eq!(shards_preference("_shards:0,2"), Some([0, 2].into_iter().collect()));
        assert_eq!(shards_preference("_shards:1|_local"), Some([1].into_iter().collect()));
        assert_eq!(shards_preference("_local"), None);
        assert!(is_shards_preference("preference=_shards%3A1"));
        assert!(is_shards_preference("preference=_shards:1"));
        assert!(!is_shards_preference("preference=abc"));
    }

    /// What `local_engine_once` does: the engine REST layer, index by name.
    async fn by_name(router: &axum::Router, uri: &str, doc: Value) -> RestResp {
        let r = Request::builder().method("PUT").uri(uri).header("content-type", "application/json").body(Body::from(doc.to_string())).unwrap();
        let resp = router.clone().oneshot(r).await.unwrap();
        let status = resp.status().as_u16();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap().to_vec();
        RestResp::new(status,Vec::new(),body)
    }

    /// FAULT_RESULTS.md section 10 (G3, majority-minority seed 1), forced
    /// deterministically: a write is in flight on a primary context while
    /// the node's engine copy of that index is removed and re-created
    /// (`mirror::apply` / `delete_local_copy` + `create_index`). The REST
    /// layer resolves the index by name, so the op is sequenced by the new
    /// empty instance: 201, a seq_no the primary already handed out. Before
    /// the guard this response was replicated "through" the primary's max
    /// and acknowledged (seq_no acked twice, the earlier doc lost).
    #[tokio::test]
    async fn g3_write_sequenced_by_a_recreated_engine_copy_is_not_acked() {
        let dir = tempfile::tempdir().unwrap();
        let engine = elasticsearch_engine::node::Node::open(dir.path()).unwrap();
        let router = elasticsearch_http::router(engine.clone());
        engine.create_index("g3", &json!({})).unwrap();
        let svc = engine.get_index("g3").unwrap();
        let p = PrimaryCtx::new("g3", 0, "aid-primary", 1, svc.clone()).unwrap();

        // Writes on the primary's own copy are sequenced above its max and pass.
        for i in 0..3 {
            assert!(engine_is(&engine, &p));
            let before = p.copy().shard.max_seq_no();
            let r = by_name(&router, &format!("/g3/_doc/c16-{i}"), json!({"v": i})).await;
            assert_eq!(r.status, 201, "{}", String::from_utf8_lossy(&r.body));
            assert_eq!(foreign_seq_no(&engine, &p, before, p.copy().shard.max_seq_no(), &r), None);
        }
        let before = p.copy().shard.max_seq_no();
        assert_eq!(before, 2);

        // The interleaving: past the gate check, the copy is wiped and the
        // mirror re-creates an empty instance under the same name.
        crate::meta_extra::delete_local_copy(&engine, "g3").unwrap();
        engine.create_index("g3", &json!({})).unwrap();
        let r = by_name(&router, "/g3/_doc/c16-452", json!({"v": 452})).await;
        assert_eq!(r.status, 201);
        let v = r.body_json().unwrap();
        assert!(v["_seq_no"].as_i64().unwrap() <= before, "op re-used a seq_no the primary already assigned: {v}");
        let why = foreign_seq_no(&engine, &p, before, p.copy().shard.max_seq_no(), &r).expect("a foreign-sequenced write must not be acknowledged");
        assert!(why.contains("replaced"), "{why}");

        // A write arriving after the swap is refused before it executes.
        assert!(!engine_is(&engine, &p));
    }

    #[test]
    fn generated_ids_look_like_es_auto_ids() {
        let (a, b) = (new_doc_id(), new_doc_id());
        assert_eq!(a.len(), 20);
        assert!(a.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_'));
        assert_ne!(a, b);
    }

    #[test]
    fn auto_id_index_becomes_a_create_with_a_fixed_id() {
        let (r, op) = prepare_write(&req("POST", "/i/_doc"), || "ID".into()).unwrap();
        assert_eq!(r.unwrap().uri, "/i/_doc/ID?op_type=create");
        assert!(op.unwrap().generated);
        let (r, _) = prepare_write(&req("POST", "/i/_doc/?refresh=true&pipeline=p"), || "ID".into()).unwrap();
        assert_eq!(r.unwrap().uri, "/i/_doc/ID?refresh=true&pipeline=p&op_type=create");
        let (r, _) = prepare_write(&req("POST", "/i/_doc?op_type=create"), || "ID".into()).unwrap();
        assert_eq!(r.unwrap().uri, "/i/_doc/ID?op_type=create");
        // The rewritten request is an explicit create: no second id on a
        // retry or on the node it is forwarded to.
        let (r, op) = prepare_write(&req("POST", "/i/_doc/ID?op_type=create"), || panic!("second id")).unwrap();
        assert!(r.is_none());
        assert!(!op.unwrap().generated);
    }

    #[test]
    fn other_writes_are_left_alone() {
        for (m, u) in [("POST", "/i/_doc?op_type=index"), ("POST", "/i/_doc?if_seq_no=1&if_primary_term=1"), ("PUT", "/i/_doc/1"), ("POST", "/i/_update/1"), ("DELETE", "/i/_doc/1"), ("POST", "/i/_doc/1")] {
            let (r, op) = prepare_write(&req(m, u), || panic!("id generated")).unwrap();
            assert!(r.is_none() && op.is_none(), "{m} {u}");
        }
        let (r, op) = prepare_write(&req("PUT", "/i/_create/a%20b"), || panic!("id generated")).unwrap();
        assert!(r.is_none());
        let op = op.unwrap();
        assert_eq!(op.raw_id, "a%20b");
        assert!(!op.generated);
    }

    #[test]
    fn only_lost_responses_are_indeterminate() {
        assert!(is_indeterminate(&RestResp::error(502, "node_disconnected_exception", "x")));
        assert!(is_indeterminate(&RestResp::error(500, "receive_timeout_transport_exception", "x")));
        assert!(!is_indeterminate(&RestResp::error(502, "connect_transport_exception", "x")));
        assert!(!is_indeterminate(&RestResp::error(409, "version_conflict_engine_exception", "x")));
    }

    /// A write the primary executed but could not replicate (replica or
    /// primary node dying) is retried on the next cluster state instead of
    /// answering a final 500, and stays indeterminate so a retry's own-echo
    /// conflict is never reported as a definite failure; never a 2xx.
    #[test]
    fn executed_but_not_acked_is_retried_and_indeterminate() {
        let r = not_acked("i", 2, "replication failed: boom");
        assert!(!r.is_success());
        assert_eq!(r.status, 500);
        let v = r.body_json().unwrap();
        assert_eq!(v["error"]["type"], "retry_on_primary_exception");
        assert_eq!(v["error"]["reason"], "[i][2] replication failed: boom");
        assert!(retryable(&r));
        assert!(is_indeterminate(&r));
        let closed = RestResp::error(500, "node_closed_exception", "node closed");
        assert!(retryable(&closed) && is_indeterminate(&closed));
        // Determinate refusals stay retryable without being indeterminate.
        assert!(retryable(&unavailable_shard("i", 0)) && !is_indeterminate(&unavailable_shard("i", 0)));
    }

    /// A forward answered by a node of the previous release while it shut
    /// down (its aborted local auto-create) is the node closing: retried on
    /// the next state like `node_closed_exception`, never a final 500.
    #[test]
    fn a_forward_aborted_by_the_targets_shutdown_is_retried() {
        let aborted = RestResp::error(500, "exception", "local handler for [internal:cluster/auto_create] failed: task 531 was cancelled");
        assert!(!retryable(&aborted), "as answered by the old release");
        let r = closing_peer_answer(aborted);
        assert_eq!(r.status, 500);
        let v = r.body_json().unwrap();
        assert_eq!(v["error"]["type"], "node_closed_exception", "{v}");
        assert!(retryable(&r) && is_indeterminate(&r));
        // Other failures, and a handler that panicked, are left as they are.
        let panicked = RestResp::error(500, "exception", "local handler for [internal:cluster/auto_create] failed: task 7 panicked with message \"boom\"");
        assert_eq!(closing_peer_answer(panicked).body_json().unwrap()["error"]["type"], "exception");
        let other = RestResp::error(500, "exception", "task 3 was cancelled");
        assert_eq!(closing_peer_answer(other).body_json().unwrap()["error"]["type"], "exception");
        assert_eq!(closing_peer_answer(unavailable_shard("i", 0)).status, unavailable_shard("i", 0).status);
    }

    /// FAULT_RESULTS.md section 10: after a lost response, a retry's
    /// conflict or delete-not-found may be the request's own doing.
    #[test]
    fn answers_a_lost_attempt_can_have_caused() {
        let not_found = json!({"result": "not_found", "_version": 1});
        assert!(own_echo(false, 409, &json!({"error": {"type": "version_conflict_engine_exception"}})));
        assert!(own_echo(true, 404, &not_found));
        // An index never answers not_found, and a missing index is definite.
        assert!(!own_echo(false, 404, &not_found));
        assert!(!own_echo(true, 404, &json!({"error": {"type": "index_not_found_exception"}})));
        assert!(!own_echo(true, 200, &json!({"result": "deleted"})));
        assert!(!own_echo(false, 400, &json!({})));
    }

    #[test]
    fn a_conditional_write_recognises_its_own_document() {
        let src = json!({"v": 7});
        let doc = json!({"found": true, "_seq_no": 9694, "_version": 12, "_source": {"v": 7}});
        let cas = Condition { if_seq_no: Some(9655), version: None };
        assert!(is_own_write(&doc, &src, cas));
        // Someone else's value, an untouched document, a missing document.
        assert!(!is_own_write(&json!({"found": true, "_seq_no": 9694, "_source": {"v": 8}}), &src, cas));
        assert!(!is_own_write(&json!({"found": true, "_seq_no": 9655, "_source": {"v": 7}}), &src, cas));
        assert!(!is_own_write(&json!({"found": false}), &src, cas));
        // External versioning: exactly the version sent.
        let ext = Condition { if_seq_no: None, version: Some(12) };
        assert!(is_own_write(&doc, &src, ext));
        assert!(!is_own_write(&doc, &src, Condition { if_seq_no: None, version: Some(11) }));
        // Without a condition there is nothing to recognise it by.
        assert!(!is_own_write(&doc, &src, Condition::default()));
    }

    /// flapping seed 1: a CAS on jepsen-a/r3 sent to n4 lost
    /// its answer while n3 (term 1) was primary; n3 applied it at seq 4337
    /// and replicated it to n4's copy, but the master failed n3 and n4 and
    /// promoted n2 (term 2), which never received it and used seq 4337
    /// itself. The retry got 409; the confirming GET was answered by n4's own
    /// failed copy, which showed the request's source, and n4 returned 200
    /// (an acknowledged lost update). The read now goes to the primary, and
    /// that primary must acknowledge the operation as its write path does.
    #[test]
    fn a_retried_write_is_confirmed_only_on_the_primary_below_its_acked_checkpoint() {
        use elasticsearch_cluster::state::{AllocationId, ClusterState, IndexMetadata, IndexRoutingTable, IndexShardRoutingTable, IndexState, ShardRouting, ShardState};
        let copy = |primary: bool, node: &str| ShardRouting { index: "jepsen-a".into(), shard: 0, primary, state: ShardState::Started, node_id: Some(node.into()),
            relocating_node_id: None, allocation_id: Some(AllocationId { id: format!("a-{node}"), relocation_id: None }), recovery_source: None, unassigned_info: None, expected_shard_size_in_bytes: None };
        let mut st = ClusterState::empty("test");
        st.metadata.indices.insert("jepsen-a".into(), IndexMetadata { name: "jepsen-a".into(), uuid: "u".into(), version: 1, mapping_version: 1,
            settings_version: 1, aliases_version: 1, state: IndexState::Open, settings: Default::default(), mappings: json!({}).into(), aliases: std::collections::BTreeMap::new(),
            number_of_shards: 1, number_of_replicas: 2, routing_num_shards: 1, primary_terms: vec![2], in_sync_allocation_ids: std::collections::BTreeMap::new(), creation_date: 0, lifecycle: None });
        st.routing_table.indices.insert("jepsen-a".into(), IndexRoutingTable { index_uuid: "u".into(), shards: vec![IndexShardRoutingTable { shard: 0, copies: vec![copy(false, "n4"), copy(true, "n2"), copy(false, "n3")] }] });
        // n4 coordinates and holds a copy: the read goes to n2, not to n4's own.
        let (preference, at) = primary_read(&st, "n4", "jepsen-a", 0).expect("a primary");
        assert_eq!(preference, "_only_nodes:n2");
        assert_eq!(at, PrimaryAt { index: "jepsen-a".into(), shard: 0, node: "n2".into(), term: 2 });
        assert_eq!(primary_read(&st, "n2", "jepsen-a", 0).map(|p| p.0).as_deref(), Some("_only_local"));
        // The stale copy shows this request's source; n2's copy does not.
        let src = json!({"v": 20000000461i64});
        let cas = Condition { if_seq_no: Some(4322), version: None };
        let stale = json!({"found": true, "_seq_no": 4337, "_primary_term": 1, "_source": {"v": 20000000461i64}});
        let on_primary = json!({"found": true, "_seq_no": 4333, "_primary_term": 2, "_source": {"v": 10000000417i64}});
        assert!(is_own_write(&stale, &src, cas) && !is_own_write(&on_primary, &src, cas));
        // Acknowledged by the primary that served the read, in its term.
        assert_eq!(acked_decision(&at, &AckReply::Acked { term: 2 }), Durability::Durable);
        assert_eq!(acked_decision(&at, &AckReply::Failed), Durability::NotYet);
        // That node lost the primary, or holds it in another term.
        assert_eq!(acked_decision(&at, &AckReply::NotPrimary { term: None }), Durability::NotCurrent);
        assert_eq!(acked_decision(&at, &AckReply::NotPrimary { term: Some(3) }), Durability::NotCurrent);
        assert_eq!(acked_decision(&at, &AckReply::Acked { term: 3 }), Durability::NotCurrent);
        // A deposed primary that has not learnt of its successor, read with a
        // coordinator's equally stale state, cannot acknowledge: its
        // `replicate_through` needs the promoted copy, which rejects its term.
    }

    /// The data stream retry (replicated_writes::
    /// an_indeterminate_data_stream_write_is_retried_where_it_was_resolved):
    /// the old primary applied the auto-id create in term 1 and replicated
    /// it to the copy then promoted (term 2). The document on the new
    /// primary carries term 1 and is part of the history it acknowledges, so
    /// the retry's 409 confirms it (201 in the index it resolved): an
    /// earlier term is no reason to report the outcome as unknown.
    #[test]
    fn an_earlier_term_operation_held_by_the_promoted_primary_is_confirmed() {
        let at = PrimaryAt { index: ".ds-logs-u-000001".into(), shard: 0, node: "c".into(), term: 2 };
        // The document carries term 1; only the primary's term and its ack count.
        assert_eq!(acked_decision(&at, &AckReply::Acked { term: 2 }), Durability::Durable);
    }

    #[test]
    fn only_index_requests_have_a_confirmable_id() {
        assert_eq!(indexed_id(&req("PUT", "/i/_doc/a?if_seq_no=3&if_primary_term=1")).as_deref(), Some("a"));
        assert_eq!(indexed_id(&req("DELETE", "/i/_doc/a")), None);
        assert_eq!(indexed_id(&req("POST", "/i/_update/a")), None);
    }

    #[test]
    fn explained_hits_name_the_answering_copy() {
        let mut part = json!({"hits": {"hits": [
            {"_index": "i", "_id": "a", "_shard": "[i][0]", "_node": "local"},
            {"_index": "i", "_id": "b"},
        ]}});
        label_explained_hits(&mut part, 3, "n2");
        assert_eq!(part["hits"]["hits"][0]["_shard"], "[i][3]");
        assert_eq!(part["hits"]["hits"][0]["_node"], "n2");
        assert!(part["hits"]["hits"][1].get("_shard").is_none());
    }
}

/// Bound only admission behind the rollback fence. Once admitted, keep the
/// original task/guard lifetime so a client timeout cannot abandon publication.
async fn enter_engine_api_bounded(node: &Arc<ClusterNode>, req: &RestReq, family: Option<&'static str>)
    -> Result<crate::node::EngineApiGuard, RestResp>
{
    let limit = req.params().get("master_timeout").and_then(|s| crate::rest::parse_timeout(s))
        .unwrap_or(Duration::from_secs(30));
    tokio::time::timeout(limit, node.engine_api_enter(family)).await.map_err(|_| {
        RestResp::error(503, "master_not_discovered_exception", "timed out waiting for engine metadata rollback")
    })
}

#[cfg(test)]
mod rollback_admission_tests {
    use super::*;

    /// A held API prevents rollback from completing. A queued request must
    /// time out without removing that fence or becoming an in-flight API.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn waiting_api_times_out_without_bypassing_rollback() {
        let dir = tempfile::tempdir().unwrap();
        let mut settings = crate::config::Settings::new();
        for (key, value) in [
            ("cluster.name", json!("rollback-admission")),
            ("node.name", json!("rollback-admission")),
            ("path.data", json!(dir.path().join("data"))),
            ("path.logs", json!(dir.path().join("logs"))),
            ("network.host", json!("127.0.0.1")),
            ("http.port", json!("0")),
            ("transport.port", json!("0")),
            ("discovery.type", json!("single-node")),
            ("xpack.security.enabled", json!("false")),
        ] { settings.insert(key.into(), value); }
        let (node, _listener) = ClusterNode::start(crate::config::NodeConfig::from_settings(settings).unwrap()).await.unwrap();
        let held = node.engine_api_in_flight(None);
        node.request_engine_rollback();
        let req = RestReq::try_new("PUT","/_ingest/pipeline/fenced?master_timeout=40ms",&[],&[],0,Dfs::None,None).unwrap();
        for _ in 0..2 {
            let response = match enter_engine_api_bounded(&node, &req, None).await {
                Err(response) => response,
                Ok(_) => panic!("rollback fence was bypassed"),
            };
            assert_eq!(response.status, 503);
            assert_eq!(response.body_json().unwrap()["error"]["type"], "master_not_discovered_exception");
        }
        drop(held);
        // The cancelled admission futures must not keep an in-flight guard.
        let guard = tokio::time::timeout(Duration::from_secs(10), node.engine_api_enter(None)).await
            .expect("rollback should complete after the original API exits");
        drop(guard);
    }
}

#[cfg(test)]
mod query_parser_scope_tests {
    use serde_json::json;

    #[test]
    fn coordinator_validation_uses_installed_query_modules_without_leaking_scope() {
        let dir = tempfile::tempdir().unwrap();
        let node = elasticsearch_engine::node::Node::open(dir.path()).unwrap();
        let query = json!({"query":{"deprecated":{}}});
        assert!(elasticsearch_http::validate_search_request(&node, &query, &Default::default()).is_err());
        node.node_settings.write().insert(elasticsearch_engine::warnings::TEST_MODULES_SETTING.into(),
            json!("x-pack-test-deprecated-query"));
        assert!(elasticsearch_http::validate_search_request(&node, &query, &Default::default()).is_ok());
        assert!(!elasticsearch_engine::warnings::has_test_module("x-pack-test-deprecated-query"));
    }
}

#[cfg(test)]
mod metadata_admission_retry_tests {
    #[test]
    fn executed_scratch_retries_only_with_proof_no_state_task_ran() {
        use std::sync::atomic::AtomicU8;
        let phase = AtomicU8::new(super::METADATA_STARTED);
        let error = || Err("node is not the elected master".to_string());
        let response = || super::RestResp::error(503, "failed_to_commit_cluster_state_exception", "not master");
        assert!(matches!(super::finish_scratch_attempt(&phase, response(), error(), true), super::MetadataAdmission::NotExecuted(_)));
        // Identical error text without proof is indeterminate, never replayed.
        assert!(matches!(super::finish_scratch_attempt(&phase, response(), error(), false), super::MetadataAdmission::Finished(_, Err(_))));
    }

    use super::*;
    use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

    #[test]
    fn election_rejects_queued_attempt_before_closure_then_retry_executes_once() {
        let executions = AtomicUsize::new(0);
        let queued = AtomicU8::new(METADATA_PENDING);
        // execute_batch rejects this queued request after the coordinator
        // changes from Leader to Candidate, before invoking its closure.
        let failed = finish_metadata_attempt(&queued,
            RestResp::error(503, "failed_to_commit_cluster_state_exception", "not elected master"),
            Err("not elected master".into()));
        assert!(matches!(failed, MetadataAdmission::NotExecuted(_)));
        assert!(!start_metadata_attempt(&queued), "late execution of the abandoned attempt is fenced");
        let retry = AtomicU8::new(METADATA_PENDING);
        if start_metadata_attempt(&retry) { executions.fetch_add(1, Ordering::SeqCst); }
        assert!(!start_metadata_attempt(&retry));
        assert!(matches!(finish_metadata_attempt(&retry,
            RestResp::json(200, &json!({"acknowledged":true})), Ok(7)), MetadataAdmission::Finished(_, Ok(7))));
        assert_eq!(executions.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn indeterminate_publication_never_authorizes_replay() {
        let phase = AtomicU8::new(METADATA_PENDING);
        assert!(start_metadata_attempt(&phase));
        let result = finish_metadata_attempt(&phase,
            RestResp::json(200, &json!({"acknowledged":true})), Err("publication outcome unknown".into()));
        match result {
            MetadataAdmission::Finished(response, Err(_)) => assert_eq!(response.status, 503),
            _ => panic!("an executed request must not be retried"),
        }
        assert!(!start_metadata_attempt(&phase));
        let timed_out = AtomicU8::new(METADATA_ABANDONED);
        assert!(matches!(finish_metadata_attempt(&timed_out,
            RestResp::error(503, "process_cluster_event_timeout_exception", "expired"), Err("expired".into())),
            MetadataAdmission::Finished(_, Err(_))));
    }

    #[test]
    fn forwarding_preserves_query_and_reduces_original_admission_budget() {
        let req = RestReq::try_new("POST","/logs/_rollover?master_timeout=30s&timeout=2s&x=a%26b",&[],&[],0,Dfs::None,None).unwrap();
        let next = remaining_master_request(&req, Duration::from_millis(123)).unwrap();
        assert_eq!(master_admission_limit(&next), Duration::from_millis(123));
        assert_eq!(next.params()["timeout"], "2s");
        assert_eq!(next.params()["x"], "a&b");
        assert_eq!(next.path(), req.path());
    }

    #[test]
    fn a_task_refused_by_a_non_master_is_a_retryable_not_master_answer() {
        let refused = task_error(&elasticsearch_cluster::coordination::CoordinationError::not_master("n1").to_string());
        assert_eq!(refused.status, 503);
        assert!(refused_not_master(&refused));
        let failed = task_error("no longer master: term bumped");
        assert_eq!((failed.status, refused_not_master(&failed)), (500, false));
    }
}

#[cfg(test)]
mod broadcast_body_tests {
    use super::*;

    /// `POST /_refresh` / `_flush` with a body (even `{}`) is refused at the
    /// front with ES's 400 before fan-out (it was a 200 with `_shards.failed`
    /// and no refresh); without a body, or on a route that takes one, it
    /// passes.
    #[test]
    fn body_on_no_body_broadcast_is_400_before_fan_out() {
        for uri in ["/_refresh", "/logs/_refresh", "/_flush", "/logs/_flush"] {
            let req = RestReq::try_new("POST", uri, &[], b"{}", 0, Dfs::None, None).unwrap();
            let e = broadcast_body_rejection(&req).unwrap_or_else(|| panic!("{uri} accepted a body"));
            let resp = err(e);
            assert_eq!(resp.status, 400, "{uri}");
            let v = resp.body_json().unwrap();
            assert_eq!(v["error"]["type"], "illegal_argument_exception");
            assert_eq!(v["error"]["reason"], format!("request [POST {uri}] does not support having a body"));
            assert_eq!(v["status"], 400);
            let bare = RestReq::try_new("POST", uri, &[], &[], 0, Dfs::None, None).unwrap();
            assert!(broadcast_body_rejection(&bare).is_none(), "{uri} without a body");
        }
        let search = RestReq::try_new("POST", "/logs/_search", &[], b"{}", 0, Dfs::None, None).unwrap();
        assert!(broadcast_body_rejection(&search).is_none());
    }
}
